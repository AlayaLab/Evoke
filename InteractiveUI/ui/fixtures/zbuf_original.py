import os
import torch
import numpy as np
from evoke.modules.geometric_state.da3_cloud import unproject_depth_torch

def _render_backward_multisrc_zbuf(store, ids_all, target_c2ws, K_pix, height, width, *,
                                   nearby=16, fill_iters=12, recall_min_cov=0.5, recall_margin=0.15,
                                   depth_thresh=0.02, topk=8,
                                   fg_covis=0.3, fg_factor=1.5,
                                   fg_scale_exempt=1.0,
                                   zbuf_despeckle=False, zbuf_despeckle_ksize=3, zbuf_despeckle_fill_iters=4,
                                   device="cuda"):


    import torch.nn.functional as _F
    H, W = int(height), int(width); F_t = int(target_c2ws.shape[0])
    Kt = torch.as_tensor(np.asarray(K_pix, np.float32), device=device)
    fx, fy, cx, cy = Kt[0, 0], Kt[1, 1], Kt[0, 2], Kt[1, 2]
    if not ids_all:
        return (torch.full((1, 3, F_t, H, W), -1.0, device=device),
                torch.zeros((1, 1, F_t, H, W), device=device))
    ids_all = sorted(ids_all)

    depth_thresh = float(os.environ.get("WARP_ZBUF_DEPTH_THRESH", str(depth_thresh)))
    topk = int(os.environ.get("WARP_ZBUF_TOPK", str(topk)))
    M = int(os.environ.get("WARP_ZBUF_COVIS_M", "2000"))
    _covis_min = float(os.environ.get("WARP_ZBUF_COVIS_MIN", "0"))
    FAR = 1e6
    id2row = {g: i for i, g in enumerate(ids_all)}

    _wseed = os.environ.get("EVOKE_WARP_SEED")
    _cgen = torch.Generator(device=device).manual_seed(int(_wseed)) if _wseed is not None else None


    P_all = torch.full((len(ids_all), M, 3), FAR, device=device)
    for g in ids_all:
        d, it, cwi, _ = store[g]; wp = unproject_depth_torch(d, it, cwi).reshape(-1, 3)


        wp = wp[(d.reshape(-1) > 1e-4) & torch.isfinite(wp).all(-1)]
        if wp.shape[0] > 0:
            _idx = (torch.randint(0, wp.shape[0], (M,), device=device, generator=_cgen)
                    if _cgen is not None else torch.randint(0, wp.shape[0], (M,), device=device))
            P_all[id2row[g]] = wp[_idx]

    def covis_vec(rows, tpose):

        if len(rows) == 0:
            return torch.zeros((0,), device=device)
        P = P_all[torch.as_tensor(rows, device=device)]
        w2c = torch.linalg.inv(tpose); R = w2c[:3, :3]; t = tpose[:3, 3]
        cam = torch.einsum('cmj,kj->cmk', P - t, R); z = cam[..., 2]
        px = cam[..., 0] / z.clamp(min=1e-6) * fx + cx; py = cam[..., 1] / z.clamp(min=1e-6) * fy + cy
        ok = (z > 1e-4) & (px >= 0) & (px < W) & (py >= 0) & (py < H) & (P[..., 0] < FAR * 0.5)
        return ok.float().mean(1)

    def splat_one(g, tpose, scale=1.0):


        d, it, cwi, rr = store[g]; h, w = d.shape
        ys, xs = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32),
                                torch.arange(w, device=device, dtype=torch.float32), indexing="ij")
        z = d.float()
        if scale != 1.0:
            z = z * scale
        Xc = (xs - it[0, 2]) / it[0, 0] * z; Yc = (ys - it[1, 2]) / it[1, 1] * z
        cam = torch.stack([Xc, Yc, z, torch.ones_like(z)], -1).reshape(-1, 4)
        world = (cwi @ cam.T).T[:, :3]
        w2c = torch.linalg.inv(tpose); ct = (w2c[:3, :3] @ world.T).T + w2c[:3, 3]; zt = ct[:, 2]
        xt = torch.round(ct[:, 0] / zt.clamp(min=1e-6) * fx + cx).long()
        yt = torch.round(ct[:, 1] / zt.clamp(min=1e-6) * fy + cy).long()
        src_flat = torch.arange(h * w, device=device)
        ok = (z.reshape(-1) > 1e-4) & (zt > 1e-4) & (xt >= 0) & (xt < W) & (yt >= 0) & (yt < H)
        col = torch.zeros(3, H, W, device=device)
        zbuf = torch.full((H * W,), float("inf"), device=device)
        if not bool(ok.any()):
            return col, zbuf, torch.zeros(H, W, dtype=torch.bool, device=device)
        key = (yt * W + xt)[ok]; zt_ok = zt[ok]; src_ok = src_flat[ok]

        zq = (zt_ok.clamp(0, 1e6) * 1000).long().clamp(0, (1 << 38) - 1)
        packed = (zq << 24) | src_ok
        INF = torch.full((H * W,), (1 << 62), dtype=torch.long, device=device)
        INF.scatter_reduce_(0, key, packed, reduce="amin", include_self=True)
        valid = INF < (1 << 62); owner = (INF & ((1 << 24) - 1)).clamp(max=h * w - 1)
        zbuf[valid] = ((INF[valid] >> 24).float()) / 1000.0
        us = (owner % w).float(); vs = (owner // w).float()
        gx = us / max(w - 1, 1) * 2 - 1; gy = vs / max(h - 1, 1) * 2 - 1
        grid = torch.stack([gx.reshape(H, W), gy.reshape(H, W)], -1)[None]
        samp = _F.grid_sample(rr[None].float(), grid, mode="bilinear", padding_mode="border", align_corners=False)[0]
        vmask = valid.reshape(H, W)
        col = torch.where(vmask[None].expand(3, H, W), samp, torch.zeros_like(samp))
        return col, zbuf, vmask

    def est_scale(g, ref_pts, tposepts_n=200):


        d, it, cwi, _ = store[g]; h, w = d.shape
        if ref_pts is None or ref_pts.shape[0] == 0:
            return 1.0, 0
        w2c = torch.linalg.inv(cwi); cam = (w2c[:3, :3] @ ref_pts.T).T + w2c[:3, 3]; z = cam[:, 2]
        px = (cam[:, 0] / z.clamp(min=1e-6) * it[0, 0] + it[0, 2]).round().long()
        py = (cam[:, 1] / z.clamp(min=1e-6) * it[1, 1] + it[1, 2]).round().long()
        ok = (z > 1e-4) & (px >= 0) & (px < w) & (py >= 0) & (py < h)
        if int(ok.sum()) < tposepts_n:
            return 1.0, int(ok.sum())
        D = torch.full((h * w,), float("inf"), device=device)
        D.scatter_reduce_(0, py[ok] * w + px[ok], z[ok], reduce="amin", include_self=True)
        df = d.reshape(-1); v = (D < float("inf")) & (df > 1e-4)
        nov = int(v.sum())
        if nov < tposepts_n:
            return 1.0, nov
        s = float((D[v] / df[v]).median())
        return (s if 0.2 < s < 5.0 else 1.0), nov


    nearby_ids = ids_all[-int(nearby):]; nearby_set = set(nearby_ids)


    _fg_covis = float(os.environ.get("WARP_ZBUF_FG_COVIS", fg_covis))
    _fg_factor = float(os.environ.get("WARP_ZBUF_FG_FACTOR", fg_factor))
    _nb_cen = torch.stack([store[g][2][:3, 3] for g in nearby_ids]).to(device).float() if (nearby_ids and _fg_covis > 0) else None
    _all_cen = {g: store[g][2][:3, 3].to(device).float() for g in ids_all} if _fg_covis > 0 else None
    _nref_l = [P_all[id2row[g]][P_all[id2row[g]][:, 0] < FAR * 0.5] for g in nearby_ids]
    _nref_l = [p for p in _nref_l if p.shape[0] > 0]
    nref = torch.cat(_nref_l) if _nref_l else None
    _es = {g: ((1.0, 0) if g in nearby_set else est_scale(g, nref)) for g in ids_all}
    src_scale = {g: _es[g][0] for g in ids_all}
    _fg_scale_exempt = float(os.environ.get("WARP_ZBUF_FG_SCALE_EXEMPT", str(fg_scale_exempt)))

    src_aligned = ({g: (g not in nearby_set and _es[g][1] >= 200 and 0.8 <= _es[g][0] <= 1.25) for g in ids_all}
                   if _fg_scale_exempt > 0 else {})

    tc = target_c2ws.to(device).float()
    all_rows = [id2row[g] for g in ids_all]
    warps = []; viss = []
    _age_dbg = bool(os.environ.get("EVOKE_WARP_AGE_DEBUG"))
    _age_rows = []
    for f in range(F_t):
        tpose = tc[f]
        cv = covis_vec(all_rows, tpose)
        k = min(int(topk), len(ids_all)); order = torch.topk(cv, k=k).indices.tolist()
        if _covis_min > 0:
            order = [i for i in order if float(cv[i]) >= _covis_min]
        if _fg_covis > 0 and _nb_cen is not None:
            _tcen = tpose[:3, 3]; _dnb = float((_nb_cen - _tcen).norm(dim=1).min())
            order = [i for i in order if float(cv[i]) >= _fg_covis
                     or float((_all_cen[ids_all[i]] - _tcen).norm()) <= _fg_factor * _dnb
                     or src_aligned.get(ids_all[i], False)]
        cand = [ids_all[i] for i in order]
        fused = torch.zeros(3, H, W, device=device)
        fused_depth = torch.full((H * W,), float("inf"), device=device)
        covered = torch.zeros(H, W, dtype=torch.bool, device=device)
        winner_row = torch.full((H * W,), -1, dtype=torch.long, device=device) if _age_dbg else None
        for g in cand:
            col, zbuf, _vm = splat_one(g, tpose, scale=src_scale[g])
            update = torch.isfinite(zbuf) & (zbuf < fused_depth - depth_thresh)
            if not bool(update.any()):
                update = torch.isfinite(zbuf) & (zbuf < fused_depth)
                if not bool(update.any()):
                    continue
            um = update.reshape(H, W)
            fused = torch.where(um[None].expand(3, H, W), col, fused)
            fused_depth = torch.where(update, zbuf, fused_depth); covered = covered | um
            if _age_dbg:
                winner_row[update] = int(id2row[g])
        if _age_dbg:
            _age_rows.append(winner_row[covered.reshape(-1)])


        if zbuf_despeckle:
            k = int(zbuf_despeckle_ksize)
            erosion = lambda x: -_F.max_pool2d(-x, k, 1, k // 2)
            dilation = lambda x: _F.max_pool2d(x, k, 1, k // 2)
            V = covered.float()[None, None]
            opened = dilation(erosion(V))
            hv = erosion(dilation(opened))
            hv_b = hv[0, 0] > 0.5
            fill_mask = hv_b & (~covered)
            removed = covered & (~hv_b)
            col = fused.clone()
            m = covered.float()[None, None]
            for _ in range(int(zbuf_despeckle_fill_iters)):
                num = _F.avg_pool2d(col[None] * m, 3, 1, 1); den = _F.avg_pool2d(m, 3, 1, 1)
                upd = fill_mask & (den[0, 0] > 1e-6) & (m[0, 0] < 0.5)
                col = torch.where(upd[None].expand(3, -1, -1), (num / den.clamp(min=1e-6))[0], col)
                m[0, 0] = torch.where(upd, torch.ones_like(m[0, 0]), m[0, 0])
            col[:, removed] = 0.0
            warps.append((col.clamp(0, 1) * 2 - 1)); viss.append(hv_b.float())
            continue
        warps.append((fused.clamp(0, 1) * 2 - 1)); viss.append(covered.float())
    vis_out = torch.stack(viss)[None, None].contiguous()

    _mean_cov = float(vis_out.mean())
    if _mean_cov < 0.25:
        _n_old = int(sum(1 for g in ids_all if g not in nearby_set))
        _n_aligned = int(sum(1 for g in ids_all if abs(float(src_scale[g]) - 1.0) > 1e-6))
        print(f"[zbuf-render] LOW-COV mean_cov={_mean_cov*100:.1f}% src={len(ids_all)} "
              f"nearby={len(nearby_ids)} old(recall)={_n_old} aligned={_n_aligned} "
              f"nref={'0' if nref is None else nref.shape[0]} F={F_t}", flush=True)
    if _age_dbg and _age_rows:
        _wr = torch.cat(_age_rows)
        if _wr.numel() > 0:
            _ids_t = torch.as_tensor(ids_all, device=_wr.device)
            _gid = _ids_t[_wr]; _age = (int(ids_all[-1]) - _gid).float() / 24.0
            _n = float(_wr.numel())
            _b = lambda lo, hi: 100.0 * float(((_age >= lo) & (_age < hi)).sum()) / _n
            _oldest = int(_gid.min().item()); _newest_win = int(_gid.max().item())
            _n_old_pool = int(sum(1 for g in ids_all if (int(ids_all[-1]) - g) > 120))
            print(f"[warp-age] winners={int(_n)} bank_gid=[{ids_all[0]}..{ids_all[-1]}] pool={len(ids_all)} old_in_pool={_n_old_pool} | "
                  f"age<2s={_b(0,2):.1f}% 2-5s={_b(2,5):.1f}% 5-10s={_b(5,10):.1f}% "
                  f"10-20s={_b(10,20):.1f}% 20s+={_b(20,1e9):.1f}% | "
                  f"winner_gid=[{_oldest}..{_newest_win}] frac_from_ge5s={_b(5,1e9):.1f}%", flush=True)
    return torch.stack(warps, 1)[None].contiguous(), vis_out
