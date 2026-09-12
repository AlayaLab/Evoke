from __future__ import annotations

import math
import json
import time


def _validate(cam, P, W, H):
    if cam.ndim != 3 or cam.shape[-1] != 3 or P.shape != cam.shape:
        raise ValueError('cam and P must have identical [N,M,3] shapes')
    if cam.device != P.device:
        raise ValueError('cam and P must be on the same device')
    if type(W) is not int or type(H) is not int or min(W, H) <= 0:
        raise ValueError('W and H must be positive integers')


def _unsupported(cam, P, intrinsics, FAR):
    import torch
    if cam.device.type != 'cuda': return 'CUDA required'
    if any(t.dtype != torch.float32 or not t.is_contiguous() for t in (cam, P)):
        return 'contiguous FP32 cam and P required'
    if not 1 <= cam.shape[1] <= 8192: return 'sample count must be 1..8192'
    for value in intrinsics:
        if isinstance(value, torch.Tensor):
            if value.ndim != 0 or value.dtype != torch.float32 or value.device != cam.device:
                return 'intrinsics must be same-device FP32 scalar tensors or Python numbers'
        elif type(value) not in (int, float):
            return 'intrinsics must be scalar tensors or Python numbers'
    if type(FAR) not in (int, float) or not math.isfinite(FAR):
        return 'FAR must be a finite Python number'
    return None


def reference_covis(cam, P, fx, fy, cx, cy, W, H, FAR=1e6, *, return_details=False):

    _validate(cam, P, W, H)
    z = cam[..., 2]
    px = cam[..., 0] / z.clamp(min=1e-6) * fx + cx
    py = cam[..., 1] / z.clamp(min=1e-6) * fy + cy
    ok = (z > 1e-4) & (px >= 0) & (px < W) & (py >= 0) & (py < H) & (P[..., 0] < FAR * .5)
    scores = ok.float().mean(1)
    return {'scores': scores, 'mask': ok, 'counts': ok.sum(1)} if return_details else scores


def fused_covis(cam, P, fx, fy, cx, cy, W, H, FAR=1e6, *, fallback=True, return_details=False):
    import torch
    _validate(cam, P, W, H)
    reason = _unsupported(cam, P, (fx, fy, cx, cy), FAR)
    if reason:
        if not fallback: raise ValueError(f'Fused covis unsupported: {reason}')
        return reference_covis(cam, P, fx, fy, cx, cy, W, H, FAR, return_details=return_details)
    scores = torch.empty((cam.shape[0],), dtype=torch.float32, device=cam.device)
    mask = torch.empty(cam.shape[:2], dtype=torch.bool, device=cam.device) if return_details else None
    counts = torch.empty((cam.shape[0],), dtype=torch.int64, device=cam.device) if return_details else None
    if cam.shape[0]:
        from .warp_covis_triton import launch
        launch(cam, P, fx, fy, cx, cy, W, H, FAR, scores, mask, counts)
    return {'scores': scores, 'mask': mask, 'counts': counts} if return_details else scores


def _failure_details(cam, P, fx, fy, cx, cy, W, H, FAR, reference, candidate, checks):

    import torch
    def safe(value):
        if isinstance(value, float) and not math.isfinite(value):
            return 'NaN' if math.isnan(value) else ('+Inf' if value > 0 else '-Inf')
        if isinstance(value, list): return [safe(item) for item in value]
        return value
    def value(tensor): return safe(tensor.detach().cpu().tolist())
    def bits(tensor):
        data = tensor.detach().contiguous().view(torch.int32).cpu().reshape(-1).tolist()
        return [f'0x{number & 0xffffffff:08x}' for number in data]
    mask_diff = reference['mask'] != candidate['mask']
    score_diff = reference['scores'].view(torch.int32) != candidate['scores'].view(torch.int32)
    count_diff = reference['counts'] != candidate['counts']
    per_row_mask_count = mask_diff.sum(1)
    differing_rows = ((per_row_mask_count > 0) | score_diff | count_diff).nonzero().flatten().cpu().tolist()
    mask_counts_cpu = per_row_mask_count.cpu().tolist()
    row_details = []
    for row in differing_rows[:16]:
        row_details.append(dict(row=row, maskDifferences=int(per_row_mask_count[row].item()),
                                referenceScore=value(reference['scores'][row]),
                                fusedScore=value(candidate['scores'][row]),
                                referenceScoreBits=bits(reference['scores'][row])[0],
                                fusedScoreBits=bits(candidate['scores'][row])[0],
                                referenceCount=int(reference['counts'][row].item()),
                                fusedCount=int(candidate['counts'][row].item())))

    coordinates = []
    for row in differing_rows:
        if not mask_counts_cpu[row]: continue
        columns = mask_diff[row].nonzero().flatten()[:16-len(coordinates)].cpu().tolist()
        coordinates.extend((row, col) for col in columns)
        if len(coordinates) == 16: break
    point_details = []
    if coordinates:
        rows = torch.tensor([x[0] for x in coordinates], device=cam.device)
        cols = torch.tensor([x[1] for x in coordinates], device=cam.device)
        selected = cam[rows, cols]
        zc = selected[:, 2].clamp(min=1e-6)
        qx = selected[:, 0] / zc; qy = selected[:, 1] / zc
        mx = qx * fx; my = qy * fy
        px = mx + cx; py = my + cy
        for offset, (row, col) in enumerate(coordinates):
            point_details.append(dict(row=row, col=col, cam=value(cam[row,col]), P=value(P[row,col]),
                                      camBits=bits(cam[row,col]), PBits=bits(P[row,col]),
                                      referenceVisible=bool(reference['mask'][row,col].item()),
                                      fusedVisible=bool(candidate['mask'][row,col].item()),
                                      referenceSubsetStages={name:{'value':value(tensor[offset]),
                                          'bits':bits(tensor[offset])[0]} for name,tensor in
                                          [('clampedZ',zc),('xDivZ',qx),('yDivZ',qy),
                                           ('xMulFx',mx),('yMulFy',my),('px',px),('py',py)]}))
    scalar = lambda x: value(x) if isinstance(x, torch.Tensor) else safe(x)
    return dict(checks=checks, shape=list(cam.shape),
                intrinsics={name:scalar(v) for name,v in [('fx',fx),('fy',fy),('cx',cx),('cy',cy)]},
                W=W,H=H,FAR=safe(FAR), totalMaskDifferences=int(per_row_mask_count.sum().item()),
                totalScoreDifferences=int(score_diff.sum().item()),
                totalCountDifferences=int(count_diff.sum().item()),
                totalDifferingRows=len(differing_rows), rows=row_details, points=point_details,
                rowLimit=16, pointLimit=16,
                subsetStageNote='Original Torch projection operators recomputed on at most16 selected points; not fused intermediates. Nonfinite values are strings; float32 hex preserves signed zero/subnormals.')


def audit_benchmark(cam, P, fx, fy, cx, cy, W, H, FAR=1e6, *, repeats=2):


    import torch
    _validate(cam, P, W, H)
    reason = _unsupported(cam, P, (fx, fy, cx, cy), FAR)
    if reason: raise ValueError(f'CUDA audit requires fused path: {reason}')
    if type(repeats) is not int or not 1 <= repeats <= 8:
        raise ValueError('repeats must be 1..8')
    if cam.shape[0] == 0: raise ValueError('audit requires at least one source')
    device = cam.device
    original_cpu = torch.get_rng_state().clone()
    original_cuda = torch.cuda.get_rng_state(device).clone()
    args = (cam, P, fx, fy, cx, cy, W, H, FAR)
    def check_rng():
        if not torch.equal(original_cpu, torch.get_rng_state()) or not torch.equal(original_cuda, torch.cuda.get_rng_state(device)):
            raise AssertionError('covis changed global RNG state')
    def exact(a, b):
        return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.view(torch.int32), b.view(torch.int32))
    samples = []
    try:
        torch.cuda.synchronize(device)
        ref = reference_covis(*args, return_details=True)
        fused = fused_covis(*args, fallback=False, return_details=True)
        torch.cuda.synchronize(device)
        checks = {'scoresBitsExact': exact(ref['scores'], fused['scores']),
                  'maskExact': torch.equal(ref['mask'], fused['mask']),
                  'countsExact': torch.equal(ref['counts'], fused['counts'])}
        if not all(checks.values()):
            details = _failure_details(*args, ref, fused, checks)
            error = AssertionError('Fused covis exact audit failed: ' + json.dumps(details, allow_nan=False))
            error.diagnostics = details
            raise error
        check_rng()

        fused_covis(*args, fallback=False); torch.cuda.synchronize(device)
        for _ in range(repeats):
            for name in ('reference', 'fused', 'fused', 'reference'):
                torch.cuda.synchronize(device)
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record(torch.cuda.current_stream(device)); started = time.perf_counter()
                result = reference_covis(*args) if name == 'reference' else fused_covis(*args, fallback=False)
                end.record(torch.cuda.current_stream(device)); end.synchronize()
                elapsed = time.perf_counter() - started
                if not exact(ref['scores'], result):
                    raise AssertionError('Fused covis ABBA score bits changed')
                check_rng()
                samples.append({'variant': name, 'wallSeconds': elapsed,
                                'cudaSeconds': begin.elapsed_time(end) / 1000, 'bitsExact': True})
        return {'sourceCount': cam.shape[0], 'sampleCount': cam.shape[1], 'checks': checks,
                'rngExact': True, 'order': 'ABBA', 'samples': samples,
                'includes': 'post-einsum projection, visibility comparisons, mean and allocations',
                'excludes': 'world-to-camera einsum, sampling, source sorting, diagnostic masks and comparisons'}
    finally:
        try: torch.cuda.synchronize(device)
        finally:
            torch.set_rng_state(original_cpu)
            torch.cuda.set_rng_state(original_cuda, device)


def boundary_cases(device='cpu', *, count=2000):


    import torch
    if type(count) is not int or not 1 <= count <= 8192:
        raise ValueError('count must be 1..8192')
    def neighbors(value):
        x = torch.tensor(value, dtype=torch.float32)
        return [torch.nextafter(x, torch.tensor(float('-inf'))).item(), x.item(),
                torch.nextafter(x, torch.tensor(float('inf'))).item()]
    rows = []
    for z in [0., -0., float('nan'), float('inf'), -float('inf'), *neighbors(1e-6), *neighbors(1e-4), 1.]:
        rows.append([0., 0., z])
    for x in [*neighbors(0.), *neighbors(640.), float('nan'), float('inf'), -float('inf')]:
        rows.append([x, 1., 1.])
    for y in [*neighbors(0.), *neighbors(384.), float('nan'), float('inf'), -float('inf')]:
        rows.append([1., y, 1.])
    base = torch.tensor(rows, dtype=torch.float32)
    cam = base[torch.arange(count) % len(rows)].unsqueeze(0).repeat(4, 1, 1).to(device)
    P = torch.zeros_like(cam)
    far_neighbors = neighbors(5e5)
    for row, value in enumerate([*far_neighbors, float('nan')]): P[row, :, 0] = value
    def case(name, camera, points, fx, fy, cx, cy):
        return dict(name=name, cam=camera, P=points, fx=fx, fy=fy, cx=cx, cy=cy, W=640, H=384, FAR=1e6)
    threshold = case('thresholds-nan-inf-signedzero', cam, P, 1., 1., 0., 0.)

    calibration = torch.tensor([491.375, 487.125, 319.5, 191.5], dtype=torch.float32, device=device)
    calibrated_rows = []
    for z in (0.00010001, .25, 1., 3., 1000.):
        for boundary in (0., 640.):
            for x in neighbors((boundary - 319.5) / 491.375 * z):
                calibrated_rows.append([x, 0., z])
        for boundary in (0., 384.):
            for y in neighbors((boundary - 191.5) / 487.125 * z):
                calibrated_rows.append([0., y, z])
    cb = torch.tensor(calibrated_rows, dtype=torch.float32)
    cc = cb[torch.arange(count) % len(cb)].unsqueeze(0).contiguous().to(device)
    calibrated = case('calibrated-division-mul-add-boundaries', cc, torch.zeros_like(cc), *calibration.unbind())

    c = torch.zeros((count + 1, count, 3), dtype=torch.float32, device=device); c[..., 2] = 1.
    p = torch.zeros_like(c)
    visible = torch.arange(count, device=device)[None, :] < torch.arange(count + 1, device=device)[:, None]
    p[..., 0] = torch.where(visible, 0., 1e6)
    reductions = case('every-visible-count', c, p, 1., 1., 0., 0.)

    tiny = torch.finfo(torch.float32).tiny
    arithmetic_rows = [[x,y,z] for z in (1.,2.,3.,16.) for x,y in
                       [(-tiny,0.),(tiny,0.),(0.,-tiny),(0.,tiny),
                        (-tiny*.5,0.),(0.,-tiny*.5),(-0.,0.),(0.,-0.)]]
    ac = torch.tensor(arithmetic_rows,dtype=torch.float32)
    ac = ac[torch.arange(count)%len(ac)].unsqueeze(0).contiguous().to(device)
    arithmetic = case('subnormal-division-and-multiplication',ac,torch.zeros_like(ac),.5,.5,0.,0.)
    return [threshold, calibrated, reductions, arithmetic]
