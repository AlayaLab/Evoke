import triton
import triton.language as tl


@triton.jit
def _covis(CAM, POINTS, FX, FY, CX, CY, SCORES, MASK, COUNTS,
           M: tl.constexpr, W: tl.constexpr, H: tl.constexpr, FAR_HALF: tl.constexpr,
           FX_PTR: tl.constexpr, FY_PTR: tl.constexpr, CX_PTR: tl.constexpr, CY_PTR: tl.constexpr,
           DETAILS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    j = tl.arange(0, BLOCK)
    active = j < M
    x = tl.load(CAM + (row * M + j) * 3, active, 0.)
    y = tl.load(CAM + (row * M + j) * 3 + 1, active, 0.)
    z = tl.load(CAM + (row * M + j) * 3 + 2, active, 0.)
    p0 = tl.load(POINTS + (row * M + j) * 3, active, 1.e6)
    fx = tl.load(FX) if FX_PTR else FX
    fy = tl.load(FY) if FY_PTR else FY
    cx = tl.load(CX) if CX_PTR else CX
    cy = tl.load(CY) if CY_PTR else CY


    visible = tl.inline_asm_elementwise(
        asm="""
        {
            .reg .f32 zc, qx, qy, mx, my, px, py;
            .reg .pred clamp_low, ok, p;
            setp.lt.f32 clamp_low, $3, 0f358637bd;
            selp.f32 zc, 0f358637bd, $3, clamp_low;
            div.rn.f32 qx, $1, zc;
            div.rn.f32 qy, $2, zc;
            mul.rn.f32 mx, qx, $5;
            mul.rn.f32 my, qy, $6;
            add.rn.f32 px, mx, $7;
            add.rn.f32 py, my, $8;
            setp.gt.f32 ok, $3, 0f38d1b717;
            setp.ge.f32 p, px, 0f00000000;
            and.pred ok, ok, p;
            setp.lt.f32 p, px, $9;
            and.pred ok, ok, p;
            setp.ge.f32 p, py, 0f00000000;
            and.pred ok, ok, p;
            setp.lt.f32 p, py, $10;
            and.pred ok, ok, p;
            setp.lt.f32 p, $4, $11;
            and.pred ok, ok, p;
            selp.u32 $0, 1, 0, ok;
        }
        """,
        constraints="=r,f,f,f,f,f,f,f,f,f,f,f",
        args=[x, y, z, p0, fx.to(tl.float32), fy.to(tl.float32),
              cx.to(tl.float32), cy.to(tl.float32), tl.full((), W, tl.float32),
              tl.full((), H, tl.float32), tl.full((), FAR_HALF, tl.float32)],
        dtype=tl.int32, is_pure=True, pack=1)
    ok = active & (visible != 0)
    count = tl.sum(ok.to(tl.int32), 0)

    reciprocal = tl.full((), 1. / M, tl.float32)
    score = count.to(tl.float32) * reciprocal
    tl.store(SCORES + row, score)
    if DETAILS:
        tl.store(MASK + row * M + j, ok, active)
        tl.store(COUNTS + row, count.to(tl.int64))


def launch(cam, P, fx, fy, cx, cy, W, H, FAR, scores, mask, counts):
    import torch
    with torch.cuda.device(cam.device):
        _covis[(cam.shape[0],)](cam, P, fx, fy, cx, cy, scores,
                               mask if mask is not None else scores,
                               counts if counts is not None else scores,
                               cam.shape[1], W, H, FAR * .5,
                               *(isinstance(value, torch.Tensor) for value in (fx, fy, cx, cy)),
                               mask is not None, triton.next_power_of_2(cam.shape[1]),
                               num_warps=4, enable_fp_fusion=False)
