import triton
import triton.language as tl


@triton.jit
def _gather(points_table, lengths, indices, output, COUNT: tl.constexpr,
            BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offset = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    present = tl.load(lengths + row) > 0
    valid = (offset < COUNT * 3) & present
    pointer = tl.load(points_table + row).to(tl.pointer_type(tl.float32))
    sampled = tl.load(indices + row * COUNT + offset // 3, mask=valid, other=0)
    value = tl.load(pointer + sampled * 3 + offset % 3, mask=valid, other=1.0e6)
    tl.store(output + row * COUNT * 3 + offset, value, mask=offset < COUNT * 3)


def gather(points_table, lengths, indices, output, count):
    _gather[(output.shape[0], triton.cdiv(count * 3, 256))](
        points_table, lengths, indices, output, count, 256)
