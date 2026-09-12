

from typing import Any, Generator, Iterable, Literal, Union
import torch
from torch import Tensor

Alignment = Literal["start", "center", "end"]
Axis = Literal["horizontal", "vertical"]
Color = Union[
    int,
    float,
    Iterable[int],
    Iterable[float],
    Tensor,
    Tensor,
]


def _sanitize_color(color: Color) -> Tensor:

    if isinstance(color, torch.Tensor):
        color = color.tolist()


    if isinstance(color, Iterable):
        color = list(color)
    else:
        color = [color]

    return torch.tensor(color, dtype=torch.float32)


def _intersperse(iterable: Iterable, delimiter: Any) -> Generator[Any, None, None]:
    it = iter(iterable)
    yield next(it)
    for item in it:
        yield delimiter
        yield item


def _get_main_dim(main_axis: Axis) -> int:
    return {
        "horizontal": 2,
        "vertical": 1,
    }[main_axis]


def _get_cross_dim(main_axis: Axis) -> int:
    return {
        "horizontal": 1,
        "vertical": 2,
    }[main_axis]


def _compute_offset(base: int, overlay: int, align: Alignment) -> slice:
    assert base >= overlay
    offset = {
        "start": 0,
        "center": (base - overlay) // 2,
        "end": base - overlay,
    }[align]
    return slice(offset, offset + overlay)


def overlay(
    base: Tensor,
    overlay: Tensor,
    main_axis: Axis,
    main_axis_alignment: Alignment,
    cross_axis_alignment: Alignment,
) -> Tensor:

    _, base_height, base_width = base.shape
    _, overlay_height, overlay_width = overlay.shape
    assert base_height >= overlay_height and base_width >= overlay_width


    main_dim = _get_main_dim(main_axis)
    main_slice = _compute_offset(
        base.shape[main_dim], overlay.shape[main_dim], main_axis_alignment
    )


    cross_dim = _get_cross_dim(main_axis)
    cross_slice = _compute_offset(
        base.shape[cross_dim], overlay.shape[cross_dim], cross_axis_alignment
    )


    selector = [..., None, None]
    selector[main_dim] = main_slice
    selector[cross_dim] = cross_slice
    result = base.clone()
    result[selector] = overlay
    return result


def cat(
    main_axis: Axis,
    *images: Iterable[Tensor],
    align: Alignment = "center",
    gap: int = 8,
    gap_color: Color = 1,
) -> Tensor:

    device = images[0].device
    gap_color = _sanitize_color(gap_color).to(device)


    cross_dim = _get_cross_dim(main_axis)
    cross_axis_length = max(image.shape[cross_dim] for image in images)


    padded_images = []
    for image in images:

        padded_shape = list(image.shape)
        padded_shape[cross_dim] = cross_axis_length
        base = torch.ones(padded_shape, dtype=torch.float32, device=device)
        base = base * gap_color[:, None, None]
        padded_images.append(overlay(base, image, main_axis, "start", align))


    if gap > 0:

        c, _, _ = images[0].shape
        separator_size = [gap, gap]
        separator_size[cross_dim - 1] = cross_axis_length
        separator = torch.ones((c, *separator_size), dtype=torch.float32, device=device)
        separator = separator * gap_color[:, None, None]


        padded_images = list(_intersperse(padded_images, separator))

    return torch.cat(padded_images, dim=_get_main_dim(main_axis))


def hcat(
    *images: Iterable[Tensor],
    align: Literal["start", "center", "end", "top", "bottom"] = "start",
    gap: int = 8,
    gap_color: Color = 1,
):

    return cat(
        "horizontal",
        *images,
        align={
            "start": "start",
            "center": "center",
            "end": "end",
            "top": "start",
            "bottom": "end",
        }[align],
        gap=gap,
        gap_color=gap_color,
    )


def vcat(
    *images: Iterable[Tensor],
    align: Literal["start", "center", "end", "left", "right"] = "start",
    gap: int = 8,
    gap_color: Color = 1,
):

    return cat(
        "vertical",
        *images,
        align={
            "start": "start",
            "center": "center",
            "end": "end",
            "left": "start",
            "right": "end",
        }[align],
        gap=gap,
        gap_color=gap_color,
    )


def add_border(
    image: Tensor,
    border: int = 8,
    color: Color = 1,
) -> Tensor:
    color = _sanitize_color(color).to(image)
    c, h, w = image.shape
    result = torch.empty(
        (c, h + 2 * border, w + 2 * border), dtype=torch.float32, device=image.device
    )
    result[:] = color[:, None, None]
    result[:, border : h + border, border : w + border] = image
    return result
