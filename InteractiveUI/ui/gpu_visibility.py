from __future__ import annotations

import argparse
from pathlib import Path
import re


def _prefix(value: str, label: str) -> list[str]:
    if not re.fullmatch(r"0(?:,[1-9][0-9]*)*", value):
        raise ValueError(f"{label} must be a contiguous GPU prefix: 0,1,...,N-1")
    devices = value.split(',')
    if devices != [str(index) for index in range(len(devices))]:
        raise ValueError(f"{label} must preserve GPU order: 0,1,...,N-1")
    return devices


def resolve_visibility(path: Path, visible: str, sp_size: int) -> str:

    if not path.exists():
        return visible

    with path.open(encoding='ascii') as handle:
        value = handle.read(4097)
    if len(value) > 4096:
        raise ValueError('visible-gpus is too large')
    value = value.removesuffix('\n')
    requested = _prefix(value, 'visible-gpus')
    available = _prefix(visible, 'EVOKE_VISIBLE_GPUS')
    if sp_size < 1:
        raise ValueError('EVOKE_SP_SIZE must be positive')
    if len(requested) < sp_size:
        raise ValueError('visible-gpus has fewer GPUs than EVOKE_SP_SIZE')
    if len(requested) > len(available):
        raise ValueError('visible-gpus exceeds EVOKE_VISIBLE_GPUS')
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--file', type=Path, required=True)
    parser.add_argument('--visible', required=True)
    parser.add_argument('--sp-size', type=int, required=True)
    args = parser.parse_args()
    try:
        result = resolve_visibility(args.file, args.visible, args.sp_size)
    except (OSError, UnicodeError, ValueError) as error:
        parser.exit(2, f'GPU visibility rejected: {error}\n')
    print(result)


if __name__ == '__main__':
    main()
