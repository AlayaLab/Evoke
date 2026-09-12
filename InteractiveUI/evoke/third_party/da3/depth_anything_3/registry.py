

from collections import OrderedDict
from pathlib import Path


def get_all_models() -> OrderedDict:


    configs_dir = Path(__file__).resolve().parent / "configs"


    configs_dir = Path(configs_dir)

    model_entries = []

    for item in configs_dir.iterdir():

        if item.is_file() and item.suffix == ".yaml":

            model_name = item.stem

            file_abs_path = str(item.resolve())
            model_entries.append((model_name, file_abs_path))


    sorted_entries = sorted(model_entries, key=lambda x: x[0])
    return OrderedDict(sorted_entries)


MODEL_REGISTRY = get_all_models()
