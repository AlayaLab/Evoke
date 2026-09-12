

import importlib
from pathlib import Path
from typing import Any, Callable, List, Union
from omegaconf import DictConfig, ListConfig, OmegaConf

try:
    OmegaConf.register_new_resolver("eval", eval)
except Exception as e:

    print(f"Error registering eval resolver: {e}")


def load_config(path: str, argv: List[str] = None) -> Union[DictConfig, ListConfig]:


    if "." in path and "/" not in path and not path.endswith(".yaml"):

        path_parts = path.split(".")[1:]
        config_path = Path(__file__).resolve().parent
        for part in path_parts:
            config_path = config_path.joinpath(part)
        config_path = config_path.with_suffix(".yaml")
        config = OmegaConf.load(str(config_path))
    else:

        config = OmegaConf.load(path)

    if argv is not None:
        config_argv = OmegaConf.from_dotlist(argv)
        config = OmegaConf.merge(config, config_argv)
    config = resolve_recursive(config, resolve_inheritance)
    return config


def resolve_recursive(
    config: Any,
    resolver: Callable[[Union[DictConfig, ListConfig]], Union[DictConfig, ListConfig]],
) -> Any:
    config = resolver(config)
    if isinstance(config, DictConfig):
        for k in config.keys():
            v = config.get(k)
            if isinstance(v, (DictConfig, ListConfig)):
                config[k] = resolve_recursive(v, resolver)
    if isinstance(config, ListConfig):
        for i in range(len(config)):
            v = config.get(i)
            if isinstance(v, (DictConfig, ListConfig)):
                config[i] = resolve_recursive(v, resolver)
    return config


def resolve_inheritance(config: Union[DictConfig, ListConfig]) -> Any:


    if isinstance(config, DictConfig):
        inherit = config.pop("__inherit__", None)

        if inherit:
            inherit_list = inherit if isinstance(inherit, ListConfig) else [inherit]

            parent_config = None
            for parent_path in inherit_list:
                assert isinstance(parent_path, str)
                parent_config = (
                    load_config(parent_path)
                    if parent_config is None
                    else OmegaConf.merge(parent_config, load_config(parent_path))
                )

            if len(config.keys()) > 0:
                config = OmegaConf.merge(parent_config, config)
            else:
                config = parent_config
    return config


def import_item(path: str, name: str) -> Any:


    return getattr(importlib.import_module(path), name)


def create_object(config: DictConfig) -> Any:


    config = DictConfig(config)
    item = import_item(
        path=config.__object__.path,
        name=config.__object__.name,
    )
    args = config.__object__.get("args", "as_config")
    if args == "as_config":
        return item(config)
    if args == "as_params":
        config = OmegaConf.to_object(config)
        config.pop("__object__")
        return item(**config)
    raise NotImplementedError(f"Unknown args type: {args}")


def create_dataset(path: str, *args, **kwargs) -> Any:


    return import_item(path, "create_dataset")(*args, **kwargs)


def to_dict_recursive(config_obj):
    if isinstance(config_obj, DictConfig):
        return {k: to_dict_recursive(v) for k, v in config_obj.items()}
    elif isinstance(config_obj, ListConfig):
        return [to_dict_recursive(item) for item in config_obj]
    return config_obj
