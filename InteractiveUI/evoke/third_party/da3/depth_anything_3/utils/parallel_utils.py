

import asyncio
import os
from functools import wraps
from multiprocessing.pool import ThreadPool
from threading import Thread
from typing import Callable, Dict, List
import imageio
from tqdm import tqdm


def async_call_func(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()

        return await loop.run_in_executor(None, func, *args, **kwargs)

    return wrapper


slice_func = lambda chunk_index, chunk_dim, chunk_size: [slice(None)] * chunk_dim + [
    slice(chunk_index, chunk_index + chunk_size)
]


def async_call(fn):
    def wrapper(*args, **kwargs):
        Thread(target=fn, args=args, kwargs=kwargs).start()

    return wrapper


def _save_image_impl(save_img, save_path):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    imageio.imwrite(save_path, save_img)


@async_call
def save_image_async(save_img, save_path):

    _save_image_impl(save_img, save_path)


def save_image(save_img, save_path):

    _save_image_impl(save_img, save_path)


def parallel_execution(
    *args,
    action: Callable,
    num_processes=32,
    print_progress=False,
    sequential=False,
    async_return=False,
    desc=None,
    **kwargs,
):


    args = list(args)

    def get_length(args: List, kwargs: Dict):
        for a in args:
            if isinstance(a, list):
                return len(a)
        for v in kwargs.values():
            if isinstance(v, list):
                return len(v)
        raise NotImplementedError

    def get_action_args(length: int, args: List, kwargs: Dict, i: int):
        action_args = [
            (arg[i] if isinstance(arg, list) and len(arg) == length else arg) for arg in args
        ]

        action_kwargs = {
            key: (
                kwargs[key][i]
                if isinstance(kwargs[key], list) and len(kwargs[key]) == length
                else kwargs[key]
            )
            for key in kwargs
        }
        return action_args, action_kwargs

    if not sequential:

        pool = ThreadPool(processes=num_processes)


        results = []
        asyncs = []
        length = get_length(args, kwargs)
        for i in range(length):
            action_args, action_kwargs = get_action_args(length, args, kwargs, i)
            async_result = pool.apply_async(action, action_args, action_kwargs)
            asyncs.append(async_result)


        if not async_return:
            for async_result in tqdm(asyncs, desc=desc, disable=not print_progress):
                results.append(async_result.get())
            pool.close()
            pool.join()
            return results
        else:
            return pool
    else:
        results = []
        length = get_length(args, kwargs)
        for i in tqdm(range(length), desc=desc, disable=not print_progress):
            action_args, action_kwargs = get_action_args(length, args, kwargs, i)
            async_result = action(*action_args, **action_kwargs)
            results.append(async_result)
        return results
