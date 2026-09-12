from contextlib import contextmanager
from types import SimpleNamespace
import threading
import weakref


_LOCKS = weakref.WeakKeyDictionary()
_LOCKS_GUARD = threading.Lock()
COMPUTE_DEFAULTS = {
    'skipZeroPad': False, 'fuseCausalInput': False, 'fuseAffine': False,
    'fuseNormDivision': False, 'memoryBlockSize': 256, 'fuseJointCache': False,
    'jointCachePlanes': False, 'jointCacheBlockSize': 1024, 'jointCacheWarps': 4,
}
CONTROL_DEFAULTS = {
    'allowGraphInitialization': False, 'benchmark': False,
    'benchmarkWarmup': 2, 'benchmarkRepeats': 3,
}


def normalize_options(options=None):
    values = dict(options or {})
    unknown = set(values) - COMPUTE_DEFAULTS.keys() - CONTROL_DEFAULTS.keys()
    if unknown:
        raise ValueError(f'Unknown VAE encoder optimization options: {sorted(unknown)}')
    result = {**COMPUTE_DEFAULTS, **CONTROL_DEFAULTS, **values}
    for key in ('skipZeroPad', 'fuseCausalInput', 'fuseAffine', 'fuseNormDivision',
                'allowGraphInitialization', 'benchmark', 'fuseJointCache', 'jointCachePlanes'):
        if type(result[key]) is not bool:
            raise ValueError(f'VAE encoder {key} must be boolean')
    if type(result['memoryBlockSize']) is not int or result['memoryBlockSize'] not in (256, 512, 1024):
        raise ValueError('VAE encoder memoryBlockSize must be 256, 512 or 1024')
    for key, upper in (('benchmarkWarmup', 10), ('benchmarkRepeats', 20)):
        if type(result[key]) is not int or not 1 <= result[key] <= upper:
            raise ValueError(f'VAE encoder {key} must be an integer in [1,{upper}]')
    if result['fuseNormDivision'] and not result['fuseAffine']:
        raise ValueError('VAE encoder fuseNormDivision requires fuseAffine')
    if type(result['jointCacheBlockSize']) is not int or result['jointCacheBlockSize'] not in (256, 512, 1024, 2048):
        raise ValueError('Invalid jointCacheBlockSize')
    if type(result['jointCacheWarps']) is not int or result['jointCacheWarps'] not in (4, 8):
        raise ValueError('Invalid jointCacheWarps')
    return result


def compute_signature(options=None):
    values = normalize_options(options)
    return tuple((key, values[key]) for key in COMPUTE_DEFAULTS)


@contextmanager
def encoder_access(vae):

    encoder = vae.encoder
    with _LOCKS_GUARD:
        lock = _LOCKS.get(encoder)
        if lock is None:
            lock = threading.RLock()
            _LOCKS[encoder] = lock
    if not lock.acquire(blocking=False):
        raise RuntimeError('Resident VAE encoder is already in use; serialize access')
    try:
        yield
    finally:
        lock.release()


@contextmanager
def encoder_mode(vae, options=None, verify=False):


    from .vae_memory import memory_mode, resolve_memory_timing
    from .vae_affine import affine_mode
    from .vae_joint_cache import joint_cache_mode

    values = normalize_options(options)
    proxy = SimpleNamespace(decoder=vae.encoder)
    stats = {'options': {key: values[key] for key in COMPUTE_DEFAULTS}}
    with encoder_access(vae):
        if getattr(vae.encoder, '_evoke_encoder_operators_active', False):
            raise RuntimeError('VAE encoder operator context is already active')
        vae.encoder._evoke_encoder_operators_active = True
        try:
            memory = {key: values[key] for key in ('skipZeroPad', 'fuseCausalInput')}
            memory.update(blockSize=values['memoryBlockSize'], verify=bool(verify), profile=False)
            with memory_mode(proxy, memory) as memory_stats, affine_mode(
                proxy, enabled=values['fuseAffine'], verify=verify,
                fuse_division=values['fuseNormDivision'],
            ) as affine_stats, joint_cache_mode(proxy, enabled=values['fuseJointCache'],
                verify=verify, block_size=values['jointCacheBlockSize'],
                planes=values['jointCachePlanes'], warps=values['jointCacheWarps']) as joint_stats:
                stats.update(memory=memory_stats, affine=affine_stats, jointCache=joint_stats)
                yield stats
            stats['memory'] = resolve_memory_timing(memory_stats)
        finally:
            del vae.encoder._evoke_encoder_operators_active
