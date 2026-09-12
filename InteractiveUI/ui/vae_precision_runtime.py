from types import SimpleNamespace


def prepare_decoder(vae):
    slot = getattr(vae, '_evoke_mixed_decoder', None)
    if slot is None:
        from .vae_precision_probe import make_replica
        decoder, metadata = make_replica(vae.decoder, 'head_fp16')
        slot = SimpleNamespace(decoder=decoder, metadata=metadata)


        vae._evoke_mixed_decoder = slot
    return slot


def decode_slice(vae, x, first_chunk, options):
    from .vae_decode_runtime import _run
    model = prepare_decoder(vae)
    output, count, detail = _run(model, x, vae._feat_map, first_chunk, options)
    vae._conv_idx = [count]
    return output, {'vaePrecision': 'fp16_mixed', 'precisionVariant': 'head_fp16',
                    'verified': False, 'profiled': False, 'benchmark': None, **detail}


def get_encoder(vae):
    slot = getattr(vae, '_evoke_mixed_encoder', None)
    if slot is None:
        from .vae_precision_probe import EncoderReplica
        slot = SimpleNamespace(encoder=EncoderReplica(vae, 'head_fp16'))
        vae._evoke_mixed_encoder = slot


    return slot.encoder
