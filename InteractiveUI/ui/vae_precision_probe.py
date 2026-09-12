import copy
import gc
import json
import math
import hashlib
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch
import torch.nn.functional as F

VARIANTS = ('fp32', 'full_bf16', 'full_fp16', 'conv_bf16', 'conv_fp16',
            'head_bf16', 'head_fp16', 'late_bf16', 'late_fp16')


def make_replica(module, variant, kind='decoder'):
    if variant not in VARIANTS:
        raise ValueError(f'Unknown precision variant: {variant}')
    replica = copy.deepcopy(module).eval().requires_grad_(False)
    low = torch.bfloat16 if variant.endswith('bf16') else torch.float16
    selected = []
    if variant.startswith('full_'):
        replica.to(dtype=low)
        return replica, {'variant': variant, 'inputDtype': str(low), 'selectedConvs': 'all',
                         'note': 'Stock low-precision path; FP32-only fusions fall back.'}
    if variant != 'fp32':
        for name, conv in replica.named_modules():
            if not isinstance(conv, (torch.nn.Conv2d, torch.nn.Conv3d)):
                continue
            if variant.startswith(('head_', 'late_')) and (
                    name in ('conv_in', 'conv_out') or 'attentions' in name):
                continue
            sensitive = ('up_blocks.3.',) if kind == 'decoder' else ('down_blocks.0.','down_blocks.1.','down_blocks.2.')
            if variant.startswith('late_') and name.startswith(sensitive):
                continue
            original = conv._conv_forward
            weight = conv.weight.detach().to(low)
            bias = conv.bias.detach().to(low) if conv.bias is not None else None
            def forward(self, value, ignored_weight, ignored_bias,
                        _original=original, _weight=weight, _bias=bias, _dtype=low):
                return _original(value.to(_dtype), _weight, _bias).to(value.dtype)
            conv._conv_forward = MethodType(forward, conv)
            selected.append(name)
    return replica, {'variant': variant, 'inputDtype': 'torch.float32', 'selectedConvs': selected,
                     'note': 'Instance _conv_forward; FP32 activations/cache/norm outside conv; preconverted weight copies.'}


def clone_cache(cache, dtype=None):
    return [x.detach().to(dtype=dtype or x.dtype, copy=True) if isinstance(x, torch.Tensor) else x for x in cache]


def append(path, data):
    with Path(path).open('a') as handle:
        handle.write(json.dumps(data, allow_nan=False) + '\n')


def tensor_error(reference, candidate):
    a, b = reference.float(), candidate.float()
    finite = bool(torch.isfinite(b).all().item())
    if not finite:
        return {'finite': False, 'nonfinite': int((~torch.isfinite(b)).sum().item())}
    delta = b - a
    return {'finite': True, 'mae': delta.abs().mean().item(),
            'rmse': delta.square().mean().sqrt().item(), 'maxAbs': delta.abs().max().item(),
            'referenceRms': a.square().mean().sqrt().item()}


def image_error(reference, candidate, previous_error=None):
    result = tensor_error(reference, candidate)
    if not result['finite']:
        return result, None
    a = reference.float().clamp(-1, 1).add(1).mul(.5)
    b = candidate.float().clamp(-1, 1).add(1).mul(.5)
    error = b-a
    per_frame_mse = error.square().mean(dim=(0, 1, 3, 4))
    psnr = -10 * torch.log10(per_frame_mse.clamp_min(1e-12))
    aa = a.permute(0,2,1,3,4).reshape(-1,3,a.shape[-2],a.shape[-1])
    bb = b.permute(0,2,1,3,4).reshape_as(aa)
    pool = lambda x: F.avg_pool2d(x, 11, stride=1)
    ma, mb = pool(aa), pool(bb)
    va, vb = pool(aa*aa)-ma*ma, pool(bb*bb)-mb*mb
    cov = pool(aa*bb)-ma*mb
    ssim = ((2*ma*mb+.01**2)*(2*cov+.03**2) /
            ((ma*ma+mb*mb+.01**2)*(va+vb+.03**2)))
    differences = error[:,:,1:]-error[:,:,:-1]
    if previous_error is not None:
        differences = torch.cat([error[:,:,:1]-previous_error, differences], dim=2)
    result.update(pixelMae=error.abs().mean().item(), pixelMaxAbs=error.abs().max().item(),
                  framePsnr=psnr.detach().cpu().tolist(), ssimUniform11=ssim.mean().item(),
                  temporalMae=differences.abs().mean().item() if differences.numel() else None)
    return result, error[:,:,-1:].detach().clone()


def rng_state(device):
    return torch.get_rng_state().clone(), torch.cuda.get_rng_state(device).clone()


def check_rng(before, device):
    after = rng_state(device)
    if not all(torch.equal(a,b) for a,b in zip(before,after)):
        raise AssertionError('Precision diagnostic changed the serving RNG stream')


class DecoderShadow:
    def __init__(self, vae, live):
        self.root = live.root / 'precision'; self.root.mkdir(exist_ok=True)
        self.options = dict(live.vae_decode_options)
        self.policy = dict(live.precision_probe)
        self.variants = tuple(self.policy.get('variants', VARIANTS))
        if 'fp32' not in self.variants:
            raise ValueError('Shadow tests require the FP32 control')
        self.models = {}; self.caches = {}; self.previous = {}; self.disabled = set(); self.calls = 0
        for variant in self.variants:
            decoder, metadata = make_replica(vae.decoder, variant)
            self.models[variant] = SimpleNamespace(decoder=decoder)
            dtype = next(decoder.parameters()).dtype
            self.caches[variant] = clone_cache(vae._feat_map, dtype)
            append(self.root/'models.jsonl', {**metadata, 'kind':'decoder',
                   'weightDtype': str(dtype), 'device':str(next(decoder.parameters()).device)})

    def run(self, vae, x, first_chunk, reference, live, slice_index):
        from .vae_decode_runtime import _run, tensor_exact, cache_exact
        before = rng_state(x.device)
        order = list(self.variants)
        shift = self.calls % len(order); order = order[shift:] + order[:shift]
        snapshots = {'reference':reference.detach().cpu()} if live.chunk in self.policy.get('saveChunks',[0,2,8]) and slice_index==4 else None
        for variant in order:
            if variant in self.disabled:
                continue
            model=self.models[variant]; dtype=next(model.decoder.parameters()).dtype
            begin=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
            begin.record()
            actual,count,detail=_run(model,x.to(dtype),self.caches[variant],first_chunk,self.options)
            actual=actual.float()
            end.record();end.synchronize()
            if variant=='fp32':
                if not tensor_exact(reference,actual) or not cache_exact(vae._feat_map,self.caches[variant]):
                    raise AssertionError('Independent FP32 shadow does not match live output/cache')
            metrics,last=image_error(reference,actual,self.previous.get(variant))
            self.previous[variant]=last
            if not metrics['finite']:
                self.disabled.add(variant)
            row={'kind':'decoder','variant':variant,'chunk':live.chunk,'slice':slice_index,
                 'call':self.calls,'firstChunk':bool(first_chunk),'cudaMs':begin.elapsed_time(end),
                 'metrics':metrics,'cacheCount':count,
                 'cacheBytes':sum(v.numel()*v.element_size() for v in self.caches[variant] if isinstance(v,torch.Tensor)),
                 'jointCalls':detail.get('jointCache',{}).get('fusedCalls',0),
                 'outputDtypeBeforeBoundary':str(dtype),'usedForGeneration':False}
            append(self.root/'decoder.jsonl',row)
            if snapshots is not None:
                snapshots[variant]=actual.detach().cpu()
        if snapshots is not None:
            torch.save(snapshots,self.root/f'pixels-{live.chunk:03d}-{slice_index}.pt')
        self.calls+=1
        check_rng(before,x.device)


def prepare_decoder(vae, live):
    if not live.precision_probe or not live.precision_probe.get('decoder',True):
        return
    if not hasattr(live,'_precision_decoder'):
        before=rng_state(next(vae.decoder.parameters()).device)
        live._precision_decoder=DecoderShadow(vae,live)
        check_rng(before,next(vae.decoder.parameters()).device)


def shadow_decode(vae,x,first_chunk,reference,live,slice_index):
    if hasattr(live,'_precision_decoder'):
        live._precision_decoder.run(vae,x,first_chunk,reference,live,slice_index)


def encoder_probe(vae,video,reference,live,options):


    from .vae_encoder_runtime import encoder_mode
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
    root=live.root/'precision';root.mkdir(exist_ok=True)
    if video.ndim!=5 or video.shape[2]!=33 or vae.config.patch_size is not None or vae.use_tiling:
        raise ValueError('Precision encoder probe requires the current untiled 33-frame Wan2.1 contract')
    before=rng_state(video.device)
    variants=tuple(live.precision_probe.get('variants',VARIANTS))
    for variant in variants:
        encoder,metadata=make_replica(vae.encoder,variant,'encoder')
        quant=copy.deepcopy(vae.quant_conv).eval().requires_grad_(False)
        dtype=next(encoder.parameters()).dtype


        proxy=SimpleNamespace(encoder=encoder)
        def execute():
            cache=[None]*vae._cached_conv_counts['encoder'];out=None
            with encoder_mode(proxy,options):
                for i in range(9):
                    inp=video[:,:,:1] if i==0 else video[:,:,1+4*(i-1):1+4*i]
                    part=encoder(inp.to(dtype),feat_cache=cache,feat_idx=[0])
                    out=part if out is None else torch.cat([out,part],2)
            return quant(out.float())
        with torch.no_grad():
            actual=execute();actual=execute();torch.cuda.current_stream().synchronize()
            if variant=='fp32' and not torch.equal(reference.view(torch.int32),actual.view(torch.int32)):
                raise AssertionError('FP32 encoder replica differs from production full posterior')
            metrics=tensor_error(reference,actual)
            if metrics['finite']:
                expected=DiagonalGaussianDistribution(reference.float())
                candidate=DiagonalGaussianDistribution(actual.float())
                generator=torch.Generator(device=video.device).manual_seed(20260911)
                epsilon=torch.randn(expected.mean.shape,device=video.device,dtype=torch.float32,generator=generator)
                metrics.update(mean=tensor_error(expected.mean,candidate.mean),
                    logvar=tensor_error(expected.logvar,candidate.logvar),
                    fixedNoiseSample=tensor_error(expected.mean+expected.std*epsilon,candidate.mean+candidate.std*epsilon))
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,capture_error_mode='thread_local'):
                graph_output=execute()
            samples=[]
            for i in range(6):
                begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                begin.record();graph.replay();end.record();end.synchronize()
                samples.append(begin.elapsed_time(end))
            graph_metrics=tensor_error(actual,graph_output)
            if variant=='fp32' and not torch.equal(reference.view(torch.int32),graph_output.view(torch.int32)):
                raise AssertionError('FP32 encoder graph control differs from production posterior')
            append(root/'encoder.jsonl',{'kind':'encoder','variant':variant,'metadata':metadata,
                   'metrics':metrics,'graphVsEager':graph_metrics,'graphCudaMs':samples,
                   'graphVsReference':tensor_error(reference,graph_output),
                   'posteriorDtype':'torch.float32','usedForGeneration':False})
            del graph,graph_output,actual,encoder,proxy,quant
            gc.collect()
    check_rng(before,video.device)


def close_probe(live):
    if hasattr(live,'_precision_decoder'):
        del live._precision_decoder
    if hasattr(live,'_precision_trial_decoder'):
        del live._precision_trial_decoder
    if hasattr(live,'_precision_trial_encoder'):
        del live._precision_trial_encoder
    gc.collect()
    if hasattr(live,'_precision_rng_scope'):
        live._precision_rng_scope.__exit__(None,None,None)
        restored=_rng_fingerprint(live._precision_rng_devices)
        append(live.root/'precision/rng.jsonl',{'phase':'restored','exact':restored==live._precision_rng_original})
        if restored!=live._precision_rng_original:
            raise AssertionError('Precision trial did not restore original RNG state')
        del live._precision_rng_scope


def prepare_trial(vae,live):

    variant=live.precision_trial.get('decoder')
    if variant is None:
        return
    if live.precision_probe:
        raise ValueError('Select shadow comparison or closed-loop trial, not both')
    if any(live.vae_decode_options.get(key) for key in ('verify','profile','benchmark','groupProbe')):
        raise ValueError('Precision trial cannot use the FP32 exact-audit flags')
    if not hasattr(live,'_precision_trial_decoder'):
        before=rng_state(next(vae.decoder.parameters()).device)
        decoder,metadata=make_replica(vae.decoder,variant)
        live._precision_trial_decoder=SimpleNamespace(decoder=decoder)
        root=live.root/'precision';root.mkdir(exist_ok=True)
        append(root/'models.jsonl',{**metadata,'kind':'closed-loop-decoder','usedForGeneration':True})
        check_rng(before,next(vae.decoder.parameters()).device)


def trial_decode(vae,x,first_chunk,live):
    from .vae_decode_runtime import _run
    model=live._precision_trial_decoder
    dtype=next(model.decoder.parameters()).dtype


    output,count,detail=_run(model,x.to(dtype),vae._feat_map,first_chunk,live.vae_decode_options)
    if not torch.isfinite(output).all().item():
        raise AssertionError('Precision trial produced nonfinite decoder pixels')
    vae._conv_idx=[count]
    return output.float(),{'precisionTrial':live.precision_trial['decoder'],
        'finite':True,'verified':False,'profiled':False,'benchmark':None,**detail}


def save_trial_pixels(output,live,slice_index):
    if slice_index==4 and live.chunk in live.precision_trial.get('saveChunks',[0,2,8,32,64,95]):
        root=live.root/'precision';root.mkdir(exist_ok=True)
        torch.save(output.detach().cpu(),root/f'trial-pixels-{live.chunk:03d}.pt')


def save_trial_geometry(state,source_chunk,live):
    if source_chunk not in live.precision_trial.get('saveChunks',[0,2,8,32,64,95]):return
    bank=state.get('da3_bank');root=live.root/'precision';root.mkdir(exist_ok=True)
    if bank is None or not bank.frames:
        append(root/'geometry.jsonl',{'chunk':source_chunk,'available':False});return
    gid=max(bank.frames);depth,intr,c2w,rgb=bank.frames[gid]
    snapshot={name:torch.as_tensor(value).detach().cpu() for name,value in
              (('depth',depth),('intrinsic',intr),('c2w',c2w),('rgb',rgb))}
    snapshot.update(gid=gid,chunk=source_chunk)
    torch.save(snapshot,root/f'geometry-{source_chunk:03d}.pt')
    d=snapshot['depth'];valid=torch.isfinite(d)&(d>0)
    append(root/'geometry.jsonl',{'chunk':source_chunk,'gid':gid,'available':True,
        'finite':bool(torch.isfinite(d).all()),'validFraction':valid.float().mean().item(),
        'medianDepth':d[valid].median().item() if valid.any() else None,
        'bankFrames':len(bank.frames)})


class EncoderReplica(torch.nn.Module):

    def __init__(self,vae,variant):
        super().__init__()
        if vae.config.patch_size is not None or vae.use_tiling:
            raise ValueError('Precision trial requires untiled Wan2.1 encoder')
        self.encoder,self.metadata=make_replica(vae.encoder,variant,'encoder')
        self.quant_conv=copy.deepcopy(vae.quant_conv).eval().requires_grad_(False)
        self.config=vae.config;self.use_tiling=False;self.use_slicing=False
        self._cached_conv_counts=dict(vae._cached_conv_counts)
        self.clear_cache();self.eval().requires_grad_(False)

    @property
    def dtype(self):return torch.float32

    @property
    def device(self):return next(self.encoder.parameters()).device

    def clear_cache(self):
        self._feat_map=[None]*self._cached_conv_counts['decoder'];self._conv_idx=[0]
        self._enc_feat_map=[None]*self._cached_conv_counts['encoder'];self._enc_conv_idx=[0]

    def encode(self,video):
        from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
        if video.shape[2] not in (1,33):
            raise ValueError('Precision encoder supports the current 1/33-frame recipe')
        self.clear_cache();out=None;dtype=next(self.encoder.parameters()).dtype
        for i in range(1+(video.shape[2]-1)//4):
            self._enc_conv_idx=[0]
            inp=video[:,:,:1] if i==0 else video[:,:,1+4*(i-1):1+4*i]
            part=self.encoder(inp.to(dtype),feat_cache=self._enc_feat_map,feat_idx=self._enc_conv_idx)
            out=part if out is None else torch.cat([out,part],2)
        params=self.quant_conv(out.float());self.clear_cache()
        return SimpleNamespace(latent_dist=DiagonalGaussianDistribution(params))


def get_trial_encoder(vae,live):
    if not hasattr(live,'_precision_trial_encoder'):
        before=rng_state(vae.device)
        live._precision_trial_encoder=EncoderReplica(vae,live.precision_trial['encoder'])
        root=live.root/'precision';root.mkdir(exist_ok=True)
        append(root/'models.jsonl',{**live._precision_trial_encoder.metadata,
            'kind':'closed-loop-encoder','posteriorDtype':'torch.float32','usedForGeneration':True})
        check_rng(before,vae.device)
    return live._precision_trial_encoder


def _rng_fingerprint(devices):
    states={'cpu':torch.get_rng_state()}
    states.update({str(d):torch.cuda.get_rng_state(d) for d in devices})
    return {k:hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest() for k,v in states.items()}


def begin_trial_rng(live,settings):


    seed=live.precision_trial.get('seed')
    if seed is None:return
    if type(seed) is not int or not 0<=seed<2**63:raise ValueError('Invalid trial seed')
    devices=sorted({torch.cuda.current_device(),4,settings.get('encodeGpu',7),*settings.get('warpGpus',[4,5,6])})
    live._precision_rng_devices=devices
    live._precision_rng_original=_rng_fingerprint(devices)
    scope=torch.random.fork_rng(devices=devices);scope.__enter__()
    live._precision_rng_scope=scope
    try:
        torch.random.default_generator.manual_seed(seed)
        for device in devices:
            with torch.cuda.device(device):torch.cuda.manual_seed(seed)
        root=live.root/'precision';root.mkdir(exist_ok=True)
        append(root/'rng.jsonl',{'phase':'start','seed':seed,'fingerprint':_rng_fingerprint(devices)})
    except BaseException:
        scope.__exit__(None,None,None)
        del live._precision_rng_scope
        raise
