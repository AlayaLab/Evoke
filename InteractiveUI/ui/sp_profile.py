from collections import defaultdict
import time

import torch


class SPProfile:
    def __init__(self, index):
        self.index = index
        self.started = time.perf_counter()
        self.cpu = defaultdict(float)
        self.events = []
        self.parents = defaultdict(set)
        self.module_stack = []
        self.detail_blocks = (0, 19, 39)

    def begin(self, name, byte_count=0):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        self.events.append((name, start, end, byte_count))
        if self.module_stack:
            self.parents[name].add(self.module_stack[-1])
        return end

    def collective(self, op, tensor, call):
        end = self.begin(op, tensor.numel() * tensor.element_size())
        started = time.perf_counter()
        result = call()
        self.cpu[op + "LaunchSeconds"] += time.perf_counter() - started
        end.record()
        return result

    def run_model(self, model, call):
        from evoke.modules import student_sp
        pending = {}
        handles = []
        def watch(module, name):
            if not isinstance(module, torch.nn.Module):
                return
            def before(module, inputs):
                pending.setdefault(name, []).append(self.begin(name))
                self.module_stack.append(name)
            def after(module, inputs, output):
                pending[name].pop().record()
                actual = self.module_stack.pop()
                if actual != name:
                    raise RuntimeError(f'SP profile hook nesting mismatch: {actual} != {name}')
            handles.extend((module.register_forward_pre_hook(before), module.register_forward_hook(after)))

        for index, block in enumerate(model.blocks):
            watch(block, f"block_{index:02d}")
            if index not in self.detail_blocks:
                continue


            prefix = f"detail.block_{index:02d}"
            for name in ('norm1', 'norm2', 'norm3', 'attn1', 'attn2', 'ffn'):
                watch(getattr(block, name, None), f'{prefix}.{name}')
            for attention_name in ('attn1', 'attn2'):
                attention = getattr(block, attention_name, None)
                if attention is None:
                    continue
                for name in ('to_q', 'to_k', 'to_v', 'to_qkv', 'to_kv', 'norm_q', 'norm_k'):
                    watch(getattr(attention, name, None), f'{prefix}.{attention_name}.{name}')
                projection = getattr(attention, 'to_out', None)
                if isinstance(projection, (torch.nn.ModuleList, torch.nn.Sequential)) and len(projection):
                    watch(projection[0], f'{prefix}.{attention_name}.to_out.0')
            ffn_net = getattr(getattr(block, 'ffn', None), 'net', None)
            if isinstance(ffn_net, (torch.nn.ModuleList, torch.nn.Sequential)):
                for net_index, layer in enumerate(ffn_net):
                    if isinstance(layer, torch.nn.Dropout):
                        continue
                    watch(layer, f'{prefix}.ffn.net.{net_index}')


                    watch(getattr(layer, 'proj', None), f'{prefix}.ffn.net.{net_index}.proj')
            for name in ('cam_inj1_down_proj', 'cam_inj1_up_proj',
                         'cam_inj2_down_proj', 'cam_inj2_up_proj',
                         'cam_scale_down_proj', 'cam_scale_up_proj',
                         'cam_shift_down_proj', 'cam_shift_up_proj'):
                watch(getattr(block, name, None), f'{prefix}.{name}')
        token = student_sp._profile_collective.set(self.collective)
        end = self.begin("model")
        self.module_stack.append('model')
        try:
            return call()
        finally:
            end.record()
            student_sp._profile_collective.reset(token)
            for handle in handles:
                handle.remove()
            self.module_stack.clear()

    def result(self):

        stages = {}
        for name, start, end, byte_count in self.events:
            item = stages.setdefault(name, {"count": 0, "gpuMilliseconds": 0., "bytes": 0})
            item["count"] += 1
            item["gpuMilliseconds"] += start.elapsed_time(end)
            item["bytes"] += byte_count
        for name, item in stages.items():
            item['parents'] = sorted(self.parents[name])
        return {"forwardIndex": self.index, "cpuSeconds": dict(self.cpu), "gpuStages": stages,
                "detailBlockIndices": list(self.detail_blocks),
                "wallSeconds": time.perf_counter() - self.started,
                "timingNote": "Nested GPU ranges; parents lists enclosing module ranges. Collective names aggregate calls across parents and include stream waits. Sparse module hooks do not isolate functional FlashAttention, RoPE, camera scatter, or FP32 AdaLN arithmetic. Profiling perturbs launch timing."}
