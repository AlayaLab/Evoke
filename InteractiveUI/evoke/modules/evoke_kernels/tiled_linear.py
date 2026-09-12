import functools
import math
from typing import Callable, List, Optional

import torch
import torch.nn as nn

from diffusers.models.activations import GEGLU, GELU, ApproximateGELU, LinearActivation, SwiGLU
from diffusers.utils import deprecate


def replace_linear_with_tiled_linear(model, num_shards=None, patch_by_names=True, patch_by_types=True):
    target_names = ["to_q", "to_k", "to_v", "add_k_proj", "add_v_proj"]
    target_types = ["FeedForward"]

    patched_count = 0

    def tiled_forward(self, x):
        compute_params = list(self.parameters())
        return apply_tiled_linear(
            fn=lambda module, input: module._original_forward(input),
            mlp_module=self,
            x=x,
            num_shards=num_shards,
            compute_params=compute_params,
        )

    for name, module in model.named_modules():
        layer_name = name.rsplit(".", 1)[-1] if "." in name else name
        module_type = type(module).__name__

        should_patch = False
        if patch_by_types and module_type in target_types:
            should_patch = True
        if patch_by_names and layer_name in target_names and isinstance(module, torch.nn.Linear):
            should_patch = True

        if should_patch:
            module._original_forward = module.forward
            module.forward = tiled_forward.__get__(module, module.__class__)
            patched_count += 1

    print(f"Patched {patched_count} FeedForward modules with TiledMLP\n")
    return model


def ensure_contiguous(fn):
    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        def maybe_to_contiguous(x):
            return x.contiguous() if isinstance(x, torch.Tensor) else x

        args = [maybe_to_contiguous(arg) for arg in args]
        kwargs = {k: maybe_to_contiguous(v) for k, v in kwargs.items()}
        return fn(ctx, *args, **kwargs)

    return wrapper


class TiledLinear(torch.autograd.Function):


    @staticmethod
    @ensure_contiguous
    def forward(
        ctx,
        fn: Callable,
        mlp_module: torch.nn.Module,
        x: torch.Tensor,
        shards: int,
        compute_params: Optional[List[torch.nn.Parameter]] = None,
    ) -> torch.Tensor:
        ctx.fn = fn
        ctx.mlp_module = mlp_module
        ctx.shards = shards
        ctx.save_for_backward(x)


        x_shards = list(torch.chunk(x, chunks=shards, dim=-2))
        with torch.no_grad():
            output_shards = [fn(mlp_module, x_shard) for x_shard in x_shards]
        output_unsharded = torch.cat(output_shards, dim=-2)

        return output_unsharded

    @staticmethod
    @ensure_contiguous
    def backward(ctx, *grads) -> tuple:
        fn = ctx.fn
        (x,) = ctx.saved_tensors
        mlp_module = ctx.mlp_module
        shards = ctx.shards

        x_requires_grad = x.requires_grad
        x = x.detach()

        x.requires_grad_(x_requires_grad)


        hidden_size = x.shape[-1]
        x_shape_orig = x.shape


        x = x.view(-1, hidden_size)
        incoming_grad = grads[0].view(-1, hidden_size)
        x_grad = torch.zeros_like(x)

        x_shards = list(torch.chunk(x, chunks=shards, dim=0))

        trainable_params = [p for p in mlp_module.parameters() if p.requires_grad]

        for i, x_shard in enumerate(x_shards):
            x_shard = x_shard.detach().requires_grad_(x_requires_grad)

            shard_step = x_shards[i].shape[0]
            shard_offset = i * x_shards[0].shape[0]

            incoming_grad_shard = incoming_grad.narrow(0, shard_offset, shard_step).view_as(x_shard)

            with torch.enable_grad():
                output = fn(mlp_module, x_shard)

            grads_tuple = torch.autograd.grad(
                outputs=output,
                inputs=[x_shard] + trainable_params,
                grad_outputs=incoming_grad_shard,
                allow_unused=True,
                retain_graph=False,
            )

            x_grad.narrow(0, shard_offset, shard_step).copy_(grads_tuple[0])

            for param, grad in zip(trainable_params, grads_tuple[1:]):
                if grad is not None:
                    if param.grad is None:
                        param.grad = grad
                    else:
                        param.grad.add_(grad)


        x_grad = x_grad.view(x_shape_orig)

        return (None, None, x_grad, None, None)


def apply_tiled_linear(
    fn: Callable,
    mlp_module: torch.nn.Module,
    x: torch.Tensor,
    num_shards: Optional[int] = None,
    compute_params: Optional[List[torch.nn.Parameter]] = None,
) -> torch.Tensor:


    if num_shards is None:

        hidden_size = x.shape[-1]
        seqlen = x.shape[-2]
        num_shards = math.ceil(seqlen / hidden_size)


    num_shards = max(1, num_shards)

    return TiledLinear.apply(
        fn,
        mlp_module,
        x,
        num_shards,
        compute_params,
    )


class FeedForward(nn.Module):


    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
        inner_dim=None,
        bias: bool = True,
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim, bias=bias)
        if activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh", bias=bias)
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim, bias=bias)
        elif activation_fn == "geglu-approximate":
            act_fn = ApproximateGELU(dim, inner_dim, bias=bias)
        elif activation_fn == "swiglu":
            act_fn = SwiGLU(dim, inner_dim, bias=bias)
        elif activation_fn == "linear-silu":
            act_fn = LinearActivation(dim, inner_dim, bias=bias, activation="silu")

        self.net = nn.ModuleList([])
        self.net.append(act_fn)
        self.net.append(nn.Dropout(dropout))
        self.net.append(nn.Linear(inner_dim, dim_out, bias=bias))
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class TiledFeedForward(nn.Module):


    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
        inner_dim: Optional[int] = None,
        bias: bool = True,
        num_shards: Optional[int] = None,
    ):
        super().__init__()

        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        self.dim = dim
        self.inner_dim = inner_dim
        self.dim_out = dim_out
        self.activation_fn = activation_fn
        self.num_shards = num_shards

        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim, bias=bias)
        if activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh", bias=bias)
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim, bias=bias)
        elif activation_fn == "geglu-approximate":
            act_fn = ApproximateGELU(dim, inner_dim, bias=bias)
        elif activation_fn == "swiglu":
            act_fn = SwiGLU(dim, inner_dim, bias=bias)
        elif activation_fn == "linear-silu":
            act_fn = LinearActivation(dim, inner_dim, bias=bias, activation="silu")

        self.net = nn.ModuleList([])
        self.net.append(act_fn)
        self.net.append(nn.Dropout(dropout))
        self.net.append(nn.Linear(inner_dim, dim_out, bias=bias))
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def _mlp_forward(self, module, x):

        for layer in module.net:
            x = layer(x)
        return x

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:

        compute_params = list(self.parameters())

        return apply_tiled_linear(
            fn=self._mlp_forward,
            mlp_module=self,
            x=hidden_states,
            num_shards=self.num_shards,
            compute_params=compute_params,
        )


if __name__ == "__main__":
    import torch
    import torch.nn as nn


    torch.manual_seed(42)

    batch_size, seq_len, hidden_dim = 2, 1024, 768
    x = torch.randn(batch_size, seq_len, hidden_dim, requires_grad=True)


    model1 = FeedForward(dim=hidden_dim)

    out1 = model1(x)
    loss1 = out1.sum()
    loss1.backward()
    grad1 = x.grad.clone()


    x.grad = None

    model2 = FeedForward(dim=hidden_dim)
    model2 = replace_linear_with_tiled_linear(model2, num_shards=4)

    model2.load_state_dict(model1.state_dict(), strict=True)
    out2 = model2(x)
    loss2 = out2.sum()
    loss2.backward()
    grad2 = x.grad.clone()


    print(f"Output diff: {(out1 - out2).abs().max().item()}")
    print(f"Gradient diff: {(grad1 - grad2).abs().max().item()}")
    print(f"Output allclose: {torch.allclose(out1, out2, atol=1e-6)}")
    print(f"Gradient allclose: {torch.allclose(grad1, grad2, atol=1e-6)}")
