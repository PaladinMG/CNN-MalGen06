from __future__ import annotations

import argparse

import torch

from data import _color_jitter_tensor, _precompute_handcrafted_features
from model import AccurateTissueNet
from train import (
    batched_color_jitter,
    complete_loss,
    device_channels_last_collate,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--forward-only", action="store_true")
    args = parser.parse_args()
    if args.size < 64 or args.size % 32:
        parser.error("--size must be a multiple of 32 and at least 64")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    device = torch.device(
        "cuda"
        if args.device == "cuda"
        or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    amp = device.type == "cuda"
    torch.manual_seed(7)

    model = AccurateTissueNet(pretrained=False)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    model.to(device).train()

    local = torch.rand(1, 3, args.size, args.size, device=device)
    context = torch.rand_like(local)
    if device.type == "cuda":
        local = local.contiguous(memory_format=torch.channels_last)
        context = context.contiguous(memory_format=torch.channels_last)

    jitter_parameters = (
        (0.91, 1.08, 0.83, 1.11),
        (1.07, 0.94, 1.18, 0.89),
    )
    geometry_codes = (10, 5)
    cached_samples = []
    for sample_index, parameters in enumerate(jitter_parameters):
        cached_rgb = (
            local[0].mul(255).round().to(torch.uint8) + sample_index
        ).clamp_max(255)
        cached_mask = torch.randint(
            0, 6, (args.size, args.size), dtype=torch.uint8, device=device
        )
        cached_samples.append(
            {
                "local": cached_rgb,
                "context": cached_rgb,
                "mask": cached_mask,
                "boundary": cached_mask.eq(1).to(torch.uint8),
                "color_jitter": parameters,
                "geometry_code": geometry_codes[sample_index],
                "name": f"cached_{sample_index}",
            }
        )
    cached_batch = device_channels_last_collate(cached_samples)
    batched_augmented = batched_color_jitter(
        cached_batch["local"],
        cached_batch["color_jitter"],
    )
    ordered_samples = sorted(
        cached_samples, key=lambda sample: sample["geometry_code"]
    )

    def apply_reference_geometry(
        image: torch.Tensor, code: int
    ) -> torch.Tensor:
        if code & 4:
            image = torch.flip(image, dims=[2])
        if code & 8:
            image = torch.flip(image, dims=[1])
        if code & 3:
            image = torch.rot90(image, code & 3, dims=(1, 2))
        return image

    reference_augmented = torch.stack(
        [
            _color_jitter_tensor(
                apply_reference_geometry(
                    sample["local"], sample["geometry_code"]
                ),
                *sample["color_jitter"],
            )
            for sample in ordered_samples
        ]
    )
    augmentation_max_error = (
        batched_augmented.sub(reference_augmented).abs().max().item()
    )
    assert augmentation_max_error < 1e-6
    assert cached_batch["mask"].dtype == torch.long
    assert cached_batch["boundary"].dtype == torch.float32
    assert cached_batch["muscle_presence"].shape == (2,)

    # Exercise the same tiled, float16 feature-cache path used by training and
    # verify that its values stay close to a direct full-image calculation.
    local_uint8 = local[0].mul(255).round().to(torch.uint8)
    with torch.no_grad():
        handcrafted = _precompute_handcrafted_features(
            local_uint8,
            model.handcrafted,
            # Force internal tile boundaries even in the smallest smoke test.
            tile_size=max(16, args.size // 2),
        ).unsqueeze(0)
        direct_handcrafted = model.handcrafted(
            local_uint8.float().div(255.0).unsqueeze(0)
        )
    feature_cache_max_error = (
        handcrafted.float()
        .sub(direct_handcrafted.float())
        .abs()
        .max()
        .item()
    )
    assert feature_cache_max_error < 0.01
    if device.type == "cpu":
        # The production cache is CUDA-only, where AMP accepts its FP16
        # storage. CPU convolution requires the input and weights to match.
        handcrafted = handcrafted.float()

    mask = torch.randint(0, 6, (1, args.size, args.size), device=device)
    boundary = torch.zeros(1, args.size, args.size, device=device)
    boundary[:, :, args.size // 2 - 1 : args.size // 2 + 2] = 1
    muscle_presence = torch.ones(1, device=device)

    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=amp,
    ):
        outputs = model(local, context, handcrafted)
        loss, components, probabilities = complete_loss(
            model,
            outputs,
            mask,
            boundary,
            muscle_presence,
            class_weights=None,
        )

    assert probabilities.shape == (1, 6, args.size, args.size)
    assert torch.isfinite(probabilities).all()
    assert torch.isfinite(loss)
    maximum_sum_error = (
        probabilities.sum(1).sub(1).abs().max().detach().item()
    )
    assert maximum_sum_error < 1e-4

    if not args.forward_only:
        loss.backward()
        finite_gradient = any(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
        assert finite_gradient

    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"PASS device={device} torch={torch.__version__} "
        f"parameters={parameters:,} loss={loss.item():.5f} "
        f"probability_sum_max_error={maximum_sum_error:.3g} "
        f"feature_cache_max_error={feature_cache_max_error:.3g} "
        f"batched_augmentation_max_error={augmentation_max_error:.3g}"
    )
    print(
        "loss_components "
        + " ".join(f"{name}={value:.5f}" for name, value in components.items())
    )
    if device.type == "cuda":
        print(
            f"gpu={torch.cuda.get_device_name(0)} "
            f"peak_allocated_mb={torch.cuda.max_memory_allocated() / 2**20:.1f}"
        )


if __name__ == "__main__":
    main()
