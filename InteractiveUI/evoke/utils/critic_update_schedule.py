

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class CriticSlot:


    critic_substep: int
    slot: int
    group_index: int
    owner_local_rank: int
    active: bool
    global_batch_size: int
    loss_scale: float

    @property
    def sample_id(self) -> Tuple[int, int] | None:
        if not self.active:
            return None
        return (self.group_index, self.owner_local_rank)


@dataclass(frozen=True)
class CriticUpdatePlan:
    world_size: int
    sp_size: int
    num_groups: int
    num_critic_steps: int
    slots_per_step: int
    outer_step: int

    assignments: Tuple[Tuple[Tuple[CriticSlot, ...], ...], ...]

    def for_group(self, group_index: int) -> Tuple[Tuple[CriticSlot, ...], ...]:
        if not 0 <= group_index < self.num_groups:
            raise IndexError(
                f"group_index={group_index} outside [0,{self.num_groups})"
            )
        return tuple(
            tuple(slot[group_index] for slot in step_slots)
            for step_slots in self.assignments
        )

    def bucket_samples(self, critic_substep: int) -> Tuple[Tuple[int, int], ...]:
        samples = []
        for slot in self.assignments[critic_substep]:
            for item in slot:
                if item.active:
                    samples.append((item.group_index, item.owner_local_rank))
        return tuple(samples)

    def bucket_sizes(self) -> Tuple[int, ...]:
        return tuple(
            len(self.bucket_samples(substep))
            for substep in range(self.num_critic_steps)
        )


def _balanced_distinct_slots(
    owners_by_group: Sequence[Sequence[int]],
    slots_per_step: int,
) -> Dict[int, Dict[int, int]]:


    loads = [0] * slots_per_step
    result: Dict[int, Dict[int, int]] = {
        group_index: {} for group_index in range(len(owners_by_group))
    }

    group_order = sorted(
        range(len(owners_by_group)),
        key=lambda group_index: (-len(owners_by_group[group_index]), group_index),
    )
    for group_index in group_order:
        owners = list(owners_by_group[group_index])
        if len(owners) > slots_per_step:
            raise AssertionError(
                f"group {group_index} has {len(owners)} samples but only "
                f"{slots_per_step} slots"
            )
        chosen_slots = sorted(
            range(slots_per_step),
            key=lambda slot: (loads[slot], slot),
        )[: len(owners)]
        for owner, slot in zip(sorted(owners), chosen_slots):
            result[group_index][slot] = int(owner)
            loads[slot] += 1
    return result


def build_critic_update_plan(
    *,
    world_size: int,
    sp_size: int,
    num_critic_steps: int,
    outer_step: int,
) -> CriticUpdatePlan:


    world_size = int(world_size)
    sp_size = int(sp_size)
    num_critic_steps = int(num_critic_steps)
    outer_step = int(outer_step)
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if sp_size <= 1 or world_size % sp_size:
        raise ValueError(
            f"need SP-on and world_size divisible by sp_size, got "
            f"world_size={world_size}, sp_size={sp_size}"
        )
    if not 1 <= num_critic_steps <= sp_size:
        raise ValueError(
            f"num_critic_steps must be in [1, sp_size], got "
            f"{num_critic_steps} for sp_size={sp_size}"
        )

    num_groups = world_size // sp_size
    slots_per_step = ceil(sp_size / num_critic_steps)
    buckets: List[List[List[int]]] = [
        [[] for _ in range(num_groups)] for _ in range(num_critic_steps)
    ]
    for group_index in range(num_groups):
        for owner_local_rank in range(sp_size):
            critic_substep = (
                outer_step + group_index + owner_local_rank
            ) % num_critic_steps
            buckets[critic_substep][group_index].append(owner_local_rank)

    assignments = []
    all_active_samples = []
    for critic_substep, owners_by_group in enumerate(buckets):
        placement = _balanced_distinct_slots(
            owners_by_group=owners_by_group,
            slots_per_step=slots_per_step,
        )
        global_batch_size = sum(len(owners) for owners in owners_by_group)
        if global_batch_size <= 0:
            raise AssertionError(f"empty critic bucket {critic_substep}")
        loss_scale = world_size / global_batch_size
        step_slots = []
        for slot in range(slots_per_step):
            group_items = []
            for group_index in range(num_groups):
                active = slot in placement[group_index]


                owner_local_rank = (
                    placement[group_index][slot]
                    if active
                    else (outer_step + critic_substep + slot + group_index)
                    % sp_size
                )
                item = CriticSlot(
                    critic_substep=critic_substep,
                    slot=slot,
                    group_index=group_index,
                    owner_local_rank=owner_local_rank,
                    active=active,
                    global_batch_size=global_batch_size,
                    loss_scale=loss_scale,
                )
                group_items.append(item)
                if item.sample_id is not None:
                    all_active_samples.append(item.sample_id)
            step_slots.append(tuple(group_items))
        assignments.append(tuple(step_slots))

    expected_samples = {
        (group_index, owner_local_rank)
        for group_index in range(num_groups)
        for owner_local_rank in range(sp_size)
    }
    active_samples = set(all_active_samples)
    if active_samples != expected_samples or len(all_active_samples) != world_size:
        raise AssertionError(
            "critic plan must cover every rollout exactly once: "
            f"active={len(all_active_samples)} unique={len(active_samples)} "
            f"expected={len(expected_samples)}"
        )

    return CriticUpdatePlan(
        world_size=world_size,
        sp_size=sp_size,
        num_groups=num_groups,
        num_critic_steps=num_critic_steps,
        slots_per_step=slots_per_step,
        outer_step=outer_step,
        assignments=tuple(assignments),
    )
