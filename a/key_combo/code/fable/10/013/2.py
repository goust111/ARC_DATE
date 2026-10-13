from __future__ import annotations

from collections import Counter, deque
import hashlib
import random

import numpy as np


_SEEN_OUTPUTS: set[str] = set()


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude is not None:
        colors -= set(exclude)
    return sorted(colors)


def _find_vessels(
    grid: np.ndarray,
    vessel_color: int,
) -> list[dict[str, int | None]]:
    h, w = map(int, grid.shape)
    bottom = h - 1
    vessels: list[dict[str, int | None]] = []
    expected_mask = np.zeros_like(grid, dtype=bool)

    c = 0
    while c < w:
        if int(grid[bottom, c]) != vessel_color:
            c += 1
            continue
        start = c
        while c < w and int(grid[bottom, c]) == vessel_color:
            c += 1
        end = c - 1
        assert end - start + 1 == 3, "each vessel base must be exactly three cells wide"

        left_c, mid_c, right_c = start, start + 1, start + 2
        left_h = 0
        r = bottom
        while r >= 0 and int(grid[r, left_c]) == vessel_color:
            left_h += 1
            r -= 1

        right_h = 0
        r = bottom
        while r >= 0 and int(grid[r, right_c]) == vessel_color:
            right_h += 1
            r -= 1

        assert left_h >= 3 and right_h >= 3, "each arm must provide a visible cavity"
        assert left_h != right_h, "container arms must have different heights"

        top = bottom - max(left_h, right_h) + 1
        for rr in range(top, bottom):
            assert int(grid[rr, mid_c]) != vessel_color, "vessel interior must stay open"

        expected_mask[bottom - left_h + 1 : bottom + 1, left_c] = True
        expected_mask[bottom - right_h + 1 : bottom + 1, right_c] = True
        expected_mask[bottom, start : end + 1] = True

        direction = -1 if left_h < right_h else 1
        capacity = min(left_h, right_h) - 1
        tall_capacity = max(left_h, right_h) - 1
        vessels.append(
            {
                "left_c": left_c,
                "mid_c": mid_c,
                "right_c": right_c,
                "left_h": left_h,
                "right_h": right_h,
                "capacity": capacity,
                "tall_capacity": tall_capacity,
                "direction": direction,
                "top": top,
            }
        )

    assert vessels, "expected at least one vessel"
    assert np.array_equal(grid == vessel_color, expected_mask), "unexpected extra vessel pixels"
    return vessels


def _settle(volumes: list[int], capacities: list[int], targets: list[int | None]) -> dict[str, object]:
    total = sum(volumes)
    final = list(volumes)
    received = [0] * len(final)
    spilled = [0] * len(final)
    q = deque(i for i, v in enumerate(final) if v > capacities[i])
    in_queue = [v > capacities[i] for i, v in enumerate(final)]
    steps = 0
    limit = total * max(1, len(final)) + 20

    while q:
        i = q.popleft()
        in_queue[i] = False
        if final[i] <= capacities[i]:
            continue
        steps += 1
        assert steps <= limit, "overflow process did not terminate"

        target = targets[i]
        assert target is not None, "overflow cannot leave the vessel system"
        excess = final[i] - capacities[i]
        final[i] = capacities[i]
        spilled[i] += excess
        final[target] += excess
        received[target] += excess
        if final[target] > capacities[target] and not in_queue[target]:
            q.append(target)
            in_queue[target] = True

    assert all(v <= cap for v, cap in zip(final, capacities))
    return {
        "final": final,
        "received": received,
        "spilled": spilled,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop the airborne droplets into the narrow uneven-arm vessels and settle every short-side overflow cascade.

    Input structure
    - The dominant color is the background.
    - Along the bottom row sit several upward-open frame vessels of one shared color; each vessel is three cells wide with a one-cell cavity and unequal left/right arm heights.
    - The only other non-background color is the droplet color, appearing as single airborne pixels in the cavity column above vessels.

    Rule
    - Count all droplet pixels above each vessel's cavity column as that vessel's initial water.
    - A vessel's effective capacity is the number of cavity cells capped by its shorter arm; any excess spills over that shorter side into the nearest vessel on that side, repeating until all vessels are within capacity.
    - Remove all airborne droplets and fill each cavity bottom-up with the droplet color to its final settled amount.

    Key insight
    - The taller arm is a decoy: both the retained capacity and the overflow direction are determined locally by the shorter arm of each individual vessel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _get_bg(g)
    colors = _colors_present(g, exclude={bg})
    assert len(colors) == 2, "expected exactly one vessel color and one droplet color"

    bottom_colors = {int(x) for x in g[-1] if int(x) != bg}
    assert len(bottom_colors) == 1, "bottom row must identify the vessel color"
    vessel_color = int(next(iter(bottom_colors)))
    water_color = int([c for c in colors if c != vessel_color][0])

    vessels = _find_vessels(g, vessel_color)
    cavity_cols = {int(v["mid_c"]) for v in vessels}
    water_coords = np.argwhere(g == water_color)
    assert water_coords.size > 0, "expected at least one droplet"
    for r, c in water_coords:
        assert int(c) in cavity_cols, "droplets must align with vessel cavities"
        assert int(r) < g.shape[0] - 1, "droplets cannot occupy the floor row"

    capacities = [int(v["capacity"]) for v in vessels]
    targets = []
    for i, vessel in enumerate(vessels):
        direction = int(vessel["direction"])
        target = i + direction
        targets.append(target if 0 <= target < len(vessels) else None)

    initial = [int(np.sum(g[:, int(v["mid_c"])] == water_color)) for v in vessels]
    settled = _settle(initial, capacities, targets)
    final = settled["final"]

    out = g.copy()
    out[out == water_color] = bg
    floor = g.shape[0] - 1
    for vessel, amount in zip(vessels, final):
        mid_c = int(vessel["mid_c"])
        for k in range(int(amount)):
            out[floor - 1 - k, mid_c] = water_color
    return out


def _draw_vessels(
    height: int,
    width: int,
    left_margin: int,
    gaps: list[int],
    heights: list[tuple[int, int]],
    vessel_color: int,
) -> tuple[np.ndarray, list[int]]:
    grid = np.zeros((height, width), dtype=int)
    mids: list[int] = []
    c = left_margin
    bottom = height - 1
    for i, (left_h, right_h) in enumerate(heights):
        left_c, mid_c, right_c = c, c + 1, c + 2
        grid[bottom - left_h + 1 : bottom + 1, left_c] = vessel_color
        grid[bottom - right_h + 1 : bottom + 1, right_c] = vessel_color
        grid[bottom, left_c : right_c + 1] = vessel_color
        mids.append(mid_c)
        if i < len(gaps):
            c += 3 + gaps[i]
    return grid, mids


def _sample_structure() -> dict[str, object] | None:
    vessel_count = random.choice((4, 5))
    root = 2 if vessel_count == 5 else random.choice((1, 2))

    vessel_color, water_color = random.sample(range(1, 10), 2)

    heights: list[tuple[int, int]] = []
    capacities: list[int] = []
    tall_caps: list[int] = []
    directions: list[int] = []
    for i in range(vessel_count):
        short_h = random.randint(5, 6) if i == root else random.randint(3, 6)
        tall_h = short_h + random.randint(1, 3)
        if i < root:
            left_h, right_h = tall_h, short_h
        elif i > root:
            left_h, right_h = short_h, tall_h
        else:
            if random.choice((False, True)):
                left_h, right_h = short_h, tall_h
            else:
                left_h, right_h = tall_h, short_h
        heights.append((left_h, right_h))
        capacities.append(min(left_h, right_h) - 1)
        tall_caps.append(max(left_h, right_h) - 1)
        directions.append(-1 if left_h < right_h else 1)

    left_margin = random.randint(1, 2)
    right_margin = random.randint(1, 2)
    gaps = [random.randint(1, 3) for _ in range(vessel_count - 1)]
    width = left_margin + right_margin + vessel_count * 3 + sum(gaps)
    if width > 30:
        return None

    left_len = root
    right_len = vessel_count - root - 1
    chain_side_choices = []
    if left_len >= 2:
        chain_side_choices.append("left")
    if right_len >= 2:
        chain_side_choices.append("right")
    if not chain_side_choices:
        return None
    chain_side = random.choice(chain_side_choices)

    if chain_side == "left":
        leaf = 0
        parent = 1
        direct = root + 1
    else:
        leaf = vessel_count - 1
        parent = vessel_count - 2
        direct = root - 1

    leaf_extra_max = min(2, tall_caps[leaf] - capacities[leaf])
    direct_extra_max = min(2, tall_caps[direct] - capacities[direct])
    if leaf_extra_max < 1 or direct_extra_max < 1:
        return None

    leaf_extra = random.randint(1, leaf_extra_max)
    parent_spill = random.randint(1, leaf_extra)
    direct_extra = random.randint(1, direct_extra_max)
    incoming_root = parent_spill + direct_extra
    if capacities[root] < incoming_root:
        return None

    counts: list[int | None] = [None] * vessel_count
    counts[leaf] = capacities[leaf] + leaf_extra
    counts[parent] = capacities[parent] - leaf_extra + parent_spill
    counts[direct] = capacities[direct] + direct_extra
    counts[root] = random.randint(0, capacities[root] - incoming_root)

    stable_choices = [i for i in range(vessel_count) if counts[i] is None]
    if stable_choices:
        stable_idx = random.choice(stable_choices)
        counts[stable_idx] = random.randint(1, capacities[stable_idx])
    for i in range(vessel_count):
        if counts[i] is None:
            counts[i] = random.randint(0, capacities[i])

    initial = [int(x) for x in counts]
    max_h = max(max(left_h, right_h) for left_h, right_h in heights)
    min_height = max(max_h + initial[i] + 1 for i in range(vessel_count))
    height = min_height + random.randint(0, 3)
    if height > 30:
        return None

    return {
        "vessel_count": vessel_count,
        "root": root,
        "heights": heights,
        "capacities": capacities,
        "tall_caps": tall_caps,
        "directions": directions,
        "initial": initial,
        "leaf": leaf,
        "direct": direct,
        "leaf_extra": leaf_extra,
        "parent": parent,
        "parent_spill": parent_spill,
        "width": width,
        "height": height,
        "left_margin": left_margin,
        "gaps": gaps,
        "vessel_color": vessel_color,
        "water_color": water_color,
    }


def _render_input(spec: dict[str, object]) -> np.ndarray:
    height = int(spec["height"])
    width = int(spec["width"])
    vessel_color = int(spec["vessel_color"])
    water_color = int(spec["water_color"])
    heights = list(spec["heights"])
    gaps = list(spec["gaps"])
    left_margin = int(spec["left_margin"])
    initial = list(spec["initial"])

    grid, mids = _draw_vessels(height, width, left_margin, gaps, heights, vessel_color)
    for mid_c, amount, (left_h, right_h) in zip(mids, initial, heights):
        top = height - max(left_h, right_h)
        available_rows = list(range(0, top - 1))
        assert len(available_rows) >= amount
        rows = sorted(random.sample(available_rows, amount))
        for r in rows:
            grid[r, mid_c] = water_color
    return grid


def _output_signature(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes() + str(grid.shape).encode()).hexdigest()


def _is_semantically_good(inp: np.ndarray, spec: dict[str, object]) -> bool:
    expected = main(inp)
    vessel_color = int(spec["vessel_color"])
    vessels = _find_vessels(inp, vessel_color)
    capacities = [int(v["capacity"]) for v in vessels]
    targets = []
    for i, vessel in enumerate(vessels):
        target = i + int(vessel["direction"])
        targets.append(target if 0 <= target < len(vessels) else None)

    water_color = int(spec["water_color"])
    initial = [int(np.sum(inp[:, int(v["mid_c"])] == water_color)) for v in vessels]
    settled = _settle(initial, capacities, targets)
    final = list(settled["final"])
    received = list(settled["received"])
    spilled = list(settled["spilled"])

    overflow_dirs = {targets[i] - i for i, amount in enumerate(spilled) if amount > 0}
    if overflow_dirs != {-1, 1}:
        return False
    if not any(received[i] > 0 and spilled[i] > 0 for i in range(len(vessels))):
        return False
    if not any(final[i] > initial[i] for i in range(len(vessels))):
        return False
    if not any(final[i] < initial[i] for i in range(len(vessels))):
        return False

    leaf = int(spec["leaf"])
    tall_caps = list(spec["tall_caps"])
    if not (capacities[leaf] < initial[leaf] <= tall_caps[leaf]):
        return False

    out_sig = _output_signature(expected)
    if out_sig in _SEEN_OUTPUTS:
        return False

    rebuilt = inp.copy()
    rebuilt[rebuilt == water_color] = 0
    for vessel, amount in zip(vessels, final):
        mid_c = int(vessel["mid_c"])
        for k in range(amount):
            rebuilt[inp.shape[0] - 2 - k, mid_c] = water_color
    if not np.array_equal(rebuilt, expected):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(400):
        spec = _sample_structure()
        if spec is None:
            continue
        inp = _render_input(spec)
        if not _is_semantically_good(inp, spec):
            continue
        _SEEN_OUTPUTS.add(_output_signature(main(inp)))
        return inp
    raise RuntimeError("failed to generate a valid uneven-arm overflow puzzle")
