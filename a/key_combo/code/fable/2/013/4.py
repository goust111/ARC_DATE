from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_TOOLBOX_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_TOOLBOX_DIR) not in sys.path:
    sys.path.append(str(_TOOLBOX_DIR))

from arc_utils import Color, find_components


BACKGROUND = Color.BLACK
SIDES = ("top", "right", "bottom", "left")
OPEN_TO_GRAVITY = {
    "top": (1, 0),
    "bottom": (-1, 0),
    "left": (0, 1),
    "right": (0, -1),
}
OPPOSITE_SIDE = {
    "top": "bottom",
    "bottom": "top",
    "left": "right",
    "right": "left",
}
GEN_RETRIES = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _is_consecutive(values: list[int]) -> bool:
    return bool(values) and values == list(range(values[0], values[-1] + 1))


def _edge_gaps(local_mask: np.ndarray, side: str) -> list[int]:
    h, w = local_mask.shape
    if side == "top":
        return [c for c in range(1, w - 1) if not local_mask[0, c]]
    if side == "bottom":
        return [c for c in range(1, w - 1) if not local_mask[h - 1, c]]
    if side == "left":
        return [r for r in range(1, h - 1) if not local_mask[r, 0]]
    if side == "right":
        return [r for r in range(1, h - 1) if not local_mask[r, w - 1]]
    raise ValueError(f"unknown side: {side}")


def _projection(cell: tuple[int, int], gravity: tuple[int, int]) -> int:
    return cell[0] * gravity[0] + cell[1] * gravity[1]


def _analyze_cup_mask(mask: np.ndarray, color: int) -> dict | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None

    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if min(h, w) < 4:
        return None

    local_mask = np.zeros((h, w), dtype=bool)
    local_mask[coords[:, 0] - r0, coords[:, 1] - c0] = True

    if h > 2 and w > 2 and np.any(local_mask[1:-1, 1:-1]):
        return None
    if not (local_mask[0, 0] and local_mask[0, -1] and local_mask[-1, 0] and local_mask[-1, -1]):
        return None

    gap_map = {side: _edge_gaps(local_mask, side) for side in SIDES}
    opening_sides = [side for side, gaps in gap_map.items() if len(gaps) >= 2 and _is_consecutive(gaps)]
    if len(opening_sides) != 1:
        return None

    open_side = opening_sides[0]
    opposite_side = OPPOSITE_SIDE[open_side]
    leak_index = None
    for side, gaps in gap_map.items():
        if side == open_side:
            if len(gaps) < 2 or not _is_consecutive(gaps):
                return None
        elif side == opposite_side:
            if len(gaps) == 1:
                leak_index = int(gaps[0])
            elif len(gaps) != 0:
                return None
        elif gaps:
            return None

    return {
        "bbox": (r0, c0, r1, c1),
        "color": int(color),
        "open_side": open_side,
        "leak_index": leak_index,
        "opening_start": int(gap_map[open_side][0]),
        "opening_len": int(len(gap_map[open_side])),
    }


def _detect_cups(grid: np.ndarray) -> list[dict]:
    components = find_components(
        grid,
        bg=BACKGROUND,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    candidate_colors = sorted({int(component["color"]) for component in components if int(component["area"]) >= 4})

    cups = []
    for color in candidate_colors:
        cup = _analyze_cup_mask(grid == color, color)
        if cup is not None:
            cups.append(cup)
    cups.sort(key=lambda cup: cup["bbox"])
    return cups


def _interior_slices(cup: dict) -> tuple[slice, slice]:
    r0, c0, r1, c1 = cup["bbox"]
    if cup["open_side"] == "top":
        return slice(r0, r1), slice(c0 + 1, c1)
    if cup["open_side"] == "bottom":
        return slice(r0 + 1, r1 + 1), slice(c0 + 1, c1)
    if cup["open_side"] == "left":
        return slice(r0 + 1, r1), slice(c0, c1)
    return slice(r0 + 1, r1), slice(c0 + 1, c1 + 1)


def _channel_cells(cup: dict) -> list[tuple[int, list[tuple[int, int]]]]:
    row_slice, col_slice = _interior_slices(cup)
    gravity = OPEN_TO_GRAVITY[cup["open_side"]]
    channels: list[tuple[int, list[tuple[int, int]]]] = []
    if gravity[0] != 0:
        for c in range(col_slice.start, col_slice.stop):
            cells = [(r, c) for r in range(row_slice.start, row_slice.stop)]
            cells.sort(key=lambda cell: _projection(cell, gravity), reverse=True)
            channels.append((c, cells))
    else:
        for r in range(row_slice.start, row_slice.stop):
            cells = [(r, c) for c in range(col_slice.start, col_slice.stop)]
            cells.sort(key=lambda cell: _projection(cell, gravity), reverse=True)
            channels.append((r, cells))
    return channels


def _leak_channel_key(cup: dict) -> int | None:
    if cup["leak_index"] is None:
        return None
    r0, c0, _r1, _c1 = cup["bbox"]
    if cup["open_side"] in ("top", "bottom"):
        return c0 + int(cup["leak_index"])
    return r0 + int(cup["leak_index"])


def _leak_hole_coord(cup: dict) -> tuple[int, int] | None:
    leak_index = cup["leak_index"]
    if leak_index is None:
        return None
    r0, c0, r1, c1 = cup["bbox"]
    if cup["open_side"] == "top":
        return r1, c0 + int(leak_index)
    if cup["open_side"] == "bottom":
        return r0, c0 + int(leak_index)
    if cup["open_side"] == "left":
        return r0 + int(leak_index), c1
    return r0 + int(leak_index), c0


def _clear_cup_interiors(grid: np.ndarray, cups: list[dict]) -> np.ndarray:
    out = grid.copy()
    for cup in cups:
        row_slice, col_slice = _interior_slices(cup)
        out[row_slice, col_slice] = BACKGROUND
    return out


def _collect_channel_colors(source: np.ndarray, cells: list[tuple[int, int]]) -> list[int]:
    return [int(source[r, c]) for r, c in cells if int(source[r, c]) != BACKGROUND]


def _leak_targets(out: np.ndarray, cup: dict, count: int) -> list[tuple[int, int]]:
    if count == 0:
        return []
    hole = _leak_hole_coord(cup)
    assert hole is not None
    dr, dc = OPEN_TO_GRAVITY[cup["open_side"]]

    run: list[tuple[int, int]] = []
    r, c = hole
    while True:
        r += dr
        c += dc
        if not _in_bounds(out.shape, r, c):
            break
        if int(out[r, c]) != BACKGROUND:
            break
        run.append((r, c))

    assert len(run) >= count, "leak path is shorter than the leaking stack"
    return list(reversed(run))[:count]


def _apply_cup_gravity(source: np.ndarray, out: np.ndarray, cup: dict) -> None:
    leak_channel = _leak_channel_key(cup)
    for channel_key, cells in _channel_cells(cup):
        colors = _collect_channel_colors(source, cells)
        if not colors:
            continue
        if leak_channel is not None and channel_key == leak_channel:
            targets = _leak_targets(out, cup, len(colors))
        else:
            targets = cells[: len(colors)]
        for color, (r, c) in zip(colors, targets):
            out[r, c] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Let every cup's interior pixels settle opposite its own opening, with leak-channel pixels draining further outside.

    Input structure:
    - A black background contains several disconnected monochrome cup walls, each a rectangular border with one wide opening on one side.
    - A black background contains several disconnected monochrome cup walls, each forming a rectangular cup with one wide open side.
    - Some cups also have a one-cell leak hole on the side opposite the opening, and colored marker pixels are scattered inside the cup interiors.
    - Additional colored distractor pixels may appear outside every cup and must remain fixed.

    Rule:
    - Detect each cup independently from its wide boundary gap, and use the direction opposite that gap as the cup's local gravity.
    - For every interior channel parallel to gravity, let the interior pixels fall and stack against the cup bottom while preserving their per-channel order.
    - If the bottom side has a one-cell leak hole, the pixels from that single channel do not remain inside: they pass through the hole and keep falling along the same direction until stopped by the canvas edge, another cup wall, or earlier leaked pixels.

    Key insight:
    - Gravity is local to each cup, and the wide opening only defines orientation while the separate one-cell hole selects exactly one draining channel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    cups = _detect_cups(g)
    assert cups, "expected at least one cup"

    out = _clear_cup_interiors(g, cups)
    for cup in cups:
        _apply_cup_gravity(g, out, cup)
    return out


def _draw_cup_walls(grid: np.ndarray, cup: dict) -> None:
    r0, c0, r1, c1 = cup["bbox"]
    color = int(cup["color"])
    if cup["open_side"] == "top":
        grid[r0 : r1 + 1, c0] = color
        grid[r0 : r1 + 1, c1] = color
        grid[r1, c0 : c1 + 1] = color
    elif cup["open_side"] == "bottom":
        grid[r0 : r1 + 1, c0] = color
        grid[r0 : r1 + 1, c1] = color
        grid[r0, c0 : c1 + 1] = color
    elif cup["open_side"] == "left":
        grid[r0, c0 : c1 + 1] = color
        grid[r1, c0 : c1 + 1] = color
        grid[r0 : r1 + 1, c1] = color
    else:
        grid[r0, c0 : c1 + 1] = color
        grid[r1, c0 : c1 + 1] = color
        grid[r0 : r1 + 1, c0] = color

    hole = _leak_hole_coord(cup)
    if hole is not None:
        grid[hole] = BACKGROUND


def _cup_wall_mask(shape: tuple[int, int], cup: dict) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    temp = np.zeros(shape, dtype=int)
    _draw_cup_walls(temp, cup)
    mask[temp != BACKGROUND] = True
    return mask


def _cup_void_mask(shape: tuple[int, int], cup: dict) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = cup["bbox"]
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    mask &= ~_cup_wall_mask(shape, cup)
    return mask


def _render_input(shape: tuple[int, int], cups: list[dict], distractors: list[tuple[int, int, int]]) -> np.ndarray:
    grid = np.full(shape, BACKGROUND, dtype=int)
    for cup in cups:
        _draw_cup_walls(grid, cup)
    for r, c, color in distractors:
        grid[r, c] = int(color)
    for cup in cups:
        for (r, c), color in cup["markers"].items():
            grid[r, c] = int(color)
    return grid


def _expected_from_specs(
    shape: tuple[int, int],
    cups: list[dict],
    distractors: list[tuple[int, int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    inp = _render_input(shape, cups, distractors)
    out = _clear_cup_interiors(inp, cups)
    for cup in cups:
        _apply_cup_gravity(inp, out, cup)
    return inp, out


def _sample_cup_geometry(height: int, width: int, taken: np.ndarray, color: int) -> dict | None:
    for _ in range(200):
        h = random.randint(5, min(9, height - 2))
        w = random.randint(5, min(9, width - 2))
        open_side = random.choice(SIDES)
        opening_len = (w - 2) if open_side in ("top", "bottom") else (h - 2)
        if opening_len < 2:
            continue

        r0 = random.randint(1, height - h - 1)
        c0 = random.randint(1, width - w - 1)
        r1 = r0 + h - 1
        c1 = c0 + w - 1
        er0 = max(0, r0 - 1)
        ec0 = max(0, c0 - 1)
        er1 = min(height - 1, r1 + 1)
        ec1 = min(width - 1, c1 + 1)
        if taken[er0 : er1 + 1, ec0 : ec1 + 1].any():
            continue

        cup = {
            "bbox": (r0, c0, r1, c1),
            "color": int(color),
            "open_side": open_side,
            "opening_start": 1,
            "opening_len": int(opening_len),
            "leak_index": None,
            "markers": {},
        }
        taken[er0 : er1 + 1, ec0 : ec1 + 1] = True
        return cup
    return None


def _compute_leak_runs(shape: tuple[int, int], cups: list[dict]) -> dict[int, list[tuple[int, int]]] | None:
    base = np.full(shape, BACKGROUND, dtype=int)
    void_mask = np.zeros(shape, dtype=bool)
    for cup in cups:
        _draw_cup_walls(base, cup)
    for cup in cups:
        void_mask |= _cup_void_mask(shape, cup)

    leak_runs: dict[int, list[tuple[int, int]]] = {}
    claimed: set[tuple[int, int]] = set()
    for idx, cup in enumerate(cups):
        if cup["leak_index"] is None:
            continue
        hole = _leak_hole_coord(cup)
        assert hole is not None
        dr, dc = OPEN_TO_GRAVITY[cup["open_side"]]
        run: list[tuple[int, int]] = []
        r, c = hole
        while True:
            r += dr
            c += dc
            if not _in_bounds(shape, r, c):
                break
            if void_mask[r, c]:
                return None
            if int(base[r, c]) != BACKGROUND:
                break
            run.append((r, c))
        if not run:
            return None
        if claimed.intersection(run):
            return None
        claimed.update(run)
        leak_runs[idx] = run
    return leak_runs


def _allocate_counts(
    capacities: dict[int, int],
    total: int,
    seed_counts: dict[int, int],
) -> dict[int, int] | None:
    counts = {key: 0 for key in capacities}
    for key, value in seed_counts.items():
        if value > capacities[key]:
            return None
        counts[key] = int(value)
    remaining = total - sum(seed_counts.values())
    while remaining > 0:
        choices = [key for key, cap in capacities.items() if counts[key] < cap]
        if not choices:
            return None
        counts[random.choice(choices)] += 1
        remaining -= 1
    return counts


def _sample_source_positions(cells: list[tuple[int, int]], count: int) -> list[tuple[int, int]]:
    preferred = list(cells[count:])
    if len(preferred) >= count:
        return random.sample(preferred, count)
    positions = preferred[:]
    remaining = count - len(positions)
    positions.extend(random.sample(cells[:count], remaining))
    return positions


def _assign_markers(cups: list[dict], leak_runs: dict[int, list[tuple[int, int]]]) -> bool:
    wall_colors = {int(cup["color"]) for cup in cups}
    marker_palette = [color for color in range(1, 10) if color not in wall_colors]
    for idx, cup in enumerate(cups):
        channels = _channel_cells(cup)
        capacities = {key: len(cells) for key, cells in channels}
        if not capacities:
            return False

        leak_key = _leak_channel_key(cup)
        total_capacity = sum(capacities.values())
        markers: dict[tuple[int, int], int] = {}
        if leak_key is not None:
            leak_run = leak_runs.get(idx)
            if leak_run is None:
                return False
            leak_cap = min(capacities[leak_key], len(leak_run), 3)
            if leak_cap < 1:
                return False
            alloc_capacities = dict(capacities)
            alloc_capacities[leak_key] = leak_cap
            leak_count = random.randint(1, leak_cap)
            other_keys = [key for key in alloc_capacities if key != leak_key]
            extra_required = 1 if other_keys else 0
            max_extra = min(4, sum(alloc_capacities.values()) - leak_count)
            if max_extra < extra_required:
                return False
            extra = random.randint(extra_required, max_extra)
            seed_counts = {leak_key: leak_count}
            if extra_required:
                seed_counts[random.choice(other_keys)] = 1
            counts = _allocate_counts(alloc_capacities, leak_count + extra, seed_counts)
        else:
            total = random.randint(2, min(6, total_capacity))
            counts = _allocate_counts(capacities, total, {random.choice(list(capacities)): 1})
        if counts is None:
            return False

        for channel_key, cells in channels:
            count = counts[channel_key]
            if count == 0:
                continue
            positions = _sample_source_positions(cells, count)
            for position in positions:
                markers[position] = int(random.choice(marker_palette))
        cup["markers"] = markers
    return True


def _place_distractors(
    shape: tuple[int, int],
    cups: list[dict],
    leak_runs: dict[int, list[tuple[int, int]]],
) -> list[tuple[int, int, int]]:
    forbidden = np.zeros(shape, dtype=bool)
    for cup in cups:
        r0, c0, r1, c1 = cup["bbox"]
        forbidden[max(0, r0 - 1) : min(shape[0], r1 + 2), max(0, c0 - 1) : min(shape[1], c1 + 2)] = True
    for run in leak_runs.values():
        for r, c in run:
            forbidden[r, c] = True

    cells = [(r, c) for r in range(shape[0]) for c in range(shape[1]) if not forbidden[r, c]]
    random.shuffle(cells)
    count = random.randint(0, min(5, len(cells)))
    distractors: list[tuple[int, int, int]] = []
    wall_colors = {int(cup["color"]) for cup in cups}
    distractor_palette = [color for color in range(1, 10) if color not in wall_colors]
    used = np.zeros(shape, dtype=bool)
    for r, c in cells:
        if len(distractors) >= count:
            break
        if used[r, c]:
            continue
        color = random.choice(distractor_palette)
        distractors.append((r, c, color))
        used[max(0, r - 1) : min(shape[0], r + 2), max(0, c - 1) : min(shape[1], c + 2)] = True
    return distractors


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(16, 30)
    width = random.randint(16, 30)
    cup_count = random.choices([1, 2, 3], weights=[2, 4, 3], k=1)[0]
    taken = np.zeros((height, width), dtype=bool)
    wall_colors = random.sample(range(1, 10), k=cup_count)

    cups: list[dict] = []
    for color in wall_colors:
        cup = _sample_cup_geometry(height, width, taken, color)
        if cup is None:
            return None
        cups.append(cup)

    leak_flags = [random.random() < 0.6 for _ in cups]
    if not any(leak_flags):
        leak_flags[random.randrange(len(cups))] = True
    for cup, leak_flag in zip(cups, leak_flags):
        if not leak_flag:
            continue
        r0, c0, r1, c1 = cup["bbox"]
        side_span = (c1 - c0 + 1) if cup["open_side"] in ("top", "bottom") else (r1 - r0 + 1)
        cup["leak_index"] = random.randint(1, side_span - 2)

    leak_runs = _compute_leak_runs((height, width), cups)
    if leak_runs is None:
        return None
    if not _assign_markers(cups, leak_runs):
        return None

    distractors = _place_distractors((height, width), cups, leak_runs)
    inp, expected = _expected_from_specs((height, width), cups, distractors)
    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid leak-cup scene")
