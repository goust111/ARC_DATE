from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
WALL = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

MIN_GRID_SIZE = 15
MAX_GRID_SIZE = 24
MIN_CHANNEL_LENGTH = 5
MAX_CHANNELS = 4
MAX_GENERATION_ATTEMPTS = 400

_LAYOUT_CACHE: dict[tuple[int, int], dict[tuple[int, int, int], list[tuple[int, ...]]]] = {}


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.asarray(coords, dtype=int))

    return components


def _growth_offsets(length: int, seed_offset: int, seed_count: int) -> list[int]:
    left_dist = seed_offset
    right_dist = length - 1 - seed_offset
    direction = 1 if left_dist <= right_dist else -1

    offsets: list[int] = []
    for step in range(1, seed_count + 1):
        nxt = seed_offset + direction * step
        if 0 <= nxt < length:
            offsets.append(nxt)
        else:
            break
    return offsets


def _parse_channels(grid: np.ndarray) -> list[dict]:
    comps = _components4(grid != BG)
    channels: list[dict] = []

    for coords in comps:
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        box = grid[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(box == BG):
            raise AssertionError("channel components must fill their bounding boxes")

        h, w = box.shape
        if h == 3 and w > 3:
            axis_r = r0 + 1
            if not np.all(grid[r0, c0 : c1 + 1] == WALL):
                raise AssertionError("horizontal channel top wall must be grey")
            if not np.all(grid[r1, c0 : c1 + 1] == WALL):
                raise AssertionError("horizontal channel bottom wall must be grey")

            seeds = []
            for c in range(c0, c1 + 1):
                color = int(grid[axis_r, c])
                if color == BG:
                    raise AssertionError("channel axis cannot contain background")
                if color != WALL:
                    seeds.append({"row": axis_r, "col": c, "offset": c - c0, "color": color})
            if not seeds:
                raise AssertionError("each channel must contain at least one seed")
            channels.append(
                {
                    "orientation": "h",
                    "bbox": (r0, c0, r1, c1),
                    "length": w,
                    "axis_row": axis_r,
                    "seeds": seeds,
                }
            )
        elif w == 3 and h > 3:
            axis_c = c0 + 1
            if not np.all(grid[r0 : r1 + 1, c0] == WALL):
                raise AssertionError("vertical channel left wall must be grey")
            if not np.all(grid[r0 : r1 + 1, c1] == WALL):
                raise AssertionError("vertical channel right wall must be grey")

            seeds = []
            for r in range(r0, r1 + 1):
                color = int(grid[r, axis_c])
                if color == BG:
                    raise AssertionError("channel axis cannot contain background")
                if color != WALL:
                    seeds.append({"row": r, "col": axis_c, "offset": r - r0, "color": color})
            if not seeds:
                raise AssertionError("each channel must contain at least one seed")
            channels.append(
                {
                    "orientation": "v",
                    "bbox": (r0, c0, r1, c1),
                    "length": h,
                    "axis_col": axis_c,
                    "seeds": seeds,
                }
            )
        else:
            raise AssertionError("each non-background component must be a width-3 channel")

    if not channels:
        raise AssertionError("expected at least one channel")
    return channels


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each colored seed along the centerline of its grey channel.

    For every width-3 horizontal or vertical grey strip, count the seeds on its
    centerline. Each seed grows away from the nearer end by that count, or
    toward the positive axis direction when exactly centered. Original seeds stay
    unchanged; only grey centerline cells are recolored.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("grid must be 2D")

    channels = _parse_channels(g)
    original_seeds = {
        (int(seed["row"]), int(seed["col"])): int(seed["color"])
        for channel in channels
        for seed in channel["seeds"]
    }

    out = g.copy()
    planned: dict[tuple[int, int], int] = {}

    for channel in channels:
        seed_count = len(channel["seeds"])
        length = int(channel["length"])
        orient = channel["orientation"]

        for seed in channel["seeds"]:
            color = int(seed["color"])
            for target_offset in _growth_offsets(length, int(seed["offset"]), seed_count):
                if orient == "h":
                    pos = (int(channel["axis_row"]), int(channel["bbox"][1]) + target_offset)
                else:
                    pos = (int(channel["bbox"][0]) + target_offset, int(channel["axis_col"]))

                if pos in original_seeds:
                    continue
                prev = planned.get(pos)
                if prev is not None and prev != color:
                    raise AssertionError("ambiguous overlapping growth")
                planned[pos] = color

    for (r, c), color in planned.items():
        out[r, c] = color
    return out


def _positions_valid(length: int, seed_count: int, positions: tuple[int, ...]) -> bool:
    if len(positions) != seed_count or len(set(positions)) != seed_count:
        return False

    seeds = set(positions)
    painted: set[int] = set()
    for pos in positions:
        for nxt in _growth_offsets(length, pos, seed_count):
            if nxt in seeds or nxt in painted:
                return False
            painted.add(nxt)
    return True


def _family_layouts(length: int, seed_count: int) -> dict[tuple[int, int, int], list[tuple[int, ...]]]:
    key = (length, seed_count)
    if key in _LAYOUT_CACHE:
        return _LAYOUT_CACHE[key]

    mid = (length - 1) // 2
    families: dict[tuple[int, int, int], list[tuple[int, ...]]] = {}

    for tie in (0, 1):
        if tie and length % 2 == 0:
            continue
        for left_count in range(seed_count - tie + 1):
            right_count = seed_count - tie - left_count
            family = (left_count, right_count, tie)

            center = (mid,) if tie else ()
            left_candidates = list(range(0, mid + (0 if tie else 1)))
            if tie and left_candidates and left_candidates[-1] == mid:
                left_candidates = left_candidates[:-1]
            right_candidates = list(range(mid + 1, length))

            left_combos = itertools.combinations(left_candidates, left_count)
            right_combos = itertools.combinations(right_candidates, right_count)
            layouts: list[tuple[int, ...]] = []

            for left_positions in left_combos:
                for right_positions in right_combos:
                    positions = tuple(sorted(left_positions + center + right_positions))
                    if _positions_valid(length, seed_count, positions):
                        layouts.append(positions)

            if layouts:
                families[family] = layouts

    _LAYOUT_CACHE[key] = families
    return families


def _channel_choices(
    grid_shape: tuple[int, int], orientation: str, seed_count: int
) -> list[tuple[int, tuple[int, int, int], list[tuple[int, ...]]]]:
    h, w = grid_shape
    max_length = (w - 2) if orientation == "h" else (h - 2)
    choices: list[tuple[int, tuple[int, int, int], list[tuple[int, ...]]]] = []

    for length in range(MIN_CHANNEL_LENGTH, max_length + 1):
        families = _family_layouts(length, seed_count)
        for family, layouts in families.items():
            choices.append((length, family, layouts))
    return choices


def _channel_positions(grid_shape: tuple[int, int], orientation: str, length: int, blocked: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid_shape
    ch_h, ch_w = (3, length) if orientation == "h" else (length, 3)
    positions: list[tuple[int, int]] = []

    for r0 in range(1, h - ch_h):
        for c0 in range(1, w - ch_w):
            if blocked[r0 : r0 + ch_h, c0 : c0 + ch_w].any():
                continue
            positions.append((r0, c0))

    return positions


def _paint_channel(grid: np.ndarray, blocked: np.ndarray, orientation: str, r0: int, c0: int, length: int, seed_offsets: tuple[int, ...]) -> None:
    ch_h, ch_w = (3, length) if orientation == "h" else (length, 3)
    grid[r0 : r0 + ch_h, c0 : c0 + ch_w] = WALL

    colors = random.sample(SEED_COLORS, k=len(seed_offsets))
    if orientation == "h":
        axis_r = r0 + 1
        for offset, color in zip(seed_offsets, colors):
            grid[axis_r, c0 + offset] = color
    else:
        axis_c = c0 + 1
        for offset, color in zip(seed_offsets, colors):
            grid[r0 + offset, axis_c] = color

    r1 = r0 + ch_h - 1
    c1 = c0 + ch_w - 1
    blocked[max(0, r0 - 1) : min(grid.shape[0], r1 + 2), max(0, c0 - 1) : min(grid.shape[1], c1 + 2)] = True


def _orientation_plan(channel_count: int) -> list[str]:
    if channel_count == 1:
        return [random.choice(("h", "v"))]

    if random.random() < 0.65:
        plan = ["h", "v"]
        while len(plan) < channel_count:
            plan.append(random.choice(("h", "v")))
        random.shuffle(plan)
        return plan

    return [random.choice(("h", "v")) for _ in range(channel_count)]


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)
    w = random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)
    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    max_channels = 3 if min(h, w) < 18 else MAX_CHANNELS
    channel_count = random.randint(1, max_channels)
    orientations = _orientation_plan(channel_count)
    seed_counts = [random.randint(1, 3) for _ in range(channel_count)]

    for orientation, seed_count in zip(orientations, seed_counts):
        choices = _channel_choices((h, w), orientation, seed_count)
        if not choices:
            return None

        random.shuffle(choices)
        placed = False
        for length, _family, layouts in choices:
            starts = _channel_positions((h, w), orientation, length, blocked)
            if not starts:
                continue
            r0, c0 = random.choice(starts)
            layout = random.choice(layouts)
            _paint_channel(grid, blocked, orientation, r0, c0, length, layout)
            placed = True
            break
        if not placed:
            return None

    try:
        _ = main(grid)
        _parse_channels(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
