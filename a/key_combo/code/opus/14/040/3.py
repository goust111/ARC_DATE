from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
FRAME_COLORS = (1, 3, 7)
TERRAIN_COLOR = 5
GEN_ATTEMPTS = 400

_BASE_SHAPES = [
    ((0, 0), (0, 1)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (1, 0), (2, 0), (1, 1)),
    ((0, 1), (0, 0), (1, 1), (2, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (1, 1), (2, 1), (2, 0)),
]


def _normalize(cells):
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _shape_variants(shape):
    cells = list(shape)
    variants = set()
    for rot in range(4):
        turned = []
        for r, c in cells:
            rr, cc = r, c
            for _ in range(rot):
                rr, cc = cc, -rr
            turned.append((rr, cc))
        for flip in (False, True):
            candidate = [((r, -c) if flip else (r, c)) for r, c in turned]
            variants.add(_normalize(candidate))
    return list(variants)


_SHAPE_VARIANTS = [_shape_variants(shape) for shape in _BASE_SHAPES]


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_frame(grid: np.ndarray, coords: np.ndarray, color: int) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if len(coords) != 2 * h + 2 * w - 4:
        return False
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            border = r in (r0, r1) or c in (c0, c1)
            value = int(grid[r, c])
            if border:
                if value != int(color):
                    return False
            elif value != BG:
                return False
    return True


def _terrain_top(grid: np.ndarray) -> int:
    rows = np.flatnonzero(np.any(grid == TERRAIN_COLOR, axis=1))
    assert rows.size > 0, "expected a grey terrain region"
    return int(rows[0])


def _terrain_starts(grid: np.ndarray, terrain_top: int) -> np.ndarray:
    h, w = grid.shape
    starts = np.empty(w, dtype=int)
    for c in range(w):
        col = grid[terrain_top:, c]
        grey_rows = np.flatnonzero(col == TERRAIN_COLOR)
        assert grey_rows.size > 0, "every terrain column must contain grey"
        start = terrain_top + int(grey_rows[0])
        assert np.all(grid[start:, c] == TERRAIN_COLOR), "terrain must be a contiguous suffix"
        assert np.all(grid[terrain_top:start, c] == BG), "terrain band above the suffix must be black"
        starts[c] = start
    assert len(set(map(int, starts - terrain_top))) == w, "terrain depths must be unique"
    assert np.all(grid[h - 1] == TERRAIN_COLOR), "bottom row must be all terrain"
    return starts


def _parse_scene(grid: np.ndarray):
    terrain_top = _terrain_top(grid)
    starts = _terrain_starts(grid, terrain_top)
    frames = []
    sprites = []
    for color in FRAME_COLORS:
        for comp in _components_mask(grid == color):
            if _is_frame(grid, comp, color):
                frames.append({"color": int(color), "coords": comp})
            else:
                sprites.append({"color": int(color), "coords": comp})

    frame_colors = sorted({frame["color"] for frame in frames})
    assert 2 <= len(frame_colors) <= 3, "expected 2-3 distinct frame colors"
    counts = {color: 0 for color in frame_colors}
    for sprite in sprites:
        color = int(sprite["color"])
        if color in counts:
            counts[color] += 1
    assert all(counts[color] > 0 for color in frame_colors), "every frame color must match sprites"
    return {
        "terrain_top": terrain_top,
        "starts": starts,
        "depths": starts - terrain_top,
        "frames": frames,
        "frame_colors": frame_colors,
        "sprites": sprites,
        "counts": counts,
    }


def _assign_columns(depths: np.ndarray, counts: dict[int, int]) -> dict[int, int]:
    order = sorted(range(len(depths)), key=lambda c: (-int(depths[c]), int(c)))
    owners: dict[int, int] = {}
    for color, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        claimed = 0
        for col in order:
            if col in owners:
                continue
            owners[col] = int(color)
            claimed += 1
            if claimed == int(count):
                break
        assert claimed == int(count), "not enough distinct terrain columns"
    return owners


def _paint_columns(out: np.ndarray, starts: np.ndarray, owners: dict[int, int]) -> None:
    for col, color in owners.items():
        segment = out[: int(starts[col]), int(col)]
        segment[segment == BG] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count same-color sprites for each hollow frame and use those tallies to recolor terrain-above columns.

    Input structure:
    - Black background dominates the grid, with 2-3 hollow rectangular frames in blue/green/orange.
    - Small monochrome 4-connected sprites of those colors are scattered above the frames.
    - The bottom band is a grey terrain profile: each column has black cells followed by a contiguous grey suffix.

    Rule:
    - For each frame color C, count how many top sprites have color C.
    - Measure each terrain column by the number of black cells above its grey suffix within the terrain band.
    - Resolve the frame colors by larger counts first: each color claims that many deepest still-unclaimed columns.
    - Erase all top sprites, keep the frames and grey terrain, and recolor every black cell above each claimed terrain column with the owning frame color.

    Key insight:
    - Sprite counts do not draw objects directly; they determine how many terrain columns each frame color controls, and terrain depth decides where those colors appear.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    scene = _parse_scene(g)

    out = g.copy()
    for sprite in scene["sprites"]:
        coords = sprite["coords"]
        out[coords[:, 0], coords[:, 1]] = BG

    owners = _assign_columns(scene["depths"], scene["counts"])
    _paint_columns(out, scene["starts"], owners)
    return out


def _random_depths(width: int, terrain_rows: int) -> np.ndarray:
    assert width <= terrain_rows
    walk = [random.uniform(-1.0, 1.0)]
    for _ in range(1, width):
        walk.append(walk[-1] + random.uniform(-1.7, 1.7))
    order = sorted(range(width), key=lambda i: (walk[i], random.random()))
    depths = np.empty(width, dtype=int)
    for rank, idx in enumerate(order):
        depths[idx] = rank
    if list(depths) == list(range(width)) or list(depths) == list(range(width - 1, -1, -1)):
        return _random_depths(width, terrain_rows)
    return depths


def _count_choices(width: int, n_colors: int) -> list[tuple[int, ...]]:
    limit = width // 2
    nums = range(1, limit + 1)
    choices = []
    if n_colors == 2:
        for a in nums:
            for b in nums:
                if a == b:
                    continue
                combo = tuple(sorted((a, b)))
                if sum(combo) <= width - 1:
                    choices.append(combo)
    else:
        for a in nums:
            for b in nums:
                for c in nums:
                    if len({a, b, c}) != 3:
                        continue
                    combo = tuple(sorted((a, b, c)))
                    if sum(combo) <= width - 1:
                        choices.append(combo)
    return sorted(set(choices))


def _frame_layout(width: int, n_frames: int):
    base_widths = [3] * n_frames
    inner_gaps = [1] * (n_frames - 1)
    remaining = width - sum(base_widths) - sum(inner_gaps)
    if remaining < 0:
        return None

    widths = base_widths[:]
    left_margin = 0
    right_margin = 0
    while remaining > 0:
        choices = ["left", "right"] + [f"w{i}" for i in range(n_frames) if widths[i] < 5]
        pick = random.choice(choices)
        if pick == "left":
            left_margin += 1
        elif pick == "right":
            right_margin += 1
        else:
            widths[int(pick[1:])] += 1
        remaining -= 1

    positions = []
    col = left_margin
    for i, frame_w in enumerate(widths):
        positions.append((col, frame_w))
        col += frame_w
        if i + 1 < n_frames:
            col += inner_gaps[i]
    assert col + right_margin == width
    random.shuffle(positions)
    return positions


def _shape_cells(shape, r0: int, c0: int):
    return [(r0 + r, c0 + c) for r, c in shape]


def _can_place_shape(occupied: np.ndarray, shape, r0: int, c0: int) -> bool:
    h, w = occupied.shape
    cells = _shape_cells(shape, r0, c0)
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w and occupied[rr, cc]:
                    return False
    return True


def _place_sprites(top_h: int, width: int, sprite_plan: list[int]) -> list[dict] | None:
    occupied = np.zeros((top_h, width), dtype=bool)
    placed = []
    colors = sprite_plan[:]
    random.shuffle(colors)
    for color in colors:
        candidates = []
        for variants in _SHAPE_VARIANTS:
            shape = random.choice(variants)
            shape_h = max(r for r, _ in shape) + 1
            shape_w = max(c for _, c in shape) + 1
            for r0 in range(top_h - shape_h + 1):
                for c0 in range(width - shape_w + 1):
                    if _can_place_shape(occupied, shape, r0, c0):
                        candidates.append((shape, r0, c0))
        if not candidates:
            return None
        shape, r0, c0 = random.choice(candidates)
        cells = _shape_cells(shape, r0, c0)
        for r, c in cells:
            occupied[r, c] = True
        placed.append({"color": int(color), "cells": cells})
    return placed


def _pixel_counts(sprites: list[dict], colors: list[int]) -> dict[int, int]:
    counts = {color: 0 for color in colors}
    for sprite in sprites:
        color = int(sprite["color"])
        if color in counts:
            counts[color] += len(sprite["cells"])
    return counts


def _build_expected(inp: np.ndarray, counts: dict[int, int]) -> np.ndarray:
    scene = _parse_scene(inp)
    assert scene["counts"] == counts
    out = inp.copy()
    for sprite in scene["sprites"]:
        coords = sprite["coords"]
        out[coords[:, 0], coords[:, 1]] = BG
    owners = _assign_columns(scene["depths"], scene["counts"])
    _paint_columns(out, scene["starts"], owners)
    return out


def _validate_sample(inp: np.ndarray, expected: np.ndarray, counts: dict[int, int], frame_colors: list[int]) -> bool:
    scene = _parse_scene(inp)
    if scene["counts"] != counts:
        return False
    if scene["frame_colors"] != sorted(frame_colors):
        return False
    if any(scene["counts"][color] <= 0 for color in frame_colors):
        return False
    if len(set(scene["counts"].values())) != len(scene["counts"]):
        return False

    owners = _assign_columns(scene["depths"], scene["counts"])
    owned_counts = Counter(owners.values())
    if any(owned_counts[color] != counts[color] for color in frame_colors):
        return False

    if list(scene["depths"]) == list(range(inp.shape[1])) or list(scene["depths"]) == list(range(inp.shape[1] - 1, -1, -1)):
        return False

    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False

    visible = Counter()
    for col, color in owners.items():
        visible[int(color)] += int(np.count_nonzero(actual[: int(scene["starts"][col]), int(col)] == int(color)))
    if any(visible[color] == 0 for color in frame_colors):
        return False

    pixel_counts = _pixel_counts(
        [{"color": sprite["color"], "cells": [(int(r), int(c)) for r, c in sprite["coords"]]} for sprite in scene["sprites"]],
        frame_colors,
    )
    if not any(pixel_counts[color] != counts[color] for color in frame_colors):
        return False

    largest_color = max(counts, key=lambda color: counts[color])
    largest_only = expected.copy()
    largest_only[:] = inp
    largest_only = inp.copy()
    for sprite in scene["sprites"]:
        coords = sprite["coords"]
        largest_only[coords[:, 0], coords[:, 1]] = BG
    largest_depths = sorted(range(inp.shape[1]), key=lambda c: (-int(scene["depths"][c]), int(c)))[: counts[largest_color]]
    for col in largest_depths:
        segment = largest_only[: int(scene["starts"][col]), int(col)]
        segment[segment == BG] = int(largest_color)
    if np.array_equal(largest_only, expected):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    width = random.randint(10, 13)
    n_frame_colors = random.choice([2, 3] if width >= 11 else [2])
    top_h = random.randint(4, 6)
    gap1 = random.randint(1, 2)
    frame_h = random.randint(4, 5)
    gap2 = random.randint(1, 2)
    terrain_h = random.randint(max(7, width), min(13, width + 2))
    total_h = top_h + gap1 + frame_h + gap2 + terrain_h
    if total_h > 30:
        return None

    frame_colors = random.sample(list(FRAME_COLORS), n_frame_colors)
    counts_combo = random.choice(_count_choices(width, n_frame_colors))
    shuffled_counts = list(counts_combo)
    random.shuffle(shuffled_counts)
    counts = {color: count for color, count in zip(frame_colors, shuffled_counts)}

    leftover = [color for color in FRAME_COLORS if color not in frame_colors]
    distractor_colors = []
    if leftover and random.random() < 0.7:
        distractor_colors = [random.choice(leftover)] * random.randint(1, 2)

    frame_slots = _frame_layout(width, n_frame_colors)
    if frame_slots is None:
        return None

    sprite_plan = []
    for color in frame_colors:
        sprite_plan.extend([color] * counts[color])
    sprite_plan.extend(distractor_colors)
    sprites = _place_sprites(top_h, width, sprite_plan)
    if sprites is None:
        return None

    grid = np.zeros((total_h, width), dtype=int)

    for sprite in sprites:
        for r, c in sprite["cells"]:
            grid[r, c] = int(sprite["color"])

    frame_row0 = top_h + gap1
    used_slots = frame_slots[:]
    random.shuffle(used_slots)
    for color, (c0, frame_w) in zip(frame_colors, used_slots):
        frame_inner_h = random.randint(3, min(4, frame_h))
        r0 = frame_row0 + random.randint(0, frame_h - frame_inner_h)
        r1 = r0 + frame_inner_h - 1
        c1 = c0 + frame_w - 1
        grid[r0, c0 : c1 + 1] = int(color)
        grid[r1, c0 : c1 + 1] = int(color)
        grid[r0 : r1 + 1, c0] = int(color)
        grid[r0 : r1 + 1, c1] = int(color)

    terrain_top = top_h + gap1 + frame_h + gap2
    depths = _random_depths(width, terrain_h)
    starts = terrain_top + depths
    for c, start in enumerate(starts):
        grid[int(start) :, c] = TERRAIN_COLOR

    expected = _build_expected(grid, counts)
    if not _validate_sample(grid, expected, counts, frame_colors):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
