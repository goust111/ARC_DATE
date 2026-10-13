from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BACKGROUND = 0
NOISE = 7
FRAME_COLORS = (1, 3, 4, 6)

GRID_MIN = 15
GRID_MAX = 20
FRAME_COUNT_OPTIONS = (3, 4)
FRAME_SIZE_MIN = 4
FRAME_SIZE_MAX = 12
JITTER_OPTIONS = (-2, -1, 0, 1, 2)
GENERATION_ATTEMPTS = 600


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _component_coords(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(comp)
    return components


def _bbox_for_color(grid: np.ndarray, color: int) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(grid == color)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_size(grid: np.ndarray, color: int) -> int:
    box = _bbox_for_color(grid, color)
    if box is None:
        return 0
    r0, c0, r1, c1 = box
    return max(r1 - r0 + 1, c1 - c0 + 1)


def _draw_square_border(grid: np.ndarray, top: int, left: int, size: int, color: int) -> None:
    bottom = top + size - 1
    right = left + size - 1
    grid[top, left : right + 1] = color
    grid[bottom, left : right + 1] = color
    grid[top : bottom + 1, left] = color
    grid[top : bottom + 1, right] = color


def _square_border_cells(top: int, left: int, size: int) -> set[tuple[int, int]]:
    bottom = top + size - 1
    right = left + size - 1
    cells: set[tuple[int, int]] = set()
    for c in range(left, right + 1):
        cells.add((top, c))
        cells.add((bottom, c))
    for r in range(top, bottom + 1):
        cells.add((r, left))
        cells.add((r, right))
    return cells


def _noise_decision(grid: np.ndarray, r: int, c: int) -> int:
    support: list[tuple[int, int, int]] = []
    for nr, nc in _neighbors4(r, c):
        if not _in_bounds(grid.shape, nr, nc):
            continue
        color = int(grid[nr, nc])
        if color not in (BACKGROUND, NOISE):
            support.append((color, nr, nc))

    if len(support) < 2:
        return BACKGROUND

    counts = Counter(color for color, _nr, _nc in support)
    best_count = max(counts.values())
    tied = [color for color, count in counts.items() if count == best_count]
    if len(tied) == 1:
        return tied[0]

    tie_breakers = []
    for color in tied:
        rows = [nr for col, nr, _nc in support if col == color]
        cols = [nc for col, nr, nc in support if col == color and nr == min(rows)]
        tie_breakers.append((min(rows), min(cols), color))
    tie_breakers.sort()
    return tie_breakers[0][2]


def _apply_noise_rule(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    orange_components = _component_coords(g == NOISE)
    decisions: dict[tuple[int, int], int] = {}
    for comp in orange_components:
        assert len(comp) <= 2, "orange noise components must have area <= 2"
        for r, c in comp:
            decisions[(r, c)] = _noise_decision(g, r, c)
    for (r, c), color in decisions.items():
        out[r, c] = color
    return out


def _build_output(layers: list[tuple[int, int]]) -> np.ndarray:
    sizes = [size for size, _color in layers]
    n = max(sizes)
    out = np.full((n, n), BACKGROUND, dtype=int)

    for size, color in sorted(layers, reverse=True):
        assert (n - size) % 2 == 0, "all layers must share the output center"
        inset = (n - size) // 2
        _draw_square_border(out, inset, inset, size, color)

    inner_size, inner_color = min(layers)
    if inner_size > 2:
        inset = (n - inner_size) // 2
        seed = (inset + 1, inset + 1)
        if out[seed] == BACKGROUND:
            queue = deque([seed])
            seen = {seed}
            while queue:
                r, c = queue.popleft()
                if out[r, c] != BACKGROUND:
                    continue
                out[r, c] = inner_color
                for nr, nc in _neighbors4(r, c):
                    if _in_bounds(out.shape, nr, nc) and (nr, nc) not in seen and out[nr, nc] == BACKGROUND:
                        seen.add((nr, nc))
                        queue.append((nr, nc))
    return out


def _repaired_layers(grid: np.ndarray) -> list[tuple[int, int]]:
    repaired = _apply_noise_rule(grid)
    colors = [color for color in np.unique(repaired) if color not in (BACKGROUND, NOISE)]
    layers = []
    for color in sorted(map(int, colors)):
        box = _bbox_for_color(repaired, color)
        assert box is not None
        r0, c0, r1, c1 = box
        layers.append((max(r1 - r0 + 1, c1 - c0 + 1), color))
    return layers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassign orange noise on overlapping square-frame fragments, then redraw the repaired layers as centered concentric rings and fill the innermost one.

    Input structure:
    - The grid has a black background, 3-4 colored square-frame fragments, and orange noise pixels.
    - Frame colors are distinct and may overlap so that only one color is visible at each shared cell.
    - Orange pixels form only 4-connected components of area 1 or 2, placed on frame cells or just beside them.

    Rule:
    - For each orange pixel, count 4-neighbor frame-color contacts in the original input. If at least two frame-color neighbors exist, recolor the orange pixel to the most-supported frame color; ties go to the color whose supporting neighbor reaches the smallest row first. Otherwise erase the orange pixel to black.
    - After this repair, compute each color's square-frame side length from its monochrome bbox side.
    - Create a square output whose side is the largest repaired length, redraw all frames with a shared center from largest to smallest so smaller frames sit on top, then flood-fill the strict interior of the smallest frame with its own color.

    Key insight:
    - The orange noise is not just clutter: its ownership determines repaired frame sizes, and those sizes determine both the overlap order and which frame becomes the filled innermost layer.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    layers = _repaired_layers(g)
    assert layers, "expected at least one repaired frame"
    sizes = [size for size, _color in layers]
    assert len(set(sizes)) == len(sizes), "repaired frame sizes must be unique"
    return _build_output(layers)


def _sample_frames(rng: random.Random) -> tuple[int, int, list[dict[str, object]]] | None:
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    count = rng.choice(FRAME_COUNT_OPTIONS)
    parity = rng.choice((0, 1))
    max_size_cap = min(FRAME_SIZE_MAX, min(h, w) - 3)
    size_pool = [size for size in range(FRAME_SIZE_MIN, max_size_cap + 1) if size % 2 == parity]
    if len(size_pool) < count:
        return None

    sizes = sorted(rng.sample(size_pool, count), reverse=True)
    max_size = sizes[0]
    margin_r = h - max_size
    margin_c = w - max_size
    if margin_r < 1 or margin_c < 1:
        return None

    base_top = rng.randint(0, margin_r)
    base_left = rng.randint(0, margin_c)
    colors = rng.sample(list(FRAME_COLORS), count)

    frames: list[dict[str, object]] = []
    offsets_seen: set[tuple[int, int]] = set()
    for size, color in zip(sizes, colors):
        ideal_top = base_top + (max_size - size) // 2
        ideal_left = base_left + (max_size - size) // 2

        valid_offsets = []
        for dr in JITTER_OPTIONS:
            for dc in JITTER_OPTIONS:
                top = ideal_top + dr
                left = ideal_left + dc
                if 0 <= top and 0 <= left and top + size <= h and left + size <= w:
                    valid_offsets.append((dr, dc, top, left))
        if not valid_offsets:
            return None
        dr, dc, top, left = rng.choice(valid_offsets)
        offsets_seen.add((dr, dc))
        frames.append(
            {
                "color": color,
                "size": size,
                "top": top,
                "left": left,
                "cells": _square_border_cells(top, left, size),
            }
        )

    if len(offsets_seen) == 1:
        return None

    overlap_pixels = 0
    for r in range(h):
        for c in range(w):
            cover = sum((r, c) in frame["cells"] for frame in frames)
            if cover >= 2:
                overlap_pixels += 1
    if overlap_pixels == 0:
        return None

    return h, w, frames


def _render_visible_grid(h: int, w: int, frames: list[dict[str, object]], rng: random.Random) -> tuple[np.ndarray, list[int]]:
    order = list(range(len(frames)))
    rng.shuffle(order)
    grid = np.full((h, w), BACKGROUND, dtype=int)
    for idx in order:
        frame = frames[idx]
        _draw_square_border(grid, int(frame["top"]), int(frame["left"]), int(frame["size"]), int(frame["color"]))
    return grid, order


def _sizes_by_color(grid: np.ndarray, colors: list[int]) -> dict[int, int]:
    return {color: _bbox_size(grid, color) for color in colors}


def _intended_sizes(frames: list[dict[str, object]]) -> dict[int, int]:
    return {int(frame["color"]): int(frame["size"]) for frame in frames}


def _matches_intended_sizes(grid: np.ndarray, frames: list[dict[str, object]]) -> bool:
    colors = [int(frame["color"]) for frame in frames]
    return _sizes_by_color(grid, colors) == _intended_sizes(frames)


def _all_frame_neighborhood_cells(frames: list[dict[str, object]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for frame in frames:
        for r, c in frame["cells"]:
            if _in_bounds(shape, r, c):
                cells.add((r, c))
                for nr, nc in _neighbors4(r, c):
                    if _in_bounds(shape, nr, nc):
                        cells.add((nr, nc))
    return cells


def _apply_noise_cells(base: np.ndarray, cells: set[tuple[int, int]]) -> np.ndarray:
    grid = base.copy()
    for r, c in cells:
        grid[r, c] = NOISE
    return grid


def _touches_existing_noise(cells: set[tuple[int, int]], chosen: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        if (r, c) in chosen:
            return True
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in chosen and (nr, nc) not in cells:
                return True
    return False


def _find_critical_noise_sets(
    base: np.ndarray,
    frames: list[dict[str, object]],
) -> list[set[tuple[int, int]]]:
    valid_sets: list[set[tuple[int, int]]] = []
    expected_sizes = _intended_sizes(frames)
    colors = [int(frame["color"]) for frame in frames]
    candidate_cells: list[tuple[int, int, int]] = []

    for color in colors:
        box = _bbox_for_color(base, color)
        if box is None:
            continue
        r0, c0, r1, c1 = box
        coords = np.argwhere(base == color)
        row_counts = Counter()
        col_counts = Counter()
        for r, c in coords:
            rr = int(r)
            cc = int(c)
            if rr in (r0, r1):
                row_counts[rr] += 1
            if cc in (c0, c1):
                col_counts[cc] += 1
        for r, c in coords:
            rr = int(r)
            cc = int(c)
            if rr not in (r0, r1) and cc not in (c0, c1):
                continue
            scarce = False
            if rr in (r0, r1) and row_counts[rr] <= 2:
                scarce = True
            if cc in (c0, c1) and col_counts[cc] <= 2:
                scarce = True
            if not scarce:
                continue
            candidate_cells.append((rr, cc, color))
            trial = _apply_noise_cells(base, {(rr, cc)})
            repaired = _apply_noise_rule(trial)
            raw_sizes = _sizes_by_color(trial, colors)
            if int(repaired[rr, cc]) == color and raw_sizes != expected_sizes and _sizes_by_color(repaired, colors) == expected_sizes:
                valid_sets.append({(rr, cc)})

    if valid_sets:
        return valid_sets

    for i in range(len(candidate_cells)):
        r0, c0, _color0 = candidate_cells[i]
        for j in range(i + 1, len(candidate_cells)):
            r1, c1, _color1 = candidate_cells[j]
            cells = {(r0, c0), (r1, c1)}
            trial = _apply_noise_cells(base, cells)
            repaired = _apply_noise_rule(trial)
            raw_sizes = _sizes_by_color(trial, colors)
            if raw_sizes != expected_sizes and _sizes_by_color(repaired, colors) == expected_sizes:
                valid_sets.append(cells)
    return valid_sets


def _find_safe_singletons(
    base: np.ndarray,
    frames: list[dict[str, object]],
    chosen_noise: set[tuple[int, int]],
    target_outcomes: set[int],
) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    candidate_cells = _all_frame_neighborhood_cells(frames, base.shape)
    expected_sizes = _intended_sizes(frames)
    colors = [int(frame["color"]) for frame in frames]

    for r, c in candidate_cells:
        if _touches_existing_noise({(r, c)}, chosen_noise):
            continue
        trial = _apply_noise_cells(base, chosen_noise | {(r, c)})
        repaired = _apply_noise_rule(trial)
        if _sizes_by_color(repaired, colors) != expected_sizes:
            continue
        if int(repaired[r, c]) in target_outcomes:
            candidates.append((r, c))
    return candidates


def _find_safe_pairs(
    base: np.ndarray,
    frames: list[dict[str, object]],
    chosen_noise: set[tuple[int, int]],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    candidate_cells = sorted(_all_frame_neighborhood_cells(frames, base.shape))
    expected_sizes = _intended_sizes(frames)
    colors = [int(frame["color"]) for frame in frames]
    candidate_set = set(candidate_cells)

    for r, c in candidate_cells:
        for nr, nc in ((r + 1, c), (r, c + 1)):
            pair = {(r, c), (nr, nc)}
            if (nr, nc) not in candidate_set:
                continue
            if _touches_existing_noise(pair, chosen_noise):
                continue
            trial = _apply_noise_cells(base, chosen_noise | pair)
            repaired = _apply_noise_rule(trial)
            if _sizes_by_color(repaired, colors) != expected_sizes:
                continue
            outcomes = {int(repaired[rr, cc]) for rr, cc in pair}
            if BACKGROUND not in outcomes:
                continue
            pairs.append(((r, c), (nr, nc)))
    return pairs


def _noise_stats(grid: np.ndarray) -> tuple[int, int, bool]:
    repaired = _apply_noise_rule(grid)
    recolored = 0
    erased = 0
    has_pair = False
    for comp in _component_coords(grid == NOISE):
        if len(comp) == 2:
            has_pair = True
        for r, c in comp:
            if int(repaired[r, c]) == BACKGROUND:
                erased += 1
            else:
                recolored += 1
    return recolored, erased, has_pair


def _attempt_generate(rng: random.Random) -> np.ndarray | None:
    sampled = _sample_frames(rng)
    if sampled is None:
        return None
    h, w, frames = sampled
    base, _order = _render_visible_grid(h, w, frames, rng)

    expected_sizes = _intended_sizes(frames)
    colors = [int(frame["color"]) for frame in frames]
    if _sizes_by_color(base, colors) != expected_sizes:
        return None

    critical = _find_critical_noise_sets(base, frames)
    if not critical:
        return None

    chosen_noise = set(rng.choice(critical))

    if rng.random() < 0.55 and not any(
        abs(r0 - r1) + abs(c0 - c1) == 1 for (r0, c0) in chosen_noise for (r1, c1) in chosen_noise if (r0, c0) != (r1, c1)
    ):
        pair_candidates = _find_safe_pairs(base, frames, chosen_noise)
        if pair_candidates:
            pair = rng.choice(pair_candidates)
            chosen_noise.update(pair)

    erase_singletons = _find_safe_singletons(base, frames, chosen_noise, {BACKGROUND})
    if erase_singletons:
        chosen_noise.add(rng.choice(erase_singletons))
    else:
        return None

    if rng.random() < 0.45:
        extra_recolors = _find_safe_singletons(base, frames, chosen_noise, set(colors))
        if extra_recolors:
            chosen_noise.add(rng.choice(extra_recolors))

    inp = _apply_noise_cells(base, chosen_noise)
    repaired = _apply_noise_rule(inp)
    if _sizes_by_color(repaired, colors) != expected_sizes:
        return None

    raw_sizes = _sizes_by_color(inp, colors)
    if raw_sizes == expected_sizes:
        return None

    recolored, erased, has_pair = _noise_stats(inp)
    if recolored == 0 or erased == 0:
        return None

    if has_pair:
        orange_components = _component_coords(inp == NOISE)
        if not any(len(comp) == 2 for comp in orange_components):
            return None

    expected = _build_output([(expected_sizes[color], color) for color in colors])
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
