from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_RETRIES = 400


def _mode(values: list[int]) -> int:
    return int(Counter(values).most_common(1)[0][0])


def _background_color(grid: np.ndarray) -> int:
    border = np.concatenate([grid[0, :], grid[-1, :], grid[1:-1, 0], grid[1:-1, -1]])
    return _mode([int(v) for v in border.tolist()])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []

    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue

        seen[sr, sc] = True
        q = deque([(sr, sc)])
        coords: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            coords.append((r, c))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        parts.append(coords)

    return parts


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _interval_gap(a0: int, a1: int, b0: int, b1: int) -> int:
    if a1 < b0:
        return b0 - a1 - 1
    if b1 < a0:
        return a0 - b1 - 1
    return 0


def _find_poles(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    poles: list[dict[str, object]] = []

    for c in range(w):
        color = int(g[0, c])
        if color == bg:
            continue

        height = 1
        while height < h and int(g[height, c]) == color:
            height += 1
        if height < 2:
            continue

        if c > 0 and any(int(g[r, c - 1]) != bg for r in range(1, height)):
            continue
        if c + 1 < w and any(int(g[r, c + 1]) != bg for r in range(1, height)):
            continue

        flags: list[int] = []
        cc = c + 1
        while cc < w and int(g[0, cc]) != bg:
            flags.append(int(g[0, cc]))
            cc += 1

        if not flags or color in flags or len(flags) != len(set(flags)):
            continue

        poles.append(
            {
                "col": int(c),
                "height": int(height),
                "color": int(color),
                "flags": flags,
            }
        )

    assert poles, "expected at least one pole with a flag sequence"
    return poles


def _find_regions(grid: np.ndarray, bg: int, poles: list[dict[str, object]]) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    mask = g != bg
    mask[0, :] = False
    for pole in poles:
        col = int(pole["col"])
        height = int(pole["height"])
        mask[:height, col] = False

    regions: list[dict[str, object]] = []
    for coords in _components(mask):
        r0, c0, r1, c1 = _bbox(coords)
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        if len(coords) != patch.size or np.any(patch == bg):
            continue
        if any(not np.all(patch[:, j] == patch[0, j]) for j in range(patch.shape[1])):
            continue

        colors = [int(patch[0, j]) for j in range(patch.shape[1])]
        if len(colors) < 2 or len(colors) != len(set(colors)):
            continue

        regions.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "patch": patch.copy(),
                "colors": colors,
                "width": int(patch.shape[1]),
            }
        )

    assert regions, "expected at least one striped rectangle below the poles"
    return regions


def _match_poles_to_regions(
    poles: list[dict[str, object]],
    regions: list[dict[str, object]],
) -> list[tuple[dict[str, object], dict[str, object]]]:
    matches: list[tuple[dict[str, object], dict[str, object]]] = []
    used: set[int] = set()

    for pole in sorted(poles, key=lambda item: int(item["col"])):
        pole_col = int(pole["col"])
        pole_bottom = int(pole["height"]) - 1
        flags = list(pole["flags"])
        flag_set = set(flags)
        flag_left = pole_col + 1
        flag_right = pole_col + len(flags)

        best: tuple[int, int, int, int] | None = None
        for idx, region in enumerate(regions):
            if idx in used:
                continue
            if int(region["width"]) != len(flags):
                continue
            if set(region["colors"]) != flag_set:
                continue

            r0, c0, _r1, c1 = region["bbox"]
            if r0 <= pole_bottom:
                continue

            vertical_gap = int(r0 - pole_bottom - 1)
            horizontal_gap = _interval_gap(int(c0), int(c1), int(flag_left), int(flag_right))
            center_delta = abs((int(c0) + int(c1)) - (int(flag_left) + int(flag_right)))
            score = (vertical_gap, horizontal_gap, center_delta, idx)
            if best is None or score < best:
                best = score

        assert best is not None, "each pole must have one matching striped rectangle"
        region = regions[best[3]]
        used.add(best[3])
        matches.append((pole, region))

    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder each striped rectangle's columns to match the colored flag sequence hanging from the pole above it.

    Input structure:
    - The background is black.
    - Along the top edge, one-pixel-wide vertical poles descend for several rows, and each pole's top cell has a horizontal rightward sequence of distinct colored flag pixels.
    - Below each pole sits one full rectangle whose columns are solid colors; its column colors are exactly the flag colors but in a different order.

    Rule:
    - Find every pole-and-flag sequence and read the flag colors from left to right.
    - Locate the striped rectangle below that uses the same set of colors and has the same number of columns.
    - Inside that rectangle, move the colored columns so that the left-to-right column order matches the flag sequence.
    - Leave the poles and the flag pixels unchanged.

    Key insight:
    - The flags are an ordering legend: their visible color order tells you how to permute the matching color columns in the rectangle below.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    poles = _find_poles(g, bg)
    regions = _find_regions(g, bg, poles)
    matches = _match_poles_to_regions(poles, regions)

    out = g.copy()
    for pole, region in matches:
        patch = np.asarray(region["patch"], dtype=int)
        color_to_col = {int(color): idx for idx, color in enumerate(region["colors"])}
        reordered = np.empty_like(patch)
        for new_col, color in enumerate(pole["flags"]):
            reordered[:, new_col] = patch[:, color_to_col[int(color)]]

        r0, c0, r1, c1 = region["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = reordered

    return out


def _deranged(values: list[int]) -> list[int]:
    shuffled = list(values)
    while True:
        random.shuffle(shuffled)
        if shuffled != values:
            return shuffled.copy()


def _sample_lengths(block_count: int) -> list[int] | None:
    for _ in range(200):
        lengths = [random.randint(2, 5) for _ in range(block_count)]
        if sum(lengths) <= len(PALETTE):
            return lengths
    return None


def _distribute_extra(extra: int, slot_count: int) -> list[int]:
    pieces = [0] * slot_count
    for _ in range(extra):
        pieces[random.randrange(slot_count)] += 1
    return pieces


def _render_case(specs: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    gaps = [random.randint(1, 4) for _ in range(len(specs) - 1)]
    left_margin = random.randint(0, 2)
    right_margin = random.randint(0, 2)

    base_w = left_margin + right_margin + sum(1 + len(spec["flags"]) for spec in specs) + sum(gaps)
    extra_w = random.randint(0, min(30 - base_w, 6))
    spread = _distribute_extra(extra_w, len(gaps) + 2)
    left_margin += spread[0]
    right_margin += spread[1]
    for i in range(len(gaps)):
        gaps[i] += spread[i + 2]

    width = left_margin + right_margin + sum(1 + len(spec["flags"]) for spec in specs) + sum(gaps)
    height = max(int(spec["pole_h"]) + int(spec["gap_v"]) + int(spec["rect_h"]) for spec in specs)
    height += random.randint(0, min(30 - height, 4))

    grid = np.full((height, width), BLACK, dtype=int)
    expected = grid.copy()

    x = left_margin
    for i, spec in enumerate(specs):
        pole_h = int(spec["pole_h"])
        rect_h = int(spec["rect_h"])
        gap_v = int(spec["gap_v"])
        flags = list(spec["flags"])
        region = list(spec["region"])
        align = str(spec["align"])

        grid[:pole_h, x] = GREY
        expected[:pole_h, x] = GREY
        for j, color in enumerate(flags, start=1):
            grid[0, x + j] = int(color)
            expected[0, x + j] = int(color)

        rect_top = pole_h + gap_v
        rect_left = x + 1 if align == "flags" else x
        for j, color in enumerate(region):
            grid[rect_top : rect_top + rect_h, rect_left + j] = int(color)
        for j, color in enumerate(flags):
            expected[rect_top : rect_top + rect_h, rect_left + j] = int(color)

        if i < len(gaps):
            x += 1 + len(flags) + gaps[i]

    return grid, expected


def _gen_attempt() -> np.ndarray | None:
    block_count = random.randint(1, 3)
    lengths = _sample_lengths(block_count)
    if lengths is None:
        return None

    colors = random.sample(PALETTE, sum(lengths))
    specs: list[dict[str, object]] = []
    color_idx = 0
    for length in lengths:
        flag_colors = colors[color_idx : color_idx + length]
        color_idx += length
        flag_order = random.sample(flag_colors, length)
        specs.append(
            {
                "flags": flag_order,
                "region": _deranged(flag_order),
                "pole_h": random.randint(2, 7),
                "gap_v": random.randint(1, 4),
                "rect_h": random.randint(2, 6),
                "align": random.choice(["flags", "pole"]),
            }
        )

    grid, expected = _render_case(specs)
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None
    if np.array_equal(grid, expected):
        return None

    try:
        if np.array_equal(main(grid), expected):
            return grid
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid flag-column permutation puzzle")
