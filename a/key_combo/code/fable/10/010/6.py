from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np


PALETTE = tuple(range(10))
GEN_HEIGHT_RANGE = (18, 30)
GEN_MAX_ATTEMPTS = 800
GEN_CENTER_GAP = 2


def _remaining_bg_color(grid: np.ndarray, top_color: int, bottom_color: int) -> int:
    colors = {int(x) for x in np.unique(grid)}
    assert top_color != bottom_color, "top and bottom border bars must use different colors"
    assert top_color in colors and bottom_color in colors
    bg_candidates = colors - {top_color, bottom_color}
    assert len(bg_candidates) == 1, "grid must use exactly one background color besides the two border-bar colors"
    return bg_candidates.pop()


def _is_square_count(count: int) -> int:
    root = math.isqrt(int(count))
    assert root * root == int(count), "color count inside a source column must be a perfect square"
    return root


def _source_columns(grid: np.ndarray, bg: int) -> list[tuple[int, np.ndarray]]:
    h, w = grid.shape
    out: list[tuple[int, np.ndarray]] = []
    for c in range(w):
        rows = np.where(grid[1 : h - 1, c] != bg)[0] + 1
        if rows.size:
            out.append((c, rows.astype(int)))
    return out


def _paint_inverted_pyramid(out: np.ndarray, center_c: int, height: int, color: int, bg: int) -> None:
    for offset in range(height):
        width = 2 * (height - offset) - 1
        r = 1 + offset
        half = width // 2
        c0, c1 = center_c - half, center_c + half
        assert 0 <= c0 <= c1 < out.shape[1]
        assert np.all(out[r, c0 : c1 + 1] == bg), "reconstructed top piles must not overlap"
        out[r, c0 : c1 + 1] = color


def _paint_upright_pyramid(out: np.ndarray, center_c: int, height: int, color: int, bg: int) -> None:
    for offset in range(height):
        width = 2 * offset + 1
        r = out.shape[0] - 1 - height + offset
        half = width // 2
        c0, c1 = center_c - half, center_c + half
        assert 0 <= c0 <= c1 < out.shape[1]
        assert np.all(out[r, c0 : c1 + 1] == bg), "reconstructed bottom piles must not overlap"
        out[r, c0 : c1 + 1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """Separate each mixed grain column by bar color and rebuild two opposite centered sand piles.

    Input structure:
    - The top row is a full-width border bar in color A and the bottom row is a full-width border bar in color B.
    - Interior cells contain background plus sparse vertical source columns made only of colors A and B.
    - In each source column, the number of A pixels and the number of B pixels are each perfect squares; either count may be zero.

    Rule:
    - Clear the interior to background while keeping the two border bars unchanged.
    - For each source column, count the A grains and rebuild them under the top bar as an inverted centered pyramid with row widths `2m-1, 2m-3, ..., 1`, where `m^2` is the A count.
    - Count the B grains and rebuild them above the bottom bar as an upright centered pyramid with row widths `1, 3, ..., 2n-1`, where `n^2` is the B count.

    Key insight:
    - Gravity is color-dependent rather than global: the same source column must be split by color into two square-number reservoirs, and each reservoir is then reshaped using the odd-width decomposition of a square.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "input must be a non-empty 2-D grid"
    h, w = g.shape
    assert h <= 30 and w <= 30, "ARC grids must fit within 30x30"
    assert h >= 4 and w >= 1

    top_color = int(g[0, 0])
    bottom_color = int(g[h - 1, 0])
    assert bool(np.all(g[0, :] == top_color)), "top row must be a full-width border bar"
    assert bool(np.all(g[h - 1, :] == bottom_color)), "bottom row must be a full-width border bar"
    bg = _remaining_bg_color(g, top_color, bottom_color)

    interior = g[1 : h - 1, :]
    assert set(map(int, np.unique(interior))) <= {bg, top_color, bottom_color}, "interior may only use background and the two bar colors"

    cols = _source_columns(g, bg)
    assert cols, "expected at least one source column"

    out = np.full_like(g, bg)
    out[0, :] = top_color
    out[h - 1, :] = bottom_color

    for center_c, rows in cols:
        colors = [int(g[r, center_c]) for r in rows]
        top_count = sum(color == top_color for color in colors)
        bottom_count = sum(color == bottom_color for color in colors)
        assert top_count + bottom_count == len(colors) and len(colors) > 0

        top_height = _is_square_count(top_count)
        bottom_height = _is_square_count(bottom_count)
        assert top_height > 0 or bottom_height > 0

        radius = max(top_height, bottom_height) - 1
        assert center_c - radius >= 0 and center_c + radius < w, "reconstructed pile must fit horizontally"

        if top_height:
            _paint_inverted_pyramid(out, center_c, top_height, top_color, bg)
        if bottom_height:
            _paint_upright_pyramid(out, center_c, bottom_height, bottom_color, bg)

    return out


def _sample_specs() -> list[dict[str, int]]:
    count = random.choices([2, 3, 4], weights=[2, 4, 3], k=1)[0]
    archetypes = ["mixed"]
    if count >= 3 and random.random() < 0.8:
        archetypes.extend(["top_only", "bottom_only"])
    while len(archetypes) < count:
        archetypes.append(random.choice(["mixed", "mixed", "top_only", "bottom_only"]))
    random.shuffle(archetypes)

    specs: list[dict[str, int]] = []
    for archetype in archetypes:
        if archetype == "mixed":
            top_h = random.choice([1, 2, 3])
            bottom_h = random.choice([1, 2, 3])
        elif archetype == "top_only":
            top_h = random.choice([1, 2, 3])
            bottom_h = 0
        else:
            top_h = 0
            bottom_h = random.choice([1, 2, 3])
        specs.append(
            {
                "top_h": top_h,
                "bottom_h": bottom_h,
                "top_count": top_h * top_h,
                "bottom_count": bottom_h * bottom_h,
                "radius": max(top_h, bottom_h) - 1,
            }
        )
    assert any(spec["top_h"] > 0 and spec["bottom_h"] > 0 for spec in specs)
    return specs


def _sample_dimensions(specs: list[dict[str, int]]) -> tuple[int, int] | None:
    max_pixels = max(spec["top_count"] + spec["bottom_count"] for spec in specs)
    max_top = max(spec["top_h"] for spec in specs)
    max_bottom = max(spec["bottom_h"] for spec in specs)

    min_h = max(GEN_HEIGHT_RANGE[0], max_pixels + 2, max_top + max_bottom + 4)
    if min_h > GEN_HEIGHT_RANGE[1]:
        return None
    h = random.randint(min_h, GEN_HEIGHT_RANGE[1])

    radii = [spec["radius"] for spec in specs]
    min_w = sum(2 * r + 1 for r in radii) + GEN_CENTER_GAP * max(0, len(radii) - 1)
    min_w = max(9, min_w)
    if min_w > 30:
        return None
    w = random.randint(min_w, 30)
    return h, w


def _sample_centers(width: int, radii: list[int]) -> list[int] | None:
    order = sorted(range(len(radii)), key=lambda i: (-radii[i], i))
    centers = [None] * len(radii)

    for idx in order:
        r = radii[idx]
        candidates = []
        for c in range(r, width - r):
            ok = True
            for other_idx, other_center in enumerate(centers):
                if other_center is None:
                    continue
                min_sep = radii[idx] + radii[other_idx] + GEN_CENTER_GAP
                if abs(c - other_center) < min_sep:
                    ok = False
                    break
            if ok:
                candidates.append(c)
        if not candidates:
            return None
        centers[idx] = random.choice(candidates)
    return [int(c) for c in centers]


def _distributed_sequence(top_count: int, bottom_count: int, top_color: int, bottom_color: int) -> list[int]:
    if top_count == 0:
        return [bottom_color] * bottom_count
    if bottom_count == 0:
        return [top_color] * top_count

    entries: list[tuple[float, int]] = []
    for i in range(top_count):
        jitter = random.uniform(-0.2, 0.2) / max(1, top_count)
        entries.append(((i + 0.5) / top_count + jitter, top_color))
    for i in range(bottom_count):
        jitter = random.uniform(-0.2, 0.2) / max(1, bottom_count)
        entries.append(((i + 0.5) / bottom_count + jitter, bottom_color))
    entries.sort(key=lambda item: item[0])

    seq = [color for _, color in entries]
    if all(color == seq[0] for color in seq):
        return seq
    switches = sum(seq[i] != seq[i - 1] for i in range(1, len(seq)))
    assert switches >= 1, "mixed columns must visibly interleave the two colors"
    return seq


def _draw_expected(
    shape: tuple[int, int],
    bg: int,
    top_color: int,
    bottom_color: int,
    specs: list[dict[str, int]],
) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), bg, dtype=int)
    out[0, :] = top_color
    out[h - 1, :] = bottom_color
    for spec in specs:
        center_c = spec["center"]
        if spec["top_h"]:
            _paint_inverted_pyramid(out, center_c, spec["top_h"], top_color, bg)
        if spec["bottom_h"]:
            _paint_upright_pyramid(out, center_c, spec["bottom_h"], bottom_color, bg)
    return out


def _build_input(
    shape: tuple[int, int],
    bg: int,
    top_color: int,
    bottom_color: int,
    specs: list[dict[str, int]],
) -> np.ndarray | None:
    h, w = shape
    inp = np.full((h, w), bg, dtype=int)
    inp[0, :] = top_color
    inp[h - 1, :] = bottom_color

    for spec in specs:
        total = spec["top_count"] + spec["bottom_count"]
        rows = sorted(random.sample(range(1, h - 1), total))
        seq = _distributed_sequence(spec["top_count"], spec["bottom_count"], top_color, bottom_color)
        if len(seq) != len(rows):
            return None
        for r, color in zip(rows, seq):
            inp[r, spec["center"]] = color

    return inp


def _recover_counts(grid: np.ndarray) -> list[tuple[int, int, int]]:
    top_color = int(grid[0, 0])
    bottom_color = int(grid[-1, 0])
    bg = _remaining_bg_color(grid, top_color, bottom_color)
    recovered: list[tuple[int, int, int]] = []
    for c, rows in _source_columns(grid, bg):
        colors = [int(grid[r, c]) for r in rows]
        recovered.append((c, sum(color == top_color for color in colors), sum(color == bottom_color for color in colors)))
    return recovered


def _semantic_checks(inp: np.ndarray, expected: np.ndarray, specs: list[dict[str, int]]) -> bool:
    if inp.ndim != 2 or expected.ndim != 2:
        return False
    if inp.shape != expected.shape or max(inp.shape) > 30:
        return False
    if not np.array_equal(main(inp), expected):
        return False

    top_color = int(inp[0, 0])
    bottom_color = int(inp[-1, 0])
    bg = _remaining_bg_color(inp, top_color, bottom_color)
    if Counter(map(int, inp.flat)).most_common(1)[0][0] != bg:
        return False

    recovered = sorted(_recover_counts(inp))
    intended = sorted((spec["center"], spec["top_count"], spec["bottom_count"]) for spec in specs)
    if recovered != intended:
        return False

    if not any(top > 0 and bottom > 0 for _, top, bottom in recovered):
        return False
    if len(recovered) >= 3 and not any(top == 0 or bottom == 0 for _, top, bottom in recovered):
        return False

    for center, top_count, bottom_count in recovered:
        top_h = math.isqrt(top_count)
        bottom_h = math.isqrt(bottom_count)
        radius = max(top_h, bottom_h) - 1
        if center - radius < 0 or center + radius >= inp.shape[1]:
            return False

    out_internal = expected[1:-1, :]
    if np.any((out_internal != bg) & (out_internal != top_color) & (out_internal != bottom_color)):
        return False

    return True


def _generate_one() -> np.ndarray | None:
    bg, top_color, bottom_color = random.sample(PALETTE, 3)
    specs = _sample_specs()
    dims = _sample_dimensions(specs)
    if dims is None:
        return None
    h, w = dims

    centers = _sample_centers(w, [spec["radius"] for spec in specs])
    if centers is None:
        return None
    for spec, center in zip(specs, centers):
        spec["center"] = int(center)

    expected = _draw_expected((h, w), bg, top_color, bottom_color, specs)
    inp = _build_input((h, w), bg, top_color, bottom_color, specs)
    if inp is None:
        return None

    if _semantic_checks(inp, expected, specs):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _generate_one()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid example after {GEN_MAX_ATTEMPTS} attempts")
