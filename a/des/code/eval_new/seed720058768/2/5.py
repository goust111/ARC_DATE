from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _is_rectangular_ring(points: list[tuple[int, int]], box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in points:
        sub[r - r0, c - c0] = True
    expected = np.zeros_like(sub, dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return np.array_equal(sub, expected)


def _draw_ring(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _ring_is_clear(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, bg: int) -> bool:
    return (
        np.all(grid[r0, c0 : c1 + 1] == bg)
        and np.all(grid[r1, c0 : c1 + 1] == bg)
        and np.all(grid[r0 : r1 + 1, c0] == bg)
        and np.all(grid[r0 : r1 + 1, c1] == bg)
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each nested rectangular frame group from outside to inside using the top-row palette order.

    Input structure:
    - The most common color is the background.
    - The top row is a palette row: distinct single-pixel colors appear left-to-right with exactly one background pixel between adjacent palette entries.
    - Below the palette row are several non-overlapping groups of concentric axis-aligned rectangular rings, each ring one pixel thick with one background cell of spacing to the next ring.
    - All rings in the input share the same foreground color, while different groups may contain different numbers of rings.

    Rule:
    - Keep the palette row unchanged.
    - Split the lower non-background pixels into individual rectangular-ring components.
    - Group rings by shared center, then sort each group's rings by ring area from largest to smallest, which is equivalent to outermost to innermost.
    - Recolor the k-th largest ring in every group with the k-th palette color, using only the available rings in shorter groups.

    Key insight:
    - The top row is a global legend ordered by layer rank, so solving the task requires separating rings, regrouping them by common center, and mapping outer-to-inner rank to palette position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 4

    bg = _get_bg(g)
    palette_cols = np.flatnonzero(g[0] != bg)
    assert palette_cols.size >= 2
    if palette_cols.size > 1:
        assert np.all(np.diff(palette_cols) == 2)
    palette = [int(g[0, c]) for c in palette_cols]
    assert len(set(palette)) == len(palette)

    mask = g != bg
    mask[0, :] = False
    rings = []
    for pts in _components(mask):
        box = _bbox(pts)
        assert _is_rectangular_ring(pts, box)
        rings.append(
            {
                "pts": pts,
                "bbox": box,
                "center": (box[0] + box[2], box[1] + box[3]),
                "area": len(pts),
            }
        )

    assert rings
    groups: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    for ring in rings:
        groups[ring["center"]].append(ring)

    out = g.copy()
    for group_rings in groups.values():
        group_rings.sort(key=lambda ring: int(ring["area"]), reverse=True)
        assert len(group_rings) <= len(palette)
        for outer, inner in zip(group_rings, group_rings[1:]):
            ro0, co0, ro1, co1 = outer["bbox"]  # type: ignore[assignment]
            ri0, ci0, ri1, ci1 = inner["bbox"]  # type: ignore[assignment]
            assert ri0 - ro0 >= 2 and ci0 - co0 >= 2 and ro1 - ri1 >= 2 and co1 - ci1 >= 2
        for color, ring in zip(palette, group_rings):
            for r, c in ring["pts"]:  # type: ignore[index]
                out[r, c] = int(color)

    out[0, :] = g[0, :]
    return out


GEN_RETRIES = 400


def _sample_group_specs(rng: np.random.Generator, palette_len: int, group_count: int) -> list[dict[str, int]]:
    depths = rng.integers(1, palette_len + 1, size=group_count).tolist()
    max_idx = int(rng.integers(group_count))
    depths[max_idx] = palette_len
    if group_count > 1 and palette_len > 1 and rng.random() < 0.7 and min(depths) == palette_len:
        other = [idx for idx in range(group_count) if idx != max_idx]
        depths[int(other[int(rng.integers(len(other)))])] = int(rng.integers(1, palette_len))

    specs: list[dict[str, int]] = []
    for depth in depths:
        if depth <= 2:
            inner_h_max, inner_w_max = 6, 7
        elif depth == 3:
            inner_h_max, inner_w_max = 5, 6
        else:
            inner_h_max, inner_w_max = 4, 5
        inner_h = int(rng.integers(3, inner_h_max + 1))
        inner_w = int(rng.integers(3, inner_w_max + 1))
        outer_h = inner_h + 4 * (depth - 1)
        outer_w = inner_w + 4 * (depth - 1)
        area = sum(2 * ((outer_h - 4 * layer) + (outer_w - 4 * layer)) - 4 for layer in range(depth))
        specs.append(
            {
                "depth": int(depth),
                "outer_h": int(outer_h),
                "outer_w": int(outer_w),
                "area": int(area),
            }
        )
    return specs


def _layout_groups(
    rng: np.random.Generator,
    specs: list[dict[str, int]],
    palette_len: int,
    occupied: int,
) -> tuple[int, int, list[tuple[int, int, int]], int] | None:
    group_count = len(specs)
    order = rng.permutation(group_count).tolist()
    row_count = 1 if group_count == 1 or rng.random() < 0.5 else 2
    rows = [list(chunk) for chunk in np.array_split(order, row_count) if len(chunk) > 0]

    row_defs = []
    for row in rows:
        gap_count = max(0, len(row) - 1)
        gaps = [int(rng.integers(2, 6)) for _ in range(gap_count)]
        left_margin = int(rng.integers(1, 4))
        right_margin = int(rng.integers(1, 4))
        max_h = max(specs[idx]["outer_h"] for idx in row)
        band_height = max_h + int(rng.integers(0, 3))
        row_width = left_margin + right_margin + sum(specs[idx]["outer_w"] for idx in row) + sum(gaps)
        row_defs.append(
            {
                "indices": row,
                "gaps": gaps,
                "left_margin": left_margin,
                "band_height": band_height,
                "row_width": row_width,
            }
        )

    top_gap = int(rng.integers(1, 4))
    bottom_margin = int(rng.integers(1, 4))
    row_gaps = [int(rng.integers(1, 5)) for _ in range(max(0, len(row_defs) - 1))]
    min_h = 1 + top_gap + bottom_margin + sum(row["band_height"] for row in row_defs) + sum(row_gaps)
    min_w = max(2 * palette_len - 1, max(row["row_width"] for row in row_defs))
    if min_h > 30 or min_w > 30:
        return None

    candidates = [(h, w) for h in range(min_h, 31) for w in range(min_w, 31) if h * w > 2 * occupied]
    if not candidates:
        return None
    min_area = min(h * w for h, w in candidates)
    near = [(h, w) for h, w in candidates if h * w <= min_area + 80]
    h, w = near[int(rng.integers(len(near)))]

    placements: list[tuple[int, int, int]] = []
    row_top = 1 + top_gap
    for row_idx, row in enumerate(row_defs):
        row_offset = int(rng.integers(0, w - row["row_width"] + 1))
        cursor = row_offset + row["left_margin"]
        for i, group_idx in enumerate(row["indices"]):
            spec = specs[group_idx]
            jitter = int(rng.integers(0, row["band_height"] - spec["outer_h"] + 1))
            placements.append((group_idx, row_top + jitter, cursor))
            cursor += spec["outer_w"]
            if i < len(row["gaps"]):
                cursor += row["gaps"][i]
        row_top += row["band_height"]
        if row_idx < len(row_gaps):
            row_top += row_gaps[row_idx]

    palette_offset = int(rng.integers(0, w - (2 * palette_len - 1) + 1))
    return h, w, placements, palette_offset


def _make_attempt(rng: np.random.Generator) -> np.ndarray | None:
    bg = int(rng.integers(0, 10))
    palette_len = int(rng.integers(2, 6))
    max_groups = 4 if palette_len <= 3 else 3
    group_count = int(rng.integers(1, max_groups + 1))

    available = [color for color in range(10) if color != bg]
    rng.shuffle(available)
    palette = [int(color) for color in available[:palette_len]]
    frame_color = int(available[palette_len])

    specs = _sample_group_specs(rng, palette_len, group_count)
    occupied = palette_len + sum(spec["area"] for spec in specs)
    layout = _layout_groups(rng, specs, palette_len, occupied)
    if layout is None:
        return None
    h, w, placements, palette_offset = layout

    grid = np.full((h, w), bg, dtype=int)
    for idx, color in enumerate(palette):
        grid[0, palette_offset + 2 * idx] = int(color)

    for group_idx, r0, c0 in placements:
        spec = specs[group_idx]
        for layer in range(spec["depth"]):
            inset = 2 * layer
            rr0 = r0 + inset
            cc0 = c0 + inset
            rr1 = r0 + spec["outer_h"] - 1 - inset
            cc1 = c0 + spec["outer_w"] - 1 - inset
            if not _ring_is_clear(grid, rr0, cc0, rr1, cc1, bg):
                return None
            _draw_ring(grid, rr0, cc0, rr1, cc1, frame_color)

    if _get_bg(grid) != bg:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _make_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
