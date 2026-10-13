from __future__ import annotations

from collections import Counter

import numpy as np


_RNG = np.random.default_rng()

_BG = 0
_BAR = 5
_DOT = 8
_FRAGMENT_COLORS = [1, 2, 3, 4, 6, 7, 9]
_DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in _DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _mask_from_coords(coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask, (r0, c0, r1, c1)


def _detect_bar_and_dots(grid: np.ndarray) -> tuple[int, np.ndarray]:
    row = np.asarray(grid[-1], dtype=int)
    values, counts = np.unique(row, return_counts=True)
    assert values.size == 2, "expected a 2-color position bar on the last row"
    dot_color = int(values[np.argmin(counts)])
    dot_cols = np.flatnonzero(row == dot_color)
    assert dot_cols.size >= 2, "expected at least two position dots"
    diffs = np.diff(dot_cols)
    assert np.all(diffs == diffs[0]), "position dots must be evenly spaced"
    return int(grid.shape[0] - 1), dot_cols.astype(int)


def _detect_axis(mask_a: np.ndarray, mask_b: np.ndarray) -> str:
    assert mask_a.shape == mask_b.shape, "paired fragments must share the same bbox shape"
    vertical = np.array_equal(mask_a, np.fliplr(mask_b))
    horizontal = np.array_equal(mask_a, np.flipud(mask_b))
    assert vertical ^ horizontal, "pair must match exactly one mirror axis"
    return "vertical" if vertical else "horizontal"


def _stitch_half(mask: np.ndarray, axis: str) -> np.ndarray:
    if axis == "vertical":
        h, w = mask.shape
        out = np.zeros((h, 2 * w - 1), dtype=bool)
        out[:, :w] |= mask
        out[:, w - 1 :] |= np.fliplr(mask)
        return out
    if axis == "horizontal":
        h, w = mask.shape
        out = np.zeros((2 * h - 1, w), dtype=bool)
        out[:h, :] |= mask
        out[h - 1 :, :] |= np.flipud(mask)
        return out
    raise ValueError(axis)


def _count_holes(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            touches_edge = False
            while stack:
                cr, cc = stack.pop()
                if cr == 0 or cr == h - 1 or cc == 0 or cc == w - 1:
                    touches_edge = True
                for dr, dc in _DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and not mask[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            if not touches_edge:
                holes += 1
    return holes


def _paint_mask(grid: np.ndarray, mask: np.ndarray, color: int, top: int, left: int) -> None:
    h, w = mask.shape
    window = grid[top : top + h, left : left + w]
    window[mask] = color


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _most_common_color(g)

    bar_row, dot_cols = _detect_bar_and_dots(g)
    work = g.copy()
    work[bar_row, :] = bg

    colors = [int(c) for c in np.unique(work) if int(c) != bg]
    merged = []
    for color in colors:
        comps = _components_of_color(work, color)
        assert len(comps) == 2, "each fragment color must appear in exactly two components"
        mask_a, bbox_a = _mask_from_coords(comps[0])
        mask_b, bbox_b = _mask_from_coords(comps[1])
        axis = _detect_axis(mask_a, mask_b)
        full_mask = _stitch_half(mask_a, axis)
        merged.append(
            {
                "color": color,
                "mask": full_mask,
                "holes": _count_holes(full_mask),
                "anchor": min(int(bbox_a[1]), int(bbox_b[1])),
            }
        )

    assert len(merged) == len(dot_cols), "dot count must equal color-pair count"
    merged.sort(key=lambda item: (int(item["holes"]), int(item["anchor"])))

    out = np.full_like(g, bg)
    for dot_col, item in zip(dot_cols, merged):
        mask = item["mask"]
        h, w = mask.shape
        top = bar_row - h + 1
        left = int(dot_col) - w // 2
        assert top >= 0 and 0 <= left and left + w <= g.shape[1], "merged shape must fit on the output bar"
        _paint_mask(out, mask, int(item["color"]), top, left)
    return out


def _segment_row(width: int, seg_width: int) -> np.ndarray:
    row = np.zeros(width, dtype=bool)
    row[width - seg_width :] = True
    return row


def _solid_rows(width: int, height: int, rng: np.random.Generator) -> list[np.ndarray]:
    rows: list[np.ndarray] = []
    prev = None
    for _ in range(height):
        if prev is None:
            seg_width = int(rng.integers(1, width + 1))
        else:
            low = max(1, prev - 1)
            high = min(width, prev + 1)
            seg_width = int(rng.integers(low, high + 1))
        rows.append(_segment_row(width, seg_width))
        prev = seg_width
    return rows


def _ring_rows(width: int, ring_width: int) -> list[np.ndarray]:
    start = width - ring_width
    top = _segment_row(width, ring_width)
    mid = np.zeros(width, dtype=bool)
    mid[start] = True
    return [top, mid, top.copy()]


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _build_base_half(hole_count: int, rng: np.random.Generator) -> np.ndarray:
    for _ in range(100):
        width = int(rng.integers(2, 5))
        rows: list[np.ndarray] = []
        if hole_count == 0:
            rows.extend(_solid_rows(width, int(rng.choice([3, 5])), rng))
        else:
            cap_height = 1 if hole_count == 2 else int(rng.choice([1, 2]))
            rows.extend(_solid_rows(width, cap_height, rng))
            for idx in range(hole_count):
                ring_width = int(rng.integers(2, width + 1))
                rows.extend(_ring_rows(width, ring_width))
                if idx != hole_count - 1:
                    rows.append(_segment_row(width, int(rng.integers(1, width + 1))))
            rows.extend(_solid_rows(width, cap_height, rng))
        half = _crop_mask(np.array(rows, dtype=bool))
        if min(half.shape) < 2:
            continue
        if _count_holes(_stitch_half(half, "vertical")) != hole_count:
            continue
        return half
    raise RuntimeError("failed to build a valid half-shape")


def _make_shape(hole_count: int, axis: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    for _ in range(100):
        base = _build_base_half(hole_count, rng)
        if axis == "vertical":
            frag_a = base
            frag_b = np.fliplr(base)
        elif axis == "horizontal":
            frag_a = base.T
            frag_b = np.flipud(base.T)
        else:
            raise ValueError(axis)
        vertical = frag_a.shape == frag_b.shape and np.array_equal(frag_a, np.fliplr(frag_b))
        horizontal = frag_a.shape == frag_b.shape and np.array_equal(frag_a, np.flipud(frag_b))
        if axis == "vertical" and not (vertical and not horizontal):
            continue
        if axis == "horizontal" and not (horizontal and not vertical):
            continue
        full = _stitch_half(frag_a, axis)
        if _count_holes(full) != hole_count:
            continue
        return frag_a, frag_b, full
    raise RuntimeError("failed to make a valid mirrored pair")


def _can_place(mask: np.ndarray, top: int, left: int, occupied: np.ndarray, bar_row: int) -> bool:
    h, w = mask.shape
    if top < 0 or left < 0 or top + h > bar_row - 1 or left + w > occupied.shape[1]:
        return False
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(occupied.shape[0], top + h + 1)
    c1 = min(occupied.shape[1], left + w + 1)
    if np.any(occupied[r0:r1, c0:c1]):
        return False
    return True


def _place_fragment(grid: np.ndarray, occupied: np.ndarray, mask: np.ndarray, color: int, top: int, left: int) -> None:
    _paint_mask(grid, mask, color, top, left)
    occupied[top : top + mask.shape[0], left : left + mask.shape[1]][mask] = True


def _expected_output(shape: tuple[int, int], dot_cols: np.ndarray, specs: list[dict[str, object]]) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), _BG, dtype=int)
    bar_row = h - 1
    ordered = sorted(specs, key=lambda item: int(item["holes"]))
    for dot_col, spec in zip(dot_cols, ordered):
        mask = np.asarray(spec["full"], dtype=bool)
        top = bar_row - mask.shape[0] + 1
        left = int(dot_col) - mask.shape[1] // 2
        _paint_mask(out, mask, int(spec["color"]), top, left)
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = _RNG
    pair_count = int(rng.choice([2, 3]))
    hole_choices = [0, 1, 2]
    hole_counts = (
        hole_choices.copy()
        if pair_count == 3
        else list(rng.choice(hole_choices, size=pair_count, replace=False))
    )
    rng.shuffle(hole_counts)

    colors = list(rng.choice(_FRAGMENT_COLORS, size=pair_count, replace=False))
    specs: list[dict[str, object]] = []
    for color, holes in zip(colors, hole_counts):
        axis = "vertical" if rng.random() < 0.5 else "horizontal"
        frag_a, frag_b, full = _make_shape(int(holes), axis, rng)
        specs.append(
            {
                "color": int(color),
                "holes": int(holes),
                "axis": axis,
                "frag_a": frag_a,
                "frag_b": frag_b,
                "full": full,
            }
        )

    ordered = sorted(specs, key=lambda item: int(item["holes"]))
    widths = [int(np.asarray(item["full"]).shape[1]) for item in ordered]
    half_spans = [(w - 1) // 2 for w in widths]
    if len(widths) > 1:
        min_spacing = max(half_spans[i] + half_spans[i + 1] + 2 for i in range(len(widths) - 1))
    else:
        min_spacing = 0
    spacing = min_spacing + int(rng.integers(0, 2))
    start = max(half_spans) + int(rng.integers(0, 2))
    dot_cols = np.array([start + i * spacing for i in range(pair_count)], dtype=int)
    grid_w = int(dot_cols[-1] + max(half_spans) + 2 + rng.integers(0, 2))
    grid_w = max(grid_w, 18)

    max_output_h = max(int(np.asarray(item["full"]).shape[0]) for item in specs)
    grid_h = max(18, max_output_h + 8 + int(rng.integers(0, 3)))
    bar_row = grid_h - 1

    if dot_cols[-1] >= grid_w - 1:
        return None

    grid = np.full((grid_h, grid_w), _BG, dtype=int)
    grid[bar_row, :] = _BAR
    grid[bar_row, dot_cols] = _DOT

    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    occupied[bar_row, :] = True
    fragments: list[tuple[int, np.ndarray]] = []
    for spec in specs:
        fragments.append((int(spec["color"]), np.asarray(spec["frag_a"], dtype=bool)))
        fragments.append((int(spec["color"]), np.asarray(spec["frag_b"], dtype=bool)))
    rng.shuffle(fragments)
    fragments.sort(key=lambda item: -int(np.sum(item[1])))

    for color, mask in fragments:
        placed = False
        h, w = mask.shape
        max_top = bar_row - h - 1
        if max_top < 0:
            return None
        for _ in range(300):
            top = int(rng.integers(0, max_top + 1))
            left = int(rng.integers(0, grid_w - w + 1))
            if not _can_place(mask, top, left, occupied, bar_row):
                continue
            _place_fragment(grid, occupied, mask, color, top, left)
            placed = True
            break
        if not placed:
            return None

    expected = _expected_output(grid.shape, dot_cols, specs)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if int(np.unique(grid).size) < 5:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
