from __future__ import annotations

import numpy as np


BG = 0
COLOR_POOL = np.arange(1, 10, dtype=int)
MAX_GRID_SIDE = 30
MAX_GENERATION_ATTEMPTS = 500
_RNG = np.random.default_rng()


def _most_common_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _border_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _find_rings(grid: np.ndarray, bg: int) -> list[dict[str, int]]:
    rings: list[dict[str, int]] = []
    for color in (int(c) for c in np.unique(grid) if int(c) != int(bg)):
        coords = np.argwhere(grid == color)
        assert coords.size > 0, "each non-background color must be present"
        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "rings need a non-empty interior"

        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = _border_mask(*patch.shape)
        assert np.all(patch[border] == color), "color bbox must be a rectangular frame"

        rings.append({"color": color, "r0": r0, "c0": c0, "r1": r1, "c1": c1})

    rings.sort(key=lambda ring: (ring["r0"], ring["c0"]))
    assert rings, "expected at least one ring"
    return rings


def _blank_suffix_start(grid: np.ndarray, bg: int) -> int:
    col = int(grid.shape[1])
    while col > 0 and bool(np.all(grid[:, col - 1] == bg)):
        col -= 1
    assert col < int(grid.shape[1]), "expected a blank recycle zone on the right"
    return col


def _repair_inner(inner: np.ndarray, color: int, bg: int) -> tuple[np.ndarray, np.ndarray]:
    mask = inner == int(color)
    defects = mask & (~mask[::-1, ::-1])
    repaired = inner.copy()
    repaired[defects] = int(bg)
    return repaired, defects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove one-sided pixels that break 180-degree rotational symmetry inside each
    rectangular ring, then count those removed pixels in right-side recycle columns.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    rings = _find_rings(g, bg)
    recycle_start = _blank_suffix_start(g, bg)
    assert g.shape[1] - recycle_start >= len(rings), "recycle zone must have one column per ring"

    out = g.copy()
    defect_counts: list[int] = []

    for ring in rings:
        r0, c0, r1, c1 = ring["r0"], ring["c0"], ring["r1"], ring["c1"]
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        repaired, defects = _repair_inner(inner, ring["color"], bg)
        out[r0 + 1 : r1, c0 + 1 : c1] = repaired
        defect_counts.append(int(defects.sum()))

    for idx, ring in enumerate(rings):
        col = recycle_start + idx
        out[:, col] = int(bg)
        out[: defect_counts[idx], col] = int(ring["color"])

    return out


def _pair_orbits(height: int, width: int) -> tuple[list[tuple[tuple[int, int], tuple[int, int]]], tuple[int, int] | None]:
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    center: tuple[int, int] | None = None
    for r in range(height):
        for c in range(width):
            rr = height - 1 - r
            cc = width - 1 - c
            if r > rr or (r == rr and c > cc):
                continue
            if r == rr and c == cc:
                center = (r, c)
            else:
                pairs.append(((r, c), (rr, cc)))
    return pairs, center


def _make_inner_pattern(height: int, width: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    pairs, center = _pair_orbits(height, width)
    assert pairs, "interior must have at least one paired orbit"

    for _ in range(100):
        fill_prob = float(_RNG.uniform(0.18, 0.48))
        filled_pairs = _RNG.random(len(pairs)) < fill_prob

        if not bool(filled_pairs.any()):
            filled_pairs[int(_RNG.integers(len(pairs)))] = True
        if bool(filled_pairs.all()):
            filled_pairs[int(_RNG.integers(len(pairs)))] = False

        empty_indices = np.flatnonzero(~filled_pairs)
        if empty_indices.size == 0:
            continue

        mask = np.zeros((height, width), dtype=bool)
        for idx, is_filled in enumerate(filled_pairs):
            if not bool(is_filled):
                continue
            (r0, c0), (r1, c1) = pairs[idx]
            mask[r0, c0] = True
            mask[r1, c1] = True

        if center is not None and bool(_RNG.integers(0, 2)):
            mask[center] = True

        if int(mask.sum()) < 2:
            continue

        defect_cap = min(5, int(empty_indices.size))
        defect_count = int(_RNG.integers(1, defect_cap + 1))
        chosen = _RNG.choice(empty_indices, size=defect_count, replace=False)

        defects: list[tuple[int, int]] = []
        for idx in np.atleast_1d(chosen):
            pair = pairs[int(idx)]
            cell = pair[int(_RNG.integers(0, 2))]
            mask[cell] = True
            defects.append((int(cell[0]), int(cell[1])))

        return mask, defects

    raise RuntimeError("failed to build a non-degenerate inner pattern")


def _draw_ring(
    grid: np.ndarray,
    top: int,
    left: int,
    outer_h: int,
    outer_w: int,
    color: int,
    inner_mask: np.ndarray,
) -> None:
    bottom = top + outer_h - 1
    right = left + outer_w - 1
    grid[top : bottom + 1, left] = int(color)
    grid[top : bottom + 1, right] = int(color)
    grid[top, left : right + 1] = int(color)
    grid[bottom, left : right + 1] = int(color)
    grid[top + 1 : bottom, left + 1 : right][inner_mask] = int(color)


def _expected_output(grid: np.ndarray, specs: list[dict[str, object]], recycle_start: int) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        for r, c in spec["defects_abs"]:
            out[int(r), int(c)] = BG

    for idx, spec in enumerate(specs):
        col = recycle_start + idx
        out[:, col] = BG
        out[: len(spec["defects_abs"]), col] = int(spec["color"])

    return out


def _gen_attempt() -> np.ndarray | None:
    ring_count = int(_RNG.integers(1, 5))
    outer_heights = [int(_RNG.integers(5, 9)) for _ in range(ring_count)]
    outer_widths = [int(_RNG.integers(5, 10)) for _ in range(ring_count)]

    top_margin = int(_RNG.integers(1, 4))
    bottom_margin = int(_RNG.integers(1, 4))
    gaps = [int(_RNG.integers(1, 4)) for _ in range(max(0, ring_count - 1))]

    height = top_margin + bottom_margin + sum(outer_heights) + sum(gaps)
    if height > MAX_GRID_SIDE:
        return None

    rows: list[int] = []
    cursor = top_margin
    for idx, outer_h in enumerate(outer_heights):
        rows.append(cursor)
        cursor += outer_h
        if idx < ring_count - 1:
            cursor += gaps[idx]

    field_width = max(outer_widths) + int(_RNG.integers(3, 11))
    cols = [int(_RNG.integers(1, field_width - outer_w + 1)) for outer_w in outer_widths]
    last_nonbg = max(col + outer_w - 1 for col, outer_w in zip(cols, outer_widths, strict=False))

    suffix_extra = int(_RNG.integers(0, 4))
    width = last_nonbg + 1 + ring_count + suffix_extra
    if width > MAX_GRID_SIDE:
        return None

    grid = np.full((height, width), BG, dtype=int)
    colors = [int(c) for c in _RNG.choice(COLOR_POOL, size=ring_count, replace=False)]
    specs: list[dict[str, object]] = []

    for row, col, outer_h, outer_w, color in zip(rows, cols, outer_heights, outer_widths, colors, strict=False):
        inner_h = outer_h - 2
        inner_w = outer_w - 2
        inner_mask, defects_rel = _make_inner_pattern(inner_h, inner_w)
        _draw_ring(grid, row, col, outer_h, outer_w, color, inner_mask)

        defects_abs = [(row + 1 + r, col + 1 + c) for r, c in defects_rel]
        specs.append(
            {
                "color": color,
                "r0": row,
                "c0": col,
                "r1": row + outer_h - 1,
                "c1": col + outer_w - 1,
                "defects_abs": defects_abs,
            }
        )

    recycle_start = last_nonbg + 1
    assert width - recycle_start >= ring_count
    assert bool(np.all(grid[:, recycle_start:] == BG))

    expected = _expected_output(grid, specs, recycle_start)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
