import math
from collections import deque

import numpy as np


BG = 0
GREY = 5
MAROON = 9
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GENERATION_ATTEMPTS = 400


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _border_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _find_color_components(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _find_intervals(mask_1d: np.ndarray) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    start = None
    for idx, value in enumerate(mask_1d.tolist()):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            intervals.append((start, idx - 1))
            start = None
    if start is not None:
        intervals.append((start, len(mask_1d) - 1))
    return intervals


def _rect_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = outer
    rr0, cc0, rr1, cc1 = inner
    return r0 < rr0 and c0 < cc0 and rr1 < r1 and cc1 < c1


def _extract_legend_and_main(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    grey_components = _find_color_components(grid, GREY)
    h, w = grid.shape
    candidates = []
    for comp in grey_components:
        r0, c0, r1, c1 = _bbox_from_coords(comp)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if hh < 3 or ww < 3:
            continue
        local = np.zeros((hh, ww), dtype=bool)
        for r, c in comp:
            local[r - r0, c - c0] = True
        if not np.array_equal(local, _border_mask(hh, ww)):
            continue
        touches_right = c1 == w - 1 and r0 == 0 and r1 == h - 1 and c0 > 0
        touches_bottom = r1 == h - 1 and c0 == 0 and c1 == w - 1 and r0 > 0
        if touches_right or touches_bottom:
            candidates.append((hh * ww, (r0, c0, r1, c1), "right" if touches_right else "bottom"))
    assert candidates, "legend not found"
    _area, bbox, side = max(candidates, key=lambda item: item[0])
    r0, c0, r1, c1 = bbox
    if side == "right":
        main = grid[:, :c0].copy()
        legend = grid[r0 : r1 + 1, c0 : c1 + 1]
    else:
        main = grid[:r0, :].copy()
        legend = grid[r0 : r1 + 1, c0 : c1 + 1]
    return main, legend


def _parse_legend(legend: np.ndarray) -> dict[int, np.ndarray]:
    inner = legend[1:-1, 1:-1]
    row_intervals = _find_intervals(np.any(inner != BG, axis=1))
    mapping: dict[int, np.ndarray] = {}
    for r0, r1 in row_intervals:
        band = inner[r0 : r1 + 1]
        count = int(np.count_nonzero(band == MAROON))
        assert count > 0
        pat_mask = (band != BG) & (band != MAROON)
        assert np.any(pat_mask), "missing pattern sample"
        rows, cols = np.where(pat_mask)
        pr0, pr1 = int(rows.min()), int(rows.max())
        pc0, pc1 = int(cols.min()), int(cols.max())
        mapping[count] = band[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()
    assert mapping, "empty legend mapping"
    return mapping


def _ring_components(grid: np.ndarray) -> list[dict]:
    rings: list[dict] = []
    for color in np.unique(grid):
        col = int(color)
        if col == BG:
            continue
        for comp in _find_color_components(grid, col):
            r0, c0, r1, c1 = _bbox_from_coords(comp)
            hh = r1 - r0 + 1
            ww = c1 - c0 + 1
            if hh < 3 or ww < 3:
                continue
            local = np.zeros((hh, ww), dtype=bool)
            for r, c in comp:
                local[r - r0, c - c0] = True
            if np.array_equal(local, _border_mask(hh, ww)):
                rings.append({"color": col, "bbox": (r0, c0, r1, c1), "area": hh * ww})
    return rings


def _tile_pattern(height: int, width: int, pattern: np.ndarray) -> np.ndarray:
    ph, pw = pattern.shape
    rep_h = math.ceil(height / ph)
    rep_w = math.ceil(width / pw)
    tiled = np.tile(pattern, (rep_h, rep_w))
    return tiled[:height, :width]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select a fill pattern from the legend by nested-frame depth and tile it
    into each structure's innermost empty rectangle. Remove the legend area.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    main_grid, legend = _extract_legend_and_main(g)
    mapping = _parse_legend(legend)
    rings = _ring_components(main_grid)
    assert rings, "no frame rings found"

    parents = [None] * len(rings)
    children: list[list[int]] = [[] for _ in rings]
    for i, ring in enumerate(rings):
        container = None
        container_area = None
        for j, other in enumerate(rings):
            if i == j:
                continue
            if _rect_contains(other["bbox"], ring["bbox"]):
                if container_area is None or other["area"] < container_area:
                    container = j
                    container_area = other["area"]
        parents[i] = container
        if container is not None:
            children[container].append(i)

    out = main_grid.copy()
    roots = [idx for idx, parent in enumerate(parents) if parent is None]
    assert roots, "no top-level frame structures found"
    for root in roots:
        chain = [root]
        node = root
        while children[node]:
            child = min(children[node], key=lambda idx: rings[idx]["area"])
            chain.append(child)
            node = child
        depth = len(chain)
        assert depth in mapping, "legend missing required depth"
        r0, c0, r1, c1 = rings[chain[-1]]["bbox"]
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        assert inner_h > 0 and inner_w > 0
        patch = _tile_pattern(inner_h, inner_w, mapping[depth])
        out[r0 + 1 : r1, c0 + 1 : c1] = patch
    return out


def _make_pattern(rng: np.random.Generator) -> np.ndarray:
    height = int(rng.integers(2, 5))
    width = int(rng.integers(2, 5))
    colors = rng.choice(COLOR_POOL, size=int(rng.integers(2, 4)), replace=False)
    a = int(colors[0])
    b = int(colors[1])
    family = int(rng.integers(0, 6))
    pattern = np.full((height, width), a, dtype=int)

    if family == 0:
        for r in range(height):
            for c in range(width):
                pattern[r, c] = a if (r + c) % 2 == 0 else b
    elif family == 1:
        pattern[:, ::2] = a
        pattern[:, 1::2] = b
    elif family == 2:
        pattern[::2, :] = a
        pattern[1::2, :] = b
    elif family == 3:
        pattern[:, :] = a
        pattern[height // 2, :] = b
        pattern[:, width // 2] = b
    elif family == 4:
        pattern[:, :] = a
        pattern[0, :] = b
        pattern[-1, :] = b
        pattern[:, 0] = b
        pattern[:, -1] = b
    else:
        c = int(colors[2]) if len(colors) > 2 else a
        pattern[:, :] = a
        pattern[: math.ceil(height / 2), : math.ceil(width / 2)] = b
        pattern[math.floor(height / 2) :, math.floor(width / 2) :] = c
    return pattern


def _draw_structure(grid: np.ndarray, top: int, left: int, inner_h: int, inner_w: int, depth: int, colors: np.ndarray) -> tuple[int, int, int, int]:
    total_h = inner_h + 4 * depth - 2
    total_w = inner_w + 4 * depth - 2
    for layer in range(depth):
        r0 = top + 2 * layer
        c0 = left + 2 * layer
        r1 = top + total_h - 1 - 2 * layer
        c1 = left + total_w - 1 - 2 * layer
        color = int(colors[layer])
        grid[r0, c0 : c1 + 1] = color
        grid[r1, c0 : c1 + 1] = color
        grid[r0 : r1 + 1, c0] = color
        grid[r0 : r1 + 1, c1] = color
    inner_top = top + 2 * depth - 1
    inner_left = left + 2 * depth - 1
    return inner_top, inner_left, total_h, total_w


def _layout_specs(
    rng: np.random.Generator,
    specs: list[dict],
    side: str,
    legend_min_h: int,
    legend_min_w: int,
) -> tuple[int, int, list[tuple[int, int]]] | None:
    n = len(specs)
    for _ in range(80):
        row_count = int(rng.integers(1, min(3, n) + 1))
        if n >= 4 and row_count == 1:
            row_count = 2
        cuts = sorted(rng.choice(np.arange(1, n), size=row_count - 1, replace=False).tolist()) if row_count > 1 else []
        order = [int(x) for x in rng.permutation(np.arange(n))]
        chunks: list[list[int]] = []
        start = 0
        for cut in cuts + [n]:
            chunks.append(order[start:cut])
            start = cut

        row_gap = int(rng.integers(2, 4))
        col_gap = int(rng.integers(2, 4))
        row_heights = [max(specs[idx]["total_h"] for idx in chunk) for chunk in chunks]
        row_widths = [sum(specs[idx]["total_w"] for idx in chunk) + col_gap * (len(chunk) - 1) for chunk in chunks]

        top_margin = int(rng.integers(1, 4))
        bottom_margin = int(rng.integers(1, 4))
        left_margin = int(rng.integers(1, 4))
        right_margin = int(rng.integers(1, 4))
        main_h = top_margin + bottom_margin + sum(row_heights) + row_gap * (row_count - 1)
        main_w = left_margin + right_margin + max(row_widths)

        if side == "right" and main_h < legend_min_h:
            main_h = legend_min_h + int(rng.integers(0, 3))
        if side == "bottom" and main_w < legend_min_w:
            main_w = legend_min_w + int(rng.integers(0, 3))
        if main_h > 30 or main_w > 30:
            continue

        placements: list[tuple[int, int] | None] = [None] * n
        row_top = top_margin
        for chunk, row_h, row_w in zip(chunks, row_heights, row_widths):
            extra_x = main_w - left_margin - right_margin - row_w
            row_left = left_margin + int(rng.integers(0, extra_x + 1))
            col = row_left
            for idx in chunk:
                spec = specs[idx]
                v_shift = int(rng.integers(0, row_h - spec["total_h"] + 1))
                placements[idx] = (row_top + v_shift, col)
                col += spec["total_w"] + col_gap
            row_top += row_h + row_gap

        return main_h, main_w, [(int(r), int(c)) for r, c in placements]  # type: ignore[arg-type]
    return None


def _build_legend(
    rng: np.random.Generator,
    side: str,
    main_h: int,
    main_w: int,
    patterns: dict[int, np.ndarray],
    order: list[int],
) -> np.ndarray | None:
    band_heights = {depth: max(depth, patterns[depth].shape[0]) for depth in patterns}
    gap_sizes = [int(rng.integers(1, 3)) for _ in range(len(order) - 1)]
    content_h = sum(band_heights[d] for d in order) + sum(gap_sizes)
    max_band_w = max(patterns[d].shape[1] for d in patterns)
    content_w = 2 + max_band_w

    if side == "right":
        if main_h < content_h + 2:
            return None
        legend_h = main_h
        legend_w = max(content_w + 2 + int(rng.integers(0, 3)), 7)
    else:
        if main_w < content_w + 2:
            return None
        legend_h = content_h + 2 + int(rng.integers(0, 3))
        legend_w = main_w

    legend = np.full((legend_h, legend_w), BG, dtype=int)
    legend[0, :] = GREY
    legend[-1, :] = GREY
    legend[:, 0] = GREY
    legend[:, -1] = GREY

    inner = legend[1:-1, 1:-1]
    if inner.shape[0] < content_h or inner.shape[1] < content_w:
        return None

    top_pad = int(rng.integers(0, inner.shape[0] - content_h + 1))
    row = top_pad
    for idx, depth in enumerate(order):
        band_h = band_heights[depth]
        band = inner[row : row + band_h]
        maroon_top = int(rng.integers(0, band_h - depth + 1))
        band[maroon_top : maroon_top + depth, 0] = MAROON
        pattern = patterns[depth]
        ph, pw = pattern.shape
        pr = int(rng.integers(0, band_h - ph + 1))
        pc = int(rng.integers(2, inner.shape[1] - pw + 1))
        band[pr : pr + ph, pc : pc + pw] = pattern
        row += band_h
        if idx < len(gap_sizes):
            row += gap_sizes[idx]
    return legend


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    patterns: dict[int, np.ndarray] = {}
    signatures = set()
    for depth in (1, 2, 3):
        for _ in range(30):
            pattern = _make_pattern(rng)
            sig = (pattern.shape, tuple(pattern.flatten().tolist()))
            if sig not in signatures:
                signatures.add(sig)
                patterns[depth] = pattern
                break
        if depth not in patterns:
            return None

    side = "right" if int(rng.integers(0, 2)) == 0 else "bottom"
    object_count = int(rng.integers(3, 5))
    depths = [1, 2, 3] + [int(rng.integers(1, 4)) for _ in range(object_count - 3)]
    rng.shuffle(depths)
    order = [int(x) for x in rng.permutation(np.array([1, 2, 3], dtype=int))]

    specs: list[dict] = []
    for depth in depths:
        inner_max = 6 - depth
        inner_h = int(rng.integers(2, inner_max + 1))
        inner_w = int(rng.integers(2, inner_max + 1))
        colors = rng.choice(COLOR_POOL, size=depth, replace=False)
        specs.append(
            {
                "depth": depth,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "colors": colors,
                "total_h": inner_h + 4 * depth - 2,
                "total_w": inner_w + 4 * depth - 2,
            }
        )

    legend_min_h = sum(max(depth, patterns[depth].shape[0]) for depth in (1, 2, 3)) + 2 + 2
    legend_min_w = 4 + max(patterns[depth].shape[1] for depth in (1, 2, 3))

    for _ in range(60):
        layout = _layout_specs(rng, specs, side, legend_min_h, legend_min_w)
        if layout is None:
            continue
        main_h, main_w, placements = layout

        main_grid = np.full((main_h, main_w), BG, dtype=int)
        fill_jobs: list[tuple[int, int, int, int, int]] = []
        for spec, (top, left) in zip(specs, placements):
            inner_top, inner_left, _total_h, _total_w = _draw_structure(
                main_grid,
                top,
                left,
                spec["inner_h"],
                spec["inner_w"],
                spec["depth"],
                spec["colors"],
            )
            fill_jobs.append((inner_top, inner_left, spec["inner_h"], spec["inner_w"], spec["depth"]))

        expected = main_grid.copy()
        for inner_top, inner_left, inner_h, inner_w, depth in fill_jobs:
            patch = _tile_pattern(inner_h, inner_w, patterns[depth])
            expected[inner_top : inner_top + inner_h, inner_left : inner_left + inner_w] = patch

        legend = _build_legend(rng, side, main_h, main_w, patterns, order)
        if legend is None:
            continue

        if side == "right":
            grid = np.concatenate([main_grid, legend], axis=1)
        else:
            grid = np.concatenate([main_grid, legend], axis=0)

        try:
            produced = main(grid)
        except Exception:
            continue
        if np.array_equal(produced, expected):
            return grid
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
