from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
GREY = 5
ORANGE = 7
VOTE_COLORS = (1, 2, 3, 4, 6, 8, 9)
MAX_GENERATION_ATTEMPTS = 500


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bands_from_lines(lines: list[int]) -> list[tuple[int, int]] | None:
    bands: list[tuple[int, int]] = []
    for a, b in zip(lines, lines[1:], strict=False):
        if int(b) - int(a) <= 1:
            return None
        bands.append((int(a + 1), int(b - 1)))
    return bands if bands else None


def _template_info(grid: np.ndarray) -> tuple[tuple[int, int, int, int], list[tuple[int, int]], list[tuple[int, int]]]:
    best = None
    for comp in _components(grid == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        row_lines = [r for r in range(patch.shape[0]) if np.all(patch[r, :] == GREY)]
        col_lines = [c for c in range(patch.shape[1]) if np.all(patch[:, c] == GREY)]
        if len(row_lines) < 2 or len(col_lines) < 2:
            continue
        if row_lines[0] != 0 or row_lines[-1] != patch.shape[0] - 1:
            continue
        if col_lines[0] != 0 or col_lines[-1] != patch.shape[1] - 1:
            continue
        row_bands = _bands_from_lines(row_lines)
        col_bands = _bands_from_lines(col_lines)
        if row_bands is None or col_bands is None:
            continue

        expected = np.zeros_like(patch, dtype=bool)
        expected[row_lines, :] = True
        expected[:, col_lines] = True
        if not np.array_equal(patch == GREY, expected):
            continue

        score = (len(row_bands) * len(col_bands), patch.size, int(comp.shape[0]))
        cand = (score, (r0, c0, r1, c1), row_bands, col_bands)
        if best is None or cand[0] > best[0]:
            best = cand

    assert best is not None, "expected a grey grid template"
    _score, bbox_info, row_bands, col_bands = best
    return bbox_info, row_bands, col_bands


def _region_components(grid: np.ndarray, template_bbox: tuple[int, int, int, int]) -> list[np.ndarray]:
    r0, c0, r1, c1 = template_bbox
    keep = grid != ORANGE
    keep[r0 : r1 + 1, c0 : c1 + 1] = False
    return _components(keep)


def _axis_labels(values: list[float], groups: int, items_per_group: int) -> list[int]:
    n = len(values)
    assert n == groups * items_per_group
    order = list(np.argsort(np.asarray(values, dtype=float)))
    labels = [-1] * n
    for gi in range(groups):
        block = order[gi * items_per_group : (gi + 1) * items_per_group]
        assert block
        for idx in block:
            labels[int(idx)] = int(gi)
    return labels


def _region_layout(components: list[np.ndarray], n_rows: int, n_cols: int) -> list[list[int]]:
    row_centers = [float(comp[:, 0].mean()) for comp in components]
    col_centers = [float(comp[:, 1].mean()) for comp in components]
    row_labels = _axis_labels(row_centers, n_rows, n_cols)
    col_labels = _axis_labels(col_centers, n_cols, n_rows)

    layout = [[-1 for _ in range(n_cols)] for _ in range(n_rows)]
    for idx, (ri, ci) in enumerate(zip(row_labels, col_labels, strict=False)):
        assert layout[ri][ci] == -1, "region centers must map bijectively to a row/col matrix"
        layout[ri][ci] = int(idx)

    for row in layout:
        assert all(idx >= 0 for idx in row)
    return layout


def _majority_color(grid: np.ndarray, comp: np.ndarray) -> int:
    votes = Counter(
        int(grid[r, c])
        for r, c in comp
        if int(grid[r, c]) not in {BG, GREY, ORANGE}
    )
    assert votes, "each fenced region must contain vote pixels"
    best = votes.most_common()
    assert len(best) == 1 or best[0][1] != best[1][1], "majority vote must be unique"
    return int(best[0][0])


def _template_patch(n_rows: int, n_cols: int, cell_size: int) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    h = n_rows * cell_size + (n_rows + 1)
    w = n_cols * cell_size + (n_cols + 1)
    patch = np.full((h, w), BG, dtype=int)

    row_lines = [i * (cell_size + 1) for i in range(n_rows + 1)]
    col_lines = [i * (cell_size + 1) for i in range(n_cols + 1)]
    patch[row_lines, :] = GREY
    patch[:, col_lines] = GREY

    row_bands = [(row_lines[i] + 1, row_lines[i + 1] - 1) for i in range(n_rows)]
    col_bands = [(col_lines[i] + 1, col_lines[i + 1] - 1) for i in range(n_cols)]
    return patch, row_bands, col_bands


def _filled_template(n_rows: int, n_cols: int, cell_size: int, colors: np.ndarray) -> np.ndarray:
    patch, row_bands, col_bands = _template_patch(n_rows, n_cols, cell_size)
    out = patch.copy()
    for ri, (r0, r1) in enumerate(row_bands):
        for ci, (c0, c1) in enumerate(col_bands):
            out[r0 : r1 + 1, c0 : c1 + 1] = int(colors[ri, ci])
    return out


def _bands_and_separators(sizes: list[int]) -> tuple[list[tuple[int, int]], list[int], int]:
    bands: list[tuple[int, int]] = []
    seps: list[int] = []
    cursor = 0
    for i, size in enumerate(sizes):
        bands.append((int(cursor), int(cursor + size - 1)))
        cursor += int(size)
        if i + 1 < len(sizes):
            seps.append(int(cursor))
            cursor += 1
    return bands, seps, int(cursor)


def _sample_sizes(
    count: int,
    host_idx: int,
    host_min: int,
    total_limit: int,
    *,
    other_min: int = 4,
    other_max: int = 6,
    host_extra: int = 4,
) -> list[int] | None:
    mins = [other_min] * count
    maxs = [other_max] * count
    mins[host_idx] = int(host_min)
    maxs[host_idx] = int(host_min + host_extra)
    if sum(mins) > total_limit:
        return None

    sizes = mins[:]
    capacity = [maxs[i] - mins[i] for i in range(count)]
    extra_budget = min(sum(capacity), int(total_limit - sum(mins)))
    extra = random.randint(0, extra_budget)
    while extra > 0:
        choices = [i for i, cap in enumerate(capacity) if cap > 0]
        if not choices:
            break
        idx = random.choice(choices)
        sizes[idx] += 1
        capacity[idx] -= 1
        extra -= 1
    return [int(v) for v in sizes]


def _corner_bbox(h: int, w: int, ph: int, pw: int, corner: str) -> tuple[int, int, int, int]:
    if corner == "tl":
        r0, c0 = 0, 0
    elif corner == "tr":
        r0, c0 = 0, w - pw
    elif corner == "bl":
        r0, c0 = h - ph, 0
    else:
        assert corner == "br"
        r0, c0 = h - ph, w - pw
    return int(r0), int(c0), int(r0 + ph - 1), int(c0 + pw - 1)


def _deform_vertical(grid: np.ndarray, row_bands: list[tuple[int, int]], sep_cols: list[int]) -> None:
    for c in sep_cols:
        for r0, r1 in row_bands:
            if r1 - r0 + 1 < 4 or random.random() < 0.45:
                continue
            a = random.randint(r0 + 1, r1 - 2)
            b = random.randint(a + 2, r1)
            dc = random.choice((-1, 1))
            grid[a + 1 : b, c] = BG
            grid[a + 1 : b, c + dc] = ORANGE
            grid[a, c + dc] = ORANGE
            grid[b, c + dc] = ORANGE


def _deform_horizontal(grid: np.ndarray, col_bands: list[tuple[int, int]], sep_rows: list[int]) -> None:
    for r in sep_rows:
        for c0, c1 in col_bands:
            if c1 - c0 + 1 < 4 or random.random() < 0.45:
                continue
            a = random.randint(c0 + 1, c1 - 2)
            b = random.randint(a + 2, c1)
            dr = random.choice((-1, 1))
            grid[r, a + 1 : b] = BG
            grid[r + dr, a + 1 : b] = ORANGE
            grid[r + dr, a] = ORANGE
            grid[r + dr, b] = ORANGE


def _vote_plan(area: int) -> tuple[list[int], list[int]] | None:
    n_colors = random.randint(2, min(4, len(VOTE_COLORS)))
    colors = random.sample(list(VOTE_COLORS), n_colors)
    max_major = min(6, area - (n_colors - 1))
    if max_major < 2:
        return None
    major_count = random.randint(2, max_major)
    counts = [major_count] + [1] * (n_colors - 1)

    slack = min(area - sum(counts), 5)
    minor_caps = [max(0, major_count - 2) for _ in range(n_colors - 1)]
    while slack > 0 and any(cap > 0 for cap in minor_caps):
        idx = random.randrange(n_colors - 1)
        if minor_caps[idx] == 0:
            continue
        counts[idx + 1] += 1
        minor_caps[idx] -= 1
        slack -= 1
    return [int(c) for c in colors], [int(n) for n in counts]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Collapse orange-fenced regions into a filled miniature vote map inside the grey template.

    Input structure:
    - The background is black.
    - Orange fence pixels form a connected 1-pixel wall network that partitions the grid into closed regions, sometimes using the outer grid edge as part of the boundary.
    - Each region contains scattered single-pixel votes in non-black, non-orange colors.
    - A separate grey rectangular grid template sits in one corner; its row/column count matches the spatial arrangement of the fenced regions.

    Rule:
    - Detect the grey template and read its number of rows and columns from the separator lines.
    - Ignore the template area, then split the remaining non-orange cells into fenced regions.
    - For each region, count the vote colors and take the unique majority color.
    - Order the regions into a row/column matrix by their centers, then fill the corresponding template cell with that region's majority color while keeping the grey separator lines.

    Key insight:
    - The orange structure encodes a coarse region layout, while the grey corner object is only an empty output stencil; each stencil cell should inherit the majority vote from its matching fenced region.
    """
    g = np.asarray(grid, dtype=int)
    template_bbox, row_bands, col_bands = _template_info(g)
    comps = _region_components(g, template_bbox)
    layout = _region_layout(comps, len(row_bands), len(col_bands))

    colors = np.zeros((len(row_bands), len(col_bands)), dtype=int)
    for ri in range(len(row_bands)):
        for ci in range(len(col_bands)):
            colors[ri, ci] = _majority_color(g, comps[layout[ri][ci]])

    r0, c0, r1, c1 = template_bbox
    out = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    for ri, (rb0, rb1) in enumerate(row_bands):
        for ci, (cb0, cb1) in enumerate(col_bands):
            out[rb0 : rb1 + 1, cb0 : cb1 + 1] = int(colors[ri, ci])
    return out


def _gen_attempt() -> np.ndarray | None:
    n_rows = random.randint(2, 4)
    n_cols = random.randint(2, 4)
    cell_size = random.choice((1, 2))
    corner = random.choice(("tl", "tr", "bl", "br"))

    template, _tpl_row_bands, _tpl_col_bands = _template_patch(n_rows, n_cols, cell_size)
    tpl_h, tpl_w = map(int, template.shape)

    host_row = 0 if corner.startswith("t") else n_rows - 1
    host_col = 0 if corner.endswith("l") else n_cols - 1

    row_sizes = _sample_sizes(n_rows, host_row, tpl_h + 2, 30 - (n_rows - 1))
    col_sizes = _sample_sizes(n_cols, host_col, tpl_w + 2, 30 - (n_cols - 1))
    if row_sizes is None or col_sizes is None:
        return None

    row_bands, sep_rows, height = _bands_and_separators(row_sizes)
    col_bands, sep_cols, width = _bands_and_separators(col_sizes)
    if height > 30 or width > 30:
        return None

    grid = np.full((height, width), BG, dtype=int)
    if sep_rows:
        grid[sep_rows, :] = ORANGE
    if sep_cols:
        grid[:, sep_cols] = ORANGE

    dent_mode = random.choice(("none", "vertical", "horizontal"))
    if dent_mode == "vertical":
        _deform_vertical(grid, row_bands, sep_cols)
    elif dent_mode == "horizontal":
        _deform_horizontal(grid, col_bands, sep_rows)

    template_bbox = _corner_bbox(height, width, tpl_h, tpl_w, corner)
    r0, c0, r1, c1 = template_bbox
    if np.any(grid[r0 : r1 + 1, c0 : c1 + 1] == ORANGE):
        return None
    grid[r0 : r1 + 1, c0 : c1 + 1][template == GREY] = GREY

    comps = _region_components(grid, template_bbox)
    if len(comps) != n_rows * n_cols:
        return None
    layout = _region_layout(comps, n_rows, n_cols)

    majority = np.zeros((n_rows, n_cols), dtype=int)
    for ri in range(n_rows):
        for ci in range(n_cols):
            comp = comps[layout[ri][ci]]
            area = int(comp.shape[0])
            if area < 6:
                return None
            plan = _vote_plan(area)
            if plan is None:
                return None
            colors, counts = plan
            total_votes = sum(counts)
            if total_votes > area:
                return None
            pts = [tuple(map(int, p)) for p in comp.tolist()]
            random.shuffle(pts)
            chosen = pts[:total_votes]
            cursor = 0
            for color, count in zip(colors, counts, strict=False):
                for _ in range(count):
                    rr, cc = chosen[cursor]
                    grid[rr, cc] = int(color)
                    cursor += 1
            majority[ri, ci] = int(colors[0])

    expected = _filled_template(n_rows, n_cols, cell_size, majority)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fenced-vote puzzle")
