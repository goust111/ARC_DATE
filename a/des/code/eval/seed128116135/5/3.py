from __future__ import annotations

import random
from typing import Any

import numpy as np


BLACK = 0
RED = 2
GREY = 5
TEXTURE_PALETTE = (1, 3, 4, 6, 7, 8, 9)
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_SHAPES = ((2, 2), (2, 3), (3, 2))
REGION_SIZE_OPTIONS = (5, 7, 9)
PATTERN_KINDS = ("h2", "h3", "v2", "v3", "checker", "lattice", "dots3", "diag3")
GENERATE_RETRIES = 400


def _bands(size: int, separators: list[int]) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start = 0
    for sep in separators + [size]:
        assert sep > start, "separator layout must leave a non-empty region band"
        bands.append((start, sep - 1))
        start = sep + 1
    return bands


def _extract_regions(
    grid: np.ndarray, row_bands: list[tuple[int, int]], col_bands: list[tuple[int, int]]
) -> list[list[dict[str, Any]]]:
    regions: list[list[dict[str, Any]]] = []
    for rb, (r0, r1) in enumerate(row_bands):
        row_regions: list[dict[str, Any]] = []
        for cb, (c0, c1) in enumerate(col_bands):
            sub = grid[r0 : r1 + 1, c0 : c1 + 1]
            assert not np.any(sub == GREY), "separator color may not appear inside a region"
            texture_colors = np.unique(sub[(sub != BLACK) & (sub != GREY) & (sub != RED)])
            assert texture_colors.size == 1, "each region must have exactly one texture color"
            marker_coords = np.argwhere(sub == RED)
            assert marker_coords.shape[0] <= 1, "each region may contain at most one red marker"
            marker = None
            if marker_coords.shape[0] == 1:
                marker = (int(r0 + marker_coords[0, 0]), int(c0 + marker_coords[0, 1]))
            row_regions.append(
                {
                    "row": rb,
                    "col": cb,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(sub.size),
                    "texture_color": int(texture_colors[0]),
                    "numerator": int(np.count_nonzero((sub != BLACK) & (sub != GREY))),
                    "marker": marker,
                    "center": ((r0 + r1) // 2, (c0 + c1) // 2),
                }
            )
        regions.append(row_regions)
    return regions


def _compare_density(a: dict[str, Any], b: dict[str, Any]) -> int:
    left = int(a["numerator"]) * int(b["area"])
    right = int(b["numerator"]) * int(a["area"])
    if left > right:
        return 1
    if left < right:
        return -1
    return 0


def _boundary_stage(
    grid: np.ndarray,
    regions: list[list[dict[str, Any]]],
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
    sep_rows: list[int],
    sep_cols: list[int],
) -> tuple[np.ndarray, int, int]:
    stage = grid.copy()
    tie_count = 0
    colored_count = 0

    for edge_row, sep_row in enumerate(sep_rows):
        for col_idx, (c0, c1) in enumerate(col_bands):
            upper = regions[edge_row][col_idx]
            lower = regions[edge_row + 1][col_idx]
            cmp = _compare_density(upper, lower)
            if cmp > 0:
                stage[sep_row, c0 : c1 + 1] = int(upper["texture_color"])
                colored_count += 1
            elif cmp < 0:
                stage[sep_row, c0 : c1 + 1] = int(lower["texture_color"])
                colored_count += 1
            else:
                stage[sep_row, c0 : c1 + 1] = GREY
                tie_count += 1

    for row_idx, (r0, r1) in enumerate(row_bands):
        for edge_col, sep_col in enumerate(sep_cols):
            left = regions[row_idx][edge_col]
            right = regions[row_idx][edge_col + 1]
            cmp = _compare_density(left, right)
            if cmp > 0:
                stage[r0 : r1 + 1, sep_col] = int(left["texture_color"])
                colored_count += 1
            elif cmp < 0:
                stage[r0 : r1 + 1, sep_col] = int(right["texture_color"])
                colored_count += 1
            else:
                stage[r0 : r1 + 1, sep_col] = GREY
                tie_count += 1

    return stage, tie_count, colored_count


def _paint_rays(
    original: np.ndarray,
    stage: np.ndarray,
    regions: list[list[dict[str, Any]]],
    sep_rows: list[int],
    sep_cols: list[int],
) -> np.ndarray:
    out = stage.copy()
    sep_row_set = set(sep_rows)
    sep_col_set = set(sep_cols)
    height, width = original.shape

    for row_regions in regions:
        for region in row_regions:
            if region["marker"] is None:
                continue
            color = int(region["texture_color"])
            mr, mc = region["marker"]
            for dr, dc in DIRS:
                r, c = mr, mc
                while True:
                    r += dr
                    c += dc
                    if not (0 <= r < height and 0 <= c < width):
                        break
                    if (r in sep_row_set or c in sep_col_set) and int(stage[r, c]) == color:
                        break
                    if int(original[r, c]) in (BLACK, GREY):
                        out[r, c] = color
    return out


def _solve(grid: np.ndarray, return_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_rows = [int(r) for r in range(g.shape[0]) if np.all(g[r, :] == GREY)]
    sep_cols = [int(c) for c in range(g.shape[1]) if np.all(g[:, c] == GREY)]
    row_bands = _bands(int(g.shape[0]), sep_rows)
    col_bands = _bands(int(g.shape[1]), sep_cols)
    regions = _extract_regions(g, row_bands, col_bands)
    stage, tie_count, colored_count = _boundary_stage(g, regions, row_bands, col_bands, sep_rows, sep_cols)
    out = _paint_rays(g, stage, regions, sep_rows, sep_cols)

    if return_meta:
        return out, stage, regions, row_bands, col_bands, sep_rows, sep_cols, tie_count, colored_count
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each separator segment by the denser adjacent region's texture color,
    then cast texture-colored rays outward from red markers until a same-colored
    boundary segment blocks them.
    """
    return _solve(grid)


def _pattern_mask(height: int, width: int, kind: str) -> np.ndarray:
    rr, cc = np.indices((height, width))
    cr = height // 2
    cm = width // 2
    dr = rr - cr
    dc = cc - cm

    if kind == "h2":
        mask = np.mod(dr, 2) == 0
    elif kind == "h3":
        mask = np.mod(dr, 3) == 0
    elif kind == "v2":
        mask = np.mod(dc, 2) == 0
    elif kind == "v3":
        mask = np.mod(dc, 3) == 0
    elif kind == "checker":
        mask = np.mod(dr + dc, 2) == 0
    elif kind == "lattice":
        mask = (np.mod(dr, 2) == 0) & (np.mod(dc, 2) == 0)
    elif kind == "dots3":
        mask = np.mod(2 * dr + dc, 3) == 0
    else:
        assert kind == "diag3"
        mask = np.mod(dr - dc, 3) == 0

    assert bool(mask[cr, cm]), "pattern must cover the region center"
    assert 0 < int(mask.sum()) < height * width
    return mask


def _adjacency_edges(n_rows: int, n_cols: int) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for r in range(n_rows):
        for c in range(n_cols - 1):
            edges.append(
                {
                    "orientation": "vertical",
                    "a": (r, c),
                    "b": (r, c + 1),
                    "dir_a": (0, 1),
                    "dir_b": (0, -1),
                }
            )
    for r in range(n_rows - 1):
        for c in range(n_cols):
            edges.append(
                {
                    "orientation": "horizontal",
                    "a": (r, c),
                    "b": (r + 1, c),
                    "dir_a": (1, 0),
                    "dir_b": (-1, 0),
                }
            )
    return edges


def _ray_first_event(
    stage: np.ndarray,
    region: dict[str, Any],
    direction: tuple[int, int],
    sep_rows: list[int],
    sep_cols: list[int],
) -> str:
    r, c = map(int, region["center"])
    dr, dc = direction
    sep_row_set = set(sep_rows)
    sep_col_set = set(sep_cols)
    height, width = stage.shape

    while True:
        r += dr
        c += dc
        if not (0 <= r < height and 0 <= c < width):
            return "edge"
        if r in sep_row_set or c in sep_col_set:
            if int(stage[r, c]) == int(region["texture_color"]):
                return "stop"
            return "pass"


def _build_attempt() -> np.ndarray | None:
    n_rows, n_cols = random.choice(GRID_SHAPES)
    heights = [random.choice(REGION_SIZE_OPTIONS) for _ in range(n_rows)]
    widths = [random.choice(REGION_SIZE_OPTIONS) for _ in range(n_cols)]
    total_regions = n_rows * n_cols
    colors = random.sample(TEXTURE_PALETTE, total_regions)

    edges = _adjacency_edges(n_rows, n_cols)
    unequal_edge = random.choice(edges)
    unequal_cells = {unequal_edge["a"], unequal_edge["b"]}
    tie_candidates = [edge for edge in edges if {edge["a"], edge["b"]}.isdisjoint(unequal_cells)]
    if not tie_candidates:
        return None
    tie_edge = random.choice(tie_candidates)

    region_patterns = {
        (r, c): random.choice(PATTERN_KINDS) for r in range(n_rows) for c in range(n_cols)
    }
    if tie_edge["orientation"] == "vertical":
        tie_kind = random.choice(("h2", "h3"))
    else:
        tie_kind = random.choice(("v2", "v3"))
    region_patterns[tie_edge["a"]] = tie_kind
    region_patterns[tie_edge["b"]] = tie_kind

    if unequal_edge["orientation"] == "vertical":
        high_kind, low_kind = "h2", "h3"
    else:
        high_kind, low_kind = "v2", "v3"
    if random.random() < 0.5:
        region_patterns[unequal_edge["a"]] = high_kind
        region_patterns[unequal_edge["b"]] = low_kind
    else:
        region_patterns[unequal_edge["a"]] = low_kind
        region_patterns[unequal_edge["b"]] = high_kind

    grid_h = sum(heights) + (n_rows - 1)
    grid_w = sum(widths) + (n_cols - 1)
    grid = np.full((grid_h, grid_w), BLACK, dtype=int)

    sep_rows: list[int] = []
    sep_cols: list[int] = []
    row_starts: list[int] = []
    col_starts: list[int] = []

    pos = 0
    for i, height in enumerate(heights):
        row_starts.append(pos)
        pos += height
        if i < n_rows - 1:
            sep_rows.append(pos)
            grid[pos, :] = GREY
            pos += 1

    pos = 0
    for i, width in enumerate(widths):
        col_starts.append(pos)
        pos += width
        if i < n_cols - 1:
            sep_cols.append(pos)
            grid[:, pos] = GREY
            pos += 1

    for r in range(n_rows):
        for c in range(n_cols):
            r0 = row_starts[r]
            c0 = col_starts[c]
            height = heights[r]
            width = widths[c]
            color = colors[r * n_cols + c]
            mask = _pattern_mask(height, width, region_patterns[(r, c)])
            window = grid[r0 : r0 + height, c0 : c0 + width]
            window[mask] = color

    _, stage, regions, _row_bands, _col_bands, parsed_sep_rows, parsed_sep_cols, tie_count, colored_count = _solve(
        grid, return_meta=True
    )
    if parsed_sep_rows != sep_rows or parsed_sep_cols != sep_cols:
        return None
    if tie_count < 1 or colored_count < 1:
        return None

    tie_a = regions[tie_edge["a"][0]][tie_edge["a"][1]]
    tie_b = regions[tie_edge["b"][0]][tie_edge["b"][1]]
    if _compare_density(tie_a, tie_b) != 0:
        return None

    reg_a = regions[unequal_edge["a"][0]][unequal_edge["a"][1]]
    reg_b = regions[unequal_edge["b"][0]][unequal_edge["b"][1]]
    cmp = _compare_density(reg_a, reg_b)
    if cmp == 0:
        return None
    if cmp > 0:
        dense_region, dense_dir = reg_a, unequal_edge["dir_a"]
        sparse_region, sparse_dir = reg_b, unequal_edge["dir_b"]
    else:
        dense_region, dense_dir = reg_b, unequal_edge["dir_b"]
        sparse_region, sparse_dir = reg_a, unequal_edge["dir_a"]

    if _ray_first_event(stage, dense_region, dense_dir, parsed_sep_rows, parsed_sep_cols) != "stop":
        return None
    if _ray_first_event(stage, sparse_region, sparse_dir, parsed_sep_rows, parsed_sep_cols) != "pass":
        return None

    for region in (dense_region, sparse_region):
        rr, cc = map(int, region["center"])
        grid[rr, cc] = RED

    remaining = [
        regions[r][c]
        for r in range(n_rows)
        for c in range(n_cols)
        if (r, c) not in unequal_cells
    ]
    random.shuffle(remaining)
    extra_markers = random.randint(0, min(2, len(remaining)))
    for region in remaining[:extra_markers]:
        rr, cc = map(int, region["center"])
        grid[rr, cc] = RED

    _out, final_stage, final_regions, _rb, _cb, final_sep_rows, final_sep_cols, final_ties, final_colored = _solve(
        grid, return_meta=True
    )
    if final_ties < 1 or final_colored < 1:
        return None
    dense_final = final_regions[dense_region["row"]][dense_region["col"]]
    sparse_final = final_regions[sparse_region["row"]][sparse_region["col"]]
    if _ray_first_event(final_stage, dense_final, dense_dir, final_sep_rows, final_sep_cols) != "stop":
        return None
    if _ray_first_event(final_stage, sparse_final, sparse_dir, final_sep_rows, final_sep_cols) != "pass":
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {GENERATE_RETRIES} attempts")
