from __future__ import annotations

import random
from collections import Counter

import numpy as np


SIDE_ORDER = ("top", "left", "right", "bottom")
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _separator_color(grid: np.ndarray, bg: int) -> int:
    values, counts = np.unique(grid, return_counts=True)
    non_bg = [(int(v), int(c)) for v, c in zip(values, counts) if int(v) != bg]
    assert non_bg, "expected at least one non-background color"
    fallback = max(non_bg, key=lambda item: item[1])[0]

    h, w = grid.shape
    best_score: tuple[int, int, int, int] | None = None
    best_color = fallback
    for color, count in non_bg:
        comps = _components(grid != color)
        if len(comps) < 2:
            continue

        id_map = np.full((h, w), -1, dtype=int)
        rect_count = 0
        for idx, pts in enumerate(comps):
            for r, c in pts:
                id_map[r, c] = idx
            r0, c0, r1, c1 = _bbox(pts)
            if len(pts) == (r1 - r0 + 1) * (c1 - c0 + 1):
                rect_count += 1

        bridges = 0
        for r, c in np.argwhere(grid == color):
            r = int(r)
            c = int(c)
            if 0 < c < w - 1:
                left = int(id_map[r, c - 1])
                right = int(id_map[r, c + 1])
                if left >= 0 and right >= 0 and left != right:
                    bridges += 1
            if 0 < r < h - 1:
                top = int(id_map[r - 1, c])
                bottom = int(id_map[r + 1, c])
                if top >= 0 and bottom >= 0 and top != bottom:
                    bridges += 1

        score = (int(rect_count == len(comps)), bridges, rect_count, count)
        if best_score is None or score > best_score:
            best_score = score
            best_color = color

    return int(best_color)


def _analyze_regions(
    grid: np.ndarray, bg: int, sep: int
) -> tuple[list[dict[str, object]], dict[int, dict[str, set[int]]]]:
    comps = _components(grid != sep)
    h, w = grid.shape
    id_map = np.full((h, w), -1, dtype=int)
    regions: list[dict[str, object]] = []
    for idx, pts in enumerate(comps):
        for r, c in pts:
            id_map[r, c] = idx
        r0, c0, r1, c1 = _bbox(pts)
        values = [int(grid[r, c]) for r, c in pts if int(grid[r, c]) not in (bg, sep)]
        colors = sorted(set(values))
        assert len(colors) <= 1, "each non-empty region must use a single marker color"
        regions.append(
            {
                "id": idx,
                "bbox": (r0, c0, r1, c1),
                "height": r1 - r0 + 1,
                "width": c1 - c0 + 1,
                "empty": len(values) == 0,
                "marker_color": None if not colors else colors[0],
                "marker_count": len(values),
            }
        )

    adjacency: dict[int, dict[str, set[int]]] = {
        idx: {side: set() for side in SIDE_ORDER} for idx in range(len(regions))
    }
    for r, c in np.argwhere(grid == sep):
        r = int(r)
        c = int(c)
        if 0 < c < w - 1:
            left = int(id_map[r, c - 1])
            right = int(id_map[r, c + 1])
            if left >= 0 and right >= 0 and left != right:
                adjacency[left]["right"].add(right)
                adjacency[right]["left"].add(left)
        if 0 < r < h - 1:
            top = int(id_map[r - 1, c])
            bottom = int(id_map[r + 1, c])
            if top >= 0 and bottom >= 0 and top != bottom:
                adjacency[top]["bottom"].add(bottom)
                adjacency[bottom]["top"].add(top)

    return regions, adjacency


def _neighbor_sort_key(region: dict[str, object], side: str) -> tuple[int, int]:
    r0, c0, _r1, _c1 = region["bbox"]  # type: ignore[index]
    if side in ("top", "bottom"):
        return int(c0), int(r0)
    return int(r0), int(c0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Partition the grid by separator lines, then fill each empty region with
    horizontal supply stripes from its non-empty neighbors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    sep = _separator_color(g, bg)
    regions, adjacency = _analyze_regions(g, bg, sep)

    out = g.copy()
    for region in regions:
        if not region["empty"]:
            continue
        r0, c0, r1, c1 = region["bbox"]  # type: ignore[index]
        height = int(region["height"])
        cursor = 0
        for side in SIDE_ORDER:
            neighbor_ids = sorted(
                adjacency[int(region["id"])][side],
                key=lambda rid: _neighbor_sort_key(regions[rid], side),
            )
            for neighbor_id in neighbor_ids:
                neighbor = regions[neighbor_id]
                if neighbor["empty"]:
                    continue
                color = neighbor["marker_color"]
                thickness = int(neighbor["marker_count"])
                if color is None or thickness <= 0 or cursor >= height:
                    continue
                top = int(r0) + cursor
                bottom = min(int(r1) + 1, top + thickness)
                out[top:bottom, int(c0) : int(c1) + 1] = int(color)
                cursor += thickness
                if cursor >= height:
                    break
            if cursor >= height:
                break

    return out


# --------------- generator ---------------

SEPARATOR_COLOR = 9
ROW_COUNT_RANGE = (2, 4)
COL_COUNT_RANGE = (2, 4)
CELL_HEIGHT_RANGE = (4, 7)
CELL_WIDTH_RANGE = (4, 7)
GRID_HEIGHT_MAX = 24
GRID_WIDTH_MAX = 24
MAX_NONEMPTY_REGIONS = 6
MARKER_COUNT_CHOICES = (1, 1, 2, 2, 3)
GEN_RETRIES = 400


def _sample_axis(rng: random.Random, n_parts: int, size_range: tuple[int, int], total_max: int) -> list[int] | None:
    for _ in range(100):
        sizes = [rng.randint(*size_range) for _ in range(n_parts)]
        if sum(sizes) + (n_parts - 1) <= total_max:
            return sizes
    return None


def _cell_bboxes(
    heights: list[int], widths: list[int]
) -> dict[tuple[int, int], tuple[int, int, int, int]]:
    row_starts = []
    col_starts = []
    pos = 0
    for height in heights:
        row_starts.append(pos)
        pos += height + 1
    pos = 0
    for width in widths:
        col_starts.append(pos)
        pos += width + 1

    bboxes: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for r, r0 in enumerate(row_starts):
        for c, c0 in enumerate(col_starts):
            bboxes[(r, c)] = (r0, c0, r0 + heights[r] - 1, c0 + widths[c] - 1)
    return bboxes


def _neighbor_cells(cell: tuple[int, int], n_rows: int, n_cols: int) -> dict[str, tuple[int, int]]:
    r, c = cell
    out: dict[str, tuple[int, int]] = {}
    if r > 0:
        out["top"] = (r - 1, c)
    if c > 0:
        out["left"] = (r, c - 1)
    if c < n_cols - 1:
        out["right"] = (r, c + 1)
    if r < n_rows - 1:
        out["bottom"] = (r + 1, c)
    return out


def _pick_focus_cell(rng: random.Random, n_rows: int, n_cols: int) -> tuple[int, int]:
    corners = [
        (r, c)
        for r in range(n_rows)
        for c in range(n_cols)
        if r in (0, n_rows - 1) and c in (0, n_cols - 1)
    ]
    edges = [
        (r, c)
        for r in range(n_rows)
        for c in range(n_cols)
        if (r in (0, n_rows - 1) or c in (0, n_cols - 1)) and (r, c) not in corners
    ]
    interior = [
        (r, c)
        for r in range(1, n_rows - 1)
        for c in range(1, n_cols - 1)
    ]

    groups = [corners]
    if edges:
        groups.append(edges)
    if interior:
        groups.append(interior)
    return rng.choice(rng.choice(groups))


def _choose_occupancy(
    rng: random.Random, n_rows: int, n_cols: int
) -> tuple[set[tuple[int, int]], tuple[int, int]] | None:
    focus = _pick_focus_cell(rng, n_rows, n_cols)
    neighbors = _neighbor_cells(focus, n_rows, n_cols)
    if not neighbors:
        return None
    chosen_sides = rng.sample(list(neighbors), rng.randint(1, len(neighbors)))

    nonempty = {neighbors[side] for side in chosen_sides}
    blocked = {cell for side, cell in neighbors.items() if side not in chosen_sides}
    all_cells = [(r, c) for r in range(n_rows) for c in range(n_cols)]
    candidates = [
        cell
        for cell in all_cells
        if cell != focus and cell not in nonempty and cell not in blocked
    ]
    rng.shuffle(candidates)
    extra_cap = min(MAX_NONEMPTY_REGIONS - len(nonempty), len(candidates))
    extra_count = rng.randint(0, extra_cap)
    for cell in candidates[:extra_count]:
        nonempty.add(cell)

    if len(nonempty) > MAX_NONEMPTY_REGIONS:
        return None
    return nonempty, focus


def _assign_marker_counts(
    rng: random.Random,
    nonempty: set[tuple[int, int]],
    bboxes: dict[tuple[int, int], tuple[int, int, int, int]],
    n_rows: int,
    n_cols: int,
) -> dict[tuple[int, int], int] | None:
    all_cells = [(r, c) for r in range(n_rows) for c in range(n_cols)]
    for _ in range(200):
        counts: dict[tuple[int, int], int] = {}
        for cell in nonempty:
            r0, c0, r1, c1 = bboxes[cell]
            max_count = min(3, (r1 - r0 + 1) * (c1 - c0 + 1))
            choices = [v for v in MARKER_COUNT_CHOICES if v <= max_count]
            counts[cell] = rng.choice(choices)

        changed = False
        valid = True
        for cell in all_cells:
            if cell in nonempty:
                continue
            r0, _c0, r1, _c1 = bboxes[cell]
            height = r1 - r0 + 1
            total = sum(counts[nbr] for nbr in _neighbor_cells(cell, n_rows, n_cols).values() if nbr in nonempty)
            if total > height:
                valid = False
                break
            if total > 0:
                changed = True
        if valid and changed:
            return counts
    return None


def _paint_expected_output(
    grid: np.ndarray,
    bboxes: dict[tuple[int, int], tuple[int, int, int, int]],
    nonempty: set[tuple[int, int]],
    colors: dict[tuple[int, int], int],
    counts: dict[tuple[int, int], int],
    n_rows: int,
    n_cols: int,
) -> np.ndarray:
    out = grid.copy()
    for r in range(n_rows):
        for c in range(n_cols):
            cell = (r, c)
            if cell in nonempty:
                continue
            r0, c0, r1, c1 = bboxes[cell]
            cursor = 0
            for side in SIDE_ORDER:
                neighbor = _neighbor_cells(cell, n_rows, n_cols).get(side)
                if neighbor is None or neighbor not in nonempty:
                    continue
                thickness = counts[neighbor]
                out[r0 + cursor : r0 + cursor + thickness, c0 : c1 + 1] = colors[neighbor]
                cursor += thickness
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = rng.choice([color for color in range(10) if color != SEPARATOR_COLOR])
    n_rows = rng.randint(*ROW_COUNT_RANGE)
    n_cols = rng.randint(*COL_COUNT_RANGE)
    heights = _sample_axis(rng, n_rows, CELL_HEIGHT_RANGE, GRID_HEIGHT_MAX)
    widths = _sample_axis(rng, n_cols, CELL_WIDTH_RANGE, GRID_WIDTH_MAX)
    if heights is None or widths is None:
        return None

    occupancy = _choose_occupancy(rng, n_rows, n_cols)
    if occupancy is None:
        return None
    nonempty, _focus = occupancy
    if not nonempty:
        return None

    available_colors = [color for color in range(10) if color not in (bg, SEPARATOR_COLOR)]
    if len(nonempty) > len(available_colors):
        return None

    bboxes = _cell_bboxes(heights, widths)
    counts = _assign_marker_counts(rng, nonempty, bboxes, n_rows, n_cols)
    if counts is None:
        return None

    total_h = sum(heights) + n_rows - 1
    total_w = sum(widths) + n_cols - 1
    grid = np.full((total_h, total_w), bg, dtype=int)

    row_seps = np.cumsum(heights[:-1]) + np.arange(len(heights) - 1)
    col_seps = np.cumsum(widths[:-1]) + np.arange(len(widths) - 1)
    grid[row_seps, :] = SEPARATOR_COLOR
    grid[:, col_seps] = SEPARATOR_COLOR

    colors = {
        cell: color for cell, color in zip(sorted(nonempty), rng.sample(available_colors, len(nonempty)))
    }
    for cell in nonempty:
        r0, c0, r1, c1 = bboxes[cell]
        cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
        markers = rng.sample(cells, counts[cell])
        for r, c in markers:
            grid[r, c] = colors[cell]

    values, freq = np.unique(grid, return_counts=True)
    if int(values[np.argmax(freq)]) != bg:
        return None

    expected = _paint_expected_output(grid, bboxes, nonempty, colors, counts, n_rows, n_cols)
    if np.array_equal(expected, grid):
        return None
    try:
        solved = main(grid)
    except Exception:
        return None
    return grid if np.array_equal(solved, expected) else None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
