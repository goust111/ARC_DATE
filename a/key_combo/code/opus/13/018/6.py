import random
from collections import Counter

import numpy as np


LATTICE_SIZE = 4
PALETTE = (1, 2, 3, 4)
SEP_COLOR = 5
BLOCK_SIZE_RANGE = (2, 5)
GEN_RETRIES = 500
EDGE_TO_CLOCKWISE_RANK = {"top": 0, "right": 1, "bottom": 2, "left": 3}
ORDERED_PAIR_CHOICES = (
    ("top", "right"),
    ("top", "left"),
    ("right", "bottom"),
    ("bottom", "left"),
)


def _runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    out = []
    start = prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        out.append((start, prev))
        start = prev = idx
    out.append((start, prev))
    return out


def _majority_color(block: np.ndarray) -> int:
    return int(Counter(map(int, block.ravel())).most_common(1)[0][0])


def _infer_separator_color(grid: np.ndarray) -> int:
    row_counts: Counter[int] = Counter()
    col_counts: Counter[int] = Counter()
    for r in range(grid.shape[0]):
        row = grid[r]
        if np.all(row == row[0]):
            row_counts[int(row[0])] += 1
    for c in range(grid.shape[1]):
        col = grid[:, c]
        if np.all(col == col[0]):
            col_counts[int(col[0])] += 1
    candidates = [color for color in row_counts if row_counts[color] >= 3 and col_counts[color] >= 3]
    assert len(candidates) == 1, "expected one separator color"
    return int(candidates[0])


def _find_bands(grid: np.ndarray, sep: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = grid.shape
    sep_rows = [r for r in range(h) if np.count_nonzero(grid[r] == sep) >= w - 3]
    sep_cols = [c for c in range(w) if np.count_nonzero(grid[:, c] == sep) >= h - 3]
    row_runs = _runs([r for r in range(h) if r not in set(sep_rows)])
    col_runs = _runs([c for c in range(w) if c not in set(sep_cols)])
    assert len(row_runs) == LATTICE_SIZE and len(col_runs) == LATTICE_SIZE
    row_heights = {r1 - r0 + 1 for r0, r1 in row_runs}
    col_widths = {c1 - c0 + 1 for c0, c1 in col_runs}
    assert len(row_heights) == 1 and len(col_widths) == 1
    return row_runs, col_runs


def _band_index(runs: list[tuple[int, int]], pos: int) -> int:
    for idx, (start, end) in enumerate(runs):
        if start <= pos <= end:
            return idx
    raise AssertionError("marker does not align with a content band")


def _perimeter_index(shape: tuple[int, int], r: int, c: int) -> int:
    h, w = shape
    if r == 0:
        return c
    if c == w - 1:
        return (w - 1) + r
    if r == h - 1:
        return (w - 1) + (h - 1) + (w - 1 - c)
    assert c == 0
    return (w - 1) + (h - 1) + (w - 1) + (h - 1 - r)


def _marker_cells(grid: np.ndarray, sep: int) -> list[tuple[int, int, int]]:
    h, w = grid.shape
    markers = []
    border = set()
    for c in range(w):
        border.add((0, c))
        border.add((h - 1, c))
    for r in range(h):
        border.add((r, 0))
        border.add((r, w - 1))
    for r, c in sorted(border):
        color = int(grid[r, c])
        if color != sep:
            markers.append((r, c, color))
    assert 2 <= len(markers) <= 3
    return sorted(markers, key=lambda rcv: _perimeter_index(grid.shape, rcv[0], rcv[1]))


def _ray_path(edge: str, band: int) -> list[tuple[int, int]]:
    if edge == "top":
        return [(r, band) for r in range(LATTICE_SIZE)]
    if edge == "bottom":
        return [(r, band) for r in range(LATTICE_SIZE - 1, -1, -1)]
    if edge == "left":
        return [(band, c) for c in range(LATTICE_SIZE)]
    assert edge == "right"
    return [(band, c) for c in range(LATTICE_SIZE - 1, -1, -1)]


def _edge_from_border(shape: tuple[int, int], r: int, c: int) -> str:
    h, w = shape
    if r == 0:
        return "top"
    if c == w - 1:
        return "right"
    if r == h - 1:
        return "bottom"
    assert c == 0
    return "left"


def _compress_blocks(grid: np.ndarray, row_runs: list[tuple[int, int]], col_runs: list[tuple[int, int]]) -> np.ndarray:
    out = np.zeros((LATTICE_SIZE, LATTICE_SIZE), dtype=int)
    for i, (r0, r1) in enumerate(row_runs):
        for j, (c0, c1) in enumerate(col_runs):
            out[i, j] = _majority_color(grid[r0 : r1 + 1, c0 : c1 + 1])
    return out


def _simulate_rays(summary: np.ndarray, markers: list[dict]) -> tuple[np.ndarray, list[dict]]:
    out = np.array(summary, dtype=int, copy=True)
    traces: list[dict] = []
    visit_counts: Counter[tuple[int, int]] = Counter()
    for marker in markers:
        color = int(marker["color"])
        path = _ray_path(marker["edge"], int(marker["band"]))
        visited = []
        absorbed = False
        recolored = 0
        for cell in path:
            visited.append(cell)
            visit_counts[cell] += 1
            r, c = cell
            if int(out[r, c]) == color:
                absorbed = True
                break
            out[r, c] = color
            recolored += 1
        traces.append(
            {
                "edge": marker["edge"],
                "band": int(marker["band"]),
                "color": color,
                "visited": visited,
                "absorbed": absorbed,
                "recolored": recolored,
            }
        )
    overlap = sum(1 for count in visit_counts.values() if count >= 2)
    for trace in traces:
        trace["overlap_cells"] = overlap
    return out, traces


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress each separator-defined block to its majority color, then let border markers fire repainting rays into the compressed grid.

    Input structure:
    - A one-cell-wide separator color forms the outer border and the internal lattice lines of a 4x4 block grid.
    - Each block contains only non-separator colors and has one unique majority color.
    - Two or three colored marker pixels sit on the outer border, aligned with block row bands or column bands.

    Rule:
    - Identify the separator rows and columns and compress every block to one cell using its majority color.
    - Build a 6x6 output with a one-cell separator border around that 4x4 summary grid.
    - Process border markers in clockwise order around the input border, starting from the top edge near the top-left corner.
    - Top and bottom markers fire vertical rays; left and right markers fire horizontal rays.
    - Along a ray, repaint each visited summary cell to the marker color until the ray reaches a cell already having that color, where it is absorbed and stops.

    Key insight:
    - The block-compression step creates the small color grid that the rays interact with, so ray depth depends on the compressed colors and on earlier rays that may repaint future absorption cells.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep = _infer_separator_color(g)
    row_runs, col_runs = _find_bands(g, sep)
    summary = _compress_blocks(g, row_runs, col_runs)
    markers = []
    for r, c, color in _marker_cells(g, sep):
        edge = _edge_from_border(g.shape, r, c)
        band = _band_index(col_runs, c) if edge in {"top", "bottom"} else _band_index(row_runs, r)
        markers.append({"edge": edge, "band": band, "color": color})

    painted, _traces = _simulate_rays(summary, markers)
    out = np.full((LATTICE_SIZE + 2, LATTICE_SIZE + 2), sep, dtype=int)
    out[1:-1, 1:-1] = painted
    return out


def _path_to_cross(edge: str, row: int, col: int) -> list[tuple[int, int]]:
    path = _ray_path(edge, col if edge in {"top", "bottom"} else row)
    idx = path.index((row, col))
    return path[: idx + 1]


def _path_after_cross(edge: str, row: int, col: int) -> list[tuple[int, int]]:
    path = _ray_path(edge, col if edge in {"top", "bottom"} else row)
    idx = path.index((row, col))
    return path[idx + 1 :]


def _choose_other_color(color: int) -> int:
    return random.choice([c for c in PALETTE if c != color])


def _initial_summary() -> np.ndarray:
    return np.array([[random.choice(PALETTE) for _ in range(LATTICE_SIZE)] for _ in range(LATTICE_SIZE)], dtype=int)


def _build_marker_plan() -> tuple[np.ndarray, list[dict]]:
    summary = _initial_summary()
    early_edge, late_edge = random.choice(ORDERED_PAIR_CHOICES)
    cross_row = random.randrange(LATTICE_SIZE)
    cross_col = random.randrange(LATTICE_SIZE)
    shared_color = random.choice(PALETTE)

    early_path = _path_to_cross(early_edge, cross_row, cross_col)
    late_path = _path_to_cross(late_edge, cross_row, cross_col)

    for r, c in early_path:
        summary[r, c] = _choose_other_color(shared_color)
    for r, c in late_path[:-1]:
        summary[r, c] = _choose_other_color(shared_color)

    early_tail = _path_after_cross(early_edge, cross_row, cross_col)
    if early_tail and random.random() < 0.5:
        stop_at = random.choice(early_tail)
        stop_idx = early_tail.index(stop_at)
        for r, c in early_tail[:stop_idx]:
            summary[r, c] = _choose_other_color(shared_color)
        summary[stop_at] = shared_color

    markers = [
        {
            "edge": early_edge,
            "band": cross_col if early_edge in {"top", "bottom"} else cross_row,
            "color": shared_color,
        },
        {
            "edge": late_edge,
            "band": cross_col if late_edge in {"top", "bottom"} else cross_row,
            "color": shared_color,
        },
    ]

    remaining_edges = [edge for edge in EDGE_TO_CLOCKWISE_RANK if edge not in {early_edge, late_edge}]
    if random.random() < 0.65:
        extra_edge = random.choice(remaining_edges)
        extra_band = random.randrange(LATTICE_SIZE)
        extra_color = random.choice(PALETTE)
        extra_path = _ray_path(extra_edge, extra_band)
        first_cell = extra_path[0]
        if summary[first_cell] == extra_color:
            summary[first_cell] = _choose_other_color(extra_color)
        markers.append({"edge": extra_edge, "band": extra_band, "color": extra_color})

    return summary, sorted(markers, key=lambda m: EDGE_TO_CLOCKWISE_RANK[m["edge"]])


def _block_noise(majority: int, block_size: int) -> np.ndarray:
    area = block_size * block_size
    minority = _choose_other_color(majority)
    minority_max = max(1, min(area // 3, area // 2 - 1))
    minority_count = random.randint(1, minority_max)
    block = np.full((block_size, block_size), majority, dtype=int)
    cells = random.sample(range(area), minority_count)
    for idx in cells:
        block[idx // block_size, idx % block_size] = minority
    return block


def _band_centers(block_size: int) -> list[int]:
    centers = []
    for idx in range(LATTICE_SIZE):
        start = 1 + idx * (block_size + 1)
        centers.append(start + block_size // 2)
    return centers


def _render_input(summary: np.ndarray, markers: list[dict], block_size: int) -> np.ndarray:
    side = LATTICE_SIZE * block_size + (LATTICE_SIZE + 1)
    grid = np.full((side, side), SEP_COLOR, dtype=int)
    for i in range(LATTICE_SIZE):
        for j in range(LATTICE_SIZE):
            r0 = 1 + i * (block_size + 1)
            c0 = 1 + j * (block_size + 1)
            grid[r0 : r0 + block_size, c0 : c0 + block_size] = _block_noise(int(summary[i, j]), block_size)

    centers = _band_centers(block_size)
    for marker in markers:
        band = int(marker["band"])
        color = int(marker["color"])
        edge = marker["edge"]
        if edge == "top":
            grid[0, centers[band]] = color
        elif edge == "bottom":
            grid[-1, centers[band]] = color
        elif edge == "left":
            grid[centers[band], 0] = color
        else:
            grid[centers[band], -1] = color
    return grid


def _valid_summary(summary: np.ndarray) -> bool:
    if len(set(map(int, summary.ravel()))) < 2:
        return False
    row_varied = any(len(set(map(int, summary[r]))) >= 2 for r in range(LATTICE_SIZE))
    col_varied = any(len(set(map(int, summary[:, c]))) >= 2 for c in range(LATTICE_SIZE))
    return row_varied and col_varied


def _marker_positions_distinct(markers: list[dict]) -> bool:
    edges = [marker["edge"] for marker in markers]
    return len(edges) == len(set(edges))


def _gen_attempt() -> np.ndarray | None:
    summary, markers = _build_marker_plan()
    if not _valid_summary(summary):
        return None
    if not _marker_positions_distinct(markers):
        return None

    painted, traces = _simulate_rays(summary, markers)
    if not any(trace["absorbed"] for trace in traces):
        return None
    if traces[0]["overlap_cells"] < 1:
        return None
    if sum(trace["recolored"] for trace in traces) < 3:
        return None

    expected = np.full((LATTICE_SIZE + 2, LATTICE_SIZE + 2), SEP_COLOR, dtype=int)
    expected[1:-1, 1:-1] = painted

    block_size = random.randint(*BLOCK_SIZE_RANGE)
    grid = _render_input(summary, markers, block_size)
    if np.array_equal(main(grid), expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid sample")
