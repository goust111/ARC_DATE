from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BLACK = 0
GREY = 5
BASE_COLORS = (BLACK, GREY)
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _label_base_components(grid: np.ndarray):
    h, w = grid.shape
    comp_ids = np.full((h, w), -1, dtype=int)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color not in BASE_COLORS or comp_ids[r, c] != -1:
                continue

            cid = len(components)
            queue = deque([(r, c)])
            comp_ids[r, c] = cid
            coords: list[tuple[int, int]] = []

            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if not _in_bounds(grid.shape, nr, nc):
                        continue
                    if comp_ids[nr, nc] != -1 or int(grid[nr, nc]) != color:
                        continue
                    comp_ids[nr, nc] = cid
                    queue.append((nr, nc))

            components.append({"color": color, "coords": coords})

    return components, comp_ids


def _build_component_adjacency(comp_ids: np.ndarray, n_components: int):
    h, w = comp_ids.shape
    adjacency = [defaultdict(int) for _ in range(n_components)]

    for r in range(h):
        for c in range(w - 1):
            a = int(comp_ids[r, c])
            b = int(comp_ids[r, c + 1])
            if a >= 0 and b >= 0 and a != b:
                adjacency[a][b] += 1
                adjacency[b][a] += 1

    for r in range(h - 1):
        for c in range(w):
            a = int(comp_ids[r, c])
            b = int(comp_ids[r + 1, c])
            if a >= 0 and b >= 0 and a != b:
                adjacency[a][b] += 1
                adjacency[b][a] += 1

    return adjacency


def _unique_max_item(counts: dict, context: str):
    assert counts, f"missing candidates for {context}"
    best_value = max(counts.values())
    winners = [key for key, value in counts.items() if value == best_value]
    assert len(winners) == 1, f"tie detected in {context}"
    return winners[0], int(best_value)


def _solve_regions(grid: np.ndarray, *, return_details: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    components, comp_ids = _label_base_components(g)
    assert components, "expected black/grey base regions"

    markers: list[tuple[int, int, int]] = [
        (int(r), int(c), int(g[r, c]))
        for r, c in np.argwhere(~np.isin(g, BASE_COLORS))
    ]
    assert markers, "expected colored marker pixels"

    adjacency = _build_component_adjacency(comp_ids, len(components))
    marker_contacts = [defaultdict(int) for _ in components]

    for marker_idx, (r, c, _color) in enumerate(markers):
        for nr, nc in _neighbors4(r, c):
            if not _in_bounds(g.shape, nr, nc):
                continue
            cid = int(comp_ids[nr, nc])
            if cid >= 0:
                marker_contacts[cid][marker_idx] += 1

    region_colors: dict[int, int] = {}
    wave_of: dict[int, int] = {}
    first_wave_competitions = 0

    for cid, counts in enumerate(marker_contacts):
        if not counts:
            continue
        if len(counts) > 1:
            first_wave_competitions += 1
        best_marker, _ = _unique_max_item(counts, f"first wave component {cid}")
        region_colors[cid] = int(markers[best_marker][2])
        wave_of[cid] = 1

    assert region_colors, "expected at least one first-wave region"

    later_wave_competitions = 0
    wave = 2
    while len(region_colors) < len(components):
        pending: dict[int, int] = {}
        any_frontier = False

        for cid in range(len(components)):
            if cid in region_colors:
                continue

            tally: dict[int, int] = defaultdict(int)
            for nid, shared_len in adjacency[cid].items():
                if nid in region_colors:
                    tally[int(region_colors[nid])] += int(shared_len)

            if not tally:
                continue

            any_frontier = True
            if len(tally) > 1:
                later_wave_competitions += 1
            best_color, _ = _unique_max_item(tally, f"wave {wave} component {cid}")
            pending[cid] = int(best_color)

        assert any_frontier, "uncolored regions were not reachable from the colored wavefront"
        for cid, color in pending.items():
            region_colors[cid] = color
            wave_of[cid] = wave
        wave += 1

    out = g.copy()
    for cid, comp in enumerate(components):
        color = int(region_colors[cid])
        for r, c in comp["coords"]:
            out[r, c] = color

    details = {
        "component_count": len(components),
        "marker_count": len(markers),
        "wave_of": wave_of,
        "max_wave": max(wave_of.values()),
        "first_wave_competitions": first_wave_competitions,
        "later_wave_competitions": later_wave_competitions,
    }
    if return_details:
        return out, details
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor black and grey regions by marker-driven majority, then let the winning colors spread one wave at a time.

    Input structure:
    - The canvas is fully tiled by 4-connected black and grey regions, and same-color regions never share an edge.
    - A few non-black/non-grey marker pixels sit inside some of those regions and act as the initial color sources.

    Rule:
    - First wave: for every region that contains or side-touches marker pixels, count shared edges to each marker pixel and recolor the region with the color of the marker with the largest shared boundary.
    - Later waves: at each step, every still-uncolored region that side-touches an already recolored region sums shared boundary length by propagated color and takes the unique longest total.
    - Repeat wave by wave until every original black/grey region is recolored, while marker pixels themselves stay unchanged.

    Key insight:
    - Distant regions are not decided directly by the nearest marker; earlier recolored regions become the next sources, so wave order and shared boundary length both matter.
    """
    return _solve_regions(grid, return_details=False)


ROW_COUNT_OPTIONS = (3, 4, 5)
COL_COUNT_OPTIONS = (3, 4, 5)
ROW_HEIGHT_OPTIONS = (3, 5)
COL_WIDTH_OPTIONS = (4, 6)
SEED_COUNT_OPTIONS = (2, 3)
MAX_GEN_ATTEMPTS = 400


def _cell_neighbors(cell: tuple[int, int], rows: int, cols: int):
    r, c = cell
    if r > 0:
        yield (r - 1, c)
    if r + 1 < rows:
        yield (r + 1, c)
    if c > 0:
        yield (r, c - 1)
    if c + 1 < cols:
        yield (r, c + 1)


def _shared_boundary_length(
    cell: tuple[int, int],
    neighbor: tuple[int, int],
    row_heights: list[int],
    col_widths: list[int],
) -> int:
    r, c = cell
    nr, nc = neighbor
    if r == nr:
        return int(row_heights[r])
    return int(col_widths[c])


def _solve_cell_model(
    rows: int,
    cols: int,
    row_heights: list[int],
    col_widths: list[int],
    seeds: dict[tuple[int, int], int],
):
    colored = dict(seeds)
    wave_of = {cell: 1 for cell in seeds}
    later_wave_competitions = 0
    wave = 2

    while len(colored) < rows * cols:
        pending: dict[tuple[int, int], int] = {}
        any_frontier = False

        for r in range(rows):
            for c in range(cols):
                cell = (r, c)
                if cell in colored:
                    continue

                tally: dict[int, int] = defaultdict(int)
                for neighbor in _cell_neighbors(cell, rows, cols):
                    if neighbor in colored:
                        color = int(colored[neighbor])
                        tally[color] += _shared_boundary_length(
                            cell, neighbor, row_heights, col_widths
                        )

                if not tally:
                    continue

                any_frontier = True
                best_value = max(tally.values())
                winners = [color for color, value in tally.items() if value == best_value]
                if len(winners) != 1:
                    return None
                if len(tally) > 1:
                    later_wave_competitions += 1
                pending[cell] = int(winners[0])

        if not any_frontier:
            return None

        for cell, color in pending.items():
            colored[cell] = color
            wave_of[cell] = wave
        wave += 1

    return {
        "cell_colors": colored,
        "wave_of": wave_of,
        "max_wave": max(wave_of.values()),
        "later_wave_competitions": later_wave_competitions,
    }


def _sample_band_sizes(count: int, options: tuple[int, ...]) -> list[int]:
    sizes = [int(random.choice(options)) for _ in range(count)]
    if count >= 2 and len(set(options)) > 1 and len(set(sizes)) == 1:
        flip = 0 if sizes[0] != options[0] else -1
        sizes[random.randrange(count)] = int(options[flip])
    return sizes


def _make_partition():
    rows = int(random.choice(ROW_COUNT_OPTIONS))
    cols = int(random.choice(COL_COUNT_OPTIONS))
    row_heights = _sample_band_sizes(rows, ROW_HEIGHT_OPTIONS)
    col_widths = _sample_band_sizes(cols, COL_WIDTH_OPTIONS)

    h = sum(row_heights)
    w = sum(col_widths)
    grid = np.zeros((h, w), dtype=int)
    cell_boxes: dict[tuple[int, int], tuple[int, int, int, int]] = {}

    r0 = 0
    for r, height in enumerate(row_heights):
        c0 = 0
        for c, width in enumerate(col_widths):
            color = BLACK if (r + c) % 2 == 0 else GREY
            grid[r0 : r0 + height, c0 : c0 + width] = color
            cell_boxes[(r, c)] = (r0, c0, r0 + height - 1, c0 + width - 1)
            c0 += width
        r0 += height

    return grid, rows, cols, row_heights, col_widths, cell_boxes


def _choose_seed_cells(rows: int, cols: int, seed_count: int):
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    weighted = []
    for cell in cells:
        r, c = cell
        border = r in {0, rows - 1} or c in {0, cols - 1}
        corner = r in {0, rows - 1} and c in {0, cols - 1}
        repeats = 4 if corner else 2 if border else 1
        weighted.extend([cell] * repeats)

    for _ in range(200):
        chosen: list[tuple[int, int]] = []
        while len(chosen) < seed_count:
            cell = random.choice(weighted)
            if cell in chosen:
                continue
            if any(abs(cell[0] - rr) + abs(cell[1] - cc) < 2 for rr, cc in chosen):
                continue
            chosen.append(cell)
        if max(
            abs(a[0] - b[0]) + abs(a[1] - b[1])
            for i, a in enumerate(chosen)
            for b in chosen[i + 1 :]
        ) >= max(rows, cols) - 1:
            return chosen
    return None


def _pick_marker_position(box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    rows = list(range(r0 + 1, r1))
    cols = list(range(c0 + 1, c1))
    return int(random.choice(rows)), int(random.choice(cols))


def _cell_anchor(box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    return r0 + 1, c0 + 1


def _expected_output_from_cells(
    rows: int,
    cols: int,
    cell_boxes: dict[tuple[int, int], tuple[int, int, int, int]],
    cell_colors: dict[tuple[int, int], int],
    markers: list[tuple[int, int, int]],
) -> np.ndarray:
    h = max(box[2] for box in cell_boxes.values()) + 1
    w = max(box[3] for box in cell_boxes.values()) + 1
    out = np.zeros((h, w), dtype=int)

    for cell, box in cell_boxes.items():
        r0, c0, r1, c1 = box
        out[r0 : r1 + 1, c0 : c1 + 1] = int(cell_colors[cell])

    for r, c, color in markers:
        out[r, c] = int(color)
    return out


def _attempt_generate() -> np.ndarray | None:
    grid, rows, cols, row_heights, col_widths, cell_boxes = _make_partition()

    seed_count = int(random.choice(SEED_COUNT_OPTIONS))
    seed_cells = _choose_seed_cells(rows, cols, seed_count)
    if seed_cells is None:
        return None

    colors = random.sample(MARKER_COLORS, seed_count)
    seeds = {cell: int(color) for cell, color in zip(seed_cells, colors)}

    cell_model = _solve_cell_model(rows, cols, row_heights, col_widths, seeds)
    if cell_model is None:
        return None
    if cell_model["max_wave"] < 3:
        return None
    if cell_model["later_wave_competitions"] == 0:
        return None

    markers: list[tuple[int, int, int]] = []
    for cell, color in seeds.items():
        r, c = _pick_marker_position(cell_boxes[cell])
        markers.append((r, c, int(color)))
        grid[r, c] = int(color)

    components, _comp_ids = _label_base_components(grid)
    if len(components) != rows * cols:
        return None

    expected = _expected_output_from_cells(
        rows, cols, cell_boxes, cell_model["cell_colors"], markers
    )

    try:
        actual, details = _solve_regions(grid, return_details=True)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None
    if details["component_count"] != rows * cols:
        return None
    if details["marker_count"] != seed_count:
        return None
    if details["max_wave"] != cell_model["max_wave"]:
        return None
    if details["later_wave_competitions"] == 0:
        return None
    if len(set(cell_model["cell_colors"].values())) < 2:
        return None

    anchors = {
        cell: _cell_anchor(cell_boxes[cell]) for cell in cell_boxes
    }
    if len({int(actual[r, c]) for r, c in anchors.values()}) < 2:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
