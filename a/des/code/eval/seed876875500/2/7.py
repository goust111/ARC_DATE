from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import random

import numpy as np


Coord = tuple[int, int]

YELLOW = 4
GREY = 5
DIRS: tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
FRAGMENT_LENGTHS = (2, 3, 4)
LEG_LENGTHS = (1, 2, 3)
TEMP_GROUP_SHAPE = (25, 25)
MAX_GROUP_EXTENT = 16
GROUP_BUILD_ATTEMPTS = 120
FRAGMENT_PLACEMENT_ATTEMPTS = 120
SCENE_GENERATION_ATTEMPTS = 400


@dataclass(frozen=True)
class Fragment:
    color: int
    cells: tuple[Coord, ...]
    entrance: Coord
    departure: Coord
    out_dir: Coord
    marker: Coord


def _add(a: Coord, b: Coord) -> Coord:
    return (a[0] + b[0], a[1] + b[1])


def _scale(d: Coord, k: int) -> Coord:
    return (d[0] * k, d[1] * k)


def _neg(d: Coord) -> Coord:
    return (-d[0], -d[1])


def _neighbors4(cell: Coord) -> tuple[Coord, ...]:
    r, c = cell
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _orthogonals(d: Coord) -> tuple[Coord, Coord]:
    if d[0] != 0:
        return ((0, -1), (0, 1))
    return ((-1, 0), (1, 0))


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4((cr, cc)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _fragment_from_component(grid: np.ndarray, color: int, coords: np.ndarray) -> Fragment:
    cells = [tuple(map(int, cell)) for cell in coords]
    rows = sorted({r for r, _c in cells})
    cols = sorted({c for _r, c in cells})
    assert 2 <= len(cells) <= 4, "each fragment must have length 2-4"

    if len(rows) == 1:
        r = rows[0]
        assert cols[-1] - cols[0] + 1 == len(cells), "horizontal fragment must be contiguous"
        ordered = tuple((r, c) for c in range(cols[0], cols[-1] + 1))
    else:
        assert len(cols) == 1, "fragment must be orthogonal"
        c = cols[0]
        assert rows[-1] - rows[0] + 1 == len(cells), "vertical fragment must be contiguous"
        ordered = tuple((r, c) for r in range(rows[0], rows[-1] + 1))

    endpoints = (ordered[0], ordered[-1])
    marker_info: list[tuple[Coord, Coord]] = []
    for endpoint in endpoints:
        markers = []
        for nr, nc in _neighbors4(endpoint):
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and int(grid[nr, nc]) == YELLOW:
                markers.append((nr, nc))
        if markers:
            assert len(markers) == 1, "marked endpoint must have exactly one yellow neighbor"
            marker_info.append((endpoint, markers[0]))

    assert len(marker_info) == 1, "each fragment must have exactly one marked endpoint"
    departure, marker = marker_info[0]
    entrance = endpoints[1] if endpoints[0] == departure else endpoints[0]
    out_dir = (marker[0] - departure[0], marker[1] - departure[1])
    assert out_dir in DIRS, "marker must sit one orthogonal step from the departure endpoint"
    return Fragment(
        color=int(color),
        cells=ordered,
        entrance=entrance,
        departure=departure,
        out_dir=out_dir,
        marker=marker,
    )


def _extract_fragments(grid: np.ndarray, color: int) -> list[Fragment]:
    fragments = [_fragment_from_component(grid, color, comp) for comp in _components_of_color(grid, color)]
    assert 3 <= len(fragments) <= 5, "each color group must contain 3-5 fragments"
    return sorted(fragments, key=lambda frag: (min(frag.cells), frag.entrance, frag.departure))


def _connector_cells(source: Coord, out_dir: Coord, target: Coord) -> list[Coord] | None:
    sr, sc = source
    tr, tc = target
    dr, dc = out_dir

    if dr != 0:
        delta = tr - sr
        if delta * dr < 1 or tc == sc:
            return None
        cells = [(sr + dr * step, sc) for step in range(1, abs(delta) + 1)]
        turn = 1 if tc > sc else -1
        cells.extend((tr, sc + turn * step) for step in range(1, abs(tc - sc)))
        return cells

    delta = tc - sc
    if delta * dc < 1 or tr == sr:
        return None
    cells = [(sr, sc + dc * step) for step in range(1, abs(delta) + 1)]
    turn = 1 if tr > sr else -1
    cells.extend((sr + turn * step, tc) for step in range(1, abs(tr - sr)))
    return cells


def _valid_path_cells(grid: np.ndarray, bg: int, source: Fragment, target: Fragment) -> list[Coord] | None:
    cells = _connector_cells(source.departure, source.out_dir, target.entrance)
    if not cells or cells[0] != source.marker:
        return None

    h, w = grid.shape
    for idx, cell in enumerate(cells):
        r, c = cell
        if not (0 <= r < h and 0 <= c < w):
            return None
        value = int(grid[r, c])
        if idx == 0:
            if value != YELLOW:
                return None
        elif value != bg:
            return None
    return cells


def _solve_group(grid: np.ndarray, bg: int, fragments: list[Fragment]) -> tuple[tuple[int | None, ...], dict[tuple[int, int], list[Coord]]]:
    n = len(fragments)
    candidates: list[list[int]] = [[] for _ in range(n)]
    paths: dict[tuple[int, int], list[Coord]] = {}

    for i, source in enumerate(fragments):
        for j, target in enumerate(fragments):
            if i == j:
                continue
            cells = _valid_path_cells(grid, bg, source, target)
            if cells is None:
                continue
            candidates[i].append(j)
            paths[(i, j)] = cells

    solutions: list[tuple[int | None, ...]] = []
    options = [[None] + sorted(cands) for cands in candidates]
    for succ in product(*options):
        if succ.count(None) != 1:
            continue
        indeg = [0] * n
        ok = True
        for i, j in enumerate(succ):
            if j is None:
                continue
            indeg[j] += 1
            if indeg[j] > 1:
                ok = False
                break
        if not ok or sum(x is not None for x in succ) != n - 1:
            continue

        starts = [idx for idx, deg in enumerate(indeg) if deg == 0]
        if len(starts) != 1:
            continue

        seen: set[int] = set()
        cur = starts[0]
        while cur not in seen and cur is not None:
            seen.add(cur)
            cur = succ[cur]
        if cur is not None or len(seen) != n:
            continue
        solutions.append(tuple(None if item is None else int(item) for item in succ))

    assert len(solutions) == 1, "fragment chain must be uniquely determined"
    return solutions[0], paths


def main(grid: np.ndarray) -> np.ndarray:
    """
    Join same-color line fragments into full polylines by following the yellow endpoint markers.

    Input structure:
    - The background is the dominant color.
    - Each non-background, non-yellow color forms 3-5 separate straight line
      fragments of length 2-4.
    - Exactly one endpoint of each fragment has an adjacent yellow marker,
      whose direction indicates the fragment's outgoing direction.

    Rule:
    - For each color, connect fragments into one chain.
    - The outgoing endpoint of a fragment links to another fragment's unmarked
      endpoint using a single-turn grey L path.
    - The first leg follows the yellow marker direction; then the path turns
      once to reach the next fragment's entrance endpoint.
    - Delete all yellow markers and keep the original fragment colors.

    Key insight:
    - The yellow marker simultaneously identifies the departure endpoint and
      fixes the first leg of the missing connector, so the full chain is
      recovered by solving a direction-constrained path order.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    colors = [int(color) for color in np.unique(g) if int(color) not in {bg, YELLOW, GREY}]
    out = g.copy()
    out[out == YELLOW] = bg

    for color in colors:
        fragments = _extract_fragments(g, color)
        succ, paths = _solve_group(g, bg, fragments)
        for i, j in enumerate(succ):
            if j is None:
                continue
            for r, c in paths[(i, j)]:
                assert int(out[r, c]) in {bg, GREY}, "connector may only cover empty cells or another grey connector"
                out[r, c] = GREY

    return out


def _inside(shape: tuple[int, int], cell: Coord) -> bool:
    return 0 <= cell[0] < shape[0] and 0 <= cell[1] < shape[1]


def _can_place_fragment(
    cells: list[Coord],
    entrance: Coord,
    arrival_dir: Coord | None,
    fragment_cells: set[Coord],
    marker_cells: set[Coord],
    connector_cells: set[Coord],
) -> bool:
    cell_set = set(cells)
    if any(cell in fragment_cells or cell in marker_cells or cell in connector_cells for cell in cells):
        return False

    allowed_connector_neighbor = None if arrival_dir is None else _add(entrance, _neg(arrival_dir))
    if arrival_dir is not None and allowed_connector_neighbor not in connector_cells:
        return False

    for cell in cells:
        for nb in _neighbors4(cell):
            if nb in fragment_cells and nb not in cell_set:
                return False
            if nb in marker_cells:
                return False
            if nb in connector_cells:
                if not (cell == entrance and nb == allowed_connector_neighbor):
                    return False
    return True


def _can_place_marker(
    marker: Coord,
    departure: Coord,
    fragment_cells: set[Coord],
    marker_cells: set[Coord],
    connector_cells: set[Coord],
) -> bool:
    if marker in fragment_cells or marker in marker_cells or marker in connector_cells:
        return False
    for nb in _neighbors4(marker):
        if nb == departure:
            continue
        if nb in fragment_cells or nb in marker_cells or nb in connector_cells:
            return False
    return True


def _can_place_path(
    path: list[Coord],
    marker: Coord,
    departure: Coord,
    fragment_cells: set[Coord],
    marker_cells: set[Coord],
    connector_cells: set[Coord],
) -> bool:
    if not path or path[0] != marker or len(set(path)) != len(path):
        return False
    path_set = set(path)
    for idx, cell in enumerate(path):
        if idx == 0:
            if cell in fragment_cells or cell in connector_cells:
                return False
        else:
            if cell in fragment_cells or cell in marker_cells or cell in connector_cells:
                return False
        allowed_neighbors = set()
        if idx == 0:
            allowed_neighbors.add(departure)
        if idx > 0:
            allowed_neighbors.add(path[idx - 1])
        if idx + 1 < len(path):
            allowed_neighbors.add(path[idx + 1])
        for nb in _neighbors4(cell):
            if nb in path_set or nb in allowed_neighbors:
                continue
            if nb in fragment_cells or nb in marker_cells or nb in connector_cells:
                return False
    return True


def _crop_patch(painted: dict[Coord, int], footprint: set[Coord]) -> dict[str, object]:
    all_cells = set(painted) | footprint
    min_r = min(r for r, _c in all_cells)
    max_r = max(r for r, _c in all_cells)
    min_c = min(c for _r, c in all_cells)
    max_c = max(c for _r, c in all_cells)
    shape = (max_r - min_r + 1, max_c - min_c + 1)
    return {
        "shape": shape,
        "paint": [((r - min_r, c - min_c), int(color)) for (r, c), color in painted.items()],
    }


def _build_group_patch(color: int, n_fragments: int) -> dict[str, object] | None:
    h, w = TEMP_GROUP_SHAPE
    for _ in range(GROUP_BUILD_ATTEMPTS):
        fragment_cells: set[Coord] = set()
        marker_cells: set[Coord] = set()
        connector_cells: set[Coord] = set()
        painted: dict[Coord, int] = {}
        entrance = (random.randint(8, h - 9), random.randint(8, w - 9))
        arrival_dir: Coord | None = None
        ok = True

        for frag_index in range(n_fragments):
            placed = False
            for _ in range(FRAGMENT_PLACEMENT_ATTEMPTS):
                seg_dir_choices = list(DIRS)
                random.shuffle(seg_dir_choices)
                if arrival_dir is not None:
                    seg_dir_choices = [d for d in seg_dir_choices if d != _neg(arrival_dir)]
                seg_dir = random.choice(seg_dir_choices)
                length = random.choice(FRAGMENT_LENGTHS)
                frag_cells = [_add(entrance, _scale(seg_dir, step)) for step in range(length)]
                if not all(_inside(TEMP_GROUP_SHAPE, cell) for cell in frag_cells):
                    continue
                if not _can_place_fragment(frag_cells, entrance, arrival_dir, fragment_cells, marker_cells, connector_cells):
                    continue

                departure = frag_cells[-1]
                out_dir_choices = [d for d in DIRS if d != _neg(seg_dir)]
                random.shuffle(out_dir_choices)
                out_dir = random.choice(out_dir_choices)
                marker = _add(departure, out_dir)
                if not _inside(TEMP_GROUP_SHAPE, marker):
                    continue
                if not _can_place_marker(marker, departure, fragment_cells, marker_cells, connector_cells):
                    continue

                if frag_index < n_fragments - 1:
                    turn_dir_choices = list(_orthogonals(out_dir))
                    random.shuffle(turn_dir_choices)
                    turn_dir = random.choice(turn_dir_choices)
                    leg1 = random.choice(LEG_LENGTHS)
                    leg2 = random.choice(LEG_LENGTHS)
                    next_entrance = _add(_add(departure, _scale(out_dir, leg1)), _scale(turn_dir, leg2))
                    if not _inside(TEMP_GROUP_SHAPE, next_entrance):
                        continue
                    if next_entrance in fragment_cells or next_entrance in marker_cells or next_entrance in connector_cells:
                        continue
                    path = _connector_cells(departure, out_dir, next_entrance)
                    if path is None:
                        continue
                    if not all(_inside(TEMP_GROUP_SHAPE, cell) for cell in path):
                        continue
                    if not _can_place_path(path, marker, departure, fragment_cells, marker_cells, connector_cells):
                        continue
                else:
                    turn_dir = None
                    next_entrance = None
                    path = []

                for cell in frag_cells:
                    fragment_cells.add(cell)
                    painted[cell] = int(color)
                marker_cells.add(marker)
                painted[marker] = YELLOW
                for cell in path:
                    connector_cells.add(cell)
                entrance = next_entrance if next_entrance is not None else entrance
                arrival_dir = turn_dir
                placed = True
                break

            if not placed:
                ok = False
                break

        if not ok:
            continue

        footprint = fragment_cells | marker_cells | connector_cells
        patch = _crop_patch(painted, footprint)
        patch_h, patch_w = patch["shape"]
        if patch_h <= MAX_GROUP_EXTENT and patch_w <= MAX_GROUP_EXTENT:
            return patch
    return None


def _paint_patch(grid: np.ndarray, patch: dict[str, object], top: int, left: int) -> None:
    for (r, c), color in patch["paint"]:
        grid[top + r, left + c] = int(color)


def _build_scene() -> np.ndarray | None:
    group_count = random.randint(2, 3)
    bg_choices = [color for color in range(10) if color not in {YELLOW, GREY}]
    bg = random.choice(bg_choices)
    group_colors = random.sample([color for color in range(10) if color not in {bg, YELLOW, GREY}], group_count)
    patches = []
    for color in group_colors:
        patch = _build_group_patch(color, random.randint(3, 5))
        if patch is None:
            return None
        patches.append(patch)
    random.shuffle(patches)

    layout = random.choice(["row", "col"])
    gaps = [random.randint(3, 6) for _ in range(group_count - 1)]
    top_margin = random.randint(1, 4)
    bottom_margin = random.randint(1, 4)
    left_margin = random.randint(1, 4)
    right_margin = random.randint(1, 4)

    heights = [patch["shape"][0] for patch in patches]
    widths = [patch["shape"][1] for patch in patches]
    if layout == "row":
        grid_h = top_margin + max(heights) + bottom_margin + random.randint(0, 3)
        grid_w = left_margin + sum(widths) + sum(gaps) + right_margin
        grid = np.full((grid_h, grid_w), int(bg), dtype=int)
        cur_left = left_margin
        for idx, patch in enumerate(patches):
            patch_h, patch_w = patch["shape"]
            top = random.randint(top_margin, grid_h - bottom_margin - patch_h)
            _paint_patch(grid, patch, top, cur_left)
            cur_left += patch_w
            if idx < len(gaps):
                cur_left += gaps[idx]
    else:
        grid_h = top_margin + sum(heights) + sum(gaps) + bottom_margin
        grid_w = left_margin + max(widths) + right_margin + random.randint(0, 3)
        grid = np.full((grid_h, grid_w), int(bg), dtype=int)
        cur_top = top_margin
        for idx, patch in enumerate(patches):
            patch_h, patch_w = patch["shape"]
            left = random.randint(left_margin, grid_w - right_margin - patch_w)
            _paint_patch(grid, patch, cur_top, left)
            cur_top += patch_h
            if idx < len(gaps):
                cur_top += gaps[idx]

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if np.array_equal(solved, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(SCENE_GENERATION_ATTEMPTS):
        grid = _build_scene()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
