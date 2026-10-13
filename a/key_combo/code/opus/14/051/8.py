from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_ATTEMPTS = 400


def neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _lattice_lines(grid: np.ndarray) -> tuple[list[int], list[int]]:
    h, w = grid.shape
    row_threshold = w // 2
    col_threshold = h // 2
    rows = [int(r) for r in range(h) if int(np.count_nonzero(grid[r] == GREY)) > row_threshold]
    cols = [int(c) for c in range(w) if int(np.count_nonzero(grid[:, c] == GREY)) > col_threshold]
    assert len(rows) >= 5 and len(cols) >= 5, "expected a lattice with at least 4x4 cells"
    assert rows[0] == 0 and rows[-1] == h - 1
    assert cols[0] == 0 and cols[-1] == w - 1
    return rows, cols


def _cell_boxes(grid: np.ndarray) -> list[list[tuple[int, int, int, int]]]:
    rows, cols = _lattice_lines(grid)
    row_sizes = [rows[i + 1] - rows[i] - 1 for i in range(len(rows) - 1)]
    col_sizes = [cols[i + 1] - cols[i] - 1 for i in range(len(cols) - 1)]
    assert len(set(row_sizes)) == 1 and len(set(col_sizes)) == 1
    cell_h = row_sizes[0]
    cell_w = col_sizes[0]
    assert cell_h in (2, 3) and cell_w == cell_h
    boxes: list[list[tuple[int, int, int, int]]] = []
    for i in range(len(rows) - 1):
        row_boxes = []
        for j in range(len(cols) - 1):
            row_boxes.append((rows[i] + 1, cols[j] + 1, rows[i + 1] - 1, cols[j + 1] - 1))
        boxes.append(row_boxes)
    return boxes


def _cell_of(boxes: list[list[tuple[int, int, int, int]]], r: int, c: int) -> tuple[int, int]:
    for i, row in enumerate(boxes):
        for j, (r0, c0, r1, c1) in enumerate(row):
            if r0 <= r <= r1 and c0 <= c <= c1:
                return i, j
    raise AssertionError("coordinate does not lie in any cell")


def _seed_counts(
    grid: np.ndarray, boxes: list[list[tuple[int, int, int, int]]]
) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]]:
    n_rows = len(boxes)
    n_cols = len(boxes[0])
    edge_seeds: list[tuple[int, int]] = []
    inner_seeds: list[tuple[int, int]] = []
    for i, row in enumerate(boxes):
        for j, (r0, c0, r1, c1) in enumerate(row):
            ys = np.argwhere(grid[r0 : r1 + 1, c0 : c1 + 1] == YELLOW)
            assert len(ys) <= 1, "at most one yellow marker per cell"
            if len(ys) == 0:
                continue
            rr, cc = ys[0]
            coord = (int(r0 + rr), int(c0 + cc))
            if i in (0, n_rows - 1) or j in (0, n_cols - 1):
                edge_seeds.append(coord)
            else:
                inner_seeds.append(coord)
    return len(edge_seeds), len(inner_seeds), edge_seeds, inner_seeds


def _simulate(
    grid: np.ndarray,
    *,
    active_kind: str | None = None,
    ignore_obstacles: bool = False,
) -> tuple[np.ndarray, dict]:
    boxes = _cell_boxes(grid)
    edge_count, inner_count, edge_seeds, inner_seeds = _seed_counts(grid, boxes)
    assert edge_count != inner_count, "edge and interior counts must differ"

    if active_kind is None:
        active_kind = "edge" if edge_count > inner_count else "inner"
    assert active_kind in {"edge", "inner"}

    if active_kind == "edge":
        active_seeds = edge_seeds
        fill_color = RED
    else:
        active_seeds = inner_seeds
        fill_color = GREEN

    out = grid.copy()
    seen = np.zeros(grid.shape, dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r, c in active_seeds:
        seen[r, c] = True
        out[r, c] = fill_color
        q.append((r, c))

    passable = {BLACK}
    if ignore_obstacles:
        passable |= {BLUE, RED}

    while q:
        r, c = q.popleft()
        for rr, cc in neighbors4(r, c):
            if not in_bounds(grid.shape, rr, cc) or seen[rr, cc]:
                continue
            if int(grid[rr, cc]) not in passable:
                continue
            seen[rr, cc] = True
            if int(grid[rr, cc]) == BLACK:
                out[rr, cc] = fill_color
            q.append((rr, cc))

    return out, {
        "boxes": boxes,
        "edge_count": edge_count,
        "inner_count": inner_count,
        "edge_seeds": edge_seeds,
        "inner_seeds": inner_seeds,
        "active_kind": active_kind,
        "fill_color": fill_color,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Flood from the yellow markers of the majority cell class across a holed lattice while preserving the losing markers.

    Input structure:
    - Black background cells are partitioned by a grey lattice into a complete 4x4 or 5x5 array of square cell interiors.
    - Some cells contain one yellow seed pixel; other cells may contain a small 4-connected red or blue blocker shape.
    - A few grey divider pixels are missing, creating one-pixel passages between neighboring cells while the rest of the lattice stays blocking.

    Rule:
    - Recover the lattice rows and columns, then classify each cell by array position as edge or strictly interior.
    - Count yellow seeds in edge cells and interior cells; if edge has more, activate the edge seeds and use red fill, otherwise activate the interior seeds and use green fill.
    - Starting from every activated seed, flood only through 4-connected black pixels; grey lattice pixels and colored blocker shapes remain impassable, activated seeds are recolored to the fill color, and non-activated yellow seeds stay yellow.

    Key insight:
    - The grey lattice simultaneously defines cell membership, the edge-versus-interior counting categories, and the topology of the flood because only its single-pixel gaps let the winning seeds escape into neighboring cells.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    out, _meta = _simulate(g)
    return out


def _build_lattice(n_cells: int, cell_size: int) -> tuple[np.ndarray, list[list[tuple[int, int, int, int]]]]:
    dim = n_cells * cell_size + (n_cells + 1)
    grid = np.full((dim, dim), GREY, dtype=int)
    boxes: list[list[tuple[int, int, int, int]]] = []
    for i in range(n_cells):
        row_boxes = []
        r0 = 1 + i * (cell_size + 1)
        for j in range(n_cells):
            c0 = 1 + j * (cell_size + 1)
            grid[r0 : r0 + cell_size, c0 : c0 + cell_size] = BLACK
            row_boxes.append((r0, c0, r0 + cell_size - 1, c0 + cell_size - 1))
        boxes.append(row_boxes)
    return grid, boxes


def _is_edge(i: int, j: int, n_cells: int) -> bool:
    return i in (0, n_cells - 1) or j in (0, n_cells - 1)


def _adjacent(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def _shared_gap_coord(
    boxes: list[list[tuple[int, int, int, int]]],
    a: tuple[int, int],
    b: tuple[int, int],
    offset: int,
) -> tuple[int, int]:
    ar, ac = a
    br, bc = b
    a_box = boxes[ar][ac]
    b_box = boxes[br][bc]
    assert _adjacent(a, b)
    if ar == br:
        gap_c = a_box[3] + 1 if bc > ac else b_box[3] + 1
        base_r = a_box[0]
        return base_r + offset, gap_c
    gap_r = a_box[2] + 1 if br > ar else b_box[2] + 1
    base_c = a_box[1]
    return gap_r, base_c + offset


def _carve_gap(
    grid: np.ndarray,
    boxes: list[list[tuple[int, int, int, int]]],
    a: tuple[int, int],
    b: tuple[int, int],
    offset: int,
) -> None:
    r, c = _shared_gap_coord(boxes, a, b, offset)
    grid[r, c] = BLACK


def _cell_coords(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _place_seed(grid: np.ndarray, box: tuple[int, int, int, int], avoid: set[tuple[int, int]]) -> tuple[int, int]:
    choices = [coord for coord in _cell_coords(box) if coord not in avoid]
    assert choices
    r, c = random.choice(choices)
    grid[r, c] = YELLOW
    return r, c


def _translate_shape(
    shape: list[tuple[int, int]], cell_size: int
) -> list[tuple[int, int]]:
    max_r = max(r for r, _ in shape)
    max_c = max(c for _, c in shape)
    shift_r = random.randint(0, cell_size - 1 - max_r)
    shift_c = random.randint(0, cell_size - 1 - max_c)
    return [(r + shift_r, c + shift_c) for r, c in shape]


def _random_obstacle_shape(cell_size: int) -> list[tuple[int, int]]:
    if cell_size == 2:
        shapes = [
            [(0, 0), (0, 1)],
            [(0, 0), (1, 0)],
            [(0, 1), (1, 1)],
            [(1, 0), (1, 1)],
            [(0, 0), (0, 1), (1, 0)],
            [(0, 0), (0, 1), (1, 1)],
            [(0, 0), (1, 0), (1, 1)],
            [(0, 1), (1, 0), (1, 1)],
        ]
        return random.choice(shapes)

    bases = [
        [(0, 0), (0, 1)],
        [(0, 0), (1, 0)],
        [(0, 0), (0, 1), (0, 2)],
        [(0, 0), (1, 0), (2, 0)],
        [(0, 0), (0, 1), (1, 0)],
        [(0, 0), (0, 1), (1, 1)],
        [(0, 0), (1, 0), (1, 1)],
        [(0, 1), (1, 0), (1, 1)],
    ]
    return _translate_shape(random.choice(bases), cell_size)


def _gate_shape(cell_size: int, orientation: str) -> list[tuple[int, int]]:
    if orientation == "h":
        if cell_size == 2:
            return [(0, 1), (1, 1)]
        return [(0, 1), (1, 1), (2, 1)]
    if cell_size == 2:
        return [(1, 0), (1, 1)]
    return [(1, 0), (1, 1), (1, 2)]


def _paint_shape(
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    shape: list[tuple[int, int]],
    color: int,
) -> None:
    r0, c0, _r1, _c1 = box
    for dr, dc in shape:
        grid[r0 + dr, c0 + dc] = color


def _cell_sets(n_cells: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    edges: list[tuple[int, int]] = []
    inners: list[tuple[int, int]] = []
    for i in range(n_cells):
        for j in range(n_cells):
            (edges if _is_edge(i, j, n_cells) else inners).append((i, j))
    return edges, inners


def _candidate_triples(n_cells: int, winner_kind: str) -> list[tuple[tuple[int, int], tuple[int, int], tuple[int, int], str]]:
    triples = []
    for i in range(n_cells):
        for j in range(n_cells - 2):
            cells = [(i, j), (i, j + 1), (i, j + 2)]
            for a, b, c in (cells, list(reversed(cells))):
                if (_is_edge(a[0], a[1], n_cells) and winner_kind == "edge") or (
                    not _is_edge(a[0], a[1], n_cells) and winner_kind == "inner"
                ):
                    triples.append((a, b, c, "h"))
    for i in range(n_cells - 2):
        for j in range(n_cells):
            cells = [(i, j), (i + 1, j), (i + 2, j)]
            for a, b, c in (cells, list(reversed(cells))):
                if (_is_edge(a[0], a[1], n_cells) and winner_kind == "edge") or (
                    not _is_edge(a[0], a[1], n_cells) and winner_kind == "inner"
                ):
                    triples.append((a, b, c, "v"))
    return triples


def _candidate_pairs(
    n_cells: int,
    loser_kind: str,
    forbidden: set[tuple[int, int]],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    pairs = []
    for i in range(n_cells):
        for j in range(n_cells):
            a = (i, j)
            if a in forbidden:
                continue
            a_is_edge = _is_edge(i, j, n_cells)
            if (loser_kind == "edge" and not a_is_edge) or (loser_kind == "inner" and a_is_edge):
                continue
            for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                b = (i + dr, j + dc)
                if not (0 <= b[0] < n_cells and 0 <= b[1] < n_cells) or b in forbidden:
                    continue
                pairs.append((a, b))
    return pairs


def _coords_to_cells(
    coords: list[tuple[int, int]], boxes: list[list[tuple[int, int, int, int]]]
) -> set[tuple[int, int]]:
    return {_cell_of(boxes, r, c) for r, c in coords}


def _changed_cells(
    inp: np.ndarray,
    out: np.ndarray,
    boxes: list[list[tuple[int, int, int, int]]],
    fill_color: int,
) -> set[tuple[int, int]]:
    changed = np.argwhere((inp != out) & (out == fill_color))
    cells: set[tuple[int, int]] = set()
    for r, c in changed:
        try:
            cells.add(_cell_of(boxes, int(r), int(c)))
        except AssertionError:
            continue
    return cells


def _obstacle_components(grid: np.ndarray) -> list[tuple[int, list[tuple[int, int]]]]:
    comps: list[tuple[int, list[tuple[int, int]]]] = []
    seen = np.zeros(grid.shape, dtype=bool)
    for color in (BLUE, RED):
        for r, c in np.argwhere(grid == color):
            rr = int(r)
            cc = int(c)
            if seen[rr, cc]:
                continue
            q = deque([(rr, cc)])
            seen[rr, cc] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc2 = q.popleft()
                comp.append((cr, cc2))
                for nr, nc in neighbors4(cr, cc2):
                    if in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append((color, comp))
    return comps


def _semantic_ok(grid: np.ndarray) -> bool:
    try:
        out, meta = _simulate(grid)
        boxes = meta["boxes"]
        edge_count = meta["edge_count"]
        inner_count = meta["inner_count"]
        active_kind = meta["active_kind"]
        fill_color = meta["fill_color"]
        active_seeds = meta["edge_seeds"] if active_kind == "edge" else meta["inner_seeds"]
        inactive_seeds = meta["inner_seeds"] if active_kind == "edge" else meta["edge_seeds"]

        if not active_seeds or not inactive_seeds:
            return False
        if np.array_equal(out, grid):
            return False

        wrong_kind = "inner" if active_kind == "edge" else "edge"
        wrong_out, wrong_meta = _simulate(grid, active_kind=wrong_kind)
        if np.array_equal(wrong_out, out):
            return False
        if active_kind == "edge" and not (edge_count > inner_count):
            return False
        if active_kind == "inner" and not (inner_count > edge_count):
            return False

        active_seed_cells = _coords_to_cells(active_seeds, boxes)
        changed_cells = _changed_cells(grid, out, boxes, fill_color)
        if not (changed_cells - active_seed_cells):
            return False

        wrong_seed_cells = _coords_to_cells(
            wrong_meta["edge_seeds"] if wrong_kind == "edge" else wrong_meta["inner_seeds"],
            boxes,
        )
        wrong_changed_cells = _changed_cells(grid, wrong_out, boxes, wrong_meta["fill_color"])
        if not (wrong_changed_cells - wrong_seed_cells):
            return False

        passable_out, _ = _simulate(grid, active_kind=active_kind, ignore_obstacles=True)
        if np.array_equal(passable_out, out):
            return False

        for r, c in inactive_seeds:
            if int(out[r, c]) != YELLOW:
                return False

        comps = _obstacle_components(grid)
        if len(comps) < 2:
            return False
        for _color, comp in comps:
            if len(comp) not in (2, 3):
                return False
            cells = {_cell_of(boxes, r, c) for r, c in comp}
            if len(cells) != 1:
                return False
            cell = next(iter(cells))
            r0, c0, r1, c1 = boxes[cell[0]][cell[1]]
            if np.any(grid[r0 : r1 + 1, c0 : c1 + 1] == YELLOW):
                return False

        return True and np.array_equal(main(grid), out)
    except Exception:
        return False


def _attempt() -> np.ndarray | None:
    n_cells = random.choice([4, 5, 5])
    cell_size = random.choice([2, 3, 3])
    grid, boxes = _build_lattice(n_cells, cell_size)
    edge_cells, inner_cells = _cell_sets(n_cells)

    winner_kind = random.choice(["edge", "inner"])
    loser_kind = "inner" if winner_kind == "edge" else "edge"

    triples = _candidate_triples(n_cells, winner_kind)
    if not triples:
        return None
    a_cell, gate_cell, far_cell, orientation = random.choice(triples)

    offset_ab = random.randrange(cell_size)
    offset_bc = random.randrange(cell_size)
    _carve_gap(grid, boxes, a_cell, gate_cell, offset_ab)
    _carve_gap(grid, boxes, gate_cell, far_cell, offset_bc)

    loser_pairs = _candidate_pairs(n_cells, loser_kind, {a_cell, gate_cell, far_cell})
    if not loser_pairs:
        return None
    loser_seed_cell, loser_reach_cell = random.choice(loser_pairs)
    _carve_gap(grid, boxes, loser_seed_cell, loser_reach_cell, random.randrange(cell_size))

    gate_color = BLUE if winner_kind == "edge" else random.choice([BLUE, RED])
    _paint_shape(grid, boxes[gate_cell[0]][gate_cell[1]], _gate_shape(cell_size, orientation), gate_color)

    reserved_for_obstacles = {gate_cell}
    winner_cells = edge_cells if winner_kind == "edge" else inner_cells
    loser_cells = inner_cells if winner_kind == "edge" else edge_cells

    winner_free = [cell for cell in winner_cells if cell not in reserved_for_obstacles]
    loser_free = [cell for cell in loser_cells if cell not in reserved_for_obstacles]
    if a_cell not in winner_free or loser_seed_cell not in loser_free:
        return None

    max_active = min(5, len(winner_free))
    max_loser = min(4, len(loser_free))
    if max_active < 2 or max_loser < 1:
        return None

    if winner_kind == "edge":
        active_count = random.randint(2, max_active)
        loser_count = random.randint(1, min(max_loser, active_count - 1))
    else:
        active_count = random.randint(2, min(max_active, 4))
        loser_count = random.randint(1, min(max_loser, active_count - 1))

    extra_active = [cell for cell in winner_free if cell != a_cell]
    if len(extra_active) < active_count - 1:
        return None
    active_seed_cells = [a_cell] + random.sample(extra_active, active_count - 1)

    extra_loser = [cell for cell in loser_free if cell != loser_seed_cell]
    if len(extra_loser) < loser_count - 1:
        return None
    loser_seed_cells = [loser_seed_cell] + random.sample(extra_loser, loser_count - 1)

    placed_cells = set(active_seed_cells) | set(loser_seed_cells) | reserved_for_obstacles

    extra_obstacle_candidates = [
        cell for cell in edge_cells + inner_cells if cell not in placed_cells and cell not in {far_cell, loser_reach_cell}
    ]
    random.shuffle(extra_obstacle_candidates)
    n_extra_obstacles = random.randint(1, min(4, len(extra_obstacle_candidates)))
    extra_obstacle_cells = extra_obstacle_candidates[:n_extra_obstacles]

    for cell in active_seed_cells + loser_seed_cells:
        _place_seed(grid, boxes[cell[0]][cell[1]], avoid=set())

    for idx, cell in enumerate(extra_obstacle_cells):
        color_choices = [BLUE, RED]
        if idx == 0 and winner_kind == "inner":
            color_choices = [RED, BLUE]
        shape = _random_obstacle_shape(cell_size)
        _paint_shape(grid, boxes[cell[0]][cell[1]], shape, random.choice(color_choices))

    if not _semantic_ok(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid new puzzle input")
