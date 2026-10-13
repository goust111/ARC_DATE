from __future__ import annotations

import random

import numpy as np


BG = 0
PIVOT = 1
ANCHOR = 2
MOVER_COLORS = (3, 4)
OUTPUT_COLOR = 7

Coord = tuple[int, int]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _rotate_ccw(point: Coord, pivot: Coord) -> Coord:
    pr, pc = pivot
    r, c = point
    dr = r - pr
    dc = c - pc
    return pr - dc, pc + dr


def _path_cells(a: Coord, b: Coord) -> list[Coord]:
    """Cells strictly between two axis-aligned coordinates."""
    ar, ac = a
    br, bc = b
    if ar == br:
        step = _sign(bc - ac)
        return [(ar, c) for c in range(ac + step, bc, step)]
    if ac == bc:
        step = _sign(br - ar)
        return [(r, ac) for r in range(ar + step, br, step)]
    raise AssertionError("expected axis-aligned path")


def _isolated(cell: Coord, others: set[Coord]) -> bool:
    return all(max(abs(cell[0] - r), abs(cell[1] - c)) > 1 for r, c in others)


def _extract_moves(grid: np.ndarray) -> list[dict[str, Coord | int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    anchors = [tuple(map(int, rc)) for rc in np.argwhere(g == ANCHOR)]
    pivots = [tuple(map(int, rc)) for rc in np.argwhere(g == PIVOT)]
    movers = [tuple(map(int, rc)) for rc in np.argwhere(np.isin(g, MOVER_COLORS))]

    assert 2 <= len(anchors) <= 3, "expected 2-3 red anchors"
    assert 2 <= len(pivots) <= 3, "expected 2-3 blue pivots"
    assert 4 <= len(movers) <= 6, "expected 4-6 scattered movers"

    for anchor in anchors:
        matches = [pivot for pivot in pivots if (anchor[0] == pivot[0]) ^ (anchor[1] == pivot[1])]
        assert len(matches) == 1, "each anchor must align with exactly one pivot"

    planned_targets: set[Coord] = set()
    moves: list[dict[str, Coord | int]] = []
    for mover in movers:
        mr, mc = mover
        matching_anchors = [anchor for anchor in anchors if (mr == anchor[0]) ^ (mc == anchor[1])]
        assert len(matching_anchors) == 1, "each mover must align with exactly one anchor"
        anchor = matching_anchors[0]
        ar, ac = anchor

        dr = _sign(mr - ar)
        dc = _sign(mc - ac)
        assert abs(dr) + abs(dc) == 1, "mover must share exactly one axis with its anchor"
        assert abs(mr - ar) + abs(mc - ac) >= 2, "mover must start at least two cells away"

        middle = (ar + dr, ac + dc)
        assert _in_bounds(g.shape, *middle), "compressed cell must stay in bounds"
        assert int(g[middle]) == BG, "compressed cell must be empty in the input"

        for cell in _path_cells(anchor, mover):
            assert cell == middle or int(g[cell]) == BG, "path from mover to anchor must be clear"

        matching_pivots = [pivot for pivot in pivots if (middle[0] == pivot[0]) ^ (middle[1] == pivot[1])]
        assert len(matching_pivots) == 1, "compressed cell must align with exactly one pivot"
        pivot = matching_pivots[0]

        target = _rotate_ccw(middle, pivot)
        assert _in_bounds(g.shape, *target), "rotated target must stay in bounds"
        assert target not in planned_targets, "rotated targets must not collide"
        assert target not in set(anchors) and target not in set(pivots), "rotated targets must land on background"
        planned_targets.add(target)

        moves.append(
            {
                "mover": mover,
                "anchor": anchor,
                "middle": middle,
                "pivot": pivot,
                "target": target,
                "color": int(g[mover]),
            }
        )

    return moves


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress each scattered pixel next to its red anchor, then rotate that intermediate position around the aligned blue pivot and repaint it orange.

    Input structure:
    - The grid has a black background with isolated red anchor pixels, isolated blue pivot pixels, and isolated green/yellow mover pixels.
    - Every mover shares exactly one row or column with a red anchor, and every anchor shares exactly one row or column with a blue pivot.
    - The cell adjacent to the anchor along the mover's ray is empty, and that compressed cell shares exactly one row or column with a blue pivot.

    Rule:
    - Keep all red anchors and blue pivots fixed.
    - For each green or yellow mover, erase its original location and slide it inward to the adjacent cell next to its unique aligned red anchor.
    - Do not keep that intermediate cell; instead rotate it 90 degrees counterclockwise around its unique aligned blue pivot and paint the rotated target orange.

    Key insight:
    - The transform is a two-step pipeline: the temporary compression next to the red anchor changes which pivot-relative offset gets rotated, so solving the task requires "slide first, rotate second" rather than rotating the original scattered pixels.
    """
    g = np.asarray(grid, dtype=int)
    moves = _extract_moves(g)

    out = g.copy()
    for move in moves:
        mr, mc = move["mover"]
        out[mr, mc] = BG
    for move in moves:
        tr, tc = move["target"]
        assert int(out[tr, tc]) == BG, "rotated target must land on background"
        out[tr, tc] = OUTPUT_COLOR
    return out


# --------------- generator ---------------

GRID_SIDE_RANGE = (13, 18)
PAIR_DISTANCE_RANGE = (2, 4)
MAX_GENERATION_ATTEMPTS = 800
PAIR_ATTEMPTS = 240


def _sample_pair_specs() -> list[tuple[str, int]]:
    if random.random() < 0.35:
        specs = [("H", 2), ("V", 2)]
        random.shuffle(specs)
        return specs

    total_movers = random.randint(4, 6)
    counts = [1, 1, 1]
    while sum(counts) < total_movers:
        idxs = [i for i, count in enumerate(counts) if count < 2]
        counts[random.choice(idxs)] += 1
    orientations = ["H", "V", random.choice(("H", "V"))]
    random.shuffle(orientations)
    return list(zip(orientations, counts))


def _try_horizontal_system(
    shape: tuple[int, int],
    count: int,
    input_cells: set[Coord],
    target_cells: set[Coord],
    used_rows: set[int],
    used_cols: set[int],
    anchor_cols: set[int],
    pivot_cols: set[int],
    forbid_anchor_rows: set[int],
    forbid_anchor_cols: set[int],
    forbid_pivot_rows: set[int],
    forbid_pivot_cols: set[int],
) -> dict | None:
    h, w = shape
    row_candidates = [
        r
        for r in range(2, h - 2)
        if r not in used_rows and r not in forbid_anchor_rows and r not in forbid_pivot_rows
    ]
    if not row_candidates:
        return None

    row = random.choice(row_candidates)
    fixed_row_cells = {cell for cell in input_cells if cell[0] == row}
    col_pairs = []
    for ac in range(2, w - 2):
        if ac in used_cols or ac in forbid_anchor_cols:
            continue
        for pc in range(2, w - 2):
            if pc == ac or pc in used_cols or pc in forbid_pivot_cols:
                continue
            if not (PAIR_DISTANCE_RANGE[0] <= abs(ac - pc) <= PAIR_DISTANCE_RANGE[1]):
                continue
            col_pairs.append((ac, pc))
    random.shuffle(col_pairs)

    for ac, pc in col_pairs:
        anchor = (row, ac)
        pivot = (row, pc)
        if anchor in target_cells or pivot in target_cells:
            continue
        if not _isolated(anchor, input_cells) or not _isolated(pivot, input_cells | {anchor}):
            continue

        side_options = []
        for side in (-1, 1):
            landing = (row, ac + side)
            if landing[1] in pivot_cols:
                continue
            target = _rotate_ccw(landing, pivot)
            if not _in_bounds(shape, *target):
                continue
            if target in target_cells or target in input_cells or target in {anchor, pivot}:
                continue

            dist_options = []
            max_dist = ac if side < 0 else (w - 1 - ac)
            for dist in range(2, max_dist + 1):
                mover = (row, ac + side * dist)
                if mover[1] in anchor_cols or mover[1] in forbid_anchor_cols:
                    continue
                if mover in target_cells:
                    continue
                if not _isolated(mover, input_cells | {anchor, pivot}):
                    continue
                segment = _path_cells(anchor, mover)
                if any(cell in fixed_row_cells or cell == pivot for cell in segment):
                    continue
                dist_options.append(dist)

            if dist_options:
                side_options.append((side, landing, target, dist_options))

        if len(side_options) < count:
            continue
        if count == 2 and {opt[0] for opt in side_options} != {-1, 1}:
            continue

        if count == 1:
            chosen = [random.choice(side_options)]
        else:
            chosen = [opt for opt in side_options if opt[0] in (-1, 1)]

        movers = []
        local_cells = {anchor, pivot}
        ok = True
        for side, landing, target, dist_options in chosen:
            valid = []
            for dist in dist_options:
                mover = (row, ac + side * dist)
                if mover in local_cells or not _isolated(mover, local_cells):
                    continue
                valid.append(dist)
            if not valid:
                ok = False
                break
            dist = random.choice(valid)
            mover = (row, ac + side * dist)
            movers.append({"mover": mover, "landing": landing, "target": target})
            local_cells.add(mover)
        if not ok:
            continue

        return {
            "orientation": "H",
            "anchor": anchor,
            "pivot": pivot,
            "movers": movers,
        }
    return None


def _try_vertical_system(
    shape: tuple[int, int],
    count: int,
    input_cells: set[Coord],
    target_cells: set[Coord],
    used_rows: set[int],
    used_cols: set[int],
    anchor_rows: set[int],
    pivot_rows: set[int],
    forbid_anchor_rows: set[int],
    forbid_anchor_cols: set[int],
    forbid_pivot_rows: set[int],
    forbid_pivot_cols: set[int],
) -> dict | None:
    h, w = shape
    col_candidates = [
        c
        for c in range(2, w - 2)
        if c not in used_cols and c not in forbid_anchor_cols and c not in forbid_pivot_cols
    ]
    if not col_candidates:
        return None

    col = random.choice(col_candidates)
    fixed_col_cells = {cell for cell in input_cells if cell[1] == col}
    row_pairs = []
    for ar in range(2, h - 2):
        if ar in used_rows or ar in forbid_anchor_rows:
            continue
        for pr in range(2, h - 2):
            if pr == ar or pr in used_rows or pr in forbid_pivot_rows:
                continue
            if not (PAIR_DISTANCE_RANGE[0] <= abs(ar - pr) <= PAIR_DISTANCE_RANGE[1]):
                continue
            row_pairs.append((ar, pr))
    random.shuffle(row_pairs)

    for ar, pr in row_pairs:
        anchor = (ar, col)
        pivot = (pr, col)
        if anchor in target_cells or pivot in target_cells:
            continue
        if not _isolated(anchor, input_cells) or not _isolated(pivot, input_cells | {anchor}):
            continue

        side_options = []
        for side in (-1, 1):
            landing = (ar + side, col)
            if landing[0] in pivot_rows:
                continue
            target = _rotate_ccw(landing, pivot)
            if not _in_bounds(shape, *target):
                continue
            if target in target_cells or target in input_cells or target in {anchor, pivot}:
                continue

            dist_options = []
            max_dist = ar if side < 0 else (h - 1 - ar)
            for dist in range(2, max_dist + 1):
                mover = (ar + side * dist, col)
                if mover[0] in anchor_rows or mover[0] in forbid_anchor_rows:
                    continue
                if mover in target_cells:
                    continue
                if not _isolated(mover, input_cells | {anchor, pivot}):
                    continue
                segment = _path_cells(anchor, mover)
                if any(cell in fixed_col_cells or cell == pivot for cell in segment):
                    continue
                dist_options.append(dist)

            if dist_options:
                side_options.append((side, landing, target, dist_options))

        if len(side_options) < count:
            continue
        if count == 2 and {opt[0] for opt in side_options} != {-1, 1}:
            continue

        if count == 1:
            chosen = [random.choice(side_options)]
        else:
            chosen = [opt for opt in side_options if opt[0] in (-1, 1)]

        movers = []
        local_cells = {anchor, pivot}
        ok = True
        for side, landing, target, dist_options in chosen:
            valid = []
            for dist in dist_options:
                mover = (ar + side * dist, col)
                if mover in local_cells or not _isolated(mover, local_cells):
                    continue
                valid.append(dist)
            if not valid:
                ok = False
                break
            dist = random.choice(valid)
            mover = (ar + side * dist, col)
            movers.append({"mover": mover, "landing": landing, "target": target})
            local_cells.add(mover)
        if not ok:
            continue

        return {
            "orientation": "V",
            "anchor": anchor,
            "pivot": pivot,
            "movers": movers,
        }
    return None


def _place_systems(shape: tuple[int, int], specs: list[tuple[str, int]]) -> list[dict] | None:
    input_cells: set[Coord] = set()
    target_cells: set[Coord] = set()
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    anchor_rows: set[int] = set()
    anchor_cols: set[int] = set()
    pivot_rows: set[int] = set()
    pivot_cols: set[int] = set()
    forbid_anchor_rows: set[int] = set()
    forbid_anchor_cols: set[int] = set()
    forbid_pivot_rows: set[int] = set()
    forbid_pivot_cols: set[int] = set()

    systems: list[dict] = []
    for orientation, count in specs:
        system = None
        for _ in range(PAIR_ATTEMPTS):
            if orientation == "H":
                system = _try_horizontal_system(
                    shape,
                    count,
                    input_cells,
                    target_cells,
                    used_rows,
                    used_cols,
                    anchor_cols,
                    pivot_cols,
                    forbid_anchor_rows,
                    forbid_anchor_cols,
                    forbid_pivot_rows,
                    forbid_pivot_cols,
                )
            else:
                system = _try_vertical_system(
                    shape,
                    count,
                    input_cells,
                    target_cells,
                    used_rows,
                    used_cols,
                    anchor_rows,
                    pivot_rows,
                    forbid_anchor_rows,
                    forbid_anchor_cols,
                    forbid_pivot_rows,
                    forbid_pivot_cols,
                )
            if system is not None:
                break
        if system is None:
            return None

        anchor = system["anchor"]
        pivot = system["pivot"]
        systems.append(system)
        input_cells.update({anchor, pivot})
        target_cells.update(mover["target"] for mover in system["movers"])
        used_rows.update({anchor[0], pivot[0]})
        used_cols.update({anchor[1], pivot[1]})
        anchor_rows.add(anchor[0])
        anchor_cols.add(anchor[1])
        pivot_rows.add(pivot[0])
        pivot_cols.add(pivot[1])

        for mover in system["movers"]:
            input_cells.add(mover["mover"])
            if system["orientation"] == "H":
                forbid_anchor_cols.add(mover["mover"][1])
                forbid_pivot_cols.add(mover["landing"][1])
            else:
                forbid_anchor_rows.add(mover["mover"][0])
                forbid_pivot_rows.add(mover["landing"][0])

    return systems


def _render_input(shape: tuple[int, int], systems: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    movers = [mover for system in systems for mover in system["movers"]]
    colors = [random.choice(MOVER_COLORS) for _ in movers]
    if len(movers) >= 2 and len(set(colors)) == 1:
        colors[random.randrange(len(colors))] = MOVER_COLORS[0] if colors[0] == MOVER_COLORS[1] else MOVER_COLORS[1]
    random.shuffle(colors)

    color_iter = iter(colors)
    for system in systems:
        ar, ac = system["anchor"]
        pr, pc = system["pivot"]
        grid[ar, ac] = ANCHOR
        grid[pr, pc] = PIVOT
        for mover in system["movers"]:
            mr, mc = mover["mover"]
            grid[mr, mc] = next(color_iter)
    return grid


def _expected_output(inp: np.ndarray, systems: list[dict]) -> np.ndarray:
    out = inp.copy()
    for system in systems:
        for mover in system["movers"]:
            mr, mc = mover["mover"]
            tr, tc = mover["target"]
            out[mr, mc] = BG
            out[tr, tc] = OUTPUT_COLOR
    return out


def _input_pixels_are_isolated(grid: np.ndarray) -> bool:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(grid != BG)]
    for i, a in enumerate(coords):
        for b in coords[i + 1 :]:
            if max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1:
                return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)
    specs = _sample_pair_specs()
    systems = _place_systems((h, w), specs)
    if systems is None:
        return None

    inp = _render_input((h, w), systems)
    if not _input_pixels_are_isolated(inp):
        return None

    expected = _expected_output(inp, systems)
    try:
        observed = main(inp)
        _extract_moves(inp)
    except AssertionError:
        return None
    if not np.array_equal(observed, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
