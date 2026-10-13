from __future__ import annotations

from collections import Counter, defaultdict, deque
import random
import sys

import numpy as np

_ARC_UTILS_DIR = "/home/lixinyu/snap/new/solvers"
if _ARC_UTILS_DIR not in sys.path:
    sys.path.append(_ARC_UTILS_DIR)

from arc_utils import bbox_h, bbox_w, find_components, get_bg, in_bounds, neighbors4


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
COLOR_POOL = list(range(10))
GEN_RETRIES = 400
SIZE_RANGE = (16, 26)
CASE_TYPES = ("diag", "horiz", "horiz", "horiz", "vert", "vert", "vert")


def _sign(value: int) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _frame_mask(shape: tuple[int, int], frame_cells: set[tuple[int, int]], marker_cells: set[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in frame_cells | marker_cells:
        mask[r, c] = True
    return mask


def _find_doorways(grid: np.ndarray) -> tuple[int, list[dict], set[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    by_color: defaultdict[int, list[dict]] = defaultdict(list)
    for comp in comps:
        by_color[int(comp["color"])].append(comp)

    candidate_colors = []
    for color, items in by_color.items():
        if len(items) != 2:
            continue
        if all(int(comp["area"]) == 7 and bbox_h(comp["bbox"]) == 3 and bbox_w(comp["bbox"]) == 3 for comp in items):
            candidate_colors.append(color)

    assert len(candidate_colors) == 1, "expected exactly one shared doorway-frame color"
    frame_color = int(candidate_colors[0])
    frame_comps = sorted(by_color[frame_color], key=lambda comp: tuple(map(int, comp["bbox"])))

    doorways: list[dict] = []
    marker_cells: set[tuple[int, int]] = set()
    for comp in frame_comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        center = (r0 + 1, c0 + 1)
        marker_color = int(g[center])
        assert marker_color not in (bg, frame_color), "doorway center must be a non-frame marker"

        expected = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
        expected.remove(center)

        gap_candidates = [(rr, cc) for rr, cc in neighbors4(*center) if int(g[rr, cc]) == bg]
        assert len(gap_candidates) == 1, "each doorway marker must have exactly one background gap"
        gap = gap_candidates[0]
        expected.remove(gap)

        comp_cells = {tuple(map(int, coord)) for coord in comp["coords"]}
        assert comp_cells == expected, "doorway frame must be a 3x3 ring with one side gap"

        dr = gap[0] - center[0]
        dc = gap[1] - center[1]
        exterior = (center[0] + 2 * dr, center[1] + 2 * dc)
        assert in_bounds(g.shape, *exterior), "exterior doorway cell must stay in bounds"
        assert int(g[exterior]) == bg, "exterior doorway cell must be background"

        doorways.append(
            {
                "center": center,
                "gap": gap,
                "exterior": exterior,
                "marker_color": marker_color,
                "bbox": (r0, c0, r1, c1),
            }
        )
        marker_cells.add(center)

    assert len(doorways) == 2
    return frame_color, doorways, marker_cells


def _reachable(
    passable: np.ndarray,
    seed: tuple[int, int],
    moves: list[tuple[int, int]],
    bounds: tuple[int, int, int, int],
) -> np.ndarray:
    r0, c0, r1, c1 = bounds

    def valid(r: int, c: int) -> bool:
        return r0 <= r <= r1 and c0 <= c <= c1 and bool(passable[r, c])

    assert valid(*seed)
    seen = np.zeros(passable.shape, dtype=bool)
    seen[seed] = True
    q = deque([seed])
    while q:
        r, c = q.popleft()
        for dr, dc in moves:
            rr, cc = r + dr, c + dc
            if valid(rr, cc) and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))
    return seen


def _corridor_mask(
    grid: np.ndarray,
    frame_color: int,
    doorways: list[dict],
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int], tuple[int, int, int, int]]:
    start = tuple(map(int, doorways[0]["exterior"]))
    end = tuple(map(int, doorways[1]["exterior"]))
    row_step = _sign(end[0] - start[0])
    col_step = _sign(end[1] - start[1])
    assert row_step != 0 or col_step != 0, "doorways must be distinct"

    moves = []
    if row_step:
        moves.append((row_step, 0))
    if col_step:
        moves.append((0, col_step))

    r0, r1 = sorted((start[0], end[0]))
    c0, c1 = sorted((start[1], end[1]))
    bounds = (r0, c0, r1, c1)

    passable = np.asarray(grid, dtype=int) != int(frame_color)
    forward = _reachable(passable, start, moves, bounds)
    backward = _reachable(passable, end, [(-dr, -dc) for dr, dc in moves], bounds)
    corridor = forward & backward
    assert corridor[start] and corridor[end], "exterior doorway cells must be connected"
    for doorway in doorways:
        corridor[tuple(map(int, doorway["gap"]))] = True
    return corridor, start, end, bounds


def _dust_mask(
    grid: np.ndarray,
    bg: int,
    frame_color: int,
    marker_cells: set[tuple[int, int]],
) -> np.ndarray:
    dust = (np.asarray(grid, dtype=int) != int(bg)) & (np.asarray(grid, dtype=int) != int(frame_color))
    for r, c in marker_cells:
        dust[r, c] = False
    return dust


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the hidden doorway corridor, let only corridor dust vote, and repaint that corridor with the winning color.

    Input structure:
    - The grid has a dominant background, two same-color 3x3 doorway frames, and one non-frame marker at each frame center.
    - Each center marker has exactly one background 4-neighbor gap, and the next cell outward is the exterior doorway cell.
    - Other non-background cells are isolated dust pixels that are not part of the doorway objects.

    Rule:
    - Identify the two doorway frames and treat only their frame-color cells as obstacles.
    - Between the two exterior doorway cells, collect every background or dust cell that lies on some row/column monotone shortest path, then also include the two doorway gaps.
    - Count dust colors only on that hidden corridor, erase all dust everywhere, and repaint the whole corridor with the unique winning vote color while keeping both frames and center markers unchanged.

    Key insight:
    - The voting region is implicit: you must reconstruct the full union of monotone shortest paths before you know which dust pixels are allowed to vote.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    frame_color, doorways, marker_cells = _find_doorways(g)
    corridor, _start, _end, _bounds = _corridor_mask(g, frame_color, doorways)
    dust = _dust_mask(g, bg, frame_color, marker_cells)

    votes = Counter(map(int, g[corridor & dust]))
    ranked = votes.most_common()
    assert ranked, "expected at least one corridor dust voter"
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "corridor vote winner must be unique"
    winner = int(ranked[0][0])

    out = g.copy()
    out[dust] = bg
    paint_mask = corridor & (g != frame_color)
    for r, c in marker_cells:
        paint_mask[r, c] = False
    out[paint_mask] = winner
    return out


def _monotone_key(start: tuple[int, int], row_step: int, col_step: int, cell: tuple[int, int]) -> tuple[int, int]:
    r, c = cell
    rr = 0 if row_step == 0 else row_step * (r - start[0])
    cc = 0 if col_step == 0 else col_step * (c - start[1])
    return rr, cc


def _cells_form_single_monotone_path(
    start: tuple[int, int],
    end: tuple[int, int],
    cells: list[tuple[int, int]],
) -> bool:
    row_step = _sign(end[0] - start[0])
    col_step = _sign(end[1] - start[1])
    if row_step == 0 or col_step == 0:
        return True
    keyed = [_monotone_key(start, row_step, col_step, cell) for cell in cells]
    for i in range(len(keyed)):
        for j in range(i + 1, len(keyed)):
            a = keyed[i]
            b = keyed[j]
            if (a[0] - b[0]) * (a[1] - b[1]) < 0:
                return False
    return True


def _interior_open_cells(bounds: tuple[int, int, int, int], corridor: np.ndarray, object_mask: np.ndarray) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bounds
    cells = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if corridor[r, c] or object_mask[r, c]:
                continue
            cells.append((r, c))
    return cells


def _filtered_positions(
    grid_shape: tuple[int, int],
    candidates: list[tuple[int, int]],
    blocked: set[tuple[int, int]],
    chosen: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    out = []
    for r, c in candidates:
        if (r, c) in blocked:
            continue
        if any(max(abs(r - rr), abs(c - cc)) <= 1 for rr, cc in chosen):
            continue
        touches_object = False
        for rr, cc in [(r, c), *list(neighbors4(r, c))]:
            if in_bounds(grid_shape, rr, cc) and (rr, cc) in blocked:
                touches_object = True
                break
        if not touches_object:
            out.append((r, c))
    random.shuffle(out)
    return out


def _pick_cells(
    grid_shape: tuple[int, int],
    base_candidates: list[tuple[int, int]],
    blocked: set[tuple[int, int]],
    chosen: list[tuple[int, int]],
    count: int,
) -> list[tuple[int, int]] | None:
    available = _filtered_positions(grid_shape, base_candidates, blocked, chosen)
    if len(available) < count:
        return None
    taken: list[tuple[int, int]] = []
    for cell in available:
        if any(max(abs(cell[0] - rr), abs(cell[1] - cc)) <= 1 for rr, cc in chosen + taken):
            continue
        taken.append(cell)
        if len(taken) == count:
            return taken
    return None


def _place_frame(
    grid: np.ndarray,
    center: tuple[int, int],
    direction: tuple[int, int],
    frame_color: int,
    marker_color: int,
) -> dict:
    r, c = center
    dr, dc = direction
    gap = (r + dr, c + dc)
    exterior = (r + 2 * dr, c + 2 * dc)

    grid[r - 1 : r + 2, c - 1 : c + 2] = frame_color
    grid[r, c] = marker_color
    grid[gap] = int(grid[exterior])
    return {"center": center, "gap": gap, "exterior": exterior}


def _valid_center(shape: tuple[int, int], center: tuple[int, int]) -> bool:
    rows, cols = shape
    r, c = center
    return 2 <= r <= rows - 3 and 2 <= c <= cols - 3


def _sample_doors(rows: int, cols: int, case_type: str) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    shape = (rows, cols)
    ext_cells = [(r, c) for r in range(rows) for c in range(cols)]
    for _ in range(300):
        if case_type == "diag":
            ext_a = random.choice(ext_cells)
            ext_b = random.choice(ext_cells)
            if ext_a[0] == ext_b[0] or ext_a[1] == ext_b[1]:
                continue
        elif case_type == "horiz":
            if cols < 17:
                continue
            row = random.randint(2, rows - 3)
            c1 = random.randint(4, cols - 13)
            c2 = random.randint(c1 + 8, cols - 5)
            ext_a, ext_b = (row, c1), (row, c2)
            center_a = (row, c1 - 2)
            center_b = (row, c2 + 2)
            return [center_a, center_b], [(0, 1), (0, -1)]
        else:
            if rows < 17:
                continue
            col = random.randint(2, cols - 3)
            r1 = random.randint(4, rows - 13)
            r2 = random.randint(r1 + 8, rows - 5)
            ext_a, ext_b = (r1, col), (r2, col)
            center_a = (r1 - 2, col)
            center_b = (r2 + 2, col)
            return [center_a, center_b], [(1, 0), (-1, 0)]

        dirs_a = random.sample(DIR4, len(DIR4))
        dirs_b = random.sample(DIR4, len(DIR4))
        for dir_a in dirs_a:
            center_a = (ext_a[0] - 2 * dir_a[0], ext_a[1] - 2 * dir_a[1])
            if not _valid_center(shape, center_a):
                continue
            bbox_a = (center_a[0] - 1, center_a[1] - 1, center_a[0] + 1, center_a[1] + 1)
            for dir_b in dirs_b:
                center_b = (ext_b[0] - 2 * dir_b[0], ext_b[1] - 2 * dir_b[1])
                if not _valid_center(shape, center_b):
                    continue
                if max(abs(center_a[0] - center_b[0]), abs(center_a[1] - center_b[1])) < 5:
                    continue
                bbox_b = (center_b[0] - 1, center_b[1] - 1, center_b[0] + 1, center_b[1] + 1)
                if not (bbox_a[2] < bbox_b[0] or bbox_b[2] < bbox_a[0] or bbox_a[3] < bbox_b[1] or bbox_b[3] < bbox_a[1]):
                    continue
                return [center_a, center_b], [dir_a, dir_b]
    return None


def _render_base_case(
    rows: int,
    cols: int,
    case_type: str,
    bg: int,
    frame_color: int,
    marker_color: int,
) -> tuple[np.ndarray, list[dict], np.ndarray, tuple[int, int], tuple[int, int], tuple[int, int, int, int], set[tuple[int, int]], bool] | None:
    sampled = _sample_doors(rows, cols, case_type)
    if sampled is None:
        return None
    centers, directions = sampled

    grid = np.full((rows, cols), bg, dtype=int)
    raw_doors = []
    for center, direction in zip(centers, directions):
        raw_doors.append(_place_frame(grid, center, direction, frame_color, marker_color))

    frame_cells = {(r, c) for r, c in np.argwhere(grid == frame_color)}
    marker_cells = {(r, c) for r, c in np.argwhere(grid == marker_color)}
    assert len(marker_cells) == 2

    doorways = []
    for item in raw_doors:
        doorways.append(
            {
                "center": tuple(map(int, item["center"])),
                "gap": tuple(map(int, item["gap"])),
                "exterior": tuple(map(int, item["exterior"])),
                "marker_color": marker_color,
            }
        )

    try:
        corridor, start, end, bounds = _corridor_mask(grid, frame_color, doorways)
    except AssertionError:
        return None
    if int(corridor.sum()) < 5:
        return None

    object_mask = _frame_mask(grid.shape, frame_cells, marker_cells)
    interior_open = _interior_open_cells(bounds, corridor, object_mask)
    diagonal_case = start[0] != end[0] and start[1] != end[1]
    if diagonal_case and len(interior_open) < 2:
        return None
    return grid, doorways, corridor, start, end, bounds, frame_cells | marker_cells, diagonal_case


def _choose_incomparable_pair(
    corridor_cells: list[tuple[int, int]],
    start: tuple[int, int],
    end: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    row_step = _sign(end[0] - start[0])
    col_step = _sign(end[1] - start[1])
    keyed = [(cell, _monotone_key(start, row_step, col_step, cell)) for cell in corridor_cells]
    random.shuffle(keyed)
    for cell_a, key_a in keyed:
        for cell_b, key_b in keyed:
            if cell_a == cell_b:
                continue
            if (key_a[0] - key_b[0]) * (key_a[1] - key_b[1]) < 0:
                return cell_a, cell_b
    return None


def _paint_dust(grid: np.ndarray, color: int, cells: list[tuple[int, int]]) -> None:
    for r, c in cells:
        grid[r, c] = color


def _analyze_instance(grid: np.ndarray) -> dict:
    bg = get_bg(grid)
    frame_color, doorways, marker_cells = _find_doorways(grid)
    corridor, start, end, bounds = _corridor_mask(grid, frame_color, doorways)
    dust = _dust_mask(grid, bg, frame_color, marker_cells)
    votes = Counter(map(int, grid[corridor & dust]))
    return {
        "bg": bg,
        "frame_color": frame_color,
        "doorways": doorways,
        "marker_cells": marker_cells,
        "corridor": corridor,
        "start": start,
        "end": end,
        "bounds": bounds,
        "dust": dust,
        "votes": votes,
    }


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        info = _analyze_instance(grid)
        out = main(grid)
    except Exception:
        return False

    if out.shape != grid.shape:
        return False

    votes = info["votes"]
    ranked = votes.most_common()
    if not ranked or (len(ranked) > 1 and ranked[0][1] == ranked[1][1]):
        return False
    winner = ranked[0][0]

    dust = info["dust"]
    total_counts = Counter(map(int, grid[dust]))
    if len(total_counts) < 3:
        return False
    global_top = total_counts.most_common(1)[0][0]
    if global_top == winner:
        return False

    corridor_cells = [tuple(map(int, cell)) for cell in np.argwhere(info["corridor"] & dust)]
    outside_cells = [tuple(map(int, cell)) for cell in np.argwhere((~info["corridor"]) & dust)]
    if not corridor_cells or not outside_cells:
        return False

    start, end = info["start"], info["end"]
    diagonal_case = start[0] != end[0] and start[1] != end[1]
    if diagonal_case:
        winner_cells = [cell for cell in corridor_cells if int(grid[cell]) == int(winner)]
        if _cells_form_single_monotone_path(start, end, winner_cells):
            return False
        frame_mask = grid == info["frame_color"]
        marker_mask = np.zeros(grid.shape, dtype=bool)
        for r, c in info["marker_cells"]:
            marker_mask[r, c] = True
        object_mask = frame_mask | marker_mask
        r0, c0, r1, c1 = info["bounds"]
        interior_false_voters = 0
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if dust[r, c] and not info["corridor"][r, c] and not object_mask[r, c]:
                    interior_false_voters += 1
        if interior_false_voters == 0:
            return False

    frame_color = info["frame_color"]
    for r, c in info["marker_cells"]:
        if out[r, c] != grid[r, c]:
            return False
    if not np.array_equal(out[grid == frame_color], grid[grid == frame_color]):
        return False
    if np.any(out[dust & (~info["corridor"])] != info["bg"]):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(*SIZE_RANGE)
    cols = random.randint(*SIZE_RANGE)
    case_type = random.choice(CASE_TYPES)

    colors = random.sample(COLOR_POOL, 6)
    bg, frame_color, marker_color, winner_color, runner_color, global_color = colors

    rendered = _render_base_case(rows, cols, case_type, bg, frame_color, marker_color)
    if rendered is None:
        return None
    grid, _doorways, corridor, start, end, bounds, blocked_cells, diagonal_case = rendered

    corridor_candidates = [tuple(map(int, cell)) for cell in np.argwhere(corridor)]
    if diagonal_case:
        pair = _choose_incomparable_pair(corridor_candidates, start, end)
        if pair is None:
            return None
        winner_seed = list(pair)
    else:
        winner_seed = random.sample(corridor_candidates, 2)

    chosen: list[tuple[int, int]] = []
    blocked = set(blocked_cells)

    winner_cells = _pick_cells(grid.shape, winner_seed + corridor_candidates, blocked, chosen, 3)
    if winner_cells is None:
        return None
    chosen.extend(winner_cells)

    runner_cells = _pick_cells(grid.shape, corridor_candidates, blocked, chosen, 2)
    if runner_cells is None:
        return None
    chosen.extend(runner_cells)

    outside_candidates = [tuple(map(int, cell)) for cell in np.argwhere(~corridor)]
    global_cells = _pick_cells(grid.shape, outside_candidates, blocked, chosen, 4)
    if global_cells is None:
        return None
    chosen.extend(global_cells)

    extra_color = next(color for color in COLOR_POOL if color not in {bg, frame_color, marker_color, winner_color, runner_color, global_color})
    extra_cells = _pick_cells(grid.shape, outside_candidates, blocked, chosen, 2)
    if extra_cells is None:
        return None
    chosen.extend(extra_cells)

    if diagonal_case:
        interior_open = _interior_open_cells(bounds, corridor, _frame_mask(grid.shape, {cell for cell in blocked if grid[cell] == frame_color}, {cell for cell in blocked if grid[cell] == marker_color}))
        interior_false = _pick_cells(grid.shape, interior_open, blocked, chosen, 1)
        if interior_false is None:
            return None
        chosen.extend(interior_false)
        global_cells[0] = interior_false[0]

    _paint_dust(grid, winner_color, winner_cells)
    _paint_dust(grid, runner_color, runner_cells)
    _paint_dust(grid, global_color, global_cells)
    _paint_dust(grid, extra_color, extra_cells)

    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid doorway-vote corridor input")
