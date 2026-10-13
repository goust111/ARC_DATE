import random

import numpy as np


BLACK = 0
MARKER_COLORS = (1, 2, 3, 4)
GREY = 5
GEN_RETRY_LIMIT = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _touches_edge(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    rows = coords[:, 0]
    cols = coords[:, 1]
    return bool(np.any((rows == 0) | (rows == h - 1) | (cols == 0) | (cols == w - 1)))


def _candidate_wall_mask(shape: tuple[int, int], row_period: int, col_period: int, grey_coords: np.ndarray) -> np.ndarray:
    h, w = shape
    row_ids = np.arange(h)[:, None] % row_period
    col_ids = np.arange(w)[None, :] % col_period
    class_ids = row_ids * col_period + col_ids
    residue_ids = np.unique((grey_coords[:, 0] % row_period) * col_period + (grey_coords[:, 1] % col_period))
    return np.isin(class_ids, residue_ids)


def _render_output_from_repaired(repaired: np.ndarray) -> np.ndarray | None:
    passable = repaired != GREY
    comps = _components(passable)
    marked_components = 0
    unmarked_components = 0
    out = repaired.copy()
    for coords in comps:
        vals = repaired[coords[:, 0], coords[:, 1]]
        marker_vals = vals[(vals != BLACK) & (vals != GREY)]
        if marker_vals.size > 1:
            return None
        if marker_vals.size == 1:
            marked_components += 1
            if _touches_edge(coords, repaired.shape):
                return None
            if int(np.count_nonzero(vals == BLACK)) == 0:
                return None
            fill_color = int(marker_vals[0])
            black_cells = coords[vals == BLACK]
            out[black_cells[:, 0], black_cells[:, 1]] = fill_color
        else:
            unmarked_components += 1
    if marked_components < 1 or unmarked_components < 1:
        return None
    return out


def _analyze_candidate(grid: np.ndarray, row_period: int, col_period: int) -> dict | None:
    grey_coords = np.argwhere(grid == GREY)
    if grey_coords.size == 0:
        return None
    wall_mask = _candidate_wall_mask(grid.shape, row_period, col_period, grey_coords)
    marker_mask = (grid != BLACK) & (grid != GREY)
    if np.any(wall_mask & marker_mask):
        return None
    repaired = np.asarray(grid, dtype=int).copy()
    repaired[wall_mask & ~marker_mask] = GREY
    output = _render_output_from_repaired(repaired)
    if output is None:
        return None
    return {
        "row_period": int(row_period),
        "col_period": int(col_period),
        "area": int(row_period * col_period),
        "output": output,
    }


def _valid_candidates(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[dict] = []
    for row_period in range(1, h + 1):
        for col_period in range(1, w + 1):
            cand = _analyze_candidate(g, row_period, col_period)
            if cand is not None:
                candidates.append(cand)
    return candidates


def _solve(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    candidates = _valid_candidates(g)
    assert candidates, "no period/topology candidate satisfies the task semantics"
    min_area = min(cand["area"] for cand in candidates)
    best = [cand for cand in candidates if cand["area"] == min_area]
    signatures = {cand["output"].tobytes(): cand for cand in best}
    assert len(signatures) == 1, "minimal valid period candidates must imply one unique output"
    chosen = next(iter(signatures.values()))
    return chosen["output"], (chosen["row_period"], chosen["col_period"])


def main(grid: np.ndarray) -> np.ndarray:
    """Repair the periodic grey wallpaper, then flood-fill each enclosed marked room with its marker color.

    Input structure:
    - A black background is partitioned by a grey periodic wall pattern made of repeated residue classes.
    - Some grey wall pixels were erased to black, but only at positions that should belong to the wallpaper.
    - Several rooms contain exactly one colored marker pixel, while at least one room has no marker.

    Rule:
    - Ignore black cells and markers when inferring the smallest row/column period whose grey residue classes are consistent.
    - Any residue class that still shows at least one grey pixel must be restored to grey everywhere, except marker cells themselves.
    - Treat the repaired grey pixels as walls and split the remaining non-grey cells into 4-connected room components.
    - If a room contains one colored marker, recolor every black cell in that room to the marker color; unmarked rooms stay black.

    Key insight:
    - Filling first is wrong because erased wall gaps merge neighboring rooms; only after periodic wall repair do the room boundaries become topologically correct.
    """
    output, _ = _solve(grid)
    return output


def _room_cells(room_r: int, room_c: int, row_period: int, col_period: int) -> tuple[slice, slice]:
    r0 = room_r * row_period + 1
    c0 = room_c * col_period + 1
    return slice(r0, r0 + row_period - 1), slice(c0, c0 + col_period - 1)


def _base_layout(
    room_rows: int, room_cols: int, row_period: int, col_period: int
) -> tuple[np.ndarray, np.ndarray]:
    h = room_rows * row_period + 1
    w = room_cols * col_period + 1
    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    wall_mask = (rows % row_period == 0) | (cols % col_period == 0)
    room_ids = np.full((h, w), -1, dtype=int)
    rid = 0
    for rr in range(room_rows):
        for cc in range(room_cols):
            rs, cs = _room_cells(rr, cc, row_period, col_period)
            room_ids[rs, cs] = rid
            rid += 1
    return wall_mask, room_ids


def _room_adjacencies(
    room_rows: int, room_cols: int, row_period: int, col_period: int
) -> list[dict]:
    edges: list[dict] = []
    for rr in range(room_rows):
        for cc in range(room_cols):
            room_id = rr * room_cols + cc
            if cc + 1 < room_cols:
                wall_col = (cc + 1) * col_period
                cells = [(r, wall_col) for r in range(rr * row_period + 1, rr * row_period + row_period)]
                edges.append(
                    {
                        "rooms": (room_id, room_id + 1),
                        "cells": cells,
                    }
                )
            if rr + 1 < room_rows:
                wall_row = (rr + 1) * row_period
                cells = [(wall_row, c) for c in range(cc * col_period + 1, cc * col_period + col_period)]
                edges.append(
                    {
                        "rooms": (room_id, room_id + room_cols),
                        "cells": cells,
                    }
                )
    return edges


def _choose_marked_rooms(total_rooms: int) -> list[int]:
    max_marked = min(4, total_rooms - 1)
    min_marked = 1 if total_rooms == 2 else 2
    marked_count = random.randint(min_marked, max_marked)
    return random.sample(list(range(total_rooms)), marked_count)


def _marker_plan(
    room_rows: int, room_cols: int, row_period: int, col_period: int
) -> tuple[dict[int, tuple[int, int]], dict[int, int]]:
    total_rooms = room_rows * room_cols
    marked_rooms = _choose_marked_rooms(total_rooms)
    colors = random.sample(list(MARKER_COLORS), len(marked_rooms))
    positions: dict[int, tuple[int, int]] = {}
    color_by_room: dict[int, int] = {}
    for room_id, color in zip(marked_rooms, colors):
        rr, cc = divmod(room_id, room_cols)
        rs, cs = _room_cells(rr, cc, row_period, col_period)
        r = random.randint(rs.start, rs.stop - 1)
        c = random.randint(cs.start, cs.stop - 1)
        positions[room_id] = (r, c)
        color_by_room[room_id] = color
    return positions, color_by_room


def _paint_output(
    wall_mask: np.ndarray,
    room_ids: np.ndarray,
    marker_positions: dict[int, tuple[int, int]],
    room_colors: dict[int, int],
) -> np.ndarray:
    out = np.full(room_ids.shape, BLACK, dtype=int)
    out[wall_mask] = GREY
    for room_id, color in room_colors.items():
        out[room_ids == room_id] = color
    for room_id, pos in marker_positions.items():
        out[pos] = room_colors[room_id]
    return out


def _paint_repaired_input(
    wall_mask: np.ndarray,
    room_ids: np.ndarray,
    marker_positions: dict[int, tuple[int, int]],
    room_colors: dict[int, int],
) -> np.ndarray:
    inp = np.full(room_ids.shape, BLACK, dtype=int)
    inp[wall_mask] = GREY
    for room_id, pos in marker_positions.items():
        inp[pos] = room_colors[room_id]
    return inp


def _erase_openings(
    inp: np.ndarray,
    edges: list[dict],
    marked_rooms: set[int],
) -> None:
    preferred = [
        edge
        for edge in edges
        if (edge["rooms"][0] in marked_rooms) ^ (edge["rooms"][1] in marked_rooms)
    ]
    if not preferred:
        preferred = [edge for edge in edges if set(edge["rooms"]) & marked_rooms]
    assert preferred, "need at least one marked-to-neighbor wall to damage"
    chosen_edges = [random.choice(preferred)]

    extra_pool = [edge for edge in edges if edge not in chosen_edges]
    random.shuffle(extra_pool)
    for edge in extra_pool[: random.randint(0, min(3, len(extra_pool)))]:
        chosen_edges.append(edge)

    for edge in chosen_edges:
        cells = edge["cells"]
        gap_len = 2 if len(cells) >= 3 and random.random() < 0.45 else 1
        start = random.randint(0, len(cells) - gap_len)
        for r, c in cells[start : start + gap_len]:
            inp[r, c] = BLACK

    if len(chosen_edges) == 1:
        edge = chosen_edges[0]
        room_a, room_b = edge["rooms"]
        if room_a in marked_rooms or room_b in marked_rooms:
            bridge_room = room_b if room_a in marked_rooms else room_a
            same_neighbors = [
                other
                for other in edges
                if bridge_room in other["rooms"] and other not in chosen_edges
            ]
            if same_neighbors and random.random() < 0.6:
                other = random.choice(same_neighbors)
                r, c = random.choice(other["cells"])
                inp[r, c] = BLACK


def _wall_classes_visible(inp: np.ndarray, wall_mask: np.ndarray, row_period: int, col_period: int) -> bool:
    wall_coords = np.argwhere(wall_mask)
    wall_ids = {
        int((r % row_period) * col_period + (c % col_period))
        for r, c in wall_coords
    }
    visible_ids = {
        int((r % row_period) * col_period + (c % col_period))
        for r, c in np.argwhere(inp == GREY)
    }
    return wall_ids <= visible_ids


def _damaged_input_needs_repair(inp: np.ndarray, room_ids: np.ndarray, marker_positions: dict[int, tuple[int, int]]) -> bool:
    passable = inp != GREY
    marker_set = set(marker_positions.values())
    for coords in _components(passable):
        room_set = {int(room_ids[r, c]) for r, c in coords}
        room_set.discard(-1)
        if len(room_set) < 2:
            continue
        if any((int(r), int(c)) in marker_set for r, c in coords):
            return True
    return False


def _unique_true_period(inp: np.ndarray, expected_output: np.ndarray, row_period: int, col_period: int) -> bool:
    candidates = _valid_candidates(inp)
    if not candidates:
        return False
    best_area = min(cand["area"] for cand in candidates)
    best = [cand for cand in candidates if cand["area"] == best_area]
    if len({cand["output"].tobytes() for cand in best}) != 1:
        return False
    matches = [
        (cand["row_period"], cand["col_period"])
        for cand in best
        if np.array_equal(cand["output"], expected_output)
    ]
    return matches == [(row_period, col_period)]


def _gen_attempt() -> np.ndarray | None:
    room_rows = random.randint(2, 4)
    room_cols = random.randint(2, 4)
    row_period = random.randint(3, 5)
    col_period = random.randint(3, 5)

    wall_mask, room_ids = _base_layout(room_rows, room_cols, row_period, col_period)
    marker_positions, room_colors = _marker_plan(room_rows, room_cols, row_period, col_period)
    marked_rooms = set(room_colors)
    expected_output = _paint_output(wall_mask, room_ids, marker_positions, room_colors)
    inp = _paint_repaired_input(wall_mask, room_ids, marker_positions, room_colors)

    edges = _room_adjacencies(room_rows, room_cols, row_period, col_period)
    _erase_openings(inp, edges, marked_rooms)

    if not _wall_classes_visible(inp, wall_mask, row_period, col_period):
        return None
    if not _damaged_input_needs_repair(inp, room_ids, marker_positions):
        return None
    candidates = _valid_candidates(inp)
    if not candidates:
        return None
    best_area = min(cand["area"] for cand in candidates)
    best = [cand for cand in candidates if cand["area"] == best_area]
    if len({cand["output"].tobytes() for cand in best}) != 1:
        return None
    solved, periods = _solve(inp)
    if periods != (row_period, col_period):
        return None
    if not np.array_equal(solved, expected_output):
        return None
    if not _unique_true_period(inp, expected_output, row_period, col_period):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid wallpaper-repair input")
