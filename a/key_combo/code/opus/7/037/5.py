from __future__ import annotations

from collections import deque
import random

import numpy as np


DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GEN_RETRIES = 500
GEN_MAX_SIDE = 18
GEN_MIN_SIDE = 14


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(cells, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _coords_to_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (r0, c0, r1, c1)


def _triangle_mask(level: int, direction: str) -> np.ndarray:
    span = int(level)
    h = span + 1
    w = 2 * span + 1
    up = np.zeros((h, w), dtype=bool)
    for r in range(h):
        c0 = span - r
        c1 = span + r
        up[r, c0 : c1 + 1] = True
    if direction == "up":
        return up
    if direction == "down":
        return np.flipud(up)
    if direction == "left":
        return np.rot90(up, 1)
    if direction == "right":
        return np.rot90(up, -1)
    raise ValueError(direction)


def _triangle_info(coords: np.ndarray) -> dict[str, object] | None:
    mask, box = _coords_to_mask(coords)
    for level in range(1, 6):
        for direction in DIRS:
            templ = _triangle_mask(level, direction)
            if mask.shape != templ.shape or not np.array_equal(mask, templ):
                continue
            r0, c0, _r1, _c1 = box
            if direction == "up":
                apex = (r0, c0 + templ.shape[1] // 2)
            elif direction == "down":
                apex = (r0 + templ.shape[0] - 1, c0 + templ.shape[1] // 2)
            elif direction == "left":
                apex = (r0 + templ.shape[0] // 2, c0)
            else:
                apex = (r0 + templ.shape[0] // 2, c0 + templ.shape[1] - 1)
            return {"direction": direction, "apex": apex}
    return None


def _cells_from_mask(mask: np.ndarray, r0: int, c0: int) -> set[tuple[int, int]]:
    pts = set()
    for rr, cc in np.argwhere(mask):
        pts.add((int(r0 + rr), int(c0 + cc)))
    return pts


def _translate(cells: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in cells}


def _in_bounds(shape: tuple[int, int], cells: set[tuple[int, int]]) -> bool:
    h, w = shape
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _paint(grid: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _expanded(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    out = set()
    for r, c in cells:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _bbox_of_cells(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _swept_cells(cells: set[tuple[int, int]], dr: int, dc: int, steps: int) -> set[tuple[int, int]]:
    swept = set()
    for k in range(steps + 1):
        swept |= {(r + dr * k, c + dc * k) for r, c in cells}
    return swept


def _all_nonzero_components(grid: np.ndarray) -> list[dict[str, object]]:
    comps = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != 0):
        for coords in _components(grid == color):
            comps.append({"color": color, "coords": coords})
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each colored triangle as a direction marker for the other same-colored object, then erase the triangles.

    Input structure:
    - The background is black.
    - Each foreground color appears in exactly two 4-connected monochrome objects.
    - For each color, one object is a filled triangle pointing up, down, left, or right, and the other is a compact non-triangular movable shape.

    Rule:
    - Match each triangle with the other object of the same color.
    - Sort the pairs by the triangle apex row, breaking ties by apex column.
    - In that order, slide the movable object one cell at a time in the triangle's pointing direction until the next step would leave the grid or hit a non-black cell.
    - After the move, erase that triangle to black; already moved objects remain as obstacles for later moves.

    Key insight:
    - The triangles are only directional controllers: same color selects the moving object, apex order selects the processing order, and the moved objects create the later collisions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    pairs = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != 0):
        parts = _components(g == color)
        assert len(parts) == 2, "each color must form exactly two objects"
        tagged = []
        for coords in parts:
            info = _triangle_info(coords)
            tagged.append((coords, info))
        triangles = [(coords, info) for coords, info in tagged if info is not None]
        movers = [coords for coords, info in tagged if info is None]
        assert len(triangles) == 1 and len(movers) == 1, "each color needs one triangle and one non-triangle"
        tri_coords, tri_info = triangles[0]
        pairs.append(
            {
                "color": color,
                "triangle": {(int(r), int(c)) for r, c in tri_coords},
                "mover": {(int(r), int(c)) for r, c in movers[0]},
                "direction": str(tri_info["direction"]),
                "apex": tuple(int(v) for v in tri_info["apex"]),
            }
        )

    pairs.sort(key=lambda item: item["apex"])
    board = g.copy()
    h, w = board.shape

    for pair in pairs:
        color = int(pair["color"])
        mover = set(pair["mover"])
        triangle = set(pair["triangle"])
        dr, dc = DIRS[str(pair["direction"])]

        occupied = {(int(r), int(c)) for r, c in np.argwhere(board != 0)}
        occupied -= mover
        current = mover
        while True:
            nxt = _translate(current, dr, dc)
            if not _in_bounds((h, w), nxt) or any(cell in occupied for cell in nxt):
                break
            current = nxt

        _paint(board, mover, 0)
        _paint(board, current, color)
        _paint(board, triangle, 0)

    return board


MOVER_TEMPLATES = [
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1], [1, 0]], dtype=bool),
]


def _unique_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants = []
    seen = set()
    for k in range(4):
        rot = np.rot90(mask, k)
        for cand in (rot, np.fliplr(rot)):
            key = cand.tobytes() + str(cand.shape).encode()
            if key not in seen:
                seen.add(key)
                variants.append(cand.copy())
    return variants


MOVER_VARIANTS = [variant for mask in MOVER_TEMPLATES for variant in _unique_variants(mask)]


def _sample_final_anchor(
    shape: tuple[int, int],
    mask: np.ndarray,
    direction: str,
    occupied: set[tuple[int, int]],
    stop_type: str,
) -> tuple[int, int] | None:
    h, w = shape
    mh, mw = mask.shape
    dr, dc = DIRS[direction]
    for _ in range(300):
        r0 = random.randint(0, h - mh)
        c0 = random.randint(0, w - mw)
        cells = _cells_from_mask(mask, r0, c0)
        if cells & occupied:
            continue
        nxt = _translate(cells, dr, dc)
        edge_block = not _in_bounds(shape, nxt)
        occ_block = any(cell in occupied for cell in nxt if 0 <= cell[0] < h and 0 <= cell[1] < w)
        if stop_type == "boundary" and edge_block:
            return r0, c0
        if stop_type == "obstacle" and (not edge_block) and occ_block:
            return r0, c0
    return None


def _valid_pair_structure(grid: np.ndarray) -> bool:
    colors = [int(v) for v in np.unique(grid) if int(v) != 0]
    if len(colors) < 3 or len(colors) > 4:
        return False
    for color in colors:
        parts = _components(grid == color)
        if len(parts) != 2:
            return False
        infos = [_triangle_info(coords) for coords in parts]
        if sum(info is not None for info in infos) != 1:
            return False
        tri = parts[0] if infos[0] is not None else parts[1]
        mov = parts[1] if infos[0] is not None else parts[0]
        tri_rows = {int(r) for r, _c in tri}
        tri_cols = {int(c) for _r, c in tri}
        mov_rows = {int(r) for r, _c in mov}
        mov_cols = {int(c) for _r, c in mov}
        if not tri_rows.isdisjoint(mov_rows) or not tri_cols.isdisjoint(mov_cols):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GEN_MIN_SIDE, GEN_MAX_SIDE)
    w = random.randint(GEN_MIN_SIDE, GEN_MAX_SIDE)
    pair_count = random.randint(3, 4)
    colors = random.sample(COLOR_POOL, pair_count)

    pairs: list[dict[str, object]] = []
    occupied_final: set[tuple[int, int]] = set()
    stop_types = ["boundary"]
    if pair_count >= 2:
        stop_types.extend(random.choices(["boundary", "obstacle"], k=pair_count - 1))
        if pair_count >= 3 and "obstacle" not in stop_types:
            stop_types[-1] = "obstacle"

    for i in range(pair_count):
        color = colors[i]
        direction = random.choice(list(DIRS))
        mover_mask = random.choice(MOVER_VARIANTS)
        stop_type = stop_types[i] if occupied_final else "boundary"
        anchor = _sample_final_anchor((h, w), mover_mask, direction, occupied_final, stop_type)
        if anchor is None and stop_type == "obstacle":
            anchor = _sample_final_anchor((h, w), mover_mask, direction, occupied_final, "boundary")
            stop_type = "boundary"
        if anchor is None:
            return None
        final_cells = _cells_from_mask(mover_mask, anchor[0], anchor[1])
        pairs.append(
            {
                "color": color,
                "direction": direction,
                "mover_mask": mover_mask,
                "final_anchor": anchor,
                "final_cells": final_cells,
                "stop_type": stop_type,
            }
        )
        occupied_final |= final_cells

    out = np.zeros((h, w), dtype=int)
    for pair in pairs:
        _paint(out, set(pair["final_cells"]), int(pair["color"]))

    input_occupied: set[tuple[int, int]] = set()
    input_expanded: set[tuple[int, int]] = set()

    for i in range(pair_count - 1, -1, -1):
        pair = pairs[i]
        mover_mask = pair["mover_mask"]
        direction = str(pair["direction"])
        dr, dc = DIRS[direction]
        fr0, fc0 = pair["final_anchor"]
        mh, mw = mover_mask.shape

        earlier_final = set()
        for j in range(i):
            earlier_final |= set(pairs[j]["final_cells"])

        max_gap = 0
        while True:
            next_anchor = (fr0 - dr * (max_gap + 1), fc0 - dc * (max_gap + 1))
            next_cells = _cells_from_mask(mover_mask, next_anchor[0], next_anchor[1])
            if not _in_bounds((h, w), next_cells):
                break
            max_gap += 1
        if max_gap < 1:
            return None

        gaps = list(range(1, min(max_gap, 7) + 1))
        random.shuffle(gaps)
        placed = False
        for gap in gaps:
            sr0 = fr0 - dr * gap
            sc0 = fc0 - dc * gap
            start_cells = _cells_from_mask(mover_mask, sr0, sc0)
            if start_cells & input_expanded:
                continue
            swept = _swept_cells(start_cells, dr, dc, gap)
            if swept & input_occupied:
                continue
            if swept & earlier_final:
                continue
            pair["start_anchor"] = (sr0, sc0)
            pair["start_cells"] = start_cells
            pair["swept"] = swept
            input_occupied |= start_cells
            input_expanded |= _expanded(start_cells, (h, w))
            placed = True
            break
        if not placed:
            return None

    all_swept = [set(pair["swept"]) for pair in pairs]
    prev_apex_row = -1
    for i, pair in enumerate(pairs):
        direction = str(pair["direction"])
        mover_cells = set(pair["start_cells"])
        mr0, mc0, mr1, mc1 = _bbox_of_cells(mover_cells)
        level_choices = [1, 2]
        random.shuffle(level_choices)
        placed = False

        other_swept = set()
        for j, swept in enumerate(all_swept):
            if j != i:
                other_swept |= swept

        remaining = pair_count - i - 1
        for level in level_choices:
            tri_mask = _triangle_mask(level, direction)
            th, tw = tri_mask.shape
            if direction == "up":
                apex_r_off, apex_c_off = 0, tw // 2
            elif direction == "down":
                apex_r_off, apex_c_off = th - 1, tw // 2
            elif direction == "left":
                apex_r_off, apex_c_off = th // 2, 0
            else:
                apex_r_off, apex_c_off = th // 2, tw - 1

            max_apex_row = h - 1 - remaining
            apex_rows = list(range(prev_apex_row + 1, max_apex_row + 1))
            random.shuffle(apex_rows)
            for apex_row in apex_rows:
                r0 = apex_row - apex_r_off
                if r0 < 0 or r0 + th > h:
                    continue
                c_options = list(range(0, w - tw + 1))
                random.shuffle(c_options)
                for c0 in c_options:
                    tri_cells = _cells_from_mask(tri_mask, r0, c0)
                    if tri_cells & input_expanded:
                        continue
                    if tri_cells & other_swept:
                        continue
                    tr0, tc0, tr1, tc1 = _bbox_of_cells(tri_cells)
                    rows_ok = tr1 < mr0 or mr1 < tr0
                    cols_ok = tc1 < mc0 or mc1 < tc0
                    if not (rows_ok and cols_ok):
                        continue
                    pair["triangle_cells"] = tri_cells
                    pair["triangle_mask"] = tri_mask
                    pair["apex"] = (apex_row, c0 + apex_c_off)
                    input_occupied |= tri_cells
                    input_expanded |= _expanded(tri_cells, (h, w))
                    prev_apex_row = apex_row
                    placed = True
                    break
                if placed:
                    break
            if placed:
                break
        if not placed:
            return None

    inp = np.zeros((h, w), dtype=int)
    for pair in pairs:
        _paint(inp, set(pair["start_cells"]), int(pair["color"]))
        _paint(inp, set(pair["triangle_cells"]), int(pair["color"]))

    if not _valid_pair_structure(inp):
        return None

    comps = _all_nonzero_components(inp)
    if not comps or any(_expanded({(int(r), int(c)) for r, c in comp["coords"]}, (h, w)) & set().union(
        *[
            {(int(r), int(c)) for r, c in other["coords"]}
            for other in comps
            if other is not comp
        ]
    ) for comp in comps):
        return None

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
