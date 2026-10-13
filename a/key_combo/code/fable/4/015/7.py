from __future__ import annotations

from collections import Counter
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def get_bg(grid: np.ndarray) -> int:
    """Return the most common color in the grid."""
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> None:
    """Paste a patch into the destination grid with optional transparency."""
    r0 = int(r0)
    c0 = int(c0)
    h, w = map(int, patch.shape)
    if r0 < 0 or c0 < 0 or r0 + h > dst.shape[0] or c0 + w > dst.shape[1]:
        raise ValueError("paste out of bounds")
    if transparent is None:
        dst[r0 : r0 + h, c0 : c0 + w] = patch
        return
    mask = patch != int(transparent)
    dst_view = dst[r0 : r0 + h, c0 : c0 + w]
    dst_view[mask] = patch[mask]


def components_from_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    """Return connected components from a boolean mask as coordinate arrays."""
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)] if int(connectivity) == 4 else [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),            (0, 1),
        (1, -1),  (1, 0),   (1, 1),
    ]
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in deltas:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _is_isolated_point_lattice(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    for r, c in coords:
        r0 = max(0, int(r) - 1)
        r1 = min(h, int(r) + 2)
        c0 = max(0, int(c) - 1)
        c1 = min(w, int(c) + 2)
        if int(np.count_nonzero(mask[r0:r1, c0:c1])) != 1:
            return False
    return True


def _lattice_signature(mask: np.ndarray) -> tuple[list[int], list[int], int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0 or not _is_isolated_point_lattice(mask):
        return None

    rows = sorted(map(int, np.unique(coords[:, 0])))
    cols = sorted(map(int, np.unique(coords[:, 1])))
    if len(rows) < 2 or len(cols) < 2:
        return None
    if len(coords) != len(rows) * len(cols):
        return None

    coord_set = {tuple(map(int, cell)) for cell in coords}
    if any((r, c) not in coord_set for r in rows for c in cols):
        return None

    row_steps = {rows[i + 1] - rows[i] for i in range(len(rows) - 1)}
    col_steps = {cols[i + 1] - cols[i] for i in range(len(cols) - 1)}
    if len(row_steps) != 1 or len(col_steps) != 1:
        return None

    pitch_r = int(next(iter(row_steps)))
    pitch_c = int(next(iter(col_steps)))
    if pitch_r != pitch_c or pitch_r < 4 or pitch_r % 2 != 0:
        return None
    return rows, cols, pitch_r


def _find_marker_lattice(grid: np.ndarray, bg: int) -> tuple[int, list[int], list[int], int]:
    candidates = []
    for color in sorted(set(map(int, np.unique(grid))) - {int(bg)}):
        signature = _lattice_signature(grid == color)
        if signature is not None:
            rows, cols, pitch = signature
            candidates.append((int(color), rows, cols, pitch))
    assert len(candidates) == 1, "expected exactly one complete marker lattice"
    return candidates[0]


def _cell_center(marker_rows: list[int], marker_cols: list[int], cell: Coord) -> Coord:
    r_idx, c_idx = map(int, cell)
    return (
        (int(marker_rows[r_idx]) + int(marker_rows[r_idx + 1])) // 2,
        (int(marker_cols[c_idx]) + int(marker_cols[c_idx + 1])) // 2,
    )


def _locate_cell(marker_rows: list[int], marker_cols: list[int], bbox: BBox) -> Coord:
    r0, c0, r1, c1 = map(int, bbox)
    row_hits = [i for i in range(len(marker_rows) - 1) if marker_rows[i] < r0 and r1 < marker_rows[i + 1]]
    col_hits = [j for j in range(len(marker_cols) - 1) if marker_cols[j] < c0 and c1 < marker_cols[j + 1]]
    assert len(row_hits) == 1 and len(col_hits) == 1, "sprite must stay inside one lattice cell"
    return int(row_hits[0]), int(col_hits[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each sprite by decoding its in-cell offset as a same-sized displacement across lattice cells.

    Input structure:
    - The background is the dominant color.
    - One foreground color forms a complete Cartesian lattice of isolated marker pixels that partitions the canvas into equal square cells.
    - Some lattice cells contain one sprite whose odd-sized bounding-box center is either centered in the cell or shifted by 1 or 2 cells along exactly one axis.

    Rule:
    - Keep the marker lattice fixed and clear every sprite from its original position.
    - A centered sprite stays in its cell; a shifted sprite moves in the same direction by as many lattice cells as the magnitude of that shift.
    - Repaint each sprite unchanged, but centered inside its destination cell.

    Key insight:
    - The sprite's small offset inside a cell is a scaled-down copy of its large movement across the lattice.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = get_bg(g)
    marker_color, marker_rows, marker_cols, _pitch = _find_marker_lattice(g, bg)

    sprite_mask = (g != bg) & (g != marker_color)
    sprites = components_from_mask(sprite_mask, connectivity=4)
    assert sprites, "expected at least one sprite"

    sprite_infos = []
    source_cells: set[Coord] = set()
    dest_cells: set[Coord] = set()
    cell_row_count = len(marker_rows) - 1
    cell_col_count = len(marker_cols) - 1

    for coords in sprites:
        bbox = bbox_coords(coords)
        r0, c0, r1, c1 = bbox
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h % 2 == 1 and w % 2 == 1, "sprite bbox must have odd height and width"

        source_cell = _locate_cell(marker_rows, marker_cols, bbox)
        assert source_cell not in source_cells, "expected at most one sprite per source cell"
        source_cells.add(source_cell)

        sprite_center = ((r0 + r1) // 2, (c0 + c1) // 2)
        cell_center = _cell_center(marker_rows, marker_cols, source_cell)
        dr = int(sprite_center[0] - cell_center[0])
        dc = int(sprite_center[1] - cell_center[1])

        assert (dr == 0 or dc == 0) and min(abs(dr), abs(dc)) == 0, "offset must be axis-aligned"
        assert max(abs(dr), abs(dc)) <= 2, "offset magnitude must be 0, 1, or 2"

        dest_cell = (int(source_cell[0] + dr), int(source_cell[1] + dc))
        assert 0 <= dest_cell[0] < cell_row_count and 0 <= dest_cell[1] < cell_col_count, "destination cell out of bounds"
        assert dest_cell not in dest_cells, "multiple sprites cannot land in the same destination cell"
        dest_cells.add(dest_cell)

        sprite_infos.append(
            {
                "source_cell": source_cell,
                "dest_cell": dest_cell,
                "offset": (dr, dc),
                "bbox": bbox,
                "patch": g[r0 : r1 + 1, c0 : c1 + 1].copy(),
            }
        )

    for info in sprite_infos:
        if info["offset"] != (0, 0):
            assert info["dest_cell"] not in source_cells, "moving sprites must land in previously empty cells"

    out = g.copy()
    out[sprite_mask] = bg

    for info in sprite_infos:
        patch = info["patch"]
        h, w = patch.shape
        target_center_r, target_center_c = _cell_center(marker_rows, marker_cols, info["dest_cell"])
        paste_r = target_center_r - h // 2
        paste_c = target_center_c - w // 2
        mask = patch != bg
        target = out[paste_r : paste_r + h, paste_c : paste_c + w]
        assert target.shape == patch.shape, "destination paste must stay in bounds"
        assert np.all(target[mask] == bg), "destination cell must be empty"
        paste(out, patch, paste_r, paste_c, transparent=bg)

    return out


# --------------- generator ---------------

BG_COLOR = 0
PITCH = 8
CELL_LAYOUTS = ((3, 3), (3, 3), (3, 2), (2, 3))
TOP_LEFT_MAX = 2
BOTTOM_PAD_MAX = 2
GEN_ATTEMPTS = 300
COLOR_POOL = tuple(range(1, 10))
MOVE_VECTORS = ((-2, 0), (-1, 0), (1, 0), (2, 0), (0, -2), (0, -1), (0, 1), (0, 2))
BASE_MASKS = (
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [1, 1, 1], [0, 0, 1]], dtype=bool),
)


def _move_orientation(move) -> str:
    return "v" if int(move[4]) != 0 else "h"


def _move_magnitude(move) -> int:
    return abs(int(move[4])) + abs(int(move[5]))


def _all_cells(cell_rows: int, cell_cols: int) -> list[Coord]:
    return [(r, c) for r in range(cell_rows) for c in range(cell_cols)]


def _candidate_moves(cell_rows: int, cell_cols: int):
    moves = []
    for sr, sc in _all_cells(cell_rows, cell_cols):
        for dr, dc in MOVE_VECTORS:
            tr = sr + dr
            tc = sc + dc
            if 0 <= tr < cell_rows and 0 <= tc < cell_cols:
                moves.append((sr, sc, tr, tc, dr, dc))
    return moves


def _compatible_move(move, used_sources: set[Coord], used_dests: set[Coord]) -> bool:
    source = (int(move[0]), int(move[1]))
    dest = (int(move[2]), int(move[3]))
    return source not in used_sources and dest not in used_dests and source not in used_dests and dest not in used_sources


def _choose_layout(cell_rows: int, cell_cols: int):
    cells = _all_cells(cell_rows, cell_cols)
    moves = _candidate_moves(cell_rows, cell_cols)
    mag2_moves = [move for move in moves if _move_magnitude(move) == 2]
    all_orientations = {_move_orientation(move) for move in moves}
    move_max = min(3, (len(cells) - 1) // 2)
    move_min = 2 if move_max >= 2 else 1

    for strict in (True, False):
        for _ in range(120):
            move_count = random.randint(move_min, move_max)
            selected = []
            used_sources: set[Coord] = set()
            used_dests: set[Coord] = set()

            if strict and mag2_moves:
                first = random.choice(mag2_moves)
                selected.append(first)
                used_sources.add((int(first[0]), int(first[1])))
                used_dests.add((int(first[2]), int(first[3])))

            if strict and move_count >= 2 and len(all_orientations) == 2:
                missing = {"h", "v"} - {_move_orientation(move) for move in selected}
                if missing:
                    options = [move for move in moves if _move_orientation(move) in missing and _compatible_move(move, used_sources, used_dests)]
                    if not options:
                        continue
                    move = random.choice(options)
                    selected.append(move)
                    used_sources.add((int(move[0]), int(move[1])))
                    used_dests.add((int(move[2]), int(move[3])))

            pool = moves[:]
            random.shuffle(pool)
            for move in pool:
                if len(selected) >= move_count:
                    break
                if _compatible_move(move, used_sources, used_dests):
                    selected.append(move)
                    used_sources.add((int(move[0]), int(move[1])))
                    used_dests.add((int(move[2]), int(move[3])))

            if len(selected) != move_count:
                continue
            if strict and mag2_moves and all(_move_magnitude(move) != 2 for move in selected):
                continue
            if strict and move_count >= 2 and len(all_orientations) == 2 and len({_move_orientation(move) for move in selected}) < 2:
                continue

            remaining = [cell for cell in cells if cell not in used_sources and cell not in used_dests]
            if not remaining:
                continue
            centered_count = random.randint(1, min(3, len(remaining)))
            centered_cells = random.sample(remaining, centered_count)
            return selected, centered_cells

    return None


def _sample_patch(bg: int, sprite_colors: list[int]) -> np.ndarray:
    mask = random.choice(BASE_MASKS).copy()
    mask = np.rot90(mask, random.randrange(4))
    if random.random() < 0.5:
        mask = np.fliplr(mask)

    patch = np.full(mask.shape, bg, dtype=int)
    if random.random() < 0.35 and len(sprite_colors) >= 2:
        colors = random.sample(sprite_colors, 2)
        coords = list(map(tuple, np.argwhere(mask)))
        random.shuffle(coords)
        split = random.randint(1, len(coords) - 1)
        for r, c in coords[:split]:
            patch[r, c] = colors[0]
        for r, c in coords[split:]:
            patch[r, c] = colors[1]
    else:
        patch[mask] = random.choice(sprite_colors)
    return patch


def _paint_markers(grid: np.ndarray, marker_rows: list[int], marker_cols: list[int], marker_color: int) -> None:
    for r in marker_rows:
        for c in marker_cols:
            grid[int(r), int(c)] = int(marker_color)


def _gen_attempt() -> np.ndarray | None:
    cell_rows, cell_cols = random.choice(CELL_LAYOUTS)
    layout = _choose_layout(cell_rows, cell_cols)
    if layout is None:
        return None
    moving, centered = layout

    first_r = random.randint(0, TOP_LEFT_MAX)
    first_c = random.randint(0, TOP_LEFT_MAX)
    marker_rows = [first_r + PITCH * i for i in range(cell_rows + 1)]
    marker_cols = [first_c + PITCH * j for j in range(cell_cols + 1)]
    height = marker_rows[-1] + 1 + random.randint(0, BOTTOM_PAD_MAX)
    width = marker_cols[-1] + 1 + random.randint(0, BOTTOM_PAD_MAX)
    if height > 30 or width > 30:
        return None

    marker_color = random.choice(COLOR_POOL)
    sprite_colors = [color for color in COLOR_POOL if color != marker_color]
    inp = np.full((height, width), BG_COLOR, dtype=int)
    out = np.full((height, width), BG_COLOR, dtype=int)
    _paint_markers(inp, marker_rows, marker_cols, marker_color)
    _paint_markers(out, marker_rows, marker_cols, marker_color)

    records = []
    for move in moving:
        records.append(
            {
                "source_cell": (int(move[0]), int(move[1])),
                "dest_cell": (int(move[2]), int(move[3])),
                "offset": (int(move[4]), int(move[5])),
            }
        )
    for cell in centered:
        records.append({"source_cell": cell, "dest_cell": cell, "offset": (0, 0)})
    random.shuffle(records)

    for record in records:
        patch = _sample_patch(BG_COLOR, sprite_colors)
        h, w = patch.shape
        in_center_r, in_center_c = _cell_center(marker_rows, marker_cols, record["source_cell"])
        in_center_r += int(record["offset"][0])
        in_center_c += int(record["offset"][1])
        paste(inp, patch, in_center_r - h // 2, in_center_c - w // 2, transparent=BG_COLOR)

        out_center_r, out_center_c = _cell_center(marker_rows, marker_cols, record["dest_cell"])
        paste(out, patch, out_center_r - h // 2, out_center_c - w // 2, transparent=BG_COLOR)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
