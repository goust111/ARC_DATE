from __future__ import annotations

import random

import numpy as np


BLACK = 0
WALL = 5
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MOVE_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

SPRITE_SHAPES = (
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (0, 3)),
    ((0, 0), (1, 0), (2, 0), (3, 0)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (1, 1), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (2, 1)),
    ((0, 0), (0, 1), (1, 0), (2, 0)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (1, 0), (1, 1), (2, 0)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 1)),
    ((0, 1), (0, 2), (1, 0), (1, 1)),
)

GEN_ROOM_ROWS = (2, 4)
GEN_ROOM_COLS = (2, 4)
GEN_ROOM_HEIGHT = (3, 6)
GEN_ROOM_WIDTH = (3, 6)
GEN_MAX_MOVES = 6
GEN_ATTEMPTS = 300


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr, nc = int(rr + dr), int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _translated_cells(
    cells: list[tuple[int, int]],
    direction: tuple[int, int],
    target_bbox: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    sr0, sc0, sr1, sc1 = _bbox(cells)
    tr0, tc0, tr1, tc1 = target_bbox
    dr, dc = direction
    if dr == -1:
        shift_r, shift_c = int(tr0 - sr0), 0
    elif dr == 1:
        shift_r, shift_c = int(tr1 - sr1), 0
    elif dc == -1:
        shift_r, shift_c = 0, int(tc0 - sc0)
    else:
        shift_r, shift_c = 0, int(tc1 - sc1)
    return [(int(r + shift_r), int(c + shift_c)) for r, c in cells]


def _extract_doors_and_sprites(
    g: np.ndarray,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    doors: list[dict[str, object]] = []
    sprites: list[dict[str, object]] = []

    for color in sorted(int(v) for v in np.unique(g)):
        if color in (BLACK, WALL):
            continue
        for comp in _components(g == color):
            if len(comp) == 1:
                doors.append({"color": int(color), "cell": comp[0]})
            else:
                assert 2 <= len(comp) <= 4, "sprites must be small 4-connected components"
                sprites.append(
                    {
                        "color": int(color),
                        "cells": comp,
                        "bbox": _bbox(comp),
                    }
                )
    return doors, sprites


def _find_rooms(
    g: np.ndarray, door_cells: set[tuple[int, int]]
) -> tuple[list[dict[str, object]], np.ndarray]:
    h, w = g.shape
    barrier = g == WALL
    for r, c in door_cells:
        barrier[int(r), int(c)] = True

    rooms: list[dict[str, object]] = []
    room_id = np.full((h, w), -1, dtype=int)
    for comp in _components(~barrier):
        r0, c0, r1, c1 = _bbox(comp)
        if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
            continue
        if len(comp) != (r1 - r0 + 1) * (c1 - c0 + 1):
            continue
        if not (
            np.all(barrier[r0 - 1, c0 : c1 + 1])
            and np.all(barrier[r1 + 1, c0 : c1 + 1])
            and np.all(barrier[r0 : r1 + 1, c0 - 1])
            and np.all(barrier[r0 : r1 + 1, c1 + 1])
        ):
            continue

        rid = len(rooms)
        for r, c in comp:
            room_id[int(r), int(c)] = int(rid)
        rooms.append({"bbox": (int(r0), int(c0), int(r1), int(c1)), "cells": comp})

    assert rooms, "expected at least one enclosed rectangular room"
    return rooms, room_id


def _door_links(
    door_cell: tuple[int, int], room_id: np.ndarray
) -> dict[int, tuple[tuple[int, int], int]]:
    r, c = door_cell
    h, w = room_id.shape
    up = int(room_id[r - 1, c]) if r > 0 else -1
    down = int(room_id[r + 1, c]) if r + 1 < h else -1
    left = int(room_id[r, c - 1]) if c > 0 else -1
    right = int(room_id[r, c + 1]) if c + 1 < w else -1

    if up >= 0 and down >= 0 and up != down and left < 0 and right < 0:
        return {
            int(up): ((1, 0), int(down)),
            int(down): ((-1, 0), int(up)),
        }
    if left >= 0 and right >= 0 and left != right and up < 0 and down < 0:
        return {
            int(left): ((0, 1), int(right)),
            int(right): ((0, -1), int(left)),
        }
    raise AssertionError("door must separate exactly two rooms across one wall")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each room's sprite to the same-colored wall door, move the sprite through
    that door's direction into the neighboring room, and slide it to the far wall.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    doors, sprites = _extract_doors_and_sprites(g)
    door_cells = {tuple(map(int, door["cell"])) for door in doors}
    rooms, room_id = _find_rooms(g, door_cells)

    for sprite in sprites:
        sprite_rooms = {int(room_id[r, c]) for r, c in sprite["cells"] if int(room_id[r, c]) >= 0}
        assert len(sprite_rooms) == 1, "every sprite must lie inside one room"
        sprite["room"] = int(next(iter(sprite_rooms)))

    for door in doors:
        door["links"] = _door_links(tuple(map(int, door["cell"])), room_id)

    moves: list[dict[str, object]] = []
    for sprite in sprites:
        room = int(sprite["room"])
        color = int(sprite["color"])
        matches = [
            door
            for door in doors
            if int(door["color"]) == color and room in door["links"]
        ]
        assert len(matches) == 1, "sprite color must identify one door on its room wall"
        door = matches[0]
        direction, target_room = door["links"][room]
        target_bbox = tuple(map(int, rooms[int(target_room)]["bbox"]))
        final_cells = _translated_cells(list(sprite["cells"]), direction, target_bbox)
        assert all(int(room_id[r, c]) == int(target_room) for r, c in final_cells)
        moves.append(
            {
                "color": int(color),
                "source_cells": list(sprite["cells"]),
                "door_cell": tuple(map(int, door["cell"])),
                "final_cells": final_cells,
            }
        )

    out = g.copy()
    for move in moves:
        for r, c in move["source_cells"]:
            out[int(r), int(c)] = BLACK
        dr, dc = move["door_cell"]
        out[int(dr), int(dc)] = BLACK
    for move in moves:
        for r, c in move["final_cells"]:
            out[int(r), int(c)] = int(move["color"])
    return out


def _build_room_grid(
    row_heights: list[int], col_widths: list[int]
) -> tuple[np.ndarray, list[int], list[int], list[dict[str, object]]]:
    row_lines = [0]
    for height in row_heights:
        row_lines.append(int(row_lines[-1] + height + 1))
    col_lines = [0]
    for width in col_widths:
        col_lines.append(int(col_lines[-1] + width + 1))

    h = int(row_lines[-1] + 1)
    w = int(col_lines[-1] + 1)
    grid = np.full((h, w), BLACK, dtype=int)
    grid[row_lines, :] = WALL
    grid[:, col_lines] = WALL

    rooms: list[dict[str, object]] = []
    ncols = len(col_widths)
    for rr, height in enumerate(row_heights):
        for cc, width in enumerate(col_widths):
            rid = rr * ncols + cc
            rooms.append(
                {
                    "id": int(rid),
                    "row": int(rr),
                    "col": int(cc),
                    "bbox": (
                        int(row_lines[rr] + 1),
                        int(col_lines[cc] + 1),
                        int(row_lines[rr + 1] - 1),
                        int(col_lines[cc + 1] - 1),
                    ),
                    "height": int(height),
                    "width": int(width),
                }
            )
    return grid, row_lines, col_lines, rooms


def _room_id(rr: int, cc: int, ncols: int) -> int:
    return int(rr * ncols + cc)


def _candidate_moves(
    row_heights: list[int], col_widths: list[int], row_lines: list[int], col_lines: list[int]
) -> list[dict[str, object]]:
    nrows = len(row_heights)
    ncols = len(col_widths)
    cands: list[dict[str, object]] = []

    for rr in range(nrows):
        for cc in range(ncols):
            src = _room_id(rr, cc, ncols)
            if rr + 1 < nrows:
                dst = _room_id(rr + 1, cc, ncols)
                wall_row = int(row_lines[rr + 1])
                cands.append(
                    {
                        "source": int(src),
                        "target": int(dst),
                        "direction": (1, 0),
                        "axis": "h",
                        "wall_key": ("h", rr, cc),
                        "fixed": int(wall_row),
                        "span": (
                            int(col_lines[cc] + 1),
                            int(col_lines[cc + 1] - 1),
                        ),
                    }
                )
                cands.append(
                    {
                        "source": int(dst),
                        "target": int(src),
                        "direction": (-1, 0),
                        "axis": "h",
                        "wall_key": ("h", rr, cc),
                        "fixed": int(wall_row),
                        "span": (
                            int(col_lines[cc] + 1),
                            int(col_lines[cc + 1] - 1),
                        ),
                    }
                )
            if cc + 1 < ncols:
                dst = _room_id(rr, cc + 1, ncols)
                wall_col = int(col_lines[cc + 1])
                cands.append(
                    {
                        "source": int(src),
                        "target": int(dst),
                        "direction": (0, 1),
                        "axis": "v",
                        "wall_key": ("v", rr, cc),
                        "fixed": int(wall_col),
                        "span": (
                            int(row_lines[rr] + 1),
                            int(row_lines[rr + 1] - 1),
                        ),
                    }
                )
                cands.append(
                    {
                        "source": int(dst),
                        "target": int(src),
                        "direction": (0, -1),
                        "axis": "v",
                        "wall_key": ("v", rr, cc),
                        "fixed": int(wall_col),
                        "span": (
                            int(row_lines[rr] + 1),
                            int(row_lines[rr + 1] - 1),
                        ),
                    }
                )
    return cands


def _select_moves(
    candidates: list[dict[str, object]], room_count: int
) -> list[dict[str, object]] | None:
    max_moves = min(GEN_MAX_MOVES, len(COLOR_POOL), room_count // 2)
    if max_moves <= 0:
        return None

    for _ in range(60):
        need = random.randint(1, max_moves)
        shuffled = candidates[:]
        random.shuffle(shuffled)
        used_rooms: set[int] = set()
        used_walls: set[tuple[object, ...]] = set()
        chosen: list[dict[str, object]] = []

        for cand in shuffled:
            src = int(cand["source"])
            dst = int(cand["target"])
            wall_key = tuple(cand["wall_key"])
            if src in used_rooms or dst in used_rooms or wall_key in used_walls:
                continue
            chosen.append(cand)
            used_rooms.add(src)
            used_rooms.add(dst)
            used_walls.add(wall_key)
            if len(chosen) == need:
                return chosen
    return None


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return max(rows) + 1, max(cols) + 1


def _pick_shape(
    direction: tuple[int, int], src_bbox: tuple[int, int, int, int], dst_bbox: tuple[int, int, int, int]
) -> tuple[tuple[int, int], ...] | None:
    sr0, sc0, sr1, sc1 = src_bbox
    tr0, tc0, tr1, tc1 = dst_bbox
    src_h, src_w = int(sr1 - sr0 + 1), int(sc1 - sc0 + 1)
    dst_h, dst_w = int(tr1 - tr0 + 1), int(tc1 - tc0 + 1)

    allowed = []
    for shape in SPRITE_SHAPES:
        sh, sw = _shape_dims(shape)
        if sh > src_h or sw > src_w:
            continue
        if direction[0] != 0 and sh > dst_h:
            continue
        if direction[1] != 0 and sw > dst_w:
            continue
        allowed.append(shape)
    return random.choice(allowed) if allowed else None


def _place_shape(
    shape: tuple[tuple[int, int], ...], room_bbox: tuple[int, int, int, int]
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = room_bbox
    sh, sw = _shape_dims(shape)
    top = random.randint(int(r0), int(r1 - sh + 1))
    left = random.randint(int(c0), int(c1 - sw + 1))
    return [(int(top + r), int(left + c)) for r, c in shape]


def _render_expected(
    inp: np.ndarray, assignments: list[dict[str, object]], rooms: list[dict[str, object]]
) -> np.ndarray:
    out = inp.copy()
    for item in assignments:
        for r, c in item["source_cells"]:
            out[int(r), int(c)] = BLACK
        dr, dc = item["door_cell"]
        out[int(dr), int(dc)] = BLACK

    for item in assignments:
        target_bbox = tuple(map(int, rooms[int(item["target"])]["bbox"]))
        final_cells = _translated_cells(list(item["source_cells"]), tuple(item["direction"]), target_bbox)
        for r, c in final_cells:
            out[int(r), int(c)] = int(item["color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    nrows = random.randint(*GEN_ROOM_ROWS)
    ncols = random.randint(*GEN_ROOM_COLS)
    row_heights = [random.randint(*GEN_ROOM_HEIGHT) for _ in range(nrows)]
    col_widths = [random.randint(*GEN_ROOM_WIDTH) for _ in range(ncols)]

    grid, row_lines, col_lines, rooms = _build_room_grid(row_heights, col_widths)
    candidates = _candidate_moves(row_heights, col_widths, row_lines, col_lines)
    moves = _select_moves(candidates, len(rooms))
    if not moves:
        return None

    colors = random.sample(COLOR_POOL, len(moves))
    assignments: list[dict[str, object]] = []
    for move, color in zip(moves, colors):
        src_bbox = tuple(map(int, rooms[int(move["source"])]["bbox"]))
        dst_bbox = tuple(map(int, rooms[int(move["target"])]["bbox"]))
        shape = _pick_shape(tuple(move["direction"]), src_bbox, dst_bbox)
        if shape is None:
            return None
        source_cells = _place_shape(shape, src_bbox)

        span_lo, span_hi = map(int, move["span"])
        if move["axis"] == "h":
            door_cell = (int(move["fixed"]), int(random.randint(span_lo, span_hi)))
        else:
            door_cell = (int(random.randint(span_lo, span_hi)), int(move["fixed"]))

        grid[door_cell] = int(color)
        for r, c in source_cells:
            grid[int(r), int(c)] = int(color)

        assignments.append(
            {
                "color": int(color),
                "source": int(move["source"]),
                "target": int(move["target"]),
                "direction": tuple(move["direction"]),
                "door_cell": door_cell,
                "source_cells": source_cells,
            }
        )

    expected = _render_expected(grid, assignments, rooms)
    try:
        actual = main(grid)
    except Exception:
        return None
    if np.array_equal(actual, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
