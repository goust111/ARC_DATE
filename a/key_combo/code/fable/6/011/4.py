from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_HELPER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_HELPER_DIR) not in sys.path:
    sys.path.append(str(_HELPER_DIR))

from arc_utils import components_mask, find_components, get_bg


GRID_MAX = 30
GEN_TRIES = 400
PLACE_TRIES = 160


CoordArray = np.ndarray


def _normalize_coords(coords: CoordArray) -> CoordArray:
    arr = np.asarray(coords, dtype=int)
    r0, c0 = arr.min(axis=0)
    out = arr.copy()
    out[:, 0] -= int(r0)
    out[:, 1] -= int(c0)
    order = np.lexsort((out[:, 1], out[:, 0]))
    return out[order]


def _shape_key(coords: CoordArray) -> tuple[tuple[int, int], ...]:
    arr = _normalize_coords(coords)
    return tuple((int(r), int(c)) for r, c in arr)


def _shape_dims(coords: CoordArray) -> tuple[int, int]:
    arr = _normalize_coords(coords)
    return int(arr[:, 0].max()) + 1, int(arr[:, 1].max()) + 1


def _flip_vertical(coords: CoordArray) -> CoordArray:
    arr = _normalize_coords(coords)
    h = int(arr[:, 0].max()) + 1
    flipped = arr.copy()
    flipped[:, 0] = h - 1 - flipped[:, 0]
    return _normalize_coords(flipped)


def _place_shape(anchor: tuple[int, int], coords: CoordArray) -> CoordArray:
    r0, c0 = map(int, anchor)
    return np.asarray(coords, dtype=int) + np.array([r0, c0], dtype=int)


def _paint(grid: np.ndarray, coords: CoordArray, color: int) -> None:
    grid[coords[:, 0], coords[:, 1]] = int(color)


def _board_mask_from_component(board: dict) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = map(int, board["bbox"])
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    coords = np.asarray(board["coords"], dtype=int)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (r0, c0, r1, c1)


def _extract_notches(board: dict) -> tuple[str, list[dict]]:
    board_mask, (r0, c0, _r1, _c1) = _board_mask_from_component(board)
    empty = ~board_mask
    h, _w = empty.shape

    top_notches: list[dict] = []
    bottom_notches: list[dict] = []

    for comp in components_mask(empty, connectivity=4):
        touches_top = bool(np.any(comp[:, 0] == 0))
        touches_bottom = bool(np.any(comp[:, 0] == h - 1))
        if touches_top == touches_bottom:
            continue
        comp_global = comp + np.array([r0, c0], dtype=int)
        nr0, nc0 = comp_global.min(axis=0)
        nr1, nc1 = comp_global.max(axis=0)
        entry = {
            "coords": comp_global,
            "bbox": (int(nr0), int(nc0), int(nr1), int(nc1)),
            "shape": _normalize_coords(comp),
            "key": _shape_key(comp),
        }
        if touches_top:
            top_notches.append(entry)
        else:
            bottom_notches.append(entry)

    if top_notches and not bottom_notches:
        return "top", sorted(top_notches, key=lambda x: (x["bbox"][1], x["bbox"][0]))
    if bottom_notches and not top_notches:
        return "bottom", sorted(bottom_notches, key=lambda x: (x["bbox"][1], x["bbox"][0]))
    raise ValueError("failed to recover a single-sided notch family")


def _solve_scene(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    if not comps:
        raise ValueError("empty scene")
    comps.sort(key=lambda comp: -int(comp["area"]))

    board = comps[0]
    side, notches = _extract_notches(board)
    pieces = comps[1:]

    piece_by_key: dict[tuple[tuple[int, int], ...], list[dict]] = {}
    for piece in pieces:
        key = _shape_key(np.asarray(piece["coords"], dtype=int))
        piece_by_key.setdefault(key, []).append(piece)

    out = g.copy()
    board_color = int(board["color"])
    board_r0, _board_c0, board_r1, _board_c1 = map(int, board["bbox"])

    for notch in notches:
        matches = piece_by_key.get(notch["key"], [])
        if len(matches) != 1:
            raise ValueError("notch-to-sprite matching is ambiguous")
        piece = matches[0]
        piece_coords = np.asarray(piece["coords"], dtype=int)
        out[piece_coords[:, 0], piece_coords[:, 1]] = int(bg)

        notch_coords = np.asarray(notch["coords"], dtype=int)
        out[notch_coords[:, 0], notch_coords[:, 1]] = int(board_color)

        flipped = _flip_vertical(np.asarray(notch["shape"], dtype=int))
        fh, _fw = _shape_dims(flipped)
        _notch_r0, notch_c0, _notch_r1, _notch_c1 = map(int, notch["bbox"])

        if side == "top":
            anchor = (board_r1 + 1, notch_c0)
        else:
            anchor = (board_r0 - fh, notch_c0)

        placed = _place_shape(anchor, flipped)
        if (
            placed[:, 0].min() < 0
            or placed[:, 1].min() < 0
            or placed[:, 0].max() >= g.shape[0]
            or placed[:, 1].max() >= g.shape[1]
        ):
            raise ValueError("extruded sprite would leave the grid")
        _paint(out, placed, int(piece["color"]))

    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Fill every board notch with the board color, then press the matching sprite through
    the board so that it reappears on the opposite side as a vertically mirrored extrusion.
    """
    return _solve_scene(np.asarray(input_grid, dtype=int))


def _make_profile_shape(width: int, max_height: int) -> CoordArray:
    for _ in range(PLACE_TRIES):
        depths = [random.randint(1, max_height) for _ in range(width)]
        if max(depths) < 2:
            continue
        if len(set(depths)) == 1:
            continue
        cells = [(r, c) for c, depth in enumerate(depths) for r in range(depth)]
        shape = _normalize_coords(np.array(cells, dtype=int))
        if _shape_key(shape) != _shape_key(_flip_vertical(shape)):
            return shape
    raise ValueError("failed to build an asymmetric notch profile")


def _reserve_clear(occupied: np.ndarray, coords: CoordArray, *, padding: int = 1) -> bool:
    h, w = occupied.shape
    r0 = max(0, int(coords[:, 0].min()) - padding)
    c0 = max(0, int(coords[:, 1].min()) - padding)
    r1 = min(h - 1, int(coords[:, 0].max()) + padding)
    c1 = min(w - 1, int(coords[:, 1].max()) + padding)
    return not bool(occupied[r0 : r1 + 1, c0 : c1 + 1].any())


def _stamp_occupied(occupied: np.ndarray, coords: CoordArray, *, padding: int = 1) -> None:
    h, w = occupied.shape
    r0 = max(0, int(coords[:, 0].min()) - padding)
    c0 = max(0, int(coords[:, 1].min()) - padding)
    r1 = min(h - 1, int(coords[:, 0].max()) + padding)
    c1 = min(w - 1, int(coords[:, 1].max()) + padding)
    occupied[r0 : r1 + 1, c0 : c1 + 1] = True


def _build_case() -> tuple[np.ndarray, np.ndarray]:
    board_h = random.randint(7, 11)
    board_w = random.randint(11, 17)
    notch_count = random.randint(2, min(4, max(2, board_w // 4)))
    side = random.choice(["top", "bottom"])

    notch_specs: list[dict] = []
    shape_keys: set[tuple[tuple[int, int], ...]] = set()

    for _ in range(PLACE_TRIES):
        notch_specs.clear()
        shape_keys.clear()
        cursor = 1
        ok = True
        for idx in range(notch_count):
            remaining = notch_count - idx - 1
            width = random.randint(2, 4)
            if cursor + width + remaining * 3 > board_w - 1:
                ok = False
                break
            max_left = board_w - 1 - width - remaining * 3
            left = random.randint(cursor, max_left)
            shape = _make_profile_shape(width, max_height=min(4, board_h - 2))
            actual_shape = shape if side == "top" else _flip_vertical(shape)
            key = _shape_key(actual_shape)
            if key in shape_keys:
                ok = False
                break
            shape_keys.add(key)
            notch_specs.append({"left": int(left), "shape": actual_shape})
            cursor = left + width + 1
        if ok and len(notch_specs) == notch_count:
            break
    else:
        raise ValueError("failed to lay out distinct notches")

    max_notch_h = max(_shape_dims(spec["shape"])[0] for spec in notch_specs)
    grid_h = board_h + random.randint(max_notch_h + 5, max_notch_h + 11)
    grid_w = board_w + random.randint(6, 12)
    if grid_h > GRID_MAX or grid_w > GRID_MAX:
        raise ValueError("generated grid exceeds ARC limits")

    board_r0 = random.randint(3, grid_h - board_h - 4)
    board_c0 = random.randint(2, grid_w - board_w - 3)
    board_r1 = board_r0 + board_h - 1

    bg = random.randint(0, 9)
    palette = [color for color in range(10) if color != bg]
    random.shuffle(palette)
    board_color = palette.pop()
    if len(palette) < notch_count:
        raise ValueError("not enough sprite colors")

    grid = np.full((grid_h, grid_w), int(bg), dtype=int)
    board_mask = np.ones((board_h, board_w), dtype=bool)

    realized_notches: list[dict] = []
    for spec in notch_specs:
        shape = np.asarray(spec["shape"], dtype=int)
        sh, _sw = _shape_dims(shape)
        notch_top = 0 if side == "top" else board_h - sh
        local_coords = _place_shape((notch_top, int(spec["left"])), shape)
        board_mask[local_coords[:, 0], local_coords[:, 1]] = False
        realized_notches.append(
            {
                "shape": shape,
                "coords": _place_shape((board_r0, board_c0), local_coords),
                "left": int(spec["left"]),
            }
        )

    board_local = np.argwhere(board_mask)
    board_global = _place_shape((board_r0, board_c0), board_local)
    _paint(grid, board_global, int(board_color))

    occupied = np.zeros_like(grid, dtype=bool)
    _stamp_occupied(occupied, board_global, padding=1)

    expected = grid.copy()

    for notch in realized_notches:
        color = palette.pop()
        shape = np.asarray(notch["shape"], dtype=int)
        sh, sw = _shape_dims(shape)
        placed_piece = None
        for _ in range(PLACE_TRIES):
            r0 = random.randint(1, grid_h - sh - 2)
            c0 = random.randint(1, grid_w - sw - 2)
            coords = _place_shape((r0, c0), shape)
            if not _reserve_clear(occupied, coords, padding=1):
                continue
            placed_piece = coords
            break
        if placed_piece is None:
            raise ValueError("failed to place a sprite")

        _paint(grid, placed_piece, int(color))
        _stamp_occupied(occupied, placed_piece, padding=1)

        notch_coords = np.asarray(notch["coords"], dtype=int)
        expected[notch_coords[:, 0], notch_coords[:, 1]] = int(board_color)
        expected[placed_piece[:, 0], placed_piece[:, 1]] = int(bg)

        flipped = _flip_vertical(shape)
        fh, _fw = _shape_dims(flipped)
        anchor = (board_r1 + 1, board_c0 + int(notch["left"])) if side == "top" else (board_r0 - fh, board_c0 + int(notch["left"]))
        extruded = _place_shape(anchor, flipped)
        _paint(expected, extruded, int(color))

    produced = main(grid)
    if not np.array_equal(produced, expected):
        raise ValueError("generated case does not round-trip through solver")
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_TRIES):
        try:
            grid, _expected = _build_case()
            return grid
        except (AssertionError, ValueError):
            continue
    raise RuntimeError("failed to generate a valid through-assembly instance")
