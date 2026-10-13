from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

import numpy as np

BG = 0
GREY = 5
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
SIDES = ("top", "right", "bottom", "left")
SIDE_TO_ROT = {"top": 2, "right": 1, "bottom": 0, "left": 3}
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]

GRID_MIN = 18
GRID_MAX = 30
PIECE_COUNT_MIN = 2
PIECE_COUNT_MAX = 4
SHAPE_MIN_AREA = 4
SHAPE_MAX_AREA = 8
SHAPE_CANVAS_MIN = 3
SHAPE_CANVAS_MAX = 5
INNER_PAD_MIN = 1
INNER_PAD_MAX = 2
PLACEMENT_MARGIN = 1
PLACEMENT_TRIES = 250
SHAPE_TRIES = 300
GEN_RETRIES = 400


def _neighbors4(r: int, c: int, h: int, w: int) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for dr, dc in DIR4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _components(mask: np.ndarray) -> List[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: List[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: List[Tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> Tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _mask_signature(mask: np.ndarray) -> Tuple[int, int, bytes]:
    return int(mask.shape[0]), int(mask.shape[1]), mask.astype(np.uint8).tobytes()


def _rotation_class_key(mask: np.ndarray) -> Tuple[int, int, bytes]:
    return min(_mask_signature(np.rot90(mask, k)) for k in range(4))


def _touches_grey(grid: np.ndarray, coords: np.ndarray) -> bool:
    h, w = grid.shape
    for r_raw, c_raw in coords:
        r, c = int(r_raw), int(c_raw)
        for nr, nc in _neighbors4(r, c, h, w):
            if int(grid[nr, nc]) == GREY:
                return True
    return False


def _extract_pieces(grid: np.ndarray) -> Dict[int, dict]:
    pieces: Dict[int, dict] = {}
    color_mask = (grid != BG) & (grid != GREY)
    for coords in _components(color_mask):
        color_values = {int(grid[int(r), int(c)]) for r, c in coords}
        if len(color_values) != 1:
            continue
        color = next(iter(color_values))
        if len(coords) == 1 and _touches_grey(grid, coords):
            continue  # border marker, not a loose piece
        pieces[color] = {
            "color": color,
            "coords": coords,
            "bbox": _bbox(coords),
            "mask": _mask_from_coords(coords),
        }
    return pieces


def _side_candidates(r: int, c: int, h: int, w: int) -> List[str]:
    sides: List[str] = []
    if r == 0:
        sides.append("top")
    if c == w - 1:
        sides.append("right")
    if r == h - 1:
        sides.append("bottom")
    if c == 0:
        sides.append("left")
    return sides


def _extract_molds(grid: np.ndarray) -> Dict[int, dict]:
    molds: Dict[int, dict] = {}
    for coords in _components(grid == GREY):
        r0, c0, r1, c1 = _bbox(coords)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        sh, sw = sub.shape

        marker_cells: List[Tuple[int, int, int]] = []
        for rr in range(sh):
            for cc in range(sw):
                if rr not in (0, sh - 1) and cc not in (0, sw - 1):
                    continue
                value = int(sub[rr, cc])
                if value not in (BG, GREY):
                    marker_cells.append((rr, cc, value))
        if len(marker_cells) != 1:
            continue
        mr, mc, color = marker_cells[0]

        hole_comps = []
        for hole in _components(sub == BG):
            if np.any(hole[:, 0] == 0) or np.any(hole[:, 0] == sh - 1):
                continue
            if np.any(hole[:, 1] == 0) or np.any(hole[:, 1] == sw - 1):
                continue
            hole_comps.append(hole)
        if len(hole_comps) != 1:
            continue
        hole = hole_comps[0]
        hr0, hc0, hr1, hc1 = _bbox(hole)
        hole_mask = np.zeros((hr1 - hr0 + 1, hc1 - hc0 + 1), dtype=bool)
        hole_mask[hole[:, 0] - hr0, hole[:, 1] - hc0] = True

        molds[color] = {
            "color": color,
            "bbox": (r0, c0, r1, c1),
            "marker": (r0 + mr, c0 + mc),
            "side_candidates": _side_candidates(mr, mc, sh, sw),
            "hole_bbox": (r0 + hr0, c0 + hc0, r0 + hr1, c0 + hc1),
            "hole_mask": hole_mask,
        }
    return molds


def _choose_rotated_piece(piece_mask: np.ndarray, hole_mask: np.ndarray, side_candidates: Sequence[str]) -> np.ndarray:
    for side in side_candidates:
        rotated = np.rot90(piece_mask, SIDE_TO_ROT[side])
        if rotated.shape == hole_mask.shape and np.array_equal(rotated, hole_mask):
            return rotated

    for k in range(4):
        rotated = np.rot90(piece_mask, k)
        if rotated.shape == hole_mask.shape and np.array_equal(rotated, hole_mask):
            return rotated

    raise AssertionError("piece does not match its mold under any rotation")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each scattered piece into the hole of the grey mold with the same color marker.

    The marker side encodes the intended rotation:
    top -> 180, right -> 90 CCW, bottom -> 0, left -> 90 CW.
    """
    g = np.asarray(grid, dtype=int)
    pieces = _extract_pieces(g)
    molds = _extract_molds(g)

    out = g.copy()
    for piece in pieces.values():
        for r_raw, c_raw in piece["coords"]:
            out[int(r_raw), int(c_raw)] = BG

    for color, mold in molds.items():
        assert color in pieces, "each mold marker must have a matching loose piece"
        piece_mask = pieces[color]["mask"]
        rotated = _choose_rotated_piece(piece_mask, mold["hole_mask"], mold["side_candidates"])
        hr0, hc0, hr1, hc1 = mold["hole_bbox"]
        out[hr0 : hr1 + 1, hc0 : hc1 + 1][rotated] = color

    return out


def _random_shape() -> np.ndarray:
    for _ in range(SHAPE_TRIES):
        h = random.randint(SHAPE_CANVAS_MIN, SHAPE_CANVAS_MAX)
        w = random.randint(SHAPE_CANVAS_MIN, SHAPE_CANVAS_MAX)
        max_area = min(SHAPE_MAX_AREA, h * w - 1)
        if max_area < SHAPE_MIN_AREA:
            continue
        target = random.randint(SHAPE_MIN_AREA, max_area)

        mask = np.zeros((h, w), dtype=bool)
        seed = (random.randrange(h), random.randrange(w))
        mask[seed] = True
        frontier = set(_neighbors4(seed[0], seed[1], h, w))
        while int(mask.sum()) < target and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            if mask[cell]:
                continue
            mask[cell] = True
            for nr, nc in _neighbors4(cell[0], cell[1], h, w):
                if not mask[nr, nc]:
                    frontier.add((nr, nc))

        if int(mask.sum()) != target:
            continue
        cropped = _mask_from_coords(np.argwhere(mask))
        ch, cw = cropped.shape
        if ch < 2 or cw < 2:
            continue
        if int(cropped.sum()) == ch * cw:
            continue
        if len({_mask_signature(np.rot90(cropped, k)) for k in range(4)}) < 4:
            continue
        return cropped

    raise RuntimeError("failed to sample a non-symmetric piece shape")


def _make_mold_patch(piece_mask: np.ndarray, color: int, side: str) -> dict:
    hole_mask = np.rot90(piece_mask, SIDE_TO_ROT[side])
    hh, hw = hole_mask.shape
    pad_top = random.randint(INNER_PAD_MIN, INNER_PAD_MAX)
    pad_bottom = random.randint(INNER_PAD_MIN, INNER_PAD_MAX)
    pad_left = random.randint(INNER_PAD_MIN, INNER_PAD_MAX)
    pad_right = random.randint(INNER_PAD_MIN, INNER_PAD_MAX)
    mh = hh + pad_top + pad_bottom
    mw = hw + pad_left + pad_right

    patch = np.full((mh, mw), GREY, dtype=int)
    patch[pad_top : pad_top + hh, pad_left : pad_left + hw][hole_mask] = BG

    if side == "top":
        marker = (0, random.randint(1, mw - 2))
    elif side == "right":
        marker = (random.randint(1, mh - 2), mw - 1)
    elif side == "bottom":
        marker = (mh - 1, random.randint(1, mw - 2))
    else:
        marker = (random.randint(1, mh - 2), 0)
    patch[marker] = color

    return {
        "mold_patch": patch,
        "hole_mask": hole_mask,
        "hole_offset": (pad_top, pad_left),
        "marker_local": marker,
        "side": side,
    }


def _can_place(occupied: np.ndarray, top: int, left: int, h: int, w: int, margin: int) -> bool:
    oh, ow = occupied.shape
    r0 = max(0, top - margin)
    c0 = max(0, left - margin)
    r1 = min(oh, top + h + margin)
    c1 = min(ow, left + w + margin)
    return not bool(occupied[r0:r1, c0:c1].any())


def _place_box(occupied: np.ndarray, h: int, w: int, margin: int = PLACEMENT_MARGIN) -> Tuple[int, int] | None:
    oh, ow = occupied.shape
    if h > oh or w > ow:
        return None

    for _ in range(PLACEMENT_TRIES):
        top = random.randint(0, oh - h)
        left = random.randint(0, ow - w)
        if _can_place(occupied, top, left, h, w, margin):
            occupied[top : top + h, left : left + w] = True
            return top, left

    candidates = [(r, c) for r in range(oh - h + 1) for c in range(ow - w + 1)]
    random.shuffle(candidates)
    for top, left in candidates:
        if _can_place(occupied, top, left, h, w, margin):
            occupied[top : top + h, left : left + w] = True
            return top, left
    return None


def _build_expected(grid: np.ndarray, specs: Sequence[dict]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        pr, pc = spec["piece_pos"]
        piece_mask = spec["piece_mask"]
        out[pr : pr + piece_mask.shape[0], pc : pc + piece_mask.shape[1]][piece_mask] = BG

    for spec in specs:
        mr, mc = spec["mold_pos"]
        hr, hc = spec["hole_offset"]
        hole_mask = spec["hole_mask"]
        color = spec["color"]
        out[mr + hr : mr + hr + hole_mask.shape[0], mc + hc : mc + hc + hole_mask.shape[1]][hole_mask] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(PIECE_COUNT_MIN, PIECE_COUNT_MAX)
    colors = random.sample(COLOR_POOL, count)

    shape_specs: List[dict] = []
    used_shapes = set()
    for color in colors:
        for _ in range(SHAPE_TRIES):
            piece_mask = _random_shape()
            key = _rotation_class_key(piece_mask)
            if key in used_shapes:
                continue
            used_shapes.add(key)
            side = random.choice(SIDES)
            mold = _make_mold_patch(piece_mask, color, side)
            shape_specs.append(
                {
                    "color": color,
                    "piece_mask": piece_mask,
                    "side": side,
                    **mold,
                }
            )
            break
        else:
            return None

    max_h = max(max(spec["piece_mask"].shape[0], spec["mold_patch"].shape[0]) for spec in shape_specs)
    max_w = max(max(spec["piece_mask"].shape[1], spec["mold_patch"].shape[1]) for spec in shape_specs)
    min_h = max(GRID_MIN, max_h + 4)
    min_w = max(GRID_MIN, max_w + 4)
    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None

    h = random.randint(min_h, GRID_MAX)
    w = random.randint(min_w, GRID_MAX)
    occupied = np.zeros((h, w), dtype=bool)

    for spec in sorted(shape_specs, key=lambda d: int(np.prod(d["mold_patch"].shape)), reverse=True):
        pos = _place_box(occupied, spec["mold_patch"].shape[0], spec["mold_patch"].shape[1])
        if pos is None:
            return None
        spec["mold_pos"] = pos

    for spec in sorted(shape_specs, key=lambda d: int(np.prod(d["piece_mask"].shape)), reverse=True):
        pos = _place_box(occupied, spec["piece_mask"].shape[0], spec["piece_mask"].shape[1])
        if pos is None:
            return None
        spec["piece_pos"] = pos

    grid = np.zeros((h, w), dtype=int)
    for spec in shape_specs:
        mr, mc = spec["mold_pos"]
        patch = spec["mold_patch"]
        grid[mr : mr + patch.shape[0], mc : mc + patch.shape[1]] = patch

    for spec in shape_specs:
        pr, pc = spec["piece_pos"]
        piece_mask = spec["piece_mask"]
        grid[pr : pr + piece_mask.shape[0], pc : pc + piece_mask.shape[1]][piece_mask] = spec["color"]

    expected = _build_expected(grid, shape_specs)
    molds = _extract_molds(grid)
    pieces = _extract_pieces(grid)
    if set(molds) != set(colors) or set(pieces) != set(colors):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
