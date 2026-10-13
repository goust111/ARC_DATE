from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
SIDES = ("top", "bottom", "left", "right")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}

# Generator configuration.
PAIR_COUNT_RANGE = (1, 3)
BASE_SIZE_RANGE = (3, 5)
COLOR_POOL = tuple(range(1, 10))
CELL_SIZE = 11
CELL_GAP = 2
CANVAS_MIN = 18
CANVAS_EXTRA_MIN = 6
CANVAS_EXTRA_MAX = 10
PLACEMENT_ATTEMPTS = 250
GEN_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(int(r), int(c))])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = int(rr + dr), int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _bbox_intersects(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> bool:
    return not (
        int(a[2]) < int(b[0])
        or int(b[2]) < int(a[0])
        or int(a[3]) < int(b[1])
        or int(b[3]) < int(a[1])
    )


def _crop_nonzero(grid: np.ndarray) -> np.ndarray:
    pts = np.argwhere(grid != BG)
    assert pts.size > 0, "expected non-empty foreground"
    r0, c0 = map(int, pts.min(axis=0))
    r1, c1 = map(int, pts.max(axis=0))
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _stamp_array(grid: np.ndarray, arr: np.ndarray, top: int, left: int) -> None:
    h, w = arr.shape
    assert 0 <= top and 0 <= left and top + h <= grid.shape[0] and left + w <= grid.shape[1]
    sub = grid[top : top + h, left : left + w]
    mask = arr != BG
    assert not np.any(sub[mask] != BG), "pieces must not overlap when stamped"
    sub[mask] = arr[mask]


def _feature_rel(side: str, offset: int, shape: tuple[int, int]) -> tuple[int, int]:
    h, w = map(int, shape)
    if side == "left":
        return int(offset), 0
    if side == "right":
        return int(offset), int(w - 1)
    if side == "top":
        return 0, int(offset)
    if side == "bottom":
        return int(h - 1), int(offset)
    raise ValueError(f"unknown side: {side}")


def _classify_piece(sub: np.ndarray) -> tuple[str, str, int]:
    mask = sub != BG
    h, w = mask.shape
    candidates: list[tuple[str, str, int]] = []

    missing = np.argwhere(~mask)
    if missing.shape[0] == 1:
        r, c = map(int, missing[0])
        sides: list[str] = []
        if r == 0:
            sides.append("top")
        if r == h - 1:
            sides.append("bottom")
        if c == 0:
            sides.append("left")
        if c == w - 1:
            sides.append("right")
        if len(sides) == 1:
            side = sides[0]
            offset = int(c if side in ("top", "bottom") else r)
            extent = int(w if side in ("top", "bottom") else h)
            if 0 < offset < extent - 1:
                candidates.append(("slot", side, offset))

    if w >= 2 and mask[:, :-1].all() and int(mask[:, -1].sum()) == 1:
        offset = int(np.flatnonzero(mask[:, -1])[0])
        if 0 < offset < h - 1:
            candidates.append(("tooth", "right", offset))
    if w >= 2 and mask[:, 1:].all() and int(mask[:, 0].sum()) == 1:
        offset = int(np.flatnonzero(mask[:, 0])[0])
        if 0 < offset < h - 1:
            candidates.append(("tooth", "left", offset))
    if h >= 2 and mask[:-1, :].all() and int(mask[-1, :].sum()) == 1:
        offset = int(np.flatnonzero(mask[-1, :])[0])
        if 0 < offset < w - 1:
            candidates.append(("tooth", "bottom", offset))
    if h >= 2 and mask[1:, :].all() and int(mask[0, :].sum()) == 1:
        offset = int(np.flatnonzero(mask[0, :])[0])
        if 0 < offset < w - 1:
            candidates.append(("tooth", "top", offset))

    assert len(candidates) == 1, "every piece must be exactly one rectangle with one tooth or one slot"
    return candidates[0]


def _extract_pieces(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    pieces: list[dict] = []
    for idx, coords in enumerate(_components(g != BG)):
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        colors = np.unique(sub[sub != BG])
        assert len(colors) == 1, "each puzzle piece must be monochrome"
        kind, side, offset = _classify_piece(sub)
        fr, fc = _feature_rel(side, offset, sub.shape)
        pieces.append(
            {
                "index": int(idx),
                "color": int(colors[0]),
                "kind": kind,
                "side": side,
                "offset": int(offset),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "coords": [(int(r), int(c)) for r, c in coords],
                "feature_abs": (int(r0 + fr), int(c0 + fc)),
            }
        )
    return pieces


def _assemble_from_pieces(pieces: list[dict]) -> np.ndarray:
    slots = [piece for piece in pieces if piece["kind"] == "slot"]
    teeth = [piece for piece in pieces if piece["kind"] == "tooth"]
    assert slots and teeth and len(slots) == len(teeth), "expected matched tooth/slot pairs only"

    translations: dict[int, tuple[int, int]] = {}
    used_teeth: set[int] = set()
    slot_order = sorted(slots, key=lambda piece: (piece["bbox"][0], piece["bbox"][1]))
    for slot in slot_order:
        matches = [
            tooth
            for tooth in teeth
            if tooth["index"] not in used_teeth
            and tooth["color"] == slot["color"]
            and tooth["side"] == OPPOSITE[slot["side"]]
        ]
        assert len(matches) == 1, "each slot must have one unique same-color tooth on the opposite side"
        tooth = matches[0]
        used_teeth.add(int(tooth["index"]))
        sr, sc = map(int, slot["feature_abs"])
        tr, tc = map(int, tooth["feature_abs"])
        translations[int(tooth["index"])] = (int(sr - tr), int(sc - tc))

    assert len(used_teeth) == len(teeth), "every tooth must be used exactly once"

    painted: dict[tuple[int, int], int] = {}
    for piece in pieces:
        dr, dc = translations.get(int(piece["index"]), (0, 0))
        for r, c in piece["coords"]:
            rr, cc = int(r + dr), int(c + dc)
            assert (rr, cc) not in painted, "assembled pieces should only meet at the filled slot cell"
            painted[(rr, cc)] = int(piece["color"])

    rows = [coord[0] for coord in painted]
    cols = [coord[1] for coord in painted]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    out = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=int)
    for (r, c), color in painted.items():
        out[int(r - r0), int(c - c0)] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Interlock monochrome rectangles by moving each toothed piece into its matching slotted piece.

    Input structure:
    - The background is black.
    - Foreground objects are separated monochrome rectangles.
    - Each rectangle has exactly one edge feature: either a one-pixel tooth that sticks out or a one-pixel slot cut into the edge.
    - Matching pieces share a color, and their tooth/slot lie on opposite facing edges.

    Rule:
    - Detect every same-color tooth-slot pair whose feature sides are opposite.
    - Keep the slotted rectangle fixed.
    - Translate the toothed rectangle so its tooth lands exactly inside the slot cell and the two rectangle edges touch.
    - If several pairs exist, match them in the top-to-bottom, left-to-right order of the slotted rectangles, then crop away outer black rows and columns.

    Key insight:
    - The single tooth or slot is both the pairing clue and the motion vector: color picks the partner, while the feature side determines how the toothed piece must be translated.
    """
    pieces = _extract_pieces(np.asarray(grid, dtype=int))
    return _assemble_from_pieces(pieces)


def _make_slot_piece(color: int, side: str, height: int, width: int, offset: int) -> np.ndarray:
    arr = np.full((int(height), int(width)), int(color), dtype=int)
    if side == "left":
        arr[int(offset), 0] = BG
    elif side == "right":
        arr[int(offset), int(width - 1)] = BG
    elif side == "top":
        arr[0, int(offset)] = BG
    elif side == "bottom":
        arr[int(height - 1), int(offset)] = BG
    else:
        raise ValueError(f"unknown side: {side}")
    return arr


def _make_tooth_piece(color: int, side: str, height: int, width: int, offset: int) -> np.ndarray:
    if side in ("left", "right"):
        arr = np.zeros((int(height), int(width + 1)), dtype=int)
        if side == "left":
            arr[:, 1:] = int(color)
            arr[int(offset), 0] = int(color)
        else:
            arr[:, :width] = int(color)
            arr[int(offset), int(width)] = int(color)
        return arr

    arr = np.zeros((int(height + 1), int(width)), dtype=int)
    if side == "top":
        arr[1:, :] = int(color)
        arr[0, int(offset)] = int(color)
    elif side == "bottom":
        arr[:height, :] = int(color)
        arr[int(height), int(offset)] = int(color)
    else:
        raise ValueError(f"unknown side: {side}")
    return arr


def _occupied_coords(arr: np.ndarray, top: int = 0, left: int = 0) -> list[tuple[int, int]]:
    pts = np.argwhere(arr != BG)
    return [(int(r + top), int(c + left)) for r, c in pts]


def _pair_spec(rng: random.Random, color: int) -> dict:
    slot_side = rng.choice(SIDES)
    tooth_side = OPPOSITE[slot_side]

    slot_h = rng.randint(*BASE_SIZE_RANGE)
    slot_w = rng.randint(*BASE_SIZE_RANGE)
    tooth_h = rng.randint(*BASE_SIZE_RANGE)
    tooth_w = rng.randint(*BASE_SIZE_RANGE)

    slot_extent = slot_w if slot_side in ("top", "bottom") else slot_h
    tooth_extent = tooth_w if tooth_side in ("top", "bottom") else tooth_h
    slot_offset = rng.randint(1, int(slot_extent - 2))
    tooth_offset = rng.randint(1, int(tooth_extent - 2))

    slot_arr = _make_slot_piece(color, slot_side, slot_h, slot_w, slot_offset)
    tooth_arr = _make_tooth_piece(color, tooth_side, tooth_h, tooth_w, tooth_offset)

    assert _classify_piece(slot_arr) == ("slot", slot_side, slot_offset)
    assert _classify_piece(tooth_arr) == ("tooth", tooth_side, tooth_offset)

    slot_feature = _feature_rel(slot_side, slot_offset, slot_arr.shape)
    tooth_feature = _feature_rel(tooth_side, tooth_offset, tooth_arr.shape)
    tooth_shift = (
        int(slot_feature[0] - tooth_feature[0]),
        int(slot_feature[1] - tooth_feature[1]),
    )

    slot_coords = _occupied_coords(slot_arr)
    tooth_coords = _occupied_coords(tooth_arr, tooth_shift[0], tooth_shift[1])
    all_coords = np.array(slot_coords + tooth_coords, dtype=int)
    r0, c0, r1, c1 = _bbox_from_coords(all_coords)

    return {
        "color": int(color),
        "slot_side": slot_side,
        "tooth_side": tooth_side,
        "slot_arr": slot_arr,
        "tooth_arr": tooth_arr,
        "tooth_shift": tooth_shift,
        "rel_bbox": (int(r0), int(c0), int(r1), int(c1)),
        "bbox_size": (int(r1 - r0 + 1), int(c1 - c0 + 1)),
        "slot_final_top": (0, 0),
        "tooth_final_top": tooth_shift,
    }


def _choose_cells(pair_count: int, rng: random.Random) -> list[tuple[int, int]]:
    if pair_count == 1:
        rows, cols = 1, 1
    elif pair_count == 2:
        rows, cols = rng.choice([(1, 2), (2, 1)])
    else:
        rows, cols = 2, 2
    cells = [(int(r), int(c)) for r in range(rows) for c in range(cols)]
    rng.shuffle(cells)
    return cells[:pair_count]


def _place_pairs_on_layout(pairs: list[dict], rng: random.Random) -> tuple[int, int]:
    cells = _choose_cells(len(pairs), rng)
    for pair, (cell_r, cell_c) in zip(pairs, cells):
        box_h, box_w = map(int, pair["bbox_size"])
        if box_h > CELL_SIZE or box_w > CELL_SIZE:
            raise AssertionError("pair bbox must fit in its placement cell")
        cell_top = int(cell_r * (CELL_SIZE + CELL_GAP))
        cell_left = int(cell_c * (CELL_SIZE + CELL_GAP))
        bbox_top = int(cell_top + rng.randint(0, CELL_SIZE - box_h))
        bbox_left = int(cell_left + rng.randint(0, CELL_SIZE - box_w))
        rel_r0, rel_c0, _, _ = map(int, pair["rel_bbox"])
        shift_r = int(bbox_top - rel_r0)
        shift_c = int(bbox_left - rel_c0)
        pair["slot_final_top"] = (shift_r, shift_c)
        pair["tooth_final_top"] = (
            int(shift_r + pair["tooth_shift"][0]),
            int(shift_c + pair["tooth_shift"][1]),
        )

    all_coords: list[tuple[int, int]] = []
    for pair in pairs:
        all_coords.extend(_occupied_coords(pair["slot_arr"], *pair["slot_final_top"]))
        all_coords.extend(_occupied_coords(pair["tooth_arr"], *pair["tooth_final_top"]))
    coords = np.array(all_coords, dtype=int)
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    for pair in pairs:
        pair["slot_final_top"] = (
            int(pair["slot_final_top"][0] - r0),
            int(pair["slot_final_top"][1] - c0),
        )
        pair["tooth_final_top"] = (
            int(pair["tooth_final_top"][0] - r0),
            int(pair["tooth_final_top"][1] - c0),
        )
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _bbox_clear(occupied: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> bool:
    h, w = occupied.shape
    r0 = max(0, int(bbox[0] - pad))
    c0 = max(0, int(bbox[1] - pad))
    r1 = min(int(h - 1), int(bbox[2] + pad))
    c1 = min(int(w - 1), int(bbox[3] + pad))
    return not bool(occupied[r0 : r1 + 1, c0 : c1 + 1].any())


def _render_expected(pairs: list[dict], shape: tuple[int, int], shift: tuple[int, int]) -> np.ndarray:
    h, w = map(int, shape)
    sr, sc = map(int, shift)
    canvas = np.zeros((h, w), dtype=int)
    for pair in pairs:
        slot_top = (int(pair["slot_final_top"][0] + sr), int(pair["slot_final_top"][1] + sc))
        tooth_top = (int(pair["tooth_final_top"][0] + sr), int(pair["tooth_final_top"][1] + sc))
        _stamp_array(canvas, pair["slot_arr"], *slot_top)
        _stamp_array(canvas, pair["tooth_arr"], *tooth_top)
    return _crop_nonzero(canvas)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.randint(*PAIR_COUNT_RANGE)
    colors = rng.sample(COLOR_POOL, pair_count)
    pairs = [_pair_spec(rng, color) for color in colors]
    layout_h, layout_w = _place_pairs_on_layout(pairs, rng)

    min_h = max(CANVAS_MIN, int(layout_h + CANVAS_EXTRA_MIN))
    min_w = max(CANVAS_MIN, int(layout_w + CANVAS_EXTRA_MIN))
    max_h = min(30, int(layout_h + CANVAS_EXTRA_MAX))
    max_w = min(30, int(layout_w + CANVAS_EXTRA_MAX))
    if min_h > max_h or min_w > max_w:
        return None

    h = rng.randint(min_h, max_h)
    w = rng.randint(min_w, max_w)
    max_top = int(h - layout_h - 1)
    max_left = int(w - layout_w - 1)
    if max_top < 1 or max_left < 1:
        return None

    layout_top = rng.randint(1, min(3, max_top))
    layout_left = rng.randint(1, min(3, max_left))
    layout_shift = (int(layout_top), int(layout_left))

    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    for pair in pairs:
        slot_top = (
            int(pair["slot_final_top"][0] + layout_shift[0]),
            int(pair["slot_final_top"][1] + layout_shift[1]),
        )
        _stamp_array(grid, pair["slot_arr"], *slot_top)
        occupied[slot_top[0] : slot_top[0] + pair["slot_arr"].shape[0], slot_top[1] : slot_top[1] + pair["slot_arr"].shape[1]] |= (
            pair["slot_arr"] != BG
        )

    layout_box = (
        max(0, int(layout_top - 1)),
        max(0, int(layout_left - 1)),
        min(int(h - 1), int(layout_top + layout_h)),
        min(int(w - 1), int(layout_left + layout_w)),
    )

    scatter_order = sorted(
        pairs,
        key=lambda pair: int(np.count_nonzero(pair["tooth_arr"])),
        reverse=True,
    )
    for pair in scatter_order:
        arr = pair["tooth_arr"]
        ph, pw = map(int, arr.shape)
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            top = rng.randint(0, int(h - ph))
            left = rng.randint(0, int(w - pw))
            bbox = (int(top), int(left), int(top + ph - 1), int(left + pw - 1))
            if _bbox_intersects(bbox, layout_box):
                continue
            if not _bbox_clear(occupied, bbox, pad=1):
                continue
            _stamp_array(grid, arr, top, left)
            occupied[top : top + ph, left : left + pw] |= arr != BG
            placed = True
            break
        if not placed:
            return None

    try:
        pieces = _extract_pieces(grid)
        if len(pieces) != pair_count * 2:
            return None
        expected = _render_expected(pairs, (h, w), layout_shift)
        solved = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {GEN_ATTEMPTS} attempts")
