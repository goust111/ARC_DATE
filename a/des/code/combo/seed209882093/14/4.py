from __future__ import annotations

import random

import numpy as np


BG = 0
FRAME_COLOR = 5
GEN_RETRIES = 400


PATTERN_POOL = [
    np.array([[1, 0, 0], [0, 0, 0]], dtype=bool),
    np.array([[0, 1, 0], [0, 0, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 0, 0]], dtype=bool),
    np.array([[1, 0, 1], [0, 0, 0]], dtype=bool),
]


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _monochrome_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "area": len(coords),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return comps


def _component_mask(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in comp["coords"]:
        mask[r - r0, c - c0] = True
    return mask


def _find_slots(grid: np.ndarray, frame: dict) -> list[dict]:
    r0, c0, r1, c1 = frame["bbox"]
    slots: list[dict] = []

    top_band = grid[r0 : r0 + 2, c0 : c1 + 1]
    top_cols = np.where(np.any(top_band == BG, axis=0))[0]
    assert top_cols.size > 0
    tc0 = int(top_cols.min())
    assert int(top_cols.max()) - tc0 + 1 == 3
    slots.append(
        {
            "side": "top",
            "r0": int(r0),
            "c0": int(c0 + tc0),
            "fill_mask": top_band[:, tc0 : tc0 + 3] == BG,
        }
    )

    bottom_band = grid[r1 - 1 : r1 + 1, c0 : c1 + 1]
    bottom_cols = np.where(np.any(bottom_band == BG, axis=0))[0]
    assert bottom_cols.size > 0
    bc0 = int(bottom_cols.min())
    assert int(bottom_cols.max()) - bc0 + 1 == 3
    slots.append(
        {
            "side": "bottom",
            "r0": int(r1 - 1),
            "c0": int(c0 + bc0),
            "fill_mask": bottom_band[:, bc0 : bc0 + 3] == BG,
        }
    )

    left_band = grid[r0 : r1 + 1, c0 : c0 + 2]
    left_rows = np.where(np.any(left_band == BG, axis=1))[0]
    assert left_rows.size > 0
    lr0 = int(left_rows.min())
    assert int(left_rows.max()) - lr0 + 1 == 3
    slots.append(
        {
            "side": "left",
            "r0": int(r0 + lr0),
            "c0": int(c0),
            "fill_mask": left_band[lr0 : lr0 + 3, :] == BG,
        }
    )

    right_band = grid[r0 : r1 + 1, c1 - 1 : c1 + 1]
    right_rows = np.where(np.any(right_band == BG, axis=1))[0]
    assert right_rows.size > 0
    rr0 = int(right_rows.min())
    assert int(right_rows.max()) - rr0 + 1 == 3
    slots.append(
        {
            "side": "right",
            "r0": int(r0 + rr0),
            "c0": int(c1 - 1),
            "fill_mask": right_band[rr0 : rr0 + 3, :] == BG,
        }
    )

    return slots


def _match_piece_to_slot(piece_mask: np.ndarray, slot_mask: np.ndarray):
    sh, sw = slot_mask.shape
    out = []
    for k in range(4):
        rotated = np.rot90(piece_mask, k)
        ph, pw = rotated.shape
        if ph > sh or pw > sw:
            continue
        for dr in range(sh - ph + 1):
            for dc in range(sw - pw + 1):
                canvas = np.zeros_like(slot_mask, dtype=bool)
                canvas[dr : dr + ph, dc : dc + pw] = rotated
                if np.array_equal(canvas, slot_mask):
                    out.append({"rotation": k, "dr": dr, "dc": dc, "canvas": canvas})
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Place each corner piece into the matching notch of the central frame.

    The match is determined by shape complement under rotation only:
    black cells in the notch are filled by the rotated piece, while remaining
    grey cells inside the notch stay visible as the slot's "teeth".
    """
    g = np.asarray(grid, dtype=int)

    comps = _monochrome_components(g)
    assert comps, "expected a frame and pieces"
    frame = max(comps, key=lambda comp: int(comp["area"]))
    slots = _find_slots(g, frame)

    pieces = []
    for comp in comps:
        if comp is frame or int(comp["color"]) == int(frame["color"]):
            continue
        pieces.append(
            {
                "color": int(comp["color"]),
                "coords": comp["coords"],
                "mask": _component_mask(comp),
            }
        )
    assert len(pieces) == len(slots), "piece count must equal slot count"

    slot_candidates: list[list[dict]] = []
    for slot in slots:
        candidates = []
        for piece_index, piece in enumerate(pieces):
            for match in _match_piece_to_slot(piece["mask"], slot["fill_mask"]):
                candidates.append({"piece_index": piece_index, "canvas": match["canvas"]})
        assert candidates, "every slot must have at least one matching piece"
        slot_candidates.append(candidates)

    order = sorted(range(len(slots)), key=lambda i: len(slot_candidates[i]))
    assignment: dict[int, dict] = {}
    used_pieces: set[int] = set()

    def backtrack(depth: int) -> bool:
        if depth == len(order):
            return True
        slot_index = order[depth]
        for cand in slot_candidates[slot_index]:
            piece_index = int(cand["piece_index"])
            if piece_index in used_pieces:
                continue
            used_pieces.add(piece_index)
            assignment[slot_index] = cand
            if backtrack(depth + 1):
                return True
            used_pieces.remove(piece_index)
            del assignment[slot_index]
        return False

    assert backtrack(0), "matching must succeed"

    out = g.copy()
    for piece in pieces:
        for r, c in piece["coords"]:
            out[r, c] = BG

    for slot_index, slot in enumerate(slots):
        piece = pieces[int(assignment[slot_index]["piece_index"])]
        canvas = assignment[slot_index]["canvas"]
        rr, cc = np.where(canvas)
        out[slot["r0"] + rr, slot["c0"] + cc] = int(piece["color"])

    return out


def _orient_pattern(pattern: np.ndarray, side: str, rng: random.Random) -> np.ndarray:
    if side in ("top", "bottom"):
        choices = [pattern.copy(), np.rot90(pattern, 2)]
    else:
        choices = [np.rot90(pattern, 1), np.rot90(pattern, 3)]
    valid = [choice for choice in choices if _protrusions_attached(choice, side)]
    assert valid, "every notch pattern must stay attached to the frame"
    return rng.choice(valid).copy()


def _protrusions_attached(protrusions: np.ndarray, side: str) -> bool:
    h, w = protrusions.shape
    cells = {(int(r), int(c)) for r, c in np.argwhere(protrusions)}
    if not cells:
        return True
    anchors: set[tuple[int, int]] = set()
    for r, c in cells:
        if side == "top":
            if r == h - 1 or (r == 0 and c in (0, w - 1)):
                anchors.add((r, c))
        elif side == "bottom":
            if r == 0 or (r == h - 1 and c in (0, w - 1)):
                anchors.add((r, c))
        elif side == "left":
            if c == w - 1 or (c == 0 and r in (0, h - 1)):
                anchors.add((r, c))
        else:
            if c == 0 or (c == w - 1 and r in (0, h - 1)):
                anchors.add((r, c))
    if not anchors:
        return False
    stack = list(anchors)
    seen = set(anchors)
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr = r + dr
            nc = c + dc
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return seen == cells


def _pick_edge_start(length: int, rng: random.Random) -> int:
    return rng.randint(2, length - 5)


def _place_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                grid[top + r, left + c] = int(color)
                coords.append((top + r, left + c))
    return coords


def _corner_position(
    corner: str,
    box: tuple[int, int, int, int],
    shape: tuple[int, int],
    rng: random.Random,
) -> tuple[int, int]:
    r_lo, r_hi, c_lo, c_hi = box
    ph, pw = shape
    assert r_hi - r_lo + 1 >= ph
    assert c_hi - c_lo + 1 >= pw
    if corner.startswith("top"):
        pr = rng.randint(r_lo, min(r_lo + 2, r_hi - ph + 1))
    else:
        pr = rng.randint(max(r_hi - ph - 1, r_lo), r_hi - ph + 1)
    if corner.endswith("left"):
        pc = rng.randint(c_lo, min(c_lo + 2, c_hi - pw + 1))
    else:
        pc = rng.randint(max(c_hi - pw - 1, c_lo), c_hi - pw + 1)
    return int(pr), int(pc)


def _build_expected(grid: np.ndarray, piece_cells: list[list[tuple[int, int]]], slot_specs: list[dict]) -> np.ndarray:
    expected = grid.copy()
    for coords in piece_cells:
        for r, c in coords:
            expected[r, c] = BG
    for spec in slot_specs:
        fill = spec["fill_mask"]
        rr, cc = np.where(fill)
        expected[spec["r0"] + rr, spec["c0"] + cc] = int(spec["color"])
    return expected


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    thickness = rng.randint(3, 4)
    inner_h = rng.randint(5, 8)
    inner_w = rng.randint(6, 10)
    outer_h = inner_h + 2 * thickness
    outer_w = inner_w + 2 * thickness

    top_margin = rng.randint(3, 5)
    bottom_margin = rng.randint(3, 5)
    left_margin = rng.randint(3, 5)
    right_margin = rng.randint(3, 5)

    h = outer_h + top_margin + bottom_margin
    w = outer_w + left_margin + right_margin
    grid = np.zeros((h, w), dtype=int)

    fr0 = top_margin
    fc0 = left_margin
    grid[fr0 : fr0 + outer_h, fc0 : fc0 + outer_w] = FRAME_COLOR
    grid[
        fr0 + thickness : fr0 + thickness + inner_h,
        fc0 + thickness : fc0 + thickness + inner_w,
    ] = BG

    chosen_patterns = rng.sample(PATTERN_POOL, 4)
    sides = ["top", "bottom", "left", "right"]
    colors = rng.sample([c for c in range(1, 10) if c != FRAME_COLOR], 4)
    slot_specs: list[dict] = []

    starts = {
        "top": _pick_edge_start(outer_w, rng),
        "bottom": _pick_edge_start(outer_w, rng),
        "left": _pick_edge_start(outer_h, rng),
        "right": _pick_edge_start(outer_h, rng),
    }

    for side, base_pattern, color in zip(sides, chosen_patterns, colors):
        protrusions = _orient_pattern(base_pattern, side, rng)
        fill_mask = ~protrusions
        if side == "top":
            sr0 = fr0
            sc0 = fc0 + starts[side]
        elif side == "bottom":
            sr0 = fr0 + outer_h - 2
            sc0 = fc0 + starts[side]
        elif side == "left":
            sr0 = fr0 + starts[side]
            sc0 = fc0
        else:
            sr0 = fr0 + starts[side]
            sc0 = fc0 + outer_w - 2

        sub = grid[sr0 : sr0 + protrusions.shape[0], sc0 : sc0 + protrusions.shape[1]]
        sub[:] = BG
        sub[protrusions] = FRAME_COLOR
        slot_specs.append(
            {
                "side": side,
                "r0": int(sr0),
                "c0": int(sc0),
                "fill_mask": fill_mask,
                "color": int(color),
            }
        )

    corners = ["top_left", "top_right", "bottom_left", "bottom_right"]
    rng.shuffle(corners)
    piece_cells: list[list[tuple[int, int]]] = []

    corner_boxes = {
        "top_left": (0, fr0 - 1, 0, fc0 - 1),
        "top_right": (0, fr0 - 1, fc0 + outer_w, w - 1),
        "bottom_left": (fr0 + outer_h, h - 1, 0, fc0 - 1),
        "bottom_right": (fr0 + outer_h, h - 1, fc0 + outer_w, w - 1),
    }

    for slot_spec, corner in zip(slot_specs, corners):
        piece_mask = slot_spec["fill_mask"]
        rotation = rng.randint(0, 3)
        rotated_piece = np.rot90(piece_mask, rotation)
        pr, pc = _corner_position(corner, corner_boxes[corner], rotated_piece.shape, rng)
        piece_cells.append(_place_mask(grid, rotated_piece, pr, pc, slot_spec["color"]))

    expected = _build_expected(grid, piece_cells, slot_specs)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
