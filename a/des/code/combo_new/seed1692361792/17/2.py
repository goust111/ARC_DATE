from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
GEN_FRAME_COLOR = 5
SIDES = ("top", "right", "bottom", "left")
OPPOSITE = {
    "top": "bottom",
    "right": "left",
    "bottom": "top",
    "left": "right",
}
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_MAX_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _boundary_side(shape: tuple[int, int], pos: tuple[int, int]) -> str | None:
    r, c = map(int, pos)
    h, w = map(int, shape)
    if r == 0 and 0 < c < w - 1:
        return "top"
    if c == w - 1 and 0 < r < h - 1:
        return "right"
    if r == h - 1 and 0 < c < w - 1:
        return "bottom"
    if c == 0 and 0 < r < h - 1:
        return "left"
    return None


def _slot_from_component(
    sub: np.ndarray,
    origin_r: int,
    origin_c: int,
    coords: np.ndarray,
    frame_color: int,
) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if r0 == 0 or c0 == 0 or r1 == sub.shape[0] - 1 or c1 == sub.shape[1] - 1:
        return None
    region = sub[r0 : r1 + 1, c0 : c1 + 1]
    if bool(np.any(region == int(frame_color))):
        return None
    if not bool(np.all((region == BG) | (region != int(frame_color)))):
        return None
    marker_positions = np.argwhere((region != BG) & (region != int(frame_color)))
    if marker_positions.shape[0] != 1:
        return None
    mr, mc = map(int, marker_positions[0])
    side = _boundary_side((h, w), (mr, mc))
    if side is None:
        return None
    color = int(region[mr, mc])
    return {
        "bbox": (int(origin_r + r0), int(origin_c + c0), int(origin_r + r1), int(origin_c + c1)),
        "shape": (h, w),
        "lock_color": color,
        "lock_pos": (int(origin_r + r0 + mr), int(origin_c + c0 + mc)),
        "lock_pos_local": (mr, mc),
        "lock_side": side,
    }


def _piece_from_component(grid: np.ndarray, coords: np.ndarray, lock_colors: set[int]) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.array(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
    if bool(np.any(patch == BG)):
        return None
    key_positions = np.argwhere(np.isin(patch, list(lock_colors)))
    if key_positions.shape[0] != 1:
        return None
    kr, kc = map(int, key_positions[0])
    key_side = _boundary_side(patch.shape, (kr, kc))
    if key_side is None:
        return None
    return {
        "bbox": (r0, c0, r1, c1),
        "patch": patch,
        "key_color": int(patch[kr, kc]),
        "key_pos": (kr, kc),
        "key_side": key_side,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Insert each keyed pattern block into the matching colored slot inside the grey frame.

    Input structure:
    - The background is black.
    - One large grey connected frame-like rectangle occupies most of the grid; inside it are several rectangular slot regions made of black cells plus one colored lock pixel on a slot edge.
    - Outside the frame are several full rectangular multicolor blocks, each with one boundary key pixel whose color matches exactly one slot lock.

    Rule:
    - Find the large grey frame and the enclosed rectangular slot regions inside its bounding box.
    - Match each outside block to a slot by lock-key color, then rotate the block by 0/90/180/270 degrees so its key side becomes opposite the slot's lock side and its rectangle matches the slot size.
    - Clear the scattered source blocks, fill each slot with its matched rotated block, and erase both the block's key pixel and the slot's lock pixel to black.

    Key insight:
    - Color determines which block belongs to which slot, but the slot-edge direction is a second independent clue that fixes the required rotation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    non_bg_components = _components(g != BG)
    assert non_bg_components, "expected at least one foreground component"
    frame_component = max(non_bg_components, key=lambda comp: int(comp.shape[0]))

    frame_colors = [int(g[r, c]) for r, c in frame_component if int(g[r, c]) != BG]
    frame_color = int(Counter(frame_colors).most_common(1)[0][0])
    fr0, fc0, fr1, fc1 = _bbox(frame_component)
    sub = np.array(g[fr0 : fr1 + 1, fc0 : fc1 + 1], dtype=int)

    slots: list[dict] = []
    for comp in _components(sub != int(frame_color)):
        slot = _slot_from_component(sub, fr0, fc0, comp, frame_color)
        if slot is not None:
            slots.append(slot)
    assert slots, "expected at least one enclosed slot"

    lock_colors = {int(slot["lock_color"]) for slot in slots}
    pieces: list[dict] = []
    for comp in non_bg_components:
        if comp is frame_component:
            continue
        piece = _piece_from_component(g, comp, lock_colors)
        if piece is not None:
            pieces.append(piece)
    assert len(pieces) == len(slots), "expected one outside piece per slot"

    slots_by_color = {int(slot["lock_color"]): slot for slot in slots}
    pieces_by_color = {int(piece["key_color"]): piece for piece in pieces}
    assert set(slots_by_color) == set(pieces_by_color), "lock and key colors must match exactly"

    out = np.array(g, copy=True)
    for piece in pieces:
        r0, c0, r1, c1 = piece["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = BG

    for color, slot in slots_by_color.items():
        piece = pieces_by_color[int(color)]
        target_side = OPPOSITE[str(slot["lock_side"])]
        placed_patch = None
        placed_key_pos = None
        for k in range(4):
            rotated = np.rot90(piece["patch"], int(k))
            key_positions = np.argwhere(rotated == int(color))
            if key_positions.shape[0] != 1:
                continue
            kr, kc = map(int, key_positions[0])
            if rotated.shape != tuple(slot["shape"]):
                continue
            if _boundary_side(rotated.shape, (kr, kc)) != target_side:
                continue
            placed_patch = rotated
            placed_key_pos = (kr, kc)
            break
        assert placed_patch is not None and placed_key_pos is not None, "piece rotation must be resolvable"

        sr0, sc0, sr1, sc1 = slot["bbox"]
        out[sr0 : sr1 + 1, sc0 : sc1 + 1] = placed_patch
        out[sr0 + placed_key_pos[0], sc0 + placed_key_pos[1]] = BG
        lr, lc = slot["lock_pos"]
        out[lr, lc] = BG

    return out


def _pattern_patch(shape: tuple[int, int], colors: list[int], rng: random.Random) -> np.ndarray:
    h, w = map(int, shape)
    mode = rng.choice(["checker", "hstripe", "vstripe", "diag", "blocks"])
    patch = np.zeros((h, w), dtype=int)
    n = len(colors)
    for r in range(h):
        for c in range(w):
            if mode == "checker":
                idx = (r + c) % n
            elif mode == "hstripe":
                idx = r % n
            elif mode == "vstripe":
                idx = c % n
            elif mode == "diag":
                idx = (2 * r + c) % n
            else:
                idx = ((r // 2) + (c // 2)) % n
            patch[r, c] = int(colors[idx])
    return patch


def _side_position(side: str, shape: tuple[int, int], offset: int) -> tuple[int, int]:
    h, w = map(int, shape)
    if side == "top":
        return 0, int(offset)
    if side == "right":
        return int(offset), w - 1
    if side == "bottom":
        return h - 1, int(offset)
    return int(offset), 0


def _make_piece(
    shape: tuple[int, int],
    lock_side: str,
    key_color: int,
    body_colors: list[int],
    rng: random.Random,
) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = map(int, shape)
    if lock_side in ("top", "bottom"):
        offset = rng.randint(1, w - 2)
    else:
        offset = rng.randint(1, h - 2)
    key_side = OPPOSITE[str(lock_side)]
    patch = _pattern_patch((h, w), body_colors, rng)
    key_pos = _side_position(key_side, (h, w), offset)
    patch[key_pos] = int(key_color)
    return patch, key_pos


def _can_place(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int, padding: int) -> bool:
    h, w = map(int, patch.shape)
    hh, ww = map(int, canvas.shape)
    if r0 < 0 or c0 < 0 or r0 + h > hh or c0 + w > ww:
        return False
    r_lo = max(0, int(r0 - padding))
    c_lo = max(0, int(c0 - padding))
    r_hi = min(hh, int(r0 + h + padding))
    c_hi = min(ww, int(c0 + w + padding))
    return not bool(np.any(canvas[r_lo:r_hi, c_lo:c_hi] != BG))


def _random_place(canvas: np.ndarray, patch: np.ndarray, padding: int, rng: random.Random) -> tuple[int, int] | None:
    hh, ww = map(int, canvas.shape)
    h, w = map(int, patch.shape)
    positions = [(r, c) for r in range(hh - h + 1) for c in range(ww - w + 1)]
    rng.shuffle(positions)
    for r0, c0 in positions:
        if _can_place(canvas, patch, int(r0), int(c0), padding):
            return int(r0), int(c0)
    return None


def _place_slots(
    board_h: int,
    board_w: int,
    specs: list[tuple[tuple[int, int], str, int]],
    rng: random.Random,
) -> list[dict] | None:
    occupied = np.zeros((board_h, board_w), dtype=bool)
    slots: list[dict] = []
    ordered = sorted(specs, key=lambda spec: -(spec[0][0] * spec[0][1]))
    for (slot_h, slot_w), side, color in ordered:
        positions = [
            (r0, c0)
            for r0 in range(1, board_h - slot_h)
            for c0 in range(1, board_w - slot_w)
        ]
        rng.shuffle(positions)
        placed = False
        for r0, c0 in positions:
            r1 = r0 + slot_h - 1
            c1 = c0 + slot_w - 1
            if r1 >= board_h - 1 or c1 >= board_w - 1:
                continue
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(board_h, r1 + 2)
            cc1 = min(board_w, c1 + 2)
            if bool(np.any(occupied[rr0:rr1, cc0:cc1])):
                continue
            occupied[r0 : r1 + 1, c0 : c1 + 1] = True
            slots.append(
                {
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "shape": (int(slot_h), int(slot_w)),
                    "lock_side": side,
                    "lock_color": int(color),
                }
            )
            placed = True
            break
        if not placed:
            return None
    return slots


def _build_board(board_h: int, board_w: int, slots: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    board_in = np.full((board_h, board_w), GEN_FRAME_COLOR, dtype=int)
    board_out = np.full((board_h, board_w), GEN_FRAME_COLOR, dtype=int)
    for slot in slots:
        r0, c0, r1, c1 = slot["bbox"]
        h, w = slot["shape"]
        board_in[r0 : r1 + 1, c0 : c1 + 1] = BG
        piece_patch = slot["piece_patch"]
        board_out[r0 : r1 + 1, c0 : c1 + 1] = piece_patch

        lock_r_local, lock_c_local = slot["lock_local"]
        key_r_local, key_c_local = slot["key_local"]
        board_in[r0 + lock_r_local, c0 + lock_c_local] = int(slot["lock_color"])
        board_out[r0 + lock_r_local, c0 + lock_c_local] = BG
        board_out[r0 + key_r_local, c0 + key_c_local] = BG

    return board_in, board_out


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    n_slots = rng.randint(2, 4)
    chosen_sides = rng.sample(list(SIDES), n_slots)
    lock_colors = rng.sample(PALETTE, n_slots)

    min_board = 11 + n_slots
    board_h = rng.randint(min_board, min(18, 23 - n_slots))
    board_w = rng.randint(min_board, min(18, 23 - n_slots))

    slot_specs: list[tuple[tuple[int, int], str, int]] = []
    for side, color in zip(chosen_sides, lock_colors):
        slot_h = rng.randint(3, 5)
        slot_w = rng.randint(3, 5)
        slot_specs.append(((slot_h, slot_w), side, color))

    slots = _place_slots(board_h, board_w, slot_specs, rng)
    if slots is None:
        return None

    remaining_body_colors = [c for c in PALETTE if c not in lock_colors]
    rng.shuffle(remaining_body_colors)
    for slot in slots:
        n_body = rng.randint(2, min(3, len(remaining_body_colors)))
        body_colors = rng.sample(remaining_body_colors, n_body)
        piece_patch, key_pos = _make_piece(
            slot["shape"],
            str(slot["lock_side"]),
            int(slot["lock_color"]),
            body_colors,
            rng,
        )
        h, w = slot["shape"]
        if slot["lock_side"] in ("top", "bottom"):
            offset = key_pos[1]
        else:
            offset = key_pos[0]
        lock_pos = _side_position(str(slot["lock_side"]), (h, w), int(offset))
        slot["piece_patch"] = piece_patch
        slot["key_local"] = key_pos
        slot["lock_local"] = lock_pos
        slot["scatter_patch"] = np.rot90(piece_patch, int(rng.randint(0, 3)))

    board_in, board_out = _build_board(board_h, board_w, slots)

    canvas_h = rng.randint(max(board_h + 6, 20), 30)
    canvas_w = rng.randint(max(board_w + 6, 20), 30)
    inp = np.zeros((canvas_h, canvas_w), dtype=int)
    expected = np.zeros((canvas_h, canvas_w), dtype=int)

    board_pos = _random_place(inp, board_in, padding=2, rng=rng)
    if board_pos is None:
        return None
    br, bc = board_pos
    inp[br : br + board_h, bc : bc + board_w] = board_in
    expected[br : br + board_h, bc : bc + board_w] = board_out

    placement_canvas = np.array(inp, copy=True)
    rng.shuffle(slots)
    for slot in slots:
        patch = slot["scatter_patch"]
        pos = _random_place(placement_canvas, patch, padding=1, rng=rng)
        if pos is None:
            return None
        pr, pc = pos
        placement_canvas[pr : pr + patch.shape[0], pc : pc + patch.shape[1]] = patch
        inp[pr : pr + patch.shape[0], pc : pc + patch.shape[1]] = patch

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after repeated attempts")
