import random

import numpy as np


BG = 0
FRAME = 5
MAX_RETRIES = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _find_separator(grid: np.ndarray, bg: int) -> tuple[int, int]:
    h, w = grid.shape
    candidates = []
    for c in range(1, w - 1):
        col = grid[:, c]
        if not np.all(col == col[0]):
            continue
        color = int(col[0])
        if color == int(bg):
            continue
        side_bg = int((grid[:, c - 1] == bg).sum()) + int((grid[:, c + 1] == bg).sum())
        candidates.append((side_bg, -abs(c - w // 2), int(c), color))
    assert candidates, "separator column not found"
    _, _, sep_col, color = max(candidates)
    return int(sep_col), int(color)


def _frame_slots(
    grid: np.ndarray, sep_col: int, frame_color: int
) -> tuple[tuple[int, int, int, int], list[dict], int, int]:
    mask = np.zeros_like(grid, dtype=bool)
    mask[:, :sep_col] = grid[:, :sep_col] == int(frame_color)
    comps = _components(mask)
    assert comps, "frame component not found"
    frame_coords = max(comps, key=lambda x: int(x.shape[0]))
    fr0, fc0, fr1, fc1 = _bbox(frame_coords)

    sub = grid[fr0 : fr1 + 1, fc0 : fc1 + 1]
    row_counts = np.sum(sub == int(frame_color), axis=1)
    col_counts = np.sum(sub == int(frame_color), axis=0)
    cross_r = int(np.argmax(row_counts[1:-1])) + 1
    cross_c = int(np.argmax(col_counts[1:-1])) + 1

    slot_h_top = int(cross_r - 1)
    slot_h_bottom = int(sub.shape[0] - cross_r - 2)
    slot_w_left = int(cross_c - 1)
    slot_w_right = int(sub.shape[1] - cross_c - 2)
    assert slot_h_top == slot_h_bottom and slot_w_left == slot_w_right
    slot_h = int(slot_h_top)
    slot_w = int(slot_w_left)
    assert slot_h >= 3 and slot_w >= 3

    hr = int(fr0 + cross_r)
    vc = int(fc0 + cross_c)
    slots = [
        {"r0": int(fr0 + 1), "c0": int(fc0 + 1), "top": int(fr0), "bottom": int(hr), "left": int(fc0), "right": int(vc)},
        {"r0": int(fr0 + 1), "c0": int(vc + 1), "top": int(fr0), "bottom": int(hr), "left": int(vc), "right": int(fc1)},
        {"r0": int(hr + 1), "c0": int(fc0 + 1), "top": int(hr), "bottom": int(fr1), "left": int(fc0), "right": int(vc)},
        {"r0": int(hr + 1), "c0": int(vc + 1), "top": int(hr), "bottom": int(fr1), "left": int(vc), "right": int(fc1)},
    ]
    return (int(fr0), int(fc0), int(fr1), int(fc1)), slots, slot_h, slot_w


def _slot_signature(
    grid: np.ndarray, slot: dict, slot_h: int, slot_w: int, frame_color: int
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    r0 = int(slot["r0"])
    c0 = int(slot["c0"])

    def top() -> tuple[int, ...]:
        vals = []
        for dc in range(slot_w):
            boundary = int(grid[int(slot["top"]), c0 + dc]) == int(frame_color)
            inner = int(grid[r0, c0 + dc]) == int(frame_color)
            vals.append(-1 if not boundary else 1 if inner else 0)
        return tuple(vals)

    def bottom() -> tuple[int, ...]:
        vals = []
        rr = int(r0 + slot_h - 1)
        for dc in range(slot_w):
            boundary = int(grid[int(slot["bottom"]), c0 + dc]) == int(frame_color)
            inner = int(grid[rr, c0 + dc]) == int(frame_color)
            vals.append(-1 if not boundary else 1 if inner else 0)
        return tuple(vals)

    def left() -> tuple[int, ...]:
        vals = []
        for dr in range(slot_h):
            boundary = int(grid[r0 + dr, int(slot["left"])]) == int(frame_color)
            inner = int(grid[r0 + dr, c0]) == int(frame_color)
            vals.append(-1 if not boundary else 1 if inner else 0)
        return tuple(vals)

    def right() -> tuple[int, ...]:
        vals = []
        cc = int(c0 + slot_w - 1)
        for dr in range(slot_h):
            boundary = int(grid[r0 + dr, int(slot["right"])]) == int(frame_color)
            inner = int(grid[r0 + dr, cc]) == int(frame_color)
            vals.append(-1 if not boundary else 1 if inner else 0)
        return tuple(vals)

    return (top(), bottom(), left(), right())


def _candidate_signature(
    mask: np.ndarray, slot_h: int, slot_w: int, top_ext: int, bottom_ext: int, left_ext: int, right_ext: int
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]] | None:
    ph, pw = mask.shape
    if top_ext + slot_h + bottom_ext != ph or left_ext + slot_w + right_ext != pw:
        return None

    allowed = np.zeros_like(mask, dtype=bool)
    allowed[top_ext : top_ext + slot_h, left_ext : left_ext + slot_w] = True
    if top_ext:
        allowed[0, left_ext : left_ext + slot_w] = True
    if bottom_ext:
        allowed[ph - 1, left_ext : left_ext + slot_w] = True
    if left_ext:
        allowed[top_ext : top_ext + slot_h, 0] = True
    if right_ext:
        allowed[top_ext : top_ext + slot_h, pw - 1] = True
    if np.any(mask & ~allowed):
        return None

    base = mask[top_ext : top_ext + slot_h, left_ext : left_ext + slot_w]
    if not base[1:-1, 1:-1].all():
        return None
    if not bool(base[0, 0] and base[0, -1] and base[-1, 0] and base[-1, -1]):
        return None

    for r in range(slot_h):
        for c in range(slot_w):
            if base[r, c]:
                continue
            if r not in (0, slot_h - 1) and c not in (0, slot_w - 1):
                return None
            if r in (0, slot_h - 1) and c in (0, slot_w - 1):
                return None

    if top_ext and (mask[0, left_ext] or mask[0, left_ext + slot_w - 1]):
        return None
    if bottom_ext and (mask[ph - 1, left_ext] or mask[ph - 1, left_ext + slot_w - 1]):
        return None
    if left_ext and (mask[top_ext, 0] or mask[top_ext + slot_h - 1, 0]):
        return None
    if right_ext and (mask[top_ext, pw - 1] or mask[top_ext + slot_h - 1, pw - 1]):
        return None

    def top() -> tuple[int, ...]:
        vals = []
        for c in range(slot_w):
            base_cell = bool(base[0, c])
            outer = bool(top_ext and mask[0, left_ext + c])
            if outer and not base_cell:
                raise ValueError
            vals.append(1 if outer else 0 if base_cell else -1)
        return tuple(vals)

    def bottom() -> tuple[int, ...]:
        vals = []
        for c in range(slot_w):
            base_cell = bool(base[slot_h - 1, c])
            outer = bool(bottom_ext and mask[ph - 1, left_ext + c])
            if outer and not base_cell:
                raise ValueError
            vals.append(1 if outer else 0 if base_cell else -1)
        return tuple(vals)

    def left() -> tuple[int, ...]:
        vals = []
        for r in range(slot_h):
            base_cell = bool(base[r, 0])
            outer = bool(left_ext and mask[top_ext + r, 0])
            if outer and not base_cell:
                raise ValueError
            vals.append(1 if outer else 0 if base_cell else -1)
        return tuple(vals)

    def right() -> tuple[int, ...]:
        vals = []
        for r in range(slot_h):
            base_cell = bool(base[r, slot_w - 1])
            outer = bool(right_ext and mask[top_ext + r, pw - 1])
            if outer and not base_cell:
                raise ValueError
            vals.append(1 if outer else 0 if base_cell else -1)
        return tuple(vals)

    try:
        return (top(), bottom(), left(), right())
    except ValueError:
        return None


def _piece_candidates(mask: np.ndarray, slot_h: int, slot_w: int) -> list[dict]:
    ph, pw = mask.shape
    extra_h = int(ph - slot_h)
    extra_w = int(pw - slot_w)
    if extra_h not in (0, 1, 2) or extra_w not in (0, 1, 2):
        return []

    found = []
    seen = set()
    for top_ext in (0, 1):
        bottom_ext = int(extra_h - top_ext)
        if bottom_ext not in (0, 1):
            continue
        for left_ext in (0, 1):
            right_ext = int(extra_w - left_ext)
            if right_ext not in (0, 1):
                continue
            sig = _candidate_signature(mask, slot_h, slot_w, top_ext, bottom_ext, left_ext, right_ext)
            if sig is None:
                continue
            key = (sig, int(top_ext), int(left_ext))
            if key in seen:
                continue
            seen.add(key)
            found.append({"signature": sig, "top_ext": int(top_ext), "left_ext": int(left_ext)})
    return found


def _extract_pieces(
    grid: np.ndarray, sep_col: int, bg: int, frame_color: int, slot_h: int, slot_w: int
) -> list[dict]:
    mask = np.zeros_like(grid, dtype=bool)
    right = grid[:, sep_col + 1 :]
    mask[:, sep_col + 1 :] = (right != int(bg)) & (right != int(frame_color))
    comps = _components(mask)
    assert len(comps) == 4, "expected exactly four pieces"

    pieces = []
    for coords in comps:
        r0, c0, r1, c1 = _bbox(coords)
        patch_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        patch_mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
        colors = np.unique(grid[coords[:, 0], coords[:, 1]])
        assert len(colors) == 1
        color = int(colors[0])
        patch = np.full(patch_mask.shape, int(bg), dtype=int)
        patch[patch_mask] = int(color)
        candidates = _piece_candidates(patch_mask, slot_h, slot_w)
        assert candidates, "piece does not match slot dimensions"
        pieces.append({"color": int(color), "patch": patch, "mask": patch_mask, "candidates": candidates})
    return pieces


def _negate_signature(
    sig: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    return tuple(tuple(-v for v in edge) for edge in sig)


def _match_slots(slot_sigs: list, pieces: list[dict]) -> list[tuple[int, int]]:
    options = []
    for slot_sig in slot_sigs:
        want = _negate_signature(slot_sig)
        cur = []
        for pi, piece in enumerate(pieces):
            for ci, cand in enumerate(piece["candidates"]):
                if cand["signature"] == want:
                    cur.append((int(pi), int(ci)))
        assert cur, "slot has no matching piece"
        options.append(cur)

    order = sorted(range(len(slot_sigs)), key=lambda i: len(options[i]))
    assignment: list[tuple[int, int] | None] = [None] * len(slot_sigs)
    used: set[int] = set()

    def dfs(idx: int) -> bool:
        if idx == len(order):
            return True
        slot_idx = int(order[idx])
        for piece_idx, cand_idx in options[slot_idx]:
            if int(piece_idx) in used:
                continue
            used.add(int(piece_idx))
            assignment[slot_idx] = (int(piece_idx), int(cand_idx))
            if dfs(idx + 1):
                return True
            used.remove(int(piece_idx))
            assignment[slot_idx] = None
        return False

    assert dfs(0), "no bijection between slots and pieces"
    return [x for x in assignment if x is not None]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    sep_col, frame_color = _find_separator(g, bg)
    frame_bbox, slots, slot_h, slot_w = _frame_slots(g, sep_col, frame_color)
    fr0, fc0, fr1, fc1 = frame_bbox

    slot_sigs = [_slot_signature(g, slot, slot_h, slot_w, frame_color) for slot in slots]
    pieces = _extract_pieces(g, sep_col, bg, frame_color, slot_h, slot_w)
    assignment = _match_slots(slot_sigs, pieces)

    out = g[fr0 : fr1 + 1, fc0 : fc1 + 1].copy()
    for slot_idx, (piece_idx, cand_idx) in enumerate(assignment):
        slot = slots[slot_idx]
        piece = pieces[piece_idx]
        cand = piece["candidates"][cand_idx]
        top = int(slot["r0"] - cand["top_ext"] - fr0)
        left = int(slot["c0"] - cand["left_ext"] - fc0)
        rr, cc = np.nonzero(piece["mask"])
        out[top + rr, left + cc] = int(piece["color"])
    return out


def _random_edge(length: int, rng: random.Random, allow_flat: bool) -> tuple[int, ...]:
    arr = [0] * length
    if allow_flat and rng.random() < 0.25:
        return tuple(arr)
    positions = list(range(1, length - 1))
    count = rng.randint(1, min(2, len(positions)))
    for pos in rng.sample(positions, count):
        arr[pos] = rng.choice([-1, 1])
    return tuple(arr)


def _generate_slot_signatures(
    slot_h: int, slot_w: int, rng: random.Random
) -> list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]]:
    zero_h = (0,) * slot_h
    zero_w = (0,) * slot_w
    for _ in range(200):
        tl_top = _random_edge(slot_w, rng, allow_flat=True)
        tl_left = _random_edge(slot_h, rng, allow_flat=True)
        tr_top = _random_edge(slot_w, rng, allow_flat=True)
        tr_right = _random_edge(slot_h, rng, allow_flat=True)
        bl_bottom = _random_edge(slot_w, rng, allow_flat=True)
        bl_left = _random_edge(slot_h, rng, allow_flat=True)
        br_bottom = _random_edge(slot_w, rng, allow_flat=True)
        br_right = _random_edge(slot_h, rng, allow_flat=True)

        if tl_top == zero_w and tl_left == zero_h:
            continue
        if tr_top == zero_w and tr_right == zero_h:
            continue
        if bl_bottom == zero_w and bl_left == zero_h:
            continue
        if br_bottom == zero_w and br_right == zero_h:
            continue

        sigs = [
            (tl_top, zero_w, tl_left, zero_h),
            (tr_top, zero_w, zero_h, tr_right),
            (zero_w, bl_bottom, bl_left, zero_h),
            (zero_w, br_bottom, zero_h, br_right),
        ]
        if len(set(sigs)) == 4:
            return sigs
    raise RuntimeError("failed to make distinct slot signatures")


def _local_slots(slot_h: int, slot_w: int) -> list[dict]:
    cross_r = int(slot_h + 1)
    cross_c = int(slot_w + 1)
    frame_h = int(2 * slot_h + 3)
    frame_w = int(2 * slot_w + 3)
    return [
        {"r0": 1, "c0": 1, "top": 0, "bottom": cross_r, "left": 0, "right": cross_c},
        {"r0": 1, "c0": cross_c + 1, "top": 0, "bottom": cross_r, "left": cross_c, "right": frame_w - 1},
        {"r0": cross_r + 1, "c0": 1, "top": cross_r, "bottom": frame_h - 1, "left": 0, "right": cross_c},
        {"r0": cross_r + 1, "c0": cross_c + 1, "top": cross_r, "bottom": frame_h - 1, "left": cross_c, "right": frame_w - 1},
    ]


def _draw_frame(
    slot_h: int,
    slot_w: int,
    slot_sigs: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]],
) -> np.ndarray:
    frame_h = int(2 * slot_h + 3)
    frame_w = int(2 * slot_w + 3)
    frame = np.full((frame_h, frame_w), BG, dtype=int)
    frame[0, :] = FRAME
    frame[-1, :] = FRAME
    frame[:, 0] = FRAME
    frame[:, -1] = FRAME
    frame[slot_h + 1, :] = FRAME
    frame[:, slot_w + 1] = FRAME

    slots = _local_slots(slot_h, slot_w)
    for slot, sig in zip(slots, slot_sigs):
        r0 = int(slot["r0"])
        c0 = int(slot["c0"])
        top, bottom, left, right = sig

        for dc, val in enumerate(top):
            if val == -1:
                frame[int(slot["top"]), c0 + dc] = BG
            elif val == 1:
                frame[r0, c0 + dc] = FRAME
        for dc, val in enumerate(bottom):
            if val == -1:
                frame[int(slot["bottom"]), c0 + dc] = BG
            elif val == 1:
                frame[r0 + slot_h - 1, c0 + dc] = FRAME
        for dr, val in enumerate(left):
            if val == -1:
                frame[r0 + dr, int(slot["left"])] = BG
            elif val == 1:
                frame[r0 + dr, c0] = FRAME
        for dr, val in enumerate(right):
            if val == -1:
                frame[r0 + dr, int(slot["right"])] = BG
            elif val == 1:
                frame[r0 + dr, c0 + slot_w - 1] = FRAME
    return frame


def _build_piece(
    slot_h: int,
    slot_w: int,
    sig: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]],
    color: int,
) -> dict:
    top, bottom, left, right = sig
    top_ext = int(any(v == 1 for v in top))
    bottom_ext = int(any(v == 1 for v in bottom))
    left_ext = int(any(v == 1 for v in left))
    right_ext = int(any(v == 1 for v in right))
    ph = int(slot_h + top_ext + bottom_ext)
    pw = int(slot_w + left_ext + right_ext)
    r0 = int(top_ext)
    c0 = int(left_ext)

    mask = np.zeros((ph, pw), dtype=bool)
    mask[r0 : r0 + slot_h, c0 : c0 + slot_w] = True

    for dc, val in enumerate(top):
        if val == -1:
            mask[r0, c0 + dc] = False
        elif val == 1:
            mask[0, c0 + dc] = True
    for dc, val in enumerate(bottom):
        if val == -1:
            mask[r0 + slot_h - 1, c0 + dc] = False
        elif val == 1:
            mask[ph - 1, c0 + dc] = True
    for dr, val in enumerate(left):
        if val == -1:
            mask[r0 + dr, c0] = False
        elif val == 1:
            mask[r0 + dr, 0] = True
    for dr, val in enumerate(right):
        if val == -1:
            mask[r0 + dr, c0 + slot_w - 1] = False
        elif val == 1:
            mask[r0 + dr, pw - 1] = True

    comps = _components(mask)
    assert len(comps) == 1
    patch = np.full((ph, pw), BG, dtype=int)
    patch[mask] = int(color)
    return {"patch": patch, "mask": mask, "top_ext": int(top_ext), "left_ext": int(left_ext), "color": int(color)}


def _place_piece(output: np.ndarray, slot: dict, piece: dict) -> None:
    top = int(slot["r0"] - piece["top_ext"])
    left = int(slot["c0"] - piece["left_ext"])
    rr, cc = np.nonzero(piece["mask"])
    output[top + rr, left + cc] = int(piece["color"])


def _build_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    slot_h = rng.randint(4, 6)
    slot_w = rng.randint(4, 6)
    slot_sigs = _generate_slot_signatures(slot_h, slot_w, rng)
    frame = _draw_frame(slot_h, slot_w, slot_sigs)

    colors = rng.sample([1, 2, 3, 4, 6, 7, 8, 9], 4)
    pieces = []
    for sig, color in zip(slot_sigs, colors):
        piece_sig = _negate_signature(sig)
        pieces.append(_build_piece(slot_h, slot_w, piece_sig, int(color)))

    expected = frame.copy()
    for slot, piece in zip(_local_slots(slot_h, slot_w), pieces):
        _place_piece(expected, slot, piece)

    piece_order = list(range(4))
    rng.shuffle(piece_order)
    max_ph = max(int(p["patch"].shape[0]) for p in pieces)
    max_pw = max(int(p["patch"].shape[1]) for p in pieces)
    bin_h = int(max_ph + 3)
    bin_w = int(max_pw + 3)

    frame_h, frame_w = frame.shape
    top_pad = rng.randint(1, 3)
    left_pad = rng.randint(1, 3)
    bottom_pad = rng.randint(1, 4)
    gap_left = rng.randint(1, 3)
    gap_right = rng.randint(1, 3)
    right_pad = rng.randint(1, 3)
    extra_right = rng.randint(2, 6)
    right_h = int(2 * bin_h + rng.randint(0, 2))
    right_w = int(2 * bin_w + extra_right)
    grid_h = max(int(frame_h + top_pad + bottom_pad), int(right_h + 2))
    grid_w = int(left_pad + frame_w + gap_left + 1 + gap_right + right_w + right_pad)

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    frame_top = rng.randint(1, max(1, grid_h - frame_h - 1))
    frame_left = int(left_pad)
    grid[frame_top : frame_top + frame_h, frame_left : frame_left + frame_w] = frame

    sep_col = int(frame_left + frame_w + gap_left)
    grid[:, sep_col] = FRAME

    right_left = int(sep_col + 1 + gap_right)
    right_top = rng.randint(1, max(1, grid_h - 2 * bin_h - 1))
    bin_origins = [
        (int(right_top), int(right_left)),
        (int(right_top), int(right_left + bin_w)),
        (int(right_top + bin_h), int(right_left)),
        (int(right_top + bin_h), int(right_left + bin_w)),
    ]

    for piece_idx, (bin_r, bin_c) in zip(piece_order, bin_origins):
        piece = pieces[piece_idx]
        ph, pw = piece["patch"].shape
        off_r = rng.randint(1, max(1, bin_h - ph - 1))
        off_c = rng.randint(1, max(1, bin_w - pw - 1))
        top = int(bin_r + off_r)
        left = int(bin_c + off_c)
        rr, cc = np.nonzero(piece["mask"])
        grid[top + rr, left + cc] = int(piece["color"])

    return grid, expected


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_RETRIES):
        try:
            grid, expected = _build_case(rng)
            if np.array_equal(main(grid), expected):
                return grid
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid puzzle")
