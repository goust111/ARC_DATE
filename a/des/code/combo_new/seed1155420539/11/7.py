from __future__ import annotations

import random

import numpy as np


BG = 0
BOARD_COLOR = 5
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = ("top", "bottom", "left", "right")
PIECE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

GRID_MIN = 22
GRID_MAX = 30
BASE_H_RANGE = (5, 7)
BASE_W_RANGE = (5, 7)
PIECE_COUNT_CHOICES = (3, 4)
SPAN_MIN = 3
SPAN_MAX = 6
THICKNESS_MIN = 2
THICKNESS_MAX = 3
EXTRA_MAX = 2
MIN_PIECE_AREA = 5
MAX_GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _component_dicts(grid: np.ndarray, bg: int = BG) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "area": len(coords),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return comps


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(mask)
    assert pts.size > 0
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask_to_patch(mask: np.ndarray, color: int) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_from_mask(mask)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]
    patch[sub] = int(color)
    return patch


def _crop_component_patch(grid: np.ndarray, comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    color = int(comp["color"])
    for r, c in comp["coords"]:
        patch[int(r) - r0, int(c) - c0] = color
    return patch


def _rotation_variants(patch: np.ndarray) -> list[np.ndarray]:
    seen: set[tuple[tuple[int, int], bytes]] = set()
    out: list[np.ndarray] = []
    for k in range(4):
        rot = np.rot90(patch, k)
        mask = rot != BG
        key = (tuple(map(int, rot.shape)), mask.tobytes())
        if key in seen:
            continue
        seen.add(key)
        out.append(rot.copy())
    return out


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    sub = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != BG
    sub[mask] = patch[mask]


def _frontier_mask(mask: np.ndarray, side: str) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    f = np.zeros_like(m, dtype=bool)
    if side == "top":
        f[:-1, :] = m[:-1, :] & ~m[1:, :]
        f[-1, :] = m[-1, :]
    elif side == "bottom":
        f[1:, :] = m[1:, :] & ~m[:-1, :]
        f[0, :] = m[0, :]
    elif side == "left":
        f[:, :-1] = m[:, :-1] & ~m[:, 1:]
        f[:, -1] = m[:, -1]
    else:
        f[:, 1:] = m[:, 1:] & ~m[:, :-1]
        f[:, 0] = m[:, 0]
    return f


def _envelopes(mask: np.ndarray) -> dict[str, np.ndarray]:
    occ = np.asarray(mask, dtype=bool)
    h, w = occ.shape
    top = np.full(w, h, dtype=int)
    bottom = np.full(w, -1, dtype=int)
    left = np.full(h, w, dtype=int)
    right = np.full(h, -1, dtype=int)
    for c in range(w):
        rows = np.where(occ[:, c])[0]
        if rows.size:
            top[c] = int(rows[0])
            bottom[c] = int(rows[-1])
    for r in range(h):
        cols = np.where(occ[r])[0]
        if cols.size:
            left[r] = int(cols[0])
            right[r] = int(cols[-1])
    return {"top": top, "bottom": bottom, "left": left, "right": right}


def _choose_board_component(comps: list[dict], h: int, w: int) -> int:
    cr = (h - 1) / 2.0
    cc = (w - 1) / 2.0
    ranked = []
    for i, comp in enumerate(comps):
        r0, c0, r1, c1 = comp["bbox"]
        contains_center = r0 <= cr <= r1 and c0 <= cc <= c1
        mean_r = sum(r for r, _ in comp["coords"]) / float(comp["area"])
        mean_c = sum(c for _, c in comp["coords"]) / float(comp["area"])
        dist = (mean_r - cr) ** 2 + (mean_c - cc) ** 2
        ranked.append((0 if contains_center else 1, dist, -comp["area"], i))
    ranked.sort()
    return int(ranked[0][-1])


def _placement_fits(
    occ: np.ndarray,
    envelopes: dict[str, np.ndarray],
    bbox: tuple[int, int, int, int],
    mask: np.ndarray,
    frontier: np.ndarray,
    side: str,
    r0: int,
    c0: int,
) -> bool:
    fr = np.argwhere(frontier)
    body = np.argwhere(mask & ~frontier)
    if fr.size == 0 or body.size == 0:
        return False
    h, w = occ.shape
    ph, pw = mask.shape
    if r0 < 0 or c0 < 0 or r0 + ph > h or c0 + pw > w:
        return False
    if side == "top":
        for rr, cc in fr:
            gr, gc = int(rr + r0), int(cc + c0)
            if not bool(occ[gr, gc]) or int(envelopes["top"][gc]) != gr:
                return False
        for rr, cc in body:
            gr, gc = int(rr + r0), int(cc + c0)
            if bool(occ[gr, gc]):
                return False
        return int(body[:, 0].min()) + r0 < int(bbox[0])
    if side == "bottom":
        for rr, cc in fr:
            gr, gc = int(rr + r0), int(cc + c0)
            if not bool(occ[gr, gc]) or int(envelopes["bottom"][gc]) != gr:
                return False
        for rr, cc in body:
            gr, gc = int(rr + r0), int(cc + c0)
            if bool(occ[gr, gc]):
                return False
        return int(body[:, 0].max()) + r0 > int(bbox[2])
    if side == "left":
        for rr, cc in fr:
            gr, gc = int(rr + r0), int(cc + c0)
            if not bool(occ[gr, gc]) or int(envelopes["left"][gr]) != gc:
                return False
        for rr, cc in body:
            gr, gc = int(rr + r0), int(cc + c0)
            if bool(occ[gr, gc]):
                return False
        return int(body[:, 1].min()) + c0 < int(bbox[1])
    for rr, cc in fr:
        gr, gc = int(rr + r0), int(cc + c0)
        if not bool(occ[gr, gc]) or int(envelopes["right"][gr]) != gc:
            return False
    for rr, cc in body:
        gr, gc = int(rr + r0), int(cc + c0)
        if bool(occ[gr, gc]):
            return False
    return int(body[:, 1].max()) + c0 > int(bbox[3])


def _candidate_placements(current: np.ndarray, patch: np.ndarray) -> list[tuple[np.ndarray, int, int]]:
    occ = np.asarray(current != BG, dtype=bool)
    bbox = _bbox_from_mask(occ)
    env = _envelopes(occ)
    candidates: list[tuple[int, int, int, int, np.ndarray]] = []
    seen: set[tuple[int, int, tuple[int, int], bytes]] = set()
    for rot_idx, rot in enumerate(_rotation_variants(patch)):
        mask = rot != BG
        ph, pw = rot.shape
        for side_idx, side in enumerate(SIDES):
            frontier = _frontier_mask(mask, side)
            overlap_count = int(frontier.sum())
            if overlap_count < 2 or overlap_count == int(mask.sum()):
                continue
            for r0 in range(current.shape[0] - ph + 1):
                for c0 in range(current.shape[1] - pw + 1):
                    if _placement_fits(occ, env, bbox, mask, frontier, side, r0, c0):
                        key = (r0, c0, tuple(map(int, rot.shape)), (rot != BG).tobytes())
                        if key in seen:
                            continue
                        seen.add(key)
                        candidates.append((-overlap_count, side_idx, r0, c0, rot.copy()))
    candidates.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4].shape, x[4].tobytes()))
    return [(cand[4], int(cand[2]), int(cand[3])) for cand in candidates]


def _find_piece_placement(current: np.ndarray, patch: np.ndarray) -> tuple[np.ndarray, int, int] | None:
    candidates = _candidate_placements(current, patch)
    return candidates[0] if candidates else None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered colored fragments onto the central grey board by matching interlocking edges.

    Input structure:
    - Background is black.
    - A single central monochrome component is the grey board; its rectangular core may already have one-cell tabs and notches on the border.
    - Other foreground components are monochrome colored fragments scattered away from the board.
    - Each fragment is a 4-connected shape whose inward-facing edge matches one side profile of the current board or of the already assembled structure.

    Rule:
    - Identify the central board and treat all other components as movable pieces.
    - Sort the pieces by area from large to small.
    - For each piece, try the four rotations and place it on one of the four outer sides so that the piece's inward frontier exactly overlaps the current side-envelope cells, while all newly added cells land on black background.
    - Paint the piece at that matched location, updating the assembled outline before processing the next piece.

    Key insight: The solve requires three linked steps: recognize the central board, process pieces in descending area order, and use rotation-aware boundary-profile matching so each piece locks to the current outer contour.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    comps = _component_dicts(g, bg=BG)
    assert comps, "grid must contain a board and pieces"

    board_idx = _choose_board_component(comps, h, w)
    out = np.full((h, w), BG, dtype=int)
    board = comps[board_idx]
    for r, c in board["coords"]:
        out[int(r), int(c)] = int(board["color"])

    pieces: list[dict] = []
    for i, comp in enumerate(comps):
        if i == board_idx:
            continue
        pieces.append(
            {
                "area": int(comp["area"]),
                "patch": _crop_component_patch(g, comp),
                "color": int(comp["color"]),
            }
        )
    pieces.sort(key=lambda d: (-d["area"], d["color"]))

    def search(k: int, canvas: np.ndarray) -> np.ndarray | None:
        if k == len(pieces):
            return canvas
        piece = pieces[k]
        for patch, r0, c0 in _candidate_placements(canvas, piece["patch"]):
            nxt = canvas.copy()
            _paste(nxt, patch, r0, c0)
            solved = search(k + 1, nxt)
            if solved is not None:
                return solved
        return None

    solved = search(0, out)
    assert solved is not None, "every piece must match the current board boundary"
    return solved


def _random_walk_profile(length: int, rng: random.Random) -> list[int]:
    vals = [rng.randint(0, min(1, EXTRA_MAX))]
    for _ in range(length - 1):
        prev = vals[-1]
        opts = [v for v in (prev - 1, prev, prev + 1) if 0 <= v <= EXTRA_MAX]
        vals.append(rng.choice(opts))
    if max(vals) == 0 and EXTRA_MAX >= 1:
        idx = rng.randrange(length)
        vals[idx] = 1
        for i in range(idx - 1, -1, -1):
            vals[i] = max(0, min(vals[i], vals[i + 1] + 1))
        for i in range(idx + 1, length):
            vals[i] = max(0, min(vals[i], vals[i - 1] + 1))
    return vals


def _make_initial_board(h: int, w: int, rng: random.Random) -> np.ndarray | None:
    mask = np.zeros((h, w), dtype=bool)
    bh = rng.randint(BASE_H_RANGE[0], BASE_H_RANGE[1])
    bw = rng.randint(BASE_W_RANGE[0], BASE_W_RANGE[1])
    r0 = h // 2 - bh // 2 + rng.randint(-1, 1)
    c0 = w // 2 - bw // 2 + rng.randint(-1, 1)
    r0 = max(3, min(h - bh - 4, r0))
    c0 = max(3, min(w - bw - 4, c0))
    mask[r0 : r0 + bh, c0 : c0 + bw] = True

    changed = False
    for _ in range(rng.randint(2, 4)):
        side = rng.choice(SIDES)
        if side in ("top", "bottom"):
            if bw < 3:
                continue
            c = rng.randint(c0 + 1, c0 + bw - 2)
            if rng.random() < 0.5:
                rr = r0 - 1 if side == "top" else r0 + bh
                if 0 <= rr < h and not bool(mask[rr, c]):
                    mask[rr, c] = True
                    changed = True
            else:
                rr = r0 if side == "top" else r0 + bh - 1
                inner = rr + 1 if side == "top" else rr - 1
                if 0 <= inner < h and bool(mask[rr, c]) and bool(mask[inner, c]):
                    mask[rr, c] = False
                    changed = True
        else:
            if bh < 3:
                continue
            r = rng.randint(r0 + 1, r0 + bh - 2)
            if rng.random() < 0.5:
                cc = c0 - 1 if side == "left" else c0 + bw
                if 0 <= cc < w and not bool(mask[r, cc]):
                    mask[r, cc] = True
                    changed = True
            else:
                cc = c0 if side == "left" else c0 + bw - 1
                inner = cc + 1 if side == "left" else cc - 1
                if 0 <= inner < w and bool(mask[r, cc]) and bool(mask[r, inner]):
                    mask[r, cc] = False
                    changed = True

    if not changed or not _is_connected(mask):
        return None
    return mask


def _side_spans(mask: np.ndarray, side: str) -> list[tuple[list[int], list[int]]]:
    bbox = _bbox_from_mask(mask)
    env = _envelopes(mask)
    spans: list[tuple[list[int], list[int]]] = []
    if side in ("top", "bottom"):
        c0, c1 = int(bbox[1]), int(bbox[3])
        positions = list(range(c0, c1 + 1))
        profile = [int(env[side][c]) for c in positions]
    else:
        r0, r1 = int(bbox[0]), int(bbox[2])
        positions = list(range(r0, r1 + 1))
        profile = [int(env[side][r]) for r in positions]

    if len(positions) < SPAN_MIN:
        return spans

    for length in range(SPAN_MIN, min(SPAN_MAX, len(positions)) + 1):
        starts = range(0, len(positions) - length + 1)
        if len(positions) - length >= 2:
            starts = range(1, len(positions) - length)
        for start in starts:
            sub_positions = positions[start : start + length]
            sub_profile = profile[start : start + length]
            if any(abs(a - b) > 1 for a, b in zip(sub_profile, sub_profile[1:])):
                continue
            spans.append((sub_positions, sub_profile))
    return spans


def _build_piece_mask(
    occ: np.ndarray,
    side: str,
    positions: list[int],
    inner_profile: list[int],
    thickness: int,
    rng: random.Random,
) -> np.ndarray | None:
    h, w = occ.shape
    mask = np.zeros((h, w), dtype=bool)
    extra = _random_walk_profile(len(positions), rng)
    if side == "top":
        for c, inner, add in zip(positions, inner_profile, extra):
            outer = inner - thickness - add
            if outer < 0:
                return None
            mask[outer : inner + 1, c] = True
    elif side == "bottom":
        for c, inner, add in zip(positions, inner_profile, extra):
            outer = inner + thickness + add
            if outer >= h:
                return None
            mask[inner : outer + 1, c] = True
    elif side == "left":
        for r, inner, add in zip(positions, inner_profile, extra):
            outer = inner - thickness - add
            if outer < 0:
                return None
            mask[r, outer : inner + 1] = True
    else:
        for r, inner, add in zip(positions, inner_profile, extra):
            outer = inner + thickness + add
            if outer >= w:
                return None
            mask[r, inner : outer + 1] = True

    overlap = mask & occ
    if int(overlap.sum()) != len(positions):
        return None
    if not _is_connected(mask):
        return None
    r0, c0, r1, c1 = _bbox_from_mask(mask)
    if int(mask.sum()) == (r1 - r0 + 1) * (c1 - c0 + 1):
        return None
    return mask


def _sample_piece(
    occ: np.ndarray,
    color: int,
    prev_area: int,
    remaining_after: int,
    rng: random.Random,
) -> dict | None:
    min_area_for_slot = MIN_PIECE_AREA + remaining_after
    candidates: list[dict] = []
    side_order = list(SIDES)
    rng.shuffle(side_order)
    thicknesses = list(range(THICKNESS_MIN, THICKNESS_MAX + 1))
    for side in side_order:
        spans = _side_spans(occ, side)
        rng.shuffle(spans)
        for positions, inner_profile in spans:
            rng.shuffle(thicknesses)
            for thickness in thicknesses:
                piece_mask = _build_piece_mask(occ, side, positions, inner_profile, thickness, rng)
                if piece_mask is None:
                    continue
                area = int(piece_mask.sum())
                if area < min_area_for_slot or area >= prev_area:
                    continue
                candidates.append(
                    {
                        "mask": piece_mask,
                        "patch": _crop_mask_to_patch(piece_mask, color),
                        "area": area,
                        "color": color,
                    }
                )
    if not candidates:
        return None
    candidates.sort(key=lambda d: (-d["area"], d["patch"].shape, d["patch"].tobytes()))
    return candidates[rng.randrange(min(6, len(candidates)))]


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = np.asarray(mask, dtype=bool).copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _scatter_pieces(inp: np.ndarray, board_mask: np.ndarray, pieces: list[dict], rng: random.Random) -> bool:
    h, w = inp.shape
    used = board_mask.copy()
    order = list(range(len(pieces)))
    order.sort(key=lambda i: -pieces[i]["area"])
    for idx in order:
        base_patch = pieces[idx]["patch"]
        rotations = _rotation_variants(base_patch)
        rng.shuffle(rotations)
        placed = False
        for rot in rotations:
            ph, pw = rot.shape
            if ph > h or pw > w:
                continue
            padded = _dilate8(used)
            for _ in range(200):
                r0 = rng.randint(0, h - ph)
                c0 = rng.randint(0, w - pw)
                mask = rot != BG
                if bool((padded[r0 : r0 + ph, c0 : c0 + pw] & mask).any()):
                    continue
                _paste(inp, rot, r0, c0)
                used[r0 : r0 + ph, c0 : c0 + pw] |= mask
                placed = True
                break
            if placed:
                break
        if not placed:
            return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    board_mask = _make_initial_board(h, w, rng)
    if board_mask is None:
        return None

    occ = board_mask.copy()

    piece_count = rng.choice(PIECE_COUNT_CHOICES)
    colors = rng.sample(PIECE_COLORS, piece_count)
    pieces: list[dict] = []
    prev_area = h * w + 1
    for idx, color in enumerate(colors):
        piece = None
        for _ in range(120):
            piece = _sample_piece(occ, color, prev_area, piece_count - idx - 1, rng)
            if piece is not None:
                break
        if piece is None:
            return None
        pieces.append(piece)
        prev_area = int(piece["area"])
        occ |= piece["mask"]

    inp = np.full((h, w), BG, dtype=int)
    inp[board_mask] = BOARD_COLOR
    if not _scatter_pieces(inp, board_mask, pieces, rng):
        return None

    comps = _component_dicts(inp, bg=BG)
    if len(comps) != piece_count + 1:
        return None
    if int(np.sum(inp == BG)) <= int(np.sum(inp != BG)):
        return None
    try:
        solved = main(inp)
    except Exception:
        return None
    if np.array_equal(solved, inp):
        return None
    if not _is_connected(solved != BG):
        return None
    if int(np.sum(solved != BG)) <= int(board_mask.sum()):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample for puzzle 7")
