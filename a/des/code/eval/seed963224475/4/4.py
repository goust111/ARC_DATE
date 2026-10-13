import random
from typing import Optional

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
PIECE_COLORS = (1, 2, 3, 4, 6, 7, 8)

WALL_FLAT = 0
WALL_SLOT = 1
WALL_TOOTH = 2

SIDE_ORDER = ("top", "right", "bottom", "left")
SIDE_TO_ROT = {"top": 0, "left": 1, "bottom": 2, "right": 3}
SIDE_INDEX = {side: idx for idx, side in enumerate(SIDE_ORDER)}

GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _mask_cells(mask: np.ndarray, r0: int = 0, c0: int = 0) -> tuple[tuple[int, int], ...]:
    rows, cols = np.where(mask)
    return tuple(sorted((int(r0 + r), int(c0 + c)) for r, c in zip(rows, cols)))


def _render_piece(mask: np.ndarray, color: int) -> np.ndarray:
    arr = np.full(mask.shape, BLACK, dtype=int)
    arr[mask] = int(color)
    return arr


def _find_separator(grid: np.ndarray) -> int:
    grey_cols = [c for c in range(grid.shape[1]) if bool(np.all(grid[:, c] == GREY))]
    assert grey_cols, "expected a full-height grey separator"
    for c in grey_cols:
        left_ok = c == 0 or not bool(np.all(grid[:, c - 1] == GREY))
        right_ok = c == grid.shape[1] - 1 or not bool(np.all(grid[:, c + 1] == GREY))
        if left_ok and right_ok:
            return int(c)
    return int(grey_cols[0])


def _extract_frame_crop(left_region: np.ndarray) -> np.ndarray:
    pts = np.argwhere(left_region == GREY)
    assert pts.size > 0, "expected a grey frame in the left region"
    r0, c0, r1, c1 = _bbox(pts)
    frame = left_region[r0 : r1 + 1, c0 : c1 + 1].copy()
    assert frame.shape[0] >= 5 and frame.shape[1] >= 5
    assert int(frame[0, 0]) == GREY and int(frame[0, -1]) == GREY
    assert int(frame[-1, 0]) == GREY and int(frame[-1, -1]) == GREY
    return frame


def _extract_pieces(right_region: np.ndarray) -> list[dict]:
    mask = (right_region != BLACK) & (right_region != GREY)
    pieces: list[dict] = []
    for coords in _components(mask):
        colors = {int(right_region[r, c]) for r, c in coords}
        assert len(colors) == 1, "piece components must be monochrome"
        color = int(next(iter(colors)))
        r0, c0, r1, c1 = _bbox(coords)
        local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in coords:
            local[int(r - r0), int(c - c0)] = True
        pieces.append({"color": color, "mask": local})
    pieces.sort(key=lambda p: (-int(p["mask"].sum()), int(p["color"]), p["mask"].shape))
    return pieces


def _required_wall_signature(mask: np.ndarray, side: str) -> Optional[tuple[int, ...]]:
    h, w = mask.shape
    if side == "top":
        if h < 2:
            return None
        return tuple(
            WALL_SLOT if bool(mask[0, c]) else WALL_FLAT if bool(mask[1, c]) else WALL_TOOTH
            for c in range(w)
        )
    if side == "bottom":
        if h < 2:
            return None
        return tuple(
            WALL_SLOT if bool(mask[h - 1, c]) else WALL_FLAT if bool(mask[h - 2, c]) else WALL_TOOTH
            for c in range(w)
        )
    if side == "left":
        if w < 2:
            return None
        return tuple(
            WALL_SLOT if bool(mask[r, 0]) else WALL_FLAT if bool(mask[r, 1]) else WALL_TOOTH
            for r in range(h)
        )
    if side == "right":
        if w < 2:
            return None
        return tuple(
            WALL_SLOT if bool(mask[r, w - 1]) else WALL_FLAT if bool(mask[r, w - 2]) else WALL_TOOTH
            for r in range(h)
        )
    raise ValueError(side)


def _wall_signature(frame: np.ndarray, side: str, offset: int, span: int) -> tuple[int, ...]:
    h, w = frame.shape
    sig: list[int] = []
    if side == "top":
        assert 1 <= offset and offset + span - 1 <= w - 2
        for c in range(offset, offset + span):
            if int(frame[1, c]) == GREY:
                sig.append(WALL_TOOTH)
            elif int(frame[0, c]) == BLACK:
                sig.append(WALL_SLOT)
            else:
                sig.append(WALL_FLAT)
        return tuple(sig)
    if side == "bottom":
        assert 1 <= offset and offset + span - 1 <= w - 2
        for c in range(offset, offset + span):
            if int(frame[h - 2, c]) == GREY:
                sig.append(WALL_TOOTH)
            elif int(frame[h - 1, c]) == BLACK:
                sig.append(WALL_SLOT)
            else:
                sig.append(WALL_FLAT)
        return tuple(sig)
    if side == "left":
        assert 1 <= offset and offset + span - 1 <= h - 2
        for r in range(offset, offset + span):
            if int(frame[r, 1]) == GREY:
                sig.append(WALL_TOOTH)
            elif int(frame[r, 0]) == BLACK:
                sig.append(WALL_SLOT)
            else:
                sig.append(WALL_FLAT)
        return tuple(sig)
    if side == "right":
        assert 1 <= offset and offset + span - 1 <= h - 2
        for r in range(offset, offset + span):
            if int(frame[r, w - 2]) == GREY:
                sig.append(WALL_TOOTH)
            elif int(frame[r, w - 1]) == BLACK:
                sig.append(WALL_SLOT)
            else:
                sig.append(WALL_FLAT)
        return tuple(sig)
    raise ValueError(side)


def _placement_origin(frame_shape: tuple[int, int], mask_shape: tuple[int, int], side: str, offset: int) -> tuple[int, int]:
    fh, fw = frame_shape
    h, w = mask_shape
    if side == "top":
        return 0, int(offset)
    if side == "bottom":
        return int(fh - h), int(offset)
    if side == "left":
        return int(offset), 0
    if side == "right":
        return int(offset), int(fw - w)
    raise ValueError(side)


def _placement_clear(frame: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> bool:
    fh, fw = frame.shape
    h, w = mask.shape
    if r0 < 0 or c0 < 0 or r0 + h > fh or c0 + w > fw:
        return False
    rows, cols = np.where(mask)
    for dr, dc in zip(rows, cols):
        if int(frame[int(r0 + dr), int(c0 + dc)]) == GREY:
            return False
    return True


def _enumerate_piece_candidates(frame: np.ndarray, piece: dict) -> list[dict]:
    candidates: list[dict] = []
    seen: set[tuple] = set()
    for rot in range(4):
        mask = np.rot90(piece["mask"], rot)
        for side in SIDE_ORDER:
            req = _required_wall_signature(mask, side)
            if req is None:
                continue
            span = int(mask.shape[1] if side in ("top", "bottom") else mask.shape[0])
            limit = int(frame.shape[1] - span if side in ("top", "bottom") else frame.shape[0] - span)
            if limit <= 1:
                continue
            for offset in range(1, limit):
                if _wall_signature(frame, side, offset, span) != req:
                    continue
                r0, c0 = _placement_origin(frame.shape, mask.shape, side, offset)
                if not _placement_clear(frame, mask, r0, c0):
                    continue
                cells = _mask_cells(mask, r0, c0)
                claims = frozenset((side, int(offset + i)) for i in range(span))
                key = (side, int(r0), int(c0), cells)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    {
                        "rot": int(rot),
                        "side": side,
                        "offset": int(offset),
                        "r0": int(r0),
                        "c0": int(c0),
                        "cells": cells,
                        "claims": claims,
                        "special": int(sum(v != WALL_FLAT for v in req)),
                    }
                )
    candidates.sort(
        key=lambda cand: (
            SIDE_INDEX[cand["side"]],
            cand["offset"],
            cand["rot"],
            cand["r0"],
            cand["c0"],
        )
    )
    return candidates


def _select_candidate_set(candidate_lists: list[list[dict]], limit: int = 2) -> tuple[Optional[list[dict]], int]:
    if any(len(cands) == 0 for cands in candidate_lists):
        return None, 0

    order = sorted(range(len(candidate_lists)), key=lambda idx: (len(candidate_lists[idx]), idx))
    used_claims: set[tuple[str, int]] = set()
    chosen: list[Optional[dict]] = [None] * len(candidate_lists)
    solutions: list[list[dict]] = []

    def dfs(pos: int) -> None:
        if len(solutions) >= limit:
            return
        if pos == len(order):
            solutions.append([cand for cand in chosen if cand is not None])
            return
        piece_idx = order[pos]
        for cand in candidate_lists[piece_idx]:
            if any(claim in used_claims for claim in cand["claims"]):
                continue
            chosen[piece_idx] = cand
            added = list(cand["claims"])
            for claim in added:
                used_claims.add(claim)
            dfs(pos + 1)
            for claim in added:
                used_claims.remove(claim)
            chosen[piece_idx] = None
            if len(solutions) >= limit:
                return

    dfs(0)
    return (solutions[0] if solutions else None), len(solutions)


def _assemble_output(frame: np.ndarray, pieces: list[dict], chosen: list[dict]) -> np.ndarray:
    out = frame.copy()
    for piece, cand in zip(pieces, chosen):
        color = int(piece["color"])
        for r, c in cand["cells"]:
            cur = int(out[r, c])
            if cur == GREY:
                raise AssertionError("piece placement overlapped the frame")
            if cur == BLACK:
                out[r, c] = color
            elif cur == MAROON:
                continue
            else:
                out[r, c] = MAROON
    return out


def _solve_frame(frame: np.ndarray, pieces: list[dict]) -> tuple[np.ndarray, int]:
    candidate_lists = [_enumerate_piece_candidates(frame, piece) for piece in pieces]
    chosen, count = _select_candidate_set(candidate_lists, limit=2)
    assert chosen is not None, "no valid way to place all pieces"
    return _assemble_output(frame, pieces, chosen), count


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    sep = _find_separator(g)
    frame = _extract_frame_crop(g[:, :sep])
    pieces = _extract_pieces(g[:, sep + 1 :])
    out, _count = _solve_frame(frame, pieces)
    return out


def _canonical_piece_from_signature(wall_sig: tuple[int, ...], depth: int) -> np.ndarray:
    span = len(wall_sig)
    mask = np.zeros((depth + 1, span), dtype=bool)
    mask[1:, :] = True
    for idx, kind in enumerate(wall_sig):
        if kind == WALL_SLOT:
            mask[0, idx] = True
        elif kind == WALL_TOOTH:
            mask[1, idx] = False
    assert bool(mask[0].any()), "each piece needs at least one tab so its anchor margin survives cropping"
    return mask


def _rotate_to_side(mask: np.ndarray, side: str) -> np.ndarray:
    return np.rot90(mask, SIDE_TO_ROT[side])


def _apply_wall_signature(frame: np.ndarray, side: str, offset: int, wall_sig: tuple[int, ...]) -> None:
    h, w = frame.shape
    if side == "top":
        for i, kind in enumerate(wall_sig):
            c = offset + i
            if kind == WALL_SLOT:
                frame[0, c] = BLACK
            elif kind == WALL_TOOTH:
                frame[1, c] = GREY
        return
    if side == "bottom":
        for i, kind in enumerate(wall_sig):
            c = offset + i
            if kind == WALL_SLOT:
                frame[h - 1, c] = BLACK
            elif kind == WALL_TOOTH:
                frame[h - 2, c] = GREY
        return
    if side == "left":
        for i, kind in enumerate(wall_sig):
            r = offset + i
            if kind == WALL_SLOT:
                frame[r, 0] = BLACK
            elif kind == WALL_TOOTH:
                frame[r, 1] = GREY
        return
    if side == "right":
        for i, kind in enumerate(wall_sig):
            r = offset + i
            if kind == WALL_SLOT:
                frame[r, w - 1] = BLACK
            elif kind == WALL_TOOTH:
                frame[r, w - 2] = GREY
        return
    raise ValueError(side)


def _random_wall_signature(span: int, force_tooth: bool) -> tuple[int, ...]:
    assert span >= 3
    while True:
        sig = [WALL_FLAT] * span
        n_special = random.randint(2, min(4, span))
        idxs = random.sample(range(span), n_special)

        slot_idx = random.choice(idxs)
        sig[slot_idx] = WALL_SLOT

        remaining = [idx for idx in idxs if idx != slot_idx]
        if force_tooth:
            tooth_idx = random.choice(remaining)
            sig[tooth_idx] = WALL_TOOTH
            remaining = [idx for idx in remaining if idx != tooth_idx]
        elif remaining and random.random() < 0.6:
            tooth_idx = random.choice(remaining)
            sig[tooth_idx] = WALL_TOOTH
            remaining = [idx for idx in remaining if idx != tooth_idx]

        for idx in remaining:
            sig[idx] = random.choice([WALL_SLOT, WALL_TOOTH])

        if all(kind == WALL_SLOT for kind in sig):
            sig[random.randrange(span)] = WALL_TOOTH
        if WALL_SLOT not in sig:
            continue
        if force_tooth and WALL_TOOTH not in sig:
            continue
        return tuple(int(v) for v in sig)


def _rect_for_candidate(cand: dict) -> tuple[int, int, int, int]:
    rows = [r for r, _c in cand["cells"]]
    cols = [c for _r, c in cand["cells"]]
    return min(rows), min(cols), max(rows), max(cols)


def _overlap_count(chosen: list[dict], frame_shape: tuple[int, int]) -> int:
    counts = np.zeros(frame_shape, dtype=int)
    for cand in chosen:
        for r, c in cand["cells"]:
            counts[r, c] += 1
    return int((counts > 1).sum())


def _place_patch(
    grid: np.ndarray,
    reserve: np.ndarray,
    patch: np.ndarray,
    col_start: int,
) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(0, h - ph + 1):
        for c0 in range(col_start, w - pw + 1):
            r_lo = max(0, r0 - 1)
            r_hi = min(h, r0 + ph + 1)
            c_lo = max(0, c0 - 1)
            c_hi = min(w, c0 + pw + 1)
            if bool(reserve[r_lo:r_hi, c_lo:c_hi].any()):
                continue
            positions.append((r0, c0))
    if not positions:
        return False
    r0, c0 = random.choice(positions)
    patch_mask = patch != BLACK
    sub = grid[r0 : r0 + ph, c0 : c0 + pw]
    sub[patch_mask] = patch[patch_mask]
    reserve[r0 : r0 + ph, c0 : c0 + pw] = True
    return True


def _gen_attempt() -> Optional[np.ndarray]:
    n_pieces = random.randint(2, 4)
    want_overlap = bool(random.getrandbits(1))

    fh = random.randint(9, 14)
    fw = random.randint(9, 14)
    frame = np.full((fh, fw), BLACK, dtype=int)
    frame[0, :] = GREY
    frame[-1, :] = GREY
    frame[:, 0] = GREY
    frame[:, -1] = GREY

    sides = random.sample(list(SIDE_ORDER), n_pieces)
    random.shuffle(sides)
    colors = random.sample(list(PIECE_COLORS), n_pieces)

    pieces: list[dict] = []
    has_tooth = False
    for idx, (side, color) in enumerate(zip(sides, colors)):
        side_len = fw - 2 if side in ("top", "bottom") else fh - 2
        cross_len = fh - 2 if side in ("top", "bottom") else fw - 2
        span_lo = max(3, side_len // 2) if want_overlap else 3
        span = random.randint(span_lo, side_len)
        if want_overlap:
            depth_lo = max(2, cross_len // 2)
            depth = random.randint(depth_lo, cross_len)
        else:
            depth_hi = max(2, (cross_len + 1) // 2)
            depth = random.randint(2, depth_hi)
        offset = random.randint(1, side_len - span + 1)
        force_tooth = not has_tooth and idx == n_pieces - 1
        wall_sig = _random_wall_signature(span, force_tooth=force_tooth)
        has_tooth = has_tooth or WALL_TOOTH in wall_sig
        _apply_wall_signature(frame, side, offset, wall_sig)
        canonical = _canonical_piece_from_signature(wall_sig, depth)
        pieces.append({"color": int(color), "mask": _rotate_to_side(canonical, side)})

    try:
        expected, solution_count = _solve_frame(frame, pieces)
    except AssertionError:
        return None
    if solution_count != 1:
        return None

    chosen, chosen_count = _select_candidate_set([_enumerate_piece_candidates(frame, piece) for piece in pieces], limit=2)
    if chosen is None or chosen_count != 1:
        return None
    overlap_count = _overlap_count(chosen, frame.shape)
    if want_overlap != (overlap_count > 0):
        return None

    piece_patches = []
    for piece in pieces:
        arr = _render_piece(piece["mask"], int(piece["color"]))
        piece_patches.append(np.rot90(arr, random.randrange(4)))

    max_piece_h = max(int(arr.shape[0]) for arr in piece_patches)
    max_piece_w = max(int(arr.shape[1]) for arr in piece_patches)
    right_w = max(sum(int(arr.shape[1]) for arr in piece_patches) + 3 * len(piece_patches), max_piece_w * 2 + 6)

    grid_h = max(fh + random.randint(2, 6), max_piece_h * 2 + 5, 14)
    frame_r0 = random.randint(1, grid_h - fh - 1)
    frame_c0 = random.randint(1, 3)
    sep_col = frame_c0 + fw + random.randint(1, 3)
    grid_w = sep_col + 1 + right_w + random.randint(2, 5)

    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    grid[frame_r0 : frame_r0 + fh, frame_c0 : frame_c0 + fw] = frame
    grid[:, sep_col] = GREY

    reserve = np.zeros_like(grid, dtype=bool)
    reserve[:, : sep_col + 2] = True
    for arr in sorted(piece_patches, key=lambda x: int(x.size), reverse=True):
        if not _place_patch(grid, reserve, arr, sep_col + 2):
            return None

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None

    sep = _find_separator(grid)
    frame2 = _extract_frame_crop(grid[:, :sep])
    pieces2 = _extract_pieces(grid[:, sep + 1 :])
    try:
        _out2, count2 = _solve_frame(frame2, pieces2)
    except AssertionError:
        return None
    if count2 != 1:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
