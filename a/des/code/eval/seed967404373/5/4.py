from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
BOARD_COLOR = 5
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
PIECE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

GEN_RETRIES = 400
SHAPE_RETRIES = 200
BOARD_RETRIES = 80
PIECE_RETRIES = 120


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIRS:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _trim(arr: np.ndarray) -> np.ndarray:
    mask = arr != BG
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    assert rows.size and cols.size
    return arr[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in np.asarray(mask, dtype=int))


def _canonical_signature(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return min(_mask_signature(np.rot90(mask, k)) for k in range(4))


def _distinct_rotation_ks(mask: np.ndarray) -> list[int]:
    seen: set[tuple[tuple[int, ...], ...]] = set()
    ks: list[int] = []
    for k in range(4):
        sig = _mask_signature(np.rot90(mask, k))
        if sig not in seen:
            seen.add(sig)
            ks.append(k)
    return ks


def _flat_edge_candidates(mask: np.ndarray) -> list[tuple[int, int, int]]:
    h, w = mask.shape
    out: list[tuple[int, int, int]] = []
    for r, c in np.argwhere(mask):
        exposed: list[int] = []
        for d, (dr, dc) in enumerate(DIRS):
            nr = int(r + dr)
            nc = int(c + dc)
            if nr < 0 or nc < 0 or nr >= h or nc >= w or not bool(mask[nr, nc]):
                exposed.append(int(d))
        if len(exposed) == 1:
            out.append((int(r), int(c), int(exposed[0])))
    return out


def _single_touch_seam_candidates(mask: np.ndarray) -> list[tuple[int, int, int]]:
    h, w = mask.shape
    out: list[tuple[int, int, int]] = []
    for r, c, d in _flat_edge_candidates(mask):
        dr, dc = DIRS[int(d)]
        ar = int(r + dr)
        ac = int(c + dc)
        touches = 0
        for dr2, dc2 in DIRS:
            nr = ar + dr2
            nc = ac + dc2
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]):
                touches += 1
        if touches == 1:
            out.append((int(r), int(c), int(d)))
    return out


def _mode_color(values: list[int]) -> int:
    counts = Counter(values)
    top = max(counts.values())
    winners = [color for color, cnt in counts.items() if cnt == top]
    assert len(winners) == 1
    return int(winners[0])


def _parse_piece(grid: np.ndarray, cells: list[tuple[int, int]]) -> dict:
    r0, c0, r1, c1 = _bbox(cells)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    colors = [int(v) for v in patch.ravel() if int(v) != BG]
    main_color = _mode_color(colors)
    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "patch": patch,
        "main_color": int(main_color),
        "cells": [(int(r), int(c)) for r, c in cells],
    }


def _parse_slots(board_patch: np.ndarray, board_bbox: tuple[int, int, int, int], board_color: int) -> list[dict]:
    br0, bc0, _br1, _bc1 = board_bbox
    h, w = board_patch.shape
    slots: list[dict] = []
    for cells in _components(board_patch == BG):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in cells):
            continue
        r0, c0, r1, c1 = _bbox(cells)
        shape = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in cells:
            shape[r - r0, c - c0] = True

        seams: list[tuple[int, int, int, int]] = []
        seams_by_cell: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for r, c in cells:
            rr = int(r - r0)
            cc = int(c - c0)
            for d, (dr, dc) in enumerate(DIRS):
                nr = int(r + dr)
                nc = int(c + dc)
                if 0 <= nr < h and 0 <= nc < w:
                    color = int(board_patch[nr, nc])
                    if color not in (BG, board_color):
                        seams.append((rr, cc, int(d), color))
                        seams_by_cell.setdefault((rr, cc), []).append((int(d), color))

        slots.append(
            {
                "bbox_abs": (int(br0 + r0), int(bc0 + c0), int(br0 + r1), int(bc0 + c1)),
                "shape": shape,
                "seams": seams,
                "seams_by_cell": seams_by_cell,
            }
        )

    slots.sort(key=lambda slot: slot["bbox_abs"])
    return slots


def _parse_instance(grid: np.ndarray) -> tuple[dict, list[dict]]:
    fg_components = _components(grid != BG)
    assert len(fg_components) >= 2, "expected one board and at least one piece"
    board_cells = max(fg_components, key=len)
    board_set = set(board_cells)
    board_bbox = _bbox(board_cells)
    br0, bc0, br1, bc1 = board_bbox
    board_patch = grid[br0 : br1 + 1, bc0 : bc1 + 1].copy()
    board_colors = [int(grid[r, c]) for r, c in board_cells]
    board_color = _mode_color(board_colors)
    slots = _parse_slots(board_patch, board_bbox, board_color)

    pieces = []
    for cells in fg_components:
        if set(cells) == board_set:
            continue
        pieces.append(_parse_piece(grid, cells))
    pieces.sort(key=lambda piece: piece["bbox"])

    board = {
        "bbox": board_bbox,
        "patch": board_patch,
        "color": int(board_color),
        "cells": board_cells,
    }
    return board, slots, pieces


def _secondary_edge_map(patch: np.ndarray, main_color: int) -> dict[tuple[int, int], tuple[int, set[int]]] | None:
    h, w = patch.shape
    out: dict[tuple[int, int], tuple[int, set[int]]] = {}
    for r, c in np.argwhere((patch != BG) & (patch != int(main_color))):
        dirs: set[int] = set()
        for d, (dr, dc) in enumerate(DIRS):
            nr = int(r + dr)
            nc = int(c + dc)
            if nr < 0 or nc < 0 or nr >= h or nc >= w or int(patch[nr, nc]) == BG:
                dirs.add(int(d))
        if not dirs:
            return None
        out[(int(r), int(c))] = (int(patch[r, c]), dirs)
    return out


def _match_variant(piece: dict, slot: dict, rot_k: int, require_seams: bool) -> np.ndarray | None:
    rotated = np.rot90(piece["patch"], int(rot_k))
    shape = rotated != BG
    if shape.shape != slot["shape"].shape or not np.array_equal(shape, slot["shape"]):
        return None
    if not require_seams:
        return rotated

    secondary = _secondary_edge_map(rotated, int(piece["main_color"]))
    if slot["seams"]:
        if not secondary:
            return None
        for rr, cc, d, color in slot["seams"]:
            item = secondary.get((int(rr), int(cc)))
            if item is None:
                return None
            piece_color, dirs = item
            if piece_color != int(color) or int(d) not in dirs:
                return None
        for (rr, cc), (piece_color, dirs) in secondary.items():
            matches = slot["seams_by_cell"].get((int(rr), int(cc)), [])
            if not any(int(d) in dirs and int(color) == piece_color for d, color in matches):
                return None
    elif secondary:
        return None
    return rotated


def _candidate_lists(pieces: list[dict], slots: list[dict], require_seams: bool) -> list[list[dict]]:
    out: list[list[dict]] = []
    for piece_idx, piece in enumerate(pieces):
        piece_cands: list[dict] = []
        for slot_idx, slot in enumerate(slots):
            for rot_k in range(4):
                rotated = _match_variant(piece, slot, rot_k, require_seams=require_seams)
                if rotated is None:
                    continue
                piece_cands.append(
                    {
                        "piece_idx": int(piece_idx),
                        "slot_idx": int(slot_idx),
                        "rot_k": int(rot_k),
                        "patch": rotated,
                    }
                )
        out.append(piece_cands)
    return out


def _solve_assignments(candidate_lists: list[list[dict]], n_slots: int, limit: int = 2) -> list[list[dict]]:
    order = sorted(range(len(candidate_lists)), key=lambda idx: len(candidate_lists[idx]))
    used = [False] * int(n_slots)
    picked: list[dict | None] = [None] * len(candidate_lists)
    solutions: list[list[dict]] = []

    def dfs(pos: int) -> None:
        if len(solutions) >= limit:
            return
        if pos == len(order):
            solutions.append([item for item in picked if item is not None])
            return
        piece_idx = order[pos]
        for cand in candidate_lists[piece_idx]:
            slot_idx = int(cand["slot_idx"])
            if used[slot_idx]:
                continue
            used[slot_idx] = True
            picked[piece_idx] = cand
            dfs(pos + 1)
            picked[piece_idx] = None
            used[slot_idx] = False

    dfs(0)
    return solutions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each board slot with the matching loose piece.

    Shape matching uses the occupied cells of each piece, while ignoring whether
    those occupied cells use the piece's primary or secondary color. If multiple
    slot/rotation pairs share the same occupancy shape, boundary seam colors fix
    the unique target slot and orientation.
    """
    g = np.asarray(grid, dtype=int)
    board, slots, pieces = _parse_instance(g)
    assert len(slots) == len(pieces) and slots, "piece count must match slot count"

    candidates = _candidate_lists(pieces, slots, require_seams=True)
    assert all(cands for cands in candidates), "every piece must match at least one slot"
    solutions = _solve_assignments(candidates, len(slots), limit=2)
    assert len(solutions) == 1, "expected a unique piece-to-slot assignment"
    assignment = solutions[0]

    out = g.copy()
    for piece in pieces:
        for r, c in piece["cells"]:
            out[int(r), int(c)] = BG

    for cand in assignment:
        slot = slots[int(cand["slot_idx"])]
        patch = np.asarray(cand["patch"], dtype=int)
        r0, c0, r1, c1 = slot["bbox_abs"]
        region = out[r0 : r1 + 1, c0 : c1 + 1]
        region[slot["shape"]] = patch[slot["shape"]]

    return out


def _random_shape() -> np.ndarray | None:
    for _ in range(SHAPE_RETRIES):
        h = random.randint(3, 4)
        w = random.randint(3, 4)
        target = random.randint(5, min(7, h * w - 1))
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for dr, dc in DIRS:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != target:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[int(r), int(c)] = True
        mask = _trim(mask)
        hh, ww = mask.shape
        if hh < 2 or ww < 2:
            continue
        if int(mask.sum()) == hh * ww:
            continue
        if len(_single_touch_seam_candidates(mask)) < 2:
            continue
        return mask
    return None


def _sample_shape_specs(ambiguous_mode: bool, n_pieces: int) -> list[dict] | None:
    specs: list[dict] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()

    if ambiguous_mode:
        for _ in range(SHAPE_RETRIES):
            base = _random_shape()
            if base is None:
                continue
            ks = _distinct_rotation_ks(base)
            if len(ks) < 2:
                continue
            slot_ks = random.sample(ks, 2)
            for slot_k in slot_ks:
                specs.append({"base_mask": base, "slot_mask": np.rot90(base, int(slot_k))})
            seen.add(_canonical_signature(base))
            break
        else:
            return None

    while len(specs) < n_pieces:
        shape = _random_shape()
        if shape is None:
            return None
        sig = _canonical_signature(shape)
        if sig in seen:
            continue
        seen.add(sig)
        slot_k = random.randrange(4)
        specs.append({"base_mask": shape, "slot_mask": np.rot90(shape, int(slot_k))})

    random.shuffle(specs)
    return specs


def _can_place_mask(used: np.ndarray, mask: np.ndarray, top: int, left: int, pad: int) -> bool:
    h, w = used.shape
    for rr, cc in np.argwhere(mask):
        r = int(top + rr)
        c = int(left + cc)
        r0 = max(0, r - pad)
        r1 = min(h, r + pad + 1)
        c0 = max(0, c - pad)
        c1 = min(w, c + pad + 1)
        if bool(np.any(used[r0:r1, c0:c1])):
            return False
    return True


def _place_slots(slot_masks: list[np.ndarray]) -> tuple[int, int, list[tuple[int, int]]] | None:
    max_h = max(mask.shape[0] for mask in slot_masks)
    max_w = max(mask.shape[1] for mask in slot_masks)
    total_area = sum(int(mask.sum()) for mask in slot_masks)
    min_h = max(9, int(max_h) + 4)
    min_w = max(9, int(max_w) + 4)
    max_h_lim = max(min_h, min(15, min_h + 4))
    max_w_lim = max(min_w, min(15, min_w + 5))

    for _ in range(BOARD_RETRIES):
        board_h = random.randint(min_h, max_h_lim)
        board_w = random.randint(min_w, max_w_lim)
        if board_h * board_w < total_area * 3:
            continue
        used = np.zeros((board_h, board_w), dtype=bool)
        placements: list[tuple[int, int] | None] = [None] * len(slot_masks)
        order = list(range(len(slot_masks)))
        order.sort(key=lambda idx: slot_masks[idx].shape[0] * slot_masks[idx].shape[1], reverse=True)

        ok = True
        for idx in order:
            mask = slot_masks[idx]
            mh, mw = mask.shape
            candidates = [(r, c) for r in range(1, board_h - mh) for c in range(1, board_w - mw)]
            random.shuffle(candidates)
            placed = False
            for top, left in candidates:
                if not _can_place_mask(used, mask, top, left, pad=1):
                    continue
                placements[idx] = (int(top), int(left))
                for rr, cc in np.argwhere(mask):
                    used[int(top + rr), int(left + cc)] = True
                placed = True
                break
            if not placed:
                ok = False
                break

        if ok and all(item is not None for item in placements):
            return int(board_h), int(board_w), [(int(r), int(c)) for r, c in placements if r is not None]
    return None


def _mark_blocked(blocked: np.ndarray, patch: np.ndarray, top: int, left: int, pad: int) -> None:
    h, w = blocked.shape
    for rr, cc in np.argwhere(patch != BG):
        r = int(top + rr)
        c = int(left + cc)
        r0 = max(0, r - pad)
        r1 = min(h, r + pad + 1)
        c0 = max(0, c - pad)
        c1 = min(w, c + pad + 1)
        blocked[r0:r1, c0:c1] = True


def _can_place_patch(blocked: np.ndarray, patch: np.ndarray, top: int, left: int) -> bool:
    h, w = blocked.shape
    ph, pw = patch.shape
    if top < 1 or left < 1 or top + ph >= h or left + pw >= w:
        return False
    for rr, cc in np.argwhere(patch != BG):
        if bool(blocked[int(top + rr), int(left + cc)]):
            return False
    return True


def _place_pieces(
    grid_h: int, grid_w: int, board_bbox: tuple[int, int, int, int], piece_patches: list[np.ndarray]
) -> list[tuple[int, int]] | None:
    blocked = np.zeros((grid_h, grid_w), dtype=bool)
    br0, bc0, br1, bc1 = board_bbox
    blocked[max(0, br0 - 1) : min(grid_h, br1 + 2), max(0, bc0 - 1) : min(grid_w, bc1 + 2)] = True

    positions: list[tuple[int, int] | None] = [None] * len(piece_patches)
    order = list(range(len(piece_patches)))
    order.sort(key=lambda idx: int(np.count_nonzero(piece_patches[idx])), reverse=True)

    for idx in order:
        patch = piece_patches[idx]
        ph, pw = patch.shape
        candidates = [(r, c) for r in range(1, grid_h - ph) for c in range(1, grid_w - pw)]
        random.shuffle(candidates)
        placed = False
        for top, left in candidates:
            if not _can_place_patch(blocked, patch, top, left):
                continue
            positions[idx] = (int(top), int(left))
            _mark_blocked(blocked, patch, top, left, pad=1)
            placed = True
            break
        if not placed:
            return None

    return [(int(r), int(c)) for r, c in positions if r is not None]


def _build_expected(
    inp: np.ndarray,
    board_top: int,
    board_left: int,
    specs: list[dict],
    piece_positions: list[tuple[int, int]],
) -> np.ndarray:
    out = inp.copy()
    for spec, (top, left) in zip(specs, piece_positions, strict=False):
        patch = np.asarray(spec["piece_patch"], dtype=int)
        region = out[top : top + patch.shape[0], left : left + patch.shape[1]]
        region[patch != BG] = BG

    for spec in specs:
        slot_top, slot_left = spec["slot_pos"]
        filled = np.asarray(spec["filled_slot_patch"], dtype=int)
        mask = np.asarray(spec["slot_mask"], dtype=bool)
        region = out[
            board_top + slot_top : board_top + slot_top + filled.shape[0],
            board_left + slot_left : board_left + slot_left + filled.shape[1],
        ]
        region[mask] = filled[mask]
    return out


def _gen_attempt() -> np.ndarray | None:
    ambiguous_mode = random.random() < 0.5
    n_pieces = random.randint(2, 4)
    specs = _sample_shape_specs(ambiguous_mode=ambiguous_mode, n_pieces=n_pieces)
    if specs is None:
        return None

    slot_masks = [np.asarray(spec["slot_mask"], dtype=bool) for spec in specs]
    placed_slots = _place_slots(slot_masks)
    if placed_slots is None:
        return None
    board_h, board_w, slot_positions = placed_slots

    board_patch = np.full((board_h, board_w), BOARD_COLOR, dtype=int)
    for spec, (slot_top, slot_left) in zip(specs, slot_positions, strict=False):
        mask = np.asarray(spec["slot_mask"], dtype=bool)
        region = board_patch[slot_top : slot_top + mask.shape[0], slot_left : slot_left + mask.shape[1]]
        region[mask] = BG
        spec["slot_pos"] = (int(slot_top), int(slot_left))

    for spec in specs:
        mask = np.asarray(spec["slot_mask"], dtype=bool)
        area = int(mask.sum())
        flat = _single_touch_seam_candidates(mask)
        if not flat:
            return None
        max_seams = min(len(flat), max(1, min(3, (area - 1) // 2)))
        if max_seams < 1:
            return None

        main_color = random.choice(PIECE_COLORS)
        seam_color = random.choice([color for color in PIECE_COLORS if color != main_color])
        seam_count = random.randint(1, max_seams)
        random.shuffle(flat)
        chosen = flat[:seam_count]

        fill = np.full(mask.shape, BG, dtype=int)
        fill[mask] = int(main_color)
        slot_top, slot_left = spec["slot_pos"]
        seen_adjacent: set[tuple[int, int]] = set()
        for rr, cc, d in chosen:
            dr, dc = DIRS[int(d)]
            ar = int(slot_top + rr + dr)
            ac = int(slot_left + cc + dc)
            if (ar, ac) in seen_adjacent or int(board_patch[ar, ac]) != BOARD_COLOR:
                return None
            seen_adjacent.add((ar, ac))
            board_patch[ar, ac] = int(seam_color)
            fill[int(rr), int(cc)] = int(seam_color)

        piece_rot = random.randrange(4)
        spec["filled_slot_patch"] = fill
        spec["piece_patch"] = np.rot90(fill, int(piece_rot))

    piece_patches = [np.asarray(spec["piece_patch"], dtype=int) for spec in specs]
    max_piece_h = max(patch.shape[0] for patch in piece_patches)
    max_piece_w = max(patch.shape[1] for patch in piece_patches)
    grid_h = board_h + max(8, max_piece_h + 4) + random.randint(0, 3)
    grid_w = board_w + max(8, max_piece_w + 4) + random.randint(0, 3)

    board_top_center = (grid_h - board_h) // 2
    board_left_center = (grid_w - board_w) // 2
    board_top = min(max(2, board_top_center + random.randint(-1, 1)), grid_h - board_h - 2)
    board_left = min(max(2, board_left_center + random.randint(-1, 1)), grid_w - board_w - 2)
    board_bbox = (board_top, board_left, board_top + board_h - 1, board_left + board_w - 1)

    piece_positions = _place_pieces(grid_h, grid_w, board_bbox, piece_patches)
    if piece_positions is None:
        return None

    inp = np.full((grid_h, grid_w), BG, dtype=int)
    inp[board_top : board_top + board_h, board_left : board_left + board_w] = board_patch
    for spec, (top, left) in zip(specs, piece_positions, strict=False):
        patch = np.asarray(spec["piece_patch"], dtype=int)
        region = inp[top : top + patch.shape[0], left : left + patch.shape[1]]
        region[patch != BG] = patch[patch != BG]

    expected = _build_expected(inp, board_top, board_left, specs, piece_positions)

    try:
        board, slots, pieces = _parse_instance(inp)
        shape_candidates = _candidate_lists(pieces, slots, require_seams=False)
        full_candidates = _candidate_lists(pieces, slots, require_seams=True)
        if ambiguous_mode:
            if not any(len(cands) > 1 for cands in shape_candidates):
                return None
        else:
            if not all(len(cands) == 1 for cands in shape_candidates):
                return None
        if len(_solve_assignments(full_candidates, len(slots), limit=2)) != 1:
            return None
        actual = main(inp)
    except Exception:
        return None

    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
