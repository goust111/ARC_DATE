from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREEN = 3
DIAGONALS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ORTHOGONALS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SHAPE_PATTERNS = [
    [".#.", "###", ".#."],
    [".#.", "###", ".#.", ".#."],
    ["##.", "###", ".#.", ".#."],
    [".#.", "##.", "###", ".#."],
    ["##.", "###", "##."],
    [".#.", "###", ".#.", "##."],
]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        steps = ORTHOGONALS
    else:
        steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in steps:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    return out


def _find_color_components(grid: np.ndarray, color: int, connectivity: int = 4) -> list[dict]:
    comps = []
    for coords in _components(grid == color, connectivity=connectivity):
        comps.append({"coords": coords, "bbox": _bbox_coords(coords), "area": int(coords.shape[0])})
    return comps


def _find_foreground_components(grid: np.ndarray, connectivity: int = 8) -> list[dict]:
    mask = grid != BLACK
    out = []
    for coords in _components(mask, connectivity=connectivity):
        r0, c0, r1, c1 = _bbox_coords(coords)
        patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
        rr = coords[:, 0] - r0
        cc = coords[:, 1] - c0
        patch[rr, cc] = grid[coords[:, 0], coords[:, 1]]
        colors = sorted(set(map(int, patch[patch != BLACK].tolist())))
        out.append(
            {
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "area": int(coords.shape[0]),
                "patch": patch,
                "mask": patch != BLACK,
                "colors": colors,
            }
        )
    return out


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, transparent: int = BLACK) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != transparent
    view[mask] = patch[mask]


def _scale_mask(mask: np.ndarray, k: int) -> np.ndarray:
    return np.kron(mask.astype(int), np.ones((k, k), dtype=int)).astype(bool)


def _dihedral_variants(patch: np.ndarray, mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    variants: list[tuple[np.ndarray, np.ndarray]] = []
    seen: set[tuple[bytes, tuple[int, int]]] = set()
    for rot in range(4):
        rp = np.rot90(patch, rot)
        rm = np.rot90(mask, rot)
        for fp, fm in ((rp, rm), (np.fliplr(rp), np.fliplr(rm))):
            key = (fm.tobytes(), tuple(map(int, fm.shape)))
            if key in seen:
                continue
            seen.add(key)
            variants.append((fp.copy(), fm.copy()))
    return variants


def _downsample_mask(mask: np.ndarray, k: int) -> np.ndarray | None:
    h, w = mask.shape
    if h % k != 0 or w % k != 0:
        return None
    out = np.zeros((h // k, w // k), dtype=bool)
    for r in range(h // k):
        for c in range(w // k):
            block = mask[r * k : (r + 1) * k, c * k : (c + 1) * k]
            if block.all():
                out[r, c] = True
            elif block.any():
                return None
    return out


def _shape_key(mask: np.ndarray) -> tuple[bytes, tuple[int, int]]:
    keys = []
    for rot in range(4):
        rm = np.rot90(mask, rot)
        for vm in (rm, np.fliplr(rm)):
            keys.append((vm.tobytes(), tuple(map(int, vm.shape))))
    return min(keys)


def _pattern_to_mask(rows: list[str]) -> np.ndarray:
    grid = np.array([[ch == "#" for ch in row] for row in rows], dtype=bool)
    coords = np.argwhere(grid)
    r0, c0, r1, c1 = _bbox_coords(coords)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _library_masks() -> list[np.ndarray]:
    masks = []
    seen = set()
    for rows in SHAPE_PATTERNS:
        mask = _pattern_to_mask(rows)
        key = _shape_key(mask)
        if key not in seen:
            seen.add(key)
            masks.append(mask)
    return masks


MASK_LIBRARY = _library_masks()
TEMPLATE_LIBRARY = [
    {
        "frame": [
            "33333333333",
            "33333333333",
            "33303330333",
            "33000300333",
            "33303300033",
            "33303330333",
            "33333333333",
            "30000000003",
            "33333333333",
        ],
        "slots": [(3, 2, 2), (1, 2, 6)],
    },
    {
        "frame": [
            "33333333333",
            "33333333333",
            "33303330333",
            "33000300033",
            "33303330333",
            "33003330333",
            "33333333333",
            "30000000003",
            "30000000003",
            "33333333333",
        ],
        "slots": [(5, 2, 2), (1, 2, 6)],
    },
    {
        "frame": [
            "33333333333",
            "30000000003",
            "30000000003",
            "33333333333",
            "33003330333",
            "33000300033",
            "33303330333",
            "33303330333",
            "33333333333",
            "33333333333",
        ],
        "slots": [(1, 4, 6), (2, 4, 2)],
    },
    {
        "frame": [
            "33333333333",
            "30000000003",
            "33333333333",
            "33303330333",
            "33003300033",
            "33000330333",
            "33303330333",
            "33333333333",
            "33333333333",
        ],
        "slots": [(1, 3, 2), (3, 3, 6)],
    },
    {
        "frame": [
            "33333333333",
            "30000000003",
            "33333333333",
            "33303330333",
            "33000300033",
            "33303330333",
            "33303300333",
            "33333333333",
            "33333333333",
        ],
        "slots": [(1, 3, 2), (5, 3, 6)],
    },
]


def _piece_from_slot_mask(slot_mask: np.ndarray, k: int, inner_color: int) -> np.ndarray:
    union = _scale_mask(slot_mask, k)
    h, w = union.shape
    patch = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            if not union[r, c]:
                continue
            is_boundary = False
            for dr, dc in ORTHOGONALS:
                nr, nc = r + dr, c + dc
                if not _in_bounds(union.shape, nr, nc) or not union[nr, nc]:
                    is_boundary = True
                    break
            patch[r, c] = GREEN if is_boundary else inner_color
    assert np.any(patch == inner_color)
    return patch


def _frame_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    candidates = []
    for comp in _find_color_components(grid, GREEN, connectivity=4):
        r0, c0, r1, c1 = comp["bbox"]
        if r1 - r0 + 1 < 6 or c1 - c0 + 1 < 6:
            continue
        crop = _crop(grid, comp["bbox"])
        if not (
            np.all(crop[0, :] == GREEN)
            and np.all(crop[-1, :] == GREEN)
            and np.all(crop[:, 0] == GREEN)
            and np.all(crop[:, -1] == GREEN)
        ):
            continue
        black_comps = _components(crop == BLACK, connectivity=4)
        if len(black_comps) < 3:
            continue
        areas = sorted(len(c) for c in black_comps)
        if areas[-1] <= areas[-2]:
            continue
        candidates.append((len(black_comps), (r1 - r0 + 1) * (c1 - c0 + 1), comp["bbox"]))
    assert candidates, "failed to find outer frame"
    return max(candidates)[2]


def _extract_slots(frame_crop: np.ndarray) -> list[dict]:
    black_comps = _components(frame_crop == BLACK, connectivity=4)
    assert len(black_comps) >= 3, "expected ambient background and 2-3 slots"
    largest = max(range(len(black_comps)), key=lambda i: len(black_comps[i]))
    slots = []
    for i, coords in enumerate(black_comps):
        if i == largest:
            continue
        r0, c0, r1, c1 = _bbox_coords(coords)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
        slots.append({"r0": r0, "c0": c0, "mask": mask, "area": int(mask.sum())})
    slots.sort(key=lambda s: (s["area"], s["r0"], s["c0"]))
    assert 2 <= len(slots) <= 3, "expected 2-3 slots"
    return slots


def _extract_pieces(grid: np.ndarray, frame_box: tuple[int, int, int, int]) -> list[dict]:
    fr0, fc0, fr1, fc1 = frame_box
    pieces = []
    for comp in _find_foreground_components(grid, connectivity=8):
        colors = comp["colors"]
        if colors == [GREEN]:
            continue
        if GREEN not in colors or len(colors) != 2:
            continue
        r0, c0, r1, c1 = comp["bbox"]
        if fr0 <= r0 and c0 >= fc0 and r1 <= fr1 and c1 <= fc1:
            continue
        inner = [color for color in colors if color != GREEN][0]
        pieces.append({"patch": comp["patch"], "mask": comp["mask"], "inner": int(inner)})
    assert 2 <= len(pieces) <= 3, "expected 2-3 external pieces"
    return pieces


def _match_slots_and_pieces(slots: list[dict], pieces: list[dict]) -> tuple[int, list[dict]]:
    assert len(slots) == len(pieces)
    solutions: list[tuple[int, list[dict]]] = []
    for k in (2, 3):
        candidate_lists: list[list[dict]] = []
        for slot in slots:
            slot_cands = []
            for piece_index, piece in enumerate(pieces):
                for patch, mask in _dihedral_variants(piece["patch"], piece["mask"]):
                    coarse = _downsample_mask(mask, k)
                    if coarse is None or coarse.shape != slot["mask"].shape:
                        continue
                    if np.array_equal(coarse, slot["mask"]):
                        slot_cands.append(
                            {
                                "piece_index": piece_index,
                                "patch": patch,
                                "mask": mask,
                                "inner": piece["inner"],
                                "slot_area": slot["area"],
                            }
                        )
            if not slot_cands:
                candidate_lists = []
                break
            candidate_lists.append(slot_cands)
        if not candidate_lists:
            continue
        order = sorted(range(len(slots)), key=lambda i: len(candidate_lists[i]))
        used: set[int] = set()
        current: list[dict | None] = [None] * len(slots)

        def backtrack(pos: int) -> None:
            if len(solutions) > 1:
                return
            if pos == len(order):
                solutions.append((k, [entry for entry in current if entry is not None]))
                return
            slot_index = order[pos]
            for cand in candidate_lists[slot_index]:
                if cand["piece_index"] in used:
                    continue
                used.add(cand["piece_index"])
                current[slot_index] = cand
                backtrack(pos + 1)
                current[slot_index] = None
                used.remove(cand["piece_index"])

        backtrack(0)
    assert len(solutions) == 1, "expected exactly one global matching"
    return solutions[0]


def _solve(grid: np.ndarray, return_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    frame_box = _frame_bbox(g)
    frame_crop = _crop(g, frame_box)
    slots = _extract_slots(frame_crop)
    pieces = _extract_pieces(g, frame_box)
    k, matching = _match_slots_and_pieces(slots, pieces)

    out_h, out_w = frame_crop.shape[0] * k, frame_crop.shape[1] * k
    out = np.zeros((out_h, out_w), dtype=int)
    out[:k, :] = GREEN
    out[-k:, :] = GREEN
    out[:, :k] = GREEN
    out[:, -k:] = GREEN

    per_piece_green = []
    for slot, match in zip(slots, matching):
        top = slot["r0"] * k
        left = slot["c0"] * k
        _paste(out, match["patch"], top, left, transparent=BLACK)
        gmask = np.zeros_like(out, dtype=bool)
        green_local = match["patch"] == GREEN
        gmask[top : top + match["patch"].shape[0], left : left + match["patch"].shape[1]] = green_local
        per_piece_green.append(
            {
                "green": gmask,
                "inner": int(match["inner"]),
                "area": int(slot["area"]),
            }
        )

    conflict_pixels = 0
    halo_pixels = 0
    for r in range(k, out_h - k):
        for c in range(k, out_w - k):
            if out[r, c] != BLACK:
                continue
            if any(_in_bounds(out.shape, r + dr, c + dc) and out[r + dr, c + dc] == GREEN for dr, dc in ORTHOGONALS):
                continue
            candidates = []
            for piece in per_piece_green:
                if any(
                    _in_bounds(out.shape, r + dr, c + dc) and piece["green"][r + dr, c + dc]
                    for dr, dc in DIAGONALS
                ):
                    candidates.append(piece)
            if not candidates:
                continue
            if len(candidates) > 1:
                conflict_pixels += 1
            chosen = min(candidates, key=lambda item: item["area"])
            out[r, c] = int(chosen["inner"])
            halo_pixels += 1

    if return_meta:
        return out, {"k": k, "halo_pixels": halo_pixels, "conflict_pixels": conflict_pixels}
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each scaled external piece to its frame slot, place it back with the right orientation, and add diagonal-only inner-color halos.

    Input structure:
    - The grid has black background, one large green rectangular frame near the center, and 2-3 external multicolor pieces.
    - Inside the frame bbox, black cells split into one large ambient background component plus 2-3 smaller slot components.
    - Each external piece is made of k×k blocks (k=2 or 3): its perimeter pixels are green and its interior pixels use one unique non-green color.

    Rule:
    - Extract the small black slot components inside the frame.
    - For each external piece, downsample its occupied mask by k and compare all 8 orientations to the slot masks to find the unique matching slot and orientation.
    - Build the output as the k-scaled frame crop: keep only the outer green border on black, then paste each matched oriented piece into its slot position at k-scale.
    - Finally, recolor every remaining black pixel that is diagonally adjacent to a pasted green piece-boundary pixel and not orthogonally adjacent to any green pixel; if multiple pieces compete, use the inner color of the smaller slot.

    Key insight:
    - The slot match fixes both which piece goes where and which rotated/flipped green boundary shape appears after placement; that boundary shape then determines the diagonal-only halo pixels.
    """
    return _solve(grid)


def _slot_with_offset(mask: np.ndarray, r0: int, c0: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    cells = {(r0 + r, c0 + c) for r, c in np.argwhere(mask)}
    h, w = mask.shape
    shell = {
        (rr, cc)
        for rr in range(r0 - 1, r0 + h + 1)
        for cc in range(c0 - 1, c0 + w + 1)
        if (rr, cc) not in cells
    }
    return cells, shell


def _place_slots(crop_h: int, crop_w: int, masks: list[np.ndarray]) -> tuple[np.ndarray, list[dict]] | None:
    for _ in range(150):
        placements: list[dict] = []
        occupied_slot: set[tuple[int, int]] = set()
        ok = True
        for mask in random.sample(masks, len(masks)):
            h, w = mask.shape
            options = []
            for r0 in range(1, crop_h - h - 1):
                for c0 in range(1, crop_w - w - 1):
                    cells, moat = _slot_with_offset(mask, r0, c0)
                    if any(r <= 0 or c <= 0 or r >= crop_h - 1 or c >= crop_w - 1 for r, c in cells | moat):
                        continue
                    if cells & occupied_slot:
                        continue
                    options.append((r0, c0, cells, moat))
            if not options:
                ok = False
                break
            r0, c0, cells, moat = random.choice(options)
            placements.append({"mask": mask, "r0": r0, "c0": c0, "cells": cells, "moat": moat})
            occupied_slot |= cells
        if not ok:
            continue

        grid = np.zeros((crop_h, crop_w), dtype=int)
        grid[0, :] = GREEN
        grid[-1, :] = GREEN
        grid[:, 0] = GREEN
        grid[:, -1] = GREEN
        for item in placements:
            for rr, cc in item["moat"]:
                grid[rr, cc] = GREEN

        black_comps = _components(grid == BLACK, connectivity=4)
        if len(black_comps) != len(masks) + 1:
            continue
        areas = sorted(len(comp) for comp in black_comps)
        if areas[-1] <= areas[-2]:
            continue
        placements.sort(key=lambda item: (item["r0"], item["c0"]))
        return grid, placements
    return None


def _random_piece_variants(slot_placements: list[dict], k: int, palette: list[int]) -> list[np.ndarray]:
    colors = random.sample(palette, len(slot_placements))
    pieces = []
    for slot, color in zip(slot_placements, colors):
        mask = slot["mask"]
        rot = random.randrange(4)
        oriented = np.rot90(mask, rot)
        if random.random() < 0.5:
            oriented = np.fliplr(oriented)
        piece = _piece_from_slot_mask(oriented, k, color)
        pieces.append(piece)
    return pieces


def _can_place_patch(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    h, w = patch.shape
    if r0 < 0 or c0 < 0 or r0 + h > canvas.shape[0] or c0 + w > canvas.shape[1]:
        return False
    occ = patch != BLACK
    rs = range(max(0, r0 - 1), min(canvas.shape[0], r0 + h + 1))
    cs = range(max(0, c0 - 1), min(canvas.shape[1], c0 + w + 1))
    if np.any(canvas[np.ix_(list(rs), list(cs))] != BLACK):
        return False
    return not np.any(canvas[r0 : r0 + h, c0 : c0 + w][occ] != BLACK)


def _build_candidate() -> np.ndarray | None:
    k = 2
    template = random.choice(TEMPLATE_LIBRARY)
    frame_crop = np.array([[int(ch) for ch in row] for row in template["frame"]], dtype=int)
    crop_h, crop_w = frame_crop.shape
    slot_placements = []
    for mask_index, r0, c0 in template["slots"]:
        slot_placements.append({"mask": MASK_LIBRARY[mask_index], "r0": r0, "c0": c0})

    piece_palette = [1, 2, 4, 6, 7, 8, 9]
    pieces = _random_piece_variants(slot_placements, k, piece_palette)

    piece_gap = random.choice([1, 2])
    total_piece_width = sum(piece.shape[1] for piece in pieces) + piece_gap * (len(pieces) - 1)
    max_piece_height = max(piece.shape[0] for piece in pieces)
    max_piece_width = max(piece.shape[1] for piece in pieces)

    if random.random() < 0.5:
        canvas_h = max(18, crop_h + max_piece_height + 4)
        canvas_w = max(18, crop_w + total_piece_width + 5)
        if canvas_h > 30 or canvas_w > 30:
            return None
        canvas = np.zeros((canvas_h, canvas_w), dtype=int)
        frame_top = canvas_h - crop_h - 1
        frame_left = 1
        canvas[frame_top : frame_top + crop_h, frame_left : frame_left + crop_w] = frame_crop
        cur_c = canvas_w - total_piece_width - 1
        for patch in pieces:
            _paste(canvas, patch, 1, cur_c, transparent=BLACK)
            cur_c += patch.shape[1] + piece_gap
    else:
        total_piece_height = sum(piece.shape[0] for piece in pieces) + piece_gap * (len(pieces) - 1)
        canvas_h = max(18, crop_h + total_piece_height + 5)
        canvas_w = max(18, crop_w + max_piece_width + 4)
        if canvas_h > 30 or canvas_w > 30:
            return None
        canvas = np.zeros((canvas_h, canvas_w), dtype=int)
        frame_top = canvas_h - crop_h - 1
        frame_left = 1
        canvas[frame_top : frame_top + crop_h, frame_left : frame_left + crop_w] = frame_crop
        cur_r = 1
        piece_left = canvas_w - max_piece_width - 1
        for patch in pieces:
            _paste(canvas, patch, cur_r, piece_left, transparent=BLACK)
            cur_r += patch.shape[0] + piece_gap

    try:
        out, meta = _solve(canvas, return_meta=True)
    except Exception:
        return None
    if meta["halo_pixels"] == 0:
        return None
    if max(out.shape) > 30:
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(600):
        grid = _build_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
