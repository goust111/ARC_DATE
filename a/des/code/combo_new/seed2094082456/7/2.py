from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
SIDES = ("up", "right", "down", "left")
SIDE_TO_INDEX = {side: idx for idx, side in enumerate(SIDES)}

CELL = 2
LOGICAL_ROWS = 3
LOGICAL_COLS = 4
OBJECT_COUNT_CHOICES = (3, 4)
GEN_RETRIES = 500
PLACEMENT_RETRIES = 120
PARTITION_LIBRARY: dict[int, list[list[set[tuple[int, int]]]]] = {
    3: [
        [
            {(0, 0), (1, 0), (2, 0), (1, 1)},
            {(0, 1), (0, 2), (0, 3), (1, 3)},
            {(1, 2), (2, 1), (2, 2), (2, 3)},
        ],
    ],
    4: [
        [
            {(0, 0), (0, 1), (1, 0)},
            {(0, 2), (0, 3), (1, 3)},
            {(1, 1), (2, 0), (2, 1)},
            {(1, 2), (2, 2), (2, 3)},
        ],
    ],
}


def _neighbors(r: int, c: int, h: int, w: int, connectivity: int) -> list[tuple[int, int]]:
    deltas4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    deltas8 = deltas4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    out: list[tuple[int, int]] = []
    for dr, dc in deltas8 if connectivity == 8 else deltas4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((int(nr), int(nc)))
    return out


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors(rr, cc, h, w, connectivity):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    assert pts.size > 0
    r0 = int(pts[:, 0].min())
    r1 = int(pts[:, 0].max())
    c0 = int(pts[:, 1].min())
    c1 = int(pts[:, 1].max())
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_from_points(points: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(points)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in points:
        out[r - r0, c - c0] = True
    return out


def _boundary_candidates(mask: np.ndarray, side: str) -> list[tuple[int, int]]:
    pts = np.argwhere(mask)
    r0 = int(pts[:, 0].min())
    r1 = int(pts[:, 0].max())
    c0 = int(pts[:, 1].min())
    c1 = int(pts[:, 1].max())
    coords: list[tuple[int, int]] = []
    if side == "up":
        coords = [(r, c) for r, c in map(tuple, pts) if int(r) == r0 and c0 < int(c) < c1]
    elif side == "down":
        coords = [(r, c) for r, c in map(tuple, pts) if int(r) == r1 and c0 < int(c) < c1]
    elif side == "left":
        coords = [(r, c) for r, c in map(tuple, pts) if int(c) == c0 and r0 < int(r) < r1]
    else:
        coords = [(r, c) for r, c in map(tuple, pts) if int(c) == c1 and r0 < int(r) < r1]
    return [(int(r), int(c)) for r, c in coords]


def _tail_candidates(body_mask: np.ndarray, side: str) -> list[tuple[int, int]]:
    h, w = body_mask.shape
    out: list[tuple[int, int]] = []
    if side == "up":
        for c in range(w):
            if body_mask[0, c]:
                continue
            diag = (c > 0 and body_mask[0, c - 1]) or (c + 1 < w and body_mask[0, c + 1])
            if diag:
                out.append((-1, int(c)))
    elif side == "down":
        for c in range(w):
            if body_mask[h - 1, c]:
                continue
            diag = (c > 0 and body_mask[h - 1, c - 1]) or (c + 1 < w and body_mask[h - 1, c + 1])
            if diag:
                out.append((int(h), int(c)))
    elif side == "left":
        for r in range(h):
            if body_mask[r, 0]:
                continue
            diag = (r > 0 and body_mask[r - 1, 0]) or (r + 1 < h and body_mask[r + 1, 0])
            if diag:
                out.append((int(r), -1))
    else:
        for r in range(h):
            if body_mask[r, w - 1]:
                continue
            diag = (r > 0 and body_mask[r - 1, w - 1]) or (r + 1 < h and body_mask[r + 1, w - 1])
            if diag:
                out.append((int(r), int(w)))
    return out


def _side_from_point_on_bbox(point: tuple[int, int], bbox: tuple[int, int, int, int]) -> str:
    r, c = point
    r0, c0, r1, c1 = bbox
    if r == r0 and c0 < c < c1:
        return "up"
    if c == c1 and r0 < r < r1:
        return "right"
    if r == r1 and c0 < c < c1:
        return "down"
    if c == c0 and r0 < r < r1:
        return "left"
    cr = (r0 + r1) / 2.0
    cc = (c0 + c1) / 2.0
    dr = r - cr
    dc = c - cc
    if abs(dr) >= abs(dc):
        return "down" if dr > 0 else "up"
    return "right" if dc > 0 else "left"


def _template_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(grid == GREY)
    assert pts.size > 0, "expected a grey template"
    return (
        int(pts[:, 0].min()),
        int(pts[:, 1].min()),
        int(pts[:, 0].max()),
        int(pts[:, 1].max()),
    )


def _extract_slots(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> dict[int, dict[str, object]]:
    r0, c0, r1, c1 = bbox
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    slot_mask = interior != GREY
    slots: dict[int, dict[str, object]] = {}
    for comp in _components(slot_mask, connectivity=4):
        global_pts = [(r0 + 1 + r, c0 + 1 + c) for r, c in comp]
        colors = {int(grid[r, c]) for r, c in global_pts if int(grid[r, c]) != BG}
        assert len(colors) == 1, "each slot must contain exactly one colored marker"
        color = colors.pop()
        marker_pts = [(r, c) for r, c in global_pts if int(grid[r, c]) == color]
        assert len(marker_pts) == 1
        comp_bbox = _bbox(global_pts)
        marker_side = _side_from_point_on_bbox(marker_pts[0], comp_bbox)
        slots[color] = {
            "bbox": comp_bbox,
            "cells": global_pts,
            "mask": _mask_from_points(global_pts),
            "marker_side": marker_side,
        }
    return slots


def _extract_pieces(grid: np.ndarray, template_bbox: tuple[int, int, int, int]) -> dict[int, dict[str, object]]:
    tr0, tc0, tr1, tc1 = template_bbox
    pieces: dict[int, dict[str, object]] = {}
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in {BG, GREY}):
        comps8 = _components(grid == color, connectivity=8)
        piece_comp = None
        for comp in comps8:
            if len(comp) > 1 and any(not (tr0 <= r <= tr1 and tc0 <= c <= tc1) for r, c in comp):
                piece_comp = comp
                break
        if piece_comp is None:
            continue
        r0, c0, r1, c1 = _bbox(piece_comp)
        patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in piece_comp:
            patch[r - r0, c - c0] = True
        comps4 = _components(patch, connectivity=4)
        assert len(comps4) == 2, "piece should split into body plus tail under 4-connectivity"
        comps4.sort(key=len)
        assert len(comps4[0]) == 1 and len(comps4[1]) > 1
        tail = comps4[0][0]
        body_pts = comps4[1]
        br0, bc0, br1, bc1 = _bbox(body_pts)
        if tail[0] < br0:
            tail_side = "up"
        elif tail[0] > br1:
            tail_side = "down"
        elif tail[1] < bc0:
            tail_side = "left"
        else:
            tail_side = "right"
        body_mask = np.zeros((br1 - br0 + 1, bc1 - bc0 + 1), dtype=bool)
        for r, c in body_pts:
            body_mask[r - br0, c - bc0] = True
        pieces[color] = {"body_mask": body_mask, "tail_side": tail_side}
    return pieces


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate scattered same-color fragments into the matching grey template slots.

    Input structure:
    - The background is black.
    - A grey rectangular template sits on the left; grey internal walls partition its interior into several connected slots.
    - Each slot contains exactly one colored single-pixel marker, and the right side contains one same-color fragment whose body is 4-connected plus one diagonally attached tail pixel.

    Rule:
    - Pair every fragment with the template slot that contains the same color marker.
    - Read the tail pixel's side relative to the fragment body's tight bbox, and read the slot marker's side relative to the slot's tight bbox.
    - Rotate the fragment body clockwise by the side difference from tail direction to slot-marker direction.
    - Place the rotated body so it exactly fills the slot, keep the grey template lines, and erase the original right-side fragments.

    Key insight:
    - Color only performs the fragment-to-slot matching; the actual rotation is encoded independently by the fragment tail and by the slot marker side.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    template_bbox = _template_bbox(g)
    slots = _extract_slots(g, template_bbox)
    pieces = _extract_pieces(g, template_bbox)
    assert slots and pieces and set(slots) == set(pieces)

    out = np.zeros_like(g)
    out[g == GREY] = GREY

    for color, slot in slots.items():
        body_mask = np.asarray(pieces[color]["body_mask"], dtype=bool)
        tail_side = str(pieces[color]["tail_side"])
        marker_side = str(slot["marker_side"])
        turns = (SIDE_TO_INDEX[marker_side] - SIDE_TO_INDEX[tail_side]) % 4
        rotated = _crop_mask(np.rot90(body_mask, -turns))
        assert np.array_equal(rotated, slot["mask"]), "rotated fragment must match its slot"
        for r, c in slot["cells"]:
            out[r, c] = int(color)
    return out


def _logical_neighbors(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    out: list[tuple[int, int]] = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < LOGICAL_ROWS and 0 <= nc < LOGICAL_COLS:
            out.append((int(nr), int(nc)))
    return out


def _is_rectangular_region(region: set[tuple[int, int]]) -> bool:
    rows = [r for r, _ in region]
    cols = [c for _, c in region]
    return len(region) == (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)


def _region_targets(rng: random.Random, n_regions: int) -> list[int]:
    del rng, n_regions
    return []


def _random_partition(rng: random.Random, n_regions: int) -> list[set[tuple[int, int]]] | None:
    bases = PARTITION_LIBRARY.get(n_regions)
    if not bases:
        return None
    regions = [set(region) for region in rng.choice(bases)]
    if rng.random() < 0.5:
        regions = [{(LOGICAL_ROWS - 1 - r, c) for r, c in region} for region in regions]
    if rng.random() < 0.5:
        regions = [{(r, LOGICAL_COLS - 1 - c) for r, c in region} for region in regions]
    rng.shuffle(regions)
    assert all(not _is_rectangular_region(region) for region in regions)
    return regions


def _slot_masks_from_regions(regions: list[set[tuple[int, int]]]) -> list[np.ndarray]:
    h = 2 + LOGICAL_ROWS * CELL + (LOGICAL_ROWS - 1)
    w = 2 + LOGICAL_COLS * CELL + (LOGICAL_COLS - 1)
    masks = [np.zeros((h, w), dtype=bool) for _ in regions]

    def block_origin(cell: tuple[int, int]) -> tuple[int, int]:
        r, c = cell
        return 1 + r * (CELL + 1), 1 + c * (CELL + 1)

    for idx, region in enumerate(regions):
        for cell in region:
            rr, cc = block_origin(cell)
            masks[idx][rr : rr + CELL, cc : cc + CELL] = True
        for r, c in region:
            if (r + 1, c) in region:
                rr, cc = block_origin((r, c))
                masks[idx][rr + CELL, cc : cc + CELL] = True
            if (r, c + 1) in region:
                rr, cc = block_origin((r, c))
                masks[idx][rr : rr + CELL, cc + CELL] = True
    return masks


def _build_template_and_pieces(
    rng: random.Random,
    slot_masks: list[np.ndarray],
    colors: list[int],
) -> tuple[np.ndarray, list[np.ndarray]] | None:
    template_h = 2 + LOGICAL_ROWS * CELL + (LOGICAL_ROWS - 1)
    template_w = 2 + LOGICAL_COLS * CELL + (LOGICAL_COLS - 1)
    template = np.full((template_h, template_w), GREY, dtype=int)
    piece_patches: list[np.ndarray] = []

    for slot_mask_full, color in zip(slot_masks, colors):
        slot_mask = _crop_mask(slot_mask_full)
        feasible: list[tuple[str, tuple[int, int], str, np.ndarray, tuple[int, int]]] = []
        for slot_side in SIDES:
            marker_cells = _boundary_candidates(slot_mask_full, slot_side)
            if not marker_cells:
                continue
            for tail_side in SIDES:
                turns = (SIDE_TO_INDEX[slot_side] - SIDE_TO_INDEX[tail_side]) % 4
                body_mask = _crop_mask(np.rot90(slot_mask, turns))
                tail_cells = _tail_candidates(body_mask, tail_side)
                if not tail_cells:
                    continue
                for marker in marker_cells:
                    for tail in tail_cells:
                        feasible.append((slot_side, marker, tail_side, body_mask, tail))
        if not feasible:
            return None
        slot_side, marker, _tail_side, body_mask, tail = rng.choice(feasible)
        del slot_side
        template[slot_mask_full] = BG
        template[marker] = int(color)

        body_pts = [(int(r), int(c)) for r, c in np.argwhere(body_mask)]
        all_pts = body_pts + [tail]
        min_r = min(r for r, _ in all_pts)
        min_c = min(c for _, c in all_pts)
        max_r = max(r for r, _ in all_pts)
        max_c = max(c for _, c in all_pts)
        patch = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=int)
        for r, c in body_pts:
            patch[r - min_r, c - min_c] = int(color)
        patch[tail[0] - min_r, tail[1] - min_c] = int(color)
        piece_patches.append(patch)
    return template, piece_patches


def _place_pieces(
    rng: random.Random,
    template: np.ndarray,
    piece_patches: list[np.ndarray],
) -> np.ndarray | None:
    template_h, template_w = template.shape
    max_piece_h = max(p.shape[0] for p in piece_patches)
    max_piece_w = max(p.shape[1] for p in piece_patches)
    min_h = max(template_h + 2, max_piece_h + 3, 20)
    min_w = max(template_w + max_piece_w + 4, 24)
    if min_h > 30 or min_w > 30:
        return None

    for _ in range(PLACEMENT_RETRIES):
        h = rng.randint(min_h, 30)
        w = rng.randint(min_w, 30)
        grid = np.zeros((h, w), dtype=int)
        top = rng.randint(1, h - template_h - 1)
        left = rng.randint(1, 2)
        grid[top : top + template_h, left : left + template_w] = template
        occ = np.zeros((h, w), dtype=bool)
        rr0 = max(0, top - 1)
        rr1 = min(h, top + template_h + 1)
        cc0 = max(0, left - 1)
        cc1 = min(w, left + template_w + 1)
        occ[rr0:rr1, cc0:cc1] = True

        positions: list[tuple[int, int]] = [(-1, -1)] * len(piece_patches)
        order = sorted(range(len(piece_patches)), key=lambda idx: piece_patches[idx].size, reverse=True)
        min_col = left + template_w + 2
        if min_col >= w - 1:
            continue

        ok = True
        for idx in order:
            patch = piece_patches[idx]
            ph, pw = patch.shape
            if pw > w - min_col - 1 or ph > h - 2:
                ok = False
                break
            candidates: list[tuple[int, int]] = []
            for _ in range(200):
                r = rng.randint(1, h - ph - 1)
                c = rng.randint(min_col, w - pw - 1)
                pad_r0 = r - 1
                pad_r1 = r + ph + 1
                pad_c0 = c - 1
                pad_c1 = c + pw + 1
                if occ[pad_r0:pad_r1, pad_c0:pad_c1].any():
                    continue
                candidates.append((r, c))
                if len(candidates) >= 10:
                    break
            if not candidates:
                ok = False
                break
            r, c = rng.choice(candidates)
            positions[idx] = (r, c)
            occ[r - 1 : r + ph + 1, c - 1 : c + pw + 1] = True
        if not ok:
            continue

        for patch, (r, c) in zip(piece_patches, positions):
            mask = patch != BG
            grid[r : r + patch.shape[0], c : c + patch.shape[1]][mask] = patch[mask]
        return grid
    return None


def _generate_candidate(rng: random.Random) -> np.ndarray | None:
    n_objects = rng.choice(OBJECT_COUNT_CHOICES)
    colors = rng.sample([c for c in range(1, 10) if c != GREY], n_objects)

    for _ in range(200):
        regions = _random_partition(rng, n_objects)
        if regions is None:
            continue
        slot_masks = _slot_masks_from_regions(regions)
        built = _build_template_and_pieces(rng, slot_masks, colors)
        if built is None:
            continue
        template, piece_patches = built
        grid = _place_pieces(rng, template, piece_patches)
        if grid is None:
            continue
        try:
            solved = main(grid)
        except Exception:
            continue
        if solved.shape == grid.shape and max(grid.shape) <= 30 and max(solved.shape) <= 30:
            return grid
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _generate_candidate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid puzzle instance")
