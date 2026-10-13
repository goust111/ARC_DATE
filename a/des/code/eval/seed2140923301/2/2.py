from __future__ import annotations

import random

import numpy as np


BG = 0
YELLOW = 4
GREY = 5
MAROON = 9
PIECE_COLORS = (1, 2, 3, 6, 7, 8)

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
NEIGHBORS8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
)


def _find_components(grid: np.ndarray, colors: set[int]) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color not in colors:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                    "area": len(coords),
                }
            )
    return comps


def _find_frame(grid: np.ndarray) -> tuple[int, int, int, int]:
    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for comp in _find_components(grid, {GREY}):
        r0, c0, r1, c1 = comp["bbox"]
        if r1 - r0 < 3 or c1 - c0 < 3:
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros(sub.shape, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if not np.all(sub[border] == GREY):
            continue
        if not np.any(sub[1:-1, 1:-1] == BG):
            continue
        candidates.append((sub.shape[0] * sub.shape[1], (r0, c0, r1, c1)))
    assert candidates, "expected a grey frame"
    max_area = max(area for area, _bbox in candidates)
    best = [bbox for area, bbox in candidates if area == max_area]
    assert len(best) == 1, "frame must be unique"
    return best[0]


def _marker_positions(grid: np.ndarray) -> list[tuple[int, int]]:
    markers = _find_components(grid, {MAROON})
    assert markers, "expected at least one marker"
    pts: list[tuple[int, int]] = []
    for comp in markers:
        assert comp["area"] == 1, "markers must be single pixels"
        pts.append(comp["coords"][0])
    return pts


def _piece_direction(piece: dict, markers: list[tuple[int, int]]) -> tuple[str, tuple[int, int], tuple[int, int]]:
    r0, c0, r1, c1 = piece["bbox"]
    hits: list[tuple[str, tuple[int, int], tuple[int, int]]] = []
    for mr, mc in markers:
        if mr == r0 - 1 and c0 <= mc <= c1:
            hits.append(("up", DIRS["up"], (mr, mc)))
        if mr == r1 + 1 and c0 <= mc <= c1:
            hits.append(("down", DIRS["down"], (mr, mc)))
        if mc == c0 - 1 and r0 <= mr <= r1:
            hits.append(("left", DIRS["left"], (mr, mc)))
        if mc == c1 + 1 and r0 <= mr <= r1:
            hits.append(("right", DIRS["right"], (mr, mc)))
    assert len(hits) == 1, "each piece must have exactly one adjacent marker"
    return hits[0]


def _cells_from_mask(mask: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    return [(top + int(r), left + int(c)) for r, c in np.argwhere(mask)]


def _bitmask_from_cells(cells: list[tuple[int, int]], frame_bbox: tuple[int, int, int, int]) -> int:
    fr0, fc0, _fr1, fc1 = frame_bbox
    width = fc1 - fc0 + 1
    bits = 0
    for r, c in cells:
        idx = (r - fr0) * width + (c - fc0)
        bits |= 1 << idx
    return bits


def _candidate_placements(grid: np.ndarray, piece: dict, direction: tuple[int, int], frame_bbox: tuple[int, int, int, int]) -> list[dict]:
    r0, c0, _r1, _c1 = piece["bbox"]
    mask = piece["mask"]
    ph, pw = mask.shape
    fr0, fc0, fr1, fc1 = frame_bbox
    dr, dc = direction
    candidates: list[dict] = []
    if dr == 0 and dc == 1:
        if not (fr0 <= r0 and r0 + ph - 1 <= fr1):
            return []
        col_iter = range(max(fc0, c0 + 1), fc1 - pw + 2)
        row_iter = [r0]
    elif dr == 0 and dc == -1:
        if not (fr0 <= r0 and r0 + ph - 1 <= fr1):
            return []
        col_iter = range(min(fc1 - pw + 1, c0 - 1), fc0 - 1, -1)
        row_iter = [r0]
    elif dr == 1 and dc == 0:
        if not (fc0 <= c0 and c0 + pw - 1 <= fc1):
            return []
        row_iter = range(max(fr0, r0 + 1), fr1 - ph + 2)
        col_iter = [c0]
    else:
        if not (fc0 <= c0 and c0 + pw - 1 <= fc1):
            return []
        row_iter = range(min(fr1 - ph + 1, r0 - 1), fr0 - 1, -1)
        col_iter = [c0]

    for nr in row_iter:
        for nc in col_iter:
            cells = _cells_from_mask(mask, int(nr), int(nc))
            if all(fr0 <= rr <= fr1 and fc0 <= cc <= fc1 and int(grid[rr, cc]) == BG for rr, cc in cells):
                candidates.append(
                    {
                        "top_left": (int(nr), int(nc)),
                        "cells": cells,
                        "bits": _bitmask_from_cells(cells, frame_bbox),
                        "distance": abs(int(nr) - r0) + abs(int(nc) - c0),
                    }
                )
    candidates.sort(key=lambda item: (int(item["distance"]), int(item["top_left"][0]), int(item["top_left"][1])))
    return candidates


def _solve_assignments(slot_bits: int, pieces: list[dict]) -> list[dict]:
    ordered = sorted(
        range(len(pieces)),
        key=lambda idx: (len(pieces[idx]["candidates"]), -int(pieces[idx]["area"]), idx),
    )
    suffix_union = [0] * (len(ordered) + 1)
    for pos in range(len(ordered) - 1, -1, -1):
        union = suffix_union[pos + 1]
        for cand in pieces[ordered[pos]]["candidates"]:
            union |= int(cand["bits"])
        suffix_union[pos] = union

    best_cost: int | None = None
    best_choice: dict[int, dict] | None = None

    def search(pos: int, covered: int, cost: int, chosen: dict[int, dict]) -> None:
        nonlocal best_cost, best_choice
        if best_cost is not None and cost >= best_cost:
            return
        if (covered | suffix_union[pos]) != slot_bits:
            return
        if pos == len(ordered):
            if covered == slot_bits:
                best_cost = cost
                best_choice = chosen.copy()
            return
        piece_idx = ordered[pos]
        for cand in pieces[piece_idx]["candidates"]:
            chosen[piece_idx] = cand
            search(pos + 1, covered | int(cand["bits"]), cost + int(cand["distance"]), chosen)
            del chosen[piece_idx]

    search(0, 0, 0, {})
    assert best_choice is not None, "could not assign pieces to slots"
    return [best_choice[idx] for idx in range(len(pieces))]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    frame_bbox = _find_frame(g)
    fr0, fc0, fr1, fc1 = frame_bbox

    slot_cells = [
        (r, c)
        for r in range(fr0, fr1 + 1)
        for c in range(fc0, fc1 + 1)
        if int(g[r, c]) == BG
    ]
    slot_bits = _bitmask_from_cells(slot_cells, frame_bbox)
    assert slot_bits != 0, "frame must contain slots"

    markers = _marker_positions(g)
    pieces = []
    piece_colors = set(int(v) for v in np.unique(g)) - {BG, GREY, MAROON}
    for comp in _find_components(g, piece_colors):
        pr0, pc0, pr1, pc1 = comp["bbox"]
        if not (pr1 < fr0 or pr0 > fr1 or pc1 < fc0 or pc0 > fc1):
            continue
        direction_name, direction, marker = _piece_direction(comp, markers)
        candidates = _candidate_placements(g, comp, direction, frame_bbox)
        assert candidates, f"piece moving {direction_name} has no slot candidate"
        comp = dict(comp)
        comp["direction"] = direction
        comp["marker"] = marker
        comp["candidates"] = candidates
        pieces.append(comp)

    assert pieces, "expected at least one colored piece"
    placements = _solve_assignments(slot_bits, pieces)

    out = g.copy()
    for piece in pieces:
        for r, c in piece["coords"]:
            out[r, c] = BG
        mr, mc = piece["marker"]
        out[mr, mc] = BG

    cover_count: dict[tuple[int, int], int] = {}
    for piece, placement in zip(pieces, placements):
        color = int(piece["color"])
        for cell in placement["cells"]:
            count = cover_count.get(cell, 0) + 1
            cover_count[cell] = count
            out[cell] = color if count == 1 else YELLOW

    return out


GRID_SIDE_RANGE = (26, 33)
FRAME_INNER_H_RANGE = (7, 12)
FRAME_INNER_W_RANGE = (7, 12)
FRAME_MARGIN = 6
PIECE_COUNT_RANGE = (2, 5)
AREA_RANGE = (3, 6)
MAX_MASK_SIDE = 4
MAX_ATTEMPTS = 400


def _random_mask() -> np.ndarray:
    for _ in range(100):
        h = random.randint(2, MAX_MASK_SIDE)
        w = random.randint(2, MAX_MASK_SIDE)
        max_area = min(h * w, AREA_RANGE[1])
        area = random.randint(AREA_RANGE[0], max_area)
        cells = {(random.randrange(h), random.randrange(w))}
        while len(cells) < area:
            r, c = random.choice(tuple(cells))
            dr, dc = random.choice(NEIGHBORS4)
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                cells.add((nr, nc))
        rows = [p[0] for p in cells]
        cols = [p[1] for p in cells]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in cells:
            mask[r - r0, c - c0] = True
        if mask.shape[0] >= 1 and mask.shape[1] >= 1 and int(mask.sum()) >= AREA_RANGE[0]:
            return mask
    raise RuntimeError("failed to build a connected mask")


def _mask_key(mask: np.ndarray) -> bytes:
    return mask.tobytes() + bytes([mask.shape[0], mask.shape[1]])


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[:-1, :] |= mask[1:, :]
    out[1:, :] |= mask[:-1, :]
    out[:, :-1] |= mask[:, 1:]
    out[:, 1:] |= mask[:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    return out


def _stamp_mask(dst: np.ndarray, mask: np.ndarray, top: int, left: int, value: int) -> None:
    h, w = mask.shape
    view = dst[top : top + h, left : left + w]
    view[mask] = value


def _place_overlap_pair(inner_h: int, inner_w: int, masks: list[np.ndarray]) -> list[tuple[int, int]] | None:
    mask_a, mask_b = masks[0], masks[1]
    ah, aw = mask_a.shape
    bh, bw = mask_b.shape
    positions_a = [
        (r, c)
        for r in range(inner_h - ah + 1)
        for c in range(inner_w - aw + 1)
    ]
    random.shuffle(positions_a)
    for ar, ac in positions_a:
        cells_a = set(_cells_from_mask(mask_a, ar, ac))
        positions_b = [
            (r, c)
            for r in range(inner_h - bh + 1)
            for c in range(inner_w - bw + 1)
        ]
        random.shuffle(positions_b)
        for br, bc in positions_b:
            cells_b = set(_cells_from_mask(mask_b, br, bc))
            overlap = len(cells_a & cells_b)
            if overlap not in (1, 2):
                continue
            if cells_a == cells_b:
                continue
            return [(ar, ac), (br, bc)]
    return None


def _nonoverlap_positions(
    inner_h: int,
    inner_w: int,
    placements: list[tuple[np.ndarray, tuple[int, int]]],
    mask: np.ndarray,
) -> tuple[int, int] | None:
    occupied = np.zeros((inner_h, inner_w), dtype=bool)
    for placed_mask, (top, left) in placements:
        _stamp_mask(occupied, placed_mask, top, left, True)
    blocked = _dilate8(occupied)
    h, w = mask.shape
    coords = [(r, c) for r in range(inner_h - h + 1) for c in range(inner_w - w + 1)]
    random.shuffle(coords)
    for top, left in coords:
        local = blocked[top : top + h, left : left + w]
        if not np.any(local[mask]):
            return top, left
    return None


def _make_expected(
    shape: tuple[int, int],
    frame_bbox: tuple[int, int, int, int],
    masks: list[np.ndarray],
    placements: list[tuple[int, int]],
    colors: list[int],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    fr0, fc0, fr1, fc1 = frame_bbox
    out[fr0 : fr1 + 1, fc0 : fc1 + 1] = GREY
    counts: dict[tuple[int, int], int] = {}
    for mask, (top, left), color in zip(masks, placements, colors):
        gtop = fr0 + 1 + top
        gleft = fc0 + 1 + left
        for cell in _cells_from_mask(mask, gtop, gleft):
            counts[cell] = counts.get(cell, 0) + 1
            out[cell] = int(color) if counts[cell] == 1 else YELLOW
    return out


def _build_frame_patch(inner_h: int, inner_w: int, masks: list[np.ndarray], placements: list[tuple[int, int]]) -> np.ndarray:
    patch = np.full((inner_h + 2, inner_w + 2), GREY, dtype=int)
    for mask, (top, left) in zip(masks, placements):
        _stamp_mask(patch[1:-1, 1:-1], mask, top, left, BG)
    return patch


def _marker_for_source(top: int, left: int, mask: np.ndarray, direction: str) -> tuple[int, int]:
    h, w = mask.shape
    if direction == "right":
        return top + h // 2, left + w
    if direction == "left":
        return top + h // 2, left - 1
    if direction == "down":
        return top + h, left + w // 2
    return top - 1, left + w // 2


def _cells_with_marker(top: int, left: int, mask: np.ndarray, marker: tuple[int, int]) -> set[tuple[int, int]]:
    return set(_cells_from_mask(mask, top, left)) | {marker}


def _can_use_source(cells: set[tuple[int, int]], occupied: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        for dr, dc in NEIGHBORS8 + ((0, 0),):
            nr, nc = r + dr, c + dc
            if (nr, nc) in occupied:
                return False
    return True


def _source_positions(
    shape: tuple[int, int],
    frame_bbox: tuple[int, int, int, int],
    masks: list[np.ndarray],
    placements: list[tuple[int, int]],
    directions: list[str],
) -> list[tuple[int, int, tuple[int, int]]] | None:
    h, w = shape
    fr0, fc0, _fr1, _fc1 = frame_bbox
    occupied: set[tuple[int, int]] = set()
    for r in range(frame_bbox[0], frame_bbox[2] + 1):
        for c in range(frame_bbox[1], frame_bbox[3] + 1):
            occupied.add((r, c))

    order = list(range(len(masks)))
    random.shuffle(order)
    result: list[tuple[int, int, tuple[int, int]] | None] = [None] * len(masks)
    for idx in order:
        mask = masks[idx]
        ph, pw = mask.shape
        top, left = placements[idx]
        gtop = fr0 + 1 + top
        gleft = fc0 + 1 + left
        direction = directions[idx]
        dr, dc = DIRS[direction]
        ds = list(range(2, max(h, w) + 1))
        random.shuffle(ds)
        placed = False
        for dist in ds:
            src_top = gtop - dr * dist
            src_left = gleft - dc * dist
            marker = _marker_for_source(src_top, src_left, mask, direction)
            cells = _cells_with_marker(src_top, src_left, mask, marker)
            if not _can_use_source(cells, occupied, shape):
                continue
            if direction == "right" and src_left + pw >= fc0:
                continue
            if direction == "left" and src_left - 1 <= frame_bbox[3]:
                continue
            if direction == "down" and src_top + ph >= fr0:
                continue
            if direction == "up" and src_top - 1 <= frame_bbox[2]:
                continue
            occupied |= cells
            result[idx] = (src_top, src_left, marker)
            placed = True
            break
        if not placed:
            return None
    assert all(item is not None for item in result)
    return [item for item in result if item is not None]


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    grid_h = random.randint(*GRID_SIDE_RANGE)
    grid_w = random.randint(*GRID_SIDE_RANGE)
    inner_h = random.randint(*FRAME_INNER_H_RANGE)
    inner_w = random.randint(*FRAME_INNER_W_RANGE)
    frame_h = inner_h + 2
    frame_w = inner_w + 2
    if grid_h < frame_h + 2 * FRAME_MARGIN or grid_w < frame_w + 2 * FRAME_MARGIN:
        return None

    r_start = FRAME_MARGIN
    r_end = grid_h - frame_h - FRAME_MARGIN
    c_start = max(grid_w // 2, FRAME_MARGIN)
    c_end = grid_w - frame_w - FRAME_MARGIN
    if r_start > r_end or c_start > c_end:
        return None

    frame_r0 = random.randint(r_start, r_end)
    frame_c0 = random.randint(c_start, c_end)
    frame_bbox = (frame_r0, frame_c0, frame_r0 + frame_h - 1, frame_c0 + frame_w - 1)

    piece_count = random.randint(*PIECE_COUNT_RANGE)
    want_overlap = piece_count >= 2 and random.random() < 0.5

    masks: list[np.ndarray] = []
    seen_shapes: set[bytes] = set()
    for _ in range(piece_count):
        for _retry in range(100):
            mask = _random_mask()
            key = _mask_key(mask)
            if key not in seen_shapes:
                seen_shapes.add(key)
                masks.append(mask)
                break
        else:
            return None

    placements: list[tuple[int, int]]
    if want_overlap:
        pair = _place_overlap_pair(inner_h, inner_w, masks[:2])
        if pair is None:
            return None
        placements = pair[:]
        used = [(masks[0], placements[0]), (masks[1], placements[1])]
        for mask in masks[2:]:
            pos = _nonoverlap_positions(inner_h, inner_w, used, mask)
            if pos is None:
                return None
            placements.append(pos)
            used.append((mask, pos))
    else:
        placements = []
        used: list[tuple[np.ndarray, tuple[int, int]]] = []
        for mask in masks:
            pos = _nonoverlap_positions(inner_h, inner_w, used, mask)
            if pos is None:
                return None
            placements.append(pos)
            used.append((mask, pos))

    colors = random.sample(PIECE_COLORS, piece_count)
    directions = [random.choice(tuple(DIRS)) for _ in range(piece_count)]
    sources = _source_positions((grid_h, grid_w), frame_bbox, masks, placements, directions)
    if sources is None:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    frame_patch = _build_frame_patch(inner_h, inner_w, masks, placements)
    grid[frame_r0 : frame_r0 + frame_h, frame_c0 : frame_c0 + frame_w] = frame_patch

    for mask, color, (src_top, src_left, marker) in zip(masks, colors, sources):
        _stamp_mask(grid, mask, src_top, src_left, int(color))
        grid[marker] = MAROON

    expected = _make_expected((grid_h, grid_w), frame_bbox, masks, placements, colors)

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    if want_overlap and not np.any(expected == YELLOW):
        return None
    if not want_overlap and np.any(expected == YELLOW):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        built = _generate_case()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid input")
