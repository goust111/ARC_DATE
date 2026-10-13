from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GRID_MIN = 17
GRID_MAX = 23
MARGIN = 3
ANCHOR_COUNTS = (3, 4)
MIN_GAP = 6
PAIR_DELTAS = (6, 8, 10)
GEN_ATTEMPTS = 400

ARM_A = ((-1, 0), (0, 1))
ARM_B = ((1, 0), (0, -1))

Anchor = tuple[int, int, int]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    """Mirror arc_utils.in_bounds with the canonical (r, c) convention."""
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _sort_anchors(anchors: list[Anchor]) -> list[Anchor]:
    return sorted(anchors, key=lambda anchor: (anchor[0], anchor[1]))


def _extract_anchors(grid: np.ndarray) -> list[Anchor]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    coords = np.argwhere(g != BG)
    assert 3 <= len(coords) <= 4, "expected 3-4 colored anchor pixels"

    anchors: list[Anchor] = []
    seen_colors: set[int] = set()
    h, w = map(int, g.shape)
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        color = int(g[rr, cc])
        assert color != BG
        assert color not in seen_colors, "expected distinct anchor colors"
        seen_colors.add(color)

        r0 = max(0, rr - 1)
        r1 = min(h, rr + 2)
        c0 = max(0, cc - 1)
        c1 = min(w, cc + 2)
        assert np.count_nonzero(g[r0:r1, c0:c1] != BG) == 1, "anchors must be isolated"
        anchors.append((rr, cc, color))

    return _sort_anchors(anchors)


def _simulate_from_anchors(
    shape: tuple[int, int],
    anchors: list[Anchor],
    *,
    order: list[Anchor] | None = None,
    record: bool = False,
) -> tuple[np.ndarray, list[dict]]:
    out = np.zeros(shape, dtype=int)
    for r, c, color in anchors:
        out[int(r), int(c)] = int(color)

    ordered = _sort_anchors(anchors) if order is None else list(order)
    records: list[dict] = []
    for order_index, (r, c, color) in enumerate(ordered):
        for arm_name, directions in (("A", ARM_A), ("B", ARM_B)):
            head_r = int(r)
            head_c = int(c)
            segments_drawn = 0

            while True:
                dr, dc = directions[segments_drawn % 2]
                cells = [
                    (head_r + dr, head_c + dc),
                    (head_r + 2 * dr, head_c + 2 * dc),
                ]

                if any(not _in_bounds(shape, rr, cc) for rr, cc in cells):
                    if record:
                        records.append(
                            {
                                "anchor": (int(r), int(c), int(color)),
                                "order_index": int(order_index),
                                "arm": arm_name,
                                "segments_drawn": int(segments_drawn),
                                "stop_reason": "boundary",
                                "stop_cells": tuple(cells),
                                "blockers": (),
                            }
                        )
                    break

                blockers = tuple(
                    (int(rr), int(cc), int(out[rr, cc]))
                    for rr, cc in cells
                    if int(out[rr, cc]) != BG
                )
                if blockers:
                    if record:
                        records.append(
                            {
                                "anchor": (int(r), int(c), int(color)),
                                "order_index": int(order_index),
                                "arm": arm_name,
                                "segments_drawn": int(segments_drawn),
                                "stop_reason": "occupied",
                                "stop_cells": tuple(cells),
                                "blockers": blockers,
                            }
                        )
                    break

                for rr, cc in cells:
                    out[rr, cc] = int(color)
                head_r, head_c = cells[-1]
                segments_drawn += 1

    return out, records


def _free_stair_cells(shape: tuple[int, int], anchor: Anchor) -> set[tuple[int, int]]:
    out, _records = _simulate_from_anchors(shape, [anchor], record=False)
    cells = {tuple(map(int, coord)) for coord in np.argwhere(out != BG)}
    cells.discard((int(anchor[0]), int(anchor[1])))
    return cells


def _render_input(shape: tuple[int, int], anchors: list[Anchor]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for r, c, color in anchors:
        grid[int(r), int(c)] = int(color)
    return grid


def _well_spaced(pos: tuple[int, int], anchors: list[Anchor]) -> bool:
    r, c = map(int, pos)
    for ar, ac, _color in anchors:
        if max(abs(r - int(ar)), abs(c - int(ac))) < MIN_GAP:
            return False
    return True


def _region_candidates(
    shape: tuple[int, int],
    anchors: list[Anchor],
    *,
    row_range: range,
    col_range: range,
) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for r in row_range:
        for c in col_range:
            if _well_spaced((r, c), anchors):
                candidates.append((int(r), int(c)))
    random.shuffle(candidates)
    return candidates


def _add_extra_anchors(shape: tuple[int, int], anchors: list[Anchor], total_count: int) -> list[Anchor] | None:
    h, w = map(int, shape)
    need = int(total_count - len(anchors))
    if need <= 0:
        return list(anchors)

    mid_r = h // 2
    mid_c = w // 2
    regions = [
        (range(MARGIN, mid_r), range(MARGIN, mid_c)),
        (range(mid_r, h - MARGIN), range(mid_c, w - MARGIN)),
    ]
    random.shuffle(regions)

    result = list(anchors)
    for row_range, col_range in regions:
        if len(result) >= total_count:
            break
        candidates = _region_candidates(shape, result, row_range=row_range, col_range=col_range)
        if candidates:
            r, c = candidates[0]
            result.append((r, c, -1))

    if len(result) < total_count:
        global_candidates = _region_candidates(
            shape,
            result,
            row_range=range(MARGIN, h - MARGIN),
            col_range=range(MARGIN, w - MARGIN),
        )
        for r, c in global_candidates:
            if len(result) >= total_count:
                break
            result.append((r, c, -1))

    if len(result) != total_count:
        return None
    return result


def _pair_positions(shape: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    h, w = map(int, shape)
    deltas = [
        delta
        for delta in PAIR_DELTAS
        if MARGIN + delta <= h - 1 - MARGIN and MARGIN + delta <= w - 1 - MARGIN
    ]
    if not deltas:
        return None

    delta = random.choice(deltas)
    lower_left_r = random.randint(MARGIN + delta, h - 1 - MARGIN)
    lower_left_c = random.randint(MARGIN, w - 1 - MARGIN - delta)
    upper_right = (lower_left_r - delta, lower_left_c + delta)
    lower_left = (lower_left_r, lower_left_c)
    return upper_right, lower_left


def _semantic_summary(grid: np.ndarray) -> dict:
    anchors = _extract_anchors(grid)
    shape = tuple(map(int, grid.shape))
    anchor_positions = {(r, c) for r, c, _color in anchors}

    free_sets = [_free_stair_cells(shape, anchor) for anchor in anchors]
    free_pairs = 0
    for i in range(len(free_sets)):
        for j in range(i + 1, len(free_sets)):
            if free_sets[i] & free_sets[j]:
                free_pairs += 1

    out, records = _simulate_from_anchors(shape, anchors, record=True)
    reverse_out, _ = _simulate_from_anchors(shape, anchors, order=list(reversed(anchors)), record=False)

    occupied_by_stair = any(
        rec["stop_reason"] == "occupied"
        and any((rr, cc) not in anchor_positions for rr, cc, _color in rec["blockers"])
        for rec in records
    )
    occupied_any = any(rec["stop_reason"] == "occupied" for rec in records)
    boundary_any = any(rec["stop_reason"] == "boundary" for rec in records)
    multi_step_any = any(rec["segments_drawn"] >= 3 for rec in records)

    return {
        "anchors": anchors,
        "out": out,
        "records": records,
        "free_pairs": int(free_pairs),
        "occupied_any": bool(occupied_any),
        "occupied_by_stair": bool(occupied_by_stair),
        "boundary_any": bool(boundary_any),
        "multi_step_any": bool(multi_step_any),
        "reverse_diff": not np.array_equal(out, reverse_out),
    }


def _semantic_ok(grid: np.ndarray) -> bool:
    try:
        info = _semantic_summary(grid)
    except Exception:
        return False

    return bool(
        info["free_pairs"] >= 1
        and info["occupied_by_stair"]
        and info["occupied_any"]
        and info["boundary_any"]
        and info["multi_step_any"]
        and info["reverse_diff"]
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow colored two-arm staircases from each anchor, truncating later stairs when they hit earlier stair territory.

    Input structure:
    - The grid is a black background containing 3 or 4 isolated colored anchor pixels.
    - Every anchor is a single non-black cell with a unique color.
    - Anchors are separated so each one has room to begin both stair arms.

    Rule:
    - Process anchors in reading order: top to bottom, then left to right.
    - For each anchor, grow arm A by repeating up-2 then right-2 segments, and grow arm B by repeating down-2 then left-2 segments.
    - Before drawing any 2-cell segment, test both cells together; if either cell is out of bounds or already non-black, skip that whole segment and stop only that arm.
    - Draw successful stair cells in the anchor's own color and keep the original anchor pixels unchanged.

    Key insight:
    - Each anchor would form a point-symmetric pair of stair arms on its own, but row-major processing lets earlier anchors claim cells that later anchors must treat as blocking territory.
    """
    g = np.asarray(grid, dtype=int)
    anchors = _extract_anchors(g)
    out, _records = _simulate_from_anchors(tuple(map(int, g.shape)), anchors, record=False)
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    shape = (h, w)

    base_pair = _pair_positions(shape)
    if base_pair is None:
        return None

    anchor_count = random.choice(ANCHOR_COUNTS)
    colors = random.sample(PALETTE, anchor_count)
    anchors: list[Anchor] = [
        (int(base_pair[0][0]), int(base_pair[0][1]), int(colors[0])),
        (int(base_pair[1][0]), int(base_pair[1][1]), int(colors[1])),
    ]

    with_placeholders = _add_extra_anchors(shape, anchors, anchor_count)
    if with_placeholders is None:
        return None

    final_anchors: list[Anchor] = anchors[:]
    extra_index = 2
    for r, c, color in with_placeholders[2:]:
        final_anchors.append((int(r), int(c), int(colors[extra_index])))
        extra_index += 1

    grid = _render_input(shape, final_anchors)
    try:
        info = _semantic_summary(grid)
    except Exception:
        return None
    if not (
        info["free_pairs"] >= 1
        and info["occupied_by_stair"]
        and info["occupied_any"]
        and info["boundary_any"]
        and info["multi_step_any"]
        and info["reverse_diff"]
    ):
        return None
    if not np.array_equal(main(grid), info["out"]):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid multi-anchor staircase collision input")
