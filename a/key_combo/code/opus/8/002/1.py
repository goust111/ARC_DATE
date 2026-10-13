from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7

ANCHOR_COLOR = RED
MARKER_COLORS = [BLUE, GREEN, YELLOW, GREY, PINK]


def _anchor_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == ANCHOR_COLOR)
    assert len(coords) == 4, "expected exactly four anchor cells"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    assert int(r1) == int(r0) + 1 and int(c1) == int(c0) + 1, "anchor must be 2x2"
    assert np.all(grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == ANCHOR_COLOR)
    return int(r0), int(c0), int(r1), int(c1)


def _marker_map(grid: np.ndarray, anchor: tuple[int, int, int, int]) -> dict[tuple[int, int], int]:
    r0, c0, r1, c1 = anchor
    markers: dict[tuple[int, int], int] = {}
    for r, c in np.argwhere(grid != BG):
        rr, cc = int(r), int(c)
        if r0 <= rr <= r1 and c0 <= cc <= c1:
            continue
        markers[(rr, cc)] = int(grid[rr, cc])
    return markers


def _ray_cells(marker: tuple[int, int], anchor: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r, c = marker
    r0, c0, r1, c1 = anchor
    if r in (r0, r1):
        if c < c0:
            return [(r, cc) for cc in range(c + 1, c0)]
        if c > c1:
            return [(r, cc) for cc in range(c - 1, c1, -1)]
    if c in (c0, c1):
        if r < r0:
            return [(rr, c) for rr in range(r + 1, r0)]
        if r > r1:
            return [(rr, c) for rr in range(r - 1, r1, -1)]
    return []


def _rotate_point(
    r: int,
    c: int,
    center_r: float,
    center_c: float,
    quarter_turns_cw: int,
) -> tuple[int, int]:
    dr = r - center_r
    dc = c - center_c
    turns = quarter_turns_cw % 4
    if turns == 0:
        nr, nc = center_r + dr, center_c + dc
    elif turns == 1:
        nr, nc = center_r + dc, center_c - dr
    elif turns == 2:
        nr, nc = center_r - dr, center_c - dc
    else:
        nr, nc = center_r - dc, center_c + dr
    return int(round(nr)), int(round(nc))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate all anchor-aligned marker rays by a global quarter-turn count and recolor ray-marker overlaps orange.

    Input structure:
    - The grid has a black background, one red 2x2 anchor square, and several isolated singleton markers.
    - Marker colors come from the non-red foreground palette, and some markers share one anchor row or one anchor column.
    - Markers that do not align with an anchor row or column are distractors that stay fixed unless hit after rotation.

    Rule:
    - Find the red 2x2 anchor square and treat its geometric center as the rotation pivot.
    - For each aligned marker, draw the horizontal or vertical ray from that marker toward the anchor, stopping at the cell adjacent to the anchor and excluding the marker itself.
    - Let K be the number of aligned markers, rotate every ray cell clockwise by K * 90 degrees around the anchor center, and discard out-of-bounds rotated cells.
    - Keep the anchor and all original marker locations; paint rotated ray cells in their source colors, except any rotated ray that lands on an original marker location turns that location orange.

    Key insight:
    - The aligned markers serve two roles at once: they decide which rays exist and their total count controls the shared rotation angle applied to every ray.
    """
    g = np.asarray(grid, dtype=int)
    anchor = _anchor_bbox(g)
    r0, c0, r1, c1 = anchor
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0

    markers = _marker_map(g, anchor)
    anchor_rows = {r0, r1}
    anchor_cols = {c0, c1}

    aligned = [pos for pos in markers if pos[0] in anchor_rows or pos[1] in anchor_cols]
    ray_pixels: list[tuple[int, int, int]] = []
    for pos in aligned:
        color = markers[pos]
        for rr, cc in _ray_cells(pos, anchor):
            ray_pixels.append((rr, cc, color))

    turns = len(aligned) % 4
    out = g.copy()
    for rr, cc, color in ray_pixels:
        nr, nc = _rotate_point(rr, cc, center_r, center_c, turns)
        if not _in_bounds(g.shape, nr, nc):
            continue
        if r0 <= nr <= r1 and c0 <= nc <= c1:
            continue
        if (nr, nc) in markers:
            out[nr, nc] = ORANGE
        elif out[nr, nc] == BG:
            out[nr, nc] = color
    return out


def _chebyshev_clear(cell: tuple[int, int], others: set[tuple[int, int]]) -> bool:
    r, c = cell
    for orow, ocol in others:
        if max(abs(r - orow), abs(c - ocol)) <= 1:
            return False
    return True


def _min_anchor_distance(cell: tuple[int, int], anchor_cells: set[tuple[int, int]]) -> int:
    r, c = cell
    return min(abs(r - ar) + abs(c - ac) for ar, ac in anchor_cells)


def _slot_to_marker(
    slot: tuple[str, int],
    length: int,
    anchor: tuple[int, int, int, int],
    shape: tuple[int, int],
) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    r0, c0, r1, c1 = anchor
    h, w = shape
    side, lane = slot
    if side == "top":
        marker = (r0 - 1 - length, lane)
    elif side == "bottom":
        marker = (r1 + 1 + length, lane)
    elif side == "left":
        marker = (lane, c0 - 1 - length)
    else:
        marker = (lane, c1 + 1 + length)
    assert 0 <= marker[0] < h and 0 <= marker[1] < w
    return marker, _ray_cells(marker, anchor)


def _max_length(slot: tuple[str, int], anchor: tuple[int, int, int, int], shape: tuple[int, int]) -> int:
    r0, c0, r1, c1 = anchor
    h, w = shape
    side, _lane = slot
    if side == "top":
        return r0 - 1
    if side == "bottom":
        return h - r1 - 2
    if side == "left":
        return c0 - 1
    return w - c1 - 2


def _build_input(
    shape: tuple[int, int],
    anchor: tuple[int, int, int, int],
    aligned_slots: list[tuple[str, int]],
    total_markers: int,
    colors: list[int],
) -> np.ndarray | None:
    h, w = shape
    grid = np.zeros((h, w), dtype=int)
    r0, c0, r1, c1 = anchor
    grid[r0 : r1 + 1, c0 : c1 + 1] = ANCHOR_COLOR

    anchor_cells = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    anchor_rows = {r0, r1}
    anchor_cols = {c0, c1}

    markers: list[tuple[int, int, int]] = []
    used_cells: set[tuple[int, int]] = set(anchor_cells)
    ray_pixels: list[tuple[int, int, int]] = []
    for i, slot in enumerate(aligned_slots):
        max_len = min(5, _max_length(slot, anchor, shape))
        if max_len < 2:
            return None
        lengths = list(range(2, max_len + 1))
        random.shuffle(lengths)
        placed = False
        for length in lengths:
            marker, ray = _slot_to_marker(slot, length, anchor, shape)
            if marker in used_cells or not _chebyshev_clear(marker, {(r, c) for r, c, _ in markers}):
                continue
            if _min_anchor_distance(marker, anchor_cells) < 3:
                continue
            markers.append((marker[0], marker[1], colors[i]))
            used_cells.add(marker)
            for rr, cc in ray:
                ray_pixels.append((rr, cc, colors[i]))
            placed = True
            break
        if not placed:
            return None

    turns = len(aligned_slots) % 4
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    rotated: dict[tuple[int, int], int] = {}
    for rr, cc, color in ray_pixels:
        nr, nc = _rotate_point(rr, cc, center_r, center_c, turns)
        if not _in_bounds(shape, nr, nc):
            return None
        if (nr, nc) in anchor_cells:
            return None
        if (nr, nc) in rotated and rotated[(nr, nc)] != color:
            return None
        rotated[(nr, nc)] = color

    unaligned_total = total_markers - len(aligned_slots)
    existing_marker_cells = {(r, c) for r, c, _ in markers}
    overlap_candidates = [
        cell
        for cell in rotated
        if cell not in existing_marker_cells
        and cell[0] not in anchor_rows
        and cell[1] not in anchor_cols
        and _chebyshev_clear(cell, existing_marker_cells)
        and _min_anchor_distance(cell, anchor_cells) >= 3
    ]
    random.shuffle(overlap_candidates)

    overlap_count = 0
    if unaligned_total > 0 and overlap_candidates and random.random() < 0.8 and turns in (2, 3):
        overlap_count = random.randint(1, min(unaligned_total, len(overlap_candidates)))

    for j in range(overlap_count):
        r, c = overlap_candidates[j]
        color = colors[len(aligned_slots) + j]
        markers.append((r, c, color))
        existing_marker_cells.add((r, c))

    remaining = unaligned_total - overlap_count
    free_cells = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if (r, c) not in used_cells
        and (r, c) not in existing_marker_cells
        and r not in anchor_rows
        and c not in anchor_cols
        and _min_anchor_distance((r, c), anchor_cells) >= 3
        and (r, c) not in rotated
    ]
    random.shuffle(free_cells)
    for idx in range(remaining):
        choices = [cell for cell in free_cells if _chebyshev_clear(cell, existing_marker_cells)]
        if not choices:
            return None
        r, c = random.choice(choices)
        color = colors[len(aligned_slots) + overlap_count + idx]
        markers.append((r, c, color))
        existing_marker_cells.add((r, c))
        free_cells.remove((r, c))

    for r, c, color in markers:
        grid[r, c] = color

    if len(markers) != total_markers:
        return None
    if not np.array_equal(main(grid), main(grid.copy())):
        return None
    return grid


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(13, 18)
    w = random.randint(13, 18)
    r0 = random.randint(4, h - 6)
    c0 = random.randint(4, max(4, min(w - 6, w // 2)))
    anchor = (r0, c0, r0 + 1, c0 + 1)

    total_markers = random.choice([4, 5])
    k_options = [k for k in (2, 3, 4) if k < total_markers]
    if total_markers == 5:
        k_options.append(4)
    k = random.choice(sorted(set(k_options)))

    slots = [
        ("top", c0),
        ("top", c0 + 1),
        ("bottom", c0),
        ("bottom", c0 + 1),
        ("left", r0),
        ("left", r0 + 1),
        ("right", r0),
        ("right", r0 + 1),
    ]
    random.shuffle(slots)
    aligned_slots = slots[:k]
    colors = random.sample(MARKER_COLORS, total_markers)
    grid = _build_input((h, w), anchor, aligned_slots, total_markers, colors)
    if grid is None:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
