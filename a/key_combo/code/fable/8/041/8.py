from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5
ORANGE = 7

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 500


def _crop_bool(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size, "expected a non-empty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _flood_background(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    exterior = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if not mask[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not mask[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in DIRS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                q.append((nr, nc))

    return exterior


def _render_union(mask: np.ndarray) -> np.ndarray:
    cropped = _crop_bool(mask)
    exterior = _flood_background(cropped)
    holes = (~cropped) & (~exterior)

    out = np.full(cropped.shape, BLACK, dtype=int)
    out[cropped] = BLUE
    out[holes] = YELLOW
    return out


def _extract_relative_mask(panel: np.ndarray) -> set[tuple[int, int]]:
    anchors = np.argwhere(panel == RED)
    assert len(anchors) == 1, "expected exactly one red anchor in each panel"
    ar, ac = map(int, anchors[0])
    orange = np.argwhere(panel == ORANGE)
    assert orange.size, "expected orange mask pixels in each panel"
    return {(int(r) - ar, int(c) - ac) for r, c in orange}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align the two orange masks by their red anchors, merge them, and fill only the holes that appear after the merge.

    Input structure:
    - A full-height grey separator column splits the grid into two equal black/orange mask panels.
    - Each panel contains exactly one red anchor pixel; the anchor itself is not part of the orange mask.
    - Blue and yellow do not appear in the input.

    Rule:
    - Split the grid at the grey separator column.
    - Translate the two panel masks so that their red anchors coincide.
    - Take the cellwise union of the two orange masks and recolor that union blue.
    - In the cropped union, recolor every black region that is fully enclosed by blue and unreachable from the crop boundary to yellow.

    Key insight:
    - The panel borders are a decoy: the correct overlay uses the red anchors as the alignment reference, and only that aligned union creates the enclosed holes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    separator_cols = np.flatnonzero(np.all(g == GREY, axis=0))
    assert len(separator_cols) == 1, "expected one full-height grey separator column"
    sep = int(separator_cols[0])

    left = g[:, :sep]
    right = g[:, sep + 1 :]
    assert left.shape == right.shape and left.size > 0, "expected two equal non-empty panels"
    assert set(map(int, np.unique(left))) <= {BLACK, RED, ORANGE}
    assert set(map(int, np.unique(right))) <= {BLACK, RED, ORANGE}

    rel_left = _extract_relative_mask(left)
    rel_right = _extract_relative_mask(right)
    union = rel_left | rel_right
    rel = np.array(list(union), dtype=int)
    r0, c0 = rel.min(axis=0)
    r1, c1 = rel.max(axis=0)

    mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    for r, c in union:
        mask[r - int(r0), c - int(c0)] = True

    return _render_union(mask)


def _rect_perimeter(top: int, left: int, height: int, width: int) -> set[tuple[int, int]]:
    bottom = top + height - 1
    right = left + width - 1
    cells = set()
    for c in range(left, right + 1):
        cells.add((top, c))
        cells.add((bottom, c))
    for r in range(top, bottom + 1):
        cells.add((r, left))
        cells.add((r, right))
    return cells


def _bar_cells(side: str, top: int, left: int, bottom: int, right: int, margin: int) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    if side == "left":
        row = random.randint(top + 1, bottom - 1)
        length = random.randint(1, min(2, left))
        return ({(row, left - d) for d in range(1, length + 1)}, (row, left))
    if side == "right":
        row = random.randint(top + 1, bottom - 1)
        length = random.randint(1, min(2, margin))
        return ({(row, right + d) for d in range(1, length + 1)}, (row, right))
    if side == "top":
        col = random.randint(left + 1, right - 1)
        length = random.randint(1, min(2, top))
        return ({(top - d, col) for d in range(1, length + 1)}, (top, col))
    col = random.randint(left + 1, right - 1)
    length = random.randint(1, min(2, margin))
    return ({(bottom + d, col) for d in range(1, length + 1)}, (bottom, col))


def _gap_cells(side: str, top: int, left: int, bottom: int, right: int, forbidden: tuple[int, int] | None) -> set[tuple[int, int]] | None:
    if side in {"top", "bottom"}:
        fixed_r = top if side == "top" else bottom
        line = [(fixed_r, c) for c in range(left + 1, right)]
    else:
        fixed_c = left if side == "left" else right
        line = [(r, fixed_c) for r in range(top + 1, bottom)]

    lengths = [2, 1]
    random.shuffle(lengths)
    for length in lengths:
        if len(line) < length:
            continue
        windows = [set(line[start : start + length]) for start in range(len(line) - length + 1)]
        random.shuffle(windows)
        for segment in windows:
            if forbidden is not None and forbidden in segment:
                continue
            return segment
    return None


def _hole_count(mask: np.ndarray) -> int:
    cropped = _crop_bool(mask)
    exterior = _flood_background(cropped)
    holes = (~cropped) & (~exterior)
    h, w = holes.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0

    for r in range(h):
        for c in range(w):
            if not holes[r, c] or seen[r, c]:
                continue
            count += 1
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            while q:
                cr, cc = q.popleft()
                for dr, dc in DIRS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and holes[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
    return count


def _coords_to_mask(coords: set[tuple[int, int]]) -> np.ndarray:
    pts = np.array(list(coords), dtype=int)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    for r, c in coords:
        mask[r - int(r0), c - int(c0)] = True
    return mask


def _anchor_candidates(
    top: int, left: int, bottom: int, right: int, occupied: set[tuple[int, int]], shape: tuple[int, int]
) -> list[tuple[int, int]]:
    h, w = shape
    candidates: list[tuple[int, int]] = []
    for r in range(top + 1, bottom):
        for c in (left - 1, right + 1):
            if 0 <= r < h and 0 <= c < w and (r, c) not in occupied:
                candidates.append((r, c))
    for c in range(left + 1, right):
        for r in (top - 1, bottom + 1):
            if 0 <= r < h and 0 <= c < w and (r, c) not in occupied:
                candidates.append((r, c))
    random.shuffle(candidates)
    return candidates


def _relative_coords(coords: set[tuple[int, int]], anchor: tuple[int, int]) -> set[tuple[int, int]]:
    ar, ac = anchor
    return {(r - ar, c - ac) for r, c in coords}


def _span_bounds(rel: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = np.array(list(rel), dtype=int)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    return int(mins[0]), int(maxs[0]), int(mins[1]), int(maxs[1])


def _choose_anchor_position(
    rel: set[tuple[int, int]], panel_h: int, panel_w: int
) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    min_r, max_r, min_c, max_c = _span_bounds(rel)
    row_lo = max(0, -min_r)
    row_hi = min(panel_h - 1, panel_h - 1 - max_r)
    col_lo = max(0, -min_c)
    col_hi = min(panel_w - 1, panel_w - 1 - max_c)
    rows = list(range(row_lo, row_hi + 1))
    cols = list(range(col_lo, col_hi + 1))
    positions = [(r, c) for r in rows for c in cols]
    return random.choice(positions), positions


def _panel_from_relative(rel: set[tuple[int, int]], anchor_pos: tuple[int, int], shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    panel = np.full((h, w), BLACK, dtype=int)
    ar, ac = anchor_pos
    for dr, dc in rel:
        panel[ar + dr, ac + dc] = ORANGE
    panel[ar, ac] = RED
    return panel


def _or_top_left(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    h = max(mask_a.shape[0], mask_b.shape[0])
    w = max(mask_a.shape[1], mask_b.shape[1])
    merged = np.zeros((h, w), dtype=bool)
    merged[: mask_a.shape[0], : mask_a.shape[1]] |= mask_a
    merged[: mask_b.shape[0], : mask_b.shape[1]] |= mask_b
    return merged


def _border_aligned_output(left_panel: np.ndarray, right_panel: np.ndarray) -> np.ndarray:
    union = (left_panel == ORANGE) | (right_panel == ORANGE)
    return _render_union(union)


def _bbox_aligned_output(left_panel: np.ndarray, right_panel: np.ndarray) -> np.ndarray:
    left_crop = _crop_bool(left_panel == ORANGE)
    right_crop = _crop_bool(right_panel == ORANGE)
    return _render_union(_or_top_left(left_crop, right_crop))


def _single_panel_output(panel: np.ndarray) -> np.ndarray:
    return _render_union(panel == ORANGE)


def _build_mask_pair() -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    outer_h = random.randint(5, 6)
    outer_w = random.randint(5, 7)
    top_margin = random.randint(2, 3)
    bottom_margin = random.randint(2, 3)
    left_margin = random.randint(2, 3)
    right_margin = random.randint(2, 3)

    top = top_margin
    left = left_margin
    bottom = top + outer_h - 1
    right = left + outer_w - 1
    work_h = bottom + bottom_margin + 1
    work_w = right + right_margin + 1

    ring = _rect_perimeter(top, left, outer_h, outer_w)

    if random.random() < 0.5:
        only1, root1 = _bar_cells("left", top, left, bottom, right, right_margin)
        only2, root2 = _bar_cells("right", top, left, bottom, right, right_margin)
    else:
        only1, root1 = _bar_cells("top", top, left, bottom, right, bottom_margin)
        only2, root2 = _bar_cells("bottom", top, left, bottom, right, bottom_margin)

    gap1 = _gap_cells(random.choice(["top", "bottom", "left", "right"]), top, left, bottom, right, root1)
    if gap1 is None:
        return None

    remaining_sides = ["top", "bottom", "left", "right"]
    random.shuffle(remaining_sides)
    gap2 = None
    for side in remaining_sides:
        gap2 = _gap_cells(side, top, left, bottom, right, root2)
        if gap2 is not None and gap2 != gap1:
            break
    if gap2 is None or gap2 == gap1:
        return None

    mask1 = (ring - gap1) | only1
    mask2 = (ring - gap2) | only2
    union = ring | only1 | only2

    anchors = _anchor_candidates(top, left, bottom, right, union, (work_h, work_w))
    if not anchors:
        return None
    anchor = random.choice(anchors)

    rel1 = _relative_coords(mask1, anchor)
    rel2 = _relative_coords(mask2, anchor)

    if len(rel1 - rel2) == 0 or len(rel2 - rel1) == 0:
        return None
    if _hole_count(_coords_to_mask(rel1)) != 0:
        return None
    if _hole_count(_coords_to_mask(rel2)) != 0:
        return None
    if _hole_count(_coords_to_mask(rel1 | rel2)) == 0:
        return None

    return rel1, rel2


def _gen_attempt() -> np.ndarray | None:
    built = _build_mask_pair()
    if built is None:
        return None
    rel1, rel2 = built

    span1_h = _span_bounds(rel1)[1] - _span_bounds(rel1)[0] + 1
    span1_w = _span_bounds(rel1)[3] - _span_bounds(rel1)[2] + 1
    span2_h = _span_bounds(rel2)[1] - _span_bounds(rel2)[0] + 1
    span2_w = _span_bounds(rel2)[3] - _span_bounds(rel2)[2] + 1
    min_h = max(span1_h, span2_h)
    min_w = max(span1_w, span2_w)

    if min_h > 12 or min_w > 12:
        return None

    panel_h = random.randint(min_h + 1, min(14, min_h + 3))
    panel_w = random.randint(min_w + 1, min(14, min_w + 3))

    anchor1, choices1 = _choose_anchor_position(rel1, panel_h, panel_w)
    anchor2, choices2 = _choose_anchor_position(rel2, panel_h, panel_w)

    tries = 0
    while (anchor1 == anchor2 or anchor1[0] == anchor2[0] or anchor1[1] == anchor2[1]) and tries < 20:
        anchor2 = random.choice(choices2)
        tries += 1
    if anchor1 == anchor2:
        return None

    left_panel = _panel_from_relative(rel1, anchor1, (panel_h, panel_w))
    right_panel = _panel_from_relative(rel2, anchor2, (panel_h, panel_w))
    separator = np.full((panel_h, 1), GREY, dtype=int)
    inp = np.concatenate([left_panel, separator, right_panel], axis=1)

    try:
        out = main(inp)
    except Exception:
        return None

    if not np.any(out == YELLOW):
        return None
    if np.array_equal(out, _single_panel_output(left_panel)):
        return None
    if np.array_equal(out, _single_panel_output(right_panel)):
        return None
    if np.array_equal(out, _border_aligned_output(left_panel, right_panel)):
        return None
    if np.array_equal(out, _bbox_aligned_output(left_panel, right_panel)):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid anchor-aligned hole-merging example")
