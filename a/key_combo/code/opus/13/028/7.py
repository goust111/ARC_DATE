from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
MAGENTA = 6
ORANGE = 7
TEAL = 8

NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
NEIGHBORS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

GEN_ATTEMPTS = 400
SOLID_AREA_RANGE = (6, 11)
GAP_RANGE = (3, 7)
MARGIN_RANGE = (1, 4)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_from_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)

    return comps


def _green_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    return _components_from_mask(grid == GREEN)


def _holes_for_component(
    shape: tuple[int, int], coords: list[tuple[int, int]]
) -> list[list[tuple[int, int]]]:
    r0, c0, r1, c1 = _bbox(coords)
    sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        sub[r - r0, c - c0] = True

    holes: list[list[tuple[int, int]]] = []
    seen = np.zeros_like(sub, dtype=bool)
    for sr in range(sub.shape[0]):
        for sc in range(sub.shape[1]):
            if sub[sr, sc] or seen[sr, sc]:
                continue
            queue: deque[tuple[int, int]] = deque([(sr, sc)])
            seen[sr, sc] = True
            region: list[tuple[int, int]] = []
            touches_edge = False
            while queue:
                rr, cc = queue.popleft()
                region.append((rr, cc))
                if rr == 0 or cc == 0 or rr == sub.shape[0] - 1 or cc == sub.shape[1] - 1:
                    touches_edge = True
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < sub.shape[0] and 0 <= nc < sub.shape[1] and not sub[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            if not touches_edge:
                holes.append([(r0 + rr, c0 + cc) for rr, cc in region])
    return holes


def _paint_outline(out: np.ndarray, solid_coords: list[tuple[int, int]]) -> None:
    h, w = out.shape
    for r, c in solid_coords:
        for dr, dc in NEIGHBORS8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and out[nr, nc] == BLACK:
                out[nr, nc] = RED


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect the hollow green object, fill its hole, outline it, and color the gap according to whether it sits on the left or right.

    Input structure:
    - A black background contains exactly two green 4-connected objects separated horizontally.
    - One object is a solid irregular shape with no enclosed background hole.
    - The other object is hollow: its green pixels enclose one or more black cells inside its bounding box.

    Rule:
    - Find the two green connected components and detect which one contains enclosed 4-connected background holes inside its bbox.
    - For the hollow object, recolor every enclosed hole cell to magenta (6), then draw a red (2) 8-neighborhood outline around the hollow object together with the newly filled hole cells, changing only background cells.
    - Leave the solid object unchanged.
    - In the rectangular region strictly between the two bboxes and across their overlapping rows, fill remaining background cells with teal (8) if the hollow object is on the left, otherwise orange (7).

    Key insight:
    - Hole detection is the single decision point: it identifies which object receives the local fill-and-outline treatment and also determines the global gap color through the hollow object's left/right position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    comps = _green_components(g)
    assert len(comps) == 2, "expected exactly two green objects"

    comp_info = []
    for comp in comps:
        holes = _holes_for_component(g.shape, comp)
        comp_info.append({"coords": comp, "bbox": _bbox(comp), "holes": holes})

    hollow = [info for info in comp_info if info["holes"]]
    solid = [info for info in comp_info if not info["holes"]]
    assert len(hollow) == 1 and len(solid) == 1, "expected exactly one hollow and one solid object"

    hollow_info = hollow[0]
    solid_info = solid[0]

    out = g.copy()
    hole_cells = [cell for region in hollow_info["holes"] for cell in region]
    for r, c in hole_cells:
        out[r, c] = MAGENTA

    solidified = list(hollow_info["coords"]) + hole_cells
    _paint_outline(out, solidified)

    hb = hollow_info["bbox"]
    sb = solid_info["bbox"]
    row0 = max(hb[0], sb[0])
    row1 = min(hb[2], sb[2])
    assert row0 <= row1, "object row ranges must overlap"

    if hb[1] < sb[1]:
        gap_col0 = hb[3] + 1
        gap_col1 = sb[1] - 1
        gap_color = TEAL
    else:
        gap_col0 = sb[3] + 1
        gap_col1 = hb[1] - 1
        gap_color = ORANGE

    for r in range(row0, row1 + 1):
        for c in range(gap_col0, gap_col1 + 1):
            if out[r, c] == BLACK:
                out[r, c] = gap_color

    return out


def _cells_to_mask(cells: set[tuple[int, int]]) -> np.ndarray:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    shifted = {(r - min_r, c - min_c) for r, c in cells}
    h = max(r for r, _ in shifted) + 1
    w = max(c for _, c in shifted) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in shifted:
        mask[r, c] = True
    return mask


def _random_transform(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = mask.copy()
    if bool(rng.integers(0, 2)):
        out = np.flipud(out)
    if bool(rng.integers(0, 2)):
        out = np.fliplr(out)
    if bool(rng.integers(0, 2)):
        out = out.T
    return out.copy()


def _random_solid_shape(rng: np.random.Generator) -> np.ndarray | None:
    area = int(rng.integers(*SOLID_AREA_RANGE))
    cells: set[tuple[int, int]] = {(0, 0)}
    for _ in range(area - 1):
        frontier = sorted(
            {
                (r + dr, c + dc)
                for r, c in cells
                for dr, dc in NEIGHBORS4
                if (r + dr, c + dc) not in cells
            }
        )
        if not frontier:
            return None
        pick = frontier[int(rng.integers(0, len(frontier)))]
        cells.add(pick)

    mask = _cells_to_mask(cells)
    mask = _random_transform(mask, rng)
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    if len(coords) != area:
        return None
    if len(_holes_for_component(mask.shape, coords)) != 0:
        return None
    box = _bbox(coords)
    if box[0] == box[2] or box[1] == box[3]:
        return None
    return mask


def _add_rect(cells: set[tuple[int, int]], r0: int, c0: int, h: int, w: int) -> None:
    for r in range(r0, r0 + h):
        for c in range(c0, c0 + w):
            cells.add((r, c))


def _random_hollow_shape(rng: np.random.Generator) -> np.ndarray | None:
    outer_h = int(rng.integers(3, 6))
    outer_w = int(rng.integers(3, 6))
    hole_h = int(rng.integers(1, min(3, outer_h - 1)))
    hole_w = int(rng.integers(1, min(3, outer_w - 1)))
    if outer_h - hole_h < 2 or outer_w - hole_w < 2:
        return None
    hole_r = int(rng.integers(1, outer_h - hole_h))
    hole_c = int(rng.integers(1, outer_w - hole_w))
    if hole_r + hole_h >= outer_h or hole_c + hole_w >= outer_w:
        return None

    cells = {
        (r, c)
        for r in range(outer_h)
        for c in range(outer_w)
        if not (hole_r <= r < hole_r + hole_h and hole_c <= c < hole_c + hole_w)
    }

    protrusions = int(rng.integers(0, 3))
    for _ in range(protrusions):
        side = int(rng.integers(0, 4))
        span = int(rng.integers(1, 3))
        length = int(rng.integers(1, 3))
        if side == 0:
            start = int(rng.integers(0, outer_w - span + 1))
            _add_rect(cells, -length, start, length, span)
        elif side == 1:
            start = int(rng.integers(0, outer_w - span + 1))
            _add_rect(cells, outer_h, start, length, span)
        elif side == 2:
            start = int(rng.integers(0, outer_h - span + 1))
            _add_rect(cells, start, -length, span, length)
        else:
            start = int(rng.integers(0, outer_h - span + 1))
            _add_rect(cells, start, outer_w, span, length)

    mask = _cells_to_mask(cells)
    mask = _random_transform(mask, rng)
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    holes = _holes_for_component(mask.shape, coords)
    hole_pixels = sum(len(region) for region in holes)
    if hole_pixels < 1 or hole_pixels > 4:
        return None
    if len(holes) != 1:
        return None
    if len(coords) < 8 or len(coords) > 18:
        return None
    return mask


def _row_overlap(
    r0_a: int, h_a: int, r0_b: int, h_b: int
) -> int:
    return max(0, min(r0_a + h_a - 1, r0_b + h_b - 1) - max(r0_a, r0_b) + 1)


def _place_masks(
    hollow_mask: np.ndarray, solid_mask: np.ndarray, hollow_left: bool, rng: np.random.Generator
) -> np.ndarray | None:
    hh, hw = hollow_mask.shape
    sh, sw = solid_mask.shape
    gap = int(rng.integers(*GAP_RANGE))
    left_margin = int(rng.integers(*MARGIN_RANGE))
    right_margin = int(rng.integers(*MARGIN_RANGE))
    top_margin = int(rng.integers(1, 4))
    bottom_margin = int(rng.integers(1, 4))

    height = max(hh, sh) + top_margin + bottom_margin + int(rng.integers(0, 4))
    height = min(height, 20)
    if height < max(hh, sh) + 2:
        return None

    hollow_rows = list(range(1, height - hh))
    solid_rows = list(range(1, height - sh))
    if not hollow_rows or not solid_rows:
        return None

    candidate_pairs = [
        (rh, rs)
        for rh in hollow_rows
        for rs in solid_rows
        if _row_overlap(rh, hh, rs, sh) >= 1
    ]
    if not candidate_pairs:
        return None
    rh, rs = candidate_pairs[int(rng.integers(0, len(candidate_pairs)))]

    if hollow_left:
        ch = left_margin
        cs = ch + hw + gap
    else:
        cs = left_margin
        ch = cs + sw + gap

    width = max(ch + hw, cs + sw) + right_margin
    width = min(width, 24)
    if width < max(ch + hw, cs + sw) + 1:
        return None

    grid = np.zeros((height, width), dtype=int)
    grid[rh : rh + hh, ch : ch + hw][hollow_mask] = GREEN
    grid[rs : rs + sh, cs : cs + sw][solid_mask] = GREEN
    return grid


def _validate_input(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if not set(map(int, np.unique(grid))).issubset({BLACK, GREEN}):
        return False

    comps = _green_components(grid)
    if len(comps) != 2:
        return False

    info = []
    for comp in comps:
        holes = _holes_for_component(grid.shape, comp)
        info.append({"coords": comp, "bbox": _bbox(comp), "holes": holes})

    hollow = [entry for entry in info if entry["holes"]]
    solid = [entry for entry in info if not entry["holes"]]
    if len(hollow) != 1 or len(solid) != 1:
        return False

    hole_pixels = sum(len(region) for region in hollow[0]["holes"])
    if not (1 <= hole_pixels <= 4):
        return False

    hb = hollow[0]["bbox"]
    sb = solid[0]["bbox"]
    if min(hb[2], sb[2]) < max(hb[0], sb[0]):
        return False
    if hb[1] < sb[1]:
        gap = sb[1] - hb[3] - 1
        expected_gap_color = TEAL
        unexpected_gap_color = ORANGE
    else:
        gap = hb[1] - sb[3] - 1
        expected_gap_color = ORANGE
        unexpected_gap_color = TEAL
    if gap < 2:
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    if not np.any(out == MAGENTA):
        return False
    if not np.any(out == RED):
        return False
    if not np.any(out == expected_gap_color):
        return False
    if np.any(out == unexpected_gap_color):
        return False
    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        hollow_mask = _random_hollow_shape(rng)
        solid_mask = _random_solid_shape(rng)
        if hollow_mask is None or solid_mask is None:
            continue

        hollow_left = bool(rng.integers(0, 2))
        grid = _place_masks(hollow_mask, solid_mask, hollow_left, rng)
        if grid is None:
            continue
        if _validate_input(grid):
            return grid

    raise RuntimeError("failed to generate a valid input")
