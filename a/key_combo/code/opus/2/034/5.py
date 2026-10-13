from __future__ import annotations

import itertools

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

GEN_GRID_MIN = 18
GEN_GRID_MAX = 25
GEN_GAP_CHOICES = (2, 3, 4)
GEN_RECT_MIN_SIDE = 5
GEN_RECT_MAX_SIDE = 11
GEN_RETRIES = 400
SUPPRESSION_PATTERNS = (
    ("tl",),
    ("tr",),
    ("bl",),
    ("br",),
    ("tl", "tr"),
    ("bl", "br"),
    ("tl", "bl"),
    ("tr", "br"),
)


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    """ARC toolbox convention: coords are (row, col) and bboxes are inclusive."""
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Inclusive bbox of a boolean mask, mirroring arc_utils.bbox()."""
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def bbox_h(box: tuple[int, int, int, int]) -> int:
    return int(box[2] - box[0] + 1)


def bbox_w(box: tuple[int, int, int, int]) -> int:
    return int(box[3] - box[1] + 1)


def _corner_specs(box: tuple[int, int, int, int]) -> dict[str, tuple[tuple[int, int], tuple[int, int]]]:
    r0, c0, r1, c1 = map(int, box)
    return {
        "tl": ((r0, c0), (-1, -1)),
        "tr": ((r0, c1), (-1, 1)),
        "bl": ((r1, c0), (1, -1)),
        "br": ((r1, c1), (1, 1)),
    }


def _box_contains(box: tuple[int, int, int, int], coord: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = box
    r, c = coord
    return r0 <= r <= r1 and c0 <= c <= c1


def _solid_rectangle_bbox(grid: np.ndarray, color: int) -> tuple[int, int, int, int]:
    mask = grid == int(color)
    box = bbox(mask)
    assert box is not None, "expected a solid rectangle"
    r0, c0, r1, c1 = box
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == int(color)), "rectangle must be solid"
    assert int(mask.sum()) == bbox_h(box) * bbox_w(box), "all green pixels must belong to the rectangle"
    return box


def _hollow_frame_bbox(grid: np.ndarray, color: int) -> tuple[int, int, int, int]:
    mask = grid == int(color)
    box = bbox(mask)
    assert box is not None, "expected a hollow frame"
    r0, c0, r1, c1 = box
    patch = mask[r0 : r1 + 1, c0 : c1 + 1]
    expected = np.zeros_like(patch, dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    assert np.array_equal(patch, expected), "blue pixels must form a one-cell-thick frame"
    return box


def _ray_hits_frame_corner(
    grid: np.ndarray,
    start: tuple[int, int],
    step: tuple[int, int],
    frame_corners: set[tuple[int, int]],
) -> tuple[int, int] | None:
    rr = int(start[0] + step[0])
    cc = int(start[1] + step[1])
    while in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) == BG:
        rr += step[0]
        cc += step[1]
    if not in_bounds(grid.shape, rr, cc):
        return None
    if int(grid[rr, cc]) != BLUE or (rr, cc) not in frame_corners:
        return None
    return rr, cc


def main(grid: np.ndarray) -> np.ndarray:
    """Highlight the green rectangle under the red-marker bbox and let only unsuppressed corners fire diagonal rays.

    Input structure:
    - The grid has black background, one solid green axis-aligned rectangle, one blue hollow rectangular frame, and several red single-pixel markers.
    - The green rectangle sits strictly inside the blue frame, and the red markers define a sparse bounding box window.

    Rule:
    - Compute the inclusive bbox of all red marker pixels and recolor every green rectangle pixel inside that bbox to yellow.
    - Inspect the four rectangle corners: yellow corners are suppressed, while remaining green corners trace the outward diagonal until the first non-black cell.
    - If that first non-black cell is a blue frame corner, continue one more step and paint every later black cell on the same diagonal green until the grid edge; otherwise do nothing.

    Key insight:
    - The same red-marker bbox has opposite roles at once: it highlights covered rectangle pixels but also disables any covered rectangle corner from launching its diagonal ray.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rect_box = _solid_rectangle_bbox(g, GREEN)
    frame_box = _hollow_frame_bbox(g, BLUE)
    marker_box = bbox(g == RED)
    assert marker_box is not None, "expected red marker pixels"

    fr0, fc0, fr1, fc1 = frame_box
    gr0, gc0, gr1, gc1 = rect_box
    assert fr0 < gr0 <= gr1 < fr1 and fc0 < gc0 <= gc1 < fc1, "rectangle must lie inside the frame"

    out = g.copy()
    mr0, mc0, mr1, mc1 = marker_box
    rr0 = max(mr0, gr0)
    cc0 = max(mc0, gc0)
    rr1 = min(mr1, gr1)
    cc1 = min(mc1, gc1)
    if rr0 <= rr1 and cc0 <= cc1:
        window = out[rr0 : rr1 + 1, cc0 : cc1 + 1]
        window[window == GREEN] = YELLOW

    frame_corners = {
        (fr0, fc0),
        (fr0, fc1),
        (fr1, fc0),
        (fr1, fc1),
    }
    for corner, step in _corner_specs(rect_box).values():
        r, c = corner
        if int(out[r, c]) != GREEN:
            continue
        hit = _ray_hits_frame_corner(g, corner, step, frame_corners)
        if hit is None:
            continue
        rr = int(hit[0] + step[0])
        cc = int(hit[1] + step[1])
        while in_bounds(g.shape, rr, cc):
            if int(g[rr, cc]) == BG:
                out[rr, cc] = GREEN
            rr += step[0]
            cc += step[1]

    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GEN_GRID_MIN, GEN_GRID_MAX + 1))
    w = int(rng.integers(GEN_GRID_MIN, GEN_GRID_MAX + 1))
    gap = int(rng.choice(GEN_GAP_CHOICES))

    size_options: list[tuple[int, int]] = []
    rect_h_max = min(GEN_RECT_MAX_SIDE, h - 2 * gap - 2)
    rect_w_max = min(GEN_RECT_MAX_SIDE, w - 2 * gap - 2)
    for rect_h in range(GEN_RECT_MIN_SIDE, rect_h_max + 1):
        for rect_w in range(GEN_RECT_MIN_SIDE, rect_w_max + 1):
            frame_h = rect_h + 2 * gap
            frame_w = rect_w + 2 * gap
            if rect_h == rect_w:
                continue
            if frame_h < 10 or frame_w < 10:
                continue
            if frame_h > h - 2 or frame_w > w - 2:
                continue
            size_options.append((rect_h, rect_w))
    if not size_options:
        return None

    rect_h, rect_w = size_options[int(rng.integers(len(size_options)))]
    frame_h = rect_h + 2 * gap
    frame_w = rect_w + 2 * gap

    fr0 = int(rng.integers(1, h - frame_h + 1))
    fc0 = int(rng.integers(1, w - frame_w + 1))
    fr1 = fr0 + frame_h - 1
    fc1 = fc0 + frame_w - 1
    gr0 = fr0 + gap
    gc0 = fc0 + gap
    gr1 = gr0 + rect_h - 1
    gc1 = gc0 + rect_w - 1

    grid = np.full((h, w), BG, dtype=int)
    grid[fr0 : fr1 + 1, fc0] = BLUE
    grid[fr0 : fr1 + 1, fc1] = BLUE
    grid[fr0, fc0 : fc1 + 1] = BLUE
    grid[fr1, fc0 : fc1 + 1] = BLUE
    grid[gr0 : gr1 + 1, gc0 : gc1 + 1] = GREEN

    suppressed = SUPPRESSION_PATTERNS[int(rng.integers(len(SUPPRESSION_PATTERNS)))]
    marker_box = _choose_marker_box(rng, grid, (gr0, gc0, gr1, gc1), (fr0, fc0, fr1, fc1), suppressed)
    if marker_box is None:
        return None

    _, markers = marker_box
    for r, c in markers:
        assert int(grid[r, c]) == BG
        grid[r, c] = RED

    if int(np.sum(grid == RED)) < 3 or int(np.sum(grid == RED)) > 5:
        return None

    if bbox(grid == RED) != marker_box[0]:
        return None

    out = main(grid)
    corners = _corner_specs((gr0, gc0, gr1, gc1))
    yellow_corners = [name for name, (coord, _step) in corners.items() if int(out[coord]) == YELLOW]
    if tuple(sorted(yellow_corners)) != tuple(sorted(suppressed)):
        return None
    if not np.any(out == YELLOW):
        return None
    if int(np.sum(out == GREEN)) <= int(np.sum(grid == GREEN)):
        return None
    if np.array_equal(out, grid):
        return None

    frame_corners = {(fr0, fc0), (fr0, fc1), (fr1, fc0), (fr1, fc1)}
    for name, (coord, step) in corners.items():
        hit = _ray_hits_frame_corner(grid, coord, step, frame_corners)
        if name in suppressed:
            continue
        if hit is None:
            return None
        rr = int(hit[0] + step[0])
        cc = int(hit[1] + step[1])
        if not in_bounds(grid.shape, rr, cc):
            return None

    return grid


def _choose_marker_box(
    rng: np.random.Generator,
    grid: np.ndarray,
    rect_box: tuple[int, int, int, int],
    frame_box: tuple[int, int, int, int],
    suppressed: tuple[str, ...],
) -> tuple[tuple[int, int, int, int], list[tuple[int, int]]] | None:
    gr0, gc0, gr1, gc1 = rect_box
    fr0, fc0, fr1, fc1 = frame_box
    corners = {name: coord for name, (coord, _step) in _corner_specs(rect_box).items()}
    target = tuple(sorted(suppressed))

    row_lo = max(fr0 + 1, gr0 - 2)
    row_hi = min(fr1 - 1, gr1 + 2)
    col_lo = max(fc0 + 1, gc0 - 2)
    col_hi = min(fc1 - 1, gc1 + 2)

    candidates: list[tuple[int, tuple[int, int, int, int], list[tuple[int, int]], list[tuple[int, int]]]] = []
    for mr0 in range(row_lo, row_hi + 1):
        for mr1 in range(mr0, row_hi + 1):
            for mc0 in range(col_lo, col_hi + 1):
                for mc1 in range(mc0, col_hi + 1):
                    box = (mr0, mc0, mr1, mc1)
                    inside = tuple(sorted(name for name, coord in corners.items() if _box_contains(box, coord)))
                    if inside != target:
                        continue

                    black_cells = [
                        (r, c)
                        for r in range(mr0, mr1 + 1)
                        for c in range(mc0, mc1 + 1)
                        if int(grid[r, c]) == BG
                    ]
                    if len(black_cells) < 3:
                        continue

                    cover = _boundary_cover(black_cells, box)
                    if cover is None:
                        continue

                    area = bbox_h(box) * bbox_w(box)
                    candidates.append((area, box, black_cells, cover))

    if not candidates:
        return None

    min_area = min(area for area, _box, _cells, _cover in candidates)
    slack = max(4, min_area // 2)
    pool = [item for item in candidates if item[0] <= min_area + slack]
    area, box, black_cells, cover = pool[int(rng.integers(len(pool)))]
    assert area >= min_area

    target_count = int(rng.integers(3, min(6, len(black_cells) + 1)))
    chosen = list(cover)
    remaining = [cell for cell in black_cells if cell not in chosen]
    if remaining:
        order = rng.permutation(len(remaining))
        for idx in order:
            if len(chosen) >= target_count:
                break
            chosen.append(remaining[int(idx)])

    chosen = chosen[:target_count]
    if len(chosen) < 3:
        return None
    if bbox(_mask_from_coords(grid.shape, chosen)) != box:
        return None
    return box, chosen


def _boundary_cover(
    black_cells: list[tuple[int, int]],
    box: tuple[int, int, int, int],
) -> list[tuple[int, int]] | None:
    mr0, mc0, mr1, mc1 = box
    boundary = [
        cell
        for cell in black_cells
        if cell[0] in (mr0, mr1) or cell[1] in (mc0, mc1)
    ]
    if not any(r == mr0 for r, _c in boundary):
        return None
    if not any(r == mr1 for r, _c in boundary):
        return None
    if not any(c == mc0 for _r, c in boundary):
        return None
    if not any(c == mc1 for _r, c in boundary):
        return None

    for size in range(3, min(5, len(boundary)) + 1):
        for combo in itertools.combinations(boundary, size):
            rows = [r for r, _c in combo]
            cols = [c for _r, c in combo]
            if min(rows) == mr0 and max(rows) == mr1 and min(cols) == mc0 and max(cols) == mc1:
                return list(combo)
    return None


def _mask_from_coords(shape: tuple[int, int], coords: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in coords:
        mask[int(r), int(c)] = True
    return mask
