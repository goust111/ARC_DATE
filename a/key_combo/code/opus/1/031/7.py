from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

BG = 0
GEN_HEIGHT_RANGE = (5, 8)
GEN_WIDTH_RANGE = (5, 8)
GEN_MAIN_AREA_RANGE = (6, 10)
GEN_ATTEMPTS = 400
GEN_MODES = [
    {"name": "top_right", "require": {"top", "right"}, "forbid": {"bottom"}, "components": 2, "weight": 3},
    {"name": "bottom_left", "require": {"bottom", "left"}, "forbid": {"right"}, "components": 2, "weight": 3},
    {"name": "bottom_right", "require": {"bottom", "right"}, "forbid": set(), "components": 1, "weight": 4},
]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _most_common_non_bg(g: np.ndarray, bg: int) -> int:
    counts = Counter(map(int, g.ravel()))
    colors = [color for color in counts if color != bg]
    assert len(colors) >= 2, "expected a main color and a marker color"
    area = max(counts[color] for color in colors)
    matches = [color for color in colors if counts[color] == area]
    assert len(matches) == 1, "expected a unique largest non-background color"
    return int(matches[0])


def _least_common_marker(g: np.ndarray, bg: int, main_color: int) -> int:
    counts = Counter(map(int, g.ravel()))
    colors = [color for color in counts if color not in {bg, main_color}]
    assert colors, "expected at least one marker color"
    area = min(counts[color] for color in colors)
    matches = [color for color in colors if counts[color] == area]
    assert len(matches) == 1, "expected a unique smallest marker color"
    return int(matches[0])


def _mirror_four(g: np.ndarray) -> np.ndarray:
    top = np.concatenate([g, g[:, ::-1]], axis=1)
    bottom = np.concatenate([g[::-1, :], g[::-1, ::-1]], axis=1)
    return np.concatenate([top, bottom], axis=0)


def _outer_background(bg_mask: np.ndarray) -> np.ndarray:
    outer = np.zeros_like(bg_mask, dtype=bool)
    q: deque[tuple[int, int]] = deque()
    h, w = bg_mask.shape

    for r in range(h):
        for c in (0, w - 1):
            if bg_mask[r, c] and not outer[r, c]:
                outer[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if bg_mask[r, c] and not outer[r, c]:
                outer[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            rr, cc = r + dr, c + dc
            if _in_bounds(bg_mask.shape, rr, cc) and bg_mask[rr, cc] and not outer[rr, cc]:
                outer[rr, cc] = True
                q.append((rr, cc))
    return outer


def _trace_outer_outline(shape_grid: np.ndarray, main_color: int, marker_color: int) -> np.ndarray:
    main_mask = shape_grid == main_color
    erased = shape_grid.copy()
    erased[(shape_grid == main_color) | (shape_grid == marker_color)] = BG
    bg_mask = (erased == BG) & ~main_mask
    outer_bg = _outer_background(bg_mask)

    outline = np.zeros(shape_grid.shape, dtype=bool)
    for r, c in np.argwhere(outer_bg):
        rr = int(r)
        cc = int(c)
        for dr, dc in DIR4:
            nr, nc = rr + dr, cc + dc
            if _in_bounds(shape_grid.shape, nr, nc) and main_mask[nr, nc]:
                outline[rr, cc] = True
                break

    out = np.zeros_like(shape_grid)
    out[outline] = marker_color
    return out


def _component_count(mask: np.ndarray, connectivity: int = 8) -> int:
    dirs = DIR8 if connectivity == 8 else DIR4
    seen = np.zeros_like(mask, dtype=bool)
    count = 0
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        count += 1
        seen[rr, cc] = True
        q: deque[tuple[int, int]] = deque([(rr, cc)])
        while q:
            cr, cc2 = q.popleft()
            for dr, dc in dirs:
                nr, nc = cr + dr, cc2 + dc
                if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
    return count


def _has_hole(mask: np.ndarray) -> bool:
    bg_mask = ~mask
    return not np.array_equal(_outer_background(bg_mask), bg_mask)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the seed tile into four quadrants, erase the fused figure, and redraw only its outer contour in the marker color.

    Input structure:
    - The input is a small black grid containing one 8-connected foreground figure in a dominant non-black color.
    - The same tile also contains one or two marker pixels in a second non-black color whose count is strictly smaller.
    - The foreground figure touches a tile edge so bilateral mirroring can make copies meet across the central seams.

    Rule:
    - Identify the largest non-black color as the figure color and the smallest remaining non-black color as the marker color.
    - Build a 2H x 2W grid with the original tile in the upper-left, its horizontal mirror in the upper-right,
      its vertical mirror in the lower-left, and its 180-degree rotation in the lower-right.
    - Erase every figure-color pixel and every marker pixel to black, but remember the mirrored figure shape.
    - Paint the marker color onto every outer-background cell that is 4-neighbor adjacent to the erased mirrored figure.

    Key insight:
    - The contour must be traced after the four mirrored copies are assembled, because seam contact can fuse them into a new larger shape whose outer boundary is not the mirror of a single-tile outline.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    main_color = _most_common_non_bg(g, BG)
    marker_color = _least_common_marker(g, BG, main_color)
    big = _mirror_four(g)
    return _trace_outer_outline(big, main_color, marker_color)


def _touches(mask: np.ndarray, edge: str) -> bool:
    if edge == "top":
        return bool(mask[0, :].any())
    if edge == "bottom":
        return bool(mask[-1, :].any())
    if edge == "left":
        return bool(mask[:, 0].any())
    if edge == "right":
        return bool(mask[:, -1].any())
    raise ValueError(edge)


def _edge_cell(h: int, w: int, edge: str, avoid_edges: set[str] | None = None) -> tuple[int, int]:
    avoid_edges = set() if avoid_edges is None else set(avoid_edges)
    if edge == "top":
        cols = list(range(w))
        if "left" in avoid_edges and len(cols) > 1:
            cols.remove(0)
        if "right" in avoid_edges and len(cols) > 1 and (w - 1) in cols:
            cols.remove(w - 1)
        return 0, random.choice(cols)
    if edge == "bottom":
        cols = list(range(w))
        if "left" in avoid_edges and len(cols) > 1:
            cols.remove(0)
        if "right" in avoid_edges and len(cols) > 1 and (w - 1) in cols:
            cols.remove(w - 1)
        return h - 1, random.choice(cols)
    if edge == "left":
        rows = list(range(h))
        if "top" in avoid_edges and len(rows) > 1:
            rows.remove(0)
        if "bottom" in avoid_edges and len(rows) > 1 and (h - 1) in rows:
            rows.remove(h - 1)
        return random.choice(rows), 0
    if edge == "right":
        rows = list(range(h))
        if "top" in avoid_edges and len(rows) > 1:
            rows.remove(0)
        if "bottom" in avoid_edges and len(rows) > 1 and (h - 1) in rows:
            rows.remove(h - 1)
        return random.choice(rows), w - 1
    raise ValueError(edge)


def _step_toward(current: tuple[int, int], target: tuple[int, int]) -> tuple[int, int]:
    r, c = current
    tr, tc = target
    dr = 0 if tr == r else (1 if tr > r else -1)
    dc = 0 if tc == c else (1 if tc > c else -1)
    if dr != 0 and dc != 0:
        if random.random() < 0.8:
            return r + dr, c + dc
        if random.random() < 0.5:
            return r + dr, c
        return r, c + dc
    if dr != 0:
        return r + dr, c
    return r, c + dc


def _connect_points(points: list[tuple[int, int]]) -> set[tuple[int, int]]:
    cells = {points[0]}
    current = points[0]
    for target in points[1:]:
        while current != target:
            current = _step_toward(current, target)
            cells.add(current)
    return cells


def _frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if mask[r, c]:
                continue
            for dr, dc in DIR8:
                rr, cc = r + dr, c + dc
                if _in_bounds(mask.shape, rr, cc) and mask[rr, cc]:
                    out.append((r, c))
                    break
    return out


def _edge_penalty(cell: tuple[int, int], h: int, w: int, forbidden: set[str]) -> int:
    r, c = cell
    penalty = 0
    if "top" in forbidden and r == 0:
        penalty += 2
    if "bottom" in forbidden and r == h - 1:
        penalty += 2
    if "left" in forbidden and c == 0:
        penalty += 2
    if "right" in forbidden and c == w - 1:
        penalty += 2
    return penalty


def _neighbor_count(mask: np.ndarray, r: int, c: int) -> int:
    total = 0
    for dr, dc in DIR8:
        rr, cc = r + dr, c + dc
        if _in_bounds(mask.shape, rr, cc) and mask[rr, cc]:
            total += 1
    return total


def _draw_top_right_family(h: int, w: int) -> np.ndarray | None:
    lo = max(1, w - h + 1)
    hi = min(3, w - 3)
    if lo > hi:
        return None
    bar_end = random.randint(lo, hi)
    mask = np.zeros((h, w), dtype=bool)
    mask[0, : bar_end + 1] = True
    mask[0, w - 1] = True
    for row in range(1, w - bar_end):
        mask[row, bar_end + row] = True
    return mask


def _draw_bottom_right_family(h: int, w: int) -> np.ndarray | None:
    lo = max(1, w - h)
    hi = min(3, w - 2)
    if lo > hi:
        return None
    bar_end = random.randint(lo, hi)
    motif_h = w - bar_end + 1
    if motif_h > h:
        return None
    row_offset = h - motif_h
    mask = np.zeros((h, w), dtype=bool)
    mask[row_offset, : bar_end + 1] = True
    for step in range(1, w - bar_end):
        mask[row_offset + step, bar_end + step] = True
    mask[h - 1, w - 2] = True
    return mask


def _grow_shape(h: int, w: int, mode: dict[str, object]) -> np.ndarray | None:
    name = str(mode["name"])
    if name == "top_right":
        mask = _draw_top_right_family(h, w)
    elif name == "bottom_left":
        seed = _draw_top_right_family(h, w)
        mask = None if seed is None else seed[::-1, ::-1]
    elif name == "bottom_right":
        mask = _draw_bottom_right_family(h, w)
    else:
        raise ValueError(name)

    if mask is None:
        return None
    if not (GEN_MAIN_AREA_RANGE[0] <= int(mask.sum()) <= GEN_MAIN_AREA_RANGE[1]):
        return None
    if _component_count(mask, connectivity=8) != 1:
        return None
    if _component_count(mask, connectivity=4) == 1:
        return None
    return mask


def _seam_fusion(mask: np.ndarray) -> tuple[bool, bool]:
    big = _mirror_four(mask.astype(int)).astype(bool)
    h, w = mask.shape
    vertical = bool(np.any(big[:, w - 1] & big[:, w]))
    horizontal = bool(np.any(big[h - 1, :] & big[h, :]))
    return vertical, horizontal


def _small_outline_then_mirror(inp: np.ndarray, main_color: int, marker_color: int) -> np.ndarray:
    small_outline = _trace_outer_outline(inp, main_color, marker_color)
    return _mirror_four(small_outline)


def _orbit_positions(h: int, w: int, r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r, c),
        (r, 2 * w - 1 - c),
        (2 * h - 1 - r, c),
        (2 * h - 1 - r, 2 * w - 1 - c),
    ]


def _place_markers(mask: np.ndarray, main_color: int, marker_color: int) -> np.ndarray | None:
    h, w = mask.shape
    grid = np.zeros((h, w), dtype=int)
    grid[mask] = main_color

    bare_output = _trace_outer_outline(_mirror_four(grid), main_color, marker_color)
    erase_candidates: list[tuple[int, int]] = []
    contour_candidates: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                continue
            orbit = _orbit_positions(h, w, r, c)
            statuses = [bare_output[rr, cc] == marker_color for rr, cc in orbit]
            if all(statuses):
                contour_candidates.append((r, c))
            elif not any(statuses):
                erase_candidates.append((r, c))

    if not erase_candidates:
        return None

    erase_candidates.sort(
        key=lambda cell: (
            min(abs(cell[0] - rr) + abs(cell[1] - cc) for rr, cc in np.argwhere(mask)),
            random.random(),
        ),
        reverse=True,
    )
    chosen = [erase_candidates[0]]

    use_two = len(erase_candidates) >= 2 and random.random() < 0.55
    if use_two:
        second_pool = contour_candidates if contour_candidates and random.random() < 0.45 else erase_candidates[1:]
        if second_pool:
            second = random.choice(second_pool)
            if second not in chosen:
                chosen.append(second)

    for r, c in chosen:
        grid[r, c] = marker_color
    return grid


def _mode_ok(mask: np.ndarray, mode: dict[str, object]) -> bool:
    seam_require = {"right", "bottom"} & set(mode["require"])
    seam_forbid = {"right", "bottom"} & set(mode["forbid"])
    vertical, horizontal = _seam_fusion(mask)
    if "right" in seam_require and not vertical:
        return False
    if "bottom" in seam_require and not horizontal:
        return False
    if "right" in seam_forbid and vertical:
        return False
    if "bottom" in seam_forbid and horizontal:
        return False

    big = _mirror_four(mask.astype(int)).astype(bool)
    if _component_count(big, connectivity=8) != int(mode["components"]):
        return False
    if not _has_hole(big):
        return False
    return True


def _random_mode() -> dict[str, object]:
    weights = [int(mode["weight"]) for mode in GEN_MODES]
    return random.choices(GEN_MODES, weights=weights, k=1)[0]


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GEN_HEIGHT_RANGE)
    w = random.randint(*GEN_WIDTH_RANGE)
    palette = random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], 2)
    main_color, marker_color = palette
    mode = _random_mode()

    mask = _grow_shape(h, w, mode)
    if mask is None or not _mode_ok(mask, mode):
        return None

    grid = _place_markers(mask, main_color, marker_color)
    if grid is None:
        return None

    out = main(grid)
    if not np.any(out == marker_color):
        return None
    if np.array_equal(out, _small_outline_then_mirror(grid, main_color, marker_color)):
        return None
    if max(grid.shape) > 15 or max(out.shape) > 30:
        return None
    if np.count_nonzero(grid == marker_color) >= np.count_nonzero(grid == main_color):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
