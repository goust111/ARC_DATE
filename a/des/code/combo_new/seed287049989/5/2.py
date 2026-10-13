from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


def _dominant_color(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.reshape(-1)))
    ranked = counts.most_common()
    assert ranked, "expected a nonempty grid"
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "background must be uniquely dominant"
    return int(ranked[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _normalize_coords(coords: list[tuple[int, int]] | np.ndarray) -> tuple[frozenset[tuple[int, int]], int, int]:
    if isinstance(coords, np.ndarray):
        pts = [(int(r), int(c)) for r, c in coords]
    else:
        pts = [(int(r), int(c)) for r, c in coords]
    assert pts, "shape must contain at least one cell"
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    r0, c0 = min(rows), min(cols)
    r1, c1 = max(rows), max(cols)
    norm = frozenset((r - r0, c - c0) for r, c in pts)
    return norm, int(r1 - r0 + 1), int(c1 - c0 + 1)


def _is_connected(mask: np.ndarray) -> bool:
    return len(_find_components(mask.astype(bool))) == 1


def _is_solid_bbox(mask: np.ndarray) -> bool:
    return int(mask.sum()) == int(mask.shape[0] * mask.shape[1])


def _rectangle_dimension_candidates(
    coords_a: list[tuple[int, int]], coords_b: list[tuple[int, int]]
) -> set[tuple[int, int]]:
    fixed = coords_a
    other = coords_b
    if len(other) > len(fixed):
        fixed, other = other, fixed

    fixed_norm, fixed_h, fixed_w = _normalize_coords(fixed)
    other_norm, other_h, other_w = _normalize_coords(other)
    fixed_pts = np.array(sorted(fixed_norm), dtype=int)
    total_area = len(coords_a) + len(coords_b)

    candidates: set[tuple[int, int]] = set()
    for out_h in range(max(fixed_h, other_h), total_area + 1):
        if total_area % out_h:
            continue
        out_w = total_area // out_h
        if out_w < max(fixed_w, other_w):
            continue
        for r0 in range(out_h - fixed_h + 1):
            for c0 in range(out_w - fixed_w + 1):
                occ = np.zeros((out_h, out_w), dtype=bool)
                occ[r0 + fixed_pts[:, 0], c0 + fixed_pts[:, 1]] = True
                comp_coords = np.argwhere(~occ)
                comp_norm, comp_h, comp_w = _normalize_coords(comp_coords)
                if (comp_h, comp_w) == (other_h, other_w) and comp_norm == other_norm:
                    candidates.add((int(out_h), int(out_w)))
                    break
            else:
                continue
            break
    return candidates


def _recover_rectangle_dims(coords_a: list[tuple[int, int]], coords_b: list[tuple[int, int]]) -> tuple[int, int]:
    candidates = _rectangle_dimension_candidates(coords_a, coords_b)
    assert len(candidates) == 1, "same-color fragments must assemble into one unique rectangle"
    return next(iter(candidates))


def _corner_marker_and_fragments(
    grid: np.ndarray, bg: int
) -> tuple[tuple[int, tuple[int, int]], dict[int, list[list[tuple[int, int]]]]]:
    h, w = grid.shape
    corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    marker: tuple[int, tuple[int, int]] | None = None
    fragments: dict[int, list[list[tuple[int, int]]]] = {}

    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        comps = _find_components(grid == color)
        if len(comps) == 1 and len(comps[0]) == 1 and comps[0][0] in corners:
            assert marker is None, "expected one unique corner marker"
            marker = (color, comps[0][0])
        else:
            assert len(comps) == 2, "each fragment color must appear in exactly two components"
            fragments[color] = comps

    assert marker is not None, "missing corner marker"
    assert fragments, "expected at least one fragment pair"
    return marker, fragments


def _render_output(rects: list[tuple[int, int, int, int]], bg: int, anchor_left: bool) -> np.ndarray:
    out_h = max(h for _color, h, _w, _area in rects)
    out_w = sum(w for _color, _h, w, _area in rects) + len(rects) - 1
    out = np.full((out_h, out_w), bg, dtype=int)

    if anchor_left:
        col = 0
        for color, h, w, _area in rects:
            out[:h, col : col + w] = color
            col += w + 1
    else:
        edge = out_w
        for color, h, w, _area in rects:
            left = edge - w
            out[:h, left:edge] = color
            edge = left - 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble each same-color fragment pair into a rectangle, then pack the rectangles by increasing area.

    Input structure:
    - The background is the uniquely most common color.
    - Non-background pixels form monochrome 4-connected fragments; every real object color appears in exactly two fragments.
    - One extra color appears as a single marker pixel in a grid corner and marks the packing anchor.

    Rule:
    - For each non-marker color, translate its two fragments without rotating or flipping them until they exactly fill one solid rectangle.
    - The rectangle dimensions are the unique ones for which the two translated fragments partition the full box with no overlap.
    - Sort the reconstructed rectangles by area from small to large.
    - Build the output canvas just large enough to hold them in one horizontal row with one background column between neighbors; rectangles are top-aligned, and the corner marker chooses whether packing starts from the left edge or the right edge.

    Key insight:
    - The task is not to compare fragment silhouettes directly, but to infer the only rectangle each same-color pair can jointly complete, then use those completed rectangle areas to determine the packing order.
    """
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    (_marker_color, marker_pos), fragments = _corner_marker_and_fragments(g, bg)
    anchor_left = marker_pos[1] == 0

    rects: list[tuple[int, int, int, int]] = []
    for color, (coords_a, coords_b) in fragments.items():
        h, w = _recover_rectangle_dims(coords_a, coords_b)
        rects.append((int(color), int(h), int(w), int(h * w)))

    areas = [area for _color, _h, _w, area in rects]
    assert len(set(areas)) == len(areas), "rectangle areas must give an unambiguous sort order"
    rects.sort(key=lambda item: item[3])
    return _render_output(rects, bg, anchor_left)


# --------------- generator ---------------

COLOR_VALUES = tuple(range(10))
ARC_MAX = 30
RECT_COUNT_RANGE = (2, 4)
RECT_H_RANGE = (2, 5)
RECT_W_RANGE = (3, 7)
RECT_AREA_MIN = 8
RECT_AREA_MAX = 20
OUTPUT_GAP = 1
INPUT_H_RANGE = (14, 24)
INPUT_W_RANGE = (16, 30)
PARTITION_ATTEMPTS = 200
PLACEMENT_ATTEMPTS = 80
GENERATION_ATTEMPTS = 300


def _grow_connected_mask(h: int, w: int, area: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    seed = (random.randrange(h), random.randrange(w))
    mask[seed] = True
    while int(mask.sum()) < area:
        frontier = set()
        for r, c in np.argwhere(mask):
            for nr, nc in _neighbors4(int(r), int(c)):
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((int(nr), int(nc)))
        assert frontier, "connected region growth frontier unexpectedly empty"
        nr, nc = random.choice(tuple(frontier))
        mask[nr, nc] = True
    return mask


def _sample_partition(h: int, w: int) -> tuple[np.ndarray, np.ndarray] | None:
    total = h * w
    low = max(2, total // 3)
    high = min(total - 2, (2 * total) // 3)
    for _ in range(PARTITION_ATTEMPTS):
        area_a = random.randint(low, high)
        mask_a = _grow_connected_mask(h, w, area_a)
        mask_b = ~mask_a
        if not _is_connected(mask_b):
            continue
        if _is_solid_bbox(mask_a) or _is_solid_bbox(mask_b):
            continue
        coords_a = [(int(r), int(c)) for r, c in np.argwhere(mask_a)]
        coords_b = [(int(r), int(c)) for r, c in np.argwhere(mask_b)]
        candidates = _rectangle_dimension_candidates(coords_a, coords_b)
        if candidates != {(h, w)}:
            continue
        return mask_a, mask_b
    return None


def _sample_rectangles(n_rects: int) -> list[tuple[int, int]] | None:
    rects: list[tuple[int, int]] = []
    used_areas: set[int] = set()
    width_budget = 0
    for i in range(n_rects):
        placed = False
        for _ in range(PARTITION_ATTEMPTS):
            h = random.randint(*RECT_H_RANGE)
            w = random.randint(*RECT_W_RANGE)
            area = h * w
            if not (RECT_AREA_MIN <= area <= RECT_AREA_MAX):
                continue
            if area in used_areas:
                continue
            projected = width_budget + w + (1 if i > 0 else 0)
            if projected > ARC_MAX:
                continue
            rects.append((h, w))
            used_areas.add(area)
            width_budget = projected
            placed = True
            break
        if not placed:
            return None
    return rects


def _dilate(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _place_patch(
    grid: np.ndarray, occupied: np.ndarray, mask: np.ndarray, color: int
) -> bool:
    h, w = mask.shape
    H, W = grid.shape
    candidates = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
    random.shuffle(candidates)
    for r0, c0 in candidates[:PLACEMENT_ATTEMPTS]:
        blocked = _dilate(occupied)
        if bool((blocked[r0 : r0 + h, c0 : c0 + w] & mask).any()):
            continue
        grid[r0 : r0 + h, c0 : c0 + w][mask] = color
        occupied[r0 : r0 + h, c0 : c0 + w][mask] = True
        return True
    return False


def _build_input(
    pieces: list[tuple[int, np.ndarray]], bg: int, marker_color: int, anchor_left: bool
) -> np.ndarray | None:
    max_h = max(mask.shape[0] for _color, mask in pieces)
    max_w = max(mask.shape[1] for _color, mask in pieces)
    min_h = max(INPUT_H_RANGE[0], max_h + 4)
    min_w = max(INPUT_W_RANGE[0], max_w + 4)
    if min_h > INPUT_H_RANGE[1] or min_w > INPUT_W_RANGE[1]:
        return None

    H = random.randint(min_h, INPUT_H_RANGE[1])
    W = random.randint(min_w, INPUT_W_RANGE[1])
    grid = np.full((H, W), bg, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    marker_pos = (0, 0) if anchor_left else (0, W - 1)
    occupied[marker_pos] = True

    order = pieces[:]
    random.shuffle(order)
    order.sort(key=lambda item: int(item[1].sum()), reverse=True)
    for color, mask in order:
        if not _place_patch(grid, occupied, mask, color):
            return None

    grid[marker_pos] = marker_color
    return grid


def _expected_output(rect_specs: list[tuple[int, int, int]], bg: int, anchor_left: bool) -> np.ndarray:
    rects = [(color, h, w, h * w) for color, h, w in rect_specs]
    rects.sort(key=lambda item: item[3])
    return _render_output(rects, bg, anchor_left)


def _gen_attempt() -> np.ndarray | None:
    n_rects = random.randint(*RECT_COUNT_RANGE)
    rect_dims = _sample_rectangles(n_rects)
    if rect_dims is None:
        return None

    if n_rects + 2 > len(COLOR_VALUES):
        return None
    palette = random.sample(COLOR_VALUES, n_rects + 2)
    bg = int(palette[0])
    marker_color = int(palette[1])
    rect_colors = [int(c) for c in palette[2:]]
    anchor_left = random.random() < 0.5

    rect_specs: list[tuple[int, int, int]] = []
    pieces: list[tuple[int, np.ndarray]] = []
    for color, (h, w) in zip(rect_colors, rect_dims):
        part = _sample_partition(h, w)
        if part is None:
            return None
        mask_a, mask_b = part
        rect_specs.append((color, h, w))
        pieces.extend([(color, mask_a), (color, mask_b)])

    grid = _build_input(pieces, bg, marker_color, anchor_left)
    if grid is None:
        return None

    expected = _expected_output(rect_specs, bg, anchor_left)
    if max(expected.shape) > ARC_MAX:
        return None
    if _dominant_color(grid) != bg:
        return None
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fragment-packing puzzle")
