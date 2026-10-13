from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2

MAX_GENERATION_ATTEMPTS = 1000
ANCHOR_COUNTS = (3, 4, 5)
TEMPLATE_HEIGHTS = (3, 4, 5, 6)
TEMPLATE_WIDTHS = (3, 4, 5, 6)


def _neighbors(r: int, c: int, connectivity: int = 4):
    if connectivity == 4:
        yield r - 1, c
        yield r + 1, c
        yield r, c - 1
        yield r, c + 1
    else:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr or dc:
                    yield r + dr, c + dc


def _is_connected(mask: np.ndarray, connectivity: int = 4) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, pts[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors(r, c, connectivity):
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    box = _bbox(mask)
    assert box is not None, "expected a nonempty template intersection"
    r0, c0, r1, c1 = box
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_to_blue(mask: np.ndarray) -> np.ndarray:
    out = np.full(mask.shape, BLACK, dtype=int)
    out[mask.astype(bool)] = BLUE
    return out


def _intersection_from_anchors(grid: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    assert len(anchors) > 0, "expected at least one red anchor"
    foreground = (grid == BLUE) | (grid == RED)
    common_h = min(h - int(r) for r, _c in anchors)
    common_w = min(w - int(c) for _r, c in anchors)
    assert common_h > 0 and common_w > 0, "anchors must be inside the grid"

    inter = np.ones((common_h, common_w), dtype=bool)
    for r, c in anchors:
        r = int(r)
        c = int(c)
        inter &= foreground[r : r + common_h, c : c + common_w]
    return inter


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover an unknown blue stamp template by intersecting the foreground translated back to every red anchor.

    Input structure:
    - A black background contains several red anchor pixels and a blue foreground region.
    - The blue foreground is the union of one hidden template stamped with each
      red anchor as the template bounding-box upper-left corner.
    - Red anchor pixels may cover blue cells and are treated as foreground while
      testing aligned template offsets.

    Rule:
    - Use every red pixel as an anchor.
    - For each nonnegative offset from the anchors, keep it exactly when all
      anchor-shifted locations contain blue or red foreground.
    - Return the tight crop of the kept offsets, painted blue on black.

    Key insight:
    - Each individual stamped region is polluted by overlaps from other stamps;
      aligning the whole foreground to every anchor and intersecting cancels the
      pollution and leaves only the common stamp template.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    anchors = np.argwhere(g == RED)
    assert len(anchors) >= 2, "expected multiple red anchors"

    template_mask = _crop_mask(_intersection_from_anchors(g, anchors))
    return _mask_to_blue(template_mask)


def _random_template() -> np.ndarray | None:
    th = random.choice(TEMPLATE_HEIGHTS)
    tw = random.choice(TEMPLATE_WIDTHS)
    area_min = max(th + tw - 1, 5)
    area_max = min(th * tw - 3, max(area_min, int(round(th * tw * 0.68))))
    if area_min > area_max:
        return None

    for _ in range(150):
        area = random.randint(area_min, area_max)
        mask = np.zeros((th, tw), dtype=bool)
        mask[0, 0] = True
        cells = {(0, 0)}

        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors(r, c, 4):
                    if 0 <= nr < th and 0 <= nc < tw and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        for r, c in cells:
            mask[r, c] = True

        if not mask[-1, :].any() or not mask[:, -1].any():
            continue
        if int(mask.sum()) >= th * tw - 1:
            continue
        if int((~mask).sum()) < 3:
            continue
        if not _is_connected(mask, connectivity=4):
            continue
        return mask
    return None


def _sample_relative_anchors(template: np.ndarray, count: int, require_background_anchor: bool) -> list[tuple[int, int]] | None:
    th, tw = template.shape
    rel: list[tuple[int, int]] = [(0, 0)]
    row_radius = min(2, th - 1)
    col_radius = min(2, tw - 1)

    if require_background_anchor:
        background_offsets = [
            (r, c)
            for r in range(min(th, row_radius + 1))
            for c in range(min(tw, col_radius + 1))
            if (r, c) != (0, 0) and not bool(template[r, c])
        ]
        if not background_offsets:
            return None
        rel.append(random.choice(background_offsets))

    pool = [
        (dr, dc)
        for dr in range(-row_radius, row_radius + 1)
        for dc in range(-col_radius, col_radius + 1)
        if (dr, dc) != (0, 0) and (dr, dc) not in rel
    ]
    random.shuffle(pool)
    if len(pool) < count - len(rel):
        return None

    tries = 0
    while len(rel) < count and tries < 200:
        tries += 1
        dr, dc = pool.pop()
        if (dr, dc) in rel:
            continue
        rel.append((dr, dc))

    if len(rel) != count:
        return None
    return rel


def _absolute_anchors(rel: list[tuple[int, int]], template_shape: tuple[int, int]) -> tuple[int, int, list[tuple[int, int]]] | None:
    th, tw = template_shape
    min_r = min(r for r, _c in rel)
    max_r = max(r for r, _c in rel)
    min_c = min(c for _r, c in rel)
    max_c = max(c for _r, c in rel)

    margin = 2
    min_h = (max_r - min_r) + th + 2 * margin
    min_w = (max_c - min_c) + tw + 2 * margin
    if min_h > 30 or min_w > 30:
        return None

    grid_h = random.randint(max(10, min_h), 30)
    grid_w = random.randint(max(10, min_w), 30)

    base_r_lo = margin - min_r
    base_r_hi = grid_h - margin - th - max_r
    base_c_lo = margin - min_c
    base_c_hi = grid_w - margin - tw - max_c
    if base_r_lo > base_r_hi or base_c_lo > base_c_hi:
        return None

    base_r = random.randint(base_r_lo, base_r_hi)
    base_c = random.randint(base_c_lo, base_c_hi)
    anchors = [(base_r + r, base_c + c) for r, c in rel]
    return grid_h, grid_w, anchors


def _render_case(template: np.ndarray, anchors: list[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    coords = np.argwhere(template)
    for ar, ac in anchors:
        for tr, tc in coords:
            grid[int(ar) + int(tr), int(ac) + int(tc)] = BLUE
    for ar, ac in anchors:
        grid[int(ar), int(ac)] = RED
    return grid


def _expected_output(template: np.ndarray) -> np.ndarray:
    return _mask_to_blue(_crop_mask(template))


def _all_stamps_polluted(template: np.ndarray, anchors: list[tuple[int, int]], grid: np.ndarray) -> bool:
    th, tw = template.shape
    foreground = (grid == BLUE) | (grid == RED)
    for ar, ac in anchors:
        observed = foreground[ar : ar + th, ac : ac + tw]
        if observed.shape != template.shape:
            return False
        if not bool((observed & ~template).any()):
            return False
    return True


def _each_stamp_overlaps_another(template: np.ndarray, anchors: list[tuple[int, int]]) -> bool:
    coords = [tuple(map(int, p)) for p in np.argwhere(template)]
    stamp_sets = []
    for ar, ac in anchors:
        stamp_sets.append({(ar + r, ac + c) for r, c in coords})
    for i, cells in enumerate(stamp_sets):
        if not any(cells & other for j, other in enumerate(stamp_sets) if i != j):
            return False
    return True


def _has_anchor_on_other_template_background(template: np.ndarray, anchors: list[tuple[int, int]]) -> bool:
    th, tw = template.shape
    for i, (ar, ac) in enumerate(anchors):
        for j, (br, bc) in enumerate(anchors):
            if i == j:
                continue
            dr = br - ar
            dc = bc - ac
            if 0 <= dr < th and 0 <= dc < tw and not bool(template[dr, dc]):
                return True
    return False


def _foreground_connected(grid: np.ndarray) -> bool:
    return _is_connected((grid == BLUE) | (grid == RED), connectivity=8)


def _solve_with_anchor_subset(grid: np.ndarray, anchors: list[tuple[int, int]]) -> np.ndarray:
    inter = _intersection_from_anchors(grid, np.array(anchors, dtype=int))
    return _mask_to_blue(_crop_mask(inter))


def _multi_anchor_intersection_is_necessary(grid: np.ndarray, anchors: list[tuple[int, int]], expected: np.ndarray) -> bool:
    for anchor in anchors:
        try:
            single = _solve_with_anchor_subset(grid, [anchor])
        except AssertionError:
            continue
        if np.array_equal(single, expected):
            return False

    any_dropped_anchor_changes = False
    for i in range(len(anchors)):
        subset = [a for j, a in enumerate(anchors) if i != j]
        try:
            reduced = _solve_with_anchor_subset(grid, subset)
        except AssertionError:
            any_dropped_anchor_changes = True
            continue
        if not np.array_equal(reduced, expected):
            any_dropped_anchor_changes = True
    return any_dropped_anchor_changes


def _crop_is_required(grid: np.ndarray, anchors: list[tuple[int, int]], expected: np.ndarray) -> bool:
    raw = _intersection_from_anchors(grid, np.array(anchors, dtype=int))
    return raw.shape != expected.shape


def _gen_attempt() -> np.ndarray | None:
    template = _random_template()
    if template is None:
        return None

    count = random.choice(ANCHOR_COUNTS)
    require_background_anchor = random.random() < 0.6
    rel = _sample_relative_anchors(template, count, require_background_anchor)
    if rel is None:
        return None

    placed = _absolute_anchors(rel, template.shape)
    if placed is None:
        return None
    grid_h, grid_w, anchors = placed

    grid = _render_case(template, anchors, (grid_h, grid_w))
    expected = _expected_output(template)

    try:
        solved = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(solved, expected):
        return None
    if not _foreground_connected(grid):
        return None
    if not _each_stamp_overlaps_another(template, anchors):
        return None
    if not _all_stamps_polluted(template, anchors, grid):
        return None
    if require_background_anchor and not _has_anchor_on_other_template_background(template, anchors):
        return None
    if not _multi_anchor_intersection_is_necessary(grid, anchors, expected):
        return None
    if not _crop_is_required(grid, anchors, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_GENERATION_ATTEMPTS} attempts")
