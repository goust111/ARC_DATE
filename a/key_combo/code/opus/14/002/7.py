from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
RED = 2
ORANGE = 7
TEMPLATE_EXTRA_COLORS = (1, 3, 4, 6, 8, 9)
GEN_RETRIES = 400


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _components_of_nonblack(grid: np.ndarray) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == BLACK:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != BLACK:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _scan_fill(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for r in range(out.shape[0]):
        cols = np.flatnonzero(out[r] == ORANGE)
        for left, right in zip(cols, cols[1:]):
            if right - left > 1:
                segment = out[r, left + 1 : right]
                segment[segment == BLACK] = RED
                out[r, left + 1 : right] = segment
    return out


def _find_template(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    candidates: list[tuple[int, int, np.ndarray, tuple[int, int]]] = []
    for coords in _components_of_nonblack(grid):
        bbox = _bbox_from_coords(coords)
        r0, c0, r1, c1 = bbox
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if patch.shape[0] == patch.shape[1]:
            continue
        if np.any(patch == BLACK):
            continue
        if len(coords) != patch.shape[0] * patch.shape[1]:
            continue
        if not np.any(patch == ORANGE):
            continue
        if not np.any((patch != BLACK) & (patch != ORANGE)):
            continue
        candidates.append((r1 + c1, r0 + c0, patch.copy(), (r0, c0)))

    assert candidates, "expected one compact non-square template"
    candidates.sort(key=lambda item: (item[0], item[1]))
    _score1, _score2, template, origin = candidates[-1]
    return template, origin


def main(grid: np.ndarray) -> np.ndarray:
    """
    Scan-fill the orange diagonal contours, rotate the corner template to each red region's bbox, and stamp it with orange treated as transparent.

    Input structure:
    - A black background contains several orange contour strokes whose row-wise scan fill forms 2-3 separate red regions.
    - A compact non-square multicolor template sits near the lower-right corner and includes orange plus other non-black colors.
    - The template is fully occupied, while the contour strokes remain sparse and do not overlap the template rows.

    Rule:
    - For each row, look at all orange pixels from left to right and recolor every black cell strictly between each consecutive pair to red.
    - Find every 4-connected red region, compute its tight bounding box, and choose the smallest clockwise rotation of the template whose shape matches that bbox.
    - Overlay that rotated template at the red region bbox's top-left, but only overwrite red cells; template orange is transparent, and original orange contour pixels stay unchanged.

    Key insight:
    - The contour scan fill creates the target geometry first, and that geometry alone determines which rotation of the corner template fits; orange serves two roles at once: contour color and template transparency color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    template, _origin = _find_template(g)
    filled = _scan_fill(g)
    out = filled.copy()

    rotations = [np.rot90(template, -k) for k in range(4)]
    for coords in sorted(_components_of_color(filled, RED), key=lambda arr: tuple(arr.min(axis=0))):
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        chosen = None
        for rotated in rotations:
            if rotated.shape == (h, w):
                chosen = rotated
                break
        assert chosen is not None, "every red region must match a template rotation by bbox size"

        for rr in range(h):
            for cc in range(w):
                if int(out[r0 + rr, c0 + cc]) == RED and int(chosen[rr, cc]) != ORANGE:
                    out[r0 + rr, c0 + cc] = int(chosen[rr, cc])

    return out


def _make_template(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    a, b = random.sample(TEMPLATE_EXTRA_COLORS, 2)
    rr, cc = np.indices((h, w))
    style = random.choice(("diag", "bands", "corners"))

    if style == "diag":
        template = np.where((rr + 2 * cc) % 3 == 0, ORANGE, np.where((rr + cc) % 2 == 0, a, b))
    elif style == "bands":
        template = np.where(rr % 2 == 0, a, b)
        template[(rr + cc) % 3 == 1] = ORANGE
    else:
        d1 = rr + cc
        d2 = rr + (w - 1 - cc)
        template = np.where(d1 <= d2, a, b)
        template[(rr == h // 2) | (cc == w // 2)] = ORANGE

    template = template.astype(int)
    if random.random() < 0.5:
        template = np.flipud(template)
    if random.random() < 0.5:
        template = np.fliplr(template)
    if random.random() < 0.5:
        template[0, 0] = ORANGE
    if random.random() < 0.5:
        template[-1, -1] = a

    colors = set(map(int, template.ravel()))
    if ORANGE not in colors:
        template[h // 2, w // 2] = ORANGE
    if a not in set(map(int, template.ravel())):
        template[0, min(1, w - 1)] = a
    if b not in set(map(int, template.ravel())):
        template[-1, max(0, w - 2)] = b
    assert np.any(template == ORANGE)
    assert np.any(template == a)
    assert np.any(template == b)
    return template


def _draw_region_outline(grid: np.ndarray, top: int, left: int, shape: tuple[int, int]) -> None:
    h, w = shape
    for dr in range(h):
        row = top + dr
        lc = left - 1 + (dr % 2)
        rc = left + w - (dr % 2)
        grid[row, lc] = ORANGE
        grid[row, rc] = ORANGE


def _region_fill_cells(top: int, left: int, shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    cells: set[tuple[int, int]] = set()
    for dr in range(h):
        start = left + (dr % 2)
        stop = left + w - (dr % 2)
        for c in range(start, stop):
            cells.add((top + dr, c))
    return cells


def _choose_region_shapes(template_shape: tuple[int, int], count: int) -> list[tuple[int, int]]:
    h, w = template_shape
    rotated = (w, h)
    shapes = [template_shape, rotated]
    if count == 3:
        shapes.append(random.choice((template_shape, rotated)))
    random.shuffle(shapes)
    return shapes


def _place_regions(
    grid_shape: tuple[int, int],
    template_top: int,
    region_shapes: list[tuple[int, int]],
) -> list[tuple[int, int, tuple[int, int]]] | None:
    H, W = grid_shape
    slack = template_top - 2 - sum(h for h, _w in region_shapes) - (len(region_shapes) - 1)
    if slack < 1:
        return None

    current = random.randint(1, slack)
    placements: list[tuple[int, int, tuple[int, int]]] = []
    for idx, shape in enumerate(region_shapes):
        h, w = shape
        min_left = 1
        max_left = W - w - 1
        if max_left < min_left:
            return None
        left = random.randint(min_left, max_left)
        placements.append((current, left, shape))
        if idx != len(region_shapes) - 1:
            remaining = sum(s[0] for s in region_shapes[idx + 1 :]) + (len(region_shapes) - idx - 2)
            latest_next = template_top - 2 - remaining
            next_min = current + h + 1
            next_max = latest_next
            if next_max < next_min:
                return None
            current = random.randint(next_min, next_max)
    return placements


def _make_candidate() -> np.ndarray | None:
    template_shape = random.choice(((3, 4), (4, 3)))
    template = _make_template(template_shape)
    region_count = random.choice((2, 3))
    region_shapes = _choose_region_shapes(template_shape, region_count)

    min_h = sum(h for h, _w in region_shapes) + len(region_shapes) + template_shape[0] + 4
    min_w = max(max(w for _h, w in region_shapes) + 3, template_shape[1] + 2, 16)
    H = random.randint(min_h, min(26, max(min_h, min_h + 5)))
    W = random.randint(min_w, min(26, max(min_w, min_w + 6)))

    bottom_margin = random.randint(0, 1)
    right_margin = random.randint(0, 1)
    template_top = H - template_shape[0] - bottom_margin
    template_left = W - template_shape[1] - right_margin
    if template_top < 6 or template_left < 6:
        return None

    placements = _place_regions((H, W), template_top, region_shapes)
    if placements is None:
        return None

    grid = np.zeros((H, W), dtype=int)
    for top, left, shape in placements:
        _draw_region_outline(grid, top, left, shape)

    grid[template_top : template_top + template_shape[0], template_left : template_left + template_shape[1]] = template

    filled = _scan_fill(grid)
    red_components = _components_of_color(filled, RED)
    if len(red_components) != region_count:
        return None

    found_shapes = []
    expected_sets = []
    for top, left, shape in placements:
        expected_sets.append(_region_fill_cells(top, left, shape))
    unmatched = expected_sets[:]
    for comp in red_components:
        comp_set = {tuple(map(int, rc)) for rc in comp}
        if comp_set not in unmatched:
            return None
        unmatched.remove(comp_set)
        bbox = _bbox_from_coords(comp)
        found_shapes.append((bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1))
    if unmatched:
        return None

    if template_shape not in found_shapes or (template_shape[1], template_shape[0]) not in found_shapes:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _make_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
