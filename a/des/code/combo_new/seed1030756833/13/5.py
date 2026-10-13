from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
LABEL_COLORS = [1, 2, 3, 4, 6, 7, 8]
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        comp: list[tuple[int, int]] = []
        q: deque[tuple[int, int]] = deque([(sr, sc)])
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for dr, dc in NEIGHBORS4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
        comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _find_divider(g: np.ndarray) -> int:
    cols = [int(c) for c in np.where(np.all(g == MAROON, axis=0))[0]]
    assert len(cols) == 1, "expected exactly one full maroon divider column"
    divider = cols[0]
    assert divider > 0 and g.shape[1] - divider - 1 == divider, "divider must split equal panels"
    return divider


def _extract_templates(left: np.ndarray) -> dict[int, dict[str, object]]:
    templates: dict[int, dict[str, object]] = {}
    label_colors: set[int] = set()
    for comp in _components(left != BLACK):
        vals = [int(left[r, c]) for r, c in comp]
        labels = [v for v in vals if v not in (GREY, BLACK, MAROON)]
        assert len(labels) == 1, "each template must contain exactly one colored label pixel"
        assert all(v in (GREY, labels[0]) for v in vals), "template may contain only grey and its label"
        color = int(labels[0])
        assert color not in label_colors, "template labels must use distinct colors"
        label_colors.add(color)

        r0, c0, _r1, _c1 = _bbox(comp)
        offsets = sorted((r - r0, c - c0) for r, c in comp)
        height = max(r for r, _ in offsets) + 1
        width = max(c for _, c in offsets) + 1
        area = len(comp)
        assert area not in templates, "template areas must be unique"
        templates[area] = {"color": color, "offsets": offsets, "height": height, "width": width}
    assert templates, "expected at least one template"
    return templates


def _extract_targets(right: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    targets: list[tuple[int, int, int, int, int]] = []
    for comp in _components(right == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        area = len(comp)
        assert area == (r1 - r0 + 1) * (c1 - c0 + 1), "target grey component must be a solid rectangle"
        assert np.all(right[r0 : r1 + 1, c0 : c1 + 1] == GREY)
        targets.append((r0, c0, r1, c1, area))
    assert targets, "expected at least one target rectangle"
    targets.sort()
    return targets


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp each labeled left-panel template onto the right-panel rectangle with the same area.

    Input structure:
    - Black background and one full-height vertical Maroon divider split the grid into equal left and right panels.
    - The left panel contains separated 4-connected Grey template shapes; each shape has exactly one embedded non-Grey color-label pixel.
    - The right panel contains the same number of separated solid Grey rectangular target regions.
    - Template areas are all distinct, and each target rectangle area equals exactly one template area.

    Rule:
    - Keep the left panel and the Maroon divider unchanged.
    - For every Grey target rectangle, count its pixels and find the left template with the same area.
    - Clear the original target rectangle to Black, then stamp the full template shape at the rectangle's top-left position.
    - Recolor every stamped template pixel with the template's embedded label color.

    Key insight:
    - The target-to-template correspondence is not positional or color-based; it is established by unique area counts before transferring and recoloring the shape.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    divider = _find_divider(g)
    left = g[:, :divider]
    right = g[:, divider + 1 :]
    templates = _extract_templates(left)
    targets = _extract_targets(right)
    assert len(targets) == len(templates), "target and template counts must match"

    out = g.copy()
    panel_h, panel_w = right.shape
    for r0, c0, r1, c1, area in targets:
        assert area in templates, "every target area must match one template"
        template = templates[area]
        color = int(template["color"])
        offsets = template["offsets"]
        th = int(template["height"])
        tw = int(template["width"])
        assert r0 + th <= panel_h and c0 + tw <= panel_w, "stamped template must fit in right panel"

        gr0 = r0
        gc0 = divider + 1 + c0
        out[gr0 : r1 + 1, gc0 : divider + 1 + c1 + 1] = BLACK
        for dr, dc in offsets:
            out[gr0 + int(dr), gc0 + int(dc)] = color
    return out


# --------------- generator ---------------

GRID_HEIGHT_RANGE = (12, 28)
PANEL_WIDTH_RANGE = (10, 14)
OBJECT_COUNT_RANGE = (2, 4)
SHAPE_H_RANGE = (2, 6)
SHAPE_W_RANGE = (2, 6)
GENERATION_ATTEMPTS = 500
PLACE_ATTEMPTS = 200


def _random_connected_shape(height: int, width: int, area: int) -> np.ndarray | None:
    if not (1 <= area <= height * width):
        return None
    cells = {(random.randrange(height), random.randrange(width))}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in NEIGHBORS4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < height and 0 <= cc < width and (rr, cc) not in cells:
                    frontier.append((rr, cc))
        if not frontier:
            return None
        cells.add(random.choice(frontier))

    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    norm = {(r - r0, c - c0) for r, c in cells}
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in norm:
        mask[r, c] = True
    return mask


def _make_shape(max_h: int, max_w: int, used_areas: set[int]) -> np.ndarray | None:
    for _ in range(100):
        h = random.randint(*SHAPE_H_RANGE)
        w = random.randint(*SHAPE_W_RANGE)
        h = min(h, max_h)
        w = min(w, max_w)
        if h < 2 or w < 2:
            continue
        lo = max(4, min(h, w))
        hi = min(h * w - 1, 17)
        if hi < lo:
            continue
        area = random.randint(lo, hi)
        if area in used_areas:
            continue
        mask = _random_connected_shape(h, w, area)
        if mask is None or int(mask.sum()) in used_areas:
            continue
        if mask.shape[0] > max_h or mask.shape[1] > max_w:
            continue
        used_areas.add(int(mask.sum()))
        return mask
    return None


def _stamp_with_padding(occupied: np.ndarray, cells: list[tuple[int, int]], pad: int = 1) -> None:
    h, w = occupied.shape
    for r, c in cells:
        for rr in range(max(0, r - pad), min(h, r + pad + 1)):
            for cc in range(max(0, c - pad), min(w, c + pad + 1)):
                occupied[rr, cc] = True


def _can_place(occupied: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    h, w = occupied.shape
    return all(0 <= r < h and 0 <= c < w and not occupied[r, c] for r, c in cells)


def _place_template(left: np.ndarray, occupied: np.ndarray, mask: np.ndarray, color: int) -> bool:
    mh, mw = mask.shape
    h, w = left.shape
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    for _ in range(PLACE_ATTEMPTS):
        r0 = random.randint(0, h - mh)
        c0 = random.randint(0, w - mw)
        cells = [(r0 + r, c0 + c) for r, c in coords]
        if not _can_place(occupied, cells):
            continue
        for r, c in cells:
            left[r, c] = GREY
        label_r, label_c = random.choice(cells)
        left[label_r, label_c] = color
        _stamp_with_padding(occupied, cells)
        return True
    return False


def _factor_pairs(area: int, max_h: int, max_w: int) -> list[tuple[int, int]]:
    pairs = []
    for h in range(1, max_h + 1):
        if area % h == 0:
            w = area // h
            if 1 <= w <= max_w:
                pairs.append((h, w))
    return pairs


def _place_target(
    right: np.ndarray,
    occupied: np.ndarray,
    mask: np.ndarray,
    area: int,
) -> bool:
    panel_h, panel_w = right.shape
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    pairs = _factor_pairs(area, panel_h, panel_w)
    random.shuffle(pairs)
    for rh, rw in pairs:
        for _ in range(PLACE_ATTEMPTS):
            max_r = panel_h - max(rh, mask.shape[0])
            max_c = panel_w - max(rw, mask.shape[1])
            if max_r < 0 or max_c < 0:
                continue
            r0 = random.randint(0, max_r)
            c0 = random.randint(0, max_c)
            rect_cells = [(r, c) for r in range(r0, r0 + rh) for c in range(c0, c0 + rw)]
            shape_cells = [(r0 + r, c0 + c) for r, c in coords]
            footprint = list(set(rect_cells + shape_cells))
            if not _can_place(occupied, footprint):
                continue
            for r, c in rect_cells:
                right[r, c] = GREY
            _stamp_with_padding(occupied, footprint)
            return True
    return False


def _expected_from_parts(left: np.ndarray, right: np.ndarray, divider_col: np.ndarray) -> np.ndarray:
    grid = np.concatenate([left, divider_col, right], axis=1)
    return main(grid)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    panel_w = random.randint(*PANEL_WIDTH_RANGE)
    count = random.randint(*OBJECT_COUNT_RANGE)
    if count > len(LABEL_COLORS):
        return None

    left = np.full((h, panel_w), BLACK, dtype=int)
    right = np.full((h, panel_w), BLACK, dtype=int)
    left_occ = np.zeros((h, panel_w), dtype=bool)
    right_occ = np.zeros((h, panel_w), dtype=bool)

    colors = random.sample(LABEL_COLORS, count)
    used_areas: set[int] = set()
    shapes: list[tuple[np.ndarray, int]] = []
    for color in colors:
        mask = _make_shape(h, panel_w, used_areas)
        if mask is None:
            return None
        shapes.append((mask, color))

    random.shuffle(shapes)
    for mask, color in shapes:
        if not _place_template(left, left_occ, mask, color):
            return None

    random.shuffle(shapes)
    for mask, _color in shapes:
        area = int(mask.sum())
        if not _place_target(right, right_occ, mask, area):
            return None

    divider_col = np.full((h, 1), MAROON, dtype=int)
    grid = np.concatenate([left, divider_col, right], axis=1)
    try:
        out = main(grid)
        expected = _expected_from_parts(left, right, divider_col)
    except Exception:
        return None
    if not np.array_equal(out, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for seed1030756833/13/5")
