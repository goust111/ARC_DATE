import random

import numpy as np


BG = 7
OBJ = 5
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
CAT_NONE = 0
CAT_ONE = 1
CAT_CORNER = 2
CAT_OPPOSITE = 3


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _legend_rows(grid: np.ndarray) -> list[int]:
    legend_mask = (grid != BG) & (grid != OBJ)
    coords = np.argwhere(legend_mask)
    assert len(coords) > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    assert r1 - r0 + 1 == 4
    assert (r0 == 0 or r1 == grid.shape[0] - 1) and (c0 == 0 or c1 == grid.shape[1] - 1)
    block = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(legend_mask[r0 : r1 + 1, c0 : c1 + 1])

    row_colors = []
    for row in block:
        colors = np.unique(row)
        assert len(colors) == 1
        row_colors.append(int(colors[0]))
    assert len(set(row_colors)) == 4
    return row_colors


def _touch_category(coords: np.ndarray, shape: tuple[int, int]) -> int:
    h, w = shape
    rows = coords[:, 0]
    cols = coords[:, 1]
    touches = {
        "top": bool(np.any(rows == 0)),
        "bottom": bool(np.any(rows == h - 1)),
        "left": bool(np.any(cols == 0)),
        "right": bool(np.any(cols == w - 1)),
    }
    sides = [name for name, flag in touches.items() if flag]
    if len(sides) == 0:
        return CAT_NONE
    if len(sides) == 1:
        return CAT_ONE
    assert len(sides) == 2
    if {"top", "bottom"} == set(sides) or {"left", "right"} == set(sides):
        return CAT_OPPOSITE
    return CAT_CORNER


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each grey object by its grid-edge contact pattern using the corner legend."""
    g = np.asarray(grid, dtype=int)
    palette = _legend_rows(g)
    out = g.copy()
    for comp in _components(g == OBJ):
        category = _touch_category(comp, g.shape)
        out[tuple(comp.T)] = palette[category]
    return out


# --------------- generator ---------------

GRID_H_RANGE = (13, 18)
GRID_W_RANGE = (13, 18)
LEGEND_WIDTH_RANGE = (2, 3)
EXTRA_OBJECTS_RANGE = (0, 2)
PLACE_TRIES = 120
GEN_RETRIES = 300
PALETTE_POOL = [1, 2, 3, 4, 6, 8, 9]
SIDES = ("top", "bottom", "left", "right")
CORNERS = ("tl", "tr", "bl", "br")


def _randint(rng: random.Random, lo: int, hi: int) -> int | None:
    if lo > hi:
        return None
    return rng.randint(lo, hi)


def _corner_rect(h: int, w: int, corner: str, box_h: int, box_w: int) -> tuple[int, int]:
    r0 = 0 if corner in ("tl", "tr") else h - box_h
    c0 = 0 if corner in ("tl", "bl") else w - box_w
    return r0, c0


def _mark_occupied(occupied: np.ndarray, cells: list[tuple[int, int]], pad: int = 1) -> None:
    h, w = occupied.shape
    for r, c in cells:
        r0 = max(0, r - pad)
        r1 = min(h, r + pad + 1)
        c0 = max(0, c - pad)
        c1 = min(w, c + pad + 1)
        occupied[r0:r1, c0:c1] = True


def _random_path(
    start: tuple[int, int], goal: tuple[int, int], rng: random.Random
) -> list[tuple[int, int]]:
    r, c = start
    gr, gc = goal
    out = []
    while (r, c) != (gr, gc):
        steps = []
        if r < gr:
            steps.append((r + 1, c))
        elif r > gr:
            steps.append((r - 1, c))
        if c < gc:
            steps.append((r, c + 1))
        elif c > gc:
            steps.append((r, c - 1))
        r, c = rng.choice(steps)
        out.append((r, c))
    return out


def _build_mask(
    box_h: int, box_w: int, anchors: list[tuple[int, int]], rng: random.Random
) -> np.ndarray:
    cells = {anchors[0]}
    for anchor in anchors[1:]:
        if anchor in cells:
            continue
        target = rng.choice(tuple(cells))
        cells.add(anchor)
        for step in _random_path(anchor, target, rng):
            cells.add(step)

    total = box_h * box_w
    min_area = len(cells)
    low = max(min_area, int(np.ceil(total * 0.35)))
    high = max(low, int(np.floor(total * 0.65)))
    target_area = rng.randint(low, high)
    while len(cells) < target_area:
        frontier = set()
        for r, c in cells:
            for dr, dc in DIR4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            break
        cells.add(rng.choice(tuple(frontier)))

    mask = np.zeros((box_h, box_w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _sample_none(rng: random.Random, h: int, w: int):
    box_h = _randint(rng, 2, min(5, h - 2))
    box_w = _randint(rng, 2, min(5, w - 2))
    if box_h is None or box_w is None:
        return None
    r0 = _randint(rng, 1, h - box_h - 1)
    c0 = _randint(rng, 1, w - box_w - 1)
    if r0 is None or c0 is None:
        return None
    anchors = [(rng.randrange(box_h), rng.randrange(box_w))]
    return r0, c0, box_h, box_w, anchors


def _sample_one(rng: random.Random, h: int, w: int):
    side = rng.choice(SIDES)
    if side in ("top", "bottom"):
        box_h = _randint(rng, 2, min(5, h - 1))
        box_w = _randint(rng, 2, min(5, w - 2))
        if box_h is None or box_w is None:
            return None
        c0 = _randint(rng, 1, w - box_w - 1)
        if c0 is None:
            return None
        r0 = 0 if side == "top" else h - box_h
        anchors = [(0 if side == "top" else box_h - 1, rng.randrange(box_w))]
    else:
        box_h = _randint(rng, 2, min(5, h - 2))
        box_w = _randint(rng, 2, min(5, w - 1))
        if box_h is None or box_w is None:
            return None
        r0 = _randint(rng, 1, h - box_h - 1)
        if r0 is None:
            return None
        c0 = 0 if side == "left" else w - box_w
        anchors = [(rng.randrange(box_h), 0 if side == "left" else box_w - 1)]
    return r0, c0, box_h, box_w, anchors


def _sample_corner(rng: random.Random, h: int, w: int, forbidden_corner: str):
    choices = [corner for corner in CORNERS if corner != forbidden_corner]
    corner = rng.choice(choices)
    box_h = _randint(rng, 2, min(5, h - 1))
    box_w = _randint(rng, 2, min(5, w - 1))
    if box_h is None or box_w is None:
        return None
    r0, c0 = _corner_rect(h, w, corner, box_h, box_w)
    if rng.random() < 0.5:
        anchors = {
            "tl": [(0, 0)],
            "tr": [(0, box_w - 1)],
            "bl": [(box_h - 1, 0)],
            "br": [(box_h - 1, box_w - 1)],
        }[corner]
    else:
        anchors = {
            "tl": [(0, rng.randrange(box_w)), (rng.randrange(box_h), 0)],
            "tr": [(0, rng.randrange(box_w)), (rng.randrange(box_h), box_w - 1)],
            "bl": [(box_h - 1, rng.randrange(box_w)), (rng.randrange(box_h), 0)],
            "br": [(box_h - 1, rng.randrange(box_w)), (rng.randrange(box_h), box_w - 1)],
        }[corner]
    return r0, c0, box_h, box_w, anchors


def _sample_opposite(rng: random.Random, h: int, w: int):
    orientation = rng.choice(("vertical", "horizontal"))
    if orientation == "vertical":
        box_h = h
        box_w = _randint(rng, 1, min(3, w - 2))
        if box_w is None:
            return None
        r0 = 0
        c0 = _randint(rng, 1, w - box_w - 1)
        if c0 is None:
            return None
        anchors = [(0, rng.randrange(box_w)), (box_h - 1, rng.randrange(box_w))]
    else:
        box_h = _randint(rng, 1, min(3, h - 2))
        if box_h is None:
            return None
        box_w = w
        r0 = _randint(rng, 1, h - box_h - 1)
        if r0 is None:
            return None
        c0 = 0
        anchors = [(rng.randrange(box_h), 0), (rng.randrange(box_h), box_w - 1)]
    return r0, c0, box_h, box_w, anchors


def _sample_spec(
    category: int, rng: random.Random, h: int, w: int, legend_corner: str
):
    if category == CAT_NONE:
        return _sample_none(rng, h, w)
    if category == CAT_ONE:
        return _sample_one(rng, h, w)
    if category == CAT_CORNER:
        return _sample_corner(rng, h, w, legend_corner)
    return _sample_opposite(rng, h, w)


def _try_place_object(
    grid: np.ndarray,
    occupied: np.ndarray,
    category: int,
    legend_corner: str,
    rng: random.Random,
) -> np.ndarray | None:
    h, w = grid.shape
    for _ in range(PLACE_TRIES):
        spec = _sample_spec(category, rng, h, w, legend_corner)
        if spec is None:
            continue
        r0, c0, box_h, box_w, anchors = spec
        if occupied[r0 : r0 + box_h, c0 : c0 + box_w].any():
            continue

        mask = _build_mask(box_h, box_w, anchors, rng)
        local = np.argwhere(mask)
        coords = local + np.array([r0, c0], dtype=int)
        actual_category = _touch_category(coords, grid.shape)
        if actual_category != category:
            continue
        cells = [tuple(map(int, xy)) for xy in coords]
        if any(occupied[r, c] for r, c in cells):
            continue

        grid[tuple(coords.T)] = OBJ
        _mark_occupied(occupied, cells)
        return coords
    return None


def _make_expected(
    grid: np.ndarray, objects: list[tuple[int, np.ndarray]], palette: list[int]
) -> np.ndarray:
    out = grid.copy()
    for category, coords in objects:
        out[tuple(coords.T)] = palette[category]
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    legend_corner = rng.choice(CORNERS)
    legend_w = rng.randint(*LEGEND_WIDTH_RANGE)
    palette = rng.sample(PALETTE_POOL, 4)

    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    legend_r0, legend_c0 = _corner_rect(h, w, legend_corner, 4, legend_w)
    for rr, color in enumerate(palette):
        grid[legend_r0 + rr, legend_c0 : legend_c0 + legend_w] = color
    legend_cells = [
        (legend_r0 + rr, legend_c0 + cc)
        for rr in range(4)
        for cc in range(legend_w)
    ]
    _mark_occupied(occupied, legend_cells)

    categories = [CAT_OPPOSITE, CAT_CORNER, CAT_ONE, CAT_NONE]
    extra_count = rng.randint(*EXTRA_OBJECTS_RANGE)
    extras = [rng.choice([CAT_NONE, CAT_ONE, CAT_CORNER, CAT_OPPOSITE]) for _ in range(extra_count)]
    extras.sort(reverse=True)
    categories.extend(extras)

    objects = []
    for i, category in enumerate(categories):
        coords = _try_place_object(grid, occupied, category, legend_corner, rng)
        if coords is None:
            if i < 4:
                return None
            continue
        objects.append((category, coords))

    expected = _make_expected(grid, objects, palette)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
