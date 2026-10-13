from __future__ import annotations

import random

import numpy as np


BG = 0
COLOR_POOL = (1, 2, 3, 4, 5, 6, 7, 8, 9)
OBJECT_COUNT_RANGE = (3, 5)
PATTERN_SIDE_OPTIONS = (3, 5, 7)
RECT_MARGIN_RANGE = (1, 2)
GRID_SIDE_RANGE = (18, 28)
PLACEMENT_GAP = 1
GEN_RETRIES = 400
PATTERN_RETRIES = 80


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_true(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _touches_edge(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = map(int, shape)
    rows = coords[:, 0]
    cols = coords[:, 1]
    return bool(
        np.any((rows == 0) | (rows == h - 1) | (cols == 0) | (cols == w - 1))
    )


def _hole_count(mask: np.ndarray) -> int:
    holes = 0
    for comp in _components(~mask):
        if not _touches_edge(comp, mask.shape):
            holes += 1
    return holes


def _extract_layers(grid: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    layers: list[tuple[int, int, np.ndarray]] = []

    for color in sorted(int(c) for c in np.unique(g) if int(c) != BG):
        comps = _components(g == color)
        assert len(comps) == 1, "each non-background color must form one rectangle"
        r0, c0, r1, c1 = _bbox_from_coords(comps[0])
        sub = g[r0 : r1 + 1, c0 : c1 + 1]

        assert np.all((sub == BG) | (sub == color))
        assert np.all(sub[0, :] == color)
        assert np.all(sub[-1, :] == color)
        assert np.all(sub[:, 0] == color)
        assert np.all(sub[:, -1] == color)

        hole = sub == BG
        assert hole.any(), "each rectangle must contain one hidden background pattern"
        assert not hole[0, :].any()
        assert not hole[-1, :].any()
        assert not hole[:, 0].any()
        assert not hole[:, -1].any()

        hole_comps = _components(hole)
        assert len(hole_comps) == 1, "hidden pattern must be connected"
        pattern = _crop_true(hole)
        layers.append((int(sub.shape[0] * sub.shape[1]), int(color), pattern))

    assert 3 <= len(layers) <= 5
    areas = [area for area, _color, _pattern in layers]
    assert len(areas) == len(set(areas)), "rectangle areas must be unique"
    return layers


def _crop_nonzero(grid: np.ndarray) -> np.ndarray:
    coords = np.argwhere(grid != BG)
    assert coords.size
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the black pattern hidden inside each colored rectangle and nest the
    recolored patterns by descending rectangle area with aligned centers.
    """
    layers = _extract_layers(grid)
    layers.sort(key=lambda item: item[0], reverse=True)

    max_h = max(int(pattern.shape[0]) for _area, _color, pattern in layers)
    max_w = max(int(pattern.shape[1]) for _area, _color, pattern in layers)
    canvas = np.zeros((max_h, max_w), dtype=int)

    for _area, color, pattern in layers:
        h, w = map(int, pattern.shape)
        r0 = (max_h - h) // 2
        c0 = (max_w - w) // 2
        window = canvas[r0 : r0 + h, c0 : c0 + w]
        window[pattern] = int(color)

    return _crop_nonzero(canvas)


def _path_between(
    start: tuple[int, int],
    end: tuple[int, int],
) -> list[tuple[int, int]]:
    r, c = start
    tr, tc = end
    cells = [(int(r), int(c))]
    row_first = random.choice([True, False])
    for axis in range(2):
        use_row = row_first if axis == 0 else not row_first
        if use_row:
            step = 1 if tr > r else -1
            while r != tr:
                r += step
                cells.append((int(r), int(c)))
        else:
            step = 1 if tc > c else -1
            while c != tc:
                c += step
                cells.append((int(r), int(c)))
    return cells


def _plus_pattern(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[:, w // 2] = True
    mask[h // 2, :] = True
    if h >= 5 and random.random() < 0.5:
        side = random.choice(["top", "bottom", "left", "right"])
        if side == "top":
            mask[1 : h // 2, random.randint(0, w - 1)] = True
        elif side == "bottom":
            mask[h // 2 + 1 : h - 1, random.randint(0, w - 1)] = True
        elif side == "left":
            mask[random.randint(0, h - 1), 1 : w // 2] = True
        else:
            mask[random.randint(0, h - 1), w // 2 + 1 : w - 1] = True
    return mask


def _tee_pattern(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    orientation = random.choice(["up", "down", "left", "right"])
    if orientation == "up":
        mask[0, :] = True
        mask[:, w // 2] = True
    elif orientation == "down":
        mask[-1, :] = True
        mask[:, w // 2] = True
    elif orientation == "left":
        mask[:, 0] = True
        mask[h // 2, :] = True
    else:
        mask[:, -1] = True
        mask[h // 2, :] = True
    return mask


def _ell_pattern(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    corner = random.choice(["tl", "tr", "bl", "br"])
    if corner == "tl":
        mask[:, 0] = True
        mask[-1, :] = True
    elif corner == "tr":
        mask[:, -1] = True
        mask[-1, :] = True
    elif corner == "bl":
        mask[:, 0] = True
        mask[0, :] = True
    else:
        mask[:, -1] = True
        mask[0, :] = True
    return mask


def _stairs_pattern(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    left_to_right = random.choice([True, False])
    c = 0 if left_to_right else w - 1
    target_c = w - 1 if left_to_right else 0
    step_c = 1 if left_to_right else -1

    mask[0, c] = True
    for r in range(h - 1):
        if c != target_c:
            run = random.randint(1, min(2, abs(target_c - c)))
            for _ in range(run):
                c += step_c
                mask[r, c] = True
                if c == target_c:
                    break
        mask[r + 1, c] = True

    while c != target_c:
        c += step_c
        mask[h - 1, c] = True
    return mask


def _organic_pattern(h: int, w: int) -> np.ndarray:
    center = (h // 2, w // 2)
    cells = {center}
    targets = [
        (0, random.randint(0, w - 1)),
        (h - 1, random.randint(0, w - 1)),
        (random.randint(0, h - 1), 0),
        (random.randint(0, h - 1), w - 1),
    ]
    for target in targets:
        cells.update(_path_between(center, target))

    limit = min(h * w - 1, max(len(cells), int(round(h * w * random.uniform(0.35, 0.6)))))
    target_area = random.randint(len(cells), limit)
    while len(cells) < target_area:
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for nr, nc in _neighbors4(r, c, h, w):
                if (nr, nc) not in cells:
                    frontier.add((int(nr), int(nc)))
        if not frontier:
            break
        cells.add(random.choice(tuple(frontier)))

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _random_pattern() -> np.ndarray:
    families = [
        _plus_pattern,
        _tee_pattern,
        _ell_pattern,
        _stairs_pattern,
        _organic_pattern,
    ]
    for _ in range(PATTERN_RETRIES):
        h = random.choice(PATTERN_SIDE_OPTIONS)
        w = random.choice(PATTERN_SIDE_OPTIONS)
        mask = random.choice(families)(h, w)
        if random.random() < 0.5:
            mask = np.flip(mask, axis=0)
        if random.random() < 0.5:
            mask = np.flip(mask, axis=1)
        mask = _crop_true(mask)
        if len(_components(mask)) != 1:
            continue
        if _hole_count(mask) != 0:
            continue
        if mask.shape[0] not in PATTERN_SIDE_OPTIONS or mask.shape[1] not in PATTERN_SIDE_OPTIONS:
            continue
        return mask
    raise RuntimeError("failed to generate a pattern")


def _place_rectangles(rects: list[dict]) -> np.ndarray | None:
    total_area = sum(int(rect["height"] * rect["width"]) for rect in rects)
    max_h = max(int(rect["height"]) for rect in rects)
    max_w = max(int(rect["width"]) for rect in rects)
    layout = random.choice(["balanced", "wide", "tall"])
    base = int(np.ceil(np.sqrt(total_area * random.uniform(1.8, 2.3))))

    if layout == "wide":
        h = random.randint(max(max_h + 4, base - 4), max(max_h + 6, base + 2))
        w = random.randint(max(max_w + 8, base + 2), max(max_w + 10, base + 8))
    elif layout == "tall":
        h = random.randint(max(max_h + 8, base + 2), max(max_h + 10, base + 8))
        w = random.randint(max(max_w + 4, base - 4), max(max_w + 6, base + 2))
    else:
        h = random.randint(max(max_h + 4, base - 2), max(max_h + 7, base + 4))
        w = random.randint(max(max_w + 4, base - 2), max(max_w + 7, base + 4))

    h = max(GRID_SIDE_RANGE[0], min(GRID_SIDE_RANGE[1], h))
    w = max(GRID_SIDE_RANGE[0], min(GRID_SIDE_RANGE[1], w))
    if h < max_h + 2 or w < max_w + 2:
        return None

    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    order = rects[:]
    random.shuffle(order)

    for rect in order:
        rh = int(rect["height"])
        rw = int(rect["width"])
        placed = False
        for _ in range(120):
            r0 = random.randint(0, h - rh)
            c0 = random.randint(0, w - rw)
            rr0 = max(0, r0 - PLACEMENT_GAP)
            cc0 = max(0, c0 - PLACEMENT_GAP)
            rr1 = min(h, r0 + rh + PLACEMENT_GAP)
            cc1 = min(w, c0 + rw + PLACEMENT_GAP)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue

            color = int(rect["color"])
            grid[r0 : r0 + rh, c0 : c0 + rw] = color
            ph, pw = map(int, rect["pattern"].shape)
            ir = r0 + int(rect["top"])
            ic = c0 + int(rect["left"])
            grid[ir : ir + ph, ic : ic + pw][rect["pattern"]] = BG
            occupied[r0 : r0 + rh, c0 : c0 + rw] = True
            placed = True
            break
        if not placed:
            return None

    return grid


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(*OBJECT_COUNT_RANGE)
    colors = random.sample(COLOR_POOL, n)
    used_areas: set[int] = set()
    rects: list[dict] = []

    for color in colors:
        placed = False
        for _ in range(80):
            pattern = _random_pattern()
            ph, pw = map(int, pattern.shape)
            top = random.randint(*RECT_MARGIN_RANGE)
            bottom = random.randint(*RECT_MARGIN_RANGE)
            left = random.randint(*RECT_MARGIN_RANGE)
            right = random.randint(*RECT_MARGIN_RANGE)
            rh = ph + top + bottom
            rw = pw + left + right
            area = int(rh * rw)
            if area in used_areas:
                continue
            used_areas.add(area)
            rects.append(
                {
                    "color": int(color),
                    "pattern": pattern,
                    "top": int(top),
                    "left": int(left),
                    "height": int(rh),
                    "width": int(rw),
                }
            )
            placed = True
            break
        if not placed:
            return None

    grid = _place_rectangles(rects)
    if grid is None:
        return None

    try:
        layers = _extract_layers(grid)
        out = main(grid)
    except AssertionError:
        return None

    if len(layers) != n or out.size == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
