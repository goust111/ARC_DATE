import random
from collections import Counter, deque

import numpy as np


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert len(coords) > 0
    r0, c0, r1, c1 = bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _mask_has_hole(mask: np.ndarray) -> bool:
    h, w = map(int, mask.shape)
    open_mask = ~mask
    seen = np.zeros((h, w), dtype=bool)
    q = deque()
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            if open_mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and open_mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return bool(np.any(open_mask & ~seen))


def _object_has_hole(grid: np.ndarray, bg: int, coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = bbox_coords(coords)
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    return _mask_has_hole(sub != bg)


def _extract_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = get_bg(g)
    objects = []
    for coords in components_mask(g != bg):
        colors = np.unique(g[coords[:, 0], coords[:, 1]])
        assert colors.size == 1, "each foreground object must be monochrome"
        color = int(colors[0])
        r0, c0, r1, c1 = bbox_coords(coords)
        sprite = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
        rel = coords - np.array([r0, c0], dtype=int)
        sprite[rel[:, 0], rel[:, 1]] = color
        objects.append(
            {
                "color": color,
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "sprite": sprite,
                "area": int(len(coords)),
                "has_hole": _object_has_hole(g, bg, coords),
            }
        )
    return objects


def _sort_key(obj: dict) -> tuple[int, int, int, int]:
    r0, c0, _, _ = obj["bbox"]
    h, w = obj["sprite"].shape
    return int(h * w), int(r0), int(c0), int(obj["color"])


def _band_size(objects: list[dict]) -> tuple[int, int]:
    if not objects:
        return 1, 1
    height = max(int(obj["sprite"].shape[0]) for obj in objects)
    width = sum(int(obj["sprite"].shape[1]) for obj in objects) + max(0, len(objects) - 1)
    return height, max(1, width)


def _paste_band(out: np.ndarray, row0: int, objects: list[dict]):
    col = 0
    for obj in objects:
        sprite = obj["sprite"]
        h, w = map(int, sprite.shape)
        out[row0 : row0 + h, col : col + w] = sprite
        col += w + 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group scattered monochrome objects by whether they contain enclosed holes.

    Hollow objects are placed on the first row band from left to right in
    descending area order. Solid objects are placed on the second row band in
    ascending area order. Objects keep their original shapes and colors.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    objects = _extract_objects(g)
    assert objects, "expected at least one foreground object"

    hollow = [obj for obj in objects if obj["has_hole"]]
    solid = [obj for obj in objects if not obj["has_hole"]]

    hollow.sort(key=lambda obj: (-int(obj["area"]),) + _sort_key(obj))
    solid.sort(key=lambda obj: (int(obj["area"]),) + _sort_key(obj))

    top_h, top_w = _band_size(hollow)
    bottom_h, bottom_w = _band_size(solid)
    out_h = top_h + 1 + bottom_h
    out_w = max(top_w, bottom_w)
    out = np.full((out_h, out_w), bg, dtype=int)

    _paste_band(out, 0, hollow)
    _paste_band(out, top_h + 1, solid)
    return out


MAX_GENERATION_ATTEMPTS = 300
GRID_HEIGHT_RANGE = (18, 29)
GRID_WIDTH_RANGE = (18, 29)
HOLLOW_COUNT_RANGE = (1, 4)
SOLID_COUNT_RANGE = (1, 4)
OBJECT_GAP = 1
PLACE_RETRIES = 150


def _random_transform(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    out = np.array(mask, dtype=bool)
    out = np.rot90(out, rng.randrange(4))
    if rng.random() < 0.5:
        out = np.fliplr(out)
    if rng.random() < 0.5:
        out = np.flipud(out)
    return _crop_mask(out)


def _grow_polyomino(rng: random.Random, min_area: int, max_area: int) -> np.ndarray | None:
    target = rng.randint(min_area, max_area)
    h = rng.randint(4, 8)
    w = rng.randint(4, 8)
    start = (rng.randrange(h), rng.randrange(w))
    cells = {start}
    for _ in range(target * 6):
        if len(cells) >= target:
            break
        frontier = []
        for r, c in cells:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(rng.choice(frontier))
    if len(cells) != target:
        return None
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    mask = _crop_mask(mask)
    if min(mask.shape) < 2 or max(mask.shape) > 7:
        return None
    if _mask_has_hole(mask):
        return None
    return _random_transform(mask, rng)


def _solid_mask(rng: random.Random) -> np.ndarray | None:
    for _ in range(80):
        mode = rng.choice(["poly", "rect", "notch"])
        if mode == "poly":
            mask = _grow_polyomino(rng, 4, 13)
        elif mode == "rect":
            h = rng.randint(2, 5)
            w = rng.randint(2, 6)
            mask = np.ones((h, w), dtype=bool)
            mask = _random_transform(mask, rng)
        else:
            h = rng.randint(3, 6)
            w = rng.randint(3, 6)
            mask = np.ones((h, w), dtype=bool)
            cut_h = rng.randint(1, h - 1)
            cut_w = rng.randint(1, w - 1)
            corner = rng.choice([(0, 0), (0, w - cut_w), (h - cut_h, 0), (h - cut_h, w - cut_w)])
            r0, c0 = corner
            mask[r0 : r0 + cut_h, c0 : c0 + cut_w] = False
            if not np.any(mask):
                continue
            if len(components_mask(mask)) != 1 or _mask_has_hole(mask):
                continue
            mask = _crop_mask(mask)
            mask = _random_transform(mask, rng)
        if mask is None:
            continue
        if len(components_mask(mask)) == 1 and not _mask_has_hole(mask):
            return mask
    return None


def _attach_protrusion(canvas: np.ndarray, side: str, rng: random.Random):
    rows, cols = np.where(canvas)
    r0, c0, r1, c1 = int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())
    if side == "top":
        base_c = rng.randint(c0, c1)
        length = rng.randint(1, 2)
        width = rng.randint(1, min(3, c1 - c0 + 1))
        left = max(0, min(base_c - width // 2, canvas.shape[1] - width))
        top = max(0, r0 - length)
        canvas[top:r0, left : left + width] = True
    elif side == "bottom":
        base_c = rng.randint(c0, c1)
        length = rng.randint(1, 2)
        width = rng.randint(1, min(3, c1 - c0 + 1))
        left = max(0, min(base_c - width // 2, canvas.shape[1] - width))
        bottom = min(canvas.shape[0], r1 + 1 + length)
        canvas[r1 + 1 : bottom, left : left + width] = True
    elif side == "left":
        base_r = rng.randint(r0, r1)
        length = rng.randint(1, 2)
        width = rng.randint(1, min(3, r1 - r0 + 1))
        top = max(0, min(base_r - width // 2, canvas.shape[0] - width))
        left = max(0, c0 - length)
        canvas[top : top + width, left:c0] = True
    else:
        base_r = rng.randint(r0, r1)
        length = rng.randint(1, 2)
        width = rng.randint(1, min(3, r1 - r0 + 1))
        top = max(0, min(base_r - width // 2, canvas.shape[0] - width))
        right = min(canvas.shape[1], c1 + 1 + length)
        canvas[top : top + width, c1 + 1 : right] = True


def _hollow_mask(rng: random.Random) -> np.ndarray | None:
    for _ in range(80):
        outer_h = rng.randint(4, 8)
        outer_w = rng.randint(4, 8)
        hole_h = rng.randint(1, outer_h - 2)
        hole_w = rng.randint(1, outer_w - 2)
        if hole_h >= outer_h or hole_w >= outer_w:
            continue
        hole_r = rng.randint(1, outer_h - hole_h - 1)
        hole_c = rng.randint(1, outer_w - hole_w - 1)

        canvas = np.zeros((outer_h + 4, outer_w + 4), dtype=bool)
        canvas[2 : 2 + outer_h, 2 : 2 + outer_w] = True
        canvas[2 + hole_r : 2 + hole_r + hole_h, 2 + hole_c : 2 + hole_c + hole_w] = False
        for _ in range(rng.randint(0, 3)):
            _attach_protrusion(canvas, rng.choice(["top", "bottom", "left", "right"]), rng)
        mask = _crop_mask(canvas)
        mask = _random_transform(mask, rng)
        if len(components_mask(mask)) != 1:
            continue
        if not _mask_has_hole(mask):
            continue
        return mask
    return None


def _make_object(kind: str, color: int, rng: random.Random) -> dict | None:
    mask = _hollow_mask(rng) if kind == "hollow" else _solid_mask(rng)
    if mask is None:
        return None
    return {
        "kind": kind,
        "color": color,
        "mask": mask,
        "area": int(mask.sum()),
    }


def _place_objects(grid: np.ndarray, objects: list[dict], bg: int, rng: random.Random) -> bool:
    h, w = map(int, grid.shape)
    blocked = np.zeros((h, w), dtype=bool)
    for obj in sorted(objects, key=lambda item: int(item["area"]), reverse=True):
        mask = obj["mask"]
        oh, ow = map(int, mask.shape)
        placed = False
        if oh > h or ow > w:
            return False
        for _ in range(PLACE_RETRIES):
            r0 = rng.randint(0, h - oh)
            c0 = rng.randint(0, w - ow)
            rr0 = max(0, r0 - OBJECT_GAP)
            cc0 = max(0, c0 - OBJECT_GAP)
            rr1 = min(h, r0 + oh + OBJECT_GAP)
            cc1 = min(w, c0 + ow + OBJECT_GAP)
            if blocked[rr0:rr1, cc0:cc1].any():
                continue
            view = grid[r0 : r0 + oh, c0 : c0 + ow]
            view[mask] = obj["color"]
            blocked[rr0:rr1, cc0:cc1] = True
            placed = True
            break
        if not placed:
            return False
    assert np.all((grid == bg) | ~blocked | (grid != bg))
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = rng.randrange(10)
    n_hollow = rng.randint(*HOLLOW_COUNT_RANGE)
    n_solid = rng.randint(*SOLID_COUNT_RANGE)
    total = n_hollow + n_solid
    colors = rng.sample([c for c in range(10) if c != bg], total)

    objects = []
    for idx in range(n_hollow):
        obj = _make_object("hollow", colors[idx], rng)
        if obj is None:
            return None
        objects.append(obj)
    for idx in range(n_solid):
        obj = _make_object("solid", colors[n_hollow + idx], rng)
        if obj is None:
            return None
        objects.append(obj)

    hollow_areas = [obj["area"] for obj in objects if obj["kind"] == "hollow"]
    solid_areas = [obj["area"] for obj in objects if obj["kind"] == "solid"]
    if len(hollow_areas) != len(set(hollow_areas)) or len(solid_areas) != len(set(solid_areas)):
        return None

    max_h = max(int(obj["mask"].shape[0]) for obj in objects)
    max_w = max(int(obj["mask"].shape[1]) for obj in objects)
    min_h = max(GRID_HEIGHT_RANGE[0], max_h + 4)
    min_w = max(GRID_WIDTH_RANGE[0], max_w + 4)
    if min_h > GRID_HEIGHT_RANGE[1] or min_w > GRID_WIDTH_RANGE[1]:
        return None

    h = rng.randint(min_h, GRID_HEIGHT_RANGE[1])
    w = rng.randint(min_w, GRID_WIDTH_RANGE[1])
    grid = np.full((h, w), bg, dtype=int)
    if not _place_objects(grid, objects, bg, rng):
        return None

    extracted = _extract_objects(grid)
    if len(extracted) != total:
        return None
    if not any(obj["has_hole"] for obj in extracted):
        return None
    if not any(not obj["has_hole"] for obj in extracted):
        return None

    ex_hollow = [obj["area"] for obj in extracted if obj["has_hole"]]
    ex_solid = [obj["area"] for obj in extracted if not obj["has_hole"]]
    if len(ex_hollow) != len(set(ex_hollow)) or len(ex_solid) != len(set(ex_solid)):
        return None

    _ = main(grid)
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
