from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
DIVIDER = 5
OBJECT_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
PAD8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 0),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

LR_ONLY = "lr_only"   # symmetric under left-right reflection only
TB_ONLY = "tb_only"   # symmetric under top-bottom reflection only
BOTH = "both"
NONE = "none"

GEN_RETRIES = 300
SHAPE_RETRIES = 120


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.asarray(coords, dtype=int))
    return parts


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    h, w = mask.shape
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _classify_mask(mask: np.ndarray) -> str:
    left_right = np.array_equal(mask, np.fliplr(mask))
    top_bottom = np.array_equal(mask, np.flipud(mask))
    if left_right and top_bottom:
        return BOTH
    if left_right:
        return LR_ONLY
    if top_bottom:
        return TB_ONLY
    return NONE


def _extract_objects(tl: np.ndarray, bg: int) -> list[dict[str, object]]:
    objects: list[dict[str, object]] = []
    for coords in _components(tl != bg):
        rows = coords[:, 0]
        cols = coords[:, 1]
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        colors = {int(tl[r, c]) for r, c in coords.tolist()}
        assert len(colors) == 1, "each object must be monochromatic"
        color = colors.pop()
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        mask[rows - r0, cols - c0] = True
        objects.append(
            {
                "r0": r0,
                "c0": c0,
                "color": color,
                "mask": mask,
                "category": _classify_mask(mask),
            }
        )
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route each top-left object by its reflection symmetry.

    The prompt's parenthetical definitions are used directly:
    - left-right symmetric only -> bottom-left, flipped top-bottom
    - top-bottom symmetric only -> bottom-right, flipped left-right
    - symmetric in both directions -> top-right, unchanged
    - no symmetry -> stays in the top-left quadrant
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    h, w = map(int, g.shape)
    assert h % 2 == 1 and w % 2 == 1, "grid must have a central divider row and column"

    mid_r, mid_c = h // 2, w // 2
    bg = _dominant_color(g)
    divider = int(g[mid_r, 0])
    assert divider != bg, "divider must differ from the background"
    assert np.all(g[mid_r, :] == divider), "expected a full divider row"
    assert np.all(g[:, mid_c] == divider), "expected a full divider column"
    assert divider == int(g[0, mid_c]), "divider cross must have one color"

    tl = g[:mid_r, :mid_c]
    objects = _extract_objects(tl, bg)
    assert objects, "top-left quadrant must contain at least one object"

    out = np.full_like(g, bg)
    out[mid_r, :] = divider
    out[:, mid_c] = divider

    quadrant_origin = {
        NONE: (0, 0),
        BOTH: (0, mid_c + 1),
        LR_ONLY: (mid_r + 1, 0),
        TB_ONLY: (mid_r + 1, mid_c + 1),
    }

    for obj in objects:
        mask = obj["mask"]
        color = int(obj["color"])
        r0 = int(obj["r0"])
        c0 = int(obj["c0"])
        category = str(obj["category"])

        patch = np.full(mask.shape, bg, dtype=int)
        patch[mask] = color
        if category == LR_ONLY:
            patch = np.flipud(patch)
        elif category == TB_ONLY:
            patch = np.fliplr(patch)

        qr0, qc0 = quadrant_origin[category]
        pr0 = qr0 + r0
        pc0 = qc0 + c0
        target = out[pr0 : pr0 + patch.shape[0], pc0 : pc0 + patch.shape[1]]
        stamp = patch != bg
        assert target.shape == patch.shape, "object placement must stay inside its destination quadrant"
        assert not np.any((target != bg) & stamp), "objects may not overlap after routing"
        target[stamp] = patch[stamp]

    return out


def _orbit_cells(h: int, w: int, r: int, c: int, category: str) -> tuple[tuple[int, int], ...]:
    rows = {r}
    cols = {c}
    if category in {TB_ONLY, BOTH}:
        rows.add(h - 1 - r)
    if category in {LR_ONLY, BOTH}:
        cols.add(w - 1 - c)
    return tuple(sorted((rr, cc) for rr in rows for cc in cols))


def _orbit_catalog(h: int, w: int, category: str) -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[tuple[int, int], ...]] = set()
    orbits: list[tuple[tuple[int, int], ...]] = []
    for r in range(h):
        for c in range(w):
            orbit = _orbit_cells(h, w, r, c, category)
            if orbit not in seen:
                seen.add(orbit)
                orbits.append(orbit)
    return orbits


def _cells_connected(cells: tuple[tuple[int, int], ...]) -> bool:
    start = cells[0]
    cell_set = set(cells)
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            nxt = (r + dr, c + dc)
            if nxt in cell_set and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(cells)


def _orbit_touches(mask: np.ndarray, orbit: tuple[tuple[int, int], ...]) -> bool:
    h, w = mask.shape
    for r, c in orbit:
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                return True
    return False


def _paint_orbit(mask: np.ndarray, orbit: tuple[tuple[int, int], ...]) -> None:
    for r, c in orbit:
        mask[r, c] = True


def _generate_symmetric_mask(category: str) -> np.ndarray:
    for _ in range(SHAPE_RETRIES):
        h = random.randint(3, 6)
        w = random.randint(3, 6)
        if category == LR_ONLY:
            w = random.randint(3, 7)
        elif category == TB_ONLY:
            h = random.randint(3, 7)

        orbits = _orbit_catalog(h, w, category)
        starts = [orbit for orbit in orbits if _cells_connected(orbit)]
        if not starts:
            continue

        mask = np.zeros((h, w), dtype=bool)
        used: set[tuple[tuple[int, int], ...]] = set()
        start = random.choice(starts)
        _paint_orbit(mask, start)
        used.add(start)

        target_area = random.randint(4, min(12, h * w))
        min_orbits = 2 if category != BOTH else 1
        while True:
            candidates = [orbit for orbit in orbits if orbit not in used and _orbit_touches(mask, orbit)]
            if not candidates:
                break
            if int(mask.sum()) >= target_area and len(used) >= min_orbits and random.random() < 0.7:
                break
            orbit = random.choice(candidates)
            _paint_orbit(mask, orbit)
            used.add(orbit)

        cropped = _crop_mask(mask)
        if int(cropped.sum()) < 3 or min(cropped.shape) < 2:
            continue
        if not _is_connected(cropped):
            continue
        if _classify_mask(cropped) != category:
            continue
        return cropped
    raise RuntimeError(f"failed to build a {category} object")


def _generate_asymmetric_mask() -> np.ndarray:
    for _ in range(SHAPE_RETRIES):
        h = random.randint(3, 7)
        w = random.randint(3, 7)
        target_area = random.randint(4, min(10, h * w - 1))
        mask = np.zeros((h, w), dtype=bool)
        start = (random.randrange(h), random.randrange(w))
        mask[start] = True

        while int(mask.sum()) < target_area:
            frontier: set[tuple[int, int]] = set()
            for r, c in np.argwhere(mask):
                rr = int(r)
                cc = int(c)
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        frontier.add((nr, nc))
            if not frontier:
                break
            nr, nc = random.choice(tuple(frontier))
            mask[nr, nc] = True

        cropped = _crop_mask(mask)
        if int(cropped.sum()) < 3 or min(cropped.shape) < 2:
            continue
        if _classify_mask(cropped) != NONE:
            continue
        return cropped
    raise RuntimeError("failed to build an asymmetric object")


def _generate_mask(category: str) -> np.ndarray:
    if category == NONE:
        return _generate_asymmetric_mask()
    return _generate_symmetric_mask(category)


def _can_place(mask: np.ndarray, blocked: np.ndarray, r0: int, c0: int) -> bool:
    for r, c in np.argwhere(mask):
        rr = r0 + int(r)
        cc = c0 + int(c)
        if blocked[rr, cc]:
            return False
    return True


def _place_with_padding(canvas: np.ndarray, blocked: np.ndarray, mask: np.ndarray, color: int) -> bool:
    h, w = canvas.shape
    mh, mw = mask.shape
    positions = [(r, c) for r in range(h - mh + 1) for c in range(w - mw + 1)]
    random.shuffle(positions)
    for r0, c0 in positions:
        if not _can_place(mask, blocked, r0, c0):
            continue
        coords = [(r0 + int(r), c0 + int(c)) for r, c in np.argwhere(mask)]
        for rr, cc in coords:
            canvas[rr, cc] = color
        for rr, cc in coords:
            for dr, dc in PAD8:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w:
                    blocked[nr, nc] = True
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    extra_count = random.randint(0, 2)
    categories = [BOTH, LR_ONLY, TB_ONLY, NONE]
    if extra_count:
        categories.extend(random.sample([BOTH, LR_ONLY, TB_ONLY, NONE], k=extra_count))
    random.shuffle(categories)

    qh = random.randint(11 + extra_count, 14)
    qw = random.randint(11 + extra_count, 14)
    colors = random.sample(OBJECT_COLORS, k=len(categories))

    objects = []
    try:
        for color, category in zip(colors, categories, strict=False):
            objects.append({"color": color, "category": category, "mask": _generate_mask(category)})
    except RuntimeError:
        return None

    objects.sort(key=lambda obj: int(np.prod(obj["mask"].shape)), reverse=True)
    tl = np.full((qh, qw), BG, dtype=int)
    blocked = np.zeros((qh, qw), dtype=bool)
    for obj in objects:
        if not _place_with_padding(tl, blocked, obj["mask"], int(obj["color"])):
            return None

    grid = np.full((2 * qh + 1, 2 * qw + 1), BG, dtype=int)
    grid[qh, :] = DIVIDER
    grid[:, qw] = DIVIDER
    grid[:qh, :qw] = tl

    parsed = _extract_objects(tl, BG)
    if len(parsed) != len(objects):
        return None
    counts = Counter(str(obj["category"]) for obj in parsed)
    if any(counts[category] == 0 for category in [BOTH, LR_ONLY, TB_ONLY, NONE]):
        return None

    try:
        _ = main(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
