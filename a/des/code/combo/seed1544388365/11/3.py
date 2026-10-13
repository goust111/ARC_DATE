from __future__ import annotations

import random
from collections import Counter

import numpy as np


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
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
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _count_holes(mask: np.ndarray) -> int:
    h, w = mask.shape
    bg = ~mask
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if not bg[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            touches_border = False
            while stack:
                cr, cc = stack.pop()
                if cr == 0 or cr == h - 1 or cc == 0 or cc == w - 1:
                    touches_border = True
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            if not touches_border:
                holes += 1
    return holes


def _detect_cross(grid: np.ndarray, bg: int) -> tuple[int, int, int]:
    h, w = grid.shape
    row_candidates: list[tuple[int, int]] = []
    col_candidates: list[tuple[int, int]] = []

    for r in range(h):
        val = int(grid[r, 0])
        if val != bg and np.all(grid[r, :] == val):
            row_candidates.append((r, val))
    for c in range(w):
        val = int(grid[0, c])
        if val != bg and np.all(grid[:, c] == val):
            col_candidates.append((c, val))

    if not row_candidates or not col_candidates:
        raise AssertionError("expected one full horizontal line and one full vertical line")

    center_r = (h - 1) / 2.0
    center_c = (w - 1) / 2.0
    matches: list[tuple[float, int, int, int]] = []
    for r, row_color in row_candidates:
        for c, col_color in col_candidates:
            if row_color == col_color:
                score = abs(r - center_r) + abs(c - center_c)
                matches.append((score, r, c, row_color))

    if not matches:
        raise AssertionError("cross lines should share a common color")

    matches.sort()
    _score, row, col, color = matches[0]
    return int(row), int(col), int(color)


def _quadrant_name(coords: np.ndarray, cross_r: int, cross_c: int) -> str:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 < cross_r:
        vertical = "top"
    elif r0 > cross_r:
        vertical = "bottom"
    else:
        raise AssertionError("object should stay inside a single horizontal half")

    if c1 < cross_c:
        horizontal = "left"
    elif c0 > cross_c:
        horizontal = "right"
    else:
        raise AssertionError("object should stay inside a single vertical half")

    return f"{vertical}-{horizontal}"


def _shift_for_quadrant(quadrant: str, steps: int) -> tuple[int, int]:
    dr = steps if quadrant.startswith("top") else -steps
    dc = steps if quadrant.endswith("left") else -steps
    return dr, dc


def _extract_objects(grid: np.ndarray, bg: int, cross_r: int, cross_c: int, cross_color: int) -> list[dict[str, object]]:
    objects: list[dict[str, object]] = []
    for color in np.unique(grid):
        color = int(color)
        if color in {bg, cross_color}:
            continue
        for coords in _components(grid == color):
            r0, c0, r1, c1 = _bbox(coords)
            local = grid[r0 : r1 + 1, c0 : c1 + 1] == color
            quadrant = _quadrant_name(coords, cross_r, cross_c)
            objects.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "quadrant": quadrant,
                    "holes": _count_holes(local),
                }
            )
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each quadrant object diagonally toward the center by its hole count.

    Objects with more holes are drawn later, so they appear on top in overlaps.
    The gray cross stays in place and can be covered by moved object pixels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _bg_color(g)
    cross_r, cross_c, cross_color = _detect_cross(g, bg)
    objects = _extract_objects(g, bg, cross_r, cross_c, cross_color)
    assert len(objects) == 4, "expected one object per quadrant"

    quadrant_order = {
        "top-left": 0,
        "top-right": 1,
        "bottom-left": 2,
        "bottom-right": 3,
    }

    out = np.full_like(g, bg)
    out[cross_r, :] = cross_color
    out[:, cross_c] = cross_color

    objects.sort(key=lambda obj: (int(obj["holes"]), quadrant_order[str(obj["quadrant"])]))
    for obj in objects:
        steps = int(obj["holes"])
        dr, dc = _shift_for_quadrant(str(obj["quadrant"]), steps)
        coords = np.asarray(obj["coords"], dtype=int)
        for r, c in coords:
            rr = int(r) + dr
            cc = int(c) + dc
            if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]:
                out[rr, cc] = int(obj["color"])

    return out


BG = 0
CROSS = 5
OBJECT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GRID_SIDE_MIN = 22
GRID_SIDE_MAX = 30
MAX_HOLES = 3
GEN_ATTEMPTS = 400
OBJECT_ATTEMPTS = 200


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask.astype(int))


def _is_connected(mask: np.ndarray) -> bool:
    comps = _components(mask)
    return len(comps) == 1


def _rects_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    separated = ar1 + 1 <= br0 or br1 + 1 <= ar0 or ac1 + 1 <= bc0 or bc1 + 1 <= ac0
    return not separated


def _generate_object_mask(rng: random.Random, hole_count: int) -> np.ndarray | None:
    min_side = {0: 3, 1: 5, 2: 6, 3: 7}[hole_count]
    max_side = 7 if hole_count <= 2 else 8

    for _ in range(OBJECT_ATTEMPTS):
        base_h = rng.randint(min_side, max_side)
        base_w = rng.randint(min_side, max_side)
        canvas = np.zeros((base_h + 4, base_w + 4), dtype=bool)
        top = 2
        left = 2
        canvas[top : top + base_h, left : left + base_w] = True

        holes: list[tuple[int, int, int, int]] = []
        ok = True
        for _hole_idx in range(hole_count):
            placed = False
            for _ in range(80):
                hole_h = rng.randint(1, min(2, base_h - 2))
                hole_w = rng.randint(1, min(2, base_w - 2))
                hr0 = rng.randint(top + 1, top + base_h - hole_h - 1)
                hc0 = rng.randint(left + 1, left + base_w - hole_w - 1)
                rect = (hr0, hc0, hr0 + hole_h, hc0 + hole_w)
                if all(not _rects_too_close(rect, other) for other in holes):
                    holes.append(rect)
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if not ok:
            continue

        for r0, c0, r1, c1 in holes:
            canvas[r0:r1, c0:c1] = False

        for _ in range(rng.randint(0, 4)):
            side = rng.choice(("top", "bottom", "left", "right"))
            if side == "top":
                ext = rng.randint(1, top)
                span = rng.randint(1, base_w)
                c0 = rng.randint(left, left + base_w - span)
                canvas[top - ext : top, c0 : c0 + span] = True
            elif side == "bottom":
                ext = rng.randint(1, canvas.shape[0] - (top + base_h))
                span = rng.randint(1, base_w)
                c0 = rng.randint(left, left + base_w - span)
                canvas[top + base_h : top + base_h + ext, c0 : c0 + span] = True
            elif side == "left":
                ext = rng.randint(1, left)
                span = rng.randint(1, base_h)
                r0 = rng.randint(top, top + base_h - span)
                canvas[r0 : r0 + span, left - ext : left] = True
            else:
                ext = rng.randint(1, canvas.shape[1] - (left + base_w))
                span = rng.randint(1, base_h)
                r0 = rng.randint(top, top + base_h - span)
                canvas[r0 : r0 + span, left + base_w : left + base_w + ext] = True

        cropped = _crop_mask(canvas)
        if not _is_connected(cropped):
            continue
        if _count_holes(cropped) != hole_count:
            continue
        if cropped.shape[0] > 9 or cropped.shape[1] > 9:
            continue
        if int(cropped.sum()) < 6 + 2 * hole_count:
            continue
        return cropped

    return None


def _sample_offset(rng: random.Random, low: int, high: int, prefer_high: bool) -> int:
    if low == high:
        return low
    if rng.random() < 0.65:
        mid = (low + high) // 2
        if prefer_high:
            return rng.randint(mid, high)
        return rng.randint(low, mid)
    return rng.randint(low, high)


def _place_mask(
    rng: random.Random,
    mask: np.ndarray,
    quadrant: str,
    h: int,
    w: int,
    cross_r: int,
    cross_c: int,
) -> tuple[int, int] | None:
    mh, mw = map(int, mask.shape)
    outer_pad_r = rng.randint(0, 1)
    outer_pad_c = rng.randint(0, 1)
    inner_pad_r = rng.randint(0, 1)
    inner_pad_c = rng.randint(0, 1)

    if quadrant == "top-left":
        row_low = outer_pad_r
        row_high = cross_r - inner_pad_r - mh
        col_low = outer_pad_c
        col_high = cross_c - inner_pad_c - mw
        row = _sample_offset(rng, row_low, row_high, prefer_high=True) if row_low <= row_high else None
        col = _sample_offset(rng, col_low, col_high, prefer_high=True) if col_low <= col_high else None
    elif quadrant == "top-right":
        row_low = outer_pad_r
        row_high = cross_r - inner_pad_r - mh
        col_low = cross_c + 1 + inner_pad_c
        col_high = w - outer_pad_c - mw
        row = _sample_offset(rng, row_low, row_high, prefer_high=True) if row_low <= row_high else None
        col = _sample_offset(rng, col_low, col_high, prefer_high=False) if col_low <= col_high else None
    elif quadrant == "bottom-left":
        row_low = cross_r + 1 + inner_pad_r
        row_high = h - outer_pad_r - mh
        col_low = outer_pad_c
        col_high = cross_c - inner_pad_c - mw
        row = _sample_offset(rng, row_low, row_high, prefer_high=False) if row_low <= row_high else None
        col = _sample_offset(rng, col_low, col_high, prefer_high=True) if col_low <= col_high else None
    else:
        row_low = cross_r + 1 + inner_pad_r
        row_high = h - outer_pad_r - mh
        col_low = cross_c + 1 + inner_pad_c
        col_high = w - outer_pad_c - mw
        row = _sample_offset(rng, row_low, row_high, prefer_high=False) if row_low <= row_high else None
        col = _sample_offset(rng, col_low, col_high, prefer_high=False) if col_low <= col_high else None

    if row is None or col is None:
        return None
    return int(row), int(col)


def _moved_cells(obj: dict[str, object]) -> list[tuple[int, int]]:
    mask = np.asarray(obj["mask"], dtype=bool)
    top = int(obj["top"])
    left = int(obj["left"])
    dr, dc = _shift_for_quadrant(str(obj["quadrant"]), int(obj["holes"]))
    cells = []
    for r, c in np.argwhere(mask):
        cells.append((top + int(r) + dr, left + int(c) + dc))
    return cells


def _build_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randrange(GRID_SIDE_MIN, GRID_SIDE_MAX + 1, 2)
    w = rng.randrange(GRID_SIDE_MIN, GRID_SIDE_MAX + 1, 2)
    cross_r = rng.choice((h // 2 - 1, h // 2))
    cross_c = rng.choice((w // 2 - 1, w // 2))

    hole_counts = [rng.randint(0, MAX_HOLES) for _ in range(4)]
    if max(hole_counts) == 0 or 0 not in hole_counts or len(set(hole_counts)) == 1:
        return None

    quadrants = ["top-left", "top-right", "bottom-left", "bottom-right"]
    colors = rng.sample(list(OBJECT_COLORS), 4)
    used_shapes: set[tuple[tuple[int, ...], ...]] = set()
    objects: list[dict[str, object]] = []

    for quadrant, holes, color in zip(quadrants, hole_counts, colors):
        mask = _generate_object_mask(rng, holes)
        if mask is None:
            return None
        sig = _mask_signature(mask)
        if sig in used_shapes:
            return None
        used_shapes.add(sig)

        placement = _place_mask(rng, mask, quadrant, h, w, cross_r, cross_c)
        if placement is None:
            return None
        top, left = placement
        objects.append(
            {
                "quadrant": quadrant,
                "holes": holes,
                "color": color,
                "mask": mask,
                "top": top,
                "left": left,
            }
        )

    coverage: Counter[tuple[int, int]] = Counter()
    hole_layers: dict[tuple[int, int], list[int]] = {}
    overlap_cells = 0
    touches_cross = False
    for obj in objects:
        cells = _moved_cells(obj)
        for cell in cells:
            coverage[cell] += 1
            hole_layers.setdefault(cell, []).append(int(obj["holes"]))
            if cell[0] == cross_r or cell[1] == cross_c:
                touches_cross = True
    for cell, count in coverage.items():
        if count > 1:
            overlap_cells += 1
            layers = hole_layers[cell]
            if len(layers) != len(set(layers)):
                return None

    require_overlap = rng.random() < 0.45
    if require_overlap and overlap_cells == 0:
        return None
    if not touches_cross:
        return None

    grid = np.full((h, w), BG, dtype=int)
    grid[cross_r, :] = CROSS
    grid[:, cross_c] = CROSS
    for obj in objects:
        mask = np.asarray(obj["mask"], dtype=bool)
        color = int(obj["color"])
        top = int(obj["top"])
        left = int(obj["left"])
        mh, mw = map(int, mask.shape)
        region = grid[top : top + mh, left : left + mw]
        region[mask] = color

    parsed = _extract_objects(grid, BG, cross_r, cross_c, CROSS)
    if len(parsed) != 4:
        return None
    if sorted(int(obj["holes"]) for obj in parsed) != sorted(hole_counts):
        return None

    out = main(grid)
    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _build_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
