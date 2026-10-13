from __future__ import annotations

import random
from collections import deque

import numpy as np


_DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_DIR8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)

# Generator configuration.
_GRID_H_RANGE = (14, 28)
_GRID_W_RANGE = (14, 26)
_OBJECT_COUNT_RANGE = (2, 5)
_SHAPE_RETRIES = 100
_GEN_RETRIES = 300


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_by_color(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue

            color = int(grid[r, c])
            q = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while q:
                cr, cc = q.pop()
                coords.append((int(cr), int(cc)))
                for dr, dc in _DIR4:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            arr = np.array(coords, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "top": int(arr[:, 0].min()),
                    "left": int(arr[:, 1].min()),
                }
            )

    return comps


def _boundary_pixel_count(grid: np.ndarray, coords: np.ndarray, bg: int) -> int:
    h, w = map(int, grid.shape)
    total = 0
    for r, c in coords:
        for dr, dc in _DIR4:
            nr, nc = int(r) + dr, int(c) + dc
            if not (0 <= nr < h and 0 <= nc < w) or int(grid[nr, nc]) == bg:
                total += 1
                break
    return total


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert each foreground object's boundary-pixel count into a horizontal bar.

    Objects are monochromatic 4-connected components on the dominant background.
    They are ordered by the top row of their bounding box. Each output row uses
    the object's color and has length equal to the number of object pixels that
    touch background in the 4-neighborhood.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    comps = _components_by_color(g, bg)
    assert comps, "expected at least one foreground object"

    items: list[tuple[int, int, int, int]] = []
    for comp in comps:
        coords = comp["coords"]
        perim = _boundary_pixel_count(g, coords, bg)
        items.append((int(comp["top"]), int(comp["left"]), int(comp["color"]), int(perim)))

    items.sort(key=lambda item: (item[0], item[1]))
    out_w = max(item[3] for item in items)
    out = np.full((len(items), out_w), bg, dtype=int)

    for row, (_top, _left, color, perim) in enumerate(items):
        out[row, :perim] = color

    return out


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _mask_connected(mask: np.ndarray) -> bool:
    points = np.argwhere(mask)
    if points.size == 0:
        return False

    h, w = map(int, mask.shape)
    start = tuple(map(int, points[0]))
    seen = {start}
    q = [start]

    while q:
        r, c = q.pop()
        for dr, dc in _DIR4:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or not mask[nr, nc]:
                continue
            nxt = (int(nr), int(nc))
            if nxt in seen:
                continue
            seen.add(nxt)
            q.append(nxt)

    return len(seen) == int(mask.sum())


def _mask_has_hole(mask: np.ndarray) -> bool:
    bg = ~mask
    if not bg.any():
        return False

    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if bg[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if bg[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in _DIR4:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if seen[nr, nc] or not bg[nr, nc]:
                continue
            seen[nr, nc] = True
            q.append((nr, nc))

    return bool(np.any(bg & ~seen))


def _has_2x2_block(mask: np.ndarray) -> bool:
    if mask.shape[0] < 2 or mask.shape[1] < 2:
        return False
    return bool(
        np.any(
            mask[:-1, :-1]
            & mask[1:, :-1]
            & mask[:-1, 1:]
            & mask[1:, 1:]
        )
    )


def _valid_shape(mask: np.ndarray) -> bool:
    if mask.size == 0 or not mask.any():
        return False
    mask = _trim_mask(mask.astype(bool))
    h, w = map(int, mask.shape)
    if h < 2 or w < 2 or int(mask.sum()) < 4:
        return False
    if not _mask_connected(mask):
        return False
    if _mask_has_hole(mask):
        return False
    if not _has_2x2_block(mask):
        return False
    return True


def _shape_rectangle(rng: random.Random, max_h: int, max_w: int) -> np.ndarray | None:
    if max_h < 2 or max_w < 2:
        return None
    h = rng.randint(2, min(6, max_h))
    w = rng.randint(2, min(7, max_w))
    return np.ones((h, w), dtype=bool)


def _shape_corner_notch(rng: random.Random, max_h: int, max_w: int) -> np.ndarray | None:
    if max_h < 4 or max_w < 4:
        return None
    h = rng.randint(4, min(7, max_h))
    w = rng.randint(4, min(8, max_w))
    mask = np.ones((h, w), dtype=bool)
    notch_h = rng.randint(1, h - 2)
    notch_w = rng.randint(1, w - 2)
    corner = rng.choice(("tl", "tr", "bl", "br"))

    if corner == "tl":
        mask[:notch_h, :notch_w] = False
    elif corner == "tr":
        mask[:notch_h, w - notch_w :] = False
    elif corner == "bl":
        mask[h - notch_h :, :notch_w] = False
    else:
        mask[h - notch_h :, w - notch_w :] = False

    return mask


def _shape_side_bite(rng: random.Random, max_h: int, max_w: int) -> np.ndarray | None:
    if max_h < 4 or max_w < 4:
        return None
    h = rng.randint(4, min(7, max_h))
    w = rng.randint(4, min(8, max_w))
    mask = np.ones((h, w), dtype=bool)
    side = rng.choice(("top", "bottom", "left", "right"))

    if side in ("top", "bottom"):
        depth = rng.randint(1, h - 2)
        bite_w = rng.randint(1, w - 2)
        start = rng.randint(1, w - bite_w - 1)
        rows = slice(0, depth) if side == "top" else slice(h - depth, h)
        mask[rows, start : start + bite_w] = False
    else:
        depth = rng.randint(1, w - 2)
        bite_h = rng.randint(1, h - 2)
        start = rng.randint(1, h - bite_h - 1)
        cols = slice(0, depth) if side == "left" else slice(w - depth, w)
        mask[start : start + bite_h, cols] = False

    return mask


def _shape_protrusion(rng: random.Random, max_h: int, max_w: int) -> np.ndarray | None:
    if max_h < 3 or max_w < 3:
        return None

    if rng.random() < 0.5:
        if max_h < 3 or max_w < 3:
            return None
        base_h = rng.randint(2, min(5, max_h - 1))
        base_w = rng.randint(3, min(7, max_w))
        protr_h = rng.randint(2, min(4, max_h - base_h + 1))
        protr_w = rng.randint(2, base_w - 1)
        offset = rng.randint(0, base_w - protr_w)
        total_h = base_h + protr_h - 1
        mask = np.zeros((total_h, base_w), dtype=bool)
        if rng.random() < 0.5:
            mask[protr_h - 1 :, :] = True
            mask[:protr_h, offset : offset + protr_w] = True
        else:
            mask[:base_h, :] = True
            mask[base_h - 1 :, offset : offset + protr_w] = True
        return mask

    base_h = rng.randint(3, min(7, max_h))
    base_w = rng.randint(2, min(5, max_w - 1))
    protr_w = rng.randint(2, min(4, max_w - base_w + 1))
    protr_h = rng.randint(2, base_h - 1)
    offset = rng.randint(0, base_h - protr_h)
    total_w = base_w + protr_w - 1
    mask = np.zeros((base_h, total_w), dtype=bool)
    if rng.random() < 0.5:
        mask[:, protr_w - 1 :] = True
        mask[offset : offset + protr_h, :protr_w] = True
    else:
        mask[:, :base_w] = True
        mask[offset : offset + protr_h, base_w - 1 :] = True
    return mask


def _generate_shape(rng: random.Random, max_h: int, max_w: int) -> np.ndarray | None:
    factories = (
        _shape_rectangle,
        _shape_corner_notch,
        _shape_side_bite,
        _shape_protrusion,
    )

    for _ in range(_SHAPE_RETRIES):
        factory = rng.choice(factories)
        mask = factory(rng, max_h, max_w)
        if mask is None:
            continue
        mask = _trim_mask(mask)
        if _valid_shape(mask):
            return mask
    return None


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = map(int, mask.shape)
    patch = grid[top : top + h, left : left + w]
    patch[mask] = color


def _validate_generated_grid(grid: np.ndarray, expected_count: int) -> bool:
    bg = _background_color(grid)
    comps = _components_by_color(grid, bg)
    if len(comps) != expected_count:
        return False

    colors = [int(comp["color"]) for comp in comps]
    if len(set(colors)) != len(colors):
        return False

    tops = [int(comp["top"]) for comp in comps]
    if len(set(tops)) != len(tops):
        return False

    bg_count = int((grid == bg).sum())
    largest_object = max(int(comp["coords"].shape[0]) for comp in comps)
    if bg_count <= largest_object:
        return False

    h, w = map(int, grid.shape)
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg:
                continue
            for dr, dc in _DIR8:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    other = int(grid[nr, nc])
                    if other != bg and other != color:
                        return False

    for comp in comps:
        coords = comp["coords"]
        r0, c0 = np.min(coords, axis=0)
        r1, c1 = np.max(coords, axis=0)
        mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
        for r, c in coords:
            mask[int(r - r0), int(c - c0)] = True
        if not _valid_shape(mask):
            return False

    out = main(grid)
    items = sorted(
        (
            int(comp["top"]),
            int(comp["left"]),
            int(comp["color"]),
            _boundary_pixel_count(grid, comp["coords"], bg),
        )
        for comp in comps
    )
    if out.shape != (len(items), max(item[3] for item in items)):
        return False

    for row, (_top, _left, color, count) in enumerate(items):
        if not np.all(out[row, :count] == color):
            return False
        if count < out.shape[1] and not np.all(out[row, count:] == bg):
            return False

    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_objects = rng.randint(*_OBJECT_COUNT_RANGE)
    min_h = max(_GRID_H_RANGE[0], 4 * n_objects + 3)
    if min_h > _GRID_H_RANGE[1]:
        return None

    grid_h = rng.randint(min_h, _GRID_H_RANGE[1])
    grid_w = rng.randint(_GRID_W_RANGE[0], _GRID_W_RANGE[1])
    bg = rng.randrange(10)
    colors = rng.sample([c for c in range(10) if c != bg], n_objects)

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    top = rng.randint(1, 2)

    for index, color in enumerate(colors):
        future = n_objects - index - 1
        max_h = min(8, grid_h - top - (3 * future + 1))
        max_w = min(9, grid_w - 2)
        if max_h < 2 or max_w < 2:
            return None

        mask = _generate_shape(rng, max_h=max_h, max_w=max_w)
        if mask is None:
            return None

        obj_h, obj_w = map(int, mask.shape)
        left = rng.randint(1, grid_w - obj_w - 1)
        _paint_mask(grid, mask, top, left, color)

        if future:
            gap_max = grid_h - (top + obj_h) - 3 * future
            if gap_max < 1:
                return None
            top += obj_h + rng.randint(1, min(3, gap_max))

    if not _validate_generated_grid(grid, n_objects):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
