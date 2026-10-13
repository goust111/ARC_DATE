from __future__ import annotations

from collections import deque
import random

import numpy as np


AZURE = 8
COLORS = [0, 1, 2, 3, 4, 5, 6, 7, 9]
MAX_GENERATION_ATTEMPTS = 300


def _components_by_color(g: np.ndarray, bg: int) -> list[dict[str, int]]:
    height, width = map(int, g.shape)
    seen = np.zeros((height, width), dtype=bool)
    objects: list[dict[str, int]] = []

    for sr in range(height):
        for sc in range(width):
            if seen[sr, sc] or int(g[sr, sc]) == bg:
                continue

            color = int(g[sr, sc])
            q = deque([(sr, sc)])
            seen[sr, sc] = True
            area = 0

            while q:
                r, c = q.popleft()
                area += 1
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr, cc = r + dr, c + dc
                    if not (0 <= rr < height and 0 <= cc < width):
                        continue
                    if seen[rr, cc] or int(g[rr, cc]) != color:
                        continue
                    seen[rr, cc] = True
                    q.append((rr, cc))

            objects.append({"color": color, "area": area})

    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """Render colored foreground objects as concentric rectangular rings ordered by object area.

    Input structure:
    - The grid background is Azure.
    - Several differently colored 4-connected monochrome foreground objects are
      scattered on the background.
    - Each foreground object has a distinct pixel area; its original shape may be arbitrary.

    Rule:
    - Find every non-Azure monochrome 4-connected foreground object.
    - Sort the objects by pixel area from largest to smallest.
    - Create a square Azure output grid with side length 2 * object_count - 1.
    - Draw the largest object's color as the outer full rectangular ring, then the
      next largest as the next inner ring, continuing until the smallest object's
      color fills the single center pixel.

    Key insight:
    - The input shapes are only counted by area; their geometry is discarded and
      rebuilt as a size-ranked stack of nested rectangular rings.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    objects = _components_by_color(g, AZURE)
    assert objects, "expected at least one foreground object"

    areas = [obj["area"] for obj in objects]
    assert len(set(areas)) == len(areas), "foreground object areas must be unique"

    colors = [obj["color"] for obj in objects]
    assert len(set(colors)) == len(colors), "foreground object colors must be unique"

    ordered = sorted(objects, key=lambda obj: obj["area"], reverse=True)
    side = 2 * len(ordered) - 1
    assert side <= 30, "output grid exceeds ARC size limit"

    out = np.full((side, side), AZURE, dtype=int)
    for layer, obj in enumerate(ordered):
        out[layer : side - layer, layer : side - layer] = int(obj["color"])
    return out


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _transform_mask(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, random.randrange(4))
    if random.random() < 0.5:
        out = out[:, ::-1]
    if random.random() < 0.5:
        out = out[::-1, :]
    return np.array(out, dtype=bool)


def _line_mask(area: int, max_h: int, max_w: int) -> np.ndarray | None:
    options = []
    if area <= max_w:
        options.append(np.ones((1, area), dtype=bool))
    if area <= max_h:
        options.append(np.ones((area, 1), dtype=bool))
    if not options:
        return None
    return random.choice(options)


def _rectangle_mask(area: int, max_h: int, max_w: int) -> np.ndarray | None:
    options = []
    for h in range(2, min(max_h, area) + 1):
        if area % h != 0:
            continue
        w = area // h
        if 2 <= w <= max_w:
            options.append(np.ones((h, w), dtype=bool))
    if not options:
        return None
    return random.choice(options)


def _growth_mask(area: int, max_h: int, max_w: int) -> np.ndarray | None:
    if area == 1:
        return np.ones((1, 1), dtype=bool)

    dims = []
    for h in range(1, min(max_h, area) + 1):
        for w in range(1, min(max_w, area) + 1):
            capacity = h * w
            if area <= capacity <= max(area + 8, int(area * 2.0)):
                dims.append((h, w))
    random.shuffle(dims)

    for h, w in dims[:40]:
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        frontier = set()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = start[0] + dr, start[1] + dc
            if 0 <= rr < h and 0 <= cc < w:
                frontier.add((rr, cc))

        while len(cells) < area and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            r, c = cell
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                nxt = (rr, cc)
                if 0 <= rr < h and 0 <= cc < w and nxt not in cells:
                    frontier.add(nxt)

        if len(cells) == area:
            mask = np.zeros((h, w), dtype=bool)
            for r, c in cells:
                mask[r, c] = True
            return _crop_mask(mask)

    return None


def _make_shape(area: int, max_h: int, max_w: int) -> np.ndarray | None:
    builders = [_growth_mask, _line_mask, _rectangle_mask]
    random.shuffle(builders)
    for builder in builders:
        mask = builder(area, max_h, max_w)
        if mask is None:
            continue
        mask = _transform_mask(mask)
        if mask.shape[0] <= max_h and mask.shape[1] <= max_w:
            return mask
    return None


def _mark_with_padding(occupied: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> None:
    height, width = map(int, occupied.shape)
    for rr, cc in np.argwhere(mask):
        r = int(r0 + rr)
        c = int(c0 + cc)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width:
                    occupied[nr, nc] = True


def _place_shape(
    grid: np.ndarray,
    occupied: np.ndarray,
    mask: np.ndarray,
    color: int,
) -> bool:
    height, width = map(int, grid.shape)
    mh, mw = map(int, mask.shape)
    if mh + 2 > height or mw + 2 > width:
        return False

    candidates = []
    for r0 in range(1, height - mh):
        for c0 in range(1, width - mw):
            window = occupied[r0 : r0 + mh, c0 : c0 + mw]
            if not np.any(window[mask]):
                candidates.append((r0, c0))

    if not candidates:
        return False

    r0, c0 = random.choice(candidates)
    window = grid[r0 : r0 + mh, c0 : c0 + mw]
    window[mask] = int(color)
    _mark_with_padding(occupied, mask, r0, c0)
    return True


def _expected_output(items: list[tuple[int, int]]) -> np.ndarray:
    ordered = sorted(items, key=lambda item: item[0], reverse=True)
    side = 2 * len(ordered) - 1
    out = np.full((side, side), AZURE, dtype=int)
    for layer, (_area, color) in enumerate(ordered):
        out[layer : side - layer, layer : side - layer] = int(color)
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(12, 30)
    width = random.randint(12, 30)
    object_count = random.randint(3, min(8, len(COLORS)))

    max_area = min(18, max(6, (height * width) // (object_count * 7)))
    if max_area < object_count:
        return None

    areas = random.sample(range(1, max_area + 1), object_count)
    colors = random.sample(COLORS, object_count)

    shapes = []
    for area, color in zip(areas, colors):
        mask = _make_shape(area, height - 2, width - 2)
        if mask is None:
            return None
        shapes.append((area, color, mask))

    grid = np.full((height, width), AZURE, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    random.shuffle(shapes)
    shapes.sort(key=lambda item: item[2].shape[0] * item[2].shape[1], reverse=True)
    for _area, color, mask in shapes:
        if not _place_shape(grid, occupied, mask, color):
            return None

    expected = _expected_output([(area, color) for area, color, _ in shapes])
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
