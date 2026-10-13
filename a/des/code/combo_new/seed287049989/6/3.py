from __future__ import annotations

import random
from typing import Iterable

import numpy as np


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components of a boolean mask as coordinate arrays."""
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


def _count_holes(mask: np.ndarray) -> int:
    """Count enclosed 4-connected background regions inside one object mask."""
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
            touches_edge = False
            while stack:
                cr, cc = stack.pop()
                if cr == 0 or cr == h - 1 or cc == 0 or cc == w - 1:
                    touches_edge = True
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            if not touches_edge:
                holes += 1
    return holes


def _object_components(grid: np.ndarray) -> list[dict[str, object]]:
    comps: list[dict[str, object]] = []
    body = grid[:-1, :]
    for color in sorted(int(v) for v in np.unique(body) if int(v) != 0):
        for coords in _connected_components(body == color):
            r0, c0, r1, c1 = _bbox(coords)
            local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            local[coords[:, 0] - r0, coords[:, 1] - c0] = True
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "holes": _count_holes(local),
                }
            )
    comps.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))  # type: ignore[index]
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw palette-colored horizontal stripes below each object based on its enclosed-hole count.

    Input structure:
    - Background is black.
    - The last row is a palette row containing distinct non-black single pixels separated by black cells.
    - Above the palette are several 4-connected monochrome objects with distinct colors; each object may enclose zero or more black holes.

    Rule:
    - Keep the original objects and the palette row unchanged.
    - For each foreground object, count its enclosed 4-connected black holes.
    - If an object has N holes, draw N full-width horizontal stripes immediately below its bounding box.
    - The stripe width equals the object's bounding-box width, and stripe k uses the k-th non-black color from the palette row, read left to right.

    Key insight:
    The visible action is driven by hidden topology: convert each object's hole count into a palette lookup, then materialize that count as stacked rows directly under the same bounding box.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2, "expected a 2D grid with a palette row"

    palette = [int(v) for v in g[-1, :] if int(v) != 0]
    assert palette, "palette row must contain non-black colors"

    out = g.copy()
    for comp in _object_components(g):
        holes = int(comp["holes"])
        if holes == 0:
            continue
        assert holes <= len(palette), "palette must be long enough for the hole count"
        r0, c0, r1, c1 = comp["bbox"]  # type: ignore[misc]
        for idx in range(holes):
            row = r1 + 1 + idx
            if row >= g.shape[0] - 1:
                break
            out[row, c0 : c1 + 1] = palette[idx]
    return out


GEN_ATTEMPTS = 400
GEN_OBJECT_COUNT = (2, 4)
GEN_PALETTE_LEN = (2, 5)
GEN_HOLE_CAP = 4
GEN_COLORS = tuple(range(1, 10))


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _add_protrusions(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    canvas = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=bool)
    canvas[1:-1, 1:-1] = mask
    h, w = mask.shape
    options: list[tuple[int, int]] = []

    for c in range(1, w + 1):
        if canvas[1, c]:
            options.append((0, c))
        if canvas[h, c]:
            options.append((h + 1, c))
    for r in range(1, h + 1):
        if canvas[r, 1]:
            options.append((r, 0))
        if canvas[r, w]:
            options.append((r, w + 1))

    rng.shuffle(options)
    for rr, cc in options[: rng.randint(0, min(3, len(options)))]:
        canvas[rr, cc] = True
    return _crop_mask(canvas)


def _room_layouts(n: int) -> list[tuple[int, int]]:
    layouts = []
    for rows in range(1, n + 1):
        if n % rows == 0:
            layouts.append((rows, n // rows))
    return layouts


def _make_holey_object(hole_count: int, rng: random.Random) -> np.ndarray:
    rows, cols = rng.choice(_room_layouts(hole_count))
    cell_h = rng.randint(1, 2)
    cell_w = rng.randint(1, 2 if hole_count <= 3 else 1)

    h = rows * cell_h + rows + 1
    w = cols * cell_w + cols + 1
    mask = np.ones((h, w), dtype=bool)
    for rr in range(rows):
        for cc in range(cols):
            r0 = 1 + rr * (cell_h + 1)
            c0 = 1 + cc * (cell_w + 1)
            mask[r0 : r0 + cell_h, c0 : c0 + cell_w] = False

    mask = _add_protrusions(mask, rng)
    assert _count_holes(mask) == hole_count
    return mask


def _solid_growth_mask(h: int, w: int, target: int, rng: random.Random) -> np.ndarray:
    start = (rng.randrange(h), rng.randrange(w))
    cells = {start}
    frontier = {pt for pt in _neighbors4(*start) if 0 <= pt[0] < h and 0 <= pt[1] < w}

    while len(cells) < target and frontier:
        cell = rng.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        for nxt in _neighbors4(*cell):
            if 0 <= nxt[0] < h and 0 <= nxt[1] < w and nxt not in cells:
                frontier.add(nxt)

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return _crop_mask(mask)


def _make_solid_object(rng: random.Random) -> np.ndarray:
    for _ in range(120):
        h = rng.randint(2, 5)
        w = rng.randint(2, 6)
        target = rng.randint(max(3, (h * w) // 3), max(4, h * w - 1))
        mask = _solid_growth_mask(h, w, target, rng)
        if min(mask.shape) < 2:
            continue
        if _count_holes(mask) != 0:
            continue
        return mask
    raise RuntimeError("failed to generate a solid object")


def _make_object(hole_count: int, rng: random.Random) -> np.ndarray:
    if hole_count == 0:
        return _make_solid_object(rng)
    return _make_holey_object(hole_count, rng)


def _sample_hole_counts(object_count: int, palette_len: int, rng: random.Random) -> list[int]:
    max_holes = min(palette_len, GEN_HOLE_CAP)
    counts = [rng.randint(0, max_holes) for _ in range(object_count)]
    if not any(v > 0 for v in counts):
        counts[rng.randrange(object_count)] = rng.randint(1, max_holes)
    if object_count > 1 and not any(v == 0 for v in counts) and rng.random() < 0.5:
        counts[rng.randrange(object_count)] = 0
    return counts


def _palette_positions(width: int, palette_len: int, rng: random.Random) -> list[int] | None:
    if palette_len == 1:
        return [rng.randrange(width)]
    gaps = [rng.randint(1, 3) for _ in range(palette_len - 1)]
    span = palette_len + sum(gaps)
    if span > width:
        return None
    start = rng.randint(0, width - span)
    pos = [start]
    cur = start
    for gap in gaps:
        cur += gap + 1
        pos.append(cur)
    return pos


def _sample_canvas(objects: list[np.ndarray], holes: list[int], palette_len: int, rng: random.Random) -> tuple[int, int] | None:
    widths = [int(obj.shape[1]) for obj in objects]
    heights = [int(obj.shape[0]) for obj in objects]
    reserved = [h + holes[i] for i, h in enumerate(heights)]

    min_palette_width = palette_len + (palette_len - 1)
    min_width = max(min_palette_width, sum(widths) + len(widths) + 1, max(widths) + 2)
    min_height = max(10, 2 * max(heights) + 2, max(reserved) + 3)
    if min_width > 30 or min_height > 30:
        return None

    width = rng.randint(min_width, min(30, min_width + 8))
    height = rng.randint(min_height, min(30, min_height + 8))
    return height, width


def _place_objects(
    grid: np.ndarray,
    objects: list[np.ndarray],
    holes: list[int],
    colors: Iterable[int],
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    occupied = np.zeros((h - 1, w), dtype=bool)
    order = sorted(range(len(objects)), key=lambda i: (objects[i].shape[0] + holes[i]) * objects[i].shape[1], reverse=True)
    placements: dict[int, tuple[int, int]] = {}

    for idx in order:
        obj = objects[idx]
        oh, ow = map(int, obj.shape)
        stripe_h = int(holes[idx])
        max_top = h // 2 - oh
        max_left = w - ow - 1
        if max_top < 1 or max_left < 1:
            return False

        placed = False
        for _ in range(250):
            top = rng.randint(1, max_top)
            left = rng.randint(1, max_left)
            end = top + oh + stripe_h
            if end > h - 1:
                continue
            r0 = max(0, top - 1)
            r1 = min(h - 1, end + 1)
            c0 = max(0, left - 1)
            c1 = min(w, left + ow + 1)
            if occupied[r0:r1, c0:c1].any():
                continue
            occupied[r0:r1, c0:c1] = True
            placements[idx] = (top, left)
            placed = True
            break
        if not placed:
            return False

    for idx, color in enumerate(colors):
        top, left = placements[idx]
        obj = objects[idx]
        rows, cols = np.where(obj)
        grid[top + rows, left + cols] = int(color)
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    object_count = rng.randint(*GEN_OBJECT_COUNT)
    max_palette = min(GEN_PALETTE_LEN[1], len(GEN_COLORS) - object_count)
    palette_len = rng.randint(GEN_PALETTE_LEN[0], max_palette)

    hole_counts = _sample_hole_counts(object_count, palette_len, rng)
    objects = [_make_object(count, rng) for count in hole_counts]
    canvas = _sample_canvas(objects, hole_counts, palette_len, rng)
    if canvas is None:
        return None

    height, width = canvas
    palette_cols = _palette_positions(width, palette_len, rng)
    if palette_cols is None:
        return None

    colors = list(GEN_COLORS)
    rng.shuffle(colors)
    object_colors = colors[:object_count]
    palette_colors = colors[object_count : object_count + palette_len]

    grid = np.zeros((height, width), dtype=int)
    if not _place_objects(grid, objects, hole_counts, object_colors, rng):
        return None

    for col, color in zip(palette_cols, palette_colors):
        grid[height - 1, col] = int(color)

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
