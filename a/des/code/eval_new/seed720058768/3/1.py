import random
from collections import Counter, deque

import numpy as np


BG = 0
WALL = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
TEMPLATE_SIZES = (3, 4)
ROOM_COUNT_RANGE = (2, 5)
TOP_H_RANGE = (10, 18)
TOP_W_MAX = 28
MIN_ROOM_SPAN = 3
GEN_RETRIES = 400


def _pattern(rows):
    return np.array([[1 if ch == "1" else 0 for ch in row] for row in rows], dtype=int)


PATTERN_LIBRARY = {
    3: [
        _pattern(["101", "010", "101"]),
        _pattern(["100", "010", "001"]),
        _pattern(["100", "110", "111"]),
        _pattern(["111", "011", "001"]),
        _pattern(["111", "101", "111"]),
        _pattern(["110", "011", "101"]),
        _pattern(["101", "011", "110"]),
    ],
    4: [
        _pattern(["1010", "0101", "1010", "0101"]),
        _pattern(["1000", "0100", "0010", "0001"]),
        _pattern(["1000", "1100", "1110", "1111"]),
        _pattern(["1111", "0111", "0011", "0001"]),
        _pattern(["1111", "1001", "1001", "1111"]),
        _pattern(["1100", "0110", "0011", "1001"]),
        _pattern(["1001", "0110", "1100", "0011"]),
    ],
}


def _most_common(values: np.ndarray) -> int:
    flat = [int(v) for v in values.reshape(-1)]
    return Counter(flat).most_common(1)[0][0]


def _intervals(mask_1d):
    spans = []
    start = None
    for i, value in enumerate(mask_1d):
        if value and start is None:
            start = i
        elif not value and start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(mask_1d)))
    return spans


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _extract_templates(bottom: np.ndarray, bg: int):
    assert bottom.size > 0, "expected template rows below the partitioned map"
    non_bg_rows = np.where(np.any(bottom != bg, axis=1))[0]
    assert len(non_bg_rows) > 0, "template strip must contain foreground pixels"
    band = bottom[non_bg_rows[0] : non_bg_rows[-1] + 1]

    non_bg_cols = np.any(band != bg, axis=0)
    spans = _intervals(non_bg_cols)
    assert spans, "failed to segment the template strip"

    templates = {}
    for c0, c1 in spans:
        tile = band[:, c0:c1].copy()
        key = int(tile[0, 0])
        assert key != bg, "template key must be its top-left color"
        colors = set(map(int, np.unique(tile)))
        assert colors <= {bg, key}, "template must be binary in bg/key colors"
        assert key not in templates, "template keys must be unique"
        templates[key] = tile
    return templates


def _partition_and_templates(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    wall = int(g[0, 0])
    bg = _most_common(g[g != wall])
    wall_rows = np.where(np.any(g == wall, axis=1))[0]
    assert len(wall_rows) > 0, "missing wall rows"
    top_h = int(wall_rows[-1]) + 1
    top = g[:top_h]
    templates = _extract_templates(g[top_h:], bg)
    return top, bg, wall, templates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each walled region with the template keyed by its seed color, then remove the template strip.

    Input structure:
    - bg is black, and the upper part of the grid is enclosed by a 1-pixel grey wall network with an outer grey border.
    - Each enclosed region inside that upper partition map contains exactly one single-pixel seed color on otherwise black cells.
    - Below the map sits a horizontal strip of separated square texture templates; each template is binary in black and one seed color, and its top-left pixel uses that seed color.

    Rule:
    - Treat grey as an immutable wall and find every 4-connected non-wall region in the upper map.
    - For each region, read its unique seed color, find the bottom template whose top-left pixel has the same color, and tile that template over the whole grid from the global origin (0,0).
    - Copy the tiled pixels only into cells of that region, overwriting the seed, keep all grey wall pixels unchanged, and discard the lower template strip from the output.

    Key insight:
    - The bottom squares are lookup templates keyed by their top-left color, and the texture phase is anchored to the whole-grid origin rather than to each region's local bounding box.
    """
    top, bg, wall, templates = _partition_and_templates(np.asarray(grid, dtype=int))
    out = top.copy()

    for room in _components(top != wall):
        seed_colors = {int(top[r, c]) for r, c in room if int(top[r, c]) not in (bg, wall)}
        assert len(seed_colors) == 1, "each region must contain exactly one seed color"
        seed = next(iter(seed_colors))
        assert seed in templates, "missing template for a region seed color"
        tile = templates[seed]
        th, tw = tile.shape
        for r, c in room:
            out[r, c] = int(tile[r % th, c % tw])

    return out


def _split_room(grid: np.ndarray, room, orient: str, rng: random.Random):
    r0, r1, c0, c1 = room
    if orient == "v":
        wc = rng.randint(c0 + MIN_ROOM_SPAN, c1 - MIN_ROOM_SPAN)
        grid[r0 - 1 : r1 + 2, wc] = WALL
        return (r0, r1, c0, wc - 1), (r0, r1, wc + 1, c1)
    wr = rng.randint(r0 + MIN_ROOM_SPAN, r1 - MIN_ROOM_SPAN)
    grid[wr, c0 - 1 : c1 + 2] = WALL
    return (r0, wr - 1, c0, c1), (wr + 1, r1, c0, c1)


def _make_partition(top_h: int, top_w: int, room_count: int, rng: random.Random):
    grid = np.full((top_h, top_w), BG, dtype=int)
    grid[0, :] = WALL
    grid[-1, :] = WALL
    grid[:, 0] = WALL
    grid[:, -1] = WALL

    rooms = [(1, top_h - 2, 1, top_w - 2)]
    for _ in range(200):
        if len(rooms) >= room_count:
            break
        splittable = []
        for idx, (r0, r1, c0, c1) in enumerate(rooms):
            opts = []
            if c1 - c0 + 1 >= 2 * MIN_ROOM_SPAN + 1:
                opts.append("v")
            if r1 - r0 + 1 >= 2 * MIN_ROOM_SPAN + 1:
                opts.append("h")
            if opts:
                splittable.append((idx, opts))
        if not splittable:
            break
        idx, opts = rng.choice(splittable)
        orient = rng.choice(opts)
        room = rooms.pop(idx)
        a, b = _split_room(grid, room, orient, rng)
        if rng.random() < 0.5:
            rooms.extend([a, b])
        else:
            rooms.extend([b, a])

    if len(rooms) != room_count:
        return None
    return grid, rooms


def _colorize_template(mask: np.ndarray, color: int):
    tile = np.full(mask.shape, BG, dtype=int)
    tile[mask.astype(bool)] = color
    return tile


def _place_templates(width: int, tiles, rng: random.Random):
    size = tiles[0].shape[0]
    count = len(tiles)
    gaps = [0] + [1] * (count - 1) + [0]
    base_width = count * size + sum(gaps)
    extra = width - base_width
    assert extra >= 0
    for _ in range(extra):
        gaps[rng.randrange(len(gaps))] += 1

    band = np.full((size, width), BG, dtype=int)
    c = gaps[0]
    for i, tile in enumerate(tiles):
        band[:, c : c + size] = tile
        c += size + gaps[i + 1]
    return band


def _validate_generated(grid: np.ndarray, expected_rooms: int):
    top, bg, wall, templates = _partition_and_templates(grid)
    rooms = _components(top != wall)
    assert len(rooms) == expected_rooms
    assert len(templates) == expected_rooms
    for room in rooms:
        seed_colors = {int(top[r, c]) for r, c in room if int(top[r, c]) not in (bg, wall)}
        assert len(seed_colors) == 1
        seed = next(iter(seed_colors))
        assert seed in templates
    out = main(grid)
    assert out.shape == top.shape
    assert max(grid.shape) <= 30 and max(out.shape) <= 30


def _gen_attempt(rng: random.Random):
    room_count = rng.randint(*ROOM_COUNT_RANGE)
    template_size = rng.choice(TEMPLATE_SIZES)
    min_width = room_count * template_size + (room_count - 1) + 2
    top_w = rng.randint(max(12, min_width), TOP_W_MAX)
    top_h = rng.randint(*TOP_H_RANGE)
    if top_h + template_size + 2 > 30:
        return None

    partition = _make_partition(top_h, top_w, room_count, rng)
    if partition is None:
        return None
    top, rooms = partition

    colors = rng.sample(SEED_COLORS, room_count)
    pattern_masks = rng.sample(PATTERN_LIBRARY[template_size], room_count)

    for room, color in zip(rooms, colors):
        r0, r1, c0, c1 = room
        sr = rng.randint(r0, r1)
        sc = rng.randint(c0, c1)
        top[sr, sc] = color

    tiles = [_colorize_template(mask, color) for mask, color in zip(pattern_masks, colors)]
    rng.shuffle(tiles)
    template_band = _place_templates(top_w, tiles, rng)

    separator_rows = rng.randint(1, 2)
    grid = np.vstack([top, np.full((separator_rows, top_w), BG, dtype=int), template_band])

    _validate_generated(grid, room_count)
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
