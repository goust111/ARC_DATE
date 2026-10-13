from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


BG = 8
SEP = 5
FOREGROUND_COLORS = (0, 1, 2, 3, 4, 6, 7, 9)
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _normalize(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _is_connected(cells: set[tuple[int, int]]) -> bool:
    start = next(iter(cells))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for rr, cc in _neighbors4(r, c):
            if (rr, cc) in cells and (rr, cc) not in seen:
                seen.add((rr, cc))
                q.append((rr, cc))
    return len(seen) == len(cells)


def _build_shape_catalog() -> list[tuple[tuple[int, int], ...]]:
    shapes: set[tuple[tuple[int, int], ...]] = set()
    for mask in range(1 << 9):
        area = mask.bit_count()
        if not (3 <= area <= 8):
            continue
        cells = {(idx // 3, idx % 3) for idx in range(9) if (mask >> idx) & 1}
        if not _is_connected(cells):
            continue
        shape = _normalize(cells)
        h = max(r for r, _ in shape) + 1
        w = max(c for _, c in shape) + 1
        if h <= 3 and w <= 3:
            shapes.add(shape)
    return sorted(shapes, key=lambda shape: (len(shape), shape))


SHAPES = _build_shape_catalog()


def _shape_to_array(shape: tuple[tuple[int, int], ...], color: int) -> np.ndarray:
    h = max(r for r, _ in shape) + 1
    w = max(c for _, c in shape) + 1
    arr = np.full((h, w), BG, dtype=int)
    for r, c in shape:
        arr[r, c] = color
    return arr


def _extract_sprites(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)

    seen = np.zeros((h, w), dtype=bool)
    sprites: list[dict] = []
    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0] or int(g[r0, c0]) == BG:
                continue
            color = int(g[r0, c0])
            q = deque([(r0, c0)])
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                coords.append((r, c))
                for rr, cc in _neighbors4(r, c):
                    if 0 <= rr < h and 0 <= cc < w and not seen[rr, cc] and int(g[rr, cc]) == color:
                        seen[rr, cc] = True
                        q.append((rr, cc))

            rs = [r for r, _ in coords]
            cs = [c for _, c in coords]
            r1, c1 = min(rs), min(cs)
            r2, c2 = max(rs), max(cs)
            box_h = r2 - r1 + 1
            box_w = c2 - c1 + 1
            area = len(coords)

            assert 3 <= area <= 8, "sprite area out of range"
            assert box_h <= 3 and box_w <= 3, "sprite bbox too large"

            crop = g[r1 : r2 + 1, c1 : c2 + 1].copy()
            sprites.append(
                {
                    "color": color,
                    "coords": tuple(sorted(coords)),
                    "bbox": (r1, c1, r2, c2),
                    "crop": crop,
                    "center_r": (r1 + r2) / 2.0,
                    "center_c": (c1 + c2) / 2.0,
                }
            )

    assert sprites, "expected at least one sprite"
    return sprites


def _pack_row(sprites: list[dict]) -> np.ndarray:
    if not sprites:
        return np.full((0, 0), BG, dtype=int)

    height = max(sprite["crop"].shape[0] for sprite in sprites)
    width = sum(sprite["crop"].shape[1] for sprite in sprites) + max(0, len(sprites) - 1)
    row = np.full((height, width), BG, dtype=int)

    col = 0
    for sprite in sprites:
        crop = sprite["crop"]
        h, w = crop.shape
        row[:h, col : col + w] = crop
        col += w + 1
    return row


def _pack_section(rows: list[np.ndarray]) -> np.ndarray:
    rows = [row for row in rows if row.size > 0]
    if not rows:
        return np.full((0, 0), BG, dtype=int)

    width = max(row.shape[1] for row in rows)
    height = sum(row.shape[0] for row in rows)
    section = np.full((height, width), BG, dtype=int)

    top = 0
    for row in rows:
        rh, rw = row.shape
        section[top : top + rh, :rw] = row
        top += rh
    return section


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group same-color sprites by vertical half, then pack each subgroup into a horizontal strip.

    The input uses azure background and several small monochrome 4-connected sprites.
    Sprites are first grouped by color. Within each color, sprites whose bbox center is
    above the horizontal midpoint go to the upper section; the rest go to the lower section.
    Inside each subgroup, sprites are ordered from left to right and concatenated with one
    azure column between consecutive bbox crops. Color rows follow the input reading order
    of the color groups, and the two sections are separated by a full grey row.
    """
    g = np.asarray(grid, dtype=int)
    sprites = _extract_sprites(g)

    colors = {sprite["color"] for sprite in sprites}
    assert 2 <= len(colors) <= 3, "expected 2 or 3 foreground colors"

    midpoint = g.shape[0] / 2.0
    color_order = sorted(
        colors,
        key=lambda color: min((sprite["bbox"][0], sprite["bbox"][1]) for sprite in sprites if sprite["color"] == color),
    )

    by_color: dict[int, dict[str, list[dict]]] = defaultdict(lambda: {"upper": [], "lower": []})
    for sprite in sprites:
        zone = "upper" if sprite["center_r"] < midpoint else "lower"
        by_color[sprite["color"]][zone].append(sprite)

    upper_rows: list[np.ndarray] = []
    lower_rows: list[np.ndarray] = []
    for color in color_order:
        upper = sorted(
            by_color[color]["upper"],
            key=lambda sprite: (sprite["center_c"], sprite["bbox"][1], sprite["bbox"][0]),
        )
        lower = sorted(
            by_color[color]["lower"],
            key=lambda sprite: (sprite["center_c"], sprite["bbox"][1], sprite["bbox"][0]),
        )
        if upper:
            upper_rows.append(_pack_row(upper))
        if lower:
            lower_rows.append(_pack_row(lower))

    upper_section = _pack_section(upper_rows)
    lower_section = _pack_section(lower_rows)

    out_h = upper_section.shape[0] + 1 + lower_section.shape[0]
    out_w = max(1, upper_section.shape[1], lower_section.shape[1])
    out = np.full((out_h, out_w), BG, dtype=int)

    if upper_section.size:
        out[: upper_section.shape[0], : upper_section.shape[1]] = upper_section
    out[upper_section.shape[0], :] = SEP
    if lower_section.size:
        start = upper_section.shape[0] + 1
        out[start : start + lower_section.shape[0], : lower_section.shape[1]] = lower_section

    return out


def _valid_positions(
    blocked: np.ndarray,
    shape: tuple[tuple[int, int], ...],
    zone: str,
) -> list[tuple[int, int]]:
    h, w = blocked.shape
    box_h = max(r for r, _ in shape) + 1
    box_w = max(c for _, c in shape) + 1
    midpoint = h / 2.0

    positions: list[tuple[int, int]] = []
    for r in range(h - box_h + 1):
        center_r = r + (box_h - 1) / 2.0
        if zone == "upper" and not (center_r < midpoint):
            continue
        if zone == "lower" and not (center_r >= midpoint):
            continue
        for c in range(w - box_w + 1):
            r_lo = max(0, r - 1)
            r_hi = min(h, r + box_h + 1)
            c_lo = max(0, c - 1)
            c_hi = min(w, c + box_w + 1)
            if blocked[r_lo:r_hi, c_lo:c_hi].any():
                continue
            positions.append((r, c))
    return positions


def _place_shape(
    grid: np.ndarray,
    blocked: np.ndarray,
    shape: tuple[tuple[int, int], ...],
    color: int,
    zone: str,
) -> bool:
    positions = _valid_positions(blocked, shape, zone)
    if not positions:
        return False

    r, c = random.choice(positions)
    for dr, dc in shape:
        grid[r + dr, c + dc] = color

    box_h = max(rr for rr, _ in shape) + 1
    box_w = max(cc for _, cc in shape) + 1
    r_lo = max(0, r - 1)
    r_hi = min(grid.shape[0], r + box_h + 1)
    c_lo = max(0, c - 1)
    c_hi = min(grid.shape[1], c + box_w + 1)
    blocked[r_lo:r_hi, c_lo:c_hi] = True
    return True


def _color_zone_counts(colors: list[int]) -> dict[int, dict[str, int]]:
    counts = {color: {"upper": 1, "lower": 1} for color in colors}
    extra_objects = random.randint(0, len(colors) + 1)
    for _ in range(extra_objects):
        color = random.choice(colors)
        zone = random.choice(("upper", "lower"))
        counts[color][zone] += 1
    return counts


def _generator_attempt() -> np.ndarray | None:
    height = random.randint(12, 18)
    width = random.randint(13, 22)
    color_count = random.randint(2, 3)
    colors = random.sample(FOREGROUND_COLORS, color_count)
    counts = _color_zone_counts(colors)

    grid = np.full((height, width), BG, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)

    tasks: list[tuple[int, str, tuple[tuple[int, int], ...]]] = []
    for color in colors:
        for zone in ("upper", "lower"):
            for _ in range(counts[color][zone]):
                tasks.append((color, zone, random.choice(SHAPES)))

    random.shuffle(tasks)
    tasks.sort(key=lambda item: len(item[2]), reverse=True)

    for color, zone, shape in tasks:
        if not _place_shape(grid, blocked, shape, color, zone):
            return None

    try:
        sprites = _extract_sprites(grid)
        recovered_colors = {sprite["color"] for sprite in sprites}
        if recovered_colors != set(colors):
            return None

        midpoint = height / 2.0
        recovered_counts = {color: {"upper": 0, "lower": 0} for color in colors}
        for sprite in sprites:
            zone = "upper" if sprite["center_r"] < midpoint else "lower"
            recovered_counts[sprite["color"]][zone] += 1
        if recovered_counts != counts:
            return None

        out = main(grid)
        sep_rows = np.where(np.all(out == SEP, axis=1))[0]
        if out.shape[0] < 3 or len(sep_rows) != 1:
            return None
    except Exception:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generator_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
