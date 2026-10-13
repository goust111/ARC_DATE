from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5
MAGENTA = 6
RECT_COLORS = (3, 4, 7, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _transform_patch(patch: np.ndarray, width: int, blue_above_red: bool) -> np.ndarray:
    odd_uses_vertical = bool(blue_above_red)
    use_vertical = bool(odd_uses_vertical) if width % 2 == 1 else not bool(odd_uses_vertical)
    return np.flipud(patch) if use_vertical else np.fliplr(patch)


def _center_cell(height: int, width: int) -> tuple[int, int]:
    return int((height - 1) // 2), int((width - 1) // 2)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read a blue/red control pair in the top band, clear the source band, then move each
    solid rectangle to its matched magenta slot in the bottom band.

    Odd-width rectangles use transform A; even-width rectangles use transform B.
    If blue is above red, A=vertical mirror and B=horizontal mirror; otherwise swapped.

    The moved rectangle is anchored by its discrete bbox center cell:
    ((r0+r1)//2, (c0+c1)//2). This matches the generator's slot construction and keeps
    even-sized rectangles well-defined on the integer grid.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h >= 5 and w >= 3

    sep_rows = [int(r) for r in range(h) if bool(np.all(g[r, :] == GREY))]
    assert len(sep_rows) == 2
    sep_top, sep_bottom = sep_rows
    assert 0 < sep_top < sep_bottom < h - 1

    top_band = g[:sep_top, :]
    src_band = g[sep_top + 1 : sep_bottom, :]
    dst_band = g[sep_bottom + 1 :, :]
    assert top_band.size and src_band.size and dst_band.size

    blue_pts = np.argwhere(top_band == BLUE)
    red_pts = np.argwhere(top_band == RED)
    assert blue_pts.shape[0] == 1 and red_pts.shape[0] == 1
    blue_above_red = bool(int(blue_pts[0, 0]) < int(red_pts[0, 0]))

    src_components = []
    for coords in _components(src_band != BLACK):
        r0, c0, r1, c1 = _bbox(coords)
        patch = src_band[r0 : r1 + 1, c0 : c1 + 1].copy()
        src_components.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "patch": patch,
            }
        )
    assert src_components
    src_components.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))

    slots = [(int(r), int(c)) for r, c in np.argwhere(dst_band == MAGENTA)]
    assert slots
    slots.sort(key=lambda rc: (rc[1], rc[0]))
    assert len(slots) == len(src_components)

    out = g.copy()
    out[sep_top + 1 : sep_bottom, :] = BLACK
    out[sep_bottom + 1 :, :] = BLACK

    dst_row0 = int(sep_bottom + 1)
    for item, (slot_r, slot_c) in zip(src_components, slots):
        patch = item["patch"]
        ph, pw = patch.shape
        moved = _transform_patch(patch, pw, blue_above_red)
        anchor_r, anchor_c = _center_cell(ph, pw)
        top_r = int(slot_r - anchor_r)
        left_c = int(slot_c - anchor_c)
        assert 0 <= top_r <= dst_band.shape[0] - ph
        assert 0 <= left_c <= w - pw
        target = out[dst_row0 + top_r : dst_row0 + top_r + ph, left_c : left_c + pw]
        assert bool(np.all(target == BLACK))
        target[:, :] = moved

    return out


def _sample_sizes(count: int) -> list[tuple[int, int]]:
    while True:
        sizes = [(random.randint(1, 4), random.randint(1, 6)) for _ in range(count)]
        widths = [w for _h, w in sizes]
        if not any(width % 2 == 0 for width in widths):
            continue
        if not any(width % 2 == 1 for width in widths):
            continue
        if not any(h * w > 1 for h, w in sizes):
            continue
        return sizes


def _sample_colors(count: int) -> list[int]:
    if random.random() < 0.55 and count <= len(RECT_COLORS):
        return random.sample(list(RECT_COLORS), count)
    colors = [random.choice(RECT_COLORS) for _ in range(count)]
    if count > 1 and len(set(colors)) == 1:
        idx = random.randrange(count)
        choices = [c for c in RECT_COLORS if c != colors[idx]]
        colors[idx] = random.choice(choices)
    return colors


def _pack_rectangles(sizes: list[tuple[int, int]], band_h: int) -> tuple[list[tuple[int, int, int, int]], int]:
    placements: list[tuple[int, int, int, int]] = []
    x = 0
    for idx, (rh, rw) in enumerate(sizes):
        if idx > 0:
            x += random.randint(1, 4)
        y = random.randint(0, band_h - rh)
        placements.append((int(y), int(x), int(rh), int(rw)))
        x += rw
    return placements, int(x)


def _shift_placements(
    placements: list[tuple[int, int, int, int]], dx: int
) -> list[tuple[int, int, int, int]]:
    return [(int(r), int(c + dx), int(h), int(w)) for r, c, h, w in placements]


def _build_case() -> tuple[np.ndarray, np.ndarray]:
    count = random.randint(2, 5)
    sizes = _sample_sizes(count)
    colors = _sample_colors(count)
    max_h = max(h for h, _w in sizes)

    top_h = random.randint(3, 6)
    src_h = max_h + random.randint(1, 4)
    dst_h = max_h + random.randint(1, 4)

    src_places, src_span = _pack_rectangles(sizes, src_h)
    dst_places, dst_span = _pack_rectangles(sizes, dst_h)

    src_left = random.randint(1, 4)
    dst_left = random.randint(1, 4)
    width = max(
        src_left + src_span + random.randint(1, 4),
        dst_left + dst_span + random.randint(1, 4),
    )
    width += random.randint(0, 4)

    src_places = _shift_placements(src_places, src_left)
    dst_places = _shift_placements(dst_places, dst_left)

    height = top_h + 1 + src_h + 1 + dst_h
    grid = np.full((height, width), BLACK, dtype=int)
    expected = np.full((height, width), BLACK, dtype=int)

    sep_top = int(top_h)
    sep_bottom = int(top_h + 1 + src_h)
    grid[sep_top, :] = GREY
    grid[sep_bottom, :] = GREY
    expected[sep_top, :] = GREY
    expected[sep_bottom, :] = GREY

    blue_above_red = bool(random.getrandbits(1))
    control_rows = sorted(random.sample(range(top_h), 2))
    if blue_above_red:
        blue_row, red_row = control_rows[0], control_rows[1]
    else:
        red_row, blue_row = control_rows[0], control_rows[1]
    grid[int(blue_row), 0] = BLUE
    grid[int(red_row), 0] = RED
    expected[int(blue_row), 0] = BLUE
    expected[int(red_row), 0] = RED

    src_row0 = int(sep_top + 1)
    dst_row0 = int(sep_bottom + 1)

    for (r, c, rh, rw), color in zip(src_places, colors):
        grid[src_row0 + r : src_row0 + r + rh, c : c + rw] = int(color)

    for (r, c, rh, rw), color in zip(dst_places, colors):
        slot_r = int(r + _center_cell(rh, rw)[0])
        slot_c = int(c + _center_cell(rh, rw)[1])
        grid[dst_row0 + slot_r, slot_c] = MAGENTA
        patch = np.full((rh, rw), int(color), dtype=int)
        moved = _transform_patch(patch, rw, blue_above_red)
        expected[dst_row0 + r : dst_row0 + r + rh, c : c + rw] = moved

    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            grid, expected = _build_case()
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
