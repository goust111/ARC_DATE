from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
GEN_RETRY_LIMIT = 400


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        coords: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            coords.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        components.append(np.array(coords, dtype=int))
    return components


def _rotate_cw(grid: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(grid, -int(turns) % 4)


def _render_output(stamp: np.ndarray, macro: np.ndarray, map_color: int) -> np.ndarray:
    size = int(stamp.shape[0])
    side = int(macro.shape[0])
    center = side // 2
    out = np.zeros((side * size, side * size), dtype=int)
    for r, c in np.argwhere(macro == map_color):
        r = int(r)
        c = int(c)
        dist = max(abs(r - center), abs(c - center))
        block = _rotate_cw(stamp, dist)
        out[r * size : (r + 1) * size, c * size : (c + 1) * size] = block
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each lit macro-cell by the control stamp rotated clockwise by its ring distance from the map center.

    Input structure:
    - A black background contains one solid square stamp sprite filled with multiple non-black colors.
    - The remaining non-black pixels form a single-color odd square macro map whose center cell is lit.
    - The stamp colors and the macro-map color are distinct.

    Rule:
    - Find the unique full square component that is entirely non-black and uses multiple colors; this is the stamp.
    - Take all other non-black pixels as the macro map, using their odd square bounding box as the map extent.
    - For each lit macro cell, compute its Chebyshev distance from the map center, rotate the stamp clockwise by that many quarter-turns, and paste it into the corresponding output block; dark macro cells stay black.

    Key insight:
    - The map is only a coarse placement scaffold: each active cell expands into a full stamp tile, and the tile orientation is determined purely by how many square rings away that cell is from the center.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.any(g == BG), "expected black background"

    non_bg = g != BG
    components = _find_components(non_bg)
    stamp_candidates: list[tuple[tuple[int, int, int, int], np.ndarray]] = []

    for coords in components:
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h != w:
            continue
        if len(coords) != h * w:
            continue
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        colors = set(map(int, np.unique(patch)))
        if BG in colors or len(colors) < 2:
            continue
        stamp_candidates.append(((r0, c0, r1, c1), patch.copy()))

    assert len(stamp_candidates) == 1, "expected one multicolor square stamp"
    (sr0, sc0, sr1, sc1), stamp = stamp_candidates[0]
    stamp_mask = np.zeros_like(non_bg, dtype=bool)
    stamp_mask[sr0 : sr1 + 1, sc0 : sc1 + 1] = True

    macro_coords = np.argwhere(non_bg & ~stamp_mask)
    assert macro_coords.size > 0, "expected macro-map pixels outside the stamp"
    macro_colors = set(map(int, g[macro_coords[:, 0], macro_coords[:, 1]]))
    assert len(macro_colors) == 1, "macro map must be monochrome"
    map_color = next(iter(macro_colors))

    mr0, mc0, mr1, mc1 = _bbox_from_coords(macro_coords)
    side_h = mr1 - mr0 + 1
    side_w = mc1 - mc0 + 1
    assert side_h == side_w and side_h % 2 == 1, "macro map bbox must be an odd square"

    macro = g[mr0 : mr1 + 1, mc0 : mc1 + 1].copy()
    allowed = {BG, map_color}
    assert set(map(int, np.unique(macro))) <= allowed
    center = side_h // 2
    assert int(macro[center, center]) == map_color, "macro-map center must be lit"

    return _render_output(stamp, macro, map_color)


def _sample_palette(exclude: set[int], count: int) -> list[int]:
    choices = [color for color in PALETTE if color not in exclude]
    return random.sample(choices, count)


def _gen_stamp(size: int, map_color: int) -> np.ndarray | None:
    max_colors = min(5, size * size, len(PALETTE) - 1)
    min_colors = min(3, max_colors)
    if min_colors < 2:
        min_colors = 2
    color_count = random.randint(min_colors, max_colors)
    colors = _sample_palette({map_color}, color_count)

    for _ in range(80):
        stamp = np.array(
            [[random.choice(colors) for _ in range(size)] for _ in range(size)],
            dtype=int,
        )
        used = set(map(int, np.unique(stamp)))
        if len(used) != color_count:
            continue
        if any(np.array_equal(stamp, np.rot90(stamp, k)) for k in (1, 2, 3)):
            continue
        row_signatures = {tuple(map(int, row)) for row in stamp}
        col_signatures = {tuple(map(int, col)) for col in stamp.T}
        if len(row_signatures) < 2 or len(col_signatures) < 2:
            continue
        return stamp
    return None


def _ring_cells(radius: int, center: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    lo = center - radius
    hi = center + radius
    for c in range(lo, hi + 1):
        cells.append((lo, c))
        cells.append((hi, c))
    for r in range(lo + 1, hi):
        cells.append((r, lo))
        cells.append((r, hi))
    return cells


def _gen_macro_map(radius: int, map_color: int) -> np.ndarray:
    side = 2 * radius + 1
    center = radius
    macro = np.zeros((side, side), dtype=int)
    lit: set[tuple[int, int]] = {(center, center)}

    for dist in range(1, radius + 1):
        ring = _ring_cells(dist, center)
        required: set[tuple[int, int]] = set()
        if dist == radius:
            required.add((center - dist, random.randint(center - dist, center + dist)))
            required.add((center + dist, random.randint(center - dist, center + dist)))
            required.add((random.randint(center - dist, center + dist), center - dist))
            required.add((random.randint(center - dist, center + dist), center + dist))

        remaining = [cell for cell in ring if cell not in required]
        max_target = min(len(ring), max(len(required), 2 + 2 * dist))
        target_count = random.randint(max(len(required), 2), max_target)
        extra_count = max(0, target_count - len(required))
        if extra_count:
            required.update(random.sample(remaining, extra_count))
        lit.update(required)

    for r, c in lit:
        macro[r, c] = map_color
    return macro


def _paste_nonzero(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    view = dst[top : top + h, left : left + w]
    mask = patch != BG
    view[mask] = patch[mask]


def _assemble_input(stamp: np.ndarray, macro: np.ndarray) -> np.ndarray:
    sh, sw = stamp.shape
    mh, mw = macro.shape
    pad_top = random.randint(1, 3)
    pad_left = random.randint(1, 3)
    pad_bottom = random.randint(1, 3)
    pad_right = random.randint(1, 3)
    gap = random.randint(2, 4)
    horizontal = random.choice([True, False])
    stamp_first = random.choice([True, False])

    if horizontal:
        inner_h = max(sh, mh)
        inner_w = sw + gap + mw
        grid = np.zeros((pad_top + inner_h + pad_bottom, pad_left + inner_w + pad_right), dtype=int)
        stamp_r = pad_top + random.randint(0, inner_h - sh)
        macro_r = pad_top + random.randint(0, inner_h - mh)
        if stamp_first:
            stamp_c = pad_left
            macro_c = pad_left + sw + gap
        else:
            macro_c = pad_left
            stamp_c = pad_left + mw + gap
    else:
        inner_h = sh + gap + mh
        inner_w = max(sw, mw)
        grid = np.zeros((pad_top + inner_h + pad_bottom, pad_left + inner_w + pad_right), dtype=int)
        stamp_c = pad_left + random.randint(0, inner_w - sw)
        macro_c = pad_left + random.randint(0, inner_w - mw)
        if stamp_first:
            stamp_r = pad_top
            macro_r = pad_top + sh + gap
        else:
            macro_r = pad_top
            stamp_r = pad_top + mh + gap

    grid[stamp_r : stamp_r + sh, stamp_c : stamp_c + sw] = stamp
    _paste_nonzero(grid, macro, macro_r, macro_c)
    return grid


def _gen_attempt() -> np.ndarray | None:
    size = random.randint(2, 5)
    max_radius = min(4, (30 // size - 1) // 2)
    if max_radius < 1:
        return None
    radius = random.randint(1, max_radius)
    map_color = random.choice(PALETTE)
    stamp = _gen_stamp(size, map_color)
    if stamp is None:
        return None

    macro = _gen_macro_map(radius, map_color)
    expected = _render_output(stamp, macro, map_color)
    if max(expected.shape) > 30:
        return None

    inp = _assemble_input(stamp, macro)
    if max(inp.shape) > 30:
        return None

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    colors = Counter(map(int, inp.ravel()))
    if colors[BG] <= max(colors[color] for color in colors if color != BG):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle-4 input")
