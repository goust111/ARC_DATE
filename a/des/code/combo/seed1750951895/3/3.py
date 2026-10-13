import random
from collections import Counter

import numpy as np


BG = 5
SEP = 9
GEN_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _find_separator_cross(grid: np.ndarray) -> tuple[int, int, int]:
    h, w = grid.shape
    solid_rows = [(r, int(grid[r, 0])) for r in range(h) if np.all(grid[r, :] == int(grid[r, 0]))]
    solid_cols = [(c, int(grid[0, c])) for c in range(w) if np.all(grid[:, c] == int(grid[0, c]))]
    crosses = [(r, c, color) for r, color in solid_rows for c, color2 in solid_cols if color == color2]
    assert len(crosses) == 1, "expected exactly one full separator row/column pair"
    r, c, color = crosses[0]
    assert 0 < r < h - 1 and 0 < c < w - 1, "separator must split the grid into four nonempty quadrants"
    return int(r), int(c), int(color)


def _is_rectangular_frame(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if len(coords) < 8:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return None
    frame = np.zeros_like(mask, dtype=bool)
    frame[r0 : r1 + 1, c0] = True
    frame[r0 : r1 + 1, c1] = True
    frame[r0, c0 : c1 + 1] = True
    frame[r1, c0 : c1 + 1] = True
    if not np.array_equal(mask, frame):
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _find_quadrant_frame(region: np.ndarray, bg: int, sep: int) -> tuple[int, tuple[int, int, int, int]]:
    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for color in np.unique(region):
        color = int(color)
        if color in {bg, sep}:
            continue
        bbox = _is_rectangular_frame(region == color)
        if bbox is None:
            continue
        r0, c0, r1, c1 = bbox
        if r0 == 0 or c0 == 0 or r1 == region.shape[0] - 1 or c1 == region.shape[1] - 1:
            continue
        candidates.append((color, bbox))
    assert len(candidates) == 1, "expected exactly one frame in each quadrant"
    return candidates[0]


def _quadrant_info(region: np.ndarray, bg: int, sep: int) -> dict:
    frame_color, (r0, c0, r1, c1) = _find_quadrant_frame(region, bg, sep)
    interior = region[r0 + 1 : r1, c0 + 1 : c1]
    colors = {int(v) for v in np.unique(interior) if int(v) not in {bg, sep, frame_color}}
    assert colors, "frame interior must contain at least one non-background color"
    return {
        "frame_color": int(frame_color),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "colors": colors,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split the board by a solid cross, compare the color sets inside the four
    framed regions, and mark each frame center with its missing color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _most_common_color(g)
    cross_r, cross_c, sep = _find_separator_cross(g)

    quadrants = [
        (slice(0, cross_r), slice(0, cross_c)),
        (slice(0, cross_r), slice(cross_c + 1, g.shape[1])),
        (slice(cross_r + 1, g.shape[0]), slice(0, cross_c)),
        (slice(cross_r + 1, g.shape[0]), slice(cross_c + 1, g.shape[1])),
    ]
    infos = [_quadrant_info(g[rs, cs], bg, sep) for rs, cs in quadrants]

    full_set = set().union(*(info["colors"] for info in infos))
    assert full_set, "complete color set must be nonempty"

    out = g.copy()
    for (rs, cs), info in zip(quadrants, infos):
        missing = full_set - info["colors"]
        assert len(missing) == 1, "each quadrant must miss exactly one color"
        miss_color = int(next(iter(missing)))
        r0, c0, r1, c1 = info["bbox"]
        center_r = (r0 + r1) // 2
        center_c = (c0 + c1) // 2
        for dr, dc in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
            rr = rs.start + center_r + dr
            cc = cs.start + center_c + dc
            if out[rr, cc] == bg:
                out[rr, cc] = miss_color
    return out


def _paint_frame(region: np.ndarray, color: int, top: int, left: int, bottom: int, right: int) -> None:
    region[top : bottom + 1, left] = color
    region[top : bottom + 1, right] = color
    region[top, left : right + 1] = color
    region[bottom, left : right + 1] = color


def _random_interior_points(h: int, w: int, count: int) -> list[tuple[int, int]]:
    pts = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(pts)
    return pts[:count]


def _make_quadrant(height: int, width: int, frame_color: int, colors: list[int]) -> np.ndarray:
    region = np.full((height, width), BG, dtype=int)

    frame_h = random.randrange(5, height if height % 2 == 1 else height - 1, 2)
    frame_w = random.randrange(5, width if width % 2 == 1 else width - 1, 2)
    top = random.randrange(1, height - frame_h)
    left = random.randrange(1, width - frame_w)
    bottom = top + frame_h - 1
    right = left + frame_w - 1
    _paint_frame(region, frame_color, top, left, bottom, right)

    inner_h = frame_h - 2
    inner_w = frame_w - 2
    center = (inner_h // 2, inner_w // 2)
    cross_cells = {
        center,
        (center[0] - 1, center[1]),
        (center[0] + 1, center[1]),
        (center[0], center[1] - 1),
        (center[0], center[1] + 1),
    }

    available = [(r, c) for r in range(inner_h) for c in range(inner_w) if (r, c) not in cross_cells]
    random.shuffle(available)

    cursor = 0
    for color in colors:
        copies = random.randint(2, min(4, max(2, len(available) // max(1, len(colors)))))
        placed = 0
        while placed < copies and cursor < len(available):
            rr, cc = available[cursor]
            cursor += 1
            region[top + 1 + rr, left + 1 + cc] = color
            placed += 1
        if placed == 0:
            rr, cc = center
            region[top + 1 + rr, left + 1 + cc] = color

    extra_noise = random.randint(0, min(4, len(available) - cursor))
    noise_palette = [c for c in range(10) if c not in {BG, SEP, frame_color} and c not in colors]
    for rr, cc in available[cursor : cursor + extra_noise]:
        if random.random() < 0.5 and noise_palette:
            region[top + 1 + rr, left + 1 + cc] = random.choice(noise_palette)

    return region


def _gen_attempt() -> np.ndarray | None:
    qh = [random.randrange(9, 15) for _ in range(2)]
    qw = [random.randrange(9, 15) for _ in range(2)]
    h = qh[0] + qh[1] + 1
    w = qw[0] + qw[1] + 1
    grid = np.full((h, w), BG, dtype=int)
    cross_r = qh[0]
    cross_c = qw[0]
    grid[cross_r, :] = SEP
    grid[:, cross_c] = SEP

    all_colors = random.sample([c for c in range(10) if c not in {BG, SEP}], 8)
    content_colors = all_colors[:4]
    frame_colors = all_colors[4:8]
    missing_perm = random.sample(content_colors, 4)

    specs = [
        (slice(0, cross_r), slice(0, cross_c), qh[0], qw[0]),
        (slice(0, cross_r), slice(cross_c + 1, w), qh[0], qw[1]),
        (slice(cross_r + 1, h), slice(0, cross_c), qh[1], qw[0]),
        (slice(cross_r + 1, h), slice(cross_c + 1, w), qh[1], qw[1]),
    ]

    for idx, (rs, cs, hh, ww) in enumerate(specs):
        present = [c for c in content_colors if c != missing_perm[idx]]
        random.shuffle(present)
        quad = _make_quadrant(hh, ww, frame_colors[idx], present)
        grid[rs, cs] = quad

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None

    infos = [_quadrant_info(grid[rs, cs], BG, SEP) for rs, cs, _, _ in specs]
    full_set = set().union(*(info["colors"] for info in infos))
    if len(full_set) != 4:
        return None
    for info in infos:
        if len(full_set - info["colors"]) != 1:
            return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
