from __future__ import annotations

import random

import numpy as np


COLORS = tuple(range(10))
QUADRANTS = ("tl", "tr", "bl", "br")
MAX_GEN_ATTEMPTS = 500
SIZE_POOL = (5, 5, 5, 7, 7, 9, 11)


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in points]
    cols = [c for _r, c in points]
    return int(min(rows)), int(min(cols)), int(max(rows)), int(max(cols))


def _find_frames(grid: np.ndarray, bg: int) -> list[tuple[int, int, int, int, int]]:
    frames: list[tuple[int, int, int, int, int]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        for comp in _components(grid == color):
            r0, c0, r1, c1 = _bbox(comp)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 5 or w < 5:
                continue
            patch = np.zeros((h, w), dtype=bool)
            for r, c in comp:
                patch[r - r0, c - c0] = True
            ring = np.zeros((h, w), dtype=bool)
            ring[0, :] = True
            ring[-1, :] = True
            ring[:, 0] = True
            ring[:, -1] = True
            if np.array_equal(patch, ring):
                frames.append((color, r0, c0, r1, c1))
    frames.sort(key=lambda item: (item[1], item[2], item[3], item[4], item[0]))
    return frames


def _quadrant_fill(seeds: dict[str, int], inner_h: int, inner_w: int, mr: int, mc: int) -> int:
    row_mid = (inner_h - 1) / 2.0
    col_mid = (inner_w - 1) / 2.0
    assert mr != row_mid and mc != col_mid, "marker must not lie on the interior center row/column"
    vertical = "top" if mr < row_mid else "bottom"
    horizontal = "left" if mc < col_mid else "right"
    return int(seeds[f"{vertical}_{horizontal}"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each framed interior from the corner seed that matches its marker quadrant, then recolor the frame with the marker color.

    Input structure:
    - A dominant background color fills the canvas outside several disjoint hollow rectangles.
    - The four corner cells are singleton seed colors for the top-left, top-right, bottom-left, and bottom-right quadrants.
    - Every rectangle is a one-pixel-thick frame whose interior contains background plus exactly one marker pixel.

    Rule:
    - Detect each rectangular frame and locate its unique interior marker.
    - Determine whether that marker lies in the frame interior's top-left, top-right, bottom-left, or bottom-right quadrant.
    - Fill the whole interior with the seed color from the matching grid corner, overwriting the marker, and recolor the frame border to the marker's original color.

    Key insight:
    The corner singletons form a four-way legend: marker position selects the fill color, while marker color itself becomes the frame's new border.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _mode_color(g)
    frames = _find_frames(g, bg)
    assert frames, "expected at least one hollow rectangular frame"

    seeds = {
        "top_left": int(g[0, 0]),
        "top_right": int(g[0, -1]),
        "bottom_left": int(g[-1, 0]),
        "bottom_right": int(g[-1, -1]),
    }

    out = g.copy()
    for border_color, r0, c0, r1, c1 in frames:
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        marker_pts = np.argwhere(inner != bg)
        assert len(marker_pts) == 1, "each frame interior must contain exactly one non-background marker"
        mr, mc = map(int, marker_pts[0])
        marker_color = int(inner[mr, mc])
        assert marker_color != border_color, "marker color must differ from frame color"

        fill_color = _quadrant_fill(seeds, inner.shape[0], inner.shape[1], mr, mc)
        out[r0 + 1 : r1, c0 + 1 : c1] = fill_color
        out[r0, c0 : c1 + 1] = marker_color
        out[r1, c0 : c1 + 1] = marker_color
        out[r0 : r1 + 1, c0] = marker_color
        out[r0 : r1 + 1, c1] = marker_color

    return out


def _choose_colors(n_rects: int) -> tuple[int, int, list[int], list[int]]:
    palette = list(COLORS)
    random.shuffle(palette)

    bg = int(palette[0])
    frame = int(palette[1])
    seed_colors = [int(v) for v in palette[2:6]]
    distinct_marker_pool = [int(v) for v in palette[6:10]]

    if n_rects <= 4:
        marker_colors = distinct_marker_pool[:n_rects]
    else:
        marker_colors = distinct_marker_pool[:]
        marker_colors.append(int(random.choice(seed_colors)))
    random.shuffle(marker_colors)
    return bg, frame, seed_colors, marker_colors


def _size_choices(limit: int, n_rects: int) -> list[int]:
    choices = [size for size in SIZE_POOL if size <= limit]
    if n_rects >= 5:
        choices = [size for size in choices if size <= 9]
    assert choices, "grid too small for any valid frame size"
    return choices


def _sample_rectangles(height: int, width: int, n_rects: int) -> list[tuple[int, int, int, int]] | None:
    h_choices = _size_choices(height - 2, n_rects)
    w_choices = _size_choices(width - 2, n_rects)
    sizes = [(random.choice(h_choices), random.choice(w_choices)) for _ in range(n_rects)]
    sizes.sort(key=lambda hw: hw[0] * hw[1], reverse=True)

    occupied = np.zeros((height, width), dtype=bool)
    rects: list[tuple[int, int, int, int]] = []
    for outer_h, outer_w in sizes:
        positions: list[tuple[int, int]] = []
        for r0 in range(1, height - outer_h):
            r1 = r0 + outer_h - 1
            for c0 in range(1, width - outer_w):
                c1 = c0 + outer_w - 1
                if occupied[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2].any():
                    continue
                positions.append((int(r0), int(c0)))
        if not positions:
            return None
        r0, c0 = random.choice(positions)
        rects.append((int(r0), int(c0), int(outer_h), int(outer_w)))
        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1
        occupied[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2] = True

    random.shuffle(rects)
    return rects


def _quadrants_for_case(n_rects: int) -> list[str]:
    quadrants: list[str] = []
    while len(quadrants) < n_rects:
        batch = list(QUADRANTS)
        random.shuffle(batch)
        quadrants.extend(batch)
    return quadrants[:n_rects]


def _marker_position(inner_h: int, inner_w: int, quadrant: str) -> tuple[int, int]:
    center_r = inner_h // 2
    center_c = inner_w // 2
    if quadrant == "tl":
        rows = range(0, center_r)
        cols = range(0, center_c)
    elif quadrant == "tr":
        rows = range(0, center_r)
        cols = range(center_c + 1, inner_w)
    elif quadrant == "bl":
        rows = range(center_r + 1, inner_h)
        cols = range(0, center_c)
    else:
        rows = range(center_r + 1, inner_h)
        cols = range(center_c + 1, inner_w)
    return int(random.choice(list(rows))), int(random.choice(list(cols)))


def _validate_case(grid: np.ndarray, n_rects: int) -> bool:
    bg = _mode_color(grid)
    seeds = [int(grid[0, 0]), int(grid[0, -1]), int(grid[-1, 0]), int(grid[-1, -1])]
    if len(set(seeds)) != 4:
        return False

    frames = _find_frames(grid, bg)
    if len(frames) != n_rects:
        return False
    if len({color for color, *_rest in frames}) != 1:
        return False
    frame_color = frames[0][0]
    if frame_color == bg or frame_color in seeds:
        return False

    marker_colors: set[int] = set()
    out = main(grid)
    for _color, r0, c0, r1, c1 in frames:
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        marker_pts = np.argwhere(inner != bg)
        if len(marker_pts) != 1:
            return False
        mr, mc = map(int, marker_pts[0])
        marker_color = int(inner[mr, mc])
        if marker_color in marker_colors:
            return False
        marker_colors.add(marker_color)

        fill = _quadrant_fill(
            {
                "top_left": seeds[0],
                "top_right": seeds[1],
                "bottom_left": seeds[2],
                "bottom_right": seeds[3],
            },
            inner.shape[0],
            inner.shape[1],
            mr,
            mc,
        )
        if not np.all(out[r0 + 1 : r1, c0 + 1 : c1] == fill):
            return False
        if not np.all(out[r0, c0 : c1 + 1] == marker_color):
            return False
        if not np.all(out[r1, c0 : c1 + 1] == marker_color):
            return False
        if not np.all(out[r0 : r1 + 1, c0] == marker_color):
            return False
        if not np.all(out[r0 : r1 + 1, c1] == marker_color):
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        n_rects = random.choice((3, 4, 4, 5))
        min_dim = {3: 18, 4: 22, 5: 24}[n_rects]
        height = random.randint(min_dim, 30)
        width = random.randint(min_dim, 30)

        rects = _sample_rectangles(height, width, n_rects)
        if rects is None:
            continue

        bg, frame_color, seed_colors, marker_colors = _choose_colors(n_rects)
        grid = np.full((height, width), bg, dtype=int)
        grid[0, 0] = seed_colors[0]
        grid[0, -1] = seed_colors[1]
        grid[-1, 0] = seed_colors[2]
        grid[-1, -1] = seed_colors[3]

        quadrants = _quadrants_for_case(n_rects)
        for (r0, c0, outer_h, outer_w), quadrant, marker_color in zip(rects, quadrants, marker_colors, strict=False):
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1
            grid[r0, c0 : c1 + 1] = frame_color
            grid[r1, c0 : c1 + 1] = frame_color
            grid[r0 : r1 + 1, c0] = frame_color
            grid[r0 : r1 + 1, c1] = frame_color

            inner_h = outer_h - 2
            inner_w = outer_w - 2
            mr, mc = _marker_position(inner_h, inner_w, quadrant)
            grid[r0 + 1 + mr, c0 + 1 + mc] = marker_color

        if _validate_case(grid, n_rects):
            return grid

    raise RuntimeError("failed to generate a valid ARC input within retry budget")
