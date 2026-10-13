import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
GREY = 5
BORDER_COLORS = (1, 3, 4, 6, 7, 8, 9)

PATTERN_POOL = (
    np.array([[GREY, BLACK], [BLACK, GREY]], dtype=int),
    np.array([[BLACK, GREY], [GREY, BLACK]], dtype=int),
    np.array([[GREY, BLACK], [GREY, BLACK]], dtype=int),
    np.array([[BLACK, GREY], [BLACK, GREY]], dtype=int),
    np.array([[GREY, GREY], [GREY, BLACK]], dtype=int),
    np.array([[GREY, GREY], [BLACK, GREY]], dtype=int),
    np.array([[GREY, BLACK], [GREY, GREY]], dtype=int),
    np.array([[BLACK, GREY], [GREY, GREY]], dtype=int),
)

GEN_RETRIES = 400
PLACE_RETRIES = 120


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _parse_legend(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert w >= 2
    c0 = w - 2
    patterns: list[np.ndarray] = []
    r = 0
    while r < h:
        if np.all(g[r, c0:w] == BLACK):
            r += 1
            continue
        assert r + 1 < h, "legend block must be 2 rows tall"
        block = g[r : r + 2, c0:w]
        vals = set(map(int, block.reshape(-1)))
        assert vals <= {BLACK, GREY}, "legend uses only black and grey"
        assert bool((block[0] == GREY).any()) and bool((block[1] == GREY).any()), "each legend row must contain grey"
        patterns.append(block.copy())
        r += 2
        if r < h and np.all(g[r, c0:w] == BLACK):
            r += 1
    assert patterns, "legend must contain at least one 2x2 pattern"
    return patterns


def _find_rectangles(grid: np.ndarray) -> list[dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    rects: list[dict[str, int]] = []
    for color in sorted(set(map(int, g.reshape(-1))) - {BLACK, RED, GREY}):
        comps = _components(g == color)
        assert len(comps) == 1, "each rectangle color should form one component"
        pts = comps[0]
        rows = [r for r, _ in pts]
        cols = [c for _, c in pts]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "rectangle needs a non-empty border box"
        expected = set()
        for rr in range(r0, r1 + 1):
            expected.add((rr, c0))
            expected.add((rr, c1))
        for cc in range(c0, c1 + 1):
            expected.add((r0, cc))
            expected.add((r1, cc))
        assert set(pts) == expected, "component must be a 1-pixel rectangular outline"
        rects.append({"color": color, "r0": r0, "c0": c0, "r1": r1, "c1": c1})
    assert rects, "expected at least one rectangle"
    return sorted(rects, key=lambda item: (item["r0"], item["c0"]))


def _tile_pattern(pattern: np.ndarray, color: int, height: int, width: int) -> np.ndarray:
    out = np.zeros((height, width), dtype=int)
    for r in range(height):
        for c in range(width):
            out[r, c] = color if int(pattern[r % 2, c % 2]) == GREY else BLACK
    return out


def _sample_interior_dims(rng: random.Random, markers: int) -> tuple[int, int]:
    min_dim = 2 if markers > 0 else 1
    for _ in range(80):
        h = rng.randint(min_dim, 6)
        w = rng.randint(min_dim, 7)
        if h * w >= max(1, markers):
            return h, w
    return max(2, markers), max(2, (markers + 1) // 2)


def _place_rectangles(
    rng: random.Random,
    height: int,
    width: int,
    colors: list[int],
    marker_counts: list[int],
) -> list[dict[str, object]] | None:
    usable_w = width - 3
    occupied = np.zeros((height, usable_w), dtype=bool)
    rects: list[dict[str, object]] = []
    for color, markers in zip(colors, marker_counts):
        placed = False
        for _ in range(PLACE_RETRIES):
            inner_h, inner_w = _sample_interior_dims(rng, markers)
            box_h, box_w = inner_h + 2, inner_w + 2
            if box_h > height or box_w > usable_w:
                continue
            r0 = rng.randint(0, height - box_h)
            c0 = rng.randint(0, usable_w - box_w)
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(height, r0 + box_h + 1)
            cc1 = min(usable_w, c0 + box_w + 1)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            occupied[rr0:rr1, cc0:cc1] = True
            cells = [(r, c) for r in range(r0 + 1, r0 + 1 + inner_h) for c in range(c0 + 1, c0 + 1 + inner_w)]
            rng.shuffle(cells)
            markers_pts = cells[:markers]
            rects.append(
                {
                    "color": color,
                    "r0": r0,
                    "c0": c0,
                    "r1": r0 + box_h - 1,
                    "c1": c0 + box_w - 1,
                    "markers": markers_pts,
                }
            )
            placed = True
            break
        if not placed:
            return None
    return rects


def _draw_input(
    height: int,
    width: int,
    legend_start: int,
    patterns: list[np.ndarray],
    rects: list[dict[str, object]],
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    c0 = width - 2
    for idx, pattern in enumerate(patterns):
        r0 = legend_start + 3 * idx
        grid[r0 : r0 + 2, c0:width] = pattern
    for rect in rects:
        color = int(rect["color"])
        r0, c0, r1, c1 = (int(rect[key]) for key in ("r0", "c0", "r1", "c1"))
        grid[r0, c0 : c1 + 1] = color
        grid[r1, c0 : c1 + 1] = color
        grid[r0 : r1 + 1, c0] = color
        grid[r0 : r1 + 1, c1] = color
        for r, c in rect["markers"]:
            grid[r, c] = RED
    return grid


def _validate_generated(grid: np.ndarray, expected_rects: int, legend_count: int) -> bool:
    patterns = _parse_legend(grid)
    rects = _find_rectangles(grid)
    if len(patterns) != legend_count or len(rects) != expected_rects:
        return False
    for rect in rects:
        r0, c0, r1, c1 = rect["r0"], rect["c0"], rect["r1"], rect["c1"]
        markers = int((grid[r0 + 1 : r1, c0 + 1 : c1] == RED).sum())
        if markers > legend_count:
            return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each rectangular outline with a tiled legend texture selected by its red marker count.

    Input structure:
    - Black background with several closed one-pixel rectangular outlines, each in a distinct non-black color.
    - Some outline interiors contain scattered Red marker pixels on otherwise black cells; other interiors have no markers.
    - The rightmost two columns contain a vertical texture legend made of separated 2x2 blocks that use only Black and Grey.

    Rule:
    - Count the Red markers inside each rectangle to obtain an index K.
    - When K >= 1, select the K-th legend texture from top to bottom and tile that 2x2 pattern across the rectangle interior starting at the interior's top-left cell.
    - Replace Grey cells of the tiled texture with the rectangle's own border color, keep Black cells black, and overwrite any original markers.
    - Rectangles with K = 0 keep black interiors, and the legend area itself is cleared to Black in the output.

    Key insight:
    - The red pixels are not decoration: their count indexes a texture in the far-right legend, and that legend's grey pixels are recolored to the local frame color when tiled.
    """
    g = np.asarray(grid, dtype=int)
    patterns = _parse_legend(g)
    rects = _find_rectangles(g)

    out = g.copy()
    out[:, -2:] = BLACK

    for rect in rects:
        r0, c0, r1, c1, color = (int(rect[key]) for key in ("r0", "c0", "r1", "c1", "color"))
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        marker_count = int((interior == RED).sum())
        if marker_count == 0:
            continue
        assert marker_count <= len(patterns), "marker count must select an existing legend entry"
        fill = _tile_pattern(patterns[marker_count - 1], color, r1 - r0 - 1, c1 - c0 - 1)
        out[r0 + 1 : r1, c0 + 1 : c1] = fill

    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        legend_count = rng.randint(3, 5)
        legend_height = 3 * legend_count - 1
        height = rng.randint(max(legend_height, 12), 30)
        width = rng.randint(15, 30)
        legend_start = rng.randint(0, height - legend_height)

        patterns = [pattern.copy() for pattern in rng.sample(PATTERN_POOL, legend_count)]
        rect_count = rng.randint(2, 4)
        colors = rng.sample(list(BORDER_COLORS), rect_count)
        marker_counts = [rng.randint(0, legend_count) for _ in range(rect_count)]
        if all(count == 0 for count in marker_counts):
            marker_counts[rng.randrange(rect_count)] = rng.randint(1, legend_count)
        if all(count > 0 for count in marker_counts):
            marker_counts[rng.randrange(rect_count)] = 0
        if len({count for count in marker_counts if count > 0}) == 1 and rect_count >= 3:
            idxs = [i for i, count in enumerate(marker_counts) if count > 0]
            if len(idxs) >= 2:
                marker_counts[idxs[0]] = 1
                marker_counts[idxs[1]] = legend_count

        rects = _place_rectangles(rng, height, width, colors, marker_counts)
        if rects is None:
            continue
        grid = _draw_input(height, width, legend_start, patterns, rects)
        if _validate_generated(grid, rect_count, legend_count):
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
