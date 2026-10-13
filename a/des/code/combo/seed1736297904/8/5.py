import math
import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9

PALETTE_CHOICES = (1, 2, 3, 4, 6, 7, 8)
BRANCH_TYPES = ("general", "row", "col", "cell")

RECT_COUNT_RANGE = (4, 6)
TOP_PANEL_HEIGHT_RANGE = (2, 5)
GRID_WIDTH_MAX = 28
GRID_HEIGHT_MAX = 30
MAX_GENERATION_ATTEMPTS = 300


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _find_separator(g: np.ndarray) -> int:
    h, _ = g.shape
    matches: list[int] = []
    for r in np.where(np.all(g == MAROON, axis=1))[0].tolist():
        if r <= 0 or r >= h - 1:
            continue
        upper = g[:r]
        content_rows = np.where(np.any(upper != BLACK, axis=1))[0]
        if len(content_rows) != 1:
            continue
        palette_r = int(content_rows[0])
        cols = np.where(upper[palette_r] != BLACK)[0]
        if cols.size == 0 or int(cols[-1] - cols[0] + 1) != int(cols.size):
            continue
        colors = [int(v) for v in upper[palette_r, cols]]
        if len(set(colors)) != len(colors):
            continue
        if not np.any(g[r + 1 :] == GREY):
            continue
        matches.append(int(r))
    assert len(matches) == 1, "expected exactly one valid separator row"
    return matches[0]


def _parse_instance(grid: np.ndarray) -> tuple[int, dict[int, int], list[tuple[int, int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    sep = _find_separator(g)

    upper = g[:sep]
    lower = g[sep + 1 :]
    assert upper.size > 0 and lower.size > 0, "both panels must exist"

    content_rows = np.where(np.any(upper != BLACK, axis=1))[0]
    assert len(content_rows) == 1, "upper panel must contain exactly one palette row"
    palette_r = int(content_rows[0])
    palette_cols = np.where(upper[palette_r] != BLACK)[0]
    assert palette_cols.size > 0, "palette row must be non-empty"
    assert int(palette_cols[-1] - palette_cols[0] + 1) == int(palette_cols.size), "palette must be compact"

    palette = [int(v) for v in upper[palette_r, palette_cols[0] : palette_cols[-1] + 1]]
    assert len(set(palette)) == len(palette), "palette colors must be distinct"
    for r in range(upper.shape[0]):
        if r == palette_r:
            continue
        assert not np.any(upper[r] != BLACK), "upper panel may only contain the palette row"

    assert np.all(np.isin(lower, [BLACK, GREY])), "lower panel may only contain background and grey rectangles"
    mapping = {k + 1: color for k, color in enumerate(palette)}

    rectangles: list[tuple[int, int, int, int]] = []
    for coords in _components_mask(lower == GREY):
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert int(coords.shape[0]) == h * w, "grey object must be a solid rectangle"
        assert 1 <= h <= len(palette) and 1 <= w <= len(palette), "rectangle dimensions must fit the palette"
        rectangles.append((sep + 1 + r0, c0, h, w))

    assert rectangles, "expected at least one rectangle"
    return sep, mapping, rectangles


def main(grid: np.ndarray) -> np.ndarray:
    """
    The top panel is a left-to-right size-to-color legend. Each grey rectangle in
    the lower panel keeps its grey interior while its horizontal sides are colored
    by its height and its vertical sides are colored by its width.
    """
    g = np.asarray(grid, dtype=int)
    sep, mapping, rectangles = _parse_instance(g)

    out = g.copy()
    for r0, c0, h, w in rectangles:
        r1 = r0 + h - 1
        c1 = c0 + w - 1
        row_color = mapping[h]
        col_color = mapping[w]

        if h == 1 and w == 1:
            out[r0, c0] = row_color
        elif h == 1:
            out[r0, c0 : c1 + 1] = row_color
        elif w == 1:
            out[r0 : r1 + 1, c0] = col_color
        else:
            out[r0, c0 : c1 + 1] = row_color
            out[r1, c0 : c1 + 1] = row_color
            out[r0 + 1 : r1, c0] = col_color
            out[r0 + 1 : r1, c1] = col_color

    assert out.shape == g.shape
    assert np.array_equal(out[sep], g[sep]), "separator row must remain unchanged"
    return out


def _sample_rectangles(n: int) -> list[tuple[int, int]]:
    count = random.randint(*RECT_COUNT_RANGE)
    kinds = list(BRANCH_TYPES)
    random.shuffle(kinds)
    kinds.extend(random.choice(BRANCH_TYPES) for _ in range(count - len(BRANCH_TYPES)))
    random.shuffle(kinds)

    rects: list[tuple[int, int]] = []
    for kind in kinds[:count]:
        if kind == "general":
            h = random.randint(2, n)
            w = random.randint(2, n)
        elif kind == "row":
            h = 1
            w = random.randint(2, n)
        elif kind == "col":
            h = random.randint(2, n)
            w = 1
        else:
            h = 1
            w = 1
        rects.append((h, w))

    if not any(h == n or w == n for h, w in rects):
        idx = random.randrange(len(rects))
        h, w = rects[idx]
        if h > 1 and w > 1:
            if random.random() < 0.5:
                h = n
            else:
                w = n
        elif h == 1 and w == 1:
            if random.random() < 0.5:
                h = n
            else:
                w = n
        elif h == 1:
            w = n
        else:
            h = n
        rects[idx] = (h, w)

    return rects


def _place_rectangles(lower_h: int, width: int, rects: list[tuple[int, int]]) -> list[tuple[int, int, int, int]] | None:
    order = sorted(range(len(rects)), key=lambda i: rects[i][0] * rects[i][1], reverse=True)
    blocked = np.zeros((lower_h, width), dtype=bool)
    placed: list[tuple[int, int, int, int] | None] = [None] * len(rects)

    for idx in order:
        h, w = rects[idx]
        placed_here = False
        for _ in range(300):
            r0 = random.randint(0, lower_h - h)
            c0 = random.randint(0, width - w)
            rs = max(0, r0 - 1)
            re = min(lower_h, r0 + h + 1)
            cs = max(0, c0 - 1)
            ce = min(width, c0 + w + 1)
            if blocked[rs:re, cs:ce].any():
                continue
            blocked[rs:re, cs:ce] = True
            placed[idx] = (r0, c0, h, w)
            placed_here = True
            break
        if not placed_here:
            return None

    return [p for p in placed if p is not None]


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(2, len(PALETTE_CHOICES))
    palette = random.sample(PALETTE_CHOICES, n)
    rects = _sample_rectangles(n)

    max_h = max(h for h, _ in rects)
    max_w = max(w for _, w in rects)
    padded_area = sum((h + 2) * (w + 2) for h, w in rects)

    top_h = random.randint(*TOP_PANEL_HEIGHT_RANGE)
    min_width = max(n + 2, max_w + 1, int(math.ceil(math.sqrt(padded_area))) - 1)
    if min_width > GRID_WIDTH_MAX:
        return None
    width = random.randint(min_width, GRID_WIDTH_MAX)

    lower_h_min = max(max_h, int(math.ceil(padded_area / max(width - 1, 1))))
    lower_h_max = GRID_HEIGHT_MAX - top_h - 1
    if lower_h_min > lower_h_max:
        return None
    lower_h = random.randint(lower_h_min, min(lower_h_max, lower_h_min + 8))

    placements = _place_rectangles(lower_h, width, rects)
    if placements is None:
        return None

    grid = np.full((top_h + 1 + lower_h, width), BLACK, dtype=int)
    palette_row = random.randint(0, top_h - 1)
    palette_start = random.randint(0, width - n)
    grid[palette_row, palette_start : palette_start + n] = palette
    grid[top_h, :] = MAROON

    for r0, c0, h, w in placements:
        grid[top_h + 1 + r0 : top_h + 1 + r0 + h, c0 : c0 + w] = GREY

    sep, mapping, parsed_rects = _parse_instance(grid)
    assert sep == top_h
    assert mapping == {k + 1: palette[k] for k in range(n)}
    assert len(parsed_rects) == len(rects)

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
