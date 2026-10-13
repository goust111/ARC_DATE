from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
SEP = 1
TEMPLATE = 5
SCATTER_COLORS = (2, 3, 4, 6, 7, 8, 9)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_ATTEMPTS = 400
GEN_HEIGHT_RANGE = (11, 18)
GEN_LEFT_WIDTH_RANGE = (8, 12)
GEN_RIGHT_WIDTH_RANGE = (4, 6)
GEN_COLOR_COUNT_RANGE = (3, 5)
GEN_TEMPLATE_AREA_RANGE = (3, 6)
GEN_TEMPLATE_MAX_H = 5
GEN_TEMPLATE_MAX_W = 4
GEN_SCATTER_FILL_MAX = 0.34


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return the inclusive bbox of a boolean mask, matching arc_utils semantics."""
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Crop a grid to an inclusive bbox, matching arc_utils semantics."""
    r0, c0, r1, c1 = box
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    """Return connected components of a boolean mask as coord arrays."""
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        dirs = DIRS4
    else:
        dirs = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]

    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in dirs:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _ordered_colors(left: np.ndarray) -> list[int]:
    counts = Counter(int(v) for v in left.ravel() if int(v) not in (BG, SEP))
    assert 3 <= len(counts) <= 5, "expected 3-5 scatter colors"
    freqs = list(counts.values())
    assert len(freqs) == len(set(freqs)), "scatter frequencies must be unique"
    return [int(color) for color, _count in sorted(counts.items(), key=lambda item: -item[1])]


def _extract_template(right: np.ndarray) -> np.ndarray:
    mask = right == TEMPLATE
    assert mask.any(), "expected a grey template on the right"
    comps = components_mask(mask, connectivity=4)
    assert len(comps) == 1, "expected one 4-connected template"
    box = bbox(mask)
    assert box is not None
    template = crop_bbox(right, box)
    assert np.all((template == BG) | (template == TEMPLATE)), "template region should only contain black and grey"
    return template


def _paint_output(template: np.ndarray, colors: list[int]) -> np.ndarray:
    h, w = map(int, template.shape)
    n = len(colors)
    out = np.zeros((h, n * w + (n - 1)), dtype=int)
    mask = template != BG
    for i, color in enumerate(colors):
        c0 = i * (w + 1)
        patch = np.zeros_like(template)
        patch[mask] = int(color)
        out[:, c0 : c0 + w] = patch
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repeat the cropped right-side template once per left-side color, ordered by descending frequency and recolored per rank.

    Input structure:
    - A single full-height blue separator column splits the grid into a left scatter zone and a right template zone.
    - The left zone is black except for 3-5 non-black, non-blue scatter colors, each appearing a unique number of times.
    - The right zone contains one small grey 4-connected template on black background, possibly padded by empty rows or columns.

    Rule:
    - Count each non-black, non-blue color in the left zone and sort the colors from most frequent to least frequent.
    - Crop the grey template in the right zone to its minimal bounding box.
    - Output one horizontal copy of the cropped template per sorted color, with one black spacer column between neighboring copies.
    - Recolor every non-black cell of the i-th copy to the i-th color in the descending-frequency order.

    Key insight:
    - The left scatter zone determines both how many copies must be produced and the exact left-to-right color order, while the right zone supplies only the shape.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_cols = np.flatnonzero(np.all(g == SEP, axis=0))
    assert len(sep_cols) == 1, "expected exactly one full-height blue separator"
    sep = int(sep_cols[0])
    assert 0 < sep < g.shape[1] - 1, "separator must split the grid into two non-empty zones"

    left = g[:, :sep]
    right = g[:, sep + 1 :]
    colors = _ordered_colors(left)
    template = _extract_template(right)
    out = _paint_output(template, colors)
    assert max(out.shape) <= 30, "output must stay within ARC bounds"
    return out


def _normalize_cells(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    return sorted((r - min_r, c - min_c) for r, c in cells)


def _random_polyomino() -> np.ndarray:
    for _ in range(200):
        area = random.randint(*GEN_TEMPLATE_AREA_RANGE)
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for dr, dc in DIRS4:
                    nxt = (r + dr, c + dc)
                    if nxt not in cells:
                        frontier.add(nxt)
            cells.add(random.choice(tuple(frontier)))

        norm = _normalize_cells(cells)
        h = max(r for r, _c in norm) + 1
        w = max(c for _r, c in norm) + 1
        if not (2 <= h <= GEN_TEMPLATE_MAX_H and 2 <= w <= GEN_TEMPLATE_MAX_W):
            continue
        patch = np.zeros((h, w), dtype=int)
        for r, c in norm:
            patch[r, c] = TEMPLATE
        return patch
    raise RuntimeError("failed to generate a valid template polyomino")


def _sample_counts(n_colors: int, left_area: int) -> list[int] | None:
    max_count = min(9, left_area // max(4, n_colors))
    if max_count < n_colors:
        return None
    pool = list(range(1, max_count + 1))
    for _ in range(120):
        counts = sorted(random.sample(pool, n_colors), reverse=True)
        total = sum(counts)
        if total < n_colors * 2 + 3:
            continue
        if total > int(left_area * GEN_SCATTER_FILL_MAX):
            continue
        return counts
    return None


def _row_major_first_seen(left: np.ndarray) -> list[int]:
    seen: list[int] = []
    for value in left.ravel():
        color = int(value)
        if color in (BG, SEP) or color in seen:
            continue
        seen.append(color)
    return seen


def _numeric_orders(colors: list[int]) -> tuple[list[int], list[int]]:
    return sorted(colors), sorted(colors, reverse=True)


def _place_scatter(left: np.ndarray, ordered_colors: list[int], counts: list[int]) -> None:
    h, w = left.shape
    cells = [(r, c) for r in range(h) for c in range(w)]
    chosen = random.sample(cells, sum(counts))
    random.shuffle(chosen)
    idx = 0
    for color, count in zip(ordered_colors, counts):
        for r, c in chosen[idx : idx + count]:
            left[r, c] = int(color)
        idx += count


def _build_expected(template: np.ndarray, ordered_colors: list[int]) -> np.ndarray:
    return _paint_output(template, ordered_colors)


def _scene_checks(grid: np.ndarray, expected: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if not isinstance(expected, np.ndarray) or expected.ndim != 2 or max(expected.shape) > 30:
        return False

    sep_cols = np.flatnonzero(np.all(grid == SEP, axis=0))
    if len(sep_cols) != 1:
        return False
    sep = int(sep_cols[0])
    left = grid[:, :sep]
    right = grid[:, sep + 1 :]

    left_colors = sorted(set(map(int, left.ravel())) - {BG})
    if SEP in left_colors or TEMPLATE in left_colors:
        return False
    if not (3 <= len(left_colors) <= 5):
        return False

    right_colors = set(map(int, right.ravel()))
    if not right_colors <= {BG, TEMPLATE}:
        return False

    template = _extract_template(right)
    ordered_colors = _ordered_colors(left)
    if not np.array_equal(_build_expected(template, ordered_colors), expected):
        return False

    first_seen = _row_major_first_seen(left)
    if first_seen == ordered_colors:
        return False
    if len(first_seen) != len(ordered_colors):
        return False

    asc = list(reversed(ordered_colors))
    if np.array_equal(_build_expected(template, asc), expected):
        return False
    if np.array_equal(_build_expected(template, first_seen), expected):
        return False

    num_asc, num_desc = _numeric_orders(ordered_colors)
    if ordered_colors == num_asc or ordered_colors == num_desc:
        return False

    try:
        solved = main(grid)
    except Exception:
        return False
    return np.array_equal(solved, expected)


def _gen_attempt() -> np.ndarray | None:
    ordered_colors = random.sample(SCATTER_COLORS, random.randint(*GEN_COLOR_COUNT_RANGE))
    template = _random_polyomino()
    th, tw = map(int, template.shape)
    n = len(ordered_colors)
    out_w = n * tw + (n - 1)
    if out_w > 30 or th > 30:
        return None

    height = random.randint(max(GEN_HEIGHT_RANGE[0], th + 4), GEN_HEIGHT_RANGE[1])
    left_w = random.randint(*GEN_LEFT_WIDTH_RANGE)
    right_w = random.randint(max(GEN_RIGHT_WIDTH_RANGE[0], tw), GEN_RIGHT_WIDTH_RANGE[1])
    width = left_w + 1 + right_w
    if width > 30:
        return None

    counts = _sample_counts(len(ordered_colors), height * left_w)
    if counts is None:
        return None

    grid = np.zeros((height, width), dtype=int)
    grid[:, left_w] = SEP
    left = grid[:, :left_w]
    _place_scatter(left, ordered_colors, counts)

    top = random.randint(0, height - th)
    left_pad = random.randint(0, right_w - tw)
    right_c0 = left_w + 1
    grid[top : top + th, right_c0 + left_pad : right_c0 + left_pad + tw] = template

    expected = _build_expected(template, ordered_colors)
    if _scene_checks(grid, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
