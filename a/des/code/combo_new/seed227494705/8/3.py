import random

import numpy as np


BG = 0
MAROON = 9


def _spans(n, lines):
    spans = []
    prev = 0
    for line in lines:
        if prev <= line - 1:
            spans.append((prev, line - 1))
        prev = line + 1
    if prev <= n - 1:
        spans.append((prev, n - 1))
    return spans


def _grid_spans(g):
    row_lines = [r for r in range(g.shape[0]) if np.all(g[r, :] == MAROON)]
    col_lines = [c for c in range(g.shape[1]) if np.all(g[:, c] == MAROON)]
    return _spans(g.shape[0], row_lines), _spans(g.shape[1], col_lines)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the maroon control tile as a blueprint for placing template tiles.

    Input structure:
    - Maroon separator rows and columns divide the canvas into an equal N by N tile grid.
    - Ordinary non-empty tiles contain one unique foreground color pattern.
    - One tile contains a maroon pattern; its pixels point to target tile coordinates.

    Rule:
    - Locate the maroon control tile and read its foreground pixels in row-major order as target tile positions.
    - Sort all ordinary template tiles by non-black pixel count from largest to smallest.
    - Clear every tile interior, then cycle the sorted templates into the indicated target positions.

    Key insight:
    - The control tile is not content; it is a miniature layout map for the whole tile grid.
    """
    g = np.asarray(grid, dtype=int)
    row_spans, col_spans = _grid_spans(g)
    n = len(row_spans)
    out = np.zeros_like(g)
    out[g == MAROON] = MAROON
    templates = []
    targets = []
    for ti, (r0, r1) in enumerate(row_spans):
        for tj, (c0, c1) in enumerate(col_spans):
            tile = g[r0 : r1 + 1, c0 : c1 + 1]
            non = int(np.count_nonzero(tile != BG))
            if non == 0:
                continue
            if np.any(tile == MAROON):
                for rr, cc in np.argwhere(tile == MAROON):
                    if int(rr) < n and int(cc) < n:
                        targets.append((int(rr), int(cc)))
            else:
                templates.append((non, tile.copy()))
    templates.sort(key=lambda x: -x[0])
    targets.sort()
    if not templates:
        return out
    for idx, (ti, tj) in enumerate(targets):
        if ti >= len(row_spans) or tj >= len(col_spans):
            continue
        r0, r1 = row_spans[ti]
        c0, c1 = col_spans[tj]
        tile = templates[idx % len(templates)][1]
        out[r0 : r1 + 1, c0 : c1 + 1] = tile
    return out


def _random_shape(rng, size, count):
    cells = {(rng.randrange(size), rng.randrange(size))}
    while len(cells) < count:
        r, c = rng.choice(list(cells))
        opts = [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
        rng.shuffle(opts)
        for nr, nc in opts:
            if 0 <= nr < size and 0 <= nc < size:
                cells.add((nr, nc))
                break
    return cells


def generate_input() -> np.ndarray:
    rng = random.Random()
    n = rng.choice([3, 4])
    tile = rng.randint(n, 6)
    size = n * tile + n + 1
    g = np.zeros((size, size), dtype=int)
    for k in range(n + 1):
        line = k * (tile + 1)
        g[line, :] = MAROON
        g[:, line] = MAROON
    spans = [(k * (tile + 1) + 1, k * (tile + 1) + tile) for k in range(n)]
    colors = [1, 2, 3, 4, 6, 7, 8]
    coords = [(r, c) for r in range(n) for c in range(n)]
    master = rng.choice(coords)
    target_count = rng.randint(2, min(n * n - 1, 7))
    targets = rng.sample([p for p in coords if p != master], target_count)
    mr0, mr1 = spans[master[0]]
    mc0, mc1 = spans[master[1]]
    for tr, tc in targets:
        g[mr0 + tr, mc0 + tc] = MAROON
    template_count = rng.randint(2, min(4, len(coords) - 1))
    template_cells = rng.sample([p for p in coords if p != master], template_count)
    for color, cell in zip(rng.sample(colors, template_count), template_cells):
        r0, r1 = spans[cell[0]]
        c0, c1 = spans[cell[1]]
        count = rng.randint(2, min(tile * tile, 9))
        for rr, cc in _random_shape(rng, tile, count):
            g[r0 + rr, c0 + cc] = color
    return g
