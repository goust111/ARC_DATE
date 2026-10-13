import itertools
import random

import numpy as np


SEP = 5
CTRL_BG = 0
CTRL_FG = 7
TOP_COLORS = (1, 2, 3, 4, 6, 8, 9)
GEN_RETRIES = 400


def _split_regions(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    sep_rows = np.flatnonzero(np.all(g == SEP, axis=1))
    assert len(sep_rows) == 1, "expected exactly one horizontal gray separator"
    r_sep = int(sep_rows[0])
    assert 0 < r_sep < g.shape[0] - 1

    lower = g[r_sep + 1 :, :]
    sep_cols = np.flatnonzero(np.all(lower == SEP, axis=0))
    assert len(sep_cols) == 1, "expected exactly one vertical gray separator in the lower band"
    c_sep = int(sep_cols[0])
    assert 0 < c_sep < g.shape[1] - 1

    top = g[:r_sep, :]
    control_a = lower[:, :c_sep]
    control_b = lower[:, c_sep + 1 :]
    assert top.size and control_a.size and control_b.size
    return top, control_a, control_b


def _find_valid_pairs(region: np.ndarray) -> list[tuple[tuple[int, int], np.ndarray]]:
    colors = sorted(int(v) for v in np.unique(region) if int(v) != SEP)
    valid = []
    for a, b in itertools.combinations(colors, 2):
        mask = np.isin(region, (a, b))
        if not np.any(mask):
            continue
        rows = mask.any(axis=1)
        cols = mask.any(axis=0)
        sub = region[np.ix_(rows, cols)]
        if sub.size == 0:
            continue
        if np.all(np.isin(sub, (a, b))) and set(map(int, np.unique(sub))) == {a, b}:
            valid.append(((a, b), sub.copy()))
    return valid


def _tile(template: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    th, tw = template.shape
    rr = np.arange(out_h) % th
    cc = np.arange(out_w) % tw
    return template[np.ix_(rr, cc)].copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the unique hidden two-color image and tile it to a control-specified size.

    Input structure:
    - A gray horizontal line separates a mixed-color upper region from a lower control band.
    - In the lower band, a gray vertical line splits two black-background control regions.
    - The upper region contains many colors; exactly one color pair forms a complete row-by-column product.
    - The left and right control regions contain orange foreground pixels whose counts encode output height and width.

    Rule:
    - In the upper region, enumerate every color pair and find the unique pair whose combined pixels occupy every cell in the cross product of their active rows and active columns.
    - Extract the induced subgrid on exactly those rows and columns; this two-color image is the template.
    - Count non-black pixels in the left control region to get the output height, and in the right control region to get the output width.
    - Periodically tile the extracted template over that output rectangle, truncating at the requested size.

    Key insight:
    - The hidden image is identified algebraically by complete row-column coverage, not by connectivity or rarity, and the extracted image then becomes the tile used in a second, separate tiling step.
    """
    top, control_a, control_b = _split_regions(np.asarray(grid, dtype=int))
    candidates = _find_valid_pairs(top)
    assert len(candidates) == 1, "expected exactly one valid two-color row/column product"
    template = candidates[0][1]

    out_h = int(np.count_nonzero(control_a != CTRL_BG))
    out_w = int(np.count_nonzero(control_b != CTRL_BG))
    assert out_h > 0 and out_w > 0
    return _tile(template, out_h, out_w)


def _sample_indices(rng: random.Random, size: int, k: int, force_gap: bool) -> list[int]:
    while True:
        picked = sorted(rng.sample(range(size), k))
        if not force_gap:
            return picked
        if any(b - a > 1 for a, b in zip(picked, picked[1:])):
            return picked


def _make_template(rng: random.Random, h: int, w: int, colors: tuple[int, int]) -> np.ndarray:
    a, b = colors
    mode = rng.choice(("checker", "row_stripes", "col_stripes", "random"))
    template = np.full((h, w), a, dtype=int)

    if mode == "checker":
        for r in range(h):
            for c in range(w):
                template[r, c] = a if (r + c) % 2 == 0 else b
    elif mode == "row_stripes":
        for r in range(h):
            template[r, :] = a if r % 2 == 0 else b
    elif mode == "col_stripes":
        for c in range(w):
            template[:, c] = a if c % 2 == 0 else b
    else:
        bias = rng.uniform(0.35, 0.65)
        for r in range(h):
            for c in range(w):
                template[r, c] = a if rng.random() < bias else b

    if len(set(map(int, template.ravel()))) == 1:
        template[-1, -1] = b if template[-1, -1] == a else a
    return template


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _grow_cells(rng: random.Random, h: int, w: int, count: int) -> set[tuple[int, int]] | None:
    start = (rng.randrange(h), rng.randrange(w))
    cells = {start}
    frontier = {
        (nr, nc)
        for nr, nc in _neighbors4(*start)
        if 0 <= nr < h and 0 <= nc < w
    }
    while len(cells) < count:
        if not frontier:
            return None
        cell = rng.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        for nr, nc in _neighbors4(*cell):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                frontier.add((nr, nc))
    return cells


def _make_control_region(rng: random.Random, h: int, w: int, count: int) -> np.ndarray:
    region = np.zeros((h, w), dtype=int)
    mode = rng.choice(("blob", "blob", "scatter"))
    cells = None
    if mode == "blob":
        cells = _grow_cells(rng, h, w, count)
    if cells is None:
        cells = set(rng.sample([(r, c) for r in range(h) for c in range(w)], count))
    for r, c in cells:
        region[r, c] = CTRL_FG
    return region


def _base_top(fillers: tuple[int, int, int], h: int, w: int) -> np.ndarray:
    top = np.empty((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            top[r, c] = fillers[(r + 2 * c) % 3]
    return top


def _paint_cross(top: np.ndarray, rows: list[int], cols: list[int], colors: tuple[int, int]) -> None:
    a, b = colors
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            top[r, c] = a if (i + j) % 2 == 0 else b


def _add_decoy(
    rng: random.Random,
    top: np.ndarray,
    used_rows: set[int],
    used_cols: set[int],
    colors: tuple[int, int],
) -> dict | None:
    free_rows = [r for r in range(top.shape[0]) if r not in used_rows]
    free_cols = [c for c in range(top.shape[1]) if c not in used_cols]
    if len(free_rows) < 2 or len(free_cols) < 2:
        return None

    dh = rng.randint(2, min(4, len(free_rows)))
    dw = rng.randint(2, min(4, len(free_cols)))
    rows = sorted(rng.sample(free_rows, dh))
    cols = sorted(rng.sample(free_cols, dw))
    mode = rng.choice(("hole", "stray"))

    hole = None
    stray = None
    if mode == "hole":
        hole = (rng.choice(rows), rng.choice(cols))

    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            if hole == (r, c):
                continue
            top[r, c] = colors[(i + j) % 2]

    if mode == "stray":
        stray_rows = [r for r in range(top.shape[0]) if r not in rows]
        stray_cols = cols[:] if rng.random() < 0.5 else [c for c in range(top.shape[1]) if c not in cols]
        if not stray_cols:
            stray_cols = cols[:]
        sr = rng.choice(stray_rows)
        sc = rng.choice(stray_cols)
        top[sr, sc] = rng.choice(colors)
        stray = (sr, sc)

    return {"mode": mode, "rows": rows, "cols": cols, "colors": colors, "hole": hole, "stray": stray}


def _validate_decoy(region: np.ndarray, decoy: dict) -> None:
    a, b = decoy["colors"]
    rows = decoy["rows"]
    cols = decoy["cols"]
    sub = region[np.ix_(rows, cols)]
    if decoy["mode"] == "hole":
        bad = np.count_nonzero(~np.isin(sub, (a, b)))
        assert bad == 1, "hole decoy must miss exactly one cross-product cell"
    else:
        sr, sc = decoy["stray"]
        assert region[sr, sc] in (a, b)
        assert sr not in rows or sc not in cols


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    top_h = rng.randint(8, 14)
    bottom_h = rng.randint(4, 6)
    left_w = rng.randint(5, 8)
    right_w = rng.randint(5, 8)
    total_w = left_w + 1 + right_w

    colors = rng.sample(TOP_COLORS, len(TOP_COLORS))
    target_colors = (colors[0], colors[1])
    decoy_colors = (colors[2], colors[3])
    fillers = (colors[4], colors[5], colors[6])

    max_template_h = min(6, top_h - 2)
    max_template_w = min(7, total_w - 2)
    if max_template_h < 2 or max_template_w < 2:
        return None
    template_h = rng.randint(2, max_template_h)
    template_w = rng.randint(2, max_template_w)

    max_out_h = min(12, bottom_h * left_w)
    max_out_w = min(12, bottom_h * right_w)
    if max_out_h < template_h or max_out_w < template_w:
        return None
    out_h = rng.randint(template_h, max_out_h)
    out_w = rng.randint(template_w, max_out_w)

    force_gap = rng.random() < 0.7
    rows = _sample_indices(rng, top_h, template_h, force_gap)
    cols = _sample_indices(rng, total_w, template_w, force_gap)
    template = _make_template(rng, template_h, template_w, target_colors)

    top = _base_top(fillers, top_h, total_w)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            top[r, c] = int(template[i, j])

    decoy = _add_decoy(rng, top, set(rows), set(cols), decoy_colors)
    if decoy is None:
        return None

    control_a = _make_control_region(rng, bottom_h, left_w, out_h)
    control_b = _make_control_region(rng, bottom_h, right_w, out_w)

    grid = np.full((top_h + 1 + bottom_h, total_w), SEP, dtype=int)
    grid[:top_h, :] = top
    grid[top_h + 1 :, :left_w] = control_a
    grid[top_h + 1 :, left_w + 1 :] = control_b

    valid = _find_valid_pairs(top)
    if len(valid) != 1:
        return None
    pair, extracted = valid[0]
    if set(pair) != set(target_colors):
        return None
    if not np.array_equal(extracted, template):
        return None

    _validate_decoy(top, decoy)

    expected = _tile(template, out_h, out_w)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
