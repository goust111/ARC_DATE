from __future__ import annotations

import random

import numpy as np


BG = 0
SEPARATOR = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

# --------------- solver ---------------


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size, "expected at least one marked cell"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_separator_row(grid: np.ndarray) -> int:
    rows: list[int] = []
    for r in range(grid.shape[0]):
        color = int(grid[r, 0])
        if color == BG:
            continue
        if np.all(grid[r] == color) and np.any(grid[:r] != BG) and np.any(grid[r + 1 :] != BG):
            rows.append(r)
    assert len(rows) == 1, f"expected exactly one separator row, got {rows}"
    return rows[0]


def _crop_non_black(region: np.ndarray) -> np.ndarray:
    mask = region != BG
    assert np.any(mask), "expected a non-background object"
    r0, c0, r1, c1 = _bbox(mask)
    return np.asarray(region[r0 : r1 + 1, c0 : c1 + 1], dtype=int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only the source pixels that sit under the template's black cut-outs.
    """
    g = np.asarray(grid, dtype=int)
    sep_r = _find_separator_row(g)

    source = _crop_non_black(g[:sep_r, :])
    template = _crop_non_black(g[sep_r + 1 :, :])

    assert source.shape == template.shape, "source and template must have matching sizes"
    assert np.all(source != BG), "source crop must be a solid rectangle"
    assert np.any(template == BG) and np.any(template != BG), "template must mix holes and solid cells"

    return np.where(template == BG, source, BG).astype(int)


# --------------- generator ---------------

RECT_H_RANGE = (4, 8)
RECT_W_RANGE = (4, 10)
TOP_PAD_RANGE = (1, 4)
BOTTOM_PAD_RANGE = (1, 4)
ABOVE_SEPARATOR_GAP = (1, 3)
BELOW_SEPARATOR_GAP = (1, 3)
WIDTH_SLACK_RANGE = (3, 10)
SOURCE_COLOR_COUNT = (2, 4)
GEN_ATTEMPTS = 400

RNG = random.Random()


def _random_partition(total: int, parts: int) -> list[int]:
    assert 1 <= parts <= total
    if parts == 1:
        return [int(total)]
    cuts = sorted(RNG.sample(range(1, total), parts - 1))
    points = [0] + cuts + [total]
    return [int(points[i + 1] - points[i]) for i in range(parts)]


def _nonrepeating_colors(length: int, colors: list[int]) -> list[int]:
    out: list[int] = []
    for _ in range(length):
        choices = [c for c in colors if not out or c != out[-1]]
        out.append(int(RNG.choice(choices)))
    if len(set(out)) == 1:
        idx = RNG.randrange(length)
        replacements = [c for c in colors if c != out[idx]]
        out[idx] = int(RNG.choice(replacements))
    return out


def _source_stripes(h: int, w: int, colors: list[int], horizontal: bool) -> np.ndarray:
    grid = np.empty((h, w), dtype=int)
    axis_len = h if horizontal else w
    n_runs = RNG.randint(2, min(4, axis_len))
    run_lengths = _random_partition(axis_len, n_runs)
    run_colors = _nonrepeating_colors(n_runs, colors)

    start = 0
    for size, color in zip(run_lengths, run_colors):
        end = start + size
        if horizontal:
            grid[start:end, :] = color
        else:
            grid[:, start:end] = color
        start = end
    return grid


def _source_blocks(h: int, w: int, colors: list[int]) -> np.ndarray:
    grid = np.empty((h, w), dtype=int)
    row_parts = RNG.randint(2, min(4, h))
    col_parts = RNG.randint(2, min(4, w))
    row_sizes = _random_partition(h, row_parts)
    col_sizes = _random_partition(w, col_parts)

    rs = 0
    for i, rh in enumerate(row_sizes):
        cs = 0
        for j, cw in enumerate(col_sizes):
            forbidden = set()
            if i > 0:
                forbidden.add(int(grid[rs - 1, cs]))
            if j > 0:
                forbidden.add(int(grid[rs, cs - 1]))
            choices = [c for c in colors if c not in forbidden] or colors
            grid[rs : rs + rh, cs : cs + cw] = int(RNG.choice(choices))
            cs += cw
        rs += rh
    return grid


def _source_voronoi(h: int, w: int, colors: list[int]) -> np.ndarray:
    grid = np.empty((h, w), dtype=int)
    n_seeds = RNG.randint(2, min(5, h * w))
    picks = RNG.sample(range(h * w), n_seeds)
    seed_colors = [int(RNG.choice(colors)) for _ in range(n_seeds)]
    if len(set(seed_colors)) == 1:
        seed_colors[-1] = int(RNG.choice([c for c in colors if c != seed_colors[0]]))

    seeds = [(p // w, p % w) for p in picks]
    for r in range(h):
        for c in range(w):
            best_idx = 0
            best_key = None
            for idx, (sr, sc) in enumerate(seeds):
                key = (abs(r - sr) + abs(c - sc), idx)
                if best_key is None or key < best_key:
                    best_key = key
                    best_idx = idx
            grid[r, c] = seed_colors[best_idx]
    return grid


def _source_checker(h: int, w: int, colors: list[int]) -> np.ndarray:
    grid = np.empty((h, w), dtype=int)
    row_parts = RNG.randint(2, min(4, h))
    col_parts = RNG.randint(2, min(4, w))
    row_sizes = _random_partition(h, row_parts)
    col_sizes = _random_partition(w, col_parts)
    palette = list(colors)
    RNG.shuffle(palette)
    offset = RNG.randrange(len(palette))

    rs = 0
    for i, rh in enumerate(row_sizes):
        cs = 0
        for j, cw in enumerate(col_sizes):
            color = palette[(i + j + offset) % len(palette)]
            grid[rs : rs + rh, cs : cs + cw] = int(color)
            cs += cw
        rs += rh
    return grid


def _make_source_pattern(h: int, w: int, colors: list[int]) -> np.ndarray:
    mode = RNG.choice(("h_stripes", "v_stripes", "blocks", "voronoi", "checker"))
    if mode == "h_stripes":
        grid = _source_stripes(h, w, colors, horizontal=True)
    elif mode == "v_stripes":
        grid = _source_stripes(h, w, colors, horizontal=False)
    elif mode == "blocks":
        grid = _source_blocks(h, w, colors)
    elif mode == "voronoi":
        grid = _source_voronoi(h, w, colors)
    else:
        grid = _source_checker(h, w, colors)

    if len(np.unique(grid)) < 2:
        grid[0, 0] = colors[0]
        grid[-1, -1] = colors[1]
    return grid


def _hole_mask_rectangles(h: int, w: int) -> np.ndarray:
    holes = np.zeros((h, w), dtype=bool)
    for _ in range(RNG.randint(1, 3)):
        hh = RNG.randint(1, h - 2)
        ww = RNG.randint(1, w - 2)
        r0 = RNG.randint(1, h - hh - 1)
        c0 = RNG.randint(1, w - ww - 1)
        holes[r0 : r0 + hh, c0 : c0 + ww] = True
    return holes


def _hole_mask_walk(h: int, w: int, target: int) -> np.ndarray:
    holes = np.zeros((h, w), dtype=bool)
    cell = (RNG.randint(1, h - 2), RNG.randint(1, w - 2))
    carved = {cell}
    while len(carved) < target:
        base_r, base_c = RNG.choice(tuple(carved))
        neighbors = []
        for dr, dc in NEIGHBORS4:
            nr, nc = base_r + dr, base_c + dc
            if 1 <= nr < h - 1 and 1 <= nc < w - 1:
                neighbors.append((nr, nc))
        pick = RNG.choice(neighbors)
        carved.add(pick)
    for r, c in carved:
        holes[r, c] = True
    return holes


def _hole_mask_bands(h: int, w: int) -> np.ndarray:
    holes = np.zeros((h, w), dtype=bool)
    for _ in range(RNG.randint(1, min(3, h - 2))):
        r = RNG.randint(1, h - 2)
        c0 = RNG.randint(1, w - 2)
        c1 = RNG.randint(c0, w - 2)
        holes[r, c0 : c1 + 1] = True
    for _ in range(RNG.randint(1, min(3, w - 2))):
        c = RNG.randint(1, w - 2)
        r0 = RNG.randint(1, h - 2)
        r1 = RNG.randint(r0, h - 2)
        holes[r0 : r1 + 1, c] = True
    return holes


def _hole_mask_windows(h: int, w: int) -> np.ndarray:
    holes = np.zeros((h, w), dtype=bool)
    inner_h, inner_w = h - 2, w - 2
    row_parts = RNG.randint(2, min(4, inner_h))
    col_parts = RNG.randint(2, min(4, inner_w))
    row_sizes = _random_partition(inner_h, row_parts)
    col_sizes = _random_partition(inner_w, col_parts)
    parity = RNG.randrange(2)

    rs = 1
    for i, rh in enumerate(row_sizes):
        cs = 1
        for j, cw in enumerate(col_sizes):
            if (i + j + parity) % 2 == 0:
                holes[rs : rs + rh, cs : cs + cw] = True
            cs += cw
        rs += rh
    return holes


def _make_hole_mask(h: int, w: int) -> np.ndarray:
    inner_area = (h - 2) * (w - 2)
    min_holes = max(1, inner_area // 4)
    max_holes = max(min_holes, inner_area - max(1, inner_area // 4))

    for _ in range(40):
        mode = RNG.choice(("rectangles", "walk", "bands", "windows"))
        if mode == "rectangles":
            holes = _hole_mask_rectangles(h, w)
        elif mode == "walk":
            target = RNG.randint(min_holes, max_holes)
            holes = _hole_mask_walk(h, w, target)
        elif mode == "bands":
            holes = _hole_mask_bands(h, w)
        else:
            holes = _hole_mask_windows(h, w)

        hole_count = int(holes[1:-1, 1:-1].sum())
        if min_holes <= hole_count < inner_area:
            return holes
    raise RuntimeError("failed to make a valid hole mask")


def _build_template(h: int, w: int, color: int) -> np.ndarray:
    template = np.full((h, w), color, dtype=int)
    holes = _make_hole_mask(h, w)
    template[holes] = BG
    return template


def _is_valid_grid(grid: np.ndarray, expected: np.ndarray) -> bool:
    try:
        sep_r = _find_separator_row(grid)
        upper = _crop_non_black(grid[:sep_r, :])
        lower = _crop_non_black(grid[sep_r + 1 :, :])
        if upper.shape != lower.shape:
            return False
        if not np.all(upper != BG):
            return False
        if not (np.all(lower[0] != BG) and np.all(lower[-1] != BG)):
            return False
        if not (np.all(lower[:, 0] != BG) and np.all(lower[:, -1] != BG)):
            return False
        actual = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False
    visible = actual[actual != BG]
    return visible.size > 0 and len(np.unique(visible)) >= 2


def _gen_attempt() -> np.ndarray | None:
    h = RNG.randint(*RECT_H_RANGE)
    w = RNG.randint(*RECT_W_RANGE)

    top_pad = RNG.randint(*TOP_PAD_RANGE)
    bottom_pad = RNG.randint(*BOTTOM_PAD_RANGE)
    gap_up = RNG.randint(*ABOVE_SEPARATOR_GAP)
    gap_down = RNG.randint(*BELOW_SEPARATOR_GAP)
    grid_h = top_pad + h + gap_up + 1 + gap_down + h + bottom_pad
    grid_w = w + RNG.randint(*WIDTH_SLACK_RANGE)

    src_r0 = top_pad
    sep_r = src_r0 + h + gap_up
    tmpl_r0 = sep_r + 1 + gap_down

    src_c0 = RNG.randint(1, grid_w - w - 1)
    tmpl_c0 = RNG.randint(1, grid_w - w - 1)

    source_colors = RNG.sample(PALETTE, RNG.randint(*SOURCE_COLOR_COUNT))
    source = _make_source_pattern(h, w, source_colors)
    template_color = int(RNG.choice([c for c in PALETTE if c not in source_colors]))
    template = _build_template(h, w, template_color)
    expected = np.where(template == BG, source, BG).astype(int)

    grid = np.zeros((grid_h, grid_w), dtype=int)
    grid[sep_r, :] = SEPARATOR
    grid[src_r0 : src_r0 + h, src_c0 : src_c0 + w] = source
    grid[tmpl_r0 : tmpl_r0 + h, tmpl_c0 : tmpl_c0 + w] = template

    if not _is_valid_grid(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
