from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
DIVIDER_COLOR = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
QUADRANTS = ("TL", "TR", "BL", "BR")
MAX_GENERATION_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_dividers(grid: np.ndarray, bg: int) -> tuple[int, int, int]:
    h, w = grid.shape
    full_rows_by_color: dict[int, list[int]] = {}
    full_cols_by_color: dict[int, list[int]] = {}

    for r in range(h):
        color = int(grid[r, 0])
        if color != bg and bool(np.all(grid[r, :] == color)):
            full_rows_by_color.setdefault(color, []).append(r)
    for c in range(w):
        color = int(grid[0, c])
        if color != bg and bool(np.all(grid[:, c] == color)):
            full_cols_by_color.setdefault(color, []).append(c)

    candidates = [
        (color, rows, cols)
        for color, rows in full_rows_by_color.items()
        if color in full_cols_by_color
        for cols in [full_cols_by_color[color]]
        if len(rows) == 1 and len(cols) == 1
    ]
    assert len(candidates) == 1, "expected exactly one full horizontal and vertical divider pair"
    color, rows, cols = candidates[0]
    return int(color), int(rows[0]), int(cols[0])


def _quadrant_bounds(h: int, w: int, div_r: int, div_c: int) -> dict[str, tuple[int, int, int, int]]:
    bounds = {
        "TL": (0, div_r - 1, 0, div_c - 1),
        "TR": (0, div_r - 1, div_c + 1, w - 1),
        "BL": (div_r + 1, h - 1, 0, div_c - 1),
        "BR": (div_r + 1, h - 1, div_c + 1, w - 1),
    }
    for r0, r1, c0, c1 in bounds.values():
        assert r0 <= r1 and c0 <= c1, "divider must create four non-empty quadrants"
    return bounds


def _quadrant_of(r: int, c: int, div_r: int, div_c: int) -> str:
    if r < div_r and c < div_c:
        return "TL"
    if r < div_r and c > div_c:
        return "TR"
    if r > div_r and c < div_c:
        return "BL"
    if r > div_r and c > div_c:
        return "BR"
    raise AssertionError("point lies on a divider")


def _flood_same_color(
    grid: np.ndarray,
    start: tuple[int, int],
    bounds: tuple[int, int, int, int],
) -> np.ndarray:
    r0, r1, c0, c1 = bounds
    sr, sc = start
    color = int(grid[sr, sc])
    stack = [(int(sr), int(sc))]
    seen = {(int(sr), int(sc))}
    coords: list[tuple[int, int]] = []

    while stack:
        r, c = stack.pop()
        coords.append((r, c))
        for dr, dc in DIRS:
            nr = r + dr
            nc = c + dc
            if not (r0 <= nr <= r1 and c0 <= nc <= c1):
                continue
            if (nr, nc) in seen:
                continue
            if int(grid[nr, nc]) != color:
                continue
            seen.add((nr, nc))
            stack.append((nr, nc))

    return np.array(coords, dtype=int)


def _extract_rulers(
    grid: np.ndarray,
) -> tuple[int, int, int, np.ndarray, dict[int, int], dict[int, str]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _most_common_color(g)
    divider_color, div_r, div_c = _find_dividers(g, bg)
    bounds = _quadrant_bounds(h, w, div_r, div_c)
    ruler_mask = np.zeros_like(g, dtype=bool)
    size_by_color: dict[int, int] = {}
    home_by_color: dict[int, str] = {}

    corners = {
        "TL": (0, 0),
        "TR": (0, w - 1),
        "BL": (h - 1, 0),
        "BR": (h - 1, w - 1),
    }

    for name in QUADRANTS:
        r0, r1, c0, c1 = bounds[name]
        sr, sc = corners[name]
        color = int(g[sr, sc])
        assert color not in (bg, divider_color), "each quadrant corner must start with a colored ruler block"

        coords = _flood_same_color(g, (sr, sc), bounds[name])
        rr0 = int(coords[:, 0].min())
        rr1 = int(coords[:, 0].max())
        cc0 = int(coords[:, 1].min())
        cc1 = int(coords[:, 1].max())
        side_h = rr1 - rr0 + 1
        side_w = cc1 - cc0 + 1
        assert side_h == side_w and 1 <= side_h <= 3, "ruler block must be a 1x1 to 3x3 square"
        assert int(coords.shape[0]) == side_h * side_w, "ruler block must be solid"

        if name == "TL":
            assert rr0 == r0 and cc0 == c0
        elif name == "TR":
            assert rr0 == r0 and cc1 == c1
        elif name == "BL":
            assert rr1 == r1 and cc0 == c0
        else:
            assert rr1 == r1 and cc1 == c1

        ruler_mask[coords[:, 0], coords[:, 1]] = True
        assert color not in size_by_color, "ruler colors must be unique"
        size_by_color[color] = int(side_h)
        home_by_color[color] = name

    return int(bg), int(divider_color), int(div_r), int(div_c), ruler_mask, size_by_color, home_by_color


def _simulate_growth(
    grid: np.ndarray,
    bg: int,
    divider_color: int,
    ruler_mask: np.ndarray,
    size_by_color: dict[int, int],
    collect_stats: bool = False,
) -> tuple[np.ndarray, int, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    seed_coords = np.argwhere((g != bg) & (g != divider_color) & (~ruler_mask))
    assert int(seed_coords.shape[0]) > 0, "expected at least one seed pixel"

    full_dirs = 0
    blocked_dirs = 0
    for r, c in seed_coords:
        color = int(g[r, c])
        assert color in size_by_color, "seed color must match a ruler block"
        ray_len = int(size_by_color[color])

        for dr, dc in DIRS:
            rr = int(r)
            cc = int(c)
            steps = 0
            for _ in range(ray_len):
                rr += dr
                cc += dc
                if not (0 <= rr < h and 0 <= cc < w):
                    break
                if bool(ruler_mask[rr, cc]) or int(out[rr, cc]) == divider_color:
                    break
                cell = int(out[rr, cc])
                if cell not in (bg, color):
                    break
                out[rr, cc] = color
                steps += 1

            if collect_stats:
                if steps == ray_len:
                    full_dirs += 1
                else:
                    blocked_dirs += 1

    return out, int(full_dirs), int(blocked_dirs)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each colored seed into four orthogonal rays whose length is read from the matching corner ruler block.

    Input structure:
    - The background is black, and one complete grey row together with one complete grey column divide the grid into four quadrants.
    - Each quadrant has a monochrome 1x1 to 3x3 square anchored in its outer corner; the four ruler-block colors are all distinct.
    - Additional single colored seed pixels are scattered through the quadrants, and every seed color matches one ruler-block color.

    Rule:
    - Keep the divider lines and the four corner ruler blocks unchanged.
    - For each seed pixel of color C, find the ruler block of the same color and read its side length L.
    - From the seed, extend color-C rays upward, downward, leftward, and rightward for at most L cells, not counting the seed cell itself.
    - A ray may continue through background or already-painted cells of the same color, but it stops before the grid edge or before any divider cell, ruler-block cell, or different-color pixel.

    Key insight:
    - The corner squares are not decorations: each one is a color-keyed ruler whose side length controls how far same-colored seeds grow.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg, divider_color, _div_r, _div_c, ruler_mask, size_by_color, _home_by_color = _extract_rulers(g)
    out, _full_dirs, _blocked_dirs = _simulate_growth(g, bg, divider_color, ruler_mask, size_by_color)
    return out


def _dilate_chebyshev(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rs0 = max(0, -dr)
            rs1 = min(h, h - dr)
            cs0 = max(0, -dc)
            cs1 = min(w, w - dc)
            rd0 = max(0, dr)
            rd1 = min(h, h + dr)
            cd0 = max(0, dc)
            cd1 = min(w, w + dc)
            out[rd0:rd1, cd0:cd1] |= mask[rs0:rs1, cs0:cs1]
    return out


def _ray_cells_static(
    shape: tuple[int, int],
    hard_block: np.ndarray,
    seeds: dict[tuple[int, int], int],
    start: tuple[int, int],
    color: int,
    ray_len: int,
    dr: int,
    dc: int,
) -> list[tuple[int, int]]:
    h, w = shape
    rr, cc = start
    cells: list[tuple[int, int]] = []
    for _ in range(ray_len):
        rr += dr
        cc += dc
        if not (0 <= rr < h and 0 <= cc < w):
            break
        if bool(hard_block[rr, cc]):
            break
        other = seeds.get((rr, cc))
        if other is not None and int(other) != int(color):
            break
        cells.append((int(rr), int(cc)))
    return cells


def _candidate_cells(
    base: np.ndarray,
    hard_block: np.ndarray,
    ruler_buffer: np.ndarray,
    seeds: dict[tuple[int, int], int],
    div_r: int,
    div_c: int,
    color: int,
    ray_len: int,
    category: str,
    home: str,
    require_off_home: bool,
) -> list[tuple[int, int]]:
    h, w = base.shape
    cells: list[tuple[int, int]] = []

    for r in range(h):
        for c in range(w):
            if int(base[r, c]) != BACKGROUND:
                continue
            if bool(ruler_buffer[r, c]):
                continue
            if any(max(abs(r - sr), abs(c - sc)) <= 1 for sr, sc in seeds):
                continue

            quadrant = _quadrant_of(r, c, div_r, div_c)
            if require_off_home and quadrant == home:
                continue

            paths = [
                _ray_cells_static(base.shape, hard_block, seeds, (r, c), color, ray_len, dr, dc)
                for dr, dc in DIRS
            ]
            if category == "open" and not all(len(path) == ray_len for path in paths):
                continue
            if category == "blocked" and not any(len(path) < ray_len for path in paths):
                continue
            cells.append((int(r), int(c)))

    random.shuffle(cells)
    return cells


def _claims_are_unambiguous(
    shape: tuple[int, int],
    hard_block: np.ndarray,
    seeds: dict[tuple[int, int], int],
    size_by_color: dict[int, int],
) -> bool:
    claims: dict[tuple[int, int], set[int]] = {}
    for start, color in seeds.items():
        ray_len = int(size_by_color[color])
        for dr, dc in DIRS:
            for cell in _ray_cells_static(shape, hard_block, seeds, start, color, ray_len, dr, dc):
                claims.setdefault(cell, set()).add(int(color))
                if len(claims[cell]) > 1:
                    return False
    return True


def _paint_seeds(base: np.ndarray, seeds: dict[tuple[int, int], int]) -> np.ndarray:
    grid = base.copy()
    for (r, c), color in seeds.items():
        grid[r, c] = int(color)
    return grid


def _build_base_grid(
    h: int,
    w: int,
    div_r: int,
    div_c: int,
    colors: list[int],
    sizes: list[int],
) -> tuple[np.ndarray, np.ndarray, dict[int, int], dict[int, str]]:
    grid = np.full((h, w), BACKGROUND, dtype=int)
    grid[div_r, :] = DIVIDER_COLOR
    grid[:, div_c] = DIVIDER_COLOR

    ruler_mask = np.zeros((h, w), dtype=bool)
    size_by_color: dict[int, int] = {}
    home_by_color: dict[int, str] = {}

    placements = {
        "TL": (0, 0),
        "TR": (0, w),
        "BL": (h, 0),
        "BR": (h, w),
    }

    for name, color, side in zip(QUADRANTS, colors, sizes):
        if name == "TL":
            rows = slice(0, side)
            cols = slice(0, side)
        elif name == "TR":
            rows = slice(0, side)
            cols = slice(w - side, w)
        elif name == "BL":
            rows = slice(h - side, h)
            cols = slice(0, side)
        else:
            rows = slice(h - side, h)
            cols = slice(w - side, w)

        grid[rows, cols] = int(color)
        ruler_mask[rows, cols] = True
        size_by_color[int(color)] = int(side)
        home_by_color[int(color)] = name

    return grid, ruler_mask, size_by_color, home_by_color


def _sample_sizes() -> list[int]:
    for _ in range(100):
        sizes = [random.randint(1, 3) for _ in QUADRANTS]
        if len(set(sizes)) >= 2:
            return sizes
    raise RuntimeError("failed to sample non-degenerate ruler sizes")


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(19, 30)
    w = random.randint(19, 30)
    div_r = random.randint(8, h - 9)
    div_c = random.randint(8, w - 9)
    colors = random.sample(PALETTE, 4)
    sizes = _sample_sizes()

    base, ruler_mask, size_by_color, home_by_color = _build_base_grid(h, w, div_r, div_c, colors, sizes)
    ruler_buffer = _dilate_chebyshev(ruler_mask)
    hard_block = ruler_mask.copy()
    hard_block[div_r, :] = True
    hard_block[:, div_c] = True

    counts = {color: random.randint(1, 3) for color in colors}
    open_color, blocked_color, offhome_color = random.sample(colors, 3)
    specs: list[tuple[int, str, bool]] = [
        (open_color, "open", False),
        (blocked_color, "blocked", False),
        (offhome_color, "any", True),
    ]

    used_special = {open_color: 1, blocked_color: 1, offhome_color: 1}
    for color in colors:
        special_used = used_special.get(color, 0)
        for _ in range(counts[color] - special_used):
            specs.append((color, random.choice(("any", "blocked")), False))

    random.shuffle(specs[3:])
    seeds: dict[tuple[int, int], int] = {}

    for color, category, require_off_home in specs:
        candidates = _candidate_cells(
            base,
            hard_block,
            ruler_buffer,
            seeds,
            div_r,
            div_c,
            color,
            int(size_by_color[color]),
            category,
            home_by_color[color],
            require_off_home,
        )
        if not candidates:
            return None
        choice = random.choice(candidates)
        seeds[choice] = int(color)

    if not any(_quadrant_of(r, c, div_r, div_c) != home_by_color[color] for (r, c), color in seeds.items()):
        return None
    if not _claims_are_unambiguous(base.shape, hard_block, seeds, size_by_color):
        return None

    inp = _paint_seeds(base, seeds)
    try:
        actual = main(inp)
    except Exception:
        return None

    expected, full_dirs, blocked_dirs = _simulate_growth(inp, BACKGROUND, DIVIDER_COLOR, ruler_mask, size_by_color, collect_stats=True)
    if not np.array_equal(actual, expected):
        return None
    if full_dirs == 0 or blocked_dirs == 0:
        return None
    if np.array_equal(actual, inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
