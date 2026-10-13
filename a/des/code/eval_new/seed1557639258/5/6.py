import random
from collections import Counter, deque

import numpy as np


NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
BG_COLOR = 0
STAMP_SIZE_RANGE = (3, 5)
BAR_LENGTH_RANGE = (3, 6)
GRID_SIDE_LIMIT = 30
MAX_GENERATION_ATTEMPTS = 300
MASK_RETRIES = 200
PALETTE = tuple(range(1, 10))


def get_bg(grid: np.ndarray) -> int:
    counts = Counter(int(x) for x in np.asarray(grid).ravel())
    return counts.most_common(1)[0][0]


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def find_monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            comps.append(
                {
                    "color": color,
                    "area": len(coords),
                    "coords": np.array(coords, dtype=int),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor one stamp template into a horizontal sequence controlled by a lower instruction bar.

    Input structure:
    - The background is the most common color.
    - The upper part contains exactly one multi-pixel monochrome stamp template on the background.
    - The remaining non-background pixels form one contiguous horizontal instruction bar, and the bar's first color matches the stamp color.

    Rule:
    - Find the unique multi-pixel monochrome component and crop it to its bounding box to get the stamp mask.
    - Read the instruction-bar colors from left to right.
    - Create one copy of the cropped stamp per instruction color, recoloring all stamp pixels to that color and separating adjacent copies by one background column.

    Key insight:
    - The repeated first color links the bar to the stamp, while the bar length and color order jointly encode how many recolored copies to emit and in what sequence.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = get_bg(g)
    comps = find_monochrome_components(g, bg)
    stamp_candidates = [comp for comp in comps if comp["area"] > 1]
    assert len(stamp_candidates) == 1, "expected exactly one multi-pixel stamp component"

    stamp = stamp_candidates[0]
    stamp_color = int(stamp["color"])
    r0, c0, r1, c1 = stamp["bbox"]
    patch = g[r0 : r1 + 1, c0 : c1 + 1]
    stamp_mask = patch == stamp_color
    assert np.any(stamp_mask), "stamp mask cannot be empty"

    remaining = (g != bg).copy()
    stamp_coords = stamp["coords"]
    remaining[stamp_coords[:, 0], stamp_coords[:, 1]] = False
    bar_coords = np.argwhere(remaining)
    assert len(bar_coords) >= 3, "instruction bar must contain at least three pixels"

    bar_rows = sorted({int(r) for r, _ in bar_coords})
    assert len(bar_rows) == 1, "instruction bar must lie on one row"
    bar_row = bar_rows[0]

    bar_cols = sorted(int(c) for _, c in bar_coords)
    assert bar_cols == list(range(bar_cols[0], bar_cols[-1] + 1)), "instruction bar must be contiguous"
    assert len(bar_cols) <= 6, "instruction bar is unexpectedly long"

    bar_colors = [int(g[bar_row, c]) for c in bar_cols]
    assert len(set(bar_colors)) == len(bar_colors), "instruction colors must be distinct"
    assert bar_colors[0] == stamp_color, "first instruction color must match the stamp color"

    out_h, out_w = stamp_mask.shape[0], len(bar_colors) * stamp_mask.shape[1] + len(bar_colors) - 1
    out = np.full((out_h, out_w), bg, dtype=int)
    tile_w = stamp_mask.shape[1]
    for idx, color in enumerate(bar_colors):
        c_start = idx * (tile_w + 1)
        block = out[:, c_start : c_start + tile_w]
        block[stamp_mask] = color
    return out


def _random_path(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = start
    rr, cc = end
    steps = ["r"] * abs(rr - r) + ["c"] * abs(cc - c)
    random.shuffle(steps)
    dr = 0 if rr == r else (1 if rr > r else -1)
    dc = 0 if cc == c else (1 if cc > c else -1)
    cells = []
    for axis in steps:
        if axis == "r":
            r += dr
        else:
            c += dc
        cells.append((r, c))
    return cells


def _has_nontrivial_symmetry(mask: np.ndarray) -> bool:
    variants = []
    for k in range(4):
        rot = np.rot90(mask, k)
        variants.append(rot)
        variants.append(np.fliplr(rot))
    for variant in variants[1:]:
        if variant.shape == mask.shape and np.array_equal(variant, mask):
            return True
    return False


def _frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells = set()
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for dr, dc in NEIGHBORS4:
            nr = rr + dr
            nc = cc + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                cells.add((nr, nc))
    return list(cells)


def _make_stamp_mask() -> np.ndarray:
    for _ in range(MASK_RETRIES):
        h = random.randint(*STAMP_SIZE_RANGE)
        w = random.randint(*STAMP_SIZE_RANGE)
        anchors = [
            (0, random.randrange(w)),
            (h - 1, random.randrange(w)),
            (random.randrange(h), 0),
            (random.randrange(h), w - 1),
        ]
        random.shuffle(anchors)

        mask = np.zeros((h, w), dtype=bool)
        cur = anchors[0]
        mask[cur] = True
        for nxt in anchors[1:]:
            for cell in _random_path(cur, nxt):
                mask[cell] = True
            cur = nxt

        if np.all(mask):
            continue

        min_fill = max(4, int(np.ceil(0.4 * h * w)), int(mask.sum()))
        max_fill = h * w - 1
        if min_fill > max_fill:
            continue
        target = random.randint(min_fill, max_fill)

        while int(mask.sum()) < target:
            choices = _frontier(mask)
            if not choices:
                break
            mask[random.choice(choices)] = True

        if int(mask.sum()) != target:
            continue
        if _has_nontrivial_symmetry(mask):
            continue
        return mask
    raise RuntimeError("failed to create an asymmetric stamp mask")


def _build_expected(mask: np.ndarray, colors: list[int], bg: int) -> np.ndarray:
    h, w = mask.shape
    out = np.full((h, len(colors) * w + len(colors) - 1), bg, dtype=int)
    for idx, color in enumerate(colors):
        c0 = idx * (w + 1)
        block = out[:, c0 : c0 + w]
        block[mask] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    mask = _make_stamp_mask()
    stamp_h, stamp_w = mask.shape

    valid_lengths = [
        n
        for n in range(BAR_LENGTH_RANGE[0], BAR_LENGTH_RANGE[1] + 1)
        if n * stamp_w + (n - 1) <= GRID_SIDE_LIMIT
    ]
    if not valid_lengths:
        return None
    bar_len = random.choice(valid_lengths)

    colors = random.sample(PALETTE, bar_len)
    stamp_color = colors[0]

    min_h = max(2 * stamp_h + 2, 10)
    grid_h = random.randint(min_h, 20)
    min_w = max(stamp_w + 2, bar_len + 2, 10)
    grid_w = random.randint(min_w, GRID_SIDE_LIMIT)

    top_row_max = grid_h // 2 - stamp_h
    if top_row_max < 0:
        return None
    stamp_r0 = random.randint(0, top_row_max)
    stamp_c0 = random.randint(0, grid_w - stamp_w)

    bar_row_min = max(grid_h // 2, stamp_r0 + stamp_h + 1)
    if bar_row_min > grid_h - 1:
        return None
    bar_row = random.randint(bar_row_min, grid_h - 1)
    bar_c0 = random.randint(0, grid_w - bar_len)

    grid = np.full((grid_h, grid_w), BG_COLOR, dtype=int)
    stamp_block = grid[stamp_r0 : stamp_r0 + stamp_h, stamp_c0 : stamp_c0 + stamp_w]
    stamp_block[mask] = stamp_color
    for idx, color in enumerate(colors):
        grid[bar_row, bar_c0 + idx] = color

    expected = _build_expected(mask, colors, BG_COLOR)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
