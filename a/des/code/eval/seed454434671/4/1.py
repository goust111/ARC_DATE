from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
WALL = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
LEGEND_WIDTH = 3
PATTERN_SIZE = 3
MAX_GENERATION_ATTEMPTS = 400

PATTERN_MASKS = [
    np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=bool),
    np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
]

NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def _dominant_color(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    ranked = counts.most_common()
    assert ranked, "expected a nonempty grid"
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "expected a unique dominant color"
    return int(ranked[0][0])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    assert rows.size, "expected a nonempty mask"
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _has_hole(mask: np.ndarray) -> bool:
    bg = ~mask
    if not np.any(bg):
        return False

    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    stack: list[tuple[int, int]] = []
    for r in range(h):
        for c in (0, w - 1):
            if bg[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if bg[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))

    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return bool(np.any(bg & ~seen))


def _region_patch(mask: np.ndarray, seed_color: int) -> tuple[np.ndarray, np.ndarray]:
    tight = _tight_mask(mask.astype(bool))
    h, w = tight.shape
    interior = np.zeros((h + 2, w + 2), dtype=bool)
    interior[1:-1, 1:-1] = tight

    wall_mask = np.zeros_like(interior, dtype=bool)
    rows, cols = np.where(interior)
    for r, c in zip(rows.tolist(), cols.tolist(), strict=False):
        for dr, dc in NEIGHBORS8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < interior.shape[0] and 0 <= nc < interior.shape[1] and not interior[nr, nc]:
                wall_mask[nr, nc] = True

    patch = np.full(interior.shape, BG, dtype=int)
    patch[wall_mask] = WALL
    cells = list(zip(rows.tolist(), cols.tolist(), strict=False))
    sr, sc = random.choice(cells)
    patch[sr, sc] = seed_color
    return patch, interior


def _legend_left(g: np.ndarray, bg: int) -> int:
    w = g.shape[1]
    c = w - 1
    while c >= 0 and np.any(g[:, c] != bg):
        c -= 1
    left = c + 1
    assert 0 < left < w, "expected a right-edge legend separated by at least one blank column"
    assert w - left == LEGEND_WIDTH, "legend should occupy exactly three columns"
    return left


def _parse_legend(g: np.ndarray, bg: int) -> tuple[dict[int, np.ndarray], int]:
    left = _legend_left(g, bg)
    legend = g[:, left:]
    h = legend.shape[0]
    used = np.zeros_like(legend, dtype=bool)
    mapping: dict[int, np.ndarray] = {}

    r = 0
    while r <= h - 4:
        nz = np.where(legend[r] != bg)[0]
        if nz.size == 1:
            block = legend[r + 1 : r + 4]
            if block.shape == (PATTERN_SIZE, LEGEND_WIDTH) and np.all(block != bg):
                key = int(legend[r, int(nz[0])])
                colors = {int(v) for v in np.unique(block)}
                if len(colors) == 2 and key in colors:
                    assert key not in mapping, "seed colors must have unique legend entries"
                    mapping[key] = block.copy()
                    used[r, int(nz[0])] = True
                    used[r + 1 : r + 4, :] = True
                    r += 4
                    continue
        r += 1

    assert mapping, "expected at least one legend entry"
    assert np.all((legend == bg) | used), "legend strip contains unparsable foreground cells"
    return mapping, left


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed region with the 3x3 legend pattern keyed by its seed color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _dominant_color(g)
    non_bg = g[g != bg]
    assert non_bg.size, "expected foreground walls and legend"
    wall = _dominant_color(non_bg)

    mapping, legend_left = _parse_legend(g, bg)
    core = g[:, :legend_left]
    out = g.copy()

    passable = core != wall
    for comp in _components(passable):
        if any(r == 0 or c == 0 or r == core.shape[0] - 1 or c == core.shape[1] - 1 for r, c in comp):
            continue

        seed_cells = [(r, c) for r, c in comp if int(core[r, c]) != bg]
        assert len(seed_cells) == 1, "each enclosed region must contain exactly one seed"
        seed_color = int(core[seed_cells[0]])
        assert seed_color in mapping, "every seed color must appear in the legend"

        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, c0 = min(rows), min(cols)
        pattern = mapping[seed_color]
        for r, c in comp:
            out[r, c] = int(pattern[(r - r0) % PATTERN_SIZE, (c - c0) % PATTERN_SIZE])

    return out


def _random_region_mask() -> np.ndarray | None:
    h = random.randint(3, 6)
    w = random.randint(3, 6)
    max_area = h * w - 1
    min_area = max(5, (h * w) // 2)
    if min_area > max_area:
        return None
    target = random.randint(min_area, max_area)

    for _ in range(80):
        mask = np.zeros((h, w), dtype=bool)
        seed = (random.randrange(h), random.randrange(w))
        mask[seed] = True
        while int(mask.sum()) < target:
            frontier = []
            for r in range(h):
                for c in range(w):
                    if mask[r, c]:
                        continue
                    touches = sum(
                        0 <= r + dr < h and 0 <= c + dc < w and mask[r + dr, c + dc]
                        for dr, dc in NEIGHBORS4
                    )
                    if touches:
                        frontier.append((r, c))
            if not frontier:
                break
            mask[random.choice(frontier)] = True

        if int(mask.sum()) != target:
            continue

        if random.random() < 0.5:
            mask = np.fliplr(mask)
        if random.random() < 0.5:
            mask = np.flipud(mask)
        mask = np.rot90(mask, random.randrange(4))
        mask = _tight_mask(mask)

        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if np.all(mask):
            continue
        if _has_hole(mask):
            continue
        return mask
    return None


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _render_expected(inp: np.ndarray, placements: list[dict]) -> np.ndarray:
    out = inp.copy()
    for item in placements:
        r0, c0 = item["origin"]
        interior = item["interior"]
        pattern = item["pattern"]
        rows, cols = np.where(interior)
        for r, c in zip(rows.tolist(), cols.tolist(), strict=False):
            if r == 0 or c == 0 or r == interior.shape[0] - 1 or c == interior.shape[1] - 1:
                continue
            out[r0 + r, c0 + c] = int(pattern[(r - 1) % PATTERN_SIZE, (c - 1) % PATTERN_SIZE])
    return out


def _gen_attempt() -> np.ndarray | None:
    region_count = random.randint(1, 4)
    colors = list(COLOR_POOL)
    random.shuffle(colors)
    seed_colors = colors[:region_count]
    secondary_colors = colors[region_count : 2 * region_count]
    pattern_masks = random.sample(PATTERN_MASKS, region_count)

    gaps = [random.randint(0, 2) for _ in range(max(0, region_count - 1))]
    top_margin = random.randint(0, 2)
    bottom_margin = random.randint(0, 2)
    legend_height = top_margin + bottom_margin + 4 * region_count + sum(gaps)

    height = random.randint(max(18, legend_height), max(24, legend_height + 4))
    play_w = random.randint(13, 19)
    sep_w = random.randint(1, 2)
    width = play_w + sep_w + LEGEND_WIDTH
    legend_left = width - LEGEND_WIDTH

    inp = np.full((height, width), BG, dtype=int)
    placements: list[dict] = []
    reserved: list[tuple[int, int, int, int]] = []

    specs = list(zip(seed_colors, secondary_colors, pattern_masks, strict=False))
    random.shuffle(specs)
    for seed_color, secondary_color, mask_bits in specs:
        region_mask = _random_region_mask()
        if region_mask is None:
            return None
        patch, interior = _region_patch(region_mask, seed_color)
        ph, pw = patch.shape
        if ph + 2 >= height or pw + 2 >= play_w:
            return None

        placed = False
        for _ in range(120):
            r0 = random.randint(1, height - ph - 1)
            c0 = random.randint(1, play_w - pw - 1)
            box = (r0 - 1, c0 - 1, r0 + ph + 1, c0 + pw + 1)
            if any(_boxes_overlap(box, other) for other in reserved):
                continue
            patch_target = inp[r0 : r0 + ph, c0 : c0 + pw]
            if np.any(patch_target != BG):
                continue
            patch_target[patch != BG] = patch[patch != BG]
            pattern = np.where(mask_bits, seed_color, secondary_color).astype(int)
            placements.append(
                {
                    "origin": (r0, c0),
                    "interior": interior,
                    "pattern": pattern,
                    "seed_color": seed_color,
                }
            )
            reserved.append(box)
            placed = True
            break
        if not placed:
            return None

    order = placements[:]
    random.shuffle(order)
    row = top_margin
    for idx, item in enumerate(order):
        seed_color = int(item["seed_color"])
        pattern = item["pattern"]
        marker_col = random.randrange(LEGEND_WIDTH)
        inp[row, legend_left + marker_col] = seed_color
        inp[row + 1 : row + 4, legend_left:] = pattern
        row += 4
        if idx < len(gaps):
            row += gaps[idx]

    expected = _render_expected(inp, placements)

    counts = Counter(map(int, inp.reshape(-1)))
    ranked = counts.most_common()
    if ranked[0][0] != BG or (len(ranked) > 1 and ranked[0][1] == ranked[1][1]):
        return None

    non_bg_counts = Counter(map(int, inp[inp != BG].reshape(-1)))
    non_bg_ranked = non_bg_counts.most_common()
    if non_bg_ranked[0][0] != WALL or (len(non_bg_ranked) > 1 and non_bg_ranked[0][1] == non_bg_ranked[1][1]):
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    """
    Randomly generate a valid walled-region + right-edge legend puzzle instance.
    """
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
