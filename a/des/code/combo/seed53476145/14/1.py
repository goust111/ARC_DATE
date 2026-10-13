from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
PALETTE = tuple(range(1, 10))
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

# --------------- solver ---------------


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
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
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _is_entry(block: np.ndarray) -> bool:
    if block.shape != (2, 4):
        return False
    left = block[:, :2]
    right = block[:, 2:4]
    seed = int(left[0, 0])
    if seed == BACKGROUND or not np.all(left == seed):
        return False
    colors = set(map(int, right.reshape(-1)))
    return BACKGROUND not in colors and len(colors) == 2


def _parse_vertical_legend(g: np.ndarray) -> tuple[dict[int, np.ndarray], np.ndarray]:
    h, w = g.shape
    if h < 2 or w < 4:
        return {}, np.zeros_like(g, dtype=bool)

    mapping: dict[int, np.ndarray] = {}
    mask = np.zeros_like(g, dtype=bool)
    r = 0
    while r + 1 < h:
        if _is_entry(g[r : r + 2, :4]):
            seed = int(g[r, 0])
            if seed in mapping:
                return {}, np.zeros_like(g, dtype=bool)
            mapping[seed] = g[r : r + 2, 2:4].copy()
            mask[r : r + 2, :4] = True
            r += 2
            while r < h and np.all(g[r, :4] == BACKGROUND):
                r += 1
        else:
            break
    return mapping, mask


def _parse_horizontal_legend(g: np.ndarray) -> tuple[dict[int, np.ndarray], np.ndarray]:
    h, w = g.shape
    if h < 2 or w < 4:
        return {}, np.zeros_like(g, dtype=bool)

    mapping: dict[int, np.ndarray] = {}
    mask = np.zeros_like(g, dtype=bool)
    c = 0
    while c + 3 < w:
        if _is_entry(g[:2, c : c + 4]):
            seed = int(g[0, c])
            if seed in mapping:
                return {}, np.zeros_like(g, dtype=bool)
            mapping[seed] = g[:2, c + 2 : c + 4].copy()
            mask[:2, c : c + 4] = True
            c += 4
            while c < w and np.all(g[:2, c] == BACKGROUND):
                c += 1
        else:
            break
    return mapping, mask


def _extract_legend(g: np.ndarray) -> tuple[dict[int, np.ndarray], np.ndarray]:
    candidates: list[tuple[int, int, dict[int, np.ndarray], np.ndarray]] = []
    for parser in (_parse_vertical_legend, _parse_horizontal_legend):
        mapping, mask = parser(g)
        if mapping:
            candidates.append((len(mapping), int(mask.sum()), mapping, mask))
    assert candidates, "expected a top-left legend built from 2x4 entries"

    best = max(candidates, key=lambda item: (item[0], item[1]))
    mapping, mask = best[2], best[3]
    assert len(mapping) >= 2, "expected several legend entries"
    return mapping, mask


def _tile_pattern(pattern: np.ndarray, height: int, width: int) -> np.ndarray:
    rows = np.arange(height) % 2
    cols = np.arange(width) % 2
    return pattern[rows[:, None], cols[None, :]]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each solid-color rectangle by the 2x2 pattern assigned to its color in the top-left legend.

    The legend is cleared in the output. Rectangles whose color has no legend entry remain unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    mapping, legend_mask = _extract_legend(g)

    work = g.copy()
    work[legend_mask] = BACKGROUND
    out = work.copy()

    for comp in _components(work != BACKGROUND):
        r0, c0, r1, c1 = _bbox(comp)
        color = int(work[comp[0, 0], comp[0, 1]])
        block = work[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(block == color), "every non-legend object must be a solid rectangle"
        assert block.size == len(comp), "every non-legend object must fill its bbox"
        if color in mapping:
            out[r0 : r1 + 1, c0 : c1 + 1] = _tile_pattern(mapping[color], block.shape[0], block.shape[1])

    return out


# --------------- generator ---------------


GRID_HEIGHT_RANGE = (12, 24)
GRID_WIDTH_RANGE = (12, 24)
LEGEND_ENTRY_RANGE = (2, 4)
LEGEND_GAP_CHOICES = (1, 2)
RECT_COUNT_RANGE = (2, 5)
RECT_HEIGHT_RANGE = (2, 7)
RECT_WIDTH_RANGE = (2, 7)
UNMAPPED_BRANCH_PROB = 0.5
PLACEMENT_GAP = 1
PLACEMENT_RETRIES = 250
GENERATION_RETRIES = 400

PATTERN_BITS = tuple(
    np.array(
        [[(mask >> 0) & 1, (mask >> 1) & 1], [(mask >> 2) & 1, (mask >> 3) & 1]],
        dtype=int,
    )
    for mask in range(16)
    if mask not in (0, 15)
)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] <= b[2] and a[2] >= b[0] and a[1] <= b[3] and a[3] >= b[1]


def _expand_box(box: tuple[int, int, int, int], pad: int, h: int, w: int) -> tuple[int, int, int, int]:
    return (
        max(0, box[0] - pad),
        max(0, box[1] - pad),
        min(h - 1, box[2] + pad),
        min(w - 1, box[3] + pad),
    )


def _random_pattern(seed: int, used: set[bytes]) -> np.ndarray | None:
    choices = [color for color in PALETTE if color != seed]
    for _ in range(120):
        c0, c1 = random.sample(choices, 2)
        bits = random.choice(PATTERN_BITS)
        pattern = np.where(bits == 0, c0, c1).astype(int)
        key = pattern.tobytes()
        if key not in used and len(np.unique(pattern)) == 2:
            used.add(key)
            return pattern
    return None


def _legend_spec() -> tuple[str, dict[int, np.ndarray], list[tuple[int, int, np.ndarray]], tuple[int, int, int, int]] | None:
    orientation = random.choice(("vertical", "horizontal"))
    count = random.randint(*LEGEND_ENTRY_RANGE)
    seeds = random.sample(PALETTE, count)

    used_patterns: set[bytes] = set()
    entries: list[tuple[int, int, np.ndarray]] = []
    mapping: dict[int, np.ndarray] = {}

    offset = 0
    for seed in seeds:
        pattern = _random_pattern(seed, used_patterns)
        if pattern is None:
            return None
        entries.append((offset, seed, pattern))
        mapping[seed] = pattern
        offset += 2 if orientation == "vertical" else 4
        if seed != seeds[-1]:
            offset += random.choice(LEGEND_GAP_CHOICES)

    if orientation == "vertical":
        legend_box = (0, 0, offset - 1, 3)
    else:
        legend_box = (0, 0, 1, offset - 1)
    return orientation, mapping, entries, legend_box


def _draw_legend(
    grid: np.ndarray, orientation: str, entries: list[tuple[int, int, np.ndarray]]
) -> None:
    for offset, seed, pattern in entries:
        if orientation == "vertical":
            r0, c0 = offset, 0
        else:
            r0, c0 = 0, offset
        grid[r0 : r0 + 2, c0 : c0 + 2] = seed
        grid[r0 : r0 + 2, c0 + 2 : c0 + 4] = pattern


def _place_rectangles(
    h: int, w: int, legend_box: tuple[int, int, int, int], colors: list[int]
) -> list[tuple[int, int, int, int, int]] | None:
    reserved = [_expand_box(legend_box, PLACEMENT_GAP, h, w)]
    rects: list[tuple[int, int, int, int, int]] = []

    for color in colors:
        placed = False
        max_h = min(RECT_HEIGHT_RANGE[1], h - 1)
        max_w = min(RECT_WIDTH_RANGE[1], w - 1)
        if max_h < RECT_HEIGHT_RANGE[0] or max_w < RECT_WIDTH_RANGE[0]:
            return None
        for _ in range(PLACEMENT_RETRIES):
            rh = random.randint(RECT_HEIGHT_RANGE[0], max_h)
            rw = random.randint(RECT_WIDTH_RANGE[0], max_w)
            r0 = random.randint(0, h - rh)
            c0 = random.randint(0, w - rw)
            box = (r0, c0, r0 + rh - 1, c0 + rw - 1)
            padded = _expand_box(box, PLACEMENT_GAP, h, w)
            if any(_boxes_overlap(padded, other) for other in reserved):
                continue
            reserved.append(padded)
            rects.append((r0, c0, rh, rw, color))
            placed = True
            break
        if not placed:
            return None
    return rects


def _choose_rectangle_colors(seed_colors: list[int]) -> list[int]:
    total = random.randint(*RECT_COUNT_RANGE)
    include_unmapped = random.random() < UNMAPPED_BRANCH_PROB
    if include_unmapped:
        matched = random.randint(1, total - 1)
    else:
        matched = total
    unmatched = total - matched

    colors = [random.choice(seed_colors) for _ in range(matched)]
    if unmatched:
        pool = [color for color in PALETTE if color not in seed_colors]
        colors.extend(random.sample(pool, unmatched))
    random.shuffle(colors)
    return colors


def _render_expected(
    shape: tuple[int, int], rects: list[tuple[int, int, int, int, int]], mapping: dict[int, np.ndarray]
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for r0, c0, rh, rw, color in rects:
        if color in mapping:
            out[r0 : r0 + rh, c0 : c0 + rw] = _tile_pattern(mapping[color], rh, rw)
        else:
            out[r0 : r0 + rh, c0 : c0 + rw] = color
    return out


def _attempt() -> np.ndarray | None:
    legend = _legend_spec()
    if legend is None:
        return None
    orientation, mapping, entries, legend_box = legend

    min_h = max(GRID_HEIGHT_RANGE[0], legend_box[2] + 4)
    min_w = max(GRID_WIDTH_RANGE[0], legend_box[3] + 4)
    if min_h > GRID_HEIGHT_RANGE[1] or min_w > GRID_WIDTH_RANGE[1]:
        return None

    h = random.randint(min_h, GRID_HEIGHT_RANGE[1])
    w = random.randint(min_w, GRID_WIDTH_RANGE[1])

    rect_colors = _choose_rectangle_colors(list(mapping))
    rects = _place_rectangles(h, w, legend_box, rect_colors)
    if rects is None:
        return None

    inp = np.zeros((h, w), dtype=int)
    _draw_legend(inp, orientation, entries)
    for r0, c0, rh, rw, color in rects:
        inp[r0 : r0 + rh, c0 : c0 + rw] = color

    expected = _render_expected((h, w), rects, mapping)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
