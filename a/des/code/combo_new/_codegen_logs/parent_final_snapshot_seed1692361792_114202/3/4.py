import random

import numpy as np


COLOR_VALUES = tuple(range(10))
MAX_GENERATION_ATTEMPTS = 300
QUAD_H_RANGE = (7, 12)
QUAD_W_RANGE = (7, 12)
CORE_AREA_MIN = 9
CORE_AREA_MAX = 18
NOISE_SCORE_CAP = 7
ANCHORS = (
    "top_left",
    "top_right",
    "bottom_left",
    "bottom_right",
    "left_top",
    "left_bottom",
    "right_top",
    "right_bottom",
)


def _most_common_color(values: np.ndarray) -> int:
    vals, counts = np.unique(values, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _scan_palette(colors: list[int], bg: int, sep: int) -> list[int]:
    palette: list[int] = []
    seen: set[int] = set()
    for color in colors:
        if color in (bg, sep):
            if palette:
                break
            return []
        if color in seen:
            break
        seen.add(color)
        palette.append(int(color))
    return palette if len(palette) >= 2 else []


def _extract_palette(grid: np.ndarray, bg: int, sep: int) -> list[int]:
    h, w = grid.shape
    scans = [
        _scan_palette([int(grid[0, c]) for c in range(w)], bg, sep),
        _scan_palette([int(grid[0, c]) for c in range(w - 1, -1, -1)], bg, sep),
        _scan_palette([int(grid[h - 1, c]) for c in range(w)], bg, sep),
        _scan_palette([int(grid[h - 1, c]) for c in range(w - 1, -1, -1)], bg, sep),
        _scan_palette([int(grid[r, 0]) for r in range(h)], bg, sep),
        _scan_palette([int(grid[r, 0]) for r in range(h - 1, -1, -1)], bg, sep),
        _scan_palette([int(grid[r, w - 1]) for r in range(h)], bg, sep),
        _scan_palette([int(grid[r, w - 1]) for r in range(h - 1, -1, -1)], bg, sep),
    ]
    lengths = [len(scan) for scan in scans]
    best = max(lengths)
    assert best >= 2, "palette must be present on an outer edge"
    assert lengths.count(best) == 1, "palette edge must be uniquely identifiable"
    return scans[lengths.index(best)]


def _shape_mask(quadrant: np.ndarray, bg: int, sep: int) -> np.ndarray:
    vals, counts = np.unique(quadrant, return_counts=True)
    ranked = [
        (int(count), int(color))
        for color, count in zip(vals, counts)
        if int(color) not in (bg, sep)
    ]
    assert ranked, "each quadrant must contain a foreground shape"
    _, color = max(ranked)
    mask = quadrant == color
    assert bool(mask.any()), "quadrant shape must be nonempty"
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay the four separator-split quadrant sprites, keep their shared pixels, and recolor the overlap from an edge legend.

    Input structure:
    - A single full row and a single full column of one separator color split the odd-sized grid into four equal quadrants.
    - Each quadrant contains one monochrome sprite on a shared background; the four sprites are aligned copies with small pixel-level disagreements.
    - One outer edge carries a contiguous legend run of distinct non-background colors ordered from index 0 inward.

    Rule:
    - Read the four quadrant sprites as boolean masks in their local quadrant coordinates.
    - Compute the pixelwise intersection of the four masks; this common mask is the output shape.
    - For each quadrant, count how many pixels differ from the common mask, sum the four counts into S, and use the S-th legend color as the output foreground.
    - Return a single-quadrant canvas with the original background color and the common mask painted in the selected legend color.

    Key insight:
    - The cross only aligns four noisy observations of the same sprite; the real content is their overlap, while the total amount of disagreement indexes the border legend to choose the output color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h % 2 == 1 and w % 2 == 1, "quadrant grid must have odd height and width"

    sep_r = h // 2
    sep_c = w // 2
    sep = int(g[sep_r, 0])
    assert np.all(g[sep_r, :] == sep), "separator row must be solid"
    assert np.all(g[:, sep_c] == sep), "separator column must be solid"
    assert int(g[sep_r, sep_c]) == sep

    non_sep = np.ones_like(g, dtype=bool)
    non_sep[sep_r, :] = False
    non_sep[:, sep_c] = False
    bg = _most_common_color(g[non_sep])

    quadrants = [
        g[:sep_r, :sep_c],
        g[:sep_r, sep_c + 1 :],
        g[sep_r + 1 :, :sep_c],
        g[sep_r + 1 :, sep_c + 1 :],
    ]
    masks = [_shape_mask(quadrant, bg, sep) for quadrant in quadrants]
    common = np.logical_and.reduce(masks)
    assert bool(common.any()), "quadrant intersection must be nonempty"

    score = sum(int(np.count_nonzero(mask != common)) for mask in masks)
    palette = _extract_palette(g, bg, sep)
    assert score < len(palette), "difference score must fit the legend"

    out = np.full(common.shape, bg, dtype=int)
    out[common] = int(palette[score])
    return out


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _generate_core_mask(qh: int, qw: int) -> np.ndarray | None:
    inner_h = qh - 2
    inner_w = qw - 2
    max_area = min(CORE_AREA_MAX, inner_h * inner_w)
    if max_area < CORE_AREA_MIN:
        return None

    target = random.randint(CORE_AREA_MIN, max_area)
    inner = np.zeros((inner_h, inner_w), dtype=bool)
    sr = random.randrange(inner_h)
    sc = random.randrange(inner_w)
    inner[sr, sc] = True
    frontier = set(_neighbors4(sr, sc, inner_h, inner_w))

    while int(inner.sum()) < target:
        if not frontier:
            return None
        r, c = random.choice(tuple(frontier))
        frontier.remove((r, c))
        inner[r, c] = True
        for nr, nc in _neighbors4(r, c, inner_h, inner_w):
            if not inner[nr, nc]:
                frontier.add((nr, nc))

    mask = np.zeros((qh, qw), dtype=bool)
    mask[1:-1, 1:-1] = inner
    return mask


def _noise_candidates(core: np.ndarray) -> list[tuple[int, int]]:
    h, w = core.shape
    cells = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if core[r, c]:
                continue
            if any(core[nr, nc] for nr, nc in _neighbors4(r, c, h, w)):
                cells.append((r, c))
    random.shuffle(cells)
    return cells


def _make_variants(core: np.ndarray, target_score: int) -> list[np.ndarray] | None:
    variants = [core.copy() for _ in range(4)]
    candidates = _noise_candidates(core)
    remaining = target_score

    while remaining > 0:
        if not candidates:
            return None
        r, c = candidates.pop()
        contribution = random.choice([k for k in (1, 2, 3) if k <= remaining])
        for idx in random.sample(range(4), contribution):
            variants[idx][r, c] = True
        remaining -= contribution

    return variants


def _anchor_axis_len(anchor: str, qh: int, qw: int) -> int:
    return qw if anchor.startswith(("top", "bottom")) else qh


def _palette_coords(anchor: str, qh: int, qw: int, length: int) -> list[tuple[int, int]]:
    h = 2 * qh + 1
    w = 2 * qw + 1
    if anchor == "top_left":
        return [(0, c) for c in range(length)]
    if anchor == "top_right":
        return [(0, w - 1 - i) for i in range(length)]
    if anchor == "bottom_left":
        return [(h - 1, c) for c in range(length)]
    if anchor == "bottom_right":
        return [(h - 1, w - 1 - i) for i in range(length)]
    if anchor == "left_top":
        return [(r, 0) for r in range(length)]
    if anchor == "left_bottom":
        return [(h - 1 - i, 0) for i in range(length)]
    if anchor == "right_top":
        return [(r, w - 1) for r in range(length)]
    return [(h - 1 - i, w - 1) for i in range(length)]


def _render_input(
    qh: int,
    qw: int,
    bg: int,
    sep: int,
    shape_colors: list[int],
    variants: list[np.ndarray],
    anchor: str,
    palette: list[int],
) -> np.ndarray:
    h = 2 * qh + 1
    w = 2 * qw + 1
    grid = np.full((h, w), bg, dtype=int)
    grid[qh, :] = sep
    grid[:, qw] = sep

    quadrant_slices = [
        (slice(0, qh), slice(0, qw)),
        (slice(0, qh), slice(qw + 1, w)),
        (slice(qh + 1, h), slice(0, qw)),
        (slice(qh + 1, h), slice(qw + 1, w)),
    ]
    for color, variant, (rs, cs) in zip(shape_colors, variants, quadrant_slices):
        region = grid[rs, cs]
        region[variant] = int(color)

    for (r, c), color in zip(_palette_coords(anchor, qh, qw, len(palette)), palette):
        grid[r, c] = int(color)
    return grid


def _expected_output(core: np.ndarray, bg: int, palette: list[int], score: int) -> np.ndarray:
    out = np.full(core.shape, bg, dtype=int)
    out[core] = int(palette[score])
    return out


def _gen_attempt() -> np.ndarray | None:
    qh = random.randint(*QUAD_H_RANGE)
    qw = random.randint(*QUAD_W_RANGE)
    assert 2 * qh + 1 <= 30 and 2 * qw + 1 <= 30

    anchor = random.choice(ANCHORS)
    axis_len = _anchor_axis_len(anchor, qh, qw)

    colors = random.sample(COLOR_VALUES, 6)
    bg = colors[0]
    sep = colors[1]
    shape_colors = colors[2:]

    core = _generate_core_mask(qh, qw)
    if core is None:
        return None

    candidates = _noise_candidates(core)
    max_score = min(NOISE_SCORE_CAP, axis_len - 2, len(candidates) * 3)
    if max_score < 0:
        return None
    target_score = random.randint(0, max_score)

    variants = _make_variants(core, target_score)
    if variants is None:
        return None

    palette_pool = [color for color in COLOR_VALUES if color not in (bg, sep)]
    min_len = max(2, target_score + 1)
    max_len = min(axis_len, len(palette_pool))
    if min_len > max_len:
        return None
    palette_len = random.randint(min_len, max_len)
    palette = random.sample(palette_pool, palette_len)

    grid = _render_input(qh, qw, bg, sep, shape_colors, variants, anchor, palette)
    expected = _expected_output(core, bg, palette, target_score)

    try:
        if not np.array_equal(main(grid), expected):
            return None
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
