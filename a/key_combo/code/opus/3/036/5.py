from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import bbox, crop_bbox, find_components, get_bg, paste, randomly_spaced_indices


BG = 0
MASK_COLOR = 3
PATTERN_SIDE_CHOICES = (9, 10, 11)
PATTERN_COLOR_POOL = (1, 2, 6, 7, 8, 9)
MASK_SHAPES = (
    (3, 3),
    (3, 4),
    (4, 3),
    (3, 5),
    (5, 3),
    (4, 4),
    (4, 5),
    (5, 4),
)
MARKER_COUNT_CHOICES = (2, 3, 4)
GENERATION_ATTEMPTS = 400


def _d4_coords(r: int, c: int, n: int) -> tuple[tuple[int, int], ...]:
    last = int(n) - 1
    return (
        (r, c),
        (r, last - c),
        (last - r, c),
        (last - r, last - c),
        (c, r),
        (last - c, last - r),
        (c, last - r),
        (last - c, r),
    )


def _bbox_is_filled_rect(mask: np.ndarray) -> bool:
    box = bbox(mask)
    if box is None:
        return False
    return bool(np.all(crop_bbox(mask, box)))


def _expand_to_square(box: tuple[int, int, int, int], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = map(int, box)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h == w:
        return r0, c0, r1, c1

    side = max(h, w)
    if h < side:
        extra = side - h
        r0 -= extra // 2
        r1 += extra - extra // 2
    if w < side:
        extra = side - w
        c0 -= extra // 2
        c1 += extra - extra // 2

    H, W = map(int, shape)
    if r0 < 0:
        r1 -= r0
        r0 = 0
    if c0 < 0:
        c1 -= c0
        c0 = 0
    if r1 >= H:
        shift = r1 - H + 1
        r0 -= shift
        r1 -= shift
    if c1 >= W:
        shift = c1 - W + 1
        c0 -= shift
        c1 -= shift
    return int(r0), int(c0), int(r1), int(c1)


def _pattern_box(grid: np.ndarray, mask: np.ndarray) -> tuple[int, int, int, int]:
    comps = find_components(grid, bg=BG, connectivity=4, monochromatic=False)
    chosen = None
    for comp in comps:
        coords = comp["coords"]
        if any(mask[int(r), int(c)] for r, c in coords):
            if chosen is None or int(comp["area"]) > int(chosen["area"]):
                chosen = comp
    assert chosen is not None, "expected one foreground component containing the mask"
    return _expand_to_square(chosen["bbox"], grid.shape)


def _recover_template(
    grid: np.ndarray,
    pattern_box: tuple[int, int, int, int],
    mask_box: tuple[int, int, int, int],
) -> np.ndarray:
    pr0, pc0, pr1, pc1 = map(int, pattern_box)
    mr0, mc0, mr1, mc1 = map(int, mask_box)
    pattern = crop_bbox(grid, pattern_box)
    side_h, side_w = map(int, pattern.shape)
    assert side_h == side_w, "expected a square symmetric pattern box"

    out = np.empty((mr1 - mr0 + 1, mc1 - mc0 + 1), dtype=int)
    for r in range(mr0, mr1 + 1):
        for c in range(mc0, mc1 + 1):
            rr = r - pr0
            cc = c - pc0
            visible = {
                int(pattern[tr, tc])
                for tr, tc in _d4_coords(rr, cc, side_h)
                if int(pattern[tr, tc]) != MASK_COLOR
            }
            assert visible, "each masked cell must have a visible symmetric copy"
            assert len(visible) == 1, "all visible symmetric copies must agree"
            out[r - mr0, c - mc0] = next(iter(visible))
    return out


def _sorted_markers(
    grid: np.ndarray,
    pattern_box: tuple[int, int, int, int],
    bg: int,
) -> tuple[list[tuple[int, int, int]], bool]:
    r0, c0, r1, c1 = map(int, pattern_box)
    outside = np.ones(grid.shape, dtype=bool)
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    coords = np.argwhere((grid != int(bg)) & outside)
    markers = [(int(r), int(c), int(grid[r, c])) for r, c in coords]
    assert 2 <= len(markers) <= 4, "expected 2-4 external marker pixels"
    assert len({color for _r, _c, color in markers}) == len(markers), "marker colors must be distinct"

    rows = {r for r, _c, _color in markers}
    cols = {c for _r, c, _color in markers}
    horizontal = len(rows) == 1
    vertical = len(cols) == 1
    assert horizontal != vertical, "markers must lie on exactly one shared row or one shared column"

    markers.sort(key=(lambda item: item[1]) if horizontal else (lambda item: item[0]))
    return markers, horizontal


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the hidden symmetric template under the green mask and concatenate recolored copies in marker order.

    Input structure:
    - A black background surrounds one central square pattern that is symmetric under the full dihedral group of the square.
    - A solid green rectangle masks part of that pattern, and a few isolated single-pixel markers sit outside the pattern.
    - The marker pixels have distinct colors and share exactly one row or one column.

    Rule:
    - Find the green rectangle and the square pattern containing it.
    - For each masked cell, inspect every rotation/reflection-equivalent position inside the square pattern and ignore still-masked cells; the remaining visible copies all give the recovery color.
    - Crop the recovered rectangle as template T.
    - Sort the markers left-to-right when they share a row, or top-to-bottom when they share a column.
    - For each sorted marker, recolor every non-black cell of T to that marker's color and concatenate the copies horizontally or vertically to match the marker axis.

    Key insight:
    - The output template is not explicitly visible in the input: it must be reconstructed from dihedral symmetry before the marker-ordered concatenation can be applied.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = int(get_bg(g))
    mask = g == MASK_COLOR
    mask_box = bbox(mask)
    assert mask_box is not None, "expected a green mask rectangle"
    assert _bbox_is_filled_rect(mask), "mask must be one solid rectangle"

    pattern_box = _pattern_box(g, mask)
    template = _recover_template(g, pattern_box, mask_box)
    markers, horizontal = _sorted_markers(g, pattern_box, bg)

    th, tw = map(int, template.shape)
    if horizontal:
        out = np.full((th, tw * len(markers)), bg, dtype=int)
        for i, (_r, _c, color) in enumerate(markers):
            patch = np.full(template.shape, bg, dtype=int)
            patch[template != bg] = int(color)
            paste(out, patch, 0, i * tw)
    else:
        out = np.full((th * len(markers), tw), bg, dtype=int)
        for i, (_r, _c, color) in enumerate(markers):
            patch = np.full(template.shape, bg, dtype=int)
            patch[template != bg] = int(color)
            paste(out, patch, i * th, 0)
    return out


def _orbits(n: int) -> list[tuple[tuple[int, int], ...]]:
    seen = np.zeros((n, n), dtype=bool)
    out = []
    for r in range(n):
        for c in range(n):
            if seen[r, c]:
                continue
            orbit = tuple(sorted(set(_d4_coords(r, c, n))))
            for rr, cc in orbit:
                seen[rr, cc] = True
            out.append(orbit)
    return out


def _build_shape_mask(n: int) -> np.ndarray:
    shape = np.zeros((n, n), dtype=bool)
    shape[0, :] = True
    shape[-1, :] = True
    shape[:, 0] = True
    shape[:, -1] = True
    for i in range(n):
        shape[i, i] = True
        shape[i, n - 1 - i] = True

    if random.choice((False, True)):
        mids = [n // 2] if n % 2 == 1 else [n // 2 - 1, n // 2]
        for mid in mids:
            shape[mid, :] = True
            shape[:, mid] = True

    interior_orbits = []
    for orbit in _orbits(n):
        if all(0 < r < n - 1 and 0 < c < n - 1 for r, c in orbit):
            if not all(shape[r, c] for r, c in orbit):
                interior_orbits.append(orbit)
    random.shuffle(interior_orbits)
    extra = random.randint(max(1, len(interior_orbits) // 6), max(2, len(interior_orbits) // 3))
    for orbit in interior_orbits[:extra]:
        for r, c in orbit:
            shape[r, c] = True
    return shape


def _colorize_pattern(shape: np.ndarray, colors: list[int]) -> np.ndarray:
    n = int(shape.shape[0])
    pattern = np.zeros((n, n), dtype=int)
    filled_orbits = [orbit for orbit in _orbits(n) if shape[orbit[0][0], orbit[0][1]]]
    random.shuffle(filled_orbits)

    orbit_colors = list(colors)
    while len(orbit_colors) < len(filled_orbits):
        orbit_colors.append(random.choice(colors))
    random.shuffle(orbit_colors[len(colors) :])

    for orbit, color in zip(filled_orbits, orbit_colors, strict=False):
        for r, c in orbit:
            pattern[r, c] = int(color)
    return pattern


def _mask_has_visible_copy(r0: int, c0: int, h: int, w: int, n: int) -> bool:
    for r in range(r0, r0 + h):
        for c in range(c0, c0 + w):
            if all(r0 <= rr < r0 + h and c0 <= cc < c0 + w for rr, cc in _d4_coords(r, c, n)):
                return False
    return True


def _candidate_mask_boxes(pattern: np.ndarray) -> list[tuple[int, int, int, int]]:
    n = int(pattern.shape[0])
    candidates = []
    for h, w in MASK_SHAPES:
        if h > n or w > n:
            continue
        for r0 in range(n - h + 1):
            for c0 in range(n - w + 1):
                if not _mask_has_visible_copy(r0, c0, h, w, n):
                    continue
                template = pattern[r0 : r0 + h, c0 : c0 + w]
                filled = int(np.count_nonzero(template))
                if filled == 0 or filled == template.size:
                    continue
                if filled < max(3, (template.size + 2) // 3):
                    continue
                candidates.append((r0, c0, h, w))
    return candidates


def _expected_output(
    template: np.ndarray,
    markers: list[tuple[int, int, int]],
    horizontal: bool,
) -> np.ndarray:
    th, tw = map(int, template.shape)
    if horizontal:
        out = np.full((th, tw * len(markers)), BG, dtype=int)
        for i, (_r, _c, color) in enumerate(markers):
            patch = np.full(template.shape, BG, dtype=int)
            patch[template != BG] = int(color)
            paste(out, patch, 0, i * tw)
    else:
        out = np.full((th * len(markers), tw), BG, dtype=int)
        for i, (_r, _c, color) in enumerate(markers):
            patch = np.full(template.shape, BG, dtype=int)
            patch[template != BG] = int(color)
            paste(out, patch, i * th, 0)
    return out


def _place_markers(
    grid: np.ndarray,
    pattern_box: tuple[int, int, int, int],
    colors: list[int],
    horizontal: bool,
) -> list[tuple[int, int, int]] | None:
    H, W = map(int, grid.shape)
    r0, c0, r1, c1 = map(int, pattern_box)
    count = len(colors)

    if horizontal:
        side_rows = []
        if r0 >= 2:
            side_rows.append(list(range(0, r0 - 1)))
        if r1 + 2 < H:
            side_rows.append(list(range(r1 + 2, H)))
        if not side_rows:
            return None
        row = random.choice(random.choice(side_rows))
        try:
            cols = randomly_spaced_indices(W, count, border_size=0, padding=1).tolist()
        except ValueError:
            return None
        return [(int(row), int(c), int(color)) for c, color in zip(cols, colors, strict=False)]

    side_cols = []
    if c0 >= 2:
        side_cols.append(list(range(0, c0 - 1)))
    if c1 + 2 < W:
        side_cols.append(list(range(c1 + 2, W)))
    if not side_cols:
        return None
    col = random.choice(random.choice(side_cols))
    try:
        rows = randomly_spaced_indices(H, count, border_size=0, padding=1).tolist()
    except ValueError:
        return None
    return [(int(r), int(col), int(color)) for r, color in zip(rows, colors, strict=False)]


def _gen_attempt() -> np.ndarray | None:
    side = random.choice(PATTERN_SIDE_CHOICES)
    count = random.choice(MARKER_COUNT_CHOICES)
    color_count = random.choice((2, 3))
    pattern_colors = random.sample(PATTERN_COLOR_POOL, color_count)
    marker_pool = [color for color in range(1, 10) if color not in {BG, MASK_COLOR, *pattern_colors}]
    marker_colors = random.sample(marker_pool, count)
    horizontal = random.choice((False, True))

    shape = _build_shape_mask(side)
    pattern = _colorize_pattern(shape, pattern_colors)
    mask_candidates = _candidate_mask_boxes(pattern)
    if not mask_candidates:
        return None
    mr0, mc0, mh, mw = random.choice(mask_candidates)
    mr1 = mr0 + mh - 1
    mc1 = mc0 + mw - 1
    template = pattern[mr0 : mr1 + 1, mc0 : mc1 + 1].copy()

    H = random.randint(side + 5, min(20, side + 8))
    W = random.randint(side + 5, min(20, side + 8))
    pr0 = random.randint(2, H - side - 2)
    pc0 = random.randint(2, W - side - 2)
    pattern_box = (pr0, pc0, pr0 + side - 1, pc0 + side - 1)

    grid = np.zeros((H, W), dtype=int)
    paste(grid, pattern, pr0, pc0)
    grid[pr0 + mr0 : pr0 + mr1 + 1, pc0 + mc0 : pc0 + mc1 + 1] = MASK_COLOR

    markers = _place_markers(grid, pattern_box, marker_colors, horizontal)
    if markers is None:
        return None
    for r, c, color in markers:
        if grid[r, c] != BG:
            return None
        grid[r, c] = int(color)

    expected = _expected_output(template, sorted(markers, key=(lambda item: item[1]) if horizontal else (lambda item: item[0])), horizontal)
    try:
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after repeated attempts")
