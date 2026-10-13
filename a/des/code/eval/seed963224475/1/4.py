from __future__ import annotations

from collections import Counter
import random

import numpy as np


AZURE = 8
COLOR_POOL = tuple(range(10))

# --------------- generator config ---------------

QUAD_H_RANGE = (6, 11)
QUAD_W_RANGE = (6, 11)
MAX_ARM_SUM = 7
GEN_ATTEMPTS = 400
TEMPLATE_RETRIES = 200


Bounds = tuple[int, int, int, int]
Arms = tuple[int, int, int, int]


def _most_common_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[cnts.argmax()])


def _find_divider_cross(grid: np.ndarray, bg: int) -> tuple[int, int, int]:
    h, w = grid.shape
    row_candidates = [
        (r, int(grid[r, 0]))
        for r in range(h)
        if np.all(grid[r, :] == grid[r, 0]) and int(grid[r, 0]) != bg
    ]
    col_candidates = [
        (c, int(grid[0, c]))
        for c in range(w)
        if np.all(grid[:, c] == grid[0, c]) and int(grid[0, c]) != bg
    ]

    matches: list[tuple[int, int, int]] = []
    for r, row_color in row_candidates:
        if not (0 < r < h - 1) or 2 * r + 1 != h:
            continue
        for c, col_color in col_candidates:
            if row_color != col_color:
                continue
            if not (0 < c < w - 1) or 2 * c + 1 != w:
                continue
            matches.append((r, c, row_color))

    assert len(matches) == 1, "expected exactly one centered divider cross"
    return matches[0]


def _quadrant_bounds(h: int, w: int, mid_r: int, mid_c: int) -> list[Bounds]:
    return [
        (0, mid_r, 0, mid_c),
        (0, mid_r, mid_c + 1, w),
        (mid_r + 1, h, 0, mid_c),
        (mid_r + 1, h, mid_c + 1, w),
    ]


def _cross_points(center: tuple[int, int], arms: Arms) -> list[tuple[int, int]]:
    r, c = center
    up, down, left, right = arms
    pts = [(r, c)]
    pts.extend((r - d, c) for d in range(1, up + 1))
    pts.extend((r + d, c) for d in range(1, down + 1))
    pts.extend((r, c - d) for d in range(1, left + 1))
    pts.extend((r, c + d) for d in range(1, right + 1))
    return pts


def _extract_seed_and_template(region: np.ndarray, color: int) -> tuple[tuple[int, int], Arms | None]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(region == color)]
    assert coords, "quadrant must contain the seed color"
    if len(coords) == 1:
        return coords[0], None

    row_counts = Counter(r for r, _ in coords)
    col_counts = Counter(c for _, c in coords)
    center_r, row_count = row_counts.most_common(1)[0]
    center_c, col_count = col_counts.most_common(1)[0]
    assert row_count >= 2 and col_count >= 2, "completed pattern must be a cross"
    assert int(region[center_r, center_c]) == color

    up = 0
    rr = center_r - 1
    while rr >= 0 and int(region[rr, center_c]) == color:
        up += 1
        rr -= 1

    down = 0
    rr = center_r + 1
    while rr < region.shape[0] and int(region[rr, center_c]) == color:
        down += 1
        rr += 1

    left = 0
    cc = center_c - 1
    while cc >= 0 and int(region[center_r, cc]) == color:
        left += 1
        cc -= 1

    right = 0
    cc = center_c + 1
    while cc < region.shape[1] and int(region[center_r, cc]) == color:
        right += 1
        cc += 1

    arms = (up, down, left, right)
    expected = set(_cross_points((center_r, center_c), arms))
    assert expected == set(coords), "seed color must form one clean cross"
    return (center_r, center_c), arms


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the missing quadrant by copying the cross template of the matching seed color.

    The grid is split into four equal quadrants by one full horizontal and one full vertical
    divider line. Each quadrant contains one non-background seed color; three quadrants already
    show a same-color cross grown from that seed, while the fourth contains only the seed pixel.
    The missing cross uses the arm lengths of the completed quadrant with the same seed color.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _most_common_color(g)
    mid_r, mid_c, divider = _find_divider_cross(g, bg)

    templates: dict[int, Arms] = {}
    missing: tuple[int, int, int, Bounds] | None = None

    for bounds in _quadrant_bounds(h, w, mid_r, mid_c):
        r0, r1, c0, c1 = bounds
        region = g[r0:r1, c0:c1]
        colors = [int(v) for v in np.unique(region) if int(v) not in (bg, divider)]
        assert len(colors) == 1, "each quadrant must contain exactly one seed color"
        color = colors[0]

        (seed_r, seed_c), arms = _extract_seed_and_template(region, color)
        if arms is None:
            assert missing is None, "there must be exactly one unfinished quadrant"
            missing = (r0 + seed_r, c0 + seed_c, color, bounds)
        else:
            if color in templates:
                assert templates[color] == arms, "same color must imply same template"
            else:
                templates[color] = arms

    assert missing is not None, "one quadrant must be unfinished"
    seed_r, seed_c, color, bounds = missing
    assert color in templates, "unfinished seed color must match a completed template"

    out = g.copy()
    r0, r1, c0, c1 = bounds
    for rr, cc in _cross_points((seed_r, seed_c), templates[color]):
        assert r0 <= rr < r1 and c0 <= cc < c1, "completed cross must stay inside its quadrant"
        if int(out[rr, cc]) == bg:
            out[rr, cc] = color
    return out


def _sample_template(qh: int, qw: int, used: set[Arms]) -> Arms | None:
    max_vertical = min(MAX_ARM_SUM, qh - 3)
    max_horizontal = min(MAX_ARM_SUM, qw - 3)
    if max_vertical < 2 or max_horizontal < 2:
        return None

    for _ in range(TEMPLATE_RETRIES):
        v_sum = random.randint(2, max_vertical)
        up = random.randint(1, v_sum - 1)
        down = v_sum - up

        h_sum = random.randint(2, max_horizontal)
        left = random.randint(1, h_sum - 1)
        right = h_sum - left

        arms = (up, down, left, right)
        if up == down and left == right:
            continue
        if arms in used:
            continue
        return arms
    return None


def _sample_seed(qh: int, qw: int, arms: Arms) -> tuple[int, int]:
    up, down, left, right = arms
    min_r = up + 1
    max_r = qh - down - 2
    min_c = left + 1
    max_c = qw - right - 2
    assert min_r <= max_r and min_c <= max_c
    return random.randint(min_r, max_r), random.randint(min_c, max_c)


def _paint_cross(grid: np.ndarray, bounds: Bounds, seed: tuple[int, int], color: int, arms: Arms, full: bool) -> None:
    r0, r1, c0, c1 = bounds
    seed_r = r0 + seed[0]
    seed_c = c0 + seed[1]
    assert int(grid[seed_r, seed_c]) != AZURE
    grid[seed_r, seed_c] = color
    if not full:
        return

    for rr, cc in _cross_points((seed_r, seed_c), arms):
        assert r0 <= rr < r1 and c0 <= cc < c1
        grid[rr, cc] = color


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    qh = random.randint(*QUAD_H_RANGE)
    qw = random.randint(*QUAD_W_RANGE)
    h = 2 * qh + 1
    w = 2 * qw + 1

    bg_choices = [c for c in COLOR_POOL if c != AZURE]
    bg = random.choice(bg_choices)
    seed_palette = [c for c in COLOR_POOL if c not in (bg, AZURE)]
    seed_colors = random.sample(seed_palette, 3)

    templates: dict[int, Arms] = {}
    used_templates: set[Arms] = set()
    for color in seed_colors:
        arms = _sample_template(qh, qw, used_templates)
        if arms is None:
            return None
        templates[color] = arms
        used_templates.add(arms)

    missing_quadrant = random.randrange(4)
    target_color = random.choice(seed_colors)
    completed_quadrants = [idx for idx in range(4) if idx != missing_quadrant]
    assigned_complete_colors = seed_colors[:]
    random.shuffle(assigned_complete_colors)

    quadrant_colors = [0] * 4
    for idx, color in zip(completed_quadrants, assigned_complete_colors, strict=False):
        quadrant_colors[idx] = color
    quadrant_colors[missing_quadrant] = target_color

    inp = np.full((h, w), bg, dtype=int)
    out = np.full((h, w), bg, dtype=int)
    inp[qh, :] = AZURE
    inp[:, qw] = AZURE
    out[qh, :] = AZURE
    out[:, qw] = AZURE

    bounds_list = _quadrant_bounds(h, w, qh, qw)
    for idx, bounds in enumerate(bounds_list):
        color = quadrant_colors[idx]
        arms = templates[color]
        seed = _sample_seed(qh, qw, arms)

        full_in_input = idx != missing_quadrant
        _paint_cross(inp, bounds, seed, color, arms, full_in_input)
        _paint_cross(out, bounds, seed, color, arms, True)

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        case = _build_case()
        if case is None:
            continue
        inp, out = case
        try:
            if not np.array_equal(main(inp), out):
                continue
        except AssertionError:
            continue
        return inp
    raise RuntimeError("failed to generate a valid input")
