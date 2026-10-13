from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
AZURE = 8
COLOR_POOL = (1, 2, 3, 4, 5, 6, 7, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 600


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        q: deque[tuple[int, int]] = deque([(r, c)])
        pts: list[tuple[int, int]] = []
        while q:
            cr, cc = q.popleft()
            pts.append((cr, cc))
            for dr, dc in DIR4:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(pts)
    return comps


def _find_cross(g: np.ndarray) -> tuple[int, int]:
    full_rows = [int(r) for r in np.where(np.all(g == AZURE, axis=1))[0]]
    full_cols = [int(c) for c in np.where(np.all(g == AZURE, axis=0))[0]]
    assert len(full_rows) == 1, "expected exactly one full azure separator row"
    assert len(full_cols) == 1, "expected exactly one full azure separator column"
    sep_r, sep_c = full_rows[0], full_cols[0]
    assert sep_r == g.shape[0] - sep_r - 1, "separator row must split equal-height subgrids"
    assert sep_c == g.shape[1] - sep_c - 1, "separator column must split equal-width subgrids"
    return sep_r, sep_c


def _colors_by_first_seen(region: np.ndarray) -> list[int]:
    colors: list[int] = []
    seen: set[int] = set()
    for r, c in np.argwhere(region != BLACK):
        color = int(region[int(r), int(c)])
        if color not in seen:
            seen.add(color)
            colors.append(color)
    return colors


def _hole_count_for_color(region: np.ndarray, color: int) -> int:
    coords = np.argwhere(region == color)
    assert coords.size > 0, "shape color must be present"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    patch = region[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    black = patch == BLACK

    holes = 0
    h, w = black.shape
    for pts in _components(black):
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in pts):
            continue
        holes += 1
    return holes


def _shape_infos(left_top: np.ndarray) -> list[dict]:
    infos: list[dict] = []
    for color in _colors_by_first_seen(left_top):
        coords = [(int(r), int(c)) for r, c in np.argwhere(left_top == color)]
        assert coords, "each left-top color must define a shape"
        holes = _hole_count_for_color(left_top, color)
        infos.append({"color": color, "coords": coords, "holes": holes})
    assert infos, "expected at least one keyed shape in the left-top subgrid"
    return infos


def _pattern_colors(right_top: np.ndarray) -> list[int]:
    patterns: list[tuple[int, int, int]] = []
    for color in _colors_by_first_seen(right_top):
        coords = np.argwhere(right_top == color)
        assert coords.size > 0, "pattern color must be present"
        min_r, min_c = coords.min(axis=0)
        patterns.append((int(min_c), int(min_r), color))
    patterns.sort(key=lambda x: (x[0], x[1]))
    assert len(patterns) == 3, "expected exactly three right-top color patterns"
    return [color for _c, _r, color in patterns]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy left-top shapes into the left-bottom quadrant, recoloring them by hole-count lookup.

    Input structure:
    - Black background with a full one-pixel azure row and a full one-pixel azure
      column crossing at the center, forming four equal subgrids.
    - The left-top subgrid contains separated monochrome shapes, each with 0, 1,
      or 2 enclosed black holes inside its colored body.
    - The right-top subgrid contains exactly three pure-color patterns ordered
      from left to right; their colors are distinct from the left-top shape colors.
    - The two bottom subgrids are black except for the shared azure separators.

    Rule:
    - Count the number of enclosed black holes H in each left-top shape.
    - Read the right-top pattern colors from left to right as indices 0, 1, and 2.
    - In the left-bottom subgrid, place a copy of every left-top shape at the
      same relative coordinates, using pattern color H while leaving the copied
      holes black.
    - Keep the separators and the entire top half unchanged.

    Key insight:
    The hole count inside each source shape is the visible index key into the
    ordered pattern legend, so color replacement is driven by structure rather
    than by matching input colors or absolute positions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    sep_r, sep_c = _find_cross(g)

    left_top = g[:sep_r, :sep_c]
    right_top = g[:sep_r, sep_c + 1 :]
    legend = _pattern_colors(right_top)

    out = g.copy()
    out[sep_r + 1 :, :sep_c] = BLACK
    for info in _shape_infos(left_top):
        holes = int(info["holes"])
        assert 0 <= holes < len(legend), "hole count must select one of the three patterns"
        fill = int(legend[holes])
        for r, c in info["coords"]:
            out[sep_r + 1 + int(r), int(c)] = fill
    return out


def _mask_hole_count(mask: np.ndarray) -> int:
    empty = ~mask
    holes = 0
    h, w = empty.shape
    for pts in _components(empty):
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in pts):
            continue
        holes += 1
    return holes


def _shape_mask(rng: random.Random, holes: int, max_side: int = 5) -> np.ndarray | None:
    max_side = max(3, min(5, max_side))
    for _ in range(100):
        min_side = 4 if holes == 2 else 3
        if max_side < min_side:
            continue
        h = rng.randint(min_side, max_side)
        w = rng.randint(min_side, max_side)
        inner = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
        if len(inner) < holes:
            continue

        chosen: list[tuple[int, int]]
        if holes == 0:
            chosen = []
        elif holes == 1:
            chosen = [rng.choice(inner)]
        else:
            pairs = [
                (a, b)
                for i, a in enumerate(inner)
                for b in inner[i + 1 :]
                if abs(a[0] - b[0]) + abs(a[1] - b[1]) > 1
            ]
            if not pairs:
                continue
            chosen = list(rng.choice(pairs))

        mask = np.ones((h, w), dtype=bool)
        for r, c in chosen:
            mask[r, c] = False
        if len(_components(mask)) == 1 and _mask_hole_count(mask) == holes:
            return mask
    return None


PATTERN_SHAPES = (
    ((1, 1), (1, 1)),
    ((1, 1, 1), (1, 1, 1)),
    ((1, 1), (1, 1), (1, 1)),
    ((1, 0), (1, 1), (1, 0)),
    ((0, 1, 0), (1, 1, 1)),
    ((1, 1, 0), (0, 1, 1)),
)


def _pattern_mask(rng: random.Random) -> np.ndarray:
    return np.array(rng.choice(PATTERN_SHAPES), dtype=bool)


def _candidate_positions(
    sub_h: int, sub_w: int, obj_h: int, obj_w: int
) -> list[tuple[int, int]]:
    row_start = 1 if sub_h - obj_h >= 2 else 0
    row_stop = sub_h - obj_h - (1 if sub_h - obj_h >= 2 else 0)
    col_start = 1 if sub_w - obj_w >= 2 else 0
    col_stop = sub_w - obj_w - (1 if sub_w - obj_w >= 2 else 0)
    return [
        (r, c)
        for r in range(row_start, row_stop + 1)
        for c in range(col_start, col_stop + 1)
    ]


def _place_shapes(
    grid: np.ndarray,
    sub_h: int,
    sub_w: int,
    specs: list[dict],
    rng: random.Random,
) -> list[dict] | None:
    for _ in range(80):
        trial = grid[:sub_h, :sub_w].copy()
        occ = np.zeros((sub_h, sub_w), dtype=bool)
        placed: list[dict] = []
        ordered = sorted(specs, key=lambda x: -int(np.sum(x["mask"])))
        for spec in ordered:
            mask = spec["mask"]
            mh, mw = mask.shape
            candidates = _candidate_positions(sub_h, sub_w, mh, mw)
            rng.shuffle(candidates)
            found: tuple[int, int] | None = None
            for r0, c0 in candidates:
                rr0, rr1 = max(0, r0 - 1), min(sub_h, r0 + mh + 1)
                cc0, cc1 = max(0, c0 - 1), min(sub_w, c0 + mw + 1)
                if not np.any(occ[rr0:rr1, cc0:cc1]):
                    found = (r0, c0)
                    break
            if found is None:
                break

            r0, c0 = found
            color = int(spec["color"])
            for rr, cc in np.argwhere(mask):
                trial[r0 + int(rr), c0 + int(cc)] = color
            rr0, rr1 = max(0, r0 - 1), min(sub_h, r0 + mh + 1)
            cc0, cc1 = max(0, c0 - 1), min(sub_w, c0 + mw + 1)
            occ[rr0:rr1, cc0:cc1] = True
            placed.append({**spec, "r0": r0, "c0": c0})

        if len(placed) == len(specs):
            grid[:sub_h, :sub_w] = trial
            return placed
    return None


def _place_patterns(
    grid: np.ndarray,
    sub_h: int,
    sub_w: int,
    sep_c: int,
    colors: list[int],
    rng: random.Random,
) -> bool:
    for _ in range(100):
        masks = [_pattern_mask(rng) for _ in range(3)]
        widths = [int(m.shape[1]) for m in masks]
        min_total = sum(widths) + 2
        if min_total > sub_w:
            continue

        extra = sub_w - min_total
        left_min = 1 if extra >= 1 else 0
        extra -= left_min
        left_gap = left_min + rng.randint(0, extra)
        extra -= left_gap - left_min
        gap1_extra = rng.randint(0, extra)
        extra -= gap1_extra
        gap2_extra = rng.randint(0, extra)
        gap1 = 1 + gap1_extra
        gap2 = 1 + gap2_extra

        starts = [
            left_gap,
            left_gap + widths[0] + gap1,
            left_gap + widths[0] + gap1 + widths[1] + gap2,
        ]
        for mask, color, local_c in zip(masks, colors, starts, strict=True):
            ph, pw = mask.shape
            if sub_h - ph >= 2:
                local_r = rng.randint(1, sub_h - ph - 1)
            else:
                local_r = rng.randint(0, sub_h - ph)
            for rr, cc in np.argwhere(mask):
                grid[local_r + int(rr), sep_c + 1 + local_c + int(cc)] = int(color)
        return True
    return False


def _expected_output(
    inp: np.ndarray,
    placed: list[dict],
    pattern_colors: list[int],
    sep_r: int,
    sep_c: int,
) -> np.ndarray:
    out = inp.copy()
    out[sep_r + 1 :, :sep_c] = BLACK
    for spec in placed:
        fill = int(pattern_colors[int(spec["holes"])])
        r0, c0 = int(spec["r0"]), int(spec["c0"])
        mask = spec["mask"]
        for rr, cc in np.argwhere(mask):
            out[sep_r + 1 + r0 + int(rr), c0 + int(cc)] = fill
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    sub_h = rng.randint(10, 12)
    sub_w = rng.randint(10, 12)
    height = 2 * sub_h + 1
    width = 2 * sub_w + 1
    grid = np.full((height, width), BLACK, dtype=int)
    grid[sub_h, :] = AZURE
    grid[:, sub_w] = AZURE

    shape_count = rng.choice([3, 4]) if min(sub_h, sub_w) >= 11 else 3

    colors = rng.sample(list(COLOR_POOL), shape_count + 3)
    shape_colors = colors[:shape_count]
    pattern_colors = colors[shape_count:]

    hole_counts = [0, 1, 2] + [rng.randrange(3) for _ in range(shape_count - 3)]
    rng.shuffle(hole_counts)

    specs: list[dict] = []
    for color, holes in zip(shape_colors, hole_counts, strict=True):
        max_side = 5 if shape_count == 3 and min(sub_h, sub_w) == 12 else 4
        mask = _shape_mask(rng, holes, max_side=max_side)
        if mask is None:
            return None
        specs.append({"color": int(color), "holes": int(holes), "mask": mask})

    placed = _place_shapes(grid, sub_h, sub_w, specs, rng)
    if placed is None:
        return None
    if not _place_patterns(grid, sub_h, sub_w, sub_w, pattern_colors, rng):
        return None

    expected = _expected_output(grid, placed, pattern_colors, sub_h, sub_w)
    try:
        if np.array_equal(main(grid), expected):
            return grid
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hole-index quadrant puzzle")
