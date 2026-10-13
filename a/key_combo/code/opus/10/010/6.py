from __future__ import annotations

import math
import random
from collections import deque

import numpy as np


BLACK = 0
YELLOW = 4
ORANGE = 7
DIVIDER_COLORS = {YELLOW, ORANGE}
REGION_COLORS = [1, 2, 3, 5, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _monochrome_components(grid: np.ndarray) -> list[dict]:
    comps: list[dict] = []
    for color in sorted(set(map(int, np.unique(grid))) - {BLACK, YELLOW, ORANGE}):
        for coords in _components_from_mask(grid == color):
            comps.append({"color": color, "coords": coords})
    return comps


def _clockwise_angle(coords: np.ndarray, center: np.ndarray) -> float:
    centroid = coords.mean(axis=0)
    dy = float(centroid[0] - center[0])
    dx = float(centroid[1] - center[1])
    return math.atan2(dy, dx) % (2 * math.pi)


def _ordered_regions(grid: np.ndarray) -> list[dict]:
    divider = np.argwhere(np.isin(grid, sorted(DIVIDER_COLORS)))
    assert divider.size > 0, "expected a yellow/orange divider structure"
    center = divider.mean(axis=0)

    regions = _monochrome_components(grid)
    assert len(regions) >= 4, "expected at least four colored regions"
    regions.sort(key=lambda item: _clockwise_angle(item["coords"], center))

    angles = [_clockwise_angle(region["coords"], center) for region in regions]
    for left, right in zip(angles, angles[1:]):
        assert not math.isclose(left, right, abs_tol=1e-6), "region angles must be unique"
    return regions


def _count_black_components(grid: np.ndarray) -> int:
    return len(_components_from_mask(grid == BLACK))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the colored sectors around a yellow/orange divider by a cyclic shift whose step count is encoded by black subregion count.

    Input structure:
    - The canvas uses black (0) as background, with a fixed central divider made of yellow (4) and orange (7) pixels.
    - Every other non-black color forms one 4-connected foreground region positioned around that divider.
    - Some regions contain black interior pockets, and some of those pockets are cut by same-color 1-pixel barriers that belong to the surrounding region.

    Rule:
    - Keep every yellow and orange divider pixel unchanged.
    - Count the number K of 4-connected black background components in the whole grid.
    - Sort all non-black, non-divider regions by centroid angle around the divider centroid, using screen coordinates with rightward = 0 degrees and clockwise increase.
    - Keep each region's geometry fixed, but recolor region i with the color of region (i-K) modulo N in that clockwise order.

    Key insight:
    - The small same-color barrier strokes inside a region are not decorative: they split black pockets into extra connected components, and that global black-component count becomes the rotation step for every region color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    regions = _ordered_regions(g)
    k = _count_black_components(g)
    assert 1 < k < len(regions), "black component count must define a non-trivial cyclic shift"

    out = g.copy()
    source_colors = [int(region["color"]) for region in regions]
    for i, region in enumerate(regions):
        rr, cc = region["coords"][:, 0], region["coords"][:, 1]
        out[rr, cc] = source_colors[(i - k) % len(regions)]
    return out


def _build_divider(h: int, w: int, topology: str, region_count: int) -> np.ndarray | None:
    cr, cc = h // 2, w // 2
    g = np.zeros((h, w), dtype=int)

    if topology == "cross":
        g[:, cc] = ORANGE
        g[cr, :] = ORANGE
        extra_needed = region_count - 4
        candidates: list[tuple[str, int]] = []
        left_cols = list(range(2, cc - 2))
        right_cols = list(range(cc + 3, w - 2))
        lower_rows = list(range(cr + 3, h - 2))
        if left_cols:
            candidates.append(("top_left_vertical", random.choice(left_cols)))
        if right_cols:
            candidates.append(("top_right_vertical", random.choice(right_cols)))
        if lower_rows:
            candidates.append(("bottom_left_horizontal", random.choice(lower_rows)))
            candidates.append(("bottom_right_horizontal", random.choice(lower_rows)))
        if len(candidates) < extra_needed:
            return None
        for name, value in random.sample(candidates, extra_needed):
            if name == "top_left_vertical":
                g[:cr, value] = ORANGE
            elif name == "top_right_vertical":
                g[:cr, value] = ORANGE
            elif name == "bottom_left_horizontal":
                g[value, :cc] = ORANGE
            else:
                g[value, cc + 1 :] = ORANGE
    else:
        g[cr, :] = ORANGE
        g[cr:, cc] = ORANGE
        extra_needed = region_count - 3
        top_candidates: list[tuple[str, int]] = []
        other_candidates: list[tuple[str, int]] = []
        left_cols = list(range(2, cc - 2))
        right_cols = list(range(cc + 3, w - 2))
        lower_rows = list(range(cr + 3, h - 2))
        if left_cols:
            top_candidates.append(("top_left_vertical", random.choice(left_cols)))
        if right_cols:
            top_candidates.append(("top_right_vertical", random.choice(right_cols)))
        if lower_rows:
            other_candidates.append(("bottom_left_horizontal", random.choice(lower_rows)))
            other_candidates.append(("bottom_right_horizontal", random.choice(lower_rows)))
        if not top_candidates or len(top_candidates) + len(other_candidates) < extra_needed:
            return None
        chosen = [random.choice(top_candidates)]
        remaining = [item for item in top_candidates + other_candidates if item not in chosen]
        if len(remaining) < extra_needed - 1:
            return None
        chosen.extend(random.sample(remaining, extra_needed - 1))
        for name, value in chosen:
            if name == "top_left_vertical":
                g[:cr, value] = ORANGE
            elif name == "top_right_vertical":
                g[:cr, value] = ORANGE
            elif name == "bottom_left_horizontal":
                g[value, :cc] = ORANGE
            else:
                g[value, cc + 1 :] = ORANGE

    g[cr - 1 : cr + 2, cc - 1 : cc + 2] = YELLOW
    return g


def _sector_components(divider_grid: np.ndarray) -> list[np.ndarray]:
    sectors = _components_from_mask(~np.isin(divider_grid, sorted(DIVIDER_COLORS)))
    sectors.sort(key=lambda coords: tuple(coords.mean(axis=0)))
    return sectors


def _find_rectangles(mask: np.ndarray, min_h: int, max_h: int, min_w: int, max_w: int) -> list[tuple[int, int, int, int]]:
    h, w = mask.shape
    options: list[tuple[int, int, int, int]] = []
    for hh in range(min_h, max_h + 1):
        for ww in range(min_w, max_w + 1):
            for r0 in range(1, h - hh):
                r1 = r0 + hh - 1
                for c0 in range(1, w - ww):
                    c1 = c0 + ww - 1
                    if not mask[r0 : r1 + 1, c0 : c1 + 1].all():
                        continue
                    if not mask[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2].all():
                        continue
                    options.append((r0, r1, c0, c1))
    random.shuffle(options)
    return options


def _feature_options(sector_mask: np.ndarray) -> tuple[list[dict], list[dict]]:
    sector_coords = np.argwhere(sector_mask)
    r0, c0 = sector_coords.min(axis=0)
    r1, c1 = sector_coords.max(axis=0)
    local = sector_mask[r0 : r1 + 1, c0 : c1 + 1]

    simple: list[dict] = []
    barrier: list[dict] = []
    for rr0, rr1, cc0, cc1 in _find_rectangles(local, 2, 4, 2, 4):
        pocket = [(r0 + r, c0 + c) for r in range(rr0, rr1 + 1) for c in range(cc0, cc1 + 1)]
        simple.append({"pocket": pocket, "barrier": []})

    for rr0, rr1, cc0, cc1 in _find_rectangles(local, 3, 5, 3, 5):
        if cc1 - cc0 + 1 >= 3:
            mid_c = random.choice(list(range(cc0 + 1, cc1)))
            pocket = [(r0 + r, c0 + c) for r in range(rr0, rr1 + 1) for c in range(cc0, cc1 + 1)]
            barrier_cells = [(r0 + r, c0 + mid_c) for r in range(rr0, rr1 + 1)]
            barrier.append({"pocket": pocket, "barrier": barrier_cells})
        if rr1 - rr0 + 1 >= 3:
            mid_r = random.choice(list(range(rr0 + 1, rr1)))
            pocket = [(r0 + r, c0 + c) for r in range(rr0, rr1 + 1) for c in range(cc0, cc1 + 1)]
            barrier_cells = [(r0 + mid_r, c0 + c) for c in range(cc0, cc1 + 1)]
            barrier.append({"pocket": pocket, "barrier": barrier_cells})

    random.shuffle(simple)
    random.shuffle(barrier)
    return simple, barrier


def _apply_region_features(output_grid: np.ndarray, sector: np.ndarray, color: int, feature: dict) -> list[tuple[int, int]]:
    rr, cc = sector[:, 0], sector[:, 1]
    output_grid[rr, cc] = color
    for r, c in feature["pocket"]:
        output_grid[r, c] = BLACK
    barrier_cells = feature["barrier"]
    for r, c in barrier_cells:
        output_grid[r, c] = color
    return barrier_cells


def _make_output_candidate(region_count: int) -> tuple[np.ndarray, list[list[tuple[int, int]]], int] | None:
    h = random.choice([15, 17, 19])
    w = random.choice([15, 17, 19])
    topology = random.choice(["cross", "t"])

    divider = _build_divider(h, w, topology, region_count)
    if divider is None:
        return None

    sectors = _sector_components(divider)
    if len(sectors) != region_count:
        return None
    if min(len(sector) for sector in sectors) < 14:
        return None

    barrier_candidates: dict[int, list[dict]] = {}
    simple_candidates: dict[int, list[dict]] = {}
    for i, sector in enumerate(sectors):
        sector_mask = np.zeros((h, w), dtype=bool)
        sector_mask[sector[:, 0], sector[:, 1]] = True
        simple, barrier = _feature_options(sector_mask)
        simple_candidates[i] = simple
        barrier_candidates[i] = barrier

    output = divider.copy()
    colors = random.sample(REGION_COLORS, region_count)

    k = random.randint(2, region_count - 1)
    possible_barrier_counts = [
        b
        for b in (1, 2)
        if 2 * b <= k and sum(bool(barrier_candidates[i]) for i in range(region_count)) >= b
    ]
    random.shuffle(possible_barrier_counts)
    if not possible_barrier_counts:
        return None

    for barrier_count in possible_barrier_counts:
        simple_count = k - 2 * barrier_count
        barrier_pool = [i for i in range(region_count) if barrier_candidates[i]]
        random.shuffle(barrier_pool)
        for barrier_indices in [barrier_pool[:barrier_count]]:
            if len(barrier_indices) < barrier_count:
                continue
            simple_pool = [i for i in range(region_count) if i not in barrier_indices and simple_candidates[i]]
            random.shuffle(simple_pool)
            simple_indices = simple_pool[:simple_count]
            if len(simple_indices) < simple_count:
                continue

            barrier_groups: list[list[tuple[int, int]]] = []
            for i, sector in enumerate(sectors):
                if i in barrier_indices:
                    feature = random.choice(barrier_candidates[i])
                    barrier_groups.append(_apply_region_features(output, sector, colors[i], feature))
                elif i in simple_indices:
                    feature = random.choice(simple_candidates[i])
                    _apply_region_features(output, sector, colors[i], feature)
                else:
                    _apply_region_features(output, sector, colors[i], {"pocket": [], "barrier": []})

            if _count_black_components(output) == k:
                return output, barrier_groups, k
            output = divider.copy()
    return None


def _inverse_rotate(output_grid: np.ndarray, k: int) -> np.ndarray:
    ordered = _ordered_regions(output_grid)
    colors = [int(region["color"]) for region in ordered]
    inp = output_grid.copy()
    for i, region in enumerate(ordered):
        rr, cc = region["coords"][:, 0], region["coords"][:, 1]
        inp[rr, cc] = colors[(i + k) % len(ordered)]
    return inp


def _validate_instance(inp: np.ndarray, out: np.ndarray, barrier_groups: list[list[tuple[int, int]]], k: int) -> bool:
    if inp.shape != out.shape or max(inp.shape) > 30:
        return False
    in_regions = _ordered_regions(inp)
    out_regions = _ordered_regions(out)
    if len(in_regions) != len(out_regions):
        return False
    if len(in_regions) <= k or k <= 1:
        return False
    if _count_black_components(inp) != k or _count_black_components(out) != k:
        return False
    if np.array_equal(inp, out):
        return False
    if not np.array_equal(main(inp), out):
        return False

    out_colors = [int(region["color"]) for region in out_regions]
    if len(set(out_colors)) != len(out_colors):
        return False

    if not barrier_groups:
        return False
    muted = inp.copy()
    for cells in barrier_groups:
        for r, c in cells:
            muted[r, c] = BLACK
    if _count_black_components(muted) >= k:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        region_count = random.choice([4, 5, 6])
        built = _make_output_candidate(region_count)
        if built is None:
            continue
        output, barrier_groups, k = built
        inp = _inverse_rotate(output, k)
        if _validate_instance(inp, output, barrier_groups, k):
            return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
