import random
from collections import Counter

import numpy as np


SIDES = ("up", "right", "down", "left")
SIDE_TO_DELTA = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
L_PATTERNS = {
    "l_tl": {(0, 1), (1, 0), (1, 1)},  # missing top-left -> opens to up/left
    "l_tr": {(0, 0), (1, 0), (1, 1)},  # missing top-right -> opens to up/right
    "l_bl": {(0, 0), (0, 1), (1, 1)},  # missing bottom-left -> opens to down/left
    "l_br": {(0, 0), (0, 1), (1, 0)},  # missing bottom-right -> opens to down/right
}
L_SIDE_CANDIDATES = {
    "l_tl": {"up", "left"},
    "l_tr": {"up", "right"},
    "l_bl": {"down", "left"},
    "l_br": {"down", "right"},
}

GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
OBJECT_COUNT_RANGE = (2, 4)
BOX_H_RANGE = (7, 10)
BOX_W_RANGE = (7, 10)
PLACEMENT_RETRIES = 200
GENERATION_RETRIES = 300


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _prefix_sum(mask: np.ndarray) -> np.ndarray:
    pref = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), dtype=int)
    pref[1:, 1:] = np.cumsum(np.cumsum(mask.astype(int), axis=0), axis=1)
    return pref


def _rect_sum(pref: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    if r0 > r1 or c0 > c1:
        return 0
    return int(pref[r1 + 1, c1 + 1] - pref[r0, c1 + 1] - pref[r1 + 1, c0] + pref[r0, c0])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "area": len(coords),
                }
            )
    return comps


def _seed_side_candidates(coords: list[tuple[int, int]]) -> set[str] | None:
    area = len(coords)
    if area not in (2, 3):
        return None
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, c0, r1, c1 = min(rows), min(cols), max(rows), max(cols)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    norm = {(r - r0, c - c0) for r, c in coords}
    if h == 1:
        return {"left", "right"}
    if w == 1:
        return {"up", "down"}
    if area == 3 and h == 2 and w == 2:
        for key, pattern in L_PATTERNS.items():
            if norm == pattern:
                return set(L_SIDE_CANDIDATES[key])
    return None


def _boundary_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells = [(r0, c) for c in range(c0, c1 + 1)]
    cells.extend((r, c1) for r in range(r0 + 1, r1))
    cells.extend((r1, c) for c in range(c1, c0 - 1, -1))
    cells.extend((r, c0) for r in range(r1 - 1, r0, -1))
    return cells


def _gap_side(r0: int, c0: int, r1: int, c1: int, gap: tuple[int, int]) -> str | None:
    r, c = gap
    if (r in (r0, r1)) and (c in (c0, c1)):
        return None
    if r == r0:
        return "up"
    if r == r1:
        return "down"
    if c == c0:
        return "left"
    if c == c1:
        return "right"
    return None


def _find_region_for_seed(
    grid: np.ndarray, bg: int, comp: dict, pref_nonbg: np.ndarray
) -> dict | None:
    h, w = grid.shape
    seed_color = int(comp["color"])
    sr0, sc0, sr1, sc1 = comp["bbox"]
    seed_area = int(comp["area"])
    best: dict | None = None
    best_area: int | None = None

    for r0 in range(sr0 - 1, -1, -1):
        for r1 in range(sr1 + 1, h):
            if r1 - r0 + 1 < 4:
                continue
            inner_nonbg_rows_ok = _rect_sum(pref_nonbg, r0 + 1, 0, r1 - 1, w - 1) >= seed_area
            if not inner_nonbg_rows_ok:
                continue
            for c0 in range(sc0 - 1, -1, -1):
                for c1 in range(sc1 + 1, w):
                    if c1 - c0 + 1 < 4:
                        continue
                    if _rect_sum(pref_nonbg, r0 + 1, c0 + 1, r1 - 1, c1 - 1) != seed_area:
                        continue

                    boundary = _boundary_cells(r0, c0, r1, c1)
                    gaps = [(r, c) for r, c in boundary if int(grid[r, c]) == bg]
                    if len(gaps) != 2:
                        continue

                    gap_sides: list[str] = []
                    for gap in gaps:
                        side = _gap_side(r0, c0, r1, c1, gap)
                        if side is None:
                            gap_sides = []
                            break
                        gap_sides.append(side)
                    if len(gap_sides) != 2 or gap_sides[0] == gap_sides[1]:
                        continue

                    nonbg_boundary = {int(grid[r, c]) for r, c in boundary if int(grid[r, c]) != bg}
                    if len(nonbg_boundary) != 1:
                        continue
                    border_color = int(next(iter(nonbg_boundary)))
                    if border_color == seed_color:
                        continue

                    area = (r1 - r0 + 1) * (c1 - c0 + 1)
                    if best_area is None or area < best_area:
                        best_area = area
                        best = {
                            "bbox": (r0, c0, r1, c1),
                            "gaps": gaps,
                            "gap_sides": tuple(gap_sides),
                            "border_color": border_color,
                        }
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """
    Choose the matching border gap from each seed cluster and shoot an outward ray through it.

    Input structure:
    - The most common color is the background.
    - The grid contains one or more one-pixel-thick rectangular frames, each drawn in a single border color.
    - Every frame boundary has exactly two one-pixel gaps on different sides, and its interior contains exactly one small seed cluster of another color.

    Rule:
    - Infer the seed cluster's directional cue: straight clusters indicate a horizontal or vertical side-family, while L clusters indicate the two sides adjacent to their opening.
    - Among the frame's two gaps, the unique gap compatible with that cue is active and the other gap is silent.
    - Fill the silent gap with the frame color, keep the active gap open, and paint a ray in the seed color from the cell just outside the active gap straight outward through background cells until a non-background cell or the grid edge stops it.

    Key insight:
    - The seed cluster does not directly paint its own direction inside the box; it only selects which of the two boundary gaps stays open, and that chosen gap determines the ray direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _most_common_color(g)
    pref_nonbg = _prefix_sum(g != bg)
    comps = _find_components(g, bg)

    regions: list[dict] = []
    seen_boxes: set[tuple[int, int, int, int]] = set()
    for comp in comps:
        side_candidates = _seed_side_candidates(comp["coords"])
        if side_candidates is None:
            continue
        region = _find_region_for_seed(g, bg, comp, pref_nonbg)
        if region is None:
            continue
        bbox = tuple(region["bbox"])
        if bbox in seen_boxes:
            continue

        matches = [i for i, side in enumerate(region["gap_sides"]) if side in side_candidates]
        if len(matches) != 1:
            continue
        active_i = matches[0]
        silent_i = 1 - active_i
        region["seed_color"] = int(comp["color"])
        region["active_gap"] = tuple(region["gaps"][active_i])
        region["silent_gap"] = tuple(region["gaps"][silent_i])
        region["active_side"] = str(region["gap_sides"][active_i])
        regions.append(region)
        seen_boxes.add(bbox)

    assert regions, "expected at least one valid region"

    out = g.copy()
    ray_paints: list[tuple[int, list[tuple[int, int]]]] = []
    h, w = g.shape
    for region in regions:
        sr, sc = region["silent_gap"]
        out[sr, sc] = int(region["border_color"])

        ar, ac = region["active_gap"]
        dr, dc = SIDE_TO_DELTA[region["active_side"]]
        rr, cc = ar + dr, ac + dc
        ray: list[tuple[int, int]] = []
        while 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == bg:
            ray.append((rr, cc))
            rr += dr
            cc += dc
        ray_paints.append((int(region["seed_color"]), ray))

    for color, ray in ray_paints:
        for r, c in ray:
            out[r, c] = color
    return out


def _sample_line_seed(
    rng: random.Random, box_h: int, box_w: int, family: str, active_side: str
) -> list[tuple[int, int]]:
    ih = box_h - 2
    iw = box_w - 2
    length = rng.choice((2, 3))
    if family == "line_h":
        row = 1 + rng.randrange(ih)
        max_start = iw - length
        if active_side == "left":
            lo = 0
            hi = max(0, max_start // 2)
        else:
            lo = min(max_start, (max_start + 1) // 2)
            hi = max_start
        start = rng.randint(lo, hi)
        return [(row, 1 + start + k) for k in range(length)]

    col = 1 + rng.randrange(iw)
    max_start = ih - length
    if active_side == "up":
        lo = 0
        hi = max(0, max_start // 2)
    else:
        lo = min(max_start, (max_start + 1) // 2)
        hi = max_start
    start = rng.randint(lo, hi)
    return [(1 + start + k, col) for k in range(length)]


def _sample_l_seed(
    rng: random.Random, box_h: int, box_w: int, family: str
) -> list[tuple[int, int]]:
    ih = box_h - 2
    iw = box_w - 2
    max_r = ih - 2
    max_c = iw - 2
    if family == "l_tl":
        r0 = rng.randint(0, max(0, max_r // 2))
        c0 = rng.randint(0, max(0, max_c // 2))
    elif family == "l_tr":
        r0 = rng.randint(0, max(0, max_r // 2))
        c0 = rng.randint(min(max_c, (max_c + 1) // 2), max_c)
    elif family == "l_bl":
        r0 = rng.randint(min(max_r, (max_r + 1) // 2), max_r)
        c0 = rng.randint(0, max(0, max_c // 2))
    else:
        r0 = rng.randint(min(max_r, (max_r + 1) // 2), max_r)
        c0 = rng.randint(min(max_c, (max_c + 1) // 2), max_c)
    return [(1 + r0 + dr, 1 + c0 + dc) for dr, dc in sorted(L_PATTERNS[family])]


def _sample_seed(
    rng: random.Random, box_h: int, box_w: int
) -> tuple[str, set[str], str, str, list[tuple[int, int]]]:
    if rng.random() < 0.5:
        family = rng.choice(("line_h", "line_v"))
        side_candidates = {"left", "right"} if family == "line_h" else {"up", "down"}
        active_side = rng.choice(sorted(side_candidates))
        silent_side = rng.choice(sorted(set(SIDES) - side_candidates))
        coords = _sample_line_seed(rng, box_h, box_w, family, active_side)
    else:
        family = rng.choice(tuple(L_PATTERNS))
        side_candidates = set(L_SIDE_CANDIDATES[family])
        active_side = rng.choice(sorted(side_candidates))
        silent_side = rng.choice(sorted(set(SIDES) - side_candidates))
        coords = _sample_l_seed(rng, box_h, box_w, family)
    return family, side_candidates, active_side, silent_side, coords


def _safe_gap_index(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def _choose_gaps(
    seed_coords: list[tuple[int, int]], box_h: int, box_w: int, active_side: str, silent_side: str
) -> tuple[tuple[int, int], tuple[int, int]]:
    seed_rows = [r for r, _ in seed_coords]
    seed_cols = [c for _, c in seed_coords]
    row_mid = int(round(sum(seed_rows) / len(seed_rows)))
    col_mid = int(round(sum(seed_cols) / len(seed_coords)))

    def side_gap(side: str, anchor: int) -> tuple[int, int]:
        if side == "up":
            return (0, _safe_gap_index(anchor, 2, box_w - 3))
        if side == "down":
            return (box_h - 1, _safe_gap_index(anchor, 2, box_w - 3))
        if side == "left":
            return (_safe_gap_index(anchor, 2, box_h - 3), 0)
        return (_safe_gap_index(anchor, 2, box_h - 3), box_w - 1)

    active_anchor = col_mid if active_side in ("up", "down") else row_mid
    silent_anchor = row_mid if silent_side in ("left", "right") else col_mid
    return side_gap(active_side, active_anchor), side_gap(silent_side, silent_anchor)


def _ray_cells(
    grid_h: int, grid_w: int, r0: int, c0: int, active_gap: tuple[int, int], active_side: str
) -> list[tuple[int, int]]:
    gap_r, gap_c = active_gap
    dr, dc = SIDE_TO_DELTA[active_side]
    rr, cc = r0 + gap_r + dr, c0 + gap_c + dc
    cells: list[tuple[int, int]] = []
    while 0 <= rr < grid_h and 0 <= cc < grid_w:
        cells.append((rr, cc))
        rr += dr
        cc += dc
    return cells


def _draw_object(
    grid: np.ndarray,
    r0: int,
    c0: int,
    box_h: int,
    box_w: int,
    border_color: int,
    seed_color: int,
    seed_coords: list[tuple[int, int]],
    active_gap: tuple[int, int],
    silent_gap: tuple[int, int],
) -> None:
    r1 = r0 + box_h - 1
    c1 = c0 + box_w - 1
    grid[r0, c0 : c1 + 1] = border_color
    grid[r1, c0 : c1 + 1] = border_color
    grid[r0 : r1 + 1, c0] = border_color
    grid[r0 : r1 + 1, c1] = border_color
    for gr, gc in (active_gap, silent_gap):
        grid[r0 + gr, c0 + gc] = grid[0, 0]
    for sr, sc in seed_coords:
        grid[r0 + sr, c0 + sc] = seed_color


def _padded_bbox(r0: int, c0: int, r1: int, c1: int, h: int, w: int) -> tuple[int, int, int, int]:
    return max(0, r0 - 1), max(0, c0 - 1), min(h - 1, r1 + 1), min(w - 1, c1 + 1)


def _clear(mask: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    return not bool(mask[r0 : r1 + 1, c0 : c1 + 1].any())


def _mark(mask: np.ndarray, box: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = box
    mask[r0 : r1 + 1, c0 : c1 + 1] = True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid_h = rng.randint(*GRID_H_RANGE)
    grid_w = rng.randint(*GRID_W_RANGE)
    bg = rng.randrange(10)
    grid = np.full((grid_h, grid_w), bg, dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    reserved_rays = np.zeros((grid_h, grid_w), dtype=bool)

    n_objects = rng.randint(*OBJECT_COUNT_RANGE)
    palette = [c for c in range(10) if c != bg]
    if len(palette) < 2 * n_objects:
        return None
    rng.shuffle(palette)
    color_pairs = [(palette[2 * i], palette[2 * i + 1]) for i in range(n_objects)]

    placed = 0
    for border_color, seed_color in color_pairs:
        success = False
        for _ in range(PLACEMENT_RETRIES):
            box_h = rng.randint(*BOX_H_RANGE)
            box_w = rng.randint(*BOX_W_RANGE)
            family, _side_candidates, active_side, silent_side, seed_coords = _sample_seed(rng, box_h, box_w)
            active_gap, silent_gap = _choose_gaps(seed_coords, box_h, box_w, active_side, silent_side)

            min_r0 = 1 if active_side != "up" else 2
            max_r0 = grid_h - box_h - (1 if active_side != "down" else 2)
            min_c0 = 1 if active_side != "left" else 2
            max_c0 = grid_w - box_w - (1 if active_side != "right" else 2)
            if min_r0 > max_r0 or min_c0 > max_c0:
                continue

            r0 = rng.randint(min_r0, max_r0)
            c0 = rng.randint(min_c0, max_c0)
            r1 = r0 + box_h - 1
            c1 = c0 + box_w - 1
            pad_box = _padded_bbox(r0, c0, r1, c1, grid_h, grid_w)
            if not _clear(occupied, pad_box):
                continue
            if not _clear(reserved_rays, pad_box):
                continue

            ray = _ray_cells(grid_h, grid_w, r0, c0, active_gap, active_side)
            if len(ray) < 2:
                continue
            if any(occupied[rr, cc] or reserved_rays[rr, cc] for rr, cc in ray):
                continue

            _draw_object(
                grid,
                r0,
                c0,
                box_h,
                box_w,
                border_color,
                seed_color,
                seed_coords,
                active_gap,
                silent_gap,
            )
            _mark(occupied, pad_box)
            for rr, cc in ray:
                reserved_rays[rr, cc] = True
            placed += 1
            success = True
            break
        if not success:
            return None

    if placed != n_objects:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        result = _gen_attempt(random.Random())
        if result is not None:
            return result
    raise RuntimeError(f"failed to generate a valid sample after {GENERATION_RETRIES} attempts")
