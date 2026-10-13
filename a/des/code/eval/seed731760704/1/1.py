import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

DIR_COLOR_TO_DELTA = {
    MAROON: (-1, 0),
    AZURE: (0, 1),
    MAGENTA: (1, 0),
    GREEN: (0, -1),
}
SEED_COLORS = [BLUE, RED, YELLOW, ORANGE]

GRID_H_RANGE = (18, 26)
GRID_W_RANGE = (26, 36)
REGION_COUNT_RANGE = (2, 4)
GEN_RETRIES = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _mask_connected(mask: np.ndarray) -> bool:
    if not mask.any():
        return False
    return len(_components(mask)) == 1


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    r_idx = np.where(rows)[0]
    c_idx = np.where(cols)[0]
    return mask[r_idx[0]:r_idx[-1] + 1, c_idx[0]:c_idx[-1] + 1]


def _parse_legend(grid: np.ndarray) -> dict[int, tuple[int, int]]:
    top = np.asarray(grid[0], dtype=int)
    mapping: dict[int, tuple[int, int]] = {}
    c = 0
    while c < top.size:
        if int(top[c]) == BLACK:
            c += 1
            continue
        start = c
        while c < top.size and int(top[c]) != BLACK:
            c += 1
        run = [int(v) for v in top[start:c]]
        for i in range(0, len(run) - 1, 2):
            seed_color = run[i]
            dir_color = run[i + 1]
            if dir_color in DIR_COLOR_TO_DELTA:
                mapping[seed_color] = DIR_COLOR_TO_DELTA[dir_color]
    return mapping


def _enclosed_regions(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    passable = g != GREY
    regions: list[list[tuple[int, int]]] = []
    for comp in _components(passable):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        regions.append(comp)
    return regions


def _ray_distance_to_wall_grid(grid: np.ndarray, r: int, c: int, dr: int, dc: int) -> int:
    h, w = grid.shape
    dist = 0
    rr, cc = r + dr, c + dc
    while 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) != GREY:
        dist += 1
        rr += dr
        cc += dc
    return dist


def _ray_distance_to_wall_mask(mask: np.ndarray, r: int, c: int, dr: int, dc: int) -> int:
    h, w = mask.shape
    dist = 0
    rr, cc = r + dr, c + dc
    while 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]):
        dist += 1
        rr += dr
        cc += dc
    return dist


def _ray_cells(mask: np.ndarray, start: tuple[int, int], delta: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = start
    dr, dc = delta
    h, w = mask.shape
    pts: list[tuple[int, int]] = []
    rr, cc = r + dr, c + dc
    while 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]):
        pts.append((rr, cc))
        rr += dr
        cc += dc
    return pts


def _simulate_growth(
    grid: np.ndarray,
    mapping: dict[int, tuple[int, int]] | None = None,
) -> tuple[np.ndarray, list[dict[str, int | str | tuple[int, int]]]]:
    g = np.asarray(grid, dtype=int)
    if mapping is None:
        mapping = _parse_legend(g)
    out = g.copy()
    h, w = g.shape
    stats: list[dict[str, int | str | tuple[int, int]]] = []

    for region in _enclosed_regions(g):
        seeds = []
        for r, c in region:
            color = int(g[r, c])
            if color in (BLACK, GREY) or color not in mapping:
                continue
            dr, dc = mapping[color]
            dist = _ray_distance_to_wall_grid(g, r, c, dr, dc)
            seeds.append((dist, r, c, color, dr, dc))
        seeds.sort(key=lambda item: (item[0], item[1], item[2]))

        for dist, r, c, color, dr, dc in seeds:
            filled = 0
            stop = "wall"
            rr, cc = r + dr, c + dc
            while 0 <= rr < h and 0 <= cc < w:
                cell = int(out[rr, cc])
                if cell != BLACK:
                    stop = "wall" if cell == GREY else "blocked"
                    break
                out[rr, cc] = color
                filled += 1
                rr += dr
                cc += dc
            stats.append(
                {
                    "seed": (r, c),
                    "color": color,
                    "dist": dist,
                    "filled": filled,
                    "stop": stop,
                }
            )
    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow colored seeds inside grey-walled regions according to a top-row legend.
    Seeds are processed by increasing distance to their target wall, then row-major.
    """
    out, _ = _simulate_growth(grid)
    return out


def _build_region_patch(mask: np.ndarray, seeds: dict[int, tuple[int, int]]) -> np.ndarray:
    h, w = mask.shape
    inner = np.zeros((h + 2, w + 2), dtype=bool)
    inner[1:-1, 1:-1] = mask
    patch = np.zeros((h + 2, w + 2), dtype=int)

    for r, c in np.argwhere(inner):
        for nr, nc in _neighbors4(int(r), int(c)):
            if 0 <= nr < h + 2 and 0 <= nc < w + 2 and not inner[nr, nc]:
                patch[nr, nc] = GREY

    for color, (r, c) in seeds.items():
        patch[r + 1, c + 1] = color
    return patch


def _make_notched_shape(rng: random.Random) -> np.ndarray | None:
    for _ in range(80):
        h = rng.randint(4, 8)
        w = rng.randint(5, 10)
        mask = np.ones((h, w), dtype=bool)
        changed = False
        for _ in range(rng.randint(1, 3)):
            side = rng.choice(["top", "bottom", "left", "right"])
            cand = mask.copy()
            if side in ("top", "bottom"):
                if cand.shape[0] <= 3:
                    continue
                max_depth = min(3, cand.shape[0] - 2)
                depth = rng.randint(1, max_depth)
                max_span = cand.shape[1] - 1
                span = rng.randint(1, max_span)
                start = rng.randint(0, cand.shape[1] - span)
                if side == "top":
                    cand[:depth, start:start + span] = False
                else:
                    cand[cand.shape[0] - depth:, start:start + span] = False
            else:
                if cand.shape[1] <= 3:
                    continue
                max_depth = min(3, cand.shape[1] - 2)
                depth = rng.randint(1, max_depth)
                max_span = cand.shape[0] - 1
                span = rng.randint(1, max_span)
                start = rng.randint(0, cand.shape[0] - span)
                if side == "left":
                    cand[start:start + span, :depth] = False
                else:
                    cand[start:start + span, cand.shape[1] - depth:] = False
            if not cand.any():
                continue
            cand = _trim_mask(cand)
            if cand.sum() < 10 or not _mask_connected(cand):
                continue
            mask = cand
            changed = True

        mask = _trim_mask(mask)
        if not changed:
            continue
        if mask.shape[0] < 4 or mask.shape[1] < 4 or mask.sum() < 10:
            continue
        if not (~mask).any():
            continue
        return mask
    return None


def _place_free_seeds(
    mask: np.ndarray,
    colors: list[int],
    mapping: dict[int, tuple[int, int]],
    rng: random.Random,
    used_positions: set[tuple[int, int]] | None = None,
    forbidden_positions: set[tuple[int, int]] | None = None,
) -> dict[int, tuple[int, int]] | None:
    if not colors:
        return {}
    used = set() if used_positions is None else set(used_positions)
    forbidden = set() if forbidden_positions is None else set(forbidden_positions)
    cells = [(int(r), int(c)) for r, c in np.argwhere(mask)]

    for _ in range(200):
        placements: dict[int, tuple[int, int]] = {}
        local_used = set(used)
        order = list(colors)
        rng.shuffle(order)
        ok = True
        for color in order:
            dr, dc = mapping[color]
            min_dist = 2 if rng.random() < 0.55 else 1
            candidates = [
                p
                for p in cells
                if p not in local_used
                and p not in forbidden
                and _ray_distance_to_wall_mask(mask, p[0], p[1], dr, dc) >= min_dist
            ]
            if not candidates:
                ok = False
                break
            pos = rng.choice(candidates)
            placements[color] = pos
            local_used.add(pos)
        if ok:
            return placements
    return None


def _distance_range(length: int, intersection_idx: int, positive_direction: bool) -> tuple[int, int]:
    if positive_direction:
        return length - intersection_idx, length - 1
    return intersection_idx + 1, length - 1


def _coord_from_distance(length: int, dist: int, positive_direction: bool) -> int:
    if positive_direction:
        return length - 1 - dist
    return dist


def _pick_conflict_positions(
    h: int,
    w: int,
    hdir: tuple[int, int],
    vdir: tuple[int, int],
    mode: str,
    rng: random.Random,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None:
    h_positive = hdir == (0, 1)
    v_positive = vdir == (1, 0)

    for _ in range(120):
        ir = rng.randint(2, h - 3)
        ic = rng.randint(2, w - 3)
        h_lo, h_hi = _distance_range(w, ic, h_positive)
        v_lo, v_hi = _distance_range(h, ir, v_positive)
        if mode == "tie":
            lo = max(h_lo, v_lo)
            hi = min(h_hi, v_hi)
            if lo > hi:
                continue
            dist_h = dist_v = rng.randint(lo, hi)
        else:
            if rng.random() < 0.5:
                near_h = range(h_lo, min(h_hi, h_lo + 1) + 1)
                far_v = range(max(v_lo, v_hi - 1), v_hi + 1)
                h_choices = list(near_h)
                v_choices = [d for d in far_v if d not in h_choices]
            else:
                far_h = range(max(h_lo, h_hi - 1), h_hi + 1)
                near_v = range(v_lo, min(v_hi, v_lo + 1) + 1)
                h_choices = [d for d in far_h if d not in near_v]
                v_choices = list(near_v)
            if not h_choices or not v_choices:
                continue
            dist_h = rng.choice(h_choices)
            dist_v = rng.choice(v_choices)
            if dist_h == dist_v:
                continue

        hc = _coord_from_distance(w, dist_h, h_positive)
        vr = _coord_from_distance(h, dist_v, v_positive)
        h_seed = (ir, hc)
        v_seed = (vr, ic)
        if h_seed == v_seed:
            continue
        return (ir, ic), h_seed, v_seed
    return None


def _add_safe_corner_notch(mask: np.ndarray, intersection: tuple[int, int], rng: random.Random) -> np.ndarray:
    h, w = mask.shape
    ir, ic = intersection
    safe_corners = []
    if ir >= 2 and ic >= 2:
        safe_corners.append(("tl", min(2, ir - 1), min(2, ic - 1)))
    if ir >= 2 and w - 1 - ic >= 1:
        safe_corners.append(("tr", min(2, ir - 1), min(2, w - 1 - ic)))
    if h - 1 - ir >= 1 and ic >= 2:
        safe_corners.append(("bl", min(2, h - 1 - ir), min(2, ic - 1)))
    if h - 1 - ir >= 1 and w - 1 - ic >= 1:
        safe_corners.append(("br", min(2, h - 1 - ir), min(2, w - 1 - ic)))
    if not safe_corners or rng.random() >= 0.7:
        return mask

    corner, max_h, max_w = rng.choice(safe_corners)
    notch_h = rng.randint(1, max_h)
    notch_w = rng.randint(1, max_w)
    out = mask.copy()
    if corner == "tl":
        out[:notch_h, :notch_w] = False
    elif corner == "tr":
        out[:notch_h, w - notch_w:] = False
    elif corner == "bl":
        out[h - notch_h:, :notch_w] = False
    else:
        out[h - notch_h:, w - notch_w:] = False
    return out


def _make_conflict_region(
    colors: list[int],
    mapping: dict[int, tuple[int, int]],
    rng: random.Random,
) -> tuple[np.ndarray, str] | None:
    h_color, v_color = colors[0], colors[1]
    hdir = mapping[h_color]
    vdir = mapping[v_color]
    assert hdir[0] == 0 and vdir[1] == 0

    for _ in range(100):
        h = rng.randint(6, 8)
        w = rng.randint(7, 10)
        mode = rng.choice(["distance", "tie"])
        picked = _pick_conflict_positions(h, w, hdir, vdir, mode, rng)
        if picked is None:
            continue
        intersection, h_seed, v_seed = picked

        mask = np.ones((h, w), dtype=bool)
        mask = _add_safe_corner_notch(mask, intersection, rng)

        seeds = {h_color: h_seed, v_color: v_seed}
        used = {h_seed, v_seed}
        forbid = set(used)
        forbid.update(_ray_cells(mask, h_seed, hdir))
        forbid.update(_ray_cells(mask, v_seed, vdir))

        extra_colors = list(colors[2:])
        if extra_colors:
            extras = _place_free_seeds(mask, extra_colors, mapping, rng, used_positions=used, forbidden_positions=forbid)
            if extras is None:
                continue
            seeds.update(extras)

        patch = _build_region_patch(mask, seeds)
        _, stats = _simulate_growth(patch, mapping)
        by_color = {int(item["color"]): item for item in stats}
        if h_color not in by_color or v_color not in by_color:
            continue
        if by_color[h_color]["stop"] != "blocked" and by_color[v_color]["stop"] != "blocked":
            continue
        if mode == "tie" and int(by_color[h_color]["dist"]) != int(by_color[v_color]["dist"]):
            continue
        if mode == "distance" and int(by_color[h_color]["dist"]) == int(by_color[v_color]["dist"]):
            continue
        return patch, mode
    return None


def _make_regular_region(
    colors: list[int],
    mapping: dict[int, tuple[int, int]],
    rng: random.Random,
) -> np.ndarray | None:
    for _ in range(100):
        mask = _make_notched_shape(rng)
        if mask is None:
            continue
        seeds = _place_free_seeds(mask, colors, mapping, rng)
        if seeds is None:
            continue
        patch = _build_region_patch(mask, seeds)
        _, stats = _simulate_growth(patch, mapping)
        if any(int(item["filled"]) > 0 for item in stats):
            return patch
    return None


def _paint_legend(
    grid: np.ndarray,
    legend_pairs: list[tuple[int, int]],
    rng: random.Random,
) -> None:
    pairs = list(legend_pairs)
    rng.shuffle(pairs)
    gaps = [rng.randint(1, 2) for _ in range(max(0, len(pairs) - 1))]
    width = 2 * len(pairs) + sum(gaps)
    start = rng.randint(0, grid.shape[1] - width)
    c = start
    for i, (seed_color, dir_color) in enumerate(pairs):
        grid[0, c] = seed_color
        grid[0, c + 1] = dir_color
        c += 2
        if i < len(gaps):
            c += gaps[i]


def _place_patches(grid: np.ndarray, patches: list[np.ndarray], rng: random.Random) -> bool:
    h, w = grid.shape
    occupied = np.zeros((h, w), dtype=bool)
    occupied[0, :] = True

    for patch in sorted(patches, key=lambda arr: arr.shape[0] * arr.shape[1], reverse=True):
        ph, pw = patch.shape
        if ph + 3 > h or pw + 2 > w:
            return False
        rows = list(range(2, h - ph))
        cols = list(range(1, w - pw))
        rng.shuffle(rows)
        placed = False
        for r0 in rows:
            cols_local = list(cols)
            rng.shuffle(cols_local)
            for c0 in cols_local:
                r1 = max(0, r0 - 1)
                r2 = min(h, r0 + ph + 1)
                c1 = max(0, c0 - 1)
                c2 = min(w, c0 + pw + 1)
                if occupied[r1:r2, c1:c2].any():
                    continue
                grid[r0:r0 + ph, c0:c0 + pw] = patch
                occupied[r0:r0 + ph, c0:c0 + pw] = True
                placed = True
                break
            if placed:
                break
        if not placed:
            return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_regions = rng.randint(*REGION_COUNT_RANGE)

    direction_colors = list(DIR_COLOR_TO_DELTA.keys())
    rng.shuffle(direction_colors)
    legend_pairs = list(zip(SEED_COLORS, direction_colors))
    delta_mapping = {seed: DIR_COLOR_TO_DELTA[dir_color] for seed, dir_color in legend_pairs}

    horizontal_colors = [color for color, delta in delta_mapping.items() if delta[0] == 0]
    vertical_colors = [color for color, delta in delta_mapping.items() if delta[1] == 0]
    h_color = rng.choice(horizontal_colors)
    v_color = rng.choice(vertical_colors)

    conflict_idx = rng.randrange(n_regions)
    region_colors: list[list[int]] = [[] for _ in range(n_regions)]
    region_colors[conflict_idx] = [h_color, v_color]

    other_idxs = [idx for idx in range(n_regions) if idx != conflict_idx]
    rng.shuffle(other_idxs)
    for i, color in enumerate([c for c in SEED_COLORS if c not in (h_color, v_color)]):
        target = other_idxs[i % len(other_idxs)] if other_idxs else conflict_idx
        region_colors[target].append(color)

    for idx in other_idxs:
        if not region_colors[idx]:
            region_colors[idx].append(rng.choice(SEED_COLORS))

    if len(region_colors[conflict_idx]) < 3 and rng.random() < 0.35:
        extras = [c for c in SEED_COLORS if c not in region_colors[conflict_idx]]
        if extras:
            region_colors[conflict_idx].append(rng.choice(extras))

    for idx in range(n_regions):
        if idx == conflict_idx:
            continue
        if len(region_colors[idx]) < 2 and rng.random() < 0.45:
            extras = [c for c in SEED_COLORS if c not in region_colors[idx]]
            if extras:
                region_colors[idx].append(rng.choice(extras))

    conflict_built = _make_conflict_region(region_colors[conflict_idx], delta_mapping, rng)
    if conflict_built is None:
        return None
    patches = [conflict_built[0]]

    for idx in range(n_regions):
        if idx == conflict_idx:
            continue
        patch = _make_regular_region(region_colors[idx], delta_mapping, rng)
        if patch is None:
            return None
        patches.append(patch)

    min_w = max(GRID_W_RANGE[0], 2 * len(legend_pairs) + 2 * (len(legend_pairs) - 1) + 2)
    grid_h = rng.randint(*GRID_H_RANGE)
    grid_w = rng.randint(min_w, GRID_W_RANGE[1])
    if n_regions >= 4:
        grid_h = max(grid_h, 22)
        grid_w = max(grid_w, 30)
    grid = np.zeros((grid_h, grid_w), dtype=int)

    if not _place_patches(grid, patches, rng):
        return None
    _paint_legend(grid, legend_pairs, rng)

    parsed = _parse_legend(grid)
    if parsed != delta_mapping:
        return None

    regions = _enclosed_regions(grid)
    if len(regions) != n_regions:
        return None

    _, stats = _simulate_growth(grid, parsed)
    if len(stats) < 4:
        return None
    if set(int(item["color"]) for item in stats) != set(SEED_COLORS):
        return None
    if not any(item["stop"] == "blocked" for item in stats):
        return None

    out, _ = _simulate_growth(grid, parsed)
    if np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exceeded retry budget")
