# Combo variant of 800d221b (index 35)
# combo_spec: D12/2+D13/0+D20/1
# Name: 多路径色+独特种子色对+边框保持P
#
# Design analysis:
# - Concrete behavior trace:
#   The original task has one path color, shared seed colors, and seed frames that recolor with the branch.
#   Here the hub and all seed frames stay in P1, branch chains may be either P1 or P2, and each of the
#   three seeds owns a private 2-color interior pair. A P1 branch and a P2 branch can both land on the same
#   seed frame and both must recolor to that seed's majority interior color, while the P1 frame itself stays P1.
# - Complexity calibration:
#   A fully general "unique pair per seed" setup would exceed the 10-color ARC palette once seed count reaches 4,
#   so the generator fixes the count at 3 seeds. That still yields 9 visible colors
#   (bg + P1 + P2 + 3 disjoint pairs), which is enough to make the fusion obvious without becoming unreadable.
#   At least one seed is forced to receive both P1 and P2 branches so the branch color is visibly not the output key.
# - Implicit-assumption audit against the original solver:
#   `P` can no longer be inferred from a single largest network component, so the solver first scores border-color
#   candidates by "how many framed edge seeds do they explain". Seed detection can no longer depend on excluding
#   all path colors upfront, so it treats each non-candidate component as a filled edge-touching rectangle with a
#   candidate-colored frame. Branch-to-seed matching can no longer use direct seed adjacency because frames are kept;
#   it now matches each branch component to the seed frame cells it touches.
# - Placement legality:
#   Seeds are edge-anchored rectangles with non-overlapping border zones and a gap from the hub. Paths are carved on
#   an occupancy canvas that forbids overlap, side-contact with unrelated objects, and T-junctions outside hub/frame.
# - Visual observability and neatness:
#   The three seed pairs are pairwise disjoint, frames all share P1, and only branches recolor, so a human can spot:
#   (1) two input path colors collapse into seed-majority colors in output, (2) output branch colors come from the
#   touched seed interior rather than from input path color, and (3) frames remain unchanged.
# - New solver assumptions mirrored by the generator:
#   Exactly three seeds exist; each seed interior is a fully filled rectangle using exactly two private colors;
#   all non-edge frame segments are P1; branch components only touch hub/frame at their ends; the hub is the unique
#   largest solid P1 square; and the hub-center winner has a strict majority in number of hub-touching branches.

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import components_mask, get_bg, label_mask


_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _pathlike_score(mask: np.ndarray) -> float:
    H, W = mask.shape
    comps = components_mask(mask, connectivity=4)
    if not comps:
        return -1e9
    pts = np.concatenate(comps, axis=0)
    deg2 = 0
    deg4 = 0
    for r, c in pts:
        r = int(r)
        c = int(c)
        d = 0
        if r > 0 and bool(mask[r - 1, c]):
            d += 1
        if r + 1 < H and bool(mask[r + 1, c]):
            d += 1
        if c > 0 and bool(mask[r, c - 1]):
            d += 1
        if c + 1 < W and bool(mask[r, c + 1]):
            d += 1
        if d == 2:
            deg2 += 1
        elif d == 4:
            deg4 += 1
    return float(deg2 - 2 * deg4)


def _largest_solid_square(mask: np.ndarray) -> tuple[int, int, int]:
    H, W = mask.shape
    dp = np.zeros((H, W), dtype=int)
    best = (0, 0, 0)
    for r in range(H):
        for c in range(W):
            if not bool(mask[r, c]):
                continue
            if r == 0 or c == 0:
                dp[r, c] = 1
            else:
                dp[r, c] = 1 + min(dp[r - 1, c], dp[r, c - 1], dp[r - 1, c - 1])
            k = int(dp[r, c])
            if k > best[0]:
                best = (k, int(r - k + 1), int(c - k + 1))
    return best


def _frame_cells(r0: int, c0: int, r1: int, c1: int, H: int, W: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    if r0 > 0:
        cells.extend((r0 - 1, c) for c in range(c0, c1 + 1))
    if r1 < H - 1:
        cells.extend((r1 + 1, c) for c in range(c0, c1 + 1))
    if c0 > 0:
        cells.extend((r, c0 - 1) for r in range(r0, r1 + 1))
    if c1 < W - 1:
        cells.extend((r, c1 + 1) for r in range(r0, r1 + 1))
    if r0 > 0 and c0 > 0:
        cells.append((r0 - 1, c0 - 1))
    if r0 > 0 and c1 < W - 1:
        cells.append((r0 - 1, c1 + 1))
    if r1 < H - 1 and c0 > 0:
        cells.append((r1 + 1, c0 - 1))
    if r1 < H - 1 and c1 < W - 1:
        cells.append((r1 + 1, c1 + 1))
    return sorted(set(cells))


def _extract_seeds(g: np.ndarray, bg: int, border_color: int) -> list[dict[str, object]]:
    H, W = g.shape
    mask = (g != int(bg)) & (g != int(border_color))
    lab, ncomp = label_mask(mask, connectivity=4)
    seeds: list[dict[str, object]] = []
    for cid in range(1, int(ncomp) + 1):
        pts = np.argwhere(lab == int(cid))
        if pts.size == 0:
            continue
        r0 = int(pts[:, 0].min())
        r1 = int(pts[:, 0].max())
        c0 = int(pts[:, 1].min())
        c1 = int(pts[:, 1].max())
        area = int((r1 - r0 + 1) * (c1 - c0 + 1))
        if int(pts.shape[0]) != area:
            continue
        if not (r0 == 0 or c0 == 0 or r1 == H - 1 or c1 == W - 1):
            continue
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        cols = sorted({int(x) for x in np.unique(sub)})
        if len(cols) != 2 or int(bg) in cols or int(border_color) in cols:
            continue
        if r0 > 0 and not np.all(g[r0 - 1, c0 : c1 + 1] == int(border_color)):
            continue
        if r1 < H - 1 and not np.all(g[r1 + 1, c0 : c1 + 1] == int(border_color)):
            continue
        if c0 > 0 and not np.all(g[r0 : r1 + 1, c0 - 1] == int(border_color)):
            continue
        if c1 < W - 1 and not np.all(g[r0 : r1 + 1, c1 + 1] == int(border_color)):
            continue
        vals = [int(x) for x in sub.flatten()]
        maj = int(Counter(vals).most_common(1)[0][0])
        seeds.append(
            {
                "bbox": (r0, c0, r1, c1),
                "colors": tuple(cols),
                "majority": maj,
                "frame_cells": _frame_cells(r0, c0, r1, c1, H, W),
            }
        )
    seeds.sort(key=lambda s: tuple(s["bbox"]))  # type: ignore[index]
    return seeds


def _pick_border_color(g: np.ndarray, bg: int) -> tuple[int, list[dict[str, object]]]:
    colors = sorted({int(x) for x in np.unique(g)} - {int(bg)})
    assert colors
    best_score = None
    best_color = None
    best_seeds: list[dict[str, object]] = []
    for col in colors:
        seeds = _extract_seeds(g, bg, int(col))
        sq, _, _ = _largest_solid_square(g == int(col))
        score = (len(seeds), sq, _pathlike_score(g == int(col)), int((g == int(col)).sum()))
        if best_score is None or score > best_score:
            best_score = score
            best_color = int(col)
            best_seeds = seeds
    assert best_color is not None
    return int(best_color), best_seeds


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a two-color path network by the majority color of the seed it reaches.

    Input structure:
    - bg is the most common color.
    - A solid P1 hub square sits near the center; seed frames are also P1 and stay unchanged.
    - Branches are thin orthogonal chains colored either P1 or P2.
    - Exactly three edge-touching seed interiors exist, each as a filled rectangle with its own private 2-color pair.

    Rule:
    - Infer P1 as the color that best explains framed edge seeds, then recover the three seed rectangles and their majority colors.
    - Infer P2 as the only remaining non-seed, non-bg color.
    - Treat all non-frame, non-hub cells of colors P1 or P2 as branch components; recolor each component to the
      majority color of the seed frame it touches.
    - Keep every seed frame pixel unchanged and recolor only the hub center pixel to the most frequent color among
      hub-adjacent recolored branches.

    Key insight:
    - Path color is no longer the label for the output. P1 and P2 branches behave identically once you separate
      them from the fixed P1 frames; the output color is determined only by which uniquely colored seed rectangle a branch reaches.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    p1, seeds = _pick_border_color(g, bg)
    assert len(seeds) == 3, "expected exactly three framed seeds"

    used_seed_colors: set[int] = set()
    for seed in seeds:
        cols = set(int(x) for x in seed["colors"])  # type: ignore[index]
        assert len(cols) == 2
        assert not (used_seed_colors & cols), "seed color pairs must be disjoint"
        used_seed_colors |= cols

    non_bg = {int(x) for x in np.unique(g)} - {int(bg)}
    remaining = sorted(non_bg - {int(p1)} - used_seed_colors)
    assert len(remaining) == 1, "expected exactly one secondary path color"
    p2 = int(remaining[0])

    k, hr0, hc0 = _largest_solid_square(g == int(p1))
    assert k >= 3, "hub square missing"
    hub = np.zeros((H, W), dtype=bool)
    hub[hr0 : hr0 + k, hc0 : hc0 + k] = True

    frame_id_grid = np.zeros((H, W), dtype=int)
    frame_mask = np.zeros((H, W), dtype=bool)
    seed_major: dict[int, int] = {}
    for sid, seed in enumerate(seeds, start=1):
        r0, c0, r1, c1 = seed["bbox"]  # type: ignore[index]
        seed_major[int(sid)] = int(seed["majority"])  # type: ignore[index]
        for r, c in seed["frame_cells"]:  # type: ignore[index]
            assert int(g[r, c]) == int(p1), "frame must stay in P1"
            frame_mask[r, c] = True
            frame_id_grid[r, c] = int(sid)

    network_mask = (g == int(p1)) | (g == int(p2))
    branch_mask = network_mask & (~hub) & (~frame_mask)
    branch_lab, branch_n = label_mask(branch_mask, connectivity=4)

    npad = np.pad(network_mask, 1)
    degree = (
        npad[2:, 1:-1].astype(int)
        + npad[:-2, 1:-1]
        + npad[1:-1, 2:]
        + npad[1:-1, :-2]
    )
    assert not np.any((degree > 2) & branch_mask), "branch network has a T-junction"

    out = g.copy()
    for bid in range(1, int(branch_n) + 1):
        pts = np.argwhere(branch_lab == int(bid))
        if pts.size == 0:
            continue
        touches = Counter()
        hub_touch = 0
        for r, c in pts:
            r = int(r)
            c = int(c)
            for dr, dc in _DIRS:
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < H and 0 <= cc < W:
                    sid = int(frame_id_grid[rr, cc])
                    if sid:
                        touches[sid] += 1
                    if bool(hub[rr, cc]):
                        hub_touch += 1
        assert touches, "every branch must reach a seed frame"
        assert hub_touch > 0, "every branch must reach the hub"
        sid = int(touches.most_common(1)[0][0])
        out[pts[:, 0], pts[:, 1]] = int(seed_major[sid])

    touch_cols = Counter()
    for r in range(H):
        for c in range(W):
            if not bool(hub[r, c]):
                continue
            for dr, dc in _DIRS:
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < H and 0 <= cc < W and int(branch_lab[rr, cc]) != 0:
                    touch_cols[int(out[rr, cc])] += 1

    assert touch_cols, "hub must touch at least one recolored branch"
    center_r = int(hr0 + k // 2)
    center_c = int(hc0 + k // 2)
    out[center_r, center_c] = int(touch_cols.most_common(1)[0][0])
    return out


# --------------- generator ---------------

GRID_SIZE_MIN = 14
GRID_SIZE_MAX = 30
HUB_SIZE = 3

SEED_COUNT = 3
SEED_INTERIOR_MIN = 2
SEED_INTERIOR_MAX = 4
SEED_ZONE_GAP = 2
SEED_PLACEMENT_RETRIES = 500

MAX_BENDS = 6
SEGMENT_LEN_MIN = 1
SEGMENT_LEN_MAX = 6
PATH_FIRST_ROUND_RETRIES = 80
PATH_EXTRA_ROUND_RETRIES = 60
PATH_DIR_RETRIES = 3

GEN_OUTER_RETRIES = 400


def _compute_seed_zone(edge: str, ir0: int, ic0: int, ir1: int, ic1: int, H: int, W: int) -> tuple[int, int, int, int]:
    if edge == "top":
        return (ir0, max(0, ic0 - 1), ir1 + 1, min(W - 1, ic1 + 1))
    if edge == "bottom":
        return (max(0, ir0 - 1), max(0, ic0 - 1), ir1, min(W - 1, ic1 + 1))
    if edge == "left":
        return (max(0, ir0 - 1), ic0, min(H - 1, ir1 + 1), ic1 + 1)
    return (max(0, ir0 - 1), max(0, ic0 - 1), min(H - 1, ir1 + 1), ic1)


def _zones_overlap(zone: tuple[int, int, int, int], others: list[tuple[int, int, int, int]], gap: int) -> bool:
    r0, c0, r1, c1 = zone
    for or0, oc0, or1, oc1 in others:
        if r0 - gap <= or1 and or0 - gap <= r1 and c0 - gap <= oc1 and oc0 - gap <= c1:
            return True
    return False


def _pick_seed_interior(edge: str, H: int, W: int) -> tuple[int, int, int, int]:
    ih = random.randint(SEED_INTERIOR_MIN, SEED_INTERIOR_MAX)
    iw = random.randint(SEED_INTERIOR_MIN, SEED_INTERIOR_MAX)
    if edge == "top":
        r0, c0 = 0, random.randint(0, W - iw)
    elif edge == "bottom":
        r0, c0 = H - ih, random.randint(0, W - iw)
    elif edge == "left":
        r0, c0 = random.randint(0, H - ih), 0
    else:
        r0, c0 = random.randint(0, H - ih), W - iw
    return int(r0), int(c0), int(r0 + ih - 1), int(c0 + iw - 1)


def _place_seed_frame(grid: np.ndarray, occ: np.ndarray, p1: int, r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    H, W = grid.shape
    landing: list[tuple[int, int]] = []
    for cond, cells in [
        (r0 > 0, [(r0 - 1, c) for c in range(c0, c1 + 1)]),
        (r1 < H - 1, [(r1 + 1, c) for c in range(c0, c1 + 1)]),
        (c0 > 0, [(r, c0 - 1) for r in range(r0, r1 + 1)]),
        (c1 < W - 1, [(r, c1 + 1) for r in range(r0, r1 + 1)]),
    ]:
        if cond:
            for rr, cc in cells:
                grid[rr, cc] = int(p1)
                occ[rr, cc] = 1
                landing.append((int(rr), int(cc)))
    for cond, rc in [
        (r0 > 0 and c0 > 0, (r0 - 1, c0 - 1)),
        (r0 > 0 and c1 < W - 1, (r0 - 1, c1 + 1)),
        (r1 < H - 1 and c0 > 0, (r1 + 1, c0 - 1)),
        (r1 < H - 1 and c1 < W - 1, (r1 + 1, c1 + 1)),
    ]:
        if cond:
            rr, cc = rc
            grid[rr, cc] = int(p1)
            occ[rr, cc] = 1
    return landing


def _fill_seed_interior(
    grid: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    majority: int,
    minority: int,
) -> None:
    cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    n = len(cells)
    max_minor = max(1, min(int(n * 0.4), n // 2 - 1))
    min_minor = 1
    target_minor = random.randint(min_minor, max_minor)
    parity = random.randint(0, 1)
    cur_minor = [(r, c) for r, c in cells if (r + c) % 2 == parity]
    cur_major = [cell for cell in cells if cell not in cur_minor]
    if len(cur_minor) > target_minor:
        random.shuffle(cur_minor)
        cur_major.extend(cur_minor[target_minor:])
        cur_minor = cur_minor[:target_minor]
    elif len(cur_minor) < target_minor:
        need = target_minor - len(cur_minor)
        random.shuffle(cur_major)
        cur_minor.extend(cur_major[:need])
        cur_major = cur_major[need:]
    for r, c in cur_minor:
        grid[r, c] = int(minority)
    for r, c in cur_major:
        grid[r, c] = int(majority)


def _try_carve_path(
    occ: np.ndarray,
    start: tuple[int, int],
    dest_border_cells: list[tuple[int, int]],
    max_bends: int = MAX_BENDS,
) -> list[tuple[int, int]] | None:
    H, W = occ.shape
    dest_set = set(dest_border_cells)

    def _check_neighbors(r: int, c: int, prev: tuple[int, int], allow_dest: bool) -> bool:
        if not (0 <= r < H and 0 <= c < W) or int(occ[r, c]) != 0:
            return False
        for dr, dc in _DIRS:
            nr = int(r + dr)
            nc = int(c + dc)
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if (nr, nc) == prev or (nr, nc) == start:
                continue
            if int(occ[nr, nc]) != 0:
                if allow_dest and (nr, nc) in dest_set:
                    continue
                return False
        return True

    sr, sc = start
    init_dirs = []
    for dr, dc in _DIRS:
        nr = int(sr + dr)
        nc = int(sc + dc)
        if 0 <= nr < H and 0 <= nc < W and int(occ[nr, nc]) == 0:
            init_dirs.append((dr, dc))
    if not init_dirs:
        return None
    random.shuffle(init_dirs)
    bends = random.randint(0, max_bends)

    def _attempt(initial_dir: tuple[int, int]) -> list[tuple[int, int]] | None:
        path: list[tuple[int, int]] = []
        cur = start
        cur_dir = initial_dir
        for seg_i in range(bends + 1):
            is_last = seg_i == bends
            max_steps = max(H, W) if is_last else random.randint(SEGMENT_LEN_MIN, SEGMENT_LEN_MAX)
            blocked = False
            for _ in range(max_steps):
                nr = int(cur[0] + cur_dir[0])
                nc = int(cur[1] + cur_dir[1])
                if any((nr + dr, nc + dc) in dest_set for dr, dc in _DIRS) and _check_neighbors(
                    nr, nc, cur, allow_dest=True
                ):
                    occ[nr, nc] = 1
                    path.append((nr, nc))
                    return path
                if _check_neighbors(nr, nc, cur, allow_dest=False):
                    occ[nr, nc] = 1
                    path.append((nr, nc))
                    cur = (nr, nc)
                else:
                    blocked = True
                    break
            if blocked or is_last:
                for rr, cc in path:
                    occ[rr, cc] = 0
                return None
            perps = [(-cur_dir[1], cur_dir[0]), (cur_dir[1], -cur_dir[0])]
            random.shuffle(perps)
            cur_dir = perps[0]
        for rr, cc in path:
            occ[rr, cc] = 0
        return None

    for direction in init_dirs:
        for _ in range(PATH_DIR_RETRIES):
            res = _attempt(direction)
            if res is not None:
                return res
    return None


def _carve_paths(
    occ: np.ndarray,
    hub_perimeter: list[tuple[int, int]],
    seeds: list[dict[str, object]],
    target_counts: list[int],
) -> dict[int, list[list[tuple[int, int]]]] | None:
    used_starts: set[tuple[int, int]] = set()
    paths: dict[int, list[list[tuple[int, int]]]] = {i: [] for i in range(len(seeds))}

    def _add_path(si: int, retries: int) -> bool:
        for _ in range(retries):
            avail = [cell for cell in hub_perimeter if cell not in used_starts]
            if not avail:
                return False
            start = random.choice(avail)
            path = _try_carve_path(occ, start, seeds[si]["landing_cells"])  # type: ignore[index]
            if path is not None:
                used_starts.add(start)
                paths[si].append(path)
                return True
        return False

    order = list(range(len(seeds)))
    random.shuffle(order)
    for si in order:
        if not _add_path(si, PATH_FIRST_ROUND_RETRIES):
            return None

    extra_order: list[int] = []
    for si, count in enumerate(target_counts):
        extra_order.extend([si] * int(count - 1))
    random.shuffle(extra_order)
    for si in extra_order:
        if not _add_path(si, PATH_EXTRA_ROUND_RETRIES):
            return None
    return paths


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    W = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    palette = random.sample(list(range(10)), 9)
    bg = int(palette[0])
    p1 = int(palette[1])
    p2 = int(palette[2])
    seed_pairs = [(int(palette[3]), int(palette[4])), (int(palette[5]), int(palette[6])), (int(palette[7]), int(palette[8]))]

    grid = np.full((H, W), bg, dtype=int)
    occ = np.zeros((H, W), dtype=int)

    hr0 = random.randint(max(2, H // 3 - 1), min(H - 5, 2 * H // 3 - 2))
    hc0 = random.randint(max(2, W // 3 - 1), min(W - 5, 2 * W // 3 - 2))
    grid[hr0 : hr0 + HUB_SIZE, hc0 : hc0 + HUB_SIZE] = int(p1)
    occ[hr0 : hr0 + HUB_SIZE, hc0 : hc0 + HUB_SIZE] = 1

    hub_perimeter = [
        (r, c)
        for r in range(hr0, hr0 + HUB_SIZE)
        for c in range(hc0, hc0 + HUB_SIZE)
        if r in (hr0, hr0 + HUB_SIZE - 1) or c in (hc0, hc0 + HUB_SIZE - 1)
    ]

    used_zones: list[tuple[int, int, int, int]] = []
    seeds: list[dict[str, object]] = []
    chosen_edges = random.sample(["top", "bottom", "left", "right"], SEED_COUNT)

    def _overlaps_hub(zone: tuple[int, int, int, int]) -> bool:
        r0, c0, r1, c1 = zone
        return not (r1 < hr0 - 1 or r0 > hr0 + HUB_SIZE or c1 < hc0 - 1 or c0 > hc0 + HUB_SIZE)

    for edge, pair in zip(chosen_edges, seed_pairs):
        placed = False
        for _ in range(SEED_PLACEMENT_RETRIES):
            r0, c0, r1, c1 = _pick_seed_interior(edge, H, W)
            zone = _compute_seed_zone(edge, r0, c0, r1, c1, H, W)
            if _overlaps_hub(zone):
                continue
            if _zones_overlap(zone, used_zones, gap=SEED_ZONE_GAP):
                continue
            maj = random.choice(pair)
            minc = int(pair[0] if maj == pair[1] else pair[1])
            _fill_seed_interior(grid, r0, c0, r1, c1, maj, minc)
            occ[r0 : r1 + 1, c0 : c1 + 1] = 1
            landing = _place_seed_frame(grid, occ, p1, r0, c0, r1, c1)
            used_zones.append(zone)
            seeds.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "landing_cells": landing,
                    "pair": pair,
                    "majority": maj,
                }
            )
            placed = True
            break
        if not placed:
            return None

    winner = random.randrange(SEED_COUNT)
    target_options = [[2, 1, 1], [3, 1, 1], [3, 2, 1]]
    target_counts = random.choice(target_options)
    if winner != 0:
        target_counts = [target_counts[(i - winner) % SEED_COUNT] for i in range(SEED_COUNT)]
    paths = _carve_paths(occ, hub_perimeter, seeds, target_counts)
    if paths is None:
        return None

    # Force the winner seed to demonstrate both input path colors.
    winner_paths = paths[winner]
    assert len(winner_paths) >= 2
    assignments: dict[tuple[int, int], int] = {}
    first_two = winner_paths[:2]
    for idx, path in enumerate(first_two):
        assignments[tuple(path[0])] = int(p1 if idx == 0 else p2)
    for path in winner_paths[2:]:
        assignments[tuple(path[0])] = random.choice([p1, p2])
    for si, path_list in paths.items():
        if si == winner:
            continue
        for path in path_list:
            assignments[tuple(path[0])] = random.choice([p1, p2])

    used_branch_colors = set(assignments.values())
    if used_branch_colors != {int(p1), int(p2)}:
        return None

    for path_list in paths.values():
        for path in path_list:
            col = int(assignments[tuple(path[0])])
            for r, c in path:
                grid[r, c] = int(col)

    counts = Counter(int(x) for x in grid.flatten())
    if counts.most_common(1)[0][0] != int(bg):
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    # Anti-degeneracy checks: both path colors appear, one seed gets both colors, and frames stay unchanged.
    frame_preserved = 0
    mixed_seed = 0
    for seed in _extract_seeds(grid, bg, p1):
        frame_cells = set(seed["frame_cells"])  # type: ignore[arg-type]
        frame_cols = {int(grid[r, c]) for r, c in frame_cells}
        if frame_cols == {int(p1)}:
            frame_preserved += len(frame_cells)
        r0, c0, r1, c1 = seed["bbox"]  # type: ignore[index]
        local_branch_cols = set()
        for rr in range(max(0, r0 - 2), min(H, r1 + 3)):
            for cc in range(max(0, c0 - 2), min(W, c1 + 3)):
                if int(grid[rr, cc]) in (int(p1), int(p2)):
                    for dr, dc in _DIRS:
                        nr = rr + dr
                        nc = cc + dc
                        if 0 <= nr < H and 0 <= nc < W and (nr, nc) in frame_cells:
                            local_branch_cols.add(int(grid[rr, cc]))
        if local_branch_cols == {int(p1), int(p2)}:
            mixed_seed += 1
    if frame_preserved == 0 or mixed_seed == 0:
        return None

    # Ensure the output visibly differs from the input on actual branches.
    if int((out != grid).sum()) < 4:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_OUTER_RETRIES):
        res = _gen_attempt()
        if res is not None:
            return res
    raise RuntimeError(f"generate_input failed after {GEN_OUTER_RETRIES} attempts")
