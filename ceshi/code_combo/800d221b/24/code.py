# Combo variant of 800d221b (index 24)
# combo_spec: D10/1+D14/2
# Name: 弯折编码+条件删除

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import components_mask, get_bg, label_mask


# Design analysis for the combo:
# - Concrete behavior: in the base task, each non-hub P component is recolored by the majority
#   color of the seed interior it reaches, and the hub center copies the dominant adjacent branch
#   color. Here the same component must first inspect only its thin trunk, count L-bends on that
#   trunk, recolor to majority for even bends or minority for odd bends, and then delete the full
#   component if it took the minority color. Example: a 2-bend top branch survives and becomes the
#   seed majority; a 3-bend right branch would first take the seed minority and then vanish to bg.
# - Complexity calibration: the generator fixes one seed per edge and one branch per seed so every
#   component is visually separable, but still keeps mixed seed interiors, varied bend counts, and
#   both surviving and deleted cases. Samples always contain both bend parities and at least one
#   deletion, so the fusion does not collapse back to the original task.
# - Implicit assumptions audited against the base solver:
#   1. A branch component still contains both the path trunk and the seed's P border.
#   2. Bend counting must ignore that border, because its rectangle corners would fake extra bends.
#   3. Each branch must correspond to exactly one seed border; the generator enforces one path per
#      seed and non-overlap so this remains true.
#   4. After deleting odd branches, hub-center voting must happen on the surviving recolored output,
#      not on the original branch labels.
#   5. The trunk must be a simple chain after removing hub and seed-border pixels; the path carver
#      forbids touching existing P except at the intended seed border and hub start.
# - Placement legality: seeds sit on distinct outer edges, border zones do not overlap the hub or
#   each other, paths start from distinct hub-side midpoints, and the carver checks bounds plus
#   neighbor conflicts to prevent overlap and T-junctions.
# - Visual observability and cleanliness: only two seed colors are used, every sample has both
#   surviving even-bend branches and deleted odd-bend branches, and the hub center has a unique
#   winner among survivors. This keeps the output readable while making bend-parity the only stable
#   explanation for which branches remain.


# ── Shared geometry helpers ──────────────────────────────────────────────────

_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
HUB_SIZE = 3


def _find_path_color(g: np.ndarray, bg: int) -> int:
    H, W = g.shape
    colors = sorted({int(x) for x in np.unique(g)} - {int(bg)})
    assert colors

    def comp_deg_score(mask: np.ndarray) -> float:
        comps = components_mask(mask, connectivity=4)
        if not comps:
            return -1e9
        best_pts = max(comps, key=lambda coords: int(coords.shape[0]))
        best_sz = int(best_pts.shape[0])

        m = np.zeros(mask.shape, dtype=bool)
        m[best_pts[:, 0], best_pts[:, 1]] = True

        deg2 = 0
        deg4 = 0
        for r, c in best_pts:
            r = int(r)
            c = int(c)
            d = 0
            if r > 0 and bool(m[r - 1, c]):
                d += 1
            if r + 1 < H and bool(m[r + 1, c]):
                d += 1
            if c > 0 and bool(m[r, c - 1]):
                d += 1
            if c + 1 < W and bool(m[r, c + 1]):
                d += 1
            if d == 2:
                deg2 += 1
            elif d == 4:
                deg4 += 1
        frac2 = deg2 / max(1, best_sz)
        frac4 = deg4 / max(1, best_sz)
        return float((frac2 - frac4) * best_sz)

    return int(max(colors, key=lambda c: comp_deg_score(g == int(c))))


def _find_hub(g: np.ndarray, path_color: int) -> tuple[np.ndarray, tuple[int, int, int]]:
    H, W = g.shape
    mP = g == int(path_color)
    ps = mP.astype(int).cumsum(axis=0).cumsum(axis=1)

    def rect_sum(r0: int, c0: int, r1: int, c1: int) -> int:
        s = int(ps[r1, c1])
        if r0 > 0:
            s -= int(ps[r0 - 1, c1])
        if c0 > 0:
            s -= int(ps[r1, c0 - 1])
        if r0 > 0 and c0 > 0:
            s += int(ps[r0 - 1, c0 - 1])
        return int(s)

    best = None
    for k in range(2, int(min(H, W)) + 1):
        found = None
        for r0 in range(0, int(H - k + 1)):
            r1 = int(r0 + k - 1)
            for c0 in range(0, int(W - k + 1)):
                c1 = int(c0 + k - 1)
                if rect_sum(int(r0), int(c0), int(r1), int(c1)) == int(k * k):
                    found = (int(k), int(r0), int(c0))
        if found is not None:
            best = found

    assert best is not None
    k, r0, c0 = map(int, best)
    hub = np.zeros((H, W), dtype=bool)
    hub[r0 : r0 + k, c0 : c0 + k] = True
    return hub, (int(k), int(r0), int(c0))


def _seed_border_cells(r0: int, c0: int, r1: int, c1: int, H: int, W: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    if r0 > 0:
        for c in range(c0, c1 + 1):
            cells.add((int(r0 - 1), int(c)))
    if r1 < H - 1:
        for c in range(c0, c1 + 1):
            cells.add((int(r1 + 1), int(c)))
    if c0 > 0:
        for r in range(r0, r1 + 1):
            cells.add((int(r), int(c0 - 1)))
    if c1 < W - 1:
        for r in range(r0, r1 + 1):
            cells.add((int(r), int(c1 + 1)))
    if r0 > 0 and c0 > 0:
        cells.add((int(r0 - 1), int(c0 - 1)))
    if r0 > 0 and c1 < W - 1:
        cells.add((int(r0 - 1), int(c1 + 1)))
    if r1 < H - 1 and c0 > 0:
        cells.add((int(r1 + 1), int(c0 - 1)))
    if r1 < H - 1 and c1 < W - 1:
        cells.add((int(r1 + 1), int(c1 + 1)))
    return cells


def _trace_chain(coords: set[tuple[int, int]]) -> list[tuple[int, int]]:
    if not coords:
        return []
    if len(coords) == 1:
        return [next(iter(coords))]

    neighbor_map: dict[tuple[int, int], list[tuple[int, int]]] = {}
    endpoints: list[tuple[int, int]] = []
    for cell in coords:
        r, c = cell
        nbrs = []
        for dr, dc in _DIRS:
            nxt = (int(r + dr), int(c + dc))
            if nxt in coords:
                nbrs.append(nxt)
        deg = len(nbrs)
        assert deg in (1, 2), "trunk must be a simple chain"
        neighbor_map[cell] = nbrs
        if deg == 1:
            endpoints.append(cell)

    assert len(endpoints) == 2, "trunk must have exactly two endpoints"
    ordered = [endpoints[0]]
    prev = None
    cur = endpoints[0]
    while True:
        nbrs = neighbor_map[cur]
        nxts = [n for n in nbrs if n != prev]
        if not nxts:
            break
        assert len(nxts) == 1, "trunk traversal branched unexpectedly"
        prev, cur = cur, nxts[0]
        ordered.append(cur)
    assert len(ordered) == len(coords), "trunk traversal missed cells"
    return ordered


def _count_bends_from_ordered_path(path: list[tuple[int, int]]) -> int:
    if len(path) < 3:
        return 0
    bends = 0
    prev_dir = None
    for i in range(1, len(path)):
        dr = int(path[i][0] - path[i - 1][0])
        dc = int(path[i][1] - path[i - 1][1])
        cur_dir = (dr, dc)
        if prev_dir is not None and cur_dir != prev_dir:
            bends += 1
        prev_dir = cur_dir
    return int(bends)


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape
    path_color = _find_path_color(g, bg)
    hub, (hub_k, hub_r0, hub_c0) = _find_hub(g, path_color)

    seed_mask = (g != int(bg)) & (g != int(path_color))
    seed_lab, seed_n = label_mask(seed_mask, connectivity=4)
    assert int(seed_n) >= 1

    seeds: dict[int, dict] = {}
    all_seed_vals: list[int] = []
    for sid in range(1, int(seed_n) + 1):
        pts = np.argwhere(seed_lab == int(sid))
        r0 = int(pts[:, 0].min())
        r1 = int(pts[:, 0].max())
        c0 = int(pts[:, 1].min())
        c1 = int(pts[:, 1].max())
        vals = [int(g[int(r), int(c)]) for r, c in pts]
        freq = Counter(vals).most_common()
        assert len(freq) >= 2, "seed interior must expose majority and minority"
        major = int(freq[0][0])
        minor = int(freq[1][0])
        border = _seed_border_cells(r0, c0, r1, c1, H, W)
        seeds[int(sid)] = {
            "id": int(sid),
            "bbox": (int(r0), int(c0), int(r1), int(c1)),
            "majority": int(major),
            "minority": int(minor),
            "border": border,
            "pts": {(int(r), int(c)) for r, c in pts},
        }
        all_seed_vals.extend(vals)

    branch_mask = (g == int(path_color)) & (~hub)
    branch_lab, branch_n = label_mask(branch_mask, connectivity=4)
    assert int(branch_n) >= 1

    branches: list[dict] = []
    for bid in range(1, int(branch_n) + 1):
        pts = {(int(r), int(c)) for r, c in np.argwhere(branch_lab == int(bid))}
        touched = [sid for sid, info in seeds.items() if pts & info["border"]]
        assert len(touched) == 1, "each component must map to exactly one seed"
        sid = int(touched[0])
        border = seeds[sid]["border"]
        trunk = pts - border
        assert trunk, "branch trunk cannot be empty"
        ordered = _trace_chain(trunk)
        bends = _count_bends_from_ordered_path(ordered)
        branches.append(
            {
                "id": int(bid),
                "seed_id": int(sid),
                "pts": pts,
                "trunk": trunk,
                "ordered_trunk": ordered,
                "bends": int(bends),
                "parity": int(bends % 2),
                "majority": int(seeds[sid]["majority"]),
                "minority": int(seeds[sid]["minority"]),
            }
        )

    return {
        "grid": g,
        "bg": int(bg),
        "P": int(path_color),
        "hub": hub,
        "hub_info": (int(hub_k), int(hub_r0), int(hub_c0)),
        "seed_lab": seed_lab,
        "seeds": seeds,
        "branches": branches,
        "all_seed_vals": all_seed_vals,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a hub-and-branch network, then delete the odd-bend branches.

    Input structure:
    - The background is the most frequent color.
    - One non-background color forms a thin orthogonal network with a solid square hub.
    - Rectangular seed interiors sit on the outer edges, each surrounded by a P border on the
      non-edge sides; branch components include both that border and the connecting path trunk.

    Rule:
    - Infer the path color and the hub as in the base family.
    - For each non-hub branch component, isolate the trunk by removing the seed border cells, count
      its L-bends, and choose the seed majority color for even bend counts or the seed minority
      color for odd bend counts.
    - Apply that recolor first, then delete every odd-bend component to background, and finally set
      the hub center pixel to the most common surviving branch color touching the hub.

    Key insight: the visible survive/delete decision is encoded by bend parity on the path trunk,
    while the seed interior still determines which color an even branch would keep.
    """
    analysis = _analyze_grid(np.array(grid, dtype=int))
    g = analysis["grid"]
    bg = int(analysis["bg"])
    hub = analysis["hub"]
    path_color = int(analysis["P"])
    k, r0, c0 = analysis["hub_info"]

    recolored = g.copy()

    for branch in analysis["branches"]:
        pts = branch["pts"]
        if int(branch["parity"]) == 0:
            branch_color = int(branch["majority"])
        else:
            branch_color = int(branch["minority"])
        for r, c in pts:
            recolored[int(r), int(c)] = int(branch_color)

    out = recolored.copy()
    for branch in analysis["branches"]:
        if int(branch["parity"]) == 0:
            continue
        for r, c in branch["pts"]:
            out[int(r), int(c)] = int(bg)

    touch_cols = Counter()
    H, W = g.shape
    for r in range(H):
        for c in range(W):
            if not bool(hub[int(r), int(c)]):
                continue
            for dr, dc in _DIRS:
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < H and 0 <= cc < W and not bool(hub[rr, cc]):
                    col = int(out[rr, cc])
                    if col not in (int(bg), int(path_color)):
                        touch_cols[int(col)] += 1

    assert touch_cols, "at least one even-bend branch must survive"
    cr = int(r0 + k // 2)
    cc = int(c0 + k // 2)
    out[int(cr), int(cc)] = int(touch_cols.most_common(1)[0][0])
    return out


# ── Generator ────────────────────────────────────────────────────────────────

GRID_SIZE_MIN = 16
GRID_SIZE_MAX = 30
SEED_INTERIOR_MIN = 2
SEED_INTERIOR_MAX = 4
MINORITY_FRAC_LO = 0.10
MINORITY_FRAC_HI = 0.35
SEED_ZONE_GAP = 2
SEED_PLACEMENT_RETRIES = 120
SEGMENT_LEN_MIN = 1
SEGMENT_LEN_MAX = 6
MAX_BENDS = 6
PATH_ATTEMPTS_PER_SEED = 80
GEN_OUTER_RETRIES = 300


def _compute_seed_zone(edge: str, ir0: int, ic0: int, ir1: int, ic1: int, H: int, W: int) -> tuple[int, int, int, int]:
    if edge == "top":
        return (int(ir0), int(max(0, ic0 - 1)), int(ir1 + 1), int(min(W - 1, ic1 + 1)))
    if edge == "bottom":
        return (int(max(0, ir0 - 1)), int(max(0, ic0 - 1)), int(ir1), int(min(W - 1, ic1 + 1)))
    if edge == "left":
        return (int(max(0, ir0 - 1)), int(ic0), int(min(H - 1, ir1 + 1)), int(ic1 + 1))
    return (int(max(0, ir0 - 1)), int(max(0, ic0 - 1)), int(min(H - 1, ir1 + 1)), int(ic1))


def _zones_overlap(zone: tuple[int, int, int, int], others: list[tuple[int, int, int, int]], gap: int) -> bool:
    r0, c0, r1, c1 = zone
    for or0, oc0, or1, oc1 in others:
        if r0 - gap <= or1 and or0 - gap <= r1 and c0 - gap <= oc1 and oc0 - gap <= c1:
            return True
    return False


def _place_seed_border(work: np.ndarray, path_color: int, ir0: int, ic0: int, ir1: int, ic1: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    H, W = work.shape
    border_cells: list[tuple[int, int]] = []
    landing_cells: list[tuple[int, int]] = []
    for cond, cells in [
        (ir0 > 0, [(ir0 - 1, c) for c in range(ic0, ic1 + 1)]),
        (ir1 < H - 1, [(ir1 + 1, c) for c in range(ic0, ic1 + 1)]),
        (ic0 > 0, [(r, ic0 - 1) for r in range(ir0, ir1 + 1)]),
        (ic1 < W - 1, [(r, ic1 + 1) for r in range(ir0, ir1 + 1)]),
    ]:
        if cond:
            for r, c in cells:
                work[int(r), int(c)] = int(path_color)
                border_cells.append((int(r), int(c)))
                landing_cells.append((int(r), int(c)))
    for cond, cell in [
        (ir0 > 0 and ic0 > 0, (ir0 - 1, ic0 - 1)),
        (ir0 > 0 and ic1 < W - 1, (ir0 - 1, ic1 + 1)),
        (ir1 < H - 1 and ic0 > 0, (ir1 + 1, ic0 - 1)),
        (ir1 < H - 1 and ic1 < W - 1, (ir1 + 1, ic1 + 1)),
    ]:
        if cond:
            work[int(cell[0]), int(cell[1])] = int(path_color)
            border_cells.append((int(cell[0]), int(cell[1])))
    return border_cells, landing_cells


def _fill_seed_interior(
    work: np.ndarray,
    seed_id_grid: np.ndarray,
    ir0: int,
    ic0: int,
    ir1: int,
    ic1: int,
    majority: int,
    minority: int,
    sid: int,
) -> None:
    cells = [(r, c) for r in range(ir0, ir1 + 1) for c in range(ic0, ic1 + 1)]
    n_total = len(cells)
    min_minor = max(1, int(round(n_total * MINORITY_FRAC_LO)))
    max_minor = max(min_minor, int(round(n_total * MINORITY_FRAC_HI)))
    n_minor = random.randint(min_minor, max_minor)
    if n_minor >= n_total - n_minor:
        n_minor = max(1, (n_total - 1) // 2)
    random.shuffle(cells)
    for r, c in cells[:n_minor]:
        work[int(r), int(c)] = int(minority)
        seed_id_grid[int(r), int(c)] = int(sid)
    for r, c in cells[n_minor:]:
        work[int(r), int(c)] = int(majority)
        seed_id_grid[int(r), int(c)] = int(sid)


def _pick_seed_interior(edge: str, H: int, W: int, hub_box: tuple[int, int, int, int], used_zones: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int] | None:
    hr0, hc0, hr1, hc1 = hub_box
    for _ in range(SEED_PLACEMENT_RETRIES):
        ih = random.randint(SEED_INTERIOR_MIN, SEED_INTERIOR_MAX)
        iw = random.randint(SEED_INTERIOR_MIN, SEED_INTERIOR_MAX)
        if edge == "top":
            ir0 = 0
            ic0 = random.randint(0, W - iw)
        elif edge == "bottom":
            ir0 = H - ih
            ic0 = random.randint(0, W - iw)
        elif edge == "left":
            ir0 = random.randint(0, H - ih)
            ic0 = 0
        else:
            ir0 = random.randint(0, H - ih)
            ic0 = W - iw
        ir1 = int(ir0 + ih - 1)
        ic1 = int(ic0 + iw - 1)
        zone = _compute_seed_zone(edge, ir0, ic0, ir1, ic1, H, W)
        zr0, zc0, zr1, zc1 = zone
        if not (zr1 < hr0 - 1 or zr0 > hr1 + 1 or zc1 < hc0 - 1 or zc0 > hc1 + 1):
            continue
        if _zones_overlap(zone, used_zones, gap=SEED_ZONE_GAP):
            continue
        return (int(ir0), int(ic0), int(ir1), int(ic1))
    return None


def _count_bends_on_generated_path(path: list[tuple[int, int]]) -> int:
    return _count_bends_from_ordered_path(path)


def _try_carve_path(canvas: np.ndarray, path_color: int, start: tuple[int, int], initial_dir: tuple[int, int], dest_border_cells: list[tuple[int, int]], n_bends: int) -> list[tuple[int, int]] | None:
    H, W = canvas.shape
    dest_set = set(dest_border_cells)

    def _check_neighbors(r: int, c: int, prev: tuple[int, int], allow_dest: bool) -> bool:
        if not (0 <= r < H and 0 <= c < W) or int(canvas[int(r), int(c)]) != 0:
            return False
        for dr, dc in _DIRS:
            nr = int(r + dr)
            nc = int(c + dc)
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if (nr, nc) == prev or (nr, nc) == start:
                continue
            v = int(canvas[nr, nc])
            if v == int(path_color):
                if allow_dest and (nr, nc) in dest_set:
                    continue
                return False
            if v != 0:
                return False
        return True

    path: list[tuple[int, int]] = []
    cur = (int(start[0]), int(start[1]))
    cur_dir = (int(initial_dir[0]), int(initial_dir[1]))

    for seg_i in range(int(n_bends) + 1):
        is_last = seg_i == int(n_bends)
        max_steps = max(H, W) if is_last else random.randint(SEGMENT_LEN_MIN, SEGMENT_LEN_MAX)
        moved = False
        blocked = False
        for _ in range(int(max_steps)):
            nr = int(cur[0] + cur_dir[0])
            nc = int(cur[1] + cur_dir[1])
            if any((int(nr + dr), int(nc + dc)) in dest_set for dr, dc in _DIRS) and _check_neighbors(nr, nc, cur, allow_dest=True):
                canvas[int(nr), int(nc)] = int(path_color)
                path.append((int(nr), int(nc)))
                return path
            if _check_neighbors(nr, nc, cur, allow_dest=False):
                canvas[int(nr), int(nc)] = int(path_color)
                path.append((int(nr), int(nc)))
                cur = (int(nr), int(nc))
                moved = True
            else:
                blocked = True
                break
        if is_last or blocked or not moved:
            for r, c in path:
                canvas[int(r), int(c)] = 0
            return None
        perps = [(-cur_dir[1], cur_dir[0]), (cur_dir[1], -cur_dir[0])]
        random.shuffle(perps)
        cur_dir = (int(perps[0][0]), int(perps[0][1]))

    for r, c in path:
        canvas[int(r), int(c)] = 0
    return None


def _carve_path_with_parity(
    canvas: np.ndarray,
    path_color: int,
    start: tuple[int, int],
    initial_dir: tuple[int, int],
    dest_border_cells: list[tuple[int, int]],
    target_parity: int,
) -> list[tuple[int, int]] | None:
    bend_choices = [0, 2, 4, 6] if int(target_parity) == 0 else [1, 3, 5]
    for _ in range(PATH_ATTEMPTS_PER_SEED):
        n_bends = random.choice(bend_choices)
        path = _try_carve_path(canvas, path_color, start, initial_dir, dest_border_cells, n_bends)
        if path is None:
            continue
        if _count_bends_on_generated_path(path) % 2 == int(target_parity):
            return path
        for r, c in path:
            canvas[int(r), int(c)] = 0
    return None


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    W = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    bg, path_color, color_a, color_b = random.sample(list(range(10)), 4)
    g = np.full((H, W), int(bg), dtype=int)

    hr0 = random.randint(max(2, H // 3 - 1), min(H - 5, 2 * H // 3 - 2))
    hc0 = random.randint(max(2, W // 3 - 1), min(W - 5, 2 * W // 3 - 2))
    g[hr0 : hr0 + HUB_SIZE, hc0 : hc0 + HUB_SIZE] = int(path_color)
    hub_box = (int(hr0), int(hc0), int(hr0 + HUB_SIZE - 1), int(hc0 + HUB_SIZE - 1))

    work = np.where(g == int(bg), 0, g).astype(int)
    seed_id_grid = np.zeros((H, W), dtype=int)
    edges = ["top", "bottom", "left", "right"]
    used_zones: list[tuple[int, int, int, int]] = []
    seeds: list[dict] = []

    odd_edge = random.choice(edges)
    even_edges = [edge for edge in edges if edge != odd_edge]
    winner_color = random.choice([int(color_a), int(color_b)])
    loser_color = int(color_b) if int(winner_color) == int(color_a) else int(color_a)
    special_even = random.choice(even_edges)

    majority_by_edge: dict[str, int] = {}
    for edge in even_edges:
        majority_by_edge[edge] = int(loser_color if edge == special_even else winner_color)
    majority_by_edge[odd_edge] = int(random.choice([color_a, color_b]))

    for edge in edges:
        picked = _pick_seed_interior(edge, H, W, hub_box, used_zones)
        if picked is None:
            return None
        ir0, ic0, ir1, ic1 = picked
        majority = int(majority_by_edge[edge])
        minority = int(color_b) if int(majority) == int(color_a) else int(color_a)
        sid = len(seeds) + 1
        _fill_seed_interior(work, seed_id_grid, ir0, ic0, ir1, ic1, majority, minority, sid)
        border_cells, landing_cells = _place_seed_border(work, path_color, ir0, ic0, ir1, ic1)
        zone = _compute_seed_zone(edge, ir0, ic0, ir1, ic1, H, W)
        used_zones.append(zone)
        seeds.append(
            {
                "id": int(sid),
                "edge": edge,
                "majority": int(majority),
                "minority": int(minority),
                "landing_cells": landing_cells,
                "border_cells": border_cells,
            }
        )

    start_map = {
        "top": ((hr0, hc0 + 1), (-1, 0)),
        "bottom": ((hr0 + HUB_SIZE - 1, hc0 + 1), (1, 0)),
        "left": ((hr0 + 1, hc0), (0, -1)),
        "right": ((hr0 + 1, hc0 + HUB_SIZE - 1), (0, 1)),
    }

    for seed in seeds:
        start, initial_dir = start_map[seed["edge"]]
        target_parity = 1 if seed["edge"] == odd_edge else 0
        path = _carve_path_with_parity(work, path_color, start, initial_dir, seed["landing_cells"], target_parity)
        if path is None:
            return None
        seed["path"] = path

    inp = np.where(work == 0, int(bg), work).astype(int)
    if int(get_bg(inp)) != int(bg):
        return None

    try:
        analysis = _analyze_grid(inp)
        branches = analysis["branches"]
        parities = sorted(int(branch["parity"]) for branch in branches)
        if parities.count(1) != 1 or parities.count(0) != 3:
            return None
        out = main(inp)
        k, r0, c0 = analysis["hub_info"]
        center = int(out[int(r0 + k // 2), int(c0 + k // 2)])
        if center != int(winner_color):
            return None
        deleted = sum(1 for branch in branches if int(branch["parity"]) == 1)
        if deleted < 1:
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_OUTER_RETRIES):
        try:
            grid = _gen_attempt()
        except Exception:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_OUTER_RETRIES} attempts")
