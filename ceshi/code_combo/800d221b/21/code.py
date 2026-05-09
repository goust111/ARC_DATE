# Combo variant of 800d221b (index 21)
# combo_spec: D7/0+D11/2
# Name: hub大小编码种子数量
#
# Design analysis:
# a) Concreteization and behavior trace:
#    - Original task: a fixed 3x3 hub feeds 3-5 seed regions; the hub size itself carries no meaning.
#    - Combo task: the hub is a solid kxk square with k in {2,3,4,5}, and that same k is the exact seed count.
#      Example: with a 4x4 hub, the input must contain exactly 4 seed interiors. The solver must first measure the
#      hub, decode "expect 4 seeds", then verify and match those 4 seeds before recoloring the branches.
#    - The recoloring rule is otherwise preserved: each non-hub P-component inherits the majority color of the seed
#      interior it touches most, and a single designated hub cell takes the hub-adjacent majority branch color.
#      For even k there is no geometric single center, so the designated cell is fixed as (r0 + k//2, c0 + k//2),
#      i.e. the lower-right cell of the central 2x2 block.
# b) Complexity calibration:
#    - k is limited to four visible sizes, so the encoded parameter is easy to notice but still nontrivial.
#    - Seed colors remain only two-valued inside the seed interiors, avoiding a separate color-mapping puzzle.
#    - One seed gets two hub-to-seed paths while the others get one, ensuring the hub-center recolor is readable
#      instead of relying on arbitrary tie-breaking.
# c) Hidden-assumption audit against the original solver:
#    - "hub is 3x3" is invalidated; hub search now accepts only k in {2,3,4,5}.
#    - "seed count is independently visible" becomes an explicit inference step; main() now asserts seed_n == k.
#    - "center pixel" becomes ambiguous for even k; main() and generator share the designated-cell convention above.
#    - Branch recoloring still assumes outside the hub there is no larger solid P square and no path T-junctions.
#      The generator preserves thin paths and framed seeds so those assumptions stay valid.
# d) Placement legality:
#    - The generator places the hub away from edges, uses edge-touching rectangular seed interiors with P borders on
#      non-edge sides, rejects zone overlaps, and only accepts paths carved through free cells.
# e) Visual observability:
#    - Across samples, counting seeds and measuring the hub side length are both directly visible.
#      Since the counts must match, a human can infer the encoded parameter from 2-4 examples.
# f) Visual cleanliness:
#    - Seeds are distributed across distinct edges when possible, hub starts prefer the matching hub side, and only
#      one extra path is added, keeping the picture sparse enough to parse by eye.
# g) Change inferability:
#    - The key fused change is structural, not numeric-color based: hub size visibly predicts seed count.
#      Color changes still have clear support from local seed-majority cues.
# Additional audit of new solver assumptions:
#    - main() assumes exactly k seed interiors exist, every seed interior is P-framed on non-edge sides, and the
#      recolored center color is uniquely determined by hub-adjacent branch counts. generate_input() constructs all
#      three properties directly.

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import components_mask, get_bg, label_mask


HUB_AND_SEED_CHOICES = [2, 3, 4, 5]
GEN_HUB_AND_SEED_CHOICES = [2, 3, 4, 4, 5, 5, 5]
_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _infer_path_color(g: np.ndarray, bg: int) -> int:
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


def _find_hub_square(g: np.ndarray, p_color: int) -> tuple[int, int, int]:
    H, W = g.shape
    mP = g == int(p_color)
    ps = mP.astype(int).cumsum(axis=0).cumsum(axis=1)

    def rect_sum(r0: int, c0: int, r1: int, c1: int) -> int:
        s = int(ps[int(r1), int(c1)])
        if r0 > 0:
            s -= int(ps[int(r0 - 1), int(c1)])
        if c0 > 0:
            s -= int(ps[int(r1), int(c0 - 1)])
        if r0 > 0 and c0 > 0:
            s += int(ps[int(r0 - 1), int(c0 - 1)])
        return int(s)

    best = None
    for k in sorted(HUB_AND_SEED_CHOICES, reverse=True):
        if k > H or k > W:
            continue
        for r0 in range(0, int(H - k + 1)):
            r1 = int(r0 + k - 1)
            for c0 in range(0, int(W - k + 1)):
                c1 = int(c0 + k - 1)
                if rect_sum(int(r0), int(c0), int(r1), int(c1)) == int(k * k):
                    best = (int(k), int(r0), int(c0))
        if best is not None:
            break

    assert best is not None
    return best


def _hub_perimeter_cells(r0: int, c0: int, k: int) -> list[tuple[int, int]]:
    return [
        (r, c)
        for r in range(int(r0), int(r0 + k))
        for c in range(int(c0), int(c0 + k))
        if r in (int(r0), int(r0 + k - 1)) or c in (int(c0), int(c0 + k - 1))
    ]


def _hub_side_cells(r0: int, c0: int, k: int) -> dict[str, list[tuple[int, int]]]:
    return {
        "top": [(int(r0), c) for c in range(int(c0), int(c0 + k))],
        "bottom": [(int(r0 + k - 1), c) for c in range(int(c0), int(c0 + k))],
        "left": [(r, int(c0)) for r in range(int(r0), int(r0 + k))],
        "right": [(r, int(c0 + k - 1)) for r in range(int(r0), int(r0 + k))],
    }


def _designated_hub_cell(r0: int, c0: int, k: int) -> tuple[int, int]:
    return int(r0 + k // 2), int(c0 + k // 2)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a path-network whose hub size encodes how many seeds exist.

    Input structure:
    - bg is the most common color.
    - A thin orthogonal path network uses one path color P and contains one solid square hub of size kxk.
    - Seed interiors are the non-bg, non-P regions, and their count is encoded by the hub size: exactly k seeds.

    Rule:
    - Infer P from the network-like foreground component, then find the solid hub square and read its size k.
    - Use that k as the expected number of seed interiors, compute each seed's majority interior color, and recolor
      every non-hub P-branch by the seed interior it touches most.
    - Leave the hub itself in P except for the designated hub cell at (r0 + k//2, c0 + k//2), which takes the color
      with the most hub-adjacent recolored branch pixels.

    Key insight:
    - The hub is not just a junction to exclude from branch recoloring; its side length is a visible parameter that
      tells you how many seeds must be found before the usual branch-to-seed recoloring rule is applied.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    P = _infer_path_color(g, bg)
    k, r0, c0 = _find_hub_square(g, P)
    assert k in HUB_AND_SEED_CHOICES

    r1 = int(r0 + k - 1)
    c1 = int(c0 + k - 1)
    hub = np.zeros((H, W), dtype=bool)
    hub[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = True

    seed_mask = (g != int(bg)) & (g != int(P))
    seed_lab, seed_n = label_mask(seed_mask, connectivity=4)
    assert int(seed_n) == int(k), "hub size must equal seed count"

    seed_major: dict[int, int] = {}
    all_seed_vals: list[int] = []
    for sid in range(1, int(seed_n) + 1):
        pts = np.argwhere(seed_lab == int(sid))
        rn, rx = int(pts[:, 0].min()), int(pts[:, 0].max())
        cn, cx = int(pts[:, 1].min()), int(pts[:, 1].max())
        if rn > 0:
            assert all(int(g[rn - 1, c]) == int(P) for c in range(cn, cx + 1)), "seed missing top P border"
        if rx < H - 1:
            assert all(int(g[rx + 1, c]) == int(P) for c in range(cn, cx + 1)), "seed missing bottom P border"
        if cn > 0:
            assert all(int(g[r, cn - 1]) == int(P) for r in range(rn, rx + 1)), "seed missing left P border"
        if cx < W - 1:
            assert all(int(g[r, cx + 1]) == int(P) for r in range(rn, rx + 1)), "seed missing right P border"
        vals = [int(g[int(r), int(c)]) for r, c in pts]
        maj = int(Counter(vals).most_common(1)[0][0])
        seed_major[int(sid)] = int(maj)
        all_seed_vals.extend(vals)

    out = g.copy()

    branch_mask = (g == int(P)) & (~hub)
    branch_lab, branch_n = label_mask(branch_mask, connectivity=4)

    pmask = g == int(P)
    ppad = np.pad(pmask, 1)
    p_deg = ppad[2:, 1:-1].astype(int) + ppad[:-2, 1:-1] + ppad[1:-1, 2:] + ppad[1:-1, :-2]
    spad = np.pad(seed_lab > 0, 1)
    adj_seed = spad[2:, 1:-1] | spad[:-2, 1:-1] | spad[1:-1, 2:] | spad[1:-1, :-2]
    assert not np.any((p_deg > 2) & pmask & ~hub & ~adj_seed), "path has T-junction"

    for bid in range(1, int(branch_n) + 1):
        pts = np.argwhere(branch_lab == int(bid))
        touches = Counter()
        for r, c in pts:
            r = int(r)
            c = int(c)
            for dr, dc in _DIRS:
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < H and 0 <= cc < W:
                    sid = int(seed_lab[rr, cc])
                    if sid != 0:
                        touches[int(sid)] += 1
        if not touches:
            continue
        sid = int(touches.most_common(1)[0][0])
        col = int(seed_major[int(sid)])
        for r, c in pts:
            out[int(r), int(c)] = int(col)

    touch_cols = Counter()
    if int(branch_n) > 0:
        for r in range(H):
            for c in range(W):
                if not bool(hub[int(r), int(c)]):
                    continue
                for dr, dc in _DIRS:
                    rr = int(r + dr)
                    cc = int(c + dc)
                    if 0 <= rr < H and 0 <= cc < W and int(branch_lab[rr, cc]) != 0:
                        touch_cols[int(out[rr, cc])] += 1

    cr, cc = _designated_hub_cell(r0, c0, k)
    assert touch_cols, "hub must be touched by at least one recolored branch"
    assert all_seed_vals, "grid must contain seed regions"
    center_col = int(touch_cols.most_common(1)[0][0])
    out[int(cr), int(cc)] = int(center_col)

    return out


# --------------- generator ---------------

# Canvas / grid
GRID_SIZE_BASE = 12
GRID_SIZE_PER_SEED = 3
GRID_SIZE_MIN = 12
GRID_SIZE_MAX = 30

# Seed regions
SEED_INTERIOR_MIN = 2
SEED_INTERIOR_MAX = 4
MINORITY_FRAC_LO = 0.06
MINORITY_FRAC_HI = 0.40
SEED_ZONE_GAP = 1
SEED_PLACEMENT_RETRIES = 160

# Path carving
PATH_RETRIES_PER_SEED = 80
PATH_EXTRA_RETRIES = 100

# Top-level generation
GEN_OUTER_RETRIES = 600


def _try_carve_path(canvas: np.ndarray, p_color: int, start: tuple[int, int], dest_border_cells: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    """Carve one 1-cell-wide orthogonal path from a hub cell to a seed border."""
    H, W = canvas.shape
    dest_set = set(dest_border_cells)

    def _candidate_ok(r: int, c: int, prev: tuple[int, int]) -> tuple[bool, bool]:
        if not (0 <= r < H and 0 <= c < W) or int(canvas[r, c]) != 0:
            return False, False

        touches_dest = False
        for dr, dc in _DIRS:
            nr = int(r + dr)
            nc = int(c + dc)
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if (nr, nc) == prev or (nr, nc) == start:
                continue

            v = int(canvas[nr, nc])
            if v == 0:
                continue
            if v == int(p_color) and (nr, nc) in dest_set:
                touches_dest = True
                continue
            return False, False
        return True, touches_dest

    frontier: list[tuple[int, int, tuple[int, int]]] = []
    parent: dict[tuple[int, int, tuple[int, int]], tuple[int, int, tuple[int, int]] | None] = {}

    init_dirs = list(_DIRS)
    random.shuffle(init_dirs)
    for dr, dc in init_dirs:
        nr = int(start[0] + dr)
        nc = int(start[1] + dc)
        ok, touches_dest = _candidate_ok(nr, nc, start)
        if not ok:
            continue
        state = (int(nr), int(nc), start)
        frontier.append(state)
        parent[state] = None
        if touches_dest:
            canvas[int(nr), int(nc)] = int(p_color)
            return [(int(nr), int(nc))]

    if not frontier:
        return None

    head = 0
    seen = set(frontier)
    while head < len(frontier):
        r, c, prev = frontier[head]
        head += 1

        next_dirs = list(_DIRS)
        random.shuffle(next_dirs)
        for dr, dc in next_dirs:
            nr = int(r + dr)
            nc = int(c + dc)
            if (nr, nc) == prev or (nr, nc) == start:
                continue

            ok, touches_dest = _candidate_ok(nr, nc, (int(r), int(c)))
            if not ok:
                continue

            state = (int(nr), int(nc), (int(r), int(c)))
            if state in seen:
                continue
            seen.add(state)
            parent[state] = (int(r), int(c), prev)
            frontier.append(state)

            if not touches_dest:
                continue

            path_rev = [(int(nr), int(nc))]
            cur = (int(r), int(c), prev)
            while cur is not None:
                path_rev.append((int(cur[0]), int(cur[1])))
                cur = parent[cur]
            path = list(reversed(path_rev))
            for pr, pc in path:
                canvas[int(pr), int(pc)] = int(p_color)
            return path

    return None


def _compute_seed_zone(edge: str, ir0: int, ic0: int, ir1: int, ic1: int, H: int, W: int) -> tuple[int, int, int, int]:
    if edge == "top":
        return int(ir0), max(0, int(ic0 - 1)), int(ir1 + 1), min(int(W - 1), int(ic1 + 1))
    if edge == "bottom":
        return max(0, int(ir0 - 1)), max(0, int(ic0 - 1)), int(ir1), min(int(W - 1), int(ic1 + 1))
    if edge == "left":
        return max(0, int(ir0 - 1)), int(ic0), min(int(H - 1), int(ir1 + 1)), int(ic1 + 1)
    return max(0, int(ir0 - 1)), max(0, int(ic0 - 1)), min(int(H - 1), int(ir1 + 1)), int(ic1)


def _place_seed_border(work: np.ndarray, p_color: int, ir0: int, ic0: int, ir1: int, ic1: int, H: int, W: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
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
                work[int(r), int(c)] = int(p_color)
                border_cells.append((int(r), int(c)))
                landing_cells.append((int(r), int(c)))
    for cond, rc in [
        (ir0 > 0 and ic0 > 0, (ir0 - 1, ic0 - 1)),
        (ir0 > 0 and ic1 < W - 1, (ir0 - 1, ic1 + 1)),
        (ir1 < H - 1 and ic0 > 0, (ir1 + 1, ic0 - 1)),
        (ir1 < H - 1 and ic1 < W - 1, (ir1 + 1, ic1 + 1)),
    ]:
        if cond:
            work[int(rc[0]), int(rc[1])] = int(p_color)
            border_cells.append((int(rc[0]), int(rc[1])))
    return border_cells, landing_cells


def _zones_overlap(zone: tuple[int, int, int, int], others: list[tuple[int, int, int, int]], gap: int) -> bool:
    r0, c0, r1, c1 = zone
    for or0, oc0, or1, oc1 in others:
        if int(r0 - gap) <= int(or1) and int(or0 - gap) <= int(r1) and int(c0 - gap) <= int(oc1) and int(oc0 - gap) <= int(c1):
            return True
    return False


def _pick_seed_interior(edge: str, H: int, W: int) -> tuple[int, int, int, int]:
    ih = random.randint(SEED_INTERIOR_MIN, SEED_INTERIOR_MAX)
    iw = random.randint(SEED_INTERIOR_MIN, SEED_INTERIOR_MAX)
    if edge == "top":
        ir0, ic0 = 0, random.randint(0, int(W - iw))
    elif edge == "bottom":
        ir0, ic0 = int(H - ih), random.randint(0, int(W - iw))
    elif edge == "left":
        ir0, ic0 = random.randint(0, int(H - ih)), 0
    else:
        ir0, ic0 = random.randint(0, int(H - ih)), int(W - iw)
    return int(ir0), int(ic0), int(ir0 + ih - 1), int(ic0 + iw - 1)


def _fill_seed_interior(work: np.ndarray, ir0: int, ic0: int, ir1: int, ic1: int, majority: int, minority: int) -> None:
    interior_cells = [(r, c) for r in range(ir0, ir1 + 1) for c in range(ic0, ic1 + 1)]
    n_int = len(interior_cells)
    min_minor = max(1, int(n_int * MINORITY_FRAC_LO))
    max_minor = max(min_minor, int(n_int * MINORITY_FRAC_HI))
    n_minor = random.randint(min_minor, max_minor)
    if n_minor >= n_int - n_minor:
        n_minor = max(1, (n_int - 1) // 2)
    random.shuffle(interior_cells)
    for r, c in interior_cells[:n_minor]:
        work[int(r), int(c)] = int(minority)
    for r, c in interior_cells[n_minor:]:
        work[int(r), int(c)] = int(majority)


def _candidate_starts(edge: str, hub_sides: dict[str, list[tuple[int, int]]], hub_perimeter: list[tuple[int, int]], used_starts: set[tuple[int, int]]) -> list[tuple[int, int]]:
    preferred = [cell for cell in hub_sides[edge] if cell not in used_starts]
    fallback = [cell for cell in hub_perimeter if cell not in used_starts and cell not in preferred]
    random.shuffle(preferred)
    random.shuffle(fallback)
    return preferred + fallback


def _carve_all_paths(
    work: np.ndarray,
    p_color: int,
    hub_perimeter: list[tuple[int, int]],
    hub_sides: dict[str, list[tuple[int, int]]],
    seeds: list[dict[str, object]],
    winner_idx: int,
) -> list[int] | None:
    used_hub_starts: set[tuple[int, int]] = set()
    path_counts = [0 for _ in seeds]

    seed_order = list(range(len(seeds)))
    random.shuffle(seed_order)
    for si in seed_order:
        ok = False
        edge = str(seeds[si]["edge"])
        starts = _candidate_starts(edge, hub_sides, hub_perimeter, used_hub_starts)
        for start in starts[:PATH_RETRIES_PER_SEED]:
            path = _try_carve_path(work, p_color, start, list(seeds[si]["landing_cells"]))
            if path is not None:
                used_hub_starts.add(start)
                path_counts[si] += 1
                ok = True
                break
        if not ok:
            return None

    edge = str(seeds[winner_idx]["edge"])
    starts = _candidate_starts(edge, hub_sides, hub_perimeter, used_hub_starts)
    for start in starts[:PATH_EXTRA_RETRIES]:
        path = _try_carve_path(work, p_color, start, list(seeds[winner_idx]["landing_cells"]))
        if path is not None:
            used_hub_starts.add(start)
            path_counts[winner_idx] += 1
            return path_counts
    return None


def _planned_edges(k: int) -> list[str]:
    edges = ["top", "bottom", "left", "right"]
    if k <= 4:
        chosen = random.sample(edges, k)
    else:
        chosen = edges + [random.choice(edges)]
    random.shuffle(chosen)
    return chosen


def _choose_majorities(color_a: int, color_b: int, path_counts: list[int]) -> list[int]:
    for _ in range(200):
        majors = [random.choice([color_a, color_b]) for _ in path_counts]
        if len(set(majors)) < 2:
            continue
        wa = sum(path_counts[i] for i, col in enumerate(majors) if int(col) == int(color_a))
        wb = sum(path_counts[i] for i, col in enumerate(majors) if int(col) == int(color_b))
        if wa != wb:
            return [int(x) for x in majors]

    majors = [int(color_a if i % 2 == 0 else color_b) for i in range(len(path_counts))]
    if len(path_counts) >= 2:
        majors[-1] = int(color_b)
    wa = sum(path_counts[i] for i, col in enumerate(majors) if int(col) == int(color_a))
    wb = sum(path_counts[i] for i, col in enumerate(majors) if int(col) == int(color_b))
    if wa == wb:
        majors[-1] = int(color_a if majors[-1] == int(color_b) else color_b)
    return majors


def _gen_attempt() -> np.ndarray | None:
    k = random.choice(GEN_HUB_AND_SEED_CHOICES)
    size_lo = max(GRID_SIZE_MIN, GRID_SIZE_BASE + GRID_SIZE_PER_SEED * (k - 2))
    H = random.randint(size_lo, GRID_SIZE_MAX)
    W = random.randint(size_lo, GRID_SIZE_MAX)
    bg, P, A, B = random.sample(list(range(10)), 4)

    g = np.full((H, W), bg, dtype=int)

    r_lo = max(2, H // 3 - k // 2)
    r_hi = min(H - k - 2, 2 * H // 3 - k // 2)
    c_lo = max(2, W // 3 - k // 2)
    c_hi = min(W - k - 2, 2 * W // 3 - k // 2)
    if r_lo > r_hi or c_lo > c_hi:
        return None

    hr0 = random.randint(int(r_lo), int(r_hi))
    hc0 = random.randint(int(c_lo), int(c_hi))
    g[int(hr0) : int(hr0 + k), int(hc0) : int(hc0 + k)] = int(P)

    hub_perimeter = _hub_perimeter_cells(hr0, hc0, k)
    hub_sides = _hub_side_cells(hr0, hc0, k)

    work = np.where(g == int(bg), 0, g).astype(int)
    seeds: list[dict[str, object]] = []
    used_zones: list[tuple[int, int, int, int]] = []

    def _overlaps_hub(zone: tuple[int, int, int, int]) -> bool:
        r0, c0, r1, c1 = zone
        return not (int(r1) < int(hr0 - 1) or int(r0) > int(hr0 + k) or int(c1) < int(hc0 - 1) or int(c0) > int(hc0 + k))

    for edge in _planned_edges(k):
        placed = False
        for _ in range(SEED_PLACEMENT_RETRIES):
            ir0, ic0, ir1, ic1 = _pick_seed_interior(edge, H, W)
            border_zone = _compute_seed_zone(edge, ir0, ic0, ir1, ic1, H, W)
            if _overlaps_hub(border_zone):
                continue
            if _zones_overlap(border_zone, used_zones, gap=SEED_ZONE_GAP):
                continue
            used_zones.append(border_zone)
            seeds.append(
                {
                    "edge": edge,
                    "bbox": (int(ir0), int(ic0), int(ir1), int(ic1)),
                    "border_zone": border_zone,
                }
            )
            placed = True
            break
        if not placed:
            return None

    assert len(seeds) == k
    winner_idx = random.randrange(k)
    planned_path_counts = [1 for _ in seeds]
    planned_path_counts[winner_idx] += 1
    majorities = _choose_majorities(A, B, planned_path_counts)

    for sid, seed in enumerate(seeds):
        ir0, ic0, ir1, ic1 = map(int, seed["bbox"])
        majority = int(majorities[sid])
        minority = int(B if majority == int(A) else A)
        _fill_seed_interior(work, ir0, ic0, ir1, ic1, majority, minority)
        border_cells, landing_cells = _place_seed_border(work, P, ir0, ic0, ir1, ic1, H, W)
        seed["majority"] = majority
        seed["minority"] = minority
        seed["border_cells"] = border_cells
        seed["landing_cells"] = landing_cells

    actual_path_counts = _carve_all_paths(work, P, hub_perimeter, hub_sides, seeds, winner_idx)
    if actual_path_counts is None:
        return None

    color_touch = Counter()
    for seed, n_paths in zip(seeds, actual_path_counts):
        color_touch[int(seed["majority"])] += int(n_paths)
    if len(color_touch) < 2 or len({int(v) for v in color_touch.values()}) != len(color_touch):
        return None

    inp = np.where(work == 0, bg, work).astype(int)
    counts = Counter(int(x) for x in inp.flatten())
    if int(counts.most_common(1)[0][0]) != int(bg):
        return None

    try:
        out = main(inp)
    except Exception:
        return None

    cr, cc = _designated_hub_cell(hr0, hc0, k)
    if int(out[cr, cc]) == int(P):
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
