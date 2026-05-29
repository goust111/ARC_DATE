# Variant of abc82100
# Dimension: D9 — 规则数量与源色-目标色关系 (meta)
# Change: 可有多条规则（多个不同的源色-目标色对），每条规则有自己的teal模板；每种源色对应一种目标色... → 同一源色对应多个目标色（条件分支）

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import components_mask, get_bg


# Design analysis for the D9 variant
# ----------------------------------
# concrete_behavior:
#   - Each source-color family now contains exactly two legend examples with the
#     same source color but different target colors and different teal stencils.
#   - Scattered source pixels keep that shared source color in the input.
#   - In the output, each scattered source pixel chooses the branch whose legend
#     source cell is spatially closest in Chebyshev distance, then stamps that
#     branch's stencil recolored to that branch's target color.
#   - Example trace:
#       original: blue source pixels always use one blue->red legend, so every
#       blue scatter becomes the red stencil.
#       variant: two blue legends exist, blue->red on the left and blue->yellow
#       on the right. A blue scatter near the left legend uses the red stencil;
#       another blue scatter near the right legend uses the yellow stencil.
#       The difference is not color-id based; it is visible as a spatial split.
# complexity_calibration:
#   - Branch count is fixed at 2 per source-color family. This is enough to make
#     the new conditional rule visible without exploding the hypothesis space.
#   - Every branch receives at least two scattered pixels, so each branch is
#     repeated inside a sample instead of appearing as a one-off exception.
# hidden_assumptions_checked:
#   - The original solver stamped every rule onto every pixel of that source
#     color. That depended on "one source color => one target color". It is no
#     longer valid, so stamping is now done per scattered source pixel after a
#     nearest-legend choice.
#   - The original "preserve bystanders" logic still works only if bystander
#     colors never reuse any source color; the generator keeps that guarantee.
#   - Deterministic tie handling is added in the solver, but the generator also
#     avoids equal-distance cases so the visible rule stays unambiguous.
# placement_legality:
#   - Legends are placed with a radius-1 Chebyshev safety halo so a legend anchor
#     never accidentally matches the wrong teal component.
#   - Same-source legends are forced far apart, and scattered pixels assigned to
#     one branch must be strictly closer to that branch's legend than to the
#     sibling legend.
#   - Output stencil footprints are kept fully in bounds and non-overlapping to
#     keep the visual parse clean and avoid order-dependent overwrites.
# observability_and_cleanliness:
#   - The same input color splits into two spatial regions that produce two
#     different target colors and usually two different shapes. This makes the
#     D9 change visible in 2-3 examples.
#   - No hidden ordering by numeric color id is used anywhere. Only visible
#     geometry matters: legend position, shared source color, and stencil shape.


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _anchor_distance(pattern: np.ndarray, anchor: tuple[int, int]) -> int:
    return min(_chebyshev((int(r), int(c)), anchor) for r, c in pattern)


def _discover_rules(grid: np.ndarray, bg: int, pattern_color: int = 8) -> list[dict]:
    h, w = grid.shape
    patterns = components_mask(grid == pattern_color, connectivity=8)
    rules: list[dict] = []
    seen: set[tuple[tuple[int, int], tuple[int, int], int]] = set()

    for r in range(h):
        for c in range(w):
            source_color = int(grid[r, c])
            if source_color in (int(bg), int(pattern_color)):
                continue

            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                tr = r + dr
                tc = c + dc
                if not (0 <= tr < h and 0 <= tc < w):
                    continue

                target_color = int(grid[tr, tc])
                if target_color in (int(bg), int(pattern_color)) or target_color == source_color:
                    continue

                anchor = (tr + dr, tc + dc)
                candidates: list[tuple[int, int, np.ndarray]] = []
                for i, pattern in enumerate(patterns):
                    dist = _anchor_distance(pattern, anchor)
                    if dist <= 1:
                        candidates.append((int(dist), int(i), pattern))

                if not candidates:
                    continue

                candidates.sort(key=lambda item: (item[0], len(item[2])))
                _dist, pattern_index, pattern = candidates[0]
                key = ((int(r), int(c)), (int(tr), int(tc)), int(pattern_index))
                if key in seen:
                    continue
                seen.add(key)
                rules.append(
                    {
                        "source_color": int(source_color),
                        "target_color": int(target_color),
                        "source_pos": (int(r), int(c)),
                        "target_pos": (int(tr), int(tc)),
                        "anchor": (int(anchor[0]), int(anchor[1])),
                        "pattern": np.asarray(pattern, dtype=int),
                    }
                )

    return rules


def _choose_rule_for_source(rules_for_color: list[dict], source_pos: tuple[int, int]) -> dict:
    sr, sc = int(source_pos[0]), int(source_pos[1])
    return min(
        rules_for_color,
        key=lambda rule: (
            _chebyshev((sr, sc), rule["source_pos"]),
            _chebyshev((sr, sc), rule["anchor"]),
            int(rule["source_pos"][0]),
            int(rule["source_pos"][1]),
            int(rule["target_color"]),
            len(rule["pattern"]),
        ),
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn local legends, then branch by nearest same-colored legend example.

    Input structure:
    - Black background with isolated colored pixels and teal (8) stencil components.
    - A legend example is still a 4-neighbor source->target pair whose direction
      continues to an anchor touching a teal stencil.
    - Unlike the base task, the same source color may appear in two different
      legend examples, each with its own target color and stencil.

    Rule:
    - Recover all legend examples exactly as before.
    - For each non-legend source pixel, only compare legends with the same source
      color and choose the nearest legend source cell.
    - Translate that legend's teal stencil so its anchor lands on the source
      pixel, recolor the stencil to the chosen legend's target color, and erase
      every legend pixel and teal stencil from the final grid.

    Key insight: Color matching no longer determines the branch by itself; when a
    source color has multiple legends, distance to the nearest same-color legend
    decides which target color and stencil should be used.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    pattern_color = 8
    h, w = g.shape

    rules = _discover_rules(g, bg=bg, pattern_color=pattern_color)
    rules_by_source: dict[int, list[dict]] = defaultdict(list)
    legend_pixels: set[tuple[int, int]] = set()
    source_colors: set[int] = set()
    for rule in rules:
        rules_by_source[int(rule["source_color"])].append(rule)
        legend_pixels.add(rule["source_pos"])
        legend_pixels.add(rule["target_pos"])
        source_colors.add(int(rule["source_color"]))

    out = np.full_like(g, int(bg))

    for r, c in map(tuple, np.argwhere((g != int(bg)) & (g != pattern_color))):
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        if (r, c) in legend_pixels or color in source_colors:
            continue
        out[r, c] = color

    for sr, sc in map(tuple, np.argwhere(np.isin(g, list(source_colors)))):
        sr = int(sr)
        sc = int(sc)
        if (sr, sc) in legend_pixels:
            continue
        source_color = int(g[sr, sc])
        chosen_rule = _choose_rule_for_source(rules_by_source[source_color], (sr, sc))
        ar, ac = chosen_rule["anchor"]
        pattern = chosen_rule["pattern"]
        dr = sr - ar
        dc = sc - ac
        rows = pattern[:, 0] + dr
        cols = pattern[:, 1] + dc
        ok = (0 <= rows) & (rows < h) & (0 <= cols) & (cols < w)
        out[rows[ok], cols[ok]] = chosen_rule["target_color"]

    return out


# --------------- generator (c) ---------------

# Grid dimensions
GRID_DIM_MAX = 20
MIN_DIM_BY_GROUPS = {1: 10, 2: 16}

# Each source-color family owns exactly two conditional branches.
GROUP_COUNT_WEIGHTS = [1, 1, 2]
BRANCHES_PER_GROUP = 2
NON_BG_NON_TEAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

# Stencil size
STENCIL_MAX_CAP = 12
STENCIL_MIN_AREA_DIV = 100
STENCIL_MAX_AREA_DIV = 25

# Scattered pixels per branch
SCATTERED_PER_BRANCH_MIN = 2
SCATTERED_PER_BRANCH_MAX_CAP = 4
SCATTERED_AREA_DIV = 55

# Placement layout
LEGEND_RETRIES = 600
SCATTERED_RETRIES_PER_PIXEL = 80
BYSTANDER_RETRIES = 80
OUTER_RETRIES = 400
BYSTANDER_MAX = 3
LEGEND_DISTANCE_MARGIN = 2
LOCAL_SCATTER_RADIUS = 5

_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _reserve_chebyshev(reserved: set[tuple[int, int]], cells, radius: int = 1) -> None:
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                reserved.add((int(r) + dr, int(c) + dc))


def _make_stencil(min_pixels: int, max_pixels: int):
    n = random.randint(min_pixels, max_pixels)
    coords = {(0, 0)}
    for _ in range(n - 1):
        frontier = set()
        for r, c in coords:
            for dr, dc in _DIRS8:
                p = (r + dr, c + dc)
                if p not in coords:
                    frontier.add(p)
        coords.add(random.choice(list(frontier)))
    coords_list = sorted(coords)
    return coords_list, random.choice(coords_list)


def _build_groups(num_groups: int, area: int):
    colors = NON_BG_NON_TEAL_COLORS[:]
    random.shuffle(colors)

    max_s = min(STENCIL_MAX_CAP, max(2, area // STENCIL_MAX_AREA_DIV))
    min_s = min(max(1, area // STENCIL_MIN_AREA_DIV), max_s)
    sc_max = min(SCATTERED_PER_BRANCH_MAX_CAP, max(SCATTERED_PER_BRANCH_MIN, area // SCATTERED_AREA_DIV))

    groups = []
    used_colors = set()
    for group_id in range(num_groups):
        base = 3 * group_id
        src = colors[base]
        targets = colors[base + 1 : base + 1 + BRANCHES_PER_GROUP]
        used_colors.add(src)
        used_colors.update(targets)
        rel_seen: set[tuple[tuple[int, int], ...]] = set()
        branches = []
        for tgt in targets:
            rel_key = None
            rel = None
            for _ in range(40):
                stencil, anchor = _make_stencil(min_s, max_s)
                candidate = tuple(sorted((int(r - anchor[0]), int(c - anchor[1])) for r, c in stencil))
                if candidate not in rel_seen:
                    rel_seen.add(candidate)
                    rel_key = candidate
                    rel = [tuple(x) for x in candidate]
                    break
            if rel is None or rel_key is None:
                return None, None
            branches.append(
                {
                    "group_id": group_id,
                    "src": int(src),
                    "tgt": int(tgt),
                    "rel": rel,
                    "n_sc": random.randint(SCATTERED_PER_BRANCH_MIN, sc_max),
                    "scattered": [],
                }
            )
        groups.append({"src": int(src), "branches": branches})
    return groups, used_colors


def _flatten_branches(groups) -> list[dict]:
    return [branch for group in groups for branch in group["branches"]]


def _place_legends(groups, inp, occupied, reserved):
    h, w = inp.shape
    branches = _flatten_branches(groups)
    random.shuffle(branches)
    min_sep = max(4, min(h, w) // 3)

    for branch in branches:
        placed = False
        same_group = [b for b in groups[branch["group_id"]]["branches"] if b is not branch and "source_pos" in b]
        for _ in range(LEGEND_RETRIES):
            dr, dc = random.choice(_DIRS4)
            sr, sc = random.randint(0, h - 1), random.randint(0, w - 1)
            tr, tc = sr + dr, sc + dc
            ar, ac = sr + 2 * dr, sc + 2 * dc
            if not (0 <= tr < h and 0 <= tc < w and 0 <= ar < h and 0 <= ac < w):
                continue

            teal_cells = {(ar + rr, ac + cc) for rr, cc in branch["rel"]}
            if any(not (0 <= nr < h and 0 <= nc < w) for nr, nc in teal_cells):
                continue

            if same_group and any(_chebyshev((sr, sc), b["source_pos"]) < min_sep for b in same_group):
                continue

            all_cells = {(sr, sc), (tr, tc), (ar, ac)} | teal_cells
            if all_cells & (occupied | reserved):
                continue

            inp[sr, sc] = branch["src"]
            inp[tr, tc] = branch["tgt"]
            for nr, nc in teal_cells:
                inp[nr, nc] = 8

            branch["source_pos"] = (int(sr), int(sc))
            branch["target_pos"] = (int(tr), int(tc))
            branch["anchor"] = (int(ar), int(ac))
            branch["teal_cells"] = teal_cells
            occupied.update(all_cells)
            _reserve_chebyshev(reserved, all_cells, radius=1)
            placed = True
            break
        if not placed:
            return False
    return True


def _candidate_footprint(branch: dict, pos: tuple[int, int], shape: tuple[int, int]) -> set[tuple[int, int]] | None:
    h, w = shape
    sr, sc = pos
    cells = {(sr + rr, sc + cc) for rr, cc in branch["rel"]}
    if any(not (0 <= r < h and 0 <= c < w) for r, c in cells):
        return None
    return cells


def _place_scattered(groups, inp, occupied, reserved):
    h, w = inp.shape
    output_occupied: set[tuple[int, int]] = set()

    for group in groups:
        branches = group["branches"]
        for branch in branches:
            need = int(branch["n_sc"])
            tries = need * SCATTERED_RETRIES_PER_PIXEL
            while len(branch["scattered"]) < need and tries > 0:
                tries -= 1
                base_r, base_c = branch["source_pos"]
                r = base_r + random.randint(-LOCAL_SCATTER_RADIUS, LOCAL_SCATTER_RADIUS)
                c = base_c + random.randint(-LOCAL_SCATTER_RADIUS, LOCAL_SCATTER_RADIUS)
                if not (0 <= r < h and 0 <= c < w):
                    continue
                if (r, c) in occupied or (r, c) in reserved:
                    continue

                d_self = _chebyshev((r, c), branch["source_pos"])
                other_ds = [_chebyshev((r, c), other["source_pos"]) for other in branches if other is not branch]
                if any(d_self + LEGEND_DISTANCE_MARGIN > d_other for d_other in other_ds):
                    continue

                footprint = _candidate_footprint(branch, (r, c), inp.shape)
                if footprint is None:
                    continue
                if footprint & output_occupied:
                    continue
                if footprint & occupied:
                    continue

                branch["scattered"].append((int(r), int(c)))
                inp[r, c] = branch["src"]
                occupied.add((r, c))
                output_occupied.update(footprint)
                _reserve_chebyshev(reserved, [(r, c)], radius=1)

            if len(branch["scattered"]) < SCATTERED_PER_BRANCH_MIN:
                return None
    return output_occupied


def _place_bystanders(used_colors, output_occupied, inp, occupied, reserved):
    bystander_colors = [c for c in NON_BG_NON_TEAL_COLORS if c not in used_colors]
    if not bystander_colors:
        return []

    h, w = inp.shape
    n = random.randint(0, min(BYSTANDER_MAX, len(bystander_colors)))
    placed = []
    for _ in range(n):
        for _ in range(BYSTANDER_RETRIES):
            r, c = random.randint(0, h - 1), random.randint(0, w - 1)
            if (r, c) in occupied or (r, c) in reserved or (r, c) in output_occupied:
                continue
            color = random.choice(bystander_colors)
            inp[r, c] = color
            placed.append((int(r), int(c), int(color)))
            occupied.add((r, c))
            _reserve_chebyshev(reserved, [(r, c)], radius=1)
            break
    return placed


def _build_output(groups, bystanders, h: int, w: int):
    out = np.zeros((h, w), dtype=int)
    for r, c, color in bystanders:
        out[r, c] = color
    for group in groups:
        for branch in group["branches"]:
            for sr, sc in branch["scattered"]:
                for rr, cc in branch["rel"]:
                    out[sr + rr, sc + cc] = branch["tgt"]
    return out


def _gen_attempt():
    num_groups = random.choice(GROUP_COUNT_WEIGHTS)
    min_dim = MIN_DIM_BY_GROUPS[num_groups]
    h = random.randint(min_dim, GRID_DIM_MAX)
    w = random.randint(min_dim, GRID_DIM_MAX)

    groups, used_colors = _build_groups(num_groups, h * w)
    if groups is None or used_colors is None:
        return None

    inp = np.zeros((h, w), dtype=int)
    occupied: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()

    if not _place_legends(groups, inp, occupied, reserved):
        return None
    output_occupied = _place_scattered(groups, inp, occupied, reserved)
    if output_occupied is None:
        return None
    bystanders = _place_bystanders(used_colors, output_occupied, inp, occupied, reserved)

    out = _build_output(groups, bystanders, h, w)

    # Variant guard: every source-color family must genuinely branch.
    for group in groups:
        targets = {branch["tgt"] for branch in group["branches"]}
        counts = [len(branch["scattered"]) for branch in group["branches"]]
        if len(targets) < 2 or min(counts) < SCATTERED_PER_BRANCH_MIN:
            return None

    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {OUTER_RETRIES} attempts")
