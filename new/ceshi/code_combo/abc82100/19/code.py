# Combo variant of abc82100 (index 19)
# combo_spec: D1/4+D9/1
# Name: 同色多目标条件分支+频率消歧

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import components_mask, get_bg


# Design analysis for the combo variant
# -------------------------------------
# concrete_behavior:
#   - Each sample contains exactly two source-color families.
#   - Each family has two legends with the same source color but different
#     target colors and different teal stencils: one visibly small stencil and
#     one visibly large stencil.
#   - Scattered pixels keep the shared source color. They do not branch by
#     proximity. Instead, we count how many times each source color appears in
#     the whole input grid. The higher-frequency source color uses its large
#     branch everywhere; the lower-frequency source color uses its small branch
#     everywhere.
#   - Example trace:
#       family A: blue has a blue->red small legend and a blue->yellow large
#       legend. family B: green has a green->pink small legend and a
#       green->orange large legend.
#       If blue appears 8 times in the grid and green appears 5 times, then all
#       scattered blue pixels stamp the yellow large stencil, while all
#       scattered green pixels stamp the pink small stencil.
#       This differs from the base task because one source color no longer has a
#       unique target, and it also differs from the D9-only nearest-legend
#       variant because spatial proximity does not decide the branch.
# complexity_calibration:
#   - The branch count is fixed at two per family and the global frequency split
#     is fixed at "higher" versus "lower". This keeps the inference space small
#     enough for ARC while making the fused behavior visually obvious.
#   - Large and small stencils use separated size bands, and both branches are
#     shown in every sample because one family is high-frequency and the other
#     is low-frequency.
#   - Each family gets multiple scattered pixels, with at least one pixel closer
#     to each legend. This prevents the puzzle from collapsing into a
#     nearest-legend explanation.
# hidden_assumptions_checked:
#   - The original solver assumed "one source color => one rule" and stamped
#     every rule onto every pixel of that color. That is invalid now, so the
#     solver first groups legends by source color, then selects a single branch
#     per family from global frequency before stamping.
#   - The D9-only nearest-legend assumption is also invalid. The generator
#     intentionally places some scattered pixels closer to the unchosen legend,
#     and it rejects any sample whose output would still match the
#     nearest-legend baseline.
#   - Bystanders still stay safe only because their colors never overlap any
#     source color; the generator preserves that guarantee.
# placement_legality:
#   - Legends keep a Chebyshev safety halo so an anchor cannot be claimed by the
#     wrong teal component.
#   - Scattered pixels are isolated from legends and from each other, and output
#     footprints are kept fully in bounds and non-overlapping to avoid
#     order-dependent overwrites.
#   - Same-family legends are separated enough that we can place scattered
#     pixels clearly closer to either branch.
# observability_and_cleanliness:
#   - The same input color visibly offers two legend targets. The output shows
#     only one of them, and the deciding cue is the count of that source color
#     in the whole grid.
#   - Because some scattered pixels sit nearer to the unused legend, a human can
#     see that distance alone is insufficient. The count difference is the only
#     stable visible discriminator.
#   - Bystander usage is kept small, stencil footprints do not overlap, and the
#     large/small size contrast keeps the grid readable.
# inferability:
#   - The rule never depends on numeric color ids. Only visible properties are
#     used: same-color legend grouping, whole-grid frequency, and stencil size.
#   - Across examples, which actual color becomes "high-frequency" changes, so a
#     solver cannot memorize a fixed color-to-size mapping.
# solver_assumptions_audit:
#   - _discover_rules assumes each legend is still a 4-neighbor source->target
#     pair pointing toward a nearby teal component; the generator keeps that
#     structure unchanged.
#   - main() assumes each source family has at least two distinguishable stencil
#     sizes so the frequency-selected branch is well-defined; the generator uses
#     disjoint size bands for that.
#   - main() uses a deterministic rank fallback for tied frequencies, but the
#     generator guarantees unequal source-color counts so no hidden tie-break is
#     needed in valid samples.


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


def _sorted_rules_by_size(rules_for_color: list[dict]) -> list[dict]:
    return sorted(
        rules_for_color,
        key=lambda rule: (
            len(rule["pattern"]),
            int(rule["target_color"]),
            int(rule["source_pos"][0]),
            int(rule["source_pos"][1]),
        ),
    )


def _choose_rule_for_color_frequency(
    source_color: int,
    rules_by_source: dict[int, list[dict]],
    source_counts: dict[int, int],
) -> dict:
    colors = sorted(
        rules_by_source,
        key=lambda color: (-int(source_counts[color]), int(color)),
    )
    size_sorted = _sorted_rules_by_size(rules_by_source[source_color])
    if len(colors) == 1:
        return size_sorted[-1]

    freq_rank = colors.index(int(source_color))
    max_rank = len(colors) - 1
    branch_index = int(round((max_rank - freq_rank) * (len(size_sorted) - 1) / max_rank))
    return size_sorted[branch_index]


def _choose_nearest_rule_for_source(rules_for_color: list[dict], source_pos: tuple[int, int]) -> dict:
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


def _render_output(
    g: np.ndarray,
    bg: int,
    rules: list[dict],
    chooser,
) -> np.ndarray:
    pattern_color = 8
    h, w = g.shape
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
        chosen_rule = chooser(rules_by_source[source_color], (sr, sc))
        ar, ac = chosen_rule["anchor"]
        pattern = chosen_rule["pattern"]
        dr = sr - ar
        dc = sc - ac
        rows = pattern[:, 0] + dr
        cols = pattern[:, 1] + dc
        ok = (0 <= rows) & (rows < h) & (0 <= cols) & (cols < w)
        out[rows[ok], cols[ok]] = chosen_rule["target_color"]

    return out


def _nearest_legend_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    rules = _discover_rules(g, bg=bg, pattern_color=8)
    return _render_output(g, bg, rules, lambda rules_for_color, pos: _choose_nearest_rule_for_source(rules_for_color, pos))


def _combo_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    rules = _discover_rules(g, bg=bg, pattern_color=8)
    chosen_by_source = _chosen_rule_by_source(g, rules)
    return _render_output(
        g,
        bg,
        rules,
        lambda rules_for_color, pos: chosen_by_source[int(rules_for_color[0]["source_color"])],
    )


def _chosen_rule_by_source(grid: np.ndarray, rules: list[dict]) -> dict[int, dict]:
    source_counts = {int(color): int(np.count_nonzero(grid == int(color))) for color in {rule["source_color"] for rule in rules}}
    rules_by_source: dict[int, list[dict]] = defaultdict(list)
    for rule in rules:
        rules_by_source[int(rule["source_color"])].append(rule)
    return {
        color: _choose_rule_for_color_frequency(int(color), rules_by_source, source_counts)
        for color in rules_by_source
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn paired same-color legends, then use global source-color frequency to
    choose which branch survives for each family.

    Input structure:
    - The grid still contains isolated colored pixels on a black background and
      teal (8) stencil components attached to local legend examples.
    - A legend is still a 4-neighbor source->target pair whose direction
      continues to an anchor touching the teal stencil.
    - Unlike the base task, a source color can appear in multiple legends with
      different target colors and different stencil sizes, typically one small
      stencil and one large stencil.

    Rule:
    - Recover all legends and group them by source color.
    - Count how often each source color appears in the whole input grid. For a
      high-frequency source color, choose that family's largest stencil rule; for
      a low-frequency source color, choose that family's smallest stencil rule.
    - Stamp only the frequency-selected stencil onto every non-legend pixel of
      that source color, recolor it to the selected target color, and erase all
      legend pixels plus every teal stencil from the final output.

    Key insight: Matching by color is no longer enough, and proximity is also
    insufficient. The decisive clue is the whole-grid frequency of each source
    color, which selects between multiple same-colored legend branches.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    rules = _discover_rules(g, bg=bg, pattern_color=8)
    chosen_by_source = _chosen_rule_by_source(g, rules)
    return _render_output(
        g,
        bg,
        rules,
        lambda rules_for_color, pos: chosen_by_source[int(rules_for_color[0]["source_color"])],
    )


# --------------- generator (combo) ---------------

GRID_DIM_MIN = 14
GRID_DIM_MAX = 20
GROUP_COUNT = 2
BRANCHES_PER_GROUP = 2
NON_BG_NON_TEAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

SMALL_STENCIL_RANGE = (1, 3)
LARGE_STENCIL_RANGE = (5, 8)

HIGH_SCATTER_RANGE = (6, 8)
LOW_SCATTER_RANGE = (3, 4)

LEGEND_RETRIES = 700
SCATTERED_RETRIES_PER_PIXEL = 140
BYSTANDER_RETRIES = 80
OUTER_RETRIES = 500
BYSTANDER_MAX = 2

SAME_GROUP_MIN_SEP = 5
LOCAL_SCATTER_RADIUS = 5
NEAR_MARGIN = 1

_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _reserve_chebyshev(reserved: set[tuple[int, int]], cells, radius: int = 1) -> None:
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                reserved.add((int(r) + dr, int(c) + dc))


def _make_connected_stencil(min_pixels: int, max_pixels: int):
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
    anchor = random.choice(coords_list)
    rel = tuple(sorted((int(r - anchor[0]), int(c - anchor[1])) for r, c in coords_list))
    return rel


def _make_size_separated_stencils() -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | tuple[None, None]:
    for _ in range(60):
        small = _make_connected_stencil(*SMALL_STENCIL_RANGE)
        large = _make_connected_stencil(*LARGE_STENCIL_RANGE)
        if small != large and len(small) < len(large):
            return [tuple(x) for x in small], [tuple(x) for x in large]
    return None, None


def _build_groups():
    colors = NON_BG_NON_TEAL_COLORS[:]
    random.shuffle(colors)
    groups = []
    used_colors = set()
    roles = ["high", "low"]
    random.shuffle(roles)

    for group_id in range(GROUP_COUNT):
        base = 3 * group_id
        src = int(colors[base])
        tgt_a = int(colors[base + 1])
        tgt_b = int(colors[base + 2])
        small_rel, large_rel = _make_size_separated_stencils()
        if small_rel is None or large_rel is None:
            return None, None

        branches = [
            {"kind": "small", "src": src, "tgt": tgt_a, "rel": small_rel},
            {"kind": "large", "src": src, "tgt": tgt_b, "rel": large_rel},
        ]
        chosen_kind = "large" if roles[group_id] == "high" else "small"
        n_sc = random.randint(*(HIGH_SCATTER_RANGE if roles[group_id] == "high" else LOW_SCATTER_RANGE))
        groups.append(
            {
                "group_id": group_id,
                "src": src,
                "role": roles[group_id],
                "chosen_kind": chosen_kind,
                "n_sc": n_sc,
                "branches": branches,
                "scattered": [],
            }
        )
        used_colors.update([src, tgt_a, tgt_b])

    counts = sorted(group["n_sc"] + BRANCHES_PER_GROUP for group in groups)
    if counts[0] == counts[1]:
        return None, None
    return groups, used_colors


def _flatten_branches(groups) -> list[dict]:
    return [branch for group in groups for branch in group["branches"]]


def _place_legends(groups, inp, occupied, reserved):
    h, w = inp.shape
    branches = _flatten_branches(groups)
    random.shuffle(branches)

    for branch in branches:
        group = groups[int(branch["group_id"])]
        placed = False
        same_group_existing = [b for b in group["branches"] if b is not branch and "source_pos" in b]
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

            if same_group_existing:
                other = same_group_existing[0]
                if _chebyshev((sr, sc), other["source_pos"]) < SAME_GROUP_MIN_SEP:
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


def _branch_by_kind(group: dict, kind: str) -> dict:
    for branch in group["branches"]:
        if branch["kind"] == kind:
            return branch
    raise KeyError(kind)


def _chosen_branch(group: dict) -> dict:
    return _branch_by_kind(group, str(group["chosen_kind"]))


def _other_branch(group: dict) -> dict:
    return _branch_by_kind(group, "small" if group["chosen_kind"] == "large" else "large")


def _candidate_footprint(branch: dict, pos: tuple[int, int], shape: tuple[int, int]) -> set[tuple[int, int]] | None:
    h, w = shape
    sr, sc = pos
    cells = {(int(sr) + rr, int(sc) + cc) for rr, cc in branch["rel"]}
    if any(not (0 <= r < h and 0 <= c < w) for r, c in cells):
        return None
    return cells


def _is_closer_to(pos: tuple[int, int], winner: dict, loser: dict) -> bool:
    d_win = _chebyshev(pos, winner["source_pos"])
    d_lose = _chebyshev(pos, loser["source_pos"])
    return d_win + NEAR_MARGIN <= d_lose


def _try_place_scatter(
    group: dict,
    near_branch: dict,
    chosen_branch: dict,
    other_branch: dict,
    inp: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
    output_occupied: set[tuple[int, int]],
) -> tuple[int, int] | None:
    h, w = inp.shape
    br, bc = near_branch["source_pos"]
    for _ in range(SCATTERED_RETRIES_PER_PIXEL):
        r = br + random.randint(-LOCAL_SCATTER_RADIUS, LOCAL_SCATTER_RADIUS)
        c = bc + random.randint(-LOCAL_SCATTER_RADIUS, LOCAL_SCATTER_RADIUS)
        if not (0 <= r < h and 0 <= c < w):
            continue
        pos = (int(r), int(c))
        if pos in occupied or pos in reserved:
            continue
        if not _is_closer_to(pos, near_branch, other_branch if near_branch is chosen_branch else chosen_branch):
            continue

        footprint = _candidate_footprint(chosen_branch, pos, inp.shape)
        if footprint is None:
            continue
        if footprint & output_occupied:
            continue
        if footprint & occupied:
            continue

        inp[r, c] = group["src"]
        occupied.add(pos)
        output_occupied.update(footprint)
        _reserve_chebyshev(reserved, [pos], radius=1)
        return pos
    return None


def _place_scattered(groups, inp, occupied, reserved):
    output_occupied: set[tuple[int, int]] = set()
    for group in groups:
        chosen = _chosen_branch(group)
        other = _other_branch(group)

        mandatory_near = [chosen, other]
        random.shuffle(mandatory_near)
        for near_branch in mandatory_near:
            pos = _try_place_scatter(group, near_branch, chosen, other, inp, occupied, reserved, output_occupied)
            if pos is None:
                return None
            group["scattered"].append(pos)

        while len(group["scattered"]) < int(group["n_sc"]):
            near_branch = random.choice([chosen, other])
            pos = _try_place_scatter(group, near_branch, chosen, other, inp, occupied, reserved, output_occupied)
            if pos is None:
                return None
            group["scattered"].append(pos)

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
            pos = (int(r), int(c))
            if pos in occupied or pos in reserved or pos in output_occupied:
                continue
            color = int(random.choice(bystander_colors))
            inp[r, c] = color
            placed.append((int(r), int(c), color))
            occupied.add(pos)
            _reserve_chebyshev(reserved, [pos], radius=1)
            break
    return placed


def _build_output(groups, bystanders, h: int, w: int):
    out = np.zeros((h, w), dtype=int)
    for r, c, color in bystanders:
        out[r, c] = color
    for group in groups:
        chosen = _chosen_branch(group)
        for sr, sc in group["scattered"]:
            for rr, cc in chosen["rel"]:
                out[sr + rr, sc + cc] = chosen["tgt"]
    return out


def _combo_metrics(inp: np.ndarray):
    rules = _discover_rules(inp, bg=get_bg(inp), pattern_color=8)
    rules_by_source: dict[int, list[dict]] = defaultdict(list)
    for rule in rules:
        rules_by_source[int(rule["source_color"])].append(rule)

    legend_pixels = {rule["source_pos"] for rule in rules} | {rule["target_pos"] for rule in rules}
    source_counts = {color: int(np.count_nonzero(inp == color)) for color in rules_by_source}
    chosen = {
        color: _choose_rule_for_color_frequency(color, rules_by_source, source_counts)
        for color in rules_by_source
    }

    conflict = 0
    total = 0
    for color, rules_for_color in rules_by_source.items():
        for sr, sc in map(tuple, np.argwhere(inp == color)):
            pos = (int(sr), int(sc))
            if pos in legend_pixels:
                continue
            total += 1
            nearest = _choose_nearest_rule_for_source(rules_for_color, pos)
            if nearest["target_color"] != chosen[color]["target_color"] or len(nearest["pattern"]) != len(chosen[color]["pattern"]):
                conflict += 1
    return {
        "source_counts": source_counts,
        "chosen_sizes": {color: len(rule["pattern"]) for color, rule in chosen.items()},
        "conflict_scattered": conflict,
        "total_scattered": total,
    }


def _gen_attempt():
    h = random.randint(GRID_DIM_MIN, GRID_DIM_MAX)
    w = random.randint(GRID_DIM_MIN, GRID_DIM_MAX)

    groups, used_colors = _build_groups()
    if groups is None or used_colors is None:
        return None

    inp = np.zeros((h, w), dtype=int)
    occupied: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()

    for group in groups:
        for branch in group["branches"]:
            branch["group_id"] = int(group["group_id"])

    if not _place_legends(groups, inp, occupied, reserved):
        return None
    output_occupied = _place_scattered(groups, inp, occupied, reserved)
    if output_occupied is None:
        return None
    bystanders = _place_bystanders(used_colors, output_occupied, inp, occupied, reserved)

    out = _build_output(groups, bystanders, h, w)

    discovered = _discover_rules(inp, bg=get_bg(inp), pattern_color=8)
    discovered_by_source: dict[int, list[dict]] = defaultdict(list)
    for rule in discovered:
        discovered_by_source[int(rule["source_color"])].append(rule)
    if len(discovered) != GROUP_COUNT * BRANCHES_PER_GROUP:
        return None
    if sorted(len(rules_for_color) for rules_for_color in discovered_by_source.values()) != [BRANCHES_PER_GROUP] * GROUP_COUNT:
        return None

    counts = sorted(int(np.count_nonzero(inp == group["src"])) for group in groups)
    if counts[0] == counts[1]:
        return None

    combo_metrics = _combo_metrics(inp)
    if combo_metrics["conflict_scattered"] < GROUP_COUNT:
        return None

    try:
        combo_out = main(inp)
        nearest_out = _nearest_legend_output(inp)
        if np.array_equal(combo_out, out) and not np.array_equal(nearest_out, out):
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
