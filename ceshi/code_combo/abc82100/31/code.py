# Combo variant of abc82100 (index 31)
# combo_spec: D1/2+D3/0+D6/1
# Name: 局部环境决定目标色+多像素色块+多属性匹配
#
# Design analysis for the fused variant
# ------------------------------------
# a) Concrete behavior and trace
#    - The legend still uses the original local mechanism: a source pixel, an
#      adjacent differently colored guide pixel, and a teal stencil in the same
#      forward direction. That teaches which stencil belongs to a source color.
#    - Free-standing sources are no longer isolated pixels. Each source object is
#      a 4-connected monochrome block of size >= 2, and each block triggers one
#      stamp aligned to its top-left occupied cell.
#    - The block anchor's 4-neighborhood now does two jobs at once:
#      1. matching: the total number of non-background 4-neighbors at the anchor
#         must equal the legend source's count, so rules are keyed by
#         (source_color, anchor_neighbor_count);
#      2. recoloring: among those same non-background neighbors, the unique most
#         frequent color becomes the stamp color for that particular block.
#    - To make this visible and stable, every source block is chosen from a shape
#      family whose top-left anchor has exactly one internal same-color neighbor.
#      The remaining anchor neighbors are external colored support pixels.
#      Count classes are therefore only 3 or 4:
#      * count 3 = {1 internal source neighbor, 2 support pixels of one color}
#      * count 4 = {1 internal source neighbor, 2 support pixels of winner color,
#                   1 support pixel of a minority color}
#    - The legend mirrors that structure. Because the legend source is only one
#      pixel, we add one same-color helper next to it to stand in for the block's
#      internal neighbor. The guide pixel is one of the winner-colored neighbors,
#      and optional extra support pixels complete count 3 or 4.
#    - Trace example:
#      a red rule of count 3 has legend neighborhood {red helper, yellow guide,
#      yellow support}. A free red domino whose top-left cell touches two yellow
#      supports also has count 3, so it matches and stamps its teal stencil in
#      yellow. Another red block with neighborhood {red internal, green, green,
#      maroon} has count 4 instead, so it matches the red count-4 rule and stamps
#      in green. The original task would have ignored both the block geometry and
#      the neighborhood split.
#
# b) Complexity calibration
#    - Neighbor counts are restricted to {3, 4}. This is small enough for humans
#      to read but still enough to split one source color into two rules.
#    - Source colors come from a tiny palette, and the generator always creates at
#      least one repeated source color across different counts. That makes the
#      extra match condition necessary rather than decorative.
#    - Vote colors come from a separate small palette and at least one rule is
#      forced to use two different majority colors across its source blocks, so
#      the per-instance recolor cannot collapse back to a fixed target color.
#
# c) Hidden assumptions that break in the original solver
#    - Iterating over every pixel of a source color is no longer valid: one block
#      must produce one stamp, not one stamp per cell.
#    - Source color alone no longer determines a rule, so rule discovery must
#      store the legend source's anchor-neighbor count as part of the key.
#    - A fixed legend target color is no longer meaningful. The guide pixel still
#      helps locate the stencil, but the actual output color must be recomputed
#      from each block anchor's neighborhood.
#    - Legend-local helper/support pixels now exist. If they are not removed from
#      consideration, the solver would mistake them for free-standing sources or
#      preserve them in the output.
#
# d) Placement legality
#    - Legends, source blocks, support pixels, bystanders, and teal stencils are
#      all placed on disjoint footprints with a reserved 4-neighborhood margin.
#      This prevents accidental merges, unstable counts, and false local rules.
#    - Every source-block anchor is kept strictly away from the border so all four
#      anchor neighbors exist and the count/majority logic is unambiguous.
#    - Teal legend cells are forbidden from occupying the legend source, the guide,
#      or any other 4-neighbor of the legend source, because those cells encode
#      the count and majority evidence.
#
# e) Visual observability
#    - Humans can see three independent cues in a few examples:
#      * source objects are obviously blobs, not single pixels;
#      * the same source color appears with two distinct local counts;
#      * blocks of the same rule can stamp in different colors when their local
#        repeated support color changes.
#    - The guide pixel in the legend is only a local witness of majority color,
#      not a global target. Seeing the same source color produce multiple output
#      colors makes that visible.
#
# f) Visual cleanliness
#    - Source blocks are small and sparse, and support pixels only touch the block
#      anchor. This keeps the neighborhoods legible instead of creating dense
#      carpets of helper pixels.
#    - Legend helper pixels are removed in the output so the example area does not
#      leave distracting debris.
#
# g) Color inference
#    - There is no hidden use of numeric color ids. Source colors identify rule
#      ownership, while the visible local majority around each block anchor picks
#      the output color.
#
# Review of new solver assumptions and generator guarantees
#    - Assumption: each non-legend source component has a recoverable top-left
#      anchor with exactly one internal same-color 4-neighbor.
#      Guarantee: source shapes come from a curated family with that property.
#    - Assumption: each (source_color, neighbor_count) pair identifies at most one
#      rule.
#      Guarantee: the generator samples unique keys and validates discovery.
#    - Assumption: the anchor majority is unique for every consumable block.
#      Guarantee: support colors are placed in a 2-vs-1 or 2-vs-0 pattern.
#    - Assumption: the legend's same-color helper is not a real source block.
#      Guarantee: solver explicitly excludes same-color neighbors of legend source
#      cells from component extraction, and generator keeps legends isolated.

from __future__ import annotations

import random
import sys

import numpy as np

SOLVER_UTILS = "/home/lixinyu/snap/new/solvers"
if SOLVER_UTILS not in sys.path:
    sys.path.append(SOLVER_UTILS)

from arc_utils import components_mask, get_bg


GRID_DIM_MAX = 20
MIN_DIM_BY_RULES = {2: 12}

RULE_COUNT_WEIGHTS = [2]
SOURCE_COLORS = [1, 2, 3]
VOTE_COLORS = [4, 5, 6, 7, 9]
PATTERN_COLOR = 8
COUNT_CLASSES = [3, 4]

STENCIL_MAX_CAP = 12
STENCIL_MIN_AREA_DIV = 100
STENCIL_MAX_AREA_DIV = 25

SOURCE_BLOCK_MIN = 3
SOURCE_BLOCK_MAX_CAP = 4
SOURCE_BLOCK_AREA_DIV = 75

BYSTANDER_MAX = 2

LEGEND_RETRIES = 700
SOURCE_BLOCK_RETRIES_PER_BLOCK = 120
BYSTANDER_RETRIES = 80
OUTER_RETRIES = 500

_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

# Every shape is normalized to top-left anchor (0,0) and gives that anchor
# exactly one internal same-color 4-neighbor.
_SOURCE_SHAPES = [
    [(0, 0), (0, 1)],
    [(0, 0), (1, 0)],
    [(0, 0), (0, 1), (0, 2)],
    [(0, 0), (1, 0), (2, 0)],
    [(0, 0), (0, 1), (1, 1)],
    [(0, 0), (1, 0), (1, 1)],
    [(0, 0), (0, 1), (0, 2), (1, 2)],
    [(0, 0), (1, 0), (2, 0), (2, 1)],
]


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _anchor_distance(pattern: np.ndarray, anchor: tuple[int, int]) -> int:
    return min(_chebyshev((int(r), int(c)), anchor) for r, c in pattern)


def _count_non_bg4(grid: np.ndarray, r: int, c: int, bg: int) -> int:
    h, w = grid.shape
    total = 0
    for dr, dc in _DIRS4:
        nr = int(r + dr)
        nc = int(c + dc)
        if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) != int(bg):
            total += 1
    return total


def _non_bg_neighbors4(grid: np.ndarray, r: int, c: int, bg: int) -> list[tuple[int, int]]:
    h, w = grid.shape
    out: list[tuple[int, int]] = []
    for dr, dc in _DIRS4:
        nr = int(r + dr)
        nc = int(c + dc)
        if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) != int(bg):
            out.append((nr, nc))
    return out


def _majority_non_bg_neighbor_color(grid: np.ndarray, r: int, c: int, bg: int) -> int | None:
    counts: dict[int, int] = {}
    for nr, nc in _non_bg_neighbors4(grid, r, c, bg):
        color = int(grid[nr, nc])
        counts[color] = counts.get(color, 0) + 1
    if not counts:
        return None
    best = max(counts.values())
    winners = [color for color, count in counts.items() if count == best]
    if len(winners) != 1:
        return None
    return int(winners[0])


def _component_anchor(component: np.ndarray) -> tuple[int, int]:
    return min((int(r), int(c)) for r, c in component)


def _discover_rules(grid: np.ndarray, bg: int, pattern_color: int = PATTERN_COLOR) -> list[dict]:
    h, w = grid.shape
    patterns = components_mask(grid == int(pattern_color), connectivity=8)
    rules: list[dict] = []
    seen: set[tuple[tuple[int, int], tuple[int, int], int]] = set()

    for r in range(h):
        for c in range(w):
            source_color = int(grid[r, c])
            if source_color in (int(bg), int(pattern_color)):
                continue

            for dr, dc in _DIRS4:
                tr = int(r + dr)
                tc = int(c + dc)
                if not (0 <= tr < h and 0 <= tc < w):
                    continue

                guide_color = int(grid[tr, tc])
                if guide_color in (int(bg), int(pattern_color)) or guide_color == source_color:
                    continue

                anchor = (int(tr + dr), int(tc + dc))
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
                        "neighbor_count": _count_non_bg4(grid, int(r), int(c), int(bg)),
                        "guide_color": int(guide_color),
                        "source_pos": (int(r), int(c)),
                        "guide_pos": (int(tr), int(tc)),
                        "anchor": (int(anchor[0]), int(anchor[1])),
                        "pattern": np.asarray(pattern, dtype=int),
                    }
                )

    return rules


def _legend_exclusions(
    grid: np.ndarray, rules: list[dict], bg: int
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[int]]:
    legend_pixels: set[tuple[int, int]] = set()
    legend_context: set[tuple[int, int]] = set()
    source_colors: set[int] = set()
    for rule in rules:
        src_pos = tuple(rule["source_pos"])
        guide_pos = tuple(rule["guide_pos"])
        source_color = int(rule["source_color"])
        legend_pixels.add(src_pos)
        legend_pixels.add(guide_pos)
        source_colors.add(source_color)
        for nr, nc in _non_bg_neighbors4(grid, src_pos[0], src_pos[1], int(bg)):
            if (nr, nc) != guide_pos:
                legend_context.add((int(nr), int(nc)))
    return legend_pixels, legend_context, source_colors


def _source_components(
    grid: np.ndarray,
    source_color: int,
    legend_pixels: set[tuple[int, int]],
    legend_context: set[tuple[int, int]],
) -> list[np.ndarray]:
    mask = grid == int(source_color)
    if legend_pixels or legend_context:
        mask = mask.copy()
        for r, c in legend_pixels | legend_context:
            if 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1]:
                mask[r, c] = False
    return components_mask(mask, connectivity=4)


def _matched_blocks(grid: np.ndarray, rules: list[dict], bg: int) -> list[dict]:
    legend_pixels, legend_context, _source_colors = _legend_exclusions(grid, rules, bg)
    matched: list[dict] = []
    for rule in rules:
        wanted_color = int(rule["source_color"])
        wanted_count = int(rule["neighbor_count"])
        for component in _source_components(grid, wanted_color, legend_pixels, legend_context):
            anchor = _component_anchor(component)
            if _count_non_bg4(grid, anchor[0], anchor[1], int(bg)) != wanted_count:
                continue
            majority = _majority_non_bg_neighbor_color(grid, anchor[0], anchor[1], int(bg))
            if majority is None:
                continue
            matched.append(
                {
                    "rule": rule,
                    "component": np.asarray(component, dtype=int),
                    "anchor_cell": anchor,
                    "target_color": int(majority),
                }
            )
    return matched


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn stencil ownership from local source-guide-pattern examples, then apply
    each stencil once per matching source block using the block anchor's context.

    Input structure:
    - Black background with teal (8) stencil components, legend source->guide
      pairs, extra legend helper pixels, and free-standing monochrome source
      blocks elsewhere in the grid.
    - Each source block is a 4-connected shape of size at least 2. Its top-left
      occupied cell acts as the block anchor.
    - The legend source and each block anchor have a visible local 4-neighborhood
      whose total non-background count is either 3 or 4, and whose majority color
      is shown by repeated neighboring pixels.

    Rule:
    - Discover a rule from each legend by reading the source color, the nearby
      teal stencil, and the legend source's non-background 4-neighbor count.
    - For every non-legend source block, look at its anchor cell. The block uses
      the rule whose source color matches and whose anchor-neighbor count matches.
    - Copy that rule's teal stencil once onto the block anchor, and recolor the
      copied stencil with the unique most frequent non-background color among the
      anchor's four neighbors. Remove the legend area, all original teal cells,
      and the source blocks themselves; preserve unrelated support/bystander
      pixels unless a stamp later covers them.

    Key insight: the same local neighborhood around a source-block anchor must be
    analyzed twice. Its size class chooses which rule applies, and its color vote
    chooses what color that rule's stencil will use for this specific block.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    h, w = g.shape

    rules = _discover_rules(g, bg=bg, pattern_color=PATTERN_COLOR)
    legend_pixels, legend_context, source_colors = _legend_exclusions(g, rules, int(bg))
    matched = _matched_blocks(g, rules, int(bg))

    out = np.full_like(g, int(bg))

    for r, c in map(tuple, np.argwhere((g != int(bg)) & (g != int(PATTERN_COLOR)))):
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        if (r, c) in legend_pixels or (r, c) in legend_context or color in source_colors:
            continue
        out[r, c] = color

    for item in matched:
        rule = item["rule"]
        sr, sc = item["anchor_cell"]
        ar, ac = rule["anchor"]
        pattern = rule["pattern"]
        dr = int(sr - ar)
        dc = int(sc - ac)
        rows = pattern[:, 0] + dr
        cols = pattern[:, 1] + dc
        ok = (0 <= rows) & (rows < h) & (0 <= cols) & (cols < w)
        out[rows[ok], cols[ok]] = int(item["target_color"])

    return out


def _reserve_with_margin(reserved: set[tuple[int, int]], cells) -> None:
    for r, c in cells:
        rr = int(r)
        cc = int(c)
        reserved.add((rr, cc))
        for dr, dc in _DIRS4:
            reserved.add((rr + dr, cc + dc))


def _make_stencil(min_pixels: int, max_pixels: int) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    n = random.randint(min_pixels, max_pixels)
    coords = {(0, 0)}
    for _ in range(n - 1):
        frontier = set()
        for r, c in coords:
            for dr, dc in _DIRS8:
                pos = (int(r + dr), int(c + dc))
                if pos not in coords:
                    frontier.add(pos)
        coords.add(random.choice(list(frontier)))
    coords_list = sorted(coords)
    return coords_list, random.choice(coords_list)


def _build_rules(num_rules: int, area: int) -> list[dict]:
    max_s = min(STENCIL_MAX_CAP, max(2, area // STENCIL_MAX_AREA_DIV))
    min_s = min(max(1, area // STENCIL_MIN_AREA_DIV), max_s)
    block_max = min(SOURCE_BLOCK_MAX_CAP, max(SOURCE_BLOCK_MIN, area // SOURCE_BLOCK_AREA_DIV))

    shared_src = random.choice(SOURCE_COLORS)
    keys = [(shared_src, COUNT_CLASSES[0]), (shared_src, COUNT_CLASSES[1])]
    remaining = [(src, cnt) for src in SOURCE_COLORS for cnt in COUNT_CLASSES if (src, cnt) not in keys]
    random.shuffle(remaining)
    keys.extend(remaining[: num_rules - 2])
    random.shuffle(keys)

    rules: list[dict] = []
    for src_color, neighbor_count in keys:
        stencil, anchor = _make_stencil(min_s, max_s)
        rel = [(int(r - anchor[0]), int(c - anchor[1])) for r, c in stencil]
        rules.append(
            {
                "src": int(src_color),
                "count": int(neighbor_count),
                "rel": rel,
                "n_blocks": random.randint(SOURCE_BLOCK_MIN, block_max),
            }
        )
    return rules


def _pick_vote_palette() -> list[int]:
    k = random.choice([2, 3])
    return random.sample(VOTE_COLORS, k)


def _assign_majority_sequences(rules: list[dict], vote_palette: list[int]) -> None:
    varying_rules = []
    for rule in rules:
        majors = [random.choice(vote_palette) for _ in range(rule["n_blocks"])]
        rule["majority_colors"] = majors
        if len(set(majors)) >= 2:
            varying_rules.append(rule)
    if varying_rules:
        return
    chosen = random.choice(rules)
    chosen["majority_colors"][0] = vote_palette[0]
    chosen["majority_colors"][1] = vote_palette[1]


def _shape_internal_dir(shape: list[tuple[int, int]]) -> tuple[int, int]:
    nbrs = [(int(r), int(c)) for r, c in shape if abs(int(r)) + abs(int(c)) == 1]
    if len(nbrs) != 1:
        raise ValueError("Source shape anchor must have exactly one internal 4-neighbor")
    return nbrs[0]


def _place_legends(
    rules: list[dict],
    vote_palette: list[int],
    inp: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> bool:
    h, w = inp.shape
    for rule in rules:
        rel = rule["rel"]
        placed = False
        for _ in range(LEGEND_RETRIES):
            dr, dc = random.choice(_DIRS4)
            sr = random.randint(1, h - 2)
            sc = random.randint(1, w - 2)
            tr = int(sr + dr)
            tc = int(sc + dc)
            ar = int(sr + 2 * dr)
            ac = int(sc + 2 * dc)
            if not (0 <= ar < h and 0 <= ac < w):
                continue

            teal_cells = set()
            ok = True
            for rr, cc in rel:
                nr = int(ar + rr)
                nc = int(ac + cc)
                if not (0 <= nr < h and 0 <= nc < w):
                    ok = False
                    break
                teal_cells.add((nr, nc))
            if not ok:
                continue

            if (sr, sc) in teal_cells or (tr, tc) in teal_cells:
                continue

            other_dirs = [d for d in _DIRS4 if d != (dr, dc)]
            same_dir = random.choice(other_dirs)
            rem_dirs = [d for d in other_dirs if d != same_dir]
            support_needed = int(rule["count"]) - 2
            if len(rem_dirs) < support_needed:
                continue
            support_dirs = random.sample(rem_dirs, support_needed)

            same_pos = (int(sr + same_dir[0]), int(sc + same_dir[1]))
            support_positions = [(int(sr + rr), int(sc + cc)) for rr, cc in support_dirs]
            context_cells = {same_pos, *support_positions}
            all_source_neighbors = {(int(sr + rr), int(sc + cc)) for rr, cc in _DIRS4}

            if teal_cells & all_source_neighbors:
                continue
            if any(not (0 <= rr < h and 0 <= cc < w) for rr, cc in context_cells):
                continue

            actual_cells = {(int(sr), int(sc)), (int(tr), int(tc))} | context_cells | teal_cells
            if actual_cells & occupied or actual_cells & reserved:
                continue

            majority_color = random.choice(rule["majority_colors"])
            inp[sr, sc] = int(rule["src"])
            inp[tr, tc] = int(majority_color)
            inp[same_pos] = int(rule["src"])
            if int(rule["count"]) == 3:
                inp[support_positions[0]] = int(majority_color)
            else:
                minority = random.choice([c for c in vote_palette if c != majority_color])
                order = support_positions[:]
                random.shuffle(order)
                inp[order[0]] = int(majority_color)
                inp[order[1]] = int(minority)
            for nr, nc in teal_cells:
                inp[nr, nc] = int(PATTERN_COLOR)

            occupied.update(actual_cells)
            _reserve_with_margin(reserved, actual_cells)
            placed = True
            break
        if not placed:
            return False
    return True


def _place_source_blocks(
    rules: list[dict],
    vote_palette: list[int],
    inp: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> bool:
    h, w = inp.shape
    for rule in rules:
        blocks = []
        support_pixels: list[tuple[int, int, int]] = []
        target_count = int(rule["n_blocks"])
        for majority_color in rule["majority_colors"]:
            placed = False
            for _ in range(SOURCE_BLOCK_RETRIES_PER_BLOCK):
                shape = random.choice(_SOURCE_SHAPES)
                max_r = max(r for r, _c in shape)
                max_c = max(c for _r, c in shape)
                ar = random.randint(1, h - max_r - 1)
                ac = random.randint(1, w - max_c - 1)
                cells = {(int(ar + rr), int(ac + cc)) for rr, cc in shape}
                internal_dir = _shape_internal_dir(shape)
                support_dirs = [d for d in _DIRS4 if d != internal_dir]
                need_supports = int(rule["count"]) - 1
                if need_supports == 2:
                    chosen_dirs = random.sample(support_dirs, 2)
                    color_plan = [int(majority_color), int(majority_color)]
                else:
                    chosen_dirs = support_dirs[:]
                    minority = random.choice([c for c in vote_palette if c != majority_color])
                    color_plan = [int(majority_color), int(majority_color), int(minority)]
                    random.shuffle(color_plan)
                support_cells = [(int(ar + dr), int(ac + dc), int(color)) for (dr, dc), color in zip(chosen_dirs, color_plan)]
                footprint = set(cells) | {(r, c) for r, c, _color in support_cells}
                if any(not (0 <= r < h and 0 <= c < w) for r, c in footprint):
                    continue
                if footprint & occupied or footprint & reserved:
                    continue

                for r, c in cells:
                    inp[r, c] = int(rule["src"])
                for r, c, color in support_cells:
                    inp[r, c] = int(color)

                occupied.update(footprint)
                _reserve_with_margin(reserved, footprint)
                blocks.append({"cells": sorted(cells), "anchor": (int(ar), int(ac)), "majority": int(majority_color)})
                support_pixels.extend(support_cells)
                placed = True
                break
            if not placed:
                return False
        if len(blocks) < SOURCE_BLOCK_MIN:
            return False
        rule["blocks"] = blocks
        rule["support_pixels"] = support_pixels
    return True


def _place_bystanders(
    inp: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> list[tuple[int, int, int]]:
    h, w = inp.shape
    n = random.randint(0, BYSTANDER_MAX)
    placed: list[tuple[int, int, int]] = []
    for _ in range(n):
        for _ in range(BYSTANDER_RETRIES):
            r = random.randint(0, h - 1)
            c = random.randint(0, w - 1)
            if (r, c) in occupied or (r, c) in reserved:
                continue
            color = random.choice(VOTE_COLORS)
            inp[r, c] = int(color)
            placed.append((int(r), int(c), int(color)))
            occupied.add((int(r), int(c)))
            _reserve_with_margin(reserved, [(int(r), int(c))])
            break
    return placed


def _build_output(rules: list[dict], bystanders: list[tuple[int, int, int]], h: int, w: int) -> np.ndarray:
    out = np.zeros((h, w), dtype=int)
    for r, c, color in bystanders:
        out[int(r), int(c)] = int(color)
    for rule in rules:
        for r, c, color in rule["support_pixels"]:
            out[int(r), int(c)] = int(color)
    for rule in rules:
        rel = rule["rel"]
        for block in rule["blocks"]:
            ar, ac = block["anchor"]
            target_color = int(block["majority"])
            for rr, cc in rel:
                nr = int(ar + rr)
                nc = int(ac + cc)
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = target_color
    return out


def _variant_metrics(inp: np.ndarray) -> dict[str, int]:
    bg = get_bg(inp)
    rules = _discover_rules(inp, bg=bg, pattern_color=PATTERN_COLOR)
    keys = {(int(rule["source_color"]), int(rule["neighbor_count"])) for rule in rules}
    repeated_source_rules = len(rules) - len({int(rule["source_color"]) for rule in rules})
    matched = _matched_blocks(inp, rules, int(bg))
    by_rule_majorities: dict[tuple[int, int], set[int]] = {}
    min_block_size = 99
    for item in matched:
        rule = item["rule"]
        key = (int(rule["source_color"]), int(rule["neighbor_count"]))
        by_rule_majorities.setdefault(key, set()).add(int(item["target_color"]))
        min_block_size = min(min_block_size, len(item["component"]))
    varying_rules = sum(1 for colors in by_rule_majorities.values() if len(colors) >= 2)
    return {
        "rules": len(rules),
        "unique_keys": len(keys),
        "repeated_source_rules": repeated_source_rules,
        "matched_blocks": len(matched),
        "varying_rules": varying_rules,
        "min_block_size": 0 if not matched else min_block_size,
    }


def _gen_attempt() -> np.ndarray | None:
    num_rules = random.choice(RULE_COUNT_WEIGHTS)
    min_dim = MIN_DIM_BY_RULES[num_rules]
    h = random.randint(min_dim, GRID_DIM_MAX)
    w = random.randint(min_dim, GRID_DIM_MAX)

    rules = _build_rules(num_rules, h * w)
    vote_palette = _pick_vote_palette()
    _assign_majority_sequences(rules, vote_palette)

    occupied: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()
    inp = np.zeros((h, w), dtype=int)

    if not _place_legends(rules, vote_palette, inp, occupied, reserved):
        return None
    if not _place_source_blocks(rules, vote_palette, inp, occupied, reserved):
        return None
    bystanders = _place_bystanders(inp, occupied, reserved)

    metrics = _variant_metrics(inp)
    expected_blocks = sum(int(rule["n_blocks"]) for rule in rules)
    if metrics["rules"] != num_rules:
        return None
    if metrics["unique_keys"] != num_rules:
        return None
    if metrics["repeated_source_rules"] < 1:
        return None
    if metrics["matched_blocks"] != expected_blocks:
        return None
    if metrics["varying_rules"] < 1:
        return None
    if metrics["min_block_size"] < 2:
        return None

    out = _build_output(rules, bystanders, h, w)
    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {OUTER_RETRIES} attempts")
