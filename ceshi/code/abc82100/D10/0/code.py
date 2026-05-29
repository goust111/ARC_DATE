# Variant of abc82100
# Dimension: D10 — 源像素与规则示例的空间分布 (meta)
# Change: 规则示例和teal模板集中在网格的某个区域，源像素散布在其他区域... → 规则示例和源像素交错分布

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg


# Design analysis for the D10 variant
# ----------------------------------
# concrete_behavior:
# - Keep the local rule semantics unchanged: each rule is still encoded by a 4-neighbor
#   source->target pair whose forward continuation reaches an anchor touching a teal stencil.
# - Change only the macro layout. Legends no longer occupy a dedicated corner/strip. Instead,
#   legends and ordinary source pixels are placed on the same board-wide field, and every
#   legend is forced to have nearby scattered source pixels at Chebyshev distance 2..4.
# - Example trace:
#   old behavior: a legend might sit in the top-left, while all reusable source pixels sit in
#   the lower-right, so visual segmentation is trivial.
#   new behavior: a blue->red legend with its teal stencil may sit near the middle, while lone
#   blue/green/yellow source pixels appear around it without touching it. The solver must still
#   identify the legend only from the adjacent colored pair plus teal stencil, then apply that
#   stencil to every other blue source pixel.
# complexity_calibration:
# - Colors remain unique per rule pair, and bystanders never reuse a learned source color.
#   This keeps the deduction about "which lone pixels are replaceable" visible and stable.
# - The generator gives each legend two nearby lone sources so the interleaving is obvious in
#   2-4 examples, but keeps the number of control categories small: there is still only one
#   stencil and one target color per source color.
# implicit_assumptions_audited:
# - _discover_rules scans every non-bg/non-teal pixel and trusts any adjacent colored pair
#   whose continuation sees a teal component. Therefore the generator must block all accidental
#   4-neighbor color pairs near teal.
# - _discover_rules picks the nearest teal component within Chebyshev distance <= 1 from the
#   anchor. Therefore different teal stencils must stay 8-disconnected so a single anchor
#   cannot ambiguously "see" two components.
# - main removes every non-legend pixel whose color matches a learned source color. Therefore
#   the generator must never place preserved bystanders using any learned source color.
# placement_legality:
# - All objects use a 4-neighbor exclusion halo so unintended adjacent colored pairs cannot
#   appear between different objects.
# - Teal cells additionally reserve their 8-neighborhood, preventing two stencils from merging
#   into one connected component or from being simultaneously reachable from one anchor.
# - Legends are scored to spread across the board, while still keeping enough nearby free cells
#   to host the forced interleaved source pixels.
# visual_observability:
# - The board stays readable because legend internals are unchanged and nearby lone sources are
#   still single isolated pixels. What changes is the segmentation challenge: legends are mixed
#   into the same neighborhoods as the source pixels they govern.


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn local source-target-stencil examples even when they are mixed among the source pixels.

    Input structure:
    - Black background with isolated colored pixels and teal (8) stencil components.
    - Rule examples are scattered across the same general area as the ordinary source pixels,
      rather than being parked in one dedicated legend zone.
    - A valid example is still a 4-neighbor colored pair source->target whose continued
      direction reaches an anchor touching a teal component.

    Rule:
    - For each discovered example, copy its teal stencil onto every other pixel of the same
      source color, recoloring the copied stencil to that example's target color.
    - The translation aligns the example's anchor with the source pixel being replaced.
    - Erase all teal stencils and the colored pixels that belong to rule examples.

    Key insight: Spatial mixing does not change the local signature of a rule example. The
    adjacent colored pair still names the source/target colors, and the nearby teal component
    still provides the stencil that must be stamped onto the other lone source pixels.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    pattern_color = 8
    h, w = g.shape

    rules = _discover_rules(g, bg=bg, pattern_color=pattern_color)
    legend_pixels: set[tuple[int, int]] = set()
    source_colors: set[int] = set()
    for rule in rules:
        legend_pixels.add(rule["source_pos"])
        legend_pixels.add(rule["target_pos"])
        source_colors.add(rule["source_color"])

    out = np.full_like(g, int(bg))

    for r, c in map(tuple, np.argwhere((g != int(bg)) & (g != pattern_color))):
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        if (r, c) in legend_pixels or color in source_colors:
            continue
        out[r, c] = color

    for rule in rules:
        ar, ac = rule["anchor"]
        pattern = rule["pattern"]
        for sr, sc in map(tuple, np.argwhere(g == rule["source_color"])):
            sr = int(sr)
            sc = int(sc)
            if (sr, sc) in legend_pixels:
                continue
            dr = sr - ar
            dc = sc - ac
            rows = pattern[:, 0] + dr
            cols = pattern[:, 1] + dc
            ok = (0 <= rows) & (rows < h) & (0 <= cols) & (cols < w)
            out[rows[ok], cols[ok]] = rule["target_color"]

    return out


# --------------- generator ---------------

# Grid dimensions
GRID_DIM_MAX = 20
MIN_DIM_BY_RULES = {1: 7, 2: 11, 3: 15, 4: 18}

# Rule sampling
RULE_COUNT_WEIGHTS = [1, 1, 2, 2, 3, 3, 4]
NON_BG_NON_TEAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

# Stencil size
STENCIL_MAX_CAP = 12
STENCIL_MIN_AREA_DIV = 100
STENCIL_MAX_AREA_DIV = 25

# Scattered source pixels per rule
SCATTERED_MIN = 3
SCATTERED_MAX_CAP = 8
SCATTERED_AREA_DIV = 30

# Bystander pixels
BYSTANDER_MAX = 5

# Placement retry budgets
LEGEND_RETRIES = 700
SCATTERED_RETRIES_PER_PIXEL = 80
BYSTANDER_RETRIES = 80
OUTER_RETRIES = 400

# Interleaving constraints
NEARBY_SOURCE_PASSES = 2
NEARBY_RING_MIN = 2
NEARBY_RING_MAX = 4
LEGEND_RING_REQUIREMENT = 4

_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def _reserve_neighbors(blocked: set[tuple[int, int]], cells, dirs) -> None:
    for r, c in cells:
        r = int(r)
        c = int(c)
        blocked.add((r, c))
        for dr, dc in dirs:
            blocked.add((r + dr, c + dc))


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


def _build_rules(num_rules: int, area: int):
    colors = NON_BG_NON_TEAL_COLORS[:]
    random.shuffle(colors)

    max_s = min(STENCIL_MAX_CAP, max(2, area // STENCIL_MAX_AREA_DIV))
    min_s = min(max(1, area // STENCIL_MIN_AREA_DIV), max_s)
    sc_max = min(SCATTERED_MAX_CAP, max(SCATTERED_MIN, area // SCATTERED_AREA_DIV))

    rules = []
    used_colors = set()
    for i in range(num_rules):
        src_c, tgt_c = colors[2 * i], colors[2 * i + 1]
        used_colors.update((src_c, tgt_c))
        stencil, anchor = _make_stencil(min_s, max_s)
        rel = [(r - anchor[0], c - anchor[1]) for r, c in stencil]
        rules.append(
            {
                "src": int(src_c),
                "tgt": int(tgt_c),
                "rel": rel,
                "n_sc": random.randint(SCATTERED_MIN, sc_max),
                "remaining": 0,
                "scattered": [],
            }
        )
    return rules, used_colors


def _legend_layout(rel, sr: int, sc: int, dr: int, dc: int, h: int, w: int):
    tr, tc = sr + dr, sc + dc
    ar, ac = sr + 2 * dr, sc + 2 * dc
    if not (0 <= tr < h and 0 <= tc < w and 0 <= ar < h and 0 <= ac < w):
        return None

    teal_cells = set()
    for rr, cc in rel:
        nr, nc = ar + rr, ac + cc
        if not (0 <= nr < h and 0 <= nc < w):
            return None
        teal_cells.add((int(nr), int(nc)))

    visible_cells = {(int(sr), int(sc)), (int(tr), int(tc))} | teal_cells
    all_cells = visible_cells | {(int(ar), int(ac))}
    return {
        "source_pos": (int(sr), int(sc)),
        "target_pos": (int(tr), int(tc)),
        "anchor": (int(ar), int(ac)),
        "teal_cells": teal_cells,
        "visible_cells": visible_cells,
        "all_cells": all_cells,
    }


def _ring_candidates(host_cells, h: int, w: int, occupied, blocked4):
    rows = [r for r, _ in host_cells]
    cols = [c for _, c in host_cells]
    candidates = []
    for r in range(max(0, min(rows) - NEARBY_RING_MAX), min(h, max(rows) + NEARBY_RING_MAX + 1)):
        for c in range(max(0, min(cols) - NEARBY_RING_MAX), min(w, max(cols) + NEARBY_RING_MAX + 1)):
            pos = (int(r), int(c))
            if pos in occupied or pos in blocked4:
                continue
            dist = min(_chebyshev(pos, cell) for cell in host_cells)
            if NEARBY_RING_MIN <= dist <= NEARBY_RING_MAX:
                candidates.append(pos)
    return candidates


def _commit_legend(rule, layout, inp, occupied, blocked4, blocked_teal8) -> None:
    sr, sc = layout["source_pos"]
    tr, tc = layout["target_pos"]
    inp[sr, sc] = rule["src"]
    inp[tr, tc] = rule["tgt"]
    for r, c in layout["teal_cells"]:
        inp[r, c] = 8

    occupied.update(layout["all_cells"])
    _reserve_neighbors(blocked4, layout["all_cells"], _DIRS4)
    _reserve_neighbors(blocked_teal8, layout["teal_cells"], _DIRS8)
    rule["legend"] = layout
    rule["remaining"] = int(rule["n_sc"])
    rule["scattered"] = []


def _place_legends(rules, inp, occupied, blocked4, blocked_teal8):
    h, w = inp.shape
    placed_layouts: list[dict] = []

    for rule in rules:
        best_layout = None
        best_score = None
        for _ in range(LEGEND_RETRIES):
            dr, dc = random.choice(_DIRS4)
            sr = random.randint(0, h - 1)
            sc = random.randint(0, w - 1)
            layout = _legend_layout(rule["rel"], sr, sc, dr, dc, h, w)
            if layout is None:
                continue
            if layout["all_cells"] & occupied:
                continue
            if layout["all_cells"] & blocked4:
                continue
            if layout["teal_cells"] & blocked_teal8:
                continue

            ring = _ring_candidates(layout["all_cells"], h, w, occupied, blocked4)
            if len(ring) < LEGEND_RING_REQUIREMENT:
                continue

            anchor = layout["anchor"]
            spread = min(
                (_chebyshev(anchor, other["anchor"]) for other in placed_layouts),
                default=max(h, w),
            )
            border_room = min(anchor[0], h - 1 - anchor[0], anchor[1], w - 1 - anchor[1])
            score = (spread, len(ring), border_room)
            if best_score is None or score > best_score:
                best_score = score
                best_layout = layout

        if best_layout is None:
            return False

        _commit_legend(rule, best_layout, inp, occupied, blocked4, blocked_teal8)
        placed_layouts.append(best_layout)

    return True


def _place_source_pixel(rule, pos: tuple[int, int], inp, occupied, blocked4) -> None:
    r, c = pos
    inp[r, c] = rule["src"]
    rule["scattered"].append((int(r), int(c)))
    rule["remaining"] -= 1
    occupied.add((int(r), int(c)))
    _reserve_neighbors(blocked4, [(int(r), int(c))], _DIRS4)


def _choose_rule_index(rules, preferred_indices):
    candidates = [i for i in preferred_indices if rules[i]["remaining"] > 0]
    if not candidates:
        candidates = [i for i, rule in enumerate(rules) if rule["remaining"] > 0]
    if not candidates:
        return None
    max_remaining = max(rules[i]["remaining"] for i in candidates)
    best = [i for i in candidates if rules[i]["remaining"] == max_remaining]
    return int(random.choice(best))


def _place_nearby_sources(rules, inp, occupied, blocked4):
    h, w = inp.shape
    legend_indices = list(range(len(rules)))
    random.shuffle(legend_indices)

    for pass_id in range(NEARBY_SOURCE_PASSES):
        for host_idx in legend_indices:
            if pass_id == 0:
                preferred = [host_idx]
            else:
                preferred = [i for i in legend_indices if i != host_idx] + [host_idx]
            target_idx = _choose_rule_index(rules, preferred)
            if target_idx is None:
                return False

            host_cells = rules[host_idx]["legend"]["all_cells"]
            candidates = _ring_candidates(host_cells, h, w, occupied, blocked4)
            random.shuffle(candidates)
            if not candidates:
                return False

            placed = False
            for pos in candidates:
                if pos in occupied or pos in blocked4:
                    continue
                _place_source_pixel(rules[target_idx], pos, inp, occupied, blocked4)
                placed = True
                break
            if not placed:
                return False

    return True


def _place_remaining_sources(rules, inp, occupied, blocked4):
    h, w = inp.shape
    all_cells = [(r, c) for r in range(h) for c in range(w)]

    for rule in rules:
        while rule["remaining"] > 0:
            random.shuffle(all_cells)
            placed = False
            budget = min(len(all_cells), rule["remaining"] * SCATTERED_RETRIES_PER_PIXEL)
            for pos in all_cells[:budget]:
                if pos in occupied or pos in blocked4:
                    continue
                _place_source_pixel(rule, pos, inp, occupied, blocked4)
                placed = True
                break
            if not placed:
                return False

        if len(rule["scattered"]) < SCATTERED_MIN:
            return False

    return True


def _place_bystanders(used_colors, inp, occupied, blocked4):
    bystander_colors = [c for c in NON_BG_NON_TEAL_COLORS if c not in used_colors]
    if not bystander_colors:
        return []

    h, w = inp.shape
    n = random.randint(0, min(BYSTANDER_MAX, len(bystander_colors) * 2))
    placed = []
    for _ in range(n):
        for _ in range(BYSTANDER_RETRIES):
            r = random.randint(0, h - 1)
            c = random.randint(0, w - 1)
            if (r, c) in occupied or (r, c) in blocked4:
                continue
            color = random.choice(bystander_colors)
            inp[r, c] = color
            placed.append((int(r), int(c), int(color)))
            occupied.add((int(r), int(c)))
            _reserve_neighbors(blocked4, [(int(r), int(c))], _DIRS4)
            break
    return placed


def _build_output(rules, bystander_pos, h: int, w: int):
    output = np.zeros((h, w), dtype=int)
    for r, c, bc in bystander_pos:
        output[r, c] = bc
    for rule in rules:
        rel = rule["rel"]
        for sr, sc in rule["scattered"]:
            for rr, cc in rel:
                nr, nc = sr + rr, sc + cc
                if 0 <= nr < h and 0 <= nc < w:
                    output[nr, nc] = rule["tgt"]
    return output


def _gen_attempt():
    num_rules = random.choice(RULE_COUNT_WEIGHTS)
    min_dim = MIN_DIM_BY_RULES[num_rules]
    h = random.randint(min_dim, GRID_DIM_MAX)
    w = random.randint(min_dim, GRID_DIM_MAX)

    rules, used_colors = _build_rules(num_rules, h * w)
    inp = np.zeros((h, w), dtype=int)
    occupied: set[tuple[int, int]] = set()
    blocked4: set[tuple[int, int]] = set()
    blocked_teal8: set[tuple[int, int]] = set()

    if not _place_legends(rules, inp, occupied, blocked4, blocked_teal8):
        return None
    if not _place_nearby_sources(rules, inp, occupied, blocked4):
        return None
    if not _place_remaining_sources(rules, inp, occupied, blocked4):
        return None

    bystander_pos = _place_bystanders(used_colors, inp, occupied, blocked4)
    output = _build_output(rules, bystander_pos, h, w)

    try:
        if np.array_equal(main(inp), output):
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
