# Combo variant of abc82100 (index 23)
# combo_spec: D9/2+D13/1
# Name: 循环颜色映射+规则链式应用

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg


# Design analysis for the combo variant
# a) Concreteization and behavior trace
#    - The legends still look like the base task: source pixel -> target pixel ->
#      anchor, with a teal stencil attached at the anchor.
#    - The three active legends are not independent. Their colors form one cycle
#      A->B, B->C, C->A, and the scattered non-legend pixels use only one start
#      color from that cycle.
#    - Small trace:
#        base task: scattered A pixels directly produce a B-colored copy of the
#        A->B legend stencil, and the process stops there.
#        combo task: step 1 turns each scattered A into a B stencil; step 2 treats
#        every B pixel of that stencil as a new source and expands it with the B->C
#        stencil; step 3 does the same with C->A. The final output is therefore the
#        three-step chained expansion, not the one-step A->B result.
# b) Complexity calibration
#    - The color relation is fixed to one 3-cycle so the repeated evidence is small
#      and visible; a larger cycle would be difficult to infer from 2-4 ARC pairs.
#    - Only the first cycle color is scattered in the input. This keeps the chain
#      order unambiguous instead of mixing several simultaneous starting phases.
#    - Each stencil has at least two pixels, and the generator rejects cases where
#      the three-step result collapses to the one-step result, so the fusion is
#      visually meaningful rather than a cosmetic relabeling.
# c) Hidden assumptions in the original main()
#    - Rule discovery from legends still works, but the assumption "each rule can be
#      applied independently to the original grid" is false now.
#    - The original preservation logic removed all source-colored pixels in one pass.
#      Here only the current step's source color should be consumed, while bystanders
#      and future-step colors must survive until their own turn.
#    - The original solver never needed an execution order. This combo adds a new
#      assumption: the chain starts from the unique cycle color actually scattered in
#      the cleaned input, and then follows the directed cycle exactly once.
# d) Placement legality
#    - Legends are separated with 8-neighbor buffers so no accidental source-target
#      pairs compete for the same teal component.
#    - Each scattered seed is accepted only if its entire 3-step footprint stays in
#      bounds and is disjoint from all previously reserved chain footprints.
#    - Bystanders are placed outside all latent chain footprints, so preservation is
#      unambiguous and they never masquerade as intermediate chain sources.
# e) Visual observability
#    - A human sees three legends whose source and target colors close a loop, plus
#      repeated isolated pixels of exactly one of those colors. The output contains
#      larger same-colored motifs, forcing the interpretation "follow the cycle for
#      several steps" rather than "apply one local substitution once".
#    - Generator rejection explicitly compares the intended chained output with the
#      one-step baseline and keeps only samples where they differ.
# f) Visual cleanliness
#    - Active chain footprints are fully in-bounds and mutually disjoint, so the
#      final motifs stay readable instead of merging into one blob.
#    - Only a few bystanders are allowed, and they use colors outside the cycle.
# g) Color-change inferability
#    - The relevant palette is only three cycle colors plus optional outsiders.
#      Because the legends themselves show A->B, B->C, C->A, the solver can infer
#      why the start color reappears after three chained applications.


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


def _find_cycle_rules(rules: list[dict]) -> list[dict]:
    for first in rules:
        for second in rules:
            if second is first or int(second["source_color"]) != int(first["target_color"]):
                continue
            for third in rules:
                if third is first or third is second:
                    continue
                if int(third["source_color"]) != int(second["target_color"]):
                    continue
                if int(third["target_color"]) != int(first["source_color"]):
                    continue
                colors = {
                    int(first["source_color"]),
                    int(second["source_color"]),
                    int(third["source_color"]),
                }
                if len(colors) == 3:
                    return [first, second, third]
    return []


def _collect_legend_pixels(rules: list[dict]) -> set[tuple[int, int]]:
    legend_pixels: set[tuple[int, int]] = set()
    for rule in rules:
        legend_pixels.add((int(rule["source_pos"][0]), int(rule["source_pos"][1])))
        legend_pixels.add((int(rule["target_pos"][0]), int(rule["target_pos"][1])))
    return legend_pixels


def _clean_input_state(
    grid: np.ndarray,
    bg: int,
    pattern_color: int,
    legend_pixels: set[tuple[int, int]],
) -> np.ndarray:
    state = np.full_like(grid, int(bg))
    for r, c in map(tuple, np.argwhere((grid != int(bg)) & (grid != pattern_color))):
        rr = int(r)
        cc = int(c)
        if (rr, cc) in legend_pixels:
            continue
        state[rr, cc] = int(grid[rr, cc])
    return state


def _ordered_cycle_from_start(cycle_rules: list[dict], start_color: int) -> list[dict]:
    by_source = {int(rule["source_color"]): rule for rule in cycle_rules}
    ordered: list[dict] = []
    current = int(start_color)
    for _ in range(len(by_source)):
        rule = by_source.get(int(current))
        if rule is None:
            return []
        ordered.append(rule)
        current = int(rule["target_color"])
    if int(current) != int(start_color):
        return []
    if len({int(rule["source_color"]) for rule in ordered}) != len(by_source):
        return []
    return ordered


def _apply_rule_once(state: np.ndarray, rule: dict, bg: int) -> np.ndarray:
    out = np.full_like(state, int(bg))
    h, w = state.shape
    src = int(rule["source_color"])
    tgt = int(rule["target_color"])
    ar, ac = map(int, rule["anchor"])
    pattern = np.asarray(rule["pattern"], dtype=int)

    keep_mask = (state != int(bg)) & (state != int(src))
    out[keep_mask] = state[keep_mask]

    for sr, sc in map(tuple, np.argwhere(state == int(src))):
        rr = int(sr)
        cc = int(sc)
        dr = rr - int(ar)
        dc = cc - int(ac)
        rows = pattern[:, 0] + int(dr)
        cols = pattern[:, 1] + int(dc)
        ok = (0 <= rows) & (rows < h) & (0 <= cols) & (cols < w)
        out[rows[ok], cols[ok]] = int(tgt)

    return out


def _solve_independent(grid: np.ndarray, bg: int, pattern_color: int, rules: list[dict]) -> np.ndarray:
    h, w = grid.shape
    legend_pixels = _collect_legend_pixels(rules)
    source_colors = {int(rule["source_color"]) for rule in rules}
    out = np.full((h, w), int(bg), dtype=int)

    for r, c in map(tuple, np.argwhere((grid != int(bg)) & (grid != pattern_color))):
        rr = int(r)
        cc = int(c)
        color = int(grid[rr, cc])
        if (rr, cc) in legend_pixels or color in source_colors:
            continue
        out[rr, cc] = color

    for rule in rules:
        ar, ac = map(int, rule["anchor"])
        pattern = np.asarray(rule["pattern"], dtype=int)
        for sr, sc in map(tuple, np.argwhere(grid == int(rule["source_color"]))):
            rr = int(sr)
            cc = int(sc)
            if (rr, cc) in legend_pixels:
                continue
            dr = rr - int(ar)
            dc = cc - int(ac)
            rows = pattern[:, 0] + int(dr)
            cols = pattern[:, 1] + int(dc)
            ok = (0 <= rows) & (rows < h) & (0 <= cols) & (cols < w)
            out[rows[ok], cols[ok]] = int(rule["target_color"])

    return out


def _solve_parallel_cycle(base_state: np.ndarray, cycle_rules: list[dict], bg: int) -> np.ndarray:
    cycle_colors = {int(rule["source_color"]) for rule in cycle_rules}
    source_positions: dict[int, list[tuple[int, int]]] = {color: [] for color in cycle_colors}
    for r, c in map(tuple, np.argwhere(base_state != int(bg))):
        rr = int(r)
        cc = int(c)
        color = int(base_state[rr, cc])
        if color in source_positions:
            source_positions[color].append((rr, cc))

    out = np.full_like(base_state, int(bg))
    for r, c in map(tuple, np.argwhere(base_state != int(bg))):
        rr = int(r)
        cc = int(c)
        color = int(base_state[rr, cc])
        if color not in cycle_colors:
            out[rr, cc] = color

    h, w = base_state.shape
    for rule in cycle_rules:
        ar, ac = map(int, rule["anchor"])
        pattern = np.asarray(rule["pattern"], dtype=int)
        for sr, sc in source_positions.get(int(rule["source_color"]), []):
            dr = int(sr) - int(ar)
            dc = int(sc) - int(ac)
            rows = pattern[:, 0] + int(dr)
            cols = pattern[:, 1] + int(dc)
            ok = (0 <= rows) & (rows < h) & (0 <= cols) & (cols < w)
            out[rows[ok], cols[ok]] = int(rule["target_color"])

    return out


def _extract_chain_start(base_state: np.ndarray, cycle_rules: list[dict], bg: int) -> int | None:
    cycle_colors = {int(rule["source_color"]) for rule in cycle_rules}
    present = sorted({int(v) for v in base_state[base_state != int(bg)] if int(v) in cycle_colors})
    if len(present) == 1:
        return int(present[0])
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn one three-color legend cycle and run it as a chained three-step expansion.

    Input structure:
    - Black background with teal (8) stencil components and local legends of the
      form source->target->anchor, where the anchor touches or belongs to that
      legend's teal stencil.
    - Exactly three legends are operative, and their colors form one directed
      loop A->B, B->C, C->A.
    - Away from the legends, isolated scattered pixels use only one of the cycle
      colors; any other non-cycle colors are bystanders.

    Rule:
    - Remove all legends and teal pixels, then treat the remaining scattered
      cycle-colored pixels as the starting state.
    - Starting from the unique scattered cycle color, follow the cycle order for
      three steps. On each step, every pixel of the current source color is
      consumed and replaced by a translated copy of that rule's teal stencil,
      recolored to the target color.
    - The whole output of one step becomes the input to the next step, so pixels
      generated at step k can themselves trigger the legend of step k+1.

    Key insight: the legends encode both a color loop and an execution order.
    Because only one loop color is initially scattered, the solver must start
    there and repeatedly feed each step's result into the next legend instead of
    treating the three rules as independent one-shot substitutions.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    pattern_color = 8

    rules = _discover_rules(g, bg=bg, pattern_color=pattern_color)
    if not rules:
        return np.full_like(g, int(bg))

    cycle_rules = _find_cycle_rules(rules)
    if len(cycle_rules) != 3:
        return _solve_independent(g, bg=bg, pattern_color=pattern_color, rules=rules)

    legend_pixels = _collect_legend_pixels(rules)
    base_state = _clean_input_state(g, bg=bg, pattern_color=pattern_color, legend_pixels=legend_pixels)
    start_color = _extract_chain_start(base_state, cycle_rules, bg=bg)
    if start_color is None:
        return _solve_parallel_cycle(base_state, cycle_rules, bg=bg)

    ordered_rules = _ordered_cycle_from_start(cycle_rules, start_color=start_color)
    if len(ordered_rules) != 3:
        return _solve_parallel_cycle(base_state, cycle_rules, bg=bg)

    state = np.array(base_state, copy=True)
    for rule in ordered_rules:
        state = _apply_rule_once(state, rule, bg=bg)
    return state


# --------------- generator ---------------

GRID_DIM_MIN = 18
GRID_DIM_MAX = 20
NON_BG_NON_TEAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

STENCIL_MAX_CAP = 3
STENCIL_MIN_AREA_DIV = 160
STENCIL_MAX_AREA_DIV = 70

SEED_MIN = 3
SEED_MAX_CAP = 4
SEED_AREA_DIV = 110
BYSTANDER_MAX = 2

LEGEND_RETRIES = 800
SEED_RETRIES_PER_PIXEL = 120
BYSTANDER_RETRIES = 120
OUTER_RETRIES = 600

MIN_CHAIN_DIFF = 8

_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _reserve_4neighbors(reserved: set[tuple[int, int]], cells) -> None:
    for r, c in cells:
        rr = int(r)
        cc = int(c)
        for dr, dc in _DIRS4:
            reserved.add((rr + dr, cc + dc))


def _reserve_8neighbors(reserved: set[tuple[int, int]], cells) -> None:
    for r, c in cells:
        rr = int(r)
        cc = int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                reserved.add((rr + dr, cc + dc))


def _make_stencil(min_pixels: int, max_pixels: int) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    n = random.randint(max(2, min_pixels), max(2, max_pixels))
    coords = {(0, 0)}
    for _ in range(n - 1):
        frontier = set()
        for r, c in coords:
            for dr, dc in _DIRS8:
                point = (int(r) + dr, int(c) + dc)
                if point not in coords:
                    frontier.add(point)
        coords.add(random.choice(list(frontier)))
    coords_list = sorted(coords)
    return coords_list, random.choice(coords_list)


def _translated_cells(rel: list[tuple[int, int]], sr: int, sc: int) -> set[tuple[int, int]]:
    return {(int(sr) + int(rr), int(sc) + int(cc)) for rr, cc in rel}


def _build_rules(area: int) -> tuple[list[dict], set[int]]:
    cycle_colors = random.sample(NON_BG_NON_TEAL_COLORS, 3)

    max_s = min(STENCIL_MAX_CAP, max(2, area // STENCIL_MAX_AREA_DIV))
    min_s = min(max(2, area // STENCIL_MIN_AREA_DIV), max_s)
    seed_max = min(SEED_MAX_CAP, max(SEED_MIN, area // SEED_AREA_DIV))

    while True:
        rel_sets: set[tuple[tuple[int, int], ...]] = set()
        rules: list[dict] = []
        ok = True
        for i, src_c in enumerate(cycle_colors):
            tgt_c = cycle_colors[(i + 1) % 3]
            stencil, anchor = _make_stencil(min_s, max_s)
            rel = tuple(sorted((int(r - anchor[0]), int(c - anchor[1])) for r, c in stencil))
            if rel in rel_sets:
                ok = False
                break
            rel_sets.add(rel)
            rules.append(
                {
                    "src": int(src_c),
                    "tgt": int(tgt_c),
                    "rel": [(int(rr), int(cc)) for rr, cc in rel],
                }
            )
        if ok:
            break

    rules[0]["n_sc"] = random.randint(SEED_MIN, seed_max)
    return rules, {int(c) for c in cycle_colors}


def _seed_chain_trace(
    seed: tuple[int, int],
    rules: list[dict],
    h: int,
    w: int,
) -> dict | None:
    current = {(int(seed[0]), int(seed[1]))}
    footprint = set(current)
    stages: list[set[tuple[int, int]]] = []

    for rule in rules:
        counts: dict[tuple[int, int], int] = {}
        for sr, sc in current:
            for rr, cc in rule["rel"]:
                nr = int(sr) + int(rr)
                nc = int(sc) + int(cc)
                if not (0 <= nr < h and 0 <= nc < w):
                    return None
                counts[(nr, nc)] = counts.get((nr, nc), 0) + 1
        if not counts or any(v > 1 for v in counts.values()):
            return None
        current = set(counts)
        stages.append(set(current))
        footprint.update(current)

    return {"stages": stages, "footprint": footprint}


def _place_legends(
    rules: list[dict],
    inp: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> bool:
    h, w = inp.shape
    for rule in rules:
        placed = False
        for _ in range(LEGEND_RETRIES):
            dr, dc = random.choice(_DIRS4)
            sr = random.randint(0, h - 1)
            sc = random.randint(0, w - 1)
            tr = int(sr) + int(dr)
            tc = int(sc) + int(dc)
            ar = int(sr) + 2 * int(dr)
            ac = int(sc) + 2 * int(dc)
            if not (0 <= tr < h and 0 <= tc < w and 0 <= ar < h and 0 <= ac < w):
                continue

            teal_cells = _translated_cells(rule["rel"], ar, ac)
            if any(not (0 <= r < h and 0 <= c < w) for r, c in teal_cells):
                continue
            if (int(sr), int(sc)) in teal_cells or (int(tr), int(tc)) in teal_cells:
                continue

            all_cells = {(int(sr), int(sc)), (int(tr), int(tc)), (int(ar), int(ac))} | teal_cells
            if all_cells & occupied or all_cells & reserved:
                continue

            inp[int(sr), int(sc)] = int(rule["src"])
            inp[int(tr), int(tc)] = int(rule["tgt"])
            for r, c in teal_cells:
                inp[int(r), int(c)] = 8
            occupied.update(all_cells)
            _reserve_8neighbors(reserved, all_cells)
            placed = True
            break
        if not placed:
            return False
    return True


def _place_seeds(
    rules: list[dict],
    inp: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], set[tuple[int, int]]] | None:
    h, w = inp.shape
    seeds: list[tuple[int, int]] = []
    future_reserved = set(occupied) | set(reserved)
    chain_footprints: set[tuple[int, int]] = set()

    for _ in range(rules[0]["n_sc"]):
        placed = False
        for _ in range(SEED_RETRIES_PER_PIXEL):
            sr = random.randint(0, h - 1)
            sc = random.randint(0, w - 1)
            seed = (int(sr), int(sc))
            if seed in occupied or seed in reserved:
                continue
            trace = _seed_chain_trace(seed, rules, h=h, w=w)
            if trace is None:
                continue
            footprint = trace["footprint"]
            if footprint & future_reserved:
                continue

            seeds.append(seed)
            inp[int(sr), int(sc)] = int(rules[0]["src"])
            occupied.add(seed)
            _reserve_4neighbors(reserved, [seed])
            chain_footprints.update(footprint)
            future_reserved.update(footprint)
            placed = True
            break
        if not placed:
            return None

    return seeds, chain_footprints


def _place_bystanders(
    used_colors: set[int],
    inp: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
    chain_footprints: set[tuple[int, int]],
) -> list[tuple[int, int, int]]:
    bystander_colors = [c for c in NON_BG_NON_TEAL_COLORS if c not in used_colors]
    if not bystander_colors:
        return []

    h, w = inp.shape
    n = random.randint(0, BYSTANDER_MAX)
    placed: list[tuple[int, int, int]] = []
    blocked = set(chain_footprints) | set(occupied) | set(reserved)
    for _ in range(n):
        for _ in range(BYSTANDER_RETRIES):
            r = random.randint(0, h - 1)
            c = random.randint(0, w - 1)
            if (int(r), int(c)) in blocked:
                continue
            color = int(random.choice(bystander_colors))
            inp[int(r), int(c)] = color
            placed.append((int(r), int(c), color))
            occupied.add((int(r), int(c)))
            _reserve_4neighbors(reserved, [(int(r), int(c))])
            blocked.add((int(r), int(c)))
            break
    return placed


def _apply_rel_rule_once(state: np.ndarray, rule: dict) -> np.ndarray:
    out = np.zeros_like(state, dtype=int)
    src = int(rule["src"])
    tgt = int(rule["tgt"])
    rel = rule["rel"]
    h, w = state.shape

    keep_mask = (state != 0) & (state != int(src))
    out[keep_mask] = state[keep_mask]

    for sr, sc in map(tuple, np.argwhere(state == int(src))):
        rr = int(sr)
        cc = int(sc)
        for dr, dc in rel:
            nr = rr + int(dr)
            nc = cc + int(dc)
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = int(tgt)

    return out


def _build_initial_state(
    rules: list[dict],
    seeds: list[tuple[int, int]],
    bystanders: list[tuple[int, int, int]],
    h: int,
    w: int,
) -> np.ndarray:
    state = np.zeros((h, w), dtype=int)
    for r, c in seeds:
        state[int(r), int(c)] = int(rules[0]["src"])
    for r, c, color in bystanders:
        state[int(r), int(c)] = int(color)
    return state


def _build_output(
    rules: list[dict],
    seeds: list[tuple[int, int]],
    bystanders: list[tuple[int, int, int]],
    h: int,
    w: int,
) -> np.ndarray:
    state = _build_initial_state(rules, seeds, bystanders, h, w)
    for rule in rules:
        state = _apply_rel_rule_once(state, rule)
    return state


def _build_one_step_output(
    rules: list[dict],
    seeds: list[tuple[int, int]],
    bystanders: list[tuple[int, int, int]],
    h: int,
    w: int,
) -> np.ndarray:
    state = _build_initial_state(rules, seeds, bystanders, h, w)
    return _apply_rel_rule_once(state, rules[0])


def chain_difference_metric(grid: np.ndarray) -> int:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    rules = _discover_rules(g, bg=bg, pattern_color=8)
    cycle_rules = _find_cycle_rules(rules)
    if len(cycle_rules) != 3:
        return 0

    legend_pixels = _collect_legend_pixels(rules)
    base_state = _clean_input_state(g, bg=bg, pattern_color=8, legend_pixels=legend_pixels)
    start_color = _extract_chain_start(base_state, cycle_rules, bg=bg)
    if start_color is None:
        return 0

    ordered_rules = _ordered_cycle_from_start(cycle_rules, start_color=start_color)
    if len(ordered_rules) != 3:
        return 0

    one_step = _apply_rule_once(base_state, ordered_rules[0], bg=bg)
    chained = np.array(base_state, copy=True)
    for rule in ordered_rules:
        chained = _apply_rule_once(chained, rule, bg=bg)
    return int(np.count_nonzero(chained != one_step))


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_DIM_MIN, GRID_DIM_MAX)
    w = random.randint(GRID_DIM_MIN, GRID_DIM_MAX)

    rules, used_colors = _build_rules(h * w)
    occupied: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()
    inp = np.zeros((h, w), dtype=int)

    if not _place_legends(rules, inp, occupied, reserved):
        return None

    placed = _place_seeds(rules, inp, occupied, reserved)
    if placed is None:
        return None
    seeds, chain_footprints = placed

    bystanders = _place_bystanders(used_colors, inp, occupied, reserved, chain_footprints)
    output = _build_output(rules, seeds, bystanders, h, w)
    one_step = _build_one_step_output(rules, seeds, bystanders, h, w)

    if int(np.count_nonzero(output)) <= int(np.count_nonzero(one_step)):
        return None
    if int(np.count_nonzero(output != one_step)) < MIN_CHAIN_DIFF:
        return None

    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(OUTER_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {OUTER_RETRIES} attempts")
