# Combo variant of abc82100 (index 42)
# combo_spec: D6/3+D9/2+D13/1
# Name: 链式应用+颜色交换+循环映射

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg


# Design analysis for the combo variant
# -------------------------------------
# a) concretization_and_trace
# - The input still contains three local legends of the original form:
#   source -> target -> anchor/stencil, with a teal component attached at the
#   anchor. Here the three legends use colors A->B, B->C, C->A, so the colors
#   visibly form a directed 3-cycle.
# - The new part is execution order. The chain starts from the top-left-most
#   legend, then follows the cycle by color matching: if a legend ends in color
#   B, the next step is the legend whose source color is B.
# - Each step is a bidirectional swap, not a one-way rewrite. For a rule A->B,
#   every A pixel stamps that legend's stencil in color B, and every B pixel
#   stamps the same stencil in color A.
# - The full output is built sequentially. Small trace with three rules
#   R1=A<->B using stencil S1, R2=B<->C using S2, R3=C<->A using S3:
#     start A pixel -> after R1: B-shaped S1
#                   -> after R2: C-shaped S1+S2
#                   -> after R3: A-shaped S1+S2+S3
#     start B pixel -> after R1: A-shaped S1
#                   -> after R2: unchanged
#                   -> after R3: C-shaped S1+S3
#     start C pixel -> after R1: unchanged
#                   -> after R2: B-shaped S2
#                   -> after R3: unchanged
# - So the earliest color in the chain undergoes three stencil applications, the
#   middle color undergoes two, and the last color undergoes one. This is the
#   visible difference from both the original parallel rules and the single-step
#   cycle variant.
#
# b) complexity_calibration
# - Exactly three active legends are used. Adding a decoy would dilute the main
#   observation, while fewer than three legends would destroy the cycle.
# - Each of the three cycle colors appears multiple times as isolated actor
#   pixels, so humans see repeated evidence for the three different chain depths.
# - Stencils are kept small but non-trivial (2-4 pixels), and the generator
#   rejects stencil triples whose composed footprints collapse to nearly the same
#   size. That keeps the chain effect visible instead of degenerating into a
#   simple recoloring.
#
# c) implicit_assumption_audit_of_original_main
# - Original code assumed rules are independent; the loop stamped each rule from
#   the immutable input and never fed one rule's output into the next. That is
#   false now, so the solver must explicitly build a step-by-step chain.
# - Original code assumed only the source color is consumed. That breaks under
#   the swap step because both endpoint colors are active at that step.
# - Original code had no need for a rule order. The combo variant does, so the
#   generator makes the first legend spatially identifiable and the solver uses
#   that visible start plus the A->B->C->A color links to order the chain.
# - Original input actors had to stay isolated forever to avoid false legends.
#   Here only the initial grid must have that property; later intermediate grids
#   may form larger components because chaining is intentional.
#
# d) placement_legality
# - Legends are placed as separated horizontal triples near the top of the grid.
#   Their 8-neighborhoods are reserved so scattered actors cannot accidentally
#   create alternate source-target examples near teal.
# - Initial actor pixels reserve 4-neighborhoods so rule discovery only sees the
#   intended legends.
# - The generator computes each actor's footprint at every chain step and accepts
#   a placement only if all step footprints stay in bounds and remain disjoint.
#   This prevents overwrite ambiguities between different chain branches.
#
# e) visual_observability
# - The cycle itself is visible because the three legend pairs reuse colors in a
#   closed loop. The start is visible because one legend is placed leftmost/topmost.
# - The chain is visible because the three source colors do not produce the same
#   kind of output: one color grows through three legends, one through two, and
#   one through one. Multiple repeated actors of each color make that contrast
#   learnable from a few examples.
#
# f) visual_cleanliness
# - Final and intermediate footprints are forced disjoint, so the output is a set
#   of clean separated shapes rather than a pile of overwritten colors.
# - Bystanders use only non-cycle colors and are kept outside all footprints.
#
# g) color_change_inferability
# - Only three active colors participate, and every one of them appears both in a
#   legend and among scattered actors. No hidden ordering by color id is used.
# - The visible cues are structural: left-to-right legend placement chooses the
#   first step, and the directed cycle tells which legend comes next.
#
# new_solver_assumptions_to_match_generator
# - Exactly one intended 3-cycle is present among the discovered legends.
# - The leftmost/topmost legend is the first rule of the chain.
# - Initial non-legend active-color pixels are isolated actors; the generator
#   guarantees this with neighborhood reservation before any chaining happens.


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


def _ordered_cycle_chain(rules: list[dict]) -> list[dict]:
    cycle_rules = _find_cycle_rules(rules)
    if not cycle_rules:
        return []

    start = min(cycle_rules, key=lambda rule: tuple(map(int, rule["source_pos"])))
    by_source = {int(rule["source_color"]): rule for rule in cycle_rules}

    chain = [start]
    while len(chain) < len(cycle_rules):
        nxt = by_source.get(int(chain[-1]["target_color"]))
        if nxt is None or nxt in chain:
            return sorted(cycle_rules, key=lambda rule: tuple(map(int, rule["source_pos"])))
        chain.append(nxt)

    if int(chain[-1]["target_color"]) != int(chain[0]["source_color"]):
        return sorted(cycle_rules, key=lambda rule: tuple(map(int, rule["source_pos"])))
    return chain


def _apply_swap_rule(grid: np.ndarray, rule: dict, bg: int) -> np.ndarray:
    current = np.asarray(grid, dtype=int)
    h, w = current.shape
    src = int(rule["source_color"])
    tgt = int(rule["target_color"])
    ar, ac = map(int, rule["anchor"])
    pattern = np.asarray(rule["pattern"], dtype=int)
    rel_rows = pattern[:, 0] - int(ar)
    rel_cols = pattern[:, 1] - int(ac)

    out = current.copy()
    active_mask = (current == int(src)) | (current == int(tgt))
    out[active_mask] = int(bg)

    for actor_color, fill_color in ((src, tgt), (tgt, src)):
        for sr, sc in map(tuple, np.argwhere(current == int(actor_color))):
            sr = int(sr)
            sc = int(sc)
            rows = sr + rel_rows
            cols = sc + rel_cols
            ok = (0 <= rows) & (rows < h) & (0 <= cols) & (cols < w)
            out[rows[ok], cols[ok]] = int(fill_color)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Execute the visible three-rule color cycle as a chained sequence of reversible stencil swaps.

    Input structure:
    - Black background with isolated colored pixels and three teal (8) stencil components.
    - Each legend is a 4-neighbor source->target pair followed by an anchor touching its teal stencil.
    - The three legends reuse three colors in a closed loop A->B, B->C, C->A, and the top-left-most
      legend is the first rule of the chain.

    Rule:
    - Remove all legend pixels and all teal template pixels, then process the three cycle legends in
      chain order: start from the visible first legend and follow target-color to next source-color.
    - At each step, the current rule swaps its two endpoint colors bidirectionally: every source-colored
      pixel stamps the rule's stencil in the target color, and every target-colored pixel stamps it back
      in the source color.
    - The full output of one step becomes the full input of the next step, so earlier colors can be
      transformed again by later legends while unrelated bystander colors stay unchanged.

    Key insight: The legends are neither independent nor one-way. They define a cyclic rule chain with
    a visible starting point, and each step is a two-color swap whose stamped output feeds the next rule.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    pattern_color = 8

    rules = _discover_rules(g, bg=bg, pattern_color=pattern_color)
    chain_rules = _ordered_cycle_chain(rules)
    if not chain_rules:
        chain_rules = sorted(rules, key=lambda rule: tuple(map(int, rule["source_pos"])))

    legend_pixels: set[tuple[int, int]] = set()
    for rule in rules:
        legend_pixels.add(tuple(map(int, rule["source_pos"])))
        legend_pixels.add(tuple(map(int, rule["target_pos"])))

    current = g.copy()
    current[current == int(pattern_color)] = int(bg)
    for r, c in legend_pixels:
        current[int(r), int(c)] = int(bg)

    for rule in chain_rules:
        current = _apply_swap_rule(current, rule, bg=int(bg))

    return current


# --------------- generator ---------------

GRID_DIM_MIN = 18
GRID_DIM_MAX = 20
NON_BG_NON_TEAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

STENCIL_MIN_PIXELS = 2
STENCIL_MAX_PIXELS = 4
ACTORS_PER_COLOR_MIN = 2
ACTORS_PER_COLOR_MAX = 3
BYSTANDER_MAX = 2

LEGEND_RETRIES = 400
ACTOR_RETRIES = 800
BYSTANDER_RETRIES = 150
STENCIL_RETRIES = 300
OUTER_RETRIES = 500

_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _reserve_4neighbors(reserved: set[tuple[int, int]], cells) -> None:
    for r, c in cells:
        reserved.add((int(r), int(c)))
        for dr, dc in _DIRS4:
            reserved.add((int(r) + dr, int(c) + dc))


def _reserve_8neighbors(reserved: set[tuple[int, int]], cells) -> None:
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                reserved.add((int(r) + dr, int(c) + dc))


def _make_stencil() -> tuple[list[tuple[int, int]], tuple[int, int]]:
    n = random.randint(STENCIL_MIN_PIXELS, STENCIL_MAX_PIXELS)
    coords = {(0, 0)}
    for _ in range(n - 1):
        frontier = set()
        for r, c in coords:
            for dr, dc in _DIRS8:
                p = (int(r) + dr, int(c) + dc)
                if p not in coords:
                    frontier.add(p)
        coords.add(random.choice(list(frontier)))
    coords_list = sorted(coords)
    return coords_list, random.choice(coords_list)


def _compose_offsets(
    first: set[tuple[int, int]],
    second: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    return {
        (int(r1) + int(r2), int(c1) + int(c2))
        for r1, c1 in first
        for r2, c2 in second
    }


def _shift_offsets(offsets: set[tuple[int, int]], center: tuple[int, int]) -> set[tuple[int, int]]:
    cr, cc = center
    return {(int(cr) + int(r), int(cc) + int(c)) for r, c in offsets}


def _fits_offsets(offsets: set[tuple[int, int]], center: tuple[int, int], h: int, w: int) -> bool:
    for r, c in _shift_offsets(offsets, center):
        if not (0 <= int(r) < h and 0 <= int(c) < w):
            return False
    return True


def _build_rules() -> tuple[list[dict], dict[int, dict[int, set[tuple[int, int]]]], set[int]] | tuple[None, None, None]:
    colors = random.sample(NON_BG_NON_TEAL_COLORS, 3)
    a, b, c = map(int, colors)

    for _ in range(STENCIL_RETRIES):
        s1_coords, s1_anchor = _make_stencil()
        s2_coords, s2_anchor = _make_stencil()
        s3_coords, s3_anchor = _make_stencil()

        s1 = {(int(r - s1_anchor[0]), int(c_ - s1_anchor[1])) for r, c_ in s1_coords}
        s2 = {(int(r - s2_anchor[0]), int(c_ - s2_anchor[1])) for r, c_ in s2_coords}
        s3 = {(int(r - s3_anchor[0]), int(c_ - s3_anchor[1])) for r, c_ in s3_coords}

        a2 = _compose_offsets(s1, s2)
        a3 = _compose_offsets(a2, s3)
        b3 = _compose_offsets(s1, s3)

        if not (len(a2) > len(s1) and len(a3) > len(a2) and len(b3) > len(s1)):
            continue
        if not (len(a3) > len(b3) > len(s2)):
            continue

        rules = [
            {"src": a, "tgt": b, "rel": sorted(s1)},
            {"src": b, "tgt": c, "rel": sorted(s2)},
            {"src": c, "tgt": a, "rel": sorted(s3)},
        ]
        footprints = {
            a: {1: set(s1), 2: set(a2), 3: set(a3)},
            b: {1: set(s1), 2: set(s1), 3: set(b3)},
            c: {1: {(0, 0)}, 2: set(s2), 3: set(s2)},
        }
        return rules, footprints, {a, b, c}

    return None, None, None


def _place_legends(
    rules: list[dict],
    inp: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> bool:
    h, w = inp.shape
    block = max(4, w // 3)

    for i, rule in enumerate(rules):
        rel = rule["rel"]
        placed = False
        c_lo = i * block
        c_hi = w - 3 if i == len(rules) - 1 else min(w - 3, (i + 1) * block - 1)
        if c_lo > c_hi:
            return False

        for _ in range(LEGEND_RETRIES):
            r = random.randint(0, max(0, h // 2 - 2))
            c = random.randint(c_lo, c_hi)
            sr, sc = int(r), int(c)
            tr, tc = int(r), int(c + 1)
            ar, ac = int(r), int(c + 2)

            teal_cells = _shift_offsets(set(rel), (ar, ac))
            if not teal_cells:
                continue
            if any(not (0 <= rr < h and 0 <= cc < w) for rr, cc in teal_cells):
                continue
            if (sr, sc) in teal_cells or (tr, tc) in teal_cells:
                continue

            all_cells = {(sr, sc), (tr, tc), (ar, ac)} | teal_cells
            if all_cells & occupied or all_cells & reserved:
                continue

            inp[sr, sc] = int(rule["src"])
            inp[tr, tc] = int(rule["tgt"])
            for rr, cc in teal_cells:
                inp[rr, cc] = 8

            occupied.update(all_cells)
            _reserve_8neighbors(reserved, all_cells)
            rule["source_pos"] = (sr, sc)
            rule["target_pos"] = (tr, tc)
            rule["anchor"] = (ar, ac)
            placed = True
            break

        if not placed:
            return False

    return True


def _place_actors(
    rules: list[dict],
    footprints: dict[int, dict[int, set[tuple[int, int]]]],
    inp: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> tuple[bool, dict[int, list[tuple[int, int]]], dict[int, set[tuple[int, int]]]]:
    h, w = inp.shape
    positions_by_color: dict[int, list[tuple[int, int]]] = {}
    step_used = {1: set(), 2: set(), 3: set()}

    color_order = [int(rules[0]["src"]), int(rules[1]["src"]), int(rules[2]["src"])]
    color_order.sort(key=lambda color: -len(footprints[color][3]))

    for color in color_order:
        count = random.randint(ACTORS_PER_COLOR_MIN, ACTORS_PER_COLOR_MAX)
        positions: list[tuple[int, int]] = []
        for _ in range(ACTOR_RETRIES):
            if len(positions) >= count:
                break
            r = random.randint(0, h - 1)
            c = random.randint(0, w - 1)
            center = (int(r), int(c))
            if center in occupied or center in reserved:
                continue

            step_cells: dict[int, set[tuple[int, int]]] = {}
            ok = True
            for step in (1, 2, 3):
                offsets = footprints[color][step]
                if not _fits_offsets(offsets, center, h, w):
                    ok = False
                    break
                cells = _shift_offsets(offsets, center)
                if cells & step_used[step]:
                    ok = False
                    break
                step_cells[step] = cells

            if not ok:
                continue

            positions.append(center)
            positions_by_color.setdefault(int(color), []).append(center)
            occupied.add(center)
            _reserve_4neighbors(reserved, [center])
            for step in (1, 2, 3):
                step_used[step].update(step_cells[step])

        if len(positions) < ACTORS_PER_COLOR_MIN:
            return False, {}, {}
        for r, c in positions:
            inp[r, c] = int(color)

    return True, positions_by_color, step_used


def _place_bystanders(
    used_colors: set[int],
    inp: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
    step_used: dict[int, set[tuple[int, int]]],
) -> list[tuple[int, int, int]]:
    bystander_colors = [c for c in NON_BG_NON_TEAL_COLORS if int(c) not in used_colors]
    if not bystander_colors:
        return []

    h, w = inp.shape
    placed: list[tuple[int, int, int]] = []
    n = random.randint(0, BYSTANDER_MAX)
    blocked = set(step_used[1]) | set(step_used[2]) | set(step_used[3])

    for _ in range(n):
        for _ in range(BYSTANDER_RETRIES):
            r = random.randint(0, h - 1)
            c = random.randint(0, w - 1)
            if (r, c) in occupied or (r, c) in reserved or (r, c) in blocked:
                continue
            color = int(random.choice(bystander_colors))
            inp[r, c] = int(color)
            placed.append((int(r), int(c), color))
            occupied.add((int(r), int(c)))
            _reserve_4neighbors(reserved, [(r, c)])
            break

    return placed


def _apply_rel_rule(grid: np.ndarray, src: int, tgt: int, rel: list[tuple[int, int]], bg: int) -> np.ndarray:
    current = np.asarray(grid, dtype=int)
    h, w = current.shape
    rel_rows = np.array([int(r) for r, _ in rel], dtype=int)
    rel_cols = np.array([int(c) for _, c in rel], dtype=int)

    out = current.copy()
    active_mask = (current == int(src)) | (current == int(tgt))
    out[active_mask] = int(bg)

    for actor_color, fill_color in ((src, tgt), (tgt, src)):
        for sr, sc in map(tuple, np.argwhere(current == int(actor_color))):
            sr = int(sr)
            sc = int(sc)
            rows = sr + rel_rows
            cols = sc + rel_cols
            ok = (0 <= rows) & (rows < h) & (0 <= cols) & (cols < w)
            out[rows[ok], cols[ok]] = int(fill_color)

    return out


def _simulate_chain_grid(base_grid: np.ndarray, rules: list[dict], bg: int = 0) -> np.ndarray:
    current = np.asarray(base_grid, dtype=int)
    for rule in rules:
        current = _apply_rel_rule(current, int(rule["src"]), int(rule["tgt"]), rule["rel"], bg=int(bg))
    return current


def _build_expected_output(
    inp: np.ndarray,
    rules: list[dict],
) -> np.ndarray:
    base = np.asarray(inp, dtype=int).copy()
    base[base == 8] = 0
    for rule in rules:
        sr, sc = map(int, rule["source_pos"])
        tr, tc = map(int, rule["target_pos"])
        base[sr, sc] = 0
        base[tr, tc] = 0
    return _simulate_chain_grid(base, rules, bg=0)


def _build_independent_baseline_output(inp: np.ndarray, rules: list[dict]) -> np.ndarray:
    base = np.asarray(inp, dtype=int).copy()
    base[base == 8] = 0
    for rule in rules:
        sr, sc = map(int, rule["source_pos"])
        tr, tc = map(int, rule["target_pos"])
        base[sr, sc] = 0
        base[tr, tc] = 0

    out = np.asarray(base, dtype=int).copy()
    active_colors = {int(rule["src"]) for rule in rules} | {int(rule["tgt"]) for rule in rules}
    for color in active_colors:
        out[out == int(color)] = 0

    for rule in rules:
        rel_rows = np.array([int(r) for r, _ in rule["rel"]], dtype=int)
        rel_cols = np.array([int(c) for _, c in rule["rel"]], dtype=int)
        for actor_color, fill_color in (
            (int(rule["src"]), int(rule["tgt"])),
            (int(rule["tgt"]), int(rule["src"])),
        ):
            for sr, sc in map(tuple, np.argwhere(base == int(actor_color))):
                sr = int(sr)
                sc = int(sc)
                rows = sr + rel_rows
                cols = sc + rel_cols
                ok = (
                    (0 <= rows)
                    & (rows < out.shape[0])
                    & (0 <= cols)
                    & (cols < out.shape[1])
                )
                out[rows[ok], cols[ok]] = int(fill_color)
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_DIM_MIN, GRID_DIM_MAX)
    w = random.randint(GRID_DIM_MIN, GRID_DIM_MAX)

    rules, footprints, used_colors = _build_rules()
    if rules is None or footprints is None or used_colors is None:
        return None

    inp = np.zeros((h, w), dtype=int)
    occupied: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()

    if not _place_legends(rules, inp, occupied, reserved):
        return None
    ok, _positions, step_used = _place_actors(rules, footprints, inp, occupied, reserved)
    if not ok:
        return None
    _place_bystanders(used_colors, inp, occupied, reserved, step_used)

    expected = _build_expected_output(inp, rules)
    baseline = _build_independent_baseline_output(inp, rules)
    if np.array_equal(expected, baseline):
        return None

    try:
        if np.array_equal(main(inp), expected):
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
