# Combo variant of abc82100 (index 15)
# combo_spec: D6/3+D13/0
# Name: 源色目标色交换+强制双向映射

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg


# Design analysis for the combo variant
# -------------------------------------
# a) concretization_and_trace:
# - The original task learns one directed legend `A -> B` next to one teal stencil,
#   then every loose A pixel stamps that stencil in color B. Loose B pixels are
#   inert and are either preserved or consumed depending on how colors are reused.
# - This combo fuses two changes into one consistent mechanism:
#   1. the execution is a color swap, not a one-way recolor;
#   2. the input is only valid when that swap is demonstrated symmetrically.
# - Concretely, one logical rule is a swap pair {A, B} backed by one teal stencil.
#   The generator shows it as two reciprocal legends around the same anchor:
#   `A B [teal-anchor] A B` along one axis. Read from left to right, the left pair
#   proves A->B; read from right to left, the right pair proves B->A.
# - Example trace:
#   stencil relative cells = {(0,0), (0,1), (1,0)}
#   swap colors = {2, 4}
#   original behavior:
#     loose 2 at (7,7) -> stamp color 4 there;
#     loose 4 at (11,3) does not activate the stencil.
#   combo behavior:
#     loose 2 at (7,7) -> stamp color 4 there;
#     loose 4 at (11,3) -> stamp the same stencil in color 2.
# - The visible difference is therefore twofold: the legend itself is explicitly
#   reciprocal, and both endpoint colors reappear later as active scattered actors.
#
# b) complexity_calibration:
# - If the generator emitted reciprocal legends but only scattered one color, the
#   task would collapse back toward the original one-way behavior. To avoid that,
#   every swap pair gets repeated support in both directions.
# - If the generator emitted only reciprocal legends, the old solver would often
#   produce the same output accidentally by learning both directed rules. To make
#   the D13 constraint observable, each sample also contains one asymmetric control
#   legend with its own scattered source pixels; the combo leaves that control
#   untouched, while the old one-way solver would incorrectly activate it.
# - Colors are still restricted to disjoint 2-color pairs from the original
#   non-teal palette, so each active color belongs to exactly one stencil.
# - Output-stamp overlap is forbidden. Without that constraint, opposite-direction
#   swaps could overwrite each other and hide the bidirectional effect.
#
# c) implicit_assumption_audit_of_original_main:
# - Original `source_colors.add(rule["source_color"])` assumed only one endpoint
#   color is consumed. That breaks here because both legend colors are active.
# - Original preservation logic `color in source_colors` is no longer sufficient;
#   it would leak scattered pixels of the opposite endpoint into the output.
# - Original stamping iterated over only `g == rule["source_color"]`, which hard-
#   coded the one-way transform. The combo must stamp both directions.
# - Original rule acceptance treated every discovered directed legend as usable.
#   That also breaks: under this combo, a single A->B legend is not enough. The
#   solver must first group legends by unordered color pair + shared anchor +
#   shared stencil, then require that both directions are present.
#
# d) placement_legality:
# - Reciprocal legends share one stencil anchor but occupy opposite sides of the
#   same axis, so they remain disjoint while proving both directions.
# - The generator reserves 4-neighbors around every legend cell and every scattered
#   actor center, preventing accidental extra adjacent color pairs.
# - Candidate actor centers are accepted only when their stamped output footprint
#   is in-bounds after clipping, non-empty, and disjoint from every other future
#   stamp and from all legend/stencil cells.
#
# e) visual_observability:
# - A human solver can infer the rule from 2-4 examples because the reciprocal
#   structure is explicit in the input: each stencil is flanked by the same two
#   colors in opposite directions, and both colors recur as isolated actors.
# - The extra asymmetric control legend makes the "must be reciprocal" condition
#   visible instead of merely implicit: one-way examples do not trigger stamping.
# - No invisible ordering by numeric color id is used. The swap is cued only by
#   the mirrored geometry and repeated paired evidence.
#
# f) visual_cleanliness:
# - Each logical rule still uses one compact teal component, one compact mirrored
#   legend, and isolated single-pixel actors. That keeps the grid readable.
# - Preventing output-stamp overlap and keeping bystanders off future stamp cells
#   avoids ambiguous mixed-color regions.
#
# g) change_inferability:
# - The color change is always a 2-color exchange tied to one shared stencil.
#   Because both colors produce the same shape in the opposite color repeatedly,
#   the mapping is structurally signaled rather than guessed from raw palette ids.
#
# h) new_solver_assumptions_to_match_generator:
# - Each active color appears in at most one reciprocal swap group.
# - Reciprocal legends that belong together share the same anchor location and the
#   same absolute teal component.
# - Every non-legend pixel of an active color is an actor center, not a hidden
#   second legend fragment. The generator enforces this with 4-neighbor spacing.
#
# i) post-change_main_audit:
# - The loop that preserves unrelated colors now assumes any non-active color is a
#   bystander and should survive unchanged; the generator guarantees bystanders use
#   colors outside all swap pairs.
# - The stamping loop assumes all actor centers of a given active color should use
#   exactly one reciprocal rule. Disjoint color pairing in generation guarantees
#   there is no ambiguity about which stencil owns a color.


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _anchor_distance(pattern: np.ndarray, anchor: tuple[int, int]) -> int:
    return min(_chebyshev((int(r), int(c)), anchor) for r, c in pattern)


def _pattern_key(pattern: np.ndarray) -> tuple[tuple[int, int], ...]:
    return tuple((int(r), int(c)) for r, c in np.asarray(pattern, dtype=int))


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


def _discover_swap_groups(rules: list[dict]) -> list[dict]:
    grouped: dict[tuple[frozenset[int], tuple[int, int], tuple[tuple[int, int], ...]], dict] = {}
    key_order: list[tuple[frozenset[int], tuple[int, int], tuple[tuple[int, int], ...]]] = []

    for rule in rules:
        color_pair = frozenset((int(rule["source_color"]), int(rule["target_color"])))
        if len(color_pair) != 2:
            continue
        key = (color_pair, tuple(rule["anchor"]), _pattern_key(rule["pattern"]))
        if key not in grouped:
            grouped[key] = {"dirs": {}, "rules": []}
            key_order.append(key)
        grouped[key]["dirs"][(int(rule["source_color"]), int(rule["target_color"]))] = rule
        grouped[key]["rules"].append(rule)

    swap_groups: list[dict] = []
    for key in key_order:
        dirs = grouped[key]["dirs"]
        first_rule = grouped[key]["rules"][0]
        a = int(first_rule["source_color"])
        b = int(first_rule["target_color"])
        if (b, a) not in dirs:
            continue
        swap_groups.append(
            {
                "colors": (a, b),
                "rules_by_source": {
                    a: dirs[(a, b)],
                    b: dirs[(b, a)],
                },
            }
        )

    return swap_groups


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use mirrored local legends to learn a stencil-backed two-color swap.

    Input structure:
    - Black background with isolated colored pixels and teal (8) stencil components.
    - A usable rule is not a lone directed pair: the same teal component must be
      flanked by reciprocal legends A->B and B->A that share one anchor region.
      Unpaired legends may appear as distractors but do not activate any stencil.
    - Away from the legend, isolated pixels may use either endpoint color of that
      learned swap pair.

    Rule:
    - Keep only legend information that forms a bidirectional pair around one
      stencil, then treat the two endpoint colors as a reversible mapping.
    - For each non-legend A pixel, stamp the associated teal stencil at that pixel
      recolored to B; for each non-legend B pixel, stamp the same stencil recolored
      back to A.
    - Remove all teal template pixels, and erase only the legend pixels that belong
      to a validated reciprocal swap. Unpaired legend colors remain ordinary pixels.

    Key insight: The reciprocal legend and the bidirectional execution describe the
    same relationship, so one shared stencil encodes a visible A<->B exchange.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    pattern_color = 8
    h, w = g.shape

    rules = _discover_rules(g, bg=bg, pattern_color=pattern_color)
    swap_groups = _discover_swap_groups(rules)

    legend_pixels: set[tuple[int, int]] = set()
    active_colors: set[int] = set()
    color_positions: dict[int, np.ndarray] = {}

    for group in swap_groups:
        for source_color, rule in group["rules_by_source"].items():
            active_colors.add(int(source_color))
            legend_pixels.add(tuple(rule["source_pos"]))
            legend_pixels.add(tuple(rule["target_pos"]))

    for color in active_colors:
        color_positions[color] = np.argwhere(g == int(color))

    out = np.full_like(g, int(bg))

    # Preserve colored pixels that are neither legend cells nor members of a
    # validated reciprocal swap pair.
    for r, c in map(tuple, np.argwhere((g != int(bg)) & (g != pattern_color))):
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        if (r, c) in legend_pixels or color in active_colors:
            continue
        out[r, c] = color

    for group in swap_groups:
        for source_color, rule in group["rules_by_source"].items():
            ar, ac = rule["anchor"]
            pattern = rule["pattern"]
            target_color = int(rule["target_color"])
            for sr, sc in map(tuple, color_positions[int(source_color)]):
                sr = int(sr)
                sc = int(sc)
                if (sr, sc) in legend_pixels:
                    continue
                dr = sr - int(ar)
                dc = sc - int(ac)
                rows = pattern[:, 0] + dr
                cols = pattern[:, 1] + dc
                ok = (0 <= rows) & (rows < h) & (0 <= cols) & (cols < w)
                out[rows[ok], cols[ok]] = target_color

    return out


# --------------- generator (c) ---------------

# Grid dimensions
GRID_DIM_MAX = 20
MIN_DIM_BY_SWAP_PAIRS = {1: 7, 2: 11, 3: 15}

# Swap-pair sampling
SWAP_PAIR_WEIGHTS = [1, 1, 2, 2, 3, 3]
NON_BG_NON_TEAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

# Stencil size
STENCIL_MAX_CAP = 12
STENCIL_MIN_AREA_DIV = 100
STENCIL_MAX_AREA_DIV = 25
STENCIL_CAP_BY_PAIRS = {1: 12, 2: 10, 3: 8, 4: 6}

# Scattered actors per color within a swap pair
SCATTERED_PER_COLOR_MIN = 2
SCATTERED_PER_COLOR_MAX_CAP = 6
SCATTERED_PER_COLOR_AREA_DIV = 40

# Asymmetric control rule: proves that one-way legends are ignored.
CONTROL_SCATTERED_MIN = 2
CONTROL_SCATTERED_MAX_CAP = 5
CONTROL_SCATTERED_AREA_DIV = 45

# Bystander pixels
BYSTANDER_MAX = 5

# Placement retry budgets
LEGEND_RETRIES = 700
SCATTERED_RETRIES_PER_PIXEL = 100
BYSTANDER_RETRIES = 80
OUTER_RETRIES = 600

_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _reserve_4neighbors(reserved: set[tuple[int, int]], cells) -> None:
    for r, c in cells:
        for dr, dc in _DIRS4:
            reserved.add((int(r) + dr, int(c) + dc))


def _make_stencil(min_pixels: int, max_pixels: int):
    n = random.randint(min_pixels, max_pixels)
    coords = {(0, 0)}
    for _ in range(n - 1):
        frontier = set()
        for r, c in coords:
            for dr, dc in _DIRS8:
                candidate = (r + dr, c + dc)
                if candidate not in coords:
                    frontier.add(candidate)
        coords.add(random.choice(list(frontier)))
    coords_list = sorted(coords)
    anchor = random.choice(coords_list)
    return coords_list, anchor


def _stamp_cells(rel, anchor_r: int, anchor_c: int, h: int, w: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for rr, cc in rel:
        nr = int(anchor_r + rr)
        nc = int(anchor_c + cc)
        if 0 <= nr < h and 0 <= nc < w:
            cells.add((nr, nc))
    return cells


def _build_swaps(num_pairs: int, area: int):
    colors = NON_BG_NON_TEAL_COLORS[:]
    random.shuffle(colors)

    max_s = min(
        STENCIL_MAX_CAP,
        STENCIL_CAP_BY_PAIRS[num_pairs],
        max(2, area // STENCIL_MAX_AREA_DIV),
    )
    min_s = min(max(1, area // STENCIL_MIN_AREA_DIV), max_s)
    sc_max = min(
        SCATTERED_PER_COLOR_MAX_CAP,
        max(SCATTERED_PER_COLOR_MIN, area // SCATTERED_PER_COLOR_AREA_DIV),
    )

    swaps = []
    used_colors = set()
    for i in range(num_pairs):
        color_a = colors[2 * i]
        color_b = colors[2 * i + 1]
        stencil, anchor = _make_stencil(min_s, max_s)
        rel = [(int(r - anchor[0]), int(c - anchor[1])) for r, c in stencil]
        swaps.append(
            {
                "a": int(color_a),
                "b": int(color_b),
                "rel": rel,
                "n_a": random.randint(SCATTERED_PER_COLOR_MIN, sc_max),
                "n_b": random.randint(SCATTERED_PER_COLOR_MIN, sc_max),
            }
        )
        used_colors.update((int(color_a), int(color_b)))

    return swaps, used_colors


def _build_control_rule(available_colors: list[int], area: int):
    if len(available_colors) < 2:
        return None

    max_s = min(STENCIL_MAX_CAP, max(2, area // STENCIL_MAX_AREA_DIV))
    min_s = min(max(1, area // STENCIL_MIN_AREA_DIV), max_s)
    sc_max = min(
        CONTROL_SCATTERED_MAX_CAP,
        max(CONTROL_SCATTERED_MIN, area // CONTROL_SCATTERED_AREA_DIV),
    )
    src, tgt = random.sample(available_colors, 2)
    stencil, anchor = _make_stencil(min_s, max_s)
    rel = [(int(r - anchor[0]), int(c - anchor[1])) for r, c in stencil]
    return {
        "src": int(src),
        "tgt": int(tgt),
        "rel": rel,
        "n_src": random.randint(CONTROL_SCATTERED_MIN, sc_max),
    }


def _place_legends(swaps, inp, occupied, reserved):
    h, w = inp.shape
    for swap in swaps:
        rel = swap["rel"]
        placed = False
        for _ in range(LEGEND_RETRIES):
            dr, dc = random.choice(_DIRS4)
            ar = random.randint(0, h - 1)
            ac = random.randint(0, w - 1)

            legend_cells = {
                (ar - 2 * dr, ac - 2 * dc),
                (ar - dr, ac - dc),
                (ar + dr, ac + dc),
                (ar + 2 * dr, ac + 2 * dc),
            }
            if not all(0 <= r < h and 0 <= c < w for r, c in legend_cells):
                continue

            teal_cells = _stamp_cells(rel, ar, ac, h, w)
            if len(teal_cells) != len(rel):
                continue

            all_cells = legend_cells | teal_cells
            if len(all_cells) != len(legend_cells) + len(teal_cells):
                continue
            if all_cells & (occupied | reserved):
                continue

            a = int(swap["a"])
            b = int(swap["b"])
            inp[ar - 2 * dr, ac - 2 * dc] = a
            inp[ar - dr, ac - dc] = b
            inp[ar + dr, ac + dc] = a
            inp[ar + 2 * dr, ac + 2 * dc] = b
            for nr, nc in teal_cells:
                inp[nr, nc] = 8

            occupied.update(all_cells)
            _reserve_4neighbors(reserved, all_cells)
            placed = True
            break

        if not placed:
            return False

    return True


def _place_control_legend(control_rule, inp, occupied, reserved):
    h, w = inp.shape
    rel = control_rule["rel"]
    for _ in range(LEGEND_RETRIES):
        dr, dc = random.choice(_DIRS4)
        sr = random.randint(0, h - 1)
        sc = random.randint(0, w - 1)
        tr = sr + dr
        tc = sc + dc
        ar = sr + 2 * dr
        ac = sc + 2 * dc
        if not (0 <= tr < h and 0 <= tc < w and 0 <= ar < h and 0 <= ac < w):
            continue

        teal_cells = _stamp_cells(rel, ar, ac, h, w)
        if len(teal_cells) != len(rel):
            continue

        all_cells = {(int(sr), int(sc)), (int(tr), int(tc))} | teal_cells
        if len(all_cells) != 2 + len(teal_cells):
            continue
        if all_cells & (occupied | reserved):
            continue

        inp[sr, sc] = int(control_rule["src"])
        inp[tr, tc] = int(control_rule["tgt"])
        for nr, nc in teal_cells:
            inp[nr, nc] = 8

        occupied.update(all_cells)
        _reserve_4neighbors(reserved, all_cells)
        control_rule["legend_source_pos"] = (int(sr), int(sc))
        control_rule["legend_target_pos"] = (int(tr), int(tc))
        return True

    return False


def _place_scattered(swaps, inp, occupied, reserved, output_reserved):
    h, w = inp.shape
    legend_reserved = set(occupied)

    for swap in swaps:
        swap["scattered"] = {}
        for color_key, count_key in (("a", "n_a"), ("b", "n_b")):
            color = int(swap[color_key])
            target_count = int(swap[count_key])
            positions = []
            for _ in range(target_count * SCATTERED_RETRIES_PER_PIXEL):
                if len(positions) >= target_count:
                    break
                r = random.randint(0, h - 1)
                c = random.randint(0, w - 1)
                if (r, c) in occupied or (r, c) in reserved:
                    continue
                stamp = _stamp_cells(swap["rel"], r, c, h, w)
                if not stamp or (stamp & output_reserved) or (stamp & legend_reserved):
                    continue
                positions.append((int(r), int(c)))
                occupied.add((int(r), int(c)))
                _reserve_4neighbors(reserved, [(r, c)])
                output_reserved.update(stamp)
            if len(positions) < SCATTERED_PER_COLOR_MIN:
                return False
            swap["scattered"][color] = positions
            for r, c in positions:
                inp[r, c] = color
    return True


def _place_control_scattered(control_rule, inp, occupied, reserved, output_reserved):
    h, w = inp.shape
    positions = []
    target_count = int(control_rule["n_src"])
    for _ in range(target_count * SCATTERED_RETRIES_PER_PIXEL):
        if len(positions) >= target_count:
            break
        r = random.randint(0, h - 1)
        c = random.randint(0, w - 1)
        if (r, c) in occupied or (r, c) in reserved:
            continue
        if (r, c) in output_reserved:
            continue
        positions.append((int(r), int(c)))
        occupied.add((int(r), int(c)))
        _reserve_4neighbors(reserved, [(r, c)])
    if len(positions) < CONTROL_SCATTERED_MIN:
        return False
    control_rule["scattered"] = positions
    for r, c in positions:
        inp[r, c] = int(control_rule["src"])
    return True


def _place_bystanders(used_colors, inp, occupied, reserved, output_reserved):
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
            if (r, c) in occupied or (r, c) in reserved or (r, c) in output_reserved:
                continue
            color = int(random.choice(bystander_colors))
            inp[r, c] = color
            placed.append((int(r), int(c), color))
            occupied.add((int(r), int(c)))
            _reserve_4neighbors(reserved, [(r, c)])
            break
    return placed


def _build_output(swaps, bystander_pos, h: int, w: int):
    output = np.zeros((h, w), dtype=int)
    for r, c, color in bystander_pos:
        output[int(r), int(c)] = int(color)
    for swap in swaps:
        for sr, sc in swap["scattered"][int(swap["a"])]:
            for rr, cc in swap["rel"]:
                nr = int(sr + rr)
                nc = int(sc + cc)
                if 0 <= nr < h and 0 <= nc < w:
                    output[nr, nc] = int(swap["b"])
        for sr, sc in swap["scattered"][int(swap["b"])]:
            for rr, cc in swap["rel"]:
                nr = int(sr + rr)
                nc = int(sc + cc)
                if 0 <= nr < h and 0 <= nc < w:
                    output[nr, nc] = int(swap["a"])
    return output


def _overlay_control_passthrough(output: np.ndarray, control_rule) -> np.ndarray:
    out = np.array(output, copy=True)
    sr, sc = control_rule["legend_source_pos"]
    tr, tc = control_rule["legend_target_pos"]
    out[int(sr), int(sc)] = int(control_rule["src"])
    out[int(tr), int(tc)] = int(control_rule["tgt"])
    for r, c in control_rule["scattered"]:
        out[int(r), int(c)] = int(control_rule["src"])
    return out


def _gen_attempt():
    num_pairs = random.choice(SWAP_PAIR_WEIGHTS)
    min_dim = MIN_DIM_BY_SWAP_PAIRS[num_pairs]
    if num_pairs == 3:
        dim_floor = max(min_dim, GRID_DIM_MAX - 3)
    else:
        dim_floor = min_dim
    h = random.randint(dim_floor, GRID_DIM_MAX)
    w = random.randint(dim_floor, GRID_DIM_MAX)

    swaps, used_colors = _build_swaps(num_pairs, h * w)
    control_rule = _build_control_rule(
        [c for c in NON_BG_NON_TEAL_COLORS if c not in used_colors],
        h * w,
    )
    if control_rule is None:
        return None
    used_colors.update((int(control_rule["src"]), int(control_rule["tgt"])))
    occupied: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()
    output_reserved: set[tuple[int, int]] = set()
    inp = np.zeros((h, w), dtype=int)

    if not _place_legends(swaps, inp, occupied, reserved):
        return None
    if not _place_control_legend(control_rule, inp, occupied, reserved):
        return None
    if not _place_scattered(swaps, inp, occupied, reserved, output_reserved):
        return None
    if not _place_control_scattered(control_rule, inp, occupied, reserved, output_reserved):
        return None
    bystander_pos = _place_bystanders(used_colors, inp, occupied, reserved, output_reserved)

    output = _build_output(swaps, bystander_pos, h, w)
    output = _overlay_control_passthrough(output, control_rule)
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
