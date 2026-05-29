# Variant of abc82100
# Dimension: D6 — 颜色重着色规则 (transform)
# Change: teal模板像素统一重着色为规则的目标色（单一颜色替换）... → 源色与目标色交换（双向映射）

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg


# Design analysis for the D6 variant
# ----------------------------------
# source_task_baseline:
# - The reference task a25697e4 uses a learned 2-color relationship where a later
#   operation can swap the two colors instead of treating one as permanently
#   dominant. We only borrow that bidirectional recoloring idea here; the geometry
#   stays the original abc82100 legend+stencil mechanism.
#
# concretization_and_trace:
# - A legend still looks like one adjacent colored pair followed by a teal stencil:
#   source -> target -> anchor/stencil.
# - The visible change is away from the legend: scattered pixels are no longer only
#   source-colored. Both colors from the legend reappear as isolated pixels.
# - Example trace for one rule:
#   source=2, target=4, stencil={(0,0),(0,1),(1,0)} around the anchor.
#   Original behavior:
#     isolated 2 at (7,7) -> stamp a 4-colored L shape there;
#     isolated 4 does nothing and disappears because it is treated as a consumed
#     legend color only.
#   Variant behavior:
#     isolated 2 at (7,7) -> stamp the same L in color 4;
#     isolated 4 at (11,3) -> stamp the same L in color 2.
# - So the changed stage is the "which scattered colors are active" step. The
#   stencil geometry and rule discovery stay the same.
#
# complexity_calibration:
# - The variant would be visually weak if the generator still produced only
#   source-colored scattered pixels, because then the outputs would match the
#   original task exactly.
# - To make the swap observable, every rule gets repeated support in both
#   directions: at least two scattered source pixels and at least two scattered
#   target pixels.
# - Color pairs remain disjoint across rules, matching the original complexity and
#   keeping each active color tied to exactly one stencil.
#
# implicit_assumption_audit_of_original_main:
# - Original `source_colors.add(rule["source_color"])` assumed only the source
#   color is consumed. That is false now; both endpoints must be removed from the
#   "preserve ordinary pixels" pass.
# - Original preserve condition `color in source_colors` was harmless before, but
#   would incorrectly keep scattered target-colored actors in the variant.
# - Original stamping loop iterated only over `g == rule["source_color"]`. That
#   encoded the one-way mapping directly, so the variant must add the mirrored
#   `target -> source` pass.
# - Original output construction tolerated stamp overlap more easily because one
#   rule always painted a single fill color. In the variant, opposite-direction
#   stamps could overwrite each other with different colors, making results depend
#   on processing order. The generator therefore forbids output-stamp overlap.
#
# placement_legality:
# - Legend colored pair cells reserve their 4-neighbors so no scattered pixel can
#   accidentally sit next to a legend or form a fake source-target example.
# - Scattered actors of either color are also 4-neighbor separated from everything
#   else, so `_discover_rules` only sees the intended legend pairs.
# - Candidate actor centers are accepted only when their translated stencil stays
#   disjoint from already planned output stamps and from legend cells, keeping the
#   input and output readable and order-independent.
#
# visual_observability_and_cleanliness:
# - Humans can infer the rule from 2-4 examples because each rule shows one local
#   pair of colors and then multiple isolated pixels of both colors flipping into
#   the opposite-colored copy of the same stencil.
# - We avoid clutter by preventing stamp overlap, by using only isolated actor
#   pixels instead of larger movable objects, and by shrinking stencil/actor
#   budgets slightly when many rules share one grid.
#
# color_inference:
# - No hidden ordering by color id is used. The only color logic is the visible
#   paired relationship from the legend: the two non-teal colors trade roles.
#
# new_solver_assumptions_to_match_generator:
# - Each active color belongs to at most one discovered rule.
# - Every relevant non-legend pixel of an active color is isolated from other
#   colored pixels, so it should be treated as an actor center rather than a new
#   legend fragment.
# - The generator enforces both assumptions by choosing disjoint rule colors and
#   reserving 4-neighborhoods during placement.


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
    Learn each legend pair as a reversible stencil-color swap.

    Input structure:
    - Black background with isolated colored pixels and teal (8) stencil components.
    - A rule example is a 4-neighbor ordered pair source->target whose continued
      direction reaches an anchor touching one teal stencil component.
    - Away from the legend, isolated pixels may use either endpoint color of the
      learned pair.

    Rule:
    - For each rule example, reuse that teal component as a stencil centered on
      every non-legend pixel of the source color and recolor it to the target color.
    - Apply the same stencil symmetrically to every non-legend pixel of the target
      color, recoloring it back to the source color.
    - Remove all legend pixels and all teal template pixels in the output.

    Key insight: The adjacent colored pair does not define a one-way replacement;
    it defines a reversible color pair, and the nearby teal component is the
    shared stencil that gets stamped in the opposite endpoint color.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    pattern_color = 8
    h, w = g.shape

    rules = _discover_rules(g, bg=bg, pattern_color=pattern_color)
    legend_pixels: set[tuple[int, int]] = set()
    active_colors: set[int] = set()
    color_positions: dict[int, np.ndarray] = {}
    for rule in rules:
        legend_pixels.add(rule["source_pos"])
        legend_pixels.add(rule["target_pos"])
        active_colors.add(rule["source_color"])
        active_colors.add(rule["target_color"])

    for color in active_colors:
        color_positions[color] = np.argwhere(g == int(color))

    out = np.full_like(g, int(bg))

    # Preserve colored pixels that are neither legend pixels nor participants in
    # any reversible rule.
    for r, c in map(tuple, np.argwhere((g != int(bg)) & (g != pattern_color))):
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        if (r, c) in legend_pixels or color in active_colors:
            continue
        out[r, c] = color

    for rule in rules:
        ar, ac = rule["anchor"]
        pattern = rule["pattern"]
        for actor_color, fill_color in (
            (rule["source_color"], rule["target_color"]),
            (rule["target_color"], rule["source_color"]),
        ):
            for sr, sc in map(tuple, color_positions[actor_color]):
                sr = int(sr)
                sc = int(sc)
                if (sr, sc) in legend_pixels:
                    continue
                dr = sr - ar
                dc = sc - ac
                rows = pattern[:, 0] + dr
                cols = pattern[:, 1] + dc
                ok = (0 <= rows) & (rows < h) & (0 <= cols) & (cols < w)
                out[rows[ok], cols[ok]] = int(fill_color)

    return out


# --------------- generator (c) ---------------

# ── CFG: tunable generation parameters ──────────────────────────────
# Grid dimensions
GRID_DIM_MAX = 20
MIN_DIM_BY_RULES = {1: 5, 2: 10, 3: 14, 4: 18}

# Rule sampling
RULE_COUNT_WEIGHTS = [1, 1, 2, 2, 3, 3, 4]
NON_BG_NON_TEAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

# Stencil size
STENCIL_MAX_CAP = 12
STENCIL_MIN_AREA_DIV = 100
STENCIL_MAX_AREA_DIV = 25
STENCIL_CAP_BY_RULES = {1: 12, 2: 10, 3: 8, 4: 6}

# Scattered actors per rule: both directions must appear.
SCATTERED_TOTAL_MIN = 4
SCATTERED_TOTAL_MAX_CAP = 8
SCATTERED_TOTAL_AREA_DIV = 30
SCATTERED_PER_ROLE_MIN = 2
SCATTERED_TOTAL_CAP_BY_RULES = {1: 8, 2: 7, 3: 6, 4: 5}

# Bystander pixels
BYSTANDER_MAX = 5

# Placement retry budgets
LEGEND_RETRIES = 700
SCATTERED_RETRIES_PER_PIXEL = 100
BYSTANDER_RETRIES = 80
OUTER_RETRIES = 500

_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _reserve_4neighbors(reserved: set[tuple[int, int]], cells) -> None:
    """Add all 4-neighbors of *cells* into *reserved*."""
    for r, c in cells:
        for dr, dc in _DIRS4:
            reserved.add((int(r + dr), int(c + dc)))


def _make_stencil(min_pixels: int, max_pixels: int):
    """Generate a random 8-connected stencil. Returns (coords_list, anchor)."""
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
    """Assign color pairs and generate stencils for *num_rules* rules."""
    colors = NON_BG_NON_TEAL_COLORS[:]
    random.shuffle(colors)

    max_s = min(
        STENCIL_MAX_CAP,
        STENCIL_CAP_BY_RULES[num_rules],
        max(2, area // STENCIL_MAX_AREA_DIV),
    )
    min_s = min(max(1, area // STENCIL_MIN_AREA_DIV), max_s)
    sc_total_max = min(
        SCATTERED_TOTAL_MAX_CAP,
        SCATTERED_TOTAL_CAP_BY_RULES[num_rules],
        max(SCATTERED_TOTAL_MIN, area // SCATTERED_TOTAL_AREA_DIV),
    )

    rules, used = [], set()
    for i in range(num_rules):
        src_c, tgt_c = colors[2 * i], colors[2 * i + 1]
        used.update((src_c, tgt_c))
        stencil, anchor = _make_stencil(min_s, max_s)
        rel = [(int(r - anchor[0]), int(c - anchor[1])) for r, c in stencil]
        total = random.randint(SCATTERED_TOTAL_MIN, sc_total_max)
        n_src = random.randint(SCATTERED_PER_ROLE_MIN, total - SCATTERED_PER_ROLE_MIN)
        n_tgt = total - n_src
        rules.append(
            {
                "src": src_c,
                "tgt": tgt_c,
                "rel": rel,
                "n_src": n_src,
                "n_tgt": n_tgt,
            }
        )
    return rules, used


def _stamp_cells(rel, center: tuple[int, int], H: int, W: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    cr, cc = center
    for rr, cc_rel in rel:
        nr = int(cr + rr)
        nc = int(cc + cc_rel)
        if 0 <= nr < H and 0 <= nc < W:
            cells.add((nr, nc))
    return cells


def _place_legends(rules, inp, occupied, reserved):
    """Place rule examples (source, target, anchor + teal stencil) on *inp*."""
    H, W = inp.shape
    for rule in rules:
        rel = rule["rel"]
        placed = False
        for _ in range(LEGEND_RETRIES):
            dr, dc = random.choice(_DIRS4)
            sr, sc = random.randint(0, H - 1), random.randint(0, W - 1)
            tr, tc = sr + dr, sc + dc
            ar, ac = sr + 2 * dr, sc + 2 * dc
            if not (0 <= tr < H and 0 <= tc < W and 0 <= ar < H and 0 <= ac < W):
                continue

            teal_cells = set()
            ok = True
            for rr, cc in rel:
                nr, nc = ar + rr, ac + cc
                if not (0 <= nr < H and 0 <= nc < W):
                    ok = False
                    break
                teal_cells.add((int(nr), int(nc)))
            if not ok:
                continue

            all_cells = {(int(sr), int(sc)), (int(tr), int(tc)), (int(ar), int(ac))} | teal_cells
            if all_cells & (occupied | reserved):
                continue

            inp[sr, sc] = rule["src"]
            inp[tr, tc] = rule["tgt"]
            for nr, nc in teal_cells:
                inp[nr, nc] = 8
            occupied.update(all_cells)
            _reserve_4neighbors(reserved, [(sr, sc), (tr, tc)])
            rule["legend_source_pos"] = (int(sr), int(sc))
            rule["legend_target_pos"] = (int(tr), int(tc))
            rule["legend_anchor"] = (int(ar), int(ac))
            placed = True
            break
        if not placed:
            return False
    return True


def _place_scattered(rules, inp, occupied, reserved):
    """Place isolated scattered pixels of both colors for each rule."""
    H, W = inp.shape
    legend_cells = set(occupied)
    output_used: set[tuple[int, int]] = set()

    for rule in rules:
        for role_name, actor_color, count in (
            ("src_positions", rule["src"], rule["n_src"]),
            ("tgt_positions", rule["tgt"], rule["n_tgt"]),
        ):
            positions = []
            attempts = count * SCATTERED_RETRIES_PER_PIXEL
            for _ in range(attempts):
                if len(positions) >= count:
                    break
                r, c = random.randint(0, H - 1), random.randint(0, W - 1)
                if (r, c) in occupied or (r, c) in reserved:
                    continue
                stamp = _stamp_cells(rule["rel"], (r, c), H, W)
                if not stamp or (stamp & output_used) or (stamp & legend_cells):
                    continue
                positions.append((int(r), int(c)))
                occupied.add((int(r), int(c)))
                _reserve_4neighbors(reserved, [(r, c)])
                output_used.update(stamp)
            if len(positions) < count:
                return False, None
            rule[role_name] = positions
            for r, c in positions:
                inp[r, c] = actor_color

    return True, output_used


def _place_bystanders(used_colors, inp, occupied, reserved, output_used):
    """Place unrelated colored pixels that stay unchanged in the output."""
    bystander_colors = [c for c in NON_BG_NON_TEAL_COLORS if c not in used_colors]
    if not bystander_colors:
        return []
    H, W = inp.shape
    n = random.randint(0, min(BYSTANDER_MAX, len(bystander_colors) * 2))
    placed = []
    for _ in range(n):
        for _ in range(BYSTANDER_RETRIES):
            r, c = random.randint(0, H - 1), random.randint(0, W - 1)
            if (r, c) in occupied or (r, c) in reserved or (r, c) in output_used:
                continue
            bc = random.choice(bystander_colors)
            inp[r, c] = bc
            placed.append((int(r), int(c), int(bc)))
            occupied.add((int(r), int(c)))
            _reserve_4neighbors(reserved, [(r, c)])
            break
    return placed


def _build_output(rules, bystander_pos, H, W):
    """Construct expected output by stamping each rule in both directions."""
    output = np.zeros((H, W), dtype=int)
    for r, c, bc in bystander_pos:
        output[r, c] = bc
    for rule in rules:
        for sr, sc in rule["src_positions"]:
            for rr, cc in rule["rel"]:
                nr, nc = sr + rr, sc + cc
                if 0 <= nr < H and 0 <= nc < W:
                    output[nr, nc] = rule["tgt"]
        for sr, sc in rule["tgt_positions"]:
            for rr, cc in rule["rel"]:
                nr, nc = sr + rr, sc + cc
                if 0 <= nr < H and 0 <= nc < W:
                    output[nr, nc] = rule["src"]
    return output


def _gen_attempt():
    """Single generation attempt. Returns input grid or None on failure."""
    num_rules = random.choice(RULE_COUNT_WEIGHTS)
    min_dim = MIN_DIM_BY_RULES[num_rules]
    if num_rules >= 4:
        dim_floor = max(min_dim, GRID_DIM_MAX - 1)
    elif num_rules == 3:
        dim_floor = max(min_dim, GRID_DIM_MAX - 3)
    else:
        dim_floor = min_dim
    H = random.randint(dim_floor, GRID_DIM_MAX)
    W = random.randint(dim_floor, GRID_DIM_MAX)

    rules, used_colors = _build_rules(num_rules, H * W)
    occupied, reserved = set(), set()
    inp = np.zeros((H, W), dtype=int)

    if not _place_legends(rules, inp, occupied, reserved):
        return None
    ok, output_used = _place_scattered(rules, inp, occupied, reserved)
    if not ok or output_used is None:
        return None
    bystander_pos = _place_bystanders(used_colors, inp, occupied, reserved, output_used)

    output = _build_output(rules, bystander_pos, H, W)
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
