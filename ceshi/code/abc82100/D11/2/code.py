# Variant of abc82100
# Dimension: D11 — teal模板的连通性和数量 (meta)
# Change: 每个规则示例附近有一个teal连通分量作为该规则的模板... → 模板使用不同颜色区分（非统一teal）

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg


# design_analysis:
#   a_concretization_and_trace:
#     original_behavior:
#       - Every rule legend used the same template color 8, so solver first collected
#         all teal components globally, then attached the nearest one to each
#         source->target pair.
#       - Example: if blue->red is followed by an anchor near a teal L-shape, every
#         scattered blue pixel stamps that L-shape in red.
#     variant_behavior:
#       - Each rule still has one nearby stencil, but the stencil color is now
#         rule-specific and distinct across rules inside the same grid.
#       - Example: blue->red may point to a teal L-shape while yellow->green points
#         to a grey zigzag. A scattered blue pixel stamps the L-shape in red; a
#         scattered yellow pixel stamps the zigzag in green.
#     flow_impact:
#       - Rule discovery can no longer hard-code pattern_color=8.
#       - Output cleanup can no longer erase templates by color id alone; it must
#         remove the exact legend/template cells recovered from the input.
#   b_complexity_calibration:
#     - To keep the color cue visible, multiple-rule instances must remain common.
#     - ARC has only 9 non-background colors, so supporting 4 rules with distinct
#       source, target, and template colors is impossible.
#     - The minimal compatible adjustment is reducing the generator to 1-3 rules,
#       which still keeps repetition and avoids palette collisions.
#   c_implicit_assumption_audit:
#     - _discover_rules in the original code assumed every stencil pixel had color 8.
#       That fails immediately once templates can be grey/orange/etc.
#     - The original preservation pass treated every non-8 non-bg pixel as a normal
#       colored pixel candidate. In the variant, template pixels may be any color, so
#       preserving by color would leak legend stencils into the output.
#     - Stamping logic itself is still valid because the copied shape depends only on
#       relative coordinates, not on the input template color.
#   d_placement_legality:
#     - Generator assigns each rule a template color disjoint from all source/target
#       colors in the same grid.
#     - Rule legends, scattered sources, and bystanders reserve an 8-neighborhood so
#       no unrelated colored pixel can sit Chebyshev-distance 1 from an anchor and be
#       mistaken for a second stencil candidate.
#     - Each template remains a single 8-connected component placed fully in bounds.
#   e_visual_observability:
#     - The new behavior is directly visible because different legends carry visibly
#       different template colors, while the output still removes all legend regions
#       and stamps only target-colored shapes.
#     - The solver never sorts or branches on raw color ids for semantics beyond
#       equality/grouping, so the rule remains human-visible rather than numeric.
#   f_visual_cleanliness:
#     - Only one template component exists per template color, and bystanders never
#       reuse those colors. This keeps the legend region readable.
#     - Scattered source pixels stay isolated with extra diagonal spacing, preserving
#       the original task's uncluttered "legend plus dots" composition.
#   g_color_inference:
#     - Template color is only an identity tag for "which stencil belongs to which
#       legend", not a hidden arithmetic transform.
#     - Repeated scattered source pixels for each rule provide enough support for a
#       human to infer that the colored stencil shape, not its legend color, is what
#       gets copied and recolored to the target color.


ALL_NON_BG_COLORS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _anchor_distance(pattern: np.ndarray, anchor: tuple[int, int]) -> int:
    return min(_chebyshev((int(r), int(c)), anchor) for r, c in pattern)


def _pattern_components_by_color(grid: np.ndarray, bg: int) -> dict[int, list[np.ndarray]]:
    components: dict[int, list[np.ndarray]] = {}
    for color in sorted(set(map(int, grid.flatten()))):
        if color == int(bg):
            continue
        components[color] = components_mask(grid == color, connectivity=8)
    return components


def _discover_rules(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    patterns_by_color = _pattern_components_by_color(grid, bg)
    rules: list[dict] = []
    seen: set[tuple[tuple[int, int], tuple[int, int], int, int]] = set()

    for r in range(h):
        for c in range(w):
            source_color = int(grid[r, c])
            if source_color == int(bg):
                continue

            for dr, dc in _DIRS4:
                tr = r + dr
                tc = c + dc
                if not (0 <= tr < h and 0 <= tc < w):
                    continue

                target_color = int(grid[tr, tc])
                if target_color == int(bg) or target_color == source_color:
                    continue

                anchor = (tr + dr, tc + dc)
                candidates: list[tuple[int, int, int, int, np.ndarray]] = []
                for pattern_color, patterns in patterns_by_color.items():
                    if pattern_color in (source_color, target_color):
                        continue
                    for i, pattern in enumerate(patterns):
                        dist = _anchor_distance(pattern, anchor)
                        if dist <= 1:
                            candidates.append((int(dist), len(pattern), int(pattern_color), int(i), pattern))

                if not candidates:
                    continue

                candidates.sort(key=lambda item: (item[0], item[1]))
                _dist, _size, pattern_color, pattern_index, pattern = candidates[0]
                key = ((int(r), int(c)), (int(tr), int(tc)), int(pattern_color), int(pattern_index))
                if key in seen:
                    continue
                seen.add(key)
                rules.append(
                    {
                        "source_color": int(source_color),
                        "target_color": int(target_color),
                        "pattern_color": int(pattern_color),
                        "source_pos": (int(r), int(c)),
                        "target_pos": (int(tr), int(tc)),
                        "anchor": (int(anchor[0]), int(anchor[1])),
                        "pattern": np.asarray(pattern, dtype=int),
                    }
                )

    return rules


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn local source-target-stencil examples and stamp their shapes elsewhere.

    Input structure:
    - Black background with isolated colored source pixels and several stencil
      components; unlike the base task, those stencil components are not all teal.
    - Each rule example is a 4-neighbor colored pair source->target whose forward
      direction reaches an anchor touching a same-colored stencil component.

    Rule:
    - For each learned rule, copy that nearby stencil shape onto every other pixel
      of the rule's source color, recoloring the copied shape to the rule's target.
    - The translation aligns the legend anchor with the scattered source pixel.
    - Remove all legend pixels and all input stencil pixels after applying the rules.

    Key insight: The adjacent colored pair still tells you which source color turns
    into which target color; the change is that stencil identity is carried by its
    own local color, so the solver must recover the exact nearby component rather
    than assume every stencil is color 8.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    h, w = g.shape

    rules = _discover_rules(g, bg=bg)
    legend_pixels: set[tuple[int, int]] = set()
    template_pixels: set[tuple[int, int]] = set()
    source_colors: set[int] = set()
    for rule in rules:
        legend_pixels.add(rule["source_pos"])
        legend_pixels.add(rule["target_pos"])
        template_pixels.update(map(tuple, rule["pattern"]))
        source_colors.add(rule["source_color"])

    out = np.full_like(g, int(bg))

    # Preserve colored pixels that are neither legend/template cells nor consumed
    # scattered sources.
    for r, c in map(tuple, np.argwhere(g != int(bg))):
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        if (r, c) in legend_pixels or (r, c) in template_pixels or color in source_colors:
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


# --------------- generator (c) ---------------

# Grid dimensions
GRID_DIM_MAX = 20
MIN_DIM_BY_RULES = {1: 5, 2: 10, 3: 14}

# Rule sampling
RULE_COUNT_WEIGHTS = [1, 1, 2, 2, 3, 3]

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
LEGEND_RETRIES = 500
SCATTERED_RETRIES_PER_PIXEL = 50
BYSTANDER_RETRIES = 50
OUTER_RETRIES = 300


def _reserve_8neighbors(reserved: set[tuple[int, int]], cells) -> None:
    for r, c in cells:
        for dr, dc in _DIRS8:
            reserved.add((r + dr, c + dc))


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
    anchor_candidates = set()
    for r, c in coords:
        for dr, dc in _DIRS8:
            p = (r + dr, c + dc)
            if p not in coords:
                anchor_candidates.add(p)
    return coords_list, random.choice(sorted(anchor_candidates))


def _build_rules(num_rules: int, area: int):
    colors = ALL_NON_BG_COLORS[:]
    random.shuffle(colors)

    max_s = min(STENCIL_MAX_CAP, max(2, area // STENCIL_MAX_AREA_DIV))
    min_s = min(max(1, area // STENCIL_MIN_AREA_DIV), max_s)
    sc_max = min(SCATTERED_MAX_CAP, max(SCATTERED_MIN, area // SCATTERED_AREA_DIV))

    rules = []
    used_colors: set[int] = set()
    for i in range(num_rules):
        src_c = colors[2 * i]
        tgt_c = colors[2 * i + 1]
        pattern_c = colors[2 * num_rules + i]
        used_colors.update((src_c, tgt_c, pattern_c))
        stencil, anchor = _make_stencil(min_s, max_s)
        rel = [(r - anchor[0], c - anchor[1]) for r, c in stencil]
        rules.append(
            {
                "src": src_c,
                "tgt": tgt_c,
                "pattern_color": pattern_c,
                "rel": rel,
                "n_sc": random.randint(SCATTERED_MIN, sc_max),
            }
        )
    return rules, used_colors


def _place_legends(rules, inp: np.ndarray, occupied: set[tuple[int, int]], reserved: set[tuple[int, int]]):
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
            reverse_anchor = (sr - dr, sc - dc)
            reverse_guard = {
                (reverse_anchor[0], reverse_anchor[1]),
                *[(reverse_anchor[0] + rr, reverse_anchor[1] + cc) for rr, cc in _DIRS8],
            }
            if any(not (0 <= rr < H and 0 <= cc < W) for rr, cc in reverse_guard):
                continue

            pattern_cells = set()
            ok = True
            for rr, cc in rel:
                nr, nc = ar + rr, ac + cc
                if not (0 <= nr < H and 0 <= nc < W):
                    ok = False
                    break
                pattern_cells.add((nr, nc))
            if not ok:
                continue
            source_neighbors = {(sr + rr, sc + cc) for rr, cc in _DIRS4}
            target_neighbors = {(tr + rr, tc + cc) for rr, cc in _DIRS4}
            if pattern_cells & (source_neighbors | target_neighbors):
                continue
            if pattern_cells & reverse_guard:
                continue

            all_cells = {(sr, sc), (tr, tc), (ar, ac)} | pattern_cells
            if all_cells & (occupied | reserved):
                continue
            if reverse_guard & occupied:
                continue
            if reverse_guard & reserved:
                continue

            inp[sr, sc] = rule["src"]
            inp[tr, tc] = rule["tgt"]
            for nr, nc in pattern_cells:
                inp[nr, nc] = rule["pattern_color"]
            occupied.update(all_cells)
            _reserve_8neighbors(reserved, all_cells)
            reserved.update(reverse_guard)
            placed = True
            break
        if not placed:
            return False
    return True


def _place_scattered(rules, inp: np.ndarray, occupied: set[tuple[int, int]], reserved: set[tuple[int, int]]):
    H, W = inp.shape
    for rule in rules:
        positions = []
        for _ in range(rule["n_sc"] * SCATTERED_RETRIES_PER_PIXEL):
            if len(positions) >= rule["n_sc"]:
                break
            r, c = random.randint(0, H - 1), random.randint(0, W - 1)
            if (r, c) in occupied or (r, c) in reserved:
                continue
            positions.append((r, c))
            occupied.add((r, c))
            _reserve_8neighbors(reserved, [(r, c)])
        if len(positions) < SCATTERED_MIN:
            return False
        rule["scattered"] = positions
        for r, c in positions:
            inp[r, c] = rule["src"]
    return True


def _place_bystanders(used_colors, inp: np.ndarray, occupied: set[tuple[int, int]], reserved: set[tuple[int, int]]):
    bystander_colors = [c for c in ALL_NON_BG_COLORS if c not in used_colors]
    if not bystander_colors:
        return []

    H, W = inp.shape
    n = random.randint(0, min(BYSTANDER_MAX, len(bystander_colors) * 2))
    placed = []
    for _ in range(n):
        for _ in range(BYSTANDER_RETRIES):
            r, c = random.randint(0, H - 1), random.randint(0, W - 1)
            if (r, c) in occupied or (r, c) in reserved:
                continue
            bc = random.choice(bystander_colors)
            inp[r, c] = bc
            placed.append((r, c, bc))
            occupied.add((r, c))
            _reserve_8neighbors(reserved, [(r, c)])
            break
    return placed


def _build_output(rules, bystander_pos, H: int, W: int):
    output = np.zeros((H, W), dtype=int)
    for r, c, bc in bystander_pos:
        output[r, c] = bc
    for rule in rules:
        rel = rule["rel"]
        for sr, sc in rule["scattered"]:
            for rr, cc in rel:
                nr, nc = sr + rr, sc + cc
                if 0 <= nr < H and 0 <= nc < W:
                    output[nr, nc] = rule["tgt"]
    return output


def _gen_attempt():
    num_rules = random.choice(RULE_COUNT_WEIGHTS)
    min_dim = MIN_DIM_BY_RULES[num_rules]
    H = random.randint(min_dim, GRID_DIM_MAX)
    W = random.randint(min_dim, GRID_DIM_MAX)

    rules, used_colors = _build_rules(num_rules, H * W)
    occupied: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()
    inp = np.zeros((H, W), dtype=int)

    if not _place_legends(rules, inp, occupied, reserved):
        return None
    if not _place_scattered(rules, inp, occupied, reserved):
        return None
    bystander_pos = _place_bystanders(used_colors, inp, occupied, reserved)

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
