# Variant of abc82100
# Dimension: D6 — 颜色重着色规则 (transform)
# Change: teal模板像素统一重着色为规则的目标色（单一颜色替换）... → 渐变着色（距锚点越远颜色越浅）

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg


# Design analysis for the D6 variant
# ----------------------------------
# design_analysis:
#   concretization:
#     - The local legend still teaches a rule with three aligned cells:
#       source -> target -> anchor, plus one teal stencil touching that anchor.
#     - The solver still learns only (source_color, target_color, anchor, stencil).
#       The change happens when the stencil is stamped onto scattered source pixels.
#     - New stamp coloring uses Manhattan distance to the anchor:
#         distance 0 -> target color
#         distance 1 -> lighter intermediate
#         distance >=2 -> lightest outer color
#     - To keep "lighter" visually meaningful without hidden color-id arithmetic,
#       target colors are restricted to four darker hues with explicit visible
#       hue-progressions:
#         blue(1)   -> teal(8)   -> yellow(4)
#         red(2)    -> orange(7) -> yellow(4)
#         green(3)  -> teal(8)   -> yellow(4)
#         maroon(9) -> red(2)    -> orange(7)
#   concrete_trace:
#     - Example legend: pink source 6, maroon target 9, anchor on the teal stencil.
#       Suppose the stencil has relative cells {(0,0), (1,0), (2,0), (1,1)}.
#     - Original task output for a scattered pink pixel at (8,8):
#         all translated stencil cells become maroon(9).
#     - Variant output:
#         (8,8)       distance 0 -> maroon(9)
#         (9,8),(9,9) distance 1/2? specifically (9,8)=1 -> red(2), (9,9)=2 -> orange(7)
#         (10,8)      distance 2 -> orange(7)
#     - The structural operation is unchanged; only the per-cell recoloring stage differs.
#   complexity_calibration:
#     - If stencils were allowed to be a single pixel or only distance-0/1 cells,
#       the variant would visually collapse back toward the original task.
#       Therefore every generated stencil contains at least one distance-1 cell
#       and one distance-2 cell.
#     - If scattered stamps overlapped heavily, later overwrites could erase the
#       gradient and make examples noisy. The generator therefore keeps stamped
#       footprints disjoint and fully in-bounds.
#   implicit_assumption_audit:
#     - _discover_rules originally only needed a nearby teal component. For the
#       gradient, the anchor must be a real stencil cell; otherwise no pixel would
#       realize the distance-0 color. The generator guarantees this by always
#       building stencils with relative cell (0,0).
#     - main() originally wrote one uniform target color. In the variant each
#       written pixel depends on its own anchor distance, so the code must carry
#       per-cell distance buckets alongside the stencil coordinates.
#     - Unconditional overwrites were harmless before because all pixels in one
#       stamp had the same color. They would now scramble gradients if many stamps
#       overlapped, so the generator avoids footprint overlap.
#   placement_legality:
#     - Legend placement rejects any direction where the teal stencil would cover
#       the legend's source or target cell.
#     - Scattered sources are only placed where the entire translated stencil fits
#       inside the grid and where that footprint does not intersect any reserved
#       region, legend, other footprint, or bystander.
#   visual_observability:
#     - Every rule repeats on 3-8 scattered sources, so each training grid shows
#       the same distance-color pattern multiple times.
#     - The outer colors are not chosen by sorting color ids; they come from a
#       small hand-authored visible gradient palette that can be inferred from the
#       examples directly.
#   cleanliness:
#     - Bystanders use only colors that never appear inside gradient stamps
#       (grey/pink when unused), which avoids output clutter.
#   solver_generator_contract:
#     - main() assumes target colors belong to the explicit gradient map and that
#       each learned stencil contains anchor-relative coordinates.
#     - generate_input() enforces those assumptions by construction and validates
#       every sample with main(input) == expected_output before returning.


TARGET_GRADIENTS: dict[int, tuple[int, int, int]] = {
    1: (1, 8, 4),
    2: (2, 7, 4),
    3: (3, 8, 4),
    9: (9, 2, 7),
}
TARGET_COLORS = list(TARGET_GRADIENTS.keys())
SOURCE_COLORS = [4, 5, 6, 7]
CLEAN_BYSTANDER_COLORS = [5, 6]


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _anchor_distance(pattern: np.ndarray, anchor: tuple[int, int]) -> int:
    return min(_chebyshev((int(r), int(c)), anchor) for r, c in pattern)


def _gradient_color(target_color: int, distance: int) -> int:
    palette = TARGET_GRADIENTS.get(int(target_color), (int(target_color), 8, 4))
    bucket = min(max(int(distance), 0), 2)
    return int(palette[bucket])


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

                pattern = np.asarray(pattern, dtype=int)
                distances = np.abs(pattern[:, 0] - int(anchor[0])) + np.abs(pattern[:, 1] - int(anchor[1]))
                rules.append(
                    {
                        "source_color": int(source_color),
                        "target_color": int(target_color),
                        "source_pos": (int(r), int(c)),
                        "target_pos": (int(tr), int(tc)),
                        "anchor": (int(anchor[0]), int(anchor[1])),
                        "pattern": pattern,
                        "distance_buckets": np.minimum(distances, 2).astype(int),
                    }
                )

    return rules


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn local source-target-stencil examples, then stamp those stencils elsewhere.

    Input structure:
    - The grid has a dominant background, isolated colored pixels, and teal(8)
      stencil components attached to local legend examples.
    - Each legend is a 4-neighbor source->target pair whose forward continuation
      lands on the anchor cell of a nearby teal stencil.

    Rule:
    - Every non-legend pixel of the learned source color becomes a translated copy
      of that legend's teal stencil, with the anchor aligned to the source pixel.
    - The copied stencil is not recolored uniformly: anchor-distance 0 keeps the
      target color, distance 1 uses a lighter intermediate, and distance >=2 uses
      the lightest outer color from that target's visible gradient palette.
    - All legend pixels and all teal template pixels are removed from the output.

    Key insight: The legend still teaches where to stamp and which target hue to
    use, but the output color of each stamped pixel now depends on its Manhattan
    distance from the stencil's anchor rather than being a single flat fill.
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

    # Preserve ordinary colored pixels that are not consumed by any learned rule.
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
        distance_buckets = rule["distance_buckets"]
        pattern_colors = np.array(
            [_gradient_color(rule["target_color"], int(dist)) for dist in distance_buckets],
            dtype=int,
        )
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
            out[rows[ok], cols[ok]] = pattern_colors[ok]

    return out


# --------------- generator (c) ---------------

# Grid dimensions
GRID_DIM_MAX = 20
MIN_DIM_BY_RULES = {1: 5, 2: 10, 3: 14, 4: 18}

# Rule sampling
RULE_COUNT_WEIGHTS = [1, 1, 2, 2, 3, 3, 4]
NON_BG_NON_TEAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

# Stencil size and shape
STENCIL_MAX_CAP = 12
STENCIL_MIN_AREA_DIV = 100
STENCIL_MAX_AREA_DIV = 25
STENCIL_RADIUS_CAP = 2

# Scattered source pixels per rule
SCATTERED_MIN = 3
SCATTERED_MAX_CAP = 8
SCATTERED_AREA_DIV = 30

# Bystander pixels
BYSTANDER_MAX = 5

# Placement retry budgets
LEGEND_RETRIES = 500
SCATTERED_RETRIES_PER_PIXEL = 80
BYSTANDER_RETRIES = 50
OUTER_RETRIES = 300

_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _reserve_4neighbors(reserved: set[tuple[int, int]], cells) -> None:
    for r, c in cells:
        for dr, dc in _DIRS4:
            reserved.add((int(r) + dr, int(c) + dc))


def _reserve_chebyshev(reserved: set[tuple[int, int]], centers, radius: int) -> None:
    for r, c in centers:
        for dr in range(-int(radius), int(radius) + 1):
            for dc in range(-int(radius), int(radius) + 1):
                reserved.add((int(r) + dr, int(c) + dc))


def _make_stencil(min_pixels: int, max_pixels: int) -> list[tuple[int, int]]:
    n = random.randint(max(3, min_pixels), max(3, max_pixels))
    dr, dc = random.choice(_DIRS4)
    coords = {(0, 0), (dr, dc), (2 * dr, 2 * dc)}
    while len(coords) < n:
        frontier = set()
        for r, c in coords:
            for rr, cc in _DIRS8:
                nr, nc = r + rr, c + cc
                if max(abs(nr), abs(nc)) > STENCIL_RADIUS_CAP:
                    continue
                if (nr, nc) not in coords:
                    frontier.add((nr, nc))
        if not frontier:
            break
        coords.add(random.choice(list(frontier)))
    return sorted(coords)


def _build_rules(num_rules: int, area: int):
    target_colors = TARGET_COLORS[:]
    source_colors = SOURCE_COLORS[:]
    random.shuffle(target_colors)
    random.shuffle(source_colors)

    max_s = min(STENCIL_MAX_CAP, max(3, area // STENCIL_MAX_AREA_DIV))
    min_s = min(max(3, area // STENCIL_MIN_AREA_DIV), max_s)
    sc_max = min(SCATTERED_MAX_CAP, max(SCATTERED_MIN, area // SCATTERED_AREA_DIV))

    rules = []
    used = set()
    for i in range(num_rules):
        src_c = source_colors[i]
        tgt_c = target_colors[i]
        used.update((src_c, tgt_c))
        rel = _make_stencil(min_s, max_s)
        rel_with_dist = [(rr, cc, min(abs(rr) + abs(cc), 2)) for rr, cc in rel]
        min_rr = min(rr for rr, _cc in rel)
        max_rr = max(rr for rr, _cc in rel)
        min_cc = min(cc for _rr, cc in rel)
        max_cc = max(cc for _rr, cc in rel)
        rules.append(
            {
                "src": src_c,
                "tgt": tgt_c,
                "rel": rel,
                "rel_with_dist": rel_with_dist,
                "n_sc": random.randint(SCATTERED_MIN, sc_max),
                "row_bounds": (-min_rr, max_rr),
                "col_bounds": (-min_cc, max_cc),
            }
        )
    return rules, used


def _footprint_cells(rel: list[tuple[int, int]], anchor_r: int, anchor_c: int) -> set[tuple[int, int]]:
    return {(int(anchor_r) + rr, int(anchor_c) + cc) for rr, cc in rel}


def _touches_anchor_region(cells: set[tuple[int, int]], anchor: tuple[int, int]) -> bool:
    return any(_chebyshev(cell, anchor) <= 1 for cell in cells)


def _place_legends(rules, inp: np.ndarray, occupied: set[tuple[int, int]], reserved: set[tuple[int, int]]) -> bool:
    H, W = inp.shape
    teal_occupied: set[tuple[int, int]] = set()
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

            teal_cells = _footprint_cells(rel, ar, ac)
            if any(not (0 <= nr < H and 0 <= nc < W) for nr, nc in teal_cells):
                continue
            if teal_cells & {(sr, sc), (tr, tc)}:
                continue
            reverse_anchor = (sr - dr, sc - dc)
            if _touches_anchor_region(teal_occupied, (ar, ac)):
                continue
            if _touches_anchor_region(teal_cells, reverse_anchor):
                continue
            if _touches_anchor_region(teal_occupied, reverse_anchor):
                continue

            all_cells = {(sr, sc), (tr, tc)} | teal_cells
            if all_cells & (occupied | reserved):
                continue

            inp[sr, sc] = rule["src"]
            inp[tr, tc] = rule["tgt"]
            for nr, nc in teal_cells:
                inp[nr, nc] = 8
            occupied.update(all_cells)
            teal_occupied.update(teal_cells)
            _reserve_4neighbors(reserved, all_cells)
            _reserve_chebyshev(reserved, [reverse_anchor], radius=1)
            placed = True
            break
        if not placed:
            return False
    return True


def _place_scattered(rules, inp: np.ndarray, occupied: set[tuple[int, int]], reserved: set[tuple[int, int]]) -> bool:
    H, W = inp.shape
    for rule in rules:
        positions = []
        row_margin_lo, row_margin_hi = rule["row_bounds"]
        col_margin_lo, col_margin_hi = rule["col_bounds"]
        row_min = int(row_margin_lo)
        row_max = H - 1 - int(row_margin_hi)
        col_min = int(col_margin_lo)
        col_max = W - 1 - int(col_margin_hi)
        if row_min > row_max or col_min > col_max:
            return False

        for _ in range(rule["n_sc"] * SCATTERED_RETRIES_PER_PIXEL):
            if len(positions) >= rule["n_sc"]:
                break
            r = random.randint(row_min, row_max)
            c = random.randint(col_min, col_max)
            footprint = _footprint_cells(rule["rel"], r, c)
            if footprint & occupied or footprint & reserved:
                continue

            positions.append((r, c))
            occupied.add((r, c))
            reserved.update(footprint)
            _reserve_4neighbors(reserved, footprint)

        if len(positions) < SCATTERED_MIN:
            return False
        rule["scattered"] = positions
        for r, c in positions:
            inp[r, c] = rule["src"]
    return True


def _place_bystanders(used_colors, inp: np.ndarray, occupied: set[tuple[int, int]], reserved: set[tuple[int, int]]):
    bystander_colors = [c for c in CLEAN_BYSTANDER_COLORS if c not in used_colors]
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
            _reserve_4neighbors(reserved, [(r, c)])
            break
    return placed


def _build_output(rules, bystander_pos, H: int, W: int) -> np.ndarray:
    output = np.zeros((H, W), dtype=int)
    for r, c, bc in bystander_pos:
        output[r, c] = bc
    for rule in rules:
        for sr, sc in rule["scattered"]:
            for rr, cc, dist in rule["rel_with_dist"]:
                nr, nc = sr + rr, sc + cc
                if 0 <= nr < H and 0 <= nc < W:
                    output[nr, nc] = _gradient_color(rule["tgt"], dist)
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
