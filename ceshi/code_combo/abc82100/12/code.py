# Combo variant of abc82100 (index 12)
# combo_spec: D4/3+D15/2
# Name: 模板按比例缩放+网格自动扩展

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg


# Design analysis for the D4/3 + D15/2 combo variant
# a) Concrete behavior and trace:
# - Original abc82100 learns rules from local legends of the form
#   `source -> target -> anchor+teal stencil`, then copies the teal stencil onto
#   every scattered source pixel at one fixed size, clipping at the border.
# - D4/3 changes the scattered source instance: it is no longer a lone pixel but
#   the NW head of a same-color SE diagonal run. The run length k in {1,2,3}
#   encodes the scale. The teal legend component becomes a prototype, and each
#   prototype cell expands into a k x k block.
# - D15/2 changes the border semantics: scaled stamps are not clipped. Instead,
#   the output canvas becomes the smallest axis-aligned rectangle that contains
#   both the entire original input frame and every scaled stamp. If a stamp
#   reaches row -2, for example, two background rows are inserted above the
#   original frame and every surviving in-frame pixel shifts downward by 2.
# - Small example:
#   legend prototype rel coords = {(-1,0), (0,0), (0,1)} around the anchor;
#   scattered marker = diagonal run at (1,1),(2,2),(3,3), so k=3.
#   Original task would stamp the unscaled 3-cell pattern around (1,1) and clip
#   the cell at row -1 away. This combo instead creates three 3x3 blocks whose
#   unshifted rows span [-2..3], then expands the grid upward by 2 rows so the
#   whole scaled shape remains visible.
# b) Complexity calibration:
# - The scale alphabet stays small: only 1/2/3. Every rule gets at least one
#   scale-1 and one larger instance, so the size cue repeats within a sample.
# - Prototypes remain compact (2-5 cells in a <=3x3 span), which keeps 3x stamps
#   readable while still letting them exceed the original frame.
# - At least one scattered placement per generated sample is forced to overflow
#   the input frame, so the expansion behavior is not optional or latent.
# c) Hidden assumptions reviewed against the original main():
# - `_discover_rules(...)` still assumes a 4-neighbor source/target pair whose
#   continuation reaches a nearby teal component. The generator keeps the anchor
#   directly on a teal cell, so the learned local rule format is unchanged.
# - The original solver treated every remaining source-colored pixel as an
#   independent source instance. That is false now because a whole diagonal run
#   shares the source color. The new solver must detect only the NW-most head and
#   infer the scale from the run length.
# - The original stamping loop clipped out-of-bounds cells against the input
#   shape. That assumption is intentionally broken: we now collect all target
#   cells in unshifted coordinates, compute the expanded frame, then render.
# - The original output size was fixed. Here the preserved bystanders and the new
#   stamps are shifted together by one global offset derived from the minimum row
#   and column reached by any stamp. The original frame [0..H-1]x[0..W-1] is
#   always included in the final bounds, so expansion only adds margins.
# d) Placement legality enforced by the generator:
# - Diagonal source runs stay fully inside the input grid and are surrounded by
#   an 8-neighbor exclusion zone, preventing accidental merging or fake longer
#   runs.
# - Output stamps never overlap each other, never cover bystanders, and never
#   cover unrelated input objects. Legends are placed only after the scattered
#   runs are fixed and are forbidden from landing on any eventual output pixel.
# - The forced-overflow placement is chosen from valid heads whose diagonal run
#   fits in-bounds but whose scaled stamp extends outside the original frame.
# e) Visual observability:
# - A human can infer scale from repeated short-vs-long diagonal runs of the same
#   source color, and can infer expansion because some outputs are strictly wider
#   or taller than their inputs while preserving the same learned prototype.
# - No invisible metadata is used: scale comes from visible geometry, and the
#   expanded canvas follows visible overflow of the scaled targets.
# f) Visual cleanliness:
# - Output stamps are disjoint, bystanders are kept away from all transformed
#   regions, and legends do not sit inside final target areas. This keeps the
#   expanded portions of the output easy to inspect.
# g) Color inference:
# - Color substitution remains explicitly taught by the local legend; colors are
#   disjoint across rules, so the new reasoning burden is only scale + expansion.
# main()/generator contract introduced by this combo:
# - Every transformable source instance must be the NW head of a SE diagonal run
#   of its own color.
# - At least one such run must create a scaled stamp that exits the original
#   input frame, otherwise the sample degenerates to the scaling-only variant.
# - The generator guarantees that no unrelated colored adjacency plus nearby teal
#   creates a spurious extra rule, which the local rule-discovery code depends on.


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


def _diagonal_scale_head(grid: np.ndarray, r: int, c: int, color: int) -> int:
    """Return the SE-diagonal run length if (r,c) is the NW-most head, else 0."""
    h, w = grid.shape
    if int(grid[r, c]) != int(color):
        return 0
    if r > 0 and c > 0 and int(grid[r - 1, c - 1]) == int(color):
        return 0

    k = 0
    rr, cc = int(r), int(c)
    while 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == int(color):
        k += 1
        rr += 1
        cc += 1
    return k


def _scaled_pattern_cells(
    rel_pattern: np.ndarray, head: tuple[int, int], scale: int
) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    hr, hc = int(head[0]), int(head[1])
    k = int(scale)
    for rr, cc in rel_pattern:
        base_r = hr + int(rr) * k
        base_c = hc + int(cc) * k
        for dr in range(k):
            for dc in range(k):
                cells.append((base_r + dr, base_c + dc))
    return cells


def _frame_union(
    bounds: tuple[int, int, int, int], cells: set[tuple[int, int]] | list[tuple[int, int]]
) -> tuple[int, int, int, int]:
    min_r, min_c, max_r, max_c = map(int, bounds)
    for r, c in cells:
        min_r = min(min_r, int(r))
        min_c = min(min_c, int(c))
        max_r = max(max_r, int(r))
        max_c = max(max_c, int(c))
    return min_r, min_c, max_r, max_c


def _render_expanded_output(
    height: int,
    width: int,
    bg: int,
    preserved: list[tuple[int, int, int]],
    painted: list[tuple[int, int, int]],
) -> np.ndarray:
    out = np.full((int(height), int(width)), int(bg), dtype=int)
    for r, c, color in preserved:
        out[int(r), int(c)] = int(color)
    for r, c, color in painted:
        out[int(r), int(c)] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn local legends, scale their teal prototypes, and expand the canvas.

    Input structure:
    - Black background with local source->target legend pairs and nearby teal (8)
      prototype components.
    - Away from the legends, each transformable source instance is a NW-to-SE
      diagonal run of one learned source color; the run head gives the placement
      point and the run length gives the scale.

    Rule:
    - Discover each legend as an adjacent colored pair whose forward anchor
      touches a teal prototype component.
    - For every non-legend diagonal run of the matching source color, place a
      scaled copy of that prototype at the run head, recolored to the rule's
      target color; scale k expands each prototype cell into a k x k block.
    - Remove legend pixels, teal prototype pixels, and all source-color marker
      runs; if any scaled copy leaves the input frame, enlarge the output grid so
      the full scaled shapes remain visible.

    Key insight: The local legend still teaches which prototype shape and target
    color to use, while the visible diagonal-run length determines how large the
    copied prototype becomes and therefore how far the final canvas must extend.
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
        source_colors.add(int(rule["source_color"]))

    preserved_raw: list[tuple[int, int, int]] = []
    for r, c in map(tuple, np.argwhere((g != int(bg)) & (g != pattern_color))):
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        if (r, c) in legend_pixels or color in source_colors:
            continue
        preserved_raw.append((r, c, color))

    painted_raw: list[tuple[int, int, int]] = []
    bounds = (0, 0, h - 1, w - 1)
    for rule in rules:
        ar, ac = rule["anchor"]
        pattern = np.asarray(rule["pattern"], dtype=int)
        rel_pattern = pattern - np.array([[int(ar), int(ac)]], dtype=int)
        for sr, sc in map(tuple, np.argwhere(g == rule["source_color"])):
            sr = int(sr)
            sc = int(sc)
            if (sr, sc) in legend_pixels:
                continue
            scale = _diagonal_scale_head(g, sr, sc, rule["source_color"])
            if scale <= 0:
                continue
            stamp_cells = _scaled_pattern_cells(rel_pattern, (sr, sc), scale)
            bounds = _frame_union(bounds, stamp_cells)
            painted_raw.extend((int(rr), int(cc), int(rule["target_color"])) for rr, cc in stamp_cells)

    min_r, min_c, max_r, max_c = bounds
    out_h = int(max_r - min_r + 1)
    out_w = int(max_c - min_c + 1)
    shift_r = -int(min_r)
    shift_c = -int(min_c)

    preserved = [(r + shift_r, c + shift_c, color) for r, c, color in preserved_raw]
    painted = [(r + shift_r, c + shift_c, color) for r, c, color in painted_raw]
    return _render_expanded_output(out_h, out_w, int(bg), preserved, painted)


# --------------- generator (combo variant) ---------------

GRID_DIM_MAX = 18
OUTPUT_DIM_MAX = 30
MIN_DIM_BY_RULES = {1: 8, 2: 10, 3: 12, 4: 14}

RULE_COUNT_WEIGHTS = [1, 1, 2, 2, 3, 3, 4]
NON_BG_NON_TEAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

PROTOTYPE_MIN_CELLS = 2
PROTOTYPE_MAX_CELLS = 5
PROTOTYPE_MAX_SPAN = 3

SCALE_CHOICES = [1, 2, 3]
SCATTERED_MIN = 3
SCATTERED_MAX_CAP = 5
SCATTERED_AREA_DIV = 55

BYSTANDER_MAX = 3

LEGEND_RETRIES = 800
BYSTANDER_RETRIES = 150
OUTER_RETRIES = 500

_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _reserve_8neighbors(reserved: set[tuple[int, int]], cells) -> None:
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                reserved.add((int(r) + dr, int(c) + dc))


def _make_prototype() -> tuple[list[tuple[int, int]], tuple[int, int]]:
    for _ in range(300):
        n = random.randint(PROTOTYPE_MIN_CELLS, PROTOTYPE_MAX_CELLS)
        coords = {(0, 0)}
        for _ in range(n - 1):
            frontier = set()
            for r, c in coords:
                for dr, dc in _DIRS8:
                    p = (r + dr, c + dc)
                    if p not in coords:
                        frontier.add(p)
            coords.add(random.choice(list(frontier)))

        arr = np.asarray(sorted(coords), dtype=int)
        arr[:, 0] -= int(arr[:, 0].min())
        arr[:, 1] -= int(arr[:, 1].min())
        span_r = int(arr[:, 0].max()) + 1
        span_c = int(arr[:, 1].max()) + 1
        if span_r > PROTOTYPE_MAX_SPAN or span_c > PROTOTYPE_MAX_SPAN:
            continue

        coords_list = [tuple(map(int, rc)) for rc in arr.tolist()]
        anchor = random.choice(coords_list)
        return coords_list, anchor
    raise RuntimeError("failed to build a compact prototype")


def _choose_scales(n_scattered: int) -> list[int]:
    scales = [1, random.choice([2, 3])]
    while len(scales) < n_scattered:
        scales.append(random.choice(SCALE_CHOICES))
    random.shuffle(scales)
    return scales


def _build_rules(num_rules: int, area: int):
    colors = NON_BG_NON_TEAL_COLORS[:]
    random.shuffle(colors)
    sc_max = min(SCATTERED_MAX_CAP, max(SCATTERED_MIN, area // SCATTERED_AREA_DIV))

    rules: list[dict] = []
    used_colors: set[int] = set()
    for i in range(num_rules):
        src_c, tgt_c = colors[2 * i], colors[2 * i + 1]
        used_colors.update((src_c, tgt_c))
        prototype, anchor = _make_prototype()
        rel = sorted((int(r - anchor[0]), int(c - anchor[1])) for r, c in prototype)
        n_sc = random.randint(SCATTERED_MIN, sc_max)
        rules.append(
            {
                "src": int(src_c),
                "tgt": int(tgt_c),
                "rel": rel,
                "scales": _choose_scales(n_sc),
            }
        )
    return rules, used_colors


def _scaled_stamp_cells(rel, head: tuple[int, int], scale: int) -> set[tuple[int, int]]:
    return set(_scaled_pattern_cells(np.asarray(rel, dtype=int), head, scale))


def _diagonal_chain(head: tuple[int, int], scale: int) -> list[tuple[int, int]]:
    sr, sc = map(int, head)
    return [(sr + i, sc + i) for i in range(int(scale))]


def _overflows_frame(cells: set[tuple[int, int]], height: int, width: int) -> bool:
    for r, c in cells:
        if int(r) < 0 or int(r) >= int(height) or int(c) < 0 or int(c) >= int(width):
            return True
    return False


def _bounds_size(bounds: tuple[int, int, int, int]) -> tuple[int, int]:
    min_r, min_c, max_r, max_c = map(int, bounds)
    return max_r - min_r + 1, max_c - min_c + 1


def _candidate_heads(height: int, width: int, scale: int) -> list[tuple[int, int]]:
    heads = [(r, c) for r in range(height - scale + 1) for c in range(width - scale + 1)]
    random.shuffle(heads)
    return heads


def _place_one_scattered(
    rule: dict,
    scale: int,
    inp: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
    output_occupied: set[tuple[int, int]],
    bounds: tuple[int, int, int, int],
    require_overflow: bool,
) -> tuple[tuple[int, int, int], tuple[int, int, int, int]] | None:
    height, width = inp.shape
    for sr, sc in _candidate_heads(height, width, int(scale)):
        chain = _diagonal_chain((sr, sc), int(scale))
        chain_set = set(chain)
        if chain_set & occupied:
            continue
        if chain_set & reserved:
            continue
        if chain_set & output_occupied:
            continue

        stamp_cells = _scaled_stamp_cells(rule["rel"], (sr, sc), int(scale))
        if stamp_cells & occupied:
            continue
        if stamp_cells & output_occupied:
            continue

        overflow = _overflows_frame(stamp_cells, height, width)
        if require_overflow and not overflow:
            continue

        new_bounds = _frame_union(bounds, stamp_cells)
        out_h, out_w = _bounds_size(new_bounds)
        if out_h > OUTPUT_DIM_MAX or out_w > OUTPUT_DIM_MAX:
            continue

        for r, c in chain:
            inp[r, c] = int(rule["src"])
        occupied.update(chain_set)
        _reserve_8neighbors(reserved, chain)
        output_occupied.update(stamp_cells)
        return (int(sr), int(sc), int(scale)), new_bounds
    return None


def _place_scattered(rules, inp, occupied, reserved, output_occupied):
    bounds = (0, 0, inp.shape[0] - 1, inp.shape[1] - 1)
    overflow_done = False
    for idx, rule in enumerate(rules):
        placements: list[tuple[int, int, int]] = []
        scales = sorted(rule["scales"], reverse=True)
        overflow_index = next((i for i, scale in enumerate(scales) if scale > 1), None)
        for i, scale in enumerate(scales):
            require_overflow = bool((not overflow_done) and idx == 0 and overflow_index == i)
            placed = _place_one_scattered(
                rule,
                int(scale),
                inp,
                occupied,
                reserved,
                output_occupied,
                bounds,
                require_overflow=require_overflow,
            )
            if placed is None:
                return None
            placement, bounds = placed
            placements.append(placement)
            if require_overflow:
                overflow_done = True
        rule["scattered"] = placements

    if not overflow_done:
        return None
    return bounds


def _place_bystanders(used_colors, inp, occupied, reserved, output_occupied):
    bystander_colors = [c for c in NON_BG_NON_TEAL_COLORS if c not in used_colors]
    if not bystander_colors:
        return []

    height, width = inp.shape
    n = random.randint(0, min(BYSTANDER_MAX, len(bystander_colors) * 2))
    placed = []
    for _ in range(n):
        success = False
        for _ in range(BYSTANDER_RETRIES):
            r = random.randint(0, height - 1)
            c = random.randint(0, width - 1)
            if (r, c) in occupied or (r, c) in reserved or (r, c) in output_occupied:
                continue
            color = random.choice(bystander_colors)
            inp[r, c] = int(color)
            placed.append((int(r), int(c), int(color)))
            occupied.add((r, c))
            output_occupied.add((r, c))
            _reserve_8neighbors(reserved, [(r, c)])
            success = True
            break
        if not success:
            break
    return placed


def _place_legends(rules, inp, occupied, reserved, output_occupied):
    height, width = inp.shape
    for rule in rules:
        rel = rule["rel"]
        placed = False
        for _ in range(LEGEND_RETRIES):
            dr, dc = random.choice(_DIRS4)
            sr = random.randint(0, height - 1)
            sc = random.randint(0, width - 1)
            tr, tc = sr + dr, sc + dc
            ar, ac = sr + 2 * dr, sc + 2 * dc
            if not (0 <= tr < height and 0 <= tc < width and 0 <= ar < height and 0 <= ac < width):
                continue

            teal_cells = {(ar + rr, ac + cc) for rr, cc in rel}
            if any(not (0 <= r < height and 0 <= c < width) for r, c in teal_cells):
                continue

            all_cells = {(sr, sc), (tr, tc)} | teal_cells
            if all_cells & occupied:
                continue
            if all_cells & reserved:
                continue
            if all_cells & output_occupied:
                continue

            inp[sr, sc] = int(rule["src"])
            inp[tr, tc] = int(rule["tgt"])
            for r, c in teal_cells:
                inp[r, c] = 8
            occupied.update(all_cells)
            _reserve_8neighbors(reserved, all_cells)
            placed = True
            break
        if not placed:
            return False
    return True


def _build_output(rules, bystander_pos, height, width):
    bounds = (0, 0, height - 1, width - 1)
    painted_raw: list[tuple[int, int, int]] = []
    for rule in rules:
        for sr, sc, scale in rule["scattered"]:
            cells = _scaled_stamp_cells(rule["rel"], (sr, sc), int(scale))
            bounds = _frame_union(bounds, cells)
            painted_raw.extend((int(r), int(c), int(rule["tgt"])) for r, c in cells)

    min_r, min_c, max_r, max_c = bounds
    shift_r = -int(min_r)
    shift_c = -int(min_c)
    out_h = int(max_r - min_r + 1)
    out_w = int(max_c - min_c + 1)

    preserved = [(r + shift_r, c + shift_c, color) for r, c, color in bystander_pos]
    painted = [(r + shift_r, c + shift_c, color) for r, c, color in painted_raw]
    return _render_expanded_output(out_h, out_w, 0, preserved, painted)


def _behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    rules = _discover_rules(g, bg=bg, pattern_color=8)
    legend_pixels = {tuple(rule["source_pos"]) for rule in rules} | {tuple(rule["target_pos"]) for rule in rules}

    max_scale = 0
    overflow_instances = 0
    bounds = (0, 0, g.shape[0] - 1, g.shape[1] - 1)
    total_instances = 0
    for rule in rules:
        ar, ac = rule["anchor"]
        rel = np.asarray(rule["pattern"], dtype=int) - np.array([[int(ar), int(ac)]], dtype=int)
        for sr, sc in map(tuple, np.argwhere(g == rule["source_color"])):
            sr = int(sr)
            sc = int(sc)
            if (sr, sc) in legend_pixels:
                continue
            scale = _diagonal_scale_head(g, sr, sc, rule["source_color"])
            if scale <= 0:
                continue
            total_instances += 1
            max_scale = max(max_scale, int(scale))
            cells = set(_scaled_pattern_cells(rel, (sr, sc), scale))
            if _overflows_frame(cells, g.shape[0], g.shape[1]):
                overflow_instances += 1
            bounds = _frame_union(bounds, cells)
    out_h, out_w = _bounds_size(bounds)
    return {
        "rule_count": len(rules),
        "instance_count": total_instances,
        "max_scale": int(max_scale),
        "overflow_instances": int(overflow_instances),
        "input_h": int(g.shape[0]),
        "input_w": int(g.shape[1]),
        "output_h": int(out_h),
        "output_w": int(out_w),
        "expanded_h": int(out_h - g.shape[0]),
        "expanded_w": int(out_w - g.shape[1]),
    }


def _gen_attempt():
    num_rules = random.choice(RULE_COUNT_WEIGHTS)
    min_dim = MIN_DIM_BY_RULES[num_rules]
    height = random.randint(min_dim, GRID_DIM_MAX)
    width = random.randint(min_dim, GRID_DIM_MAX)

    rules, used_colors = _build_rules(num_rules, height * width)
    occupied: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()
    output_occupied: set[tuple[int, int]] = set()
    inp = np.zeros((height, width), dtype=int)

    bounds = _place_scattered(rules, inp, occupied, reserved, output_occupied)
    if bounds is None:
        return None

    bystander_pos = _place_bystanders(used_colors, inp, occupied, reserved, output_occupied)
    if not _place_legends(rules, inp, occupied, reserved, output_occupied):
        return None

    output = _build_output(rules, bystander_pos, height, width)
    metrics = _behavior_metrics(inp)
    try:
        if (
            np.array_equal(main(inp), output)
            and metrics["max_scale"] >= 2
            and metrics["overflow_instances"] >= 1
            and (metrics["output_h"] > metrics["input_h"] or metrics["output_w"] > metrics["input_w"])
        ):
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
