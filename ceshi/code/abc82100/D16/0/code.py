# Variant of abc82100
# Dimension: D16 — 规则示例中颜色对的空间关系 (meta)
# Change: 源色像素和目标色像素相邻（4连通或8连通），构成紧凑的颜色对... → 源色和目标色像素间隔固定距离

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg

# Design analysis for the D16 variant
# a) Concrete behavior:
#    Each legend still teaches one color-substitution rule and one teal stencil,
#    but the colored cue is no longer a tight adjacent pair. Instead, source and
#    target lie on the same cardinal ray with a fixed span of 3 or 4 cells, and
#    the anchor stays one more step beyond the target. Example with span=3:
#    S . . T A, where A touches the teal stencil. The output behavior is
#    unchanged: every scattered source pixel receives the translated stencil,
#    recolored to the legend target color.
#    Original flow: scan one-step neighbors to find source->target, then look
#    one more step for the anchor. Variant flow: scan 3-step or 4-step targets,
#    then look one more step beyond that for the anchor. The teal stencil copy
#    stage is unchanged.
# b) Complexity calibration:
#    The span is sampled from {3, 4} for the whole grid, so examples repeat one
#    consistent long-distance relation instead of introducing per-rule freedom.
#    This is enough to make the visual change obvious while keeping the task
#    inferable from 2-4 training pairs.
# c) Old implicit assumptions that break:
#    - `_discover_rules` assumed the target was a 4-neighbor. That must change
#      to a small set of longer offsets.
#    - Legend placement assumed `source,target,anchor` occupied consecutive
#      cells. The generator must now reserve a longer ray and handle the larger
#      footprint.
#    - Spurious-rule avoidance based only on local 4-neighbor isolation is no
#      longer enough, because accidental same-row/same-column colored pixels at
#      distance 3 or 4 could look like legends. The generator therefore reserves
#      cardinal rays out to the maximum tested span around every colored cell.
# d) Placement legality:
#    Legends use `source + span*dir = target` and `target + dir = anchor`, all
#    in bounds. Stencil cells must fit in bounds and avoid occupied or reserved
#    cells. Scattered pixels and bystanders avoid occupied cells, local 4-near
#    collisions, and long cardinal alignments that would create false legends.
# e) Visual observability:
#    Human solvers can now see that the teaching source and target colors are
#    separated by visible gaps rather than touching. Keeping the span constant
#    within a grid makes that relation easy to notice.
# f) Visual cleanliness:
#    Colored cells stay isolated and legends are spaced by reservations, so the
#    long-distance cue is readable instead of being buried in clutter.
# g) Color inference:
#    Colors still do not encode hidden ordering. The only cue is the visible
#    legend relation plus the nearby teal stencil, exactly as in the original
#    task but with a longer source-target spacing.
# h) Anti-degeneration metric:
#    The generator should only emit inputs whose discovered legend spans are all
#    3 or 4. If the original adjacent-pair solver is run on these inputs, it
#    should usually miss the rules or produce a different output.
# i) New solver assumptions and generator guarantees:
#    - `main()` assumes legends use one of the tested spans and that all true
#      rules in one grid agree on the dominant span. The generator enforces one
#      span per grid.
#    - `main()` assumes no non-legend colored cells accidentally form another
#      valid long-distance pair with an anchor near teal. The generator enforces
#      this with ray reservations around every colored placement.


PAIR_SPAN_OPTIONS = (3, 4)


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _anchor_distance(pattern: np.ndarray, anchor: tuple[int, int]) -> int:
    return min(_chebyshev((int(r), int(c)), anchor) for r, c in pattern)


def _discover_rules(grid: np.ndarray, bg: int, pattern_color: int = 8) -> list[dict]:
    h, w = grid.shape
    patterns = components_mask(grid == pattern_color, connectivity=8)
    candidates_by_span: dict[int, list[dict]] = {span: [] for span in PAIR_SPAN_OPTIONS}
    seen_by_span: dict[int, set[tuple[tuple[int, int], tuple[int, int], int]]] = {
        span: set() for span in PAIR_SPAN_OPTIONS
    }

    for r in range(h):
        for c in range(w):
            source_color = int(grid[r, c])
            if source_color in (int(bg), int(pattern_color)):
                continue

            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                for span in PAIR_SPAN_OPTIONS:
                    tr = r + span * dr
                    tc = c + span * dc
                    ar = tr + dr
                    ac = tc + dc
                    if not (0 <= tr < h and 0 <= tc < w and 0 <= ar < h and 0 <= ac < w):
                        continue

                    target_color = int(grid[tr, tc])
                    if target_color in (int(bg), int(pattern_color)) or target_color == source_color:
                        continue

                    anchor = (int(ar), int(ac))
                    pattern_hits: list[tuple[int, int, np.ndarray]] = []
                    for i, pattern in enumerate(patterns):
                        dist = _anchor_distance(pattern, anchor)
                        if dist <= 1:
                            pattern_hits.append((int(dist), int(i), pattern))

                    if not pattern_hits:
                        continue

                    pattern_hits.sort(key=lambda item: (item[0], len(item[2])))
                    _dist, pattern_index, pattern = pattern_hits[0]
                    key = ((int(r), int(c)), (int(tr), int(tc)), int(pattern_index))
                    if key in seen_by_span[span]:
                        continue
                    seen_by_span[span].add(key)
                    candidates_by_span[span].append(
                        {
                            "source_color": int(source_color),
                            "target_color": int(target_color),
                            "source_pos": (int(r), int(c)),
                            "target_pos": (int(tr), int(tc)),
                            "anchor": anchor,
                            "pattern": np.asarray(pattern, dtype=int),
                            "span": int(span),
                        }
                    )

    best_span = max(PAIR_SPAN_OPTIONS, key=lambda span: len(candidates_by_span[span]))
    return candidates_by_span[best_span]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use spaced source-target-pattern examples to replace matching source pixels.

    Input structure:
    - Black background with isolated colored pixels and teal (8) stencil
      components.
    - A rule example is a cardinal source pixel and a target pixel of a
      different color located 3 or 4 cells away on the same line; the next cell
      past the target is an anchor touching the teal stencil.

    Rule:
    - For each discovered long-distance example, copy that teal component onto
      every other pixel of the source color, recoloring the copied component to
      the example's target color.
    - The translation places the legend anchor onto the scattered source pixel.
    - Remove all teal stencil cells and the colored pixels that belong to legend
      examples.

    Key insight: The teaching pair is not adjacent anymore; the informative cue
    is the repeated source-to-target spacing together with the stencil anchored
    just beyond the distant target.
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

    # Keep colored pixels that are not legend members and are not consumed
    # as scattered sources by any discovered rule.
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


# --------------- generator (c) ---------------

# ── CFG: tunable generation parameters ──────────────────────────────
GRID_DIM_MAX = 20
MIN_DIM_BY_RULES = {1: 6, 2: 10, 3: 14, 4: 18}

RULE_COUNT_WEIGHTS = [1, 1, 2, 2, 3, 3, 4]
NON_BG_NON_TEAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

STENCIL_MAX_CAP = 12
STENCIL_MIN_AREA_DIV = 100
STENCIL_MAX_AREA_DIV = 25

SCATTERED_MIN = 3
SCATTERED_MAX_CAP = 8
SCATTERED_AREA_DIV = 30

BYSTANDER_MAX = 5

LEGEND_RETRIES = 700
SCATTERED_RETRIES_PER_PIXEL = 80
BYSTANDER_RETRIES = 80
OUTER_RETRIES = 400

_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _reserve_4neighbors(reserved: set[tuple[int, int]], cells) -> None:
    for r, c in cells:
        for dr, dc in _DIRS4:
            reserved.add((int(r) + dr, int(c) + dc))


def _reserve_color_rays(
    reserved: set[tuple[int, int]], cells, max_span: int = max(PAIR_SPAN_OPTIONS)
) -> None:
    for r, c in cells:
        r = int(r)
        c = int(c)
        for dr, dc in _DIRS4:
            for step in range(1, max_span + 1):
                reserved.add((r + step * dr, c + step * dc))


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
    pair_span = random.choice(PAIR_SPAN_OPTIONS)

    rules, used = [], set()
    for i in range(num_rules):
        src_c, tgt_c = colors[2 * i], colors[2 * i + 1]
        used.update((src_c, tgt_c))
        stencil, anchor = _make_stencil(min_s, max_s)
        rel = [(r - anchor[0], c - anchor[1]) for r, c in stencil]
        rules.append(
            {
                "src": src_c,
                "tgt": tgt_c,
                "rel": rel,
                "n_sc": random.randint(SCATTERED_MIN, sc_max),
                "span": pair_span,
            }
        )
    return rules, used, pair_span


def _place_legends(rules, pair_span: int, inp, occupied, reserved):
    H, W = inp.shape
    for rule in rules:
        rel = rule["rel"]
        placed = False
        for _ in range(LEGEND_RETRIES):
            dr, dc = random.choice(_DIRS4)
            sr, sc = random.randint(0, H - 1), random.randint(0, W - 1)
            tr, tc = sr + pair_span * dr, sc + pair_span * dc
            ar, ac = tr + dr, tc + dc
            if not (0 <= tr < H and 0 <= tc < W and 0 <= ar < H and 0 <= ac < W):
                continue

            teal_cells = set()
            ok = True
            for rr, cc in rel:
                nr, nc = ar + rr, ac + cc
                if not (0 <= nr < H and 0 <= nc < W):
                    ok = False
                    break
                teal_cells.add((nr, nc))
            if not ok:
                continue

            all_cells = {(sr, sc), (tr, tc), (ar, ac)} | teal_cells
            if all_cells & (occupied | reserved):
                continue

            inp[sr, sc] = rule["src"]
            inp[tr, tc] = rule["tgt"]
            for nr, nc in teal_cells:
                inp[nr, nc] = 8
            occupied.update(all_cells)
            _reserve_4neighbors(reserved, all_cells)
            _reserve_color_rays(reserved, [(sr, sc), (tr, tc)])
            placed = True
            break
        if not placed:
            return False
    return True


def _place_scattered(rules, inp, occupied, reserved):
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
            _reserve_4neighbors(reserved, [(r, c)])
            _reserve_color_rays(reserved, [(r, c)])
        if len(positions) < SCATTERED_MIN:
            return False
        rule["scattered"] = positions
        for r, c in positions:
            inp[r, c] = rule["src"]
    return True


def _place_bystanders(used_colors, inp, occupied, reserved):
    bystander_colors = [c for c in NON_BG_NON_TEAL_COLORS if c not in used_colors]
    if not bystander_colors:
        return []
    H, W = inp.shape
    n = random.randint(0, min(BYSTANDER_MAX, len(bystander_colors) * 2))
    placed = []
    for _ in range(n):
        for _ in range(BYSTANDER_RETRIES):
            r, c = random.randint(0, H - 1), random.randint(0, W - 1)
            if (r, c) not in occupied and (r, c) not in reserved:
                bc = random.choice(bystander_colors)
                inp[r, c] = bc
                placed.append((r, c, bc))
                occupied.add((r, c))
                _reserve_4neighbors(reserved, [(r, c)])
                _reserve_color_rays(reserved, [(r, c)])
                break
    return placed


def _build_output(rules, bystander_pos, H, W):
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

    rules, used_colors, pair_span = _build_rules(num_rules, H * W)
    occupied, reserved = set(), set()
    inp = np.zeros((H, W), dtype=int)

    if not _place_legends(rules, pair_span, inp, occupied, reserved):
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
    raise RuntimeError("Failed to generate valid input after %d attempts" % OUTER_RETRIES)
