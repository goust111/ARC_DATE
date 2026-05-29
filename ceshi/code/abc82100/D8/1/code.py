# Variant of abc82100
# Dimension: D8 — 输出网格形状 (grid_shape)
# Change: 输出网格尺寸与输入相同（same）... → 输出尺寸扩大以容纳所有模板

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg


# Design analysis for the D8 variant
# a) Concrete behavior and trace:
#    The legend format is unchanged: a source-color pixel touches a target-color
#    pixel, and one more step in the same direction reaches an anchor touching a
#    teal stencil. The old solver translated that stencil onto every scattered
#    source pixel but clipped every stamped cell back into the input frame. Here
#    stamping happens on an implicit larger canvas. Example: on a 5x5 grid, if a
#    rule stencil has relative cells [(-1, 0), (0, 0), (1, 1)] and a scattered
#    source sits at (0, 4), the old logic kept only in-bounds cells (0,4) and
#    (1,5) was lost; the new logic keeps all three cells by expanding the output
#    to cover rows -1..4 and cols 0..5, then shifting everything by (+1, +0).
# b) Complexity calibration:
#    Only the canvas bounds change. Rule discovery, color mapping, stencil
#    translation, and bystander preservation stay identical. To make the new
#    behavior visible, the generator forbids 1-pixel stencils and forces at least
#    one scattered source to sit in an edge band where its stencil crosses a
#    border. This keeps the task readable while ensuring the D8 effect appears.
# c) Hidden assumptions in the original solver:
#    - `out = np.full_like(g, bg)` assumed the final canvas was the input canvas.
#    - `ok = ... in bounds` silently clipped every translated stencil cell.
#    - Preserved bystanders were written at unchanged coordinates with no need for
#      a global shift.
#    Those assumptions break once stamps may extend above/left/below/right of the
#    input. The new solver first collects all translated cells, computes the union
#    of the original input rectangle and all stamp coordinates, then renders once
#    with a single offset.
# d) Placement legality:
#    Legends are still placed fully inside the input so rule discovery remains
#    unchanged. Scattered sources remain isolated from legends and one another via
#    the same occupancy/reservation scheme. Only their implied output stamps may
#    leave the canvas, which is legal in this variant because the output expands.
# e) Visual observability:
#    A human still sees the same "learn local source->target + teal template"
#    mechanism, but now some outputs visibly gain rows/cols on the sides where
#    edge-near sources would have been clipped before. This is a stable, visible
#    change across a few examples.
# f) Visual cleanliness:
#    The variant does not add new marker types or overlapping legends. Expansion
#    merely adds black margin on the needed sides, so the scene remains legible.
# g) Color inference:
#    Color mapping is unchanged and still fully explained by the local legends, so
#    D8 does not introduce any hidden color rule.
# Solver-side assumptions introduced by the variant:
#    - The output canvas must include the full original input rectangle even if no
#      colored result lands there, otherwise top/left expansion would look like an
#      unrelated crop rather than a border growth.
#    - Rule patterns remain non-empty and all preserved non-rule pixels originate
#      inside the input rectangle; the generator guarantees this.


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


def _render_expanded(
    h: int,
    w: int,
    bg: int,
    preserved_cells: list[tuple[int, int, int]],
    stamped_cells: list[tuple[int, int, int]],
) -> np.ndarray:
    min_r = 0
    min_c = 0
    max_r = h - 1
    max_c = w - 1

    for r, c, _color in stamped_cells:
        min_r = min(min_r, int(r))
        min_c = min(min_c, int(c))
        max_r = max(max_r, int(r))
        max_c = max(max_c, int(c))

    out = np.full((int(max_r - min_r + 1), int(max_c - min_c + 1)), int(bg), dtype=int)
    row_shift = -int(min_r)
    col_shift = -int(min_c)

    for r, c, color in preserved_cells:
        out[int(r) + row_shift, int(c) + col_shift] = int(color)
    for r, c, color in stamped_cells:
        out[int(r) + row_shift, int(c) + col_shift] = int(color)
    return out


def _rel_bounds(rel: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [int(r) for r, _ in rel]
    cols = [int(c) for _, c in rel]
    return min(rows), max(rows), min(cols), max(cols)


def _stamp_overflows(rule: dict, sr: int, sc: int, h: int, w: int) -> bool:
    min_rr, max_rr, min_cc, max_cc = _rel_bounds(rule["rel"])
    return (
        int(sr) + min_rr < 0
        or int(sr) + max_rr >= h
        or int(sc) + min_cc < 0
        or int(sc) + max_cc >= w
    )


def _overflow_candidates(rule: dict, h: int, w: int) -> list[tuple[int, int]]:
    min_rr, max_rr, min_cc, max_cc = _rel_bounds(rule["rel"])
    candidates: set[tuple[int, int]] = set()

    if min_rr < 0:
        for r in range(min(h, -min_rr)):
            for c in range(w):
                candidates.add((int(r), int(c)))
    if max_rr > 0:
        for r in range(max(0, h - max_rr), h):
            for c in range(w):
                candidates.add((int(r), int(c)))
    if min_cc < 0:
        for r in range(h):
            for c in range(min(w, -min_cc)):
                candidates.add((int(r), int(c)))
    if max_cc > 0:
        for r in range(h):
            for c in range(max(0, w - max_cc), w):
                candidates.add((int(r), int(c)))

    return list(candidates)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn local source-target-pattern rules, then stamp them on an expanded canvas.

    Input structure:
    - Black background with isolated colored pixels and teal (8) stencil components.
    - A rule example is a 4-neighbor source->target pair whose continued direction
      reaches an anchor touching a teal stencil component.

    Rule:
    - For each rule example, copy that teal component onto every other pixel of
      the same source color, recoloring the copy to the rule's target color.
    - Translation still maps the rule anchor onto the source pixel being replaced,
      and all legend pixels plus teal stencil pixels are removed.
    - If a translated stencil extends beyond the input border, enlarge the output
      grid on the necessary sides instead of clipping those cells away.

    Key insight: The legend still teaches both the color substitution and the
    stencil shape; the only new step is to render all translated stencils onto the
    union of the original grid rectangle and every stamped stencil coordinate.
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

    preserved_cells: list[tuple[int, int, int]] = []
    for r, c in map(tuple, np.argwhere((g != int(bg)) & (g != pattern_color))):
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        if (r, c) in legend_pixels or color in source_colors:
            continue
        preserved_cells.append((r, c, color))

    stamped_cells: list[tuple[int, int, int]] = []
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
            for pr, pc in pattern:
                stamped_cells.append((int(pr) + dr, int(pc) + dc, int(rule["target_color"])))

    return _render_expanded(h, w, int(bg), preserved_cells, stamped_cells)


# --------------- generator (c) ---------------

# ── CFG: tunable generation parameters ──────────────────────────────
GRID_DIM_MAX = 20
MIN_DIM_BY_RULES = {1: 5, 2: 10, 3: 14, 4: 18}

RULE_COUNT_WEIGHTS = [1, 1, 2, 2, 3, 3, 4]
NON_BG_NON_TEAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

STENCIL_MAX_CAP = 12
STENCIL_MIN_AREA_DIV = 100
STENCIL_MAX_AREA_DIV = 25
STENCIL_MIN_PIXELS = 2

SCATTERED_MIN = 3
SCATTERED_MAX_CAP = 8
SCATTERED_AREA_DIV = 30

BYSTANDER_MAX = 5

LEGEND_RETRIES = 500
SCATTERED_RETRIES_PER_PIXEL = 50
BYSTANDER_RETRIES = 50
OUTER_RETRIES = 300

_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _reserve_4neighbors(reserved: set, cells) -> None:
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
                p = (int(r) + dr, int(c) + dc)
                if p not in coords:
                    frontier.add(p)
        coords.add(random.choice(list(frontier)))
    coords_list = sorted(coords)
    return coords_list, random.choice(coords_list)


def _build_rules(num_rules: int, area: int):
    colors = NON_BG_NON_TEAL_COLORS[:]
    random.shuffle(colors)

    max_s = min(STENCIL_MAX_CAP, max(STENCIL_MIN_PIXELS, area // STENCIL_MAX_AREA_DIV))
    min_s = min(max(STENCIL_MIN_PIXELS, area // STENCIL_MIN_AREA_DIV), max_s)
    sc_max = min(SCATTERED_MAX_CAP, max(SCATTERED_MIN, area // SCATTERED_AREA_DIV))

    rules, used = [], set()
    for i in range(num_rules):
        src_c, tgt_c = colors[2 * i], colors[2 * i + 1]
        used.update((src_c, tgt_c))
        stencil, anchor = _make_stencil(min_s, max_s)
        rel = [(int(r - anchor[0]), int(c - anchor[1])) for r, c in stencil]
        rules.append(
            {
                "src": int(src_c),
                "tgt": int(tgt_c),
                "rel": rel,
                "n_sc": random.randint(SCATTERED_MIN, sc_max),
                "force_overflow": False,
            }
        )
    return rules, used


def _place_legends(rules, inp, occupied, reserved):
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
            _reserve_4neighbors(reserved, all_cells)
            placed = True
            break
        if not placed:
            return False
    return True


def _place_scattered(rules, inp, occupied, reserved):
    H, W = inp.shape
    for rule in rules:
        positions = []

        if rule.get("force_overflow"):
            candidates = _overflow_candidates(rule, H, W)
            random.shuffle(candidates)
            forced = None
            for r, c in candidates:
                if (r, c) in occupied or (r, c) in reserved:
                    continue
                if not _stamp_overflows(rule, r, c, H, W):
                    continue
                forced = (int(r), int(c))
                break
            if forced is None:
                return False
            positions.append(forced)
            occupied.add(forced)
            _reserve_4neighbors(reserved, [forced])

        for _ in range(rule["n_sc"] * SCATTERED_RETRIES_PER_PIXEL):
            if len(positions) >= rule["n_sc"]:
                break
            r, c = random.randint(0, H - 1), random.randint(0, W - 1)
            if (r, c) in occupied or (r, c) in reserved:
                continue
            positions.append((int(r), int(c)))
            occupied.add((int(r), int(c)))
            _reserve_4neighbors(reserved, [(int(r), int(c))])

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
                placed.append((int(r), int(c), int(bc)))
                occupied.add((int(r), int(c)))
                _reserve_4neighbors(reserved, [(int(r), int(c))])
                break
    return placed


def _build_output(rules, bystander_pos, H, W):
    preserved_cells = [(int(r), int(c), int(color)) for r, c, color in bystander_pos]
    stamped_cells: list[tuple[int, int, int]] = []
    for rule in rules:
        rel = rule["rel"]
        for sr, sc in rule["scattered"]:
            for rr, cc in rel:
                stamped_cells.append((int(sr) + int(rr), int(sc) + int(cc), int(rule["tgt"])))
    return _render_expanded(H, W, 0, preserved_cells, stamped_cells)


def _gen_attempt():
    num_rules = random.choice(RULE_COUNT_WEIGHTS)
    min_dim = MIN_DIM_BY_RULES[num_rules]
    H = random.randint(min_dim, GRID_DIM_MAX)
    W = random.randint(min_dim, GRID_DIM_MAX)

    rules, used_colors = _build_rules(num_rules, H * W)
    expandable = [rule for rule in rules if _overflow_candidates(rule, H, W)]
    if not expandable:
        return None
    for rule in random.sample(expandable, random.randint(1, min(2, len(expandable)))):
        rule["force_overflow"] = True

    occupied, reserved = set(), set()
    inp = np.zeros((H, W), dtype=int)

    if not _place_legends(rules, inp, occupied, reserved):
        return None
    if not _place_scattered(rules, inp, occupied, reserved):
        return None
    bystander_pos = _place_bystanders(used_colors, inp, occupied, reserved)

    output = _build_output(rules, bystander_pos, H, W)
    if output.shape == inp.shape:
        return None
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
