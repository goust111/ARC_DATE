# Variant of abc82100
# Dimension: D1 — 源像素与规则的匹配条件 (match_cond)
# Change: s.color == d.source_color：源像素的颜色必须等于规则示例中源色像素的颜色... → 多属性联合匹配（颜色+邻域特征）

# Design analysis for this variant
# --------------------------------
# a) Concrete behavior and trace
#    - Each rule legend still contains a source pixel, an adjacent target pixel,
#      and a nearby teal stencil. The new part is that the legend source also
#      encodes a small context class: its 4-neighborhood contains exactly 1, 2,
#      or 3 non-background cells.
#    - This count is made visible with gray support pixels. In the legend, the
#      target pixel already contributes one neighbor, so counts 2 and 3 add one
#      or two gray pixels on the orthogonal sides of the source. For scattered
#      source tokens, the full count is provided by 1/2/3 gray support pixels.
#    - Trace example: if a red legend source touches a blue target and also one
#      gray helper, then its non-background 4-neighbor count is 2. A scattered
#      red pixel with two gray neighbors matches that rule and receives the blue
#      stencil; another red pixel with only one gray neighbor does not.
#    - Difference from the original task: the original solver would stamp every
#      red source with the same rule. Here, same-color sources can split across
#      multiple rules based on the visible local support count.
#
# b) Complexity calibration
#    - Counts are restricted to {1,2,3}. This keeps the context vocabulary small
#      enough for humans, but large enough to distinguish multiple rules sharing
#      one source color.
#    - The generator always includes at least two rules with the same source
#      color but different counts, so the new condition is necessary rather than
#      decorative.
#
# c) Old assumptions that break
#    - _discover_rules() originally stored only source_color and target_color.
#      That assumes one source color implies one rule. This is no longer valid.
#    - main() originally iterated over all pixels of a source color and stamped
#      them all. Now it must filter by the source pixel's local 4-neighbor count.
#    - main() originally preserved every non-source-color non-teal pixel. The
#      new gray support pixels are input-only structure and must be removed when
#      attached to a matched source or a legend source, otherwise the output
#      would retain the context markers.
#
# d) Placement legality
#    - Legend markers are only allowed on sides orthogonal to the source->target
#      direction. Putting a helper on the opposite side would create a false
#      colored pair whose continuation sits next to the teal stencil.
#    - Scattered source tokens reserve their whole footprint and surrounding
#      4-neighbors so support pixels never overlap, counts stay stable, and no
#      unrelated bystander ends up adjacent to a consumable source.
#
# e) Visual observability
#    - Reused source colors with different gray-neighbor counts create a clear
#      before/after contrast across examples. A human can see that two red tokens
#      with different support counts transform differently.
#
# f) Visual cleanliness
#    - Support pixels use a single dedicated gray color and are removed in the
#      output. This keeps the output close to the original task: only bystanders
#      and recolored stencil copies remain.
#
# g) Color inference
#    - Source colors come from a small pool, target colors from a disjoint small
#      pool, and gray is reserved for the neighborhood encoding. No hidden use of
#      color ids is involved in ordering or transformation.
#
# 9) New solver assumptions and generator guarantees
#    - Assumption: every matched scattered source has only its dedicated support
#      pixels as non-background 4-neighbors.
#      Guarantee: _place_scattered() reserves the surrounding area so no extra
#      bystander or another token touches the source.
#    - Assumption: each (source_color, neighbor_count) key identifies at most one
#      rule.
#      Guarantee: _build_rules() samples unique keys and validates with main().
#    - Assumption: gray legend supports are not misread as separate legends.
#      Guarantee: legend supports are restricted to orthogonal positions only.

from __future__ import annotations

import random
import sys

import numpy as np

SOLVER_UTILS = "/home/lixinyu/snap/new/solvers"
if SOLVER_UTILS not in sys.path:
    sys.path.append(SOLVER_UTILS)

from arc_utils import components_mask, get_bg


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _anchor_distance(pattern: np.ndarray, anchor: tuple[int, int]) -> int:
    return min(_chebyshev((int(r), int(c)), anchor) for r, c in pattern)


def _count_non_bg4(grid: np.ndarray, r: int, c: int, bg: int) -> int:
    h, w = grid.shape
    total = 0
    for dr, dc in _DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) != int(bg):
            total += 1
    return total


def _non_bg_neighbors4(grid: np.ndarray, r: int, c: int, bg: int) -> list[tuple[int, int]]:
    h, w = grid.shape
    out: list[tuple[int, int]] = []
    for dr, dc in _DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) != int(bg):
            out.append((int(nr), int(nc)))
    return out


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

            for dr, dc in _DIRS4:
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
                        "neighbor_count": _count_non_bg4(grid, int(r), int(c), int(bg)),
                        "source_pos": (int(r), int(c)),
                        "target_pos": (int(tr), int(tc)),
                        "anchor": (int(anchor[0]), int(anchor[1])),
                        "pattern": np.asarray(pattern, dtype=int),
                    }
                )

    return rules


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use local source-target-pattern examples plus source neighborhood counts.

    Input structure:
    - Black background with colored pixels, teal (8) stencil components, and gray
      support pixels that make a source pixel's 4-neighbor count visible.
    - A rule example is still a 4-neighbor colored pair source->target whose
      continued direction reaches a teal stencil, but the source's local count of
      non-background 4-neighbors is also part of the encoded rule.

    Rule:
    - For each legend, copy its teal stencil onto every other source pixel that
      matches both the legend source color and the same non-background 4-neighbor
      count, recoloring the copied stencil to the legend target color.
    - Translate the stencil by aligning the legend anchor with the matched source
      pixel, and remove the legend, teal stencil, and gray support pixels that
      belong to matched source tokens.

    Key insight: The legend no longer teaches only a color substitution; it also
    teaches a local context class, visible as how many adjacent non-background
    cells surround the source pixel.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    pattern_color = 8
    h, w = g.shape

    rules = _discover_rules(g, bg=bg, pattern_color=pattern_color)
    legend_pixels: set[tuple[int, int]] = set()
    consumed_support: set[tuple[int, int]] = set()
    source_colors: set[int] = set()
    for rule in rules:
        src_pos = rule["source_pos"]
        tgt_pos = rule["target_pos"]
        legend_pixels.add(src_pos)
        legend_pixels.add(tgt_pos)
        source_colors.add(rule["source_color"])
        for nr, nc in _non_bg_neighbors4(g, src_pos[0], src_pos[1], int(bg)):
            if (nr, nc) != tgt_pos:
                consumed_support.add((int(nr), int(nc)))

    out = np.full_like(g, int(bg))

    matched_sources: list[tuple[dict, tuple[int, int]]] = []
    for rule in rules:
        wanted_color = int(rule["source_color"])
        wanted_count = int(rule["neighbor_count"])
        for sr, sc in map(tuple, np.argwhere(g == wanted_color)):
            sr = int(sr)
            sc = int(sc)
            if (sr, sc) in legend_pixels:
                continue
            if _count_non_bg4(g, sr, sc, int(bg)) != wanted_count:
                continue
            matched_sources.append((rule, (sr, sc)))
            for nr, nc in _non_bg_neighbors4(g, sr, sc, int(bg)):
                consumed_support.add((int(nr), int(nc)))

    # Preserve ordinary colored pixels that are not part of a legend or source token.
    for r, c in map(tuple, np.argwhere((g != int(bg)) & (g != pattern_color))):
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        if (r, c) in legend_pixels or (r, c) in consumed_support or color in source_colors:
            continue
        out[r, c] = color

    for rule, (sr, sc) in matched_sources:
        ar, ac = rule["anchor"]
        pattern = rule["pattern"]
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
MIN_DIM_BY_RULES = {2: 10, 3: 14, 4: 18}

RULE_COUNT_WEIGHTS = [2, 2, 3, 3, 4]
SOURCE_COLORS = [1, 2, 3]
TARGET_COLORS = [4, 6, 7, 9]
MARKER_COLOR = 5
PATTERN_COLOR = 8
NEIGHBOR_COUNTS = [1, 2, 3]

STENCIL_MAX_CAP = 12
STENCIL_MIN_AREA_DIV = 100
STENCIL_MAX_AREA_DIV = 25

SCATTERED_MIN = 3
SCATTERED_MAX_CAP = 8
SCATTERED_AREA_DIV = 30

BYSTANDER_MAX = 4

LEGEND_RETRIES = 500
SCATTERED_RETRIES_PER_PIXEL = 70
BYSTANDER_RETRIES = 60
OUTER_RETRIES = 400

_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _reserve_4neighbors(reserved: set[tuple[int, int]], cells) -> None:
    for r, c in cells:
        reserved.add((int(r), int(c)))
        for dr, dc in _DIRS4:
            reserved.add((int(r + dr), int(c + dc)))


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
    max_s = min(STENCIL_MAX_CAP, max(2, area // STENCIL_MAX_AREA_DIV))
    min_s = min(max(1, area // STENCIL_MIN_AREA_DIV), max_s)
    sc_max = min(SCATTERED_MAX_CAP, max(SCATTERED_MIN, area // SCATTERED_AREA_DIV))

    shared_src = random.choice(SOURCE_COLORS)
    shared_counts = random.sample(NEIGHBOR_COUNTS, 2)
    keys = [(shared_src, shared_counts[0]), (shared_src, shared_counts[1])]
    remaining = [(src, cnt) for src in SOURCE_COLORS for cnt in NEIGHBOR_COUNTS if (src, cnt) not in keys]
    random.shuffle(remaining)
    keys.extend(remaining[: num_rules - 2])
    random.shuffle(keys)

    targets = TARGET_COLORS[:]
    random.shuffle(targets)

    rules: list[dict] = []
    used_colors = {MARKER_COLOR, PATTERN_COLOR}
    for i, (src_c, neighbor_count) in enumerate(keys):
        tgt_c = targets[i]
        used_colors.update((src_c, tgt_c))
        stencil, anchor = _make_stencil(min_s, max_s)
        rel = [(r - anchor[0], c - anchor[1]) for r, c in stencil]
        rules.append(
            {
                "src": int(src_c),
                "tgt": int(tgt_c),
                "count": int(neighbor_count),
                "rel": rel,
                "n_sc": random.randint(SCATTERED_MIN, sc_max),
            }
        )
    return rules, used_colors


def _orthogonal_dirs(dr: int, dc: int) -> list[tuple[int, int]]:
    if dr == 0:
        return [(-1, 0), (1, 0)]
    return [(0, -1), (0, 1)]


def _legend_support_offsets(dr: int, dc: int, neighbor_count: int) -> list[tuple[int, int]]:
    orth = _orthogonal_dirs(dr, dc)
    if neighbor_count == 1:
        return []
    if neighbor_count == 2:
        return [random.choice(orth)]
    return orth


def _scatter_support_offsets(neighbor_count: int) -> list[tuple[int, int]]:
    return random.sample(_DIRS4, neighbor_count)


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

            support_cells = set()
            for rr, cc in _legend_support_offsets(dr, dc, int(rule["count"])):
                nr, nc = sr + rr, sc + cc
                if not (0 <= nr < H and 0 <= nc < W):
                    ok = False
                    break
                support_cells.add((int(nr), int(nc)))
            if not ok:
                continue

            all_cells = {(int(sr), int(sc)), (int(tr), int(tc)), (int(ar), int(ac))} | teal_cells | support_cells
            if all_cells & occupied or all_cells & reserved:
                continue

            inp[sr, sc] = rule["src"]
            inp[tr, tc] = rule["tgt"]
            for nr, nc in support_cells:
                inp[nr, nc] = MARKER_COLOR
            for nr, nc in teal_cells:
                inp[nr, nc] = PATTERN_COLOR

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
        support_groups = []
        for _ in range(rule["n_sc"] * SCATTERED_RETRIES_PER_PIXEL):
            if len(positions) >= rule["n_sc"]:
                break
            r, c = random.randint(0, H - 1), random.randint(0, W - 1)
            support_cells = {(int(r + dr), int(c + dc)) for dr, dc in _scatter_support_offsets(int(rule["count"]))}
            cells = {(int(r), int(c))} | support_cells
            if any(not (0 <= nr < H and 0 <= nc < W) for nr, nc in cells):
                continue
            if cells & occupied or cells & reserved:
                continue
            positions.append((int(r), int(c)))
            support_groups.append(sorted(support_cells))
            occupied.update(cells)
            _reserve_4neighbors(reserved, cells)
        if len(positions) < SCATTERED_MIN:
            return False
        rule["scattered"] = positions
        rule["supports"] = support_groups
        for r, c in positions:
            inp[r, c] = rule["src"]
        for group in support_groups:
            for r, c in group:
                inp[r, c] = MARKER_COLOR
    return True


def _place_bystanders(used_colors, inp, occupied, reserved):
    bystander_colors = [c for c in TARGET_COLORS if c not in used_colors]
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
            placed.append((int(r), int(c), int(bc)))
            occupied.add((int(r), int(c)))
            _reserve_4neighbors(reserved, [(int(r), int(c))])
            break
    return placed


def _build_output(rules, bystander_pos, H, W):
    output = np.zeros((H, W), dtype=int)
    for r, c, bc in bystander_pos:
        output[r, c] = bc
    for rule in rules:
        for sr, sc in rule["scattered"]:
            for rr, cc in rule["rel"]:
                nr, nc = sr + rr, sc + cc
                if 0 <= nr < H and 0 <= nc < W:
                    output[nr, nc] = rule["tgt"]
    return output


def _variant_metrics(inp: np.ndarray) -> dict[str, int]:
    bg = get_bg(inp)
    rules = _discover_rules(inp, bg=bg, pattern_color=PATTERN_COLOR)
    key_count = len({(int(rule["source_color"]), int(rule["neighbor_count"])) for rule in rules})
    repeated_source = len(rules) - len({int(rule["source_color"]) for rule in rules})
    return {"rules": len(rules), "unique_keys": key_count, "repeated_source_rules": repeated_source}


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

    metrics = _variant_metrics(inp)
    if metrics["rules"] != num_rules or metrics["unique_keys"] != num_rules or metrics["repeated_source_rules"] < 1:
        return None

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
