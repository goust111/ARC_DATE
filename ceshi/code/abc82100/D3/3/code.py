# Variant of abc82100
# Dimension: D3 — 源对象类型 (src)
# Change: 散落的单个彩色像素（不属于任何规则示例的非背景像素）... → 带方向标记的源像素

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg


# Design analysis for this D3 variant
# -----------------------------------
# a) Concrete behavior
#    - The legend is kept from the base task: src pixel, adjacent tgt pixel, then
#      one more step in the same direction reaches the anchor of a teal stencil.
#    - Every free-standing source occurrence is now a 2-pixel object:
#      {source_color pixel} + {fixed marker-color pixel in one cardinal neighbor}.
#    - The marker's side is the placement cue. The learned stencil is translated
#      so that its anchor lands on the marker cell, not on the source cell.
#      Example: if a rule's anchor-relative stencil is {(0,0),(1,0)} and a
#      scattered source at (8,8) has its marker at (8,9), the output stamp is
#      placed at {(8,9),(9,9)}. In the base task it would have been {(8,8),(9,8)}.
#      That single-step shift is the whole behavioral change.
# b) Complexity calibration
#    - The marker color is fixed and unique so the input visibly contains a new
#      object type rather than "just another colored neighbor".
#    - Marker direction has only four values (up/down/left/right); each rule gets
#      repeated support from several scattered occurrences, and the generator
#      enforces at least two directions per rule so the new degree of freedom is
#      observable instead of degenerate.
# c) Hidden assumptions broken in the base solver
#    - Base rule discovery treated every non-bg/non-teal color as a possible
#      legend participant. That would incorrectly read marker pixels as legends,
#      so marker_color must be excluded when discovering rules.
#    - Base stamping aligned anchor -> source pixel. That old invariant no longer
#      holds; stamping must align anchor -> marker pixel.
#    - Base output preservation copied any non-legend non-source colored pixel.
#      Marker pixels are neither legends nor source colors, so they would leak
#      into the output unless they are explicitly excluded.
# d) Placement legality
#    - Generator places each scattered object as an atomic 2-cell source+marker
#      pair, checks bounds for both cells, and reserves 4-neighbors around both
#      cells to avoid accidental 4-adjacent color pairs.
#    - Legend examples stay separated from scattered objects, so no scattered
#      source can accidentally borrow a legend target or teal cluster.
# e) Visual observability
#    - The same rule appears multiple times with different marker directions, so
#      a solver must notice the side-marker and cannot explain the examples by
#      the original "stamp on the source pixel" rule.
#    - No color-id ordering or hidden numeric convention is used; only visible
#      adjacency and direction matter.
# f) Visual cleanliness
#    - Inputs remain sparse: source+marker pairs are separated by reserved
#      spacing, legend triplets are isolated, and bystanders avoid all rule
#      colors plus the marker color. This keeps the new 2-pixel objects legible.
# g) Color-change inferability
#    - Color substitution is still taught by local src->tgt legends exactly as in
#      the base task. The marker affects placement only, not recoloring, so the
#      added complexity is a single visible variable rather than a new color code.
# New solver assumptions checked against the generator
#    - Every non-legend source occurrence has exactly one 4-neighbor marker.
#    - Marker color never appears in legends, targets, or bystanders.
#    - Legend source colors are unique across rules, so source_color alone picks
#      the correct rule during stamping.


MARKER_COLOR = 2
PATTERN_COLOR = 8


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _anchor_distance(pattern: np.ndarray, anchor: tuple[int, int]) -> int:
    return min(_chebyshev((int(r), int(c)), anchor) for r, c in pattern)


def _discover_rules(
    grid: np.ndarray,
    bg: int,
    pattern_color: int = PATTERN_COLOR,
    marker_color: int = MARKER_COLOR,
) -> list[dict]:
    h, w = grid.shape
    patterns = components_mask(grid == pattern_color, connectivity=8)
    rules: list[dict] = []
    seen: set[tuple[tuple[int, int], tuple[int, int], int]] = set()

    for r in range(h):
        for c in range(w):
            source_color = int(grid[r, c])
            if source_color in (int(bg), int(pattern_color), int(marker_color)):
                continue

            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                tr = r + dr
                tc = c + dc
                if not (0 <= tr < h and 0 <= tc < w):
                    continue

                target_color = int(grid[tr, tc])
                if target_color in (int(bg), int(pattern_color), int(marker_color)):
                    continue
                if target_color == source_color:
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


def _find_marker(
    grid: np.ndarray,
    source_pos: tuple[int, int],
    marker_color: int,
) -> tuple[int, int] | None:
    sr, sc = source_pos
    hits = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        mr = int(sr + dr)
        mc = int(sc + dc)
        if 0 <= mr < grid.shape[0] and 0 <= mc < grid.shape[1] and int(grid[mr, mc]) == int(marker_color):
            hits.append((mr, mc))
    if len(hits) != 1:
        return None
    return hits[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn color-to-stencil rules from local examples, then place each stencil by a side marker.

    Input structure:
    - Black background with teal (8) stencil components and colored pixels.
    - A legend example is still a 4-neighbor source->target pair whose continued
      direction reaches the anchor of a teal stencil.
    - Every free-standing source occurrence is a source-colored pixel with one
      adjacent red marker pixel; the marker's side encodes the placement direction.

    Rule:
    - Each legend teaches which teal stencil belongs to a source color and which
      target color the stamped stencil should use.
    - For every non-legend source pixel, find its unique adjacent red marker and
      translate the learned stencil so the legend anchor lands on that marker.
    - Remove all legend pixels, teal stencil pixels, source pixels, and markers;
      preserve unrelated bystanders.

    Key insight: The legend still determines the color substitution and stencil
    shape, but the extra adjacent marker changes where each copied stencil is attached.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    h, w = g.shape

    rules = _discover_rules(g, bg=bg, pattern_color=PATTERN_COLOR, marker_color=MARKER_COLOR)
    legend_pixels: set[tuple[int, int]] = set()
    source_colors: set[int] = set()
    for rule in rules:
        legend_pixels.add(rule["source_pos"])
        legend_pixels.add(rule["target_pos"])
        source_colors.add(rule["source_color"])

    out = np.full_like(g, int(bg))

    # Preserve non-rule pixels, but discard all markers and all source objects.
    for r, c in map(tuple, np.argwhere((g != int(bg)) & (g != PATTERN_COLOR))):
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        if color == int(MARKER_COLOR):
            continue
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
            marker = _find_marker(g, (sr, sc), MARKER_COLOR)
            if marker is None:
                continue
            mr, mc = marker
            dr = int(mr - ar)
            dc = int(mc - ac)
            rows = pattern[:, 0] + dr
            cols = pattern[:, 1] + dc
            ok = (0 <= rows) & (rows < h) & (0 <= cols) & (cols < w)
            out[rows[ok], cols[ok]] = rule["target_color"]

    return out


# --------------- generator (D3 marker variant) ---------------

# Grid dimensions
GRID_DIM_MAX = 20
MIN_DIM_BY_RULES = {1: 6, 2: 11, 3: 15, 4: 19}

# Rule sampling
RULE_COUNT_WEIGHTS = [1, 1, 2, 2, 3, 3, 4]
NON_BG_NON_TEAL_COLORS = [1, 3, 4, 5, 6, 7, 9]

# Stencil size
STENCIL_MAX_CAP = 12
STENCIL_MIN_AREA_DIV = 100
STENCIL_MAX_AREA_DIV = 25

# Scattered source objects per rule
SCATTERED_MIN = 3
SCATTERED_MAX_CAP = 8
SCATTERED_AREA_DIV = 32

# Bystander pixels
BYSTANDER_MAX = 5

# Placement retry budgets
LEGEND_RETRIES = 500
SCATTERED_RETRIES_PER_PIXEL = 70
BYSTANDER_RETRIES = 60
OUTER_RETRIES = 400

# Cardinal directions (4-connected)
_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
# 8-connected neighbor offsets
_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _reserve_4neighbors(reserved: set[tuple[int, int]], cells) -> None:
    for r, c in cells:
        for dr, dc in _DIRS4:
            reserved.add((int(r + dr), int(c + dc)))


def _make_stencil(min_pixels: int, max_pixels: int):
    n = random.randint(min_pixels, max_pixels)
    coords = {(0, 0)}
    for _ in range(n - 1):
        frontier = set()
        for r, c in coords:
            for dr, dc in _DIRS8:
                p = (int(r + dr), int(c + dc))
                if p not in coords:
                    frontier.add(p)
        coords.add(random.choice(list(frontier)))
    coords_list = sorted(coords)
    return coords_list, random.choice(coords_list)


def _build_rules(num_rules: int, area: int):
    colors = NON_BG_NON_TEAL_COLORS[:]
    random.shuffle(colors)
    src_colors = colors[:num_rules]
    tgt_pool = [c for c in colors if c not in src_colors]

    max_s = min(STENCIL_MAX_CAP, max(2, area // STENCIL_MAX_AREA_DIV))
    min_s = min(max(1, area // STENCIL_MIN_AREA_DIV), max_s)
    sc_max = min(SCATTERED_MAX_CAP, max(SCATTERED_MIN, area // SCATTERED_AREA_DIV))

    rules = []
    used_colors = {MARKER_COLOR}
    for src_c in src_colors:
        tgt_c = random.choice(tgt_pool)
        used_colors.update((src_c, tgt_c))
        stencil, anchor = _make_stencil(min_s, max_s)
        rel = [(int(r - anchor[0]), int(c - anchor[1])) for r, c in stencil]
        rules.append(
            {
                "src": int(src_c),
                "tgt": int(tgt_c),
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
            sr = random.randint(0, H - 1)
            sc = random.randint(0, W - 1)
            tr, tc = int(sr + dr), int(sc + dc)
            ar, ac = int(sr + 2 * dr), int(sc + 2 * dc)
            if not (0 <= tr < H and 0 <= tc < W and 0 <= ar < H and 0 <= ac < W):
                continue

            teal_cells = set()
            ok = True
            for rr, cc in rel:
                nr, nc = int(ar + rr), int(ac + cc)
                if not (0 <= nr < H and 0 <= nc < W):
                    ok = False
                    break
                teal_cells.add((nr, nc))
            if not ok:
                continue

            all_cells = {(int(sr), int(sc)), (tr, tc), (ar, ac)} | teal_cells
            if all_cells & occupied:
                continue
            if all_cells & reserved:
                continue

            inp[int(sr), int(sc)] = rule["src"]
            inp[tr, tc] = rule["tgt"]
            for nr, nc in teal_cells:
                inp[nr, nc] = PATTERN_COLOR
            occupied.update(all_cells)
            _reserve_4neighbors(reserved, all_cells)
            placed = True
            break
        if not placed:
            return False
    return True


def _direction_plan(n_sc: int) -> list[tuple[int, int]]:
    dirs = _DIRS4[:]
    random.shuffle(dirs)
    chosen = [dirs[0], dirs[1]]
    while len(chosen) < n_sc:
        chosen.append(random.choice(dirs))
    random.shuffle(chosen)
    return chosen


def _place_scattered(rules, inp: np.ndarray, occupied: set[tuple[int, int]], reserved: set[tuple[int, int]]):
    H, W = inp.shape
    for rule in rules:
        objects = []
        for dr, dc in _direction_plan(rule["n_sc"]):
            placed = False
            for _ in range(SCATTERED_RETRIES_PER_PIXEL):
                sr = random.randint(0, H - 1)
                sc = random.randint(0, W - 1)
                mr, mc = int(sr + dr), int(sc + dc)
                if not (0 <= mr < H and 0 <= mc < W):
                    continue
                cells = {(int(sr), int(sc)), (mr, mc)}
                if cells & occupied:
                    continue
                if cells & reserved:
                    continue
                inp[int(sr), int(sc)] = rule["src"]
                inp[mr, mc] = MARKER_COLOR
                occupied.update(cells)
                _reserve_4neighbors(reserved, cells)
                objects.append({"source": (int(sr), int(sc)), "marker": (mr, mc)})
                placed = True
                break
            if not placed:
                return False
        if len(objects) < SCATTERED_MIN:
            return False
        rule["scattered"] = objects
    return True


def _place_bystanders(used_colors, inp: np.ndarray, occupied: set[tuple[int, int]], reserved: set[tuple[int, int]]):
    bystander_colors = [c for c in NON_BG_NON_TEAL_COLORS if c not in used_colors]
    if not bystander_colors:
        return []
    H, W = inp.shape
    n = random.randint(0, min(BYSTANDER_MAX, len(bystander_colors) * 2))
    placed = []
    for _ in range(n):
        for _ in range(BYSTANDER_RETRIES):
            r = random.randint(0, H - 1)
            c = random.randint(0, W - 1)
            if (r, c) in occupied or (r, c) in reserved:
                continue
            bc = random.choice(bystander_colors)
            inp[r, c] = int(bc)
            placed.append((int(r), int(c), int(bc)))
            occupied.add((int(r), int(c)))
            _reserve_4neighbors(reserved, [(int(r), int(c))])
            break
    return placed


def _build_output(rules, bystander_pos, H: int, W: int):
    output = np.zeros((H, W), dtype=int)
    for r, c, bc in bystander_pos:
        output[int(r), int(c)] = int(bc)
    for rule in rules:
        rel = rule["rel"]
        for obj in rule["scattered"]:
            mr, mc = obj["marker"]
            for rr, cc in rel:
                nr, nc = int(mr + rr), int(mc + cc)
                if 0 <= nr < H and 0 <= nc < W:
                    output[nr, nc] = rule["tgt"]
    return output


def _intended_rule_pairs(rules) -> set[tuple[int, int]]:
    return {(int(rule["src"]), int(rule["tgt"])) for rule in rules}


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
        discovered = _discover_rules(inp, bg=get_bg(inp), pattern_color=PATTERN_COLOR, marker_color=MARKER_COLOR)
        discovered_pairs = {(int(rule["source_color"]), int(rule["target_color"])) for rule in discovered}
        if len(discovered) != num_rules:
            return None
        if discovered_pairs != _intended_rule_pairs(rules):
            return None
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
