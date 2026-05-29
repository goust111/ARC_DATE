# Combo variant of abc82100 (index 46)
# combo_spec: D5/1+D14/2
# Name: 极端位置锚点+角落对齐

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg


# Design analysis for the combo variant
# a) Concrete behavior:
#    Seed task: a legend contains source -> target -> anchor, and the nearby teal component is
#    copied onto every scattered source pixel by moving the legend anchor onto that source. Here
#    the legend anchor is no longer an arbitrary touch point: it is the stencil's upper-left teal
#    pixel. Placement also changes to "upper-left corner onto source". Because generation forces
#    that upper-left corner to be an occupied teal cell, the two changes fuse into one rule:
#    the copied stencil starts exactly at the scattered source pixel and then extends only
#    rightward/downward. Example: legend stencil cells relative to its corner are
#    {(0,0),(0,1),(1,0),(1,2)}. Old seed behavior with a non-corner anchor such as (1,0) would
#    stamp offsets {(-1,0),(-1,1),(0,0),(0,2)} around a source. The combo instead always stamps
#    {(0,0),(0,1),(1,0),(1,2)} because the designated anchor is the corner pixel itself.
# b) Complexity calibration:
#    The combo could become visually trivial if the stencil had one pixel or if the corner were
#    empty. To avoid that, generation always uses stencils of at least two pixels, anchored at an
#    occupied (0,0) cell, with the rest of the cells grown in the nonnegative quadrant. This keeps
#    the "grow from the source toward lower-right" behavior visible in every rule.
# c) Hidden assumptions review:
#    The seed solver assumed that `target + direction` may land next to a stencil rather than on a
#    specific stencil pixel, and that this anchor controls placement. Both assumptions change. The
#    new solver therefore maps a legend anchor position to exactly one teal component whose
#    lexicographically smallest pixel is that anchor. Stamping no longer depends on an arbitrary
#    learned offset; it aligns the stencil's designated upper-left teal pixel to the scattered
#    source. This also means the generator must guarantee an occupied top-left corner so "corner"
#    and "upper-left pixel" are identical rather than ambiguous.
# d) Placement legality:
#    Legend placement uses source/target/anchor as a 3-cell 4-connected line, then attaches the
#    teal stencil by adding nonnegative offsets to the anchor. We explicitly reject placements
#    where teal would cover the source or target legend cells, and we reserve 4-neighborhoods
#    around all placed objects so scattered pixels do not accidentally form extra legend pairs.
# e) Visual observability:
#    In every generated sample, each rule's teal template visibly starts at the continued third
#    cell of the legend and spreads only right/down from there; scattered outputs do the same from
#    their source pixels. A human can therefore infer both the fixed anchor location and the corner
#    alignment from 2-4 examples without reading any invisible metadata.
# f) Visual cleanliness:
#    Objects remain sparse on a black field; legends are separated from one another and from
#    scattered pixels via reserved 4-neighborhoods. Output stamp overlaps are still allowed, but
#    legend regions stay clean and decipherable because the teal template never overwrites its own
#    colored teaching pair.
# g) Change inferability:
#    Colors still act only as local source->target labels taught by adjacent legend pairs. No
#    ordering on color ids is used. The new inference burden is purely spatial: identify that the
#    teal component's own upper-left pixel is the designated anchor and that copies start there.


def _pattern_anchor(pattern: np.ndarray) -> tuple[int, int]:
    """Return the designated anchor: the lexicographically smallest teal pixel."""
    ordered = sorted((int(r), int(c)) for r, c in pattern)
    return ordered[0]


def _discover_rules(grid: np.ndarray, bg: int, pattern_color: int = 8) -> list[dict]:
    h, w = grid.shape
    patterns = [np.asarray(p, dtype=int) for p in components_mask(grid == pattern_color, connectivity=8)]
    anchor_to_pattern = {_pattern_anchor(pattern): pattern for pattern in patterns}
    rules: list[dict] = []
    seen: set[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = set()

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

                anchor = (int(tr + dr), int(tc + dc))
                pattern = anchor_to_pattern.get(anchor)
                if pattern is None:
                    continue

                key = ((int(r), int(c)), (int(tr), int(tc)), anchor)
                if key in seen:
                    continue
                seen.add(key)
                rules.append(
                    {
                        "source_color": int(source_color),
                        "target_color": int(target_color),
                        "source_pos": (int(r), int(c)),
                        "target_pos": (int(tr), int(tc)),
                        "anchor": anchor,
                        "pattern": pattern,
                    }
                )

    return rules


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn local source-to-target legends and restamp their teal templates elsewhere.

    Input structure:
    - The grid has a black background, isolated colored pixels, and teal (8) stencil components.
    - Each legend is a 4-neighbor colored pair followed by one more step in the same direction.
      That third cell is the stencil's designated anchor: the teal component's upper-left pixel.

    Rule:
    - For every non-legend pixel whose color matches a legend source color, copy that legend's
      teal stencil and recolor it to the legend target color.
    - Place the copy so the stencil's upper-left anchor pixel lands on the scattered source pixel,
      meaning the recreated shape grows only to the right and downward from that source.
    - Remove all teal pixels and erase the legend source/target teaching pixels.

    Key insight: the continued cell after a source->target pair is not just near the stencil; it
    marks the stencil's own top-left teal pixel, and copied shapes use that same corner pixel as
    the placement reference on every scattered source.
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

    for r, c in map(tuple, np.argwhere((g != int(bg)) & (g != pattern_color))):
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        if (r, c) in legend_pixels or color in source_colors:
            continue
        out[r, c] = color

    for rule in rules:
        pattern = rule["pattern"]
        ar, ac = _pattern_anchor(pattern)
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
MIN_DIM_BY_RULES = {1: 5, 2: 10, 3: 14, 4: 18}

# Rule sampling
RULE_COUNT_WEIGHTS = [1, 1, 2, 2, 3, 3, 4]
NON_BG_NON_TEAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

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

_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _reserve_4neighbors(reserved: set[tuple[int, int]], cells) -> None:
    for r, c in cells:
        for dr, dc in _DIRS4:
            reserved.add((r + dr, c + dc))


def _make_stencil(min_pixels: int, max_pixels: int) -> list[tuple[int, int]]:
    """Generate an 8-connected stencil whose occupied upper-left cell is (0, 0)."""
    n = random.randint(min_pixels, max_pixels)
    coords = {(0, 0)}
    while len(coords) < n:
        frontier = set()
        for r, c in coords:
            for dr, dc in _DIRS8:
                nr, nc = r + dr, c + dc
                if nr < 0 or nc < 0 or (nr, nc) in coords:
                    continue
                frontier.add((nr, nc))
        coords.add(random.choice(list(frontier)))
    return sorted(coords)


def _build_rules(num_rules: int, area: int):
    colors = NON_BG_NON_TEAL_COLORS[:]
    random.shuffle(colors)

    max_s = min(STENCIL_MAX_CAP, max(2, area // STENCIL_MAX_AREA_DIV))
    min_s = min(max(2, area // STENCIL_MIN_AREA_DIV), max_s)
    sc_max = min(SCATTERED_MAX_CAP, max(SCATTERED_MIN, area // SCATTERED_AREA_DIV))

    rules, used = [], set()
    for i in range(num_rules):
        src_c, tgt_c = colors[2 * i], colors[2 * i + 1]
        used.update((src_c, tgt_c))
        rel = _make_stencil(min_s, max_s)
        rules.append(
            {
                "src": src_c,
                "tgt": tgt_c,
                "rel": rel,
                "n_sc": random.randint(SCATTERED_MIN, sc_max),
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
            teal_cells = {(ar + rr, ac + cc) for rr, cc in rel}
            if any(not (0 <= nr < H and 0 <= nc < W) for nr, nc in teal_cells):
                continue
            if {(sr, sc), (tr, tc)} & teal_cells:
                continue
            all_cells = {(sr, sc), (tr, tc)} | teal_cells
            if all_cells & (occupied | reserved):
                continue

            inp[sr, sc] = rule["src"]
            inp[tr, tc] = rule["tgt"]
            for nr, nc in teal_cells:
                inp[nr, nc] = 8
            rule["source_pos"] = (sr, sc)
            rule["target_pos"] = (tr, tc)
            rule["anchor_pos"] = (ar, ac)
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
        for _ in range(rule["n_sc"] * SCATTERED_RETRIES_PER_PIXEL):
            if len(positions) >= rule["n_sc"]:
                break
            r, c = random.randint(0, H - 1), random.randint(0, W - 1)
            if (r, c) in occupied or (r, c) in reserved:
                continue
            positions.append((r, c))
            occupied.add((r, c))
            _reserve_4neighbors(reserved, [(r, c)])
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
                color = random.choice(bystander_colors)
                inp[r, c] = color
                placed.append((r, c, color))
                occupied.add((r, c))
                _reserve_4neighbors(reserved, [(r, c)])
                break
    return placed


def _build_output(rules, bystander_pos, H, W):
    output = np.zeros((H, W), dtype=int)
    for r, c, color in bystander_pos:
        output[r, c] = color
    for rule in rules:
        for sr, sc in rule["scattered"]:
            for rr, cc in rule["rel"]:
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
    occupied, reserved = set(), set()
    inp = np.zeros((H, W), dtype=int)

    if not _place_legends(rules, inp, occupied, reserved):
        return None
    if not _place_scattered(rules, inp, occupied, reserved):
        return None
    bystander_pos = _place_bystanders(used_colors, inp, occupied, reserved)

    discovered = _discover_rules(inp, bg=0, pattern_color=8)
    expected_legends = {
        (rule["source_pos"], rule["target_pos"], rule["anchor_pos"], rule["src"], rule["tgt"])
        for rule in rules
    }
    found_legends = {
        (rule["source_pos"], rule["target_pos"], rule["anchor"], rule["source_color"], rule["target_color"])
        for rule in discovered
    }
    if found_legends != expected_legends:
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
