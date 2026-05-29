# Variant of abc82100
# Dimension: D5 — 模板放置的对齐方式 (transform)
# Change: d.anchor（规则示例中的对齐参考点）对齐到源像素s的位置，模板整体平移
#   → 模板角落对齐到源像素：把模板左上角对齐到源像素，图案只向右下方展开。

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg


# Design analysis for the D5 variant
# a) Concrete behavior:
#    Old task: each scattered source pixel receives a recolored copy of the teal stencil
#    by moving the legend anchor onto that source pixel. New task: the legend still teaches
#    the same source->target color mapping and still has an anchor next to / inside the teal
#    stencil, but the copied stencil no longer uses that anchor for placement. Instead, the
#    stencil's upper-left occupied cell becomes the alignment point, so every stamped shape
#    grows rightward/downward from the source pixel. Example: if the legend stencil cells are
#    {(0,0),(0,1),(1,2)} and the legend anchor sits on cell (0,1), the old logic would place
#    the scattered copy at offsets {(0,-1),(0,0),(1,1)} from the source; the new logic places
#    it at {(0,0),(0,1),(1,2)}. The same local rule example now implies a visibly shifted stamp.
# b) Complexity calibration:
#    To keep the change observable, generated stencils always have at least two cells and the
#    chosen legend anchor is never the stencil's upper-left cell. That avoids accidental
#    degeneration back to the original anchor-aligned behavior.
# c) Hidden assumptions review:
#    The original main() depends on `dr = sr - ar`, `dc = sc - ac`, which assumes the anchor
#    is the placement reference. That is exactly the property that changes here, so stamping
#    must instead use the stencil corner `(min_row, min_col)`. Rule discovery can stay mostly
#    unchanged because the legend layout still exposes the same source/target pair and a nearby
#    teal component; only the application step changes.
# d) Placement legality:
#    Generator stencils are built in nonnegative coordinates with (0,0) occupied, so "corner"
#    is visually meaningful. Legend placement computes the stencil corner from the chosen anchor
#    offset, checks bounds, and reserves 4-neighbors around legend/scattered/bystander pixels to
#    avoid collisions and spurious extra rules.
# e) Visual observability:
#    Every rule's anchor offset from the corner is nonzero, and generation rejects samples whose
#    new output matches the old anchor-aligned output. Thus the D5 change is not merely latent.
# f) Visual cleanliness:
#    Objects remain sparse on black background, legends stay separated, and bystanders use colors
#    outside all source colors. Overlaps among output stamps are still allowed, matching the seed.
# g) Color inference:
#    Colors still serve only as local source->target labels taught by adjacent legend pairs; no
#    ordering or numeric color-id logic is introduced.


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _anchor_distance(pattern: np.ndarray, anchor: tuple[int, int]) -> int:
    return min(_chebyshev((int(r), int(c)), anchor) for r, c in pattern)


def _pattern_corner(pattern: np.ndarray) -> tuple[int, int]:
    return int(pattern[:, 0].min()), int(pattern[:, 1].min())


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
    Learn local color-substitution examples and stamp their teal templates elsewhere.

    Input structure:
    - A black field contains isolated colored pixels plus several teal (8) components.
    - Each rule example is a 4-neighbor source/target color pair whose forward direction
      reaches an anchor cell attached to a teal template component.

    Rule:
    - Match every non-legend pixel of the example's source color.
    - Recolor the nearby teal component to the example's target color and place it so the
      template's upper-left cell, not the legend anchor, lands on the matched source pixel.
    - Erase all teal pixels and all legend source/target pixels from the final grid.

    Key insight: the legend anchor still identifies which teal component belongs to a color
    pair, but it no longer controls placement; the copied stencil is positioned by its own
    upper-left corner, so each recreated shape extends to the right and downward.
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
        cr, cc = _pattern_corner(pattern)
        for sr, sc in map(tuple, np.argwhere(g == rule["source_color"])):
            sr = int(sr)
            sc = int(sc)
            if (sr, sc) in legend_pixels:
                continue
            dr = sr - cr
            dc = sc - cc
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


def _make_stencil(min_pixels: int, max_pixels: int) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    """Generate a quadrant stencil with occupied upper-left cell and non-corner anchor."""
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
    coords_list = sorted(coords)
    anchor_choices = [p for p in coords_list if p != (0, 0)]
    return coords_list, random.choice(anchor_choices)


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
        rel, anchor_offset = _make_stencil(min_s, max_s)
        rules.append(
            {
                "src": src_c,
                "tgt": tgt_c,
                "rel": rel,
                "anchor_offset": anchor_offset,
                "n_sc": random.randint(SCATTERED_MIN, sc_max),
            }
        )
    return rules, used


def _place_legends(rules, inp, occupied, reserved):
    H, W = inp.shape
    for rule in rules:
        rel = rule["rel"]
        ar_off, ac_off = rule["anchor_offset"]
        placed = False
        for _ in range(LEGEND_RETRIES):
            dr, dc = random.choice(_DIRS4)
            sr, sc = random.randint(0, H - 1), random.randint(0, W - 1)
            tr, tc = sr + dr, sc + dc
            ar, ac = sr + 2 * dr, sc + 2 * dc
            cr, cc = ar - ar_off, ac - ac_off
            if not (0 <= tr < H and 0 <= tc < W and 0 <= ar < H and 0 <= ac < W):
                continue
            teal_cells = set()
            ok = True
            for rr, cc_rel in rel:
                nr, nc = cr + rr, cc + cc_rel
                if not (0 <= nr < H and 0 <= nc < W):
                    ok = False
                    break
                teal_cells.add((nr, nc))
            if not ok:
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
                bc = random.choice(bystander_colors)
                inp[r, c] = bc
                placed.append((r, c, bc))
                occupied.add((r, c))
                _reserve_4neighbors(reserved, [(r, c)])
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


def _build_old_alignment_output(rules, bystander_pos, H, W):
    """Reference output under the seed task's anchor-aligned placement."""
    output = np.zeros((H, W), dtype=int)
    for r, c, bc in bystander_pos:
        output[r, c] = bc
    for rule in rules:
        rel = rule["rel"]
        ar_off, ac_off = rule["anchor_offset"]
        for sr, sc in rule["scattered"]:
            for rr, cc in rel:
                nr, nc = sr + rr - ar_off, sc + cc - ac_off
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
        (rule["source_pos"], rule["target_pos"], rule["src"], rule["tgt"])
        for rule in rules
    }
    found_legends = {
        (rule["source_pos"], rule["target_pos"], rule["source_color"], rule["target_color"])
        for rule in discovered
    }
    if found_legends != expected_legends:
        return None

    output = _build_output(rules, bystander_pos, H, W)
    old_output = _build_old_alignment_output(rules, bystander_pos, H, W)
    if np.array_equal(output, old_output):
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
    raise RuntimeError(f"Failed to generate valid input after {OUTER_RETRIES} attempts")
