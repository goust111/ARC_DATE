# Variant of abc82100
# Dimension: D16 — 规则示例中颜色对的空间关系 (meta)
# Change: 源色像素和目标色像素相邻（4连通或8连通），构成紧凑的颜色对... → 三像素编码（源色-分隔符-目标色）

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg


# Design analysis for the D16 variant
# ===================================
# a) Concrete behavior:
#    - Old legend: `source -> target -> anchor`, where source/target were directly
#      adjacent colored cells.
#    - New legend: `source -> gray(5) -> target -> anchor`, all four cells
#      collinear in one cardinal direction, with the teal stencil touching the
#      anchor exactly as before.
#    - Small trace:
#        old: 1 2 8  means "color-1 pixels become the nearby teal stencil in 2".
#        new: 1 5 2 8 means the same replacement, but the separator explicitly
#        marks that the first and third colored cells are one encoded rule.
#      The changed step is rule discovery only; stencil translation/recoloring is
#      intentionally preserved.
# b) Complexity calibration:
#    - The separator adds visible structure without adding many free cases:
#      it is always color 5 and always sits between source and target.
#    - Source colors remain unique per rule so scattered pixels are unambiguous.
#      Targets may repeat when there are 4 rules because gray is no longer
#      available as a source/target color.
# c) Hidden assumptions from the old main():
#    - Old `_discover_rules` assumed every valid rule started with two adjacent
#      non-special colors; that is false now, so discovery must require a gray
#      separator at offset +1 and the target at offset +2.
#    - Old preservation logic treated every non-bg/non-teal color as a possible
#      survivor unless it matched a learned source color. That would wrongly keep
#      or inspect separator pixels, so gray(5) must now be treated as structural.
#    - The solver now assumes non-legend gray pixels do not appear; generator
#      enforces this by reserving color 5 exclusively for separators.
# d) Placement legality:
#    - Each legend occupies four collinear cells plus one teal component.
#    - Generator checks bounds for source, separator, target, anchor, and every
#      stencil cell, and reserves 4-neighbors to avoid collisions and false rules.
# e) Visual observability:
#    - Across examples, the most obvious change is the repeated gray cell between
#      the two informative colors. Human solvers can still infer the same mapping
#      because each legend keeps one source color, one target color, and one teal
#      stencil, but the boundary cue is now explicit.
# f) Visual cleanliness:
#    - Gray is removed from the general palette so the grid does not contain
#      ambiguous standalone separators or bystanders.
#    - Reservation keeps legends and scattered pixels visually separated.
# g) Color inferability:
#    - The mapping clue is still local and repeated: one legend teaches one
#      source->target substitution, and several scattered source pixels reuse it.
# h) New solver assumptions audited against the generator:
#    - Every true legend has a unique gray separator at +1 step: `_place_legends`.
#    - Every source color belongs to only one rule: `_build_rules`.
#    - No bystander uses gray or any source color: `_place_bystanders`.


PATTERN_COLOR = 8
SEPARATOR_COLOR = 5


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _anchor_distance(pattern: np.ndarray, anchor: tuple[int, int]) -> int:
    return min(_chebyshev((int(r), int(c)), anchor) for r, c in pattern)


def _discover_rules(
    grid: np.ndarray,
    bg: int,
    pattern_color: int = PATTERN_COLOR,
    separator_color: int = SEPARATOR_COLOR,
) -> list[dict]:
    h, w = grid.shape
    patterns = components_mask(grid == pattern_color, connectivity=8)
    color_counts = {int(color): int(np.sum(grid == int(color))) for color in np.unique(grid)}
    rules: list[dict] = []
    seen: set[tuple[tuple[int, int], tuple[int, int], tuple[int, int], int]] = set()

    for r in range(h):
        for c in range(w):
            source_color = int(grid[r, c])
            if source_color in (int(bg), int(pattern_color), int(separator_color)):
                continue
            if color_counts.get(source_color, 0) < 2:
                continue

            for dr, dc in _DIRS4:
                mr = r + dr
                mc = c + dc
                tr = r + 2 * dr
                tc = c + 2 * dc
                ar = r + 3 * dr
                ac = c + 3 * dc
                if not (0 <= mr < h and 0 <= mc < w and 0 <= tr < h and 0 <= tc < w):
                    continue
                if int(grid[mr, mc]) != int(separator_color):
                    continue

                target_color = int(grid[tr, tc])
                if target_color in (int(bg), int(pattern_color), int(separator_color)):
                    continue
                if target_color == source_color or not (0 <= ar < h and 0 <= ac < w):
                    continue

                anchor = (ar, ac)
                candidates: list[tuple[int, int, np.ndarray]] = []
                for i, pattern in enumerate(patterns):
                    dist = _anchor_distance(pattern, anchor)
                    if dist <= 1:
                        candidates.append((int(dist), int(i), pattern))

                if not candidates:
                    continue

                candidates.sort(key=lambda item: (item[0], len(item[2])))
                _dist, pattern_index, pattern = candidates[0]
                key = ((int(r), int(c)), (int(mr), int(mc)), (int(tr), int(tc)), int(pattern_index))
                if key in seen:
                    continue
                seen.add(key)
                rules.append(
                    {
                        "source_color": int(source_color),
                        "target_color": int(target_color),
                        "source_pos": (int(r), int(c)),
                        "separator_pos": (int(mr), int(mc)),
                        "target_pos": (int(tr), int(tc)),
                        "anchor": (int(anchor[0]), int(anchor[1])),
                        "pattern": np.asarray(pattern, dtype=int),
                    }
                )

    return rules


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use local source-separator-target-pattern examples to replace matching pixels.

    Input structure:
    - Black background with isolated colored source pixels and teal (8) stencil
      components.
    - A rule example is four collinear cells: source color, gray separator (5),
      target color, then an anchor that touches a teal component.

    Rule:
    - For each learned example, stamp that teal component onto every other pixel
      of the same source color, recoloring the stamp to the rule's target color.
    - The translation still aligns the legend anchor with the scattered source
      pixel being replaced.
    - Erase the rule examples, separators, anchors, and teal template pixels.

    Key insight: The gray cell is not part of the output; it is a structural cue
    that the two colored cells on its sides define one source->target mapping.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    pattern_color = PATTERN_COLOR
    separator_color = SEPARATOR_COLOR
    h, w = g.shape

    rules = _discover_rules(g, bg=bg, pattern_color=pattern_color, separator_color=separator_color)
    legend_pixels: set[tuple[int, int]] = set()
    source_colors: set[int] = set()
    for rule in rules:
        legend_pixels.add(rule["source_pos"])
        legend_pixels.add(rule["separator_pos"])
        legend_pixels.add(rule["target_pos"])
        source_colors.add(rule["source_color"])

    out = np.full_like(g, int(bg))

    # Preserve ordinary colored pixels that are not structural and not consumed.
    keep_mask = (g != int(bg)) & (g != pattern_color) & (g != separator_color)
    for r, c in map(tuple, np.argwhere(keep_mask)):
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
NON_SPECIAL_COLORS = [1, 2, 3, 4, 6, 7, 9]

STENCIL_MAX_CAP = 12
STENCIL_MIN_AREA_DIV = 100
STENCIL_MAX_AREA_DIV = 25

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
    """Add all 4-neighbors of *cells* into *reserved*."""
    for r, c in cells:
        for dr, dc in _DIRS4:
            reserved.add((r + dr, c + dc))


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
    """Assign source/target pairs and generate stencils for *num_rules* rules."""
    colors = NON_SPECIAL_COLORS[:]
    random.shuffle(colors)

    max_s = min(STENCIL_MAX_CAP, max(2, area // STENCIL_MAX_AREA_DIV))
    min_s = min(max(1, area // STENCIL_MIN_AREA_DIV), max_s)
    sc_max = min(SCATTERED_MAX_CAP, max(SCATTERED_MIN, area // SCATTERED_AREA_DIV))

    source_colors = colors[:num_rules]
    rules: list[dict] = []
    used_pairs: set[tuple[int, int]] = set()
    used_targets: set[int] = set()
    for src_c in source_colors:
        candidates = [c for c in NON_SPECIAL_COLORS if c != src_c and (src_c, c) not in used_pairs]
        source_avoiding = [c for c in candidates if c not in source_colors]
        fresh = [c for c in source_avoiding if c not in used_targets]
        if not fresh:
            fresh = [c for c in candidates if c not in used_targets]
        tgt_c = random.choice(fresh or candidates)
        used_pairs.add((src_c, tgt_c))
        used_targets.add(tgt_c)
        stencil, anchor = _make_stencil(min_s, max_s)
        rel = [(r - anchor[0], c - anchor[1]) for r, c in stencil]
        rules.append(
            {
                "src": src_c,
                "tgt": tgt_c,
                "rel": rel,
                "n_sc": random.randint(SCATTERED_MIN, sc_max),
            }
        )

    used_colors = set(source_colors) | used_targets | {SEPARATOR_COLOR}
    return rules, used_colors


def _place_legends(rules, inp, occupied, reserved):
    """Place rule examples `source, gray, target, anchor+teal` on *inp*."""
    H, W = inp.shape
    for rule in rules:
        rel = rule["rel"]
        placed = False
        for _ in range(LEGEND_RETRIES):
            dr, dc = random.choice(_DIRS4)
            sr, sc = random.randint(0, H - 1), random.randint(0, W - 1)
            mr, mc = sr + dr, sc + dc
            tr, tc = sr + 2 * dr, sc + 2 * dc
            ar, ac = sr + 3 * dr, sc + 3 * dc
            if not (
                0 <= mr < H
                and 0 <= mc < W
                and 0 <= tr < H
                and 0 <= tc < W
                and 0 <= ar < H
                and 0 <= ac < W
            ):
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
            reverse_anchor = (sr - dr, sc - dc)
            if any(_chebyshev(cell, reverse_anchor) <= 1 for cell in teal_cells):
                continue

            all_cells = {(sr, sc), (mr, mc), (tr, tc), (ar, ac)} | teal_cells
            if all_cells & (occupied | reserved):
                continue

            inp[sr, sc] = rule["src"]
            inp[mr, mc] = SEPARATOR_COLOR
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


def _place_scattered(rules, inp, occupied, reserved):
    """Place isolated scattered source pixels for each rule."""
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
    """Place unrelated colored pixels that should survive unchanged."""
    bystander_colors = [c for c in NON_SPECIAL_COLORS if c not in used_colors]
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
    """Construct the expected output by stamping recolored stencils."""
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
    """Single generation attempt. Returns input grid or None on failure."""
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
