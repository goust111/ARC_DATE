# Variant of abc82100
# Dimension: D16 — 规则示例中颜色对的空间关系 (meta)
# Change: 源色像素和目标色像素相邻（4连通或8连通），构成紧凑的颜色对... → 颜色对的排列方向编码模板旋转

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg

# Design analysis for the D16 variant
# ===================================
# a) Concrete behavior
#    - We keep the original object vocabulary: black background, several local
#      rule legends, teal stencil components, scattered source pixels, and
#      optional bystanders.
#    - The local legend still uses three collinear cells:
#        source -> target -> anchor
#      but now the source/target direction is semantically important.
#    - The teal stencil shown near the legend is always the canonical stencil,
#      not pre-rotated. The legend direction tells the solver how much to rotate
#      that canonical stencil before stamping it onto scattered source pixels.
#    - Rotation code (clockwise quarter-turns) is:
#        source left of target  -> 0 turns
#        source above target    -> 1 turn
#        source right of target -> 2 turns
#        source below target    -> 3 turns
#    - Example:
#        Legend: red source at (5,5), yellow target at (4,5), so source is
#        below target. In the original task, the teal stencil would be copied
#        unmodified onto every other red pixel. In this variant, every copied
#        stencil is rotated 270 degrees clockwise first. The color mapping is
#        still red -> yellow; only the geometry changes.
# b) Complexity calibration
#    - If the stencil were rotationally symmetric, D16 would be invisible.
#      Therefore the generator rejects symmetric stencils and requires a shape
#      with both row and column extent.
#    - Directions are sampled with diversity across rules so a single grid often
#      contains multiple rotation cases instead of repeating one trivial case.
# c) Audit of old assumptions
#    - Old `_discover_rules` assumed direction mattered only for finding the
#      anchor. That remains true for anchor discovery, but the direction must
#      now also be stored as a rotation code.
#    - Old stamping logic used raw `pattern` coordinates directly. That relied
#      on "legend geometry == output geometry". This is no longer valid, so
#      stamping must convert the pattern into anchor-relative offsets and rotate
#      those offsets before translation.
#    - The preservation logic for bystanders still works because only source-
#      colored scattered pixels are consumed; other colors remain untouched.
# d) Placement legality
#    - Legends are still kept disjoint by occupancy/reservation checks.
#    - Scattered source pixels stay 4-isolated so they cannot accidentally form
#      extra rule pairs with neighbors.
#    - Each legend’s anchor continues one more step past the target, preserving
#      the original local cue that lets the solver bind a pair to the nearby
#      teal component.
# e) Visual observability
#    - Humans can compare the canonical teal stencil in the legend with the
#      recolored output copies and see that different pair directions yield
#      different orientations.
#    - Rotations depend only on visible spatial arrangement, never on color ids.
# f) Visual cleanliness
#    - The object types remain separated exactly as in the original task:
#      legends occupy one region, scattered sources are isolated, teal only
#      appears in legends, and bystanders use uninvolved colors.
# g) Color-change inferability
#    - Color change stays local and explicit: each legend still shows a source
#      color adjacent to a target color. Direction adds geometry, not a hidden
#      recoloring rule.
# h) Generator-side anti-degeneration guard
#    - Every stencil is asymmetric under 90/180/270-degree rotations around its
#      anchor, so the chosen direction truly affects the produced output.
# i) New solver assumptions checked against the generator
#    - The solver assumes every useful colored pair has a unique nearby teal
#      component at the continued anchor position; generator placement enforces
#      that with spacing.
#    - The solver assumes rotated stamps can be clipped at the border; generator
#      allows that just like the original task.


_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
_STEP_BY_ROT = {
    0: (0, 1),
    1: (1, 0),
    2: (0, -1),
    3: (-1, 0),
}
_ROT_BY_STEP = {step: rot for rot, step in _STEP_BY_ROT.items()}


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _anchor_distance(pattern: np.ndarray, anchor: tuple[int, int]) -> int:
    return min(_chebyshev((int(r), int(c)), anchor) for r, c in pattern)


def _rotate_offset(offset: tuple[int, int], turns: int) -> tuple[int, int]:
    r, c = map(int, offset)
    for _ in range(int(turns) % 4):
        r, c = c, -r
    return int(r), int(c)


def _rotate_offsets(offsets: list[tuple[int, int]] | np.ndarray, turns: int) -> np.ndarray:
    return np.asarray([_rotate_offset((int(r), int(c)), turns) for r, c in offsets], dtype=int)


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
                        "source_pos": (int(r), int(c)),
                        "target_pos": (int(tr), int(tc)),
                        "anchor": (int(anchor[0]), int(anchor[1])),
                        "rotation": int(_ROT_BY_STEP[(int(dr), int(dc))]),
                        "pattern": np.asarray(pattern, dtype=int),
                    }
                )

    return rules


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use local source-target-pattern examples to replace matching source pixels.

    Input structure:
    - Black background with isolated colored pixels and teal (8) stencil components.
    - A rule example is a 4-neighbor source/target color pair followed by an
      anchor in the same direction, with a teal stencil touching that anchor.
    - The teal stencil shows the canonical shape; the pair direction varies
      across legends and encodes a rotation.

    Rule:
    - For each legend, recolor the nearby teal stencil to the legend's target
      color and stamp it onto every other pixel of the legend's source color.
    - Before stamping, rotate the stencil according to the source-target
      direction: left-of-target=0, above=90, right-of-target=180, below=270.
    - Translate the rotated stencil so the legend anchor lands on the scattered
      source pixel, then erase all legend pixels and all teal pixels.

    Key insight: A legend now encodes two things at once: the color mapping
    comes from the source/target colors, while their orientation supplies the
    quarter-turn applied to the teal template before copying it elsewhere.
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
        ar, ac = rule["anchor"]
        pattern = rule["pattern"]
        rel = [(int(pr) - int(ar), int(pc) - int(ac)) for pr, pc in pattern]
        rot_rel = _rotate_offsets(rel, rule["rotation"])
        for sr, sc in map(tuple, np.argwhere(g == rule["source_color"])):
            sr = int(sr)
            sc = int(sc)
            if (sr, sc) in legend_pixels:
                continue
            rows = sr + rot_rel[:, 0]
            cols = sc + rot_rel[:, 1]
            ok = (0 <= rows) & (rows < h) & (0 <= cols) & (cols < w)
            out[rows[ok], cols[ok]] = rule["target_color"]

    return out


# --------------- generator (D16 variant) ---------------

# Grid dimensions
GRID_DIM_MAX = 20
MIN_DIM_BY_RULES = {1: 5, 2: 10, 3: 14, 4: 18}

# Rule sampling
RULE_COUNT_WEIGHTS = [1, 2, 2, 3, 3, 4, 4]
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

# Retry budgets
LEGEND_RETRIES = 500
SCATTERED_RETRIES_PER_PIXEL = 50
BYSTANDER_RETRIES = 50
OUTER_RETRIES = 300
ASYMMETRIC_STENCIL_RETRIES = 200


def _reserve_4neighbors(reserved: set[tuple[int, int]], cells) -> None:
    for r, c in cells:
        for dr, dc in _DIRS4:
            reserved.add((int(r + dr), int(c + dc)))


def _is_asymmetric(rel: list[tuple[int, int]]) -> bool:
    if len({int(r) for r, _ in rel}) < 2 or len({int(c) for _, c in rel}) < 2:
        return False
    base = tuple(sorted((int(r), int(c)) for r, c in rel))
    for turns in (1, 2, 3):
        rotated = tuple(sorted(_rotate_offset((int(r), int(c)), turns) for r, c in rel))
        if rotated == base:
            return False
    return True


def _make_stencil(min_pixels: int, max_pixels: int) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    lo = max(3, int(min_pixels))
    hi = max(lo, int(max_pixels))
    for _ in range(ASYMMETRIC_STENCIL_RETRIES):
        n = random.randint(lo, hi)
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
        anchor = random.choice(coords_list)
        rel = [(int(r - anchor[0]), int(c - anchor[1])) for r, c in coords_list]
        if _is_asymmetric(rel):
            return coords_list, anchor

    coords_list = [(0, 0), (1, 0), (1, 1)]
    anchor = (1, 0)
    return coords_list, anchor


def _choose_rotations(num_rules: int) -> list[int]:
    if num_rules == 1:
        return [random.choice([1, 2, 3])]
    dirs = [0, 1, 2, 3]
    random.shuffle(dirs)
    return dirs[:num_rules]


def _build_rules(num_rules: int, area: int):
    colors = NON_BG_NON_TEAL_COLORS[:]
    random.shuffle(colors)

    max_s = min(STENCIL_MAX_CAP, max(3, area // STENCIL_MAX_AREA_DIV))
    min_s = min(max(1, area // STENCIL_MIN_AREA_DIV), max_s)
    sc_max = min(SCATTERED_MAX_CAP, max(SCATTERED_MIN, area // SCATTERED_AREA_DIV))
    rotations = _choose_rotations(num_rules)

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
                "rot": int(rotations[i]),
                "n_sc": random.randint(SCATTERED_MIN, sc_max),
            }
        )
    return rules, used


def _place_legends(rules, inp: np.ndarray, occupied: set, reserved: set) -> bool:
    H, W = inp.shape
    for rule in rules:
        rel = rule["rel"]
        dr, dc = _STEP_BY_ROT[int(rule["rot"])]
        placed = False
        for _ in range(LEGEND_RETRIES):
            sr, sc = random.randint(0, H - 1), random.randint(0, W - 1)
            tr, tc = sr + dr, sc + dc
            ar, ac = sr + 2 * dr, sc + 2 * dc
            if not (0 <= tr < H and 0 <= tc < W and 0 <= ar < H and 0 <= ac < W):
                continue

            teal_cells = set()
            ok = True
            for rr, cc in rel:
                nr, nc = ar + int(rr), ac + int(cc)
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


def _place_scattered(rules, inp: np.ndarray, occupied: set, reserved: set) -> bool:
    H, W = inp.shape
    for rule in rules:
        positions = []
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


def _place_bystanders(used_colors, inp: np.ndarray, occupied: set, reserved: set):
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
                inp[r, c] = int(bc)
                placed.append((int(r), int(c), int(bc)))
                occupied.add((int(r), int(c)))
                _reserve_4neighbors(reserved, [(int(r), int(c))])
                break
    return placed


def _build_output(rules, bystander_pos, H: int, W: int) -> np.ndarray:
    output = np.zeros((H, W), dtype=int)
    for r, c, bc in bystander_pos:
        output[int(r), int(c)] = int(bc)
    for rule in rules:
        rot_rel = _rotate_offsets(rule["rel"], rule["rot"])
        for sr, sc in rule["scattered"]:
            rows = int(sr) + rot_rel[:, 0]
            cols = int(sc) + rot_rel[:, 1]
            ok = (0 <= rows) & (rows < H) & (0 <= cols) & (cols < W)
            output[rows[ok], cols[ok]] = int(rule["tgt"])
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
