# Variant of abc82100
# Dimension: D12 — 多模板重叠处理 (multi_step)
# Change: 多个源像素各自独立放置模板，重叠区域后放置的覆盖先放置的... → 重叠区域变为特殊颜色

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg


# Design analysis for the D12 variant
# a) Concrete behavior:
#    Each learned rule is still "source pixel -> target pixel -> nearby teal stencil".
#    The new behavior only changes what happens after all scattered source pixels stamp
#    their translated stencils. Example: one blue source may stamp a red shape and one
#    green source may stamp a yellow shape; if both translated shapes cover the same
#    output cell, that cell becomes a fixed grey conflict marker instead of whichever
#    stamp happened to be written later.
# b) Complexity calibration:
#    The task needs at least two rules, otherwise no cross-color overlap can happen.
#    Grey (5) is reserved as the conflict color and is never used for rules, so the
#    collision marker is visually stable across samples. Each grid also keeps several
#    non-overlapping source pixels so the solver can still infer the base stamping rule.
# c) Hidden assumptions from the original solver:
#    `out[rows, cols] = target` assumed last-write-wins and therefore erased evidence of
#    overlap. That is invalid here. Also, ordinary preserved pixels cannot participate in
#    conflict detection, so stamping must happen on a separate layer before merging with
#    bystanders. The generator must now force at least one overlap; random placement is
#    not enough because a collision-free sample would collapse back to the old behavior.
# d) Placement legality:
#    Legend triplets still avoid one another and avoid scattered sources via occupied /
#    reserved cells. The forced conflict is created by choosing one relative stencil cell
#    from rule A and one from rule B and aligning both onto the same in-bounds board cell,
#    while keeping the two source pixels distinct and non-4-adjacent.
# e) Visual observability:
#    Grey appears only in output and only at overlap cells, so a human can see that the
#    normal target-colored stamps are still present while cross-rule collisions are marked
#    specially. Samples include both ordinary stamps and at least one grey collision.
# f) Visual cleanliness:
#    Only the output templates overlap; input legends remain separated and the scattered
#    source pixels stay isolated. This keeps the input readable while making the output
#    show a small number of crisp conflict regions rather than dense clutter.
# g) Color inference:
#    The recoloring rule is still given explicitly by the local examples. Grey is not a
#    hidden color-id trick; it is a visible structural marker that depends on whether two
#    different target colors occupy the same output cell.

BG_COLOR = 0
PATTERN_COLOR = 8
CONFLICT_COLOR = 5


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _anchor_distance(pattern: np.ndarray, anchor: tuple[int, int]) -> int:
    return min(_chebyshev((int(r), int(c)), anchor) for r, c in pattern)


def _discover_rules(grid: np.ndarray, bg: int, pattern_color: int = PATTERN_COLOR) -> list[dict]:
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


def _accumulate_stamp(
    stamp_color: np.ndarray,
    conflict_mask: np.ndarray,
    row: int,
    col: int,
    target_color: int,
    bg: int,
) -> None:
    current = int(stamp_color[row, col])
    if current == int(bg):
        stamp_color[row, col] = int(target_color)
    elif current != int(target_color):
        conflict_mask[row, col] = True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn local source-target-stencil examples and stamp them onto matching sources,
    turning cross-color overlaps into a dedicated conflict color.

    Input structure:
    - Black background with isolated colored pixels and teal (8) stencil components.
    - Each rule example is a 4-neighbor source/target pair whose direction continues
      to an anchor touching the teal stencil for that rule.

    Rule:
    - For every non-legend source pixel of a learned source color, translate that
      rule's teal stencil so the anchor lands on the source pixel, then recolor the
      stencil to the rule's target color.
    - If stamps of different target colors land on the same output cell, replace that
      cell with the fixed conflict color grey (5); same-color overlaps stay unchanged.
    - Remove all legend pixels and all teal stencil pixels from the final output.

    Key insight: The local colored pair still teaches which stencil to copy and which
    color to paint, but overlapping translated copies now encode an extra interaction:
    disagreement between target colors is shown explicitly as grey.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    h, w = g.shape

    rules = _discover_rules(g, bg=bg, pattern_color=PATTERN_COLOR)
    legend_pixels: set[tuple[int, int]] = set()
    source_colors: set[int] = set()
    for rule in rules:
        legend_pixels.add(rule["source_pos"])
        legend_pixels.add(rule["target_pos"])
        source_colors.add(rule["source_color"])

    out = np.full_like(g, int(bg))

    for r, c in map(tuple, np.argwhere((g != int(bg)) & (g != PATTERN_COLOR))):
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        if (r, c) in legend_pixels or color in source_colors:
            continue
        out[r, c] = color

    stamp_color = np.full_like(g, int(bg))
    conflict_mask = np.zeros((h, w), dtype=bool)

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
            for rr, cc in zip(rows[ok], cols[ok]):
                _accumulate_stamp(
                    stamp_color,
                    conflict_mask,
                    int(rr),
                    int(cc),
                    int(rule["target_color"]),
                    int(bg),
                )

    stamp_mask = stamp_color != int(bg)
    out[stamp_mask] = stamp_color[stamp_mask]
    out[conflict_mask] = CONFLICT_COLOR
    return out


# --------------- generator (c) ---------------

# Grid dimensions
GRID_DIM_MAX = 20
MIN_DIM_BY_RULES = {2: 10, 3: 14}

# Rule sampling
RULE_COUNT_WEIGHTS = [2, 2, 3, 3]
NON_BG_NON_TEAL_COLORS = [1, 2, 3, 4, 6, 7, 9]

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
CONFLICT_PAIR_RETRIES = 800
BYSTANDER_RETRIES = 50
OUTER_RETRIES = 300

_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _reserve_4neighbors(reserved: set[tuple[int, int]], cells) -> None:
    for r, c in cells:
        for dr, dc in _DIRS4:
            reserved.add((int(r) + dr, int(c) + dc))


def _reserve_chebyshev1(reserved: set[tuple[int, int]], cells) -> None:
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                reserved.add((int(r) + dr, int(c) + dc))


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
            }
        )
    return rules, used


def _stamp_cells(rel, sr: int, sc: int, h: int, w: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for rr, cc in rel:
        nr = int(sr + rr)
        nc = int(sc + cc)
        if 0 <= nr < h and 0 <= nc < w:
            cells.append((nr, nc))
    return cells


def _place_legends(rules, inp, occupied, reserved):
    h, w = inp.shape
    for rule in rules:
        rel = rule["rel"]
        placed = False
        for _ in range(LEGEND_RETRIES):
            dr, dc = random.choice(_DIRS4)
            sr, sc = random.randint(0, h - 1), random.randint(0, w - 1)
            tr, tc = sr + dr, sc + dc
            ar, ac = sr + 2 * dr, sc + 2 * dc
            if not (0 <= tr < h and 0 <= tc < w and 0 <= ar < h and 0 <= ac < w):
                continue
            teal_cells = set()
            ok = True
            for rr, cc in rel:
                nr, nc = ar + rr, ac + cc
                if not (0 <= nr < h and 0 <= nc < w):
                    ok = False
                    break
                teal_cells.add((nr, nc))
            if not ok:
                continue
            reverse_anchor = (sr - dr, sc - dc)
            teal_arr = np.asarray(sorted(teal_cells), dtype=int)
            if _anchor_distance(teal_arr, reverse_anchor) <= 1:
                continue
            reverse_zone = {
                (reverse_anchor[0] + rr, reverse_anchor[1] + cc)
                for rr in (-1, 0, 1)
                for cc in (-1, 0, 1)
            }
            if reverse_zone & (occupied | reserved):
                continue
            all_cells = {(sr, sc), (tr, tc), (ar, ac)} | teal_cells
            if all_cells & (occupied | reserved):
                continue
            inp[sr, sc] = rule["src"]
            inp[tr, tc] = rule["tgt"]
            for nr, nc in teal_cells:
                inp[nr, nc] = PATTERN_COLOR
            occupied.update(all_cells)
            _reserve_4neighbors(reserved, all_cells)
            _reserve_chebyshev1(reserved, [reverse_anchor])
            placed = True
            break
        if not placed:
            return False
    return True


def _add_scattered(rule, pos: tuple[int, int], inp, occupied, reserved) -> None:
    r, c = int(pos[0]), int(pos[1])
    rule["scattered"].append((r, c))
    inp[r, c] = rule["src"]
    occupied.add((r, c))
    _reserve_4neighbors(reserved, [(r, c)])


def _place_conflict_pair(rules, inp, occupied, reserved) -> bool:
    h, w = inp.shape
    pair_indices = random.sample(range(len(rules)), 2)
    ia, ib = int(pair_indices[0]), int(pair_indices[1])
    rule_a = rules[ia]
    rule_b = rules[ib]

    rel_a = rule_a["rel"][:]
    rel_b = rule_b["rel"][:]
    random.shuffle(rel_a)
    random.shuffle(rel_b)

    for _ in range(CONFLICT_PAIR_RETRIES):
        rr_a, cc_a = random.choice(rel_a)
        rr_b, cc_b = random.choice(rel_b)
        overlap_r = random.randint(0, h - 1)
        overlap_c = random.randint(0, w - 1)
        src_a = (overlap_r - rr_a, overlap_c - cc_a)
        src_b = (overlap_r - rr_b, overlap_c - cc_b)

        if not (0 <= src_a[0] < h and 0 <= src_a[1] < w and 0 <= src_b[0] < h and 0 <= src_b[1] < w):
            continue
        if src_a == src_b:
            continue
        if abs(src_a[0] - src_b[0]) + abs(src_a[1] - src_b[1]) == 1:
            continue
        if src_a in occupied or src_a in reserved or src_b in occupied or src_b in reserved:
            continue

        _add_scattered(rule_a, src_a, inp, occupied, reserved)
        if src_b in occupied or src_b in reserved:
            return False
        _add_scattered(rule_b, src_b, inp, occupied, reserved)
        return True

    return False


def _place_scattered(rules, inp, occupied, reserved):
    h, w = inp.shape
    for rule in rules:
        rule["scattered"] = []

    if len(rules) < 2 or not _place_conflict_pair(rules, inp, occupied, reserved):
        return False

    for rule in rules:
        for _ in range(rule["n_sc"] * SCATTERED_RETRIES_PER_PIXEL):
            if len(rule["scattered"]) >= rule["n_sc"]:
                break
            r, c = random.randint(0, h - 1), random.randint(0, w - 1)
            if (r, c) in occupied or (r, c) in reserved:
                continue
            _add_scattered(rule, (r, c), inp, occupied, reserved)
        if len(rule["scattered"]) < SCATTERED_MIN:
            return False

    return True


def _collect_stamp_footprint(rules, h: int, w: int) -> set[tuple[int, int]]:
    footprint: set[tuple[int, int]] = set()
    for rule in rules:
        for sr, sc in rule["scattered"]:
            footprint.update(_stamp_cells(rule["rel"], sr, sc, h, w))
    return footprint


def _place_bystanders(used_colors, inp, occupied, reserved, forbidden):
    bystander_colors = [c for c in NON_BG_NON_TEAL_COLORS if c not in used_colors]
    if not bystander_colors:
        return []
    h, w = inp.shape
    n = random.randint(0, min(BYSTANDER_MAX, len(bystander_colors) * 2))
    placed = []
    for _ in range(n):
        for _ in range(BYSTANDER_RETRIES):
            r, c = random.randint(0, h - 1), random.randint(0, w - 1)
            if (r, c) in occupied or (r, c) in reserved or (r, c) in forbidden:
                continue
            bc = random.choice(bystander_colors)
            inp[r, c] = bc
            placed.append((r, c, bc))
            occupied.add((r, c))
            _reserve_4neighbors(reserved, [(r, c)])
            break
    return placed


def _build_output(rules, bystander_pos, h: int, w: int):
    output = np.zeros((h, w), dtype=int)
    for r, c, bc in bystander_pos:
        output[int(r), int(c)] = int(bc)

    stamp_color = np.zeros((h, w), dtype=int)
    conflict_mask = np.zeros((h, w), dtype=bool)
    for rule in rules:
        for sr, sc in rule["scattered"]:
            for nr, nc in _stamp_cells(rule["rel"], sr, sc, h, w):
                _accumulate_stamp(stamp_color, conflict_mask, nr, nc, int(rule["tgt"]), BG_COLOR)

    stamp_mask = stamp_color != BG_COLOR
    output[stamp_mask] = stamp_color[stamp_mask]
    output[conflict_mask] = CONFLICT_COLOR
    return output


def _gen_attempt():
    num_rules = random.choice(RULE_COUNT_WEIGHTS)
    min_dim = MIN_DIM_BY_RULES[num_rules]
    h = random.randint(min_dim, GRID_DIM_MAX)
    w = random.randint(min_dim, GRID_DIM_MAX)

    rules, used_colors = _build_rules(num_rules, h * w)
    occupied, reserved = set(), set()
    inp = np.zeros((h, w), dtype=int)

    if not _place_legends(rules, inp, occupied, reserved):
        return None
    if not _place_scattered(rules, inp, occupied, reserved):
        return None

    footprint = _collect_stamp_footprint(rules, h, w)
    bystander_pos = _place_bystanders(used_colors, inp, occupied, reserved, footprint)
    output = _build_output(rules, bystander_pos, h, w)

    if not np.any(output == CONFLICT_COLOR):
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
