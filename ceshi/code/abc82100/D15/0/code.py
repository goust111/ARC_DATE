# Variant of abc82100
# Dimension: D15 — 边界处理规则 (meta)
# Change: 模板放置时超出网格边界的像素被截断/忽略... → 模板不允许超出边界（跳过超界源像素）

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg

# Design analysis
# a) Concrete behavior and trace:
#    The input still contains local "legends": a source-colored pixel, its 4-neighbor
#    target-colored pixel, and a nearby teal stencil whose anchor lies one step farther
#    along the same direction. Other isolated source-colored pixels are the instances to
#    transform. The D15 change only affects what happens when translating the teal stencil
#    onto a scattered source pixel near an edge.
#    Example: suppose a rule's relative stencil cells are {(0, 0), (0, 1), (1, 0)} with
#    the anchor at the source pixel. For a scattered source at (4, 4) in a 5x5 grid, the
#    original solver would clip the translated cells to {(4, 4)} and paint that partial
#    shape. In this variant, because (4, 5) and (5, 4) would leave the grid, the entire
#    transform is skipped and the output leaves that source with no replacement at all.
#    For an interior source such as (2, 2), the full 3-cell shape is still stamped.
# b) Complexity calibration:
#    The generator keeps 1-4 rules, unique source->target color pairs, and small teal
#    connected stencils. To make the new boundary rule visible, each rule gets both:
#    - interior valid scattered sources that produce full recolored stamps
#    - edge-biased invalid scattered sources whose translated stencil would cross a border
#    This avoids degeneration back to the original behavior while keeping the task family
#    recognizable.
# c) Audit of original assumptions:
#    - The original `ok = ...` mask in `main()` assumed per-pixel clipping was legal.
#      That is no longer true; partial writes would violate the new rule.
#    - Preserving non-source bystanders remains valid because bystander colors never equal
#      any learned source color, so skipped invalid sources still disappear cleanly.
#    - Rule discovery does not depend on clipping, so `_discover_rules()` can stay intact.
# d) Placement legality:
#    Legend triplets still require in-bounds source, target, anchor, and a fully in-bounds
#    teal stencil component. Scattered source pixels remain isolated by 4-neighbor
#    reservation so they do not create spurious source-target pairs. Invalid scattered
#    sources are legal input objects even though their implied output stamp would leave the
#    board; the solver is expected to skip them.
# e) Visual observability:
#    The generator biases invalid scattered sources toward the outer border and valid
#    scattered sources toward the interior. Across examples, a human can see that interior
#    source pixels expand into the target-colored stencil while edge-hugging ones vanish.
# f) Visual cleanliness:
#    Legends, scattered sources, and bystanders are separated in the input by reservation.
#    Bystanders are also kept away from valid output footprints so preserved pixels remain
#    visually distinct in the output.
# g) Color inference:
#    Colors still work exactly as in the base task: the adjacent legend pair teaches which
#    source color maps to which target color. The boundary decision uses only visible
#    geometry relative to the grid edge, never hidden color ordering.
# h) Variant metric used in testing:
#    Count scattered source pixels whose full translated stencil would leave the grid.
#    Samples should have this count > 0, and a clipped-output baseline should differ from
#    the strict-output solver on many generated inputs.
# i) New solver assumptions and generator guarantees:
#    - `main()` now assumes a skipped source should leave no residue at all.
#      The generator therefore models outputs from a blank grid plus valid stamps only.
#    - `main()` still removes every source-colored scattered pixel regardless of validity.
#      The generator never uses source colors for bystanders, so this removal is safe.


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
                        "pattern": np.asarray(pattern, dtype=int),
                    }
                )

    return rules


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn local source-target-stencil examples and apply them to matching pixels.

    Input structure:
    - The grid has a black background, isolated colored pixels, and teal (8) stencil
      components.
    - Each rule example is a 4-neighbor colored pair source->target, followed in the
      same direction by an anchor touching the teal stencil for that rule.

    Rule:
    - Every non-legend pixel with the learned source color tries to receive a translated
      copy of the teal stencil, recolored to the learned target color.
    - The translation aligns the rule anchor with the source pixel being processed.
    - If any cell of that translated stencil would leave the grid, skip that source
      entirely; do not place a clipped partial stencil. Remove all legend pixels and all
      teal stencil pixels from the final output.

    Key insight:
    - The legend pair tells you both the color substitution and which teal component is
      the stencil, while the board edges determine whether each scattered source is valid
      enough to receive the whole translated copy.
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

    # Preserve ordinary colored pixels that are not consumed by any learned rule.
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
            if not ((0 <= rows).all() and (rows < h).all() and (0 <= cols).all() and (cols < w).all()):
                continue
            out[rows, cols] = rule["target_color"]

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
VALID_SCATTERED_MIN = 2
INVALID_SCATTERED_MIN = 1
INVALID_SCATTERED_MAX = 2

# Bystander pixels
BYSTANDER_MAX = 5

# Placement retry budgets
LEGEND_RETRIES = 500
OUTER_RETRIES = 300

# Cardinal directions (4-connected)
_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
# 8-connected neighbor offsets
_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _reserve_4neighbors(reserved: set[tuple[int, int]], cells) -> None:
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
                p = (r + dr, c + dc)
                if p not in coords:
                    frontier.add(p)
        coords.add(random.choice(list(frontier)))
    coords_list = sorted(coords)
    return coords_list, random.choice(coords_list)


def _fits_rel(rel: list[tuple[int, int]], sr: int, sc: int, H: int, W: int) -> bool:
    for rr, cc in rel:
        nr = int(sr) + int(rr)
        nc = int(sc) + int(cc)
        if not (0 <= nr < H and 0 <= nc < W):
            return False
    return True


def _stamp_cells(rel: list[tuple[int, int]], sr: int, sc: int) -> set[tuple[int, int]]:
    return {(int(sr) + int(rr), int(sc) + int(cc)) for rr, cc in rel}


def _edge_distance(r: int, c: int, H: int, W: int) -> int:
    return min(int(r), int(c), H - 1 - int(r), W - 1 - int(c))


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
        stencil, anchor = _make_stencil(min_s, max_s)
        rel = [(r - anchor[0], c - anchor[1]) for r, c in stencil]
        n_sc = random.randint(SCATTERED_MIN, sc_max)
        n_invalid = random.randint(INVALID_SCATTERED_MIN, min(INVALID_SCATTERED_MAX, n_sc - VALID_SCATTERED_MIN))
        rules.append(
            {
                "src": src_c,
                "tgt": tgt_c,
                "rel": rel,
                "n_sc": n_sc,
                "n_invalid": n_invalid,
                "n_valid": n_sc - n_invalid,
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
            placed = True
            break
        if not placed:
            return False
    return True


def _choose_candidate(candidates, H, W, prefer_edge: bool):
    if not candidates:
        return None
    decorated = [(_edge_distance(r, c, H, W), (r, c)) for r, c in candidates]
    decorated.sort(key=lambda item: item[0], reverse=not prefer_edge)
    window = max(1, min(len(decorated), 6))
    return random.choice([pos for _dist, pos in decorated[:window]])


def _place_scattered(rules, inp, occupied, reserved):
    H, W = inp.shape
    valid_output_cells: set[tuple[int, int]] = set()

    for rule in rules:
        rel = rule["rel"]
        valid_positions: list[tuple[int, int]] = []
        invalid_positions: list[tuple[int, int]] = []
        all_positions: list[tuple[int, int]] = []

        for _ in range(rule["n_invalid"]):
            candidates = [
                (r, c)
                for r in range(H)
                for c in range(W)
                if (r, c) not in occupied and (r, c) not in reserved and not _fits_rel(rel, r, c, H, W)
            ]
            pos = _choose_candidate(candidates, H, W, prefer_edge=True)
            if pos is None:
                return None
            invalid_positions.append(pos)
            all_positions.append(pos)
            occupied.add(pos)
            _reserve_4neighbors(reserved, [pos])

        for _ in range(rule["n_valid"]):
            candidates = []
            for r in range(H):
                for c in range(W):
                    if (r, c) in occupied or (r, c) in reserved:
                        continue
                    if not _fits_rel(rel, r, c, H, W):
                        continue
                    stamp = _stamp_cells(rel, r, c)
                    if stamp & valid_output_cells:
                        continue
                    candidates.append((r, c))
            pos = _choose_candidate(candidates, H, W, prefer_edge=False)
            if pos is None:
                return None
            valid_positions.append(pos)
            all_positions.append(pos)
            occupied.add(pos)
            _reserve_4neighbors(reserved, [pos])
            valid_output_cells.update(_stamp_cells(rel, pos[0], pos[1]))

        if len(valid_positions) < VALID_SCATTERED_MIN or len(invalid_positions) < INVALID_SCATTERED_MIN:
            return None

        rule["valid_scattered"] = valid_positions
        rule["invalid_scattered"] = invalid_positions
        rule["scattered"] = all_positions
        for r, c in all_positions:
            inp[r, c] = rule["src"]

    return valid_output_cells


def _place_bystanders(used_colors, inp, occupied, reserved, blocked_output):
    bystander_colors = [c for c in NON_BG_NON_TEAL_COLORS if c not in used_colors]
    if not bystander_colors:
        return []
    H, W = inp.shape
    n = random.randint(0, min(BYSTANDER_MAX, len(bystander_colors) * 2))
    placed = []
    for _ in range(n):
        candidates = [
            (r, c)
            for r in range(H)
            for c in range(W)
            if (r, c) not in occupied and (r, c) not in reserved and (r, c) not in blocked_output
        ]
        if not candidates:
            break
        r, c = random.choice(candidates)
        bc = random.choice(bystander_colors)
        inp[r, c] = bc
        placed.append((r, c, bc))
        occupied.add((r, c))
        _reserve_4neighbors(reserved, [(r, c)])
    return placed


def _build_output(rules, bystander_pos, H, W):
    output = np.zeros((H, W), dtype=int)
    for r, c, bc in bystander_pos:
        output[r, c] = bc
    for rule in rules:
        rel = rule["rel"]
        for sr, sc in rule["valid_scattered"]:
            for rr, cc in rel:
                nr, nc = sr + rr, sc + cc
                output[nr, nc] = rule["tgt"]
    return output


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
    valid_output_cells = _place_scattered(rules, inp, occupied, reserved)
    if valid_output_cells is None:
        return None
    if sum(len(rule["invalid_scattered"]) for rule in rules) <= 0:
        return None
    bystander_pos = _place_bystanders(used_colors, inp, occupied, reserved, valid_output_cells)

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
