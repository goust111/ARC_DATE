# Combo variant of abc82100 (index 47)
# combo_spec: D3/2+D15/0
# Name: 跳过超界源像素+边缘源像素

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg

# Design analysis
# a) Concrete behavior and trace:
#    The base task learns each rule from an interior legend: a source-colored pixel, an
#    adjacent target-colored pixel, and a teal stencil anchored one step farther in the
#    same direction. In the original task, every other pixel of that source color can act
#    as an instance anywhere on the board, and out-of-bounds stencil cells are clipped.
#    In this combo, only non-legend source pixels on the outer frame are actionable, and
#    an edge source is processed only if the whole translated stencil stays in bounds.
#    Example: suppose a learned relative stencil is {(0, 0), (0, 1), (1, 0)}. A top-edge
#    source at (0, 3) is valid and paints those three cells in the target color. A top
#    edge source at (0, 4) is invalid because (0, 5) would leave the grid, so the source
#    simply disappears in the output instead of producing a clipped fragment.
# b) Complexity calibration:
#    The generator keeps the original 1-4 rules, unique source->target color pairs, and
#    small 8-connected teal stencils, but now every scattered source lies on the border.
#    Each rule is forced to have both valid edge sources and invalid edge sources so the
#    learner must combine edge detection with boundary-fit checking rather than seeing a
#    degenerate all-valid or all-invalid sample.
# c) Audit of original assumptions:
#    - The original `main()` treated every non-legend source-colored pixel as actionable;
#      that is no longer correct because interior source-colored pixels should not trigger
#      stamping in this variant. The solver now explicitly filters to border cells.
#    - The original `ok = ...` clipping write assumed partial stencils were legal. That
#      assumption is broken by D15/0, so the solver now requires the full translated
#      stencil to fit or skips the source entirely.
#    - Preserving non-source bystanders still works because the generator never reuses a
#      learned source color for bystanders, and bystanders are kept away from the border
#      for clarity.
# d) Placement legality:
#    Legends are placed strictly in the interior so the "edge sources" cue is clean.
#    Scattered source pixels are placed only on border cells, remain 4-neighbor isolated,
#    and never overlap legends or each other. Valid stamps are also kept away from other
#    valid output footprints and previously placed source pixels so the output stays legible.
# e) Visual observability:
#    Across samples, the border contains the actionable source pixels while the legend sits
#    inside the board. Some edge sources generate the full target-colored stencil and some
#    vanish, making the fit-vs-overflow distinction visible from the geometry alone.
# f) Visual cleanliness:
#    Legend regions, edge sources, valid output stamps, and bystanders are all separated by
#    reservation. Bystanders are interior-only and avoid output footprints so preserved
#    pixels do not masquerade as failed edge sources.
# g) Color inference:
#    Colors still follow the base task: the adjacent legend pair teaches which source color
#    maps to which target color, and the nearby teal component is the stencil. The new
#    decision point is purely spatial: actionable instances are on the border, and only
#    those whose full stencil fits are realized.
# h) Variant metric used in testing:
#    Count border source pixels whose translated stencil is invalid. Samples should have a
#    positive invalid count and should differ from a clipped-stamping baseline on many
#    generated boards; otherwise D15/0 has visually degenerated.
# i) New solver assumptions and generator guarantees:
#    - The solver assumes non-legend actionable instances are border cells only.
#      The generator therefore places every scattered source on the outer frame.
#    - The solver assumes a skipped source leaves no residue at all.
#      The generator builds outputs from bystanders plus valid edge stamps only.
#    - The solver assumes legends are discoverable without interference from scattered
#      pixels. The generator enforces 4-neighbor isolation and interior legend placement.


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
RULE_BUILD_RETRIES = 80

# Scattered source pixels per rule
SCATTERED_MIN = 3
SCATTERED_MAX_CAP = 8
EXTRA_VALID_BIAS = 0.6

# Bystander pixels
BYSTANDER_MAX = 5

# Placement retry budgets
LEGEND_RETRIES = 500
OUTER_RETRIES = 300

# Cardinal directions (4-connected)
_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
# 8-connected neighbor offsets
_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _anchor_distance(pattern: np.ndarray, anchor: tuple[int, int]) -> int:
    return min(_chebyshev((int(r), int(c)), anchor) for r, c in pattern)


def _is_edge_cell(r: int, c: int, H: int, W: int) -> bool:
    return int(r) in (0, H - 1) or int(c) in (0, W - 1)


def _reserve_4neighbors(reserved: set[tuple[int, int]], cells) -> None:
    for r, c in cells:
        for dr, dc in _DIRS4:
            reserved.add((int(r) + dr, int(c) + dc))


def _make_stencil(min_pixels: int, max_pixels: int) -> list[tuple[int, int]]:
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
    return sorted(coords)


def _fit_sides(rel: list[tuple[int, int]]) -> list[str]:
    rows = [int(rr) for rr, _ in rel]
    cols = [int(cc) for _, cc in rel]
    sides: list[str] = []
    if min(rows) >= 0:
        sides.append("top")
    if max(rows) <= 0:
        sides.append("bottom")
    if min(cols) >= 0:
        sides.append("left")
    if max(cols) <= 0:
        sides.append("right")
    return sides


def _fits_rel(rel: list[tuple[int, int]], sr: int, sc: int, H: int, W: int) -> bool:
    for rr, cc in rel:
        nr = int(sr) + int(rr)
        nc = int(sc) + int(cc)
        if not (0 <= nr < H and 0 <= nc < W):
            return False
    return True


def _stamp_cells(rel: list[tuple[int, int]], sr: int, sc: int) -> set[tuple[int, int]]:
    return {(int(sr) + int(rr), int(sc) + int(cc)) for rr, cc in rel}


def _side_positions(side: str, H: int, W: int) -> list[tuple[int, int]]:
    if side == "top":
        return [(0, c) for c in range(W)]
    if side == "bottom":
        return [(H - 1, c) for c in range(W)]
    if side == "left":
        return [(r, 0) for r in range(H)]
    return [(r, W - 1) for r in range(H)]


def _edge_positions(H: int, W: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    cells.extend((0, c) for c in range(W))
    cells.extend((H - 1, c) for c in range(W) if H > 1)
    cells.extend((r, 0) for r in range(1, H - 1))
    cells.extend((r, W - 1) for r in range(1, H - 1) if W > 1)
    return cells


def _edge_margin(r: int, c: int, H: int, W: int) -> int:
    if int(r) in (0, H - 1):
        return min(int(c), W - 1 - int(c))
    return min(int(r), H - 1 - int(r))


def _pick_ranked(candidates: list[tuple[int, int]], H: int, W: int, prefer_large_margin: bool):
    if not candidates:
        return None
    ranked = sorted(
        candidates,
        key=lambda pos: (_edge_margin(pos[0], pos[1], H, W), pos[0], pos[1]),
        reverse=prefer_large_margin,
    )
    window = min(len(ranked), 6)
    return random.choice(ranked[:window])


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
    Learn interior source-target-stencil examples and apply them only from the border.

    Input structure:
    - The board has a black background, interior rule legends, teal (8) stencil
      components, and scattered source-colored pixels that appear only on the outer edge.
    - Each legend is a 4-neighbor source->target pair, with the anchor one more step in
      the same direction touching that rule's teal stencil.

    Rule:
    - For each learned rule, inspect every non-legend pixel of the source color that lies
      on the grid border.
    - Translate the teal stencil so the rule anchor lands on that border source pixel and
      recolor the whole stencil to the learned target color, but only if every translated
      stencil cell stays in bounds. Otherwise skip that source completely.
    - Remove all legend pixels and all teal stencil pixels from the output; preserve only
      unrelated bystanders.

    Key insight:
    - The legend still teaches the color substitution and stencil shape, but the actual
      instances are restricted to border pixels, and the decisive question is whether the
      full translated stencil fits inside the board from that edge position.
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
        for sr, sc in map(tuple, np.argwhere(g == rule["source_color"])):
            sr = int(sr)
            sc = int(sc)
            if (sr, sc) in legend_pixels or not _is_edge_cell(sr, sc, h, w):
                continue
            dr = sr - ar
            dc = sc - ac
            rows = pattern[:, 0] + dr
            cols = pattern[:, 1] + dc
            if not ((0 <= rows).all() and (rows < h).all() and (0 <= cols).all() and (cols < w).all()):
                continue
            out[rows, cols] = rule["target_color"]

    return out


def _build_rules(num_rules: int, area: int):
    colors = NON_BG_NON_TEAL_COLORS[:]
    random.shuffle(colors)

    max_s = min(STENCIL_MAX_CAP, max(2, area // STENCIL_MAX_AREA_DIV))
    min_s = min(max(2, area // STENCIL_MIN_AREA_DIV), max_s)

    rules: list[dict] = []
    used: set[int] = set()
    for i in range(num_rules):
        src_c, tgt_c = colors[2 * i], colors[2 * i + 1]
        used.update((src_c, tgt_c))
        chosen = None
        for _ in range(RULE_BUILD_RETRIES):
            stencil = _make_stencil(min_s, max_s)
            candidates = []
            for anchor in stencil:
                rel = [(r - anchor[0], c - anchor[1]) for r, c in stencil]
                sides = _fit_sides(rel)
                if sides:
                    candidates.append((len(sides), rel, sides))
            if candidates:
                best = max(count for count, _rel, _sides in candidates)
                _count, rel, sides = random.choice([item for item in candidates if item[0] == best])
                chosen = {"src": src_c, "tgt": tgt_c, "rel": rel, "valid_sides": sides}
                break
        if chosen is None:
            return None, None
        rules.append(chosen)

    return rules, used


def _place_legends(rules, inp, occupied, reserved):
    H, W = inp.shape
    legend_cells: set[tuple[int, int]] = set()

    for rule in rules:
        rel = rule["rel"]
        placed = False
        for _ in range(LEGEND_RETRIES):
            dr, dc = random.choice(_DIRS4)
            sr, sc = random.randint(1, H - 2), random.randint(1, W - 2)
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
            if any(_is_edge_cell(r, c, H, W) for r, c in all_cells):
                continue
            if all_cells & (occupied | reserved):
                continue
            inp[sr, sc] = rule["src"]
            inp[tr, tc] = rule["tgt"]
            for nr, nc in teal_cells:
                inp[nr, nc] = 8
            occupied.update(all_cells)
            _reserve_4neighbors(reserved, all_cells)
            legend_cells.update(all_cells)
            placed = True
            break
        if not placed:
            return None

    return legend_cells


def _valid_side_candidates(rule, H, W, occupied, reserved, blocked_output, source_cells, valid_output_cells):
    rel = rule["rel"]
    side_to_candidates: dict[str, list[tuple[int, int]]] = {}
    for side in rule["valid_sides"]:
        side_candidates: list[tuple[int, int]] = []
        for r, c in _side_positions(side, H, W):
            if (r, c) in occupied or (r, c) in reserved:
                continue
            stamp = _stamp_cells(rel, r, c)
            if stamp & blocked_output or stamp & valid_output_cells or stamp & source_cells:
                continue
            if _fits_rel(rel, r, c, H, W):
                side_candidates.append((r, c))
        if side_candidates:
            side_to_candidates[side] = side_candidates
    return side_to_candidates


def _invalid_edge_candidates(rel, H, W, occupied, reserved, valid_output_cells):
    candidates: list[tuple[int, int]] = []
    for r, c in _edge_positions(H, W):
        if (r, c) in occupied or (r, c) in reserved or (r, c) in valid_output_cells:
            continue
        if not _fits_rel(rel, r, c, H, W):
            candidates.append((r, c))
    return candidates


def _place_scattered(rules, inp, occupied, reserved, blocked_output):
    H, W = inp.shape
    valid_output_cells: set[tuple[int, int]] = set()
    source_cells: set[tuple[int, int]] = set()

    for rule in rules:
        rel = rule["rel"]
        valid_positions: list[tuple[int, int]] = []
        invalid_positions: list[tuple[int, int]] = []
        all_positions: list[tuple[int, int]] = []

        side_to_candidates = _valid_side_candidates(
            rule, H, W, occupied, reserved, blocked_output, source_cells, valid_output_cells
        )
        if not side_to_candidates:
            return None
        best_count = max(len(cands) for cands in side_to_candidates.values())
        chosen_side = random.choice([side for side, cands in side_to_candidates.items() if len(cands) == best_count])
        pos = _pick_ranked(side_to_candidates[chosen_side], H, W, prefer_large_margin=True)
        if pos is None:
            return None
        vr, vc = pos
        stamp = _stamp_cells(rel, vr, vc)
        valid_positions.append(pos)
        all_positions.append(pos)
        source_cells.add(pos)
        valid_output_cells.update(stamp)
        occupied.add(pos)
        _reserve_4neighbors(reserved, [pos])
        inp[vr, vc] = rule["src"]

        invalid_candidates = _invalid_edge_candidates(rel, H, W, occupied, reserved, valid_output_cells)
        pos = _pick_ranked(invalid_candidates, H, W, prefer_large_margin=False)
        if pos is None:
            return None
        ir, ic = pos
        invalid_positions.append(pos)
        all_positions.append(pos)
        source_cells.add(pos)
        occupied.add(pos)
        _reserve_4neighbors(reserved, [pos])
        inp[ir, ic] = rule["src"]

        target_total = random.randint(SCATTERED_MIN, SCATTERED_MAX_CAP)
        while len(all_positions) < target_total:
            wants_valid = random.random() < EXTRA_VALID_BIAS
            if wants_valid:
                side_to_candidates = _valid_side_candidates(
                    rule, H, W, occupied, reserved, blocked_output, source_cells, valid_output_cells
                )
                candidates = side_to_candidates.get(chosen_side, [])
                if not candidates:
                    remaining = [pos for vals in side_to_candidates.values() for pos in vals]
                    candidates = remaining
                pos = _pick_ranked(candidates, H, W, prefer_large_margin=True)
                if pos is not None:
                    r, c = pos
                    stamp = _stamp_cells(rel, r, c)
                    valid_positions.append(pos)
                    all_positions.append(pos)
                    source_cells.add(pos)
                    valid_output_cells.update(stamp)
                    occupied.add(pos)
                    _reserve_4neighbors(reserved, [pos])
                    inp[r, c] = rule["src"]
                    continue
            invalid_candidates = _invalid_edge_candidates(rel, H, W, occupied, reserved, valid_output_cells)
            pos = _pick_ranked(invalid_candidates, H, W, prefer_large_margin=False)
            if pos is not None:
                r, c = pos
                invalid_positions.append(pos)
                all_positions.append(pos)
                source_cells.add(pos)
                occupied.add(pos)
                _reserve_4neighbors(reserved, [pos])
                inp[r, c] = rule["src"]
                continue
            side_to_candidates = _valid_side_candidates(
                rule, H, W, occupied, reserved, blocked_output, source_cells, valid_output_cells
            )
            remaining = [pos for vals in side_to_candidates.values() for pos in vals]
            pos = _pick_ranked(remaining, H, W, prefer_large_margin=True)
            if pos is None:
                break
            r, c = pos
            stamp = _stamp_cells(rel, r, c)
            valid_positions.append(pos)
            all_positions.append(pos)
            source_cells.add(pos)
            valid_output_cells.update(stamp)
            occupied.add(pos)
            _reserve_4neighbors(reserved, [pos])
            inp[r, c] = rule["src"]

        if len(all_positions) < SCATTERED_MIN or not valid_positions or not invalid_positions:
            return None

        rule["valid_scattered"] = valid_positions
        rule["invalid_scattered"] = invalid_positions
        rule["scattered"] = all_positions
        rule["chosen_side"] = chosen_side

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
            for r in range(1, H - 1)
            for c in range(1, W - 1)
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
        for sr, sc in rule["valid_scattered"]:
            for rr, cc in rule["rel"]:
                nr, nc = sr + rr, sc + cc
                output[nr, nc] = rule["tgt"]
    return output


def _build_clipped_baseline(rules, bystander_pos, H, W):
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


def _gen_attempt():
    num_rules = random.choice(RULE_COUNT_WEIGHTS)
    min_dim = MIN_DIM_BY_RULES[num_rules]
    H = random.randint(min_dim, GRID_DIM_MAX)
    W = random.randint(min_dim, GRID_DIM_MAX)

    rules, used_colors = _build_rules(num_rules, H * W)
    if rules is None:
        return None

    occupied: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()
    inp = np.zeros((H, W), dtype=int)

    legend_cells = _place_legends(rules, inp, occupied, reserved)
    if legend_cells is None:
        return None
    valid_output_cells = _place_scattered(rules, inp, occupied, reserved, legend_cells)
    if valid_output_cells is None:
        return None

    total_valid = sum(len(rule["valid_scattered"]) for rule in rules)
    total_invalid = sum(len(rule["invalid_scattered"]) for rule in rules)
    if total_valid <= 0 or total_invalid <= 0:
        return None

    blocked_output = legend_cells | valid_output_cells
    bystander_pos = _place_bystanders(used_colors, inp, occupied, reserved, blocked_output)

    output = _build_output(rules, bystander_pos, H, W)
    clipped = _build_clipped_baseline(rules, bystander_pos, H, W)
    if np.array_equal(output, clipped):
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
