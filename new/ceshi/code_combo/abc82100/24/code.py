# Combo variant of abc82100 (index 24)
# combo_spec: D5/2+D15/1
# Name: 超界环绕+象限对齐

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg

# Design analysis for the combo variant
# concretization:
#   The learned legend is still a local source -> target pair with a nearby teal stencil,
#   but each scattered source now uses two coupled global effects at once.
#   First choose the stencil bbox corner from the source pixel's quadrant:
#   top-left -> bbox top-left, top-right -> bbox top-right, bottom-left -> bbox bottom-left,
#   bottom-right -> bbox bottom-right.
#   Then stamp the whole stencil so that chosen corner lands on the source, and wrap any
#   off-grid pixels modulo the grid size instead of clipping them away.
# behavior_trace:
#   Example on an 8x9 grid with a 6x4 stencil whose filled cells include all four bbox corners:
#   a top-left scattered source at (3, 1) uses the stencil's top-left corner as the reference.
#   Original task: anchor translation would use one fixed legend anchor and clip off-grid cells.
#   D5-only: the top-left corner would align to (3, 1), but out-of-bounds cells would vanish.
#   This combo: the same corner aligns to (3, 1), so the stencil extends downward far enough
#   to cross row 7; those cells reappear on row 0. A bottom-right source of the same color uses
#   the opposite bbox corner, so it produces a differently shifted wrapped copy of the same shape.
# complexity_calibration:
#   The fused rule is easy to state but physically easy to miss: quadrant-corner alignment tends
#   to push the stencil inward, which would suppress wrapping on large grids. To make the combo
#   observable, the generator uses a single rule, a compact 8xN or Nx8 grid, a stencil whose bbox
#   has one long dimension of length 6, and one scattered source in every quadrant. That keeps the
#   branch count at four visible cases while guaranteeing repeated wrapped examples.
# implicit_assumption_audit:
#   - Rule discovery still assumes an in-bounds 4-neighbor source/target legend and a teal component
#     touching the continued anchor cell. The generator preserves that exactly.
#   - The original stamping loop assumed one fixed anchor reference for every scattered source.
#     That is no longer true; the solver must recompute a quadrant-dependent reference corner per source.
#   - The original clipping mask encoded a hidden boundary rule. The combo replaces that with modulo
#     indexing, so the generator must keep bystanders off every wrapped footprint, not just off visible
#     in-bounds clipped cells.
#   - The combo introduces a new requirement that each selected bbox corner is actually occupied; the
#     generator therefore emits stencils whose four bbox corners are filled and whose chosen anchor is
#     a non-corner cell, preventing collapse back to anchor-based placement.
# placement_legality:
#   Legends are placed fully in bounds using the original collinear geometry. Scattered source pixels
#   remain 4-neighbor isolated to avoid accidental extra legends. Wrapped output overlap is allowed,
#   but with one rule and a few bystanders the scene stays readable.
# visual_observability:
#   Every instance shows the same color rule in all four quadrants, so the solver can compare how the
#   output object shifts when the input source moves between quadrants. At least one wrapped source is
#   forced, and in practice the mandatory quadrant sources all wrap along the long stencil dimension.
# cleanliness_and_inferability:
#   The palette still carries only explicit source -> target substitution. No numeric color ordering is
#   used. Restricting to one rule avoids clutter from multiple large legends in a small toroidal grid.


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


def _quadrant_key(r: int, c: int, h: int, w: int) -> str:
    top = 2 * int(r) < int(h)
    left = 2 * int(c) < int(w)
    if top and left:
        return "tl"
    if top and not left:
        return "tr"
    if not top and left:
        return "bl"
    return "br"


def _pattern_corner(pattern: np.ndarray, quadrant: str) -> tuple[int, int]:
    rows = pattern[:, 0]
    cols = pattern[:, 1]
    r0 = int(rows.min())
    r1 = int(rows.max())
    c0 = int(cols.min())
    c1 = int(cols.max())
    if quadrant == "tl":
        return r0, c0
    if quadrant == "tr":
        return r0, c1
    if quadrant == "bl":
        return r1, c0
    return r1, c1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn local color-change legends, then replay the teal stencil with quadrant-chosen corners and toroidal wrapping.

    Input structure:
    - The grid has a black background, isolated colored pixels, and one teal (8) connected component that acts as a stencil.
    - A rule example is a 4-neighbor source->target pair whose continued direction reaches an anchor touching that teal stencil.
    - Other scattered pixels reuse the source color and can appear in different global quadrants of the same grid.

    Rule:
    - Each rule example teaches a source color, a target color, and which teal component to copy.
    - For every non-legend source pixel, choose the stencil bbox corner from that pixel's quadrant:
      upper-left, upper-right, lower-left, or lower-right.
    - Translate the stencil so the chosen corner lands on the source pixel, recolor it to the target color,
      wrap any off-grid cells to the opposite edge, and erase the legend pixels plus all teal stencil pixels.

    Key insight:
    - The local legend still tells you which teal shape belongs to which color substitution, but placement is controlled by a
      global quadrant choice instead of the legend anchor, and the copied shape lives on a torus rather than being clipped.
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
        for sr, sc in map(tuple, np.argwhere(g == rule["source_color"])):
            sr = int(sr)
            sc = int(sc)
            if (sr, sc) in legend_pixels:
                continue
            corner = _pattern_corner(pattern, _quadrant_key(sr, sc, h, w))
            dr = sr - int(corner[0])
            dc = sc - int(corner[1])
            rows = (pattern[:, 0] + dr) % h
            cols = (pattern[:, 1] + dc) % w
            out[rows, cols] = rule["target_color"]

    return out


# --------------- generator (c) ---------------

GRID_SHORT_DIM = 8
GRID_LONG_DIM_OPTIONS = [8, 9, 10]

RULE_COUNT_WEIGHTS = [1]
NON_BG_NON_TEAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

STENCIL_MIN_PIXELS = 9
STENCIL_MAX_CAP = 12
STENCIL_EXTRA_PIXELS_MAX = 3

SCATTERED_MIN = 4
SCATTERED_MAX = 6
BYSTANDER_MAX = 3

LEGEND_RETRIES = 400
SCATTERED_EXTRA_RETRIES = 120
BYSTANDER_RETRIES = 80
OUTER_RETRIES = 500

_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
_QUADRANTS = ("tl", "tr", "bl", "br")


def _reserve_4neighbors(reserved: set[tuple[int, int]], cells) -> None:
    for r, c in cells:
        for dr, dc in _DIRS4:
            reserved.add((int(r) + dr, int(c) + dc))


def _king_path(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = start
    tr, tc = end
    path = [(int(r), int(c))]
    while (r, c) != (tr, tc):
        if r < tr:
            r += 1
        elif r > tr:
            r -= 1
        if c < tc:
            c += 1
        elif c > tc:
            c -= 1
        path.append((int(r), int(c)))
    return path


def _make_stencil(orientation: str):
    if orientation == "vertical":
        dim_options = [(6, 3), (6, 4), (6, 5)]
    else:
        dim_options = [(3, 6), (4, 6), (5, 6)]

    for _ in range(100):
        ph, pw = random.choice(dim_options)
        center = (random.randint(1, ph - 2), random.randint(1, pw - 2))
        corners = {(0, 0), (0, pw - 1), (ph - 1, 0), (ph - 1, pw - 1)}

        coords = set()
        for corner in corners:
            coords.update(_king_path(center, corner))
        if len(coords) > STENCIL_MAX_CAP:
            continue

        target = random.randint(
            max(STENCIL_MIN_PIXELS, len(coords)),
            min(STENCIL_MAX_CAP, ph * pw, len(coords) + STENCIL_EXTRA_PIXELS_MAX),
        )
        while len(coords) < target:
            frontier = set()
            for r, c in coords:
                for dr, dc in _DIRS8:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < ph and 0 <= nc < pw and (nr, nc) not in coords:
                        frontier.add((nr, nc))
            if not frontier:
                break
            coords.add(random.choice(sorted(frontier)))

        non_corner_cells = sorted(coords - corners)
        if len(coords) >= STENCIL_MIN_PIXELS and non_corner_cells:
            anchor = random.choice(non_corner_cells)
            return sorted(coords), anchor, (ph, pw)

    raise RuntimeError("failed to build combo stencil")


def _build_rule():
    colors = NON_BG_NON_TEAL_COLORS[:]
    random.shuffle(colors)
    orientation = random.choice(["vertical", "horizontal"])
    cells, anchor, size = _make_stencil(orientation)
    return {
        "src": colors[0],
        "tgt": colors[1],
        "cells": cells,
        "anchor": anchor,
        "size": size,
        "orientation": orientation,
        "n_sc": random.randint(SCATTERED_MIN, SCATTERED_MAX),
    }, {colors[0], colors[1]}


def _combo_origin(rule: dict, sr: int, sc: int, h: int, w: int) -> tuple[int, int]:
    ph, pw = rule["size"]
    quadrant = _quadrant_key(sr, sc, h, w)
    if quadrant == "tl":
        return sr, sc
    if quadrant == "tr":
        return sr, sc - (pw - 1)
    if quadrant == "bl":
        return sr - (ph - 1), sc
    return sr - (ph - 1), sc - (pw - 1)


def _combo_stamp_cells(
    rule: dict, sr: int, sc: int, h: int, w: int
) -> tuple[set[tuple[int, int]], bool, bool]:
    top, left = _combo_origin(rule, sr, sc, h, w)
    cells: set[tuple[int, int]] = set()
    wraps_row = False
    wraps_col = False
    for rr, cc in rule["cells"]:
        raw_r = int(top + rr)
        raw_c = int(left + cc)
        if not (0 <= raw_r < h):
            wraps_row = True
        if not (0 <= raw_c < w):
            wraps_col = True
        cells.add((raw_r % h, raw_c % w))
    return cells, wraps_row, wraps_col


def _combo_clipped_cells(rule: dict, sr: int, sc: int, h: int, w: int) -> set[tuple[int, int]]:
    top, left = _combo_origin(rule, sr, sc, h, w)
    cells: set[tuple[int, int]] = set()
    for rr, cc in rule["cells"]:
        nr = int(top + rr)
        nc = int(left + cc)
        if 0 <= nr < h and 0 <= nc < w:
            cells.add((nr, nc))
    return cells


def _anchor_wrap_cells(rule: dict, sr: int, sc: int, h: int, w: int) -> set[tuple[int, int]]:
    ar, ac = rule["anchor"]
    cells: set[tuple[int, int]] = set()
    for rr, cc in rule["cells"]:
        raw_r = int(sr + rr - ar)
        raw_c = int(sc + cc - ac)
        cells.add((raw_r % h, raw_c % w))
    return cells


def _quadrant_ranges(h: int, w: int) -> dict[str, tuple[range, range]]:
    r_split = (h + 1) // 2
    c_split = (w + 1) // 2
    return {
        "tl": (range(0, r_split), range(0, c_split)),
        "tr": (range(0, r_split), range(c_split, w)),
        "bl": (range(r_split, h), range(0, c_split)),
        "br": (range(r_split, h), range(c_split, w)),
    }


def _mandatory_row(rule: dict, quadrant: str, h: int) -> int:
    ph, _pw = rule["size"]
    if quadrant in ("tl", "tr"):
        return max(0, h - ph + 1)
    return min(h - 1, ph - 2)


def _mandatory_col(rule: dict, quadrant: str, w: int) -> int:
    _ph, pw = rule["size"]
    if quadrant in ("tl", "bl"):
        return max(0, w - pw + 1)
    return min(w - 1, pw - 2)


def _quadrant_candidates(rule: dict, h: int, w: int, quadrant: str, require_wrap: bool) -> list[tuple[int, int]]:
    row_range, col_range = _quadrant_ranges(h, w)[quadrant]
    candidates: list[tuple[int, int]] = []
    orientation = rule["orientation"]

    if orientation == "vertical":
        rows = [r for r in row_range if r == _mandatory_row(rule, quadrant, h)]
        cols = list(col_range)
    else:
        rows = list(row_range)
        cols = [c for c in col_range if c == _mandatory_col(rule, quadrant, w)]

    if not rows or not cols:
        return []

    random.shuffle(rows)
    random.shuffle(cols)
    for r in rows:
        for c in cols:
            cells, wraps_row, wraps_col = _combo_stamp_cells(rule, int(r), int(c), h, w)
            if require_wrap and not (wraps_row or wraps_col):
                continue
            if not require_wrap and (wraps_row or wraps_col):
                continue
            if cells:
                candidates.append((int(r), int(c)))
    random.shuffle(candidates)
    return candidates


def _place_legends(rules, inp, occupied, reserved):
    h, w = inp.shape
    for rule in rules:
        cells = rule["cells"]
        ar_local, ac_local = rule["anchor"]
        placed = False
        for _ in range(LEGEND_RETRIES):
            dr, dc = random.choice(_DIRS4)
            sr, sc = random.randint(0, h - 1), random.randint(0, w - 1)
            tr, tc = sr + dr, sc + dc
            ar, ac = sr + 2 * dr, sc + 2 * dc
            if not (0 <= tr < h and 0 <= tc < w and 0 <= ar < h and 0 <= ac < w):
                continue

            top = ar - ar_local
            left = ac - ac_local
            teal_cells = {(top + rr, left + cc) for rr, cc in cells}
            if any(not (0 <= nr < h and 0 <= nc < w) for nr, nc in teal_cells):
                continue

            all_cells = {(sr, sc), (tr, tc), (ar, ac)} | teal_cells
            if all_cells & (occupied | reserved):
                continue

            inp[sr, sc] = rule["src"]
            inp[tr, tc] = rule["tgt"]
            for nr, nc in teal_cells:
                inp[nr, nc] = 8
            rule["legend"] = {"source": (sr, sc), "target": (tr, tc), "anchor_abs": (ar, ac)}
            occupied.update(all_cells)
            _reserve_4neighbors(reserved, all_cells)
            placed = True
            break
        if not placed:
            return False
    return True


def _place_scattered(rules, inp, occupied, reserved):
    h, w = inp.shape
    for rule in rules:
        positions: list[tuple[int, int]] = []

        for quadrant in _QUADRANTS:
            candidates = _quadrant_candidates(rule, h, w, quadrant, require_wrap=True)
            placed = False
            for r, c in candidates:
                if (r, c) in occupied or (r, c) in reserved:
                    continue
                positions.append((r, c))
                occupied.add((r, c))
                _reserve_4neighbors(reserved, [(r, c)])
                placed = True
                break
            if not placed:
                return False

        extra_candidates = []
        for quadrant in _QUADRANTS:
            extra_candidates.extend(_quadrant_candidates(rule, h, w, quadrant, require_wrap=False))
            extra_candidates.extend(_quadrant_candidates(rule, h, w, quadrant, require_wrap=True))
        random.shuffle(extra_candidates)

        for r, c in extra_candidates[:SCATTERED_EXTRA_RETRIES]:
            if len(positions) >= rule["n_sc"]:
                break
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


def _all_output_cells(rules, h: int, w: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for rule in rules:
        for sr, sc in rule["scattered"]:
            stamp_cells, _wraps_row, _wraps_col = _combo_stamp_cells(rule, int(sr), int(sc), h, w)
            cells.update(stamp_cells)
    return cells


def _place_bystanders(used_colors, inp, occupied, reserved, forbidden_output_cells):
    bystander_colors = [c for c in NON_BG_NON_TEAL_COLORS if c not in used_colors]
    if not bystander_colors:
        return []
    h, w = inp.shape
    n = random.randint(0, BYSTANDER_MAX)
    placed = []
    for _ in range(n):
        for _ in range(BYSTANDER_RETRIES):
            r, c = random.randint(0, h - 1), random.randint(0, w - 1)
            if (r, c) in occupied or (r, c) in reserved or (r, c) in forbidden_output_cells:
                continue
            bc = random.choice(bystander_colors)
            inp[r, c] = bc
            placed.append((r, c, bc))
            occupied.add((r, c))
            _reserve_4neighbors(reserved, [(r, c)])
            break
    return placed


def _build_combo_output(rules, bystanders, h: int, w: int) -> np.ndarray:
    out = np.zeros((h, w), dtype=int)
    for r, c, bc in bystanders:
        out[int(r), int(c)] = int(bc)
    for rule in rules:
        for sr, sc in rule["scattered"]:
            cells, _wraps_row, _wraps_col = _combo_stamp_cells(rule, int(sr), int(sc), h, w)
            for nr, nc in cells:
                out[nr, nc] = int(rule["tgt"])
    return out


def _build_anchor_wrap_output(rules, bystanders, h: int, w: int) -> np.ndarray:
    out = np.zeros((h, w), dtype=int)
    for r, c, bc in bystanders:
        out[int(r), int(c)] = int(bc)
    for rule in rules:
        for sr, sc in rule["scattered"]:
            for nr, nc in _anchor_wrap_cells(rule, int(sr), int(sc), h, w):
                out[nr, nc] = int(rule["tgt"])
    return out


def _build_combo_clipped_output(rules, bystanders, h: int, w: int) -> np.ndarray:
    out = np.zeros((h, w), dtype=int)
    for r, c, bc in bystanders:
        out[int(r), int(c)] = int(bc)
    for rule in rules:
        for sr, sc in rule["scattered"]:
            for nr, nc in _combo_clipped_cells(rule, int(sr), int(sc), h, w):
                out[nr, nc] = int(rule["tgt"])
    return out


def _combo_metrics(rules, h: int, w: int) -> dict[str, int]:
    metrics = {
        "sources": 0,
        "wrapped_sources": 0,
        "row_wrap_sources": 0,
        "col_wrap_sources": 0,
        "anchor_diff_sources": 0,
        "clipped_diff_sources": 0,
        "quadrant_sources": 0,
    }
    for rule in rules:
        quadrants = {_quadrant_key(int(sr), int(sc), h, w) for sr, sc in rule["scattered"]}
        metrics["quadrant_sources"] += len(quadrants)
        for sr, sc in rule["scattered"]:
            combo_cells, wraps_row, wraps_col = _combo_stamp_cells(rule, int(sr), int(sc), h, w)
            anchor_cells = _anchor_wrap_cells(rule, int(sr), int(sc), h, w)
            clipped_cells = _combo_clipped_cells(rule, int(sr), int(sc), h, w)
            metrics["sources"] += 1
            if wraps_row or wraps_col:
                metrics["wrapped_sources"] += 1
            if wraps_row:
                metrics["row_wrap_sources"] += 1
            if wraps_col:
                metrics["col_wrap_sources"] += 1
            if combo_cells != anchor_cells:
                metrics["anchor_diff_sources"] += 1
            if combo_cells != clipped_cells:
                metrics["clipped_diff_sources"] += 1
    return metrics


def _gen_attempt():
    num_rules = random.choice(RULE_COUNT_WEIGHTS)
    rules = []
    used_colors: set[int] = set()
    for _ in range(num_rules):
        rule, used = _build_rule()
        rules.append(rule)
        used_colors.update(used)

    if any(rule["orientation"] == "vertical" for rule in rules):
        h = GRID_SHORT_DIM
        w = random.choice(GRID_LONG_DIM_OPTIONS)
    else:
        h = random.choice(GRID_LONG_DIM_OPTIONS)
        w = GRID_SHORT_DIM

    occupied: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()
    inp = np.zeros((h, w), dtype=int)

    if not _place_legends(rules, inp, occupied, reserved):
        return None
    if not _place_scattered(rules, inp, occupied, reserved):
        return None

    output_cells = _all_output_cells(rules, h, w)
    bystanders = _place_bystanders(used_colors, inp, occupied, reserved, output_cells)

    combo_output = _build_combo_output(rules, bystanders, h, w)
    anchor_wrap_output = _build_anchor_wrap_output(rules, bystanders, h, w)
    combo_clipped_output = _build_combo_clipped_output(rules, bystanders, h, w)
    metrics = _combo_metrics(rules, h, w)

    if metrics["quadrant_sources"] < 4 * len(rules):
        return None
    if metrics["wrapped_sources"] < len(rules):
        return None
    if metrics["anchor_diff_sources"] < len(rules):
        return None
    if metrics["clipped_diff_sources"] < len(rules):
        return None
    if np.array_equal(combo_output, anchor_wrap_output):
        return None
    if np.array_equal(combo_output, combo_clipped_output):
        return None

    try:
        if np.array_equal(main(inp), combo_output):
            return inp
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid combo input after {OUTER_RETRIES} attempts")
