# Variant of abc82100
# Dimension: D8 — 输出网格形状 (grid_shape)
# Change: 输出网格尺寸与输入相同（same）... → 输出仅保留变换区域（排除规则区域）

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg


# Design analysis for the D8 crop variant
# a) Concrete behavior and trace:
#    The input still mixes two roles: a compact legend area with local
#    source->target->teal examples, and a separate work area with scattered source
#    pixels. The solver first performs the original full-grid substitution:
#    learn each color rule from its local example, stamp the teal stencil on every
#    non-legend source pixel, and erase the examples. The D8 change happens after
#    that: instead of returning the full canvas with a blank legend band, crop the
#    intermediate output to the tight bounding box of the non-background result.
#    Example trace: in the old task, a 16x18 input with legends in the top strip
#    produced a 16x18 output whose top strip became all black. In the variant, the
#    same intermediate result is cropped down to the lower occupied region, so the
#    output might become 8x11.
# b) Complexity calibration:
#    The variant keeps 1-4 rules, repeated scattered source colors, and distinct
#    teal stencils. The generator only changes layout, not the rule semantics.
#    Legends are grouped in one top band while all transformed content lives in one
#    work box; this makes the crop visually obvious without introducing extra
#    symbolic machinery or many subcases.
# c) Hidden assumptions in main():
#    - Rule discovery assumes teal components encode instructions. The generator
#      therefore keeps every teal component inside the legend band and places no
#      teal in the work area, so scattered pixels cannot create spurious rules.
#    - Preserving ordinary colored pixels assumes that bystanders are the only
#      non-rule colored cells that should survive. The generator restricts all
#      bystanders to the work box so the crop is not contaminated by stray colors.
#    - Stamping every pixel of a learned source color assumes legend source pixels
#      can be filtered by exact coordinates. The generator stores one explicit
#      example pair per rule and isolates those cells from scattered sources.
#    - The new crop step assumes every intended output pixel is non-background and
#      every non-background pixel after stamping belongs to the result region. The
#      generator enforces non-empty stamped output and avoids output overlaps with
#      bystanders that would create ambiguity.
# d) Placement legality:
#    Legend examples are placed only inside a dedicated top strip, with enough room
#    for the source-target pair and the full teal stencil. Scattered source pixels
#    are placed only where the translated stencil stays entirely inside the work
#    box. Output stamps do not overlap each other or bystanders, and source pixels
#    are kept 4-neighbor-separated for cleaner inputs.
# e) Visual observability:
#    Across examples, the human sees a clear "instruction strip" above and a
#    cluster of source pixels below. The output removes the strip entirely and
#    keeps only the transformed lower region. This is observable in 2-3 examples
#    without relying on color-id ordering or hidden numeric conventions.
# f) Visual cleanliness:
#    The work box keeps all surviving pixels compact, while the legend strip is
#    separated by empty rows. Non-overlapping stamps prevent the cropped outputs
#    from becoming unreadable blobs.
# g) Color inference:
#    Color changes remain locally explained by the source-target pair beside each
#    teal stencil. There are only a few rule colors per puzzle, each repeated by
#    several scattered source pixels, so the mapping remains learnable.
# Main()/generator assumption audit:
#    The solver now assumes the post-transform result is bounded away from the
#    input border often enough for cropping to matter; the generator guarantees
#    that by using an interior work box and by rejecting cases whose cropped
#    output accidentally matches the full input shape.


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


def _crop_to_non_bg(grid: np.ndarray, bg: int) -> np.ndarray:
    coords = np.argwhere(grid != int(bg))
    if coords.size == 0:
        return np.full((1, 1), int(bg), dtype=int)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return np.asarray(grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1], dtype=int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn local source-target-stencil examples, stamp them onto matching source pixels,
    then return only the occupied result region.

    Input structure:
    - Black background with a top instruction strip of colored source->target pairs
      and nearby teal (8) stencil components.
    - Elsewhere, isolated scattered source pixels repeat the legend source colors,
      and optional bystander pixels use unrelated colors.

    Rule:
    - For each legend pair, copy its associated teal stencil onto every non-legend
      pixel of the same source color, recoloring that stencil to the target color.
    - Remove the legend pairs and all teal instruction pixels, preserve unrelated
      bystanders, then crop away the empty border and blank legend strip.

    Key insight: The legend still teaches a color substitution plus a stencil anchor;
    the new twist is that the true answer is the tight crop of the transformed work area,
    not the original full canvas with empty instruction space.
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
            if (sr, sc) in legend_pixels:
                continue
            dr = sr - ar
            dc = sc - ac
            rows = pattern[:, 0] + dr
            cols = pattern[:, 1] + dc
            ok = (0 <= rows) & (rows < h) & (0 <= cols) & (cols < w)
            out[rows[ok], cols[ok]] = rule["target_color"]

    return _crop_to_non_bg(out, bg=bg)


# --------------- generator (D8 crop variant) ---------------

GRID_DIM_MAX = 20
NON_BG_NON_TEAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]
RULE_COUNT_WEIGHTS = [1, 1, 2, 2, 3, 3, 4]
MIN_HEIGHT_BY_RULES = {1: 12, 2: 13, 3: 15, 4: 16}
MIN_WIDTH_BY_RULES = {1: 10, 2: 13, 3: 16, 4: 18}

STENCIL_MIN_PIXELS = 2
STENCIL_MAX_PIXELS = 7
STENCIL_SPAN_MAX = 4

SCATTERED_MIN = 3
SCATTERED_MAX_CAP = 6
BYSTANDER_MAX = 4

LEGEND_GAP = 2
EDGE_MARGIN = 1
LEGEND_RETRIES = 500
SCATTERED_RETRIES_PER_PIXEL = 80
BYSTANDER_RETRIES = 80
OUTER_RETRIES = 400

_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _reserve_4neighbors(reserved: set[tuple[int, int]], cells: set[tuple[int, int]] | list[tuple[int, int]]) -> None:
    for r, c in cells:
        for dr, dc in _DIRS4:
            reserved.add((int(r + dr), int(c + dc)))


def _reserve_chebyshev(
    reserved: set[tuple[int, int]], cells: set[tuple[int, int]] | list[tuple[int, int]], radius: int
) -> None:
    for r, c in cells:
        for dr in range(-int(radius), int(radius) + 1):
            for dc in range(-int(radius), int(radius) + 1):
                reserved.add((int(r + dr), int(c + dc)))


def _rel_bounds(rel: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [int(r) for r, _ in rel]
    cols = [int(c) for _, c in rel]
    return min(rows), max(rows), min(cols), max(cols)


def _make_stencil() -> tuple[list[tuple[int, int]], tuple[int, int]]:
    for _ in range(200):
        n = random.randint(STENCIL_MIN_PIXELS, STENCIL_MAX_PIXELS)
        coords = {(0, 0)}
        for _ in range(n - 1):
            frontier = set()
            for r, c in coords:
                for dr, dc in _DIRS8:
                    pos = (int(r + dr), int(c + dc))
                    if pos not in coords:
                        frontier.add(pos)
            coords.add(random.choice(list(frontier)))
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        if max(rows) - min(rows) + 1 > STENCIL_SPAN_MAX:
            continue
        if max(cols) - min(cols) + 1 > STENCIL_SPAN_MAX:
            continue
        coords_list = sorted(coords)
        return coords_list, random.choice(coords_list)
    raise RuntimeError("Failed to sample bounded stencil")


def _build_rules(num_rules: int) -> tuple[list[dict], set[int]]:
    colors = NON_BG_NON_TEAL_COLORS[:]
    random.shuffle(colors)
    rules: list[dict] = []
    used_colors: set[int] = set()
    for i in range(num_rules):
        src_c = int(colors[2 * i])
        tgt_c = int(colors[2 * i + 1])
        stencil, anchor = _make_stencil()
        rel = [(int(r - anchor[0]), int(c - anchor[1])) for r, c in stencil]
        min_r, max_r, min_c, max_c = _rel_bounds(rel)
        rules.append(
            {
                "src": src_c,
                "tgt": tgt_c,
                "rel": rel,
                "bounds": (min_r, max_r, min_c, max_c),
            }
        )
        used_colors.update((src_c, tgt_c))
    return rules, used_colors


def _choose_grid(num_rules: int, rules: list[dict]) -> tuple[int, int, int]:
    max_pattern_h = max(int(rule["bounds"][1] - rule["bounds"][0] + 1) for rule in rules)
    legend_h = max(6, max_pattern_h + 2)
    min_h = max(MIN_HEIGHT_BY_RULES[num_rules], legend_h + LEGEND_GAP + 5 + EDGE_MARGIN)
    min_w = MIN_WIDTH_BY_RULES[num_rules]
    if min_h > GRID_DIM_MAX or min_w > GRID_DIM_MAX:
        raise RuntimeError("Grid bounds incompatible with variant layout")
    h = random.randint(min_h, GRID_DIM_MAX)
    w = random.randint(min_w, GRID_DIM_MAX)
    return h, w, legend_h


def _assign_scattered_counts(rules: list[dict], result_area: int) -> None:
    soft_max = min(SCATTERED_MAX_CAP, max(SCATTERED_MIN, result_area // 18))
    for rule in rules:
        rule["n_sc"] = random.randint(SCATTERED_MIN, soft_max)


def _choose_result_box(h: int, w: int, legend_h: int, rules: list[dict]) -> tuple[int, int, int, int] | None:
    start_row = legend_h + LEGEND_GAP
    max_stamp_h = max(int(rule["bounds"][1] - rule["bounds"][0] + 1) for rule in rules)
    max_stamp_w = max(int(rule["bounds"][3] - rule["bounds"][2] + 1) for rule in rules)

    max_result_h = h - start_row - EDGE_MARGIN
    max_result_w = w - 2 * EDGE_MARGIN
    min_result_h = max(5, max_stamp_h + 1)
    min_result_w = max(6, max_stamp_w + 2)
    if max_result_h < min_result_h or max_result_w < min_result_w:
        return None

    result_h = random.randint(min_result_h, max_result_h)
    result_w = random.randint(min_result_w, max_result_w)
    slack_down = h - EDGE_MARGIN - result_h - start_row
    top = start_row + random.randint(0, max(0, slack_down))
    left = random.randint(EDGE_MARGIN, w - EDGE_MARGIN - result_w)
    return int(top), int(left), int(top + result_h - 1), int(left + result_w - 1)


def _place_legends_top(rules: list[dict], inp: np.ndarray, legend_h: int) -> bool:
    _h, w = inp.shape
    occupied: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()

    for rule in rules:
        rel = rule["rel"]
        min_r, max_r, min_c, max_c = rule["bounds"]
        row_lo = max(0, -min_r)
        row_hi = legend_h - 1 - max_r
        col_lo = max(0, -2 - min_c)
        col_hi = min(w - 2, w - 3 - max_c)
        if row_lo > row_hi or col_lo > col_hi:
            return False

        placed = False
        for _ in range(LEGEND_RETRIES):
            sr = random.randint(row_lo, row_hi)
            sc = random.randint(col_lo, col_hi)
            teal_cells = {(int(sr + rr), int(sc + 2 + cc)) for rr, cc in rel}
            if (int(sr), int(sc)) in teal_cells or (int(sr), int(sc + 1)) in teal_cells:
                continue
            reverse_anchor = (int(sr), int(sc - 1))
            if min(_chebyshev(cell, reverse_anchor) for cell in teal_cells) <= 1:
                continue
            all_cells = {(int(sr), int(sc)), (int(sr), int(sc + 1))} | teal_cells
            if all_cells & (occupied | reserved):
                continue
            inp[int(sr), int(sc)] = int(rule["src"])
            inp[int(sr), int(sc + 1)] = int(rule["tgt"])
            for r, c in teal_cells:
                inp[int(r), int(c)] = 8
            occupied.update(all_cells)
            _reserve_chebyshev(reserved, all_cells, radius=2)
            placed = True
            break
        if not placed:
            return False
    return True


def _place_scattered_and_output(rules: list[dict], inp: np.ndarray, out: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = map(int, box)
    occupied_src: set[tuple[int, int]] = set()
    reserved_src: set[tuple[int, int]] = set()
    occupied_out: set[tuple[int, int]] = set()

    for rule in rules:
        min_r, max_r, min_c, max_c = rule["bounds"]
        src_row_lo = r0 - min_r
        src_row_hi = r1 - max_r
        src_col_lo = c0 - min_c
        src_col_hi = c1 - max_c
        if src_row_lo > src_row_hi or src_col_lo > src_col_hi:
            return False

        positions: list[tuple[int, int]] = []
        for _ in range(int(rule["n_sc"]) * SCATTERED_RETRIES_PER_PIXEL):
            if len(positions) >= int(rule["n_sc"]):
                break
            sr = random.randint(src_row_lo, src_row_hi)
            sc = random.randint(src_col_lo, src_col_hi)
            src_pos = (int(sr), int(sc))
            if src_pos in occupied_src or src_pos in reserved_src:
                continue

            stamp_cells = {(int(sr + rr), int(sc + cc)) for rr, cc in rule["rel"]}
            if any(not (r0 <= r <= r1 and c0 <= c <= c1) for r, c in stamp_cells):
                continue
            if stamp_cells & occupied_out:
                continue

            positions.append(src_pos)
            occupied_src.add(src_pos)
            occupied_out.update(stamp_cells)
            _reserve_4neighbors(reserved_src, [src_pos])
            inp[int(sr), int(sc)] = int(rule["src"])
            for r, c in stamp_cells:
                out[int(r), int(c)] = int(rule["tgt"])

        if len(positions) < SCATTERED_MIN:
            return False
        rule["scattered"] = positions

    for rule in rules:
        rule["occupied_out"] = {
            (int(sr + rr), int(sc + cc))
            for sr, sc in rule["scattered"]
            for rr, cc in rule["rel"]
        }
    return True


def _place_bystanders(
    used_colors: set[int],
    inp: np.ndarray,
    out: np.ndarray,
    box: tuple[int, int, int, int],
) -> list[tuple[int, int, int]]:
    bystander_colors = [c for c in NON_BG_NON_TEAL_COLORS if c not in used_colors]
    if not bystander_colors:
        return []

    result_cells = {(int(r), int(c)) for r in range(box[0], box[2] + 1) for c in range(box[1], box[3] + 1)}
    blocked = set(map(tuple, np.argwhere(out != 0)))
    blocked.update(map(tuple, np.argwhere(inp != 0)))
    free_cells = list(result_cells - blocked)
    random.shuffle(free_cells)

    n = random.randint(0, min(BYSTANDER_MAX, len(free_cells)))
    placed: list[tuple[int, int, int]] = []
    for r, c in free_cells[:n]:
        color = int(random.choice(bystander_colors))
        inp[int(r), int(c)] = color
        out[int(r), int(c)] = color
        placed.append((int(r), int(c), color))
    return placed


def _gen_attempt() -> np.ndarray | None:
    num_rules = random.choice(RULE_COUNT_WEIGHTS)
    rules, used_colors = _build_rules(num_rules)
    h, w, legend_h = _choose_grid(num_rules, rules)
    result_box = _choose_result_box(h, w, legend_h, rules)
    if result_box is None:
        return None

    _assign_scattered_counts(rules, (result_box[2] - result_box[0] + 1) * (result_box[3] - result_box[1] + 1))
    inp = np.zeros((h, w), dtype=int)
    full_output = np.zeros((h, w), dtype=int)

    if not _place_legends_top(rules, inp, legend_h):
        return None
    if not _place_scattered_and_output(rules, inp, full_output, result_box):
        return None
    _place_bystanders(used_colors, inp, full_output, result_box)

    expected = _crop_to_non_bg(full_output, bg=0)
    if expected.shape == inp.shape:
        return None

    try:
        if np.array_equal(main(inp), expected):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {OUTER_RETRIES} attempts")
