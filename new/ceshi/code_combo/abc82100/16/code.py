# Combo variant of abc82100 (index 16)
# combo_spec: D7/0+D8/2
# Name: 保留规则示例+仅输出变换区域

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace:
#    The input still has two semantic zones. In a top legend strip, each rule is
#    shown as a source-color pixel adjacent to a target-color pixel, followed in
#    the same direction by a teal stencil anchor/component. In a lower work box,
#    isolated source-color pixels are the real transformation triggers.
#    Small trace: if a top-strip legend shows blue -> red with a teal L-shape,
#    and three blue pixels appear in the lower box, then the full transformed
#    canvas would keep the blue/red legend pair, erase teal, and stamp red
#    L-shapes on the three lower blue pixels. The combo twist is that the final
#    returned output is not that full canvas; it is only the cropped work result.
#    So preserving legends matters during reasoning, but those preserved legends
#    must be excluded from the crop.
# b) Complexity calibration:
#    We keep 1-4 rules, repeated scattered source pixels, and small 8-connected
#    stencils. The work box contains all surviving non-legend content; the legend
#    strip is separated by empty rows. This keeps the crop visually obvious while
#    retaining enough repeated evidence per color rule to avoid ambiguity.
# c) Implicit-assumption audit for main():
#    - Rule discovery assumes teal exists only near legends. The generator places
#      all teal in the top strip and nowhere in the work box.
#    - The preservation pass assumes any non-legend pixel of a learned source
#      color is a transform base that should disappear. The generator therefore
#      forbids bystanders from using source colors.
#    - D7 breaks the old D8 shortcut "crop the non-background output", because
#      legend pixels now survive. The solver must instead crop only on surviving
#      non-legend pixels. The generator makes this necessary by ensuring legend
#      pixels remain non-background and lie outside the work box.
#    - Restoring legend colors after stamping assumes translated stencils should
#      not be allowed to redefine legend cells. The generator keeps the two zones
#      disjoint so this restoration is logically inert but still faithful.
# d) Placement legality:
#    Legends are sampled only inside a bounded top strip with room for the full
#    teal stencil. Scattered sources are sampled only where their translated
#    stencil stays inside the work box. Output stamps are non-overlapping and
#    bystanders are placed only on free cells inside the work box.
# e) Visual observability:
#    Across examples, humans see a stable instruction strip above and a separate
#    source-pixel cluster below. The output keeps only the lower transformed area.
#    Because the legend colors would survive on the full canvas, the solver must
#    actively identify and exclude the rule area; this is visible from paired
#    examples without using numeric color identities.
# f) Visual cleanliness:
#    Restricting all true output pixels to one compact work box prevents clutter.
#    Non-overlapping stamps and in-box bystanders keep the cropped outputs easy
#    to parse, while the empty gap between strips makes the two roles legible.
# g) Color/change inferability:
#    Each source color appears several times in the work box and is explained by a
#    nearby source->target legend plus a unique stencil. That repeated local cue
#    is enough for humans to infer both recoloring and shape transfer.


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


def _crop_to_mask(grid: np.ndarray, mask: np.ndarray, bg: int) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.full((1, 1), int(bg), dtype=int)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return np.asarray(grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1], dtype=int)


def _crop_result_region(grid: np.ndarray, legend_pixels: set[tuple[int, int]], bg: int) -> np.ndarray:
    mask = np.asarray(grid != int(bg), dtype=bool)
    for r, c in legend_pixels:
        if 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1]:
            mask[int(r), int(c)] = False
    return _crop_to_mask(grid, mask, bg=bg)


def _crop_to_non_bg(grid: np.ndarray, bg: int) -> np.ndarray:
    return _crop_to_mask(grid, np.asarray(grid != int(bg), dtype=bool), bg=bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn local source-target-stencil examples, keep their colored legend pairs,
    and return only the cropped transformation region.

    Input structure:
    - A black-background grid with a separated top instruction strip and a lower
      work region.
    - Each legend is a 4-neighbor source->target pair whose forward continuation
      reaches a teal (8) stencil component; elsewhere, isolated source-color
      pixels repeat those source colors inside the work region.

    Rule:
    - For every discovered legend, translate its teal stencil onto every non-legend
      pixel of the same source color, recoloring the stencil to the legend's
      target color and aligning the legend anchor with that source pixel.
    - On the full transformed canvas, teal is removed but the colored legend pair
      remains.
    - The final answer is then cropped to the surviving non-legend result area,
      excluding the preserved rule strip.

    Key insight: The colored legend pair is still a valid part of the transformed
    full canvas, so a naive non-background crop is wrong. You must separate
    "instruction pixels" from "result pixels" and crop only around the latter.
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
        if (r, c) in legend_pixels:
            out[r, c] = color
            continue
        if color in source_colors:
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

    for rule in rules:
        sr, sc = rule["source_pos"]
        tr, tc = rule["target_pos"]
        out[int(sr), int(sc)] = int(rule["source_color"])
        out[int(tr), int(tc)] = int(rule["target_color"])

    return _crop_result_region(out, legend_pixels=legend_pixels, bg=bg)


# --------------- generator (combo D7/0 + D8/2) ---------------

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
            rule["source_pos"] = (int(sr), int(sc))
            rule["target_pos"] = (int(sr), int(sc + 1))
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


def _legend_pixels_from_rules(rules: list[dict]) -> set[tuple[int, int]]:
    legend_pixels: set[tuple[int, int]] = set()
    for rule in rules:
        legend_pixels.add(tuple(rule["source_pos"]))
        legend_pixels.add(tuple(rule["target_pos"]))
    return legend_pixels


def _seed_legends_into_output(rules: list[dict], out: np.ndarray) -> None:
    for rule in rules:
        sr, sc = rule["source_pos"]
        tr, tc = rule["target_pos"]
        out[int(sr), int(sc)] = int(rule["src"])
        out[int(tr), int(tc)] = int(rule["tgt"])


def _gen_attempt() -> np.ndarray | None:
    num_rules = random.choice(RULE_COUNT_WEIGHTS)
    rules, used_colors = _build_rules(num_rules)
    h, w, legend_h = _choose_grid(num_rules, rules)
    result_box = _choose_result_box(h, w, legend_h, rules)
    if result_box is None:
        return None

    result_area = (result_box[2] - result_box[0] + 1) * (result_box[3] - result_box[1] + 1)
    _assign_scattered_counts(rules, result_area)

    inp = np.zeros((h, w), dtype=int)
    full_output = np.zeros((h, w), dtype=int)

    if not _place_legends_top(rules, inp, legend_h):
        return None
    if not _place_scattered_and_output(rules, inp, full_output, result_box):
        return None
    _place_bystanders(used_colors, inp, full_output, result_box)
    _seed_legends_into_output(rules, full_output)

    legend_pixels = _legend_pixels_from_rules(rules)
    expected = _crop_result_region(full_output, legend_pixels=legend_pixels, bg=0)
    naive = _crop_to_non_bg(full_output, bg=0)

    if expected.shape == inp.shape:
        return None
    if np.array_equal(expected, naive):
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
