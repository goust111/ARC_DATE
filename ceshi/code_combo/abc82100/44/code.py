# Combo variant of abc82100 (index 44)
# combo_spec: D3/2+D7/1+D8/0
# Name: 边缘源像素+规则移至边缘+紧凑裁剪

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import bbox_coords, components_mask, get_bg
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "solvers"))
    from arc_utils import bbox_coords, components_mask, get_bg


# Design analysis for the combo variant:
# - Concrete behavior / execution trace:
#   Start from the original "learn local source->target + teal stencil, then
#   stamp that stencil on matching source-color pixels". The fusion changes two
#   spatial roles at once. In the input, every true rule example sits on the
#   top border, while every activatable source pixel sits on the border as a
#   control marker; the generator uses the bottom border for these controls.
#   A bottom-edge source no longer means "stamp at that same border cell".
#   Instead, it projects inward to the middle row while keeping its column, so
#   a source at (H-1, c) triggers the rule's stencil at (H//2, c). After those
#   central stamps are painted, the learned rule examples are rebuilt as a
#   compact legend strip on the top edge of the final answer, and the answer is
#   tight-cropped. Compared with the original task, both the consumed sources
#   and the original edge legends disappear from their input locations, the
#   painted result lives in the middle, and the rule examples reappear only as a
#   relocated reference band.
# - Complexity calibration:
#   The puzzle still needs repeated evidence for color substitution, so each
#   accepted sample keeps multiple bottom-edge controls per rule; the current
#   generator uses 3 controls per rule and reliably supports single-rule and
#   double-rule layouts within the 20x20 board limit. Legends are consistently
#   grouped on the top edge and sources are consistently isolated on the bottom
#   edge. That avoids a noisy "everything everywhere on the border" look while
#   still forcing the solver to use edge markers rather than interior source
#   pixels.
# - Hidden assumptions review of the original `main()`:
#   `for sr, sc in np.argwhere(g == source_color)` used to assume the output
#   stamp anchor is exactly the source location. That is no longer valid, so the
#   translated anchor is now computed by `_project_edge_source(...)`.
#   `out = np.full_like(g, bg)` is still useful as an intermediate core canvas,
#   but returning that full-size canvas is no longer correct because D8 now
#   demands a tight crop. The original "erase rule examples" assumption also no
#   longer holds; we must explicitly rebuild them as a moved legend strip.
#   The preservation loop for bystanders remains valid, but only for colors that
#   are not part of any learned source color.
# - Placement legality:
#   The generator enforces four non-overlap regimes: top-edge legends do not
#   overlap each other, bottom-edge source pixels are 4-neighbor separated,
#   projected middle-row stencil footprints do not overlap, and bystanders avoid
#   both the input legend/source neighborhoods and the projected footprints.
#   Stencils are generated with their anchor on the stencil's top row, which
#   guarantees that a downward-pointing top-edge legend always fits.
# - Visual observability:
#   Across examples, humans can see three stable cues: rule examples live on the
#   input top edge, source controls live on the input bottom edge, and the
#   output foreground appears in a central band with the rule examples reissued
#   as a legend strip at the output top edge. Tight crop then removes the dead
#   border, so the changed roles of "edge as control" and "edge as legend" are
#   both visible.
# - Neatness / color inference:
#   Source colors and target colors still come only from local legends, so color
#   inference is structural rather than numeric. The generator keeps bystanders
#   on colors outside all rule source colors and keeps projected stencil
#   footprints disjoint, preventing the center from becoming visually tangled.
# - New assumptions introduced here and matched by the generator:
#   1. A true legend is a top-edge source->target vertical pair with a nearby
#      teal stencil below it.
#   2. Activatable non-legend source pixels lie on the border; the generator
#      uses the bottom border subset to keep the projection rule obvious.
#   3. The moved output legend is packed left-to-right in the same visible order
#      as the input legends.
#   4. Tight crop is taken after combining the relocated legend strip with the
#      projected central painting.


PATTERN_COLOR = 8
GRID_DIM_MAX = 20
MIN_DIM_BY_RULES = {1: 12, 2: 14, 3: 16, 4: 18}
RULE_COUNT_WEIGHTS = [1, 1, 2, 2, 3, 3, 4]
NON_BG_NON_TEAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

STENCIL_MIN_PIXELS = 2
STENCIL_MAX_CAP = 6
STENCIL_MAX_H = 4
STENCIL_MAX_W = 4
STENCIL_SHAPE_RETRIES = 80

SCATTERED_MIN = 3
SCATTERED_MAX = 3
BYSTANDER_MAX = 2

LEGEND_GAP = 1
STACK_GAP = 1
OUTER_RETRIES = 500

_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _anchor_distance(pattern: np.ndarray, anchor: tuple[int, int]) -> int:
    return min(_chebyshev((int(r), int(c)), anchor) for r, c in pattern)


def _tight_crop_non_bg(grid: np.ndarray, bg: int) -> np.ndarray:
    nz = np.argwhere(grid != int(bg))
    if nz.size == 0:
        return np.full((0, 0), int(bg), dtype=int)
    r0, c0, r1, c1 = bbox_coords(nz)
    return np.array(grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1], dtype=int)


def _legend_bbox_from_rel(rel: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [-2, -1] + [int(rr) for rr, _ in rel]
    cols = [0, 0] + [int(cc) for _, cc in rel]
    return min(rows), min(cols), max(rows), max(cols)


def _legend_size_from_rel(rel: list[tuple[int, int]]) -> tuple[int, int]:
    min_r, min_c, max_r, max_c = _legend_bbox_from_rel(rel)
    return int(max_r - min_r + 1), int(max_c - min_c + 1)


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


def _project_edge_source(h: int, w: int, sr: int, sc: int) -> tuple[int, int]:
    sr = int(sr)
    sc = int(sc)
    if sr == 0 or sr == h - 1:
        return int(h // 2), int(sc)
    if sc == 0 or sc == w - 1:
        return int(sr), int(w // 2)
    return int(sr), int(sc)


def _render_legend_block(rule: dict, bg: int, pattern_color: int = PATTERN_COLOR) -> np.ndarray:
    cells: list[tuple[int, int, int]] = [
        (int(rule["source_pos"][0]), int(rule["source_pos"][1]), int(rule["source_color"])),
        (int(rule["target_pos"][0]), int(rule["target_pos"][1]), int(rule["target_color"])),
    ]
    for r, c in rule["pattern"]:
        cells.append((int(r), int(c), int(pattern_color)))

    rs = [r for r, _c, _v in cells]
    cs = [c for _r, c, _v in cells]
    min_r, max_r = min(rs), max(rs)
    min_c, max_c = min(cs), max(cs)
    block = np.full((int(max_r - min_r + 1), int(max_c - min_c + 1)), int(bg), dtype=int)
    for r, c, value in cells:
        block[int(r - min_r), int(c - min_c)] = int(value)
    return block


def _build_legend_strip(rules: list[dict], bg: int, gap: int = LEGEND_GAP) -> np.ndarray:
    if not rules:
        return np.full((0, 0), int(bg), dtype=int)

    ordered = sorted(rules, key=lambda rule: (int(rule["source_pos"][1]), int(rule["source_pos"][0])))
    blocks = [_render_legend_block(rule, bg=bg) for rule in ordered]
    height = max(int(block.shape[0]) for block in blocks)
    width = sum(int(block.shape[1]) for block in blocks) + gap * max(0, len(blocks) - 1)
    strip = np.full((height, width), int(bg), dtype=int)

    cur = 0
    for block in blocks:
        bh, bw = block.shape
        mask = block != int(bg)
        strip[0:bh, cur : cur + bw][mask] = block[mask]
        cur += int(bw) + gap
    return strip


def _stack_legend_and_core(legend_strip: np.ndarray, core_crop: np.ndarray, bg: int) -> np.ndarray:
    if legend_strip.size == 0:
        return core_crop
    if core_crop.size == 0:
        return legend_strip

    lh, lw = legend_strip.shape
    ch, cw = core_crop.shape
    out = np.full((int(lh + STACK_GAP + ch), int(max(lw, cw))), int(bg), dtype=int)

    legend_mask = legend_strip != int(bg)
    core_mask = core_crop != int(bg)
    out[0:lh, 0:lw][legend_mask] = legend_strip[legend_mask]
    out[int(lh + STACK_GAP) : int(lh + STACK_GAP + ch), 0:cw][core_mask] = core_crop[core_mask]
    return _tight_crop_non_bg(out, bg=int(bg))


def _solve_core(grid: np.ndarray) -> tuple[np.ndarray, list[dict], int]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    h, w = g.shape

    rules = _discover_rules(g, bg=bg, pattern_color=PATTERN_COLOR)
    legend_pixels: set[tuple[int, int]] = set()
    source_colors: set[int] = set()
    for rule in rules:
        legend_pixels.add((int(rule["source_pos"][0]), int(rule["source_pos"][1])))
        legend_pixels.add((int(rule["target_pos"][0]), int(rule["target_pos"][1])))
        source_colors.add(int(rule["source_color"]))

    out = np.full((h, w), int(bg), dtype=int)

    for r, c in map(tuple, np.argwhere((g != int(bg)) & (g != int(PATTERN_COLOR)))):
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        if (r, c) in legend_pixels or color in source_colors:
            continue
        out[r, c] = color

    for rule in rules:
        ar, ac = int(rule["anchor"][0]), int(rule["anchor"][1])
        pattern = np.asarray(rule["pattern"], dtype=int)
        for sr, sc in map(tuple, np.argwhere(g == int(rule["source_color"]))):
            sr = int(sr)
            sc = int(sc)
            if (sr, sc) in legend_pixels:
                continue
            dst_r, dst_c = _project_edge_source(h, w, sr, sc)
            dr = int(dst_r - ar)
            dc = int(dst_c - ac)
            rows = pattern[:, 0] + dr
            cols = pattern[:, 1] + dc
            ok = (0 <= rows) & (rows < h) & (0 <= cols) & (cols < w)
            out[rows[ok], cols[ok]] = int(rule["target_color"])

    return out, rules, int(bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn edge legends, project edge controls inward, then crop the rebuilt scene.

    Input structure:
    - A black-background grid contains teal (8) stencil components and colored
      pixels, but the true rule examples live on the border rather than in the
      interior.
    - Each rule example is still a 4-neighbor source->target pair whose forward
      continuation reaches an anchor touching a teal stencil; in generated
      samples these legends sit on the top edge, while the activatable source
      pixels appear as isolated edge controls on the border.

    Rule:
    - Learn the same source-color -> target-color + teal-stencil mapping from
      each local legend.
    - For every non-legend source pixel, do not stamp at the border cell
      itself; instead project that edge pixel inward onto the grid's middle row
      or middle column, then translate the stencil there and recolor it with
      the target color.
    - Preserve non-rule bystanders, rebuild the visible rule legends as a
      compact strip along the top edge of the result, and finally tight-crop to
      the minimal rectangle containing all non-background output pixels.

    Key insight: Edge pixels now play two different roles. Some edge pixels are
    legends that teach the color/stencil rule, while other edge pixels are
    control markers whose row/column is projected into the center where the
    actual painted result appears.
    """
    core, rules, bg = _solve_core(np.asarray(grid, dtype=int))
    core_crop = _tight_crop_non_bg(core, bg=bg)
    legend_strip = _build_legend_strip(rules, bg=bg)
    return _stack_legend_and_core(legend_strip, core_crop, bg=bg)


def _reserve_4neighbors(reserved: set[tuple[int, int]], cells) -> None:
    for r, c in cells:
        r = int(r)
        c = int(c)
        reserved.add((r, c))
        for dr, dc in _DIRS4:
            reserved.add((int(r + dr), int(c + dc)))


def _make_stencil(min_pixels: int, max_pixels: int) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    for _ in range(STENCIL_SHAPE_RETRIES):
        target = random.randint(int(min_pixels), int(max_pixels))
        coords = {(0, 0)}
        while len(coords) < target:
            frontier = set()
            for r, c in coords:
                for dr, dc in _DIRS8:
                    cand = (int(r + dr), int(c + dc))
                    if cand in coords:
                        continue
                    trial = coords | {cand}
                    rs = [pt[0] for pt in trial]
                    cs = [pt[1] for pt in trial]
                    if max(rs) - min(rs) + 1 > STENCIL_MAX_H:
                        continue
                    if max(cs) - min(cs) + 1 > STENCIL_MAX_W:
                        continue
                    frontier.add(cand)
            if not frontier:
                break
            coords.add(random.choice(list(frontier)))

        if len(coords) < min_pixels:
            continue

        coords_list = sorted(coords)
        min_r = min(r for r, _c in coords_list)
        anchors = [pt for pt in coords_list if int(pt[0]) == int(min_r)]
        anchor = random.choice(anchors)
        return coords_list, anchor

    raise RuntimeError("failed to build stencil")


def _rel_bounds(rel: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [int(r) for r, _ in rel]
    cs = [int(c) for _, c in rel]
    return min(rs), max(rs), min(cs), max(cs)


def _build_rules(num_rules: int, area: int) -> tuple[list[dict], set[int]]:
    colors = NON_BG_NON_TEAL_COLORS[:]
    random.shuffle(colors)

    max_pixels = min(STENCIL_MAX_CAP, max(STENCIL_MIN_PIXELS, area // 35))
    max_pixels = max(STENCIL_MIN_PIXELS, max_pixels)

    rules: list[dict] = []
    used_colors: set[int] = set()
    for i in range(num_rules):
        src_c = int(colors[2 * i])
        tgt_c = int(colors[2 * i + 1])
        used_colors.update((src_c, tgt_c))
        stencil, anchor = _make_stencil(STENCIL_MIN_PIXELS, max_pixels)
        rel = [(int(r - anchor[0]), int(c - anchor[1])) for r, c in stencil]
        rule = {
            "src": src_c,
            "tgt": tgt_c,
            "rel": rel,
            "bounds": _rel_bounds(rel),
            "n_sc": random.randint(SCATTERED_MIN, SCATTERED_MAX),
        }
        leg_h, leg_w = _legend_size_from_rel(rel)
        rule["legend_h"] = int(leg_h)
        rule["legend_w"] = int(leg_w)
        rules.append(rule)
    return rules, used_colors


def _required_width_for_legends(rules: list[dict]) -> int:
    return sum(int(rule["legend_w"]) for rule in rules) + LEGEND_GAP * max(0, len(rules) - 1)


def _place_top_legends(rules: list[dict], inp: np.ndarray, occupied: set[tuple[int, int]], reserved: set[tuple[int, int]]) -> bool:
    h, w = inp.shape
    required_w = _required_width_for_legends(rules)
    if required_w > w:
        return False

    left_margin = random.randint(0, w - required_w)
    cur = int(left_margin)
    ordered = rules[:]
    random.shuffle(ordered)
    ordered.sort(key=lambda rule: int(rule["legend_w"]), reverse=True)

    for rule in ordered:
        min_r, min_c, max_r, max_c = _legend_bbox_from_rel(rule["rel"])
        anchor_c = int(cur - min_c)
        source_pos = (0, int(anchor_c))
        target_pos = (1, int(anchor_c))
        anchor = (2, int(anchor_c))
        teal_cells = {(int(anchor[0] + rr), int(anchor[1] + cc)) for rr, cc in rule["rel"]}
        visible_cells = {source_pos, target_pos} | teal_cells

        if max(r for r, _c in visible_cells) >= h:
            return False
        if min(c for _r, c in visible_cells) < 0 or max(c for _r, c in visible_cells) >= w:
            return False
        if visible_cells & occupied:
            return False

        inp[source_pos] = int(rule["src"])
        inp[target_pos] = int(rule["tgt"])
        for r, c in teal_cells:
            inp[int(r), int(c)] = int(PATTERN_COLOR)

        occupied.update(visible_cells)
        _reserve_4neighbors(reserved, visible_cells)

        rule["source_pos"] = source_pos
        rule["target_pos"] = target_pos
        rule["anchor"] = anchor
        rule["pattern"] = np.asarray(sorted(teal_cells), dtype=int)

        cur += int(rule["legend_w"]) + LEGEND_GAP

    rules.sort(key=lambda rule: int(rule["source_pos"][1]))
    return True


def _projected_cells_for_bottom_source(rule: dict, h: int, source_col: int) -> set[tuple[int, int]]:
    mid_r = int(h // 2)
    return {
        (int(mid_r + rr), int(source_col + cc))
        for rr, cc in rule["rel"]
    }


def _place_bottom_sources(
    rules: list[dict],
    inp: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> tuple[bool, set[tuple[int, int]]]:
    h, w = inp.shape
    projected_occupied: set[tuple[int, int]] = set()
    row = int(h - 1)

    ordered = sorted(
        rules,
        key=lambda rule: (
            (int(rule["bounds"][3]) - int(rule["bounds"][2]) + 1),
            len(rule["rel"]),
        ),
        reverse=True,
    )
    for rule in ordered:
        min_r, max_r, min_c, max_c = rule["bounds"]
        mid_r = int(h // 2)
        if mid_r + int(max_r) >= h:
            return False, set()

        c0 = max(1, int(-min_c) + 1)
        c1 = min(w - 2, int(w - 2 - max_c))
        if c0 > c1:
            return False, set()

        cols = list(range(int(c0), int(c1) + 1))
        random.shuffle(cols)
        placed: list[tuple[int, int]] = []
        for col in cols:
            if len(placed) >= int(rule["n_sc"]):
                break
            pos = (int(row), int(col))
            stamp = _projected_cells_for_bottom_source(rule, h, col)
            if pos in occupied or pos in reserved:
                continue
            if any(not (0 <= r < h and 0 <= c < w) for r, c in stamp):
                continue
            if stamp & projected_occupied:
                continue

            placed.append(pos)
            occupied.add(pos)
            _reserve_4neighbors(reserved, [pos])
            projected_occupied.update(stamp)

        if len(placed) < SCATTERED_MIN:
            return False, set()

        rule["scattered"] = placed
        for r, c in placed:
            inp[int(r), int(c)] = int(rule["src"])

    return True, projected_occupied


def _place_bystanders(
    used_colors: set[int],
    inp: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
    projected_occupied: set[tuple[int, int]],
) -> list[tuple[int, int, int]]:
    colors = [int(c) for c in NON_BG_NON_TEAL_COLORS if int(c) not in used_colors]
    if not colors:
        return []

    h, w = inp.shape
    mid_r = int(h // 2)
    candidates = [
        (int(r), int(c))
        for r in range(max(3, mid_r - 1), min(h - 1, mid_r + 3))
        for c in range(1, w - 1)
    ]
    random.shuffle(candidates)

    wanted = random.randint(0, BYSTANDER_MAX)
    placed: list[tuple[int, int, int]] = []
    for r, c in candidates:
        if len(placed) >= wanted:
            break
        if (r, c) in occupied or (r, c) in reserved or (r, c) in projected_occupied:
            continue
        color = int(random.choice(colors))
        inp[int(r), int(c)] = color
        placed.append((int(r), int(c), color))
        occupied.add((int(r), int(c)))
        _reserve_4neighbors(reserved, [(int(r), int(c))])
    return placed


def _build_expected_output(rules: list[dict], bystanders: list[tuple[int, int, int]], h: int, w: int) -> np.ndarray:
    core = np.zeros((h, w), dtype=int)
    for r, c, color in bystanders:
        core[int(r), int(c)] = int(color)

    for rule in rules:
        for sr, sc in rule["scattered"]:
            dst_r, dst_c = _project_edge_source(h, w, int(sr), int(sc))
            for rr, cc in rule["rel"]:
                core[int(dst_r + rr), int(dst_c + cc)] = int(rule["tgt"])

    core_crop = _tight_crop_non_bg(core, 0)
    legend_rules = [
        {
            "source_color": int(rule["src"]),
            "target_color": int(rule["tgt"]),
            "source_pos": tuple(map(int, rule["source_pos"])),
            "target_pos": tuple(map(int, rule["target_pos"])),
            "anchor": tuple(map(int, rule["anchor"])),
            "pattern": np.asarray(rule["pattern"], dtype=int),
        }
        for rule in rules
    ]
    legend_strip = _build_legend_strip(legend_rules, bg=0)
    return _stack_legend_and_core(legend_strip, core_crop, bg=0)


def _fusion_metrics(inp: np.ndarray, out: np.ndarray) -> dict[str, int]:
    h, w = inp.shape
    edge_non_bg = sum(
        int(inp[r, c] != 0)
        for r in range(h)
        for c in range(w)
        if r in (0, h - 1) or c in (0, w - 1)
    )
    interior_non_bg = int(np.count_nonzero(inp)) - edge_non_bg

    moved_strip_height = 0
    for r in range(out.shape[0]):
        if int(np.any(out[r, :] == PATTERN_COLOR)):
            moved_strip_height = int(r + 1)
        elif moved_strip_height:
            break

    return {
        "input_edge_non_bg": int(edge_non_bg),
        "input_interior_non_bg": int(interior_non_bg),
        "output_has_teal": int(np.any(out == PATTERN_COLOR)),
        "output_smaller_h": int(out.shape[0] < h),
        "output_smaller_w": int(out.shape[1] < w),
        "moved_strip_height": int(moved_strip_height),
    }


def _gen_attempt() -> np.ndarray | None:
    num_rules = random.choice(RULE_COUNT_WEIGHTS)
    rules, used_colors = _build_rules(num_rules, area=GRID_DIM_MAX * GRID_DIM_MAX)

    max_rel_r = max(int(rule["bounds"][1]) for rule in rules)
    min_h = max(MIN_DIM_BY_RULES[num_rules], int(2 * max_rel_r + 10))
    min_w = max(MIN_DIM_BY_RULES[num_rules], int(_required_width_for_legends(rules) + 2))
    if min_h > GRID_DIM_MAX or min_w > GRID_DIM_MAX:
        return None

    h = random.randint(int(min_h), GRID_DIM_MAX)
    w = random.randint(int(min_w), GRID_DIM_MAX)
    inp = np.zeros((h, w), dtype=int)

    occupied: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()

    if not _place_top_legends(rules, inp, occupied, reserved):
        return None
    ok, projected_occupied = _place_bottom_sources(rules, inp, occupied, reserved)
    if not ok:
        return None
    bystanders = _place_bystanders(used_colors, inp, occupied, reserved, projected_occupied)

    expected = _build_expected_output(rules, bystanders, h, w)
    if expected.size == 0:
        return None
    if expected.shape[0] >= h or expected.shape[1] >= w:
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    metrics = _fusion_metrics(inp, actual)
    if metrics["input_edge_non_bg"] <= metrics["input_interior_non_bg"]:
        return None
    if not metrics["output_has_teal"]:
        return None
    if not (metrics["output_smaller_h"] and metrics["output_smaller_w"]):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(OUTER_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {OUTER_RETRIES} attempts")
