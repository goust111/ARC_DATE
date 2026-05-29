# Combo variant of 291dc1e1 (index 4)
# combo_spec: D1/0+D13/0
# Name: 面积排序+分色多输出

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste


# Design analysis for the combo variant
# ====================================
# a) Concrete behavior and trace:
#    The source scene keeps the original corner-marker frame: one blue edge line,
#    one red edge line, and several interior solid rectangles. The original task
#    globally sorted every rectangle by (distance to blue, distance to red), then
#    stacked everything into one vertical pile. This combo changes two linked
#    stages at once:
#    - D1: the ordering key becomes rectangle area.
#    - D13: rectangles no longer merge into one stack; they first split by
#      dominant color, then each color-group stacks independently, and the
#      stacks are placed side by side.
#    A concrete canonical example is:
#      top band:    A=2x5 green, B=2x2 yellow, C=2x4 green
#      bottom band: D=3x3 maroon, E=3x5 yellow
#    Original order would be A,B,C,D,E because the whole top band is nearer to
#    the blue edge. The combo instead computes areas 10,4,8,9,15, giving the
#    global area sequence B(yellow), C(green), D(maroon), A(green), E(yellow).
#    The output is therefore three blocks from left to right by first encounter
#    in that area sequence:
#      yellow block: B then E
#      green block:  C then A
#      maroon block: D
#    If the blue line is vertical, each rectangle still rotates 90 degrees CCW
#    before entering its color block.
# b) Complexity calibration:
#    If every rectangle had the same height, area would collapse to width and
#    the new ordering would look too close to the old distance order. The
#    generator therefore uses two bands of different heights (2 and 3), making
#    area a true counting task. To avoid weak many:many examples, at least two
#    dominant colors repeat, and colors are interleaved across the scene so the
#    multi-block output is visually necessary rather than accidental.
# c) Implicit-assumption audit:
#    - Corner detection still assumes a unique corner marker distinct from its
#      two adjacent edge colors. The generator preserves that exactly.
#    - Blue-line inference still relies on color 1 if present, else on the
#      longer edge. The generator still uses color 1 as blue.
#    - Rectangle extraction still assumes every target component fully fills its
#      bbox with no background holes. The generator keeps each object a solid
#      rectangle and only adds sparse non-background accent pixels.
#    - The original `rects.sort(...)` assumed distance keys. That assumption is
#      removed; we now sort by area and keep spatial tiebreakers only for
#      determinism.
#    - The original single `seq` / single output canvas assumed many:1. That is
#      replaced with per-color sequences and then side-by-side block assembly.
#    - Grouping cannot assume monochrome rectangles because accents may appear
#      inside them; grouping therefore uses the dominant color instead.
# d) Placement legality:
#    Rectangles are placed on two fixed horizontal bands with at least two
#    background columns between neighbors. The canonical grid is sized from the
#    chosen widths before any placement. Whole-grid flips and rotations preserve
#    the non-overlap and keep every object away from the edge lines.
# e) Visual observability:
#    A human sees two simultaneous signals in examples: the output is split into
#    multiple side-by-side stacks by dominant color, and within each stack the
#    rectangle sizes increase by area rather than following near-edge order.
#    The generator rejects samples where the combo output matches the D13-only
#    distance-grouped output or the D1-only single-stack output, so both changes
#    remain visible in the observed training pairs.
# f) Visual cleanliness:
#    The input stays close to the source task: one corner frame plus two neat
#    bands of rectangles. Accent pixels are sparse and never remove the large
#    dominant color field, so object outlines and color groups remain readable.
#    Output blocks are separated by one full background column to avoid merging.
# g) Color inferability:
#    There is no recoloring rule: dominant rectangle color is both visible in
#    the input and reused as the grouping key in the output. Only 2-3 dominant
#    colors appear per sample, and minority accents are limited to blue/red so
#    they cannot overpower the dominant color cue.
# h) Generator validation:
#    Each candidate input is rendered with metadata, then compared against three
#    behaviors on the same input:
#    - combo target: area-sorted, color-grouped multi-output
#    - D13-only control: distance-sorted, color-grouped multi-output
#    - D1-only control: area-sorted, single-stack output
#    The generator keeps only candidates where `main(input)` matches the combo
#    target and differs from both controls.
# i) New solver assumptions after the edit:
#    The solver now assumes every object has a well-defined dominant color and a
#    measurable area, and that side-by-side blocks can be ordered by first
#    appearance in the global area order. The generator guarantees strict
#    dominant-color majorities, unique areas for every rectangle, at least two
#    repeated color groups, and no touching components.


BG = 8
BLUE = 1
RED = 2
MARKER = 0
DOMINANT_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
MINORITY_COLORS = np.array([BLUE, RED], dtype=int)

TOP_ROW = 2
BOTTOM_ROW = 6
TOP_H = 2
BOTTOM_H = 3
CANONICAL_H = 11


def _locate_frame(g: np.ndarray) -> tuple[int, tuple[int, int], int, str, str]:
    H, W = g.shape
    bg = int(get_bg(g))
    corners = [(0, 0), (0, int(W - 1)), (int(H - 1), 0), (int(H - 1), int(W - 1))]
    corner = None
    for r, c in corners:
        if int(g[r, c]) == int(bg):
            continue
        nbr = []
        if r == 0:
            nbr.append(int(g[r + 1, c]))
        if r == H - 1:
            nbr.append(int(g[r - 1, c]))
        if c == 0:
            nbr.append(int(g[r, c + 1]))
        if c == W - 1:
            nbr.append(int(g[r, c - 1]))
        if nbr and all(int(n) != int(g[r, c]) for n in nbr):
            corner = (int(r), int(c))
            break
    assert corner is not None

    cr, cc = map(int, corner)
    marker = int(g[cr, cc])
    hstep = 1 if cc == 0 else -1
    vstep = 1 if cr == 0 else -1
    hcol = int(g[cr, cc + hstep])
    vcol = int(g[cr + vstep, cc])

    if hcol == BLUE or vcol == BLUE:
        blue_col = BLUE
    else:
        hlen = 0
        c = int(cc + hstep)
        while 0 <= c < W and int(g[cr, c]) == int(hcol):
            hlen += 1
            c += int(hstep)
        vlen = 0
        r = int(cr + vstep)
        while 0 <= r < H and int(g[r, cc]) == int(vcol):
            vlen += 1
            r += int(vstep)
        blue_col = int(vcol if vlen >= hlen else hcol)

    blue_orient = "h" if int(hcol) == int(blue_col) else "v"
    red_orient = "v" if blue_orient == "h" else "h"
    return int(bg), (int(cr), int(cc)), int(marker), blue_orient, red_orient


def _dominant_color(arr: np.ndarray) -> int:
    vals, counts = np.unique(arr, return_counts=True)
    return int(vals[np.argmax(counts)])


def _along_from_corner(box: tuple[int, int, int, int], orient: str, cr: int, cc: int) -> int:
    r0, c0, r1, c1 = map(int, box)
    if orient == "v":
        return int(r0 if cr == 0 else -r1)
    return int(c0 if cc == 0 else -c1)


def _dist_to_edge(box: tuple[int, int, int, int], orient: str, edge_idx: int) -> int:
    r0, c0, r1, c1 = map(int, box)
    if orient == "v":
        col = int(edge_idx)
        if c0 > col:
            return int(c0 - col - 1)
        if c1 < col:
            return int(col - c1 - 1)
        return 0
    row = int(edge_idx)
    if r0 > row:
        return int(r0 - row - 1)
    if r1 < row:
        return int(row - r1 - 1)
    return 0


def _extract_rectangles(g: np.ndarray) -> tuple[int, tuple[int, int], str, str, list[dict]]:
    bg, (cr, cc), marker, blue_orient, red_orient = _locate_frame(g)
    H, W = g.shape

    line_mask = np.zeros((H, W), dtype=bool)
    line_mask[cr, :] = True
    line_mask[:, cc] = True

    mask = (g != int(bg)) & (~line_mask) & (g != int(marker))
    rects = []
    for coords in components_mask(mask, connectivity=4):
        if coords.size == 0:
            continue
        box = bbox_coords(coords)
        sub = crop_bbox(g, box)
        if bool((sub == int(bg)).any()):
            continue
        rects.append(
            {
                "box": tuple(map(int, box)),
                "arr": sub.copy(),
                "area": int(np.count_nonzero(sub != int(bg))),
                "group": int(_dominant_color(sub)),
            }
        )

    assert rects
    return int(bg), (int(cr), int(cc)), blue_orient, red_orient, rects


def _stack_sequence(seq: list[np.ndarray], bg: int) -> np.ndarray:
    out_w = int(max(int(a.shape[1]) for a in seq))
    out_h = int(sum(int(a.shape[0]) for a in seq))
    out = np.full((out_h, out_w), int(bg), dtype=int)
    out_r = 0
    for a in seq:
        h, w = map(int, a.shape)
        paste(out, a, int(out_r), int((out_w - w) // 2))
        out_r += int(h)
    return out


def _stack_blocks(blocks: list[np.ndarray], bg: int) -> np.ndarray:
    out_h = int(max(int(block.shape[0]) for block in blocks))
    out_w = int(sum(int(block.shape[1]) for block in blocks) + max(0, len(blocks) - 1))
    out = np.full((out_h, out_w), int(bg), dtype=int)
    out_c = 0
    for block in blocks:
        paste(out, block, 0, int(out_c))
        out_c += int(block.shape[1]) + 1
    return out


def _compose_output(
    rects_by_id: dict[int, dict],
    ordered_ids: list[int],
    blue_orient: str,
    bg: int,
    *,
    group_by_color: bool,
) -> np.ndarray:
    if not group_by_color:
        seq = []
        for rid in ordered_ids:
            arr = rects_by_id[int(rid)]["input_patch"]
            if blue_orient == "v":
                arr = np.rot90(arr, k=1)
            seq.append(arr)
        return _stack_sequence(seq, bg)

    groups: dict[int, list[np.ndarray]] = {}
    group_order: list[int] = []
    for rid in ordered_ids:
        rect = rects_by_id[int(rid)]
        grp = int(rect["dominant"])
        if grp not in groups:
            groups[grp] = []
            group_order.append(grp)
        arr = rect["input_patch"]
        if blue_orient == "v":
            arr = np.rot90(arr, k=1)
        groups[grp].append(arr)

    blocks = [_stack_sequence(groups[grp], bg) for grp in group_order]
    return _stack_blocks(blocks, bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group rectangles by dominant color, sort each group by area, and output the
    groups as parallel vertical stacks.
    """
    g = np.array(grid, dtype=int)
    bg, (cr, cc), blue_orient, red_orient, rects = _extract_rectangles(g)

    rects.sort(
        key=lambda r: (
            int(r["area"]),
            _along_from_corner(r["box"], blue_orient, cr, cc),
            _along_from_corner(r["box"], red_orient, cr, cc),
            tuple(map(int, r["box"])),
        )
    )

    groups: dict[int, list[np.ndarray]] = {}
    group_order: list[int] = []
    for rect in rects:
        grp = int(rect["group"])
        if grp not in groups:
            groups[grp] = []
            group_order.append(grp)
        arr = rect["arr"]
        if blue_orient == "v":
            arr = np.rot90(arr, k=1)
        groups[grp].append(arr)

    blocks = [_stack_sequence(groups[grp], bg) for grp in group_order]
    return _stack_blocks(blocks, bg)


def _transform(grid: np.ndarray, blue_orient: str, corner: str) -> np.ndarray:
    if blue_orient == "h":
        if corner == "tl":
            return grid
        if corner == "tr":
            return np.fliplr(grid)
        if corner == "bl":
            return np.flipud(grid)
        return np.rot90(grid, 2)

    base = np.rot90(grid, 1)
    if corner == "bl":
        return base
    if corner == "tl":
        return np.flipud(base)
    if corner == "tr":
        return np.rot90(grid, 3)
    return np.flipud(np.rot90(grid, 3))


def _decorate_patch(h: int, w: int, dominant: int) -> np.ndarray:
    patch = np.full((h, w), int(dominant), dtype=int)
    area = int(h * w)
    max_minor = min(2, max(0, (area - 1) // 3))
    if max_minor == 0 or np.random.rand() < 0.45:
        return patch
    candidates = [(0, 0), (h - 1, w - 1), (h - 1, 0), (0, w - 1)]
    n_minor = int(np.random.randint(1, max_minor + 1))
    cells = np.random.permutation(len(candidates))[:n_minor]
    colors = np.random.choice(MINORITY_COLORS, size=n_minor, replace=True)
    for idx, color in zip(cells, colors):
        r, c = candidates[int(idx)]
        patch[int(r), int(c)] = int(color)
    return patch


def _sample_specs(n_top: int, n_bottom: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    widths = np.arange(2, 7, dtype=int)
    for _ in range(200):
        top_ws = [int(x) for x in np.random.permutation(widths)[:n_top]]
        bot_ws = [int(x) for x in np.random.permutation(widths)[:n_bottom]]
        areas = [TOP_H * w for w in top_ws] + [BOTTOM_H * w for w in bot_ws]
        if len(set(areas)) != len(areas):
            continue
        top_specs = [(TOP_H, w) for w in top_ws]
        bottom_specs = [(BOTTOM_H, w) for w in bot_ws]
        return top_specs, bottom_specs
    return None


def _run_count(seq: list[int]) -> int:
    runs = 0
    prev = None
    for x in seq:
        if x != prev:
            runs += 1
            prev = x
    return int(runs)


def _build_color_sequence(n_top: int, n_bottom: int) -> list[int]:
    total = n_top + n_bottom
    n_groups = 3 if total >= 5 else 2
    colors = [int(c) for c in np.random.choice(DOMINANT_COLORS, size=n_groups, replace=False)]

    if total == 5:
        counts = [2, 2, 1]
    elif total == 6:
        counts = [2, 2, 2] if np.random.rand() < 0.7 else [3, 2, 1]
    else:
        counts = [2, 2]
        for _ in range(total - 4):
            counts[int(np.random.randint(len(counts)))] += 1

    bag = []
    for color, count in zip(colors, counts):
        bag.extend([int(color)] * int(count))

    for _ in range(300):
        seq = [int(x) for x in np.random.permutation(bag)]
        top = seq[:n_top]
        bottom = seq[n_top:]
        count_map = {color: seq.count(color) for color in set(seq)}
        repeat_groups = sum(int(v >= 2) for v in count_map.values())
        if len(set(seq)) < 2:
            continue
        if repeat_groups < 2:
            continue
        if len(set(top)) < 2 or len(set(bottom)) < 2:
            continue
        if not any((color in top) and (color in bottom) for color in set(seq)):
            continue
        if _run_count(seq) <= len(set(seq)):
            continue
        return seq
    raise RuntimeError("failed to build color sequence")


def _place_band(
    grid: np.ndarray,
    row: int,
    specs: list[tuple[int, int]],
    colors: list[int],
    start_idx: int,
) -> list[dict]:
    rects = []
    c = 2
    for offset, ((h, w), dominant) in enumerate(zip(specs, colors)):
        patch = _decorate_patch(int(h), int(w), int(dominant))
        paste(grid, patch, int(row), int(c))
        rects.append(
            {
                "id": int(start_idx + offset),
                "patch": patch,
                "area": int(h * w),
                "dominant": int(dominant),
                "box": (int(row), int(c), int(row + h - 1), int(c + w - 1)),
            }
        )
        c += int(w) + 2
    return rects


def _render_single_rect(
    patch: np.ndarray,
    box: tuple[int, int, int, int],
    canonical_shape: tuple[int, int],
    blue_orient: str,
    corner: str,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    canvas = np.full(canonical_shape, BG, dtype=int)
    r0, c0, _r1, _c1 = map(int, box)
    paste(canvas, patch, int(r0), int(c0))
    transformed = _transform(canvas, blue_orient, corner)
    coords = np.argwhere(transformed != BG)
    bbox = bbox_coords(coords)
    return crop_bbox(transformed, bbox), tuple(map(int, bbox))


def _corner_coords(shape: tuple[int, int], corner: str) -> tuple[int, int]:
    h, w = map(int, shape)
    mapping = {
        "tl": (0, 0),
        "tr": (0, int(w - 1)),
        "bl": (int(h - 1), 0),
        "br": (int(h - 1), int(w - 1)),
    }
    return mapping[corner]


def _distance_order_ids(rects: list[dict], input_shape: tuple[int, int], blue_orient: str, corner: str) -> list[int]:
    cr, cc = _corner_coords(input_shape, corner)
    red_orient = "v" if blue_orient == "h" else "h"
    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    red_edge_idx = int(cc if red_orient == "v" else cr)
    ordered = sorted(
        rects,
        key=lambda r: (
            _dist_to_edge(r["input_box"], blue_orient, blue_edge_idx),
            _dist_to_edge(r["input_box"], red_orient, red_edge_idx),
            _along_from_corner(r["input_box"], blue_orient, cr, cc),
            _along_from_corner(r["input_box"], red_orient, cr, cc),
            int(r["id"]),
        ),
    )
    return [int(r["id"]) for r in ordered]


def _area_order_ids(rects: list[dict], input_shape: tuple[int, int], blue_orient: str, corner: str) -> list[int]:
    cr, cc = _corner_coords(input_shape, corner)
    red_orient = "v" if blue_orient == "h" else "h"
    ordered = sorted(
        rects,
        key=lambda r: (
            int(r["area"]),
            _along_from_corner(r["input_box"], blue_orient, cr, cc),
            _along_from_corner(r["input_box"], red_orient, cr, cc),
            int(r["id"]),
        ),
    )
    return [int(r["id"]) for r in ordered]


def _build_attempt() -> tuple[np.ndarray, np.ndarray, dict] | None:
    count_options = [(2, 3), (3, 2), (3, 3)]
    n_top, n_bottom = count_options[int(np.random.randint(len(count_options)))]

    specs = _sample_specs(n_top, n_bottom)
    if specs is None:
        return None
    top_specs, bottom_specs = specs

    color_seq = _build_color_sequence(n_top, n_bottom)
    top_colors = color_seq[:n_top]
    bottom_colors = color_seq[n_top:]

    need_w_top = 2 + sum(w for _h, w in top_specs) + 2 * (len(top_specs) - 1) + int(np.random.randint(1, 4))
    need_w_bottom = 2 + sum(w for _h, w in bottom_specs) + 2 * (len(bottom_specs) - 1) + int(np.random.randint(1, 4))
    width = int(max(need_w_top, need_w_bottom))

    canonical = np.full((CANONICAL_H, width), BG, dtype=int)
    canonical[0, :] = BLUE
    canonical[:, 0] = RED
    canonical[0, 0] = MARKER

    rects = []
    rects.extend(_place_band(canonical, TOP_ROW, top_specs, top_colors, 0))
    rects.extend(_place_band(canonical, BOTTOM_ROW, bottom_specs, bottom_colors, len(rects)))

    blue_orient = str(np.random.choice(["h", "v"]))
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    inp = _transform(canonical, blue_orient, corner)

    transformed_rects = []
    for rect in rects:
        input_patch, input_box = _render_single_rect(rect["patch"], rect["box"], canonical.shape, blue_orient, corner)
        transformed_rects.append(
            {
                "id": int(rect["id"]),
                "area": int(rect["area"]),
                "dominant": int(rect["dominant"]),
                "input_patch": input_patch,
                "input_box": input_box,
            }
        )

    rects_by_id = {int(r["id"]): r for r in transformed_rects}
    distance_ids = _distance_order_ids(transformed_rects, inp.shape, blue_orient, corner)
    area_ids = _area_order_ids(transformed_rects, inp.shape, blue_orient, corner)

    combo_out = _compose_output(rects_by_id, area_ids, blue_orient, BG, group_by_color=True)
    distance_group_out = _compose_output(rects_by_id, distance_ids, blue_orient, BG, group_by_color=True)
    area_single_out = _compose_output(rects_by_id, area_ids, blue_orient, BG, group_by_color=False)

    counts = {}
    for color in color_seq:
        counts[int(color)] = int(counts.get(int(color), 0) + 1)
    repeat_groups = sum(int(v >= 2) for v in counts.values())

    if len(counts) < 2 or repeat_groups < 2:
        return None
    if combo_out.shape == area_single_out.shape and np.array_equal(combo_out, area_single_out):
        return None
    if combo_out.shape == distance_group_out.shape and np.array_equal(combo_out, distance_group_out):
        return None

    actual = main(inp)
    if not np.array_equal(actual, combo_out):
        return None

    meta = {
        "blue_orient": blue_orient,
        "corner": corner,
        "area_ids": area_ids,
        "distance_ids": distance_ids,
        "group_count": len(counts),
        "repeat_groups": repeat_groups,
        "combo_differs_from_distance_group": not (
            combo_out.shape == distance_group_out.shape and np.array_equal(combo_out, distance_group_out)
        ),
        "combo_differs_from_area_single": not (
            combo_out.shape == area_single_out.shape and np.array_equal(combo_out, area_single_out)
        ),
    }
    return inp, combo_out, meta


def generate_input() -> np.ndarray:
    for _ in range(800):
        built = _build_attempt()
        if built is not None:
            inp, _expected, _meta = built
            return inp
    raise RuntimeError("failed to generate input for 291dc1e1 combo D1/0+D13/0")
