# Combo variant of 291dc1e1 (index 24)
# combo_spec: D1/0+D3/2
# Name: 面积排序+色组矩形

from __future__ import annotations

import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
ROOT = None
for parent in HERE.parents:
    if (parent / "solvers" / "arc_utils.py").exists():
        ROOT = parent
        break
assert ROOT is not None
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste


"""
Design analysis for combo variant D1/0 + D3/2

a) Concrete behavior and trace
- Original task: every interior rectangle is treated as an independent object,
  sorted by `(distance to blue, distance to red, tie-breaks from the corner)`,
  then stacked. If blue is vertical, each rectangle is rotated 90 degrees CCW.
- This combo adds two coupled changes.
  1. Same-colored rectangles form a group.
  2. Inside each group, rectangles are sorted by area instead of by the original
     distance key.
  3. Whole groups are still ordered by the original geometric key of the first
     member that would appear in the input's geometric scan order.
- Example in canonical order: slot sequence
  `green 2x6, yellow 2x3, green 3x2, yellow 3x5`.
  Original output order is the same slot order.
  Pure color-grouping would become `green 2x6, green 3x2, yellow 2x3, yellow 3x5`.
  Pure area-sorting would become `yellow 2x3, green 3x2, green 2x6, yellow 3x5`.
  Combo output becomes `green 3x2, green 2x6, yellow 2x3, yellow 3x5`:
  groups are `green` then `yellow` by geometric lead member, but inside each
  group the smaller-area member comes first.

b) Complexity calibration
- To keep area meaningfully different from width, the two source bands use
  heights 2 and 3 instead of a single shared height.
- To keep color grouping visible, only 2 rectangle colors are used per sample
  and each color appears at least twice.
- To avoid hidden ambiguity, every rectangle is monochrome and every sample is
  rejected if the combo order collapses to any simpler baseline.

c) Implicit-assumption audit on the original solver
- Corner detection assumes one marked corner adjacent to two full edge lines.
  The generator preserves that exactly.
- Blue-line detection assumes color 1 is the blue edge when present. Preserved.
- Full-row/full-column masking assumes rectangles never touch the axis lines.
  The generator keeps a one-cell gap from both axes.
- Rectangle extraction assumes each component fills its bounding box with no
  background holes. Preserved by using solid monochrome rectangles.
- Original `rects.sort(...)` assumed one flat ordering rule. That no longer
  holds: the solver must compute both a geometric key and an area key, then use
  geometry only for ordering groups and area only for ordering inside groups.
- Rotation remains valid after grouping because rotation changes orientation but
  not color membership or area.

d) Placement legality
- Canonical grids use a fixed left axis column and top axis row, then place the
  top band at rows `2:4` and the bottom band at rows `6:9`.
- Rectangles are separated by 2 background columns inside a band, so components
  never touch or merge.
- Grid width is derived from actual used width plus a trailing border, so every
  paste stays in bounds. Whole-grid flips/rotations preserve legality.

e) Visual observability
- A human can see repeated colors in the input, then see contiguous color blocks
  in the output. That exposes the grouping layer.
- Inside a color block, the smaller rectangle appears above the larger one. That
  exposes the area-ordering layer.
- The generator rejects samples where repeated colors are already contiguous in
  the original geometric order or where within-group area order already matches
  the geometric order for every group.

f) Visual cleanliness
- The scene contains only the two axis lines, one corner marker, and solid
  rectangles. No internal decorations are added, because grouping depends on
  color identity and should stay visually obvious.
- Using only 2 repeated rectangle colors keeps the image readable even when the
  rectangles are numerous.

g) Color-change inferability
- No recoloring happens. Input colors persist to the output and act only as
  grouping labels.
- Restricting to 2 repeated rectangle colors gives enough repetition for humans
  to infer grouping from 2-4 examples without adding noisy extra colors.

h) Generator validation and anti-degeneration checks
- For every attempt, the generator derives transformed per-rectangle metadata and
  constructs the intended combo output independently from that metadata.
- It rejects any sample whose combo order matches:
  1. the original flat geometric order,
  2. the pure flat area order, or
  3. the D3-only grouped-by-color geometric-inside-group order.
- It also requires the original geometric color sequence to have more runs than
  the number of color groups, so the grouping effect is visible.

i) New solver assumptions after the edit
- Each meaningful component must have a stable dominant color; the generator uses
  monochrome rectangles so dominant color is unambiguous.
- Area must be distinguishable inside at least one color group; the generator
  guarantees this by rejecting attempts where no group's area order changes.
- Group order assumes each color group has a unique geometric leader. That is
  guaranteed because rectangles have distinct bounding boxes and the geometric
  key includes spatial tie-breaks.
"""


BG = 8
BLUE = 1
RED = 2
MARKER = 0
RECT_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
WIDTH_OPTIONS = np.arange(2, 7, dtype=int)

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
        red_col = int(vcol if hcol == BLUE else hcol)
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
        if vlen >= hlen:
            blue_col = int(vcol)
            red_col = int(hcol)
        else:
            blue_col = int(hcol)
            red_col = int(vcol)

    blue_orient = "h" if int(hcol) == int(blue_col) else "v"
    red_orient = "h" if int(hcol) == int(red_col) else "v"
    return int(bg), (int(cr), int(cc)), int(marker), blue_orient, red_orient


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


def _along_from_corner(box: tuple[int, int, int, int], orient: str, cr: int, cc: int) -> int:
    r0, c0, r1, c1 = map(int, box)
    if orient == "v":
        return int(r0 if cr == 0 else -r1)
    return int(c0 if cc == 0 else -c1)


def _extract_rectangles(grid: np.ndarray) -> tuple[int, tuple[int, int], str, str, list[dict]]:
    g = np.array(grid, dtype=int)
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
        box = tuple(map(int, bbox_coords(coords)))
        sub = crop_bbox(g, box)
        if bool((sub == int(bg)).any()):
            continue
        vals = [int(v) for v in sub.reshape(-1) if int(v) != int(bg)]
        assert vals
        rects.append(
            {
                "box": box,
                "arr": sub.copy(),
                "area": int(np.count_nonzero(sub != int(bg))),
                "color": int(Counter(vals).most_common(1)[0][0]),
            }
        )

    assert rects
    return int(bg), (int(cr), int(cc)), blue_orient, red_orient, rects


def _annotate_rectangles(rects: list[dict], cr: int, cc: int, blue_orient: str, red_orient: str) -> None:
    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    red_edge_idx = int(cc if red_orient == "v" else cr)
    for rect in rects:
        box = tuple(map(int, rect["box"]))
        along_blue = int(_along_from_corner(box, blue_orient, cr, cc))
        along_red = int(_along_from_corner(box, red_orient, cr, cc))
        rect["geom_key"] = (
            int(_dist_to_edge(box, blue_orient, blue_edge_idx)),
            int(_dist_to_edge(box, red_orient, red_edge_idx)),
            along_blue,
            along_red,
            box,
        )
        rect["area_key"] = (
            int(rect["area"]),
            along_blue,
            along_red,
            box,
        )


def _ordered_rectangles(rects: list[dict], *, group_by_color: bool, intra_sort: str) -> list[dict]:
    assert intra_sort in {"geom", "area"}
    if not group_by_color:
        flat_key = "geom_key" if intra_sort == "geom" else "area_key"
        return sorted(rects, key=lambda rect: rect[flat_key])

    grouped: dict[int, list[dict]] = {}
    for rect in rects:
        grouped.setdefault(int(rect["color"]), []).append(rect)

    groups = []
    for color, items in grouped.items():
        items_sorted = sorted(items, key=lambda rect: rect["geom_key" if intra_sort == "geom" else "area_key"])
        group_key = min((rect["geom_key"] for rect in items), key=lambda key: key)
        groups.append((group_key, int(color), items_sorted))

    groups.sort(key=lambda item: (item[0], int(item[1])))
    ordered = []
    for _group_key, _color, items in groups:
        ordered.extend(items)
    return ordered


def _stack_sequence(seq: list[np.ndarray], bg: int) -> np.ndarray:
    out_w = int(max(int(arr.shape[1]) for arr in seq))
    out_h = int(sum(int(arr.shape[0]) for arr in seq))
    out = np.full((out_h, out_w), int(bg), dtype=int)
    out_r = 0
    for arr in seq:
        h, w = map(int, arr.shape)
        out_c = int((out_w - w) // 2)
        paste(out, arr, int(out_r), int(out_c))
        out_r += int(h)
    return out


def _solve(grid: np.ndarray, *, group_by_color: bool, intra_sort: str) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg, (cr, cc), blue_orient, red_orient, rects = _extract_rectangles(g)
    _annotate_rectangles(rects, cr, cc, blue_orient, red_orient)
    ordered = _ordered_rectangles(rects, group_by_color=group_by_color, intra_sort=intra_sort)

    seq = []
    for rect in ordered:
        arr = rect["arr"]
        if blue_orient == "v":
            arr = np.rot90(arr, k=1)
        seq.append(arr)
    return _stack_sequence(seq, bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - Detect the same interior rectangles as in the source task.
    - Group rectangles by dominant color.
    - Order the color groups by the earliest member of each group under the
      original geometric `(distance to blue, distance to red, tie-breaks)` rule.
    - Within each color group, sort rectangles by area from small to large.
    - If the blue edge is vertical, rotate every rectangle 90 degrees CCW.
    - Stack the resulting sequence top-to-bottom, centered horizontally.
    """
    return _solve(grid, group_by_color=True, intra_sort="area")


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


def _sample_band_counts() -> tuple[int, int]:
    options = [(2, 3), (3, 2), (2, 2)]
    return tuple(map(int, options[int(np.random.randint(len(options)))]))


def _sample_band_specs() -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    n_top, n_bottom = _sample_band_counts()
    top_widths = [int(w) for w in np.random.permutation(WIDTH_OPTIONS)[:n_top]]
    bottom_widths = [int(w) for w in np.random.permutation(WIDTH_OPTIONS)[:n_bottom]]
    top_specs = [(TOP_H, int(w)) for w in top_widths]
    bottom_specs = [(BOTTOM_H, int(w)) for w in bottom_widths]
    all_areas = [int(h * w) for h, w in top_specs + bottom_specs]
    if len(set(all_areas)) != len(all_areas):
        return None
    non_square = sum(int(h != w) for h, w in top_specs + bottom_specs)
    if non_square < 2:
        return None
    return top_specs, bottom_specs


def _runs(seq: list[int]) -> int:
    if not seq:
        return 0
    return 1 + sum(int(seq[i] != seq[i - 1]) for i in range(1, len(seq)))


def _choose_slot_colors(slot_areas: list[int]) -> list[int] | None:
    total = len(slot_areas)
    best_labels = None
    best_score = None
    size_options = [(2, total - 2)]

    for size0, _size1 in size_options:
        for combo in combinations(range(total), size0):
            labels = [1] * total
            for idx in combo:
                labels[int(idx)] = 0
            run_count = _runs(labels)
            if run_count <= 2:
                continue

            disorder_groups = 0
            for label in (0, 1):
                group_areas = [int(slot_areas[i]) for i, cur in enumerate(labels) if int(cur) == label]
                if group_areas != sorted(group_areas):
                    disorder_groups += 1

            score = (int(disorder_groups), int(run_count))
            if best_score is None or score > best_score:
                best_score = score
                best_labels = list(labels)

    if best_labels is None or int(best_score[0]) == 0:
        return None

    colors = [int(c) for c in np.random.choice(RECT_COLORS, size=2, replace=False)]
    return [int(colors[int(label)]) for label in best_labels]


def _place_band(
    grid: np.ndarray,
    row: int,
    specs: list[tuple[int, int]],
    colors: list[int],
    start_idx: int,
) -> list[dict]:
    rects = []
    c = 2
    for offset, (h, w) in enumerate(specs):
        color = int(colors[start_idx + offset])
        patch = np.full((int(h), int(w)), int(color), dtype=int)
        paste(grid, patch, int(row), int(c))
        rects.append(
            {
                "id": int(start_idx + offset),
                "patch": patch,
                "box": (int(row), int(c), int(row + h - 1), int(c + w - 1)),
                "area": int(h * w),
                "color": int(color),
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
    bbox = tuple(map(int, bbox_coords(coords)))
    return crop_bbox(transformed, bbox), bbox


def _corner_coords(shape: tuple[int, int], corner: str) -> tuple[int, int]:
    h, w = map(int, shape)
    mapping = {
        "tl": (0, 0),
        "tr": (0, int(w - 1)),
        "bl": (int(h - 1), 0),
        "br": (int(h - 1), int(w - 1)),
    }
    return mapping[corner]


def _ordered_ids_from_meta(
    rects: list[dict],
    input_shape: tuple[int, int],
    blue_orient: str,
    corner: str,
    *,
    group_by_color: bool,
    intra_sort: str,
) -> list[int]:
    assert intra_sort in {"geom", "area"}
    cr, cc = _corner_coords(input_shape, corner)
    red_orient = "v" if blue_orient == "h" else "h"
    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    red_edge_idx = int(cc if red_orient == "v" else cr)

    annotated = []
    for rect in rects:
        box = tuple(map(int, rect["input_box"]))
        along_blue = int(_along_from_corner(box, blue_orient, cr, cc))
        along_red = int(_along_from_corner(box, red_orient, cr, cc))
        geom_key = (
            int(_dist_to_edge(box, blue_orient, blue_edge_idx)),
            int(_dist_to_edge(box, red_orient, red_edge_idx)),
            along_blue,
            along_red,
            box,
        )
        area_key = (
            int(rect["area"]),
            along_blue,
            along_red,
            box,
        )
        annotated.append(
            {
                "id": int(rect["id"]),
                "color": int(rect["color"]),
                "geom_key": geom_key,
                "area_key": area_key,
            }
        )

    if not group_by_color:
        flat_key = "geom_key" if intra_sort == "geom" else "area_key"
        ordered = sorted(annotated, key=lambda rect: rect[flat_key])
        return [int(rect["id"]) for rect in ordered]

    grouped: dict[int, list[dict]] = {}
    for rect in annotated:
        grouped.setdefault(int(rect["color"]), []).append(rect)

    groups = []
    for color, items in grouped.items():
        items_sorted = sorted(items, key=lambda rect: rect["geom_key" if intra_sort == "geom" else "area_key"])
        group_key = min((rect["geom_key"] for rect in items), key=lambda key: key)
        groups.append((group_key, int(color), items_sorted))

    groups.sort(key=lambda item: (item[0], int(item[1])))
    ordered_ids = []
    for _group_key, _color, items in groups:
        ordered_ids.extend(int(rect["id"]) for rect in items)
    return ordered_ids


def _color_runs_for_ids(ids: list[int], rect_by_id: dict[int, dict]) -> int:
    return _runs([int(rect_by_id[int(rid)]["color"]) for rid in ids])


def _intra_group_area_reorders(
    rects: list[dict],
    input_shape: tuple[int, int],
    blue_orient: str,
    corner: str,
) -> int:
    geom_ids = _ordered_ids_from_meta(
        rects, input_shape, blue_orient, corner, group_by_color=False, intra_sort="geom"
    )
    area_ids = _ordered_ids_from_meta(
        rects, input_shape, blue_orient, corner, group_by_color=False, intra_sort="area"
    )
    rect_by_id = {int(rect["id"]): rect for rect in rects}

    changed = 0
    for color in sorted({int(rect["color"]) for rect in rects}):
        geom_group = [int(rid) for rid in geom_ids if int(rect_by_id[int(rid)]["color"]) == int(color)]
        area_group = [int(rid) for rid in area_ids if int(rect_by_id[int(rid)]["color"]) == int(color)]
        if geom_group != area_group:
            changed += 1
    return int(changed)


def _build_attempt() -> tuple[np.ndarray, np.ndarray, dict] | None:
    sampled = _sample_band_specs()
    if sampled is None:
        return None
    top_specs, bottom_specs = sampled
    slot_specs = top_specs + bottom_specs
    slot_areas = [int(h * w) for h, w in slot_specs]
    slot_colors = _choose_slot_colors(slot_areas)
    if slot_colors is None:
        return None

    need_w_top = 2 + sum(int(w) for _h, w in top_specs) + 2 * (len(top_specs) - 1) + int(np.random.randint(1, 4))
    need_w_bottom = 2 + sum(int(w) for _h, w in bottom_specs) + 2 * (len(bottom_specs) - 1) + int(np.random.randint(1, 4))
    width = int(max(need_w_top, need_w_bottom))

    canonical = np.full((CANONICAL_H, width), BG, dtype=int)
    canonical[0, :] = BLUE
    canonical[:, 0] = RED
    canonical[0, 0] = MARKER

    rects = []
    rects.extend(_place_band(canonical, TOP_ROW, top_specs, slot_colors, 0))
    rects.extend(_place_band(canonical, BOTTOM_ROW, bottom_specs, slot_colors, len(top_specs)))

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
                "color": int(rect["color"]),
                "input_patch": input_patch,
                "input_box": input_box,
            }
        )

    flat_geom_ids = _ordered_ids_from_meta(
        transformed_rects, inp.shape, blue_orient, corner, group_by_color=False, intra_sort="geom"
    )
    flat_area_ids = _ordered_ids_from_meta(
        transformed_rects, inp.shape, blue_orient, corner, group_by_color=False, intra_sort="area"
    )
    group_geom_ids = _ordered_ids_from_meta(
        transformed_rects, inp.shape, blue_orient, corner, group_by_color=True, intra_sort="geom"
    )
    combo_ids = _ordered_ids_from_meta(
        transformed_rects, inp.shape, blue_orient, corner, group_by_color=True, intra_sort="area"
    )

    if combo_ids == flat_geom_ids or combo_ids == flat_area_ids or combo_ids == group_geom_ids:
        return None

    rect_by_id = {int(rect["id"]): rect for rect in transformed_rects}
    color_groups = len({int(rect["color"]) for rect in transformed_rects})
    if _color_runs_for_ids(flat_geom_ids, rect_by_id) <= color_groups:
        return None

    intra_changed = _intra_group_area_reorders(transformed_rects, inp.shape, blue_orient, corner)
    if intra_changed < 1:
        return None

    seq = []
    for rid in combo_ids:
        arr = rect_by_id[int(rid)]["input_patch"]
        if blue_orient == "v":
            arr = np.rot90(arr, k=1)
        seq.append(arr)
    expected = _stack_sequence(seq, BG)

    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None

    meta = {
        "blue_orient": blue_orient,
        "corner": corner,
        "rect_count": int(len(rects)),
        "group_count": int(color_groups),
        "flat_geom_ids": [int(x) for x in flat_geom_ids],
        "flat_area_ids": [int(x) for x in flat_area_ids],
        "group_geom_ids": [int(x) for x in group_geom_ids],
        "combo_ids": [int(x) for x in combo_ids],
        "flat_geom_color_runs": int(_color_runs_for_ids(flat_geom_ids, rect_by_id)),
        "intra_group_area_reorders": int(intra_changed),
    }
    return inp, expected, meta


def generate_input() -> np.ndarray:
    for _ in range(800):
        built = _build_attempt()
        if built is not None:
            inp, _expected, _meta = built
            return inp
    raise RuntimeError("failed to generate input for 291dc1e1 combo variant 24")
