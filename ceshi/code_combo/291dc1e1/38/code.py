# Combo variant of 291dc1e1 (index 38)
# combo_spec: D1/0+D3/2+D13/0
# Name: 面积排序+色组+分色输出

from __future__ import annotations

from collections import Counter

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste


"""
Design analysis for combo D1/0 + D3/2 + D13/0

a) Concrete behavior and trace
- The original task detects the corner marker plus the two edge lines, extracts the
  interior rectangles, sorts all rectangles in one flat list by
  `(distance to blue edge, distance to red edge, tie-breaks along the corner)`,
  optionally rotates each rectangle 90 degrees CCW when the blue edge is vertical,
  then stacks everything into one single column.
- This combo keeps the same scene skeleton and the same rotation trigger, but changes
  both ordering and assembly:
  1. Rectangles are grouped by dominant fill color.
  2. Inside each color group, rectangles are sorted by area from small to large.
  3. Each color group is stacked independently into its own vertical block.
  4. The blocks are placed side by side from left to right.
- Concrete example in canonical orientation:
    slot order under the original geometry = [A(red, area 10), B(green, area 8),
                                             C(red, area 4),  D(green, area 15)]
  Original output: a single stack `A, B, C, D`.
  Combo output:
    red group   -> `C(area 4), A(area 10)`
    green group -> `B(area 8), D(area 15)`
    block order -> red block first because its smallest member (area 4) is globally
                   earlier than green's smallest member (area 8).
  The final output is therefore two columns: `[C over A] [B over D]`.
- The visual difference is therefore threefold at once: color now defines grouping,
  area replaces distance as the within-group ordering key, and the target is no
  longer one many-to-one stack but several parallel stacks.

b) Complexity calibration
- If every rectangle had the same height, "area order" would collapse to width order,
  which is too close to the old along-band geometry. To avoid that, the generator
  uses two bands of different fixed heights: top-band rectangles are 2 high and
  bottom-band rectangles are 3 high.
- To keep the scene clean, only 2 or 3 rectangle colors are used, and every chosen
  color appears at least twice. This gives enough support for humans to infer the
  grouping rule without flooding the puzzle with too many categories.
- Areas are unique within a sample. That removes ambiguous ties and makes the area
  cue visually decisive.
- The generator rejects samples where any key combo dimension degenerates:
  grouped-area output must differ from grouped-distance output, and the flattened
  grouped-area sequence must differ from the plain flat area sequence.

c) Implicit-assumption audit of the original solver
- Corner detection still assumes exactly one corner marker different from both
  neighboring line colors. The generator preserves that.
- Blue-line detection still assumes color 1 is the blue line when present; the
  generator preserves blue=1 and red=2.
- Full row/column masking around the corner still assumes rectangles never touch the
  edge lines. The generator preserves at least one background gap from both lines.
- Rectangle extraction assumes each connected component fills its bbox with no
  background holes. The generator keeps every object as a solid rectangle.
- The original `rects.sort(...)` assumed a single global flat order and distance as
  the primary key. In the combo that assumption fails twice: the primary key becomes
  area, and ordering is only meaningful inside a color group.
- The original single `seq` and single output canvas assumed many:1. In the combo,
  each color group must first be assembled into its own block, then those blocks are
  concatenated horizontally with a 1-cell background gap.
- A new combo-specific assumption is that each rectangle has a stable dominant color.
  The generator guarantees this by using monochrome rectangles.
- Another new assumption is that group order is inferable. The solver orders groups
  by the first member under the same area key, with color used only as a deterministic
  final tie-break. The generator keeps areas unique so this tie-break is rarely active.

d) Placement legality
- Canonical generation uses a fixed 11-row scaffold:
  row 0 = blue edge, col 0 = red edge, top band at rows 2:4, bottom band at rows 6:9.
- Rectangles inside a band are separated by 2 background columns and start at column 2,
  so components never touch or overlap.
- Grid width is derived from the actual band usage plus 2-3 trailing background cells,
  so every paste stays in bounds.
- Final flips/rotations are whole-grid transforms, so they preserve non-overlap and
  keep the edge/corner structure valid.

e) Visual observability
- From a few examples, a human can see that equal-colored rectangles gather into the
  same output column, not just near each other.
- Inside a column, smaller rectangles appear above larger same-colored ones, even when
  the original spatial order for that color is the opposite.
- Different colors become different parallel output blocks, so D13 is directly visible
  in the output shape itself.

f) Visual cleanliness
- Rectangles stay plain solid rectangles; there are no internal decorations that might
  blur the grouping signal.
- Only the original edge lines, the corner marker, and the rectangles are present.
- Output blocks are separated by exactly one background column, so adjacent color groups
  remain visually distinct.

g) Color-change inferability
- There is no recoloring. Color is a stable identity feature used only for grouping.
- Because colors repeat and remain unchanged from input to output, the grouping rule is
  structurally inferable from the samples.

h) Generator validation
- Each attempt builds metadata for every rectangle, transforms the whole scene, derives
  the intended grouped-area multi-block output from the metadata, and checks it against
  `main(input)`.
- The attempt is rejected if any combo dimension is inactive on that sample.

i) New solver assumptions after the edit
- The solver assumes every meaningful component is a solid rectangle whose bbox area is
  the intended area signal. The generator guarantees solid rectangles with no holes.
- The solver assumes there are at least two color groups for the multi-block output to
  make sense visually. The generator guarantees that.
- The solver assumes rectangles of the same dominant color should be grouped, even if
  they are spatially separated. The generator makes repeated colors explicit and clean.
"""


BG = 8
BLUE = 1
RED = 2
MARKER = 0
RECT_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)

TOP_ROW = 2
TOP_H = 2
BOTTOM_ROW = 6
BOTTOM_H = 3

TOP_WIDTHS = np.array([2, 4, 5, 6], dtype=int)      # areas: 4,8,10,12
BOTTOM_WIDTHS = np.array([2, 3, 5, 6], dtype=int)   # areas: 6,9,15,18


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


def _dominant_color(arr: np.ndarray) -> int:
    vals = [int(v) for v in arr.flatten()]
    return int(Counter(vals).most_common(1)[0][0])


def _extract_rectangles(g: np.ndarray) -> tuple[int, tuple[int, int], str, str, list[dict]]:
    bg, (cr, cc), marker, blue_orient, red_orient = _locate_frame(g)
    H, W = g.shape
    line_mask = np.zeros((H, W), dtype=bool)
    line_mask[cr, :] = True
    line_mask[:, cc] = True

    mask = (g != int(bg)) & (~line_mask) & (g != int(marker))
    rects: list[dict] = []
    for coords in components_mask(mask, connectivity=4):
        if coords.size == 0:
            continue
        box = tuple(map(int, bbox_coords(coords)))
        sub = crop_bbox(g, box)
        if bool((sub == int(bg)).any()):
            continue
        rects.append(
            {
                "box": box,
                "arr": sub.copy(),
                "area": int(sub.shape[0] * sub.shape[1]),
                "color": int(_dominant_color(sub)),
            }
        )

    assert rects
    return int(bg), (int(cr), int(cc)), blue_orient, red_orient, rects


def _along_from_corner(box: tuple[int, int, int, int], orient: str, cr: int, cc: int) -> int:
    r0, c0, r1, c1 = map(int, box)
    if orient == "v":
        return int(r0 if cr == 0 else -r1)
    return int(c0 if cc == 0 else -c1)


def _dist_to_edge(box: tuple[int, int, int, int], orient: str, edge_idx: int) -> int:
    r0, c0, r1, c1 = map(int, box)
    if orient == "v":
        if c0 > edge_idx:
            return int(c0 - edge_idx - 1)
        if c1 < edge_idx:
            return int(edge_idx - c1 - 1)
        return 0
    if r0 > edge_idx:
        return int(r0 - edge_idx - 1)
    if r1 < edge_idx:
        return int(edge_idx - r1 - 1)
    return 0


def _area_key(rect: dict, blue_orient: str, red_orient: str, cr: int, cc: int) -> tuple[int, int, int, tuple[int, int, int, int]]:
    box = tuple(map(int, rect["box"]))
    return (
        int(rect["area"]),
        int(_along_from_corner(box, blue_orient, cr, cc)),
        int(_along_from_corner(box, red_orient, cr, cc)),
        box,
    )


def _distance_key(
    rect: dict, blue_orient: str, red_orient: str, blue_edge_idx: int, red_edge_idx: int, cr: int, cc: int
) -> tuple[int, int, int, int, tuple[int, int, int, int]]:
    box = tuple(map(int, rect["box"]))
    return (
        int(_dist_to_edge(box, blue_orient, blue_edge_idx)),
        int(_dist_to_edge(box, red_orient, red_edge_idx)),
        int(_along_from_corner(box, blue_orient, cr, cc)),
        int(_along_from_corner(box, red_orient, cr, cc)),
        box,
    )


def _stack_sequence(seq: list[np.ndarray], bg: int) -> np.ndarray:
    out_w = int(max(int(a.shape[1]) for a in seq))
    out_h = int(sum(int(a.shape[0]) for a in seq))
    out = np.full((out_h, out_w), int(bg), dtype=int)
    out_r = 0
    for arr in seq:
        h, w = map(int, arr.shape)
        out_c = int((out_w - w) // 2)
        paste(out, arr, int(out_r), int(out_c))
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


def _solve(
    grid: np.ndarray,
    *,
    sort_mode: str,
    group_by_color: bool,
    multi_output: bool,
) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg, (cr, cc), blue_orient, red_orient, rects = _extract_rectangles(g)
    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    red_edge_idx = int(cc if red_orient == "v" else cr)

    for rect in rects:
        rect["area_key"] = _area_key(rect, blue_orient, red_orient, cr, cc)
        rect["distance_key"] = _distance_key(rect, blue_orient, red_orient, blue_edge_idx, red_edge_idx, cr, cc)
        rect["sort_key"] = rect["area_key"] if sort_mode == "area" else rect["distance_key"]

    if group_by_color:
        grouped: dict[int, list[dict]] = {}
        for rect in rects:
            grouped.setdefault(int(rect["color"]), []).append(rect)

        ordered_groups: list[list[dict]] = []
        group_meta = []
        for color, items in grouped.items():
            items.sort(key=lambda rect: rect["sort_key"])
            group_meta.append((items[0]["sort_key"], int(color), items))
        group_meta.sort(key=lambda item: (item[0], int(item[1])))
        for _first_key, _color, items in group_meta:
            ordered_groups.append(items)
    else:
        ordered_groups = [sorted(rects, key=lambda rect: rect["sort_key"])]

    if blue_orient == "v":
        for group in ordered_groups:
            for rect in group:
                rect["out_arr"] = np.rot90(rect["arr"], k=1)
    else:
        for group in ordered_groups:
            for rect in group:
                rect["out_arr"] = rect["arr"]

    if multi_output:
        blocks = []
        for group in ordered_groups:
            seq = [rect["out_arr"] for rect in group]
            blocks.append(_stack_sequence(seq, bg))
        return _stack_blocks(blocks, bg)

    seq = [rect["out_arr"] for group in ordered_groups for rect in group]
    return _stack_sequence(seq, bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - Detect the same interior rectangles and edge-frame as in the original task.
    - Group rectangles by dominant fill color.
    - Sort inside each color group by area from small to large.
    - Order color groups by the first member under that same area key.
    - If the blue edge is vertical, rotate each rectangle 90 degrees CCW.
    - Stack each group into its own vertical block and place the blocks left-to-right.
    """
    return _solve(grid, sort_mode="area", group_by_color=True, multi_output=True)


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
    choices = [(2, 2), (2, 3), (3, 2), (3, 3)]
    idx = int(np.random.randint(len(choices)))
    return choices[idx]


def _sample_group_sizes(total: int) -> list[int]:
    options = {
        4: [[2, 2]],
        5: [[2, 3], [3, 2]],
        6: [[3, 3], [2, 2, 2], [2, 4], [4, 2]],
    }
    candidates = [sizes for sizes in options[total] if min(sizes) >= 2]
    choice = candidates[int(np.random.randint(len(candidates)))]
    return [int(v) for v in choice]


def _run_count(seq: list[int]) -> int:
    runs = 0
    prev = None
    for value in seq:
        if value != prev:
            runs += 1
            prev = value
    return int(runs)


def _interleaved_color_sequence(group_sizes: list[int], colors: np.ndarray, n_top: int) -> list[int] | None:
    bag = []
    for color, size in zip(colors, group_sizes):
        bag.extend([int(color)] * int(size))
    total = len(bag)
    for _ in range(300):
        seq = [int(v) for v in np.random.permutation(bag)]
        if _run_count(seq) <= len(group_sizes):
            continue
        top = seq[:n_top]
        bottom = seq[n_top:]
        if len(set(top)) < 2 or len(set(bottom)) < 2:
            continue
        if not any(color in top and color in bottom for color in set(seq)):
            continue
        if total >= 5 and max(seq.count(color) for color in set(seq)) == total:
            continue
        return seq
    return None


def _corner_coords(shape: tuple[int, int], corner: str) -> tuple[int, int]:
    h, w = map(int, shape)
    mapping = {
        "tl": (0, 0),
        "tr": (0, int(w - 1)),
        "bl": (int(h - 1), 0),
        "br": (int(h - 1), int(w - 1)),
    }
    return mapping[corner]


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


def _meta_sort_keys(
    rects: list[dict], input_shape: tuple[int, int], blue_orient: str, corner: str
) -> tuple[list[dict], int, int, str]:
    cr, cc = _corner_coords(input_shape, corner)
    red_orient = "v" if blue_orient == "h" else "h"
    blue_edge_idx = int(cc if blue_orient == "v" else cr)
    red_edge_idx = int(cc if red_orient == "v" else cr)
    enriched = []
    for rect in rects:
        box = tuple(map(int, rect["input_box"]))
        area_key = (
            int(rect["area"]),
            int(_along_from_corner(box, blue_orient, cr, cc)),
            int(_along_from_corner(box, red_orient, cr, cc)),
            box,
        )
        distance_key = (
            int(_dist_to_edge(box, blue_orient, blue_edge_idx)),
            int(_dist_to_edge(box, red_orient, red_edge_idx)),
            int(_along_from_corner(box, blue_orient, cr, cc)),
            int(_along_from_corner(box, red_orient, cr, cc)),
            box,
        )
        item = dict(rect)
        item["area_key"] = area_key
        item["distance_key"] = distance_key
        enriched.append(item)
    return enriched, cr, cc, red_orient


def _grouped_ids(rects: list[dict], *, sort_mode: str) -> list[list[int]]:
    groups: dict[int, list[dict]] = {}
    for rect in rects:
        groups.setdefault(int(rect["color"]), []).append(rect)

    meta = []
    key_name = "area_key" if sort_mode == "area" else "distance_key"
    for color, items in groups.items():
        items = sorted(items, key=lambda rect: rect[key_name])
        meta.append((items[0][key_name], int(color), [int(r["id"]) for r in items]))
    meta.sort(key=lambda item: (item[0], int(item[1])))
    return [ids for _first_key, _color, ids in meta]


def _flat_ids(rects: list[dict], *, sort_mode: str) -> list[int]:
    key_name = "area_key" if sort_mode == "area" else "distance_key"
    return [int(rect["id"]) for rect in sorted(rects, key=lambda rect: rect[key_name])]


def _assemble_from_groups(rect_by_id: dict[int, dict], grouped_ids: list[list[int]], blue_orient: str) -> np.ndarray:
    blocks = []
    for ids in grouped_ids:
        seq = []
        for rid in ids:
            arr = rect_by_id[int(rid)]["input_patch"]
            if blue_orient == "v":
                arr = np.rot90(arr, k=1)
            seq.append(arr)
        blocks.append(_stack_sequence(seq, BG))
    return _stack_blocks(blocks, BG)


def _assemble_single_column(rect_by_id: dict[int, dict], ids: list[int], blue_orient: str) -> np.ndarray:
    seq = []
    for rid in ids:
        arr = rect_by_id[int(rid)]["input_patch"]
        if blue_orient == "v":
            arr = np.rot90(arr, k=1)
        seq.append(arr)
    return _stack_sequence(seq, BG)


def _build_canonical() -> tuple[np.ndarray, list[dict]] | None:
    n_top, n_bottom = _sample_band_counts()

    top_widths = [int(v) for v in np.random.permutation(TOP_WIDTHS)[:n_top]]
    bottom_widths = [int(v) for v in np.random.permutation(BOTTOM_WIDTHS)[:n_bottom]]
    total = int(n_top + n_bottom)

    group_sizes = _sample_group_sizes(total)
    colors = np.random.choice(RECT_COLORS, size=len(group_sizes), replace=False)
    color_seq = _interleaved_color_sequence(group_sizes, colors, n_top)
    if color_seq is None:
        return None

    need_top = 2 + sum(top_widths) + 2 * (n_top - 1) + int(np.random.randint(2, 4))
    need_bottom = 2 + sum(bottom_widths) + 2 * (n_bottom - 1) + int(np.random.randint(2, 4))
    width = int(max(need_top, need_bottom))
    height = 11

    grid = np.full((height, width), BG, dtype=int)
    grid[0, :] = BLUE
    grid[:, 0] = RED
    grid[0, 0] = MARKER

    rects: list[dict] = []
    slot_idx = 0
    for row, band_h, widths in ((TOP_ROW, TOP_H, top_widths), (BOTTOM_ROW, BOTTOM_H, bottom_widths)):
        c = 2
        for w in widths:
            color = int(color_seq[slot_idx])
            patch = np.full((int(band_h), int(w)), color, dtype=int)
            paste(grid, patch, int(row), int(c))
            rects.append(
                {
                    "id": int(slot_idx),
                    "color": int(color),
                    "area": int(band_h * w),
                    "patch": patch,
                    "box": (int(row), int(c), int(row + band_h - 1), int(c + w - 1)),
                }
            )
            c += int(w) + 2
            slot_idx += 1

    assert slot_idx == total
    if sum(int(r["patch"].shape[0] != r["patch"].shape[1]) for r in rects) < 2:
        return None
    return grid, rects


def _build_attempt() -> tuple[np.ndarray, dict] | None:
    built = _build_canonical()
    if built is None:
        return None
    canonical, rects = built
    blue_orient = str(np.random.choice(["h", "v"]))
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    inp = _transform(canonical, blue_orient, corner)

    transformed_rects = []
    for rect in rects:
        input_patch, input_box = _render_single_rect(rect["patch"], rect["box"], canonical.shape, blue_orient, corner)
        transformed_rects.append(
            {
                "id": int(rect["id"]),
                "color": int(rect["color"]),
                "area": int(rect["area"]),
                "input_patch": input_patch,
                "input_box": input_box,
            }
        )

    transformed_rects, _cr, _cc, _red_orient = _meta_sort_keys(transformed_rects, inp.shape, blue_orient, corner)
    grouped_area = _grouped_ids(transformed_rects, sort_mode="area")
    grouped_distance = _grouped_ids(transformed_rects, sort_mode="distance")
    flat_area = _flat_ids(transformed_rects, sort_mode="area")
    flat_distance = _flat_ids(transformed_rects, sort_mode="distance")

    flattened_grouped_area = [rid for group in grouped_area for rid in group]
    if len(grouped_area) < 2:
        return None
    if grouped_area == grouped_distance:
        return None
    if flattened_grouped_area == flat_area:
        return None

    rect_by_id = {int(rect["id"]): rect for rect in transformed_rects}
    expected = _assemble_from_groups(rect_by_id, grouped_area, blue_orient)
    grouped_distance_out = _assemble_from_groups(rect_by_id, grouped_distance, blue_orient)
    flat_area_out = _assemble_single_column(rect_by_id, flat_area, blue_orient)
    flat_distance_out = _assemble_single_column(rect_by_id, flat_distance, blue_orient)

    if np.array_equal(expected, grouped_distance_out):
        return None
    if np.array_equal(expected, flat_area_out):
        return None
    if np.array_equal(expected, flat_distance_out):
        return None

    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None

    metrics = {
        "group_count": int(len(grouped_area)),
        "rect_count": int(len(transformed_rects)),
        "area_vs_distance_changed": int(grouped_area != grouped_distance),
        "grouping_vs_flat_changed": int(flattened_grouped_area != flat_area),
        "output_diff_vs_distance_multi": int(not np.array_equal(expected, grouped_distance_out)),
        "output_diff_vs_area_single": int(not np.array_equal(expected, flat_area_out)),
        "output_diff_vs_original": int(not np.array_equal(expected, flat_distance_out)),
    }
    return inp, metrics


def generate_input() -> np.ndarray:
    for _ in range(500):
        built = _build_attempt()
        if built is not None:
            inp, _metrics = built
            return inp
    raise RuntimeError("failed to generate input for 291dc1e1 combo variant 38")
