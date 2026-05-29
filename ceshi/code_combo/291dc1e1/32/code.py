# Combo variant of 291dc1e1 (index 32)
# combo_spec: D1/0+D13/0+D14/2
# Name: 面积排序+分色输出+面积去重
#
# Design analysis
# ===============
# a) Concrete behavior and trace
# - The source scene keeps the original frame: one marked corner, one full blue
#   edge line, one full red edge line, and several full rectangles inside.
# - Original task:
#   1. extract every interior rectangle;
#   2. sort all of them by (distance to blue, distance to red);
#   3. rotate each rectangle 90 CCW iff the blue edge is vertical;
#   4. stack everything into one vertical column.
# - Fused task:
#   1. extract every interior rectangle and measure its bbox area;
#   2. discard rectangles whose area is unique in the input;
#   3. sort the surviving rectangles by area (small to large), with spatial
#      tiebreaks only for equal areas;
#   4. partition that sorted stream by dominant color; each color forms its own
#      output column, and each column preserves the area order of its members;
#   5. place the columns left-to-right in the order their first surviving member
#      appears in the global area-sorted stream.
# - Small concrete example in canonical orientation:
#   top band:    A(2x6), B(2x6), A(2x2)
#   bottom band: A(3x2), B(3x2), B(3x3)
#   Areas are    12,     12,     4,      6,      6,      9
#   Original output order would be roughly
#     A12 -> B12 -> A4 -> A6 -> B6 -> B9
#   because the top band is nearer the blue edge.
#   Fused output first removes unique areas 4 and 9, keeping only
#     A12, B12, A6, B6.
#   Then it area-sorts survivors to
#     A6, B6, A12, B12.
#   Then it groups by dominant color, producing two independent stacks:
#     column A: A6 over A12
#     column B: B6 over B12.
#   The visible difference is therefore not one change but a pipeline:
#   some rectangles disappear, the remaining ones split into multiple columns,
#   and within each column the smaller-area rectangle rises above the larger one.
#
# b) Complexity calibration
# - If all rectangles had height 2, area would collapse to width and D1 would be
#   too weak. The generator therefore uses two fixed band heights, 2 and 3, so
#   area reasoning is genuine.
# - To make D14 visible, the generator always includes at least two unique-area
#   distractors that share colors with kept rectangles; otherwise the discard
#   step could be mistaken for a pure color rule.
# - To make D13 visible, at least two color groups survive the filter, and each
#   surviving color appears at least twice.
# - To make D1 visible after grouping, each surviving color group contains two
#   different areas, and the nearer rectangle is the larger one, so distance
#   order and area order disagree inside each color column.
#
# c) Implicit-assumption audit on the original main()
# - Corner detection assumed a unique marker at a grid corner distinct from the
#   adjacent edge colors. Preserved.
# - Blue-line detection assumed color 1 if present, else the longer edge line.
#   Preserved.
# - Rectangle extraction assumed every object fills its bbox with non-bg colors.
#   Preserved; sparse accent pixels never use bg or marker color.
# - Original global `rects.sort(...)` assumed every extracted rectangle belongs
#   to one output stream. Invalid after fusion: filtering can drop objects and
#   grouping creates several output streams.
# - Original single `seq` / `out_r` / one-column assembly depended on many:1.
#   Invalid after fusion: we need one stack per dominant color, then a horizontal
#   concatenation of those stacks.
# - New assumptions introduced here:
#   1. at least one area is repeated, so the filter leaves something;
#   2. dominant color of each rectangle is unique;
#   3. at least one surviving color group exists.
#   The generator enforces stronger conditions than that: exactly two repeated
#   area classes survive, two color groups survive, and each group has two areas.
#
# d) Placement legality
# - Canonical layout uses three rectangles in the top band and three in the
#   bottom band. Rectangles in a band are separated by 2 background columns.
# - The grid width is computed from the widest band plus trailing background, so
#   nothing overlaps or exits the canvas.
# - Whole-grid rotations/flips preserve those legality guarantees.
#
# e) Visual observability
# - From examples, a human can see that some same-colored rectangles vanish while
#   other rectangles of the same color remain; that points to a non-color filter.
# - The kept rectangles reappear as two side-by-side stacks rather than one
#   column; that exposes the color grouping.
# - Within each color stack, the smaller rectangle is above the larger one even
#   though in the input the larger one is nearer the blue line; that exposes area
#   sorting rather than distance sorting.
#
# f) Visual cleanliness
# - The scene still uses the original sparse geometry: edge frame + two neat
#   interior bands. There are no extra glyphs or floating noise objects.
# - Accent pixels are sparse and stay inside rectangles, so object boundaries and
#   dominant colors remain obvious.
#
# g) Color inferability
# - Colors are not remapped; they are only used as grouping keys after filtering.
# - Because unique-area distractors deliberately reuse the same dominant colors
#   as surviving rectangles, the examples visually imply that color alone is not
#   the keep/drop rule.
#
# h) Generator validation
# - Each attempted input is converted to an intended output from explicit
#   metadata, then checked against `main(input)`.
# - The generator also rejects degenerate samples whose fused output matches the
#   original task or any natural partial ablation (area-only, group-only,
#   filter-only, and the pairwise combinations).
#
# i) New solver assumptions after the edit
# - The solver assumes full rectangles, repeated areas, and a strict dominant
#   color per rectangle. The generator guarantees all three by construction.

from __future__ import annotations

from collections import Counter
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste


BG = 8
BLUE = 1
RED = 2
MARKER = 0
DOMINANT_COLORS = np.array([3, 4, 5, 6, 7, 9], dtype=int)
ACCENT_COLORS = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)

TOP_ROW = 2
BOTTOM_ROW = 6
TOP_H = 2
BOTTOM_H = 3


def _locate_frame(g: np.ndarray) -> tuple[int, tuple[int, int], int, str, str]:
    H, W = map(int, g.shape)
    bg = int(get_bg(g))
    corners = [(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)]
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


def _stack_sequence(seq: list[np.ndarray], bg: int) -> np.ndarray:
    out_w = int(max(int(a.shape[1]) for a in seq))
    out_h = int(sum(int(a.shape[0]) for a in seq))
    out = np.full((out_h, out_w), int(bg), dtype=int)
    out_r = 0
    for a in seq:
        h, w = map(int, a.shape)
        out_c = int((out_w - w) // 2)
        paste(out, a, int(out_r), int(out_c))
        out_r += int(h)
    return out


def _stack_blocks(blocks: list[np.ndarray], bg: int) -> np.ndarray:
    out_h = int(max(int(block.shape[0]) for block in blocks))
    out_w = int(sum(int(block.shape[1]) for block in blocks) + max(0, len(blocks) - 1))
    out = np.full((out_h, out_w), int(bg), dtype=int)
    out_c = 0
    for block in blocks:
        h, w = map(int, block.shape)
        paste(out, block, 0, int(out_c))
        out_c += int(w) + 1
    return out


def _extract_rectangles(grid: np.ndarray) -> tuple[int, tuple[int, int], str, str, list[dict]]:
    g = np.array(grid, dtype=int)
    H, W = map(int, g.shape)
    bg, (cr, cc), marker, blue_orient, red_orient = _locate_frame(g)

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
                "area": int(sub.shape[0] * sub.shape[1]),
                "group": _dominant_color(sub),
            }
        )

    assert rects
    return int(bg), (int(cr), int(cc)), blue_orient, red_orient, rects


def _solve_pipeline(
    rects: list[dict],
    *,
    bg: int,
    cr: int,
    cc: int,
    blue_orient: str,
    red_orient: str,
    filter_unique: bool,
    group_by_color: bool,
    sort_mode: str,
) -> np.ndarray:
    chosen = list(rects)
    if filter_unique:
        area_freq = Counter(int(r["area"]) for r in chosen)
        chosen = [r for r in chosen if int(area_freq[int(r["area"])]) >= 2]
        assert chosen

    if sort_mode == "area":
        ordered = sorted(
            chosen,
            key=lambda r: (
                int(r["area"]),
                _along_from_corner(r["box"], blue_orient, cr, cc),
                _along_from_corner(r["box"], red_orient, cr, cc),
                tuple(map(int, r["box"])),
            ),
        )
    else:
        blue_edge_idx = int(cc if blue_orient == "v" else cr)
        red_edge_idx = int(cc if red_orient == "v" else cr)
        ordered = sorted(
            chosen,
            key=lambda r: (
                _dist_to_edge(r["box"], blue_orient, blue_edge_idx),
                _dist_to_edge(r["box"], red_orient, red_edge_idx),
                _along_from_corner(r["box"], blue_orient, cr, cc),
                _along_from_corner(r["box"], red_orient, cr, cc),
                tuple(map(int, r["box"])),
            ),
        )

    if not group_by_color:
        seq = []
        for rect in ordered:
            arr = rect["arr"]
            if blue_orient == "v":
                arr = np.rot90(arr, k=1)
            seq.append(arr)
        return _stack_sequence(seq, bg)

    groups: dict[int, list[np.ndarray]] = {}
    group_order: list[int] = []
    for rect in ordered:
        grp = int(rect["group"])
        arr = rect["arr"]
        if blue_orient == "v":
            arr = np.rot90(arr, k=1)
        if grp not in groups:
            groups[grp] = []
            group_order.append(grp)
        groups[grp].append(arr)

    blocks = [_stack_sequence(groups[grp], bg) for grp in group_order]
    return _stack_blocks(blocks, bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only rectangles whose area is repeated, then output one area-sorted
    stack per dominant color.

    Rule:
    - Extract full interior rectangles.
    - Discard rectangles whose bbox area occurs only once.
    - Sort surviving rectangles by area, small to large.
    - Split that sorted stream into dominant-color groups.
    - If the blue edge is vertical, rotate each surviving rectangle 90 CCW.
    - Stack each color group top-to-bottom and place the stacks left-to-right in
      first-appearance order from the global area-sorted stream.
    """
    bg, (cr, cc), blue_orient, red_orient, rects = _extract_rectangles(grid)
    return _solve_pipeline(
        rects,
        bg=bg,
        cr=cr,
        cc=cc,
        blue_orient=blue_orient,
        red_orient=red_orient,
        filter_unique=True,
        group_by_color=True,
        sort_mode="area",
    )


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


def _corner_coords(shape: tuple[int, int], corner: str) -> tuple[int, int]:
    H, W = map(int, shape)
    return {
        "tl": (0, 0),
        "tr": (0, W - 1),
        "bl": (H - 1, 0),
        "br": (H - 1, W - 1),
    }[corner]


def _decorate_patch(h: int, w: int, dominant: int) -> np.ndarray:
    patch = np.full((h, w), int(dominant), dtype=int)
    area = int(h * w)
    max_accents = min(2, max(0, (area - 1) // 3))
    if max_accents == 0 or np.random.rand() < 0.35:
        return patch

    n_accents = 1 + int(max_accents == 2 and np.random.rand() < 0.45)
    choices = [int(c) for c in ACCENT_COLORS if int(c) != int(dominant)]
    np.random.shuffle(choices)
    spots = [(0, 0), (h - 1, w - 1), (min(1, h - 1), max(0, w - 2))]
    for idx, (r, c) in enumerate(spots[:n_accents]):
        patch[int(r), int(c)] = int(choices[idx])
    return patch


def _place_band(grid: np.ndarray, row: int, specs: list[dict]) -> list[dict]:
    placed = []
    col = 2
    for spec in specs:
        h = int(spec["h"])
        w = int(spec["w"])
        patch = _decorate_patch(h, w, int(spec["dominant"]))
        paste(grid, patch, int(row), int(col))
        placed.append(
            {
                "id": int(spec["id"]),
                "role": str(spec["role"]),
                "dominant": int(spec["dominant"]),
                "patch": patch,
                "area": int(h * w),
                "box": (int(row), int(col), int(row + h - 1), int(col + w - 1)),
            }
        )
        col += int(w) + 2
    return placed


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


def _build_specs() -> tuple[list[dict], list[dict], dict]:
    group_colors = [int(x) for x in np.random.choice(DOMINANT_COLORS, size=2, replace=False)]
    color_a, color_b = group_colors
    uniq_top_w = int(np.random.choice([2, 5]))   # areas 4 or 10
    uniq_bottom_w = int(np.random.choice([3, 5]))  # areas 9 or 15
    unique_colors = [color_a, color_b]
    np.random.shuffle(unique_colors)

    top_specs = [
        {"id": 0, "role": "keep", "h": TOP_H, "w": 6, "dominant": color_a},
        {"id": 1, "role": "keep", "h": TOP_H, "w": 6, "dominant": color_b},
        {"id": 2, "role": "drop", "h": TOP_H, "w": uniq_top_w, "dominant": unique_colors[0]},
    ]
    bottom_specs = [
        {"id": 3, "role": "keep", "h": BOTTOM_H, "w": 2, "dominant": color_a},
        {"id": 4, "role": "keep", "h": BOTTOM_H, "w": 2, "dominant": color_b},
        {"id": 5, "role": "drop", "h": BOTTOM_H, "w": uniq_bottom_w, "dominant": unique_colors[1]},
    ]

    top_order = np.random.permutation(len(top_specs))
    bottom_order = np.random.permutation(len(bottom_specs))
    top_specs = [top_specs[int(i)] for i in top_order]
    bottom_specs = [bottom_specs[int(i)] for i in bottom_order]
    meta = {
        "survivor_areas": [6, 12],
        "drop_areas": [int(TOP_H * uniq_top_w), int(BOTTOM_H * uniq_bottom_w)],
        "group_colors": tuple(group_colors),
    }
    return top_specs, bottom_specs, meta


def _build_canonical() -> tuple[np.ndarray, list[dict], dict]:
    top_specs, bottom_specs, meta = _build_specs()
    top_used = int(sum(int(spec["w"]) for spec in top_specs) + 2 * (len(top_specs) - 1))
    bottom_used = int(sum(int(spec["w"]) for spec in bottom_specs) + 2 * (len(bottom_specs) - 1))
    trailing = int(np.random.randint(2, 5))
    width = int(2 + max(top_used, bottom_used) + trailing)
    height = 11

    grid = np.full((height, width), BG, dtype=int)
    grid[0, :] = BLUE
    grid[:, 0] = RED
    grid[0, 0] = MARKER

    rects = []
    rects.extend(_place_band(grid, TOP_ROW, top_specs))
    rects.extend(_place_band(grid, BOTTOM_ROW, bottom_specs))
    meta = dict(meta)
    meta["rect_count"] = len(rects)
    return grid, rects, meta


def _build_attempt() -> tuple[np.ndarray, np.ndarray, dict] | None:
    canonical, rects, meta = _build_canonical()
    blue_orient = str(np.random.choice(["h", "v"]))
    corner = str(np.random.choice(["tl", "tr", "bl", "br"]))
    inp = _transform(canonical, blue_orient, corner)

    transformed_rects = []
    for rect in rects:
        input_patch, input_box = _render_single_rect(rect["patch"], rect["box"], canonical.shape, blue_orient, corner)
        transformed_rects.append(
            {
                "id": int(rect["id"]),
                "role": str(rect["role"]),
                "box": input_box,
                "arr": input_patch,
                "area": int(rect["area"]),
                "group": _dominant_color(input_patch),
            }
        )

    cr, cc = _corner_coords(inp.shape, corner)
    red_orient = "v" if blue_orient == "h" else "h"
    expected = _solve_pipeline(
        transformed_rects,
        bg=BG,
        cr=cr,
        cc=cc,
        blue_orient=blue_orient,
        red_orient=red_orient,
        filter_unique=True,
        group_by_color=True,
        sort_mode="area",
    )

    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None

    baselines = {
        "original": _solve_pipeline(
            transformed_rects,
            bg=BG,
            cr=cr,
            cc=cc,
            blue_orient=blue_orient,
            red_orient=red_orient,
            filter_unique=False,
            group_by_color=False,
            sort_mode="distance",
        ),
        "d1_only": _solve_pipeline(
            transformed_rects,
            bg=BG,
            cr=cr,
            cc=cc,
            blue_orient=blue_orient,
            red_orient=red_orient,
            filter_unique=False,
            group_by_color=False,
            sort_mode="area",
        ),
        "d13_only": _solve_pipeline(
            transformed_rects,
            bg=BG,
            cr=cr,
            cc=cc,
            blue_orient=blue_orient,
            red_orient=red_orient,
            filter_unique=False,
            group_by_color=True,
            sort_mode="distance",
        ),
        "d14_only": _solve_pipeline(
            transformed_rects,
            bg=BG,
            cr=cr,
            cc=cc,
            blue_orient=blue_orient,
            red_orient=red_orient,
            filter_unique=True,
            group_by_color=False,
            sort_mode="distance",
        ),
        "filter_group_only": _solve_pipeline(
            transformed_rects,
            bg=BG,
            cr=cr,
            cc=cc,
            blue_orient=blue_orient,
            red_orient=red_orient,
            filter_unique=True,
            group_by_color=True,
            sort_mode="distance",
        ),
        "area_filter_only": _solve_pipeline(
            transformed_rects,
            bg=BG,
            cr=cr,
            cc=cc,
            blue_orient=blue_orient,
            red_orient=red_orient,
            filter_unique=True,
            group_by_color=False,
            sort_mode="area",
        ),
        "area_group_only": _solve_pipeline(
            transformed_rects,
            bg=BG,
            cr=cr,
            cc=cc,
            blue_orient=blue_orient,
            red_orient=red_orient,
            filter_unique=False,
            group_by_color=True,
            sort_mode="area",
        ),
    }
    if any(np.array_equal(expected, base) for base in baselines.values()):
        return None

    area_freq = Counter(int(r["area"]) for r in transformed_rects)
    kept = [r for r in transformed_rects if int(area_freq[int(r["area"])]) >= 2]
    groups: dict[int, list[int]] = {}
    for rect in sorted(
        kept,
        key=lambda r: (
            int(r["area"]),
            _along_from_corner(r["box"], blue_orient, cr, cc),
            _along_from_corner(r["box"], red_orient, cr, cc),
            tuple(map(int, r["box"])),
        ),
    ):
        groups.setdefault(int(rect["group"]), []).append(int(rect["area"]))

    if len(groups) != 2:
        return None
    if sorted(sorted(v) for v in groups.values()) != [[6, 12], [6, 12]]:
        return None
    if sum(int(freq == 1) for freq in area_freq.values()) < 2:
        return None

    meta = dict(meta)
    meta.update(
        {
            "blue_orient": blue_orient,
            "corner": corner,
            "group_area_sequences": [tuple(v) for v in groups.values()],
            "removed_count": len(transformed_rects) - len(kept),
            "baseline_diffs": {
                name: int(not np.array_equal(expected, base)) for name, base in baselines.items()
            },
        }
    )
    return inp, expected, meta


def generate_input() -> np.ndarray:
    for _ in range(600):
        built = _build_attempt()
        if built is not None:
            inp, _expected, _meta = built
            return inp
    raise RuntimeError("failed to generate input for 291dc1e1 combo variant 32")
