# Variant of 136b0064
# Dimension: D2 — 字形编码方式——面积到指令的映射 (transform)
# Change: 字形像素面积唯一决定线段：area 4→垂直向下2格，area 5→水平向右3格，area 6→水平向左4格，area 7→水平向左2格
#         → 字形形状（而非面积）决定指令——引入同面积不同形状

"""
Design analysis for the D2 variant
==================================

a) Concrete behavior and flow impact
   - The input layout stays the same: two columns of 3x3 glyph slots on the left, one full-height
     divider in column 7, and one gray start pixel on the right.
   - The change happens at the decoding step. The original solver read each 3x3 component, counted
     its pixels, and mapped area to a segment. The variant keeps the same four segment instructions
     but reassigns them to four fixed 3x3 masks, with two masks sharing area 4 and two masks sharing
     area 5.
   - Small example:
       Original logic:
         area-4 glyph -> always draw vertical down length 2.
       Variant logic:
         area-4 "hook"  [[1,1,0],[0,1,0],[0,0,1]] -> down 2
         area-4 "tripod"[[1,0,0],[0,1,0],[1,0,1]] -> left 2
       Difference:
         Two glyphs with the same pixel count now generate different path segments, so counting
         pixels is no longer sufficient.

b) Complexity calibration
   - Only four instruction classes are kept, so the path grammar remains small and learnable.
   - The template library uses exactly two shared-area pairs: this is enough to visibly break the
     old area rule without exploding the number of cases.
   - Generator constraints require at least one same-area/different-shape pair per sample, plus at
     least three distinct templates overall, so the new mechanism is repeatedly visible.

c) Review of old implicit assumptions
   - Old `main()` assumed `comp["area"]` uniquely identified the move. That assumption is false now.
   - Old `area_to_len` / `area_to_dir` tables must be replaced by exact 3x3 mask lookup.
   - The old color-shape binding was "harmless" because color happened to match area templates.
     In this variant that would let the solver cheat via color, so glyph colors are randomized and
     solver logic ignores them except when painting the output segment color.
   - The bbox-size assumption (every glyph fills a 3x3 bbox) is preserved on purpose, so the
     component extraction strategy still works.

d) Placement legality
   - Every glyph template is 8-connected, monochromatic, and has a full 3x3 bbox.
   - Tile positions, divider placement, and gray-start placement remain unchanged from the base task.
   - The generator still samples a path first, rejecting any sequence whose decoded walk exits the
     left canvas or the bottom row. It also rejects heavily self-overlapping traces to keep outputs
     readable.

e) Visual observability
   - The new rule is visible because examples contain distinct glyph shapes with identical area.
   - Colors do not encode the move: the same shape can appear in different colors across samples,
     and a sample is forced to contain at least one reused color on different shapes.
   - A human can compare the left glyph masks with the right polyline and infer that contour, not
     area and not color, determines the displacement.

f) Visual cleanliness
   - The left-side layout is unchanged and sparse, so glyphs remain easy to isolate.
   - The output trace is still a single polyline on a clean 7-column canvas.
   - Generator rejects samples with too many repaint overlaps, preventing unreadable tangles.

g) Color-change inferability
   - Output color still copies the glyph's own color, but color is sampled independently from shape
     using a small palette that excludes background/divider/start colors.
   - Because the motion and the color are disentangled, the structural cue remains the glyph mask.

h) New solver assumptions and generator guarantees
   - Solver assumes each non-divider/non-gray component on the left is a 3x3 template from the
     known library; generator guarantees that by stamping only those templates.
   - Solver assumes column-major ordering by tile origin; generator keeps the original slot layout.
   - Solver assumes the decoded walk stays in bounds; generator enforces that by path simulation
     before rendering.
"""

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg


def find_full_height_constant_col(grid: np.ndarray, *, bg: int | None = None) -> tuple[int | None, int | None]:
    """
    Local helper (only used by this solver): find a column where all values are identical and != bg.
    Returns (col_idx, color) or (None, None).
    """
    H, W = grid.shape
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    for c in range(int(W)):
        col = grid[:, int(c)]
        if (col == col[0]).all() and int(col[0]) != bg:
            return int(c), int(col[0])
    return None, None


# --- canvas geometry ---
N_MIN, N_MAX = 2, 5
GRID_W = 15
DIVIDER_COL = 7
DIVIDER_COLOR = 4
LEFT_W = 7
TILE_COLS = (0, 4)
TILE_ROW_STRIDE = 4
GRAY = 5
GRAY_START_ROW = 0
PALETTE = (1, 2, 3, 6, 7, 8, 9)


def _mask_key(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(int(x) for x in mask.astype(int).reshape(-1))


_GLYPH_LIBRARY: dict[str, dict[str, object]] = {
    "hook4": {
        "template": np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=int),
        "seg_len": 2,
        "dir_h": 0,
        "kind": "vertical",
        "delta": (2, 0),
        "area": 4,
    },
    "tripod4": {
        "template": np.array([[1, 0, 0], [0, 1, 0], [1, 0, 1]], dtype=int),
        "seg_len": 2,
        "dir_h": -1,
        "kind": "horizontal",
        "delta": (1, -1),
        "area": 4,
    },
    "tee5": {
        "template": np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
        "seg_len": 3,
        "dir_h": 1,
        "kind": "horizontal",
        "delta": (1, 2),
        "area": 5,
    },
    "stair5": {
        "template": np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=int),
        "seg_len": 4,
        "dir_h": -1,
        "kind": "horizontal",
        "delta": (1, -3),
        "area": 5,
    },
}
_MASK_TO_GLYPH = {_mask_key(spec["template"]): name for name, spec in _GLYPH_LIBRARY.items()}
_GLYPH_IDS = tuple(_GLYPH_LIBRARY.keys())
MAX_TRACE_OVERLAPS = 2
MIN_DIRECTION_CHANGES = 2
MAX_ATTEMPTS = 600


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw a polyline by decoding 3x3 glyph instructions read across a divided grid.

    Input structure:
    - Left of a full-height divider: 3x3 monochromatic glyphs placed in fixed tile slots on bg.
    - Right of divider: a single gray (5) start pixel whose shifted column marks the trace origin.

    Rule:
    - Remove the divider and project the gray start into the 7-column left canvas.
    - Read the left glyphs in column-major order (left tile column top→bottom, then right column).
    - For each 3x3 glyph, identify its exact shape template; same-area templates can decode to
      different segments.
    - Step down one row, then draw that glyph's colored segment: one template means down 2, one
      means right 3, one means left 4, and one means left 2.

    Key insight: Pixel count is no longer enough. The solver must match each glyph's full 3x3 mask,
    because different contours with the same area now encode different directions and lengths.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    div_c, div_color = find_full_height_constant_col(g, bg=bg)
    assert div_c is not None

    out = np.full((H, int(div_c)), bg, dtype=int)

    gray = np.argwhere(g == GRAY)
    if gray.size == 0:
        return out
    r_cur, c_cur = map(int, gray[0])
    c_cur = int(c_cur - (int(div_c) + 1))
    if not (0 <= c_cur < int(div_c)):
        return out
    out[int(r_cur), int(c_cur)] = GRAY

    left = g[:, : int(div_c)]
    ignore = {int(bg), GRAY}
    if div_color is not None:
        ignore.add(int(div_color))

    glyphs: list[dict[str, object]] = []
    for comp in find_components(left, bg=bg, connectivity=8, monochromatic=True, return_format="dict"):
        color = int(comp["color"])
        if color in ignore:
            continue
        b = comp["bbox"]
        if bbox_h(b) != 3 or bbox_w(b) != 3:
            continue
        r0, c0, r1, c1 = map(int, b)
        mask = (left[r0 : r1 + 1, c0 : c1 + 1] == color).astype(int)
        key = _mask_key(mask)
        assert key in _MASK_TO_GLYPH, f"unknown glyph template: {mask.tolist()}"
        glyphs.append({"color": color, "r0": r0, "c0": c0, "glyph_id": _MASK_TO_GLYPH[key]})

    glyphs.sort(key=lambda d: (int(d["c0"]), int(d["r0"])))

    for glyph in glyphs:
        color = int(glyph["color"])
        spec = _GLYPH_LIBRARY[str(glyph["glyph_id"])]
        seg_len = int(spec["seg_len"])
        dir_h = int(spec["dir_h"])
        kind = str(spec["kind"])

        r_cur = int(r_cur + 1)
        assert r_cur < H, f"segment exits bottom: r_cur={r_cur}, H={H}"
        out[r_cur, int(c_cur)] = color

        if kind == "vertical":
            end_r = r_cur + (seg_len - 1)
            assert end_r < H, f"vertical segment exits bottom: end_r={end_r}, H={H}"
            for _ in range(seg_len - 1):
                r_cur = int(r_cur + 1)
                out[r_cur, int(c_cur)] = color
        else:
            end_c = int(c_cur) + dir_h * (seg_len - 1)
            assert 0 <= end_c < int(div_c), (
                f"horizontal segment exits canvas: c_cur={c_cur}, dir={dir_h}, "
                f"seg_len={seg_len}, div_c={div_c}"
            )
            for _ in range(seg_len - 1):
                c_cur = int(c_cur + dir_h)
                out[r_cur, c_cur] = color

    return out


def _trace_output(H: int, start_col: int, glyph_ids: list[str], colors: list[int]) -> tuple[np.ndarray, int]:
    out = np.zeros((H, LEFT_W), dtype=int)
    out[GRAY_START_ROW, start_col] = GRAY
    r_cur, c_cur = GRAY_START_ROW, int(start_col)
    overlaps = 0

    for glyph_id, color in zip(glyph_ids, colors):
        spec = _GLYPH_LIBRARY[glyph_id]
        seg_len = int(spec["seg_len"])
        dir_h = int(spec["dir_h"])
        kind = str(spec["kind"])

        r_cur += 1
        assert r_cur < H
        if out[r_cur, c_cur] not in (0, GRAY):
            overlaps += 1
        out[r_cur, c_cur] = int(color)

        if kind == "vertical":
            for _ in range(seg_len - 1):
                r_cur += 1
                assert r_cur < H
                if out[r_cur, c_cur] not in (0, GRAY):
                    overlaps += 1
                out[r_cur, c_cur] = int(color)
        else:
            for _ in range(seg_len - 1):
                c_cur += dir_h
                assert 0 <= c_cur < LEFT_W
                if out[r_cur, c_cur] not in (0, GRAY):
                    overlaps += 1
                out[r_cur, c_cur] = int(color)

    return out, overlaps


def _sample_glyph_ids(N: int, start_col: int) -> list[str] | None:
    H = 4 * N - 1
    r, c = GRAY_START_ROW, start_col
    glyph_ids: list[str] = []

    for _ in range(2 * N):
        options = []
        for glyph_id in _GLYPH_IDS:
            dr, dc = _GLYPH_LIBRARY[glyph_id]["delta"]
            if r + int(dr) < H and 0 <= c + int(dc) < LEFT_W:
                options.append(glyph_id)
        if not options:
            return None
        glyph_id = random.choice(options)
        glyph_ids.append(glyph_id)
        dr, dc = _GLYPH_LIBRARY[glyph_id]["delta"]
        r += int(dr)
        c += int(dc)

    kinds = [str(_GLYPH_LIBRARY[g]["kind"]) for g in glyph_ids]
    changes = sum(1 for i in range(1, len(kinds)) if kinds[i] != kinds[i - 1])
    if changes < MIN_DIRECTION_CHANGES:
        return None

    if len(set(glyph_ids)) < 3:
        return None

    areas = [int(_GLYPH_LIBRARY[g]["area"]) for g in glyph_ids]
    same_area_diff_shape = any(
        areas[i] == areas[j] and glyph_ids[i] != glyph_ids[j]
        for i in range(len(glyph_ids))
        for j in range(i + 1, len(glyph_ids))
    )
    if not same_area_diff_shape:
        return None

    return glyph_ids


def _assign_colors(glyph_ids: list[str]) -> list[int]:
    colors = [random.choice(PALETTE) for _ in glyph_ids]
    diff_pairs = [(i, j) for i in range(len(glyph_ids)) for j in range(i + 1, len(glyph_ids)) if glyph_ids[i] != glyph_ids[j]]
    if diff_pairs:
        i, j = random.choice(diff_pairs)
        shared = random.choice(PALETTE)
        colors[i] = shared
        colors[j] = shared
    return colors


def _render_grid(N: int, start_col: int, glyph_ids: list[str], colors: list[int]) -> np.ndarray:
    H = 4 * N - 1
    grid = np.zeros((H, GRID_W), dtype=int)
    grid[:, DIVIDER_COL] = DIVIDER_COLOR
    grid[GRAY_START_ROW, DIVIDER_COL + 1 + start_col] = GRAY

    for i, (glyph_id, color) in enumerate(zip(glyph_ids, colors)):
        tile_col = TILE_COLS[i // N]
        r0 = TILE_ROW_STRIDE * (i % N)
        template = np.array(_GLYPH_LIBRARY[glyph_id]["template"], dtype=int)
        grid[r0 : r0 + 3, tile_col : tile_col + 3] = template * int(color)
    return grid


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    N = random.randint(N_MIN, N_MAX)
    H = 4 * N - 1
    start_col = random.randint(0, LEFT_W - 1)
    glyph_ids = _sample_glyph_ids(N, start_col)
    if glyph_ids is None:
        return None

    colors = _assign_colors(glyph_ids)
    expected, overlaps = _trace_output(H, start_col, glyph_ids, colors)
    if overlaps > MAX_TRACE_OVERLAPS:
        return None

    inp = _render_grid(N, start_col, glyph_ids, colors)
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        try:
            attempt = _gen_attempt()
            if attempt is None:
                continue
            inp, expected = attempt
            out = main(inp)
            if out.shape != expected.shape:
                continue
            if not np.array_equal(out, expected):
                continue
            return inp
        except Exception:
            continue
    raise RuntimeError(f"generate_input: failed after {MAX_ATTEMPTS} attempts")
