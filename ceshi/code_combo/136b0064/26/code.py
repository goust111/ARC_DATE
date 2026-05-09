# Combo variant of 136b0064 (index 26)
# combo_spec: D2/0+D3/0+D11/0
# Name: 大字形形状识别配合图例

"""
Design analysis for the combo variant
=====================================

a) Concrete behavior and flow impact
   - The base task had two columns of 3x3 glyphs on the left, a divider, and only one gray start
     pixel on the right. Area alone decoded each move.
   - This combo enlarges the glyph family and adds a legend. The left half now uses two columns of
     5x5 slots; each slot contains either a full 5x5 glyph or a 4x4 glyph anchored in the slot's
     upper-left corner. The full grid is widened to 23 columns so both the large-glyph canvas and a
     readable legend fit cleanly.
   - The right half contains the gray start pixel on row 0 and, below it, four 5x11 legend strips.
     Each strip shows one glyph sample on the left and one 5x5 arrow exemplar on the right:
     glyph-shape -> direction for this instance.
   - Small trace example:
       Original logic:
         any area-6 glyph -> a fixed horizontal-left segment of fixed length.
       New logic:
         two different area-6 glyphs may appear; one has length 2, the other length 4.
         The legend might map the first area-6 shape to right and the second area-6 shape to down.
       Difference:
         counting pixels fails twice: same-area shapes can have different lengths, and the current
         direction must be read from the legend rather than hard-coded.

b) Complexity calibration
   - The instruction library still has exactly four glyph identities, matching the original task's
     overall scale. This keeps the concept learnable.
   - To make the D2 change visible, the four glyphs are organized as two same-area pairs:
     two 4x4 shapes with area 6 and two 5x5 shapes with area 8.
   - To make the D11 change visible, every sample includes all four legend rows and uses all four
     glyph identities at least once in the actual instruction stream.
   - Direction choices are restricted to the multiset {down, right, left, left}. This preserves the
     original path flavor while still allowing the legend to reshuffle meanings.

c) Hidden assumption audit against the original `main()`
   - Original assumption: bbox is always 3x3. No longer true; new solver must accept 4x4 and 5x5.
   - Original assumption: `comp["area"]` uniquely determines the move. False now because two shapes
     share area 6 and two shapes share area 8.
   - Original assumption: direction is a fixed table. False now because direction comes from the
     right-side legend and changes across samples.
   - Original assumption: right side contains only the gray start. False now; solver must parse the
     legend while still locating the gray pixel.
   - New interaction check: because the legend keys off shape identity, the solver must normalize a
     legend sample by its occupied bbox before matching it against the shape library. This handles
     both 4x4 and 5x5 glyphs inside fixed 5x5 legend windows.

d) Placement legality
   - Large glyphs require a wider left canvas. The combo uses width 11 on the left:
     two 5x5 slot columns at c0=0 and c0=6 with a blank separator column 5.
   - The full grid width is 23: 11 left + 1 divider + 11 right.
   - Glyph rows use stride 6, so slots do not overlap even for 5x5 shapes.
   - The legend uses four fixed 5-row bands starting at rows 2, 8, 14, and 20. To fit those bands,
     the generator restricts N to 5..6, yielding heights 29 or 35.
   - The path is simulated before rendering. Any instruction sequence that exits the left canvas or
     the bottom edge is rejected.

e) Visual observability
   - The left panel visibly contains both 4x4 and 5x5 glyphs, and some distinct glyphs have the
     same number of filled cells. A human can see that area alone cannot explain the output.
   - The legend explicitly teaches the per-instance direction assignment. Because legend row order is
     shuffled, the mapping must be read from shape correspondence rather than from row position.
   - The generator forces both same-area pairs to receive different directions in the legend, so the
     new control variable is exercised rather than hidden.

f) Visual cleanliness
   - Each region has one job: left = instruction stream, divider = separator, right = start + legend.
   - The 5x11 legend strips use a blank spacer column between glyph sample and arrow sample, keeping
     the lookup readable despite the larger symbols.
   - The output remains a single polyline on the cleaned left canvas, and the generator rejects
     samples with too many repaint overlaps so traces stay legible.

g) Color-change inferability
   - Output color still comes directly from each left glyph's own color, which keeps the color rule
     simple. The structural reasoning burden stays on shape and legend lookup.
   - Left glyph colors are randomized from a small palette, with constraints that at least one shape
     appears in multiple colors and at least one color is reused across different shapes. This blocks
     accidental fallback to a hidden color->instruction shortcut.
   - Legend colors are purely stylistic: all legend shapes share one color, while arrow colors only
     help distinguish arrow exemplars visually.

h) New solver assumptions and generator guarantees
   - Solver assumes every left instruction component is one of four known 4x4/5x5 templates;
     generator stamps only those templates.
   - Solver assumes the legend occupies the fixed row bands and fixed local columns for
     shape-vs-arrow samples; generator uses exactly that layout.
   - Solver assumes the gray start remains a single cell on the right at row 0; generator preserves
     that placement.
   - Solver assumes the decoded walk stays in bounds; generator guarantees this by path-first
     simulation plus an oracle check against `main()`.
"""

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg


def find_full_height_constant_col(grid: np.ndarray, *, bg: int | None = None) -> tuple[int | None, int | None]:
    """
    Local helper: find a column where all values are identical and != bg.
    Returns (col_idx, color) or (None, None).
    """
    _h, w = grid.shape
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    for c in range(int(w)):
        col = grid[:, int(c)]
        if (col == col[0]).all() and int(col[0]) != bg:
            return int(c), int(col[0])
    return None, None


# --- geometry ---
GRID_W = 23
DIVIDER_COL = 11
DIVIDER_COLOR = 4
LEFT_W = 11
RIGHT_W = GRID_W - DIVIDER_COL - 1
TILE_COLS = (0, 6)
TILE_ROW_STRIDE = 6
GRAY = 5
GRAY_START_ROW = 0
N_MIN, N_MAX = 5, 6

# Right-side legend layout: one 5x5 glyph sample + spacer + one 5x5 arrow sample.
LEGEND_ROW0S = (2, 8, 14, 20)
LEGEND_SHAPE_C0 = 0
LEGEND_ARROW_C0 = 6

DRAW_COLORS = (1, 2, 3, 6, 7, 8, 9)
LEGEND_SHAPE_COLOR = 6
LEGEND_ARROW_COLORS = {"D": 8, "R": 1, "L": 2}

MIN_DIRECTION_CHANGES = 3
MAX_TRACE_OVERLAPS = 4
MAX_ATTEMPTS = 800


def _mask_key(mask: np.ndarray) -> tuple[int, ...]:
    arr = np.asarray(mask, dtype=int)
    return (int(arr.shape[0]), int(arr.shape[1]), *[int(x) for x in arr.reshape(-1)])


_GLYPH_LIBRARY: dict[str, dict[str, object]] = {
    "hook6": {
        "template": np.array(
            [
                [1, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
            ],
            dtype=int,
        ),
        "seg_len": 2,
        "area": 6,
    },
    "bend6": {
        "template": np.array(
            [
                [1, 0, 0, 0],
                [1, 1, 0, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
            ],
            dtype=int,
        ),
        "seg_len": 4,
        "area": 6,
    },
    "trail8": {
        "template": np.array(
            [
                [1, 1, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 1, 1, 0, 0],
                [0, 0, 1, 0, 1],
                [0, 0, 0, 1, 0],
            ],
            dtype=int,
        ),
        "seg_len": 3,
        "area": 8,
    },
    "fork8": {
        "template": np.array(
            [
                [1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 1, 1, 0, 0],
                [0, 0, 0, 1, 1],
            ],
            dtype=int,
        ),
        "seg_len": 4,
        "area": 8,
    },
}
_GLYPH_IDS = tuple(_GLYPH_LIBRARY.keys())
_MASK_TO_GLYPH = {_mask_key(spec["template"]): glyph_id for glyph_id, spec in _GLYPH_LIBRARY.items()}
_PAIR_MATE = {
    "hook6": "bend6",
    "bend6": "hook6",
    "trail8": "fork8",
    "fork8": "trail8",
}

_DIR_TEMPLATES = {
    "D": np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
        ],
        dtype=int,
    ),
    "R": np.array(
        [
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 0],
        ],
        dtype=int,
    ),
    "L": np.array(
        [
            [0, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
        ],
        dtype=int,
    ),
}


def _crop_binary_patch(patch: np.ndarray, bg: int) -> np.ndarray:
    mask = (np.asarray(patch, dtype=int) != int(bg)).astype(int)
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty patch"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _decode_glyph_patch(patch: np.ndarray, bg: int) -> str:
    key = _mask_key(_crop_binary_patch(patch, bg))
    assert key in _MASK_TO_GLYPH, f"unknown glyph patch: {np.asarray(patch, dtype=int).tolist()}"
    return _MASK_TO_GLYPH[key]


def _decode_dir_patch(patch: np.ndarray, bg: int) -> str:
    mask = (np.asarray(patch, dtype=int) != int(bg)).astype(int)
    for name, tmpl in _DIR_TEMPLATES.items():
        if np.array_equal(mask, tmpl):
            return name
    raise AssertionError(f"unrecognized legend arrow patch:\n{mask}")


def _read_legend(right: np.ndarray, bg: int) -> dict[str, str]:
    glyph_to_dir: dict[str, str] = {}
    for r0 in LEGEND_ROW0S:
        if r0 + 4 >= right.shape[0]:
            continue
        shape_patch = right[r0 : r0 + 5, LEGEND_SHAPE_C0 : LEGEND_SHAPE_C0 + 5]
        arrow_patch = right[r0 : r0 + 5, LEGEND_ARROW_C0 : LEGEND_ARROW_C0 + 5]
        if np.count_nonzero(shape_patch != bg) == 0:
            continue
        glyph_id = _decode_glyph_patch(shape_patch, bg)
        glyph_to_dir[glyph_id] = _decode_dir_patch(arrow_patch, bg)
    assert set(glyph_to_dir) == set(_GLYPH_IDS), f"incomplete legend: got keys {sorted(glyph_to_dir)}"
    return glyph_to_dir


def _glyph_delta(glyph_id: str, direction: str) -> tuple[int, int]:
    seg_len = int(_GLYPH_LIBRARY[glyph_id]["seg_len"])
    if direction == "D":
        return seg_len, 0
    if direction == "R":
        return 1, seg_len - 1
    return 1, -(seg_len - 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw a colored polyline by decoding large glyphs through a right-side legend.

    Input structure:
    - Left of a full-height divider: two columns of large monochromatic glyphs in fixed 5x5 slots on
      the background; some glyphs occupy 4x4 bboxes, others occupy 5x5 bboxes.
    - Right of the divider: one gray start pixel on row 0, plus a four-entry legend where each row
      shows a glyph sample and a 5x5 arrow sample.

    Rule:
    - Remove the divider and shift the gray start horizontally into the left canvas.
    - Read the legend first to obtain glyph-shape -> direction for this particular input.
    - Read the left glyphs in column-major order (left tile column top->bottom, then right column).
    - For each glyph, identify its exact shape, step down one row, and then draw that glyph's
      colored segment using the glyph's own fixed length and the legend-provided direction.

    Key insight: The instruction is now split across two visible cues. Exact shape identity matters
    because same-area glyphs can have different lengths, and the legend must be consulted because
    the current direction is assigned per shape on the right side.
    """
    g = np.array(grid, dtype=int)
    h, _w = g.shape
    bg = int(get_bg(g))

    div_c, div_color = find_full_height_constant_col(g, bg=bg)
    assert div_c is not None and int(div_c) == DIVIDER_COL

    out = np.full((h, int(div_c)), bg, dtype=int)

    gray = np.argwhere(g == GRAY)
    if gray.size == 0:
        return out
    r_cur, c_cur = map(int, gray[0])
    c_cur = int(c_cur - (int(div_c) + 1))
    if not (0 <= c_cur < int(div_c)):
        return out
    out[int(r_cur), int(c_cur)] = GRAY

    left = g[:, : int(div_c)]
    right = g[:, int(div_c) + 1 :]
    glyph_to_dir = _read_legend(right, bg)

    ignore = {int(bg), GRAY}
    if div_color is not None:
        ignore.add(int(div_color))

    glyphs: list[dict[str, object]] = []
    for comp in find_components(left, bg=bg, connectivity=8, monochromatic=True, return_format="dict"):
        color = int(comp["color"])
        if color in ignore:
            continue
        b = comp["bbox"]
        if (bbox_h(b), bbox_w(b)) not in {(4, 4), (5, 5)}:
            continue
        r0, c0, r1, c1 = map(int, b)
        mask = (left[r0 : r1 + 1, c0 : c1 + 1] == color).astype(int)
        key = _mask_key(mask)
        assert key in _MASK_TO_GLYPH, f"unknown left glyph mask: {mask.tolist()}"
        glyphs.append({"color": color, "r0": r0, "c0": c0, "glyph_id": _MASK_TO_GLYPH[key]})

    glyphs.sort(key=lambda d: (int(d["c0"]), int(d["r0"])))

    for glyph in glyphs:
        glyph_id = str(glyph["glyph_id"])
        color = int(glyph["color"])
        seg_len = int(_GLYPH_LIBRARY[glyph_id]["seg_len"])
        direction = glyph_to_dir[glyph_id]

        r_cur = int(r_cur + 1)
        assert r_cur < h, f"segment exits bottom: r_cur={r_cur}, H={h}"
        out[r_cur, int(c_cur)] = color

        if direction == "D":
            end_r = r_cur + (seg_len - 1)
            assert end_r < h, f"vertical segment exits bottom: end_r={end_r}, H={h}"
            for _ in range(seg_len - 1):
                r_cur = int(r_cur + 1)
                out[r_cur, int(c_cur)] = color
        else:
            dir_h = 1 if direction == "R" else -1
            end_c = int(c_cur) + dir_h * (seg_len - 1)
            assert 0 <= end_c < int(div_c), (
                f"horizontal segment exits canvas: c_cur={c_cur}, dir={direction}, "
                f"seg_len={seg_len}, div_c={div_c}"
            )
            for _ in range(seg_len - 1):
                c_cur = int(c_cur + dir_h)
                out[r_cur, c_cur] = color

    return out


def _trace_output(
    h: int,
    start_col: int,
    glyph_ids: list[str],
    colors: list[int],
    glyph_to_dir: dict[str, str],
) -> tuple[np.ndarray, int]:
    out = np.zeros((h, LEFT_W), dtype=int)
    out[GRAY_START_ROW, start_col] = GRAY
    r_cur, c_cur = GRAY_START_ROW, int(start_col)
    overlaps = 0

    for glyph_id, color in zip(glyph_ids, colors):
        seg_len = int(_GLYPH_LIBRARY[glyph_id]["seg_len"])
        direction = glyph_to_dir[glyph_id]

        r_cur += 1
        assert r_cur < h
        if out[r_cur, c_cur] not in (0, GRAY):
            overlaps += 1
        out[r_cur, c_cur] = int(color)

        if direction == "D":
            for _ in range(seg_len - 1):
                r_cur += 1
                assert r_cur < h
                if out[r_cur, c_cur] not in (0, GRAY):
                    overlaps += 1
                out[r_cur, c_cur] = int(color)
        else:
            dir_h = 1 if direction == "R" else -1
            for _ in range(seg_len - 1):
                c_cur += dir_h
                assert 0 <= c_cur < LEFT_W
                if out[r_cur, c_cur] not in (0, GRAY):
                    overlaps += 1
                out[r_cur, c_cur] = int(color)

    return out, overlaps


def _sample_dir_map() -> dict[str, str]:
    while True:
        dirs = ["D", "R", "L", "L"]
        random.shuffle(dirs)
        glyph_to_dir = {glyph_id: dirs[i] for i, glyph_id in enumerate(_GLYPH_IDS)}
        if glyph_to_dir["hook6"] == glyph_to_dir["bend6"]:
            continue
        if glyph_to_dir["trail8"] == glyph_to_dir["fork8"]:
            continue
        return glyph_to_dir


def _sample_path(n: int, start_col: int, glyph_to_dir: dict[str, str]) -> list[str] | None:
    h = TILE_ROW_STRIDE * n - 1

    for _ in range(160):
        r, c = GRAY_START_ROW, start_col
        glyph_ids: list[str] = []
        counts = {glyph_id: 0 for glyph_id in _GLYPH_IDS}
        prev_dir: str | None = None
        changes = 0

        for _step in range(2 * n):
            options = []
            for glyph_id in _GLYPH_IDS:
                dr, dc = _glyph_delta(glyph_id, glyph_to_dir[glyph_id])
                if r + dr < h and 0 <= c + dc < LEFT_W:
                    options.append(glyph_id)
            if not options:
                glyph_ids = []
                break

            def score(glyph_id: str) -> tuple[int, int, int, float]:
                direction = glyph_to_dir[glyph_id]
                unseen_bonus = 1 if counts[glyph_id] == 0 else 0
                pair_bonus = 1 if counts[_PAIR_MATE[glyph_id]] == 0 else 0
                change_bonus = 1 if prev_dir is not None and direction != prev_dir else 0
                return (unseen_bonus, pair_bonus, change_bonus, random.random())

            glyph_id = max(options, key=score)
            glyph_ids.append(glyph_id)
            counts[glyph_id] += 1

            direction = glyph_to_dir[glyph_id]
            if prev_dir is not None and direction != prev_dir:
                changes += 1
            prev_dir = direction

            dr, dc = _glyph_delta(glyph_id, direction)
            r += dr
            c += dc

        if not glyph_ids:
            continue
        if any(counts[glyph_id] == 0 for glyph_id in _GLYPH_IDS):
            continue
        if changes < MIN_DIRECTION_CHANGES:
            continue
        return glyph_ids

    return None


def _sample_glyph_colors(glyph_ids: list[str]) -> list[int] | None:
    for _ in range(120):
        colors = [random.choice(DRAW_COLORS) for _ in glyph_ids]
        shape_to_colors: dict[str, set[int]] = {glyph_id: set() for glyph_id in _GLYPH_IDS}
        color_to_shapes: dict[int, set[str]] = {}

        for glyph_id, color in zip(glyph_ids, colors):
            shape_to_colors[glyph_id].add(int(color))
            color_to_shapes.setdefault(int(color), set()).add(glyph_id)

        varied_shape = any(len(cols) >= 2 for cols in shape_to_colors.values())
        reused_color = any(len(shapes) >= 2 for shapes in color_to_shapes.values())
        if varied_shape and reused_color and len(set(colors)) >= 3:
            return colors
    return None


def _render_grid(
    n: int,
    start_col: int,
    glyph_ids: list[str],
    glyph_colors: list[int],
    glyph_to_dir: dict[str, str],
    legend_order: list[str],
) -> np.ndarray:
    h = TILE_ROW_STRIDE * n - 1
    grid = np.zeros((h, GRID_W), dtype=int)
    grid[:, DIVIDER_COL] = DIVIDER_COLOR
    grid[GRAY_START_ROW, DIVIDER_COL + 1 + start_col] = GRAY

    for i, (glyph_id, color) in enumerate(zip(glyph_ids, glyph_colors)):
        tile_col = TILE_COLS[i // n]
        r0 = TILE_ROW_STRIDE * (i % n)
        template = np.asarray(_GLYPH_LIBRARY[glyph_id]["template"], dtype=int)
        hh, ww = template.shape
        grid[r0 : r0 + hh, tile_col : tile_col + ww] = template * int(color)

    for idx, glyph_id in enumerate(legend_order):
        r0 = LEGEND_ROW0S[idx]
        shape = np.asarray(_GLYPH_LIBRARY[glyph_id]["template"], dtype=int)
        hh, ww = shape.shape
        grid[
            r0 : r0 + hh,
            DIVIDER_COL + 1 + LEGEND_SHAPE_C0 : DIVIDER_COL + 1 + LEGEND_SHAPE_C0 + ww,
        ] = shape * LEGEND_SHAPE_COLOR
        direction = glyph_to_dir[glyph_id]
        arrow = np.asarray(_DIR_TEMPLATES[direction], dtype=int)
        grid[
            r0 : r0 + 5,
            DIVIDER_COL + 1 + LEGEND_ARROW_C0 : DIVIDER_COL + 1 + LEGEND_ARROW_C0 + 5,
        ] = arrow * int(LEGEND_ARROW_COLORS[direction])

    return grid


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    n = random.randint(N_MIN, N_MAX)
    h = TILE_ROW_STRIDE * n - 1
    start_col = random.randint(0, LEFT_W - 1)
    glyph_to_dir = _sample_dir_map()
    glyph_ids = _sample_path(n, start_col, glyph_to_dir)
    if glyph_ids is None:
        return None

    glyph_colors = _sample_glyph_colors(glyph_ids)
    if glyph_colors is None:
        return None

    expected, overlaps = _trace_output(h, start_col, glyph_ids, glyph_colors, glyph_to_dir)
    if overlaps > MAX_TRACE_OVERLAPS:
        return None

    legend_order = random.sample(list(_GLYPH_IDS), k=len(_GLYPH_IDS))
    inp = _render_grid(n, start_col, glyph_ids, glyph_colors, glyph_to_dir, legend_order)
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
