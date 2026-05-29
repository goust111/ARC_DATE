# Combo variant of 136b0064 (index 10)
# combo_spec: D2/0+D11/0
# Name: 图例指导的形状解码

"""
Design analysis for the combo variant
=====================================

a) Concrete behavior and trace
   - The base task had two fixed columns of 3x3 glyphs on the left, a full-height divider in
     column 7, and one gray start pixel on the right. A glyph's pixel count directly determined the
     whole move.
   - This combo breaks that in two places at once.
     1. The four instruction glyphs are now four exact 3x3 masks split into two same-area pairs:
        two shapes have area 4 and two shapes have area 5.
     2. The right side now contains a four-row legend. Each row shows one glyph sample and one
        arrow sample, so direction must be looked up from shape -> arrow, not hard-coded.
   - Concretely, consider two area-4 glyphs:
       hook4   = [[1,1,0],[0,1,0],[0,0,1]]
       tripod4 = [[1,0,0],[0,1,0],[1,0,1]]
     If the legend maps hook4 -> right and tripod4 -> down, then counting pixels only tells "area
     4" and is useless. The solver must identify which contour it is, then consult the legend, then
     use that shape's fixed length.
   - In this combo, shape determines segment length, while the legend determines segment direction.
     The final instruction is therefore a fusion of left-side local shape recognition and right-side
     legend lookup.

b) Complexity calibration
   - Keep exactly four glyph classes and exactly three directions {down, right, left}, with legend
     direction multiset {D, R, L, L}. This stays close to the source path language.
   - Preserve the original segment-length multiset {2, 2, 3, 4}, but reassign lengths to the new
     shape library so that area no longer predicts length.
   - Require every generated sample to use all four shapes at least once. That ensures both same-
     area contrasts are visible and the legend is repeatedly exercised.

c) Hidden assumption audit
   - Original `main()` assumed `comp["area"]` uniquely identified the move. False now: area 4 and
     area 5 each correspond to two different glyphs.
   - Original `area_to_dir` was hard-coded. False now: direction is instance-specific and must be
     read from the legend.
   - Original color-shape binding was harmless. False now: if colors stayed fixed, color could act
     as a shortcut for shape. The generator therefore randomizes left glyph colors independently.
   - The original bbox/tiling assumptions still hold on purpose: every actionable glyph remains one
     monochromatic 8-connected 3x3 component in the original tile slots, so component extraction and
     column-major ordering remain valid.

d) Placement legality
   - The right-side legend needs four 3x7 strips, so `N` is restricted to 4..5 just as in the
     single D11 variant. Then `H = 4*N - 1` is 15 or 19, which comfortably fits the legend rows at
     2, 5, 8, 11 plus the gray start at row 0.
   - The generator simulates the entire walk before stamping glyphs, rejecting any sequence whose
     decoded path exits the 7-column left canvas or the bottom boundary.
   - A small overlap budget is allowed for readability, but heavily tangled traces are rejected.

e) Visual observability
   - Same-area pairs are guaranteed to appear inside each sample, so a human sees that area alone is
     insufficient.
   - The legend explicitly shows which contour maps to which direction in this instance, and legend
     row order is shuffled, so row position cannot be used as a shortcut.
   - Because left glyph colors are decoupled from shape, the only stable structural cue is the 3x3
     contour itself, which matches the intended reasoning path.

f) Visual cleanliness
   - The left side keeps the sparse original layout: two 3x3 tile columns separated by blank space.
   - The right side uses a fixed strip layout with one blank spacer column between shape and arrow,
     keeping the two symbol systems visually separated.
   - Legend samples use fixed styling colors that are separate from the random draw colors used by
     the actionable left glyphs.

g) Change inferability
   - The new color behavior remains inferable because color is not transformed by a hidden rule: the
     output segment still uses the left glyph's own color.
   - The only nontrivial indirection is structural and visible: exact glyph contour -> legend row ->
     direction, plus contour -> fixed length.
   - Generator constraints explicitly block collapse back to a fixed shape->direction solver by
     comparing the combo output to a baseline non-legend mapping and rejecting accidental matches.

h) New solver assumptions and generator guarantees
   - Solver assumes the right-side legend occupies the fixed row bands in `LEGEND_ROW0S`, with a
     3x3 shape patch at local cols 0..2 and a 3x3 arrow patch at local cols 4..6.
   - Solver assumes every left actionable component is one of the four known 3x3 masks.
   - Solver assumes the legend covers all four glyph identities exactly once; the generator stamps
     all four entries every time and verifies the produced input with the solver itself.
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
    H, W = grid.shape
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    for c in range(int(W)):
        col = grid[:, int(c)]
        if (col == col[0]).all() and int(col[0]) != bg:
            return int(c), int(col[0])
    return None, None


# --- shared geometry ---
GRID_W = 15
DIVIDER_COL = 7
DIVIDER_COLOR = 4
LEFT_W = 7
TILE_COLS = (0, 4)
TILE_ROW_STRIDE = 4
GRAY = 5
GRAY_START_ROW = 0
N_MIN, N_MAX = 4, 5

# Right-side legend layout: one 3x3 shape sample + spacer + one 3x3 arrow sample.
LEGEND_ROW0S = (2, 5, 8, 11)
LEGEND_SHAPE_C0 = 0
LEGEND_ARROW_C0 = 4

DRAW_COLORS = (1, 2, 3, 6, 7, 8, 9)
MAX_ATTEMPTS = 800
MAX_COLOR_ATTEMPTS = 160
MAX_TRACE_OVERLAPS = 3
MIN_DIRECTION_CHANGES = 2


def _mask_key(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(int(x) for x in np.asarray(mask, dtype=int).reshape(-1))


_GLYPH_LIBRARY: dict[str, dict[str, object]] = {
    "hook4": {
        "template": np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=int),
        "area": 4,
        "seg_len": 2,
    },
    "tripod4": {
        "template": np.array([[1, 0, 0], [0, 1, 0], [1, 0, 1]], dtype=int),
        "area": 4,
        "seg_len": 4,
    },
    "tee5": {
        "template": np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
        "area": 5,
        "seg_len": 3,
    },
    "stair5": {
        "template": np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=int),
        "area": 5,
        "seg_len": 2,
    },
}
_MASK_TO_GLYPH = {_mask_key(spec["template"]): glyph_id for glyph_id, spec in _GLYPH_LIBRARY.items()}
_GLYPH_IDS = tuple(_GLYPH_LIBRARY.keys())
_AREA_PAIRS = (("hook4", "tripod4"), ("tee5", "stair5"))

_DIR_TEMPLATES = {
    "D": np.array([[0, 1, 0], [0, 1, 0], [1, 1, 1]], dtype=int),
    "R": np.array([[0, 0, 1], [1, 1, 1], [0, 0, 1]], dtype=int),
    "L": np.array([[1, 0, 0], [1, 1, 1], [1, 0, 0]], dtype=int),
}
_LEGEND_ARROW_COLORS = {"D": 8, "R": 1, "L": 2}
_LEGEND_SHAPE_COLORS = {"hook4": 6, "tripod4": 7, "tee5": 8, "stair5": 9}

# Baseline fixed map only exists to prevent the generator from collapsing into a non-legend task.
_BASELINE_DIR_MAP = {"hook4": "D", "tripod4": "L", "tee5": "R", "stair5": "L"}


def _decode_shape_patch(patch: np.ndarray, bg: int) -> str:
    mask = (np.asarray(patch, dtype=int) != int(bg)).astype(int)
    key = _mask_key(mask)
    assert key in _MASK_TO_GLYPH, f"unknown glyph template: {mask.tolist()}"
    return _MASK_TO_GLYPH[key]


def _decode_dir_patch(patch: np.ndarray, bg: int) -> str:
    mask = (np.asarray(patch, dtype=int) != int(bg)).astype(int)
    for name, tmpl in _DIR_TEMPLATES.items():
        if np.array_equal(mask, tmpl):
            return name
    raise AssertionError(f"unknown legend direction patch: {mask.tolist()}")


def _shape_delta(glyph_id: str, direction: str) -> tuple[int, int]:
    seg_len = int(_GLYPH_LIBRARY[glyph_id]["seg_len"])
    if direction == "D":
        return seg_len, 0
    if direction == "R":
        return 1, seg_len - 1
    return 1, -(seg_len - 1)


def _read_legend(right: np.ndarray, bg: int) -> dict[str, str]:
    glyph_to_dir: dict[str, str] = {}
    for r0 in LEGEND_ROW0S:
        if r0 + 2 >= right.shape[0]:
            continue
        shape_patch = right[r0 : r0 + 3, LEGEND_SHAPE_C0 : LEGEND_SHAPE_C0 + 3]
        arrow_patch = right[r0 : r0 + 3, LEGEND_ARROW_C0 : LEGEND_ARROW_C0 + 3]
        if int(np.count_nonzero(shape_patch != bg)) == 0:
            continue
        glyph_id = _decode_shape_patch(shape_patch, bg)
        glyph_to_dir[glyph_id] = _decode_dir_patch(arrow_patch, bg)
    assert set(glyph_to_dir) == set(_GLYPH_IDS), f"incomplete legend keys: {sorted(glyph_to_dir)}"
    return glyph_to_dir


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw a colored polyline by decoding shape lengths through a right-side legend.

    Input structure:
    - Left of the full-height divider: 3x3 monochromatic glyphs in the original two tile columns.
    - Right of the divider: one gray start pixel plus a four-row legend; each legend row contains a
      glyph sample and a 3x3 arrow sample.

    Rule:
    - Remove the divider and shift the gray start into the 7-column left canvas.
    - Read the legend first to obtain shape -> direction for this instance.
    - Read the left glyphs in column-major order. For each glyph, match its exact 3x3 contour to get
      its fixed segment length, step down one row, then draw the segment in that glyph's own color
      using the direction supplied by the legend.

    Key insight: Neither pixel count nor a fixed hard-coded shape mapping is sufficient anymore.
    The full contour identifies which length class the glyph belongs to, and the right-side legend
    separately tells which direction that contour means in this grid.
    """
    g = np.array(grid, dtype=int)
    H, _W = g.shape
    bg = int(get_bg(g))

    div_c, div_color = find_full_height_constant_col(g, bg=bg)
    assert div_c is not None and int(div_c) == DIVIDER_COL

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
        if bbox_h(b) != 3 or bbox_w(b) != 3:
            continue
        r0, c0, r1, c1 = map(int, b)
        glyph_id = _decode_shape_patch(left[r0 : r1 + 1, c0 : c1 + 1], bg)
        glyphs.append({"color": color, "r0": r0, "c0": c0, "glyph_id": glyph_id})

    glyphs.sort(key=lambda d: (int(d["c0"]), int(d["r0"])))

    for glyph in glyphs:
        color = int(glyph["color"])
        glyph_id = str(glyph["glyph_id"])
        seg_len = int(_GLYPH_LIBRARY[glyph_id]["seg_len"])
        direction = glyph_to_dir[glyph_id]

        r_cur = int(r_cur + 1)
        assert r_cur < H, f"segment exits bottom: r_cur={r_cur}, H={H}"
        out[r_cur, int(c_cur)] = color

        if direction == "D":
            end_r = r_cur + (seg_len - 1)
            assert end_r < H, f"vertical segment exits bottom: end_r={end_r}, H={H}"
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
    H: int,
    start_col: int,
    glyph_ids: list[str],
    colors: list[int],
    glyph_to_dir: dict[str, str],
) -> tuple[np.ndarray, int]:
    out = np.zeros((H, LEFT_W), dtype=int)
    out[GRAY_START_ROW, int(start_col)] = GRAY
    r_cur, c_cur = GRAY_START_ROW, int(start_col)
    overlaps = 0

    for glyph_id, color in zip(glyph_ids, colors):
        seg_len = int(_GLYPH_LIBRARY[glyph_id]["seg_len"])
        direction = glyph_to_dir[glyph_id]

        r_cur += 1
        assert r_cur < H
        if out[r_cur, c_cur] not in (0, GRAY):
            overlaps += 1
        out[r_cur, c_cur] = int(color)

        if direction == "D":
            for _ in range(seg_len - 1):
                r_cur += 1
                assert r_cur < H
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
        if not all(glyph_to_dir[a] != glyph_to_dir[b] for a, b in _AREA_PAIRS):
            continue
        diff_from_baseline = sum(int(glyph_to_dir[g] != _BASELINE_DIR_MAP[g]) for g in _GLYPH_IDS)
        if diff_from_baseline >= 2:
            return glyph_to_dir


def _sample_path(N: int, start_col: int, glyph_to_dir: dict[str, str]) -> list[str] | None:
    H = 4 * N - 1
    r, c = GRAY_START_ROW, int(start_col)
    glyph_ids: list[str] = []
    counts = {glyph_id: 0 for glyph_id in _GLYPH_IDS}
    prev_dir: str | None = None
    direction_changes = 0

    for _ in range(2 * N):
        options = []
        for glyph_id in _GLYPH_IDS:
            dr, dc = _shape_delta(glyph_id, glyph_to_dir[glyph_id])
            if r + dr < H and 0 <= c + dc < LEFT_W:
                options.append(glyph_id)
        if not options:
            return None

        def score(glyph_id: str) -> tuple[int, int, int, float]:
            direction = glyph_to_dir[glyph_id]
            missing_bonus = 2 if counts[glyph_id] == 0 else 0
            pair_bonus = 0
            for a, b in _AREA_PAIRS:
                if glyph_id == a and counts[a] == 0 and counts[b] > 0:
                    pair_bonus = 1
                if glyph_id == b and counts[b] == 0 and counts[a] > 0:
                    pair_bonus = 1
            change_bonus = 1 if prev_dir is not None and direction != prev_dir else 0
            return (missing_bonus, pair_bonus, change_bonus, random.random())

        glyph_id = max(options, key=score)
        glyph_ids.append(glyph_id)
        counts[glyph_id] += 1

        direction = glyph_to_dir[glyph_id]
        if prev_dir is not None and direction != prev_dir:
            direction_changes += 1
        prev_dir = direction

        dr, dc = _shape_delta(glyph_id, direction)
        r += dr
        c += dc

    if any(counts[glyph_id] == 0 for glyph_id in _GLYPH_IDS):
        return None
    if direction_changes < MIN_DIRECTION_CHANGES:
        return None
    return glyph_ids


def _sample_colors(glyph_ids: list[str]) -> list[int] | None:
    counts = {glyph_id: glyph_ids.count(glyph_id) for glyph_id in _GLYPH_IDS}
    repeated_shapes = {glyph_id for glyph_id, count in counts.items() if count >= 2}
    for _ in range(MAX_COLOR_ATTEMPTS):
        colors = [random.choice(DRAW_COLORS) for _ in glyph_ids]
        shape_to_colors: dict[str, set[int]] = {glyph_id: set() for glyph_id in _GLYPH_IDS}
        color_to_shapes: dict[int, set[str]] = {}
        for glyph_id, color in zip(glyph_ids, colors):
            shape_to_colors[glyph_id].add(int(color))
            color_to_shapes.setdefault(int(color), set()).add(glyph_id)

        diversified_repeat = any(len(shape_to_colors[glyph_id]) >= 2 for glyph_id in repeated_shapes)
        shared_color = any(len(shapes) >= 2 for shapes in color_to_shapes.values())
        if diversified_repeat and shared_color:
            return colors
    return None


def _render_grid(
    N: int,
    start_col: int,
    glyph_ids: list[str],
    glyph_colors: list[int],
    glyph_to_dir: dict[str, str],
    legend_order: list[str],
) -> np.ndarray:
    H = 4 * N - 1
    grid = np.zeros((H, GRID_W), dtype=int)
    right_c0 = DIVIDER_COL + 1

    grid[:, DIVIDER_COL] = DIVIDER_COLOR
    grid[GRAY_START_ROW, right_c0 + int(start_col)] = GRAY

    for i, (glyph_id, color) in enumerate(zip(glyph_ids, glyph_colors)):
        tile_col = TILE_COLS[i // N]
        r0 = TILE_ROW_STRIDE * (i % N)
        template = np.array(_GLYPH_LIBRARY[glyph_id]["template"], dtype=int)
        grid[r0 : r0 + 3, tile_col : tile_col + 3] = template * int(color)

    for idx, glyph_id in enumerate(legend_order):
        r0 = LEGEND_ROW0S[idx]
        shape_template = np.array(_GLYPH_LIBRARY[glyph_id]["template"], dtype=int)
        arrow_template = _DIR_TEMPLATES[glyph_to_dir[glyph_id]]
        grid[r0 : r0 + 3, right_c0 + LEGEND_SHAPE_C0 : right_c0 + LEGEND_SHAPE_C0 + 3] = (
            shape_template * int(_LEGEND_SHAPE_COLORS[glyph_id])
        )
        grid[r0 : r0 + 3, right_c0 + LEGEND_ARROW_C0 : right_c0 + LEGEND_ARROW_C0 + 3] = (
            arrow_template * int(_LEGEND_ARROW_COLORS[glyph_to_dir[glyph_id]])
        )

    return grid


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    N = random.randint(N_MIN, N_MAX)
    H = 4 * N - 1
    start_col = random.randint(0, LEFT_W - 1)
    glyph_to_dir = _sample_dir_map()
    glyph_ids = _sample_path(N, start_col, glyph_to_dir)
    if glyph_ids is None:
        return None

    glyph_colors = _sample_colors(glyph_ids)
    if glyph_colors is None:
        return None

    expected, overlaps = _trace_output(H, start_col, glyph_ids, glyph_colors, glyph_to_dir)
    if overlaps > MAX_TRACE_OVERLAPS:
        return None

    baseline, _ = _trace_output(H, start_col, glyph_ids, glyph_colors, _BASELINE_DIR_MAP)
    if np.array_equal(expected, baseline):
        return None

    legend_order = random.sample(list(_GLYPH_IDS), k=len(_GLYPH_IDS))
    inp = _render_grid(N, start_col, glyph_ids, glyph_colors, glyph_to_dir, legend_order)
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
