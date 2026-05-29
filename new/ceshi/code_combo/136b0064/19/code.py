# Combo variant of 136b0064 (index 19)
# combo_spec: D8/2+D11/1
# Name: 分隔线编码步进模式

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg


# Design analysis
# a) Concrete behavior:
#    Keep the original 3x3 glyph language and gray start pixel, but the connector step
#    before each glyph is no longer always "down 1". Example with start_col=2 and
#    areas=[5,4,7,5]:
#      - original: down, draw right-3; down, draw down-2; down, draw left-2; ...
#      - fused, ctrl=8: down, draw right-3; right, draw down-2; down, draw left-2; ...
#      - fused, ctrl=9: right, draw right-3; down, draw down-2; right, draw left-2; ...
#    The only new parameter is which axis comes first; after that the connector axes
#    alternate with a fixed period 2.
# b) Complexity calibration:
#    The control space is intentionally small: exactly two divider-only control colors,
#    repeated twice for visibility, and only one global bit is encoded
#    (vertical-first vs horizontal-first). This makes the change visible without
#    exploding the case count.
# c) Broken assumptions from the original solver:
#    - The divider is no longer constant-colored, so we must detect a full-height
#      non-background column rather than a constant column.
#    - The pre-segment connector no longer always increases the row, so validity and
#      drawing must be simulated per phase.
#    - Generator legality depends jointly on glyph area and current connector phase.
# d) Placement legality:
#    The generator samples the output walk first on the 7-column left canvas and rejects
#    any instruction whose connector landing cell or subsequent segment would leave the
#    canvas. Divider control pixels live only on the divider and never overlap glyphs.
# e) Visual observability:
#    The divider still looks like the original yellow split bar, but now contains two
#    cyan/maroon pixels. Across examples, that local divider cue flips whether the first
#    connector is vertical or horizontal, so the extra rule is visible from a few pairs.
# f) Visual neatness:
#    The new symbols are confined to the divider column, keeping glyph tiles, blank
#    spacing, and the right-side gray start uncluttered.
# g) New solver assumptions and generator guarantees:
#    - exactly one full-height non-bg column exists -> generator uses only the divider;
#    - divider control pixels all share one color in {8,9} -> generator stamps exactly
#      two such pixels;
#    - glyph instructions remain isolated 3x3 monochrome components on the left;
#    - the decoded alternating walk is in-bounds and control-sensitive -> generator
#      rejects samples that match the legacy always-down walk or ignore the control bit.


GRID_W = 15
LEFT_W = 7
DIVIDER_COL = 7
BASE_DIVIDER_COLOR = 4
GRAY = 5
GRAY_START_ROW = 0
TILE_COLS = (0, 4)
TILE_ROW_STRIDE = 4
N_MIN, N_MAX = 2, 5
MAX_ATTEMPTS = 400
MIN_DIRECTION_CHANGES = 2

CTRL_VERTICAL_FIRST = 8
CTRL_HORIZONTAL_FIRST = 9
NEXT_VERTICAL_FIRST = True

AREA_TO_COLOR = {4: 6, 5: 1, 6: 3, 7: 2}
AREA_TO_EXTRA_DELTA = {
    4: (1, 0),   # connector landing cell already counts as first cell
    5: (0, 2),
    6: (0, -3),
    7: (0, -1),
}
AREA_TO_TEMPLATE = {
    4: np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
    5: np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int),
    6: np.array([[1, 1, 1], [0, 1, 0], [1, 0, 1]], dtype=int),
    7: np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=int),
}
AREAS = (4, 5, 6, 7)


def find_full_height_filled_col(grid: np.ndarray, *, bg: int | None = None) -> tuple[int | None, np.ndarray | None]:
    """Find a column whose every cell is non-background."""
    H, W = grid.shape
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    for c in range(int(W)):
        col = grid[:, int(c)]
        if int(col.shape[0]) == int(H) and bool((col != bg).all()):
            return int(c), col.copy()
    return None, None


def _control_rows(H: int) -> tuple[int, int]:
    return 1, int(H - 2)


def _divider_phase_vertical_first(divider_vals: np.ndarray) -> bool:
    vals = [int(v) for v in divider_vals]
    base = int(Counter(vals).most_common(1)[0][0])
    ctrl = [v for v in vals if int(v) != int(base)]
    assert ctrl, "divider is missing control pixels"
    assert len(set(ctrl)) == 1, f"divider has mixed control colors: {sorted(set(ctrl))}"
    ctrl_color = int(ctrl[0])
    if ctrl_color == CTRL_VERTICAL_FIRST:
        return True
    if ctrl_color == CTRL_HORIZONTAL_FIRST:
        return False
    raise AssertionError(f"unknown control color: {ctrl_color}")


def _step_is_vertical(idx: int, vertical_first: bool) -> bool:
    return (idx % 2 == 0) if bool(vertical_first) else (idx % 2 == 1)


def _instruction_cells(r: int, c: int, area: int, idx: int, vertical_first: bool) -> list[tuple[int, int]]:
    dr0, dc0 = (1, 0) if _step_is_vertical(idx, vertical_first) else (0, 1)
    rr = int(r + dr0)
    cc = int(c + dc0)
    cells = [(int(rr), int(cc))]

    dr1, dc1 = AREA_TO_EXTRA_DELTA[int(area)]
    if int(area) == 4:
        cells.append((int(rr + dr1), int(cc + dc1)))
    else:
        step = 1 if int(dc1) > 0 else -1
        for off in range(1, abs(int(dc1)) + 1):
            cells.append((int(rr), int(cc + step * off)))
    return cells


def _simulate_output(
    H: int,
    start_col: int,
    areas: list[int],
    colors: list[int],
    *,
    vertical_first: bool,
) -> np.ndarray:
    out = np.zeros((int(H), LEFT_W), dtype=int)
    r_cur, c_cur = 0, int(start_col)
    out[int(r_cur), int(c_cur)] = GRAY

    for idx, (area, color) in enumerate(zip(areas, colors)):
        cells = _instruction_cells(int(r_cur), int(c_cur), int(area), int(idx), bool(vertical_first))
        for rr, cc in cells:
            assert 0 <= int(rr) < int(H), f"row out of bounds: {(rr, cc)} vs H={H}"
            assert 0 <= int(cc) < LEFT_W, f"col out of bounds: {(rr, cc)} vs LEFT_W={LEFT_W}"
            out[int(rr), int(cc)] = int(color)
        r_cur, c_cur = map(int, cells[-1])

    return out


def _simulate_legacy_output(H: int, start_col: int, areas: list[int], colors: list[int]) -> np.ndarray:
    out = np.zeros((int(H), LEFT_W), dtype=int)
    r_cur, c_cur = 0, int(start_col)
    out[int(r_cur), int(c_cur)] = GRAY

    for area, color in zip(areas, colors):
        r_cur = int(r_cur + 1)
        assert r_cur < int(H)
        out[int(r_cur), int(c_cur)] = int(color)
        if int(area) == 4:
            r_cur = int(r_cur + 1)
            assert r_cur < int(H)
            out[int(r_cur), int(c_cur)] = int(color)
        else:
            extra_dc = int(AREA_TO_EXTRA_DELTA[int(area)][1])
            step = 1 if extra_dc > 0 else -1
            for _ in range(abs(extra_dc)):
                c_cur = int(c_cur + step)
                assert 0 <= int(c_cur) < LEFT_W
                out[int(r_cur), int(c_cur)] = int(color)
    return out


def _extract_glyphs(left: np.ndarray, bg: int) -> list[dict[str, int]]:
    glyphs: list[dict[str, int]] = []
    for comp in find_components(left, bg=bg, connectivity=8, monochromatic=True, return_format="dict"):
        b = comp["bbox"]
        if bbox_h(b) != 3 or bbox_w(b) != 3:
            continue
        r0, c0, _r1, _c1 = map(int, b)
        glyphs.append(
            {
                "color": int(comp["color"]),
                "r0": int(r0),
                "c0": int(c0),
                "area": int(comp["area"]),
            }
        )
    glyphs.sort(key=lambda d: (int(d["c0"]), int(d["r0"])))
    return glyphs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw a colored polyline by reading 3x3 glyphs and a divider-embedded step-phase code.

    Input structure:
    - Left of a full-height non-background divider: 3x3 monochrome glyphs on bg.
    - The divider is mostly color 4, but it contains a couple of control pixels whose
      color encodes whether the alternating connector pattern starts vertical or horizontal.
    - Right of the divider: one gray (5) start pixel on an otherwise empty field.

    Rule:
    - Remove the divider and shift the gray pixel into the 7-column left canvas.
    - Read glyphs in column-major order (left column top→bottom, then right column).
    - Each glyph's area still gives the segment instruction:
      area 4 → down 2, area 5 → right 3, area 6 → left 4, area 7 → left 2.
    - Before each segment, take a 1-cell connector step whose axis alternates between
      vertical and horizontal; the divider control chooses whether the first connector is
      down or right.

    Key insight: The glyphs still encode the segment geometry, but the divider is now an
    active symbol layer: its control pixels set the phase of the alternating connector walk.
    """
    g = np.array(grid, dtype=int)
    H, _ = g.shape
    bg = int(get_bg(g))

    div_c, divider_vals = find_full_height_filled_col(g, bg=bg)
    assert div_c is not None and divider_vals is not None, "missing full-height divider"
    vertical_first = _divider_phase_vertical_first(divider_vals)

    out = np.zeros((int(H), int(div_c)), dtype=int)

    gray = np.argwhere(g == GRAY)
    if gray.size == 0:
        return out
    r0, c0 = map(int, gray[0])
    start_col = int(c0 - (int(div_c) + 1))
    if not (0 <= start_col < int(div_c)):
        return out

    left = g[:, : int(div_c)]
    glyphs = _extract_glyphs(left, bg)
    areas = [int(glyph["area"]) for glyph in glyphs]
    colors = [int(glyph["color"]) for glyph in glyphs]
    return _simulate_output(int(H), int(start_col), areas, colors, vertical_first=bool(vertical_first))


def _sample_path(N: int, start_col: int, vertical_first: bool) -> list[int] | None:
    H = 4 * int(N) - 1
    r, c = 0, int(start_col)
    areas: list[int] = []

    for idx in range(2 * int(N)):
        opts = []
        for area in AREAS:
            cells = _instruction_cells(int(r), int(c), int(area), int(idx), bool(vertical_first))
            if all(0 <= int(rr) < int(H) and 0 <= int(cc) < LEFT_W for rr, cc in cells):
                opts.append(int(area))
        if not opts:
            return None
        area = int(random.choice(opts))
        areas.append(area)
        r, c = map(int, _instruction_cells(int(r), int(c), int(area), int(idx), bool(vertical_first))[-1])

    types = [int(area) == 4 for area in areas]
    changes = sum(1 for i in range(1, len(types)) if bool(types[i]) != bool(types[i - 1]))
    if changes < MIN_DIRECTION_CHANGES:
        return None
    if 4 not in areas or not any(int(area) != 4 for area in areas):
        return None
    return areas


def _render_grid(N: int, start_col: int, areas: list[int], vertical_first: bool) -> np.ndarray:
    H = 4 * int(N) - 1
    grid = np.zeros((int(H), GRID_W), dtype=int)
    grid[:, DIVIDER_COL] = BASE_DIVIDER_COLOR

    ctrl_color = CTRL_VERTICAL_FIRST if bool(vertical_first) else CTRL_HORIZONTAL_FIRST
    for rr in _control_rows(int(H)):
        grid[int(rr), DIVIDER_COL] = int(ctrl_color)

    grid[GRAY_START_ROW, DIVIDER_COL + 1 + int(start_col)] = GRAY

    for i, area in enumerate(areas):
        tile_col = TILE_COLS[i // int(N)]
        r0 = TILE_ROW_STRIDE * (i % int(N))
        grid[r0 : r0 + 3, tile_col : tile_col + 3] = AREA_TO_TEMPLATE[int(area)] * AREA_TO_COLOR[int(area)]
    return grid


def _gen_attempt(vertical_first: bool) -> np.ndarray | None:
    N = random.randint(N_MIN, N_MAX)
    H = 4 * int(N) - 1
    start_col = random.randint(0, LEFT_W - 1)
    areas = _sample_path(int(N), int(start_col), bool(vertical_first))
    if areas is None:
        return None

    colors = [int(AREA_TO_COLOR[int(area)]) for area in areas]
    fused_out = _simulate_output(int(H), int(start_col), areas, colors, vertical_first=bool(vertical_first))
    legacy_out = _simulate_legacy_output(int(H), int(start_col), areas, colors)
    toggled_out = _simulate_output(int(H), int(start_col), areas, colors, vertical_first=not bool(vertical_first))

    if np.array_equal(fused_out, legacy_out):
        return None
    if np.array_equal(fused_out, toggled_out):
        return None

    return _render_grid(int(N), int(start_col), areas, bool(vertical_first))


def generate_input() -> np.ndarray:
    global NEXT_VERTICAL_FIRST
    target_vertical_first = bool(NEXT_VERTICAL_FIRST)
    for _ in range(MAX_ATTEMPTS):
        try:
            inp = _gen_attempt(bool(target_vertical_first))
            if inp is None:
                continue
            out = main(inp)
            if out.shape != (inp.shape[0], LEFT_W):
                continue
            NEXT_VERTICAL_FIRST = not bool(target_vertical_first)
            return inp
        except Exception:
            continue
    raise RuntimeError(f"generate_input: failed after {MAX_ATTEMPTS} attempts")
