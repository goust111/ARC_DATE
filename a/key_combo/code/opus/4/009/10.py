from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, colors_present, crop_bbox, find_components, paste


BG = Color.BLACK
PANEL_COLOR = Color.TEAL
CROSS_COLOR = Color.ORANGE
PANEL_W = 5
CANVAS_W_RANGE = (11, 18)
LEGEND_COUNT_CHOICES = (4, 5)
GEN_ATTEMPTS = 500
MARKER_CROSS_PROB = 0.45
PALETTE = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.GREY, Color.PINK, Color.MAROON]


def _mask(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


# Areas overlap across the two classes so pixel-count shortcuts do not work.
HORIZONTAL_SHAPES = [
    _mask(["111"]),
    _mask(["111", "010"]),
    _mask(["111", "100"]),
    _mask(["111", "001"]),
    _mask(["110", "011"]),
    _mask(["011", "110"]),
]

VERTICAL_SHAPES = [
    _mask(["1", "1", "1"]),
    _mask(["1", "1", "1", "1"]),
    _mask(["10", "11", "10"]),
    _mask(["01", "11", "01"]),
    _mask(["10", "11", "01"]),
    _mask(["01", "11", "10"]),
]


def _panel_bounds(grid: np.ndarray) -> tuple[int, int]:
    full_cols = [int(c) for c in range(grid.shape[1]) if np.all(grid[:, c] == PANEL_COLOR)]
    assert len(full_cols) >= 2, "expected two full-height panel border columns"
    left = full_cols[0]
    right = next(c for c in full_cols[1:] if c > left + 1)
    return left, right


def _legend_orientation_map(grid: np.ndarray) -> dict[int, str]:
    g = np.asarray(grid, dtype=int)
    panel_c0, panel_c1 = _panel_bounds(g)
    sep_rows = [int(r) for r in range(g.shape[0]) if np.all(g[r, panel_c0 : panel_c1 + 1] == PANEL_COLOR)]
    assert len(sep_rows) >= 2, "expected legend separators"

    mapping: dict[int, str] = {}
    for top, bottom in zip(sep_rows, sep_rows[1:]):
        if bottom - top <= 1:
            continue
        patch = g[top + 1 : bottom, panel_c0 + 1 : panel_c1]
        cols = colors_present(patch, exclude={BG, PANEL_COLOR})
        if not cols:
            continue
        assert len(cols) == 1, "each legend slot must contain exactly one color"
        color = int(cols[0])
        comps = find_components(
            patch,
            bg=BG,
            colors=[color],
            connectivity=4,
            monochromatic=True,
            return_format="dict",
        )
        assert len(comps) == 1, "each legend slot must contain exactly one component"
        sprite = crop_bbox(patch, comps[0]["bbox"])
        mask = sprite == color
        height, width = map(int, mask.shape)
        assert width != height, "legend bbox must not be square"
        mapping[color] = "H" if width > height else "V"

    assert mapping, "expected at least one legend entry"
    return mapping


def _markers_from_canvas(grid: np.ndarray, mapping: dict[int, str]) -> list[tuple[int, int, int, str]]:
    g = np.asarray(grid, dtype=int)
    _panel_c0, panel_c1 = _panel_bounds(g)
    canvas_c0 = panel_c1 + 1
    canvas = g[:, canvas_c0:]
    markers: list[tuple[int, int, int, str]] = []
    for color, orient in mapping.items():
        for rr, cc in np.argwhere(canvas == color):
            markers.append((int(rr), int(cc) + canvas_c0, int(color), orient))
    return markers


def _render_output(grid: np.ndarray, markers: list[tuple[int, int, int, str]]) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    _panel_c0, panel_c1 = _panel_bounds(g)
    canvas_c0 = panel_c1 + 1
    out = g.copy()

    row_color: dict[int, int] = {}
    col_color: dict[int, int] = {}
    marker_color: dict[tuple[int, int], int] = {}
    for r, c, color, orient in markers:
        assert (r, c) not in marker_color, "markers must occupy distinct cells"
        marker_color[(r, c)] = int(color)
        if orient == "H":
            assert r not in row_color or row_color[r] == color, "ambiguous horizontal row"
            row_color[r] = int(color)
        else:
            assert c not in col_color or col_color[c] == color, "ambiguous vertical column"
            col_color[c] = int(color)

    for r, color in row_color.items():
        out[r, canvas_c0:] = int(color)
    for c, color in col_color.items():
        out[:, c] = int(color)
    for r in row_color:
        for c in col_color:
            if (r, c) not in marker_color:
                out[r, c] = CROSS_COLOR
    for (r, c), color in marker_color.items():
        out[r, c] = int(color)
    out[:, :canvas_c0] = g[:, :canvas_c0]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Classify legend colors by shape aspect ratio, emit full-span rays, and fill their cross points.

    Input structure:
    - The left 5 columns form a teal legend panel with full-height teal side borders and teal separator rows.
    - Each non-empty legend slot contains one 4-connected monochrome shape whose bbox is strictly wider or taller.
    - The canvas to the right is black except for scattered single-pixel colored markers whose colors appear in the legend.

    Rule:
    - Read each legend slot and crop the shape to its minimal bbox.
    - If the bbox width is greater than its height, that color is a horizontal emitter; if the height is greater, it is a vertical emitter.
    - Every horizontal marker paints its entire canvas row in its own color, and every vertical marker paints its entire canvas column in its own color.
    - Every horizontal/vertical crossing becomes orange, except that original marker pixels always keep their own input color.
    - Preserve the legend panel unchanged.

    Key insight:
    - The legend does not translate colors into new shapes; the only signal it carries is whether each color's sample shape is wider or taller, which determines ray direction.
    """
    mapping = _legend_orientation_map(np.asarray(grid, dtype=int))
    markers = _markers_from_canvas(np.asarray(grid, dtype=int), mapping)
    return _render_output(np.asarray(grid, dtype=int), markers)


def _entry_height(shape_h: int) -> int:
    if shape_h == 1:
        return 2
    if shape_h >= 4:
        return shape_h
    return shape_h + random.randint(0, 1)


def _sample_spaced(count: int, size: int, min_delta: int) -> list[int] | None:
    if count == 0:
        return []
    if count > size:
        return None
    population = list(range(size))
    for _ in range(250):
        picks = sorted(random.sample(population, count))
        if all(b - a >= min_delta for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _marker_cell_ok(existing: set[tuple[int, int]], r: int, c: int) -> bool:
    for rr, cc in existing:
        if max(abs(int(r) - rr), abs(int(c) - cc)) <= 1:
            return False
    return True


def _pick_marker_colors(colors: list[int], count: int) -> list[int]:
    caps = {int(color): 2 for color in colors}
    chosen: list[int] = []
    while len(chosen) < count:
        options = [color for color in colors if caps[int(color)] > 0]
        if not options:
            return []
        color = int(random.choice(options))
        caps[color] -= 1
        chosen.append(color)
    random.shuffle(chosen)
    return chosen


def _build_panel_and_meta(entries: list[tuple[int, str]]) -> tuple[np.ndarray, dict[int, str]]:
    layouts: list[dict[str, int | str | np.ndarray]] = []
    for color, orient in entries:
        shape = random.choice(HORIZONTAL_SHAPES if orient == "H" else VERTICAL_SHAPES)
        shape_h, shape_w = map(int, shape.shape)
        slot_h = _entry_height(shape_h)
        top_pad = random.randint(0, slot_h - shape_h)
        left_pad = random.randint(0, 3 - shape_w)
        layouts.append(
            {
                "color": int(color),
                "orient": orient,
                "shape": shape,
                "slot_h": slot_h,
                "top_pad": top_pad,
                "left_pad": left_pad,
            }
        )

    height = 1 + sum(int(layout["slot_h"]) for layout in layouts) + len(layouts)
    panel = np.full((height, PANEL_W), BG, dtype=int)
    panel[:, 0] = PANEL_COLOR
    panel[:, -1] = PANEL_COLOR

    cursor = 0
    panel[cursor, :] = PANEL_COLOR
    for layout in layouts:
        slot_h = int(layout["slot_h"])
        top = cursor + 1
        bottom = top + slot_h
        color = int(layout["color"])
        shape = np.where(layout["shape"], color, BG).astype(int)
        paste(
            panel,
            shape,
            top + int(layout["top_pad"]),
            1 + int(layout["left_pad"]),
            transparent=BG,
        )
        cursor = bottom
        panel[cursor, :] = PANEL_COLOR

    mapping = {int(layout["color"]): str(layout["orient"]) for layout in layouts}
    return panel, mapping


def _marker_positions(
    height: int,
    canvas_w: int,
    h_colors: list[int],
    v_colors: list[int],
    need_cross_marker: bool,
) -> list[dict[str, int | str]] | None:
    if not h_colors or not v_colors:
        return None

    max_h = min(3, len(h_colors) * 2)
    max_v = min(3, len(v_colors) * 2)
    if max_h == 0 or max_v == 0:
        return None

    for _ in range(250):
        n_h = random.randint(1, max_h)
        n_v = random.randint(1, max_v)
        if n_h + n_v < 3:
            continue
        if n_h * n_v > 6:
            continue
        if need_cross_marker and n_h * n_v < 2:
            continue

        row_gap = 3 if height >= 10 else 2
        col_gap = 3 if canvas_w >= 10 else 2
        h_rows = _sample_spaced(n_h, height, row_gap) or _sample_spaced(n_h, height, 2) or _sample_spaced(n_h, height, 1)
        v_cols_local = _sample_spaced(n_v, canvas_w, col_gap) or _sample_spaced(n_v, canvas_w, 2) or _sample_spaced(n_v, canvas_w, 1)
        if h_rows is None or v_cols_local is None:
            continue

        h_marker_colors = _pick_marker_colors(h_colors, n_h)
        v_marker_colors = _pick_marker_colors(v_colors, n_v)
        if not h_marker_colors or not v_marker_colors:
            continue

        occupied: set[tuple[int, int]] = set()
        markers: list[dict[str, int | str]] = []
        full_v_cols = [PANEL_W + c for c in v_cols_local]
        cross_marker_seen = False

        ok = True
        for row, color in zip(h_rows, h_marker_colors):
            prefer_cross = need_cross_marker and not cross_marker_seen
            if not prefer_cross:
                prefer_cross = bool(full_v_cols) and random.random() < MARKER_CROSS_PROB
            pref_cols = random.sample(full_v_cols, len(full_v_cols)) if prefer_cross else []
            all_cols = list(range(PANEL_W, PANEL_W + canvas_w))
            rem_cols = [c for c in all_cols if c not in pref_cols]
            choices = pref_cols + random.sample(rem_cols, len(rem_cols))
            chosen_col = None
            for col in choices:
                if _marker_cell_ok(occupied, row, col):
                    chosen_col = int(col)
                    break
            if chosen_col is None:
                ok = False
                break
            occupied.add((int(row), int(chosen_col)))
            if chosen_col in full_v_cols:
                cross_marker_seen = True
            markers.append({"row": int(row), "col": int(chosen_col), "color": int(color), "orient": "H"})
        if not ok:
            continue

        full_h_rows = [int(r) for r in h_rows]
        for col_local, color in zip(v_cols_local, v_marker_colors):
            col = PANEL_W + int(col_local)
            prefer_cross = need_cross_marker and not cross_marker_seen
            if not prefer_cross:
                prefer_cross = bool(full_h_rows) and random.random() < MARKER_CROSS_PROB
            pref_rows = random.sample(full_h_rows, len(full_h_rows)) if prefer_cross else []
            all_rows = list(range(height))
            rem_rows = [r for r in all_rows if r not in pref_rows]
            choices = pref_rows + random.sample(rem_rows, len(rem_rows))
            chosen_row = None
            for row in choices:
                if _marker_cell_ok(occupied, row, col):
                    chosen_row = int(row)
                    break
            if chosen_row is None:
                ok = False
                break
            occupied.add((int(chosen_row), int(col)))
            if chosen_row in full_h_rows:
                cross_marker_seen = True
            markers.append({"row": int(chosen_row), "col": int(col), "color": int(color), "orient": "V"})
        if not ok:
            continue

        marker_cells = {(int(marker["row"]), int(marker["col"])) for marker in markers}
        orange_cells = {
            (int(h_marker["row"]), int(v_marker["col"]))
            for h_marker in markers
            for v_marker in markers
            if h_marker["orient"] == "H" and v_marker["orient"] == "V"
        } - marker_cells
        if not orange_cells:
            continue
        if need_cross_marker and not cross_marker_seen:
            continue
        return markers

    return None


def _expected_from_meta(inp: np.ndarray, markers: list[dict[str, int | str]]) -> np.ndarray:
    marker_tuples = [
        (int(marker["row"]), int(marker["col"]), int(marker["color"]), str(marker["orient"]))
        for marker in markers
    ]
    return _render_output(inp, marker_tuples)


def _validate_generated(inp: np.ndarray, legend_map: dict[int, str], markers: list[dict[str, int | str]]) -> None:
    assert inp.ndim == 2
    assert max(inp.shape) <= 30
    assert np.all(inp[:, 0] == PANEL_COLOR) and np.all(inp[:, PANEL_W - 1] == PANEL_COLOR)
    assert set(legend_map.values()) == {"H", "V"}
    assert _legend_orientation_map(inp) == legend_map

    canvas = inp[:, PANEL_W:]
    used_colors = colors_present(canvas, exclude={BG})
    assert used_colors, "expected markers on canvas"
    assert set(used_colors) <= set(legend_map)
    assert any(legend_map[color] == "H" for color in used_colors)
    assert any(legend_map[color] == "V" for color in used_colors)

    counts_by_color = {color: int(np.sum(canvas == color)) for color in used_colors}
    assert all(count <= 2 for count in counts_by_color.values())

    h_markers = [marker for marker in markers if marker["orient"] == "H"]
    v_markers = [marker for marker in markers if marker["orient"] == "V"]
    assert h_markers and v_markers
    assert len({int(marker["row"]) for marker in h_markers}) == len(h_markers)
    assert len({int(marker["col"]) for marker in v_markers}) == len(v_markers)

    marker_cells = {(int(marker["row"]), int(marker["col"])) for marker in markers}
    orange_cells = {(int(h["row"]), int(v["col"])) for h in h_markers for v in v_markers} - marker_cells
    assert orange_cells, "need at least one visible orange intersection"

    expected = _expected_from_meta(inp, markers)
    actual = main(inp)
    assert np.array_equal(actual, expected)
    assert np.array_equal(actual[:, :PANEL_W], inp[:, :PANEL_W])
    assert np.all(actual[[r for r, _c in orange_cells], [c for _r, c in orange_cells]] == CROSS_COLOR)
    for marker in markers:
        r = int(marker["row"])
        c = int(marker["col"])
        color = int(marker["color"])
        assert int(actual[r, c]) == color


def _gen_attempt() -> np.ndarray | None:
    legend_count = random.choice(LEGEND_COUNT_CHOICES)
    legend_colors = random.sample(PALETTE, legend_count)
    while True:
        legend_map = {int(color): random.choice(["H", "V"]) for color in legend_colors}
        if set(legend_map.values()) == {"H", "V"}:
            break

    entries = [(color, legend_map[color]) for color in legend_colors]
    panel, legend_map = _build_panel_and_meta(entries)
    height = int(panel.shape[0])
    if height > 30:
        return None

    canvas_w = random.randint(*CANVAS_W_RANGE)
    total_w = PANEL_W + canvas_w
    if total_w > 30:
        return None

    h_colors = [color for color, orient in legend_map.items() if orient == "H"]
    v_colors = [color for color, orient in legend_map.items() if orient == "V"]
    need_cross_marker = random.random() < 0.55
    markers = _marker_positions(height, canvas_w, h_colors, v_colors, need_cross_marker)
    if markers is None:
        return None

    inp = np.zeros((height, total_w), dtype=int)
    inp[:, :PANEL_W] = panel
    for marker in markers:
        inp[int(marker["row"]), int(marker["col"])] = int(marker["color"])

    _validate_generated(inp, legend_map, markers)
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
