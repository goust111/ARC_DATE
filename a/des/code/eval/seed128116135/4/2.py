import numpy as np


BG = 8  # azure
OPPOSITE = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}
COLOR_POOL = [c for c in range(10) if c != BG]
RETRY_LIMIT = 300


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _extract_legend(grid: np.ndarray, bg: int) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    comps = _components(grid != bg)
    candidates = []
    for comp in comps:
        colors = {int(grid[r, c]) for r, c in comp}
        if len(comp) == 2 and len(colors) == 2:
            r0, c0, r1, c1 = _bbox(comp)
            if r0 == r1 and c1 == c0 + 1:
                candidates.append(comp)
    assert len(candidates) == 1, "expected a unique 1x2 legend bar"
    comp = candidates[0]
    ordered = sorted(comp, key=lambda rc: (rc[1], rc[0]))
    colors = tuple(int(grid[r, c]) for r, c in ordered)
    return colors, comp


def _analyze_piece(grid: np.ndarray, comp: list[tuple[int, int]]) -> dict:
    r0, c0, r1, c1 = _bbox(comp)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in comp:
        mask[r - r0, c - c0] = True

    row_counts = mask.sum(axis=1)
    col_counts = mask.sum(axis=0)
    row_one = np.flatnonzero(row_counts == 1)
    col_one = np.flatnonzero(col_counts == 1)
    assert (len(row_one) == 1) ^ (len(col_one) == 1), "piece must have exactly one tab axis"

    area = int(mask.sum())
    if len(row_one) == 1:
        tab_row = int(row_one[0])
        tab_side = "top" if tab_row == 0 else "bottom"
        base_horiz_shift = 0
        base_vert_shift = 1 if tab_side == "top" else 0
        base_w = int(row_counts.max())
        assert base_w >= 3 and area % base_w == 0
        base_h = area // base_w
        tab_offset = int(np.flatnonzero(mask[tab_row])[0]) - base_horiz_shift
        if col_counts[0] == base_h - 1:
            notch_side = "left"
            notch_col = 0
        elif col_counts[-1] == base_h - 1:
            notch_side = "right"
            notch_col = mask.shape[1] - 1
        else:
            raise AssertionError("expected a left/right notch")
        misses = [
            rr
            for rr in range(base_h)
            if not mask[base_vert_shift + rr, notch_col]
        ]
        assert len(misses) == 1
        notch_offset = int(misses[0])
    else:
        tab_col = int(col_one[0])
        tab_side = "left" if tab_col == 0 else "right"
        base_horiz_shift = 1 if tab_side == "left" else 0
        base_vert_shift = 0
        base_h = int(col_counts.max())
        assert base_h >= 3 and area % base_h == 0
        base_w = area // base_h
        tab_offset = int(np.flatnonzero(mask[:, tab_col])[0]) - base_vert_shift
        if row_counts[0] == base_w - 1:
            notch_side = "top"
            notch_row = 0
        elif row_counts[-1] == base_w - 1:
            notch_side = "bottom"
            notch_row = mask.shape[0] - 1
        else:
            raise AssertionError("expected a top/bottom notch")
        misses = [
            cc
            for cc in range(base_w)
            if not mask[notch_row, base_horiz_shift + cc]
        ]
        assert len(misses) == 1
        notch_offset = int(misses[0])

    color = int(grid[comp[0][0], comp[0][1]])
    assert all(int(grid[r, c]) == color for r, c in comp), "pieces must be monochrome"
    return {
        "color": color,
        "mask": mask,
        "cells": [(int(r), int(c)) for r, c in zip(*np.where(mask))],
        "base_h": int(base_h),
        "base_w": int(base_w),
        "base_r0": int(base_vert_shift),
        "base_c0": int(base_horiz_shift),
        "tab_side": tab_side,
        "tab_offset": int(tab_offset),
        "notch_side": notch_side,
        "notch_offset": int(notch_offset),
    }


def _render_layout(layout: dict[str, dict], horiz_color: int, vert_color: int) -> np.ndarray:
    tl = layout["TL"]
    tr = layout["TR"]
    bl = layout["BL"]
    br = layout["BR"]

    row1 = tl["base_h"]
    row2 = bl["base_h"]
    col1 = tl["base_w"]
    col2 = tr["base_w"]

    assert tr["base_h"] == row1 and br["base_h"] == row2
    assert bl["base_w"] == col1 and br["base_w"] == col2

    h = row1 + row2
    w = col1 + col2
    owner = np.full((h, w), -1, dtype=int)
    out = np.full((h, w), -1, dtype=int)
    placements = {
        "TL": (0, 0, 0),
        "TR": (0, col1, 1),
        "BL": (row1, 0, 2),
        "BR": (row1, col1, 3),
    }

    for name, (base_r, base_c, owner_id) in placements.items():
        piece = layout[name]
        anchor_r = base_r - piece["base_r0"]
        anchor_c = base_c - piece["base_c0"]
        for rr, cc in piece["cells"]:
            ar = anchor_r + rr
            ac = anchor_c + cc
            assert 0 <= ar < h and 0 <= ac < w
            assert owner[ar, ac] == -1
            owner[ar, ac] = owner_id
            out[ar, ac] = piece["color"]

    assert np.all(owner != -1), "assembled pieces must cover the output rectangle"

    horiz_mask = np.zeros((h, w), dtype=bool)
    vert_mask = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            cur = owner[r, c]
            if (r > 0 and owner[r - 1, c] != cur) or (r + 1 < h and owner[r + 1, c] != cur):
                horiz_mask[r, c] = True
            if (c > 0 and owner[r, c - 1] != cur) or (c + 1 < w and owner[r, c + 1] != cur):
                vert_mask[r, c] = True

    out[horiz_mask] = horiz_color
    out[vert_mask] = vert_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble four scattered jigsaw-like quadrants into a rectangle and color seams from the 1x2 legend bar.

    The four monochrome pieces are translated only. Each has one tab and one notch on distinct sides, and
    the assembled rectangle is always a 2x2 arrangement: every corner piece touches exactly one horizontal
    and one vertical neighbor.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    (horiz_color, vert_color), legend_comp = _extract_legend(g, bg)
    legend_set = set(legend_comp)

    all_comps = _components(g != bg)
    piece_comps = [comp for comp in all_comps if set(comp) != legend_set]
    assert len(piece_comps) == 4, "expected four puzzle pieces"
    pieces = [_analyze_piece(g, comp) for comp in piece_comps]

    left_of: dict[int, int] = {}
    right_of: dict[int, int] = {}
    up_of: dict[int, int] = {}
    down_of: dict[int, int] = {}

    for i, piece in enumerate(pieces):
        matches = []
        for j, other in enumerate(pieces):
            if i == j:
                continue
            if other["notch_side"] != OPPOSITE[piece["tab_side"]]:
                continue
            if other["notch_offset"] != piece["tab_offset"]:
                continue
            if piece["tab_side"] in ("left", "right"):
                if piece["base_h"] != other["base_h"]:
                    continue
            else:
                if piece["base_w"] != other["base_w"]:
                    continue
            matches.append(j)
        assert len(matches) == 1, "every tab must match a unique complementary notch"
        j = matches[0]

        if piece["tab_side"] == "right":
            right_of[i] = j
            left_of[j] = i
        elif piece["tab_side"] == "left":
            left_of[i] = j
            right_of[j] = i
        elif piece["tab_side"] == "bottom":
            down_of[i] = j
            up_of[j] = i
        else:
            up_of[i] = j
            down_of[j] = i

    top_left = [
        i
        for i in range(4)
        if i in right_of and i in down_of and i not in left_of and i not in up_of
    ]
    assert len(top_left) == 1, "layout must contain a unique top-left piece"
    tl = top_left[0]
    tr = right_of[tl]
    bl = down_of[tl]
    br = right_of[bl]
    assert down_of[tr] == br and left_of[tr] == tl and up_of[bl] == tl and up_of[br] == tr
    assert left_of[br] == bl and down_of[tl] == bl

    layout = {"TL": pieces[tl], "TR": pieces[tr], "BL": pieces[bl], "BR": pieces[br]}
    return _render_layout(layout, horiz_color, vert_color)


def _build_piece(h: int, w: int, color: int, tab_side: str, tab_offset: int, notch_side: str, notch_offset: int) -> dict:
    top_extra = 1 if tab_side == "top" else 0
    bottom_extra = 1 if tab_side == "bottom" else 0
    left_extra = 1 if tab_side == "left" else 0
    right_extra = 1 if tab_side == "right" else 0
    mask = np.zeros((h + top_extra + bottom_extra, w + left_extra + right_extra), dtype=bool)
    base_r0 = top_extra
    base_c0 = left_extra
    mask[base_r0 : base_r0 + h, base_c0 : base_c0 + w] = True

    if notch_side == "top":
        mask[base_r0, base_c0 + notch_offset] = False
    elif notch_side == "bottom":
        mask[base_r0 + h - 1, base_c0 + notch_offset] = False
    elif notch_side == "left":
        mask[base_r0 + notch_offset, base_c0] = False
    else:
        mask[base_r0 + notch_offset, base_c0 + w - 1] = False

    if tab_side == "top":
        mask[0, base_c0 + tab_offset] = True
    elif tab_side == "bottom":
        mask[base_r0 + h, base_c0 + tab_offset] = True
    elif tab_side == "left":
        mask[base_r0 + tab_offset, 0] = True
    else:
        mask[base_r0 + tab_offset, base_c0 + w] = True

    return {
        "color": int(color),
        "mask": mask,
        "cells": [(int(r), int(c)) for r, c in zip(*np.where(mask))],
        "base_h": int(h),
        "base_w": int(w),
        "base_r0": int(base_r0),
        "base_c0": int(base_c0),
        "tab_side": tab_side,
        "tab_offset": int(tab_offset),
        "notch_side": notch_side,
        "notch_offset": int(notch_offset),
    }


def _boxes_far_enough(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0


def _paint_patch(canvas: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    sub = canvas[top : top + h, left : left + w]
    mask = patch != BG
    assert np.all(sub[mask] == BG)
    sub[mask] = patch[mask]


def _piece_patch(piece: dict) -> np.ndarray:
    patch = np.full(piece["mask"].shape, BG, dtype=int)
    patch[piece["mask"]] = piece["color"]
    return patch


def _sample_distinct_offsets(rng: np.random.Generator, len_a: int, len_b: int) -> tuple[int, int]:
    for _ in range(50):
        a = int(rng.integers(1, len_a - 1))
        b = int(rng.integers(1, len_b - 1))
        if len_a != len_b or a != b:
            return a, b
    raise AssertionError("failed to sample non-ambiguous offsets")


def _make_case(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    seam_h, seam_v, *piece_colors = rng.choice(COLOR_POOL, size=6, replace=False).tolist()
    h_top = int(rng.integers(4, 8))
    h_bottom = int(rng.integers(4, 8))
    w_left = int(rng.integers(4, 8))
    w_right = int(rng.integers(4, 8))
    y_top, y_bottom = _sample_distinct_offsets(rng, h_top, h_bottom)
    x_left, x_right = _sample_distinct_offsets(rng, w_left, w_right)

    clockwise = bool(rng.integers(0, 2))
    if clockwise:
        layout = {
            "TL": _build_piece(h_top, w_left, piece_colors[0], "right", y_top, "bottom", x_left),
            "TR": _build_piece(h_top, w_right, piece_colors[1], "bottom", x_right, "left", y_top),
            "BL": _build_piece(h_bottom, w_left, piece_colors[2], "top", x_left, "right", y_bottom),
            "BR": _build_piece(h_bottom, w_right, piece_colors[3], "left", y_bottom, "top", x_right),
        }
    else:
        layout = {
            "TL": _build_piece(h_top, w_left, piece_colors[0], "bottom", x_left, "right", y_top),
            "TR": _build_piece(h_top, w_right, piece_colors[1], "left", y_top, "bottom", x_right),
            "BL": _build_piece(h_bottom, w_left, piece_colors[2], "right", y_bottom, "top", x_left),
            "BR": _build_piece(h_bottom, w_right, piece_colors[3], "top", x_right, "left", y_bottom),
        }

    expected = _render_layout(layout, seam_h, seam_v)

    patches = [_piece_patch(layout[name]) for name in ("TL", "TR", "BL", "BR")]
    patch_boxes = []
    grid_h = int(max(20, expected.shape[0] + 10) + rng.integers(0, 5))
    grid_w = int(max(20, expected.shape[1] + 10) + rng.integers(0, 5))
    canvas = np.full((grid_h, grid_w), BG, dtype=int)

    corner = int(rng.integers(0, 4))
    if corner == 0:
        legend_cells = [(0, 0, seam_h), (0, 1, seam_v)]
    elif corner == 1:
        legend_cells = [(0, grid_w - 2, seam_h), (0, grid_w - 1, seam_v)]
    elif corner == 2:
        legend_cells = [(grid_h - 1, 0, seam_h), (grid_h - 1, 1, seam_v)]
    else:
        legend_cells = [(grid_h - 1, grid_w - 2, seam_h), (grid_h - 1, grid_w - 1, seam_v)]
    for r, c, color in legend_cells:
        canvas[r, c] = color
    legend_box = (
        min(r for r, _, _ in legend_cells),
        min(c for _, c, _ in legend_cells),
        max(r for r, _, _ in legend_cells),
        max(c for _, c, _ in legend_cells),
    )
    patch_boxes.append(legend_box)

    order = rng.permutation(4).tolist()
    for idx in order:
        patch = patches[idx]
        ph, pw = patch.shape
        placed = False
        for _ in range(240):
            top = int(rng.integers(1, grid_h - ph))
            left = int(rng.integers(1, grid_w - pw))
            box = (top, left, top + ph - 1, left + pw - 1)
            if all(_boxes_far_enough(box, other, pad=1) for other in patch_boxes):
                _paint_patch(canvas, patch, top, left)
                patch_boxes.append(box)
                placed = True
                break
        assert placed, "failed to scatter a piece without overlap"

    return canvas, expected


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(RETRY_LIMIT):
        try:
            grid, expected = _make_case(rng)
        except AssertionError:
            continue
        try:
            got = main(grid)
        except AssertionError:
            continue
        if np.array_equal(got, expected):
            return grid
    raise RuntimeError("failed to generate a solver-consistent input")
