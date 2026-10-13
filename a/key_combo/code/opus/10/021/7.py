from __future__ import annotations

import hashlib
import random

import numpy as np


BACKGROUND = 0
SEPARATOR = 5
PANEL_COLORS = (2, 1, 3)
PANEL_SIZE = 9
AXIS_ROW = PANEL_SIZE // 2
HALF_HEIGHT_CHOICES = (3, 4)
PATCH_WIDTH_RANGE = (5, 8)
HOLE_SIZE_PREFERENCE = ((3, 3), (2, 3), (3, 2), (2, 2))
MAX_ATTEMPTS = 400

_SEEN_INPUT_SIGS: set[str] = set()
_SEEN_OUTPUT_SIGS: set[str] = set()
_SELECTED_SIDES = ("top", "bottom", "top")
_HOLE_SIDE_PATTERNS = [
    pattern
    for pattern in (
        ("top", "top", "top"),
        ("top", "top", "bottom"),
        ("top", "bottom", "bottom"),
        ("bottom", "top", "top"),
        ("bottom", "top", "bottom"),
        ("bottom", "bottom", "top"),
        ("bottom", "bottom", "bottom"),
        ("top", "bottom", "top"),
    )
    if pattern not in {_SELECTED_SIDES, tuple("bottom" if side == "top" else "top" for side in _SELECTED_SIDES)}
]


def _sig(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes() + str(grid.shape).encode()).hexdigest()


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _mask_connected(mask: np.ndarray) -> bool:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if not coords:
        return False
    seen = {coords[0]}
    stack = [coords[0]]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c, h, w):
            if mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(coords)


def _hole_candidates(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = mask.shape
    candidates: list[tuple[int, int, int, int]] = []
    for hole_h, hole_w in HOLE_SIZE_PREFERENCE:
        if hole_h > h or hole_w > w:
            continue
        for r0 in range(h - hole_h + 1):
            for c0 in range(w - hole_w + 1):
                if np.all(mask[r0 : r0 + hole_h, c0 : c0 + hole_w]):
                    candidates.append((r0, c0, hole_h, hole_w))
    return candidates


def _generate_top_mask(half_h: int, width: int) -> np.ndarray:
    for _ in range(200):
        mask = np.ones((half_h, width), dtype=bool)
        for _ in range(random.randint(2, 4)):
            kind = random.choice(("top", "bottom", "left", "right", "corner"))
            if kind in {"top", "bottom"} and width >= 3:
                bite_w = random.randint(1, min(2, width - 1))
                c0 = random.randint(0, width - bite_w)
                row = 0 if kind == "top" else half_h - 1
                mask[row, c0 : c0 + bite_w] = False
            elif kind in {"left", "right"} and half_h >= 2:
                bite_h = random.randint(1, min(2, half_h - 1))
                r0 = random.randint(0, half_h - bite_h)
                col = 0 if kind == "left" else width - 1
                mask[r0 : r0 + bite_h, col] = False
            else:
                mask[random.choice((0, half_h - 1)), random.choice((0, width - 1))] = False

        if not mask[0].any() or not mask[-1].any():
            continue
        if np.array_equal(mask, np.flipud(mask)):
            continue
        if int(mask.sum()) < half_h * width - 5:
            continue
        if not _mask_connected(mask):
            continue
        if not _hole_candidates(mask):
            continue
        return mask
    raise AssertionError("failed to build an asymmetric top-half mask")


def _separator_cols(grid: np.ndarray) -> tuple[int, int]:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == SEPARATOR)]
    assert len(cols) == 2, "expected exactly two solid separator columns"
    return int(cols[0]), int(cols[1])


def _split_panels(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    sep0, sep1 = _separator_cols(g)
    widths = [sep0, sep1 - sep0 - 1, g.shape[1] - sep1 - 1]
    assert len(set(widths)) == 1, "separator bars must split the grid into equal-width panels"
    panels = [
        g[:, :sep0].copy(),
        g[:, sep0 + 1 : sep1].copy(),
        g[:, sep1 + 1 :].copy(),
    ]
    assert all(panel.shape == (g.shape[0], widths[0]) for panel in panels)
    return panels


def _panel_foreground_color(panel: np.ndarray) -> int:
    colors = sorted(set(map(int, np.unique(panel))) - {BACKGROUND})
    assert len(colors) == 1, "each panel must contain exactly one non-black foreground color"
    return colors[0]


def _repair_panel(panel: np.ndarray) -> np.ndarray:
    g = np.asarray(panel, dtype=int)
    color = _panel_foreground_color(g)
    h, _ = g.shape
    axis = h // 2
    repaired = g.copy()
    for r in range(h):
        mirror_r = 2 * axis - r
        if not (0 <= mirror_r < h):
            continue
        fill_mask = (g[r] == BACKGROUND) & (g[mirror_r] == color)
        repaired[r, fill_mask] = color
    return repaired


def _crop_symmetric_halves(panel: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    coords = np.argwhere(panel != BACKGROUND)
    assert len(coords) > 0, "panel must contain a foreground motif"
    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))
    axis = panel.shape[0] // 2
    assert not np.any(panel[axis] != BACKGROUND), "the symmetry axis row must remain empty"
    assert r0 < axis < r1, "motif must straddle the horizontal symmetry axis"

    top = panel[r0:axis, c0 : c1 + 1]
    bottom = panel[axis + 1 : r1 + 1, c0 : c1 + 1]
    assert top.shape == bottom.shape and top.size > 0
    assert np.array_equal(top, np.flipud(bottom)), "repaired motif must be mirror-symmetric"
    return top.copy(), bottom.copy(), (r0, c0, r1, c1)


def _piece_from_bbox(panel: np.ndarray, bbox: tuple[int, int, int, int], side: str) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    axis = panel.shape[0] // 2
    if side == "top":
        return panel[r0:axis, c0 : c1 + 1].copy()
    if side == "bottom":
        return panel[axis + 1 : r1 + 1, c0 : c1 + 1].copy()
    raise ValueError(f"unknown side: {side}")


def _hole_info(panel: np.ndarray, repaired: np.ndarray) -> tuple[str, tuple[int, int, int, int]]:
    hole_mask = (panel == BACKGROUND) & (repaired != BACKGROUND)
    coords = np.argwhere(hole_mask)
    assert len(coords) > 0, "each panel must contain a repairable hole"
    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))
    assert np.all(hole_mask[r0 : r1 + 1, c0 : c1 + 1]), "the hole must be one compact rectangle"
    axis = panel.shape[0] // 2
    if r1 < axis:
        return "top", (r0, c0, r1, c1)
    assert r0 > axis, "holes cannot cross the symmetry axis"
    return "bottom", (r0, c0, r1, c1)


def main(grid: np.ndarray) -> np.ndarray:
    """Repair each panel's horizontal mirror-symmetric motif, then merge the axis-selected half from each panel.

    Input structure:
    - The grid has a black background and two solid grey separator columns that split it into three equal square panels.
    - Each panel contains one monochrome motif whose top and bottom parts are horizontal-mirror copies around the panel's middle row.
    - One compact black rectangular patch is missing from either the top or bottom half of each panel, while the mirrored counterpart stays intact.

    Rule:
    - Identify the three panels from left to right and repair every missing black cell by copying the color from its mirror position across the panel's horizontal midline.
    - Crop the repaired motif in each panel to its foreground bounding box and split it into the top half above the axis and the bottom half below the axis.
    - Panel 0 contributes its top half, panel 1 contributes its bottom half, and panel 2 contributes its top half.
    - Concatenate those three selected half-motifs from left to right to form the output.

    Key insight:
    - The hole must be repaired before selection because the chosen half is not always the intact one, and the panel index determines which mirrored half survives.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    pieces: list[np.ndarray] = []
    for panel_index, panel in enumerate(_split_panels(g)):
        repaired = _repair_panel(panel)
        top, bottom, _bbox = _crop_symmetric_halves(repaired)
        piece = top if panel_index % 2 == 0 else bottom
        pieces.append(piece)

    heights = {piece.shape[0] for piece in pieces}
    assert len(heights) == 1, "all selected half-motifs must share the same height"
    return np.concatenate(pieces, axis=1)


def _pick_hole_candidate(candidates: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    weighted = sorted(candidates, key=lambda cand: cand[2] * cand[3], reverse=True)
    pool = weighted[: max(1, min(6, len(weighted)))]
    return random.choice(pool)


def _absolute_hole_bbox(top_r: int, left_c: int, candidate: tuple[int, int, int, int], side: str) -> tuple[str, int, int, int, int]:
    r0, c0, hole_h, hole_w = candidate
    if side == "top":
        abs_r0 = top_r + r0
        abs_r1 = abs_r0 + hole_h - 1
    else:
        top_abs_r0 = top_r + r0
        top_abs_r1 = top_abs_r0 + hole_h - 1
        abs_r0 = 2 * AXIS_ROW - top_abs_r1
        abs_r1 = 2 * AXIS_ROW - top_abs_r0
    abs_c0 = left_c + c0
    abs_c1 = abs_c0 + hole_w - 1
    return side, abs_r0, abs_c0, abs_r1, abs_c1


def _render_panel(
    color: int,
    mask: np.ndarray,
    top_r: int,
    left_c: int,
    hole_side: str,
    hole_candidate: tuple[int, int, int, int],
) -> np.ndarray:
    panel = np.full((PANEL_SIZE, PANEL_SIZE), BACKGROUND, dtype=int)
    half_h, width = mask.shape

    for r in range(half_h):
        for c in range(width):
            if not mask[r, c]:
                continue
            abs_r = top_r + r
            abs_c = left_c + c
            panel[abs_r, abs_c] = color
            panel[2 * AXIS_ROW - abs_r, abs_c] = color

    hole_r0, hole_c0, hole_h, hole_w = hole_candidate
    for rr in range(hole_r0, hole_r0 + hole_h):
        for cc in range(hole_c0, hole_c0 + hole_w):
            abs_top_r = top_r + rr
            abs_c = left_c + cc
            if hole_side == "top":
                panel[abs_top_r, abs_c] = BACKGROUND
            else:
                panel[2 * AXIS_ROW - abs_top_r, abs_c] = BACKGROUND

    return panel


def _validate_semantics(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.shape == (PANEL_SIZE, 3 * PANEL_SIZE + 2)

    panels = _split_panels(g)
    pieces: list[np.ndarray] = []
    opposite_pieces: list[np.ndarray] = []
    hole_signatures: list[tuple[str, int, int, int, int]] = []
    saw_selected_hole = False
    saw_unselected_hole = False

    for panel_index, panel in enumerate(panels):
        assert _panel_foreground_color(panel) == PANEL_COLORS[panel_index]

        repaired = _repair_panel(panel)
        top, bottom, bbox = _crop_symmetric_halves(repaired)
        assert not np.array_equal(top, bottom), "a solver must distinguish orientation, not only content set"

        hole_side, hole_bbox = _hole_info(panel, repaired)
        hole_signatures.append((hole_side, *hole_bbox))

        selected_side = _SELECTED_SIDES[panel_index]
        selected_piece = top if selected_side == "top" else bottom
        opposite_piece = bottom if selected_side == "top" else top
        pieces.append(selected_piece)
        opposite_pieces.append(opposite_piece)

        raw_selected = _piece_from_bbox(panel, bbox, selected_side)
        raw_opposite = _piece_from_bbox(panel, bbox, "bottom" if selected_side == "top" else "top")

        if hole_side == selected_side:
            saw_selected_hole = True
            assert not np.array_equal(selected_piece, raw_selected), "repair must affect at least one chosen half"
            assert not np.array_equal(selected_piece, raw_opposite), "cannot replace the chosen half by the intact opposite half"
        else:
            saw_unselected_hole = True

    assert saw_selected_hole and saw_unselected_hole, "samples must cover both selected-half and unselected-half holes"
    assert len(set(hole_signatures)) == 3, "the three hole positions must not repeat"

    out = main(g)
    expected = np.concatenate(pieces, axis=1)
    assert np.array_equal(out, expected)
    assert not np.array_equal(out, np.concatenate(opposite_pieces, axis=1)), "panel parity must matter"


def _generate_one() -> np.ndarray:
    half_h = random.choice(HALF_HEIGHT_CHOICES)
    top_r = AXIS_ROW - half_h
    hole_sides = random.choice(_HOLE_SIDE_PATTERNS)

    panels: list[np.ndarray] = []
    used_holes: set[tuple[str, int, int, int, int]] = set()
    for color, hole_side in zip(PANEL_COLORS, hole_sides):
        for _ in range(120):
            width = random.randint(max(PATCH_WIDTH_RANGE[0], half_h + 2), PATCH_WIDTH_RANGE[1])
            mask = _generate_top_mask(half_h, width)
            hole_candidate = _pick_hole_candidate(_hole_candidates(mask))
            left_c = random.randint(0, PANEL_SIZE - width)
            hole_sig = _absolute_hole_bbox(top_r, left_c, hole_candidate, hole_side)
            if hole_sig in used_holes:
                continue

            panel = _render_panel(color, mask, top_r, left_c, hole_side, hole_candidate)
            repaired = _repair_panel(panel)
            top, bottom, _bbox = _crop_symmetric_halves(repaired)
            if top.shape[0] != half_h or bottom.shape[0] != half_h:
                continue
            used_holes.add(hole_sig)
            panels.append(panel)
            break
        else:
            raise AssertionError("failed to build panel")

    grid = np.full((PANEL_SIZE, 3 * PANEL_SIZE + 2), BACKGROUND, dtype=int)
    grid[:, PANEL_SIZE] = SEPARATOR
    grid[:, 2 * PANEL_SIZE + 1] = SEPARATOR
    for panel_index, panel in enumerate(panels):
        c0 = panel_index * PANEL_SIZE + panel_index
        grid[:, c0 : c0 + PANEL_SIZE] = panel

    _validate_semantics(grid)

    inp_sig = _sig(grid)
    out_sig = _sig(main(grid))
    assert inp_sig not in _SEEN_INPUT_SIGS, "duplicate input sample"
    assert out_sig not in _SEEN_OUTPUT_SIGS, "duplicate output sample"
    _SEEN_INPUT_SIGS.add(inp_sig)
    _SEEN_OUTPUT_SIGS.add(out_sig)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        try:
            return _generate_one()
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid panel-symmetry ARC input")
