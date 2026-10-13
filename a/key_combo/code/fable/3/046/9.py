from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import bbox, colors_present, components_mask, get_bg, paste


def _row_shift(row: int, anchor_row: int, marker_side: int, gap: int) -> int:
    dist = abs(int(row) - int(anchor_row))
    if dist % 2 == 0:
        return 0
    moving_index = (dist + 1) // 2
    direction = int(marker_side) if moving_index % 2 == 1 else -int(marker_side)
    return direction * int(gap)


def _find_structure(grid: np.ndarray) -> dict:
    bg = get_bg(grid)
    non_bg = colors_present(grid, exclude={bg})
    assert len(non_bg) == 2, "expected one foreground color and one marker color"

    counts = {color: int(np.count_nonzero(grid == int(color))) for color in non_bg}
    marker_colors = [color for color, count in counts.items() if count == 1]
    assert len(marker_colors) == 1, "expected exactly one singleton marker pixel"
    marker_color = int(marker_colors[0])
    fg = int([color for color in non_bg if int(color) != marker_color][0])
    assert counts[fg] > 1, "foreground object must contain multiple pixels"

    marker_coords = np.argwhere(grid == marker_color)
    assert marker_coords.shape == (1, 2)
    marker_r, marker_c = map(int, marker_coords[0])

    fg_mask = grid == fg
    fg_components = components_mask(fg_mask, connectivity=4)
    assert len(fg_components) == 1, "foreground pixels must form one 4-connected object"

    box = bbox(fg_mask)
    assert box is not None
    r0, c0, r1, c1 = map(int, box)
    for r in range(r0, r1 + 1):
        assert bool(np.any(fg_mask[r, c0 : c1 + 1])), "each bbox row needs foreground"

    assert r0 <= marker_r <= r1, "marker row must lie inside the foreground bbox rows"
    if marker_c < c0:
        marker_side = -1
        gap = c0 - marker_c - 1
        between = grid[marker_r, marker_c + 1 : c0]
    elif marker_c > c1:
        marker_side = 1
        gap = marker_c - c1 - 1
        between = grid[marker_r, c1 + 1 : marker_c]
    else:
        raise AssertionError("marker must sit outside the foreground bbox horizontally")

    assert gap >= 1, "marker and bbox must be separated by at least one background column"
    assert bool(np.all(between == int(bg))), "marker-to-bbox gap must be all background"

    return {
        "bg": int(bg),
        "fg": int(fg),
        "marker_color": int(marker_color),
        "marker": (int(marker_r), int(marker_c)),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "side": int(marker_side),
        "gap": int(gap),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Bend the single foreground lattice into a marker-anchored horizontal row wave.

    Input structure:
    - The background is the most common color. There is exactly one foreground color
      forming one 4-connected lattice-like object, and every row of its bounding box
      contains foreground pixels.
    - A single marker pixel of a third color is outside the foreground bounding box
      on the left or right, with one or more background columns between marker and box.
    - The marker row lies within the foreground bounding-box row range.

    Rule:
    - Erase the marker pixel to background.
    - The marker row is the anchor row; rows at even distance from it keep their
      foreground cells in place.
    - Rows at odd distance shift horizontally by the marker-to-box gap. On each side
      of the anchor, the first, second, third, ... moving rows shift toward the marker
      side, away from it, toward it, ... respectively.

    Key insight:
    - The marker pixel jointly encodes the anchor row, first shift direction, and
      wave amplitude; the phase is not determined from the top or bottom of the object.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    info = _find_structure(g)
    bg = int(info["bg"])
    fg = int(info["fg"])
    marker_r, marker_c = info["marker"]
    r0, _c0, r1, _c1 = info["bbox"]
    side = int(info["side"])
    gap = int(info["gap"])

    fg_mask = g == fg
    out = g.copy()
    out[marker_r, marker_c] = bg

    for r in range(r0, r1 + 1):
        cols = np.flatnonzero(fg_mask[r])
        assert cols.size > 0, "each row inside the bbox must contain foreground"
        shift = _row_shift(r, marker_r, side, gap)
        if shift == 0:
            continue

        new_cols = cols + shift
        assert int(new_cols.min()) >= 0 and int(new_cols.max()) < g.shape[1]
        out[r, cols] = bg
        assert bool(np.all(out[r, new_cols] == bg)), "shifted row must land on background"
        out[r, new_cols] = fg

    return out


# --------------- generator ---------------

_GEN_MAX_ATTEMPTS = 500
_GEN_GRID_HEIGHT_RANGE = (10, 24)
_GEN_GRID_WIDTH_RANGE = (16, 30)
_GEN_OBJECT_HEIGHT_RANGE = (7, 13)
_GEN_OBJECT_WIDTH_RANGE = (6, 12)
_GEN_MAX_GAP = 4
_GEN_BRANCH_ANCHORS = ("interior", "bottom", "top", "near_top", "near_bottom", "interior")
_GEN_CALL_COUNT = 0


def _gen_lattice_patch(height: int, width: int, fg: int, bg: int) -> np.ndarray:
    patch = np.full((int(height), int(width)), int(bg), dtype=int)

    left_w = random.choice((1, 2)) if width >= 8 else 1
    right_w = random.choice((1, 2)) if width >= 8 else 1
    bar_cols = set(range(left_w))
    bar_cols.update(range(width - right_w, width))

    if width >= 9 and random.random() < 0.8:
        mid_w = random.choice((1, 2)) if width >= 11 else 1
        lo = left_w + 2
        hi = width - right_w - mid_w - 2
        if lo <= hi:
            mid = random.randint(lo, hi)
            bar_cols.update(range(mid, mid + mid_w))

    hbar_rows = {0, height - 1}
    spacing = random.choice((2, 3, 4))
    phase = random.randint(1, min(spacing, height - 2))
    hbar_rows.update(range(phase, height - 1, spacing))
    if len(hbar_rows) < 3:
        hbar_rows.add(random.randint(1, height - 2))

    patch[sorted(hbar_rows), :] = int(fg)
    patch[:, sorted(bar_cols)] = int(fg)
    return patch


def _gen_anchor_offset(height: int, mode: str) -> int:
    if mode == "top":
        return 0
    if mode == "bottom":
        return int(height) - 1
    if mode == "near_top":
        return 1
    if mode == "near_bottom":
        return int(height) - 2
    return random.randint(2, int(height) - 3)


def _gen_apply_known_wave(
    grid: np.ndarray,
    *,
    bg: int,
    fg: int,
    marker: tuple[int, int],
    box: tuple[int, int, int, int],
    side: int,
    gap: int,
) -> np.ndarray:
    marker_r, marker_c = marker
    r0, _c0, r1, _c1 = box
    fg_mask = grid == int(fg)
    out = grid.copy()
    out[marker_r, marker_c] = int(bg)

    for r in range(r0, r1 + 1):
        cols = np.flatnonzero(fg_mask[r])
        shift = _row_shift(r, marker_r, side, gap)
        if shift == 0:
            continue
        out[r, cols] = int(bg)
        out[r, cols + shift] = int(fg)
    return out


def _gen_marker_col(box: tuple[int, int, int, int], side: int, gap: int) -> int:
    _r0, c0, _r1, c1 = box
    if int(side) < 0:
        return int(c0 - gap - 1)
    return int(c1 + gap + 1)


def _gen_shift_set(box: tuple[int, int, int, int], marker_r: int, side: int, gap: int) -> set[int]:
    r0, _c0, r1, _c1 = box
    return {_row_shift(r, marker_r, side, gap) for r in range(r0, r1 + 1)}


def _gen_counterfactual_output(
    grid: np.ndarray,
    *,
    old_marker: tuple[int, int],
    new_marker: tuple[int, int],
    marker_color: int,
) -> np.ndarray | None:
    alt = grid.copy()
    old_r, old_c = old_marker
    new_r, new_c = new_marker
    bg = get_bg(grid)
    if not (0 <= new_r < alt.shape[0] and 0 <= new_c < alt.shape[1]):
        return None
    alt[old_r, old_c] = int(bg)
    if int(alt[new_r, new_c]) != int(bg):
        return None
    alt[new_r, new_c] = int(marker_color)
    try:
        return main(alt)
    except AssertionError:
        return None


def _gen_semantic_checks(grid: np.ndarray, expected: np.ndarray, spec: dict) -> bool:
    try:
        actual = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False

    bg = int(spec["bg"])
    fg = int(spec["fg"])
    marker = tuple(spec["marker"])
    marker_r, marker_c = marker
    box = tuple(spec["bbox"])
    r0, _c0, r1, _c1 = box
    side = int(spec["side"])
    gap = int(spec["gap"])
    marker_color = int(spec["marker_color"])

    erased = grid.copy()
    erased[marker_r, marker_c] = bg
    if np.array_equal(expected, erased):
        return False

    shifts = _gen_shift_set(box, marker_r, side, gap)
    if side * gap not in shifts or -side * gap not in shifts:
        return False

    old_mask = grid == fg
    new_mask = expected == fg
    for r in range(r0, r1 + 1):
        old_cols = np.flatnonzero(old_mask[r])
        shift = _row_shift(r, marker_r, side, gap)
        new_cols = np.flatnonzero(new_mask[r])
        if not np.array_equal(new_cols, old_cols + shift):
            return False

    c_side_flip = _gen_marker_col(box, -side, gap)
    flipped = _gen_counterfactual_output(
        grid,
        old_marker=marker,
        new_marker=(marker_r, c_side_flip),
        marker_color=marker_color,
    )
    if flipped is None or np.array_equal(flipped, expected):
        return False

    alt_anchor_r = r0 if marker_r != r0 else r1
    anchored = _gen_counterfactual_output(
        grid,
        old_marker=marker,
        new_marker=(alt_anchor_r, marker_c),
        marker_color=marker_color,
    )
    if anchored is None or np.array_equal(anchored, expected):
        return False

    alt_gap = 2 if gap != 2 else 3
    c_gap = _gen_marker_col(box, side, alt_gap)
    gapped = _gen_counterfactual_output(
        grid,
        old_marker=marker,
        new_marker=(marker_r, c_gap),
        marker_color=marker_color,
    )
    if gapped is None or np.array_equal(gapped, expected):
        return False

    return True


def _gen_attempt(branch: int) -> np.ndarray | None:
    gap = (branch % _GEN_MAX_GAP) + 1
    side = -1 if branch % 2 == 0 else 1
    anchor_mode = _GEN_BRANCH_ANCHORS[branch % len(_GEN_BRANCH_ANCHORS)]

    height = random.randint(*_GEN_GRID_HEIGHT_RANGE)
    width = random.randint(*_GEN_GRID_WIDTH_RANGE)
    obj_h = random.randint(*_GEN_OBJECT_HEIGHT_RANGE)
    obj_w = random.randint(*_GEN_OBJECT_WIDTH_RANGE)

    side_margin = _GEN_MAX_GAP + 1
    if obj_h + 2 > height or obj_w + 2 * side_margin > width:
        return None

    colors = list(range(10))
    bg = random.choice(colors)
    colors.remove(bg)
    fg, marker_color = random.sample(colors, 2)

    patch = _gen_lattice_patch(obj_h, obj_w, fg, bg)
    grid = np.full((height, width), int(bg), dtype=int)

    r0 = random.randint(1, height - obj_h - 1)
    c0 = random.randint(side_margin, width - obj_w - side_margin)
    paste(grid, patch, r0, c0, transparent=bg)

    box = (r0, c0, r0 + obj_h - 1, c0 + obj_w - 1)
    marker_r = r0 + _gen_anchor_offset(obj_h, anchor_mode)
    marker_c = _gen_marker_col(box, side, gap)
    if not (0 <= marker_c < width):
        return None
    if int(grid[marker_r, marker_c]) != int(bg):
        return None
    grid[marker_r, marker_c] = int(marker_color)

    values, counts = np.unique(grid, return_counts=True)
    if int(values[int(np.argmax(counts))]) != int(bg):
        return None

    spec = {
        "bg": int(bg),
        "fg": int(fg),
        "marker_color": int(marker_color),
        "marker": (int(marker_r), int(marker_c)),
        "bbox": tuple(map(int, box)),
        "side": int(side),
        "gap": int(gap),
    }
    expected = _gen_apply_known_wave(
        grid,
        bg=bg,
        fg=fg,
        marker=(marker_r, marker_c),
        box=box,
        side=side,
        gap=gap,
    )
    if not _gen_semantic_checks(grid, expected, spec):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    start_branch = _GEN_CALL_COUNT
    for attempt in range(_GEN_MAX_ATTEMPTS):
        branch = start_branch + attempt
        grid = _gen_attempt(branch)
        if grid is not None:
            _GEN_CALL_COUNT = branch + 1
            return grid
    raise RuntimeError("failed to generate valid marker-anchored wave input")
