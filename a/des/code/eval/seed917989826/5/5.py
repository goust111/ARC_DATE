from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _full_height_groups(grid: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    height, width = grid.shape
    groups: list[tuple[int, int, int]] = []
    col = 0
    while col < width:
        color = int(grid[0, col])
        if color != bg and np.all(grid[:, col] == color):
            start = col
            while (
                col + 1 < width
                and int(grid[0, col + 1]) == color
                and np.all(grid[:, col + 1] == color)
            ):
                col += 1
            groups.append((start, col, color))
        col += 1
    return groups


def _split_panels(grid: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups = _full_height_groups(grid, bg)
    width = int(grid.shape[1])
    candidates: list[tuple[int, int, int, int]] = []
    for idx1, (s1, e1, color1) in enumerate(groups):
        for s2, e2, color2 in groups[idx1 + 1 :]:
            if color1 != color2:
                continue
            panel_w1 = s1
            panel_w2 = s2 - e1 - 1
            panel_w3 = width - e2 - 1
            if panel_w1 > 0 and panel_w1 == panel_w2 == panel_w3:
                candidates.append((s1, e1, s2, e2))
    assert len(candidates) == 1, "expected exactly two separator bands splitting three equal panels"
    s1, e1, s2, e2 = candidates[0]
    left = grid[:, :s1].copy()
    middle = grid[:, e1 + 1 : s2].copy()
    right = grid[:, e2 + 1 :].copy()
    return left, middle, right


def _find_anchor(panel: np.ndarray, bg: int) -> tuple[int, int]:
    colors = Counter(map(int, panel[panel != bg].reshape(-1)))
    singleton_colors = [color for color, count in colors.items() if count == 1]
    assert len(singleton_colors) == 1, "expected exactly one singleton anchor color in each panel"
    anchor_color = singleton_colors[0]
    coords = np.argwhere(panel == anchor_color)
    assert len(coords) == 1, "anchor color must occupy exactly one cell"
    return int(coords[0, 0]), int(coords[0, 1])


def _reflect_panel(
    panel: np.ndarray, axis_kind: str, anchor: tuple[int, int], bg: int
) -> np.ndarray:
    height, width = panel.shape
    anchor_r, anchor_c = anchor
    out = np.full_like(panel, bg)
    for r_raw, c_raw in np.argwhere(panel != bg):
        r = int(r_raw)
        c = int(c_raw)
        if axis_kind == "vertical":
            rr, cc = r, 2 * anchor_c - c
        else:
            rr, cc = 2 * anchor_r - r, c
        assert 0 <= rr < height and 0 <= cc < width, "reflected pixel must remain inside the panel"
        color = int(panel[r, c])
        assert int(out[rr, cc]) in {bg, color}, "reflection cannot create conflicting colors"
        out[rr, cc] = color
    return out


def _maybe_reflect_panel(
    panel: np.ndarray, axis_kind: str, anchor: tuple[int, int], bg: int
) -> np.ndarray | None:
    try:
        return _reflect_panel(panel, axis_kind, anchor, bg)
    except AssertionError:
        return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn whether the training pair uses a vertical-axis or horizontal-axis mirror,
    with the axis passing through the anchor pixel, then apply the same axis type
    to the test panel using its own anchor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _dominant_color(g)
    left, middle, right = _split_panels(g, bg)

    train_anchor = _find_anchor(left, bg)
    vertical = _maybe_reflect_panel(left, "vertical", train_anchor, bg)
    horizontal = _maybe_reflect_panel(left, "horizontal", train_anchor, bg)
    vertical_match = vertical is not None and np.array_equal(vertical, middle)
    horizontal_match = horizontal is not None and np.array_equal(horizontal, middle)
    assert vertical_match != horizontal_match, "training pair must determine a unique mirror-axis type"

    axis_kind = "vertical" if vertical_match else "horizontal"
    test_anchor = _find_anchor(right, bg)
    return _reflect_panel(right, axis_kind, test_anchor, bg)


# --------------- generator ---------------

_COLORS = tuple(range(10))
_PANEL_H_RANGE = (6, 10)
_PANEL_W_RANGE = (6, 10)
_SEPARATOR_WIDTHS = (1, 2)
_SHAPE_SIZE_RANGE = (3, 8)
_GEN_RETRIES = 400
_SHAPE_RETRIES = 120
_DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _neighbors4(r: int, c: int, height: int, width: int):
    for dr, dc in _DIRS4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < height and 0 <= cc < width:
            yield rr, cc


def _allowed_mask(
    height: int, width: int, axis_kind: str, anchor: tuple[int, int], side: str
) -> np.ndarray:
    anchor_r, anchor_c = anchor
    mask = np.zeros((height, width), dtype=bool)
    if axis_kind == "vertical":
        if side == "left":
            lo = max(0, 2 * anchor_c - (width - 1))
            hi = anchor_c
        else:
            lo = anchor_c
            hi = min(width - 1, 2 * anchor_c)
        mask[:, lo : hi + 1] = True
    else:
        if side == "up":
            lo = max(0, 2 * anchor_r - (height - 1))
            hi = anchor_r
        else:
            lo = anchor_r
            hi = min(height - 1, 2 * anchor_r)
        mask[lo : hi + 1, :] = True
    return mask


def _has_off_axis_cell(
    coords: set[tuple[int, int]], axis_kind: str, anchor: tuple[int, int]
) -> bool:
    anchor_r, anchor_c = anchor
    if axis_kind == "vertical":
        return any(c != anchor_c for _r, c in coords)
    return any(r != anchor_r for r, _c in coords)


def _grow_shape(
    height: int,
    width: int,
    axis_kind: str,
    anchor: tuple[int, int],
    side: str,
) -> set[tuple[int, int]] | None:
    allowed = _allowed_mask(height, width, axis_kind, anchor, side)
    allowed_count = int(allowed.sum())
    max_size = min(_SHAPE_SIZE_RANGE[1], allowed_count)
    if max_size < _SHAPE_SIZE_RANGE[0]:
        return None

    target_size = random.randint(_SHAPE_SIZE_RANGE[0], max_size)
    for _ in range(_SHAPE_RETRIES):
        selected = {anchor}
        frontier = {
            cell
            for cell in _neighbors4(anchor[0], anchor[1], height, width)
            if allowed[cell]
        }
        while len(selected) < target_size:
            if not frontier:
                break
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            selected.add(cell)
            for nxt in _neighbors4(cell[0], cell[1], height, width):
                if allowed[nxt] and nxt not in selected:
                    frontier.add(nxt)
        if len(selected) == target_size and _has_off_axis_cell(selected, axis_kind, anchor):
            return selected
    return None


def _render_panel(
    height: int,
    width: int,
    coords: set[tuple[int, int]],
    anchor: tuple[int, int],
    shape_color: int,
    anchor_color: int,
    bg: int,
) -> np.ndarray:
    panel = np.full((height, width), bg, dtype=int)
    for r, c in coords:
        panel[r, c] = shape_color
    panel[anchor] = anchor_color
    return panel


def _has_spurious_full_height_column(panel: np.ndarray, bg: int) -> bool:
    for col in range(panel.shape[1]):
        color = int(panel[0, col])
        if color != bg and np.all(panel[:, col] == color):
            return True
    return False


def _random_panel(
    height: int,
    width: int,
    axis_kind: str,
    shape_color: int,
    anchor_color: int,
    bg: int,
) -> np.ndarray | None:
    side_choices = ("left", "right") if axis_kind == "vertical" else ("up", "down")
    for _ in range(_SHAPE_RETRIES):
        if axis_kind == "vertical":
            anchor = (random.randrange(height), random.randint(1, width - 2))
        else:
            anchor = (random.randint(1, height - 2), random.randrange(width))
        side = random.choice(side_choices)
        coords = _grow_shape(height, width, axis_kind, anchor, side)
        if coords is None:
            continue
        panel = _render_panel(height, width, coords, anchor, shape_color, anchor_color, bg)
        if _has_spurious_full_height_column(panel, bg):
            continue
        return panel
    return None


def _assemble_input(
    left: np.ndarray, middle: np.ndarray, right: np.ndarray, separator: int, sep_w: int, bg: int
) -> np.ndarray:
    height, panel_w = left.shape
    total_w = 3 * panel_w + 2 * sep_w
    grid = np.full((height, total_w), bg, dtype=int)
    grid[:, :panel_w] = left
    grid[:, panel_w : panel_w + sep_w] = separator
    grid[:, panel_w + sep_w : 2 * panel_w + sep_w] = middle
    grid[:, 2 * panel_w + sep_w : 2 * panel_w + 2 * sep_w] = separator
    grid[:, 2 * panel_w + 2 * sep_w :] = right
    return grid


def _training_axis_is_unique(left: np.ndarray, middle: np.ndarray, bg: int) -> bool:
    anchor = _find_anchor(left, bg)
    vertical = _maybe_reflect_panel(left, "vertical", anchor, bg)
    horizontal = _maybe_reflect_panel(left, "horizontal", anchor, bg)
    return (vertical is not None and np.array_equal(vertical, middle)) != (
        horizontal is not None and np.array_equal(horizontal, middle)
    )


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*_PANEL_H_RANGE)
    width = random.randint(*_PANEL_W_RANGE)
    if height < 3 or width < 3:
        return None

    axis_kind = random.choice(("vertical", "horizontal"))
    sep_w = random.choice(_SEPARATOR_WIDTHS)
    bg, separator, anchor_color, left_color, right_color = random.sample(_COLORS, 5)

    left = _random_panel(height, width, axis_kind, left_color, anchor_color, bg)
    right = _random_panel(height, width, axis_kind, right_color, anchor_color, bg)
    if left is None or right is None:
        return None

    left_anchor = _find_anchor(left, bg)
    middle = _reflect_panel(left, axis_kind, left_anchor, bg)
    if np.array_equal(left, middle):
        return None
    if _has_spurious_full_height_column(middle, bg) or _has_spurious_full_height_column(right, bg):
        return None
    if not _training_axis_is_unique(left, middle, bg):
        return None

    grid = _assemble_input(left, middle, right, separator, sep_w, bg)
    expected = _reflect_panel(right, axis_kind, _find_anchor(right, bg), bg)

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-anchor ARC input")
