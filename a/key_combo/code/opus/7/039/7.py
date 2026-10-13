import random

import numpy as np


BG = 0
SPRITE = 2
MARKER = 4
FRAME = 5

_GEN_PANEL_COUNTS = (3, 4)
_GEN_INTERIOR_WIDTHS = (3, 4)
_GEN_STEPS = (1, 2, 3)
_GEN_EXTRA_TAIL = (0, 1, 2)
_GEN_MAX_ATTEMPTS = 400

_GEN_SPRITE_MASKS = (
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _full_color_columns(grid: np.ndarray) -> dict[int, list[int]]:
    by_color: dict[int, list[int]] = {}
    for c in range(grid.shape[1]):
        values = np.unique(grid[:, c])
        if values.size == 1:
            color = int(values[0])
            by_color.setdefault(color, []).append(int(c))
    return by_color


def _frame_layout(grid: np.ndarray) -> tuple[int, list[int], int]:
    full_cols = _full_color_columns(grid)
    candidates = [
        (color, cols)
        for color, cols in full_cols.items()
        if len(cols) >= 3 and cols[0] == 0
    ]
    assert candidates, "expected shared frame separators"

    frame_color, starts = max(candidates, key=lambda item: len(item[1]))
    diffs = [b - a for a, b in zip(starts, starts[1:])]
    assert diffs and len(set(diffs)) == 1, "frame separators must be equally spaced"
    panel_w = diffs[0] + 1
    return int(frame_color), starts, int(panel_w)


def _infer_colors(grid: np.ndarray, frame_color: int) -> tuple[int, int, int]:
    values, counts = np.unique(grid, return_counts=True)
    freq = {int(v): int(c) for v, c in zip(values, counts)}
    bg = max(freq, key=freq.get)
    others = [color for color in freq if color not in (bg, frame_color)]
    assert len(others) == 2, "expected one sprite color and one marker color"
    marker_color = next(color for color in others if freq[color] == 1)
    sprite_color = next(color for color in others if color != marker_color)
    return int(bg), int(sprite_color), int(marker_color)


def _extract_sprite(panel: np.ndarray, sprite_color: int, bg: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    mask = panel == sprite_color
    box = _bbox(mask)
    assert box is not None, "each panel must contain a sprite"
    r0, c0, r1, c1 = box
    sprite = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    local = np.argwhere(mask[r0 : r1 + 1, c0 : c1 + 1])
    sprite[local[:, 0], local[:, 1]] = sprite_color
    return sprite, sprite == sprite_color, int(r0), int(c0)


def _make_output(height: int, panel_w: int, sprite: np.ndarray, top: int, left: int) -> np.ndarray:
    out = np.full((height, panel_w), BG, dtype=int)
    out[0, :] = FRAME
    out[-1, :] = FRAME
    out[:, 0] = FRAME
    out[:, -1] = FRAME
    mask = sprite == SPRITE
    target = out[top : top + sprite.shape[0], left : left + sprite.shape[1]]
    target[mask] = SPRITE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Decode the sprite's downward speed from the last frame's sightline marker and extrapolate one more frame.

    Input structure:
    - The grid contains a horizontal strip of equal-width frames that share vertical border columns.
    - Every frame contains the same red sprite at the same local columns, while its row increases by a constant step.
    - A single yellow marker sits to the right of the last frame, aligned with the sprite through a background-only sightline.

    Rule:
    - Read the horizontal background gap between the marker and the rightmost aligned sprite pixel in the last frame; that gap is the step size d.
    - Verify that consecutive frame-to-frame sprite row offsets are all equal to d.
    - Output one standalone framed panel of the same size, with the same sprite columns and the sprite moved downward by another d rows.

    Key insight:
    - The motion step is redundantly encoded both by the frame sequence and by the marker-to-sprite sightline gap, so the marker gives the speed needed for extrapolation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frame_color, starts, panel_w = _frame_layout(g)
    bg, sprite_color, marker_color = _infer_colors(g, frame_color)
    assert sprite_color == SPRITE and marker_color == MARKER

    height, width = g.shape
    frame_right = starts[-1] + panel_w - 1
    assert frame_right < width, "last frame must fit in the grid"

    panels = [g[:, start : start + panel_w] for start in starts]
    sprite, sprite_mask, first_top, local_left = _extract_sprite(panels[0], sprite_color, bg)
    top_rows = [first_top]

    for panel in panels[1:]:
        cur_sprite, cur_mask, top, left = _extract_sprite(panel, sprite_color, bg)
        assert left == local_left, "sprite columns must stay fixed across frames"
        assert np.array_equal(cur_mask, sprite_mask), "sprite shape must stay identical"
        assert np.array_equal(cur_sprite, sprite), "sprite pattern must stay identical"
        top_rows.append(int(top))

    marker_coords = np.argwhere(g == marker_color)
    assert marker_coords.shape == (1, 2), "expected a single marker"
    marker_r, marker_c = map(int, marker_coords[0])
    assert marker_c > frame_right, "marker must be to the right of the last frame"

    last_mask = panels[-1] == sprite_color
    same_row_cols = [int(c) for r, c in np.argwhere(last_mask) if int(r) == marker_r]
    assert same_row_cols, "marker must align with a sprite row in the last frame"
    rightmost_local_c = max(same_row_cols)
    rightmost_global_c = starts[-1] + rightmost_local_c
    step = marker_c - rightmost_global_c - 1
    assert 1 <= step <= 30, "marker gap must be positive"
    assert np.all(g[marker_r, rightmost_global_c + 1 : marker_c] == bg), "sightline gap must be background only"

    deltas = [b - a for a, b in zip(top_rows, top_rows[1:])]
    assert deltas and len(set(deltas)) == 1, "motion must be constant"
    assert deltas[0] == step, "marker gap must match frame-to-frame motion"

    next_top = top_rows[-1] + step
    assert next_top + sprite.shape[0] <= height - 1, "extrapolated sprite must fit inside the frame"
    return _make_output(height, panel_w, sprite, next_top, local_left)


def _random_sprite() -> tuple[np.ndarray, int]:
    mask = random.choice(_GEN_SPRITE_MASKS)
    rows = [r for r in range(mask.shape[0]) if mask[r, -1]]
    align_row = random.choice(rows)
    sprite = np.full(mask.shape, BG, dtype=int)
    sprite[mask] = SPRITE
    return sprite, int(align_row)


def _build_input(
    *,
    height: int,
    num_panels: int,
    interior_w: int,
    step: int,
    start_top: int,
    sprite: np.ndarray,
    align_row: int,
    extra_tail: int,
) -> np.ndarray:
    panel_w = interior_w + 2
    stride = panel_w - 1
    frame_region_w = num_panels * stride + 1
    total_w = frame_region_w + step + extra_tail
    grid = np.full((height, total_w), BG, dtype=int)

    starts = [i * stride for i in range(num_panels)]
    for start in starts:
        end = start + panel_w - 1
        grid[0, start : end + 1] = FRAME
        grid[-1, start : end + 1] = FRAME
        grid[:, start] = FRAME
    grid[:, frame_region_w - 1] = FRAME

    sprite_h, sprite_w = sprite.shape
    local_left = panel_w - sprite_w - 1
    for i, start in enumerate(starts):
        top = start_top + i * step
        view = grid[top : top + sprite_h, start + local_left : start + local_left + sprite_w]
        view[sprite == SPRITE] = SPRITE

    last_top = start_top + (num_panels - 1) * step
    hole_row = last_top + align_row
    grid[hole_row, frame_region_w - 1] = BG
    marker_c = frame_region_w - 1 + step
    grid[hole_row, marker_c] = MARKER
    return grid


def _attempt() -> np.ndarray | None:
    num_panels = random.choice(_GEN_PANEL_COUNTS)
    interior_w = random.choice(_GEN_INTERIOR_WIDTHS)
    step = random.choice(_GEN_STEPS)
    sprite, align_row = _random_sprite()

    sprite_h = sprite.shape[0]
    min_height = sprite_h + num_panels * step + 2
    height = random.randint(min_height, min(min_height + 4, 16))

    max_top = height - sprite_h - 1
    if max_top < 1 + num_panels * step:
        return None
    start_top = random.randint(1, max_top - num_panels * step)
    extra_tail = random.choice(_GEN_EXTRA_TAIL)

    inp = _build_input(
        height=height,
        num_panels=num_panels,
        interior_w=interior_w,
        step=step,
        start_top=start_top,
        sprite=sprite,
        align_row=align_row,
        extra_tail=extra_tail,
    )

    panel_w = interior_w + 2
    local_left = panel_w - sprite.shape[1] - 1
    expected = _make_output(height, panel_w, sprite, start_top + num_panels * step, local_left)
    if np.array_equal(main(inp), expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 7")
