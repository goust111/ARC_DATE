import math
import random
from collections import Counter
from fractions import Fraction

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    GRAY = 5
    PINK = 6
    ORANGE = 7
    TEAL = 8
    MAROON = 9
    NOT_BLACK = [BLUE, RED, GREEN, YELLOW, GREY, PINK, ORANGE, TEAL, MAROON]


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def paste(
    dst: np.ndarray,
    patch: np.ndarray,
    r0: int,
    c0: int,
    *,
    transparent: int | None = None,
) -> np.ndarray:
    h, w = patch.shape
    rr = slice(int(r0), int(r0) + h)
    cc = slice(int(c0), int(c0) + w)
    view = dst[rr, cc]
    if transparent is None:
        view[:, :] = patch
    else:
        mask = patch != int(transparent)
        view[mask] = patch[mask]
    return dst


def _unique_most_common(values) -> int:
    counts = Counter(map(int, values))
    ranked = counts.most_common()
    assert ranked
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1]
    return int(ranked[0][0])


def _round_half_up(value: Fraction) -> int:
    return math.floor(value + Fraction(1, 2))


def _split_panels(grid: np.ndarray) -> tuple[list[np.ndarray], int, int, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    frame = int(g[0, 0])
    assert h >= 5 and w >= 9
    assert np.all(g[0, :] == frame)
    assert np.all(g[-1, :] == frame)

    separators = [c for c in range(w) if np.all(g[:, c] == frame)]
    assert len(separators) >= 4
    assert separators[0] == 0 and separators[-1] == w - 1
    strides = [b - a for a, b in zip(separators, separators[1:])]
    assert len(set(strides)) == 1
    stride = strides[0]
    panel_w = stride + 1

    panels = []
    interior_values = []
    for c0, c1 in zip(separators, separators[1:]):
        panel = g[:, c0 : c1 + 1]
        assert panel.shape == (h, panel_w)
        assert np.all(panel[:, 0] == frame)
        assert np.all(panel[:, -1] == frame)
        panels.append(panel)
        interior_values.extend(panel[1:-1, 1:-1].ravel())
    fill = _unique_most_common(interior_values)
    return panels, frame, fill, stride


def _extract_panel_parts(
    panel: np.ndarray, fill: int
) -> tuple[np.ndarray, np.ndarray, int, int, np.ndarray, np.ndarray, int, int]:
    interior = panel[1:-1, 1:-1]

    bar_mask_local = interior == Color.GREY
    bar_box = bbox(bar_mask_local)
    assert bar_box is not None
    br0, bc0, br1, bc1 = bar_box
    bar_top = br0 + 1
    bar_left = bc0 + 1
    bar_patch = panel[bar_top : br1 + 2, bar_left : bc1 + 2].copy()
    bar_mask = bar_patch == Color.GREY
    assert np.all(bar_patch[bar_mask] == Color.GREY)

    sprite_mask_local = (interior != fill) & (interior != Color.GREY)
    sprite_box = bbox(sprite_mask_local)
    assert sprite_box is not None
    sr0, sc0, sr1, sc1 = sprite_box
    sprite_top = sr0 + 1
    sprite_left = sc0 + 1
    sprite_patch = panel[sprite_top : sr1 + 2, sprite_left : sc1 + 2].copy()
    sprite_mask = sprite_patch != fill
    assert np.any(sprite_mask)
    return sprite_patch, sprite_mask, sprite_top, sprite_left, bar_patch, bar_mask, bar_top, bar_left


def _overlap(
    r0_a: int,
    c0_a: int,
    mask_a: np.ndarray,
    r0_b: int,
    c0_b: int,
    mask_b: np.ndarray,
) -> bool:
    ah, aw = mask_a.shape
    bh, bw = mask_b.shape
    rr0 = max(r0_a, r0_b)
    cc0 = max(c0_a, c0_b)
    rr1 = min(r0_a + ah, r0_b + bh)
    cc1 = min(c0_a + aw, c0_b + bw)
    if rr0 >= rr1 or cc0 >= cc1:
        return False
    a = mask_a[rr0 - r0_a : rr1 - r0_a, cc0 - c0_a : cc1 - c0_a]
    b = mask_b[rr0 - r0_b : rr1 - r0_b, cc0 - c0_b : cc1 - c0_b]
    return bool((a & b).any())


def _build_empty_panel(height: int, panel_w: int, frame: int, fill: int) -> np.ndarray:
    out = np.full((height, panel_w), int(fill), dtype=int)
    out[0, :] = frame
    out[-1, :] = frame
    out[:, 0] = frame
    out[:, -1] = frame
    return out


def _simulate_output(
    height: int,
    panel_w: int,
    frame: int,
    fill: int,
    sprite_patch: np.ndarray,
    sprite_mask: np.ndarray,
    sprite_top: int,
    sprite_left: int,
    bar_patch: np.ndarray,
    bar_mask: np.ndarray,
    bar_top: int,
    bar_left: int,
    signed_steps: int,
) -> tuple[np.ndarray, int, int]:
    cur_top = int(sprite_top)
    cur_bar_top = int(bar_top)
    if signed_steps == 0:
        out = _build_empty_panel(height, panel_w, frame, fill)
        paste(out, bar_patch, cur_bar_top, bar_left, transparent=fill)
        paste(out, sprite_patch, cur_top, sprite_left, transparent=fill)
        return out, cur_top, cur_bar_top

    direction = 1 if signed_steps > 0 else -1
    remaining = 0
    for step_idx in range(abs(int(signed_steps))):
        next_top = cur_top + direction
        if _overlap(next_top, sprite_left, sprite_mask, cur_bar_top, bar_left, bar_mask):
            remaining = abs(int(signed_steps)) - step_idx
            break
        cur_top = next_top

    if remaining:
        if direction > 0:
            max_shift = (height - 2) - (cur_bar_top + bar_mask.shape[0] - 1)
        else:
            max_shift = cur_bar_top - 1
        cur_bar_top += direction * min(remaining, max_shift)

    out = _build_empty_panel(height, panel_w, frame, fill)
    paste(out, bar_patch, cur_bar_top, bar_left, transparent=fill)
    paste(out, sprite_patch, cur_top, sprite_left, transparent=fill)
    return out, cur_top, cur_bar_top


def main(grid: np.ndarray) -> np.ndarray:
    """Extrapolate the moving sprite into the next standalone panel, transferring blocked motion to the grey bar.

    Input structure:
    - The grid is a horizontal sequence of equal-width framed panels that share vertical border columns.
    - Every panel has the same fill color, the same grey horizontal obstacle bar, and the same inner sprite.
    - The sprite keeps the same local columns and its top row changes by a constant amount from panel to panel.

    Rule:
    - Parse the shared-border frame sequence and infer the per-panel vertical displacement from successive sprite positions.
    - Convert that displacement to the standalone next frame by scaling from shared-border stride to full panel width.
    - Starting from the last observed panel, move the sprite one cell at a time in that vertical direction; if the next step would overlap the grey bar, stop the sprite.
    - Transfer every unspent step to the grey bar, moving the bar in the same direction until the requested steps are exhausted or the bar hits the inner frame border.

    Key insight:
    - This is not pure extrapolation: the last frame first predicts a naive next sprite position, but any blocked remainder becomes a second motion applied to the obstacle itself.
    """
    panels, frame, fill, stride = _split_panels(grid)
    height, panel_w = panels[0].shape

    first = _extract_panel_parts(panels[0], fill)
    sprite_patch, sprite_mask, first_top, sprite_left, bar_patch, bar_mask, bar_top, bar_left = first

    top_rows = [first_top]
    for panel in panels[1:]:
        cur = _extract_panel_parts(panel, fill)
        cur_sprite_patch, cur_sprite_mask, cur_top, cur_left, cur_bar_patch, cur_bar_mask, cur_bar_top, cur_bar_left = cur
        assert cur_left == sprite_left
        assert np.array_equal(cur_sprite_mask, sprite_mask)
        assert np.array_equal(cur_sprite_patch[cur_sprite_mask], sprite_patch[cur_sprite_mask])
        assert cur_bar_top == bar_top and cur_bar_left == bar_left
        assert np.array_equal(cur_bar_mask, bar_mask)
        top_rows.append(cur_top)

    deltas = [b - a for a, b in zip(top_rows, top_rows[1:])]
    assert deltas and len(set(deltas)) == 1
    total_steps = _round_half_up(Fraction(deltas[0] * panel_w, stride))
    assert total_steps != 0

    out, _sprite_top, _bar_top = _simulate_output(
        height,
        panel_w,
        frame,
        fill,
        sprite_patch,
        sprite_mask,
        top_rows[-1],
        sprite_left,
        bar_patch,
        bar_mask,
        bar_top,
        bar_left,
        total_steps,
    )
    return out


_GEN_CALL_COUNT = 0
_GEN_COLOR_POOL = [Color.BLACK, Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
_GEN_SPRITES = [
    np.array([[1, 1], [1, 0]], dtype=int),
    np.array([[1, 1], [0, 1]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[1, 0], [1, 1], [0, 1]], dtype=int),
]


def _sprite_patch_for_color(mask: np.ndarray, fill: int, color: int) -> np.ndarray:
    patch = np.full(mask.shape, int(fill), dtype=int)
    patch[mask.astype(bool)] = int(color)
    return patch


def _make_input_from_spec(
    height: int,
    panel_w: int,
    num_panels: int,
    frame: int,
    fill: int,
    sprite_patch: np.ndarray,
    top_rows: list[int],
    sprite_left: int,
    bar_patch: np.ndarray,
    bar_top: int,
    bar_left: int,
) -> np.ndarray:
    stride = panel_w - 1
    width = num_panels * stride + 1
    grid = np.full((height, width), int(fill), dtype=int)
    grid[0, :] = frame
    grid[-1, :] = frame
    for i in range(num_panels + 1):
        grid[:, i * stride] = frame
    for i, top in enumerate(top_rows):
        offset = i * stride
        paste(grid, bar_patch, bar_top, offset + bar_left, transparent=fill)
        paste(grid, sprite_patch, top, offset + sprite_left, transparent=fill)
    return grid


def _make_case(want_collision: bool, direction: int) -> tuple[np.ndarray, np.ndarray]:
    direction = 1 if direction > 0 else -1
    panel_w = random.choice([5, 6])
    stride = panel_w - 1

    shape_options = [mask for mask in _GEN_SPRITES if mask.shape[1] <= panel_w - 2]
    sprite_mask = random.choice(shape_options).astype(bool)
    sprite_h, sprite_w = sprite_mask.shape

    feasible_panel_counts = []
    for num_panels in [3, 4]:
        base_height = 2 + sprite_h + panel_w + (num_panels - 1) * stride
        if base_height + 2 <= 30:
            feasible_panel_counts.append((num_panels, base_height))
    assert feasible_panel_counts
    num_panels, base_height = random.choice(feasible_panel_counts)
    height = random.randint(base_height + 2, min(30, base_height + 5))

    colors = random.sample(_GEN_COLOR_POOL, 3)
    fill, frame, sprite_color = colors
    bar_patch = np.full((1, random.randint(max(2, sprite_w), panel_w - 2)), Color.GREY, dtype=int)
    sprite_patch = _sprite_patch_for_color(sprite_mask, fill, sprite_color)

    local_left_choices = list(range(1, panel_w - sprite_w))
    sprite_left = random.choice(local_left_choices)

    bar_left_choices = []
    for left in range(1, panel_w - bar_patch.shape[1]):
        if not (left + bar_patch.shape[1] - 1 < sprite_left or sprite_left + sprite_w - 1 < left):
            bar_left_choices.append(left)
    assert bar_left_choices
    bar_left = random.choice(bar_left_choices)

    if direction > 0:
        last_top_min = 1 + (num_panels - 1) * stride
        last_top_max = height - 1 - sprite_h - panel_w
    else:
        last_top_min = 1 + panel_w
        last_top_max = height - 1 - sprite_h - (num_panels - 1) * stride
    assert last_top_min <= last_top_max
    last_top = random.randint(last_top_min, last_top_max)

    top_rows = [last_top - (num_panels - 1 - i) * direction * stride for i in range(num_panels)]
    total_steps = direction * panel_w
    target_top = last_top + total_steps

    if want_collision:
        success_steps = random.randint(0, panel_w - 1)
        final_top = last_top + direction * success_steps
        remaining = panel_w - success_steps
        if direction > 0:
            bar_top = final_top + sprite_h
            pushed_top = min(bar_top + remaining, height - 2)
        else:
            bar_top = final_top - 1
            pushed_top = max(bar_top - remaining, 1)
        assert 1 <= bar_top <= height - 2
        assert final_top != target_top
    else:
        if direction > 0:
            candidates = list(range(target_top + sprite_h, height - 1))
        else:
            candidates = list(range(1, target_top))
        assert candidates
        bar_top = random.choice(candidates)
        pushed_top = bar_top
        final_top = target_top
        remaining = 0

    inp = _make_input_from_spec(
        height,
        panel_w,
        num_panels,
        frame,
        fill,
        sprite_patch,
        top_rows,
        sprite_left,
        bar_patch,
        bar_top,
        bar_left,
    )
    expected, solved_top, solved_bar_top = _simulate_output(
        height,
        panel_w,
        frame,
        fill,
        sprite_patch,
        sprite_mask,
        last_top,
        sprite_left,
        bar_patch,
        bar_patch == Color.GREY,
        bar_top,
        bar_left,
        total_steps,
    )

    assert solved_top == final_top
    assert solved_bar_top == pushed_top
    if want_collision:
        assert solved_top != target_top
        assert remaining > 0
    else:
        assert solved_top == target_top
        assert solved_bar_top == bar_top
        assert remaining == 0

    actual = main(inp)
    assert np.array_equal(actual, expected)
    return inp, expected


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    branch = _GEN_CALL_COUNT % 4
    _GEN_CALL_COUNT += 1

    want_collision = branch in (0, 2)
    direction = 1 if branch in (0, 1) else -1

    for _ in range(200):
        try:
            inp, _expected = _make_case(want_collision, direction)
            return inp
        except AssertionError:
            continue
    raise RuntimeError("failed to generate valid collision-transfer extrapolation case")
