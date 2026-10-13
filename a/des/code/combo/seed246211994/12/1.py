from __future__ import annotations

from collections import Counter, deque
from fractions import Fraction
import itertools
import math

import numpy as np


def _round_half_up(value: Fraction) -> int:
    return math.floor(value + Fraction(1, 2))


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _most_common(values) -> int:
    return int(Counter(map(int, values)).most_common(1)[0][0])


def _extract_sprite(panel: np.ndarray, fill: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    interior_mask = panel[1:-1, 1:-1] != fill
    box = _bbox(interior_mask)
    assert box is not None
    r0, c0, r1, c1 = box
    r0 += 1
    r1 += 1
    c0 += 1
    c1 += 1
    sprite = panel[r0 : r1 + 1, c0 : c1 + 1].copy()
    sprite_mask = sprite != fill
    assert int(np.count_nonzero(sprite_mask)) == int(np.count_nonzero(interior_mask))
    return sprite, sprite_mask, int(r0), int(c0)


def _paste(panel: np.ndarray, sprite: np.ndarray, top: int, left: int, fill: int) -> None:
    view = panel[top : top + sprite.shape[0], left : left + sprite.shape[1]]
    mask = sprite != fill
    view[mask] = sprite[mask]


def _solve_horizontal(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 4 and w >= 7

    frame = int(g[0, 0])
    assert bool(np.all(g[0, :] == frame))
    assert bool(np.all(g[-1, :] == frame))

    separators = [c for c in range(w) if bool(np.all(g[:, c] == frame))]
    assert len(separators) >= 4
    assert separators[0] == 0 and separators[-1] == w - 1
    strides = [int(b - a) for a, b in zip(separators, separators[1:])]
    assert len(set(strides)) == 1
    stride = int(strides[0])
    assert stride >= 3
    panel_w = int(stride + 1)

    panels: list[np.ndarray] = []
    interior_values: list[int] = []
    for c0, c1 in zip(separators, separators[1:]):
        panel = g[:, c0 : c1 + 1]
        assert panel.shape == (h, panel_w)
        assert bool(np.all(panel[0, :] == frame))
        assert bool(np.all(panel[-1, :] == frame))
        assert bool(np.all(panel[:, 0] == frame))
        assert bool(np.all(panel[:, -1] == frame))
        panels.append(panel)
        interior_values.extend(map(int, panel[1:-1, 1:-1].ravel()))

    non_frame_values = [v for v in interior_values if int(v) != frame]
    assert non_frame_values
    fill = _most_common(non_frame_values)

    blank = [bool(np.all(panel[1:-1, 1:-1] == fill)) for panel in panels]
    assert blank[-1]
    assert not any(blank[:-1])
    visible = panels[:-1]
    assert len(visible) >= 2

    template_sprite, template_mask, first_top, template_left = _extract_sprite(visible[0], fill)
    top_rows = [int(first_top)]
    for panel in visible[1:]:
        sprite, sprite_mask, top, left = _extract_sprite(panel, fill)
        assert left == template_left
        assert sprite_mask.shape == template_mask.shape
        assert bool(np.array_equal(sprite_mask, template_mask))
        assert bool(np.array_equal(sprite[sprite_mask], template_sprite[template_mask]))
        top_rows.append(int(top))

    deltas = [int(b - a) for a, b in zip(top_rows, top_rows[1:])]
    assert deltas and len(set(deltas)) == 1
    next_top = _round_half_up(
        Fraction(top_rows[-1], 1) + Fraction(deltas[0] * panel_w, stride)
    )

    sprite_h, sprite_w = template_sprite.shape
    assert 1 <= next_top and next_top + sprite_h <= h - 1
    assert 1 <= template_left and template_left + sprite_w <= panel_w - 1

    out = np.full((h, panel_w), fill, dtype=int)
    out[0, :] = frame
    out[-1, :] = frame
    out[:, 0] = frame
    out[:, -1] = frame
    _paste(out, template_sprite, next_top, template_left, fill)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extrapolate the moving sprite sequence from shared-border panels into one standalone next frame.

    The sprite keeps the same appearance while its row position changes by a constant amount
    between adjacent visible frames. The final input panel is blank; the output is the detached
    next panel, so the last step is scaled by full panel width rather than the shared-border
    spacing used inside the input strip.
    """
    g = np.asarray(grid, dtype=int)
    try:
        return _solve_horizontal(g)
    except AssertionError:
        return _solve_horizontal(g.T).T


_RNG = np.random.default_rng()
_MASK_CACHE: dict[tuple[int, int], list[np.ndarray]] = {}
_COLOR_VALUES = tuple(range(10))
_HEIGHT_RANGE = (8, 16)
_INTERIOR_WIDTH_RANGE = (4, 7)
_VISIBLE_PANEL_COUNTS = (2, 3, 4)
_SPRITE_HEIGHTS = (1, 2, 3)
_SPRITE_WIDTHS = (2, 3, 4)
_DELTA_CHOICES = (-3, -2, -1, 1, 2, 3)
_MAX_ATTEMPTS = 400


def _randint(lo: int, hi: int) -> int:
    return int(_RNG.integers(int(lo), int(hi) + 1))


def _choice(seq):
    return seq[int(_RNG.integers(0, len(seq)))]


def _sample(seq, k: int) -> list[int]:
    idx = _RNG.choice(len(seq), size=int(k), replace=False)
    return [int(seq[int(i)]) for i in idx]


def _mask_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    start = tuple(map(int, pts[0]))
    seen = {start}
    q = deque([start])
    h, w = mask.shape
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]) and (rr, cc) not in seen:
                seen.add((rr, cc))
                q.append((rr, cc))
    return len(seen) == int(pts.shape[0])


def _contiguous_masks(sprite_h: int, sprite_w: int) -> list[np.ndarray]:
    key = (int(sprite_h), int(sprite_w))
    if key in _MASK_CACHE:
        return _MASK_CACHE[key]

    cells = [(r, c) for r in range(sprite_h) for c in range(sprite_w)]
    masks: list[np.ndarray] = []
    max_pixels = min(sprite_h * sprite_w, 6)
    for count in range(2, max_pixels + 1):
        for combo in itertools.combinations(cells, count):
            rows = {r for r, _ in combo}
            cols = {c for _, c in combo}
            if len(rows) != sprite_h or len(cols) != sprite_w:
                continue
            mask = np.zeros((sprite_h, sprite_w), dtype=bool)
            for r, c in combo:
                mask[r, c] = True
            if _mask_connected(mask):
                masks.append(mask)
    assert masks
    _MASK_CACHE[key] = masks
    return masks


def _choose_palette() -> tuple[int, int]:
    frame, fill = _sample(_COLOR_VALUES, 2)
    return int(frame), int(fill)


def _make_sprite(fill: int, frame: int) -> np.ndarray:
    sprite_h = _choice(_SPRITE_HEIGHTS)
    sprite_w = _choice(_SPRITE_WIDTHS)
    mask = _choice(_contiguous_masks(sprite_h, sprite_w))
    sprite_colors = _choose_sprite_colors(frame, fill)

    sprite = np.full((sprite_h, sprite_w), fill, dtype=int)
    true_cells = np.argwhere(mask)
    perm = _RNG.permutation(len(true_cells))
    ordered = [tuple(map(int, true_cells[int(i)])) for i in perm]
    for i, color in enumerate(sprite_colors):
        r, c = ordered[i]
        sprite[r, c] = int(color)
    for r, c in ordered[len(sprite_colors) :]:
        sprite[r, c] = int(_choice(sprite_colors))
    return sprite


def _choose_sprite_colors(frame: int, fill: int) -> list[int]:
    remaining = [c for c in _COLOR_VALUES if c not in (frame, fill)]
    use_frame = bool(_RNG.random() < 0.3)
    color_count = 1 if bool(_RNG.random() < 0.6) else 2
    if use_frame and color_count == 1:
        sprite_colors = [frame]
    elif use_frame:
        sprite_colors = [frame] + _sample(remaining, color_count - 1)
    else:
        sprite_colors = _sample(remaining, color_count)
    return [int(c) for c in sprite_colors]


def _build_panel(height: int, panel_w: int, frame: int, fill: int) -> np.ndarray:
    panel = np.full((height, panel_w), fill, dtype=int)
    panel[0, :] = frame
    panel[-1, :] = frame
    panel[:, 0] = frame
    panel[:, -1] = frame
    return panel


def _motion_choices(height: int, sprite_h: int, panel_w: int, visible_count: int) -> list[tuple[int, int, int]]:
    max_top = height - 1 - sprite_h
    stride = panel_w - 1
    choices: list[tuple[int, int, int]] = []
    for delta in _DELTA_CHOICES:
        for start in range(1, max_top + 1):
            top_rows = [start + i * delta for i in range(visible_count)]
            if not all(1 <= top <= max_top for top in top_rows):
                continue
            next_top = _round_half_up(
                Fraction(top_rows[-1], 1) + Fraction(delta * panel_w, stride)
            )
            if 1 <= next_top <= max_top:
                choices.append((int(delta), int(start), int(next_top)))
    return choices


def _make_input(
    height: int,
    panel_w: int,
    visible_count: int,
    frame: int,
    fill: int,
    sprite: np.ndarray,
    start_top: int,
    delta: int,
    local_left: int,
) -> np.ndarray:
    total_panels = visible_count + 1
    stride = panel_w - 1
    width = total_panels * stride + 1
    grid = np.full((height, width), fill, dtype=int)
    grid[0, :] = frame
    grid[-1, :] = frame
    for panel_i in range(total_panels + 1):
        grid[:, panel_i * stride] = frame
    for panel_i in range(visible_count):
        top = start_top + panel_i * delta
        _paste(grid, sprite, top, panel_i * stride + local_left, fill)
    return grid


def _gen_attempt() -> np.ndarray | None:
    height = _randint(*_HEIGHT_RANGE)
    panel_w = _randint(*_INTERIOR_WIDTH_RANGE) + 2
    frame, fill = _choose_palette()
    sprite = _make_sprite(fill, frame)
    sprite_h, sprite_w = sprite.shape
    if sprite_w > panel_w - 2:
        return None

    visible_count = _choice(_VISIBLE_PANEL_COUNTS)
    local_left_max = panel_w - sprite_w - 1
    if local_left_max < 1:
        return None
    local_left = _randint(1, local_left_max)

    choices = _motion_choices(height, sprite_h, panel_w, visible_count)
    if not choices:
        return None
    delta, start_top, next_top = _choice(choices)

    expected = _build_panel(height, panel_w, frame, fill)
    _paste(expected, sprite, next_top, local_left, fill)
    inp = _make_input(
        height=height,
        panel_w=panel_w,
        visible_count=visible_count,
        frame=frame,
        fill=fill,
        sprite=sprite,
        start_top=start_top,
        delta=delta,
        local_left=local_left,
    )
    if bool(_RNG.random() < 0.35):
        inp = inp.T
        expected = expected.T

    if np.array_equal(main(inp), expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
