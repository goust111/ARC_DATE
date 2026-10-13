from __future__ import annotations

import random

import numpy as np


BG = 0
OCCLUDER = 5


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_occluder_band(grid: np.ndarray, bg: int) -> tuple[int, int, int]:
    best: tuple[int, int, int, int] | None = None
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        cols = np.flatnonzero(np.all(grid == color, axis=0))
        if cols.size == 0:
            continue
        start = 0
        while start < cols.size:
            end = start
            while end + 1 < cols.size and int(cols[end + 1]) == int(cols[end]) + 1:
                end += 1
            c0 = int(cols[start])
            c1 = int(cols[end])
            run = (c1 - c0 + 1, color, c0, c1)
            if best is None or run[0] > best[0]:
                best = run
            start = end + 1
    assert best is not None, "expected one full-height vertical occluder band"
    _width, color, c0, c1 = best
    return color, c0, c1


def _row_intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    rows = np.any(mask, axis=1)
    intervals: list[tuple[int, int]] = []
    r = 0
    while r < mask.shape[0]:
        while r < mask.shape[0] and not bool(rows[r]):
            r += 1
        if r >= mask.shape[0]:
            break
        r0 = r
        while r < mask.shape[0] and bool(rows[r]):
            r += 1
        intervals.append((r0, r - 1))
    return intervals


def _group_snapshot(
    grid: np.ndarray, fg_mask: np.ndarray, r0: int, r1: int, band_c0: int, band_c1: int
) -> dict:
    rows, cols = np.where(fg_mask[r0 : r1 + 1])
    assert rows.size > 0
    rows = rows.astype(int) + int(r0)
    cols = cols.astype(int)
    colors = grid[rows, cols].astype(int)

    left_cols = cols[cols < band_c0]
    right_cols = cols[cols > band_c1]
    return {
        "r0": int(r0),
        "r1": int(r1),
        "height": int(r1 - r0 + 1),
        "pixels": list(zip(map(int, rows), map(int, cols), map(int, colors))),
        "left_edge": None if left_cols.size == 0 else int(left_cols.min()),
        "right_edge": None if right_cols.size == 0 else int(right_cols.max()),
    }


def _constant_step(samples: list[tuple[int, int]]) -> int | None:
    if len(samples) < 2:
        return None
    step: int | None = None
    for i, value_i in samples:
        for j, value_j in samples:
            if j <= i:
                continue
            diff_index = j - i
            diff_value = value_j - value_i
            if diff_value % diff_index != 0:
                return None
            candidate = diff_value // diff_index
            if step is None:
                step = candidate
            elif step != candidate:
                return None
    return step


def _render_with_occlusion(
    shape: tuple[int, int],
    bg: int,
    band_color: int,
    band_c0: int,
    band_c1: int,
    sprite: np.ndarray,
    positions: list[tuple[int, int]],
) -> np.ndarray | None:
    out = np.full(shape, bg, dtype=int)
    h, w = sprite.shape
    for top, left in positions:
        if top < 0 or left < 0 or top + h > shape[0] or left + w > shape[1]:
            return None
        patch = out[top : top + h, left : left + w]
        conflict = (patch != bg) & (patch != sprite)
        if bool(np.any(conflict)):
            return None
        patch[:] = np.where(patch == bg, sprite, patch)
    out[:, band_c0 : band_c1 + 1] = band_color
    return out


def _solve_with_order(
    grid: np.ndarray,
    bg: int,
    band_color: int,
    band_c0: int,
    band_c1: int,
    ordered_groups: list[dict],
) -> np.ndarray | None:
    if len(ordered_groups) < 2:
        return None

    tops = [group["r0"] for group in ordered_groups]
    dr = tops[1] - tops[0]
    if dr == 0 or any(tops[i + 1] - tops[i] != dr for i in range(len(tops) - 1)):
        return None

    heights = {group["height"] for group in ordered_groups}
    if len(heights) != 1:
        return None
    sprite_h = heights.pop()

    left_samples = [
        (idx, int(group["left_edge"]))
        for idx, group in enumerate(ordered_groups)
        if group["left_edge"] is not None
    ]
    right_samples = [
        (idx, int(group["right_edge"]))
        for idx, group in enumerate(ordered_groups)
        if group["right_edge"] is not None
    ]
    if not left_samples or not right_samples:
        return None

    dc: int | None = None
    for samples in (left_samples, right_samples):
        if len(samples) < 2:
            continue
        candidate = _constant_step(samples)
        if candidate is None or candidate == 0:
            return None
        if dc is None:
            dc = candidate
        elif dc != candidate:
            return None
    if dc is None:
        return None

    anchor_index, anchor_left = left_samples[0]
    left_positions = [anchor_left + (idx - anchor_index) * dc for idx in range(len(ordered_groups))]
    for idx, left_edge in left_samples:
        if left_positions[idx] != left_edge:
            return None

    widths = [right_edge - left_positions[idx] + 1 for idx, right_edge in right_samples]
    if not widths or min(widths) <= 0 or len(set(widths)) != 1:
        return None
    sprite_w = widths[0]

    sprite = np.full((sprite_h, sprite_w), -1, dtype=int)
    positions: list[tuple[int, int]] = []
    for idx, group in enumerate(ordered_groups):
        top = int(group["r0"])
        left = int(left_positions[idx])
        positions.append((top, left))
        for row, col, color in group["pixels"]:
            local_r = row - top
            local_c = col - left
            if not (0 <= local_r < sprite_h and 0 <= local_c < sprite_w):
                return None
            if sprite[local_r, local_c] == -1:
                sprite[local_r, local_c] = color
            elif sprite[local_r, local_c] != color:
                return None

    if bool(np.any(sprite < 0)):
        return None

    rerendered = _render_with_occlusion(
        grid.shape, bg, band_color, band_c0, band_c1, sprite, positions
    )
    if rerendered is None or not np.array_equal(rerendered, grid):
        return None

    last_top, last_left = positions[-1]
    next_top = last_top + dr
    next_left = last_left + dc
    if next_top < 0 or next_left < 0 or next_top + sprite_h > grid.shape[0] or next_left + sprite_w > grid.shape[1]:
        return None
    if dc > 0 and not (next_left > band_c1):
        return None
    if dc < 0 and not (next_left + sprite_w - 1 < band_c0):
        return None

    target = grid[next_top : next_top + sprite_h, next_left : next_left + sprite_w]
    if bool(np.any(target != bg)):
        return None

    out = grid.copy()
    out[next_top : next_top + sprite_h, next_left : next_left + sprite_w] = sprite
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the fully occluded moving sprite from partial snapshots and stamp one complete next frame.

    Input structure:
    - A black background contains one full-height vertical grey occluder band covering several consecutive columns.
    - One multicolor sprite appears in several equally spaced motion snapshots along a fixed direction.
    - Each snapshot is partially hidden by the occluder band, and no single snapshot shows the whole sprite.

    Rule:
    - Identify the grey band and treat the remaining colored pixels as ordered motion snapshots of one rigid sprite.
    - Use the equal spacing between snapshots to align their visible fragments and take the union of those aligned fragments to recover the full sprite.
    - Extrapolate one more step in the same direction and stamp the complete sprite at that next position, while leaving the original snapshots and the occluder unchanged.

    Key insight:
    - The full sprite never appears directly in the input; it only exists after aligning the partially visible frames by their motion step and merging their complementary visible parts.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _most_common_color(g)
    band_color, band_c0, band_c1 = _find_occluder_band(g, bg)
    fg_mask = (g != bg) & (g != band_color)
    intervals = _row_intervals(fg_mask)
    assert len(intervals) >= 2, "expected several separated snapshot rows"

    groups = [_group_snapshot(g, fg_mask, r0, r1, band_c0, band_c1) for r0, r1 in intervals]
    for ordered in (groups, list(reversed(groups))):
        out = _solve_with_order(g, bg, band_color, band_c0, band_c1, ordered)
        if out is not None:
            return out

    raise AssertionError("failed to recover the moving sprite and extrapolated frame")


def _sprite_palette() -> list[int]:
    colors = [1, 2, 3, 4, 6, 7, 8, 9]
    random.shuffle(colors)
    count = random.randint(2, 4)
    return colors[:count]


def _make_sprite(height: int, width: int) -> np.ndarray:
    palette = _sprite_palette()
    mode = random.choice(("checker", "diagonal", "vertical", "rowmix"))
    offset = random.randint(0, len(palette) - 1)
    sprite = np.zeros((height, width), dtype=int)
    for r in range(height):
        for c in range(width):
            if mode == "checker":
                idx = (r + c + offset) % len(palette)
            elif mode == "diagonal":
                idx = (2 * r + c + offset) % len(palette)
            elif mode == "vertical":
                idx = (c + offset) % len(palette)
            else:
                idx = (r + 2 * c + offset) % len(palette)
            sprite[r, c] = palette[idx]
    if len(set(map(int, sprite.ravel()))) == 1:
        sprite[0, 0] = palette[-1]
    return sprite


def _columns_fully_recoverable(width: int, band_width: int, offsets: list[int]) -> bool:
    for local_c in range(width):
        visible_somewhere = False
        for offset in offsets:
            if not (offset <= local_c <= offset + band_width - 1):
                visible_somewhere = True
                break
        if not visible_somewhere:
            return False
    return True


def _pick_motion_profile() -> tuple[int, int, int, int, int]:
    for _ in range(400):
        frame_count = random.randint(3, 4)
        sprite_w = random.randint(5, 7)
        band_w = random.randint(2, min(4, sprite_w - 1))
        dc = random.randint(1, 3)
        possible_d0 = []
        for d0 in range(1, sprite_w + 4):
            offsets = [d0 - i * dc for i in range(frame_count)]
            next_offset = d0 - frame_count * dc
            if not all(-(band_w - 1) <= offset <= sprite_w - 1 for offset in offsets):
                continue
            if next_offset > -band_w:
                continue
            if sum(offset > 0 for offset in offsets) < 2:
                continue
            if sum(offset < sprite_w - band_w for offset in offsets) < 1:
                continue
            if not any(0 < offset < sprite_w - band_w for offset in offsets):
                continue
            if not _columns_fully_recoverable(sprite_w, band_w, offsets):
                continue
            possible_d0.append(d0)
        if possible_d0:
            return frame_count, sprite_w, band_w, dc, random.choice(possible_d0)
    raise RuntimeError("failed to sample a recoverable motion profile")


def _make_case() -> tuple[np.ndarray, np.ndarray]:
    for _ in range(400):
        sprite_h = random.randint(3, 5)
        frame_count, sprite_w, band_w, dc, d0 = _pick_motion_profile()
        sprite = _make_sprite(sprite_h, sprite_w)

        dr = random.randint(sprite_h + 1, sprite_h + 3)
        top_margin = random.randint(1, 3)
        bottom_margin = random.randint(1, 3)
        left_margin = random.randint(1, 3)
        right_margin = random.randint(1, 3)

        x_positions = [left_margin + i * dc for i in range(frame_count + 1)]
        y_positions = [top_margin + i * dr for i in range(frame_count + 1)]
        band_c0 = left_margin + d0
        band_c1 = band_c0 + band_w - 1

        height = y_positions[-1] + sprite_h + bottom_margin
        width = x_positions[-1] + sprite_w + right_margin
        if height > 30 or width > 30:
            continue

        inp = _render_with_occlusion(
            (height, width),
            BG,
            OCCLUDER,
            band_c0,
            band_c1,
            sprite,
            list(zip(y_positions[:frame_count], x_positions[:frame_count])),
        )
        if inp is None:
            continue

        expected = inp.copy()
        next_top = y_positions[frame_count]
        next_left = x_positions[frame_count]
        if bool(np.any(expected[next_top : next_top + sprite_h, next_left : next_left + sprite_w] != BG)):
            continue
        expected[next_top : next_top + sprite_h, next_left : next_left + sprite_w] = sprite

        if random.random() < 0.5:
            inp = np.fliplr(inp)
            expected = np.fliplr(expected)
        if random.random() < 0.5:
            inp = np.flipud(inp)
            expected = np.flipud(expected)

        actual = main(inp)
        if np.array_equal(actual, expected):
            return inp, expected
    raise RuntimeError("failed to generate a valid input/output pair")


def generate_input() -> np.ndarray:
    inp, _expected = _make_case()
    return inp
