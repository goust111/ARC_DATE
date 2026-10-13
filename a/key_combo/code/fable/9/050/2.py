from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
GEN_RETRIES = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def _count_true_components(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
    return count


def _foreground_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] == BG or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            colors = set()
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                colors.add(int(grid[rr, cc]))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] != BG and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "colors": colors,
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _count_enclosed_holes(sprite: np.ndarray) -> int:
    h, w = sprite.shape
    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if sprite[r, c] == BG and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if sprite[r, c] == BG and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))

    while q:
        rr, cc = q.popleft()
        for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
            if 0 <= nr < h and 0 <= nc < w and sprite[nr, nc] == BG and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))

    holes = 0
    for r in range(h):
        for c in range(w):
            if sprite[r, c] != BG or seen[r, c]:
                continue
            holes += 1
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and sprite[nr, nc] == BG and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
    return holes


def _split_candidates(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[dict] = []

    for r in range(1, h - 1):
        if not np.all(g[r, :] == BG):
            continue
        top = g[:r, :]
        bottom = g[r + 1 :, :]
        if top.size == 0 or bottom.size == 0 or top.size == bottom.size:
            continue
        control, pattern = (top, bottom) if top.size < bottom.size else (bottom, top)
        comps = _foreground_components(control)
        if len(comps) != 1:
            continue
        comp = comps[0]
        if len(comp["colors"]) != 1:
            continue
        cr0, _cc0, cr1, _cc1 = comp["bbox"]
        if cr0 != 0 or cr1 != control.shape[0] - 1:
            continue
        sprite = _crop_bbox(control, comp["bbox"])
        holes = _count_enclosed_holes(sprite)
        if holes < 1:
            continue
        candidates.append(
            {
                "kind": "row",
                "index": r,
                "control": control,
                "pattern": pattern,
                "sprite": sprite,
                "holes": holes,
            }
        )

    for c in range(1, w - 1):
        if not np.all(g[:, c] == BG):
            continue
        left = g[:, :c]
        right = g[:, c + 1 :]
        if left.size == 0 or right.size == 0 or left.size == right.size:
            continue
        control, pattern = (left, right) if left.size < right.size else (right, left)
        comps = _foreground_components(control)
        if len(comps) != 1:
            continue
        comp = comps[0]
        if len(comp["colors"]) != 1:
            continue
        _cr0, cc0, _cr1, cc1 = comp["bbox"]
        if cc0 != 0 or cc1 != control.shape[1] - 1:
            continue
        sprite = _crop_bbox(control, comp["bbox"])
        holes = _count_enclosed_holes(sprite)
        if holes < 1:
            continue
        candidates.append(
            {
                "kind": "col",
                "index": c,
                "control": control,
                "pattern": pattern,
                "sprite": sprite,
                "holes": holes,
            }
        )

    return candidates


def _parse_instance(grid: np.ndarray) -> tuple[np.ndarray, int, int]:
    candidates = _split_candidates(grid)
    assert len(candidates) == 1, "expected exactly one valid separator/control interpretation"
    info = candidates[0]
    sprite = np.asarray(info["sprite"], dtype=int)
    sh, sw = sprite.shape
    assert sh != sw, "control sprite must be non-square"
    axis = 1 if sw > sh else 0
    return np.asarray(info["pattern"], dtype=int), int(info["holes"]), int(axis)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the control sprite's enclosed holes, use aspect ratio to choose an axis, and cyclically roll the main pattern.

    Input structure:
    - A full black separator row or column splits the grid into a larger colorful pattern region and a smaller control region.
    - The smaller region contains exactly one monochrome hollow control sprite on black background.
    - The control sprite encloses one or more disconnected black hole regions.

    Rule:
    - Count the number of enclosed black hole components inside the control sprite; call it h.
    - If the sprite's bounding box is wider than it is tall, roll the pattern region h columns to the right.
    - If the sprite's bounding box is taller than it is wide, roll the pattern region h rows downward.
    - Drop the separator band and the control region; output only the rolled pattern region.

    Key insight:
    - The control sprite uses two independent visible properties: enclosed-hole count gives the roll amount, while aspect ratio selects whether the cyclic roll is horizontal or vertical.
    """
    pattern, shift, axis = _parse_instance(np.asarray(grid, dtype=int))
    return np.roll(pattern, shift=shift, axis=axis)


def _random_composition(total: int, parts: int, min_part: int = 1, max_part: int | None = None) -> list[int]:
    if max_part is None:
        max_part = total
    values = [min_part] * parts
    remaining = total - min_part * parts
    while remaining > 0:
        idx = random.randrange(parts)
        if values[idx] >= max_part:
            continue
        values[idx] += 1
        remaining -= 1
    random.shuffle(values)
    return values


def _make_control_sprite(hole_count: int, horizontal_roll: bool, color: int) -> np.ndarray:
    if horizontal_roll:
        inner_h = random.randint(2, 5)
        chamber_widths = [random.randint(1, 3) for _ in range(hole_count)]
        inner_w = sum(chamber_widths) + (hole_count - 1)
        while inner_w + 2 <= inner_h + 2:
            chamber_widths[random.randrange(hole_count)] += 1
            inner_w += 1
        sprite = np.full((inner_h + 2, inner_w + 2), color, dtype=int)
        sprite[1:-1, 1:-1] = BG
        cursor = 1 + chamber_widths[0]
        for width in chamber_widths[1:]:
            sprite[1:-1, cursor] = color
            cursor += width + 1
        return sprite

    inner_w = random.randint(2, 5)
    chamber_heights = [random.randint(1, 3) for _ in range(hole_count)]
    inner_h = sum(chamber_heights) + (hole_count - 1)
    while inner_h + 2 <= inner_w + 2:
        chamber_heights[random.randrange(hole_count)] += 1
        inner_h += 1
    sprite = np.full((inner_h + 2, inner_w + 2), color, dtype=int)
    sprite[1:-1, 1:-1] = BG
    cursor = 1 + chamber_heights[0]
    for height in chamber_heights[1:]:
        sprite[cursor, 1:-1] = color
        cursor += height + 1
    return sprite


def _random_pattern(h: int, w: int, colors: list[int]) -> np.ndarray:
    pattern = np.random.choice(colors, size=(h, w)).astype(int)
    for r in range(h):
        if len(set(map(int, pattern[r, :]))) == 1:
            pattern[r, random.randrange(w)] = random.choice(colors)
    for c in range(w):
        if len(set(map(int, pattern[:, c]))) == 1:
            pattern[random.randrange(h), c] = random.choice(colors)
    return pattern


def _has_roll_period(grid: np.ndarray, axis: int) -> bool:
    axis_len = grid.shape[axis]
    return any(np.array_equal(np.roll(grid, s, axis=axis), grid) for s in range(1, axis_len))


def _semantically_distinct(pattern: np.ndarray, shift: int, axis: int) -> bool:
    axis_len = pattern.shape[axis]
    if not (0 < shift < axis_len):
        return False
    if _has_roll_period(pattern, 0) or _has_roll_period(pattern, 1):
        return False
    expected = np.roll(pattern, shift, axis=axis)
    if np.array_equal(expected, pattern):
        return False
    other_axis = 1 - axis
    for s in range(pattern.shape[other_axis]):
        if np.array_equal(np.roll(pattern, s, axis=other_axis), expected):
            return False
    return True


def _build_input(pattern: np.ndarray, sprite: np.ndarray, separator_is_column: bool) -> np.ndarray:
    ph, pw = pattern.shape
    sh, sw = sprite.shape

    if separator_is_column:
        control_w = sw
        full_h = ph
        full_w = pw + 1 + control_w
        grid = np.zeros((full_h, full_w), dtype=int)
        side_left = bool(random.getrandbits(1))
        sprite_top = random.randint(0, full_h - sh)
        if side_left:
            grid[:, control_w + 1 :] = pattern
            grid[sprite_top : sprite_top + sh, :control_w] = sprite
        else:
            grid[:, :pw] = pattern
            grid[sprite_top : sprite_top + sh, pw + 1 :] = sprite
        return grid

    control_h = sh
    full_h = ph + 1 + control_h
    full_w = pw
    grid = np.zeros((full_h, full_w), dtype=int)
    side_top = bool(random.getrandbits(1))
    sprite_left = random.randint(0, full_w - sw)
    if side_top:
        grid[control_h + 1 :, :] = pattern
        grid[:control_h, sprite_left : sprite_left + sw] = sprite
    else:
        grid[:ph, :] = pattern
        grid[ph + 1 :, sprite_left : sprite_left + sw] = sprite
    return grid


def _validate_generated(grid: np.ndarray, pattern: np.ndarray, holes: int, axis: int) -> bool:
    if grid.ndim != 2 or pattern.ndim != 2:
        return False
    if max(grid.shape) > 30 or max(pattern.shape) > 30:
        return False
    candidates = _split_candidates(grid)
    if len(candidates) != 1:
        return False
    parsed_pattern, parsed_holes, parsed_axis = _parse_instance(grid)
    if parsed_holes != holes or parsed_axis != axis:
        return False
    if not np.array_equal(parsed_pattern, pattern):
        return False
    if not _semantically_distinct(pattern, holes, axis):
        return False
    expected = np.roll(pattern, holes, axis=axis)
    if not np.array_equal(main(grid), expected):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    holes = random.randint(1, 4)
    horizontal_roll = bool(random.getrandbits(1))
    separator_is_column = bool(random.getrandbits(1))

    control_color = random.choice(PALETTE)
    sprite = _make_control_sprite(holes, horizontal_roll, control_color)
    sh, sw = sprite.shape

    if separator_is_column:
        ph_min = max(sh, holes + 3, 6)
        ph_max = 24
        pw_min = max(sw + 1, holes + 3, 6)
        pw_max = 29 - sw
        if ph_min > ph_max or pw_min > pw_max:
            return None
        ph = random.randint(ph_min, ph_max)
        pw = random.randint(pw_min, pw_max)
    else:
        pw_min = max(sw, holes + 3, 6)
        pw_max = 24
        ph_min = max(sh + 1, holes + 3, 6)
        ph_max = 29 - sh
        if pw_min > pw_max or ph_min > ph_max:
            return None
        pw = random.randint(pw_min, pw_max)
        ph = random.randint(ph_min, ph_max)

    pattern_colors = [c for c in PALETTE if c != control_color]
    random.shuffle(pattern_colors)
    color_count = random.randint(4, min(6, len(pattern_colors)))
    pattern = _random_pattern(ph, pw, pattern_colors[:color_count])
    axis = 1 if horizontal_roll else 0

    grid = _build_input(pattern, sprite, separator_is_column)
    if not _validate_generated(grid, pattern, holes, axis):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
