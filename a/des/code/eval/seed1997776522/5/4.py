from __future__ import annotations

from collections import Counter

import numpy as np


BLACK = 0
GREY = 5
PIP_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
_RNG = np.random.default_rng()
_MAX_GENERATION_ATTEMPTS = 400

PIP_LAYOUTS = {
    1: ((2, 2),),
    2: ((1, 1), (3, 3)),
    3: ((1, 1), (2, 2), (3, 3)),
    4: ((1, 1), (1, 3), (3, 1), (3, 3)),
    5: ((1, 1), (1, 3), (2, 2), (3, 1), (3, 3)),
    6: ((1, 1), (2, 1), (3, 1), (1, 3), (2, 3), (3, 3)),
}
PIP_LAYOUT_SETS = {value: set(coords) for value, coords in PIP_LAYOUTS.items()}


def _background_color(grid: np.ndarray) -> int:
    border = np.concatenate((grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]))
    return int(Counter(map(int, border)).most_common(1)[0][0])


def _render_die(value: int, pip_color: int, grey_color: int = GREY) -> np.ndarray:
    tile = np.full((5, 5), int(grey_color), dtype=int)
    for r, c in PIP_LAYOUTS[int(value)]:
        tile[int(r), int(c)] = int(pip_color)
    return tile


def _decode_die(patch: np.ndarray, bg: int) -> dict[str, int] | None:
    values, counts = np.unique(patch, return_counts=True)
    if len(values) != 2:
        return None

    order = np.argsort(-counts)
    grey_color = int(values[int(order[0])])
    pip_color = int(values[int(order[1])])
    pip_count = int(counts[int(order[1])])

    if grey_color == bg or pip_color == bg or not (1 <= pip_count <= 6):
        return None

    coords = {tuple(map(int, coord)) for coord in np.argwhere(patch == pip_color)}
    if coords != PIP_LAYOUT_SETS[pip_count]:
        return None

    return {"grey": grey_color, "pip": pip_color, "value": pip_count}


def _find_dice(grid: np.ndarray, bg: int) -> list[dict[str, int]]:
    h, w = grid.shape
    dice: list[dict[str, int]] = []

    for r in range(h - 4):
        for c in range(w - 4):
            patch = grid[r : r + 5, c : c + 5]
            if bool((patch == bg).any()):
                continue

            decoded = _decode_die(patch, bg)
            if decoded is None:
                continue

            dice.append({"r": int(r), "c": int(c), **decoded})

    if not dice:
        raise ValueError("no dice found")

    positions = {(die["r"], die["c"]) for die in dice}
    if len(positions) != len(dice):
        raise ValueError("duplicate dice detections")

    pip_colors = [die["pip"] for die in dice]
    if len(set(pip_colors)) != len(pip_colors):
        raise ValueError("pip colors must be unique")

    return dice


def _contiguous_segment_indices(row: np.ndarray, bg: int) -> list[np.ndarray]:
    cols = np.flatnonzero(row != bg)
    if cols.size == 0:
        return []

    splits = np.where(np.diff(cols) > 1)[0] + 1
    return [segment for segment in np.split(cols, splits) if segment.size > 0]


def _read_bottom_strip(grid: np.ndarray, bg: int, expected_len: int) -> list[int]:
    for r in range(grid.shape[0] - 1, -1, -1):
        segments = _contiguous_segment_indices(grid[r], bg)
        for segment in segments:
            if int(segment.size) != int(expected_len):
                continue
            colors = [int(grid[r, c]) for c in segment]
            if len(set(colors)) != len(colors):
                continue
            return colors
    raise ValueError("could not identify bottom strip")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder scattered colored dice by the bottom color strip, then increment each die value by one cyclically.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)

    dice = _find_dice(g, bg)
    strip_colors = _read_bottom_strip(g, bg, expected_len=len(dice))

    by_color = {die["pip"]: die for die in dice}
    if set(strip_colors) != set(by_color):
        raise ValueError("strip colors must match the dice colors")

    out = np.full((5, 5 * len(strip_colors)), int(bg), dtype=int)
    for idx, color in enumerate(strip_colors):
        die = by_color[int(color)]
        new_value = 1 + (int(die["value"]) % 6)
        out[:, idx * 5 : (idx + 1) * 5] = _render_die(new_value, pip_color=int(color), grey_color=int(die["grey"]))

    return out


def _place_dice_positions(height: int, width: int, count: int) -> list[tuple[int, int]] | None:
    occupied = np.zeros((height, width), dtype=bool)
    occupied[-1, :] = True
    if height >= 2:
        occupied[-2, :] = True

    candidates = [(r, c) for r in range(1, height - 6) for c in range(1, width - 5)]
    if len(candidates) < count:
        return None

    _RNG.shuffle(candidates)
    positions: list[tuple[int, int]] = []

    for r, c in candidates:
        rr0 = max(0, r - 1)
        rr1 = min(height, r + 6)
        cc0 = max(0, c - 1)
        cc1 = min(width, c + 6)
        if bool(occupied[rr0:rr1, cc0:cc1].any()):
            continue

        positions.append((int(r), int(c)))
        occupied[rr0:rr1, cc0:cc1] = True
        if len(positions) == count:
            return positions

    return None


def _generate_attempt() -> np.ndarray | None:
    count = int(_RNG.integers(2, 7))
    colors = list(map(int, _RNG.choice(np.array(PIP_COLORS, dtype=int), size=count, replace=False)))
    values = list(map(int, _RNG.integers(1, 7, size=count)))

    height = int(_RNG.integers(10 + max(0, count - 4), 19 + max(0, count - 4)))
    width_min = max(14, count + 4)
    width = int(_RNG.integers(width_min, 31))
    if width <= count + 2:
        return None

    strip_start = int(_RNG.integers(1, width - count))
    positions = _place_dice_positions(height, width, count)
    if positions is None:
        return None

    grid = np.full((height, width), BLACK, dtype=int)

    perm = list(map(int, _RNG.permutation(count)))
    for die_index, pos_index in enumerate(perm):
        r, c = positions[pos_index]
        grid[r : r + 5, c : c + 5] = _render_die(values[die_index], pip_color=colors[die_index], grey_color=GREY)

    order = list(map(int, _RNG.permutation(count)))
    for idx, color_index in enumerate(order):
        grid[-1, strip_start + idx] = colors[color_index]

    try:
        detected = _find_dice(grid, BLACK)
        strip = _read_bottom_strip(grid, BLACK, expected_len=count)
        if len(detected) != count or set(strip) != set(colors):
            return None
    except ValueError:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
