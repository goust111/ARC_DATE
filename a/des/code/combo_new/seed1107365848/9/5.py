from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
SEPARATOR = 5
DAMAGE = 8
PATTERN_COLORS = (1, 2, 3, 4, 6, 7, 9)
MAX_ATTEMPTS = 300
MASK_ATTEMPTS = 200


def _split_panels(grid: np.ndarray) -> tuple[list[np.ndarray], int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h % 2 == 1 and w % 2 == 1, "grid must be split by a centered cross"

    sep_r = h // 2
    sep_c = w // 2
    sep = int(g[sep_r, sep_c])
    assert np.all(g[sep_r, :] == sep), "middle row must be a full separator"
    assert np.all(g[:, sep_c] == sep), "middle column must be a full separator"

    panels = [
        g[:sep_r, :sep_c],
        g[:sep_r, sep_c + 1 :],
        g[sep_r + 1 :, :sep_c],
        g[sep_r + 1 :, sep_c + 1 :],
    ]
    shapes = {panel.shape for panel in panels}
    assert len(shapes) == 1, "the four panels must have equal size"
    return [panel.copy() for panel in panels], sep


def _repair_panels(panels: list[np.ndarray]) -> tuple[np.ndarray, int]:
    ph, pw = panels[0].shape
    out = np.empty((ph, pw), dtype=int)
    damage_colors: set[int] = set()

    for r in range(ph):
        for c in range(pw):
            counts = Counter(int(panel[r, c]) for panel in panels)
            if len(counts) == 1:
                out[r, c] = next(iter(counts))
                continue

            common = counts.most_common()
            assert len(common) == 2, "each local position may differ in only one panel"
            (majority_color, majority_count), (minority_color, minority_count) = common
            assert (majority_count, minority_count) == (3, 1), "damage must affect exactly one panel per position"
            out[r, c] = int(majority_color)
            damage_colors.add(int(minority_color))

    assert damage_colors, "at least one damaged pixel is required"
    assert len(damage_colors) == 1, "all damaged pixels must share one marker color"
    return out, next(iter(damage_colors))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the complete pattern by reconciling four damaged panel copies.

    Input structure:
    - A rectangular grid is split by one full horizontal separator row and one full vertical separator column into four equal panels.
    - Each panel contains the same underlying picture, but some cells are replaced by one dedicated damage color.
    - At any local coordinate, at most one of the four panels is damaged.

    Rule:
    - Treat the middle full row and middle full column as separators and discard them.
    - For each local panel coordinate, compare the four visible colors from the four panels.
    - If all four colors match, keep that color; otherwise the valid disagreement is a 3-to-1 split, where the singleton color is the damage marker and the majority color is the true pixel.
    - Output the repaired single panel with no separator lines.

    Key insight:
    - The four panels are redundant noisy replicas, so every disagreement exposes the damage marker as the unique outlier and the true image as the repeated color.
    """
    panels, _sep = _split_panels(np.asarray(grid, dtype=int))
    repaired, _damage = _repair_panels(panels)
    return repaired


def _paint_rectangle(mask: np.ndarray) -> None:
    h, w = mask.shape
    rh = random.randint(1, h)
    rw = random.randint(1, w)
    r0 = random.randint(0, h - rh)
    c0 = random.randint(0, w - rw)
    mask[r0 : r0 + rh, c0 : c0 + rw] = True


def _paint_frame(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h < 3 or w < 3:
        return False
    rh = random.randint(3, h)
    rw = random.randint(3, w)
    r0 = random.randint(0, h - rh)
    c0 = random.randint(0, w - rw)
    mask[r0, c0 : c0 + rw] = True
    mask[r0 + rh - 1, c0 : c0 + rw] = True
    mask[r0 : r0 + rh, c0] = True
    mask[r0 : r0 + rh, c0 + rw - 1] = True
    return True


def _paint_hbar(mask: np.ndarray) -> None:
    h, w = mask.shape
    r = random.randrange(h)
    length = random.randint(2, w)
    c0 = random.randint(0, w - length)
    mask[r, c0 : c0 + length] = True


def _paint_vbar(mask: np.ndarray) -> None:
    h, w = mask.shape
    c = random.randrange(w)
    length = random.randint(2, h)
    r0 = random.randint(0, h - length)
    mask[r0 : r0 + length, c] = True


def _paint_plus(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h < 3 or w < 3:
        return False
    r = random.randrange(h)
    c = random.randrange(w)
    up = random.randint(0, r)
    down = random.randint(0, h - 1 - r)
    left = random.randint(0, c)
    right = random.randint(0, w - 1 - c)
    if up + down + left + right == 0:
        return False
    mask[r - up : r + down + 1, c] = True
    mask[r, c - left : c + right + 1] = True
    return True


def _paint_l(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h < 2 or w < 2:
        return False
    r0 = random.randrange(h - 1)
    c0 = random.randrange(w - 1)
    vh = random.randint(2, h - r0)
    hw = random.randint(2, w - c0)
    orientation = random.choice(("dr", "dl", "ur", "ul"))
    if orientation == "dr":
        mask[r0 : r0 + vh, c0] = True
        mask[r0 + vh - 1, c0 : c0 + hw] = True
    elif orientation == "dl":
        mask[r0 : r0 + vh, c0 + hw - 1] = True
        mask[r0 + vh - 1, c0 : c0 + hw] = True
    elif orientation == "ur":
        mask[r0 : r0 + vh, c0] = True
        mask[r0, c0 : c0 + hw] = True
    else:
        mask[r0 : r0 + vh, c0 + hw - 1] = True
        mask[r0, c0 : c0 + hw] = True
    return True


def _make_mask(height: int, width: int) -> np.ndarray | None:
    area_total = height * width
    min_area = max(8, area_total // 6)
    max_area = max(min_area, area_total - max(3, area_total // 5))

    painters = (
        _paint_rectangle,
        _paint_frame,
        _paint_hbar,
        _paint_vbar,
        _paint_plus,
        _paint_l,
    )

    for _ in range(MASK_ATTEMPTS):
        mask = np.zeros((height, width), dtype=bool)
        primitive_count = random.randint(2, 5)
        for _ in range(primitive_count):
            for _ in range(10):
                painter = random.choice(painters)
                probe = mask.copy()
                ok = painter(probe)
                if ok is False:
                    continue
                mask = probe
                break

        area = int(mask.sum())
        if not (min_area <= area <= max_area):
            continue
        if not np.any(~mask):
            continue
        if np.count_nonzero(mask.any(axis=1)) < 2 or np.count_nonzero(mask.any(axis=0)) < 2:
            continue
        return mask
    return None


def _colorize_mask(mask: np.ndarray) -> np.ndarray | None:
    height, width = mask.shape
    pattern = np.zeros((height, width), dtype=int)
    scheme = random.choice(("mono", "mono", "row", "col", "diag"))

    if scheme == "mono":
        color = random.choice(PATTERN_COLORS)
        pattern[mask] = color
        return pattern

    palette_size = random.choice((2, 3))
    palette = random.sample(PATTERN_COLORS, k=palette_size)
    rows, cols = np.nonzero(mask)
    for r, c in zip(rows, cols):
        if scheme == "row":
            color = palette[r % palette_size]
        elif scheme == "col":
            color = palette[c % palette_size]
        else:
            color = palette[(r + c) % palette_size]
        pattern[r, c] = color

    used_colors = [int(c) for c in np.unique(pattern[mask])]
    if len(used_colors) < 2:
        return None
    if min(int(np.count_nonzero(pattern == color)) for color in used_colors) < 2:
        return None
    return pattern


def _make_pattern(height: int, width: int) -> np.ndarray | None:
    mask = _make_mask(height, width)
    if mask is None:
        return None
    pattern = _colorize_mask(mask)
    if pattern is None:
        return None
    if np.count_nonzero(pattern) < 8:
        return None
    return pattern


def _damage_pattern(pattern: np.ndarray) -> list[np.ndarray] | None:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(pattern != BG)]
    if len(coords) < 8:
        return None

    panels = [pattern.copy() for _ in range(4)]
    max_damage = min(len(coords), max(4, int(round(len(coords) * 0.6))))
    total_damage = random.randint(4, max_damage)
    damaged_coords = random.sample(coords, k=total_damage)
    random.shuffle(damaged_coords)

    for panel_idx, coord in enumerate(damaged_coords[:4]):
        panels[panel_idx][coord] = DAMAGE
    for coord in damaged_coords[4:]:
        panels[random.randrange(4)][coord] = DAMAGE

    return panels


def _assemble_grid(panels: list[np.ndarray]) -> np.ndarray:
    ph, pw = panels[0].shape
    grid = np.full((2 * ph + 1, 2 * pw + 1), BG, dtype=int)
    grid[ph, :] = SEPARATOR
    grid[:, pw] = SEPARATOR
    grid[:ph, :pw] = panels[0]
    grid[:ph, pw + 1 :] = panels[1]
    grid[ph + 1 :, :pw] = panels[2]
    grid[ph + 1 :, pw + 1 :] = panels[3]
    return grid


def _gen_attempt() -> np.ndarray | None:
    panel_h = random.randint(4, 14)
    panel_w = random.randint(4, 14)

    pattern = _make_pattern(panel_h, panel_w)
    if pattern is None:
        return None
    panels = _damage_pattern(pattern)
    if panels is None:
        return None

    grid = _assemble_grid(panels)
    try:
        if not np.array_equal(main(grid), pattern):
            return None
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid panel-repair puzzle")
