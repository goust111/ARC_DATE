from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
SYMMETRY_TYPES = ("lr", "ud", "rot", "none")
GEN_ATTEMPTS = 600
PATTERN_ATTEMPTS = 400
PLACE_ATTEMPTS = 300


def _classify_pattern(pattern: np.ndarray) -> str:
    assert pattern.ndim == 2
    h, w = pattern.shape
    assert h == w and h >= 3 and h % 2 == 1
    assert np.any(pattern != BLACK), "window interior must contain a visible pattern"
    assert not np.any(pattern == GREY), "grey is reserved for window borders"

    if np.array_equal(pattern, np.fliplr(pattern)):
        return "lr"
    if np.array_equal(pattern, np.flipud(pattern)):
        return "ud"
    if np.array_equal(pattern, np.rot90(pattern, 2)):
        return "rot"
    return "none"


def _palette_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    mask[h - 2 : h, w - 4 : w] = True
    return mask


def _mark_border(mask: np.ndarray, r: int, c: int, side: int) -> None:
    mask[r, c : c + side] = True
    mask[r + side - 1, c : c + side] = True
    mask[r : r + side, c] = True
    mask[r : r + side, c + side - 1] = True


def _border_is_color(g: np.ndarray, r: int, c: int, side: int, color: int) -> bool:
    return bool(
        np.all(g[r, c : c + side] == color)
        and np.all(g[r + side - 1, c : c + side] == color)
        and np.all(g[r : r + side, c] == color)
        and np.all(g[r : r + side, c + side - 1] == color)
    )


def _find_windows(g: np.ndarray) -> list[tuple[int, int, int]]:
    h, w = g.shape
    palette = _palette_mask(g.shape)
    candidates: list[tuple[int, int, int]] = []

    for side in range(5, min(h, w) + 1, 2):
        for r in range(h - side + 1):
            for c in range(w - side + 1):
                if np.any(palette[r : r + side, c : c + side]):
                    continue
                if not _border_is_color(g, r, c, side, GREY):
                    continue
                interior = g[r + 1 : r + side - 1, c + 1 : c + side - 1]
                if np.any(interior == GREY):
                    continue
                if not np.any(interior != BLACK):
                    continue
                candidates.append((r, c, side))

    covered = np.zeros(g.shape, dtype=bool)
    for r, c, side in candidates:
        _mark_border(covered, r, c, side)

    grey_outside_palette = (g == GREY) & ~palette
    assert np.array_equal(covered, grey_outside_palette), "grey pixels must belong to complete square window borders"
    assert candidates, "expected at least one grey-bordered window"
    return sorted(candidates)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey window frame by the symmetry class of the pattern inside it.

    Input structure:
    - The background is black. The bottom-right 2x4 palette has four distinct non-black, non-grey colors in its upper row.
    - The lower palette row is a four-position icon row; from left to right the columns stand for left-right mirror symmetry, up-down mirror symmetry, 180-degree rotational symmetry, and no symmetry.
    - Elsewhere are one or more disjoint square windows with one-cell grey borders and odd square interiors of side at least 3.
    - Each window interior contains a non-black, non-grey pattern and no grey pixels.

    Rule:
    - For each window, compare its full interior array with its left-right flip, up-down flip, and 180-degree rotation.
    - Classify with priority left-right mirror, then up-down mirror, then 180-degree rotation, then none.
    - Use the corresponding palette column's upper color to repaint the entire grey border of that window, leaving the interior and palette unchanged.

    Key insight:
    - The frame color is not determined by the pattern color; it is determined by the pattern's visual symmetry class and the fixed column order of the palette icons.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 7 and w >= 7

    palette_top = [int(v) for v in g[h - 2, w - 4 : w]]
    assert len(set(palette_top)) == 4, "palette top row colors must be distinct"
    assert all(v not in (BLACK, GREY) for v in palette_top), "palette colors cannot be black or grey"
    assert all(int(v) != BLACK for v in g[h - 1, w - 4 : w]), "palette icons must be visible"
    color_for_type = dict(zip(SYMMETRY_TYPES, palette_top))

    out = g.copy()
    for r, c, side in _find_windows(g):
        interior = g[r + 1 : r + side - 1, c + 1 : c + side - 1]
        color = color_for_type[_classify_pattern(interior)]
        out[r, c : c + side] = color
        out[r + side - 1, c : c + side] = color
        out[r : r + side, c] = color
        out[r : r + side, c + side - 1] = color

    return out


def _bool_class(mask: np.ndarray) -> str:
    return _classify_pattern(mask.astype(int))


def _good_mask(mask: np.ndarray) -> bool:
    count = int(mask.sum())
    return 2 <= count <= mask.size - 2


def _random_lr_mask(n: int) -> np.ndarray | None:
    for _ in range(PATTERN_ATTEMPTS):
        mask = np.zeros((n, n), dtype=bool)
        p = random.uniform(0.25, 0.55)
        for r in range(n):
            for c in range((n + 1) // 2):
                value = random.random() < p
                mask[r, c] = value
                mask[r, n - 1 - c] = value
        if _good_mask(mask) and _bool_class(mask) == "lr":
            return mask
    return None


def _random_ud_mask(n: int) -> np.ndarray | None:
    for _ in range(PATTERN_ATTEMPTS):
        mask = np.zeros((n, n), dtype=bool)
        p = random.uniform(0.25, 0.55)
        for r in range((n + 1) // 2):
            for c in range(n):
                value = random.random() < p
                mask[r, c] = value
                mask[n - 1 - r, c] = value
        if _good_mask(mask) and _bool_class(mask) == "ud":
            return mask
    return None


def _random_rot_mask(n: int) -> np.ndarray | None:
    for _ in range(PATTERN_ATTEMPTS):
        mask = np.zeros((n, n), dtype=bool)
        p = random.uniform(0.25, 0.55)
        seen: set[tuple[int, int]] = set()
        for r in range(n):
            for c in range(n):
                if (r, c) in seen:
                    continue
                rr, cc = n - 1 - r, n - 1 - c
                value = random.random() < p
                mask[r, c] = value
                mask[rr, cc] = value
                seen.add((r, c))
                seen.add((rr, cc))
        if _good_mask(mask) and _bool_class(mask) == "rot":
            return mask
    return None


def _random_none_mask(n: int) -> np.ndarray | None:
    for _ in range(PATTERN_ATTEMPTS):
        p = random.uniform(0.25, 0.55)
        mask = np.random.random((n, n)) < p
        if _good_mask(mask) and _bool_class(mask) == "none":
            return mask
    return None


def _random_mask(n: int, kind: str) -> np.ndarray | None:
    if kind == "lr":
        return _random_lr_mask(n)
    if kind == "ud":
        return _random_ud_mask(n)
    if kind == "rot":
        return _random_rot_mask(n)
    if kind == "none":
        return _random_none_mask(n)
    raise ValueError(kind)


def _reserve(mask: np.ndarray, r: int, c: int, height: int, width: int, pad: int = 1) -> None:
    r0 = max(0, r - pad)
    c0 = max(0, c - pad)
    r1 = min(mask.shape[0], r + height + pad)
    c1 = min(mask.shape[1], c + width + pad)
    mask[r0:r1, c0:c1] = True


def _paint_window(grid: np.ndarray, r: int, c: int, interior: np.ndarray, color: int) -> None:
    n = interior.shape[0]
    side = n + 2
    grid[r, c : c + side] = GREY
    grid[r + side - 1, c : c + side] = GREY
    grid[r : r + side, c] = GREY
    grid[r : r + side, c + side - 1] = GREY
    grid[r + 1 : r + side - 1, c + 1 : c + side - 1][interior] = int(color)


def _place_windows(
    grid: np.ndarray,
    items: list[tuple[str, int, np.ndarray, int]],
) -> bool:
    h, w = grid.shape
    occupied = np.zeros((h, w), dtype=bool)
    _reserve(occupied, h - 2, w - 4, 2, 4, pad=1)

    items = sorted(items, key=lambda item: item[1], reverse=True)
    for _kind, n, interior, color in items:
        side = n + 2
        placed = False
        for _ in range(PLACE_ATTEMPTS):
            r = random.randint(0, h - side)
            c = random.randint(0, w - side)
            if np.any(occupied[r : r + side, c : c + side]):
                continue
            _paint_window(grid, r, c, interior, color)
            _reserve(occupied, r, c, side, side, pad=1)
            placed = True
            break
        if not placed:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(20, 30)
    width = random.randint(20, 30)
    grid = np.zeros((height, width), dtype=int)

    palette_top = random.sample(PALETTE, 4)
    icon_pool = [color for color in PALETTE if color not in palette_top]
    palette_icons = random.sample(icon_pool, 4)
    grid[height - 2, width - 4 : width] = palette_top
    grid[height - 1, width - 4 : width] = palette_icons

    kinds = list(SYMMETRY_TYPES)
    kinds.extend(random.choice(SYMMETRY_TYPES) for _ in range(random.randint(0, 3)))
    random.shuffle(kinds)

    items: list[tuple[str, int, np.ndarray, int]] = []
    for kind in kinds:
        n = random.choice((3, 5, 7))
        mask = _random_mask(n, kind)
        if mask is None:
            return None
        border_color = palette_top[SYMMETRY_TYPES.index(kind)]
        pattern_choices = [color for color in PALETTE if color != border_color]
        pattern_color = random.choice(pattern_choices)
        items.append((kind, n, mask, pattern_color))

    if not _place_windows(grid, items):
        return None

    try:
        solved = main(grid)
    except Exception:
        return None
    if solved.shape != grid.shape or max(solved.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid symmetry-window input")
