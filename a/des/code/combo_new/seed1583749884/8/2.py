import random

import numpy as np


GEN_RETRIES = 400
SHAPE_RETRIES = 120
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _background_color(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(values).argmax())


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _mask_signature(mask: np.ndarray) -> frozenset[tuple[int, int]]:
    return frozenset((int(r), int(c)) for r, c in np.argwhere(mask))


def _horizontal_signature(coords: np.ndarray) -> frozenset[tuple[int, int]]:
    left = int(coords[:, 1].min())
    return frozenset((int(r), int(c - left)) for r, c in coords)


def _shape_groups(grid: np.ndarray) -> tuple[int, dict[int, np.ndarray], dict[frozenset[tuple[int, int]], list[int]]]:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    coords_by_color = {color: np.argwhere(g == color) for color in colors}

    groups: dict[frozenset[tuple[int, int]], list[int]] = {}
    for color, coords in coords_by_color.items():
        groups.setdefault(_horizontal_signature(coords), []).append(color)
    return bg, coords_by_color, groups


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge horizontally translated duplicate color layers back onto their leftmost original position.

    Input structure:
    - The background is black.
    - There are 3-4 foreground colors, each forming one visible shape.
    - One same-row shape appears in 2-3 colors with exactly the same per-row mask, only shifted horizontally; the leftmost instance is the original and the others are copies.
    - One extra foreground color is an unrelated distractor shape.

    Rule:
    - Compare foreground colors after removing only their column offset, so horizontal copies share one signature while the distractor does not.
    - In the repeated-shape family, keep the leftmost color layer as the original position.
    - Remove every other shifted copy from its old location; the merged result therefore keeps just the original-colored shape at the original columns.
    - Leave the unrelated distractor unchanged.

    Key insight:
    - The repeated colors are not different objects: they are parallel horizontal layers of the same mask, so solving means grouping equal row-patterns and collapsing them onto the leftmost anchor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, coords_by_color, groups = _shape_groups(g)
    family_groups = [colors for colors in groups.values() if len(colors) >= 2]
    assert len(family_groups) == 1, "expected one duplicated-shape family"

    family = family_groups[0]
    original = min(family, key=lambda color: int(coords_by_color[color][:, 1].min()))

    out = g.copy()
    for color in family:
        if color != original:
            out[g == color] = bg
    return out


def _frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    options: set[tuple[int, int]] = set()
    for r, c in np.argwhere(mask):
        for dr, dc in NEIGHBORS4:
            nr = int(r) + dr
            nc = int(c) + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                options.add((nr, nc))
    return list(options)


def _random_connected_mask() -> np.ndarray | None:
    for _ in range(SHAPE_RETRIES):
        h = random.randint(2, 5)
        w = random.randint(2, 5)
        target = random.randint(3, min(h * w, 9))

        mask = np.zeros((h, w), dtype=bool)
        mask[random.randrange(h), random.randrange(w)] = True
        while int(mask.sum()) < target:
            frontier = _frontier(mask)
            if not frontier:
                break
            nr, nc = random.choice(frontier)
            mask[nr, nc] = True

        if int(mask.sum()) != target:
            continue

        mask = _crop_mask(mask)
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        return mask
    return None


def _mask_cells(mask: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    return [(top + int(r), left + int(c)) for r, c in np.argwhere(mask)]


def _can_place(mask: np.ndarray, top: int, left: int, occupied: np.ndarray, pad: int = 1) -> bool:
    h, w = occupied.shape
    for r, c in _mask_cells(mask, top, left):
        if not (0 <= r < h and 0 <= c < w):
            return False
        r0 = max(0, r - pad)
        r1 = min(h, r + pad + 1)
        c0 = max(0, c - pad)
        c1 = min(w, c + pad + 1)
        if bool(occupied[r0:r1, c0:c1].any()):
            return False
    return True


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int, occupied: np.ndarray) -> None:
    for r, c in _mask_cells(mask, top, left):
        grid[r, c] = color
        occupied[r, c] = True


def _random_distractor(family_mask: np.ndarray) -> np.ndarray | None:
    family_sig = _mask_signature(family_mask)
    for _ in range(SHAPE_RETRIES):
        mask = _random_connected_mask()
        if mask is None:
            continue
        if _mask_signature(mask) != family_sig:
            return mask
    return None


def _valid_generated_grid(grid: np.ndarray, family_size: int) -> bool:
    try:
        bg, coords_by_color, groups = _shape_groups(grid)
        colors = [color for color in coords_by_color if color != bg]
        if len(colors) != family_size + 1:
            return False
        family_groups = [group for group in groups.values() if len(group) >= 2]
        if len(family_groups) != 1 or len(family_groups[0]) != family_size:
            return False
        out = main(grid)
    except AssertionError:
        return False

    return bool(np.any(out != grid))


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        family_size = random.choice([2, 3])
        family_mask = _random_connected_mask()
        if family_mask is None:
            continue

        distractor_mask = _random_distractor(family_mask)
        if distractor_mask is None:
            continue

        fh, fw = family_mask.shape
        dh, dw = distractor_mask.shape

        gaps = [random.randint(1, 4) for _ in range(family_size - 1)]
        offsets = [0]
        for gap in gaps:
            offsets.append(offsets[-1] + fw + gap)
        family_span = offsets[-1] + fw

        min_h = max(fh, dh) + 4
        min_w = max(family_span, dw) + 4
        if min_h > 30 or min_w > 30:
            continue

        height = random.randint(min_h, min(30, min_h + 8))
        width = random.randint(min_w, min(30, min_w + 8))

        top = random.randint(0, height - fh)
        left = random.randint(0, width - family_span)

        colors = random.sample(range(1, 10), family_size + 1)
        family_colors = colors[:family_size]
        distractor_color = colors[-1]

        grid = np.zeros((height, width), dtype=int)
        occupied = np.zeros((height, width), dtype=bool)

        for color, offset in zip(family_colors, offsets):
            _paint_mask(grid, family_mask, top, left + offset, color, occupied)

        candidates = [
            (r, c)
            for r in range(height - dh + 1)
            for c in range(width - dw + 1)
            if _can_place(distractor_mask, r, c, occupied, pad=1)
        ]
        if not candidates:
            continue

        distractor_top, distractor_left = random.choice(candidates)
        _paint_mask(grid, distractor_mask, distractor_top, distractor_left, distractor_color, occupied)

        if _valid_generated_grid(grid, family_size):
            return grid

    raise RuntimeError("failed to generate a valid input for task 2")
