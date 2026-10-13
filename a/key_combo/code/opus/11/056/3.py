from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
INPUT_COLORS = (1, 2, 3)
ORANGE = 7
CYAN = 8

GRID_SIDE_MIN = 10
GRID_SIDE_MAX = 15
MASK_COUNT_CHOICES = (2, 3)
BBOX_SIDE_CHOICES = (4, 5, 6)
BORDER = 1
GAP = 1
PIXELS_MIN = 8
PIXELS_MAX = 15
GENERATION_RETRIES = 600
PLACEMENT_RETRIES = 200
MASK_RETRIES = 300


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _foreground_colors(grid: np.ndarray) -> list[int]:
    return sorted(int(color) for color in np.unique(grid) if int(color) != BACKGROUND)


def _compare_with_vertical_mirror(mask: np.ndarray, color: int) -> np.ndarray:
    mirrored = np.flipud(mask)
    out = np.zeros(mask.shape, dtype=int)
    out[mask & mirrored] = color
    out[mask & ~mirrored] = ORANGE
    out[mirrored & ~mask] = CYAN
    return out


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return (
        a[2] + GAP < b[0]
        or b[2] + GAP < a[0]
        or a[3] + GAP < b[1]
        or b[3] + GAP < a[1]
    )


def _is_adjacent(mask: np.ndarray, r: int, c: int) -> bool:
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc]:
            return True
    return False


def _validate_local_mask(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if _bbox(mask) != (0, 0, h - 1, w - 1):
        return False
    pixel_count = int(mask.sum())
    if not (PIXELS_MIN <= pixel_count <= PIXELS_MAX):
        return False

    mirrored = np.flipud(mask)
    symmetric = mask & mirrored
    orange = mask & ~mirrored
    cyan = mirrored & ~mask

    if int(symmetric.sum()) < 2 or int(orange.sum()) < 1 or int(cyan.sum()) < 1:
        return False
    if np.array_equal(mask, mirrored):
        return False

    orange_rows = np.argwhere(orange)[:, 0]
    cyan_rows = np.argwhere(cyan)[:, 0]
    axis = (h - 1) / 2.0
    if orange_rows.size == 0 or cyan_rows.size == 0:
        return False
    if not np.all(orange_rows < axis):
        return False
    if not np.all(cyan_rows > axis):
        return False

    return True


def _choose_placement(mask: np.ndarray, placements: list[list[tuple[int, int]]]) -> list[tuple[int, int]] | None:
    if not placements:
        return None
    scored = []
    for cells in placements:
        adjacency = sum(1 for r, c in cells if _is_adjacent(mask, r, c))
        boundary_hits = sum(
            1 for r, c in cells if r in (0, mask.shape[0] - 1) or c in (0, mask.shape[1] - 1)
        )
        scored.append(((adjacency, boundary_hits, len(cells), random.random()), cells))
    scored.sort(key=lambda item: item[0], reverse=True)
    return random.choice([cells for _score, cells in scored[: min(6, len(scored))]])


def _make_mask(h: int, w: int) -> np.ndarray | None:
    mask = np.zeros((h, w), dtype=bool)
    half = h // 2

    sym_col = random.randrange(1, w)
    mask[0, 0] = True
    mask[0, sym_col] = True
    mask[h - 1, sym_col] = True
    if sym_col != w - 1:
        mask[random.randrange(half), w - 1] = True
    if h % 2 == 1 and random.random() < 0.5:
        mask[h // 2, random.randrange(w)] = True

    target = random.randint(max(PIXELS_MIN, int(mask.sum())), min(PIXELS_MAX, h * w))
    while int(mask.sum()) < target:
        remaining = target - int(mask.sum())
        placements: list[list[tuple[int, int]]] = []

        for r in range(half):
            mr = h - 1 - r
            for c in range(w):
                if not mask[r, c]:
                    placements.append([(r, c)])
                if remaining >= 2 and not mask[r, c] and not mask[mr, c]:
                    placements.append([(r, c), (mr, c)])

        if h % 2 == 1:
            center_r = h // 2
            for c in range(w):
                if not mask[center_r, c]:
                    placements.append([(center_r, c)])

        chosen = _choose_placement(mask, placements)
        if chosen is None:
            break
        for r, c in chosen:
            mask[r, c] = True

    return mask if _validate_local_mask(mask) else None


def _place_bboxes(
    rows: int, cols: int, sizes: list[tuple[int, int]]
) -> list[tuple[int, int, int, int]] | None:
    for _ in range(PLACEMENT_RETRIES):
        placed: list[tuple[int, int, int, int]] = []
        order = list(range(len(sizes)))
        random.shuffle(order)
        slots: list[tuple[int, int, int, int] | None] = [None] * len(sizes)

        ok = True
        for idx in order:
            h, w = sizes[idx]
            candidates = [
                (r, c, r + h - 1, c + w - 1)
                for r in range(BORDER, rows - h - BORDER + 1)
                for c in range(BORDER, cols - w - BORDER + 1)
            ]
            random.shuffle(candidates)
            box = next(
                (cand for cand in candidates if all(_boxes_separated(cand, other) for other in placed)),
                None,
            )
            if box is None:
                ok = False
                break
            placed.append(box)
            slots[idx] = box

        if ok:
            return [box for box in slots if box is not None]

    return None


def _expected_patch(mask: np.ndarray, color: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mirrored = np.flipud(mask)
    symmetric = mask & mirrored
    orange = mask & ~mirrored
    cyan = mirrored & ~mask
    patch = np.zeros(mask.shape, dtype=int)
    patch[symmetric] = color
    patch[orange] = ORANGE
    patch[cyan] = CYAN
    return patch, symmetric, orange, cyan


def _crop_with_border(grid: np.ndarray, border: int = 1) -> np.ndarray:
    box = _bbox(grid != BACKGROUND)
    assert box is not None
    r0, c0, r1, c1 = box
    r0 = max(0, r0 - border)
    c0 = max(0, c0 - border)
    r1 = min(grid.shape[0] - 1, r1 + border)
    c1 = min(grid.shape[1] - 1, c1 + border)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _validate_scene(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    if max(grid.shape) > 30:
        return False

    colors = _foreground_colors(grid)
    if len(colors) not in MASK_COUNT_CHOICES:
        return False
    if not set(colors).issubset(INPUT_COLORS):
        return False

    bboxes: list[tuple[int, int, int, int]] = []
    out = main(grid)

    for color in colors:
        mask = grid == color
        box = _bbox(mask)
        if box is None:
            return False
        r0, c0, r1, c1 = box
        if not (4 <= (r1 - r0 + 1) <= 6 and 4 <= (c1 - c0 + 1) <= 6):
            return False
        for other in bboxes:
            if not _boxes_separated(box, other):
                return False
        bboxes.append(box)

        local = mask[r0 : r1 + 1, c0 : c1 + 1]
        if not _validate_local_mask(local):
            return False
        expected, symmetric, orange, cyan = _expected_patch(local, color)
        if not np.array_equal(out[r0 : r1 + 1, c0 : c1 + 1], expected):
            return False
        if int(symmetric.sum()) < 2 or int(orange.sum()) < 1 or int(cyan.sum()) < 1:
            return False

    covered = np.zeros_like(grid, dtype=bool)
    for r0, c0, r1, c1 in bboxes:
        covered[r0 : r1 + 1, c0 : c1 + 1] = True
    if np.any(out[~covered] != BACKGROUND):
        return False

    return True


def main(grid: np.ndarray) -> np.ndarray:
    """Mark each color's self-mirror asymmetry inside its own bounding box.

    Input structure:
    - The grid has a black background with 2 or 3 non-overlapping monochrome color masks.
    - Each foreground color is treated as one mask even if it splits into multiple components.
    - Every color occupies its own compact bounding box and is not perfectly vertically symmetric.

    Rule:
    - For each non-background color independently, find the bounding box of all pixels of that color.
    - Mirror that color mask vertically inside its own box, mapping `(r, c)` to `(r0 + r1 - r, c)`.
    - Compare the original mask with its mirrored version inside the box:
      overlapping pixels keep the original color, original-only pixels become orange, and mirrored-only pixels become cyan.
    - Pixels outside every foreground bounding box stay black.

    Key insight:
    - The answer is not the mirrored shape itself; it is a direction-aware comparison map between the original mask and its own per-color vertical reflection.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = np.zeros_like(g)
    for color in _foreground_colors(g):
        mask = g == color
        box = _bbox(mask)
        assert box is not None
        r0, c0, r1, c1 = box
        local = mask[r0 : r1 + 1, c0 : c1 + 1]
        patch = _compare_with_vertical_mirror(local, color)
        region = out[r0 : r1 + 1, c0 : c1 + 1]
        assert not np.any((region != BACKGROUND) & (patch != BACKGROUND))
        region[patch != BACKGROUND] = patch[patch != BACKGROUND]

    return out


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        count = random.choice(MASK_COUNT_CHOICES)
        colors = random.sample(list(INPUT_COLORS), count)
        sizes = [(random.choice(BBOX_SIDE_CHOICES), random.choice(BBOX_SIDE_CHOICES)) for _ in range(count)]
        grid = np.zeros((15, 15), dtype=int)
        slot_origins = [(1, 1), (1, 8), (8, 1), (8, 8)]
        chosen_slots = random.sample(slot_origins, count)

        ok = True
        for color, (h, w), (slot_r, slot_c) in zip(colors, sizes, chosen_slots):
            mask = _make_mask(h, w)
            if mask is None:
                ok = False
                break
            r0 = slot_r + random.randint(0, 6 - h)
            c0 = slot_c + random.randint(0, 6 - w)
            r1 = r0 + h - 1
            c1 = c0 + w - 1
            grid[r0 : r1 + 1, c0 : c1 + 1][mask] = color

        if not ok:
            continue

        cropped = _crop_with_border(grid, border=1)
        if _validate_scene(cropped):
            return cropped

    raise RuntimeError("failed to generate a valid input grid")
