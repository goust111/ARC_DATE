from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected nonempty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_single_component(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    seen = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for rr, cc in _neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                queue.append((rr, cc))
    return len(seen) == int(mask.sum())


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(mask)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _path(a: tuple[int, int], b: tuple[int, int], vertical_first: bool) -> list[tuple[int, int]]:
    r, c = map(int, a)
    tr, tc = map(int, b)
    pts = [(r, c)]
    if vertical_first:
        while r != tr:
            r += 1 if tr > r else -1
            pts.append((r, c))
        while c != tc:
            c += 1 if tc > c else -1
            pts.append((r, c))
    else:
        while c != tc:
            c += 1 if tc > c else -1
            pts.append((r, c))
        while r != tr:
            r += 1 if tr > r else -1
            pts.append((r, c))
    return pts


def _paint_path(mask: np.ndarray, pts: list[tuple[int, int]]) -> None:
    for r, c in pts:
        mask[int(r), int(c)] = True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align all scattered monochrome masks at their top-left corners and keep only uniquely owned cells.

    Input structure:
    - The dominant color is the background.
    - Each non-background color forms one separated 4-connected monochrome mask.
    - Cropping every mask to its own bounding box yields several top-left-alignable mask windows of varying sizes.

    Rule:
    - Crop each colored mask to its bounding box.
    - Pad all cropped masks into the common window given by the maximum observed height and width, aligned at the top-left.
    - In the common window, keep a cell only when exactly one cropped mask covers it, and color it with that mask's color; cells covered by zero or at least two masks become background.

    Key insight:
    - This looks like partial-template completion, but overlap is destructive: shared cells vanish instead of merging, so the output is the multi-mask "unique owner only" stamp.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    assert len(colors) >= 2, "expected multiple foreground masks"

    locals_by_color: list[tuple[int, np.ndarray]] = []
    for color in colors:
        mask = g == color
        assert np.any(mask), "foreground color missing"
        assert _is_single_component(mask), "each foreground color must form exactly one 4-connected mask"
        local = _crop_mask(mask)
        assert _is_single_component(local), "cropped mask must remain connected"
        locals_by_color.append((color, local))

    out_h = max(local.shape[0] for _, local in locals_by_color)
    out_w = max(local.shape[1] for _, local in locals_by_color)
    stack = np.zeros((len(locals_by_color), out_h, out_w), dtype=bool)
    for i, (_color, local) in enumerate(locals_by_color):
        h, w = local.shape
        stack[i, :h, :w] = local

    count = stack.sum(axis=0)
    out = np.full((out_h, out_w), bg, dtype=int)
    for i, (color, _local) in enumerate(locals_by_color):
        out[stack[i] & (count == 1)] = int(color)
    return out


GRID_HEIGHT_RANGE = (18, 25)
GRID_WIDTH_RANGE = (18, 25)
WINDOW_HEIGHT_RANGE = (6, 9)
WINDOW_WIDTH_RANGE = (6, 9)
MASK_COUNT_CHOICES = (3, 4)
GAP = 2
GEN_RETRIES = 400
PALETTE = tuple(range(10))


def _extra_mask_base() -> list[tuple[int, int]]:
    return [(0, 4), (1, 4), (2, 4), (2, 3), (2, 2), (2, 1), (2, 0)]


def _build_local_masks(rng: random.Random) -> tuple[list[np.ndarray], tuple[int, int], np.ndarray] | None:
    count = rng.choice(MASK_COUNT_CHOICES)
    out_h = rng.randint(*WINDOW_HEIGHT_RANGE)
    out_w = rng.randint(*WINDOW_WIDTH_RANGE)
    if out_h < 6 or out_w < 6:
        return None

    height_choices = list(range(max(5, out_h - 2), out_h + 1))
    width_choices = list(range(max(5, out_w - 2), out_w + 1))
    pairs = [(h, w) for h in height_choices for w in width_choices]
    if len(pairs) < count:
        return None
    extents = rng.sample(pairs, count - 1) + [(out_h, out_w)]
    rng.shuffle(extents)

    masks = [np.zeros((h, w), dtype=bool) for h, w in extents]

    triple_path = rng.choice(
        [
            [(0, 0), (0, 1), (1, 1)],
            [(0, 0), (1, 0), (1, 1)],
        ]
    )
    pair_path = [(1, 1), (1, 2), (1, 3)]

    for idx in range(3):
        _paint_path(masks[idx], triple_path)
    for idx in (0, 1):
        _paint_path(masks[idx], pair_path)

    goals = [(h - 1, w - 1) for h, w in extents]
    _paint_path(masks[0], _path((1, 3), goals[0], vertical_first=True))
    _paint_path(masks[1], _path((1, 3), goals[1], vertical_first=False))
    _paint_path(masks[2], _path((1, 1), goals[2], vertical_first=True))

    if count == 4:
        base = _extra_mask_base()
        if extents[3][0] <= 2 or extents[3][1] <= 4:
            return None
        _paint_path(masks[3], base)
        _paint_path(masks[3], _path((2, 2), goals[3], vertical_first=False))

    for i in range(count):
        local = masks[i]
        if not _is_single_component(local):
            return None
        if local[0].sum() == 0 or local[:, 0].sum() == 0:
            return None
        if local[-1].sum() == 0 or local[:, -1].sum() == 0:
            return None
        if local.shape != _crop_mask(local).shape:
            return None

    if len({m.shape[0] for m in masks}) == 1:
        return None
    if len({m.shape[1] for m in masks}) == 1:
        return None

    stack = np.zeros((count, out_h, out_w), dtype=bool)
    for i, local in enumerate(masks):
        h, w = local.shape
        stack[i, :h, :w] = local
    cover = stack.sum(axis=0)
    if not np.any(cover == 2):
        return None
    if not np.any(cover == 3):
        return None
    if np.any(cover > 3):
        return None
    for i in range(count):
        if not np.any(stack[i] & (cover == 1)):
            return None

    return masks, (out_h, out_w), cover


def _compute_expected(masks: list[np.ndarray], colors: list[int], bg: int, shape: tuple[int, int]) -> np.ndarray:
    out_h, out_w = map(int, shape)
    stack = np.zeros((len(masks), out_h, out_w), dtype=bool)
    for i, local in enumerate(masks):
        h, w = local.shape
        stack[i, :h, :w] = local
    cover = stack.sum(axis=0)
    out = np.full((out_h, out_w), int(bg), dtype=int)
    for i, color in enumerate(colors):
        out[stack[i] & (cover == 1)] = int(color)
    return out


def _separated(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int], gap: int = GAP) -> bool:
    return (
        box_a[2] + gap < box_b[0]
        or box_b[2] + gap < box_a[0]
        or box_a[3] + gap < box_b[1]
        or box_b[3] + gap < box_a[1]
    )


def _place_masks(
    rng: random.Random,
    locals_by_color: list[tuple[int, np.ndarray]],
    grid_h: int,
    grid_w: int,
    bg: int,
) -> np.ndarray | None:
    grid = np.full((grid_h, grid_w), int(bg), dtype=int)
    boxes: list[tuple[int, int, int, int]] = []

    items = locals_by_color[:]
    rng.shuffle(items)
    for color, local in items:
        h, w = local.shape
        candidates: list[tuple[int, int]] = []
        for r0 in range(grid_h - h + 1):
            for c0 in range(grid_w - w + 1):
                box = (r0, c0, r0 + h - 1, c0 + w - 1)
                if all(_separated(box, prev) for prev in boxes):
                    candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = rng.choice(candidates)
        patch = grid[r0 : r0 + h, c0 : c0 + w]
        patch[local] = int(color)
        boxes.append((r0, c0, r0 + h - 1, c0 + w - 1))

    return grid


def _valid_input(grid: np.ndarray, expected: np.ndarray, cover: np.ndarray, mask_count: int, bg: int) -> bool:
    if _get_bg(grid) != bg:
        return False
    if Counter(map(int, grid.ravel())).most_common(1)[0][1] <= grid.size // 2:
        return False

    colors = [int(c) for c in np.unique(grid) if int(c) != bg]
    if len(colors) != mask_count:
        return False
    for color in colors:
        if not _is_single_component(grid == color):
            return False

    if not np.any(cover == 2) or not np.any(cover == 3):
        return False
    if np.unique(expected).size <= 2:
        return False

    try:
        solved = main(grid)
    except AssertionError:
        return False
    return np.array_equal(solved, expected)


def _generate_attempt() -> np.ndarray | None:
    rng = random.Random()
    built = _build_local_masks(rng)
    if built is None:
        return None
    masks, out_shape, cover = built
    count = len(masks)

    bg = rng.choice(PALETTE)
    colors = rng.sample([c for c in PALETTE if c != bg], count)
    expected = _compute_expected(masks, colors, bg, out_shape)

    grid_h = rng.randint(*GRID_HEIGHT_RANGE)
    grid_w = rng.randint(*GRID_WIDTH_RANGE)
    if grid_h < max(m.shape[0] for m in masks) or grid_w < max(m.shape[1] for m in masks):
        return None

    inp = _place_masks(rng, list(zip(colors, masks)), grid_h, grid_w, bg)
    if inp is None:
        return None

    if not _valid_input(inp, expected, cover, count, bg):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid unique-ownership stamp input")
