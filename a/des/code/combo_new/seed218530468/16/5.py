from __future__ import annotations

import random

import numpy as np


NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
NEIGHBORS8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
)

GRID_HEIGHT_RANGE = (8, 24)
GRID_WIDTH_RANGE = (8, 24)
OBJECT_COUNT_CHOICES = (1, 2, 3, 4)
MAX_SHAPE_HEIGHT = 8
MAX_SHAPE_WIDTH = 9
MIN_OBJECT_FRACTION = 0.08
MAX_OBJECT_FRACTION = 0.35
PLACEMENT_TRIES = 80
GENERATION_RETRIES = 400


def _striped_background(height: int, width: int, even_color: int, odd_color: int) -> np.ndarray:
    bg = np.empty((height, width), dtype=int)
    bg[0::2, :] = even_color
    bg[1::2, :] = odd_color
    return bg


def _foreground_candidate(
    g: np.ndarray, fg: int
) -> list[tuple[tuple[int, int, int, int], int, int]]:
    h, _w = g.shape
    row_colors: list[int | None] = []
    for r in range(h):
        visible = {int(v) for v in g[r, :] if int(v) != fg}
        if len(visible) > 1:
            return []
        row_colors.append(next(iter(visible)) if visible else None)

    stripes = tuple(sorted({v for v in row_colors if v is not None}))
    if len(stripes) != 2:
        return []

    visible_rows = sum(color is not None for color in row_colors)
    visible_parities = int(any(row_colors[r] is not None for r in range(0, h, 2))) + int(
        any(row_colors[r] is not None for r in range(1, h, 2))
    )
    fg_rows = np.flatnonzero(np.any(g == fg, axis=1))
    fg_parities = len({int(r % 2) for r in fg_rows.tolist()})
    visible_cells = int(np.count_nonzero(g != fg))
    score = (visible_rows, visible_parities, fg_parities, visible_cells)

    solutions: list[tuple[tuple[int, int, int, int], int, int]] = []
    for even_color, odd_color in ((stripes[0], stripes[1]), (stripes[1], stripes[0])):
        if all(
            color is None or color == (even_color if r % 2 == 0 else odd_color)
            for r, color in enumerate(row_colors)
        ):
            solutions.append((score, even_color, odd_color))
    return solutions


def _infer_scene(g: np.ndarray) -> tuple[int, int, int]:
    colors = [int(c) for c in np.unique(g)]
    assert len(colors) >= 3, "expected two stripe colors and one foreground color"

    candidates: list[tuple[tuple[int, int, int, int], int, int, int]] = []
    for fg in colors:
        for score, even_color, odd_color in _foreground_candidate(g, fg):
            candidates.append((score, fg, even_color, odd_color))

    assert candidates, "could not infer stripe colors and foreground color"
    best_score = max(score for score, _fg, _even, _odd in candidates)
    best = [(fg, even_color, odd_color) for score, fg, even_color, odd_color in candidates if score == best_score]
    assert len(best) == 1, "scene is ambiguous under the stripe-occlusion model"
    return best[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove the foreground objects and reveal them as local stripe inversions.

    Input structure:
    - The whole grid is covered by width-1 horizontal stripes using exactly two alternating colors.
    - One additional color draws one or more 4-connected foreground objects on top of those stripes.
    - Away from the objects, every row shows only its native stripe color.

    Rule:
    - Infer which color is the overlaid foreground and which two colors form the horizontal stripe background.
    - Reconstruct the full alternating stripe pattern for the entire canvas.
    - At every cell that was occupied by the foreground, swap that row's stripe color to the other stripe color; leave all other cells unchanged.

    Key insight:
    - The object color disappears completely; the output keeps only the original stripe field, except that the hidden object area is encoded by locally flipping the stripe colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    fg, even_color, odd_color = _infer_scene(g)
    base = _striped_background(g.shape[0], g.shape[1], even_color, odd_color)
    swapped = np.where(base == even_color, odd_color, even_color)
    out = base.copy()
    out[g == fg] = swapped[g == fg]
    return out


def _random_blob(height: int, width: int, target: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    start = (random.randrange(height), random.randrange(width))
    mask[start] = True
    frontier = {start}

    while int(mask.sum()) < target and frontier:
        r, c = random.choice(tuple(frontier))
        neighbors = []
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                neighbors.append((nr, nc))
        if not neighbors:
            frontier.remove((r, c))
            continue
        nr, nc = random.choice(neighbors)
        mask[nr, nc] = True
        frontier.add((nr, nc))

    return mask


def _shape_rect(max_h: int, max_w: int) -> np.ndarray:
    height = random.randint(2, min(6, max_h))
    width = random.randint(2, min(7, max_w))
    return np.ones((height, width), dtype=bool)


def _shape_l(max_h: int, max_w: int) -> np.ndarray:
    height = random.randint(3, min(7, max_h))
    width = random.randint(3, min(7, max_w))
    mask = np.zeros((height, width), dtype=bool)
    orientation = random.randrange(4)
    if orientation == 0:
        mask[:, 0] = True
        mask[-1, :] = True
    elif orientation == 1:
        mask[:, -1] = True
        mask[-1, :] = True
    elif orientation == 2:
        mask[:, 0] = True
        mask[0, :] = True
    else:
        mask[:, -1] = True
        mask[0, :] = True
    return mask


def _shape_plus(max_h: int, max_w: int) -> np.ndarray:
    max_h = min(max_h, 7)
    max_w = min(max_w, 7)
    height_choices = [v for v in range(3, max_h + 1, 2)]
    width_choices = [v for v in range(3, max_w + 1, 2)]
    height = random.choice(height_choices)
    width = random.choice(width_choices)
    mask = np.zeros((height, width), dtype=bool)
    mask[height // 2, :] = True
    mask[:, width // 2] = True
    return mask


def _shape_blob(max_h: int, max_w: int) -> np.ndarray:
    height = random.randint(2, min(7, max_h))
    width = random.randint(2, min(8, max_w))
    min_area = max(4, height + width - 3)
    max_area = max(min_area, min(height * width - 1, max(height * width // 2, min_area)))
    target = random.randint(min_area, max_area)
    return _random_blob(height, width, target)


def _sample_shape(grid_h: int, grid_w: int) -> np.ndarray:
    max_h = min(MAX_SHAPE_HEIGHT, grid_h)
    max_w = min(MAX_SHAPE_WIDTH, grid_w)
    kind = random.choice(("rect", "l", "plus", "blob"))
    if kind == "rect":
        return _shape_rect(max_h, max_w)
    if kind == "l":
        return _shape_l(max_h, max_w)
    if kind == "plus":
        return _shape_plus(max_h, max_w)
    return _shape_blob(max_h, max_w)


def _fits(mask: np.ndarray, blocked: np.ndarray, top: int, left: int) -> bool:
    coords = np.argwhere(mask)
    rows = coords[:, 0] + top
    cols = coords[:, 1] + left
    return not bool(blocked[rows, cols].any())


def _paint_object(scene_mask: np.ndarray, blocked: np.ndarray, mask: np.ndarray, top: int, left: int) -> None:
    for rr, cc in np.argwhere(mask):
        r = int(top + rr)
        c = int(left + cc)
        scene_mask[r, c] = True
        for dr, dc in NEIGHBORS8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < scene_mask.shape[0] and 0 <= nc < scene_mask.shape[1]:
                blocked[nr, nc] = True


def _sample_scene_mask(height: int, width: int) -> np.ndarray | None:
    scene_mask = np.zeros((height, width), dtype=bool)
    blocked = np.zeros((height, width), dtype=bool)
    target_objects = random.choice(OBJECT_COUNT_CHOICES)

    placed = 0
    for _ in range(target_objects):
        shape = _sample_shape(height, width)
        sh, sw = shape.shape
        positions = [
            (top, left)
            for top in range(height - sh + 1)
            for left in range(width - sw + 1)
            if _fits(shape, blocked, top, left)
        ]
        if not positions:
            continue
        top, left = random.choice(positions)
        _paint_object(scene_mask, blocked, shape, top, left)
        placed += 1

    if placed == 0:
        return None
    return scene_mask


def _scene_is_clear(mask: np.ndarray) -> bool:
    height, width = mask.shape
    area = int(mask.sum())
    if area == 0:
        return False
    if area < int(np.ceil(MIN_OBJECT_FRACTION * height * width)):
        return False
    if area > int(np.floor(MAX_OBJECT_FRACTION * height * width)):
        return False
    if bool(np.any(mask.all(axis=1))):
        return False
    if len({int(r % 2) for r in np.flatnonzero(mask.any(axis=1)).tolist()}) < 2:
        return False
    if not any((~mask[r, :]).any() for r in range(0, height, 2)):
        return False
    if not any((~mask[r, :]).any() for r in range(1, height, 2)):
        return False
    # Avoid scenes where every row of one parity contains only a single object pixel.
    if min(int(mask[0::2, :].sum()), int(mask[1::2, :].sum())) < 2:
        return False
    return True


def _expected_output(
    height: int, width: int, even_color: int, odd_color: int, object_mask: np.ndarray
) -> np.ndarray:
    base = _striped_background(height, width, even_color, odd_color)
    swapped = np.where(base == even_color, odd_color, even_color)
    out = base.copy()
    out[object_mask] = swapped[object_mask]
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    even_color, odd_color, object_color = random.sample(range(10), 3)

    for _ in range(PLACEMENT_TRIES):
        object_mask = _sample_scene_mask(height, width)
        if object_mask is None or not _scene_is_clear(object_mask):
            continue

        grid = _striped_background(height, width, even_color, odd_color)
        grid[object_mask] = object_color
        expected = _expected_output(height, width, even_color, odd_color, object_mask)
        try:
            if np.array_equal(main(grid), expected):
                return grid
        except Exception:
            continue
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid stripe-imprint scene after {GENERATION_RETRIES} attempts")
