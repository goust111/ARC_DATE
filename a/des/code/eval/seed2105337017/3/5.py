import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
ORANGE = 7
FRAGMENT_COLORS = (1, 2, 3, 4, 6, 8, 9)

LAYOUT_CHOICES = ((2, 2), (2, 3), (3, 2))
CELL_SIZE_MIN = 6
CELL_SIZE_MAX = 8
OUTER_BORDER_PROB = 0.5

GEN_ATTEMPTS = 400
SHAPE_ATTEMPTS = 120
FRAGMENT_ATTEMPTS = 120

ALL_RECIPES = tuple((rot, flip) for rot in range(4) for flip in (False, True))


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False

    h, w = mask.shape
    start_r, start_c = map(int, coords[0])
    seen = np.zeros_like(mask, dtype=bool)
    seen[start_r, start_c] = True
    queue = deque([(start_r, start_c)])
    reached = 0

    while queue:
        r, c = queue.popleft()
        reached += 1
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                queue.append((nr, nc))

    return reached == int(mask.sum())


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.nonzero(mask)
    assert len(rows) > 0, "mask must be non-empty"
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(mask)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _apply_recipe(mask: np.ndarray, recipe: tuple[int, bool]) -> np.ndarray:
    rot, flip = recipe
    out = np.rot90(mask, rot)
    if flip:
        out = np.fliplr(out)
    return out.copy()


def _dihedral_masks(mask: np.ndarray) -> list[np.ndarray]:
    variants: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}
    for recipe in ALL_RECIPES:
        candidate = _apply_recipe(mask, recipe)
        key = (candidate.shape, candidate.tobytes())
        variants.setdefault(key, candidate)
    return list(variants.values())


def _spans_from_lines(lines: list[int], length: int) -> list[tuple[int, int]]:
    bounds = [-1] + sorted(set(int(x) for x in lines)) + [int(length)]
    spans: list[tuple[int, int]] = []
    for a, b in zip(bounds, bounds[1:]):
        start = int(a) + 1
        end = int(b) - 1
        if start <= end:
            spans.append((start, end))
    return spans


def _extract_cells(grid: np.ndarray) -> tuple[list[dict], int, int]:
    g = np.asarray(grid, dtype=int)
    wall_rows = [r for r in range(g.shape[0]) if bool(np.all(g[r, :] == GREY))]
    wall_cols = [c for c in range(g.shape[1]) if bool(np.all(g[:, c] == GREY))]

    row_spans = _spans_from_lines(wall_rows, g.shape[0])
    col_spans = _spans_from_lines(wall_cols, g.shape[1])
    assert row_spans and col_spans, "expected grey divider lines"

    cell_heights = {int(r1 - r0 + 1) for r0, r1 in row_spans}
    cell_widths = {int(c1 - c0 + 1) for c0, c1 in col_spans}
    assert len(cell_heights) == 1 and len(cell_widths) == 1, "cells must be equally sized"

    cells: list[dict] = []
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            colors = [int(v) for v in np.unique(patch) if int(v) not in (BG, GREY)]
            assert len(colors) == 1, "each cell must contain exactly one fragment color"
            color = colors[0]
            assert bool(np.all(np.isin(patch, (BG, color)))), "cell contents must be fragment or background"

            mask = patch == color
            assert _is_connected(mask), "fragment must stay 4-connected"
            cells.append(
                {
                    "r0": int(r0),
                    "r1": int(r1),
                    "c0": int(c0),
                    "c1": int(c1),
                    "color": color,
                    "mask": mask,
                    "area": int(mask.sum()),
                }
            )

    assert len(cells) >= 2, "expected multiple cells"
    return cells, next(iter(cell_heights)), next(iter(cell_widths))


def _completion_fill(reference_mask: np.ndarray, current_mask: np.ndarray, cell_shape: tuple[int, int]) -> np.ndarray:
    cell_h, cell_w = map(int, cell_shape)
    fills: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}

    for oriented in _dihedral_masks(reference_mask):
        oh, ow = oriented.shape
        if oh > cell_h or ow > cell_w:
            continue
        for top in range(cell_h - oh + 1):
            for left in range(cell_w - ow + 1):
                full = np.zeros((cell_h, cell_w), dtype=bool)
                full[top : top + oh, left : left + ow] = oriented
                if bool(np.all(~current_mask | full)):
                    missing = full & ~current_mask
                    key = (missing.shape, missing.tobytes())
                    fills.setdefault(key, missing.copy())

    assert fills, "fragment must match a transformed reference piece"
    assert len(fills) == 1, "fragment completion must be unambiguous"
    return next(iter(fills.values()))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the hidden prototype from the unique complete fragment, then fill
    each incomplete fragment's missing cells with orange.
    """
    g = np.asarray(grid, dtype=int)
    cells, cell_h, cell_w = _extract_cells(g)

    max_area = max(int(cell["area"]) for cell in cells)
    reference_indices = [idx for idx, cell in enumerate(cells) if int(cell["area"]) == max_area]
    assert len(reference_indices) == 1, "largest fragment must be unique"
    reference_idx = reference_indices[0]
    reference_mask = _crop_mask(cells[reference_idx]["mask"])

    out = g.copy()
    for idx, cell in enumerate(cells):
        if idx == reference_idx:
            continue
        missing = _completion_fill(reference_mask, cell["mask"], (cell_h, cell_w))
        patch = out[cell["r0"] : cell["r1"] + 1, cell["c0"] : cell["c1"] + 1]
        patch[missing] = ORANGE

    return out


def _grow_connected_mask(height: int, width: int, area: int) -> np.ndarray | None:
    mask = np.zeros((height, width), dtype=bool)
    start = (random.randrange(height), random.randrange(width))
    mask[start] = True
    frontier = set()

    def add_frontier(r: int, c: int) -> None:
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                frontier.add((nr, nc))

    add_frontier(*start)
    while int(mask.sum()) < area:
        if not frontier:
            return None
        r, c = random.choice(tuple(frontier))
        frontier.remove((r, c))
        if mask[r, c]:
            continue
        mask[r, c] = True
        add_frontier(r, c)

    return mask


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(
        np.any(mask[0, :])
        and np.any(mask[-1, :])
        and np.any(mask[:, 0])
        and np.any(mask[:, -1])
    )


def _generate_prototype(cell_h: int, cell_w: int) -> np.ndarray | None:
    max_dim = min(cell_h, cell_w) - 2
    if max_dim < 3:
        return None

    for _ in range(SHAPE_ATTEMPTS):
        box_h = random.randint(3, max_dim)
        box_w = random.randint(3, max_dim)
        area_min = max(5, (box_h * box_w + 2) // 3)
        area_max = min(box_h * box_w - 1, (box_h * box_w * 3) // 4)
        if area_min > area_max:
            continue

        area = random.randint(area_min, area_max)
        mask = _grow_connected_mask(box_h, box_w, area)
        if mask is None:
            continue
        if not _touches_all_sides(mask):
            continue
        if len(_dihedral_masks(mask)) != 8:
            continue
        return mask

    return None


def _embed_mask(mask: np.ndarray, cell_h: int, cell_w: int, top: int, left: int) -> np.ndarray:
    out = np.zeros((cell_h, cell_w), dtype=bool)
    h, w = mask.shape
    out[top : top + h, left : left + w] = mask
    return out


def _random_full_fragment(mask: np.ndarray, cell_h: int, cell_w: int) -> np.ndarray | None:
    h, w = mask.shape
    if h > cell_h - 2 or w > cell_w - 2:
        return None

    top = random.randint(1, cell_h - h - 1)
    left = random.randint(1, cell_w - w - 1)
    return _embed_mask(mask, cell_h, cell_w, top, left)


def _build_incomplete_fragment(
    full_mask: np.ndarray, reference_mask: np.ndarray, cell_shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray] | None:
    full_coords = [tuple(map(int, coord)) for coord in np.argwhere(full_mask)]
    full_area = len(full_coords)
    max_remove = min(max(2, full_area // 3), full_area - 1)

    if max_remove < 1:
        return None

    for _ in range(FRAGMENT_ATTEMPTS):
        remove_count = random.randint(1, max_remove)
        removed = set(random.sample(full_coords, remove_count))
        current = full_mask.copy()
        for r, c in removed:
            current[r, c] = False
        if not _is_connected(current):
            continue

        try:
            missing = _completion_fill(reference_mask, current, cell_shape)
        except AssertionError:
            continue

        if np.array_equal(current | missing, full_mask):
            return current, missing

    return None


def _grid_shape(rows: int, cols: int, cell_h: int, cell_w: int, outer_border: bool) -> tuple[int, int]:
    if outer_border:
        return rows * cell_h + rows + 1, cols * cell_w + cols + 1
    return rows * cell_h + rows - 1, cols * cell_w + cols - 1


def _cell_origin(index: int, cell_size: int, outer_border: bool) -> int:
    if outer_border:
        return 1 + index * (cell_size + 1)
    return index * (cell_size + 1)


def _render_input(
    fragments: list[np.ndarray],
    colors: list[int],
    rows: int,
    cols: int,
    cell_h: int,
    cell_w: int,
    outer_border: bool,
) -> np.ndarray:
    grid_h, grid_w = _grid_shape(rows, cols, cell_h, cell_w, outer_border)
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    if outer_border:
        for k in range(rows + 1):
            grid[k * (cell_h + 1), :] = GREY
        for k in range(cols + 1):
            grid[:, k * (cell_w + 1)] = GREY
    else:
        for k in range(1, rows):
            grid[k * (cell_h + 1) - 1, :] = GREY
        for k in range(1, cols):
            grid[:, k * (cell_w + 1) - 1] = GREY

    for idx, (mask, color) in enumerate(zip(fragments, colors)):
        rr = idx // cols
        cc = idx % cols
        r0 = _cell_origin(rr, cell_h, outer_border)
        c0 = _cell_origin(cc, cell_w, outer_border)
        patch = grid[r0 : r0 + cell_h, c0 : c0 + cell_w]
        patch[mask] = int(color)

    return grid


def _render_expected_output(
    grid: np.ndarray,
    missing_masks: list[np.ndarray],
    rows: int,
    cols: int,
    cell_h: int,
    cell_w: int,
    outer_border: bool,
) -> np.ndarray:
    out = grid.copy()
    for idx, missing in enumerate(missing_masks):
        rr = idx // cols
        cc = idx % cols
        r0 = _cell_origin(rr, cell_h, outer_border)
        c0 = _cell_origin(cc, cell_w, outer_border)
        patch = out[r0 : r0 + cell_h, c0 : c0 + cell_w]
        patch[missing] = ORANGE
    return out


def _gen_attempt() -> np.ndarray | None:
    rows, cols = random.choice(LAYOUT_CHOICES)
    cell_count = rows * cols
    cell_h = random.randint(CELL_SIZE_MIN, CELL_SIZE_MAX)
    cell_w = random.randint(CELL_SIZE_MIN, CELL_SIZE_MAX)
    outer_border = random.random() < OUTER_BORDER_PROB

    prototype = _generate_prototype(cell_h, cell_w)
    if prototype is None:
        return None

    recipes = random.sample(list(ALL_RECIPES), cell_count)
    reference_idx = random.randrange(cell_count)
    if not any(flip for idx, (_, flip) in enumerate(recipes) if idx != reference_idx):
        return None
    if not any(rot != 0 for idx, (rot, _) in enumerate(recipes) if idx != reference_idx):
        return None

    reference_mask = _apply_recipe(prototype, recipes[reference_idx])
    colors = random.sample(FRAGMENT_COLORS, cell_count)
    fragments: list[np.ndarray] = []
    missing_masks: list[np.ndarray] = []

    for idx, recipe in enumerate(recipes):
        oriented = _apply_recipe(prototype, recipe)
        full_mask = _random_full_fragment(oriented, cell_h, cell_w)
        if full_mask is None:
            return None

        if idx == reference_idx:
            fragments.append(full_mask)
            missing_masks.append(np.zeros((cell_h, cell_w), dtype=bool))
            continue

        result = _build_incomplete_fragment(full_mask, reference_mask, (cell_h, cell_w))
        if result is None:
            return None
        current_mask, missing = result
        fragments.append(current_mask)
        missing_masks.append(missing)

    grid = _render_input(fragments, colors, rows, cols, cell_h, cell_w, outer_border)
    expected = _render_expected_output(grid, missing_masks, rows, cols, cell_h, cell_w, outer_border)

    try:
        solved = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(solved, expected):
        return None
    if np.array_equal(grid, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {GEN_ATTEMPTS} attempts")
