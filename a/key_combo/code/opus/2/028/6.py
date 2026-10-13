from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
LEFT_COLOR = 1
RIGHT_COLOR = 3
SEPARATOR_COLOR = 5
RAY_COLOR = 7

HEIGHT_RANGE = (9, 13)
HALF_WIDTH_RANGE = (5, 7)
SHAPE_SIZE_RANGE = (6, 10)
OVERLAP_RANGE = (2, 4)
MAX_GENERATION_ATTEMPTS = 2000


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int, height: int, width: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < height and 0 <= nc < width:
            yield nr, nc


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, height, width):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _is_connected(mask: np.ndarray) -> bool:
    comps = _mask_components(mask)
    return len(comps) == 1


def _is_irregular(cells: set[tuple[int, int]]) -> bool:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    height = max(rows) - min(rows) + 1
    width = max(cols) - min(cols) + 1
    return height > 1 and width > 1 and height * width != len(cells)


def _separator_col(grid: np.ndarray, bg: int) -> int:
    height, width = map(int, grid.shape)
    cols = []
    for c in range(width):
        color = int(grid[0, c])
        if color == bg:
            continue
        if np.all(grid[:, c] == color):
            cols.append(c)
    assert len(cols) == 1, "expected exactly one full-height non-background separator column"
    sep = int(cols[0])
    assert sep > 0 and sep < width - 1, "separator cannot lie on the border"
    assert np.all(grid[:, sep] == int(grid[0, sep]))
    return sep


def _nearest_direction(r: int, c: int, height: int, width: int) -> tuple[int, int]:
    options = [
        ((-1, 0), r),
        ((0, 1), width - 1 - c),
        ((1, 0), height - 1 - r),
        ((0, -1), c),
    ]
    best_dist = min(dist for _, dist in options)
    for direction, dist in options:
        if dist == best_dist:
            return direction
    raise AssertionError("unreachable")


def _direction_name(direction: tuple[int, int]) -> str:
    mapping = {
        (-1, 0): "up",
        (0, 1): "right",
        (1, 0): "down",
        (0, -1): "left",
    }
    return mapping[direction]


def _fold_xor(left: np.ndarray, right_input: np.ndarray, bg: int) -> np.ndarray:
    right = np.fliplr(np.asarray(right_input, dtype=int))
    out = np.full(left.shape, bg, dtype=int)
    left_fg = left != bg
    right_fg = right != bg
    out[left_fg & ~right_fg] = left[left_fg & ~right_fg]
    out[right_fg & ~left_fg] = right[right_fg & ~left_fg]
    return out


def _apply_rays(mid: np.ndarray, bg: int) -> np.ndarray:
    out = np.asarray(mid, dtype=int).copy()
    height, width = map(int, out.shape)
    source_mask = out != bg
    coords = np.argwhere(source_mask)
    for r, c in coords:
        dr, dc = _nearest_direction(int(r), int(c), height, width)
        rr, cc = int(r) + dr, int(c) + dc
        while 0 <= rr < height and 0 <= cc < width:
            if not source_mask[rr, cc]:
                out[rr, cc] = RAY_COLOR
            rr += dr
            cc += dc
    return out


def _decode_scene(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "input must be a non-empty 2D grid"
    bg = _get_bg(g)
    sep = _separator_col(g, bg)
    height, width = map(int, g.shape)
    assert 2 * sep + 1 == width, "separator must split the grid into two equal panels"
    left = g[:, :sep].copy()
    right_input = g[:, sep + 1 :].copy()
    assert left.shape == right_input.shape == (height, sep)
    return g, left, right_input, bg


def _folded_output(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, left, right_input, bg = _decode_scene(grid)
    mid = _fold_xor(left, right_input, bg)
    out = _apply_rays(mid, bg)
    return mid, out


def main(grid: np.ndarray) -> np.ndarray:
    """Fold the separator-split panels with XOR cancellation, then extend each surviving pixel to its nearest folded boundary.

    Input structure:
    - A black background is split by one full-height grey separator column into equal-width left and right panels.
    - The left panel contains one blue 4-connected foreground shape, and the right panel contains one green 4-connected foreground shape.
    - The two panel shapes partly match after mirroring the right panel across the separator, but they do not match perfectly.

    Rule:
    - Remove the separator column and mirror the right panel horizontally into the left panel coordinate system.
    - Overlay the mirrored right panel onto the left panel using XOR-style foreground cancellation: cells occupied by exactly one side survive in their original color, while cells occupied on both sides become black.
    - For every surviving foreground cell in the folded grid, choose the nearest boundary of the folded grid using distances to top, right, bottom, and left, with tie priority top > right > bottom > left.
    - Extend an orange ray from that cell step by step toward the chosen boundary until the grid edge; rays may pass over one another, but the original surviving blue and green cells stay unchanged.

    Key insight:
    The fold both changes which pixels survive and changes the coordinate system used for boundary distances, so the final rays are determined by the XOR fold result in the reduced-width grid rather than by the original two-panel layout.
    """
    _, out = _folded_output(grid)
    return out


def _grow_connected_cells(height: int, width: int, size: int) -> set[tuple[int, int]]:
    cells = {(random.randrange(height), random.randrange(width))}
    while len(cells) < size:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c, height, width):
                if (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            raise RuntimeError("frontier unexpectedly empty while growing a connected shape")
        cells.add(random.choice(frontier))
    return cells


def _paint_cells(grid: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _full_grid_direction(r: int, c: int, height: int, width: int) -> tuple[int, int]:
    return _nearest_direction(r, c, height, width)


def _folded_coord_to_input_coord(
    cell: tuple[int, int], color: int, half_width: int
) -> tuple[int, int]:
    r, c = cell
    if color == LEFT_COLOR:
        return r, c
    assert color == RIGHT_COLOR
    return r, 2 * half_width - c


def _survivor_data(mid: np.ndarray) -> list[tuple[int, int, int]]:
    data = []
    for r, c in np.argwhere(mid != BG):
        data.append((int(r), int(c), int(mid[r, c])))
    return data


def _direction_mismatches(mid: np.ndarray) -> int:
    height, half_width = map(int, mid.shape)
    full_width = 2 * half_width + 1
    mismatches = 0
    for r, c, color in _survivor_data(mid):
        folded_dir = _nearest_direction(r, c, height, half_width)
        src_r, src_c = _folded_coord_to_input_coord((r, c), color, half_width)
        full_dir = _full_grid_direction(src_r, src_c, height, full_width)
        if folded_dir != full_dir:
            mismatches += 1
    return mismatches


def _ray_cells(mid: np.ndarray) -> set[tuple[int, int]]:
    rays: set[tuple[int, int]] = set()
    height, width = map(int, mid.shape)
    source = {(int(r), int(c)) for r, c in np.argwhere(mid != BG)}
    for r, c in source:
        dr, dc = _nearest_direction(r, c, height, width)
        rr, cc = r + dr, c + dc
        while 0 <= rr < height and 0 <= cc < width:
            if (rr, cc) not in source:
                rays.add((rr, cc))
            rr += dr
            cc += dc
    return rays


def _has_tie_position(mid: np.ndarray) -> bool:
    height, width = map(int, mid.shape)
    for r, c, _ in _survivor_data(mid):
        distances = [r, width - 1 - c, height - 1 - r, c]
        best = min(distances)
        if distances.count(best) >= 2:
            return True
    return False


def _direction_count(mid: np.ndarray) -> int:
    height, width = map(int, mid.shape)
    return len(
        {
            _direction_name(_nearest_direction(r, c, height, width))
            for r, c, _ in _survivor_data(mid)
        }
    )


def _naive_fold_union(left: np.ndarray, right_input: np.ndarray, bg: int) -> np.ndarray:
    right = np.fliplr(right_input)
    out = left.copy()
    mask = right != bg
    out[mask] = right[mask]
    return out


def _semantic_accepts(grid: np.ndarray, *, require_tie: bool, min_direction_count: int) -> bool:
    try:
        g, left, right_input, bg = _decode_scene(grid)
    except AssertionError:
        return False
    if bg != BG:
        return False
    if int(g[:, left.shape[1]][0]) != SEPARATOR_COLOR:
        return False
    if set(map(int, np.unique(left))) - {BG, LEFT_COLOR}:
        return False
    if set(map(int, np.unique(right_input))) - {BG, RIGHT_COLOR}:
        return False
    left_mask = left == LEFT_COLOR
    right_mask_input = right_input == RIGHT_COLOR
    if left_mask.sum() < SHAPE_SIZE_RANGE[0] or left_mask.sum() > SHAPE_SIZE_RANGE[1]:
        return False
    if right_mask_input.sum() < SHAPE_SIZE_RANGE[0] or right_mask_input.sum() > SHAPE_SIZE_RANGE[1]:
        return False
    if not _is_connected(left_mask) or not _is_connected(right_mask_input):
        return False
    if not _is_irregular(set(map(tuple, np.argwhere(left_mask)))) or not _is_irregular(
        set(map(tuple, np.argwhere(right_mask_input)))
    ):
        return False

    mid = _fold_xor(left, right_input, bg)
    right_fold_mask = np.fliplr(right_input == RIGHT_COLOR)
    overlap = int(np.sum(left_mask & right_fold_mask))
    left_only = int(np.sum((left == LEFT_COLOR) & ~right_fold_mask))
    right_only = int(np.sum((np.fliplr(right_input) == RIGHT_COLOR) & ~left_mask))
    if not (OVERLAP_RANGE[0] <= overlap <= OVERLAP_RANGE[1]):
        return False
    if left_only < 1 or right_only < 1 or left_only + right_only < 3:
        return False
    if np.array_equal(mid, left) or np.array_equal(mid, np.fliplr(right_input)):
        return False
    if np.array_equal(mid, _naive_fold_union(left, right_input, bg)):
        return False

    survivor_colors = set(map(int, np.unique(mid[mid != BG])))
    if survivor_colors != {LEFT_COLOR, RIGHT_COLOR}:
        return False
    if _direction_count(mid) < min_direction_count:
        return False
    if require_tie and not _has_tie_position(mid):
        return False
    if _direction_mismatches(mid) < 1:
        return False

    ray_cells = _ray_cells(mid)
    if len(ray_cells) < 4:
        return False

    out = main(grid)
    if out.shape != mid.shape:
        return False
    if set(map(int, np.unique(out))) - {BG, LEFT_COLOR, RIGHT_COLOR, RAY_COLOR}:
        return False
    if not np.array_equal(out[mid == LEFT_COLOR], np.full(int(np.sum(mid == LEFT_COLOR)), LEFT_COLOR)):
        return False
    if not np.array_equal(out[mid == RIGHT_COLOR], np.full(int(np.sum(mid == RIGHT_COLOR)), RIGHT_COLOR)):
        return False
    actual_ray_cells = {(int(r), int(c)) for r, c in np.argwhere(out == RAY_COLOR)}
    if actual_ray_cells != ray_cells:
        return False
    return True


def _make_input_from_folded(left_cells: set[tuple[int, int]], right_fold_cells: set[tuple[int, int]], height: int, width: int) -> np.ndarray:
    left = np.full((height, width), BG, dtype=int)
    right_fold = np.full((height, width), BG, dtype=int)
    _paint_cells(left, left_cells, LEFT_COLOR)
    _paint_cells(right_fold, right_fold_cells, RIGHT_COLOR)
    grid = np.full((height, 2 * width + 1), BG, dtype=int)
    grid[:, :width] = left
    grid[:, width] = SEPARATOR_COLOR
    grid[:, width + 1 :] = np.fliplr(right_fold)
    return grid


def _gen_attempt(require_tie: bool, min_direction_count: int) -> np.ndarray | None:
    height = random.randint(*HEIGHT_RANGE)
    width = random.randint(*HALF_WIDTH_RANGE)
    left_size = random.randint(*SHAPE_SIZE_RANGE)
    right_size = random.randint(*SHAPE_SIZE_RANGE)

    try:
        left_cells = _grow_connected_cells(height, width, left_size)
        right_fold_cells = _grow_connected_cells(height, width, right_size)
    except RuntimeError:
        return None

    if not _is_irregular(left_cells) or not _is_irregular(right_fold_cells):
        return None

    grid = _make_input_from_folded(left_cells, right_fold_cells, height, width)
    if not _semantic_accepts(grid, require_tie=require_tie, min_direction_count=min_direction_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    modes = [
        (True, 3),
        (True, 2),
        (False, 3),
    ]
    for _ in range(MAX_GENERATION_ATTEMPTS):
        require_tie, min_direction_count = random.choice(modes)
        grid = _gen_attempt(require_tie=require_tie, min_direction_count=min_direction_count)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fold-xor-ray instance")
