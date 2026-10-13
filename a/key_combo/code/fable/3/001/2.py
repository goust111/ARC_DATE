from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import bbox_h, bbox_w, colors_present, find_components, get_bg, paste, random_free_location


ORTHOGONAL = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIAGONAL = ((-1, -1), (-1, 1), (1, -1), (1, 1))
ARC_COLORS = tuple(range(10))
GENERATION_ATTEMPTS = 600
SPRITE_ATTEMPTS = 120

_BRANCH_QUEUE: list[bool] = []


def _compact_cross_centers(g: np.ndarray, bg: int, color: int) -> list[tuple[int, int]]:
    centers: list[tuple[int, int]] = []
    height, width = map(int, g.shape)
    for r in range(1, height - 1):
        for c in range(1, width - 1):
            if int(g[r, c]) != int(color):
                continue
            if not all(int(g[r + dr, c + dc]) == int(color) for dr, dc in ORTHOGONAL):
                continue
            if not all(int(g[r + dr, c + dc]) == int(bg) for dr, dc in DIAGONAL):
                continue
            centers.append((int(r), int(c)))
    return centers


def _detect_marker_color(g: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int]]]:
    candidates: list[tuple[int, list[tuple[int, int]]]] = []
    for color in colors_present(g, exclude={bg}):
        centers = _compact_cross_centers(g, bg, int(color))
        if len(centers) == 4:
            candidates.append((int(color), centers))
    assert len(candidates) == 1, "expected exactly one color with four compact cross centers"
    return candidates[0]


def _target_bbox_from_centers(centers: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in centers]
    cols = [c for _r, c in centers]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    corner_set = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    assert set(centers) == corner_set, "cross centers must occupy the four extrema corners"
    target = (r0 + 1, c0 + 1, r1 - 1, c1 - 1)
    assert target[0] <= target[2] and target[1] <= target[3], "inset window must be non-empty"
    return target


def _object_patch(
    g: np.ndarray, bg: int, color: int, component: dict
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    obj_bbox = tuple(map(int, component["bbox"]))
    r0, c0, _r1, _c1 = obj_bbox
    patch = np.full((bbox_h(obj_bbox), bbox_w(obj_bbox)), int(bg), dtype=int)
    coords = component["coords"]
    rel = coords - np.array([r0, c0])
    patch[rel[:, 0], rel[:, 1]] = int(color)
    return patch, obj_bbox


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the second foreground object into the inset window defined by four hidden compact crosses, rotating it when its bbox is transposed.

    Input structure:
    - The dominant background surrounds one drawing color that forms irregular blobs and exactly four compact cross centers.
    - A compact cross center is a drawing-color pixel whose four orthogonal neighbors are drawing color and whose four diagonals are background.
    - One independent second foreground color forms a single 4-connected object whose bounding box matches the target window or has the window dimensions swapped.

    Rule:
    - Detect the four compact cross centers, take the extrema of their rows and columns, and inset that rectangle by one cell to get the target window.
    - Clear the second-color object from its original position.
    - If the object bbox already has the target window size, translate it into the window; if the bbox dimensions are swapped, rotate the object clockwise 90 degrees before placing it.
    - Leave the drawing-color crosses and blobs unchanged.

    Key insight:
    - The corner markers are not a separate marker color; they are local cross structures hidden inside same-color clutter, and their inset extrema determine both the destination and whether the object must rotate.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = int(get_bg(g))

    marker_color, centers = _detect_marker_color(g, bg)
    target_bbox = _target_bbox_from_centers(centers)

    fg_colors = colors_present(g, exclude={bg})
    object_colors = [int(color) for color in fg_colors if int(color) != int(marker_color)]
    assert len(object_colors) == 1, "expected one non-marker foreground object color"
    object_color = object_colors[0]

    components = find_components(g, bg=bg, colors=[object_color], connectivity=4)
    assert len(components) == 1, "expected one 4-connected moving object"
    assert int(components[0]["area"]) == int(np.count_nonzero(g == object_color))
    source_patch, source_bbox = _object_patch(g, bg, object_color, components[0])

    target_h, target_w = bbox_h(target_bbox), bbox_w(target_bbox)
    source_h, source_w = bbox_h(source_bbox), bbox_w(source_bbox)
    if (source_h, source_w) == (target_h, target_w):
        placed_patch = source_patch
    elif (source_h, source_w) == (target_w, target_h):
        placed_patch = np.rot90(source_patch, k=-1)
    else:
        raise AssertionError("object bbox must match the target window or its transpose")

    assert placed_patch.shape == (target_h, target_w)
    out = g.copy()
    out[g == object_color] = bg

    tr0, tc0, tr1, tc1 = target_bbox
    target_region = out[tr0 : tr1 + 1, tc0 : tc1 + 1]
    object_mask = placed_patch != bg
    assert np.all(target_region[object_mask] == bg), "moving object must land on empty target cells"
    target_region[object_mask] = object_color
    return out


def _next_rotate_branch() -> bool:
    global _BRANCH_QUEUE
    if not _BRANCH_QUEUE:
        _BRANCH_QUEUE = [False, True, random.choice([False, True])]
        random.shuffle(_BRANCH_QUEUE)
    return bool(_BRANCH_QUEUE.pop())


def _mask_cross_centers(mask: np.ndarray) -> list[tuple[int, int]]:
    centers: list[tuple[int, int]] = []
    h, w = map(int, mask.shape)
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if not bool(mask[r, c]):
                continue
            if not all(bool(mask[r + dr, c + dc]) for dr, dc in ORTHOGONAL):
                continue
            if not all(not bool(mask[r + dr, c + dc]) for dr, dc in DIAGONAL):
                continue
            centers.append((int(r), int(c)))
    return centers


def _carve_path(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
    r, c = map(int, start)
    er, ec = map(int, end)
    mask[r, c] = True
    while (r, c) != (er, ec):
        moves: list[tuple[int, int]] = []
        if r < er:
            moves.append((1, 0))
        elif r > er:
            moves.append((-1, 0))
        if c < ec:
            moves.append((0, 1))
        elif c > ec:
            moves.append((0, -1))
        dr, dc = random.choice(moves)
        r += dr
        c += dc
        mask[r, c] = True


def _frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = map(int, mask.shape)
    cells: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if bool(mask[r, c]):
                continue
            if any(0 <= r + dr < h and 0 <= c + dc < w and bool(mask[r + dr, c + dc]) for dr, dc in ORTHOGONAL):
                cells.append((int(r), int(c)))
    return cells


def _gen_object_mask(h: int, w: int) -> np.ndarray | None:
    area = h * w
    for _ in range(SPRITE_ATTEMPTS):
        anchors = [
            (0, random.randrange(w)),
            (h - 1, random.randrange(w)),
            (random.randrange(h), 0),
            (random.randrange(h), w - 1),
        ]
        random.shuffle(anchors)
        mask = np.zeros((h, w), dtype=bool)
        for a, b in zip(anchors, anchors[1:]):
            _carve_path(mask, a, b)

        min_target = int(mask.sum())
        if min_target > area - 2:
            continue
        max_target = min(area - 2, max(min_target, int(area * 0.72)))
        target = random.randint(min_target, max_target)
        while int(mask.sum()) < target:
            candidates = _frontier(mask)
            if not candidates:
                break
            r, c = random.choice(candidates)
            mask[r, c] = True

        if int(mask.sum()) != target:
            continue
        if not (mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if int((~mask).sum()) < 2:
            continue
        if _mask_cross_centers(mask):
            continue
        if h == w and np.array_equal(mask, np.rot90(mask, k=-1)):
            continue
        return mask
    return None


def _target_cells(target_bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = target_bbox
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _cross_cells(center: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    return {(r, c)} | {(r + dr, c + dc) for dr, dc in ORTHOGONAL}


def _forbidden_noise_cells(
    shape: tuple[int, int], centers: set[tuple[int, int]], target_bbox: tuple[int, int, int, int]
) -> set[tuple[int, int]]:
    height, width = map(int, shape)
    forbidden = _target_cells(target_bbox)
    for r, c in centers:
        for dr, dc in DIAGONAL:
            rr, cc = r + dr, c + dc
            if 0 <= rr < height and 0 <= cc < width:
                forbidden.add((rr, cc))
    return forbidden


def _add_blob(
    grid: np.ndarray, color: int, start: tuple[int, int], size: int, forbidden: set[tuple[int, int]]
) -> None:
    height, width = map(int, grid.shape)
    if start in forbidden or int(grid[start]) != int(get_bg(grid)):
        return
    cells = {start}
    grid[start] = int(color)
    for _ in range(size - 1):
        candidates: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in ORTHOGONAL:
                rr, cc = r + dr, c + dc
                if not (0 <= rr < height and 0 <= cc < width):
                    continue
                if (rr, cc) in forbidden or int(grid[rr, cc]) != int(get_bg(grid)):
                    continue
                candidates.append((int(rr), int(cc)))
        if not candidates:
            break
        nxt = random.choice(candidates)
        cells.add(nxt)
        grid[nxt] = int(color)


def _attachment_candidates(center: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = center
    return [
        (r - 2, c - 1),
        (r - 2, c + 1),
        (r + 2, c - 1),
        (r + 2, c + 1),
        (r - 1, c - 2),
        (r + 1, c - 2),
        (r - 1, c + 2),
        (r + 1, c + 2),
    ]


def _add_drawing_noise(
    grid: np.ndarray,
    bg: int,
    draw_color: int,
    centers: set[tuple[int, int]],
    target_bbox: tuple[int, int, int, int],
) -> bool:
    height, width = map(int, grid.shape)
    forbidden = _forbidden_noise_cells(grid.shape, centers, target_bbox)

    for center in centers:
        for r, c in _cross_cells(center):
            grid[r, c] = int(draw_color)

    attached = []
    for center in centers:
        candidates = [
            cell
            for cell in _attachment_candidates(center)
            if 0 <= cell[0] < height and 0 <= cell[1] < width and cell not in forbidden and int(grid[cell]) == int(bg)
        ]
        random.shuffle(candidates)
        attached.extend(candidates[: random.randint(1, min(2, len(candidates)))])
    random.shuffle(attached)
    for start in attached[: random.randint(4, min(8, len(attached)))]:
        _add_blob(grid, draw_color, start, random.randint(2, 5), forbidden)

    free = [
        (r, c)
        for r in range(height)
        for c in range(width)
        if (r, c) not in forbidden and int(grid[r, c]) == int(bg)
    ]
    random.shuffle(free)
    for start in free[: random.randint(2, 5)]:
        _add_blob(grid, draw_color, start, random.randint(3, 8), forbidden)

    found = set(_compact_cross_centers(grid, bg, draw_color))
    if found != centers:
        return False
    r0, c0, r1, c1 = target_bbox
    return bool(np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == bg))


def _choose_target_geometry(rotate_branch: bool) -> tuple[int, int, int, int, tuple[int, int, int, int]] | None:
    for _ in range(80):
        target_h = random.randint(4, 8)
        target_w = random.randint(4, 8)
        if rotate_branch and target_h == target_w:
            continue
        source_h, source_w = (target_w, target_h) if rotate_branch else (target_h, target_w)
        min_h = max(16, target_h + source_h + 8)
        min_w = max(16, target_w + source_w + 8)
        if min_h > 30 or min_w > 30:
            continue
        height = random.randint(min_h, 30)
        width = random.randint(min_w, 30)
        margin = 3
        if height - target_h - margin - 1 < margin or width - target_w - margin - 1 < margin:
            continue
        tr0 = random.randint(margin, height - target_h - margin - 1)
        tc0 = random.randint(margin, width - target_w - margin - 1)
        target_bbox = (tr0, tc0, tr0 + target_h - 1, tc0 + target_w - 1)
        return height, width, source_h, source_w, target_bbox
    return None


def _place_source_object(
    grid: np.ndarray,
    bg: int,
    object_color: int,
    mask: np.ndarray,
    target_bbox: tuple[int, int, int, int],
) -> tuple[int, int] | None:
    occupancy = np.zeros_like(grid, dtype=int)
    occupancy[grid != bg] = 1
    r0, c0, r1, c1 = target_bbox
    occupancy[r0 : r1 + 1, c0 : c1 + 1] = 1
    footprint = np.ones(mask.shape, dtype=int)
    try:
        sr, sc = random_free_location(occupancy, footprint, transparent=0, border=1, padding=1)
    except ValueError:
        return None

    patch = np.full(mask.shape, int(bg), dtype=int)
    patch[mask] = int(object_color)
    paste(grid, patch, sr, sc, transparent=bg)
    return int(sr), int(sc)


def _expected_output(
    grid: np.ndarray,
    bg: int,
    object_color: int,
    source_patch: np.ndarray,
    target_bbox: tuple[int, int, int, int],
    rotate_branch: bool,
) -> np.ndarray:
    out = grid.copy()
    out[grid == object_color] = bg
    patch = np.rot90(source_patch, k=-1) if rotate_branch else source_patch
    paste(out, patch, target_bbox[0], target_bbox[1], transparent=bg)
    return out


def _valid_generated(
    grid: np.ndarray,
    expected: np.ndarray,
    draw_color: int,
    object_color: int,
    rotate_branch: bool,
) -> bool:
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False
    bg = int(get_bg(grid))
    if int(np.count_nonzero(grid == bg)) <= max(int(np.count_nonzero(grid == c)) for c in colors_present(grid, exclude={bg})):
        return False
    if set(colors_present(grid, exclude={bg})) != {int(draw_color), int(object_color)}:
        return False
    if _compact_cross_centers(grid, bg, object_color):
        return False

    try:
        marker_color, centers = _detect_marker_color(grid, bg)
        target_bbox = _target_bbox_from_centers(centers)
        components = find_components(grid, bg=bg, colors=[object_color], connectivity=4)
        actual = main(grid)
    except AssertionError:
        return False
    if marker_color != int(draw_color) or len(components) != 1:
        return False
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(grid, expected):
        return False

    source_bbox = tuple(map(int, components[0]["bbox"]))
    target_size = (bbox_h(target_bbox), bbox_w(target_bbox))
    source_size = (bbox_h(source_bbox), bbox_w(source_bbox))
    if rotate_branch:
        if target_size[0] == target_size[1] or source_size != (target_size[1], target_size[0]):
            return False
    elif source_size != target_size:
        return False

    r0, c0, r1, c1 = target_bbox
    if not np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == bg):
        return False
    if not np.array_equal(grid == draw_color, expected == draw_color):
        return False
    return True


def _gen_attempt(rotate_branch: bool) -> np.ndarray | None:
    geometry = _choose_target_geometry(rotate_branch)
    if geometry is None:
        return None
    height, width, source_h, source_w, target_bbox = geometry

    bg, draw_color, object_color = random.sample(ARC_COLORS, 3)
    grid = np.full((height, width), int(bg), dtype=int)

    tr0, tc0, tr1, tc1 = target_bbox
    centers = {
        (tr0 - 1, tc0 - 1),
        (tr0 - 1, tc1 + 1),
        (tr1 + 1, tc0 - 1),
        (tr1 + 1, tc1 + 1),
    }
    if not _add_drawing_noise(grid, bg, draw_color, centers, target_bbox):
        return None

    mask = _gen_object_mask(source_h, source_w)
    if mask is None:
        return None
    source_origin = _place_source_object(grid, bg, object_color, mask, target_bbox)
    if source_origin is None:
        return None

    source_patch = np.full((source_h, source_w), int(bg), dtype=int)
    source_patch[mask] = int(object_color)
    expected = _expected_output(grid, bg, object_color, source_patch, target_bbox, rotate_branch)

    if not _valid_generated(grid, expected, draw_color, object_color, rotate_branch):
        return None
    return grid


def generate_input() -> np.ndarray:
    rotate_branch = _next_rotate_branch()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rotate_branch)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid compact-cross window input")
