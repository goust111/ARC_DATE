from __future__ import annotations

import random
from collections import deque

import numpy as np


BLUE = 1
GREEN = 3

MAX_GENERATION_ATTEMPTS = 400


def _rotate_cell_once(r: int, c: int, center2: tuple[int, int]) -> tuple[int, int] | None:
    cr2, cc2 = center2
    rr2 = cr2 + (2 * c + 1 - cc2)
    cc2_new = cc2 - (2 * r + 1 - cr2)
    if (rr2 & 1) == 0 or (cc2_new & 1) == 0:
        return None
    return (rr2 - 1) // 2, (cc2_new - 1) // 2


def _rotate_cells_once(
    cells: set[tuple[int, int]],
    center2: tuple[int, int],
    shape: tuple[int, int] | None = None,
) -> set[tuple[int, int]] | None:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        rotated = _rotate_cell_once(r, c, center2)
        if rotated is None:
            return None
        if shape is not None:
            rr, cc = rotated
            if not (0 <= rr < shape[0] and 0 <= cc < shape[1]):
                return None
        out.add(rotated)
    return out


def _bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_clear(blue: set[tuple[int, int]], missing: set[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(missing)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if (r, c) in blue:
                return False
    return True


def _center_candidates(shape: tuple[int, int]):
    height, width = shape
    for cr2 in range(1, 2 * height):
        for cc2 in range(1, 2 * width):
            if (cr2 & 1) != (cc2 & 1):
                continue
            yield cr2, cc2


def _find_missing_completion(
    blue: set[tuple[int, int]],
    shape: tuple[int, int],
) -> tuple[tuple[int, int], set[tuple[int, int]]]:
    best_center: tuple[int, int] | None = None
    best_missing: set[tuple[int, int]] | None = None
    best_score: tuple[int, ...] | None = None

    for center2 in _center_candidates(shape):
        rotated_once = _rotate_cells_once(blue, center2, shape)
        if rotated_once is None:
            continue

        missing = rotated_once - blue
        if not missing:
            continue

        closed = set(blue)
        current = set(blue)
        valid = True
        for _ in range(4):
            current = _rotate_cells_once(current, center2, shape)
            if current is None:
                valid = False
                break
            closed |= current
        if not valid or current != blue:
            continue

        added_total = len(closed - blue)
        r0, c0, r1, c1 = _bbox(missing)
        score = (
            int(added_total == len(missing)),
            int(_bbox_clear(blue, missing)),
            int(r1 > r0 and c1 > c0),
            len(rotated_once & blue),
            -added_total,
            -len(missing),
        )
        if best_score is None or score > best_score:
            best_center = center2
            best_missing = missing
            best_score = score

    if best_center is None or best_missing is None:
        raise ValueError("failed to detect rotational completion")
    return best_center, best_missing


def _draw_frame(out: np.ndarray, missing: set[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(missing)
    out[r0, c0 : c1 + 1] = GREEN
    out[r1, c0 : c1 + 1] = GREEN
    out[r0 : r1 + 1, c0] = GREEN
    out[r0 : r1 + 1, c1] = GREEN
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate the incomplete blue object to recover its missing part, then replace that part by a green bounding frame.

    Input structure:
    - Black is the background.
    - Blue pixels form one nearly 4-fold rotationally symmetric connected object.
    - One directional portion of that object is missing, so rotating the blue object clockwise exposes black completion cells.

    Rule:
    - Detect the object's 90-degree rotational center, which may lie on a pixel center or a four-pixel intersection.
    - Rotate every blue pixel clockwise; every rotated target that lands on black belongs to the missing part.
    - Compute the missing part's row/column extrema and draw a one-pixel-wide green rectangle on those extrema.
    - Erase the temporary missing-part markers, leaving the original blue object plus the green frame.

    Key insight:
    - The frame is not anchored by explicit markers in the input; it is determined by the extrema of the pixels created by mentally completing the object's clockwise rotation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    blue = {tuple(map(int, rc)) for rc in np.argwhere(g == BLUE)}
    assert blue, "expected at least one blue object"

    _center2, missing = _find_missing_completion(blue, g.shape)

    out = g.copy()
    return _draw_frame(out, missing)


def _is_single_component(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    seen = {start}
    queue: deque[tuple[int, int]] = deque([start])
    while queue:
        r, c = queue.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == len(cells)


def _pixel_seed() -> tuple[set[tuple[int, int]], set[tuple[int, int]], tuple[int, int]]:
    plate_h, plate_w, stem_len = random.choice(
        [
            (2, 2, 2),
            (2, 3, 1),
            (2, 3, 2),
            (3, 2, 1),
            (3, 2, 2),
        ]
    )
    valid_tops = [top for top in range(-plate_h + 1, 1) if top <= 0 <= top + plate_h - 1]
    top = random.choice(valid_tops)

    core = {(0, 0)}
    seed = {(0, c) for c in range(1, stem_len + 1)}
    for r in range(top, top + plate_h):
        for c in range(stem_len, stem_len + plate_w):
            seed.add((r, c))
    return core, seed, (1, 1)


def _intersection_seed() -> tuple[set[tuple[int, int]], set[tuple[int, int]], tuple[int, int]]:
    plate_h, plate_w = random.choice([(2, 2), (2, 3), (3, 2)])
    valid_tops = [top for top in range(-plate_h + 1, 2) if max(top, 0) <= min(top + plate_h - 1, 1)]
    top = random.choice(valid_tops)

    core = {(0, 0), (0, 1), (1, 0), (1, 1)}
    seed = {(r, c) for r in range(top, top + plate_h) for c in range(2, 2 + plate_w)}
    if len(seed) == 6 and random.random() < 0.5:
        far_col = 1 + plate_w
        corner_row = top if random.random() < 0.5 else top + plate_h - 1
        seed.remove((corner_row, far_col))
    return core, seed, (2, 2)
def _build_shape() -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    core, seed, center2 = random.choice([_pixel_seed, _intersection_seed])()

    arms: list[set[tuple[int, int]]] = []
    current = set(seed)
    for _ in range(4):
        arms.append(current)
        current = _rotate_cells_once(current, center2)
        if current is None:
            raise ValueError("seed rotation failed")
    if current != seed:
        raise ValueError("four rotations should return to the seed")

    full = set(core)
    for arm in arms:
        if full & arm:
            raise ValueError("core and arms must not overlap")
        full |= arm
    for i in range(4):
        for j in range(i + 1, 4):
            if arms[i] & arms[j]:
                raise ValueError("rotated arms must not overlap")

    r0, c0, _r1, _c1 = _bbox(full)
    normalized_full = {(r - r0, c - c0) for r, c in full}
    normalized_arms = [{(r - r0, c - c0) for r, c in arm} for arm in arms]

    missing = set(random.choice(normalized_arms))
    inp = normalized_full - missing

    if not _is_single_component(inp):
        raise ValueError("input object must stay connected")
    if not (15 <= len(inp) <= 30):
        raise ValueError("blue area out of target range")
    if not (3 <= len(missing) <= 8):
        raise ValueError("missing area out of target range")
    mr0, mc0, mr1, mc1 = _bbox(missing)
    if mr1 == mr0 or mc1 == mc0:
        raise ValueError("missing bbox must have positive area")
    if not _bbox_clear(inp, missing):
        raise ValueError("missing bbox must stay clear of blue pixels")

    return inp, missing


def _paint_cells(
    grid: np.ndarray,
    cells: set[tuple[int, int]],
    color: int,
    offset: tuple[int, int] = (0, 0),
) -> None:
    dr, dc = offset
    for r, c in cells:
        grid[r + dr, c + dc] = color


def _expected_output(grid: np.ndarray, missing: set[tuple[int, int]]) -> np.ndarray:
    out = grid.copy()
    return _draw_frame(out, missing)


def _attempt_generate() -> np.ndarray | None:
    try:
        input_cells, missing_cells = _build_shape()
    except ValueError:
        return None

    all_cells = input_cells | missing_cells
    patch_r0, patch_c0, patch_r1, patch_c1 = _bbox(all_cells)
    patch_h = patch_r1 - patch_r0 + 1
    patch_w = patch_c1 - patch_c0 + 1

    min_h = max(14, patch_h + 1)
    min_w = max(14, patch_w + 1)
    max_h = min(19, patch_h + 7, 30)
    max_w = min(19, patch_w + 7, 30)
    if min_h > max_h or min_w > max_w:
        return None

    height = random.randint(min_h, max_h)
    width = random.randint(min_w, max_w)
    offset = (random.randint(0, height - patch_h), random.randint(0, width - patch_w))

    grid = np.zeros((height, width), dtype=int)
    _paint_cells(grid, input_cells, BLUE, offset)

    blue_abs = {(r + offset[0], c + offset[1]) for r, c in input_cells}
    missing_abs = {(r + offset[0], c + offset[1]) for r, c in missing_cells}

    if not _is_single_component(blue_abs):
        return None
    if not _bbox_clear(blue_abs, missing_abs):
        return None

    expected = _expected_output(grid, missing_abs)
    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
