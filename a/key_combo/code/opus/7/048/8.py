from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
HOLE = 6
PATTERN_COLORS = (BLUE, RED, GREEN)
GEN_MAX_ATTEMPTS = 400


def _swap_red_blue(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    blue = out == BLUE
    red = out == RED
    out[blue] = RED
    out[red] = BLUE
    return out


def _pattern_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(np.asarray(grid) != BLACK)
    assert coords.size > 0, "expected a visible pattern"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _axis_stats(
    grid: np.ndarray,
    *,
    vertical: bool,
    axis_sum: int,
    bbox: tuple[int, int, int, int],
    wildcard: int = HOLE,
) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    conflicts = 0
    evidence = 0
    if vertical:
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                mate_c = axis_sum - c
                if not (c0 <= mate_c <= c1) or c > mate_c:
                    continue
                a = int(grid[r, c])
                b = int(grid[r, mate_c])
                if a == wildcard or b == wildcard:
                    continue
                if a != b:
                    conflicts += 1
                evidence += 1 if c == mate_c else 2
    else:
        for r in range(r0, r1 + 1):
            mate_r = axis_sum - r
            if not (r0 <= mate_r <= r1) or r > mate_r:
                continue
            for c in range(c0, c1 + 1):
                a = int(grid[r, c])
                b = int(grid[mate_r, c])
                if a == wildcard or b == wildcard:
                    continue
                if a != b:
                    conflicts += 1
                evidence += 1 if r == mate_r else 2
    return conflicts, evidence


def _detect_axis(grid: np.ndarray) -> tuple[str, int]:
    bbox = _pattern_bbox(grid)
    r0, c0, r1, c1 = bbox
    candidates = []
    for name, vertical, axis_sum in (
        ("vertical", True, c0 + c1),
        ("horizontal", False, r0 + r1),
    ):
        conflicts, evidence = _axis_stats(
            grid, vertical=vertical, axis_sum=axis_sum, bbox=bbox
        )
        if conflicts == 0 and evidence > 0:
            candidates.append((name, axis_sum, evidence))

    assert candidates, "expected at least one reflection axis"
    if len(candidates) == 1:
        return candidates[0][0], candidates[0][1]

    candidates.sort(key=lambda item: item[2], reverse=True)
    assert candidates[0][2] > candidates[1][2], "expected a unique reflection axis"
    return candidates[0][0], candidates[0][1]


def _reflect_cell(row: int, col: int, axis_kind: str, axis_sum: int) -> tuple[int, int]:
    if axis_kind == "vertical":
        return row, axis_sum - col
    return axis_sum - row, col


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap red and blue, repair the magenta holes by reflection, then swap the repaired colors back.

    Input structure:
    - A black square grid contains one centered multicolor pattern made from blue, red, and green cells.
    - Some pattern cells are replaced by magenta hole pixels that do not belong to the final image.
    - After swapping blue and red, the non-hole pattern is reflection-symmetric about one central vertical or horizontal axis.

    Rule:
    - Copy the grid and swap every blue cell with red while leaving green, black, and magenta unchanged.
    - Detect the unique reflection axis of the swapped pattern while treating magenta holes as wildcards.
    - Replace each magenta hole by the swapped-space color at its reflected counterpart, then swap blue and red back.

    Key insight:
    - The magenta cells are missing pixels inside a reflected pattern, so once the red/blue palette is normalized, each hole has a forced mirror partner.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    swapped = _swap_red_blue(g)
    axis_kind, axis_sum = _detect_axis(swapped)

    out = swapped.copy()
    height, width = out.shape
    for row, col in np.argwhere(out == HOLE):
        row = int(row)
        col = int(col)
        mate_row, mate_col = _reflect_cell(row, col, axis_kind, axis_sum)
        if 0 <= mate_row < height and 0 <= mate_col < width and int(out[mate_row, mate_col]) != HOLE:
            out[row, col] = int(out[mate_row, mate_col])

    assert not np.any(out == HOLE), "every hole must have a visible mirror mate"
    return _swap_red_blue(out)


def _exact_reflection(grid: np.ndarray, axis_kind: str, axis_sum: int) -> bool:
    r0, c0, r1, c1 = _pattern_bbox(grid)
    if axis_kind == "vertical":
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                mate_c = axis_sum - c
                if not (c0 <= mate_c <= c1):
                    return False
                if int(grid[r, c]) != int(grid[r, mate_c]):
                    return False
    else:
        for r in range(r0, r1 + 1):
            mate_r = axis_sum - r
            if not (r0 <= mate_r <= r1):
                return False
            for c in range(c0, c1 + 1):
                if int(grid[r, c]) != int(grid[mate_r, c]):
                    return False
    return True


def _build_symmetric_patch(
    height: int,
    width: int,
    axis_kind: str,
) -> np.ndarray | None:
    patch = np.zeros((height, width), dtype=int)
    if axis_kind == "vertical":
        mid = width // 2
        arm_lengths = [random.randint(0, mid) for _ in range(height)]
        arm_lengths[random.randrange(height)] = mid
        if sum(arm_lengths) < 4 or arm_lengths == arm_lengths[::-1]:
            return None
        for r in range(height):
            patch[r, mid] = random.choice(PATTERN_COLORS)
            for dist in range(1, arm_lengths[r] + 1):
                color = random.choice(PATTERN_COLORS)
                patch[r, mid - dist] = color
                patch[r, mid + dist] = color
    else:
        mid = height // 2
        arm_lengths = [random.randint(0, mid) for _ in range(width)]
        arm_lengths[random.randrange(width)] = mid
        if sum(arm_lengths) < 4 or arm_lengths == arm_lengths[::-1]:
            return None
        for c in range(width):
            patch[mid, c] = random.choice(PATTERN_COLORS)
            for dist in range(1, arm_lengths[c] + 1):
                color = random.choice(PATTERN_COLORS)
                patch[mid - dist, c] = color
                patch[mid + dist, c] = color

    occupied = np.argwhere(patch != BLACK)
    if occupied.size == 0:
        return None
    r0, c0 = occupied.min(axis=0)
    r1, c1 = occupied.max(axis=0)
    if (int(r1 - r0 + 1), int(c1 - c0 + 1)) != (height, width):
        return None
    if not all(np.any(patch == color) for color in PATTERN_COLORS):
        coords = [tuple(map(int, rc)) for rc in occupied]
        random.shuffle(coords)
        for color, (r, c) in zip(PATTERN_COLORS, coords):
            if axis_kind == "vertical":
                mate = (r, width - 1 - c)
            else:
                mate = (height - 1 - r, c)
            patch[r, c] = color
            patch[mate] = color

    other_axis = "horizontal" if axis_kind == "vertical" else "vertical"
    own_sum = width - 1 if axis_kind == "vertical" else height - 1
    other_sum = height - 1 if axis_kind == "vertical" else width - 1
    if not _exact_reflection(patch, axis_kind, own_sum):
        return None
    if _exact_reflection(patch, other_axis, other_sum):
        return None
    return patch


def _hole_candidates(patch: np.ndarray, axis_kind: str) -> list[tuple[int, int]]:
    height, width = patch.shape
    candidates = []
    if axis_kind == "vertical":
        mid = width // 2
        for r in range(height):
            for c in range(mid):
                mate_c = width - 1 - c
                if patch[r, c] != BLACK and patch[r, mate_c] != BLACK:
                    candidates.append((r, c))
    else:
        mid = height // 2
        for r in range(mid):
            for c in range(width):
                mate_r = height - 1 - r
                if patch[r, c] != BLACK and patch[mate_r, c] != BLACK:
                    candidates.append((r, c))
    random.shuffle(candidates)
    return candidates


def _embed_patch(
    patch: np.ndarray,
    grid_size: int,
) -> tuple[np.ndarray, int, int]:
    height, width = patch.shape
    base_r = grid_size // 2 - height // 2
    base_c = grid_size // 2 - width // 2
    r0 = min(max(base_r + random.choice((-1, 0, 1)), 0), grid_size - height)
    c0 = min(max(base_c + random.choice((-1, 0, 1)), 0), grid_size - width)
    grid = np.zeros((grid_size, grid_size), dtype=int)
    grid[r0 : r0 + height, c0 : c0 + width] = patch
    return grid, r0, c0


def _generate_attempt() -> np.ndarray | None:
    grid_size = random.choice((11, 13, 15))
    axis_kind = random.choice(("vertical", "horizontal"))
    side = random.choice((5, 7, 9))
    patch = _build_symmetric_patch(side, side, axis_kind)
    if patch is None:
        return None

    complete_swapped, r0, c0 = _embed_patch(patch, grid_size)
    complete_input = _swap_red_blue(complete_swapped)

    candidates = _hole_candidates(patch, axis_kind)
    hole_count = random.randint(2, 5)
    if len(candidates) < hole_count:
        return None

    grid = complete_input.copy()
    for pr, pc in candidates[:hole_count]:
        grid[r0 + pr, c0 + pc] = HOLE

    if not np.array_equal(main(grid), complete_input):
        return None
    swapped = _swap_red_blue(grid)
    detected_axis, _axis_sum = _detect_axis(swapped)
    if detected_axis != axis_kind:
        return None
    if _exact_reflection(grid, axis_kind, (_pattern_bbox(grid)[1] + _pattern_bbox(grid)[3]) if axis_kind == "vertical" else (_pattern_bbox(grid)[0] + _pattern_bbox(grid)[2])):
        return None
    if not all(np.any(complete_input == color) for color in PATTERN_COLORS):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
