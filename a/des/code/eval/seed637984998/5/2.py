from __future__ import annotations

import random

import numpy as np


BLACK = 0
DIRECTIONS = ("up", "down", "left", "right")
SIZE_CHOICES = (3, 4)
ALL_COLORS = tuple(range(10))

GRID_H_RANGE = (20, 24)
GRID_W_RANGE = (20, 24)
OBJECT_COUNT_RANGE = (4, 7)
PLACEMENT_PADDING = 1
PLACEMENT_MARGIN = 3
MAX_GENERATION_RETRIES = 400
MAX_OBJECT_ATTEMPTS = 250

PATTERN_MASKS_3 = [np.array([[True]], dtype=bool)]
PATTERN_MASKS_4 = [
    np.array([[True, True], [False, False]], dtype=bool),
    np.array([[True, False], [True, False]], dtype=bool),
    np.array([[True, True], [True, False]], dtype=bool),
    np.array([[True, True], [False, True]], dtype=bool),
    np.array([[True, False], [True, True]], dtype=bool),
    np.array([[False, True], [True, True]], dtype=bool),
    np.array([[True, True], [True, True]], dtype=bool),
]


def _majority_non_black(values: np.ndarray) -> int:
    vals = [int(v) for v in values if int(v) != BLACK]
    assert vals, "expected non-black wall pixels on each edge"
    counts: dict[int, int] = {}
    for v in vals:
        counts[v] = counts.get(v, 0) + 1
    return max(counts.items(), key=lambda item: item[1])[0]


def _wall_colors(grid: np.ndarray) -> dict[str, int]:
    return {
        "up": _majority_non_black(grid[0, :]),
        "down": _majority_non_black(grid[-1, :]),
        "left": _majority_non_black(grid[:, 0]),
        "right": _majority_non_black(grid[:, -1]),
    }


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = int(rr + dr), int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _shell_direction(shell_mask: np.ndarray) -> str | None:
    h, w = shell_mask.shape
    if h != w or h not in SIZE_CHOICES:
        return None
    if int(shell_mask.sum()) != 4 * h - 5:
        return None
    if np.any(shell_mask[1:-1, 1:-1]):
        return None
    if not (bool(shell_mask[0, 0]) and bool(shell_mask[0, -1]) and bool(shell_mask[-1, 0]) and bool(shell_mask[-1, -1])):
        return None

    side_missing = {
        "up": sum(not bool(shell_mask[0, c]) for c in range(1, w - 1)),
        "down": sum(not bool(shell_mask[h - 1, c]) for c in range(1, w - 1)),
        "left": sum(not bool(shell_mask[r, 0]) for r in range(1, h - 1)),
        "right": sum(not bool(shell_mask[r, w - 1]) for r in range(1, h - 1)),
    }
    if sorted(side_missing.values()) != [0, 0, 0, 1]:
        return None
    return next(side for side, missing in side_missing.items() if missing == 1)


def _find_shell_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    walls = _wall_colors(g)
    objects: list[dict] = []

    for wall_color in set(walls.values()):
        for comp in _components(g == int(wall_color)):
            if any(r in (0, h - 1) or c in (0, w - 1) for r, c in comp):
                continue
            r0, c0, r1, c1 = _bbox(comp)
            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            shell_mask = patch == int(wall_color)
            direction = _shell_direction(shell_mask)
            if direction is None or int(wall_color) != int(walls[direction]):
                continue

            border_mask = np.zeros_like(shell_mask, dtype=bool)
            border_mask[0, :] = True
            border_mask[-1, :] = True
            border_mask[:, 0] = True
            border_mask[:, -1] = True

            pattern_mask = (patch != BLACK) & (patch != int(wall_color))
            if not bool(pattern_mask.any()) or bool(np.any(pattern_mask & border_mask)):
                continue

            pattern_cells = [
                (int(r0 + rr), int(c0 + cc), int(patch[rr, cc]))
                for rr, cc in np.argwhere(pattern_mask)
            ]
            objects.append(
                {
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "direction": direction,
                    "shell_color": int(wall_color),
                    "pattern_cells": pattern_cells,
                }
            )

    objects.sort(key=lambda obj: obj["bbox"])
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each inner pattern toward the wall indicated by its shell gap.

    The shell color redundantly matches the destination wall color. After
    projection, all original shell and interior pattern pixels are cleared.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    objects = _find_shell_objects(g)
    assert objects, "expected at least one gapped shell object"

    out = g.copy()
    projected: list[tuple[int, int, int]] = []

    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = BLACK

        rows = [r for r, _c, _v in obj["pattern_cells"]]
        cols = [c for _r, c, _v in obj["pattern_cells"]]
        if obj["direction"] == "up":
            shift_r, shift_c = 1 - min(rows), 0
        elif obj["direction"] == "down":
            shift_r, shift_c = (h - 2) - max(rows), 0
        elif obj["direction"] == "left":
            shift_r, shift_c = 0, 1 - min(cols)
        else:
            shift_r, shift_c = 0, (w - 2) - max(cols)

        for r, c, color in obj["pattern_cells"]:
            rr = int(r + shift_r)
            cc = int(c + shift_c)
            assert 0 <= rr < h and 0 <= cc < w
            projected.append((rr, cc, int(color)))

    for rr, cc, color in projected:
        out[rr, cc] = int(color)
    return out


def _base_grid(height: int, width: int, walls: dict[str, int]) -> np.ndarray:
    g = np.full((height, width), BLACK, dtype=int)
    g[0, :] = int(walls["up"])
    g[-1, :] = int(walls["down"])
    g[1:-1, 0] = int(walls["left"])
    g[1:-1, -1] = int(walls["right"])
    return g


def _pattern_masks(size: int) -> list[np.ndarray]:
    return PATTERN_MASKS_3 if size == 3 else PATTERN_MASKS_4


def _gap_index(size: int) -> int:
    return 1 if size == 3 else random.choice((1, 2))


def _build_patch(size: int, direction: str, gap_idx: int, shell_color: int, pattern_color: int, pattern_mask: np.ndarray) -> np.ndarray:
    patch = np.full((size, size), BLACK, dtype=int)
    patch[0, :] = int(shell_color)
    patch[-1, :] = int(shell_color)
    patch[:, 0] = int(shell_color)
    patch[:, -1] = int(shell_color)

    if direction == "up":
        patch[0, gap_idx] = BLACK
    elif direction == "down":
        patch[-1, gap_idx] = BLACK
    elif direction == "left":
        patch[gap_idx, 0] = BLACK
    else:
        patch[gap_idx, -1] = BLACK

    inner = patch[1 : size - 1, 1 : size - 1]
    inner[pattern_mask] = int(pattern_color)
    return patch


def _pattern_cells(top: int, left: int, pattern_mask: np.ndarray, pattern_color: int) -> list[tuple[int, int, int]]:
    cells = []
    for rr, cc in np.argwhere(pattern_mask):
        cells.append((int(top + 1 + rr), int(left + 1 + cc), int(pattern_color)))
    return cells


def _project_pattern(cells: list[tuple[int, int, int]], direction: str, height: int, width: int) -> list[tuple[int, int, int]]:
    rows = [r for r, _c, _v in cells]
    cols = [c for _r, c, _v in cells]
    if direction == "up":
        shift_r, shift_c = 1 - min(rows), 0
    elif direction == "down":
        shift_r, shift_c = (height - 2) - max(rows), 0
    elif direction == "left":
        shift_r, shift_c = 0, 1 - min(cols)
    else:
        shift_r, shift_c = 0, (width - 2) - max(cols)
    return [(int(r + shift_r), int(c + shift_c), int(v)) for r, c, v in cells]


def _boxes_conflict(a: tuple[int, int, int, int], b: tuple[int, int, int, int], padding: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + padding < br0
        or br1 + padding < ar0
        or ac1 + padding < bc0
        or bc1 + padding < ac0
    )


def _direction_list(n_objects: int) -> list[str]:
    dirs = list(DIRECTIONS)
    while len(dirs) < n_objects:
        dirs.append(random.choice(DIRECTIONS))
    random.shuffle(dirs)
    return dirs


def _attempt_generate() -> tuple[np.ndarray, np.ndarray] | None:
    n_objects = random.randint(*OBJECT_COUNT_RANGE)
    min_side = max(GRID_H_RANGE[0], GRID_W_RANGE[0], 18 + max(0, n_objects - 5))
    height = random.randint(min_side, GRID_H_RANGE[1])
    width = random.randint(min_side, GRID_W_RANGE[1])

    wall_colors = random.sample(list(range(1, 10)), 4)
    walls = dict(zip(DIRECTIONS, wall_colors))
    pattern_palette = [color for color in ALL_COLORS if color not in wall_colors and color != BLACK]
    grid = _base_grid(height, width, walls)
    expected = _base_grid(height, width, walls)

    boxes: list[tuple[int, int, int, int]] = []
    used_output: set[tuple[int, int]] = set()

    for direction in _direction_list(n_objects):
        placed = False
        for _ in range(MAX_OBJECT_ATTEMPTS):
            size = random.choice(SIZE_CHOICES)
            top_lo = PLACEMENT_MARGIN
            left_lo = PLACEMENT_MARGIN
            top_hi = height - size - PLACEMENT_MARGIN
            left_hi = width - size - PLACEMENT_MARGIN
            if top_hi < top_lo or left_hi < left_lo:
                return None

            top = random.randint(top_lo, top_hi)
            left = random.randint(left_lo, left_hi)
            box = (int(top), int(left), int(top + size - 1), int(left + size - 1))
            if any(_boxes_conflict(box, other, PLACEMENT_PADDING) for other in boxes):
                continue

            pattern_mask = random.choice(_pattern_masks(size))
            pattern_color = random.choice(pattern_palette)
            gap_idx = _gap_index(size)
            patch = _build_patch(size, direction, gap_idx, walls[direction], pattern_color, pattern_mask)
            pattern_cells = _pattern_cells(top, left, pattern_mask, pattern_color)
            projected = _project_pattern(pattern_cells, direction, height, width)

            projected_positions = {(r, c) for r, c, _ in projected}
            if any((r, c) in used_output for r, c in projected_positions):
                continue
            if any(r in (0, height - 1) or c in (0, width - 1) for r, c in projected_positions):
                continue

            target = grid[top : top + size, left : left + size]
            if not bool(np.all(target == BLACK)):
                continue

            grid[top : top + size, left : left + size] = patch
            for r, c, color in projected:
                expected[r, c] = int(color)
                used_output.add((int(r), int(c)))
            boxes.append(box)
            placed = True
            break
        if not placed:
            return None

    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_RETRIES):
        built = _attempt_generate()
        if built is None:
            continue
        grid, expected = built
        try:
            actual = main(grid)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
