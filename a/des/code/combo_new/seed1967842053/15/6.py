import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each red object by whether it touches the grid edge, a grey wall, or both.

    Input structure:
    The background is black, grey one-cell-wide horizontal or vertical segments act as
    internal walls, and every red 4-connected component is a separate object. Red
    objects may touch the outside border of the grid, be 4-adjacent to a grey wall, or
    satisfy both conditions.

    Rule:
    Copy the grid, keep black background and grey walls unchanged, and replace each red
    component independently: edge-only contact becomes blue, grey-wall-only contact
    becomes green, and contact with both the outer edge and a grey wall becomes yellow.

    Key insight:
    The red shapes themselves do not determine the output color; each component is
    classified only by its visible contact with the two boundary types, the grid edge
    and the grey internal walls.
    """
    arr = np.asarray(grid)
    out = arr.copy()
    rows, cols = arr.shape
    seen = np.zeros(arr.shape, dtype=bool)

    for sr in range(rows):
        for sc in range(cols):
            if arr[sr, sc] != RED or seen[sr, sc]:
                continue

            q = deque([(sr, sc)])
            seen[sr, sc] = True
            cells = []
            touches_edge = False
            touches_grey = False

            while q:
                r, c = q.popleft()
                cells.append((r, c))
                if r == 0 or c == 0 or r == rows - 1 or c == cols - 1:
                    touches_edge = True

                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < rows and 0 <= nc < cols):
                        continue
                    if arr[nr, nc] == GREY:
                        touches_grey = True
                    elif arr[nr, nc] == RED and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            if touches_edge and touches_grey:
                color = YELLOW
            elif touches_grey:
                color = GREEN
            else:
                color = BLUE

            for r, c in cells:
                out[r, c] = color

    return out


def _stamp(grid, cells, color):
    for r, c in cells:
        grid[r, c] = color


def _free_for_object(grid, cells, rows, cols):
    for r, c in cells:
        if not (0 <= r < rows and 0 <= c < cols):
            return False
        if grid[r, c] != BLACK:
            return False
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == RED:
                    return False
    return True


def _random_shape(anchor, rows, cols, max_cells=5):
    cells = {anchor}
    target = random.randint(1, max_cells)
    for _ in range(target * 8):
        if len(cells) >= target:
            break
        r, c = random.choice(tuple(cells))
        random.shuffle(_DIRS)
        for dr, dc in _DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                cells.add((nr, nc))
                break
    return cells


def _place_shape(grid, candidates, rows, cols, max_cells=5):
    random.shuffle(candidates)
    for anchor in candidates:
        for _ in range(20):
            cells = _random_shape(anchor, rows, cols, max_cells)
            if anchor not in cells:
                cells.add(anchor)
            if _free_for_object(grid, cells, rows, cols):
                _stamp(grid, cells, RED)
                return True
    return False


def _place_single(grid, candidates, rows, cols):
    random.shuffle(candidates)
    for cell in candidates:
        if _free_for_object(grid, {cell}, rows, cols):
            _stamp(grid, {cell}, RED)
            return True
    return False


def _edge_candidates(rows, cols):
    cells = []
    for c in range(1, cols - 1):
        cells.append((0, c))
        cells.append((rows - 1, c))
    for r in range(1, rows - 1):
        cells.append((r, 0))
        cells.append((r, cols - 1))
    return cells


def _edge_only_candidates(grid, rows, cols):
    out = []
    for r, c in _edge_candidates(rows, cols):
        if grid[r, c] != BLACK:
            continue
        near_grey = False
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == GREY:
                near_grey = True
                break
        if not near_grey:
            out.append((r, c))
    return out


def _wall_only_candidates(grid, rows, cols):
    out = []
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if grid[r, c] != BLACK:
                continue
            if any(
                grid[r + dr, c + dc] == GREY
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
            ):
                out.append((r, c))
    return out


def _both_candidates(grid, rows, cols):
    out = []
    for r, c in _edge_candidates(rows, cols):
        if grid[r, c] != BLACK:
            continue
        if any(
            0 <= r + dr < rows
            and 0 <= c + dc < cols
            and grid[r + dr, c + dc] == GREY
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
        ):
            out.append((r, c))
    return out


def _add_walls(grid, rows, cols):
    orientation = random.choice(("vertical", "horizontal"))
    if orientation == "vertical":
        c = random.choice((1, cols - 2))
        r0 = random.randint(0, 2)
        r1 = random.randint(rows - 3, rows - 1)
        grid[r0 : r1 + 1, c] = GREY
        other_c = random.randint(3, cols - 4)
        a = random.randint(2, rows - 6)
        b = random.randint(a + 2, min(rows - 2, a + 7))
        grid[a : b + 1, other_c] = GREY
        if random.random() < 0.6:
            rr = random.randint(3, rows - 4)
            cc0 = random.randint(3, max(3, cols // 2 - 1))
            cc1 = random.randint(max(cc0 + 2, cols // 2), cols - 4)
            grid[rr, cc0 : cc1 + 1] = GREY
    else:
        r = random.choice((1, rows - 2))
        c0 = random.randint(0, 2)
        c1 = random.randint(cols - 3, cols - 1)
        grid[r, c0 : c1 + 1] = GREY
        other_r = random.randint(3, rows - 4)
        a = random.randint(2, cols - 6)
        b = random.randint(a + 2, min(cols - 2, a + 7))
        grid[other_r, a : b + 1] = GREY
        if random.random() < 0.6:
            cc = random.randint(3, cols - 4)
            rr0 = random.randint(3, max(3, rows // 2 - 1))
            rr1 = random.randint(max(rr0 + 2, rows // 2), rows - 4)
            grid[rr0 : rr1 + 1, cc] = GREY


def generate_input() -> np.ndarray:
    rows = random.randint(14, 30)
    cols = random.randint(14, 30)
    grid = np.full((rows, cols), BLACK, dtype=int)
    _add_walls(grid, rows, cols)

    branch_counts = {
        "edge": random.randint(1, 3),
        "wall": random.randint(1, 3),
        "both": random.randint(1, 3),
    }
    extra = random.randint(0, 3)
    for _ in range(extra):
        branch_counts[random.choice(("edge", "wall", "both"))] += 1

    for _ in range(branch_counts["both"]):
        if not _place_single(grid, _both_candidates(grid, rows, cols), rows, cols):
            break

    for _ in range(branch_counts["wall"]):
        if not _place_shape(grid, _wall_only_candidates(grid, rows, cols), rows, cols, 4):
            break

    for _ in range(branch_counts["edge"]):
        if not _place_shape(grid, _edge_only_candidates(grid, rows, cols), rows, cols, 4):
            break

    return grid


_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]

