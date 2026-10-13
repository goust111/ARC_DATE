import random
from collections import Counter

import numpy as np


BACKGROUND = 0
ARROW_COLORS = (1, 2, 3, 4)
FRAME = 5
MAROON = 9
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == color
    seen = np.zeros_like(mask, dtype=bool)
    height, width = grid.shape
    components: list[np.ndarray] = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _build_arrow(direction: str, r0: int, c0: int) -> dict:
    if direction == "right":
        coords = [(r0 + 1, c0 + k) for k in range(4)]
        coords += [(r0, c0 + 2), (r0 + 2, c0 + 2)]
        tip = (r0 + 1, c0 + 3)
        axis = ("H", r0 + 1)
        bbox = (r0, c0, r0 + 2, c0 + 3)
    elif direction == "left":
        coords = [(r0 + 1, c0 + k) for k in range(4)]
        coords += [(r0, c0 + 1), (r0 + 2, c0 + 1)]
        tip = (r0 + 1, c0)
        axis = ("H", r0 + 1)
        bbox = (r0, c0, r0 + 2, c0 + 3)
    elif direction == "down":
        coords = [(r0 + k, c0 + 1) for k in range(4)]
        coords += [(r0 + 2, c0), (r0 + 2, c0 + 2)]
        tip = (r0 + 3, c0 + 1)
        axis = ("V", c0 + 1)
        bbox = (r0, c0, r0 + 3, c0 + 2)
    else:
        assert direction == "up"
        coords = [(r0 + k, c0 + 1) for k in range(4)]
        coords += [(r0 + 1, c0), (r0 + 1, c0 + 2)]
        tip = (r0, c0 + 1)
        axis = ("V", c0 + 1)
        bbox = (r0, c0, r0 + 3, c0 + 2)

    return {
        "direction": direction,
        "coords": np.array(coords, dtype=int),
        "tip": tip,
        "axis": axis,
        "bbox": bbox,
    }


def _detect_arrow(coords: np.ndarray) -> dict:
    box = _bbox(coords)
    r0, c0, r1, c1 = box
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    row_counts = Counter(int(r) for r, _ in coords)
    col_counts = Counter(int(c) for _, c in coords)

    if height == 3 and width == 4:
        bar_rows = [r for r, count in row_counts.items() if count == 4]
        cross_cols = [c for c, count in col_counts.items() if count == 3]
        assert len(bar_rows) == 1 and len(cross_cols) == 1, "invalid horizontal arrow"
        bar_row = bar_rows[0]
        cross_col = cross_cols[0]
        offset = cross_col - c0
        assert offset in (1, 2), "horizontal crossbar must be near one end"
        direction = "left" if offset == 1 else "right"
        tip = (bar_row, c0 if direction == "left" else c1)
        axis = ("H", bar_row)
    else:
        assert height == 4 and width == 3, "arrow bbox must be 3x4 or 4x3"
        bar_cols = [c for c, count in col_counts.items() if count == 4]
        cross_rows = [r for r, count in row_counts.items() if count == 3]
        assert len(bar_cols) == 1 and len(cross_rows) == 1, "invalid vertical arrow"
        bar_col = bar_cols[0]
        cross_row = cross_rows[0]
        offset = cross_row - r0
        assert offset in (1, 2), "vertical crossbar must be near one end"
        direction = "up" if offset == 1 else "down"
        tip = (r0 if direction == "up" else r1, bar_col)
        axis = ("V", bar_col)

    return {
        "direction": direction,
        "coords": np.array(sorted(map(tuple, coords)), dtype=int),
        "tip": tip,
        "axis": axis,
        "bbox": box,
    }


def _forward_cells(arrow: dict, shape: tuple[int, int]) -> list[tuple[int, int]]:
    height, width = shape
    dr, dc = DIRS[arrow["direction"]]
    r, c = arrow["tip"]
    cells = []
    while True:
        r += dr
        c += dc
        if r <= 0 or r >= height - 1 or c <= 0 or c >= width - 1:
            break
        cells.append((r, c))
    return cells


def _opposite_border(arrow: dict, shape: tuple[int, int]) -> tuple[int, int]:
    height, width = shape
    r, c = arrow["tip"]
    if arrow["direction"] == "right":
        return r, 0
    if arrow["direction"] == "left":
        return r, width - 1
    if arrow["direction"] == "down":
        return 0, c
    return height - 1, c


def _border_hit(arrow: dict, shape: tuple[int, int]) -> tuple[int, int]:
    height, width = shape
    r, c = arrow["tip"]
    if arrow["direction"] == "right":
        return r, width - 1
    if arrow["direction"] == "left":
        return r, 0
    if arrow["direction"] == "down":
        return height - 1, c
    return 0, c


def _push_along_arrow(state: np.ndarray, arrow: dict) -> None:
    direction = arrow["direction"]
    dr, dc = DIRS[direction]
    tip_r, tip_c = arrow["tip"]
    positions = []
    if direction == "right":
        row = tip_r
        for c in range(tip_c + 1, state.shape[1] - 1):
            if state[row, c] == MAROON:
                positions.append((row, c))
        positions.sort(key=lambda pos: pos[1], reverse=True)
    elif direction == "left":
        row = tip_r
        for c in range(1, tip_c):
            if state[row, c] == MAROON:
                positions.append((row, c))
        positions.sort(key=lambda pos: pos[1])
    elif direction == "down":
        col = tip_c
        for r in range(tip_r + 1, state.shape[0] - 1):
            if state[r, col] == MAROON:
                positions.append((r, col))
        positions.sort(key=lambda pos: pos[0], reverse=True)
    else:
        col = tip_c
        for r in range(1, tip_r):
            if state[r, col] == MAROON:
                positions.append((r, col))
        positions.sort(key=lambda pos: pos[0])

    for start_r, start_c in positions:
        if state[start_r, start_c] != MAROON:
            continue
        state[start_r, start_c] = BACKGROUND
        r, c = start_r, start_c
        while state[r + dr, c + dc] == BACKGROUND:
            r += dr
            c += dc
        state[r, c] = MAROON


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push maroon pixels along each asymmetric T-arrow's pointing axis, then mark only the frame endpoints reached by unobstructed rays.

    Input structure:
    - The grid has a one-pixel solid outer frame and a black interior background.
    - Inside the frame are 3 or 4 distinct-color six-pixel asymmetric T arrows plus scattered maroon single pixels.
    - Each arrow uses a length-4 bar and a length-3 offset crossbar whose nearer end indicates the tip direction.

    Rule:
    - Detect every arrow and decode its direction from the crossbar offset.
    - On the arrow's forward row or column, slide every maroon pixel ahead of the tip in that same direction, processing the one nearest the border first and stopping each one just before the first obstacle.
    - After all pushes, cast a ray from each tip toward the border; if all interior cells on that ray are background, paint the hit border cell with the arrow color and paint the opposite border cell on the same axis black, otherwise leave the frame unchanged for that arrow.

    Key insight:
    - The arrow direction controls both the maroon-push and the later ray, so a pushed maroon can end up blocking the very border mark that the same arrow would otherwise create.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape
    assert height >= 7 and width >= 7, "grid is too small for the puzzle objects"

    frame = int(g[0, 0])
    assert np.all(g[0, :] == frame) and np.all(g[-1, :] == frame), "top and bottom must be solid frame"
    assert np.all(g[:, 0] == frame) and np.all(g[:, -1] == frame), "left and right must be solid frame"

    arrow_components = []
    for color in sorted(set(map(int, np.unique(g))) & set(ARROW_COLORS)):
        for comp in _components_of_color(g, color):
            assert comp.shape[0] == 6, "every arrow component must have six pixels"
            arrow = _detect_arrow(comp)
            arrow["color"] = color
            arrow_components.append(arrow)
    assert 3 <= len(arrow_components) <= 4, "expected 3 or 4 arrows"

    state = g.copy()
    for arrow in sorted(arrow_components, key=lambda item: item["color"]):
        _push_along_arrow(state, arrow)

    assignments: dict[tuple[int, int], int] = {}
    for arrow in arrow_components:
        blocked = False
        r, c = arrow["tip"]
        dr, dc = DIRS[arrow["direction"]]
        while True:
            r += dr
            c += dc
            if r <= 0 or r >= height - 1 or c <= 0 or c >= width - 1:
                hit = (r, c)
                break
            if state[r, c] != BACKGROUND:
                blocked = True
                break
        if blocked:
            continue
        opposite = _opposite_border(arrow, g.shape)
        assert hit not in assignments or assignments[hit] == arrow["color"], "conflicting frame hit"
        assert opposite not in assignments or assignments[opposite] == BACKGROUND, "conflicting opposite mark"
        assignments[hit] = arrow["color"]
        assignments[opposite] = BACKGROUND

    out = state.copy()
    for coord, value in assignments.items():
        out[coord] = value
    return out


def _candidate_arrow(direction: str, shape: tuple[int, int], min_forward: int) -> dict | None:
    height, width = shape
    if direction == "right":
        max_c0 = width - 5 - min_forward
        if max_c0 < 1:
            return None
        r0 = random.randint(1, height - 4)
        c0 = random.randint(1, max_c0)
    elif direction == "left":
        min_c0 = 1 + min_forward
        max_c0 = width - 5
        if min_c0 > max_c0:
            return None
        r0 = random.randint(1, height - 4)
        c0 = random.randint(min_c0, max_c0)
    elif direction == "down":
        max_r0 = height - 5 - min_forward
        if max_r0 < 1:
            return None
        r0 = random.randint(1, max_r0)
        c0 = random.randint(1, width - 4)
    else:
        assert direction == "up"
        min_r0 = 1 + min_forward
        max_r0 = height - 5
        if min_r0 > max_r0:
            return None
        r0 = random.randint(min_r0, max_r0)
        c0 = random.randint(1, width - 4)
    return _build_arrow(direction, r0, c0)


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    return not (a[2] + gap < b[0] or b[2] + gap < a[0] or a[3] + gap < b[1] or b[3] + gap < a[1])


def _placeable(candidate: dict, arrows: list[dict], shape: tuple[int, int]) -> bool:
    for other in arrows:
        if _boxes_too_close(candidate["bbox"], other["bbox"], gap=1):
            return False
        if candidate["axis"][0] == "H" and other["axis"][0] == "H" and candidate["axis"][1] == other["axis"][1]:
            return False
        if candidate["axis"][0] == "V" and other["axis"][0] == "V" and candidate["axis"][1] == other["axis"][1]:
            return False
        if _border_hit(candidate, shape) in {_border_hit(other, shape), _opposite_border(other, shape)}:
            return False
        if _opposite_border(candidate, shape) in {_border_hit(other, shape), _opposite_border(other, shape)}:
            return False
        other_cells = {tuple(pos) for pos in other["coords"]}
        cand_cells = {tuple(pos) for pos in candidate["coords"]}
        if any(cell in other_cells for cell in _forward_cells(candidate, shape)):
            return False
        if any(cell in cand_cells for cell in _forward_cells(other, shape)):
            return False
    return True


def _choose_maroons(corridor: list[tuple[int, int]], count: int) -> list[tuple[int, int]] | None:
    if count == 0:
        return []
    usable = corridor[: len(corridor) - count]
    if len(usable) < count:
        return None
    if count == 1:
        return [random.choice(usable)]

    indices = list(range(len(usable)))
    random.shuffle(indices)
    best = None
    for i in indices:
        for j in indices:
            if i >= j:
                continue
            if j - i >= 2:
                return sorted([usable[i], usable[j]])
            if best is None:
                best = [usable[i], usable[j]]
    if best is None:
        return None
    return sorted(best)


def _extra_maroons(grid: np.ndarray, arrows: list[dict], occupied: set[tuple[int, int]]) -> bool:
    horizontal_rows = {arrow["axis"][1] for arrow in arrows if arrow["axis"][0] == "H"}
    vertical_cols = {arrow["axis"][1] for arrow in arrows if arrow["axis"][0] == "V"}
    forbidden = set()
    for arrow in arrows:
        forbidden.update(_forward_cells(arrow, grid.shape))
    candidates = []
    for r in range(1, grid.shape[0] - 1):
        for c in range(1, grid.shape[1] - 1):
            if (r, c) in occupied or (r, c) in forbidden:
                continue
            if r in horizontal_rows or c in vertical_cols:
                continue
            candidates.append((r, c))
    random.shuffle(candidates)
    extras = random.randint(0, 3)
    placed = 0
    for r, c in candidates:
        if any(max(abs(r - rr), abs(c - cc)) <= 1 for rr, cc in occupied if grid[rr, cc] == MAROON):
            continue
        grid[r, c] = MAROON
        occupied.add((r, c))
        placed += 1
        if placed == extras:
            break
    return placed == extras


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(13, 18)
    width = random.randint(13, 18)
    grid = np.full((height, width), BACKGROUND, dtype=int)
    grid[0, :] = FRAME
    grid[-1, :] = FRAME
    grid[:, 0] = FRAME
    grid[:, -1] = FRAME

    num_arrows = random.choice((3, 4))
    directions = random.sample(list(DIRS), num_arrows)
    colors = random.sample(list(ARROW_COLORS), num_arrows)
    blocked_count = random.randint(1, num_arrows - 1)
    blocked_directions = set(random.sample(directions, blocked_count))
    maroon_counts = {
        direction: 0 if direction not in blocked_directions else random.choice((1, 2))
        for direction in directions
    }

    arrows = []
    for direction, color in zip(directions, colors):
        min_forward = 0 if maroon_counts[direction] == 0 else maroon_counts[direction] + 2
        placed = None
        for _ in range(150):
            candidate = _candidate_arrow(direction, grid.shape, min_forward)
            if candidate is None:
                return None
            if not _placeable(candidate, arrows, grid.shape):
                continue
            candidate["color"] = color
            placed = candidate
            break
        if placed is None:
            return None
        arrows.append(placed)

    occupied: set[tuple[int, int]] = set()
    for arrow in arrows:
        for r, c in arrow["coords"]:
            grid[r, c] = arrow["color"]
            occupied.add((int(r), int(c)))

    for arrow in arrows:
        corridor = _forward_cells(arrow, grid.shape)
        chosen = _choose_maroons(corridor, maroon_counts[arrow["direction"]])
        if chosen is None:
            return None
        for r, c in chosen:
            if (r, c) in occupied:
                return None
            grid[r, c] = MAROON
            occupied.add((r, c))

    if not _extra_maroons(grid, arrows, occupied):
        return None

    solved = main(grid)
    changed_border = np.argwhere((grid != solved) & (
        (np.indices(grid.shape)[0] == 0)
        | (np.indices(grid.shape)[0] == grid.shape[0] - 1)
        | (np.indices(grid.shape)[1] == 0)
        | (np.indices(grid.shape)[1] == grid.shape[1] - 1)
    ))
    if changed_border.shape[0] < 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
