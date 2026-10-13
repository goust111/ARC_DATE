from __future__ import annotations

import random
from collections import deque

import numpy as np


GRAY = 5
GREEN = 3
BLUE = 1
RED = 2
YELLOW = 4
ORANGE = 7

HEIGHT_RANGE = (12, 22)
WIDTH_RANGE = (12, 22)
OBJECT_COUNT_RANGE = (3, 6)
OBJECT_MARGIN = 1
MAX_SHAPE_EXTENT = 6
GEN_ATTEMPTS = 400

DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
BRANCHES = ("gray", "green", "tie")


def _normalize_cells(cells: set[tuple[int, int]] | list[tuple[int, int]]) -> list[tuple[int, int]]:
    rows = [int(r) for r, _c in cells]
    cols = [int(c) for _r, c in cells]
    min_r = min(rows)
    min_c = min(cols)
    return sorted((int(r - min_r), int(c - min_c)) for r, c in cells)


def _bbox_size(cells: list[tuple[int, int]]) -> tuple[int, int]:
    return max(r for r, _c in cells) + 1, max(c for _r, c in cells) + 1


def _parity_delta(cells: list[tuple[int, int]]) -> int:
    return sum(1 if (r + c) % 2 == 0 else -1 for r, c in cells)


def _transformed_cells(
    cells: list[tuple[int, int]],
    rot: int,
    flip_h: bool,
    flip_v: bool,
) -> list[tuple[int, int]]:
    pts = [(int(r), int(c)) for r, c in cells]
    for _ in range(rot % 4):
        pts = [(int(c), int(-r)) for r, c in pts]
    if flip_h:
        pts = [(int(r), int(-c)) for r, c in pts]
    if flip_v:
        pts = [(int(-r), int(c)) for r, c in pts]
    return _normalize_cells(pts)


def _random_polyomino(area: int, rng: random.Random) -> list[tuple[int, int]] | None:
    cells: set[tuple[int, int]] = {(0, 0)}

    while len(cells) < area:
        frontier: list[tuple[int, int]] = []
        weights: list[int] = []
        seen: set[tuple[int, int]] = set()

        for r, c in cells:
            for dr, dc in DIR4:
                nr, nc = int(r + dr), int(c + dc)
                cand = (nr, nc)
                if cand in cells or cand in seen:
                    continue
                seen.add(cand)
                trial = _normalize_cells(list(cells) + [cand])
                h, w = _bbox_size(trial)
                if max(h, w) > MAX_SHAPE_EXTENT:
                    continue
                adj = sum((nr + rr, nc + cc) in cells for rr, cc in DIR4)
                frontier.append(cand)
                weights.append(1 + adj * adj)

        if not frontier:
            return None

        pick = rng.choices(frontier, weights=weights, k=1)[0]
        cells.add((int(pick[0]), int(pick[1])))

    transformed = _transformed_cells(
        _normalize_cells(cells),
        rot=rng.randrange(4),
        flip_h=rng.random() < 0.5,
        flip_v=rng.random() < 0.5,
    )
    h, w = _bbox_size(transformed)
    if max(h, w) > MAX_SHAPE_EXTENT:
        return None
    return transformed


def _shape_for_branch(branch: str, rng: random.Random) -> list[tuple[int, int]] | None:
    for _ in range(200):
        if branch == "tie":
            area = rng.choice((2, 4, 6, 8))
        else:
            area = rng.choice((3, 5, 7, 9))
        cells = _random_polyomino(area, rng)
        if cells is None:
            continue
        delta = _parity_delta(cells)
        if branch == "tie" and delta == 0:
            return cells
        if branch != "tie" and delta != 0:
            return cells
    return None


def _make_checkerboard(height: int, width: int, gray_on_even: bool) -> np.ndarray:
    rows, cols = np.indices((height, width))
    parity = (rows + cols) % 2
    gray_parity = 0 if gray_on_even else 1
    return np.where(parity == gray_parity, GRAY, GREEN).astype(int)


def _foreground_and_checker_colors(grid: np.ndarray) -> tuple[int, int, int]:
    g = np.asarray(grid, dtype=int)
    rows, cols = np.indices(g.shape)
    parity = (rows + cols) % 2
    candidates: list[tuple[int, int, int]] = []

    for fg in np.unique(g):
        visible = g != int(fg)
        even_vals = np.unique(g[(parity == 0) & visible])
        odd_vals = np.unique(g[(parity == 1) & visible])
        if even_vals.size != 1 or odd_vals.size != 1:
            continue
        even_color = int(even_vals[0])
        odd_color = int(odd_vals[0])
        if even_color == odd_color:
            continue
        candidates.append((int(fg), even_color, odd_color))

    assert len(candidates) == 1, "expected one foreground color over a visible checkerboard"
    fg, even_color, odd_color = candidates[0]
    assert {even_color, odd_color} == {GRAY, GREEN}, "checkerboard must use the gray and green colors"
    return fg, even_color, odd_color


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    mask = np.asarray(grid == int(color), dtype=bool)
    visited = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    components: list[list[tuple[int, int]]] = []

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or visited[r, c]:
                continue
            visited[r, c] = True
            queue: deque[tuple[int, int]] = deque([(int(r), int(c))])
            coords: list[tuple[int, int]] = []

            while queue:
                rr, cc = queue.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = int(rr + dr), int(cc + dc)
                    if not (0 <= nr < height and 0 <= nc < width):
                        continue
                    if not mask[nr, nc] or visited[nr, nc]:
                        continue
                    visited[nr, nc] = True
                    queue.append((nr, nc))

            components.append(coords)

    return components


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each blue object by whether it covers more gray or more green cells of the checkerboard.

    Input structure:
    - The background is a full 1x1 checkerboard made of gray and green cells.
    - Several 4-connected foreground objects are drawn in one solid blue color on top of that checkerboard.
    - The blue objects may have arbitrary connected shapes and hide part of the alternating background pattern.

    Rule:
    - Infer the checkerboard parity from the visible gray and green background cells.
    - For each blue connected component, determine for every covered pixel whether the hidden background cell there would be gray or green.
    - Count the component's covered gray cells and covered green cells.
    - Recolor the whole component red if gray is the majority, orange if green is the majority, and yellow if the counts are equal.
    - Leave the checkerboard background unchanged.

    Key insight:
    The checkerboard acts as a parity grid: a component's output color depends only on how its shape sits on the hidden gray/green parity classes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    fg_color, even_color, odd_color = _foreground_and_checker_colors(g)
    gray_parity = 0 if even_color == GRAY else 1

    out = g.copy()
    components = _components_of_color(g, fg_color)
    assert components, "expected at least one foreground object"

    for comp in components:
        gray_count = 0
        for r, c in comp:
            covered_is_gray = (int(r + c) % 2) == gray_parity
            gray_count += int(covered_is_gray)
        green_count = len(comp) - gray_count

        if gray_count > green_count:
            new_color = RED
        elif green_count > gray_count:
            new_color = ORANGE
        else:
            new_color = YELLOW

        for r, c in comp:
            out[r, c] = int(new_color)

    return out


def _required_top_left_parity(branch: str, cells: list[tuple[int, int]], gray_on_even: bool) -> int | None:
    if branch == "tie":
        return None

    delta = _parity_delta(cells)
    assert delta != 0, "majority branches require an imbalanced shape"
    gray_parity = 0 if gray_on_even else 1

    if branch == "gray":
        top_left_on_gray = delta > 0
    else:
        top_left_on_gray = delta < 0

    return int(gray_parity if top_left_on_gray else 1 - gray_parity)


def _can_place(
    top: int,
    left: int,
    cells: list[tuple[int, int]],
    blocked: np.ndarray,
) -> bool:
    height, width = blocked.shape
    for r, c in cells:
        rr = int(top + r)
        cc = int(left + c)
        if not (0 <= rr < height and 0 <= cc < width):
            return False
        if bool(blocked[rr, cc]):
            return False
    return True


def _mark_blocked(blocked: np.ndarray, abs_cells: list[tuple[int, int]]) -> None:
    height, width = blocked.shape
    for r, c in abs_cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < height and 0 <= cc < width:
                    blocked[rr, cc] = True


def _place_object(
    grid: np.ndarray,
    blocked: np.ndarray,
    cells: list[tuple[int, int]],
    branch: str,
    gray_on_even: bool,
    rng: random.Random,
) -> bool:
    height, width = grid.shape
    obj_h, obj_w = _bbox_size(cells)
    max_top = height - obj_h - OBJECT_MARGIN
    max_left = width - obj_w - OBJECT_MARGIN
    if max_top < OBJECT_MARGIN or max_left < OBJECT_MARGIN:
        return False

    required_parity = _required_top_left_parity(branch, cells, gray_on_even)
    candidates = [
        (int(top), int(left))
        for top in range(OBJECT_MARGIN, max_top + 1)
        for left in range(OBJECT_MARGIN, max_left + 1)
        if required_parity is None or (top + left) % 2 == required_parity
    ]
    rng.shuffle(candidates)

    for top, left in candidates:
        if not _can_place(top, left, cells, blocked):
            continue

        abs_cells = [(int(top + r), int(left + c)) for r, c in cells]
        for r, c in abs_cells:
            grid[r, c] = BLUE
        _mark_blocked(blocked, abs_cells)
        return True

    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(*HEIGHT_RANGE)
    width = rng.randint(*WIDTH_RANGE)
    gray_on_even = rng.random() < 0.5
    grid = _make_checkerboard(height, width, gray_on_even)
    blocked = np.zeros((height, width), dtype=bool)

    object_count = rng.randint(*OBJECT_COUNT_RANGE)
    branches = ["gray", "green", "tie"]
    while len(branches) < object_count:
        branches.append(rng.choice(BRANCHES))
    rng.shuffle(branches)

    specs: list[tuple[str, list[tuple[int, int]]]] = []
    for branch in branches:
        shape = _shape_for_branch(branch, rng)
        if shape is None:
            return None
        specs.append((branch, shape))

    specs.sort(key=lambda item: (-len(item[1]), rng.random()))

    for branch, shape in specs:
        if not _place_object(grid, blocked, shape, branch, gray_on_even, rng):
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if not {RED, ORANGE, YELLOW}.issubset(set(int(v) for v in np.unique(out))):
        return None

    if len(_components_of_color(grid, BLUE)) != object_count:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
