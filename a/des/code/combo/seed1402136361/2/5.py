from __future__ import annotations

import random

import numpy as np


RED = 2
YELLOW = 4
BLUE = 1
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# --------------- solver ---------------


def _majority_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _role_colors(component: list[tuple[int, int]]) -> dict[tuple[int, int], int]:
    cells = set(component)
    roles: dict[tuple[int, int], int] = {}
    endpoints = 0

    for r, c in component:
        neighbors = []
        for dr, dc in DIR4:
            nbr = (r + dr, c + dc)
            if nbr in cells:
                neighbors.append(nbr)

        degree = len(neighbors)
        assert degree in (1, 2), "each path pixel must have degree 1 or 2"
        if degree == 1:
            roles[(r, c)] = RED
            endpoints += 1
            continue

        same_row = neighbors[0][0] == neighbors[1][0]
        same_col = neighbors[0][1] == neighbors[1][1]
        roles[(r, c)] = BLUE if same_row or same_col else YELLOW

    assert endpoints == 2, "each object must have exactly two endpoints"
    return roles


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each path pixel by its local topological role inside its own object:
    endpoints -> red, corners -> yellow, straight segments -> blue.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _majority_color(g)
    colors = sorted(set(map(int, g.ravel())) - {bg})
    assert len(colors) == 1, "expected exactly one foreground color"
    fg = colors[0]

    mask = g == fg
    comps = _components(mask)
    assert comps, "expected at least one foreground object"

    out = g.copy()
    for comp in comps:
        for (r, c), color in _role_colors(comp).items():
            out[r, c] = color
    return out


# --------------- generator ---------------

MIN_HEIGHT = 12
MAX_HEIGHT = 20
MIN_WIDTH = 12
MAX_WIDTH = 20
MIN_OBJECTS = 2
MAX_OBJECTS = 5
MIN_STRAIGHT_LEN = 3
MAX_STRAIGHT_LEN = 8
MIN_WALK_LEN = 4
MAX_WALK_LEN = 11
INPUT_PALETTE = (0, 3, 5, 6, 7, 8, 9)
PATH_ATTEMPTS = 120
PLACEMENT_ATTEMPTS = 200
GENERATION_ATTEMPTS = 400


def _normalize_path(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in path)
    min_c = min(c for _, c in path)
    return [(r - min_r, c - min_c) for r, c in path]


def _bbox(path: list[tuple[int, int]]) -> tuple[int, int]:
    rows = [r for r, _ in path]
    cols = [c for _, c in path]
    return max(rows) - min(rows) + 1, max(cols) - min(cols) + 1


def _role_histogram(path: list[tuple[int, int]]) -> dict[int, int]:
    counts = {RED: 0, YELLOW: 0, BLUE: 0}
    for color in _role_colors(path).values():
        counts[color] += 1
    return counts


def _straight_path() -> list[tuple[int, int]]:
    length = random.randint(MIN_STRAIGHT_LEN, MAX_STRAIGHT_LEN)
    if random.random() < 0.5:
        return [(0, c) for c in range(length)]
    return [(r, 0) for r in range(length)]


def _valid_extension(candidate: tuple[int, int], tail: tuple[int, int], used: set[tuple[int, int]]) -> bool:
    if candidate in used:
        return False
    for dr, dc in DIR4:
        nbr = (candidate[0] + dr, candidate[1] + dc)
        if nbr in used and nbr != tail:
            return False
    return True


def _walk_path() -> list[tuple[int, int]] | None:
    for _ in range(PATH_ATTEMPTS):
        target_len = random.randint(MIN_WALK_LEN, MAX_WALK_LEN)
        path = [(0, 0)]
        used = {(0, 0)}
        dirs_taken: list[tuple[int, int]] = []
        tail = (0, 0)

        for _step in range(target_len - 1):
            options: list[tuple[tuple[int, int], int]] = []
            prev_dir = dirs_taken[-1] if dirs_taken else None
            for dr, dc in DIR4:
                nxt = (tail[0] + dr, tail[1] + dc)
                if not _valid_extension(nxt, tail, used):
                    continue
                weight = 4 if prev_dir == (dr, dc) else 2
                options.append(((dr, dc), weight))
            if not options:
                break

            dirs, weights = zip(*options)
            dr, dc = random.choices(dirs, weights=weights, k=1)[0]
            tail = (tail[0] + dr, tail[1] + dc)
            path.append(tail)
            used.add(tail)
            dirs_taken.append((dr, dc))

        if len(path) != target_len:
            continue

        roles = _role_histogram(path)
        if roles[YELLOW] == 0:
            continue
        return _normalize_path(path)
    return None


def _translate(path: list[tuple[int, int]], top: int, left: int) -> list[tuple[int, int]]:
    return [(top + r, left + c) for r, c in path]


def _mark_blocked(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        r0 = max(0, r - 1)
        r1 = min(h, r + 2)
        c0 = max(0, c - 1)
        c1 = min(w, c + 2)
        blocked[r0:r1, c0:c1] = True


def _place_paths(height: int, width: int, bg: int, fg: int, paths: list[list[tuple[int, int]]]) -> np.ndarray | None:
    grid = np.full((height, width), bg, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)

    ordered = sorted(paths, key=lambda path: (-len(path), -_bbox(path)[0] * _bbox(path)[1]))
    for path in ordered:
        box_h, box_w = _bbox(path)
        if box_h > height or box_w > width:
            return None

        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            top = random.randint(0, height - box_h)
            left = random.randint(0, width - box_w)
            cells = _translate(path, top, left)
            if any(blocked[r, c] for r, c in cells):
                continue
            for r, c in cells:
                grid[r, c] = fg
            _mark_blocked(blocked, cells)
            placed = True
            break

        if not placed:
            return None
    return grid


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(MIN_HEIGHT, MAX_HEIGHT)
    width = random.randint(MIN_WIDTH, MAX_WIDTH)
    bg, fg = random.sample(INPUT_PALETTE, 2)
    object_count = random.randint(MIN_OBJECTS, MAX_OBJECTS)

    paths: list[list[tuple[int, int]]] = [_straight_path()]
    walk = _walk_path()
    if walk is None:
        return None
    paths.append(walk)

    for _ in range(object_count - 2):
        if random.random() < 0.5:
            paths.append(_straight_path())
        else:
            extra = _walk_path()
            if extra is None:
                return None
            paths.append(extra)

    random.shuffle(paths)
    inp = _place_paths(height, width, bg, fg, paths)
    if inp is None:
        return None

    if int(np.count_nonzero(inp == bg)) <= int(np.count_nonzero(inp != bg)):
        return None

    comps = _components(inp == fg)
    if len(comps) != object_count:
        return None

    out = main(inp)
    present = set(map(int, out.ravel()))
    if not {RED, YELLOW, BLUE}.issubset(present):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
