import random
from collections import Counter

import numpy as np


AZURE = 8
YELLOW = 4
ANCHOR_COLORS = (1, 2, 3, 5, 6, 7, 9)
GEN_RETRIES = 800
PAIR_PLACEMENT_RETRIES = 2500


def _base_color(r: int, c: int, azure_on_even: bool) -> int:
    even_parity = (r + c) % 2 == 0
    azure_here = even_parity if azure_on_even else not even_parity
    return AZURE if azure_here else YELLOW


def _l_path(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    (r1, c1), (r2, c2) = a, b
    if c1 > c2:
        r1, c1, r2, c2 = r2, c2, r1, c1
    assert c1 < c2
    assert r1 != r2

    path = [(int(r1), int(c)) for c in range(int(c1), int(c2) + 1)]
    step = 1 if r2 > r1 else -1
    for r in range(int(r1 + step), int(r2 + step), step):
        path.append((int(r), int(c2)))
    return path


def _infer_checkerboard_phase(grid: np.ndarray) -> bool:
    scores = {}
    for azure_on_even in (False, True):
        mismatches = 0
        observed = 0
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                val = int(grid[r, c])
                if val not in (AZURE, YELLOW):
                    continue
                observed += 1
                if val != _base_color(int(r), int(c), azure_on_even):
                    mismatches += 1
        assert observed > 0
        scores[azure_on_even] = (mismatches, -observed)
    return min(scores, key=scores.get)


def _anchor_pairs(grid: np.ndarray) -> list[tuple[int, tuple[int, int], tuple[int, int]]]:
    positions: dict[int, list[tuple[int, int]]] = {}
    for color in np.unique(grid):
        color = int(color)
        if color in (AZURE, YELLOW):
            continue
        coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == color)]
        assert len(coords) == 2, f"expected exactly two anchors for color {color}"
        coords.sort(key=lambda rc: (rc[1], rc[0]))
        assert coords[0][1] != coords[1][1], f"anchor pair {color} cannot share a column"
        assert coords[0][0] != coords[1][0], f"anchor pair {color} cannot share a row"
        positions[color] = coords

    pairs = [(color, coords[0], coords[1]) for color, coords in positions.items()]
    pairs.sort(key=lambda item: (item[1][0], item[1][1], item[0]))
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor only the selected checkerboard cells along each same-colored anchor pair's L-shaped path.

    Input structure:
    - The whole grid is an azure/yellow 1x1 checkerboard, except for scattered colored anchor pixels.
    - Every non-checkerboard color appears exactly twice, forming one anchor pair.
    - The two anchors of each pair are in different rows and different columns.

    Rule:
    - For each anchor pair, read the left anchor first and the right anchor second.
    - Build the unique L-shaped path that goes horizontally from the left anchor to the right anchor's column, then vertically to the right anchor's row, including the corner cell.
    - Count the underlying azure and yellow checkerboard cells along that path.
    - If azure is the strict majority, recolor every yellow path cell to the anchor color; otherwise recolor every azure path cell to the anchor color.
    - Keep both anchor pixels unchanged and leave all cells outside the path untouched.

    Key insight:
    - The anchors define deterministic one-bend paths, and the output depends on which checkerboard parity is in the minority or tied subset along each path.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    azure_on_even = _infer_checkerboard_phase(g)

    out = g.copy()
    for color, a, b in _anchor_pairs(g):
        path = _l_path(a, b)
        azure_count = sum(1 for r, c in path if _base_color(int(r), int(c), azure_on_even) == AZURE)
        yellow_count = len(path) - azure_count
        target_checker = YELLOW if azure_count > yellow_count else AZURE
        anchors = {a, b}
        for r, c in path:
            if (int(r), int(c)) in anchors:
                continue
            if _base_color(int(r), int(c), azure_on_even) == target_checker:
                out[int(r), int(c)] = int(color)
    return out


def _checkerboard(height: int, width: int, azure_on_even: bool) -> np.ndarray:
    grid = np.empty((height, width), dtype=int)
    for r in range(height):
        for c in range(width):
            grid[r, c] = _base_color(int(r), int(c), azure_on_even)
    return grid


def _branch_kind(path: list[tuple[int, int]], azure_on_even: bool) -> str:
    azure_count = sum(1 for r, c in path if _base_color(int(r), int(c), azure_on_even) == AZURE)
    yellow_count = len(path) - azure_count
    if azure_count > yellow_count:
        return "azure_majority"
    if yellow_count > azure_count:
        return "yellow_majority"
    return "tie"


def _sample_branch_plan(rng: random.Random, pair_count: int) -> list[str]:
    if pair_count == 1:
        if rng.random() < 0.5:
            return ["azure_majority"]
        return [rng.choice(["yellow_majority", "tie"])]

    plan = []
    azure_slots = pair_count // 2
    non_azure_slots = pair_count - azure_slots
    plan.extend(["azure_majority"] * azure_slots)
    plan.extend(rng.choice(["yellow_majority", "tie"]) for _ in range(non_azure_slots))
    rng.shuffle(plan)

    if rng.random() < 0.5:
        for idx in range(pair_count):
            if rng.random() < 0.35:
                plan[idx] = "azure_majority" if plan[idx] != "azure_majority" else rng.choice(["yellow_majority", "tie"])
    return plan


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.randint(1, 4)
    min_side = 8 + 2 * pair_count
    max_side = min(24, min_side + 8)
    height = rng.randint(min_side, max_side)
    width = rng.randint(min_side, max_side)
    azure_on_even = rng.choice([False, True])

    grid = _checkerboard(height, width, azure_on_even)
    pair_colors = rng.sample(ANCHOR_COLORS, pair_count)
    branch_plan = _sample_branch_plan(rng, pair_count)

    used_cells: set[tuple[int, int]] = set()
    placed: list[tuple[int, tuple[int, int], tuple[int, int], list[tuple[int, int]], str]] = []

    for color, desired_kind in zip(pair_colors, branch_plan):
        chosen = None
        for _ in range(PAIR_PLACEMENT_RETRIES):
            c1 = rng.randrange(0, width - 1)
            c2 = rng.randrange(c1 + 1, width)
            r1 = rng.randrange(height)
            r2 = rng.randrange(height)
            if r1 == r2:
                continue
            a = (int(r1), int(c1))
            b = (int(r2), int(c2))
            path = _l_path(a, b)
            if any(cell in used_cells for cell in path):
                continue
            if _branch_kind(path, azure_on_even) != desired_kind:
                continue
            chosen = (int(color), a, b, path, desired_kind)
            break
        if chosen is None:
            return None
        placed.append(chosen)
        used_cells.update(chosen[3])

    for color, a, b, _path, _kind in placed:
        grid[a] = int(color)
        grid[b] = int(color)

    non_checker = [int(color) for color in np.unique(grid) if int(color) not in (AZURE, YELLOW)]
    assert len(non_checker) == pair_count
    assert all(int((grid == color).sum()) == 2 for color in non_checker)

    kinds_seen = Counter(kind for *_rest, kind in placed)
    assert sum(kinds_seen.values()) == pair_count

    out = main(grid)
    assert out.shape == grid.shape
    assert np.any(out != grid)
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
