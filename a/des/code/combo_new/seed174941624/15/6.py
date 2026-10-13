import random
from collections import deque

import numpy as np


BG = 0
PATH = 5
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]

GRID_MIN = 15
GRID_MAX = 24
HUB_MARGIN = 4
GEN_ATTEMPTS = 400
PATH_ATTEMPTS = 80


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS:
        yield r + dr, c + dc


def _in_bounds(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    r, c = cell
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _degree_map(occupied: set[tuple[int, int]]) -> dict[tuple[int, int], int]:
    degrees: dict[tuple[int, int], int] = {}
    for cell in occupied:
        r, c = cell
        degrees[cell] = sum((nr, nc) in occupied for nr, nc in _neighbors4(r, c))
    return degrees


def _analyze_tree(mask: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int]]]:
    nodes = {tuple(map(int, coord)) for coord in np.argwhere(mask)}
    assert nodes, "expected a non-empty tree"

    neighbors: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for cell in nodes:
        r, c = cell
        nbrs = [(nr, nc) for nr, nc in _neighbors4(r, c) if (nr, nc) in nodes]
        neighbors[cell] = nbrs

    start = next(iter(nodes))
    seen = {start}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        for nbr in neighbors[cell]:
            if nbr not in seen:
                seen.add(nbr)
                queue.append(nbr)

    assert seen == nodes, "tree pixels must be 4-connected"
    edges = sum(len(nbrs) for nbrs in neighbors.values()) // 2
    assert edges == len(nodes) - 1, "path structure must be acyclic"
    return neighbors


def _turns(direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    if dr != 0:
        return [(0, -1), (0, 1)]
    return [(-1, 0), (1, 0)]


def _step(cell: tuple[int, int], direction: tuple[int, int]) -> tuple[int, int]:
    return cell[0] + direction[0], cell[1] + direction[1]


def _free_dirs(
    cell: tuple[int, int], occupied: set[tuple[int, int]], shape: tuple[int, int]
) -> list[tuple[int, int]]:
    free = []
    for direction in DIRS:
        nxt = _step(cell, direction)
        if _in_bounds(shape, nxt) and nxt not in occupied:
            free.append(direction)
    return free


def _is_valid_extension(
    start: tuple[int, int],
    path: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
    shape: tuple[int, int],
) -> bool:
    if not path or len(set(path)) != len(path):
        return False

    path_set = set(path)
    for idx, cell in enumerate(path):
        if not _in_bounds(shape, cell) or cell in occupied:
            return False

        allowed = {start if idx == 0 else path[idx - 1]}
        if idx + 1 < len(path):
            allowed.add(path[idx + 1])

        for nbr in _neighbors4(*cell):
            if (nbr in occupied or nbr in path_set) and nbr not in allowed:
                return False
    return True


def _propose_path(
    start: tuple[int, int],
    occupied: set[tuple[int, int]],
    shape: tuple[int, int],
    rng: random.Random,
    *,
    first_dir: tuple[int, int] | None = None,
    min_len: int,
    max_len: int,
    bend_prob: float,
) -> list[tuple[int, int]] | None:
    free = _free_dirs(start, occupied, shape)
    if first_dir is not None and first_dir not in free:
        return None
    if first_dir is None and not free:
        return None

    for _ in range(PATH_ATTEMPTS):
        direction1 = first_dir if first_dir is not None else rng.choice(free)
        use_bend = max_len >= 2 and rng.random() < bend_prob

        segments: list[tuple[tuple[int, int], int]] = []
        if use_bend:
            len1 = rng.randint(1, max_len - 1)
            len2 = rng.randint(1, max_len - len1)
            direction2 = rng.choice(_turns(direction1))
            segments = [(direction1, len1), (direction2, len2)]
        else:
            segments = [(direction1, rng.randint(min_len, max_len))]

        path: list[tuple[int, int]] = []
        cell = start
        for direction, length in segments:
            for _ in range(length):
                cell = _step(cell, direction)
                path.append(cell)

        if not (min_len <= len(path) <= max_len):
            continue
        if _is_valid_extension(start, path, occupied, shape):
            return path
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate the root color through a grey tree, switching to branch-marker colors when they are encountered.

    Input structure:
    - The background is black.
    - All non-background pixels form one 4-connected single-pixel-wide tree with grey path cells.
    - Exactly one colored endpoint is the root, and some branch points are colored markers instead of grey.

    Rule:
    - Start at the colored root endpoint and propagate its color away from the root along the tree.
    - Grey path pixels are recolored to the currently propagated color.
    - A colored branch marker keeps its own color, and every descendant branch beyond it uses that marker color.

    Key insight:
    - The task is a rooted tree traversal: colors do not alternate by distance, they are inherited along edges and only change when the traversal passes through a colored branch node.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    mask = g != BG
    neighbors = _analyze_tree(mask)
    degrees = {cell: len(nbrs) for cell, nbrs in neighbors.items()}
    branch_nodes = [cell for cell, degree in degrees.items() if degree >= 3]
    assert branch_nodes, "expected at least one branch node"

    colored_nodes = [cell for cell in neighbors if int(g[cell]) not in (BG, PATH)]
    root_candidates = [cell for cell in colored_nodes if degrees[cell] == 1]
    assert len(root_candidates) == 1, "expected exactly one colored root endpoint"
    root = root_candidates[0]
    root_color = int(g[root])

    assert any(cell != root for cell in colored_nodes), "expected at least one colored branch marker"
    for cell in colored_nodes:
        if cell != root:
            assert degrees[cell] >= 3, "non-root colored pixels must be branch markers"
    for cell, degree in degrees.items():
        if degree == 1 and cell != root:
            assert int(g[cell]) == PATH, "non-root leaves must be grey before propagation"

    out = np.full_like(g, BG)
    seen: set[tuple[int, int]] = set()
    stack = [(root, None, root_color)]

    while stack:
        cell, parent, current_color = stack.pop()
        assert cell not in seen, "tree traversal encountered a cycle"
        seen.add(cell)

        cell_color = int(g[cell])
        if cell_color == PATH:
            out[cell] = current_color
            next_color = current_color
        else:
            out[cell] = cell_color
            next_color = cell_color

        for nbr in neighbors[cell]:
            if nbr != parent:
                stack.append((nbr, cell, next_color))

    assert seen == set(neighbors), "traversal must reach the whole tree"
    return out


def _build_grid(
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    root: tuple[int, int],
    colored_branches: set[tuple[int, int]],
    rng: random.Random,
) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for r, c in occupied:
        grid[r, c] = PATH

    colors = rng.sample(COLOR_POOL, 1 + len(colored_branches))
    grid[root] = colors[0]
    for cell, color in zip(sorted(colored_branches), colors[1:]):
        grid[cell] = color
    return grid


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    shape = (rng.randint(GRID_MIN, GRID_MAX), rng.randint(GRID_MIN, GRID_MAX))

    if shape[0] <= 2 * HUB_MARGIN or shape[1] <= 2 * HUB_MARGIN:
        return None

    hub1 = (
        rng.randint(HUB_MARGIN, shape[0] - HUB_MARGIN - 1),
        rng.randint(HUB_MARGIN, shape[1] - HUB_MARGIN - 1),
    )
    occupied = {hub1}

    trunk = _propose_path(
        hub1,
        occupied,
        shape,
        rng,
        min_len=3,
        max_len=6,
        bend_prob=0.55,
    )
    if trunk is None:
        return None
    occupied.update(trunk)
    root = trunk[-1]

    free1 = _free_dirs(hub1, occupied, shape)
    if len(free1) < 2:
        return None
    child1_count = 2 if len(free1) == 2 else rng.choice([2, 3])
    child1_dirs = rng.sample(free1, child1_count)

    first_level_paths: list[list[tuple[int, int]]] = []
    for direction in child1_dirs:
        path = _propose_path(
            hub1,
            occupied,
            shape,
            rng,
            first_dir=direction,
            min_len=2,
            max_len=5,
            bend_prob=0.60,
        )
        if path is None:
            return None
        occupied.update(path)
        first_level_paths.append(path)

    hub2_candidates = [
        path[-1]
        for path in first_level_paths
        if len(_free_dirs(path[-1], occupied, shape)) >= 2
    ]
    if not hub2_candidates:
        return None
    hub2 = rng.choice(hub2_candidates)

    free2 = _free_dirs(hub2, occupied, shape)
    if len(free2) < 2:
        return None
    child2_count = 2 if len(free2) == 2 else rng.choice([2, 3])
    child2_dirs = rng.sample(free2, child2_count)

    for direction in child2_dirs:
        path = _propose_path(
            hub2,
            occupied,
            shape,
            rng,
            first_dir=direction,
            min_len=1,
            max_len=4,
            bend_prob=0.60,
        )
        if path is None:
            return None
        occupied.update(path)

    extra_steps = rng.randint(0, 2)
    for _ in range(extra_steps):
        degrees = _degree_map(occupied)
        attach_candidates = [
            cell
            for cell, degree in degrees.items()
            if cell != root and degree < 4 and _free_dirs(cell, occupied, shape)
        ]
        if not attach_candidates:
            break
        attach = rng.choice(attach_candidates)
        extra = _propose_path(
            attach,
            occupied,
            shape,
            rng,
            min_len=1,
            max_len=4,
            bend_prob=0.50,
        )
        if extra is not None:
            occupied.update(extra)

    degrees = _degree_map(occupied)
    branch_nodes = [cell for cell, degree in degrees.items() if degree >= 3]
    leaves = [cell for cell, degree in degrees.items() if degree == 1]
    if root not in leaves or len(branch_nodes) < 2 or len(leaves) < 4:
        return None
    if not (14 <= len(occupied) <= 45):
        return None

    max_colored = min(len(branch_nodes) - 1, len(COLOR_POOL) - 1)
    if max_colored < 1:
        return None
    colored_count = rng.randint(1, max_colored)
    colored_branches = set(rng.sample(branch_nodes, colored_count))

    grid = _build_grid(shape, occupied, root, colored_branches, rng)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.any(out == PATH):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid tree-color propagation input")
