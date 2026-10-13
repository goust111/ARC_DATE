import random
from collections import Counter

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
OPPOSITE = {
    (-1, 0): (1, 0),
    (1, 0): (-1, 0),
    (0, -1): (0, 1),
    (0, 1): (0, -1),
}


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _connected_components(points: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    remaining = set(points)
    comps: list[set[tuple[int, int]]] = []
    while remaining:
        start = remaining.pop()
        comp = {start}
        stack = [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in DIR4:
                nxt = (r + dr, c + dc)
                if nxt in remaining:
                    remaining.remove(nxt)
                    comp.add(nxt)
                    stack.append(nxt)
        comps.append(comp)
    return comps


def _color_components(grid: np.ndarray, color: int) -> list[set[tuple[int, int]]]:
    points = {
        (int(r), int(c))
        for r, c in zip(*np.where(np.asarray(grid, dtype=int) == int(color)))
    }
    return _connected_components(points)


def _legend_candidates(grid: np.ndarray, bg: int) -> list[list[tuple[int, int]]]:
    height, width = grid.shape
    if height < 4 or width < 1:
        return []

    raw = [
        [(r, 0) for r in range(4)],
        [(r, width - 1) for r in range(4)],
        [(height - 4 + r, 0) for r in range(4)],
        [(height - 4 + r, width - 1) for r in range(4)],
    ]
    deduped = []
    seen = set()
    for coords in raw:
        key = tuple(coords)
        if key not in seen:
            deduped.append(coords)
            seen.add(key)

    candidates: list[list[tuple[int, int]]] = []
    for coords in deduped:
        vals = [int(grid[r, c]) for r, c in coords]
        if all(v != int(bg) for v in vals) and len(set(vals)) == 4:
            candidates.append(coords)
    return candidates


def _normalized(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _c in coords)
    min_c = min(c for _r, c in coords)
    return {(r - min_r, c - min_c) for r, c in coords}


def _is_joint_pixel(coords: set[tuple[int, int]], pixel: tuple[int, int]) -> bool:
    neighbors = []
    r, c = pixel
    for dr, dc in DIR4:
        nxt = (r + dr, c + dc)
        if nxt in coords:
            neighbors.append((dr, dc))

    if len(neighbors) <= 1:
        return False
    if len(neighbors) == 2 and neighbors[0] == OPPOSITE[neighbors[1]]:
        return False

    remainder = set(coords)
    remainder.remove(pixel)
    if not remainder:
        return False
    return len(_connected_components(remainder)) > 1


def _branch_count(coords: set[tuple[int, int]]) -> int:
    joints = {pixel for pixel in coords if _is_joint_pixel(coords, pixel)}
    if not joints:
        return 1
    remainder = coords - joints
    assert remainder, "removing joints must leave at least one branch pixel"
    return len(_connected_components(remainder))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each foreground object by its branch count, using the corner legend.

    Input structure:
    - The dominant color is the background.
    - One corner contains a 4x1 vertical legend whose top-to-bottom colors map to
      branch counts 1, 2, 3, 4.
    - All other non-background pixels belong to 4-connected monochrome objects of
      a single placeholder foreground color.

    Rule:
    - For each object, identify its joint pixels: articulation pixels that are not
      simple straight-through line interiors.
    - If an object has no such joint pixels, its branch count is 1.
    - Otherwise, remove all joint pixels and count the remaining 4-connected
      components; that count is the object's branch count.
    - Recolor the whole object with the legend color for that branch count.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _bg_color(g)
    candidates = _legend_candidates(g, bg)
    assert len(candidates) == 1, "expected exactly one corner legend"
    legend = candidates[0]
    branch_to_color = {i + 1: int(g[r, c]) for i, (r, c) in enumerate(legend)}
    legend_cells = set(legend)

    remaining_colors = {
        int(g[r, c])
        for r in range(g.shape[0])
        for c in range(g.shape[1])
        if (r, c) not in legend_cells and int(g[r, c]) != int(bg)
    }
    assert len(remaining_colors) == 1, "expected a single foreground object color"
    fg = next(iter(remaining_colors))

    components = _color_components(g, fg)
    assert components, "expected at least one foreground object"

    out = g.copy()
    for comp in components:
        count = _branch_count(comp)
        assert count in branch_to_color, "branch count must be covered by the legend"
        color = branch_to_color[count]
        for r, c in comp:
            out[r, c] = color
    return out


# --------------- generator (att1) ---------------

GRID_MIN = 14
GRID_MAX = 22
COPIES_PER_BRANCH_CHOICES = (1, 2)
LINE_LEN_RANGE = (2, 6)
L_ARM_RANGE = (2, 5)
T_ARM_RANGE = (2, 4)
CROSS_ARM_RANGE = (2, 4)
PLACEMENT_PADDING = 1
GENERATION_ATTEMPTS = 400


def _make_line() -> set[tuple[int, int]]:
    length = random.randint(*LINE_LEN_RANGE)
    if random.random() < 0.5:
        return {(0, c) for c in range(length)}
    return {(r, 0) for r in range(length)}


def _make_l() -> set[tuple[int, int]]:
    vertical = random.choice([(-1, 0), (1, 0)])
    horizontal = random.choice([(0, -1), (0, 1)])
    arm_a = random.randint(*L_ARM_RANGE)
    arm_b = random.randint(*L_ARM_RANGE)
    coords = {(0, 0)}
    for step in range(1, arm_a):
        coords.add((vertical[0] * step, vertical[1] * step))
    for step in range(1, arm_b):
        coords.add((horizontal[0] * step, horizontal[1] * step))
    return _normalized(coords)


def _make_t() -> set[tuple[int, int]]:
    missing = random.choice(DIR4)
    coords = {(0, 0)}
    for direction in DIR4:
        if direction == missing:
            continue
        arm = random.randint(*T_ARM_RANGE)
        for step in range(1, arm):
            coords.add((direction[0] * step, direction[1] * step))
    return _normalized(coords)


def _make_cross() -> set[tuple[int, int]]:
    coords = {(0, 0)}
    for direction in DIR4:
        arm = random.randint(*CROSS_ARM_RANGE)
        for step in range(1, arm):
            coords.add((direction[0] * step, direction[1] * step))
    return _normalized(coords)


def _shape_for_branch(branch_count: int) -> set[tuple[int, int]]:
    if branch_count == 1:
        coords = _make_line()
    elif branch_count == 2:
        coords = _make_l()
    elif branch_count == 3:
        coords = _make_t()
    else:
        assert branch_count == 4
        coords = _make_cross()
    assert _branch_count(coords) == branch_count
    return coords


def _bbox(coords: set[tuple[int, int]]) -> tuple[int, int]:
    return max(r for r, _c in coords) + 1, max(c for _r, c in coords) + 1


def _translated(coords: set[tuple[int, int]], top: int, left: int) -> set[tuple[int, int]]:
    return {(top + r, left + c) for r, c in coords}


def _mark_blocked(blocked: np.ndarray, coords: set[tuple[int, int]], radius: int = 1) -> None:
    height, width = blocked.shape
    for r, c in coords:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < height and 0 <= cc < width:
                    blocked[rr, cc] = True


def _legend_positions(height: int, width: int, corner: str) -> list[tuple[int, int]]:
    if corner == "tl":
        return [(r, 0) for r in range(4)]
    if corner == "tr":
        return [(r, width - 1) for r in range(4)]
    if corner == "bl":
        return [(height - 4 + r, 0) for r in range(4)]
    assert corner == "br"
    return [(height - 4 + r, width - 1) for r in range(4)]


def _all_candidates(height: int, width: int, shape: set[tuple[int, int]]) -> list[tuple[int, int]]:
    shape_h, shape_w = _bbox(shape)
    candidates = [
        (top, left)
        for top in range(height - shape_h + 1)
        for left in range(width - shape_w + 1)
    ]
    random.shuffle(candidates)
    return candidates


def _place_shape(
    grid: np.ndarray,
    blocked: np.ndarray,
    shape: set[tuple[int, int]],
    color: int,
) -> bool:
    height, width = grid.shape
    for top, left in _all_candidates(height, width, shape):
        placed = _translated(shape, top, left)
        if any(blocked[r, c] for r, c in placed):
            continue
        for r, c in placed:
            grid[r, c] = color
        _mark_blocked(blocked, placed, radius=PLACEMENT_PADDING)
        return True
    return False


def _sample_palette() -> tuple[int, int, dict[int, int]]:
    colors = random.sample(range(10), 6)
    bg = colors[0]
    fg = colors[1]
    branch_to_color = {branch: colors[branch + 1] for branch in range(1, 5)}
    return bg, fg, branch_to_color


def _object_plan() -> list[tuple[int, set[tuple[int, int]]]]:
    copies = random.choice(COPIES_PER_BRANCH_CHOICES)
    plan = []
    for branch in range(1, 5):
        for _ in range(copies):
            plan.append((branch, _shape_for_branch(branch)))
    random.shuffle(plan)
    plan.sort(key=lambda item: len(item[1]), reverse=True)
    return plan


def _grid_shape(copies_per_branch: int) -> tuple[int, int]:
    if copies_per_branch == 1:
        return random.randint(GRID_MIN, 18), random.randint(GRID_MIN, 18)
    return random.randint(17, GRID_MAX), random.randint(17, GRID_MAX)


def _gen_attempt() -> np.ndarray | None:
    copies = random.choice(COPIES_PER_BRANCH_CHOICES)
    height, width = _grid_shape(copies)
    bg, fg, branch_to_color = _sample_palette()
    legend_colors = [branch_to_color[branch] for branch in range(1, 5)]

    grid = np.full((height, width), bg, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)

    legend_corner = random.choice(["tl", "tr", "bl", "br"])
    legend = _legend_positions(height, width, legend_corner)
    for (r, c), color in zip(legend, legend_colors):
        grid[r, c] = color
    _mark_blocked(blocked, set(legend), radius=PLACEMENT_PADDING)

    shapes: list[tuple[int, set[tuple[int, int]]]] = []
    for branch in range(1, 5):
        for _ in range(copies):
            shapes.append((branch, _shape_for_branch(branch)))
    random.shuffle(shapes)
    shapes.sort(key=lambda item: len(item[1]), reverse=True)

    for _branch, shape in shapes:
        if not _place_shape(grid, blocked, shape, fg):
            return None

    counts = Counter(map(int, grid.ravel()))
    if counts[bg] != max(counts.values()) or list(counts.values()).count(counts[bg]) != 1:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    for branch, legend_color in branch_to_color.items():
        if legend_color == fg:
            return None
        if int(np.count_nonzero(out == legend_color)) == 0:
            return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
