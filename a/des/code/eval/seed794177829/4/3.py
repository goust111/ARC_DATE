from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7

GUIDE_COLORS = (RED, BLUE, YELLOW, ORANGE)
DIR_ORDER = ("up", "right", "down", "left")
DIRS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
GEN_RETRIES = 400
GRID_H_RANGE = (10, 26)
GRID_W_RANGE = (10, 26)
SEED_DIM_RANGE = (1, 5)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIRS.values():
                    nr, nc = int(rr + dr), int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _child_dirs(parent_dir: str) -> tuple[str, str]:
    if parent_dir in ("up", "down"):
        options = {"left", "right"}
    else:
        options = {"up", "down"}
    return tuple(direction for direction in DIR_ORDER if direction in options)


def _draw_branch(
    out: np.ndarray,
    start_r: int,
    start_c: int,
    direction: str,
    length: int,
    color: int,
) -> tuple[int, int]:
    h, w = map(int, out.shape)
    dr, dc = DIRS[direction]
    r = int(start_r)
    c = int(start_c)

    # Branch geometry continues in abstract coordinates; the grid only clips visibility.
    for _ in range(int(length)):
        if 0 <= r < h and 0 <= c < w:
            out[r, c] = int(color)
        r += int(dr)
        c += int(dc)

    return int(start_r + (length - 1) * dr), int(start_c + (length - 1) * dc)


def _find_seed_and_guides(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int], dict[str, tuple[int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    seed_components = _components(g == GREEN)
    assert len(seed_components) == 1, "expected exactly one green seed"
    seed = seed_components[0]
    box = _bbox(seed)
    r0, c0, r1, c1 = box

    guides: dict[str, tuple[int, int, int]] = {}
    for r, c in np.argwhere((g != BG) & (g != GREEN)):
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        assert color in GUIDE_COLORS, "unexpected guide color"

        if r < r0:
            assert r == r0 - 1 and c0 <= c <= c1
            direction = "up"
        elif c > c1:
            assert c == c1 + 1 and r0 <= r <= r1
            direction = "right"
        elif r > r1:
            assert r == r1 + 1 and c0 <= c <= c1
            direction = "down"
        else:
            assert c == c0 - 1 and r0 <= r <= r1
            direction = "left"

        assert direction not in guides, "expected at most one guide per direction"
        assert any(
            0 <= r + dr < g.shape[0] and 0 <= c + dc < g.shape[1] and int(g[r + dr, c + dc]) == GREEN
            for dr, dc in DIRS.values()
        ), "guide must touch the seed"
        guides[direction] = (r, c, color)

    return seed, box, guides


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a recursive branch structure from a green seed whose bounding-box dimensions control branch lengths.

    Input structure:
    - Black background.
    - One green connected seed region.
    - Up to four adjacent single-pixel guide blocks around the seed, one per
      cardinal direction, using red/blue/yellow/orange.

    Rule:
    - A guide launches a first-generation straight branch in its direction.
    - Horizontal branch length uses the seed height; vertical branch length uses
      the seed width.
    - Each branch tip becomes a growth point whose next generation branches grow
      in the two perpendicular directions with the same color and lengths reduced
      by one, floored at one.
    - Branches are drawn generation by generation; within a generation, growth
      points are processed in clockwise order inherited from the previous step.
    - Drawing is clipped by the canvas, but recursion follows the unclipped tips.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _seed, (r0, c0, r1, c1), guides = _find_seed_and_guides(g)
    seed_h = int(r1 - r0 + 1)
    seed_w = int(c1 - c0 + 1)

    out = g.copy()
    current: list[tuple[int, int, str, int]] = [
        (guides[direction][0], guides[direction][1], direction, guides[direction][2])
        for direction in DIR_ORDER
        if direction in guides
    ]
    generation = 1

    while current:
        horizontal_len = max(seed_h - generation + 1, 1)
        vertical_len = max(seed_w - generation + 1, 1)
        next_growth_points: list[tuple[int, int, str, int]] = []

        for start_r, start_c, direction, color in current:
            length = horizontal_len if direction in ("left", "right") else vertical_len
            tip_r, tip_c = _draw_branch(out, start_r, start_c, direction, length, color)
            next_growth_points.append((tip_r, tip_c, direction, color))

        if horizontal_len == 1 and vertical_len == 1:
            break

        next_branches: list[tuple[int, int, str, int]] = []
        for tip_r, tip_c, parent_dir, color in next_growth_points:
            for child_dir in _child_dirs(parent_dir):
                dr, dc = DIRS[child_dir]
                next_branches.append((int(tip_r + dr), int(tip_c + dc), child_dir, color))

        current = next_branches
        generation += 1

    return out


def _random_manhattan_path(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = start
    path = [(int(r), int(c))]
    while (r, c) != end:
        steps: list[tuple[int, int]] = []
        if r < end[0]:
            steps.append((1, 0))
        elif r > end[0]:
            steps.append((-1, 0))
        if c < end[1]:
            steps.append((0, 1))
        elif c > end[1]:
            steps.append((0, -1))
        dr, dc = random.choice(steps)
        r += int(dr)
        c += int(dc)
        path.append((int(r), int(c)))
    return path


def _seed_mask(height: int, width: int) -> np.ndarray:
    anchors = [
        (0, random.randrange(width)),
        (height - 1, random.randrange(width)),
        (random.randrange(height), 0),
        (random.randrange(height), width - 1),
    ]
    random.shuffle(anchors)
    cells = {anchors[0]}

    for anchor in anchors[1:]:
        start = random.choice(tuple(cells))
        cells.update(_random_manhattan_path(start, anchor))

    target_area = random.randint(len(cells), height * width)
    while len(cells) < target_area:
        frontier = []
        for r, c in cells:
            for dr, dc in DIRS.values():
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(random.choice(frontier))

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[int(r), int(c)] = True
    return mask


def _biased_position(lo: int, hi: int) -> int:
    assert lo <= hi
    if lo == hi:
        return int(lo)
    edge_slack = min(2, hi - lo)
    mode = random.random()
    if mode < 1 / 3:
        return int(lo + random.randint(0, edge_slack))
    if mode < 2 / 3:
        return int(hi - random.randint(0, edge_slack))
    return int(random.randint(lo, hi))


def _guide_sites(mask: np.ndarray, directions: set[str]) -> dict[str, tuple[int, int]]:
    height, width = map(int, mask.shape)
    sites: dict[str, tuple[int, int]] = {}

    if "up" in directions:
        cols = [int(c) for c in range(width) if bool(mask[0, c])]
        sites["up"] = (0, random.choice(cols))
    if "right" in directions:
        rows = [int(r) for r in range(height) if bool(mask[r, width - 1])]
        sites["right"] = (random.choice(rows), width - 1)
    if "down" in directions:
        cols = [int(c) for c in range(width) if bool(mask[height - 1, c])]
        sites["down"] = (height - 1, random.choice(cols))
    if "left" in directions:
        rows = [int(r) for r in range(height) if bool(mask[r, 0])]
        sites["left"] = (random.choice(rows), 0)

    return sites


def _build_candidate() -> np.ndarray:
    seed_h = random.randint(*SEED_DIM_RANGE)
    seed_w = random.randint(*SEED_DIM_RANGE)
    guide_count = random.randint(1, 4)
    guide_dirs = set(random.sample(DIR_ORDER, guide_count))

    min_grid_h = max(GRID_H_RANGE[0], seed_h + (1 if "up" in guide_dirs else 0) + (1 if "down" in guide_dirs else 0))
    min_grid_w = max(GRID_W_RANGE[0], seed_w + (1 if "left" in guide_dirs else 0) + (1 if "right" in guide_dirs else 0))
    grid_h = random.randint(min_grid_h, GRID_H_RANGE[1])
    grid_w = random.randint(min_grid_w, GRID_W_RANGE[1])

    top_margin = 1 if "up" in guide_dirs else 0
    bottom_margin = 1 if "down" in guide_dirs else 0
    left_margin = 1 if "left" in guide_dirs else 0
    right_margin = 1 if "right" in guide_dirs else 0

    seed_r0 = _biased_position(top_margin, grid_h - seed_h - bottom_margin)
    seed_c0 = _biased_position(left_margin, grid_w - seed_w - right_margin)

    mask = _seed_mask(seed_h, seed_w)
    guide_sites = _guide_sites(mask, guide_dirs)

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    seed_patch = grid[seed_r0 : seed_r0 + seed_h, seed_c0 : seed_c0 + seed_w]
    seed_patch[mask] = GREEN

    for direction in DIR_ORDER:
        if direction not in guide_dirs:
            continue
        rel_r, rel_c = guide_sites[direction]
        color = random.choice(GUIDE_COLORS)
        if direction == "up":
            rr, cc = seed_r0 - 1, seed_c0 + rel_c
        elif direction == "right":
            rr, cc = seed_r0 + rel_r, seed_c0 + seed_w
        elif direction == "down":
            rr, cc = seed_r0 + seed_h, seed_c0 + rel_c
        else:
            rr, cc = seed_r0 + rel_r, seed_c0 - 1
        grid[int(rr), int(cc)] = int(color)

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_candidate()
        try:
            out = main(grid)
        except AssertionError:
            continue

        changed = int(np.count_nonzero(out != grid))
        foreground = int(np.count_nonzero(out != BG))
        if changed < 3:
            continue
        if foreground > int(grid.size * 0.45):
            continue
        return grid

    raise RuntimeError("failed to generate a valid input")
