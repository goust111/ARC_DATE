from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


_DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as (N, 2) arrays of (row, column)."""
    seen = np.zeros(mask.shape, dtype=bool)
    result: list[np.ndarray] = []
    height, width = mask.shape
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        queue = deque([(sr, sc)])
        cells = []
        while queue:
            r, c = queue.popleft()
            cells.append((r, c))
            for dr, dc in _DIR4:
                rr, cc = r + dr, c + dc
                if (0 <= rr < height and 0 <= cc < width and mask[rr, cc]
                        and not seen[rr, cc]):
                    seen[rr, cc] = True
                    queue.append((rr, cc))
        result.append(np.asarray(cells, dtype=int))
    return result


def _tight_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    result = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    result[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return result


def _scale(mask: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(mask, np.ones((factor, factor), dtype=bool))


def _scale_factor(small: np.ndarray, large: np.ndarray) -> int | None:
    """Return k when large is the exact per-pixel k-upscale of small."""
    sh, sw = small.shape
    lh, lw = large.shape
    if lh % sh or lw % sw or lh // sh != lw // sw:
        return None
    factor = lh // sh
    if factor < 1 or not np.array_equal(_scale(small, factor), large):
        return None
    return factor


def _separator(grid: np.ndarray, bg: int) -> tuple[int, str, int]:
    """Find the structural full line and return (wall color, axis, index)."""
    candidates = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for r in range(grid.shape[0]):
            if np.all(grid[r, :] == color):
                candidates.append((color, "row", r))
        for c in range(grid.shape[1]):
            if np.all(grid[:, c] == color):
                candidates.append((color, "col", c))
    assert len(candidates) == 1, "expected one full wall separator"
    return candidates[0]


def _best_assignment(edges: list[tuple[int, int, int]], n_legends: int):
    """Unique maximum matching, also requiring distinct scale factors."""
    by_legend = [[] for _ in range(n_legends)]
    for legend, chamber, factor in edges:
        by_legend[legend].append((chamber, factor))

    best_size = -1
    best: set[tuple[tuple[int, int, int], ...]] = set()

    def search(i: int, used_chambers: set[int], used_factors: set[int], chosen):
        nonlocal best_size, best
        if i == n_legends:
            key = tuple(chosen)
            size = len(key)
            if size > best_size:
                best_size, best = size, {key}
            elif size == best_size:
                best.add(key)
            return
        search(i + 1, used_chambers, used_factors, chosen)
        for chamber, factor in by_legend[i]:
            if chamber in used_chambers or factor in used_factors:
                continue
            search(i + 1, used_chambers | {chamber}, used_factors | {factor},
                   chosen + ((i, chamber, factor),))

    search(0, set(), set(), ())
    assert best_size > 0 and len(best) == 1, "matching must have a unique maximum solution"
    return next(iter(best))


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each maze chamber with the color of its uniquely assigned scaled legend shape.

    Input structure:
    - A dominant background is split by one full line of maze-wall color.
    - The maze side contains wall-enclosed 4-connected background chambers; the
      other side contains separated, monochromatic legend shapes.

    Rule:
    - Compare every chamber's tight negative-space mask with every legend mask
      enlarged by integer per-pixel replication.  Choose the unique maximum
      one-to-one assignment whose matched scale factors are all different.
    - Fill assigned chambers with their legend colors, leave unmatched chambers
      as background, clear the entire legend side, and preserve every wall cell.

    Key insight:
    - The objects being claimed are negative-space chambers, and the locally
      ambiguous cross-scale matches are resolved only by globally distinct factors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(Counter(map(int, g.ravel())).most_common(1)[0][0])
    wall, axis, cut = _separator(g, bg)

    if axis == "col":
        left_walls = int(np.sum(g[:, :cut] == wall))
        right_walls = int(np.sum(g[:, cut + 1:] == wall))
        assert left_walls != right_walls
        maze_side = (slice(None), slice(0, cut)) if left_walls > right_walls else (slice(None), slice(cut + 1, None))
        legend_side = (slice(None), slice(cut + 1, None)) if left_walls > right_walls else (slice(None), slice(0, cut))
    else:
        upper_walls = int(np.sum(g[:cut, :] == wall))
        lower_walls = int(np.sum(g[cut + 1:, :] == wall))
        assert upper_walls != lower_walls
        maze_side = (slice(0, cut), slice(None)) if upper_walls > lower_walls else (slice(cut + 1, None), slice(None))
        legend_side = (slice(cut + 1, None), slice(None)) if upper_walls > lower_walls else (slice(0, cut), slice(None))

    # Chambers are precisely background components that cannot reach the canvas edge.
    chambers = []
    height, width = g.shape
    maze_selector = np.zeros(g.shape, dtype=bool)
    maze_selector[maze_side] = True
    for coords in _components((g == bg) & maze_selector):
        touches_edge = np.any((coords[:, 0] == 0) | (coords[:, 0] == height - 1)
                              | (coords[:, 1] == 0) | (coords[:, 1] == width - 1))
        if not touches_edge:
            chambers.append(coords)
    assert chambers, "expected enclosed negative-space chambers"
    chambers.sort(key=lambda x: (int(x[:, 0].min()), int(x[:, 1].min())))

    legend_selector = np.zeros(g.shape, dtype=bool)
    legend_selector[legend_side] = True
    legends = []
    for color in np.unique(g[legend_side]):
        color = int(color)
        if color in (bg, wall):
            continue
        for coords in _components((g == color) & legend_selector):
            legends.append((coords, color))
    legends.sort(key=lambda item: (int(item[0][:, 0].min()), int(item[0][:, 1].min())))
    assert legends, "expected legend shapes"

    chamber_masks = [_tight_mask(coords) for coords in chambers]
    legend_masks = [_tight_mask(coords) for coords, _color in legends]
    edges = []
    for li, small in enumerate(legend_masks):
        for ci, large in enumerate(chamber_masks):
            factor = _scale_factor(small, large)
            if factor is not None:
                edges.append((li, ci, factor))
    assignment = _best_assignment(edges, len(legends))

    out = g.copy()
    out[legend_side] = np.where(out[legend_side] == wall, wall, bg)
    for li, ci, _factor in assignment:
        coords = chambers[ci]
        out[coords[:, 0], coords[:, 1]] = legends[li][1]
    return out


# Shapes are all 4-connected.  The first family is deliberately represented at
# two legend scales, making the global distinct-factor rule indispensable.
_FAMILY_BASES = (
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
)
_OTHER_SHAPES = (
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1], [0, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1]], dtype=bool),
)
_UNMATCHED = (
    np.array([[1, 1, 1], [1, 0, 1]], dtype=bool),
    np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
)


def _place_chambers(grid: np.ndarray, masks: list[np.ndarray], maze_width: int,
                    wall: int, rng: random.Random) -> list[np.ndarray] | None:
    """Place negative masks and their one-cell 4-neighbor wall boundaries."""
    occupied = np.zeros(grid.shape, dtype=bool)
    placed: list[np.ndarray] = [np.empty((0, 2), dtype=int) for _ in masks]
    order = sorted(range(len(masks)), key=lambda i: -masks[i].size)

    def backtrack(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        mask = masks[idx]
        mh, mw = mask.shape
        choices = [(r, c) for r in range(1, grid.shape[0] - mh)
                   for c in range(1, maze_width - mw)]
        rng.shuffle(choices)
        for r0, c0 in choices[:180]:
            cells = {(r0 + int(r), c0 + int(c)) for r, c in np.argwhere(mask)}
            boundary = {(r + dr, c + dc) for r, c in cells for dr, dc in _DIR4} - cells
            footprint = cells | boundary
            if any(occupied[r, c] for r, c in footprint):
                continue
            for r, c in footprint:
                occupied[r, c] = True
            for r, c in boundary:
                grid[r, c] = wall
            placed[idx] = np.asarray(sorted(cells), dtype=int)
            if backtrack(pos + 1):
                return True
            for r, c in footprint:
                occupied[r, c] = False
            for r, c in boundary:
                grid[r, c] = grid[0, 0]
        return False

    return placed if backtrack(0) else None


def _render_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(25, 30)
    maze_width = rng.randint(15, 18)
    legend_width = rng.randint(8, min(10, 29 - maze_width))
    width = maze_width + 1 + legend_width
    bg, wall = rng.sample(range(10), 2)
    colors = rng.sample([c for c in range(10) if c not in (bg, wall)], 4)

    family = rng.choice(_FAMILY_BASES)
    third, distractor = rng.sample(_OTHER_SHAPES, 2)
    third_factor = rng.choice((2, 3))
    legend_masks = [family, _scale(family, 2), third, distractor]
    chamber_masks = [_scale(family, 4), _scale(family, 2),
                     _scale(third, third_factor), rng.choice(_UNMATCHED)]

    grid = np.full((height, width), bg, dtype=int)
    grid[:, maze_width] = wall
    chamber_coords = _place_chambers(grid, chamber_masks, maze_width, wall, rng)
    if chamber_coords is None:
        return None

    # Stack legends with visible gaps, but scramble their spatial/color order.
    items = list(zip(legend_masks, colors))
    rng.shuffle(items)
    total_h = sum(mask.shape[0] for mask, _ in items) + len(items) - 1
    start_r = rng.randint(1, height - total_h - 1)
    r = start_r
    for mask, color in items:
        max_offset = legend_width - mask.shape[1] - 1
        if max_offset < 1:
            return None
        c0 = maze_width + 1 + rng.randint(1, max_offset)
        rows, cols = np.where(mask)
        grid[r + rows, c0 + cols] = color
        r += mask.shape[0] + 1

    expected = grid.copy()
    expected[:, maze_width + 1:] = bg
    # family-small legend claims the large chamber at factor 4; its 2x legend
    # claims the smaller chamber at factor 1.  The tempting alternative repeats 2.
    expected[chamber_coords[0][:, 0], chamber_coords[0][:, 1]] = colors[0]
    expected[chamber_coords[1][:, 0], chamber_coords[1][:, 1]] = colors[1]
    expected[chamber_coords[2][:, 0], chamber_coords[2][:, 1]] = colors[2]

    # Apply whole-scene symmetries; matching remains in the visible grid frame.
    transform = rng.randrange(8)
    if transform & 1:
        grid, expected = np.fliplr(grid), np.fliplr(expected)
    if transform & 2:
        grid, expected = np.flipud(grid), np.flipud(expected)
    if transform & 4:
        grid, expected = grid.T.copy(), expected.T.copy()

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    # Semantic checks: exactly three necessary fills, one distractor of each kind,
    # and at least two different output colors prevent degenerate constant answers.
    changed_to_color = (actual != grid) & (actual != bg)
    if len(np.unique(actual[changed_to_color])) != 3:
        return None
    if np.array_equal(actual, grid) or not np.any(actual == bg):
        return None
    return grid.astype(int, copy=False)


def generate_input() -> np.ndarray:
    """Generate a varied, oracle-checked instance with ambiguity and distractors."""
    rng = random.Random()
    for _ in range(400):
        grid = _render_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid scaled negative-space puzzle")
