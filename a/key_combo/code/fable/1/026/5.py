from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_arc_utils():
    path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    spec = importlib.util.spec_from_file_location("arc_utils_shared_fable_1_026_5", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ARC_UTILS = _load_arc_utils()
Color = _ARC_UTILS.Color
bbox = _ARC_UTILS.bbox
colors_present = _ARC_UTILS.colors_present
label_mask = _ARC_UTILS.label_mask


BLACK = Color.BLACK
PALETTE = np.array(
    [
        Color.BLUE,
        Color.RED,
        Color.GREEN,
        Color.YELLOW,
        Color.GREY,
        Color.PINK,
        Color.ORANGE,
        Color.TEAL,
        Color.MAROON,
    ],
    dtype=int,
)

MAX_GENERATION_ATTEMPTS = 400
MASK_ATTEMPTS = 100
COLORING_ATTEMPTS = 60


def _neighbors4(r: int, c: int, n: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < n:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < n:
        out.append((r, c + 1))
    return out


def _color_cycle(grid: np.ndarray) -> list[int]:
    cycle: list[int] = []
    seen: set[int] = set()
    for value in grid.ravel():
        color = int(value)
        if color == BLACK or color in seen:
            continue
        cycle.append(color)
        seen.add(color)
    return cycle


def _indexed_nonblack_positions(grid: np.ndarray, mode: str) -> list[tuple[int, int, int]]:
    n = int(grid.shape[0])
    indexed: list[tuple[int, int, int]] = []
    idx = 0

    if mode == "row_nonblack":
        for r in range(n):
            for c in range(n):
                if int(grid[r, c]) != BLACK:
                    indexed.append((r, c, idx))
                    idx += 1
        return indexed

    if mode not in {"snake_nonblack", "snake_all"}:
        raise ValueError(f"unknown order mode: {mode}")

    for r in range(n):
        cols = range(n) if r % 2 == 0 else range(n - 1, -1, -1)
        for c in cols:
            if int(grid[r, c]) != BLACK:
                indexed.append((r, c, idx))
                if mode == "snake_nonblack":
                    idx += 1
            if mode == "snake_all":
                idx += 1
    return indexed


def _recolor_sprite(sprite: np.ndarray, cycle: list[int], shift: int) -> np.ndarray:
    if not cycle:
        return sprite.copy()

    k = len(cycle)
    mapping = {cycle[i]: cycle[(i + shift) % k] for i in range(k)}
    out = sprite.copy()
    for src, dst in mapping.items():
        out[sprite == src] = dst
    return out


def _render(grid: np.ndarray, *, order_mode: str, force_shift: int | None = None) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    n = int(g.shape[0])
    out = np.full((n * n, n * n), BLACK, dtype=int)
    cycle = _color_cycle(g)
    cache: dict[int, np.ndarray] = {}

    for r, c, idx in _indexed_nonblack_positions(g, order_mode):
        shift = idx if force_shift is None else int(force_shift)
        shift %= max(len(cycle), 1)
        if shift not in cache:
            cache[shift] = _recolor_sprite(g, cycle, shift)
        out[r * n : (r + 1) * n, c * n : (c + 1) * n] = cache[shift]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the sprite by self-stamping into blocks, with each copy cyclically recolored by its boustrophedon index.

    Input structure:
    - The input is an n x n sprite on a black background, using one or more non-black colors.
    - Each input cell corresponds to one n x n block in the output.
    - The output therefore has shape n^2 x n^2.

    Rule:
    - If an input cell is black, its output block stays entirely black.
    - If an input cell is non-black, stamp the full input sprite into that block.
    - Number only the non-black input cells in boustrophedon order: even rows left-to-right, odd rows right-to-left.
    - Read the sprite's distinct non-black colors by first appearance in row-major order, treat that list as a cycle, and shift every stamped copy forward by its block index within that cycle.

    Key insight:
    - The placement pattern comes from the sprite's own non-black mask, while the color variation is not positional noise: it is a cyclic recoloring keyed by each stamped block's boustrophedon serial number.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert g.shape[0] == g.shape[1]
    n = int(g.shape[0])
    assert 1 <= n <= 5
    assert n * n <= 30

    cycle = _color_cycle(g)
    assert cycle
    return _render(g, order_mode="snake_nonblack")


def _mask_is_good(mask: np.ndarray) -> bool:
    labels, n_comp = label_mask(mask, connectivity=4)
    if n_comp != 1:
        return False

    box = bbox(mask)
    if box is None:
        return False
    r0, c0, r1, c1 = box
    if r0 == r1 or c0 == c1:
        return False

    cropped = mask[r0 : r1 + 1, c0 : c1 + 1]
    if cropped.all():
        return False

    odd_row_has_reversal = any(int(mask[r].sum()) >= 2 for r in range(1, mask.shape[0], 2))
    if not odd_row_has_reversal:
        return False

    snake = [(r, c) for r, c, _ in _indexed_nonblack_positions(mask.astype(int), "snake_nonblack")]
    row_major = [(r, c) for r, c, _ in _indexed_nonblack_positions(mask.astype(int), "row_nonblack")]
    return snake != row_major


def _grow_connected_mask(rng: np.random.Generator, n: int, cells_needed: int) -> np.ndarray | None:
    for _ in range(MASK_ATTEMPTS):
        start = (int(rng.integers(0, n)), int(rng.integers(0, n)))
        cells = {start}
        frontier = set(_neighbors4(start[0], start[1], n))

        while len(cells) < cells_needed and frontier:
            frontier_list = list(frontier)
            weights = np.array(
                [1 + sum((nr, nc) in cells for nr, nc in _neighbors4(r, c, n)) for r, c in frontier_list],
                dtype=float,
            )
            weights /= weights.sum()
            choice_idx = int(rng.choice(len(frontier_list), p=weights))
            cell = frontier_list[choice_idx]
            frontier.discard(cell)
            cells.add(cell)
            for nb in _neighbors4(cell[0], cell[1], n):
                if nb not in cells:
                    frontier.add(nb)

        if len(cells) != cells_needed:
            continue

        mask = np.zeros((n, n), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if _mask_is_good(mask):
            return mask
    return None


def _assign_colors(rng: np.random.Generator, mask: np.ndarray, color_count: int) -> np.ndarray | None:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    coords.sort()
    colors = [int(v) for v in rng.choice(PALETTE, size=color_count, replace=False)]

    for _ in range(COLORING_ATTEMPTS):
        anchor_indices = sorted(rng.choice(len(coords), size=color_count, replace=False).tolist())
        anchors = [coords[i] for i in anchor_indices]
        sprite = np.zeros(mask.shape, dtype=int)

        for r, c in coords:
            best = min(
                range(color_count),
                key=lambda idx: (
                    abs(r - anchors[idx][0]) + abs(c - anchors[idx][1]),
                    anchor_indices[idx],
                    anchors[idx][0],
                    anchors[idx][1],
                ),
            )
            sprite[r, c] = colors[best]

        cycle = _color_cycle(sprite)
        if len(cycle) != color_count:
            continue

        counts = [int(np.sum(sprite == color)) for color in cycle]
        if min(counts) < 1:
            continue
        if sum(count == 1 for count in counts) > 1:
            continue
        return sprite
    return None


def _choose_color_count(rng: np.random.Generator, n: int) -> int:
    if n == 3:
        options = np.array([2, 3], dtype=int)
        weights = np.array([1.0, 3.0], dtype=float)
    elif n == 4:
        options = np.array([2, 3, 4], dtype=int)
        weights = np.array([1.0, 4.0, 2.0], dtype=float)
    else:
        options = np.array([3, 4], dtype=int)
        weights = np.array([3.0, 2.0], dtype=float)
    weights /= weights.sum()
    return int(rng.choice(options, p=weights))


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n = int(rng.choice(np.array([3, 4, 5], dtype=int), p=np.array([0.3, 0.4, 0.3], dtype=float)))
    color_count = _choose_color_count(rng, n)

    min_cells = max(color_count + 2, 2 * color_count - 1, n + 1)
    max_cells = min((n * n) - 2, int(np.floor(0.64 * n * n)))
    if min_cells > max_cells:
        return None

    cells_needed = int(rng.integers(min_cells, max_cells + 1))
    mask = _grow_connected_mask(rng, n, cells_needed)
    if mask is None:
        return None

    sprite = _assign_colors(rng, mask, color_count)
    if sprite is None:
        return None

    if len(colors_present(sprite, exclude={BLACK})) != color_count:
        return None

    oracle = main(sprite)
    if np.array_equal(oracle, _render(sprite, order_mode="row_nonblack")):
        return None
    if np.array_equal(oracle, _render(sprite, order_mode="snake_all")):
        return None
    if np.array_equal(oracle, _render(sprite, order_mode="snake_nonblack", force_shift=0)):
        return None

    if np.unique(oracle).size < 3:
        return None
    return sprite


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fractal-gradient-stamp input")
