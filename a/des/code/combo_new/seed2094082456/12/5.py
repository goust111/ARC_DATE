from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0

# Generation bounds.
GRID_MIN = 16
GRID_MAX = 28
SEED_COUNT_MIN = 4
SEED_COUNT_MAX = 9
TEMPLATE_RETRIES = 200
LAYOUT_RETRIES = 250
GEN_RETRIES = 300

# Keep seeds comfortably away from the corner legends.
SEED_MARGIN = 4


FILLED_BASES = (
    np.array(
        [
            [1, 1, 1],
            [0, 1, 0],
            [0, 1, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 1, 0],
            [0, 1, 1],
            [0, 0, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 0, 1],
        ],
        dtype=int,
    ),
)

EMPTY_BASES = (
    np.array(
        [
            [1, 1, 1],
            [0, 0, 0],
            [0, 0, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 0, 0],
            [1, 0, 0],
            [1, 1, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 1, 0],
            [0, 0, 0],
            [1, 0, 1],
        ],
        dtype=int,
    ),
)


def _corner_origins(height: int, width: int) -> list[tuple[int, int]]:
    return [(0, 0), (0, width - 3), (height - 3, 0), (height - 3, width - 3)]


def _reserved_corners(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    for r0, c0 in _corner_origins(height, width):
        mask[r0 : r0 + 3, c0 : c0 + 3] = True
    return mask


def _mask_key(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(int(v) for v in mask.flatten())


def _transform_mask(mask: np.ndarray, rot_k: int, flip_lr: bool) -> np.ndarray:
    out = np.rot90(mask, k=rot_k)
    if flip_lr:
        out = np.fliplr(out)
    return out.copy()


def _template_anchor(mask: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(mask != 0)
    assert len(coords) > 0, "template must contain at least one colored pixel"
    if bool(mask[1, 1]):
        return (1, 1)

    center = np.array([1, 1], dtype=int)
    dist2 = np.sum((coords - center) ** 2, axis=1)
    nearest = coords[dist2 == dist2.min()]
    assert len(nearest) == 1, "center-empty template must have a unique nearest anchor"
    return tuple(int(v) for v in nearest[0])


def _template_offsets(mask: np.ndarray) -> list[tuple[int, int]]:
    ar, ac = _template_anchor(mask)
    coords = np.argwhere(mask != 0)
    return [(int(r) - ar, int(c) - ac) for r, c in coords]


def _seed_footprint(seed: tuple[int, int], offsets: list[tuple[int, int]]) -> set[tuple[int, int]]:
    sr, sc = seed
    return {(sr + dr, sc + dc) for dr, dc in offsets}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each central seed by stamping the same-colored 3x3 corner template onto the grid.

    Input structure:
    - Background is black.
    - Each corner contains a 3x3 monochrome template on black, and the four template colors are all different.
    - Away from the corner legends, the grid contains isolated single-pixel seeds whose colors match one of the corner templates.

    Rule:
    - Read the 3x3 template in each corner and map its shape by color.
    - For each seed, stamp the same-colored template so that the template center aligns to the seed.
    - If that template has a black center, instead align the unique colored template cell nearest to the center to the seed.
    - Paint only translated template pixels that stay inside the grid and do not land on an input foreground pixel or on any corner 3x3 legend cell.
    - Keep the original corner templates and seed pixels unchanged.

    Key insight:
    - The corner miniatures are a color-keyed legend: each seed grows by printing the exact shape shown in the same-colored corner.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 6, "expected a 2D grid large enough for four 3x3 corners"
    height, width = g.shape

    reserved = _reserved_corners(height, width)
    templates: dict[int, list[tuple[int, int]]] = {}
    for r0, c0 in _corner_origins(height, width):
        patch = g[r0 : r0 + 3, c0 : c0 + 3]
        assert patch.shape == (3, 3), "each corner legend must be exactly 3x3"
        colors = [int(color) for color in np.unique(patch) if int(color) != BACKGROUND]
        assert len(colors) == 1, "each corner legend must be monochrome"
        color = colors[0]
        assert color not in templates, "corner legend colors must be distinct"
        templates[color] = _template_offsets(patch == color)

    assert len(templates) == 4, "expected four distinct corner legends"

    blockers = reserved | (g != BACKGROUND)
    out = g.copy()
    for r, c in np.argwhere((g != BACKGROUND) & ~reserved):
        color = int(g[r, c])
        assert color in templates, "every seed color must match a corner legend"
        for dr, dc in templates[color]:
            rr = int(r) + dr
            cc = int(c) + dc
            if 0 <= rr < height and 0 <= cc < width and not blockers[rr, cc]:
                out[rr, cc] = color
    return out


def _sample_templates(rng: random.Random) -> list[np.ndarray]:
    for _ in range(TEMPLATE_RETRIES):
        chosen: list[np.ndarray] = []
        for pool in (FILLED_BASES, EMPTY_BASES):
            for base in rng.sample(pool, 2):
                mask = _transform_mask(base, rng.randrange(4), rng.random() < 0.5)
                try:
                    _template_anchor(mask)
                except AssertionError:
                    break
                chosen.append(mask)
            else:
                continue
            break
        if len(chosen) != 4:
            continue
        if len({_mask_key(mask) for mask in chosen}) != 4:
            continue
        rng.shuffle(chosen)
        return chosen
    raise RuntimeError("failed to sample four distinct corner templates")


def _sample_seed_positions(
    rng: random.Random, height: int, width: int, count: int
) -> list[tuple[int, int]] | None:
    available = {
        (r, c)
        for r in range(SEED_MARGIN, height - SEED_MARGIN)
        for c in range(SEED_MARGIN, width - SEED_MARGIN)
    }
    chosen: list[tuple[int, int]] = []
    while available and len(chosen) < count:
        pos = rng.choice(tuple(available))
        chosen.append(pos)
        pr, pc = pos
        for r in range(pr - 1, pr + 2):
            for c in range(pc - 1, pc + 2):
                available.discard((r, c))
    if len(chosen) != count:
        return None
    return chosen


def _valid_layout(
    seeds: list[tuple[int, int, int]],
    template_map: dict[int, list[tuple[int, int]]],
    reserved: np.ndarray,
) -> bool:
    height, width = reserved.shape
    seed_positions = {(r, c) for r, c, _ in seeds}
    painted_cells: set[tuple[int, int]] = set()

    for r, c, color in seeds:
        footprint = _seed_footprint((r, c), template_map[color])
        if any(rr < 0 or rr >= height or cc < 0 or cc >= width for rr, cc in footprint):
            return False
        if any(reserved[rr, cc] for rr, cc in footprint):
            return False

        new_cells = {cell for cell in footprint if cell not in seed_positions}
        if not new_cells:
            return False
        if painted_cells & new_cells:
            return False
        painted_cells |= new_cells
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(GRID_MIN, GRID_MAX)
    width = rng.randint(GRID_MIN, GRID_MAX)
    grid = np.full((height, width), BACKGROUND, dtype=int)
    reserved = _reserved_corners(height, width)

    colors = rng.sample(range(1, 10), 4)
    masks = _sample_templates(rng)
    template_map: dict[int, list[tuple[int, int]]] = {}
    for color, mask, (r0, c0) in zip(colors, masks, _corner_origins(height, width)):
        patch = grid[r0 : r0 + 3, c0 : c0 + 3]
        patch[mask != 0] = color
        template_map[color] = _template_offsets(mask)

    inner_rows = height - 2 * SEED_MARGIN
    inner_cols = width - 2 * SEED_MARGIN
    max_by_area = max(SEED_COUNT_MIN, min(SEED_COUNT_MAX, (inner_rows * inner_cols) // 7))
    count = rng.randint(SEED_COUNT_MIN, max_by_area)

    base_colors = colors.copy()
    extra_colors = [rng.choice(colors) for _ in range(count - len(colors))]
    seed_colors = base_colors + extra_colors
    rng.shuffle(seed_colors)

    for _ in range(LAYOUT_RETRIES):
        positions = _sample_seed_positions(rng, height, width, count)
        if positions is None:
            return None
        seeds = [(r, c, color) for (r, c), color in zip(positions, seed_colors)]
        if not _valid_layout(seeds, template_map, reserved):
            continue

        for r, c, color in seeds:
            grid[r, c] = color
        return grid
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
