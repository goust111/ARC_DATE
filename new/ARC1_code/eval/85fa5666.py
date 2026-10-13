from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, bbox_w, colors_present, find_components, get_bg, neighbors4
import random
from functools import lru_cache
from itertools import combinations


Ray = tuple[tuple[int, int], int, list[tuple[int, int]]]

CORNER_OFFSETS = {
    "NW": (-1, -1),
    "NE": (-1, 2),
    "SW": (2, -1),
    "SE": (2, 2),
}

DIAGONAL_DIRS = {
    "NW": (-1, -1),
    "NE": (-1, 1),
    "SW": (1, -1),
    "SE": (1, 1),
}

ROTATED_SOURCE = {
    "NW": "SW",
    "NE": "NW",
    "SE": "NE",
    "SW": "SE",
}


def _block_color(g: np.ndarray, bg: int) -> int:
    candidates: list[int] = []
    for color in colors_present(g, exclude={bg}):
        comps = find_components(
            g,
            bg=bg,
            connectivity=4,
            monochromatic=True,
            colors=[color],
            return_format="dict",
        )
        if comps and all(
            int(comp["area"]) == 4
            and bbox_h(comp["bbox"]) == 2
            and bbox_w(comp["bbox"]) == 2
            for comp in comps
        ):
            candidates.append(int(color))

    assert len(candidates) == 1, "expected exactly one color forming solid 2x2 blocks"
    return candidates[0]


def _blocks(g: np.ndarray, color: int, bg: int) -> list[tuple[int, int]]:
    comps = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[color],
        return_format="dict",
    )
    blocks: list[tuple[int, int]] = []
    for comp in comps:
        b = comp["bbox"]
        assert int(comp["area"]) == 4 and bbox_h(b) == 2 and bbox_w(b) == 2
        blocks.append((int(b[0]), int(b[1])))

    assert blocks, "expected at least one 2x2 block"
    return sorted(blocks)


def _ray_path(g: np.ndarray, start: tuple[int, int], direction: tuple[int, int], block_color: int) -> list[tuple[int, int]]:
    h, w = g.shape
    dr, dc = direction
    r, c = start
    path: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w and int(g[r, c]) != block_color:
        path.append((r, c))
        r += dr
        c += dc

    assert path, "expected every corner ray to start in bounds outside the block"
    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each 2x2 block's four diagonal corner colors clockwise, then extend the rotated colors outward as diagonal rays.

    Input structure
    - The background is the dominant color.
    - One foreground color forms one or more solid 2x2 blocks.
    - Each block has four non-background, non-block corner pixels diagonally adjacent to it.

    Rule
    - Around every block, move the NW corner color to NE, NE to SE, SE to SW, and SW to NW.
    - From each rotated corner, paint along the outward diagonal until the grid edge or another 2x2 block is reached.
    - If a ray reaches a same-colored outgoing corner of another block that has its own visible continuation, stop the incoming ray at that junction.

    Key insight
    - The 2x2 blocks are color rotators and ray blockers; the corner pixels are not independent dots but the four ports of each block.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))
    block_color = _block_color(g, bg)

    rays: list[Ray] = []
    h, w = g.shape
    for r0, c0 in _blocks(g, block_color, bg):
        corner_colors: dict[str, int] = {}
        for corner, (dr, dc) in CORNER_OFFSETS.items():
            r = r0 + dr
            c = c0 + dc
            assert 0 <= r < h and 0 <= c < w, "expected all four block corners in bounds"
            color = int(g[r, c])
            assert color != bg and color != block_color, "expected colored corner ports around each block"
            corner_colors[corner] = color

        for corner, offset in CORNER_OFFSETS.items():
            start = (r0 + offset[0], c0 + offset[1])
            color = corner_colors[ROTATED_SOURCE[corner]]
            path = _ray_path(g, start, DIAGONAL_DIRS[corner], block_color)
            rays.append((start, color, path))

    starts_by_color: dict[int, set[tuple[int, int]]] = {}
    path_len: dict[tuple[int, tuple[int, int]], int] = {}
    for start, color, path in rays:
        starts_by_color.setdefault(color, set()).add(start)
        path_len[(color, start)] = len(path)

    planned: dict[tuple[int, int], int] = {}
    for start, color, path in rays:
        keep = len(path)
        other_starts = starts_by_color[color] - {start}
        for i, cell in enumerate(path):
            if i == 0:
                continue
            for nr, nc in neighbors4(*cell):
                other_start = (int(nr), int(nc))
                if other_start in other_starts and path_len[(color, other_start)] > 1:
                    keep = i + 1
                    break
            if keep != len(path):
                break

        for cell in path[:keep]:
            previous = planned.get(cell)
            assert previous is None or previous == color, "conflicting rays must not overlap"
            planned[cell] = color

    out = g.copy()
    for (r, c), color in planned.items():
        out[r, c] = color
    return out

# --------------- generator (o5x) ---------------

# Canvas, layout, and retry bounds.
_GEN_GRID_SIZE_RANGE = (8, 12)
_GEN_BLOCK_COUNT_RANGE = (1, 3)
_GEN_MAX_ATTEMPTS = 300
# Block geometry and palette.
_GEN_BACKGROUND_COLOR = 0
_GEN_BLOCK_COLOR = 2
_GEN_BLOCK_SIZE = 2
_GEN_BOX_SIZE = 4
_GEN_BOX_TO_BLOCK_OFFSET = 1
_GEN_CORNER_COLORS = (3, 6, 7, 8)
_GEN_CORNER_OFFSETS = ((-1, -1), (-1, 2), (2, -1), (2, 2))
_GEN_DIAGONAL_DIRS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
# Acceptance checks.
_GEN_MIN_LONG_RAY_LENGTH = 2

def _gen_boxes_disjoint(a: tuple[int, int], b: tuple[int, int]) -> bool:
    ar, ac = a
    br, bc = b
    last = _GEN_BOX_SIZE - 1
    return ar + last < br or br + last < ar or ac + last < bc or bc + last < ac

@lru_cache(maxsize=None)
def _gen_valid_box_sets(n: int, k: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    positions = tuple((r, c) for r in range(n - _GEN_BOX_SIZE + 1) for c in range(n - _GEN_BOX_SIZE + 1))
    valid: list[tuple[tuple[int, int], ...]] = []
    for combo in combinations(positions, k):
        if all(_gen_boxes_disjoint(a, b) for a, b in combinations(combo, 2)):
            valid.append(combo)
    return tuple(valid)


def _gen_has_long_ray(g: np.ndarray, block_tops: list[tuple[int, int]]) -> bool:
    h, w = g.shape
    for r0, c0 in block_tops:
        for (orow, ocol), (drow, dcol) in zip(_GEN_CORNER_OFFSETS, _GEN_DIAGONAL_DIRS):
            r, c = r0 + orow, c0 + ocol
            length = 0
            while 0 <= r < h and 0 <= c < w and int(g[r, c]) != _GEN_BLOCK_COLOR:
                length += 1
                if length >= _GEN_MIN_LONG_RAY_LENGTH:
                    return True
                r += drow
                c += dcol
    return False

def _gen_render_grid(n: int, box_set: tuple[tuple[int, int], ...]) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Render one chosen box layout and return the 2x2 block top-left cells."""
    g = np.full((n, n), _GEN_BACKGROUND_COLOR, dtype=int)
    block_tops: list[tuple[int, int]] = []
    for box_r, box_c in box_set:
        r0, c0 = box_r + _GEN_BOX_TO_BLOCK_OFFSET, box_c + _GEN_BOX_TO_BLOCK_OFFSET
        block_tops.append((r0, c0))
        g[r0 : r0 + _GEN_BLOCK_SIZE, c0 : c0 + _GEN_BLOCK_SIZE] = _GEN_BLOCK_COLOR
        for color, (dr, dc) in zip(random.sample(_GEN_CORNER_COLORS, 4), _GEN_CORNER_OFFSETS):
            g[r0 + dr, c0 + dc] = color
    return g, block_tops

def _gen_passes_oracle(g: np.ndarray) -> bool:
    """Accept only grids whose solver result is deterministic and non-conflicting."""
    try:
        out = main(g)
        return bool(np.array_equal(main(g.copy()), out))
    except Exception:
        return False

def _gen_attempt() -> np.ndarray | None:
    n = random.randint(*_GEN_GRID_SIZE_RANGE)
    k = random.randint(*_GEN_BLOCK_COUNT_RANGE)
    box_sets = _gen_valid_box_sets(n, k)
    if not box_sets:
        return None

    g, block_tops = _gen_render_grid(n, random.choice(box_sets))
    if not _gen_has_long_ray(g, block_tops) or not _gen_passes_oracle(g):
        return None

    return g

def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("failed to generate valid 85fa5666 input")
