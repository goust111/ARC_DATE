from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
ALL_COLORS = tuple(range(1, 10))
GENERATION_ATTEMPTS = 300
MIN_MARGIN = 2
MAX_MARGIN = 6
MAX_SIDE_OFFSET = 2
MIN_PRISM_SIDE = 3
MAX_PRISM_SIDE = 7
MAX_RAYS_PER_AXIS = 2


Ray = tuple[str, int, int]  # (side, aligned row/col, dot color)
BBox = tuple[int, int, int, int]  # (r0, c0, r1, c1)


def _detect_prism_bbox(grid: np.ndarray) -> BBox:
    interior = grid != BACKGROUND
    interior = interior.copy()
    interior[0, :] = False
    interior[-1, :] = False
    interior[:, 0] = False
    interior[:, -1] = False

    coords = np.argwhere(interior)
    assert coords.size > 0, "expected a non-boundary prism"

    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = int(r0), int(c0), int(r1), int(c1)

    prism_mask = interior[r0 : r1 + 1, c0 : c1 + 1]
    assert prism_mask.all(), "prism must be a solid non-black rectangle"
    assert int(prism_mask.sum()) == (r1 - r0 + 1) * (c1 - c0 + 1)
    return r0, c0, r1, c1


def _classify_boundary_dot(shape: tuple[int, int], bbox: BBox, r: int, c: int) -> tuple[str, int]:
    h, w = shape
    r0, c0, r1, c1 = bbox

    candidates: list[tuple[str, int]] = []
    if r == 0 and c0 <= c <= c1:
        candidates.append(("top", c))
    if r == h - 1 and c0 <= c <= c1:
        candidates.append(("bottom", c))
    if c == 0 and r0 <= r <= r1:
        candidates.append(("left", r))
    if c == w - 1 and r0 <= r <= r1:
        candidates.append(("right", r))

    assert len(candidates) == 1, "each boundary dot must align with the prism on exactly one axis"
    return candidates[0]


def _extract_rays(grid: np.ndarray, bbox: BBox) -> list[Ray]:
    h, w = grid.shape
    seen: set[tuple[int, int]] = set()
    rays: list[Ray] = []

    for c in range(w):
        if grid[0, c] != BACKGROUND:
            seen.add((0, c))
        if grid[h - 1, c] != BACKGROUND:
            seen.add((h - 1, c))
    for r in range(1, h - 1):
        if grid[r, 0] != BACKGROUND:
            seen.add((r, 0))
        if grid[r, w - 1] != BACKGROUND:
            seen.add((r, w - 1))

    for r, c in sorted(seen):
        side, line = _classify_boundary_dot(grid.shape, bbox, r, c)
        rays.append((side, line, int(grid[r, c])))

    assert rays, "expected at least one boundary dot"
    return rays


def _apply_rays(grid: np.ndarray, bbox: BBox, rays: list[Ray]) -> np.ndarray:
    out = grid.copy()
    r0, c0, r1, c1 = bbox
    prism = grid[r0 : r1 + 1, c0 : c1 + 1]

    for side, line, dot_color in rays:
        if side == "left":
            row = line
            local_r = row - r0
            near = int(prism[local_r, 0])
            far = int(prism[local_r, -1])
            out[row, :c0] = dot_color
            if dot_color != near:
                out[row, c1 + 1 :] = far
        elif side == "right":
            row = line
            local_r = row - r0
            near = int(prism[local_r, -1])
            far = int(prism[local_r, 0])
            out[row, c1 + 1 :] = dot_color
            if dot_color != near:
                out[row, :c0] = far
        elif side == "top":
            col = line
            local_c = col - c0
            near = int(prism[0, local_c])
            far = int(prism[-1, local_c])
            out[:r0, col] = dot_color
            if dot_color != near:
                out[r1 + 1 :, col] = far
        else:
            assert side == "bottom"
            col = line
            local_c = col - c0
            near = int(prism[-1, local_c])
            far = int(prism[0, local_c])
            out[r1 + 1 :, col] = dot_color
            if dot_color != near:
                out[:r0, col] = far

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace border rays through a multicolor prism, absorbing near-color matches and recoloring transmitted exits by the far edge.

    Input structure:
    - The grid has a black background with one solid multicolor rectangular prism away from the border.
    - Several single-pixel colored dots lie on the outer border, and each dot is aligned with one prism row or one prism column.
    - No pre-drawn rays are present in the input; only the prism and the border dots are colored.

    Rule:
    - Each border dot shoots straight toward the prism along its row or column, coloring the incoming segment from the dot up to the cell just before the prism with the dot's own color.
    - Compare the dot color with the first prism cell hit on that side; if they match, the ray is absorbed and nothing is drawn past the prism.
    - Otherwise the prism stays unchanged and the outgoing segment on the opposite side is drawn from just beyond the far prism edge to the border using the color of that far edge cell.

    Key insight:
    - The prism is a color filter rather than a blocker: the near edge decides whether the ray survives, and the far edge decides the exit color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bbox = _detect_prism_bbox(g)
    rays = _extract_rays(g, bbox)
    return _apply_rays(g, bbox, rays)


def _paired_margins() -> tuple[int, int]:
    first = random.randint(MIN_MARGIN, MAX_MARGIN)
    lo = max(MIN_MARGIN, first - MAX_SIDE_OFFSET)
    hi = min(MAX_MARGIN, first + MAX_SIDE_OFFSET)
    return first, random.randint(lo, hi)


def _make_prism(height: int, width: int) -> np.ndarray:
    for _ in range(100):
        palette = random.sample(ALL_COLORS, random.randint(3, 5))
        prism = np.array(
            [[random.choice(palette) for _ in range(width)] for _ in range(height)],
            dtype=int,
        )
        if len(set(map(int, np.unique(prism)))) >= 3:
            return prism
    raise RuntimeError("failed to create a varied prism")


def _ray_end_colors(prism: np.ndarray, side: str, line: int, bbox: BBox) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if side == "left":
        local_r = line - r0
        return int(prism[local_r, 0]), int(prism[local_r, -1])
    if side == "right":
        local_r = line - r0
        return int(prism[local_r, -1]), int(prism[local_r, 0])
    if side == "top":
        local_c = line - c0
        return int(prism[0, local_c]), int(prism[-1, local_c])
    assert side == "bottom"
    local_c = line - c0
    return int(prism[-1, local_c]), int(prism[0, local_c])


def _dot_coord(side: str, line: int, shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    if side == "left":
        return line, 0
    if side == "right":
        return line, w - 1
    if side == "top":
        return 0, line
    assert side == "bottom"
    return h - 1, line


def _choose_rays(prism: np.ndarray, bbox: BBox) -> list[Ray]:
    r0, c0, r1, c1 = bbox
    prism_h, prism_w = prism.shape
    row_count = random.randint(1, min(MAX_RAYS_PER_AXIS, prism_h))
    col_count = random.randint(1, min(MAX_RAYS_PER_AXIS, prism_w))

    rays: list[Ray] = []
    for local_r in random.sample(range(prism_h), row_count):
        side = random.choice(("left", "right"))
        rays.append((side, r0 + local_r, -1))
    for local_c in random.sample(range(prism_w), col_count):
        side = random.choice(("top", "bottom"))
        rays.append((side, c0 + local_c, -1))

    branches = ["absorb", "pass"]
    while len(branches) < len(rays):
        branches.append(random.choice(("absorb", "pass")))
    random.shuffle(branches)

    colored_rays: list[Ray] = []
    for (side, line, _), branch in zip(rays, branches):
        near, far = _ray_end_colors(prism, side, line, bbox)
        if branch == "absorb":
            dot_color = near
        else:
            choices = [color for color in ALL_COLORS if color not in {near, far}]
            if not choices:
                choices = [color for color in ALL_COLORS if color != near]
            dot_color = random.choice(choices)
        colored_rays.append((side, line, dot_color))

    return colored_rays


def _gen_attempt() -> np.ndarray | None:
    prism_h = random.randint(MIN_PRISM_SIDE, MAX_PRISM_SIDE)
    prism_w = random.randint(MIN_PRISM_SIDE, MAX_PRISM_SIDE)
    top_margin, bottom_margin = _paired_margins()
    left_margin, right_margin = _paired_margins()

    height = top_margin + prism_h + bottom_margin
    width = left_margin + prism_w + right_margin
    if max(height, width) > 30:
        return None

    r0 = top_margin
    c0 = left_margin
    r1 = r0 + prism_h - 1
    c1 = c0 + prism_w - 1
    bbox = (r0, c0, r1, c1)

    prism = _make_prism(prism_h, prism_w)
    rays = _choose_rays(prism, bbox)

    grid = np.zeros((height, width), dtype=int)
    grid[r0 : r1 + 1, c0 : c1 + 1] = prism
    for side, line, dot_color in rays:
        rr, cc = _dot_coord(side, line, grid.shape)
        grid[rr, cc] = dot_color

    expected = _apply_rays(grid, bbox, rays)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid prism-ray input")
