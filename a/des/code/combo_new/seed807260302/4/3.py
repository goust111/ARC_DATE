import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

SPRITE_COLORS = [BLUE, RED, GREEN, YELLOW, 6, 7, 8, 9]
TRANSFORM_CODES = {
    "hflip": (RED, BLUE),
    "vflip": (BLUE, RED),
    "rot90": (GREEN, YELLOW),
}
TRANSFORMS = ("none", "hflip", "vflip", "rot90")
MAX_GENERATION_ATTEMPTS = 300


def _row_regions(grid: np.ndarray) -> list[tuple[int, int]]:
    sep_rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == GREY)]
    regions: list[tuple[int, int]] = []
    start = 0
    for r in sep_rows:
        if start <= r - 1:
            regions.append((start, r - 1))
        start = r + 1
    if start <= grid.shape[0] - 1:
        regions.append((start, grid.shape[0] - 1))
    assert regions, "expected at least one non-separator row region"
    for r0, r1 in regions:
        assert r1 - r0 + 1 >= 2, "each row region needs a two-pixel legend"
    return regions


def _transform_from_legend(top: int, bottom: int) -> str:
    if top == bottom:
        return "none"
    if (top, bottom) == TRANSFORM_CODES["hflip"]:
        return "hflip"
    if (top, bottom) == TRANSFORM_CODES["vflip"]:
        return "vflip"
    if (top, bottom) == TRANSFORM_CODES["rot90"]:
        return "rot90"
    raise AssertionError(f"unknown legend color pair: {(top, bottom)}")


def _apply_transform(patch: np.ndarray, transform: str) -> np.ndarray:
    if transform == "none":
        return patch.copy()
    if transform == "hflip":
        return np.fliplr(patch).copy()
    if transform == "vflip":
        return np.flipud(patch).copy()
    if transform == "rot90":
        assert patch.shape[0] == patch.shape[1], "90-degree sprite bboxes must be square"
        return np.rot90(patch, -1).copy()
    raise AssertionError(f"unknown transform: {transform}")


def _component_bboxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    seen = np.zeros(mask.shape, dtype=bool)
    bboxes: list[tuple[int, int, int, int]] = []
    height, width = mask.shape
    for r in range(height):
        for c in range(width):
            if seen[r, c] or not mask[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rows = [cell[0] for cell in cells]
            cols = [cell[1] for cell in cells]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            assert r1 - r0 + 1 <= 5 and c1 - c0 + 1 <= 5, "sprite bbox exceeds 5x5"
            bboxes.append((r0, c0, r1, c1))
    return bboxes


def main(grid: np.ndarray) -> np.ndarray:
    """Transform each row-region sprite according to that row's two-pixel color legend.

    Input structure:
    - The background is black, and complete horizontal grey rows split the grid
      into several row regions.
    - At the far left of each row region, the first two cells form a vertical
      legend bar with color A above color B.
    - The right side of each row region contains separated small foreground
      sprites whose bounding boxes are at most 5x5.

    Rule:
    - A matching legend pair means no transform; red above blue means horizontal
      reflection; blue above red means vertical reflection; green above yellow
      means clockwise 90-degree rotation.
    - In each row region, transform every sprite inside its own bounding box and
      paste the transformed patch back at the same bounding-box position.
    - Keep grey separator rows and the two-pixel legend bars unchanged.

    Key insight:
    - The same tiny color bar controls every independent sprite in its horizontal
      band, so the solver must read the legend before applying local geometry.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"
    out = g.copy()

    for r0, r1 in _row_regions(g):
        transform = _transform_from_legend(int(g[r0, 0]), int(g[r0 + 1, 0]))
        region = g[r0 : r1 + 1, :]
        mask = (region != BLACK) & (region != GREY)
        mask[0:2, 0] = False
        for br0, bc0, br1, bc1 in _component_bboxes(mask):
            gr0, gr1 = r0 + br0, r0 + br1
            patch = g[gr0 : gr1 + 1, bc0 : bc1 + 1]
            out[gr0 : gr1 + 1, bc0 : bc1 + 1] = _apply_transform(patch, transform)

    return out


def _manhattan_path(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = a
    tr, tc = b
    cells = [(r, c)]
    first_axis = random.randrange(2)
    axes = (first_axis, 1 - first_axis)
    for axis in axes:
        if axis == 0:
            while r != tr:
                r += 1 if tr > r else -1
                cells.append((r, c))
        else:
            while c != tc:
                c += 1 if tc > c else -1
                cells.append((r, c))
    return cells


def _connected_mask(height: int, width: int) -> np.ndarray:
    anchors = [
        (0, random.randrange(width)),
        (height - 1, random.randrange(width)),
        (random.randrange(height), 0),
        (random.randrange(height), width - 1),
    ]
    random.shuffle(anchors)
    mask = np.zeros((height, width), dtype=bool)
    for a, b in zip(anchors, anchors[1:]):
        for r, c in _manhattan_path(a, b):
            mask[r, c] = True

    max_cells = height * width
    target = random.randint(int(mask.sum()), max_cells)
    frontier = {
        (nr, nc)
        for r, c in zip(*np.where(mask))
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
        if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]
    }
    while int(mask.sum()) < target and frontier:
        r, c = random.choice(tuple(frontier))
        frontier.remove((r, c))
        mask[r, c] = True
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                frontier.add((nr, nc))
    return mask


def _color_sprite(mask: np.ndarray) -> np.ndarray:
    patch = np.full(mask.shape, BLACK, dtype=int)
    cells = list(zip(*np.where(mask)))
    color_count = 1 if random.random() < 0.45 else random.randint(2, min(4, len(SPRITE_COLORS)))
    colors = random.sample(SPRITE_COLORS, color_count)
    for r, c in cells:
        patch[r, c] = random.choice(colors)
    return patch


def _random_sprite(transform: str, max_height: int) -> np.ndarray | None:
    for _ in range(100):
        if transform == "rot90":
            side = random.randint(2, min(5, max_height))
            height = width = side
        else:
            height = random.randint(2, min(5, max_height))
            width = random.randint(2, 5)
        patch = _color_sprite(_connected_mask(height, width))
        transformed = _apply_transform(patch, transform)
        if transform == "none" or not np.array_equal(patch, transformed):
            return patch
    return None


def _can_place(occupied: np.ndarray, top: int, left: int, height: int, width: int) -> bool:
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(occupied.shape[0], top + height + 1)
    c1 = min(occupied.shape[1], left + width + 1)
    return not np.any(occupied[r0:r1, c0:c1])


def _place_row_sprites(grid: np.ndarray, r0: int, r1: int, transform: str) -> bool:
    region_h = r1 - r0 + 1
    width = grid.shape[1]
    occupied = np.zeros((region_h, width), dtype=bool)
    occupied[0:2, 0] = True
    count = random.randint(2, 4)
    for _ in range(count):
        max_sprite_h = min(5, region_h - 2)
        if max_sprite_h < 2:
            return False
        patch = _random_sprite(transform, max_sprite_h)
        if patch is None:
            return False
        ph, pw = patch.shape
        candidates = [
            (rr, cc)
            for rr in range(1, region_h - ph)
            for cc in range(3, width - pw)
            if _can_place(occupied, rr, cc, ph, pw)
        ]
        if not candidates:
            return False
        rr, cc = random.choice(candidates)
        grid[r0 + rr : r0 + rr + ph, cc : cc + pw] = patch
        occupied[rr : rr + ph, cc : cc + pw] = True
    return True


def _legend_for(transform: str) -> tuple[int, int]:
    if transform == "none":
        color = random.choice(SPRITE_COLORS)
        return color, color
    return TRANSFORM_CODES[transform]


def _gen_attempt() -> np.ndarray | None:
    row_heights = [random.randint(5, 7) for _ in TRANSFORMS]
    if sum(row_heights) + len(row_heights) - 1 > 30:
        return None
    width = random.randint(18, 30)
    height = sum(row_heights) + len(row_heights) - 1
    grid = np.full((height, width), BLACK, dtype=int)

    transforms = list(TRANSFORMS)
    random.shuffle(transforms)
    row_start = 0
    for i, (row_h, transform) in enumerate(zip(row_heights, transforms)):
        r0, r1 = row_start, row_start + row_h - 1
        top, bottom = _legend_for(transform)
        grid[r0, 0] = top
        grid[r0 + 1, 0] = bottom
        if not _place_row_sprites(grid, r0, r1, transform):
            return None
        row_start = r1 + 1
        if i + 1 < len(row_heights):
            grid[row_start, :] = GREY
            row_start += 1

    expected = main(grid)
    assert expected.shape == grid.shape
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
