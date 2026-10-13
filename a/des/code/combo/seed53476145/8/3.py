import random
from collections import deque

import numpy as np


BG = 0
RED = 2
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# --------------- solver ---------------


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _perimeter_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _extract_regions(grid: np.ndarray) -> list[dict[str, int]]:
    wall_mask = (grid != BG) & (grid != RED)
    regions: list[dict[str, int]] = []
    for coords in _components(wall_mask):
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        if height < 3 or width < 3:
            raise AssertionError("every enclosed region must be at least 3x3")

        local = np.zeros((height, width), dtype=bool)
        local[coords[:, 0] - r0, coords[:, 1] - c0] = True
        if not np.array_equal(local, _perimeter_mask(height, width)):
            raise AssertionError("expected each wall component to be a one-pixel-thick rectangular frame")

        red_count = int(np.count_nonzero(grid[r0 + 1 : r1, c0 + 1 : c1] == RED))
        if red_count > 3:
            raise AssertionError("each region may contain at most three red control pixels")
        regions.append(
            {
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r1),
                "c1": int(c1),
                "height": height,
                "width": width,
                "red_count": red_count,
            }
        )

    covered_red = np.zeros_like(grid, dtype=bool)
    for region in regions:
        covered_red[
            region["r0"] + 1 : region["r1"],
            region["c0"] + 1 : region["c1"],
        ] = True
    stray_red = (grid == RED) & (~covered_red)
    if np.any(stray_red):
        raise AssertionError("red control pixels must lie strictly inside a region")

    return sorted(regions, key=lambda item: (item["r0"], item["c0"]))


def _rotated_bbox_top_left(r0: int, c0: int, height: int, width: int, turns: int) -> tuple[int, int]:
    new_height, new_width = (height, width) if turns % 2 == 0 else (width, height)
    center_r2 = 2 * r0 + height - 1
    center_c2 = 2 * c0 + width - 1
    top_num = center_r2 - new_height + 1
    left_num = center_c2 - new_width + 1
    if top_num % 2 or left_num % 2:
        raise AssertionError("90-degree rotation requires matching row/column parity")
    return top_num // 2, left_num // 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each rectangular enclosed region clockwise by 90 degrees per red control pixel,
    then erase the rotated control pixels back to the black background.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    regions = _extract_regions(g)

    for region in regions:
        out[region["r0"] : region["r1"] + 1, region["c0"] : region["c1"] + 1] = BG

    for region in regions:
        r0, c0 = region["r0"], region["c0"]
        height, width = region["height"], region["width"]
        turns = region["red_count"] % 4

        patch = g[r0 : r0 + height, c0 : c0 + width].copy()
        signal_mask = np.zeros_like(patch, dtype=bool)
        signal_mask[1:-1, 1:-1] = patch[1:-1, 1:-1] == RED

        rotated_patch = np.rot90(patch, -turns)
        rotated_signal = np.rot90(signal_mask, -turns)
        rotated_patch[rotated_signal] = BG

        nr0, nc0 = _rotated_bbox_top_left(r0, c0, height, width, turns)
        nh, nw = rotated_patch.shape
        if not (0 <= nr0 and 0 <= nc0 and nr0 + nh <= g.shape[0] and nc0 + nw <= g.shape[1]):
            raise AssertionError("rotated region must stay inside the canvas")

        dest = out[nr0 : nr0 + nh, nc0 : nc0 + nw]
        overlap = (dest != BG) & (rotated_patch != BG)
        if np.any(overlap):
            raise AssertionError("rotated regions must not overlap")
        dest[rotated_patch != BG] = rotated_patch[rotated_patch != BG]

    return out


# --------------- generator ---------------

PALETTE = tuple(color for color in range(10) if color not in {BG, RED})
REGION_COUNT_CHOICES = (2, 3, 4, 5)
GRID_SIDE_RANGE = (18, 30)
REGION_SIDE_CHOICES = (5, 6, 7, 8, 9)
MARKER_LENGTH_CHOICES = (2, 3)
SIDES = ("top", "right", "bottom", "left")
RESERVE_GAP = 2
BORDER_MARGIN = 1
PLACE_ATTEMPTS = 200
GEN_ATTEMPTS = 400

_GEN_BRANCH_CURSOR = 0


def _sample_red_counts(region_count: int, rng: random.Random) -> list[int]:
    global _GEN_BRANCH_CURSOR
    counts = [(_GEN_BRANCH_CURSOR + idx) % 4 for idx in range(region_count)]
    _GEN_BRANCH_CURSOR = (_GEN_BRANCH_CURSOR + region_count) % 4
    rng.shuffle(counts)
    return counts


def _sample_region_size(rng: random.Random) -> tuple[int, int]:
    parity = rng.choice((0, 1))
    options = [side for side in REGION_SIDE_CHOICES if side % 2 == parity]
    height = rng.choice(options)
    if rng.random() < 0.65:
        width_options = [side for side in options if side != height]
        width = rng.choice(width_options or options)
    else:
        width = rng.choice(options)
    return height, width


def _marker_cells(height: int, width: int, side: str, length: int, rng: random.Random) -> list[tuple[int, int]]:
    if side == "top":
        start = rng.randint(1, width - length - 1)
        return [(0, col) for col in range(start, start + length)]
    if side == "bottom":
        start = rng.randint(1, width - length - 1)
        return [(height - 1, col) for col in range(start, start + length)]
    if side == "left":
        start = rng.randint(1, height - length - 1)
        return [(row, 0) for row in range(start, start + length)]
    start = rng.randint(1, height - length - 1)
    return [(row, width - 1) for row in range(start, start + length)]


def _build_region_patch(
    height: int,
    width: int,
    wall_color: int,
    marker_color: int,
    side: str,
    marker_length: int,
    red_count: int,
    rng: random.Random,
) -> np.ndarray:
    patch = np.full((height, width), BG, dtype=int)
    patch[0, :] = wall_color
    patch[-1, :] = wall_color
    patch[:, 0] = wall_color
    patch[:, -1] = wall_color

    for row, col in _marker_cells(height, width, side, marker_length, rng):
        patch[row, col] = marker_color

    interior_cells = [(row, col) for row in range(1, height - 1) for col in range(1, width - 1)]
    for row, col in rng.sample(interior_cells, red_count):
        patch[row, col] = RED
    return patch


def _reserve_bounds(top: int, left: int, size: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    r0 = max(0, top - RESERVE_GAP)
    c0 = max(0, left - RESERVE_GAP)
    r1 = min(h, top + size + RESERVE_GAP)
    c1 = min(w, left + size + RESERVE_GAP)
    return r0, r1, c0, c1


def _place_region(
    grid: np.ndarray,
    blocked: np.ndarray,
    patch: np.ndarray,
    reserve_size: int,
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    max_top = h - BORDER_MARGIN - reserve_size
    max_left = w - BORDER_MARGIN - reserve_size
    min_top = BORDER_MARGIN
    min_left = BORDER_MARGIN
    if max_top < min_top or max_left < min_left:
        return False

    for _ in range(PLACE_ATTEMPTS):
        top = rng.randint(min_top, max_top)
        left = rng.randint(min_left, max_left)
        r0, r1, c0, c1 = _reserve_bounds(top, left, reserve_size, grid.shape)
        if np.any(blocked[r0:r1, c0:c1]):
            continue

        patch_top = top + (reserve_size - ph) // 2
        patch_left = left + (reserve_size - pw) // 2
        target = grid[patch_top : patch_top + ph, patch_left : patch_left + pw]
        if np.any(target != BG):
            continue
        target[:, :] = patch
        blocked[r0:r1, c0:c1] = True
        return True
    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid_h = rng.randint(GRID_SIDE_RANGE[0], GRID_SIDE_RANGE[1])
    grid_w = rng.randint(GRID_SIDE_RANGE[0], GRID_SIDE_RANGE[1])
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    blocked = np.zeros((grid_h, grid_w), dtype=bool)

    region_count = rng.choice(REGION_COUNT_CHOICES)
    red_counts = _sample_red_counts(region_count, rng)
    specs = []
    for red_count in red_counts:
        height, width = _sample_region_size(rng)
        wall_color, marker_color = rng.sample(PALETTE, 2)
        side = rng.choice(SIDES)
        marker_length = rng.choice(MARKER_LENGTH_CHOICES)
        patch = _build_region_patch(
            height=height,
            width=width,
            wall_color=wall_color,
            marker_color=marker_color,
            side=side,
            marker_length=marker_length,
            red_count=red_count,
            rng=rng,
        )
        specs.append((max(height, width), patch))

    specs.sort(key=lambda item: (item[0], item[1].shape[0] * item[1].shape[1]), reverse=True)
    for reserve_size, patch in specs:
        if not _place_region(grid, blocked, patch, reserve_size, rng):
            return None

    regions = _extract_regions(grid)
    if len(regions) != region_count:
        return None
    if sorted(region["red_count"] for region in regions) != sorted(red_counts):
        return None

    out = main(grid)
    if np.any(out == RED):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rotating-rectangle input")
