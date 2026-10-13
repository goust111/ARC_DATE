from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
GREY = 5
FILL_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

# Generator configuration.
FRAME_COUNT_MIN = 1
FRAME_COUNT_MAX = 4
GRID_BASE_MIN = 12
GRID_GROWTH_PER_FRAME = 3
GRID_SIDE_MAX = 30
OUTER_HEIGHT_RANGE = (4, 10)
OUTER_WIDTH_RANGE = (4, 11)
PLACEMENT_RETRIES = 120
NOISE_COUNT_RANGE = (1, 8)
GENERATION_RETRIES = 400


FrameSpec = tuple[int, int, int, int, int, int]


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _classify_frame(g: np.ndarray, coords: np.ndarray) -> FrameSpec | None:
    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    if height < 3 or width < 3:
        return None

    region = g[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros((height, width), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    if int(coords.shape[0]) != int(border.sum()):
        return None
    if not bool(np.all(region[border] == GREY)):
        return None
    if bool(np.any(region[~border] == GREY)):
        return None

    interior = region[1:-1, 1:-1]
    colors = [int(v) for v in np.unique(interior) if int(v) not in (BLACK, GREY)]
    if len(colors) != 2:
        return None

    counts = Counter(int(v) for v in interior.ravel() if int(v) not in (BLACK, GREY))
    a, b = colors
    if counts[a] == counts[b]:
        return None

    majority = int(a if counts[a] > counts[b] else b)
    minority = int(b if majority == a else a)
    return (r0, c0, r1, c1, majority, minority)


def _find_voting_frames(g: np.ndarray) -> list[FrameSpec]:
    frames: list[FrameSpec] = []
    for coords in _components(g == GREY):
        spec = _classify_frame(g, coords)
        if spec is not None:
            frames.append(spec)
    frames.sort(key=lambda item: item[:4])
    return frames


def _apply_specs(g: np.ndarray, frames: list[FrameSpec]) -> np.ndarray:
    out = np.array(g, dtype=int).copy()
    for r0, c0, r1, c1, majority, minority in frames:
        out[r0, c0 : c1 + 1] = minority
        out[r1, c0 : c1 + 1] = minority
        out[r0 : r1 + 1, c0] = minority
        out[r0 : r1 + 1, c1] = minority

        src_inner = g[r0 + 1 : r1, c0 + 1 : c1]
        dst_inner = out[r0 + 1 : r1, c0 + 1 : c1]
        dst_inner[src_inner == minority] = majority
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each true grey frame with the losing interior color and flip minority votes inside.

    Input structure:
    - The background is black.
    - Some grey connected components are one-cell-thick hollow rectangles, and their interiors contain exactly two non-black colors with unequal counts plus black empty cells.
    - Additional grey single pixels may appear outside those rectangles as distractors and do not form closed frames.

    Rule:
    - Detect only grey components that are complete axis-aligned rectangular borders.
    - For each detected frame, count the two interior colors to find majority color M and minority color m.
    - Recolor the entire grey border to m.
    - Change every interior m pixel to M, while leaving black interior cells and all non-frame grey distractors unchanged.

    Key insight:
    - The border takes the minority color rather than the majority one, so the solver must first separate real rectangular frames from grey noise and then invert the vote result.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")
    return _apply_specs(g, _find_voting_frames(g))


def _reserve_halo(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    h, w = mask.shape
    rr0 = max(0, int(r0) - 1)
    cc0 = max(0, int(c0) - 1)
    rr1 = min(h - 1, int(r1) + 1)
    cc1 = min(w - 1, int(c1) + 1)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _sample_interior(height: int, width: int, majority: int, minority: int) -> np.ndarray:
    total = int(height * width)
    assert total >= 4

    black_count = random.randint(1, total - 3)
    colored_count = total - black_count
    minority_max = (colored_count - 1) // 2
    minority_count = random.randint(1, minority_max)
    majority_count = colored_count - minority_count

    cells = [(r, c) for r in range(height) for c in range(width)]
    random.shuffle(cells)
    patch = np.full((height, width), BLACK, dtype=int)

    for r, c in cells[:majority_count]:
        patch[r, c] = majority
    for r, c in cells[majority_count : majority_count + minority_count]:
        patch[r, c] = minority
    return patch


def _place_frame(grid: np.ndarray, reserved: np.ndarray) -> FrameSpec | None:
    h, w = grid.shape

    for _ in range(PLACEMENT_RETRIES):
        outer_h = random.randint(
            OUTER_HEIGHT_RANGE[0], min(OUTER_HEIGHT_RANGE[1], int(h) - 2)
        )
        outer_w = random.randint(
            OUTER_WIDTH_RANGE[0], min(OUTER_WIDTH_RANGE[1], int(w) - 2)
        )
        if (outer_h - 2) * (outer_w - 2) < 4:
            continue

        r0 = random.randint(1, int(h) - outer_h - 1)
        c0 = random.randint(1, int(w) - outer_w - 1)
        r1 = int(r0 + outer_h - 1)
        c1 = int(c0 + outer_w - 1)

        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(h - 1, r1 + 1)
        cc1 = min(w - 1, c1 + 1)
        if bool(np.any(reserved[rr0 : rr1 + 1, cc0 : cc1 + 1])):
            continue

        majority, minority = random.sample(FILL_COLORS, 2)
        interior = _sample_interior(outer_h - 2, outer_w - 2, majority, minority)

        grid[r0, c0 : c1 + 1] = GREY
        grid[r1, c0 : c1 + 1] = GREY
        grid[r0 : r1 + 1, c0] = GREY
        grid[r0 : r1 + 1, c1] = GREY
        grid[r0 + 1 : r1, c0 + 1 : c1] = interior

        _reserve_halo(reserved, r0, c0, r1, c1)
        return (r0, c0, r1, c1, majority, minority)
    return None


def _place_noise(grid: np.ndarray, reserved: np.ndarray) -> int:
    target = random.randint(*NOISE_COUNT_RANGE)
    placed = 0

    for _ in range(target):
        candidates = np.argwhere(~reserved)
        if int(candidates.shape[0]) == 0:
            break
        r, c = map(int, candidates[random.randrange(int(candidates.shape[0]))])
        grid[r, c] = GREY
        _reserve_halo(reserved, r, c, r, c)
        placed += 1
    return placed


def _attempt() -> np.ndarray | None:
    frame_count = random.randint(FRAME_COUNT_MIN, FRAME_COUNT_MAX)
    side_min = min(GRID_SIDE_MAX, GRID_BASE_MIN + GRID_GROWTH_PER_FRAME * (frame_count - 1))
    height = random.randint(side_min, GRID_SIDE_MAX)
    width = random.randint(side_min, GRID_SIDE_MAX)

    grid = np.zeros((height, width), dtype=int)
    reserved = np.zeros((height, width), dtype=bool)
    specs: list[FrameSpec] = []

    for _ in range(frame_count):
        spec = _place_frame(grid, reserved)
        if spec is None:
            return None
        specs.append(spec)

    noise_count = _place_noise(grid, reserved)
    if noise_count == 0:
        return None

    found = _find_voting_frames(grid)
    expected = _apply_specs(grid, specs)
    if found != sorted(specs, key=lambda item: item[:4]):
        return None
    if not bool(np.array_equal(main(grid), expected)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
