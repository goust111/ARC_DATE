from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


_CARDINALS = ((1, 0), (-1, 0), (0, 1), (0, -1))
_DIR_TO_STEP = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

_GEN_BG = 0
_GEN_FENCE = 5
_GEN_SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
_GEN_GRID_MIN = 12
_GEN_GRID_MAX = 30
_GEN_FENCE_COUNT_MAX = 4
_GEN_PLACEMENT_ATTEMPTS = 150
_GEN_ATTEMPTS = 400


@dataclass(frozen=True)
class _Fence:
    bbox: tuple[int, int, int, int]
    gap: tuple[int, int]
    step: tuple[int, int]
    seed_color: int


@dataclass(frozen=True)
class _PlacedFence:
    top: int
    left: int
    height: int
    width: int
    side: str
    gap: tuple[int, int]
    step: tuple[int, int]
    seed: tuple[int, int]
    seed_color: int


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == color
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in _CARDINALS:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.asarray(coords, dtype=int))
    return components


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _detect_fences(grid: np.ndarray, bg: int) -> list[_Fence]:
    fences: list[_Fence] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for coords in _components_of_color(grid, color):
            rows = coords[:, 0]
            cols = coords[:, 1]
            r0 = int(rows.min())
            r1 = int(rows.max())
            c0 = int(cols.min())
            c1 = int(cols.max())
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue

            border = _border_cells(r0, c0, r1, c1)
            cells = {tuple(map(int, rc)) for rc in coords}
            if len(cells) != len(border) - 1 or not cells <= border:
                continue

            missing = list(border - cells)
            if len(missing) != 1:
                continue
            gap = missing[0]
            on_horizontal = gap[0] in (r0, r1)
            on_vertical = gap[1] in (c0, c1)
            if on_horizontal and on_vertical:
                continue

            interior = grid[r0 + 1 : r1, c0 + 1 : c1]
            non_bg = np.argwhere(interior != bg)
            if non_bg.shape[0] != 1:
                continue

            seed_r, seed_c = map(int, non_bg[0])
            seed_color = int(interior[seed_r, seed_c])
            if seed_color == color:
                continue

            if gap[0] == r0:
                step = (-1, 0)
            elif gap[0] == r1:
                step = (1, 0)
            elif gap[1] == c0:
                step = (0, -1)
            else:
                step = (0, 1)

            fences.append(
                _Fence(
                    bbox=(r0, c0, r1, c1),
                    gap=gap,
                    step=step,
                    seed_color=seed_color,
                )
            )
    assert fences, "expected at least one fenced seed"
    return fences


def _emit_rays(grid: np.ndarray, fences: list[_Fence], bg: int) -> np.ndarray:
    out = grid.copy()
    h, w = grid.shape
    claimed: dict[tuple[int, int], int] = {}

    for fence in fences:
        r, c = fence.gap
        dr, dc = fence.step
        while 0 <= r < h and 0 <= c < w and int(grid[r, c]) == bg:
            prev = claimed.get((r, c))
            assert prev is None or prev == fence.seed_color, "overlapping rays must agree in color"
            claimed[(r, c)] = fence.seed_color
            r += dr
            c += dc

    for (r, c), color in claimed.items():
        out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Cast each fenced seed's color outward through the fence gap until the grid edge.

    Input structure:
    - The background is a dominant color, typically black.
    - Several 4-connected fence objects are monochrome rectangular borders with exactly one missing border cell.
    - Each fence encloses exactly one interior non-background seed pixel whose color is different from the fence.

    Rule:
    - For each fence, recover its bounding rectangle and locate the unique missing border cell that forms the gap.
    - The side containing that gap determines the outward direction, while the enclosed seed supplies the ray color.
    - Starting at the gap cell, paint background cells along that outward row or column until leaving the grid or reaching an original non-background obstacle.

    Key insight:
    - The gap encodes both the launch position and direction, while the interior seed contributes only the ray color, so solving requires combining the fence geometry with the separate seed marker.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    fences = _detect_fences(g, bg)
    return _emit_rays(g, fences, bg)


def _ray_cells(height: int, width: int, gap: tuple[int, int], step: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = gap
    dr, dc = step
    cells: list[tuple[int, int]] = []
    while 0 <= r < height and 0 <= c < width:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _expanded_bbox_cells(height: int, width: int, top: int, left: int, box_h: int, box_w: int) -> list[tuple[int, int]]:
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(height - 1, top + box_h)
    c1 = min(width - 1, left + box_w)
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _choose_seed_position(spec_top: int, spec_left: int, spec_h: int, spec_w: int, side: str, gap: tuple[int, int]) -> tuple[int, int]:
    interior = [
        (r, c)
        for r in range(spec_top + 1, spec_top + spec_h - 1)
        for c in range(spec_left + 1, spec_left + spec_w - 1)
    ]
    if side in ("up", "down"):
        filtered = [rc for rc in interior if rc[1] != gap[1]]
    else:
        filtered = [rc for rc in interior if rc[0] != gap[0]]
    return random.choice(filtered or interior)


def _render_input(height: int, width: int, specs: list[_PlacedFence]) -> np.ndarray:
    grid = np.full((height, width), _GEN_BG, dtype=int)
    for spec in specs:
        top = spec.top
        left = spec.left
        bottom = top + spec.height - 1
        right = left + spec.width - 1
        for r, c in _border_cells(top, left, bottom, right):
            if (r, c) != spec.gap:
                grid[r, c] = _GEN_FENCE
        sr, sc = spec.seed
        grid[sr, sc] = spec.seed_color
    return grid


def _expected_output(grid: np.ndarray, specs: list[_PlacedFence]) -> np.ndarray:
    fences = [
        _Fence(
            bbox=(spec.top, spec.left, spec.top + spec.height - 1, spec.left + spec.width - 1),
            gap=spec.gap,
            step=spec.step,
            seed_color=spec.seed_color,
        )
        for spec in specs
    ]
    return _emit_rays(grid, fences, _GEN_BG)


def _sample_grid_shape(fence_count: int) -> tuple[int, int]:
    min_dim = max(_GEN_GRID_MIN, 10 + 4 * fence_count)
    height = random.randint(min_dim, _GEN_GRID_MAX)
    width = random.randint(min_dim, _GEN_GRID_MAX)
    return height, width


def _sample_box_size(fence_count: int) -> tuple[int, int]:
    max_size = 8 if fence_count <= 2 else 7 if fence_count == 3 else 6
    return random.randint(4, max_size), random.randint(4, max_size)


def _place_fences(height: int, width: int, fence_count: int) -> list[_PlacedFence] | None:
    occupied = np.zeros((height, width), dtype=bool)
    sides = random.sample(list(_DIR_TO_STEP), k=fence_count)
    colors = random.sample(_GEN_SEED_COLORS, k=fence_count)
    specs: list[_PlacedFence] = []

    for side, seed_color in zip(sides, colors, strict=True):
        placed = False
        for _ in range(_GEN_PLACEMENT_ATTEMPTS):
            box_h, box_w = _sample_box_size(fence_count)
            if height - box_h - 1 < 1 or width - box_w - 1 < 1:
                return None

            top = random.randint(1, height - box_h - 1)
            left = random.randint(1, width - box_w - 1)
            bottom = top + box_h - 1
            right = left + box_w - 1

            if side == "up":
                gap = (top, random.randint(left + 1, right - 1))
            elif side == "down":
                gap = (bottom, random.randint(left + 1, right - 1))
            elif side == "left":
                gap = (random.randint(top + 1, bottom - 1), left)
            else:
                gap = (random.randint(top + 1, bottom - 1), right)

            step = _DIR_TO_STEP[side]
            seed = _choose_seed_position(top, left, box_h, box_w, side, gap)
            reserve = _expanded_bbox_cells(height, width, top, left, box_h, box_w)
            reserve.extend(_ray_cells(height, width, gap, step))
            if any(occupied[r, c] for r, c in reserve):
                continue

            for r, c in reserve:
                occupied[r, c] = True
            specs.append(
                _PlacedFence(
                    top=top,
                    left=left,
                    height=box_h,
                    width=box_w,
                    side=side,
                    gap=gap,
                    step=step,
                    seed=seed,
                    seed_color=seed_color,
                )
            )
            placed = True
            break
        if not placed:
            return None

    return specs


def _gen_attempt() -> np.ndarray | None:
    fence_count = random.randint(1, _GEN_FENCE_COUNT_MAX)
    height, width = _sample_grid_shape(fence_count)
    specs = _place_fences(height, width, fence_count)
    if specs is None:
        return None

    grid = _render_input(height, width, specs)
    expected = _expected_output(grid, specs)

    try:
        actual = main(grid)
    except Exception:
        return None

    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fenced-ray sample")
