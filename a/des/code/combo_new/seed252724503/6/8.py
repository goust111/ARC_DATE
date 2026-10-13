from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


BLACK = 0
GREY = 5
TOWER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

MIN_GRID_SIZE = 18
MAX_GRID_SIZE = 28
MIN_FENCE_SIZE = 4
MAX_FENCE_SIZE = 8
MAX_FENCES = 5

SCENARIO_RETRIES = 120
GENERATION_RETRIES = 300
EXTRA_PLACEMENT_RETRIES = 160


@dataclass(frozen=True)
class Fence:
    r0: int
    c0: int
    r1: int
    c1: int
    tower_r: int
    tower_c: int
    color: int

    @property
    def height(self) -> int:
        return self.r1 - self.r0 + 1

    @property
    def width(self) -> int:
        return self.c1 - self.c0 + 1


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _frame_mask(height: int, width: int) -> np.ndarray:
    frame = np.zeros((height, width), dtype=bool)
    frame[0, :] = True
    frame[-1, :] = True
    frame[:, 0] = True
    frame[:, -1] = True
    return frame


def _parse_fences(grid: np.ndarray) -> list[Fence]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    grey_mask = g == GREY
    comps = _components(grey_mask)
    assert comps, "expected at least one grey rectangular fence"

    fences: list[Fence] = []
    tower_positions: set[tuple[int, int]] = set()
    tower_colors: set[int] = set()

    for coords in comps:
        r0 = int(coords[:, 0].min())
        c0 = int(coords[:, 1].min())
        r1 = int(coords[:, 0].max())
        c1 = int(coords[:, 1].max())
        assert 0 < r0 < r1 < h - 1, "fence must stay away from top/bottom borders"
        assert 0 < c0 < c1 < w - 1, "fence must stay away from left/right borders"

        height = r1 - r0 + 1
        width = c1 - c0 + 1
        assert height >= 3 and width >= 3, "fence must enclose an interior"
        patch = grey_mask[r0 : r1 + 1, c0 : c1 + 1]
        assert np.array_equal(patch, _frame_mask(height, width)), "grey component must be a 1-pixel rectangular frame"

        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        assert interior.size > 0
        assert not np.any(interior == GREY), "fence interior must not contain grey wall pixels"

        tower_rel = np.argwhere((interior != BLACK) & (interior != GREY))
        assert tower_rel.shape[0] == 1, "each fence must contain exactly one colored tower"
        tr = r0 + 1 + int(tower_rel[0, 0])
        tc = c0 + 1 + int(tower_rel[0, 1])
        color = int(g[tr, tc])
        assert color not in (BLACK, GREY)
        assert color not in tower_colors, "tower colors must be unique"

        fence = Fence(r0, c0, r1, c1, tr, tc, color)
        fences.append(fence)
        tower_positions.add((tr, tc))
        tower_colors.add(color)

    colored = {
        (int(r), int(c))
        for r, c in np.argwhere((g != BLACK) & (g != GREY))
    }
    assert colored == tower_positions, "all non-black non-grey cells must be towers"
    return fences


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each fenced tower's color along its row and column, skipping its own grey wall and continuing outside until another wall or the grid edge.

    Input structure:
    - The background is black.
    - Several disjoint grey 1-pixel-thick rectangular fences sit strictly inside the grid.
    - Each fence interior contains exactly one non-black, non-grey colored tower pixel, and tower colors are unique.

    Rule:
    - For every fence, locate its tower at position (r, c) with color C.
    - Fill the tower's interior row and interior column inside the same fence with C, stopping before the grey wall.
    - The grey wall itself is never recolored.
    - Beyond that same fence's wall, continue the projection outward in the same four directions, recoloring black cells with C until another grey wall or the grid boundary is reached.
    - If multiple outside projections reach the same cell, the tower with the larger row index wins; if rows tie, the tower with the larger column index wins.

    Key insight:
    - Each tower creates a same-color cross, but the cross behaves differently inside and outside its fence: it fills the interior, skips the wall, then resumes in open space until a new wall blocks it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"
    fences = _parse_fences(g)

    out = g.copy()

    for fence in fences:
        out[fence.tower_r, fence.c0 + 1 : fence.c1] = fence.color
        out[fence.r0 + 1 : fence.r1, fence.tower_c] = fence.color

    for fence in sorted(fences, key=lambda item: (item.tower_r, item.tower_c)):
        for rr in range(fence.r0 - 1, -1, -1):
            if g[rr, fence.tower_c] == GREY:
                break
            assert g[rr, fence.tower_c] == BLACK
            out[rr, fence.tower_c] = fence.color
        for rr in range(fence.r1 + 1, g.shape[0]):
            if g[rr, fence.tower_c] == GREY:
                break
            assert g[rr, fence.tower_c] == BLACK
            out[rr, fence.tower_c] = fence.color
        for cc in range(fence.c0 - 1, -1, -1):
            if g[fence.tower_r, cc] == GREY:
                break
            assert g[fence.tower_r, cc] == BLACK
            out[fence.tower_r, cc] = fence.color
        for cc in range(fence.c1 + 1, g.shape[1]):
            if g[fence.tower_r, cc] == GREY:
                break
            assert g[fence.tower_r, cc] == BLACK
            out[fence.tower_r, cc] = fence.color

    return out


def _make_fence(r0: int, c0: int, height: int, width: int, tower_r: int, tower_c: int, color: int) -> Fence:
    return Fence(r0, c0, r0 + height - 1, c0 + width - 1, tower_r, tower_c, color)


def _separated(a: Fence, b: Fence) -> bool:
    return a.r1 + 1 < b.r0 or b.r1 + 1 < a.r0 or a.c1 + 1 < b.c0 or b.c1 + 1 < a.c0


def _can_add(candidate: Fence, fences: list[Fence], forbidden: np.ndarray) -> bool:
    if forbidden[candidate.r0 : candidate.r1 + 1, candidate.c0 : candidate.c1 + 1].any():
        return False
    return all(_separated(candidate, other) for other in fences)


def _draw_fences(shape: tuple[int, int], fences: list[Fence]) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for fence in fences:
        grid[fence.r0 : fence.r1 + 1, fence.c0] = GREY
        grid[fence.r0 : fence.r1 + 1, fence.c1] = GREY
        grid[fence.r0, fence.c0 : fence.c1 + 1] = GREY
        grid[fence.r1, fence.c0 : fence.c1 + 1] = GREY
        grid[fence.tower_r, fence.tower_c] = fence.color
    return grid


def _mark_horizontal(mask: np.ndarray, row: int, c0: int, c1: int) -> None:
    if 0 <= row < mask.shape[0]:
        lo = max(0, min(c0, c1))
        hi = min(mask.shape[1] - 1, max(c0, c1))
        if lo <= hi:
            mask[row, lo : hi + 1] = True


def _mark_vertical(mask: np.ndarray, col: int, r0: int, r1: int) -> None:
    if 0 <= col < mask.shape[1]:
        lo = max(0, min(r0, r1))
        hi = min(mask.shape[0] - 1, max(r0, r1))
        if lo <= hi:
            mask[lo : hi + 1, col] = True


def _scenario_cross(h: int, w: int, colors: list[int]) -> tuple[list[Fence], np.ndarray] | None:
    for _ in range(SCENARIO_RETRIES):
        ah = random.randint(5, min(8, h - 6))
        aw = random.randint(5, min(8, w - 6))
        bh = random.randint(5, min(8, h - 6))
        bw = random.randint(5, min(8, w - 6))

        ar0_hi = h - ah - bh - 3
        ac0_hi = w - aw - bw - 3
        if ar0_hi < 1 or ac0_hi < 1:
            continue

        ar0 = random.randint(1, ar0_hi)
        ac0 = random.randint(1, ac0_hi)
        a = _make_fence(
            ar0,
            ac0,
            ah,
            aw,
            random.randint(ar0 + 1, ar0 + ah - 2),
            random.randint(ac0 + 1, ac0 + aw - 2),
            colors[0],
        )

        br0_lo = a.r1 + 2
        bc0_lo = a.c1 + 2
        br0_hi = h - bh - 1
        bc0_hi = w - bw - 1
        if br0_lo > br0_hi or bc0_lo > bc0_hi:
            continue

        br0 = random.randint(br0_lo, br0_hi)
        bc0 = random.randint(bc0_lo, bc0_hi)
        b = _make_fence(
            br0,
            bc0,
            bh,
            bw,
            random.randint(br0 + 1, br0 + bh - 2),
            random.randint(bc0 + 1, bc0 + bw - 2),
            colors[1],
        )

        forbidden = np.zeros((h, w), dtype=bool)
        _mark_horizontal(forbidden, a.tower_r, a.c1 + 1, w - 1)
        _mark_vertical(forbidden, b.tower_c, 0, b.r0 - 1)
        return [a, b], forbidden
    return None


def _scenario_faceoff(h: int, w: int, colors: list[int]) -> tuple[list[Fence], np.ndarray] | None:
    for _ in range(SCENARIO_RETRIES):
        height = random.randint(5, min(8, h - 4))
        aw = random.randint(4, min(8, w - 8))
        bw = random.randint(4, min(8, w - aw - 4))

        r0_hi = h - height - 1
        if r0_hi < 1:
            continue
        r0 = random.randint(1, r0_hi)
        shared_row = random.randint(r0 + 1, r0 + height - 2)

        ac0_hi = w - aw - bw - 3
        if ac0_hi < 1:
            continue
        ac0 = random.randint(1, ac0_hi)
        a = _make_fence(
            r0,
            ac0,
            height,
            aw,
            shared_row,
            random.randint(ac0 + 1, ac0 + aw - 2),
            colors[0],
        )

        bc0_lo = a.c1 + 2
        bc0_hi = w - bw - 1
        if bc0_lo > bc0_hi:
            continue
        bc0 = random.randint(bc0_lo, bc0_hi)
        b = _make_fence(
            r0,
            bc0,
            height,
            bw,
            shared_row,
            random.randint(bc0 + 1, bc0 + bw - 2),
            colors[1],
        )

        forbidden = np.zeros((h, w), dtype=bool)
        _mark_horizontal(forbidden, shared_row, 0, w - 1)
        return [a, b], forbidden
    return None


def _scenario_blocked(h: int, w: int, colors: list[int]) -> tuple[list[Fence], np.ndarray] | None:
    for _ in range(SCENARIO_RETRIES):
        ah = random.randint(4, min(7, h - 6))
        aw = random.randint(4, min(7, w - 8))
        bh = random.randint(5, min(8, h - 4))
        bw = random.randint(4, min(8, w - aw - 4))

        ar0_hi = h - ah - 1
        ac0_hi = w - aw - bw - 3
        if ar0_hi < 1 or ac0_hi < 1:
            continue
        ar0 = random.randint(1, ar0_hi)
        ac0 = random.randint(1, ac0_hi)
        tower_r = random.randint(ar0 + 1, ar0 + ah - 2)
        a = _make_fence(
            ar0,
            ac0,
            ah,
            aw,
            tower_r,
            random.randint(ac0 + 1, ac0 + aw - 2),
            colors[0],
        )

        br0_lo = max(1, tower_r - (bh - 2))
        br0_hi = min(tower_r - 1, h - bh - 1)
        bc0_lo = a.c1 + 2
        bc0_hi = w - bw - 1
        if br0_lo > br0_hi or bc0_lo > bc0_hi:
            continue

        br0 = random.randint(br0_lo, br0_hi)
        bc0 = random.randint(bc0_lo, bc0_hi)
        btower_choices = [r for r in range(br0 + 1, br0 + bh - 1) if r != tower_r]
        if not btower_choices:
            continue
        b = _make_fence(
            br0,
            bc0,
            bh,
            bw,
            random.choice(btower_choices),
            random.randint(bc0 + 1, bc0 + bw - 2),
            colors[1],
        )

        forbidden = np.zeros((h, w), dtype=bool)
        _mark_horizontal(forbidden, tower_r, a.c1 + 1, b.c0 - 1)
        return [a, b], forbidden
    return None


def _add_extra_fences(h: int, w: int, fences: list[Fence], forbidden: np.ndarray, colors: list[int], target_total: int) -> bool:
    pool = colors[len(fences) :]
    attempts = 0
    while len(fences) < target_total and attempts < EXTRA_PLACEMENT_RETRIES:
        attempts += 1
        if not pool:
            return False

        height = random.randint(MIN_FENCE_SIZE, min(MAX_FENCE_SIZE, h - 2))
        width = random.randint(MIN_FENCE_SIZE, min(MAX_FENCE_SIZE, w - 2))
        r0_hi = h - height - 1
        c0_hi = w - width - 1
        if r0_hi < 1 or c0_hi < 1:
            continue

        color = pool[0]
        candidate = _make_fence(
            random.randint(1, r0_hi),
            random.randint(1, c0_hi),
            height,
            width,
            0,
            0,
            color,
        )
        tower_r = random.randint(candidate.r0 + 1, candidate.r1 - 1)
        tower_c = random.randint(candidate.c0 + 1, candidate.c1 - 1)
        candidate = Fence(candidate.r0, candidate.c0, candidate.r1, candidate.c1, tower_r, tower_c, color)

        if not _can_add(candidate, fences, forbidden):
            continue

        fences.append(candidate)
        pool = pool[1:]

    return len(fences) == target_total


def _random_transform(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, copy=True)
    out = np.rot90(out, random.randint(0, 3))
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    return np.ascontiguousarray(out)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)
    w = random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)
    colors = list(TOWER_COLORS)
    random.shuffle(colors)

    scenario = random.choice(("cross", "faceoff", "blocked"))
    if scenario == "cross":
        built = _scenario_cross(h, w, colors)
    elif scenario == "faceoff":
        built = _scenario_faceoff(h, w, colors)
    else:
        built = _scenario_blocked(h, w, colors)
    if built is None:
        return None

    fences, forbidden = built
    target_total = random.randint(2, MAX_FENCES)
    if not _add_extra_fences(h, w, fences, forbidden, colors, target_total):
        return None

    grid = _draw_fences((h, w), fences)
    grid = _random_transform(grid)

    parsed = _parse_fences(grid)
    assert len(parsed) == len(fences)
    _ = main(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fenced-tower projection input")
