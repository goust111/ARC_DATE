from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np


BLACK = 0
GREY = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SIDE_TO_DIR = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

GRID_MAX = 30
GEN_ATTEMPTS = 400
PER_OBJECT_ATTEMPTS = 80
TARGET_COUNT_CHOICES = (1, 2, 3, 4)
TARGET_COUNT_WEIGHTS = (1, 3, 3, 1)
INNER_SIZE_CHOICES = (3, 5)


@dataclass(frozen=True)
class FrameSpec:
    bbox: tuple[int, int, int, int]
    gap: tuple[int, int]
    direction: tuple[int, int]
    seed_pos: tuple[int, int]
    seed_color: int
    area: int


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _bbox_halo(r0: int, c0: int, r1: int, c1: int, h: int, w: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for r in range(max(0, r0 - 1), min(h, r1 + 2)):
        for c in range(max(0, c0 - 1), min(w, c1 + 2)):
            cells.add((r, c))
    return cells


def _halo_cells(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for nr in range(max(0, r - 1), min(h, r + 2)):
            for nc in range(max(0, c - 1), min(w, c + 2)):
                out.add((nr, nc))
    return out


def _ray_cells(
    start: tuple[int, int],
    direction: tuple[int, int],
    length: int,
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = shape
    dr, dc = direction
    r, c = start
    cells: list[tuple[int, int]] = []
    for _ in range(length):
        if not (0 <= r < h and 0 <= c < w):
            break
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _gap_from_side(r0: int, c0: int, r1: int, c1: int, side: str) -> tuple[int, int]:
    if side == "up":
        return (r0, (c0 + c1) // 2)
    if side == "down":
        return (r1, (c0 + c1) // 2)
    if side == "left":
        return ((r0 + r1) // 2, c0)
    if side == "right":
        return ((r0 + r1) // 2, c1)
    raise ValueError(f"unknown side: {side}")


def _parse_frame_component(grid: np.ndarray, comp: list[tuple[int, int]]) -> FrameSpec:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must enclose a non-empty interior"

    border = _border_cells(r0, c0, r1, c1)
    comp_set = set(comp)
    assert comp_set <= border, "grey component must lie on a rectangle border"

    missing = border - comp_set
    assert len(missing) == 1, "rectangle border must have exactly one gap"
    gap = next(iter(missing))
    gr, gc = gap

    if gr == r0 and c0 < gc < c1:
        assert (c1 - c0) % 2 == 0 and gc == (c0 + c1) // 2, "top gap must be centered"
        direction = SIDE_TO_DIR["up"]
    elif gr == r1 and c0 < gc < c1:
        assert (c1 - c0) % 2 == 0 and gc == (c0 + c1) // 2, "bottom gap must be centered"
        direction = SIDE_TO_DIR["down"]
    elif gc == c0 and r0 < gr < r1:
        assert (r1 - r0) % 2 == 0 and gr == (r0 + r1) // 2, "left gap must be centered"
        direction = SIDE_TO_DIR["left"]
    elif gc == c1 and r0 < gr < r1:
        assert (r1 - r0) % 2 == 0 and gr == (r0 + r1) // 2, "right gap must be centered"
        direction = SIDE_TO_DIR["right"]
    else:
        raise AssertionError("gap must sit at the center of exactly one side")

    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    assert inner.size > 0, "frame interior must exist"
    assert not np.any(inner == GREY), "frame interior must not contain grey pixels"

    seed_mask = (inner != BLACK) & (inner != GREY)
    seed_coords = np.argwhere(seed_mask)
    assert seed_coords.shape[0] == 1, "frame interior must contain exactly one seed pixel"
    seed_r, seed_c = map(int, seed_coords[0])
    seed_pos = (r0 + 1 + seed_r, c0 + 1 + seed_c)
    seed_color = int(grid[seed_pos])

    return FrameSpec(
        bbox=(r0, c0, r1, c1),
        gap=gap,
        direction=direction,
        seed_pos=seed_pos,
        seed_color=seed_color,
        area=(r1 - r0 - 1) * (c1 - c0 - 1),
    )


def _extract_frames(grid: np.ndarray) -> list[FrameSpec]:
    grey_comps = _components_mask(grid == GREY)
    frames: list[FrameSpec] = []
    for comp in grey_comps:
        frames.append(_parse_frame_component(grid, comp))
    assert frames, "expected at least one grey rectangle"
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a seed-colored ray outward through each grey rectangle's single border gap, with ray length equal to the rectangle's interior area.

    Input structure:
    - The background is black.
    - One or more grey 1-pixel-thick rectangular borders appear on the grid.
    - Each grey rectangle has exactly one centered 1-pixel gap on one side.
    - Each rectangle interior contains exactly one non-black, non-grey seed pixel.

    Rule:
    - For each grey rectangle, identify its bounding box, its unique border gap, and the unique seed color inside.
    - Compute the interior cavity area as the number of cells strictly inside the border.
    - Starting at the gap cell itself, draw a straight ray outward in the gap's facing direction using the seed color.
    - The ray length equals the cavity area, truncated if it reaches the grid boundary; keep the original grey borders, seed pixels, and all unrelated cells unchanged.

    Key insight:
    - The gap contributes only the outward direction, while the interior seed supplies the paint color and the interior area supplies the ray length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h <= GRID_MAX and w <= GRID_MAX, "ARC grids must be at most 30x30"

    out = g.copy()
    for frame in _extract_frames(g):
        for r, c in _ray_cells(frame.gap, frame.direction, frame.area, g.shape):
            if out[r, c] == BLACK:
                out[r, c] = frame.seed_color
    return out


def _sample_target_count(rng: random.Random) -> int:
    return rng.choices(TARGET_COUNT_CHOICES, weights=TARGET_COUNT_WEIGHTS, k=1)[0]


def _sample_sides(rng: random.Random, n: int) -> list[str]:
    sides = list(SIDE_TO_DIR)
    rng.shuffle(sides)
    chosen = sides[: min(n, len(sides))]
    while len(chosen) < n:
        chosen.append(rng.choice(sides))
    rng.shuffle(chosen)
    return chosen


def _sample_trunc_flags(rng: random.Random, n: int) -> list[bool]:
    if n == 1:
        return [bool(rng.randrange(2))]
    flags = [False, True]
    while len(flags) < n:
        flags.append(bool(rng.randrange(2)))
    rng.shuffle(flags)
    return flags


def _candidate_placements(
    h: int,
    w: int,
    inner_h: int,
    inner_w: int,
    side: str,
    truncated: bool,
    protected: set[tuple[int, int]],
) -> list[tuple[int, int, list[tuple[int, int]], set[tuple[int, int]]]]:
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    candidates: list[tuple[int, int, list[tuple[int, int]], set[tuple[int, int]]]] = []
    for r0 in range(1, h - outer_h):
        for c0 in range(1, w - outer_w):
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1
            gap = _gap_from_side(r0, c0, r1, c1, side)
            direction = SIDE_TO_DIR[side]
            ray = _ray_cells(gap, direction, inner_h * inner_w, (h, w))
            if truncated != (len(ray) < inner_h * inner_w):
                continue

            reserved = _bbox_halo(r0, c0, r1, c1, h, w) | _halo_cells(set(ray), h, w)
            if reserved & protected:
                continue
            candidates.append((r0, c0, ray, reserved))
    return candidates


def _place_object(
    grid: np.ndarray,
    protected: set[tuple[int, int]],
    side: str,
    truncated: bool,
    color: int,
    rng: random.Random,
) -> dict[str, object] | None:
    h, w = map(int, grid.shape)
    for _ in range(PER_OBJECT_ATTEMPTS):
        inner_h = rng.choice(INNER_SIZE_CHOICES)
        inner_w = rng.choice(INNER_SIZE_CHOICES)
        candidates = _candidate_placements(h, w, inner_h, inner_w, side, truncated, protected)
        if not candidates:
            continue

        r0, c0, ray, reserved = rng.choice(candidates)
        r1 = r0 + inner_h + 1
        c1 = c0 + inner_w + 1
        gap = _gap_from_side(r0, c0, r1, c1, side)

        for r, c in _border_cells(r0, c0, r1, c1):
            if (r, c) != gap:
                grid[r, c] = GREY

        seed_r = rng.randint(r0 + 1, r1 - 1)
        seed_c = rng.randint(c0 + 1, c1 - 1)
        grid[seed_r, seed_c] = color
        protected.update(reserved)

        return {
            "bbox": (r0, c0, r1, c1),
            "gap": gap,
            "direction": SIDE_TO_DIR[side],
            "seed_pos": (seed_r, seed_c),
            "seed_color": color,
            "area": inner_h * inner_w,
            "truncated": truncated,
            "ray": ray,
        }
    return None


def _validate_generated(grid: np.ndarray, expected_count: int) -> bool:
    try:
        frames = _extract_frames(grid)
    except AssertionError:
        return False

    if len(frames) != expected_count:
        return False
    if len({frame.seed_color for frame in frames}) != expected_count:
        return False

    out = main(grid)
    if out.shape != grid.shape or np.array_equal(out, grid):
        return False

    expected = grid.copy()
    changed = 0
    trunc_seen = False
    full_seen = False
    for frame in frames:
        ray = _ray_cells(frame.gap, frame.direction, frame.area, grid.shape)
        if len(ray) < frame.area:
            trunc_seen = True
        else:
            full_seen = True
        for r, c in ray:
            if expected[r, c] == BLACK:
                expected[r, c] = frame.seed_color
                changed += 1

    if changed == 0 or not np.array_equal(out, expected):
        return False

    if expected_count >= 2 and not (trunc_seen and full_seen):
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    target_count = _sample_target_count(rng)
    min_side = min(GRID_MAX, 12 + 4 * (target_count - 1))
    h = rng.randint(min_side, GRID_MAX)
    w = rng.randint(min_side, GRID_MAX)
    grid = np.full((h, w), BLACK, dtype=int)

    colors = list(SEED_COLORS)
    rng.shuffle(colors)
    colors = colors[:target_count]

    sides = _sample_sides(rng, target_count)
    trunc_flags = _sample_trunc_flags(rng, target_count)

    protected: set[tuple[int, int]] = set()
    placed: list[dict[str, object]] = []
    for side, truncated, color in zip(sides, trunc_flags, colors):
        spec = _place_object(grid, protected, side, truncated, color, rng)
        if spec is None:
            return None
        placed.append(spec)

    if len(placed) != target_count:
        return None
    if not _validate_generated(grid, target_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
