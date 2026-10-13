from __future__ import annotations

import random

import numpy as np


SIDES = ("top", "bottom", "left", "right")
PALETTE = tuple(range(10))

GRID_HEIGHT_RANGE = (18, 31)
GRID_WIDTH_RANGE = (18, 31)
FENCE_SIZE_RANGE = (5, 11)
SEED_COUNT_RANGE = (1, 6)
MAX_FENCES = 4
MIN_CLEARANCE = 7
CLIPPED_SAMPLE_PROB = 0.5
PLACEMENT_TRIES = 200
GEN_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _border_positions(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    pts: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        pts.append((r0, c))
    for r in range(r0 + 1, r1):
        pts.append((r, c1))
    for c in range(c1, c0 - 1, -1):
        pts.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        pts.append((r, c0))
    return pts


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _frame_components_for_color(grid: np.ndarray, color: int) -> list[tuple[int, int, int, int, list[tuple[int, int]]]] | None:
    frames: list[tuple[int, int, int, int, list[tuple[int, int]]]] = []
    mask = grid == color
    for pts in _components(_dilate4(mask)):
        frame_pts = [(r, c) for r, c in pts if mask[r, c]]
        if not frame_pts:
            continue
        r0, c0, r1, c1 = _bbox(frame_pts)
        if r1 - r0 < 2 or c1 - c0 < 2:
            return None
        border = set(_border_positions(r0, c0, r1, c1))
        comp = set(frame_pts)
        if not comp <= border:
            return None
        gaps = [pos for pos in border if pos not in comp]
        if not 1 <= len(gaps) <= 4:
            return None
        frames.append((r0, c0, r1, c1, gaps))
    return frames or None


def _find_fence_color_and_frames(
    grid: np.ndarray, bg: int
) -> tuple[int, list[tuple[int, int, int, int, list[tuple[int, int]]]]]:
    candidates = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        frames = _frame_components_for_color(grid, color)
        if frames is not None:
            candidates.append((len(frames), int(np.count_nonzero(grid == color)), color, frames))
    assert candidates, "expected at least one frame color"
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return candidates[0][2], candidates[0][3]


def _gap_direction(r: int, c: int, r0: int, c0: int, r1: int, c1: int) -> tuple[int, int]:
    if r == r0:
        return -1, 0
    if r == r1:
        return 1, 0
    if c == c0:
        return 0, -1
    if c == c1:
        return 0, 1
    raise AssertionError("gap must lie on the frame boundary")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow maroon branches outward from each fence gap.

    The gap side gives the growth direction and the number of non-background,
    non-fence seed pixels inside that fence gives every branch length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = get_bg(g)
    fence_color, frames = _find_fence_color_and_frames(g, bg)

    out = g.copy()
    for r0, c0, r1, c1, gaps in frames:
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        seed_count = int(np.count_nonzero((interior != bg) & (interior != fence_color)))
        for gr, gc in gaps:
            dr, dc = _gap_direction(gr, gc, r0, c0, r1, c1)
            for step in range(1, seed_count + 1):
                nr = gr + dr * step
                nc = gc + dc * step
                if not (0 <= nr < h and 0 <= nc < w):
                    break
                out[nr, nc] = fence_color
    return out


def _row_gap(a0: int, a1: int, b0: int, b1: int) -> int:
    if a1 < b0:
        return b0 - a1 - 1
    if b1 < a0:
        return a0 - b1 - 1
    return 0


def _has_clearance(box: tuple[int, int, int, int], others: list[tuple[int, int, int, int]]) -> bool:
    r0, c0, r1, c1 = box
    for or0, oc0, or1, oc1 in others:
        row_gap = _row_gap(r0, r1, or0, or1)
        col_gap = _row_gap(c0, c1, oc0, oc1)
        if max(row_gap, col_gap) < MIN_CLEARANCE:
            return False
    return True


def _sample_gap_sides(rng: random.Random) -> dict[str, int]:
    gap_count = rng.randint(1, 4)
    chosen = rng.sample(SIDES, gap_count)
    return {side: 0 for side in chosen}


def _assign_gap_positions(
    rng: random.Random, fence_h: int, fence_w: int, gaps: dict[str, int]
) -> dict[str, tuple[int, int]]:
    positions: dict[str, tuple[int, int]] = {}
    for side in gaps:
        if side == "top":
            positions[side] = (0, rng.randint(1, fence_w - 2))
        elif side == "bottom":
            positions[side] = (fence_h - 1, rng.randint(1, fence_w - 2))
        elif side == "left":
            positions[side] = (rng.randint(1, fence_h - 2), 0)
        else:
            positions[side] = (rng.randint(1, fence_h - 2), fence_w - 1)
    return positions


def _sample_object_count(rng: random.Random, h: int, w: int) -> int:
    area = h * w
    if area >= 600:
        max_fences = 4
    elif area >= 420:
        max_fences = 3
    elif area >= 280:
        max_fences = 2
    else:
        max_fences = 1
    return rng.randint(1, min(MAX_FENCES, max_fences))


def _sample_dimensions(rng: random.Random, h: int, w: int, seed_count: int) -> tuple[int, int] | None:
    max_h = min(FENCE_SIZE_RANGE[1] - 1, h - 2)
    max_w = min(FENCE_SIZE_RANGE[1] - 1, w - 2)
    if max_h < FENCE_SIZE_RANGE[0] or max_w < FENCE_SIZE_RANGE[0]:
        return None
    for _ in range(40):
        fence_h = rng.randint(FENCE_SIZE_RANGE[0], max_h)
        fence_w = rng.randint(FENCE_SIZE_RANGE[0], max_w)
        if (fence_h - 2) * (fence_w - 2) >= seed_count:
            return fence_h, fence_w
    return None


def _box_has_clipped_branch(
    box: tuple[int, int, int, int], sides: set[str], seed_count: int, h: int, w: int
) -> bool:
    r0, c0, r1, c1 = box
    return (
        ("top" in sides and r0 < seed_count)
        or ("bottom" in sides and (h - 1 - r1) < seed_count)
        or ("left" in sides and c0 < seed_count)
        or ("right" in sides and (w - 1 - c1) < seed_count)
    )


def _sample_position(
    rng: random.Random,
    h: int,
    w: int,
    fence_h: int,
    fence_w: int,
    seed_count: int,
    sides: set[str],
    placed_boxes: list[tuple[int, int, int, int]],
    force_clip: bool,
) -> tuple[int, int] | None:
    max_r0 = h - fence_h - 1
    max_c0 = w - fence_w - 1
    if max_r0 < 1 or max_c0 < 1:
        return None

    clip_choices = list(sides) if force_clip else []
    for _ in range(PLACEMENT_TRIES):
        if clip_choices:
            clip_side = rng.choice(clip_choices)
            if clip_side == "top":
                upper = min(max_r0, seed_count - 1)
                if upper < 1:
                    continue
                r0 = rng.randint(1, upper)
                c0 = rng.randint(1, max_c0)
            elif clip_side == "bottom":
                upper = min(max_r0, seed_count - 1)
                if upper < 1:
                    continue
                dist = rng.randint(1, upper)
                r0 = h - fence_h - 1 - dist + 1
                c0 = rng.randint(1, max_c0)
            elif clip_side == "left":
                upper = min(max_c0, seed_count - 1)
                if upper < 1:
                    continue
                c0 = rng.randint(1, upper)
                r0 = rng.randint(1, max_r0)
            else:
                upper = min(max_c0, seed_count - 1)
                if upper < 1:
                    continue
                dist = rng.randint(1, upper)
                c0 = w - fence_w - 1 - dist + 1
                r0 = rng.randint(1, max_r0)
        else:
            r0 = rng.randint(1, max_r0)
            c0 = rng.randint(1, max_c0)

        box = (r0, c0, r0 + fence_h - 1, c0 + fence_w - 1)
        if not _has_clearance(box, placed_boxes):
            continue
        if force_clip and not _box_has_clipped_branch(box, sides, seed_count, h, w):
            continue
        return r0, c0
    return None


def _paint_frame(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _render_expected(
    grid: np.ndarray, specs: list[dict[str, object]], fence_color: int
) -> np.ndarray:
    out = grid.copy()
    h, w = out.shape
    for spec in specs:
        seed_count = int(spec["seed_count"])
        gaps_abs = spec["gaps_abs"]
        bbox = spec["bbox"]
        r0, c0, r1, c1 = bbox
        for gr, gc in gaps_abs.values():
            dr, dc = _gap_direction(gr, gc, r0, c0, r1, c1)
            for step in range(1, seed_count + 1):
                nr = gr + dr * step
                nc = gc + dc * step
                if not (0 <= nr < h and 0 <= nc < w):
                    break
                out[nr, nc] = fence_color
    return out


def _validate_generated(
    grid: np.ndarray,
    specs: list[dict[str, object]],
    bg: int,
    fence_color: int,
) -> bool:
    if get_bg(grid) != bg:
        return False
    colors = set(map(int, np.unique(grid)))
    if colors != {bg, fence_color, int(specs[0]["seed_color"])}:
        return False
    found_fence, frames = _find_fence_color_and_frames(grid, bg)
    if found_fence != fence_color or len(frames) != len(specs):
        return False
    expected = _render_expected(grid, specs, fence_color)
    return np.array_equal(main(grid), expected)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_HEIGHT_RANGE)
    w = rng.randint(*GRID_WIDTH_RANGE)
    bg, fence_color, seed_color = rng.sample(PALETTE, 3)

    target_count = _sample_object_count(rng, h, w)
    want_clip = rng.random() < CLIPPED_SAMPLE_PROB
    clipped_present = False

    placed_boxes: list[tuple[int, int, int, int]] = []
    specs: list[dict[str, object]] = []

    for index in range(target_count):
        placed = False
        for _ in range(PLACEMENT_TRIES):
            seed_count = rng.randint(*SEED_COUNT_RANGE)
            dims = _sample_dimensions(rng, h, w, seed_count)
            if dims is None:
                return None
            fence_h, fence_w = dims
            gap_sides = _sample_gap_sides(rng)
            gap_positions = _assign_gap_positions(rng, fence_h, fence_w, gap_sides)
            force_clip = want_clip and not clipped_present and index == 0
            position = _sample_position(
                rng,
                h,
                w,
                fence_h,
                fence_w,
                seed_count,
                set(gap_sides),
                placed_boxes,
                force_clip,
            )
            if position is None:
                continue
            r0, c0 = position
            r1 = r0 + fence_h - 1
            c1 = c0 + fence_w - 1
            box = (r0, c0, r1, c1)

            interior_cells = [
                (rr, cc)
                for rr in range(r0 + 1, r1)
                for cc in range(c0 + 1, c1)
            ]
            if len(interior_cells) < seed_count:
                continue
            seeds = rng.sample(interior_cells, seed_count)
            gaps_abs = {
                side: (r0 + rel_r, c0 + rel_c) for side, (rel_r, rel_c) in gap_positions.items()
            }

            specs.append(
                {
                    "bbox": box,
                    "gaps_abs": gaps_abs,
                    "seed_count": seed_count,
                    "seed_cells": seeds,
                    "seed_color": seed_color,
                }
            )
            placed_boxes.append(box)
            clipped_present = clipped_present or _box_has_clipped_branch(
                box, set(gap_sides), seed_count, h, w
            )
            placed = True
            break
        if not placed:
            return None

    if want_clip and not clipped_present:
        return None

    grid = np.full((h, w), bg, dtype=int)
    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        _paint_frame(grid, r0, c0, r1, c1, fence_color)
        for gr, gc in spec["gaps_abs"].values():
            grid[gr, gc] = bg
        for sr, sc in spec["seed_cells"]:
            grid[sr, sc] = seed_color

    if not _validate_generated(grid, specs, bg, fence_color):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
