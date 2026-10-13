from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
SIDES = ("top", "bottom", "left", "right")

GRID_MIN = 17
GRID_MAX = 30
OUTER_SIZE_MIN = 6
OUTER_SIZE_MAX = 8
FRAME_COUNT_MIN = 1
FRAME_COUNT_MAX = 4
GENERATION_ATTEMPTS = 400


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _color_components(grid: np.ndarray, bg: int = BG) -> list[tuple[int, list[tuple[int, int]]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[tuple[int, list[tuple[int, int]]]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append((color, pts))
    return comps


def _side_positions(r0: int, c0: int, r1: int, c1: int) -> dict[str, list[tuple[int, int]]]:
    return {
        "top": [(r0, c) for c in range(c0, c1 + 1)],
        "bottom": [(r1, c) for c in range(c0, c1 + 1)],
        "left": [(r, c0) for r in range(r0, r1 + 1)],
        "right": [(r, c1) for r in range(r0, r1 + 1)],
    }


def _is_contiguous_gap(points: list[tuple[int, int]], side: str) -> bool:
    if not points:
        return False
    coords = sorted({c for _, c in points} if side in ("top", "bottom") else {r for r, _ in points})
    return coords[-1] - coords[0] + 1 == len(coords)


def _analyze_frame_component(
    grid: np.ndarray,
    color: int,
    points: list[tuple[int, int]],
) -> dict[str, object] | None:
    r0, c0, r1, c1 = _bbox(points)
    if r1 - r0 + 1 < 5 or c1 - c0 + 1 < 5:
        return None

    border_by_side = _side_positions(r0, c0, r1, c1)
    border_cells = set().union(*border_by_side.values())
    point_set = set(points)
    if not point_set.issubset(border_cells):
        return None

    missing = sorted(border_cells - point_set)
    if not missing:
        return None

    candidate_sides = [
        side
        for side, cells in border_by_side.items()
        if set(missing).issubset(set(cells)) and _is_contiguous_gap(missing, side)
    ]
    if len(candidate_sides) != 1:
        return None
    gap_side = candidate_sides[0]

    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    if inner.size == 0:
        return None
    inner_colors = {int(v) for v in inner.reshape(-1) if int(v) != BG}
    if inner_colors != {v for v in inner_colors if v != color} or len(inner_colors) != 1:
        return None

    fill_color = next(iter(inner_colors))
    pattern_mask = inner == fill_color
    if (
        np.any(pattern_mask[0, :])
        or np.any(pattern_mask[-1, :])
        or np.any(pattern_mask[:, 0])
        or np.any(pattern_mask[:, -1])
    ):
        return None

    return {
        "border_color": color,
        "fill_color": fill_color,
        "bbox": (r0, c0, r1, c1),
        "gap_side": gap_side,
        "pattern_mask": pattern_mask,
    }


def _find_frames(grid: np.ndarray) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for color, points in _color_components(grid, bg=BG):
        info = _analyze_frame_component(grid, color, points)
        if info is not None:
            frames.append(info)
    frames.sort(key=lambda item: tuple(item["bbox"]))  # stable iteration only
    return frames


def _projection_counts(pattern_mask: np.ndarray, gap_side: str) -> np.ndarray:
    if gap_side in ("top", "bottom"):
        return pattern_mask.sum(axis=0).astype(int)
    return pattern_mask.sum(axis=1).astype(int)


def _paint_projection_fill(out: np.ndarray, frame: dict[str, object]) -> None:
    r0, c0, r1, c1 = frame["bbox"]
    fill_color = int(frame["fill_color"])
    gap_side = str(frame["gap_side"])
    pattern_mask = np.asarray(frame["pattern_mask"], dtype=bool)
    counts = _projection_counts(pattern_mask, gap_side)

    inner_top, inner_bottom = r0 + 1, r1 - 1
    inner_left, inner_right = c0 + 1, c1 - 1
    out[inner_top : inner_bottom + 1, inner_left : inner_right + 1] = BG

    if gap_side == "top":
        for offset, count in enumerate(counts.tolist()):
            if count > 0:
                c = inner_left + offset
                out[inner_top : inner_top + count, c] = fill_color
    elif gap_side == "bottom":
        for offset, count in enumerate(counts.tolist()):
            if count > 0:
                c = inner_left + offset
                out[inner_bottom - count + 1 : inner_bottom + 1, c] = fill_color
    elif gap_side == "left":
        for offset, count in enumerate(counts.tolist()):
            if count > 0:
                r = inner_top + offset
                out[r, inner_left : inner_left + count] = fill_color
    else:
        for offset, count in enumerate(counts.tolist()):
            if count > 0:
                r = inner_top + offset
                out[r, inner_right - count + 1 : inner_right + 1] = fill_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the side with the border gap to choose a projection axis, then refill the
    frame interior with projection-depth bars grown inward from that side.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frames = _find_frames(g)
    assert frames, "expected at least one gapped rectangular frame"

    out = g.copy()
    for frame in frames:
        _paint_projection_fill(out, frame)
    return out


def _random_connected_mask(height: int, width: int, rng: random.Random) -> np.ndarray | None:
    area = height * width
    if area < 2:
        return None

    min_cells = 2 if area <= 4 else 3
    max_cells = max(min_cells, min(area - 1, int(np.ceil(area * 0.7))))
    target = rng.randint(min_cells, max_cells)

    mask = np.zeros((height, width), dtype=bool)
    start = (rng.randrange(height), rng.randrange(width))
    mask[start] = True
    cells = [start]

    stalled = 0
    while len(cells) < target and stalled < area * 20:
        stalled += 1
        base_r, base_c = rng.choice(cells)
        neighbors = []
        for dr, dc in DIRS4:
            nr, nc = base_r + dr, base_c + dc
            if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                neighbors.append((nr, nc))
        if not neighbors:
            continue
        nr, nc = rng.choice(neighbors)
        mask[nr, nc] = True
        cells.append((nr, nc))
        stalled = 0

    if len(cells) != target:
        return None
    if mask.any(axis=0).sum() < 2 or mask.any(axis=1).sum() < 2:
        return None

    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    if np.all(mask[r0 : r1 + 1, c0 : c1 + 1]):
        return None
    return mask


def _build_pattern(inner_h: int, inner_w: int, gap_side: str, fill_color: int, rng: random.Random) -> np.ndarray | None:
    if inner_h < 4 or inner_w < 4:
        return None

    core_h, core_w = inner_h - 2, inner_w - 2
    for _ in range(60):
        core = _random_connected_mask(core_h, core_w, rng)
        if core is None:
            continue
        pattern = np.zeros((inner_h, inner_w), dtype=int)
        pattern[1:-1, 1:-1][core] = fill_color
        counts = _projection_counts(pattern == fill_color, gap_side)
        if np.count_nonzero(counts) < 2:
            continue
        return pattern
    return None


def _gap_spec(side: str, outer_h: int, outer_w: int, rng: random.Random) -> tuple[int, int]:
    side_len = outer_w if side in ("top", "bottom") else outer_h
    usable = side_len - 2
    gap_len = rng.randint(1, max(1, usable - 1))
    start = rng.randint(1, side_len - gap_len - 1)
    return start, gap_len


def _make_frame_spec(border_color: int, fill_color: int, gap_side: str, rng: random.Random) -> dict[str, object] | None:
    outer_h = rng.randint(OUTER_SIZE_MIN, OUTER_SIZE_MAX)
    outer_w = rng.randint(OUTER_SIZE_MIN, OUTER_SIZE_MAX)
    inner_h, inner_w = outer_h - 2, outer_w - 2

    pattern = _build_pattern(inner_h, inner_w, gap_side, fill_color, rng)
    if pattern is None:
        return None

    gap_start, gap_len = _gap_spec(gap_side, outer_h, outer_w, rng)
    return {
        "border_color": border_color,
        "fill_color": fill_color,
        "gap_side": gap_side,
        "outer_h": outer_h,
        "outer_w": outer_w,
        "gap_start": gap_start,
        "gap_len": gap_len,
        "pattern": pattern,
    }


def _mark_occupied(occ: np.ndarray, r0: int, c0: int, height: int, width: int, pad: int = 1) -> None:
    h, w = occ.shape
    rr0 = max(0, r0 - pad)
    rr1 = min(h, r0 + height + pad)
    cc0 = max(0, c0 - pad)
    cc1 = min(w, c0 + width + pad)
    occ[rr0:rr1, cc0:cc1] = True


def _candidate_positions(occ: np.ndarray, height: int, width: int) -> list[tuple[int, int]]:
    h, w = occ.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(1, h - height):
        for c0 in range(1, w - width):
            rr0 = max(0, r0 - 1)
            rr1 = min(h, r0 + height + 1)
            cc0 = max(0, c0 - 1)
            cc1 = min(w, c0 + width + 1)
            if not occ[rr0:rr1, cc0:cc1].any():
                positions.append((r0, c0))
    return positions


def _paint_frame(grid: np.ndarray, top: int, left: int, spec: dict[str, object]) -> None:
    border_color = int(spec["border_color"])
    gap_side = str(spec["gap_side"])
    outer_h = int(spec["outer_h"])
    outer_w = int(spec["outer_w"])
    gap_start = int(spec["gap_start"])
    gap_len = int(spec["gap_len"])
    pattern = np.asarray(spec["pattern"], dtype=int)

    bottom = top + outer_h - 1
    right = left + outer_w - 1

    grid[top, left : right + 1] = border_color
    grid[bottom, left : right + 1] = border_color
    grid[top : bottom + 1, left] = border_color
    grid[top : bottom + 1, right] = border_color

    if gap_side == "top":
        grid[top, left + gap_start : left + gap_start + gap_len] = BG
    elif gap_side == "bottom":
        grid[bottom, left + gap_start : left + gap_start + gap_len] = BG
    elif gap_side == "left":
        grid[top + gap_start : top + gap_start + gap_len, left] = BG
    else:
        grid[top + gap_start : top + gap_start + gap_len, right] = BG

    inner = grid[top + 1 : bottom, left + 1 : right]
    inner[:] = BG
    inner[pattern != BG] = pattern[pattern != BG]


def _sample_directions(count: int, rng: random.Random) -> list[str]:
    sides = [rng.choice(SIDES) for _ in range(count)]
    if count > 1 and len(set(sides)) == 1:
        idx = rng.randrange(count)
        sides[idx] = rng.choice([side for side in SIDES if side != sides[idx]])
    return sides


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    frame_count = rng.randint(FRAME_COUNT_MIN, FRAME_COUNT_MAX)
    min_dim = max(GRID_MIN, 13 + 3 * frame_count)
    max_dim = min(GRID_MAX, 19 + 3 * frame_count)
    if min_dim > max_dim:
        return None

    height = rng.randint(min_dim, max_dim)
    width = rng.randint(min_dim, max_dim)
    grid = np.full((height, width), BG, dtype=int)
    occ = np.zeros((height, width), dtype=bool)

    colors = rng.sample(list(range(1, 10)), 2 * frame_count)
    directions = _sample_directions(frame_count, rng)

    specs: list[dict[str, object]] = []
    for i in range(frame_count):
        spec = _make_frame_spec(colors[2 * i], colors[2 * i + 1], directions[i], rng)
        if spec is None:
            return None
        specs.append(spec)

    specs.sort(key=lambda item: int(item["outer_h"]) * int(item["outer_w"]), reverse=True)
    for spec in specs:
        positions = _candidate_positions(occ, int(spec["outer_h"]), int(spec["outer_w"]))
        if not positions:
            return None
        top, left = rng.choice(positions)
        _paint_frame(grid, top, left, spec)
        _mark_occupied(occ, top, left, int(spec["outer_h"]), int(spec["outer_w"]))

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    if len(_find_frames(grid)) != frame_count:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
