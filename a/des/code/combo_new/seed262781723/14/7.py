from collections import Counter
from dataclasses import dataclass
import random

import numpy as np


BG = 0
FENCE = 5
SEED_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 400


@dataclass(frozen=True)
class FenceRecord:
    r0: int
    c0: int
    r1: int
    c1: int
    gap_r: int
    gap_c: int
    direction: str
    seed_r: int
    seed_c: int
    seed_color: int


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(pts)
    return components


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _detect_fences(grid: np.ndarray, bg: int, fence_color: int) -> list[FenceRecord]:
    fences: list[FenceRecord] = []
    for pts in _components4(grid == fence_color):
        r0, c0, r1, c1 = _bbox(pts)
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "fence must enclose an interior"

        pts_set = set(pts)
        expected_perimeter = set()
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if r in (r0, r1) or c in (c0, c1):
                    expected_perimeter.add((r, c))

        missing = sorted(expected_perimeter - pts_set)
        assert len(missing) == 1, "expected exactly one border gap"
        gap_r, gap_c = missing[0]

        if gap_r == r0 and c0 < gap_c < c1:
            direction = "up"
        elif gap_r == r1 and c0 < gap_c < c1:
            direction = "down"
        elif gap_c == c0 and r0 < gap_r < r1:
            direction = "left"
        elif gap_c == c1 and r0 < gap_r < r1:
            direction = "right"
        else:
            raise AssertionError("gap must lie on a non-corner edge cell")

        assert len(pts_set) == len(expected_perimeter) - 1, "fence must be a rectangle border with one gap"

        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        seed_locs = np.argwhere((interior != bg) & (interior != fence_color))
        assert seed_locs.shape[0] == 1, "each fence must contain exactly one seed pixel"
        sr = int(seed_locs[0, 0]) + r0 + 1
        sc = int(seed_locs[0, 1]) + c0 + 1
        fences.append(
            FenceRecord(
                r0=r0,
                c0=c0,
                r1=r1,
                c1=c1,
                gap_r=gap_r,
                gap_c=gap_c,
                direction=direction,
                seed_r=sr,
                seed_c=sc,
                seed_color=int(grid[sr, sc]),
            )
        )

    fences.sort(key=lambda f: (f.r0, f.c0, f.r1, f.c1))
    return fences


def _band_cells(grid: np.ndarray, fence: FenceRecord, bg: int, fence_color: int) -> tuple[set[tuple[int, int]], str]:
    h, w = grid.shape
    cells: set[tuple[int, int]] = {(fence.gap_r, fence.gap_c)}
    stop_reason = "edge"

    if fence.direction in ("up", "down"):
        cols = slice(fence.c0 + 1, fence.c1)
        rows = range(fence.r0 - 1, -1, -1) if fence.direction == "up" else range(fence.r1 + 1, h)
        for r in rows:
            if np.any(grid[r, cols] == fence_color):
                stop_reason = "fence"
                break
            for c in range(fence.c0 + 1, fence.c1):
                if int(grid[r, c]) == bg:
                    cells.add((r, c))
    else:
        rows = slice(fence.r0 + 1, fence.r1)
        cols = range(fence.c0 - 1, -1, -1) if fence.direction == "left" else range(fence.c1 + 1, w)
        for c in cols:
            if np.any(grid[rows, c] == fence_color):
                stop_reason = "fence"
                break
            for r in range(fence.r0 + 1, fence.r1):
                if int(grid[r, c]) == bg:
                    cells.add((r, c))

    return cells, stop_reason


def _paint_band(out: np.ndarray, grid: np.ndarray, fence: FenceRecord, bg: int, fence_color: int) -> None:
    cells, _ = _band_cells(grid, fence, bg, fence_color)
    for r, c in cells:
        out[r, c] = fence.seed_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each fenced seed color outward through the fence opening as a full-width band.

    Input structure:
    - The background is black.
    - Several gray 1-pixel rectangular fences appear as disconnected components, and each fence has exactly one 1-pixel gap on one side.
    - Each fence interior contains exactly one non-black, non-gray seed pixel, with different fences using different seed colors.

    Rule:
    - Identify the gray fence color as the dominant non-background color, then recover each fence's bounding box and its unique missing border pixel.
    - Use that missing border pixel to determine whether the opening faces up, down, left, or right, and keep the gray border and the seed pixel unchanged.
    - Paint the gap cell and extend the seed color outward in a band aligned with the fence's interior span on that side; stop before the first gray fence pixel in that path or at the grid edge.

    Key insight:
    - The gap is an orientation cue rather than the whole painted width: it tells which side opens, while the actual fill is a full interior-width strip colored by the interior seed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = BG if np.any(g == BG) else _most_common_color(g)
    counts = Counter(map(int, g.flat))
    non_bg = [color for color in counts if color != bg]
    if not non_bg:
        return g.copy()
    fence_color = max(non_bg, key=lambda color: counts[color])

    out = g.copy()
    for fence in _detect_fences(g, bg, fence_color):
        _paint_band(out, g, fence, bg, fence_color)
    return out


def _make_fence(r0: int, c0: int, height: int, width: int, direction: str, seed_color: int, rng: random.Random) -> FenceRecord:
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    if direction == "up":
        gap_r = r0
        gap_c = rng.randint(c0 + 1, c1 - 1)
    elif direction == "down":
        gap_r = r1
        gap_c = rng.randint(c0 + 1, c1 - 1)
    elif direction == "left":
        gap_r = rng.randint(r0 + 1, r1 - 1)
        gap_c = c0
    elif direction == "right":
        gap_r = rng.randint(r0 + 1, r1 - 1)
        gap_c = c1
    else:
        raise AssertionError("unknown direction")

    seed_r = rng.randint(r0 + 1, r1 - 1)
    seed_c = rng.randint(c0 + 1, c1 - 1)
    return FenceRecord(r0, c0, r1, c1, gap_r, gap_c, direction, seed_r, seed_c, seed_color)


def _draw_fence(grid: np.ndarray, fence: FenceRecord) -> None:
    grid[fence.r0, fence.c0 : fence.c1 + 1] = FENCE
    grid[fence.r1, fence.c0 : fence.c1 + 1] = FENCE
    grid[fence.r0 : fence.r1 + 1, fence.c0] = FENCE
    grid[fence.r0 : fence.r1 + 1, fence.c1] = FENCE
    grid[fence.gap_r, fence.gap_c] = BG
    grid[fence.seed_r, fence.seed_c] = fence.seed_color


def _build_vertical_input(rng: random.Random) -> tuple[np.ndarray, list[FenceRecord]]:
    lane_count = rng.randint(2, 3)
    pair_count = rng.randint(0, min(2, lane_count))
    paired_lanes = set(rng.sample(range(lane_count), pair_count))

    if lane_count == 3:
        width_lo, width_hi = 5, 6
        margin_lo, margin_hi = 1, 2
        gap_lo, gap_hi = 2, 3
    else:
        width_lo, width_hi = 5, 7
        margin_lo, margin_hi = 1, 3
        gap_lo, gap_hi = 2, 4

    lane_widths = [rng.randint(width_lo, width_hi) for _ in range(lane_count)]
    left_margin = rng.randint(margin_lo, margin_hi)
    right_margin = rng.randint(margin_lo, margin_hi)
    lane_gaps = [rng.randint(gap_lo, gap_hi) for _ in range(lane_count - 1)]
    base_w = left_margin + right_margin + sum(lane_widths) + sum(lane_gaps)
    if base_w > 30:
        raise RuntimeError("vertical layout too wide")
    grid_w = base_w + rng.randint(0, min(30 - base_w, 3))

    columns: list[int] = []
    cur_c = left_margin
    for idx, width in enumerate(lane_widths):
        columns.append(cur_c)
        cur_c += width
        if idx < lane_count - 1:
            cur_c += lane_gaps[idx]

    lane_specs: list[dict[str, int | str | bool]] = []
    required_h = 0
    for lane_idx in range(lane_count):
        direction = rng.choice(["up", "down"])
        paired = lane_idx in paired_lanes
        source_h = rng.randint(5, 7)
        spec: dict[str, int | str | bool] = {
            "direction": direction,
            "paired": paired,
            "source_h": source_h,
        }
        if direction == "up":
            if paired:
                blocker_h = rng.randint(5, 7)
                blocker_margin = rng.randint(1, 3)
                span = rng.randint(2, 6)
                bottom_margin = rng.randint(1, 3)
                lane_h = blocker_margin + blocker_h + span + source_h + bottom_margin
                spec.update(
                    {
                        "blocker_h": blocker_h,
                        "blocker_margin": blocker_margin,
                        "span": span,
                        "bottom_margin": bottom_margin,
                    }
                )
            else:
                top_open = rng.randint(2, 8)
                bottom_margin = rng.randint(1, 3)
                lane_h = top_open + source_h + bottom_margin
                spec.update({"top_open": top_open, "bottom_margin": bottom_margin})
        else:
            top_margin = rng.randint(1, 3)
            if paired:
                blocker_h = rng.randint(5, 7)
                span = rng.randint(2, 6)
                bottom_open = rng.randint(2, 8)
                lane_h = top_margin + source_h + span + blocker_h + bottom_open
                spec.update(
                    {
                        "blocker_h": blocker_h,
                        "top_margin": top_margin,
                        "span": span,
                        "bottom_open": bottom_open,
                    }
                )
            else:
                bottom_open = rng.randint(2, 8)
                lane_h = top_margin + source_h + bottom_open
                spec.update({"top_margin": top_margin, "bottom_open": bottom_open})
        required_h = max(required_h, lane_h)
        lane_specs.append(spec)

    if required_h > 30:
        raise RuntimeError("vertical layout too tall")
    grid_h = required_h + rng.randint(0, min(30 - required_h, 3))

    colors = rng.sample(SEED_COLORS, lane_count + pair_count)
    color_idx = 0
    fences: list[FenceRecord] = []

    for lane_idx, spec in enumerate(lane_specs):
        lane_w = lane_widths[lane_idx]
        c0 = columns[lane_idx]
        extra = grid_h - (
            (int(spec["blocker_margin"]) + int(spec["blocker_h"]) + int(spec["span"]) + int(spec["source_h"]) + int(spec["bottom_margin"]))
            if spec["direction"] == "up" and spec["paired"]
            else (int(spec["top_open"]) + int(spec["source_h"]) + int(spec["bottom_margin"]))
            if spec["direction"] == "up"
            else (int(spec["top_margin"]) + int(spec["source_h"]) + int(spec["span"]) + int(spec["blocker_h"]) + int(spec["bottom_open"]))
            if spec["paired"]
            else (int(spec["top_margin"]) + int(spec["source_h"]) + int(spec["bottom_open"]))
        )
        shift = rng.randint(0, extra)

        direction = str(spec["direction"])
        source_h = int(spec["source_h"])
        if direction == "up":
            if bool(spec["paired"]):
                blocker_margin = int(spec["blocker_margin"]) + shift
                blocker_h = int(spec["blocker_h"])
                span = int(spec["span"])
                blocker_r0 = blocker_margin
                blocker_r1 = blocker_r0 + blocker_h - 1
                source_r0 = blocker_r1 + span + 1
                source = _make_fence(source_r0, c0, source_h, lane_w, direction, colors[color_idx], rng)
                color_idx += 1
                blocker = _make_fence(blocker_r0, c0, blocker_h, lane_w, direction, colors[color_idx], rng)
                color_idx += 1
                fences.extend([source, blocker])
            else:
                top_open = int(spec["top_open"]) + shift
                source = _make_fence(top_open, c0, source_h, lane_w, direction, colors[color_idx], rng)
                color_idx += 1
                fences.append(source)
        else:
            top_margin = int(spec["top_margin"]) + shift
            source_r0 = top_margin
            if bool(spec["paired"]):
                span = int(spec["span"])
                blocker_h = int(spec["blocker_h"])
                source = _make_fence(source_r0, c0, source_h, lane_w, direction, colors[color_idx], rng)
                color_idx += 1
                blocker_r0 = source.r1 + span + 1
                blocker = _make_fence(blocker_r0, c0, blocker_h, lane_w, direction, colors[color_idx], rng)
                color_idx += 1
                fences.extend([source, blocker])
            else:
                source = _make_fence(source_r0, c0, source_h, lane_w, direction, colors[color_idx], rng)
                color_idx += 1
                fences.append(source)

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for fence in fences:
        _draw_fence(grid, fence)
    return grid, fences


def _build_horizontal_input(rng: random.Random) -> tuple[np.ndarray, list[FenceRecord]]:
    lane_count = rng.randint(2, 3)
    pair_count = rng.randint(0, min(2, lane_count))
    paired_lanes = set(rng.sample(range(lane_count), pair_count))

    if lane_count == 3:
        height_lo, height_hi = 5, 6
        margin_lo, margin_hi = 1, 2
        gap_lo, gap_hi = 2, 3
    else:
        height_lo, height_hi = 5, 7
        margin_lo, margin_hi = 1, 3
        gap_lo, gap_hi = 2, 4

    lane_heights = [rng.randint(height_lo, height_hi) for _ in range(lane_count)]
    top_margin = rng.randint(margin_lo, margin_hi)
    bottom_margin = rng.randint(margin_lo, margin_hi)
    lane_gaps = [rng.randint(gap_lo, gap_hi) for _ in range(lane_count - 1)]
    base_h = top_margin + bottom_margin + sum(lane_heights) + sum(lane_gaps)
    if base_h > 30:
        raise RuntimeError("horizontal layout too tall")
    grid_h = base_h + rng.randint(0, min(30 - base_h, 3))

    rows: list[int] = []
    cur_r = top_margin
    for idx, height in enumerate(lane_heights):
        rows.append(cur_r)
        cur_r += height
        if idx < lane_count - 1:
            cur_r += lane_gaps[idx]

    lane_specs: list[dict[str, int | str | bool]] = []
    required_w = 0
    for lane_idx in range(lane_count):
        direction = rng.choice(["left", "right"])
        paired = lane_idx in paired_lanes
        source_w = rng.randint(5, 7)
        spec: dict[str, int | str | bool] = {
            "direction": direction,
            "paired": paired,
            "source_w": source_w,
        }
        if direction == "left":
            if paired:
                blocker_w = rng.randint(5, 7)
                blocker_margin = rng.randint(1, 3)
                span = rng.randint(2, 6)
                right_margin = rng.randint(1, 3)
                lane_w = blocker_margin + blocker_w + span + source_w + right_margin
                spec.update(
                    {
                        "blocker_w": blocker_w,
                        "blocker_margin": blocker_margin,
                        "span": span,
                        "right_margin": right_margin,
                    }
                )
            else:
                left_open = rng.randint(2, 8)
                right_margin = rng.randint(1, 3)
                lane_w = left_open + source_w + right_margin
                spec.update({"left_open": left_open, "right_margin": right_margin})
        else:
            left_margin = rng.randint(1, 3)
            if paired:
                blocker_w = rng.randint(5, 7)
                span = rng.randint(2, 6)
                right_open = rng.randint(2, 8)
                lane_w = left_margin + source_w + span + blocker_w + right_open
                spec.update(
                    {
                        "blocker_w": blocker_w,
                        "left_margin": left_margin,
                        "span": span,
                        "right_open": right_open,
                    }
                )
            else:
                right_open = rng.randint(2, 8)
                lane_w = left_margin + source_w + right_open
                spec.update({"left_margin": left_margin, "right_open": right_open})
        required_w = max(required_w, lane_w)
        lane_specs.append(spec)

    if required_w > 30:
        raise RuntimeError("horizontal layout too wide")
    grid_w = required_w + rng.randint(0, min(30 - required_w, 3))

    colors = rng.sample(SEED_COLORS, lane_count + pair_count)
    color_idx = 0
    fences: list[FenceRecord] = []

    for lane_idx, spec in enumerate(lane_specs):
        lane_h = lane_heights[lane_idx]
        r0 = rows[lane_idx]
        extra = grid_w - (
            (int(spec["blocker_margin"]) + int(spec["blocker_w"]) + int(spec["span"]) + int(spec["source_w"]) + int(spec["right_margin"]))
            if spec["direction"] == "left" and spec["paired"]
            else (int(spec["left_open"]) + int(spec["source_w"]) + int(spec["right_margin"]))
            if spec["direction"] == "left"
            else (int(spec["left_margin"]) + int(spec["source_w"]) + int(spec["span"]) + int(spec["blocker_w"]) + int(spec["right_open"]))
            if spec["paired"]
            else (int(spec["left_margin"]) + int(spec["source_w"]) + int(spec["right_open"]))
        )
        shift = rng.randint(0, extra)

        direction = str(spec["direction"])
        source_w = int(spec["source_w"])
        if direction == "left":
            if bool(spec["paired"]):
                blocker_margin = int(spec["blocker_margin"]) + shift
                blocker_w = int(spec["blocker_w"])
                span = int(spec["span"])
                blocker_c0 = blocker_margin
                blocker_c1 = blocker_c0 + blocker_w - 1
                source_c0 = blocker_c1 + span + 1
                source = _make_fence(r0, source_c0, lane_h, source_w, direction, colors[color_idx], rng)
                color_idx += 1
                blocker = _make_fence(r0, blocker_c0, lane_h, blocker_w, direction, colors[color_idx], rng)
                color_idx += 1
                fences.extend([source, blocker])
            else:
                left_open = int(spec["left_open"]) + shift
                source = _make_fence(r0, left_open, lane_h, source_w, direction, colors[color_idx], rng)
                color_idx += 1
                fences.append(source)
        else:
            left_margin = int(spec["left_margin"]) + shift
            source_c0 = left_margin
            if bool(spec["paired"]):
                span = int(spec["span"])
                blocker_w = int(spec["blocker_w"])
                source = _make_fence(r0, source_c0, lane_h, source_w, direction, colors[color_idx], rng)
                color_idx += 1
                blocker_c0 = source.c1 + span + 1
                blocker = _make_fence(r0, blocker_c0, lane_h, blocker_w, direction, colors[color_idx], rng)
                color_idx += 1
                fences.extend([source, blocker])
            else:
                source = _make_fence(r0, source_c0, lane_h, source_w, direction, colors[color_idx], rng)
                color_idx += 1
                fences.append(source)

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for fence in fences:
        _draw_fence(grid, fence)
    return grid, fences


def _expected_output(grid: np.ndarray, fences: list[FenceRecord]) -> tuple[np.ndarray, list[str]]:
    out = grid.copy()
    reasons: list[str] = []
    for fence in fences:
        _paint_band(out, grid, fence, BG, FENCE)
        _, reason = _band_cells(grid, fence, BG, FENCE)
        reasons.append(reason)
    return out, reasons


def _validate_generated(grid: np.ndarray, fences: list[FenceRecord]) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    try:
        detected = _detect_fences(grid, BG, FENCE)
    except AssertionError:
        return False

    expected_keys = sorted((f.r0, f.c0, f.r1, f.c1, f.direction, f.seed_color) for f in fences)
    detected_keys = sorted((f.r0, f.c0, f.r1, f.c1, f.direction, f.seed_color) for f in detected)
    if expected_keys != detected_keys:
        return False

    all_band_cells: set[tuple[int, int]] = set()
    has_blocked = False
    has_boundary = False
    for fence in fences:
        band_cells, stop_reason = _band_cells(grid, fence, BG, FENCE)
        if len(band_cells) <= 1:
            return False
        if all_band_cells & band_cells:
            return False
        all_band_cells |= band_cells
        has_blocked |= stop_reason == "fence"
        has_boundary |= stop_reason == "edge"

    if len(fences) >= 3 and not (has_blocked and has_boundary):
        return False

    expected, _ = _expected_output(grid, fences)
    try:
        solved = main(grid)
    except AssertionError:
        return False
    return np.array_equal(expected, solved)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = random.Random()
        builder = _build_vertical_input if rng.random() < 0.5 else _build_horizontal_input
        try:
            grid, fences = builder(rng)
        except RuntimeError:
            continue
        if _validate_generated(grid, fences):
            return grid
    raise RuntimeError("failed to generate a valid fence-gap puzzle input")
