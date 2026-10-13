import random
from collections import Counter

import numpy as np


GEN_RETRIES = 400
PLACEMENT_RETRIES = 300
GRID_HEIGHT_RANGE = (14, 22)
GRID_WIDTH_RANGE = (14, 24)
LEGEND_LENGTH_RANGE = (2, 6)
FRAME_COUNT_RANGE = (1, 4)
FRAME_HEIGHT_RANGE = (4, 8)
FRAME_WIDTH_RANGE = (4, 9)
ALL_COLORS = tuple(range(10))
BACKGROUND = 0


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []

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
            comps.append(pts)

    return comps


def _legend_info(grid: np.ndarray, bg: int) -> tuple[int, int, list[int]]:
    bottom = grid[-1]
    cols = [int(c) for c in np.flatnonzero(bottom != bg)]
    assert cols, "expected a bottom-row legend"

    runs: list[tuple[int, int]] = []
    start = cols[0]
    prev = cols[0]
    for col in cols[1:]:
        if col == prev + 1:
            prev = col
            continue
        runs.append((start, prev))
        start = col
        prev = col
    runs.append((start, prev))

    run_start, run_end = max(runs, key=lambda item: (item[1] - item[0] + 1, -item[0]))
    colors = [int(v) for v in bottom[run_start : run_end + 1]]
    assert len(colors) >= 1
    return run_start, run_end, colors


def _is_frame_component(points: list[tuple[int, int]]) -> tuple[bool, tuple[int, int, int, int] | None]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return False, None

    expected_size = 2 * (height + width) - 4
    if len(points) != expected_size:
        return False, None

    point_set = set(points)
    expected = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))

    if point_set != expected:
        return False, None
    return True, (r0, c0, r1, c1)


def _find_frames(grid: np.ndarray, bg: int) -> list[tuple[int, int, int, int]]:
    work = np.array(grid, copy=True)
    work[-1, :] = bg
    frames: list[tuple[int, int, int, int]] = []

    for color in np.unique(work):
        color = int(color)
        if color == bg:
            continue
        for pts in _components(work == color):
            ok, bbox = _is_frame_component(pts)
            if ok and bbox is not None:
                frames.append(bbox)

    frames.sort()
    return frames


def _paint_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = BACKGROUND
    _, _, legend = _legend_info(g, bg)
    frames = _find_frames(g, bg)
    assert frames, "expected at least one rectangular frame"

    out = g.copy()
    for bbox in frames:
        r0, c0, r1, c1 = bbox
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        marker_count = int(np.count_nonzero(interior != bg))
        assert 1 <= marker_count <= len(legend), "marker count must map into the legend"
        _paint_frame(out, bbox, legend[marker_count - 1])

    return out


def _sample_marker_positions(
    rng: random.Random, r0: int, c0: int, r1: int, c1: int, count: int
) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    assert len(cells) >= count

    for _ in range(50):
        order = list(cells)
        rng.shuffle(order)
        chosen: list[tuple[int, int]] = []
        for cell in order:
            if all(abs(cell[0] - r) + abs(cell[1] - c) > 1 for r, c in chosen):
                chosen.append(cell)
                if len(chosen) == count:
                    return chosen

    return rng.sample(cells, count)


def _frame_sizes_for_count(grid_h: int, grid_w: int, count: int) -> list[tuple[int, int]]:
    sizes = []
    for height in range(FRAME_HEIGHT_RANGE[0], FRAME_HEIGHT_RANGE[1] + 1):
        for width in range(FRAME_WIDTH_RANGE[0], FRAME_WIDTH_RANGE[1] + 1):
            if height > grid_h - 3 or width > grid_w - 2:
                continue
            if (height - 2) * (width - 2) < count:
                continue
            sizes.append((height, width))
    return sizes


def _count_choices(rng: random.Random, frame_count: int, legend_len: int) -> list[int]:
    if frame_count == 1:
        return [rng.randint(1, legend_len)]

    target_distinct = min(frame_count, legend_len, 2)
    for _ in range(40):
        counts = [rng.randint(1, legend_len) for _ in range(frame_count)]
        if len(set(counts)) >= target_distinct:
            return counts
    return [rng.randint(1, legend_len) for _ in range(frame_count)]


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        bg = BACKGROUND
        start, end, legend = _legend_info(grid, bg)
        if len(legend) < 2:
            return False
        if len(set(legend)) != len(legend):
            return False
        if np.any(grid[-1, :start] != bg) or np.any(grid[-1, end + 1 :] != bg):
            return False

        frames = _find_frames(grid, bg)
        if not frames:
            return False

        out = main(grid)
        if not np.array_equal(out[-1], grid[-1]):
            return False

        for bbox in frames:
            r0, c0, r1, c1 = bbox
            interior = grid[r0 + 1 : r1, c0 + 1 : c1]
            marker_count = int(np.count_nonzero(interior != bg))
            if not (1 <= marker_count <= len(legend)):
                return False
            expected = legend[marker_count - 1]
            border = np.zeros_like(grid, dtype=bool)
            border[r0, c0 : c1 + 1] = True
            border[r1, c0 : c1 + 1] = True
            border[r0 : r1 + 1, c0] = True
            border[r0 : r1 + 1, c1] = True
            if not np.all(out[border] == expected):
                return False
            if not np.array_equal(out[r0 + 1 : r1, c0 + 1 : c1], interior):
                return False

        return True
    except Exception:
        return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(*GRID_HEIGHT_RANGE)
    width = rng.randint(*GRID_WIDTH_RANGE)
    legend_len = rng.randint(*LEGEND_LENGTH_RANGE)
    frame_count = rng.randint(*FRAME_COUNT_RANGE)

    grid = np.full((height, width), BACKGROUND, dtype=int)
    occupied = np.zeros_like(grid, dtype=bool)

    frame_color = rng.choice([c for c in ALL_COLORS if c != BACKGROUND])
    legend_colors = rng.sample([c for c in ALL_COLORS if c not in (BACKGROUND, frame_color)], legend_len)
    legend_start = rng.randint(0, width - legend_len)
    grid[-1, legend_start : legend_start + legend_len] = np.array(legend_colors, dtype=int)

    counts = _count_choices(rng, frame_count, legend_len)
    rng.shuffle(counts)

    frames: list[tuple[int, int, int, int, int]] = []
    for marker_count in counts:
        sizes = _frame_sizes_for_count(height, width, marker_count)
        if not sizes:
            return None

        placed = False
        for _ in range(PLACEMENT_RETRIES):
            frame_h, frame_w = rng.choice(sizes)
            r0 = rng.randint(1, height - frame_h - 2)
            c0 = rng.randint(1, width - frame_w - 1)
            r1 = r0 + frame_h - 1
            c1 = c0 + frame_w - 1

            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(height, r1 + 2)
            cc1 = min(width, c1 + 2)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue

            occupied[rr0:rr1, cc0:cc1] = True
            _paint_frame(grid, (r0, c0, r1, c1), frame_color)

            marker_positions = _sample_marker_positions(rng, r0, c0, r1, c1, marker_count)
            marker_colors = rng.sample([c for c in ALL_COLORS if c not in (BACKGROUND, frame_color)], marker_count)
            for (mr, mc), color in zip(marker_positions, marker_colors):
                grid[mr, mc] = color

            frames.append((r0, c0, r1, c1, marker_count))
            placed = True
            break

        if not placed:
            return None

    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        try:
            grid = _gen_attempt(rng)
        except Exception:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
