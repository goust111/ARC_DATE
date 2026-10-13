import random
from collections import deque

import numpy as np


GEN_RETRIES = 400
FRAME_RETRIES = 200
PLACEMENT_TRIES = 300
GRID_MIN = 16
GRID_MAX = 30
LEGEND_LEN_MIN = 2
LEGEND_LEN_MAX = 5
FRAME_COUNT_MIN = 2
FRAME_COUNT_MAX = 4


def _background_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(coords)
    return parts


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_legend(grid: np.ndarray, bg: int) -> tuple[set[tuple[int, int]], dict[int, int]]:
    h, w = grid.shape
    candidates: list[list[tuple[int, int]]] = []
    for comp in _components(grid != bg):
        r0, c0, r1, c1 = _bbox(comp)
        if c0 != c1:
            continue
        if c0 not in (0, w - 1):
            continue
        if r0 != 0 and r1 != h - 1:
            continue
        if len(comp) != r1 - r0 + 1:
            continue
        candidates.append(comp)

    assert len(candidates) == 1, "expected exactly one corner legend column"
    legend = sorted(candidates[0])
    mapping = {idx + 1: int(grid[r, c]) for idx, (r, c) in enumerate(legend)}
    return set(legend), mapping


def _holes_in_component(
    grid: np.ndarray, bg: int, coords: list[tuple[int, int]]
) -> list[list[tuple[int, int]]]:
    r0, c0, r1, c1 = _bbox(coords)
    hh, ww = r1 - r0 + 1, c1 - c0 + 1
    if hh < 3 or ww < 3:
        return []

    obj = np.zeros((hh, ww), dtype=bool)
    for r, c in coords:
        obj[r - r0, c - c0] = True

    if not (obj[0, :].all() and obj[-1, :].all() and obj[:, 0].all() and obj[:, -1].all()):
        return []

    bg_mask = grid[r0 : r1 + 1, c0 : c1 + 1] == bg
    holes: list[list[tuple[int, int]]] = []
    for region in _components(bg_mask):
        if any(rr == 0 or rr == hh - 1 or cc == 0 or cc == ww - 1 for rr, cc in region):
            continue
        holes.append([(r0 + rr, c0 + cc) for rr, cc in region])
    return holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each frame's enclosed background holes using the corner legend color
    indexed by that frame's hole count, then erase the legend.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _background_color(g)
    legend_cells, mapping = _find_legend(g, bg)

    out = g.copy()
    for r, c in legend_cells:
        out[r, c] = bg

    frame_count = 0
    for comp in _components(g != bg):
        comp_set = set(comp)
        if comp_set == legend_cells:
            continue
        holes = _holes_in_component(g, bg, comp)
        if not holes:
            continue
        hole_count = len(holes)
        assert hole_count in mapping, "legend must define a color for every hole count"
        fill_color = mapping[hole_count]
        for hole in holes:
            for r, c in hole:
                out[r, c] = fill_color
        frame_count += 1

    assert frame_count > 0, "expected at least one frame with enclosed holes"
    return out


def _make_frame(rng: random.Random, bg: int, color: int, hole_count: int) -> np.ndarray:
    for _ in range(FRAME_RETRIES):
        inner_h_low = max(3, hole_count)
        inner_w_low = max(3, hole_count)
        inner_h = rng.randint(inner_h_low, min(8, inner_h_low + 3))
        inner_w = rng.randint(inner_w_low, min(9, inner_w_low + 4))

        frame = np.full((inner_h + 2, inner_w + 2), bg, dtype=int)
        frame[0, :] = color
        frame[-1, :] = color
        frame[:, 0] = color
        frame[:, -1] = color

        rooms: list[tuple[int, int, int, int]] = [(1, 1, inner_h, inner_w)]
        ok = True
        for _split in range(hole_count - 1):
            options: list[tuple[int, str]] = []
            for idx, (r0, c0, r1, c1) in enumerate(rooms):
                if c1 - c0 + 1 >= 3:
                    options.append((idx, "v"))
                if r1 - r0 + 1 >= 3:
                    options.append((idx, "h"))
            if not options:
                ok = False
                break

            room_idx, axis = rng.choice(options)
            r0, c0, r1, c1 = rooms.pop(room_idx)
            if axis == "v":
                wall_c = rng.randint(c0 + 1, c1 - 1)
                frame[r0 : r1 + 1, wall_c] = color
                rooms.append((r0, c0, r1, wall_c - 1))
                rooms.append((r0, wall_c + 1, r1, c1))
            else:
                wall_r = rng.randint(r0 + 1, r1 - 1)
                frame[wall_r, c0 : c1 + 1] = color
                rooms.append((r0, c0, wall_r - 1, c1))
                rooms.append((wall_r + 1, c0, r1, c1))

        if not ok:
            continue

        frame = np.rot90(frame, rng.randrange(4))
        if rng.random() < 0.5:
            frame = np.fliplr(frame)
        if rng.random() < 0.5:
            frame = np.flipud(frame)
        return frame.copy()

    raise RuntimeError("failed to build frame")


def _legend_cells(
    corner: str, legend_len: int, height: int, width: int
) -> list[tuple[int, int]]:
    if "t" in corner:
        rows = list(range(legend_len))
    else:
        rows = list(range(height - legend_len, height))
    col = 0 if "l" in corner else width - 1
    return [(r, col) for r in rows]


def _expected_output(
    grid: np.ndarray,
    bg: int,
    legend_coords: list[tuple[int, int]],
    legend_colors: list[int],
    placements: list[tuple[int, int, np.ndarray, int]],
) -> np.ndarray:
    out = grid.copy()
    for r, c in legend_coords:
        out[r, c] = bg
    for top, left, frame, hole_count in placements:
        fill_color = legend_colors[hole_count - 1]
        target = out[top : top + frame.shape[0], left : left + frame.shape[1]]
        target[frame == bg] = fill_color
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = rng.randrange(10)
    remaining = [c for c in range(10) if c != bg]
    rng.shuffle(remaining)

    legend_len = rng.randint(LEGEND_LEN_MIN, LEGEND_LEN_MAX)
    frame_count = rng.randint(FRAME_COUNT_MIN, min(FRAME_COUNT_MAX, legend_len + 1))
    if legend_len + frame_count > len(remaining):
        return None

    legend_colors = remaining[:legend_len]
    frame_colors = remaining[legend_len : legend_len + frame_count]

    distinct = rng.sample(range(1, legend_len + 1), k=min(frame_count, legend_len))
    hole_counts = distinct[:]
    while len(hole_counts) < frame_count:
        hole_counts.append(rng.randint(1, legend_len))
    rng.shuffle(hole_counts)

    frames: list[tuple[np.ndarray, int]] = []
    for color, hole_count in zip(frame_colors, hole_counts):
        frames.append((_make_frame(rng, bg, color, hole_count), hole_count))

    padded_area = sum((frame.shape[0] + 2) * (frame.shape[1] + 2) for frame, _ in frames)
    side_hint = int(np.ceil(np.sqrt(padded_area))) + 4
    max_h = max(frame.shape[0] for frame, _ in frames)
    max_w = max(frame.shape[1] for frame, _ in frames)
    min_h = max(GRID_MIN, legend_len + 4, max_h + 3, side_hint - 2)
    min_w = max(GRID_MIN, max_w + 3, side_hint - 2)
    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None

    grid_h = rng.randint(min_h, GRID_MAX)
    grid_w = rng.randint(min_w, GRID_MAX)
    if rng.random() < 0.5:
        grid_h, grid_w = grid_w, grid_h
        if not (legend_len + 4 <= grid_h <= GRID_MAX and max_h + 3 <= grid_h and max_w + 3 <= grid_w):
            return None

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    blocked = np.zeros((grid_h, grid_w), dtype=bool)

    corner = rng.choice(("tl", "tr", "bl", "br"))
    legend_coords = _legend_cells(corner, legend_len, grid_h, grid_w)
    for idx, (r, c) in enumerate(legend_coords):
        grid[r, c] = legend_colors[idx]
    lr0, lc0 = legend_coords[0]
    lr1, lc1 = legend_coords[-1]
    blocked[max(0, lr0 - 1) : min(grid_h, lr1 + 2), max(0, lc0 - 1) : min(grid_w, lc1 + 2)] = True

    placements: list[tuple[int, int, np.ndarray, int]] = []
    for frame, hole_count in sorted(frames, key=lambda item: item[0].shape[0] * item[0].shape[1], reverse=True):
        fh, fw = frame.shape
        if grid_h - fh - 1 < 1 or grid_w - fw - 1 < 1:
            return None
        placed = False
        for _ in range(PLACEMENT_TRIES):
            top = rng.randint(1, grid_h - fh - 1)
            left = rng.randint(1, grid_w - fw - 1)
            if blocked[top - 1 : top + fh + 1, left - 1 : left + fw + 1].any():
                continue
            patch = grid[top : top + fh, left : left + fw]
            patch[frame != bg] = frame[frame != bg]
            blocked[top - 1 : top + fh + 1, left - 1 : left + fw + 1] = True
            placements.append((top, left, frame, hole_count))
            placed = True
            break
        if not placed:
            return None

    expected = _expected_output(grid, bg, legend_coords, legend_colors, placements)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
