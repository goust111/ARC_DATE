from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
BAR_COLOR_POOL = (1, 2, 6, 7, 8, 9)
MARKER = 3
FILL = 4
FRAME = 5

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
LENGTH_FAMILIES = (
    (2, 4),
    (3, 6),
    (2, 3, 5),
    (2, 4, 5),
    (2, 3, 6),
    (2, 4, 6),
    (3, 5, 6),
)
MAX_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _sign(n: int) -> int:
    if n > 0:
        return 1
    if n < 0:
        return -1
    return 0


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _find_frames(grid: np.ndarray) -> list[dict]:
    markers = [tuple(map(int, rc)) for rc in np.argwhere(grid == MARKER)]
    assert len(markers) == 2, "expected exactly two green frame markers"

    frames: list[dict] = []
    for r, c in markers:
        gap_dir: tuple[int, int] | None = None
        frame_neighbors: list[int] = []
        for dr, dc in DIRS4:
            rr, cc = r + dr, c + dc
            assert _in_bounds(grid.shape, rr, cc), "marker must not touch border"
            val = int(grid[rr, cc])
            if val == BG:
                assert gap_dir is None, "marker must have exactly one doorway gap"
                gap_dir = (dr, dc)
            else:
                frame_neighbors.append(val)

        assert gap_dir is not None
        assert len(frame_neighbors) == 3
        assert len(set(frame_neighbors)) == 1
        frame_color = frame_neighbors[0]

        gap = (r + gap_dir[0], c + gap_dir[1])
        exterior = (r + 2 * gap_dir[0], c + 2 * gap_dir[1])
        assert _in_bounds(grid.shape, *exterior), "exterior doorway cell must be in bounds"
        assert int(grid[exterior]) == BG, "exterior doorway cell must be background"

        r0, r1 = r - 1, r + 1
        c0, c1 = c - 1, c + 1
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                if (rr, cc) == (r, c):
                    assert int(grid[rr, cc]) == MARKER
                elif (rr, cc) == gap:
                    assert int(grid[rr, cc]) == BG
                else:
                    assert int(grid[rr, cc]) == frame_color

        frames.append(
            {
                "center": (r, c),
                "gap": gap,
                "exterior": exterior,
                "frame_color": frame_color,
                "bbox": (r0, c0, r1, c1),
            }
        )

    assert len({frame["frame_color"] for frame in frames}) == 1
    frames.sort(key=lambda frame: frame["exterior"])
    return frames


def _find_bars(grid: np.ndarray, frames: list[dict]) -> list[dict]:
    ignore = np.zeros_like(grid, dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        ignore[r0 : r1 + 1, c0 : c1 + 1] = True

    bars: list[dict] = []
    h, w = grid.shape
    for r in range(h):
        c = 0
        while c < w:
            if ignore[r, c] or int(grid[r, c]) == BG:
                c += 1
                continue
            color = int(grid[r, c])
            c0 = c
            while c < w and (not ignore[r, c]) and int(grid[r, c]) == color:
                c += 1
            c1 = c - 1
            if c1 > c0:
                bars.append({"row": r, "c0": c0, "c1": c1, "length": c1 - c0 + 1, "color": color})

    assert len(bars) >= 2, "expected 2-3 color bars"
    return bars


def _monotone_union(grid: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    row_step = _sign(end[0] - start[0])
    col_step = _sign(end[1] - start[1])
    assert row_step != 0 and col_step != 0, "endpoints must differ in both row and column"

    moves = []
    if row_step:
        moves.append((row_step, 0))
    if col_step:
        moves.append((0, col_step))

    r0, r1 = sorted((start[0], end[0]))
    c0, c1 = sorted((start[1], end[1]))

    def reachable(seed: tuple[int, int], step_list: list[tuple[int, int]]) -> np.ndarray:
        seen = np.zeros_like(g, dtype=bool)
        q: deque[tuple[int, int]] = deque([seed])
        seen[seed] = True
        while q:
            r, c = q.popleft()
            for dr, dc in step_list:
                rr, cc = r + dr, c + dc
                if (
                    r0 <= rr <= r1
                    and c0 <= cc <= c1
                    and not seen[rr, cc]
                    and int(g[rr, cc]) == BG
                ):
                    seen[rr, cc] = True
                    q.append((rr, cc))
        return seen

    forward = reachable(start, moves)
    backward = reachable(end, [(-dr, -dc) for dr, dc in moves])
    path = forward & backward
    assert path[start] and path[end]
    return path


def _distance_on_mask(mask: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    dist = np.full(mask.shape, -1, dtype=int)
    q: deque[tuple[int, int]] = deque([start])
    dist[start] = 0
    while q:
        r, c = q.popleft()
        nd = int(dist[r, c]) + 1
        for rr, cc in _neighbors4(r, c):
            if _in_bounds(mask.shape, rr, cc) and mask[rr, cc] and dist[rr, cc] < 0:
                dist[rr, cc] = nd
                q.append((rr, cc))
    return dist


def _analyze(grid: np.ndarray) -> dict:
    frames = _find_frames(grid)
    bars = _find_bars(grid, frames)
    start = frames[0]["exterior"]
    end = frames[1]["exterior"]
    base_path = _monotone_union(grid, start, end)
    channel = base_path.copy()
    for frame in frames:
        channel[frame["gap"]] = True
    dist = _distance_on_mask(channel, start)
    assert np.all(dist[channel] >= 0)
    return {
        "frames": frames,
        "bars": bars,
        "start": start,
        "end": end,
        "base_path": base_path,
        "channel": channel,
        "dist": dist,
    }


def _paint_periodic(
    base: np.ndarray, bars: list[dict], channel: np.ndarray, dist: np.ndarray
) -> np.ndarray:
    out = base.copy()
    painted = np.zeros_like(base, dtype=bool)
    for bar in bars:
        mask = channel & (dist >= 0) & (dist % bar["length"] == 0) & ~painted
        out[mask] = bar["color"]
        painted |= mask
    out[channel & ~painted] = FILL
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project periodic color bars onto the union of all shortest monotone doorway paths.

    Input structure:
    - The grid has black background, two grey 3x3 doorway frames, and one green center marker in each frame.
    - Each marker has exactly one black 4-neighbor gap; the next cell outward is that frame's exterior doorway cell.
    - Two or three horizontal colored bars sit near one edge and their lengths are all distinct.

    Rule:
    - Detect the two exterior doorway cells and order them by row-major position; the earlier one is A and the other is B.
    - Take the union of all shortest monotone 4-neighbor paths from A to B that stay on black cells and only step toward B in row or column.
    - Add both doorway-gap cells to that channel, compute each channel cell's shortest 4-neighbor distance from A, then process the bars from shorter length to longer length.
    - For a bar of color C and length L, paint every still-uncolored channel cell with distance d satisfying d mod L == 0; paint any remaining channel cells yellow.

    Key insight:
    - The path region is the full branching union of monotone shortest routes, and the bar length itself is the visible period used to color anti-diagonal distance layers from the row-major first doorway.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze(g)
    bars = sorted(info["bars"], key=lambda bar: (bar["length"], bar["row"], bar["c0"]))
    return _paint_periodic(g, bars, info["channel"], info["dist"])


def _sample_bars(
    rng: np.random.Generator, h: int, w: int
) -> tuple[np.ndarray, list[dict], str, int]:
    bar_edge = str(rng.choice(["top", "bottom"]))
    if bar_edge == "top":
        bar_row = int(rng.integers(1, 3))
    else:
        bar_row = int(h - rng.integers(2, 4))

    families = [
        family
        for family in LENGTH_FAMILIES
        if sum(family) + (len(family) - 1) + 2 <= w
    ]
    assert families
    lengths = list(map(int, families[int(rng.integers(0, len(families)))]))
    rng.shuffle(lengths)

    colors = list(map(int, rng.choice(BAR_COLOR_POOL, size=len(lengths), replace=False)))

    gap_count = len(lengths) + 1
    extra = w - (sum(lengths) + (len(lengths) - 1) + 2)
    slots = np.zeros(gap_count, dtype=int)
    for _ in range(extra):
        slots[int(rng.integers(0, gap_count))] += 1

    base = np.zeros((h, w), dtype=int)
    cursor = 1 + int(slots[0])
    bars: list[dict] = []
    for idx, (length, color) in enumerate(zip(lengths, colors)):
        c0 = cursor
        c1 = c0 + length - 1
        base[bar_row, c0 : c1 + 1] = color
        bars.append({"row": bar_row, "c0": c0, "c1": c1, "length": length, "color": color})
        cursor = c1 + 2 + int(slots[idx + 1])

    return base, bars, bar_edge, bar_row


def _center_ranges(
    h: int, w: int, bar_edge: str, bar_row: int
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
    top_min = max(3, bar_row + 3) if bar_edge == "top" else 3
    top_max = h // 2 - 2
    bot_min = h // 2 + 1
    bot_max = min(h - 3, bar_row - 3) if bar_edge == "bottom" else h - 3
    left_min = 3
    left_max = w // 2 - 2
    right_min = w // 2 + 1
    right_max = w - 4
    return (top_min, top_max), (bot_min, bot_max), (left_min, left_max), (right_min, right_max)


def _valid_center_ranges(ranges: tuple[tuple[int, int], ...]) -> bool:
    return all(lo <= hi for lo, hi in ranges)


def _sample_centers(
    rng: np.random.Generator, h: int, w: int, bar_edge: str, bar_row: int
) -> list[tuple[int, int]]:
    top_rows, bot_rows, left_cols, right_cols = _center_ranges(h, w, bar_edge, bar_row)
    assert _valid_center_ranges((top_rows, bot_rows, left_cols, right_cols))

    if bool(rng.integers(0, 2)):
        return [
            (int(rng.integers(top_rows[0], top_rows[1] + 1)), int(rng.integers(left_cols[0], left_cols[1] + 1))),
            (int(rng.integers(bot_rows[0], bot_rows[1] + 1)), int(rng.integers(right_cols[0], right_cols[1] + 1))),
        ]

    return [
        (int(rng.integers(top_rows[0], top_rows[1] + 1)), int(rng.integers(right_cols[0], right_cols[1] + 1))),
        (int(rng.integers(bot_rows[0], bot_rows[1] + 1)), int(rng.integers(left_cols[0], left_cols[1] + 1))),
    ]


def _frame_cells(center: tuple[int, int], gap_dir: tuple[int, int]) -> dict:
    r, c = center
    gap = (r + gap_dir[0], c + gap_dir[1])
    exterior = (r + 2 * gap_dir[0], c + 2 * gap_dir[1])
    pixels = []
    for rr in range(r - 1, r + 2):
        for cc in range(c - 1, c + 2):
            if (rr, cc) not in (center, gap):
                pixels.append((rr, cc))
    return {"center": center, "gap": gap, "exterior": exterior, "pixels": pixels}


def _place_frames(
    rng: np.random.Generator,
    grid: np.ndarray,
    centers: list[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    placements = []
    for center in centers:
        choices = []
        for gap_dir in DIRS4:
            frame = _frame_cells(center, gap_dir)
            if not _in_bounds((h, w), *frame["gap"]) or not _in_bounds((h, w), *frame["exterior"]):
                continue
            if any(int(grid[rr, cc]) != BG for rr, cc in frame["pixels"]):
                continue
            if int(grid[frame["gap"]]) != BG or int(grid[frame["exterior"]]) != BG:
                continue
            choices.append(frame)
        if not choices:
            return False
        placements.append(choices[int(rng.integers(0, len(choices)))])

    ext0, ext1 = placements[0]["exterior"], placements[1]["exterior"]
    if abs(ext0[0] - ext1[0]) < 3 or abs(ext0[1] - ext1[1]) < 3:
        return False

    for frame in placements:
        for rr, cc in frame["pixels"]:
            grid[rr, cc] = FRAME
        cr, cc = frame["center"]
        grid[cr, cc] = MARKER
        gr, gc = frame["gap"]
        grid[gr, gc] = BG
    return True


def _validate_instance(grid: np.ndarray) -> bool:
    try:
        info = _analyze(grid)
        bars = sorted(info["bars"], key=lambda bar: (bar["length"], bar["row"], bar["c0"]))
        lengths = [bar["length"] for bar in bars]
        if len(lengths) != len(set(lengths)):
            return False
        if any(
            not np.all(grid[bar["row"], bar["c0"] : bar["c1"] + 1] == bar["color"])
            for bar in bars
        ):
            return False

        channel = info["channel"]
        dist = info["dist"]
        if not (
            np.any(channel)
            and np.sum(channel)
            > abs(info["end"][0] - info["start"][0]) + abs(info["end"][1] - info["start"][1]) + 2
        ):
            return False

        overlap = False
        for rr, cc in np.argwhere(channel):
            d = int(dist[rr, cc])
            if d <= 0:
                continue
            hits = sum(d % length == 0 for length in lengths)
            if hits >= 2:
                overlap = True
                break
        if not overlap:
            return False

        painted = np.zeros_like(grid, dtype=bool)
        for bar in bars:
            hits = channel & (dist >= 0) & (dist % bar["length"] == 0) & ~painted
            if not np.any(hits):
                return False
            painted |= hits

        if not np.any(channel & ~painted):
            return False

        out = main(grid)
        alt_dist = _distance_on_mask(channel, info["end"])
        alt_out = _paint_periodic(grid, bars, channel, alt_dist)
        if np.array_equal(out, grid):
            return False
        if np.array_equal(out, alt_out):
            return False
        if np.max(grid.shape) > 30 or np.max(out.shape) > 30:
            return False
        return True
    except Exception:
        return False


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        rng = np.random.default_rng()
        h = int(rng.integers(18, 25))
        w = int(rng.integers(18, 25))

        grid, _bars, bar_edge, bar_row = _sample_bars(rng, h, w)
        center_ranges = _center_ranges(h, w, bar_edge, bar_row)
        if not _valid_center_ranges(center_ranges):
            continue

        centers = _sample_centers(rng, h, w, bar_edge, bar_row)
        if not _place_frames(rng, grid, centers):
            continue

        if _validate_instance(grid):
            return grid

    raise RuntimeError("failed to generate a valid monotone periodic-path ARC input")
