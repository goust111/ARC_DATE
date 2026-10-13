import random
from typing import Iterable

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8)
GEN_RETRIES = 400


def _neighbors4(r: int, c: int, h: int, w: int) -> Iterable[tuple[int, int]]:
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _line_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    (r0, c0), (r1, c1) = a, b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    raise ValueError("segments must be axis-aligned")


def _find_legend(grid: np.ndarray) -> tuple[list[int], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[list[int], np.ndarray]] = []
    for col in (0, w - 1):
        start = 0
        while start < h:
            color = int(g[start, col])
            if color in (BLACK, GREY, MAROON):
                start += 1
                continue
            end = start
            while end + 1 < h and int(g[end + 1, col]) not in (BLACK, GREY, MAROON):
                end += 1
            colors = [int(g[r, col]) for r in range(start, end + 1)]
            if len(colors) >= 3 and len(set(colors)) == len(colors):
                mask = np.zeros_like(g, dtype=bool)
                mask[start : end + 1, col] = True
                candidates.append((colors, mask))
            start = end + 1
    assert candidates, "expected a vertical priority legend on a grid edge"
    return max(candidates, key=lambda item: len(item[0]))


def _find_starts(grid: np.ndarray, legend_mask: np.ndarray) -> dict[int, tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    starts: dict[int, tuple[int, int]] = {}
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            if legend_mask[r, c]:
                continue
            color = int(g[r, c])
            if color in (BLACK, GREY, MAROON):
                continue
            grey_neighbors = [
                (nr, nc)
                for nr, nc in _neighbors4(r, c, h, w)
                if int(g[nr, nc]) == GREY
            ]
            assert len(grey_neighbors) == 1, "each start must touch one grey path cell"
            starts[color] = (r, c)
    assert starts, "expected edge start markers"
    return starts


def _find_frames(grid: np.ndarray) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    centers: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(g[r, c]) != BLACK:
                continue
            window = g[r - 1 : r + 2, c - 1 : c + 2]
            if int(window[1, 1]) != BLACK:
                continue
            border = [
                int(window[0, 0]),
                int(window[0, 1]),
                int(window[0, 2]),
                int(window[1, 0]),
                int(window[1, 2]),
                int(window[2, 0]),
                int(window[2, 1]),
                int(window[2, 2]),
            ]
            if all(v == MAROON for v in border):
                centers.append((r, c))
    assert centers, "expected terminal frames"
    return centers


def _frame_endpoints(grid: np.ndarray, centers: list[tuple[int, int]]) -> dict[tuple[int, int], tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    mapping: dict[tuple[int, int], tuple[int, int]] = {}
    for r, c in centers:
        candidates = []
        for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == GREY:
                candidates.append((rr, cc))
        assert len(candidates) == 1, "each frame must have one adjacent grey endpoint"
        mapping[candidates[0]] = (r, c)
    return mapping


def _trace_path(
    grid: np.ndarray,
    start: tuple[int, int],
    *,
    stop_on_branch: bool,
) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    gray_neighbors = [
        (nr, nc)
        for nr, nc in _neighbors4(start[0], start[1], h, w)
        if int(g[nr, nc]) == GREY
    ]
    assert len(gray_neighbors) == 1
    prev = start
    cur = gray_neighbors[0]
    path = [cur]
    seen = {cur}
    while True:
        candidates = [
            (nr, nc)
            for nr, nc in _neighbors4(cur[0], cur[1], h, w)
            if int(g[nr, nc]) == GREY and (nr, nc) != prev
        ]
        if stop_on_branch and len(candidates) != 1:
            break
        if not candidates:
            break
        if len(candidates) == 1:
            nxt = candidates[0]
        else:
            in_dir = (cur[0] - prev[0], cur[1] - prev[1])
            straight = [
                cell
                for cell in candidates
                if cell[0] - cur[0] == in_dir[0] and cell[1] - cur[1] == in_dir[1]
            ]
            assert straight, "ambiguous branch without a straight continuation"
            nxt = straight[0]
        assert nxt not in seen, "paths must not loop"
        seen.add(nxt)
        path.append(nxt)
        prev, cur = cur, nxt
    return path


def _first_frame_after(
    main_path: list[tuple[int, int]],
    endpoint_to_frame: dict[tuple[int, int], tuple[int, int]],
    start_index: int,
    used: set[tuple[int, int]],
) -> tuple[int, int]:
    for cell in main_path[start_index:]:
        if cell in endpoint_to_frame and endpoint_to_frame[cell] not in used:
            return endpoint_to_frame[cell]
    raise AssertionError("expected a later unused trunk frame")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route each edge-start color through a shared grey network and fill the terminal frame reached after priority-controlled merges.

    Input structure:
    - The background is black.
    - Several single-pixel colored starts lie on distinct grid edges and each touches a grey orthogonal path.
    - A vertical edge legend lists the participating colors from top to bottom in priority order.
    - Maroon 3x3 hollow frames sit beside grey path endpoints; the shared horizontal trunk may be fed by top/bottom side paths.

    Rule:
    - Read the legend so earlier colors have higher priority than later colors.
    - The left-edge start is the main trunk signal: when a feeder path meets that trunk, the higher-priority trunk keeps going straight and the lower-priority feeder turns onto the forward branch of the trunk.
    - The main signal fills the first trunk frame, and each feeder fills the first still-empty trunk frame that appears after its merge point; any separate non-intersecting path fills its own unique frame.
    - Leave the grey network, the legend, the starts, and the maroon frame borders unchanged.

    Key insight:
    - The legend is not a paint palette; it identifies which signal owns the shared trunk at each merge, and the frame order along that trunk determines where later merged colors finally terminate.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    legend_colors, legend_mask = _find_legend(g)
    priority = {color: idx for idx, color in enumerate(legend_colors)}
    starts = _find_starts(g, legend_mask)
    frames = _find_frames(g)
    endpoint_to_frame = _frame_endpoints(g, frames)

    left_starts = [(color, pos) for color, pos in starts.items() if pos[1] == 0]
    top_bottom_starts = [
        (color, pos)
        for color, pos in starts.items()
        if pos[1] != 0 and pos[1] != g.shape[1] - 1
    ]
    right_starts = [(color, pos) for color, pos in starts.items() if pos[1] == g.shape[1] - 1]

    assert len(left_starts) == 1, "expected one left-edge main start"
    assert len(top_bottom_starts) == 2, "expected two feeder starts on top/bottom edges"
    assert len(right_starts) <= 1, "expected at most one right-edge independent start"

    main_color, main_start = left_starts[0]
    assert all(priority[main_color] < priority[color] for color, _pos in top_bottom_starts), (
        "the trunk owner must outrank the feeders in the legend"
    )

    main_path = _trace_path(g, main_start, stop_on_branch=False)
    main_index = {cell: idx for idx, cell in enumerate(main_path)}
    trunk_frames = [
        (idx, endpoint_to_frame[cell])
        for idx, cell in enumerate(main_path)
        if cell in endpoint_to_frame
    ]
    assert len(trunk_frames) == 1 + len(top_bottom_starts), "expected one trunk frame per trunk-using color"

    feeder_info = []
    for color, pos in top_bottom_starts:
        feeder_path = _trace_path(g, pos, stop_on_branch=True)
        junction = feeder_path[-1]
        assert junction in main_index, "feeder must merge into the main trunk"
        feeder_info.append((main_index[junction], color))
    feeder_info.sort()

    out = g.copy()
    used_frames: set[tuple[int, int]] = set()

    main_frame = trunk_frames[0][1]
    out[main_frame] = main_color
    used_frames.add(main_frame)

    for junction_index, color in feeder_info:
        frame = _first_frame_after(main_path, endpoint_to_frame, junction_index, used_frames)
        out[frame] = color
        used_frames.add(frame)

    for color, pos in right_starts:
        path = _trace_path(g, pos, stop_on_branch=False)
        assert path[-1] in endpoint_to_frame, "independent right path must end at a frame"
        frame = endpoint_to_frame[path[-1]]
        out[frame] = color
        used_frames.add(frame)

    return out


def _draw_frame(grid: np.ndarray, center: tuple[int, int]) -> None:
    r, c = center
    grid[r - 1 : r + 2, c - 1] = MAROON
    grid[r - 1 : r + 2, c + 1] = MAROON
    grid[r - 1, c - 1 : c + 2] = MAROON
    grid[r + 1, c - 1 : c + 2] = MAROON


def _place_trunk_frame(
    grid: np.ndarray,
    row: int,
    col: int,
    side: str,
) -> tuple[int, int]:
    if side == "up":
        center = (row - 2, col)
    else:
        center = (row + 2, col)
    _draw_frame(grid, center)
    return center


def _spaced_positions(low: int, high: int, count: int, min_gap: int) -> list[int] | None:
    values = list(range(low, high + 1))
    if len(values) < count:
        return None
    for _ in range(200):
        picks = sorted(random.sample(values, count))
        if all(b - a >= min_gap for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _priority_order(main_color: int, feeder_colors: list[int], extra: list[int]) -> list[int]:
    colors = [main_color] + feeder_colors + extra
    for _ in range(200):
        perm = random.sample(colors, len(colors))
        if all(perm.index(main_color) < perm.index(color) for color in feeder_colors):
            return perm
    raise RuntimeError("failed to sample a compatible priority order")


def _reserve_rows(
    height: int,
    legend_rows: list[int],
    trunk_row: int,
    frame_sides: list[str],
) -> set[int]:
    reserved = set(legend_rows)
    reserved.add(trunk_row)
    for side in frame_sides:
        if side == "up":
            reserved.update({trunk_row - 3, trunk_row - 2, trunk_row - 1})
        else:
            reserved.update({trunk_row + 1, trunk_row + 2, trunk_row + 3})
    return {r for r in reserved if 0 <= r < height}


def _expected_output(
    inp: np.ndarray,
    main_color: int,
    feeder_assignments: list[tuple[int, int]],
    trunk_frame_centers: list[tuple[int, int]],
    right_assignment: tuple[int, tuple[int, int]] | None,
) -> np.ndarray:
    out = inp.copy()
    out[trunk_frame_centers[0]] = main_color
    for (_, color), frame in zip(feeder_assignments, trunk_frame_centers[1:]):
        out[frame] = color
    if right_assignment is not None:
        color, frame = right_assignment
        out[frame] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    count = random.choice((3, 4))
    width = random.randint(24, 30)
    height = random.randint(15, 24)

    frame_sides = [random.choice(("up", "down")) for _ in range(3)]
    if "up" in frame_sides:
        min_trunk = 4
    else:
        min_trunk = 2
    if "down" in frame_sides:
        max_trunk = height - 5
    else:
        max_trunk = height - 3
    if min_trunk > max_trunk:
        return None
    trunk_row = random.randint(min_trunk, max_trunk)

    legend_at_top = random.choice((False, True))
    legend_rows = list(range(0, count)) if legend_at_top else list(range(height - count, height))

    cols = _spaced_positions(5, width - 6, 5, 3)
    if cols is None:
        return None
    frame_cols = [cols[0], cols[2], cols[4]]
    merge_cols = [cols[1], cols[3]]

    all_colors = random.sample(COLOR_POOL, count)
    main_color = all_colors[0]
    feeder_colors = [all_colors[1], all_colors[2]]
    extra_colors = [all_colors[3]] if count == 4 else []
    legend_order = _priority_order(main_color, feeder_colors, extra_colors)

    grid = np.full((height, width), BLACK, dtype=int)

    legend_col = width - 1
    for row, color in zip(legend_rows, legend_order):
        grid[row, legend_col] = color

    trunk_end = frame_cols[-1]
    for r, c in _line_cells((trunk_row, 1), (trunk_row, trunk_end)):
        grid[r, c] = GREY
    grid[trunk_row, 0] = main_color

    feeder_specs = list(zip(merge_cols, feeder_colors, ("top", "bottom")))
    if random.random() < 0.5:
        feeder_specs[0] = (merge_cols[0], feeder_colors[0], "bottom")
        feeder_specs[1] = (merge_cols[1], feeder_colors[1], "top")
    random.shuffle(feeder_specs)

    for col, color, side in feeder_specs:
        if side == "top":
            for r, c in _line_cells((1, col), (trunk_row, col)):
                grid[r, c] = GREY
            grid[0, col] = color
        else:
            for r, c in _line_cells((height - 2, col), (trunk_row, col)):
                grid[r, c] = GREY
            grid[height - 1, col] = color

    trunk_frame_centers = []
    for col, side in zip(frame_cols, frame_sides):
        center = _place_trunk_frame(grid, trunk_row, col, side)
        trunk_frame_centers.append(center)

    right_assignment: tuple[int, tuple[int, int]] | None = None
    if count == 4:
        independent_color = extra_colors[0]
        reserved_rows = _reserve_rows(height, legend_rows, trunk_row, frame_sides)
        valid_rows = [
            r
            for r in range(2, height - 2)
            if r not in reserved_rows and all(abs(r - rr) >= 2 for rr in legend_rows)
        ]
        if not valid_rows:
            return None
        right_row = random.choice(valid_rows)
        end_col = random.randint(max(5, width - 12), width - 5)
        for r, c in _line_cells((right_row, width - 2), (right_row, end_col)):
            grid[r, c] = GREY
        grid[right_row, width - 1] = independent_color

        side = random.choice(("up", "down"))
        if side == "up" and right_row < 4:
            side = "down"
        if side == "down" and right_row > height - 5:
            side = "up"
        center = _place_trunk_frame(grid, right_row, end_col, side)
        right_assignment = (independent_color, center)

    feeder_assignments = sorted(
        ((merge_col, color) for merge_col, color, _side in feeder_specs),
        key=lambda item: item[0],
    )

    expected = _expected_output(
        grid,
        main_color,
        feeder_assignments,
        trunk_frame_centers,
        right_assignment,
    )
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid routing input")
