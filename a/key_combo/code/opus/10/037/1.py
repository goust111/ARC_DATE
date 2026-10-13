from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
GREY = 5
MARKER_COLORS = (2, 3, 4, 6, 7, 8, 9)

MAX_GENERATION_ATTEMPTS = 400
GRID_RANGE_BY_FRAME_COUNT = {
    2: ((20, 25), (20, 25)),
    3: ((23, 28), (23, 28)),
}
FRAME_GAP = 2


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.asarray(grid).ravel()))
    if exclude is not None:
        colors -= set(exclude)
    return sorted(colors)


def _corner_cells(top: int, left: int, bottom: int, right: int) -> set[tuple[int, int]]:
    return {
        (top, left),
        (top, left + 1),
        (top + 1, left),
        (top, right - 1),
        (top, right),
        (top + 1, right),
        (bottom - 1, left),
        (bottom, left),
        (bottom, left + 1),
        (bottom - 1, right),
        (bottom, right - 1),
        (bottom, right),
    }


def _find_frames(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    frames: list[dict[str, object]] = []
    for color in colors_present(g, exclude={BLACK, BLUE, GREY}):
        coords = np.argwhere(g == color)
        assert len(coords) == 12, f"marker color {color} should occupy exactly 12 cells"
        top, left = coords.min(axis=0)
        bottom, right = coords.max(axis=0)
        top, left, bottom, right = map(int, (top, left, bottom, right))
        assert bottom - top + 1 >= 5 and right - left + 1 >= 5
        expected = _corner_cells(top, left, bottom, right)
        actual = {tuple(map(int, rc)) for rc in coords}
        assert actual == expected, f"marker color {color} is not a valid corner frame"
        frames.append(
            {
                "color": int(color),
                "bbox": (top, left, bottom, right),
                "direction": "right" if (right - left) > (bottom - top) else "down",
            }
        )
    frames.sort(key=lambda frame: tuple(frame["bbox"]))  # type: ignore[arg-type]
    return frames


def _sort_blues(blues: list[tuple[int, int]], direction: str) -> list[tuple[int, int]]:
    if direction == "right":
        return sorted(blues, key=lambda rc: (-rc[1], rc[0]))
    return sorted(blues, key=lambda rc: (-rc[0], rc[1]))


def _settle_blues(
    blues: list[tuple[int, int]],
    obstacles: set[tuple[int, int]],
    bbox: tuple[int, int, int, int],
    direction: str,
) -> list[tuple[int, int]]:
    top, left, bottom, right = bbox
    occupied = set(obstacles)
    settled: list[tuple[int, int]] = []
    for r, c in _sort_blues(blues, direction):
        if direction == "right":
            nc = c
            while nc + 1 <= right - 1 and (r, nc + 1) not in occupied:
                nc += 1
            occupied.add((r, nc))
            settled.append((r, nc))
        else:
            nr = r
            while nr + 1 <= bottom - 1 and (nr + 1, c) not in occupied:
                nr += 1
            occupied.add((nr, c))
            settled.append((nr, c))
    return settled


def _simulate_scene(
    grid: np.ndarray,
    *,
    direction_override: str | None = None,
    ignore_obstacles: bool = False,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    frames = _find_frames(g)
    for frame in frames:
        bbox = frame["bbox"]  # type: ignore[assignment]
        top, left, bottom, right = bbox
        direction = direction_override or str(frame["direction"])

        blues = [
            (int(r), int(c))
            for r, c in np.argwhere(g[top + 1 : bottom, left + 1 : right] == BLUE)
            for _ in (None,)
            for r, c in [(int(r + top + 1), int(c + left + 1))]
        ]
        obstacles = set()
        if not ignore_obstacles:
            obstacles = {
                (int(r + top + 1), int(c + left + 1))
                for r, c in np.argwhere(g[top + 1 : bottom, left + 1 : right] == GREY)
            }

        interior = out[top + 1 : bottom, left + 1 : right]
        interior[interior == BLUE] = BLACK
        for r, c in _settle_blues(blues, obstacles, bbox, direction):
            out[r, c] = BLUE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Slide the blue pixels inside each corner-marked rectangle using a gravity direction chosen by that rectangle's aspect ratio.

    Input structure:
    - A black background contains 2-3 non-overlapping rectangular frames.
    - Each frame is defined only by four same-colored 2x2 L-corner triominoes, one at each rectangle corner.
    - Inside each frame are scattered blue pixels and fixed grey obstacle pixels.

    Rule:
    - Detect each frame from its four same-colored corner markers and measure its bounding-box width and height.
    - Wide frames (W > H) apply gravity to the right; tall frames (H > W) apply gravity downward.
    - Keep every marker pixel and every grey obstacle fixed.
    - Move each blue pixel cell-by-cell along that frame's gravity direction until the next step would hit the frame edge, a grey obstacle, or a previously settled blue pixel; then stop on the previous cell.
    - Process blue pixels from far to near along the gravity axis: rightward uses rightmost-to-leftmost columns, downward uses bottom-to-top rows.

    Key insight:
    - The corner markers are only a geometric cue: first recover each rectangle, then its aspect ratio determines a frame-local gravity direction, and only then can the internal blue motion be resolved.
    """
    return _simulate_scene(np.asarray(grid, dtype=int))


def _rectangles_far_enough(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = FRAME_GAP
) -> bool:
    at, al, ab, ar = a
    bt, bl, bb, br = b
    return ab + gap < bt or bb + gap < at or ar + gap < bl or br + gap < al


def _sample_frame_size(direction: str) -> tuple[int, int]:
    if direction == "right":
        height = random.randint(7, 9)
        width = random.randint(height + 1, min(12, height + 4))
    else:
        width = random.randint(7, 9)
        height = random.randint(width + 1, min(12, width + 4))
    return height, width


def _line_slots(start: int, stop: int, *, offset: int) -> list[int]:
    return list(range(start + offset, stop + 1, 2))


def _choose_two_spaced(values: list[int]) -> tuple[int, int] | None:
    choices = [(a, b) for a in values for b in values if abs(a - b) >= 2]
    if not choices:
        return None
    return random.choice(choices)


def _populate_wide_frame(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    top, left, bottom, right = bbox
    inner_rows = list(range(top + 1, bottom))
    inner_cols = list(range(left + 1, right))
    pair = _choose_two_spaced(inner_rows)
    if pair is None:
        return False
    primary_row, secondary_row = pair

    width_capacity = (len(inner_cols) + 1) // 2
    blue_total = random.randint(3, min(6, 2 * width_capacity))
    primary_min = max(2, blue_total - width_capacity)
    primary_max = min(3, width_capacity, blue_total - 1)
    if primary_min > primary_max:
        return False
    primary_count = random.randint(primary_min, primary_max)
    secondary_count = blue_total - primary_count

    obstacle_col = random.choice(inner_cols[-2:])
    primary_slots = [c for c in _line_slots(left + 1, obstacle_col - 2, offset=0) if c <= obstacle_col - 2]
    if len(primary_slots) < primary_count + 1:
        primary_slots = [c for c in _line_slots(left + 1, obstacle_col - 2, offset=1) if c <= obstacle_col - 2]
    if len(primary_slots) < primary_count:
        return False
    if len(primary_slots) > primary_count:
        primary_blue_cols = sorted(random.sample(primary_slots[:-1], primary_count))
    else:
        primary_blue_cols = sorted(primary_slots[:primary_count])

    secondary_offset = random.randint(0, 1)
    secondary_slots = _line_slots(left + 1, right - 1, offset=secondary_offset)
    if len(secondary_slots) < secondary_count:
        secondary_slots = _line_slots(left + 1, right - 1, offset=1 - secondary_offset)
    if len(secondary_slots) < secondary_count:
        return False
    secondary_blue_cols = sorted(random.sample(secondary_slots, secondary_count))

    blue_cells = {(primary_row, c) for c in primary_blue_cols} | {(secondary_row, c) for c in secondary_blue_cols}
    obstacle_cells = {(primary_row, obstacle_col)}

    extra_obstacles = random.randint(0, 2)
    candidates = [
        (r, c)
        for r in inner_rows
        for c in inner_cols
        if (r, c) not in blue_cells and (r, c) not in obstacle_cells and not (r == primary_row and c > max(primary_blue_cols))
    ]
    random.shuffle(candidates)
    for cell in candidates[:extra_obstacles]:
        obstacle_cells.add(cell)

    for r, c in blue_cells:
        grid[r, c] = BLUE
    for r, c in obstacle_cells:
        grid[r, c] = GREY
    return True


def _populate_tall_frame(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    top, left, bottom, right = bbox
    inner_rows = list(range(top + 1, bottom))
    inner_cols = list(range(left + 1, right))
    pair = _choose_two_spaced(inner_cols)
    if pair is None:
        return False
    primary_col, secondary_col = pair

    height_capacity = (len(inner_rows) + 1) // 2
    blue_total = random.randint(3, min(6, 2 * height_capacity))
    primary_min = max(2, blue_total - height_capacity)
    primary_max = min(3, height_capacity, blue_total - 1)
    if primary_min > primary_max:
        return False
    primary_count = random.randint(primary_min, primary_max)
    secondary_count = blue_total - primary_count

    obstacle_row = random.choice(inner_rows[-2:])
    primary_slots = [r for r in _line_slots(top + 1, obstacle_row - 2, offset=0) if r <= obstacle_row - 2]
    if len(primary_slots) < primary_count + 1:
        primary_slots = [r for r in _line_slots(top + 1, obstacle_row - 2, offset=1) if r <= obstacle_row - 2]
    if len(primary_slots) < primary_count:
        return False
    if len(primary_slots) > primary_count:
        primary_blue_rows = sorted(random.sample(primary_slots[:-1], primary_count))
    else:
        primary_blue_rows = sorted(primary_slots[:primary_count])

    secondary_offset = random.randint(0, 1)
    secondary_slots = _line_slots(top + 1, bottom - 1, offset=secondary_offset)
    if len(secondary_slots) < secondary_count:
        secondary_slots = _line_slots(top + 1, bottom - 1, offset=1 - secondary_offset)
    if len(secondary_slots) < secondary_count:
        return False
    secondary_blue_rows = sorted(random.sample(secondary_slots, secondary_count))

    blue_cells = {(r, primary_col) for r in primary_blue_rows} | {(r, secondary_col) for r in secondary_blue_rows}
    obstacle_cells = {(obstacle_row, primary_col)}

    extra_obstacles = random.randint(0, 2)
    candidates = [
        (r, c)
        for r in inner_rows
        for c in inner_cols
        if (r, c) not in blue_cells and (r, c) not in obstacle_cells and not (c == primary_col and r > max(primary_blue_rows))
    ]
    random.shuffle(candidates)
    for cell in candidates[:extra_obstacles]:
        obstacle_cells.add(cell)

    for r, c in blue_cells:
        grid[r, c] = BLUE
    for r, c in obstacle_cells:
        grid[r, c] = GREY
    return True


def _frame_counts(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    top, left, bottom, right = bbox
    patch = np.asarray(grid[top + 1 : bottom, left + 1 : right], dtype=int)
    blues = [(int(r + top + 1), int(c + left + 1)) for r, c in np.argwhere(patch == BLUE)]
    obstacles = [(int(r + top + 1), int(c + left + 1)) for r, c in np.argwhere(patch == GREY)]
    return blues, obstacles


def _nonadjacent(coords: list[tuple[int, int]]) -> bool:
    pts = set(coords)
    for r, c in pts:
        if (r + 1, c) in pts or (r - 1, c) in pts or (r, c + 1) in pts or (r, c - 1) in pts:
            return False
    return True


def _validate_instance(grid: np.ndarray) -> bool:
    try:
        frames = _find_frames(grid)
    except AssertionError:
        return False

    if len(frames) not in (2, 3):
        return False

    directions = [str(frame["direction"]) for frame in frames]
    if "right" not in directions or "down" not in directions:
        return False

    if np.any(grid == BLUE):
        out = main(grid)
    else:
        return False

    if np.array_equal(out, grid):
        return False

    if np.array_equal(out, _simulate_scene(grid, direction_override="right")):
        return False
    if np.array_equal(out, _simulate_scene(grid, direction_override="down")):
        return False
    if np.array_equal(out, _simulate_scene(grid, ignore_obstacles=True)):
        return False

    for frame in frames:
        bbox = frame["bbox"]  # type: ignore[assignment]
        top, left, bottom, right = bbox
        blues, obstacles = _frame_counts(grid, bbox)
        if not (3 <= len(blues) <= 6 and 1 <= len(obstacles) <= 3):
            return False
        if not _nonadjacent(blues):
            return False
        patch = grid[top + 1 : bottom, left + 1 : right]
        if not np.all(np.isin(patch, [BLACK, BLUE, GREY])):
            return False
        if str(frame["direction"]) == "right":
            if len({r for r, _ in blues}) == len(blues):
                return False
            if not any((r, c) in obstacles for r, c in {(r, c + 1) for r, c in _settle_blues(blues, set(obstacles), bbox, "right")}):
                # Require at least one obstacle-mediated landing row in some wide frame.
                if any((r, c) in obstacles for r, c in blues):
                    return False
        else:
            if len({c for _, c in blues}) == len(blues):
                return False
            if not any((r + 1, c) in obstacles for r, c in _settle_blues(blues, set(obstacles), bbox, "down")):
                if any((r, c) in obstacles for r, c in blues):
                    return False

    return True


def _place_frame_markers(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    for r, c in _corner_cells(*bbox):
        grid[r, c] = color


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.randint(2, 3)
    (hmin, hmax), (wmin, wmax) = GRID_RANGE_BY_FRAME_COUNT[frame_count]
    H = random.randint(hmin, hmax)
    W = random.randint(wmin, wmax)
    grid = np.zeros((H, W), dtype=int)

    if frame_count == 2:
        directions = ["right", "down"]
    else:
        directions = random.choice(
            [["right", "right", "down"], ["right", "down", "down"]]
        )
    random.shuffle(directions)
    colors = random.sample(MARKER_COLORS, frame_count)

    frames: list[dict[str, object]] = []
    for direction, color in zip(directions, colors):
        placed = False
        for _ in range(120):
            height, width = _sample_frame_size(direction)
            if height > H or width > W:
                continue
            top = random.randint(0, H - height)
            left = random.randint(0, W - width)
            bbox = (top, left, top + height - 1, left + width - 1)
            if all(_rectangles_far_enough(bbox, other["bbox"]) for other in frames):  # type: ignore[arg-type]
                frames.append({"bbox": bbox, "direction": direction, "color": color})
                placed = True
                break
        if not placed:
            return None

    for frame in frames:
        _place_frame_markers(grid, frame["bbox"], int(frame["color"]))  # type: ignore[arg-type]

    for frame in frames:
        bbox = frame["bbox"]  # type: ignore[assignment]
        ok = _populate_wide_frame(grid, bbox) if frame["direction"] == "right" else _populate_tall_frame(grid, bbox)
        if not ok:
            return None

    if not _validate_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for the aspect-ratio gravity puzzle")
