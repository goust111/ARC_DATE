import random

import numpy as np


BLACK = 0
GREY = 5
AZURE = 8
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


# --- solver helpers ---


def _find_separator_and_legend(grid: np.ndarray) -> tuple[int, int, list[int]]:
    h, w = grid.shape

    separator = None
    for r in range(h - 2, -1, -1):
        if np.all(grid[r, :] == GREY):
            separator = r
            break
    assert separator is not None, "expected a full-width grey separator row"

    legend_colors = None
    for r in range(separator + 1, h):
        cols = np.flatnonzero(grid[r, :] != BLACK)
        if cols.size == 0:
            continue
        if cols.size == 5 and np.all(cols == np.arange(cols[0], cols[0] + 5)):
            colors = [int(grid[r, c]) for c in cols]
            assert len(set(colors)) == 5, "legend colors must be distinct"
            legend_colors = colors
            break
        raise AssertionError("expected exactly one contiguous 5-cell legend strip")

    assert legend_colors is not None, "legend strip not found below separator"
    return int(separator), int(r), legend_colors


def _interior_black_components(area: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = area.shape
    black = area == BLACK
    seen = np.zeros_like(black, dtype=bool)
    rooms: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not black[r, c] or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            touches_border = False

            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                if rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1:
                    touches_border = True
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and black[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            if not touches_border:
                rooms.append(coords)

    assert rooms, "expected at least one enclosed room"
    return rooms


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count azure window pixels on each enclosed room, use the bottom legend as
    a 1..5 lookup table for floor color, fill the room, and close the windows.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    separator, _legend_row, legend_colors = _find_separator_and_legend(g)
    rooms = _interior_black_components(g[:separator, :])

    out = g.copy()
    windows_to_close: set[tuple[int, int]] = set()

    for room in rooms:
        room_windows: set[tuple[int, int]] = set()
        for r, c in room:
            for dr, dc in DIR4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < separator and 0 <= cc < g.shape[1] and g[rr, cc] == AZURE:
                    room_windows.add((rr, cc))

        count = len(room_windows)
        assert 1 <= count <= 5, "each room must have between 1 and 5 windows"
        fill_color = int(legend_colors[count - 1])

        for r, c in room:
            out[r, c] = fill_color
        windows_to_close.update(room_windows)

    for r, c in windows_to_close:
        out[r, c] = GREY

    return out


# --- generator helpers ---


GEN_ATTEMPTS = 400
ROOM_GRID_SHAPES = ((1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2))
ROOM_SIZE_RANGE = (3, 6)
MARGIN_RANGE = (1, 3)
EXTRA_CANVAS_RANGE = (0, 4)
TARGET_WINDOW_RANGE = (1, 5)
LEGEND_COLORS_POOL = (1, 2, 3, 4, 6, 7, 9)
TARGET_SAMPLE_ATTEMPTS = 80
WINDOW_PLAN_ATTEMPTS = 120


def _sample_targets(room_count: int) -> list[int]:
    for _ in range(TARGET_SAMPLE_ATTEMPTS):
        targets = [random.randint(*TARGET_WINDOW_RANGE) for _ in range(room_count)]
        if len(set(targets)) > 1 or random.random() < 0.25:
            return targets
    return targets


def _choose_positions(cells: list[tuple[int, int]], count: int) -> list[tuple[int, int]]:
    assert 0 <= count <= len(cells)
    if count == 0:
        return []
    if count == len(cells):
        return list(cells)

    indexed = list(range(len(cells)))
    random.shuffle(indexed)

    chosen: list[int] = []
    used = set()
    for idx in indexed:
        if idx - 1 in used or idx + 1 in used:
            continue
        used.add(idx)
        chosen.append(idx)
        if len(chosen) == count:
            break

    if len(chosen) < count:
        remaining = [idx for idx in indexed if idx not in used]
        chosen.extend(remaining[: count - len(chosen)])

    return [cells[idx] for idx in chosen]


def _solve_window_plan(
    outer_positions: list[list[tuple[int, int]]],
    shared_segments: list[dict],
) -> tuple[list[int], set[tuple[int, int]]] | None:
    room_count = len(outer_positions)
    outer_caps = [len(cells) for cells in outer_positions]
    assert all(cap > 0 for cap in outer_caps)

    for _ in range(WINDOW_PLAN_ATTEMPTS):
        targets = _sample_targets(room_count)
        require_shared = bool(shared_segments) and random.random() < 0.7

        order = list(range(len(shared_segments)))
        random.shuffle(order)
        ordered_segments = [shared_segments[idx] for idx in order]

        suffix_caps = [[0] * room_count for _ in range(len(ordered_segments) + 1)]
        for idx in range(len(ordered_segments) - 1, -1, -1):
            suffix_caps[idx] = suffix_caps[idx + 1].copy()
            a, b = ordered_segments[idx]["rooms"]
            cap = len(ordered_segments[idx]["positions"])
            suffix_caps[idx][a] += cap
            suffix_caps[idx][b] += cap

        def dfs(
            idx: int,
            remaining: tuple[int, ...],
            used_shared: bool,
        ) -> list[int] | None:
            for room, rem in enumerate(remaining):
                if rem < 0:
                    return None
                if rem > outer_caps[room] + suffix_caps[idx][room]:
                    return None

            if idx == len(ordered_segments):
                if require_shared and not used_shared:
                    return None
                if all(rem <= outer_caps[room] for room, rem in enumerate(remaining)):
                    return []
                return None

            a, b = ordered_segments[idx]["rooms"]
            cap = min(
                len(ordered_segments[idx]["positions"]),
                remaining[a],
                remaining[b],
            )

            positives = list(range(1, cap + 1))
            random.shuffle(positives)
            if require_shared and not used_shared:
                choices = positives + [0]
            else:
                choices = [0] + positives if random.random() < 0.5 else positives + [0]

            for use in choices:
                next_remaining = list(remaining)
                next_remaining[a] -= use
                next_remaining[b] -= use
                result = dfs(idx + 1, tuple(next_remaining), used_shared or use > 0)
                if result is not None:
                    return [use] + result
            return None

        shared_counts = dfs(0, tuple(targets), False)
        if shared_counts is None:
            continue

        outer_counts = targets[:]
        chosen: set[tuple[int, int]] = set()
        for segment, count in zip(ordered_segments, shared_counts):
            if count == 0:
                continue
            a, b = segment["rooms"]
            outer_counts[a] -= count
            outer_counts[b] -= count
            chosen.update(_choose_positions(segment["positions"], count))

        if any(count < 0 or count > len(outer_positions[room]) for room, count in enumerate(outer_counts)):
            continue

        for room, count in enumerate(outer_counts):
            chosen.update(_choose_positions(outer_positions[room], count))

        if require_shared and not any(count > 0 for count in shared_counts):
            continue
        return targets, chosen

    return None


def _gen_attempt() -> np.ndarray | None:
    room_rows, room_cols = random.choice(ROOM_GRID_SHAPES)
    heights = [random.randint(*ROOM_SIZE_RANGE) for _ in range(room_rows)]
    widths = [random.randint(*ROOM_SIZE_RANGE) for _ in range(room_cols)]

    box_h = sum(heights) + room_rows + 1
    box_w = sum(widths) + room_cols + 1

    top_margin = random.randint(*MARGIN_RANGE)
    bottom_margin = random.randint(*MARGIN_RANGE)
    left_margin = random.randint(*MARGIN_RANGE)
    right_margin = random.randint(*MARGIN_RANGE)
    extra_h = random.randint(*EXTRA_CANVAS_RANGE)
    extra_w = random.randint(*EXTRA_CANVAS_RANGE)

    box_top = top_margin
    box_left = left_margin
    box_bottom = box_top + box_h - 1
    box_right = box_left + box_w - 1

    main_h = top_margin + box_h + bottom_margin + extra_h
    grid_h = main_h + 2
    grid_w = box_left + box_w + right_margin + extra_w
    grid_w = max(grid_w, 7)

    separator_row = main_h
    legend_row = main_h + 1

    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    grid[separator_row, :] = GREY

    row_lines = [box_top]
    for size in heights:
        row_lines.append(row_lines[-1] + size + 1)
    col_lines = [box_left]
    for size in widths:
        col_lines.append(col_lines[-1] + size + 1)

    for r in row_lines:
        grid[r, box_left : box_right + 1] = GREY
    for c in col_lines:
        grid[box_top : box_bottom + 1, c] = GREY

    rooms: list[dict] = []
    outer_positions: list[list[tuple[int, int]]] = []

    for i in range(room_rows):
        for j in range(room_cols):
            room_id = i * room_cols + j
            r0 = row_lines[i] + 1
            r1 = row_lines[i + 1] - 1
            c0 = col_lines[j] + 1
            c1 = col_lines[j + 1] - 1
            rooms.append({"id": room_id, "r0": r0, "r1": r1, "c0": c0, "c1": c1})

            side_cells: list[tuple[int, int]] = []
            if i == 0:
                side_cells.extend((row_lines[i], c) for c in range(c0, c1 + 1))
            if i == room_rows - 1:
                side_cells.extend((row_lines[i + 1], c) for c in range(c0, c1 + 1))
            if j == 0:
                side_cells.extend((r, col_lines[j]) for r in range(r0, r1 + 1))
            if j == room_cols - 1:
                side_cells.extend((r, col_lines[j + 1]) for r in range(r0, r1 + 1))
            outer_positions.append(side_cells)

    shared_segments: list[dict] = []
    for i in range(room_rows - 1):
        for j in range(room_cols):
            upper = i * room_cols + j
            lower = (i + 1) * room_cols + j
            row = row_lines[i + 1]
            c0 = col_lines[j] + 1
            c1 = col_lines[j + 1] - 1
            shared_segments.append(
                {
                    "rooms": (upper, lower),
                    "positions": [(row, c) for c in range(c0, c1 + 1)],
                }
            )
    for i in range(room_rows):
        for j in range(room_cols - 1):
            left = i * room_cols + j
            right = i * room_cols + j + 1
            col = col_lines[j + 1]
            r0 = row_lines[i] + 1
            r1 = row_lines[i + 1] - 1
            shared_segments.append(
                {
                    "rooms": (left, right),
                    "positions": [(r, col) for r in range(r0, r1 + 1)],
                }
            )

    plan = _solve_window_plan(outer_positions, shared_segments)
    if plan is None:
        return None
    targets, windows = plan

    for r, c in windows:
        grid[r, c] = AZURE

    legend_colors = random.sample(list(LEGEND_COLORS_POOL), 5)
    legend_start = random.randint(0, grid_w - 5)
    grid[legend_row, legend_start : legend_start + 5] = legend_colors

    out = main(grid)
    assert np.array_equal(out[separator_row + 1 :, :], grid[separator_row + 1 :, :])
    assert not np.any(out[:separator_row, :] == AZURE)

    for room, count in zip(rooms, targets):
        patch = out[room["r0"] : room["r1"] + 1, room["c0"] : room["c1"] + 1]
        assert np.all(patch == legend_colors[count - 1])

    for r, c in windows:
        assert out[r, c] == GREY

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
