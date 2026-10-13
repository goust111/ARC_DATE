from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
MAROON = 9

NODE_COLORS = (GREEN, BLUE, YELLOW)
CONDUCTIVE = {GREY, RED, GREEN, BLUE, YELLOW, MAROON}
DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(shape: tuple[int, int], r: int, c: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if _in_bounds(shape, nr, nc):
            out.append((nr, nc))
    return out


def _left(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return (-dc, dr)


def _right(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return (dc, -dr)


def _gray_neighbors(grid: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    return [(nr, nc) for nr, nc in _neighbors4(grid.shape, r, c) if int(grid[nr, nc]) == GREY]


def _signal_neighbors(grid: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    return [
        (nr, nc)
        for nr, nc in _neighbors4(grid.shape, r, c)
        if int(grid[nr, nc]) in CONDUCTIVE
    ]


def _trace_from(grid: np.ndarray, out: np.ndarray, prev: tuple[int, int], cur: tuple[int, int]) -> None:
    seen: set[tuple[int, int, int, int]] = set()

    while True:
        pr, pc = prev
        r, c = cur
        state = (pr, pc, r, c)
        if state in seen:
            return
        seen.add(state)

        value = int(grid[r, c])
        if value == GREY:
            out[r, c] = ORANGE
            candidates = [nb for nb in _signal_neighbors(grid, r, c) if nb != prev]
            if not candidates:
                return
            if len(candidates) == 1:
                prev, cur = cur, candidates[0]
                continue

            forward = (r - pr, c - pc)
            straight = [(nr, nc) for nr, nc in candidates if (nr - r, nc - c) == forward]
            if len(straight) == 1:
                prev, cur = cur, straight[0]
                continue
            return

        if value in NODE_COLORS:
            neighbors = _gray_neighbors(grid, r, c)
            if prev not in neighbors:
                return
            if len(neighbors) == 2:
                nxt = neighbors[0] if neighbors[1] == prev else neighbors[1]
                prev, cur = cur, nxt
                continue
            if len(neighbors) != 3:
                return

            incoming = (r - pr, c - pc)
            if value == GREEN:
                chosen_dir = incoming
            elif value == BLUE:
                chosen_dir = _left(incoming)
            else:
                chosen_dir = _right(incoming)

            nr, nc = r + chosen_dir[0], c + chosen_dir[1]
            if (nr, nc) not in neighbors:
                return
            prev, cur = cur, (nr, nc)
            continue

        if value == MAROON:
            out[r, c] = ORANGE
            return

        return


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace signals from red edge sources through grey paths and light only the traversed routes and reached terminals.

    Input structure:
    - The background is black.
    - Grey pixels form single-cell-wide horizontal or vertical path segments.
    - Non-black non-grey colored pixels on the paths are green, blue, or yellow nodes;
      each node touches exactly two or three grey path branches.
    - One or more red source pixels lie on the grid edge at grey path endpoints.
    - Maroon pixels mark some path terminals at grey path endpoints.

    Rule:
    - Start a signal from every red source and move along the adjacent grey path.
    - On plain grey path cells, continue along the path; on a two-branch colored node,
      continue through the only non-entry branch regardless of node color.
    - On a three-branch node, green means go straight, blue means turn left, and yellow
      means turn right relative to the incoming signal direction.
    - Recolor every grey cell visited by any signal to orange.
    - Recolor a maroon terminal to orange if a signal reaches it; leave unreached terminals maroon.
    - Keep red sources and colored nodes unchanged, and leave all unvisited grey paths grey.

    Key insight:
    - The node colors encode relative routing instructions, so the path must be simulated with
      orientation-aware left/right turns instead of reading colors as absolute directions.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for r, c in np.argwhere(g == RED):
        rr, cc = int(r), int(c)
        starts = _gray_neighbors(g, rr, cc)
        for first in starts:
            _trace_from(g, out, (rr, cc), first)
    return out


# --------------- generator ---------------

MAX_GENERATION_ATTEMPTS = 400
BAND_STEP = 10
TOP_OFF, MID_OFF, BOT_OFF = 2, 4, 6
WIDTH_RANGE = (16, 29)
TRACK_COUNT_RANGE = (1, 3)
NODE_GAP_RANGE = (3, 5)


def _set_black_only(grid: np.ndarray, r: int, c: int, color: int) -> bool:
    if not _in_bounds(grid.shape, r, c) or int(grid[r, c]) != BLACK:
        return False
    grid[r, c] = color
    return True


def _paint_segment(
    grid: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    include_end: bool = False,
) -> bool:
    r0, c0 = start
    r1, c1 = end
    if r0 != r1 and c0 != c1:
        return False
    dr = 0 if r0 == r1 else (1 if r1 > r0 else -1)
    dc = 0 if c0 == c1 else (1 if c1 > c0 else -1)

    r, c = r0 + dr, c0 + dc
    while (r, c) != (r1, c1):
        if not _set_black_only(grid, r, c, GREY):
            return False
        r += dr
        c += dc

    if include_end:
        return _set_black_only(grid, r1, c1, GREY)
    return True


def _random_node_color() -> int:
    return random.choice(NODE_COLORS)


def _add_stub(
    grid: np.ndarray,
    node: tuple[int, int],
    direction: tuple[int, int],
    *,
    allow_terminal: bool,
) -> bool:
    r, c = node
    dr, dc = direction
    r1, c1 = r + dr, c + dc
    if not _set_black_only(grid, r1, c1, GREY):
        return False

    if allow_terminal:
        r2, c2 = r + 2 * dr, c + 2 * dc
        if _in_bounds(grid.shape, r2, c2) and int(grid[r2, c2]) == BLACK and random.random() < 0.6:
            grid[r2, c2] = MAROON
    return True


def _branch_from_row(base: int, row: int) -> tuple[int, int] | None:
    if row == base + TOP_OFF:
        return (-1, 0)
    if row == base + BOT_OFF:
        return (1, 0)
    return None


def _track_rows(track_index: int) -> tuple[int, int, int, int]:
    base = track_index * BAND_STEP
    return base, base + TOP_OFF, base + MID_OFF, base + BOT_OFF


def _choose_event(current_row: int, top: int, mid: int, bot: int, need_detour: bool) -> str:
    if current_row == mid:
        options = ["straight_branch", "detour_up", "detour_down", "inline_two"]
    else:
        options = ["straight_branch", "detour_mid", "inline_two"]

    if need_detour:
        detours = [name for name in options if name.startswith("detour")]
        return random.choice(detours)
    return random.choice(options)


def _place_track(grid: np.ndarray, track_index: int, *, must_end_terminal: bool) -> bool:
    base, top, mid, bot = _track_rows(track_index)
    width = grid.shape[1]

    grid[mid, 0] = RED
    current_row = mid
    current_col = 0

    route_limit_lo = max(10, width - 6)
    route_limit = random.randint(route_limit_lo, width - 1)
    max_events = min(4, (route_limit - 2) // 3)
    if max_events < 2:
        return False
    event_count = random.randint(2, max_events)

    detour_used = False

    for event_idx in range(event_count):
        remaining_after = event_count - event_idx - 1
        max_node_col = route_limit - (2 + 3 * remaining_after)
        lo = current_col + NODE_GAP_RANGE[0]
        hi = min(current_col + NODE_GAP_RANGE[1], max_node_col)
        if lo > hi:
            return False
        node_col = random.randint(lo, hi)
        node = (current_row, node_col)

        if not _paint_segment(grid, (current_row, current_col), node, include_end=False):
            return False

        need_detour = not detour_used and event_idx == event_count - 1
        event = _choose_event(current_row, top, mid, bot, need_detour)

        if event == "inline_two":
            grid[current_row, node_col] = _random_node_color()
            current_col = node_col
            continue

        if event == "straight_branch":
            branch_dir = _branch_from_row(base, current_row)
            if branch_dir is None:
                branch_dir = random.choice([(-1, 0), (1, 0)])
                allow_terminal = False
            else:
                allow_terminal = True
            grid[current_row, node_col] = GREEN
            if not _add_stub(grid, node, branch_dir, allow_terminal=allow_terminal):
                return False
            current_col = node_col
            continue

        detour_used = True
        if event == "detour_up":
            target_row = top
        elif event == "detour_down":
            target_row = bot
        else:
            target_row = mid

        threeway_at_start = random.choice([True, False])

        if threeway_at_start:
            start_color = BLUE if target_row < current_row else YELLOW
            grid[current_row, node_col] = start_color
            extra_dir = (1, 0) if target_row < current_row else (-1, 0)
            allow_terminal = _branch_from_row(base, current_row) == extra_dir
            if not _add_stub(grid, node, extra_dir, allow_terminal=allow_terminal):
                return False
        else:
            grid[current_row, node_col] = _random_node_color()

        end_node = (target_row, node_col)
        if not _paint_segment(grid, node, end_node, include_end=False):
            return False

        if threeway_at_start:
            grid[target_row, node_col] = _random_node_color()
        else:
            moving_down = target_row > current_row
            end_color = BLUE if moving_down else YELLOW
            grid[target_row, node_col] = end_color
            extra_dir = (1, 0) if moving_down else (-1, 0)
            allow_terminal = _branch_from_row(base, target_row) == extra_dir
            if not _add_stub(grid, end_node, extra_dir, allow_terminal=allow_terminal):
                return False

        current_row = target_row
        current_col = node_col

    end_is_terminal = must_end_terminal or random.random() < 0.7
    if end_is_terminal:
        terminal = (current_row, route_limit)
        if not _paint_segment(grid, (current_row, current_col), terminal, include_end=False):
            return False
        if not _set_black_only(grid, terminal[0], terminal[1], MAROON):
            return False
    else:
        endpoint = (current_row, route_limit)
        if not _paint_segment(grid, (current_row, current_col), endpoint, include_end=True):
            return False

    return True


def _validate_structure(grid: np.ndarray) -> bool:
    source_count = 0
    terminal_count = 0
    deg2_nodes = 0
    deg3_nodes = 0

    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            value = int(grid[r, c])
            neighbors = _signal_neighbors(grid, r, c)

            if value == BLACK:
                continue

            if value == RED:
                source_count += 1
                if r not in {0, grid.shape[0] - 1} and c not in {0, grid.shape[1] - 1}:
                    return False
                if len(neighbors) != 1 or int(grid[neighbors[0]]) != GREY:
                    return False
                continue

            if value == MAROON:
                terminal_count += 1
                if len(neighbors) != 1 or int(grid[neighbors[0]]) != GREY:
                    return False
                continue

            if value in NODE_COLORS:
                if any(int(grid[nr, nc]) != GREY for nr, nc in neighbors):
                    return False
                if len(neighbors) == 2:
                    deg2_nodes += 1
                elif len(neighbors) == 3:
                    deg3_nodes += 1
                else:
                    return False
                continue

            if value != GREY:
                return False

            if not 1 <= len(neighbors) <= 2:
                return False
            if len(neighbors) == 2:
                dirs = {(nr - r, nc - c) for nr, nc in neighbors}
                if dirs not in ({(-1, 0), (1, 0)}, {(0, -1), (0, 1)}):
                    return False

    return source_count >= 1 and terminal_count >= 1 and deg2_nodes >= 1 and deg3_nodes >= 1


def _interesting_enough(inp: np.ndarray, out: np.ndarray) -> bool:
    traversed_gray = np.any((inp == GREY) & (out == ORANGE))
    untouched_gray = np.any((inp == GREY) & (out == GREY))
    lit_terminal = np.any((inp == MAROON) & (out == ORANGE))
    return traversed_gray and untouched_gray and lit_terminal and not np.array_equal(inp, out)


def _random_transform(grid: np.ndarray) -> np.ndarray:
    out = np.rot90(grid, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    return out.copy()


def _gen_attempt() -> np.ndarray | None:
    track_count = random.randint(*TRACK_COUNT_RANGE)
    height = BAND_STEP * track_count - 1
    width = random.randint(*WIDTH_RANGE)
    grid = np.full((height, width), BLACK, dtype=int)

    for track_index in range(track_count):
        must_end_terminal = track_index == 0
        if not _place_track(grid, track_index, must_end_terminal=must_end_terminal):
            return None

    if not _validate_structure(grid):
        return None

    transformed = _random_transform(grid)
    out = main(transformed)
    if not _interesting_enough(transformed, out):
        return None
    return transformed


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid signal-routing input")
