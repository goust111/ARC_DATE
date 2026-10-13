import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7
AZURE = 8

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIR4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _has_block(coords: list[tuple[int, int]]) -> bool:
    pts = set(coords)
    for r, c in coords:
        if (r + 1, c) in pts and (r, c + 1) in pts and (r + 1, c + 1) in pts:
            return True
    return False


def _source_coord(grid: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    coords = [(int(r), int(c)) for r, c in np.argwhere(grid == BLUE)]
    border = [
        (r, c)
        for r, c in coords
        if r == 0 or c == 0 or r == h - 1 or c == w - 1
    ]
    assert len(border) == 1, "expected one border source"
    return border[0]


def _event_kind(grid: np.ndarray, rc: tuple[int, int]) -> str:
    color = int(grid[rc])
    if color == BLUE:
        return "source"
    if color == RED:
        return "node"
    if color == YELLOW:
        return "terminal"
    raise AssertionError("unexpected event color")


def _branch_width(
    comp: list[tuple[int, int]],
    node: tuple[int, int],
    h: int,
    w: int,
) -> int:
    nr, nc = node
    comp_set = set(comp)
    touches = sum((ar, ac) in comp_set for ar, ac in _neighbors4(nr, nc, h, w))
    if touches >= 2 or _has_block(comp):
        return 2
    return 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow the signal from the blue source, choose the widest branch at each red node,
    recolor visited path pixels to azure, and turn reached yellow terminals orange.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    source = _source_coord(g)
    green_components = _components(g == GREEN)
    assert green_components, "expected path components"

    event_to_components: dict[tuple[int, int], list[int]] = {}
    comp_events: list[set[tuple[int, int]]] = []
    for idx, comp in enumerate(green_components):
        events: set[tuple[int, int]] = set()
        for r, c in comp:
            for nr, nc in _neighbors4(r, c, h, w):
                color = int(g[nr, nc])
                if color in (BLUE, RED, YELLOW):
                    events.add((nr, nc))
        assert len(events) >= 1, "each path component must touch an event"
        comp_events.append(events)
        for event in events:
            event_to_components.setdefault(event, []).append(idx)

    assert source in event_to_components, "source must touch a path"

    out = g.copy()
    visited_components: set[int] = set()
    processed_arrivals: set[tuple[tuple[int, int], int | None]] = set()

    def follow_component(comp_idx: int, from_event: tuple[int, int]) -> None:
        if comp_idx in visited_components:
            return
        visited_components.add(comp_idx)
        for r, c in green_components[comp_idx]:
            out[r, c] = AZURE

        other_events = [event for event in comp_events[comp_idx] if event != from_event]
        assert len(other_events) == 1, "path component must connect exactly two events"
        target = other_events[0]
        kind = _event_kind(g, target)
        if kind == "terminal":
            out[target] = ORANGE
            return
        if kind == "node":
            traverse_event(target, comp_idx)

    def traverse_event(event: tuple[int, int], incoming: int | None) -> None:
        key = (event, incoming)
        if key in processed_arrivals:
            return
        processed_arrivals.add(key)

        kind = _event_kind(g, event)
        if kind == "terminal":
            out[event] = ORANGE
            return
        if kind == "node":
            out[event] = AZURE

        options: list[tuple[int, int]] = []
        for comp_idx in event_to_components.get(event, []):
            if comp_idx == incoming or comp_idx in visited_components:
                continue
            width = _branch_width(green_components[comp_idx], event, h, w)
            options.append((width, comp_idx))
        if not options:
            return

        max_width = max(width for width, _ in options)
        for width, comp_idx in options:
            if width == max_width:
                follow_component(comp_idx, event)

    traverse_event(source, None)
    return out


GEN_ATTEMPTS = 500
H_RANGE = (17, 25)
STEP_RANGE = (5, 7)
ROOT_COL_RANGE = (3, 4)
NODE_COUNT_RANGE = (1, 3)
SHIFT_MIN = 4
TOP_MARGIN = 2
BOTTOM_MARGIN = 2


def _rotate_point(r: int, c: int, shape: tuple[int, int], turns: int) -> tuple[int, int]:
    h, w = shape
    turns %= 4
    if turns == 0:
        return r, c
    if turns == 1:
        return c, h - 1 - r
    if turns == 2:
        return h - 1 - r, w - 1 - c
    return w - 1 - c, r


def _rotate_grid(grid: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(grid, turns)


def _right_offset_row(row: int, top: int, bottom: int) -> int:
    choices = []
    if row - 1 >= top:
        choices.append(row - 1)
    if row + 1 <= bottom:
        choices.append(row + 1)
    assert choices, "right branch needs room for width-2 strip"
    return random.choice(choices)


def _edge_cells(
    start: tuple[int, int],
    end: tuple[int, int],
    width: int,
    top: int,
    bottom: int,
) -> set[tuple[int, int]]:
    sr, sc = start
    er, ec = end
    assert ec > sc
    cells: set[tuple[int, int]] = set()

    if er == sr:
        for c in range(sc + 1, ec):
            cells.add((sr, c))
        if width == 2:
            extra_r = _right_offset_row(sr, top, bottom)
            for c in range(sc + 2, ec):
                cells.add((extra_r, c))
        return cells

    if er < sr:
        for r in range(er, sr):
            cells.add((r, sc))
    else:
        for r in range(sr + 1, er + 1):
            cells.add((r, sc))
    for c in range(sc + 1, ec):
        cells.add((er, c))

    if width == 2:
        extra_r = er + 1 if er < sr else er - 1
        assert top <= extra_r <= bottom
        for c in range(sc + 2, ec):
            cells.add((extra_r, c))
    return cells


def _min_band_height(remaining_nodes: int) -> int:
    return 1 + remaining_nodes * SHIFT_MIN + 4


def _pick_direction(
    row: int,
    top: int,
    bottom: int,
    remaining_nodes: int,
) -> str | None:
    candidates = ["right"]
    if row - top >= SHIFT_MIN and row - 3 - top >= _min_band_height(remaining_nodes):
        candidates.append("up")
    if bottom - row >= SHIFT_MIN and bottom - (row + 3) >= _min_band_height(remaining_nodes):
        candidates.append("down")
    if not candidates:
        return None
    return random.choice(candidates)


def _pick_branch_rows(
    row: int,
    top: int,
    bottom: int,
    dirs: list[str],
) -> dict[str, int] | None:
    rows: dict[str, int] = {}
    if "right" in dirs:
        rows["right"] = row
    if "up" in dirs:
        lo = top + 1
        hi = row - SHIFT_MIN
        if lo > hi:
            return None
        rows["up"] = random.randint(lo, hi)
    if "down" in dirs:
        lo = row + SHIFT_MIN
        hi = bottom - 1
        if lo > hi:
            return None
        rows["down"] = random.randint(lo, hi)
    return rows


def _decision_pattern(main_dir: str, branch_dirs: list[str]) -> tuple[set[str], dict[str, int]]:
    side_dirs = [d for d in branch_dirs if d != main_dir]
    modes = ["unique"]
    if side_dirs:
        modes.append("tie_one_side")
        modes.append("tie_all")
    mode = random.choice(modes)

    active = {main_dir}
    widths = {d: 1 for d in branch_dirs}
    if mode == "unique":
        widths[main_dir] = 2
        return active, widths
    if mode == "tie_one_side":
        extra = random.choice(side_dirs)
        active.add(extra)
        for d in active:
            widths[d] = 2
        return active, widths

    active = set(branch_dirs)
    tie_width = random.choice([1, 2])
    widths = {d: tie_width for d in branch_dirs}
    return active, widths


def _build_canonical() -> np.ndarray | None:
    node_count = random.randint(*NODE_COUNT_RANGE)
    step = random.randint(*STEP_RANGE)
    root_col = random.randint(*ROOT_COL_RANGE)
    min_w = root_col + node_count * step + 2
    h = random.randint(*H_RANGE)
    max_w = max(min_w, min(32, min_w + 6))
    w = random.randint(min_w, max_w)

    root_row = random.randint(TOP_MARGIN + SHIFT_MIN, h - 1 - (BOTTOM_MARGIN + SHIFT_MIN))
    source = (root_row, 0)
    top = TOP_MARGIN
    bottom = h - 1 - BOTTOM_MARGIN

    nodes = [(root_row, root_col)]
    terminals: set[tuple[int, int]] = set()
    edges: list[dict[str, object]] = []

    source_width = random.choice([1, 2])
    edges.append(
        {
            "start": source,
            "end": nodes[0],
            "width": source_width,
            "top": top,
            "bottom": bottom,
        }
    )

    current_row = root_row
    current_col = root_col
    current_top = top
    current_bottom = bottom

    for i in range(node_count):
        remaining_nodes = node_count - i - 1
        main_dir = _pick_direction(current_row, current_top, current_bottom, remaining_nodes)
        if main_dir is None:
            return None

        side_candidates = [
            d
            for d in ("up", "right", "down")
            if d != main_dir
            and (
                d == "right"
                or (d == "up" and current_row - current_top >= SHIFT_MIN)
                or (d == "down" and current_bottom - current_row >= SHIFT_MIN)
            )
        ]
        if not side_candidates:
            return None

        branch_dirs = [main_dir]
        random.shuffle(side_candidates)
        side_count = 1 if len(side_candidates) == 1 else random.choice([1, 2])
        branch_dirs.extend(side_candidates[:side_count])

        branch_rows = _pick_branch_rows(current_row, current_top, current_bottom, branch_dirs)
        if branch_rows is None:
            return None
        active_dirs, widths = _decision_pattern(main_dir, branch_dirs)
        assert main_dir in active_dirs

        next_col = current_col + step
        next_node = None
        for direction in branch_dirs:
            end = (branch_rows[direction], next_col)
            is_main_child = direction == main_dir and i < node_count - 1
            if is_main_child:
                next_node = end
                nodes.append(end)
            else:
                terminals.add(end)
            edges.append(
                {
                    "start": (current_row, current_col),
                    "end": end,
                    "width": widths[direction],
                    "top": current_top,
                    "bottom": current_bottom,
                }
            )

        if i == node_count - 1:
            break
        assert next_node is not None
        if main_dir == "up":
            current_bottom = current_row - 3
        elif main_dir == "down":
            current_top = current_row + 3
        current_row, current_col = next_node

    grid = np.full((h, w), BLACK, dtype=int)
    grid[source] = BLUE
    for node in nodes:
        if grid[node] != BLACK:
            return None
        grid[node] = RED
    for terminal in terminals:
        if grid[terminal] != BLACK:
            return None
        grid[terminal] = YELLOW

    for edge in edges:
        coords = _edge_cells(
            edge["start"],
            edge["end"],
            int(edge["width"]),
            int(edge["top"]),
            int(edge["bottom"]),
        )
        if not coords:
            return None
        for r, c in coords:
            if not (0 <= r < h and 0 <= c < w):
                return None
            if grid[r, c] != BLACK:
                return None
        for r, c in coords:
            grid[r, c] = GREEN

    green_comps = _components(grid == GREEN)
    if len(green_comps) != len(edges):
        return None
    source_touch = sum(
        abs(r - source[0]) + abs(c - source[1]) == 1
        for r, c in np.argwhere(grid == GREEN)
    )
    if source_touch != 1:
        return None
    return grid


def _count_reached_terminals(inp: np.ndarray, out: np.ndarray) -> int:
    return int(np.sum((inp == YELLOW) & (out == ORANGE)))


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        canonical = _build_canonical()
        if canonical is None:
            continue
        turns = random.randrange(4)
        grid = _rotate_grid(canonical, turns)
        try:
            out = main(grid)
        except AssertionError:
            continue
        if _count_reached_terminals(grid, out) == 0:
            continue
        if not np.any(out == AZURE):
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
