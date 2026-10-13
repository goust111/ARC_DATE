from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7

_LEGEND_CORNERS = ("tl", "tr", "bl", "br")
_PATTERNS = (
    {
        "name": "gap_neutral",
        "tokens": "RUUUYR",
        "required_unmarked_finals": {1, -1, 0},
        "neutral_mode": "none",
    },
    {
        "name": "sandwiched_neutral",
        "tokens": "RRUYYR",
        "required_unmarked_finals": {0},
        "neutral_mode": "both",
    },
    {
        "name": "mixed_inherit",
        "tokens": "RURYYU",
        "required_unmarked_finals": {1, -1},
        "neutral_mode": None,
    },
    {
        "name": "double_opposite",
        "tokens": "RUYRRY",
        "required_unmarked_finals": {0},
        "neutral_mode": "both",
    },
    {
        "name": "long_gap_neutral",
        "tokens": "RUUUYYR",
        "required_unmarked_finals": {1, -1, 0},
        "neutral_mode": "none",
    },
)
_GEN_RETRIES = 400


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _legend_cells(grid: np.ndarray) -> set[tuple[int, int]]:
    h, w = grid.shape
    candidates: list[set[tuple[int, int]]] = []
    if h >= 2 and grid[0, 0] == GREEN and grid[1, 0] == ORANGE:
        candidates.append({(0, 0), (1, 0)})
    if h >= 2 and grid[0, w - 1] == GREEN and grid[1, w - 1] == ORANGE:
        candidates.append({(0, w - 1), (1, w - 1)})
    if h >= 2 and grid[h - 2, 0] == GREEN and grid[h - 1, 0] == ORANGE:
        candidates.append({(h - 2, 0), (h - 1, 0)})
    if h >= 2 and grid[h - 2, w - 1] == GREEN and grid[h - 1, w - 1] == ORANGE:
        candidates.append({(h - 2, w - 1), (h - 1, w - 1)})
    assert len(candidates) == 1, "expected exactly one 2x1 corner legend"
    return candidates[0]


def _wall_color(polarity_a: int, polarity_b: int) -> int:
    if polarity_a == 0 or polarity_b == 0:
        return BLUE
    return ORANGE if polarity_a == polarity_b else GREEN


def _analyze_rooms(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    passable = (g == BLACK) | (g == RED) | (g == YELLOW)

    room_id = np.full((h, w), -1, dtype=int)
    rooms: list[list[tuple[int, int]]] = []
    initial: list[int] = []

    for comp in _components(passable):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        idx = len(rooms)
        rooms.append(comp)
        has_red = any(g[r, c] == RED for r, c in comp)
        has_yellow = any(g[r, c] == YELLOW for r, c in comp)
        assert not (has_red and has_yellow), "a room cannot contain both marker colors"
        initial.append(1 if has_red else -1 if has_yellow else 0)
        for r, c in comp:
            room_id[r, c] = idx

    assert rooms, "expected at least one enclosed room"

    neighbors = [set() for _ in rooms]
    wall_pairs: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r in range(h):
        for c in range(w):
            if g[r, c] != BLUE:
                continue
            pairs = set()
            if 0 < c < w - 1:
                left = int(room_id[r, c - 1])
                right = int(room_id[r, c + 1])
                if left >= 0 and right >= 0 and left != right:
                    pairs.add(tuple(sorted((left, right))))
            if 0 < r < h - 1:
                up = int(room_id[r - 1, c])
                down = int(room_id[r + 1, c])
                if up >= 0 and down >= 0 and up != down:
                    pairs.add(tuple(sorted((up, down))))
            if not pairs:
                continue
            pair_list = sorted(pairs)
            wall_pairs[(r, c)] = pair_list
            for a, b in pair_list:
                neighbors[a].add(b)
                neighbors[b].add(a)

    initial_pos = {i for i, pol in enumerate(initial) if pol == 1}
    initial_neg = {i for i, pol in enumerate(initial) if pol == -1}
    final = initial[:]
    for idx, pol in enumerate(initial):
        if pol != 0:
            continue
        has_pos = any(nb in initial_pos for nb in neighbors[idx])
        has_neg = any(nb in initial_neg for nb in neighbors[idx])
        if has_pos and not has_neg:
            final[idx] = 1
        elif has_neg and not has_pos:
            final[idx] = -1
        else:
            final[idx] = 0

    return {
        "room_id": room_id,
        "rooms": rooms,
        "initial": initial,
        "final": final,
        "neighbors": neighbors,
        "wall_pairs": wall_pairs,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer room polarity from interior markers, then recolor the shared walls.

    Rooms are 4-connected black/red/yellow components enclosed by blue walls.
    Red-marked rooms are positive and yellow-marked rooms are negative. Unmarked
    rooms inherit polarity from marked neighbors in exactly one non-iterative
    round. Shared blue walls between opposite polarities become green; shared
    walls between equal non-neutral polarities become orange; any wall touching a
    neutral room stays blue. Markers and the corner legend are erased.
    """
    g = np.asarray(grid, dtype=int)
    _legend_cells(g)
    info = _analyze_rooms(g)
    final = info["final"]
    wall_pairs = info["wall_pairs"]

    out = g.copy()
    out[(out == RED) | (out == YELLOW) | (out == GREEN) | (out == ORANGE)] = BLACK

    for (r, c), pairs in wall_pairs.items():
        colors = {_wall_color(final[a], final[b]) for a, b in pairs}
        if len(colors) == 1:
            out[r, c] = next(iter(colors))
        else:
            out[r, c] = BLUE

    return out


def _pattern_value(token: str) -> int:
    if token == "R":
        return 1
    if token == "Y":
        return -1
    return 0


def _partition_total(total: int, count: int, min_each: int, max_each: int) -> list[int]:
    parts = [min_each] * count
    remaining = total - count * min_each
    capacity = [max_each - min_each] * count
    while remaining > 0:
        choices = [i for i, cap in enumerate(capacity) if cap > 0]
        if not choices:
            break
        idx = random.choice(choices)
        parts[idx] += 1
        capacity[idx] -= 1
        remaining -= 1
    random.shuffle(parts)
    return parts


def _draw_horizontal_chain(
    grid: np.ndarray, top: int, left: int, room_height: int, room_widths: list[int]
) -> list[tuple[int, int, int, int]]:
    frame_height = room_height + 2
    frame_width = sum(room_widths) + len(room_widths) + 1
    bottom = top + frame_height - 1
    right = left + frame_width - 1

    grid[top : bottom + 1, left] = BLUE
    grid[top : bottom + 1, right] = BLUE
    grid[top, left : right + 1] = BLUE
    grid[bottom, left : right + 1] = BLUE

    rooms: list[tuple[int, int, int, int]] = []
    cur = left + 1
    for idx, width in enumerate(room_widths):
        rooms.append((top + 1, cur, bottom - 1, cur + width - 1))
        cur += width
        if idx != len(room_widths) - 1:
            grid[top : bottom + 1, cur] = BLUE
            cur += 1
    return rooms


def _draw_vertical_chain(
    grid: np.ndarray, top: int, left: int, room_width: int, room_heights: list[int]
) -> list[tuple[int, int, int, int]]:
    frame_height = sum(room_heights) + len(room_heights) + 1
    frame_width = room_width + 2
    bottom = top + frame_height - 1
    right = left + frame_width - 1

    grid[top : bottom + 1, left] = BLUE
    grid[top : bottom + 1, right] = BLUE
    grid[top, left : right + 1] = BLUE
    grid[bottom, left : right + 1] = BLUE

    rooms: list[tuple[int, int, int, int]] = []
    cur = top + 1
    for idx, height in enumerate(room_heights):
        rooms.append((cur, left + 1, cur + height - 1, right - 1))
        cur += height
        if idx != len(room_heights) - 1:
            grid[cur, left : right + 1] = BLUE
            cur += 1
    return rooms


def _legend_margin(corner: str) -> tuple[int, int, int, int]:
    top = 1 + int(corner in ("tl", "tr"))
    bottom = 1 + int(corner in ("bl", "br"))
    left = 1 + int(corner in ("tl", "bl"))
    right = 1 + int(corner in ("tr", "br"))
    return top, bottom, left, right


def _place_legend(grid: np.ndarray, corner: str) -> None:
    h, w = grid.shape
    if corner == "tl":
        grid[0, 0] = GREEN
        grid[1, 0] = ORANGE
    elif corner == "tr":
        grid[0, w - 1] = GREEN
        grid[1, w - 1] = ORANGE
    elif corner == "bl":
        grid[h - 2, 0] = GREEN
        grid[h - 1, 0] = ORANGE
    else:
        grid[h - 2, w - 1] = GREEN
        grid[h - 1, w - 1] = ORANGE


def _scatter_markers(grid: np.ndarray, room: tuple[int, int, int, int], token: str) -> None:
    if token == "U":
        return
    color = RED if token == "R" else YELLOW
    r0, c0, r1, c1 = room
    cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    limit = min(3, max(1, len(cells) // 4 + 1))
    count = random.randint(1, limit)
    for r, c in random.sample(cells, count):
        grid[r, c] = color


def _validate_generated(grid: np.ndarray, pattern: dict[str, object]) -> bool:
    try:
        _legend_cells(grid)
        info = _analyze_rooms(grid)
        out = main(grid)
    except Exception:
        return False

    tokens = pattern["tokens"]
    initial = info["initial"]
    final = info["final"]
    neighbors = info["neighbors"]
    if len(initial) != len(tokens):
        return False
    for idx, token in enumerate(tokens):
        if initial[idx] != _pattern_value(token):
            return False

    unmarked_finals = {final[idx] for idx, token in enumerate(tokens) if token == "U"}
    if not pattern["required_unmarked_finals"].issubset(unmarked_finals):
        return False

    neutral_mode = pattern["neutral_mode"]
    if neutral_mode is not None:
        matched_mode = False
        for idx, token in enumerate(tokens):
            if token != "U" or final[idx] != 0:
                continue
            has_pos = any(initial[nb] == 1 for nb in neighbors[idx])
            has_neg = any(initial[nb] == -1 for nb in neighbors[idx])
            mode = "both" if has_pos and has_neg else "none" if (not has_pos and not has_neg) else "single"
            if mode == neutral_mode:
                matched_mode = True
                break
        if not matched_mode:
            return False

    colors = set(map(int, np.unique(out)))
    return GREEN in colors and ORANGE in colors and BLUE in colors and RED not in colors and YELLOW not in colors


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(16, 25)
    width = random.randint(16, 25)
    orientation = random.choice(("horizontal", "vertical"))
    corner = random.choice(_LEGEND_CORNERS)
    pattern = random.choice(_PATTERNS)
    tokens = pattern["tokens"]
    room_count = len(tokens)
    top_min, bottom_min, left_min, right_min = _legend_margin(corner)

    grid = np.full((height, width), BLACK, dtype=int)

    if orientation == "horizontal":
        max_frame_height = height - top_min - bottom_min
        max_frame_width = width - left_min - right_min
        if max_frame_height < 5:
            return None
        max_room_height = min(6, max_frame_height - 2)
        if max_room_height < 3:
            return None
        room_height = random.randint(3, max_room_height)
        max_total_width = max_frame_width - (room_count + 1)
        if max_total_width < 2 * room_count:
            return None
        total_width = random.randint(2 * room_count, min(4 * room_count, max_total_width))
        room_widths = _partition_total(total_width, room_count, 2, 4)
        frame_height = room_height + 2
        frame_width = total_width + room_count + 1
        extra_top = height - top_min - bottom_min - frame_height
        extra_left = width - left_min - right_min - frame_width
        top = top_min + random.randint(0, extra_top)
        left = left_min + random.randint(0, extra_left)
        rooms = _draw_horizontal_chain(grid, top, left, room_height, room_widths)
    else:
        max_frame_height = height - top_min - bottom_min
        max_frame_width = width - left_min - right_min
        if max_frame_width < 5:
            return None
        max_room_width = min(6, max_frame_width - 2)
        if max_room_width < 3:
            return None
        room_width = random.randint(3, max_room_width)
        max_total_height = max_frame_height - (room_count + 1)
        if max_total_height < 2 * room_count:
            return None
        total_height = random.randint(2 * room_count, min(4 * room_count, max_total_height))
        room_heights = _partition_total(total_height, room_count, 2, 4)
        frame_height = total_height + room_count + 1
        frame_width = room_width + 2
        extra_top = height - top_min - bottom_min - frame_height
        extra_left = width - left_min - right_min - frame_width
        top = top_min + random.randint(0, extra_top)
        left = left_min + random.randint(0, extra_left)
        rooms = _draw_vertical_chain(grid, top, left, room_width, room_heights)

    _place_legend(grid, corner)
    for room, token in zip(rooms, tokens, strict=True):
        _scatter_markers(grid, room, token)

    if _validate_generated(grid, pattern):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
