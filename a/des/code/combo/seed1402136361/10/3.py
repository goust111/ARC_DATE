import random
from collections import Counter, deque

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 400
LEGEND_MAX_WEIGHT = 5
LEGEND_WIDTH = 1 + LEGEND_MAX_WEIGHT
SEPARATOR_WIDTH = 1
ROOM_GAP = 1
INNER_H_RANGE = (4, 8)
INNER_W_RANGE = (5, 10)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bg_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _parse_legend(grid: np.ndarray, bg: int) -> tuple[dict[int, int], int]:
    h, w = grid.shape
    candidates: list[tuple[int, int, int]] = []
    for r in range(h):
        if w < 2 or int(grid[r, 0]) == bg or int(grid[r, 1]) == bg:
            continue
        key = int(grid[r, 0])
        bar = int(grid[r, 1])
        if key == bar:
            continue
        c = 1
        while c < w and int(grid[r, c]) == bar:
            c += 1
        weight = c - 1
        if 1 <= weight <= LEGEND_MAX_WEIGHT:
            candidates.append((key, bar, weight))
    assert candidates, "expected left legend rows"

    bar_color = Counter(bar for _key, bar, _weight in candidates).most_common(1)[0][0]
    weights: dict[int, int] = {}
    legend_end = 0
    for key, bar, weight in candidates:
        if bar != bar_color:
            continue
        if key in weights:
            assert weights[key] == weight, "legend color weight must be unique"
        weights[key] = weight
        legend_end = max(legend_end, weight)
    assert weights, "failed to recover legend mapping"

    crop_col = None
    for c in range(legend_end + 1, w):
        if np.any(grid[:, c] != bg):
            crop_col = c
            break
    assert crop_col is not None, "failed to locate main area"
    return weights, crop_col


def _solve_main_area(main_grid: np.ndarray, bg: int, weights: dict[int, int]) -> np.ndarray:
    g = np.asarray(main_grid, dtype=int)
    non_bg = [int(v) for v in g.flat if int(v) != bg]
    assert non_bg, "main area must contain walls"
    wall = Counter(non_bg).most_common(1)[0][0]

    out = g.copy()
    h, w = g.shape
    for comp in _components(g == bg):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        marker_cells: set[tuple[int, int]] = set()
        first_pos: dict[int, tuple[int, int]] = {}
        for r, c in comp:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    value = int(g[nr, nc])
                    if value != bg and value != wall:
                        marker_cells.add((nr, nc))
        assert marker_cells, "each region must have boundary markers"

        counts: Counter[int] = Counter()
        for pos in sorted(marker_cells):
            color = int(g[pos])
            assert color in weights, "marker color must appear in legend"
            counts[color] += 1
            if color not in first_pos:
                first_pos[color] = pos

        best_color = None
        best_score = None
        best_pos = None
        for color, count in counts.items():
            score = (int(weights[color]), int(count))
            pos = first_pos[color]
            if (
                best_color is None
                or score > best_score
                or (score == best_score and pos < best_pos)
            ):
                best_color = color
                best_score = score
                best_pos = pos
        assert best_color is not None

        for r, c in comp:
            out[r, c] = int(best_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    weights, crop_col = _parse_legend(g, bg)
    main_grid = g[:, crop_col:]
    return _solve_main_area(main_grid, bg, weights)


def _sample_colors(rng: random.Random, n: int, forbidden: set[int]) -> list[int]:
    pool = [c for c in range(10) if c not in forbidden]
    rng.shuffle(pool)
    return pool[:n]


def _spaced_rows(rng: random.Random, height: int, n: int) -> list[int] | None:
    rows = list(range(height))
    rng.shuffle(rows)
    chosen: list[int] = []
    for row in sorted(rows):
        if all(abs(row - other) >= 2 for other in chosen):
            chosen.append(row)
            if len(chosen) == n:
                return sorted(chosen)
    return None


def _choose_weights(rng: random.Random, colors: list[int]) -> dict[int, int] | None:
    for _ in range(100):
        values = [rng.randint(1, LEGEND_MAX_WEIGHT) for _ in colors]
        if len(set(values)) < 2:
            continue
        groups = {}
        for color, value in zip(colors, values):
            groups.setdefault(value, []).append(color)
        if not any(len(group) >= 2 for group in groups.values()):
            continue
        return {color: value for color, value in zip(colors, values)}
    return None


def _room_boundary_cells(room: dict[str, int]) -> list[tuple[int, int]]:
    top = int(room["top"])
    left = int(room["left"])
    inner_h = int(room["inner_h"])
    inner_w = int(room["inner_w"])
    cells: list[tuple[int, int]] = []
    for c in range(left + 1, left + inner_w + 1):
        cells.append((top, c))
        cells.append((top + inner_h + 1, c))
    for r in range(top + 1, top + inner_h + 1):
        cells.append((r, left))
        cells.append((r, left + inner_w + 1))
    return cells


def _choose_layout(rng: random.Random, n_rooms: int) -> tuple[int, int]:
    if n_rooms == 2:
        return rng.choice([(1, 2), (2, 1)])
    if n_rooms == 3:
        return rng.choice([(1, 3), (3, 1)])
    return (2, 2)


def _build_rooms(rng: random.Random) -> tuple[np.ndarray, list[dict[str, int]]] | None:
    n_rooms = rng.randint(2, 4)
    rows, cols = _choose_layout(rng, n_rooms)
    rooms: list[dict[str, int]] = []
    outer_hs: list[int] = []
    outer_ws: list[int] = []
    for _ in range(n_rooms):
        inner_h = rng.randint(*INNER_H_RANGE)
        inner_w = rng.randint(*INNER_W_RANGE)
        rooms.append({"inner_h": inner_h, "inner_w": inner_w})
        outer_hs.append(inner_h + 2)
        outer_ws.append(inner_w + 2)

    row_heights = [0] * rows
    col_widths = [0] * cols
    for idx, room in enumerate(rooms):
        rr = idx // cols
        cc = idx % cols
        row_heights[rr] = max(row_heights[rr], int(room["inner_h"]) + 2)
        col_widths[cc] = max(col_widths[cc], int(room["inner_w"]) + 2)

    main_h = sum(row_heights) + ROOM_GAP * (rows - 1)
    main_w = sum(col_widths) + ROOM_GAP * (cols - 1)
    if main_h < 10 or main_w < 10 or main_h > 24 or main_w > 24:
        return None

    grid = np.zeros((main_h, main_w), dtype=int)
    row_starts = []
    cur = 0
    for height in row_heights:
        row_starts.append(cur)
        cur += height + ROOM_GAP
    col_starts = []
    cur = 0
    for width in col_widths:
        col_starts.append(cur)
        cur += width + ROOM_GAP

    for idx, room in enumerate(rooms):
        rr = idx // cols
        cc = idx % cols
        outer_h = int(room["inner_h"]) + 2
        outer_w = int(room["inner_w"]) + 2
        slack_r = row_heights[rr] - outer_h
        slack_c = col_widths[cc] - outer_w
        top = row_starts[rr] + (rng.randint(0, slack_r) if slack_r > 0 else 0)
        left = col_starts[cc] + (rng.randint(0, slack_c) if slack_c > 0 else 0)
        room["top"] = top
        room["left"] = left
        grid[top + 1 : top + outer_h - 1, left + 1 : left + outer_w - 1] = 1

    return grid, rooms


def _choose_tie_plan(
    rng: random.Random,
    weights: dict[int, int],
) -> tuple[dict[int, int], int] | None:
    groups: dict[int, list[int]] = {}
    for color, weight in weights.items():
        groups.setdefault(int(weight), []).append(int(color))
    tie_weights = [weight for weight, colors in groups.items() if len(colors) >= 2]
    if not tie_weights:
        return None
    weight = rng.choice(tie_weights)
    contenders = rng.sample(groups[weight], 2)
    winner = rng.choice(contenders)
    loser = contenders[1] if contenders[0] == winner else contenders[0]
    win_count = rng.randint(3, 5)
    lose_count = rng.randint(1, win_count - 1)
    plan = {winner: win_count, loser: lose_count}
    lower = [color for color, value in weights.items() if value < weight]
    if lower and rng.random() < 0.8:
        plan[rng.choice(lower)] = rng.randint(1, 2)
    return plan, winner


def _choose_weight_plan(
    rng: random.Random,
    weights: dict[int, int],
) -> tuple[dict[int, int], int] | None:
    pairs = [(hi, lo) for hi, wh in weights.items() for lo, wl in weights.items() if wh > wl]
    if not pairs:
        return None
    winner, loser = rng.choice(pairs)
    if rng.random() < 0.5:
        win_count = rng.randint(1, 2)
        lose_count = rng.randint(win_count + 1, win_count + 3)
    else:
        win_count = rng.randint(2, 4)
        lose_count = rng.randint(1, max(1, win_count - 1))
    plan = {winner: win_count, loser: lose_count}
    lower = [color for color, value in weights.items() if value < weights[winner] and color != loser]
    if lower and rng.random() < 0.8:
        plan[rng.choice(lower)] = rng.randint(1, 2)
    return plan, winner


def _paint_room_markers(
    rng: random.Random,
    grid: np.ndarray,
    room: dict[str, int],
    plan: dict[int, int],
    wall: int,
) -> bool:
    cells = _room_boundary_cells(room)
    if sum(plan.values()) > len(cells):
        return False
    rng.shuffle(cells)
    idx = 0
    for color, count in plan.items():
        for _ in range(count):
            if idx >= len(cells):
                return False
            r, c = cells[idx]
            idx += 1
            assert int(grid[r, c]) == wall
            grid[r, c] = int(color)
    return True


def _build_expected(main_grid: np.ndarray, rooms: list[dict[str, int]], winners: list[int]) -> np.ndarray:
    out = np.array(main_grid, copy=True)
    for room, winner in zip(rooms, winners):
        top = int(room["top"])
        left = int(room["left"])
        inner_h = int(room["inner_h"])
        inner_w = int(room["inner_w"])
        out[top + 1 : top + inner_h + 1, left + 1 : left + inner_w + 1] = int(winner)
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    room_data = _build_rooms(rng)
    if room_data is None:
        return None
    mask, rooms = room_data
    main_h, main_w = mask.shape

    base_forbidden = set()
    bg, wall, bar = _sample_colors(rng, 3, base_forbidden)
    n_colors = rng.randint(3, min(5, main_h // 2))
    legend_colors = _sample_colors(rng, n_colors, {bg, wall, bar})
    if len(legend_colors) != n_colors:
        return None
    weights = _choose_weights(rng, legend_colors)
    if weights is None:
        return None

    case_types = ["tie", "weight"]
    while len(case_types) < len(rooms):
        case_types.append(rng.choice(["tie", "weight"]))
    rng.shuffle(case_types[2:])

    main_grid = np.full((main_h, main_w), wall, dtype=int)
    main_grid[mask == 1] = bg
    winners: list[int] = []
    for room, case_type in zip(rooms, case_types):
        if case_type == "tie":
            choice = _choose_tie_plan(rng, weights)
        else:
            choice = _choose_weight_plan(rng, weights)
        if choice is None:
            return None
        plan, winner = choice
        if not _paint_room_markers(rng, main_grid, room, plan, wall):
            return None
        winners.append(winner)

    expected = _build_expected(main_grid, rooms, winners)
    if np.count_nonzero(main_grid == bg) <= np.count_nonzero(main_grid == wall):
        return None

    legend_rows = _spaced_rows(rng, main_h, len(legend_colors))
    if legend_rows is None:
        return None

    full_w = LEGEND_WIDTH + SEPARATOR_WIDTH + main_w
    grid = np.full((main_h, full_w), bg, dtype=int)
    main_start = LEGEND_WIDTH + SEPARATOR_WIDTH
    grid[:, main_start:] = main_grid
    for row, color in zip(legend_rows, legend_colors):
        weight = int(weights[color])
        grid[row, 0] = int(color)
        grid[row, 1 : 1 + weight] = int(bar)

    if np.count_nonzero(grid == bg) <= np.count_nonzero(grid == wall):
        return None
    solved = main(grid)
    if solved.shape != expected.shape or not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
