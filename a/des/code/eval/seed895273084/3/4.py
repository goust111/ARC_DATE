from __future__ import annotations

import random
from bisect import bisect_left
from collections import Counter, defaultdict

import numpy as np


WALL = 5
DIRS = ("up", "down", "left", "right")
GEN_MAX_ATTEMPTS = 600
GEN_GROUP_MIN = 2
GEN_GROUP_MAX = 4
GEN_SIZE_MIN = 13
GEN_SIZE_MAX = 20
GEN_STEM_CHOICES = (1, 2)
GEN_SCATTER_EXTRA_MAX = 3
GEN_PALETTE = tuple(color for color in range(10) if color != WALL)


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_for_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    mask = grid == int(color)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(int(cr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _match_arrow(coords: list[tuple[int, int]]) -> str | None:
    if len(coords) not in (4, 5):
        return None

    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    rel = {(int(r - r0), int(c - c0)) for r, c in coords}

    if w == 3 and h in (2, 3):
        up = {(0, 0), (0, 1), (0, 2)} | {(rr, 1) for rr in range(1, h)}
        down = {(h - 1, 0), (h - 1, 1), (h - 1, 2)} | {(rr, 1) for rr in range(h - 1)}
        if rel == up:
            return "up"
        if rel == down:
            return "down"

    if h == 3 and w in (2, 3):
        left = {(0, 0), (1, 0), (2, 0)} | {(1, cc) for cc in range(1, w)}
        right = {(0, w - 1), (1, w - 1), (2, w - 1)} | {(1, cc) for cc in range(w - 1)}
        if rel == left:
            return "left"
        if rel == right:
            return "right"

    return None


def _find_arrows(grid: np.ndarray, bg: int) -> dict[int, dict[str, object]]:
    arrows: dict[int, dict[str, object]] = {}
    for color in sorted(int(x) for x in np.unique(grid)):
        if color in (int(bg), WALL):
            continue
        matches: list[tuple[str, set[tuple[int, int]]]] = []
        for comp in _components_for_color(grid, color):
            direction = _match_arrow(comp)
            if direction is not None:
                matches.append((direction, set(comp)))
        if len(matches) == 1:
            direction, coords = matches[0]
            arrows[int(color)] = {"direction": direction, "coords": coords}
    return arrows


def _project_positions(
    points: list[tuple[int, int]],
    direction: str,
    wall_mask: np.ndarray,
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = shape
    if not points:
        return []

    wall_rows = [np.flatnonzero(wall_mask[:, c]).tolist() for c in range(w)]
    wall_cols = [np.flatnonzero(wall_mask[r, :]).tolist() for r in range(h)]
    buckets: dict[tuple[int, int, int], int] = defaultdict(int)

    if direction in ("up", "down"):
        for r, c in points:
            barriers = wall_rows[int(c)]
            idx = bisect_left(barriers, int(r))
            lo = int(barriers[idx - 1]) if idx > 0 else -1
            hi = int(barriers[idx]) if idx < len(barriers) else h
            buckets[(int(c), lo, hi)] += 1

        out: list[tuple[int, int]] = []
        for (c, lo, hi), count in buckets.items():
            capacity = max(0, int(hi - lo - 1))
            count = min(int(count), capacity)
            if direction == "up":
                out.extend((int(lo + 1 + i), int(c)) for i in range(count))
            else:
                out.extend((int(hi - count + i), int(c)) for i in range(count))
        return out

    for r, c in points:
        barriers = wall_cols[int(r)]
        idx = bisect_left(barriers, int(c))
        lo = int(barriers[idx - 1]) if idx > 0 else -1
        hi = int(barriers[idx]) if idx < len(barriers) else w
        buckets[(int(r), lo, hi)] += 1

    out = []
    for (r, lo, hi), count in buckets.items():
        capacity = max(0, int(hi - lo - 1))
        count = min(int(count), capacity)
        if direction == "left":
            out.extend((int(r), int(lo + 1 + i)) for i in range(count))
        else:
            out.extend((int(r), int(hi - count + i)) for i in range(count))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each color's scattered pixels in the direction indicated by its matching T-arrow.

    Grey line segments are fixed walls. Every non-wall color has exactly one arrow component,
    and the remaining pixels of that color are the scattered particles to move. Particles slide
    until the nearest wall or grid edge and then compress into stacks within each free row/column
    segment. Arrow pixels are erased in the output.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)
    wall_mask = g == WALL
    arrows = _find_arrows(g, bg)

    out = np.full((h, w), int(bg), dtype=int)
    out[wall_mask] = WALL

    for color, info in arrows.items():
        arrow_cells = info["coords"]
        points = [
            (int(r), int(c))
            for r, c in np.argwhere(g == int(color))
            if (int(r), int(c)) not in arrow_cells
        ]
        for r, c in _project_positions(points, str(info["direction"]), wall_mask, g.shape):
            out[int(r), int(c)] = int(color)

    return out


def _expand(cells: set[tuple[int, int]], shape: tuple[int, int], radius: int = 1) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(int(r - radius), int(r + radius + 1)):
            for cc in range(int(c - radius), int(c + radius + 1)):
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((int(rr), int(cc)))
    return out


def _sample_spaced(values: list[int], count: int) -> list[int] | None:
    vals = list(values)
    random.shuffle(vals)
    chosen: list[int] = []
    for value in vals:
        if all(abs(int(value) - int(prev)) > 1 for prev in chosen):
            chosen.append(int(value))
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _split_count(total: int, bins: int) -> list[int]:
    counts = [1] * bins
    for _ in range(total - bins):
        counts[random.randrange(bins)] += 1
    random.shuffle(counts)
    return counts


def _arrow_cells(direction: str, stem_len: int, top: int, left: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    if direction == "up":
        cells.update((int(top), int(left + cc)) for cc in range(3))
        cells.update((int(top + rr), int(left + 1)) for rr in range(1, int(stem_len) + 1))
    elif direction == "down":
        cells.update((int(top + stem_len), int(left + cc)) for cc in range(3))
        cells.update((int(top + rr), int(left + 1)) for rr in range(int(stem_len)))
    elif direction == "left":
        cells.update((int(top + rr), int(left)) for rr in range(3))
        cells.update((int(top + 1), int(left + cc)) for cc in range(1, int(stem_len) + 1))
    else:
        cells.update((int(top + rr), int(left + stem_len)) for rr in range(3))
        cells.update((int(top + 1), int(left + cc)) for cc in range(int(stem_len)))
    return cells


def _arrow_dims(direction: str, stem_len: int) -> tuple[int, int]:
    if direction in ("up", "down"):
        return int(stem_len + 1), 3
    return 3, int(stem_len + 1)


def _candidate_ok(cells: set[tuple[int, int]], forbidden: set[tuple[int, int]]) -> bool:
    return not (cells & forbidden)


def _make_vertical_group(
    h: int,
    w: int,
    state: dict[str, set[int] | set[tuple[int, int]]],
    color: int,
    direction: str,
    use_wall: bool,
) -> dict[str, object] | None:
    assert direction in ("up", "down")
    forbidden = state["forbidden"]
    used_vertical_cols = state["used_vertical_cols"]
    used_horizontal_rows = state["used_horizontal_rows"]
    wall_cols = state["wall_cols"]
    wall_rows = state["wall_rows"]

    for _ in range(180):
        lane_count = random.randint(1, 2)
        total = random.randint(lane_count + 1, lane_count + GEN_SCATTER_EXTRA_MAX)
        candidate_cols = [
            c for c in range(1, w - 1) if c not in used_vertical_cols and c not in wall_cols
        ]
        cols = _sample_spaced(candidate_cols, lane_count)
        if cols is None:
            continue

        counts = _split_count(total, lane_count)
        max_count = max(counts)
        wall_cells: set[tuple[int, int]] = set()

        if use_wall:
            possible_rows: list[int] = []
            if direction == "up":
                for wr in range(2, h - 2):
                    if wr in wall_rows or wr in used_horizontal_rows:
                        continue
                    if h - wr - 3 < max_count:
                        continue
                    possible_rows.append(int(wr))
            else:
                for wr in range(1, h - 3):
                    if wr in wall_rows or wr in used_horizontal_rows:
                        continue
                    if wr - 2 < max_count:
                        continue
                    possible_rows.append(int(wr))
            if not possible_rows:
                continue
            wr = random.choice(possible_rows)
            wall_left = max(0, min(cols) - random.randint(1, 2))
            wall_right = min(w - 1, max(cols) + random.randint(1, 2))
            if any(c in used_vertical_cols for c in range(wall_left, wall_right + 1)):
                continue
            wall_cells = {(int(wr), int(c)) for c in range(int(wall_left), int(wall_right + 1))}
        else:
            wr = None

        scatter_cells: set[tuple[int, int]] = set()
        ok = True
        for col, count in zip(cols, counts):
            if direction == "up":
                low = int(wr + 3) if wr is not None else 2
                high = h - 2
            else:
                low = 1
                high = int(wr - 3) if wr is not None else h - 3
            rows = _sample_spaced(list(range(low, high + 1)), count) if low <= high else None
            if rows is None:
                ok = False
                break
            scatter_cells.update((int(r), int(col)) for r in rows)
        if not ok:
            continue

        pre_arrow = scatter_cells | wall_cells
        if not _candidate_ok(pre_arrow, forbidden):
            continue

        local_forbidden = _expand(pre_arrow, (h, w), radius=1)
        for _arrow_try in range(120):
            stem_len = random.choice(GEN_STEM_CHOICES)
            ah, aw = _arrow_dims(direction, stem_len)
            top = random.randint(0, h - ah)
            left = random.randint(0, w - aw)
            arrow_cells = _arrow_cells(direction, stem_len, top, left)
            if arrow_cells & local_forbidden:
                continue
            all_cells = pre_arrow | arrow_cells
            if not _candidate_ok(all_cells, forbidden):
                continue
            return {
                "color": int(color),
                "direction": direction,
                "use_wall": bool(use_wall),
                "scatter": scatter_cells,
                "wall": wall_cells,
                "arrow": arrow_cells,
                "cols": set(int(c) for c in cols),
            }
    return None


def _make_horizontal_group(
    h: int,
    w: int,
    state: dict[str, set[int] | set[tuple[int, int]]],
    color: int,
    direction: str,
    use_wall: bool,
) -> dict[str, object] | None:
    assert direction in ("left", "right")
    forbidden = state["forbidden"]
    used_vertical_cols = state["used_vertical_cols"]
    used_horizontal_rows = state["used_horizontal_rows"]
    wall_cols = state["wall_cols"]
    wall_rows = state["wall_rows"]

    for _ in range(180):
        lane_count = random.randint(1, 2)
        total = random.randint(lane_count + 1, lane_count + GEN_SCATTER_EXTRA_MAX)
        candidate_rows = [
            r for r in range(1, h - 1) if r not in used_horizontal_rows and r not in wall_rows
        ]
        rows = _sample_spaced(candidate_rows, lane_count)
        if rows is None:
            continue

        counts = _split_count(total, lane_count)
        max_count = max(counts)
        wall_cells: set[tuple[int, int]] = set()

        if use_wall:
            possible_cols: list[int] = []
            if direction == "left":
                for wc in range(2, w - 2):
                    if wc in wall_cols or wc in used_vertical_cols:
                        continue
                    if w - wc - 3 < max_count:
                        continue
                    possible_cols.append(int(wc))
            else:
                for wc in range(1, w - 3):
                    if wc in wall_cols or wc in used_vertical_cols:
                        continue
                    if wc - 2 < max_count:
                        continue
                    possible_cols.append(int(wc))
            if not possible_cols:
                continue
            wc = random.choice(possible_cols)
            wall_top = max(0, min(rows) - random.randint(1, 2))
            wall_bottom = min(h - 1, max(rows) + random.randint(1, 2))
            if any(r in used_horizontal_rows for r in range(wall_top, wall_bottom + 1)):
                continue
            wall_cells = {(int(r), int(wc)) for r in range(int(wall_top), int(wall_bottom + 1))}
        else:
            wc = None

        scatter_cells: set[tuple[int, int]] = set()
        ok = True
        for row, count in zip(rows, counts):
            if direction == "left":
                low = int(wc + 3) if wc is not None else 2
                high = w - 2
            else:
                low = 1
                high = int(wc - 3) if wc is not None else w - 3
            cols = _sample_spaced(list(range(low, high + 1)), count) if low <= high else None
            if cols is None:
                ok = False
                break
            scatter_cells.update((int(row), int(c)) for c in cols)
        if not ok:
            continue

        pre_arrow = scatter_cells | wall_cells
        if not _candidate_ok(pre_arrow, forbidden):
            continue

        local_forbidden = _expand(pre_arrow, (h, w), radius=1)
        for _arrow_try in range(120):
            stem_len = random.choice(GEN_STEM_CHOICES)
            ah, aw = _arrow_dims(direction, stem_len)
            top = random.randint(0, h - ah)
            left = random.randint(0, w - aw)
            arrow_cells = _arrow_cells(direction, stem_len, top, left)
            if arrow_cells & local_forbidden:
                continue
            all_cells = pre_arrow | arrow_cells
            if not _candidate_ok(all_cells, forbidden):
                continue
            return {
                "color": int(color),
                "direction": direction,
                "use_wall": bool(use_wall),
                "scatter": scatter_cells,
                "wall": wall_cells,
                "arrow": arrow_cells,
                "rows": set(int(r) for r in rows),
            }
    return None


def _validate_generated(grid: np.ndarray, groups: list[dict[str, object]]) -> bool:
    bg = _get_bg(grid)
    arrows = _find_arrows(grid, bg)
    colors = {int(group["color"]) for group in groups}
    if set(arrows) != colors:
        return False

    wall_mask = grid == WALL
    out = main(grid)
    if not np.all(out[wall_mask] == WALL):
        return False

    for group in groups:
        color = int(group["color"])
        scatter = list(group["scatter"])
        expected = set(_project_positions(scatter, str(group["direction"]), wall_mask, grid.shape))
        actual = {tuple(map(int, rc)) for rc in np.argwhere(out == color)}
        if actual != expected:
            return False
        if len(actual) != len(scatter):
            return False
        if any(int(out[r, c]) != int(bg) for r, c in group["arrow"]):
            return False
        if actual == set(scatter):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GEN_SIZE_MIN, GEN_SIZE_MAX)
    w = random.randint(GEN_SIZE_MIN, GEN_SIZE_MAX)
    bg = random.choice(GEN_PALETTE)
    palette = [color for color in GEN_PALETTE if color != bg]

    group_count = random.randint(GEN_GROUP_MIN, GEN_GROUP_MAX)
    directions = random.sample(DIRS, group_count)
    colors = random.sample(palette, group_count)

    wall_count = random.randint(1, group_count - 1)
    wall_flags = [True] * wall_count + [False] * (group_count - wall_count)
    random.shuffle(wall_flags)

    order = list(range(group_count))
    random.shuffle(order)

    state: dict[str, set[int] | set[tuple[int, int]]] = {
        "forbidden": set(),
        "used_vertical_cols": set(),
        "used_horizontal_rows": set(),
        "wall_cols": set(),
        "wall_rows": set(),
    }
    groups: list[dict[str, object]] = []

    for idx in order:
        direction = directions[idx]
        color = colors[idx]
        use_wall = wall_flags[idx]
        if direction in ("up", "down"):
            group = _make_vertical_group(h, w, state, color, direction, use_wall)
        else:
            group = _make_horizontal_group(h, w, state, color, direction, use_wall)
        if group is None:
            return None

        cells = set(group["scatter"]) | set(group["wall"]) | set(group["arrow"])
        state["forbidden"].update(_expand(cells, (h, w), radius=1))
        state["wall_cols"].update(int(c) for _, c in group["wall"])
        state["wall_rows"].update(int(r) for r, _ in group["wall"])
        if direction in ("up", "down"):
            state["used_vertical_cols"].update(int(c) for c in group["cols"])
        else:
            state["used_horizontal_rows"].update(int(r) for r in group["rows"])
        groups.append(group)

    grid = np.full((h, w), int(bg), dtype=int)
    for group in groups:
        for r, c in group["wall"]:
            grid[int(r), int(c)] = WALL
        for r, c in group["scatter"]:
            grid[int(r), int(c)] = int(group["color"])
        for r, c in group["arrow"]:
            grid[int(r), int(c)] = int(group["color"])

    if np.count_nonzero(grid == WALL) == 0:
        return None
    if not _validate_generated(grid, groups):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
