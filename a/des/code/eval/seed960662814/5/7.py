from __future__ import annotations

import random

import numpy as np


Coord = tuple[int, int]


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[Coord] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _bbox(coords: list[Coord]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _container_wall_color(grid: np.ndarray, bg: int) -> int:
    colors = [int(v) for v in np.unique(grid) if int(v) != bg]
    assert colors, "expected at least one foreground color"
    return max(colors, key=lambda color: int(np.sum(grid == color)))


def _parse_containers(grid: np.ndarray) -> tuple[int, list[dict[str, int]]]:
    bg = _get_bg(grid)
    wall_color = _container_wall_color(grid, bg)
    containers: list[dict[str, int]] = []

    for comp in _components(grid == wall_color):
        r0, c0, r1, c1 = _bbox(comp)
        expected = set()
        for r in range(r0, r1 + 1):
            expected.add((r, c0))
            expected.add((r, c1))
        for c in range(c0, c1 + 1):
            expected.add((r1, c))
        if set(comp) != expected:
            continue

        cavity = [(r, c) for r in range(r0, r1) for c in range(c0 + 1, c1)]
        marker_cells = [
            (r, c)
            for r, c in cavity
            if int(grid[r, c]) not in (bg, wall_color)
        ]
        assert len(marker_cells) == 1, "each container must contain exactly one marker"
        mr, mc = marker_cells[0]
        containers.append(
            {
                "top": int(r0),
                "left": int(c0),
                "bottom": int(r1),
                "right": int(c1),
                "marker_row": int(mr),
                "marker_col": int(mc),
                "marker_color": int(grid[mr, mc]),
            }
        )

    assert containers, "expected at least one U-shaped container"
    return wall_color, containers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each ball to the container whose bottom marker has the same color, then stack the
    balls vertically above that marker.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    wall_color, containers = _parse_containers(g)

    marker_positions = {
        (info["marker_row"], info["marker_col"]) for info in containers
    }
    by_color = {}
    for info in containers:
        color = int(info["marker_color"])
        assert color not in by_color, "marker colors must be unique"
        by_color[color] = info

    balls: list[tuple[int, int, int]] = []
    h, w = g.shape
    for r in range(h):
        for c in range(w):
            value = int(g[r, c])
            if value in (bg, wall_color):
                continue
            if (r, c) in marker_positions:
                continue
            balls.append((int(r), int(c), value))

    out = g.copy()
    for r, c, _ in balls:
        out[r, c] = bg

    balls_by_color: dict[int, list[tuple[int, int]]] = {}
    for r, c, color in balls:
        assert color in by_color, "every ball color must match a container marker"
        balls_by_color.setdefault(color, []).append((r, c))

    for color, info in by_color.items():
        ordered = sorted(balls_by_color.get(color, []), key=lambda rc: (rc[1], rc[0]))
        for i, _ in enumerate(ordered, start=1):
            rr = info["marker_row"] - i
            cc = info["marker_col"]
            assert rr >= info["top"], "stack must stay inside the container"
            assert out[rr, cc] == bg, "ball landing cell must be empty"
            out[rr, cc] = color

    return out


# --------------- generator ---------------

GRID_H_RANGE = (14, 22)
GRID_W_RANGE = (16, 28)
CONTAINER_COUNT_RANGE = (2, 4)
OUTER_WIDTH_RANGE = (3, 5)
INNER_DEPTH_RANGE = (3, 5)  # open cavity height
MAX_ATTEMPTS = 400


def _draw_container(
    grid: np.ndarray,
    *,
    top: int,
    left: int,
    outer_h: int,
    outer_w: int,
    wall_color: int,
    marker_color: int,
    marker_offset: int,
) -> dict[str, int]:
    right = left + outer_w - 1
    bottom = top + outer_h - 1
    grid[top : bottom + 1, left] = wall_color
    grid[top : bottom + 1, right] = wall_color
    grid[bottom, left : right + 1] = wall_color
    marker_col = left + 1 + marker_offset
    marker_row = bottom - 1
    grid[marker_row, marker_col] = marker_color
    return {
        "top": int(top),
        "left": int(left),
        "bottom": int(bottom),
        "right": int(right),
        "marker_row": int(marker_row),
        "marker_col": int(marker_col),
        "marker_color": int(marker_color),
    }


def _layout_columns(widths: list[int], grid_w: int) -> list[int]:
    n = len(widths)
    gap_sizes = [1] + [2] * (n - 1) + [1]
    slack = grid_w - (sum(widths) + sum(gap_sizes))
    assert slack >= 0
    for _ in range(slack):
        gap_sizes[random.randrange(len(gap_sizes))] += 1

    starts = []
    cur = gap_sizes[0]
    for idx, width in enumerate(widths):
        starts.append(int(cur))
        cur += width + gap_sizes[idx + 1]
    return starts


def _can_place_ball(candidate: Coord, occupied: set[Coord]) -> bool:
    r, c = candidate
    for orow, ocol in occupied:
        if abs(r - orow) <= 1 and abs(c - ocol) <= 1:
            return False
    return True


def _place_balls(
    grid: np.ndarray,
    containers: list[dict[str, int]],
    ball_counts: list[int],
) -> list[tuple[int, int, int, int]]:
    top_band_bottom = min(info["top"] for info in containers) - 2
    assert top_band_bottom >= 0
    grid_w = grid.shape[1]
    occupied: set[Coord] = set()
    placed: list[tuple[int, int, int, int]] = []

    requests = []
    for idx, count in enumerate(ball_counts):
        requests.extend([idx] * count)
    random.shuffle(requests)

    forced_remote = True
    remote_used = False

    for target_idx in requests:
        target = containers[target_idx]
        target_span = set(range(target["left"], target["right"] + 1))
        other_span = set()
        for idx, info in enumerate(containers):
            if idx == target_idx:
                continue
            other_span.update(range(info["left"], info["right"] + 1))
        wrong_cols = set(range(grid_w)) - target_span

        mode_cols: list[set[int]] = []
        if forced_remote and wrong_cols:
            if other_span - target_span:
                mode_cols.append(other_span - target_span)
            mode_cols.append(wrong_cols)
            mode_cols.append(set(range(grid_w)))
        elif random.random() < 0.7 and wrong_cols:
            if other_span - target_span and random.random() < 0.6:
                mode_cols.append(other_span - target_span)
            mode_cols.append(wrong_cols)
            mode_cols.append(set(range(grid_w)))
        else:
            mode_cols.append(set(range(grid_w)))
            if wrong_cols:
                mode_cols.append(wrong_cols)

        chosen: Coord | None = None
        for cols in mode_cols:
            candidates = [
                (r, c)
                for r in range(top_band_bottom + 1)
                for c in cols
                if _can_place_ball((r, c), occupied)
            ]
            random.shuffle(candidates)
            if candidates:
                chosen = candidates[0]
                break
        if chosen is None:
            raise AssertionError("failed to place balls with required spacing")

        row, col = chosen
        color = target["marker_color"]
        grid[row, col] = color
        occupied.add((row, col))
        placed.append((row, col, color, target_idx))
        if col not in target_span:
            remote_used = True
            forced_remote = False

    assert remote_used, "at least one ball should not start above its target container"
    return placed


def _build_expected(
    inp: np.ndarray,
    containers: list[dict[str, int]],
    balls: list[tuple[int, int, int, int]],
    bg: int,
) -> np.ndarray:
    out = inp.copy()
    for r, c, _color, _target_idx in balls:
        out[r, c] = bg

    by_target: dict[int, list[tuple[int, int, int]]] = {}
    for r, c, color, target_idx in balls:
        by_target.setdefault(target_idx, []).append((r, c, color))

    for target_idx, info in enumerate(containers):
        ordered = sorted(by_target.get(target_idx, []), key=lambda item: (item[1], item[0]))
        for level, (_r, _c, color) in enumerate(ordered, start=1):
            rr = info["marker_row"] - level
            out[rr, info["marker_col"]] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    container_count = random.randint(*CONTAINER_COUNT_RANGE)
    outer_widths = [random.randint(*OUTER_WIDTH_RANGE) for _ in range(container_count)]
    inner_depths = [random.randint(*INNER_DEPTH_RANGE) for _ in range(container_count)]
    outer_heights = [depth + 1 for depth in inner_depths]

    min_w = sum(outer_widths) + (container_count - 1) * 2 + 2
    if min_w > GRID_W_RANGE[1]:
        return None
    grid_w = random.randint(max(GRID_W_RANGE[0], min_w), GRID_W_RANGE[1])

    min_h = max(GRID_H_RANGE[0], max(outer_heights) + 6)
    if min_h > GRID_H_RANGE[1]:
        return None
    grid_h = random.randint(min_h, GRID_H_RANGE[1])

    colors = random.sample(list(range(10)), 2 + container_count)
    bg = colors[0]
    wall_color = colors[1]
    marker_colors = colors[2:]

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    starts = _layout_columns(outer_widths, grid_w)
    containers = []
    for outer_w, outer_h, start, marker_color in zip(
        outer_widths, outer_heights, starts, marker_colors
    ):
        top_lo = max(4, grid_h // 2 - 1)
        top_hi = grid_h - outer_h - 1
        if top_lo > top_hi:
            return None
        top = random.randint(top_lo, top_hi)
        marker_offset = random.randint(0, outer_w - 3)
        containers.append(
            _draw_container(
                grid,
                top=top,
                left=start,
                outer_h=outer_h,
                outer_w=outer_w,
                wall_color=wall_color,
                marker_color=marker_color,
                marker_offset=marker_offset,
            )
        )

    ball_counts = [random.randint(1, min(3, depth - 1)) for depth in inner_depths]
    growable = [idx for idx, depth in enumerate(inner_depths) if depth >= 4]
    if growable and max(ball_counts) == 1 and random.random() < 0.65:
        idx = random.choice(growable)
        ball_counts[idx] = random.randint(2, min(3, inner_depths[idx] - 1))

    try:
        balls = _place_balls(grid, containers, ball_counts)
    except AssertionError:
        return None

    expected = _build_expected(grid, containers, balls, bg)
    try:
        if np.array_equal(main(grid), expected):
            return grid
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid container-and-balls grid")
