from __future__ import annotations

import heapq
import random
from collections import Counter

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
TEAL = 8

SOURCE_COLORS = (RED, GREEN, YELLOW, ORANGE)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
START_DIR = 4

BOX_H = 4
WORK_W_RANGE = (13, 15)
BOX_W_RANGE = (11, 14)
TOP_MARGIN_CHOICES = (0, 1)
BOTTOM_MARGIN_CHOICES = (0, 1)
GAP_CHOICES = (1, 2)
GENERATION_ATTEMPTS = 400
BEND_TRIPLETS = (
    (0, 1, 2),
    (0, 1, 3),
    (0, 2, 3),
    (1, 2, 3),
    (0, 0, 2),
    (0, 0, 3),
    (0, 1, 1),
    (1, 1, 3),
    (2, 2, 3),
    (0, 2, 2),
)


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _split_grid(grid: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    h, w = grid.shape
    assert h >= 3 and w >= 5, "grid is too small"
    grey_cols = [c for c in range(w) if np.all(grid[:, c] == GREY)]
    assert grey_cols, "missing grey separator"
    sep = next((c for c in grey_cols if w - c - 1 == 3), grey_cols[-1])
    assert w - sep - 1 >= 3, "output area must be at least three columns wide"
    return sep, grid[:, :sep], grid[:, sep + 1 : sep + 4]


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    mask = grid == int(color)
    visited = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(sorted(comp))
    comps.sort(key=lambda comp: comp[0])
    return comps


def _top_left(coords: list[tuple[int, int]]) -> tuple[int, int]:
    return min(coords, key=lambda rc: (rc[0], rc[1]))


def _best_route(
    work: np.ndarray,
    color: int,
    comp_a: list[tuple[int, int]],
    comp_b: list[tuple[int, int]],
) -> tuple[int, int, int]:
    allowed = (work == BLACK) | (work == int(color))
    h, w = work.shape
    inf = 10**9

    best_steps = np.full((h, w, 5), inf, dtype=int)
    best_bends = np.full((h, w, 5), inf, dtype=int)
    ways = np.zeros((h, w, 5), dtype=int)
    pq: list[tuple[int, int, int, int, int]] = []

    for r, c in comp_a:
        best_steps[r, c, START_DIR] = 0
        best_bends[r, c, START_DIR] = 0
        ways[r, c, START_DIR] = 1
        heapq.heappush(pq, (0, 0, r, c, START_DIR))

    while pq:
        steps, bends, r, c, prev_dir = heapq.heappop(pq)
        if steps != best_steps[r, c, prev_dir] or bends != best_bends[r, c, prev_dir]:
            continue

        for next_dir, (dr, dc) in enumerate(DIRS4):
            nr = r + dr
            nc = c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if not allowed[nr, nc]:
                continue
            nbends = bends + int(prev_dir != START_DIR and prev_dir != next_dir)
            nsteps = steps + 1
            better = (
                nsteps < best_steps[nr, nc, next_dir]
                or (
                    nsteps == best_steps[nr, nc, next_dir]
                    and nbends < best_bends[nr, nc, next_dir]
                )
            )
            equal = (
                nsteps == best_steps[nr, nc, next_dir]
                and nbends == best_bends[nr, nc, next_dir]
            )
            if better:
                best_steps[nr, nc, next_dir] = nsteps
                best_bends[nr, nc, next_dir] = nbends
                ways[nr, nc, next_dir] = ways[r, c, prev_dir]
                heapq.heappush(pq, (nsteps, nbends, nr, nc, next_dir))
            elif equal:
                ways[nr, nc, next_dir] = min(2, ways[nr, nc, next_dir] + ways[r, c, prev_dir])

    best_cost: tuple[int, int] | None = None
    count = 0
    for r, c in comp_b:
        for last_dir in range(4):
            steps = int(best_steps[r, c, last_dir])
            bends = int(best_bends[r, c, last_dir])
            if steps >= inf:
                continue
            cost = (steps, bends)
            if best_cost is None or cost < best_cost:
                best_cost = cost
                count = int(ways[r, c, last_dir])
            elif cost == best_cost:
                count = min(2, count + int(ways[r, c, last_dir]))

    assert best_cost is not None, f"color {color} has no connecting path"
    return best_cost[0], best_cost[1], count


def _analyze_scene(grid: np.ndarray) -> list[dict]:
    sep, work, _out_zone = _split_grid(grid)
    bg = get_bg(work)
    assert bg == BLACK, "work area background should be black"

    colors = [int(c) for c in np.unique(work) if int(c) not in (BLACK, GREY, TEAL)]
    assert len(colors) == 3, "expected exactly three source colors"

    infos: list[dict] = []
    for color in colors:
        comps = _components_of_color(work, color)
        assert len(comps) == 2, f"color {color} should have exactly two components"
        assert all(2 <= len(comp) <= 4 for comp in comps), f"color {color} has invalid component size"
        merged = [coord for comp in comps for coord in comp]
        steps, bends, count = _best_route(work, color, comps[0], comps[1])
        infos.append(
            {
                "color": color,
                "components": comps,
                "top_left": _top_left(merged),
                "steps": steps,
                "bends": bends,
                "optimal_path_count": count,
                "separator_col": sep,
            }
        )

    infos.sort(key=lambda info: info["top_left"])
    return infos


def _oracle_output(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    sep, _work, _out_zone = _split_grid(out)
    infos = _analyze_scene(out)
    out[:, sep + 1 : sep + 4] = BLACK
    for row, info in enumerate(infos):
        bends = int(info["bends"])
        if bends > 0:
            out[row, sep + 1 : sep + 1 + bends] = int(info["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Render each color's obstacle-forced connector bend count into the 3-column output strip.

    Input structure:
    - The grid is split into a left work area and a right 3-column output area,
      separated by a full-height grey vertical line.
    - The work area has black background, teal wall cells, and exactly three
      non-black non-teal colors chosen from red, green, yellow, and orange.
    - Each source color appears as exactly two disconnected 4-connected components.

    Rule:
    - For each source color, connect its two components by the shortest orthogonal
      path that may pass only through black or that same color.
    - Among shortest paths, use the one with the fewest direction changes, and
      count those bends.
    - Order the three colors by the row-major position of their topmost-leftmost
      pixel in the work area, then render one output row per color: paint the
      leftmost B cells of that row with the color, where B is its bend count.

    Key insight:
    - The visible teal walls encode an invisible routing problem whose path
      complexity is reduced to a small count and then redrawn as a compact row.
    """
    return _oracle_output(np.asarray(grid, dtype=int))


def _build_box(box_w: int, color: int, bends: int) -> tuple[np.ndarray, dict]:
    box = np.full((BOX_H, int(box_w)), BLACK, dtype=int)
    box[0, :] = TEAL
    box[-1, :] = TEAL
    box[:, 0] = TEAL
    box[:, -1] = TEAL

    for _ in range(60):
        left_len = random.randint(2, 4)
        right_len = random.randint(2, 4)
        start_anchor = left_len
        end_anchor = int(box_w) - 1 - right_len
        if end_anchor - start_anchor >= 2:
            break
    else:
        raise AssertionError("failed to size the box corridor")

    modes: dict[int, int] = {}
    mode_name = ""

    if bends == 0:
        start_row = end_row = random.choice((1, 2))
        for c in range(start_anchor, end_anchor + 1):
            modes[c] = start_row
        mode_name = "straight"
    elif bends == 1:
        start_row = random.choice((1, 2))
        end_row = 3 - start_row
        if random.random() < 0.5:
            modes[start_anchor] = 0
            for c in range(start_anchor + 1, end_anchor + 1):
                modes[c] = end_row
            mode_name = "vh"
        else:
            for c in range(start_anchor, end_anchor):
                modes[c] = start_row
            modes[end_anchor] = 0
            mode_name = "hv"
    elif bends == 2:
        if random.random() < 0.5:
            start_row = random.choice((1, 2))
            end_row = 3 - start_row
            pivot = random.randint(start_anchor + 1, end_anchor - 1)
            for c in range(start_anchor, pivot):
                modes[c] = start_row
            modes[pivot] = 0
            for c in range(pivot + 1, end_anchor + 1):
                modes[c] = end_row
            mode_name = "hvh"
        else:
            start_row = end_row = random.choice((1, 2))
            mid_row = 3 - start_row
            modes[start_anchor] = 0
            for c in range(start_anchor + 1, end_anchor):
                modes[c] = mid_row
            modes[end_anchor] = 0
            mode_name = "vhv"
    elif bends == 3:
        start_row = end_row = random.choice((1, 2))
        mid_row = 3 - start_row
        pivot = random.randint(start_anchor + 1, end_anchor - 1)
        if random.random() < 0.5:
            for c in range(start_anchor, pivot):
                modes[c] = start_row
            modes[pivot] = 0
            for c in range(pivot + 1, end_anchor):
                modes[c] = mid_row
            modes[end_anchor] = 0
            mode_name = "hvhv"
        else:
            modes[start_anchor] = 0
            for c in range(start_anchor + 1, pivot):
                modes[c] = mid_row
            modes[pivot] = 0
            for c in range(pivot + 1, end_anchor + 1):
                modes[c] = start_row
            mode_name = "vhvh"
    else:
        raise AssertionError("bend count must stay in 0..3")

    for c in range(1, start_anchor):
        modes[c] = start_row
    for c in range(end_anchor + 1, int(box_w) - 1):
        modes[c] = end_row

    for c, mode in modes.items():
        if mode == 0:
            continue
        if mode == 1:
            box[2, c] = TEAL
        elif mode == 2:
            box[1, c] = TEAL
        else:
            raise AssertionError("invalid corridor mode")

    box[start_row, 1 : start_anchor + 1] = int(color)
    box[end_row, end_anchor : int(box_w) - 1] = int(color)

    return box, {
        "color": int(color),
        "bends": int(bends),
        "left_len": int(left_len),
        "right_len": int(right_len),
        "mode": mode_name,
    }


def _expected_from_meta(grid: np.ndarray, metas: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    sep, _work, _out_zone = _split_grid(out)
    out[:, sep + 1 : sep + 4] = BLACK
    for row, meta in enumerate(metas):
        bends = int(meta["bends"])
        if bends > 0:
            out[row, sep + 1 : sep + 1 + bends] = int(meta["color"])
    return out


def _validate_sample(grid: np.ndarray, metas: list[dict]) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2 and max(grid.shape) <= 30
    sep, work, out_zone = _split_grid(grid)
    assert out_zone.shape == (grid.shape[0], 3)
    assert np.all(grid[:, sep] == GREY)
    assert np.all(out_zone == BLACK), "input output strip must start empty"

    infos = _analyze_scene(grid)
    assert [info["color"] for info in infos] == [meta["color"] for meta in metas]
    assert [info["bends"] for info in infos] == [meta["bends"] for meta in metas]
    assert all(info["optimal_path_count"] == 1 for info in infos), "paths must be uniquely determined"
    assert all(0 <= info["bends"] <= 3 for info in infos)
    assert len({info["bends"] for info in infos}) > 1, "bend counts may not all match"
    assert any(info["bends"] >= 2 for info in infos), "at least one color should require a real detour"

    no_wall = grid.copy()
    no_wall[:, :sep][no_wall[:, :sep] == TEAL] = BLACK
    no_wall_infos = _analyze_scene(no_wall)
    assert any(
        left["bends"] != right["bends"] or left["steps"] != right["steps"]
        for left, right in zip(infos, no_wall_infos)
    ), "walls must matter semantically"

    expected = _expected_from_meta(grid, metas)
    actual = main(grid)
    assert np.array_equal(actual, expected)
    assert np.array_equal(actual[:, :sep], grid[:, :sep]), "work area must remain unchanged"
    assert np.array_equal(actual[:, sep], grid[:, sep]), "separator must remain unchanged"
    for row, meta in enumerate(metas):
        expected_row = np.array(
            [meta["color"]] * int(meta["bends"]) + [BLACK] * (3 - int(meta["bends"])),
            dtype=int,
        )
        assert np.array_equal(actual[row, sep + 1 : sep + 4], expected_row)
    assert np.all(actual[len(metas) :, sep + 1 : sep + 4] == BLACK)


def _attempt_generate() -> tuple[np.ndarray, list[dict]]:
    work_w = random.randint(*WORK_W_RANGE)
    gaps = [random.choice(GAP_CHOICES), random.choice(GAP_CHOICES)]
    top = random.choice(TOP_MARGIN_CHOICES)
    bottom = random.choice(BOTTOM_MARGIN_CHOICES)
    work_h = top + BOX_H * 3 + sum(gaps) + bottom

    work = np.full((work_h, work_w), BLACK, dtype=int)
    colors = random.sample(SOURCE_COLORS, 3)
    bend_triplet = list(random.choice(BEND_TRIPLETS))

    metas: list[dict] = []
    r0 = top
    for idx, (color, bends) in enumerate(zip(colors, bend_triplet)):
        box_w = random.randint(BOX_W_RANGE[0], min(BOX_W_RANGE[1], work_w))
        c0 = random.randint(0, work_w - box_w)
        box, meta = _build_box(box_w, color, bends)
        work[r0 : r0 + BOX_H, c0 : c0 + box_w] = box
        meta["top_left"] = (r0 + 1, c0 + 1)
        metas.append(meta)
        r0 += BOX_H
        if idx < 2:
            r0 += gaps[idx]

    grid = np.full((work_h, work_w + 4), BLACK, dtype=int)
    grid[:, :work_w] = work
    grid[:, work_w] = GREY
    metas.sort(key=lambda meta: meta["top_left"])
    return grid, metas


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        try:
            grid, metas = _attempt_generate()
            _validate_sample(grid, metas)
            return grid
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid input")
