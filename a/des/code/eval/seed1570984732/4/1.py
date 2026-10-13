from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


WALL_COLOR = 5
MAROON_COLOR = 9
PALETTE = [0, 1, 2, 3, 4, 6, 7, 8]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 400
PLACEMENT_RETRIES = 60
SEGMENT_LENGTHS = [2, 3, 4, 5]


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _most_common_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(cnts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(int(rr), int(cc)):
                    if _in_bounds(h, w, nr, nc) and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _straight_orientation(comp: list[tuple[int, int]]) -> str | None:
    rows = {r for r, _ in comp}
    cols = {c for _, c in comp}
    if len(rows) == 1:
        want = list(range(min(cols), max(cols) + 1))
        if sorted(cols) == want:
            return "h"
    if len(cols) == 1:
        want = list(range(min(rows), max(rows) + 1))
        if sorted(rows) == want:
            return "v"
    return None


def _find_wall_color(grid: np.ndarray, bg: int) -> int:
    h, w = grid.shape
    candidates: list[int] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != int(bg)):
        comps = _components(grid == int(color))
        if not comps:
            continue
        ok = True
        has_segment = False
        for comp in comps:
            if len(comp) < 2:
                ok = False
                break
            if _straight_orientation(comp) is None:
                ok = False
                break
            if any(r in (0, h - 1) or c in (0, w - 1) for r, c in comp):
                ok = False
                break
            has_segment = True
        if ok and has_segment:
            candidates.append(int(color))
    assert len(candidates) == 1, "expected one wall color made of interior straight segments"
    return int(candidates[0])


def _range_to_opposite_edge(shape: tuple[int, int], seed: tuple[int, int], direction: tuple[int, int]) -> int:
    h, w = shape
    r, c = seed
    dr, dc = direction
    if (dr, dc) == (-1, 0):
        return int(r)
    if (dr, dc) == (1, 0):
        return int(h - 1 - r)
    if (dr, dc) == (0, -1):
        return int(c)
    assert (dr, dc) == (0, 1)
    return int(w - 1 - c)


def _seed_specs(grid: np.ndarray, bg: int, wall_color: int) -> list[dict]:
    h, w = grid.shape
    seeds: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in (int(bg), int(wall_color))):
        pts = np.argwhere(grid == int(color))
        assert pts.shape == (1, 2), "each seed color must appear exactly once"
        r, c = map(int, pts[0])
        wall_neighbors: list[tuple[int, int]] = []
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if _in_bounds(h, w, nr, nc) and int(grid[nr, nc]) == int(wall_color):
                wall_neighbors.append((int(dr), int(dc)))
        assert len(wall_neighbors) == 1, "each seed must touch exactly one wall side"
        wdr, wdc = wall_neighbors[0]
        direction = (-int(wdr), -int(wdc))
        seeds.append(
            {
                "color": int(color),
                "seed": (int(r), int(c)),
                "direction": (int(direction[0]), int(direction[1])),
                "steps": _range_to_opposite_edge(grid.shape, (int(r), int(c)), direction),
            }
        )
    assert seeds, "expected at least one seed"
    return seeds


def _intersection_color(grid: np.ndarray, bg: int, wall_color: int) -> int:
    used = {int(v) for v in np.unique(grid)}
    if int(MAROON_COLOR) not in used:
        return int(MAROON_COLOR)
    for color in range(10):
        if int(color) not in used and int(color) not in (int(bg), int(wall_color)):
            return int(color)
    raise AssertionError("no free color available for intersections")


def _simulate_seed(
    grid: np.ndarray,
    bg: int,
    wall_color: int,
    seed: tuple[int, int],
    direction: tuple[int, int],
    steps: int,
) -> tuple[list[tuple[int, int]], bool]:
    h, w = grid.shape
    r, c = seed
    dr, dc = direction
    visited: list[tuple[int, int]] = []
    bounced = False
    for _ in range(int(steps)):
        nr, nc = r + dr, c + dc
        if not _in_bounds(h, w, nr, nc) or int(grid[nr, nc]) == int(wall_color):
            bounced = True
            dr, dc = -dr, -dc
            nr, nc = r + dr, c + dc
            assert _in_bounds(h, w, nr, nc), "reflection must remain inside the grid"
            assert int(grid[nr, nc]) != int(wall_color), "reflected step must not enter a wall"
        r, c = int(nr), int(nc)
        if int(grid[r, c]) == int(bg):
            visited.append((int(r), int(c)))
    return visited, bounced


def _solve(grid: np.ndarray, return_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _most_common_color(g)
    wall_color = _find_wall_color(g, bg)
    seeds = _seed_specs(g, bg, wall_color)
    merge_color = _intersection_color(g, bg, wall_color)

    visits: dict[tuple[int, int], set[int]] = defaultdict(set)
    bounced_by_color: dict[int, bool] = {}
    for seed in seeds:
        visited, bounced = _simulate_seed(
            g,
            int(bg),
            int(wall_color),
            seed["seed"],
            seed["direction"],
            int(seed["steps"]),
        )
        bounced_by_color[int(seed["color"])] = bool(bounced)
        for cell in visited:
            visits[cell].add(int(seed["color"]))

    out = g.copy()
    for (r, c), colors in visits.items():
        out[int(r), int(c)] = int(merge_color if len(colors) >= 2 else next(iter(colors)))

    if return_meta:
        meta = {
            "bg": int(bg),
            "wall_color": int(wall_color),
            "merge_color": int(merge_color),
            "seed_count": int(len(seeds)),
            "maroon_count": int(sum(1 for colors in visits.values() if len(colors) >= 2)),
            "bounced_by_color": bounced_by_color,
        }
        return out, meta
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each seed's reflected ray path on background cells, turning overlaps into a third color.
    """
    return _solve(grid, return_meta=False)


def _spaced_choices(values: list[int], count: int, min_gap: int) -> list[int] | None:
    pool = values[:]
    random.shuffle(pool)
    chosen: list[int] = []
    for value in pool:
        if all(abs(int(value) - int(other)) >= int(min_gap) for other in chosen):
            chosen.append(int(value))
            if len(chosen) == int(count):
                return chosen
    return None


def _request_segment_cells(orientation: str, fixed: int, start: int, length: int) -> list[tuple[int, int]]:
    if orientation == "v":
        return [(int(r), int(fixed)) for r in range(int(start), int(start + length))]
    return [(int(fixed), int(c)) for c in range(int(start), int(start + length))]


def _place_segment(
    wall_mask: np.ndarray,
    orientation: str,
    fixed: int,
    through: int,
) -> bool:
    h, w = wall_mask.shape
    lengths = SEGMENT_LENGTHS[:]
    random.shuffle(lengths)
    for length in lengths:
        if orientation == "v":
            lo = max(1, int(through) - int(length) + 1)
            hi = min(int(through), int(h) - int(length) - 1)
        else:
            lo = max(1, int(through) - int(length) + 1)
            hi = min(int(through), int(w) - int(length) - 1)
        if hi < lo:
            continue
        starts = list(range(int(lo), int(hi) + 1))
        random.shuffle(starts)
        for start in starts:
            cells = _request_segment_cells(orientation, int(fixed), int(start), int(length))
            cell_set = set(cells)
            ok = True
            for r, c in cells:
                if not (0 < r < h - 1 and 0 < c < w - 1):
                    ok = False
                    break
                if bool(wall_mask[r, c]):
                    ok = False
                    break
                for nr, nc in _neighbors4(int(r), int(c)):
                    if (int(nr), int(nc)) in cell_set:
                        continue
                    if _in_bounds(h, w, nr, nc) and bool(wall_mask[nr, nc]):
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            for r, c in cells:
                wall_mask[r, c] = True
            return True
    return False


def _materialize_scene(
    h: int,
    w: int,
    bg: int,
    seeds: list[dict],
    wall_requests: list[tuple[str, int, int]],
) -> np.ndarray | None:
    for _ in range(PLACEMENT_RETRIES):
        wall_mask = np.zeros((h, w), dtype=bool)
        ok = True
        for orientation, fixed, through in wall_requests:
            if not _place_segment(wall_mask, orientation, int(fixed), int(through)):
                ok = False
                break
        if not ok:
            continue

        grid = np.full((h, w), int(bg), dtype=int)
        grid[wall_mask] = int(WALL_COLOR)

        used_seed_cells: set[tuple[int, int]] = set()
        for seed in seeds:
            r, c = int(seed["r"]), int(seed["c"])
            if not (0 < r < h - 1 and 0 < c < w - 1):
                ok = False
                break
            if bool(wall_mask[r, c]) or (r, c) in used_seed_cells:
                ok = False
                break
            used_seed_cells.add((r, c))
            grid[r, c] = int(seed["color"])
        if not ok:
            continue

        for seed in seeds:
            r, c = int(seed["r"]), int(seed["c"])
            wall_neighbors = 0
            for nr, nc in _neighbors4(r, c):
                if _in_bounds(h, w, nr, nc) and int(grid[nr, nc]) == int(WALL_COLOR):
                    wall_neighbors += 1
            if wall_neighbors != 1:
                ok = False
                break
        if ok:
            return grid
    return None


def _candidate_seed_columns(width: int, direction: int, cross_col: int | None = None) -> list[int]:
    if direction == 1:
        lo, hi = 2, width - 3
        if cross_col is not None:
            hi = min(int(hi), int(cross_col) - 2)
    else:
        lo, hi = 2, width - 3
        if cross_col is not None:
            lo = max(int(lo), int(cross_col) + 2)
    if hi < lo:
        return []
    return list(range(int(lo), int(hi) + 1))


def _candidate_seed_rows(height: int, direction: int, cross_row: int | None = None) -> list[int]:
    if direction == 1:
        lo, hi = 2, height - 3
        if cross_row is not None:
            hi = min(int(hi), int(cross_row) - 2)
    else:
        lo, hi = 2, height - 3
        if cross_row is not None:
            lo = max(int(lo), int(cross_row) + 2)
    if hi < lo:
        return []
    return list(range(int(lo), int(hi) + 1))


def _add_horizontal_seed(
    seeds: list[dict],
    wall_requests: list[tuple[str, int, int]],
    row: int,
    color: int,
    width: int,
    cross_col: int | None = None,
    force_bounce: bool = False,
) -> bool:
    direction = random.choice([-1, 1])
    cols = _candidate_seed_columns(width, direction, cross_col)
    if not cols:
        return False
    seed_col = random.choice(cols)
    seeds.append({"color": int(color), "r": int(row), "c": int(seed_col)})
    wall_requests.append(("v", int(seed_col - direction), int(row)))

    bounce_candidates: list[int] = []
    if direction == 1:
        lo = max(int(seed_col) + 2, 1 if cross_col is None else int(cross_col) + 1)
        hi = width - 2
        if lo <= hi:
            bounce_candidates = list(range(int(lo), int(hi) + 1))
    else:
        lo = 1
        hi = min(int(seed_col) - 2, width - 2 if cross_col is None else int(cross_col) - 1)
        if lo <= hi:
            bounce_candidates = list(range(int(lo), int(hi) + 1))

    want_bounce = bool(force_bounce or random.random() < 0.5)
    if want_bounce and bounce_candidates:
        wall_requests.append(("v", int(random.choice(bounce_candidates)), int(row)))
    elif force_bounce:
        return False
    return True


def _add_vertical_seed(
    seeds: list[dict],
    wall_requests: list[tuple[str, int, int]],
    col: int,
    color: int,
    height: int,
    cross_row: int | None = None,
    force_bounce: bool = False,
) -> bool:
    direction = random.choice([-1, 1])
    rows = _candidate_seed_rows(height, direction, cross_row)
    if not rows:
        return False
    seed_row = random.choice(rows)
    seeds.append({"color": int(color), "r": int(seed_row), "c": int(col)})
    wall_requests.append(("h", int(seed_row - direction), int(col)))

    bounce_candidates: list[int] = []
    if direction == 1:
        lo = max(int(seed_row) + 2, 1 if cross_row is None else int(cross_row) + 1)
        hi = height - 2
        if lo <= hi:
            bounce_candidates = list(range(int(lo), int(hi) + 1))
    else:
        lo = 1
        hi = min(int(seed_row) - 2, height - 2 if cross_row is None else int(cross_row) - 1)
        if lo <= hi:
            bounce_candidates = list(range(int(lo), int(hi) + 1))

    want_bounce = bool(force_bounce or random.random() < 0.5)
    if want_bounce and bounce_candidates:
        wall_requests.append(("h", int(random.choice(bounce_candidates)), int(col)))
    elif force_bounce:
        return False
    return True


def _build_cross_scene(height: int, width: int, colors: list[int], mixed: bool) -> np.ndarray | None:
    if height < 8 or width < 8:
        return None
    rows = list(range(2, height - 2))
    cols = list(range(2, width - 2))
    if not rows or not cols:
        return None

    cross_row = random.choice(rows)
    cross_col = random.choice(cols)
    seeds: list[dict] = []
    wall_requests: list[tuple[str, int, int]] = []

    if not _add_horizontal_seed(seeds, wall_requests, cross_row, colors[0], width, cross_col, force_bounce=False):
        return None
    if not _add_vertical_seed(seeds, wall_requests, cross_col, colors[1], height, cross_row, force_bounce=bool(mixed)):
        return None

    remaining = colors[2:]
    if mixed and remaining:
        extra_axis = random.choice(["h", "v"])
        if extra_axis == "h":
            choices = [r for r in rows if abs(int(r) - int(cross_row)) >= 2]
            if not choices:
                return None
            extra_row = random.choice(choices)
            if not _add_horizontal_seed(seeds, wall_requests, extra_row, remaining[0], width, None, force_bounce=True):
                return None
        else:
            choices = [c for c in cols if abs(int(c) - int(cross_col)) >= 2]
            if not choices:
                return None
            extra_col = random.choice(choices)
            if not _add_vertical_seed(seeds, wall_requests, extra_col, remaining[0], height, None, force_bounce=True):
                return None

    bg = random.choice([c for c in PALETTE if c not in colors])
    return _materialize_scene(height, width, bg, seeds, wall_requests)


def _build_parallel_scene(height: int, width: int, colors: list[int]) -> np.ndarray | None:
    axis = random.choice(["h", "v"])
    seed_count = random.randint(2, min(4, len(colors)))
    seeds: list[dict] = []
    wall_requests: list[tuple[str, int, int]] = []

    if axis == "h":
        lanes = _spaced_choices(list(range(2, height - 2)), seed_count, 2)
        if lanes is None:
            return None
        for row, color in zip(lanes, colors[:seed_count]):
            if not _add_horizontal_seed(seeds, wall_requests, int(row), int(color), width, None, force_bounce=False):
                return None
    else:
        lanes = _spaced_choices(list(range(2, width - 2)), seed_count, 2)
        if lanes is None:
            return None
        for col, color in zip(lanes, colors[:seed_count]):
            if not _add_vertical_seed(seeds, wall_requests, int(col), int(color), height, None, force_bounce=False):
                return None

    bg = random.choice([c for c in PALETTE if c not in colors[:seed_count]])
    return _materialize_scene(height, width, bg, seeds, wall_requests)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        height = random.randint(10, 17)
        width = random.randint(10, 17)
        scene_type = random.choice(["cross", "parallel", "mixed"])

        if scene_type == "parallel":
            seed_count = random.randint(2, 4)
        elif scene_type == "mixed":
            seed_count = 3
        else:
            seed_count = 2
        colors = random.sample(PALETTE, seed_count + 1)

        if scene_type == "parallel":
            grid = _build_parallel_scene(height, width, colors)
        elif scene_type == "mixed":
            grid = _build_cross_scene(height, width, colors, mixed=True)
        else:
            grid = _build_cross_scene(height, width, colors, mixed=False)
        if grid is None:
            continue

        out, meta = _solve(grid, return_meta=True)
        if np.array_equal(grid, out):
            continue
        if scene_type == "parallel" and int(meta["maroon_count"]) != 0:
            continue
        if scene_type in {"cross", "mixed"} and int(meta["maroon_count"]) == 0:
            continue
        if scene_type == "mixed" and not any(bool(v) for v in meta["bounced_by_color"].values()):
            continue
        return grid

    raise AssertionError("failed to generate a valid puzzle instance")
