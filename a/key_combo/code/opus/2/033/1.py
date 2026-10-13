from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

BG = 0
RED = 2
GREEN = 3
GRAY = 5

GRID_OPTIONS = (15, 17, 19)
GEN_ATTEMPTS = 300


@dataclass(frozen=True)
class RunSpec:
    color: int
    orientation: str
    line: int
    length: int


@dataclass
class Run:
    color: int
    coords: tuple[tuple[int, int], ...]
    orientation: str
    direction: tuple[int, int]
    front: tuple[int, int]
    length: int
    distance: float
    edge: str


def _components_of_color(grid: np.ndarray, color: int) -> list[tuple[tuple[int, int], ...]]:
    """4-connected monochrome components, following the (r, c) convention from arc_utils."""
    height, width = grid.shape
    seen: set[tuple[int, int]] = set()
    components: list[tuple[tuple[int, int], ...]] = []
    for r, c in map(tuple, np.argwhere(grid == color)):
        if (r, c) in seen:
            continue
        stack = [(int(r), int(c))]
        seen.add((int(r), int(c)))
        coords: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            coords.append((rr, cc))
            for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                if 0 <= nr < height and 0 <= nc < width and int(grid[nr, nc]) == int(color):
                    if (nr, nc) not in seen:
                        seen.add((nr, nc))
                        stack.append((nr, nc))
        components.append(tuple(sorted(coords)))
    components.sort()
    return components


def _reflect_barriers(grid: np.ndarray) -> set[tuple[int, int]]:
    height, width = grid.shape
    full: set[tuple[int, int]] = set()
    for r, c in map(tuple, np.argwhere(grid == GRAY)):
        for rr in (int(r), height - 1 - int(r)):
            for cc in (int(c), width - 1 - int(c)):
                full.add((rr, cc))
    return full


def _center_target(shape: tuple[int, int], direction: tuple[int, int]) -> int:
    height, width = shape
    dr, dc = direction
    if dc > 0:
        return (width - 1) // 2
    if dc < 0:
        return width // 2
    if dr > 0:
        return (height - 1) // 2
    return height // 2


def _front_for_spec(shape: tuple[int, int], spec: RunSpec) -> tuple[int, int]:
    height, width = shape
    if spec.orientation == "h":
        if spec.color == GREEN:
            return spec.line, spec.length - 1
        return spec.line, width - spec.length
    if spec.color == GREEN:
        return spec.length - 1, spec.line
    return height - spec.length, spec.line


def _distance_for_spec(shape: tuple[int, int], spec: RunSpec) -> float:
    center_r = (shape[0] - 1) / 2.0
    center_c = (shape[1] - 1) / 2.0
    r, c = _front_for_spec(shape, spec)
    return abs(r - center_r) + abs(c - center_c)


def _cells_for_spec(shape: tuple[int, int], spec: RunSpec) -> list[tuple[int, int]]:
    height, width = shape
    if spec.orientation == "h":
        if spec.color == GREEN:
            return [(spec.line, c) for c in range(spec.length)]
        return [(spec.line, c) for c in range(width - spec.length, width)]
    if spec.color == GREEN:
        return [(r, spec.line) for r in range(spec.length)]
    return [(r, spec.line) for r in range(height - spec.length, height)]


def _extract_runs(grid: np.ndarray) -> list[Run]:
    height, width = grid.shape
    center_r = (height - 1) / 2.0
    center_c = (width - 1) / 2.0
    runs: list[Run] = []

    for color in (RED, GREEN):
        for comp in _components_of_color(grid, color):
            rows = sorted({r for r, _ in comp})
            cols = sorted({c for _, c in comp})
            r0, r1 = rows[0], rows[-1]
            c0, c1 = cols[0], cols[-1]

            if len(rows) == 1:
                if len(comp) != c1 - c0 + 1:
                    raise AssertionError("horizontal run must be contiguous")
                orientation = "h"
                if color == GREEN:
                    if c0 != 0:
                        raise AssertionError("green horizontal runs must touch the left edge")
                    direction = (0, 1)
                    front = (r0, c1)
                    edge = "left"
                else:
                    if c1 != width - 1:
                        raise AssertionError("red horizontal runs must touch the right edge")
                    direction = (0, -1)
                    front = (r0, c0)
                    edge = "right"
            elif len(cols) == 1:
                if len(comp) != r1 - r0 + 1:
                    raise AssertionError("vertical run must be contiguous")
                orientation = "v"
                if color == GREEN:
                    if r0 != 0:
                        raise AssertionError("green vertical runs must touch the top edge")
                    direction = (1, 0)
                    front = (r1, c0)
                    edge = "top"
                else:
                    if r1 != height - 1:
                        raise AssertionError("red vertical runs must touch the bottom edge")
                    direction = (-1, 0)
                    front = (r0, c0)
                    edge = "bottom"
            else:
                raise AssertionError("every movable run must be a straight segment")

            distance = abs(front[0] - center_r) + abs(front[1] - center_c)
            runs.append(
                Run(
                    color=color,
                    coords=comp,
                    orientation=orientation,
                    direction=direction,
                    front=front,
                    length=len(comp),
                    distance=distance,
                    edge=edge,
                )
            )

    runs.sort(key=lambda run: (-run.distance, run.front[0], run.front[1], run.color))
    return runs


def _solve(grid: np.ndarray, *, return_trace: bool = False):
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    barriers = _reflect_barriers(g)

    out = np.full((height, width), BG, dtype=int)
    for r, c in barriers:
        out[r, c] = GRAY

    occupied: dict[tuple[int, int], str] = {coord: "gray" for coord in barriers}
    traces: list[dict] = []

    for run in _extract_runs(g):
        dr, dc = run.direction
        fr, fc = run.front
        target = _center_target(g.shape, run.direction)

        if run.orientation == "h":
            blockers = [
                (c, kind)
                for (r, c), kind in occupied.items()
                if r == fr and ((dc > 0 and c > fc) or (dc < 0 and c < fc))
            ]
            if dc > 0:
                if blockers:
                    blocker_c, kind = min(blockers, key=lambda item: item[0])
                    final_front = min(target, blocker_c - 1)
                    reason = kind if final_front == blocker_c - 1 else "center"
                else:
                    final_front = target
                    reason = "center"
                shift = final_front - fc
            else:
                if blockers:
                    blocker_c, kind = max(blockers, key=lambda item: item[0])
                    final_front = max(target, blocker_c + 1)
                    reason = kind if final_front == blocker_c + 1 else "center"
                else:
                    final_front = target
                    reason = "center"
                shift = fc - final_front
        else:
            blockers = [
                (r, kind)
                for (r, c), kind in occupied.items()
                if c == fc and ((dr > 0 and r > fr) or (dr < 0 and r < fr))
            ]
            if dr > 0:
                if blockers:
                    blocker_r, kind = min(blockers, key=lambda item: item[0])
                    final_front = min(target, blocker_r - 1)
                    reason = kind if final_front == blocker_r - 1 else "center"
                else:
                    final_front = target
                    reason = "center"
                shift = final_front - fr
            else:
                if blockers:
                    blocker_r, kind = max(blockers, key=lambda item: item[0])
                    final_front = max(target, blocker_r + 1)
                    reason = kind if final_front == blocker_r + 1 else "center"
                else:
                    final_front = target
                    reason = "center"
                shift = fr - final_front

        new_coords: list[tuple[int, int]] = []
        for r, c in run.coords:
            nr = int(r + dr * shift)
            nc = int(c + dc * shift)
            if (nr, nc) in occupied:
                raise AssertionError("moved runs may not overlap barriers or earlier runs")
            new_coords.append((nr, nc))

        for coord in new_coords:
            occupied[coord] = "run"
            out[coord] = run.color

        traces.append(
            {
                "run": run,
                "reason": reason,
                "shift": int(shift),
                "coords": tuple(new_coords),
            }
        )

    if return_trace:
        return out, traces
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the hidden symmetric obstacle field, then slide the edge runs inward until that field stops them.

    Input structure:
    - The grid uses black background, gray obstacle pixels, and red/green movable runs.
    - Only an upper-left subset of the gray obstacle field is visible in the input.
    - Each movable object is a 1-3 cell straight horizontal or vertical segment attached to one outer edge.

    Rule:
    - Reflect every visible gray obstacle across the horizontal axis, vertical axis, and both axes to reconstruct the full obstacle field.
    - Move green runs right or down and red runs left or up, always toward the grid center.
    - Process runs from the front pixel farthest from the center to the nearest; a run stops when its front becomes 4-adjacent to a gray obstacle, or to an already stopped run, or at the center line if nothing blocks it first.
    - Output the completed gray obstacle field together with all runs in their final positions.

    Key insight:
    - The visible obstacles are only one quadrant of a hidden biaxially symmetric stopper layout, so the slide endpoints are impossible to infer until that symmetry is completed first.
    """
    return _solve(np.asarray(grid, dtype=int), return_trace=False)


def _gen_visible_barriers(shape: tuple[int, int]) -> set[tuple[int, int]]:
    height, width = shape
    row_choices = list(range(2, height // 2))
    col_choices = list(range(2, width // 2))
    options = [
        (row_count, col_count)
        for row_count in range(2, min(4, len(row_choices)) + 1)
        for col_count in range(2, min(4, len(col_choices)) + 1)
        if row_count * col_count >= 5
    ]
    row_count, col_count = random.choice(options)
    rows = sorted(random.sample(row_choices, row_count))
    cols = sorted(random.sample(col_choices, col_count))

    n_pixels = random.randint(max(5, row_count + col_count - 1), min(8, row_count * col_count))
    visible: set[tuple[int, int]] = set()
    for row in rows:
        visible.add((row, random.choice(cols)))
    for col in cols:
        visible.add((random.choice(rows), col))

    leftovers = [(row, col) for row in rows for col in cols if (row, col) not in visible]
    random.shuffle(leftovers)
    while len(visible) < n_pixels and leftovers:
        visible.add(leftovers.pop())
    return visible


def _place_spec(grid: np.ndarray, spec: RunSpec) -> bool:
    cells = _cells_for_spec(grid.shape, spec)
    for r, c in cells:
        if int(grid[r, c]) != BG:
            return False
    for r, c in cells:
        grid[r, c] = spec.color
    return True


def _search_blocking_pair(
    base: np.ndarray, green_remaining: int, red_remaining: int
) -> tuple[RunSpec, RunSpec] | None:
    full = _reflect_barriers(base)
    candidate_rows = sorted({r for r, _ in full})
    candidate_cols = sorted({c for _, c in full})

    colors: list[int] = []
    if green_remaining > 0:
        colors.append(GREEN)
    if red_remaining > 0:
        colors.append(RED)

    best: tuple[tuple[int, int, float], RunSpec, RunSpec] | None = None
    for h_color in colors:
        for row in candidate_rows:
            for h_len in (1, 2, 3):
                horizontal = RunSpec(h_color, "h", row, h_len)
                for v_color in colors:
                    need_green = int(h_color == GREEN) + int(v_color == GREEN)
                    need_red = int(h_color == RED) + int(v_color == RED)
                    if need_green > green_remaining or need_red > red_remaining:
                        continue
                    for col in candidate_cols:
                        for v_len in (1, 2, 3):
                            vertical = RunSpec(v_color, "v", col, v_len)
                            h_dist = _distance_for_spec(base.shape, horizontal)
                            v_dist = _distance_for_spec(base.shape, vertical)
                            if h_dist == v_dist:
                                continue

                            trial = base.copy()
                            if not _place_spec(trial, horizontal):
                                continue
                            if not _place_spec(trial, vertical):
                                continue

                            try:
                                _out, trace = _solve(trial, return_trace=True)
                            except Exception:
                                continue
                            if len(trace) != 2:
                                continue
                            if any(item["shift"] <= 0 for item in trace):
                                continue
                            run_blocks = sum(item["reason"] == "run" for item in trace)
                            if run_blocks < 1:
                                continue
                            gray_blocks = sum(item["reason"] == "gray" for item in trace)
                            score = (run_blocks, gray_blocks, abs(h_dist - v_dist))
                            if best is None or score > best[0]:
                                best = (score, horizontal, vertical)

    if best is None:
        return None
    return best[1], best[2]


def _reserved_ok(spec: RunSpec, reserved_rows: set[int], reserved_cols: set[int]) -> bool:
    if spec.orientation == "h" and spec.line in reserved_rows:
        return False
    if spec.orientation == "v" and spec.line in reserved_cols:
        return False
    return True


def _add_random_specs(
    base: np.ndarray,
    seed_specs: list[RunSpec],
    green_remaining: int,
    red_remaining: int,
    reserved_rows: set[int],
    reserved_cols: set[int],
) -> tuple[np.ndarray, list[RunSpec]] | None:
    full = _reflect_barriers(base)
    candidate_rows = sorted({r for r, _ in full})
    candidate_cols = sorted({c for _, c in full})

    grid = base.copy()
    specs = list(seed_specs)
    used = {
        GREEN: {"h": set(), "v": set()},
        RED: {"h": set(), "v": set()},
    }
    existing_distances: set[float] = set()

    for spec in specs:
        if not _place_spec(grid, spec):
            return None
        used[spec.color][spec.orientation].add(spec.line)
        existing_distances.add(_distance_for_spec(base.shape, spec))

    pending_colors = [GREEN] * green_remaining + [RED] * red_remaining
    random.shuffle(pending_colors)

    for color in pending_colors:
        choices: list[RunSpec] = []
        for orientation in ("h", "v"):
            lines = candidate_rows if orientation == "h" else candidate_cols
            for line in lines:
                if line in used[color][orientation]:
                    continue
                for length in (1, 2, 3):
                    spec = RunSpec(color, orientation, line, length)
                    if not _reserved_ok(spec, reserved_rows, reserved_cols):
                        continue
                    dist = _distance_for_spec(base.shape, spec)
                    if dist in existing_distances:
                        continue
                    choices.append(spec)

        random.shuffle(choices)
        placed = False
        for spec in choices:
            trial = grid.copy()
            if _place_spec(trial, spec):
                grid = trial
                specs.append(spec)
                used[spec.color][spec.orientation].add(spec.line)
                existing_distances.add(_distance_for_spec(base.shape, spec))
                placed = True
                break
        if not placed:
            return None

    return grid, specs


def _validate_generated(grid: np.ndarray, *, require_run_block: bool) -> bool:
    try:
        _out, trace = _solve(grid, return_trace=True)
    except Exception:
        return False

    if any(item["shift"] <= 0 for item in trace):
        return False

    distances = [item["run"].distance for item in trace]
    if len(set(distances)) != len(distances):
        return False

    counts = {GREEN: 0, RED: 0}
    for item in trace:
        counts[item["run"].color] += 1
    if not (1 <= counts[GREEN] <= 3 and 1 <= counts[RED] <= 3):
        return False

    if require_run_block and not any(item["reason"] == "run" for item in trace):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    shape = (random.choice(GRID_OPTIONS), random.choice(GRID_OPTIONS))
    base = np.full(shape, BG, dtype=int)
    for r, c in _gen_visible_barriers(shape):
        base[r, c] = GRAY

    green_total = random.randint(1, 3)
    red_total = random.randint(1, 3)
    force_run_block = random.random() < 0.45

    seed_specs: list[RunSpec] = []
    reserved_rows: set[int] = set()
    reserved_cols: set[int] = set()

    if force_run_block:
        pair = _search_blocking_pair(base, green_total, red_total)
        if pair is None:
            return None
        horizontal, vertical = pair
        seed_specs = [horizontal, vertical]
        green_total -= int(horizontal.color == GREEN) + int(vertical.color == GREEN)
        red_total -= int(horizontal.color == RED) + int(vertical.color == RED)
        reserved_rows.add(horizontal.line)
        reserved_cols.add(vertical.line)

    added = _add_random_specs(
        base,
        seed_specs,
        green_total,
        red_total,
        reserved_rows,
        reserved_cols,
    )
    if added is None:
        return None
    grid, _specs = added

    if not _validate_generated(grid, require_run_block=force_run_block):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetric-slide input")
