from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
GRAY = 5

GEN_ATTEMPTS = 800
GEN_SCENARIOS = [
    ("gray", "row"),
    ("gray", "col"),
    ("gray", "both"),
    ("gray", "mixed"),
    ("boundary", "row"),
    ("boundary", "col"),
    ("boundary", "mixed"),
]
GEN_SCENARIO_WEIGHTS = [2, 2, 4, 2, 1, 1, 1]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _empty_lines(grid: np.ndarray) -> tuple[set[int], set[int]]:
    h, w = grid.shape
    empty_rows = {r for r in range(h) if np.all(grid[r] == BLACK)}
    empty_cols = {c for c in range(w) if np.all(grid[:, c] == BLACK)}
    return empty_rows, empty_cols


def _find_seed(grid: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(grid == GREEN)
    assert coords.shape[0] == 1, "expected exactly one green seed"
    return tuple(map(int, coords[0]))


def _event_kind(event: dict[str, object]) -> str:
    row_hit = bool(event["row"])
    col_hit = bool(event["col"])
    if row_hit and col_hit:
        return "both"
    if row_hit:
        return "row"
    return "col"


def _profile_of(reflections: list[dict[str, object]], stop_index: int) -> str | None:
    kinds = [_event_kind(event) for event in reflections if int(event["index_before"]) <= int(stop_index)]
    if not kinds:
        return None
    unique = set(kinds)
    if unique == {"row"}:
        return "row"
    if unique == {"col"}:
        return "col"
    if unique == {"both"}:
        return "both"
    if unique == {"row", "col"}:
        return "mixed"
    return "other"


def _seed_far_enough(seed: tuple[int, int], cell: tuple[int, int]) -> bool:
    return abs(int(seed[0]) - int(cell[0])) + abs(int(seed[1]) - int(cell[1])) >= 3


def _trace_ray_on_lines(
    shape: tuple[int, int],
    seed: tuple[int, int],
    empty_rows: set[int],
    empty_cols: set[int],
    blockers: set[tuple[int, int]] | None = None,
) -> dict[str, object] | None:
    r, c = map(int, seed)
    dr = 1
    dc = 1
    blockers = set() if blockers is None else {tuple(map(int, rc)) for rc in blockers}

    path: list[tuple[int, int]] = []
    reflections: list[dict[str, object]] = []
    seen_states: set[tuple[int, int, int, int]] = set()
    seen_cells: set[tuple[int, int]] = {seed}

    while True:
        state = (r, c, dr, dc)
        if state in seen_states:
            return None
        seen_states.add(state)

        nr = r + dr
        nc = c + dc
        if not _in_bounds(shape, nr, nc):
            return {
                "path": path,
                "reflections": reflections,
                "stop": "boundary",
                "stop_cell": None,
            }

        hit_row = nr in empty_rows
        hit_col = nc in empty_cols
        if hit_row or hit_col:
            reflections.append(
                {
                    "from": (r, c),
                    "next": (nr, nc),
                    "row": hit_row,
                    "col": hit_col,
                    "index_before": len(path),
                }
            )
            if hit_row:
                dr = -dr
            if hit_col:
                dc = -dc
            continue

        if (nr, nc) in blockers:
            return {
                "path": path,
                "reflections": reflections,
                "stop": "gray",
                "stop_cell": (nr, nc),
            }

        if (nr, nc) in seen_cells:
            return None

        seen_cells.add((nr, nc))
        path.append((nr, nc))
        r, c = nr, nc


def _trace_input(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    empty_rows, empty_cols = _empty_lines(g)
    seed = _find_seed(g)
    blockers = {tuple(map(int, rc)) for rc in np.argwhere(g == GRAY)}
    trace = _trace_ray_on_lines(g.shape, seed, empty_rows, empty_cols, blockers)
    assert trace is not None, "valid inputs must not loop or self-intersect"
    return trace


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reveal blank rows and columns as red mirrors, then extend the green diagonal ray through them.

    Input structure:
    - A black background contains sparse gray foreground pixels and exactly one green seed pixel.
    - Some rows and columns are completely black: they contain neither gray nor the seed.
    - The seed lies on a non-blank row and a non-blank column.

    Rule:
    - Detect every completely black row and column in the input, then recolor all cells on those rows
      and columns red to make the implicit mirrors visible.
    - Starting from the green seed, march one step at a time in the down-right diagonal direction.
    - If the next cell is on a red mirror line, do not paint it; instead reflect by flipping the row
      direction for a blank row hit, the column direction for a blank column hit, or both at an
      empty-row/empty-column intersection. If the next cell is gray or out of bounds, stop. Otherwise
      paint that black cell green and continue.

    Key insight:
    - The mirrors are not explicit objects in the input: they emerge from the negative space formed by
      the fully blank rows and columns, and the ray must react to that inferred line structure.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    empty_rows, empty_cols = _empty_lines(g)
    for r in empty_rows:
        out[int(r), :] = RED
    for c in empty_cols:
        out[:, int(c)] = RED

    trace = _trace_input(g)
    for r, c in trace["path"]:
        out[int(r), int(c)] = GREEN
    return out


def _find_plan(
    h: int,
    w: int,
    empty_rows: set[int],
    empty_cols: set[int],
    stop_kind: str,
    profile: str,
) -> dict[str, object] | None:
    seeds = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if r not in empty_rows and c not in empty_cols
    ]
    random.shuffle(seeds)

    candidates: list[dict[str, object]] = []
    for seed in seeds:
        free_trace = _trace_ray_on_lines((h, w), seed, empty_rows, empty_cols, blockers=None)
        if free_trace is None or free_trace["stop"] != "boundary":
            continue

        path = list(free_trace["path"])
        reflections = list(free_trace["reflections"])
        if len(path) < 3 or not reflections:
            continue

        if stop_kind == "gray":
            for blocker_index, blocker in enumerate(path):
                if blocker_index < 2 or not _seed_far_enough(seed, blocker):
                    continue
                if _profile_of(reflections, blocker_index) == profile:
                    candidates.append(
                        {
                            "seed": seed,
                            "stop": stop_kind,
                            "profile": profile,
                            "blocker_index": blocker_index,
                            "free_trace": free_trace,
                        }
                    )
        else:
            if len(path) >= 4 and _profile_of(reflections, len(path)) == profile:
                candidates.append(
                    {
                        "seed": seed,
                        "stop": stop_kind,
                        "profile": profile,
                        "blocker_index": None,
                        "free_trace": free_trace,
                    }
                )

    if not candidates:
        return None
    return random.choice(candidates)


def _build_gray_cells(
    h: int,
    w: int,
    empty_rows: set[int],
    empty_cols: set[int],
    seed: tuple[int, int],
    stop_kind: str,
    free_trace: dict[str, object],
    blocker_index: int | None,
) -> set[tuple[int, int]] | None:
    free_path = list(free_trace["path"])
    if stop_kind == "gray":
        assert blocker_index is not None
        blocker = tuple(free_path[blocker_index])
        reserved_black = set(free_path[:blocker_index])
    else:
        blocker = None
        reserved_black = set(free_path)

    reserved_black.add(seed)

    def allowed(cell: tuple[int, int]) -> bool:
        r, c = cell
        return (
            r not in empty_rows
            and c not in empty_cols
            and cell not in reserved_black
            and _seed_far_enough(seed, cell)
        )

    gray_cells: set[tuple[int, int]] = set()
    if blocker is not None:
        if (
            blocker[0] in empty_rows
            or blocker[1] in empty_cols
            or not _seed_far_enough(seed, blocker)
        ):
            return None
        gray_cells.add(blocker)

    covered_rows = {r for r, _c in gray_cells}
    covered_cols = {c for _r, c in gray_cells}
    rows_needed = {r for r in range(h) if r not in empty_rows and r != seed[0]}
    cols_needed = {c for c in range(w) if c not in empty_cols and c != seed[1]}

    universe = [(r, c) for r in range(h) for c in range(w) if allowed((r, c))]
    if blocker is not None and blocker not in universe:
        universe.append(blocker)

    occupied = set(gray_cells)
    while not rows_needed <= covered_rows or not cols_needed <= covered_cols:
        best_gain = 0
        best_cells: list[tuple[int, int]] = []
        for cell in universe:
            if cell in occupied:
                continue
            r, c = cell
            gain = int(r in rows_needed and r not in covered_rows) + int(c in cols_needed and c not in covered_cols)
            if gain > best_gain:
                best_gain = gain
                best_cells = [cell]
            elif gain == best_gain and gain > 0:
                best_cells.append(cell)

        if best_gain == 0:
            return None

        cell = random.choice(best_cells)
        gray_cells.add(cell)
        occupied.add(cell)
        covered_rows.add(cell[0])
        covered_cols.add(cell[1])
        if len(gray_cells) > 20:
            return None

    extras = [cell for cell in universe if cell not in occupied]
    min_total = max(10, len(gray_cells))
    preferred_max = 18 if len(gray_cells) <= 18 else 20
    max_total = min(20, max(preferred_max, min_total), len(gray_cells) + len(extras))
    if min_total > max_total:
        return None

    target_total = random.randint(min_total, max_total)
    random.shuffle(extras)
    gray_cells.update(extras[: target_total - len(gray_cells)])
    return gray_cells


def _metadata_output(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    empty_rows: set[int],
    empty_cols: set[int],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for r in empty_rows:
        out[int(r), :] = RED
    for c in empty_cols:
        out[:, int(c)] = RED
    for r, c in path:
        out[int(r), int(c)] = GREEN
    return out


def _validate_instance(
    grid: np.ndarray,
    plan: dict[str, object],
    empty_rows: set[int],
    empty_cols: set[int],
) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if np.count_nonzero(grid == GREEN) != 1:
        return False

    seed = _find_seed(grid)
    gray_cells = [tuple(map(int, rc)) for rc in np.argwhere(grid == GRAY)]
    if not 10 <= len(gray_cells) <= 20:
        return False
    if not all(_seed_far_enough(seed, cell) for cell in gray_cells):
        return False

    actual_empty_rows, actual_empty_cols = _empty_lines(grid)
    if actual_empty_rows != empty_rows or actual_empty_cols != empty_cols:
        return False
    if seed[0] in empty_rows or seed[1] in empty_cols:
        return False

    free_trace = dict(plan["free_trace"])
    stop_kind = str(plan["stop"])
    blocker_index = plan["blocker_index"]
    if stop_kind == "gray":
        assert blocker_index is not None
        expected_path = list(free_trace["path"][:blocker_index])
        expected_stop_cell = tuple(free_trace["path"][blocker_index])
        stop_index = int(blocker_index)
    else:
        expected_path = list(free_trace["path"])
        expected_stop_cell = None
        stop_index = len(expected_path)

    actual_trace = _trace_input(grid)
    if actual_trace["stop"] != stop_kind:
        return False
    if list(actual_trace["path"]) != expected_path:
        return False
    if actual_trace["stop_cell"] != expected_stop_cell:
        return False
    if _profile_of(list(actual_trace["reflections"]), stop_index) != str(plan["profile"]):
        return False
    if not list(actual_trace["reflections"]):
        return False

    expected_out = _metadata_output(grid, expected_path, empty_rows, empty_cols)
    return np.array_equal(main(grid), expected_out)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        h = random.randint(11, 15)
        w = random.randint(11, 15)
        empty_rows = set(random.sample(range(1, h - 1), random.randint(2, 3)))
        empty_cols = set(random.sample(range(1, w - 1), random.randint(2, 3)))

        stop_kind, profile = random.choices(
            GEN_SCENARIOS,
            weights=GEN_SCENARIO_WEIGHTS,
            k=1,
        )[0]
        plan = _find_plan(h, w, empty_rows, empty_cols, stop_kind, profile)
        if plan is None:
            continue

        seed = tuple(plan["seed"])
        gray_cells = _build_gray_cells(
            h,
            w,
            empty_rows,
            empty_cols,
            seed,
            stop_kind,
            dict(plan["free_trace"]),
            plan["blocker_index"],
        )
        if gray_cells is None:
            continue

        grid = np.zeros((h, w), dtype=int)
        for r, c in gray_cells:
            grid[int(r), int(c)] = GRAY
        grid[int(seed[0]), int(seed[1])] = GREEN

        if _validate_instance(grid, plan, empty_rows, empty_cols):
            return grid

    raise RuntimeError("failed to generate a valid ray-mirror puzzle instance")
