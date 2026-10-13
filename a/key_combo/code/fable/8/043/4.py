from __future__ import annotations

import random
from collections import Counter

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    GREY = 5
    ORANGE = 7
    TEAL = 8


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def _runs_of_color(line: np.ndarray, color: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    i = 0
    while i < len(line):
        if int(line[i]) != int(color):
            i += 1
            continue
        j = i
        while j + 1 < len(line) and int(line[j + 1]) == int(color):
            j += 1
        runs.append((int(i), int(j)))
        i = j + 1
    return runs


def _line_task(line: np.ndarray, fixed: int, axis: str) -> dict | None:
    blues = [int(x) for x in np.flatnonzero(line == Color.BLUE)]
    grey_runs = _runs_of_color(line, Color.GREY)
    if len(blues) != 2 or len(grey_runs) != 1:
        return None

    marker0, marker1 = blues
    bar0, bar1 = grey_runs[0]
    if not np.all(line[marker0 + 1 : marker1] == Color.BLACK):
        return None
    if bar1 < marker0:
        if bar1 == marker0 - 1:
            return None
        side = "left" if axis == "row" else "top"
    elif bar0 > marker1:
        if bar0 == marker1 + 1:
            return None
        side = "right" if axis == "row" else "bottom"
    else:
        return None

    gap = int(marker1 - marker0 - 1)
    charge = int(bar1 - bar0 + 1)
    return {
        "axis": axis,
        "fixed": int(fixed),
        "marker0": int(marker0),
        "marker1": int(marker1),
        "bar0": int(bar0),
        "bar1": int(bar1),
        "gap": gap,
        "charge": charge,
        "side": side,
    }


def _find_tasks(grid: np.ndarray) -> list[dict]:
    tasks: list[dict] = []
    h, w = map(int, grid.shape)
    for r in range(h):
        task = _line_task(grid[r, :], r, "row")
        if task is not None:
            tasks.append(task)
    for c in range(w):
        task = _line_task(grid[:, c], c, "col")
        if task is not None:
            tasks.append(task)
    return tasks


def _apply_task(out: np.ndarray, src: np.ndarray, task: dict) -> None:
    if task["gap"] > task["charge"]:
        return

    gap = int(task["gap"])
    if task["axis"] == "row":
        r = int(task["fixed"])
        c0 = int(task["marker0"])
        c1 = int(task["marker1"])
        assert np.all(src[r, c0 + 1 : c1] == Color.BLACK)
        out[r, c0 + 1 : c1] = Color.TEAL
        if task["side"] == "left" and gap:
            out[r, task["bar1"] - gap + 1 : task["bar1"] + 1] = Color.ORANGE
        if task["side"] == "right" and gap:
            out[r, task["bar0"] : task["bar0"] + gap] = Color.ORANGE
    else:
        c = int(task["fixed"])
        r0 = int(task["marker0"])
        r1 = int(task["marker1"])
        assert np.all(src[r0 + 1 : r1, c] == Color.BLACK)
        out[r0 + 1 : r1, c] = Color.TEAL
        if task["side"] == "top" and gap:
            out[task["bar1"] - gap + 1 : task["bar1"] + 1, c] = Color.ORANGE
        if task["side"] == "bottom" and gap:
            out[task["bar0"] : task["bar0"] + gap, c] = Color.ORANGE


def _render_expected(grid: np.ndarray, tasks: list[dict]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for task in tasks:
        _apply_task(out, grid, task)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect blue marker pairs only when their same-axis battery bar can pay the full distance.

    Input structure:
    - The background is black.
    - Some rows or columns contain exactly one pair of blue marker pixels.
    - The same row or column also contains one contiguous gray battery bar that is separated from the markers and lies entirely on one side of the pair.

    Rule:
    - For each valid same-row or same-column pair, count the black cells strictly between the two blue markers and call that gap g.
    - Let n be the length of that line's gray battery bar.
    - If g <= n, fill the entire interior gap with teal and recolor exactly g battery cells from the bar end nearest the markers to orange.
    - If g > n, leave that pair unchanged; all other pixels always stay as they are.

    Key insight:
    - The battery does not merely signal a connection; its visible gray length gates whether the line exists, and the orange/gray split records how much energy was spent.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"
    assert _get_bg(g) == Color.BLACK, "expected black background"

    tasks = _find_tasks(g)
    assert tasks, "expected at least one energy-gated pair"
    return _render_expected(g, tasks)


GRID_MIN = 13
GRID_MAX = 24
TOTAL_TASK_OPTIONS = ((1, 2), (2, 1), (2, 2), (1, 3), (3, 1), (2, 3), (3, 2))
MIN_GAP = 2
MAX_GAP = 7
MAX_CHARGE = 9
MAX_ATTEMPTS = 500


def _sample_spaced_positions(start: int, stop: int, count: int, min_gap: int) -> list[int] | None:
    values = list(range(start, stop))
    for _ in range(200):
        chosen = sorted(random.sample(values, count))
        if all(b - a >= min_gap for a, b in zip(chosen, chosen[1:])):
            return chosen
    return None


def _sample_success_flags(count: int) -> list[bool]:
    flags = [True, False] + [bool(random.getrandbits(1)) for _ in range(count - 2)]
    random.shuffle(flags)
    return flags


def _sample_sides(axis: str, count: int) -> list[str]:
    if axis == "row":
        pool = ["left", "right"]
    else:
        pool = ["top", "bottom"]
    if count == 1:
        return [random.choice(pool)]
    sides = [pool[0], pool[1]]
    while len(sides) < count:
        sides.append(random.choice(pool))
    random.shuffle(sides)
    return sides


def _sample_gap_and_charge(success: bool) -> tuple[int, int]:
    gap = random.randint(MIN_GAP, MAX_GAP)
    if success:
        charge = random.randint(gap, min(MAX_CHARGE, gap + 3))
    else:
        charge = random.randint(1, gap - 1)
    return gap, charge


def _horizontal_candidates(
    width: int,
    v_cols: set[int],
    gap: int,
    charge: int,
    side: str,
) -> list[dict]:
    candidates: list[dict] = []
    max_start = width - gap - charge - 3
    for start in range(max_start + 1):
        if side == "left":
            bar0 = start
            bar1 = start + charge - 1
            marker0 = bar1 + 2
            marker1 = marker0 + gap + 1
        else:
            marker0 = start
            marker1 = marker0 + gap + 1
            bar0 = marker1 + 2
            bar1 = bar0 + charge - 1
        occupied_cols = {marker0, marker1, *range(bar0, bar1 + 1)}
        if occupied_cols & v_cols:
            continue
        interior = set(range(marker0 + 1, marker1))
        candidates.append(
            {
                "marker0": marker0,
                "marker1": marker1,
                "bar0": bar0,
                "bar1": bar1,
                "cross_cols": sorted(interior & v_cols),
            }
        )
    random.shuffle(candidates)
    return candidates


def _vertical_candidates(
    height: int,
    h_rows: set[int],
    gap: int,
    charge: int,
    side: str,
) -> list[dict]:
    candidates: list[dict] = []
    max_start = height - gap - charge - 3
    for start in range(max_start + 1):
        if side == "top":
            bar0 = start
            bar1 = start + charge - 1
            marker0 = bar1 + 2
            marker1 = marker0 + gap + 1
        else:
            marker0 = start
            marker1 = marker0 + gap + 1
            bar0 = marker1 + 2
            bar1 = bar0 + charge - 1
        occupied_rows = {marker0, marker1, *range(bar0, bar1 + 1)}
        if occupied_rows & h_rows:
            continue
        interior = set(range(marker0 + 1, marker1))
        candidates.append(
            {
                "marker0": marker0,
                "marker1": marker1,
                "bar0": bar0,
                "bar1": bar1,
                "cross_rows": sorted(interior & h_rows),
            }
        )
    random.shuffle(candidates)
    return candidates


def _normalized_task(task: dict) -> tuple:
    return (
        task["axis"],
        int(task["fixed"]),
        int(task["marker0"]),
        int(task["marker1"]),
        int(task["bar0"]),
        int(task["bar1"]),
        int(task["gap"]),
        int(task["charge"]),
        task["side"],
    )


def _crossing_count(tasks: list[dict]) -> int:
    rows = [t for t in tasks if t["axis"] == "row"]
    cols = [t for t in tasks if t["axis"] == "col"]
    count = 0
    for row_task in rows:
        for col_task in cols:
            if row_task["marker0"] < col_task["fixed"] < row_task["marker1"] and col_task["marker0"] < row_task["fixed"] < col_task["marker1"]:
                count += 1
    return count


def _validate_semantics(inp: np.ndarray, tasks: list[dict]) -> bool:
    parsed = _find_tasks(inp)
    if sorted(map(_normalized_task, parsed)) != sorted(map(_normalized_task, tasks)):
        return False

    out_expected = _render_expected(inp, tasks)
    out_actual = main(inp)
    if not np.array_equal(out_actual, out_expected):
        return False
    if np.array_equal(inp, out_expected):
        return False

    rows = sum(t["axis"] == "row" for t in tasks)
    cols = len(tasks) - rows
    if rows < 1 or cols < 1:
        return False
    if not any(t["gap"] <= t["charge"] for t in tasks):
        return False
    if not any(t["gap"] > t["charge"] for t in tasks):
        return False

    crossings = _crossing_count(tasks)
    if crossings < 1 or crossings > 4:
        return False

    for task in tasks:
        if task["axis"] == "row":
            line_in = inp[task["fixed"], :]
            line_out = out_expected[task["fixed"], :]
        else:
            line_in = inp[:, task["fixed"]]
            line_out = out_expected[:, task["fixed"]]

        if not np.all(line_in[task["marker0"] + 1 : task["marker1"]] == Color.BLACK):
            return False
        if task["gap"] <= task["charge"]:
            if not np.all(line_out[task["marker0"] + 1 : task["marker1"]] == Color.TEAL):
                return False
            if task["side"] in ("left", "top"):
                spent = line_out[task["bar1"] - task["gap"] + 1 : task["bar1"] + 1]
                left_over = line_out[task["bar0"] : task["bar1"] - task["gap"] + 1]
            else:
                spent = line_out[task["bar0"] : task["bar0"] + task["gap"]]
                left_over = line_out[task["bar0"] + task["gap"] : task["bar1"] + 1]
            if task["gap"] and not np.all(spent == Color.ORANGE):
                return False
            if left_over.size and not np.all(left_over == Color.GREY):
                return False
        else:
            if not np.array_equal(line_in, line_out):
                return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    row_count, col_count = random.choice(TOTAL_TASK_OPTIONS)

    h_rows = _sample_spaced_positions(2, height - 2, row_count, 3)
    v_cols = _sample_spaced_positions(2, width - 2, col_count, 3)
    if h_rows is None or v_cols is None:
        return None

    flags = _sample_success_flags(row_count + col_count)
    row_sides = _sample_sides("row", row_count)
    col_sides = _sample_sides("col", col_count)

    tasks: list[dict] = []
    grid = np.full((height, width), Color.BLACK, dtype=int)

    for r, side, success in zip(h_rows, row_sides, flags[:row_count]):
        gap, charge = _sample_gap_and_charge(success)
        candidates = _horizontal_candidates(width, set(v_cols), gap, charge, side)
        if not candidates:
            return None
        weighted = []
        for cand in candidates:
            cross_bonus = 3 if cand["cross_cols"] else 1
            if success and charge == gap:
                cross_bonus += 1
            weighted.extend([cand] * cross_bonus)
        chosen = random.choice(weighted)
        task = {
            "axis": "row",
            "fixed": int(r),
            "marker0": int(chosen["marker0"]),
            "marker1": int(chosen["marker1"]),
            "bar0": int(chosen["bar0"]),
            "bar1": int(chosen["bar1"]),
            "gap": int(gap),
            "charge": int(charge),
            "side": side,
        }
        grid[r, task["marker0"]] = Color.BLUE
        grid[r, task["marker1"]] = Color.BLUE
        grid[r, task["bar0"] : task["bar1"] + 1] = Color.GREY
        tasks.append(task)

    for c, side, success in zip(v_cols, col_sides, flags[row_count:]):
        gap, charge = _sample_gap_and_charge(success)
        candidates = _vertical_candidates(height, set(h_rows), gap, charge, side)
        if not candidates:
            return None
        weighted = []
        for cand in candidates:
            cross_bonus = 3 if cand["cross_rows"] else 1
            if success and charge == gap:
                cross_bonus += 1
            weighted.extend([cand] * cross_bonus)
        chosen = random.choice(weighted)
        task = {
            "axis": "col",
            "fixed": int(c),
            "marker0": int(chosen["marker0"]),
            "marker1": int(chosen["marker1"]),
            "bar0": int(chosen["bar0"]),
            "bar1": int(chosen["bar1"]),
            "gap": int(gap),
            "charge": int(charge),
            "side": side,
        }
        grid[task["marker0"], c] = Color.BLUE
        grid[task["marker1"], c] = Color.BLUE
        grid[task["bar0"] : task["bar1"] + 1, c] = Color.GREY
        tasks.append(task)

    if not _validate_semantics(grid, tasks):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
