from __future__ import annotations

import random

import numpy as np


BG = 0
SHELF = 5
MARBLE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def _grey_runs(row: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    width = int(row.shape[0])
    c = 0
    while c < width:
        if int(row[c]) != SHELF:
            c += 1
            continue
        start = c
        while c + 1 < width and int(row[c + 1]) == SHELF:
            c += 1
        runs.append((int(start), int(c)))
        c += 1
    return runs


def _parse_shelves(grid: np.ndarray) -> list[tuple[int, int, int, int | None]]:
    shelves: list[tuple[int, int, int, int | None]] = []
    height, width = grid.shape
    for r in range(1, height - 1):
        runs = _grey_runs(grid[r])
        if not runs:
            continue
        if len(runs) == 1:
            a, b = runs[0]
            gap = None
        elif len(runs) == 2 and runs[1][0] - runs[0][1] == 2:
            a = int(runs[0][0])
            b = int(runs[1][1])
            gap = int(runs[0][1] + 1)
        else:
            raise AssertionError("each shelf row must contain one shelf, with at most one single-cell gap")
        assert 0 <= a <= b < width
        shelves.append((int(r), int(a), int(b), gap))
    return shelves


def _top_marbles(grid: np.ndarray) -> list[tuple[int, int]]:
    marbles: list[tuple[int, int]] = []
    for c, value in enumerate(grid[0]):
        color = int(value)
        if color in (BG, SHELF):
            assert color != SHELF, "shelves do not appear on the top row"
            continue
        marbles.append((int(c), color))
    assert marbles, "expected at least one top-row marble"
    colors = [color for _c, color in marbles]
    assert len(colors) == len(set(colors)), "top-row marbles must have distinct colors"
    return marbles


def _trace_marble(
    shelves: list[tuple[int, int, int, int | None]],
    start_col: int,
    width: int,
    *,
    ignore_gaps: bool = False,
    invert_preference: bool = False,
) -> dict[str, object]:
    col = int(start_col)
    current_row = 0
    impacts: list[dict[str, int | bool | None]] = []

    while True:
        hit = None
        for shelf_row, a, b, gap in shelves:
            if shelf_row <= current_row:
                continue
            if a <= col <= b and col != gap:
                hit = (shelf_row, a, b, gap)
                break
        if hit is None:
            return {
                "final_col": int(col),
                "impact_count": int(len(impacts)),
                "used_gap": bool(any(bool(info["used_gap"]) for info in impacts)),
                "used_end": bool(any(not bool(info["used_gap"]) for info in impacts)),
                "first_dir": None if not impacts else int(impacts[0]["dir"]),
                "impacts": impacts,
            }

        shelf_row, a, b, gap = hit
        left_dist = int(col - a)
        right_dist = int(b - col)
        if left_dist == right_dist:
            raise AssertionError("the puzzle guarantees a unique nearer end at every impact")

        choose_left = left_dist < right_dist
        if invert_preference:
            choose_left = not choose_left
        direction = -1 if choose_left else 1
        overflow_col = (a - 1) if choose_left else (b + 1)
        assert 0 <= overflow_col < width, "chosen shelf end must drop inside the grid"

        used_gap = False
        exit_col = overflow_col
        pos = col
        while True:
            pos += direction
            if gap is not None and pos == gap and not ignore_gaps:
                used_gap = True
                exit_col = int(pos)
                break
            if pos == overflow_col:
                break

        impacts.append(
            {
                "shelf_row": int(shelf_row),
                "start_col": int(col),
                "left_dist": int(left_dist),
                "right_dist": int(right_dist),
                "dir": int(direction),
                "gap": None if gap is None else int(gap),
                "used_gap": bool(used_gap),
                "exit_col": int(exit_col),
            }
        )
        col = int(exit_col)
        current_row = int(shelf_row)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop each top-row marble through grey shelves and keep only its final bottom landing cell.

    Input structure:
    - The background is black.
    - Distinct single-cell marbles of different colors appear on the top row.
    - Lower rows contain horizontal grey shelves; a shelf row is either one contiguous bar or two grey runs split by one black gap cell.

    Rule:
    - Each marble falls vertically until it would hit a shelf cell in its column.
    - On that shelf, compare the horizontal distances to the left and right shelf ends and roll toward the nearer end; ties do not occur.
    - If the marble reaches a shelf gap before it reaches that chosen end, it falls from the gap column; otherwise it rolls past the chosen end and falls from the adjacent outside column.
    - Repeat this fall-roll process until the marble reaches the bottom row.
    - Output preserves all shelves and gaps, removes the top-row marbles, and places each marble only at its final bottom landing cell.

    Key insight:
    - The trajectory is determined by two coupled decisions at every impact: first choose the nearer shelf end, then let a gap on that same route intercept the motion before the end can be reached.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    height, width = g.shape
    assert 1 <= height <= 30 and 1 <= width <= 30
    assert not np.any(g[1:, :] == SHELF) or height >= 3
    assert set(map(int, np.unique(g[1:, :]))) <= {BG, SHELF}, "only shelves may appear below the top row"

    shelves = _parse_shelves(g)
    marbles = _top_marbles(g)

    out = g.copy()
    for c, _color in marbles:
        out[0, c] = BG

    used_final_cols: set[int] = set()
    for start_col, color in marbles:
        info = _trace_marble(shelves, start_col, width)
        final_col = int(info["final_col"])
        assert final_col not in used_final_cols, "final landing columns must be unique"
        used_final_cols.add(final_col)
        assert int(out[height - 1, final_col]) == BG, "bottom landing cells must be empty in the input"
        out[height - 1, final_col] = int(color)

    return out


def _sample_shelf_rows(height: int, count: int) -> list[int] | None:
    candidates = list(range(3, height - 2))
    for _ in range(60):
        rows = sorted(random.sample(candidates, count))
        if all(b - a >= 2 for a, b in zip(rows, rows[1:])):
            return rows
    return None


def _build_layout() -> tuple[int, int, list[tuple[int, int, int, int | None]]] | None:
    height = random.randint(12, 22)
    width = random.randint(12, 24)
    shelf_count = random.randint(3, min(6, height - 5))
    rows = _sample_shelf_rows(height, shelf_count)
    if rows is None:
        return None

    shelves: list[tuple[int, int, int, int | None]] = []
    for row in rows:
        max_len = min(12, width - 2)
        min_len = 5
        if min_len > max_len:
            return None
        length = random.randint(min_len, max_len)
        left = random.randint(1, width - length - 1)
        right = left + length - 1
        gap = None
        if length >= 6 and random.random() < 0.6:
            gap = random.randint(left + 1, right - 1)
        shelves.append((int(row), int(left), int(right), gap))
    return height, width, shelves


def _candidate_infos(
    shelves: list[tuple[int, int, int, int | None]],
    width: int,
) -> list[dict[str, object]]:
    infos: list[dict[str, object]] = []
    for start_col in range(width):
        try:
            info = _trace_marble(shelves, start_col, width)
            gapless = _trace_marble(shelves, start_col, width, ignore_gaps=True)
            inverted = _trace_marble(shelves, start_col, width, invert_preference=True)
        except AssertionError:
            continue
        if int(info["impact_count"]) == 0:
            continue
        enriched = dict(info)
        enriched["start_col"] = int(start_col)
        enriched["gapless_final_col"] = int(gapless["final_col"])
        enriched["inverted_final_col"] = int(inverted["final_col"])
        infos.append(enriched)
    return infos


def _covers_constraints(chosen: list[dict[str, object]]) -> bool:
    if len(chosen) < 3:
        return False
    first_dirs = {int(info["first_dir"]) for info in chosen}
    has_gap_effect = any(
        bool(info["used_gap"]) and int(info["gapless_final_col"]) != int(info["final_col"])
        for info in chosen
    )
    has_nearer_effect = any(
        int(info["inverted_final_col"]) != int(info["final_col"]) for info in chosen
    )
    has_multistage = any(int(info["impact_count"]) >= 2 for info in chosen)
    has_end_fall = any(bool(info["used_end"]) for info in chosen)
    return first_dirs == {-1, 1} and has_gap_effect and has_nearer_effect and has_multistage and has_end_fall


def _pick_marbles(candidates: list[dict[str, object]]) -> list[dict[str, object]] | None:
    pool = candidates[:]
    random.shuffle(pool)
    pool.sort(key=lambda info: (-int(info["impact_count"]), -int(bool(info["used_gap"])), random.random()))

    for target_count in random.sample([3, 4], 2):
        if target_count > len(pool):
            continue
        answer: list[dict[str, object]] | None = None

        def search(
            index: int,
            chosen: list[dict[str, object]],
            start_cols: set[int],
            final_cols: set[int],
        ) -> None:
            nonlocal answer
            if answer is not None:
                return
            if len(chosen) == target_count:
                if _covers_constraints(chosen):
                    answer = [dict(item) for item in chosen]
                return
            if index == len(pool) or len(chosen) + len(pool) - index < target_count:
                return

            info = pool[index]
            start_col = int(info["start_col"])
            final_col = int(info["final_col"])
            if (
                final_col not in final_cols
                and all(abs(start_col - old) >= 2 for old in start_cols)
            ):
                chosen.append(info)
                start_cols.add(start_col)
                final_cols.add(final_col)
                search(index + 1, chosen, start_cols, final_cols)
                chosen.pop()
                start_cols.remove(start_col)
                final_cols.remove(final_col)
            search(index + 1, chosen, start_cols, final_cols)

        search(0, [], set(), set())
        if answer is not None:
            return answer
    return None


def _render_grid(
    height: int,
    width: int,
    shelves: list[tuple[int, int, int, int | None]],
    marbles: list[dict[str, object]],
) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for row, left, right, gap in shelves:
        for c in range(left, right + 1):
            if c != gap:
                grid[row, c] = SHELF

    colors = random.sample(MARBLE_COLORS, len(marbles))
    for color, info in zip(colors, marbles):
        grid[0, int(info["start_col"])] = int(color)
    return grid


def _validate_generated_grid(grid: np.ndarray) -> bool:
    try:
        shelves = _parse_shelves(grid)
        marbles = _top_marbles(grid)
        if len(marbles) < 3:
            return False
        if any(int(info[0]) == 0 or int(info[0]) == grid.shape[1] - 1 for info in marbles):
            return False

        infos = []
        for start_col, color in marbles:
            traced = _trace_marble(shelves, start_col, grid.shape[1])
            gapless = _trace_marble(shelves, start_col, grid.shape[1], ignore_gaps=True)
            inverted = _trace_marble(shelves, start_col, grid.shape[1], invert_preference=True)
            enriched = dict(traced)
            enriched["start_col"] = int(start_col)
            enriched["color"] = int(color)
            enriched["gapless_final_col"] = int(gapless["final_col"])
            enriched["inverted_final_col"] = int(inverted["final_col"])
            infos.append(enriched)

        if len({int(info["final_col"]) for info in infos}) != len(infos):
            return False
        if not _covers_constraints(infos):
            return False

        out = main(grid)
        if not np.array_equal(out[1:-1], grid[1:-1]):
            return False
        if np.any(out[0] != BG):
            return False

        bottom = [(c, int(out[-1, c])) for c in range(out.shape[1]) if int(out[-1, c]) != BG]
        if len(bottom) != len(marbles):
            return False
        expected = sorted((int(info["final_col"]), int(info["color"])) for info in infos)
        actual = sorted(bottom)
        if actual != expected:
            return False
        return True
    except Exception:
        return False


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        layout = _build_layout()
        if layout is None:
            continue
        height, width, shelves = layout
        candidates = _candidate_infos(shelves, width)
        marbles = _pick_marbles(candidates)
        if marbles is None:
            continue
        grid = _render_grid(height, width, shelves, marbles)
        if _validate_generated_grid(grid):
            return grid
    raise RuntimeError("failed to generate a valid marble-waterfall puzzle")
