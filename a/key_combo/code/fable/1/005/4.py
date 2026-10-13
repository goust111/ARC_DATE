from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists():
    sys.path.append(str(_ARC_UTILS_DIR))

try:
    from arc_utils import Color, bbox, get_bg
except Exception:  # pragma: no cover - fallback keeps SourceFileLoader useable
    class Color:
        BLACK = 0
        RED = 2
        YELLOW = 4
        GREY = 5
        TEAL = 8

    def get_bg(grid: np.ndarray) -> int:
        values, counts = np.unique(grid, return_counts=True)
        return int(values[np.argmax(counts)])

    def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
        coords = np.argwhere(mask)
        if coords.size == 0:
            return None
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return int(r0), int(c0), int(r1), int(c1)


BG = int(Color.BLACK)
RED = int(Color.RED)
YELLOW = int(Color.YELLOW)
DIVIDER = int(Color.GREY)
OCCLUDER = int(Color.TEAL)

MIN_MARGIN = 1
HEIGHT_RANGE = (5, 10)  # high exclusive
LEFT_WIDTH_RANGE = (7, 14)
RIGHT_WIDTH_RANGE = (7, 14)
MAX_TOTAL_LEN = 10
MAX_GAP = 4
MAX_GENERATION_RETRIES = 500


def _single_run(mask: np.ndarray) -> tuple[int, int] | None:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return None
    assert np.all(np.diff(idx) == 1), "expected one contiguous run"
    return int(idx[0]), int(idx[-1])


def _find_divider_col(grid: np.ndarray) -> int:
    divider_cols = np.flatnonzero(np.all(grid == DIVIDER, axis=0))
    assert divider_cols.size == 1, "expected one full grey divider column"
    return int(divider_cols[0])


def _parse_rows(grid: np.ndarray, divider_col: int) -> list[dict[str, int | bool | None]]:
    rows: list[dict[str, int | bool | None]] = []
    for r in range(grid.shape[0]):
        left = grid[r, :divider_col]
        right = grid[r, divider_col + 1 :]

        assert np.all(np.isin(left, [BG, YELLOW])), "left side must contain only black/yellow"
        yellow_run = _single_run(left == YELLOW)
        assert yellow_run is not None, "every row needs one yellow bar"
        y0, y1 = yellow_run
        gap = int(divider_col - y1 - 1)
        ylen = int(y1 - y0 + 1)

        if np.all(right == OCCLUDER):
            rows.append({"gap": gap, "ylen": ylen, "rlen": None, "occluded": True})
            continue

        assert np.all(np.isin(right, [BG, RED])), "visible right side must contain only black/red"
        red_run = _single_run(right == RED)
        assert red_run is not None, "every visible row needs one red bar"
        r0, r1 = red_run
        red_gap = int(r0)
        rlen = int(r1 - r0 + 1)
        assert red_gap == gap, "red and yellow near-end distances must match"
        rows.append({"gap": gap, "ylen": ylen, "rlen": rlen, "occluded": False})
    return rows


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the hidden right-side red bars by combining mirror-position and complementary-length conservation laws.

    Input structure:
    - Black background with one full-height grey vertical divider.
    - Each row has one yellow horizontal bar on the left and one red horizontal bar on the right.
    - One or two rows have their entire right side replaced by one solid teal rectangle.

    Rule:
    - Visible rows share two invariants: the red bar starts as far from the divider as the yellow bar ends from it, and yellow length plus red length is the same constant on every row.
    - Infer that constant from the unobscured rows.
    - For each teal-covered row, erase the teal to black, then place the missing red bar at the mirrored gap with the complementary length.

    Key insight:
    - Position is determined row-by-row by the yellow bar's gap to the divider, but length is determined globally by a separate conserved total; both are needed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    divider_col = _find_divider_col(g)
    rows = _parse_rows(g, divider_col)
    visible = [row for row in rows if not bool(row["occluded"])]
    assert visible, "need at least one visible row"

    totals = {int(row["ylen"]) + int(row["rlen"]) for row in visible}
    assert len(totals) == 1, "visible rows must share one conserved total length"
    total_len = totals.pop()

    out = g.copy()
    bg = int(get_bg(g))
    occ_box = bbox(g == OCCLUDER)
    if occ_box is not None:
        r0, c0, r1, c1 = occ_box
        out[r0 : r1 + 1, c0 : c1 + 1] = bg

    for r, row in enumerate(rows):
        if not bool(row["occluded"]):
            continue
        gap = int(row["gap"])
        ylen = int(row["ylen"])
        rlen = int(total_len - ylen)
        red_start = divider_col + 1 + gap
        out[r, divider_col + 1 :] = bg
        out[r, red_start : red_start + rlen] = RED

    return out


def _feasible_y_lengths(left_width: int, right_width: int, total_len: int) -> dict[int, list[int]]:
    feasible: dict[int, list[int]] = {}
    max_gap = min(MAX_GAP, left_width - MIN_MARGIN - 1, right_width - MIN_MARGIN - 1)
    for gap in range(1, max_gap + 1):
        ys = []
        for ylen in range(1, total_len):
            rlen = total_len - ylen
            if ylen + gap <= left_width - MIN_MARGIN and rlen + gap <= right_width - MIN_MARGIN:
                ys.append(int(ylen))
        if ys:
            feasible[int(gap)] = ys
    return feasible


def _choose_plan(
    rng: np.random.Generator,
    height: int,
    occluded_count: int,
    left_width: int,
    right_width: int,
) -> tuple[int, list[tuple[int, int]], set[int]] | None:
    total_candidates = [
        total_len
        for total_len in range(4, min(MAX_TOTAL_LEN, left_width + right_width) + 1, 2)
    ]
    rng.shuffle(total_candidates)

    for total_len in total_candidates:
        equal_y = total_len // 2
        feasible = _feasible_y_lengths(left_width, right_width, total_len)
        gaps = [gap for gap, ys in feasible.items() if equal_y in ys]
        if len(gaps) < 2:
            continue

        for gap_a in rng.permutation(gaps):
            ys_a = set(feasible[int(gap_a)])
            for gap_b in rng.permutation([gap for gap in feasible if gap != int(gap_a)]):
                ys_b = set(feasible[int(gap_b)])
                shared_non_equal = sorted((ys_a & ys_b) - {equal_y})
                if not shared_non_equal:
                    continue

                if occluded_count == 1:
                    y_a = int(rng.choice(shared_non_equal))
                    y_b_choices = sorted(ys_b - {equal_y, y_a})
                    if not y_b_choices:
                        continue
                    y_b = int(rng.choice(y_b_choices))
                    visible_seed = [(int(gap_a), equal_y), (int(gap_a), y_a), (int(gap_b), y_b)]
                    occluded_seed = [(int(gap_b), y_a)]
                else:
                    if len(shared_non_equal) < 2:
                        continue
                    y_a, y_b = map(int, rng.choice(shared_non_equal, size=2, replace=False))
                    visible_seed = [(int(gap_a), equal_y), (int(gap_a), y_a), (int(gap_b), y_b)]
                    occluded_seed = [(int(gap_b), y_a), (int(gap_a), y_b)]

                occ_start = int(rng.integers(0, height - occluded_count + 1))
                occluded_rows = set(range(occ_start, occ_start + occluded_count))
                visible_rows = [r for r in range(height) if r not in occluded_rows]
                if len(visible_rows) < len(visible_seed):
                    continue

                all_pairs = [(gap, ylen) for gap, ys in feasible.items() for ylen in ys]
                row_pairs: list[tuple[int, int] | None] = [None] * height

                visible_pool = visible_seed[:]
                while len(visible_pool) < len(visible_rows):
                    visible_pool.append(all_pairs[int(rng.integers(len(all_pairs)))])

                for r, pair in zip(rng.permutation(visible_rows), rng.permutation(visible_pool)):
                    row_pairs[int(r)] = tuple(map(int, pair))
                for r, pair in zip(sorted(occluded_rows), rng.permutation(occluded_seed)):
                    row_pairs[int(r)] = tuple(map(int, pair))

                assert all(pair is not None for pair in row_pairs)
                return int(total_len), [pair for pair in row_pairs if pair is not None], occluded_rows

    return None


def _render_case(
    height: int,
    left_width: int,
    right_width: int,
    total_len: int,
    row_pairs: list[tuple[int, int]],
    occluded_rows: set[int],
) -> tuple[np.ndarray, np.ndarray]:
    width = left_width + 1 + right_width
    inp = np.full((height, width), BG, dtype=int)
    out = np.full((height, width), BG, dtype=int)
    inp[:, left_width] = DIVIDER
    out[:, left_width] = DIVIDER

    for r, (gap, ylen) in enumerate(row_pairs):
        rlen = total_len - ylen
        y_start = left_width - gap - ylen
        red_start = left_width + 1 + gap

        inp[r, y_start : y_start + ylen] = YELLOW
        out[r, y_start : y_start + ylen] = YELLOW
        out[r, red_start : red_start + rlen] = RED

        if r in occluded_rows:
            inp[r, left_width + 1 :] = OCCLUDER
        else:
            inp[r, red_start : red_start + rlen] = RED

    return inp, out


def _equal_length_guess(grid: np.ndarray) -> np.ndarray:
    guess = np.asarray(grid, dtype=int).copy()
    divider_col = _find_divider_col(guess)
    rows = _parse_rows(guess, divider_col)
    guess[guess == OCCLUDER] = BG
    for r, row in enumerate(rows):
        if not bool(row["occluded"]):
            continue
        gap = int(row["gap"])
        ylen = int(row["ylen"])
        red_start = divider_col + 1 + gap
        guess[r, divider_col + 1 :] = BG
        guess[r, red_start : red_start + ylen] = RED
    return guess


def _semantic_checks(inp: np.ndarray, out: np.ndarray, total_len: int, occluded_rows: set[int]) -> bool:
    if not np.array_equal(main(inp), out):
        return False

    divider_col = _find_divider_col(inp)
    inp_rows = _parse_rows(inp, divider_col)
    out_rows = _parse_rows(out, divider_col)

    visible_rows = [r for r, row in enumerate(inp_rows) if not bool(row["occluded"])]
    hidden_rows = [r for r, row in enumerate(inp_rows) if bool(row["occluded"])]

    if set(hidden_rows) != set(occluded_rows):
        return False
    if len(hidden_rows) not in (1, 2) or len(visible_rows) < 3:
        return False
    if len({int(inp_rows[r]["gap"]) for r in visible_rows}) < 2:
        return False
    if len({int(inp_rows[r]["ylen"]) for r in visible_rows}) < 3:
        return False
    if not any(int(inp_rows[r]["ylen"]) * 2 == total_len for r in visible_rows):
        return False
    if not any(int(inp_rows[r]["ylen"]) * 2 != total_len for r in visible_rows):
        return False
    if any(bool(row["occluded"]) for row in out_rows):
        return False
    if any(int(row["ylen"]) + int(row["rlen"]) != total_len for row in out_rows):
        return False

    visible_specs = [(int(inp_rows[r]["gap"]), int(inp_rows[r]["ylen"])) for r in visible_rows]
    for r in hidden_rows:
        gap = int(inp_rows[r]["gap"])
        ylen = int(inp_rows[r]["ylen"])
        if ylen * 2 == total_len:
            return False
        if not any(v_y == ylen and v_gap != gap for v_gap, v_y in visible_specs):
            return False
        if not any(v_gap == gap and v_y != ylen for v_gap, v_y in visible_specs):
            return False

    if np.array_equal(_equal_length_guess(inp), out):
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(*HEIGHT_RANGE))
    occluded_count = int(rng.choice([1, 2]))
    if height < occluded_count + 3:
        return None

    left_width = int(rng.integers(*LEFT_WIDTH_RANGE))
    right_width = int(rng.integers(*RIGHT_WIDTH_RANGE))
    if left_width + 1 + right_width > 30:
        return None

    plan = _choose_plan(rng, height, occluded_count, left_width, right_width)
    if plan is None:
        return None

    total_len, row_pairs, occluded_rows = plan
    inp, out = _render_case(height, left_width, right_width, total_len, row_pairs, occluded_rows)

    if set(np.unique(inp)) != {BG, RED, YELLOW, DIVIDER, OCCLUDER}:
        return None
    if int(get_bg(inp)) != BG:
        return None
    if not _semantic_checks(inp, out, total_len, occluded_rows):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
