import random
from typing import Literal

import numpy as np


BG = 0
BLUE = 1
RED = 2
AXIS = 5
LINE_COLORS = (BLUE, RED)

MAX_ATTEMPTS = 400
HEIGHT_RANGE = (13, 19)
WIDTH_RANGE = (13, 19)
LINE_COUNT_RANGE = (4, 6)
LINE_LEN_RANGE = (7, 10)
GAP_LEN_RANGE = (1, 3)


def _mirror_col(axis_col: int, col: int) -> int:
    return 2 * axis_col - col


def _line_mask(value: int) -> bool:
    return int(value) in LINE_COLORS


def _assert_axis_symmetry(grid: np.ndarray, axis_col: int) -> None:
    h, w = grid.shape
    for r in range(h):
        assert int(grid[r, axis_col]) == AXIS
        for c in range(axis_col):
            mc = _mirror_col(axis_col, c)
            assert 0 <= mc < w
            assert int(grid[r, c]) == int(grid[r, mc]), "output must mirror across the gray axis"


def _find_axis_col(grid: np.ndarray) -> int:
    g = np.asarray(grid, dtype=int)
    cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == AXIS)]
    assert len(cols) == 1, "expected exactly one full gray column"
    return int(cols[0])


def _fill_from_mirror(grid: np.ndarray, axis_col: int) -> tuple[np.ndarray, set[tuple[int, int]]]:
    out = np.asarray(grid, dtype=int).copy()
    filled: set[tuple[int, int]] = set()
    h, _w = out.shape
    for r in range(h):
        for c in range(axis_col):
            mc = _mirror_col(axis_col, c)
            left = int(out[r, c])
            right = int(out[r, mc])
            if left == BG and _line_mask(right):
                out[r, c] = right
                filled.add((r, c))
            elif right == BG and _line_mask(left):
                out[r, mc] = left
                filled.add((r, mc))
    return out, filled


def _choose_supported_color(
    first: tuple[int, int] | None,
    second: tuple[int, int] | None,
    prefer: Literal["first", "second"],
) -> int | None:
    if first is None or second is None:
        return None
    dist_first, color_first = first
    dist_second, color_second = second
    if dist_first < dist_second:
        return int(color_first)
    if dist_second < dist_first:
        return int(color_second)
    if color_first == color_second:
        return int(color_first)
    return int(color_first if prefer == "first" else color_second)


def _scan_h_support(
    grid: np.ndarray, row: int, start: int, end: int, side_min: int, side_max: int
) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    left_info = None
    for c in range(start - 1, side_min - 1, -1):
        value = int(grid[row, c])
        if _line_mask(value) and c - 1 >= side_min and int(grid[row, c - 1]) == value:
            left_info = (start - c, value)
            break

    right_info = None
    for c in range(end + 1, side_max + 1):
        value = int(grid[row, c])
        if _line_mask(value) and c + 1 <= side_max and int(grid[row, c + 1]) == value:
            right_info = (c - end, value)
            break

    return left_info, right_info


def _scan_v_support(
    grid: np.ndarray, col: int, start: int, end: int
) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    h = grid.shape[0]
    up_info = None
    for r in range(start - 1, -1, -1):
        value = int(grid[r, col])
        if _line_mask(value) and r - 1 >= 0 and int(grid[r - 1, col]) == value:
            up_info = (start - r, value)
            break

    down_info = None
    for r in range(end + 1, h):
        value = int(grid[r, col])
        if _line_mask(value) and r + 1 < h and int(grid[r + 1, col]) == value:
            down_info = (r - end, value)
            break

    return up_info, down_info


def _fill_local_gaps(grid: np.ndarray, axis_col: int) -> tuple[np.ndarray, set[tuple[int, int]]]:
    base = np.asarray(grid, dtype=int)
    out = base.copy()
    filled: set[tuple[int, int]] = set()
    h, _w = base.shape
    left_max = axis_col - 1
    right_min = axis_col + 1
    right_max = base.shape[1] - 1

    for r in range(h):
        c = 0
        while c <= left_max:
            if int(base[r, c]) != BG:
                c += 1
                continue
            start = c
            while c <= left_max and int(base[r, c]) == BG:
                c += 1
            end = c - 1
            mirror_start = _mirror_col(axis_col, end)
            mirror_end = _mirror_col(axis_col, start)
            if not np.all(base[r, mirror_start : mirror_end + 1] == BG):
                continue
            left_info, right_info = _scan_h_support(base, r, start, end, 0, left_max)
            color = _choose_supported_color(left_info, right_info, prefer="second")
            if color is None:
                continue
            out[r, start : end + 1] = color
            out[r, mirror_start : mirror_end + 1] = color
            for cc in range(start, end + 1):
                filled.add((r, cc))
            for cc in range(mirror_start, mirror_end + 1):
                filled.add((r, cc))

    for c in range(0, axis_col):
        r = 0
        while r < h:
            if int(base[r, c]) != BG:
                r += 1
                continue
            start = r
            while r < h and int(base[r, c]) == BG:
                r += 1
            end = r - 1
            mc = _mirror_col(axis_col, c)
            if not np.all(base[start : end + 1, mc] == BG):
                continue
            up_info, down_info = _scan_v_support(base, c, start, end)
            color = _choose_supported_color(up_info, down_info, prefer="first")
            if color is None:
                continue
            out[start : end + 1, c] = color
            out[start : end + 1, mc] = color
            for rr in range(start, end + 1):
                filled.add((rr, c))
                filled.add((rr, mc))

    return out, filled


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair a bilaterally damaged rectilinear line network so it becomes perfectly symmetric around the gray axis.

    Input structure:
    - The background is black.
    - Exactly one full gray column marks the vertical reflection axis.
    - Blue and red pixels form one-cell-thick horizontal and vertical line traces on both sides of that axis.
    - The complete traces should mirror perfectly across the axis, but some internal runs are missing on one side only and some mirrored runs are missing on both sides.

    Rule:
    - Keep the gray axis unchanged.
    - First repair every one-sided gap by copying the mirrored non-background pixels from the opposite side cell-for-cell.
    - Then repair every still-black mirrored gap pair by scanning along its row for a horizontal gap, or along its column for a vertical gap, until the first intact same-direction line segment is found on each side.
    - Fill the whole mirrored gap pair with the nearer supporting segment's color; if both supports are equally far, use the support on the side closer to the gray axis.

    Key insight:
    - Single-side damage is resolved by global mirror correspondence, while double-side damage can only be resolved from local line-direction evidence that remains consistent with the same global symmetry.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    axis_col = _find_axis_col(g)
    out, _reflected = _fill_from_mirror(g, axis_col)
    out, _local = _fill_local_gaps(out, axis_col)
    _assert_axis_symmetry(out, axis_col)
    mask = (g != BG) & (g != AXIS)
    assert np.array_equal(out[mask], g[mask]), "existing line pixels must stay unchanged"
    return out


def _random_run_sequence(length: int) -> list[int]:
    n_runs = random.randint(1, min(4, length))
    cuts = sorted(random.sample(range(1, length), n_runs - 1))
    bounds = [0, *cuts, length]
    colors: list[int] = []
    current = random.choice(LINE_COLORS)
    for i in range(n_runs):
        run_len = bounds[i + 1] - bounds[i]
        colors.extend([current] * run_len)
        if i + 1 < n_runs:
            current = RED if current == BLUE else BLUE
    return colors


def _occupancy_hits(occupied: set[tuple[int, int]], cells: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    for r, c in cells:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                if (rr, cc) in occupied:
                    return True
    return False


def _paint_line_pair(grid: np.ndarray, line: dict) -> None:
    axis_col = int(line["axis_col"])
    seq = list(map(int, line["seq"]))
    if line["orientation"] == "h":
        row = int(line["fixed"])
        start = int(line["start"])
        for i, color in enumerate(seq):
            grid[row, start + i] = color
        mirror_seq = list(reversed(seq))
        right_start = _mirror_col(axis_col, start + len(seq) - 1)
        for i, color in enumerate(mirror_seq):
            grid[row, right_start + i] = color
    else:
        col = int(line["fixed"])
        start = int(line["start"])
        for i, color in enumerate(seq):
            grid[start + i, col] = color
        mc = _mirror_col(axis_col, col)
        for i, color in enumerate(seq):
            grid[start + i, mc] = color


def _line_cells(line: dict) -> list[tuple[int, int]]:
    axis_col = int(line["axis_col"])
    seq = list(line["seq"])
    cells: list[tuple[int, int]] = []
    if line["orientation"] == "h":
        row = int(line["fixed"])
        start = int(line["start"])
        for i in range(len(seq)):
            c = start + i
            cells.append((row, c))
            cells.append((row, _mirror_col(axis_col, c)))
    else:
        col = int(line["fixed"])
        mc = _mirror_col(axis_col, col)
        start = int(line["start"])
        for i in range(len(seq)):
            r = start + i
            cells.append((r, col))
            cells.append((r, mc))
    return cells


def _make_line(axis_col: int, h: int, w: int, orientation: str, used_rows: set[int], used_cols: set[int]) -> dict | None:
    max_len = min(LINE_LEN_RANGE[1], axis_col - 1, h - 2 if orientation == "v" else axis_col - 1)
    min_len = min(LINE_LEN_RANGE[0], max_len)
    if max_len < LINE_LEN_RANGE[0]:
        return None

    if orientation == "h":
        rows = [r for r in range(1, h - 1) if r not in used_rows]
        if not rows:
            return None
        row = random.choice(rows)
        length = random.randint(min_len, max_len)
        start_min = 1
        start_max = axis_col - length
        if start_min > start_max:
            return None
        start = random.randint(start_min, start_max)
        used_rows.add(row)
        return {
            "orientation": "h",
            "fixed": row,
            "start": start,
            "seq": _random_run_sequence(length),
            "axis_col": axis_col,
        }

    cols = [c for c in range(1, axis_col - 1) if c not in used_cols]
    if not cols:
        return None
    col = random.choice(cols)
    max_len = min(LINE_LEN_RANGE[1], h - 2)
    min_len = LINE_LEN_RANGE[0]
    if max_len < min_len:
        return None
    length = random.randint(min_len, max_len)
    start_min = 1
    start_max = h - 1 - length
    if start_min > start_max:
        return None
    start = random.randint(start_min, start_max)
    used_cols.add(col)
    return {
        "orientation": "v",
        "fixed": col,
        "start": start,
        "seq": _random_run_sequence(length),
        "axis_col": axis_col,
    }


def _rewrite_double_gap(line: dict, branch: str) -> tuple[int, int]:
    seq = list(map(int, line["seq"]))
    length = len(seq)

    colors = [BLUE, RED]
    fill = random.choice(colors)
    other = RED if fill == BLUE else BLUE

    if branch == "same":
        left_dist = random.choice([1, 2])
        right_dist = random.choice([1, 2])
        left_color = fill
        right_color = fill
    elif branch == "tie_axis":
        assert line["orientation"] == "h"
        left_dist = right_dist = random.choice([1, 2])
        left_color = other
        right_color = fill
    elif branch == "near_axis":
        assert line["orientation"] == "h"
        left_dist = 2
        right_dist = 1
        left_color = other
        right_color = fill
    elif branch == "near_outer":
        assert line["orientation"] == "h"
        left_dist = 1
        right_dist = 2
        left_color = fill
        right_color = other
    elif branch == "near_top":
        assert line["orientation"] == "v"
        left_dist = 1
        right_dist = 2
        left_color = fill
        right_color = other
    elif branch == "near_bottom":
        assert line["orientation"] == "v"
        left_dist = 2
        right_dist = 1
        left_color = other
        right_color = fill
    else:
        raise AssertionError(branch)

    left_need = left_dist + 1
    right_need = right_dist + 1
    feasible_gap_lengths = [
        gap_len
        for gap_len in range(GAP_LEN_RANGE[0], GAP_LEN_RANGE[1] + 1)
        if left_need <= length - gap_len - right_need
    ]
    assert feasible_gap_lengths
    gap_len = random.choice(feasible_gap_lengths)
    gap_start = random.randint(left_need, length - gap_len - right_need)
    gap_end = gap_start + gap_len - 1
    seq[gap_start : gap_end + 1] = [fill] * gap_len

    def set_left(dist: int, color: int) -> None:
        if dist == 1:
            seq[gap_start - 1] = color
            seq[gap_start - 2] = color
        else:
            seq[gap_start - 1] = other if color == fill else fill
            seq[gap_start - 2] = color
            seq[gap_start - 3] = color

    def set_right(dist: int, color: int) -> None:
        if dist == 1:
            seq[gap_end + 1] = color
            seq[gap_end + 2] = color
        else:
            seq[gap_end + 1] = other if color == fill else fill
            seq[gap_end + 2] = color
            seq[gap_end + 3] = color

    set_left(left_dist, left_color)
    set_right(right_dist, right_color)
    line["seq"] = seq
    return gap_start, gap_end


def _choose_gap(line: dict) -> tuple[int, int]:
    length = len(line["seq"])
    gap_len = random.randint(*GAP_LEN_RANGE)
    start = random.randint(1, length - gap_len - 1)
    return start, start + gap_len - 1


def _gap_cells(line: dict, gap: tuple[int, int], mode: str) -> set[tuple[int, int]]:
    axis_col = int(line["axis_col"])
    start_idx, end_idx = gap
    cells: set[tuple[int, int]] = set()
    if line["orientation"] == "h":
        row = int(line["fixed"])
        start = int(line["start"])
        cols = [start + i for i in range(start_idx, end_idx + 1)]
        if mode in {"left", "both"}:
            for c in cols:
                cells.add((row, c))
        if mode in {"right", "both"}:
            for c in cols:
                cells.add((row, _mirror_col(axis_col, c)))
    else:
        col = int(line["fixed"])
        mc = _mirror_col(axis_col, col)
        start = int(line["start"])
        rows = [start + i for i in range(start_idx, end_idx + 1)]
        if mode in {"left", "both"}:
            for r in rows:
                cells.add((r, col))
        if mode in {"right", "both"}:
            for r in rows:
                cells.add((r, mc))
    return cells


def _build_output_and_specs() -> tuple[np.ndarray, list[dict], dict] | None:
    h = random.choice([n for n in range(HEIGHT_RANGE[0], HEIGHT_RANGE[1] + 1)])
    w = random.choice([n for n in range(WIDTH_RANGE[0], WIDTH_RANGE[1] + 1) if n % 2 == 1])
    axis_col = w // 2
    out = np.full((h, w), BG, dtype=int)
    out[:, axis_col] = AXIS

    occupied: set[tuple[int, int]] = {(r, axis_col) for r in range(h)}
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    lines: list[dict] = []
    orientations = ["h", "v"]
    target_count = random.randint(*LINE_COUNT_RANGE)
    pending = orientations.copy()

    for _ in range(120):
        if len(lines) >= target_count:
            break
        if pending:
            orientation = pending.pop(0)
        else:
            orientation = random.choice(orientations)
        line = _make_line(axis_col, h, w, orientation, used_rows, used_cols)
        if line is None:
            continue
        cells = _line_cells(line)
        if _occupancy_hits(occupied, cells, out.shape):
            continue
        lines.append(line)
        for cell in cells:
            occupied.add(cell)

    if len(lines) < 4 or not any(line["orientation"] == "h" for line in lines) or not any(line["orientation"] == "v" for line in lines):
        return None

    line_specs: list[dict] = []
    double_candidates = [line for line in lines if len(line["seq"]) >= 7]
    if len(double_candidates) < 1:
        return None

    primary_double = random.choice(double_candidates)
    if primary_double["orientation"] == "h":
        branch = random.choices(
            ["same", "tie_axis", "near_axis", "near_outer"],
            weights=[2, 3, 2, 2],
        )[0]
    else:
        branch = random.choices(["same", "near_top", "near_bottom"], weights=[2, 2, 2])[0]
    gap = _rewrite_double_gap(primary_double, branch)
    line_specs.append({"line": primary_double, "gap": gap, "mode": "both", "branch": branch})

    single_candidates = [line for line in lines if line is not primary_double]
    if not single_candidates:
        return None
    primary_single = random.choice(single_candidates)
    line_specs.append(
        {
            "line": primary_single,
            "gap": _choose_gap(primary_single),
            "mode": random.choice(["left", "right"]),
            "branch": "mirror_only",
        }
    )

    remaining = [line for line in lines if line is not primary_double and line is not primary_single]
    random.shuffle(remaining)
    if remaining:
        extra = remaining[0]
        if random.random() < 0.55:
            if extra["orientation"] == "h":
                branch = random.choices(
                    ["same", "tie_axis", "near_axis", "near_outer"],
                    weights=[2, 3, 2, 2],
                )[0]
            else:
                branch = random.choices(["same", "near_top", "near_bottom"], weights=[2, 2, 2])[0]
            gap = _rewrite_double_gap(extra, branch)
            line_specs.append({"line": extra, "gap": gap, "mode": "both", "branch": branch})
        else:
            line_specs.append(
                {
                    "line": extra,
                    "gap": _choose_gap(extra),
                    "mode": random.choice(["left", "right"]),
                    "branch": "mirror_only",
                }
            )

    for line in lines:
        _paint_line_pair(out, line)

    if not (np.any(out == BLUE) and np.any(out == RED)):
        return None

    spec = {"axis_col": axis_col, "line_specs": line_specs}
    return out, lines, spec


def _solve_with_debug(grid: np.ndarray) -> tuple[np.ndarray, set[tuple[int, int]], set[tuple[int, int]]]:
    axis_col = _find_axis_col(grid)
    after_reflect, reflect_cells = _fill_from_mirror(grid, axis_col)
    solved, local_cells = _fill_local_gaps(after_reflect, axis_col)
    return solved, reflect_cells, local_cells


def _validate_case(inp: np.ndarray, out: np.ndarray, spec: dict) -> bool:
    if inp.ndim != 2 or out.ndim != 2 or max(inp.shape) > 30 or max(out.shape) > 30:
        return False

    _assert_axis_symmetry(out, int(spec["axis_col"]))

    single_cells: set[tuple[int, int]] = set()
    double_cells: set[tuple[int, int]] = set()
    saw_single = False
    saw_double = False
    for item in spec["line_specs"]:
        cells = _gap_cells(item["line"], item["gap"], item["mode"])
        if item["mode"] == "both":
            saw_double = True
            double_cells |= cells
        else:
            saw_single = True
            single_cells |= cells

    if not saw_single or not saw_double:
        return False

    solved, reflect_cells, local_cells = _solve_with_debug(inp)
    if not np.array_equal(solved, out):
        return False
    if reflect_cells != single_cells:
        return False
    if local_cells != double_cells:
        return False
    if np.count_nonzero(inp == BG) >= np.count_nonzero(out == BG):
        pass
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        built = _build_output_and_specs()
        if built is None:
            continue
        out, _lines, spec = built
        inp = out.copy()
        for item in spec["line_specs"]:
            for r, c in _gap_cells(item["line"], item["gap"], item["mode"]):
                inp[r, c] = BG
        if _validate_case(inp, out, spec):
            return inp
    raise RuntimeError("failed to generate a valid symmetric line-repair input")
