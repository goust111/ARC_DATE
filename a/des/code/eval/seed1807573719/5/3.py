from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BG = 0
YELLOW = 4
DIRS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
INPUT_COLORS = [1, 2, 3, 5, 6, 7, 8, 9]
GEN_ATTEMPTS = 300
MOVE_SAMPLE_LIMIT = 24


def _component_cells(grid: np.ndarray) -> list[tuple[int, list[tuple[int, int]]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[tuple[int, list[tuple[int, int]]]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((int(cr), int(cc)))
                for dr, dc in DIRS.values():
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append((color, cells))
    return components


def _square_cells(r0: int, c0: int) -> set[tuple[int, int]]:
    return {(int(r0 + dr), int(c0 + dc)) for dr in range(3) for dc in range(3)}


def _antenna_cells(r0: int, c0: int, direction: str, length: int) -> list[tuple[int, int]]:
    if direction == "U":
        return [(int(r0 - 1 - k), int(c0 + 1)) for k in range(length)]
    if direction == "D":
        return [(int(r0 + 3 + k), int(c0 + 1)) for k in range(length)]
    if direction == "L":
        return [(int(r0 + 1), int(c0 - 1 - k)) for k in range(length)]
    if direction == "R":
        return [(int(r0 + 1), int(c0 + 3 + k)) for k in range(length)]
    raise AssertionError(f"bad direction {direction}")


def _parse_object(color: int, cells: list[tuple[int, int]]) -> dict[str, int | tuple[int, int] | str]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0 = min(rows)
    r1 = max(rows)
    c0 = min(cols)
    c1 = max(cols)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    area = len(cells)
    cell_set = set(cells)

    candidates: list[dict[str, int | tuple[int, int] | str]] = []
    if w == 3 and 4 <= h <= 7 and area == 9 + (h - 3):
        length = h - 3
        sq_r = r1 - 2
        expected = _square_cells(sq_r, c0) | set(_antenna_cells(sq_r, c0, "U", length))
        if expected == cell_set:
            candidates.append(
                {
                    "color": int(color),
                    "square_r": int(sq_r),
                    "square_c": int(c0),
                    "dir": "U",
                    "length": int(length),
                    "tip": _antenna_cells(sq_r, c0, "U", length)[-1],
                }
            )
        sq_r = r0
        expected = _square_cells(sq_r, c0) | set(_antenna_cells(sq_r, c0, "D", length))
        if expected == cell_set:
            candidates.append(
                {
                    "color": int(color),
                    "square_r": int(sq_r),
                    "square_c": int(c0),
                    "dir": "D",
                    "length": int(length),
                    "tip": _antenna_cells(sq_r, c0, "D", length)[-1],
                }
            )
    if h == 3 and 4 <= w <= 7 and area == 9 + (w - 3):
        length = w - 3
        sq_c = c1 - 2
        expected = _square_cells(r0, sq_c) | set(_antenna_cells(r0, sq_c, "L", length))
        if expected == cell_set:
            candidates.append(
                {
                    "color": int(color),
                    "square_r": int(r0),
                    "square_c": int(sq_c),
                    "dir": "L",
                    "length": int(length),
                    "tip": _antenna_cells(r0, sq_c, "L", length)[-1],
                }
            )
        sq_c = c0
        expected = _square_cells(r0, sq_c) | set(_antenna_cells(r0, sq_c, "R", length))
        if expected == cell_set:
            candidates.append(
                {
                    "color": int(color),
                    "square_r": int(r0),
                    "square_c": int(sq_c),
                    "dir": "R",
                    "length": int(length),
                    "tip": _antenna_cells(r0, sq_c, "R", length)[-1],
                }
            )

    assert len(candidates) == 1, f"invalid object for color {color}"
    return candidates[0]


def _extract_objects(grid: np.ndarray) -> list[dict[str, int | tuple[int, int] | str]]:
    objs = [_parse_object(color, cells) for color, cells in _component_cells(grid)]
    objs.sort(key=lambda obj: (int(obj["length"]), int(obj["square_r"]), int(obj["square_c"])))
    return objs


def _segment_cells(cur: dict, nxt: dict) -> list[tuple[int, int]]:
    sr = int(cur["square_r"])
    sc = int(cur["square_c"])
    nr = int(nxt["square_r"])
    nc = int(nxt["square_c"])
    direction = str(cur["dir"])
    tip_r, tip_c = map(int, cur["tip"])

    if direction == "R":
        assert nr <= tip_r <= nr + 2 and tip_c < nc
        return [(int(tip_r), int(c)) for c in range(int(tip_c + 1), int(nc))]
    if direction == "L":
        assert nr <= tip_r <= nr + 2 and tip_c > nc + 2
        return [(int(tip_r), int(c)) for c in range(int(nc + 3), int(tip_c))]
    if direction == "D":
        assert nc <= tip_c <= nc + 2 and tip_r < nr
        return [(int(r), int(tip_c)) for r in range(int(tip_r + 1), int(nr))]
    assert direction == "U"
    assert nc <= tip_c <= nc + 2 and tip_r > sr - 1 and tip_r > nr + 2
    return [(int(r), int(tip_c)) for r in range(int(nr + 3), int(tip_r))]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    line_colors: dict[tuple[int, int], set[int]] = defaultdict(set)
    objects = _extract_objects(g)

    for i in range(len(objects) - 1):
        cur = objects[i]
        nxt = objects[i + 1]
        color = int(cur["color"])
        for cell in _segment_cells(cur, nxt):
            if int(g[cell]) == BG:
                line_colors[cell].add(color)

    for (r, c), colors in line_colors.items():
        out[r, c] = YELLOW if len(colors) >= 2 else next(iter(colors))
    return out


def _valid_cells(cells: list[tuple[int, int]] | set[tuple[int, int]], h: int, w: int) -> bool:
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _length_pattern(count: int, allow_ties: bool) -> list[int]:
    unique_patterns = {
        3: [[1, 2, 3], [1, 2, 4]],
        4: [[1, 2, 3, 4]],
        5: [[1, 1, 2, 3, 4]],
    }
    tie_patterns = {
        3: [[1, 1, 2], [1, 2, 2], [2, 3, 3]],
        4: [[1, 1, 2, 3], [1, 2, 2, 4], [1, 2, 3, 3], [1, 1, 2, 2]],
        5: [[1, 1, 2, 3, 4], [1, 2, 2, 3, 4], [1, 2, 3, 3, 4], [1, 1, 2, 2, 4]],
    }
    pool = tie_patterns[count] if allow_ties else unique_patterns[count]
    return list(random.choice(pool))


def _ordered_ties_ok(positions: list[tuple[int, int]], lengths: list[int]) -> bool:
    for i in range(1, len(positions)):
        if lengths[i - 1] == lengths[i] and positions[i - 1] > positions[i]:
            return False
    return True


def _line_background_cells(cur_r: int, cur_c: int, direction: str, length: int, nxt_r: int, nxt_c: int) -> list[tuple[int, int]]:
    tip_r, tip_c = _antenna_cells(cur_r, cur_c, direction, length)[-1]
    if direction == "R":
        return [(int(tip_r), int(c)) for c in range(int(tip_c + 1), int(nxt_c))]
    if direction == "L":
        return [(int(tip_r), int(c)) for c in range(int(nxt_c + 3), int(tip_c))]
    if direction == "D":
        return [(int(r), int(tip_c)) for r in range(int(tip_r + 1), int(nxt_r))]
    return [(int(r), int(tip_c)) for r in range(int(nxt_r + 3), int(tip_r))]


def _sample_axis(values: list[int]) -> list[int]:
    if len(values) <= MOVE_SAMPLE_LIMIT:
        random.shuffle(values)
        return values
    return random.sample(values, MOVE_SAMPLE_LIMIT)


def _candidate_moves(
    cur_pos: tuple[int, int],
    length: int,
    h: int,
    w: int,
    occupied: set[tuple[int, int]],
    reserved_lines: set[tuple[int, int]],
) -> list[dict[str, object]]:
    cur_r, cur_c = cur_pos
    moves: list[dict[str, object]] = []
    directions = list(DIRS)
    random.shuffle(directions)

    for direction in directions:
        antenna = _antenna_cells(cur_r, cur_c, direction, length)
        if not _valid_cells(antenna, h, w):
            continue
        if any(cell in occupied or cell in reserved_lines for cell in antenna):
            continue

        tip_r, tip_c = antenna[-1]
        if direction in {"L", "R"}:
            row_options = [cur_r - 1, cur_r, cur_r + 1]
            row_options = [r for r in row_options if 0 <= r <= h - 3]
            random.shuffle(row_options)
            if direction == "R":
                col_options = list(range(int(tip_c + 2), int(w - 2)))
            else:
                col_options = list(range(0, int(tip_c - 3)))
            col_options = _sample_axis(col_options)
            for nr in row_options:
                for nc in col_options:
                    square = _square_cells(nr, nc)
                    if square & occupied or square & reserved_lines:
                        continue
                    line_cells = _line_background_cells(cur_r, cur_c, direction, length, nr, nc)
                    if any(cell in occupied for cell in line_cells):
                        continue
                    moves.append(
                        {
                            "dir": direction,
                            "antenna": antenna,
                            "next_pos": (int(nr), int(nc)),
                            "line_cells": line_cells,
                            "crosses": any(cell in reserved_lines for cell in line_cells),
                        }
                    )
        else:
            col_options = [cur_c - 1, cur_c, cur_c + 1]
            col_options = [c for c in col_options if 0 <= c <= w - 3]
            random.shuffle(col_options)
            if direction == "D":
                row_options = list(range(int(tip_r + 2), int(h - 2)))
            else:
                row_options = list(range(0, int(tip_r - 3)))
            row_options = _sample_axis(row_options)
            for nc in col_options:
                for nr in row_options:
                    square = _square_cells(nr, nc)
                    if square & occupied or square & reserved_lines:
                        continue
                    line_cells = _line_background_cells(cur_r, cur_c, direction, length, nr, nc)
                    if any(cell in occupied for cell in line_cells):
                        continue
                    moves.append(
                        {
                            "dir": direction,
                            "antenna": antenna,
                            "next_pos": (int(nr), int(nc)),
                            "line_cells": line_cells,
                            "crosses": any(cell in reserved_lines for cell in line_cells),
                        }
                    )
    random.shuffle(moves)
    return moves


def _final_directions(
    pos: tuple[int, int],
    length: int,
    h: int,
    w: int,
    occupied: set[tuple[int, int]],
    reserved_lines: set[tuple[int, int]],
) -> list[tuple[str, list[tuple[int, int]]]]:
    result: list[tuple[str, list[tuple[int, int]]]] = []
    directions = list(DIRS)
    random.shuffle(directions)
    for direction in directions:
        antenna = _antenna_cells(pos[0], pos[1], direction, length)
        if _valid_cells(antenna, h, w) and not any(cell in occupied or cell in reserved_lines for cell in antenna):
            result.append((direction, antenna))
    return result


def _render_scene(
    positions: list[tuple[int, int]],
    lengths: list[int],
    directions: list[str],
    colors: list[int],
    h: int,
    w: int,
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for (r, c), length, direction, color in zip(positions, lengths, directions, colors):
        for cell in _square_cells(r, c):
            grid[cell] = color
        for cell in _antenna_cells(r, c, direction, length):
            grid[cell] = color
    return grid


def _validate_scene(
    grid: np.ndarray,
    positions: list[tuple[int, int]],
    lengths: list[int],
    directions: list[str],
    want_cross: bool,
) -> bool:
    try:
        parsed = _extract_objects(grid)
        parsed_sig = [
            (int(obj["length"]), int(obj["square_r"]), int(obj["square_c"]), str(obj["dir"]))
            for obj in parsed
        ]
        expected_sig = [
            (int(lengths[i]), int(positions[i][0]), int(positions[i][1]), directions[i])
            for i in range(len(positions))
        ]
        if parsed_sig != expected_sig:
            return False
        out = main(grid)
    except AssertionError:
        return False
    return bool(np.any(out == YELLOW)) == want_cross


def _gen_cross_attempt() -> np.ndarray | None:
    h = random.randint(20, 26)
    w = random.randint(20, 26)
    lengths = list(random.choice([[1, 1, 2, 3, 4], [1, 2, 2, 3, 4]]))
    colors = random.sample(INPUT_COLORS, 5)
    directions = ["R", "D", "L", "U"]

    ra = random.randint(4, 7)
    ca = random.randint(2, 4)
    l0, l1, l2, l3, l4 = lengths
    tip0 = _antenna_cells(ra, ca, "R", l0)[-1][1]
    y0 = ra + 1

    cb_min = tip0 + 5
    cb_max = w - 8
    if cb_min > cb_max:
        return None
    cb = random.randint(cb_min, cb_max)
    rb = random.choice([r for r in [ra - 1, ra, ra + 1] if 0 <= r <= h - 3])

    tip1 = _antenna_cells(rb, cb, "D", l1)[-1][0]
    rc_min = tip1 + 3
    rc_max = h - 7
    if rc_min > rc_max:
        return None
    rc = random.randint(rc_min, rc_max)
    cc_options = [c for c in [cb - 1, cb, cb + 1] if 0 <= c <= w - 3]
    if not cc_options:
        return None
    cc = random.choice(cc_options)

    tip2 = _antenna_cells(rc, cc, "L", l2)[-1][1]
    cd_min = tip0
    cd_max = min(cb - 2, tip2 - 4)
    if cd_min > cd_max:
        return None
    cd = random.randint(cd_min, cd_max)
    rd = random.choice([r for r in [rc - 1, rc, rc + 1] if 0 <= r <= h - 3])

    tip3_r, tip3_c = _antenna_cells(rd, cd, "U", l3)[-1]
    if not (tip0 + 1 <= tip3_c <= cb - 1):
        return None
    re_max = min(y0 - 3, tip3_r - 4)
    if re_max < 0:
        return None
    re = random.randint(0, re_max)
    ce_options = [c for c in [cd - 1, cd, cd + 1] if 0 <= c <= w - 3]
    if not ce_options:
        return None
    ce = random.choice(ce_options)

    positions = [(ra, ca), (rb, cb), (rc, cc), (rd, cd), (re, ce)]
    if not _ordered_ties_ok(positions, lengths):
        return None

    occupied: set[tuple[int, int]] = set()
    for pos in positions:
        square = _square_cells(pos[0], pos[1])
        if occupied & square:
            return None
        occupied |= square

    reserved: set[tuple[int, int]] = set()
    for i in range(4):
        antenna = set(_antenna_cells(positions[i][0], positions[i][1], directions[i], lengths[i]))
        if not _valid_cells(antenna, h, w):
            return None
        if (occupied - _square_cells(positions[i][0], positions[i][1])) & antenna:
            return None
        if reserved & antenna:
            return None
        occupied |= antenna

        line_cells = set(_line_background_cells(
            positions[i][0], positions[i][1], directions[i], lengths[i], positions[i + 1][0], positions[i + 1][1]
        ))
        if line_cells & occupied:
            return None
        reserved |= line_cells

    final_options = _final_directions(positions[4], l4, h, w, occupied, reserved)
    if not final_options:
        return None
    final_dir, final_antenna = random.choice(final_options)
    if reserved & set(final_antenna):
        return None

    full_dirs = directions + [final_dir]
    grid = _render_scene(positions, lengths, full_dirs, colors, h, w)
    if not _validate_scene(grid, positions, lengths, full_dirs, True):
        return None
    return grid


def _gen_plain_attempt() -> np.ndarray | None:
    allow_ties = bool(random.getrandbits(1))
    count_choices = [3, 4]
    count = random.choice(count_choices)

    h = random.randint(17, 23)
    w = random.randint(17, 23)
    lengths = _length_pattern(count, allow_ties)
    colors = random.sample(INPUT_COLORS, count)

    start_r = random.randint(2, h - 5)
    start_c = random.randint(2, w - 5)
    positions = [(int(start_r), int(start_c))]
    occupied = _square_cells(start_r, start_c)
    reserved_lines: set[tuple[int, int]] = set()
    directions: list[str | None] = [None] * count

    def search(index: int, occupied_now: set[tuple[int, int]], reserved_now: set[tuple[int, int]]) -> bool:
        if index == count - 1:
            for direction, antenna in _final_directions(
                positions[index], lengths[index], h, w, occupied_now, reserved_now
            ):
                directions[index] = direction
                return True
            directions[index] = None
            return False

        moves = _candidate_moves(positions[index], lengths[index], h, w, occupied_now, reserved_now)
        non_crossing = [move for move in moves if not bool(move["crosses"])]
        if non_crossing:
            moves = non_crossing

        for move in moves:
            next_pos = move["next_pos"]
            positions.append(next_pos)
            if not _ordered_ties_ok(positions, lengths[: len(positions)]):
                positions.pop()
                continue

            directions[index] = str(move["dir"])
            next_square = _square_cells(next_pos[0], next_pos[1])
            occ2 = occupied_now | set(move["antenna"]) | next_square
            res2 = reserved_now | set(move["line_cells"])

            if search(index + 1, occ2, res2):
                return True

            directions[index] = None
            positions.pop()
        return False

    if not search(0, occupied, reserved_lines):
        return None

    final_dirs = [direction for direction in directions if direction is not None]
    if len(final_dirs) != count:
        return None

    grid = _render_scene(positions, lengths, final_dirs, colors, h, w)

    try:
        if not _validate_scene(grid, positions, lengths, final_dirs, False):
            return None
    except AssertionError:
        return None
    return grid


def _gen_attempt() -> np.ndarray | None:
    if bool(random.getrandbits(1)):
        return _gen_cross_attempt()
    return _gen_plain_attempt()


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
