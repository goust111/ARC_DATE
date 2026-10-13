import random
from collections import Counter

import numpy as np


Color = type(
    "Color",
    (),
    {
        "BLACK": 0,
        "BLUE": 1,
        "RED": 2,
        "GREEN": 3,
        "YELLOW": 4,
        "PINK": 6,
    },
)


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude=()) -> list[int]:
    return sorted(set(map(int, grid.ravel())) - set(map(int, exclude)))


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _diag_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    dr = int(b[0] - a[0])
    dc = int(b[1] - a[1])
    assert dr != 0 and abs(dr) == abs(dc)
    sr = _sign(dr)
    sc = _sign(dc)
    return [(a[0] + k * sr, a[1] + k * sc) for k in range(abs(dr) + 1)]


def _parse_pointer_direction(grid: np.ndarray) -> str:
    red = np.argwhere(grid == Color.RED)
    assert red.size > 0, "missing red pointer"
    r0, c0, r1, c1 = bbox_coords(red)
    sub = grid[r0 : r1 + 1, c0 : c1 + 1] == Color.RED
    h, w = sub.shape

    if h == 2 and w > 2:
        col_counts = sub.sum(axis=0).astype(int).tolist()
        if col_counts[:2] == [2, 2] and all(v == 1 for v in col_counts[2:]):
            return "right"
        if col_counts[-2:] == [2, 2] and all(v == 1 for v in col_counts[:-2]):
            return "left"
    if w == 2 and h > 2:
        row_counts = sub.sum(axis=1).astype(int).tolist()
        if row_counts[:2] == [2, 2] and all(v == 1 for v in row_counts[2:]):
            return "down"
        if row_counts[-2:] == [2, 2] and all(v == 1 for v in row_counts[:-2]):
            return "up"
    raise AssertionError("red object must be a 2x2 body with a straight rod")


def _parse_board(grid: np.ndarray) -> tuple[tuple[int, int, int, int], int, list[int]]:
    movable_mask = (grid != Color.BLACK) & (grid != Color.RED)
    coords = np.argwhere(movable_mask)
    assert coords.size > 0, "missing checkerboard"
    r0, c0, r1, c1 = bbox_coords(coords)
    parity = int((coords[0, 0] + coords[0, 1]) % 2)
    marker_colors = colors_present(grid, exclude={Color.BLACK, Color.BLUE, Color.RED})
    movable_colors = {Color.BLUE, *marker_colors}

    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            v = int(grid[r, c])
            if (r + c) % 2 == parity:
                assert v in movable_colors, "every moving-parity cell in the board must be occupied"
            else:
                assert v == Color.BLACK, "every opposite-parity board cell must be black"

    return (r0, c0, r1, c1), parity, marker_colors


def _compress_board(
    grid: np.ndarray,
    board_bbox: tuple[int, int, int, int],
    direction: str,
    movable_colors: set[int],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    r0, c0, r1, c1 = board_bbox
    out[r0 : r1 + 1, c0 : c1 + 1] = Color.BLACK

    if direction in ("left", "right"):
        for r in range(r0, r1 + 1):
            vals = [int(grid[r, c]) for c in range(c0, c1 + 1) if int(grid[r, c]) in movable_colors]
            if direction == "left":
                target = list(range(c0, c0 + len(vals)))
            else:
                target = list(range(c1 - len(vals) + 1, c1 + 1))
            for c, v in zip(target, vals):
                out[r, c] = v
    else:
        for c in range(c0, c1 + 1):
            vals = [int(grid[r, c]) for r in range(r0, r1 + 1) if int(grid[r, c]) in movable_colors]
            if direction == "up":
                target = list(range(r0, r0 + len(vals)))
            else:
                target = list(range(r1 - len(vals) + 1, r1 + 1))
            for r, v in zip(target, vals):
                out[r, c] = v

    return out


def _connect_marker_pairs(grid: np.ndarray, marker_colors: list[int]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for color in marker_colors:
        coords = np.argwhere(out == int(color))
        assert coords.shape == (2, 2), "each marker color must appear exactly twice before linking"
        a = tuple(map(int, coords[0]))
        b = tuple(map(int, coords[1]))
        for r, c in _diag_cells(a, b):
            out[r, c] = int(color)
    return out


def _solve_with_direction(
    grid: np.ndarray,
    direction: str,
    board_bbox: tuple[int, int, int, int],
    marker_colors: list[int],
) -> np.ndarray:
    movable_colors = {Color.BLUE, *marker_colors}
    erased = np.asarray(grid, dtype=int).copy()
    erased[erased == Color.RED] = Color.BLACK
    compressed = _compress_board(erased, board_bbox, direction, movable_colors)
    return _connect_marker_pairs(compressed, marker_colors)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the red rod direction to compact the checkerboard, then connect each same-colored pair on the compressed board.

    Input structure:
    - A black background contains one checkerboard rectangle made from blue cells on one parity and black cells on the other.
    - Some blue cells are replaced by 2 or 3 marker-color pairs.
    - A separate red object consists of one 2x2 body plus a straight 1-cell-wide rod pointing toward one grid side.

    Rule:
    - Read the rod direction from the red object.
    - Compact every blue-or-marker cell of the checkerboard toward that direction, row-wise for left/right and column-wise for up/down, preserving marker colors.
    - Erase the entire red object to black, then connect each same-colored marker pair by filling the full diagonal segment between its two compressed endpoints.

    Key insight:
    - The red rod is not part of the final picture; it is a direction cue that changes where the marker endpoints land so the compressed board reveals the diagonals to draw.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    board_bbox, _parity, marker_colors = _parse_board(g)
    assert len(marker_colors) in (2, 3), "expected 2 or 3 marker pairs"
    direction = _parse_pointer_direction(g)
    return _solve_with_direction(g, direction, board_bbox, marker_colors)


def _source_positions_for_line(
    board_bbox: tuple[int, int, int, int],
    parity: int,
    direction: str,
    index: int,
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = board_bbox
    if direction in ("left", "right"):
        r = index
        return [(r, c) for c in range(c0, c1 + 1) if (r + c) % 2 == parity]
    c = index
    return [(r, c) for r in range(r0, r1 + 1) if (r + c) % 2 == parity]


def _destination_positions_for_line(
    board_bbox: tuple[int, int, int, int],
    direction: str,
    index: int,
    count: int,
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = board_bbox
    if direction == "left":
        return [(index, c0 + k) for k in range(count)]
    if direction == "right":
        return [(index, c1 - count + 1 + k) for k in range(count)]
    if direction == "up":
        return [(r0 + k, index) for k in range(count)]
    return [(r1 - count + 1 + k, index) for k in range(count)]


def _build_compaction_map(
    board_bbox: tuple[int, int, int, int],
    parity: int,
    direction: str,
) -> dict[tuple[int, int], tuple[int, int]]:
    mapping: dict[tuple[int, int], tuple[int, int]] = {}
    r0, c0, r1, c1 = board_bbox
    if direction in ("left", "right"):
        for r in range(r0, r1 + 1):
            src = _source_positions_for_line(board_bbox, parity, direction, r)
            dst = _destination_positions_for_line(board_bbox, direction, r, len(src))
            mapping.update(dict(zip(src, dst)))
    else:
        for c in range(c0, c1 + 1):
            src = _source_positions_for_line(board_bbox, parity, direction, c)
            dst = _destination_positions_for_line(board_bbox, direction, c, len(src))
            mapping.update(dict(zip(src, dst)))
    return mapping


def _enumerate_candidate_pairs(
    board_bbox: tuple[int, int, int, int],
    mapping: dict[tuple[int, int], tuple[int, int]],
) -> list[dict]:
    cells = list(mapping)
    candidates = []
    for i, a in enumerate(cells):
        for b in cells[i + 1 :]:
            src_dr = b[0] - a[0]
            src_dc = b[1] - a[1]
            if src_dr == 0 and src_dc == 0:
                continue
            da = mapping[a]
            db = mapping[b]
            dst_dr = db[0] - da[0]
            dst_dc = db[1] - da[1]
            if dst_dr == 0 or abs(dst_dr) != abs(dst_dc):
                continue
            if abs(dst_dr) < 2 or abs(dst_dr) > 5:
                continue
            source_is_diag = src_dr != 0 and abs(src_dr) == abs(src_dc)
            source_line = set(_diag_cells(a, b)) if source_is_diag else {a, b}
            dest_line = set(_diag_cells(da, db))
            if source_line == dest_line:
                continue
            candidates.append(
                {
                    "source": (a, b),
                    "dest": (da, db),
                    "source_line": source_line,
                    "dest_line": dest_line,
                    "moved": int(a != da) + int(b != db),
                    "length": len(dest_line),
                    "source_is_diag": source_is_diag,
                }
            )
    return candidates


def _choose_pair_set(candidates: list[dict], pair_count: int, rng: random.Random) -> list[dict] | None:
    pool = candidates[:]
    rng.shuffle(pool)
    pool.sort(key=lambda cand: (-cand["moved"], -cand["length"]))
    selected: list[dict] = []
    used_source: set[tuple[int, int]] = set()
    used_source_line: set[tuple[int, int]] = set()
    used_dest_line: set[tuple[int, int]] = set()

    def rec(start: int) -> bool:
        if len(selected) == pair_count:
            return True
        remaining = pair_count - len(selected)
        for idx in range(start, len(pool)):
            if len(pool) - idx < remaining:
                return False
            cand = pool[idx]
            src_cells = set(cand["source"])
            if src_cells & used_source:
                continue
            if cand["source_line"] & used_source_line:
                continue
            if cand["dest_line"] & used_dest_line:
                continue

            selected.append(cand)
            used_source.update(src_cells)
            used_source_line.update(cand["source_line"])
            used_dest_line.update(cand["dest_line"])
            if rec(idx + 1):
                return True
            selected.pop()
            used_source.difference_update(src_cells)
            used_source_line.difference_update(cand["source_line"])
            used_dest_line.difference_update(cand["dest_line"])
        return False

    return selected if rec(0) else None


def _paint_checkerboard(
    grid: np.ndarray,
    board_bbox: tuple[int, int, int, int],
    parity: int,
) -> None:
    r0, c0, r1, c1 = board_bbox
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            grid[r, c] = Color.BLUE if (r + c) % 2 == parity else Color.BLACK


def _place_pointer(
    grid: np.ndarray,
    board_bbox: tuple[int, int, int, int],
    direction: str,
    rod_len: int,
    rng: random.Random,
) -> None:
    h, w = grid.shape
    r0, c0, r1, c1 = board_bbox
    attach = rng.randrange(2)

    if direction == "left":
        body_c = rod_len
        row_hi = h - 2
        body_r = rng.randint(1, row_hi - 1)
        grid[body_r : body_r + 2, body_c : body_c + 2] = Color.RED
        grid[body_r + attach, :body_c] = Color.RED
        assert body_c + 2 < c0
    elif direction == "right":
        body_c = w - rod_len - 2
        row_hi = h - 2
        body_r = rng.randint(1, row_hi - 1)
        grid[body_r : body_r + 2, body_c : body_c + 2] = Color.RED
        grid[body_r + attach, body_c + 2 :] = Color.RED
        assert c1 < body_c - 1
    elif direction == "up":
        body_r = rod_len
        col_hi = w - 2
        body_c = rng.randint(1, col_hi - 1)
        grid[body_r : body_r + 2, body_c : body_c + 2] = Color.RED
        grid[:body_r, body_c + attach] = Color.RED
        assert body_r + 2 < r0
    else:
        body_r = h - rod_len - 2
        col_hi = w - 2
        body_c = rng.randint(1, col_hi - 1)
        grid[body_r : body_r + 2, body_c : body_c + 2] = Color.RED
        grid[body_r + 2 :, body_c + attach] = Color.RED
        assert r1 < body_r - 1


def _wrong_direction_outputs(
    grid: np.ndarray,
    board_bbox: tuple[int, int, int, int],
    marker_colors: list[int],
    correct_direction: str,
) -> list[np.ndarray | None]:
    out = []
    for direction in ("left", "right", "up", "down"):
        if direction == correct_direction:
            continue
        try:
            out.append(_solve_with_direction(grid, direction, board_bbox, marker_colors))
        except AssertionError:
            out.append(None)
    return out


def _validate_generated_case(
    grid: np.ndarray,
    board_bbox: tuple[int, int, int, int],
    marker_colors: list[int],
    direction: str,
) -> None:
    out = main(grid)
    assert max(grid.shape) <= 30 and max(out.shape) <= 30
    assert not np.array_equal(grid, out)

    erased = np.asarray(grid, dtype=int).copy()
    erased[erased == Color.RED] = Color.BLACK
    assert not np.array_equal(erased, out), "the transform must do more than erase the pointer"

    compressed_only = _compress_board(erased, board_bbox, direction, {Color.BLUE, *marker_colors})
    assert not np.array_equal(compressed_only, out), "pair-linking step must matter"

    for wrong in _wrong_direction_outputs(grid, board_bbox, marker_colors, direction):
        if wrong is not None:
            assert not np.array_equal(wrong, out), "pointer direction must be uniquely decisive"

    for color in marker_colors:
        src = sorted(map(tuple, np.argwhere(grid == color)))
        dst = sorted(map(tuple, np.argwhere(compressed_only == color)))
        assert len(src) == 2 and len(dst) == 2
        dr = dst[1][0] - dst[0][0]
        dc = dst[1][1] - dst[0][1]
        assert dr != 0 and abs(dr) == abs(dc) and abs(dr) >= 2
        src_dr = src[1][0] - src[0][0]
        src_dc = src[1][1] - src[0][1]
        if src_dr != 0 and abs(src_dr) == abs(src_dc):
            assert set(_diag_cells(src[0], src[1])) != set(_diag_cells(dst[0], dst[1]))


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(13, 18)
    w = rng.randint(13, 18)
    direction = rng.choice(["left", "right", "up", "down"])
    rod_len = rng.randint(2, 4)
    pair_count = rng.choice([2, 3])
    marker_colors = [Color.GREEN, Color.YELLOW] + ([Color.PINK] if pair_count == 3 else [])
    parity = rng.randint(0, 1)

    max_board_h = min(11, h - 3)
    max_board_w = min(11, w - 3)
    if max_board_h < 7 or max_board_w < 7:
        return None

    side_gap = rod_len + 3
    for _ in range(80):
        board_h = rng.randint(7, max_board_h)
        board_w = rng.randint(7, max_board_w)

        r0_lo, c0_lo = 1, 1
        r0_hi = h - board_h - 1
        c0_hi = w - board_w - 1
        if direction == "left":
            c0_lo = side_gap
        elif direction == "right":
            c0_hi = w - board_w - side_gap
        elif direction == "up":
            r0_lo = side_gap
        else:
            r0_hi = h - board_h - side_gap

        if r0_lo > r0_hi or c0_lo > c0_hi:
            continue
        r0 = rng.randint(r0_lo, r0_hi)
        c0 = rng.randint(c0_lo, c0_hi)
        board_bbox = (r0, c0, r0 + board_h - 1, c0 + board_w - 1)
        mapping = _build_compaction_map(board_bbox, parity, direction)
        candidates = _enumerate_candidate_pairs(board_bbox, mapping)
        if len(candidates) < pair_count:
            continue
        chosen = _choose_pair_set(candidates, pair_count, rng)
        if chosen is None:
            continue

        grid = np.zeros((h, w), dtype=int)
        _paint_checkerboard(grid, board_bbox, parity)
        for color, cand in zip(marker_colors, chosen):
            for cell in cand["source"]:
                grid[cell] = color
        _place_pointer(grid, board_bbox, direction, rod_len, rng)

        try:
            _validate_generated_case(grid, board_bbox, marker_colors, direction)
        except AssertionError:
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
