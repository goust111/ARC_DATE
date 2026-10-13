from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5
ORANGE = 7


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _frame_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    mask = grid == ORANGE
    box = _bbox(mask)
    r0, c0, r1, c1 = box
    assert r1 - r0 >= 6 and c1 - c0 >= 7, "frame must enclose two nontrivial halves"

    expected = np.zeros_like(mask, dtype=bool)
    expected[r0, c0 : c1 + 1] = True
    expected[r1, c0 : c1 + 1] = True
    expected[r0 : r1 + 1, c0] = True
    expected[r0 : r1 + 1, c1] = True
    assert np.array_equal(mask, expected), "orange pixels must form one hollow rectangle"
    return box


def _separator_row(grid: np.ndarray, box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = box
    rows = [
        r
        for r in range(r0 + 1, r1)
        if np.all(grid[r, c0 + 1 : c1] == GREY)
    ]
    assert len(rows) == 1, "expected exactly one full grey separator row"
    return int(rows[0])


def _markers(
    grid: np.ndarray, box: tuple[int, int, int, int], sep_r: int
) -> list[tuple[int, int, int]]:
    r0, c0, r1, c1 = box
    coords = np.argwhere((grid == BLUE) | (grid == RED))
    markers = [(int(r), int(c), int(grid[r, c])) for r, c in coords]
    assert 4 <= len(markers) <= 8, "expected 4..8 control markers"

    cols = set()
    above = 0
    below = 0
    for r, c, color in markers:
        assert color in (BLUE, RED)
        assert r0 < r < r1 and c0 < c < c1, "markers must lie inside the frame"
        assert r != sep_r, "markers cannot lie on the separator"
        assert c not in cols, "marker columns must be unique"
        cols.add(c)
        if r < sep_r:
            above += 1
        else:
            below += 1
    assert above >= 2 and below >= 2, "each half must contain at least two markers"

    allowed = {BLACK, BLUE, RED, GREY, ORANGE}
    assert set(map(int, np.unique(grid))) <= allowed, "unexpected colors present"
    return markers


def _draw_ray(
    out: np.ndarray,
    box: tuple[int, int, int, int],
    sep_r: int,
    marker: tuple[int, int, int],
) -> None:
    r0, _c0, r1, _c1 = box
    r, c, color = marker
    if color == RED:
        step = 1 if r < sep_r else -1
        rows = range(r + step, sep_r, step)
    else:
        if r < sep_r:
            rows = range(r - 1, r0, -1)
        else:
            rows = range(r + 1, r1, 1)
    for rr in rows:
        out[int(rr), int(c)] = int(color)


def _vote_fill(left_color: int | None, right_color: int | None) -> int | None:
    red_votes = int(left_color == RED) + int(right_color == RED)
    blue_votes = int(left_color == BLUE) + int(right_color == BLUE)
    if red_votes == 0 and blue_votes == 0:
        return None
    if red_votes > blue_votes:
        return RED
    if blue_votes > red_votes:
        return BLUE
    return GREY


def _fill_half(
    out: np.ndarray,
    row_lo: int,
    row_hi: int,
    box: tuple[int, int, int, int],
    half_markers: list[tuple[int, int, int]],
) -> None:
    if row_lo > row_hi:
        return

    _r0, c0, _r1, c1 = box
    boundaries = [(int(c0), None)]
    boundaries.extend(sorted((int(c), int(color)) for _r, c, color in half_markers))
    boundaries.append((int(c1), None))

    for (left_c, left_color), (right_c, right_color) in zip(boundaries, boundaries[1:]):
        if right_c - left_c <= 1:
            continue
        fill_color = _vote_fill(left_color, right_color)
        if fill_color is None:
            continue
        patch = out[row_lo : row_hi + 1, left_c + 1 : right_c]
        patch[patch == BLACK] = int(fill_color)


def _apply_spec(
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    sep_r: int,
    markers: list[tuple[int, int, int]],
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for marker in markers:
        _draw_ray(out, box, sep_r, marker)

    top_markers = [m for m in markers if m[0] < sep_r]
    bottom_markers = [m for m in markers if m[0] > sep_r]
    r0, c0, r1, c1 = box
    _fill_half(out, r0 + 1, sep_r - 1, box, top_markers)
    _fill_half(out, sep_r + 1, r1 - 1, box, bottom_markers)

    out[sep_r, c0 + 1 : c1] = BLACK
    for r, c, _color in markers:
        out[r, c] = BLACK
    return out


def _strip_votes(
    box: tuple[int, int, int, int], half_markers: list[tuple[int, int, int]]
) -> list[int]:
    _r0, c0, _r1, c1 = box
    boundaries = [(int(c0), None)]
    boundaries.extend(sorted((int(c), int(color)) for _r, c, color in half_markers))
    boundaries.append((int(c1), None))
    votes: list[int] = []
    for (left_c, left_color), (right_c, right_color) in zip(boundaries, boundaries[1:]):
        if right_c - left_c <= 1:
            continue
        fill = _vote_fill(left_color, right_color)
        if fill is not None:
            votes.append(int(fill))
    return votes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn directional red/blue control pixels into a ray-partitioned band fill inside the orange frame.

    Input structure:
    - A black background contains one orange hollow rectangle.
    - Inside the frame, a single grey horizontal separator spans from the left inner wall to the right inner wall.
    - Red and blue isolated marker pixels lie strictly above or below the separator, with unique columns.

    Rule:
    - From each red marker, draw a vertical red ray toward the grey separator, stopping one cell before it and not overwriting the marker itself.
    - From each blue marker, draw a vertical blue ray away from the separator toward the nearest orange wall, again stopping before the wall and not overwriting the marker.
    - In the upper half and lower half separately, treat the sorted marker columns together with the two orange side walls as vertical strip boundaries; fill each strip's black cells by voting on its left and right boundary colors: red-majority gives red, blue-majority gives blue, ties give grey, and wall-wall contributes no fill.
    - Erase the grey separator and the original marker pixels to black, while keeping the orange frame, the ray segments, and the strip fills.

    Key insight:
    - The markers matter twice: first they determine where directional rays are drawn, and then their columns become the conceptual boundaries that slice each half into strips whose fill color is decided by the two neighboring boundary rays.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    box = _frame_bbox(g)
    sep_r = _separator_row(g, box)
    markers = _markers(g, box, sep_r)
    out = _apply_spec(g, box, sep_r, markers)
    assert out.shape == g.shape
    return out


def _sample_dims() -> tuple[int, int, int, int, int, int]:
    inner_w = random.randint(12, 18)
    top_h = random.randint(4, 7)
    bot_h = random.randint(4, 7)
    frame_h = top_h + bot_h + 3
    frame_w = inner_w + 2

    grid_h = random.randint(frame_h + 2, min(30, frame_h + 8))
    grid_w = random.randint(frame_w + 2, min(30, frame_w + 8))

    r0 = random.randint(1, grid_h - frame_h - 1)
    c0 = random.randint(1, grid_w - frame_w - 1)
    return grid_h, grid_w, r0, c0, top_h, bot_h


def _choose_columns(inner_w: int, count: int) -> list[int] | None:
    rel_cols = list(range(1, inner_w - 1))
    random.shuffle(rel_cols)
    chosen: list[int] = []
    for rel in rel_cols:
        if all(abs(rel - prev) >= 2 for prev in chosen):
            chosen.append(int(rel))
            if len(chosen) == count:
                break
    if len(chosen) != count:
        return None
    chosen.sort()
    return chosen


def _color_sequence(kind: str, count: int) -> list[int]:
    if kind == "mono_red":
        return [RED] * count
    if kind == "mono_blue":
        return [BLUE] * count
    if kind == "alternating_red":
        return [RED if i % 2 == 0 else BLUE for i in range(count)]
    if kind == "alternating_blue":
        return [BLUE if i % 2 == 0 else RED for i in range(count)]
    if kind == "rich":
        if count == 2:
            return [RED, BLUE]
        if count == 3:
            return random.choice(([RED, RED, BLUE], [BLUE, BLUE, RED], [RED, BLUE, BLUE], [BLUE, RED, RED]))
        if count == 4:
            return random.choice(
                (
                    [RED, RED, BLUE, RED],
                    [BLUE, BLUE, RED, BLUE],
                    [RED, BLUE, BLUE, RED],
                    [BLUE, RED, RED, BLUE],
                )
            )
    raise AssertionError("unsupported color-sequence request")


def _sample_patterns(n_top: int, n_bottom: int) -> tuple[list[int], list[int]]:
    scenario = random.choices(
        ["rich", "mixed", "mono_split"],
        weights=[0.45, 0.35, 0.20],
        k=1,
    )[0]

    if scenario == "rich":
        rich_top = n_top >= 3 and (n_bottom < 3 or random.random() < 0.5)
        if rich_top:
            top = _color_sequence("rich", n_top)
            bottom_kind = random.choice(
                ["alternating_red", "alternating_blue", "mono_red", "mono_blue"]
            )
            bottom = _color_sequence(bottom_kind, n_bottom)
        else:
            bottom = _color_sequence("rich", n_bottom)
            top_kind = random.choice(
                ["alternating_red", "alternating_blue", "mono_red", "mono_blue"]
            )
            top = _color_sequence(top_kind, n_top)
    elif scenario == "mixed":
        top = _color_sequence(random.choice(["alternating_red", "alternating_blue"]), n_top)
        bottom = _color_sequence(random.choice(["alternating_red", "alternating_blue"]), n_bottom)
    else:
        if random.random() < 0.5:
            top = _color_sequence("mono_red", n_top)
            bottom = _color_sequence("mono_blue", n_bottom)
        else:
            top = _color_sequence("mono_blue", n_top)
            bottom = _color_sequence("mono_red", n_bottom)

    all_colors = set(top) | set(bottom)
    if all_colors != {RED, BLUE}:
        return _sample_patterns(n_top, n_bottom)
    return top, bottom


def _sample_rows(r0: int, sep_r: int, r1: int, n_top: int, n_bottom: int) -> tuple[list[int], list[int]]:
    top_rows = [random.randint(r0 + 2, sep_r - 2) for _ in range(n_top)]
    bottom_rows = [random.randint(sep_r + 2, r1 - 2) for _ in range(n_bottom)]
    return top_rows, bottom_rows


def _build_input_from_spec(
    grid_h: int,
    grid_w: int,
    r0: int,
    c0: int,
    top_h: int,
    bot_h: int,
    top_cols: list[int],
    bot_cols: list[int],
    top_colors: list[int],
    bot_colors: list[int],
    top_rows: list[int],
    bot_rows: list[int],
) -> tuple[np.ndarray, tuple[int, int, int, int], int, list[tuple[int, int, int]]]:
    frame_h = top_h + bot_h + 3
    inner_w = max(top_cols + bot_cols) + 2
    frame_w = inner_w + 2
    r1 = r0 + frame_h - 1
    c1 = c0 + frame_w - 1
    sep_r = r0 + 1 + top_h

    grid = np.zeros((grid_h, grid_w), dtype=int)
    grid[r0, c0 : c1 + 1] = ORANGE
    grid[r1, c0 : c1 + 1] = ORANGE
    grid[r0 : r1 + 1, c0] = ORANGE
    grid[r0 : r1 + 1, c1] = ORANGE
    grid[sep_r, c0 + 1 : c1] = GREY

    markers: list[tuple[int, int, int]] = []
    for rel_c, color, row in zip(top_cols, top_colors, top_rows):
        col = c0 + rel_c
        grid[row, col] = int(color)
        markers.append((int(row), int(col), int(color)))
    for rel_c, color, row in zip(bot_cols, bot_colors, bot_rows):
        col = c0 + rel_c
        grid[row, col] = int(color)
        markers.append((int(row), int(col), int(color)))

    return grid, (r0, c0, r1, c1), sep_r, markers


def _semantic_checks(
    inp: np.ndarray,
    out: np.ndarray,
    box: tuple[int, int, int, int],
    sep_r: int,
    markers: list[tuple[int, int, int]],
) -> bool:
    r0, c0, r1, c1 = box
    top = [m for m in markers if m[0] < sep_r]
    bottom = [m for m in markers if m[0] > sep_r]
    top_votes = _strip_votes(box, top)
    bottom_votes = _strip_votes(box, bottom)
    all_votes = top_votes + bottom_votes

    if not all_votes:
        return False
    if out[sep_r, c0 + 1 : c1].any():
        return False
    if not np.any(out == RED) or not np.any(out == BLUE):
        return False
    if np.array_equal(inp, out):
        return False
    if sum(v == GREY for v in all_votes) == 0 and all(v in (RED, BLUE) for v in all_votes):
        if not (
            any(top_votes[i] != top_votes[i + 1] for i in range(len(top_votes) - 1))
            or any(bottom_votes[i] != bottom_votes[i + 1] for i in range(len(bottom_votes) - 1))
        ):
            return False

    # Ensure the two-sided vote really matters in at least one strip.
    boundary_effect = False
    for half_markers in (top, bottom):
        seq = [color for _r, _c, color in sorted(half_markers, key=lambda x: x[1])]
        if len(seq) >= 3:
            has_same = any(seq[i] == seq[i + 1] for i in range(len(seq) - 1))
            has_diff = any(seq[i] != seq[i + 1] for i in range(len(seq) - 1))
            if has_same and has_diff:
                boundary_effect = True
                break
    if not boundary_effect and not (set(top_votes + bottom_votes) >= {RED, GREY} or set(top_votes + bottom_votes) >= {BLUE, GREY}):
        return False

    inside = out[r0 + 1 : r1, c0 + 1 : c1]
    if np.count_nonzero(inside) < (r1 - r0 + c1 - c0):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    grid_h, grid_w, r0, c0, top_h, bot_h = _sample_dims()
    inner_w = random.randint(12, min(18, grid_w - c0 - 3))
    frame_w = inner_w + 2
    if c0 + frame_w >= grid_w:
        return None

    capacity = (inner_w - 2 + 1) // 2
    max_total = min(8, capacity)
    if max_total < 4:
        return None

    total = random.randint(4, max_total)
    n_top_min = max(2, total - 4)
    n_top_max = min(4, total - 2)
    if n_top_min > n_top_max:
        return None
    n_top = random.randint(n_top_min, n_top_max)
    n_bottom = total - n_top

    cols = _choose_columns(inner_w, total)
    if cols is None:
        return None
    top_idx = set(random.sample(range(total), n_top))
    top_cols = [cols[i] for i in range(total) if i in top_idx]
    bot_cols = [cols[i] for i in range(total) if i not in top_idx]
    top_cols.sort()
    bot_cols.sort()

    top_colors, bot_colors = _sample_patterns(n_top, n_bottom)
    top_rows, bot_rows = _sample_rows(r0, r0 + 1 + top_h, r0 + top_h + bot_h + 2, n_top, n_bottom)

    inp, box, sep_r, markers = _build_input_from_spec(
        grid_h,
        grid_w,
        r0,
        c0,
        top_h,
        bot_h,
        top_cols,
        bot_cols,
        top_colors,
        bot_colors,
        top_rows,
        bot_rows,
    )

    oracle = _apply_spec(inp, box, sep_r, markers)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, oracle):
        return None
    if not _semantic_checks(inp, oracle, box, sep_r, markers):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-partition fill input")
