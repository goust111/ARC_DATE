from __future__ import annotations

import random

import numpy as np


GREY = 5
PALETTE = tuple(range(10))
KIND_DIRS = {
    "H": (0, 1),
    "V": (1, 0),
    "D1": (1, 1),
    "D2": (1, -1),
}
GEN_KINDS = ("H", "V", "D1")
MAX_GENERATION_ATTEMPTS = 300


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _color_coords(grid: np.ndarray, color: int) -> list[tuple[int, int]]:
    return [tuple(map(int, rc)) for rc in np.argwhere(grid == color)]


def _raw_orientation(coords: list[tuple[int, int]]) -> str | None:
    if len(coords) <= 1:
        return "POINT"
    rows = {r for r, _c in coords}
    cols = {c for _r, c in coords}
    diffs = {r - c for r, c in coords}
    sums = {r + c for r, c in coords}
    if len(rows) == 1:
        return "H"
    if len(cols) == 1:
        return "V"
    if len(diffs) == 1:
        return "D1"
    if len(sums) == 1:
        return "D2"
    return None


def _sort_on_line(coords: list[tuple[int, int]], kind: str) -> list[tuple[int, int]]:
    if kind == "H":
        return sorted(coords, key=lambda p: p[1])
    if kind == "V":
        return sorted(coords, key=lambda p: p[0])
    if kind == "D1":
        return sorted(coords, key=lambda p: (p[0], p[1]))
    if kind == "D2":
        return sorted(coords, key=lambda p: (p[0], -p[1]))
    raise ValueError(f"unknown kind: {kind}")


def _line_id(point: tuple[int, int], kind: str) -> int:
    r, c = point
    if kind == "H":
        return r
    if kind == "V":
        return c
    if kind == "D1":
        return r - c
    if kind == "D2":
        return r + c
    raise ValueError(f"unknown kind: {kind}")


def _verify_fragment(coords: list[tuple[int, int]], kind: str) -> None:
    if not coords:
        raise ValueError("empty fragment")
    if len(coords) == 1:
        return
    if _raw_orientation(coords) != kind:
        raise ValueError("fragment orientation mismatch")
    ordered = _sort_on_line(coords, kind)
    dr, dc = KIND_DIRS[kind]
    for prev, cur in zip(ordered, ordered[1:]):
        if (cur[0] - prev[0], cur[1] - prev[1]) != (dr, dc):
            raise ValueError("fragment must be a contiguous line segment")


def _same_line(a: tuple[int, int], b: tuple[int, int], kind: str) -> bool:
    return _line_id(a, kind) == _line_id(b, kind)


def _inclusive_span(a: tuple[int, int], b: tuple[int, int], kind: str) -> int | None:
    if not _same_line(a, b, kind):
        return None
    if kind == "H":
        return abs(a[1] - b[1]) + 1
    if kind == "V":
        return abs(a[0] - b[0]) + 1
    if kind in {"D1", "D2"} and abs(a[0] - b[0]) == abs(a[1] - b[1]):
        return abs(a[0] - b[0]) + 1
    return None


def _ordered_endpoints(
    a: tuple[int, int], b: tuple[int, int], kind: str
) -> tuple[tuple[int, int], tuple[int, int]]:
    if kind == "H":
        pts = sorted((a, b), key=lambda p: p[1])
    elif kind == "V":
        pts = sorted((a, b), key=lambda p: p[0])
    elif kind == "D1":
        pts = sorted((a, b), key=lambda p: (p[0], p[1]))
    elif kind == "D2":
        pts = sorted((a, b), key=lambda p: (p[0], -p[1]))
    else:
        raise ValueError(f"unknown kind: {kind}")
    return pts[0], pts[1]


def _step_between(
    start: tuple[int, int], end: tuple[int, int]
) -> tuple[int, int]:
    span = max(abs(end[0] - start[0]), abs(end[1] - start[1])) + 1
    if span == 1:
        return 0, 0
    return (
        (end[0] - start[0]) // (span - 1),
        (end[1] - start[1]) // (span - 1),
    )


def _find_kind_and_fragments(
    grid: np.ndarray, bg: int
) -> tuple[str, list[dict[str, object]]]:
    fragment_colors = [
        int(color) for color in np.unique(grid) if int(color) not in {bg, GREY}
    ]
    if not fragment_colors:
        raise ValueError("expected at least one colored fragment")

    raw_kinds: set[str] = set()
    raw_data: list[tuple[int, list[tuple[int, int]], str | None]] = []
    for color in fragment_colors:
        coords = _color_coords(grid, color)
        raw_kind = _raw_orientation(coords)
        if raw_kind is None:
            raise ValueError("fragment is not a straight segment")
        if raw_kind != "POINT":
            raw_kinds.add(raw_kind)
        raw_data.append((color, coords, raw_kind))

    if len(raw_kinds) == 1:
        kind = next(iter(raw_kinds))
    elif len(raw_kinds) > 1:
        raise ValueError("all fragments must be parallel")
    else:
        gray_coords = _color_coords(grid, GREY)
        total_len = sum(len(coords) for _color, coords, _kind in raw_data)
        possible: list[str] = []
        for candidate in ("H", "V", "D1", "D2"):
            found = False
            for i, a in enumerate(gray_coords):
                for b in gray_coords[i + 1 :]:
                    if _inclusive_span(a, b, candidate) == total_len:
                        possible.append(candidate)
                        found = True
                        break
                if found:
                    break
        if len(set(possible)) != 1:
            raise ValueError("unable to infer fragment direction")
        kind = possible[0]

    fragments: list[dict[str, object]] = []
    for color, coords, _raw_kind in raw_data:
        _verify_fragment(coords, kind)
        fragments.append(
            {
                "color": color,
                "coords": coords,
                "length": len(coords),
                "offset": _line_id(coords[0], kind),
                "line": _line_id(coords[0], kind),
            }
        )
    fragments.sort(key=lambda frag: int(frag["offset"]))
    return kind, fragments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble parallel colored fragments onto the gray stake line in offset order.

    Input structure:
    - The most common color is the background.
    - Every non-background, non-gray color forms one straight fragment, and all
      fragments are parallel horizontal, vertical, or diagonal segments.
    - Exactly two gray pixels lie on another parallel line whose inclusive span
      equals the total fragment length; extra gray dots may appear elsewhere as
      distractors.

    Rule:
    - Remove all colored fragments from their original lines.
    - Find the unique gray-pixel pair that lies on a fragment-parallel line with
      the correct total span and no colored fragment on that line.
    - Starting from the upper-left/left/top endpoint of that pair, redraw the
      fragments end-to-end on the stake line, sorted by their carrier-line
      offset from top to bottom or left to right.

    Key insight:
    - The fragment colors and lengths are preserved; only the ordering matters,
      and that ordering is encoded by which parallel line each fragment occupies.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    kind, fragments = _find_kind_and_fragments(g, bg)
    total_len = sum(int(frag["length"]) for frag in fragments)

    gray_coords = _color_coords(g, GREY)
    fragment_lines = {int(frag["line"]) for frag in fragments}
    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for i, a in enumerate(gray_coords):
        for b in gray_coords[i + 1 :]:
            if _inclusive_span(a, b, kind) != total_len:
                continue
            line = _line_id(a, kind)
            if line in fragment_lines:
                continue
            if sum(1 for p in gray_coords if _line_id(p, kind) == line) != 2:
                continue
            candidates.append((a, b))
    if len(candidates) != 1:
        raise ValueError("expected a unique stake pair")

    start, end = _ordered_endpoints(candidates[0][0], candidates[0][1], kind)
    step_r, step_c = _step_between(start, end)

    out = g.copy()
    for frag in fragments:
        rr, cc = np.array(frag["coords"], dtype=int).T
        out[rr, cc] = bg

    cursor_r, cursor_c = start
    for frag in fragments:
        for _ in range(int(frag["length"])):
            out[cursor_r, cursor_c] = int(frag["color"])
            cursor_r += step_r
            cursor_c += step_c
    return out


def _d1_origin(shape: tuple[int, int], line_id: int) -> tuple[int, int]:
    _h, _w = shape
    return max(line_id, 0), max(-line_id, 0)


def _d1_line_length(shape: tuple[int, int], line_id: int) -> int:
    h, w = shape
    r0, c0 = _d1_origin(shape, line_id)
    return min(h - r0, w - c0)


def _choose_spaced_values(values: list[int], count: int, gap: int) -> list[int] | None:
    if len(values) < count:
        return None
    for _ in range(200):
        sample = sorted(random.sample(values, count))
        if all(b - a >= gap for a, b in zip(sample, sample[1:])):
            return sample
    return None


def _choose_bg() -> int:
    return random.choice([color for color in PALETTE if color != GREY])


def _choose_fragment_colors(bg: int, count: int) -> list[int]:
    pool = [color for color in PALETTE if color not in {bg, GREY}]
    return random.sample(pool, count)


def _paint_horizontal(
    grid: np.ndarray, row: int, start_c: int, length: int, color: int
) -> None:
    grid[row, start_c : start_c + length] = color


def _paint_vertical(
    grid: np.ndarray, col: int, start_r: int, length: int, color: int
) -> None:
    grid[start_r : start_r + length, col] = color


def _paint_d1(
    grid: np.ndarray, line_id: int, start_t: int, length: int, color: int
) -> tuple[int, int]:
    r0, c0 = _d1_origin(grid.shape, line_id)
    for i in range(length):
        grid[r0 + start_t + i, c0 + start_t + i] = color
    return r0 + start_t, c0 + start_t


def _reassembled_output(
    shape: tuple[int, int],
    bg: int,
    grey_points: list[tuple[int, int]],
    kind: str,
    start: tuple[int, int],
    fragments: list[tuple[int, int]],
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for r, c in grey_points:
        out[r, c] = GREY
    dr, dc = KIND_DIRS[kind]
    cursor_r, cursor_c = start
    for length, color in fragments:
        for _ in range(length):
            out[cursor_r, cursor_c] = color
            cursor_r += dr
            cursor_c += dc
    return out


def _oracle_checked_input(inp: np.ndarray, out: np.ndarray) -> np.ndarray | None:
    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def _gen_horizontal() -> np.ndarray | None:
    count = random.randint(2, 5)
    lengths = [random.randint(2, 5) for _ in range(count)]
    total_len = sum(lengths)
    h = random.randint(max(12, 2 * count + 5), 24)
    w_min = max(total_len + 2, 12)
    w = random.randint(w_min, min(30, w_min + 10))

    line_rows = _choose_spaced_values(list(range(1, h - 1)), count + 1, gap=2)
    if line_rows is None:
        return None
    mother_row = random.choice(line_rows)
    fragment_rows = sorted(row for row in line_rows if row != mother_row)

    bg = _choose_bg()
    colors = _choose_fragment_colors(bg, count)
    ordered_fragments = list(zip(lengths, colors, fragment_rows))

    inp = np.full((h, w), bg, dtype=int)
    start_c = random.randint(0, w - total_len)
    inp[mother_row, start_c] = GREY
    inp[mother_row, start_c + total_len - 1] = GREY

    for length, color, row in ordered_fragments:
        seg_start = random.randint(0, w - length)
        _paint_horizontal(inp, row, seg_start, length, color)

    distractor_rows = [row for row in range(h) if row not in line_rows]
    grey_points: list[tuple[int, int]] = []
    for row in random.sample(distractor_rows, k=min(len(distractor_rows), random.randint(0, 3))):
        col = random.randint(0, w - 1)
        inp[row, col] = GREY
        grey_points.append((row, col))

    out = _reassembled_output(
        inp.shape,
        bg,
        grey_points,
        "H",
        (mother_row, start_c),
        [(length, color) for length, color, _row in ordered_fragments],
    )
    return _oracle_checked_input(inp, out)


def _gen_vertical() -> np.ndarray | None:
    count = random.randint(2, 5)
    lengths = [random.randint(2, 5) for _ in range(count)]
    total_len = sum(lengths)
    w = random.randint(max(12, 2 * count + 5), 24)
    h_min = max(total_len + 2, 12)
    h = random.randint(h_min, min(30, h_min + 10))

    line_cols = _choose_spaced_values(list(range(1, w - 1)), count + 1, gap=2)
    if line_cols is None:
        return None
    mother_col = random.choice(line_cols)
    fragment_cols = sorted(col for col in line_cols if col != mother_col)

    bg = _choose_bg()
    colors = _choose_fragment_colors(bg, count)
    ordered_fragments = list(zip(lengths, colors, fragment_cols))

    inp = np.full((h, w), bg, dtype=int)
    start_r = random.randint(0, h - total_len)
    inp[start_r, mother_col] = GREY
    inp[start_r + total_len - 1, mother_col] = GREY

    for length, color, col in ordered_fragments:
        seg_start = random.randint(0, h - length)
        _paint_vertical(inp, col, seg_start, length, color)

    distractor_cols = [col for col in range(w) if col not in line_cols]
    grey_points: list[tuple[int, int]] = []
    for col in random.sample(distractor_cols, k=min(len(distractor_cols), random.randint(0, 3))):
        row = random.randint(0, h - 1)
        inp[row, col] = GREY
        grey_points.append((row, col))

    out = _reassembled_output(
        inp.shape,
        bg,
        grey_points,
        "V",
        (start_r, mother_col),
        [(length, color) for length, color, _col in ordered_fragments],
    )
    return _oracle_checked_input(inp, out)


def _gen_d1() -> np.ndarray | None:
    count = random.randint(2, 4)
    lengths = [random.randint(2, 4) for _ in range(count)]
    total_len = sum(lengths)
    h = random.randint(max(14, total_len + 2), 24)
    w = random.randint(max(14, total_len + 2), 24)

    candidates = [
        line_id
        for line_id in range(-(w - 1), h)
        if _d1_line_length((h, w), line_id) >= 2
    ]
    line_ids = _choose_spaced_values(candidates, count + 1, gap=2)
    if line_ids is None:
        return None

    feasible_mothers = [
        line_id for line_id in line_ids if _d1_line_length((h, w), line_id) >= total_len
    ]
    if not feasible_mothers:
        return None
    mother_line = random.choice(feasible_mothers)
    fragment_lines = sorted(line_id for line_id in line_ids if line_id != mother_line)

    bg = _choose_bg()
    colors = _choose_fragment_colors(bg, count)
    ordered_fragments = list(zip(lengths, colors, fragment_lines))

    inp = np.full((h, w), bg, dtype=int)
    mother_len = _d1_line_length(inp.shape, mother_line)
    mother_t = random.randint(0, mother_len - total_len)
    start_r, start_c = _paint_d1(inp, mother_line, mother_t, 1, GREY)
    end_r, end_c = _paint_d1(inp, mother_line, mother_t + total_len - 1, 1, GREY)
    inp[start_r, start_c] = GREY
    inp[end_r, end_c] = GREY

    for length, color, line_id in ordered_fragments:
        line_len = _d1_line_length(inp.shape, line_id)
        if line_len < length:
            return None
        seg_t = random.randint(0, line_len - length)
        _paint_d1(inp, line_id, seg_t, length, color)

    distractor_candidates = [line_id for line_id in candidates if line_id not in line_ids]
    distractor_lines = random.sample(
        distractor_candidates,
        k=min(len(distractor_candidates), random.randint(0, 3)),
    )
    grey_points: list[tuple[int, int]] = []
    for line_id in distractor_lines:
        line_len = _d1_line_length(inp.shape, line_id)
        t = random.randint(0, line_len - 1)
        r0, c0 = _d1_origin(inp.shape, line_id)
        rr, cc = r0 + t, c0 + t
        inp[rr, cc] = GREY
        grey_points.append((rr, cc))

    out = _reassembled_output(
        inp.shape,
        bg,
        grey_points,
        "D1",
        (start_r, start_c),
        [(length, color) for length, color, _line in ordered_fragments],
    )
    return _oracle_checked_input(inp, out)


def generate_input() -> np.ndarray:
    generators = {
        "H": _gen_horizontal,
        "V": _gen_vertical,
        "D1": _gen_d1,
    }
    for _ in range(MAX_GENERATION_ATTEMPTS):
        kind = random.choice(GEN_KINDS)
        grid = generators[kind]()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
