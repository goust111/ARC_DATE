from __future__ import annotations

import numpy as np


BG = 0
RED = 2
GREEN = 3

FRAME_COLORS = (1, 4, 5, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")

GRID_MIN = 16
GRID_MAX = 30
OBJECT_COUNT_MIN = 2
OBJECT_COUNT_MAX = 4
OUTER_MIN = 5
OUTER_MAX = 13
MARKER_MAX = 6
EDGE_MARGIN = 1
PLACEMENT_PAD = 1
PLACEMENT_TRIES = 80
GENERATION_TRIES = 400


def _cell(g: np.ndarray, r: int, c: int) -> int | None:
    h, w = g.shape
    if 0 <= r < h and 0 <= c < w:
        return int(g[r, c])
    return None


def _scan_run(g: np.ndarray, r: int, c: int, dr: int, dc: int, color: int) -> int:
    h, w = g.shape
    length = 0
    rr, cc = r, c
    while 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == color:
        length += 1
        rr += dr
        cc += dc
    return length


def _door_side_and_color(g: np.ndarray, r: int, c: int) -> tuple[str, int]:
    up = _cell(g, r - 1, c)
    down = _cell(g, r + 1, c)
    left = _cell(g, r, c - 1)
    right = _cell(g, r, c + 1)

    horizontal = left is not None and right is not None and left == right and left not in {BG, RED, GREEN}
    vertical = up is not None and down is not None and up == down and up not in {BG, RED, GREEN}
    assert horizontal != vertical, "door marker must belong to exactly one frame edge"

    if horizontal:
        frame_color = int(left)
        left_run = _scan_run(g, r, c - 1, 0, -1, frame_color)
        right_run = _scan_run(g, r, c + 1, 0, 1, frame_color)
        c0 = c - left_run
        c1 = c + right_run
        up_cont = _cell(g, r - 1, c0) == frame_color and _cell(g, r - 1, c1) == frame_color
        down_cont = _cell(g, r + 1, c0) == frame_color and _cell(g, r + 1, c1) == frame_color
        assert up_cont != down_cont, "horizontal door must be on the top or bottom edge"
        return ("top" if down_cont else "bottom"), frame_color

    frame_color = int(up)
    up_run = _scan_run(g, r - 1, c, -1, 0, frame_color)
    down_run = _scan_run(g, r + 1, c, 1, 0, frame_color)
    r0 = r - up_run
    r1 = r + down_run
    left_cont = _cell(g, r0, c - 1) == frame_color and _cell(g, r1, c - 1) == frame_color
    right_cont = _cell(g, r0, c + 1) == frame_color and _cell(g, r1, c + 1) == frame_color
    assert left_cont != right_cont, "vertical door must be on the left or right edge"
    return ("left" if right_cont else "right"), frame_color


def _frame_bbox(g: np.ndarray, r: int, c: int, side: str, frame_color: int) -> tuple[int, int, int, int]:
    h, w = g.shape

    if side in {"top", "bottom"}:
        left_run = _scan_run(g, r, c - 1, 0, -1, frame_color)
        right_run = _scan_run(g, r, c + 1, 0, 1, frame_color)
        c0 = c - left_run
        c1 = c + right_run
        assert 0 <= c0 < c < c1 < w
        if side == "top":
            r0 = r
            r1 = r
            while r1 + 1 < h and int(g[r1 + 1, c0]) == frame_color and int(g[r1 + 1, c1]) == frame_color:
                r1 += 1
        else:
            r1 = r
            r0 = r
            while r0 - 1 >= 0 and int(g[r0 - 1, c0]) == frame_color and int(g[r0 - 1, c1]) == frame_color:
                r0 -= 1
        return r0, c0, r1, c1

    up_run = _scan_run(g, r - 1, c, -1, 0, frame_color)
    down_run = _scan_run(g, r + 1, c, 1, 0, frame_color)
    r0 = r - up_run
    r1 = r + down_run
    assert 0 <= r0 < r < r1 < h
    if side == "left":
        c0 = c
        c1 = c
        while c1 + 1 < w and int(g[r0, c1 + 1]) == frame_color and int(g[r1, c1 + 1]) == frame_color:
            c1 += 1
    else:
        c1 = c
        c0 = c
        while c0 - 1 >= 0 and int(g[r0, c0 - 1]) == frame_color and int(g[r1, c0 - 1]) == frame_color:
            c0 -= 1
    return r0, c0, r1, c1


def _validate_frame(
    g: np.ndarray,
    bbox: tuple[int, int, int, int],
    frame_color: int,
    side: str,
    door: tuple[int, int],
) -> None:
    r0, c0, r1, c1 = bbox
    dr, dc = door
    assert r1 - r0 + 1 >= 5 and c1 - c0 + 1 >= 5, "frame must leave a nontrivial interior"

    if side == "top":
        assert dr == r0
        assert (c1 - c0 + 1) % 2 == 1 and dc == (c0 + c1) // 2
    elif side == "bottom":
        assert dr == r1
        assert (c1 - c0 + 1) % 2 == 1 and dc == (c0 + c1) // 2
    elif side == "left":
        assert dc == c0
        assert (r1 - r0 + 1) % 2 == 1 and dr == (r0 + r1) // 2
    else:
        assert dc == c1
        assert (r1 - r0 + 1) % 2 == 1 and dr == (r0 + r1) // 2

    for rr in range(r0, r1 + 1):
        for cc in range(c0, c1 + 1):
            on_border = rr in {r0, r1} or cc in {c0, c1}
            val = int(g[rr, cc])
            if on_border:
                if (rr, cc) == door:
                    assert val == GREEN, "door marker must be green"
                else:
                    assert val == frame_color, "border must be monochromatic except for the door marker"
            else:
                assert val in {BG, RED}, "frame interior may only contain background and red markers"


def _door_cells(door: tuple[int, int], side: str, count: int) -> list[tuple[int, int]]:
    before = (count - 1) // 2
    after = count // 2
    r, c = door
    if side in {"top", "bottom"}:
        return [(r, cc) for cc in range(c - before, c + after + 1)]
    return [(rr, c) for rr in range(r - before, r + after + 1)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove a door-sized gap from each framed rectangle based on the number of internal red markers.

    Input structure:
    - Black background.
    - Several one-pixel-thick colored rectangular frames.
    - Each frame has one green door marker on the midpoint of one edge.
    - Each frame interior contains red marker pixels on black cells.

    Rule:
    - Count red markers inside each frame.
    - Open a gap of that width on the edge containing the green marker.
    - For even widths, bias the gap one cell to the right or downward from the green cell.
    - Remove all red markers and the green marker itself.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    green_coords = [tuple(map(int, coord)) for coord in np.argwhere(g == GREEN)]
    assert green_coords, "expected at least one green door marker"

    for door in sorted(green_coords):
        r, c = door
        side, frame_color = _door_side_and_color(g, r, c)
        bbox = _frame_bbox(g, r, c, side, frame_color)
        _validate_frame(g, bbox, frame_color, side, door)

        r0, c0, r1, c1 = bbox
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        marker_coords = np.argwhere(interior == RED)
        count = int(len(marker_coords))
        assert count >= 1, "each frame must contain at least one red marker"
        door_len = (c1 - c0 + 1) if side in {"top", "bottom"} else (r1 - r0 + 1)
        assert count <= door_len - 2, "door opening must stay away from the frame corners"

        out_interior = out[r0 + 1 : r1, c0 + 1 : c1]
        out_interior[interior == RED] = BG
        for rr, cc in _door_cells(door, side, count):
            out[rr, cc] = BG

    return out


def _balanced_sides(rng: np.random.Generator, count: int) -> list[str]:
    order = list(rng.permutation(SIDES))
    return [str(order[i]) for i in range(count)]


def _balanced_parities(rng: np.random.Generator, count: int) -> list[int]:
    parities = [0, 1] * (count // 2)
    if count % 2 == 1:
        parities.append(int(rng.integers(0, 2)))
    rng.shuffle(parities)
    return [int(x) for x in parities]


def _size_options(grid_h: int, grid_w: int, side: str, parity: int) -> list[tuple[int, int]]:
    max_h = min(OUTER_MAX, grid_h - 2 * EDGE_MARGIN)
    max_w = min(OUTER_MAX, grid_w - 2 * EDGE_MARGIN)
    options: list[tuple[int, int]] = []

    for h in range(OUTER_MIN, max_h + 1):
        for w in range(OUTER_MIN, max_w + 1):
            if side in {"top", "bottom"} and w % 2 == 0:
                continue
            if side in {"left", "right"} and h % 2 == 0:
                continue
            door_len = w if side in {"top", "bottom"} else h
            interior_area = (h - 2) * (w - 2)
            max_count = min(MARKER_MAX, door_len - 2, interior_area)
            if max_count < 1:
                continue
            if not any(n % 2 == parity for n in range(1, max_count + 1)):
                continue
            options.append((h, w))
    return options


def _sample_marker_count(rng: np.random.Generator, h: int, w: int, side: str, parity: int) -> int | None:
    door_len = w if side in {"top", "bottom"} else h
    interior_area = (h - 2) * (w - 2)
    max_count = min(MARKER_MAX, door_len - 2, interior_area)
    counts = [n for n in range(1, max_count + 1) if n % 2 == parity]
    if not counts:
        return None
    return int(rng.choice(counts))


def _is_free(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    H, W = occupied.shape
    rr0 = max(0, r0 - PLACEMENT_PAD)
    rr1 = min(H, r0 + h + PLACEMENT_PAD)
    cc0 = max(0, c0 - PLACEMENT_PAD)
    cc1 = min(W, c0 + w + PLACEMENT_PAD)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _occupy(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    H, W = occupied.shape
    rr0 = max(0, r0 - PLACEMENT_PAD)
    rr1 = min(H, r0 + h + PLACEMENT_PAD)
    cc0 = max(0, c0 - PLACEMENT_PAD)
    cc1 = min(W, c0 + w + PLACEMENT_PAD)
    occupied[rr0:rr1, cc0:cc1] = True


def _door_position(spec: dict[str, object]) -> tuple[int, int]:
    r0 = int(spec["r0"])
    c0 = int(spec["c0"])
    h = int(spec["h"])
    w = int(spec["w"])
    side = str(spec["side"])

    if side == "top":
        return r0, c0 + w // 2
    if side == "bottom":
        return r0 + h - 1, c0 + w // 2
    if side == "left":
        return r0 + h // 2, c0
    return r0 + h // 2, c0 + w - 1


def _sample_red_positions(
    rng: np.random.Generator, r0: int, c0: int, h: int, w: int, count: int
) -> list[tuple[int, int]]:
    rows = np.arange(r0 + 1, r0 + h - 1)
    cols = np.arange(c0 + 1, c0 + w - 1)
    cells = [(int(rr), int(cc)) for rr in rows for cc in cols]
    picks = rng.choice(len(cells), size=count, replace=False)
    return [cells[int(idx)] for idx in picks]


def _render_input(grid_h: int, grid_w: int, specs: list[dict[str, object]]) -> np.ndarray:
    g = np.full((grid_h, grid_w), BG, dtype=int)
    for spec in specs:
        r0 = int(spec["r0"])
        c0 = int(spec["c0"])
        h = int(spec["h"])
        w = int(spec["w"])
        color = int(spec["color"])
        reds = list(spec["reds"])

        r1 = r0 + h - 1
        c1 = c0 + w - 1
        g[r0, c0 : c1 + 1] = color
        g[r1, c0 : c1 + 1] = color
        g[r0 : r1 + 1, c0] = color
        g[r0 : r1 + 1, c1] = color
        dr, dc = _door_position(spec)
        g[dr, dc] = GREEN
        for rr, cc in reds:
            g[int(rr), int(cc)] = RED
    return g


def _render_output(inp: np.ndarray, specs: list[dict[str, object]]) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        side = str(spec["side"])
        reds = list(spec["reds"])
        count = int(spec["count"])
        for rr, cc in reds:
            out[int(rr), int(cc)] = BG
        for rr, cc in _door_cells(_door_position(spec), side, count):
            out[int(rr), int(cc)] = BG
    return out


def _build_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    object_count = int(rng.integers(OBJECT_COUNT_MIN, OBJECT_COUNT_MAX + 1))
    min_side = max(GRID_MIN, 10 + 3 * object_count)
    grid_h = int(rng.integers(min_side, GRID_MAX + 1))
    grid_w = int(rng.integers(min_side, GRID_MAX + 1))

    side_choices = _balanced_sides(rng, object_count)
    parity_choices = _balanced_parities(rng, object_count)
    colors = [int(c) for c in rng.choice(np.array(FRAME_COLORS), size=object_count, replace=False)]

    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    specs: list[dict[str, object]] = []

    for side, parity, color in zip(side_choices, parity_choices, colors):
        options = _size_options(grid_h, grid_w, side, parity)
        if not options:
            return None

        placed = False
        for _ in range(PLACEMENT_TRIES):
            h, w = options[int(rng.integers(0, len(options)))]
            count = _sample_marker_count(rng, h, w, side, parity)
            if count is None:
                continue

            if grid_h - h <= EDGE_MARGIN or grid_w - w <= EDGE_MARGIN:
                continue
            r0 = int(rng.integers(EDGE_MARGIN, grid_h - h - EDGE_MARGIN + 1))
            c0 = int(rng.integers(EDGE_MARGIN, grid_w - w - EDGE_MARGIN + 1))
            if not _is_free(occupied, r0, c0, h, w):
                continue

            reds = _sample_red_positions(rng, r0, c0, h, w, count)
            spec = {
                "r0": r0,
                "c0": c0,
                "h": h,
                "w": w,
                "color": color,
                "side": side,
                "count": count,
                "reds": reds,
            }
            specs.append(spec)
            _occupy(occupied, r0, c0, h, w)
            placed = True
            break

        if not placed:
            return None

    inp = _render_input(grid_h, grid_w, specs)
    expected = _render_output(inp, specs)
    try:
        if np.array_equal(main(inp), expected):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_TRIES):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
