from __future__ import annotations

from collections import deque
import random

import numpy as np


GREY = 5
BG_OPTIONS = (0, 7)
DIRECTIONS = ("left", "right", "up", "down")
GEN_RETRIES = 400

INTERIOR_H_RANGE = (7, 16)
INTERIOR_W_RANGE = (7, 16)
MARGIN_RANGE = (3, 5)
OP_COUNT_RANGE = (3, 5)
SHIFT_RANGE = (3, 5)


def _most_common(values: np.ndarray) -> int:
    vals, counts = np.unique(values, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_grey_frame(g: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(g == GREY)
    assert pts.size, "expected a Grey rectangular border"
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "Grey frame must enclose an interior"

    border = np.zeros(g.shape, dtype=bool)
    border[r0, c0 : c1 + 1] = True
    border[r1, c0 : c1 + 1] = True
    border[r0 : r1 + 1, c0] = True
    border[r0 : r1 + 1, c1] = True
    assert bool(np.all(g[border] == GREY)), "Grey pixels must form a complete rectangle"
    assert not bool(np.any((g == GREY) & (~border))), "Grey is reserved for the border"
    return r0, c0, r1, c1


def _outside_frame_mask(shape: tuple[int, int], frame: tuple[int, int, int, int]) -> np.ndarray:
    h, w = shape
    r0, c0, r1, c1 = frame
    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    return (rows < r0) | (rows > r1) | (cols < c0) | (cols > c1)


def _components_for_color(g: np.ndarray, color: int, allowed: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = (g == int(color)) & allowed
    h, w = g.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            q: deque[tuple[int, int]] = deque([(int(r), int(c))])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _contiguous(values: list[int]) -> bool:
    vals = sorted(values)
    return vals == list(range(vals[0], vals[-1] + 1))


def _candidate_from_body(
    head: tuple[int, int], body: set[tuple[int, int]]
) -> list[dict[str, int | str]]:
    hr, hc = head
    amount = len(body)
    out: list[dict[str, int | str]] = []

    if amount == 1:
        (br, bc), = tuple(body)
        dr, dc = hr - br, hc - bc
        if abs(dr) + abs(dc) != 1:
            return out
        if dc != 0:
            out.append(
                {
                    "axis": "row",
                    "target": int(hr),
                    "direction": "right" if dc > 0 else "left",
                    "amount": int(amount),
                }
            )
        else:
            out.append(
                {
                    "axis": "col",
                    "target": int(hc),
                    "direction": "down" if dr > 0 else "up",
                    "amount": int(amount),
                }
            )
        return out

    rows = [r for r, _c in body]
    cols = [c for _r, c in body]
    if len(set(cols)) == 1 and _contiguous(rows):
        bc = cols[0]
        for br in (min(rows), max(rows)):
            if hr == br and abs(hc - bc) == 1:
                out.append(
                    {
                        "axis": "row",
                        "target": int(hr),
                        "direction": "right" if hc > bc else "left",
                        "amount": int(amount),
                    }
                )
    if len(set(rows)) == 1 and _contiguous(cols):
        br = rows[0]
        for bc in (min(cols), max(cols)):
            if hc == bc and abs(hr - br) == 1:
                out.append(
                    {
                        "axis": "col",
                        "target": int(hc),
                        "direction": "down" if hr > br else "up",
                        "amount": int(amount),
                    }
                )
    return out


def _parse_arrow(
    coords: list[tuple[int, int]], frame: tuple[int, int, int, int]
) -> dict[str, int | str]:
    r0, c0, r1, c1 = frame
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    if max(cols) < c0 or min(cols) > c1:
        expected_axis = "row"
    elif max(rows) < r0 or min(rows) > r1:
        expected_axis = "col"
    else:
        raise AssertionError("arrow must lie outside the Grey frame")

    cells = set(coords)
    candidates: list[dict[str, int | str]] = []
    for head in cells:
        body = set(cells)
        body.remove(head)
        candidates.extend(_candidate_from_body(head, body))

    valid: list[dict[str, int | str]] = []
    for cand in candidates:
        if cand["axis"] != expected_axis:
            continue
        target = int(cand["target"])
        if expected_axis == "row" and r0 < target < r1:
            valid.append(cand)
        if expected_axis == "col" and c0 < target < c1:
            valid.append(cand)

    unique: list[dict[str, int | str]] = []
    seen: set[tuple[str, int, str, int]] = set()
    for cand in valid:
        key = (
            str(cand["axis"]),
            int(cand["target"]),
            str(cand["direction"]),
            int(cand["amount"]),
        )
        if key not in seen:
            seen.add(key)
            unique.append(cand)

    assert len(unique) == 1, "each arrow must have one unambiguous head and body"
    return unique[0]


def _signed_shift(direction: str, amount: int) -> int:
    if direction in ("right", "down"):
        return int(amount)
    if direction in ("left", "up"):
        return -int(amount)
    raise AssertionError("unknown arrow direction")


def _apply_shifts(
    base: np.ndarray,
    frame: tuple[int, int, int, int],
    row_shifts: dict[int, int],
    col_shifts: dict[int, int],
) -> np.ndarray:
    r0, c0, r1, c1 = frame
    out = base.copy()
    width = int(c1 - c0 - 1)
    height = int(r1 - r0 - 1)

    for r in sorted(row_shifts):
        shift = int(row_shifts[r]) % width
        out[int(r), c0 + 1 : c1] = np.roll(out[int(r), c0 + 1 : c1], shift)

    for c in sorted(col_shifts):
        shift = int(col_shifts[c]) % height
        out[r0 + 1 : r1, int(c)] = np.roll(out[r0 + 1 : r1, int(c)], shift)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Apply arrow-coded cyclic row and column shifts inside a Grey frame.

    Input structure:
    - A complete one-pixel Grey rectangular border surrounds a multicolor interior pattern.
    - The area outside that Grey border is background except for same-color L-shaped arrows.
    - Arrows on the left or right side control one interior row and point left or right.
    - Arrows above or below the frame control one interior column and point up or down.

    Rule:
    - Read each arrow as a straight body plus one offset head pixel; the body length is the shift amount.
    - Shift the controlled interior row left or right by that amount with cyclic wraparound.
    - After all row shifts, shift the controlled interior columns up or down with cyclic wraparound.
    - Keep the Grey border fixed and erase the outside arrow marks back to the background color.

    Key insight:
    - The marker shape encodes both a direction and a numeric amount, and the row operations must be completed before the column operations.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frame = _find_grey_frame(g)
    outside = _outside_frame_mask(g.shape, frame)
    assert bool(outside.any()), "expected background outside the Grey frame"
    bg = _most_common(g[outside])

    arrow_mask = outside & (g != bg) & (g != GREY)
    arrow_colors = sorted(int(v) for v in np.unique(g[arrow_mask]))

    row_shifts: dict[int, int] = {}
    col_shifts: dict[int, int] = {}
    for color in arrow_colors:
        for comp in _components_for_color(g, color, arrow_mask):
            arrow = _parse_arrow(comp, frame)
            axis = str(arrow["axis"])
            target = int(arrow["target"])
            shift = _signed_shift(str(arrow["direction"]), int(arrow["amount"]))
            if axis == "row":
                row_shifts[target] = row_shifts.get(target, 0) + shift
            else:
                col_shifts[target] = col_shifts.get(target, 0) + shift

    out = g.copy()
    out[outside] = int(bg)
    return _apply_shifts(out, frame, row_shifts, col_shifts)


def _make_pattern(h: int, w: int, colors: list[int]) -> np.ndarray:
    tile_r = random.randint(1, 3)
    tile_c = random.randint(1, 3)
    pat = np.empty((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            pat[r, c] = colors[(r // tile_r + 2 * (c // tile_c) + r + c) % len(colors)]

    for _ in range(random.randint(4, 9)):
        rh = random.randint(1, min(4, h))
        rw = random.randint(1, min(4, w))
        r0 = random.randint(0, h - rh)
        c0 = random.randint(0, w - rw)
        pat[r0 : r0 + rh, c0 : c0 + rw] = random.choice(colors)

    for _ in range(random.randint(0, 8)):
        pat[random.randrange(h), random.randrange(w)] = random.choice(colors)
    return pat


def _arrow_padding(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    pad: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                pad.add((int(rr), int(cc)))
    return pad


def _row_arrow_cells(
    frame: tuple[int, int, int, int],
    side: str,
    direction: str,
    target: int,
    amount: int,
    body_step: int,
) -> set[tuple[int, int]]:
    _r0, c0, _r1, c1 = frame
    body_col = c0 - 2 if side == "left" else c1 + 2
    head_col = body_col + (1 if direction == "right" else -1)
    cells = {(int(target + body_step * i), int(body_col)) for i in range(amount)}
    cells.add((int(target), int(head_col)))
    return cells


def _col_arrow_cells(
    frame: tuple[int, int, int, int],
    side: str,
    direction: str,
    target: int,
    amount: int,
    body_step: int,
) -> set[tuple[int, int]]:
    r0, _c0, r1, _c1 = frame
    body_row = r0 - 2 if side == "top" else r1 + 2
    head_row = body_row + (1 if direction == "down" else -1)
    cells = {(int(body_row), int(target + body_step * i)) for i in range(amount)}
    cells.add((int(head_row), int(target)))
    return cells


def _cells_available(
    grid: np.ndarray, bg: int, cells: set[tuple[int, int]], reserved: set[tuple[int, int]]
) -> bool:
    h, w = grid.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) != int(bg):
            return False
    return not bool(_arrow_padding(cells, h, w) & reserved)


def _place_arrow(
    grid: np.ndarray,
    frame: tuple[int, int, int, int],
    bg: int,
    color: int,
    direction: str,
    used_rows: set[int],
    used_cols: set[int],
    reserved: set[tuple[int, int]],
) -> dict[str, int | str] | None:
    r0, c0, r1, c1 = frame
    inner_h = int(r1 - r0 - 1)
    inner_w = int(c1 - c0 - 1)

    for _ in range(100):
        if direction in ("left", "right"):
            max_amount = min(SHIFT_RANGE[1], inner_h, inner_w - 1)
            if max_amount < SHIFT_RANGE[0]:
                return None
            amount = random.randint(SHIFT_RANGE[0], max_amount)
            rows = []
            for target in range(r0 + 1, r1):
                if target in used_rows:
                    continue
                steps = []
                if target - (amount - 1) >= r0 + 1:
                    steps.append(-1)
                if target + (amount - 1) <= r1 - 1:
                    steps.append(1)
                if steps:
                    rows.append((target, steps))
            if not rows:
                continue
            target, steps = random.choice(rows)
            side = random.choice(("left", "right"))
            body_step = random.choice(steps)
            cells = _row_arrow_cells(frame, side, direction, target, amount, body_step)
            if not _cells_available(grid, bg, cells, reserved):
                continue
            for cell in cells:
                grid[cell] = int(color)
            reserved.update(_arrow_padding(cells, *grid.shape))
            used_rows.add(int(target))
            return {
                "axis": "row",
                "target": int(target),
                "direction": direction,
                "amount": int(amount),
            }

        max_amount = min(SHIFT_RANGE[1], inner_w, inner_h - 1)
        if max_amount < SHIFT_RANGE[0]:
            return None
        amount = random.randint(SHIFT_RANGE[0], max_amount)
        cols = []
        for target in range(c0 + 1, c1):
            if target in used_cols:
                continue
            steps = []
            if target - (amount - 1) >= c0 + 1:
                steps.append(-1)
            if target + (amount - 1) <= c1 - 1:
                steps.append(1)
            if steps:
                cols.append((target, steps))
        if not cols:
            continue
        target, steps = random.choice(cols)
        side = random.choice(("top", "bottom"))
        body_step = random.choice(steps)
        cells = _col_arrow_cells(frame, side, direction, target, amount, body_step)
        if not _cells_available(grid, bg, cells, reserved):
            continue
        for cell in cells:
            grid[cell] = int(color)
        reserved.update(_arrow_padding(cells, *grid.shape))
        used_cols.add(int(target))
        return {
            "axis": "col",
            "target": int(target),
            "direction": direction,
            "amount": int(amount),
        }
    return None


def _directions_for_count(count: int) -> list[str]:
    dirs = random.sample(list(DIRECTIONS), min(4, count))
    while len(dirs) < count:
        dirs.append(random.choice(list(DIRECTIONS)))
    random.shuffle(dirs)
    return dirs


def _expected_from_specs(
    grid: np.ndarray,
    frame: tuple[int, int, int, int],
    bg: int,
    specs: list[dict[str, int | str]],
) -> np.ndarray:
    outside = _outside_frame_mask(grid.shape, frame)
    out = grid.copy()
    out[outside] = int(bg)
    row_shifts: dict[int, int] = {}
    col_shifts: dict[int, int] = {}
    for spec in specs:
        target = int(spec["target"])
        shift = _signed_shift(str(spec["direction"]), int(spec["amount"]))
        if spec["axis"] == "row":
            row_shifts[target] = row_shifts.get(target, 0) + shift
        else:
            col_shifts[target] = col_shifts.get(target, 0) + shift
    return _apply_shifts(out, frame, row_shifts, col_shifts)


def _gen_attempt() -> np.ndarray | None:
    inner_h = random.randint(*INTERIOR_H_RANGE)
    inner_w = random.randint(*INTERIOR_W_RANGE)
    top = random.randint(*MARGIN_RANGE)
    bottom = random.randint(*MARGIN_RANGE)
    left = random.randint(*MARGIN_RANGE)
    right = random.randint(*MARGIN_RANGE)

    h = int(top + inner_h + 2 + bottom)
    w = int(left + inner_w + 2 + right)
    if h > 30 or w > 30:
        return None

    bg = random.choice(BG_OPTIONS)
    frame = (int(top), int(left), int(top + inner_h + 1), int(left + inner_w + 1))
    r0, c0, r1, c1 = frame

    palette = [c for c in range(10) if c not in {bg, GREY}]
    random.shuffle(palette)
    pattern_colors = palette[:3]
    op_count = random.randint(*OP_COUNT_RANGE)
    arrow_colors = palette[3 : 3 + op_count]
    if len(arrow_colors) < op_count:
        return None

    grid = np.full((h, w), int(bg), dtype=int)
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY
    grid[r0 + 1 : r1, c0 + 1 : c1] = _make_pattern(inner_h, inner_w, pattern_colors)

    used_rows: set[int] = set()
    used_cols: set[int] = set()
    reserved: set[tuple[int, int]] = set()
    specs: list[dict[str, int | str]] = []
    for direction, color in zip(_directions_for_count(op_count), arrow_colors):
        spec = _place_arrow(grid, frame, bg, color, direction, used_rows, used_cols, reserved)
        if spec is None:
            return None
        specs.append(spec)

    expected = _expected_from_specs(grid, frame, bg, specs)
    clean = grid.copy()
    clean[_outside_frame_mask(grid.shape, frame)] = int(bg)
    if np.array_equal(expected, clean):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
