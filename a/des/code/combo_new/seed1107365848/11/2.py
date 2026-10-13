from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
COLOR_POOL = [1, 3, 4, 5, 6, 7, 8, 9]

GRID_SIDE_MIN = 10
GRID_SIDE_MAX = 24
ARROW_COUNT_RANGE = (1, 5)
MOVE_PIXEL_COUNT_RANGE = (1, 5)
NOISE_PIXEL_COUNT_RANGE = (2, 12)
GEN_RETRIES = 400

Coord = tuple[int, int]
ArrowInfo = tuple[str, int, int, Coord]


def _normalize(coords: set[Coord]) -> set[Coord]:
    r0 = min(r for r, _c in coords)
    c0 = min(c for _r, c in coords)
    return {(r - r0, c - c0) for r, c in coords}


_PATTERNS: dict[frozenset[Coord], tuple[str, int, int]] = {}
for extra_dr in (-1, 1):
    right = {(0, 0), (0, 1), (extra_dr, 0)}
    left = {(0, 0), (0, 1), (extra_dr, 1)}
    down = {(0, 0), (1, 0), (0, extra_dr)}
    up = {(0, 0), (1, 0), (1, extra_dr)}
    _PATTERNS[frozenset(_normalize(right))] = ("right", 0, 1)
    _PATTERNS[frozenset(_normalize(left))] = ("left", 0, 0)
    _PATTERNS[frozenset(_normalize(down))] = ("down", 1, 0)
    _PATTERNS[frozenset(_normalize(up))] = ("up", 0, 0)


def _neighbors8(r: int, c: int) -> tuple[Coord, ...]:
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((r + dr, c + dc))
    return tuple(out)


def _red_components(grid: np.ndarray) -> list[list[Coord]]:
    h, w = map(int, grid.shape)
    seen = np.zeros_like(grid, dtype=bool)
    comps: list[list[Coord]] = []

    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != RED or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors8(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == RED:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _extract_arrows(grid: np.ndarray) -> list[ArrowInfo]:
    arrows: list[ArrowInfo] = []
    seen_rows: set[int] = set()
    seen_cols: set[int] = set()

    for coords_list in _red_components(grid):
        coords = set(coords_list)
        assert len(coords) == 3, "each red marker must contain exactly three pixels"
        key = frozenset(_normalize(coords))
        assert key in _PATTERNS, f"unrecognized arrow shape: {sorted(coords)}"
        direction, tip_rel_r, tip_rel_c = _PATTERNS[key]

        r0 = min(r for r, _c in coords)
        c0 = min(c for _r, c in coords)
        tip = (r0 + tip_rel_r, c0 + tip_rel_c)

        if direction in ("left", "right"):
            line_index = tip[0]
            assert line_index not in seen_rows, "expected at most one horizontal arrow per row"
            seen_rows.add(line_index)
        else:
            line_index = tip[1]
            assert line_index not in seen_cols, "expected at most one vertical arrow per column"
            seen_cols.add(line_index)

        arrows.append((direction, line_index, 0 if direction in ("left", "right") else 1, tip))

    assert arrows, "expected at least one red arrow"
    arrows.sort(key=lambda item: (item[2], item[1], item[0]))
    return arrows


def _packed_values(line: np.ndarray) -> list[int]:
    return [int(v) for v in line if int(v) not in (BLACK, RED)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide the colored pixels on each arrow-marked row or column toward the arrow tip side, then erase the arrows.

    Input structure:
    - The background is black.
    - Non-black colored pixels are scattered across the grid.
    - Some red 3-pixel markers form small right-triangle arrows; each arrow belongs to
      one row or one column and points left, right, up, or down.

    Rule:
    - Detect every red arrow from its 3-pixel geometry and infer both its direction and
      the row or column it controls.
    - On each controlled row or column, read all non-black, non-red pixels in their
      natural line order and remove them from their original cells.
    - Repack that ordered list tightly against the grid edge on the arrow-tip side:
      left/up pack to index 0, right/down pack to the last cells.
    - Erase all red arrow pixels to black; pixels on untouched rows and columns stay put.

    Key insight:
    - The red marker is not output content; its asymmetric 3-pixel shape encodes both the
      motion direction and which entire row or column should be stably compacted.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)

    arrows = _extract_arrows(g)
    active_rows = {line for direction, line, axis, _tip in arrows if axis == 0}
    active_cols = {line for direction, line, axis, _tip in arrows if axis == 1}

    out = g.copy()

    for coords in _red_components(g):
        for r, c in coords:
            out[r, c] = BLACK

    for r in active_rows:
        for c in range(w):
            if int(out[r, c]) != RED:
                out[r, c] = BLACK

    for c in active_cols:
        for r in range(h):
            if int(out[r, c]) != RED:
                out[r, c] = BLACK

    for direction, line, axis, _tip in arrows:
        if axis == 0:
            values = _packed_values(g[line, :])
            if direction == "left":
                out[line, : len(values)] = values
            else:
                out[line, w - len(values) :] = values
        else:
            values = _packed_values(g[:, line])
            if direction == "up":
                out[: len(values), line] = values
            else:
                out[h - len(values) :, line] = values

    return out


def _arrow_shape(direction: str, tip: Coord, extra_sign: int) -> set[Coord]:
    r, c = tip
    if direction == "right":
        return {(r, c), (r, c - 1), (r + extra_sign, c - 1)}
    if direction == "left":
        return {(r, c), (r, c + 1), (r + extra_sign, c + 1)}
    if direction == "down":
        return {(r, c), (r - 1, c), (r - 1, c + extra_sign)}
    if direction == "up":
        return {(r, c), (r + 1, c), (r + 1, c + extra_sign)}
    raise AssertionError(direction)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _line_cells(shape: tuple[int, int], direction: str, line_index: int) -> list[Coord]:
    h, w = shape
    if direction in ("left", "right"):
        return [(line_index, c) for c in range(w)]
    return [(r, line_index) for r in range(h)]


def _choose_tip(shape: tuple[int, int], direction: str) -> Coord:
    h, w = shape
    if direction == "right":
        return random.randint(1, h - 2), random.randint(1, w - 2)
    if direction == "left":
        return random.randint(1, h - 2), random.randint(0, w - 2)
    if direction == "down":
        return random.randint(1, h - 2), random.randint(0, w - 2)
    if direction == "up":
        return random.randint(0, h - 2), random.randint(1, w - 2)
    raise AssertionError(direction)


def _touches_reserved(cells: set[Coord], reserved: set[Coord]) -> bool:
    for r, c in cells:
        if (r, c) in reserved:
            return True
        for nr, nc in _neighbors8(r, c):
            if (nr, nc) in reserved:
                return True
    return False


def _place_arrows(grid: np.ndarray) -> list[tuple[str, int]]:
    h, w = map(int, grid.shape)
    arrow_count = random.randint(*ARROW_COUNT_RANGE)
    direction_pool = ["left", "right", "up", "down"]
    directions = [random.choice(direction_pool) for _ in range(arrow_count)]
    if arrow_count >= 4 and random.random() < 0.5:
        random.shuffle(direction_pool)
        directions[:4] = direction_pool
        random.shuffle(directions)

    reserved: set[Coord] = set()
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    line_specs: list[tuple[str, int]] = []

    for direction in directions:
        placed = False
        for _ in range(100):
            tip = _choose_tip((h, w), direction)
            shape = _arrow_shape(direction, tip, random.choice((-1, 1)))
            if not all(_in_bounds((h, w), r, c) for r, c in shape):
                continue
            if _touches_reserved(shape, reserved):
                continue

            line_index = tip[0] if direction in ("left", "right") else tip[1]
            if direction in ("left", "right"):
                if line_index in used_rows:
                    continue
            else:
                if line_index in used_cols:
                    continue

            for r, c in shape:
                grid[r, c] = RED
            reserved |= shape
            if direction in ("left", "right"):
                used_rows.add(line_index)
            else:
                used_cols.add(line_index)
            line_specs.append((direction, line_index))
            placed = True
            break
        if not placed:
            raise RuntimeError("failed to place a non-overlapping arrow")

    return line_specs


def _sample_line_positions(length: int, count: int) -> list[int]:
    if count <= 0:
        return []
    positions = random.sample(range(length), count)
    positions.sort()
    return positions


def _eligible_source_cells(grid: np.ndarray, line_specs: list[tuple[str, int]]) -> dict[tuple[str, int], list[Coord]]:
    h, w = map(int, grid.shape)
    active_rows = {line for direction, line in line_specs if direction in ("left", "right")}
    active_cols = {line for direction, line in line_specs if direction in ("up", "down")}
    out: dict[tuple[str, int], list[Coord]] = {}

    for direction, line in line_specs:
        cells = _line_cells((h, w), direction, line)
        arrow_cells = {(r, c) for r, c in cells if int(grid[r, c]) == RED}
        free_cells = [cell for cell in cells if cell not in arrow_cells]
        out[(direction, line)] = [
            cell
            for cell in free_cells
            if not (
                direction in ("left", "right")
                and cell[1] in active_cols
                or direction in ("up", "down")
                and cell[0] in active_rows
            )
        ]
    return out


def _occupied_indices(direction: str, line_length: int, count: int) -> set[int]:
    if direction in ("left", "up"):
        return set(range(count))
    return set(range(line_length - count, line_length))


def _sample_line_counts(
    shape: tuple[int, int],
    line_specs: list[tuple[str, int]],
    eligible: dict[tuple[str, int], list[Coord]],
) -> dict[tuple[str, int], int] | None:
    h, w = shape
    rows = [(direction, line) for direction, line in line_specs if direction in ("left", "right")]
    cols = [(direction, line) for direction, line in line_specs if direction in ("up", "down")]

    ranges: dict[tuple[str, int], list[int]] = {}
    for key, cells in eligible.items():
        max_count = min(MOVE_PIXEL_COUNT_RANGE[1], len(cells))
        if max_count < MOVE_PIXEL_COUNT_RANGE[0]:
            return None
        ranges[key] = list(range(MOVE_PIXEL_COUNT_RANGE[0], max_count + 1))

    for _ in range(200):
        counts = {key: random.choice(ranges[key]) for key in line_specs}
        row_spans = {
            key: _occupied_indices(key[0], w, counts[key])
            for key in rows
        }
        col_spans = {
            key: _occupied_indices(key[0], h, counts[key])
            for key in cols
        }
        ok = True
        for row_key in rows:
            row_dir, row_idx = row_key
            _ = row_dir
            for col_key in cols:
                col_dir, col_idx = col_key
                _ = col_dir
                if col_idx in row_spans[row_key] and row_idx in col_spans[col_key]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return counts
    return None


def _fill_active_lines(
    grid: np.ndarray,
    line_specs: list[tuple[str, int]],
    counts: dict[tuple[str, int], int],
    eligible: dict[tuple[str, int], list[Coord]],
) -> None:
    for direction, line in line_specs:
        cells = eligible[(direction, line)]
        count = counts[(direction, line)]
        for idx in _sample_line_positions(len(cells), count):
            r, c = cells[idx]
            grid[r, c] = random.choice(COLOR_POOL)


def _place_noise(grid: np.ndarray, line_specs: list[tuple[str, int]]) -> None:
    h, w = map(int, grid.shape)
    active_rows = {line for direction, line in line_specs if direction in ("left", "right")}
    active_cols = {line for direction, line in line_specs if direction in ("up", "down")}
    candidates = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if int(grid[r, c]) == BLACK and r not in active_rows and c not in active_cols
    ]
    if not candidates:
        return
    noise_count = min(random.randint(*NOISE_PIXEL_COUNT_RANGE), len(candidates))
    for r, c in random.sample(candidates, noise_count):
        grid[r, c] = random.choice(COLOR_POOL)


def _valid_generator_case(grid: np.ndarray, line_specs: list[tuple[str, int]]) -> bool:
    arrows = _extract_arrows(grid)
    found = sorted((direction, line) for direction, line, _axis, _tip in arrows)
    expected = sorted(line_specs)
    if found != expected:
        return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    if not np.all(out[np.asarray(grid) == RED] == BLACK):
        return False
    return True


def _gen_attempt() -> np.ndarray:
    h = random.randint(GRID_SIDE_MIN, GRID_SIDE_MAX)
    w = random.randint(GRID_SIDE_MIN, GRID_SIDE_MAX)
    grid = np.full((h, w), BLACK, dtype=int)
    line_specs = _place_arrows(grid)
    eligible = _eligible_source_cells(grid, line_specs)
    counts = _sample_line_counts((h, w), line_specs, eligible)
    if counts is None:
        raise RuntimeError("failed to choose non-overlapping packed counts")
    _fill_active_lines(grid, line_specs, counts, eligible)
    _place_noise(grid, line_specs)
    if not _valid_generator_case(grid, line_specs):
        raise RuntimeError("generator produced an invalid case")
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            return _gen_attempt()
        except Exception:
            continue
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
