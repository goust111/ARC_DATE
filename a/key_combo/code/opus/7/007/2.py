from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5
MAGENTA = 6
ORANGE = 7

LEFT_WIDTH = 6
SEP_WIDTH = 1
ROOM_LAYOUTS = {
    2: [(2, 1), (1, 2)],
    3: [(3, 1)],
    4: [(4, 1), (2, 2)],
    5: [(5, 1)],
}
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _count_enclosed_regions(walls: np.ndarray) -> int:
    h, w = walls.shape
    free = ~walls
    seen = np.zeros_like(free, dtype=bool)
    enclosed = 0

    for r in range(h):
        for c in range(w):
            if seen[r, c] or not free[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            touches_edge = r in (0, h - 1) or c in (0, w - 1)

            while queue:
                rr, cc = queue.popleft()
                for nr, nc in _neighbors4(rr, cc):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or not free[nr, nc]:
                        continue
                    seen[nr, nc] = True
                    if nr in (0, h - 1) or nc in (0, w - 1):
                        touches_edge = True
                    queue.append((nr, nc))

            if not touches_edge:
                enclosed += 1

    return enclosed


def _find_separator(grid: np.ndarray) -> int:
    g = np.asarray(grid, dtype=int)
    cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == GREY)]
    assert len(cols) == 1, "expected one full-height grey separator column"
    sep = cols[0]
    assert sep > 0 and sep < g.shape[1] - 1
    return sep


def _reconstruct_left_walls(left: np.ndarray) -> np.ndarray:
    orange = np.asarray(left, dtype=int) == ORANGE
    h, w = orange.shape
    row_major = orange.sum(axis=1) > (w / 2.0)
    col_major = orange.sum(axis=0) > (h / 2.0)

    walls = np.zeros_like(orange, dtype=bool)
    walls[row_major, :] = True
    walls[:, col_major] = True
    return walls


def main(grid: np.ndarray) -> np.ndarray:
    """Count enclosed regions in the damaged left frame, then use that count as a downward cyclic shift on the right pattern.

    Input structure:
    - A black background grid is split by one full-height grey column into a left frame zone and a right pattern zone.
    - The left zone contains orange rectilinear wall lines, possibly interrupted or covered by magenta noise pixels.
    - The right zone is a full-height blue/red pattern occupying every row.

    Rule:
    - Ignore magenta pixels and inspect only orange evidence in the left zone.
    - If a left-zone row has orange in more than half of its columns, reconstruct that entire row as wall; do the analogous fill for any column with orange in more than half of its rows.
    - Count the 4-connected non-wall regions fully enclosed by those reconstructed walls; call the count N.
    - Remove the left zone and grey separator, keep only the right zone, and cyclically shift it downward by N rows.

    Key insight:
    - The left drawing is a noisy room-count encoder: reconstructing its majority-supported rows and columns is necessary before the enclosed-region count can drive the right-side roll amount.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep = _find_separator(g)
    left = g[:, :sep]
    right = g[:, sep + 1 :]
    assert left.shape[1] >= 1 and right.shape[1] >= 1

    walls = _reconstruct_left_walls(left)
    shift = _count_enclosed_regions(walls)
    return np.roll(right, shift=shift, axis=0)


def _positive_partition(total: int, parts: int) -> list[int]:
    assert total >= parts >= 1
    cuts = sorted(random.sample(range(1, total), parts - 1))
    values = []
    prev = 0
    for cut in cuts + [total]:
        values.append(cut - prev)
        prev = cut
    return values


def _wall_positions(total_size: int, cell_count: int) -> list[int]:
    free_cells = total_size - (cell_count + 1)
    assert free_cells >= cell_count
    spans = _positive_partition(free_cells, cell_count)
    walls = [0]
    cur = 0
    for span in spans:
        cur += span + 1
        walls.append(cur)
    assert walls[-1] == total_size - 1
    return walls


def _make_left_frame(height: int, row_cells: int, col_cells: int) -> tuple[np.ndarray, list[int], list[int]]:
    rows = _wall_positions(height, row_cells)
    cols = _wall_positions(LEFT_WIDTH, col_cells)

    left = np.zeros((height, LEFT_WIDTH), dtype=int)
    left[rows, :] = ORANGE
    left[:, cols] = ORANGE
    return left, rows, cols


def _pick_critical_gap(height: int, rows: list[int], cols: list[int]) -> tuple[int, int]:
    inner_rows = [r for r in rows[1:-1]]
    inner_cols = [c for c in cols[1:-1]]

    if inner_rows and (not inner_cols or random.random() < 0.65):
        r = random.choice(inner_rows)
        c_choices = [c for c in range(LEFT_WIDTH) if c not in cols]
        return r, random.choice(c_choices)

    assert inner_cols
    c = random.choice(inner_cols)
    r_choices = [r for r in range(height) if r not in rows]
    return random.choice(r_choices), c


def _can_damage_wall(pos: tuple[int, int], row_damage: dict[int, int], col_damage: dict[int, int], height: int) -> bool:
    r, c = pos
    if row_damage.get(r, 0) >= 2:
        return False
    if col_damage.get(c, 0) >= max(2, height // 4):
        return False
    return True


def _add_noise(left: np.ndarray, rows: list[int], cols: list[int]) -> np.ndarray:
    noisy = left.copy()
    h, _w = noisy.shape
    row_damage: dict[int, int] = {}
    col_damage: dict[int, int] = {}

    def apply_wall_damage(pos: tuple[int, int]) -> None:
        r, c = pos
        noisy[r, c] = MAGENTA
        row_damage[r] = row_damage.get(r, 0) + 1
        col_damage[c] = col_damage.get(c, 0) + 1

    apply_wall_damage(_pick_critical_gap(h, rows, cols))

    wall_non_intersections = [
        (r, c)
        for r in range(h)
        for c in range(LEFT_WIDTH)
        if left[r, c] == ORANGE and not (r in rows and c in cols)
    ]
    random.shuffle(wall_non_intersections)
    extra_wall_damages = random.randint(0, 2)
    for pos in wall_non_intersections:
        if extra_wall_damages == 0:
            break
        if noisy[pos] != ORANGE or not _can_damage_wall(pos, row_damage, col_damage, h):
            continue
        apply_wall_damage(pos)
        extra_wall_damages -= 1

    distractor_cells = [
        (r, c)
        for r in range(h)
        for c in range(LEFT_WIDTH)
        if noisy[r, c] == BLACK and random.random() < 0.45
    ]
    random.shuffle(distractor_cells)
    for pos in distractor_cells[: random.randint(1, 4)]:
        noisy[pos] = MAGENTA

    return noisy


def _build_right_pattern(height: int, width: int) -> np.ndarray:
    seen: set[int] = set()
    rows: list[int] = []
    while len(rows) < height:
        bits = random.getrandbits(width)
        pop = bits.bit_count()
        if bits in seen or pop < 2 or pop > width - 2:
            continue
        seen.add(bits)
        rows.append(bits)

    random.shuffle(rows)
    out = np.full((height, width), BLUE, dtype=int)
    for r, bits in enumerate(rows):
        for c in range(width):
            if (bits >> c) & 1:
                out[r, c] = RED
    return out


def _semantic_checks(inp: np.ndarray, expected_shift: int, right: np.ndarray) -> bool:
    try:
        sep = _find_separator(inp)
        left = inp[:, :sep]
        recon = _reconstruct_left_walls(left)
        visible = left == ORANGE
        out = main(inp)
    except AssertionError:
        return False

    if _count_enclosed_regions(recon) != expected_shift:
        return False
    if _count_enclosed_regions(visible) == expected_shift:
        return False
    if not np.array_equal(out, np.roll(right, shift=expected_shift, axis=0)):
        return False

    matches = [k for k in range(inp.shape[0]) if np.array_equal(np.roll(right, shift=k, axis=0), out)]
    if matches != [expected_shift]:
        return False

    if inp.shape[1] != LEFT_WIDTH + SEP_WIDTH + right.shape[1]:
        return False
    if not np.all(inp[:, LEFT_WIDTH] == GREY):
        return False
    return True


def _generate_candidate() -> np.ndarray | None:
    room_count = random.randint(2, 5)
    row_cells, col_cells = random.choice(ROOM_LAYOUTS[room_count])

    min_height = max(12, 2 * (row_cells + 1), 2 * row_cells + 1)
    height = random.randint(min_height, 16)
    right_width = random.randint(7, 11)

    left, rows, cols = _make_left_frame(height, row_cells, col_cells)
    left = _add_noise(left, rows, cols)
    right = _build_right_pattern(height, right_width)

    grid = np.zeros((height, LEFT_WIDTH + SEP_WIDTH + right_width), dtype=int)
    grid[:, :LEFT_WIDTH] = left
    grid[:, LEFT_WIDTH] = GREY
    grid[:, LEFT_WIDTH + 1 :] = right

    if not _semantic_checks(grid, room_count, right):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for ARC puzzle 2")
