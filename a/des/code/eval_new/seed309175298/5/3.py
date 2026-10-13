from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
FRAME = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRECTIONS = ("up", "down", "left", "right")
GEN_RETRIES = 400
PLACEMENT_RETRIES = 120


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    mask = grid == color
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _frame_info(grid: np.ndarray, component: list[tuple[int, int]], bg: int) -> dict[str, object] | None:
    rows = [r for r, _ in component]
    cols = [c for _, c in component]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 < 3 or c1 - c0 < 3:
        return None

    frame_color = int(grid[component[0]])
    border = _perimeter_cells(r0, c0, r1, c1)
    comp_set = set(component)
    if not comp_set <= border:
        return None

    missing = list(border - comp_set)
    if len(missing) != 1:
        return None

    gap = missing[0]
    if gap in {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}:
        return None

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    if interior.size == 0 or np.any(interior == frame_color):
        return None

    seed_colors = {int(v) for v in np.unique(interior) if int(v) not in (bg, frame_color)}
    if len(seed_colors) != 1:
        return None

    seed_color = next(iter(seed_colors))
    if np.any((interior != bg) & (interior != seed_color)):
        return None

    seed_count = int(np.count_nonzero(interior == seed_color))
    if seed_count == 0:
        return None

    if gap[0] == r0:
        direction = (-1, 0)
    elif gap[0] == r1:
        direction = (1, 0)
    elif gap[1] == c0:
        direction = (0, -1)
    elif gap[1] == c1:
        direction = (0, 1)
    else:
        return None

    return {
        "gap": gap,
        "direction": direction,
        "seed_color": seed_color,
        "seed_count": seed_count,
    }


def _sprout_cells(gap: tuple[int, int], direction: tuple[int, int], length: int) -> list[tuple[int, int]]:
    gr, gc = gap
    dr, dc = direction
    return [(gr + dr * step, gc + dc * step) for step in range(1, length + 1)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a same-colored sprout out of each single-gap rectangular frame, with length equal to the number of seeds inside.

    Input structure:
    - The background is black.
    - One or more grey 1-pixel-thick rectangular frames appear on the grid; each frame has exactly one missing border cell on one side.
    - Inside each frame are one or more seed pixels of a single non-grey, non-black color placed on the black interior.

    Rule:
    - Detect each grey frame and locate its unique border gap.
    - The side containing the gap determines the growth direction, and the number of interior seed pixels determines the growth length.
    - Keep the original frame and seed pixels unchanged, and paint a straight line of the seed color outside the gap, starting just beyond the frame and extending outward by that many cells.

    Key insight:
    The puzzle combines two independent codes: gap position gives direction, while seed count gives distance.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    h, w = g.shape
    bg = _most_common_color(g)

    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        for component in _components_of_color(g, color):
            info = _frame_info(g, component, bg)
            if info is None:
                continue
            gap = info["gap"]
            direction = info["direction"]
            seed_color = int(info["seed_color"])
            seed_count = int(info["seed_count"])
            for r, c in _sprout_cells(gap, direction, seed_count):
                if 0 <= r < h and 0 <= c < w:
                    out[r, c] = seed_color
    return out


def _gap_data(
    top: int,
    left: int,
    frame_h: int,
    frame_w: int,
    side: str,
    offset: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    bottom = top + frame_h - 1
    right = left + frame_w - 1
    if side == "up":
        return (top, left + offset), (-1, 0)
    if side == "down":
        return (bottom, left + offset), (1, 0)
    if side == "left":
        return (top + offset, left), (0, -1)
    return (top + offset, right), (0, 1)


def _frame_cells(
    top: int,
    left: int,
    frame_h: int,
    frame_w: int,
    gap: tuple[int, int],
) -> list[tuple[int, int]]:
    bottom = top + frame_h - 1
    right = left + frame_w - 1
    cells = _perimeter_cells(top, left, bottom, right)
    cells.remove(gap)
    return sorted(cells)


def _reserve_box(
    grid_h: int,
    grid_w: int,
    top: int,
    left: int,
    frame_h: int,
    frame_w: int,
    side: str,
    sprout_len: int,
) -> tuple[int, int, int, int]:
    bottom = top + frame_h - 1
    right = left + frame_w - 1
    if side == "up":
        raw_r0, raw_r1, raw_c0, raw_c1 = top - sprout_len, bottom, left, right
    elif side == "down":
        raw_r0, raw_r1, raw_c0, raw_c1 = top, bottom + sprout_len, left, right
    elif side == "left":
        raw_r0, raw_r1, raw_c0, raw_c1 = top, bottom, left - sprout_len, right
    else:
        raw_r0, raw_r1, raw_c0, raw_c1 = top, bottom, left, right + sprout_len
    return (
        max(0, raw_r0 - 1),
        min(grid_h - 1, raw_r1 + 1),
        max(0, raw_c0 - 1),
        min(grid_w - 1, raw_c1 + 1),
    )


def _candidate_positions(
    blocked: np.ndarray,
    frame_h: int,
    frame_w: int,
    side: str,
    sprout_len: int,
) -> list[tuple[int, int]]:
    grid_h, grid_w = blocked.shape
    positions: list[tuple[int, int]] = []
    for top in range(grid_h - frame_h + 1):
        if side == "up" and top - sprout_len < 0:
            continue
        if side == "down" and top + frame_h - 1 + sprout_len >= grid_h:
            continue
        for left in range(grid_w - frame_w + 1):
            if side == "left" and left - sprout_len < 0:
                continue
            if side == "right" and left + frame_w - 1 + sprout_len >= grid_w:
                continue
            r0, r1, c0, c1 = _reserve_box(grid_h, grid_w, top, left, frame_h, frame_w, side, sprout_len)
            if not blocked[r0 : r1 + 1, c0 : c1 + 1].any():
                positions.append((top, left))
    return positions


def _oracle_output(grid: np.ndarray, objects: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    for obj in objects:
        for r, c in _sprout_cells(obj["gap"], obj["direction"], int(obj["seed_count"])):
            assert out[r, c] == BG
            out[r, c] = int(obj["seed_color"])
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    num_objects = rng.randint(1, 4)
    min_dim = min(30, 12 + 4 * (num_objects - 1))
    h = rng.randint(min_dim, 30)
    w = rng.randint(min_dim, 30)

    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    objects: list[dict[str, object]] = []
    directions = rng.sample(DIRECTIONS, k=num_objects)

    for side in directions:
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            frame_h = rng.randint(4, 8)
            frame_w = rng.randint(4, 8)
            max_seed_count = min((frame_h - 2) * (frame_w - 2), 5)
            if max_seed_count < 1:
                continue
            seed_count = rng.randint(1, max_seed_count)

            positions = _candidate_positions(blocked, frame_h, frame_w, side, seed_count)
            if not positions:
                continue

            top, left = rng.choice(positions)
            if side in ("up", "down"):
                offset = rng.randint(1, frame_w - 2)
            else:
                offset = rng.randint(1, frame_h - 2)

            gap, direction = _gap_data(top, left, frame_h, frame_w, side, offset)
            frame_cells = _frame_cells(top, left, frame_h, frame_w, gap)
            interior_cells = [
                (r, c)
                for r in range(top + 1, top + frame_h - 1)
                for c in range(left + 1, left + frame_w - 1)
            ]
            seed_cells = rng.sample(interior_cells, k=seed_count)
            seed_color = rng.choice(SEED_COLORS)

            for r, c in frame_cells:
                grid[r, c] = FRAME
            for r, c in seed_cells:
                grid[r, c] = seed_color

            r0, r1, c0, c1 = _reserve_box(h, w, top, left, frame_h, frame_w, side, seed_count)
            blocked[r0 : r1 + 1, c0 : c1 + 1] = True

            objects.append(
                {
                    "gap": gap,
                    "direction": direction,
                    "seed_color": seed_color,
                    "seed_count": seed_count,
                }
            )
            placed = True
            break
        if not placed:
            return None

    expected = _oracle_output(grid, objects)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
