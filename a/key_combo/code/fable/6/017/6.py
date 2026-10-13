from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_UTILS_SPEC = importlib.util.spec_from_file_location("_arc_utils_shared", _ARC_UTILS_PATH)
if _ARC_UTILS_SPEC is None or _ARC_UTILS_SPEC.loader is None:
    raise ImportError(f"could not load arc_utils from {_ARC_UTILS_PATH}")
_arc_utils = importlib.util.module_from_spec(_ARC_UTILS_SPEC)
_ARC_UTILS_SPEC.loader.exec_module(_arc_utils)

Color = _arc_utils.Color
find_components = _arc_utils.find_components
get_bg = _arc_utils.get_bg


GREY = Color.GREY
RED = Color.RED
MAX_GENERATION_ATTEMPTS = 800
MAX_GROUP_PLACEMENT_ATTEMPTS = 120
GRID_SIZE_BY_GROUPS = {
    1: (12, 18),
    2: (16, 24),
    3: (20, 28),
}


Coord = tuple[int, int]


def _rotate_rel(coords: set[Coord], steps: int) -> set[Coord]:
    out = {(int(r), int(c)) for r, c in coords}
    for _ in range(steps % 4):
        out = {(int(c), int(-r)) for r, c in out}
    return out


def _rotate_abs(coords: set[Coord], pivot: Coord, steps: int) -> set[Coord]:
    pr, pc = pivot
    rel = {(int(r - pr), int(c - pc)) for r, c in coords}
    return {(int(pr + dr), int(pc + dc)) for dr, dc in _rotate_rel(rel, steps)}


def _is_adjacent8(cell: Coord, sprite: set[Coord]) -> bool:
    r, c = cell
    return any(max(abs(r - sr), abs(c - sc)) == 1 for sr, sc in sprite)


def _color_components(grid: np.ndarray, color: int, bg: int) -> list[set[Coord]]:
    comps = find_components(grid, bg=bg, colors=[int(color)], connectivity=4, monochromatic=True)
    return [{(int(r), int(c)) for r, c in comp["coords"]} for comp in comps]


def _extract_groups(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    colors = sorted(set(map(int, g.flatten())) - {bg, int(GREY), int(RED)})
    grey_points = [tuple(map(int, rc)) for rc in np.argwhere(g == int(GREY))]
    red_points = {tuple(map(int, rc)) for rc in np.argwhere(g == int(RED))}
    groups: list[dict] = []

    for color in colors:
        sprites = _color_components(g, color, bg)
        assert len(sprites) == 2, f"expected exactly two {color}-colored sprites"

        start_hits: list[tuple[set[Coord], list[Coord], set[Coord]]] = []
        for i, sprite in enumerate(sprites):
            attached = [pt for pt in red_points if _is_adjacent8(pt, sprite)]
            if attached:
                start_hits.append((sprite, attached, sprites[1 - i]))
        assert len(start_hits) == 1, f"expected exactly one marked start sprite for color {color}"

        start, attached, second = start_hits[0]
        assert 1 <= len(attached) <= 3, "red markers must count out 1..3 steps"

        matches: list[tuple[Coord, int]] = []
        for pivot in grey_points:
            if _rotate_abs(start, pivot, 1) == second:
                matches.append((pivot, 1))
            if _rotate_abs(start, pivot, -1) == second:
                matches.append((pivot, -1))
        assert len(matches) == 1, f"expected a unique pivot+direction match for color {color}"

        pivot, direction = matches[0]
        final = set(second)
        for _ in range(len(attached)):
            final = _rotate_abs(final, pivot, direction)

        groups.append(
            {
                "color": int(color),
                "pivot": pivot,
                "direction": int(direction),
                "steps": int(len(attached)),
                "start": start,
                "second": second,
                "final": final,
                "markers": set(attached),
            }
        )

    return groups


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Keep the two observed frames, the grey pivot, and the red markers unchanged,
    then add the skipped final frame obtained by continuing the inferred 90-degree
    rotation from the second frame for k more steps, where k is the number of red
    markers attached to the start frame.
    """
    grid = np.asarray(input_grid, dtype=int)
    out = grid.copy()
    for group in _extract_groups(grid):
        for r, c in group["final"]:
            out[int(r), int(c)] = int(group["color"])
    return out


def _connected(mask: set[Coord]) -> bool:
    todo = [next(iter(mask))]
    seen = set()
    while todo:
        cell = todo.pop()
        if cell in seen:
            continue
        seen.add(cell)
        r, c = cell
        for nbr in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if nbr in mask and nbr not in seen:
                todo.append(nbr)
    return len(seen) == len(mask)


def _random_shape() -> set[Coord]:
    for _ in range(200):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        area = random.randint(3, min(6, h * w - 1))
        cells = {(random.randrange(h), random.randrange(w))}
        while len(cells) < area:
            frontier = [
                (r + dr, c + dc)
                for r, c in cells
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                if 0 <= r + dr < h and 0 <= c + dc < w and (r + dr, c + dc) not in cells
            ]
            if not frontier:
                break
            cells.add(random.choice(frontier))
        shifted = {(int(r + 1), int(c + 1)) for r, c in cells}
        if len(shifted) != area or not _connected(shifted):
            continue
        if any(_rotate_rel(shifted, turns) == shifted for turns in (1, 2, 3)):
            continue
        return shifted
    raise RuntimeError("failed to sample an asymmetric sprite")


def _sign(x: int) -> int:
    return 1 if x > 0 else -1


def _marker_candidates(start_rel: set[Coord], all_frames: set[Coord]) -> list[Coord]:
    candidates: set[Coord] = set()
    for r, c in start_rel:
        for rr, cc in ((r + _sign(r), c), (r, c + _sign(c))):
            if (rr, cc) != (0, 0) and (rr, cc) not in all_frames:
                candidates.add((int(rr), int(cc)))
    return sorted(candidates, key=lambda rc: (abs(rc[0]) + abs(rc[1]), rc[0], rc[1]))


def _build_group() -> dict | None:
    for _ in range(200):
        base = _random_shape()
        frames = [_rotate_rel(base, turns) for turns in range(4)]
        all_frames = set().union(*frames)
        direction = random.choice((-1, 1))
        start_turn = random.randrange(4)
        start_rel = frames[start_turn]
        second_rel = frames[(start_turn + direction) % 4]
        step_count = random.randint(1, 2)
        final_rel = frames[(start_turn + direction * (1 + step_count)) % 4]
        marker_candidates = _marker_candidates(start_rel, all_frames)
        if len(marker_candidates) < step_count:
            continue
        markers_rel = set(random.sample(marker_candidates, step_count))
        reserved_rel = set(all_frames) | markers_rel | {(0, 0)}
        if final_rel & markers_rel:
            continue
        return {
            "direction": int(direction),
            "steps": int(step_count),
            "start_rel": start_rel,
            "second_rel": second_rel,
            "final_rel": final_rel,
            "markers_rel": markers_rel,
            "reserved_rel": reserved_rel,
        }
    return None


def _fits_with_padding(cells: set[Coord], occupied: set[Coord], shape: tuple[int, int]) -> bool:
    h, w = shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if (rr, cc) in occupied:
                    return False
    return True


def _translate(cells: set[Coord], delta: Coord) -> set[Coord]:
    dr, dc = delta
    return {(int(r + dr), int(c + dc)) for r, c in cells}


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        n_groups = random.randint(1, 3)
        side_lo, side_hi = GRID_SIZE_BY_GROUPS[n_groups]
        height = random.randint(side_lo, side_hi)
        width = random.randint(side_lo, side_hi)
        bg = random.choice([c for c in Color.ALL_COLORS if c not in {int(GREY), int(RED)}])
        palette = [c for c in Color.ALL_COLORS if c not in {int(bg), int(GREY), int(RED)}]
        if len(palette) < n_groups:
            continue

        grid = np.full((height, width), int(bg), dtype=int)
        occupied: set[Coord] = set()
        success = True

        for color in random.sample(palette, n_groups):
            group = _build_group()
            if group is None:
                success = False
                break

            reserved_rel = group["reserved_rel"]
            rs = [r for r, _ in reserved_rel]
            cs = [c for _, c in reserved_rel]
            row_min = -min(rs)
            row_max = height - 1 - max(rs)
            col_min = -min(cs)
            col_max = width - 1 - max(cs)
            if row_min > row_max or col_min > col_max:
                success = False
                break

            placed = False
            for _ in range(MAX_GROUP_PLACEMENT_ATTEMPTS):
                pivot = (random.randint(row_min, row_max), random.randint(col_min, col_max))
                reserved_abs = _translate(reserved_rel, pivot)
                if not _fits_with_padding(reserved_abs, occupied, grid.shape):
                    continue

                for r, c in _translate(group["start_rel"], pivot):
                    grid[r, c] = int(color)
                for r, c in _translate(group["second_rel"], pivot):
                    grid[r, c] = int(color)
                for r, c in _translate(group["markers_rel"], pivot):
                    grid[r, c] = int(RED)
                grid[pivot] = int(GREY)
                occupied |= reserved_abs
                placed = True
                break

            if not placed:
                success = False
                break

        if not success:
            continue

        if int(np.sum(grid != int(bg))) == 0:
            continue
        if int(np.sum(grid == int(bg))) <= (height * width) // 2:
            continue

        try:
            groups = _extract_groups(grid)
            if len(groups) != n_groups:
                continue
            out = main(grid)
        except AssertionError:
            continue

        if not isinstance(out, np.ndarray) or out.ndim != 2 or out.shape != grid.shape:
            continue
        if out.shape[0] > 30 or out.shape[1] > 30:
            continue
        if not np.all((out == grid) | np.isin(out, [group["color"] for group in groups])):
            continue
        return grid

    raise RuntimeError("failed to generate a valid ARC instance")
