from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path
import random
import sys
from typing import Iterable

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if ARC_UTILS_DIR.exists():
    sys.path.insert(0, str(ARC_UTILS_DIR))

try:
    from arc_utils import Color, in_bounds
except Exception:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        YELLOW = 4
        GREY = 5
        PINK = 6
        ORANGE = 7
        TEAL = 8
        MAROON = 9

    def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
        h, w = map(int, shape)
        return 0 <= int(r) < h and 0 <= int(c) < w


BG = int(Color.BLACK)
PIVOT = int(Color.GREY)
PALETTE = [
    int(Color.BLUE),
    int(Color.RED),
    int(Color.GREEN),
    int(Color.YELLOW),
    int(Color.PINK),
    int(Color.ORANGE),
    int(Color.TEAL),
    int(Color.MAROON),
]

GEN_SIZE_RANGE = (11, 15)
GEN_VISIBLE_PATTERNS = [(1, 1), (1, 2), (2, 1), (2, 2)]
GEN_MAX_ATTEMPTS = 600


def _quarter_turn(dr: int, dc: int) -> tuple[int, int]:
    return -dc, dr


def _orbit_offsets(dr: int, dc: int) -> list[tuple[int, int]]:
    offsets = []
    cur = (int(dr), int(dc))
    for _ in range(4):
        offsets.append(cur)
        cur = _quarter_turn(*cur)
    return offsets


def _orbit_positions(pivot: tuple[int, int], dr: int, dc: int) -> list[tuple[int, int]]:
    pr, pc = map(int, pivot)
    return [(pr + rr, pc + cc) for rr, cc in _orbit_offsets(dr, dc)]


def _sorted_emitters(
    emitters: Iterable[tuple[int, int, int]], pivot: tuple[int, int]
) -> list[tuple[int, int, int]]:
    pr, pc = map(int, pivot)
    return sorted(
        ((int(r), int(c), int(color)) for r, c, color in emitters),
        key=lambda item: (
            max(abs(int(item[0]) - pr), abs(int(item[1]) - pc)),
            int(item[0]),
            int(item[1]),
        ),
    )


def _direction_away(pos: tuple[int, int], pivot: tuple[int, int]) -> tuple[int, int]:
    r, c = map(int, pos)
    pr, pc = map(int, pivot)
    assert r != pr and c != pc, "emitters may not share the pivot row or column"
    return (-1 if r < pr else 1, -1 if c < pc else 1)


def _find_pivot(grid: np.ndarray) -> tuple[int, int]:
    pivots = np.argwhere(grid == PIVOT)
    assert pivots.shape == (1, 2), "expected exactly one grey pivot"
    return int(pivots[0, 0]), int(pivots[0, 1])


def _complete_rotational(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    pivot = _find_pivot(out)
    pr, pc = pivot
    colored = np.argwhere((out != BG) & (out != PIVOT))
    for r, c in colored:
        color = int(out[r, c])
        dr, dc = int(r) - pr, int(c) - pc
        assert dr != 0 and dc != 0, "emitter cannot lie on the pivot row or column"
        cur_dr, cur_dc = dr, dc
        for _ in range(3):
            cur_dr, cur_dc = _quarter_turn(cur_dr, cur_dc)
            nr, nc = pr + cur_dr, pc + cur_dc
            assert in_bounds(out.shape, nr, nc), "rotated counterpart left the grid"
            if int(out[nr, nc]) == BG:
                out[nr, nc] = color
            else:
                assert int(out[nr, nc]) == color, "visible pixels contradict rotational completion"
    return out


def _emitters_from_grid(grid: np.ndarray) -> list[tuple[int, int, int]]:
    coords = np.argwhere((grid != BG) & (grid != PIVOT))
    return [(int(r), int(c), int(grid[r, c])) for r, c in coords]


def _render_rays(
    grid: np.ndarray,
    pivot: tuple[int, int],
    emitters: Iterable[tuple[int, int, int]],
) -> tuple[np.ndarray, list[dict]]:
    out = np.asarray(grid, dtype=int).copy()
    occupied: dict[tuple[int, int], tuple[str, int]] = {pivot: ("pivot", PIVOT)}
    for r, c, color in _emitters_from_grid(out):
        occupied[(r, c)] = ("emitter", int(color))

    analyses: list[dict] = []
    for r, c, color in _sorted_emitters(emitters, pivot):
        dr, dc = _direction_away((r, c), pivot)
        nr, nc = r + dr, c + dc
        drawn: list[tuple[int, int]] = []
        blocked_kind = None
        blocked_color = None
        blocked_at = None
        while in_bounds(out.shape, nr, nc) and int(out[nr, nc]) == BG:
            out[nr, nc] = int(color)
            occupied[(nr, nc)] = ("ray", int(color))
            drawn.append((nr, nc))
            nr += dr
            nc += dc
        if in_bounds(out.shape, nr, nc):
            blocked_at = (int(nr), int(nc))
            blocked_kind, blocked_color = occupied[(int(nr), int(nc))]
        analyses.append(
            {
                "emitter": (int(r), int(c)),
                "color": int(color),
                "drawn": drawn,
                "blocked_at": blocked_at,
                "blocked_kind": blocked_kind,
                "blocked_color": None if blocked_color is None else int(blocked_color),
            }
        )
    return out, analyses


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a partial 90-degree pattern around the grey pivot, then let every restored pixel emit an outward diagonal ray.

    Input structure:
    - Black (0) is background and there is exactly one grey (5) pivot pixel.
    - A small set of non-grey colored pixels lies off the pivot row and column.
    - The visible colored pixels are consistent with a single 90-degree rotationally symmetric pattern, but some rotated counterparts are missing as black cells.

    Rule:
    - Around the grey pivot, rotate every visible colored pixel by 90, 180, and 270 degrees.
    - Whenever a rotated counterpart lands on black, fill it with the source color; existing colored counterparts must agree.
    - After completion, every colored pixel emits a ray from the next cell onward in the diagonal direction away from the pivot.
    - Rays are drawn in ascending Chebyshev distance from the pivot, breaking ties by row then column, and each ray stops before the first non-black cell or the grid edge.

    Key insight:
    - The hidden rotational counterparts are also ray emitters, so the final ray layout cannot be determined correctly unless the symmetry completion happens first.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    completed = _complete_rotational(g)
    pivot = _find_pivot(completed)
    emitters = _emitters_from_grid(completed)
    out, _ = _render_rays(completed, pivot, emitters)
    return out


def _grid_from_emitters(
    shape: tuple[int, int],
    pivot: tuple[int, int],
    emitters: Iterable[tuple[int, int, int]],
) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    pr, pc = map(int, pivot)
    grid[pr, pc] = PIVOT
    for r, c, color in emitters:
        grid[int(r), int(c)] = int(color)
    return grid


def _pairwise_separated(emitters: Iterable[tuple[int, int, int]]) -> bool:
    pts = [(int(r), int(c)) for r, c, _ in emitters]
    for i, (r1, c1) in enumerate(pts):
        for r2, c2 in pts[i + 1 :]:
            if max(abs(r1 - r2), abs(c1 - c2)) <= 1:
                return False
    return True


def _choose_pivot(shape: tuple[int, int]) -> tuple[int, int]:
    h, w = map(int, shape)
    row_choices = [r for r in range(max(3, h // 2 - 1), min(h - 3, h // 2 + 1) + 1)]
    col_choices = [c for c in range(max(3, w // 2 - 1), min(w - 3, w // 2 + 1) + 1)]
    return random.choice(row_choices), random.choice(col_choices)


def _candidate_offsets(shape: tuple[int, int], pivot: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    pr, pc = map(int, pivot)
    candidates = []
    for dr in range(-(h - 1), h):
        for dc in range(-(w - 1), w):
            if dr == 0 or dc == 0:
                continue
            orbit = _orbit_positions(pivot, dr, dc)
            if len(set(orbit)) != 4:
                continue
            if not all(in_bounds(shape, r, c) for r, c in orbit):
                continue
            if max(abs(dr), abs(dc)) > min(pr, h - 1 - pr, pc, w - 1 - pc) - 1:
                continue
            candidates.append((int(dr), int(dc)))
    return candidates


def _full_emitters_from_offsets(
    pivot: tuple[int, int],
    offsets: list[tuple[int, int]],
    colors: list[int],
) -> list[tuple[int, int, int]]:
    emitters: list[tuple[int, int, int]] = []
    for (dr, dc), color in zip(offsets, colors):
        for r, c in _orbit_positions(pivot, dr, dc):
            emitters.append((int(r), int(c), int(color)))
    return emitters


def _geometric_ray_cells(
    shape: tuple[int, int], pivot: tuple[int, int], emitter: tuple[int, int, int]
) -> set[tuple[int, int]]:
    r, c, _color = emitter
    dr, dc = _direction_away((r, c), pivot)
    nr, nc = int(r) + dr, int(c) + dc
    cells = set()
    while in_bounds(shape, nr, nc):
        cells.add((int(nr), int(nc)))
        nr += dr
        nc += dc
    return cells


def _has_cross_color_geometric_overlap(
    shape: tuple[int, int], pivot: tuple[int, int], emitters: list[tuple[int, int, int]]
) -> bool:
    rays = [
        (int(color), _geometric_ray_cells(shape, pivot, (int(r), int(c), int(color))))
        for r, c, color in emitters
    ]
    for i, (color1, cells1) in enumerate(rays):
        for color2, cells2 in rays[i + 1 :]:
            if color1 != color2 and cells1 & cells2:
                return True
    return False


def _has_cross_color_block(analyses: list[dict]) -> bool:
    for item in analyses:
        if item["blocked_color"] is not None and item["blocked_color"] != item["color"]:
            return True
    return False


def _has_nontrivial_rays(analyses: list[dict]) -> bool:
    colored_paths = {}
    for item in analyses:
        colored_paths.setdefault(int(item["color"]), 0)
        colored_paths[int(item["color"])] += len(item["drawn"])
    return len(colored_paths) >= 2 and all(length > 0 for length in colored_paths.values())


def _remove_hidden_critical(
    shape: tuple[int, int],
    pivot: tuple[int, int],
    full_emitters: list[tuple[int, int, int]],
    hidden: list[tuple[int, int, int]],
    target_output: np.ndarray,
) -> bool:
    for removed in hidden:
        reduced = [item for item in full_emitters if item != removed]
        reduced_grid = _grid_from_emitters(shape, pivot, reduced)
        reduced_out, _ = _render_rays(reduced_grid, pivot, reduced)
        if not np.array_equal(reduced_out, target_output):
            return True
    return False


def _visible_only_output(inp: np.ndarray) -> np.ndarray:
    pivot = _find_pivot(inp)
    emitters = _emitters_from_grid(inp)
    out, _ = _render_rays(inp, pivot, emitters)
    return out


def _select_visible(full_emitters: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    by_color: dict[int, list[tuple[int, int, int]]] = {}
    for item in full_emitters:
        by_color.setdefault(int(item[2]), []).append(item)
    colors = sorted(by_color)
    counts = random.choice(GEN_VISIBLE_PATTERNS)
    visible = []
    for color, count in zip(colors, counts):
        visible.extend(random.sample(by_color[color], count))
    return visible


def _valid_offset_pair(
    shape: tuple[int, int], pivot: tuple[int, int]
) -> tuple[list[tuple[int, int]], list[int]] | None:
    candidates = _candidate_offsets(shape, pivot)
    if len(candidates) < 2:
        return None

    want_same_radius = random.random() < 0.5
    random.shuffle(candidates)
    for dr1, dc1 in candidates:
        radius1 = max(abs(dr1), abs(dc1))
        others = candidates[:]
        random.shuffle(others)
        for dr2, dc2 in others:
            if (dr1, dc1) == (dr2, dc2):
                continue
            if want_same_radius and max(abs(dr2), abs(dc2)) != radius1:
                continue
            if not want_same_radius and max(abs(dr2), abs(dc2)) == radius1:
                continue
            colors = random.sample(PALETTE, 2)
            full_emitters = _full_emitters_from_offsets(pivot, [(dr1, dc1), (dr2, dc2)], colors)
            if len({(r, c) for r, c, _ in full_emitters}) != len(full_emitters):
                continue
            if not _pairwise_separated(full_emitters):
                continue

            full_grid = _grid_from_emitters(shape, pivot, full_emitters)
            output, analyses = _render_rays(full_grid, pivot, full_emitters)
            if not _has_cross_color_geometric_overlap(shape, pivot, full_emitters):
                continue
            if not _has_cross_color_block(analyses):
                continue
            if not _has_nontrivial_rays(analyses):
                continue
            if np.array_equal(full_grid, output):
                continue
            return [(dr1, dc1), (dr2, dc2)], colors
    return None


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(*GEN_SIZE_RANGE)
    w = random.randint(*GEN_SIZE_RANGE)
    if h % 2 == 0:
        h += 1
    if w % 2 == 0:
        w += 1
    shape = (min(h, 15), min(w, 15))
    pivot = _choose_pivot(shape)

    pair = _valid_offset_pair(shape, pivot)
    if pair is None:
        return None
    offsets, colors = pair
    full_emitters = _full_emitters_from_offsets(pivot, offsets, colors)
    full_grid = _grid_from_emitters(shape, pivot, full_emitters)
    full_output = main(full_grid)

    visible = _select_visible(full_emitters)
    hidden = [item for item in full_emitters if item not in visible]
    if not hidden:
        return None

    inp = _grid_from_emitters(shape, pivot, visible)
    if np.array_equal(_complete_rotational(inp), inp):
        return None
    if np.array_equal(main(inp), _complete_rotational(inp)):
        return None
    if np.array_equal(_visible_only_output(inp), full_output):
        return None
    if not _remove_hidden_critical(shape, pivot, full_emitters, hidden, full_output):
        return None
    if not np.array_equal(main(inp), full_output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetry-ray puzzle input")


if __name__ == "__main__":
    path = Path(__file__)
    mod = SourceFileLoader("self_check_module", str(path)).load_module()
    sample = mod.generate_input()
    print(sample)
    print(mod.main(sample))
