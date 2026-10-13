from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import Color, find_components, get_bg, in_bounds


GRID_SIZE_RANGE = (16, 24)
MIRROR_LENGTH_RANGE = (3, 6)
SERIES_COUNT_RANGE = (4, 7)
SEED_LENGTH_RANGE = (2, 5)
TOTAL_NEW_BLUE_RANGE = (6, 16)
DECOY_COUNT_CHOICES = (0, 1, 2, 3)
MAX_GENERATION_ATTEMPTS = 600
NON_SPECIAL_COLORS = [c for c in Color.NOT_BLACK if c not in {Color.BLUE, Color.RED}]


def _coord_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in coords}


def _is_contiguous_diagonal(coords: set[tuple[int, int]]) -> bool:
    if len(coords) < 2:
        return False
    if len({r - c for r, c in coords}) == 1 or len({r + c for r, c in coords}) == 1:
        rows = sorted(r for r, _c in coords)
        cols = sorted(c for _r, c in coords)
        return (
            rows == list(range(rows[0], rows[-1] + 1))
            and cols == list(range(cols[0], cols[-1] + 1))
            and len(coords) == rows[-1] - rows[0] + 1
        )
    return False


def _ray_seed(g: np.ndarray, bg: int) -> tuple[int, set[tuple[int, int]]]:
    candidates: list[tuple[int, set[tuple[int, int]]]] = []
    for comp in find_components(g, bg=bg, connectivity=8, monochromatic=True):
        coords = _coord_set(comp["coords"])
        if _is_contiguous_diagonal(coords):
            candidates.append((int(comp["color"]), coords))
    assert len(candidates) == 1, "expected one diagonal ray seed"
    return candidates[0]


def _initial_state(shape: tuple[int, int], coords: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    if len({r - c for r, c in coords}) == 1:
        first = min(coords)
        last = max(coords)
        endpoints = ((first, (-1, -1)), (last, (1, 1)))
    else:
        assert len({r + c for r, c in coords}) == 1, "seed must be diagonal"
        top = min(coords)
        bottom = max(coords)
        endpoints = ((top, (-1, 1)), (bottom, (1, -1)))

    options = []
    for (r, c), (dr, dc) in endpoints:
        if in_bounds(shape, r + dr, c + dc):
            options.append((r, c, dr, dc))
    assert len(options) == 1, "seed must have exactly one in-bounds extension"
    return options[0]


def _segment_cells(anchor: tuple[int, int], orientation: str, length: int) -> list[tuple[int, int]]:
    r0, c0 = map(int, anchor)
    if orientation == "horizontal":
        return [(r0, c0 + j) for j in range(int(length))]
    return [(r0 + j, c0) for j in range(int(length))]


def _segment_fits(shape: tuple[int, int], anchor: tuple[int, int], orientation: str, length: int) -> bool:
    return all(in_bounds(shape, r, c) for r, c in _segment_cells(anchor, orientation, length))


def _segment_signature(coords: np.ndarray) -> tuple[str, tuple[int, int], int]:
    pts = sorted(_coord_set(coords))
    rows = sorted({r for r, _c in pts})
    cols = sorted({c for _r, c in pts})
    if len(rows) == 1:
        assert cols == list(range(cols[0], cols[-1] + 1)), "horizontal mirror must be contiguous"
        return "horizontal", (rows[0], cols[0]), len(cols)
    assert len(cols) == 1, "mirror must be a straight segment"
    assert rows == list(range(rows[0], rows[-1] + 1)), "vertical mirror must be contiguous"
    return "vertical", (rows[0], cols[0]), len(rows)


def _visible_mirror_components(g: np.ndarray, bg: int) -> tuple[str, int, tuple[int, int], tuple[int, int]]:
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, colors=[Color.RED])
    assert len(comps) == 2, "expected exactly two visible red mirror segments"
    parsed = [_segment_signature(comp["coords"]) for comp in comps]
    orientations = {item[0] for item in parsed}
    lengths = {item[2] for item in parsed}
    assert len(orientations) == 1 and len(lengths) == 1, "visible mirrors must be parallel and equal-length"
    orientation = parsed[0][0]
    length = parsed[0][2]
    anchors = sorted([item[1] for item in parsed])
    delta = (anchors[1][0] - anchors[0][0], anchors[1][1] - anchors[0][1])
    assert delta != (0, 0), "period vector must be non-zero"
    return orientation, length, anchors[0], delta


def _completed_series(
    shape: tuple[int, int],
    visible_anchor: tuple[int, int],
    orientation: str,
    length: int,
    delta: tuple[int, int],
) -> list[tuple[int, int]]:
    dr, dc = map(int, delta)
    assert dr != 0 or dc != 0
    start = visible_anchor
    prev = (start[0] - dr, start[1] - dc)
    while _segment_fits(shape, prev, orientation, length):
        start = prev
        prev = (start[0] - dr, start[1] - dc)

    anchors = []
    cur = start
    while _segment_fits(shape, cur, orientation, length):
        anchors.append(cur)
        cur = (cur[0] + dr, cur[1] + dc)
    assert len(anchors) >= 2
    return anchors


def _mirror_map(anchors: list[tuple[int, int]], orientation: str, length: int) -> dict[tuple[int, int], str]:
    mapping: dict[tuple[int, int], str] = {}
    for anchor in anchors:
        for cell in _segment_cells(anchor, orientation, length):
            assert cell not in mapping, "mirror segments must not overlap"
            mapping[cell] = orientation
    return mapping


def _trace_ray(
    base: np.ndarray,
    ray_color: int,
    seed: set[tuple[int, int]],
    mirrors: dict[tuple[int, int], str],
) -> np.ndarray:
    out = np.asarray(base, dtype=int).copy()
    r, c, dr, dc = _initial_state(out.shape, seed)
    seen: set[tuple[int, int, int, int]] = set()
    while True:
        state = (r, c, dr, dc)
        assert state not in seen, "ray path must not loop"
        seen.add(state)

        nr, nc = r + dr, c + dc
        if not in_bounds(out.shape, nr, nc):
            break
        if (nr, nc) in mirrors:
            if mirrors[(nr, nc)] == "horizontal":
                dr = -dr
            else:
                dc = -dc
            continue
        out[nr, nc] = ray_color
        r, c = nr, nc
    return out


def _render_with_completed_mirrors(g: np.ndarray) -> tuple[np.ndarray, dict[tuple[int, int], str]]:
    bg = get_bg(g)
    orientation, length, visible_anchor, delta = _visible_mirror_components(g, bg)
    anchors = _completed_series(g.shape, visible_anchor, orientation, length, delta)
    mirrors = _mirror_map(anchors, orientation, length)
    out = np.asarray(g, dtype=int).copy()
    for r, c in mirrors:
        out[r, c] = Color.RED
    return out, mirrors


def _visible_only_output(g: np.ndarray) -> np.ndarray:
    bg = get_bg(g)
    ray_color, seed = _ray_seed(g, bg)
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, colors=[Color.RED])
    mirrors: dict[tuple[int, int], str] = {}
    for comp in comps:
        orientation, anchor, length = _segment_signature(comp["coords"])
        for cell in _segment_cells(anchor, orientation, length):
            mirrors[cell] = orientation
    return _trace_ray(g, ray_color, seed, mirrors)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the diagonal blue ray while reflecting it on the red mirror series completed from the first two visible mirrors.

    Input structure:
    - A black-background grid contains one 8-connected diagonal blue ray seed.
    - Exactly two red horizontal or vertical mirror segments of the same length are shown; they are consecutive samples
      from a longer equally spaced parallel mirror series.
    - Optional straight bars in other colors are decoys and stay transparent to the ray.

    Rule:
    - Infer the mirror-series period vector from the offset between the two visible red segments, then keep translating that
      segment pattern in both directions until the next copy would leave the grid.
    - Paint every implied mirror copy red in the output.
    - From the seed endpoint whose outward diagonal continuation stays in bounds, extend the blue ray; whenever the next
      cell is a real or implied red mirror cell, do not paint that cell and reflect by flipping row direction on horizontal
      mirrors or column direction on vertical mirrors, then continue until the ray exits the grid.

    Key insight:
    - The second visible mirror reveals the array period, so reflections happen at invisible but implied mirror positions as
      well; tracing against only the drawn red cells is insufficient.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)
    ray_color, seed = _ray_seed(g, bg)
    completed, mirrors = _render_with_completed_mirrors(g)
    return _trace_ray(completed, ray_color, seed, mirrors)


def _full_series_components_ok(
    shape: tuple[int, int],
    anchors: list[tuple[int, int]],
    orientation: str,
    length: int,
) -> bool:
    trial = np.full(shape, Color.BLACK, dtype=int)
    for anchor in anchors:
        for r, c in _segment_cells(anchor, orientation, length):
            if trial[r, c] != Color.BLACK:
                return False
            trial[r, c] = Color.RED
    comps = find_components(trial, bg=Color.BLACK, connectivity=4, monochromatic=True, colors=[Color.RED])
    return len(comps) == len(anchors)


def _sample_delta(orientation: str, length: int) -> tuple[int, int]:
    while True:
        dr = random.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
        dc = random.choice([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
        if dr == 0 and dc == 0:
            continue
        if orientation == "horizontal" and dr == 0 and abs(dc) <= length:
            continue
        if orientation == "vertical" and dc == 0 and abs(dr) <= length:
            continue
        return dr, dc


def _sample_visible_anchor(
    shape: tuple[int, int],
    orientation: str,
    length: int,
    delta: tuple[int, int],
) -> tuple[int, int] | None:
    h, w = map(int, shape)
    dr, dc = delta
    if orientation == "horizontal":
        row_min = max(0, -dr)
        row_max = min(h - 1, h - 1 - dr)
        col_min = max(0, -dc)
        col_max = min(w - length, w - length - dc)
    else:
        row_min = max(0, -dr)
        row_max = min(h - length, h - length - dr)
        col_min = max(0, -dc)
        col_max = min(w - 1, w - 1 - dc)
    if row_min > row_max or col_min > col_max:
        return None
    return random.randint(row_min, row_max), random.randint(col_min, col_max)


def _boundary_start(shape: tuple[int, int], pre: tuple[int, int], dr: int, dc: int) -> tuple[int, int]:
    r, c = pre
    while in_bounds(shape, r - dr, c - dc):
        r -= dr
        c -= dc
    return r, c


def _line_cells(start: tuple[int, int], stop: tuple[int, int], dr: int, dc: int) -> list[tuple[int, int]]:
    cells = []
    r, c = start
    while True:
        cells.append((r, c))
        if (r, c) == stop:
            return cells
        r += dr
        c += dc


def _path_after_reflection(
    shape: tuple[int, int],
    pre: tuple[int, int],
    dr: int,
    dc: int,
    mirrors: dict[tuple[int, int], str],
    orientation: str,
) -> list[tuple[int, int]] | None:
    if orientation == "horizontal":
        dr, dc = -dr, dc
    else:
        dr, dc = dr, -dc
    cells = []
    r, c = pre
    while True:
        nr, nc = r + dr, c + dc
        if not in_bounds(shape, nr, nc):
            return cells
        if (nr, nc) in mirrors:
            return None
        cells.append((nr, nc))
        r, c = nr, nc


def _search_seed_and_hit(
    shape: tuple[int, int],
    anchors: list[tuple[int, int]],
    orientation: str,
    length: int,
    visible_indices: tuple[int, int],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    mirrors = _mirror_map(anchors, orientation, length)
    hidden_indices = [i for i in range(len(anchors)) if i not in visible_indices]
    hidden_indices.sort(key=lambda i: abs(i - visible_indices[0]), reverse=True)
    random.shuffle(hidden_indices)
    dirs = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    for idx in hidden_indices:
        hit_cells = _segment_cells(anchors[idx], orientation, length)
        random.shuffle(hit_cells)
        for mr, mc in hit_cells:
            trial_dirs = dirs[:]
            random.shuffle(trial_dirs)
            for dr, dc in trial_dirs:
                pre = (mr - dr, mc - dc)
                if not in_bounds(shape, pre[0], pre[1]) or pre in mirrors:
                    continue
                start = _boundary_start(shape, pre, dr, dc)
                before = _line_cells(start, pre, dr, dc)
                if len(before) < 4 or any(cell in mirrors for cell in before):
                    continue
                after = _path_after_reflection(shape, pre, dr, dc, mirrors, orientation)
                if after is None or len(after) < 2:
                    continue
                seed_max = min(SEED_LENGTH_RANGE[1], len(before) - 2)
                seed_choices = []
                for seed_len in range(SEED_LENGTH_RANGE[0], seed_max + 1):
                    new_blue = (len(before) - seed_len) + len(after)
                    if TOTAL_NEW_BLUE_RANGE[0] <= new_blue <= TOTAL_NEW_BLUE_RANGE[1]:
                        seed_choices.append(seed_len)
                if not seed_choices:
                    continue
                seed_len = random.choice(seed_choices)
                seed = before[:seed_len]
                future = before[seed_len:] + after
                if len(set(future)) != len(future):
                    continue
                return seed, future
    return None


def _place_decoys(
    grid: np.ndarray,
    forbidden: set[tuple[int, int]],
    future_path: list[tuple[int, int]],
) -> None:
    h, w = grid.shape
    future_set = set(future_path)
    taken = {tuple(map(int, rc)) for rc in np.argwhere(grid != Color.BLACK)}
    count = random.choice(DECOY_COUNT_CHOICES)
    if count == 0:
        return
    want_overlap = random.random() < 0.6 and len(future_path) >= 4
    overlap_done = False

    for _ in range(count):
        color = random.choice(NON_SPECIAL_COLORS)
        for _try in range(80):
            orientation = random.choice(["horizontal", "vertical"])
            length = random.randint(2, 7)
            if orientation == "horizontal":
                r = random.randint(0, h - 1)
                c0 = random.randint(0, w - length)
                cells = [(r, c0 + j) for j in range(length)]
            else:
                r0 = random.randint(0, h - length)
                c = random.randint(0, w - 1)
                cells = [(r0 + j, c) for j in range(length)]
            cell_set = set(cells)
            if cell_set & forbidden:
                continue
            if cell_set & taken:
                continue
            overlap = len(cell_set & future_set)
            if want_overlap and not overlap_done:
                if overlap == 0 or overlap > 2:
                    continue
            elif overlap > 2:
                continue
            for r, c in cells:
                grid[r, c] = color
            taken |= cell_set
            overlap_done |= overlap > 0
            break


def _generate_layout() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(*GRID_SIZE_RANGE)
    w = random.randint(*GRID_SIZE_RANGE)
    shape = (h, w)
    orientation = random.choice(["horizontal", "vertical"])
    length = random.randint(*MIRROR_LENGTH_RANGE)
    delta = _sample_delta(orientation, length)
    visible_anchor = _sample_visible_anchor(shape, orientation, length, delta)
    if visible_anchor is None:
        return None

    anchors = _completed_series(shape, visible_anchor, orientation, length, delta)
    if not (SERIES_COUNT_RANGE[0] <= len(anchors) <= SERIES_COUNT_RANGE[1]):
        return None
    if visible_anchor not in anchors:
        return None
    visible_index = anchors.index(visible_anchor)
    if visible_index == 0 or visible_index + 1 >= len(anchors) - 0:
        return None
    if visible_index + 1 >= len(anchors):
        return None
    if visible_index == 0 or visible_index + 1 == len(anchors) - 1:
        return None
    if not _full_series_components_ok(shape, anchors, orientation, length):
        return None

    path = _search_seed_and_hit(shape, anchors, orientation, length, (visible_index, visible_index + 1))
    if path is None:
        return None
    seed_cells, future_blue = path

    grid = np.full(shape, Color.BLACK, dtype=int)
    for r, c in seed_cells:
        grid[r, c] = Color.BLUE
    for idx in (visible_index, visible_index + 1):
        for r, c in _segment_cells(anchors[idx], orientation, length):
            grid[r, c] = Color.RED

    forbidden = set(_mirror_map(anchors, orientation, length)) | set(seed_cells)
    _place_decoys(grid, forbidden, future_blue)

    full_output = main(grid)
    visible_only = _visible_only_output(grid)
    full_blue = {tuple(map(int, rc)) for rc in np.argwhere(full_output == Color.BLUE)}
    visible_blue = {tuple(map(int, rc)) for rc in np.argwhere(visible_only == Color.BLUE)}
    if full_blue == visible_blue:
        return None

    hidden_red = set(_mirror_map(anchors, orientation, length)) - {
        cell
        for idx in (visible_index, visible_index + 1)
        for cell in _segment_cells(anchors[idx], orientation, length)
    }
    if not hidden_red:
        return None
    if not all(full_output[r, c] == Color.RED for r, c in hidden_red):
        return None
    return grid, full_output


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _generate_layout()
        if result is not None:
            grid, _output = result
            return grid
    raise RuntimeError("failed to generate valid implicit-mirror ARC input")
