from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, colors_present, find_components, get_bg, in_bounds


RAY_COLOR = Color.BLUE
BG_COLOR = Color.BLACK
SEPARATOR_COLOR = Color.GREY
BAR_ORIENTATIONS = ("horizontal", "vertical")
DIAGONAL_DIRS = ((-1, -1), (-1, 1), (1, -1), (1, 1))

GEN_RETRIES = 500
SCENE_RETRIES = 120
LOCAL_PLACEMENT_RETRIES = 80

UPPER_HEIGHT_RANGE = (7, 10)
LOWER_HEIGHT_RANGE = (11, 16)
WIDTH_RANGE = (14, 20)
SEED_LEN_RANGE = (2, 4)

UPPER_REFLECTIONS = 1
LOWER_REFLECTIONS_RANGE = (1, 3)
UPPER_TRANSPARENTS = 1
LOWER_TRANSPARENTS_RANGE = (1, 3)


def _single_separator_row(g: np.ndarray, bg: int) -> tuple[int, int]:
    rows = [
        r
        for r in range(g.shape[0])
        if int(g[r, 0]) != bg and np.all(g[r, :] == int(g[r, 0]))
    ]
    assert len(rows) == 1, "expected exactly one solid separator row"
    r = rows[0]
    return r, int(g[r, 0])


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


def _ray_seed(g: np.ndarray, bg: int, ray_color: int = RAY_COLOR) -> set[tuple[int, int]]:
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True)
    candidates: list[set[tuple[int, int]]] = []
    for comp in comps:
        if int(comp["color"]) != int(ray_color):
            continue
        coords = _coord_set(comp["coords"])
        if _is_contiguous_diagonal(coords):
            candidates.append(coords)

    assert len(candidates) == 1, "expected one diagonal ray seed"
    return candidates[0]


def _initial_state(
    shape: tuple[int, int], coords: set[tuple[int, int]]
) -> tuple[int, int, int, int]:
    if len({r - c for r, c in coords}) == 1:
        first = min(coords)
        last = max(coords)
        endpoints = ((first, (-1, -1)), (last, (1, 1)))
    else:
        assert len({r + c for r, c in coords}) == 1
        top = min(coords)
        bottom = max(coords)
        endpoints = ((top, (-1, 1)), (bottom, (1, -1)))

    candidates = []
    for (r, c), (dr, dc) in endpoints:
        if in_bounds(shape, r + dr, c + dc):
            candidates.append((r, c, dr, dc))

    assert len(candidates) == 1, "ray seed must have exactly one in-bounds extension"
    return candidates[0]


def _segment_orientation(coords: set[tuple[int, int]]) -> str:
    rows = sorted({r for r, _c in coords})
    cols = sorted({c for _r, c in coords})
    if len(rows) == 1 and cols == list(range(cols[0], cols[-1] + 1)):
        return "horizontal"
    if len(cols) == 1 and rows == list(range(rows[0], rows[-1] + 1)):
        return "vertical"
    raise AssertionError("line segment must be straight and contiguous")


def _segment_orientations(g: np.ndarray, color: int, bg: int) -> dict[tuple[int, int], str]:
    orientations: dict[tuple[int, int], str] = {}
    found = False
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        if int(comp["color"]) != int(color):
            continue
        found = True
        coords = _coord_set(comp["coords"])
        orientation = _segment_orientation(coords)
        for rc in coords:
            orientations[rc] = orientation
    assert found, "expected at least one segment of the requested color"
    return orientations


def _all_components_straight(g: np.ndarray, color: int, bg: int) -> bool:
    found = False
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        if int(comp["color"]) != int(color):
            continue
        found = True
        try:
            _segment_orientation(_coord_set(comp["coords"]))
        except AssertionError:
            return False
    return found


def _turn_probe_dirs(dirs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    drs = {dr for dr, _dc in dirs}
    dcs = {dc for _dr, dc in dirs}
    if len(drs) == 1:
        probe_dr = -next(iter(drs))
        return [(probe_dr, -1), (probe_dr, 1)]
    assert len(dcs) == 1
    probe_dc = -next(iter(dcs))
    return [(-1, probe_dc), (1, probe_dc)]


def _blue_turns(panel: np.ndarray, ray_color: int) -> list[tuple[int, int, list[tuple[int, int]]]]:
    blue_set = {tuple(map(int, rc)) for rc in np.argwhere(panel == int(ray_color))}
    turns = []
    for r, c in blue_set:
        dirs = [(dr, dc) for dr, dc in DIAGONAL_DIRS if (r + dr, c + dc) in blue_set]
        if len(dirs) != 2:
            continue
        if dirs[0][0] + dirs[1][0] == 0 and dirs[0][1] + dirs[1][1] == 0:
            continue
        turns.append((r, c, dirs))
    return turns


def _looks_transparent_demo_color(
    panel: np.ndarray, color: int, ray_color: int, bg: int
) -> bool:
    comps = [
        _coord_set(comp["coords"])
        for comp in find_components(panel, bg=bg, connectivity=4, monochromatic=True)
        if int(comp["color"]) == int(color)
    ]
    if len(comps) < 2:
        return False

    coords = set().union(*comps)
    rows = {r for r, _c in coords}
    cols = {c for _r, c in coords}

    if len(rows) == 1:
        row = next(iter(rows))
        c0 = min(c for _r, c in coords)
        c1 = max(c for _r, c in coords)
        segment = [int(panel[row, c]) for c in range(c0, c1 + 1)]
    elif len(cols) == 1:
        col = next(iter(cols))
        r0 = min(r for r, _c in coords)
        r1 = max(r for r, _c in coords)
        segment = [int(panel[r, col]) for r in range(r0, r1 + 1)]
    else:
        return False

    return any(v == int(ray_color) for v in segment) and all(
        v in (int(color), int(ray_color)) for v in segment
    )


def _infer_demo_roles(panel: np.ndarray, bg: int) -> tuple[int, int]:
    candidates = colors_present(panel, exclude={bg, RAY_COLOR})
    assert len(candidates) == 2, "demo panel must expose exactly two non-blue colors"

    mirror_votes = []
    for r, c, dirs in _blue_turns(panel, RAY_COLOR):
        visible = []
        for dr, dc in _turn_probe_dirs(dirs):
            rr, cc = r + dr, c + dc
            if in_bounds(panel.shape, rr, cc):
                cell = int(panel[rr, cc])
                if cell not in (int(bg), int(RAY_COLOR)):
                    visible.append(cell)
        if len(set(visible)) == 1:
            mirror_votes.append(visible[0])

    if mirror_votes:
        assert len(set(mirror_votes)) == 1
        mirror_color = mirror_votes[0]
        assert mirror_color in candidates
        transparent_color = candidates[0] if candidates[1] == mirror_color else candidates[1]
        assert _looks_transparent_demo_color(panel, transparent_color, RAY_COLOR, bg)
        return int(mirror_color), int(transparent_color)

    transparent_candidates = [
        color
        for color in candidates
        if _looks_transparent_demo_color(panel, color, RAY_COLOR, bg)
    ]
    assert len(transparent_candidates) == 1, "demo must uniquely identify the transparent color"
    transparent_color = transparent_candidates[0]
    mirror_color = candidates[0] if candidates[1] == transparent_color else candidates[1]
    return int(mirror_color), int(transparent_color)


def _extend_panel_ray(
    panel: np.ndarray,
    mirror_color: int,
    bg: int,
    ray_color: int = RAY_COLOR,
) -> np.ndarray:
    g = np.asarray(panel, dtype=int)
    seed = _ray_seed(g, bg, ray_color)
    r, c, dr, dc = _initial_state(g.shape, seed)
    mirrors = _segment_orientations(g, mirror_color, bg)

    out = g.copy()
    seen: set[tuple[int, int, int, int]] = set()
    while True:
        state = (r, c, dr, dc)
        assert state not in seen, "ray path must not loop"
        seen.add(state)

        nr, nc = r + dr, c + dc
        if not in_bounds(g.shape, nr, nc):
            break

        if int(g[nr, nc]) == int(mirror_color):
            orientation = mirrors.get((nr, nc))
            assert orientation is not None, "mirror cell must belong to a straight mirror segment"
            if orientation == "horizontal":
                dr = -dr
            else:
                dc = -dc
            continue

        out[nr, nc] = int(ray_color)
        r, c = nr, nc

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn which color is reflective from the upper demo panel, then extend the lower blue diagonal ray on the cropped output panel.

    Input structure:
    - A solid separator row splits the grid into an upper demonstration panel and a lower main panel.
    - The upper panel contains a completed blue diagonal ray path together with exactly two other segment colors.
    - The lower panel contains a short blue diagonal seed plus horizontal or vertical segments in those same two colors.

    Rule:
    - Infer from the upper panel which color acts as a mirror: it is the color adjacent to the blue turning point, while the other color is the one the blue path visibly passes through and overwrites.
    - Remove the separator and the upper demo, keeping only the lower panel for the output.
    - Extend the lower blue seed along its only in-bounds diagonal continuation; paint every non-mirror cell blue, overwrite transparent segments, and when the next cell is a mirror segment reflect by flipping the row direction on horizontal mirrors or the column direction on vertical mirrors.

    Key insight:
    - The mirror color is not fixed globally; the upper finished example is a local legend that defines how the same two colors must be interpreted in the lower panel.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    sep_r, _sep_color = _single_separator_row(g, bg)
    upper = g[:sep_r, :]
    lower = g[sep_r + 1 :, :]
    mirror_color, _transparent_color = _infer_demo_roles(upper, bg)
    return _extend_panel_ray(lower, mirror_color, bg, RAY_COLOR)


def _random_start_state(shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    edge = random.choice(("top", "bottom", "left", "right"))
    if edge == "top":
        return 0, random.randint(0, w - 1), 1, random.choice((-1, 1))
    if edge == "bottom":
        return h - 1, random.randint(0, w - 1), -1, random.choice((-1, 1))
    if edge == "left":
        return random.randint(0, h - 1), 0, random.choice((-1, 1)), 1
    return random.randint(0, h - 1), w - 1, random.choice((-1, 1)), -1


def _build_seed_cells(
    shape: tuple[int, int], r0: int, c0: int, dr: int, dc: int, seed_len: int
) -> list[tuple[int, int]] | None:
    cells = []
    for i in range(seed_len):
        rr = r0 + i * dr
        cc = c0 + i * dc
        if not in_bounds(shape, rr, cc):
            return None
        cells.append((rr, cc))
    return cells


def _segment_start_bounds(anchor: int, length: int, limit: int) -> tuple[int, int] | None:
    start_min = max(0, anchor - length + 1)
    start_max = min(anchor, limit - length)
    if start_min > start_max:
        return None

    interior_min = max(start_min, 1)
    interior_max = min(start_max, limit - length - 1) if length < limit else start_max
    if interior_min <= interior_max:
        return interior_min, interior_max
    return start_min, start_max


def _bar_cells_through(
    shape: tuple[int, int], orientation: str, r: int, c: int, length: int
) -> list[tuple[int, int]] | None:
    h, w = shape
    if orientation == "horizontal":
        bounds = _segment_start_bounds(c, length, w)
        if bounds is None:
            return None
        start = random.randint(*bounds)
        return [(r, start + j) for j in range(length)]

    bounds = _segment_start_bounds(r, length, h)
    if bounds is None:
        return None
    start = random.randint(*bounds)
    return [(start + j, c) for j in range(length)]


def _reflected_direction(orientation: str, dr: int, dc: int) -> tuple[int, int]:
    if orientation == "horizontal":
        return -dr, dc
    return dr, -dc


def _walk_segment(
    shape: tuple[int, int],
    r: int,
    c: int,
    dr: int,
    dc: int,
    seg_len: int,
    blocked: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    cells = []
    rr, cc = r, c
    for _ in range(seg_len):
        nr, nc = rr + dr, cc + dc
        if not in_bounds(shape, nr, nc) or (nr, nc) in blocked:
            return None
        cells.append((nr, nc))
        rr, cc = nr, nc
    return cells, (rr, cc)


def _bar_length_choices(shape: tuple[int, int], orientation: str) -> list[int]:
    limit = shape[1] if orientation == "horizontal" else shape[0]
    lo = 3
    hi = min(8, limit)
    if lo > hi:
        return []
    lengths = list(range(lo, hi + 1))
    random.shuffle(lengths)
    return lengths


def _try_place_mirror(
    shape: tuple[int, int],
    r: int,
    c: int,
    dr: int,
    dc: int,
    blocked: set[tuple[int, int]],
) -> tuple[str, list[tuple[int, int]], tuple[int, int], tuple[int, int]] | None:
    mr, mc = r + dr, c + dc
    if not in_bounds(shape, mr, mc) or (mr, mc) in blocked:
        return None

    orientations = list(BAR_ORIENTATIONS)
    random.shuffle(orientations)
    for orientation in orientations:
        new_dr, new_dc = _reflected_direction(orientation, dr, dc)
        next_r, next_c = r + new_dr, c + new_dc
        if not in_bounds(shape, next_r, next_c) or (next_r, next_c) in blocked:
            continue
        for bar_len in _bar_length_choices(shape, orientation):
            bar_cells = _bar_cells_through(shape, orientation, mr, mc, bar_len)
            if bar_cells is None:
                continue
            bar_set = set(bar_cells)
            if bar_set & blocked:
                continue
            return orientation, bar_cells, (mr, mc), (new_dr, new_dc)
    return None


def _place_transparent_bars(
    shape: tuple[int, int],
    full_path: list[tuple[int, int]],
    forbidden: set[tuple[int, int]],
    occupied_bars: set[tuple[int, int]],
    count: int,
) -> tuple[list[list[tuple[int, int]]], list[tuple[int, int]]] | None:
    path_set = set(full_path)
    candidates = [cell for cell in full_path[1:-1] if cell not in forbidden]
    random.shuffle(candidates)
    if len(candidates) < count:
        return None

    bars: list[list[tuple[int, int]]] = []
    anchors: list[tuple[int, int]] = []
    used_anchors: set[tuple[int, int]] = set()

    for _ in range(count):
        placed = False
        random.shuffle(candidates)
        for anchor in candidates:
            if anchor in used_anchors:
                continue
            ar, ac = anchor
            orientations = list(BAR_ORIENTATIONS)
            random.shuffle(orientations)
            for orientation in orientations:
                for length in _bar_length_choices(shape, orientation):
                    cells = _bar_cells_through(shape, orientation, ar, ac, length)
                    if cells is None:
                        continue
                    cell_set = set(cells)
                    if cell_set & occupied_bars:
                        continue
                    if cell_set & path_set != {anchor}:
                        continue
                    bars.append(cells)
                    anchors.append(anchor)
                    used_anchors.add(anchor)
                    occupied_bars |= cell_set
                    placed = True
                    break
                if placed:
                    break
            if placed:
                break
        if not placed:
            return None

    return bars, anchors


def _paint(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _render_scene(
    shape: tuple[int, int],
    mirror_color: int,
    transparent_color: int,
    mirror_bars: list[tuple[str, list[tuple[int, int]]]],
    transparent_bars: list[list[tuple[int, int]]],
    blue_cells: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.full(shape, BG_COLOR, dtype=int)
    for cells in transparent_bars:
        _paint(grid, cells, transparent_color)
    for _orientation, cells in mirror_bars:
        _paint(grid, cells, mirror_color)
    _paint(grid, blue_cells, RAY_COLOR)
    return grid


def _generate_scene(
    shape: tuple[int, int],
    mirror_color: int,
    transparent_color: int,
    n_reflections: int,
    n_transparents: int,
    seed_len: int,
    *,
    forbid_transparent_on_seed: bool,
    min_total_len: int,
) -> dict[str, object] | None:
    for _ in range(SCENE_RETRIES):
        r0, c0, dr, dc = _random_start_state(shape)
        if in_bounds(shape, r0 - dr, c0 - dc):
            continue
        seed_cells = _build_seed_cells(shape, r0, c0, dr, dc, seed_len)
        if seed_cells is None:
            continue

        path_cells = list(seed_cells)
        path_set = set(seed_cells)
        occupied_bars: set[tuple[int, int]] = set()
        mirror_bars: list[tuple[str, list[tuple[int, int]]]] = []
        mirror_anchors: list[tuple[int, int]] = []
        turn_cells: list[tuple[int, int]] = []

        r, c = seed_cells[-1]
        cur_dr, cur_dc = dr, dc
        ok = True

        for _mirror_idx in range(n_reflections):
            placed = None
            seg_choices = list(range(2, min(shape) + 1))
            random.shuffle(seg_choices)
            for seg_len in seg_choices[:LOCAL_PLACEMENT_RETRIES]:
                attempt = _walk_segment(shape, r, c, cur_dr, cur_dc, seg_len, path_set | occupied_bars)
                if attempt is None:
                    continue
                seg_cells, endpoint = attempt
                mirror = _try_place_mirror(
                    shape,
                    endpoint[0],
                    endpoint[1],
                    cur_dr,
                    cur_dc,
                    path_set | occupied_bars | set(seg_cells),
                )
                if mirror is None:
                    continue
                placed = (seg_cells, endpoint, mirror)
                break
            if placed is None:
                ok = False
                break

            seg_cells, (r, c), mirror = placed
            orientation, bar_cells, mirror_anchor, (cur_dr, cur_dc) = mirror
            path_cells.extend(seg_cells)
            path_set |= set(seg_cells)
            occupied_bars |= set(bar_cells)
            mirror_bars.append((orientation, bar_cells))
            mirror_anchors.append(mirror_anchor)
            turn_cells.append((r, c))

        if not ok:
            continue

        final_cells = []
        rr, cc = r, c
        while True:
            nr, nc = rr + cur_dr, cc + cur_dc
            if not in_bounds(shape, nr, nc):
                break
            if (nr, nc) in path_set or (nr, nc) in occupied_bars:
                ok = False
                break
            final_cells.append((nr, nc))
            path_set.add((nr, nc))
            rr, cc = nr, nc

        if not ok or len(final_cells) < 2:
            continue

        path_cells.extend(final_cells)
        if len(path_cells) < min_total_len:
            continue

        forbidden = set(turn_cells)
        if forbid_transparent_on_seed:
            forbidden |= set(seed_cells)
        placed_transparent = _place_transparent_bars(
            shape,
            path_cells,
            forbidden,
            occupied_bars.copy(),
            n_transparents,
        )
        if placed_transparent is None:
            continue
        transparent_bars, transparent_anchors = placed_transparent

        return {
            "shape": shape,
            "seed_cells": seed_cells,
            "path_cells": path_cells,
            "mirror_bars": mirror_bars,
            "mirror_anchors": mirror_anchors,
            "turn_cells": turn_cells,
            "transparent_bars": transparent_bars,
            "transparent_anchors": transparent_anchors,
            "mirror_color": mirror_color,
            "transparent_color": transparent_color,
        }

    return None


def _scene_input(scene: dict[str, object], *, full_path: bool) -> np.ndarray:
    blue_cells = scene["path_cells"] if full_path else scene["seed_cells"]
    return _render_scene(
        scene["shape"],
        int(scene["mirror_color"]),
        int(scene["transparent_color"]),
        scene["mirror_bars"],
        scene["transparent_bars"],
        blue_cells,
    )


def _choose_panel_sizes() -> tuple[int, int, int]:
    for _ in range(100):
        upper_h = random.randint(*UPPER_HEIGHT_RANGE)
        lower_h = random.randint(*LOWER_HEIGHT_RANGE)
        width = random.randint(*WIDTH_RANGE)
        if upper_h + 1 + lower_h <= 30:
            return upper_h, lower_h, width
    raise RuntimeError("failed to sample legal panel sizes")


def _semantic_check(
    full_grid: np.ndarray,
    upper_scene: dict[str, object],
    lower_scene: dict[str, object],
) -> bool:
    upper_h = int(upper_scene["shape"][0])
    upper = full_grid[:upper_h, :]
    lower = full_grid[upper_h + 1 :, :]
    bg = int(get_bg(full_grid))

    mirror_color = int(lower_scene["mirror_color"])
    transparent_color = int(lower_scene["transparent_color"])
    expected = _scene_input(lower_scene, full_path=True)

    try:
        inferred = _infer_demo_roles(upper, bg)
        output = main(full_grid)
    except Exception:
        return False

    if inferred != (mirror_color, transparent_color):
        return False
    if not np.array_equal(output, expected):
        return False
    if np.array_equal(output, lower):
        return False
    if not _all_components_straight(upper, mirror_color, bg):
        return False
    if not _all_components_straight(upper, transparent_color, bg):
        return False
    if not _all_components_straight(lower, mirror_color, bg):
        return False
    if not _all_components_straight(lower, transparent_color, bg):
        return False

    if not _looks_transparent_demo_color(upper, transparent_color, RAY_COLOR, bg):
        return False
    if _looks_transparent_demo_color(upper, mirror_color, RAY_COLOR, bg):
        return False

    try:
        wrong_output = _extend_panel_ray(lower, transparent_color, bg, RAY_COLOR)
    except Exception:
        wrong_output = None
    if wrong_output is not None and np.array_equal(wrong_output, expected):
        return False

    for anchor in lower_scene["transparent_anchors"]:
        r, c = anchor
        if int(lower[r, c]) != transparent_color or int(expected[r, c]) != int(RAY_COLOR):
            return False

    for anchor in lower_scene["mirror_anchors"]:
        r, c = anchor
        if int(lower[r, c]) != mirror_color or int(expected[r, c]) != mirror_color:
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    upper_h, lower_h, width = _choose_panel_sizes()
    mirror_color, transparent_color = random.sample(
        [Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON],
        2,
    )

    upper_scene = _generate_scene(
        (upper_h, width),
        mirror_color,
        transparent_color,
        UPPER_REFLECTIONS,
        UPPER_TRANSPARENTS,
        seed_len=random.randint(1, 2),
        forbid_transparent_on_seed=False,
        min_total_len=6,
    )
    if upper_scene is None:
        return None

    lower_scene = _generate_scene(
        (lower_h, width),
        mirror_color,
        transparent_color,
        random.randint(*LOWER_REFLECTIONS_RANGE),
        random.randint(*LOWER_TRANSPARENTS_RANGE),
        seed_len=random.randint(*SEED_LEN_RANGE),
        forbid_transparent_on_seed=True,
        min_total_len=10,
    )
    if lower_scene is None:
        return None

    upper = _scene_input(upper_scene, full_path=True)
    lower = _scene_input(lower_scene, full_path=False)
    full_grid = np.vstack(
        [
            upper,
            np.full((1, width), SEPARATOR_COLOR, dtype=int),
            lower,
        ]
    )

    if not _semantic_check(full_grid, upper_scene, lower_scene):
        return None
    return full_grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid optical-demo ARC input")
