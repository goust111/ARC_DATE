from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import Color, find_components, get_bg, in_bounds, neighbors4, paste


NW_SE = "nw_se"
NE_SW = "ne_sw"
BLUE = Color.BLUE
GREY = Color.GREY
BG = Color.BLACK
MAJ_COLOR = Color.TEAL
MIN_COLOR = Color.RED

DIAG8 = (
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
)

GEN_RETRIES = 400
CANVAS_MIN = 14
CANVAS_MAX = 18
MARKER_COUNT_CHOICES = (3, 4, 5)
GAP_RANGE = (2, 4)


def _adjacent_to_obstacle(mask: np.ndarray, pos: tuple[int, int]) -> bool:
    return any(in_bounds(mask.shape, rr, cc) and bool(mask[rr, cc]) for rr, cc in neighbors4(*pos))


def _unique_sightline(
    grid: np.ndarray, start: tuple[int, int], obstacle_color: int, bg: int
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    hits: list[tuple[tuple[int, int], tuple[int, int]]] = []
    sr, sc = start
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        r, c = sr + dr, sc + dc
        last_bg: tuple[int, int] | None = None
        while in_bounds(grid.shape, r, c):
            value = int(grid[r, c])
            if value == obstacle_color:
                if last_bg is not None:
                    hits.append(((dr, dc), last_bg))
                break
            if value != bg:
                break
            last_bg = (r, c)
            r += dr
            c += dc
    if len(hits) != 1:
        return None
    return hits[0]


def _contour_options(
    grid: np.ndarray,
    landing: tuple[int, int],
    sight_dir: tuple[int, int],
    obstacle_color: int,
    bg: int,
) -> list[dict]:
    obstacle_mask = grid == obstacle_color
    if sight_dir[0] == 0:
        turns = [(-1, sight_dir[1]), (1, sight_dir[1])]
    else:
        turns = [(sight_dir[0], -1), (sight_dir[0], 1)]

    paths: list[dict] = []
    for turn in turns:
        r, c = landing
        length = 0
        while True:
            nr, nc = r + turn[0], c + turn[1]
            if not in_bounds(grid.shape, nr, nc):
                break
            if int(grid[nr, nc]) != bg:
                break
            if not _adjacent_to_obstacle(obstacle_mask, (nr, nc)):
                break
            r, c = nr, nc
            length += 1
        paths.append({"turn": turn, "length": length, "end": (r, c)})
    return paths


def _orientation_of_component(coords: np.ndarray) -> str:
    cells = {(int(r), int(c)) for r, c in coords}
    has_nw_se = any((r - 1, c - 1) in cells or (r + 1, c + 1) in cells for r, c in cells)
    has_ne_sw = any((r - 1, c + 1) in cells or (r + 1, c - 1) in cells for r, c in cells)
    assert has_nw_se ^ has_ne_sw, "each blue component must be a single diagonal orientation"
    return NW_SE if has_nw_se else NE_SW


def _bar_endpoints(coords: np.ndarray) -> list[tuple[int, int]]:
    cells = {(int(r), int(c)) for r, c in coords}
    endpoints = []
    for r, c in cells:
        deg = sum((r + dr, c + dc) in cells for dr, dc in DIAG8)
        if deg == 1:
            endpoints.append((r, c))
    return endpoints


def _obstacle_color(grid: np.ndarray, bg: int) -> int:
    colors = {int(v) for v in np.unique(grid)}
    colors.discard(int(bg))
    colors.discard(int(BLUE))
    assert len(colors) == 1, "expected exactly one non-blue foreground obstacle color"
    return colors.pop()


def _extract_marker_infos(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    obstacle_color = _obstacle_color(g, bg)
    blue_components = find_components(
        g,
        bg=bg,
        connectivity=8,
        monochromatic=True,
        colors=[BLUE],
    )
    assert 3 <= len(blue_components) <= 5, "expected 3-5 blue diagonal bars"

    infos: list[dict] = []
    for component in blue_components:
        coords = component["coords"]
        assert 2 <= int(coords.shape[0]) <= 3, "bars must be length 2 or 3"
        orientation = _orientation_of_component(coords)
        endpoints = _bar_endpoints(coords)
        assert len(endpoints) == 2, "each bar must have two endpoints"

        visible = []
        for cell in {(int(r), int(c)) for r, c in coords}:
            sight = _unique_sightline(g, cell, obstacle_color, bg)
            if sight is not None:
                visible.append((cell, sight))
        assert len(visible) == 1, "each bar must have exactly one sightline endpoint"

        marker_pos, (sight_dir, landing) = visible[0]
        assert marker_pos in endpoints, "the visible pixel must be an endpoint"
        paths = _contour_options(g, landing, sight_dir, obstacle_color, bg)
        assert len(paths) == 2
        assert paths[0]["length"] != paths[1]["length"], "long and short contour paths must differ"
        long_path = max(paths, key=lambda item: item["length"])
        short_path = min(paths, key=lambda item: item["length"])

        infos.append(
            {
                "coords": tuple(sorted((int(r), int(c)) for r, c in coords)),
                "orientation": orientation,
                "marker": marker_pos,
                "landing": landing,
                "sight_dir": sight_dir,
                "long_end": long_path["end"],
                "short_end": short_path["end"],
                "long_len": int(long_path["length"]),
                "short_len": int(short_path["length"]),
            }
        )

    return infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each diagonal-bar marker to the long or short contour endpoint chosen by its orientation majority.

    Input structure:
    - Black background surrounds one central grey 4-connected obstacle region.
    - Three to five blue one-cell-thick diagonal bars sit around the obstacle.
    - In each blue bar, exactly one endpoint has a unique horizontal or vertical background sightline to the obstacle.

    Rule:
    - Classify each blue bar by diagonal orientation: NW-SE if it has a same-color diagonal neighbor on that axis, otherwise NE-SW.
    - Count bars per orientation; the more frequent orientation is the majority group and the other is the minority group.
    - For each bar's visible endpoint, slide along its unique sightline to the last background cell before the obstacle.
    - From that landing cell, follow the two possible diagonal contour-hugging directions that stay on background cells 4-adjacent to the obstacle.
    - Move majority-group markers to the longer contour endpoint and minority-group markers to the shorter contour endpoint, erase all original blue bars, then recolor majority markers teal and minority markers red.

    Key insight:
    The diagonal orientation does not directly determine output color; it decides whether that marker uses the long or short obstacle-contour route before the final recoloring.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    infos = _extract_marker_infos(g)
    counts = Counter(info["orientation"] for info in infos)
    assert len(counts) == 2 and counts[NW_SE] != counts[NE_SW], "two unequal orientation groups are required"
    majority = NW_SE if counts[NW_SE] > counts[NE_SW] else NE_SW

    out = g.copy()
    out[out == BLUE] = bg
    for info in infos:
        is_majority = info["orientation"] == majority
        dest = info["long_end"] if is_majority else info["short_end"]
        out[dest] = MAJ_COLOR if is_majority else MIN_COLOR
    return out


def _random_obstacle_patch() -> np.ndarray | None:
    for _ in range(120):
        height = random.randint(6, 9)
        lefts = [random.randint(0, 2)]
        widths = [random.randint(2, 4)]
        ok = True
        for _row in range(1, height):
            prev_left = lefts[-1]
            prev_width = widths[-1]
            prev_right = prev_left + prev_width - 1
            choices = []
            for d_left in (-1, 0, 1):
                new_left = prev_left + d_left
                if not 0 <= new_left <= 4:
                    continue
                for d_width in (-1, 0, 1):
                    new_width = prev_width + d_width
                    if not 2 <= new_width <= 4:
                        continue
                    new_right = new_left + new_width - 1
                    if new_left <= prev_right and new_right >= prev_left:
                        choices.append((new_left, new_width))
            if not choices:
                ok = False
                break
            left, width = random.choice(choices)
            lefts.append(left)
            widths.append(width)
        if not ok:
            continue

        area = int(sum(widths))
        if not 15 <= area <= 25:
            continue
        if len(set(lefts)) == 1 and len(set(widths)) == 1:
            continue

        patch_width = max(left + width for left, width in zip(lefts, widths))
        patch = np.full((height, patch_width), BG, dtype=int)
        for r, (left, width) in enumerate(zip(lefts, widths)):
            patch[r, left : left + width] = GREY
        return patch
    return None


def _transform_grid(grid: np.ndarray) -> np.ndarray:
    out = np.rot90(grid, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    return out.copy()


def _place_obstacle_patch(patch: np.ndarray) -> np.ndarray | None:
    h, w = patch.shape
    min_rows = max(CANVAS_MIN, h + 8)
    min_cols = max(CANVAS_MIN, w + 8)
    if min_rows > CANVAS_MAX or min_cols > CANVAS_MAX:
        return None
    rows = random.randint(min_rows, CANVAS_MAX)
    cols = random.randint(min_cols, CANVAS_MAX)
    grid = np.full((rows, cols), BG, dtype=int)
    r0 = random.randint(3, rows - h - 4)
    c0 = random.randint(3, cols - w - 4)
    patch_colored = np.where(patch == GREY, GREY, BG)
    paste(grid, patch_colored, r0, c0)
    return grid


def _bar_options(
    obstacle_grid: np.ndarray,
    marker: tuple[int, int],
    obstacle_color: int,
    bg: int,
) -> list[dict]:
    options: list[dict] = []
    for orientation, step_choices in (
        (NW_SE, [(-1, -1), (1, 1)]),
        (NE_SW, [(-1, 1), (1, -1)]),
    ):
        for dr, dc in step_choices:
            for length in (2, 3):
                coords = [(marker[0] + i * dr, marker[1] + i * dc) for i in range(length)]
                if len(set(coords)) != length:
                    continue
                if not all(in_bounds(obstacle_grid.shape, r, c) for r, c in coords):
                    continue
                if any(int(obstacle_grid[r, c]) != bg for r, c in coords):
                    continue
                trial = obstacle_grid.copy()
                for r, c in coords:
                    trial[r, c] = BLUE
                visible = [
                    cell
                    for cell in coords
                    if _unique_sightline(trial, cell, obstacle_color, bg) is not None
                ]
                if visible != [marker]:
                    continue
                options.append({"orientation": orientation, "coords": tuple(coords)})
    return options


def _enumerate_candidates(obstacle_grid: np.ndarray) -> list[dict]:
    bg = int(get_bg(obstacle_grid))
    obstacle_color = _obstacle_color(obstacle_grid, bg)
    candidates: list[dict] = []
    for r, c in np.argwhere(obstacle_grid == bg):
        marker = (int(r), int(c))
        sight = _unique_sightline(obstacle_grid, marker, obstacle_color, bg)
        if sight is None:
            continue
        sight_dir, landing = sight
        paths = _contour_options(obstacle_grid, landing, sight_dir, obstacle_color, bg)
        lengths = [int(path["length"]) for path in paths]
        if lengths[0] == lengths[1]:
            continue
        if max(lengths) == 0:
            continue
        long_path = max(paths, key=lambda item: item["length"])
        short_path = min(paths, key=lambda item: item["length"])
        for option in _bar_options(obstacle_grid, marker, obstacle_color, bg):
            candidates.append(
                {
                    "marker": marker,
                    "orientation": option["orientation"],
                    "coords": option["coords"],
                    "long_end": long_path["end"],
                    "short_end": short_path["end"],
                    "long_len": int(long_path["length"]),
                    "short_len": int(short_path["length"]),
                }
            )
    return candidates


def _expand8(coords: tuple[tuple[int, int], ...]) -> set[tuple[int, int]]:
    blocked = set()
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                blocked.add((r + dr, c + dc))
    return blocked


def _select_markers(candidates: list[dict]) -> list[dict] | None:
    by_orientation = {
        NW_SE: [candidate for candidate in candidates if candidate["orientation"] == NW_SE],
        NE_SW: [candidate for candidate in candidates if candidate["orientation"] == NE_SW],
    }
    if not by_orientation[NW_SE] or not by_orientation[NE_SW]:
        return None

    for _ in range(300):
        count = random.choice(MARKER_COUNT_CHOICES)
        majority = random.choice([NW_SE, NE_SW])
        minority = NE_SW if majority == NW_SE else NW_SE
        valid_majority_counts = [k for k in range(count // 2 + 1, count) if count - k >= 1]
        if not valid_majority_counts:
            continue
        majority_count = random.choice(valid_majority_counts)
        need = [majority] * majority_count + [minority] * (count - majority_count)
        random.shuffle(need)

        chosen: list[dict] = []
        used_markers: set[tuple[int, int]] = set()
        blocked_cells: set[tuple[int, int]] = set()
        used_dests: set[tuple[int, int]] = set()
        used_bar_cells: set[tuple[int, int]] = set()

        for orientation in need:
            pool = by_orientation[orientation][:]
            random.shuffle(pool)
            pick = None
            for candidate in pool:
                if candidate["marker"] in used_markers:
                    continue
                if any(cell in blocked_cells for cell in candidate["coords"]):
                    continue
                dest = candidate["long_end"] if orientation == majority else candidate["short_end"]
                if dest in used_dests:
                    continue
                if dest in used_bar_cells or dest in candidate["coords"]:
                    continue
                pick = candidate
                break
            if pick is None:
                break
            chosen.append(pick)
            used_markers.add(pick["marker"])
            blocked_cells.update(_expand8(pick["coords"]))
            used_dests.add(pick["long_end"] if orientation == majority else pick["short_end"])
            used_bar_cells.update(pick["coords"])

        if len(chosen) == count:
            return chosen
    return None


def _validate_instance(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    assert set(np.unique(g)).issubset({BG, BLUE, GREY})
    assert 15 <= int(np.count_nonzero(g == GREY)) <= 25

    infos = _extract_marker_infos(g)
    counts = Counter(info["orientation"] for info in infos)
    assert len(infos) in MARKER_COUNT_CHOICES
    assert len(counts) == 2 and counts[NW_SE] != counts[NE_SW]
    majority = NW_SE if counts[NW_SE] > counts[NE_SW] else NE_SW

    dests = []
    for info in infos:
        assert info["long_len"] != info["short_len"]
        dests.append(info["long_end"] if info["orientation"] == majority else info["short_end"])
    assert len(set(dests)) == len(dests), "final destinations must be unique"

    out = main(g)
    assert np.count_nonzero(out == BLUE) == 0
    assert np.array_equal(out == GREY, g == GREY)
    assert int(np.count_nonzero(out == MAJ_COLOR)) == counts[majority]
    assert int(np.count_nonzero(out == MIN_COLOR)) == counts[NE_SW if majority == NW_SE else NW_SE]


def _generate_attempt() -> np.ndarray | None:
    patch = _random_obstacle_patch()
    if patch is None:
        return None
    patch = _transform_grid(patch)
    obstacle_grid = _place_obstacle_patch(patch)
    if obstacle_grid is None:
        return None

    candidates = _enumerate_candidates(obstacle_grid)
    chosen = _select_markers(candidates)
    if chosen is None:
        return None

    grid = obstacle_grid.copy()
    for candidate in chosen:
        for r, c in candidate["coords"]:
            grid[r, c] = BLUE

    grid = _transform_grid(grid)
    try:
        _validate_instance(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for the diagonal-contour puzzle")
