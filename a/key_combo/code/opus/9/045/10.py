from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
RED = 2
YELLOW = 4
GREY = 5
PINK = 6
TEAL = 8

MARKER_COLORS = (RED, YELLOW)
GAP_RANGE = tuple(range(3, 8))
GEN_RETRIES = 800
_GEN_CALL_COUNT = 0


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _is_connected(mask: np.ndarray) -> bool:
    comps = _components(mask)
    return len(comps) == 1


def _normalize(coords: set[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return sorted((r - min_r, c - min_c) for r, c in coords)


def _shape_mask(coords: list[tuple[int, int]]) -> np.ndarray:
    h = max(r for r, _ in coords) + 1
    w = max(c for _, c in coords) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    return mask


def _counts_by_axis(coords: list[tuple[int, int]]) -> tuple[Counter, Counter]:
    row_counts = Counter(r for r, _ in coords)
    col_counts = Counter(c for _, c in coords)
    return row_counts, col_counts


def _touches_with_gap(a: set[tuple[int, int]], b: set[tuple[int, int]], gap: int = 0) -> bool:
    for ar, ac in a:
        for br, bc in b:
            if max(abs(ar - br), abs(ac - bc)) <= gap:
                return True
    return False


def _render(shape: tuple[int, int], cells_by_color: dict[int, set[tuple[int, int]]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for color, cells in cells_by_color.items():
        for r, c in cells:
            grid[r, c] = color
    return grid


def _center_direction(shape: tuple[int, int], marker: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    mr, mc = marker
    center_r = (h - 1) / 2.0
    center_c = (w - 1) / 2.0
    dr = 1 if mr < center_r else -1
    dc = 1 if mc < center_c else -1
    return dr, dc


def _reflect_from_marker(
    source_coords: list[tuple[int, int]],
    marker: tuple[int, int],
    *,
    shape: tuple[int, int] | None = None,
) -> dict:
    mr, mc = marker
    row_matches = [(r, c) for r, c in source_coords if r == mr]
    col_matches = [(r, c) for r, c in source_coords if c == mc]
    row_ok = len(row_matches) == 1 and len(col_matches) == 0
    col_ok = len(col_matches) == 1 and len(row_matches) == 0
    assert row_ok ^ col_ok, "marker must align with exactly one source pixel by row xor column"

    if row_ok:
        anchor = row_matches[0]
        ac = anchor[1]
        target = {(r, mc + ac - c) for r, c in source_coords}
        axis = "row"
    else:
        anchor = col_matches[0]
        ar = anchor[0]
        target = {(mr + ar - r, c) for r, c in source_coords}
        axis = "col"

    if shape is not None:
        assert all(_in_bounds(shape, r, c) for r, c in target)
    assert marker in target
    return {"axis": axis, "anchor": anchor, "marker": marker, "target": target}


def _trace_ray(grid: np.ndarray, marker: tuple[int, int], max_bounces: int = 3) -> dict:
    h, w = grid.shape
    dr, dc = _center_direction((h, w), marker)
    r, c = marker
    last = (r, c)
    path: list[tuple[int, int]] = []
    hits: list[dict] = []
    bounces = 0
    seen_states = set()

    while True:
        state = (r, c, dr, dc, bounces)
        if state in seen_states:
            break
        seen_states.add(state)

        nr, nc = r + dr, c + dc
        if not _in_bounds((h, w), nr, nc):
            break

        cell = int(grid[nr, nc])
        if cell not in (BLACK, GREY):
            hits.append({"obstacle": (nr, nc), "color": cell, "bounce_index": bounces + 1})
            if bounces >= max_bounces:
                break
            if bounces % 2 == 0:
                dr = -dr
            else:
                dc = -dc
            bounces += 1
            continue

        r, c = nr, nc
        last = (r, c)
        path.append(last)

    return {"path": path, "hits": hits, "last": last, "bounces": bounces}


def _solve_with_details(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert max(h, w) <= 30
    assert np.all(g[-1] == BLACK), "input bottom row must be empty"
    assert set(map(int, np.unique(g))) <= {BLACK, RED, YELLOW, TEAL}

    source_coords = [tuple(map(int, rc)) for rc in np.argwhere(g == TEAL)]
    assert 5 <= len(source_coords) <= 8
    source_mask = g == TEAL
    assert _is_connected(source_mask)

    marker_pos: dict[int, tuple[int, int]] = {}
    reflections: dict[int, dict] = {}
    for color in MARKER_COLORS:
        coords = np.argwhere(g == color)
        assert len(coords) == 1
        marker_pos[color] = tuple(map(int, coords[0]))
        reflections[color] = _reflect_from_marker(source_coords, marker_pos[color], shape=g.shape)

    assert reflections[RED]["anchor"] != reflections[YELLOW]["anchor"]

    source_set = set(source_coords)
    cells_by_color = {
        TEAL: source_set,
        RED: set(reflections[RED]["target"]),
        YELLOW: set(reflections[YELLOW]["target"]),
    }
    scene = _render(g.shape, cells_by_color)
    out = scene.copy()

    ray_info: dict[int, dict] = {}
    for color in MARKER_COLORS:
        info = _trace_ray(scene, marker_pos[color])
        ray_info[color] = info
        for r, c in info["path"]:
            out[r, c] = GREY

    red_col = ray_info[RED]["last"][1]
    yellow_col = ray_info[YELLOW]["last"][1]
    if red_col == yellow_col:
        out[h - 1, red_col] = PINK
    else:
        out[h - 1, red_col] = RED
        out[h - 1, yellow_col] = YELLOW

    return {
        "output": out,
        "scene": scene,
        "source": source_set,
        "markers": marker_pos,
        "reflections": reflections,
        "rays": ray_info,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Mirror the teal source into two marker-colored copies, bounce diagonal rays off those copies, and project the ray endpoints onto the bottom row.

    Input structure:
    - A black background grid contains one irregular 4-connected teal object and exactly two singleton markers: one red and one yellow.
    - Each marker shares exactly one row with exactly one teal pixel or exactly one column with exactly one teal pixel, but not both.
    - The bottom row is empty in the input.

    Rule:
    - For each marker, use its unique row or column alignment with the teal object to determine the mirror axis midway between the marker and the aligned teal pixel.
    - Reflect the full teal object across that axis and recolor the reflected copy with the marker color, keeping the original teal object.
    - From each marker, shoot a 45-degree ray toward the grid center; draw traversed cells in grey, and whenever the next diagonal step would enter a non-black, non-grey obstacle, bounce instead.
    - On the first and third obstacle hits flip only the vertical direction component; on the second hit flip only the horizontal component; stop at the boundary or after the third bounce, then paint the bottom-row projection at the ray's final column in the marker color, using magenta if both projections land on the same column.

    Key insight:
    - The same marker first creates a reflected copy and then launches a ray whose path depends on the reflected copies already being present, so the mirror stage and the ray stage cannot be solved independently.
    """
    return _solve_with_details(grid)["output"]


def _random_shape() -> list[tuple[int, int]]:
    for _ in range(300):
        size = random.randint(5, 8)
        coords = {(0, 0)}
        while len(coords) < size:
            frontier = set()
            for r, c in coords:
                for nr, nc in _neighbors4(r, c):
                    if -4 <= nr <= 4 and -4 <= nc <= 4 and (nr, nc) not in coords:
                        frontier.add((nr, nc))
            coords.add(random.choice(tuple(frontier)))

        norm = _normalize(coords)
        mask = _shape_mask(norm)
        h, w = mask.shape
        if h < 2 or w < 2 or h > 5 or w > 5:
            continue
        if np.array_equal(mask, np.flipud(mask)) or np.array_equal(mask, np.fliplr(mask)):
            continue
        row_counts, col_counts = _counts_by_axis(norm)
        if sum(v == 1 for v in row_counts.values()) + sum(v == 1 for v in col_counts.values()) < 3:
            continue
        return norm
    raise RuntimeError("failed to generate a usable source shape")


def _marker_options(
    source_coords: list[tuple[int, int]],
    shape: tuple[int, int],
    axis_kind: str,
) -> list[dict]:
    h, w = shape
    row_counts, col_counts = _counts_by_axis(source_coords)
    source_set = set(source_coords)
    options = []
    for anchor in source_coords:
        ar, ac = anchor
        if axis_kind == "row":
            if row_counts[ar] != 1:
                continue
            for side in (-1, 1):
                for gap in GAP_RANGE:
                    marker = (ar, ac + side * (gap + 1))
                    mr, mc = marker
                    if not (0 <= mr < h - 1 and 0 <= mc < w):
                        continue
                    if mr * 2 == h - 1 or mc * 2 == w - 1:
                        continue
                    if any(c == mc for _, c in source_coords):
                        continue
                    target = {(r, mc + ac - c) for r, c in source_coords}
                    if not all(_in_bounds(shape, r, c) and r < h - 1 for r, c in target):
                        continue
                    if not source_set.isdisjoint(target):
                        continue
                    if _touches_with_gap(source_set, target, gap=1):
                        continue
                    options.append(
                        {
                            "axis": axis_kind,
                            "anchor": anchor,
                            "marker": marker,
                            "target": target,
                        }
                    )
        else:
            if col_counts[ac] != 1:
                continue
            for side in (-1, 1):
                for gap in GAP_RANGE:
                    marker = (ar + side * (gap + 1), ac)
                    mr, mc = marker
                    if not (0 <= mr < h - 1 and 0 <= mc < w):
                        continue
                    if mr * 2 == h - 1 or mc * 2 == w - 1:
                        continue
                    if any(r == mr for r, _ in source_coords):
                        continue
                    target = {(mr + ar - r, c) for r, c in source_coords}
                    if not all(_in_bounds(shape, r, c) and r < h - 1 for r, c in target):
                        continue
                    if not source_set.isdisjoint(target):
                        continue
                    if _touches_with_gap(source_set, target, gap=1):
                        continue
                    options.append(
                        {
                            "axis": axis_kind,
                            "anchor": anchor,
                            "marker": marker,
                            "target": target,
                        }
                    )
    random.shuffle(options)
    return options


def _role_at(
    pos: tuple[int, int],
    source: set[tuple[int, int]],
    reflections: dict[int, dict],
) -> str:
    if pos in source:
        return "source"
    if pos in reflections[RED]["target"]:
        return "red_copy"
    if pos in reflections[YELLOW]["target"]:
        return "yellow_copy"
    return "empty"


def _scene_without_copy(details: dict, removed_color: int) -> np.ndarray:
    shape = details["scene"].shape
    cells_by_color = {
        TEAL: set(details["source"]),
    }
    for color in MARKER_COLORS:
        if color == removed_color:
            cells_by_color[color] = {details["markers"][color]}
        else:
            cells_by_color[color] = set(details["reflections"][color]["target"])
    return _render(shape, cells_by_color)


def _valid_scene(details: dict, scenario: dict) -> bool:
    reflections = details["reflections"]
    rays = details["rays"]
    source = details["source"]

    if scenario["axes"] is not None:
        if reflections[RED]["axis"] != scenario["axes"][0]:
            return False
        if reflections[YELLOW]["axis"] != scenario["axes"][1]:
            return False

    for color in MARKER_COLORS:
        other = YELLOW if color == RED else RED
        if not rays[color]["hits"]:
            return False
        first_hit = rays[color]["hits"][0]["obstacle"]
        if _role_at(first_hit, source, reflections) != ("yellow_copy" if color == RED else "red_copy"):
            return False
        if len(rays[color]["path"]) < 3:
            return False
        without_other = _trace_ray(_scene_without_copy(details, other), details["markers"][color])
        if without_other["last"][1] == rays[color]["last"][1]:
            return False

    if _touches_with_gap(set(reflections[RED]["target"]), set(reflections[YELLOW]["target"]), gap=1):
        return False

    red_end = rays[RED]["last"][1]
    yellow_end = rays[YELLOW]["last"][1]
    if scenario["merge"] and red_end != yellow_end:
        return False
    if scenario["separate"] and red_end == yellow_end:
        return False
    if scenario["multi"] and max(rays[RED]["bounces"], rays[YELLOW]["bounces"]) < 2:
        return False

    overlap = set(rays[RED]["path"]) & set(rays[YELLOW]["path"])
    if len(overlap) > 4:
        return False
    return True


def _scenario() -> dict:
    global _GEN_CALL_COUNT
    scenarios = [
        {"axes": ("row", "row"), "merge": False, "separate": True, "multi": False},
        {"axes": ("col", "col"), "merge": False, "separate": True, "multi": False},
        {"axes": ("row", "col"), "merge": False, "separate": True, "multi": False},
        {"axes": ("col", "row"), "merge": False, "separate": True, "multi": False},
        {"axes": None, "merge": True, "separate": False, "multi": False},
        {"axes": None, "merge": False, "separate": True, "multi": True},
    ]
    scenario = scenarios[_GEN_CALL_COUNT % len(scenarios)]
    _GEN_CALL_COUNT += 1
    return scenario


def _attempt_generate(scenario: dict) -> np.ndarray | None:
    h = random.randint(14, 18)
    w = random.randint(14, 18)
    shape = _random_shape()
    sh = max(r for r, _ in shape) + 1
    sw = max(c for _, c in shape) + 1

    max_r = h - sh - 2
    max_c = w - sw - 1
    if max_r < 1 or max_c < 1:
        return None

    for _ in range(30):
        r0 = random.randint(1, max_r)
        c0 = random.randint(1, max_c)
        source_coords = [(r0 + r, c0 + c) for r, c in shape]
        axes = scenario["axes"] if scenario["axes"] is not None else random.choice(
            [("row", "row"), ("col", "col"), ("row", "col"), ("col", "row")]
        )
        red_opts = _marker_options(source_coords, (h, w), axes[0])
        yellow_opts = _marker_options(source_coords, (h, w), axes[1])
        if not red_opts or not yellow_opts:
            continue

        sample_red = red_opts[: min(40, len(red_opts))]
        sample_yellow = yellow_opts[: min(40, len(yellow_opts))]
        random.shuffle(sample_red)
        random.shuffle(sample_yellow)
        for red_opt in sample_red:
            for yellow_opt in sample_yellow:
                if red_opt["anchor"] == yellow_opt["anchor"]:
                    continue
                if red_opt["marker"] == yellow_opt["marker"]:
                    continue
                if red_opt["marker"] in yellow_opt["target"] or yellow_opt["marker"] in red_opt["target"]:
                    continue
                if not set(red_opt["target"]).isdisjoint(set(yellow_opt["target"])):
                    continue

                inp = np.zeros((h, w), dtype=int)
                for r, c in source_coords:
                    inp[r, c] = TEAL
                inp[red_opt["marker"]] = RED
                inp[yellow_opt["marker"]] = YELLOW

                try:
                    details = _solve_with_details(inp)
                except AssertionError:
                    continue

                if not _valid_scene(details, scenario):
                    continue

                recomputed = main(inp)
                if not np.array_equal(recomputed, details["output"]):
                    continue
                return inp
    return None


def generate_input() -> np.ndarray:
    scenario = _scenario()
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate(scenario)
        if grid is not None:
            return grid

    fallback = {"axes": None, "merge": False, "separate": True, "multi": False}
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate(fallback)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
