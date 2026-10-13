from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

TRACK_COUNT = 4
ZONE_WIDTH = 6
ZONE_GAP = 1
BASE_CONTENT_WIDTH = TRACK_COUNT * ZONE_WIDTH + (TRACK_COUNT - 1) * ZONE_GAP
MAX_GEN_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _neighbors(connectivity: int) -> tuple[tuple[int, int], ...]:
    if int(connectivity) == 4:
        return ((-1, 0), (1, 0), (0, -1), (0, 1))
    return (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )


def _find_components_by_color(
    grid: np.ndarray,
    colors: list[int] | tuple[int, ...],
    *,
    connectivity: int,
) -> list[dict]:
    allowed = {int(c) for c in colors}
    if not allowed:
        return []

    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    offsets = _neighbors(connectivity)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color not in allowed:
                continue
            seen[r, c] = True
            queue: deque[tuple[int, int]] = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in offsets:
                    nr = rr + dr
                    nc = cc + dc
                    if not in_bounds(grid.shape, nr, nc):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": len(coords),
                }
            )
    return components


def _slope_from_component(component: dict) -> dict:
    coords = [tuple(map(int, rc)) for rc in component["coords"]]
    assert len(coords) >= 2, "slopes must have at least two cells"

    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    area = len(coords)
    assert len(set(rows)) == area and max(rows) - min(rows) + 1 == area
    assert len(set(cols)) == area and max(cols) - min(cols) + 1 == area

    diff_keys = {c - r for r, c in coords}
    sum_keys = {c + r for r, c in coords}
    if len(diff_keys) == 1:
        dc = 1
    else:
        assert len(sum_keys) == 1, "non-sand components must be straight diagonal ramps"
        dc = -1

    ordered = sorted(coords)
    for (r0, c0), (r1, c1) in zip(ordered, ordered[1:]):
        assert r1 == r0 + 1 and c1 == c0 + dc, "ramp cells must be contiguous along one diagonal"

    return {
        "color": int(component["color"]),
        "coords": ordered,
        "coord_set": set(ordered),
        "dc": dc,
        "top": ordered[0],
        "tail": ordered[-1],
    }


def _parse_scene(grid: np.ndarray) -> tuple[list[dict], dict[tuple[int, int], dict], set[tuple[int, int]], int]:
    bg = get_bg(grid)
    assert bg == BLACK, "expected black background"

    platform_components = _find_components_by_color(grid, [GREY], connectivity=4)
    assert platform_components, "expected at least one grey platform"
    platform_cells: set[tuple[int, int]] = set()
    for component in platform_components:
        r0, _c0, r1, _c1 = component["bbox"]
        assert r0 == r1, "platforms must be horizontal one-cell-thick segments"
        assert component["area"] >= 2, "platforms should have visible length"
        platform_cells.update(component["coords"])

    fg_colors = sorted(set(map(int, np.unique(grid))) - {bg, GREY})
    components = _find_components_by_color(grid, fg_colors, connectivity=8)
    sands: list[dict] = []
    slopes: list[dict] = []
    slope_lookup: dict[tuple[int, int], dict] = {}

    for component in components:
        if component["area"] == 1:
            r, c = component["coords"][0]
            sands.append({"pos": (int(r), int(c)), "color": int(component["color"])})
            continue
        slope = _slope_from_component(component)
        slopes.append(slope)
        for cell in slope["coords"]:
            assert cell not in slope_lookup, "ramps must not overlap"
            slope_lookup[cell] = slope

    assert sands, "expected at least one sand grain"
    assert slopes, "expected at least one ramp"
    return sands, slope_lookup, platform_cells, bg


def _settle_sand(
    start: tuple[int, int],
    start_color: int,
    shape: tuple[int, int],
    platform_cells: set[tuple[int, int]],
    slope_lookup: dict[tuple[int, int], dict],
) -> tuple[int, int, int]:
    h, _w = map(int, shape)
    r, c = map(int, start)
    color = int(start_color)

    for _ in range(h * max(shape) * 2):
        assert in_bounds(shape, r, c)
        below = (r + 1, c)
        assert below[0] < h, "every grain must settle before leaving the grid"
        if below in platform_cells:
            return r, c, color
        slope = slope_lookup.get(below)
        if slope is not None:
            tail_r, tail_c = slope["tail"]
            r = int(tail_r)
            c = int(tail_c + slope["dc"])
            assert in_bounds(shape, r, c), "ramp slide left the grid"
            color = int(slope["color"])
            continue
        r += 1

    raise AssertionError("sand path did not terminate")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Let each sand grain fall onto grey platforms, slide along colored diagonal ramps, and adopt the last ramp's color.

    Input structure:
    - The grid has a black background with one-cell grey horizontal platforms.
    - Colored ramps are monochromatic diagonal components made of contiguous 45-degree cells.
    - Colored single-cell sand grains start above the structures in distinct columns.

    Rule:
    - Remove the starting sand grains from their source cells in the output.
    - Each grain falls vertically until the cell directly below it is either a ramp cell or a grey platform.
    - If the cell below is a ramp, the grain slides diagonally downward along that ramp's downhill direction until it passes the ramp tail, then resumes falling vertically.
    - A grain stops when the cell directly below it is grey; its final color is the last ramp color it touched, or its original color if it never touched any ramp.
    - Platforms and ramps themselves stay unchanged.

    Key insight:
    - The output records the grains' motion history: geometry determines the resting column, and the last ramp on the route stamps the resting color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sands, slope_lookup, platform_cells, bg = _parse_scene(g)
    out = g.copy()
    for sand in sands:
        sr, sc = sand["pos"]
        out[sr, sc] = bg

    occupied: set[tuple[int, int]] = set()
    for sand in sorted(sands, key=lambda item: item["pos"]):
        rr, cc, color = _settle_sand(sand["pos"], sand["color"], g.shape, platform_cells, slope_lookup)
        assert (rr, cc) not in occupied, "settled grains must not overlap"
        occupied.add((rr, cc))
        out[rr, cc] = int(color)

    return out


def _zone_bounds(left_margin: int, idx: int) -> tuple[int, int]:
    left = int(left_margin) + idx * (ZONE_WIDTH + ZONE_GAP)
    return left, left + ZONE_WIDTH - 1


def _search_lengths(
    current_c: int,
    orientations: list[int],
    left: int,
    right: int,
) -> list[int] | None:
    if not orientations:
        return []
    lengths = [2, 3]
    random.shuffle(lengths)
    dc = int(orientations[0])
    for length in lengths:
        next_c = current_c + dc * length
        if not (left + 1 <= next_c <= right - 1):
            continue
        tail = _search_lengths(next_c, orientations[1:], left, right)
        if tail is not None:
            return [length] + tail
    return None


def _sample_orientations(kind: str) -> list[int]:
    if kind == "direct":
        return []
    if kind == "single":
        return [random.choice((-1, 1))]
    if kind == "multi":
        first = random.choice((-1, 1))
        count = random.choice((2, 3))
        seq = [first, -first]
        if count == 3:
            seq.append(random.choice((-1, 1)))
        return seq
    assert kind == "extra"
    if random.random() < 0.5:
        return [random.choice((-1, 1))]
    return [random.choice((-1, 1)), random.choice((-1, 1))]


def _build_track(
    kind: str,
    group: str,
    zone: tuple[int, int],
    orientations: list[int],
    slope_colors: list[int],
) -> dict | None:
    left, right = zone
    start_cols = [left + 2, left + 3]
    random.shuffle(start_cols)

    for start_c in start_cols:
        lengths = _search_lengths(start_c, orientations, left, right)
        if lengths is None:
            continue
        start_r = random.randint(0, 2)
        current_r = start_r
        current_c = start_c
        slopes = []
        for dc, length, color in zip(orientations, lengths, slope_colors):
            gap = random.randint(1, 3)
            top_r = current_r + gap + 1
            cells = [(top_r + step, current_c + dc * step) for step in range(length)]
            slopes.append(
                {
                    "color": int(color),
                    "dc": int(dc),
                    "length": int(length),
                    "top_r": int(top_r),
                    "hit_c": int(current_c),
                    "cells": cells,
                }
            )
            current_r = top_r + length - 1
            current_c = current_c + dc * length
        return {
            "kind": kind,
            "group": group,
            "zone": zone,
            "start": (int(start_r), int(start_c)),
            "slopes": slopes,
            "arrival": (int(current_r), int(current_c)),
        }
    return None


def _assign_sand_colors(tracks: list[dict]) -> None:
    direct_a = next(track for track in tracks if track["group"] == "A" and track["kind"] == "direct")
    multi_a = next(track for track in tracks if track["group"] == "A" and track["kind"] == "multi")
    ordered = [direct_a, multi_a] + [track for track in tracks if track is not direct_a and track is not multi_a]

    used: set[int] = set()
    for track in ordered:
        forbidden = set()
        if track["slopes"]:
            forbidden.add(int(track["slopes"][-1]["color"]))
        if track is direct_a:
            forbidden.add(int(multi_a["slopes"][-1]["color"]))
        if track is multi_a:
            forbidden.add(int(track["slopes"][0]["color"]))
            forbidden.add(int(direct_a.get("sand_color", -1)))
        candidates = [c for c in COLOR_POOL if c not in forbidden and c not in used]
        if not candidates:
            candidates = [c for c in COLOR_POOL if c not in forbidden]
        track["sand_color"] = int(random.choice(candidates))
        used.add(track["sand_color"])
        track["final_color"] = track["sand_color"] if not track["slopes"] else int(track["slopes"][-1]["color"])


def _set_platform_rows(tracks: list[dict]) -> bool:
    group_a = [track for track in tracks if track["group"] == "A"]
    group_b = [track for track in tracks if track["group"] == "B"]

    platform_a = max(track["arrival"][0] for track in group_a) + random.randint(3, 5)
    platform_b = max(
        max(track["arrival"][0] for track in group_b) + random.randint(3, 5),
        platform_a + random.randint(3, 5),
    )
    if platform_b > 28:
        return False

    for track in group_a:
        track["platform_row"] = int(platform_a)
    for track in group_b:
        track["platform_row"] = int(platform_b)
    for track in tracks:
        final_r = int(track["platform_row"] - 1)
        final_c = int(track["arrival"][1])
        track["final"] = (final_r, final_c)
    return True


def _platform_spans(tracks: list[dict]) -> dict[str, tuple[int, int]]:
    spans: dict[str, tuple[int, int]] = {}
    for group in ("A", "B"):
        cols = [track["final"][1] for track in tracks if track["group"] == group]
        spans[group] = (min(cols) - 1, max(cols) + 1)
    return spans


def _render_input(tracks: list[dict], spans: dict[str, tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)

    for group in ("A", "B"):
        platform_row = next(track["platform_row"] for track in tracks if track["group"] == group)
        c0, c1 = spans[group]
        for c in range(c0, c1 + 1):
            assert grid[platform_row, c] == BLACK
            grid[platform_row, c] = GREY

    for track in tracks:
        for slope in track["slopes"]:
            for r, c in slope["cells"]:
                assert grid[r, c] == BLACK
                grid[r, c] = int(slope["color"])

    for track in tracks:
        sr, sc = track["start"]
        assert grid[sr, sc] == BLACK
        grid[sr, sc] = int(track["sand_color"])

    return grid


def _render_expected(inp: np.ndarray, tracks: list[dict]) -> np.ndarray:
    out = inp.copy()
    for track in tracks:
        sr, sc = track["start"]
        out[sr, sc] = BLACK
    for track in tracks:
        fr, fc = track["final"]
        out[fr, fc] = int(track["final_color"])
    return out


def _validate_semantics(grid: np.ndarray, tracks: list[dict], spans: dict[str, tuple[int, int]]) -> None:
    h, w = map(int, grid.shape)
    assert len({track["start"][1] for track in tracks}) == len(tracks)
    assert any(not track["slopes"] for track in tracks)
    assert any(len(track["slopes"]) == 1 for track in tracks)
    assert any(len(track["slopes"]) >= 2 for track in tracks)
    assert {slope["dc"] for track in tracks for slope in track["slopes"]} == {-1, 1}

    multi_a = next(track for track in tracks if track["group"] == "A" and track["kind"] == "multi")
    direct_a = next(track for track in tracks if track["group"] == "A" and track["kind"] == "direct")
    assert len(multi_a["slopes"]) >= 2
    assert multi_a["slopes"][0]["color"] != multi_a["slopes"][-1]["color"]
    assert direct_a["final_color"] != multi_a["final_color"]

    finals = [track["final"] for track in tracks]
    assert len(set(finals)) == len(finals)
    for group in ("A", "B"):
        group_tracks = [track for track in tracks if track["group"] == group]
        assert len({track["platform_row"] for track in group_tracks}) == 1
        c0, c1 = spans[group]
        assert 0 <= c0 <= c1 < w

    for track in tracks:
        assert in_bounds(grid.shape, *track["start"])
        assert in_bounds(grid.shape, *track["final"])
        if track["slopes"]:
            assert track["sand_color"] != track["final_color"]

        current_r, current_c = track["start"]
        for slope in track["slopes"]:
            for rr in range(current_r + 1, slope["top_r"]):
                assert grid[rr, current_c] == BLACK
            for step, (rr, cc) in enumerate(slope["cells"]):
                assert grid[rr, cc] == slope["color"]
                path_c = current_c + slope["dc"] * (step + 1)
                assert grid[rr, path_c] == BLACK
            current_r = slope["top_r"] + slope["length"] - 1
            current_c = current_c + slope["dc"] * slope["length"]
        platform_row = track["platform_row"]
        for rr in range(current_r + 1, platform_row):
            assert grid[rr, current_c] == BLACK
        assert grid[platform_row, current_c] == GREY
        assert track["final"] == (platform_row - 1, current_c)
        assert 0 <= platform_row < h


def _build_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    left_margin = random.randint(0, 2)
    max_right_margin = 30 - (left_margin + BASE_CONTENT_WIDTH)
    right_margin = random.randint(0, max_right_margin)
    width = left_margin + BASE_CONTENT_WIDTH + right_margin

    group_a_kinds = ["direct", "multi"]
    group_b_kinds = ["single", "extra"]
    random.shuffle(group_a_kinds)
    random.shuffle(group_b_kinds)
    kinds = group_a_kinds + group_b_kinds
    groups = ["A", "A", "B", "B"]

    orientation_sets = [_sample_orientations(kind) for kind in kinds]
    total_slopes = sum(len(seq) for seq in orientation_sets)
    assert total_slopes <= len(COLOR_POOL)
    slope_colors = iter(random.sample(COLOR_POOL, total_slopes))

    tracks: list[dict] = []
    for idx, (kind, group, orientations) in enumerate(zip(kinds, groups, orientation_sets)):
        zone = _zone_bounds(left_margin, idx)
        colors = [next(slope_colors) for _ in range(len(orientations))]
        track = _build_track(kind, group, zone, orientations, colors)
        if track is None:
            return None
        tracks.append(track)

    if not _set_platform_rows(tracks):
        return None
    _assign_sand_colors(tracks)

    spans = _platform_spans(tracks)
    height = next(track["platform_row"] for track in tracks if track["group"] == "B") + 2
    if height > 30:
        return None

    try:
        inp = _render_input(tracks, spans, (height, width))
        expected = _render_expected(inp, tracks)
        _validate_semantics(inp, tracks, spans)
    except AssertionError:
        return None

    if not np.array_equal(main(inp), expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        built = _build_attempt()
        if built is not None:
            inp, _expected = built
            return inp
    raise RuntimeError("failed to generate a valid slope-guided sand example")
