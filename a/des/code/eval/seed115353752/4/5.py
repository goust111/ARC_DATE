import random

import numpy as np


BLACK = 0
RED = 2
YELLOW = 4
GREY = 5
MAROON = 9
SPECIAL_COLORS = {BLACK, RED, YELLOW, GREY, MAROON}
TRACK_COLORS = [1, 3, 6, 7, 8]

GRID_H_RANGE = (22, 30)
GRID_W_RANGE = (28, 38)
ROOT_H_RANGE = (8, 11)
ROOT_W_RANGE = (12, 17)
ISOLATED_COUNT_CHOICES = [1, 1, 2]
ISOLATED_H_RANGE = (5, 8)
ISOLATED_W_RANGE = (5, 9)
GENERATION_RETRIES = 400
MARKER_RETRIES = 200
PLACEMENT_RETRIES = 200


def _perimeter_cycle(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cycle: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cycle.append((r0, c))
    for r in range(r0 + 1, r1 + 1):
        cycle.append((r, c1))
    for c in range(c1 - 1, c0 - 1, -1):
        cycle.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        cycle.append((r, c0))
    return cycle


def _bbox_of_color(grid: np.ndarray, color: int) -> tuple[int, int, int, int]:
    pts = np.argwhere(grid == int(color))
    assert pts.size, "track color must remain visible"
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _build_track(color: int, bbox: tuple[int, int, int, int]) -> dict[str, object]:
    r0, c0, r1, c1 = bbox
    cycle = _perimeter_cycle(r0, c0, r1, c1)
    return {
        "color": int(color),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "cycle": cycle,
        "coord_to_idx": {coord: idx for idx, coord in enumerate(cycle)},
    }


def _reconstruct_tracks(
    grid: np.ndarray,
) -> tuple[dict[int, dict[str, object]], dict[tuple[int, int], list[int]], dict[tuple[int, int], tuple[int, int]], dict[tuple[int, int], int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    track_colors = [int(c) for c in np.unique(g) if int(c) not in SPECIAL_COLORS]
    assert track_colors, "expected at least one track"

    tracks: dict[int, dict[str, object]] = {}
    by_coord: dict[tuple[int, int], list[int]] = {}
    for color in track_colors:
        bbox = _bbox_of_color(g, color)
        track = _build_track(color, bbox)
        allowed = {int(color), YELLOW, GREY, RED, MAROON}
        for r, c in track["cycle"]:
            assert int(g[r, c]) in allowed, "perimeter reconstruction failed"
            by_coord.setdefault((int(r), int(c)), []).append(int(color))
        tracks[int(color)] = track

    yellow_positions = [(int(r), int(c)) for r, c in np.argwhere(g == YELLOW)]
    assert len(yellow_positions) == len(tracks), "expected one marker per track"
    for pos in yellow_positions:
        owners = by_coord.get(pos, [])
        assert len(owners) == 1, "marker must belong to exactly one track"
        tracks[int(owners[0])]["marker"] = pos
    for track in tracks.values():
        assert "marker" in track, "every track must own a marker"

    intersections: dict[tuple[int, int], tuple[int, int]] = {}
    for r, c in np.argwhere(g == GREY):
        pos = (int(r), int(c))
        owners = by_coord.get(pos, [])
        assert len(owners) == 2, "grey pixel must belong to exactly two tracks"
        intersections[pos] = (int(owners[0]), int(owners[1]))

    switches: dict[tuple[int, int], int] = {}
    for r, c in intersections:
        found: list[int] = []
        if r > 0 and int(g[r - 1, c]) in (RED, MAROON):
            found.append(int(g[r - 1, c]))
        if r + 1 < h and int(g[r + 1, c]) in (RED, MAROON):
            found.append(int(g[r + 1, c]))
        assert len(found) == 1, "each intersection must have one vertical switch"
        switches[(int(r), int(c))] = int(found[0])

    for color, track in tracks.items():
        track["intersections"] = [pos for pos, owners in intersections.items() if int(color) in owners]
    return tracks, by_coord, intersections, switches


def _first_intersection(track: dict[str, object]) -> tuple[int, int] | None:
    intersections = track["intersections"]
    if not intersections:
        return None
    cycle = track["cycle"]
    coord_to_idx = track["coord_to_idx"]
    marker = track["marker"]
    marker_idx = int(coord_to_idx[marker])
    best_delta = None
    best_pos = None
    for pos in intersections:
        delta = (int(coord_to_idx[pos]) - marker_idx) % len(cycle)
        if delta == 0:
            delta = len(cycle)
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_pos = pos
    return best_pos


def _final_positions(
    tracks: dict[int, dict[str, object]],
    intersections: dict[tuple[int, int], tuple[int, int]],
    switches: dict[tuple[int, int], int],
) -> dict[int, tuple[int, int]]:
    finals: dict[int, tuple[int, int]] = {}
    for color, track in tracks.items():
        hit = _first_intersection(track)
        if hit is None:
            finals[int(color)] = track["marker"]
            continue
        if int(switches[hit]) == RED:
            finals[int(color)] = hit
            continue
        owners = intersections[hit]
        other_color = int(owners[0] if owners[1] == int(color) else owners[1])
        other_track = tracks[other_color]
        other_idx = int(other_track["coord_to_idx"][hit])
        other_cycle = other_track["cycle"]
        finals[int(color)] = other_cycle[(other_idx + 1) % len(other_cycle)]
    return finals


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each yellow marker clockwise to its first grey crossing.
    Red switches keep the marker on the same track; maroon switches transfer it
    to the other track and advance one clockwise step there.
    """
    g = np.asarray(grid, dtype=int)
    tracks, by_coord, intersections, switches = _reconstruct_tracks(g)
    finals = _final_positions(tracks, intersections, switches)

    out = g.copy()
    for color, track in tracks.items():
        src = track["marker"]
        if len(by_coord[src]) >= 2:
            out[src] = GREY
        else:
            out[src] = int(color)
    for pos in finals.values():
        out[pos] = YELLOW
    return out


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _branch_kinds(
    tracks: dict[int, dict[str, object]],
    switches: dict[tuple[int, int], int],
) -> dict[int, str]:
    kinds: dict[int, str] = {}
    for color, track in tracks.items():
        hit = _first_intersection(track)
        if hit is None:
            kinds[int(color)] = "none"
        elif int(switches[hit]) == RED:
            kinds[int(color)] = "red"
        else:
            kinds[int(color)] = "maroon"
    return kinds


def _choose_root_bbox(h: int, w: int) -> tuple[int, int, int, int]:
    for _ in range(PLACEMENT_RETRIES):
        root_h = random.randint(*ROOT_H_RANGE)
        root_w = random.randint(*ROOT_W_RANGE)
        if root_h + 12 > h or root_w + 18 > w:
            continue
        r0 = random.randint(5, h - root_h - 6)
        c0 = random.randint(8, w - root_w - 9)
        r1 = r0 + root_h - 1
        c1 = c0 + root_w - 1
        return r0, c0, r1, c1
    raise AssertionError("failed to place root track")


def _place_isolated_boxes(
    h: int,
    w: int,
    occupied_boxes: list[tuple[int, int, int, int]],
    count: int,
) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    for _ in range(count):
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            ih = random.randint(*ISOLATED_H_RANGE)
            iw = random.randint(*ISOLATED_W_RANGE)
            r0 = random.randint(1, h - ih - 1)
            c0 = random.randint(1, w - iw - 1)
            box = (int(r0), int(c0), int(r0 + ih - 1), int(c0 + iw - 1))
            if any(not _boxes_separated(box, other, 2) for other in occupied_boxes + boxes):
                continue
            boxes.append(box)
            placed = True
            break
        if not placed:
            raise AssertionError("failed to place isolated track")
    return boxes


def _logical_tracks_and_switches() -> tuple[
    int,
    int,
    list[dict[str, object]],
    dict[tuple[int, int], tuple[int, int]],
    dict[tuple[int, int], int],
    dict[tuple[int, int], int],
]:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    isolated_count = random.choice(ISOLATED_COUNT_CHOICES)
    total_tracks = 3 + isolated_count
    colors = random.sample(TRACK_COLORS, total_tracks)

    root_bbox = _choose_root_bbox(h, w)
    rr0, rc0, rr1, rc1 = root_bbox
    root_w = rc1 - rc0 + 1
    left_x_max = rc0 + root_w // 2 - 1
    right_x_min = rc0 + root_w // 2 + 1
    assert left_x_max >= rc0 + 2 and right_x_min <= rc1 - 2

    left_x = random.randint(rc0 + 2, left_x_max)
    right_x = random.randint(max(right_x_min, left_x + 4), rc1 - 2)

    left_bbox = (
        random.randint(1, rr0 - 3),
        random.randint(1, rc0 - 4),
        random.randint(rr1 + 3, h - 2),
        left_x,
    )
    right_bbox = (
        random.randint(1, rr0 - 3),
        right_x,
        random.randint(rr1 + 3, h - 2),
        random.randint(rc1 + 3, w - 2),
    )

    occupied_boxes = [root_bbox, left_bbox, right_bbox]
    isolated_boxes = _place_isolated_boxes(h, w, occupied_boxes, isolated_count)

    track_specs = [
        _build_track(colors[0], root_bbox),
        _build_track(colors[1], left_bbox),
        _build_track(colors[2], right_bbox),
    ]
    for color, box in zip(colors[3:], isolated_boxes):
        track_specs.append(_build_track(color, box))

    intersections = {
        (rr0, left_x): (track_specs[0]["color"], track_specs[1]["color"]),
        (rr1, left_x): (track_specs[0]["color"], track_specs[1]["color"]),
        (rr0, right_x): (track_specs[0]["color"], track_specs[2]["color"]),
        (rr1, right_x): (track_specs[0]["color"], track_specs[2]["color"]),
    }
    if random.random() < 0.5:
        left_switch_color, right_switch_color = RED, MAROON
    else:
        left_switch_color, right_switch_color = MAROON, RED
    switch_colors = {
        (rr0, left_x): int(left_switch_color),
        (rr1, left_x): int(left_switch_color),
        (rr0, right_x): int(right_switch_color),
        (rr1, right_x): int(right_switch_color),
    }
    switch_cells = {
        (rr0 - 1, left_x): int(left_switch_color),
        (rr1 - 1, left_x): int(left_switch_color),
        (rr0 + 1, right_x): int(right_switch_color),
        (rr1 + 1, right_x): int(right_switch_color),
    }

    coord_tracks: dict[tuple[int, int], list[int]] = {}
    for track in track_specs:
        for pos in track["cycle"]:
            coord_tracks.setdefault(pos, []).append(int(track["color"]))
    for pos, owners in coord_tracks.items():
        if len(owners) == 1:
            continue
        assert pos in intersections, "unexpected shared perimeter cell"
        assert len(owners) == 2 and set(owners) == set(intersections[pos]), "bad intersection ownership"
    for pos in switch_cells:
        assert pos in coord_tracks and len(coord_tracks[pos]) == 1, "switch must sit on exactly one track branch"

    return h, w, track_specs, intersections, switch_colors, switch_cells


def _render_input(
    h: int,
    w: int,
    track_specs: list[dict[str, object]],
    intersections: dict[tuple[int, int], tuple[int, int]],
    switch_cells: dict[tuple[int, int], int],
) -> np.ndarray:
    grid = np.full((h, w), BLACK, dtype=int)
    coord_tracks: dict[tuple[int, int], list[int]] = {}
    for track in track_specs:
        for pos in track["cycle"]:
            coord_tracks.setdefault(pos, []).append(int(track["color"]))
    for pos, owners in coord_tracks.items():
        if len(owners) == 1:
            grid[pos] = int(owners[0])
        else:
            grid[pos] = GREY
    for pos, color in switch_cells.items():
        grid[pos] = int(color)
    for track in track_specs:
        grid[track["marker"]] = YELLOW
    return grid


def _pick_markers(
    track_specs: list[dict[str, object]],
    intersections: dict[tuple[int, int], tuple[int, int]],
    switch_colors: dict[tuple[int, int], int],
    switch_cells: dict[tuple[int, int], int],
) -> None:
    switch_positions = set(switch_cells)
    intersection_cells = set(intersections)
    for _ in range(MARKER_RETRIES):
        logical_tracks: dict[int, dict[str, object]] = {}
        for track in track_specs:
            candidates = [pos for pos in track["cycle"] if pos not in switch_positions and pos not in intersection_cells]
            assert candidates, "track needs at least one marker location"
            track["marker"] = random.choice(candidates)
            logical_tracks[int(track["color"])] = track
            track["intersections"] = [pos for pos, owners in intersections.items() if int(track["color"]) in owners]
        finals = _final_positions(logical_tracks, intersections, switch_colors)
        if len(set(finals.values())) != len(finals):
            continue
        kinds = _branch_kinds(logical_tracks, switch_colors)
        if {"red", "maroon", "none"} <= set(kinds.values()):
            return
    raise AssertionError("failed to pick non-degenerate markers")


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        try:
            h, w, track_specs, intersections, switch_colors, switch_cells = _logical_tracks_and_switches()
            _pick_markers(track_specs, intersections, switch_colors, switch_cells)
            grid = _render_input(h, w, track_specs, intersections, switch_cells)
            tracks, _by_coord, reconstructed_intersections, reconstructed_switches = _reconstruct_tracks(grid)
            assert len(tracks) == len(track_specs), "track count mismatch"
            assert set(reconstructed_intersections) == set(intersections), "intersection reconstruction mismatch"
            assert reconstructed_switches == switch_colors, "switch reconstruction mismatch"
            kinds = _branch_kinds(tracks, reconstructed_switches)
            assert {"red", "maroon", "none"} <= set(kinds.values()), "branch coverage missing"
            finals = _final_positions(tracks, reconstructed_intersections, reconstructed_switches)
            assert len(set(finals.values())) == len(finals), "final marker positions must stay distinct"
            return grid
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid input")
