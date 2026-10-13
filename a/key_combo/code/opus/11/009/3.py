from __future__ import annotations

import itertools
import random

import numpy as np


BLACK = 0
RED = 2
ORANGE = 7

NW_SE = "nw_se"
NE_SW = "ne_sw"
ORIENTATIONS = (NW_SE, NE_SW)

GEN_ATTEMPTS = 600
GRID_SIDE_RANGE = (14, 18)
SEGMENTS_PER_CLASS = (2, 3)
LENGTH_RANGE = (3, 6)
SHIFT_VALUES = (1, 2, 3, 4)


def neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def components8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in neighbors8(rr, cc):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def segment_coords(orientation: str, start_r: int, start_c: int, length: int) -> list[tuple[int, int]]:
    if orientation == NW_SE:
        return [(start_r + i, start_c + i) for i in range(length)]
    assert orientation == NE_SW
    return [(start_r + i, start_c - i) for i in range(length)]


def chebyshev_touches(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1


def sets_are_separated(coords_a: list[tuple[int, int]], coords_b: list[tuple[int, int]]) -> bool:
    return all(not chebyshev_touches(a, b) for a in coords_a for b in coords_b)


def classify_segment(coords: list[tuple[int, int]]) -> str:
    pts = sorted((int(r), int(c)) for r, c in coords)
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    if len(set(r - c for r, c in pts)) == 1:
        assert rows == list(range(rows[0], rows[0] + len(pts)))
        assert cols == list(range(cols[0], cols[0] + len(pts)))
        return NW_SE
    if len(set(r + c for r, c in pts)) == 1:
        assert rows == list(range(rows[0], rows[0] + len(pts)))
        assert cols == list(range(cols[0], cols[0] - len(pts), -1))
        return NE_SW
    raise AssertionError("orange component is not a straight diagonal segment")


def parse_scene(grid: np.ndarray) -> tuple[list[dict], dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    orange_components = components8(g == ORANGE)
    assert orange_components, "expected orange diagonal segments"

    segments: list[dict] = []
    for comp in orange_components:
        orientation = classify_segment(comp)
        segments.append(
            {
                "coords": sorted(comp),
                "coord_set": set(comp),
                "orientation": orientation,
                "length": len(comp),
            }
        )

    red_coords = [tuple(map(int, rc)) for rc in np.argwhere(g == RED)]
    marker_counts = [0] * len(segments)
    for rr, cc in red_coords:
        owners = []
        for idx, seg in enumerate(segments):
            if any(abs(rr - sr) <= 1 and abs(cc - sc) <= 1 for sr, sc in seg["coords"]):
                owners.append(idx)
        assert len(owners) == 1, "every red marker must belong to exactly one diagonal segment"
        marker_counts[owners[0]] += 1

    class_totals = {NW_SE: 0, NE_SW: 0}
    for seg, count in zip(segments, marker_counts):
        seg["marker_count"] = int(count)
        class_totals[seg["orientation"]] += int(count)

    for orientation in ORIENTATIONS:
        assert any(seg["orientation"] == orientation for seg in segments), "both orientations are required"
    return segments, class_totals


def shifted_coords(seg: dict, distance: int) -> list[tuple[int, int]]:
    if seg["orientation"] == NW_SE:
        return [(r, c + distance) for r, c in seg["coords"]]
    return [(r + distance, c) for r, c in seg["coords"]]


def render_shifted(grid: np.ndarray, segments: list[dict], distances: dict[str, int]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    out[(out == RED) | (out == ORANGE)] = BLACK
    occupied: set[tuple[int, int]] = set()
    for seg in segments:
        moved = shifted_coords(seg, distances[seg["orientation"]])
        for r, c in moved:
            assert in_bounds(out.shape, r, c), "shifted segment must stay in bounds"
            assert (r, c) not in occupied, "shifted segments must not overlap"
            occupied.add((r, c))
            out[r, c] = ORANGE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each diagonal orientation group by a marker-count distance chosen from its attached red pixels.

    Input structure
    - The grid has a black background, orange one-cell-wide diagonal segments, and red single-pixel markers.
    - Orange segments appear in both diagonal orientations: NW-SE (`\\`) and NE-SW (`/`).
    - Every red marker is 8-neighbor adjacent to exactly one orange segment.

    Rule
    - Identify each separate orange diagonal segment and classify its orientation.
    - Count the red markers attached to each segment, then sum those counts within each orientation class.
    - Shift all NW-SE segments right by their class total and all NE-SW segments down by their class total.
    - Erase every red marker and clear the original orange positions.

    Key insight
    - The orientation class chooses the movement direction, while the class-wide total of attached markers chooses the movement distance.
    """
    g = np.asarray(grid, dtype=int)
    segments, class_totals = parse_scene(g)
    return render_shifted(g, segments, class_totals)


def compositions(total: int, parts: int) -> list[tuple[int, ...]]:
    if parts == 1:
        return [(total,)]
    out = []
    for first in range(total + 1):
        for rest in compositions(total - first, parts - 1):
            out.append((first,) + rest)
    return out


def candidate_starts(height: int, width: int, orientation: str, length: int, distance: int) -> list[tuple[int, int]]:
    starts = []
    if orientation == NW_SE:
        for r in range(1, height - length):
            for c in range(1, width - length - distance):
                starts.append((r, c))
    else:
        for r in range(1, height - length - distance):
            for c in range(length, width - 1):
                starts.append((r, c))
    random.shuffle(starts)
    return starts


def place_segments(height: int, width: int, specs: list[tuple[str, int]], distances: dict[str, int]) -> list[dict] | None:
    ordered = sorted(
        enumerate(specs),
        key=lambda item: (-item[1][1], 0 if item[1][0] == NW_SE else 1, random.random()),
    )
    placed: list[dict | None] = [None] * len(specs)

    def rec(pos: int, input_segments: list[list[tuple[int, int]]], output_segments: list[list[tuple[int, int]]]) -> bool:
        if pos == len(ordered):
            return True
        spec_idx, (orientation, length) = ordered[pos]
        distance = distances[orientation]
        for start_r, start_c in candidate_starts(height, width, orientation, length, distance):
            coords = segment_coords(orientation, start_r, start_c, length)
            moved = [(r, c + distance) for r, c in coords] if orientation == NW_SE else [(r + distance, c) for r, c in coords]
            if any(not sets_are_separated(coords, other) for other in input_segments):
                continue
            if any(not sets_are_separated(moved, other) for other in output_segments):
                continue
            placed[spec_idx] = {"orientation": orientation, "coords": coords, "length": length}
            input_segments.append(coords)
            output_segments.append(moved)
            if rec(pos + 1, input_segments, output_segments):
                return True
            input_segments.pop()
            output_segments.pop()
            placed[spec_idx] = None
        return False

    if not rec(0, [], []):
        return None
    return [item for item in placed if item is not None]


def marker_candidates(height: int, width: int, segments: list[dict]) -> dict[int, list[tuple[int, int]]]:
    out = {idx: set() for idx in range(len(segments))}
    orange_cells = set()
    for seg in segments:
        orange_cells.update(seg["coords"])
    for r in range(height):
        for c in range(width):
            if (r, c) in orange_cells:
                continue
            owners = []
            for idx, seg in enumerate(segments):
                if any(abs(r - sr) <= 1 and abs(c - sc) <= 1 for sr, sc in seg["coords"]):
                    owners.append(idx)
            if len(owners) == 1:
                out[owners[0]].add((r, c))
    return {idx: sorted(cells) for idx, cells in out.items()}


def valid_marker_combos(cells: list[tuple[int, int]], count: int) -> list[tuple[tuple[int, int], ...]]:
    if count == 0:
        return [tuple()]
    combos = []
    for combo in itertools.combinations(cells, count):
        if all(not chebyshev_touches(a, b) for a, b in itertools.combinations(combo, 2)):
            combos.append(combo)
    random.shuffle(combos)
    return combos


def assign_markers(height: int, width: int, segments: list[dict], counts: list[int]) -> list[tuple[int, int]] | None:
    cand_map = marker_candidates(height, width, segments)
    per_segment_choices = []
    for idx, need in enumerate(counts):
        combos = valid_marker_combos(cand_map[idx], need)
        if not combos:
            return None
        per_segment_choices.append((idx, combos))
    per_segment_choices.sort(key=lambda item: len(item[1]))

    chosen: list[tuple[int, int]] = []

    def rec(pos: int) -> bool:
        if pos == len(per_segment_choices):
            return True
        idx, combos = per_segment_choices[pos]
        for combo in combos:
            if any(chebyshev_touches(a, b) for a in combo for b in chosen):
                continue
            chosen.extend(combo)
            if rec(pos + 1):
                return True
            del chosen[-len(combo) :]
        return False

    if not rec(0):
        return None
    assert all(in_bounds((height, width), r, c) for r, c in chosen)
    return chosen


def sampled_count_splits(totals: dict[str, int], segments: list[dict]) -> list[int] | None:
    counts_by_orientation: dict[str, list[int]] = {}
    index_by_orientation: dict[str, list[int]] = {NW_SE: [], NE_SW: []}
    for idx, seg in enumerate(segments):
        index_by_orientation[seg["orientation"]].append(idx)

    non_uniform_classes = 0
    result = [0] * len(segments)
    for orientation in ORIENTATIONS:
        count = len(index_by_orientation[orientation])
        choices = compositions(totals[orientation], count)
        choices = [choice for choice in choices if sum(choice) == totals[orientation]]
        random.shuffle(choices)
        if not choices:
            return None
        selected = choices[0]
        counts_by_orientation[orientation] = list(selected)
        if len(set(selected)) > 1:
            non_uniform_classes += 1
        for idx, value in zip(index_by_orientation[orientation], selected):
            result[idx] = value

    if non_uniform_classes == 0:
        return None
    return result


def wrong_per_segment_output(grid: np.ndarray, segments: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    out[(out == RED) | (out == ORANGE)] = BLACK
    for seg in segments:
        moved = shifted_coords(seg, seg["marker_count"])
        for r, c in moved:
            if in_bounds(out.shape, r, c):
                out[r, c] = ORANGE
    return out


def semantic_validation(grid: np.ndarray) -> None:
    segments, class_totals = parse_scene(grid)
    assert len(segments) >= 4
    assert {seg["orientation"] for seg in segments} == {NW_SE, NE_SW}
    assert all(2 <= sum(seg["orientation"] == ori for seg in segments) <= 3 for ori in ORIENTATIONS)
    assert all(LENGTH_RANGE[0] <= seg["length"] <= LENGTH_RANGE[1] for seg in segments)
    assert class_totals[NW_SE] in SHIFT_VALUES and class_totals[NE_SW] in SHIFT_VALUES
    assert class_totals[NW_SE] != class_totals[NE_SW]

    red_coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == RED)]
    assert all(not chebyshev_touches(a, b) for a, b in itertools.combinations(red_coords, 2))

    for i, seg_a in enumerate(segments):
        for seg_b in segments[i + 1 :]:
            assert sets_are_separated(seg_a["coords"], seg_b["coords"])

    truth = render_shifted(grid, segments, class_totals)
    assert np.array_equal(main(grid), truth)
    assert not np.array_equal(grid, truth)

    swapped_dirs = render_shifted(grid, [{**seg, "orientation": NE_SW if seg["orientation"] == NW_SE else NW_SE} for seg in segments], class_totals)
    assert not np.array_equal(swapped_dirs, truth)

    swapped_dists = render_shifted(grid, segments, {NW_SE: class_totals[NE_SW], NE_SW: class_totals[NW_SE]})
    assert not np.array_equal(swapped_dists, truth)

    per_segment = wrong_per_segment_output(grid, segments)
    assert not np.array_equal(per_segment, truth)


def build_example() -> np.ndarray | None:
    height = random.randint(*GRID_SIDE_RANGE)
    width = random.randint(*GRID_SIDE_RANGE)
    d_nw, d_ne = random.sample(SHIFT_VALUES, 2)
    distances = {NW_SE: d_nw, NE_SW: d_ne}

    nw_count = random.choice(SEGMENTS_PER_CLASS)
    ne_count = random.choice(SEGMENTS_PER_CLASS)
    specs = [(NW_SE, random.randint(*LENGTH_RANGE)) for _ in range(nw_count)]
    specs += [(NE_SW, random.randint(*LENGTH_RANGE)) for _ in range(ne_count)]

    segments = place_segments(height, width, specs, distances)
    if segments is None:
        return None

    counts = sampled_count_splits(distances, segments)
    if counts is None:
        return None

    markers = assign_markers(height, width, segments, counts)
    if markers is None:
        return None

    grid = np.zeros((height, width), dtype=int)
    for seg, marker_count in zip(segments, counts):
        seg["marker_count"] = marker_count
        for r, c in seg["coords"]:
            grid[r, c] = ORANGE
    for r, c in markers:
        grid[r, c] = RED

    try:
        semantic_validation(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = build_example()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid diagonal-slide ARC input")
