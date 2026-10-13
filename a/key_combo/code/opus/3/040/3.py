from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BACKGROUND = 0
OBSERVER = 2
PALETTE = (1, 3, 4, 5, 6, 7, 8, 9)

LANE_IDS = (
    "left_top",
    "left_bottom",
    "right_top",
    "right_bottom",
    "up_left",
    "up_right",
    "down_left",
    "down_right",
)

VISIBLE_COUNT_PATTERNS = (
    (1, 2),
    (1, 2),
    (1, 3),
    (1, 3),
    (2, 3),
    (1, 2, 3),
    (1, 4),
    (2, 4),
)

GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_color_components(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue

            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == int(color)
                    ):
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            arr = np.array(coords, dtype=int)
            comps.append(
                {
                    "color": int(color),
                    "coords": arr,
                    "bbox": (
                        int(arr[:, 0].min()),
                        int(arr[:, 1].min()),
                        int(arr[:, 0].max()),
                        int(arr[:, 1].max()),
                    ),
                    "area": int(arr.shape[0]),
                }
            )

    return comps


def _find_observer(grid: np.ndarray) -> tuple[list[tuple[int, int]], tuple[int, int, int, int]]:
    comps = _find_color_components(grid, OBSERVER)
    assert len(comps) == 1, "expected exactly one red observer block"
    comp = comps[0]
    r0, c0, r1, c1 = comp["bbox"]
    assert comp["area"] == 4, "observer must have four pixels"
    assert r1 - r0 == 1 and c1 - c0 == 1, "observer must be a 2x2 block"
    cells = [tuple(map(int, rc)) for rc in comp["coords"]]
    return cells, (r0, c0, r1, c1)


def _ray_is_clear(grid: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> bool:
    sr, sc = start
    er, ec = end

    if sr == er:
        c0, c1 = sorted((sc, ec))
        return bool(np.all(grid[sr, c0 + 1 : c1] == BACKGROUND))

    if sc == ec:
        r0, r1 = sorted((sr, er))
        return bool(np.all(grid[r0 + 1 : r1, sc] == BACKGROUND))

    return False


def _component_is_visible(
    grid: np.ndarray,
    observer_cells: list[tuple[int, int]],
    comp: dict,
) -> bool:
    targets = [tuple(map(int, rc)) for rc in comp["coords"]]
    return any(_ray_is_clear(grid, obs, tgt) for obs in observer_cells for tgt in targets)


def _visible_color_stats(grid: np.ndarray) -> tuple[list[tuple[int, int]], dict[int, int], dict[int, int], list[dict]]:
    observer_cells, _bbox = _find_observer(grid)
    all_components: list[dict] = []
    visible_components: list[dict] = []

    for color in map(int, np.unique(grid)):
        if color in (BACKGROUND, OBSERVER):
            continue
        comps = _find_color_components(grid, color)
        all_components.extend(comps)
        for comp in comps:
            if _component_is_visible(grid, observer_cells, comp):
                visible_components.append(comp)

    total_by_color: dict[int, int] = defaultdict(int)
    visible_by_color: dict[int, int] = defaultdict(int)
    visible_groups: dict[int, list[dict]] = defaultdict(list)

    for comp in all_components:
        total_by_color[int(comp["color"])] += 1
    for comp in visible_components:
        color = int(comp["color"])
        visible_by_color[color] += 1
        visible_groups[color].append(comp)

    rows: list[tuple[int, int, int, int]] = []
    for color, comps in visible_groups.items():
        first_r, first_c = min((comp["bbox"][0], comp["bbox"][1]) for comp in comps)
        rows.append((len(comps), first_r, first_c, int(color)))
    rows.sort(key=lambda item: (-item[0], item[1], item[2]))

    return (
        [(color, count) for count, _r, _c, color in rows],
        dict(total_by_color),
        dict(visible_by_color),
        all_components,
    )


def main(grid: np.ndarray) -> np.ndarray:
    """Count only the line-of-sight-visible objects per color and render them as a right-aligned histogram.

    Input structure
    - Black background with one red 2x2 observer block away from the border.
    - Several small monochromatic 4-connected foreground objects of non-black, non-red colors.
    - Some objects share a row or column with an observer pixel, while others are hidden behind earlier foreground objects on those sight lines.
    Rule
    - A foreground object is visible iff some observer pixel and some object pixel are aligned horizontally or vertically and every strictly intermediate cell is black.
    - Count only visible 4-connected objects, grouping counts by color and discarding colors whose objects are all hidden.
    - Sort visible colors by descending visible-object count and draw one histogram row per color, right-aligned to the maximum count.
    Key insight
    - The output is not a census of every object: line of sight first filters which connected components are eligible to be counted at all.
    """
    g = np.asarray(grid, dtype=int)
    visible_counts, _total_by_color, _visible_by_color, _all_components = _visible_color_stats(g)
    assert visible_counts, "expected at least one visible foreground color"

    height = len(visible_counts)
    width = max(count for _color, count in visible_counts)
    out = np.full((height, width), BACKGROUND, dtype=int)

    for r, (color, count) in enumerate(visible_counts):
        out[r, width - count : width] = int(color)

    return out


def _lane_cells(
    shape: tuple[int, int],
    observer_bbox: tuple[int, int, int, int],
    lane_id: str,
) -> list[tuple[int, int]]:
    h, w = shape
    r0, c0, r1, c1 = observer_bbox

    if lane_id == "left_top":
        return [(r0, c) for c in range(c0 - 2, -1, -1)]
    if lane_id == "left_bottom":
        return [(r1, c) for c in range(c0 - 2, -1, -1)]
    if lane_id == "right_top":
        return [(r0, c) for c in range(c1 + 2, w)]
    if lane_id == "right_bottom":
        return [(r1, c) for c in range(c1 + 2, w)]
    if lane_id == "up_left":
        return [(r, c0) for r in range(r0 - 2, -1, -1)]
    if lane_id == "up_right":
        return [(r, c1) for r in range(r0 - 2, -1, -1)]
    if lane_id == "down_left":
        return [(r, c0) for r in range(r1 + 2, h)]
    if lane_id == "down_right":
        return [(r, c1) for r in range(r1 + 2, h)]
    raise ValueError(f"unknown lane_id: {lane_id}")


def _can_place(occupied: np.ndarray, coords: list[tuple[int, int]]) -> bool:
    h, w = occupied.shape
    coord_set = set(coords)
    for r, c in coords:
        if occupied[r, c]:
            return False
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and occupied[nr, nc] and (nr, nc) not in coord_set:
                return False
    return True


def _sample_lane_segments(capacity: int, count: int) -> list[tuple[int, int]] | None:
    for _ in range(50):
        lengths = [random.randint(2, min(5, capacity)) for _ in range(count)]
        need = sum(lengths) + (count - 1)
        if need > capacity:
            continue

        slack = capacity - need
        if count == 1:
            before = random.randint(0, slack)
            return [(before, lengths[0])]

        before = random.randint(0, slack)
        gap_extra = random.randint(0, slack - before)
        return [
            (before, lengths[0]),
            (before + lengths[0] + 1 + gap_extra, lengths[1]),
        ]

    return None


def _render_segments(
    grid: np.ndarray,
    occupied: np.ndarray,
    lane_cells: list[tuple[int, int]],
    segment_specs: list[tuple[int, int]],
) -> list[list[tuple[int, int]]] | None:
    placed: list[list[tuple[int, int]]] = []
    for start, length in segment_specs:
        coords = lane_cells[start : start + length]
        if len(coords) != length or not _can_place(occupied, coords):
            return None
        placed.append(coords)

    for coords in placed:
        for r, c in coords:
            occupied[r, c] = True
    return placed


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(15, 18)
    w = random.randint(15, 18)
    observer_r0 = random.randint(6, h - 8)
    observer_c0 = random.randint(6, w - 8)
    observer_bbox = (observer_r0, observer_c0, observer_r0 + 1, observer_c0 + 1)

    visible_pattern = random.choice(VISIBLE_COUNT_PATTERNS)
    visible_colors = random.sample(PALETTE, len(visible_pattern))
    hidden_color = random.choice([color for color in PALETTE if color not in visible_colors])

    lane_ids = list(LANE_IDS)
    random.shuffle(lane_ids)
    used_lanes = lane_ids[: sum(visible_pattern)]

    front_color_by_lane: dict[str, int] = {}
    idx = 0
    for color, count in zip(visible_colors, visible_pattern):
        for _ in range(count):
            front_color_by_lane[used_lanes[idx]] = int(color)
            idx += 1

    hidden_count = random.randint(1, min(2, len(used_lanes)))
    hidden_lanes = set(random.sample(used_lanes, hidden_count))

    grid = np.full((h, w), BACKGROUND, dtype=int)
    grid[observer_r0 : observer_r0 + 2, observer_c0 : observer_c0 + 2] = OBSERVER
    occupied = np.zeros((h, w), dtype=bool)
    occupied[observer_r0 : observer_r0 + 2, observer_c0 : observer_c0 + 2] = True

    objects: list[tuple[int, list[tuple[int, int]]]] = []
    lane_order = used_lanes[:]
    random.shuffle(lane_order)

    for lane_id in lane_order:
        cells = _lane_cells((h, w), observer_bbox, lane_id)
        segment_count = 2 if lane_id in hidden_lanes else 1
        if len(cells) < 2:
            return None

        placed = None
        for _ in range(160):
            specs = _sample_lane_segments(len(cells), segment_count)
            if specs is None:
                break
            placed = _render_segments(grid, occupied, cells, specs)
            if placed is not None:
                break
        if placed is None:
            return None

        objects.append((front_color_by_lane[lane_id], placed[0]))
        if lane_id in hidden_lanes:
            objects.append((hidden_color, placed[1]))

    for color, coords in objects:
        for r, c in coords:
            grid[r, c] = int(color)

    visible_counts, total_by_color, visible_by_color, all_components = _visible_color_stats(grid)
    if sorted((count for _color, count in visible_counts), reverse=True) != sorted(
        visible_pattern, reverse=True
    ):
        return None
    if len(visible_counts) != len(visible_pattern):
        return None
    if len({count for _color, count in visible_counts}) != len(visible_counts):
        return None
    if hidden_color in visible_by_color:
        return None
    if total_by_color.get(hidden_color, 0) != hidden_count:
        return None
    if len(total_by_color) < 3:
        return None
    if any(not (2 <= int(comp["area"]) <= 5) for comp in all_components):
        return None
    if any(not (1 <= count <= 4) for count in total_by_color.values()):
        return None

    out = main(grid)
    if out.shape[0] != len(visible_counts) or out.shape[1] != max(count for _color, count in visible_counts):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid visibility-histogram scene")
