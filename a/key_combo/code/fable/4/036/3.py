"""ARC task: downward rays enter, fill, and are rerouted by hollow containers."""

from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
WALL = 5
SOURCE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
def _find_containers(grid: np.ndarray) -> list[dict]:
    """Reconstruct containers from their rectangular geometry and openings."""
    height, width = grid.shape
    containers: list[dict] = []
    gray_sets: list[set[tuple[int, int]]] = []

    # A two-opening border is not necessarily one 4-connected component.  Full
    # vertical sides and the constrained horizontal edges uniquely recover it.
    for r0 in range(height - 2):
        for r1 in range(r0 + 2, height):
            side_columns = [
                c
                for c in range(width)
                if int(grid[r0, c]) == WALL
                and int(grid[r1, c]) == WALL
                and np.all(grid[r0 : r1 + 1, c] == WALL)
            ]
            for i, c0 in enumerate(side_columns):
                for c1 in side_columns[i + 1 :]:
                    if c1 - c0 + 1 < 4:
                        continue
                    top = grid[r0, c0 + 1 : c1]
                    bottom = grid[r1, c0 + 1 : c1]
                    if np.any((top != WALL) & (top != BACKGROUND)):
                        continue
                    if np.any((bottom != WALL) & (bottom != BACKGROUND)):
                        continue
                    top_gaps = np.where(top == BACKGROUND)[0]
                    bottom_gaps = np.where(bottom == BACKGROUND)[0]
                    if len(top_gaps) != 1 or len(bottom_gaps) > 1:
                        continue
                    if not np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == BACKGROUND):
                        continue

                    top_open = c0 + 1 + int(top_gaps[0])
                    bottom_open = None
                    if len(bottom_gaps) == 1:
                        bottom_open = c0 + 1 + int(bottom_gaps[0])
                    gray_cells = {(r, c0) for r in range(r0, r1 + 1)}
                    gray_cells |= {(r, c1) for r in range(r0, r1 + 1)}
                    gray_cells |= {(r0, c) for c in range(c0, c1 + 1) if c != top_open}
                    gray_cells |= {(r1, c) for c in range(c0, c1 + 1) if c != bottom_open}
                    containers.append(
                        {
                            "bbox": (r0, c0, r1, c1),
                            "top_open": (r0, top_open),
                            "bottom_open": None if bottom_open is None else (r1, bottom_open),
                        }
                    )
                    gray_sets.append(gray_cells)

    assert containers, "expected at least one gray container"
    all_gray = {tuple(map(int, p)) for p in np.argwhere(grid == WALL)}
    covered: set[tuple[int, int]] = set()
    for cells in gray_sets:
        assert not (covered & cells), "container borders must not overlap"
        covered |= cells
    assert covered == all_gray, "every gray wall must belong to exactly one container"
    containers.sort(key=lambda item: item["bbox"])
    openings = [item["top_open"] for item in containers]
    assert len(openings) == len(set(openings)), "top openings must be distinct"
    return containers


def _paint_background(out: np.ndarray, original: np.ndarray, r: int, c: int, color: int) -> None:
    """Paint one originally black cell, rejecting ambiguous differently colored rays."""
    assert int(original[r, c]) == BACKGROUND, "rays may paint only background cells"
    old = int(out[r, c])
    assert old in (BACKGROUND, color), "different colored rays must not overlap"
    out[r, c] = color


def _solve(grid: np.ndarray, *, with_trace: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30, "expected an ARC-sized 2D grid"
    containers = _find_containers(g)
    first_container_row = min(item["bbox"][0] for item in containers)

    source_coords = np.argwhere((g != BACKGROUND) & (g != WALL))
    assert len(source_coords) >= 1, "expected at least one colored source pixel"
    sources = [(int(r), int(c), int(g[r, c])) for r, c in source_coords]
    assert all(r < first_container_row for r, _c, _color in sources), "sources must lie above all containers"
    assert len({c for _r, c, _color in sources}) == len(sources), "sources must occupy distinct columns"
    sources.sort()

    by_top_open = {item["top_open"]: item for item in containers}
    out = g.copy()
    traces: list[dict] = []
    height, _width = g.shape

    for source_r, source_c, color in sources:
        r, c = source_r + 1, source_c
        events: list[tuple] = []
        while r < height:
            container = by_top_open.get((r, c))
            if container is not None:
                r0, c0, r1, c1 = container["bbox"]
                _paint_background(out, g, r0, c, color)
                for rr in range(r0 + 1, r1):
                    for cc in range(c0 + 1, c1):
                        _paint_background(out, g, rr, cc, color)
                events.append(("enter", container["bbox"]))

                bottom_open = container["bottom_open"]
                if bottom_open is None:
                    events.append(("terminate", container["bbox"]))
                    break
                _paint_background(out, g, bottom_open[0], bottom_open[1], color)
                new_c = int(bottom_open[1])
                events.append(("shift", c, new_c))
                r, c = r1 + 1, new_c
                continue

            cell = int(g[r, c])
            if cell == WALL:
                events.append(("wall", (r, c)))
                break
            assert cell == BACKGROUND, "unexpected non-container foreground below a source"
            _paint_background(out, g, r, c, color)
            r += 1
        else:
            events.append(("bottom", c))
        traces.append({"source": (source_r, source_c), "color": color, "events": events})

    if with_trace:
        return out, traces, containers
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Extend each top source downward, with hollow containers filling and rerouting entered rays.

    Input structure:
    - A black grid has colored singleton sources above several separate gray hollow rectangular containers.
    - Every container has one black top-edge opening and either zero or one black bottom-edge opening.
    - Source columns are distinct; container interiors are empty, and differently colored ray paths do not overlap.

    Rule:
    - Extend each source color downward through black cells. A gray wall stops the ray without changing the container.
    - A ray aligned with a top opening enters and colors that opening plus the whole interior.
    - With a bottom opening, color it and continue below in its column; without one, terminate inside the container.
    - Rays not stopped by a wall or closed container continue to the bottom edge.

    Key insight:
    - A container is a gated rerouter: its top opening controls entry, while its optional bottom opening controls both continuation and the new column for the remaining conduction chain.
    """
    return _solve(grid)


def _draw_container(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    top_col: int,
    bottom_col: int | None,
) -> None:
    r0, c0, r1, c1 = bbox
    assert c0 < top_col < c1
    assert bottom_col is None or c0 < bottom_col < c1
    grid[r0, c0 : c1 + 1] = WALL
    grid[r1, c0 : c1 + 1] = WALL
    grid[r0 : r1 + 1, c0] = WALL
    grid[r0 : r1 + 1, c1] = WALL
    grid[r0, top_col] = BACKGROUND
    if bottom_col is not None:
        grid[r1, bottom_col] = BACKGROUND


def _random_bbox(zone: tuple[int, int], top: int, height: int) -> tuple[int, int, int, int]:
    lo, hi = zone
    width = random.randint(5, min(8, hi - lo + 1))
    left = random.randint(lo, hi - width + 1)
    return top, left, top + height - 1, left + width - 1


def _bbox_around_column(
    zone: tuple[int, int], top: int, height: int, opening_col: int
) -> tuple[int, int, int, int] | None:
    lo, hi = zone
    choices: list[tuple[int, int, int, int]] = []
    for width in range(5, min(8, hi - lo + 1) + 1):
        for left in range(lo, hi - width + 2):
            if left < opening_col < left + width - 1:
                choices.append((top, left, top + height - 1, left + width - 1))
    return random.choice(choices) if choices else None


def _opening_choices(bbox: tuple[int, int, int, int]) -> list[int]:
    return list(range(bbox[1] + 1, bbox[3]))


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(27, 30)
    width = random.randint(24, 30)
    grid = np.zeros((height, width), dtype=int)

    middle = width // 2
    left_zone = (3, middle - 2)
    right_zone = (middle + 2, width - 4)
    if min(left_zone[1] - left_zone[0], right_zone[1] - right_zone[0]) < 4:
        return None
    chain_zone, other_zone = random.choice(
        [(left_zone, right_zone), (right_zone, left_zone)]
    )

    first_top = random.randint(5, 7)
    first_height = random.randint(4, 5)
    second_top = random.randint(max(13, first_top + first_height + 2), 15)
    second_height = random.randint(4, 5)
    if second_top + second_height >= height - 5:
        return None

    first = _random_bbox(chain_zone, first_top, first_height)
    first_openings = _opening_choices(first)
    top_col = random.choice(first_openings)
    exit_col = random.choice([c for c in first_openings if c != top_col])
    second = _bbox_around_column(chain_zone, second_top, second_height, exit_col)
    if second is None:
        return None

    blocker = _random_bbox(other_zone, first_top, random.randint(4, 5))
    blocker_inner = _opening_choices(blocker)
    blocked_col = random.choice(blocker_inner)
    blocker_open = random.choice([c for c in blocker_inner if c != blocked_col])
    distractor = _random_bbox(other_zone, second_top, random.randint(4, 5))
    distractor_top = random.choice(_opening_choices(distractor))
    distractor_bottom = random.choice([None, random.choice(_opening_choices(distractor))])

    _draw_container(grid, first, top_col, exit_col)
    _draw_container(grid, second, exit_col, None)
    _draw_container(
        grid,
        blocker,
        blocker_open,
        random.choice([None, random.choice(blocker_inner)]),
    )
    _draw_container(grid, distractor, distractor_top, distractor_bottom)

    extra_count = random.randint(0, 2)
    third_top = height - 5
    extra_bboxes: list[tuple[int, int, int, int]] = []
    for zone in random.sample([left_zone, right_zone], extra_count):
        box = _random_bbox(zone, third_top, 4)
        openings = _opening_choices(box)
        _draw_container(
            grid,
            box,
            random.choice(openings),
            random.choice([None, random.choice(openings)]),
        )
        extra_bboxes.append(box)

    source_row = random.randint(0, min(2, first_top - 3))
    source_columns = [top_col, blocked_col]
    open_ray_columns = [1, width - 2]
    source_columns.extend(random.sample(open_ray_columns, random.randint(0, 2)))
    colors = random.sample(SOURCE_COLORS, len(source_columns))
    for col, color in zip(source_columns, colors):
        grid[source_row, col] = color

    try:
        out, traces, parsed = _solve(grid, with_trace=True)
    except AssertionError:
        return None
    if len(parsed) != 4 + extra_count:
        return None

    by_source_col = {trace["source"][1]: trace for trace in traces}
    chain_events = by_source_col[top_col]["events"]
    expected_chain = [
        ("enter", first),
        ("shift", top_col, exit_col),
        ("enter", second),
        ("terminate", second),
    ]
    if chain_events != expected_chain:
        return None
    if by_source_col[blocked_col]["events"] != [("wall", (first_top, blocked_col))]:
        return None
    if any(by_source_col[c]["events"] != [("bottom", c)] for c in source_columns[2:]):
        return None

    entered = {event[1] for trace in traces for event in trace["events"] if event[0] == "enter"}
    if entered != {first, second} or distractor in entered or any(box in entered for box in extra_bboxes):
        return None

    chain_color = by_source_col[top_col]["color"]
    if not np.all(out[first[0] + 1 : first[2], first[1] + 1 : first[3]] == chain_color):
        return None
    if not np.all(out[second[0] + 1 : second[2], second[1] + 1 : second[3]] == chain_color):
        return None
    if int(out[first[2] + 1, exit_col]) != chain_color:
        return None
    if int(out[first[2] + 1, top_col]) != BACKGROUND:
        return None
    if np.any(out[second[2] + 1 :, exit_col] == chain_color):
        return None
    if not np.array_equal(out[distractor[0] : distractor[2] + 1, distractor[1] : distractor[3] + 1],
                          grid[distractor[0] : distractor[2] + 1, distractor[1] : distractor[3] + 1]):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a varied, non-degenerate instance covering every rule branch."""
    for _ in range(300):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rerouting-container input")
