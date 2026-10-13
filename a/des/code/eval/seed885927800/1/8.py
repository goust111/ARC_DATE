from __future__ import annotations

import random

import numpy as np


def _border_background(g: np.ndarray) -> int:
    border = np.concatenate((g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]))
    vals, counts = np.unique(border, return_counts=True)
    return int(vals[counts.argmax()])


def _find_divider_row(g: np.ndarray, bg: int) -> int:
    candidates = [
        r
        for r in range(g.shape[0])
        if int(g[r, 0]) != bg and np.all(g[r, :] == g[r, 0])
    ]
    assert len(candidates) == 1, "expected exactly one full-width non-background divider row"
    divider_r = int(candidates[0])
    assert 0 < divider_r < g.shape[0] - 1, "divider must split the grid into upper and lower parts"
    return divider_r


def _row_runs(row: np.ndarray, color: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for c, value in enumerate(row.tolist()):
        if int(value) == color:
            if start is None:
                start = c
        elif start is not None:
            runs.append((int(start), int(c - 1)))
            start = None
    if start is not None:
        runs.append((int(start), int(len(row) - 1)))
    return runs


def _components(g: np.ndarray, bg: int) -> list[dict]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if seen[r, c] or color == bg:
                continue

            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [rr for rr, _cc in coords]
            cols = [cc for _rr, cc in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "area": len(coords),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )

    return comps


def _is_solid_rectangle(component: dict) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    return int(component["area"]) == (int(r1) - int(r0) + 1) * (int(c1) - int(c0) + 1)


def _parse_case(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _border_background(g)
    divider_r = _find_divider_row(g, bg)

    shadow_r = divider_r + 1
    assert shadow_r < g.shape[0], "shadow row must exist directly below the divider"
    shadow_row = g[shadow_r, :]
    shadow_colors = {int(v) for v in shadow_row.tolist() if int(v) != bg}
    assert len(shadow_colors) == 1, "expected exactly one shadow color on the projection row"
    shadow_color = int(next(iter(shadow_colors)))
    assert np.all((shadow_row == bg) | (shadow_row == shadow_color)), "projection row must contain only bg and shadow pixels"
    shadow_spans = _row_runs(shadow_row, shadow_color)
    assert shadow_spans, "expected at least one shadow segment"

    scene = g[:divider_r, :]
    components = _components(scene, bg)
    assert components, "expected scene objects above the divider"

    sources: dict[int, tuple[int, int]] = {}
    obstacles: list[dict] = []

    for component in components:
        color = int(component["color"])
        r0, c0, r1, c1 = component["bbox"]
        height = int(r1) - int(r0) + 1
        width = int(c1) - int(c0) + 1

        if int(component["area"]) == 1:
            assert r0 == 0 and r1 == 0, "sources must be single pixels on the top row"
            assert color not in sources, "source colors must be unique"
            sources[color] = (int(r0), int(c0))
            continue

        assert _is_solid_rectangle(component), "obstacles must be solid rectangles"
        assert 1 <= height <= 2 and 2 <= width <= 4, "unexpected obstacle dimensions"
        assert r0 >= 1, "obstacles must be below the source row"
        obstacles.append(component)

    obstacle_colors = [int(obstacle["color"]) for obstacle in obstacles]
    assert sources, "expected at least one source"
    assert len(obstacles) == len(set(obstacle_colors)), "obstacle colors must be unique"
    assert set(sources) == set(obstacle_colors), "each obstacle must match exactly one source color"
    assert len(obstacles) == len(shadow_spans), "each obstacle must have exactly one shadow segment"

    obstacle_by_span: dict[tuple[int, int], dict] = {}
    for obstacle in obstacles:
        _r0, c0, _r1, c1 = obstacle["bbox"]
        key = (int(c0), int(c1))
        assert key not in obstacle_by_span, "shadow matching requires unique obstacle spans"
        obstacle_by_span[key] = obstacle

    matched: list[dict] = []
    for span in shadow_spans:
        assert span in obstacle_by_span, "every shadow span must correspond to an obstacle span"
        matched.append(obstacle_by_span[span])

    return {
        "grid": g,
        "bg": bg,
        "divider_r": divider_r,
        "shadow_color": shadow_color,
        "sources": sources,
        "matched_obstacles": matched,
    }


def _apply_rule(grid: np.ndarray) -> np.ndarray:
    parsed = _parse_case(grid)
    g = parsed["grid"]
    bg = int(parsed["bg"])
    out = g.copy()

    for obstacle in parsed["matched_obstacles"]:
        color = int(obstacle["color"])
        src_r, _src_c = parsed["sources"][color]
        r0, c0, _r1, c1 = obstacle["bbox"]
        beam_top = int(src_r) + 1
        beam_bottom = int(r0)
        if beam_top >= beam_bottom:
            continue
        beam = out[beam_top:beam_bottom, int(c0) : int(c1) + 1]
        beam[beam == bg] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Verify source/occluder pairings through the existing shadows, then draw light paths in the scene.

    Input structure
    - bg = black.
    - A full-width horizontal divider row separates the upper scene from the lower projection area.
    - The top row contains distinct single-pixel colored sources.
    - Each source color also appears exactly once as a small solid rectangle obstacle in the scene.
    - The row immediately below the divider contains shadow segments whose spans match the obstacles' column ranges.

    Rule
    - Match each shadow segment to the unique obstacle with the same column span.
    - Use that obstacle's color to identify the paired source.
    - In the obstacle's columns, fill the background cells from just below the source row down to just above the obstacle.
    - Keep the sources, obstacles, divider, and shadows unchanged.
    """
    return _apply_rule(np.asarray(grid, dtype=int))


# --------------- generator ---------------

GEN_BG = 0
GEN_DIVIDER = 5
GEN_SHADOW = 9
GEN_SOURCE_COLORS = [1, 2, 3, 4, 6, 7, 8]

GEN_SCENE_HEIGHT_RANGE = (6, 11)
GEN_BOTTOM_EXTRA_RANGE = (0, 3)
GEN_WIDTH_RANGE = (12, 24)
GEN_MIN_OBJECTS = 2
GEN_MAX_OBJECTS = 5

GEN_EDGE_MARGIN = 1
GEN_MIN_GAP = 1
GEN_RETRY_LIMIT = 400


def _random_composition(total: int, parts: int) -> list[int]:
    if total <= 0:
        return [0] * parts
    return list(np.random.multinomial(int(total), [1.0 / parts] * parts))


def _pack_intervals(widths: list[int], total_width: int) -> list[tuple[int, int]] | None:
    count = len(widths)
    min_total = sum(widths) + 2 * GEN_EDGE_MARGIN + (count - 1) * GEN_MIN_GAP
    if min_total > total_width:
        return None

    slack = int(total_width - min_total)
    extras = _random_composition(slack, count + 1)
    intervals: list[tuple[int, int]] = []

    pos = GEN_EDGE_MARGIN + int(extras[0])
    for i, width in enumerate(widths):
        start = int(pos)
        end = int(start + width - 1)
        intervals.append((start, end))
        pos = end + 1
        if i + 1 < count:
            pos += GEN_MIN_GAP + int(extras[i + 1])
        else:
            pos += GEN_EDGE_MARGIN + int(extras[i + 1])

    assert pos == total_width
    return intervals


def _count_options(total_width: int) -> list[int]:
    return [
        count
        for count in range(GEN_MIN_OBJECTS, GEN_MAX_OBJECTS + 1)
        if 2 * GEN_EDGE_MARGIN + count * 2 + (count - 1) * GEN_MIN_GAP <= total_width
    ]


def _sample_layout(total_width: int, count: int) -> tuple[list[int], list[tuple[int, int]]] | None:
    for _ in range(80):
        widths = [random.randint(2, 4) for _ in range(count)]
        intervals = _pack_intervals(widths, total_width)
        if intervals is not None:
            return widths, intervals
    return None


def _generate_case() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        scene_h = random.randint(*GEN_SCENE_HEIGHT_RANGE)
        bottom_extra = random.randint(*GEN_BOTTOM_EXTRA_RANGE)
        width = random.randint(*GEN_WIDTH_RANGE)
        count_options = _count_options(width)
        if not count_options:
            continue

        count = random.choice(count_options)
        packed = _sample_layout(width, count)
        if packed is None:
            continue
        widths, intervals = packed

        height = int(scene_h + 2 + bottom_extra)
        colors = random.sample(GEN_SOURCE_COLORS, count)
        source_cols = random.sample(range(width), count)
        obstacle_heights = [random.randint(1, 2) for _ in range(count)]

        obstacle_tops = []
        valid = True
        for obstacle_h in obstacle_heights:
            max_top = int(scene_h - obstacle_h - 1)
            if max_top < 2:
                valid = False
                break
            obstacle_tops.append(random.randint(2, max_top))
        if not valid:
            continue

        grid = np.full((height, width), GEN_BG, dtype=int)
        grid[scene_h, :] = GEN_DIVIDER

        for color, src_c in zip(colors, source_cols, strict=False):
            grid[0, int(src_c)] = int(color)

        for color, obstacle_top, obstacle_h, obstacle_w, span in zip(
            colors,
            obstacle_tops,
            obstacle_heights,
            widths,
            intervals,
            strict=False,
        ):
            c0, c1 = span
            assert obstacle_w == c1 - c0 + 1
            grid[int(obstacle_top) : int(obstacle_top + obstacle_h), int(c0) : int(c1) + 1] = int(color)
            grid[int(scene_h + 1), int(c0) : int(c1) + 1] = GEN_SHADOW

        try:
            parsed = _parse_case(grid)
            assert len(parsed["matched_obstacles"]) == count
            assert np.any(_apply_rule(grid) != grid), "expected at least one light beam to be drawn"
        except AssertionError:
            continue

        return grid

    raise RuntimeError("failed to generate a valid puzzle instance")


def generate_input() -> np.ndarray:
    return _generate_case()
