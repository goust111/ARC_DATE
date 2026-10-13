from __future__ import annotations

import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]
SIDES = ("top", "bottom", "left", "right")


def _background(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    winners = colors[counts == counts.max()]
    assert len(winners) == 1, "expected a unique most-common background color"
    return int(winners[0])


def _components(mask: np.ndarray) -> list[list[Coord]]:
    """Return 4-connected True components using the shared (row, column) convention."""
    h, w = mask.shape
    unseen = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    result: list[list[Coord]] = []
    while unseen:
        start = unseen.pop()
        stack = [start]
        component = [start]
        while stack:
            r, c = stack.pop()
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) in unseen:
                    unseen.remove((nr, nc))
                    stack.append((nr, nc))
                    component.append((nr, nc))
        result.append(component)
    return result


def _bbox(coords: list[Coord]) -> BBox:
    arr = np.asarray(coords, dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _decode_filled_rings(
    grid: np.ndarray, component: list[Coord], bg: int
) -> tuple[BBox, list[int]] | None:
    """Decode a completely filled rectangle whose successive perimeters are monochrome."""
    r0, c0, r1, c1 = _bbox(component)
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return None
    region = grid[r0 : r1 + 1, c0 : c1 + 1]
    if len(component) != region.size or np.any(region == bg):
        return None

    colors: list[int] = []
    top, left, bottom, right = 0, 0, region.shape[0] - 1, region.shape[1] - 1
    while top <= bottom and left <= right:
        perimeter: list[int] = []
        perimeter.extend(map(int, region[top, left : right + 1]))
        if bottom > top:
            perimeter.extend(map(int, region[bottom, left : right + 1]))
        if bottom - top >= 2:
            perimeter.extend(map(int, region[top + 1 : bottom, left]))
            if right > left:
                perimeter.extend(map(int, region[top + 1 : bottom, right]))
        if len(set(perimeter)) != 1:
            return None
        colors.append(perimeter[0])
        top += 1
        left += 1
        bottom -= 1
        right -= 1

    if len(colors) < 2 or len(set(colors)) != len(colors) or bg in colors:
        return None
    return (r0, c0, r1, c1), colors


def _find_legend(
    grid: np.ndarray, bg: int
) -> tuple[list[Coord], BBox, list[int], list[list[Coord]]]:
    components = _components(grid != bg)
    candidates = []
    for component in components:
        decoded = _decode_filled_rings(grid, component, bg)
        if decoded is not None:
            candidates.append((component, *decoded))
    assert len(candidates) == 1, "expected one uniquely decodable nested rectangle"
    legend_component, box, colors = candidates[0]
    return legend_component, box, colors, components


def _destination(side: str, depth: int, align: int, box: BBox) -> Coord:
    r0, c0, r1, c1 = box
    if side == "top":
        return r0 - depth, align
    if side == "bottom":
        return r1 + depth, align
    if side == "left":
        return align, c0 - depth
    if side == "right":
        return align, c1 + depth
    raise ValueError(f"unknown side: {side}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each free marker toward a nested rectangle by the distance encoded by its ring color.

    Input structure:
    - A unique solid rectangular legend consists of distinct monochrome rings on a dominant background.
    - Every other foreground object is an isolated marker aligned with the legend's row or column span.

    Rule:
    - Number legend colors from the outer ring at depth 1 inward. Clear all free markers.
    - A marker with the depth-k ring color moves straight toward the legend and stops k cells beyond its outer edge,
      leaving exactly k-1 background cells; a marker whose color is absent from the legend disappears.
    - The nested rectangular legend itself is unchanged.

    Key insight:
    - The concentric rings are an ordinal ruler: visible nesting order, rather than color identity, determines each stop distance.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = _background(g)
    legend, box, ring_colors, components = _find_legend(g, bg)
    legend_set = set(legend)
    depth_of = {color: depth for depth, color in enumerate(ring_colors, start=1)}
    r0, c0, r1, c1 = box

    markers: list[Coord] = []
    for component in components:
        if set(component) == legend_set:
            continue
        assert len(component) == 1, "every object outside the legend must be a single pixel"
        markers.append(component[0])

    out = g.copy()
    for r, c in markers:
        out[r, c] = bg

    placements: dict[Coord, int] = {}
    for r, c in markers:
        color = int(g[r, c])
        horizontal = r0 <= r <= r1 and (c < c0 or c > c1)
        vertical = c0 <= c <= c1 and (r < r0 or r > r1)
        assert horizontal != vertical, "marker must have exactly one straight approach to the legend"
        if color not in depth_of:
            continue

        depth = depth_of[color]
        if horizontal and c < c0:
            side, align = "left", r
            assert c < c0 - depth, "left marker must begin beyond its prescribed stop"
        elif horizontal:
            side, align = "right", r
            assert c > c1 + depth, "right marker must begin beyond its prescribed stop"
        elif r < r0:
            side, align = "top", c
            assert r < r0 - depth, "top marker must begin beyond its prescribed stop"
        else:
            side, align = "bottom", c
            assert r > r1 + depth, "bottom marker must begin beyond its prescribed stop"

        target = _destination(side, depth, align, box)
        tr, tc = target
        assert 0 <= tr < g.shape[0] and 0 <= tc < g.shape[1]
        assert target not in placements, "two markers must not collapse onto one destination"
        placements[target] = color

    for (r, c), color in placements.items():
        assert out[r, c] == bg, "a destination must be an unambiguous background cell"
        out[r, c] = color
    return out


def _render_legend(grid: np.ndarray, box: BBox, colors: list[int]) -> None:
    r0, c0, r1, c1 = box
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            depth = min(r - r0, r1 - r, c - c0, c1 - c)
            grid[r, c] = colors[min(depth, len(colors) - 1)]


def _source_candidates(
    side: str, target: Coord, shape: tuple[int, int]
) -> list[Coord]:
    h, w = shape
    tr, tc = target
    if side == "top":
        return [(r, tc) for r in range(0, tr - 1)]
    if side == "bottom":
        return [(r, tc) for r in range(tr + 2, h)]
    if side == "left":
        return [(tr, c) for c in range(0, tc - 1)]
    return [(tr, c) for c in range(tc + 2, w)]


def _separated(cell: Coord, others: set[Coord]) -> bool:
    r, c = cell
    return all(max(abs(r - rr), abs(c - cc)) > 1 for rr, cc in others)


def _manual_expected(
    inp: np.ndarray,
    bg: int,
    matching: list[tuple[Coord, Coord, int, int, str]],
    distractors: list[Coord],
) -> np.ndarray:
    out = inp.copy()
    for source, _target, _color, _depth, _side in matching:
        out[source] = bg
    for source in distractors:
        out[source] = bg
    for _source, target, color, _depth, _side in matching:
        out[target] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    h, w = random.randint(24, 30), random.randint(24, 30)
    ring_count = random.randint(3, 5)
    base = random.choice((2 * ring_count - 1, 2 * ring_count))
    if random.random() < 0.5:
        rect_h, rect_w = base, base + random.randint(0, 3)
    else:
        rect_h, rect_w = base + random.randint(0, 3), base

    margin = ring_count + 4
    if rect_h + 2 * margin > h or rect_w + 2 * margin > w:
        return None
    r0 = random.randint(margin, h - rect_h - margin)
    c0 = random.randint(margin, w - rect_w - margin)
    box = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)

    ring_colors = random.sample(range(1, 10), ring_count)
    grid = np.full((h, w), bg, dtype=int)
    _render_legend(grid, box, ring_colors)

    marker_count = random.randint(max(7, ring_count + 2), 10)
    depths = list(range(1, ring_count + 1))
    depths.extend(random.choices(range(1, ring_count + 1), k=marker_count - len(depths)))
    sides = list(SIDES)
    sides.extend(random.choices(SIDES, k=marker_count - len(sides)))
    random.shuffle(depths)
    random.shuffle(sides)

    sources: set[Coord] = set()
    targets: set[Coord] = set()
    matching: list[tuple[Coord, Coord, int, int, str]] = []
    for depth, side in zip(depths, sides):
        aligns = list(range(c0, box[3] + 1)) if side in {"top", "bottom"} else list(range(r0, box[2] + 1))
        random.shuffle(aligns)
        placed = False
        for align in aligns:
            target = _destination(side, depth, align, box)
            if target in targets or target in sources:
                continue
            candidates = _source_candidates(side, target, grid.shape)
            random.shuffle(candidates)
            for source in candidates:
                if source in targets or grid[source] != bg or not _separated(source, sources):
                    continue
                color = ring_colors[depth - 1]
                grid[source] = color
                sources.add(source)
                targets.add(target)
                matching.append((source, target, color, depth, side))
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    unused_colors = [color for color in range(1, 10) if color not in ring_colors]
    distractors: list[Coord] = []
    for _ in range(random.randint(1, 3)):
        placed = False
        side_order = list(SIDES)
        random.shuffle(side_order)
        for side in side_order:
            aligns = list(range(c0, box[3] + 1)) if side in {"top", "bottom"} else list(range(r0, box[2] + 1))
            random.shuffle(aligns)
            for align in aligns:
                # A fictitious depth-one target supplies an aligned outside ray.
                near = _destination(side, 1, align, box)
                candidates = _source_candidates(side, near, grid.shape)
                random.shuffle(candidates)
                for source in candidates:
                    if source in targets or grid[source] != bg or not _separated(source, sources):
                        continue
                    grid[source] = random.choice(unused_colors)
                    sources.add(source)
                    distractors.append(source)
                    placed = True
                    break
                if placed:
                    break
            if placed:
                break
        if not placed:
            return None

    expected = _manual_expected(grid, bg, matching, distractors)
    try:
        actual = main(grid)
    except (AssertionError, ValueError):
        return None

    # Semantic checks: every depth and direction matter, unknown colors vanish,
    # every marker really moves, and the independently rendered result agrees.
    if {depth for _s, _t, _c, depth, _side in matching} != set(range(1, ring_count + 1)):
        return None
    if {side for _s, _t, _c, _depth, side in matching} != set(SIDES):
        return None
    if any(source == target for source, target, _c, _d, _side in matching):
        return None
    if any(expected[source] != bg for source in distractors):
        return None
    if np.array_equal(grid, expected) or not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-ring ruler input")
