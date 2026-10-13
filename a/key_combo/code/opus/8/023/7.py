from __future__ import annotations

from collections import Counter
import random

import numpy as np


MARKER = 5
ODD_SIDES = (1, 3, 5, 7, 9)
SIDE_CYCLE = (9, 7, 5, 3, 1)
PALETTE = (1, 2, 3, 4, 6)

GRID_SIDE_RANGE = (25, 29)
OBJECT_COUNT_RANGE = (3, 5)
PLACEMENT_GAP = 1
MAX_GENERATION_ATTEMPTS = 500

COUNT_PATTERNS = {
    3: [(0, 1, 3), (0, 2, 4), (1, 2, 4), (1, 3, 4)],
    4: [(0, 1, 2, 4), (0, 1, 3, 4), (0, 2, 3, 4), (1, 2, 3, 4)],
    5: [(0, 1, 2, 3, 4)],
}


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _square_bbox(center: tuple[int, int], side: int) -> tuple[int, int, int, int]:
    r, c = center
    d = side // 2
    return int(r - d), int(c - d), int(r + d), int(c + d)


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 0) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _cycle_side(side: int, steps: int) -> int:
    idx = SIDE_CYCLE.index(int(side))
    return int(SIDE_CYCLE[(idx + int(steps)) % len(SIDE_CYCLE)])


def _extract_objects(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    colors = sorted(int(color) for color in np.unique(g) if color not in (bg, MARKER))
    assert 3 <= len(colors) <= 5, "expected 3-5 colored squares"

    objects: list[dict] = []
    marker_cover = np.zeros(g.shape, dtype=int)
    used_boxes: list[tuple[int, int, int, int]] = []

    for color in colors:
        coords = np.argwhere(g == color)
        assert coords.size, "every foreground color must appear"
        r0, c0, r1, c1 = _bbox_coords(coords)
        side = int(r1 - r0 + 1)
        assert side == int(c1 - c0 + 1), "colored object bbox must be square"
        assert side in ODD_SIDES, "square side must lie on the odd cycle"

        region = g[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all((region == color) | (region == MARKER)), "square bbox must contain only color or gray markers"
        assert region.size == side * side, "square bbox must be fully filled"

        box = (r0, c0, r1, c1)
        assert all(_boxes_separated(box, other, gap=0) for other in used_boxes), "input squares must not overlap"
        used_boxes.append(box)

        marker_mask = region == MARKER
        marker_cover[r0 : r1 + 1, c0 : c1 + 1] += marker_mask.astype(int)
        marker_count = int(marker_mask.sum())
        if side == 1:
            assert marker_count == 0, "1x1 square cannot contain markers"
        else:
            assert 0 <= marker_count <= 4, "marker count must stay within the puzzle range"

        objects.append(
            {
                "color": color,
                "side": side,
                "center": ((r0 + r1) // 2, (c0 + c1) // 2),
                "bbox": box,
                "marker_count": marker_count,
            }
        )

    gray_mask = g == MARKER
    assert np.all(marker_cover[gray_mask] == 1), "every gray marker must belong to exactly one square"
    return bg, objects


def _render_output(shape: tuple[int, int], bg: int, objects: list[dict]) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    painted = np.zeros(shape, dtype=bool)
    h, w = shape

    for obj in objects:
        side = _cycle_side(obj["side"], obj["marker_count"])
        r0, c0, r1, c1 = _square_bbox(obj["center"], side)
        assert 0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w, "transformed square must fit in bounds"
        region = (slice(r0, r1 + 1), slice(c0, c1 + 1))
        assert not painted[region].any(), "transformed squares must not overlap"
        out[region] = int(obj["color"])
        painted[region] = True
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count gray markers inside each colored square and cycle that square's side length by that many shrink steps.

    Input structure:
    - The grid has a black background and 3-5 separated solid odd-sided colored squares.
    - Square colors are distinct, and some squares contain gray marker pixels overlaid inside their square bbox.
    - Valid square side lengths belong to the odd cycle {1, 3, 5, 7, 9}; 1x1 squares contain no markers.

    Rule:
    - For each colored square, count gray pixels inside its bbox to obtain K.
    - Keep the square's center and color fixed, then apply K side-cycle steps: 9->7->5->3->1->9.
    - Remove all gray markers and repaint only the transformed solid squares on the background.

    Key insight:
    - The hidden control signal is local: each square uses its own gray-marker count as the number of times its size advances along the odd-side cycle.
    """
    bg, objects = _extract_objects(np.asarray(grid, dtype=int))
    return _render_output(grid.shape, bg, objects)


def _marker_positions(center: tuple[int, int], side: int, count: int) -> list[tuple[int, int]]:
    if count == 0:
        return []
    r, c = center
    inner = [
        (rr, cc)
        for rr in range(r - side // 2 + 1, r + side // 2)
        for cc in range(c - side // 2 + 1, c + side // 2)
    ]
    assert len(inner) >= count, "marker count exceeds strict interior capacity"
    return random.sample(inner, count)


def _feasible_sides(marker_count: int) -> list[int]:
    if marker_count == 0:
        return [1]
    return [side for side in ODD_SIDES if side >= 3 and (side - 2) * (side - 2) >= marker_count]


def _weighted_choice(options: list[int], weights: list[int]) -> int:
    return int(random.choices(options, weights=weights, k=1)[0])


def _sample_counts() -> list[int]:
    n = random.randint(*OBJECT_COUNT_RANGE)
    return list(random.choice(COUNT_PATTERNS[n]))


def _assign_sides(counts: list[int]) -> list[int] | None:
    sides: list[int | None] = [None] * len(counts)
    positive = [i for i, count in enumerate(counts) if count > 0]
    assert len(positive) >= 2

    shared_pair = random.sample(positive, 2)
    common = sorted(set(_feasible_sides(counts[shared_pair[0]])) & set(_feasible_sides(counts[shared_pair[1]])))
    if not common:
        return None
    shared_side = _weighted_choice(common, [3 if side >= 5 else 1 for side in common])
    for idx in shared_pair:
        sides[idx] = shared_side

    wrap_candidates = [i for i, count in enumerate(counts) if count in (3, 4)]
    if wrap_candidates and random.random() < 0.7:
        idx = random.choice(wrap_candidates)
        if counts[idx] == 3:
            sides[idx] = 5
        else:
            sides[idx] = random.choice([5, 7])

    for idx, count in enumerate(counts):
        if sides[idx] is not None:
            continue
        options = _feasible_sides(count)
        if count == 0:
            sides[idx] = 1
        elif count == 1:
            sides[idx] = _weighted_choice(options, [1, 3, 3, 2][: len(options)])
        elif count == 2:
            sides[idx] = _weighted_choice(options, [3, 3, 2][: len(options)])
        elif count == 3:
            sides[idx] = _weighted_choice(options, [5, 3, 2][: len(options)])
        else:
            sides[idx] = _weighted_choice(options, [4, 4, 2][: len(options)])

    out_sides = [_cycle_side(side, count) for side, count in zip(sides, counts)]
    if len(set(int(side) for side in sides)) < 2:
        for idx, count in enumerate(counts):
            alternatives = [side for side in _feasible_sides(count) if side != sides[idx]]
            if alternatives:
                sides[idx] = random.choice(alternatives)
                break
        out_sides = [_cycle_side(side, count) for side, count in zip(sides, counts)]

    if len(set(int(side) for side in sides)) < 2:
        return None
    if len(set(out_sides)) < 2:
        return None
    return [int(side) for side in sides]


def _place_centers(specs: list[dict], shape: tuple[int, int]) -> bool:
    h, w = shape
    occupied: list[tuple[int, int, int, int]] = []

    for spec in sorted(specs, key=lambda item: item["footprint"], reverse=True):
        margin = spec["footprint"] // 2
        centers = [(r, c) for r in range(margin, h - margin) for c in range(margin, w - margin)]
        random.shuffle(centers)
        placed = False
        for center in centers:
            box = _square_bbox(center, spec["footprint"])
            if all(_boxes_separated(box, other, gap=PLACEMENT_GAP) for other in occupied):
                spec["center"] = center
                occupied.append(box)
                placed = True
                break
        if not placed:
            return False
    return True


def _render_input(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for spec in specs:
        r0, c0, r1, c1 = _square_bbox(spec["center"], spec["side"])
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(spec["color"])
    for spec in specs:
        for r, c in spec["markers"]:
            grid[r, c] = MARKER
    return grid


def _semantic_checks(specs: list[dict]) -> bool:
    counts = [int(spec["marker_count"]) for spec in specs]
    in_sides = [int(spec["side"]) for spec in specs]
    out_sides = [int(spec["out_side"]) for spec in specs]
    repeated_input = len(set(in_sides)) < len(in_sides)
    wrapped = any(out_side > in_side for in_side, out_side in zip(in_sides, out_sides))
    has_zero = 0 in counts
    return (
        len(set(counts)) == len(counts)
        and repeated_input
        and len(set(in_sides)) >= 2
        and len(set(out_sides)) >= 2
        and any(out_side != in_side for in_side, out_side in zip(in_sides, out_sides))
        and sum([has_zero, wrapped, repeated_input]) >= 2
    )


def _build_specs() -> list[dict] | None:
    counts = _sample_counts()
    sides = _assign_sides(counts)
    if sides is None:
        return None

    colors = random.sample(PALETTE, len(counts))
    specs: list[dict] = []
    for color, side, count in zip(colors, sides, counts):
        out_side = _cycle_side(side, count)
        specs.append(
            {
                "color": int(color),
                "side": int(side),
                "marker_count": int(count),
                "out_side": int(out_side),
                "footprint": int(max(side, out_side)),
            }
        )
    return specs if _semantic_checks(specs) else None


def _validate_generated_scene(grid: np.ndarray, planned: list[dict]) -> bool:
    try:
        bg, parsed = _extract_objects(grid)
        if bg != 0 or len(parsed) != len(planned):
            return False
        by_color = {int(obj["color"]): obj for obj in parsed}
        for spec in planned:
            obj = by_color.get(int(spec["color"]))
            if obj is None:
                return False
            if obj["side"] != spec["side"] or obj["marker_count"] != spec["marker_count"]:
                return False
            if obj["center"] != spec["center"]:
                return False
        expected = _render_output(grid.shape, 0, parsed)
        if np.array_equal(expected, grid):
            return False
        return np.array_equal(main(grid), expected)
    except Exception:
        return False


def _generate_attempt() -> np.ndarray | None:
    shape = (
        random.randint(*GRID_SIDE_RANGE),
        random.randint(*GRID_SIDE_RANGE),
    )
    specs = _build_specs()
    if specs is None:
        return None
    if not _place_centers(specs, shape):
        return None

    for spec in specs:
        spec["markers"] = _marker_positions(spec["center"], spec["side"], spec["marker_count"])

    grid = _render_input(shape, specs)
    return grid if _validate_generated_scene(grid, specs) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid counting-driven square-cycle scene")
