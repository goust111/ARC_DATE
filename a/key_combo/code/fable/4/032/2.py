"""ARC task 032-2: same-colored singleton pixels control square outlines."""

from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]  # inclusive (r0, c0, r1, c1)


def _background(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _neighbors(r: int, c: int, connectivity: int = 4) -> Iterable[Coord]:
    if connectivity == 4:
        yield r - 1, c
        yield r + 1, c
        yield r, c - 1
        yield r, c + 1
        return
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr or dc:
                yield r + dr, c + dc


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[Coord]]:
    """Return connected components using the common ARC (row, column) convention."""
    h, w = mask.shape
    unseen = set(map(tuple, np.argwhere(mask)))
    result: list[list[Coord]] = []
    while unseen:
        start = unseen.pop()
        component = [start]
        queue = deque([start])
        while queue:
            r, c = queue.popleft()
            for nr, nc in _neighbors(r, c, connectivity):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) in unseen:
                    unseen.remove((nr, nc))
                    component.append((nr, nc))
                    queue.append((nr, nc))
        result.append(component)
    return result


def _bbox(coords: Iterable[Coord]) -> BBox:
    points = list(coords)
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _outline_coords(box: BBox) -> set[Coord]:
    r0, c0, r1, c1 = box
    points = {(r0, c) for c in range(c0, c1 + 1)}
    points.update((r1, c) for c in range(c0, c1 + 1))
    points.update((r, c0) for r in range(r0, r1 + 1))
    points.update((r, c1) for r in range(r0, r1 + 1))
    return points


def _is_square_outline(coords: Iterable[Coord]) -> bool:
    points = set(coords)
    if not points:
        return False
    r0, c0, r1, c1 = _bbox(points)
    side = r1 - r0 + 1
    return side >= 3 and side == c1 - c0 + 1 and points == _outline_coords((r0, c0, r1, c1))


def _strictly_inside(point: Coord, box: BBox) -> bool:
    r, c = point
    r0, c0, r1, c1 = box
    return r0 < r < r1 and c0 < c < c1


def main(grid: np.ndarray) -> np.ndarray:
    """Resize every hollow square by the count and location of its same-colored singleton controls.

    Input structure:
    A dominant single-color background contains mutually separated, one-cell-thick
    hollow square outlines with distinct colors.  Any remaining foreground objects
    are isolated single pixels; each has the color of exactly one outline, and all
    controls of one color lie either strictly inside that outline or outside it.

    Rule:
    For each outline, let k be its number of same-colored singleton pixels.  Controls
    inside the outline shrink all four sides inward by k cells, while controls outside
    expand all four sides outward by k cells.  With k=0 the outline is unchanged.
    Erase every control pixel and redraw each one-cell outline at its original center.

    Key insight:
    The apparent noise is a consumed instruction: its color chooses the square, its
    count gives the number of layers, and the inside/outside relation gives the sign.
    """
    g = np.asarray(grid)
    assert g.ndim == 2, "grid must be two-dimensional"
    h, w = g.shape
    bg = _background(g)
    colors = [int(v) for v in np.unique(g) if int(v) != bg]
    assert colors, "at least one square outline is required"

    targets: list[tuple[int, BBox]] = []
    all_controls: list[Coord] = []

    for color in colors:
        comps = _components(g == color, connectivity=4)
        frame_comps = [comp for comp in comps if _is_square_outline(comp)]
        assert len(frame_comps) == 1, "each foreground color must identify one square outline"

        frame = frame_comps[0]
        frame_box = _bbox(frame)
        controls = [comp[0] for comp in comps if comp is not frame and len(comp) == 1]
        assert len(controls) == len(comps) - 1, "non-frame components must be single pixels"

        # A control is isolated in the visible, color-independent 8-neighborhood.
        for r, c in controls:
            for nr, nc in _neighbors(r, c, connectivity=8):
                if 0 <= nr < h and 0 <= nc < w:
                    assert int(g[nr, nc]) == bg, "control pixels must be isolated"

        k = len(controls)
        inside_flags = [_strictly_inside(p, frame_box) for p in controls]
        assert not inside_flags or all(inside_flags) or not any(inside_flags), (
            "same-colored controls cannot mix inside and outside"
        )

        r0, c0, r1, c1 = frame_box
        delta = k if inside_flags and inside_flags[0] else -k
        target = (r0 + delta, c0 + delta, r1 - delta, c1 - delta)
        tr0, tc0, tr1, tc1 = target
        assert 0 <= tr0 <= tr1 < h and 0 <= tc0 <= tc1 < w, "resized square leaves canvas"
        assert tr1 - tr0 + 1 == tc1 - tc0 + 1 >= 3, "contracted square is too small"
        targets.append((color, target))
        all_controls.extend(controls)

    # Validate the promised non-contact property before painting, independent of color ids.
    occupied: set[Coord] = set()
    for _color, box in targets:
        outline = _outline_coords(box)
        for r, c in outline:
            assert all((nr, nc) not in occupied for nr, nc in _neighbors(r, c, 8)), (
                "resized outlines must not touch"
            )
        occupied.update(outline)

    out = np.full_like(g, bg)
    for color, box in targets:
        for r, c in _outline_coords(box):
            out[r, c] = color
    return out


# ---------------------------------------------------------------------------
# Generator: it deliberately includes counterexamples to the common shortcuts.
# Every sample has k=1 and k=2 for both signs, plus an unchanged k=0 square.
# ---------------------------------------------------------------------------


_RNG = np.random.default_rng()
_MAX_ATTEMPTS = 200


def _paint_outline(grid: np.ndarray, box: BBox, color: int) -> None:
    for r, c in _outline_coords(box):
        grid[r, c] = color


def _zone_grid(h: int, w: int) -> list[BBox]:
    row_edges = (0, h // 2, h)
    col_edges = (0, w // 3, 2 * w // 3, w)
    return [
        (row_edges[i], col_edges[j], row_edges[i + 1] - 1, col_edges[j + 1] - 1)
        for i in range(2)
        for j in range(3)
    ]


def _random_square_in_zone(zone: BBox, side: int) -> BBox:
    zr0, zc0, zr1, zc1 = zone
    # One empty cell around each zone footprint separates both input and output objects.
    r0 = int(_RNG.integers(zr0 + 1, zr1 - side + 2))
    c0 = int(_RNG.integers(zc0 + 1, zc1 - side + 2))
    return r0, c0, r0 + side - 1, c0 + side - 1


def _inset(box: BBox, amount: int) -> BBox:
    r0, c0, r1, c1 = box
    return r0 + amount, c0 + amount, r1 - amount, c1 - amount


def _cell_is_isolated_background(grid: np.ndarray, point: Coord, bg: int) -> bool:
    h, w = grid.shape
    r, c = point
    if int(grid[r, c]) != bg:
        return False
    return all(
        int(grid[nr, nc]) == bg
        for nr, nc in _neighbors(r, c, 8)
        if 0 <= nr < h and 0 <= nc < w
    )


def _choose_spaced(candidates: list[Coord], count: int, grid: np.ndarray, bg: int) -> list[Coord] | None:
    order = _RNG.permutation(len(candidates))
    chosen: list[Coord] = []
    for index in order:
        point = candidates[int(index)]
        if not _cell_is_isolated_background(grid, point, bg):
            continue
        r, c = point
        if any(max(abs(r - pr), abs(c - pc)) <= 1 for pr, pc in chosen):
            continue
        chosen.append(point)
        if len(chosen) == count:
            return chosen
    return None


def _validate_generated(inp: np.ndarray, expected: np.ndarray, specs: list[dict], bg: int) -> bool:
    """Check the described semantics directly, then compare the solver to an independent oracle."""
    required = {("shrink", 1), ("shrink", 2), ("dilate", 1), ("dilate", 2), ("same", 0)}
    if not required.issubset({(s["direction"], s["k"]) for s in specs}):
        return False
    if len({s["color"] for s in specs}) != len(specs):
        return False

    rebuilt = np.full_like(inp, bg)
    for spec in specs:
        original = spec["input_box"]
        target = spec["output_box"]
        controls = spec["controls"]
        color = spec["color"]
        k = spec["k"]
        original_side = original[2] - original[0] + 1
        target_side = target[2] - target[0] + 1

        if len(controls) != k:
            return False
        if spec["direction"] == "shrink":
            if not all(_strictly_inside(p, original) for p in controls):
                return False
            if target_side != original_side - 2 * k:
                return False
        elif spec["direction"] == "dilate":
            if not all(not _strictly_inside(p, original) for p in controls):
                return False
            if target_side != original_side + 2 * k:
                return False
        else:
            if controls or target != original:
                return False

        if set(map(tuple, np.argwhere(inp == color))) != _outline_coords(original) | set(controls):
            return False
        for point in controls:
            r, c = point
            if any(
                int(inp[nr, nc]) != bg
                for nr, nc in _neighbors(r, c, 8)
                if 0 <= nr < inp.shape[0] and 0 <= nc < inp.shape[1]
            ):
                return False
        _paint_outline(rebuilt, target, color)

    if not np.array_equal(rebuilt, expected) or np.array_equal(inp, expected):
        return False
    try:
        if not np.array_equal(main(inp), expected):
            return False
        # Removing one control must alter that square's result: count is causally necessary.
        for spec in specs:
            if spec["k"] == 0:
                continue
            counterfactual = inp.copy()
            r, c = spec["controls"][0]
            counterfactual[r, c] = bg
            if np.array_equal(main(counterfactual), expected):
                return False
    except (AssertionError, ValueError):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    h = int(_RNG.integers(27, 31))
    w = int(_RNG.integers(27, 31))
    palette = list(map(int, _RNG.permutation(10)))
    bg = palette.pop()

    roles: list[tuple[str, int]] = [
        ("shrink", 1),
        ("shrink", 2),
        ("dilate", 1),
        ("dilate", 2),
        ("same", 0),
    ]
    if bool(_RNG.integers(0, 2)):
        direction = ("shrink", "dilate", "same")[int(_RNG.integers(0, 3))]
        roles.append((direction, 0 if direction == "same" else int(_RNG.integers(1, 3))))

    zones = _zone_grid(h, w)
    _RNG.shuffle(zones)
    _RNG.shuffle(roles)
    inp = np.full((h, w), bg, dtype=int)
    expected = np.full((h, w), bg, dtype=int)
    specs: list[dict] = []

    for index, (direction, k) in enumerate(roles):
        zone = zones[index]
        zr0, zc0, zr1, zc1 = zone
        max_foot = min(zr1 - zr0 - 1, zc1 - zc0 - 1)
        if direction == "shrink":
            min_side = 2 * k + 3
            if min_side > max_foot:
                return None
            input_side = int(_RNG.integers(min_side, max_foot + 1))
            input_box = _random_square_in_zone(zone, input_side)
            output_box = _inset(input_box, k)
        elif direction == "dilate":
            max_input_side = max_foot - 2 * k
            if max_input_side < 3:
                return None
            input_side = int(_RNG.integers(3, max_input_side + 1))
            output_side = input_side + 2 * k
            output_box = _random_square_in_zone(zone, output_side)
            input_box = _inset(output_box, k)
        else:
            input_side = int(_RNG.integers(3, max_foot + 1))
            input_box = _random_square_in_zone(zone, input_side)
            output_box = input_box

        color = palette[index]
        _paint_outline(inp, input_box, color)
        _paint_outline(expected, output_box, color)
        specs.append(
            {
                "direction": direction,
                "k": k,
                "color": color,
                "zone": zone,
                "input_box": input_box,
                "output_box": output_box,
                "controls": [],
            }
        )

    # Add controls only after every frame exists, so isolation is color-independent.
    for spec in specs:
        k = spec["k"]
        if k == 0:
            continue
        r0, c0, r1, c1 = spec["input_box"]
        if spec["direction"] == "shrink":
            candidates = [
                (r, c)
                for r in range(r0 + 1, r1)
                for c in range(c0 + 1, c1)
            ]
        else:
            zr0, zc0, zr1, zc1 = spec["zone"]
            target = spec["output_box"]
            tr0, tc0, tr1, tc1 = target
            # Keeping controls beyond the final contour makes erasure visibly distinct.
            candidates = [
                (r, c)
                for r in range(zr0, zr1 + 1)
                for c in range(zc0, zc1 + 1)
                if not (tr0 <= r <= tr1 and tc0 <= c <= tc1)
            ]
        chosen = _choose_spaced(candidates, k, inp, bg)
        if chosen is None:
            return None
        for r, c in chosen:
            inp[r, c] = spec["color"]
        spec["controls"] = chosen

    return inp if _validate_generated(inp, expected, specs, bg) else None


def generate_input() -> np.ndarray:
    """Generate a varied, unambiguous grid with all essential semantic branches."""
    for _ in range(_MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid noise-controls-square sample")
