from __future__ import annotations

import random

import numpy as np


GRID_SHAPE = (16, 16)
BACKGROUND = 7
MARKER = 4
DIAMOND_COLORS = (1, 2, 3, 5)
MAX_GENERATION_ATTEMPTS = 500
MIN_CENTER_GAP = 1


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _diamond_outline(center: tuple[int, int], radius: int) -> set[tuple[int, int]]:
    cr, cc = map(int, center)
    radius = int(radius)
    return {
        (r, c)
        for r in range(GRID_SHAPE[0])
        for c in range(GRID_SHAPE[1])
        if abs(r - cr) + abs(c - cc) == radius
    }


def _diamond_filled(center: tuple[int, int], radius: int) -> set[tuple[int, int]]:
    cr, cc = map(int, center)
    radius = int(radius)
    return {
        (r, c)
        for r in range(GRID_SHAPE[0])
        for c in range(GRID_SHAPE[1])
        if abs(r - cr) + abs(c - cc) <= radius
    }


def _diamond_interior(center: tuple[int, int], radius: int) -> list[tuple[int, int]]:
    cr, cc = map(int, center)
    radius = int(radius)
    return [
        (r, c)
        for r in range(GRID_SHAPE[0])
        for c in range(GRID_SHAPE[1])
        if abs(r - cr) + abs(c - cc) < radius
    ]


def _infer_diamond(coords: np.ndarray) -> tuple[tuple[int, int], int]:
    assert coords.ndim == 2 and coords.shape[1] == 2 and coords.shape[0] >= 4
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    assert height == width and height % 2 == 1 and height >= 3

    center = (int((r0 + r1) // 2), int((c0 + c1) // 2))
    radius = height // 2
    expected = _diamond_outline(center, radius)
    actual = {tuple(map(int, pt)) for pt in coords}
    assert actual == expected, "non-clipped full Manhattan diamond outline expected"
    return center, radius


def main(grid: np.ndarray) -> np.ndarray:
    """Count interior yellow markers and shrink or expand each diamond outline by one layer based on parity.

    Input structure:
    - A 16x16 orange background contains 3-4 one-pixel-thick Manhattan-diamond outlines.
    - Each outline uses a distinct color chosen from blue, red, green, and gray.
    - Yellow marker pixels lie strictly inside the diamonds, not on their outlines.

    Rule:
    - For each diamond, count the yellow markers whose Manhattan distance to that diamond's center is smaller than its radius.
    - If the count is odd, redraw that outline with radius minus one; if the count is even, redraw it with radius plus one.
    - Keep the original center and outline color, and erase all yellow markers to the orange background.

    Key insight:
    - The yellow pixels are only an encoded control signal: their parity selects whether the matched diamond contracts or expands.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == GRID_SHAPE
    assert int(np.bincount(g.ravel(), minlength=10).argmax()) == BACKGROUND

    object_colors = sorted(
        int(color) for color in np.unique(g) if int(color) not in (BACKGROUND, MARKER)
    )
    assert 3 <= len(object_colors) <= 4
    assert set(object_colors) <= set(DIAMOND_COLORS)

    diamonds: list[dict] = []
    for color in object_colors:
        coords = np.argwhere(g == int(color))
        center, radius = _infer_diamond(coords)
        diamonds.append({"color": int(color), "center": center, "radius": int(radius)})

    marker_coords = [tuple(map(int, pt)) for pt in np.argwhere(g == MARKER)]
    assert marker_coords

    for mr, mc in marker_coords:
        owners = [
            diamond
            for diamond in diamonds
            if _manhattan((mr, mc), diamond["center"]) < int(diamond["radius"])
        ]
        assert len(owners) == 1, "each marker must belong to exactly one diamond interior"

    out = g.copy()
    out[out == MARKER] = BACKGROUND

    for diamond in diamonds:
        color = int(diamond["color"])
        center = diamond["center"]
        radius = int(diamond["radius"])
        count = sum(1 for pt in marker_coords if _manhattan(pt, center) < radius)
        assert 1 <= count <= 4
        new_radius = radius - 1 if count % 2 == 1 else radius + 1
        assert new_radius >= 1

        out[out == color] = BACKGROUND
        for r, c in _diamond_outline(center, new_radius):
            out[r, c] = color

    return out


def _render_grid(specs: list[dict], *, output: bool) -> np.ndarray:
    grid = np.full(GRID_SHAPE, BACKGROUND, dtype=int)
    for spec in specs:
        radius = int(spec["radius_out"] if output else spec["radius_in"])
        for r, c in _diamond_outline(spec["center"], radius):
            assert grid[r, c] == BACKGROUND
            grid[r, c] = int(spec["color"])
    if not output:
        for spec in specs:
            for r, c in spec["markers"]:
                assert grid[r, c] == BACKGROUND
                grid[r, c] = MARKER
    return grid


def _sample_branch_counts(n_objects: int) -> list[int]:
    counts = [random.choice((1, 3)), random.choice((2, 4))]
    while len(counts) < int(n_objects):
        counts.append(random.choice((1, 2, 3, 4)))
    random.shuffle(counts)
    return counts


def _sample_radius_in(count: int, n_objects: int) -> int:
    if count % 2 == 1:
        pool = [2, 3, 4, 5] if n_objects == 3 else [2, 3, 4]
    else:
        pool = [2, 3, 4, 5] if n_objects == 3 else [2, 3, 4]
    return int(random.choice(pool))


def _place_specs(base_specs: list[dict]) -> list[dict] | None:
    placed: list[dict] = []
    order = sorted(
        base_specs,
        key=lambda spec: max(int(spec["radius_in"]), int(spec["radius_out"])),
        reverse=True,
    )

    for spec in order:
        safe_radius = max(int(spec["radius_in"]), int(spec["radius_out"]))
        centers = [
            (r, c)
            for r in range(safe_radius, GRID_SHAPE[0] - safe_radius)
            for c in range(safe_radius, GRID_SHAPE[1] - safe_radius)
        ]
        random.shuffle(centers)

        chosen = None
        for center in centers:
            if all(
                _manhattan(center, other["center"])
                > safe_radius + other["safe_radius"] + MIN_CENTER_GAP
                for other in placed
            ):
                chosen = center
                break
        if chosen is None:
            return None

        spec = dict(spec)
        spec["center"] = chosen
        spec["safe_radius"] = safe_radius
        placed.append(spec)

    return placed


def _add_markers(specs: list[dict]) -> bool:
    used = set()
    for spec in specs:
        interior = _diamond_interior(spec["center"], int(spec["radius_in"]))
        if len(interior) < int(spec["count"]):
            return False
        choices = [pt for pt in interior if pt not in used]
        if len(choices) < int(spec["count"]):
            return False
        spec["markers"] = random.sample(choices, int(spec["count"]))
        used.update(spec["markers"])
    return True


def _independent_expected(specs: list[dict]) -> np.ndarray:
    expected = np.full(GRID_SHAPE, BACKGROUND, dtype=int)
    for spec in specs:
        for r, c in _diamond_outline(spec["center"], int(spec["radius_out"])):
            assert expected[r, c] == BACKGROUND
            expected[r, c] = int(spec["color"])
    return expected


def _validate_specs(specs: list[dict]) -> bool:
    all_markers = [pt for spec in specs for pt in spec["markers"]]
    if len(set(all_markers)) != len(all_markers):
        return False

    for spec in specs:
        center = spec["center"]
        radius_in = int(spec["radius_in"])
        radius_out = int(spec["radius_out"])
        count = int(spec["count"])

        if count not in (1, 2, 3, 4):
            return False
        if count % 2 == 1 and radius_out != radius_in - 1:
            return False
        if count % 2 == 0 and radius_out != radius_in + 1:
            return False
        if radius_out < 1:
            return False

        for marker in spec["markers"]:
            if _manhattan(marker, center) >= radius_in:
                return False

    for i, left in enumerate(specs):
        for right in specs[i + 1 :]:
            if not (
                _diamond_filled(left["center"], left["safe_radius"])
                .isdisjoint(_diamond_filled(right["center"], right["safe_radius"]))
            ):
                return False

    return True


def _gen_attempt() -> np.ndarray | None:
    n_objects = random.choice((3, 3, 4))
    colors = random.sample(DIAMOND_COLORS, n_objects)
    counts = _sample_branch_counts(n_objects)
    base_specs = []
    for color, count in zip(colors, counts):
        radius_in = _sample_radius_in(count, n_objects)
        radius_out = radius_in - 1 if count % 2 == 1 else radius_in + 1
        base_specs.append(
            {
                "color": int(color),
                "count": int(count),
                "radius_in": int(radius_in),
                "radius_out": int(radius_out),
            }
        )

    specs = _place_specs(base_specs)
    if specs is None or not _add_markers(specs) or not _validate_specs(specs):
        return None

    inp = _render_grid(specs, output=False)
    expected = _independent_expected(specs)

    if not np.array_equal(expected, _render_grid(specs, output=True)):
        return None
    if not np.array_equal(main(inp), expected):
        return None

    marker_total = sum(spec["count"] for spec in specs)
    if marker_total < n_objects + 1:
        return None
    if not any(spec["count"] % 2 == 1 for spec in specs):
        return None
    if not any(spec["count"] % 2 == 0 for spec in specs):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid diamond-parity sample")
