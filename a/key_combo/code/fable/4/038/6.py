from __future__ import annotations

import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return the 4-connected components of a boolean mask as (r, c) arrays."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    result: list[np.ndarray] = []
    for start_r, start_c in np.argwhere(mask):
        sr, sc = int(start_r), int(start_c)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        stack = [(sr, sc)]
        coords: list[Coord] = []
        while stack:
            r, c = stack.pop()
            coords.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        result.append(np.asarray(coords, dtype=int))
    return result


def _bbox(mask: np.ndarray) -> BBox:
    coords = np.argwhere(mask)
    assert coords.size, "cannot take the bounding box of an empty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: BBox) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _background(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    largest = int(counts.max())
    assert int(np.count_nonzero(counts == largest)) == 1, "background must be uniquely dominant"
    return int(colors[int(np.argmax(counts))])


def _monochrome_components(grid: np.ndarray, bg: int) -> list[tuple[int, np.ndarray]]:
    result: list[tuple[int, np.ndarray]] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        result.extend((color, coords) for coords in _components(grid == color))
    return result


def main(grid: np.ndarray) -> np.ndarray:
    """Use the exterior point count to select a cavity by its remaining background area, then crop its sprite.

    Input structure:
    - A dominant background contains several differently colored, closed hollow shapes.
    - Each shape encloses one 4-connected background cavity and one non-wall-touching colored sprite.
    - A further color occurs only as isolated exterior pixels; cavity remaining-background areas are distinct.

    Rule:
    - Count the same-colored isolated exterior pixels, find the unique cavity containing exactly that many
      background cells after excluding its sprite, and return the sprite's inclusive bounding-box crop.

    Key insight:
    - The points are a two-hop pointer to a hidden sprite, and they match the cavity's remaining negative
      space rather than its total interior area.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = _background(g)

    by_color: dict[int, list[np.ndarray]] = {}
    for color in map(int, np.unique(g)):
        if color != bg:
            by_color[color] = _components(g == color)

    point_colors = [
        color
        for color, comps in by_color.items()
        if len(comps) >= 2 and all(len(comp) == 1 for comp in comps)
    ]
    assert len(point_colors) == 1, "expected one color used only by isolated count points"
    point_color = point_colors[0]
    point_count = len(by_color[point_color])

    h, w = g.shape
    cavities: list[tuple[int, np.ndarray]] = []
    for color, wall_coords in _monochrome_components(g, bg):
        if color == point_color or len(wall_coords) < 4:
            continue
        wall = np.zeros(g.shape, dtype=bool)
        wall[wall_coords[:, 0], wall_coords[:, 1]] = True
        enclosed = []
        for region_coords in _components(~wall):
            rows = region_coords[:, 0]
            cols = region_coords[:, 1]
            if not np.any((rows == 0) | (rows == h - 1) | (cols == 0) | (cols == w - 1)):
                enclosed.append(region_coords)
        if len(enclosed) != 1:
            continue

        region = np.zeros(g.shape, dtype=bool)
        coords = enclosed[0]
        region[coords[:, 0], coords[:, 1]] = True
        sprite = region & (g != bg)
        if not sprite.any() or np.any(g[sprite] == point_color):
            continue
        remaining_background = int(np.count_nonzero(region & (g == bg)))
        cavities.append((remaining_background, sprite))

    assert len(cavities) >= 2, "expected several closed sprite-bearing cavities"
    assert len({area for area, _ in cavities}) == len(cavities), "cavity background areas must be distinct"
    selected = [sprite for area, sprite in cavities if area == point_count]
    assert len(selected) == 1, "point count must select exactly one cavity"
    return _crop_bbox(g, _bbox(selected[0]))


# ------------------------------ generator ------------------------------

_PALETTE = tuple(range(10))
_MAX_ATTEMPTS = 500
_OUTPUT_HISTORY_LIMIT = 256
_used_output_signatures: set[tuple[tuple[int, int], bytes]] = set()


def _oriented(shape: tuple[int, int]) -> tuple[int, int]:
    return shape if random.choice((False, True)) else shape[::-1]


def _grow_mask(h: int, w: int, area: int) -> np.ndarray:
    """Grow a random 4-connected, hole-free small sprite inside h x w."""
    assert 1 <= area <= h * w
    mask = np.zeros((h, w), dtype=bool)
    r, c = random.randrange(h), random.randrange(w)
    mask[r, c] = True
    while int(mask.sum()) < area:
        frontier: set[Coord] = set()
        for rr, cc in np.argwhere(mask):
            for nr, nc in ((int(rr) - 1, int(cc)), (int(rr) + 1, int(cc)),
                           (int(rr), int(cc) - 1), (int(rr), int(cc) + 1)):
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((nr, nc))
        nr, nc = random.choice(tuple(frontier))
        mask[nr, nc] = True
    return mask


def _base_specs() -> tuple[int, list[dict[str, int | str]]]:
    # In every configuration: target total - target sprite = n, while trap total = n.
    configurations = {
        12: ((4, 4), 4, (3, 4)),
        15: ((4, 5), 5, (3, 5)),
        16: ((4, 5), 4, (4, 4)),
        18: ((4, 6), 6, (3, 6)),
    }
    n = random.choice(tuple(configurations))
    target_shape, target_area, trap_shape = configurations[n]
    target_h, target_w = _oriented(target_shape)
    trap_h, trap_w = _oriented(trap_shape)
    trap_capacity = (trap_h - 2) * (trap_w - 2)
    trap_area = random.randint(2, min(4, trap_capacity))

    specs: list[dict[str, int | str]] = [
        {"role": "target", "ih": target_h, "iw": target_w, "sprite_area": target_area},
        {"role": "trap", "ih": trap_h, "iw": trap_w, "sprite_area": trap_area},
    ]
    used_remaining = {n, n - trap_area}
    desired_count = random.choice((3, 4))
    while len(specs) < desired_count:
        ih, iw = random.randint(3, 6), random.randint(3, 6)
        total = ih * iw
        capacity = (ih - 2) * (iw - 2)
        if total == n or capacity < 2:
            continue
        sprite_area = random.randint(2, min(6, capacity))
        remaining = total - sprite_area
        if remaining in used_remaining:
            continue
        used_remaining.add(remaining)
        specs.append({"role": "distractor", "ih": ih, "iw": iw, "sprite_area": sprite_area})
    random.shuffle(specs)
    return n, specs


def _place_boxes(h: int, w: int, specs: list[dict[str, int | str]]) -> bool:
    occupied = np.zeros((h, w), dtype=bool)
    ordered = sorted(specs, key=lambda s: (int(s["ih"]) + 2) * (int(s["iw"]) + 2), reverse=True)
    for spec in ordered:
        oh, ow = int(spec["ih"]) + 2, int(spec["iw"]) + 2
        candidates: list[Coord] = []
        for r0 in range(2, h - oh - 1):
            for c0 in range(2, w - ow - 1):
                rr0, cc0 = max(0, r0 - 2), max(0, c0 - 2)
                rr1, cc1 = min(h, r0 + oh + 2), min(w, c0 + ow + 2)
                if not occupied[rr0:rr1, cc0:cc1].any():
                    candidates.append((r0, c0))
        if not candidates:
            return False
        r0, c0 = random.choice(candidates)
        occupied[r0 : r0 + oh, c0 : c0 + ow] = True
        spec["r0"], spec["c0"] = r0, c0
    return True


def _render_frames(
    grid: np.ndarray,
    specs: list[dict[str, int | str]],
    shape_colors: list[int],
    sprite_colors: list[int],
) -> np.ndarray:
    target_sprite = np.zeros(grid.shape, dtype=bool)
    for spec, shape_color, sprite_color in zip(specs, shape_colors, sprite_colors):
        r0, c0 = int(spec["r0"]), int(spec["c0"])
        ih, iw = int(spec["ih"]), int(spec["iw"])
        oh, ow = ih + 2, iw + 2
        grid[r0, c0 : c0 + ow] = shape_color
        grid[r0 + oh - 1, c0 : c0 + ow] = shape_color
        grid[r0 : r0 + oh, c0] = shape_color
        grid[r0 : r0 + oh, c0 + ow - 1] = shape_color

        sprite_mask = _grow_mask(ih - 2, iw - 2, int(spec["sprite_area"]))
        view = grid[r0 + 2 : r0 + ih, c0 + 2 : c0 + iw]
        view[sprite_mask] = sprite_color
        spec["shape_color"] = shape_color
        spec["sprite_color"] = sprite_color
        spec["sprite_mask"] = sprite_mask  # type: ignore[assignment]
        if spec["role"] == "target":
            local_coords = np.argwhere(sprite_mask)
            target_sprite[r0 + 2 + local_coords[:, 0], c0 + 2 + local_coords[:, 1]] = True
    return target_sprite


def _scatter_points(grid: np.ndarray, bg: int, color: int, count: int,
                    specs: list[dict[str, int | str]]) -> bool:
    h, w = grid.shape
    boxes = [
        (int(s["r0"]), int(s["c0"]), int(s["r0"]) + int(s["ih"]) + 1,
         int(s["c0"]) + int(s["iw"]) + 1)
        for s in specs
    ]
    for _ in range(count):
        candidates: list[Coord] = []
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                if any(r0 <= r <= r1 and c0 <= c <= c1 for r0, c0, r1, c1 in boxes):
                    continue
                if grid[r, c] == bg and np.all(grid[r - 1 : r + 2, c - 1 : c + 2] == bg):
                    candidates.append((r, c))
        if not candidates:
            return False
        r, c = random.choice(candidates)
        grid[r, c] = color
    return True


def _validate_scene(grid: np.ndarray, expected: np.ndarray, bg: int, point_color: int,
                    n: int, specs: list[dict[str, int | str]]) -> None:
    """Independent generator-side checks for the stated geometry and both competing selectors."""
    assert grid.ndim == 2 and max(grid.shape) <= 30
    assert int(np.count_nonzero(grid == point_color)) == n
    point_components = _components(grid == point_color)
    assert len(point_components) == n and all(len(comp) == 1 for comp in point_components)

    remaining: list[int] = []
    totals: list[int] = []
    target_index = trap_index = -1
    for i, spec in enumerate(specs):
        r0, c0 = int(spec["r0"]), int(spec["c0"])
        ih, iw = int(spec["ih"]), int(spec["iw"])
        shape_color, sprite_color = int(spec["shape_color"]), int(spec["sprite_color"])
        patch = grid[r0 : r0 + ih + 2, c0 : c0 + iw + 2]
        wall = np.zeros(patch.shape, dtype=bool)
        wall[[0, -1], :] = True
        wall[:, [0, -1]] = True
        assert np.all(patch[wall] == shape_color)
        interior = patch[1:-1, 1:-1]
        assert np.all(interior[0, :] == bg) and np.all(interior[-1, :] == bg)
        assert np.all(interior[:, 0] == bg) and np.all(interior[:, -1] == bg)
        sprite = interior == sprite_color
        assert int(sprite.sum()) == int(spec["sprite_area"])
        assert len(_components(sprite)) == 1
        assert len(_components(interior == bg)) == 1
        assert set(map(int, np.unique(interior))) == {bg, sprite_color}
        totals.append(ih * iw)
        remaining.append(int(np.count_nonzero(interior == bg)))
        if spec["role"] == "target":
            target_index = i
        elif spec["role"] == "trap":
            trap_index = i

    assert target_index >= 0 and trap_index >= 0 and target_index != trap_index
    assert len(set(remaining)) == len(remaining)
    assert [i for i, area in enumerate(remaining) if area == n] == [target_index]
    assert [i for i, area in enumerate(totals) if area == n] == [trap_index]
    assert totals[target_index] != n and remaining[trap_index] != n
    solved = main(grid)
    assert np.array_equal(solved, expected)
    assert point_color not in set(map(int, solved.ravel()))
    assert int(specs[target_index]["sprite_color"]) in set(map(int, solved.ravel()))


def _generate_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    n, specs = _base_specs()
    h, w = random.randint(24, 30), random.randint(24, 30)
    if not _place_boxes(h, w, specs):
        return None

    bg = random.choice(_PALETTE)
    colors = random.sample([color for color in _PALETTE if color != bg], 1 + 2 * len(specs))
    point_color = colors[0]
    shape_colors = colors[1 : 1 + len(specs)]
    sprite_colors = colors[1 + len(specs) :]
    grid = np.full((h, w), bg, dtype=int)
    target_sprite = _render_frames(grid, specs, shape_colors, sprite_colors)
    if not _scatter_points(grid, bg, point_color, n, specs):
        return None
    expected = _crop_bbox(grid, _bbox(target_sprite))
    try:
        _validate_scene(grid, expected, bg, point_color, n, specs)
    except AssertionError:
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    global _used_output_signatures
    if len(_used_output_signatures) >= _OUTPUT_HISTORY_LIMIT:
        _used_output_signatures.clear()
    for _ in range(_MAX_ATTEMPTS):
        generated = _generate_attempt()
        if generated is None:
            continue
        grid, expected = generated
        signature = (tuple(map(int, expected.shape)), expected.tobytes())
        if signature in _used_output_signatures:
            continue
        _used_output_signatures.add(signature)
        return grid
    raise RuntimeError("failed to generate a valid count-selects-cavity input")
