from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

LEGEND_PATCH = np.array([[BLUE, RED], [GREEN, YELLOW]], dtype=int)
ORTHOGONALS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAGONALS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

GRID_SIZE_RANGE = (14, 18)
OBJECT_COUNT_CHOICES = [4, 5, 6]
MARGIN = 2
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rect_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _expand_bbox(bbox: tuple[int, int, int, int], pad: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return r0 - pad, c0 - pad, r1 + pad, c1 + pad


def _grey_components(grid: np.ndarray) -> list[dict]:
    mask = grid == GREY
    seen = np.zeros_like(mask, dtype=bool)
    h, w = grid.shape
    components = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in ORTHOGONALS:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(grid.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            coord_array = np.array(coords, dtype=int)
            components.append(
                {
                    "coords": coord_array,
                    "coord_set": set(coords),
                    "bbox": _bbox_from_coords(coord_array),
                    "area": len(coords),
                }
            )
    return components


def _legend_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    return 0, grid.shape[1] - 2, 1, grid.shape[1] - 1


def _legend_colors(grid: np.ndarray) -> dict[int, int]:
    legend = np.asarray(grid[:2, -2:], dtype=int)
    assert legend.shape == (2, 2)
    assert set(map(int, legend.ravel())) == {BLUE, RED, GREEN, YELLOW}
    return {
        1: int(legend[0, 0]),
        2: int(legend[0, 1]),
        3: int(legend[1, 0]),
        4: int(legend[1, 1]),
    }


def _diagonal_only_targets(grid: np.ndarray, component: dict) -> set[tuple[int, int]]:
    coord_set = component["coord_set"]
    orth_neighbors = set()
    diag_neighbors = set()
    for r, c in coord_set:
        for dr, dc in ORTHOGONALS:
            nr, nc = r + dr, c + dc
            if _in_bounds(grid.shape, nr, nc):
                orth_neighbors.add((nr, nc))
        for dr, dc in DIAGONALS:
            nr, nc = r + dr, c + dc
            if _in_bounds(grid.shape, nr, nc):
                diag_neighbors.add((nr, nc))
    return {
        cell
        for cell in diag_neighbors
        if cell not in coord_set and cell not in orth_neighbors and int(grid[cell]) == BLACK
    }


def _apply_rule(grid: np.ndarray, components: list[dict], legend_by_size: dict[int, int]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for component in components:
        area = int(component["area"])
        color = int(legend_by_size[area])
        coords = component["coords"]
        coord_set = component["coord_set"]
        r0, c0, r1, c1 = component["bbox"]

        if area == 1:
            r, c = map(int, coords[0])
            for dr, dc in ORTHOGONALS:
                nr, nc = r + dr, c + dc
                assert _in_bounds(grid.shape, nr, nc)
                assert int(grid[nr, nc]) == BLACK
                out[nr, nc] = color
        elif area == 2:
            for nr, nc in _diagonal_only_targets(grid, component):
                out[nr, nc] = color
        elif area == 3:
            for rr in range(r0, r1 + 1):
                for cc in range(c0, c1 + 1):
                    if rr in (r0, r1) or cc in (c0, c1):
                        if (rr, cc) not in coord_set:
                            assert int(grid[rr, cc]) == BLACK
                            out[rr, cc] = color
        elif area == 4:
            out[r0 : r1 + 1, c0 : c1 + 1] = color
        else:
            raise AssertionError(f"unexpected grey component size: {area}")
    return out


def _validate_input_structure(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert max(h, w) <= 30
    assert h >= 8 and w >= 8

    legend = g[:2, -2:]
    assert np.array_equal(legend, LEGEND_PATCH)

    allowed = {BLACK, BLUE, RED, GREEN, YELLOW, GREY}
    assert set(map(int, np.unique(g))).issubset(allowed)

    components = _grey_components(g)
    assert 4 <= len(components) <= 6
    areas = sorted(int(component["area"]) for component in components)
    assert all(area in {1, 2, 3, 4} for area in areas)
    assert {1, 2, 3, 4}.issubset(set(areas))

    legend_zone = _expand_bbox(_legend_bbox(g), MARGIN)
    zones = []
    for component in components:
        coords = component["coords"]
        bbox = component["bbox"]
        area = int(component["area"])
        r0, c0, r1, c1 = bbox
        assert r0 >= MARGIN and c0 >= MARGIN
        assert r1 <= h - 1 - MARGIN and c1 <= w - 1 - MARGIN
        assert not _rect_overlap(_expand_bbox(bbox, MARGIN), legend_zone)
        assert np.all(g[coords[:, 0], coords[:, 1]] == GREY)
        for zone in zones:
            assert not _rect_overlap(_expand_bbox(bbox, MARGIN), zone)
        zones.append(_expand_bbox(bbox, MARGIN))

        bbox_area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if area == 1:
            assert bbox_area == 1
        elif area == 2:
            assert bbox_area == 2
        elif area == 3:
            assert bbox_area == 4
        elif area == 4:
            assert bbox_area == 4

    return components


def main(grid: np.ndarray) -> np.ndarray:
    """Decorate each grey component using its pixel count as both legend index and operation selector.

    Input structure:
    - The grid has black background, a fixed 2x2 legend in the top-right corner,
      and 4 to 6 grey 4-connected foreground components.
    - The legend is blue/red on the top row and green/yellow on the bottom row.
      Every grey component has size 1, 2, 3, or 4, with each size present at least once.

    Rule:
    - Count the pixels in each grey component.
    - Use size 1/2/3/4 to read the legend at top-left/top-right/bottom-left/bottom-right.
    - Apply the size-specific decoration with that legend color:
      size 1 paints the four orthogonal neighbors, size 2 paints diagonal-only
      black neighbors, size 3 paints the component bbox outline except on the
      original grey cells, and size 4 fills the entire bbox.
    - Keep the legend unchanged.

    Key insight:
    - Component size drives two linked decisions at once: which legend color to
      use and which geometric operation branch to execute.
    """
    components = _validate_input_structure(grid)
    legend_by_size = _legend_colors(np.asarray(grid, dtype=int))
    return _apply_rule(np.asarray(grid, dtype=int), components, legend_by_size)


def _shape_variants(size: int, *, mandatory: bool = False) -> list[np.ndarray]:
    if size == 1:
        return [np.array([[1]], dtype=int)]
    if size == 2:
        return [np.array([[1, 1]], dtype=int), np.array([[1], [1]], dtype=int)]
    if size == 3:
        return [
            np.array([[0, 1], [1, 1]], dtype=int),
            np.array([[1, 0], [1, 1]], dtype=int),
            np.array([[1, 1], [0, 1]], dtype=int),
            np.array([[1, 1], [1, 0]], dtype=int),
        ]
    if size == 4:
        if mandatory:
            return [np.array([[1, 1], [1, 1]], dtype=int)]
        return [
            np.array([[1, 1], [1, 1]], dtype=int),
            np.array([[1, 1, 1, 1]], dtype=int),
            np.array([[1], [1], [1], [1]], dtype=int),
        ]
    raise ValueError(size)


def _sample_object_specs() -> list[dict]:
    count = random.choice(OBJECT_COUNT_CHOICES)
    sizes = [1, 2, 3, 4]
    for _ in range(count - 4):
        sizes.append(random.choice([1, 2, 3, 4]))
    random.shuffle(sizes)

    mandatory_four_used = False
    specs = []
    for size in sizes:
        mandatory = size == 4 and not mandatory_four_used
        shape = random.choice(_shape_variants(size, mandatory=mandatory))
        mandatory_four_used = mandatory_four_used or size == 4
        coords = np.argwhere(shape == 1)
        bbox = _bbox_from_coords(coords)
        height = bbox[2] - bbox[0] + 1
        width = bbox[3] - bbox[1] + 1
        specs.append(
            {
                "size": size,
                "shape": shape,
                "height": height,
                "width": width,
                "zone_area": (height + 2 * MARGIN) * (width + 2 * MARGIN),
            }
        )
    specs.sort(key=lambda spec: (-spec["zone_area"], -spec["size"], random.random()))
    return specs


def _candidate_positions(h: int, w: int, spec: dict) -> list[tuple[int, int]]:
    height = int(spec["height"])
    width = int(spec["width"])
    positions = [
        (r, c)
        for r in range(MARGIN, h - height - MARGIN + 1)
        for c in range(MARGIN, w - width - MARGIN + 1)
    ]
    random.shuffle(positions)
    return positions


def _place_objects(h: int, w: int, specs: list[dict]) -> list[dict] | None:
    legend_zone = _expand_bbox((0, w - 2, 1, w - 1), MARGIN)
    placed = []
    used_zones = []

    for spec in specs:
        coords_local = np.argwhere(spec["shape"] == 1)
        options = []
        for top_r, top_c in _candidate_positions(h, w, spec):
            abs_coords = coords_local + np.array([[top_r, top_c]])
            bbox = _bbox_from_coords(abs_coords)
            zone = _expand_bbox(bbox, MARGIN)
            if _rect_overlap(zone, legend_zone):
                continue
            if any(_rect_overlap(zone, other) for other in used_zones):
                continue
            options.append((abs_coords, bbox, zone))
        if not options:
            return None
        abs_coords, bbox, zone = random.choice(options)
        used_zones.append(zone)
        placed.append({"size": spec["size"], "coords": abs_coords, "bbox": bbox})
    return placed


def _build_input_grid(h: int, w: int, placed: list[dict]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    grid[:2, -2:] = LEGEND_PATCH
    for item in placed:
        coords = item["coords"]
        grid[coords[:, 0], coords[:, 1]] = GREY
    return grid


def _validate_output_semantics(inp: np.ndarray, out: np.ndarray, components: list[dict]) -> None:
    assert np.array_equal(out[:2, -2:], LEGEND_PATCH)
    legend_by_size = _legend_colors(inp)
    for component in components:
        area = int(component["area"])
        coords = component["coords"]
        bbox = component["bbox"]
        color = int(legend_by_size[area])

        if area == 1:
            r, c = map(int, coords[0])
            expected = {(r + dr, c + dc) for dr, dc in ORTHOGONALS}
            changed = {
                (rr, cc)
                for rr, cc in expected
                if int(out[rr, cc]) == color and int(inp[rr, cc]) == BLACK
            }
            assert changed == expected
            assert int(out[r, c]) == GREY
        elif area == 2:
            targets = _diagonal_only_targets(inp, component)
            assert len(targets) == 4
            for rr, cc in targets:
                assert int(out[rr, cc]) == color
            assert np.all(out[coords[:, 0], coords[:, 1]] == GREY)
        elif area == 3:
            r0, c0, r1, c1 = bbox
            outline = {
                (rr, cc)
                for rr in range(r0, r1 + 1)
                for cc in range(c0, c1 + 1)
                if rr in (r0, r1) or cc in (c0, c1)
            }
            added = outline - component["coord_set"]
            assert len(added) == 1
            only_added = next(iter(added))
            assert int(out[only_added]) == color
            assert np.all(out[coords[:, 0], coords[:, 1]] == GREY)
        elif area == 4:
            r0, c0, r1, c1 = bbox
            assert np.all(out[r0 : r1 + 1, c0 : c1 + 1] == color)


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIZE_RANGE)
    w = random.randint(*GRID_SIZE_RANGE)
    specs = _sample_object_specs()
    placed = _place_objects(h, w, specs)
    if placed is None:
        return None

    grid = _build_input_grid(h, w, placed)
    components = _validate_input_structure(grid)

    if not any(component["area"] == 4 and component["bbox"][2] - component["bbox"][0] == 1 and component["bbox"][3] - component["bbox"][1] == 1 for component in components):
        return None

    if _get_bg(grid) != BLACK:
        return None

    out = _apply_rule(grid, components, _legend_colors(grid))
    try:
        oracle = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(out, oracle):
        return None

    _validate_output_semantics(grid, out, components)
    assert not np.array_equal(grid, out)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
