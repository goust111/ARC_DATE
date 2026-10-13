from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


Coord = tuple[int, int]
ShapeKey = tuple[str, ...]


def _get_bg(grid: np.ndarray) -> int:
    counts = Counter(map(int, np.asarray(grid, dtype=int).ravel()))
    return int(counts.most_common(1)[0][0])


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size, "expected a nonempty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].astype(bool)


def _mask_key(mask: np.ndarray) -> ShapeKey:
    tight = _trim_mask(mask)
    return tuple("".join("1" if cell else "0" for cell in row) for row in tight)


def _unique_orientations(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[ShapeKey] = set()
    base = _trim_mask(mask)
    for k in range(4):
        rotated = np.rot90(base, k)
        for candidate in (rotated, np.fliplr(rotated)):
            key = _mask_key(candidate)
            if key in seen:
                continue
            seen.add(key)
            variants.append(_trim_mask(candidate))
    return variants


def _shape_signature(mask: np.ndarray) -> ShapeKey:
    return min(_mask_key(variant) for variant in _unique_orientations(mask))


def _solid_cross(grid: np.ndarray) -> tuple[int, int, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    solid_rows = [(r, int(g[r, 0])) for r in range(h) if np.all(g[r, :] == int(g[r, 0]))]
    solid_cols = [(c, int(g[0, c])) for c in range(w) if np.all(g[:, c] == int(g[0, c]))]
    matches = [
        (r, c, color_r)
        for r, color_r in solid_rows
        for c, color_c in solid_cols
        if color_r == color_c
    ]
    assert len(matches) == 1, "expected exactly one solid same-color separator cross"
    r, c, color = matches[0]
    assert 0 < r < h - 1 and 0 < c < w - 1, "separator must split four nonempty quadrants"
    return int(r), int(c), int(color)


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            coord_array = np.array(coords, dtype=int)
            r0, c0 = coord_array.min(axis=0)
            r1, c1 = coord_array.max(axis=0)
            mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
            mask[coord_array[:, 0] - r0, coord_array[:, 1] - c0] = True
            components.append(
                {
                    "color": color,
                    "coords": coord_array,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "mask": mask,
                    "area": int(coord_array.shape[0]),
                }
            )
    return components


def _quadrant_specs(height: int, width: int, cross_r: int, cross_c: int) -> list[tuple[tuple[slice, slice], tuple[int, int]]]:
    return [
        ((slice(0, cross_r), slice(0, cross_c)), (0, 0)),
        ((slice(0, cross_r), slice(cross_c + 1, width)), (0, 2)),
        ((slice(cross_r + 1, height), slice(0, cross_c)), (2, 0)),
        ((slice(cross_r + 1, height), slice(cross_c + 1, width)), (2, 2)),
    ]


def _quadrant_lonely_color(quad: np.ndarray, bg: int) -> int:
    components = _find_components(quad, bg)
    assert components and len(components) % 2 == 1, "each quadrant must contain an odd number of objects"

    groups: dict[ShapeKey, list[dict]] = defaultdict(list)
    for component in components:
        groups[_shape_signature(component["mask"])].append(component)

    singleton_groups = [members for members in groups.values() if len(members) == 1]
    assert len(singleton_groups) == 1, "each quadrant must have exactly one unpaired shape class"
    for members in groups.values():
        if len(members) == 1:
            continue
        assert len(members) % 2 == 0, "all non-selected shape classes must pair off evenly"
    return int(singleton_groups[0][0]["color"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress each quadrant's unpaired dihedral shape color into the corners of a 3x3 cross map.

    Input structure:
    - Exactly one full row and one full column share the same separator color and split the grid into four rectangular quadrants.
    - Each quadrant contains an odd number of non-touching monochromatic objects on a common background.
    - Within a quadrant, every object except one has a partner with the same tight mask up to rotation or reflection.

    Rule:
    - Detect the unique separator cross and isolate the four quadrants.
    - In each quadrant, normalize every object to its tight boolean mask and group masks by dihedral-equivalence signature.
    - Identify the only shape class that appears once in that quadrant and take that object's color.
    - Return a 3x3 grid whose middle row and middle column are the separator color and whose corners are the four selected colors in quadrant order.

    Key insight:
    - The chosen object is not marked by size, color frequency, or position; it appears only after eliminating same-shape pairs inside each quadrant under rotation/flip equivalence.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    bg = _get_bg(g)
    cross_r, cross_c, separator = _solid_cross(g)

    out = np.full((3, 3), int(separator), dtype=int)
    for (rows, cols), dst in _quadrant_specs(h, w, cross_r, cross_c):
        out[dst] = _quadrant_lonely_color(g[rows, cols], bg)
    return out


def _shape_from_rows(rows: list[str]) -> np.ndarray:
    return np.array([[cell == "1" for cell in row] for row in rows], dtype=bool)


def _build_shape_catalog() -> list[dict]:
    base_rows = [
        ["10", "11"],
        ["111"],
        ["100", "111"],
        ["111", "010"],
        ["011", "110"],
        ["110", "111"],
        ["101", "111"],
        ["100", "100", "111"],
        ["100", "110", "011"],
        ["010", "111", "010"],
        ["1110", "0011"],
    ]
    catalog: list[dict] = []
    seen: set[ShapeKey] = set()
    for rows in base_rows:
        mask = _shape_from_rows(rows)
        signature = _shape_signature(mask)
        if signature in seen:
            continue
        orientations = _unique_orientations(mask)
        if len(orientations) < 2:
            continue
        seen.add(signature)
        catalog.append(
            {
                "signature": signature,
                "area": int(mask.sum()),
                "orientations": orientations,
            }
        )
    assert len(catalog) >= 8, "expected a sufficiently rich shape catalog"
    return catalog


SHAPES = _build_shape_catalog()
SHAPES_BY_AREA: dict[int, list[dict]] = defaultdict(list)
for _shape in SHAPES:
    SHAPES_BY_AREA[int(_shape["area"])].append(_shape)


BG = 0
PAIR_COUNT_PATTERNS = ([1, 1, 2, 2], [1, 1, 1, 2], [1, 2, 2, 2])
QUADRANT_SIZE_RANGE = (10, 14)
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_RETRIES = 80


def _sample_pair_counts() -> list[int]:
    counts = list(random.choice(PAIR_COUNT_PATTERNS))
    random.shuffle(counts)
    return counts


def _choose_shapes_for_quadrant(lone_shape: dict, pair_count: int) -> list[dict]:
    same_area = [
        shape
        for shape in SHAPES_BY_AREA[int(lone_shape["area"])]
        if shape["signature"] != lone_shape["signature"]
    ]
    assert same_area, "need at least one different shape with the same area"
    chosen = [random.choice(same_area)]

    available = [
        shape
        for shape in SHAPES
        if shape["signature"] not in {lone_shape["signature"], chosen[0]["signature"]}
    ]
    while len(chosen) < pair_count:
        candidate = random.choice(available)
        if candidate["signature"] in {shape["signature"] for shape in chosen}:
            continue
        chosen.append(candidate)
    return chosen


def _random_orientation(shape: dict) -> np.ndarray:
    return random.choice(shape["orientations"]).copy()


def _make_quadrant_objects(lone_shape: dict, lone_color: int, pair_shapes: list[dict], palette: list[int]) -> list[dict]:
    objects = [
        {
            "mask": _random_orientation(lone_shape),
            "color": int(lone_color),
            "signature": lone_shape["signature"],
            "area": int(lone_shape["area"]),
            "is_lone": True,
        }
    ]
    for shape in pair_shapes:
        idx_a, idx_b = random.sample(range(len(shape["orientations"])), 2)
        color_a, color_b = random.sample(palette, 2)
        objects.append(
            {
                "mask": shape["orientations"][idx_a].copy(),
                "color": int(color_a),
                "signature": shape["signature"],
                "area": int(shape["area"]),
                "is_lone": False,
            }
        )
        objects.append(
            {
                "mask": shape["orientations"][idx_b].copy(),
                "color": int(color_b),
                "signature": shape["signature"],
                "area": int(shape["area"]),
                "is_lone": False,
            }
        )
    random.shuffle(objects)
    return objects


def _place_quadrant(qh: int, qw: int, objects: list[dict]) -> tuple[np.ndarray, list[dict]] | None:
    canvas = np.full((qh, qw), BG, dtype=int)
    blocked = np.zeros((qh, qw), dtype=bool)
    placements: list[dict | None] = [None] * len(objects)
    order = sorted(range(len(objects)), key=lambda idx: (-objects[idx]["area"], objects[idx]["mask"].shape[0] * objects[idx]["mask"].shape[1]))

    for idx in order:
        obj = objects[idx]
        mask = obj["mask"]
        oh, ow = mask.shape
        if qh - oh < 2 or qw - ow < 2:
            return None

        positions = [(r, c) for r in range(1, qh - oh) for c in range(1, qw - ow)]
        random.shuffle(positions)
        placed = False
        for r0, c0 in positions[:PLACEMENT_RETRIES * 4]:
            coords = np.argwhere(mask) + np.array([r0, c0], dtype=int)
            if any(blocked[int(r), int(c)] for r, c in coords):
                continue

            canvas_slice = canvas[r0 : r0 + oh, c0 : c0 + ow]
            if np.any(canvas_slice[mask] != BG):
                continue

            canvas_slice[mask] = int(obj["color"])
            for r, c in coords:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = int(r + dr), int(c + dc)
                        if 0 <= nr < qh and 0 <= nc < qw:
                            blocked[nr, nc] = True
            placements[idx] = {
                **obj,
                "top": int(r0),
                "left": int(c0),
                "bbox": (int(r0), int(c0), int(r0 + oh - 1), int(c0 + ow - 1)),
            }
            placed = True
            break
        if not placed:
            return None

    return canvas, [item for item in placements if item is not None]


def _assemble_candidate() -> tuple[np.ndarray, np.ndarray, list[dict]] | None:
    top_h = random.randint(*QUADRANT_SIZE_RANGE)
    bottom_h = random.randint(*QUADRANT_SIZE_RANGE)
    left_w = random.randint(*QUADRANT_SIZE_RANGE)
    right_w = random.randint(*QUADRANT_SIZE_RANGE)
    height = top_h + 1 + bottom_h
    width = left_w + 1 + right_w
    if height > 30 or width > 30:
        return None

    separator = random.choice(list(range(1, 10)))
    object_palette = [color for color in range(1, 10) if color != separator]
    lone_colors = random.sample(object_palette, 4)

    lone_shapes = random.sample(SHAPES, 4)
    pair_counts = _sample_pair_counts()

    quadrants = [
        (top_h, left_w),
        (top_h, right_w),
        (bottom_h, left_w),
        (bottom_h, right_w),
    ]
    quadrant_infos: list[dict] = []
    quadrant_grids: list[np.ndarray] = []
    for idx, ((qh, qw), lone_shape, lone_color, pair_count) in enumerate(
        zip(quadrants, lone_shapes, lone_colors, pair_counts)
    ):
        pair_shapes = _choose_shapes_for_quadrant(lone_shape, pair_count)
        objects = _make_quadrant_objects(lone_shape, lone_color, pair_shapes, object_palette)
        placed = _place_quadrant(int(qh), int(qw), objects)
        if placed is None:
            return None
        quad_grid, placed_objects = placed
        quadrant_grids.append(quad_grid)
        quadrant_infos.append(
            {
                "index": idx,
                "shape": (int(qh), int(qw)),
                "lone_color": int(lone_color),
                "lone_signature": lone_shape["signature"],
                "objects": placed_objects,
            }
        )

    grid = np.full((height, width), BG, dtype=int)
    cross_r = top_h
    cross_c = left_w
    grid[cross_r, :] = separator
    grid[:, cross_c] = separator
    grid[0:top_h, 0:left_w] = quadrant_grids[0]
    grid[0:top_h, cross_c + 1 : width] = quadrant_grids[1]
    grid[cross_r + 1 : height, 0:left_w] = quadrant_grids[2]
    grid[cross_r + 1 : height, cross_c + 1 : width] = quadrant_grids[3]

    expected = np.array(
        [
            [lone_colors[0], separator, lone_colors[1]],
            [separator, separator, separator],
            [lone_colors[2], separator, lone_colors[3]],
        ],
        dtype=int,
    )
    return grid, expected, quadrant_infos


def _check_lone_not_extreme(components: list[dict], lone_signature: ShapeKey) -> None:
    areas = [int(component["area"]) for component in components]
    lone_area = next(int(component["area"]) for component in components if _shape_signature(component["mask"]) == lone_signature)
    min_area = min(areas)
    max_area = max(areas)
    if lone_area == min_area:
        assert areas.count(min_area) > 1, "lone object should not be the unique smallest object"
    if lone_area == max_area:
        assert areas.count(max_area) > 1, "lone object should not be the unique largest object"


def _validate_candidate(grid: np.ndarray, expected: np.ndarray, quadrant_infos: list[dict]) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    bg = _get_bg(g)
    assert bg == BG, "generator assumes black background"

    cross_r, cross_c, separator = _solid_cross(g)
    assert np.array_equal(main(g), expected)

    observed_lone_signatures: list[ShapeKey] = []
    for info, ((rows, cols), _dst) in zip(quadrant_infos, _quadrant_specs(g.shape[0], g.shape[1], cross_r, cross_c)):
        quad = g[rows, cols]
        components = _find_components(quad, bg)
        assert len(components) == len(info["objects"]), "object count changed after rendering"
        groups: dict[ShapeKey, list[dict]] = defaultdict(list)
        for component in components:
            groups[_shape_signature(component["mask"])].append(component)

        lone_groups = [group for group in groups.values() if len(group) == 1]
        assert len(lone_groups) == 1
        lone_component = lone_groups[0][0]
        lone_signature = _shape_signature(lone_component["mask"])
        observed_lone_signatures.append(lone_signature)
        assert int(lone_component["color"]) == int(info["lone_color"])
        assert lone_signature == info["lone_signature"]

        _check_lone_not_extreme(components, lone_signature)

        for signature, group in groups.items():
            if len(group) == 1:
                continue
            assert len(group) == 2, "generator should create one explicit pair per repeated shape"
            assert len({int(member["color"]) for member in group}) == 2, "paired objects must use different colors"
            assert len({_mask_key(member["mask"]) for member in group}) == 2, "paired objects must use different orientations"
            assert signature != lone_signature

    assert len(set(observed_lone_signatures)) == 4, "quadrant leftovers must be shape-distinct across quadrants"
    assert int(separator) == int(expected[1, 1])


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        candidate = _assemble_candidate()
        if candidate is None:
            continue
        grid, expected, quadrant_infos = candidate
        try:
            _validate_candidate(grid, expected, quadrant_infos)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("generate_input: failed to build a valid quadrant-pairing puzzle")
