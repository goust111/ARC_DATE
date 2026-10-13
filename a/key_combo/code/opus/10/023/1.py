from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

DEPTH_TO_COLOR = {1: BLUE, 2: RED, 3: GREEN}
DIHEDRAL_TRANSFORMS = tuple(range(8))


def _normalize(coords: list[tuple[int, int]] | np.ndarray) -> tuple[tuple[int, int], ...]:
    pts = [(int(r), int(c)) for r, c in coords]
    min_r = min(r for r, _ in pts)
    min_c = min(c for _, c in pts)
    return tuple(sorted((r - min_r, c - min_c) for r, c in pts))


def _transform(coords: list[tuple[int, int]] | np.ndarray, variant: int) -> tuple[tuple[int, int], ...]:
    pts = [(int(r), int(c)) for r, c in coords]
    transformed = []
    for r, c in pts:
        if variant == 0:
            rr, cc = r, c
        elif variant == 1:
            rr, cc = r, -c
        elif variant == 2:
            rr, cc = -r, c
        elif variant == 3:
            rr, cc = -r, -c
        elif variant == 4:
            rr, cc = c, r
        elif variant == 5:
            rr, cc = c, -r
        elif variant == 6:
            rr, cc = -c, r
        else:
            rr, cc = -c, -r
        transformed.append((rr, cc))
    return _normalize(transformed)


def _canonical_shape(coords: list[tuple[int, int]] | np.ndarray) -> tuple[tuple[int, int], ...]:
    return min(_transform(coords, variant) for variant in DIHEDRAL_TRANSFORMS)


def _bbox(coords: list[tuple[int, int]] | np.ndarray) -> tuple[int, int, int, int]:
    pts = [(int(r), int(c)) for r, c in coords]
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(np.asarray(coords, dtype=int))
    return components


def _grey_components(grid: np.ndarray) -> list[np.ndarray]:
    return _components_from_mask(np.asarray(grid == GREY, dtype=bool))


def _component_depth(coords: list[tuple[int, int]] | np.ndarray) -> int:
    r0, c0, r1, c1 = _bbox(coords)
    depth = 0
    for r, c in coords:
        layer = 1 + min(int(r) - r0, r1 - int(r), int(c) - c0, c1 - int(c))
        depth = max(depth, layer)
    return int(depth)


def _find_divider(grid: np.ndarray) -> tuple[np.ndarray, int, int]:
    candidates: list[tuple[np.ndarray, int, int]] = []
    for coords in _grey_components(grid):
        pts = {(int(r), int(c)) for r, c in coords}
        if not any(r == 0 for r, _ in pts):
            continue
        if not any(c == 0 for _, c in pts):
            continue
        divider_r = max(r for r, _ in pts)
        divider_c = max(c for _, c in pts)
        expected = {(r, divider_c) for r in range(divider_r + 1)} | {
            (divider_r, c) for c in range(divider_c + 1)
        }
        if pts == expected:
            candidates.append((coords, divider_r, divider_c))
    assert len(candidates) == 1, "expected exactly one grey L-shaped divider"
    return candidates[0]


def _extract_templates(
    grid: np.ndarray, divider_r: int, divider_c: int
) -> list[dict[str, object]]:
    legend_mask = np.zeros_like(grid, dtype=bool)
    legend_mask[:divider_r, :divider_c] = grid[:divider_r, :divider_c] != BG
    components = _components_from_mask(legend_mask)
    templates: list[dict[str, object]] = []
    seen_shapes: set[tuple[tuple[int, int], ...]] = set()
    for coords in components:
        depth = _component_depth(coords)
        assert depth in DEPTH_TO_COLOR, "template depth must be 1, 2, or 3"
        r0, c0, r1, c1 = _bbox(coords)
        for r, c in coords:
            layer = 1 + min(int(r) - r0, r1 - int(r), int(c) - c0, c1 - int(c))
            assert int(grid[r, c]) == DEPTH_TO_COLOR[layer], "template colors must encode bbox depth"
        shape = _canonical_shape(coords)
        assert shape not in seen_shapes, "legend shapes must be distinct up to rotation/reflection"
        seen_shapes.add(shape)
        templates.append(
            {
                "coords": coords,
                "shape": shape,
                "depth": depth,
                "bbox": (r0, c0, r1, c1),
                "bbox_span": tuple(sorted((r1 - r0 + 1, c1 - c0 + 1))),
                "area": int(len(coords)),
                "area_bbox": (int(len(coords)), tuple(sorted((r1 - r0 + 1, c1 - c0 + 1)))),
            }
        )
    assert 3 <= len(templates) <= 4, "expected 3-4 legend templates"
    assert {int(t["depth"]) for t in templates} >= {1, 2, 3}, "legend must cover depths 1, 2, and 3"
    return templates


def _extract_queries(
    grid: np.ndarray, divider_coords: np.ndarray
) -> list[dict[str, object]]:
    grey_mask = np.asarray(grid == GREY, dtype=bool)
    grey_mask[divider_coords[:, 0], divider_coords[:, 1]] = False
    components = _components_from_mask(grey_mask)
    queries = []
    for coords in components:
        bbox = _bbox(coords)
        queries.append(
            {
                "coords": coords,
                "shape": _canonical_shape(coords),
                "bbox": bbox,
                "bbox_span": tuple(sorted((bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1))),
                "area": int(len(coords)),
                "area_bbox": (int(len(coords)), tuple(sorted((bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1)))),
                "depth": _component_depth(coords),
            }
        )
    assert 3 <= len(queries) <= 5, "expected 3-5 query sprites"
    return queries


def _scene(grid: np.ndarray) -> dict[str, object]:
    divider_coords, divider_r, divider_c = _find_divider(grid)
    templates = _extract_templates(grid, divider_r, divider_c)
    queries = _extract_queries(grid, divider_coords)
    return {
        "divider_coords": divider_coords,
        "divider_r": divider_r,
        "divider_c": divider_c,
        "templates": templates,
        "queries": queries,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each grey query sprite by matching its shape to a depth-encoded legend template.

    Input structure:
    A black background contains a single-pixel grey L divider that touches the top and left edges and isolates
    an upper-left legend rectangle. The legend holds 3-4 multicolor 4-connected template sprites whose outer
    layer is blue, next layer red, and third layer green. Elsewhere, 3-5 grey query sprites appear as
    rotation/reflection copies of the legend shapes.

    Rule:
    Find the legend template whose shape matches each grey query sprite up to rotation/reflection. For the
    matched template, compute the maximum bbox-edge depth among its cells: outer layer depth 1, next layer
    depth 2, next layer depth 3. Recolor the entire grey query sprite to blue for depth 1, red for depth 2,
    and green for depth 3. Keep the divider, legend, and background unchanged.

    Key insight:
    The legend is not a direct color dictionary: shape identity selects a template, and that template's hidden
    concentric depth encodes the output color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    info = _scene(g)

    depth_by_shape = {template["shape"]: int(template["depth"]) for template in info["templates"]}
    out = g.copy()
    for query in info["queries"]:
        shape = query["shape"]
        assert shape in depth_by_shape, "every query must match a legend template"
        out_color = DEPTH_TO_COLOR[depth_by_shape[shape]]
        coords = query["coords"]
        out[coords[:, 0], coords[:, 1]] = out_color
    return out


# --------------- generator ---------------

GRID_MIN = 21
GRID_MAX = 27
DIVIDER_MIN = 13
DIVIDER_MAX = 15
QUERY_COUNT_MAX = 5
QUERY_GAP = 2
GEN_ATTEMPTS = 400

SHAPE_LIBRARY = [
    {
        "name": "d1_l5",
        "coords": ((0, 0), (0, 1), (0, 2), (1, 0), (2, 0)),
        "depth": 1,
    },
    {
        "name": "d1_c7",
        "coords": ((0, 0), (0, 1), (0, 2), (1, 0), (2, 0), (2, 1), (2, 2)),
        "depth": 1,
    },
    {
        "name": "d1_u5",
        "coords": ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2)),
        "depth": 1,
    },
    {
        "name": "d2_plus5",
        "coords": ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
        "depth": 2,
    },
    {
        "name": "d2_t7",
        "coords": ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 1)),
        "depth": 2,
    },
    {
        "name": "d2_p6",
        "coords": ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 1)),
        "depth": 2,
    },
    {
        "name": "d3_cross9",
        "coords": (
            (0, 2),
            (1, 2),
            (2, 0),
            (2, 1),
            (2, 2),
            (2, 3),
            (2, 4),
            (3, 2),
            (4, 2),
        ),
        "depth": 3,
    },
]


def _shape_patch(coords: tuple[tuple[int, int], ...], depth_colors: bool) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    patch = np.full((h, w), BG, dtype=int)
    for r, c in coords:
        rr = r - r0
        cc = c - c0
        if depth_colors:
            layer = 1 + min(rr, h - 1 - rr, cc, w - 1 - cc)
            patch[rr, cc] = DEPTH_TO_COLOR[layer]
        else:
            patch[rr, cc] = GREY
    return patch


def _shape_variants(shape: dict[str, object]) -> list[tuple[tuple[int, int], ...]]:
    coords = shape["coords"]
    return sorted({_transform(coords, variant) for variant in DIHEDRAL_TRANSFORMS})


def _slot_origins(divider_r: int, divider_c: int) -> list[tuple[int, int]]:
    return [
        (1, 1),
        (1, divider_c - 6),
        (divider_r - 6, 1),
        (divider_r - 6, divider_c - 6),
    ]


def _draw_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    region = grid[top : top + h, left : left + w]
    mask = patch != BG
    assert not np.any(region[mask] != BG)
    region[mask] = patch[mask]


def _bbox_gap_ok(
    bbox: tuple[int, int, int, int], others: list[tuple[int, int, int, int]], gap: int
) -> bool:
    r0, c0, r1, c1 = bbox
    for or0, oc0, or1, oc1 in others:
        if r1 < or0:
            dr = or0 - r1 - 1
        elif or1 < r0:
            dr = r0 - or1 - 1
        else:
            dr = 0
        if c1 < oc0:
            dc = oc0 - c1 - 1
        elif oc1 < c0:
            dc = c0 - oc1 - 1
        else:
            dc = 0
        if max(dr, dc) < gap:
            return False
    return True


def _query_regions(divider_r: int, divider_c: int, height: int, width: int) -> list[tuple[int, int, int, int]]:
    regions = [
        (1, divider_r - 2, divider_c + 2, width - 2),
        (divider_r + 2, height - 2, 1, divider_c - 2),
        (divider_r + 2, height - 2, divider_c + 2, width - 2),
    ]
    return [region for region in regions if region[0] <= region[1] and region[2] <= region[3]]


def _pick_template_set() -> list[dict[str, object]]:
    pair_options = [
        ("d1_l5", "d2_plus5"),
        ("d1_c7", "d2_t7"),
    ]
    pair = random.choice(pair_options)
    by_name = {shape["name"]: shape for shape in SHAPE_LIBRARY}
    chosen = [by_name[pair[0]], by_name[pair[1]], by_name["d3_cross9"]]

    extras = [
        by_name["d1_u5"],
        by_name["d2_p6"],
    ]
    if random.random() < 0.55:
        chosen.append(random.choice(extras))
    assert len({_canonical_shape(shape["coords"]) for shape in chosen}) == len(chosen)
    return chosen


def _place_templates(
    grid: np.ndarray, divider_r: int, divider_c: int, templates: list[dict[str, object]]
) -> list[dict[str, object]]:
    slots = _slot_origins(divider_r, divider_c)
    random.shuffle(slots)
    placed = []
    for shape, (base_r, base_c) in zip(templates, slots):
        coords = random.choice(_shape_variants(shape))
        patch = _shape_patch(coords, depth_colors=True)
        h, w = patch.shape
        max_dr = 5 - h
        max_dc = 5 - w
        top = base_r + random.randint(0, max_dr)
        left = base_c + random.randint(0, max_dc)
        _draw_patch(grid, patch, top, left)
        abs_coords = np.argwhere(grid[top : top + h, left : left + w] != BG)
        abs_coords[:, 0] += top
        abs_coords[:, 1] += left
        placed.append(
            {
                "name": shape["name"],
                "depth": shape["depth"],
                "shape": _canonical_shape(abs_coords),
                "coords": abs_coords,
                "bbox": _bbox(abs_coords),
                "area": int(len(abs_coords)),
            }
        )
    return placed


def _place_queries(
    grid: np.ndarray,
    divider_r: int,
    divider_c: int,
    templates: list[dict[str, object]],
) -> bool:
    regions = _query_regions(divider_r, divider_c, *grid.shape)
    query_shapes = templates.copy()
    target_count = random.randint(len(templates), QUERY_COUNT_MAX)
    while len(query_shapes) < target_count:
        query_shapes.append(random.choice(templates))
    random.shuffle(query_shapes)

    existing_bboxes: list[tuple[int, int, int, int]] = []
    for template in query_shapes:
        coords = random.choice(_shape_variants({"coords": _normalize(template["coords"])}))
        patch = _shape_patch(coords, depth_colors=False)
        h, w = patch.shape
        candidates = []
        random_regions = regions[:]
        random.shuffle(random_regions)
        for r0, r1, c0, c1 in random_regions:
            for top in range(r0, r1 - h + 2):
                for left in range(c0, c1 - w + 2):
                    bbox = (top, left, top + h - 1, left + w - 1)
                    if not _bbox_gap_ok(bbox, existing_bboxes, QUERY_GAP):
                        continue
                    region = grid[top : top + h, left : left + w]
                    if np.any(region[patch != BG] != BG):
                        continue
                    candidates.append((top, left))
        if not candidates:
            return False
        top, left = random.choice(candidates)
        _draw_patch(grid, patch, top, left)
        existing_bboxes.append((top, left, top + h - 1, left + w - 1))
    return True


def _draw_divider(grid: np.ndarray, divider_r: int, divider_c: int) -> None:
    grid[: divider_r + 1, divider_c] = GREY
    grid[divider_r, : divider_c + 1] = GREY


def _shortcut_works(templates: list[dict[str, object]], queries: list[dict[str, object]], key: str) -> bool:
    table: dict[object, set[int]] = defaultdict(set)
    for template in templates:
        table[template[key]].add(DEPTH_TO_COLOR[int(template["depth"])])
    lookup = {signature: next(iter(colors)) for signature, colors in table.items() if len(colors) == 1}
    if len(lookup) != len(table):
        return False
    return all(lookup.get(query[key]) == DEPTH_TO_COLOR[int(query["depth"])] for query in queries)


def _validate_semantics(grid: np.ndarray) -> None:
    info = _scene(grid)
    templates = info["templates"]
    queries = info["queries"]

    assert {int(t["depth"]) for t in templates} >= {1, 2, 3}
    assert len({t["shape"] for t in templates}) == len(templates)
    assert len({DEPTH_TO_COLOR[int(t["depth"])] for t in templates}) == 3
    assert not _shortcut_works(templates, queries, "area"), "generator must defeat area-only shortcut"
    assert not _shortcut_works(templates, queries, "bbox_span"), "generator must defeat bbox-only shortcut"
    assert not _shortcut_works(
        templates,
        queries,
        "area_bbox",
    ), "generator must defeat area+bbox shortcut"

    # The puzzle statement makes query bbox-depth equal to template bbox-depth under shape equivalence.
    # Validation therefore rejects simpler surrogates that are not part of the stated two-step mechanism.
    out = main(grid)
    for query in queries:
        coords = query["coords"]
        colors = {int(out[r, c]) for r, c in coords}
        assert len(colors) == 1
        matched = [template for template in templates if template["shape"] == query["shape"]]
        assert len(matched) == 1
        assert colors == {DEPTH_TO_COLOR[int(matched[0]["depth"])]}


def _generate_one() -> np.ndarray | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    divider_r = random.randint(DIVIDER_MIN, DIVIDER_MAX)
    divider_c = random.randint(DIVIDER_MIN, DIVIDER_MAX)
    if divider_r >= height - 6 or divider_c >= width - 6:
        return None

    grid = np.full((height, width), BG, dtype=int)
    templates = _pick_template_set()
    _draw_divider(grid, divider_r, divider_c)
    placed_templates = _place_templates(grid, divider_r, divider_c, templates)
    if not _place_queries(grid, divider_r, divider_c, placed_templates):
        return None

    try:
        _validate_semantics(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_one()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
