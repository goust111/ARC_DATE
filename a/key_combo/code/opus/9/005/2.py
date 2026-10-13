from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components, get_bg, paste


BG = Color.ORANGE
DIVIDER = Color.GREY
QUERY = Color.TEAL
LEFT = Color.RED
RIGHT = Color.BLUE
DIHEDRAL = (
    (1, 0, 0, 1),
    (1, 0, 0, -1),
    (-1, 0, 0, 1),
    (-1, 0, 0, -1),
    (0, 1, 1, 0),
    (0, 1, -1, 0),
    (0, -1, 1, 0),
    (0, -1, -1, 0),
)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_ATTEMPTS = 500


def _normalize(coords: list[tuple[int, int]] | np.ndarray) -> tuple[tuple[int, int], ...]:
    pts = [(int(r), int(c)) for r, c in coords]
    min_r = min(r for r, _ in pts)
    min_c = min(c for _, c in pts)
    return tuple(sorted((r - min_r, c - min_c) for r, c in pts))


def _canonical_shape(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    pts = [(int(r), int(c)) for r, c in coords]
    variants = []
    for rr_r, rr_c, cc_r, cc_c in DIHEDRAL:
        transformed = []
        for r, c in pts:
            transformed.append((rr_r * r + rr_c * c, cc_r * r + cc_c * c))
        variants.append(_normalize(transformed))
    return min(variants)


def _apply_variant(
    shape: tuple[tuple[int, int], ...], variant: tuple[int, int, int, int]
) -> tuple[tuple[int, int], ...]:
    rr_r, rr_c, cc_r, cc_c = variant
    return _normalize([(rr_r * r + rr_c * c, cc_r * r + cc_c * c) for r, c in shape])


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _shape_patch(shape: tuple[tuple[int, int], ...], color: int) -> np.ndarray:
    h, w = _shape_dims(shape)
    patch = np.full((h, w), BG, dtype=int)
    for r, c in shape:
        patch[r, c] = int(color)
    return patch


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _row_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0])


def _paint_coords(grid: np.ndarray, coords: np.ndarray, color: int) -> None:
    grid[coords[:, 0], coords[:, 1]] = int(color)


def _translate(coords: np.ndarray, dr: int, dc: int) -> np.ndarray:
    out = coords.copy()
    out[:, 0] += int(dr)
    out[:, 1] += int(dc)
    return out


def _parse_input(grid: np.ndarray) -> tuple[int, dict[tuple[tuple[int, int], ...], int], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)
    assert bg == BG, "expected orange background"

    divider_rows = [r for r in range(g.shape[0]) if np.all(g[r] == DIVIDER)]
    assert len(divider_rows) == 1, "expected one full-width horizontal divider"
    divider_r = int(divider_rows[0])
    assert 2 <= divider_r < g.shape[0] - 3, "divider must split legend and query regions"

    legend = g[:divider_r, :]
    queries = g[divider_r + 1 :, :]

    template_by_shape: dict[tuple[tuple[int, int], ...], int] = {}
    for comp in find_components(legend, bg=bg, connectivity=4, monochromatic=True, colors=[LEFT, RIGHT]):
        key = _canonical_shape(comp["coords"])
        assert key not in template_by_shape, "legend shapes must be unique up to rotation/reflection"
        template_by_shape[key] = int(comp["color"])
    assert 2 <= len(template_by_shape) <= 4, "legend must contain 2-4 templates"

    query_objects = []
    for comp in find_components(queries, bg=bg, connectivity=4, monochromatic=True, colors=[QUERY]):
        coords = comp["coords"].copy()
        coords[:, 0] += divider_r + 1
        key = _canonical_shape(coords)
        assert key in template_by_shape, "query shape must match exactly one legend shape"
        query_objects.append(
            {
                "shape": key,
                "coords": coords,
                "bbox": _bbox(coords),
                "target_color": template_by_shape[key],
            }
        )
    assert len(query_objects) == len(template_by_shape), "query count must equal template count"
    assert len({obj["shape"] for obj in query_objects}) == len(query_objects), "query shapes must be one-to-one"
    return divider_r, template_by_shape, query_objects


def _place_sliding_queries(
    query_objects: list[dict], width: int
) -> list[dict]:
    placed: dict[int, list[dict]] = {LEFT: [], RIGHT: []}
    result: list[dict] = []

    for color in (LEFT, RIGHT):
        group = [obj.copy() for obj in query_objects if int(obj["target_color"]) == color]
        for obj in group:
            r0, c0, r1, c1 = obj["bbox"]
            obj["distance"] = c0 if color == LEFT else (width - 1 - c1)
        group.sort(key=lambda obj: (int(obj["distance"]), int(obj["bbox"][0])))

        for obj in group:
            overlaps = [prev for prev in placed[color] if _row_overlap(obj["bbox"], prev["new_bbox"])]
            r0, c0, r1, c1 = obj["bbox"]
            if color == LEFT:
                new_c0 = max([0] + [int(prev["new_bbox"][3]) + 1 for prev in overlaps])
                dc = new_c0 - c0
            else:
                new_c1 = min([width - 1] + [int(prev["new_bbox"][1]) - 1 for prev in overlaps])
                dc = new_c1 - c1
            new_coords = _translate(obj["coords"], 0, dc)
            new_bbox = _bbox(new_coords)
            assert 0 <= new_bbox[1] <= new_bbox[3] < width, "slid query left grid bounds"
            obj["new_coords"] = new_coords
            obj["new_bbox"] = new_bbox
            placed[color].append(obj)
            result.append(obj)

    return result


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each lower query sprite from the upper legend, then slide it horizontally to the matching edge.

    Input structure:
    - The grid has an orange background and one full-width gray horizontal divider.
    - Above the divider sit 2-4 red/blue legend sprites, each a small 4-connected monochrome shape.
    - Below the divider sit the same number of teal query sprites; each query matches exactly one legend shape up to rotation/reflection.

    Rule:
    - Match every teal query sprite to its unique legend template under all rotations and reflections.
    - Recolor the query to the matched legend color.
    - Red queries slide left until their bbox touches column 0; blue queries slide right until their bbox touches the last column.
    - If same-color queries have overlapping row spans, process the closer-to-edge one first and pack later ones immediately behind the overlapping predecessors.

    Key insight:
    - Shape matching does double duty: it determines the visible recolor and also the movement direction, so collision resolution only makes sense after the legend lookup.
    """
    g = np.asarray(grid, dtype=int)
    divider_r, _template_by_shape, query_objects = _parse_input(g)
    out = g.copy()
    for obj in query_objects:
        _paint_coords(out, obj["coords"], BG)

    placements = _place_sliding_queries(query_objects, g.shape[1])
    for obj in placements:
        coords = obj["new_coords"]
        assert np.all(out[coords[:, 0], coords[:, 1]] == BG), "placed queries must not overlap"
        _paint_coords(out, coords, int(obj["target_color"]))

    assert np.array_equal(out[: divider_r + 1], g[: divider_r + 1]), "legend and divider must stay unchanged"
    return out


def _generate_shape_catalog() -> list[tuple[tuple[int, int], ...]]:
    by_size: dict[int, set[tuple[tuple[int, int], ...]]] = {1: {((0, 0),)}}
    for size in range(2, 7):
        shapes = set()
        for shape in by_size[size - 1]:
            cells = set(shape)
            for r, c in shape:
                for dr, dc in NEIGHBORS4:
                    nxt = (r + dr, c + dc)
                    if nxt not in cells:
                        shapes.add(_normalize(list(cells | {nxt})))
        by_size[size] = shapes

    catalog = []
    seen = set()
    for size in range(3, 7):
        for shape in by_size[size]:
            key = _canonical_shape(np.array(shape, dtype=int))
            if key in seen:
                continue
            h, w = _shape_dims(key)
            if h <= 4 and w <= 4:
                seen.add(key)
                catalog.append(key)
    catalog.sort(key=lambda shape: (len(shape), _shape_dims(shape), shape))
    return catalog


SHAPE_CATALOG = _generate_shape_catalog()


def _can_place_cells(
    occupied: np.ndarray,
    cells: tuple[tuple[int, int], ...],
    r0: int,
    c0: int,
    *,
    forbid_edge: bool,
) -> bool:
    h, w = occupied.shape
    pts = []
    for dr, dc in cells:
        r = int(r0 + dr)
        c = int(c0 + dc)
        if not (0 <= r < h and 0 <= c < w):
            return False
        if forbid_edge and (r == 0 or r == h - 1 or c == 0 or c == w - 1):
            return False
        pts.append((r, c))
    for r, c in pts:
        r0n = max(0, r - 1)
        r1n = min(h, r + 2)
        c0n = max(0, c - 1)
        c1n = min(w, c + 2)
        if bool(occupied[r0n:r1n, c0n:c1n].any()):
            return False
    return True


def _stamp_cells(occupied: np.ndarray, cells: tuple[tuple[int, int], ...], r0: int, c0: int) -> None:
    for dr, dc in cells:
        occupied[int(r0 + dr), int(c0 + dc)] = True


def _all_positions(
    occupied: np.ndarray,
    shape: tuple[tuple[int, int], ...],
    row_lo: int,
    row_hi: int,
    col_lo: int,
    col_hi: int,
    *,
    forbid_edge: bool,
) -> list[tuple[int, int]]:
    h, w = _shape_dims(shape)
    positions = []
    for r0 in range(int(row_lo), int(row_hi) - h + 2):
        for c0 in range(int(col_lo), int(col_hi) - w + 2):
            if _can_place_cells(occupied, shape, r0, c0, forbid_edge=forbid_edge):
                positions.append((r0, c0))
    return positions


def _pick_template_colors(count: int) -> list[int]:
    repeated = random.choice([LEFT, RIGHT])
    other = RIGHT if repeated == LEFT else LEFT
    if count == 2:
        colors = [repeated, repeated]
    else:
        colors = [repeated, repeated, other]
        while len(colors) < count:
            colors.append(random.choice([repeated, other]))
    random.shuffle(colors)
    return colors


def _choose_query_variants(shapes: list[tuple[tuple[int, int], ...]]) -> list[tuple[tuple[int, int], ...]]:
    variants = []
    for shape in shapes:
        transformed = [_apply_variant(shape, variant) for variant in DIHEDRAL]
        variants.append(random.choice(transformed))
    return variants


def _place_conflict_pair(
    occupied: np.ndarray,
    query_shapes: list[tuple[tuple[int, int], ...]],
    pair: tuple[int, int],
    pair_color: int,
    divider_r: int,
) -> dict[int, tuple[int, int]] | None:
    h, w = occupied.shape
    row_lo = divider_r + 2
    row_hi = h - 2
    idx_a, idx_b = pair
    shape_a = query_shapes[idx_a]
    shape_b = query_shapes[idx_b]
    ha, wa = _shape_dims(shape_a)
    hb, wb = _shape_dims(shape_b)

    for _ in range(200):
        top_a = random.randint(row_lo, row_hi - ha + 1)
        top_b_lo = max(row_lo, top_a - hb + 1)
        top_b_hi = min(row_hi - hb + 1, top_a + ha - 1)
        if top_b_lo > top_b_hi:
            continue
        top_b = random.randint(top_b_lo, top_b_hi)

        if pair_color == LEFT:
            near_c = random.randint(1, max(1, min(4, w - wa - 2)))
            far_lo = near_c + wa + 2
            far_hi = w - wb - 2
            if far_lo > far_hi:
                continue
            far_c = random.randint(far_lo, far_hi)
            pos = {idx_a: (top_a, near_c), idx_b: (top_b, far_c)}
            if random.random() < 0.5:
                pos = {idx_a: (top_a, far_c), idx_b: (top_b, near_c)}
        else:
            near_c = w - wa - 1 - random.randint(0, min(3, w - wa - 2))
            far_limit = near_c - wb - 2
            if far_limit < 1:
                continue
            far_c = random.randint(1, far_limit)
            pos = {idx_a: (top_a, near_c), idx_b: (top_b, far_c)}
            if random.random() < 0.5:
                pos = {idx_a: (top_a, far_c), idx_b: (top_b, near_c)}

        ok = True
        for idx, (r0, c0) in pos.items():
            if not _can_place_cells(occupied, query_shapes[idx], r0, c0, forbid_edge=True):
                ok = False
                break
        if not ok:
            continue
        for idx, (r0, c0) in pos.items():
            _stamp_cells(occupied, query_shapes[idx], r0, c0)
        return pos

    return None


def _assemble_grid(
    h: int,
    w: int,
    divider_r: int,
    shapes: list[tuple[tuple[int, int], ...]],
    colors: list[int],
) -> np.ndarray | None:
    grid = np.full((h, w), BG, dtype=int)
    grid[divider_r, :] = DIVIDER

    template_occupied = np.zeros((h, w), dtype=bool)
    query_occupied = np.zeros((h, w), dtype=bool)

    template_shapes = [_apply_variant(shape, random.choice(DIHEDRAL)) for shape in shapes]
    query_shapes = _choose_query_variants(shapes)

    order = list(range(len(shapes)))
    random.shuffle(order)
    for idx in order:
        patch = _shape_patch(template_shapes[idx], colors[idx])
        cells = template_shapes[idx]
        spots = _all_positions(
            template_occupied,
            cells,
            1,
            divider_r - 2,
            1,
            w - 2,
            forbid_edge=False,
        )
        if not spots:
            return None
        r0, c0 = random.choice(spots)
        paste(grid, patch, r0, c0, transparent=BG)
        _stamp_cells(template_occupied, cells, r0, c0)

    repeated_color = next(color for color in (LEFT, RIGHT) if colors.count(color) >= 2)
    repeated_indices = [i for i, color in enumerate(colors) if color == repeated_color][:2]
    pair_pos = _place_conflict_pair(
        query_occupied,
        query_shapes,
        (repeated_indices[0], repeated_indices[1]),
        repeated_color,
        divider_r,
    )
    if pair_pos is None:
        return None

    query_positions: dict[int, tuple[int, int]] = dict(pair_pos)
    for idx in range(len(shapes)):
        if idx in query_positions:
            continue
        cells = query_shapes[idx]
        patch = _shape_patch(cells, QUERY)
        spots = _all_positions(
            query_occupied,
            cells,
            divider_r + 1,
            h - 1,
            0,
            w - 1,
            forbid_edge=True,
        )
        if not spots:
            return None
        r0, c0 = random.choice(spots)
        query_positions[idx] = (r0, c0)
        _stamp_cells(query_occupied, cells, r0, c0)
        paste(grid, patch, r0, c0, transparent=BG)

    for idx in repeated_indices:
        patch = _shape_patch(query_shapes[idx], QUERY)
        r0, c0 = query_positions[idx]
        paste(grid, patch, r0, c0, transparent=BG)

    return grid


def _output_query_map(grid: np.ndarray, divider_r: int) -> dict[tuple[tuple[int, int], ...], dict]:
    lower = grid[divider_r + 1 :, :]
    out = {}
    for comp in find_components(lower, bg=BG, connectivity=4, monochromatic=True, colors=[LEFT, RIGHT]):
        coords = comp["coords"].copy()
        coords[:, 0] += divider_r + 1
        key = _canonical_shape(coords)
        out[key] = {
            "color": int(comp["color"]),
            "coords": coords,
            "bbox": _bbox(coords),
        }
    return out


def _validate_instance(inp: np.ndarray, out: np.ndarray) -> None:
    divider_r, template_by_shape, query_objects = _parse_input(inp)
    assert np.array_equal(out[: divider_r + 1], inp[: divider_r + 1]), "upper half must remain unchanged"

    for obj in query_objects:
        coords = obj["coords"]
        assert np.all(out[coords[:, 0], coords[:, 1]] == BG), "original query cells must be cleared"

    out_map = _output_query_map(out, divider_r)
    assert len(out_map) == len(query_objects), "output must contain one lower object per query"
    assert set(out_map) == set(template_by_shape), "output shapes must match the input query shapes"

    repeated_overlap = False
    for i, obj_i in enumerate(query_objects):
        for obj_j in query_objects[i + 1 :]:
            if (
                int(obj_i["target_color"]) == int(obj_j["target_color"])
                and _row_overlap(obj_i["bbox"], obj_j["bbox"])
            ):
                repeated_overlap = True
    assert repeated_overlap, "generator must include at least one same-color row-overlap conflict"

    width = inp.shape[1]
    for obj in query_objects:
        key = obj["shape"]
        dest = out_map[key]
        assert int(dest["color"]) == int(obj["target_color"]), "matched template color must recolor the query"
        assert dest["bbox"][0] == obj["bbox"][0] and dest["bbox"][2] == obj["bbox"][2], "rows must stay fixed"
        delta = dest["coords"] - obj["coords"]
        assert np.all(delta[:, 0] == 0), "movement must be purely horizontal"
        assert np.all(delta[:, 1] == delta[0, 1]), "all query pixels must translate together"

    for color in (LEFT, RIGHT):
        group = [obj for obj in query_objects if int(obj["target_color"]) == color]
        for obj in group:
            r0, c0, r1, c1 = obj["bbox"]
            obj["distance"] = c0 if color == LEFT else (width - 1 - c1)
        group.sort(key=lambda obj: (int(obj["distance"]), int(obj["bbox"][0])))
        placed = []
        for obj in group:
            dest = out_map[obj["shape"]]
            overlaps = [prev for prev in placed if _row_overlap(obj["bbox"], prev["bbox"])]
            if color == LEFT:
                expected_c0 = max([0] + [int(prev["bbox"][3]) + 1 for prev in overlaps])
                assert dest["bbox"][1] == expected_c0, "left-sliding conflicts must pack from the edge"
            else:
                expected_c1 = min([width - 1] + [int(prev["bbox"][1]) - 1 for prev in overlaps])
                assert dest["bbox"][3] == expected_c1, "right-sliding conflicts must pack from the edge"
            placed.append(dest)


def generate_input() -> np.ndarray:
    assert len(SHAPE_CATALOG) >= 8, "shape catalog unexpectedly small"

    for _ in range(MAX_ATTEMPTS):
        count = random.choices([2, 3, 4], weights=[1, 4, 3])[0]
        h = random.randint(18, 24)
        w = random.randint(20, 28)
        divider_r = random.randint(6, min(9, h - 8))

        shapes = random.sample(SHAPE_CATALOG, count)
        colors = _pick_template_colors(count)
        grid = _assemble_grid(h, w, divider_r, shapes, colors)
        if grid is None:
            continue

        try:
            out = main(grid)
            _validate_instance(grid, out)
        except AssertionError:
            continue

        if max(grid.shape) > 30 or max(out.shape) > 30:
            continue
        return grid

    raise RuntimeError("failed to generate a valid ARC instance")
