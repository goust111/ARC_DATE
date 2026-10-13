import numpy as np
from collections import defaultdict, deque


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _mask_signature(mask: np.ndarray) -> tuple[int, int, tuple[int, ...]]:
    trimmed = _trim_mask(mask)
    return (
        int(trimmed.shape[0]),
        int(trimmed.shape[1]),
        tuple(int(x) for x in trimmed.astype(int).ravel()),
    )


def _all_transforms(mask: np.ndarray) -> list[np.ndarray]:
    out = []
    for k in range(4):
        rot = np.rot90(mask, k)
        out.append(_trim_mask(rot))
        out.append(_trim_mask(np.fliplr(rot)))
    return out


def _unique_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants = []
    seen = set()
    for variant in _all_transforms(mask):
        sig = _mask_signature(variant)
        if sig in seen:
            continue
        seen.add(sig)
        variants.append(variant)
    return variants


def _canonical_shape_key(mask: np.ndarray) -> tuple[int, int, tuple[int, ...]]:
    return min(_mask_signature(variant) for variant in _all_transforms(mask))


def _component_mask(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = comp["coords"] - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _collect_components(grid: np.ndarray, bg: int) -> tuple[list[dict], np.ndarray]:
    h, w = grid.shape
    cell_to_comp = np.full((h, w), -1, dtype=int)
    comps = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for coords in _find_components(grid == color):
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            comp = {
                "id": len(comps),
                "color": int(color),
                "coords": coords,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
            }
            comp["shape_key"] = _canonical_shape_key(_component_mask(comp))
            comps.append(comp)
            cell_to_comp[coords[:, 0], coords[:, 1]] = comp["id"]
    return comps, cell_to_comp


def _path_endpoints(coords: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]] | None:
    cells = {tuple(map(int, cell)) for cell in coords}
    degrees = {}
    for r, c in cells:
        deg = 0
        for dr, dc in DIR4:
            if (r + dr, c + dc) in cells:
                deg += 1
        degrees[(r, c)] = deg
    if not degrees or max(degrees.values()) > 2:
        return None
    endpoints = sorted(pos for pos, deg in degrees.items() if deg == 1)
    if len(endpoints) != 2:
        return None
    return endpoints[0], endpoints[1]


def _adjacent_component_ids(
    coords: np.ndarray,
    grid: np.ndarray,
    bg: int,
    line_color: int,
    cell_to_comp: np.ndarray,
) -> set[int]:
    h, w = grid.shape
    touched = set()
    for r, c in coords:
        for dr, dc in DIR4:
            nr, nc = int(r + dr), int(c + dc)
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            color = int(grid[nr, nc])
            if color in (bg, line_color):
                continue
            touched.add(int(cell_to_comp[nr, nc]))
    return touched


def _adjacent_component_ids_for_cell(
    pos: tuple[int, int],
    grid: np.ndarray,
    bg: int,
    line_color: int,
    cell_to_comp: np.ndarray,
) -> set[int]:
    h, w = grid.shape
    r, c = pos
    touched = set()
    for dr, dc in DIR4:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            continue
        color = int(grid[nr, nc])
        if color in (bg, line_color):
            continue
        touched.add(int(cell_to_comp[nr, nc]))
    return touched


def _identify_sample_pairs(grid: np.ndarray, bg: int, comps: list[dict], cell_to_comp: np.ndarray) -> tuple[int, list[dict]]:
    by_color: dict[int, list[dict]] = defaultdict(list)
    for comp in comps:
        by_color[int(comp["color"])].append(comp)

    candidates = []
    for color, color_comps in by_color.items():
        sample_pairs = []
        for comp in color_comps:
            endpoints = _path_endpoints(comp["coords"])
            if endpoints is None:
                continue
            touched = _adjacent_component_ids(comp["coords"], grid, bg, color, cell_to_comp)
            if len(touched) != 2:
                continue
            obj_a, obj_b = [comps[cid] for cid in sorted(touched)]
            if int(obj_a["color"]) == int(obj_b["color"]):
                continue
            if obj_a["shape_key"] != obj_b["shape_key"]:
                continue

            endpoint_to_obj = {}
            valid = True
            for endpoint in endpoints:
                adj = _adjacent_component_ids_for_cell(endpoint, grid, bg, color, cell_to_comp) & touched
                if len(adj) != 1:
                    valid = False
                    break
                endpoint_to_obj[endpoint] = next(iter(adj))
            if not valid or len(set(endpoint_to_obj.values())) != 2:
                continue

            sample_pairs.append(
                {
                    "line_comp": comp,
                    "endpoints": endpoints,
                    "endpoint_to_obj": endpoint_to_obj,
                }
            )
        if len(sample_pairs) == 2 and len(sample_pairs) == len(color_comps):
            candidates.append((int(color), sample_pairs))

    assert candidates, "failed to identify the grey connection lines"
    candidates.sort(key=lambda item: item[0])
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor target objects by matching their shape to the implicit sample-pair dictionary."""
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    comps, cell_to_comp = _collect_components(g, bg)
    line_color, sample_pairs = _identify_sample_pairs(g, bg, comps, cell_to_comp)

    out = g.copy()
    sample_object_ids = set()
    mapping: dict[tuple[int, int, tuple[int, ...]], int] = {}

    for pair in sample_pairs:
        line_comp = pair["line_comp"]
        endpoints = pair["endpoints"]
        endpoint_to_obj = pair["endpoint_to_obj"]
        source_endpoint, target_endpoint = sorted(endpoints)
        source_id = int(endpoint_to_obj[source_endpoint])
        target_id = int(endpoint_to_obj[target_endpoint])
        source_obj = comps[source_id]
        target_obj = comps[target_id]
        assert source_obj["shape_key"] == target_obj["shape_key"]
        mapping[source_obj["shape_key"]] = int(target_obj["color"])
        sample_object_ids.add(source_id)
        sample_object_ids.add(target_id)
        out[line_comp["coords"][:, 0], line_comp["coords"][:, 1]] = bg

    for comp in comps:
        if int(comp["color"]) == line_color or comp["id"] in sample_object_ids:
            continue
        new_color = mapping.get(comp["shape_key"])
        if new_color is None:
            continue
        out[comp["coords"][:, 0], comp["coords"][:, 1]] = int(new_color)

    return out


# --------------- generator ---------------

RNG = np.random.default_rng()
LINE_COLOR = 5
GEN_ATTEMPTS = 400
SHAPE_ATTEMPTS = 200
PLACE_ATTEMPTS = 250


def _normalize_coords(coords: set[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return sorted((r - min_r, c - min_c) for r, c in coords)


def _coords_to_mask(coords: list[tuple[int, int]]) -> np.ndarray:
    h = max(r for r, _ in coords) + 1
    w = max(c for _, c in coords) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    return mask


def _random_shape(min_area: int = 4, max_area: int = 6, min_variants: int = 4) -> np.ndarray | None:
    for _ in range(SHAPE_ATTEMPTS):
        area = int(RNG.integers(min_area, max_area + 1))
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for dr, dc in DIR4:
                    nxt = (r + dr, c + dc)
                    if nxt not in cells:
                        frontier.add(nxt)
            if not frontier:
                break
            frontier_list = sorted(frontier)
            cells.add(frontier_list[int(RNG.integers(len(frontier_list)))])
        coords = _normalize_coords(cells)
        mask = _coords_to_mask(coords)
        variants = _unique_variants(mask)
        if min(mask.shape) < 2 or max(mask.shape) > 5:
            continue
        if len(variants) < min_variants:
            continue
        return mask
    return None


def _pick_distinct_colors(bg: int, count: int) -> list[int]:
    palette = [c for c in range(10) if c not in (bg, LINE_COLOR)]
    choice = RNG.choice(palette, size=count, replace=False)
    return [int(x) for x in np.atleast_1d(choice)]


def _pick_color(forbidden: set[int]) -> int:
    palette = [c for c in range(10) if c not in forbidden]
    return int(RNG.choice(palette))


def _boundary_cells(mask: np.ndarray, side: str) -> list[tuple[int, int]]:
    coords = np.argwhere(mask)
    if side == "right":
        edge = int(coords[:, 1].max())
        pts = coords[coords[:, 1] == edge]
    elif side == "left":
        edge = int(coords[:, 1].min())
        pts = coords[coords[:, 1] == edge]
    elif side == "top":
        edge = int(coords[:, 0].min())
        pts = coords[coords[:, 0] == edge]
    else:
        edge = int(coords[:, 0].max())
        pts = coords[coords[:, 0] == edge]
    return [tuple(map(int, pt)) for pt in pts]


def _paint_mask(patch: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> list[tuple[int, int]]:
    coords = []
    for r, c in np.argwhere(mask):
        rr, cc = int(top + r), int(left + c)
        assert patch[rr, cc] == -1
        patch[rr, cc] = int(color)
        coords.append((rr, cc))
    return coords


def _draw_path(
    patch: np.ndarray,
    points: list[tuple[int, int]],
    color: int,
) -> list[tuple[int, int]]:
    coords = []
    for r, c in points:
        assert patch[r, c] == -1
        patch[r, c] = int(color)
        coords.append((int(r), int(c)))
    return coords


def _build_pair_entity(shape_mask: np.ndarray, source_color: int, target_color: int) -> dict:
    variants = _unique_variants(shape_mask)
    src_idx = int(RNG.integers(len(variants)))
    other_choices = [i for i in range(len(variants)) if i != src_idx]
    dst_idx = int(RNG.choice(other_choices)) if other_choices else src_idx
    src_mask = variants[src_idx]
    dst_mask = variants[dst_idx]

    mode = str(RNG.choice(["horizontal_first", "vertical_first"]))
    row_gap = int(RNG.integers(2, 6))
    col_gap = int(RNG.integers(2, 7))
    src_h, src_w = src_mask.shape
    dst_h, dst_w = dst_mask.shape
    dst_top = src_h + row_gap
    dst_left = src_w + col_gap
    patch = np.full((dst_top + dst_h, dst_left + dst_w), -1, dtype=int)

    _paint_mask(patch, src_mask, 0, 0, source_color)
    _paint_mask(patch, dst_mask, dst_top, dst_left, target_color)

    if mode == "horizontal_first":
        src_candidates = _boundary_cells(src_mask, "right")
        dst_candidates = _boundary_cells(dst_mask, "left")
        src_cell = src_candidates[int(RNG.integers(len(src_candidates)))]
        dst_cell = dst_candidates[int(RNG.integers(len(dst_candidates)))]
        p1 = (int(src_cell[0]), int(src_cell[1] + 1))
        p2 = (int(dst_top + dst_cell[0]), int(dst_left + dst_cell[1] - 1))
        line = [(p1[0], c) for c in range(p1[1], p2[1] + 1)]
        line.extend((r, p2[1]) for r in range(p1[0] + 1, p2[0] + 1))
    else:
        src_candidates = _boundary_cells(src_mask, "bottom")
        dst_candidates = _boundary_cells(dst_mask, "top")
        src_cell = src_candidates[int(RNG.integers(len(src_candidates)))]
        dst_cell = dst_candidates[int(RNG.integers(len(dst_candidates)))]
        p1 = (int(src_cell[0] + 1), int(src_cell[1]))
        p2 = (int(dst_top + dst_cell[0] - 1), int(dst_left + dst_cell[1]))
        line = [(r, p1[1]) for r in range(p1[0], p2[0] + 1)]
        line.extend((p2[0], c) for c in range(p1[1] + 1, p2[1] + 1))

    line_coords = _draw_path(patch, line, LINE_COLOR)
    return {
        "patch": patch,
        "area": int(np.count_nonzero(patch >= 0)),
        "line_coords_local": line_coords,
        "recolor_groups": [],
        "variant_count": len(variants),
        "used_variant_ids": {src_idx, dst_idx},
    }


def _build_target_entity(mask: np.ndarray, color: int, new_color: int | None) -> dict:
    patch = np.full(mask.shape, -1, dtype=int)
    coords = _paint_mask(patch, mask, 0, 0, color)
    recolor_groups = []
    if new_color is not None:
        recolor_groups.append((coords, int(new_color)))
    return {
        "patch": patch,
        "area": int(np.count_nonzero(patch >= 0)),
        "line_coords_local": [],
        "recolor_groups": recolor_groups,
    }


def _can_place(reserved: np.ndarray, patch: np.ndarray, top: int, left: int) -> bool:
    h, w = reserved.shape
    ph, pw = patch.shape
    if top < 0 or left < 0 or top + ph > h or left + pw > w:
        return False
    for r, c in np.argwhere(patch >= 0):
        if reserved[top + int(r), left + int(c)]:
            return False
    return True


def _mark_reserved(reserved: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = reserved.shape
    for r, c in np.argwhere(patch >= 0):
        rr, cc = top + int(r), left + int(c)
        for nr in range(max(0, rr - 1), min(h, rr + 2)):
            for nc in range(max(0, cc - 1), min(w, cc + 2)):
                reserved[nr, nc] = True


def _place_entities(grid: np.ndarray, entities: list[dict]) -> list[dict] | None:
    h, w = grid.shape
    reserved = np.zeros((h, w), dtype=bool)
    placed = []
    order = sorted(entities, key=lambda entity: entity["area"], reverse=True)
    for entity in order:
        patch = entity["patch"]
        ph, pw = patch.shape
        success = False
        for _ in range(PLACE_ATTEMPTS):
            top = int(RNG.integers(0, h - ph + 1))
            left = int(RNG.integers(0, w - pw + 1))
            if not _can_place(reserved, patch, top, left):
                continue
            occ = np.argwhere(patch >= 0)
            grid[top + occ[:, 0], left + occ[:, 1]] = patch[occ[:, 0], occ[:, 1]]
            _mark_reserved(reserved, patch, top, left)
            placed_entity = dict(entity)
            placed_entity["top"] = top
            placed_entity["left"] = left
            placed.append(placed_entity)
            success = True
            break
        if not success:
            return None
    return placed


def _globalize(coords: list[tuple[int, int]], top: int, left: int) -> list[tuple[int, int]]:
    return [(int(top + r), int(left + c)) for r, c in coords]


def _build_expected(grid: np.ndarray, bg: int, placed: list[dict]) -> np.ndarray:
    out = grid.copy()
    for entity in placed:
        top, left = int(entity["top"]), int(entity["left"])
        for r, c in _globalize(entity["line_coords_local"], top, left):
            out[r, c] = int(bg)
        for coords, color in entity["recolor_groups"]:
            glob = _globalize(coords, top, left)
            rr = [r for r, _ in glob]
            cc = [c for _, c in glob]
            out[rr, cc] = int(color)
    return out


def _gen_attempt() -> np.ndarray | None:
    bg = int(RNG.choice([c for c in range(10) if c != LINE_COLOR]))

    sample_shapes = []
    sample_keys = set()
    while len(sample_shapes) < 2:
        mask = _random_shape()
        if mask is None:
            return None
        key = _canonical_shape_key(mask)
        if key in sample_keys:
            continue
        sample_shapes.append({"mask": mask, "key": key, "variants": _unique_variants(mask)})
        sample_keys.add(key)

    unmatched_shapes = []
    unmatched_count = int(RNG.integers(1, 3))
    unmatched_keys = set(sample_keys)
    while len(unmatched_shapes) < unmatched_count:
        mask = _random_shape(min_area=3, max_area=6, min_variants=2)
        if mask is None:
            return None
        key = _canonical_shape_key(mask)
        if key in unmatched_keys:
            continue
        unmatched_shapes.append({"mask": mask, "key": key, "variants": _unique_variants(mask)})
        unmatched_keys.add(key)

    entities = []
    mapping_colors = {}
    for shape in sample_shapes:
        src_color, dst_color = _pick_distinct_colors(bg, 2)
        pair_entity = _build_pair_entity(shape["mask"], src_color, dst_color)
        entities.append(pair_entity)
        mapping_colors[shape["key"]] = int(dst_color)

        target_count = int(RNG.integers(1, 4))
        forced_ids = [i for i in range(len(shape["variants"])) if i not in pair_entity["used_variant_ids"]]
        first_variant_id = int(RNG.choice(forced_ids)) if forced_ids else int(RNG.integers(len(shape["variants"])))
        for idx in range(target_count):
            if idx == 0:
                variant = shape["variants"][first_variant_id]
            else:
                variant = shape["variants"][int(RNG.integers(len(shape["variants"])))]
            color = _pick_color({bg, LINE_COLOR, dst_color})
            entities.append(_build_target_entity(variant, color, dst_color))

    for shape in unmatched_shapes:
        variant = shape["variants"][int(RNG.integers(len(shape["variants"])))]
        color = _pick_color({bg, LINE_COLOR})
        entities.append(_build_target_entity(variant, color, None))

    max_h = max(entity["patch"].shape[0] for entity in entities)
    max_w = max(entity["patch"].shape[1] for entity in entities)
    total_area = sum(entity["area"] for entity in entities)
    min_side = max(max_h + 2, max_w + 2, int(np.ceil(np.sqrt(total_area * 8))))
    if min_side > 30:
        return None

    h = int(RNG.integers(min_side, 31))
    w = int(RNG.integers(min_side, 31))
    grid = np.full((h, w), int(bg), dtype=int)
    placed = _place_entities(grid, entities)
    if placed is None:
        return None
    if np.count_nonzero(grid == bg) <= grid.size // 2:
        return None

    expected = _build_expected(grid, bg, placed)
    produced = main(grid)
    if not np.array_equal(produced, expected):
        return None
    if np.array_equal(produced, grid):
        return None
    if np.any(produced == LINE_COLOR):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
