from __future__ import annotations

import random
from collections import deque

import numpy as np


GREY = 5
BG = 0
OBJECT_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


SHAPE_LIBRARY = {
    "L": ((0, 0), (1, 0), (2, 0), (2, 1)),
    "T": ((0, 0), (0, 1), (0, 2), (1, 1)),
    "S": ((0, 1), (0, 2), (1, 0), (1, 1)),
    "O": ((0, 0), (0, 1), (1, 0), (1, 1)),
}

TEMPLATE_LIBRARY = {
    3: [
        np.array([[1, 0, 1], [1, 1, 1], [0, 1, 0]], dtype=bool),
        np.array([[1, 1, 0], [0, 1, 1], [1, 1, 0]], dtype=bool),
    ],
    4: [
        np.array(
            [[1, 0, 1, 0], [1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 1]],
            dtype=bool,
        ),
        np.array(
            [[1, 1, 0, 1], [0, 1, 1, 1], [1, 1, 0, 0], [0, 1, 1, 0]],
            dtype=bool,
        ),
    ],
}


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(grid: np.ndarray, colors: set[int]) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color not in colors:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _mask_from_shape(shape: tuple[tuple[int, int], ...]) -> np.ndarray:
    arr = np.array(shape, dtype=int)
    h = int(arr[:, 0].max()) + 1
    w = int(arr[:, 1].max()) + 1
    mask = np.zeros((h, w), dtype=bool)
    mask[arr[:, 0], arr[:, 1]] = True
    return mask


def _normalize_coords(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    coords = np.array(coords, dtype=int)
    coords = coords - coords.min(axis=0)
    return tuple(sorted((int(r), int(c)) for r, c in coords))


def _normalize_mask(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    return _normalize_coords(np.argwhere(mask))


def _dihedral_variants(shape: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    mask = _mask_from_shape(shape)
    out: list[tuple[tuple[int, int], ...]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for rot in range(4):
        cur = np.rot90(mask, rot)
        for variant in (cur, np.fliplr(cur)):
            key = _normalize_mask(variant)
            if key not in seen:
                seen.add(key)
                out.append(key)
    return out


def _column_groups(mask: np.ndarray) -> list[tuple[int, int]]:
    cols = np.where(mask.any(axis=0))[0]
    if cols.size == 0:
        return []
    groups = []
    start = int(cols[0])
    prev = int(cols[0])
    for col in cols[1:]:
        col = int(col)
        if col != prev + 1:
            groups.append((start, prev))
            start = col
        prev = col
    groups.append((start, prev))
    return groups


def _parse_legend(grid: np.ndarray) -> list[dict]:
    grey_rows = np.where((grid == GREY).any(axis=1))[0]
    assert grey_rows.size > 0
    legend_end = int(grey_rows.max())
    legend = grid[: legend_end + 1]
    grey = legend == GREY
    items = []
    for c0, c1 in _column_groups(grey):
        block = grey[:, c0 : c1 + 1]
        k = int(c1 - c0 + 1)
        rows = np.where(block.any(axis=1))[0]
        assert rows.size > 0 and k in (3, 4)
        mask_bottom = int(rows.max())
        mask_top = mask_bottom - k + 1
        assert mask_top >= 0
        template = block[mask_top : mask_bottom + 1].copy()
        assert template.shape == (k, k)
        assert template.any() and template.any(axis=0).all() and template.any(axis=1).all()
        shape_zone = block[: max(mask_top - 1, 0)]
        shape_coords = np.argwhere(shape_zone)
        assert shape_coords.size > 0
        shape = _normalize_coords(shape_coords)
        items.append(
            {
                "shape": shape,
                "variants": set(_dihedral_variants(shape)),
                "template": template,
                "k": k,
            }
        )
    assert len(items) >= 2
    return items


def _scatter_objects(grid: np.ndarray) -> list[dict]:
    colors = set(int(v) for v in np.unique(grid) if int(v) not in (BG, GREY))
    comps = _find_components(grid, colors)
    out = []
    for comp in comps:
        r0, c0, _r1, _c1 = comp["bbox"]
        shape = _normalize_coords(comp["coords"] - np.array([r0, c0], dtype=int))
        out.append({**comp, "shape": shape})
    assert out
    return out


def _match_items(objects: list[dict], legend: list[dict]) -> list[dict]:
    matched = []
    for obj in objects:
        hits = [item for item in legend if obj["shape"] in item["variants"]]
        assert len(hits) == 1
        matched.append({**obj, "match": hits[0]})
    return matched


def _render_output(objects: list[dict], input_shape: tuple[int, int]) -> tuple[np.ndarray, int]:
    max_k = max(int(obj["match"]["k"]) for obj in objects)
    canvas = np.zeros((input_shape[0] * max_k + max_k, input_shape[1] * max_k + max_k), dtype=int)
    occupied = np.zeros_like(canvas, dtype=bool)
    blocked = 0
    for obj in sorted(objects, key=lambda d: (int(d["bbox"][0]), int(d["bbox"][1]))):
        color = int(obj["color"])
        template = obj["match"]["template"]
        for r, c in obj["coords"]:
            base_r = int(r) * int(obj["match"]["k"])
            base_c = int(c) * int(obj["match"]["k"])
            for tr, tc in np.argwhere(template):
                rr = base_r + int(tr)
                cc = base_c + int(tc)
                if occupied[rr, cc]:
                    blocked += 1
                    continue
                occupied[rr, cc] = True
                canvas[rr, cc] = color
    coords = np.argwhere(occupied)
    r0, c0, r1, c1 = _bbox_coords(coords)
    return canvas[r0 : r1 + 1, c0 : c1 + 1].copy(), blocked


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each lower object to a grey legend shape, then enlarge it with that shape's paired template mask.

    Input structure:
    - Background is black.
    - The top legend contains 2 or 3 grey pairs laid out left-to-right; each pair has a small 4-connected index shape above a grey k×k template mask, with a blank row between them.
    - Below the legend, separated by background rows, sit several non-grey monochrome 4-connected objects.

    Rule:
    - Parse every legend pair by its shared column block: the block width is the template size k, the lower k rows are the template mask, and the grey cells above define the index shape.
    - For each lower object, normalize its geometry and match it against every legend index shape under the 8 dihedral orientations; the unique match chooses that object's template mask.
    - Replace every object pixel by a k×k copy of the matched mask, painting mask-on cells with the object's color and leaving mask-off cells black.
    - Place enlarged objects in ascending order of original bbox top-left; each object's pixels are anchored at their original coordinates multiplied by k, and any later enlarged pixel that lands on an already painted output cell is skipped.
    - Return the tight crop around all painted output pixels.

    Key insight:
    - Color does not select the enlargement pattern; geometry does. The solver must read the legend, do rotation/flip-invariant shape matching, and only then apply the correct pixel-expansion mask.
    """
    g = np.asarray(grid, dtype=int)
    legend = _parse_legend(g)
    objects = _match_items(_scatter_objects(g), legend)
    out, _blocked = _render_output(objects, g.shape)
    return out


def _build_pair_block(shape: tuple[tuple[int, int], ...], template: np.ndarray) -> np.ndarray:
    shape_arr = np.array(shape, dtype=int)
    shape_h = int(shape_arr[:, 0].max()) + 1
    k = int(template.shape[0])
    block = np.zeros((shape_h + 1 + k, k), dtype=int)
    col_shift = (k - (int(shape_arr[:, 1].max()) + 1)) // 2
    for r, c in shape:
        block[int(r), int(c) + col_shift] = GREY
    tpl = block[shape_h + 1 : shape_h + 1 + k, :]
    tpl[template] = GREY
    return block


def _transformed_shape(shape: tuple[tuple[int, int], ...], key: tuple[tuple[int, int], ...]) -> np.ndarray:
    coords = np.array(key, dtype=int)
    return coords


def _place_object_cells(
    placed: list[np.ndarray],
    shape: np.ndarray,
    top: int,
    left: int,
) -> np.ndarray | None:
    coords = shape + np.array([top, left], dtype=int)
    occupied = set()
    for arr in placed:
        for cell in arr:
            occupied.add((int(cell[0]), int(cell[1])))
    for r, c in coords:
        rc = (int(r), int(c))
        if rc in occupied:
            return None
        for rr, cc in occupied:
            if abs(rr - int(r)) + abs(cc - int(c)) == 1:
                return None
    return coords


def _attempt_layout(
    specs: list[dict],
    n_objects: int,
    row_offset: int,
) -> tuple[list[dict], tuple[int, int]] | None:
    h_limit = random.randint(5, 7)
    w_limit = random.randint(7, 10)
    color_pool = random.sample(OBJECT_COLORS, n_objects)
    idx_by_k = {3: [i for i, s in enumerate(specs) if int(s["k"]) == 3], 4: [i for i, s in enumerate(specs) if int(s["k"]) == 4]}
    if not idx_by_k[3] or not idx_by_k[4]:
        return None
    chosen = [random.choice(idx_by_k[3]), random.choice(idx_by_k[4])]
    while len(chosen) < n_objects:
        chosen.append(random.randrange(len(specs)))
    random.shuffle(chosen)
    objects: list[dict] = []
    placed: list[np.ndarray] = []
    used_nonbase = False
    for color, spec_idx in zip(color_pool, chosen):
        spec = specs[spec_idx]
        variants = _dihedral_variants(spec["shape"])
        variant = random.choice(variants)
        if variant != spec["shape"]:
            used_nonbase = True
        shape = _transformed_shape(spec["shape"], variant)
        rh = int(shape[:, 0].max()) + 1
        rw = int(shape[:, 1].max()) + 1
        spots = [(r, c) for r in range(h_limit - rh + 1) for c in range(w_limit - rw + 1)]
        random.shuffle(spots)
        coords = None
        for top, left in spots:
            placed_coords = _place_object_cells(placed, shape, top, left)
            if placed_coords is not None:
                coords = placed_coords
                break
        if coords is None:
            return None
        placed.append(coords)
        bbox = _bbox_coords(coords)
        objects.append(
            {
                "color": int(color),
                "coords": coords,
                "bbox": bbox,
                "shape": variant,
                "match": spec,
            }
        )
    if not used_nonbase:
        return None
    objects.sort(key=lambda d: (int(d["bbox"][0]), int(d["bbox"][1])))
    absolute_objects = []
    for obj in objects:
        coords = obj["coords"] + np.array([row_offset, 0], dtype=int)
        absolute_objects.append(
            {
                **obj,
                "coords": coords,
                "bbox": _bbox_coords(coords),
            }
        )
    out, blocked = _render_output(absolute_objects, (row_offset + h_limit, w_limit))
    if blocked <= 0 or max(out.shape) > 30:
        return None
    return absolute_objects, (row_offset + h_limit, w_limit)


def _build_grid(specs: list[dict], objects: list[dict], base_shape: tuple[int, int]) -> np.ndarray:
    gap = 2
    legend_blocks = [_build_pair_block(spec["shape"], spec["template"]) for spec in specs]
    legend_h = max(block.shape[0] for block in legend_blocks)
    legend_w = sum(block.shape[1] for block in legend_blocks) + gap * (len(legend_blocks) - 1)
    total_h = max(legend_h, base_shape[0], max(int(obj["bbox"][2]) + 1 for obj in objects))
    total_w = max(legend_w, base_shape[1], max(int(obj["bbox"][3]) + 1 for obj in objects))
    grid = np.zeros((total_h, total_w), dtype=int)
    col = 0
    for block in legend_blocks:
        br, bc = block.shape
        grid[:br, col : col + bc] = np.maximum(grid[:br, col : col + bc], block)
        col += bc + gap
    for obj in objects:
        for r, c in obj["coords"]:
            grid[int(r), int(c)] = int(obj["color"])
    return grid


def _canonical_shape(shape: tuple[tuple[int, int], ...]) -> bytes:
    variants = _dihedral_variants(shape)
    return min(repr(v).encode() for v in variants)


def _validate_generated(grid: np.ndarray, specs: list[dict], objects: list[dict]) -> None:
    shape_keys = [_canonical_shape(spec["shape"]) for spec in specs]
    assert len(shape_keys) == len(set(shape_keys))
    assert len(objects) >= 2
    assert len({int(obj["color"]) for obj in objects}) == len(objects)
    assert any(int(obj["match"]["k"]) == 3 for obj in objects)
    assert any(int(obj["match"]["k"]) == 4 for obj in objects)
    for obj in objects:
        hits = [spec for spec in specs if obj["shape"] in set(_dihedral_variants(spec["shape"]))]
        assert len(hits) == 1
    out = main(grid)
    exp, blocked = _render_output(objects, grid.shape)
    assert blocked > 0
    assert np.array_equal(out, exp)
    assert max(grid.shape) <= 30 and max(out.shape) <= 30


def generate_input() -> np.ndarray:
    for _ in range(500):
        n_pairs = random.choice([2, 3])
        shape_names = random.sample(list(SHAPE_LIBRARY), n_pairs)
        if n_pairs == 2:
            k_plan = [3, 4]
        else:
            k_plan = random.choice([[3, 4, 4], [3, 3, 4]])
        random.shuffle(k_plan)
        available_templates = {3: TEMPLATE_LIBRARY[3][:], 4: TEMPLATE_LIBRARY[4][:]}
        random.shuffle(available_templates[3])
        random.shuffle(available_templates[4])
        specs = []
        used_templates = {3: 0, 4: 0}
        for name, k in zip(shape_names, k_plan):
            template = available_templates[k][used_templates[k] % len(available_templates[k])]
            used_templates[k] += 1
            specs.append(
                {
                    "shape": SHAPE_LIBRARY[name],
                    "template": template.copy(),
                    "k": int(k),
                }
            )
        legend_h = max(_build_pair_block(spec["shape"], spec["template"]).shape[0] for spec in specs)
        n_objects = random.randint(2, 4)
        row_offset = legend_h + 2
        layout = _attempt_layout(specs, n_objects, row_offset)
        if layout is None:
            continue
        objects, base_shape = layout
        grid = _build_grid(specs, objects, base_shape)
        try:
            _validate_generated(grid, specs, objects)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid sample")
