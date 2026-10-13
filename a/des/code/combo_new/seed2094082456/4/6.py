from __future__ import annotations

import random

import numpy as np


BG = 0
SEPARATOR = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

CELL_H = 5
CELL_W = 4
LEGEND_TOP = 1
LEGEND_BOTTOM_GAP = 1
LEGEND_GROUP_GAP = 1
HEADER_H = LEGEND_TOP + CELL_H + LEGEND_BOTTOM_GAP + 1

MIN_SCENE_H = 10
MAX_SCENE_H = 22
MIN_TILE_H = 7
MIN_TILE_W = 7
MAX_GENERATION_ATTEMPTS = 400
OBJECT_ATTEMPTS = 120


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_bool(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _monochrome_components(grid: np.ndarray, ignore: set[int]) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in ignore or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(
                {
                    "color": color,
                    "bbox": _bbox_from_coords(coords),
                    "coords": np.array(coords, dtype=int),
                }
            )
    return comps


def _component_mask(comp: dict[str, object], grid: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    color = int(comp["color"])
    return grid[r0 : r1 + 1, c0 : c1 + 1] == color


def _crop_true_bbox(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert len(coords) > 0
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _hole_mask(object_mask: np.ndarray) -> np.ndarray:
    h, w = object_mask.shape
    seen = np.zeros((h, w), dtype=bool)
    holes: list[np.ndarray] = []
    bg_mask = ~object_mask

    for r in range(h):
        for c in range(w):
            if not bg_mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            touches_boundary = False
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                if cr == 0 or cc == 0 or cr == h - 1 or cc == w - 1:
                    touches_boundary = True
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if bg_mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            if not touches_boundary:
                hole = np.zeros((h, w), dtype=bool)
                for hr, hc in coords:
                    hole[hr, hc] = True
                holes.append(_crop_true_bbox(hole))

    assert len(holes) == 1, "expected exactly one enclosed hole"
    return holes[0]


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def _canonical_scaled_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    best = mask.copy()
    best_key = (h * w, h, w)

    for base_h in _divisors(h):
        row_scale = h // base_h
        for base_w in _divisors(w):
            col_scale = w // base_w
            base = np.zeros((base_h, base_w), dtype=bool)
            ok = True
            for br in range(base_h):
                for bc in range(base_w):
                    block = mask[
                        br * row_scale : (br + 1) * row_scale,
                        bc * col_scale : (bc + 1) * col_scale,
                    ]
                    if block.all():
                        base[br, bc] = True
                    elif not block.any():
                        base[br, bc] = False
                    else:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            if np.array_equal(np.kron(base, np.ones((row_scale, col_scale), dtype=bool)), mask):
                key = (base_h * base_w, base_h, base_w)
                if key < best_key:
                    best = base
                    best_key = key
    return best


def _mask_signature(mask: np.ndarray) -> tuple[int, int, bytes]:
    canon = _canonical_scaled_mask(_crop_true_bbox(mask))
    return canon.shape[0], canon.shape[1], canon.astype(np.uint8).tobytes()


def _scale_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    src_h, src_w = mask.shape
    dst_h, dst_w = shape
    row_idx = (np.arange(dst_h) * src_h) // dst_h
    col_idx = (np.arange(dst_w) * src_w) // dst_w
    return mask[row_idx[:, None], col_idx[None, :]]


def _find_separator_row(grid: np.ndarray) -> int:
    candidates = []
    for r in range(grid.shape[0]):
        row = grid[r]
        if len(set(map(int, row))) != 1:
            continue
        color = int(row[0])
        if color == BG:
            continue
        if r == 0 or r == grid.shape[0] - 1:
            continue
        if np.any(grid[:r] != BG) and np.any(grid[r + 1 :] != BG):
            candidates.append(r)
    assert len(candidates) == 1, "expected one solid separator row"
    return int(candidates[0])


def _legend_mapping(legend: np.ndarray) -> dict[tuple[int, int, bytes], tuple[np.ndarray, int]]:
    comps = _monochrome_components(legend, ignore={BG})
    comps.sort(key=lambda comp: (comp["bbox"][1], comp["bbox"][0]))
    assert len(comps) >= 4 and len(comps) % 2 == 0

    mapping: dict[tuple[int, int, bytes], tuple[np.ndarray, int]] = {}
    for idx in range(0, len(comps), 2):
        key_comp = comps[idx]
        value_comp = comps[idx + 1]
        kr0, kc0, kr1, kc1 = key_comp["bbox"]
        vr0, vc0, vr1, vc1 = value_comp["bbox"]
        assert kr0 == vr0 and kr1 == vr1, "legend key/value blocks must align"
        assert vc0 - kc1 == 1, "legend key/value blocks must be adjacent"
        if idx + 2 < len(comps):
            next_left = comps[idx + 2]["bbox"][1]
            assert next_left - vc1 >= 2, "legend pairs must be separated by black space"

        key_mask = _component_mask(key_comp, legend)
        value_mask = _component_mask(value_comp, legend)
        signature = _mask_signature(_hole_mask(key_mask))
        assert signature not in mapping, "hole shapes must uniquely identify legend entries"
        mapping[signature] = (value_mask, int(value_comp["color"]))

    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """Replace each hollow object by the legend value paired with its enclosed-hole shape.

    Input structure:
    - The background is black, and a single solid grey horizontal row separates the legend from the main area.
    - Above the separator, the legend contains left-to-right key/value pairs: each key is one monochrome component with one enclosed black hole, and each adjacent value is another monochrome template component.
    - Below the separator, the main area contains several non-overlapping monochrome 4-connected objects, each with exactly one enclosed black hole.

    Rule:
    - Remove the legend and separator from the output canvas.
    - For every lower object, extract its unique enclosed hole, normalize that hole by undoing any integer row/column scaling, and use the normalized hole shape as the lookup key.
    - Find the legend key with the same normalized hole shape, scale the paired legend value template to the object's bounding box, and draw that scaled template in the paired value color.

    Key insight:
    - The object's visible outer color and size are distractions; the true lookup key is the geometry of the enclosed black hole.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    separator_row = _find_separator_row(g)
    legend = g[:separator_row, :]
    scene = g[separator_row + 1 :, :]
    mapping = _legend_mapping(legend)

    out = np.zeros_like(scene)
    for comp in _monochrome_components(scene, ignore={BG}):
        r0, c0, r1, c1 = comp["bbox"]
        obj_mask = _component_mask(comp, scene)
        signature = _mask_signature(_hole_mask(obj_mask))
        assert signature in mapping, "every hole must match a legend key"
        value_mask, value_color = mapping[signature]
        scaled = _scale_mask(value_mask, (r1 - r0 + 1, c1 - c0 + 1))
        out[r0 : r1 + 1, c0 : c1 + 1][scaled] = value_color
    return out


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    comp = _components_bool(mask)[0]
    return len(comp) == len(coords)


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _all_small_patterns() -> tuple[list[np.ndarray], list[np.ndarray]]:
    hole_masks: list[np.ndarray] = []
    value_masks: list[np.ndarray] = []
    for bits in range(1, 1 << 6):
        flat = np.array([(bits >> i) & 1 for i in range(6)], dtype=bool)
        mask = flat.reshape(3, 2)
        if not _touches_all_sides(mask):
            continue
        if not _is_connected(mask):
            continue
        area = int(mask.sum())
        if area < 3:
            continue
        if area < 6:
            hole_masks.append(mask.copy())
        value_masks.append(mask.copy())
    assert len(hole_masks) >= 3 and len(value_masks) >= 4
    return hole_masks, value_masks


HOLE_LIBRARY, VALUE_LIBRARY = _all_small_patterns()


def _random_partition(total: int, parts: int, min_size: int) -> list[int]:
    assert total >= parts * min_size
    slack = total - parts * min_size
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(1, slack + parts), parts - 1))
    bounds = [0, *cuts, slack + parts]
    return [bounds[i + 1] - bounds[i] - 1 + min_size for i in range(parts)]


def _tile_boxes(height: int, width: int, rows: int, cols: int) -> list[tuple[int, int, int, int]]:
    row_sizes = _random_partition(height, rows, MIN_TILE_H)
    col_sizes = _random_partition(width, cols, MIN_TILE_W)
    row_starts = [0]
    col_starts = [0]
    for size in row_sizes[:-1]:
        row_starts.append(row_starts[-1] + size)
    for size in col_sizes[:-1]:
        col_starts.append(col_starts[-1] + size)

    boxes = []
    for rr, r0 in enumerate(row_starts):
        for cc, c0 in enumerate(col_starts):
            h = row_sizes[rr]
            w = col_sizes[cc]
            boxes.append((r0, c0, r0 + h - 1, c0 + w - 1))
    return boxes


def _render_key_patch(hole_mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full((CELL_H, CELL_W), color, dtype=int)
    patch[1:4, 1:3][hole_mask] = BG
    return patch


def _render_value_patch(value_mask: np.ndarray, color: int) -> np.ndarray:
    scaled = _scale_mask(value_mask, (CELL_H, CELL_W))
    return np.where(scaled, color, BG).astype(int)


def _sample_pair_defs(pair_count: int) -> list[dict[str, object]]:
    hole_choices = random.sample(HOLE_LIBRARY, pair_count)
    value_choices = random.sample(VALUE_LIBRARY, pair_count)
    defs: list[dict[str, object]] = []

    for hole_base, value_base in zip(hole_choices, value_choices):
        key_color = random.choice(PALETTE)
        value_color = random.choice([c for c in PALETTE if c != key_color])
        defs.append(
            {
                "hole_base": hole_base.copy(),
                "value_base": value_base.copy(),
                "value_template": _scale_mask(value_base, (CELL_H, CELL_W)),
                "key_color": int(key_color),
                "value_color": int(value_color),
            }
        )
    return defs


def _sample_layout(object_count: int, pair_count: int) -> tuple[int, int, int, int] | None:
    options = []
    legend_body_w = pair_count * (2 * CELL_W) + (pair_count - 1) * LEGEND_GROUP_GAP
    legend_total_min_w = legend_body_w + 2
    for rows, cols in ((1, 2), (1, 3), (1, 4), (2, 2), (2, 3)):
        if rows * cols < object_count:
            continue
        min_w = max(legend_total_min_w, cols * MIN_TILE_W)
        min_h = max(MIN_SCENE_H, rows * MIN_TILE_H)
        if min_w <= 30 and min_h <= MAX_SCENE_H:
            options.append((rows, cols, min_h, min_w))
    return random.choice(options) if options else None


def _build_object_in_tile(
    tile_h: int,
    tile_w: int,
    hole_base: np.ndarray,
    color: int,
) -> tuple[int, int, np.ndarray] | None:
    hole_base_h, hole_base_w = hole_base.shape
    avail_h = tile_h - 2
    avail_w = tile_w - 2
    if avail_h < hole_base_h + 2 or avail_w < hole_base_w + 2:
        return None

    for _ in range(OBJECT_ATTEMPTS):
        max_scale_h = (avail_h - 2) // hole_base_h
        max_scale_w = (avail_w - 2) // hole_base_w
        if max_scale_h < 1 or max_scale_w < 1:
            return None
        scale_h = random.randint(1, max_scale_h)
        scale_w = random.randint(1, max_scale_w)
        hole = np.kron(hole_base, np.ones((scale_h, scale_w), dtype=bool))
        hole_h, hole_w = hole.shape

        max_border_h = avail_h - hole_h
        max_border_w = avail_w - hole_w
        if max_border_h < 2 or max_border_w < 2:
            continue
        total_border_h = random.randint(2, max_border_h)
        total_border_w = random.randint(2, max_border_w)
        top = random.randint(1, total_border_h - 1)
        left = random.randint(1, total_border_w - 1)
        bottom = total_border_h - top
        right = total_border_w - left

        obj_h = hole_h + top + bottom
        obj_w = hole_w + left + right
        assert obj_h <= avail_h and obj_w <= avail_w

        patch = np.full((obj_h, obj_w), color, dtype=int)
        patch[top : top + hole_h, left : left + hole_w][hole] = BG

        place_r = random.randint(1, tile_h - obj_h - 1)
        place_c = random.randint(1, tile_w - obj_w - 1)
        return place_r, place_c, patch
    return None


def _compose_legend(width: int, pair_defs: list[dict[str, object]]) -> np.ndarray:
    legend = np.zeros((HEADER_H, width), dtype=int)
    legend[-1, :] = SEPARATOR

    legend_body_w = len(pair_defs) * (2 * CELL_W) + (len(pair_defs) - 1) * LEGEND_GROUP_GAP
    extra = width - legend_body_w
    left_margin = random.randint(1, extra - 1)

    col = left_margin
    for pair_def in pair_defs:
        key_patch = _render_key_patch(pair_def["hole_base"], int(pair_def["key_color"]))
        value_patch = _render_value_patch(pair_def["value_base"], int(pair_def["value_color"]))
        legend[LEGEND_TOP : LEGEND_TOP + CELL_H, col : col + CELL_W] = key_patch
        legend[LEGEND_TOP : LEGEND_TOP + CELL_H, col + CELL_W : col + 2 * CELL_W] = value_patch
        col += 2 * CELL_W + LEGEND_GROUP_GAP
    return legend


def _expected_scene(
    shape: tuple[int, int],
    objects: list[dict[str, object]],
    pair_defs: list[dict[str, object]],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for obj in objects:
        pair_def = pair_defs[int(obj["pair_index"])]
        r0, c0, r1, c1 = obj["bbox"]
        value_mask = _scale_mask(pair_def["value_template"], (r1 - r0 + 1, c1 - c0 + 1))
        out[r0 : r1 + 1, c0 : c1 + 1][value_mask] = int(pair_def["value_color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(2, 3)
    object_count = random.randint(pair_count, 5)
    layout = _sample_layout(object_count, pair_count)
    if layout is None:
        return None
    tile_rows, tile_cols, min_scene_h, min_scene_w = layout
    scene_h = random.randint(min_scene_h, MAX_SCENE_H)
    scene_w = random.randint(min_scene_w, 30)

    pair_defs = _sample_pair_defs(pair_count)
    tiles = _tile_boxes(scene_h, scene_w, tile_rows, tile_cols)
    chosen_tiles = random.sample(tiles, object_count)

    pair_indexes = list(range(pair_count))
    pair_indexes.extend(random.randrange(pair_count) for _ in range(object_count - pair_count))
    random.shuffle(pair_indexes)

    scene = np.zeros((scene_h, scene_w), dtype=int)
    objects: list[dict[str, object]] = []
    for tile, pair_index in zip(chosen_tiles, pair_indexes):
        tr0, tc0, tr1, tc1 = tile
        color = random.choice(PALETTE)
        pair_def = pair_defs[pair_index]
        built = _build_object_in_tile(
            tr1 - tr0 + 1,
            tc1 - tc0 + 1,
            pair_def["hole_base"],
            int(color),
        )
        if built is None:
            return None
        local_r, local_c, patch = built
        ph, pw = patch.shape
        sr0 = tr0 + local_r
        sc0 = tc0 + local_c
        scene[sr0 : sr0 + ph, sc0 : sc0 + pw] = patch
        objects.append(
            {
                "pair_index": int(pair_index),
                "bbox": (sr0, sc0, sr0 + ph - 1, sc0 + pw - 1),
            }
        )

    legend = _compose_legend(scene_w, pair_defs)
    grid = np.vstack([legend, scene])
    expected = _expected_scene(scene.shape, objects, pair_defs)

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
