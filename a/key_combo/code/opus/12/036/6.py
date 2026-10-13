from __future__ import annotations

import random
from collections import defaultdict
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_UTILS = SourceFileLoader("arc_utils_arc_036_6", str(_ARC_UTILS_PATH)).load_module()

Color = _ARC_UTILS.Color
bbox_coords = _ARC_UTILS.bbox_coords
crop_bbox = _ARC_UTILS.crop_bbox
draw_line = _ARC_UTILS.draw_line
find_components = _ARC_UTILS.find_components
paste = _ARC_UTILS.paste


BACKGROUND = Color.BLACK
PAIR_COUNTS = (2, 3)
PAIR_ORIENTATIONS = ("horizontal", "vertical")
PALETTE = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE)
PAIR_GAP_RANGE = (4, 6)
GRID_RANGE_BY_PAIR_COUNT = {
    2: ((20, 28), (20, 28)),
    3: ((24, 30), (24, 30)),
}
MAX_BBOX_SIDE = 4
MIN_OBJECT_GAP = 2
MAX_GENERATION_ATTEMPTS = 400
PAIR_PLACEMENT_ATTEMPTS = 200

NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _normalize_cells(cells: set[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    pts = list(cells)
    min_r = min(r for r, _ in pts)
    min_c = min(c for _, c in pts)
    return tuple(sorted((int(r - min_r), int(c - min_c)) for r, c in pts))


def _cells_to_mask(cells: tuple[tuple[int, int], ...]) -> np.ndarray:
    h = max(r for r, _ in cells) + 1
    w = max(c for _, c in cells) + 1
    mask = np.zeros((h, w), dtype=bool)
    rr, cc = np.array(cells, dtype=int).T
    mask[rr, cc] = True
    return mask


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _unique_mask_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for base in (mask, np.fliplr(mask)):
        for k in range(4):
            variant = _crop_mask(np.rot90(base, k))
            key = _mask_key(variant)
            if key not in seen:
                seen.add(key)
                variants.append(variant)
    return variants


def _canonical_shape_key(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    return min(_mask_key(variant) for variant in _unique_mask_variants(mask))


def _has_full_dihedral_orbit(mask: np.ndarray) -> bool:
    return len(_unique_mask_variants(mask)) == 8


def _build_shape_catalog() -> list[dict]:
    by_size: dict[int, set[tuple[tuple[int, int], ...]]] = {1: {((0, 0),)}}
    for size in range(2, 8):
        shapes: set[tuple[tuple[int, int], ...]] = set()
        for cells in by_size[size - 1]:
            cell_set = set(cells)
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    nxt = (r + dr, c + dc)
                    if nxt not in cell_set:
                        shapes.add(_normalize_cells(cell_set | {nxt}))
        by_size[size] = shapes

    catalog: list[dict] = []
    seen_keys: set[tuple[tuple[int, int], ...]] = set()
    for size in range(3, 8):
        for cells in by_size[size]:
            mask = _cells_to_mask(cells)
            h, w = mask.shape
            if h < 2 or w < 2 or h > MAX_BBOX_SIDE or w > MAX_BBOX_SIDE:
                continue
            if not _has_full_dihedral_orbit(mask):
                continue

            key = _canonical_shape_key(mask)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            variants_by_dims: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
            for variant in _unique_mask_variants(mask):
                variants_by_dims[variant.shape].append(variant.astype(bool))

            catalog.append(
                {
                    "key": key,
                    "area": size,
                    "variants_by_dims": dict(variants_by_dims),
                }
            )

    assert len(catalog) >= 10
    return catalog


SHAPE_CATALOG = _build_shape_catalog()
SQUARE_SHAPES = [entry for entry in SHAPE_CATALOG if any(h == w for h, w in entry["variants_by_dims"])]


def _component_mask(grid: np.ndarray, comp: dict) -> np.ndarray:
    return crop_bbox(grid, comp["bbox"]) != BACKGROUND


def _center2(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return int(r0 + r1), int(c0 + c1)


def _choose_edge_pixel(
    coords: np.ndarray,
    bbox: tuple[int, int, int, int],
    side: str,
) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    center2_r, center2_c = _center2(bbox)
    if side == "left":
        candidates = coords[coords[:, 1] == c0]
        scores = np.abs(2 * candidates[:, 0] - center2_r)
        idx = min(range(len(candidates)), key=lambda i: (int(scores[i]), int(candidates[i, 0])))
    elif side == "right":
        candidates = coords[coords[:, 1] == c1]
        scores = np.abs(2 * candidates[:, 0] - center2_r)
        idx = min(range(len(candidates)), key=lambda i: (int(scores[i]), int(candidates[i, 0])))
    elif side == "top":
        candidates = coords[coords[:, 0] == r0]
        scores = np.abs(2 * candidates[:, 1] - center2_c)
        idx = min(range(len(candidates)), key=lambda i: (int(scores[i]), int(candidates[i, 1])))
    elif side == "bottom":
        candidates = coords[coords[:, 0] == r1]
        scores = np.abs(2 * candidates[:, 1] - center2_c)
        idx = min(range(len(candidates)), key=lambda i: (int(scores[i]), int(candidates[i, 1])))
    else:
        raise ValueError(f"unknown side {side}")
    r, c = candidates[idx]
    return int(r), int(c)


def _bridge_patch(height: int, width: int, color_a: int, color_b: int, orientation: str) -> np.ndarray:
    patch = np.zeros((height, width), dtype=int)
    if orientation == "horizontal":
        split = (width + 1) // 2
        patch[0, :split] = color_a
        patch[-1, :split] = color_a
        patch[0, split:] = color_b
        patch[-1, split:] = color_b
        patch[:, 0] = color_a
        patch[:, -1] = color_b
    else:
        split = (height + 1) // 2
        patch[:split, 0] = color_a
        patch[:split, -1] = color_a
        patch[split:, 0] = color_b
        patch[split:, -1] = color_b
        patch[0, :] = color_a
        patch[-1, :] = color_b
    return patch


def _bridge_top_left(
    bbox_a: tuple[int, int, int, int],
    bbox_b: tuple[int, int, int, int],
    height: int,
    width: int,
) -> tuple[int, int]:
    center2_a_r, center2_a_c = _center2(bbox_a)
    center2_b_r, center2_b_c = _center2(bbox_b)
    num_r = center2_a_r + center2_b_r - 2 * (height - 1)
    num_c = center2_a_c + center2_b_c - 2 * (width - 1)
    if num_r % 4 == 0:
        r0 = num_r // 4
    else:
        r0 = int(round((center2_a_r + center2_b_r) / 4 - (height - 1) / 2))
    if num_c % 4 == 0:
        c0 = num_c // 4
    else:
        c0 = int(round((center2_a_c + center2_b_c) / 4 - (width - 1) / 2))
    return int(r0), int(c0)


def _order_pair(comp_a: dict, comp_b: dict) -> tuple[dict, dict, str]:
    center2_a_r, center2_a_c = _center2(comp_a["bbox"])
    center2_b_r, center2_b_c = _center2(comp_b["bbox"])
    dr = abs(center2_a_r - center2_b_r)
    dc = abs(center2_a_c - center2_b_c)
    orientation = "horizontal" if dc >= dr else "vertical"

    if orientation == "horizontal":
        key_a = (center2_a_c, center2_a_r, comp_a["bbox"][1], comp_a["bbox"][0])
        key_b = (center2_b_c, center2_b_r, comp_b["bbox"][1], comp_b["bbox"][0])
    else:
        key_a = (center2_a_r, center2_a_c, comp_a["bbox"][0], comp_a["bbox"][1])
        key_b = (center2_b_r, center2_b_c, comp_b["bbox"][0], comp_b["bbox"][1])

    return (comp_a, comp_b, orientation) if key_a <= key_b else (comp_b, comp_a, orientation)


def _build_pair_spec(comp_a: dict, comp_b: dict) -> dict:
    first, second, orientation = _order_pair(comp_a, comp_b)
    mask_a = first["mask"]
    mask_b = second["mask"]
    assert mask_a.shape == mask_b.shape, "each generated pair must share one bbox orientation"
    height, width = mask_a.shape
    bridge_r0, bridge_c0 = _bridge_top_left(first["bbox"], second["bbox"], height, width)
    bridge_bbox = (bridge_r0, bridge_c0, bridge_r0 + height - 1, bridge_c0 + width - 1)
    bridge_patch = _bridge_patch(height, width, int(first["color"]), int(second["color"]), orientation)

    if orientation == "horizontal":
        start_a = _choose_edge_pixel(first["coords"], first["bbox"], "right")
        start_b = _choose_edge_pixel(second["coords"], second["bbox"], "left")
        assert bridge_r0 <= start_a[0] <= bridge_bbox[2]
        assert bridge_r0 <= start_b[0] <= bridge_bbox[2]
        lines = [
            (start_a, (start_a[0], bridge_c0), int(first["color"])),
            (start_b, (start_b[0], bridge_bbox[3]), int(second["color"])),
        ]
    else:
        start_a = _choose_edge_pixel(first["coords"], first["bbox"], "bottom")
        start_b = _choose_edge_pixel(second["coords"], second["bbox"], "top")
        assert bridge_c0 <= start_a[1] <= bridge_bbox[3]
        assert bridge_c0 <= start_b[1] <= bridge_bbox[3]
        lines = [
            (start_a, (bridge_r0, start_a[1]), int(first["color"])),
            (start_b, (bridge_bbox[2], start_b[1]), int(second["color"])),
        ]

    return {
        "key": first["shape_key"],
        "orientation": orientation,
        "first": first,
        "second": second,
        "bridge_bbox": bridge_bbox,
        "bridge_patch": bridge_patch,
        "lines": lines,
    }


def _parse_pairs(grid: np.ndarray, *, strict: bool) -> list[dict]:
    comps = find_components(
        grid,
        bg=BACKGROUND,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    if strict:
        assert len(comps) in {4, 6}, "expected 4 or 6 monochrome sprites"
    else:
        assert len(comps) >= 2 and len(comps) % 2 == 0

    enriched = []
    for comp in comps:
        mask = _component_mask(grid, comp)
        enriched.append(
            {
                "color": int(comp["color"]),
                "coords": comp["coords"].astype(int),
                "bbox": tuple(map(int, comp["bbox"])),
                "area": int(comp["area"]),
                "mask": mask,
                "shape_key": _canonical_shape_key(mask),
            }
        )

    groups: dict[tuple[tuple[int, int], ...], list[dict]] = defaultdict(list)
    for comp in enriched:
        groups[comp["shape_key"]].append(comp)

    pairs = []
    for group in groups.values():
        assert len(group) == 2, "every shape must appear exactly twice"
        pairs.append(_build_pair_spec(group[0], group[1]))

    pairs.sort(key=lambda pair: pair["bridge_bbox"])
    if strict:
        assert len(pairs) in PAIR_COUNTS
    return pairs


def _render_pairs(grid: np.ndarray, pairs: list[dict]) -> np.ndarray:
    out = grid.copy()
    for pair in pairs:
        bridge_r0, bridge_c0, _bridge_r1, _bridge_c1 = pair["bridge_bbox"]
        paste(out, pair["bridge_patch"], bridge_r0, bridge_c0, transparent=BACKGROUND)
        for (start_r, start_c), (end_r, end_c), color in pair["lines"]:
            draw_line(out, r=start_r, c=start_c, end_r=end_r, end_c=end_c, color=color)
    return out


def _solve(grid: np.ndarray, *, strict: bool) -> np.ndarray:
    pairs = _parse_pairs(grid, strict=strict)
    return _render_pairs(grid, pairs)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair rotation/reflection-equivalent sprites and bridge each pair with a split-color hollow rectangle.

    Input structure:
    - A black background contains 4 or 6 separated monochrome 4-connected sprites.
    - Each sprite shape appears exactly twice, ignoring color and allowing rotations/reflections.
    - Paired sprites are spaced apart and aligned mainly horizontally or vertically.

    Rule:
    - Crop every sprite to its bbox and pair sprites whose foreground masks are equivalent under the 8 dihedral transforms.
    - For each pair, compute the midpoint of the two bbox centers and place there a hollow rectangle whose size matches the shared sprite bbox.
    - Split the rectangle by pair direction: left/right colors for horizontal pairs, top/bottom colors for vertical pairs, then connect each sprite's nearest side pixel to the corresponding rectangle side with a straight line of that sprite's color.

    Key insight:
    - Shape matching decides both which sprites belong together and how large each bridge must be, so pairing, bridge geometry, and color split are inseparable.
    """
    g = np.asarray(grid, dtype=int)
    return _solve(g, strict=True)


def _bbox_gap_ok(new_bbox: tuple[int, int, int, int], existing: list[tuple[int, int, int, int]], gap: int = MIN_OBJECT_GAP) -> bool:
    r0, c0, r1, c1 = new_bbox
    for er0, ec0, er1, ec1 in existing:
        if r1 < er0:
            dr = er0 - r1 - 1
        elif er1 < r0:
            dr = r0 - er1 - 1
        else:
            dr = 0

        if c1 < ec0:
            dc = ec0 - c1 - 1
        elif ec1 < c0:
            dc = c0 - ec1 - 1
        else:
            dc = 0

        if max(dr, dc) < gap:
            return False
    return True


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    sprite = np.zeros(mask.shape, dtype=int)
    sprite[mask] = int(color)
    paste(grid, sprite, top, left, transparent=BACKGROUND)


def _pick_shape_entries(pair_count: int) -> list[dict]:
    if pair_count == 3 and SQUARE_SHAPES and random.random() < 0.6:
        square_entry = random.choice(SQUARE_SHAPES)
        pool = [entry for entry in SHAPE_CATALOG if entry["key"] != square_entry["key"]]
        others = random.sample(pool, pair_count - 1)
        entries = [square_entry, *others]
        random.shuffle(entries)
        return entries
    return random.sample(SHAPE_CATALOG, pair_count)


def _pick_orientations(pair_count: int) -> list[str]:
    if pair_count == 2:
        options = [
            ["horizontal", "horizontal"],
            ["vertical", "vertical"],
            ["horizontal", "vertical"],
            ["vertical", "horizontal"],
        ]
        return random.choice(options)

    orientations = ["horizontal", "vertical", random.choice(PAIR_ORIENTATIONS)]
    random.shuffle(orientations)
    return orientations


def _pick_variants(entry: dict) -> tuple[np.ndarray, np.ndarray]:
    dims = random.choice(list(entry["variants_by_dims"]))
    variants = entry["variants_by_dims"][dims]
    if len(variants) >= 2 and random.random() < 0.85:
        a, b = random.sample(variants, 2)
    else:
        a = random.choice(variants)
        b = random.choice(variants)
    return a.copy(), b.copy()


def _pair_input_grid(
    shape_a: np.ndarray,
    shape_b: np.ndarray,
    color_a: int,
    color_b: int,
    a_top: int,
    a_left: int,
    b_top: int,
    b_left: int,
    shape: tuple[int, int],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    _paint_mask(grid, shape_a, a_top, a_left, color_a)
    _paint_mask(grid, shape_b, b_top, b_left, color_b)
    return grid


def _sample_pair_layout(
    grid_h: int,
    grid_w: int,
    sprite_h: int,
    sprite_w: int,
    orientation: str,
) -> dict | None:
    gap_a = random.randint(*PAIR_GAP_RANGE)
    gap_b = random.randint(*PAIR_GAP_RANGE)

    if orientation == "horizontal":
        min_bridge_c = gap_a + sprite_w - 2
        max_bridge_c = grid_w - (2 * sprite_w + gap_b - 2)
        if min_bridge_c > max_bridge_c or sprite_h > grid_h:
            return None
        bridge_r0 = random.randint(0, grid_h - sprite_h)
        bridge_c0 = random.randint(min_bridge_c, max_bridge_c)
        a_top = bridge_r0
        b_top = bridge_r0
        a_left = bridge_c0 - gap_a - sprite_w + 2
        b_left = bridge_c0 + sprite_w + gap_b - 2
    else:
        min_bridge_r = gap_a + sprite_h - 2
        max_bridge_r = grid_h - (2 * sprite_h + gap_b - 2)
        if min_bridge_r > max_bridge_r or sprite_w > grid_w:
            return None
        bridge_r0 = random.randint(min_bridge_r, max_bridge_r)
        bridge_c0 = random.randint(0, grid_w - sprite_w)
        a_top = bridge_r0 - gap_a - sprite_h + 2
        b_top = bridge_r0 + sprite_h + gap_b - 2
        a_left = bridge_c0
        b_left = bridge_c0

    return {
        "bridge_top_left": (bridge_r0, bridge_c0),
        "a_top_left": (a_top, a_left),
        "b_top_left": (b_top, b_left),
    }


def _validate_generated_grid(grid: np.ndarray) -> bool:
    pairs = _parse_pairs(grid, strict=True)
    out = main(grid)
    expected = _render_pairs(grid, pairs)
    assert np.array_equal(out, expected)
    assert out.shape == grid.shape
    assert not np.array_equal(out, grid)

    used_bridge = np.zeros_like(grid, dtype=bool)
    for pair in pairs:
        bridge_r0, bridge_c0, bridge_r1, bridge_c1 = pair["bridge_bbox"]
        bridge_patch = pair["bridge_patch"]
        assert np.count_nonzero(grid[bridge_r0 : bridge_r1 + 1, bridge_c0 : bridge_c1 + 1]) == 0

        bridge_mask = np.zeros_like(grid, dtype=bool)
        bridge_mask[bridge_r0 : bridge_r1 + 1, bridge_c0 : bridge_c1 + 1] = bridge_patch != 0
        assert not np.any(used_bridge & bridge_mask)
        used_bridge |= bridge_mask

        sub = out[bridge_r0 : bridge_r1 + 1, bridge_c0 : bridge_c1 + 1]
        assert np.array_equal(np.where(bridge_patch != 0, sub, 0), bridge_patch)

        for (start_r, start_c), (end_r, end_c), color in pair["lines"]:
            if start_r == end_r:
                c0, c1 = sorted((start_c, end_c))
                segment = out[start_r, c0 : c1 + 1]
            else:
                r0, r1 = sorted((start_r, end_r))
                segment = out[r0 : r1 + 1, start_c]
            assert np.all(segment == color)

    return True


def _attempt_generate() -> np.ndarray | None:
    pair_count = random.choice(PAIR_COUNTS)
    (h_lo, h_hi), (w_lo, w_hi) = GRID_RANGE_BY_PAIR_COUNT[pair_count]
    grid_h = random.randint(h_lo, h_hi)
    grid_w = random.randint(w_lo, w_hi)
    colors = random.sample(PALETTE, 2 * pair_count)
    shape_entries = _pick_shape_entries(pair_count)
    orientations = _pick_orientations(pair_count)

    grid = np.zeros((grid_h, grid_w), dtype=int)
    reserved_output = np.zeros((grid_h, grid_w), dtype=bool)
    object_bboxes: list[tuple[int, int, int, int]] = []

    for index, (entry, orientation) in enumerate(zip(shape_entries, orientations)):
        color_a = colors[2 * index]
        color_b = colors[2 * index + 1]
        placed = False

        for _ in range(PAIR_PLACEMENT_ATTEMPTS):
            shape_a, shape_b = _pick_variants(entry)
            sprite_h, sprite_w = shape_a.shape
            if shape_b.shape != (sprite_h, sprite_w):
                continue

            layout = _sample_pair_layout(grid_h, grid_w, sprite_h, sprite_w, orientation)
            if layout is None:
                continue

            a_top, a_left = layout["a_top_left"]
            b_top, b_left = layout["b_top_left"]
            a_bbox = (a_top, a_left, a_top + sprite_h - 1, a_left + sprite_w - 1)
            b_bbox = (b_top, b_left, b_top + sprite_h - 1, b_left + sprite_w - 1)
            if not _bbox_gap_ok(a_bbox, object_bboxes):
                continue
            if not _bbox_gap_ok(b_bbox, object_bboxes + [a_bbox]):
                continue

            pair_input = _pair_input_grid(
                shape_a,
                shape_b,
                color_a,
                color_b,
                a_top,
                a_left,
                b_top,
                b_left,
                (grid_h, grid_w),
            )
            pair_output = _solve(pair_input, strict=False)
            occupied = pair_output != BACKGROUND
            if np.any(reserved_output & occupied):
                continue

            grid |= pair_input
            reserved_output |= occupied
            object_bboxes.extend([a_bbox, b_bbox])
            placed = True
            break

        if not placed:
            return None

    if not _validate_generated_grid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid split-bridge pairing puzzle")
