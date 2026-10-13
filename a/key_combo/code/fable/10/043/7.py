from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np


BLACK = 0
GREY = 5
PALETTE = tuple(range(10))
OBJECT_COLORS = tuple(c for c in PALETTE if c not in {BLACK, GREY})

ICON_HEIGHT = 3
ICON_ZONE_RIGHT = 5
DOT_GAP = 2
ROW_STEP = 4
BOX_TOP_MARGIN = 1
BOX_SIDE_MARGIN = 1
COUNT_RANGE = (1, 4)
KEY_RANGE = (2, 4)
GRID_RANGE = (22, 30)
BOX_SLACK_RANGE = (0, 3)
GEN_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.array(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[:] = patch
        return
    mask = patch != int(transparent)
    view[mask] = patch[mask]


def _component_coords(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    H, W = mask.shape
    seen = np.zeros((H, W), dtype=bool)
    if connectivity == 8:
        neighbors = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
    else:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    comps: list[np.ndarray] = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in neighbors:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _mask_connected(mask: np.ndarray) -> bool:
    return len(_component_coords(mask.astype(bool), connectivity=8)) == 1


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_from_coords(coords)
    return np.array(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def _shape_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}
    for k in range(4):
        rot = np.rot90(mask, k)
        for cand in (rot, np.fliplr(rot)):
            trim = _trim_mask(cand)
            key = (tuple(map(int, trim.shape)), trim.astype(np.uint8).tobytes())
            variants.setdefault(key, trim.copy())
    return list(variants.values())


def _canonical_shape(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    best = None
    for variant in _shape_variants(mask):
        key = (tuple(map(int, variant.shape)), variant.astype(np.uint8).tobytes())
        if best is None or key < best:
            best = key
    assert best is not None
    return best


def _mask_from_component(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = bbox_from_coords(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _extract_colored_components(grid: np.ndarray, colors: list[int]) -> list[dict]:
    comps: list[dict] = []
    for color in colors:
        for coords in _component_coords(grid == int(color), connectivity=8):
            bbox = bbox_from_coords(coords)
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": bbox,
                    "shape_key": _canonical_shape(_mask_from_component(coords)),
                }
            )
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint the legend box by turning each grey icon row into a left-to-right color record of matching shapes.

    Input structure:
    - The grid has a non-black background, one solid black legend box, and several grey icon components inside that box.
    - Each grey icon is a small connected shape; icons are pairwise different up to rotation and reflection.
    - Outside the box, every monochrome 8-connected object has a shape equivalent to exactly one grey icon, but its color is arbitrary.

    Rule:
    - Find the black legend box and keep only that box area in the output.
    - For each grey icon, match all outside objects whose binary shape is equivalent under rotation or reflection.
    - Sort those matched objects by their leftmost column, breaking ties by top row.
    - Keep the grey icon, then on the icon's middle row place one colored dot every other cell to its right; the k-th dot takes the k-th matched object's color.

    Key insight:
    - Shape class, not color, is the legend key; the colored dots record both how many matching objects exist and their spatial order in the full scene.
    """
    g = np.asarray(grid, dtype=int)

    black_comps = _component_coords(g == BLACK, connectivity=8)
    assert black_comps, "missing black legend box"
    legend_coords = max(black_comps, key=lambda pts: pts.shape[0])
    legend_bbox = bbox_from_coords(legend_coords)
    r0, c0, r1, c1 = legend_bbox
    box = crop_bbox(g, legend_bbox)

    icon_comps = _component_coords(box == GREY, connectivity=8)
    assert icon_comps, "missing grey icons"

    icons = []
    for coords in icon_comps:
        b = bbox_from_coords(coords)
        rr0, cc0, rr1, cc1 = b
        assert (rr1 - rr0 + 1) % 2 == 1, "icon height must have a middle row"
        icons.append(
            {
                "bbox": b,
                "row": int((rr0 + rr1) // 2),
                "start_col": int(cc1 + 2),
                "shape_key": _canonical_shape(_mask_from_component(coords)),
            }
        )
    icons.sort(key=lambda item: (item["row"], item["bbox"][1]))

    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = get_bg(g)
    colors = [int(c) for c in np.unique(outside) if c not in {get_bg(g), BLACK, GREY}]
    objects = _extract_colored_components(outside, colors)

    keyed_objects: dict[tuple[tuple[int, int], bytes], list[dict]] = defaultdict(list)
    for obj in objects:
        keyed_objects[obj["shape_key"]].append(obj)

    out = np.full(box.shape, BLACK, dtype=int)
    out[box == GREY] = GREY

    for icon in icons:
        matches = keyed_objects.get(icon["shape_key"], [])
        matches.sort(key=lambda obj: (obj["bbox"][1], obj["bbox"][0]))
        for idx, obj in enumerate(matches):
            cc = int(icon["start_col"] + DOT_GAP * idx)
            if cc < out.shape[1]:
                out[icon["row"], cc] = int(obj["color"])
    return out


def _icon_library() -> list[np.ndarray]:
    library: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}
    for w in (2, 3, 4):
        cells = ICON_HEIGHT * w
        for bits in range(1 << cells):
            mask = np.zeros((ICON_HEIGHT, w), dtype=bool)
            for idx in range(cells):
                if (bits >> idx) & 1:
                    mask[idx // w, idx % w] = True
            area = int(mask.sum())
            if area < 4 or area > 7:
                continue
            if not mask[0].any() or not mask[1].any() or not mask[2].any():
                continue
            if not _mask_connected(mask):
                continue
            if len(_shape_variants(mask)) < 2:
                continue
            key = _canonical_shape(mask)
            library.setdefault(key, mask.copy())
    masks = list(library.values())
    masks.sort(key=lambda m: (m.shape[1], int(m.sum()), m.astype(np.uint8).tobytes()))
    return masks


ICON_LIBRARY = _icon_library()


def _sample_shapes(k: int) -> list[np.ndarray]:
    assert len(ICON_LIBRARY) >= k
    return [ICON_LIBRARY[i].copy() for i in random.sample(range(len(ICON_LIBRARY)), k)]


def _build_legend(shapes: list[np.ndarray], counts: list[int]) -> tuple[np.ndarray, list[dict]]:
    max_count = max(counts)
    box_h = BOX_TOP_MARGIN * 2 + ICON_HEIGHT + ROW_STEP * (len(shapes) - 1)
    dot_start = ICON_ZONE_RIGHT + 2
    box_w = dot_start + DOT_GAP * max_count + random.randint(*BOX_SLACK_RANGE)
    legend = np.full((box_h, box_w), BLACK, dtype=int)
    info = []
    for i, shape in enumerate(shapes):
        top = BOX_TOP_MARGIN + i * ROW_STEP
        left = ICON_ZONE_RIGHT - shape.shape[1] + 1
        patch = np.full(shape.shape, BLACK, dtype=int)
        patch[shape] = GREY
        paste(legend, patch, top, left, transparent=BLACK)
        info.append(
            {
                "shape": shape.copy(),
                "row": int(top + shape.shape[0] // 2),
                "left": int(left),
                "right": int(left + shape.shape[1] - 1),
                "shape_key": _canonical_shape(shape),
            }
        )
    return legend, info


def _expanded_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar0 - gap <= br1 and br0 - gap <= ar1 and ac0 - gap <= bc1 and bc0 - gap <= ac1


def _variant_for_icon(shape: np.ndarray, *, different_from_icon: bool = False) -> tuple[np.ndarray, bool]:
    variants = _shape_variants(shape)
    base_key = (tuple(map(int, shape.shape)), shape.astype(np.uint8).tobytes())
    others = [v for v in variants if (tuple(map(int, v.shape)), v.astype(np.uint8).tobytes()) != base_key]
    if different_from_icon and others:
        picked = random.choice(others)
        return picked.copy(), True
    picked = random.choice(variants)
    changed = (tuple(map(int, picked.shape)), picked.astype(np.uint8).tobytes()) != base_key
    return picked.copy(), changed


def _render_object(shape: np.ndarray, color: int, bg: int) -> np.ndarray:
    patch = np.full(shape.shape, bg, dtype=int)
    patch[shape] = int(color)
    return patch


def _candidate_positions(
    H: int,
    W: int,
    patch_shape: tuple[int, int],
    placed_boxes: list[tuple[int, int, int, int]],
    *,
    fixed_c0: int | None = None,
) -> list[tuple[int, int]]:
    h, w = patch_shape
    cols = [fixed_c0] if fixed_c0 is not None else list(range(W - w + 1))
    rows = list(range(H - h + 1))
    random.shuffle(cols)
    random.shuffle(rows)
    out: list[tuple[int, int]] = []
    for c0 in cols:
        if c0 is None or c0 < 0 or c0 + w > W:
            continue
        for r0 in rows:
            cand = (r0, c0, r0 + h - 1, c0 + w - 1)
            if any(_expanded_overlap(cand, box, gap=1) for box in placed_boxes):
                continue
            out.append((r0, c0))
    return out


def _place_tie_pair(
    H: int,
    W: int,
    first_shape: np.ndarray,
    second_shape: np.ndarray,
    placed_boxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int]] | None:
    h1, w1 = first_shape.shape
    h2, w2 = second_shape.shape
    cols = list(range(W - max(w1, w2) + 1))
    random.shuffle(cols)
    for c0 in cols:
        first_positions = _candidate_positions(H, W, (h1, w1), placed_boxes, fixed_c0=c0)
        random.shuffle(first_positions)
        for r1, _ in first_positions[:50]:
            box1 = (r1, c0, r1 + h1 - 1, c0 + w1 - 1)
            second_positions = _candidate_positions(H, W, (h2, w2), placed_boxes + [box1], fixed_c0=c0)
            if second_positions:
                r2, _ = random.choice(second_positions)
                return [(r1, c0), (r2, c0)]
    return None


def _force_color_baits(records: list[dict], active_colors: list[int], counts: list[int]) -> None:
    group_ids = sorted({int(rec["shape_idx"]) for rec in records})
    if len(group_ids) >= 2:
        shared_color = random.choice(active_colors)
        a, b = random.sample(group_ids, 2)
        next(rec for rec in records if rec["shape_idx"] == a)["color"] = shared_color
        next(rec for rec in records if rec["shape_idx"] == b)["color"] = shared_color

    multi_groups = [idx for idx, count in enumerate(counts) if count >= 2]
    if multi_groups and len(active_colors) >= 2:
        group = random.choice(multi_groups)
        group_records = [rec for rec in records if rec["shape_idx"] == group]
        chosen = random.sample(active_colors, 2)
        group_records[0]["color"] = chosen[0]
        group_records[1]["color"] = chosen[1]


def _semantic_expected(legend: np.ndarray, icon_info: list[dict], object_records: list[dict]) -> np.ndarray:
    out = legend.copy()
    by_shape: dict[tuple[tuple[int, int], bytes], list[dict]] = defaultdict(list)
    for rec in object_records:
        by_shape[rec["shape_key"]].append(rec)
    for icon in icon_info:
        row = int(icon["row"])
        start = int(icon["right"] + 2)
        matches = by_shape[icon["shape_key"]]
        matches.sort(key=lambda rec: (rec["bbox"][1], rec["bbox"][0]))
        for idx, rec in enumerate(matches):
            out[row, start + DOT_GAP * idx] = int(rec["color"])
    return out


def _generate_once() -> np.ndarray | None:
    bg = random.choice([c for c in OBJECT_COLORS if c != GREY])
    key_count = random.randint(*KEY_RANGE)
    counts = [random.randint(*COUNT_RANGE) for _ in range(key_count)]
    if max(counts) < 2:
        counts[random.randrange(key_count)] += 1

    shapes = _sample_shapes(key_count)
    legend, icon_info = _build_legend(shapes, counts)
    box_h, box_w = legend.shape

    H = random.randint(max(GRID_RANGE[0], box_h + 6), GRID_RANGE[1])
    W = random.randint(max(GRID_RANGE[0], box_w + 6), GRID_RANGE[1])
    if H > 30 or W > 30:
        return None

    grid = np.full((H, W), bg, dtype=int)
    legend_r0 = random.randint(1, H - box_h - 1)
    legend_c0 = random.randint(1, W - box_w - 1)
    paste(grid, legend, legend_r0, legend_c0)
    placed_boxes = [(legend_r0, legend_c0, legend_r0 + box_h - 1, legend_c0 + box_w - 1)]

    n_active = random.randint(3, min(5, len(OBJECT_COLORS) - 1))
    active_colors = random.sample([c for c in OBJECT_COLORS if c != bg], n_active)

    object_records = []
    for shape_idx, count in enumerate(counts):
        for order_idx in range(count):
            object_records.append({"shape_idx": shape_idx, "order_idx": order_idx, "color": random.choice(active_colors)})
    _force_color_baits(object_records, active_colors, counts)

    forced_transform = random.randrange(len(object_records))
    transformed_seen = False

    tie_groups = [idx for idx, count in enumerate(counts) if count >= 2]
    want_tie = bool(tie_groups) and random.random() < 0.55
    tie_group = random.choice(tie_groups) if want_tie else None

    if tie_group is not None:
        group_records = [rec for rec in object_records if rec["shape_idx"] == tie_group][:2]
        var1, changed1 = _variant_for_icon(shapes[tie_group], different_from_icon=(group_records[0] is object_records[forced_transform]))
        var2, changed2 = _variant_for_icon(shapes[tie_group], different_from_icon=(group_records[1] is object_records[forced_transform]))
        pair_positions = _place_tie_pair(H, W, var1, var2, placed_boxes)
        if pair_positions is None:
            return None
        for rec, variant, changed, (r0, c0) in zip(group_records, (var1, var2), (changed1, changed2), pair_positions):
            patch = _render_object(variant, rec["color"], bg)
            paste(grid, patch, r0, c0, transparent=bg)
            rec["shape_key"] = _canonical_shape(variant)
            rec["bbox"] = (r0, c0, r0 + patch.shape[0] - 1, c0 + patch.shape[1] - 1)
            rec["variant_key"] = (tuple(map(int, variant.shape)), variant.astype(np.uint8).tobytes())
            placed_boxes.append(rec["bbox"])
            transformed_seen |= changed

    remaining = [rec for rec in object_records if "bbox" not in rec]
    remaining.sort(key=lambda rec: shapes[rec["shape_idx"]].size, reverse=True)
    for idx, rec in enumerate(remaining):
        must_change = (object_records.index(rec) == forced_transform) and not transformed_seen
        variant, changed = _variant_for_icon(shapes[rec["shape_idx"]], different_from_icon=must_change)
        patch = _render_object(variant, rec["color"], bg)
        positions = _candidate_positions(H, W, patch.shape, placed_boxes)
        if not positions:
            return None
        r0, c0 = random.choice(positions)
        paste(grid, patch, r0, c0, transparent=bg)
        rec["shape_key"] = _canonical_shape(variant)
        rec["bbox"] = (r0, c0, r0 + patch.shape[0] - 1, c0 + patch.shape[1] - 1)
        rec["variant_key"] = (tuple(map(int, variant.shape)), variant.astype(np.uint8).tobytes())
        placed_boxes.append(rec["bbox"])
        transformed_seen |= changed

    if not transformed_seen:
        return None

    shape_keys = [_canonical_shape(shape) for shape in shapes]
    if len(set(shape_keys)) != key_count:
        return None

    for rec in object_records:
        if rec["shape_key"] not in shape_keys:
            return None

    expected = _semantic_expected(legend, icon_info, object_records)
    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None

    shared_color_ok = any(
        len({rec["shape_idx"] for rec in object_records if rec["color"] == color}) >= 2 for color in active_colors
    )
    multicolor_group_ok = any(
        len({rec["color"] for rec in object_records if rec["shape_idx"] == idx}) >= 2 for idx, count in enumerate(counts) if count >= 2
    )
    if not shared_color_ok or not multicolor_group_ok:
        return None

    if want_tie:
        tie_matches = [rec for rec in object_records if rec["shape_idx"] == tie_group]
        tie_matches.sort(key=lambda rec: (rec["bbox"][1], rec["bbox"][0]))
        if not any(
            tie_matches[i]["bbox"][1] == tie_matches[i + 1]["bbox"][1]
            for i in range(len(tie_matches) - 1)
        ):
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid legend-shape task input")
