from __future__ import annotations

from collections import Counter, defaultdict, deque
import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
TEAL = 8

LEGEND_COLORS = (BLUE, RED, GREEN)
VALID_COUNT_TRIPLETS = (
    (1, 2, 4),
    (1, 2, 5),
    (1, 3, 4),
    (1, 3, 5),
    (2, 3, 4),
    (1, 4, 5),
    (2, 3, 5),
)

_RNG = np.random.default_rng()


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(arr: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return arr[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        neigh = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        neigh = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]

    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in neigh:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    return out


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0, r1, c1 = _bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].astype(bool)


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    box = _bbox_coords(coords)
    r0, c0, r1, c1 = box
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    shifted = coords - np.array([r0, c0])
    out[shifted[:, 0], shifted[:, 1]] = True
    return out


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    tight = _tight_mask(np.asarray(mask, dtype=bool))
    return tuple(map(int, tight.shape)), tight.tobytes()


def _rotations(mask: np.ndarray) -> list[np.ndarray]:
    base = _tight_mask(mask)
    seen = set()
    out = []
    for k in range(4):
        rot = _tight_mask(np.rot90(base, k))
        key = _mask_key(rot)
        if key not in seen:
            seen.add(key)
            out.append(rot)
    return out


def _d4_variants(mask: np.ndarray) -> list[np.ndarray]:
    base = _tight_mask(mask)
    seen = set()
    out = []
    for flipped in (base, np.fliplr(base)):
        for k in range(4):
            var = _tight_mask(np.rot90(flipped, k))
            key = _mask_key(var)
            if key not in seen:
                seen.add(key)
                out.append(var)
    return out


def _canonical_shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min(_mask_key(var) for var in _d4_variants(mask))


def _patch_from_mask(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _parse_mask(rows: list[str]) -> np.ndarray:
    mask = np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)
    return _tight_mask(mask)


SHAPE_LIBRARY = [
    _parse_mask(["111"]),
    _parse_mask(["10", "11"]),
    _parse_mask(["111", "010"]),
    _parse_mask(["110", "011"]),
    _parse_mask(["10", "10", "11"]),
    _parse_mask(["11", "11", "10"]),
    _parse_mask(["111", "110"]),
    _parse_mask(["010", "111", "001"]),
]

_UNIQUE_SHAPE_LIBRARY: list[np.ndarray] = []
_seen_shape_keys: set[tuple[tuple[int, int], bytes]] = set()
for _mask in SHAPE_LIBRARY:
    _key = _canonical_shape_key(_mask)
    if _key not in _seen_shape_keys:
        _seen_shape_keys.add(_key)
        _UNIQUE_SHAPE_LIBRARY.append(_mask)


def _extract_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    teal_coords = np.argwhere(g == TEAL)
    assert teal_coords.size > 0
    panel_box = _bbox_coords(teal_coords)
    pr0, pc0, pr1, pc1 = panel_box
    panel = _crop_bbox(g, panel_box)
    assert panel.shape[1] <= 8 and panel.shape[0] <= 13
    assert pc0 <= 1
    assert np.all(np.isin(panel, [TEAL, *LEGEND_COLORS]))

    legends = []
    for color in LEGEND_COLORS:
        comps = _components(panel == color, connectivity=8)
        assert len(comps) == 1
        coords = comps[0]
        mask = _coords_to_mask(coords)
        box = _bbox_coords(coords)
        legends.append(
            {
                "color": int(color),
                "bbox": (pr0 + box[0], pc0 + box[1], pr0 + box[2], pc0 + box[3]),
                "mask": mask,
                "canon": _canonical_shape_key(mask),
            }
        )

    legends.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    for idx, item in enumerate(legends):
        item["panel_order"] = idx
    assert len({item["canon"] for item in legends}) == 3

    legend_by_color = {item["color"]: item for item in legends}

    grey_components = _components(g == GREY, connectivity=4)
    assert 7 <= len(grey_components) <= 10
    objects = []
    for coords in grey_components:
        box = _bbox_coords(coords)
        mask = _coords_to_mask(coords)
        canon = _canonical_shape_key(mask)
        matches = [item for item in legends if item["canon"] == canon]
        assert len(matches) == 1
        objects.append(
            {
                "bbox": box,
                "mask": mask,
                "canon": canon,
                "match_color": matches[0]["color"],
                "panel_order": matches[0]["panel_order"],
            }
        )

    groups: dict[int, list[dict]] = defaultdict(list)
    for obj in objects:
        groups[int(obj["match_color"])].append(obj)

    assert set(groups) == set(LEGEND_COLORS)
    counts = {color: len(items) for color, items in groups.items()}
    assert len(set(counts.values())) == 3

    ordered_groups = []
    for color, items in groups.items():
        items = sorted(items, key=lambda item: (item["bbox"][0], item["bbox"][1]))
        ordered_groups.append(
            {
                "color": int(color),
                "objects": items,
                "count": len(items),
                "panel_order": legend_by_color[color]["panel_order"],
            }
        )
    ordered_groups.sort(key=lambda item: (-item["count"], item["panel_order"]))

    return {
        "grid": g,
        "panel_box": panel_box,
        "legends": legends,
        "legend_by_color": legend_by_color,
        "objects": objects,
        "ordered_groups": ordered_groups,
    }


def _assemble_output(scene: dict) -> np.ndarray:
    cols = []
    for group in scene["ordered_groups"]:
        patches = [_patch_from_mask(obj["mask"], group["color"]) for obj in group["objects"]]
        widths = [patch.shape[1] for patch in patches]
        heights = [patch.shape[0] for patch in patches]
        col_w = max(widths)
        col_h = sum(heights) + len(patches) - 1
        cols.append({"patches": patches, "width": col_w, "height": col_h})

    out_h = max(col["height"] for col in cols)
    out_w = sum(col["width"] for col in cols) + len(cols) - 1
    out = np.zeros((out_h, out_w), dtype=int)

    c0 = 0
    for col in cols:
        r0 = 0
        for patch in col["patches"]:
            h, w = patch.shape
            out[r0 : r0 + h, c0 : c0 + w] = np.maximum(out[r0 : r0 + h, c0 : c0 + w], patch)
            r0 += h + 1
        c0 += col["width"] + 1

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the legend panel, count matching gray shapes, then stack the groups by frequency.

    Input structure:
    - Black background with a teal rectangle on the left acting as a legend panel.
    - The panel contains exactly three separated colored shapes in blue, red, and green.
    - To the right are 7-10 gray 4-connected objects whose shapes match one legend entry up to rotation and/or reflection.

    Rule:
    - Extract the three legend shapes and use D4-equivalence (rotation/reflection) to classify every gray object by legend color.
    - Count how many gray objects belong to each legend color, then sort the three groups from largest count to smallest; ties are broken by the legend's top-to-bottom order in the panel.
    - Produce a compact output where each group becomes one column: recolor each matched object with its legend color, keep its input orientation, and stack the group's objects top-to-bottom with one black row between objects and one black column between groups.

    Key insight:
    - The teal panel is not copied into the answer; it is a shape dictionary whose D4 matches determine color labels, and those labels are only useful after counting because the counts decide the final column order.
    """
    scene = _extract_scene(np.asarray(grid, dtype=int))
    return _assemble_output(scene)


def _rand_partition(total: int, parts: int, min_each: int) -> list[int] | None:
    spare = total - parts * min_each
    if spare < 0:
        return None
    bins = [min_each] * parts
    if spare == 0:
        return bins
    extras = _RNG.multinomial(spare, np.full(parts, 1 / parts))
    return [bins[i] + int(extras[i]) for i in range(parts)]


def _same_mask(a: np.ndarray, b: np.ndarray) -> bool:
    return _mask_key(a) == _mask_key(b)


def _choose_legend_bases() -> list[np.ndarray]:
    by_area: dict[int, list[np.ndarray]] = defaultdict(list)
    for mask in _UNIQUE_SHAPE_LIBRARY:
        by_area[int(mask.sum())].append(mask)

    if _RNG.random() < 0.7:
        candidate_areas = [area for area, masks in by_area.items() if len(masks) >= 2]
        area = int(_RNG.choice(candidate_areas))
        picks = _RNG.choice(len(by_area[area]), size=2, replace=False)
        first_two = [by_area[area][int(idx)] for idx in picks]
        remaining = [
            mask
            for mask in _UNIQUE_SHAPE_LIBRARY
            if all(_canonical_shape_key(mask) != _canonical_shape_key(prev) for prev in first_two)
        ]
        third = remaining[int(_RNG.integers(len(remaining)))]
        chosen = first_two + [third]
    else:
        idxs = _RNG.choice(len(_UNIQUE_SHAPE_LIBRARY), size=3, replace=False)
        chosen = [_UNIQUE_SHAPE_LIBRARY[int(idx)] for idx in idxs]

    assert len({_canonical_shape_key(mask) for mask in chosen}) == 3
    return [_tight_mask(mask) for mask in chosen]


def _pick_group_variants(display_masks: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    colors = list(display_masks)
    picked: dict[int, np.ndarray] = {}
    special_color = None
    mirror_variants = {}

    for color in colors:
        display = display_masks[color]
        rots = _rotations(display)
        d4 = _d4_variants(display)
        mirror_only = [var for var in d4 if not any(_same_mask(var, rot) for rot in rots)]
        if mirror_only:
            mirror_variants[color] = mirror_only

    if mirror_variants and _RNG.random() < 0.6:
        special_color = int(_RNG.choice(list(mirror_variants)))
        picked[special_color] = mirror_variants[special_color][int(_RNG.integers(len(mirror_variants[special_color])))]

    if special_color is None:
        diffable = [
            color
            for color in colors
            if any(not _same_mask(var, display_masks[color]) for var in _d4_variants(display_masks[color]))
        ]
        if diffable:
            special_color = int(_RNG.choice(diffable))
            options = [var for var in _d4_variants(display_masks[special_color]) if not _same_mask(var, display_masks[special_color])]
            picked[special_color] = options[int(_RNG.integers(len(options)))]

    for color in colors:
        if color in picked:
            continue
        options = _d4_variants(display_masks[color])
        picked[color] = options[int(_RNG.integers(len(options)))]

    return {int(color): _tight_mask(mask) for color, mask in picked.items()}


def _place_patches(
    grid: np.ndarray,
    work_c0: int,
    items: list[dict],
) -> bool:
    h, w = grid.shape
    blocked = np.zeros((h, w), dtype=bool)

    def mark(box: tuple[int, int, int, int]) -> None:
        r0, c0, r1, c1 = box
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(h - 1, r1 + 1)
        cc1 = min(w - 1, c1 + 1)
        blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True

    order = sorted(range(len(items)), key=lambda i: (-int(items[i]["mask"].sum()), -items[i]["mask"].shape[0], -items[i]["mask"].shape[1]))
    for idx in order:
        mask = items[idx]["mask"]
        ph, pw = mask.shape
        candidates = [
            (r0, c0)
            for r0 in range(0, h - ph + 1)
            for c0 in range(work_c0, w - pw + 1)
            if not blocked[r0 : r0 + ph, c0 : c0 + pw].any()
        ]
        if not candidates:
            return False
        _RNG.shuffle(candidates)
        r0, c0 = candidates[0]
        patch = _patch_from_mask(mask, GREY)
        grid[r0 : r0 + ph, c0 : c0 + pw] = np.maximum(grid[r0 : r0 + ph, c0 : c0 + pw], patch)
        items[idx]["bbox"] = (r0, c0, r0 + ph - 1, c0 + pw - 1)
        mark(items[idx]["bbox"])

    return True


def _build_input() -> np.ndarray | None:
    base_masks = _choose_legend_bases()
    color_order = list(_RNG.permutation(np.array(LEGEND_COLORS)))
    display_masks = {int(color): _d4_variants(base_masks[i])[int(_RNG.integers(len(_d4_variants(base_masks[i]))))] for i, color in enumerate(color_order)}
    counts = list(VALID_COUNT_TRIPLETS[int(_RNG.integers(len(VALID_COUNT_TRIPLETS)))])
    _RNG.shuffle(counts)

    required_panel_h = sum(mask.shape[0] for mask in display_masks.values()) + 4
    panel_h_max = min(13, max(required_panel_h, 10) + 1)
    if required_panel_h > 13:
        return None
    panel_h = int(_RNG.integers(required_panel_h, panel_h_max + 1))
    panel_w = int(_RNG.integers(max(6, max(mask.shape[1] for mask in display_masks.values()) + 2), 9))

    H = int(_RNG.integers(max(20, panel_h + 2), 27))
    W = int(_RNG.integers(max(22, panel_w + 13), 29))
    panel_r0 = int(_RNG.integers(0, H - panel_h + 1))
    panel_c0 = int(_RNG.integers(0, 2))
    panel_r1, panel_c1 = panel_r0 + panel_h - 1, panel_c0 + panel_w - 1

    row_gaps = _rand_partition(panel_h - sum(display_masks[color].shape[0] for color in color_order), 4, 1)
    if row_gaps is None:
        return None

    grid = np.zeros((H, W), dtype=int)
    grid[panel_r0 : panel_r1 + 1, panel_c0 : panel_c1 + 1] = TEAL

    cur_r = panel_r0 + row_gaps[0]
    for gap_idx, color in enumerate(color_order):
        mask = display_masks[color]
        mh, mw = mask.shape
        c = int(panel_c0 + _RNG.integers(1, panel_w - mw))
        patch = _patch_from_mask(mask, color)
        grid[cur_r : cur_r + mh, c : c + mw] = np.where(patch, patch, grid[cur_r : cur_r + mh, c : c + mw])
        cur_r += mh + row_gaps[gap_idx + 1]

    group_variants = _pick_group_variants(display_masks)
    items = []
    for color, count in zip(color_order, counts):
        for _ in range(int(count)):
            items.append({"color": int(color), "mask": group_variants[int(color)].copy()})
    _RNG.shuffle(items)

    work_c0 = panel_c1 + 2
    if work_c0 >= W - 2:
        return None
    if not _place_patches(grid, work_c0, items):
        return None

    if not _validate_input(grid):
        return None
    return grid


def _validate_input(grid: np.ndarray) -> bool:
    try:
        scene = _extract_scene(grid)
    except AssertionError:
        return False

    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False
    if np.any((g != BG) & (g != BLUE) & (g != RED) & (g != GREEN) & (g != GREY) & (g != TEAL)):
        return False

    legends = scene["legend_by_color"]
    transformed_objects = 0
    reflection_objects = 0
    for obj in scene["objects"]:
        legend_mask = legends[obj["match_color"]]["mask"]
        if not _same_mask(obj["mask"], legend_mask):
            transformed_objects += 1
        if not any(_same_mask(obj["mask"], rot) for rot in _rotations(legend_mask)):
            reflection_objects += 1

    if transformed_objects == 0:
        return False

    boxes = [obj["bbox"] for obj in scene["objects"]]
    for i in range(len(boxes)):
        r0, c0, r1, c1 = boxes[i]
        if not (3 <= (r1 - r0 + 1) * (c1 - c0 + 1)):
            return False
        for j in range(i + 1, len(boxes)):
            s0, t0, s1, t1 = boxes[j]
            if not (r1 < s0 - 1 or s1 < r0 - 1 or c1 < t0 - 1 or t1 < c0 - 1):
                return False

    out = _assemble_output(scene)
    if out.ndim != 2 or max(out.shape) > 30:
        return False
    if not np.array_equal(main(g), out):
        return False

    counts = [group["count"] for group in scene["ordered_groups"]]
    if counts != sorted(counts, reverse=True):
        return False

    colors_in_out = set(map(int, np.unique(out))) - {BG}
    if colors_in_out != set(LEGEND_COLORS):
        return False

    # Make area-based shortcuts unsafe in many samples.
    legend_areas = [int(item["mask"].sum()) for item in scene["legends"]]
    if len(set(legend_areas)) == 3 and reflection_objects == 0 and transformed_objects < 2:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _build_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
