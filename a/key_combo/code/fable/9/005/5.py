import random
from collections import Counter, deque

import numpy as np


AXIS_COLOR = 4
QUERY_COLOR = 3
DIVIDER_COLOR = 5
BG_COLOR = 0
LEGEND_COLORS = (1, 2, 6, 7, 8, 9)

GRID_HEIGHT_RANGE = (14, 23)
GRID_WIDTH_RANGE = (15, 24)
NUM_TEMPLATES_RANGE = (2, 4)
QUERY_COUNT_RANGE = (3, 7)
QUERY_GAP = 1
GEN_ATTEMPTS = 300


def main(grid: np.ndarray) -> np.ndarray:
    """Complete each half-drawn query by mirroring it across the yellow axis, then recolor it from the legend.

    Input structure:
    - An L-shaped divider splits the grid into an upper-left legend region and a query region.
    - The legend region contains several complete monochrome sprites, each in its own color.
    - Every query object is a placeholder-colored half-shape attached to a short yellow axis marker.

    Rule:
    - For each query object, reflect the placeholder half across its yellow axis marker to recover the full shape.
    - Match that completed shape against the legend up to rotation and reflection, and take the legend sprite's color.
    - Redraw the completed shape in place using the matched color, erase the yellow axis marker, and leave the legend, divider, and background unchanged.

    Key insight:
    - The query shape must be completed before lookup; the visible half alone never identifies the correct legend sprite.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    divider_color, divider_r, divider_c = _find_divider(g, bg)

    legend = g[:divider_r, :divider_c]
    template_by_shape: dict[tuple[tuple[int, int], ...], int] = {}
    legend_colors: set[int] = set()
    for comp in _find_components(legend, bg=bg, connectivity=4, monochromatic=True):
        color = int(comp["color"])
        assert color not in {bg, divider_color, AXIS_COLOR, QUERY_COLOR}
        key = _canonical_shape(comp["coords"])
        assert key not in template_by_shape
        template_by_shape[key] = color
        legend_colors.add(color)
    assert template_by_shape

    role_colors = set(_colors_present(g, exclude={bg, divider_color, AXIS_COLOR}))
    query_colors = role_colors - legend_colors
    assert len(query_colors) == 1
    query_color = next(iter(query_colors))

    out = g.copy()
    query_components = _find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=False,
        colors=[query_color, AXIS_COLOR],
    )
    assert query_components

    for comp in query_components:
        coords = comp["coords"]
        vals = g[coords[:, 0], coords[:, 1]]
        shape_coords = coords[vals == query_color]
        axis_coords = coords[vals == AXIS_COLOR]
        assert len(shape_coords) > 0 and len(axis_coords) > 0

        full_coords = _complete_shape(shape_coords, axis_coords)
        key = _canonical_shape(full_coords)
        assert key in template_by_shape
        color = template_by_shape[key]

        out[axis_coords[:, 0], axis_coords[:, 1]] = bg
        out[full_coords[:, 0], full_coords[:, 1]] = color

    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")


def _gen_attempt() -> np.ndarray | None:
    num_templates = random.randint(*NUM_TEMPLATES_RANGE)
    templates = _make_templates(num_templates)
    if templates is None:
        return None

    colors = random.sample(list(LEGEND_COLORS), num_templates)
    legend_patches = []
    legend_width = 0
    legend_height = 0
    for template in templates:
        patch = _apply_transform(template["out_patch"], random.randrange(8))
        legend_patches.append(patch)
        legend_width += patch.shape[1]
        legend_height = max(legend_height, patch.shape[0])
    legend_width += num_templates - 1

    divider_r = legend_height + random.randint(1, 2)
    divider_c = legend_width

    min_h = max(GRID_HEIGHT_RANGE[0], divider_r + 6)
    min_w = max(GRID_WIDTH_RANGE[0], divider_c + 6)
    if min_h > GRID_HEIGHT_RANGE[1] or min_w > GRID_WIDTH_RANGE[1]:
        return None
    h = random.randint(min_h, GRID_HEIGHT_RANGE[1])
    w = random.randint(min_w, GRID_WIDTH_RANGE[1])

    out = np.full((h, w), BG_COLOR, dtype=int)
    inp = out.copy()

    col = 0
    template_canon_to_color: dict[tuple[tuple[int, int], ...], int] = {}
    for patch, color in zip(legend_patches, colors):
        max_row = divider_r - patch.shape[0]
        row = random.randint(0, max_row)
        _paint_patch(out, patch, row, col, {1: color})
        _paint_patch(inp, patch, row, col, {1: color})
        key = _canonical_shape(np.argwhere(patch == 1))
        template_canon_to_color[key] = color
        col += patch.shape[1] + 1

    out[divider_r, : divider_c + 1] = DIVIDER_COLOR
    out[: divider_r + 1, divider_c] = DIVIDER_COLOR
    inp[divider_r, : divider_c + 1] = DIVIDER_COLOR
    inp[: divider_r + 1, divider_c] = DIVIDER_COLOR

    min_queries = max(QUERY_COUNT_RANGE[0], num_templates)
    query_count = random.randint(min_queries, QUERY_COUNT_RANGE[1])
    assignments = list(range(num_templates))
    assignments.extend(random.randrange(num_templates) for _ in range(query_count - num_templates))
    random.shuffle(assignments)

    placed_boxes: list[tuple[int, int, int, int]] = []
    used_top_right = False
    axis_orientations: list[str] = []

    for i, template_index in enumerate(assignments):
        variants = _query_variants(templates[template_index])
        if not variants:
            return None

        desired_axis = None
        if i == 0:
            desired_axis = "v"
        elif i == 1:
            desired_axis = "h"
        if desired_axis is not None:
            filtered = [variant for variant in variants if variant["axis_orientation"] == desired_axis]
            if filtered:
                variants = filtered
        variant = random.choice(variants)
        axis_orientations.append(variant["axis_orientation"])

        input_patch = variant["inp_patch"]
        output_patch = variant["out_patch"]
        patch_h, patch_w = output_patch.shape

        regions = []
        if not used_top_right and random.random() < 0.55 and patch_h <= divider_r and divider_c + 1 + patch_w <= w:
            regions.append(("top_right", (0, divider_r, divider_c + 1, w)))
        regions.append(("below", (divider_r + 1, h, 0, w)))

        pos = None
        for region_name, region in regions:
            pos = _sample_position(region, patch_h, patch_w, placed_boxes, QUERY_GAP)
            if pos is not None:
                if region_name == "top_right":
                    used_top_right = True
                break
        if pos is None:
            return None
        r0, c0 = pos
        placed_boxes.append((r0, c0, r0 + patch_h - 1, c0 + patch_w - 1))

        _paint_patch(out, output_patch, r0, c0, {1: colors[template_index]})
        _paint_patch(inp, input_patch, r0, c0, {1: QUERY_COLOR, 2: AXIS_COLOR})

        half_key = _canonical_shape(np.argwhere(input_patch == 1))
        if half_key in template_canon_to_color:
            return None

    if "v" not in axis_orientations or "h" not in axis_orientations:
        return None
    if _get_bg(inp) != BG_COLOR:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def _make_templates(count: int) -> list[dict] | None:
    templates: list[dict] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    kinds = ["v", "h"] if count >= 2 else [random.choice(["v", "h"])]

    for forced_kind in kinds:
        template = _make_one_template(forced_kind, seen)
        if template is None:
            return None
        templates.append(template)
        seen.add(template["canon"])

    while len(templates) < count:
        template = _make_one_template(random.choice(["v", "h"]), seen)
        if template is None:
            return None
        templates.append(template)
        seen.add(template["canon"])
    return templates


def _make_one_template(kind: str, seen: set[tuple[tuple[int, int], ...]]) -> dict | None:
    for _ in range(200):
        if kind == "v":
            half_h = random.randint(2, 4)
            half_w = random.randint(2, 3)
            area = random.randint(3, min(5, half_h * half_w))
            half = _random_connected_mask((half_h, half_w), area, require_side="right")
            if half is None:
                continue
            full = np.zeros((half_h, 2 * half_w), dtype=int)
            coords = np.argwhere(half)
            full[coords[:, 0], coords[:, 1]] = 1
            full[coords[:, 0], 2 * half_w - 1 - coords[:, 1]] = 1
            inp = np.zeros_like(full, dtype=int)
            inp[coords[:, 0], coords[:, 1]] = 1
            r0, _, r1, _ = _bbox_from_mask(full == 1)
            inp[r0 : r1 + 1, half_w] = 2
        else:
            half_h = random.randint(2, 3)
            half_w = random.randint(2, 4)
            area = random.randint(3, min(5, half_h * half_w))
            half = _random_connected_mask((half_h, half_w), area, require_side="bottom")
            if half is None:
                continue
            full = np.zeros((2 * half_h, half_w), dtype=int)
            coords = np.argwhere(half)
            full[coords[:, 0], coords[:, 1]] = 1
            full[2 * half_h - 1 - coords[:, 0], coords[:, 1]] = 1
            inp = np.zeros_like(full, dtype=int)
            inp[coords[:, 0], coords[:, 1]] = 1
            _, c0, _, c1 = _bbox_from_mask(full == 1)
            inp[half_h, c0 : c1 + 1] = 2

        if int(np.sum(full == 1)) < 6:
            continue
        union_mask = (inp > 0) | (full > 0)
        bbox = _bbox_from_mask(union_mask)
        if bbox is None:
            continue
        inp_patch = _crop_bbox(inp, bbox)
        out_patch = _crop_bbox(full, bbox)
        full_bbox = _bbox_from_mask(out_patch == 1)
        if full_bbox is None:
            continue
        r0, c0, r1, c1 = full_bbox
        if r1 - r0 + 1 < 2 or c1 - c0 + 1 < 2:
            continue
        canon = _canonical_shape(np.argwhere(out_patch == 1))
        if canon in seen:
            continue
        return {"inp_patch": inp_patch, "out_patch": out_patch, "canon": canon}
    return None


def _query_variants(template: dict) -> list[dict]:
    variants = []
    seen_pairs = set()
    for t in range(8):
        inp_patch = _apply_transform(template["inp_patch"], t)
        out_patch = _apply_transform(template["out_patch"], t)
        key = (
            inp_patch.shape,
            inp_patch.tobytes(),
            out_patch.shape,
            out_patch.tobytes(),
        )
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        axis_coords = np.argwhere(inp_patch == 2)
        if len(axis_coords) == 0:
            continue
        axis_bbox = _bbox_coords(axis_coords)
        axis_orientation = "v" if axis_bbox[1] == axis_bbox[3] else "h"
        variants.append(
            {
                "inp_patch": inp_patch,
                "out_patch": out_patch,
                "axis_orientation": axis_orientation,
            }
        )
    return variants


def _sample_position(
    region: tuple[int, int, int, int],
    patch_h: int,
    patch_w: int,
    boxes: list[tuple[int, int, int, int]],
    gap: int,
) -> tuple[int, int] | None:
    r_min, r_max, c_min, c_max = region
    if r_min + patch_h > r_max or c_min + patch_w > c_max:
        return None
    for _ in range(200):
        r0 = random.randint(r_min, r_max - patch_h)
        c0 = random.randint(c_min, c_max - patch_w)
        bbox = (r0, c0, r0 + patch_h - 1, c0 + patch_w - 1)
        if all(not _boxes_too_close(bbox, other, gap) for other in boxes):
            return r0, c0
    return None


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _random_connected_mask(shape: tuple[int, int], area: int, require_side: str) -> np.ndarray | None:
    h, w = shape
    boundary = []
    if require_side == "right":
        boundary = [(r, w - 1) for r in range(h)]
    elif require_side == "bottom":
        boundary = [(h - 1, c) for c in range(w)]
    else:
        raise ValueError("bad boundary side")

    for _ in range(100):
        cells = {random.choice(boundary)}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != area:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        coords = np.argwhere(mask)
        r0, c0, r1, c1 = _bbox_coords(coords)
        if r1 - r0 + 1 < 2 or c1 - c0 + 1 < 2:
            continue
        if require_side == "right" and not np.any(mask[:, w - 1]):
            continue
        if require_side == "bottom" and not np.any(mask[h - 1, :]):
            continue
        return mask
    return None


def _complete_shape(shape_coords: np.ndarray, axis_coords: np.ndarray) -> np.ndarray:
    shape_box = _bbox_coords(shape_coords)
    axis_box = _bbox_coords(axis_coords)

    if axis_box[1] == axis_box[3]:
        axis_col = axis_box[1]
        if shape_box[3] < axis_col:
            mirrored = [(int(r), int(2 * axis_col - 1 - c)) for r, c in shape_coords]
        else:
            assert shape_box[1] > axis_col
            mirrored = [(int(r), int(2 * axis_col + 1 - c)) for r, c in shape_coords]
    else:
        axis_row = axis_box[0]
        if shape_box[2] < axis_row:
            mirrored = [(int(2 * axis_row - 1 - r), int(c)) for r, c in shape_coords]
        else:
            assert shape_box[0] > axis_row
            mirrored = [(int(2 * axis_row + 1 - r), int(c)) for r, c in shape_coords]

    full = {tuple(map(int, rc)) for rc in shape_coords.tolist()}
    full.update(mirrored)
    arr = np.array(sorted(full), dtype=int)
    return arr


def _find_divider(grid: np.ndarray, bg: int) -> tuple[int, int, int]:
    for color in _colors_present(grid, exclude={bg}):
        coords = np.argwhere(grid == color)
        if len(coords) == 0:
            continue
        r0, c0, r1, c1 = _bbox_coords(coords)
        expected = np.zeros_like(grid, dtype=bool)
        expected[r1, : c1 + 1] = True
        expected[: r1 + 1, c1] = True
        if r0 == 0 and c0 == 0 and np.array_equal(grid == color, expected):
            return int(color), int(r1), int(c1)
    raise AssertionError("divider not found")


def _paint_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, color_map: dict[int, int]) -> None:
    h, w = patch.shape
    view = grid[r0 : r0 + h, c0 : c0 + w]
    for src, dst in color_map.items():
        view[patch == src] = dst


def _apply_transform(arr: np.ndarray, t: int) -> np.ndarray:
    rot = np.rot90(arr, t % 4)
    if t >= 4:
        rot = np.fliplr(rot)
    return rot.copy()


def _canonical_shape(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    coords = np.asarray(coords, dtype=int)
    assert len(coords) > 0
    base = np.zeros((coords[:, 0].max() - coords[:, 0].min() + 1, coords[:, 1].max() - coords[:, 1].min() + 1), dtype=bool)
    shifted = coords.copy()
    shifted[:, 0] -= shifted[:, 0].min()
    shifted[:, 1] -= shifted[:, 1].min()
    base[shifted[:, 0], shifted[:, 1]] = True

    variants = []
    for k in range(4):
        rot = np.rot90(base, k)
        for cand in (rot, np.fliplr(rot)):
            pts = tuple(map(tuple, np.argwhere(cand).tolist()))
            variants.append(pts)
    return min(variants)


def _find_components(
    grid: np.ndarray,
    *,
    bg: int,
    connectivity: int = 4,
    monochromatic: bool = True,
    colors: list[int] | tuple[int, ...] | None = None,
) -> list[dict]:
    if monochromatic:
        if colors is None:
            colors = _colors_present(grid, exclude={bg})
        out = []
        for color in colors:
            mask = grid == color
            for coords in _components_from_mask(mask, connectivity):
                out.append(
                    {
                        "color": int(color),
                        "coords": coords,
                        "bbox": _bbox_coords(coords),
                    }
                )
        return out

    mask = grid != bg
    if colors is not None:
        mask &= np.isin(grid, list(colors))
    return [
        {
            "color": None,
            "coords": coords,
            "bbox": _bbox_coords(coords),
        }
        for coords in _components_from_mask(mask, connectivity)
    ]


def _components_from_mask(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
    else:
        directions = (
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1),
        )

    comps = []
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
                for dr, dc in directions:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return None
    return _bbox_coords(coords)


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(arr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return arr[r0 : r1 + 1, c0 : c1 + 1].copy()
