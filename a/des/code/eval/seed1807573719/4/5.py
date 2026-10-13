import itertools
from collections import Counter, deque

import numpy as np


BG_COLOR = 0
PANEL_COLOR = 5
ARROW_COLOR = 9
GENERATION_ATTEMPTS = 300
GRID_SIDE_RANGE = (24, 33)
OBJECT_COUNT_CHOICES = (1, 2, 3)
TEMPLATE_COUNT_CHOICES = (3, 4)
OBJECT_MARGIN_RANGE = (2, 5)
PANEL_CORNERS = ("tl", "tr", "bl", "br")
PALETTE_COLORS = [1, 2, 3, 4, 6, 7, 8]
SIDE_OFFSETS = (-1, 1)


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 4:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        neighbors = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            coords = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for dr, dc in neighbors:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _mask_from_coords(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask, (r0, c0, r1, c1)


def _touches_local_boundary(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    rows = coords[:, 0]
    cols = coords[:, 1]
    return bool(
        np.any(rows == 0)
        or np.any(rows == h - 1)
        or np.any(cols == 0)
        or np.any(cols == w - 1)
    )


def _separator_spans(lines: list[int], size: int) -> list[tuple[int, int]]:
    lines = sorted(set(lines))
    assert lines and lines[0] == 0 and lines[-1] == size - 1
    spans = []
    for a, b in zip(lines, lines[1:]):
        if b - a > 1:
            spans.append((a + 1, b - 1))
    return spans


def _rect_distance(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    dr = 0 if not (ar1 < br0 or br1 < ar0) else max(br0 - ar1, ar0 - br1)
    dc = 0 if not (ac1 < bc0 or bc1 < ac0) else max(bc0 - ac1, ac0 - bc1)
    return int(dr + dc)


def _boxes_too_close(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0)


def _canonical_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    variants = [(m.shape, m.tobytes()) for m in (np.rot90(mask, k) for k in range(4))]
    return min(variants)


def _has_four_distinct_rotations(mask: np.ndarray) -> bool:
    return len({(m.shape, m.tobytes()) for m in (np.rot90(mask, k) for k in range(4))}) == 4


def _build_shape_library() -> list[np.ndarray]:
    classes: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}
    for h, w in ((3, 3), (3, 4), (4, 3), (4, 4)):
        cells = h * w
        for area in (4, 5, 6):
            for picks in itertools.combinations(range(cells), area):
                mask = np.zeros((h, w), dtype=bool)
                for idx in picks:
                    mask[idx // w, idx % w] = True
                comps = _components(mask, connectivity=4)
                if len(comps) != 1:
                    continue
                cropped, _ = _mask_from_coords(comps[0])
                ch, cw = cropped.shape
                if min(ch, cw) < 2 or max(ch, cw) > 4:
                    continue
                if not _has_four_distinct_rotations(cropped):
                    continue
                key = _canonical_key(cropped)
                classes.setdefault(key, cropped.copy())
    library = [classes[key] for key in sorted(classes)]
    assert len(library) >= 12
    return library


SHAPE_LIBRARY = _build_shape_library()


def _find_panel_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == PANEL_COLOR)
    assert len(coords) > 0, "panel border must exist"
    return _bbox(coords)


def _extract_templates(
    grid: np.ndarray, panel_bbox: tuple[int, int, int, int], bg: int
) -> list[dict]:
    r0, c0, r1, c1 = panel_bbox
    panel = grid[r0 : r1 + 1, c0 : c1 + 1]
    sep_rows = [r for r in range(panel.shape[0]) if np.all(panel[r] == PANEL_COLOR)]
    sep_cols = [c for c in range(panel.shape[1]) if np.all(panel[:, c] == PANEL_COLOR)]
    row_spans = _separator_spans(sep_rows, panel.shape[0])
    col_spans = _separator_spans(sep_cols, panel.shape[1])

    templates = []
    for rs, re in row_spans:
        for cs, ce in col_spans:
            cell = panel[rs : re + 1, cs : ce + 1]
            colors = [int(v) for v in np.unique(cell) if int(v) not in {bg, PANEL_COLOR}]
            assert len(colors) == 1, "each panel cell must contain exactly one template color"
            color = colors[0]
            mask = cell == color
            coords = np.argwhere(mask)
            tight, _ = _mask_from_coords(coords)
            templates.append({"color": color, "mask": tight})
    assert templates, "panel must expose at least one template"
    return templates


def _find_shapes(
    grid: np.ndarray, panel_bbox: tuple[int, int, int, int], bg: int
) -> list[dict]:
    mask = (grid != bg) & (grid != PANEL_COLOR) & (grid != ARROW_COLOR)
    pr0, pc0, pr1, pc1 = panel_bbox
    mask[pr0 : pr1 + 1, pc0 : pc1 + 1] = False
    comps = _components(mask, connectivity=8)
    shapes = []
    for coords in comps:
        values = {int(grid[r, c]) for r, c in coords}
        assert len(values) == 1, "each large shape must be monochrome"
        obj_mask, bbox = _mask_from_coords(coords)
        inv = ~obj_mask
        holes = []
        for hole_coords in _components(inv, connectivity=4):
            if _touches_local_boundary(hole_coords, obj_mask.shape):
                continue
            hole_mask, hole_bbox_local = _mask_from_coords(hole_coords)
            hr0, hc0, hr1, hc1 = hole_bbox_local
            r0, c0, _, _ = bbox
            holes.append(
                {
                    "mask": hole_mask,
                    "bbox_global": (r0 + hr0, c0 + hc0, r0 + hr1, c0 + hc1),
                }
            )
        assert len(holes) == 1, "each large shape must contain exactly one hole"
        shapes.append(
            {
                "color": values.pop(),
                "coords": coords,
                "bbox": bbox,
                "hole_mask": holes[0]["mask"],
                "hole_bbox": holes[0]["bbox_global"],
            }
        )
    assert shapes, "expected at least one large shape"
    return shapes


def _find_arrows(grid: np.ndarray) -> list[dict]:
    arrows = []
    for coords in _components(grid == ARROW_COLOR, connectivity=8):
        assert len(coords) == 3, "each arrow must be a 3-pixel L shape"
        r0, c0, r1, c1 = _bbox(coords)
        assert (r1 - r0, c1 - c0) == (1, 1), "arrow must occupy a 2x2 box with one missing corner"
        arrows.append({"coords": coords})
    return arrows


def _arrow_side(shape_bbox: tuple[int, int, int, int], arrow_bbox: tuple[int, int, int, int]) -> str:
    sr0, sc0, sr1, sc1 = shape_bbox
    ar0, ac0, ar1, ac1 = arrow_bbox
    if ar1 < sr0:
        return "up"
    if ar0 > sr1:
        return "down"
    if ac1 < sc0:
        return "left"
    if ac0 > sc1:
        return "right"
    raise AssertionError("arrow must sit strictly outside its shape")


def _assign_arrows_to_shapes(shapes: list[dict], arrows: list[dict]) -> None:
    assert len(shapes) == len(arrows), "arrow count must match shape count"
    n = len(shapes)
    distances = np.zeros((n, n), dtype=int)
    for ai, arrow in enumerate(arrows):
        arrow_bbox = _bbox(arrow["coords"])
        for si, shape in enumerate(shapes):
            distances[ai, si] = _rect_distance(arrow_bbox, shape["bbox"])
    best_perm = min(
        itertools.permutations(range(n)),
        key=lambda perm: sum(int(distances[ai, perm[ai]]) for ai in range(n)),
    )
    for ai, si in enumerate(best_perm):
        shapes[si]["direction"] = _arrow_side(shapes[si]["bbox"], _bbox(arrows[ai]["coords"]))


def _match_template(hole_mask: np.ndarray, templates: list[dict]) -> tuple[dict, int, np.ndarray]:
    matches = []
    for template in templates:
        for k in range(4):
            rotated = np.rot90(template["mask"], k)
            if rotated.shape == hole_mask.shape and np.array_equal(rotated, hole_mask):
                matches.append((template, k, rotated))
    assert len(matches) == 1, "hole shape must match exactly one rotated template"
    return matches[0]


def _stamp_origin(
    shape_bbox: tuple[int, int, int, int],
    hole_bbox: tuple[int, int, int, int],
    stamp_shape: tuple[int, int],
    direction: str,
) -> tuple[int, int]:
    sr0, sc0, sr1, sc1 = shape_bbox
    hr0, hc0, hr1, hc1 = hole_bbox
    stamp_h, stamp_w = stamp_shape
    hole_center_r = (hr0 + hr1) / 2.0
    hole_center_c = (hc0 + hc1) / 2.0
    top = int(round(hole_center_r - (stamp_h - 1) / 2.0))
    left = int(round(hole_center_c - (stamp_w - 1) / 2.0))
    if direction == "up":
        top = sr0 - 1 - stamp_h
    elif direction == "down":
        top = sr1 + 2
    elif direction == "left":
        left = sc0 - 1 - stamp_w
    elif direction == "right":
        left = sc1 + 2
    else:
        raise AssertionError("unknown direction")
    return top, left


def _paste_stamp(
    out: np.ndarray,
    panel_bbox: tuple[int, int, int, int],
    stamp_mask: np.ndarray,
    color: int,
    top: int,
    left: int,
) -> None:
    pr0, pc0, pr1, pc1 = panel_bbox
    h, w = out.shape
    for rr, cc in np.argwhere(stamp_mask):
        r = top + int(rr)
        c = left + int(cc)
        if 0 <= r < h and 0 <= c < w and not (pr0 <= r <= pr1 and pc0 <= c <= pc1):
            out[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each large shape's internal hole to a template in the corner stamp panel,
    rotate the template the same way, project it outside the shape along the nearby
    maroon arrow direction, erase the arrows, and keep the panel unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _mode_color(g)
    panel_bbox = _find_panel_bbox(g)
    templates = _extract_templates(g, panel_bbox, bg)
    shapes = _find_shapes(g, panel_bbox, bg)
    arrows = _find_arrows(g)
    _assign_arrows_to_shapes(shapes, arrows)

    out = g.copy()
    out[out == ARROW_COLOR] = bg

    for shape in shapes:
        template, _rot, stamp_mask = _match_template(shape["hole_mask"], templates)
        top, left = _stamp_origin(shape["bbox"], shape["hole_bbox"], stamp_mask.shape, shape["direction"])
        _paste_stamp(out, panel_bbox, stamp_mask, int(template["color"]), top, left)

    return out


def _randint(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(lo, hi))


def _panel_layout_options(n_templates: int) -> list[tuple[int, int]]:
    if n_templates == 3:
        return [(1, 3), (3, 1)]
    if n_templates == 4:
        return [(2, 2), (1, 4), (4, 1)]
    raise AssertionError("unsupported template count")


def _draw_panel(
    grid: np.ndarray,
    corner: str,
    templates: list[dict],
    rng: np.random.Generator,
) -> tuple[int, int, int, int]:
    rows, cols = _panel_layout_options(len(templates))[_randint(rng, 0, len(_panel_layout_options(len(templates))))]
    max_h = max(int(t["mask"].shape[0]) for t in templates)
    max_w = max(int(t["mask"].shape[1]) for t in templates)
    cell_h = max_h + _randint(rng, 2, 4)
    cell_w = max_w + _randint(rng, 2, 4)
    panel_h = rows * cell_h + rows + 1
    panel_w = cols * cell_w + cols + 1
    h, w = grid.shape
    assert panel_h <= h and panel_w <= w

    if corner == "tl":
        r0, c0 = 0, 0
    elif corner == "tr":
        r0, c0 = 0, w - panel_w
    elif corner == "bl":
        r0, c0 = h - panel_h, 0
    elif corner == "br":
        r0, c0 = h - panel_h, w - panel_w
    else:
        raise AssertionError("bad corner")
    r1 = r0 + panel_h - 1
    c1 = c0 + panel_w - 1

    panel = np.full((panel_h, panel_w), BG_COLOR, dtype=int)
    for r in range(0, panel_h, cell_h + 1):
        panel[r, :] = PANEL_COLOR
    panel[-1, :] = PANEL_COLOR
    for c in range(0, panel_w, cell_w + 1):
        panel[:, c] = PANEL_COLOR
    panel[:, -1] = PANEL_COLOR

    idx = 0
    for rr in range(rows):
        for cc in range(cols):
            cell_r0 = rr * (cell_h + 1) + 1
            cell_c0 = cc * (cell_w + 1) + 1
            template = templates[idx]
            idx += 1
            mask = template["mask"]
            th, tw = mask.shape
            off_r = cell_r0 + (cell_h - th) // 2
            off_c = cell_c0 + (cell_w - tw) // 2
            patch = panel[off_r : off_r + th, off_c : off_c + tw]
            patch[mask] = int(template["color"])

    grid[r0 : r1 + 1, c0 : c1 + 1] = panel
    return (r0, c0, r1, c1)


def _object_array(hole_mask: np.ndarray, color: int, margins: tuple[int, int, int, int]) -> np.ndarray:
    top_m, bottom_m, left_m, right_m = margins
    hh, hw = hole_mask.shape
    obj = np.full((top_m + hh + bottom_m, left_m + hw + right_m), color, dtype=int)
    hole = obj[top_m : top_m + hh, left_m : left_m + hw]
    hole[hole_mask] = BG_COLOR
    return obj


def _arrow_cells(
    obj_top: int,
    obj_left: int,
    obj_shape: tuple[int, int],
    hole_shape: tuple[int, int],
    margins: tuple[int, int, int, int],
    direction: str,
    side_offset: int,
) -> list[tuple[int, int]]:
    obj_h, obj_w = obj_shape
    hh, hw = hole_shape
    top_m, _bottom_m, left_m, _right_m = margins
    anchor_r = obj_top + top_m + (hh - 1) // 2
    anchor_c = obj_left + left_m + (hw - 1) // 2

    if direction == "up":
        return [(obj_top - 2, anchor_c), (obj_top - 1, anchor_c), (obj_top - 2, anchor_c + side_offset)]
    if direction == "down":
        return [
            (obj_top + obj_h, anchor_c),
            (obj_top + obj_h + 1, anchor_c),
            (obj_top + obj_h + 1, anchor_c + side_offset),
        ]
    if direction == "left":
        return [(anchor_r, obj_left - 2), (anchor_r, obj_left - 1), (anchor_r + side_offset, obj_left - 2)]
    if direction == "right":
        return [
            (anchor_r, obj_left + obj_w),
            (anchor_r, obj_left + obj_w + 1),
            (anchor_r + side_offset, obj_left + obj_w + 1),
        ]
    raise AssertionError("bad direction")


def _coords_in_bounds(coords: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return all(0 <= r < h and 0 <= c < w for r, c in coords)


def _visible_stamp_pixels(
    panel_bbox: tuple[int, int, int, int],
    grid_shape: tuple[int, int],
    stamp_mask: np.ndarray,
    top: int,
    left: int,
) -> int:
    pr0, pc0, pr1, pc1 = panel_bbox
    h, w = grid_shape
    visible = 0
    for rr, cc in np.argwhere(stamp_mask):
        r = top + int(rr)
        c = left + int(cc)
        if 0 <= r < h and 0 <= c < w and not (pr0 <= r <= pr1 and pc0 <= c <= pc1):
            visible += 1
    return visible


def _clipped_bbox(
    top: int, left: int, shape: tuple[int, int], grid_shape: tuple[int, int]
) -> tuple[int, int, int, int] | None:
    h, w = shape
    gh, gw = grid_shape
    r0 = max(0, top)
    c0 = max(0, left)
    r1 = min(gh - 1, top + h - 1)
    c1 = min(gw - 1, left + w - 1)
    if r0 > r1 or c0 > c1:
        return None
    return (r0, c0, r1, c1)


def _build_expected_output(
    grid: np.ndarray,
    panel_bbox: tuple[int, int, int, int],
    cases: list[dict],
) -> np.ndarray:
    out = grid.copy()
    out[out == ARROW_COLOR] = BG_COLOR
    for case in cases:
        top, left = _stamp_origin(case["shape_bbox"], case["hole_bbox"], case["stamp_mask"].shape, case["direction"])
        _paste_stamp(out, panel_bbox, case["stamp_mask"], int(case["template_color"]), top, left)
    return out


def _gen_attempt(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    num_objects = OBJECT_COUNT_CHOICES[_randint(rng, 0, len(OBJECT_COUNT_CHOICES))]
    num_templates = TEMPLATE_COUNT_CHOICES[_randint(rng, 0, len(TEMPLATE_COUNT_CHOICES))]
    if num_templates < num_objects or num_objects + num_templates > len(PALETTE_COLORS):
        return None

    template_masks = [SHAPE_LIBRARY[i].copy() for i in rng.choice(len(SHAPE_LIBRARY), size=num_templates, replace=False)]
    template_colors = [PALETTE_COLORS[i] for i in rng.choice(len(PALETTE_COLORS), size=num_templates, replace=False)]
    remaining_colors = [c for c in PALETTE_COLORS if c not in template_colors]
    object_colors = [remaining_colors[i] for i in rng.choice(len(remaining_colors), size=num_objects, replace=False)]

    templates = [{"mask": m, "color": c} for m, c in zip(template_masks, template_colors)]
    chosen_template_ids = rng.choice(num_templates, size=num_objects, replace=False)

    object_specs = []
    for i in range(num_objects):
        tidx = int(chosen_template_ids[i])
        rot = _randint(rng, 0, 4)
        hole_mask = np.rot90(templates[tidx]["mask"], rot)
        margins = tuple(_randint(rng, *OBJECT_MARGIN_RANGE) for _ in range(4))
        direction = ("up", "down", "left", "right")[_randint(rng, 0, 4)]
        object_specs.append(
            {
                "template_idx": tidx,
                "template_color": int(templates[tidx]["color"]),
                "hole_mask": hole_mask,
                "margins": margins,
                "direction": direction,
                "color": int(object_colors[i]),
                "clip_goal": bool(rng.random() < 0.35),
            }
        )

    max_obj_h = max(sum(spec["margins"][:2]) + spec["hole_mask"].shape[0] for spec in object_specs)
    max_obj_w = max(sum(spec["margins"][2:]) + spec["hole_mask"].shape[1] for spec in object_specs)

    min_side = max(24, max_obj_h + 8, max_obj_w + 8)
    h = _randint(rng, min_side, GRID_SIDE_RANGE[1] + 1)
    w = _randint(rng, min_side, GRID_SIDE_RANGE[1] + 1)
    grid = np.full((h, w), BG_COLOR, dtype=int)

    corner = PANEL_CORNERS[_randint(rng, 0, len(PANEL_CORNERS))]
    try:
        panel_bbox = _draw_panel(grid, corner, templates, rng)
    except AssertionError:
        return None

    blocked_boxes = [panel_bbox]
    reserved_output_boxes = [panel_bbox]
    cases = []

    for spec in object_specs:
        obj = _object_array(spec["hole_mask"], spec["color"], spec["margins"])
        obj_h, obj_w = obj.shape
        direction = spec["direction"]
        hole_h, hole_w = spec["hole_mask"].shape
        stamp_mask = spec["hole_mask"]

        if direction == "up":
            row_min, row_max = 2, h - obj_h
            col_min, col_max = 0, w - obj_w
        elif direction == "down":
            row_min, row_max = 0, h - obj_h - 2
            col_min, col_max = 0, w - obj_w
        elif direction == "left":
            row_min, row_max = 0, h - obj_h
            col_min, col_max = 2, w - obj_w
        else:
            row_min, row_max = 0, h - obj_h
            col_min, col_max = 0, w - obj_w - 2

        if row_min > row_max or col_min > col_max:
            return None

        placed = False
        for _ in range(140):
            if direction == "up" and spec["clip_goal"]:
                obj_top = _randint(rng, row_min, min(row_max, 5) + 1)
            elif direction == "down" and spec["clip_goal"]:
                obj_top = _randint(rng, max(row_min, row_max - 3), row_max + 1)
            else:
                obj_top = _randint(rng, row_min, row_max + 1)

            if direction == "left" and spec["clip_goal"]:
                obj_left = _randint(rng, col_min, min(col_max, 5) + 1)
            elif direction == "right" and spec["clip_goal"]:
                obj_left = _randint(rng, max(col_min, col_max - 3), col_max + 1)
            else:
                obj_left = _randint(rng, col_min, col_max + 1)

            obj_bbox = (obj_top, obj_left, obj_top + obj_h - 1, obj_left + obj_w - 1)
            if any(_boxes_too_close(obj_bbox, box, gap=1) for box in blocked_boxes):
                continue

            for side_offset in rng.permutation(SIDE_OFFSETS):
                arrow = _arrow_cells(
                    obj_top,
                    obj_left,
                    obj.shape,
                    spec["hole_mask"].shape,
                    spec["margins"],
                    direction,
                    int(side_offset),
                )
                if not _coords_in_bounds(arrow, grid.shape):
                    continue
                if any(grid[r, c] != BG_COLOR for r, c in arrow):
                    continue
                arrow_bbox = _bbox(np.array(arrow, dtype=int))
                union_bbox = (
                    min(obj_bbox[0], arrow_bbox[0]),
                    min(obj_bbox[1], arrow_bbox[1]),
                    max(obj_bbox[2], arrow_bbox[2]),
                    max(obj_bbox[3], arrow_bbox[3]),
                )
                if any(_boxes_too_close(union_bbox, box, gap=1) for box in blocked_boxes):
                    continue

                top_m, _bottom_m, left_m, _right_m = spec["margins"]
                hole_bbox = (
                    obj_top + top_m,
                    obj_left + left_m,
                    obj_top + top_m + hole_h - 1,
                    obj_left + left_m + hole_w - 1,
                )
                stamp_top, stamp_left = _stamp_origin(obj_bbox, hole_bbox, stamp_mask.shape, direction)
                visible = _visible_stamp_pixels(panel_bbox, grid.shape, stamp_mask, stamp_top, stamp_left)
                if visible == 0:
                    continue
                clipped = _clipped_bbox(stamp_top, stamp_left, stamp_mask.shape, grid.shape)
                if clipped is not None and any(_boxes_too_close(clipped, box, gap=0) for box in reserved_output_boxes):
                    continue

                obj_rows, obj_cols = np.indices(obj.shape)
                for rr, cc in zip(obj_rows.ravel(), obj_cols.ravel()):
                    value = int(obj[rr, cc])
                    if value != BG_COLOR:
                        grid[obj_top + int(rr), obj_left + int(cc)] = value
                for r, c in arrow:
                    grid[r, c] = ARROW_COLOR

                blocked_boxes.append(union_bbox)
                if clipped is not None:
                    reserved_output_boxes.append(clipped)
                cases.append(
                    {
                        "shape_bbox": obj_bbox,
                        "hole_bbox": hole_bbox,
                        "stamp_mask": stamp_mask,
                        "direction": direction,
                        "template_color": spec["template_color"],
                    }
                )
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    expected = _build_expected_output(grid, panel_bbox, cases)
    return grid, expected


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_ATTEMPTS):
        built = _gen_attempt(rng)
        if built is None:
            continue
        grid, expected = built
        try:
            if np.array_equal(main(grid), expected):
                return grid
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid input")
