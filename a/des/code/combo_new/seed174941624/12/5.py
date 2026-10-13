import random
from collections import deque

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bool_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _monochromatic_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rows = [p[0] for p in pts]
            cols = [p[1] for p in pts]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for pr, pc in pts:
                mask[pr - r0, pc - c0] = True
            comps.append(
                {
                    "color": color,
                    "coords": pts,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                }
            )
    return comps


def _normalize_shape(coords) -> tuple[tuple[int, int], ...]:
    pts = [(int(r), int(c)) for r, c in coords]
    min_r = min(r for r, _ in pts)
    min_c = min(c for _, c in pts)
    return tuple(sorted((r - min_r, c - min_c) for r, c in pts))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _rotate_shape_once(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    h, _w = _shape_dims(shape)
    return _normalize_shape((c, h - 1 - r) for r, c in shape)


def _rotate_shape(shape: tuple[tuple[int, int], ...], turns: int) -> tuple[tuple[int, int], ...]:
    out = shape
    for _ in range(turns % 4):
        out = _rotate_shape_once(out)
    return out


def _shape_variants(shape: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    out = []
    cur = shape
    seen = set()
    for _ in range(4):
        if cur not in seen:
            seen.add(cur)
            out.append(cur)
        cur = _rotate_shape_once(cur)
    return out


def _canonical_rotation_key(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return min(_shape_variants(shape))


def _hole_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    holes = []
    for comp in _bool_components(~mask):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        holes.append(comp)
    return holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each object's enclosed hole with the color of the edge template whose shape matches that hole.

    Input structure:
    - The background is the dominant color.
    - Several larger monochrome 4-connected objects each contain exactly one enclosed background hole.
    - Separate small monochrome templates sit near the grid edges; each template's shape matches one hole up to rotation.

    Rule:
    - Split foreground into monochrome connected components.
    - Components with one enclosed hole are the large objects; components without enclosed holes are the small templates.
    - Match every object's hole shape to a template shape under 0/90/180/270 degree rotation, fill the hole with that template's color, and erase all templates to background.

    Key insight:
    - The decisive feature is the negative-space shape of each object's hole, not the object's outer silhouette.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    comps = _monochromatic_components(g, bg)
    assert comps, "expected at least one foreground component"

    objects = []
    templates = []
    for comp in comps:
        holes = _hole_components(comp["mask"])
        if len(holes) == 1:
            objects.append(
                {
                    "color": comp["color"],
                    "hole_coords": [(comp["bbox"][0] + r, comp["bbox"][1] + c) for r, c in holes[0]],
                    "hole_shape": _normalize_shape(holes[0]),
                }
            )
        else:
            templates.append(
                {
                    "color": comp["color"],
                    "coords": comp["coords"],
                    "shape": _normalize_shape(np.argwhere(comp["mask"])),
                }
            )

    assert objects, "expected large hollow objects"
    assert templates, "expected edge templates"

    template_by_shape = {}
    for template in templates:
        key = _canonical_rotation_key(template["shape"])
        assert key not in template_by_shape, "template shapes must be unique up to rotation"
        template_by_shape[key] = template

    out = g.copy()
    for template in templates:
        for r, c in template["coords"]:
            out[r, c] = bg

    for obj in objects:
        key = _canonical_rotation_key(obj["hole_shape"])
        assert key in template_by_shape, "every hole must match a template"
        fill_color = int(template_by_shape[key]["color"])
        for r, c in obj["hole_coords"]:
            out[r, c] = fill_color

    return out


BG = 0
GRID_H_RANGE = (22, 30)
GRID_W_RANGE = (22, 30)
OBJECT_COUNT_RANGE = (2, 4)
EDGE_SIDES = ("top", "bottom", "left", "right")
OBJECT_EDGE_MARGIN = 3
PLACEMENT_PAD = 1
MAX_SHAPE_ATTEMPTS = 80
MAX_GEN_ATTEMPTS = 400


def _shape_to_mask(shape: tuple[tuple[int, int], ...]) -> np.ndarray:
    h, w = _shape_dims(shape)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in shape:
        mask[r, c] = True
    return mask


def _random_connected_shape(area: int, rng: random.Random, max_span: int = 4):
    for _ in range(MAX_SHAPE_ATTEMPTS):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) not in cells:
                        frontier.add((nr, nc))
            frontier = list(frontier)
            rng.shuffle(frontier)
            chosen = None
            for nr, nc in frontier:
                trial = _normalize_shape(list(cells) + [(nr, nc)])
                th, tw = _shape_dims(trial)
                if max(th, tw) <= max_span:
                    chosen = (nr, nc)
                    break
            if chosen is None:
                break
            cells.add(chosen)
        if len(cells) != area:
            continue
        shape = _normalize_shape(cells)
        h, w = _shape_dims(shape)
        if max(h, w) > max_span:
            continue
        if area >= 4 and h * w == area and rng.random() < 0.65:
            continue
        return shape
    return None


def _sample_unique_shapes(count: int, rng: random.Random):
    shapes = []
    used = set()
    for _ in range(count):
        for _ in range(MAX_SHAPE_ATTEMPTS * 3):
            area = rng.randint(2, 6)
            shape = _random_connected_shape(area, rng)
            if shape is None:
                continue
            key = _canonical_rotation_key(shape)
            if key in used:
                continue
            used.add(key)
            shapes.append(shape)
            break
        else:
            return None
    return shapes


def _paint_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    mask = patch != bg
    grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]][mask] = patch[mask]


def _bbox_free(occ: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> bool:
    oh, ow = occ.shape
    return not occ[max(0, r0 - pad) : min(oh, r0 + h + pad), max(0, c0 - pad) : min(ow, c0 + w + pad)].any()


def _mark_bbox(occ: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> None:
    oh, ow = occ.shape
    occ[max(0, r0 - pad) : min(oh, r0 + h + pad), max(0, c0 - pad) : min(ow, c0 + w + pad)] = True


def _make_object_patch(
    hole_shape: tuple[tuple[int, int], ...],
    object_color: int,
    fill_color: int,
    rng: random.Random,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    hole_h, hole_w = _shape_dims(hole_shape)
    margin_top = rng.randint(1, 2)
    margin_bottom = rng.randint(1, 2)
    margin_left = rng.randint(1, 2)
    margin_right = rng.randint(1, 2)
    base_h = hole_h + margin_top + margin_bottom
    base_w = hole_w + margin_left + margin_right

    extra_top = rng.randint(0, 1)
    extra_bottom = rng.randint(0, 1)
    extra_left = rng.randint(0, 1)
    extra_right = rng.randint(0, 1)
    total_h = base_h + extra_top + extra_bottom
    total_w = base_w + extra_left + extra_right

    mask = np.zeros((total_h, total_w), dtype=bool)
    base_r0 = extra_top
    base_c0 = extra_left
    mask[base_r0 : base_r0 + base_h, base_c0 : base_c0 + base_w] = True

    if extra_top and rng.random() < 0.7:
        width = rng.randint(max(1, base_w // 3), base_w)
        start = rng.randint(0, base_w - width)
        mask[:extra_top, base_c0 + start : base_c0 + start + width] = True
    if extra_bottom and rng.random() < 0.7:
        width = rng.randint(max(1, base_w // 3), base_w)
        start = rng.randint(0, base_w - width)
        mask[base_r0 + base_h :, base_c0 + start : base_c0 + start + width] = True
    if extra_left and rng.random() < 0.7:
        height = rng.randint(max(1, base_h // 3), base_h)
        start = rng.randint(0, base_h - height)
        mask[base_r0 + start : base_r0 + start + height, :extra_left] = True
    if extra_right and rng.random() < 0.7:
        height = rng.randint(max(1, base_h // 3), base_h)
        start = rng.randint(0, base_h - height)
        mask[base_r0 + start : base_r0 + start + height, base_c0 + base_w :] = True

    hole_r0 = base_r0 + margin_top
    hole_c0 = base_c0 + margin_left
    hole_cells = [(hole_r0 + r, hole_c0 + c) for r, c in hole_shape]
    for r, c in hole_cells:
        mask[r, c] = False

    patch = np.full((total_h, total_w), BG, dtype=int)
    patch[mask] = int(object_color)
    for r, c in hole_cells:
        patch[r, c] = int(fill_color)
    return patch, hole_cells


def _make_template_patch(
    shape: tuple[tuple[int, int], ...],
    color: int,
) -> np.ndarray:
    mask = _shape_to_mask(shape)
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = int(color)
    return patch


def _object_positions(h: int, w: int, patch_h: int, patch_w: int) -> list[tuple[int, int]]:
    row_start = OBJECT_EDGE_MARGIN
    col_start = OBJECT_EDGE_MARGIN
    row_stop = h - patch_h - OBJECT_EDGE_MARGIN
    col_stop = w - patch_w - OBJECT_EDGE_MARGIN
    if row_stop < row_start or col_stop < col_start:
        return []
    return [(r0, c0) for r0 in range(row_start, row_stop + 1) for c0 in range(col_start, col_stop + 1)]


def _edge_positions(side: str, h: int, w: int, patch_h: int, patch_w: int) -> list[tuple[int, int]]:
    if side == "top":
        return [(0, c0) for c0 in range(w - patch_w + 1)]
    if side == "bottom":
        return [(h - patch_h, c0) for c0 in range(w - patch_w + 1)]
    if side == "left":
        return [(r0, 0) for r0 in range(h - patch_h + 1)]
    return [(r0, w - patch_w) for r0 in range(h - patch_h + 1)]


def _place_objects(
    output: np.ndarray,
    occ: np.ndarray,
    object_specs: list[dict],
    rng: random.Random,
):
    placements = []
    order = list(range(len(object_specs)))
    order.sort(key=lambda i: object_specs[i]["patch"].shape[0] * object_specs[i]["patch"].shape[1], reverse=True)
    rng.shuffle(order)
    order.sort(key=lambda i: object_specs[i]["patch"].shape[0] * object_specs[i]["patch"].shape[1], reverse=True)

    for idx in order:
        spec = object_specs[idx]
        patch = spec["patch"]
        ph, pw = patch.shape
        candidates = _object_positions(output.shape[0], output.shape[1], ph, pw)
        rng.shuffle(candidates)
        chosen = None
        for r0, c0 in candidates:
            if _bbox_free(occ, r0, c0, ph, pw, PLACEMENT_PAD):
                chosen = (r0, c0)
                break
        if chosen is None:
            return None
        r0, c0 = chosen
        _paint_patch(output, patch, r0, c0, BG)
        _mark_bbox(occ, r0, c0, ph, pw, PLACEMENT_PAD)
        placements.append(
            {
                "index": idx,
                "r0": r0,
                "c0": c0,
            }
        )
    return placements


def _place_templates(
    grid: np.ndarray,
    occ: np.ndarray,
    template_specs: list[dict],
    rng: random.Random,
) -> bool:
    order = list(range(len(template_specs)))
    rng.shuffle(order)
    for idx in order:
        spec = template_specs[idx]
        patch = spec["patch"]
        ph, pw = patch.shape
        sides = list(EDGE_SIDES)
        rng.shuffle(sides)
        chosen = None
        for side in sides:
            candidates = _edge_positions(side, grid.shape[0], grid.shape[1], ph, pw)
            rng.shuffle(candidates)
            for r0, c0 in candidates:
                if _bbox_free(occ, r0, c0, ph, pw, PLACEMENT_PAD):
                    chosen = (r0, c0)
                    break
            if chosen is not None:
                break
        if chosen is None:
            return False
        r0, c0 = chosen
        _paint_patch(grid, patch, r0, c0, BG)
        _mark_bbox(occ, r0, c0, ph, pw, PLACEMENT_PAD)
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    n_objects = rng.randint(*OBJECT_COUNT_RANGE)
    hole_shapes = _sample_unique_shapes(n_objects, rng)
    if hole_shapes is None:
        return None

    colors = list(range(1, 10))
    rng.shuffle(colors)
    if len(colors) < 2 * n_objects:
        return None
    object_colors = colors[:n_objects]
    template_colors = colors[n_objects : 2 * n_objects]

    object_specs = []
    template_specs = []
    total_input_fg = 0
    for i, base_shape in enumerate(hole_shapes):
        hole_shape = _rotate_shape(base_shape, rng.randrange(4))
        template_shape = _rotate_shape(base_shape, rng.randrange(4))
        object_patch, hole_cells = _make_object_patch(hole_shape, object_colors[i], template_colors[i], rng)
        template_patch = _make_template_patch(template_shape, template_colors[i])
        object_specs.append(
            {
                "patch": object_patch,
                "hole_cells": hole_cells,
            }
        )
        template_specs.append({"patch": template_patch})
        total_input_fg += int(np.sum(object_patch == object_colors[i])) + int(np.sum(template_patch != BG))

    dims = []
    max_obj_h = max(spec["patch"].shape[0] for spec in object_specs)
    max_obj_w = max(spec["patch"].shape[1] for spec in object_specs)
    max_tpl_h = max(spec["patch"].shape[0] for spec in template_specs)
    max_tpl_w = max(spec["patch"].shape[1] for spec in template_specs)
    min_h = max(GRID_H_RANGE[0], max_obj_h + 2 * OBJECT_EDGE_MARGIN, max_tpl_h + 2)
    min_w = max(GRID_W_RANGE[0], max_obj_w + 2 * OBJECT_EDGE_MARGIN, max_tpl_w + 2)
    for h in range(min_h, GRID_H_RANGE[1] + 1):
        for w in range(min_w, GRID_W_RANGE[1] + 1):
            if h * w > 2 * total_input_fg + 60:
                dims.append((h, w))
    if not dims:
        return None
    h, w = rng.choice(dims)

    output = np.full((h, w), BG, dtype=int)
    occ = np.zeros((h, w), dtype=bool)
    placements = _place_objects(output, occ, object_specs, rng)
    if placements is None:
        return None

    inp = output.copy()
    for placement in placements:
        spec = object_specs[placement["index"]]
        for hr, hc in spec["hole_cells"]:
            inp[placement["r0"] + hr, placement["c0"] + hc] = BG

    if not _place_templates(inp, occ, template_specs, rng):
        return None

    if _get_bg(inp) != BG:
        return None

    expected = output
    try:
        actual = main(inp)
    except Exception:
        return None
    if np.array_equal(actual, inp):
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
