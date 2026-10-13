import random
from collections import deque

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_GENERATION_ATTEMPTS = 400
MAX_OBJECT_DIM = 8


def _neighbors4(r, c):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _get_bg(grid):
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_mask(mask):
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox_from_pts(pts):
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _trim_mask(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return mask[rows][:, cols]


def _hole_count(shape):
    shape = np.asarray(shape, dtype=bool)
    h, w = shape.shape
    holes = 0
    for comp in _components_mask(~shape):
        touches_edge = np.any(
            (comp[:, 0] == 0)
            | (comp[:, 0] == h - 1)
            | (comp[:, 1] == 0)
            | (comp[:, 1] == w - 1)
        )
        if not touches_edge:
            holes += 1
    return holes


def _is_connected(mask):
    return len(_components_mask(mask)) == 1


def _extract_objects(grid):
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    objects = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        for pts in _components_mask(g == color):
            r0, c0, r1, c1 = _bbox_from_pts(pts)
            rel = pts - np.array([r0, c0], dtype=int)
            shape = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            shape[rel[:, 0], rel[:, 1]] = True
            objects.append(
                {
                    "color": int(color),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "shape": shape,
                    "holes": _hole_count(shape),
                }
            )
    objects.sort(key=lambda obj: (obj["bbox"][1], obj["bbox"][0]))
    return bg, objects


def _class_from_holes(hole_count):
    if hole_count == 0:
        return 0
    if hole_count == 1:
        return 1
    return 2


def _band_ranges(height):
    split1 = height // 3
    split2 = (2 * height) // 3
    return ((0, split1), (split1, split2), (split2, height))


def _paste_shape(canvas, shape, top, left, color):
    rr, cc = np.nonzero(shape)
    canvas[top + rr, left + cc] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group monochrome objects by hole count and repack them into top/middle/bottom bands.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg, objects = _extract_objects(g)
    bands = _band_ranges(h)

    grouped = {0: [], 1: [], 2: []}
    for obj in objects:
        grouped[_class_from_holes(obj["holes"])].append(obj)

    out = np.full((h, w), bg, dtype=int)
    for band_idx in (0, 1, 2):
        band_top, band_bottom = bands[band_idx]
        band_height = band_bottom - band_top
        col = 0
        for obj in grouped[band_idx]:
            shape = obj["shape"]
            oh, ow = shape.shape
            assert oh <= band_height
            assert col + ow <= w
            _paste_shape(out, shape, band_top, col, obj["color"])
            col += ow + 1

    return out


def _mask_bbox(mask):
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _boundary_cells(mask):
    h, w = mask.shape
    cells = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            for nr, nc in _neighbors4(r, c):
                if nr < 0 or nr >= h or nc < 0 or nc >= w or not mask[nr, nc]:
                    cells.append((r, c))
                    break
    return cells


def _random_reorient(mask, rng):
    if rng.random() < 0.5:
        mask = np.fliplr(mask)
    if rng.random() < 0.5:
        mask = np.flipud(mask)
    mask = np.rot90(mask, rng.randrange(4))
    return np.asarray(mask, dtype=bool)


def _add_tabs(mask, rng, max_tabs):
    pad = 3
    canvas = np.zeros((mask.shape[0] + 2 * pad, mask.shape[1] + 2 * pad), dtype=bool)
    canvas[pad : pad + mask.shape[0], pad : pad + mask.shape[1]] = mask
    for _ in range(rng.randint(0, max_tabs)):
        r0, c0, r1, c1 = _mask_bbox(canvas)
        box_h = r1 - r0 + 1
        box_w = c1 - c0 + 1
        side = rng.choice(("top", "bottom", "left", "right"))
        if side in ("top", "bottom"):
            depth = rng.randint(1, 2)
            span = rng.randint(1, max(1, box_w - 1))
            start = rng.randint(c0, c1 - span + 1)
            if side == "top":
                if r0 - depth < 0:
                    continue
                canvas[r0 - depth : r0, start : start + span] = True
            else:
                if r1 + 1 + depth > canvas.shape[0]:
                    continue
                canvas[r1 + 1 : r1 + 1 + depth, start : start + span] = True
        else:
            depth = rng.randint(1, 2)
            span = rng.randint(1, max(1, box_h - 1))
            start = rng.randint(r0, r1 - span + 1)
            if side == "left":
                if c0 - depth < 0:
                    continue
                canvas[start : start + span, c0 - depth : c0] = True
            else:
                if c1 + 1 + depth > canvas.shape[1]:
                    continue
                canvas[start : start + span, c1 + 1 : c1 + 1 + depth] = True
    return _trim_mask(canvas)


def _erode_outline(mask, expected_holes, rng, max_steps):
    work = np.asarray(mask, dtype=bool)
    for _ in range(rng.randint(0, max_steps)):
        candidates = _boundary_cells(work)
        rng.shuffle(candidates)
        changed = False
        for r, c in candidates:
            test = work.copy()
            test[r, c] = False
            if not test.any():
                continue
            test = _trim_mask(test)
            if _is_connected(test) and _hole_count(test) == expected_holes:
                work = test
                changed = True
                break
        if not changed:
            break
    return work


def _shape_ok(mask, expected_holes):
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return False
    mask = _trim_mask(mask)
    h, w = mask.shape
    if h < 2 or w < 2 or h > MAX_OBJECT_DIM or w > MAX_OBJECT_DIM:
        return False
    if mask.sum() < 4:
        return False
    if not _is_connected(mask):
        return False
    return _hole_count(mask) == expected_holes


def _make_solid_shape(rng):
    for _ in range(150):
        base = np.ones((rng.randint(2, 5), rng.randint(2, 5)), dtype=bool)
        shape = _add_tabs(base, rng, max_tabs=4)
        shape = _erode_outline(shape, 0, rng, max_steps=2)
        shape = _random_reorient(shape, rng)
        if _shape_ok(shape, 0):
            return shape
    raise RuntimeError("failed to build solid shape")


def _make_one_hole_shape(rng):
    for _ in range(150):
        top = rng.randint(1, 2)
        bottom = rng.randint(1, 2)
        left = rng.randint(1, 2)
        right = rng.randint(1, 2)
        hole_h = rng.randint(1, 3)
        hole_w = rng.randint(1, 3)
        outer_h = top + hole_h + bottom
        outer_w = left + hole_w + right
        shape = np.ones((outer_h, outer_w), dtype=bool)
        hole_r = rng.randint(top, outer_h - bottom - hole_h)
        hole_c = rng.randint(left, outer_w - right - hole_w)
        shape[hole_r : hole_r + hole_h, hole_c : hole_c + hole_w] = False
        shape = _add_tabs(shape, rng, max_tabs=3)
        shape = _erode_outline(shape, 1, rng, max_steps=2)
        shape = _random_reorient(shape, rng)
        if _shape_ok(shape, 1):
            return shape
    raise RuntimeError("failed to build one-hole shape")


def _make_line_holes(rng, hole_count):
    orient = rng.choice(("horizontal", "vertical"))
    hole_sizes = [(rng.randint(1, 2), rng.randint(1, 2)) for _ in range(hole_count)]
    if orient == "horizontal":
        top = rng.randint(1, 2)
        bottom = rng.randint(1, 2)
        gap = rng.randint(1, 2)
        outer_h = top + max(hh for hh, _ in hole_sizes) + bottom
        outer_w = 1 + sum(ww for _, ww in hole_sizes) + gap * (hole_count - 1) + 1
        shape = np.ones((outer_h, outer_w), dtype=bool)
        col = 1
        for hole_h, hole_w in hole_sizes:
            row = rng.randint(top, outer_h - bottom - hole_h)
            shape[row : row + hole_h, col : col + hole_w] = False
            col += hole_w + gap
        return shape

    left = rng.randint(1, 2)
    right = rng.randint(1, 2)
    gap = rng.randint(1, 2)
    outer_h = 1 + sum(hh for hh, _ in hole_sizes) + gap * (hole_count - 1) + 1
    outer_w = left + max(ww for _, ww in hole_sizes) + right
    shape = np.ones((outer_h, outer_w), dtype=bool)
    row = 1
    for hole_h, hole_w in hole_sizes:
        col = rng.randint(left, outer_w - right - hole_w)
        shape[row : row + hole_h, col : col + hole_w] = False
        row += hole_h + gap
    return shape


def _make_grid_holes(rng, hole_count):
    row_heights = [rng.randint(1, 2), rng.randint(1, 2)]
    col_widths = [rng.randint(1, 2), rng.randint(1, 2)]
    top = rng.randint(1, 2)
    bottom = rng.randint(1, 2)
    left = rng.randint(1, 2)
    right = rng.randint(1, 2)
    gap = 1
    outer_h = top + row_heights[0] + gap + row_heights[1] + bottom
    outer_w = left + col_widths[0] + gap + col_widths[1] + right
    shape = np.ones((outer_h, outer_w), dtype=bool)
    cells = [(0, 0), (0, 1), (1, 0), (1, 1)]
    keep = rng.sample(cells, hole_count)
    row_starts = [top, top + row_heights[0] + gap]
    col_starts = [left, left + col_widths[0] + gap]
    for rr, cc in keep:
        r0 = row_starts[rr]
        c0 = col_starts[cc]
        shape[r0 : r0 + row_heights[rr], c0 : c0 + col_widths[cc]] = False
    return shape


def _make_multi_hole_shape(rng, hole_count):
    for _ in range(200):
        if hole_count == 3 and rng.random() < 0.4:
            shape = _make_grid_holes(rng, 3)
        else:
            shape = _make_line_holes(rng, hole_count)
        shape = _add_tabs(shape, rng, max_tabs=3)
        shape = _erode_outline(shape, hole_count, rng, max_steps=2)
        shape = _random_reorient(shape, rng)
        if _shape_ok(shape, hole_count):
            return shape
    raise RuntimeError("failed to build multi-hole shape")


def _packed_width(shapes):
    if not shapes:
        return 0
    return sum(shape.shape[1] for shape in shapes) + len(shapes) - 1


def _band_index_for_center(row, bands):
    for idx, (top, bottom) in enumerate(bands):
        if top <= row < bottom:
            return idx
    return 2


def _place_objects(grid_shape, objects, bg, rng):
    h, w = grid_shape
    grid = np.full((h, w), bg, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    bands = _band_ranges(h)
    placed = [None] * len(objects)

    order = sorted(range(len(objects)), key=lambda idx: int(objects[idx]["shape"].sum()), reverse=True)
    for idx in order:
        obj = objects[idx]
        shape = obj["shape"]
        oh, ow = shape.shape
        candidates = []
        for top in range(1, h - oh):
            for left in range(1, w - ow):
                if blocked[top : top + oh, left : left + ow].any():
                    continue
                candidates.append((top, left))
        if not candidates:
            return None

        preferred = []
        for top, left in candidates:
            center_band = _band_index_for_center(top + oh // 2, bands)
            if center_band != obj["class"]:
                preferred.append((top, left))
        if preferred and rng.random() < 0.8:
            candidates = preferred

        top, left = rng.choice(candidates)
        _paste_shape(grid, shape, top, left, obj["color"])
        r0 = max(0, top - 1)
        c0 = max(0, left - 1)
        r1 = min(h, top + oh + 1)
        c1 = min(w, left + ow + 1)
        blocked[r0:r1, c0:c1] = True
        placed[idx] = {"top": top, "left": left, "height": oh, "width": ow, "class": obj["class"]}

    return grid, placed


def _build_objects(rng):
    counts = {0: 1, 1: 1, 2: 1}
    for _ in range(rng.randint(1, 4)):
        counts[rng.choice((0, 1, 2))] += 1

    objects = []
    for cls in (0, 1, 2):
        for _ in range(counts[cls]):
            if cls == 0:
                hole_count = 0
                shape = _make_solid_shape(rng)
            elif cls == 1:
                hole_count = 1
                shape = _make_one_hole_shape(rng)
            else:
                hole_count = rng.choice((2, 3))
                shape = _make_multi_hole_shape(rng, hole_count)
            objects.append({"class": cls, "holes": hole_count, "shape": shape})
    return objects


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            objects = _build_objects(rng)
        except RuntimeError:
            continue

        max_obj_h = max(obj["shape"].shape[0] for obj in objects)
        band_h = max(6, max_obj_h + rng.randint(0, 2))
        grid_h = band_h * 3
        if grid_h > 30:
            continue

        widths_by_class = {cls: [] for cls in (0, 1, 2)}
        for obj in objects:
            widths_by_class[obj["class"]].append(obj["shape"])
        required_w = max(_packed_width(widths_by_class[cls]) for cls in (0, 1, 2))
        grid_w = max(18, required_w + rng.randint(2, 6))
        if grid_w > 30:
            continue

        bg = rng.randrange(10)
        colors = rng.sample([c for c in range(10) if c != bg], len(objects))
        for obj, color in zip(objects, colors):
            obj["color"] = int(color)

        placed = _place_objects((grid_h, grid_w), objects, bg, rng)
        if placed is None:
            continue
        grid, placements = placed

        left_edges = [info["left"] for info in placements]
        if len(set(left_edges)) != len(left_edges):
            continue

        bands = _band_ranges(grid_h)
        moved_classes = set()
        for info in placements:
            center_row = info["top"] + info["height"] // 2
            if _band_index_for_center(center_row, bands) != info["class"]:
                moved_classes.add(info["class"])
        if moved_classes != {0, 1, 2}:
            continue

        out = main(grid)
        if np.array_equal(out, grid):
            continue
        return grid

    raise RuntimeError("failed to generate a valid input")
