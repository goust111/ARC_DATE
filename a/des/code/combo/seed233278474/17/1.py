from __future__ import annotations

from collections import Counter
import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(points: np.ndarray) -> tuple[int, int, int, int]:
    rows = points[:, 0]
    cols = points[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _shape_signature(points: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = _bbox(points)
    rel = sorted((int(r - r0), int(c - c0)) for r, c in points.tolist())
    return (r1 - r0 + 1, c1 - c0 + 1, tuple(rel))


def _mask_signature(mask: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    pts = np.argwhere(mask)
    return _shape_signature(np.asarray(pts, dtype=int))


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _outline_color(grid: np.ndarray, bg: int) -> int:
    counts = Counter(int(v) for v in grid.reshape(-1) if int(v) != bg)
    assert counts, "expected at least one foreground color"
    ranked = counts.most_common()
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "expected a unique dominant outline color"
    return int(ranked[0][0])


def _component_patch(grid: np.ndarray, points: np.ndarray, bg: int) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(points)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    for r, c in points.tolist():
        patch[int(r) - r0, int(c) - c0] = int(grid[int(r), int(c)])
    return patch


def _holes_in_outline(points: np.ndarray) -> list[np.ndarray]:
    r0, c0, r1, c1 = _bbox(points)
    obj_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in points.tolist():
        obj_mask[int(r) - r0, int(c) - c0] = True

    holes: list[np.ndarray] = []
    for comp in _components(~obj_mask):
        if np.any(comp[:, 0] == 0) or np.any(comp[:, 0] == obj_mask.shape[0] - 1):
            continue
        if np.any(comp[:, 1] == 0) or np.any(comp[:, 1] == obj_mask.shape[1] - 1):
            continue
        holes.append(comp + np.array([r0, c0], dtype=int))
    return holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match hole shapes to edge templates and transplant the matching colored template into each hole.
    """
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    outline = _outline_color(g, bg)

    out = g.copy()
    objects = _components(g == outline)
    template_components = _components((g != bg) & (g != outline))
    assert objects, "expected at least one outline object"
    assert template_components, "expected at least one template component"

    templates: dict[tuple[int, int, tuple[tuple[int, int], ...]], np.ndarray] = {}
    for comp in template_components:
        sig = _shape_signature(comp)
        assert sig not in templates, "template shapes must be unique"
        patch = _component_patch(g, comp, bg)
        assert np.array_equal(patch != bg, _component_patch((g != bg).astype(int), comp, 0).astype(bool))
        templates[sig] = patch

    for obj in objects:
        for hole in _holes_in_outline(obj):
            sig = _shape_signature(hole)
            assert sig in templates, "every hole must have a matching template"
            patch = templates[sig]
            hr0, hc0, hr1, hc1 = _bbox(hole)
            hole_mask = np.zeros((hr1 - hr0 + 1, hc1 - hc0 + 1), dtype=bool)
            for r, c in hole.tolist():
                hole_mask[int(r) - hr0, int(c) - hc0] = True
            assert patch.shape == hole_mask.shape, "template bbox must equal hole bbox"
            assert np.array_equal(patch != bg, hole_mask), "template occupied mask must match the hole exactly"
            target = out[hr0 : hr1 + 1, hc0 : hc1 + 1]
            target[hole_mask] = patch[hole_mask]

    for comp in template_components:
        out[comp[:, 0], comp[:, 1]] = bg
    return out


def _crop_bool(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1].copy()


def _normalize_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return {(r - r0, c - c0) for r, c in cells}


def _cells_to_mask(cells: set[tuple[int, int]]) -> np.ndarray:
    h = max(r for r, _ in cells) + 1
    w = max(c for _, c in cells) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _random_shape(rng: random.Random) -> np.ndarray | None:
    target_area = rng.randint(4, 8)
    max_h = rng.randint(2, 5)
    max_w = rng.randint(2, 5)
    cells: set[tuple[int, int]] = {(0, 0)}
    while len(cells) < target_area:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in DIR4:
                cand = (r + dr, c + dc)
                if cand in cells or cand in frontier:
                    continue
                trial = _normalize_cells(cells | {cand})
                h = max(rr for rr, _ in trial) + 1
                w = max(cc for _, cc in trial) + 1
                if h <= max_h and w <= max_w:
                    frontier.append(cand)
        if not frontier:
            return None
        cells.add(rng.choice(frontier))

    mask = _cells_to_mask(_normalize_cells(cells))
    k = rng.randrange(4)
    if k:
        mask = np.rot90(mask, k)
    if rng.random() < 0.5:
        mask = np.fliplr(mask)
    if rng.random() < 0.5:
        mask = np.flipud(mask)
    mask = _crop_bool(mask)
    h, w = mask.shape
    area = int(mask.sum())
    if h < 2 or w < 2:
        return None
    if area == h * w and rng.random() < 0.85:
        return None
    return mask.copy()


def _choose_shapes(count: int, rng: random.Random) -> list[np.ndarray]:
    for _ in range(300):
        shapes: list[np.ndarray] = []
        seen: set[tuple[int, int, tuple[tuple[int, int], ...]]] = set()
        while len(shapes) < count:
            shape = _random_shape(rng)
            if shape is None:
                continue
            sig = _mask_signature(shape)
            if sig in seen:
                continue
            seen.add(sig)
            shapes.append(shape)

        areas = [int(shape.sum()) for shape in shapes]
        dims = [shape.shape for shape in shapes]
        same_area = len(set(areas)) < len(areas)
        same_bbox = len(set(dims)) < len(dims)
        if same_area or same_bbox:
            return shapes
    raise RuntimeError("failed to generate a non-degenerate set of unique hole shapes")


def _color_template(mask: np.ndarray, bg: int, palette: list[int], rng: random.Random) -> np.ndarray:
    patch = np.full(mask.shape, int(bg), dtype=int)
    area = int(mask.sum())
    n_colors = rng.choice([1, 2, 2, 3]) if area >= 5 else rng.choice([1, 2, 2])
    colors = rng.sample(palette, n_colors)
    mode = rng.choice(["checker", "rows", "cols", "diagonal"])
    cells = np.argwhere(mask)
    for r, c in cells.tolist():
        if mode == "checker":
            idx = (r + c) % n_colors
        elif mode == "rows":
            idx = r % n_colors
        elif mode == "cols":
            idx = c % n_colors
        else:
            idx = (2 * r + c) % n_colors
        patch[int(r), int(c)] = colors[idx]
    if n_colors > 1 and len(set(int(patch[r, c]) for r, c in cells.tolist())) == 1:
        rr, cc = cells[-1]
        patch[int(rr), int(cc)] = colors[-1]
    return patch


def _partition(total: int, parts: int, rng: random.Random) -> list[int]:
    sizes = [1] * parts
    for _ in range(total - parts):
        sizes[rng.randrange(parts)] += 1
    rng.shuffle(sizes)
    return sizes


def _boxes_touch(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 - pad or br1 + pad < ar0 - pad or ac1 + pad < bc0 - pad or bc1 + pad < ac0 - pad)


def _build_object(
    hole_indices: list[int],
    shapes: list[np.ndarray],
    templates: list[np.ndarray],
    outline: int,
    bg: int,
    rng: random.Random,
) -> dict:
    group_shapes = [shapes[i] for i in hole_indices]
    group_templates = [templates[i] for i in hole_indices]
    areas = [int(shape.sum()) for shape in group_shapes]
    order = sorted(range(len(group_shapes)), key=lambda idx: (-areas[idx], idx))

    for grow in range(18):
        base = int(np.ceil(np.sqrt(sum(areas) * 4.0)))
        patch_h = max(max(shape.shape[0] for shape in group_shapes) + 2, base + rng.randint(0, 3) + grow // 3)
        patch_w = max(max(shape.shape[1] for shape in group_shapes) + 2, base + rng.randint(0, 3) + grow // 3)
        if rng.random() < 0.5:
            patch_h += rng.randint(0, len(group_shapes))
        else:
            patch_w += rng.randint(0, len(group_shapes))
        placements: list[tuple[int, int] | None] = [None] * len(group_shapes)
        boxes: list[tuple[int, int, int, int]] = []
        ok = True
        for idx in order:
            shape = group_shapes[idx]
            sh, sw = shape.shape
            candidates = [
                (r0, c0)
                for r0 in range(1, patch_h - sh)
                for c0 in range(1, patch_w - sw)
            ]
            rng.shuffle(candidates)
            placed = False
            for r0, c0 in candidates:
                box = (r0, c0, r0 + sh - 1, c0 + sw - 1)
                if any(_boxes_touch(box, other, pad=1) for other in boxes):
                    continue
                placements[idx] = (r0, c0)
                boxes.append(box)
                placed = True
                break
            if not placed:
                ok = False
                break
        if not ok:
            continue

        patch = np.full((patch_h, patch_w), int(outline), dtype=int)
        holes = []
        for local_idx, (shape, template, pos) in enumerate(zip(group_shapes, group_templates, placements)):
            assert pos is not None
            r0, c0 = pos
            sh, sw = shape.shape
            patch[r0 : r0 + sh, c0 : c0 + sw][shape] = int(bg)
            holes.append(
                {
                    "top": int(r0),
                    "left": int(c0),
                    "shape": shape.copy(),
                    "template": template.copy(),
                    "hole_index": int(hole_indices[local_idx]),
                }
            )

        return {"patch": patch, "holes": holes}
    raise RuntimeError("failed to build an outline object")


def _pack_rectangles(
    sizes: list[tuple[int, int]],
    region_h: int,
    region_w: int,
    rng: random.Random,
) -> dict[int, tuple[int, int]] | None:
    order = sorted(range(len(sizes)), key=lambda idx: -(sizes[idx][0] * sizes[idx][1]))
    for _ in range(240):
        placed: dict[int, tuple[int, int]] = {}
        boxes: list[tuple[int, int, int, int]] = []
        ok = True
        for idx in order:
            h, w = sizes[idx]
            candidates = [
                (r0, c0)
                for r0 in range(0, region_h - h + 1)
                for c0 in range(0, region_w - w + 1)
            ]
            rng.shuffle(candidates)
            found = False
            for r0, c0 in candidates:
                box = (r0, c0, r0 + h - 1, c0 + w - 1)
                if any(_boxes_touch(box, other, pad=1) for other in boxes):
                    continue
                boxes.append(box)
                placed[idx] = (r0, c0)
                found = True
                break
            if not found:
                ok = False
                break
        if ok:
            return placed
    return None


def _pack_objects(objects: list[dict], rng: random.Random) -> tuple[int, int, dict[int, tuple[int, int]]]:
    sizes = [obj["patch"].shape for obj in objects]
    total = sum((h + 2) * (w + 2) for h, w in sizes)
    region_h = max(max(h for h, _ in sizes) + 2, int(np.ceil(np.sqrt(total))) + rng.randint(0, 2))
    region_w = max(max(w for _, w in sizes) + 2, int(np.ceil(np.sqrt(total))) + rng.randint(0, 2))
    for _ in range(16):
        packed = _pack_rectangles(sizes, region_h, region_w, rng)
        if packed is not None:
            return region_h, region_w, packed
        if rng.random() < 0.5:
            region_h += 2
        else:
            region_w += 2
    raise RuntimeError("failed to pack outline objects")


def _template_band_layout(
    templates: list[np.ndarray],
    side: str,
    rng: random.Random,
) -> tuple[list[int], list[int], int]:
    order = list(range(len(templates)))
    rng.shuffle(order)
    gaps = [rng.randint(1, 2) for _ in range(max(0, len(templates) - 1))]
    if side in ("left", "right"):
        span = sum(templates[idx].shape[0] for idx in order) + sum(gaps)
    else:
        span = sum(templates[idx].shape[1] for idx in order) + sum(gaps)
    return order, gaps, int(span)


def _render_pair(
    side: str,
    bg: int,
    outline: int,
    objects: list[dict],
    object_region_h: int,
    object_region_w: int,
    object_positions: dict[int, tuple[int, int]],
    templates: list[np.ndarray],
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray]:
    order, gaps, template_span = _template_band_layout(templates, side, rng)
    band_h = max(template.shape[0] for template in templates)
    band_w = max(template.shape[1] for template in templates)
    sep_gap = rng.randint(1, 2)

    if side in ("left", "right"):
        top_margin = rng.randint(1, 3)
        bottom_margin = rng.randint(1, 3)
        other_margin = rng.randint(1, 3)
        H = max(object_region_h, template_span) + top_margin + bottom_margin
        W = band_w + sep_gap + object_region_w + other_margin
        if side == "left":
            band_c = 0
            object_c0 = band_w + sep_gap
        else:
            band_c = W - band_w
            object_c0 = other_margin
        object_r0 = rng.randint(top_margin, H - bottom_margin - object_region_h)
        template_r = rng.randint(0, H - template_span)
        template_positions: dict[int, tuple[int, int]] = {}
        for j, idx in enumerate(order):
            template = templates[idx]
            c0 = band_c if side == "left" else W - template.shape[1]
            template_positions[idx] = (template_r, c0)
            template_r += template.shape[0]
            if j < len(gaps):
                template_r += gaps[j]
        object_offset = (object_r0, object_c0)
    else:
        left_margin = rng.randint(1, 3)
        right_margin = rng.randint(1, 3)
        other_margin = rng.randint(1, 3)
        H = band_h + sep_gap + object_region_h + other_margin
        W = max(object_region_w, template_span) + left_margin + right_margin
        if side == "top":
            band_r = 0
            object_r0 = band_h + sep_gap
        else:
            band_r = H - band_h
            object_r0 = other_margin
        object_c0 = rng.randint(left_margin, W - right_margin - object_region_w)
        template_c = rng.randint(0, W - template_span)
        template_positions = {}
        for j, idx in enumerate(order):
            template = templates[idx]
            r0 = band_r if side == "top" else H - template.shape[0]
            template_positions[idx] = (r0, template_c)
            template_c += template.shape[1]
            if j < len(gaps):
                template_c += gaps[j]
        object_offset = (object_r0, object_c0)

    assert H <= 32 and W <= 32, "keep generated grids visually manageable"
    inp = np.full((H, W), int(bg), dtype=int)
    out = np.full((H, W), int(bg), dtype=int)

    for obj_idx, obj in enumerate(objects):
        patch = obj["patch"]
        pr0 = object_offset[0] + object_positions[obj_idx][0]
        pc0 = object_offset[1] + object_positions[obj_idx][1]
        ph, pw = patch.shape
        inp[pr0 : pr0 + ph, pc0 : pc0 + pw] = patch
        out[pr0 : pr0 + ph, pc0 : pc0 + pw] = patch
        for hole in obj["holes"]:
            hr0 = pr0 + hole["top"]
            hc0 = pc0 + hole["left"]
            template = hole["template"]
            sh, sw = hole["shape"].shape
            out[hr0 : hr0 + sh, hc0 : hc0 + sw][hole["shape"]] = template[hole["shape"]]

    for idx, template in enumerate(templates):
        tr0, tc0 = template_positions[idx]
        th, tw = template.shape
        region = inp[tr0 : tr0 + th, tc0 : tc0 + tw]
        region[template != bg] = template[template != bg]

    counts = Counter(int(v) for v in inp.reshape(-1) if int(v) != bg)
    ranked = counts.most_common()
    assert ranked and ranked[0][0] == outline
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1]
    return inp, out


def _generate_pair(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    bg, outline = rng.sample(range(10), 2)
    palette = [color for color in range(10) if color not in (bg, outline)]

    n_holes = rng.randint(2, 5)
    n_objects = rng.randint(1, min(3, n_holes))
    side = rng.choice(["left", "right", "top", "bottom"])

    shapes = _choose_shapes(n_holes, rng)
    templates = [_color_template(shape, bg, palette, rng) for shape in shapes]

    groups = _partition(n_holes, n_objects, rng)
    shuffled_indices = list(range(n_holes))
    rng.shuffle(shuffled_indices)
    objects = []
    start = 0
    for group_size in groups:
        hole_indices = shuffled_indices[start : start + group_size]
        start += group_size
        objects.append(_build_object(hole_indices, shapes, templates, outline, bg, rng))

    object_region_h, object_region_w, object_positions = _pack_objects(objects, rng)
    return _render_pair(
        side=side,
        bg=bg,
        outline=outline,
        objects=objects,
        object_region_h=object_region_h,
        object_region_w=object_region_w,
        object_positions=object_positions,
        templates=templates,
        rng=rng,
    )


def generate_input() -> np.ndarray:
    for _ in range(400):
        rng = random.Random(random.randrange(1 << 30))
        try:
            inp, out = _generate_pair(rng)
        except (AssertionError, RuntimeError, ValueError):
            continue
        try:
            if np.array_equal(main(inp), out):
                return inp
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid input")
