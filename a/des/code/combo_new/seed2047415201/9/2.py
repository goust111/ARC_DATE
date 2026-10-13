from collections import deque
import random

import numpy as np


BG = 0
GREY = 5
AZURE = 8
LABEL_COLORS = (1, 2, 3, 4, 6, 7, 9)


def _neighbors4(r, c):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask):
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []

    for sr in range(h):
        for sc in range(w):
            if not mask[sr, sc] or seen[sr, sc]:
                continue
            q = deque([(sr, sc)])
            seen[sr, sc] = True
            pts = []
            while q:
                r, c = q.popleft()
                pts.append((r, c))
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(pts):
    rs = [r for r, _c in pts]
    cs = [c for _r, c in pts]
    return min(rs), min(cs), max(rs), max(cs)


def _mask_from_coords(pts):
    r0, c0, r1, c1 = _bbox(pts)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in pts:
        mask[r - r0, c - c0] = True
    return mask


def _crop_mask(mask):
    pts = np.argwhere(mask)
    if pts.size == 0:
        return mask[:0, :0].astype(bool)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def _shape_key(mask):
    """Canonical binary shape key, invariant to translation, rotation, and reflection."""
    mask = _crop_mask(np.asarray(mask, dtype=bool))
    variants = []
    for k in range(4):
        rot = np.rot90(mask, k)
        for var in (rot, np.fliplr(rot)):
            var = _crop_mask(var)
            pts = tuple((int(r), int(c)) for r, c in np.argwhere(var))
            variants.append((int(var.shape[0]), int(var.shape[1]), pts))
    return min(variants)


def _oriented_masks(mask):
    mask = _crop_mask(mask)
    out = []
    seen = set()
    for k in range(4):
        rot = np.rot90(mask, k)
        for var in (rot, np.fliplr(rot)):
            var = _crop_mask(var)
            key = tuple((int(r), int(c)) for r, c in np.argwhere(var))
            full_key = (int(var.shape[0]), int(var.shape[1]), key)
            if full_key not in seen:
                seen.add(full_key)
                out.append(var)
    return out


def _legend_mapping(g):
    h, w = g.shape
    mapping = {}
    legend_cells = set()
    label_cells = set()

    for pts in _components(g == GREY):
        r0, c0, r1, c1 = _bbox(pts)
        label_r = r1 + 1
        assert label_r < h, "each grey template must have a label directly below it"

        candidates = []
        for c in range(c0, c1 + 1):
            v = int(g[label_r, c])
            if v not in (BG, GREY):
                candidates.append((label_r, c, v))
        assert len(candidates) == 1, "each grey template must have exactly one color label below it"

        lr, lc, label = candidates[0]
        key = _shape_key(_mask_from_coords(pts))
        assert key not in mapping, "legend templates must be unique up to rotation and reflection"
        mapping[key] = label
        legend_cells.update((int(r), int(c)) for r, c in pts)
        label_cells.add((int(lr), int(lc)))

    assert mapping, "grid must contain at least one grey template legend entry"
    return mapping, legend_cells, label_cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor same-colored objects by matching their shapes to a left-side template legend.

    Input structure:
    - The background is Black. A left legend column contains Grey mask templates.
    - Each Grey template has exactly one colored label pixel directly below its bounding box.
    - The right main area contains separate 4-connected Azure foreground objects.

    Rule:
    - Normalize each Grey template and each Azure object to its bounding box.
    - Match shapes up to translation, rotation, and reflection.
    - Recolor each Azure object to the color of the label below the matching Grey template.
    - Erase the Grey templates and their label pixels from the output.

    Key insight: The Azure objects all have the same color, so only their shape identifies which legend color to use.
    """
    g = np.asarray(grid, dtype=int)
    mapping, legend_cells, label_cells = _legend_mapping(g)

    out = g.copy()
    for r, c in legend_cells | label_cells:
        out[r, c] = BG

    object_mask = g == AZURE
    for r, c in label_cells:
        object_mask[r, c] = False

    for pts in _components(object_mask):
        key = _shape_key(_mask_from_coords(pts))
        assert key in mapping, "every Azure object shape must appear in the legend"
        new_color = int(mapping[key])
        for r, c in pts:
            out[r, c] = new_color

    return out


def _random_connected_mask(rng):
    for _ in range(100):
        box_h = rng.randint(2, 4)
        box_w = rng.randint(2, 5)
        max_area = box_h * box_w
        target_area = rng.randint(4, min(max_area, 10))
        cells = {(rng.randrange(box_h), rng.randrange(box_w))}

        while len(cells) < target_area:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))

        if len(cells) != target_area:
            continue
        mask = _crop_mask(
            np.array(
                [[(r, c) in cells for c in range(box_w)] for r in range(box_h)],
                dtype=bool,
            )
        )
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if int(mask.sum()) == int(mask.size) and rng.random() < 0.75:
            continue
        return mask
    raise RuntimeError("failed to build a connected random mask")


def _unique_shapes(count, rng):
    shapes = []
    keys = set()
    for _ in range(500):
        mask = _random_connected_mask(rng)
        key = _shape_key(mask)
        if key in keys:
            continue
        keys.add(key)
        shapes.append(mask)
        if len(shapes) == count:
            return shapes
    raise RuntimeError("failed to build unique legend shapes")


def _can_place(grid, mask, top, left):
    h, w = grid.shape
    mh, mw = mask.shape
    if top < 0 or left < 0 or top + mh > h or left + mw > w:
        return False

    for rr, cc in np.argwhere(mask):
        r = top + int(rr)
        c = left + int(cc)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) != BG:
                    return False
    return True


def _place_mask(grid, mask, color, top, left):
    for rr, cc in np.argwhere(mask):
        grid[top + int(rr), left + int(cc)] = int(color)


def _gen_attempt(rng):
    n_templates = rng.choice([3, 4, 5])
    shapes = _unique_shapes(n_templates, rng)
    label_colors = rng.sample(list(LABEL_COLORS), n_templates)

    total_legend_h = sum(mask.shape[0] + 1 for mask in shapes) + (n_templates - 1)
    if total_legend_h > 30:
        return None

    min_h = max(18, total_legend_h)
    height = rng.randint(min_h, 30)
    width = rng.randint(20, 30)
    legend_w = max(mask.shape[1] for mask in shapes) + 2
    main_c0 = legend_w + 1
    if width - main_c0 < 10:
        return None

    grid = np.full((height, width), BG, dtype=int)
    row = rng.randint(0, height - total_legend_h)

    for mask, color in zip(shapes, label_colors):
        mh, mw = mask.shape
        col = rng.randint(1, legend_w - mw)
        _place_mask(grid, mask, GREY, row, col)

        bottom_cols = [int(c) for c in np.flatnonzero(mask[-1])]
        label_c = col + rng.choice(bottom_cols)
        grid[row + mh, label_c] = int(color)
        row += mh + 2

    n_objects = rng.randint(n_templates, min(8, n_templates + 4))
    object_ids = list(range(n_templates))
    object_ids.extend(rng.randrange(n_templates) for _ in range(n_objects - n_templates))
    rng.shuffle(object_ids)

    used_orientations = list(range(8))
    rng.shuffle(used_orientations)
    for i, shape_idx in enumerate(object_ids):
        variants = _oriented_masks(shapes[shape_idx])
        orientation_hint = used_orientations[i % len(used_orientations)]
        mask = variants[orientation_hint % len(variants)]

        placed = False
        for _ in range(300):
            top = rng.randint(0, height - mask.shape[0])
            left = rng.randint(main_c0, width - mask.shape[1])
            if _can_place(grid, mask, top, left):
                _place_mask(grid, mask, AZURE, top, left)
                placed = True
                break
        if not placed:
            return None

    main(grid)
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(300):
        try:
            grid = _gen_attempt(rng)
        except Exception:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
