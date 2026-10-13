from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_TOOLBOX = Path("/home/lixinyu/snap/new/solvers")
if str(_TOOLBOX) not in sys.path:
    sys.path.append(str(_TOOLBOX))

from arc_utils import Color, components_mask, crop_bbox, get_bg, paste, random_free_location


BG = Color.BLACK
PALETTE = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
GEN_RETRIES = 400
SHAPE_RETRIES = 80
LAYOUT_RETRIES = 80
INDEX_MODES = ("iid", "rows", "cols", "checker", "bands")


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _component_records(grid: np.ndarray, bg: int) -> list[dict]:
    records = []
    fg_mask = grid != bg
    for coords in components_mask(fg_mask, connectivity=4):
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        colors = sorted({int(grid[r, c]) for r, c in coords})
        records.append(
            {
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "colors": colors,
                "area": int(coords.shape[0]),
            }
        )
    return records


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _is_connected(mask: np.ndarray) -> bool:
    comps = components_mask(mask.astype(bool), connectivity=4)
    return len(comps) == 1


def _mask_key(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(int(v) for v in mask.ravel())


def _extract_scene(grid: np.ndarray) -> tuple[int, np.ndarray, int, dict[int, np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))
    records = _component_records(g, bg)
    multi = [rec for rec in records if len(rec["colors"]) >= 2]
    assert len(multi) == 1, "expected exactly one multicolor connected component"
    index_rec = multi[0]
    index_patch = crop_bbox(g, index_rec["bbox"])
    assert np.any(index_patch[0, :] != bg)
    assert np.any(index_patch[-1, :] != bg)
    assert np.any(index_patch[:, 0] != bg)
    assert np.any(index_patch[:, -1] != bg)

    mono = [rec for rec in records if rec is not index_rec]
    assert mono, "expected monochrome dictionary objects"
    side_pairs = {(rec["bbox"][2] - rec["bbox"][0] + 1, rec["bbox"][3] - rec["bbox"][1] + 1) for rec in mono}
    assert len(side_pairs) == 1, "all dictionary objects must share the same bbox size"
    (k_h, k_w), = side_pairs
    assert k_h == k_w, "dictionary objects must have square bboxes"
    k = int(k_h)

    masks: dict[int, np.ndarray] = {}
    for rec in mono:
        assert len(rec["colors"]) == 1, "dictionary objects must be monochrome"
        color = int(rec["colors"][0])
        assert color not in masks, "dictionary object colors must be unique"
        patch = crop_bbox(g, rec["bbox"])
        mask = patch == color
        assert mask.shape == (k, k)
        assert _is_connected(mask), "each dictionary shape must be 4-connected"
        masks[color] = mask

    used_colors = {int(v) for v in np.unique(index_patch) if int(v) != bg}
    assert len(used_colors) >= 2, "index grid must be multicolor"
    assert used_colors <= set(masks), "every non-black index color must have a matching dictionary object"
    return bg, index_patch, k, masks


def _expand_index(index_patch: np.ndarray, bg: int, k: int, masks: dict[int, np.ndarray]) -> np.ndarray:
    h, w = index_patch.shape
    out = np.full((h * k, w * k), bg, dtype=int)
    for r in range(h):
        for c in range(w):
            color = int(index_patch[r, c])
            if color == bg:
                continue
            block = out[r * k : (r + 1) * k, c * k : (c + 1) * k]
            block[masks[color]] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each pixel of the multicolor index rectangle into the same-color scene object shape.

    Input structure:
    - Black is the background.
    - Exactly one foreground connected component is multicolor; its bbox is the rectangular index grid and it may contain black cells inside that rectangle.
    - Every other foreground component is a separated monochrome object, with a unique color and a square k x k bbox.

    Rule:
    - Infer k from the shared bbox size of the monochrome objects.
    - Treat each monochrome object's occupancy pattern inside its k x k bbox as that color's stencil.
    - Crop the multicolor component's bbox as the index grid, then replace each non-black index cell with the matching color's stencil recolored in that same color; black index cells become all-black k x k blocks.

    Key insight:
    - The scattered monochrome scene objects are not targets to move; they form a diegetic legend whose colors index reusable k x k shape masks.
    """
    bg, index_patch, k, masks = _extract_scene(np.asarray(grid, dtype=int))
    return _expand_index(index_patch, bg, k, masks)


def _random_shape(k: int, *, require_non_full: bool) -> np.ndarray | None:
    min_cells = max(k + 1, (k * k + 1) // 2)
    max_cells = k * k - 1 if require_non_full else k * k
    if max_cells < min_cells:
        max_cells = min_cells
    for _ in range(SHAPE_RETRIES):
        target = random.randint(min_cells, max_cells)
        mask = np.ones((k, k), dtype=bool)
        cells = [(r, c) for r in range(k) for c in range(k)]
        random.shuffle(cells)
        for r, c in cells:
            if int(mask.sum()) <= target:
                break
            mask[r, c] = False
            if not _touches_all_sides(mask) or not _is_connected(mask):
                mask[r, c] = True
        if int(mask.sum()) == target and _touches_all_sides(mask) and _is_connected(mask):
            if require_non_full and bool(mask.all()):
                continue
            return mask
    return None


def _sample_unique_shapes(k: int, total: int, used_count: int) -> list[np.ndarray] | None:
    shapes: list[np.ndarray] = []
    seen = set()
    for idx in range(total):
        require_non_full = idx < used_count
        for _ in range(SHAPE_RETRIES):
            mask = _random_shape(k, require_non_full=require_non_full)
            if mask is None:
                continue
            key = _mask_key(mask)
            if key in seen:
                continue
            seen.add(key)
            shapes.append(mask)
            break
        else:
            return None
    return shapes


def _make_connected_index_mask(h: int, w: int, min_cells: int, include_black: bool) -> np.ndarray | None:
    mask = np.ones((h, w), dtype=bool)
    removable = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(removable)
    max_remove = h * w - min_cells
    if max_remove < 0:
        return None
    min_remove = 1 if include_black else 0
    if max_remove < min_remove:
        return None
    target_remove = random.randint(min_remove, max_remove)
    removed = 0
    for r, c in removable:
        if removed >= target_remove:
            break
        mask[r, c] = False
        if not _touches_all_sides(mask) or not _is_connected(mask):
            mask[r, c] = True
            continue
        removed += 1
    if removed != target_remove:
        return None
    return mask


def _fill_index_colors(active: np.ndarray, colors: list[int]) -> np.ndarray:
    h, w = active.shape
    patch = np.zeros((h, w), dtype=int)
    mode = random.choice(INDEX_MODES)
    if mode == "rows":
        row_colors = [random.choice(colors) for _ in range(h)]
        for r in range(h):
            patch[r, active[r]] = row_colors[r]
    elif mode == "cols":
        col_colors = [random.choice(colors) for _ in range(w)]
        for c in range(w):
            patch[active[:, c], c] = col_colors[c]
    elif mode == "checker":
        order = random.sample(colors, len(colors))
        offset = random.randrange(len(order))
        for r in range(h):
            for c in range(w):
                if active[r, c]:
                    patch[r, c] = order[(r + c + offset) % len(order)]
    elif mode == "bands":
        if random.random() < 0.5:
            split = random.randint(1, h - 1)
            band_colors = [random.choice(colors), random.choice(colors)]
            for r in range(h):
                color = band_colors[0] if r < split else band_colors[1]
                patch[r, active[r]] = color
        else:
            split = random.randint(1, w - 1)
            band_colors = [random.choice(colors), random.choice(colors)]
            for c in range(w):
                color = band_colors[0] if c < split else band_colors[1]
                patch[active[:, c], c] = color
    else:
        coords = np.argwhere(active)
        for r, c in coords:
            patch[r, c] = random.choice(colors)

    active_coords = [tuple(rc) for rc in np.argwhere(active)]
    random.shuffle(active_coords)
    for color, (r, c) in zip(colors, active_coords):
        patch[r, c] = color
    return patch


def _make_index_patch(h: int, w: int, colors: list[int], include_black: bool) -> np.ndarray | None:
    min_cells = len(colors) + (1 if include_black else 0)
    active = _make_connected_index_mask(h, w, min_cells, include_black)
    if active is None:
        return None
    patch = _fill_index_colors(active, colors)
    used = {int(v) for v in np.unique(patch) if int(v) != BG}
    if used != set(colors):
        return None
    if len(used) < 2:
        return None
    if include_black and not np.any(patch == BG):
        return None
    return patch


def _render_mask(mask: np.ndarray, color: int) -> np.ndarray:
    return np.where(mask, int(color), BG)


def _semantic_check(
    grid: np.ndarray,
    expected: np.ndarray,
    k: int,
    index_patch: np.ndarray,
    used_colors: list[int],
    masks: dict[int, np.ndarray],
    distractor_colors: list[int],
) -> bool:
    try:
        actual = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False
    if expected.shape != (index_patch.shape[0] * k, index_patch.shape[1] * k):
        return False
    if len({_mask_key(masks[c]) for c in used_colors}) < 2:
        return False
    if all(bool(masks[c].all()) for c in used_colors):
        return False
    naive = np.kron(index_patch, np.ones((k, k), dtype=int))
    if np.array_equal(expected, naive):
        return False
    if distractor_colors and any(color in np.unique(index_patch) for color in distractor_colors):
        return False
    for color in used_colors:
        if color not in expected:
            return False
    return True


def _generate_attempt() -> np.ndarray | None:
    k = random.randint(2, 5)
    max_dim = 30 // k
    if max_dim < 3:
        return None
    max_total_objects = 5 if k == 2 else 6

    index_h = random.randint(3, min(6, max_dim))
    index_w = random.randint(3, min(6, max_dim))
    max_colors = min(4, len(PALETTE) - 1, index_h * index_w - 1, max_total_objects)
    if max_colors < 2:
        return None
    used_count = random.randint(2, max_colors)
    distractor_count = 0
    max_distractors = min(2, len(PALETTE) - used_count, max_total_objects - used_count)
    if max_distractors > 0 and random.random() < 0.8:
        distractor_count = random.randint(1, max_distractors)
    total_objects = used_count + distractor_count
    colors = random.sample(PALETTE, total_objects)
    used_colors = colors[:used_count]
    distractor_colors = colors[used_count:]

    shapes = _sample_unique_shapes(k, total_objects, used_count)
    if shapes is None:
        return None
    masks = {color: shape for color, shape in zip(colors, shapes)}

    include_black = bool(random.random() < 0.7 and index_h >= 3 and index_w >= 3 and index_h * index_w > used_count)
    index_patch = _make_index_patch(index_h, index_w, used_colors, include_black)
    if index_patch is None:
        return None
    expected = _expand_index(index_patch, BG, k, masks)
    if max(expected.shape) > 30:
        return None

    min_h = max(index_h, k) + 4
    min_w = max(index_w, k) + 4
    for _ in range(LAYOUT_RETRIES):
        h = random.randint(max(16, min_h), 30)
        w = random.randint(max(16, min_w), 30)
        grid = np.full((h, w), BG, dtype=int)
        occ = np.zeros((h, w), dtype=int)
        try:
            index_loc = random_free_location(occ, np.ones(index_patch.shape, dtype=int), transparent=0, border=1, padding=1)
            paste(occ, np.ones(index_patch.shape, dtype=int), *index_loc)
            paste(grid, index_patch, *index_loc, transparent=BG)

            items = list(colors)
            random.shuffle(items)
            for color in items:
                shape = masks[color]
                reserve = np.ones((k, k), dtype=int)
                loc = random_free_location(occ, reserve, transparent=0, border=1, padding=1)
                paste(occ, reserve, *loc)
                paste(grid, _render_mask(shape, color), *loc, transparent=BG)
        except ValueError:
            continue

        if _semantic_check(grid, expected, k, index_patch, used_colors, masks, distractor_colors):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid scene-expansion puzzle")
