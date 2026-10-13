from __future__ import annotations

import random

import numpy as np


BLACK = 0
AZURE = 8

MIN_GRID_H = 22
MAX_GRID_H = 30
MIN_GRID_W = 24
MAX_GRID_W = 30
GEN_RETRIES = 400
PLACE_RETRIES = 120

NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
NEIGHBORS8 = (
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
)

CATEGORIES = ("rot4", "h", "v", "both", "none")
TEXTURE_COLORS = [1, 2, 3, 4, 6, 7, 9]

SHAPE_LIBRARY: dict[str, tuple[np.ndarray, ...]] = {
    "rot4": (
        np.array([[1, 1], [1, 1]], dtype=bool),
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
        np.ones((3, 3), dtype=bool),
    ),
    "both": (
        np.ones((2, 3), dtype=bool),
        np.ones((2, 4), dtype=bool),
        np.array(
            [
                [0, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
                [0, 1, 1, 1, 0],
            ],
            dtype=bool,
        ),
    ),
    "h": (
        np.array([[1, 1, 0], [1, 1, 1], [1, 1, 0]], dtype=bool),
        np.array([[1, 1, 0, 0], [1, 1, 1, 1], [1, 1, 0, 0]], dtype=bool),
        np.array([[1, 0, 0], [1, 1, 0], [1, 0, 0]], dtype=bool),
    ),
    "v": (
        np.array([[1, 1, 0], [1, 1, 1], [1, 1, 0]], dtype=bool).T,
        np.array([[1, 1, 0, 0], [1, 1, 1, 1], [1, 1, 0, 0]], dtype=bool).T,
        np.array([[1, 0, 0], [1, 1, 0], [1, 0, 0]], dtype=bool).T,
    ),
    "none": (
        np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
        np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
        np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool),
    ),
}


def _mode_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[counts.argmax()])


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _tight_crop(mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_from_mask(mask)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    dirs = NEIGHBORS4 if connectivity == 4 else NEIGHBORS8
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            src_r0 = max(0, -dr)
            src_r1 = min(h, h - dr)
            src_c0 = max(0, -dc)
            src_c1 = min(w, w - dc)
            dst_r0 = src_r0 + dr
            dst_r1 = src_r1 + dr
            dst_c0 = src_c0 + dc
            dst_c1 = src_c1 + dc
            out[dst_r0:dst_r1, dst_c0:dst_c1] |= mask[src_r0:src_r1, src_c0:src_c1]
    return out


def _outline_patch_from_interior(interior: np.ndarray) -> np.ndarray:
    padded = np.pad(interior.astype(bool), 1)
    outline = _dilate8(padded) & ~padded
    patch = np.full(outline.shape, AZURE, dtype=int)
    patch[outline] = BLACK
    return patch


def _symmetry_type(mask: np.ndarray) -> str:
    tight = _tight_crop(mask.astype(bool))
    has_h = bool(np.array_equal(tight, np.flipud(tight)))
    has_v = bool(np.array_equal(tight, np.fliplr(tight)))
    has_r90 = tight.shape[0] == tight.shape[1] and bool(np.array_equal(tight, np.rot90(tight)))
    if has_r90:
        return "rot4"
    if has_h and has_v:
        return "both"
    if has_h:
        return "h"
    if has_v:
        return "v"
    return "none"


def _extract_outline_objects(grid: np.ndarray, bg: int) -> list[dict]:
    mask = grid == BLACK
    objs: list[dict] = []
    for coords in _components(mask, connectivity=8):
        rows = coords[:, 0]
        cols = coords[:, 1]
        r0, c0, r1, c1 = int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())
        local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        local[rows - r0, cols - c0] = True
        open_cells = ~local
        outside = np.zeros_like(local, dtype=bool)
        stack: list[tuple[int, int]] = []
        for rr in range(local.shape[0]):
            for cc in (0, local.shape[1] - 1):
                if open_cells[rr, cc] and not outside[rr, cc]:
                    outside[rr, cc] = True
                    stack.append((rr, cc))
        for cc in range(local.shape[1]):
            for rr in (0, local.shape[0] - 1):
                if open_cells[rr, cc] and not outside[rr, cc]:
                    outside[rr, cc] = True
                    stack.append((rr, cc))
        while stack:
            rr, cc = stack.pop()
            for dr, dc in NEIGHBORS4:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < local.shape[0] and 0 <= nc < local.shape[1]:
                    if open_cells[nr, nc] and not outside[nr, nc]:
                        outside[nr, nc] = True
                        stack.append((nr, nc))
        interior = open_cells & ~outside
        if not interior.any():
            continue
        ir0, ic0, ir1, ic1 = _bbox_from_mask(interior)
        objs.append(
            {
                "bbox": (r0, c0, r1, c1),
                "outline_mask": local,
                "interior_mask": interior,
                "inner_bbox": (ir0, ic0, ir1, ic1),
                "tight_mask": interior[ir0 : ir1 + 1, ic0 : ic1 + 1].copy(),
            }
        )
    return objs


def _find_legend_and_objects(grid: np.ndarray, bg: int) -> tuple[list[dict], set[int], dict[str, np.ndarray]]:
    h, w = grid.shape
    objects = _extract_outline_objects(grid, bg)
    legend_indices: set[int] = set()
    legend_textures: dict[str, np.ndarray] = {}
    for idx, obj in enumerate(objects):
        r0, c0, r1, c1 = obj["bbox"]
        ir0, ic0, ir1, ic1 = obj["inner_bbox"]
        tex_h = int(ir1 - ir0 + 1)
        tex_w = int(ic1 - ic0 + 1)
        sample_r0 = int(r0 + ir0)
        sample_c0 = int(c1 + 2)
        if sample_r0 < 0 or sample_c0 < 0:
            continue
        if sample_r0 + tex_h > h or sample_c0 + tex_w > w:
            continue
        if c1 + 1 >= w:
            continue
        if not np.all(grid[sample_r0 : sample_r0 + tex_h, c1 + 1] == bg):
            continue
        texture = grid[sample_r0 : sample_r0 + tex_h, sample_c0 : sample_c0 + tex_w]
        if texture.shape != (tex_h, tex_w):
            continue
        if np.any(texture == bg) or np.any(texture == BLACK):
            continue
        if len(set(map(int, texture.ravel()))) < 2:
            continue
        category = _symmetry_type(obj["tight_mask"])
        assert category not in legend_textures
        legend_textures[category] = texture.copy()
        legend_indices.add(idx)
    return objects, legend_indices, legend_textures


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each hollow black outline with the legend entry of the same symmetry type,
    then fill the interior by tiling that entry's texture sample.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    objects, legend_indices, legend_textures = _find_legend_and_objects(g, bg)
    out = g.copy()
    for idx, obj in enumerate(objects):
        if idx in legend_indices:
            continue
        category = _symmetry_type(obj["tight_mask"])
        assert category in legend_textures
        texture = legend_textures[category]
        tr0, tc0, _tr1, _tc1 = obj["inner_bbox"]
        r0, c0, _r1, _c1 = obj["bbox"]
        interior = obj["interior_mask"]
        tex_h, tex_w = texture.shape
        for rr, cc in np.argwhere(interior):
            out[r0 + rr, c0 + cc] = int(texture[(rr - tr0) % tex_h, (cc - tc0) % tex_w])
    return out


def _variants(mask: np.ndarray) -> list[np.ndarray]:
    seen: set[bytes] = set()
    out: list[np.ndarray] = []
    for k in range(4):
        rot = np.rot90(mask, k)
        for candidate in (rot, np.fliplr(rot)):
            key = candidate.tobytes() + bytes(candidate.shape)
            if key in seen:
                continue
            seen.add(key)
            out.append(candidate.copy())
    return out


def _sample_shape(category: str, scale_choices: tuple[int, ...]) -> np.ndarray:
    base = random.choice(SHAPE_LIBRARY[category])
    options = [variant for variant in _variants(base) if _symmetry_type(variant) == category]
    mask = _tight_crop(random.choice(options))
    usable_scales = [scale for scale in scale_choices if scale == 1 or max(mask.shape) <= 3]
    scale = random.choice(usable_scales)
    if scale > 1:
        block = np.ones((scale, scale), dtype=bool)
        mask = np.kron(mask, block).astype(bool)
    return mask


def _texture_pattern(kind: str, colors: list[int], shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    out = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            if kind == "checker":
                idx = (r + c) % len(colors)
            elif kind == "vstripe":
                idx = c % len(colors)
            elif kind == "hstripe":
                idx = r % len(colors)
            elif kind == "diag_down":
                idx = (r + 2 * c) % len(colors)
            elif kind == "diag_up":
                idx = (r - c) % len(colors)
            else:
                idx = ((r // 2) + (c // 2)) % len(colors)
            out[r, c] = colors[idx]
    return out


def _sample_texture(shape: tuple[int, int], used: set[bytes]) -> np.ndarray:
    kinds = ("checker", "vstripe", "hstripe", "diag_down", "diag_up", "block")
    for _ in range(80):
        n_colors = random.choice((2, 2, 3))
        colors = random.sample(TEXTURE_COLORS, n_colors)
        texture = _texture_pattern(random.choice(kinds), colors, shape)
        if len(set(map(int, texture.ravel()))) < 2:
            continue
        key = texture.tobytes() + bytes(texture.shape)
        if key in used:
            continue
        used.add(key)
        return texture
    raise RuntimeError("failed to sample unique texture")


def _can_place(reserved: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> bool:
    h, w = reserved.shape
    if top < 0 or left < 0 or top + height > h or left + width > w:
        return False
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    return not bool(reserved[r0:r1, c0:c1].any())


def _reserve(reserved: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> None:
    h, w = reserved.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    reserved[r0:r1, c0:c1] = True


def _paste_non_bg(dst: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    h, w = patch.shape
    view = dst[top : top + h, left : left + w]
    mask = patch != bg
    view[mask] = patch[mask]


def _fill_target(out: np.ndarray, top: int, left: int, interior: np.ndarray, texture: np.ndarray) -> None:
    tex_h, tex_w = texture.shape
    for rr, cc in np.argwhere(interior):
        out[top + 1 + rr, left + 1 + cc] = int(texture[rr % tex_h, cc % tex_w])


def _validate_library() -> None:
    for category, masks in SHAPE_LIBRARY.items():
        for mask in masks:
            assert _symmetry_type(mask) == category


_validate_library()


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    bg = AZURE
    legend_shapes = {category: _sample_shape(category, (1,)) for category in CATEGORIES}
    target_categories = list(CATEGORIES)
    extra_count = random.randint(0, 1)
    target_categories.extend(random.choices(CATEGORIES, k=extra_count))
    random.shuffle(target_categories)
    target_shapes = {i: _sample_shape(category, (1, 1, 2)) for i, category in enumerate(target_categories)}

    legend_entries: list[dict] = []
    used_textures: set[bytes] = set()
    for category in random.sample(list(CATEGORIES), len(CATEGORIES)):
        interior = legend_shapes[category]
        outline = _outline_patch_from_interior(interior)
        texture = _sample_texture(interior.shape, used_textures)
        legend_entries.append(
            {
                "category": category,
                "interior": interior,
                "outline": outline,
                "texture": texture,
                "group_h": int(outline.shape[0]),
                "group_w": int(outline.shape[1] + 1 + texture.shape[1]),
            }
        )

    target_entries: list[dict] = []
    for idx, category in enumerate(target_categories):
        interior = target_shapes[idx]
        outline = _outline_patch_from_interior(interior)
        target_entries.append(
            {
                "category": category,
                "interior": interior,
                "outline": outline,
                "h": int(outline.shape[0]),
                "w": int(outline.shape[1]),
            }
        )
    target_entries.sort(key=lambda entry: -(entry["h"] * entry["w"]))

    legend_height = sum(entry["group_h"] for entry in legend_entries) + (len(legend_entries) - 1)
    legend_width = max(entry["group_w"] for entry in legend_entries)
    max_target_w = max(entry["w"] for entry in target_entries)

    min_h = max(MIN_GRID_H, legend_height)
    min_w = max(MIN_GRID_W, legend_width + max_target_w + 4)
    if min_h > MAX_GRID_H or min_w > MAX_GRID_W:
        raise RuntimeError("sampled shapes do not fit the canvas bounds")

    h = random.randint(min_h, MAX_GRID_H)
    w = random.randint(min_w, MAX_GRID_W)
    grid = np.full((h, w), bg, dtype=int)
    expected = grid.copy()
    reserved = np.zeros((h, w), dtype=bool)

    legend_left = random.randint(0, 1)
    legend_top = random.randint(0, h - legend_height)
    cursor = legend_top
    legend_textures: dict[str, np.ndarray] = {}
    legend_right = 0
    for entry in legend_entries:
        outline = entry["outline"]
        texture = entry["texture"]
        _paste_non_bg(grid, outline, cursor, legend_left, bg)
        _paste_non_bg(expected, outline, cursor, legend_left, bg)
        tex_top = cursor + 1
        tex_left = legend_left + outline.shape[1] + 1
        grid[tex_top : tex_top + texture.shape[0], tex_left : tex_left + texture.shape[1]] = texture
        expected[tex_top : tex_top + texture.shape[0], tex_left : tex_left + texture.shape[1]] = texture
        _reserve(reserved, cursor, legend_left, entry["group_h"], entry["group_w"], pad=1)
        legend_right = max(legend_right, tex_left + texture.shape[1] - 1)
        legend_textures[entry["category"]] = texture
        cursor += entry["group_h"] + 1

    scene_left = legend_right + 2
    if scene_left >= w - 2:
        raise RuntimeError("not enough room for target shapes")

    for entry in target_entries:
        if scene_left > w - entry["w"]:
            raise RuntimeError("canvas too narrow for sampled target shapes")
        candidates: list[tuple[int, int]] = []
        for top in range(0, h - entry["h"] + 1):
            for left in range(scene_left, w - entry["w"] + 1):
                if _can_place(reserved, top, left, entry["h"], entry["w"], pad=1):
                    candidates.append((top, left))
        if not candidates:
            raise RuntimeError("failed to place target shape")
        top, left = random.choice(candidates)
        _paste_non_bg(grid, entry["outline"], top, left, bg)
        _paste_non_bg(expected, entry["outline"], top, left, bg)
        _fill_target(expected, top, left, entry["interior"], legend_textures[entry["category"]])
        _reserve(reserved, top, left, entry["h"], entry["w"], pad=1)

    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            grid, expected = _generate_case()
        except RuntimeError:
            continue
        try:
            actual = main(grid)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
