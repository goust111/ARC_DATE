from __future__ import annotations

import random

import numpy as np


BLUE = 1
GREY = 5
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SCENE_COLOR_POOL = [c for c in range(10) if c != GREY]
BLOCK_COLOR_POOL = [c for c in range(10) if c not in (BLUE, GREY)]
GEN_ATTEMPTS = 300


def _most_common_color(arr: np.ndarray) -> int:
    vals, counts = np.unique(arr, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


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
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].astype(bool)


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    out[rel[:, 0], rel[:, 1]] = True
    return out


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    tight = _crop_mask(mask.astype(bool))
    return tight.shape, tight.astype(np.uint8).tobytes()


def _shape_key_from_coords(coords: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return _shape_key(_coords_to_mask(coords))


def _find_separator_row(grid: np.ndarray) -> int:
    h = grid.shape[0]
    candidates: list[int] = []
    for r in range(h):
        row = grid[r]
        if not np.all(row == row[0]):
            continue
        color = int(row[0])
        if r == 0 or r == h - 1:
            continue
        if not np.any(grid[:r] == color):
            continue
        below_bg = _most_common_color(grid[r + 1 :])
        if below_bg == color:
            continue
        candidates.append(r)
    assert candidates, "expected a full-width separator row"
    return max(candidates)


def _find_color_blocks(legend: np.ndarray, legend_bg: int, template_color: int) -> list[tuple[int, int, int]]:
    h, w = legend.shape
    blocks: list[tuple[int, int, int]] = []
    for r in range(h - 1):
        for c in range(w - 1):
            color = int(legend[r, c])
            if color in (legend_bg, template_color):
                continue
            if not np.all(legend[r : r + 2, c : c + 2] == color):
                continue
            if r > 0 and np.any(legend[r - 1, c : c + 2] == color):
                continue
            if r + 2 < h and np.any(legend[r + 2, c : c + 2] == color):
                continue
            if c > 0 and np.any(legend[r : r + 2, c - 1] == color):
                continue
            if c + 2 < w and np.any(legend[r : r + 2, c + 2] == color):
                continue
            blocks.append((int(r), int(c), color))
    return blocks


def _parse_legend(legend: np.ndarray, legend_bg: int, template_color: int) -> dict[tuple[tuple[int, int], bytes], int]:
    comps = []
    for coords in _components(legend == template_color):
        comps.append({"coords": coords, "bbox": _bbox(coords), "key": _shape_key_from_coords(coords)})

    blocks = _find_color_blocks(legend, legend_bg=legend_bg, template_color=template_color)
    assert blocks, "expected at least one legend color block"

    mapping: dict[tuple[tuple[int, int], bytes], int] = {}
    used_components: set[int] = set()
    for br, bc, color in sorted(blocks):
        candidates: list[tuple[tuple[int, int, int], int]] = []
        for idx, comp in enumerate(comps):
            r0, c0, r1, c1 = comp["bbox"]
            if c0 < bc - 3 or c1 > bc - 1:
                continue
            if r0 < br - 1 or r1 > br + 2:
                continue
            center_dist = abs((r0 + r1) - (2 * br + 1))
            candidates.append(((center_dist, -(c1), idx), idx))
        assert candidates, "every legend block must have a template on its left"
        candidates.sort()
        idx = candidates[0][1]
        assert idx not in used_components, "legend template paired with multiple blocks"
        used_components.add(idx)
        key = comps[idx]["key"]
        if key in mapping:
            assert mapping[key] == color
        else:
            mapping[key] = int(color)
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace scene-object colors by looking up their cropped shape in the legend.

    The legend is above a full-width separator row. Each legend pair has a small
    grey template on blue background and a 2x2 color block immediately to its right.
    The output removes the legend and separator, leaving only the recolored scene.
    """
    g = np.asarray(grid, dtype=int)
    sep_row = _find_separator_row(g)
    legend = g[:sep_row]
    scene = g[sep_row + 1 :]
    assert legend.size and scene.size

    legend_bg = int(g[sep_row, 0])
    scene_bg = _most_common_color(scene)
    mapping = _parse_legend(legend, legend_bg=legend_bg, template_color=scene_bg)

    out = scene.copy()
    for coords in _components(scene != scene_bg):
        colors = {int(scene[r, c]) for r, c in coords}
        assert len(colors) == 1, "scene objects must be monochrome"
        old_color = int(next(iter(colors)))
        new_color = mapping.get(_shape_key_from_coords(coords), old_color)
        out[coords[:, 0], coords[:, 1]] = new_color
    return out


def _is_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    return len(_components(mask.astype(bool))) == 1


def _shape_library() -> list[np.ndarray]:
    seen: set[tuple[tuple[int, int], bytes]] = set()
    shapes: list[np.ndarray] = []
    for bits in range(1, 1 << 9):
        area = bits.bit_count()
        if area < 2 or area > 5:
            continue
        mask = np.zeros((3, 3), dtype=bool)
        for idx in range(9):
            if (bits >> idx) & 1:
                mask[idx // 3, idx % 3] = True
        if not _is_connected(mask):
            continue
        tight = _crop_mask(mask)
        key = _shape_key(tight)
        if key in seen:
            continue
        seen.add(key)
        shapes.append(tight)
    return shapes


SHAPE_LIBRARY = _shape_library()


def _mask_from_key(key: tuple[tuple[int, int], bytes]) -> np.ndarray:
    shape, payload = key
    return np.frombuffer(payload, dtype=np.uint8).reshape(shape).astype(bool)


def _dedupe_masks(masks: list[np.ndarray]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for mask in masks:
        key = _shape_key(mask)
        if key in seen:
            continue
        seen.add(key)
        out.append(_crop_mask(mask))
    return out


def _rotated_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants = []
    for k in range(1, 4):
        variants.append(_crop_mask(np.rot90(mask, k)))
    return _dedupe_masks(variants)


def _render_pair(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full((3, 5), BLUE, dtype=int)
    h, w = mask.shape
    r0 = random.randint(0, 3 - h)
    c0 = random.randint(0, 3 - w)
    patch[r0 : r0 + h, c0 : c0 + w][mask] = GREY
    br = random.randint(0, 1)
    patch[br : br + 2, 3:5] = int(color)
    return patch


def _choose_nonmatching_shapes(legend_shapes: list[np.ndarray], legend_keys: set[tuple[tuple[int, int], bytes]]) -> list[np.ndarray]:
    pool: list[np.ndarray] = []
    for mask in legend_shapes:
        for rotated in _rotated_variants(mask):
            if _shape_key(rotated) not in legend_keys:
                pool.append(rotated)
    for mask in SHAPE_LIBRARY:
        if _shape_key(mask) not in legend_keys:
            pool.append(mask)
    return _dedupe_masks(pool)


def _paste_masked(grid: np.ndarray, mask: np.ndarray, color: int, top: int, left: int) -> None:
    h, w = mask.shape
    window = grid[top : top + h, left : left + w]
    window[mask] = int(color)


def _place_object(scene: np.ndarray, blocked: np.ndarray, mask: np.ndarray, color: int) -> bool:
    h, w = mask.shape
    H, W = scene.shape
    positions = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
    random.shuffle(positions)
    coords = np.argwhere(mask)
    for top, left in positions:
        if any(blocked[top + r, left + c] for r, c in coords):
            continue
        _paste_masked(scene, mask, color, top, left)
        for r, c in coords:
            rr = top + int(r)
            cc = left + int(c)
            for nr in range(max(0, rr - 1), min(H, rr + 2)):
                for nc in range(max(0, cc - 1), min(W, cc + 2)):
                    blocked[nr, nc] = True
        return True
    return False


def _build_scene(
    scene_h: int,
    scene_w: int,
    specs: list[tuple[np.ndarray, int]],
) -> np.ndarray | None:
    scene = np.full((scene_h, scene_w), GREY, dtype=int)
    blocked = np.zeros((scene_h, scene_w), dtype=bool)
    order = list(specs)
    random.shuffle(order)
    for mask, color in order:
        if not _place_object(scene, blocked, mask, color):
            return None
    return scene


def _render_legend(
    grid_w: int,
    legend_shapes: list[np.ndarray],
    legend_colors: list[int],
) -> np.ndarray | None:
    legend = np.full((3, grid_w), BLUE, dtype=int)
    left_margin = random.randint(1, 2)
    col = left_margin
    for idx, (shape, color) in enumerate(zip(legend_shapes, legend_colors, strict=True)):
        patch = _render_pair(shape, color)
        if col + patch.shape[1] > grid_w - 1:
            return None
        legend[:, col : col + patch.shape[1]] = patch
        col += patch.shape[1]
        if idx != len(legend_shapes) - 1:
            col += random.randint(1, 2)
    if np.any(np.all(legend == BLUE, axis=1)):
        return None
    return legend


def _expected_scene(scene: np.ndarray, legend_map: dict[tuple[tuple[int, int], bytes], int]) -> np.ndarray:
    out = scene.copy()
    for coords in _components(scene != GREY):
        key = _shape_key_from_coords(coords)
        color = legend_map.get(key)
        if color is not None:
            out[coords[:, 0], coords[:, 1]] = int(color)
    return out


def _sample_color(exclude: set[int]) -> int:
    choices = [c for c in SCENE_COLOR_POOL if c not in exclude]
    return int(random.choice(choices))


def _gen_attempt() -> np.ndarray | None:
    entry_count = random.randint(2, 4)
    legend_shapes = random.sample(SHAPE_LIBRARY, entry_count)
    legend_keys = {_shape_key(shape) for shape in legend_shapes}
    unmatched_pool = _choose_nonmatching_shapes(legend_shapes, legend_keys)
    if len(unmatched_pool) < 2:
        return None

    legend_colors = random.sample(BLOCK_COLOR_POOL, entry_count)
    legend_map = {_shape_key(shape): color for shape, color in zip(legend_shapes, legend_colors, strict=True)}

    matched_specs: list[tuple[np.ndarray, int]] = []
    for shape, mapped_color in zip(legend_shapes, legend_colors, strict=True):
        original = _sample_color({GREY})
        if len(SCENE_COLOR_POOL) > 1 and random.random() < 0.75:
            tries = [c for c in SCENE_COLOR_POOL if c not in (GREY, mapped_color)]
            if tries:
                original = int(random.choice(tries))
        matched_specs.append((shape, original))

    extra_matched = random.randint(0, 2)
    for _ in range(extra_matched):
        idx = random.randrange(entry_count)
        shape = legend_shapes[idx]
        mapped_color = legend_colors[idx]
        original = _sample_color({GREY})
        if len(SCENE_COLOR_POOL) > 1:
            tries = [c for c in SCENE_COLOR_POOL if c not in (GREY, mapped_color)]
            if tries and random.random() < 0.75:
                original = int(random.choice(tries))
        matched_specs.append((shape, original))

    unmatched_count = random.randint(1, max(2, entry_count))
    unmatched_shapes = random.sample(unmatched_pool, unmatched_count)
    unmatched_specs = [(shape, _sample_color({GREY})) for shape in unmatched_shapes]

    scene_specs = matched_specs + unmatched_specs
    random.shuffle(scene_specs)

    min_w = 1 + entry_count * 5 + (entry_count - 1) + 1
    grid_w = random.randint(max(14, min_w), max(18, min_w + 8))
    scene_h = random.randint(10, 17)

    legend = _render_legend(grid_w, legend_shapes, legend_colors)
    if legend is None:
        return None
    scene = _build_scene(scene_h, grid_w, scene_specs)
    if scene is None:
        return None

    separator = np.full((1, grid_w), BLUE, dtype=int)
    grid = np.vstack([legend, separator, scene])
    expected = _expected_scene(scene, legend_map)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
