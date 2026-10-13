from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BACKGROUND = 0
MAX_GENERATION_ATTEMPTS = 400
KEY_TEMPLATES = (
    (2, 3, 4, 4, 3, 2, 1, 1, 2, 2, 3, 4, 4, 3, 2),
    (3, 4, 5, 5, 4, 3, 1, 1, 1, 2, 3, 4, 5, 5, 4, 3),
    (2, 3, 4, 5, 5, 4, 2, 1, 1, 2, 3, 4, 5, 4, 3),
)
PAIR_COUNT_OPTIONS = (1, 2, 2, 3)
TARGET_DIM_OPTIONS = (
    (4, 8),
    (5, 7),
    (5, 8),
    (6, 6),
    (6, 7),
)

Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: BBox) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q: deque[Coord] = deque([(r, c)])
            seen[r, c] = True
            cells: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    return out


def _find_foreground_components(grid: np.ndarray, bg: int) -> list[dict]:
    parts = []
    for coords in _components(grid != bg):
        parts.append({"coords": coords, "bbox": _bbox_coords(coords), "area": int(coords.shape[0])})
    return parts


def _mask_from_coords(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _extract_swap_pairs(patch: np.ndarray, bg: int) -> tuple[list[tuple[int, int]], list[tuple[int, int, int]]]:
    h, w = patch.shape
    raw: list[tuple[int, int, int]] = []
    unique: set[tuple[int, int]] = set()
    for r in range(h):
        for c in range(w // 2):
            mc = w - 1 - c
            left = int(patch[r, c])
            right = int(patch[r, mc])
            if left == right:
                continue
            assert left != bg and right != bg, "shape symmetry must be preserved; only colors may differ"
            raw.append((r, left, right))
            unique.add(tuple(sorted((left, right))))

    assert raw, "expected at least one mirror defect"
    pairs = sorted(unique)
    used: set[int] = set()
    for a, b in pairs:
        assert a != b
        assert a not in used and b not in used, "color pairs must be disjoint"
        used.add(a)
        used.add(b)
    return pairs, raw


def _apply_swaps(grid: np.ndarray, target_mask: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    out = grid.copy()
    for a, b in pairs:
        mask_a = (grid == a) & target_mask
        mask_b = (grid == b) & target_mask
        out[mask_a] = b
        out[mask_b] = a
    return out


def _vertical_mismatch_count(patch: np.ndarray) -> int:
    h, w = patch.shape
    mismatches = 0
    for r in range(h):
        for c in range(w // 2):
            if int(patch[r, c]) != int(patch[r, w - 1 - c]):
                mismatches += 1
    return mismatches


def main(grid: np.ndarray) -> np.ndarray:
    """Read bidirectional color-swap pairs from a nearly mirror-symmetric key and apply them to the separate target object.

    Input structure:
    - The dominant background separates exactly two foreground components: a larger key-shaped component and a smaller multicolor target object.
    - The key is almost vertically mirror-symmetric inside its own bounding box; only a few mirrored cell pairs have different colors.
    - The target object uses the defect-pair colors, plus optionally one extra base color that is not part of any swap pair.

    Rule:
    - Identify the larger foreground component as the key and crop it to its own bounding box.
    - Compare each key cell with its vertical mirror partner inside that bounding box; every color mismatch defines one unordered two-color swap pair.
    - Simultaneously swap each such color pair only inside the smaller target component, leaving the key, its defects, and the background unchanged.

    Key insight:
    - The key's asymmetries are not damage to repair; they are the hidden legend encoding the target recoloring swaps.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    components = _find_foreground_components(g, bg)
    assert len(components) == 2, "expected exactly two foreground objects"

    components.sort(key=lambda comp: (comp["area"], comp["bbox"][2] - comp["bbox"][0], comp["bbox"][3] - comp["bbox"][1]), reverse=True)
    key = components[0]
    target = components[1]

    key_patch = _crop_bbox(g, key["bbox"])
    pairs, _raw = _extract_swap_pairs(key_patch, bg)

    target_mask = _mask_from_coords(g.shape, target["coords"])
    return _apply_swaps(g, target_mask, pairs)


def _make_key_patch(key_body: int, pair_specs: list[tuple[int, int]]) -> tuple[np.ndarray, list[dict]]:
    extents = list(random.choice(KEY_TEMPLATES))
    h = len(extents)
    w = 2 * max(extents) + 1
    mid = w // 2

    patch = np.full((h, w), BACKGROUND, dtype=int)
    for r, extent in enumerate(extents):
        patch[r, mid - extent : mid + extent + 1] = key_body

    depth_by_row: dict[int, list[int]] = {}
    for r in range(1, h - 1):
        depths: list[int] = []
        for depth in range(1, extents[r]):
            if depth <= extents[r - 1] and depth <= extents[r + 1]:
                depths.append(depth)
        if depths:
            depth_by_row[r] = depths

    needed_rows = 2 * len(pair_specs)
    if len(depth_by_row) < needed_rows:
        raise AssertionError("not enough key interior rows for defect placement")

    chosen_rows = sorted(random.sample(list(depth_by_row), needed_rows))
    placements: list[dict] = []
    for idx, (a, b) in enumerate(pair_specs):
        row_a = chosen_rows[2 * idx]
        row_b = chosen_rows[2 * idx + 1]
        depth_a = random.choice(depth_by_row[row_a])
        depth_b = random.choice(depth_by_row[row_b])

        left_a = mid - depth_a
        right_a = mid + depth_a
        left_b = mid - depth_b
        right_b = mid + depth_b

        patch[row_a, left_a] = a
        patch[row_a, right_a] = b
        patch[row_b, left_b] = b
        patch[row_b, right_b] = a
        placements.append(
            {
                "pair": tuple(sorted((a, b))),
                "oriented": ((row_a, int(left_a), a, b), (row_b, int(left_b), b, a)),
            }
        )
    return patch, placements


def _pick_target_dims(required_area: int, max_area: int) -> tuple[int, int]:
    choices = [(h, w) for h, w in TARGET_DIM_OPTIONS if required_area <= h * w <= max_area]
    if not choices:
        raise AssertionError("no target size fits the requested area bounds")
    return random.choice(choices)


def _make_target_mask(h: int, w: int) -> np.ndarray:
    mask = np.ones((h, w), dtype=bool)
    corner = random.choice(((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)))
    if random.random() < 0.75:
        mask[corner] = False
    if random.random() < 0.35:
        r, c = corner
        if r == 0 and c == 0 and w >= 4:
            mask[0, 1] = False
        elif r == 0 and c == w - 1 and w >= 4:
            mask[0, w - 2] = False
        elif r == h - 1 and c == 0 and w >= 4:
            mask[h - 1, 1] = False
        elif w >= 4:
            mask[h - 1, w - 2] = False
    return mask


def _grow_cluster(mask: np.ndarray, reserved: np.ndarray, size: int) -> list[Coord] | None:
    free = np.argwhere(mask & ~reserved)
    if free.shape[0] < size:
        return None

    start = tuple(map(int, free[random.randrange(free.shape[0])]))
    cluster = {start}
    frontier = [start]
    while len(cluster) < size and frontier:
        rr, cc = random.choice(frontier)
        options = []
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and bool(mask[nr, nc]) and not bool(reserved[nr, nc]) and (nr, nc) not in cluster:
                options.append((nr, nc))
        if options:
            cluster.add(random.choice(options))
        else:
            frontier.remove((rr, cc))
            continue
        frontier = list(cluster)

    if len(cluster) != size:
        return None
    return sorted(cluster)


def _make_target_patch(pair_colors: list[int], target_base: int | None, key_area: int) -> tuple[np.ndarray, dict]:
    overlay_colors = list(pair_colors if target_base is not None else pair_colors[1:])
    min_fill_visible = 4
    required_area = 2 * len(overlay_colors) + min_fill_visible + 4
    max_area = key_area - 8
    h, w = _pick_target_dims(required_area, max_area)

    for _ in range(60):
        mask = _make_target_mask(h, w)
        area = int(mask.sum())
        if area < required_area:
            continue
        fill_color = int(target_base if target_base is not None else pair_colors[0])
        patch = np.full((h, w), BACKGROUND, dtype=int)
        patch[mask] = fill_color

        reserved = np.zeros((h, w), dtype=bool)
        sizes: list[int] = []
        free_budget = area - min_fill_visible
        for i, _color in enumerate(overlay_colors):
            remaining = len(overlay_colors) - i - 1
            max_size = min(5, free_budget - 2 * remaining)
            if max_size < 2:
                break
            size = random.randint(2, max_size)
            sizes.append(size)
            free_budget -= size
        else:
            placed: dict[int, list[Coord]] = {}
            ok = True
            for color, size in zip(overlay_colors, sizes):
                cluster = _grow_cluster(mask, reserved, size)
                if cluster is None:
                    ok = False
                    break
                for r, c in cluster:
                    patch[r, c] = color
                    reserved[r, c] = True
                placed[int(color)] = cluster

            if not ok:
                continue
            if fill_color not in map(int, patch[mask]):
                continue
            if _vertical_mismatch_count(patch) < 3:
                # The target should not masquerade as another nearly symmetric legend.
                continue
            return patch, {"mask": mask, "fill": fill_color, "placed": placed}

    raise AssertionError("failed to build a target patch")


def _paste(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    dst[top : top + h, left : left + w] = patch


def _compose_scene(key_patch: np.ndarray, target_patch: np.ndarray) -> tuple[np.ndarray, dict]:
    kh, kw = key_patch.shape
    th, tw = target_patch.shape
    horizontal = random.random() < 0.6

    if horizontal:
        gap = random.randint(2, 4)
        left_pad = random.randint(1, 3)
        right_pad = random.randint(1, 3)
        top_pad = random.randint(1, 3)
        bottom_pad = random.randint(1, 3)
        inner_h = max(kh, th)
        grid_h = top_pad + inner_h + bottom_pad
        grid_w = left_pad + kw + gap + tw + right_pad
        if grid_h > 30 or grid_w > 30:
            raise AssertionError("horizontal layout exceeds ARC bounds")

        key_top = top_pad + random.randint(0, inner_h - kh)
        target_top = top_pad + random.randint(0, inner_h - th)
        if random.random() < 0.5:
            key_left = left_pad
            target_left = left_pad + kw + gap
        else:
            target_left = left_pad
            key_left = left_pad + tw + gap
    else:
        gap = random.randint(2, 3)
        left_pad = random.randint(1, 2)
        right_pad = random.randint(1, 2)
        top_pad = random.randint(1, 2)
        bottom_pad = random.randint(1, 2)
        inner_w = max(kw, tw)
        grid_h = top_pad + kh + gap + th + bottom_pad
        grid_w = left_pad + inner_w + right_pad
        if grid_h > 30 or grid_w > 30:
            raise AssertionError("vertical layout exceeds ARC bounds")

        key_left = left_pad + random.randint(0, inner_w - kw)
        target_left = left_pad + random.randint(0, inner_w - tw)
        if random.random() < 0.5:
            key_top = top_pad
            target_top = top_pad + kh + gap
        else:
            target_top = top_pad
            key_top = top_pad + th + gap

    grid = np.full((grid_h, grid_w), BACKGROUND, dtype=int)
    _paste(grid, key_patch, key_top, key_left)
    _paste(grid, target_patch, target_top, target_left)
    return grid, {
        "key_bbox": (key_top, key_left, key_top + kh - 1, key_left + kw - 1),
        "target_bbox": (target_top, target_left, target_top + th - 1, target_left + tw - 1),
    }


def _expected_output(grid: np.ndarray, target_bbox: BBox, pairs: list[tuple[int, int]]) -> np.ndarray:
    patch = _crop_bbox(grid, target_bbox)
    target_mask = patch != BACKGROUND
    global_mask = np.zeros_like(grid, dtype=bool)
    r0, c0, r1, c1 = target_bbox
    global_mask[r0 : r1 + 1, c0 : c1 + 1] = target_mask
    return _apply_swaps(grid, global_mask, pairs)


def _validate_scene(grid: np.ndarray, meta: dict) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2 and max(grid.shape) <= 30

    key_bbox = meta["layout"]["key_bbox"]
    target_bbox = meta["layout"]["target_bbox"]
    key_patch = _crop_bbox(grid, key_bbox)
    target_patch = _crop_bbox(grid, target_bbox)
    intended_pairs = sorted(meta["pairs"])

    fg = _find_foreground_components(grid, BACKGROUND)
    assert len(fg) == 2, "scene must contain exactly two components"
    areas = sorted(comp["area"] for comp in fg)
    assert areas[1] > areas[0], "key must be strictly larger than target"

    extracted_pairs, raw = _extract_swap_pairs(key_patch, BACKGROUND)
    assert extracted_pairs == intended_pairs, "key defects must decode exactly the intended swap pairs"

    orientation_seen: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for _row, left, right in raw:
        pair = tuple(sorted((left, right)))
        orientation_seen.setdefault(pair, set()).add((left, right))
    for pair in intended_pairs:
        a, b = pair
        assert orientation_seen.get(pair) == {(a, b), (b, a)}, "each pair must appear in both directions"

    target_colors = set(map(int, target_patch[target_patch != BACKGROUND]))
    required_target = set(meta["pair_colors"])
    if meta["target_base"] is not None:
        required_target.add(int(meta["target_base"]))
    assert target_colors == required_target, "target colors must be exactly the swap colors plus optional base"
    for color in meta["pair_colors"]:
        assert int((target_patch == color).sum()) >= 1, "every swap color must appear on the target"

    expected = _expected_output(grid, target_bbox, intended_pairs)
    actual = main(grid)
    assert np.array_equal(actual, expected), "main() must match the scene semantics"
    assert np.array_equal(actual[key_bbox[0] : key_bbox[2] + 1, key_bbox[1] : key_bbox[3] + 1], key_patch), "key must remain unchanged"
    assert not np.array_equal(actual, grid), "target recoloring must change the output"

    wrong_all_fg = grid.copy()
    wrong_all_fg[wrong_all_fg != BACKGROUND] = _apply_swaps(grid, grid != BACKGROUND, intended_pairs)[wrong_all_fg != BACKGROUND]
    assert not np.array_equal(wrong_all_fg, expected), "recoloring the key would be a wrong shortcut"


def _gen_attempt() -> tuple[np.ndarray, dict] | None:
    pair_count = random.choice(PAIR_COUNT_OPTIONS)
    use_target_base = random.random() < 0.7

    needed_colors = 1 + 2 * pair_count + (1 if use_target_base else 0)
    colors = random.sample(list(range(1, 10)), needed_colors)
    key_body = colors[0]
    pair_colors = colors[1 : 1 + 2 * pair_count]
    target_base = colors[-1] if use_target_base else None
    pair_specs = [tuple(pair_colors[2 * i : 2 * i + 2]) for i in range(pair_count)]
    intended_pairs = [tuple(sorted(pair)) for pair in pair_specs]

    try:
        key_patch, defect_meta = _make_key_patch(key_body, pair_specs)
    except AssertionError:
        return None

    key_area = int((key_patch != BACKGROUND).sum())
    try:
        target_patch, target_meta = _make_target_patch(pair_colors, target_base, key_area)
        grid, layout_meta = _compose_scene(key_patch, target_patch)
    except AssertionError:
        return None

    meta = {
        "pairs": intended_pairs,
        "pair_colors": list(pair_colors),
        "target_base": target_base,
        "layout": layout_meta,
        "defects": defect_meta,
        "target": target_meta,
    }
    try:
        _validate_scene(grid, meta)
    except AssertionError:
        return None
    return grid, meta


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            grid, _meta = result
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
