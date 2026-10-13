from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
FRAME_COLOR = 5
SPRITE_POOL = [1, 2, 3, 4]
LATTICE_STEP = 2
GEN_ATTEMPTS = 800


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, mask: np.ndarray | None = None) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if mask is None:
        view[:] = patch
    else:
        view[np.asarray(mask, dtype=bool)] = patch[np.asarray(mask, dtype=bool)]


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _is_sparse_rect_perimeter(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = bbox_coords(coords)
    rel = coords - np.array([r0, c0])
    if np.any(rel % 2 != 0):
        return False

    h = (r1 - r0) // 2 + 1
    w = (c1 - c0) // 2 + 1
    if h < 2 or w < 2:
        return False
    expected = 2 * h + 2 * w - 4
    if coords.shape[0] != expected:
        return False

    pts = {tuple(map(int, p)) for p in coords}
    for r in range(r0, r1 + 1, 2):
        for c in range(c0, c1 + 1, 2):
            on_border = r in (r0, r1) or c in (c0, c1)
            if ((r, c) in pts) != on_border:
                return False
    return True


def _logical_to_dense(logical: np.ndarray) -> np.ndarray:
    logical = np.asarray(logical, dtype=bool)
    return np.kron(logical, np.ones((2, 2), dtype=bool))[: 2 * logical.shape[0] - 1, : 2 * logical.shape[1] - 1]


def _dense_mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(coords)
    rel = coords - np.array([r0, c0])
    assert np.all(rel % 2 == 0), "sparse sprite pixels must lie on a step-2 lattice"
    logical_h = (r1 - r0) // 2 + 1
    logical_w = (c1 - c0) // 2 + 1
    logical = np.zeros((logical_h, logical_w), dtype=bool)
    for dr, dc in rel:
        logical[int(dr) // 2, int(dc) // 2] = True
    return _logical_to_dense(logical)


def _stamp_sparse_frame(grid: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = frame_bbox
    grid[r0 : r1 + 1 : 2, c0] = FRAME_COLOR
    grid[r0 : r1 + 1 : 2, c1] = FRAME_COLOR
    grid[r0, c0 : c1 + 1 : 2] = FRAME_COLOR
    grid[r1, c0 : c1 + 1 : 2] = FRAME_COLOR


def _stamp_sparse_sprite(grid: np.ndarray, logical: np.ndarray, anchor: tuple[int, int], color: int) -> None:
    r0, c0 = anchor
    pts = np.argwhere(logical)
    for rr, cc in pts:
        grid[r0 + 2 * int(rr), c0 + 2 * int(cc)] = int(color)


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(coords)
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def _transform_mask(mask: np.ndarray, rot_k: int, flip: bool) -> np.ndarray:
    out = np.rot90(mask, rot_k)
    if flip:
        out = np.fliplr(out)
    return _normalize_mask(out)


BASE_MASKS: dict[str, np.ndarray] = {
    "line4": np.array([[1, 1, 1, 1]], dtype=bool),
    "block2": np.array([[1, 1], [1, 1]], dtype=bool),
    "L4": np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    "T4": np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    "line5": np.array([[1, 1, 1, 1, 1]], dtype=bool),
    "P5": np.array([[1, 1], [1, 1], [1, 0]], dtype=bool),
    "plus5": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    "rect6": np.array([[1, 1, 1], [1, 1, 1]], dtype=bool),
    "ring8": np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),
}


SHAPE_INFO = {
    name: {
        "name": name,
        "logical": mask,
        "sparse_count": int(mask.sum()),
        "dense_area": int(_logical_to_dense(mask).sum()),
    }
    for name, mask in BASE_MASKS.items()
}


def _sprite_sort_key(sprite: dict) -> tuple[int, int, int]:
    return int(sprite["area"]), int(sprite["bbox"][0]), int(sprite["bbox"][1])


def _source_order_key(sprite: dict, horizontal: bool) -> tuple[int, int]:
    if horizontal:
        return int(sprite["bbox"][1]), int(sprite["bbox"][0])
    return int(sprite["bbox"][0]), int(sprite["bbox"][1])


def _detect_objects(grid: np.ndarray) -> tuple[int, tuple[int, int, int, int], list[dict], int]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    non_bg = colors_present(g, exclude={bg})
    assert len(non_bg) >= 4, "expected one frame and at least three sprites"

    frame_candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    coords_by_color = {color: np.argwhere(g == color) for color in non_bg}
    for color, coords in coords_by_color.items():
        if coords.size and _is_sparse_rect_perimeter(coords):
            frame_candidates.append((int(color), bbox_coords(coords)))
    assert frame_candidates, "expected a sparse rectangular frame"

    max_area = max(_bbox_area(bbox) for _color, bbox in frame_candidates)
    largest = [(color, bbox) for color, bbox in frame_candidates if _bbox_area(bbox) == max_area]
    assert len(largest) == 1, "frame must be the unique largest sparse rectangle"
    frame_color, frame_bbox = largest[0]

    sprites = []
    for color in non_bg:
        if color == frame_color:
            continue
        coords = coords_by_color[color]
        dense = _dense_mask_from_coords(coords)
        sprites.append(
            {
                "color": int(color),
                "bbox": bbox_coords(coords),
                "mask": dense,
                "area": int(dense.sum()),
                "sparse_count": int(coords.shape[0]),
            }
        )
    assert 3 <= len(sprites) <= 4, "expected 3-4 monochrome sparse sprites"
    return int(bg), frame_bbox, sprites, int(frame_color)


def _render_solution(grid: np.ndarray, *, order_mode: str) -> np.ndarray:
    bg, frame_bbox, sprites, frame_color = _detect_objects(grid)
    r0, c0, r1, c1 = frame_bbox
    interior_h = r1 - r0 - 1
    interior_w = c1 - c0 - 1
    horizontal = interior_w >= interior_h

    if order_mode == "area":
        ordered = sorted(sprites, key=_sprite_sort_key)
    elif order_mode == "source":
        ordered = sorted(sprites, key=lambda sp: _source_order_key(sp, horizontal))
    else:
        raise ValueError(order_mode)

    out = np.asarray(grid, dtype=int).copy()
    for sprite in sprites:
        rr0, cc0, rr1, cc1 = sprite["bbox"]
        out[rr0 : rr1 + 1, cc0 : cc1 + 1][out[rr0 : rr1 + 1, cc0 : cc1 + 1] == sprite["color"]] = bg

    out[r0 : r1 + 1, c0 : c1 + 1] = bg
    out[r0, c0 : c1 + 1] = frame_color
    out[r1, c0 : c1 + 1] = frame_color
    out[r0 : r1 + 1, c0] = frame_color
    out[r0 : r1 + 1, c1] = frame_color

    if horizontal:
        total_along = sum(int(sp["mask"].shape[1]) for sp in ordered)
        slack = interior_w - total_along
        assert slack > 0 and slack % (len(ordered) + 1) == 0
        gap = slack // (len(ordered) + 1)
        cursor = c0 + 1 + gap
        for sp in ordered:
            mask = sp["mask"]
            h, w = mask.shape
            cross_slack = interior_h - h
            assert cross_slack >= 0 and cross_slack % 2 == 0
            rr = r0 + 1 + cross_slack // 2
            patch = np.full((h, w), int(sp["color"]), dtype=int)
            paste(out, patch, rr, cursor, mask=mask)
            cursor += w + gap
    else:
        total_along = sum(int(sp["mask"].shape[0]) for sp in ordered)
        slack = interior_h - total_along
        assert slack > 0 and slack % (len(ordered) + 1) == 0
        gap = slack // (len(ordered) + 1)
        cursor = r0 + 1 + gap
        for sp in ordered:
            mask = sp["mask"]
            h, w = mask.shape
            cross_slack = interior_w - w
            assert cross_slack >= 0 and cross_slack % 2 == 0
            cc = c0 + 1 + cross_slack // 2
            patch = np.full((h, w), int(sp["color"]), dtype=int)
            paste(out, patch, cursor, cc, mask=mask)
            cursor += h + gap
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Densify sparse sprites, sort them by densified area, and repack them into the sparse frame.

    Input structure:
    - A black background contains one large sparse rectangular frame and 3-4 external sparse monochrome sprites.
    - The frame is a grey step-2 rectangular perimeter; each sprite is a single non-grey color drawn on its own step-2 lattice.
    - Different sprites may show similarly sparse visible pixel counts even when their densified solid areas differ.

    Rule:
    - Find the unique largest sparse rectangular perimeter and thicken it into a solid one-pixel border, clearing its interior.
    - Densify every external sparse sprite by filling the gaps between its step-2 lattice pixels inside its bounding box.
    - Compute each densified sprite's pixel area, sort sprites by ascending area, and break impossible ties by original top row.
    - Place the sorted dense sprites inside the frame with equal wall/object gaps along the long axis and centered on the cross axis, then erase their original locations.

    Key insight:
    - The sorting signal is hidden in the densified shapes rather than in the raw sparse pixels, so densification must happen before ranking and slot assignment.
    """
    return _render_solution(grid, order_mode="area")


def _choose_shapes(rng: random.Random, n_sprites: int) -> list[dict] | None:
    vis4 = [SHAPE_INFO[name] for name in ("line4", "block2", "L4", "T4")]
    pair = rng.sample(vis4, 2)
    pair_areas = {item["dense_area"] for item in pair}
    if len(pair_areas) != 2:
        return None

    chosen = [dict(item) for item in pair]
    used_areas = set(pair_areas)
    used_names = {item["name"] for item in pair}

    pool = [info for info in SHAPE_INFO.values() if info["dense_area"] not in used_areas and info["name"] not in used_names]
    if len(pool) < n_sprites - 2:
        return None
    chosen.extend(dict(item) for item in rng.sample(pool, n_sprites - 2))

    rng.shuffle(chosen)
    return chosen


def _realize_shapes(rng: random.Random, shape_infos: list[dict]) -> list[dict]:
    realized = []
    for info in shape_infos:
        logical = _transform_mask(info["logical"], rng.randrange(4), bool(rng.randrange(2)))
        dense = _logical_to_dense(logical)
        realized.append(
            {
                "name": info["name"],
                "logical": logical,
                "dense": dense,
                "dense_area": int(dense.sum()),
                "sparse_count": int(logical.sum()),
                "dense_h": int(dense.shape[0]),
                "dense_w": int(dense.shape[1]),
            }
        )
    return realized


def _frame_dims_for(realized: list[dict], horizontal: bool, rng: random.Random) -> tuple[int, int, int] | None:
    n = len(realized)
    gap_choices = [1, 2, 3] if n % 2 == 1 else [1, 3]
    rng.shuffle(gap_choices)
    sidepads = [0, 1, 2]
    rng.shuffle(sidepads)

    for gap in gap_choices:
        if horizontal:
            along_sum = sum(sp["dense_w"] for sp in realized)
            cross_base = max(sp["dense_h"] for sp in realized)
        else:
            along_sum = sum(sp["dense_h"] for sp in realized)
            cross_base = max(sp["dense_w"] for sp in realized)
        for sidepad in sidepads:
            interior_along = along_sum + gap * (n + 1)
            interior_cross = cross_base + 2 * sidepad
            if horizontal:
                if interior_along < interior_cross:
                    continue
            else:
                if interior_cross >= interior_along:
                    continue
            return interior_cross + 2, interior_along + 2, gap
    return None


def _layout_horizontal(
    realized: list[dict],
    frame_h: int,
    frame_w: int,
    rng: random.Random,
) -> tuple[np.ndarray, list[dict]] | None:
    n = len(realized)
    strip_h = max(sp["dense_h"] for sp in realized)
    source_gap = rng.randint(3, 5)
    source_total_w = sum(sp["dense_w"] for sp in realized) + source_gap * (n - 1)
    sep = rng.randint(3, 5)

    min_h = frame_h + strip_h + sep + 2
    min_w = max(frame_w, source_total_w) + 2
    if min_h > 30 or min_w > 30:
        return None

    H = rng.randint(min_h, min(30, min_h + 4))
    W = rng.randint(min_w, min(30, min_w + 4))
    source_on_top = bool(rng.randrange(2))

    frame_r0 = 1 + rng.randint(0, H - (frame_h + strip_h + sep + 2))
    if source_on_top:
        source_r0 = frame_r0 - sep - strip_h
    else:
        source_r0 = frame_r0 + frame_h + sep
    if source_r0 < 1 or source_r0 + strip_h > H - 1:
        return None

    frame_c0 = 1 + rng.randint(0, W - frame_w - 1)
    source_c0 = 1 + rng.randint(0, W - source_total_w - 1)

    area_sorted = sorted(range(n), key=lambda i: realized[i]["dense_area"])
    perm = area_sorted[:]
    while perm == area_sorted:
        rng.shuffle(perm)

    grid = np.zeros((H, W), dtype=int)
    frame_bbox = (frame_r0, frame_c0, frame_r0 + frame_h - 1, frame_c0 + frame_w - 1)
    _stamp_sparse_frame(grid, frame_bbox)

    cursor = source_c0
    placed = []
    colors = rng.sample(SPRITE_POOL, n)
    for color, sprite_idx in zip(colors, perm):
        sp = realized[sprite_idx]
        sr = source_r0 + (strip_h - sp["dense_h"]) // 2
        sc = cursor
        _stamp_sparse_sprite(grid, sp["logical"], (sr, sc), color)
        placed.append(
            {
                "color": color,
                "bbox": (sr, sc, sr + sp["dense_h"] - 1, sc + sp["dense_w"] - 1),
                "area": sp["dense_area"],
                "sparse_count": sp["sparse_count"],
            }
        )
        cursor += sp["dense_w"] + source_gap
    return grid, placed


def _layout_vertical(
    realized: list[dict],
    frame_h: int,
    frame_w: int,
    rng: random.Random,
) -> tuple[np.ndarray, list[dict]] | None:
    n = len(realized)
    strip_w = max(sp["dense_w"] for sp in realized)
    source_gap = rng.randint(3, 5)
    source_total_h = sum(sp["dense_h"] for sp in realized) + source_gap * (n - 1)
    sep = rng.randint(3, 5)

    min_h = max(frame_h, source_total_h) + 2
    min_w = frame_w + strip_w + sep + 2
    if min_h > 30 or min_w > 30:
        return None

    H = rng.randint(min_h, min(30, min_h + 4))
    W = rng.randint(min_w, min(30, min_w + 4))
    source_on_left = bool(rng.randrange(2))

    frame_c0 = 1 + rng.randint(0, W - (frame_w + strip_w + sep + 2))
    if source_on_left:
        source_c0 = frame_c0 - sep - strip_w
    else:
        source_c0 = frame_c0 + frame_w + sep
    if source_c0 < 1 or source_c0 + strip_w > W - 1:
        return None

    frame_r0 = 1 + rng.randint(0, H - frame_h - 1)
    source_r0 = 1 + rng.randint(0, H - source_total_h - 1)

    area_sorted = sorted(range(n), key=lambda i: realized[i]["dense_area"])
    perm = area_sorted[:]
    while perm == area_sorted:
        rng.shuffle(perm)

    grid = np.zeros((H, W), dtype=int)
    frame_bbox = (frame_r0, frame_c0, frame_r0 + frame_h - 1, frame_c0 + frame_w - 1)
    _stamp_sparse_frame(grid, frame_bbox)

    cursor = source_r0
    placed = []
    colors = rng.sample(SPRITE_POOL, n)
    for color, sprite_idx in zip(colors, perm):
        sp = realized[sprite_idx]
        sr = cursor
        sc = source_c0 + (strip_w - sp["dense_w"]) // 2
        _stamp_sparse_sprite(grid, sp["logical"], (sr, sc), color)
        placed.append(
            {
                "color": color,
                "bbox": (sr, sc, sr + sp["dense_h"] - 1, sc + sp["dense_w"] - 1),
                "area": sp["dense_area"],
                "sparse_count": sp["sparse_count"],
            }
        )
        cursor += sp["dense_h"] + source_gap
    return grid, placed


def _semantic_checks(grid: np.ndarray, placed: list[dict]) -> bool:
    try:
        out = main(grid)
        alt = _render_solution(grid, order_mode="source")
    except Exception:
        return False

    if np.array_equal(out, alt):
        return False

    sparse_counts = Counter(sp["sparse_count"] for sp in placed)
    repeated_sparse = [count for count, freq in sparse_counts.items() if freq >= 2]
    if not repeated_sparse:
        return False

    by_sparse: dict[int, set[int]] = {}
    for sp in placed:
        by_sparse.setdefault(int(sp["sparse_count"]), set()).add(int(sp["area"]))
    if not any(len(areas) >= 2 for areas in by_sparse.values()):
        return False

    areas = [int(sp["area"]) for sp in placed]
    if len(set(areas)) != len(areas):
        return False

    bg, frame_bbox, sprites, _frame_color = _detect_objects(grid)
    r0, c0, r1, c1 = frame_bbox
    if bg != BG:
        return False
    if not (0 <= r0 < r1 < grid.shape[0] and 0 <= c0 < c1 < grid.shape[1]):
        return False
    if any(sp["bbox"][2] >= grid.shape[0] or sp["bbox"][3] >= grid.shape[1] for sp in sprites):
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_sprites = rng.choice([3, 4])
    shape_infos = _choose_shapes(rng, n_sprites)
    if shape_infos is None:
        return None
    realized = _realize_shapes(rng, shape_infos)
    if len({sp["dense_area"] for sp in realized}) != n_sprites:
        return None

    horizontal = bool(rng.randrange(2))
    dims = _frame_dims_for(realized, horizontal, rng)
    if dims is None:
        return None
    frame_cross, frame_along, _gap = dims
    if horizontal:
        laid_out = _layout_horizontal(realized, frame_cross, frame_along, rng)
    else:
        laid_out = _layout_vertical(realized, frame_along, frame_cross, rng)
    if laid_out is None:
        return None
    grid, placed = laid_out

    if not _semantic_checks(grid, placed):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sparse-sprite sorting input")
