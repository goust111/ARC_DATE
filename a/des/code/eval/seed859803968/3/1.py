from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
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
SIDES = ("top", "bottom", "left", "right")


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _crop_patch(patch: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    if mask is None:
        mask = patch != BG
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return patch[r0 : r1 + 1, c0 : c1 + 1].copy(), mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    neigh = NEIGHBORS4 if connectivity == 4 else NEIGHBORS8
    out: list[np.ndarray] = []
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
                for dr, dc in neigh:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.asarray(pts, dtype=int))
    return out


def _boundary_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    boundary = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            for dr, dc in NEIGHBORS8:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w) or not mask[nr, nc]:
                    boundary[r, c] = True
                    break
    return boundary


def _interior_mask(mask: np.ndarray) -> np.ndarray:
    return mask & ~_boundary_mask(mask)


def _transform(arr: np.ndarray, transform_id: int) -> np.ndarray:
    if transform_id < 4:
        out = np.rot90(arr, transform_id)
    else:
        out = np.fliplr(np.rot90(arr, transform_id - 4))
    return np.array(out, copy=True)


def _apply_transform(patch: np.ndarray, mask: np.ndarray, transform_id: int) -> tuple[np.ndarray, np.ndarray]:
    return _crop_patch(_transform(patch, transform_id), _transform(mask, transform_id))


def _mask_key(mask: np.ndarray) -> tuple[str, ...]:
    return tuple("".join("1" if cell else "0" for cell in row) for row in mask.astype(bool))


def _all_mask_keys(mask: np.ndarray) -> set[tuple[str, ...]]:
    keys = set()
    for transform_id in range(8):
        keys.add(_mask_key(_crop_mask(_transform(mask, transform_id))))
    return keys


def _canonical_signature(mask: np.ndarray) -> tuple[str, ...]:
    return min(_all_mask_keys(mask))


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(counts.argmax())])


def _extract_objects(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    objects: list[dict[str, object]] = []
    for coords in _components(grid != bg, connectivity=4):
        r0, c0, r1, c1 = _bbox(coords)
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
        rel = coords - np.array([r0, c0], dtype=int)
        patch[rel[:, 0], rel[:, 1]] = grid[coords[:, 0], coords[:, 1]]
        mask = patch != bg
        boundary = _boundary_mask(mask)
        colors = sorted(int(v) for v in np.unique(patch[mask]))
        objects.append(
            {
                "bbox": (r0, c0, r1, c1),
                "patch": patch,
                "mask": mask,
                "boundary": boundary,
                "colors": colors,
                "signature": _canonical_signature(boundary),
            }
        )
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each grey filled object to the legend block with the same boundary shape
    up to rotation/reflection, then stamp the transformed legend pattern into it.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    out = np.full_like(g, bg)

    objects = _extract_objects(g, bg)
    grey_objects: list[dict[str, object]] = []
    legends_by_signature: dict[tuple[str, ...], list[dict[str, object]]] = {}

    for obj in objects:
        colors = obj["colors"]
        patch = obj["patch"]
        boundary = obj["boundary"]
        if colors == [GREY]:
            grey_objects.append(obj)
            continue
        boundary_colors = {int(v) for v in patch[boundary]}
        if len(colors) >= 2 and len(boundary_colors) == 1:
            legends_by_signature.setdefault(obj["signature"], []).append(obj)

    used_ids: set[int] = set()
    for grey_obj in grey_objects:
        candidates = legends_by_signature.get(grey_obj["signature"], [])
        match_patch = None
        match_index = None
        grey_mask = grey_obj["mask"]
        grey_boundary = grey_obj["boundary"]
        for legend in candidates:
            marker = id(legend)
            if marker in used_ids:
                continue
            legend_patch = legend["patch"]
            legend_mask = legend["mask"]
            for transform_id in range(8):
                transformed_patch, transformed_mask = _apply_transform(legend_patch, legend_mask, transform_id)
                if transformed_mask.shape != grey_mask.shape:
                    continue
                if not np.array_equal(_boundary_mask(transformed_mask), grey_boundary):
                    continue
                if not np.array_equal(transformed_mask, grey_mask):
                    continue
                match_patch = transformed_patch
                match_index = marker
                break
            if match_patch is not None:
                break
        assert match_patch is not None and match_index is not None, "every grey shape must match one legend"
        used_ids.add(match_index)
        r0, c0, r1, c1 = grey_obj["bbox"]
        sub = out[r0 : r1 + 1, c0 : c1 + 1]
        sub[grey_mask] = match_patch[grey_mask]

    return out


def _paint(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    mask = patch != BG
    sub = grid[top : top + patch.shape[0], left : left + patch.shape[1]]
    sub[mask] = patch[mask]


def _is_connected(mask: np.ndarray) -> bool:
    return len(_components(mask, connectivity=4)) == 1


def _interior_count(mask: np.ndarray) -> int:
    return int(_interior_mask(mask).sum())


def _carve_notch(mask: np.ndarray, side: str, start: int, span: int, depth: int) -> np.ndarray:
    trial = mask.copy()
    if side == "top":
        trial[:depth, start : start + span] = False
    elif side == "bottom":
        trial[-depth:, start : start + span] = False
    elif side == "left":
        trial[start : start + span, :depth] = False
    else:
        trial[start : start + span, -depth:] = False
    if not bool(trial.any()):
        return mask
    return _crop_mask(trial)


def _random_shape(rng: random.Random, used_signatures: set[tuple[str, ...]]) -> np.ndarray:
    for _ in range(500):
        height = rng.randint(4, 6)
        width = rng.randint(4, 6)
        mask = np.ones((height, width), dtype=bool)
        target_notches = rng.randint(1, 4)
        carved = 0
        for _ in range(target_notches * 4):
            if carved >= target_notches:
                break
            side = rng.choice(SIDES)
            dim_major = mask.shape[1] if side in ("top", "bottom") else mask.shape[0]
            dim_minor = mask.shape[0] if side in ("top", "bottom") else mask.shape[1]
            if dim_minor <= 3 or dim_major <= 1:
                continue
            max_depth = min(2, dim_minor - 3)
            if max_depth < 1:
                continue
            depth = rng.randint(1, max_depth)
            span = rng.randint(1, dim_major - 1)
            start = rng.randint(0, dim_major - span)
            trial = _carve_notch(mask, side, start, span, depth)
            if not _is_connected(trial):
                continue
            if _interior_count(trial) < 2:
                continue
            mask = trial
            carved += 1
        boundary = _boundary_mask(mask)
        signature = _canonical_signature(boundary)
        if carved == 0:
            continue
        if _interior_count(mask) < 2:
            continue
        if len(_all_mask_keys(boundary)) != 8:
            continue
        if signature in used_signatures:
            continue
        return mask
    raise RuntimeError("failed to generate a distinct asymmetric shape")


def _pattern_sort_key(mode: str):
    if mode == "row_major":
        return lambda rc: (rc[0], rc[1])
    if mode == "col_major":
        return lambda rc: (rc[1], rc[0])
    if mode == "diag":
        return lambda rc: (rc[0] + rc[1], rc[0], rc[1])
    return lambda rc: (rc[0] - rc[1], rc[0], rc[1])


def _make_legend_patch(mask: np.ndarray, border_color: int, rng: random.Random) -> np.ndarray:
    boundary = _boundary_mask(mask)
    interior = np.argwhere(mask & ~boundary)
    assert len(interior) >= 2
    patch = np.full(mask.shape, BG, dtype=int)
    patch[boundary] = border_color
    pattern_color_count = 2 if len(interior) == 2 else rng.choice((2, 3))
    pattern_colors = rng.sample([c for c in PALETTE if c != border_color], pattern_color_count)
    mode = rng.choice(("row_major", "col_major", "diag", "anti_diag"))
    ordered = sorted(((int(r), int(c)) for r, c in interior.tolist()), key=_pattern_sort_key(mode))
    for idx, (r, c) in enumerate(ordered):
        patch[r, c] = pattern_colors[idx % len(pattern_colors)]
    return patch


def _bbox_conflicts(box: tuple[int, int, int, int], others: list[tuple[int, int, int, int]], gap: int = 1) -> bool:
    r0, c0, r1, c1 = box
    for or0, oc0, or1, oc1 in others:
        if not (r1 + gap < or0 or or1 + gap < r0 or c1 + gap < oc0 or oc1 + gap < c0):
            return True
    return False


def _place_edge_box(
    grid_h: int,
    grid_w: int,
    box_h: int,
    box_w: int,
    reserved: list[tuple[int, int, int, int]],
    rng: random.Random,
) -> tuple[int, int]:
    for _ in range(200):
        side = rng.choice(SIDES)
        if side == "top":
            r0, c0 = 0, rng.randint(0, grid_w - box_w)
        elif side == "bottom":
            r0, c0 = grid_h - box_h, rng.randint(0, grid_w - box_w)
        elif side == "left":
            r0, c0 = rng.randint(0, grid_h - box_h), 0
        else:
            r0, c0 = rng.randint(0, grid_h - box_h), grid_w - box_w
        box = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
        if not _bbox_conflicts(box, reserved, gap=1):
            reserved.append(box)
            return r0, c0
    raise RuntimeError("failed to place legend on an edge")


def _place_inner_box(
    grid_h: int,
    grid_w: int,
    box_h: int,
    box_w: int,
    reserved: list[tuple[int, int, int, int]],
    rng: random.Random,
) -> tuple[int, int]:
    for _ in range(300):
        r0 = rng.randint(2, grid_h - box_h - 3)
        c0 = rng.randint(2, grid_w - box_w - 3)
        box = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
        if not _bbox_conflicts(box, reserved, gap=1):
            reserved.append(box)
            return r0, c0
    raise RuntimeError("failed to place grey object in the interior")


def _generate_case(rng: random.Random | None = None) -> tuple[np.ndarray, np.ndarray]:
    rng = rng or random.Random()
    for _ in range(300):
        grid_h = rng.randint(22, 30)
        grid_w = rng.randint(22, 30)
        pair_count = rng.randint(2, 4)

        used_signatures: set[tuple[str, ...]] = set()
        shapes: list[np.ndarray] = []
        for _ in range(pair_count):
            shape = _random_shape(rng, used_signatures)
            used_signatures.add(_canonical_signature(_boundary_mask(shape)))
            shapes.append(shape)

        border_colors = rng.sample(list(PALETTE), pair_count)
        reserved: list[tuple[int, int, int, int]] = []
        placements: list[dict[str, object]] = []
        try:
            for shape, border_color in zip(shapes, border_colors, strict=False):
                canonical_patch = _make_legend_patch(shape, border_color, rng)
                legend_transform = rng.randrange(8)
                grey_transform = rng.randrange(8)
                legend_patch, _legend_mask = _apply_transform(canonical_patch, shape, legend_transform)
                grey_patch, _grey_mask = _apply_transform(
                    np.where(shape, GREY, BG).astype(int),
                    shape,
                    grey_transform,
                )
                expected_patch, _ = _apply_transform(canonical_patch, shape, grey_transform)
                legend_top, legend_left = _place_edge_box(
                    grid_h,
                    grid_w,
                    legend_patch.shape[0],
                    legend_patch.shape[1],
                    reserved,
                    rng,
                )
                grey_top, grey_left = _place_inner_box(
                    grid_h,
                    grid_w,
                    grey_patch.shape[0],
                    grey_patch.shape[1],
                    reserved,
                    rng,
                )
                placements.append(
                    {
                        "legend_patch": legend_patch,
                        "legend_pos": (legend_top, legend_left),
                        "grey_patch": grey_patch,
                        "grey_pos": (grey_top, grey_left),
                        "expected_patch": expected_patch,
                    }
                )
        except RuntimeError:
            continue

        inp = np.full((grid_h, grid_w), BG, dtype=int)
        out = np.full((grid_h, grid_w), BG, dtype=int)
        for item in placements:
            legend_top, legend_left = item["legend_pos"]
            grey_top, grey_left = item["grey_pos"]
            _paint(inp, item["legend_patch"], legend_top, legend_left)
            _paint(inp, item["grey_patch"], grey_top, grey_left)
            _paint(out, item["expected_patch"], grey_top, grey_left)

        if np.array_equal(inp, out):
            continue
        counts = Counter(inp.ravel())
        if counts[BG] <= counts[GREY]:
            continue
        return inp, out
    raise RuntimeError("failed to generate a valid case")


def generate_input() -> np.ndarray:
    return _generate_case()[0]
