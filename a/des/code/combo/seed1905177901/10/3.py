from __future__ import annotations

from collections import deque

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split each framed two-color object into a stationary border and a movable fill.

    Input structure
    - Background is black.
    - Several objects are 1px rectangular frames with one fill color somewhere inside.
    - One empty rectangular frame on the right is the stacking zone.

    Rule
    - Keep every colored frame in place.
    - Remove each object's fill from its original location.
    - Sort the extracted fill shapes by area, descending.
    - Stack them inside the right-hand empty frame from bottom to top, left aligned,
      leaving one empty row between consecutive shapes.
    - Clip anything that exceeds the stacking zone.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    scene = _parse_scene(g)
    out = g.copy()
    bg = int(scene["bg"])

    for obj in scene["objects"]:
        r0, c0, r1, c1 = obj["bbox"]
        inner = out[r0 + 1 : r1, c0 + 1 : c1]
        inner[obj["fill_mask_full"]] = bg

    zone_r0, zone_c0, zone_r1, zone_c1 = scene["zone"]["bbox"]
    zone = out[zone_r0 + 1 : zone_r1, zone_c0 + 1 : zone_c1]
    zone[:, :] = bg

    zone_h, zone_w = zone.shape
    cursor_bottom = zone_h - 1
    pieces = sorted(
        scene["objects"],
        key=lambda item: (-int(item["area"]), int(item["bbox"][0]), int(item["bbox"][1])),
    )

    for piece in pieces:
        mask = piece["shape_mask"]
        color = int(piece["fill_color"])
        ph, pw = map(int, mask.shape)
        top = int(cursor_bottom - ph + 1)
        left = 0

        src_r0 = max(0, -top)
        src_c0 = max(0, -left)
        src_r1 = min(ph, zone_h - top)
        src_c1 = min(pw, zone_w - left)
        if src_r0 < src_r1 and src_c0 < src_c1:
            dst_r0 = max(0, top)
            dst_c0 = max(0, left)
            dst_r1 = dst_r0 + (src_r1 - src_r0)
            dst_c1 = dst_c0 + (src_c1 - src_c0)
            submask = mask[src_r0:src_r1, src_c0:src_c1]
            zone_view = zone[dst_r0:dst_r1, dst_c0:dst_c1]
            zone_view[submask] = color

        cursor_bottom = int(top - 2)

    return out


# --------------- helpers ---------------

_RNG = np.random.default_rng()
_BG = 0
_STACK_BORDER = 5
_MAX_RETRIES = 400
_DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(int(r), int(c))])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((int(rr), int(cc)))
                for dr, dc in _DIRS4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))

    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    return _bbox(coords)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _mask_bbox(mask)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _is_rect_frame(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return False
    expected = 2 * (h + w) - 4
    if int(coords.shape[0]) != expected:
        return False

    mask = np.zeros((h, w), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    expected_mask = np.zeros((h, w), dtype=bool)
    expected_mask[0, :] = True
    expected_mask[-1, :] = True
    expected_mask[:, 0] = True
    expected_mask[:, -1] = True
    return bool(np.array_equal(mask, expected_mask))


def _background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)

    zone_candidates: list[dict] = []
    objects: list[dict] = []

    for color in sorted(set(map(int, g.reshape(-1))) - {int(bg)}):
        for comp in _components(g == int(color)):
            if not _is_rect_frame(comp):
                continue

            r0, c0, r1, c1 = _bbox(comp)
            inner = g[r0 + 1 : r1, c0 + 1 : c1]
            inner_colors = sorted(set(map(int, inner.reshape(-1))) - {int(bg)})

            if not inner_colors:
                zone_candidates.append(
                    {
                        "bbox": (int(r0), int(c0), int(r1), int(c1)),
                        "border_color": int(color),
                    }
                )
                continue

            if len(inner_colors) != 1:
                continue

            fill_color = int(inner_colors[0])
            fill_mask_full = inner == int(fill_color)
            if not bool(fill_mask_full.any()):
                continue
            shape_mask = _crop_mask(fill_mask_full)
            objects.append(
                {
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "border_color": int(color),
                    "fill_color": int(fill_color),
                    "fill_mask_full": fill_mask_full.copy(),
                    "shape_mask": shape_mask,
                    "area": int(fill_mask_full.sum()),
                }
            )

    assert zone_candidates, "expected an empty rectangular stacking zone"
    zone = max(zone_candidates, key=lambda item: (int(item["bbox"][3]), int(item["bbox"][1])))
    assert objects, "expected at least one framed object with interior fill"
    return {"bg": int(bg), "zone": zone, "objects": objects}


def _weighted_choice(weights: np.ndarray) -> int:
    probs = weights.astype(float)
    probs /= probs.sum()
    return int(_RNG.choice(np.arange(len(probs)), p=probs))


def _random_shape(area: int) -> np.ndarray:
    assert area >= 2
    for _ in range(200):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in _DIRS4:
                    nxt = (int(r + dr), int(c + dc))
                    if nxt not in cells:
                        frontier.append(nxt)
            frontier = sorted(set(frontier))
            weights = []
            for r, c in frontier:
                adj = sum((int(r + dr), int(c + dc)) in cells for dr, dc in _DIRS4)
                weights.append(1 + 3 * adj)
            chosen = frontier[_weighted_choice(np.asarray(weights, dtype=float))]
            cells.add(chosen)

        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        rmin, cmin = min(rows), min(cols)
        norm = {(r - rmin, c - cmin) for r, c in cells}
        h = 1 + max(r for r, _ in norm)
        w = 1 + max(c for _, c in norm)
        if max(h, w) > 7:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in norm:
            mask[r, c] = True

        if bool(_RNG.integers(0, 2)):
            mask = np.flipud(mask)
        if bool(_RNG.integers(0, 2)):
            mask = np.fliplr(mask)
        if bool(_RNG.integers(0, 2)):
            mask = mask.T
        return _crop_mask(mask)

    raise RuntimeError("failed to generate a compact fill shape")


def _build_object(shape_mask: np.ndarray, border_color: int, fill_color: int) -> np.ndarray:
    sh, sw = map(int, shape_mask.shape)
    top_pad = int(_RNG.integers(0, 3))
    bottom_pad = int(_RNG.integers(0, 3))
    left_pad = int(_RNG.integers(0, 3))
    right_pad = int(_RNG.integers(0, 3))
    inner_h = int(sh + top_pad + bottom_pad)
    inner_w = int(sw + left_pad + right_pad)

    obj = np.full((inner_h + 2, inner_w + 2), _BG, dtype=int)
    obj[0, :] = border_color
    obj[-1, :] = border_color
    obj[:, 0] = border_color
    obj[:, -1] = border_color

    patch = obj[1 : 1 + inner_h, 1 : 1 + inner_w]
    view = patch[top_pad : top_pad + sh, left_pad : left_pad + sw]
    view[shape_mask] = fill_color
    return obj


def _choose_zone_size(shape_masks: list[np.ndarray]) -> tuple[int, int]:
    heights = [int(mask.shape[0]) for mask in shape_masks]
    widths = [int(mask.shape[1]) for mask in shape_masks]
    total_h = int(sum(heights) + max(0, len(heights) - 1))
    max_h = int(max(heights))
    max_w = int(max(widths))

    modes = ["fit", "vclip"]
    if max_w >= 3:
        modes.append("wclip")
        modes.append("mixed")
    mode = str(_RNG.choice(modes))

    if mode == "fit":
        zone_h = int(_RNG.integers(total_h, total_h + 5))
        zone_w = int(_RNG.integers(max_w, max_w + 4))
    elif mode == "vclip":
        upper = max(max_h + 1, total_h)
        zone_h = int(_RNG.integers(max(2, max_h // 2 + 1), upper))
        zone_w = int(_RNG.integers(max_w, max_w + 4))
    elif mode == "wclip":
        zone_h = int(_RNG.integers(total_h, total_h + 5))
        zone_w = int(_RNG.integers(max(2, max_w // 2), max_w))
    else:
        upper = max(max_h + 1, total_h)
        zone_h = int(_RNG.integers(max(2, max_h // 2 + 1), upper))
        zone_w = int(_RNG.integers(max(2, max_w // 2), max_w))

    return max(1, zone_h), max(1, zone_w)


def _can_place(occ: np.ndarray, top: int, left: int, h: int, w: int) -> bool:
    hh, ww = map(int, occ.shape)
    if top < 0 or left < 0 or top + h > hh or left + w > ww:
        return False
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(hh, top + h + 1)
    c1 = min(ww, left + w + 1)
    return not bool(occ[r0:r1, c0:c1].any())


def _place_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = map(int, patch.shape)
    sub = grid[top : top + h, left : left + w]
    mask = patch != _BG
    sub[mask] = patch[mask]


def _gen_attempt() -> np.ndarray | None:
    n_objects = int(_RNG.integers(2, 5))
    palette = [1, 2, 3, 4, 6, 7, 8, 9]
    fill_colors = list(map(int, _RNG.choice(palette, size=n_objects, replace=False)))
    border_candidates = [color for color in palette if color not in fill_colors]
    pool_size = int(_RNG.integers(1, min(3, len(border_candidates)) + 1))
    border_pool = list(map(int, _RNG.choice(border_candidates, size=pool_size, replace=False)))

    area_pool = np.arange(2, 15)
    areas = list(map(int, _RNG.choice(area_pool, size=n_objects, replace=False)))
    _RNG.shuffle(areas)
    shape_masks = [_random_shape(area) for area in areas]
    objects = [
        _build_object(shape_masks[i], int(_RNG.choice(border_pool)), fill_colors[i]) for i in range(n_objects)
    ]

    zone_h, zone_w = _choose_zone_size(shape_masks)
    zone = np.full((zone_h + 2, zone_w + 2), _BG, dtype=int)
    zone[0, :] = _STACK_BORDER
    zone[-1, :] = _STACK_BORDER
    zone[:, 0] = _STACK_BORDER
    zone[:, -1] = _STACK_BORDER

    max_obj_h = max(int(obj.shape[0]) for obj in objects)
    max_obj_w = max(int(obj.shape[1]) for obj in objects)
    grid_h = max(18, zone.shape[0] + int(_RNG.integers(2, 8)), max_obj_h + 6)
    grid_w = max(int(zone.shape[1]) + max_obj_w + 6, 24) + int(_RNG.integers(0, 5))
    if grid_h > 30 or grid_w > 34:
        return None

    grid = np.full((grid_h, grid_w), _BG, dtype=int)
    occ = np.zeros((grid_h, grid_w), dtype=bool)

    zone_top_max = grid_h - int(zone.shape[0]) - 1
    if zone_top_max < 1:
        return None
    zone_top = int(_RNG.integers(1, zone_top_max + 1))
    zone_left = int(grid_w - zone.shape[1] - _RNG.integers(1, 3))
    if zone_left <= 8:
        return None
    _place_patch(grid, zone, zone_top, zone_left)
    occ[zone_top : zone_top + zone.shape[0], zone_left : zone_left + zone.shape[1]] = True

    order = list(range(n_objects))
    _RNG.shuffle(order)
    left_limit = zone_left - 2
    if left_limit <= 2:
        return None

    for idx in order:
        obj = objects[idx]
        h, w = map(int, obj.shape)
        if w >= left_limit:
            return None
        placed = False
        for _ in range(120):
            top = int(_RNG.integers(1, grid_h - h))
            left = int(_RNG.integers(1, left_limit - w + 1))
            if _can_place(occ, top, left, h, w):
                _place_patch(grid, obj, top, left)
                occ[top : top + h, left : left + w] = True
                placed = True
                break
        if not placed:
            return None

    scene = _parse_scene(grid)
    parsed_areas = sorted(int(obj["area"]) for obj in scene["objects"])
    if len(scene["objects"]) != n_objects or len(set(parsed_areas)) != n_objects:
        return None
    if scene["zone"]["bbox"][1] <= max(obj["bbox"][3] for obj in scene["objects"]):
        return None

    return grid


# --------------- generator ---------------


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
