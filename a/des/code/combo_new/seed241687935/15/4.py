from __future__ import annotations

from collections import deque
import numpy as np


_RNG = np.random.default_rng()
_BG = 0
_FRAME = 5
_KEY_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
_SIDES = ("top", "bottom", "left", "right")
_NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_NEIGHBORS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return np.zeros((0, 0), dtype=bool)
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    trimmed = _trim_mask(mask)
    return tuple(tuple(int(x) for x in row) for row in trimmed.astype(int))


def _canonical_shape_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    keys = []
    cur = _trim_mask(mask)
    for _ in range(4):
        keys.append(_mask_key(cur))
        cur = _trim_mask(np.rot90(cur))
    return min(keys)


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in _NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _color_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in _NEIGHBORS8:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            r0, c0, r1, c1 = _bbox_from_coords(coords)
            local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                local[rr - r0, cc - c0] = True
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "mask": local,
                    "area": len(coords),
                }
            )
    return comps


def _bg_components_in_bbox(mask: np.ndarray) -> list[dict]:
    bg = ~mask
    h, w = bg.shape
    comps = []
    for coords in _connected_components(bg):
        local = np.zeros((h, w), dtype=bool)
        touches = False
        for r, c in coords:
            local[r, c] = True
            if r == 0 or r == h - 1 or c == 0 or c == w - 1:
                touches = True
        comps.append({"coords": coords, "mask": local, "touches_bbox": touches, "area": len(coords)})
    return comps


def _analyze_frame(comp: dict) -> dict | None:
    mask = comp["mask"]
    h, w = mask.shape
    if h < 6 or w < 6:
        return None
    bg_comps = _bg_components_in_bbox(mask)
    edge_comps = [x for x in bg_comps if x["touches_bbox"]]
    inner_comps = [x for x in bg_comps if not x["touches_bbox"]]
    if len(edge_comps) != 1 or not inner_comps:
        return None
    notch = edge_comps[0]
    if not (3 <= int(notch["area"]) <= 5):
        return None
    shape_key = _canonical_shape_key(notch["mask"])
    r0, c0, _r1, _c1 = comp["bbox"]
    notch_coords = [(r0 + rr, c0 + cc) for rr, cc in notch["coords"]]
    return {
        "color": int(comp["color"]),
        "bbox": comp["bbox"],
        "notch_coords": notch_coords,
        "notch_mask": notch["mask"],
        "shape_key": shape_key,
    }


def _extract_scene(grid: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    bg = _most_common_color(grid)
    comps = _color_components(grid, bg)
    frames: list[dict] = []
    keys: list[dict] = []
    for comp in comps:
        frame = _analyze_frame(comp)
        if frame is not None:
            frames.append(frame)
            continue
        if 3 <= int(comp["area"]) <= 5:
            keys.append(
                {
                    "color": int(comp["color"]),
                    "coords": comp["coords"],
                    "shape_key": _canonical_shape_key(comp["mask"]),
                }
            )
    return bg, frames, keys


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match colored keys to complementary frame notches, then insert each color into its lockhole.

    Input structure:
    - The background is black.
    - Several small monochrome 4-connected polyominoes of area 3-5 act as keys, with distinct colors.
    - Several grey one-pixel-wide frame components look like rectangles with one side indented by a key-shaped notch.
    - Each frame notch is visible only as a black side-connected cavity inside the frame bbox.

    Rule:
    - Find every colored key and compute its shape up to 0°, 90°, 180°, and 270° rotation.
    - For every grey frame, recover the notch as the unique black component inside its bbox that touches the bbox boundary.
    - Match keys to notches by rotation-invariant shape, color each notch with the matched key color, and erase the original key cells to black.
    - Leave the grey frames and all unrelated pixels unchanged.

    Key insight:
    - The lockhole is not drawn directly; it is the side-connected negative-space cavity carved from a grey frame, so solving requires extracting that cavity and matching it to a rotated key shape.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bg, frames, keys = _extract_scene(g)
    assert frames and len(frames) == len(keys), "scene must contain equally many frames and keys"

    frame_by_shape: dict[tuple[tuple[int, ...], ...], dict] = {}
    for frame in frames:
        key = frame["shape_key"]
        assert key not in frame_by_shape, "frame shapes must be unique"
        frame_by_shape[key] = frame

    seen_shapes = set()
    for key in keys:
        shape_key = key["shape_key"]
        assert shape_key in frame_by_shape, "every key must match a frame notch"
        assert shape_key not in seen_shapes, "key shapes must be unique"
        seen_shapes.add(shape_key)
        for r, c in key["coords"]:
            out[int(r), int(c)] = bg
        for r, c in frame_by_shape[shape_key]["notch_coords"]:
            out[int(r), int(c)] = int(key["color"])

    return out


def _normalize_coords(coords: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    shifted = sorted((r - min_r, c - min_c) for r, c in coords)
    return tuple(shifted)


def _shape_coords_to_mask(coords: tuple[tuple[int, int], ...]) -> np.ndarray:
    h = 1 + max(r for r, _ in coords)
    w = 1 + max(c for _, c in coords)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    return mask


def _all_polyominoes_upto_rotations() -> list[np.ndarray]:
    shapes_by_size: dict[int, set[tuple[tuple[int, int], ...]]] = {1: {((0, 0),)}}
    for size in range(2, 6):
        out: set[tuple[tuple[int, int], ...]] = set()
        for shape in shapes_by_size[size - 1]:
            base = set(shape)
            for r, c in list(base):
                for dr, dc in _NEIGHBORS4:
                    cand = (r + dr, c + dc)
                    if cand in base:
                        continue
                    grown = set(base)
                    grown.add(cand)
                    out.add(_normalize_coords(grown))
        shapes_by_size[size] = out

    catalog: dict[tuple[tuple[int, ...], ...], np.ndarray] = {}
    for size in (3, 4, 5):
        for shape in shapes_by_size[size]:
            mask = _shape_coords_to_mask(shape)
            catalog.setdefault(_canonical_shape_key(mask), mask)
    return list(catalog.values())


_SHAPE_CATALOG = _all_polyominoes_upto_rotations()


def _rotate_trim(mask: np.ndarray, k: int) -> np.ndarray:
    return _trim_mask(np.rot90(mask, k))


def _boundary_of_solid(solid: np.ndarray) -> np.ndarray:
    h, w = solid.shape
    boundary = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not solid[r, c]:
                continue
            for dr, dc in _NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w) or not solid[nr, nc]:
                    boundary[r, c] = True
                    break
    return boundary


def _make_lock_patch(shape_mask: np.ndarray, side: str) -> np.ndarray:
    sh, sw = shape_mask.shape
    if side in ("top", "bottom"):
        box_h = max(6, sh + int(_RNG.integers(3, 6)))
        before = int(_RNG.integers(1, 4))
        after = int(_RNG.integers(1, 4))
        box_w = max(6, sw + before + after)
        row0 = 0 if side == "top" else box_h - sh
        col0 = before
    else:
        box_w = max(6, sw + int(_RNG.integers(3, 6)))
        before = int(_RNG.integers(1, 4))
        after = int(_RNG.integers(1, 4))
        box_h = max(6, sh + before + after)
        row0 = before
        col0 = 0 if side == "left" else box_w - sw

    notch = np.zeros((box_h, box_w), dtype=bool)
    notch[row0 : row0 + sh, col0 : col0 + sw] = shape_mask
    solid = np.ones((box_h, box_w), dtype=bool)
    solid[notch] = False
    frame = _boundary_of_solid(solid)
    patch = np.full((box_h, box_w), _BG, dtype=int)
    patch[frame] = _FRAME
    return patch


def _make_key_patch(shape_mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(shape_mask.shape, _BG, dtype=int)
    patch[shape_mask] = int(color)
    return patch


def _paint_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    mask = patch != _BG
    h, w = patch.shape
    window = grid[top : top + h, left : left + w]
    window[mask] = patch[mask]


def _try_place_items(items: list[dict], height: int, width: int) -> np.ndarray | None:
    grid = np.full((height, width), _BG, dtype=int)
    reserved = np.zeros((height, width), dtype=bool)
    for item in sorted(items, key=lambda x: -(x["h"] * x["w"])):
        h = int(item["h"])
        w = int(item["w"])
        positions = [(r, c) for r in range(0, height - h + 1) for c in range(0, width - w + 1)]
        _RNG.shuffle(positions)
        placed = False
        for r0, c0 in positions:
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(height, r0 + h + 1)
            cc1 = min(width, c0 + w + 1)
            if reserved[rr0:rr1, cc0:cc1].any():
                continue
            _paint_patch(grid, item["patch"], r0, c0)
            reserved[rr0:rr1, cc0:cc1] = True
            placed = True
            break
        if not placed:
            return None
    return grid


def _build_items() -> list[dict]:
    pair_count = int(_RNG.integers(2, 5))
    colors = list(_RNG.choice(_KEY_COLORS, size=pair_count, replace=False))
    shape_ids = _RNG.choice(len(_SHAPE_CATALOG), size=pair_count, replace=False)
    sides = list(_SIDES)
    _RNG.shuffle(sides)

    items: list[dict] = []
    for idx in range(pair_count):
        base_shape = _SHAPE_CATALOG[int(shape_ids[idx])]
        key_shape = _rotate_trim(base_shape, int(_RNG.integers(0, 4)))
        lock_shape = _rotate_trim(base_shape, int(_RNG.integers(0, 4)))
        lock_patch = _make_lock_patch(lock_shape, sides[idx])
        key_patch = _make_key_patch(key_shape, int(colors[idx]))
        items.append({"patch": lock_patch, "h": lock_patch.shape[0], "w": lock_patch.shape[1]})
        items.append({"patch": key_patch, "h": key_patch.shape[0], "w": key_patch.shape[1]})
    return items


def generate_input() -> np.ndarray:
    for _ in range(400):
        items = _build_items()
        pair_count = len(items) // 2
        lo = min(30, 16 + 3 * pair_count)
        height = int(_RNG.integers(lo, 31))
        width = int(_RNG.integers(lo, 31))
        grid = _try_place_items(items, height, width)
        if grid is None:
            continue
        bg, frames, keys = _extract_scene(grid)
        if bg != _BG:
            continue
        if len(frames) != pair_count or len(keys) != pair_count:
            continue
        frame_shapes = sorted(frame["shape_key"] for frame in frames)
        key_shapes = sorted(key["shape_key"] for key in keys)
        if frame_shapes != key_shapes or len(set(frame_shapes)) != pair_count:
            continue
        return grid
    raise RuntimeError("failed to generate a valid key-lock sample")
