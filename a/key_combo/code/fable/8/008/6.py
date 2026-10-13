from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

BG = 0
MODES = ("vertical", "horizontal", "rot180")
GLYPH_MASKS = {
    "vertical": np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [1, 0, 1],
        ],
        dtype=bool,
    ),
    "horizontal": np.array(
        [
            [0, 1, 1],
            [1, 1, 0],
            [0, 1, 1],
        ],
        dtype=bool,
    ),
    "rot180": np.array(
        [
            [1, 1, 0],
            [0, 1, 0],
            [0, 1, 1],
        ],
        dtype=bool,
    ),
}


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: BBox) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _classify_glyph(mask: np.ndarray) -> str:
    vertical = bool(np.array_equal(mask, np.fliplr(mask)))
    horizontal = bool(np.array_equal(mask, np.flipud(mask)))
    rot180 = bool(np.array_equal(mask, np.rot90(mask, 2)))

    if vertical and not horizontal and not rot180:
        return "vertical"
    if horizontal and not vertical and not rot180:
        return "horizontal"
    if rot180 and not vertical and not horizontal:
        return "rot180"
    raise AssertionError("glyph must have exactly one target symmetry")


def _transform_coords(coords: np.ndarray, mode: str, anchor_r: int, anchor_c: int) -> np.ndarray:
    out = coords.copy()
    if mode == "vertical":
        out[:, 1] = 2 * int(anchor_c) - coords[:, 1]
    elif mode == "horizontal":
        out[:, 0] = 2 * int(anchor_r) - coords[:, 0]
    elif mode == "rot180":
        out[:, 0] = 2 * int(anchor_r) - coords[:, 0]
        out[:, 1] = 2 * int(anchor_c) - coords[:, 1]
    else:
        raise ValueError(f"unknown mode: {mode}")
    return out


def _render_transformed(
    grid: np.ndarray,
    bg: int,
    glyph_coords: np.ndarray,
    obj_coords: np.ndarray,
    obj_color: int,
    mode: str,
    anchor_r: int,
    anchor_c: int,
) -> np.ndarray | None:
    h, w = grid.shape
    target = _transform_coords(obj_coords, mode, anchor_r, anchor_c)
    if np.any(target[:, 0] < 0) or np.any(target[:, 0] >= h) or np.any(target[:, 1] < 0) or np.any(target[:, 1] >= w):
        return None

    glyph_set = {tuple(map(int, rc)) for rc in glyph_coords}
    if any((int(r), int(c)) in glyph_set for r, c in target):
        return None

    out = grid.copy()
    out[obj_coords[:, 0], obj_coords[:, 1]] = int(bg)
    out[target[:, 0], target[:, 1]] = int(obj_color)
    return out


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)

    components = _components_mask(g != bg)
    assert len(components) == 2, "expected one glyph and one large object"

    infos = []
    for coords in components:
        bbox = _bbox_coords(coords)
        patch = _crop_bbox(g, bbox)
        colors = {int(g[r, c]) for r, c in coords}
        assert len(colors) == 1, "each component must be monochrome"
        infos.append(
            {
                "coords": coords,
                "bbox": bbox,
                "patch": patch,
                "color": int(next(iter(colors))),
            }
        )

    glyph_candidates = [
        info
        for info in infos
        if info["patch"].shape == (3, 3) and np.count_nonzero(info["patch"] != bg) >= 4
    ]
    assert len(glyph_candidates) == 1, "expected exactly one 3x3 marker glyph"
    glyph = glyph_candidates[0]
    obj = next(info for info in infos if info is not glyph)

    glyph_mask = glyph["patch"] != bg
    mode = _classify_glyph(glyph_mask)
    anchor_r = int(glyph["bbox"][0] + 1)
    anchor_c = int(glyph["bbox"][1] + 1)

    return {
        "grid": g,
        "bg": int(bg),
        "glyph": glyph,
        "object": obj,
        "mode": mode,
        "anchor_r": anchor_r,
        "anchor_c": anchor_c,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the large object by reflecting or rotating it according to the marker glyph's unique symmetry.

    Input structure:
    - The grid has a dominant background, one larger asymmetric monochrome foreground object, and one separate 3x3 monochrome marker glyph.
    - The glyph has exactly one symmetry: only left-right mirror symmetry, only up-down mirror symmetry, or only 180-degree rotational symmetry.
    - The glyph stays fixed and its center row, center column, or center cell provides the geometric anchor for the move.

    Rule:
    - Identify the 3x3 glyph and classify its unique symmetry type.
    - If the glyph is only left-right symmetric, reflect the large object across the full vertical line through the glyph's center column.
    - If the glyph is only up-down symmetric, reflect the large object across the full horizontal line through the glyph's center row.
    - If the glyph is only 180-degree symmetric, rotate the large object 180 degrees around the glyph's center cell.
    - Erase the object's source cells, keep the glyph unchanged, and draw the transformed object at its new location.

    Key insight:
    - The glyph encodes both the operation and the anchor: the symmetry that leaves the glyph unchanged is exactly the rigid transformation applied to the large object.
    """
    scene = _parse_scene(np.asarray(grid, dtype=int))
    out = _render_transformed(
        scene["grid"],
        scene["bg"],
        scene["glyph"]["coords"],
        scene["object"]["coords"],
        scene["object"]["color"],
        scene["mode"],
        scene["anchor_r"],
        scene["anchor_c"],
    )
    assert out is not None, "transformed object must stay in bounds and avoid the glyph"
    return out


def _mask_frontier(mask: np.ndarray) -> list[Coord]:
    h, w = mask.shape
    frontier: set[Coord] = set()
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                frontier.add((nr, nc))
    return list(frontier)


def _random_object_mask() -> np.ndarray:
    for _ in range(400):
        h = random.randint(4, 7)
        w = random.randint(4, 7)
        area_lo = max(6, h + w - 2)
        area_hi = min(h * w - 2, (3 * h * w) // 4)
        if area_lo > area_hi:
            continue

        mask = np.zeros((h, w), dtype=bool)
        seed = (random.randrange(h), random.randrange(w))
        mask[seed] = True
        target_area = random.randint(area_lo, area_hi)
        while int(mask.sum()) < target_area:
            frontier = _mask_frontier(mask)
            mask[random.choice(frontier)] = True

        if _bbox_coords(np.argwhere(mask)) != (0, 0, h - 1, w - 1):
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        if np.array_equal(mask, np.flipud(mask)):
            continue
        if np.array_equal(mask, np.rot90(mask, 2)):
            continue
        if max(np.sum(mask, axis=0)) < 2 or max(np.sum(mask, axis=1)) < 2:
            continue
        return mask
    raise RuntimeError("failed to sample an asymmetric object")


def _bbox_intersects(b1: BBox, b2: BBox, pad: int = 0) -> bool:
    return not (
        b1[2] + pad < b2[0]
        or b2[2] + pad < b1[0]
        or b1[3] + pad < b2[1]
        or b2[3] + pad < b1[1]
    )


def _enumerate_object_placements(
    grid_shape: tuple[int, int],
    obj_mask: np.ndarray,
    mode: str,
    anchor_r: int,
    anchor_c: int,
    glyph_bbox: BBox,
) -> list[tuple[int, int]]:
    h, w = map(int, grid_shape)
    oh, ow = obj_mask.shape
    rel = np.argwhere(obj_mask)
    placements: list[tuple[int, int]] = []

    for r0 in range(h - oh + 1):
        for c0 in range(w - ow + 1):
            obj_coords = rel + np.array([r0, c0], dtype=int)
            source_bbox = (r0, c0, r0 + oh - 1, c0 + ow - 1)

            if source_bbox[0] == 0 or source_bbox[1] == 0 or source_bbox[2] == h - 1 or source_bbox[3] == w - 1:
                continue
            if _bbox_intersects(source_bbox, glyph_bbox, pad=2):
                continue

            if mode == "vertical" and not (source_bbox[3] < anchor_c - 1 or source_bbox[1] > anchor_c + 1):
                continue
            if mode == "horizontal" and not (source_bbox[2] < anchor_r - 1 or source_bbox[0] > anchor_r + 1):
                continue
            if mode == "rot180":
                row_side = source_bbox[2] < anchor_r - 1 or source_bbox[0] > anchor_r + 1
                col_side = source_bbox[3] < anchor_c - 1 or source_bbox[1] > anchor_c + 1
                if not (row_side and col_side):
                    continue

            target = _transform_coords(obj_coords, mode, anchor_r, anchor_c)
            if np.any(target[:, 0] <= 0) or np.any(target[:, 0] >= h - 1) or np.any(target[:, 1] <= 0) or np.any(target[:, 1] >= w - 1):
                continue
            target_bbox = _bbox_coords(target)
            if _bbox_intersects(target_bbox, glyph_bbox, pad=2):
                continue
            if _bbox_intersects(source_bbox, target_bbox, pad=1):
                continue
            if {tuple(map(int, rc)) for rc in obj_coords} & {tuple(map(int, rc)) for rc in target}:
                continue

            placements.append((r0, c0))

    random.shuffle(placements)
    return placements


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    rows, cols = np.where(mask)
    grid[rows + int(top), cols + int(left)] = int(color)


def _wrong_anchor_offsets(mode: str) -> list[tuple[int, int]]:
    if mode == "vertical":
        return [(0, -1), (0, 1)]
    if mode == "horizontal":
        return [(-1, 0), (1, 0)]
    return [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        scene = _parse_scene(grid)
        out = main(grid)
    except AssertionError:
        return False

    obj_coords = scene["object"]["coords"]
    glyph_coords = scene["glyph"]["coords"]
    obj_color = scene["object"]["color"]
    bg = scene["bg"]
    mode = scene["mode"]
    anchor_r = scene["anchor_r"]
    anchor_c = scene["anchor_c"]

    transformed = _transform_coords(obj_coords, mode, anchor_r, anchor_c)
    if {tuple(map(int, rc)) for rc in obj_coords} & {tuple(map(int, rc)) for rc in transformed}:
        return False
    if np.array_equal(out, grid):
        return False
    if not np.array_equal(out[glyph_coords[:, 0], glyph_coords[:, 1]], grid[glyph_coords[:, 0], glyph_coords[:, 1]]):
        return False
    if int(np.sum(out == obj_color)) != int(np.sum(grid == obj_color)):
        return False

    alt_mode_valid = 0
    for alt_mode in MODES:
        if alt_mode == mode:
            continue
        alt = _render_transformed(grid, bg, glyph_coords, obj_coords, obj_color, alt_mode, anchor_r, anchor_c)
        if alt is not None:
            alt_mode_valid += 1
            if np.array_equal(alt, out):
                return False
    if alt_mode_valid == 0:
        return False

    wrong_anchor_valid = 0
    for dr, dc in _wrong_anchor_offsets(mode):
        alt = _render_transformed(
            grid,
            bg,
            glyph_coords,
            obj_coords,
            obj_color,
            mode,
            anchor_r + dr,
            anchor_c + dc,
        )
        if alt is not None:
            wrong_anchor_valid += 1
            if np.array_equal(alt, out):
                return False
    if wrong_anchor_valid == 0:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    mode = random.choice(MODES)
    height = random.randint(16, 24)
    width = random.randint(16, 24)
    obj_mask = _random_object_mask()
    obj_color, glyph_color = random.sample(range(1, 10), 2)

    glyph_top_choices = list(range(1, height - 3))
    glyph_left_choices = list(range(1, width - 3))
    random.shuffle(glyph_top_choices)
    random.shuffle(glyph_left_choices)

    for glyph_top in glyph_top_choices[:18]:
        for glyph_left in glyph_left_choices[:18]:
            anchor_r = glyph_top + 1
            anchor_c = glyph_left + 1
            if mode == "vertical" and not (4 <= anchor_c <= width - 5):
                continue
            if mode == "horizontal" and not (4 <= anchor_r <= height - 5):
                continue
            if mode == "rot180" and not (4 <= anchor_r <= height - 5 and 4 <= anchor_c <= width - 5):
                continue

            glyph_bbox = (glyph_top, glyph_left, glyph_top + 2, glyph_left + 2)
            placements = _enumerate_object_placements((height, width), obj_mask, mode, anchor_r, anchor_c, glyph_bbox)
            if not placements:
                continue

            obj_top, obj_left = random.choice(placements)
            grid = np.full((height, width), BG, dtype=int)
            _paint_mask(grid, GLYPH_MASKS[mode], glyph_top, glyph_left, glyph_color)
            _paint_mask(grid, obj_mask, obj_top, obj_left, obj_color)

            if _semantic_checks(grid):
                return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid glyph-spell sample")
