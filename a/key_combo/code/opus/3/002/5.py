from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
SPRITE = 1
WALL = 2

_QUADRANTS = ("UL", "UR", "LL", "LR")
_ROTATION_K = {"UL": 2, "UR": 3, "LL": 1, "LR": 0}

_GRID_MIN = 20
_GRID_MAX = 26
_REGION_COUNT_WEIGHTS = ((3, 0.4), (4, 0.6))
_GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return _bbox_from_coords(coords)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _component_dicts(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue

            queue = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            touches_border = False

            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                if rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1:
                    touches_border = True
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            coords = np.array(cells, dtype=int)
            comp_mask = np.zeros((h, w), dtype=bool)
            comp_mask[coords[:, 0], coords[:, 1]] = True
            out.append(
                {
                    "coords": coords,
                    "mask": comp_mask,
                    "bbox": _bbox_from_coords(coords),
                    "area": int(coords.shape[0]),
                    "touches_border": touches_border,
                }
            )
    return out


def _find_enclosed_regions(grid: np.ndarray) -> list[dict]:
    comps = _component_dicts(np.asarray(grid) != WALL)
    return [comp for comp in comps if not comp["touches_border"]]


def _quadrant_of_bbox(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> str:
    r0, c0, r1, c1 = bbox
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    row_mid = (shape[0] - 1) / 2.0
    col_mid = (shape[1] - 1) / 2.0
    top = center_r < row_mid
    left = center_c < col_mid
    if top and left:
        return "UL"
    if top and not left:
        return "UR"
    if not top and left:
        return "LL"
    return "LR"


def _tile_mask_over_region(
    out: np.ndarray,
    region_mask: np.ndarray,
    region_bbox: tuple[int, int, int, int],
    tile_mask: np.ndarray,
) -> None:
    r0, c0, _r1, _c1 = region_bbox
    tile_h, tile_w = tile_mask.shape
    out[region_mask] = BG
    for rr, cc in np.argwhere(region_mask):
        if tile_mask[(int(rr) - r0) % tile_h, (int(cc) - c0) % tile_w]:
            out[int(rr), int(cc)] = SPRITE


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each enclosed-region sprite by its global quadrant and tile it within that same region.

    Input structure:
    - The grid has a black background, plus several disjoint red closed boundaries.
    - Each red boundary encloses one 4-connected internal region of black cells containing a small blue sprite.
    - The sprite is aligned to the enclosed region's internal top-left corner.

    Rule:
    - Detect every 4-connected non-red region that is fully enclosed by red and does not touch the outer border.
    - Inside each enclosed region, extract the blue sprite as a minimal bbox template.
    - Use the enclosed region bbox center in the full grid to choose the rotation: upper-left -> 180 degrees, upper-right -> 90 degrees clockwise, lower-left -> 90 degrees counterclockwise, lower-right -> unchanged.
    - Starting from that region's internal top-left corner, tile the rotated sprite bbox across the region and clip the copies to the region mask.
    - Leave the red boundary pixels and the exterior black background unchanged.

    Key insight:
    - The local fill pattern comes from the sprite inside each enclosure, but the sprite orientation is controlled globally by which quadrant the enclosure center occupies.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    for region in _find_enclosed_regions(g):
        sprite_mask = (g == SPRITE) & region["mask"]
        sprite_bbox = _bbox_from_mask(sprite_mask)
        assert sprite_bbox is not None, "each enclosed region must contain one blue sprite"
        sprite = _crop_bbox(sprite_mask.astype(np.uint8), sprite_bbox).astype(bool)
        quadrant = _quadrant_of_bbox(region["bbox"], g.shape)
        rotated = np.rot90(sprite, _ROTATION_K[quadrant])
        _tile_mask_over_region(out, region["mask"], region["bbox"], rotated)
    return out


def _tile_signature(mask: np.ndarray) -> bytes:
    tiled = np.tile(mask.astype(np.uint8), (3, 3))[:6, :6]
    return tiled.tobytes()


def _build_sprite_bank() -> list[np.ndarray]:
    candidates = [
        np.array([[1, 1], [1, 0]], dtype=np.uint8),
        np.array([[1, 1, 1], [1, 0, 0]], dtype=np.uint8),
        np.array([[1, 1, 1], [0, 1, 0]], dtype=np.uint8),
        np.array([[1, 0, 1], [1, 1, 1]], dtype=np.uint8),
        np.array([[1, 1, 0], [1, 1, 1]], dtype=np.uint8),
    ]
    bank: list[np.ndarray] = []
    for sprite in candidates:
        sigs = {_tile_signature(np.rot90(sprite, k).astype(bool)) for k in range(4)}
        if len(sigs) == 4:
            bank.append(sprite.astype(bool))
    assert bank, "sprite bank must contain asymmetric tiling templates"
    return bank


_SPRITE_BANK = _build_sprite_bank()


def _build_container(local_h: int, local_w: int, notch: tuple[str, int, int] | None) -> dict | None:
    box = np.zeros((local_h + 2, local_w + 2), dtype=int)
    box[0, :] = WALL
    box[-1, :] = WALL
    box[:, 0] = WALL
    box[:, -1] = WALL

    if notch is not None:
        kind, anchor, length = notch
        if kind == "vertical":
            box[local_h + 1 - length : local_h + 1, anchor + 1] = WALL
        else:
            box[anchor + 1, local_w + 1 - length : local_w + 1] = WALL

    regions = _find_enclosed_regions(box)
    if len(regions) != 1:
        return None

    region = regions[0]
    red_count = int(np.count_nonzero(box == WALL))
    if not (14 <= red_count <= 30):
        return None

    bbox = region["bbox"]
    if bbox[0] != 1 or bbox[1] != 1:
        return None

    return {
        "box": box,
        "region_mask_local": region["mask"],
        "region_bbox_local": bbox,
        "region_area": region["area"],
        "red_count": red_count,
    }


def _build_container_bank() -> list[dict]:
    bank: list[dict] = []
    for local_h in range(4, 7):
        for local_w in range(4, 8):
            bank.append(_build_container(local_h, local_w, None))
            for col in range(1, local_w - 1):
                for length in range(1, local_h - 1):
                    bank.append(_build_container(local_h, local_w, ("vertical", col, length)))
            for row in range(1, local_h - 1):
                for length in range(1, local_w - 1):
                    bank.append(_build_container(local_h, local_w, ("horizontal", row, length)))
    return [entry for entry in bank if entry is not None]


_CONTAINER_BANK = _build_container_bank()


def _choose_weighted_region_count() -> int:
    values = [item[0] for item in _REGION_COUNT_WEIGHTS]
    weights = [item[1] for item in _REGION_COUNT_WEIGHTS]
    return int(random.choices(values, weights=weights, k=1)[0])


def _assign_sprites(quadrants: list[str]) -> dict[str, np.ndarray]:
    shared = random.choice(_SPRITE_BANK)
    shared_count = 2 if len(quadrants) == 3 or random.random() < 0.7 else 3
    shared_quadrants = set(random.sample(quadrants, shared_count))
    assignments: dict[str, np.ndarray] = {}
    for quadrant in quadrants:
        if quadrant in shared_quadrants:
            assignments[quadrant] = shared

    alternatives = [sprite for sprite in _SPRITE_BANK if not np.array_equal(sprite, shared)]
    remaining = [quadrant for quadrant in quadrants if quadrant not in assignments]
    for i, quadrant in enumerate(remaining):
        if i == 0 and alternatives:
            assignments[quadrant] = random.choice(alternatives)
        else:
            assignments[quadrant] = random.choice(_SPRITE_BANK)

    distinct = {assignments[q].tobytes() for q in quadrants}
    assert len(distinct) >= 2 or len(quadrants) == 1
    return assignments


def _pick_container(sprite: np.ndarray) -> dict:
    min_area = 4 * int(np.count_nonzero(sprite))
    choices = [entry for entry in _CONTAINER_BANK if entry["region_area"] >= min_area]
    assert choices
    return random.choice(choices)


def _placement_ranges(
    quadrant: str, shape: tuple[int, int], local_shape: tuple[int, int]
) -> tuple[range, range] | None:
    h, w = shape
    box_h, box_w = local_shape
    row_mid = h // 2
    col_mid = w // 2

    if quadrant in {"UL", "UR"}:
        row_lo, row_hi = 1, row_mid - box_h - 1
    else:
        row_lo, row_hi = row_mid + 1, h - box_h - 1

    if quadrant in {"UL", "LL"}:
        col_lo, col_hi = 1, col_mid - box_w - 1
    else:
        col_lo, col_hi = col_mid + 1, w - box_w - 1

    if row_lo > row_hi or col_lo > col_hi:
        return None
    return range(row_lo, row_hi + 1), range(col_lo, col_hi + 1)


def _place_region(
    grid: np.ndarray,
    quadrant: str,
    sprite: np.ndarray,
    container: dict,
) -> dict | None:
    ranges = _placement_ranges(quadrant, grid.shape, container["box"].shape)
    if ranges is None:
        return None
    row_range, col_range = ranges
    top = random.choice(list(row_range))
    left = random.choice(list(col_range))

    box = container["box"]
    box_h, box_w = box.shape
    if np.any(grid[top : top + box_h, left : left + box_w] != BG):
        return None

    grid[top : top + box_h, left : left + box_w] = box
    sprite_h, sprite_w = sprite.shape
    sprite_slice = grid[top + 1 : top + 1 + sprite_h, left + 1 : left + 1 + sprite_w]
    sprite_slice[sprite] = SPRITE

    region_mask = np.zeros_like(grid, dtype=bool)
    local_mask = container["region_mask_local"]
    rr, cc = np.nonzero(local_mask)
    region_mask[top + rr, left + cc] = True

    red_mask = np.zeros_like(grid, dtype=bool)
    wall_rr, wall_cc = np.nonzero(box == WALL)
    red_mask[top + wall_rr, left + wall_cc] = True

    return {
        "quadrant": quadrant,
        "sprite": sprite.copy(),
        "region_mask": region_mask,
        "region_bbox": (top + 1, left + 1, top + box_h - 2, left + box_w - 2),
        "red_mask": red_mask,
    }


def _expected_output_from_specs(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        rotated = np.rot90(spec["sprite"], _ROTATION_K[spec["quadrant"]])
        _tile_mask_over_region(out, spec["region_mask"], spec["region_bbox"], rotated)
    return out


def _validate_generated_scene(grid: np.ndarray, specs: list[dict]) -> None:
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert _GRID_MIN <= grid.shape[0] <= 30 and _GRID_MIN <= grid.shape[1] <= 30
    assert len(specs) in {3, 4}

    detected_regions = _find_enclosed_regions(grid)
    detected_walls = _component_dicts(grid == WALL)
    assert len(detected_regions) == len(specs)
    assert len(detected_walls) == len(specs)

    spec_by_quadrant = {spec["quadrant"]: spec for spec in specs}
    found_quadrants = {_quadrant_of_bbox(region["bbox"], grid.shape) for region in detected_regions}
    assert found_quadrants == set(spec_by_quadrant)
    assert len(found_quadrants) >= 3

    for wall in detected_walls:
        assert 14 <= wall["area"] <= 30

    for region in detected_regions:
        quadrant = _quadrant_of_bbox(region["bbox"], grid.shape)
        spec = spec_by_quadrant[quadrant]
        assert np.array_equal(region["mask"], spec["region_mask"])
        assert region["bbox"] == spec["region_bbox"]

        sprite_mask = (grid == SPRITE) & region["mask"]
        sprite_comps = _component_dicts(sprite_mask)
        assert len(sprite_comps) == 1
        sprite_bbox = sprite_comps[0]["bbox"]
        assert sprite_bbox[:2] == region["bbox"][:2]
        assert sprite_comps[0]["area"] == int(np.count_nonzero(spec["sprite"]))
        assert region["area"] >= 4 * sprite_comps[0]["area"]

    output = main(grid)
    expected = _expected_output_from_specs(grid, specs)
    assert np.array_equal(output, expected)
    assert not np.array_equal(output, grid)

    region_signatures = set()
    for spec in specs:
        rotated = np.rot90(spec["sprite"], _ROTATION_K[spec["quadrant"]])
        region_signatures.add((spec["quadrant"], _tile_signature(rotated)))
    assert len(region_signatures) >= 3


def _generate_scene() -> tuple[np.ndarray, list[dict]] | None:
    height = random.randint(_GRID_MIN, _GRID_MAX)
    width = random.randint(_GRID_MIN, _GRID_MAX)
    grid = np.zeros((height, width), dtype=int)

    region_count = _choose_weighted_region_count()
    quadrants = list(_QUADRANTS if region_count == 4 else random.sample(_QUADRANTS, 3))
    assignments = _assign_sprites(quadrants)

    specs: list[dict] = []
    for quadrant in quadrants:
        sprite = assignments[quadrant]
        container = _pick_container(sprite)
        placed = _place_region(grid, quadrant, sprite, container)
        if placed is None:
            return None
        specs.append(placed)

    if len({spec["sprite"].tobytes() for spec in specs}) < 2:
        return None
    return grid, specs


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        built = _generate_scene()
        if built is None:
            continue
        grid, specs = built
        try:
            _validate_generated_scene(grid, specs)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
