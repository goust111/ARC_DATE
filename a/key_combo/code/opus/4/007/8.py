from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox_coords, components_mask, crop_bbox, get_bg, paste, scale_sprite


BG = Color.BLACK
BLUE = Color.BLUE
RED = Color.RED
BASE_TILE_SIZE = 3
GEN_RETRIES = 400
GRID_MIN = 22
GRID_MAX = 30
GRID_BORDER = 1
PLACEMENT_GAP = 2
THREE_COLLAGE_PROB = 0.8
BIG_SCALE = 2


def _assemble_collage(tile: np.ndarray) -> np.ndarray:
    return np.block(
        [
            [tile, np.rot90(tile, -1)],
            [np.rot90(tile, 1), np.rot90(tile, 2)],
        ]
    )


def _connected8(mask: np.ndarray) -> bool:
    coords = components_mask(mask.astype(bool), connectivity=8)
    return len(coords) == 1


def _rotation_class(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    rots = []
    for k in range(4):
        rots.append(tuple(int(v) for v in np.rot90(mask, k).ravel()))
    return tuple(sorted(rots))


def _build_prototype_candidates() -> dict[int, list[np.ndarray]]:
    by_area: dict[int, list[np.ndarray]] = {}
    seen: set[tuple[tuple[int, ...], ...]] = set()
    for bits in range(1 << (BASE_TILE_SIZE * BASE_TILE_SIZE)):
        arr = np.array(
            [(bits >> idx) & 1 for idx in range(BASE_TILE_SIZE * BASE_TILE_SIZE)],
            dtype=int,
        ).reshape(BASE_TILE_SIZE, BASE_TILE_SIZE)
        if not (4 <= int(arr.sum()) <= 5):
            continue
        if arr[-1, -1] != 1:
            continue
        if not arr[0, :].any() or not arr[:, 0].any():
            continue
        if np.array_equal(arr, np.rot90(arr, 1)) or np.array_equal(arr, np.rot90(arr, 2)):
            continue
        if not _connected8(arr):
            continue
        collage = _assemble_collage(arr)
        if not _connected8(collage):
            continue
        key = _rotation_class(arr)
        if key in seen:
            continue
        seen.add(key)
        by_area.setdefault(int(arr.sum()), []).append(arr.astype(bool))
    return by_area


PROTOTYPE_CANDIDATES = _build_prototype_candidates()


def _choose_prototypes(
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray]:
    valid_areas = [area for area, masks in PROTOTYPE_CANDIDATES.items() if len(masks) >= 2]
    area = rng.choice(valid_areas)
    proto_a, proto_b = rng.sample(PROTOTYPE_CANDIDATES[area], 2)
    proto_a = np.rot90(proto_a, rng.randrange(4))
    proto_b = np.rot90(proto_b, rng.randrange(4))
    return proto_a.astype(bool), proto_b.astype(bool)


def _tile_from_mask(mask: np.ndarray, color: int, scale: int) -> np.ndarray:
    scaled = scale_sprite(mask.astype(int), scale).astype(bool)
    tile = np.full(scaled.shape, BG, dtype=int)
    tile[scaled] = int(color)
    return tile


def _extract_collages(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    collages: list[dict] = []
    for coords in components_mask(g != bg, connectivity=8):
        r0, c0, r1, c1 = bbox_coords(coords)
        patch = crop_bbox(g, (r0, c0, r1, c1))
        height, width = map(int, patch.shape)
        assert height == width and height % 2 == 0, "each collage must occupy an even square bbox"
        colors = sorted(set(map(int, np.unique(patch))) - {bg})
        assert len(colors) == 1 and colors[0] in {BLUE, RED}, "collage colors must be only blue or red"
        half = height // 2
        tl = patch[:half, :half]
        tr = patch[:half, half:]
        bl = patch[half:, :half]
        br = patch[half:, half:]
        src_mask = tl != bg
        assert np.array_equal(tr != bg, np.rot90(src_mask, -1)), "top-right quadrant must be clockwise rotation"
        assert np.array_equal(bl != bg, np.rot90(src_mask, 1)), "bottom-left quadrant must be counterclockwise rotation"
        assert np.array_equal(br != bg, np.rot90(src_mask, 2)), "bottom-right quadrant must be 180-degree rotation"
        collages.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "color": int(colors[0]),
                "tile_size": int(half),
                "source_mask": src_mask.copy(),
            }
        )
    assert 2 <= len(collages) <= 3, "expected 2 or 3 collage objects"
    return collages


def _infer_templates(collages: list[dict]) -> dict[int, dict]:
    by_color = {BLUE: [], RED: []}
    for collage in collages:
        by_color[collage["color"]].append(collage)
    assert by_color[BLUE] and by_color[RED], "both prototype colors must appear at least once"

    templates: dict[int, dict] = {}
    for color, group in by_color.items():
        min_size = min(item["tile_size"] for item in group)
        bases = [item["source_mask"] for item in group if item["tile_size"] == min_size]
        base = bases[0]
        for other in bases[1:]:
            assert np.array_equal(base, other), "smallest same-color tiles must agree on the prototype"
        templates[color] = {"size": int(min_size), "mask": base.copy()}
    return templates


def _swapped_tile(collage: dict, templates: dict[int, dict]) -> np.ndarray:
    color = int(collage["color"])
    other = BLUE if color == RED else RED
    tile_size = int(collage["tile_size"])

    src_size = int(templates[color]["size"])
    dst_size = int(templates[other]["size"])
    assert tile_size % src_size == 0, "tile size must be an integer multiple of its own prototype base"
    assert tile_size % dst_size == 0, "tile size must be compatible with the alternate prototype base"

    scaled_other = scale_sprite(templates[other]["mask"].astype(int), tile_size // dst_size).astype(bool)
    assert scaled_other.shape == (tile_size, tile_size)
    tile = np.full((tile_size, tile_size), BG, dtype=int)
    tile[scaled_other] = other
    return tile


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decompose each rotational collage, swap its prototype, then rebuild the rotated four-quadrant collage.

    Input structure:
    - A black background contains 2 or 3 non-overlapping monochrome collage objects.
    - Each collage occupies one even square bbox split into four equal quadrants.
    - The top-left quadrant is a source tile; the other three quadrants are its 90-degree, -90-degree, and 180-degree rotations.
    - Red collages and blue collages encode the two prototype families.

    Rule:
    - Detect every collage whose four quadrants satisfy the rotational arrangement.
    - Extract the top-left source tile from each collage.
    - For each color family, use the smallest source tile in that family as the prototype template.
    - Replace every source tile with the opposite color family's prototype, scaling that prototype to the same tile size when needed.
    - Rebuild the collage in place using the swapped source tile in the top-left and its rotated copies in the other three quadrants.

    Key insight:
    - Rotation is used twice: first to recognize and factor each collage into one canonical source tile, then again to synthesize the swapped collage from that new source tile.
    """
    g = np.asarray(grid, dtype=int)
    collages = _extract_collages(g)
    templates = _infer_templates(collages)

    out = g.copy()
    for collage in collages:
        r0, c0, _r1, _c1 = collage["bbox"]
        swapped = _swapped_tile(collage, templates)
        rebuilt = _assemble_collage(swapped)
        paste(out, rebuilt, r0, c0)
    return out


def _rect_ok(placed: list[dict], r0: int, c0: int, side: int, height: int, width: int) -> bool:
    if r0 < GRID_BORDER or c0 < GRID_BORDER:
        return False
    if r0 + side > height - GRID_BORDER or c0 + side > width - GRID_BORDER:
        return False
    for item in placed:
        pr0, pc0 = item["top_left"]
        pside = item["side"]
        if not (
            r0 + side + PLACEMENT_GAP <= pr0
            or pr0 + pside + PLACEMENT_GAP <= r0
            or c0 + side + PLACEMENT_GAP <= pc0
            or pc0 + pside + PLACEMENT_GAP <= c0
        ):
            return False
    return True


def _place_collages(
    rng: random.Random,
    patches: list[dict],
    height: int,
    width: int,
) -> list[dict] | None:
    placed: list[dict] = []
    for patch in sorted(patches, key=lambda item: (-item["side"], item["color"])):
        side = int(patch["side"])
        slots = [
            (r0, c0)
            for r0 in range(GRID_BORDER, height - side - GRID_BORDER + 1)
            for c0 in range(GRID_BORDER, width - side - GRID_BORDER + 1)
        ]
        rng.shuffle(slots)
        for r0, c0 in slots:
            if _rect_ok(placed, r0, c0, side, height, width):
                placed.append({**patch, "top_left": (int(r0), int(c0))})
                break
        else:
            return None
    return placed


def _expected_swapped_patch(meta: dict) -> np.ndarray:
    other_color = BLUE if meta["color"] == RED else RED
    other_mask = meta["proto_b"] if meta["color"] == RED else meta["proto_a"]
    tile = _tile_from_mask(other_mask, other_color, meta["scale"])
    return _assemble_collage(tile)


def _validate_case(grid: np.ndarray, expected: np.ndarray, meta: dict) -> bool:
    collages = _extract_collages(grid)
    if len(collages) != len(meta["placements"]):
        return False
    templates = _infer_templates(collages)
    if not np.array_equal(templates[RED]["mask"], meta["proto_a"]):
        return False
    if not np.array_equal(templates[BLUE]["mask"], meta["proto_b"]):
        return False

    by_bbox = {tuple(item["bbox"]): item for item in meta["placements"]}
    for collage in collages:
        bbox = tuple(collage["bbox"])
        placed = by_bbox.get(bbox)
        if placed is None:
            return False
        source_patch = crop_bbox(grid, bbox)
        expected_source = _assemble_collage(
            _tile_from_mask(
                placed["proto_a"] if placed["color"] == RED else placed["proto_b"],
                placed["color"],
                placed["scale"],
            )
        )
        if not np.array_equal(source_patch, expected_source):
            return False
        swapped_patch = crop_bbox(expected, bbox)
        expected_swapped = _expected_swapped_patch(placed)
        if not np.array_equal(swapped_patch, expected_swapped):
            return False
        if np.array_equal(source_patch, swapped_patch):
            return False

    return True


def _generate_case(rng: random.Random | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    rng = rng or random.Random()
    proto_a, proto_b = _choose_prototypes(rng)
    count = 3 if rng.random() < THREE_COLLAGE_PROB else 2

    specs = [(RED, 1), (BLUE, 1)]
    if count == 3:
        specs.append((rng.choice([RED, BLUE]), BIG_SCALE))
    rng.shuffle(specs)

    patches: list[dict] = []
    for color, scale in specs:
        proto = proto_a if color == RED else proto_b
        tile = _tile_from_mask(proto, color, scale)
        patch = _assemble_collage(tile)
        side = int(patch.shape[0])
        patches.append(
            {
                "color": int(color),
                "scale": int(scale),
                "side": side,
                "patch": patch,
                "proto_a": proto_a,
                "proto_b": proto_b,
            }
        )

    placed = None
    height = width = 0
    for _ in range(40):
        height = rng.randint(GRID_MIN, GRID_MAX)
        width = rng.randint(GRID_MIN, GRID_MAX)
        placed = _place_collages(rng, patches, height, width)
        if placed is not None:
            break
    if placed is None:
        raise RuntimeError("placement failed")

    grid = np.full((height, width), BG, dtype=int)
    expected = grid.copy()
    meta_placements: list[dict] = []

    for item in placed:
        r0, c0 = item["top_left"]
        paste(grid, item["patch"], r0, c0)
        swapped_patch = _expected_swapped_patch(item)
        paste(expected, swapped_patch, r0, c0)
        meta_placements.append(
            {
                "bbox": (r0, c0, r0 + item["side"] - 1, c0 + item["side"] - 1),
                "color": item["color"],
                "scale": item["scale"],
                "proto_a": proto_a,
                "proto_b": proto_b,
            }
        )

    meta = {"proto_a": proto_a, "proto_b": proto_b, "placements": meta_placements}
    if not _validate_case(grid, expected, meta):
        raise RuntimeError("semantic validation failed")
    return grid, expected, meta


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            grid, expected, _meta = _generate_case()
        except RuntimeError:
            continue
        if np.array_equal(main(grid), expected):
            return grid
    raise RuntimeError("failed to generate a valid collage-swap instance")
