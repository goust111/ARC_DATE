from __future__ import annotations

from collections import Counter
import random

import numpy as np


Coord = tuple[int, int]
Shape = tuple[Coord, ...]


BACKGROUND = 0
COLOR_POOL = list(range(1, 10))
OBJECT_COUNT_RANGE = (4, 6)
GRID_SIZE_RANGE = (15, 20)
TARGET_KINDS = ("vertical", "horizontal", "rot180")
PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 1
MAX_GENERATION_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _scale_sprite(sprite: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(sprite, np.ones((int(factor), int(factor)), dtype=sprite.dtype))


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.asarray(coords, dtype=int))
    return components


def _symmetry_flags(sprite: np.ndarray) -> tuple[bool, bool, bool]:
    mask = np.asarray(sprite) != BACKGROUND
    return (
        bool(np.array_equal(mask, np.fliplr(mask))),
        bool(np.array_equal(mask, np.flipud(mask))),
        bool(np.array_equal(mask, np.rot90(mask, 2))),
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep the only symmetric sprite and enlarge it by the number of rejected sprites.

    Input structure:
    - A black background contains 4 to 6 separated monochromatic 4-connected sprites.
    - Each sprite is small, uses its own color, and can be tightly cropped to a minimal bbox.
    - Exactly one sprite has at least one symmetry among vertical reflection, horizontal reflection, or 180-degree rotation.

    Rule:
    - Extract every foreground connected component and crop it to its bbox.
    - Test each cropped sprite for vertical, horizontal, and 180-degree rotational symmetry.
    - Select the unique symmetric sprite as the target.
    - Let N be the number of non-target sprites, then replace every target pixel by an N x N block of the same color.
    - Output only the scaled target sprite on a black background, with size equal to the target bbox scaled by N.

    Key insight:
    - The symmetric-object filter and the scale factor are coupled: the discarded sprite count determines how much the surviving sprite expands.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    components = _components_mask(g != bg)
    assert components, "expected at least one foreground component"

    symmetric_sprites: list[np.ndarray] = []
    for coords in components:
        sprite = _crop_bbox(g, _bbox_coords(coords))
        if any(_symmetry_flags(sprite)):
            symmetric_sprites.append(sprite)

    assert len(symmetric_sprites) == 1, "expected exactly one symmetric sprite"
    factor = len(components) - 1
    assert factor >= 1
    return _scale_sprite(symmetric_sprites[0], factor)


def _normalize_shape(cells: set[Coord]) -> Shape:
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _shape_to_mask(shape: Shape) -> np.ndarray:
    h = max(r for r, _c in shape) + 1
    w = max(c for _r, c in shape) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in shape:
        mask[r, c] = True
    return mask


def _enumerate_polyominoes(max_area: int = 6) -> dict[int, set[Shape]]:
    shapes: dict[int, set[Shape]] = {1: {((0, 0),)}}
    for area in range(2, max_area + 1):
        expanded: set[Shape] = set()
        for shape in shapes[area - 1]:
            cells = set(shape)
            frontier: set[Coord] = set()
            for r, c in shape:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) not in cells:
                        frontier.add((nr, nc))
            for cell in frontier:
                expanded.add(_normalize_shape(cells | {cell}))
        shapes[area] = expanded
    return shapes


def _build_shape_pools() -> tuple[dict[str, list[Shape]], list[Shape]]:
    polyominoes = _enumerate_polyominoes(6)
    target_shapes = {kind: [] for kind in TARGET_KINDS}
    asymmetric_shapes: list[Shape] = []

    for area in range(3, 7):
        for shape in sorted(polyominoes[area]):
            vertical, horizontal, rot180 = _symmetry_flags(_shape_to_mask(shape))
            if vertical and not horizontal and not rot180:
                target_shapes["vertical"].append(shape)
            elif horizontal and not vertical and not rot180:
                target_shapes["horizontal"].append(shape)
            elif rot180 and not vertical and not horizontal:
                target_shapes["rot180"].append(shape)
            elif not (vertical or horizontal or rot180):
                asymmetric_shapes.append(shape)

    assert asymmetric_shapes
    assert all(target_shapes[kind] for kind in TARGET_KINDS)
    return target_shapes, asymmetric_shapes


TARGET_SHAPES, ASYMMETRIC_SHAPES = _build_shape_pools()


def _shape_to_sprite(shape: Shape, color: int) -> np.ndarray:
    return np.where(_shape_to_mask(shape), int(color), BACKGROUND).astype(int)


def _dilate_mask(mask: np.ndarray, padding: int, connectivity: int = 8) -> np.ndarray:
    if padding <= 0:
        return mask.copy()

    h, w = mask.shape
    out = mask.copy()
    if connectivity == 4:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        offsets = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    for _ in range(padding):
        grown = out.copy()
        for r, c in np.argwhere(out):
            rr = int(r)
            cc = int(c)
            for dr, dc in offsets:
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w:
                    grown[nr, nc] = True
        out = grown
    return out


def _random_free_location(
    canvas_shape: tuple[int, int],
    patch_mask: np.ndarray,
    occupied: np.ndarray,
    *,
    border: int,
    padding: int,
    padding_connectivity: int = 8,
) -> Coord:
    h, w = canvas_shape
    ph, pw = patch_mask.shape
    forbidden = _dilate_mask(occupied, padding, connectivity=padding_connectivity)

    candidates: list[Coord] = []
    for r0 in range(border, h - border - ph + 1):
        for c0 in range(border, w - border - pw + 1):
            window = forbidden[r0 : r0 + ph, c0 : c0 + pw]
            if not np.any(window & patch_mask):
                candidates.append((r0, c0))

    if not candidates:
        raise ValueError("no free location")
    return random.choice(candidates)


def _paste_sprite(canvas: np.ndarray, sprite: np.ndarray, top: int, left: int) -> None:
    mask = sprite != BACKGROUND
    h, w = sprite.shape
    canvas[top : top + h, left : left + w][mask] = sprite[mask]


def _expected_output(target_sprite: np.ndarray, object_count: int) -> np.ndarray:
    scale = object_count - 1
    out = _scale_sprite(target_sprite, scale)
    assert out.ndim == 2 and max(out.shape) <= 30
    return out


def _sample_scene() -> tuple[np.ndarray, np.ndarray] | None:
    object_count = random.randint(*OBJECT_COUNT_RANGE)
    target_kind = random.choice(TARGET_KINDS)
    target_shape = random.choice(TARGET_SHAPES[target_kind])
    distractor_shapes = random.sample(ASYMMETRIC_SHAPES, object_count - 1)
    colors = random.sample(COLOR_POOL, object_count)

    target_sprite = _shape_to_sprite(target_shape, colors[0])
    records = [{"sprite": target_sprite, "is_target": True}]
    for shape, color in zip(distractor_shapes, colors[1:], strict=False):
        records.append({"sprite": _shape_to_sprite(shape, color), "is_target": False})

    random.shuffle(records)
    order = records[:]
    random.shuffle(order)
    order.sort(key=lambda rec: rec["sprite"].shape[0] * rec["sprite"].shape[1], reverse=True)

    h = random.randint(*GRID_SIZE_RANGE)
    w = random.randint(*GRID_SIZE_RANGE)
    grid = np.full((h, w), BACKGROUND, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    for rec in order:
        sprite = rec["sprite"]
        mask = sprite != BACKGROUND
        try:
            r0, c0 = _random_free_location(
                grid.shape,
                mask,
                occupied,
                border=PLACEMENT_BORDER,
                padding=PLACEMENT_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        _paste_sprite(grid, sprite, r0, c0)
        sh, sw = sprite.shape
        occupied[r0 : r0 + sh, c0 : c0 + sw] |= mask

    expected = _expected_output(target_sprite, object_count)
    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        scene = _sample_scene()
        if scene is not None:
            return scene[0]
    raise RuntimeError("failed to generate a valid input grid")
