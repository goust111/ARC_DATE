from __future__ import annotations

from collections import Counter, deque

import numpy as np


BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
SHAPE_LIBRARY = [
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0], [1, 1], [0, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1], [1, 0], [1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 1, 0]], dtype=bool),
]
GEN_ATTEMPTS = 300


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _find_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for color in sorted(set(map(int, grid.ravel())) - {bg}):
        for coords in _components_from_mask(grid == color):
            out.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return out


def _shape_signature(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    rel = coords - coords.min(axis=0)
    return tuple(sorted((int(r), int(c)) for r, c in rel))


def _detect_container(
    grid: np.ndarray, comp: dict[str, object], bg: int
) -> dict[str, object] | None:
    coords = np.asarray(comp["coords"], dtype=int)
    r0, c0, r1, c1 = comp["bbox"]
    patch_h, patch_w = r1 - r0 + 1, c1 - c0 + 1
    local = np.zeros((patch_h, patch_w), dtype=bool)
    rel = coords - np.array([r0, c0])
    local[rel[:, 0], rel[:, 1]] = True

    holes = []
    for open_coords in _components_from_mask(~local):
        if (
            np.any(open_coords[:, 0] == 0)
            or np.any(open_coords[:, 0] == patch_h - 1)
            or np.any(open_coords[:, 1] == 0)
            or np.any(open_coords[:, 1] == patch_w - 1)
        ):
            continue
        hole = open_coords + np.array([r0, c0])
        key_mask = grid[hole[:, 0], hole[:, 1]] != bg
        if int(np.sum(key_mask)) != 1:
            return None
        key_r, key_c = hole[np.flatnonzero(key_mask)[0]]
        holes.append(
            {
                "hole_coords": hole,
                "key_color": int(grid[key_r, key_c]),
                "signature": _shape_signature(hole),
            }
        )
    if len(holes) != 1:
        return None
    return holes[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Move each outside sprite into the enclosed hole whose shape and key color both match it.

    Input structure:
    - The background is the dominant color.
    - Several large monochrome components each surround exactly one enclosed cavity.
    - Every cavity contains exactly one isolated key pixel of some color, and matching monochrome sprites lie elsewhere outside the containers.

    Rule:
    - Detect each holed container and treat its whole cavity, including the key-pixel cell, as the destination mask.
    - Find the unique outside sprite whose translated shape exactly matches that cavity and whose sprite color equals the cavity key color.
    - Erase all source sprites from their original locations, preserve the large containers, and paint every matched cavity with its sprite color.

    Key insight:
    - Shape alone is ambiguous because multiple holes can share one cavity shape, and color alone is ambiguous because multiple sprites can share one color, so the correct assignment is determined only by the joint `(shape, color)` key.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    components = _find_components(g, bg)

    holes: list[dict[str, object]] = []
    container_indices: set[int] = set()
    hole_cells = set()
    for idx, comp in enumerate(components):
        hole_info = _detect_container(g, comp, bg)
        if hole_info is None:
            continue
        container_indices.add(idx)
        holes.append(hole_info)
        for r, c in hole_info["hole_coords"]:
            hole_cells.add((int(r), int(c)))

    assert holes, "expected at least one holed container"

    sprites = []
    for idx, comp in enumerate(components):
        if idx in container_indices:
            continue
        coords = np.asarray(comp["coords"], dtype=int)
        if all((int(r), int(c)) in hole_cells for r, c in coords):
            continue
        sprites.append(
            {
                "coords": coords,
                "color": int(comp["color"]),
                "signature": _shape_signature(coords),
            }
        )

    assert sprites and len(sprites) == len(holes)

    sprite_map: dict[tuple[tuple[tuple[int, int], ...], int], dict[str, object]] = {}
    for sprite in sprites:
        key = (sprite["signature"], sprite["color"])
        assert key not in sprite_map, "sprite shape/color pairs must be unique"
        sprite_map[key] = sprite

    out = g.copy()
    for sprite in sprites:
        coords = sprite["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    used_keys = set()
    for hole in holes:
        key = (hole["signature"], hole["key_color"])
        assert key in sprite_map, "every hole must have a matching sprite"
        assert key not in used_keys, "hole shape/color pairs must be unique"
        used_keys.add(key)
        hole_coords = np.asarray(hole["hole_coords"], dtype=int)
        out[hole_coords[:, 0], hole_coords[:, 1]] = int(hole["key_color"])

    return out


def _transform_mask(rng: np.random.Generator, mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, int(rng.integers(4)))
    if int(rng.integers(2)):
        out = np.fliplr(out)
    return out.astype(bool)


def _sample_shapes(rng: np.random.Generator, count: int) -> list[np.ndarray]:
    indices = rng.permutation(len(SHAPE_LIBRARY))
    shapes = []
    signatures = set()
    for idx in indices:
        shape = _transform_mask(rng, SHAPE_LIBRARY[int(idx)])
        sig = _shape_signature(np.argwhere(shape))
        if sig in signatures:
            continue
        signatures.add(sig)
        shapes.append(shape)
        if len(shapes) == count:
            return shapes
    raise RuntimeError("failed to sample distinct shapes")


def _make_container_patch(shape: np.ndarray, outer_color: int, key_color: int, key_index: int) -> np.ndarray:
    h, w = shape.shape
    patch = np.full((h + 2, w + 2), outer_color, dtype=int)
    cells = np.argwhere(shape)
    for idx, (r, c) in enumerate(cells):
        patch[r + 1, c + 1] = key_color if idx == key_index else BG
    return patch


def _make_sprite_patch(shape: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(shape.shape, BG, dtype=int)
    patch[shape] = color
    return patch


def _place_patches_in_slots(
    rng: np.random.Generator, patches: list[np.ndarray]
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    total = len(patches)
    slot_h = max(int(p.shape[0]) for p in patches) + 2
    slot_w = max(int(p.shape[1]) for p in patches) + 2

    layouts = []
    for rows in range(2, min(total, 4) + 1):
        cols = (total + rows - 1) // rows
        base_h = rows * slot_h
        base_w = cols * slot_w
        if base_h <= 30 and base_w <= 30:
            layouts.append((rows, cols, base_h, base_w))
    if not layouts:
        raise RuntimeError("no slot layout fits inside 30x30")

    rows, cols, base_h, base_w = layouts[int(rng.integers(len(layouts)))]
    extra_h = min(4, 30 - base_h)
    extra_w = min(4, 30 - base_w)
    top = int(rng.integers(extra_h + 1))
    bottom = int(rng.integers(extra_h - top + 1))
    left = int(rng.integers(extra_w + 1))
    right = int(rng.integers(extra_w - left + 1))
    grid_h = base_h + top + bottom
    grid_w = base_w + left + right

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    positions: list[tuple[int, int]] = []
    order = rng.permutation(total)
    for slot_idx, patch_idx in enumerate(order):
        patch = patches[int(patch_idx)]
        ph, pw = patch.shape
        slot_r = slot_idx // cols
        slot_c = slot_idx % cols
        base_r = top + slot_r * slot_h
        base_c = left + slot_c * slot_w
        r = base_r + int(rng.integers(slot_h - ph))
        c = base_c + int(rng.integers(slot_w - pw))
        mask = patch != BG
        grid[r : r + ph, c : c + pw][mask] = patch[mask]
        positions.append((r, c))

    unshuffled = [None] * total
    for slot_idx, patch_idx in enumerate(order):
        unshuffled[int(patch_idx)] = positions[slot_idx]
    return grid, unshuffled


def _expected_output(
    grid: np.ndarray,
    pair_specs: list[dict[str, object]],
    container_positions: list[tuple[int, int]],
    sprite_positions: list[tuple[int, int]],
) -> np.ndarray:
    out = grid.copy()
    for spec, (sr, sc) in zip(pair_specs, sprite_positions):
        sprite_patch = spec["sprite_patch"]
        sprite_mask = sprite_patch != BG
        out[sr : sr + sprite_patch.shape[0], sc : sc + sprite_patch.shape[1]][sprite_mask] = BG

    for spec, (cr, cc) in zip(pair_specs, container_positions):
        shape = spec["shape"]
        color = int(spec["key_color"])
        cells = np.argwhere(shape)
        for r, c in cells:
            out[cr + 1 + int(r), cc + 1 + int(c)] = color
    return out


def _generate_case(rng: np.random.Generator) -> np.ndarray | None:
    shape_count, color_count = [(2, 2), (3, 2), (2, 3)][int(rng.integers(3))]
    shapes = _sample_shapes(rng, shape_count)

    key_colors = rng.choice(PALETTE, size=color_count, replace=False).tolist()
    container_palette = [c for c in PALETTE if c not in key_colors]
    pair_specs: list[dict[str, object]] = []

    pairs = [(si, ci) for si in range(shape_count) for ci in range(color_count)]
    rng.shuffle(pairs)
    container_colors = rng.choice(container_palette, size=len(pairs), replace=True).tolist()

    for outer_color, (shape_idx, color_idx) in zip(container_colors, pairs):
        shape = shapes[int(shape_idx)]
        key_color = int(key_colors[int(color_idx)])
        key_index = int(rng.integers(np.argwhere(shape).shape[0]))
        pair_specs.append(
            {
                "shape": shape,
                "key_color": key_color,
                "container_patch": _make_container_patch(shape, int(outer_color), key_color, key_index),
                "sprite_patch": _make_sprite_patch(shape, key_color),
            }
        )

    container_patches = [spec["container_patch"] for spec in pair_specs]
    sprite_patches = [spec["sprite_patch"] for spec in pair_specs]
    all_patches = container_patches + sprite_patches
    if max(p.shape[0] for p in all_patches) + 2 > 10 or max(p.shape[1] for p in all_patches) + 2 > 10:
        return None

    grid, positions = _place_patches_in_slots(rng, all_patches)
    container_positions = positions[: len(pair_specs)]
    sprite_positions = positions[len(pair_specs) :]
    expected = _expected_output(grid, pair_specs, container_positions, sprite_positions)

    shape_sigs = [_shape_signature(np.argwhere(spec["shape"])) for spec in pair_specs]
    color_list = [int(spec["key_color"]) for spec in pair_specs]
    if max(Counter(shape_sigs).values()) < 2 or max(Counter(color_list).values()) < 2:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_case(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
