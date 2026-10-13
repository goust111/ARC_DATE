from __future__ import annotations

from collections import Counter

import numpy as np


BG = 0
TRANSFORM_NAMES = ("id", "rot180", "flip_lr", "flip_ud")
PALETTE = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400
MAX_MASK_ATTEMPTS = 200


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_components(grid: np.ndarray, bg: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_component(grid: np.ndarray, coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    bbox = _bbox_from_coords(coords)
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy(), bbox


def _apply_transform(sprite: np.ndarray, name: str) -> np.ndarray:
    if name == "id":
        return sprite.copy()
    if name == "rot180":
        return np.rot90(sprite, 2).copy()
    if name == "flip_lr":
        return np.fliplr(sprite).copy()
    if name == "flip_ud":
        return np.flipud(sprite).copy()
    raise ValueError(f"unknown transform: {name}")


def _signature(sprite: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in sprite)


def _orbit_signature(sprite: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return min(_signature(_apply_transform(sprite, name)) for name in TRANSFORM_NAMES)


def _is_nonsymmetric(sprite: np.ndarray) -> bool:
    return all(not np.array_equal(sprite, _apply_transform(sprite, name)) for name in TRANSFORM_NAMES[1:])


def _transform_name(base: np.ndarray, target: np.ndarray) -> str:
    matches = [name for name in TRANSFORM_NAMES if np.array_equal(_apply_transform(base, name), target)]
    assert len(matches) == 1
    return matches[0]


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, transparent: int = BG) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != transparent
    assert not np.any(view[mask] != transparent)
    view[mask] = patch[mask]


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    stack = [tuple(int(v) for v in coords[0])]
    seen[stack[0]] = True
    count = 0
    while stack:
        r, c = stack.pop()
        count += 1
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return count == int(mask.sum())


def _sample_mask(rng: np.random.Generator) -> np.ndarray:
    for _ in range(MAX_MASK_ATTEMPTS):
        h = int(rng.integers(3, 5))
        w = int(rng.integers(3, 5))
        density = float(rng.uniform(0.42, 0.72))
        mask = rng.random((h, w)) < density
        area = int(mask.sum())
        if area < 4 or area >= h * w:
            continue
        if not mask[0, :].any() or not mask[-1, :].any() or not mask[:, 0].any() or not mask[:, -1].any():
            continue
        if not _is_connected(mask):
            continue
        if not _is_nonsymmetric(mask.astype(int)):
            continue
        return mask
    raise RuntimeError("failed to sample a nonsymmetric connected mask")


def _colorize_mask(rng: np.random.Generator, mask: np.ndarray, colors: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    rng.shuffle(coords)
    color_count = int(rng.integers(2, min(4, len(coords) + 1)))
    chosen = np.array(rng.choice(colors, size=color_count, replace=False), dtype=int)
    sprite = np.zeros(mask.shape, dtype=int)
    for idx, (r, c) in enumerate(coords):
        sprite[int(r), int(c)] = int(chosen[idx % color_count])
    return sprite


def _sample_program(rng: np.random.Generator, count: int) -> list[str]:
    if count < 2:
        return ["id"]
    rest = list(rng.choice(np.array(TRANSFORM_NAMES[1:]), size=min(3, count - 1), replace=False))
    while len(rest) < count - 1:
        rest.append(str(rng.choice(np.array(TRANSFORM_NAMES))))
    rng.shuffle(rest)
    return ["id", *rest]


def _sample_margin_pair(rng: np.random.Generator, base_extent: int) -> tuple[int, int, int]:
    slack = 30 - int(base_extent)
    assert slack >= 2
    total_margin = int(rng.integers(2, slack + 1))
    first = int(rng.integers(1, total_margin))
    second = total_margin - first
    return first, second, int(base_extent + total_margin)


def _build_case(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    family_mask = _sample_mask(rng)
    unique_mask = _sample_mask(rng)
    if _orbit_signature(family_mask.astype(int)) == _orbit_signature(unique_mask.astype(int)):
        raise ValueError("family and unique masks must differ")

    palette = np.array(rng.choice(PALETTE, size=6, replace=False), dtype=int)
    family_sprite = _colorize_mask(rng, family_mask, palette[:3])
    unique_sprite = _colorize_mask(rng, unique_mask, palette[3:])

    k = int(rng.integers(3, 6))
    program = _sample_program(rng, k)
    axis = str(rng.choice(np.array(["horizontal", "vertical"])))
    gap = int(rng.integers(1, 3))
    between = int(rng.integers(2, 5))

    family_parts = [_apply_transform(family_sprite, name) for name in program]
    expected = np.concatenate(
        [_apply_transform(unique_sprite, name) for name in program],
        axis=1 if axis == "horizontal" else 0,
    )
    assert max(expected.shape) <= 30

    fh, fw = family_sprite.shape
    uh, uw = unique_sprite.shape

    if axis == "horizontal":
        family_span = k * fw + (k - 1) * gap
        core_w = max(family_span, uw)
        base_h = fh + between + uh
        left, right, width = _sample_margin_pair(rng, core_w)
        top, bottom, height = _sample_margin_pair(rng, base_h)
        grid = np.zeros((height, width), dtype=int)

        family_offset = int(rng.integers(0, core_w - family_span + 1))
        unique_offset = int(rng.integers(0, core_w - uw + 1))
        unique_above = bool(rng.integers(0, 2))

        if unique_above:
            unique_r0 = top
            family_r0 = top + uh + between
        else:
            family_r0 = top
            unique_r0 = top + fh + between

        unique_c0 = left + unique_offset
        _paste(grid, unique_sprite, unique_r0, unique_c0)

        for idx, part in enumerate(family_parts):
            c0 = left + family_offset + idx * (fw + gap)
            _paste(grid, part, family_r0, c0)
    else:
        family_span = k * fh + (k - 1) * gap
        core_h = max(family_span, uh)
        base_w = fw + between + uw
        top, bottom, height = _sample_margin_pair(rng, core_h)
        left, right, width = _sample_margin_pair(rng, base_w)
        grid = np.zeros((height, width), dtype=int)

        family_offset = int(rng.integers(0, core_h - family_span + 1))
        unique_offset = int(rng.integers(0, core_h - uh + 1))
        unique_left = bool(rng.integers(0, 2))

        if unique_left:
            unique_c0 = left
            family_c0 = left + uw + between
        else:
            family_c0 = left
            unique_c0 = left + fw + between

        unique_r0 = top + unique_offset
        _paste(grid, unique_sprite, unique_r0, unique_c0)

        for idx, part in enumerate(family_parts):
            r0 = top + family_offset + idx * (fh + gap)
            _paste(grid, part, r0, family_c0)

    assert max(grid.shape) <= 30
    out = main(grid)
    assert np.array_equal(out, expected)
    return grid, expected


def main(grid: np.ndarray) -> np.ndarray:
    """
    Apply the pose program encoded by a repeated sprite family to the unique sprite and concatenate the transformed copies.

    Input structure:
    - The grid has a dominant background plus several separated multicolor 4-connected sprites.
    - One transform-equivalent family contains k copies of the same nonsymmetric sprite, arranged in a single horizontal row or vertical column.
    - Exactly one other nonsymmetric sprite belongs to no repeated family.

    Rule:
    - Crop every foreground sprite to its bounding box and group sprites up to the four allowed poses: identity, 180-degree rotation, left-right mirror, and up-down mirror.
    - Identify the repeated family, sort its copies in reading order along the arrangement axis, and use the first family copy as the reference pose.
    - For each family copy, read which of the four poses maps the first copy to it; apply that same pose to the unique sprite crop, then concatenate the transformed unique copies along the family's axis.

    Key insight:
    - The repeated sprites are a pose program: they encode copy count, concatenation direction, and per-copy orientation, while the singleton sprite provides the content to transform.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    components = _find_components(g, bg)
    assert len(components) >= 4

    groups: dict[tuple[tuple[int, ...], ...], list[dict[str, object]]] = {}
    for coords in components:
        crop, bbox = _crop_component(g, coords)
        groups.setdefault(_orbit_signature(crop), []).append({"crop": crop, "bbox": bbox})

    family_groups = [items for items in groups.values() if len(items) > 1]
    unique_groups = [items for items in groups.values() if len(items) == 1]
    assert len(groups) == 2
    assert len(family_groups) == 1
    assert len(unique_groups) == 1

    family = family_groups[0]
    unique_crop = unique_groups[0][0]["crop"]
    assert isinstance(unique_crop, np.ndarray)
    base_crop = family[0]["crop"]
    assert isinstance(base_crop, np.ndarray)
    assert _is_nonsymmetric(base_crop)
    assert _is_nonsymmetric(unique_crop)

    row_centers = [0.5 * (item["bbox"][0] + item["bbox"][2]) for item in family]
    col_centers = [0.5 * (item["bbox"][1] + item["bbox"][3]) for item in family]
    axis = 1 if max(col_centers) - min(col_centers) > max(row_centers) - min(row_centers) else 0

    if axis == 1:
        ordered = sorted(family, key=lambda item: (item["bbox"][1], item["bbox"][0]))
    else:
        ordered = sorted(family, key=lambda item: (item["bbox"][0], item["bbox"][1]))

    reference = ordered[0]["crop"]
    assert isinstance(reference, np.ndarray)
    program = []
    for item in ordered:
        crop = item["crop"]
        assert isinstance(crop, np.ndarray)
        program.append(_transform_name(reference, crop))

    parts = [_apply_transform(unique_crop, name) for name in program]
    return np.concatenate(parts, axis=axis)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            grid, _expected = _build_case(rng)
            return grid
        except (AssertionError, ValueError, RuntimeError):
            continue
    raise RuntimeError("failed to generate a valid pose-program ARC input")
