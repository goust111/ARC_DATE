import random
from typing import Iterable

import numpy as np


BACKGROUND = 0
TEMPLATE_COLORS = (1, 2)
FRAME_COLOR = 5
SPRITE_COLORS = (3, 4, 6)


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int] | None:
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _quadrant_slices(shape: tuple[int, int]) -> dict[str, tuple[slice, slice]]:
    h, w = map(int, shape)
    if h != w or h < 8 or (h - 2) % 2 != 0:
        raise ValueError("expected a square grid with a 1-cell border and equal inner quadrants")
    q = (h - 2) // 2
    return {
        "tl": (slice(1, 1 + q), slice(1, 1 + q)),
        "tr": (slice(1, 1 + q), slice(1 + q, 1 + 2 * q)),
        "bl": (slice(1 + q, 1 + 2 * q), slice(1, 1 + q)),
        "br": (slice(1 + q, 1 + 2 * q), slice(1 + q, 1 + 2 * q)),
    }


def _centered_mask(sprite_coords: np.ndarray, canvas_size: int) -> np.ndarray:
    q = int(canvas_size)
    out = np.zeros((q, q), dtype=bool)
    box = _bbox_from_coords(sprite_coords)
    if box is None:
        return out
    r0, c0, r1, c1 = box
    for r, c in sprite_coords:
        rr_num = (q - 1) + 2 * int(r) - (r0 + r1)
        cc_num = (q - 1) + 2 * int(c) - (c0 + c1)
        if rr_num % 2 != 0 or cc_num % 2 != 0:
            continue
        rr = rr_num // 2
        cc = cc_num // 2
        if 0 <= rr < q and 0 <= cc < q:
            out[rr, cc] = True
    return out


def _extract_sprite_mask_and_color(patch: np.ndarray) -> tuple[np.ndarray, int]:
    gray_coords = np.argwhere(patch == FRAME_COLOR)
    box = _bbox_from_coords(gray_coords)
    if box is None:
        raise ValueError("missing gray frame")
    r0, c0, r1, c1 = box
    inner = _crop_bbox(patch, (r0 + 1, c0 + 1, r1 - 1, c1 - 1))
    sprite_colors = sorted(set(map(int, np.unique(inner))) - {BACKGROUND, FRAME_COLOR})
    if len(sprite_colors) != 1:
        raise ValueError("expected exactly one sprite color inside the frame")
    color = int(sprite_colors[0])
    sprite_coords = np.argwhere(patch == color)
    if sprite_coords.size == 0:
        raise ValueError("empty sprite")
    return _centered_mask(sprite_coords, patch.shape[0]), color


def _paint_relative(points: Iterable[tuple[int, int]], q: int, color_values: list[int]) -> np.ndarray:
    center = q // 2
    arr = np.zeros((q, q), dtype=int)
    for (dr, dc), value in zip(points, color_values):
        arr[center + int(dr), center + int(dc)] = int(value)
    return arr


def _paint_binary(points: Iterable[tuple[int, int]], q: int) -> np.ndarray:
    center = q // 2
    arr = np.zeros((q, q), dtype=bool)
    for dr, dc in points:
        arr[center + int(dr), center + int(dc)] = True
    return arr


def _transform_template(template: np.ndarray, quadrant: str) -> np.ndarray:
    if quadrant == "tr":
        return np.fliplr(template)
    if quadrant == "bl":
        return np.flipud(template)
    if quadrant == "br":
        return np.flipud(np.fliplr(template))
    raise ValueError(f"unsupported quadrant: {quadrant}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the template into three quadrants, then carve each copy with the boxed sprite mask and recolor it.

    Input structure:
    - A black grid with a 1-cell outer border and four equal inner quadrants.
    - The upper-left quadrant contains a sparse blue/red template on black.
    - Each other quadrant contains one centered gray frame with a single connected green, yellow, or magenta sprite inside.

    Rule:
    - Keep the upper-left template unchanged.
    - Reflect the template horizontally into the upper-right quadrant, vertically into the lower-left quadrant, and on both axes into the lower-right quadrant.
    - In each non-template quadrant, extract the boxed sprite's shape, align the sprite bbox center to the quadrant center, and use that centered shape as a keep-mask over the reflected template.
    - Retain only reflected template pixels under the centered mask, recolor every retained pixel to the sprite color, and clear the gray frame and all other pixels to black.

    Key insight:
    - The sprite contributes two things at once: its bbox-centered shape decides where reflected template pixels survive, and its own color decides how those surviving pixels are recolored.
    """
    g = np.asarray(grid, dtype=int)
    quads = _quadrant_slices(g.shape)
    template = g[quads["tl"]].copy()
    out = np.zeros_like(g)
    out[quads["tl"]] = template

    for name in ("tr", "bl", "br"):
        patch = g[quads[name]]
        centered_mask, color = _extract_sprite_mask_and_color(patch)
        transformed = _transform_template(template, name)
        keep = centered_mask & (transformed != BACKGROUND)
        out_patch = np.zeros_like(transformed)
        out_patch[keep] = int(color)
        out[quads[name]] = out_patch

    return out


_TEMPLATE_CORE_BANK: tuple[tuple[tuple[int, int], ...], ...] = (
    ((-1, -1), (-1, 0), (0, 0), (1, 0), (1, 1)),
    ((-1, 1), (-1, 0), (0, 0), (1, 0), (1, -1)),
    ((-1, -1), (0, -1), (0, 0), (0, 1), (1, 1)),
    ((-1, 1), (0, -1), (0, 0), (0, 1), (1, -1)),
    ((-1, 0), (0, -1), (0, 0), (1, 0), (1, 1)),
    ((-1, 0), (0, 1), (0, 0), (1, 0), (1, -1)),
)

_RING_POINTS: tuple[tuple[int, int], ...] = (
    (-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1),
)

_SPRITE_BANK: tuple[dict[str, object], ...] = (
    {"name": "line_h", "coords": ((0, -1), (0, 0), (0, 1)), "shifts": ((-1, 0), (0, 0), (1, 0))},
    {"name": "line_v", "coords": ((-1, 0), (0, 0), (1, 0)), "shifts": ((0, -1), (0, 0), (0, 1))},
    {"name": "plus", "coords": ((-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)), "shifts": ((0, 0),)},
    {"name": "zig_a", "coords": ((-1, -1), (-1, 0), (0, 0), (1, 0), (1, 1)), "shifts": ((0, 0),)},
    {"name": "zig_b", "coords": ((-1, 1), (-1, 0), (0, 0), (1, 0), (1, -1)), "shifts": ((0, 0),)},
    {"name": "hook_a", "coords": ((-1, -1), (0, -1), (0, 0), (0, 1), (1, 1)), "shifts": ((0, 0),)},
    {"name": "hook_b", "coords": ((-1, 1), (0, -1), (0, 0), (0, 1), (1, -1)), "shifts": ((0, 0),)},
)


def _shape_to_centered_mask(spec: dict[str, object], shift: tuple[int, int], q: int) -> tuple[np.ndarray, np.ndarray]:
    frame_top = (q - 5) // 2
    frame_left = (q - 5) // 2
    sprite_coords = []
    for dr, dc in spec["coords"]:
        rr = frame_top + 2 + int(dr) + int(shift[0])
        cc = frame_left + 2 + int(dc) + int(shift[1])
        sprite_coords.append((rr, cc))
    coords_arr = np.array(sprite_coords, dtype=int)
    return coords_arr, _centered_mask(coords_arr, q)


def _build_template(q: int) -> tuple[np.ndarray, np.ndarray]:
    base = list(random.choice(_TEMPLATE_CORE_BANK))
    extras_count = random.choice((1, 2))
    extras = random.sample(_RING_POINTS, k=extras_count)
    rel_points = tuple(base + extras)
    binary = _paint_binary(rel_points, q)
    color_values = [random.choice(TEMPLATE_COLORS) for _ in rel_points]
    if len(set(color_values)) == 1:
        flip_idx = random.randrange(len(color_values))
        color_values[flip_idx] = TEMPLATE_COLORS[0] if color_values[flip_idx] == TEMPLATE_COLORS[1] else TEMPLATE_COLORS[1]
    template = _paint_relative(rel_points, q, color_values)
    return template, binary


def _binary_signature(arr: np.ndarray) -> bytes:
    return np.packbits(arr.astype(np.uint8)).tobytes() + str(arr.shape).encode()


def _build_expected(template: np.ndarray, sprite_info: dict[str, tuple[np.ndarray, int]], size: int) -> np.ndarray:
    q = template.shape[0]
    out = np.zeros((size, size), dtype=int)
    quads = _quadrant_slices((size, size))
    out[quads["tl"]] = template
    for name in ("tr", "bl", "br"):
        centered_mask, color = sprite_info[name]
        transformed = _transform_template(template, name)
        keep = centered_mask & (transformed != BACKGROUND)
        patch = np.zeros((q, q), dtype=int)
        patch[keep] = int(color)
        out[quads[name]] = patch
    return out


def _semantic_checks(
    template_binary: np.ndarray,
    sprite_specs: dict[str, tuple[str, np.ndarray]],
    centered_masks: dict[str, np.ndarray],
    expected_binary: dict[str, np.ndarray],
) -> bool:
    template_variants = {
        "tl": template_binary,
        "tr": np.fliplr(template_binary),
        "bl": np.flipud(template_binary),
        "br": np.flipud(np.fliplr(template_binary)),
    }
    if len({_binary_signature(v) for v in template_variants.values()}) != 4:
        return False

    template_area = int(template_binary.sum())
    if template_area < 6 or template_area > 7:
        return False

    if len({name for name, _mask in sprite_specs.values()}) != 3:
        return False

    if len({_binary_signature(mask) for mask in centered_masks.values()}) != 3:
        return False

    if len({_binary_signature(arr) for arr in expected_binary.values()}) != 3:
        return False

    for name in ("tr", "bl", "br"):
        actual = expected_binary[name]
        centered = centered_masks[name]
        if int(actual.sum()) < 2:
            return False
        if int(actual.sum()) * 10 < 3 * template_area:
            return False
        if np.array_equal(actual, centered):
            return False
        for other_name, other_template in template_variants.items():
            if other_name == name:
                continue
            if np.array_equal(actual, centered & other_template):
                return False
    return True


def _gen_attempt() -> np.ndarray | None:
    q = random.choice((7, 9))
    size = 2 * q + 2
    template, template_binary = _build_template(q)

    sprite_specs_raw = random.sample(list(_SPRITE_BANK), k=3)
    sprite_specs: dict[str, tuple[str, np.ndarray]] = {}
    centered_masks: dict[str, np.ndarray] = {}
    sprite_draw_coords: dict[str, np.ndarray] = {}
    sprite_info: dict[str, tuple[np.ndarray, int]] = {}
    expected_binary: dict[str, np.ndarray] = {}

    for quadrant, spec, color in zip(("tr", "bl", "br"), sprite_specs_raw, random.sample(SPRITE_COLORS, k=3)):
        shift = random.choice(spec["shifts"])
        coords_arr, centered_mask = _shape_to_centered_mask(spec, shift, q)
        transformed_binary = _transform_template(template_binary.astype(int), quadrant).astype(bool)
        actual = centered_mask & transformed_binary
        sprite_specs[quadrant] = (str(spec["name"]), centered_mask)
        centered_masks[quadrant] = centered_mask
        sprite_draw_coords[quadrant] = coords_arr
        sprite_info[quadrant] = (centered_mask, int(color))
        expected_binary[quadrant] = actual

    if not _semantic_checks(template_binary, sprite_specs, centered_masks, expected_binary):
        return None

    grid = np.zeros((size, size), dtype=int)
    quads = _quadrant_slices(grid.shape)
    grid[quads["tl"]] = template

    frame_top = (q - 5) // 2
    frame_left = (q - 5) // 2
    frame = np.full((5, 5), FRAME_COLOR, dtype=int)
    frame[1:-1, 1:-1] = BACKGROUND

    for quadrant, color in zip(("tr", "bl", "br"), (sprite_info["tr"][1], sprite_info["bl"][1], sprite_info["br"][1])):
        patch = np.zeros((q, q), dtype=int)
        patch[frame_top : frame_top + 5, frame_left : frame_left + 5] = frame
        coords_arr = sprite_draw_coords[quadrant]
        for r, c in coords_arr:
            patch[int(r), int(c)] = int(color)
        grid[quads[quadrant]] = patch

    expected = _build_expected(template, sprite_info, size)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(2000):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
