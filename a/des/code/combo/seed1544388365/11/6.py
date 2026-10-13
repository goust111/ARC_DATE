from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE_TRANSFORMS = ("identity", "hflip", "vflip", "rot_cw", "rot_ccw")
ALL_COLORS = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_TRIES = 250
SHAPE_TRIES = 160


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _mask_from_coords(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return patch, (r0, c0, r1, c1)


def _apply_transform(mask: np.ndarray, transform: str) -> np.ndarray:
    if transform == "identity":
        return mask.copy()
    if transform == "hflip":
        return np.fliplr(mask)
    if transform == "vflip":
        return np.flipud(mask)
    if transform == "rot_cw":
        return np.rot90(mask, k=3)
    if transform == "rot_ccw":
        return np.rot90(mask, k=1)
    raise AssertionError(f"unknown transform {transform}")


def _parse_palette(g: np.ndarray) -> dict[int, str]:
    row = g[-1]
    cols = np.where(row != BG)[0]
    assert 1 <= len(cols) <= len(PALETTE_TRANSFORMS), "palette must contain 1-5 colors"
    if len(cols) > 1:
        assert bool(np.all(np.diff(cols) >= 2)), "palette colors must be separated by black cells"
    colors = [int(row[c]) for c in cols]
    assert len(colors) == len(set(colors)), "palette colors must be unique"
    return {color: PALETTE_TRANSFORMS[i] for i, color in enumerate(colors)}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Apply a palette-coded geometric transform to each object and restore the instruction pixel to the base color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2 and g.shape[1] >= 3
    assert int(g[-1].min()) >= 0 and int(g[-1].max()) <= 9

    palette_map = _parse_palette(g)
    object_mask = g[:-1] != BG
    components = _connected_components(object_mask)
    assert components, "expected at least one object"

    out = np.zeros_like(g)
    out[-1, :] = g[-1, :]
    occupied = np.zeros_like(object_mask, dtype=bool)

    for coords in components:
        colors = [int(g[r, c]) for r, c in coords]
        instruction_indices = [i for i, color in enumerate(colors) if color in palette_map]
        assert len(instruction_indices) == 1, "each object must contain exactly one instruction pixel"
        instruction_color = colors[instruction_indices[0]]
        transform = palette_map[instruction_color]

        base_colors = [color for color in colors if color != instruction_color]
        assert base_colors, "object must retain at least one base-color pixel"
        base_color = base_colors[0]
        assert all(color == base_color for color in base_colors), "object must be monochrome aside from instruction"

        patch, (r0, c0, r1, c1) = _mask_from_coords(coords)
        transformed = _apply_transform(patch, transform)
        new_h, new_w = transformed.shape

        top = (r0 + r1 - (new_h - 1)) // 2
        left = (c0 + c1 - (new_w - 1)) // 2

        for rr, cc in np.argwhere(transformed):
            r = int(top + rr)
            c = int(left + cc)
            assert 0 <= r < g.shape[0] - 1 and 0 <= c < g.shape[1], "transformed object must stay above palette row"
            assert not occupied[r, c], "transformed objects must not overlap"
            occupied[r, c] = True
            out[r, c] = base_color

    return out


def _frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    frontier: set[tuple[int, int]] = set()
    for r, c in np.argwhere(mask):
        for nr, nc in _neighbors4(int(r), int(c)):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                frontier.add((nr, nc))
    return list(frontier)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1].copy()


def _random_shape() -> np.ndarray | None:
    for _ in range(SHAPE_TRIES):
        raw_h = random.randint(3, 7)
        raw_w = random.randint(3, 7)
        mask = np.zeros((raw_h, raw_w), dtype=bool)
        sr = random.randrange(raw_h)
        sc = random.randrange(raw_w)
        mask[sr, sc] = True

        min_area = max(4, (raw_h * raw_w) // 3)
        max_area = raw_h * raw_w - 1
        if min_area > max_area:
            continue
        target_area = random.randint(min_area, max_area)

        while int(mask.sum()) < target_area:
            cells = _frontier(mask)
            if not cells:
                break
            r, c = random.choice(cells)
            mask[r, c] = True

        cropped = _crop_mask(mask)
        h, w = cropped.shape
        if h < 2 or w < 2:
            continue
        if h % 2 != w % 2:
            continue
        if int(cropped.sum()) < 4:
            continue
        if bool(cropped.all()) and random.random() < 0.8:
            continue
        return cropped
    return None


def _shape_for_transform(transform: str) -> np.ndarray | None:
    for _ in range(SHAPE_TRIES):
        shape = _random_shape()
        if shape is None:
            continue
        transformed = _apply_transform(shape, transform)
        if transform != "identity" and np.array_equal(shape, transformed):
            continue
        return shape
    return None


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _find_placement(
    shape_h: int,
    shape_w: int,
    patch: np.ndarray,
    transformed: np.ndarray,
    input_boxes: list[tuple[int, int, int, int]],
    output_boxes: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    ph, pw = patch.shape
    th, tw = transformed.shape

    if ph > shape_h or pw > shape_w or th > shape_h or tw > shape_w:
        return None

    for _ in range(PLACEMENT_TRIES):
        r0 = random.randint(0, shape_h - ph)
        c0 = random.randint(0, shape_w - pw)
        out_r0 = r0 + (ph - th) // 2
        out_c0 = c0 + (pw - tw) // 2
        if out_r0 < 0 or out_c0 < 0 or out_r0 + th > shape_h or out_c0 + tw > shape_w:
            continue

        input_box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
        output_box = (out_r0, out_c0, out_r0 + th - 1, out_c0 + tw - 1)

        if any(_boxes_overlap(input_box, other, pad=1) for other in input_boxes):
            continue
        if any(_boxes_overlap(output_box, other, pad=1) for other in output_boxes):
            continue
        return r0, c0, out_r0, out_c0
    return None


def _object_records(base_color: int, palette_colors: list[int]) -> list[dict] | None:
    transforms = list(PALETTE_TRANSFORMS)
    extra_count = random.randint(0, 2)
    transforms.extend(random.choices(PALETTE_TRANSFORMS, k=extra_count))
    random.shuffle(transforms)

    records: list[dict] = []
    for transform in transforms:
        patch = _shape_for_transform(transform)
        if patch is None:
            return None
        cells = np.argwhere(patch)
        instr_r, instr_c = map(int, cells[random.randrange(len(cells))])
        instruction_color = palette_colors[PALETTE_TRANSFORMS.index(transform)]
        records.append(
            {
                "patch": patch,
                "transform": transform,
                "instruction_pos": (instr_r, instr_c),
                "instruction_color": instruction_color,
                "base_color": base_color,
            }
        )
    return records


def _build_candidate() -> np.ndarray | None:
    height = random.randint(16, 27)
    width = random.randint(18, 30)
    shape_h = height - 2
    if shape_h < 8:
        return None

    sampled = random.sample(ALL_COLORS, 6)
    palette_colors = sampled[:5]
    base_color = sampled[5]

    records = _object_records(base_color, palette_colors)
    if records is None:
        return None

    input_boxes: list[tuple[int, int, int, int]] = []
    output_boxes: list[tuple[int, int, int, int]] = []

    for record in records:
        transformed = _apply_transform(record["patch"], record["transform"])
        placement = _find_placement(shape_h, width, record["patch"], transformed, input_boxes, output_boxes)
        if placement is None:
            return None
        r0, c0, out_r0, out_c0 = placement
        record["top_left"] = (r0, c0)
        record["out_top_left"] = (out_r0, out_c0)
        ph, pw = record["patch"].shape
        th, tw = transformed.shape
        input_boxes.append((r0, c0, r0 + ph - 1, c0 + pw - 1))
        output_boxes.append((out_r0, out_c0, out_r0 + th - 1, out_c0 + tw - 1))

    grid = np.zeros((height, width), dtype=int)

    palette_span = 2 * len(palette_colors) - 1
    start_col = random.randint(0, width - palette_span)
    for i, color in enumerate(palette_colors):
        grid[-1, start_col + 2 * i] = color

    for record in records:
        patch = record["patch"]
        r0, c0 = record["top_left"]
        grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]][patch] = record["base_color"]
        instr_r, instr_c = record["instruction_pos"]
        grid[r0 + instr_r, c0 + instr_c] = record["instruction_color"]

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    if not np.array_equal(out[-1], grid[-1]):
        return None
    if len(_connected_components(out[:-1] != BG)) != len(records):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
