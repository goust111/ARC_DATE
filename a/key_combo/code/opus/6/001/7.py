from __future__ import annotations

from collections import deque
import random

import numpy as np


BG_COLOR = 0
OBJECT_COLOR = 5
PALETTE_COLORS = (1, 2, 3, 4)

AREA_MIN, AREA_MAX = 8, 20
GRID_H_RANGE = (18, 24)
GRID_W_RANGE = (18, 26)
OBJECT_COUNT_RANGE = (3, 5)
MAX_GENERATION_ATTEMPTS = 1200

_SEEN_INPUT_SIGNATURES: set[tuple[tuple[int, int], bytes]] = set()
_SEEN_OUTPUT_SIGNATURES: set[tuple[tuple[int, int], bytes]] = set()


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def neighbors4(r: int, c: int, shape: tuple[int, int]):
    h, w = shape
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc, mask.shape):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(np.array(coords, dtype=int))

    return components


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return bbox_coords(coords)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, box)
    return np.asarray(grid, dtype=int)[r0 : r1 + 1, c0 : c1 + 1].copy()


def bbox_w(box: tuple[int, int, int, int]) -> int:
    return int(box[3] - box[1] + 1)


def row_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    if a[2] < b[0]:
        return int(b[0] - a[2] - 1)
    if b[2] < a[0]:
        return int(a[0] - b[2] - 1)
    return 0


def bbox_far(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    pad: int = 1,
) -> bool:
    return (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _signature(grid: np.ndarray) -> tuple[tuple[int, int], bytes]:
    arr = np.asarray(grid, dtype=int)
    return tuple(map(int, arr.shape)), arr.tobytes()


def _is_point_symmetric(mask: np.ndarray) -> bool:
    return bool(np.array_equal(mask, mask[::-1, ::-1]))


def _is_connected(mask: np.ndarray) -> bool:
    return bool(mask.any()) and len(components_mask(mask)) == 1


def _local_mask_from_coords(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    box = bbox_coords(coords)
    h = int(box[2] - box[0] + 1)
    w = int(box[3] - box[1] + 1)
    local = np.zeros((h, w), dtype=bool)
    rel = coords - np.array([box[0], box[1]])
    local[rel[:, 0], rel[:, 1]] = True
    return local, box


def _extract_palette(g: np.ndarray, bg: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    palette_mask = (g != int(bg)) & (g != OBJECT_COLOR)
    palette_components = components_mask(palette_mask)
    assert len(palette_components) == 1

    palette_box = bbox_coords(palette_components[0])
    assert palette_box[0] == palette_box[2]
    assert palette_box[0] in {0, g.shape[0] - 1}

    palette = crop_bbox(g, palette_box)[0]
    assert palette.size == palette_components[0].shape[0]
    assert 2 <= len(set(map(int, palette.tolist()))) <= 4
    assert bool(np.all(np.isin(palette, PALETTE_COLORS)))
    return palette, palette_box


def _analyze_instance(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    palette, palette_box = _extract_palette(g, bg)

    components = []
    for coords in components_mask(g == OBJECT_COLOR):
        local, box = _local_mask_from_coords(coords)
        components.append(
            {
                "coords": coords,
                "bbox": box,
                "local": local,
                "area": int(local.sum()),
                "width": int(local.shape[1]),
                "symmetric": _is_point_symmetric(local),
            }
        )

    assert OBJECT_COUNT_RANGE[0] <= len(components) <= OBJECT_COUNT_RANGE[1]
    assert all(AREA_MIN <= int(comp["area"]) <= AREA_MAX for comp in components)

    symmetric = [comp for comp in components if bool(comp["symmetric"])]
    assert len(symmetric) == 1
    target = symmetric[0]
    assert int(target["width"]) == int(palette.size)

    return {
        "grid": g,
        "bg": int(bg),
        "palette": palette,
        "palette_box": palette_box,
        "components": components,
        "target": target,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the unique 180-degree symmetric grey object, crop it, and recolor its columns from the palette strip.

    Input structure:
    - The background is black, with 3 to 5 grey 4-connected foreground objects placed apart from one another.
    - Exactly one grey object is invariant under a 180-degree rotation within its own bounding box.
    - A separate 1-row palette strip of non-grey colors sits on the top or bottom edge, and its width matches the symmetric object's bbox width.

    Rule:
    - Test each grey object for 180-degree rotational symmetry by comparing its bbox-local mask with its 180-degree rotation.
    - Select the unique symmetric object and crop its bounding box.
    - Recolor each grey pixel in the crop using the palette color from the same output column, while leaving bbox-local background holes black.

    Key insight:
    - Point symmetry is not used to create a new copy here; it is the discriminator that identifies which object should be cropped before the column-wise recoloring rule is applied.
    """
    analysis = _analyze_instance(np.asarray(grid, dtype=int))
    palette = np.asarray(analysis["palette"], dtype=int)
    target_mask = np.asarray(analysis["target"]["local"], dtype=bool)

    out = np.zeros(target_mask.shape, dtype=int)
    for col, color in enumerate(palette):
        out[target_mask[:, col], col] = int(color)
    return out


def _orbit_groups(height: int, width: int) -> list[list[tuple[int, int]]]:
    seen: set[tuple[tuple[int, int], ...]] = set()
    groups: list[list[tuple[int, int]]] = []
    for r in range(height):
        for c in range(width):
            orbit = tuple(sorted({(r, c), (height - 1 - r, width - 1 - c)}))
            if orbit not in seen:
                seen.add(orbit)
                groups.append(list(orbit))
    return groups


def _make_symmetric_mask(width: int | None = None) -> np.ndarray | None:
    for _ in range(400):
        h = random.randint(3, 6)
        w = random.randint(4, 7) if width is None else int(width)
        if h * w < AREA_MIN:
            continue

        desired_area = random.randint(AREA_MIN, min(AREA_MAX, h * w))
        mask = np.ones((h, w), dtype=bool)
        groups = _orbit_groups(h, w)
        removed = 0
        changed = True

        while int(mask.sum()) > desired_area and changed:
            changed = False
            random.shuffle(groups)
            for group in groups:
                group_area = sum(int(mask[r, c]) for r, c in group)
                if group_area == 0 or int(mask.sum()) - group_area < desired_area:
                    continue
                candidate = mask.copy()
                for r, c in group:
                    candidate[r, c] = False
                if bbox(candidate) != (0, 0, h - 1, w - 1):
                    continue
                if not _is_connected(candidate):
                    continue
                mask = candidate
                removed += group_area
                changed = True
                if int(mask.sum()) == desired_area:
                    break

        if (
            AREA_MIN <= int(mask.sum()) <= AREA_MAX
            and removed > 0
            and _is_connected(mask)
            and _is_point_symmetric(mask)
        ):
            return mask
    return None


def _make_nonsymmetric_mask(
    width: int | None = None,
    area_min: int = AREA_MIN,
    area_max: int = AREA_MAX,
) -> np.ndarray | None:
    for _ in range(600):
        w = random.randint(3, 7) if width is None else int(width)
        heights = [
            h
            for h in range(3, 7)
            if h * w >= int(area_min) and h * w <= max(int(area_max) + 4, int(area_min) + 1)
        ]
        if not heights:
            continue

        h = random.choice(heights)
        desired_area = random.randint(int(area_min), min(int(area_max), h * w))
        mask = np.ones((h, w), dtype=bool)
        cells = [(r, c) for r in range(h) for c in range(w)]
        removed = 0
        changed = True

        while int(mask.sum()) > desired_area and changed:
            changed = False
            random.shuffle(cells)
            for r, c in cells:
                if not mask[r, c] or int(mask.sum()) - 1 < desired_area:
                    continue
                candidate = mask.copy()
                candidate[r, c] = False
                if bbox(candidate) != (0, 0, h - 1, w - 1):
                    continue
                if not _is_connected(candidate):
                    continue
                mask = candidate
                removed += 1
                changed = True
                if int(mask.sum()) == desired_area:
                    break

        if (
            AREA_MIN <= int(mask.sum()) <= AREA_MAX
            and removed > 0
            and _is_connected(mask)
            and not _is_point_symmetric(mask)
        ):
            return mask
    return None


def _make_palette(width: int) -> np.ndarray:
    distinct_count = random.randint(2, min(4, int(width)))
    colors = random.sample(PALETTE_COLORS, distinct_count)
    palette = np.array([random.choice(colors) for _ in range(int(width))], dtype=int)
    for color in colors:
        palette[random.randrange(int(width))] = int(color)
    random.shuffle(palette)
    return palette


def _place_bbox(
    blocked: list[tuple[int, int, int, int]],
    grid_h: int,
    grid_w: int,
    patch_h: int,
    patch_w: int,
    row_band: tuple[int, int] | None = None,
) -> tuple[int, int, tuple[int, int, int, int]] | None:
    if row_band is None:
        row_lo, row_hi = 0, grid_h - patch_h
    else:
        row_lo = max(0, int(row_band[0]))
        row_hi = min(grid_h - patch_h, int(row_band[1]))
    if row_lo > row_hi:
        return None

    candidates = []
    for r0 in range(row_lo, row_hi + 1):
        for c0 in range(0, grid_w - patch_w + 1):
            box = (r0, c0, r0 + patch_h - 1, c0 + patch_w - 1)
            if all(bbox_far(box, other, pad=1) for other in blocked):
                candidates.append((r0, c0, box))
    if not candidates:
        return None
    return random.choice(candidates)


def _paint_patch(grid: np.ndarray, patch_mask: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch_mask.shape
    view = grid[r0 : r0 + h, c0 : c0 + w]
    view[patch_mask] = OBJECT_COLOR


def _column_recolor(mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    out = np.zeros(mask.shape, dtype=int)
    for col, color in enumerate(np.asarray(palette, dtype=int)):
        out[mask[:, col], col] = int(color)
    return out


def _validate_instance(grid: np.ndarray, expected: np.ndarray) -> bool:
    analysis = _analyze_instance(grid)
    palette_box = analysis["palette_box"]
    target = analysis["target"]
    distractors = [comp for comp in analysis["components"] if comp is not target]

    assert distractors
    assert any(int(comp["width"]) == int(target["width"]) for comp in distractors)
    assert any(int(comp["width"]) != int(target["width"]) for comp in distractors)
    assert max(int(comp["area"]) for comp in distractors) >= int(target["area"])
    assert any(row_gap(comp["bbox"], palette_box) < row_gap(target["bbox"], palette_box) for comp in distractors)
    assert np.array_equal(main(grid), expected)
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid_h = random.randint(*GRID_H_RANGE)
        grid_w = random.randint(*GRID_W_RANGE)

        target_mask = _make_symmetric_mask()
        if target_mask is None:
            continue
        target_h, target_w = map(int, target_mask.shape)
        target_area = int(target_mask.sum())

        special_mask = _make_nonsymmetric_mask(width=target_w, area_min=target_area, area_max=AREA_MAX)
        if special_mask is None:
            continue

        object_count = random.randint(*OBJECT_COUNT_RANGE)
        object_masks = [target_mask, special_mask]
        while len(object_masks) < object_count:
            same_width = random.random() < 0.35
            extra = _make_nonsymmetric_mask(width=target_w if same_width else None)
            if extra is None:
                break
            object_masks.append(extra)
        if len(object_masks) != object_count:
            continue
        if all(mask.shape[1] == target_w for mask in object_masks[2:]):
            continue

        palette = _make_palette(target_w)
        palette_top = random.choice([True, False])
        palette_row = 0 if palette_top else grid_h - 1
        palette_col = random.randint(0, grid_w - target_w)
        palette_box = (palette_row, palette_col, palette_row, palette_col + target_w - 1)

        blocked = [palette_box]
        if palette_top:
            target_band = (grid_h // 2, grid_h - target_h - 2)
            special_band = (2, grid_h // 2 - special_mask.shape[0])
        else:
            target_band = (0, grid_h // 2 - target_h)
            special_band = (grid_h // 2, grid_h - special_mask.shape[0] - 3)

        target_pos = _place_bbox(blocked, grid_h, grid_w, target_h, target_w, target_band)
        if target_pos is None:
            continue
        blocked.append(target_pos[2])

        special_pos = _place_bbox(
            blocked,
            grid_h,
            grid_w,
            int(special_mask.shape[0]),
            int(special_mask.shape[1]),
            special_band,
        )
        if special_pos is None:
            continue
        blocked.append(special_pos[2])

        placements = [
            (target_mask, int(target_pos[0]), int(target_pos[1])),
            (special_mask, int(special_pos[0]), int(special_pos[1])),
        ]

        failed = False
        for mask in object_masks[2:]:
            placed = _place_bbox(
                blocked,
                grid_h,
                grid_w,
                int(mask.shape[0]),
                int(mask.shape[1]),
            )
            if placed is None:
                failed = True
                break
            blocked.append(placed[2])
            placements.append((mask, int(placed[0]), int(placed[1])))
        if failed:
            continue

        grid = np.zeros((grid_h, grid_w), dtype=int)
        grid[palette_row, palette_col : palette_col + target_w] = palette
        for mask, r0, c0 in placements:
            _paint_patch(grid, mask, r0, c0)

        expected = _column_recolor(target_mask, palette)
        try:
            _validate_instance(grid, expected)
        except AssertionError:
            continue

        input_signature = _signature(grid)
        output_signature = _signature(expected)
        if input_signature in _SEEN_INPUT_SIGNATURES or output_signature in _SEEN_OUTPUT_SIGNATURES:
            continue
        _SEEN_INPUT_SIGNATURES.add(input_signature)
        _SEEN_OUTPUT_SIGNATURES.add(output_signature)
        return grid

    raise RuntimeError("failed to generate a valid unique puzzle instance")
