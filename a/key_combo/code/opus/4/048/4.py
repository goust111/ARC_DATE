from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    return out


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return _bbox_coords(coords)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_shape(mask: np.ndarray) -> np.ndarray:
    b = _bbox(mask)
    assert b is not None
    return _crop_bbox(mask, b)


def _coords_from_mask(mask: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, cell)) for cell in np.argwhere(mask)]


def _parse_regions(grid: np.ndarray):
    template_components = _components(grid == GREY)
    assert len(template_components) == 1, "expected exactly one grey template sprite"
    template_shape = _mask_shape(grid == GREY)

    region_components = _components((grid != BLACK) & (grid != GREY))
    assert len(region_components) >= 3, "expected 3-4 candidate regions"

    regions = []
    for coords in region_components:
        bbox = _bbox_coords(coords)
        crop = _crop_bbox(grid, bbox)
        assert not np.any(crop == GREY)
        assert not np.any(crop[0, :] == BLACK)
        assert not np.any(crop[-1, :] == BLACK)
        assert not np.any(crop[:, 0] == BLACK)
        assert not np.any(crop[:, -1] == BLACK)
        hole_mask = crop == BLACK
        hole_bbox = _bbox(hole_mask)
        assert hole_bbox is not None, "each candidate region must contain holes"
        hole_shape = _crop_bbox(hole_mask, hole_bbox)
        regions.append(
            {
                "bbox": bbox,
                "crop": crop,
                "hole_mask": hole_mask,
                "hole_shape": hole_shape,
            }
        )
    return template_shape, regions


def _infer_phase_colors(region: np.ndarray) -> dict[int, int]:
    colors = [int(x) for x in np.unique(region) if int(x) != BLACK]
    k = len(colors)
    assert 2 <= k <= 3, "expected a 2- or 3-color diagonal pattern"
    phase_to_color: dict[int, int] = {}
    for r, c in np.argwhere(region != BLACK):
        phase = int((int(r) + int(c)) % k)
        color = int(region[r, c])
        prev = phase_to_color.get(phase)
        if prev is None:
            phase_to_color[phase] = color
        else:
            assert prev == color, "visible cells must agree on the diagonal color phase"
    assert len(phase_to_color) == k, "each phase must remain visible after holes are removed"
    return phase_to_color


def _repair_region(region: np.ndarray) -> np.ndarray:
    out = np.array(region, copy=True)
    phase_to_color = _infer_phase_colors(out)
    k = len(phase_to_color)
    for r, c in np.argwhere(out == BLACK):
        out[r, c] = phase_to_color[(int(r) + int(c)) % k]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the rectangle whose black-hole layout matches the grey template and repair its diagonal stripe holes.

    Input structure:
    - The grid has a black background, 3-4 separated multicolor rectangular regions, and one grey template sprite.
    - Each rectangle contains a full diagonal-stripe pattern made from 2 or 3 non-black colors, with several interior black holes.
    - The grey template is a separate 4-connected shape whose cropped pixel layout matches exactly one region's cropped hole layout.

    Rule:
    - Extract the grey template shape in its own minimal bounding box.
    - For each colored rectangle, collect all black-hole pixels, crop them to their own minimal bounding box, and compare that boolean shape to the template.
    - Select the unique rectangle whose hole shape matches the template, crop that whole rectangle, and infer its diagonal color rule from the visible cells using local coordinates.
    - Fill every black hole in the crop with the color dictated by the rectangle's `(row + col) mod K` diagonal phase pattern, then return the repaired crop.

    Key insight:
    - The same black holes serve two jobs at once: their spatial arrangement identifies which rectangle to crop, and those very same cells are the missing pixels that must be reconstructed from the diagonal periodic pattern.
    """
    g = np.asarray(grid, dtype=int)
    template_shape, regions = _parse_regions(g)

    matches = [region for region in regions if np.array_equal(region["hole_shape"], template_shape)]
    assert len(matches) == 1, "expected exactly one region whose hole layout matches the template"
    return _repair_region(matches[0]["crop"])


def _is_connected(mask: np.ndarray) -> bool:
    cells = int(mask.sum())
    return cells > 0 and len(_components(mask)) == 1


def _random_mask_exact(
    height: int,
    width: int,
    area: int,
    *,
    require_connected: bool,
    require_full_bbox: bool,
    avoid: np.ndarray | None = None,
) -> np.ndarray | None:
    cells = [(r, c) for r in range(height) for c in range(width)]
    for _ in range(300):
        chosen = random.sample(cells, area)
        mask = np.zeros((height, width), dtype=bool)
        for r, c in chosen:
            mask[r, c] = True
        if require_full_bbox:
            if _bbox(mask) != (0, 0, height - 1, width - 1):
                continue
        if require_connected and not _is_connected(mask):
            continue
        if area == height * width:
            continue
        if avoid is not None and np.array_equal(mask, avoid):
            continue
        return mask
    return None


def _random_template_mask() -> np.ndarray:
    bbox_options = [(2, 2), (2, 3), (3, 2), (3, 3)]
    for _ in range(400):
        height, width = random.choice(bbox_options)
        max_area = min(6, height * width - 1)
        if max_area < 3:
            continue
        area = random.randint(3, max_area)
        mask = _random_mask_exact(
            height,
            width,
            area,
            require_connected=True,
            require_full_bbox=True,
        )
        if mask is not None:
            return mask
    raise RuntimeError("failed to generate a template mask")


def _make_pattern(height: int, width: int, colors: tuple[int, ...]) -> np.ndarray:
    k = len(colors)
    region = np.zeros((height, width), dtype=int)
    for r in range(height):
        for c in range(width):
            region[r, c] = colors[(r + c) % k]
    return region


def _embed_holes(region: np.ndarray, hole_shape: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    out = np.array(region, copy=True)
    hh, hw = hole_shape.shape
    top = random.randint(1, region.shape[0] - hh - 1)
    left = random.randint(1, region.shape[1] - hw - 1)
    for r, c in _coords_from_mask(hole_shape):
        out[top + r, left + c] = BLACK
    return out, (top, left)


def _pick_region_size(hole_shape: np.ndarray) -> tuple[int, int]:
    hh, hw = hole_shape.shape
    height = random.randint(max(5, hh + 2), 7)
    width = random.randint(max(5, hw + 2), 7)
    return height, width


def _can_place(occupied: np.ndarray, top: int, left: int, height: int, width: int, gap: int) -> bool:
    h, w = occupied.shape
    if top < 0 or left < 0 or top + height > h or left + width > w:
        return False
    r0 = max(0, top - gap)
    c0 = max(0, left - gap)
    r1 = min(h, top + height + gap)
    c1 = min(w, left + width + gap)
    return not bool(np.any(occupied[r0:r1, c0:c1]))


def _mark_box(occupied: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    occupied[top : top + height, left : left + width] = True


def _place_template(grid: np.ndarray, occupied: np.ndarray, template_shape: np.ndarray) -> tuple[int, int]:
    th, tw = template_shape.shape
    corners = [
        (0, 0),
        (0, grid.shape[1] - tw),
        (grid.shape[0] - th, 0),
        (grid.shape[0] - th, grid.shape[1] - tw),
    ]
    random.shuffle(corners)
    for top, left in corners:
        if _can_place(occupied, top, left, th, tw, 2):
            for r, c in _coords_from_mask(template_shape):
                grid[top + r, left + c] = GREY
            _mark_box(occupied, top, left, th, tw)
            return top, left
    raise RuntimeError("failed to place the template")


def _place_regions(
    grid: np.ndarray,
    occupied: np.ndarray,
    region_crops: list[np.ndarray],
) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    order = list(range(len(region_crops)))
    random.shuffle(order)
    temp_positions: list[tuple[int, int] | None] = [None] * len(region_crops)
    for idx in order:
        region = region_crops[idx]
        rh, rw = region.shape
        candidates = [
            (top, left)
            for top in range(grid.shape[0] - rh + 1)
            for left in range(grid.shape[1] - rw + 1)
            if _can_place(occupied, top, left, rh, rw, 2)
        ]
        if not candidates:
            raise RuntimeError("failed to place all regions")
        top, left = random.choice(candidates)
        grid[top : top + rh, left : left + rw] = region
        _mark_box(occupied, top, left, rh, rw)
        temp_positions[idx] = (top, left)
    for pos in temp_positions:
        assert pos is not None
        positions.append(pos)
    return positions


def _build_region_from_shape(hole_shape: np.ndarray, colors: tuple[int, ...]) -> np.ndarray:
    height, width = _pick_region_size(hole_shape)
    pattern = _make_pattern(height, width, colors)
    region, _ = _embed_holes(pattern, hole_shape)
    return region


def _target_and_distractors() -> tuple[np.ndarray, list[np.ndarray]]:
    target = _random_template_mask()
    area = int(target.sum())
    same_bbox = _random_mask_exact(
        target.shape[0],
        target.shape[1],
        area,
        require_connected=False,
        require_full_bbox=True,
        avoid=target,
    )
    if same_bbox is None:
        raise RuntimeError("failed to generate a same-bbox distractor")

    distractors = [same_bbox]
    for _ in range(random.randint(1, 2)):
        for _attempt in range(300):
            height, width = random.choice([(2, 2), (2, 3), (3, 2), (3, 3)])
            max_area = min(6, height * width - 1)
            if max_area < 3:
                continue
            area2 = random.randint(3, max_area)
            mask = _random_mask_exact(
                height,
                width,
                area2,
                require_connected=False,
                require_full_bbox=True,
            )
            if mask is None or np.array_equal(mask, target):
                continue
            distractors.append(mask)
            break
        else:
            raise RuntimeError("failed to generate distractor holes")
    return target, distractors


def _validate_instance(grid: np.ndarray, expected: np.ndarray) -> None:
    template_shape, regions = _parse_regions(grid)
    assert len(regions) in (3, 4)
    matches = [region for region in regions if np.array_equal(region["hole_shape"], template_shape)]
    assert len(matches) == 1

    hole_shapes = [region["hole_shape"] for region in regions]
    target_shape = matches[0]["hole_shape"]
    assert any(
        shape.shape == target_shape.shape
        and int(shape.sum()) == int(target_shape.sum())
        and not np.array_equal(shape, target_shape)
        for shape in hole_shapes
    ), "need a distractor with the same hole bbox and hole count"

    target_crop = matches[0]["crop"]
    visible_colors = [int(x) for x in np.unique(target_crop) if int(x) != BLACK]
    assert len(visible_colors) in (2, 3)
    phase_to_color = _infer_phase_colors(target_crop)
    assert len(phase_to_color) == len(visible_colors)
    assert np.array_equal(_repair_region(target_crop), expected)
    assert np.array_equal(main(grid), expected)


def generate_input() -> np.ndarray:
    for _ in range(400):
        try:
            grid_h = random.randint(18, 24)
            grid_w = random.randint(20, 26)
            grid = np.zeros((grid_h, grid_w), dtype=int)
            occupied = np.zeros((grid_h, grid_w), dtype=bool)

            k = random.randint(2, 3)
            colors = tuple(random.sample(PALETTE, k))
            target_shape, distractor_shapes = _target_and_distractors()

            region_count = 1 + len(distractor_shapes)
            target_index = random.randrange(region_count)
            hole_shapes = list(distractor_shapes)
            hole_shapes.insert(target_index, target_shape)

            region_crops = [_build_region_from_shape(shape, colors) for shape in hole_shapes]
            for region in region_crops:
                if np.any(region[0, :] == BLACK) or np.any(region[-1, :] == BLACK):
                    raise RuntimeError("holes leaked onto the boundary")
                if np.any(region[:, 0] == BLACK) or np.any(region[:, -1] == BLACK):
                    raise RuntimeError("holes leaked onto the boundary")
                if len(_components(region != BLACK)) != 1:
                    raise RuntimeError("region must remain connected")
                if len({int(region[r, c]) for r, c in np.argwhere(region != BLACK)}) != k:
                    raise RuntimeError("all diagonal phases must stay visible")

            expected = _make_pattern(*region_crops[target_index].shape, colors)
            _place_template(grid, occupied, target_shape)
            _place_regions(grid, occupied, region_crops)
            _validate_instance(grid, expected)
            return grid
        except (AssertionError, RuntimeError, ValueError):
            continue
    raise RuntimeError("failed to generate a valid ARC instance")
