from __future__ import annotations

import random
from collections import Counter

import numpy as np


ORTHO4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ORTHO4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _monochrome_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    comps: list[dict[str, object]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for coords in _components_from_mask(grid == color):
            bbox = _bbox(coords)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": bbox,
                    "area": int(len(coords)),
                }
            )
    return comps


def _rect_size(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _extract_cavities(
    crop: np.ndarray,
    bg: int,
) -> list[dict[str, object]]:
    cavities: list[dict[str, object]] = []
    for coords in _components_from_mask(crop == bg):
        if np.any(coords[:, 0] == 0) or np.any(coords[:, 0] == crop.shape[0] - 1):
            continue
        if np.any(coords[:, 1] == 0) or np.any(coords[:, 1] == crop.shape[1] - 1):
            continue
        bbox = _bbox(coords)
        h, w = _rect_size(bbox)
        if len(coords) != h * w:
            raise AssertionError("template cavities must be solid rectangles")
        cavities.append({"bbox": bbox, "size": (h, w), "coords": coords})
    return cavities


def _component_patch(comp: dict[str, object], bg: int) -> tuple[np.ndarray, np.ndarray]:
    bbox = comp["bbox"]
    color = int(comp["color"])
    coords = np.asarray(comp["coords"], dtype=int)
    r0, c0, r1, c1 = bbox
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    rel = coords - np.array([r0, c0], dtype=int)
    patch[rel[:, 0], rel[:, 1]] = color
    return patch, patch == color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each rectangular cavity inside the largest monochrome template with the
    outside sprite whose bounding box has the same size, then crop to the template.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    comps = _monochrome_components(g, bg)
    if not comps:
        return g.copy()

    template_idx = max(range(len(comps)), key=lambda i: int(comps[i]["area"]))
    template = comps[template_idx]
    template_bbox = template["bbox"]

    out = _crop_bbox(g, template_bbox).copy()
    cavities = _extract_cavities(out, bg)

    sprites_by_size: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for idx, comp in enumerate(comps):
        if idx == template_idx:
            continue
        size = _rect_size(comp["bbox"])
        if size in sprites_by_size:
            raise AssertionError("sprite bbox sizes must identify a unique match")
        sprites_by_size[size] = _component_patch(comp, bg)

    for cavity in cavities:
        size = cavity["size"]
        if size not in sprites_by_size:
            raise AssertionError("every cavity must have a matching outside sprite")
        patch, mask = sprites_by_size[size]
        r0, c0, r1, c1 = cavity["bbox"]
        region = out[r0 : r1 + 1, c0 : c1 + 1]
        region[mask] = patch[mask]

    return out


GEN_RETRIES = 300
HOLE_COUNT_RANGE = (2, 4)
HOLE_HEIGHT_RANGE = (2, 5)
HOLE_WIDTH_RANGE = (2, 5)
FRAME_EXTRA_H_RANGE = (5, 10)
FRAME_EXTRA_W_RANGE = (6, 12)
GRID_EXTRA_H_RANGE = (9, 15)
GRID_EXTRA_W_RANGE = (10, 18)
PLACEMENT_TRIES = 250
SPRITE_SOLID_PROB = 0.2
SPRITE_EXTRA_FILL_RANGE = (0, 5)


def _random_path(
    start: tuple[int, int],
    end: tuple[int, int],
    vertical_first: bool,
) -> list[tuple[int, int]]:
    r, c = start
    tr, tc = end
    path = [(r, c)]
    axes = ((tr - r, 0), (0, tc - c))
    if not vertical_first:
        axes = axes[::-1]
    for dr_total, dc_total in axes:
        if dr_total:
            step = 1 if dr_total > 0 else -1
            for _ in range(abs(dr_total)):
                r += step
                path.append((r, c))
        if dc_total:
            step = 1 if dc_total > 0 else -1
            for _ in range(abs(dc_total)):
                c += step
                path.append((r, c))
    return path


def _make_sprite_mask(height: int, width: int) -> np.ndarray:
    if random.random() < SPRITE_SOLID_PROB:
        return np.ones((height, width), dtype=bool)

    mask = np.zeros((height, width), dtype=bool)
    center = (random.randrange(height), random.randrange(width))
    anchors = {
        (0, random.randrange(width)),
        (height - 1, random.randrange(width)),
        (random.randrange(height), 0),
        (random.randrange(height), width - 1),
        center,
    }
    for anchor in anchors:
        for rr, cc in _random_path(center, anchor, vertical_first=bool(random.getrandbits(1))):
            mask[rr, cc] = True

    extra_steps = random.randint(*SPRITE_EXTRA_FILL_RANGE)
    filled = np.argwhere(mask)
    for _ in range(extra_steps):
        if len(filled) == 0:
            break
        rr, cc = map(int, filled[random.randrange(len(filled))])
        neighbors = []
        for dr, dc in ORTHO4:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                neighbors.append((nr, nc))
        if not neighbors:
            continue
        nr, nc = neighbors[random.randrange(len(neighbors))]
        mask[nr, nc] = True
        filled = np.argwhere(mask)

    assert np.any(mask[0, :]) and np.any(mask[-1, :])
    assert np.any(mask[:, 0]) and np.any(mask[:, -1])
    assert len(_components_from_mask(mask)) == 1
    return mask


def _choose_sizes(count: int) -> list[tuple[int, int]]:
    candidates = [
        (h, w)
        for h in range(HOLE_HEIGHT_RANGE[0], HOLE_HEIGHT_RANGE[1] + 1)
        for w in range(HOLE_WIDTH_RANGE[0], HOLE_WIDTH_RANGE[1] + 1)
    ]
    random.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    used_areas: set[int] = set()
    for h, w in candidates:
        area = h * w
        if area in used_areas:
            continue
        chosen.append((h, w))
        used_areas.add(area)
        if len(chosen) == count:
            return chosen
    raise RuntimeError("failed to choose enough unique cavity sizes")


def _place_rectangles(frame_h: int, frame_w: int, sizes: list[tuple[int, int]]) -> list[tuple[int, int, int, int]] | None:
    reserved = np.zeros((frame_h, frame_w), dtype=bool)
    placed: list[tuple[int, int, int, int]] = []
    for height, width in sorted(sizes, key=lambda x: x[0] * x[1], reverse=True):
        found = False
        for _ in range(PLACEMENT_TRIES):
            top = random.randint(1, frame_h - height - 1)
            left = random.randint(1, frame_w - width - 1)
            r0 = max(0, top - 1)
            r1 = min(frame_h, top + height + 1)
            c0 = max(0, left - 1)
            c1 = min(frame_w, left + width + 1)
            if reserved[r0:r1, c0:c1].any():
                continue
            reserved[r0:r1, c0:c1] = True
            placed.append((top, left, height, width))
            found = True
            break
        if not found:
            return None
    return placed


def _reserve_box(mask: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    r0 = max(0, top - 1)
    r1 = min(mask.shape[0], top + height + 1)
    c0 = max(0, left - 1)
    c1 = min(mask.shape[1], left + width + 1)
    mask[r0:r1, c0:c1] = True


def _expected_output(
    frame_crop: np.ndarray,
    holes: list[tuple[int, int, int, int]],
    sprite_patches: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    out = frame_crop.copy()
    for top, left, height, width in holes:
        patch, patch_mask = sprite_patches[(height, width)]
        region = out[top : top + height, left : left + width]
        region[patch_mask] = patch[patch_mask]
    return out


def _generate_instance() -> tuple[np.ndarray, np.ndarray] | None:
    hole_count = random.randint(*HOLE_COUNT_RANGE)
    sizes = _choose_sizes(hole_count)
    max_h = max(h for h, _ in sizes)
    max_w = max(w for _, w in sizes)
    sum_h = sum(h for h, _ in sizes)
    sum_w = sum(w for _, w in sizes)

    frame_h_min = max(max_h + 2, max_h + 4, sum_h // 2 + 3)
    frame_w_min = max(max_w + 2, max_w + 4, sum_w // 2 + 4)
    frame_h = frame_h_min + random.randint(*FRAME_EXTRA_H_RANGE)
    frame_w = frame_w_min + random.randint(*FRAME_EXTRA_W_RANGE)

    holes = _place_rectangles(frame_h, frame_w, sizes)
    if holes is None:
        return None

    colors = random.sample(range(10), hole_count + 2)
    bg = colors[0]
    frame_color = colors[1]
    sprite_colors = colors[2:]

    frame_crop = np.full((frame_h, frame_w), frame_color, dtype=int)
    sprite_patches: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for (top, left, height, width), sprite_color in zip(holes, sprite_colors):
        frame_crop[top : top + height, left : left + width] = bg
        sprite_mask = _make_sprite_mask(height, width)
        sprite_patch = np.full((height, width), bg, dtype=int)
        sprite_patch[sprite_mask] = sprite_color
        sprite_patches[(height, width)] = (sprite_patch, sprite_mask)

    expected = _expected_output(frame_crop, holes, sprite_patches)

    grid_h = frame_h + random.randint(*GRID_EXTRA_H_RANGE)
    grid_w = frame_w + random.randint(*GRID_EXTRA_W_RANGE)
    if grid_h <= frame_h + 2 or grid_w <= frame_w + 2:
        return None

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    frame_top = random.randint(1, grid_h - frame_h - 1)
    frame_left = random.randint(1, grid_w - frame_w - 1)
    grid[frame_top : frame_top + frame_h, frame_left : frame_left + frame_w] = frame_crop

    reserved = np.zeros((grid_h, grid_w), dtype=bool)
    _reserve_box(reserved, frame_top, frame_left, frame_h, frame_w)

    for size in sizes:
        sprite_patch, sprite_mask = sprite_patches[size]
        height, width = size
        placed = False
        for _ in range(PLACEMENT_TRIES):
            top = random.randint(0, grid_h - height)
            left = random.randint(0, grid_w - width)
            if reserved[top : top + height, left : left + width].any():
                continue
            window = grid[top : top + height, left : left + width]
            window[sprite_mask] = sprite_patch[sprite_mask]
            _reserve_box(reserved, top, left, height, width)
            placed = True
            break
        if not placed:
            return None

    if _background_color(grid) != bg:
        return None

    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None

    comps = _monochrome_components(grid, bg)
    if not comps:
        return None
    areas = sorted((int(comp["area"]) for comp in comps), reverse=True)
    if len(areas) < 2 or areas[0] == areas[1]:
        return None

    template_idx = max(range(len(comps)), key=lambda i: int(comps[i]["area"]))
    size_counts = Counter(
        _rect_size(comp["bbox"]) for idx, comp in enumerate(comps) if idx != template_idx
    )
    if any(count != 1 for count in size_counts.values()):
        return None

    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _generate_instance()
        if result is not None:
            grid, _expected = result
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
