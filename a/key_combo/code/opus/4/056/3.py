from __future__ import annotations

import random

import numpy as np


BG = 0
FRAME = 1
RED = 2
GREEN = 3
MARKER = 4

Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

_GEN_CORNER_INDEX = 0
_CORNER_TO_K = {
    "tl": 0,
    "tr": 1,
    "br": 2,
    "bl": 3,
}


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: BBox) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == int(color)
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
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
            out.append(np.array(coords, dtype=int))
    return out


def _is_rect_frame(grid: np.ndarray, coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        return False
    expected_area = 2 * (h + w) - 4
    if int(coords.shape[0]) != expected_area:
        return False
    patch = _crop_bbox(grid, (r0, c0, r1, c1))
    boundary = np.zeros_like(patch, dtype=bool)
    boundary[0, :] = True
    boundary[-1, :] = True
    boundary[:, 0] = True
    boundary[:, -1] = True
    return bool(np.all(patch[boundary] == FRAME) and np.all(patch[~boundary] != FRAME))


def _extract_frames(grid: np.ndarray) -> list[dict]:
    frames = []
    for coords in _components_of_color(grid, FRAME):
        if not _is_rect_frame(grid, coords):
            continue
        bbox = _bbox_coords(coords)
        r0, c0, r1, c1 = bbox
        frames.append(
            {
                "coords": coords,
                "bbox": bbox,
                "height": r1 - r0 + 1,
                "width": c1 - c0 + 1,
                "bbox_area": (r1 - r0 + 1) * (c1 - c0 + 1),
            }
        )
    return frames


def _marker_rotation_k(grid: np.ndarray, bbox: BBox) -> int:
    markers = np.argwhere(grid == MARKER)
    assert markers.shape == (1, 2), "expected one yellow marker pixel"
    mr, mc = map(int, markers[0])
    r0, c0, r1, c1 = bbox
    mapping = {
        (r0 - 1, c0 - 1): 0,
        (r0 - 1, c1 + 1): 1,
        (r1 + 1, c1 + 1): 2,
        (r1 + 1, c0 - 1): 3,
    }
    assert (mr, mc) in mapping, "marker must sit diagonally outside one corner of the largest frame"
    return mapping[(mr, mc)]


def _interior_bbox(frame_bbox: BBox) -> BBox:
    r0, c0, r1, c1 = frame_bbox
    return r0 + 1, c0 + 1, r1 - 1, c1 - 1


def main(grid: np.ndarray) -> np.ndarray:
    """Crop the largest rectangular frame interior, then rotate it according to the yellow corner marker.

    Input structure:
    - The background is black.
    - There are 3 or 4 disjoint blue one-cell-thick rectangular frames of different bbox areas.
    - Each frame contains a small red/green pattern inside it, and exactly one yellow pixel appears diagonally outside one corner of the largest frame.

    Rule:
    - Find the unique blue frame whose bounding-box area is largest.
    - Crop only that frame's interior, excluding the blue border itself.
    - Decode the yellow pixel position relative to the largest frame:
      top-left outside corner -> rotate 0 times, top-right -> rotate 90 degrees counterclockwise,
      bottom-right -> rotate 180 degrees, bottom-left -> rotate 270 degrees counterclockwise.
    - Return the rotated crop, without the frame and without the yellow marker.

    Key insight:
    - The largest frame decides which content to crop, while the yellow diagonal corner marker encodes how many quarter-turns to apply to that cropped content.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames = _extract_frames(g)
    assert 3 <= len(frames) <= 4, "expected 3 or 4 blue rectangular frames"
    largest = max(frames, key=lambda item: item["bbox_area"])
    largest_area = int(largest["bbox_area"])
    assert sum(int(frame["bbox_area"]) == largest_area for frame in frames) == 1, "largest frame must be unique"

    interior = _crop_bbox(g, _interior_bbox(largest["bbox"]))
    k = _marker_rotation_k(g, largest["bbox"])
    return np.rot90(interior, k=k).astype(int)


_SIZE_CANDIDATES = [
    (6, 6),
    (5, 8),
    (6, 7),
    (6, 8),
    (7, 7),
    (7, 8),
    (8, 8),
]


def _next_corner_name() -> str:
    global _GEN_CORNER_INDEX
    corner = ("tl", "tr", "br", "bl")[_GEN_CORNER_INDEX % 4]
    _GEN_CORNER_INDEX += 1
    return corner


def _frame_candidates(height: int, width: int, fh: int, fw: int, require_outer_margin: bool) -> list[tuple[int, int]]:
    r_start = 1 if require_outer_margin else 0
    c_start = 1 if require_outer_margin else 0
    r_stop = height - fh - (1 if require_outer_margin else 0)
    c_stop = width - fw - (1 if require_outer_margin else 0)
    if r_start > r_stop or c_start > c_stop:
        return []
    return [(r, c) for r in range(r_start, r_stop + 1) for c in range(c_start, c_stop + 1)]


def _bbox_with_halo(bbox: BBox, shape: tuple[int, int]) -> BBox:
    h, w = shape
    r0, c0, r1, c1 = bbox
    return max(0, r0 - 1), max(0, c0 - 1), min(h - 1, r1 + 1), min(w - 1, c1 + 1)


def _boxes_intersect(a: BBox, b: BBox) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _draw_frame(grid: np.ndarray, bbox: BBox) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME


def _grow_connected_mask(mask_shape: tuple[int, int], area: int) -> np.ndarray:
    h, w = mask_shape
    mask = np.zeros((h, w), dtype=bool)
    start = (random.randrange(h), random.randrange(w))
    mask[start] = True
    cells = [start]
    while int(mask.sum()) < area:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.append((nr, nc))
        frontier = list(dict.fromkeys(frontier))
        if not frontier:
            break
        chosen = random.choice(frontier)
        mask[chosen] = True
        cells.append(chosen)
    return mask


def _paint_pattern(crop_shape: tuple[int, int], require_rotation_distinct: bool) -> np.ndarray | None:
    crop_h, crop_w = crop_shape
    usable_h = crop_h - 2
    usable_w = crop_w - 2
    if usable_h < 1 or usable_w < 1 or usable_h * usable_w < 3:
        return None

    max_area = min(8, usable_h * usable_w)
    min_area = 3
    for _ in range(100):
        area = random.randint(min_area, max_area)
        mask = _grow_connected_mask((usable_h, usable_w), area)
        if int(mask.sum()) != area:
            continue

        patch = np.zeros((crop_h, crop_w), dtype=int)
        coords = np.argwhere(mask)
        colors = [RED] * area
        green_count = random.randint(1, area - 1)
        for idx in random.sample(range(area), green_count):
            colors[idx] = GREEN
        random.shuffle(colors)
        for (rr, cc), color in zip(coords, colors):
            patch[int(rr) + 1, int(cc) + 1] = int(color)

        if require_rotation_distinct and not _rotations_all_distinct(patch):
            continue
        return patch
    return None


def _rotations_all_distinct(grid: np.ndarray) -> bool:
    seen = set()
    for k in range(4):
        rotated = np.rot90(grid, k=k)
        key = (rotated.shape, rotated.tobytes())
        if key in seen:
            return False
        seen.add(key)
    return True


def _choose_frame_sizes(n_frames: int) -> list[tuple[int, int]]:
    chosen = random.sample(_SIZE_CANDIDATES, n_frames)
    largest = max(chosen, key=lambda size: size[0] * size[1])
    others = [size for size in chosen if size != largest]
    result = [largest]
    for size in others:
        if size[0] != size[1] and random.random() < 0.5:
            result.append((size[1], size[0]))
        else:
            result.append(size)
    return result


def _place_frames(shape: tuple[int, int], sizes: list[tuple[int, int]]) -> list[BBox] | None:
    h, w = shape
    order = [0] + random.sample(list(range(1, len(sizes))), len(sizes) - 1)
    placed: dict[int, BBox] = {}
    occupied: list[BBox] = []
    for index in order:
        fh, fw = sizes[index]
        positions = _frame_candidates(h, w, fh, fw, require_outer_margin=(index == 0))
        random.shuffle(positions)
        chosen = None
        for r0, c0 in positions:
            bbox = (r0, c0, r0 + fh - 1, c0 + fw - 1)
            halo = _bbox_with_halo(bbox, shape)
            if any(_boxes_intersect(halo, other) for other in occupied):
                continue
            chosen = bbox
            break
        if chosen is None:
            return None
        placed[index] = chosen
        occupied.append(_bbox_with_halo(chosen, shape))
    return [placed[i] for i in range(len(sizes))]


def _marker_cell(bbox: BBox, corner_name: str) -> Coord:
    r0, c0, r1, c1 = bbox
    if corner_name == "tl":
        return r0 - 1, c0 - 1
    if corner_name == "tr":
        return r0 - 1, c1 + 1
    if corner_name == "br":
        return r1 + 1, c1 + 1
    return r1 + 1, c0 - 1


def _build_expected(grid: np.ndarray, largest_bbox: BBox, corner_name: str) -> np.ndarray:
    crop = _crop_bbox(grid, _interior_bbox(largest_bbox))
    return np.rot90(crop, k=_CORNER_TO_K[corner_name]).astype(int)


def _mask_is_4_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _semantic_ok(grid: np.ndarray, largest_bbox: BBox, corner_name: str) -> bool:
    frames = _extract_frames(grid)
    if not (3 <= len(frames) <= 4):
        return False
    frame_areas = [int(frame["bbox_area"]) for frame in frames]
    if frame_areas.count(max(frame_areas)) != 1:
        return False

    markers = np.argwhere(grid == MARKER)
    if markers.shape != (1, 2):
        return False
    if tuple(map(int, markers[0])) != _marker_cell(largest_bbox, corner_name):
        return False

    for frame in frames:
        crop = _crop_bbox(grid, _interior_bbox(frame["bbox"]))
        fg = crop != BG
        if not (3 <= int(fg.sum()) <= 8):
            return False
        if not _mask_is_4_connected(fg):
            return False
        if not set(map(int, crop[fg])) <= {RED, GREEN}:
            return False
        if not ({RED, GREEN} <= set(map(int, crop[fg]))):
            return False
        if np.any(crop[0, :] != BG) or np.any(crop[-1, :] != BG) or np.any(crop[:, 0] != BG) or np.any(crop[:, -1] != BG):
            return False

    target_crop = _crop_bbox(grid, _interior_bbox(largest_bbox))
    if not _rotations_all_distinct(target_crop):
        return False

    target_out = np.rot90(target_crop, k=_CORNER_TO_K[corner_name])
    for frame in frames:
        crop = _crop_bbox(grid, _interior_bbox(frame["bbox"]))
        for k in range(4):
            if frame["bbox"] != largest_bbox and np.array_equal(np.rot90(crop, k=k), target_out):
                return False
    return np.array_equal(main(grid), target_out)


def _attempt_generate() -> np.ndarray | None:
    n_frames = random.randint(3, 4)
    sizes = _choose_frame_sizes(n_frames)
    height = random.randint(18, 24)
    width = random.randint(18, 24)
    bboxes = _place_frames((height, width), sizes)
    if bboxes is None:
        return None

    grid = np.zeros((height, width), dtype=int)
    for bbox in bboxes:
        _draw_frame(grid, bbox)

    corner_name = _next_corner_name()
    largest_bbox = bboxes[0]
    marker_r, marker_c = _marker_cell(largest_bbox, corner_name)
    if not (0 <= marker_r < height and 0 <= marker_c < width):
        return None
    if int(grid[marker_r, marker_c]) != BG:
        return None
    grid[marker_r, marker_c] = MARKER

    for idx, bbox in enumerate(bboxes):
        r0, c0, r1, c1 = bbox
        crop_h = r1 - r0 - 1
        crop_w = c1 - c0 - 1
        pattern = _paint_pattern((crop_h, crop_w), require_rotation_distinct=(idx == 0))
        if pattern is None:
            return None
        grid[r0 + 1 : r1, c0 + 1 : c1] = pattern

    if not _semantic_ok(grid, largest_bbox, corner_name):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
