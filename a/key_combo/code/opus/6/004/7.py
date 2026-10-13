from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


PALETTE = (1, 2, 3, 4, 7)
SPRITE_SHAPES = (
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (2, 0)),
    ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2)),
    ((0, 0), (1, 0), (1, 1), (1, 2), (2, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
)
GEN_RETRIES = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
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
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _frame_border_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _is_hollow_frame(grid: np.ndarray, color: int, coords: np.ndarray, bg: int) -> bool:
    bbox = bbox_coords(coords)
    sprite = crop_bbox(grid, bbox)
    h, w = sprite.shape
    if h < 5 or w < 7:
        return False
    border = _frame_border_mask((h, w))
    if not np.all(sprite[border] == color):
        return False
    if not np.all(sprite[~border] == bg):
        return False
    return int(np.count_nonzero(sprite == color)) == int(np.count_nonzero(border))


def _extract_scene(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    items: list[dict] = []
    for color in colors:
        comps = components_mask(g == color)
        frame_comps = [coords for coords in comps if _is_hollow_frame(g, color, coords, bg)]
        assert len(frame_comps) == 1, f"expected exactly one frame for color {color}"
        frame_coords = frame_comps[0]
        frame_bbox = bbox_coords(frame_coords)
        frame_crop = crop_bbox(g, frame_bbox)
        sprite_count = sum(1 for coords in comps if not np.array_equal(coords, frame_coords))
        assert sprite_count >= 1, f"expected at least one sprite for color {color}"
        items.append(
            {
                "color": color,
                "count": int(sprite_count),
                "bbox": frame_bbox,
                "frame": frame_crop,
            }
        )
    counts = [item["count"] for item in items]
    assert 3 <= len(items) <= 5
    assert len(set(counts)) == len(counts), "counts must be unique"
    return bg, items


def _draw_count_dots(frame: np.ndarray, count: int, color: int, bg: int) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape
    inner_h = h - 2
    inner_w = w - 2
    assert inner_h >= 3 and inner_w >= 5
    assert inner_w >= 2 * count - 1
    dot_row = 1 + (inner_h - 1) // 2
    for i in range(count):
        out[dot_row, 1 + 2 * i] = color
    return out


def _stack_frames(items: list[dict], bg: int) -> np.ndarray:
    ordered = sorted(items, key=lambda item: -int(item["count"]))
    width = max(int(item["frame"].shape[1]) for item in ordered)
    height = sum(int(item["frame"].shape[0]) for item in ordered) + len(ordered) - 1
    out = np.full((height, width), bg, dtype=int)
    r0 = 0
    for item in ordered:
        frame = _draw_count_dots(item["frame"], int(item["count"]), int(item["color"]), bg)
        h, w = frame.shape
        out[r0 : r0 + h, :w] = frame
        r0 += h + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count same-colored sprites for each hollow frame, then reorder the frames into a vertical stack.

    Input structure:
    - The background is black.
    - Three to five distinct colors each form exactly one hollow rectangular frame.
    - Additional 4-connected monochrome sprites of those same colors appear outside the frames.

    Rule:
    - For each frame color, count how many separate same-colored sprites exist outside the frame.
    - Erase the sprites conceptually, crop each frame to its bounding box, and sort the frames by that count in descending order.
    - Stack the cropped frames top-to-bottom with one background row between neighbors, left-aligned to the widest frame.
    - Inside each stacked frame, draw that many same-colored dots on the cavity's middle row, starting at the left inner wall and skipping one column between consecutive dots.

    Key insight:
    - The same per-color count has two roles at once: it determines the frame's rank in the stack and also how many dots must be rendered inside that frame.
    """
    bg, items = _extract_scene(np.asarray(grid, dtype=int))
    return _stack_frames(items, bg)


def _normalize_shape(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=int)
    coords = coords - coords.min(axis=0)
    order = np.lexsort((coords[:, 1], coords[:, 0]))
    return coords[order]


def _transform_shape(shape: tuple[tuple[int, int], ...]) -> np.ndarray:
    coords = np.array(shape, dtype=int)
    if random.random() < 0.5:
        coords[:, 1] *= -1
    rot = random.randrange(4)
    for _ in range(rot):
        coords = np.column_stack((-coords[:, 1], coords[:, 0]))
    return _normalize_shape(coords)


def _choose_inner_heights(n: int) -> list[int]:
    max_inner_sum = 31 - 3 * n
    heights = [3] * n
    even_idx = random.randrange(n)
    heights[even_idx] = 4
    budget = max_inner_sum - sum(heights)
    indices = list(range(n))
    random.shuffle(indices)
    for idx in indices:
        while budget > 0:
            step = random.choice([0, 1, 2, 3])
            if step == 0 or heights[idx] + step > 6 or step > budget:
                break
            heights[idx] += step
            budget -= step
            if random.random() < 0.5:
                break
    random.shuffle(heights)
    return heights


def _paint_frame(grid: np.ndarray, top: int, left: int, h: int, w: int, color: int) -> None:
    grid[top, left : left + w] = color
    grid[top + h - 1, left : left + w] = color
    grid[top : top + h, left] = color
    grid[top : top + h, left + w - 1] = color


def _reserve_box(mask: np.ndarray, top: int, left: int, h: int, w: int, pad: int) -> None:
    h0, w0 = mask.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h0, top + h + pad)
    c1 = min(w0, left + w + pad)
    mask[r0:r1, c0:c1] = True


def _free_box(mask: np.ndarray, top: int, left: int, h: int, w: int, pad: int) -> bool:
    h0, w0 = mask.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h0, top + h + pad)
    c1 = min(w0, left + w + pad)
    return not bool(mask[r0:r1, c0:c1].any())


def _place_rectangles(
    canvas_shape: tuple[int, int], specs: list[dict]
) -> tuple[np.ndarray, list[dict]] | None:
    h0, w0 = canvas_shape
    reserved = np.zeros((h0, w0), dtype=bool)
    placed: list[dict] = []
    order = sorted(specs, key=lambda item: item["h"] * item["w"], reverse=True)
    for spec in order:
        positions = [
            (r, c)
            for r in range(h0 - spec["h"] + 1)
            for c in range(w0 - spec["w"] + 1)
            if _free_box(reserved, r, c, spec["h"], spec["w"], pad=1)
        ]
        if not positions:
            return None
        top, left = random.choice(positions)
        spec = dict(spec)
        spec["top"] = int(top)
        spec["left"] = int(left)
        _reserve_box(reserved, top, left, spec["h"], spec["w"], pad=1)
        placed.append(spec)
    return reserved, placed


def _place_sprites(
    grid: np.ndarray,
    reserved: np.ndarray,
    frame_specs: list[dict],
) -> bool:
    h0, w0 = grid.shape
    all_sprite_cells: set[tuple[int, int]] = set()
    for spec in frame_specs:
        for _ in range(spec["count"]):
            shape = _transform_shape(random.choice(SPRITE_SHAPES))
            sh, sw = shape.max(axis=0) + 1
            positions = [
                (r, c)
                for r in range(h0 - sh + 1)
                for c in range(w0 - sw + 1)
                if _free_box(reserved, r, c, sh, sw, pad=1)
            ]
            random.shuffle(positions)
            placed = False
            for top, left in positions:
                cells = {(top + int(dr), left + int(dc)) for dr, dc in shape}
                if any(cell in all_sprite_cells for cell in cells):
                    continue
                for rr, cc in cells:
                    grid[rr, cc] = int(spec["color"])
                _reserve_box(reserved, top, left, sh, sw, pad=1)
                all_sprite_cells.update(cells)
                placed = True
                break
            if not placed:
                return False
    return True


def _build_expected_from_specs(specs: list[dict], bg: int) -> np.ndarray:
    items = []
    for spec in specs:
        frame = np.full((spec["h"], spec["w"]), bg, dtype=int)
        _paint_frame(frame, 0, 0, spec["h"], spec["w"], int(spec["color"]))
        items.append({"color": spec["color"], "count": spec["count"], "frame": frame})
    return _stack_frames(items, bg)


def _validate_generated(grid: np.ndarray, planned_specs: list[dict]) -> None:
    bg, parsed = _extract_scene(grid)
    planned_counts = {int(spec["color"]): int(spec["count"]) for spec in planned_specs}
    parsed_counts = {int(item["color"]): int(item["count"]) for item in parsed}
    assert planned_counts == parsed_counts
    for spec in planned_specs:
        assert spec["inner_w"] >= 2 * spec["count"] - 1
        assert spec["inner_h"] >= 3
    expected = _build_expected_from_specs(planned_specs, bg)
    actual = main(grid)
    assert np.array_equal(actual, expected)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        n = random.randint(3, 5)
        colors = random.sample(PALETTE, n)
        counts = random.sample([1, 2, 3, 4, 5], n)
        inner_heights = _choose_inner_heights(n)
        frame_specs: list[dict] = []
        for color, count, inner_h in zip(colors, counts, inner_heights):
            min_inner_w = max(5, 2 * count - 1)
            inner_w = random.randint(min_inner_w, min(10, min_inner_w + 3))
            frame_specs.append(
                {
                    "color": int(color),
                    "count": int(count),
                    "inner_h": int(inner_h),
                    "inner_w": int(inner_w),
                    "h": int(inner_h + 2),
                    "w": int(inner_w + 2),
                }
            )

        total_frame_area = sum(spec["h"] * spec["w"] for spec in frame_specs)
        total_sprites = sum(spec["count"] for spec in frame_specs)
        side = max(22, min(30, int(round((total_frame_area + 9 * total_sprites) ** 0.5)) + 8))
        canvas_h = random.randint(max(22, side - 2), min(30, side + 2))
        canvas_w = random.randint(max(22, side - 2), min(30, side + 2))
        placement = _place_rectangles((canvas_h, canvas_w), frame_specs)
        if placement is None:
            continue
        reserved, placed_specs = placement

        grid = np.zeros((canvas_h, canvas_w), dtype=int)
        for spec in placed_specs:
            _paint_frame(grid, spec["top"], spec["left"], spec["h"], spec["w"], int(spec["color"]))
        if not _place_sprites(grid, reserved.copy(), placed_specs):
            continue

        try:
            _validate_generated(grid, placed_specs)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate valid input")
