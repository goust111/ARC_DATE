from __future__ import annotations

import numpy as np


BG = 0
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]
RETRY_LIMIT = 600


def get_bg(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    counts = np.bincount(values)
    return int(np.argmax(counts))


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = sorted(set(int(v) for v in np.asarray(grid, dtype=int).ravel()))
    if exclude is None:
        return colors
    return [c for c in colors if c not in exclude]


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int] | None:
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def bbox_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    return bbox_coords(np.argwhere(mask))


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.asarray(grid, dtype=int)[r0 : r1 + 1, c0 : c1 + 1].copy()


def _component_coords(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = DIR4 if connectivity == 4 else DIR8
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
                for dr, dc in dirs:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _find_foreground_components(grid: np.ndarray, bg: int) -> list[dict]:
    comps = _component_coords(np.asarray(grid, dtype=int) != int(bg), connectivity=8)
    out = []
    for coords in comps:
        box = bbox_coords(coords)
        assert box is not None
        out.append({"coords": coords, "bbox": box, "area": int(coords.shape[0])})
    return out


def _draw_rect_border(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _fill_rect(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _border_len(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    return int(2 * (h + w) - 4)


def _render_frames(shape: tuple[int, int], bboxes: dict[int, tuple[int, int, int, int]], order: list[int]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for color in order:
        _draw_rect_border(out, bboxes[int(color)], int(color))
    return out


def _left_order(left: np.ndarray, bg: int) -> list[int]:
    info = []
    for color in colors_present(left, exclude={bg}):
        mask = left == int(color)
        box = bbox_mask(mask)
        assert box is not None
        r0, c0, r1, c1 = box
        info.append((-(r1 - r0 + 1) * (c1 - c0 + 1), -(r1 - r0 + 1), -(c1 - c0 + 1), int(color)))
    info.sort()
    return [color for *_rest, color in info]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the nested ring order on the left and use it to restack the right-side hollow rectangles.

    Input structure:
    - Black background with exactly two separated foreground groups.
    - The left group is a concentric nested rectangle pattern whose colors encode an outer-to-inner order.
    - The right group contains one hollow rectangle frame per left-side color, but their visible overlap order is different.

    Rule:
    - Split the foreground into the left nested-ring component and the right overlapping-frame component.
    - Read the left colors by sorting the nested layers from largest bbox (outer ring) to smallest bbox (inner core).
    - For each right-side color, recover its full rectangle bbox from the visible frame pixels, then redraw all full 1-pixel rectangle borders from low to high priority using the left-side order.

    Key insight:
    - The left geometry is not transformed directly; it is an external instruction panel whose nesting depth defines the right-side occlusion stack.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = BG if np.any(g == BG) else int(get_bg(g))

    comps = _find_foreground_components(g, bg)
    assert len(comps) == 2, "expected exactly two separated foreground groups"
    comps.sort(key=lambda comp: (comp["bbox"][1], comp["bbox"][0]))
    left = crop_bbox(g, comps[0]["bbox"])
    right = crop_bbox(g, comps[1]["bbox"])

    order = _left_order(left, bg)
    bboxes: dict[int, tuple[int, int, int, int]] = {}
    for color in order:
        mask = right == int(color)
        box = bbox_mask(mask)
        assert box is not None, "every instruction color must appear on the right"
        bboxes[int(color)] = box

    return _render_frames(right.shape, bboxes, order)


def _make_left_panel(height: int, width: int, order: list[int]) -> np.ndarray:
    left = np.full((height, width), BG, dtype=int)
    for depth, color in enumerate(order):
        box = (depth, depth, height - 1 - depth, width - 1 - depth)
        if depth == len(order) - 1:
            _fill_rect(left, box, int(color))
        else:
            _draw_rect_border(left, box, int(color))
    return left


def _pairwise_intersections(bboxes: list[tuple[int, int, int, int]]) -> int:
    masks = []
    h = max(b[2] for b in bboxes) + 1
    w = max(b[3] for b in bboxes) + 1
    for box in bboxes:
        m = np.zeros((h, w), dtype=bool)
        _draw_rect_border(m, box, True)
        masks.append(m)
    count = 0
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            if np.any(masks[i] & masks[j]):
                count += 1
    return count


def _sample_frame_bboxes(height: int, width: int, count: int, rng: np.random.Generator) -> dict[int, tuple[int, int, int, int]] | None:
    top_pool = list(range(0, max(2, height // 2)))
    bottom_pool = list(range(max(2, height // 2), height))
    left_pool = list(range(0, max(2, width // 2)))
    right_pool = list(range(max(2, width // 2), width))
    if not top_pool or not bottom_pool or not left_pool or not right_pool:
        return None

    for _ in range(300):
        tops = [int(rng.choice(top_pool)) for _ in range(count)]
        bottoms = [int(rng.choice(bottom_pool)) for _ in range(count)]
        lefts = [int(rng.choice(left_pool)) for _ in range(count)]
        rights = [int(rng.choice(right_pool)) for _ in range(count)]

        tops[0] = 0
        bottoms[min(1, count - 1)] = height - 1
        lefts[min(2, count - 1)] = 0
        rights[min(3, count - 1)] = width - 1
        rng.shuffle(tops)
        rng.shuffle(bottoms)
        rng.shuffle(lefts)
        rng.shuffle(rights)

        boxes = []
        ok = True
        for i in range(count):
            r0 = int(tops[i])
            r1 = int(bottoms[i])
            c0 = int(lefts[i])
            c1 = int(rights[i])
            if r1 - r0 < 3 or c1 - c0 < 3:
                ok = False
                break
            boxes.append((r0, c0, r1, c1))
        if not ok or len(set(boxes)) != count:
            continue
        if _pairwise_intersections(boxes) < max(2, count - 1):
            continue
        union = np.full((height, width), BG, dtype=int)
        for box in boxes:
            _draw_rect_border(union, box, 1)
        fg_box = bbox_mask(union != BG)
        if fg_box != (0, 0, height - 1, width - 1):
            continue
        if len(_component_coords(union != BG, connectivity=8)) != 1:
            continue
        return {i: box for i, box in enumerate(boxes)}
    return None


def _visible_bbox_checks(rendered: np.ndarray, color: int, bbox: tuple[int, int, int, int]) -> bool:
    mask = rendered == int(color)
    if not np.any(mask):
        return False
    if bbox_mask(mask) != bbox:
        return False
    r0, c0, r1, c1 = bbox
    if not np.any(mask[r0, c0 : c1 + 1]):
        return False
    if not np.any(mask[r1, c0 : c1 + 1]):
        return False
    if not np.any(mask[r0 : r1 + 1, c0]):
        return False
    if not np.any(mask[r0 : r1 + 1, c1]):
        return False
    visible = int(mask.sum())
    return visible >= max(6, _border_len(bbox) // 3)


def _simple_orders(colors: list[int], bboxes: dict[int, tuple[int, int, int, int]]) -> list[list[int]]:
    return [
        sorted(colors, key=lambda color: (bboxes[color][0], bboxes[color][1], bboxes[color][2], bboxes[color][3])),
        sorted(colors, key=lambda color: (bboxes[color][1], bboxes[color][0], bboxes[color][3], bboxes[color][2])),
        sorted(colors, key=lambda color: (_border_len(bboxes[color]),) + bboxes[color]),
        sorted(colors, key=lambda color: (-_border_len(bboxes[color]),) + bboxes[color]),
    ]


def _build_attempt(rng: np.random.Generator) -> np.ndarray | None:
    count = int(rng.integers(3, 6))
    height = int(rng.integers(max(2 * count, 12), 19))
    min_left = max(2 * count, 8)
    max_left = min(13, 28 - min_left - 2)
    if min_left > max_left:
        return None
    left_w = int(rng.integers(min_left, max_left + 1))
    max_right = min(15, 30 - left_w - 2)
    if max_right < 11:
        return None
    right_w = int(rng.integers(11, max_right + 1))

    colors = [int(v) for v in rng.choice(np.arange(1, 10), size=count, replace=False)]
    target_order = colors.copy()
    left = _make_left_panel(height, left_w, target_order)

    indexed_boxes = _sample_frame_bboxes(height, right_w, count, rng)
    if indexed_boxes is None:
        return None
    bboxes = {int(color): indexed_boxes[i] for i, color in enumerate(target_order)}

    simple_orders = _simple_orders(target_order, bboxes)
    if any(target_order == order for order in simple_orders):
        return None

    current_order = target_order.copy()
    rng.shuffle(current_order)
    if current_order == target_order:
        return None

    current = _render_frames((height, right_w), bboxes, current_order)
    target = _render_frames((height, right_w), bboxes, target_order)
    if np.array_equal(current, target):
        return None
    if bbox_mask(current != BG) != (0, 0, height - 1, right_w - 1):
        return None
    if len(_component_coords(current != BG, connectivity=8)) != 1:
        return None

    occluded = False
    for color in target_order:
        bbox = bboxes[color]
        if not _visible_bbox_checks(current, color, bbox):
            return None
        if int((current == color).sum()) < _border_len(bbox):
            occluded = True
    if not occluded:
        return None

    grid = np.full((height, left_w + 2 + right_w), BG, dtype=int)
    grid[:, :left_w] = left
    grid[:, left_w + 2 :] = current

    comps = _find_foreground_components(grid, BG)
    if len(comps) != 2:
        return None
    comps.sort(key=lambda comp: comp["bbox"][1])
    if comps[0]["bbox"] != (0, 0, height - 1, left_w - 1):
        return None
    if comps[1]["bbox"] != (0, left_w + 2, height - 1, left_w + 1 + right_w):
        return None

    if _left_order(left, BG) != target_order:
        return None
    if not np.array_equal(main(grid), target):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        grid = _build_attempt(np.random.default_rng())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
