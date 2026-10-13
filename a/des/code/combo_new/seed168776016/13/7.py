import random
from collections import Counter, deque

import numpy as np


BG = 0
FRAME_COLOR = 5
FRAME_SIZE = 5
TEMPLATE_SIZE = 3
MAX_GENERATION_ATTEMPTS = 400
GRID_H_RANGE = (14, 24)
GRID_W_RANGE = (19, 30)
FRAME_GAP_RANGE = (1, 3)
REGION_COUNT_CHOICES = (3, 4, 5)
REGION_SIZE = 3
PADDING = 1
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _mask_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    start = tuple(map(int, pts[0]))
    seen = {start}
    q = deque([start])
    h, w = mask.shape
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert len(coords) > 0, "expected a non-empty mask"
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _canonical_rotation_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    base = _crop_mask(mask)
    keys = []
    for k in range(4):
        rot = _crop_mask(np.rot90(base, k))
        keys.append((int(rot.shape[0]), int(rot.shape[1]), rot.astype(np.uint8).tobytes()))
    return min(keys)


def _find_color_components(
    grid: np.ndarray,
    exclude: set[int] | None = None,
    allowed_mask: np.ndarray | None = None,
) -> list[dict]:
    exclude = exclude or set()
    h, w = grid.shape
    if allowed_mask is None:
        allowed_mask = np.ones((h, w), dtype=bool)
    visited = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if visited[r, c] or not allowed_mask[r, c] or color in exclude:
                continue
            visited[r, c] = True
            q = deque([(r, c)])
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if visited[nr, nc] or not allowed_mask[nr, nc]:
                        continue
                    if int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    q.append((nr, nc))
            coords = np.asarray(pts, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                }
            )
    return comps


def _component_mask(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    coords = comp["coords"]
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _is_rect_frame(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h < 3 or w < 3:
        return False
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return np.array_equal(mask, border)


def _extract_templates(grid: np.ndarray, bg: int) -> tuple[list[np.ndarray], list[tuple[int, int, int, int]]]:
    comps = _find_color_components(grid, exclude={bg})
    grouped: dict[tuple[int, int, int], list[dict]] = {}
    for comp in comps:
        mask = _component_mask(comp)
        if not _is_rect_frame(mask):
            continue
        r0, c0, r1, c1 = comp["bbox"]
        grouped.setdefault((comp["color"], mask.shape[0], mask.shape[1]), []).append(comp)

    chosen: list[dict] | None = None
    best_score: tuple[int, int, int] | None = None
    for group in grouped.values():
        if len(group) < 3:
            continue
        group = sorted(group, key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))
        trio = sorted(group, key=lambda comp: comp["bbox"][2], reverse=True)[:3]
        trio = sorted(trio, key=lambda comp: comp["bbox"][1])
        score = (
            min(comp["bbox"][0] for comp in trio),
            sum(comp["bbox"][2] for comp in trio),
            len(group),
        )
        if best_score is None or score > best_score:
            best_score = score
            chosen = trio
    assert chosen is not None, "expected three bottom template frames"

    frame_color = chosen[0]["color"]
    templates = []
    boxes = []
    for comp in chosen:
        assert comp["color"] == frame_color, "template frames must share a border color"
        r0, c0, r1, c1 = comp["bbox"]
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        pattern = (interior != bg) & (interior != frame_color)
        assert np.any(pattern), "each template box must contain a pattern"
        templates.append(_crop_mask(pattern))
        boxes.append((r0, c0, r1, c1))
    return templates, boxes


def _rotate_coords_cw(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    rsum = r0 + r1
    csum = c0 + c1
    out = []
    for r, c in coords:
        nr2 = rsum + 2 * int(c) - csum
        nc2 = csum - 2 * int(r) + rsum
        assert nr2 % 2 == 0 and nc2 % 2 == 0, "90-degree rotation must land on grid cells"
        out.append((nr2 // 2, nc2 // 2))
    return out


def _flip_coords_horizontal(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    _, c0, _, c1 = bbox
    return [(int(r), c0 + c1 - int(c)) for r, c in coords]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Classify each upper region by matching its shape to one of three bottom seed templates, then keep, rotate, or flip it.

    Input structure:
    - The background is black.
    - Along the bottom sit three grey 5x5 bordered boxes arranged left to right; each box contains a small colored pattern inside its 3x3 interior.
    - Above those boxes are several separated monochrome regions. Each region's cropped shape matches exactly one template shape up to rotation.

    Rule:
    - Read the three bottom template shapes from left to right; they represent keep, 90-degree clockwise rotation, and horizontal flip.
    - For every upper monochrome region, crop it to its bounding box and compare its shape against the three templates, allowing 0/90/180/270-degree rotations when matching.
    - Leave keep-matched regions unchanged, rotate rotate-matched regions 90 degrees clockwise about their own bounding-box center, and flip flip-matched regions across their own bounding-box vertical midline. Keep the template boxes unchanged.

    Key insight:
    - The bottom patterns are not icons for actions; they are shape labels. A region is transformed according to which template class its normalized silhouette belongs to.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    templates, boxes = _extract_templates(g, bg)
    template_keys = [_canonical_rotation_key(mask) for mask in templates]
    assert len(set(template_keys)) == 3, "template shapes must be distinct up to rotation"

    protected = np.zeros_like(g, dtype=bool)
    for r0, c0, r1, c1 in boxes:
        protected[r0 : r1 + 1, c0 : c1 + 1] = True

    regions = _find_color_components(g, exclude={bg}, allowed_mask=~protected)
    out = g.copy()
    h, w = g.shape

    for comp in regions:
        coords = comp["coords"]
        bbox = comp["bbox"]
        mask = _component_mask(comp)
        key = _canonical_rotation_key(mask)
        matches = [idx for idx, template_key in enumerate(template_keys) if template_key == key]
        assert len(matches) == 1, "each region must match exactly one template class"
        op = matches[0]
        if op == 0:
            continue

        color = comp["color"]
        for r, c in coords:
            out[int(r), int(c)] = bg
        if op == 1:
            new_coords = _rotate_coords_cw(coords, bbox)
        else:
            new_coords = _flip_coords_horizontal(coords, bbox)
        for nr, nc in new_coords:
            assert 0 <= nr < h and 0 <= nc < w, "transformed region must stay inside the grid"
            out[nr, nc] = color
    return out


def _template_library() -> list[np.ndarray]:
    masks = []
    seen = set()
    for bits in range(1, 1 << 9):
        flat = [bool((bits >> i) & 1) for i in range(9)]
        mask = np.asarray(flat, dtype=bool).reshape(3, 3)
        count = int(mask.sum())
        if not (3 <= count <= 6):
            continue
        if not _mask_connected(mask):
            continue
        rows, cols = np.where(mask)
        if rows.min() != 0 or rows.max() != 2 or cols.min() != 0 or cols.max() != 2:
            continue
        key = _canonical_rotation_key(mask)
        if key in seen:
            continue
        seen.add(key)
        masks.append(mask)
    masks.sort(key=lambda mask: (int(mask.sum()), mask.astype(np.uint8).tobytes()))
    return masks


TEMPLATE_LIBRARY = _template_library()
ROTATE_TEMPLATES = [mask for mask in TEMPLATE_LIBRARY if not np.array_equal(mask, np.rot90(mask, -1))]
FLIP_TEMPLATES = [
    mask
    for mask in TEMPLATE_LIBRARY
    if all(not np.array_equal(np.rot90(mask, k), np.fliplr(np.rot90(mask, k))) for k in range(4))
]


def _choose_templates(rng: random.Random) -> list[np.ndarray]:
    flip_mask = rng.choice(FLIP_TEMPLATES)
    flip_key = _canonical_rotation_key(flip_mask)

    rotate_choices = [mask for mask in ROTATE_TEMPLATES if _canonical_rotation_key(mask) != flip_key]
    rotate_mask = rng.choice(rotate_choices)
    used = {flip_key, _canonical_rotation_key(rotate_mask)}

    keep_choices = [mask for mask in TEMPLATE_LIBRARY if _canonical_rotation_key(mask) not in used]
    keep_mask = rng.choice(keep_choices)
    return [keep_mask.copy(), rotate_mask.copy(), flip_mask.copy()]


def _reserve(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = PADDING) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occupied.shape[0], r0 + h + pad)
    cc1 = min(occupied.shape[1], c0 + w + pad)
    occupied[rr0:rr1, cc0:cc1] = True


def _place_regions(
    rng: random.Random,
    grid_h: int,
    grid_w: int,
    frame_boxes: list[tuple[int, int, int, int]],
    count: int,
) -> list[tuple[int, int]] | None:
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    frame_top = min(box[0] for box in frame_boxes)
    for r0, c0, r1, c1 in frame_boxes:
        _reserve(occupied, r0, c0, r1 - r0 + 1, c1 - c0 + 1)

    max_top = frame_top - REGION_SIZE - 1
    if max_top < 0 or grid_w < REGION_SIZE:
        return None

    positions = []
    for _ in range(count):
        placed = False
        for _ in range(200):
            r0 = rng.randint(0, max_top)
            c0 = rng.randint(0, grid_w - REGION_SIZE)
            if occupied[r0 : r0 + REGION_SIZE, c0 : c0 + REGION_SIZE].any():
                continue
            positions.append((r0, c0))
            _reserve(occupied, r0, c0, REGION_SIZE, REGION_SIZE)
            placed = True
            break
        if not placed:
            return None
    return positions


def _operation_counts(rng: random.Random) -> list[int]:
    total = rng.choice(REGION_COUNT_CHOICES)
    counts = [1, 1, 1]
    for _ in range(total - 3):
        counts[rng.randrange(3)] += 1
    return counts


def _region_mask_for_template(template: np.ndarray, rng: random.Random) -> np.ndarray:
    k = rng.randrange(4)
    return np.rot90(template, -k)


def _draw_frame(grid: np.ndarray, top: int, left: int, template: np.ndarray, color: int) -> None:
    grid[top : top + FRAME_SIZE, left] = FRAME_COLOR
    grid[top : top + FRAME_SIZE, left + FRAME_SIZE - 1] = FRAME_COLOR
    grid[top, left : left + FRAME_SIZE] = FRAME_COLOR
    grid[top + FRAME_SIZE - 1, left : left + FRAME_SIZE] = FRAME_COLOR
    interior = grid[top + 1 : top + 1 + TEMPLATE_SIZE, left + 1 : left + 1 + TEMPLATE_SIZE]
    interior[template] = color


def _apply_local_transform(mask: np.ndarray, op: int) -> np.ndarray:
    if op == 0:
        return mask.copy()
    if op == 1:
        return np.rot90(mask, -1)
    return np.fliplr(mask)


def _frame_positions(rng: random.Random, grid_w: int, grid_h: int) -> list[tuple[int, int, int, int]] | None:
    gap1 = rng.randint(*FRAME_GAP_RANGE)
    gap2 = rng.randint(*FRAME_GAP_RANGE)
    total_w = FRAME_SIZE * 3 + gap1 + gap2
    if total_w > grid_w:
        return None
    left = rng.randint(0, grid_w - total_w)
    top = grid_h - FRAME_SIZE
    cols = [left, left + FRAME_SIZE + gap1, left + FRAME_SIZE * 2 + gap1 + gap2]
    return [(top, c, top + FRAME_SIZE - 1, c + FRAME_SIZE - 1) for c in cols]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid_h = rng.randint(*GRID_H_RANGE)
    grid_w = rng.randint(*GRID_W_RANGE)
    frame_boxes = _frame_positions(rng, grid_w, grid_h)
    if frame_boxes is None:
        return None

    templates = _choose_templates(rng)
    counts = _operation_counts(rng)
    ops = [0] * counts[0] + [1] * counts[1] + [2] * counts[2]
    rng.shuffle(ops)

    region_positions = _place_regions(rng, grid_h, grid_w, frame_boxes, len(ops))
    if region_positions is None:
        return None

    needed_colors = 3 + len(ops)
    if needed_colors > len(COLOR_POOL):
        return None
    colors = rng.sample(COLOR_POOL, needed_colors)
    template_colors = colors[:3]
    region_colors = colors[3:]

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    expected = grid.copy()

    for idx, (r0, c0, _, _) in enumerate(frame_boxes):
        _draw_frame(grid, r0, c0, templates[idx], template_colors[idx])
        _draw_frame(expected, r0, c0, templates[idx], template_colors[idx])

    for idx, op in enumerate(ops):
        top, left = region_positions[idx]
        color = region_colors[idx]
        local_mask = _region_mask_for_template(templates[op], rng)
        patch = grid[top : top + REGION_SIZE, left : left + REGION_SIZE]
        patch[local_mask] = color

        out_patch = expected[top : top + REGION_SIZE, left : left + REGION_SIZE]
        out_patch[:, :] = BG
        out_patch[_apply_local_transform(local_mask, op)] = color

    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None
    if np.array_equal(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            grid = _gen_attempt(rng)
        except Exception:
            continue
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
