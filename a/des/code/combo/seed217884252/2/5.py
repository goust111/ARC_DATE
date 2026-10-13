import random

import numpy as np


BG = 7
FRAME = 5
COLOR_POOL = [0, 1, 2, 3, 4, 6, 8, 9]
SIDES = ("top", "bottom", "left", "right")

GRID_H_RANGE = (22, 30)
GRID_W_RANGE = (22, 30)
FRAME_COUNT_CHOICES = [4, 5]
INNER_H_MAX = 5
INNER_W_MAX = 5
PALETTE_ENTRY_H_RANGE = (2, 3)
PALETTE_ENTRY_W_RANGE = (2, 4)
PALETTE_GAP = 1
PLACEMENT_PAD = 1
PLACEMENT_RETRIES = 300
GENERATION_RETRIES = 400


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _touches_border(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    h, w = shape
    return r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _is_rectangular_frame(grid: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(cells)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if len(cells) != 2 * (h + w) - 4:
        return False
    if not np.all(grid[r0, c0 : c1 + 1] == FRAME):
        return False
    if not np.all(grid[r1, c0 : c1 + 1] == FRAME):
        return False
    if not np.all(grid[r0 : r1 + 1, c0] == FRAME):
        return False
    if not np.all(grid[r0 : r1 + 1, c1] == FRAME):
        return False
    if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] == FRAME):
        return False
    return True


def _find_palette(grid: np.ndarray) -> dict[int, int]:
    h, w = grid.shape
    candidates: list[tuple[int, int, int, int, int]] = []
    for color in np.unique(grid):
        color = int(color)
        if color in (BG, FRAME):
            continue
        for comp in _connected_components(grid == color):
            r0, c0, r1, c1 = _bbox(comp)
            if len(comp) != (r1 - r0 + 1) * (c1 - c0 + 1):
                continue
            if not _touches_border((r0, c0, r1, c1), (h, w)):
                continue
            candidates.append((r0, c0, r1, c1, color))

    assert len(candidates) == 4, "expected exactly four border-touching palette rectangles"
    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))

    same_left = all(c0 == candidates[0][1] for _r0, c0, _r1, _c1, _color in candidates)
    same_right = all(c1 == candidates[0][3] for _r0, _c0, _r1, c1, _color in candidates)
    assert same_left or same_right, "palette entries must align on one vertical edge"
    assert candidates[0][0] == 0 or candidates[-1][2] == h - 1, "palette must sit in a grid corner"

    for i in range(3):
        assert candidates[i][2] < candidates[i + 1][0], "palette entries must be vertically separated"

    return {i + 1: color for i, (_r0, _c0, _r1, _c1, color) in enumerate(candidates)}


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for comp in _connected_components(grid == FRAME):
        if _is_rectangular_frame(grid, comp):
            frames.append(_bbox(comp))
    assert frames, "expected at least one grey frame"
    return frames


def _inner_block_bbox(grid: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = frame_bbox
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    mask = (inner != BG) & (inner != FRAME)
    assert np.any(mask), "each frame must contain an inner colored block"
    vals = np.unique(inner[mask])
    assert len(vals) == 1, "inner block must use a single color"
    rows, cols = np.where(mask)
    br0 = int(rows.min()) + r0 + 1
    bc0 = int(cols.min()) + c0 + 1
    br1 = int(rows.max()) + r0 + 1
    bc1 = int(cols.max()) + c0 + 1
    assert np.all(mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]), "inner block must be solid"
    return br0, bc0, br1, bc1


def _contact_count(
    frame_bbox: tuple[int, int, int, int], block_bbox: tuple[int, int, int, int]
) -> int:
    fr0, fc0, fr1, fc1 = frame_bbox
    br0, bc0, br1, bc1 = block_bbox
    return sum(
        [
            br0 == fr0 + 1,
            br1 == fr1 - 1,
            bc0 == fc0 + 1,
            bc1 == fc1 - 1,
        ]
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each frame by how many inner sides its solid block touches, then recolor the frame
    with the palette color assigned to that touch count.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    palette = _find_palette(g)
    out = g.copy()
    for frame_bbox in _find_frames(g):
        block_bbox = _inner_block_bbox(g, frame_bbox)
        count = _contact_count(frame_bbox, block_bbox)
        assert 1 <= count <= 4, "inner block must touch between one and four sides"
        _draw_frame(out, frame_bbox, palette[count])
    return out


def _mark(blocked: np.ndarray, bbox: tuple[int, int, int, int], pad: int) -> None:
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(blocked.shape[0] - 1, r1 + pad)
    cc1 = min(blocked.shape[1] - 1, c1 + pad)
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _can_place(blocked: np.ndarray, bbox: tuple[int, int, int, int], pad: int) -> bool:
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(blocked.shape[0] - 1, r1 + pad)
    cc1 = min(blocked.shape[1] - 1, c1 + pad)
    return not np.any(blocked[rr0 : rr1 + 1, cc0 : cc1 + 1])


def _build_axis(span: int, touch_low: bool, touch_high: bool) -> tuple[int, int, int]:
    low_gap = 0 if touch_low else 1
    high_gap = 0 if touch_high else 1
    block = 1
    extra = span - (low_gap + high_gap + block)
    assert extra >= 0
    slots = ["block"]
    if not touch_low:
        slots.append("low")
    if not touch_high:
        slots.append("high")
    for _ in range(extra):
        slot = random.choice(slots)
        if slot == "block":
            block += 1
        elif slot == "low":
            low_gap += 1
        else:
            high_gap += 1
    return low_gap, block, high_gap


def _choose_block_geometry(
    inner_h: int, inner_w: int, contacts: set[str]
) -> tuple[int, int, int, int]:
    top_gap, block_h, bottom_gap = _build_axis(inner_h, "top" in contacts, "bottom" in contacts)
    left_gap, block_w, right_gap = _build_axis(inner_w, "left" in contacts, "right" in contacts)
    assert top_gap + block_h + bottom_gap == inner_h
    assert left_gap + block_w + right_gap == inner_w
    return top_gap, left_gap, block_h, block_w


def _palette_layout(
    h: int, w: int, entry_h: int, entry_w: int, corner: str
) -> list[tuple[int, int, int, int]]:
    total_h = 4 * entry_h + 3 * PALETTE_GAP
    start_r = 0 if "top" in corner else h - total_h
    start_c = 0 if "left" in corner else w - entry_w
    rects = []
    for i in range(4):
        r0 = start_r + i * (entry_h + PALETTE_GAP)
        c0 = start_c
        rects.append((r0, c0, r0 + entry_h - 1, c0 + entry_w - 1))
    return rects


def _expected_output(
    grid: np.ndarray,
    frame_records: list[dict[str, object]],
    palette_colors: list[int],
) -> np.ndarray:
    out = grid.copy()
    for record in frame_records:
        bbox = record["bbox"]
        count = int(record["count"])
        _draw_frame(out, bbox, palette_colors[count - 1])
    return out


def _generate_one() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    entry_h = random.randint(*PALETTE_ENTRY_H_RANGE)
    entry_w = random.randint(*PALETTE_ENTRY_W_RANGE)
    if 4 * entry_h + 3 * PALETTE_GAP > h:
        return None

    corner = random.choice(["top_left", "top_right", "bottom_left", "bottom_right"])
    palette_colors = random.sample(COLOR_POOL, 4)
    palette_rects = _palette_layout(h, w, entry_h, entry_w, corner)
    for rect, color in zip(palette_rects, palette_colors):
        r0, c0, r1, c1 = rect
        grid[r0 : r1 + 1, c0 : c1 + 1] = color
    _mark(blocked, (palette_rects[0][0], palette_rects[0][1], palette_rects[-1][2], palette_rects[-1][3]), 2)

    n_frames = random.choice(FRAME_COUNT_CHOICES)
    counts = [1, 2, 3, 4]
    while len(counts) < n_frames:
        counts.append(random.randint(1, 4))
    random.shuffle(counts)
    block_colors = random.sample(COLOR_POOL, n_frames)

    frame_records: list[dict[str, object]] = []
    for count, block_color in zip(counts, block_colors):
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            contacts = set(random.sample(SIDES, count))
            min_inner_h = max(2, 1 + int("top" not in contacts) + int("bottom" not in contacts))
            min_inner_w = max(2, 1 + int("left" not in contacts) + int("right" not in contacts))
            if min_inner_h > INNER_H_MAX or min_inner_w > INNER_W_MAX:
                continue
            inner_h = random.randint(min_inner_h, INNER_H_MAX)
            inner_w = random.randint(min_inner_w, INNER_W_MAX)
            outer_h = inner_h + 2
            outer_w = inner_w + 2
            r0 = random.randint(1, h - outer_h - 1)
            c0 = random.randint(1, w - outer_w - 1)
            bbox = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
            if not _can_place(blocked, bbox, PLACEMENT_PAD):
                continue

            top_gap, left_gap, block_h, block_w = _choose_block_geometry(inner_h, inner_w, contacts)
            br0 = r0 + 1 + top_gap
            bc0 = c0 + 1 + left_gap
            br1 = br0 + block_h - 1
            bc1 = bc0 + block_w - 1

            _draw_frame(grid, bbox, FRAME)
            grid[br0 : br1 + 1, bc0 : bc1 + 1] = block_color
            _mark(blocked, bbox, PLACEMENT_PAD)
            frame_records.append({"bbox": bbox, "count": count})
            placed = True
            break
        if not placed:
            return None

    expected = _expected_output(grid, frame_records, palette_colors)
    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        result = _generate_one()
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate a valid input")
