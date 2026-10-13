from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
GREY = 5
FRAME_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

ROOM_COUNTS = (1, 2, 3, 4, 5, 6)
MIN_ROOM_SIZE = 2
MAX_ROOM_SIZE = 7
MAX_TEMPLATE_SIDE = 4
FRAME_GAP = 2
TEMPLATE_GAP = 1

GENERATION_RETRIES = 400
SHAPE_RETRIES = 200
FRAME_RETRIES = 120
PLACEMENT_RETRIES = 300


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _coords_to_mask(
    coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask, (r0, c0, r1, c1)


def _trim(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    if rows.size == 0:
        return np.zeros((0, 0), dtype=bool)
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1].copy()


def _template_lookup(grid: np.ndarray) -> dict[int, np.ndarray]:
    templates: dict[int, np.ndarray] = {}
    for comp in _components(grid == GREY):
        mask, _ = _coords_to_mask(comp)
        area = int(mask.sum())
        templates[area] = mask
    return templates


def _enclosed_rooms(occupied: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = occupied.shape
    bg_mask = ~occupied
    seen = np.zeros((h, w), dtype=bool)
    rooms: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bg_mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            touches_border = False
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                if rr == 0 or cc == 0 or rr == h - 1 or cc == w - 1:
                    touches_border = True
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if bg_mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches_border:
                rooms.append(comp)
    return rooms


def _room_key(room: list[tuple[int, int]]) -> tuple[int, int, int]:
    top = min(r for r, _ in room)
    left = min(c for _, c in room)
    return (len(room), -top, -left)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each colored frame to the grey template whose area equals the frame's room count,
    recolor that template with the frame color, and place it at the top-left of the largest room.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    templates = _template_lookup(g)

    colors = sorted(int(c) for c in np.unique(g) if int(c) not in (BG, GREY))
    for color in colors:
        for comp in _components(g == color):
            occupied, (r0, c0, _r1, _c1) = _coords_to_mask(comp)
            rooms = _enclosed_rooms(occupied)
            if not rooms:
                continue

            room_count = len(rooms)
            if room_count not in templates:
                continue

            template = templates[room_count]
            room = max(rooms, key=_room_key)
            room_top = min(r for r, _ in room)
            room_left = min(c for _, c in room)
            room_set = set(room)

            for dr, dc in zip(*np.where(template), strict=False):
                rr = room_top + int(dr)
                cc = room_left + int(dc)
                if (rr, cc) in room_set:
                    out[r0 + rr, c0 + cc] = color

    return out


def _random_connected_shape(area: int) -> np.ndarray:
    if area == 1:
        return np.array([[True]], dtype=bool)

    side = MAX_TEMPLATE_SIDE
    for _ in range(SHAPE_RETRIES):
        canvas = np.zeros((side, side), dtype=bool)
        start = (random.randrange(side), random.randrange(side))
        canvas[start] = True

        while int(canvas.sum()) < area:
            frontier: set[tuple[int, int]] = set()
            for r, c in zip(*np.where(canvas), strict=False):
                for nr, nc in _neighbors4(int(r), int(c), side, side):
                    if not canvas[nr, nc]:
                        frontier.add((int(nr), int(nc)))
            if not frontier:
                break
            rr, cc = random.choice(tuple(frontier))
            canvas[rr, cc] = True

        if int(canvas.sum()) != area:
            continue

        mask = _trim(canvas)
        if max(mask.shape) > MAX_TEMPLATE_SIDE:
            continue
        if area > 2 and area * 2 < mask.shape[0] * mask.shape[1]:
            continue

        rot = random.randrange(4)
        if rot:
            mask = np.rot90(mask, rot)
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        return _trim(mask)

    width = min(area, MAX_TEMPLATE_SIDE)
    height = (area + width - 1) // width
    fallback = np.zeros((height, width), dtype=bool)
    cells = area
    for r in range(height):
        for c in range(width):
            if cells > 0:
                fallback[r, c] = True
                cells -= 1
    return _trim(fallback)


def _choose_room_grid(room_count: int) -> tuple[int, int]:
    if room_count == 1:
        return 1, 1

    options: dict[str, list[tuple[int, int]]] = {
        "horizontal": [(1, room_count)],
        "vertical": [(room_count, 1)],
    }
    mixed = [(r, room_count // r) for r in range(2, room_count + 1) if room_count % r == 0 and room_count // r > 1]
    if mixed:
        options["mixed"] = mixed

    kind = random.choice(sorted(options))
    return random.choice(options[kind])


def _sample_sizes(count: int, required: int) -> list[int]:
    required = max(int(required), MIN_ROOM_SIZE)
    if count == 1:
        high = min(MAX_ROOM_SIZE, required + 3)
        return [random.randint(required, high)]

    largest_low = max(required, MIN_ROOM_SIZE + 1)
    largest_high = min(MAX_ROOM_SIZE, largest_low + 2)
    largest = random.randint(largest_low, largest_high)
    sizes = [random.randint(MIN_ROOM_SIZE, largest - 1) for _ in range(count)]
    sizes[random.randrange(count)] = largest
    return sizes


def _build_frame_patch(color: int, room_count: int, template: np.ndarray) -> np.ndarray | None:
    template_h, template_w = map(int, template.shape)
    for _ in range(FRAME_RETRIES):
        room_rows, room_cols = _choose_room_grid(room_count)
        row_sizes = _sample_sizes(room_rows, template_h)
        col_sizes = _sample_sizes(room_cols, template_w)
        height = 2 + sum(row_sizes) + (room_rows - 1)
        width = 2 + sum(col_sizes) + (room_cols - 1)

        patch = np.zeros((height, width), dtype=int)
        patch[0, :] = color
        patch[-1, :] = color
        patch[:, 0] = color
        patch[:, -1] = color

        rr = 1
        for size in row_sizes[:-1]:
            rr += int(size)
            patch[rr, 1:-1] = color
            rr += 1

        cc = 1
        for size in col_sizes[:-1]:
            cc += int(size)
            patch[1:-1, cc] = color
            cc += 1

        occupied = patch == color
        rooms = _enclosed_rooms(occupied)
        if len(rooms) != room_count:
            continue

        largest = max(rooms, key=_room_key)
        room_mask, _ = _coords_to_mask(largest)
        if room_mask.shape[0] < template_h or room_mask.shape[1] < template_w:
            continue
        return patch
    return None


def _boxes_conflict(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    return not (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _place_frames(
    patches: list[np.ndarray], region_h: int, region_w: int
) -> list[tuple[int, int]] | None:
    order = sorted(range(len(patches)), key=lambda i: int(patches[i].shape[0] * patches[i].shape[1]), reverse=True)
    positions: list[tuple[int, int] | None] = [None] * len(patches)
    boxes: list[tuple[int, int, int, int]] = []

    for idx in order:
        patch = patches[idx]
        h, w = map(int, patch.shape)
        if h > region_h or w > region_w:
            return None

        row_margin = 1 if region_h - h >= 2 else 0
        col_margin = 1 if region_w - w >= 2 else 0
        r_min = row_margin
        r_max = region_h - h - row_margin
        c_min = col_margin
        c_max = region_w - w - col_margin
        if r_min > r_max:
            r_min, r_max = 0, region_h - h
        if c_min > c_max:
            c_min, c_max = 0, region_w - w

        placed = False
        for _ in range(PLACEMENT_RETRIES):
            r0 = random.randint(r_min, r_max)
            c0 = random.randint(c_min, c_max)
            box = (r0, c0, r0 + h - 1, c0 + w - 1)
            if any(_boxes_conflict(box, other, FRAME_GAP) for other in boxes):
                continue
            positions[idx] = (r0, c0)
            boxes.append(box)
            placed = True
            break
        if not placed:
            return None

    return [pos for pos in positions if pos is not None]


def _paint_templates(
    grid: np.ndarray,
    side: str,
    templates: list[tuple[int, np.ndarray]],
    slot_size: int,
    band_size: int,
) -> None:
    count = len(templates)
    total_len = count * slot_size + (count - 1) * TEMPLATE_GAP

    if side in {"top", "bottom"}:
        start_col = random.randint(0, grid.shape[1] - total_len)
        base_row = 1 if side == "top" else grid.shape[0] - band_size + 1
        col = start_col
        for _area, mask in templates:
            h, w = map(int, mask.shape)
            rr = base_row + random.randint(0, slot_size - h)
            cc = col + random.randint(0, slot_size - w)
            grid[rr : rr + h, cc : cc + w][mask] = GREY
            col += slot_size + TEMPLATE_GAP
    else:
        start_row = random.randint(0, grid.shape[0] - total_len)
        base_col = 1 if side == "left" else grid.shape[1] - band_size + 1
        row = start_row
        for _area, mask in templates:
            h, w = map(int, mask.shape)
            rr = row + random.randint(0, slot_size - h)
            cc = base_col + random.randint(0, slot_size - w)
            grid[rr : rr + h, cc : cc + w][mask] = GREY
            row += slot_size + TEMPLATE_GAP


def _main_offset(side: str, band_size: int) -> tuple[int, int]:
    if side == "top":
        return band_size, 0
    if side == "left":
        return 0, band_size
    return 0, 0


def _gen_attempt() -> np.ndarray | None:
    n_frames = random.randint(2, 4)
    n_templates = random.randint(max(3, n_frames), 5)
    template_counts = sorted(random.sample(ROOM_COUNTS, n_templates))

    frame_counts = [random.choice(template_counts) for _ in range(n_frames)]
    if n_frames > 1 and len(set(frame_counts)) == 1 and len(template_counts) > 1:
        frame_counts[-1] = random.choice([c for c in template_counts if c != frame_counts[0]])

    templates = {count: _random_connected_shape(count) for count in template_counts}
    colors = random.sample(FRAME_COLORS, n_frames)

    frame_patches: list[np.ndarray] = []
    for color, room_count in zip(colors, frame_counts, strict=False):
        patch = _build_frame_patch(color, room_count, templates[room_count])
        if patch is None:
            return None
        frame_patches.append(patch)

    max_frame_h = max(int(p.shape[0]) for p in frame_patches)
    max_frame_w = max(int(p.shape[1]) for p in frame_patches)
    padded_area = sum((int(p.shape[0]) + FRAME_GAP) * (int(p.shape[1]) + FRAME_GAP) for p in frame_patches)
    base_span = max(int(np.ceil(np.sqrt(padded_area))), max_frame_h + 2, max_frame_w + 2, 16)

    template_items = list(templates.items())
    random.shuffle(template_items)
    slot_size = max(max(mask.shape) for _count, mask in template_items) + 1
    band_size = slot_size + 2
    strip_len = len(template_items) * slot_size + (len(template_items) - 1) * TEMPLATE_GAP
    side = random.choice(("top", "bottom", "left", "right"))

    main_h = random.randint(base_span, base_span + 8)
    main_w = random.randint(base_span, base_span + 8)
    if side in {"top", "bottom"}:
        main_w = max(main_w, strip_len)
        grid_h = main_h + band_size
        grid_w = main_w
    else:
        main_h = max(main_h, strip_len)
        grid_h = main_h
        grid_w = main_w + band_size

    positions = _place_frames(frame_patches, main_h, main_w)
    if positions is None:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    row_off, col_off = _main_offset(side, band_size)

    for patch, (r0, c0) in zip(frame_patches, positions, strict=False):
        h, w = patch.shape
        grid[row_off + r0 : row_off + r0 + h, col_off + c0 : col_off + c0 + w] = patch

    _paint_templates(grid, side, template_items, slot_size, band_size)

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
