from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG_COLOR = 8
FRAME_COLOR = 5
MARKER_COLORS = [0, 1, 2, 3, 4, 6, 7, 9]

GRID_HEIGHT_RANGE = (10, 21)
GRID_WIDTH_RANGE = (10, 21)
MAX_OBJECTS = 4
MAX_FRAME_SIDE = 11
GEN_ATTEMPTS = 400
MIN_GAP = 1


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((int(rr), int(cc)))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            components.append(pts)
    return components


def _is_rectangular_frame(points: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    perimeter = 2 * (h + w) - 4
    if len(points) != perimeter:
        return None

    point_set = set(points)
    for rr in range(r0, r1 + 1):
        for cc in range(c0, c1 + 1):
            on_border = rr in (r0, r1) or cc in (c0, c1)
            if on_border != ((rr, cc) in point_set):
                return None
    return int(r0), int(c0), int(r1), int(c1)


def _extract_objects(grid: np.ndarray) -> list[dict[str, int | tuple[int, int, int, int]]]:
    bg = _most_common_color(grid)
    objects: list[dict[str, int | tuple[int, int, int, int]]] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for comp in _connected_components(grid == color):
            bbox = _is_rectangular_frame(comp)
            if bbox is None:
                continue
            r0, c0, r1, c1 = bbox
            inside = grid[r0 + 1 : r1, c0 + 1 : c1]
            marker_colors = set(map(int, np.unique(inside))) - {bg, color}
            assert len(marker_colors) == 1, "frame interior must contain one marker color"
            marker_color = marker_colors.pop()
            count = int(np.count_nonzero(inside == marker_color))
            assert 1 <= count <= 5, "marker count must be between 1 and 5"
            objects.append(
                {
                    "frame": (int(r0), int(c0), int(r1), int(c1)),
                    "marker_color": int(marker_color),
                    "count": int(count),
                }
            )
    objects.sort(key=lambda obj: tuple(obj["frame"]))  # deterministic overlay when frames are disjoint
    assert objects, "expected at least one rectangular frame"
    return objects


def _square_origin(frame: tuple[int, int, int, int], size: int) -> tuple[int, int]:
    r0, c0, r1, c1 = frame
    fh = r1 - r0 + 1
    fw = c1 - c0 + 1
    return int(r0 + (fh - size) // 2), int(c0 + (fw - size) // 2)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    objects = _extract_objects(g)
    out = g.copy()

    for obj in objects:
        r0, c0, r1, c1 = obj["frame"]
        out[r0 : r1 + 1, c0 : c1 + 1] = bg

    h, w = out.shape
    for obj in objects:
        size = int(obj["count"])
        color = int(obj["marker_color"])
        top, left = _square_origin(obj["frame"], size)
        rr0 = max(0, top)
        cc0 = max(0, left)
        rr1 = min(h, top + size)
        cc1 = min(w, left + size)
        if rr0 < rr1 and cc0 < cc1:
            out[rr0:rr1, cc0:cc1] = color

    return out


def _rect_union(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def _rect_too_close(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = MIN_GAP
) -> bool:
    return not (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _square_rect(
    shape: tuple[int, int], frame: tuple[int, int, int, int], size: int
) -> tuple[tuple[int, int, int, int], bool]:
    h, w = shape
    top, left = _square_origin(frame, size)
    bottom = top + size - 1
    right = left + size - 1
    clipped = top < 0 or left < 0 or bottom >= h or right >= w
    return (
        max(0, top),
        max(0, left),
        min(h - 1, bottom),
        min(w - 1, right),
    ), clipped


def _marker_slot_sets(ih: int, iw: int) -> list[list[tuple[int, int]]]:
    slot_sets: list[list[tuple[int, int]]] = []
    for row_offset in (0, 1):
        if row_offset >= ih:
            continue
        for col_offset in (0, 1):
            if col_offset >= iw:
                continue
            rows = list(range(row_offset, ih, 2))
            cols = list(range(col_offset, iw, 2))
            slots = [(r, c) for r in rows for c in cols]
            if slots:
                slot_sets.append(slots)
    return slot_sets


def _frame_dim_options(shape: tuple[int, int], count: int) -> list[tuple[int, int]]:
    h, w = shape
    options: list[tuple[int, int]] = []
    max_h = min(h, MAX_FRAME_SIDE)
    max_w = min(w, MAX_FRAME_SIDE)
    for fh in range(3, max_h + 1):
        if fh % 2 != count % 2:
            continue
        ih = fh - 2
        for fw in range(3, max_w + 1):
            if fw % 2 != count % 2:
                continue
            iw = fw - 2
            if ih * iw < count:
                continue
            if not any(len(slots) >= count for slots in _marker_slot_sets(ih, iw)):
                continue
            options.append((int(fh), int(fw)))
    random.shuffle(options)
    return options


def _placement_options(
    shape: tuple[int, int],
    frame_size: tuple[int, int],
    count: int,
    want_clipped: bool,
    occupied: list[tuple[int, int, int, int]],
) -> list[dict[str, tuple[int, int, int, int] | bool]]:
    h, w = shape
    fh, fw = frame_size
    options: list[dict[str, tuple[int, int, int, int] | bool]] = []
    for r0 in range(h - fh + 1):
        for c0 in range(w - fw + 1):
            frame = (int(r0), int(c0), int(r0 + fh - 1), int(c0 + fw - 1))
            square, clipped = _square_rect(shape, frame, count)
            if clipped != want_clipped:
                continue
            influence = _rect_union(frame, square)
            if any(_rect_too_close(influence, other) for other in occupied):
                continue
            options.append({"frame": frame, "square": square, "influence": influence, "clipped": clipped})
    random.shuffle(options)
    return options


def _choose_markers(frame: tuple[int, int, int, int], count: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame
    ih = r1 - r0 - 1
    iw = c1 - c0 - 1
    candidates = [slots for slots in _marker_slot_sets(ih, iw) if len(slots) >= count]
    assert candidates, "frame interior must support the requested number of isolated markers"
    slots = random.choice(candidates)
    chosen = random.sample(slots, count)
    return [(int(r0 + 1 + rr), int(c0 + 1 + cc)) for rr, cc in chosen]


def _max_objects_for_shape(shape: tuple[int, int]) -> int:
    area = shape[0] * shape[1]
    if area < 150:
        return 2
    if area < 250:
        return 3
    return MAX_OBJECTS


def _object_specs(num_objects: int, force_clipped: bool) -> list[dict[str, int | bool]]:
    colors = random.sample(MARKER_COLORS, num_objects)
    clipped_idx = random.randrange(num_objects) if force_clipped else -1
    specs: list[dict[str, int | bool]] = []
    for idx in range(num_objects):
        if idx == clipped_idx:
            count = random.randint(2, 5)
        else:
            count = random.randint(1, 5)
        specs.append({"count": int(count), "color": int(colors[idx]), "want_clipped": idx == clipped_idx})
    specs.sort(key=lambda spec: int(bool(spec["want_clipped"])), reverse=True)
    return specs


def _gen_attempt(force_clipped: bool) -> np.ndarray | None:
    shape = (
        random.randrange(*GRID_HEIGHT_RANGE),
        random.randrange(*GRID_WIDTH_RANGE),
    )
    max_objects = _max_objects_for_shape(shape)
    num_objects = random.randint(1, max_objects)
    specs = _object_specs(num_objects, force_clipped)

    grid = np.full(shape, BG_COLOR, dtype=int)
    occupied: list[tuple[int, int, int, int]] = []
    placed: list[dict[str, int | tuple[int, int, int, int] | list[tuple[int, int]]]] = []

    for spec in specs:
        count = int(spec["count"])
        want_clipped = bool(spec["want_clipped"])
        placed_here = False
        for frame_size in _frame_dim_options(shape, count):
            placements = _placement_options(shape, frame_size, count, want_clipped, occupied)
            if not placements:
                continue
            placement = placements[0]
            frame = placement["frame"]
            markers = _choose_markers(frame, count)
            placed.append(
                {
                    "frame": frame,
                    "marker_color": int(spec["color"]),
                    "count": count,
                    "markers": markers,
                }
            )
            occupied.append(placement["influence"])
            placed_here = True
            break
        if not placed_here:
            return None

    for obj in placed:
        r0, c0, r1, c1 = obj["frame"]
        grid[r0, c0 : c1 + 1] = FRAME_COLOR
        grid[r1, c0 : c1 + 1] = FRAME_COLOR
        grid[r0 : r1 + 1, c0] = FRAME_COLOR
        grid[r0 : r1 + 1, c1] = FRAME_COLOR
        for rr, cc in obj["markers"]:
            grid[rr, cc] = int(obj["marker_color"])

    if _most_common_color(grid) != BG_COLOR:
        return None

    found = _extract_objects(grid)
    if len(found) != len(placed):
        return None

    clipped_found = False
    for obj in found:
        square, clipped = _square_rect(grid.shape, obj["frame"], int(obj["count"]))
        _ = square
        clipped_found = clipped_found or clipped

    if force_clipped != clipped_found:
        return None

    return grid


def generate_input() -> np.ndarray:
    force_clipped = bool(random.getrandbits(1))
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(force_clipped)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
