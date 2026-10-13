from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
FENCE_COLOR = 5
OBJECT_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 300


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _is_rectangular_frame(grid: np.ndarray, color: int) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(grid == int(color))
    if coords.size == 0:
        return None
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    perimeter = 2 * h + 2 * w - 4
    if int(coords.shape[0]) != perimeter:
        return None
    if not np.all(grid[r0, c0 : c1 + 1] == int(color)):
        return None
    if not np.all(grid[r1, c0 : c1 + 1] == int(color)):
        return None
    if not np.all(grid[r0 : r1 + 1, c0] == int(color)):
        return None
    if not np.all(grid[r0 : r1 + 1, c1] == int(color)):
        return None
    if h > 2 and w > 2 and np.any(grid[r0 + 1 : r1, c0 + 1 : c1] == int(color)):
        return None
    return r0, c0, r1, c1


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _grow_connected_mask(h: int, w: int, area: int) -> np.ndarray | None:
    cells = {(random.randrange(h), random.randrange(w))}
    while len(cells) < area:
        frontier = set()
        for r, c in cells:
            for dr, dc in NEIGHBORS4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(tuple(frontier)))
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return _trim_mask(mask)


def _random_object_mask() -> np.ndarray:
    for _ in range(80):
        h = random.randint(1, 4)
        w = random.randint(1, 4)
        if h == 1 and w == 1:
            continue
        min_area = max(2, (h * w + 1) // 2)
        area = random.randint(min_area, h * w)
        mask = _grow_connected_mask(h, w, area)
        if mask is None:
            continue
        if mask.shape == (1, 1):
            continue
        return mask
    raise RuntimeError("failed to sample object mask")


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = mask.shape
    sub = grid[r0 : r0 + h, c0 : c0 + w]
    sub[mask] = int(color)


def _reserve_rect(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occupied.shape[0], r0 + h + pad)
    cc1 = min(occupied.shape[1], c0 + w + pad)
    occupied[rr0:rr1, cc0:cc1] = True


def _rect_is_free(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occupied.shape[0], r0 + h + pad)
    cc1 = min(occupied.shape[1], c0 + w + pad)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _random_free_position(occupied: np.ndarray, h: int, w: int, pad: int = 1) -> tuple[int, int] | None:
    spots = []
    max_r = occupied.shape[0] - h
    max_c = occupied.shape[1] - w
    for r0 in range(max_r + 1):
        for c0 in range(max_c + 1):
            if _rect_is_free(occupied, r0, c0, h, w, pad=pad):
                spots.append((r0, c0))
    if not spots:
        return None
    return random.choice(spots)


def _flags_from_frame(
    grid: np.ndarray, frame_bbox: tuple[int, int, int, int], frame_color: int, bg: int
) -> tuple[str, list[int], dict[int, tuple[int, int]]]:
    r0, c0, r1, c1 = frame_bbox
    flags = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            value = int(grid[r, c])
            if value not in (int(bg), int(frame_color)):
                flags.append((r, c, value))
    assert len(flags) >= 2, "expected at least two inner flag pixels"

    colors = [value for _r, _c, value in flags]
    assert len(colors) == len(set(colors)), "flag colors must be unique"

    rows = {r for r, _c, _value in flags}
    cols = {c for _r, c, _value in flags}
    if len(rows) == 1:
        flags.sort(key=lambda item: item[1])
        orientation = "horizontal"
    else:
        assert len(cols) == 1, "flags must form one row or one column"
        flags.sort(key=lambda item: item[0])
        orientation = "vertical"

    order = [value for _r, _c, value in flags]
    positions = {value: (r, c) for r, c, value in flags}
    return orientation, order, positions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Chain the outside objects in the fence-flag order while preserving the fence and flags.

    Input structure:
    - The background is black, and one grey one-pixel rectangular fence encloses a small interior.
    - Inside the fence, distinct single-pixel colored flags appear in one row or one column with background gaps between them.
    - Outside the fence, each flag color appears again as exactly one same-colored connected object.

    Rule:
    - Detect the fence, then read the interior flag colors in their visible order.
    - The flag row/column orientation decides the chain axis: a horizontal flag sequence packs objects left-to-right, and a vertical flag sequence packs objects top-to-bottom.
    - Clear the outside objects from their original locations, keep the first ordered object fixed, and translate every later object so its bounding box sits immediately to the right of or immediately below the previous placed object.
    - Preserve the fence and the flags unchanged.

    Key insight:
    - The fence interior encodes two separate controls at once: flag order selects which object moves next, and flag orientation selects the packing direction.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    colors = [int(c) for c in np.unique(g) if int(c) != int(bg)]

    frame_candidates = []
    for color in colors:
        bbox = _is_rectangular_frame(g, int(color))
        if bbox is not None:
            frame_candidates.append((int(color), bbox))
    assert len(frame_candidates) == 1, "expected a unique rectangular fence"

    frame_color, frame_bbox = frame_candidates[0]
    orientation, ordered_colors, flag_positions = _flags_from_frame(g, frame_bbox, frame_color, bg)

    objects = []
    for color in ordered_colors:
        flag_r, flag_c = flag_positions[int(color)]
        coords = np.argwhere(g == int(color))
        object_coords = np.array(
            [(int(r), int(c)) for r, c in coords if not (int(r) == flag_r and int(c) == flag_c)],
            dtype=int,
        )
        assert object_coords.size > 0, "each flag color must have one outside object"
        r0, c0, r1, c1 = _bbox(object_coords)
        rel = object_coords - np.array([r0, c0], dtype=int)
        objects.append(
            {
                "color": int(color),
                "coords": object_coords,
                "rel": rel,
                "bbox": (r0, c0, r1, c1),
                "shape": (r1 - r0 + 1, c1 - c0 + 1),
            }
        )

    out = g.copy()
    for obj in objects:
        coords = obj["coords"]
        out[coords[:, 0], coords[:, 1]] = int(bg)

    prev_bbox = None
    for index, obj in enumerate(objects):
        if index == 0:
            new_r0, new_c0, new_r1, new_c1 = obj["bbox"]
        elif orientation == "horizontal":
            assert prev_bbox is not None
            h, w = obj["shape"]
            new_r0 = int(prev_bbox[0])
            new_c0 = int(prev_bbox[3]) + 1
            new_r1 = new_r0 + h - 1
            new_c1 = new_c0 + w - 1
        else:
            assert prev_bbox is not None
            h, w = obj["shape"]
            new_r0 = int(prev_bbox[2]) + 1
            new_c0 = int(prev_bbox[1])
            new_r1 = new_r0 + h - 1
            new_c1 = new_c0 + w - 1

        assert 0 <= new_r0 <= new_r1 < g.shape[0]
        assert 0 <= new_c0 <= new_c1 < g.shape[1]
        for dr, dc in obj["rel"]:
            rr = int(new_r0 + int(dr))
            cc = int(new_c0 + int(dc))
            assert int(out[rr, cc]) == int(bg), "target placement must land on background"
            out[rr, cc] = int(obj["color"])
        prev_bbox = (int(new_r0), int(new_c0), int(new_r1), int(new_c1))

    return out


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(3, 5)
    orientation = random.choice(["horizontal", "vertical"])
    colors = random.sample(OBJECT_COLORS, count)
    objects = [{"color": int(color), "mask": _random_object_mask()} for color in colors]

    if orientation == "horizontal":
        chain_h = max(obj["mask"].shape[0] for obj in objects)
        chain_w = sum(obj["mask"].shape[1] for obj in objects)
        inner_h = random.randint(3, 5)
        inner_w = random.randint(max(2 * count - 1, 6), min(12, 2 * count + 6))
        fence_h = inner_h + 2
        fence_w = inner_w + 2
        min_h = chain_h + fence_h + 5
        min_w = max(chain_w, fence_w) + 4
        if min_h > 30 or min_w > 30:
            return None
        H = random.randint(max(18, min_h), 30)
        W = random.randint(max(18, min_w), 30)
        chain_on_top = random.choice([True, False])
        final_c0 = random.randint(1, W - chain_w - 1)
        fence_c0 = random.randint(1, W - fence_w - 1)
        if chain_on_top:
            final_r0 = random.randint(1, min(3, H - chain_h - 1))
            fence_r_lo = final_r0 + chain_h + 2
            fence_r_hi = H - fence_h - 1
            if fence_r_lo > fence_r_hi:
                return None
            fence_r0 = random.randint(fence_r_lo, fence_r_hi)
        else:
            fence_r_hi = H - chain_h - fence_h - 3
            if fence_r_hi < 1:
                return None
            fence_r0 = random.randint(1, fence_r_hi)
            final_r_lo = fence_r0 + fence_h + 2
            final_r_hi = H - chain_h - 1
            if final_r_lo > final_r_hi:
                return None
            final_r0 = random.randint(final_r_lo, final_r_hi)
        final_bbox = (final_r0, final_c0, final_r0 + chain_h - 1, final_c0 + chain_w - 1)
    else:
        chain_h = sum(obj["mask"].shape[0] for obj in objects)
        chain_w = max(obj["mask"].shape[1] for obj in objects)
        inner_h = random.randint(max(2 * count - 1, 6), min(12, 2 * count + 6))
        inner_w = random.randint(3, 5)
        fence_h = inner_h + 2
        fence_w = inner_w + 2
        min_h = max(chain_h, fence_h) + 4
        min_w = chain_w + fence_w + 5
        if min_h > 30 or min_w > 30:
            return None
        H = random.randint(max(18, min_h), 30)
        W = random.randint(max(18, min_w), 30)
        chain_on_left = random.choice([True, False])
        final_r0 = random.randint(1, H - chain_h - 1)
        fence_r0 = random.randint(1, H - fence_h - 1)
        if chain_on_left:
            final_c0 = random.randint(1, min(3, W - chain_w - 1))
            fence_c_lo = final_c0 + chain_w + 2
            fence_c_hi = W - fence_w - 1
            if fence_c_lo > fence_c_hi:
                return None
            fence_c0 = random.randint(fence_c_lo, fence_c_hi)
        else:
            fence_c_hi = W - chain_w - fence_w - 3
            if fence_c_hi < 1:
                return None
            fence_c0 = random.randint(1, fence_c_hi)
            final_c_lo = fence_c0 + fence_w + 2
            final_c_hi = W - chain_w - 1
            if final_c_lo > final_c_hi:
                return None
            final_c0 = random.randint(final_c_lo, final_c_hi)
        final_bbox = (final_r0, final_c0, final_r0 + chain_h - 1, final_c0 + chain_w - 1)

    grid = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    _reserve_rect(occupied, frame_r0 := fence_r0, frame_c0 := fence_c0, fence_h, fence_w, pad=1)
    _reserve_rect(
        occupied,
        final_bbox[0],
        final_bbox[1],
        final_bbox[2] - final_bbox[0] + 1,
        final_bbox[3] - final_bbox[1] + 1,
        pad=1,
    )

    frame_r1 = frame_r0 + fence_h - 1
    frame_c1 = frame_c0 + fence_w - 1
    grid[frame_r0, frame_c0 : frame_c1 + 1] = FENCE_COLOR
    grid[frame_r1, frame_c0 : frame_c1 + 1] = FENCE_COLOR
    grid[frame_r0 : frame_r1 + 1, frame_c0] = FENCE_COLOR
    grid[frame_r0 : frame_r1 + 1, frame_c1] = FENCE_COLOR

    if orientation == "horizontal":
        flag_row = frame_r0 + 1 + random.randint(0, inner_h - 1)
        start_offset = random.randint(0, inner_w - (2 * count - 1))
        flag_cells = [(flag_row, frame_c0 + 1 + start_offset + 2 * i) for i in range(count)]
    else:
        flag_col = frame_c0 + 1 + random.randint(0, inner_w - 1)
        start_offset = random.randint(0, inner_h - (2 * count - 1))
        flag_cells = [(frame_r0 + 1 + start_offset + 2 * i, flag_col) for i in range(count)]

    for (r, c), color in zip(flag_cells, colors):
        grid[r, c] = int(color)

    if orientation == "horizontal":
        cursor_r = final_bbox[0]
        cursor_c = final_bbox[1]
        for index, obj in enumerate(objects):
            h, w = obj["mask"].shape
            obj["final_pos"] = (cursor_r, cursor_c)
            if index == 0:
                _paint_mask(grid, obj["mask"], cursor_r, cursor_c, obj["color"])
            cursor_c += w
    else:
        cursor_r = final_bbox[0]
        cursor_c = final_bbox[1]
        for index, obj in enumerate(objects):
            h, w = obj["mask"].shape
            obj["final_pos"] = (cursor_r, cursor_c)
            if index == 0:
                _paint_mask(grid, obj["mask"], cursor_r, cursor_c, obj["color"])
            cursor_r += h

    first_h, first_w = objects[0]["mask"].shape
    _reserve_rect(occupied, objects[0]["final_pos"][0], objects[0]["final_pos"][1], first_h, first_w, pad=1)

    for obj in objects[1:]:
        h, w = obj["mask"].shape
        pos = _random_free_position(occupied, h, w, pad=1)
        if pos is None:
            return None
        r0, c0 = pos
        _paint_mask(grid, obj["mask"], r0, c0, obj["color"])
        _reserve_rect(occupied, r0, c0, h, w, pad=1)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
