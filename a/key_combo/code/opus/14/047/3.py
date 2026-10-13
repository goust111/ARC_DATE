import random
from collections import Counter, deque

import numpy as np


YELLOW = 4
ORANGE = 7
BACKGROUND = 0
INPUT_COLORS = [1, 2, 3, 5, 6, 8, 9]
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 400


def get_bg(grid: np.ndarray) -> int:
    counts = Counter(map(int, np.asarray(grid).ravel()))
    return int(counts.most_common(1)[0][0])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= int(r) < int(h) and 0 <= int(c) < int(w)


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _frame_border_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = bbox
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _center_rows_cols(h: int, w: int) -> tuple[set[int], set[int]]:
    if (h - 2) % 2 == 1:
        center_rows = {h // 2}
    else:
        center_rows = {h // 2 - 1, h // 2}
    if (w - 2) % 2 == 1:
        center_cols = {w // 2}
    else:
        center_cols = {w // 2 - 1, w // 2}
    return center_rows, center_cols


def _is_non_center_local(local_r: int, local_c: int, h: int, w: int) -> bool:
    center_rows, center_cols = _center_rows_cols(h, w)
    return not (int(local_r) in center_rows and int(local_c) in center_cols)


def _max_ring_depth_local(local_control: tuple[int, int], h: int, w: int) -> int:
    cr, cc = local_control
    return int(max(cr - 1, (h - 2) - cr, cc - 1, (w - 2) - cc))


def _find_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    objects: list[dict] = []
    for color in sorted(set(map(int, g.ravel())) - {bg}):
        color_mask = g == int(color)
        candidates = []
        for comp in components_mask(color_mask):
            r0, c0, r1, c1 = bbox_coords(comp)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 3 or w < 3:
                continue
            bbox = (r0, c0, r1, c1)
            border_mask = _frame_border_mask(g.shape, bbox)
            if not np.all(g[border_mask] == int(color)):
                continue
            bbox_mask = np.zeros_like(color_mask, dtype=bool)
            bbox_mask[r0 : r1 + 1, c0 : c1 + 1] = True
            interior_mask = bbox_mask & ~border_mask
            inside_color = np.argwhere(color_mask & interior_mask)
            outside_color = np.argwhere(color_mask & ~bbox_mask)
            border_count = 2 * (h + w) - 4
            if int(color_mask.sum()) != border_count + 2:
                continue
            if inside_color.shape[0] != 1 or outside_color.shape[0] != 1:
                continue
            interior = g[r0 + 1 : r1, c0 + 1 : c1]
            if interior.size == 0:
                continue
            if not np.all((interior == bg) | (interior == int(color))):
                continue
            local_control = tuple(map(int, inside_color[0]))
            if not _is_non_center_local(local_control[0] - r0, local_control[1] - c0, h, w):
                continue
            candidates.append(
                {
                    "color": int(color),
                    "bbox": bbox,
                    "control": tuple(map(int, inside_color[0])),
                    "marker": tuple(map(int, outside_color[0])),
                }
            )
        assert len(candidates) == 1, f"expected exactly one frame object for color {color}"
        objects.append(candidates[0])
    assert 2 <= len(objects) <= 3, "expected 2-3 rectangle objects"
    return objects


def _sightline_info(grid: np.ndarray, marker: tuple[int, int], color: int, bg: int) -> tuple[tuple[int, int], tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    hits = []
    for dr, dc in DIRS:
        r, c = marker
        last_bg = None
        steps = 0
        while True:
            r += dr
            c += dc
            if not in_bounds(g.shape, r, c):
                break
            value = int(g[r, c])
            if value == bg:
                last_bg = (r, c)
                steps += 1
                continue
            if value == int(color) and steps >= 1:
                hits.append(((dr, dc), last_bg))
            break
    assert len(hits) == 1, "marker must have exactly one clear sightline to its frame"
    return hits[0]


def _contour_lengths_from_start(
    bbox: tuple[int, int, int, int], start: tuple[int, int]
) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    sr, sc = start
    if sr == r0 - 1 and c0 <= sc <= c1:
        a = sc - c0
        b = c1 - sc
    elif sr == r1 + 1 and c0 <= sc <= c1:
        a = sc - c0
        b = c1 - sc
    elif sc == c0 - 1 and r0 <= sr <= r1:
        a = sr - r0
        b = r1 - sr
    elif sc == c1 + 1 and r0 <= sr <= r1:
        a = sr - r0
        b = r1 - sr
    else:
        raise AssertionError("contour start must lie on exactly one exterior side segment")
    long_len = int(max(a, b))
    short_len = int(min(a, b))
    assert long_len != short_len, "two contour directions must not tie"
    return long_len, short_len


def main(grid: np.ndarray) -> np.ndarray:
    """
    Measure the longer exterior side-run from each marker and use it as the depth of concentric interior ring filling.

    Input structure:
    - Black background contains 2-3 disjoint one-pixel rectangular frames, each in a distinct color.
    - Every frame has exactly one same-colored control pixel inside it and exactly one same-colored marker pixel outside it.
    - Each marker has exactly one horizontal or vertical background sightline to its own frame.

    Rule:
    - For each frame, slide its outer marker along that unique sightline to the last background cell before the frame wall.
    - From that arrival cell, compare the two possible 4-neighbor walks along the same exterior side segment that stay adjacent to the frame; let the longer walk length be L.
    - Inside the frame, keep the wall, marker, and control pixel unchanged, and recolor only background pixels whose Chebyshev distance d from the control pixel satisfies 1 <= d <= L.
    - Paint odd d with yellow (4), even d with orange (7); leave all interior background pixels with d > L black.

    Key insight:
    The outside marker does not indicate where to paint directly; it encodes a numeric depth by measuring the longer accessible contour run, and that depth controls how many concentric square rings are filled around the inner control pixel.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()
    objects = _find_objects(g)
    for obj in objects:
        color = int(obj["color"])
        bbox = obj["bbox"]
        control = obj["control"]
        marker = obj["marker"]
        _direction, contour_start = _sightline_info(g, marker, color, bg)
        depth, _shorter = _contour_lengths_from_start(bbox, contour_start)
        r0, c0, r1, c1 = bbox
        cr, cc = control
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if int(g[r, c]) != bg:
                    continue
                dist = max(abs(r - cr), abs(c - cc))
                if 1 <= dist <= depth:
                    out[r, c] = YELLOW if dist % 2 == 1 else ORANGE
    return out


def _slot_bounds(h: int, w: int) -> list[tuple[int, int, int, int]]:
    mid_r = h // 2
    mid_c = w // 2
    return [
        (1, mid_r - 2, 1, mid_c - 2),
        (1, mid_r - 2, mid_c + 1, w - 2),
        (mid_r + 1, h - 2, 1, mid_c - 2),
        (mid_r + 1, h - 2, mid_c + 1, w - 2),
    ]


def _offset_candidates(side_len: int, max_depth: int) -> list[tuple[int, int]]:
    out = []
    for idx in range(side_len):
        a = idx
        b = side_len - 1 - idx
        depth = max(a, b)
        if a != b and 3 <= depth <= max_depth:
            out.append((idx, depth))
    return out


def _local_control_candidates(h: int, w: int, min_cap_exclusive: int) -> list[tuple[int, int]]:
    out = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if not _is_non_center_local(r, c, h, w):
                continue
            cap = _max_ring_depth_local((r, c), h, w)
            if cap > min_cap_exclusive:
                out.append((r, c))
    return out


def _top_left_ranges(
    slot: tuple[int, int, int, int], h: int, w: int, direction: tuple[int, int], gap: int
) -> tuple[range, range] | None:
    sr0, sr1, sc0, sc1 = slot
    rmin = sr0 + 1
    rmax = sr1 - h
    cmin = sc0 + 1
    cmax = sc1 - w
    dr, dc = direction
    if dr == -1:
        rmin = max(rmin, sr0 + gap + 1)
    elif dr == 1:
        rmax = min(rmax, sr1 - h - gap)
    elif dc == -1:
        cmin = max(cmin, sc0 + gap + 1)
    elif dc == 1:
        cmax = min(cmax, sc1 - w - gap)
    if rmin > rmax or cmin > cmax:
        return None
    return range(rmin, rmax + 1), range(cmin, cmax + 1)


def _sample_object_for_slot(
    rng: random.Random, slot: tuple[int, int, int, int], color: int
) -> dict | None:
    directions = DIRS[:]
    rng.shuffle(directions)
    for _ in range(200):
        h = rng.choice([9, 10])
        w = rng.choice([9, 10])
        direction = rng.choice(directions)
        gap = rng.choice([1, 2])
        ranges = _top_left_ranges(slot, h, w, direction, gap)
        if ranges is None:
            continue
        max_possible_depth = max(h - 3, w - 3) - 1
        side_len = w if direction[0] != 0 else h
        offset_choices = _offset_candidates(side_len, max_possible_depth)
        if not offset_choices:
            continue
        offset, depth = rng.choice(offset_choices)
        control_choices = _local_control_candidates(h, w, depth)
        if not control_choices:
            continue
        local_control = rng.choice(control_choices)
        r0 = rng.choice(list(ranges[0]))
        c0 = rng.choice(list(ranges[1]))
        r1 = r0 + h - 1
        c1 = c0 + w - 1
        if direction == (-1, 0):
            marker = (r0 - gap - 1, c0 + offset)
        elif direction == (1, 0):
            marker = (r1 + gap + 1, c0 + offset)
        elif direction == (0, -1):
            marker = (r0 + offset, c0 - gap - 1)
        else:
            marker = (r0 + offset, c1 + gap + 1)
        return {
            "color": int(color),
            "bbox": (r0, c0, r1, c1),
            "control": (r0 + local_control[0], c0 + local_control[1]),
            "marker": marker,
            "direction": direction,
            "gap": int(gap),
            "depth": int(depth),
        }
    return None


def _draw_object(grid: np.ndarray, obj: dict) -> None:
    r0, c0, r1, c1 = obj["bbox"]
    color = int(obj["color"])
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    cr, cc = obj["control"]
    mr, mc = obj["marker"]
    grid[cr, cc] = color
    grid[mr, mc] = color


def _validate_generated_grid(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    assert get_bg(g) == BACKGROUND
    objects = _find_objects(g)
    assert len(objects) in {2, 3}
    axes = {("vertical" if _sightline_info(g, obj["marker"], obj["color"], BACKGROUND)[0][0] != 0 else "horizontal") for obj in objects}
    assert len(axes) == 2, "each sample should include both horizontal and vertical sightlines"
    out = main(g)
    assert out.shape == g.shape
    assert np.any(out != g), "output must change the grid"
    for obj in objects:
        color = int(obj["color"])
        bbox = obj["bbox"]
        control = obj["control"]
        marker = obj["marker"]
        direction, contour_start = _sightline_info(g, marker, color, BACKGROUND)
        depth, shorter = _contour_lengths_from_start(bbox, contour_start)
        r0, c0, r1, c1 = bbox
        assert r1 - r0 - 1 >= 7 and c1 - c0 - 1 >= 7
        assert depth >= 3 and depth > shorter
        local_control = (control[0] - r0, control[1] - c0)
        max_depth = _max_ring_depth_local(local_control, r1 - r0 + 1, c1 - c0 + 1)
        assert depth < max_depth, "there must be deeper unfilled interior than the measured contour length"
        filled = 0
        far_black = 0
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if (r, c) == control:
                    assert out[r, c] == color
                    continue
                if int(g[r, c]) != BACKGROUND:
                    continue
                d = max(abs(r - control[0]), abs(c - control[1]))
                if 1 <= d <= depth:
                    expected = YELLOW if d % 2 == 1 else ORANGE
                    assert int(out[r, c]) == expected
                    filled += 1
                else:
                    assert int(out[r, c]) == BACKGROUND
                    if d > depth:
                        far_black += 1
        assert filled > 0
        assert far_black > 0
        assert g[marker] == color and out[marker] == color
        assert direction in DIRS


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        h = random.randint(28, 30)
        w = random.randint(28, 30)
        grid = np.zeros((h, w), dtype=int)
        slots = _slot_bounds(h, w)
        n_objects = random.choice([2, 3])
        chosen_slots = random.sample(range(4), n_objects)
        colors = random.sample(INPUT_COLORS, n_objects)
        objects = []
        ok = True
        for slot_idx, color in zip(chosen_slots, colors):
            obj = _sample_object_for_slot(random, slots[slot_idx], color)
            if obj is None:
                ok = False
                break
            objects.append(obj)
            _draw_object(grid, obj)
        if not ok:
            continue
        try:
            _validate_generated_grid(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate valid input for puzzle 3")
