from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
DIVIDER = 4
MARKER = 9
OBJECT_COLORS = (1, 2, 3, 6)
GRID_HEIGHTS = (17, 19, 21, 23, 25)
GRID_WIDTHS = (19, 21, 23, 25, 27)
GEN_ATTEMPTS = 500
MASK_ATTEMPTS = 200
PLACEMENT_ATTEMPTS = 120
MIN_ROW_GAP = 2


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _mask_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, pts[0]))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(pts)


def _components(mask: np.ndarray) -> list[np.ndarray]:
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
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _mask_from_coords(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _side_gap_rows(col_mask: np.ndarray) -> list[int]:
    rows = []
    for r in range(1, len(col_mask) - 1):
        if not col_mask[r] and col_mask[r - 1] and col_mask[r + 1]:
            rows.append(r)
    return rows


def _infer_notch_direction(local_mask: np.ndarray) -> int:
    left_gaps = _side_gap_rows(local_mask[:, 0])
    right_gaps = _side_gap_rows(local_mask[:, -1])
    has_left = len(left_gaps) == 1 and int(np.count_nonzero(~local_mask[:, 0])) == 1
    has_right = len(right_gaps) == 1 and int(np.count_nonzero(~local_mask[:, -1])) == 1
    assert has_left != has_right, "each object must have a unique left or right notch"
    return -1 if has_left else 1


def _shear_coords(
    coords: np.ndarray,
    direction: int,
    divider_col: int,
    height: int,
    width: int,
) -> np.ndarray:
    del height, width
    bottom = int(coords[:, 0].max())
    out: list[tuple[int, int]] = []
    for r, c in coords:
        shift = bottom - int(r)
        new_c = int(c) + direction * shift
        if 0 <= new_c < divider_col:
            out.append((int(r), new_c))
    assert out, "shear must leave at least one pixel"
    return np.asarray(out, dtype=int)


def _parse_objects(grid: np.ndarray) -> list[dict]:
    mask = (grid != BG) & (grid != DIVIDER) & (grid != MARKER)
    comps = _components(mask)
    assert 2 <= len(comps) <= 3, "expected 2-3 left-side objects"

    objects = []
    for coords in comps:
        assert np.all(coords[:, 1] < grid.shape[1] // 2), "source objects must stay left of the divider"
        bbox = _bbox(coords)
        r0, c0, r1, c1 = bbox
        color = int(grid[coords[0, 0], coords[0, 1]])
        local_mask = _mask_from_coords(coords, bbox)
        direction = _infer_notch_direction(local_mask)
        objects.append(
            {
                "coords": coords,
                "bbox": bbox,
                "color": color,
                "direction": direction,
                "center_row": (r0 + r1) / 2.0,
                "top": r0,
                "local_mask": local_mask,
            }
        )

    objects.sort(key=lambda obj: (obj["center_row"], obj["top"], obj["bbox"][1]))
    return objects


def _apply_transform(grid: np.ndarray, directions: list[int] | None = None, reflect: bool = True) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    cr, cc = h // 2, w // 2
    out = g.copy()
    objects = _parse_objects(g)

    for obj in objects:
        for r, c in obj["coords"]:
            out[r, c] = BG

    if directions is None:
        directions = [int(obj["direction"]) for obj in objects]
    assert len(directions) == len(objects)

    sheared_sets: list[tuple[int, np.ndarray]] = []
    for obj, direction in zip(objects, directions):
        sheared = _shear_coords(
            obj["coords"],
            int(direction),
            cc,
            h,
            w,
        )
        sheared_sets.append((obj["color"], sheared))
        for r, c in sheared:
            if out[r, c] != MARKER and out[r, c] != DIVIDER:
                out[r, c] = obj["color"]

    if reflect:
        for obj, (color, sheared) in zip(objects, sheared_sets):
            del obj
            for r, c in sheared:
                rr = 2 * cr - int(r)
                rc = 2 * cc - int(c)
                if 0 <= rr < h and 0 <= rc < w and rc != cc and out[rr, rc] != MARKER:
                    out[rr, rc] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shear each left-side object according to its border notch, then complete its 180-degree counterpart across the center marker.

    Input structure:
    - Black background surrounds a vertical yellow divider in the center column, with maroon center markers at the left edge, center, and right edge of the center row.
    - Two or three small 4-connected colored objects appear only on the left side of the divider.
    - Each object's bounding box has exactly one flanked black gap on either its left border or its right border.

    Rule:
    - Keep the yellow divider and the three maroon center markers unchanged.
    - For each left-side object, use the notch side to choose a shear direction: left-border notch means shear left, right-border notch means shear right; the bottom row is the anchor and each row shifts by its vertical distance from that bottom row.
    - Replace the original left-side object with its sheared version, dropping pixels that leave the grid or reach the divider, then paint the 180-degree rotated copy of the sheared pixels across the center marker on the right side.

    Key insight:
    - The notch does not merely label the object: it determines the shear direction, and that changed shape is exactly what gets point-reflected to the other side.
    """
    return _apply_transform(grid)


def _has_any_symmetry(mask: np.ndarray) -> bool:
    return (
        np.array_equal(mask, np.fliplr(mask))
        or np.array_equal(mask, np.flipud(mask))
        or np.array_equal(mask, np.rot90(mask, 2))
    )


def _sample_mask(direction: int) -> np.ndarray | None:
    for _ in range(MASK_ATTEMPTS):
        h = random.choice((3, 4))
        w = random.choice((2, 3, 4))
        mask = np.ones((h, w), dtype=bool)
        notch_row = random.randint(1, h - 2)
        notch_col = 0 if direction < 0 else w - 1
        mask[notch_row, notch_col] = False

        base_area = int(mask.sum())
        target_area = random.randint(4, 8)
        remove_needed = base_area - target_area
        interior = [(r, c) for r in range(h) for c in range(1, w - 1)]
        if remove_needed < 0 or remove_needed > len(interior):
            continue
        if remove_needed:
            for r, c in random.sample(interior, remove_needed):
                mask[r, c] = False

        if int(mask.sum()) < 4 or int(mask.sum()) > 8:
            continue
        if not _mask_connected(mask):
            continue
        if _infer_notch_direction(mask) != direction:
            continue
        if _has_any_symmetry(mask):
            continue

        local_coords = np.argwhere(mask)
        sheared_local = _shear_coords(local_coords, direction, divider_col=999, height=h, width=w)
        if len(sheared_local) < 3:
            continue
        if set(map(tuple, local_coords)) == set(map(tuple, sheared_local)):
            continue
        return mask
    return None


def _distribute_row_starts(heights: list[int], total_h: int) -> list[int] | None:
    required = sum(heights) + MIN_ROW_GAP * (len(heights) - 1)
    if required > total_h:
        return None
    slack = total_h - required
    gaps = [0] * (len(heights) + 1)
    for _ in range(slack):
        gaps[random.randrange(len(gaps))] += 1

    starts: list[int] = []
    r = gaps[0]
    for i, height in enumerate(heights):
        starts.append(r)
        r += height
        if i + 1 < len(heights):
            r += MIN_ROW_GAP + gaps[i + 1]
    return starts


def _coords_from_mask(mask: np.ndarray, r0: int, c0: int) -> np.ndarray:
    coords = np.argwhere(mask).copy()
    coords[:, 0] += r0
    coords[:, 1] += c0
    return coords


def _candidate_columns(
    mask: np.ndarray,
    r0: int,
    cc: int,
    cr: int,
    direction: int,
    want_clip: bool,
) -> list[int]:
    h, w = mask.shape
    cols = []
    local = np.argwhere(mask)
    for c0 in range(0, cc - w + 1):
        coords = local.copy()
        coords[:, 0] += r0
        coords[:, 1] += c0
        sheared = _shear_coords(coords, direction, divider_col=cc, height=h, width=w)
        kept = {tuple(map(int, rc)) for rc in sheared}
        if len(kept) != len(sheared):
            continue
        if len(kept) < 3:
            continue
        lost = len(coords) - len(sheared)
        if want_clip and lost == 0:
            continue
        if not want_clip and lost != 0:
            continue
        if any((r == cr and c == 0) for r, c in kept):
            continue
        reflected = {(2 * cr - r, 2 * cc - c) for r, c in kept}
        if any(rr == cr and rc == 2 * cc for rr, rc in reflected):
            continue
        cols.append(c0)
    return cols


def _objects_touch(coords_a: np.ndarray, coords_b: np.ndarray) -> bool:
    cells_b = {tuple(map(int, rc)) for rc in coords_b}
    for r, c in coords_a:
        if (int(r), int(c)) in cells_b:
            return True
        for nr, nc in _neighbors8(int(r), int(c)):
            if (nr, nc) in cells_b:
                return True
    return False


def _build_input() -> np.ndarray | None:
    h = random.choice(GRID_HEIGHTS)
    w = random.choice(GRID_WIDTHS)
    cr, cc = h // 2, w // 2

    obj_count = random.choice((2, 3))
    if obj_count == 2:
        directions = [-1, 1]
    else:
        directions = random.choice(([-1, -1, 1], [-1, 1, 1]))
        random.shuffle(directions)

    masks = []
    for direction in directions:
        mask = _sample_mask(direction)
        if mask is None:
            return None
        masks.append(mask)

    starts = _distribute_row_starts([mask.shape[0] for mask in masks], h)
    if starts is None:
        return None

    left_indices = [i for i, direction in enumerate(directions) if direction < 0]
    right_indices = [i for i, direction in enumerate(directions) if direction > 0]
    clip_left = random.choice(left_indices) if left_indices and random.random() < 0.35 else None
    clip_right = random.choice(right_indices) if right_indices and random.random() < 0.35 else None

    objects = []
    for i, (mask, r0, direction) in enumerate(zip(masks, starts, directions)):
        cols = _candidate_columns(
            mask,
            r0,
            cc,
            cr,
            direction,
            want_clip=(i == clip_left or i == clip_right),
        )
        if not cols:
            return None
        c0 = random.choice(cols)
        coords = _coords_from_mask(mask, r0, c0)
        if any(_objects_touch(coords, other["coords"]) for other in objects):
            return None
        objects.append(
            {
                "coords": coords,
                "color": OBJECT_COLORS[i],
                "direction": direction,
            }
        )

    grid = np.zeros((h, w), dtype=int)
    grid[:, cc] = DIVIDER
    grid[cr, 0] = MARKER
    grid[cr, cc] = MARKER
    grid[cr, w - 1] = MARKER

    colors = random.sample(OBJECT_COLORS, len(objects))
    for obj, color in zip(objects, colors):
        for r, c in obj["coords"]:
            grid[r, c] = color

    return grid


def _semantic_checks(grid: np.ndarray) -> bool:
    h, w = grid.shape
    cr, cc = h // 2, w // 2
    objects = _parse_objects(grid)
    directions = [int(obj["direction"]) for obj in objects]
    if -1 not in directions or 1 not in directions:
        return False

    centers = [obj["center_row"] for obj in objects]
    if len(set(centers)) != len(centers):
        return False

    expected = main(grid)
    if np.array_equal(expected, grid):
        return False

    if np.any(expected[:, cc] != grid[:, cc]):
        return False
    if not np.array_equal(expected[cr, [0, cc, w - 1]], np.array([MARKER, MARKER, MARKER])):
        return False
    if not np.any(expected[:, cc + 1 :] != BG):
        return False

    sheared_only = _apply_transform(grid, reflect=False)
    if np.array_equal(expected, sheared_only):
        return False

    original_reflect = grid.copy()
    for obj in objects:
        for r, c in obj["coords"]:
            rr = 2 * cr - int(r)
            rc = 2 * cc - int(c)
            if 0 <= rr < h and 0 <= rc < w and rc != cc and original_reflect[rr, rc] != MARKER:
                original_reflect[rr, rc] = obj["color"]
    if np.array_equal(expected, original_reflect):
        return False

    if np.array_equal(expected, _apply_transform(grid, directions=[-1] * len(objects))):
        return False
    if np.array_equal(expected, _apply_transform(grid, directions=[1] * len(objects))):
        return False

    left_half = expected[:, :cc]
    right_half = expected[:, cc + 1 :]
    if np.array_equal(left_half, grid[:, :cc]):
        return False
    if np.count_nonzero(right_half) == 0:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_input()
        if grid is None:
            continue
        if _semantic_checks(grid):
            return grid
    raise RuntimeError("failed to generate a valid notch-shear-symmetry input")
