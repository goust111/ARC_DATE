from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


COLOR_COUNT = 10
DIRECTIONS = ("up", "right", "down", "left")
GEN_OBJECT_COUNTS = (4, 5)
GEN_GRID_RANGE = (22, 31)
GEN_AREA_RANGE = (7, 22)
GEN_DIM_RANGE = (4, 9)
GEN_RETRIES = 500
SHAPE_RETRIES = 250
PLACEMENT_PADDING = 1
PLACEMENT_BORDER = 1


def _most_common_color(g: np.ndarray) -> int:
    vals, counts = np.unique(g, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            comp: list[tuple[int, int]] = []
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(int(cr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return int(min(rows)), int(min(cols)), int(max(rows)), int(max(cols))


def _marker_direction(marker: tuple[int, int], bbox: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = bbox
    rr = int(marker[0] - r0)
    cc = int(marker[1] - c0)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)

    dr2 = int(2 * rr - (h - 1))
    dc2 = int(2 * cc - (w - 1))
    assert dr2 != 0 or dc2 != 0, "marker must not lie at the bbox center"

    if abs(dr2) >= abs(dc2):
        return "up" if dr2 < 0 else "down"
    return "left" if dc2 < 0 else "right"


def _normalized_stamps(g: np.ndarray, bg: int) -> list[dict[str, object]]:
    comps = _components(g != int(bg))
    assert comps, "expected at least one stamp"

    rotation_to_up = {"up": 0, "right": 1, "down": 2, "left": 3}
    stamps: list[dict[str, object]] = []
    for comp in comps:
        vals = [int(g[r, c]) for r, c in comp]
        counts = Counter(vals)
        assert len(counts) == 2, "each stamp must contain one body color and one marker color"

        marker_colors = [color for color, count in counts.items() if int(count) == 1]
        assert len(marker_colors) == 1, "each stamp must have exactly one singleton marker color"
        marker_color = int(marker_colors[0])
        body_colors = [color for color in counts if int(color) != marker_color]
        assert len(body_colors) == 1, "each stamp must have exactly one body color"
        body_color = int(body_colors[0])

        marker_cells = [(r, c) for r, c in comp if int(g[r, c]) == marker_color]
        marker = marker_cells[0]
        r0, c0, r1, c1 = _bbox(comp)
        direction = _marker_direction(marker, (r0, c0, r1, c1))

        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
        for r, c in comp:
            patch[int(r - r0), int(c - c0)] = body_color

        normalized = np.rot90(patch, int(rotation_to_up[direction]))
        stamps.append(
            {
                "area": int(len(comp)),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "patch": normalized,
            }
        )

    stamps.sort(key=lambda rec: (-int(rec["area"]), tuple(rec["bbox"])))
    return stamps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate marked stamps upright, then center-nest them from largest to smallest.

    Input structure:
    - The most common color is the background.
    - Several separated 4-connected foreground components are stamps.
    - Each stamp has one body color and exactly one differently colored marker
      pixel inside its bounding box.
    - The marker lies clearly above, right of, below, or left of the stamp
      bounding-box center, indicating the stamp's current facing direction.

    Rule:
    - For each stamp, infer its direction from the marker's position relative
      to the stamp bounding-box center.
    - Rotate the whole stamp bounding box by 0, 90, 180, or 270 degrees so the
      marker would be on the upper half, and restore the marker to the body color.
    - Sort the upright stamps by area from largest to smallest.
    - Create a square background canvas whose side equals the largest upright
      stamp extent, then draw all stamps centered on it in sorted order so later
      smaller stamps overwrite earlier larger ones.

    Key insight:
    - The singleton odd-colored pixel is an orientation marker, not decoration;
      every object must be individually unrotated before the visible area order
      determines the centered overlay.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    stamps = _normalized_stamps(g, bg)
    side = max(max(map(int, stamp["patch"].shape)) for stamp in stamps)
    out = np.full((side, side), int(bg), dtype=int)

    for stamp in stamps:
        patch = np.asarray(stamp["patch"], dtype=int)
        h, w = map(int, patch.shape)
        r0 = int((side - h) // 2)
        c0 = int((side - w) // 2)
        mask = patch != int(bg)
        target = out[r0 : r0 + h, c0 : c0 + w]
        target[mask] = patch[mask]

    return out


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    assert rows.size and cols.size, "cannot trim an empty mask"
    return mask[int(rows[0]) : int(rows[-1]) + 1, int(cols[0]) : int(cols[-1]) + 1]


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    cropped = _trim_mask(np.asarray(mask, dtype=bool))
    return tuple(sorted((int(r), int(c)) for r, c in np.argwhere(cropped)))


def _canonical_rotation_key(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    return min(_mask_key(np.rot90(mask, k)) for k in range(4))


def _is_rotation_asymmetric(mask: np.ndarray) -> bool:
    return len({_mask_key(np.rot90(mask, k)) for k in range(4)}) == 4


def _is_connected(mask: np.ndarray) -> bool:
    return len(_components(np.asarray(mask, dtype=bool))) == 1


def _direction_margin(marker: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int]:
    h, w = map(int, shape)
    r, c = map(int, marker)
    dr2 = int(2 * r - (h - 1))
    dc2 = int(2 * c - (w - 1))
    return dr2, dc2


def _grow_body(h: int, w: int, area: int, seed: tuple[int, int], forbidden: set[tuple[int, int]]) -> np.ndarray | None:
    if area < 1 or area > h * w - len(forbidden):
        return None

    body: set[tuple[int, int]] = {seed}
    while len(body) < area:
        frontier: set[tuple[int, int]] = set()
        for r, c in body:
            for nr, nc in _neighbors4(int(r), int(c)):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in body and (nr, nc) not in forbidden:
                    frontier.add((int(nr), int(nc)))
        if not frontier:
            return None
        body.add(random.choice(sorted(frontier)))

    mask = np.zeros((h, w), dtype=bool)
    for r, c in body:
        mask[int(r), int(c)] = True
    return mask


def _make_canonical_shape(area: int, used_keys: set[tuple[tuple[int, int], ...]]) -> tuple[np.ndarray, tuple[int, int]] | None:
    dims = [
        (h, w)
        for h in range(*GEN_DIM_RANGE)
        for w in range(*GEN_DIM_RANGE)
        if h * w >= area and h >= 4 and w >= 4
    ]
    random.shuffle(dims)

    for _ in range(SHAPE_RETRIES):
        if not dims:
            return None
        h, w = random.choice(dims)
        mid_choices = [c for c in range(1, w - 1)]
        random.shuffle(mid_choices)
        for mid in mid_choices:
            marker = (0, int(mid))
            seed = (1, int(mid))
            body = _grow_body(h, w, int(area - 1), seed, {marker})
            if body is None:
                continue

            full = body.copy()
            full[marker] = True

            rows = np.where(full.any(axis=1))[0]
            cols = np.where(full.any(axis=0))[0]
            r0, c0 = int(rows[0]), int(cols[0])
            cropped = full[r0 : int(rows[-1]) + 1, c0 : int(cols[-1]) + 1]
            marker_rel = (int(marker[0] - r0), int(marker[1] - c0))

            body_cropped = cropped.copy()
            body_cropped[marker_rel] = False
            if not _is_connected(body_cropped):
                continue

            dr2, dc2 = _direction_margin(marker_rel, cropped.shape)
            if not (dr2 < 0 and abs(dr2) > abs(dc2)):
                continue
            if min(map(int, cropped.shape)) < 3:
                continue
            if not _is_rotation_asymmetric(cropped):
                continue

            key = _canonical_rotation_key(cropped)
            if key in used_keys:
                continue
            used_keys.add(key)
            return cropped, marker_rel

    return None


def _paint_stamp(mask: np.ndarray, marker: tuple[int, int], body_color: int, marker_color: int, bg: int) -> np.ndarray:
    patch = np.full(mask.shape, int(bg), dtype=int)
    patch[np.asarray(mask, dtype=bool)] = int(body_color)
    patch[int(marker[0]), int(marker[1])] = int(marker_color)
    return patch


def _dilate8(mask: np.ndarray) -> np.ndarray:
    src = np.asarray(mask, dtype=bool)
    out = src.copy()
    out[1:, :] |= src[:-1, :]
    out[:-1, :] |= src[1:, :]
    out[:, 1:] |= src[:, :-1]
    out[:, :-1] |= src[:, 1:]
    out[1:, 1:] |= src[:-1, :-1]
    out[:-1, :-1] |= src[1:, 1:]
    out[1:, :-1] |= src[:-1, 1:]
    out[:-1, 1:] |= src[1:, :-1]
    return out


def _padded(mask: np.ndarray, steps: int) -> np.ndarray:
    out = np.asarray(mask, dtype=bool)
    for _ in range(int(steps)):
        out = _dilate8(out)
    return out


def _place_patch(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray, bg: int) -> bool:
    h, w = map(int, patch.shape)
    H, W = map(int, grid.shape)
    if h + 2 * PLACEMENT_BORDER > H or w + 2 * PLACEMENT_BORDER > W:
        return False

    blocked = _padded(occupied, PLACEMENT_PADDING)
    mask = patch != int(bg)
    origins = [
        (r, c)
        for r in range(PLACEMENT_BORDER, H - h - PLACEMENT_BORDER + 1)
        for c in range(PLACEMENT_BORDER, W - w - PLACEMENT_BORDER + 1)
    ]
    random.shuffle(origins)
    for r0, c0 in origins:
        if bool((blocked[r0 : r0 + h, c0 : c0 + w] & mask).any()):
            continue
        target = grid[r0 : r0 + h, c0 : c0 + w]
        target[mask] = patch[mask]
        occupied[r0 : r0 + h, c0 : c0 + w] |= mask
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(GEN_OBJECT_COUNTS)
    bg = random.randrange(COLOR_COUNT)
    marker_color = random.choice([c for c in range(COLOR_COUNT) if c != bg])
    body_colors = random.sample([c for c in range(COLOR_COUNT) if c not in {bg, marker_color}], n)
    areas = random.sample(range(*GEN_AREA_RANGE), n)

    directions = list(DIRECTIONS)
    if n > len(DIRECTIONS):
        directions.extend(random.choices(DIRECTIONS, k=n - len(DIRECTIONS)))
    random.shuffle(directions)

    used_keys: set[tuple[tuple[int, int], ...]] = set()
    stamps: list[np.ndarray] = []
    rotate_to_direction = {"up": 0, "right": 3, "down": 2, "left": 1}
    for area, body_color, direction in zip(areas, body_colors, directions):
        shape = _make_canonical_shape(int(area), used_keys)
        if shape is None:
            return None
        mask, marker = shape
        upright = _paint_stamp(mask, marker, int(body_color), int(marker_color), int(bg))
        stamps.append(np.rot90(upright, int(rotate_to_direction[direction])))

    H = random.randrange(*GEN_GRID_RANGE)
    W = random.randrange(*GEN_GRID_RANGE)
    grid = np.full((H, W), int(bg), dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    order = list(range(n))
    random.shuffle(order)
    order.sort(key=lambda i: int(np.count_nonzero(stamps[i] != bg)), reverse=True)
    for idx in order:
        if not _place_patch(grid, occupied, stamps[idx], int(bg)):
            return None

    counts = Counter(map(int, grid.ravel()))
    if counts[int(bg)] != max(counts.values()):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.ndim != 2 or max(map(int, grid.shape)) > 30 or max(map(int, out.shape)) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
