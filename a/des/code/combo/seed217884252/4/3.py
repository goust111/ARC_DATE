from __future__ import annotations

from collections import deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

ORANGE = 7
BLUE = 1
RED = 2
GREY = 5

GRID_HEIGHT_RANGE = (15, 23)
GRID_WIDTH_RANGE = (15, 23)
OBJECT_COUNT_RANGE = (2, 5)
PLACEMENT_MARGIN = 2
PLACEMENT_GAP = 1
SHAPE_ATTEMPTS = 200
GENERATION_ATTEMPTS = 400


def _randint(rng: np.random.Generator, low: int, high: int) -> int:
    assert low <= high
    return int(rng.integers(low, high + 1))


def _dominant_color(grid: np.ndarray) -> int:
    flat = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(flat, minlength=10).argmax())


def _components_from_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    m = np.asarray(mask, dtype=bool)
    h, w = m.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not m[r, c] or seen[r, c]:
                continue

            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            coords: list[tuple[int, int]] = []

            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or not m[nr, nc]:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            comps.append(coords)

    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert len(coords) > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def _hole_count(mask: np.ndarray) -> int:
    shape = np.asarray(mask, dtype=bool)
    h, w = shape.shape
    bg = ~shape
    seen = np.zeros((h, w), dtype=bool)
    holes = 0

    for r in range(h):
        for c in range(w):
            if not bg[r, c] or seen[r, c]:
                continue

            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            touches_border = False

            while q:
                cr, cc = q.popleft()
                if cr == 0 or cc == 0 or cr == h - 1 or cc == w - 1:
                    touches_border = True
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or not bg[nr, nc]:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            if not touches_border:
                holes += 1

    return holes


def _random_transform(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    transformed = np.asarray(mask, dtype=bool)
    transformed = np.rot90(transformed, k=_randint(rng, 0, 3))
    if _randint(rng, 0, 1):
        transformed = np.fliplr(transformed)
    if _randint(rng, 0, 1):
        transformed = np.flipud(transformed)
    return np.asarray(transformed, dtype=bool)


def _expand_box(box: tuple[int, int, int, int], pad: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return r0 - pad, c0 - pad, r1 + pad, c1 + pad


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _carve_notch(
    canvas: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int,
    side: str,
    rng: np.random.Generator,
) -> None:
    if side in {"top", "bottom"}:
        if width < 3 or height < 3:
            return
        seg = _randint(rng, 1, width - 2)
        start = _randint(rng, 1, width - seg - 1)
        depth = _randint(rng, 1, min(2, height - 2))
        cols = slice(left + start, left + start + seg)
        if side == "top":
            canvas[top : top + depth, cols] = False
        else:
            canvas[top + height - depth : top + height, cols] = False
        return

    if height < 3 or width < 3:
        return
    seg = _randint(rng, 1, height - 2)
    start = _randint(rng, 1, height - seg - 1)
    depth = _randint(rng, 1, min(2, width - 2))
    rows = slice(top + start, top + start + seg)
    if side == "left":
        canvas[rows, left : left + depth] = False
    else:
        canvas[rows, left + width - depth : left + width] = False


def _add_protrusion(
    canvas: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int,
    side: str,
    rng: np.random.Generator,
) -> None:
    if side in {"top", "bottom"}:
        if width < 1:
            return
        thick = _randint(rng, 1, 2)
        seg = _randint(rng, 1, width)
        start = _randint(rng, 0, width - seg)
        cols = slice(left + start, left + start + seg)
        if side == "top":
            if top - thick < 0:
                return
            canvas[top - thick : top, cols] = True
        else:
            if top + height + thick > canvas.shape[0]:
                return
            canvas[top + height : top + height + thick, cols] = True
        return

    if height < 1:
        return
    thick = _randint(rng, 1, 2)
    seg = _randint(rng, 1, height)
    start = _randint(rng, 0, height - seg)
    rows = slice(top + start, top + start + seg)
    if side == "left":
        if left - thick < 0:
            return
        canvas[rows, left - thick : left] = True
    else:
        if left + width + thick > canvas.shape[1]:
            return
        canvas[rows, left + width : left + width + thick] = True


def _place_holes(
    canvas: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int,
    rng: np.random.Generator,
) -> bool:
    hole_boxes: list[tuple[int, int, int, int]] = []
    hole_total = 1
    if min(height, width) >= 7 and _randint(rng, 0, 1):
        hole_total = 2

    for _ in range(hole_total):
        hh_max = min(3, height - 2)
        ww_max = min(3, width - 2)
        if hh_max < 1 or ww_max < 1:
            return False

        placed = False
        for _ in range(40):
            hh = _randint(rng, 1, hh_max)
            ww = _randint(rng, 1, ww_max)
            r_hi = top + height - hh - 1
            c_hi = left + width - ww - 1
            if r_hi < top + 1 or c_hi < left + 1:
                continue

            r0 = _randint(rng, top + 1, r_hi)
            c0 = _randint(rng, left + 1, c_hi)
            box = (r0, c0, r0 + hh - 1, c0 + ww - 1)
            if any(_boxes_overlap(_expand_box(box, 1), other) for other in hole_boxes):
                continue
            hole_boxes.append(box)
            placed = True
            break

        if not placed:
            return False

    for r0, c0, r1, c1 in hole_boxes:
        canvas[r0 : r1 + 1, c0 : c1 + 1] = False

    return True


def _make_shape(
    rng: np.random.Generator,
    has_hole: bool,
    crowded: bool,
) -> np.ndarray | None:
    canvas_size = 16 if crowded else 18
    min_side = 5 if has_hole else 4
    max_side = 7 if crowded else 9

    for _ in range(SHAPE_ATTEMPTS):
        canvas = np.zeros((canvas_size, canvas_size), dtype=bool)

        height = _randint(rng, min_side, max_side)
        width = _randint(rng, min_side, max_side)
        top = _randint(rng, 3, canvas_size - height - 3)
        left = _randint(rng, 3, canvas_size - width - 3)
        canvas[top : top + height, left : left + width] = True

        for _ in range(_randint(rng, 0, 3)):
            _add_protrusion(canvas, top, left, height, width, str(rng.choice(["top", "bottom", "left", "right"])), rng)

        max_notches = 1 if has_hole else 2
        for _ in range(_randint(rng, 0, max_notches)):
            _carve_notch(canvas, top, left, height, width, str(rng.choice(["top", "bottom", "left", "right"])), rng)

        if has_hole and not _place_holes(canvas, top, left, height, width, rng):
            continue

        if not np.any(canvas):
            continue

        mask = _crop_mask(canvas)
        mask = _random_transform(mask, rng)
        mask = _crop_mask(mask)

        if len(_components_from_mask(mask)) != 1:
            continue

        holes = _hole_count(mask)
        if has_hole and holes == 0:
            continue
        if not has_hole and holes != 0:
            continue

        area = int(mask.sum())
        if area < 8:
            continue
        if mask.shape[0] < 3 or mask.shape[1] < 3:
            continue
        if area / mask.size < 0.35:
            continue

        return mask

    return None


def _balanced_labels(rng: np.random.Generator, count: int) -> list[bool]:
    if count == 2:
        labels = [False, True]
    else:
        hole_count = count // 2 + int(count % 2 and _randint(rng, 0, 1))
        labels = [True] * hole_count + [False] * (count - hole_count)
    rng.shuffle(labels)
    return labels


def _border_positions(height: int, width: int) -> list[tuple[int, int]]:
    coords = {(0, c) for c in range(width)}
    coords |= {(height - 1, c) for c in range(width)}
    coords |= {(r, 0) for r in range(height)}
    coords |= {(r, width - 1) for r in range(height)}
    return sorted(coords)


def _choose_markers(height: int, width: int, rng: np.random.Generator) -> tuple[tuple[int, int], tuple[int, int]]:
    border = _border_positions(height, width)
    blue_pos = border[_randint(rng, 0, len(border) - 1)]
    red_choices = [pos for pos in border if pos != blue_pos and abs(pos[0] - blue_pos[0]) + abs(pos[1] - blue_pos[1]) > 1]
    red_pos = red_choices[_randint(rng, 0, len(red_choices) - 1)]
    return blue_pos, red_pos


def _place_objects(
    height: int,
    width: int,
    items: list[tuple[np.ndarray, bool]],
    rng: np.random.Generator,
) -> list[tuple[np.ndarray, bool, int, int]] | None:
    placements: list[tuple[np.ndarray, bool, int, int] | None] = [None] * len(items)
    blocked_boxes: list[tuple[int, int, int, int]] = []
    order = sorted(range(len(items)), key=lambda i: int(items[i][0].sum()), reverse=True)

    for idx in order:
        patch, has_hole = items[idx]
        ph, pw = patch.shape
        rows = list(range(PLACEMENT_MARGIN, height - ph - PLACEMENT_MARGIN + 1))
        cols = list(range(PLACEMENT_MARGIN, width - pw - PLACEMENT_MARGIN + 1))
        if not rows or not cols:
            return None

        candidates = [(r, c) for r in rows for c in cols]
        rng.shuffle(candidates)
        placed = False

        for top, left in candidates:
            box = (
                top - PLACEMENT_GAP,
                left - PLACEMENT_GAP,
                top + ph - 1 + PLACEMENT_GAP,
                left + pw - 1 + PLACEMENT_GAP,
            )
            if any(_boxes_overlap(box, other) for other in blocked_boxes):
                continue

            blocked_boxes.append(box)
            placements[idx] = (patch, has_hole, top, left)
            placed = True
            break

        if not placed:
            return None

    assert all(item is not None for item in placements)
    return [item for item in placements if item is not None]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey object by whether it encloses an orange hole.

    Input structure:
    - Orange background.
    - Several grey 4-connected foreground objects.
    - One blue edge marker and one red edge marker acting as a legend.

    Rule:
    - Grey objects with no enclosed background holes become blue.
    - Grey objects with one or more enclosed background holes become red.
    - The edge markers are erased to the background.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _dominant_color(g)
    out = g.copy()

    border = np.zeros(g.shape, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    out[border & ((g == BLUE) | (g == RED))] = bg

    grey_components = _components_from_mask(g == GREY)
    assert grey_components, "expected at least one grey object"

    for coords in grey_components:
        r0, c0, r1, c1 = _bbox(coords)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in coords:
            mask[r - r0, c - c0] = True

        new_color = RED if _hole_count(mask) > 0 else BLUE
        for r, c in coords:
            out[r, c] = new_color

    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = _randint(rng, *GRID_HEIGHT_RANGE)
    width = _randint(rng, *GRID_WIDTH_RANGE)

    max_objects = OBJECT_COUNT_RANGE[1]
    if height * width < 320:
        max_objects = min(max_objects, 4)
    count = _randint(rng, OBJECT_COUNT_RANGE[0], max_objects)
    labels = _balanced_labels(rng, count)
    crowded = count >= 4 or min(height, width) <= 17

    items: list[tuple[np.ndarray, bool]] = []
    for has_hole in labels:
        patch = _make_shape(rng, has_hole, crowded)
        if patch is None:
            return None
        items.append((patch, has_hole))

    placements = _place_objects(height, width, items, rng)
    if placements is None:
        return None

    grid = np.full((height, width), ORANGE, dtype=int)
    expected = np.full((height, width), ORANGE, dtype=int)

    blue_pos, red_pos = _choose_markers(height, width, rng)
    grid[blue_pos] = BLUE
    grid[red_pos] = RED

    seen_hole = False
    seen_solid = False
    for patch, has_hole, top, left in placements:
        ph, pw = patch.shape
        region = grid[top : top + ph, left : left + pw]
        region[patch] = GREY

        out_region = expected[top : top + ph, left : left + pw]
        out_region[patch] = RED if has_hole else BLUE

        seen_hole |= has_hole
        seen_solid |= not has_hole

    assert seen_hole and seen_solid
    assert int(np.sum(grid == BLUE)) == 1
    assert int(np.sum(grid == RED)) == 1
    if int(np.sum(grid == ORANGE)) <= int(np.sum(grid != ORANGE)):
        return None
    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate input after {GENERATION_ATTEMPTS} attempts")
