from __future__ import annotations

import random
import numpy as np


def _most_common_color(arr: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(arr, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
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
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _sorted_by_palette(colors: list[int], palette_order: list[int]) -> list[int]:
    ordered: list[int] = []
    known = set(palette_order)
    for color in palette_order:
        ordered.extend(v for v in colors if v == color)
    ordered.extend(v for v in colors if v not in known)
    return ordered


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder each foreground object's row pixels using the color priority read
    from the left-edge palette column, then remove that palette column.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[1] >= 2, "expected a 2D grid with a palette column"

    main_region = g[:, 1:]
    bg = _most_common_color(main_region)

    palette_order: list[int] = []
    seen_palette: set[int] = set()
    for r in range(g.shape[0]):
        color = int(g[r, 0])
        if color == bg or color in seen_palette:
            continue
        palette_order.append(color)
        seen_palette.add(color)
    assert palette_order, "expected at least one palette color on the left edge"

    out = main_region.copy()
    for comp in _components(main_region != bg):
        rows: dict[int, list[int]] = {}
        for r, c in comp:
            rows.setdefault(r, []).append(c)
        for r, cols in rows.items():
            cols.sort()
            row_colors = [int(main_region[r, c]) for c in cols]
            sorted_colors = _sorted_by_palette(row_colors, palette_order)
            for c, color in zip(cols, sorted_colors):
                out[r, c] = color

    return out


# --------------- generator ---------------

_COLORS = tuple(range(10))
_GRID_H_RANGE = (8, 15)
_MAIN_W_RANGE = (10, 21)
_PALETTE_LEN_RANGE = (3, 5)
_OBJECT_COUNT_RANGE = (2, 4)
_OBJECT_H_RANGE = (2, 5)
_OBJECT_W_RANGE = (3, 7)
_MAX_OBJECT_COLORS = 4
_BBOX_MARGIN = 1
_MAX_GENERATION_ATTEMPTS = 400


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _is_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    start = tuple(map(int, pts[0]))
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _row_has_gap(row_mask: np.ndarray) -> bool:
    cols = np.flatnonzero(row_mask)
    return bool(cols.size >= 2 and np.any(np.diff(cols) > 1))


def _maybe_carve_gap(mask: np.ndarray) -> np.ndarray:
    candidates: list[np.ndarray] = []
    for r in range(mask.shape[0]):
        cols = np.flatnonzero(mask[r])
        for c in cols[1:-1]:
            trial = mask.copy()
            trial[r, c] = False
            if not _is_connected(trial):
                continue
            trial = _trim_mask(trial)
            if any(_row_has_gap(trial[rr]) for rr in range(trial.shape[0])):
                candidates.append(trial)
    if candidates and random.random() < 0.5:
        return random.choice(candidates)
    return mask


def _generate_object_mask() -> np.ndarray | None:
    for _ in range(120):
        h = random.randint(*_OBJECT_H_RANGE)
        w = random.randint(*_OBJECT_W_RANGE)
        full_area = h * w
        irregular = random.random() < 0.75 and full_area >= 6
        area_min = max(4, (full_area + 1) // 2)
        area_max = full_area - 1 if irregular else full_area
        if area_max < area_min:
            area_max = full_area
        target_area = random.randint(area_min, area_max)

        mask = np.ones((h, w), dtype=bool)
        while int(mask.sum()) > target_area:
            cells = list(map(tuple, np.argwhere(mask)))
            random.shuffle(cells)
            changed = False
            for r, c in cells:
                trial = mask.copy()
                trial[r, c] = False
                if not _is_connected(trial):
                    continue
                trial = _trim_mask(trial)
                if trial.shape[0] < 2 or trial.shape[1] < 2 or int(trial.sum()) < 4:
                    continue
                mask = trial
                changed = True
                break
            if not changed:
                break

        mask = _maybe_carve_gap(mask)
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if int(mask.sum()) < 4:
            continue
        if int(mask.sum(axis=1).max()) < 2:
            continue
        return mask
    return None


def _make_unsorted_row(length: int, object_colors: list[int], palette_order: list[int]) -> list[int]:
    left_color, right_color = random.sample(object_colors, 2)
    order_index = {color: idx for idx, color in enumerate(palette_order)}
    if order_index[left_color] < order_index[right_color]:
        left_color, right_color = right_color, left_color
    row = [random.choice(object_colors) for _ in range(length)]
    row[0] = left_color
    row[-1] = right_color
    assert row != _sorted_by_palette(row, palette_order)
    return row


def _paint_object(mask: np.ndarray, palette_order: list[int], bg: int) -> np.ndarray | None:
    usable = min(len(palette_order), _MAX_OBJECT_COLORS)
    if usable < 2:
        return None
    object_colors = random.sample(palette_order, random.randint(2, usable))
    patch = np.full(mask.shape, bg, dtype=int)

    candidate_rows = [r for r in range(mask.shape[0]) if int(mask[r].sum()) >= 2]
    if not candidate_rows:
        return None
    featured_row = random.choice(candidate_rows)

    changed = False
    for r in range(mask.shape[0]):
        cols = list(map(int, np.flatnonzero(mask[r])))
        if not cols:
            continue
        if r == featured_row:
            row_colors = _make_unsorted_row(len(cols), object_colors, palette_order)
            changed = True
        else:
            row_colors = [random.choice(object_colors) for _ in cols]
        for c, color in zip(cols, row_colors):
            patch[r, c] = color

    if len(set(map(int, patch[mask]))) < 2:
        return None
    if not changed:
        return None
    return patch


def _boxes_too_close(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
    margin: int,
) -> bool:
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b
    return not (
        ar1 + margin < br0
        or br1 + margin < ar0
        or ac1 + margin < bc0
        or bc1 + margin < ac0
    )


def _place_objects(
    main_h: int,
    main_w: int,
    objects: list[dict[str, np.ndarray]],
) -> list[tuple[int, int]] | None:
    placements: list[tuple[int, int] | None] = [None] * len(objects)
    placed_boxes: list[tuple[int, int, int, int]] = []
    order = sorted(
        range(len(objects)),
        key=lambda idx: int(objects[idx]["mask"].sum()),
        reverse=True,
    )

    for idx in order:
        mask = objects[idx]["mask"]
        h, w = mask.shape
        if h > main_h or w > main_w:
            return None
        candidates: list[tuple[int, int]] = []
        for r0 in range(main_h - h + 1):
            for c0 in range(main_w - w + 1):
                box = (r0, c0, r0 + h - 1, c0 + w - 1)
                if any(_boxes_too_close(box, other, _BBOX_MARGIN) for other in placed_boxes):
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        placements[idx] = (r0, c0)
        placed_boxes.append((r0, c0, r0 + h - 1, c0 + w - 1))

    return [pos for pos in placements if pos is not None]


def _transform_patch(patch: np.ndarray, bg: int, palette_order: list[int]) -> np.ndarray:
    out = patch.copy()
    mask = patch != bg
    for r in range(patch.shape[0]):
        cols = list(map(int, np.flatnonzero(mask[r])))
        if not cols:
            continue
        row_colors = [int(patch[r, c]) for c in cols]
        sorted_colors = _sorted_by_palette(row_colors, palette_order)
        for c, color in zip(cols, sorted_colors):
            out[r, c] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    main_h = random.randint(*_GRID_H_RANGE)
    main_w = random.randint(*_MAIN_W_RANGE)
    bg = random.choice(_COLORS)

    palette_len = random.randint(*_PALETTE_LEN_RANGE)
    palette_pool = [color for color in _COLORS if color != bg]
    palette_order = random.sample(palette_pool, palette_len)

    object_count = random.randint(*_OBJECT_COUNT_RANGE)
    objects: list[dict[str, np.ndarray]] = []
    for _ in range(object_count):
        mask = _generate_object_mask()
        if mask is None:
            return None
        patch = _paint_object(mask, palette_order, bg)
        if patch is None:
            return None
        objects.append({"mask": mask, "patch": patch})

    placements = _place_objects(main_h, main_w, objects)
    if placements is None:
        return None

    grid = np.full((main_h, main_w + 1), bg, dtype=int)
    expected = np.full((main_h, main_w), bg, dtype=int)
    palette_start = random.randint(0, main_h - palette_len)
    grid[palette_start : palette_start + palette_len, 0] = np.asarray(palette_order, dtype=int)

    visible_change = False
    for obj, (r0, c0) in zip(objects, placements):
        mask = obj["mask"]
        patch = obj["patch"]
        h, w = mask.shape

        in_view = grid[r0 : r0 + h, c0 + 1 : c0 + 1 + w]
        in_view[mask] = patch[mask]

        transformed = _transform_patch(patch, bg, palette_order)
        out_view = expected[r0 : r0 + h, c0 : c0 + w]
        out_view[mask] = transformed[mask]
        if not np.array_equal(patch[mask], transformed[mask]):
            visible_change = True

    if not visible_change:
        return None
    if _most_common_color(grid[:, 1:]) != bg:
        return None

    main_region = grid[:, 1:]
    comps = _components(main_region != bg)
    if len(comps) != object_count:
        return None
    for comp in comps:
        comp_colors = {int(main_region[r, c]) for r, c in comp}
        if len(comp_colors) < 2:
            return None

    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
