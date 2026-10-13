import random
from collections import Counter

import numpy as np


BACKGROUND = 0
LEFT_COLOR = 6
RIGHT_COLOR = 5
OBJECT_COLORS = (1, 2, 3, 7, 8)
OBJECT_AREAS = (2, 3, 4)
MAX_ATTEMPTS = 600

SHAPES_BY_AREA = {
    2: (
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
    ),
    3: (
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (1, 0), (2, 0)),
        ((0, 0), (1, 0), (1, 1)),
        ((0, 1), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (1, 0)),
        ((0, 0), (0, 1), (1, 1)),
    ),
    4: (
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (0, 2), (0, 3)),
        ((0, 0), (1, 0), (2, 0), (3, 0)),
        ((0, 0), (1, 0), (2, 0), (2, 1)),
        ((0, 1), (1, 1), (2, 0), (2, 1)),
    ),
}


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


def _find_components(grid: np.ndarray, bg: int = BACKGROUND) -> list[dict]:
    comps: list[dict] = []
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            visited[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(cells, dtype=int)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _bbox_center(coords: np.ndarray) -> tuple[int, int]:
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    center = (mins + maxs) // 2
    return int(center[0]), int(center[1])


def _connected(mask: np.ndarray) -> bool:
    points = np.argwhere(mask)
    if points.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, points[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _shape_to_coords(shape: tuple[tuple[int, int], ...], top: int, left: int) -> np.ndarray:
    return np.array([(top + r, left + c) for r, c in shape], dtype=int)


def _shape_center(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    coords = np.array(shape, dtype=int)
    return _bbox_center(coords)


def _targets_from_grid(grid: np.ndarray) -> tuple[list[tuple[int, int]], np.ndarray]:
    h, w = grid.shape
    top_h = h // 2
    panel_w = w // 2
    left = grid[:top_h, :panel_w]
    right = grid[:top_h, panel_w:]
    targets = [(r, c) for r in range(top_h) for c in range(panel_w) if int(left[r, c]) == BACKGROUND and int(right[r, c]) == BACKGROUND]
    return targets, grid[top_h:, :]


def _render_bottom(bottom: np.ndarray, targets: list[tuple[int, int]], *, order: str = "asc") -> np.ndarray:
    objects = _find_components(bottom, bg=BACKGROUND)
    if order == "asc":
        objects.sort(key=lambda obj: (int(obj["area"]), _bbox_center(obj["coords"]), int(obj["color"])))
    elif order == "desc":
        objects.sort(key=lambda obj: (-int(obj["area"]), _bbox_center(obj["coords"]), int(obj["color"])))
    else:
        raise ValueError(f"unsupported order: {order}")

    out = np.zeros_like(bottom)
    occupied = np.zeros_like(bottom, dtype=bool)
    available = list(targets)
    for obj in objects:
        if not available:
            continue
        center = _bbox_center(obj["coords"])
        ranked = sorted(
            available,
            key=lambda target: (
                abs(center[0] - target[0]) + abs(center[1] - target[1]),
                target[0],
                target[1],
            ),
        )
        for target in ranked:
            shift = np.array([target[0] - center[0], target[1] - center[1]], dtype=int)
            moved = obj["coords"] + shift
            if (
                moved[:, 0].min() < 0
                or moved[:, 0].max() >= bottom.shape[0]
                or moved[:, 1].min() < 0
                or moved[:, 1].max() >= bottom.shape[1]
            ):
                continue
            if occupied[moved[:, 0], moved[:, 1]].any():
                continue
            out[moved[:, 0], moved[:, 1]] = int(obj["color"])
            occupied[moved[:, 0], moved[:, 1]] = True
            available.remove(target)
            break
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the overlap of the two upper-panel backgrounds as target coordinates for moving the lower objects.

    Input structure:
    - The grid has a black background and is split into an upper half and a lower half.
    - The upper half contains two aligned side-by-side panels: a pink mask on the left and a grey mask on the right.
    - The lower half contains several separated monochrome 4-connected objects.

    Rule:
    - Compare the two upper panels cell by cell and record every position that is black in both panels.
    - Treat each such shared-background position as a target point in the lower-half coordinate system with the same row and column.
    - Process lower objects from smaller area to larger area; each object moves so its bbox-center lands on the nearest still-unused target point, and objects that cannot be placed are erased.

    Key insight:
    - The upper panels do not directly become output pixels; they encode hidden coordinates that control where the independent lower objects must be translated.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert g.shape[0] % 2 == 0 and g.shape[1] % 2 == 0
    targets, bottom = _targets_from_grid(g)
    return _render_bottom(bottom, targets, order="asc")


def _make_top_masks(
    panel_h: int,
    panel_w: int,
    targets: list[tuple[int, int]],
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray] | None:
    target_set = set(targets)
    total = panel_h * panel_w
    cells = [(r, c) for r in range(panel_h) for c in range(panel_w) if (r, c) not in target_set]
    for _ in range(200):
        left_fg = np.zeros((panel_h, panel_w), dtype=bool)
        right_fg = np.zeros((panel_h, panel_w), dtype=bool)
        left_only = 0
        right_only = 0
        for r, c in cells:
            state = rng.choices((1, 2, 3), weights=(3, 3, 5))[0]
            if state in (1, 3):
                left_fg[r, c] = True
            if state in (2, 3):
                right_fg[r, c] = True
            if state == 1:
                left_only += 1
            elif state == 2:
                right_only += 1
        if left_only < 2 or right_only < 2:
            continue
        if not _connected(left_fg) or not _connected(right_fg):
            continue
        left_count = int(left_fg.sum())
        right_count = int(right_fg.sum())
        if left_count < total // 2 or right_count < total // 2:
            continue
        left = np.where(left_fg, LEFT_COLOR, BACKGROUND)
        right = np.where(right_fg, RIGHT_COLOR, BACKGROUND)
        if len([(r, c) for r in range(panel_h) for c in range(panel_w) if left[r, c] == 0 and right[r, c] == 0]) != len(targets):
            continue
        return left.astype(int), right.astype(int)
    return None


def _can_place_shape(existing: set[tuple[int, int]], coords: np.ndarray, h: int, w: int) -> bool:
    if coords[:, 0].min() < 0 or coords[:, 0].max() >= h or coords[:, 1].min() < 0 or coords[:, 1].max() >= w:
        return False
    for r, c in map(tuple, coords):
        if (r, c) in existing:
            return False
        for nr, nc in _neighbors8(r, c):
            if (nr, nc) in existing:
                return False
    return True


def _place_bottom_objects(
    bottom_h: int,
    bottom_w: int,
    rng: random.Random,
) -> np.ndarray | None:
    bottom = np.zeros((bottom_h, bottom_w), dtype=int)
    existing: set[tuple[int, int]] = set()
    colors = list(rng.sample(OBJECT_COLORS, len(OBJECT_AREAS)))
    for area, color in zip(OBJECT_AREAS, colors):
        shapes = list(SHAPES_BY_AREA[area])
        rng.shuffle(shapes)
        placed = False
        for shape in shapes:
            shape_arr = np.array(shape, dtype=int)
            shape_h = int(shape_arr[:, 0].max()) + 1
            shape_w = int(shape_arr[:, 1].max()) + 1
            col_starts = list(range(max(0, bottom_w // 2 - 1), bottom_w - shape_w + 1))
            if not col_starts:
                col_starts = list(range(0, bottom_w - shape_w + 1))
            row_starts = list(range(0, bottom_h - shape_h + 1))
            rng.shuffle(row_starts)
            rng.shuffle(col_starts)
            for top in row_starts:
                for left in col_starts:
                    coords = _shape_to_coords(shape, top, left)
                    if _can_place_shape(existing, coords, bottom_h, bottom_w):
                        bottom[coords[:, 0], coords[:, 1]] = color
                        existing.update(map(tuple, coords))
                        placed = True
                        break
                if placed:
                    break
            if placed:
                break
        if not placed:
            return None
    return bottom


def _place_bottom_objects_spread(
    bottom_h: int,
    bottom_w: int,
    rng: random.Random,
) -> np.ndarray | None:
    bottom = np.zeros((bottom_h, bottom_w), dtype=int)
    existing: set[tuple[int, int]] = set()
    colors = list(rng.sample(OBJECT_COLORS, len(OBJECT_AREAS)))
    desired_rows = [0, max(0, bottom_h // 2 - 1), max(0, bottom_h - 2)]
    desired_cols = [max(bottom_w // 2, bottom_w - 3), max(bottom_w // 2, bottom_w - 5), max(bottom_w // 2 - 1, bottom_w - 7)]
    row_col_pairs = list(zip(desired_rows, desired_cols))
    rng.shuffle(row_col_pairs)

    for (area, color), (goal_r, goal_c) in zip(zip(OBJECT_AREAS, colors), row_col_pairs):
        shapes = [shape for shape in SHAPES_BY_AREA[area] if max(r for r, _ in shape) <= 1]
        if not shapes:
            shapes = list(SHAPES_BY_AREA[area])
        rng.shuffle(shapes)
        placed = False
        for shape in shapes:
            shape_arr = np.array(shape, dtype=int)
            shape_h = int(shape_arr[:, 0].max()) + 1
            shape_w = int(shape_arr[:, 1].max()) + 1
            candidates = []
            for top in range(0, bottom_h - shape_h + 1):
                for left in range(0, bottom_w - shape_w + 1):
                    coords = _shape_to_coords(shape, top, left)
                    if _can_place_shape(existing, coords, bottom_h, bottom_w):
                        score = abs(top - goal_r) + abs(left - goal_c)
                        score += rng.random() * 0.25
                        candidates.append((score, coords))
            if not candidates:
                continue
            candidates.sort(key=lambda item: item[0])
            coords = candidates[0][1]
            bottom[coords[:, 0], coords[:, 1]] = color
            existing.update(map(tuple, coords))
            placed = True
            break
        if not placed:
            return None
    return bottom


def _object_nearest_targets(bottom: np.ndarray, targets: list[tuple[int, int]]) -> list[tuple[int, ...]]:
    result = []
    for obj in _find_components(bottom, bg=BACKGROUND):
        center = _bbox_center(obj["coords"])
        dists = tuple(sorted(abs(center[0] - tr) + abs(center[1] - tc) for tr, tc in targets))
        result.append(dists)
    return result


def _nearest_target_ids(bottom: np.ndarray, targets: list[tuple[int, int]]) -> list[int]:
    out = []
    objects = sorted(_find_components(bottom, bg=BACKGROUND), key=lambda obj: int(obj["area"]))
    for obj in objects:
        center = _bbox_center(obj["coords"])
        best_idx = min(
            range(len(targets)),
            key=lambda i: (
                abs(center[0] - targets[i][0]) + abs(center[1] - targets[i][1]),
                targets[i][0],
                targets[i][1],
            ),
        )
        out.append(int(best_idx))
    return out


def _valid_target_choice(
    bottom: np.ndarray,
    targets: list[tuple[int, int]],
    require_conflict: bool,
) -> bool:
    if len(targets) != len(OBJECT_AREAS):
        return False
    dist_lists = _object_nearest_targets(bottom, targets)
    if any(len(d) < 2 or len(set(d)) != len(d) for d in dist_lists):
        return False
    asc = _render_bottom(bottom, targets, order="asc")
    if len(_find_components(asc, bg=BACKGROUND)) != len(OBJECT_AREAS):
        return False
    if np.array_equal(asc, bottom):
        return False
    desc = _render_bottom(bottom, targets, order="desc")
    nearest_ids = _nearest_target_ids(bottom, targets)
    has_conflict = len(set(nearest_ids)) < len(nearest_ids)
    if require_conflict:
        return has_conflict and not np.array_equal(asc, desc)
    return (not has_conflict) and np.array_equal(asc, desc)


def _build_candidate_targets(panel_h: int, panel_w: int) -> list[tuple[int, int]]:
    rows = range(1, panel_h - 1)
    cols = range(1, panel_w - 1)
    return [(r, c) for r in rows for c in cols]


def _assemble_grid(
    panel_h: int,
    panel_w: int,
    targets: list[tuple[int, int]],
    bottom: np.ndarray,
    rng: random.Random,
) -> np.ndarray | None:
    masks = _make_top_masks(panel_h, panel_w, targets, rng)
    if masks is None:
        return None
    left, right = masks
    top = np.concatenate([left, right], axis=1)
    return np.concatenate([top, bottom], axis=0)


def _semantic_check(grid: np.ndarray, require_conflict: bool) -> bool:
    targets, bottom = _targets_from_grid(grid)
    top_h = grid.shape[0] // 2
    panel_w = grid.shape[1] // 2
    left = grid[:top_h, :panel_w]
    right = grid[:top_h, panel_w:]
    if set(np.unique(left)) - {BACKGROUND, LEFT_COLOR}:
        return False
    if set(np.unique(right)) - {BACKGROUND, RIGHT_COLOR}:
        return False
    if len(targets) != len(OBJECT_AREAS):
        return False
    left_bg = {(r, c) for r in range(top_h) for c in range(panel_w) if int(left[r, c]) == BACKGROUND}
    right_bg = {(r, c) for r in range(top_h) for c in range(panel_w) if int(right[r, c]) == BACKGROUND}
    if set(targets) != left_bg & right_bg:
        return False
    if set(targets) == left_bg or set(targets) == right_bg:
        return False
    bottom_objects = _find_components(bottom, bg=BACKGROUND)
    if sorted(int(obj["area"]) for obj in bottom_objects) != list(OBJECT_AREAS):
        return False
    if len({int(obj["color"]) for obj in bottom_objects}) != len(OBJECT_AREAS):
        return False
    out = main(grid)
    if out.shape != bottom.shape:
        return False
    if len(_find_components(out, bg=BACKGROUND)) != len(OBJECT_AREAS):
        return False
    return _valid_target_choice(bottom, targets, require_conflict)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        panel_h = rng.randint(4, 6)
        panel_w = rng.randint(4, 6)
        require_conflict = rng.random() < 0.2
        if require_conflict:
            bottom = _place_bottom_objects(panel_h, panel_w * 2, rng)
        else:
            bottom = _place_bottom_objects_spread(panel_h, panel_w * 2, rng)
        if bottom is None:
            continue
        candidates = _build_candidate_targets(panel_h, panel_w)
        if len(candidates) < len(OBJECT_AREAS):
            continue
        rng.shuffle(candidates)
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                for k in range(j + 1, len(candidates)):
                    targets = [candidates[i], candidates[j], candidates[k]]
                    if not _valid_target_choice(bottom, targets, require_conflict):
                        continue
                    grid = _assemble_grid(panel_h, panel_w, targets, bottom, rng)
                    if grid is not None and _semantic_check(grid, require_conflict):
                        return grid.astype(int)
    raise RuntimeError("failed to generate a valid input")
