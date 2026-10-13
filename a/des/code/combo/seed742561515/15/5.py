from __future__ import annotations

from collections import deque

import numpy as np


DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
BG = 0
COLORS = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400
MASK_RETRIES = 120
PLACEMENT_RETRIES = 40
GRID_SIDE_RANGE = (18, 30)

_RNG = np.random.default_rng()


def _randint(lo: int, hi: int) -> int:
    return int(_RNG.integers(int(lo), int(hi) + 1))


def _choice(seq):
    return seq[_randint(0, len(seq) - 1)]


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool).copy()


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    seen = {start}
    queue: deque[tuple[int, int]] = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIRS4:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                queue.append((rr, cc))
    return len(seen) == int(mask.sum())


def _grow_mask(
    shape: tuple[int, int],
    target: int,
    seeds: list[tuple[int, int]],
    allowed: np.ndarray | None = None,
) -> np.ndarray | None:
    h, w = map(int, shape)
    if allowed is None:
        allowed = np.ones((h, w), dtype=bool)
    mask = np.zeros((h, w), dtype=bool)
    frontier: set[tuple[int, int]] = set()

    def add_frontier(r: int, c: int) -> None:
        for dr, dc in DIRS4:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w and allowed[rr, cc] and not mask[rr, cc]:
                frontier.add((rr, cc))

    for r, c in seeds:
        if not (0 <= r < h and 0 <= c < w and allowed[r, c]):
            return None
        mask[r, c] = True

    for r, c in seeds:
        add_frontier(int(r), int(c))

    while int(mask.sum()) < target:
        if not frontier:
            return None
        cell = tuple(frontier)[_randint(0, len(frontier) - 1)]
        frontier.remove(cell)
        r, c = cell
        mask[r, c] = True
        add_frontier(r, c)

    return mask


def _symmetry_type(mask: np.ndarray) -> str:
    horiz = bool(np.array_equal(mask, np.flipud(mask)))
    vert = bool(np.array_equal(mask, np.fliplr(mask)))
    if horiz and vert:
        return "both"
    if horiz:
        return "h_only"
    if vert:
        return "v_only"
    return "none"


def _transform_mask(mask: np.ndarray, kind: str) -> np.ndarray:
    if kind == "h_only":
        return np.fliplr(mask).copy()
    if kind == "v_only":
        return np.flipud(mask).copy()
    if kind == "both":
        top = np.asarray(mask[: mask.shape[0] // 2, :], dtype=bool)
        assert top.shape[0] > 0 and np.any(top), "double-symmetric object must keep a non-empty upper half"
        return _crop_mask(top)
    assert kind == "none"
    return mask.copy()


def _extract_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    objects = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BG or seen[r, c]:
                continue
            seen[r, c] = True
            queue: deque[tuple[int, int]] = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            pts = np.asarray(coords, dtype=int)
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
            rel = pts - np.array([r0, c0])
            mask[rel[:, 0], rel[:, 1]] = True
            objects.append({"color": color, "r0": int(r0), "c0": int(c0), "mask": mask})

    objects.sort(key=lambda obj: (int(obj["r0"]), int(obj["c0"])))
    return objects


def _stack_objects(items: list[dict]) -> np.ndarray:
    width = max(int(item["mask"].shape[1]) for item in items)
    height = sum(int(item["mask"].shape[0]) for item in items)
    out = np.zeros((height, width), dtype=int)
    row = 0
    for item in items:
        mask = np.asarray(item["mask"], dtype=bool)
        h, w = mask.shape
        col = (width - w) // 2
        out[row : row + h, col : col + w][mask] = int(item["color"])
        row += h
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Classify each object by axial symmetry, transform it, then stack the transformed objects by descending area.

    Input structure
    - Black background.
    - 3 to 5 non-overlapping monochrome 4-connected foreground objects.
    - Each object has a distinct color and one of four symmetry classes:
      horizontal-only, vertical-only, both axes, or none.

    Rule
    - Horizontal-only -> left-right flip.
    - Vertical-only -> up-down flip.
    - Both axes -> keep only the upper half.
    - None -> unchanged.
    - Sort transformed objects by transformed pixel area descending and stack them tightly, centered horizontally.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    objects = _extract_objects(g)
    assert objects, "expected at least one foreground object"

    transformed = []
    for obj in objects:
        mask = np.asarray(obj["mask"], dtype=bool)
        kind = _symmetry_type(mask)
        tmask = _transform_mask(mask, kind)
        transformed.append(
            {
                "color": int(obj["color"]),
                "mask": tmask,
                "area": int(tmask.sum()),
                "r0": int(obj["r0"]),
                "c0": int(obj["c0"]),
            }
        )

    transformed.sort(key=lambda item: (-int(item["area"]), int(item["r0"]), int(item["c0"])))
    return _stack_objects(transformed)


def _sample_target(total: int, min_fill: float, max_fill: float, min_cells: int) -> int:
    lo = max(int(min_cells), int(np.ceil(total * min_fill)))
    hi = min(int(total), int(np.floor(total * max_fill)))
    hi = max(lo, hi)
    return _randint(lo, hi)


def _reflect_horizontal(proto: np.ndarray, full_h: int) -> np.ndarray:
    out = np.zeros((full_h, proto.shape[1]), dtype=bool)
    for r, c in np.argwhere(proto):
        rr = int(r)
        cc = int(c)
        out[rr, cc] = True
        out[full_h - 1 - rr, cc] = True
    return _crop_mask(out)


def _reflect_vertical(proto: np.ndarray, full_w: int) -> np.ndarray:
    out = np.zeros((proto.shape[0], full_w), dtype=bool)
    for r, c in np.argwhere(proto):
        rr = int(r)
        cc = int(c)
        out[rr, cc] = True
        out[rr, full_w - 1 - cc] = True
    return _crop_mask(out)


def _reflect_both(proto: np.ndarray, full_h: int, full_w: int) -> np.ndarray:
    out = np.zeros((full_h, full_w), dtype=bool)
    for r, c in np.argwhere(proto):
        rr = int(r)
        cc = int(c)
        for ar in {rr, full_h - 1 - rr}:
            for ac in {cc, full_w - 1 - cc}:
                out[ar, ac] = True
    return _crop_mask(out)


def _make_h_only_mask() -> np.ndarray | None:
    for _ in range(MASK_RETRIES):
        h = _randint(4, 8)
        w = _randint(3, 8)
        upper_h = (h + 1) // 2
        seed = (upper_h - 1, _randint(0, w - 1))
        target = _sample_target(upper_h * w, 0.3, 0.75, max(2, upper_h))
        proto = _grow_mask((upper_h, w), target, [seed])
        if proto is None:
            continue
        mask = _reflect_horizontal(proto, h)
        if _symmetry_type(mask) != "h_only" or not _is_connected(mask):
            continue
        return np.fliplr(mask).copy() if _randint(0, 1) else mask
    return None


def _make_v_only_mask() -> np.ndarray | None:
    for _ in range(MASK_RETRIES):
        h = _randint(3, 8)
        w = _randint(4, 8)
        left_w = (w + 1) // 2
        seed = (_randint(0, h - 1), left_w - 1)
        target = _sample_target(h * left_w, 0.3, 0.75, max(2, left_w))
        proto = _grow_mask((h, left_w), target, [seed])
        if proto is None:
            continue
        mask = _reflect_vertical(proto, w)
        if _symmetry_type(mask) != "v_only" or not _is_connected(mask):
            continue
        return np.flipud(mask).copy() if _randint(0, 1) else mask
    return None


def _make_both_mask() -> np.ndarray | None:
    for _ in range(MASK_RETRIES):
        h = _randint(4, 8)
        w = _randint(4, 8)
        quad_h = (h + 1) // 2
        quad_w = (w + 1) // 2
        seed = (quad_h - 1, quad_w - 1)
        target = _sample_target(quad_h * quad_w, 0.25, 0.75, max(2, min(quad_h, quad_w)))
        proto = _grow_mask((quad_h, quad_w), target, [seed])
        if proto is None:
            continue
        mask = _reflect_both(proto, h, w)
        if mask.shape[0] < 2:
            continue
        if _symmetry_type(mask) != "both" or not _is_connected(mask):
            continue
        transformed = _transform_mask(mask, "both")
        if not _is_connected(transformed):
            continue
        if _randint(0, 1):
            mask = np.flipud(mask)
        if _randint(0, 1):
            mask = np.fliplr(mask)
        return mask
    return None


def _make_none_mask() -> np.ndarray | None:
    for _ in range(MASK_RETRIES):
        h = _randint(3, 8)
        w = _randint(3, 8)
        seed = (_randint(0, h - 1), _randint(0, w - 1))
        target = _sample_target(h * w, 0.25, 0.7, 3)
        mask = _grow_mask((h, w), target, [seed])
        if mask is None:
            continue
        mask = _crop_mask(mask)
        if _symmetry_type(mask) != "none" or not _is_connected(mask):
            continue
        if _randint(0, 1):
            mask = np.flipud(mask)
        if _randint(0, 1):
            mask = np.fliplr(mask)
        return mask
    return None


def _make_mask(kind: str) -> np.ndarray | None:
    if kind == "h_only":
        return _make_h_only_mask()
    if kind == "v_only":
        return _make_v_only_mask()
    if kind == "both":
        return _make_both_mask()
    return _make_none_mask()


def _sample_kinds(count: int) -> list[str]:
    others = ["h_only", "v_only", "none"]
    if count == 3:
        kinds = ["both"] + list(_RNG.choice(others, size=2, replace=False))
    elif count == 4:
        kinds = ["both", "h_only", "v_only", "none"]
    else:
        kinds = ["both", "h_only", "v_only", "none", _choice(["both", "h_only", "v_only", "none"])]
    _RNG.shuffle(kinds)
    return list(kinds)


def _dilate(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _place_objects(objects: list[dict], height: int, width: int) -> list[tuple[int, int]] | None:
    positions: list[tuple[int, int] | None] = [None] * len(objects)
    occupied = np.zeros((height, width), dtype=bool)
    order = list(range(len(objects)))
    _RNG.shuffle(order)
    order.sort(key=lambda i: objects[i]["mask"].shape[0] * objects[i]["mask"].shape[1], reverse=True)

    for idx in order:
        mask = np.asarray(objects[idx]["mask"], dtype=bool)
        h, w = mask.shape
        blocked = _dilate(occupied)
        rows = list(range(1, height - h)) if height - h >= 2 else list(range(0, height - h + 1))
        cols = list(range(1, width - w)) if width - w >= 2 else list(range(0, width - w + 1))
        candidates = []
        for r in rows:
            for c in cols:
                if not np.any(blocked[r : r + h, c : c + w] & mask):
                    candidates.append((r, c))
        if not candidates:
            return None
        r0, c0 = candidates[_randint(0, len(candidates) - 1)]
        positions[idx] = (int(r0), int(c0))
        occupied[r0 : r0 + h, c0 : c0 + w] |= mask

    return [tuple(pos) for pos in positions if pos is not None]


def _build_input(objects: list[dict], positions: list[tuple[int, int]], height: int, width: int) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for obj, (r0, c0) in zip(objects, positions):
        mask = np.asarray(obj["mask"], dtype=bool)
        h, w = mask.shape
        grid[r0 : r0 + h, c0 : c0 + w][mask] = int(obj["color"])
    return grid


def _expected_output(objects: list[dict], positions: list[tuple[int, int]]) -> np.ndarray:
    items = []
    for obj, (r0, c0) in zip(objects, positions):
        items.append(
            {
                "color": int(obj["color"]),
                "mask": np.asarray(obj["transformed"], dtype=bool),
                "area": int(obj["transformed_area"]),
                "r0": int(r0),
                "c0": int(c0),
            }
        )
    items.sort(key=lambda item: (-int(item["area"]), int(item["r0"]), int(item["c0"])))
    return _stack_objects(items)


def _ranking(values: list[int]) -> tuple[int, ...]:
    return tuple(sorted(range(len(values)), key=lambda i: -int(values[i])))


def _generate_objects() -> list[dict] | None:
    count = _randint(3, 5)
    kinds = _sample_kinds(count)
    colors = list(_RNG.choice(COLORS, size=count, replace=False))
    objects = []
    used_transformed_areas: set[int] = set()

    for kind, color in zip(kinds, colors):
        for _ in range(MASK_RETRIES):
            mask = _make_mask(kind)
            if mask is None:
                continue
            transformed = _transform_mask(mask, kind)
            transformed_area = int(transformed.sum())
            if transformed_area in used_transformed_areas:
                continue
            if kind in {"h_only", "v_only"} and np.array_equal(mask, transformed):
                continue
            if kind == "none" and not np.array_equal(mask, transformed):
                continue
            if kind == "both" and np.array_equal(mask, transformed):
                continue
            used_transformed_areas.add(transformed_area)
            objects.append(
                {
                    "kind": kind,
                    "color": int(color),
                    "mask": mask,
                    "area": int(mask.sum()),
                    "transformed": transformed,
                    "transformed_area": transformed_area,
                }
            )
            break
        else:
            return None

    if _ranking([obj["area"] for obj in objects]) == _ranking([obj["transformed_area"] for obj in objects]):
        return None
    return objects


def _gen_attempt() -> np.ndarray | None:
    objects = _generate_objects()
    if objects is None:
        return None

    total_bbox_area = sum(int(obj["mask"].shape[0] * obj["mask"].shape[1]) for obj in objects)
    max_h = max(int(obj["mask"].shape[0]) for obj in objects)
    max_w = max(int(obj["mask"].shape[1]) for obj in objects)
    min_side = max(GRID_SIDE_RANGE[0], max_h + 4, max_w + 4, int(np.ceil(np.sqrt(total_bbox_area) * 1.55)))
    if min_side > GRID_SIDE_RANGE[1]:
        return None

    for _ in range(PLACEMENT_RETRIES):
        height = _randint(min_side, GRID_SIDE_RANGE[1])
        width = _randint(min_side, GRID_SIDE_RANGE[1])
        positions = _place_objects(objects, height, width)
        if positions is None:
            continue

        grid = _build_input(objects, positions, height, width)
        if int(np.count_nonzero(grid == BG)) <= int(np.count_nonzero(grid != BG)):
            continue

        found = _extract_objects(grid)
        if len(found) != len(objects):
            continue
        expected = _expected_output(objects, positions)
        try:
            actual = main(grid)
        except AssertionError:
            continue
        if not np.array_equal(actual, expected):
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
