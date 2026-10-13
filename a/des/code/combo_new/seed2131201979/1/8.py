from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


_NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# --------------- generator config ---------------

_BG_COLOR = 0
_FRAME_COLOR = 5
_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]

_GRID_H_RANGE = (16, 31)
_GRID_W_RANGE = (16, 31)
_MAX_OBJECTS_BY_MIN_SIDE = [(18, 2), (24, 3), (31, 4)]
_OBJECT_MARGIN_RANGE = (1, 4)
_GEN_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return 4-connected components of a boolean mask."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                pts.append((int(cr), int(cc)))
                for dr, dc in _NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(pts)

    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _coords_to_mask(coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask, (r0, c0, r1, c1)


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    arr = np.asarray(mask, dtype=np.uint8)
    return tuple(map(int, arr.shape)), arr.tobytes()


def _outer_frame_color(grid: np.ndarray) -> int:
    h, w = grid.shape
    edge = [
        *map(int, grid[0, :]),
        *map(int, grid[h - 1, :]),
        *map(int, grid[1 : h - 1, 0]),
        *map(int, grid[1 : h - 1, w - 1]),
    ]
    return Counter(edge).most_common(1)[0][0]


def _inner_ring_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    mask[1, 1 : w - 1] = True
    mask[h - 2, 1 : w - 1] = True
    mask[1 : h - 1, 1] = True
    mask[1 : h - 1, w - 2] = True
    return mask


def _background_color(grid: np.ndarray, frame_color: int) -> int:
    inner_ring = _inner_ring_mask(grid.shape)
    notch_values = [int(v) for v in grid[inner_ring & (grid != frame_color)]]
    if notch_values:
        return Counter(notch_values).most_common(1)[0][0]
    others = [int(v) for v in grid[grid != frame_color]]
    assert others, "grid must contain non-frame pixels"
    return Counter(others).most_common(1)[0][0]


def _extract_notches(grid: np.ndarray, frame_color: int) -> list[dict[str, object]]:
    ring = _inner_ring_mask(grid.shape)
    comps = _components(ring & (grid != frame_color))
    notches = []
    for coords in comps:
        mask, bbox = _coords_to_mask(coords)
        notches.append({"coords": coords, "mask": mask, "bbox": bbox})
    return notches


def _extract_objects(grid: np.ndarray, frame_color: int, bg_color: int) -> list[dict[str, object]]:
    obj_mask = (grid != frame_color) & (grid != bg_color)
    comps = _components(obj_mask)
    objects = []
    for coords in comps:
        colors = {int(grid[r, c]) for r, c in coords}
        assert len(colors) == 1, "each object must be monochrome"
        color = next(iter(colors))
        r0, c0, r1, c1 = _bbox(coords)
        local_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in coords:
            local_mask[r - r0, c - c0] = True

        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        bg_patch = patch == bg_color
        holes = []
        for hole_coords in _components(bg_patch):
            if any(
                rr == 0 or cc == 0 or rr == bg_patch.shape[0] - 1 or cc == bg_patch.shape[1] - 1
                for rr, cc in hole_coords
            ):
                continue
            holes.append(hole_coords)

        assert len(holes) == 1, "each object must contain exactly one enclosed hole"
        hole_mask, _ = _coords_to_mask(holes[0])
        objects.append({"color": color, "hole_mask": hole_mask})

    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each border notch with the color of the interior object whose hole has the same shape.

    Input structure:
    - Black background fills the interior canvas.
    - A gray frame of thickness 2 surrounds the grid, and some cells on its inner ring are missing to form small connected notches.
    - Inside the frame are several monochrome 4-connected objects, each containing exactly one enclosed background hole.

    Rule:
    - Extract every notch from the frame's second layer and record its exact pixel shape without rotating or flipping it.
    - For each interior object, extract its unique enclosed hole shape.
    - Match hole shapes to notch shapes one-to-one and recolor the matched notch cells with that object's color.
    - Leave the original objects, background, and the rest of the frame unchanged.

    Key insight:
    - The object's enclosed hole is the lookup key, while the border notch is the target to be painted with that object's color.
    """
    g = np.asarray(grid, dtype=int)
    frame_color = _outer_frame_color(g)
    bg_color = _background_color(g, frame_color)

    notches = _extract_notches(g, frame_color)
    objects = _extract_objects(g, frame_color, bg_color)
    assert len(notches) == len(objects), "hole count and notch count must match"

    notch_by_key: dict[tuple[tuple[int, int], bytes], dict[str, object]] = {}
    for notch in notches:
        key = _shape_key(np.asarray(notch["mask"], dtype=bool))
        assert key not in notch_by_key, "notch shapes must be unique"
        notch_by_key[key] = notch

    out = g.copy()
    for obj in objects:
        key = _shape_key(np.asarray(obj["hole_mask"], dtype=bool))
        assert key in notch_by_key, "every hole must match a notch"
        notch = notch_by_key.pop(key)
        for r, c in notch["coords"]:
            out[r, c] = int(obj["color"])

    assert not notch_by_key, "all notches must be matched"
    return out


def _candidate_notches(h: int, w: int) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []

    def add(coords: list[tuple[int, int]], kind: str) -> None:
        mask, bbox = _coords_to_mask(coords)
        candidates.append(
            {
                "coords": coords,
                "mask": mask,
                "bbox": bbox,
                "kind": kind,
                "key": _shape_key(mask),
            }
        )

    for length in range(1, 5):
        for c0 in range(1, w - length):
            add([(1, c) for c in range(c0, c0 + length)], "straight")
            add([(h - 2, c) for c in range(c0, c0 + length)], "straight")
        for r0 in range(1, h - length):
            add([(r, 1) for r in range(r0, r0 + length)], "straight")
            add([(r, w - 2) for r in range(r0, r0 + length)], "straight")

    corner_specs = [(2, 2), (2, 3), (3, 2)]
    for arm_h, arm_w in corner_specs:
        add(sorted({(1, c) for c in range(1, 1 + arm_w)} | {(r, 1) for r in range(1, 1 + arm_h)}), "corner")
        add(
            sorted({(1, c) for c in range(w - arm_w - 1, w - 1)} | {(r, w - 2) for r in range(1, 1 + arm_h)}),
            "corner",
        )
        add(
            sorted({(h - 2, c) for c in range(w - arm_w - 1, w - 1)} | {(r, w - 2) for r in range(h - arm_h - 1, h - 1)}),
            "corner",
        )
        add(
            sorted({(h - 2, c) for c in range(1, 1 + arm_w)} | {(r, 1) for r in range(h - arm_h - 1, h - 1)}),
            "corner",
        )

    return candidates


def _blocked_coords(coords: list[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    ring = _inner_ring_mask((h, w))
    blocked = set(coords)
    for r, c in coords:
        for dr, dc in _NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and bool(ring[nr, nc]):
                blocked.add((int(nr), int(nc)))
    return blocked


def _max_objects(min_side: int) -> int:
    for limit, max_objects in _MAX_OBJECTS_BY_MIN_SIDE:
        if min_side < limit:
            return max_objects
    return 4


def _notch_kind_plan(count: int, rng: random.Random) -> list[str]:
    if count == 1:
        return [rng.choice(["straight", "corner"])]
    mode = rng.choice(["straight_only", "corner_only", "mixed"])
    if mode == "straight_only":
        return ["straight"] * count
    if mode == "corner_only":
        return ["corner"] * count
    plan = ["straight", "corner"]
    plan.extend(rng.choice(["straight", "corner"]) for _ in range(count - 2))
    rng.shuffle(plan)
    return plan


def _choose_notches(h: int, w: int, count: int, rng: random.Random) -> list[dict[str, object]] | None:
    candidates = _candidate_notches(h, w)
    selected: list[dict[str, object]] = []
    used_keys: set[tuple[tuple[int, int], bytes]] = set()
    blocked: set[tuple[int, int]] = set()

    for kind in _notch_kind_plan(count, rng):
        pool = [
            cand
            for cand in candidates
            if cand["kind"] == kind
            and cand["key"] not in used_keys
            and all(coord not in blocked for coord in cand["coords"])
        ]
        if not pool:
            return None
        cand = rng.choice(pool)
        selected.append(cand)
        used_keys.add(cand["key"])
        blocked |= _blocked_coords(cand["coords"], h, w)

    return selected


def _make_object_patch(hole_mask: np.ndarray, color: int, rng: random.Random) -> np.ndarray:
    hole_h, hole_w = hole_mask.shape
    top = rng.randrange(*_OBJECT_MARGIN_RANGE)
    bottom = rng.randrange(*_OBJECT_MARGIN_RANGE)
    left = rng.randrange(*_OBJECT_MARGIN_RANGE)
    right = rng.randrange(*_OBJECT_MARGIN_RANGE)

    patch = np.full((top + hole_h + bottom, left + hole_w + right), color, dtype=int)
    inner = patch[top : top + hole_h, left : left + hole_w]
    inner[np.asarray(hole_mask, dtype=bool)] = _BG_COLOR
    return patch


def _place_objects(grid: np.ndarray, entries: list[dict[str, object]], rng: random.Random) -> bool:
    h, w = grid.shape
    occ = np.zeros((h, w), dtype=bool)
    order = sorted(range(len(entries)), key=lambda idx: -int(np.asarray(entries[idx]["patch"]).size))

    for idx in order:
        patch = np.asarray(entries[idx]["patch"], dtype=int)
        ph, pw = patch.shape
        positions = []
        for r0 in range(3, h - ph - 2):
            for c0 in range(3, w - pw - 2):
                if occ[r0 - 1 : r0 + ph + 1, c0 - 1 : c0 + pw + 1].any():
                    continue
                positions.append((r0, c0))
        if not positions:
            return False
        r0, c0 = rng.choice(positions)
        grid[r0 : r0 + ph, c0 : c0 + pw] = patch
        occ[r0 - 1 : r0 + ph + 1, c0 - 1 : c0 + pw + 1] = True

    return True


def _expected_output(grid: np.ndarray, entries: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    for entry in entries:
        for r, c in entry["coords"]:
            out[r, c] = int(entry["color"])
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randrange(*_GRID_H_RANGE)
    w = rng.randrange(*_GRID_W_RANGE)
    grid = np.full((h, w), _BG_COLOR, dtype=int)
    grid[:2, :] = _FRAME_COLOR
    grid[h - 2 :, :] = _FRAME_COLOR
    grid[:, :2] = _FRAME_COLOR
    grid[:, w - 2 :] = _FRAME_COLOR

    count = rng.randint(1, _max_objects(min(h, w)))
    notches = _choose_notches(h, w, count, rng)
    if notches is None:
        return None

    colors = rng.sample(_COLOR_POOL, count)
    entries: list[dict[str, object]] = []
    for notch, color in zip(notches, colors):
        hole_mask = np.asarray(notch["mask"], dtype=bool)
        patch = _make_object_patch(hole_mask, color, rng)
        for r, c in notch["coords"]:
            grid[r, c] = _BG_COLOR
        entries.append({"coords": notch["coords"], "color": color, "patch": patch, "hole_mask": hole_mask})

    if not _place_objects(grid, entries, rng):
        return None

    expected = _expected_output(grid, entries)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
