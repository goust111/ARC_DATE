import random
from collections import Counter

import numpy as np


MAX_GENERATION_ATTEMPTS = 400
PAIR_COUNT_CHOICES = (2, 3)
GRID_MIN = 18
GRID_MAX = 30
LOCK_SIZE_RANGE = (5, 9)  # inclusive
PLACEMENT_TRIES = 120
SHAPE_TRIES = 100
PADDING = 1


def _get_bg(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                if rr > 0 and mask[rr - 1, cc] and not seen[rr - 1, cc]:
                    seen[rr - 1, cc] = True
                    stack.append((rr - 1, cc))
                if rr + 1 < h and mask[rr + 1, cc] and not seen[rr + 1, cc]:
                    seen[rr + 1, cc] = True
                    stack.append((rr + 1, cc))
                if cc > 0 and mask[rr, cc - 1] and not seen[rr, cc - 1]:
                    seen[rr, cc - 1] = True
                    stack.append((rr, cc - 1))
                if cc + 1 < w and mask[rr, cc + 1] and not seen[rr, cc + 1]:
                    seen[rr, cc + 1] = True
                    stack.append((rr, cc + 1))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _tight_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (r0, c0, r1, c1)


def _shape_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(mask.shape), mask.astype(np.uint8).tobytes()


def _border_positions(h: int, w: int) -> list[tuple[int, int]]:
    pts = [(0, c) for c in range(w)]
    pts += [(h - 1, c) for c in range(w) if h - 1 != 0]
    pts += [(r, 0) for r in range(1, h - 1)]
    pts += [(r, w - 1) for r in range(1, h - 1) if w - 1 != 0]
    return pts


def _is_inside_bbox(bbox: tuple[int, int, int, int], point: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    r, c = point
    return r0 <= r <= r1 and c0 <= c <= c1


def _bbox_inside_any_lock(
    bbox: tuple[int, int, int, int], lock_bboxes: list[tuple[int, int, int, int]]
) -> bool:
    r0, c0, r1, c1 = bbox
    for lr0, lc0, lr1, lc1 in lock_bboxes:
        if lr0 <= r0 <= r1 <= lr1 and lc0 <= c0 <= c1 <= lc1:
            return True
    return False


def _parse_lock_component(
    grid: np.ndarray, bg: int, coords: np.ndarray
) -> dict[str, object] | None:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        return None

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    if np.any(sub[0, :] == bg) or np.any(sub[-1, :] == bg):
        return None
    if np.any(sub[:, 0] == bg) or np.any(sub[:, -1] == bg):
        return None

    fg = sub[sub != bg]
    colors, counts = np.unique(fg, return_counts=True)
    if len(colors) != 2:
        return None

    singleton_idx = np.flatnonzero(counts == 1)
    if len(singleton_idx) != 1:
        return None

    hint_color = int(colors[int(singleton_idx[0])])
    lock_color = int(colors[1 - int(singleton_idx[0])])

    hint_cells = np.argwhere(sub == hint_color)
    if hint_cells.shape[0] != 1:
        return None
    hr, hc = map(int, hint_cells[0])
    if hr not in {0, h - 1} and hc not in {0, w - 1}:
        return None
    if h <= 2 or w <= 2:
        return None
    if np.any(sub[1:-1, 1:-1] == hint_color):
        return None

    interior = sub[1:-1, 1:-1]
    if interior.size == 0:
        return None
    if not np.all((interior == bg) | (interior == lock_color)):
        return None

    hole_mask = interior == bg
    hole_comps = _components(hole_mask)
    if len(hole_comps) != 1:
        return None

    hole_local = hole_comps[0]
    hole_tight, hole_bbox_local = _tight_mask(hole_local)
    if hole_tight.sum() == 0:
        return None

    local_r0, local_c0, local_r1, local_c1 = hole_bbox_local
    hole_bbox = (
        r0 + 1 + local_r0,
        c0 + 1 + local_c0,
        r0 + 1 + local_r1,
        c0 + 1 + local_c1,
    )

    return {
        "bbox": (r0, c0, r1, c1),
        "lock_color": lock_color,
        "key_color": hint_color,
        "hint": (r0 + hr, c0 + hc),
        "hole_mask": hole_tight,
        "hole_bbox": hole_bbox,
    }


def _find_locks(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    locks: list[dict[str, object]] = []
    for comp in _components(grid != bg):
        lock = _parse_lock_component(grid, bg, comp)
        if lock is not None:
            locks.append(lock)
    locks.sort(key=lambda d: tuple(d["bbox"]))
    return locks


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each key sprite to the lock whose border hint uses the same color, then fill the lock solid.

    Input structure:
    - The background is black or otherwise the dominant color.
    - Each lock is a single connected rectangular object: mostly one lock color, with one interior background-shaped hole and a one-pixel hint on the outer border in a different color.
    - Each key is a separate monochrome sprite whose color matches exactly one lock hint and whose shape matches that lock's hole.

    Rule:
    - Detect every lock by finding connected rectangular components with a full non-background border, a single off-color border hint, and exactly one enclosed background hole.
    - For each lock, use the hint color to find the separate key component of that color, and verify that the key's tight shape equals the hole's tight shape.
    - Remove the key from its original location and repaint the entire lock rectangle, including the former hole and hint, with the lock color.

    Key insight:
    - The border hint color is an indirect pairing key: it does not belong to the finished lock, but instead identifies which remote same-colored sprite should be translated into the hole.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    locks = _find_locks(g, bg)
    assert locks, "expected at least one lock"

    color_components: dict[int, list[dict[str, object]]] = {}
    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        comps: list[dict[str, object]] = []
        for coords in _components(g == color):
            mask, bbox = _tight_mask(coords)
            comps.append({"coords": coords, "bbox": bbox, "mask": mask})
        color_components[color] = comps

    lock_bboxes = [tuple(lock["bbox"]) for lock in locks]
    out = g.copy()

    for lock in locks:
        key_color = int(lock["key_color"])
        hole_sig = _shape_signature(np.asarray(lock["hole_mask"], dtype=bool))
        hint = tuple(lock["hint"])
        matches: list[dict[str, object]] = []

        for comp in color_components.get(key_color, []):
            coords = np.asarray(comp["coords"], dtype=int)
            bbox = tuple(comp["bbox"])
            if coords.shape[0] == 1 and tuple(map(int, coords[0])) == hint:
                continue
            if _bbox_inside_any_lock(bbox, lock_bboxes):
                continue
            if _shape_signature(np.asarray(comp["mask"], dtype=bool)) == hole_sig:
                matches.append(comp)

        assert len(matches) == 1, "each lock must match exactly one same-colored key"
        key_comp = matches[0]
        key_coords = np.asarray(key_comp["coords"], dtype=int)
        out[key_coords[:, 0], key_coords[:, 1]] = bg

        r0, c0, r1, c1 = map(int, lock["bbox"])
        out[r0 : r1 + 1, c0 : c1 + 1] = int(lock["lock_color"])

    return out


def _random_connected_shape(rng: random.Random, max_h: int, max_w: int) -> np.ndarray:
    for _ in range(SHAPE_TRIES):
        h = rng.randint(2, max_h)
        w = rng.randint(2, max_w)
        if h * w <= 3:
            continue

        target = rng.randint(3, h * w - 1)
        mask = np.zeros((h, w), dtype=bool)
        start = (rng.randrange(h), rng.randrange(w))
        mask[start] = True
        cells = {start}
        frontier: set[tuple[int, int]] = set()

        def add_frontier(rr: int, cc: int) -> None:
            for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((nr, nc))

        add_frontier(*start)
        while len(cells) < target and frontier:
            rr, cc = rng.choice(tuple(frontier))
            frontier.remove((rr, cc))
            if mask[rr, cc]:
                continue
            mask[rr, cc] = True
            cells.add((rr, cc))
            add_frontier(rr, cc)

        if int(mask.sum()) != target:
            continue

        tight, _ = _tight_mask(np.argwhere(mask))
        if tight.shape[0] < 2 or tight.shape[1] < 2:
            continue
        if bool(tight.all()):
            continue
        return tight

    raise RuntimeError("failed to build a non-rectangular connected hole")


def _can_place(blocked: np.ndarray, top: int, left: int, h: int, w: int, pad: int = PADDING) -> bool:
    hh, ww = blocked.shape
    if top < 0 or left < 0 or top + h > hh or left + w > ww:
        return False
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(hh, top + h + pad)
    c1 = min(ww, left + w + pad)
    return not bool(blocked[r0:r1, c0:c1].any())


def _mark_blocked(blocked: np.ndarray, top: int, left: int, h: int, w: int, pad: int = PADDING) -> None:
    hh, ww = blocked.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(hh, top + h + pad)
    c1 = min(ww, left + w + pad)
    blocked[r0:r1, c0:c1] = True


def _place_random_bbox(
    rng: random.Random, blocked: np.ndarray, h: int, w: int
) -> tuple[int, int] | None:
    gh, gw = blocked.shape
    for _ in range(PLACEMENT_TRIES):
        top = rng.randint(0, gh - h)
        left = rng.randint(0, gw - w)
        if _can_place(blocked, top, left, h, w):
            _mark_blocked(blocked, top, left, h, w)
            return top, left
    return None


def _random_hint_cell(rng: random.Random, h: int, w: int) -> tuple[int, int]:
    side = rng.choice(["top", "bottom", "left", "right"])
    if side == "top":
        return 0, rng.randrange(w)
    if side == "bottom":
        return h - 1, rng.randrange(w)
    if side == "left":
        return rng.randrange(h), 0
    return rng.randrange(h), w - 1


def _gen_attempt(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    pair_count = rng.choice(PAIR_COUNT_CHOICES)
    grid_h = rng.randint(max(GRID_MIN, 16 + 2 * pair_count), GRID_MAX)
    grid_w = rng.randint(max(GRID_MIN, 16 + 2 * pair_count), GRID_MAX)
    grid = np.zeros((grid_h, grid_w), dtype=int)
    expected = np.zeros_like(grid)
    blocked = np.zeros_like(grid, dtype=bool)

    palette = rng.sample(list(range(1, 10)), 2 * pair_count)
    rng.shuffle(palette)
    lock_colors = palette[:pair_count]
    key_colors = palette[pair_count:]

    placements: list[dict[str, object]] = []

    for lock_color, key_color in zip(lock_colors, key_colors):
        lock_h = rng.randint(*LOCK_SIZE_RANGE)
        lock_w = rng.randint(*LOCK_SIZE_RANGE)
        inner_h = lock_h - 2
        inner_w = lock_w - 2
        if inner_h < 2 or inner_w < 2:
            return None

        hole_mask = _random_connected_shape(rng, inner_h, inner_w)
        hole_h, hole_w = hole_mask.shape
        hole_top = rng.randint(0, inner_h - hole_h)
        hole_left = rng.randint(0, inner_w - hole_w)

        lock_pos = _place_random_bbox(rng, blocked, lock_h, lock_w)
        if lock_pos is None:
            return None
        lock_top, lock_left = lock_pos

        key_pos = _place_random_bbox(rng, blocked, hole_h, hole_w)
        if key_pos is None:
            return None
        key_top, key_left = key_pos

        grid[lock_top : lock_top + lock_h, lock_left : lock_left + lock_w] = lock_color
        expected[lock_top : lock_top + lock_h, lock_left : lock_left + lock_w] = lock_color

        hole_rows, hole_cols = np.where(hole_mask)
        global_hole_rows = lock_top + 1 + hole_top + hole_rows
        global_hole_cols = lock_left + 1 + hole_left + hole_cols
        grid[global_hole_rows, global_hole_cols] = 0

        hint_r, hint_c = _random_hint_cell(rng, lock_h, lock_w)
        grid[lock_top + hint_r, lock_left + hint_c] = key_color

        key_patch = np.zeros((hole_h, hole_w), dtype=int)
        key_patch[hole_mask] = key_color
        region = grid[key_top : key_top + hole_h, key_left : key_left + hole_w]
        region[hole_mask] = key_color

        placements.append(
            {
                "lock_bbox": (lock_top, lock_left, lock_top + lock_h - 1, lock_left + lock_w - 1),
                "key_color": key_color,
                "key_bbox": (key_top, key_left, key_top + hole_h - 1, key_left + hole_w - 1),
                "key_mask": hole_mask,
            }
        )

    for item in placements:
        kr0, kc0, _, _ = map(int, item["key_bbox"])
        key_mask = np.asarray(item["key_mask"], dtype=bool)
        key_rows, key_cols = np.where(key_mask)
        expected[kr0 + key_rows, kc0 + key_cols] = 0

    if np.array_equal(grid, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            result = _gen_attempt(rng)
        except Exception:
            continue
        if result is None:
            continue
        grid, expected = result
        if np.array_equal(main(grid), expected):
            return grid
    raise RuntimeError("failed to generate a valid key-lock puzzle")
