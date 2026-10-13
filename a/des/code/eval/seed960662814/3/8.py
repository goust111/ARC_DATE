from __future__ import annotations

import random

import numpy as np


BG_COLOR = 0
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
SIDES = ("top", "right", "bottom", "left")


def _boundary_bg(grid: np.ndarray) -> int:
    border = np.concatenate((grid[0, :], grid[-1, :], grid[1:-1, 0], grid[1:-1, -1]))
    values, counts = np.unique(border, return_counts=True)
    return int(values[counts.argmax()])


def _components_of_color(grid: np.ndarray, color: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            r0 = r1 = r
            c0 = c1 = c

            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                r0 = min(r0, cr)
                c0 = min(c0, cc)
                r1 = max(r1, cr)
                c1 = max(c1, cc)

                for dr, dc in DIRS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            components.append(
                {
                    "coords": np.array(coords, dtype=int),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(len(coords)),
                }
            )

    return components


def _bbox_size(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _is_solid_rectangle(comp: dict[str, object]) -> bool:
    bbox = comp["bbox"]
    area = int(comp["area"])
    height, width = _bbox_size(bbox)
    return area == height * width


def _component_mask(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1] == color, dtype=bool)


def _extract_pairs(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    pairs: list[dict[str, object]] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue

        components = _components_of_color(grid, color)
        lock_candidates = []
        for comp in components:
            bbox = comp["bbox"]
            height, width = _bbox_size(bbox)
            if _is_solid_rectangle(comp) and height >= 3 and width >= 3 and int(comp["area"]) > 5:
                lock_candidates.append(comp)

        assert len(lock_candidates) == 1, "each foreground color must have one lock rectangle"
        lock = lock_candidates[0]
        keys = [comp for comp in components if comp is not lock]
        assert len(keys) == 1, "each foreground color must have one key"
        key = keys[0]
        assert 2 <= int(key["area"]) <= 5, "key size must stay within the intended range"
        pairs.append({"color": color, "lock": lock, "key": key})

    assert pairs, "expected at least one lock/key pair"
    return pairs


def _relative_side(lock_bbox: tuple[int, int, int, int], key_bbox: tuple[int, int, int, int]) -> str:
    lr0, lc0, lr1, lc1 = lock_bbox
    kr0, kc0, kr1, kc1 = key_bbox

    row_overlap = not (kr1 < lr0 or lr1 < kr0)
    col_overlap = not (kc1 < lc0 or lc1 < kc0)
    candidates: list[str] = []

    if col_overlap and kr1 < lr0:
        candidates.append("top")
    if row_overlap and kc0 > lc1:
        candidates.append("right")
    if col_overlap and kr0 > lr1:
        candidates.append("bottom")
    if row_overlap and kc1 < lc0:
        candidates.append("left")

    if len(candidates) == 1:
        return candidates[0]

    distances: dict[str, int] = {}
    if kr1 < lr0:
        distances["top"] = int(lr0 - kr1)
    if kc0 > lc1:
        distances["right"] = int(kc0 - lc1)
    if kr0 > lr1:
        distances["bottom"] = int(kr0 - lr1)
    if kc1 < lc0:
        distances["left"] = int(lc0 - kc1)

    assert distances, "key must lie outside its lock"
    best = min(distances.values())
    sides = [side for side, dist in distances.items() if dist == best]
    assert len(sides) == 1, "key direction must be unambiguous"
    return sides[0]


def _anchor_for_side(
    lock_bbox: tuple[int, int, int, int],
    key_shape: tuple[int, int],
    side: str,
) -> tuple[int, int]:
    lr0, lc0, lr1, lc1 = lock_bbox
    kh, kw = key_shape
    interior_h = lr1 - lr0 - 1
    interior_w = lc1 - lc0 - 1
    assert kh <= interior_h and kw <= interior_w, "key must fit inside the lock interior"

    if side == "top":
        return int(lr0 + 1), int(lc0 + 1)
    if side == "right":
        return int(lr0 + 1), int(lc1 - kw)
    if side == "bottom":
        return int(lr1 - kh), int(lc1 - kw)
    assert side == "left"
    return int(lr1 - kh), int(lc0 + 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each solid-color lock rectangle with the same-color key sprite, then punch the
    key shape into the corresponding interior corner determined by the key's side.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _boundary_bg(g)
    pairs = _extract_pairs(g, bg)

    out = g.copy()
    for pair in pairs:
        color = int(pair["color"])
        lock = pair["lock"]
        key = pair["key"]
        lock_bbox = lock["bbox"]
        key_bbox = key["bbox"]
        key_coords = np.asarray(key["coords"], dtype=int)
        key_mask = _component_mask(g, key_bbox, color)
        side = _relative_side(lock_bbox, key_bbox)
        start_r, start_c = _anchor_for_side(lock_bbox, key_mask.shape, side)

        out[key_coords[:, 0], key_coords[:, 1]] = bg
        region = out[start_r : start_r + key_mask.shape[0], start_c : start_c + key_mask.shape[1]]
        region[key_mask] = bg

    return out


KEY_LIBRARY = (
    np.array([[1, 1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 1, 0]], dtype=bool),
)
GRID_HEIGHT_RANGE = (18, 28)
GRID_WIDTH_RANGE = (18, 28)
PAIR_COUNT_CHOICES = (2, 3, 4)
PAIR_BOX_PAD = 1
PLACEMENT_TRIES = 80
GENERATION_ATTEMPTS = 400
LOCK_EXTRA_RANGE = (2, 5)
GAP_RANGE = (1, 3)


def _random_key_mask() -> np.ndarray:
    mask = random.choice(KEY_LIBRARY).copy()
    if random.random() < 0.5:
        mask = mask.T
    if random.random() < 0.5:
        mask = np.flipud(mask)
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    return np.array(mask, dtype=bool, copy=True)


def _rects_clear(rect: tuple[int, int, int, int], used: list[tuple[int, int, int, int]], pad: int) -> bool:
    r0, c0, r1, c1 = rect
    for ur0, uc0, ur1, uc1 in used:
        if not (r1 + pad < ur0 or ur1 + pad < r0 or c1 + pad < uc0 or uc1 + pad < c0):
            return False
    return True


def _draw_pair(
    grid: np.ndarray,
    color: int,
    direction: str,
    key_mask: np.ndarray,
    lock_h: int,
    lock_w: int,
    gap: int,
    base_r: int,
    base_c: int,
) -> None:
    kh, kw = key_mask.shape
    if direction == "top":
        lock_r = base_r + kh + gap
        lock_c = base_c
        key_r = base_r
        key_c = base_c + random.randint(0, lock_w - kw)
    elif direction == "bottom":
        lock_r = base_r
        lock_c = base_c
        key_r = base_r + lock_h + gap
        key_c = base_c + random.randint(0, lock_w - kw)
    elif direction == "left":
        lock_r = base_r
        lock_c = base_c + kw + gap
        key_r = base_r + random.randint(0, lock_h - kh)
        key_c = base_c
    else:
        assert direction == "right"
        lock_r = base_r
        lock_c = base_c
        key_r = base_r + random.randint(0, lock_h - kh)
        key_c = base_c + lock_w + gap

    grid[lock_r : lock_r + lock_h, lock_c : lock_c + lock_w] = color
    grid[key_r : key_r + kh, key_c : key_c + kw][key_mask] = color


def _sample_pair_specs(colors: list[int], directions: list[str]) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for color, direction in zip(colors, directions, strict=True):
        key_mask = _random_key_mask()
        kh, kw = key_mask.shape
        lock_h = random.randint(max(4, kh + 2), max(5, kh + LOCK_EXTRA_RANGE[1]))
        lock_w = random.randint(max(4, kw + 2), max(5, kw + LOCK_EXTRA_RANGE[1]))
        gap = random.randint(*GAP_RANGE)

        if direction in {"top", "bottom"}:
            pair_h = kh + gap + lock_h
            pair_w = lock_w
        else:
            pair_h = lock_h
            pair_w = kw + gap + lock_w

        specs.append(
            {
                "color": color,
                "direction": direction,
                "key_mask": key_mask,
                "lock_h": lock_h,
                "lock_w": lock_w,
                "gap": gap,
                "pair_h": pair_h,
                "pair_w": pair_w,
            }
        )
    random.shuffle(specs)
    return specs


def _generate_once() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    pair_count = random.choice(PAIR_COUNT_CHOICES)
    colors = random.sample(list(range(1, 10)), k=pair_count)
    directions = random.sample(list(SIDES), k=pair_count)
    specs = _sample_pair_specs(colors, directions)

    grid = np.full((height, width), BG_COLOR, dtype=int)
    used: list[tuple[int, int, int, int]] = []

    for spec in specs:
        pair_h = int(spec["pair_h"])
        pair_w = int(spec["pair_w"])
        if pair_h > height - 2 or pair_w > width - 2:
            return None

        placed = False
        for _ in range(PLACEMENT_TRIES):
            base_r = random.randint(1, height - pair_h - 1)
            base_c = random.randint(1, width - pair_w - 1)
            rect = (base_r, base_c, base_r + pair_h - 1, base_c + pair_w - 1)
            if not _rects_clear(rect, used, PAIR_BOX_PAD):
                continue

            _draw_pair(
                grid,
                color=int(spec["color"]),
                direction=str(spec["direction"]),
                key_mask=np.asarray(spec["key_mask"], dtype=bool),
                lock_h=int(spec["lock_h"]),
                lock_w=int(spec["lock_w"]),
                gap=int(spec["gap"]),
                base_r=base_r,
                base_c=base_c,
            )
            used.append(rect)
            placed = True
            break

        if not placed:
            return None

    pairs = _extract_pairs(grid, BG_COLOR)
    assert len(pairs) == pair_count
    solved = main(grid)
    if np.array_equal(solved, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid lock/key puzzle")
