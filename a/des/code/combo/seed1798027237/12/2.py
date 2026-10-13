from __future__ import annotations

from collections import deque

import numpy as np


_RNG = np.random.default_rng()
_NEIGHBORS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

_SIZE_CHOICES = [
    (3, 3),
    (3, 4),
    (4, 4),
    (3, 5),
    (4, 5),
    (5, 5),
    (3, 6),
    (4, 6),
]
_PAIR_COUNT_CHOICES = (2, 3, 4)
_GRID_H_RANGE = (20, 31)
_GRID_W_RANGE = (20, 31)
_PLACEMENT_TRIES = 250
_SHAPE_TRIES = 400
_GENERATION_TRIES = 500


def _get_bg(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(values, minlength=10).argmax())


def _mask_connected(mask: np.ndarray) -> bool:
    mask = np.asarray(mask, dtype=bool)
    points = np.argwhere(mask)
    if len(points) == 0:
        return False

    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    start_r, start_c = map(int, points[0])
    seen[start_r, start_c] = True
    queue: deque[tuple[int, int]] = deque([(start_r, start_c)])
    seen_count = 1

    while queue:
        r, c = queue.popleft()
        for dr, dc in _NEIGHBORS8:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                seen_count += 1
                queue.append((rr, cc))

    return seen_count == int(mask.sum())


def _valid_fragment_mask(mask: np.ndarray) -> bool:
    mask = np.asarray(mask, dtype=bool)
    total = int(mask.size)
    area = int(mask.sum())
    if area <= 0 or area >= total:
        return False
    if area < max(3, total // 4) or area > min(total - 3, (3 * total) // 4):
        return False
    if not np.all(mask.any(axis=0)) or not np.all(mask.any(axis=1)):
        return False
    comp = ~mask
    if not np.all(comp.any(axis=0)) or not np.all(comp.any(axis=1)):
        return False
    if not _mask_connected(mask) or not _mask_connected(comp):
        return False
    return True


def _make_random_fragment(h: int, w: int) -> np.ndarray:
    for _ in range(_SHAPE_TRIES):
        p = float(_RNG.uniform(0.35, 0.65))
        mask = _RNG.random((h, w)) < p
        if _valid_fragment_mask(mask):
            return mask
    raise RuntimeError(f"failed to make a valid {h}x{w} fragment")


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            queue: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in _NEIGHBORS8:
                    nr = rr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True
            comps.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                    "area": int(mask.sum()),
                }
            )

    comps.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["bbox"][2], comp["bbox"][3]))
    return comps


def _valid_rotations(mask_a: np.ndarray, mask_b: np.ndarray) -> list[int]:
    rots: list[int] = []
    for k in range(4):
        rotated = np.rot90(mask_b, k)
        if rotated.shape == mask_a.shape and np.array_equal(mask_a, ~rotated):
            rots.append(k)
    return rots


def _assemble_pair(anchor: dict, other: dict) -> np.ndarray | None:
    if anchor["color"] == other["color"]:
        return None
    rots = _valid_rotations(anchor["mask"], other["mask"])
    if not rots:
        return None
    k = min(rots)
    other_mask = np.rot90(other["mask"], k)
    rect = np.full(anchor["mask"].shape, -1, dtype=int)
    rect[anchor["mask"]] = int(anchor["color"])
    rect[other_mask] = int(other["color"])
    return rect


def _pair_candidates(components: list[dict]) -> dict[tuple[int, int], np.ndarray]:
    candidates: dict[tuple[int, int], np.ndarray] = {}
    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            rect = _assemble_pair(components[i], components[j])
            if rect is not None:
                candidates[(i, j)] = rect
    return candidates


def _find_pairings(components: list[dict], limit: int = 1) -> list[list[tuple[int, int]]]:
    candidates = _pair_candidates(components)
    count = len(components)
    used = [False] * count
    solutions: list[list[tuple[int, int]]] = []

    def rec(current: list[tuple[int, int]]) -> None:
        if len(solutions) >= limit:
            return
        i = 0
        while i < count and used[i]:
            i += 1
        if i == count:
            solutions.append(current.copy())
            return

        used[i] = True
        for j in range(i + 1, count):
            if used[j] or (i, j) not in candidates:
                continue
            used[j] = True
            current.append((i, j))
            rec(current)
            current.pop()
            used[j] = False
            if len(solutions) >= limit:
                break
        used[i] = False

    rec([])
    return solutions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair monochrome 8-connected fragments by rotated complement shape, complete each rectangle,
    then pack the finished rectangles from smallest area to largest.

    When a pair can be aligned in several ways, keep the earlier fragment in reading order fixed
    and rotate its mate by the smallest valid quarter-turn.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    components = _find_components(g, bg)
    assert components and len(components) % 2 == 0

    pairings = _find_pairings(components, limit=1)
    assert pairings, "no valid complement pairing found"

    candidates = _pair_candidates(components)
    rects: list[tuple[tuple[int, int, int], np.ndarray]] = []
    for i, j in pairings[0]:
        rect = candidates[(i, j)]
        area = int(rect.size)
        left_anchor = min(int(components[i]["bbox"][1]), int(components[j]["bbox"][1]))
        top_anchor = min(int(components[i]["bbox"][0]), int(components[j]["bbox"][0]))
        rects.append(((area, left_anchor, top_anchor), rect))

    rects.sort(key=lambda item: item[0])
    out_h = max(int(rect.shape[0]) for _, rect in rects)
    out_w = sum(int(rect.shape[1]) for _, rect in rects) + max(0, len(rects) - 1)
    out = np.full((out_h, out_w), bg, dtype=int)

    c0 = 0
    for _, rect in rects:
        h, w = rect.shape
        out[out_h - h : out_h, c0 : c0 + w] = rect
        c0 += w + 1

    return out


def _sample_pair_sizes(pair_count: int) -> list[tuple[int, int]]:
    choices = list(_SIZE_CHOICES)
    _RNG.shuffle(choices)
    used_areas: set[int] = set()
    picked: list[tuple[int, int]] = []
    for h, w in choices:
        area = h * w
        if area in used_areas:
            continue
        picked.append((h, w))
        used_areas.add(area)
        if len(picked) == pair_count:
            return picked
    raise RuntimeError("not enough unique rectangle areas")


def _place_objects(objects: list[dict], bg: int) -> np.ndarray | None:
    max_h = max(int(obj["patch"].shape[0]) for obj in objects)
    max_w = max(int(obj["patch"].shape[1]) for obj in objects)
    for _ in range(_PLACEMENT_TRIES):
        grid_h = int(_RNG.integers(max(max_h + 3, _GRID_H_RANGE[0]), _GRID_H_RANGE[1]))
        grid_w = int(_RNG.integers(max(max_w + 3, _GRID_W_RANGE[0]), _GRID_W_RANGE[1]))
        grid = np.full((grid_h, grid_w), bg, dtype=int)
        blocked = np.zeros((grid_h, grid_w), dtype=bool)
        placements: list[tuple[int, int] | None] = [None] * len(objects)

        order = sorted(range(len(objects)), key=lambda idx: objects[idx]["patch"].size, reverse=True)
        ok = True
        for idx in order:
            patch = objects[idx]["patch"]
            h, w = patch.shape
            placed = False
            for _ in range(_PLACEMENT_TRIES):
                r0 = int(_RNG.integers(0, grid_h - h + 1))
                c0 = int(_RNG.integers(0, grid_w - w + 1))
                rr0 = max(0, r0 - 1)
                rr1 = min(grid_h, r0 + h + 1)
                cc0 = max(0, c0 - 1)
                cc1 = min(grid_w, c0 + w + 1)
                if blocked[rr0:rr1, cc0:cc1].any():
                    continue
                color = int(objects[idx]["color"])
                grid_slice = grid[r0 : r0 + h, c0 : c0 + w]
                grid_slice[patch] = color
                blocked[rr0:rr1, cc0:cc1] = True
                placements[idx] = (r0, c0)
                placed = True
                break
            if not placed:
                ok = False
                break

        if not ok:
            continue
        assert all(pos is not None for pos in placements)
        return grid
    return None


def _scene_is_valid(grid: np.ndarray, expected_pairs: int) -> bool:
    bg = _get_bg(grid)
    comps = _find_components(grid, bg)
    if len(comps) != 2 * expected_pairs:
        return False
    pairings = _find_pairings(comps, limit=2)
    if len(pairings) != 1:
        return False
    try:
        out = main(grid)
    except AssertionError:
        return False
    if not isinstance(out, np.ndarray) or out.size == 0:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    pair_count = int(_RNG.choice(_PAIR_COUNT_CHOICES))
    bg = int(_RNG.integers(0, 10))
    colors = [c for c in range(10) if c != bg]
    _RNG.shuffle(colors)
    obj_colors = colors[: 2 * pair_count]

    objects: list[dict] = []
    for pair_idx, (h, w) in enumerate(_sample_pair_sizes(pair_count)):
        mask = _make_random_fragment(h, w)
        comp = ~mask
        color_a = int(obj_colors[2 * pair_idx])
        color_b = int(obj_colors[2 * pair_idx + 1])

        for piece_mask, color in ((mask, color_a), (comp, color_b)):
            rot = int(_RNG.integers(0, 4))
            patch = np.rot90(piece_mask, rot)
            objects.append({"patch": patch, "color": color})

    _RNG.shuffle(objects)
    grid = _place_objects(objects, bg)
    if grid is None:
        return None
    if not _scene_is_valid(grid, pair_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GENERATION_TRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
