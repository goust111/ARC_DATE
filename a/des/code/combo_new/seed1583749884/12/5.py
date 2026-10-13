import random
from collections import Counter, deque

import numpy as np


BG = 8
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
COLORS = (1, 2, 3, 4, 6, 7, 9)
GEN_RETRIES = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIR4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _unique_transforms(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[bytes, tuple[int, int]]] = set()
    for base in (mask, np.fliplr(mask)):
        for k in range(4):
            cur = _normalize_mask(np.rot90(base, k))
            key = (cur.tobytes(), cur.shape)
            if key not in seen:
                seen.add(key)
                variants.append(cur)
    return variants


_BASE_MASKS = [
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1, 1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
]
_SHAPES_BY_AREA: dict[int, list[np.ndarray]] = {1: [], 2: [], 3: [], 4: []}
for _base in _BASE_MASKS:
    for _variant in _unique_transforms(_base):
        _SHAPES_BY_AREA[int(_variant.sum())].append(_variant)


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rows = [rr for rr, _ in cells]
            cols = [cc for _, cc in cells]
            comps.append(
                {
                    "color": color,
                    "cells": cells,
                    "cell_set": set(cells),
                    "area": len(cells),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return comps


def _ordered_objects(grid: np.ndarray, bg: int, reverse_ties: bool = False) -> list[dict]:
    groups: dict[int, list[dict]] = {}
    for obj in _components(grid, bg):
        groups.setdefault(int(obj["area"]), []).append(obj)

    ordered: list[dict] = []
    for area in sorted(groups, reverse=True):
        group = sorted(groups[area], key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))
        if reverse_ties:
            group = list(reversed(group))
        ordered.extend(group)
    return ordered


def _apply_diffusion(grid: np.ndarray, objects: list[dict], bg: int) -> np.ndarray:
    out = np.array(grid, copy=True)
    h, w = out.shape
    for obj in objects:
        color = int(obj["color"])
        radius = int(obj["area"])
        cell_set = obj["cell_set"]
        for r, c in sorted(obj["cells"]):
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if (nr, nc) in cell_set:
                    continue
                rr, cc = nr, nc
                for _ in range(radius):
                    if not (0 <= rr < h and 0 <= cc < w):
                        break
                    if int(out[rr, cc]) != bg:
                        break
                    out[rr, cc] = color
                    rr += dr
                    cc += dc
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Diffuse each small foreground object outward by its own area while larger objects claim shared corridors first.

    Input structure:
    - The background is azure and occupies most of the grid.
    - Several separated 4-connected monochrome foreground objects appear in other colors.
    - Every object has area 1 to 4 and may have any small polyomino shape.

    Rule:
    - Keep every original object unchanged and compute its area N.
    - For each exposed object side, extend a straight orthogonal ray outward for up to N cells.
    - Recolor only background cells on that ray; stop a ray as soon as it reaches the grid edge or any already-colored cell.
    - Process larger-area objects before smaller-area objects so earlier influence regions block later ones.

    Key insight:
    - Object area controls both the diffusion distance and the priority, so the task is a size-ordered ray expansion from every boundary edge.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    objects = _ordered_objects(g, bg)
    return _apply_diffusion(g, objects, bg)


def _can_place(mask: np.ndarray, top: int, left: int, guard: np.ndarray) -> bool:
    h, w = guard.shape
    ph, pw = mask.shape
    if top < 0 or left < 0 or top + ph > h or left + pw > w:
        return False
    for rr, cc in np.argwhere(mask):
        if guard[top + int(rr), left + int(cc)]:
            return False
    return True


def _stamp_object(
    grid: np.ndarray,
    guard: np.ndarray,
    mask: np.ndarray,
    top: int,
    left: int,
    color: int,
) -> None:
    h, w = grid.shape
    for rr, cc in np.argwhere(mask):
        r = top + int(rr)
        c = left + int(cc)
        grid[r, c] = color
    for rr, cc in np.argwhere(mask):
        r = top + int(rr)
        c = left + int(cc)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    guard[nr, nc] = True


def _place_random_object(
    grid: np.ndarray,
    guard: np.ndarray,
    color: int,
    rng: random.Random,
    allowed_areas: list[int],
) -> int | None:
    h, w = grid.shape
    for _ in range(120):
        area = int(rng.choice(allowed_areas))
        mask = rng.choice(_SHAPES_BY_AREA[area])
        ph, pw = mask.shape
        top = rng.randint(0, h - ph)
        left = rng.randint(0, w - pw)
        if not _can_place(mask, top, left, guard):
            continue
        _stamp_object(grid, guard, mask, top, left, color)
        return area
    return None


def _place_edge_object(
    grid: np.ndarray,
    guard: np.ndarray,
    color: int,
    rng: random.Random,
    allowed_areas: list[int],
) -> int | None:
    h, w = grid.shape
    sides = ["top", "bottom", "left", "right"]
    for _ in range(160):
        area = int(rng.choice(allowed_areas))
        mask = rng.choice(_SHAPES_BY_AREA[area])
        ph, pw = mask.shape
        side = rng.choice(sides)
        if side == "top":
            top = 0
            left = rng.randint(0, w - pw)
        elif side == "bottom":
            top = h - ph
            left = rng.randint(0, w - pw)
        elif side == "left":
            top = rng.randint(0, h - ph)
            left = 0
        else:
            top = rng.randint(0, h - ph)
            left = w - pw
        if not _can_place(mask, top, left, guard):
            continue
        _stamp_object(grid, guard, mask, top, left, color)
        return area
    return None


def _place_blocking_pair(
    grid: np.ndarray,
    guard: np.ndarray,
    colors: list[int],
    rng: random.Random,
) -> list[int] | None:
    h, w = grid.shape
    axis = rng.choice(["horizontal", "vertical"])
    large_area = rng.choice([3, 4])
    large_mask = np.ones((1, large_area), dtype=bool) if axis == "horizontal" else np.ones((large_area, 1), dtype=bool)
    small_mask = np.ones((1, 1), dtype=bool)
    gap = large_area

    for _ in range(120):
        if axis == "horizontal":
            total_w = large_mask.shape[1] + gap + small_mask.shape[1]
            if total_w > w:
                return None
            row = rng.randint(0, h - 1)
            left = rng.randint(0, w - total_w)
            placements = [
                (large_mask, row, left, colors[0]),
                (small_mask, row, left + large_mask.shape[1] + gap, colors[1]),
            ]
        else:
            total_h = large_mask.shape[0] + gap + small_mask.shape[0]
            if total_h > h:
                return None
            top = rng.randint(0, h - total_h)
            col = rng.randint(0, w - 1)
            placements = [
                (large_mask, top, col, colors[0]),
                (small_mask, top + large_mask.shape[0] + gap, col, colors[1]),
            ]

        if all(_can_place(mask, top0, left0, guard) for mask, top0, left0, _ in placements):
            for mask, top0, left0, color in placements:
                _stamp_object(grid, guard, mask, top0, left0, color)
            return [large_area, 1]
    return None


def _uses_every_rule_branch(inp: np.ndarray, out: np.ndarray) -> bool:
    bg = _bg_color(inp)
    objects = _components(inp, bg)
    if not objects:
        return False

    new_cells = np.argwhere((inp == bg) & (out != bg))
    if len(new_cells) < 6:
        return False

    touches_edge = any(
        obj["bbox"][0] == 0
        or obj["bbox"][1] == 0
        or obj["bbox"][2] == inp.shape[0] - 1
        or obj["bbox"][3] == inp.shape[1] - 1
        for obj in objects
    )
    blocked = False
    for obj in objects:
        color = int(obj["color"])
        for r, c in obj["cells"]:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if (nr, nc) in obj["cell_set"]:
                    continue
                if 0 <= nr < inp.shape[0] and 0 <= nc < inp.shape[1] and int(inp[nr, nc]) == bg and int(out[nr, nc]) == bg:
                    blocked = True
                    break
            if blocked:
                break
        if blocked:
            break
    return touches_edge or blocked


def _tie_invariant(inp: np.ndarray) -> bool:
    bg = _bg_color(inp)
    objs = _components(inp, bg)
    counts = Counter(int(obj["area"]) for obj in objs)
    if all(v == 1 for v in counts.values()):
        return True
    forward = _apply_diffusion(inp, _ordered_objects(inp, bg, reverse_ties=False), bg)
    backward = _apply_diffusion(inp, _ordered_objects(inp, bg, reverse_ties=True), bg)
    return np.array_equal(forward, backward)


def _valid_input(inp: np.ndarray) -> bool:
    if not isinstance(inp, np.ndarray) or inp.ndim != 2:
        return False
    if max(inp.shape) > 30 or _bg_color(inp) != BG:
        return False

    objects = _components(inp, BG)
    if not (3 <= len(objects) <= 4):
        return False
    if any(not (1 <= obj["area"] <= 4) for obj in objects):
        return False

    out = main(inp)
    if not np.any(out != inp):
        return False
    if not _tie_invariant(inp):
        return False
    return _uses_every_rule_branch(inp, out)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(12, 22)
    w = rng.randint(12, 22)
    grid = np.full((h, w), BG, dtype=int)
    guard = np.zeros((h, w), dtype=bool)

    scenario = rng.choice(["blocked", "edge", "mixed", "sparse"])
    object_count = 4 if scenario == "mixed" else rng.randint(3, 4)
    colors = rng.sample(list(COLORS), object_count)
    color_idx = 0
    remaining_areas = [1, 2, 3, 4]
    used_areas: list[int] = []

    def pick_areas(exclude: list[int]) -> list[int]:
        return [area for area in [1, 2, 3, 4] if area not in exclude] or [1, 2, 3, 4]

    if scenario in ("blocked", "mixed"):
        pair_areas = _place_blocking_pair(grid, guard, colors[color_idx : color_idx + 2], rng)
        if pair_areas is None:
            return None
        used_areas.extend(pair_areas)
        color_idx += 2

    if scenario in ("edge", "mixed"):
        area = _place_edge_object(grid, guard, colors[color_idx], rng, pick_areas(used_areas))
        if area is None:
            return None
        used_areas.append(area)
        color_idx += 1

    while color_idx < object_count:
        area = _place_random_object(grid, guard, colors[color_idx], rng, pick_areas(used_areas))
        if area is None:
            return None
        used_areas.append(area)
        color_idx += 1

    if scenario == "sparse":
        # Add one more empty margin ring around a random object-free border when possible.
        if h <= 14 or w <= 14:
            return None

    return grid if _valid_input(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid diffusion puzzle")
