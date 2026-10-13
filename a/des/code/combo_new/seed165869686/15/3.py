from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 300
PLACE_TRIES = 120
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            parts.append(comp)
    return parts


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _hole_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    holes: list[list[tuple[int, int]]] = []
    for comp in _components(~mask):
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in comp):
            continue
        holes.append(comp)
    return holes


def _hole_count(mask: np.ndarray) -> int:
    return len(_hole_components(mask))


def _analyze_objects(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    objects = []
    hole_pixels: set[tuple[int, int]] = set()

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BG or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))

            r0, c0, r1, c1 = _bbox(coords)
            local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                local[rr - r0, cc - c0] = True

            holes_local = _hole_components(local)
            obj_holes: set[tuple[int, int]] = set()
            for hole in holes_local:
                for rr, cc in hole:
                    obj_holes.add((int(r0 + rr), int(c0 + cc)))

            hole_pixels |= obj_holes
            objects.append(
                {
                    "color": color,
                    "coords": tuple(coords),
                    "hole_count": len(holes_local),
                    "holes": obj_holes,
                }
            )

    return objects, hole_pixels


def _growth_claims(grid: np.ndarray, objects, hole_pixels: set[tuple[int, int]]):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    claims: dict[tuple[int, int], list[int]] = {}

    for obj_idx, obj in enumerate(objects):
        steps = int(obj["hole_count"])
        if steps <= 0:
            continue
        dist: dict[tuple[int, int], int] = {}
        queue: deque[tuple[int, int]] = deque()

        for r, c in obj["coords"]:
            for nr, nc in _neighbors4(r, c):
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                pos = (int(nr), int(nc))
                if int(g[nr, nc]) != BG or pos in hole_pixels or pos in dist:
                    continue
                dist[pos] = 1
                queue.append(pos)

        while queue:
            pos = queue.popleft()
            d = dist[pos]
            claims.setdefault(pos, []).append(int(obj_idx))
            if d >= steps:
                continue
            r, c = pos
            for nr, nc in _neighbors4(r, c):
                nxt = (int(nr), int(nc))
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if int(g[nr, nc]) != BG or nxt in hole_pixels or nxt in dist:
                    continue
                dist[nxt] = d + 1
                queue.append(nxt)

    return claims


def _solve(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    objects, hole_pixels = _analyze_objects(g)
    claims = _growth_claims(g, objects, hole_pixels)

    out = g.copy()
    conflict_count = 0
    unique_count = 0
    for pos, claimers in claims.items():
        if len(claimers) == 1:
            obj = objects[claimers[0]]
            out[pos] = int(obj["color"])
            unique_count += 1
        else:
            conflict_count += 1

    return out, {"objects": objects, "claims": claims, "conflict_count": conflict_count, "unique_count": unique_count}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each monochrome object outward by as many layers as the number of black holes inside it.

    Input structure:
    - The background is black.
    - The grid contains several separated 4-connected monochrome objects of non-black colors.
    - Some objects contain one or more enclosed black holes fully surrounded by the object's color, while others have no holes.

    Rule:
    - For each colored object, count the number N of enclosed black hole components inside that object.
    - Expand the object into the external black background by N four-neighbor layers of the same color.
    - Keep internal holes black, and do not let growth pass through other colored objects.
    - If multiple objects try to paint the same black pixel, leave that pixel black.

    Key insight:
    - The amount of outward growth is controlled by topology rather than size: an object's hole count determines how many border rings it gains.
    """
    out, _meta = _solve(np.asarray(grid, dtype=int))
    return out


def _random_orient(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    out = np.rot90(mask, rng.randrange(4))
    if rng.random() < 0.5:
        out = np.fliplr(out)
    if rng.random() < 0.5:
        out = np.flipud(out)
    return np.asarray(out, dtype=bool).copy()


def _add_tabs(mask: np.ndarray, rng: random.Random, max_tabs: int) -> np.ndarray:
    pad = 4
    canvas = np.zeros((mask.shape[0] + 2 * pad, mask.shape[1] + 2 * pad), dtype=bool)
    canvas[pad : pad + mask.shape[0], pad : pad + mask.shape[1]] = mask

    for _ in range(rng.randint(0, max_tabs)):
        pts = np.argwhere(canvas)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        side = rng.choice(["top", "bottom", "left", "right"])
        thickness = 1 if rng.random() < 0.75 else 2

        if side in ("top", "bottom"):
            width = rng.randint(1, max(1, (c1 - c0 + 1) // 2))
            start = rng.randint(int(c0), int(c1 - width + 1))
            if side == "top" and r0 - thickness >= 0:
                canvas[r0 - thickness : r0, start : start + width] = True
            elif side == "bottom" and r1 + thickness < canvas.shape[0]:
                canvas[r1 + 1 : r1 + 1 + thickness, start : start + width] = True
        else:
            height = rng.randint(1, max(1, (r1 - r0 + 1) // 2))
            start = rng.randint(int(r0), int(r1 - height + 1))
            if side == "left" and c0 - thickness >= 0:
                canvas[start : start + height, c0 - thickness : c0] = True
            elif side == "right" and c1 + thickness < canvas.shape[1]:
                canvas[start : start + height, c1 + 1 : c1 + 1 + thickness] = True

    return _crop_mask(canvas)


def _make_solid_object(rng: random.Random) -> np.ndarray:
    for _ in range(80):
        canvas = np.zeros((16, 16), dtype=bool)
        base_h = rng.randint(2, 4)
        base_w = rng.randint(2, 4)
        r0 = rng.randint(5, 8)
        c0 = rng.randint(5, 8)
        canvas[r0 : r0 + base_h, c0 : c0 + base_w] = True

        for _ in range(rng.randint(1, 4)):
            pts = np.argwhere(canvas)
            br0, bc0 = pts.min(axis=0)
            br1, bc1 = pts.max(axis=0)
            side = rng.choice(["top", "bottom", "left", "right"])
            if side in ("top", "bottom"):
                span = rng.randint(1, max(1, int(bc1 - bc0 + 1)))
                start = rng.randint(int(bc0), int(bc1 - span + 1))
                thick = rng.randint(1, 3)
                if side == "top" and br0 - thick >= 0:
                    canvas[br0 - thick : br0, start : start + span] = True
                elif side == "bottom" and br1 + thick < canvas.shape[0]:
                    canvas[br1 + 1 : br1 + 1 + thick, start : start + span] = True
            else:
                span = rng.randint(1, max(1, int(br1 - br0 + 1)))
                start = rng.randint(int(br0), int(br1 - span + 1))
                thick = rng.randint(1, 3)
                if side == "left" and bc0 - thick >= 0:
                    canvas[start : start + span, bc0 - thick : bc0] = True
                elif side == "right" and bc1 + thick < canvas.shape[1]:
                    canvas[start : start + span, bc1 + 1 : bc1 + 1 + thick] = True

        mask = _random_orient(_crop_mask(canvas), rng)
        if mask.size == 0:
            continue
        if _hole_count(mask) != 0:
            continue
        if mask.sum() < 6 or mask.sum() > 28:
            continue
        if max(mask.shape) > 11:
            continue
        return mask

    raise RuntimeError("failed to make solid object")


def _make_holey_object(rng: random.Random, holes: int) -> np.ndarray:
    assert holes >= 1
    for _ in range(120):
        orient = rng.choice(["vertical", "horizontal"])

        if orient == "vertical":
            chamber_widths = [rng.randint(1, 3) for _ in range(holes)]
            inner_h = rng.randint(2, 5)
            inner_w = sum(chamber_widths) + (holes - 1)
            core = np.zeros((inner_h + 2, inner_w + 2), dtype=bool)
            core[0, :] = True
            core[-1, :] = True
            core[:, 0] = True
            core[:, -1] = True
            col = 1
            for width in chamber_widths[:-1]:
                col += width
                core[1:-1, col] = True
                col += 1
        else:
            chamber_heights = [rng.randint(1, 3) for _ in range(holes)]
            inner_w = rng.randint(2, 5)
            inner_h = sum(chamber_heights) + (holes - 1)
            core = np.zeros((inner_h + 2, inner_w + 2), dtype=bool)
            core[0, :] = True
            core[-1, :] = True
            core[:, 0] = True
            core[:, -1] = True
            row = 1
            for height in chamber_heights[:-1]:
                row += height
                core[row, 1:-1] = True
                row += 1

        mask = _add_tabs(core, rng, max_tabs=2)
        mask = _random_orient(mask, rng)
        if mask.size == 0:
            continue
        if _hole_count(mask) != holes:
            continue
        if max(mask.shape) > 13:
            continue
        if mask.sum() < 8:
            continue
        return mask

    raise RuntimeError(f"failed to make {holes}-hole object")


def _make_object_mask(rng: random.Random, holes: int) -> np.ndarray:
    if holes == 0:
        return _make_solid_object(rng)
    return _make_holey_object(rng, holes)


def _stamp_blocked(blocked: np.ndarray, mask: np.ndarray, top: int, left: int):
    h, w = blocked.shape
    for rr, cc in np.argwhere(mask):
        gr = int(top + rr)
        gc = int(left + cc)
        r0 = max(0, gr - 1)
        r1 = min(h, gr + 2)
        c0 = max(0, gc - 1)
        c1 = min(w, gc + 2)
        blocked[r0:r1, c0:c1] = True


def _can_place(blocked: np.ndarray, mask: np.ndarray, top: int, left: int) -> bool:
    coords = np.argwhere(mask)
    rows = top + coords[:, 0]
    cols = left + coords[:, 1]
    return not bool(np.any(blocked[rows, cols]))


def _paint_object(grid: np.ndarray, blocked: np.ndarray, mask: np.ndarray, color: int, top: int, left: int):
    coords = np.argwhere(mask)
    rows = top + coords[:, 0]
    cols = left + coords[:, 1]
    grid[rows, cols] = color
    _stamp_blocked(blocked, mask, top, left)


def _sample_hole_counts(rng: random.Random, want_conflict: bool) -> list[int]:
    if want_conflict:
        count = rng.choices([3, 4, 5], weights=[3, 3, 2])[0]
        pair = rng.choice([(1, 2), (2, 1), (2, 2), (1, 3), (3, 1)])
        rest = [0]
        while len(rest) < count - 2:
            rest.append(rng.choices([0, 1, 2, 3], weights=[3, 3, 2, 1])[0])
        return [pair[0], pair[1], *rest[: count - 2]]

    count = rng.choices([2, 3, 4, 5], weights=[2, 3, 3, 2])[0]
    counts = [0, rng.choices([1, 2, 3], weights=[4, 3, 1])[0]]
    while len(counts) < count:
        counts.append(rng.choices([0, 1, 2, 3], weights=[4, 3, 2, 1])[0])
    rng.shuffle(counts)
    return counts


def _place_remaining(grid: np.ndarray, blocked: np.ndarray, masks: list[np.ndarray], colors: list[int], indices: list[int], rng: random.Random) -> bool:
    h, w = grid.shape
    for idx in sorted(indices, key=lambda i: int(masks[i].sum()), reverse=True):
        mh, mw = masks[idx].shape
        if mh + 2 > h or mw + 2 > w:
            return False
        placed = False
        for _ in range(PLACE_TRIES):
            top = rng.randint(1, h - mh - 1)
            left = rng.randint(1, w - mw - 1)
            if not _can_place(blocked, masks[idx], top, left):
                continue
            _paint_object(grid, blocked, masks[idx], colors[idx], top, left)
            placed = True
            break
        if not placed:
            return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    want_conflict = rng.random() < 0.5
    hole_counts = _sample_hole_counts(rng, want_conflict)
    masks = [_make_object_mask(rng, holes) for holes in hole_counts]
    colors = rng.sample(COLOR_POOL, len(masks))

    min_h = max(18, max(mask.shape[0] for mask in masks) + 4)
    min_w = max(18, max(mask.shape[1] for mask in masks) + 4)
    if want_conflict:
        min_h = max(min_h, max(masks[0].shape[0], masks[1].shape[0]) + 4)
        min_w = max(min_w, masks[0].shape[1] + masks[1].shape[1] + 8)
    if min_h > 30 or min_w > 30:
        return None

    h = rng.randint(min_h, 30)
    w = rng.randint(min_w, 30)
    grid = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    if want_conflict:
        left_mask, right_mask = masks[0], masks[1]
        gap = 2
        pair_h = max(left_mask.shape[0], right_mask.shape[0])
        top = rng.randint(1, h - pair_h - 1)
        max_left = w - (left_mask.shape[1] + right_mask.shape[1] + gap) - 1
        if max_left < 1:
            return None
        left = rng.randint(1, max_left)
        right = left + left_mask.shape[1] + gap
        _paint_object(grid, blocked, left_mask, colors[0], top, left)
        _paint_object(grid, blocked, right_mask, colors[1], top, right)
        remaining = list(range(2, len(masks)))
    else:
        remaining = list(range(len(masks)))

    if not _place_remaining(grid, blocked, masks, colors, remaining, rng):
        return None

    out, meta = _solve(grid)
    hole_multiset = sorted(int(obj["hole_count"]) for obj in meta["objects"])
    if hole_multiset != sorted(hole_counts):
        return None
    if meta["unique_count"] == 0:
        return None
    if want_conflict != (meta["conflict_count"] > 0):
        return None
    if not np.any(out != grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
