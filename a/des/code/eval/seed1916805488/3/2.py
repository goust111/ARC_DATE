from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _component_points(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _is_connected(mask: np.ndarray) -> bool:
    count = int(mask.sum())
    if count == 0:
        return False
    return len(_component_points(mask)) == 1


def _bbox_from_points(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in points]
    cs = [c for _, c in points]
    return min(rs), min(cs), max(rs), max(cs)


def _extract_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    objects: list[dict] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != 0):
        comps = _component_points(g == color)
        for pts in comps:
            r0, c0, r1, c1 = _bbox_from_points(pts)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for r, c in pts:
                mask[r - r0, c - c0] = True
            objects.append(
                {
                    "color": int(color),
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                }
            )
    return objects


def _placed_mask(mask: np.ndarray, H: int, W: int, r0: int, c0: int) -> np.ndarray:
    out = np.zeros((H, W), dtype=bool)
    h, w = mask.shape
    out[r0 : r0 + h, c0 : c0 + w] = mask
    return out


def _complement_fit(mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[int, int] | None:
    ha, wa = mask_a.shape
    hb, wb = mask_b.shape
    max_H = ha + hb
    max_W = wa + wb
    best: tuple[int, int] | None = None
    for H in range(max(ha, hb), max_H + 1):
        for W in range(max(wa, wb), max_W + 1):
            for ra in range(H - ha + 1):
                for ca in range(W - wa + 1):
                    placed_a = _placed_mask(mask_a, H, W, ra, ca)
                    for rb in range(H - hb + 1):
                        for cb in range(W - wb + 1):
                            placed_b = _placed_mask(mask_b, H, W, rb, cb)
                            if np.any(placed_a & placed_b):
                                continue
                            if np.all(placed_a | placed_b):
                                cand = (H, W)
                                if best is None or H * W < best[0] * best[1] or (
                                    H * W == best[0] * best[1] and cand < best
                                ):
                                    best = cand
            if best is not None and H * W > best[0] * best[1]:
                break
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-bbox fragments by geometric complementarity, merge each pair into a
    two-color solid rectangle, then pack the rectangles left-to-right in
    ascending area order with one black separator column.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    objects = _extract_objects(g)
    assert objects and len(objects) % 2 == 0

    pairs: list[dict] = []
    remaining = set(range(len(objects)))
    while remaining:
        i = min(remaining)
        best_j = None
        best_dims = None
        for j in sorted(remaining):
            if j == i:
                continue
            dims = _complement_fit(objects[i]["mask"], objects[j]["mask"])
            if dims is None:
                continue
            if best_dims is None or dims[0] * dims[1] < best_dims[0] * best_dims[1] or (
                dims[0] * dims[1] == best_dims[0] * best_dims[1] and dims < best_dims
            ):
                best_j = j
                best_dims = dims
        assert best_j is not None and best_dims is not None
        pair_anchor = min(objects[i]["bbox"][:2], objects[best_j]["bbox"][:2])
        pairs.append(
            {
                "dims": best_dims,
                "a": objects[i],
                "b": objects[best_j],
                "area": best_dims[0] * best_dims[1],
                "anchor": pair_anchor,
            }
        )
        remaining.remove(i)
        remaining.remove(best_j)

    pairs.sort(key=lambda item: (item["area"], item["dims"][0], item["dims"][1], item["anchor"]))

    out_h = max(item["dims"][0] for item in pairs)
    out_w = sum(item["dims"][1] for item in pairs) + len(pairs) - 1
    out = np.zeros((out_h, out_w), dtype=int)

    c = 0
    for item in pairs:
        h, w = item["dims"]
        a_mask = item["a"]["mask"]
        b_mask = item["b"]["mask"]
        placed = False
        for ra in range(h - a_mask.shape[0] + 1):
            for ca in range(w - a_mask.shape[1] + 1):
                pa = _placed_mask(a_mask, h, w, ra, ca)
                for rb in range(h - b_mask.shape[0] + 1):
                    for cb in range(w - b_mask.shape[1] + 1):
                        pb = _placed_mask(b_mask, h, w, rb, cb)
                        if np.any(pa & pb):
                            continue
                        if np.all(pa | pb):
                            out[:h, c : c + w][pa] = int(item["a"]["color"])
                            out[:h, c : c + w][pb] = int(item["b"]["color"])
                            placed = True
                            break
                    if placed:
                        break
                if placed:
                    break
            if placed:
                break
        assert placed
        c += w + 1
    return out


# --------------- generator ---------------

GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
PAIR_COUNT_RANGE = (2, 4)
BBOX_H_RANGE = (2, 6)
BBOX_W_RANGE = (2, 7)
PLACEMENT_PAD = 1
GEN_RETRIES = 400
PARTITION_RETRIES = 200
PLACEMENT_RETRIES = 200


def _randint(bounds: tuple[int, int]) -> int:
    return random.randint(bounds[0], bounds[1])


def _spans_bbox(mask: np.ndarray) -> bool:
    return bool(
        np.any(mask[0, :])
        and np.any(mask[-1, :])
        and np.any(mask[:, 0])
        and np.any(mask[:, -1])
    )


def _random_manhattan_path(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = a
    tr, tc = b
    steps: list[tuple[int, int]] = []
    while r != tr or c != tc:
        options: list[tuple[int, int]] = []
        if r < tr:
            options.append((r + 1, c))
        elif r > tr:
            options.append((r - 1, c))
        if c < tc:
            options.append((r, c + 1))
        elif c > tc:
            options.append((r, c - 1))
        r, c = random.choice(options)
        steps.append((r, c))
    return steps


def _random_partition(h: int, w: int) -> np.ndarray | None:
    area = h * w
    cells = [(r, c) for r in range(h) for c in range(w)]
    for _ in range(PARTITION_RETRIES):
        target = random.randint(2, area - 2)
        seed = random.choice(cells)
        chosen = {seed}
        frontier = set()
        for cell in tuple(chosen):
            for nxt in _neighbors4(*cell):
                if 0 <= nxt[0] < h and 0 <= nxt[1] < w and nxt not in chosen:
                    frontier.add(nxt)
        while len(chosen) < target and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            chosen.add(cell)
            for nxt in _neighbors4(*cell):
                if 0 <= nxt[0] < h and 0 <= nxt[1] < w and nxt not in chosen:
                    frontier.add(nxt)

        mask = np.zeros((h, w), dtype=bool)
        for r, c in chosen:
            mask[r, c] = True
        other = ~mask
        if not _is_connected(other):
            continue
        return mask
    return None


def _place_patch(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    for _ in range(PLACEMENT_RETRIES):
        r0 = random.randint(0, h - ph)
        c0 = random.randint(0, w - pw)
        rr0 = max(0, r0 - PLACEMENT_PAD)
        cc0 = max(0, c0 - PLACEMENT_PAD)
        rr1 = min(h, r0 + ph + PLACEMENT_PAD)
        cc1 = min(w, c0 + pw + PLACEMENT_PAD)
        if occupied[rr0:rr1, cc0:cc1].any():
            continue
        mask = patch != 0
        if not mask.any():
            continue
        grid[r0 : r0 + ph, c0 : c0 + pw][mask] = patch[mask]
        occupied[r0 : r0 + ph, c0 : c0 + pw] |= mask
        return True
    return False


def _build_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _sample_dims(n_pairs: int) -> list[tuple[int, int]] | None:
    candidates = [
        (h, w)
        for h in range(BBOX_H_RANGE[0], BBOX_H_RANGE[1] + 1)
        for w in range(BBOX_W_RANGE[0], BBOX_W_RANGE[1] + 1)
        if h * w >= 4
    ]
    random.shuffle(candidates)
    dims: list[tuple[int, int]] = []
    used_areas: set[int] = set()
    for h, w in candidates:
        # Prefer unique areas so output ordering is visually unambiguous.
        if h * w in used_areas and random.random() < 0.8:
            continue
        dims.append((h, w))
        used_areas.add(h * w)
        if len(dims) == n_pairs:
            return dims
    return None


def _generate_one() -> np.ndarray | None:
    n_pairs = _randint(PAIR_COUNT_RANGE)
    dims = _sample_dims(n_pairs)
    if dims is None:
        return None

    needed_colors = 2 * n_pairs
    colors = random.sample(list(range(1, 10)), needed_colors)
    H = _randint(GRID_H_RANGE)
    W = _randint(GRID_W_RANGE)
    grid = np.zeros((H, W), dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    pieces: list[np.ndarray] = []
    for i, (h, w) in enumerate(dims):
        mask = _random_partition(h, w)
        if mask is None:
            return None
        pieces.append(_build_patch(mask, colors[2 * i]))
        pieces.append(_build_patch(~mask, colors[2 * i + 1]))

    random.shuffle(pieces)
    for patch in pieces:
        if not _place_patch(grid, occupied, patch):
            return None

    try:
        out = main(grid)
    except Exception:
        return None

    if out.size == 0 or np.count_nonzero(grid) == 0:
        return None
    if len(_extract_objects(grid)) != 2 * n_pairs:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_one()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
