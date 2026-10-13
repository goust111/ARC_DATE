from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GENERATION_ATTEMPTS = 400
PAIR_COUNT_CHOICES = (2, 3, 4)
COLOR_POOL = tuple(range(1, 10))
RECT_HEIGHT_RANGE = (2, 6)  # inclusive low, exclusive high
RECT_WIDTH_RANGE = (3, 7)
INPUT_SIDE_MIN = 18
INPUT_SIDE_MAX = 30
PLACEMENT_PADDING = 1


def _normalize_coords(coords: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    points = [(int(r), int(c)) for r, c in coords]
    r0 = min(r for r, _ in points)
    c0 = min(c for _, c in points)
    return tuple(sorted((r - r0, c - c0) for r, c in points))


def _bbox_from_coords(coords: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    points = [(int(r), int(c)) for r, c in coords]
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1].copy()


def _mask_connected(mask: np.ndarray) -> bool:
    cells = np.argwhere(mask)
    if cells.size == 0:
        return False
    h, w = mask.shape
    seen = set()
    stack = [tuple(map(int, cells[0]))]
    seen.add(stack[0])
    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _find_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    visited = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(g[r, c]) == BG:
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(g[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            r0, c0, r1, c1 = _bbox_from_coords(coords)
            rel = tuple(sorted((rr - r0, cc - c0) for rr, cc in coords))
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in rel:
                mask[rr, cc] = True
            objects.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "rel": rel,
                    "mask": mask,
                    "h": int(mask.shape[0]),
                    "w": int(mask.shape[1]),
                    "area": int(len(rel)),
                }
            )

    return objects


def _factor_pairs(area: int) -> list[tuple[int, int]]:
    dims = set()
    for h in range(1, int(area**0.5) + 1):
        if area % h == 0:
            w = area // h
            dims.add((h, w))
            dims.add((w, h))
    return sorted(dims)


def _assembly_options(obj_a: dict, obj_b: dict) -> list[np.ndarray]:
    total_area = int(obj_a["area"] + obj_b["area"])
    if total_area <= 1:
        return []

    rel_a = obj_a["rel"]
    rel_b = obj_b["rel"]
    out: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}

    for rect_h, rect_w in _factor_pairs(total_area):
        if rect_h < max(obj_a["h"], obj_b["h"]) or rect_w < max(obj_a["w"], obj_b["w"]):
            continue

        for ar in range(rect_h - obj_a["h"] + 1):
            for ac in range(rect_w - obj_a["w"] + 1):
                filled = np.zeros((rect_h, rect_w), dtype=bool)
                for rr, cc in rel_a:
                    filled[ar + rr, ac + cc] = True
                if int(filled.sum()) != obj_a["area"]:
                    continue

                comp = np.argwhere(~filled)
                if comp.shape[0] != obj_b["area"]:
                    continue
                r0 = int(comp[:, 0].min())
                c0 = int(comp[:, 1].min())
                r1 = int(comp[:, 0].max())
                c1 = int(comp[:, 1].max())
                if r1 - r0 + 1 != obj_b["h"] or c1 - c0 + 1 != obj_b["w"]:
                    continue

                norm_comp = tuple(sorted((int(r - r0), int(c - c0)) for r, c in comp))
                if norm_comp != rel_b:
                    continue

                patch = np.full((rect_h, rect_w), BG, dtype=int)
                patch[filled] = int(obj_a["color"])
                patch[~filled] = int(obj_b["color"])
                out[(tuple(map(int, patch.shape)), patch.tobytes())] = patch

    return [out[key] for key in sorted(out)]


def _pack_rectangles(choice: list[tuple[int, int, np.ndarray]], objects: list[dict]) -> np.ndarray:
    rects = []
    for i, j, patch in choice:
        pos_a = tuple(map(int, objects[i]["bbox"][:2]))
        pos_b = tuple(map(int, objects[j]["bbox"][:2]))
        area = int(np.count_nonzero(patch))
        rects.append((area, min(pos_a, pos_b), max(pos_a, pos_b), patch))

    rects.sort(key=lambda item: (item[0], item[1], item[2]))
    out_h = max(int(patch.shape[0]) for _, _, _, patch in rects)
    out_w = sum(int(patch.shape[1]) for _, _, _, patch in rects) + len(rects) - 1
    out = np.full((out_h, out_w), BG, dtype=int)

    col = 0
    for _, _, _, patch in rects:
        h, w = map(int, patch.shape)
        out[:h, col : col + w] = patch
        col += w + 1
    return out


def _solve_outputs(objects: list[dict], limit: int = 2) -> list[np.ndarray]:
    n = len(objects)
    edges: dict[tuple[int, int], list[np.ndarray]] = {}
    neighbors = {i: set() for i in range(n)}

    for i in range(n):
        for j in range(i + 1, n):
            if int(objects[i]["color"]) == int(objects[j]["color"]):
                continue
            options = _assembly_options(objects[i], objects[j])
            if not options:
                continue
            edges[(i, j)] = options
            neighbors[i].add(j)
            neighbors[j].add(i)

    if any(not neighbors[i] for i in range(n)):
        return []

    outputs: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}

    def backtrack(remaining: set[int], choice: list[tuple[int, int, np.ndarray]]) -> None:
        if len(outputs) >= limit:
            return
        if not remaining:
            packed = _pack_rectangles(choice, objects)
            outputs[(tuple(map(int, packed.shape)), packed.tobytes())] = packed
            return

        i = min(remaining, key=lambda idx: len(neighbors[idx] & remaining))
        partners = sorted(neighbors[i] & remaining)
        if not partners:
            return
        for j in partners:
            key = (i, j) if i < j else (j, i)
            for patch in edges[key]:
                choice.append((key[0], key[1], patch))
                backtrack(remaining - {i, j}, choice)
                choice.pop()
                if len(outputs) >= limit:
                    return

    backtrack(set(range(n)), [])
    return [outputs[key] for key in sorted(outputs)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair scattered monochrome fragments by rectangle complementarity, then pack the
    assembled bicolor rectangles from smallest area to largest.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    objects = _find_components(g)
    assert objects and len(objects) % 2 == 0

    outputs = _solve_outputs(objects, limit=2)
    assert outputs, "expected at least one valid complementary pairing"
    assert len(outputs) == 1, "pairing must determine a unique packed output"
    return outputs[0]


def _paint_mask(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = int(color)
    return patch


def _random_partition(rect_h: int, rect_w: int) -> tuple[np.ndarray, np.ndarray] | None:
    drops = [0] * (rect_h - 1)
    for _ in range(rect_w - 1):
        drops[random.randrange(rect_h - 1)] += 1

    boundary = [rect_w - 1]
    cur = rect_w - 1
    for drop in drops:
        cur -= drop
        boundary.append(cur)

    mask_a = np.zeros((rect_h, rect_w), dtype=bool)
    for r, last_c in enumerate(boundary):
        mask_a[r, : last_c + 1] = True

    if random.random() < 0.5:
        mask_a = np.flipud(mask_a)
    if random.random() < 0.5:
        mask_a = np.fliplr(mask_a)
    if random.random() < 0.5:
        mask_a = mask_a.T.copy()

    mask_b = ~mask_a
    if int(mask_a.sum()) < 2 or int(mask_b.sum()) < 2:
        return None
    if not _mask_connected(mask_a) or not _mask_connected(mask_b):
        return None
    return mask_a, mask_b


def _make_pair(color_a: int, color_b: int, used_areas: set[int]) -> dict | None:
    for _ in range(200):
        rect_h = random.randrange(*RECT_HEIGHT_RANGE)
        rect_w = random.randrange(*RECT_WIDTH_RANGE)
        parts = _random_partition(rect_h, rect_w)
        if parts is None:
            continue
        mask_a, mask_b = parts
        if random.random() < 0.5:
            mask_a, mask_b = mask_b, mask_a

        rect_h2, rect_w2 = map(int, mask_a.shape)
        area = rect_h2 * rect_w2
        if area in used_areas:
            continue

        frag_a = _crop_mask(mask_a)
        frag_b = _crop_mask(mask_b)
        out_patch = np.full((rect_h2, rect_w2), BG, dtype=int)
        out_patch[mask_a] = int(color_a)
        out_patch[mask_b] = int(color_b)
        return {
            "area": area,
            "output": out_patch,
            "fragments": [
                _paint_mask(frag_a, color_a),
                _paint_mask(frag_b, color_b),
            ],
        }
    return None


def _paste_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    mask = patch != BG
    h, w = patch.shape
    grid[r0 : r0 + h, c0 : c0 + w][mask] = patch[mask]


def _choose_position(blocked: np.ndarray, patch_shape: tuple[int, int], border: int) -> tuple[int, int] | None:
    h, w = map(int, patch_shape)
    H, W = blocked.shape
    positions = []
    r_start = border
    r_stop = H - h - border
    c_start = border
    c_stop = W - w - border
    if r_stop < r_start or c_stop < c_start:
        return None

    for r0 in range(r_start, r_stop + 1):
        for c0 in range(c_start, c_stop + 1):
            rr0 = max(0, r0 - PLACEMENT_PADDING)
            rr1 = min(H, r0 + h + PLACEMENT_PADDING)
            cc0 = max(0, c0 - PLACEMENT_PADDING)
            cc1 = min(W, c0 + w + PLACEMENT_PADDING)
            if not blocked[rr0:rr1, cc0:cc1].any():
                positions.append((r0, c0))
    if not positions:
        return None
    return random.choice(positions)


def _scatter_fragments(fragments: list[np.ndarray]) -> np.ndarray | None:
    max_h = max(int(patch.shape[0]) for patch in fragments)
    max_w = max(int(patch.shape[1]) for patch in fragments)
    bbox_area = sum(int(patch.shape[0] * patch.shape[1]) for patch in fragments)
    min_side = max(INPUT_SIDE_MIN, max_h + 2, max_w + 2, int(np.ceil((bbox_area * 2.6) ** 0.5)))
    if min_side > INPUT_SIDE_MAX:
        return None

    H = random.randint(min_side, INPUT_SIDE_MAX)
    W = random.randint(min_side, INPUT_SIDE_MAX)
    grid = np.full((H, W), BG, dtype=int)
    blocked = np.zeros((H, W), dtype=bool)

    order = list(fragments)
    random.shuffle(order)
    order.sort(key=lambda patch: (int(patch.shape[0] * patch.shape[1]), int(np.count_nonzero(patch))), reverse=True)

    border = 1 if H >= max_h + 2 and W >= max_w + 2 and random.random() < 0.8 else 0
    for patch in order:
        pos = _choose_position(blocked, patch.shape, border)
        if pos is None and border > 0:
            pos = _choose_position(blocked, patch.shape, 0)
        if pos is None:
            return None
        r0, c0 = pos
        _paste_patch(grid, patch, r0, c0)
        h, w = patch.shape
        blocked[r0 : r0 + h, c0 : c0 + w] = True

    return grid


def _build_expected(pairs: list[dict]) -> np.ndarray:
    ordered = sorted(pairs, key=lambda pair: int(pair["area"]))
    out_h = max(int(pair["output"].shape[0]) for pair in ordered)
    out_w = sum(int(pair["output"].shape[1]) for pair in ordered) + len(ordered) - 1
    out = np.full((out_h, out_w), BG, dtype=int)
    col = 0
    for pair in ordered:
        patch = pair["output"]
        h, w = patch.shape
        out[:h, col : col + w] = patch
        col += w + 1
    return out


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choice(PAIR_COUNT_CHOICES)
    colors = random.sample(COLOR_POOL, 2 * pair_count)

    used_areas: set[int] = set()
    pairs: list[dict] = []
    for i in range(pair_count):
        pair = _make_pair(colors[2 * i], colors[2 * i + 1], used_areas)
        if pair is None:
            return None
        used_areas.add(int(pair["area"]))
        pairs.append(pair)

    expected = _build_expected(pairs)
    if expected.shape[0] > INPUT_SIDE_MAX or expected.shape[1] > INPUT_SIDE_MAX:
        return None

    fragments = [patch for pair in pairs for patch in pair["fragments"]]
    grid = _scatter_fragments(fragments)
    if grid is None:
        return None

    found = _find_components(grid)
    if len(found) != 2 * pair_count:
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if np.array_equal(actual, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid complementary-rectangle sample")
