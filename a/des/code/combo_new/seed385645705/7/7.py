from __future__ import annotations

from collections import Counter, deque
from itertools import combinations
import random

import numpy as np


def _background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _neighbors(r: int, c: int, connectivity: int = 8):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            if connectivity == 4 and abs(dr) + abs(dc) != 1:
                continue
            yield r + dr, c + dc


def _components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q: deque[tuple[int, int]] = deque([(sr, sc)])
        seen[sr, sc] = True
        pts: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            pts.append((r, c))
            for nr, nc in _neighbors(r, c, connectivity):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct equal damaged templates and color each template's private missing cells from a bottom legend.

    Input structure:
    - The dominant color is the background.
    - Above the final row are several non-overlapping objects drawn with one shared foreground color.
    - The objects have the same bounding-box size and the same complete template, but each has background-colored missing cells.
    - The bottom row contains one distinct legend pixel per object, read from left to right.

    Rule:
    - Find the foreground objects, align their bounding boxes, and take the union of their foreground pixels as the complete template.
    - For each object, compare its aligned mask with the complete template to count all missing template pixels.
    - A private missing cell is absent from exactly that one object and present in every other object.
    - Sort objects by missing-pixel count from largest to smallest and assign the bottom legend colors in that order.
    - In each object bbox, restore every non-private missing template cell to the shared foreground color and fill private missing cells with the object's assigned legend color.

    Key insight: The missing template cells are measured only after all copies are overlaid; the bottom color strip labels objects by missing amount, while only one-object gaps receive label colors.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _background_color(g)

    legend = [(c, int(g[h - 1, c])) for c in range(w) if int(g[h - 1, c]) != bg]
    assert legend, "bottom legend must contain at least one non-background pixel"
    legend_colors = [color for _c, color in legend]
    n = len(legend_colors)

    body = g[: h - 1, :]
    non_bg = [int(v) for v in body.ravel() if int(v) != bg]
    assert non_bg, "body must contain foreground objects"
    fg = Counter(non_bg).most_common(1)[0][0]

    comps = _components(body == fg, connectivity=8)
    comps = [comp for comp in comps if int(comp.shape[0]) > 0]
    if len(comps) != n:
        comps = sorted(comps, key=lambda comp: (-int(comp.shape[0]), _bbox(comp)))[:n]
    assert len(comps) == n, "foreground object count must match legend length"

    objects: list[dict[str, object]] = []
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        objects.append(
            {
                "bbox": (r0, c0, r1, c1),
                "area": int(comp.shape[0]),
            }
        )
    objects.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))  # type: ignore[index]

    heights = {int(item["bbox"][2]) - int(item["bbox"][0]) + 1 for item in objects}  # type: ignore[index]
    widths = {int(item["bbox"][3]) - int(item["bbox"][1]) + 1 for item in objects}  # type: ignore[index]
    assert len(heights) == 1 and len(widths) == 1, "all object bboxes must have the same size"
    th = heights.pop()
    tw = widths.pop()

    presence = np.zeros((n, th, tw), dtype=bool)
    for i, item in enumerate(objects):
        r0, c0, r1, c1 = item["bbox"]  # type: ignore[assignment]
        presence[i] = g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == fg

    complete = presence.any(axis=0)
    missing_counts = [int((complete & (~presence[i])).sum()) for i in range(n)]

    private_missing: list[np.ndarray] = []
    for i in range(n):
        if n == 1:
            private_missing.append(np.zeros((th, tw), dtype=bool))
            continue
        other_idxs = [j for j in range(n) if j != i]
        present_in_others = presence[other_idxs].all(axis=0)
        private_missing.append(complete & (~presence[i]) & present_in_others)

    ranked = sorted(
        range(n),
        key=lambda i: (
            -missing_counts[i],
            int(objects[i]["bbox"][0]),  # type: ignore[index]
            int(objects[i]["bbox"][1]),  # type: ignore[index]
        ),
    )
    assigned = {obj_i: int(legend_colors[rank]) for rank, obj_i in enumerate(ranked)}

    out = g.copy()
    for i, item in enumerate(objects):
        r0, c0, r1, c1 = item["bbox"]  # type: ignore[assignment]
        rr0, cc0, rr1, cc1 = int(r0), int(c0), int(r1), int(c1)
        sub = out[rr0 : rr1 + 1, cc0 : cc1 + 1]
        private = private_missing[i]
        sub[complete & (~private)] = fg
        sub[private] = assigned[i]

    return out


_GEN_RETRIES = 400
_COLORS = tuple(range(10))


def _is_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    return len(_components(mask, connectivity=8)) == 1


def _uses_full_bbox(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    r0, c0, r1, c1 = _bbox(pts)
    return r0 == 0 and c0 == 0 and r1 == mask.shape[0] - 1 and c1 == mask.shape[1] - 1


def _make_template(n_objects: int) -> np.ndarray | None:
    if n_objects >= 5:
        h = random.randint(7, 8)
        w = random.randint(7, 8)
    elif n_objects == 4:
        h = random.randint(6, 8)
        w = random.randint(6, 8)
    else:
        h = random.randint(5, 8)
        w = random.randint(5, 8)

    mask = np.ones((h, w), dtype=bool)

    corner_candidates = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    random.shuffle(corner_candidates)
    for r, c in corner_candidates[: random.randint(0, 3)]:
        mask[r, c] = False

    edge_candidates: list[tuple[int, int]] = []
    if w >= 6:
        edge_candidates.extend([(0, random.randint(1, w - 2)), (h - 1, random.randint(1, w - 2))])
    if h >= 6:
        edge_candidates.extend([(random.randint(1, h - 2), 0), (random.randint(1, h - 2), w - 1)])
    random.shuffle(edge_candidates)
    for r, c in edge_candidates[: random.randint(0, min(2, len(edge_candidates)))]:
        test = mask.copy()
        test[r, c] = False
        if _is_connected(test) and _uses_full_bbox(test):
            mask = test

    transforms = random.randint(0, 7)
    if transforms & 1:
        mask = np.flipud(mask)
    if transforms & 2:
        mask = np.fliplr(mask)
    if transforms & 4:
        mask = np.rot90(mask)

    if not _is_connected(mask) or not _uses_full_bbox(mask):
        return None
    return np.array(mask, dtype=bool)


def _interior_template_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if bool(mask[r, c]):
                cells.append((r, c))
    return cells


def _choose_missing_sets(mask: np.ndarray, n_objects: int) -> list[set[tuple[int, int]]] | None:
    cells = _interior_template_cells(mask)
    if len(cells) < n_objects + 1:
        return None

    max_shared = min(4, max(1, len(cells) // 5))
    for _ in range(80):
        shared_count = random.randint(1, max_shared)
        available_for_private = len(cells) - shared_count
        upper = min(7, available_for_private)
        count_options = list(range(1, upper + 1))
        combos = [combo for combo in combinations(count_options, n_objects) if sum(combo) <= available_for_private]
        if not combos:
            continue
        private_counts = list(random.choice(combos))
        random.shuffle(private_counts)

        shuffled = cells[:]
        random.shuffle(shuffled)
        cursor = 0
        missing: list[set[tuple[int, int]]] = [set() for _ in range(n_objects)]

        for obj_i, count in enumerate(private_counts):
            for cell in shuffled[cursor : cursor + count]:
                missing[obj_i].add(cell)
            cursor += count

        shared_cells = shuffled[cursor : cursor + shared_count]
        for cell in shared_cells:
            subset_size = random.randint(2, n_objects - 1)
            for obj_i in random.sample(range(n_objects), subset_size):
                missing[obj_i].add(cell)

        totals = [len(s) for s in missing]
        if len(set(totals)) != n_objects:
            continue

        ok = True
        for obj_missing in missing:
            visible = mask.copy()
            for r, c in obj_missing:
                visible[r, c] = False
            if not _is_connected(visible) or not _uses_full_bbox(visible):
                ok = False
                break
        if ok:
            return missing
    return None


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _grid_size_for_objects(n_objects: int, th: int, tw: int) -> tuple[int, int] | None:
    layouts: list[tuple[int, int, int, int]] = []
    for cols in range(1, n_objects + 1):
        rows = (n_objects + cols - 1) // cols
        min_h = rows * th + max(0, rows - 1) + 3
        min_w = cols * tw + max(0, cols - 1) + 2
        if min_h <= 30 and min_w <= 30:
            layouts.append((rows, cols, min_h, min_w))
    if not layouts:
        return None
    _rows, _cols, min_h, min_w = random.choice(layouts)
    return random.randint(min_h, 30), random.randint(min_w, 30)


def _place_rectangles(grid_h: int, grid_w: int, th: int, tw: int, n_objects: int) -> list[tuple[int, int]] | None:
    rects: list[tuple[int, int, int, int]] = []
    locs: list[tuple[int, int]] = []
    max_r = grid_h - th - 2
    max_c = grid_w - tw
    if max_r < 0 or max_c < 0:
        return None
    for _ in range(n_objects):
        placed = False
        for _try in range(500):
            r = random.randint(0, max_r)
            c = random.randint(0, max_c)
            rect = (r, c, r + th - 1, c + tw - 1)
            if all(not _rects_overlap(rect, other, pad=1) for other in rects):
                rects.append(rect)
                locs.append((r, c))
                placed = True
                break
        if not placed:
            return None
    return locs


def _gen_attempt() -> np.ndarray | None:
    n_objects = random.randint(3, 5)
    template = _make_template(n_objects)
    if template is None:
        return None
    th, tw = template.shape

    missing_sets = _choose_missing_sets(template, n_objects)
    if missing_sets is None:
        return None

    size = _grid_size_for_objects(n_objects, th, tw)
    if size is None:
        return None
    grid_h, grid_w = size
    locs = _place_rectangles(grid_h, grid_w, th, tw, n_objects)
    if locs is None:
        return None

    colors = list(_COLORS)
    bg = random.choice(colors)
    available = [c for c in colors if c != bg]
    fg = random.choice(available)
    marker_pool = [c for c in available if c != fg]
    if len(marker_pool) < n_objects:
        return None
    marker_colors = random.sample(marker_pool, n_objects)

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    for obj_i, (r0, c0) in enumerate(locs):
        visible = template.copy()
        for rr, cc in missing_sets[obj_i]:
            visible[rr, cc] = False
        sub = grid[r0 : r0 + th, c0 : c0 + tw]
        sub[visible] = fg

    gap = random.choice([1, 2])
    needed_w = 1 + (n_objects - 1) * gap
    if needed_w > grid_w:
        return None
    start_c = random.randint(0, grid_w - needed_w)
    for i, color in enumerate(marker_colors):
        grid[grid_h - 1, start_c + i * gap] = int(color)

    vals, counts = np.unique(grid, return_counts=True)
    if int(vals[int(np.argmax(counts))]) != int(bg):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid seed385645705/7 input")
