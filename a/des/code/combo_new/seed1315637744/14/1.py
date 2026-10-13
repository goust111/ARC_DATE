import random
from collections import deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 400
SHAPE_RETRIES = 160
PLACEMENT_RETRIES = 120


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _boundary(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w) or not mask[nr, nc]:
                    out[r, c] = True
                    break
    return out


def _erode4(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w), dtype=bool)
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if mask[r, c] and all(mask[r + dr, c + dc] for dr, dc in DIR4):
                out[r, c] = True
    return out


def _nonzero_runs(arr: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for i, v in enumerate(arr):
        if int(v) != 0 and start is None:
            start = i
        elif int(v) == 0 and start is not None:
            runs.append((int(start), int(i)))
            start = None
    if start is not None:
        runs.append((int(start), int(len(arr))))
    return runs


def _legend_info(grid: np.ndarray) -> tuple[list[int], np.ndarray]:
    h, w = grid.shape
    candidates: list[tuple[int, str, int, int]] = []
    for start, end in _nonzero_runs(grid[h - 1, :]):
        candidates.append((end - start, "bottom", start, end))
    for start, end in _nonzero_runs(grid[:, w - 1]):
        candidates.append((end - start, "right", start, end))
    assert candidates, "expected a legend strip on the bottom row or right column"
    _, side, start, end = max(candidates, key=lambda x: (x[0], x[1] == "bottom"))

    mask = np.zeros_like(grid, dtype=bool)
    if side == "bottom":
        mask[h - 1, start:end] = True
        colors = [int(v) for v in grid[h - 1, start:end]]
    else:
        mask[start:end, w - 1] = True
        colors = [int(v) for v in grid[start:end, w - 1]]
    return colors, mask


def _classify_regions(walls: np.ndarray) -> tuple[np.ndarray, set[int], set[int]]:
    h, w = walls.shape
    padded = np.zeros((h + 2, w + 2), dtype=bool)
    padded[1:-1, 1:-1] = walls
    free = ~padded

    labels = -np.ones_like(padded, dtype=int)
    next_label = 0
    for r in range(h + 2):
        for c in range(w + 2):
            if not free[r, c] or labels[r, c] != -1:
                continue
            q = deque([(r, c)])
            labels[r, c] = next_label
            while q:
                cr, cc = q.popleft()
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h + 2 and 0 <= nc < w + 2 and free[nr, nc] and labels[nr, nc] == -1:
                        labels[nr, nc] = next_label
                        q.append((nr, nc))
            next_label += 1

    adj = [set() for _ in range(next_label)]
    for r in range(h + 2):
        for c in range(w + 2):
            if not padded[r, c]:
                continue
            nbrs = []
            seen = set()
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h + 2 and 0 <= nc < w + 2 and labels[nr, nc] >= 0:
                    lab = int(labels[nr, nc])
                    if lab not in seen:
                        seen.add(lab)
                        nbrs.append(lab)
            for i in range(len(nbrs)):
                for j in range(i + 1, len(nbrs)):
                    a, b = nbrs[i], nbrs[j]
                    adj[a].add(b)
                    adj[b].add(a)

    outside = int(labels[0, 0])
    dist = {outside: 0}
    q = deque([outside])
    while q:
        cur = q.popleft()
        for nxt in adj[cur]:
            if nxt not in dist:
                dist[nxt] = dist[cur] + 1
                q.append(nxt)

    fill_labels = {lab for lab, d in dist.items() if d % 2 == 1}
    hole_labels = {lab for lab, d in dist.items() if d > 0 and d % 2 == 0}
    return labels[1:-1, 1:-1], fill_labels, hole_labels


def main(grid: np.ndarray) -> np.ndarray:
    """
    Refill each closed boundary according to how many black holes it contains.

    Input structure:
    - Background is black (0).
    - The main objects are several separated closed boundary drawings in one non-background color; each object may contain zero or more enclosed black holes.
    - A palette strip lies either on the bottom row or on the rightmost column, with one visible color per index for hole counts 0, 1, 2, ...

    Rule:
    - Detect and remove the edge palette strip.
    - For every remaining closed boundary component, analyze the black regions separated by that boundary: odd-depth enclosed regions are fillable interior, even-depth enclosed regions are holes.
    - Count the holes, use that count as a zero-based index into the palette strip, and recolor only the fillable interior black cells with the chosen palette color.
    - Keep all boundary pixels and all holes unchanged, and restore the palette strip itself to background.

    Key insight:
    - The boundary network defines alternating inside/outside layers; the number of even-depth enclosed black regions selects the refill color from the edge legend.
    """
    g = np.asarray(grid, dtype=int)
    legend_colors, legend_mask = _legend_info(g)

    work = g.copy()
    work[legend_mask] = 0
    out = work.copy()

    for comp in _components_mask(work != 0):
        r0, c0, r1, c1 = _bbox(comp)
        walls = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in comp:
            walls[r - r0, c - c0] = True

        region_labels, fill_labels, hole_labels = _classify_regions(walls)
        hole_count = len(hole_labels)
        if hole_count >= len(legend_colors):
            continue
        fill_color = int(legend_colors[hole_count])
        if not fill_labels:
            continue
        fill_mask = (~walls) & np.isin(region_labels, list(fill_labels))
        for lr, lc in np.argwhere(fill_mask):
            out[r0 + int(lr), c0 + int(lc)] = fill_color

    return out


def _rect_mask(shape: tuple[int, int], top: int, left: int, height: int, width: int) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    out[top : top + height, left : left + width] = True
    return out


def _find_bridge(
    outer: np.ndarray,
    outer_boundary: np.ndarray,
    rect: np.ndarray,
    full_holes: np.ndarray,
    bridges: np.ndarray,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    rows, cols = np.where(rect)
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    sides = [
        ((-1, 0), [(r0, c) for c in range(c0, c1 + 1)]),
        ((1, 0), [(r1, c) for c in range(c0, c1 + 1)]),
        ((0, -1), [(r, c0) for r in range(r0, r1 + 1)]),
        ((0, 1), [(r, c1) for r in range(r0, r1 + 1)]),
    ]
    rng.shuffle(sides)

    best_path = None
    for (dr, dc), starts in sides:
        rng.shuffle(starts)
        for sr, sc in starts:
            path: list[tuple[int, int]] = []
            r, c = sr, sc
            while 0 <= r < outer.shape[0] and 0 <= c < outer.shape[1] and outer[r, c]:
                path.append((int(r), int(c)))
                r += dr
                c += dc
            if not path or not outer_boundary[path[-1]]:
                continue
            ok = True
            for pr, pc in path[1:-1]:
                if full_holes[pr, pc] or bridges[pr, pc]:
                    ok = False
                    break
            if not ok or bridges[path[-1]]:
                continue
            if best_path is None or len(path) < len(best_path):
                best_path = path
    return best_path


def _make_shape(hole_count: int, rng: random.Random) -> np.ndarray | None:
    for _ in range(SHAPE_RETRIES):
        base_h = rng.randint(max(7, 6 + hole_count), min(13, 10 + hole_count))
        base_w = rng.randint(max(7, 7 + hole_count), min(14, 11 + hole_count))
        canvas = np.zeros((base_h + 6, base_w + 6), dtype=bool)
        top, left = 3, 3

        outer = np.zeros_like(canvas, dtype=bool)
        outer[top : top + base_h, left : left + base_w] = True

        for _ in range(rng.randint(0, 2)):
            side = rng.choice(["top", "bottom", "left", "right"])
            if side in ("top", "bottom"):
                span = rng.randint(max(2, base_w // 3), base_w)
                start = rng.randint(left, left + base_w - span)
                thick = rng.randint(1, 2)
                if side == "top":
                    outer[top - thick : top, start : start + span] = True
                else:
                    outer[top + base_h : top + base_h + thick, start : start + span] = True
            else:
                span = rng.randint(max(2, base_h // 3), base_h)
                start = rng.randint(top, top + base_h - span)
                thick = rng.randint(1, 2)
                if side == "left":
                    outer[start : start + span, left - thick : left] = True
                else:
                    outer[start : start + span, left + base_w : left + base_w + thick] = True

        outer_boundary = _boundary(outer)
        core = _erode4(outer)
        if int(core.sum()) < 9 + 6 * hole_count:
            continue

        hole_rects: list[np.ndarray] = []
        reserved = np.zeros_like(canvas, dtype=bool)
        ok = True
        for _hole_idx in range(hole_count):
            placed = False
            for _place_try in range(80):
                qh = rng.choice([3, 4, 5])
                qw = rng.choice([3, 4, 5])
                if base_h - qh < 2 or base_w - qw < 2:
                    continue
                rr = rng.randint(top + 1, top + base_h - qh - 1)
                cc = rng.randint(left + 1, left + base_w - qw - 1)
                rect = _rect_mask(canvas.shape, rr, cc, qh, qw)
                if not np.all(core[rect]):
                    continue
                r0 = max(0, rr - 1)
                r1 = min(canvas.shape[0], rr + qh + 1)
                c0 = max(0, cc - 1)
                c1 = min(canvas.shape[1], cc + qw + 1)
                if reserved[r0:r1, c0:c1].any():
                    continue
                reserved[r0:r1, c0:c1] = True
                hole_rects.append(rect)
                placed = True
                break
            if not placed:
                ok = False
                break
        if not ok:
            continue

        full_holes = np.zeros_like(canvas, dtype=bool)
        hole_boundaries = np.zeros_like(canvas, dtype=bool)
        for rect in hole_rects:
            full_holes |= rect
            hole_boundaries |= _boundary(rect)

        bridges = np.zeros_like(canvas, dtype=bool)
        rng.shuffle(hole_rects)
        for rect in hole_rects:
            path = _find_bridge(outer, outer_boundary, rect, full_holes, bridges, rng)
            if path is None:
                ok = False
                break
            for r, c in path:
                bridges[r, c] = True
        if not ok:
            continue

        walls = outer_boundary | hole_boundaries | bridges
        if len(_components_mask(walls)) != 1:
            continue

        region_labels, fill_labels, hole_labels = _classify_regions(walls)
        if len(hole_labels) != hole_count or not fill_labels:
            continue
        fill_mask = (~walls) & np.isin(region_labels, list(fill_labels))
        if not fill_mask.any():
            continue

        pts = np.argwhere(walls)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        return walls[r0 : r1 + 1, c0 : c1 + 1]
    return None


def _expected_output(
    grid: np.ndarray,
    placements: list[tuple[np.ndarray, int, int, int]],
    legend_colors: list[int],
) -> np.ndarray:
    out = np.zeros_like(grid)
    for walls, top, left, hole_count in placements:
        h, w = walls.shape
        out[top : top + h, left : left + w][walls] = grid[top : top + h, left : left + w][walls]
        region_labels, fill_labels, _ = _classify_regions(walls)
        fill_mask = (~walls) & np.isin(region_labels, list(fill_labels))
        out[top : top + h, left : left + w][fill_mask] = int(legend_colors[hole_count])
    return out


def _sample_hole_counts(legend_len: int, shape_count: int, rng: random.Random) -> list[int]:
    counts = [0, rng.randint(1, legend_len - 1)]
    while len(counts) < shape_count:
        counts.append(rng.randint(0, legend_len - 1))
    rng.shuffle(counts)
    return counts


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    orientation = rng.choice(["bottom", "right"])
    shape_color = rng.randint(1, 9)
    legend_len = rng.randint(3, 5)
    legend_pool = [c for c in range(1, 10) if c != shape_color]
    legend_colors = rng.sample(legend_pool, legend_len)
    shape_count = rng.randint(2, min(4, legend_len + 1))
    hole_counts = _sample_hole_counts(legend_len, shape_count, rng)

    walls_list: list[tuple[np.ndarray, int]] = []
    for hole_count in hole_counts:
        walls = _make_shape(hole_count, rng)
        if walls is None:
            return None
        walls_list.append((walls, hole_count))

    max_h = max(w.shape[0] for w, _ in walls_list)
    max_w = max(w.shape[1] for w, _ in walls_list)
    min_side = max(18, max(max_h, max_w) + 4)

    for _ in range(PLACEMENT_RETRIES):
        h = rng.randint(min_side, 30)
        w = rng.randint(min_side, 30)
        if orientation == "bottom" and w < legend_len:
            continue
        if orientation == "right" and h < legend_len:
            continue

        grid = np.zeros((h, w), dtype=int)
        blocked = np.zeros((h, w), dtype=bool)
        placements: list[tuple[np.ndarray, int, int, int]] = []
        ok = True

        order = sorted(range(shape_count), key=lambda i: -walls_list[i][0].sum())
        for idx in order:
            walls, hole_count = walls_list[idx]
            sh, sw = walls.shape
            max_top = h - sh - 1
            max_left = w - sw - 1
            if max_top < 1 or max_left < 1:
                ok = False
                break

            placed = False
            for _place_try in range(160):
                top = rng.randint(1, max_top)
                left = rng.randint(1, max_left)
                r0 = max(0, top - 1)
                r1 = min(h, top + sh + 1)
                c0 = max(0, left - 1)
                c1 = min(w, left + sw + 1)
                if blocked[r0:r1, c0:c1].any():
                    continue
                grid[top : top + sh, left : left + sw][walls] = shape_color
                blocked[r0:r1, c0:c1] = True
                placements.append((walls, top, left, hole_count))
                placed = True
                break
            if not placed:
                ok = False
                break
        if not ok:
            continue

        if orientation == "bottom":
            start = rng.randint(0, w - legend_len)
            grid[h - 1, start : start + legend_len] = np.array(legend_colors, dtype=int)
        else:
            start = rng.randint(0, h - legend_len)
            grid[start : start + legend_len, w - 1] = np.array(legend_colors, dtype=int)

        expected = _expected_output(grid, placements, legend_colors)
        actual = main(grid)
        if np.array_equal(actual, expected):
            return grid
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {GEN_RETRIES} attempts")
