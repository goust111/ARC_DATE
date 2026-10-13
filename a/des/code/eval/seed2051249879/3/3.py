import random
from collections import Counter, deque

import numpy as np


_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_N8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

_GEN_ATTEMPTS = 300
_COLORS = tuple(range(10))


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    nbrs = _N8 if connectivity == 8 else _N4
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in nbrs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.asarray(pts, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0, r1, c1 = _bbox(coords)
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def _hole_count(mask: np.ndarray) -> int:
    obj = _crop_mask(mask)
    if obj.size == 0:
        return 0
    bg = ~obj
    h, w = bg.shape
    seen = np.zeros_like(bg, dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if not bg[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            touches_border = False
            while q:
                cr, cc = q.popleft()
                if cr == 0 or cr == h - 1 or cc == 0 or cc == w - 1:
                    touches_border = True
                for dr, dc in _N4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches_border:
                holes += 1
    return holes


def _outer_ring(mask: np.ndarray) -> list[tuple[int, int]]:
    obj = np.asarray(mask, dtype=bool)
    padded = np.pad(obj, 1, constant_values=False)
    bg = ~padded
    h, w = bg.shape
    exterior = np.zeros_like(bg, dtype=bool)
    q = deque()

    for r in range(h):
        for c in (0, w - 1):
            if bg[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if bg[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))

    while q:
        cr, cc = q.popleft()
        for dr, dc in _N4:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                q.append((nr, nc))

    ring = []
    for r in range(h):
        for c in range(w):
            if not exterior[r, c]:
                continue
            for dr, dc in _N8:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and padded[nr, nc]:
                    ring.append((r - 1, c - 1))
                    break
    return ring


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group blue objects by their number of enclosed background holes and draw
    a one-pixel exterior frame using the legend colors from the rightmost column.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _most_common_color(g)

    legend_rows = [r for r in range(h) if int(g[r, w - 1]) != bg]
    assert len(legend_rows) == 4
    legend_colors = [int(g[r, w - 1]) for r in legend_rows]

    object_colors = {int(v) for v in np.unique(g[:, : w - 1]) if int(v) != bg}
    assert len(object_colors) == 1
    obj_color = next(iter(object_colors))

    obj_mask = g == obj_color
    obj_mask[:, w - 1] = False
    comps = _components(obj_mask, connectivity=4)
    comps.sort(key=lambda pts: (int(pts[:, 0].min()), int(pts[:, 1].min())))

    out = g.copy()
    for pts in comps:
        r0, c0, r1, c1 = _bbox(pts)
        sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        sub[pts[:, 0] - r0, pts[:, 1] - c0] = True
        holes = _hole_count(sub)
        assert 0 <= holes < len(legend_colors)
        border_color = legend_colors[holes]
        for rr, cc in _outer_ring(sub):
            gr, gc = r0 + rr, c0 + cc
            if 0 <= gr < h and 0 <= gc < w and int(g[gr, gc]) == bg and int(out[gr, gc]) == bg:
                out[gr, gc] = border_color
    return out


def _component_count(mask: np.ndarray) -> int:
    return len(_components(mask, connectivity=4))


def _boundary_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            for dr, dc in _N4:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w) or not mask[nr, nc]:
                    cells.append((r, c))
                    break
    return cells


def _random_transform(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    out = np.asarray(mask, dtype=bool)
    if rng.random() < 0.5:
        out = np.flipud(out)
    if rng.random() < 0.5:
        out = np.fliplr(out)
    if rng.random() < 0.5:
        out = out.T
    return np.asarray(out, dtype=bool)


def _hole_shapes(hole_count: int) -> list[list[tuple[int, int]]]:
    if hole_count >= 3:
        return [[(0, 0)]]
    if hole_count == 2:
        return [[(0, 0)], [(0, 0), (0, 1)], [(0, 0), (1, 0)]]
    return [
        [(0, 0)],
        [(0, 0), (0, 1)],
        [(0, 0), (1, 0)],
        [(0, 0), (0, 1), (1, 0)],
        [(0, 0), (0, 1), (1, 1)],
        [(0, 0), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
    ]


def _initial_dims(rng: random.Random, hole_count: int) -> tuple[int, int]:
    if hole_count == 0:
        lo = 6
    elif hole_count == 1:
        lo = 6
    else:
        lo = 7
    return rng.randint(lo, 8), rng.randint(lo, 8)


def _place_holes(mask: np.ndarray, hole_count: int, rng: random.Random) -> np.ndarray | None:
    h, w = mask.shape
    inner_h, inner_w = h - 2, w - 2
    forbidden = np.zeros((inner_h, inner_w), dtype=bool)

    for _hole_idx in range(hole_count):
        placed = False
        shapes = _hole_shapes(hole_count)
        for _ in range(120):
            shape = rng.choice(shapes)
            sh = 1 + max(r for r, _ in shape)
            sw = 1 + max(c for _, c in shape)
            if sh > inner_h or sw > inner_w:
                continue
            r0 = rng.randint(0, inner_h - sh)
            c0 = rng.randint(0, inner_w - sw)
            cells = [(r0 + dr, c0 + dc) for dr, dc in shape]
            if any(forbidden[r, c] for r, c in cells):
                continue
            for r, c in cells:
                mask[r + 1, c + 1] = False
            for r, c in cells:
                r_lo = max(0, r - 1)
                r_hi = min(inner_h, r + 2)
                c_lo = max(0, c - 1)
                c_hi = min(inner_w, c + 2)
                forbidden[r_lo:r_hi, c_lo:c_hi] = True
            placed = True
            break
        if not placed:
            return None
    return mask


def _try_remove(mask: np.ndarray, cell: tuple[int, int], hole_count: int) -> np.ndarray | None:
    r, c = cell
    trial = mask.copy()
    trial[r, c] = False
    if not trial.any() or _component_count(trial) != 1:
        return None
    cropped = _crop_mask(trial)
    if cropped.shape[0] < 5 or cropped.shape[1] < 5:
        return None
    if _hole_count(cropped) != hole_count:
        return None
    min_area = 12 if hole_count <= 1 else 16
    if int(cropped.sum()) < min_area:
        return None
    return trial


def _make_object_mask(hole_count: int, rng: random.Random) -> np.ndarray:
    for _ in range(120):
        h, w = _initial_dims(rng, hole_count)
        base = np.ones((h, w), dtype=bool)
        base = _place_holes(base, hole_count, rng)
        if base is None or _hole_count(base) != hole_count:
            continue

        target_removals = rng.randint(1 if hole_count == 0 else 0, 4)
        current = base
        removals = 0
        for _step in range(40):
            candidates = _boundary_cells(current)
            rng.shuffle(candidates)
            changed = False
            for cell in candidates:
                updated = _try_remove(current, cell, hole_count)
                if updated is None:
                    continue
                current = updated
                removals += 1
                changed = True
                break
            if removals >= target_removals or not changed:
                break

        final = _crop_mask(current)
        final = _random_transform(final, rng)
        final = _crop_mask(final)
        if final.shape[0] < 5 or final.shape[1] < 5 or final.shape[0] > 8 or final.shape[1] > 8:
            continue
        if _component_count(final) != 1 or _hole_count(final) != hole_count:
            continue
        if hole_count == 0 and removals == 0:
            continue
        return final.astype(bool)
    raise RuntimeError(f"failed to generate object with {hole_count} holes")


def _legend_rows(h: int, rng: random.Random) -> list[int] | None:
    gaps = [rng.randint(1, 2) for _ in range(3)]
    span = 4 + sum(gaps)
    if h < span + 2:
        return None
    start = rng.randint(1, h - span - 1)
    rows = [start]
    for gap in gaps:
        rows.append(rows[-1] + gap + 1)
    return rows


def _reserve_neighborhood(blocked: np.ndarray, coords: np.ndarray) -> None:
    h, w = blocked.shape
    for r, c in coords:
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                nr, nc = int(r) + dr, int(c) + dc
                if 0 <= nr < h and 0 <= nc < w:
                    blocked[nr, nc] = True


def _place_objects(
    grid: np.ndarray,
    masks: list[np.ndarray],
    obj_color: int,
    rng: random.Random,
) -> list[tuple[np.ndarray, int, int]] | None:
    h, w = grid.shape
    blocked = np.zeros((h, w), dtype=bool)
    blocked[0, :] = True
    blocked[h - 1, :] = True
    blocked[:, 0] = True
    blocked[:, 1] = True
    blocked[:, w - 1] = True
    blocked[:, w - 2] = True
    blocked[:, w - 3] = True
    placements: list[tuple[np.ndarray, int, int]] = []

    order = list(range(len(masks)))
    rng.shuffle(order)
    for idx in order:
        mask = masks[idx]
        mh, mw = mask.shape
        placed = False
        row_lo, row_hi = 1, h - mh - 2
        col_lo, col_hi = 2, w - mw - 4
        if row_lo > row_hi or col_lo > col_hi:
            return None
        for _ in range(240):
            r0 = rng.randint(row_lo, row_hi)
            c0 = rng.randint(col_lo, col_hi)
            coords = np.argwhere(mask) + np.array([r0, c0])
            if blocked[coords[:, 0], coords[:, 1]].any():
                continue
            grid[coords[:, 0], coords[:, 1]] = obj_color
            _reserve_neighborhood(blocked, coords)
            placements.append((coords, r0, c0))
            placed = True
            break
        if not placed:
            return None
    return placements


def _scene_is_valid(
    grid: np.ndarray,
    obj_color: int,
    expected_holes: list[int],
    legend_rows: list[int],
    legend_colors: list[int],
) -> bool:
    bg = _most_common_color(grid)
    h, w = grid.shape
    if [r for r in range(h) if int(grid[r, w - 1]) != bg] != legend_rows:
        return False
    if [int(grid[r, w - 1]) for r in legend_rows] != legend_colors:
        return False

    obj_mask = grid == obj_color
    comps = _components(obj_mask, connectivity=4)
    if len(comps) != len(expected_holes):
        return False
    actual_holes = []
    used_ring: set[tuple[int, int]] = set()
    out = main(grid)
    for pts in comps:
        r0, c0, r1, c1 = _bbox(pts)
        sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        sub[pts[:, 0] - r0, pts[:, 1] - c0] = True
        holes = _hole_count(sub)
        actual_holes.append(holes)
        ring = _outer_ring(sub)
        for rr, cc in ring:
            gr, gc = r0 + rr, c0 + cc
            if not (0 <= gr < h and 0 <= gc < w):
                return False
            if int(grid[gr, gc]) != bg:
                return False
            if (gr, gc) in used_ring:
                return False
            used_ring.add((gr, gc))
            if int(out[gr, gc]) != legend_colors[holes]:
                return False
    if Counter(actual_holes) != Counter(expected_holes):
        return False
    if set(actual_holes) != {0, 1, 2, 3}:
        return False
    if not np.array_equal(out[grid == obj_color], grid[grid == obj_color]):
        return False
    if not np.array_equal(out[:, w - 1], grid[:, w - 1]):
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    obj_count = rng.randint(4, 5)
    hole_counts = [0, 1, 2, 3]
    while len(hole_counts) < obj_count:
        hole_counts.append(rng.randint(0, 3))
    rng.shuffle(hole_counts)

    masks = [_make_object_mask(hc, rng) for hc in hole_counts]

    min_h = max(18, max(mask.shape[0] for mask in masks) + 8)
    min_w = max(22, max(mask.shape[1] for mask in masks) + 10)
    h = rng.randint(min_h, 28)
    w = rng.randint(min_w, 30)

    bg = rng.choice(_COLORS)
    remaining = [c for c in _COLORS if c != bg]
    legend_colors = rng.sample(remaining, 4)
    remaining = [c for c in remaining if c not in legend_colors]
    obj_color = rng.choice(remaining)

    grid = np.full((h, w), bg, dtype=int)
    legend_rows = _legend_rows(h, rng)
    if legend_rows is None:
        return None
    for r, color in zip(legend_rows, legend_colors):
        grid[r, w - 1] = color

    placements = _place_objects(grid, masks, obj_color, rng)
    if placements is None:
        return None

    if not _scene_is_valid(grid, obj_color, hole_counts, legend_rows, legend_colors):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
