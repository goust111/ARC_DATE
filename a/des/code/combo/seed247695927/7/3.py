import random
from collections import Counter, deque

import numpy as np


GREY = 5
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _hole_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            touches_border = r == 0 or r == h - 1 or c == 0 or c == w - 1
            while q:
                cr, cc = q.popleft()
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if mask[nr, nc] or seen[nr, nc]:
                        continue
                    seen[nr, nc] = True
                    if nr == 0 or nr == h - 1 or nc == 0 or nc == w - 1:
                        touches_border = True
                    q.append((nr, nc))
            if not touches_border:
                holes += 1
    return int(holes)


def _component_hole_count(coords: np.ndarray) -> int:
    r0, c0, r1, c1 = _bbox(coords)
    sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        sub[r - r0, c - c0] = True
    return _hole_count(sub)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)

    legend = {}
    for color in np.unique(g):
        color = int(color)
        if color in (bg, GREY):
            continue
        for coords in _components(g == color):
            legend[_component_hole_count(coords)] = color

    out = g.copy()
    for coords in _components(g == GREY):
        holes = _component_hole_count(coords)
        new_color = legend.get(holes, bg)
        for r, c in coords:
            out[r, c] = int(new_color)
    return out


def _rand_blob(rng: random.Random, min_h: int = 3, max_h: int = 7, min_w: int = 3, max_w: int = 7) -> np.ndarray:
    for _ in range(200):
        h = rng.randint(min_h, max_h)
        w = rng.randint(min_w, max_w)
        target = rng.randint(max(4, (h * w) // 3), max(4, (2 * h * w) // 3))
        pts = {(rng.randrange(h), rng.randrange(w))}
        frontier = list(pts)
        while len(pts) < target and frontier:
            br, bc = rng.choice(frontier)
            cand = []
            for dr, dc in DIR4:
                nr, nc = br + dr, bc + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in pts:
                    cand.append((nr, nc))
            if not cand:
                frontier.remove((br, bc))
                continue
            npix = rng.choice(cand)
            pts.add(npix)
            frontier.append(npix)
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        mask = np.zeros((max(rs) - min(rs) + 1, max(cs) - min(cs) + 1), dtype=bool)
        for r, c in pts:
            mask[r - min(rs), c - min(cs)] = True
        if _hole_count(mask) != 0:
            continue
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        return mask
    return np.array([[1, 1], [1, 0], [1, 1]], dtype=bool)


def _base_holey_shape(hole_count: int, rng: random.Random) -> np.ndarray:
    if hole_count == 0:
        return _rand_blob(rng)

    if rng.random() < 0.5:
        rows, cols = 1, hole_count
    else:
        rows = rng.randint(1, hole_count)
        while hole_count % rows != 0:
            rows = rng.randint(1, hole_count)
        cols = hole_count // rows
    h = rows * 4 - 1
    w = cols * 4 - 1
    mask = np.zeros((h, w), dtype=bool)
    for rr in range(rows):
        for cc in range(cols):
            r0 = rr * 4
            c0 = cc * 4
            mask[r0 : r0 + 3, c0 : c0 + 3] = True
            mask[r0 + 1, c0 + 1] = False
    for rr in range(rows):
        for cc in range(cols - 1):
            mask[rr * 4 + 1, cc * 4 + 3] = True
    for rr in range(rows - 1):
        for cc in range(cols):
            mask[rr * 4 + 3, cc * 4 + 1] = True
    return mask


def _add_protrusions(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    border = []
    for r in range(h):
        for c in range(w):
            if not out[r, c]:
                continue
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not out[nr, nc]:
                    border.append((r, c, dr, dc))
    rng.shuffle(border)
    for r, c, dr, dc in border[: rng.randint(0, min(4, len(border)))]:
        length = rng.randint(1, 2)
        cr, cc = r, c
        for _ in range(length):
            cr += dr
            cc += dc
            if not (0 <= cr < h and 0 <= cc < w):
                break
            out[cr, cc] = True
    rows = np.any(out, axis=1)
    cols = np.any(out, axis=0)
    out = out[rows][:, cols]
    return out if _hole_count(out) == _hole_count(mask) else mask


def _shape_with_holes(hole_count: int, rng: random.Random) -> np.ndarray:
    mask = _base_holey_shape(hole_count, rng)
    mask = _add_protrusions(mask, rng)
    if rng.random() < 0.5:
        mask = np.rot90(mask, rng.randrange(4))
    else:
        mask = np.fliplr(np.rot90(mask, rng.randrange(4)))
    return mask.astype(bool)


def _dilate_bbox(bbox: tuple[int, int, int, int], pad: int, h: int, w: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _bbox_overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _paint(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    sub = grid[top : top + h, left : left + w]
    sub[mask] = int(color)


def _candidate_position(
    mask: np.ndarray,
    grid_shape: tuple[int, int],
    occupied: list[tuple[int, int, int, int]],
    rng: random.Random,
    edge_mode: str | None,
) -> tuple[int, int] | None:
    gh, gw = grid_shape
    h, w = mask.shape
    if h >= gh or w >= gw:
        return None
    for _ in range(200):
        if edge_mode == "top":
            top = 0
            left = rng.randint(0, gw - w)
        elif edge_mode == "bottom":
            top = gh - h
            left = rng.randint(0, gw - w)
        elif edge_mode == "left":
            top = rng.randint(0, gh - h)
            left = 0
        elif edge_mode == "right":
            top = rng.randint(0, gh - h)
            left = gw - w
        else:
            top_lo = 1 if gh - h >= 2 else 0
            top_hi = gh - h - 1 if gh - h >= 2 else gh - h
            left_lo = 1 if gw - w >= 2 else 0
            left_hi = gw - w - 1 if gw - w >= 2 else gw - w
            top = rng.randint(top_lo, top_hi)
            left = rng.randint(left_lo, left_hi)
        bbox = (top, left, top + h - 1, left + w - 1)
        grown = _dilate_bbox(bbox, 1, gh, gw)
        if any(_bbox_overlaps(grown, ob) for ob in occupied):
            continue
        return int(top), int(left)
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = rng.choice([0, 1, 2, 3])
    palette = [c for c in range(10) if c not in (bg, GREY)]
    rng.shuffle(palette)

    gh = rng.randint(18, 30)
    gw = rng.randint(18, 30)
    grid = np.full((gh, gw), bg, dtype=int)

    available_holes = [0, 1, 2, 3, 4]
    legend_count = rng.randint(2, 4)
    legend_holes = sorted(rng.sample(available_holes[:-1] if min(gh, gw) < 22 else available_holes, legend_count))
    distractor_holes = [h for h in available_holes if h not in legend_holes]
    if not distractor_holes:
        return None

    occupied = []
    legend_colors = palette[:legend_count]
    edges = ["top", "bottom", "left", "right"]
    rng.shuffle(edges)

    for idx, hole_count in enumerate(legend_holes):
        mask = _shape_with_holes(hole_count, rng)
        pos = _candidate_position(mask, grid.shape, occupied, rng, edges[idx % len(edges)])
        if pos is None:
            return None
        top, left = pos
        _paint(grid, mask, top, left, legend_colors[idx])
        bbox = (top, left, top + mask.shape[0] - 1, left + mask.shape[1] - 1)
        occupied.append(_dilate_bbox(bbox, 1, gh, gw))

    sprite_holes = []
    match_total = rng.randint(2, 5)
    for _ in range(match_total):
        sprite_holes.append(rng.choice(legend_holes))
    distract_total = rng.randint(1, 3)
    for _ in range(distract_total):
        sprite_holes.append(rng.choice(distractor_holes))
    rng.shuffle(sprite_holes)

    matched_seen = False
    unmatched_seen = False
    for hole_count in sprite_holes:
        mask = _shape_with_holes(hole_count, rng)
        pos = _candidate_position(mask, grid.shape, occupied, rng, None)
        if pos is None:
            return None
        top, left = pos
        _paint(grid, mask, top, left, GREY)
        bbox = (top, left, top + mask.shape[0] - 1, left + mask.shape[1] - 1)
        occupied.append(_dilate_bbox(bbox, 1, gh, gw))
        matched_seen |= hole_count in legend_holes
        unmatched_seen |= hole_count not in legend_holes

    if not (matched_seen and unmatched_seen):
        return None

    out = main(grid)
    recolored = Counter(int(v) for v in out[grid == GREY])
    if sum(recolored[c] for c in legend_colors if c in recolored) == 0:
        return None
    if recolored.get(bg, 0) == 0:
        return None
    if len(_components(grid == GREY)) != len(sprite_holes):
        return None
    if len({(_component_hole_count(c), int(grid[c[0, 0], c[0, 1]])) for color in legend_colors for c in _components(grid == color)}) < legend_count:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
