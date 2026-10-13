from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
WALLPAPER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
TARGET_DIMS = (
    (3, 4),
    (4, 3),
    (3, 5),
    (5, 3),
    (3, 6),
    (6, 3),
    (4, 4),
    (4, 5),
    (5, 4),
    (4, 6),
    (6, 4),
    (5, 6),
    (6, 5),
)
GEN_RETRIES = 400


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_shape(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            out.append(np.asarray(coords, dtype=int))
    return out


def _gray_rectangles(grid: np.ndarray) -> list[dict]:
    rects: list[dict] = []
    for coords in _components(grid == GREY):
        bbox = _bbox_coords(coords)
        h, w = _bbox_shape(bbox)
        r0, c0, r1, c1 = bbox
        assert coords.shape[0] == h * w, "grey component must be a solid rectangle"
        assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == GREY), "grey bbox must be solid"
        rects.append({"coords": coords, "bbox": bbox, "shape": (h, w)})
    return rects


def _tile_fill(shape: tuple[int, int], tile: np.ndarray, offset: tuple[int, int]) -> np.ndarray:
    h, w = shape
    th, tw = tile.shape
    off_r, off_c = offset
    rr = ((np.arange(h) - off_r) % th)[:, None]
    cc = ((np.arange(w) - off_c) % tw)[None, :]
    return tile[rr, cc]


def _period_tile(
    observed_coords: np.ndarray,
    observed_vals: np.ndarray,
    ph: int,
    pw: int,
) -> np.ndarray | None:
    tile = np.full((ph, pw), -1, dtype=int)
    for (r, c), value in zip(observed_coords, observed_vals):
        tr = int(r) % ph
        tc = int(c) % pw
        old = int(tile[tr, tc])
        val = int(value)
        if old == -1:
            tile[tr, tc] = val
        elif old != val:
            return None
    if np.any(tile < 0):
        return None
    return tile


def _infer_unique_period(grid: np.ndarray) -> tuple[int, int, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    observed_mask = (g != BLACK) & (g != GREY)
    observed_coords = np.argwhere(observed_mask)
    observed_vals = g[observed_mask]
    assert observed_coords.size > 0, "need visible wallpaper cells"

    candidates: list[tuple[int, int, int, np.ndarray]] = []
    h, w = g.shape
    for ph in range(1, h + 1):
        for pw in range(1, w + 1):
            tile = _period_tile(observed_coords, observed_vals, ph, pw)
            if tile is not None:
                candidates.append((ph * pw, ph, pw, tile))

    assert candidates, "no consistent wallpaper period"
    best_area = min(item[0] for item in candidates)
    best = [item for item in candidates if item[0] == best_area]
    assert len(best) == 1, "wallpaper period must be uniquely minimal"
    _area, ph, pw, tile = best[0]
    return int(ph), int(pw), tile


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the periodic wallpaper hole and transfer that hole mask into the only same-sized gray rectangle.

    Input structure:
    - The grid is a multicolor wallpaper formed by repeating a small rectangular tile.
    - One connected black hole interrupts the wallpaper.
    - One or more solid gray rectangles also appear; exactly one has the same height and width
      as the black hole's bounding box.

    Rule:
    - Infer the wallpaper's uniquely minimal 2D period from the visible non-black, non-gray cells.
    - Reconstruct every black hole cell from that period.
    - Find the gray rectangle whose size matches the hole bounding box and paint black exactly at
      the hole mask's relative positions inside that rectangle; leave its other cells gray.
    - Keep all other gray rectangles unchanged.

    Key insight:
    - The anomaly is dual-purpose: its missing wallpaper cells must be healed by periodicity, and
      the hole's bounding-box-relative mask is the unseen stamp that must be copied into the
      matching gray canvas.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    hole_components = _components(g == BLACK)
    assert len(hole_components) == 1, "expected exactly one connected black hole"
    hole_coords = hole_components[0]
    hole_bbox = _bbox_coords(hole_coords)
    hole_shape = _bbox_shape(hole_bbox)
    hole_mask = _crop_bbox(g == BLACK, hole_bbox)

    rects = _gray_rectangles(g)
    assert rects, "expected at least one gray rectangle"
    matched = [rect for rect in rects if rect["shape"] == hole_shape]
    assert len(matched) == 1, "expected one uniquely size-matched gray rectangle"
    target_bbox = matched[0]["bbox"]

    ph, pw, tile = _infer_unique_period(g)

    out = g.copy()
    for r, c in hole_coords:
        out[int(r), int(c)] = int(tile[int(r) % ph, int(c) % pw])

    tr0, tc0, tr1, tc1 = target_bbox
    stamp = np.full(hole_shape, GREY, dtype=int)
    stamp[hole_mask] = BLACK
    out[tr0 : tr1 + 1, tc0 : tc1 + 1] = stamp
    return out


def _random_path(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = start
    er, ec = end
    path = [(r, c)]
    while (r, c) != (er, ec):
        options = []
        if r < er:
            options.append((r + 1, c))
        elif r > er:
            options.append((r - 1, c))
        if c < ec:
            options.append((r, c + 1))
        elif c > ec:
            options.append((r, c - 1))
        r, c = random.choice(options)
        path.append((r, c))
    return path


def _frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells = set()
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for dr, dc in DIRS4:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                cells.add((nr, nc))
    return list(cells)


def _hole_mask(h: int, w: int) -> np.ndarray | None:
    area = h * w
    for _ in range(120):
        mask = np.zeros((h, w), dtype=bool)
        hub = (random.randrange(h), random.randrange(w))
        anchors = [
            (0, random.randrange(w)),
            (h - 1, random.randrange(w)),
            (random.randrange(h), 0),
            (random.randrange(h), w - 1),
        ]
        mask[hub] = True
        for anchor in anchors:
            for r, c in _random_path(hub, anchor):
                mask[r, c] = True

        if int(mask.sum()) >= area:
            continue

        min_area = max(int(mask.sum()), max(4, int(np.ceil(0.40 * area))))
        max_area = min(area - 1, int(np.floor(0.78 * area)))
        if min_area > max_area:
            continue
        target_area = random.randint(min_area, max_area)
        while int(mask.sum()) < target_area:
            edge = _frontier(mask)
            if not edge:
                break
            rr, cc = random.choice(edge)
            mask[rr, cc] = True

        coords = np.argwhere(mask)
        if coords.size == 0 or int(mask.sum()) >= area:
            continue
        if _bbox_coords(coords) != (0, 0, h - 1, w - 1):
            continue
        return mask
    return None


def _minimal_period_shape(grid: np.ndarray) -> tuple[int, int]:
    ph, pw, _tile = _infer_unique_period(grid)
    return int(ph), int(pw)


def _random_tile(shape: tuple[int, int], palette: list[int]) -> np.ndarray | None:
    th, tw = shape
    for _ in range(160):
        tile = np.random.choice(palette, size=(th, tw)).astype(int)
        if len(set(map(int, tile.ravel()))) < min(3, len(palette)):
            continue
        if _minimal_period_shape(tile) != (th, tw):
            continue
        return tile
    return None


def _same_area_dims(target_shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = target_shape
    area = h * w
    out = []
    for rh in range(2, 9):
        if area % rh != 0:
            continue
        rw = area // rh
        if 2 <= rw <= 8 and (rh, rw) != (h, w):
            out.append((int(rh), int(rw)))
    return out


def _choose_rectangles(target_shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = target_shape
    total = random.choice((2, 3, 3, 4))
    distractor_count = total - 1
    dims = [target_shape]
    used = {target_shape}

    same_area = _same_area_dims(target_shape)
    if same_area and len(dims) <= distractor_count:
        choice = random.choice(same_area)
        dims.append(choice)
        used.add(choice)

    side_candidates = []
    for ww in range(2, 9):
        if ww != w:
            side_candidates.append((h, ww))
    for hh in range(2, 9):
        if hh != h:
            side_candidates.append((hh, w))
    random.shuffle(side_candidates)
    for choice in side_candidates:
        if len(dims) > distractor_count:
            break
        if choice not in used:
            dims.append(choice)
            used.add(choice)

    all_choices = [(hh, ww) for hh in range(2, 9) for ww in range(2, 9) if (hh, ww) != target_shape]
    random.shuffle(all_choices)
    for choice in all_choices:
        if len(dims) > distractor_count:
            break
        if choice not in used:
            dims.append(choice)
            used.add(choice)

    assert len(dims) == total
    return dims


def _place_boxes(shape: tuple[int, int], sizes: list[tuple[int, int]], border: int = 1, gap: int = 1):
    h, w = shape
    placement_order = sorted(range(len(sizes)), key=lambda i: sizes[i][0] * sizes[i][1], reverse=True)
    occupied = np.zeros(shape, dtype=bool)
    positions: list[tuple[int, int] | None] = [None] * len(sizes)

    for idx in placement_order:
        bh, bw = sizes[idx]
        candidates = []
        for r0 in range(border, h - border - bh + 1):
            for c0 in range(border, w - border - bw + 1):
                rr0 = max(0, r0 - gap)
                cc0 = max(0, c0 - gap)
                rr1 = min(h, r0 + bh + gap)
                cc1 = min(w, c0 + bw + gap)
                if not occupied[rr0:rr1, cc0:cc1].any():
                    candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        positions[idx] = (r0, c0)
        rr0 = max(0, r0 - gap)
        cc0 = max(0, c0 - gap)
        rr1 = min(h, r0 + bh + gap)
        cc1 = min(w, c0 + bw + gap)
        occupied[rr0:rr1, cc0:cc1] = True

    return [pos for pos in positions if pos is not None]


def _render_instance(
    grid_shape: tuple[int, int],
    tile: np.ndarray,
    offset: tuple[int, int],
    hole_mask: np.ndarray,
    placements: list[tuple[int, int]],
    rect_sizes: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    wallpaper = _tile_fill(grid_shape, tile, offset)
    inp = wallpaper.copy()
    out = wallpaper.copy()

    hole_r0, hole_c0 = placements[0]
    hole_h, hole_w = rect_sizes[0]
    hole_coords = np.argwhere(hole_mask)
    for r, c in hole_coords:
        inp[hole_r0 + int(r), hole_c0 + int(c)] = BLACK

    for (r0, c0), (rh, rw) in zip(placements[1:], rect_sizes[1:]):
        inp[r0 : r0 + rh, c0 : c0 + rw] = GREY
        out[r0 : r0 + rh, c0 : c0 + rw] = GREY

    target_r0, target_c0 = placements[1]
    stamp = np.full((hole_h, hole_w), GREY, dtype=int)
    stamp[hole_mask] = BLACK
    out[target_r0 : target_r0 + hole_h, target_c0 : target_c0 + hole_w] = stamp
    return inp, out


def _validate_instance(
    inp: np.ndarray,
    out: np.ndarray,
    wallpaper: np.ndarray,
    period_shape: tuple[int, int],
    hole_mask: np.ndarray,
    rect_sizes: list[tuple[int, int]],
) -> bool:
    hole_components = _components(inp == BLACK)
    if len(hole_components) != 1:
        return False
    hole_coords = hole_components[0]
    hole_bbox = _bbox_coords(hole_coords)
    if _bbox_shape(hole_bbox) != hole_mask.shape:
        return False
    if not np.array_equal(_crop_bbox(inp == BLACK, hole_bbox), hole_mask):
        return False

    rects = _gray_rectangles(inp)
    if len(rects) != len(rect_sizes) - 1 or len(rects) < 2:
        return False
    target_shape = tuple(map(int, hole_mask.shape))
    matched = [rect for rect in rects if rect["shape"] == target_shape]
    if len(matched) != 1:
        return False

    try:
        inferred_shape = _minimal_period_shape(inp)
    except AssertionError:
        return False
    if inferred_shape != period_shape:
        return False
    if _minimal_period_shape(wallpaper) != period_shape:
        return False

    repaired_colors = {int(wallpaper[int(r), int(c)]) for r, c in hole_coords}
    if len(repaired_colors) < 2:
        return False
    row_residues = {int(r) % period_shape[0] for r, _c in hole_coords}
    col_residues = {int(c) % period_shape[1] for _r, c in hole_coords}
    if len(row_residues) < 2 or len(col_residues) < 2:
        return False

    distractors = rect_sizes[2:]
    if not any(
        h == target_shape[0] or w == target_shape[1] or h * w == target_shape[0] * target_shape[1]
        for h, w in distractors
    ):
        return False

    target_bbox = matched[0]["bbox"]
    tr0, tc0, tr1, tc1 = target_bbox
    target_patch = out[tr0 : tr1 + 1, tc0 : tc1 + 1]
    if not np.array_equal(target_patch == BLACK, hole_mask):
        return False
    if not np.any(target_patch == BLACK) or not np.any(target_patch == GREY):
        return False

    for rect in rects:
        if rect["bbox"] == target_bbox:
            continue
        r0, c0, r1, c1 = rect["bbox"]
        if not np.all(out[r0 : r1 + 1, c0 : c1 + 1] == GREY):
            return False

    expected = wallpaper.copy()
    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        expected[r0 : r1 + 1, c0 : c1 + 1] = GREY
    expected_target = expected[tr0 : tr1 + 1, tc0 : tc1 + 1]
    expected_target[hole_mask] = BLACK
    if not np.array_equal(out, expected):
        return False

    try:
        predicted = main(inp)
    except AssertionError:
        return False
    return np.array_equal(predicted, out)


def _generate_attempt() -> np.ndarray | None:
    period_shape = (random.randint(2, 5), random.randint(2, 5))
    palette = random.sample(WALLPAPER_COLORS, random.choice((3, 3, 4)))
    tile = _random_tile(period_shape, palette)
    if tile is None:
        return None

    hole_shape = random.choice(TARGET_DIMS)
    hole_mask = _hole_mask(*hole_shape)
    if hole_mask is None:
        return None

    gray_sizes = _choose_rectangles(hole_shape)
    rect_sizes = [hole_shape, *gray_sizes]
    max_h = max(size[0] for size in rect_sizes)
    max_w = max(size[1] for size in rect_sizes)
    grid_h = random.randint(max(16, max_h + 6), 28)
    grid_w = random.randint(max(16, max_w + 6), 28)
    if max(grid_h, grid_w) > 30:
        return None

    placements = _place_boxes((grid_h, grid_w), rect_sizes, border=1, gap=random.choice((1, 2)))
    if placements is None:
        return None

    offset = (random.randrange(period_shape[0]), random.randrange(period_shape[1]))
    wallpaper = _tile_fill((grid_h, grid_w), tile, offset)
    inp, out = _render_instance((grid_h, grid_w), tile, offset, hole_mask, placements, rect_sizes)

    if not _validate_instance(inp, out, wallpaper, period_shape, hole_mask, rect_sizes):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid wallpaper-transfer instance")
