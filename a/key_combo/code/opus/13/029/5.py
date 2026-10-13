from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
TARGET_COLORS = (3, 4, 6, 7)

GRID_ROWS = (17, 22)
GRID_COLS = (17, 22)
TARGET_COUNT_CHOICES = (3, 4)
SOURCE_SIZE = (5, 10)
SOURCE_BBOX_CHOICES = (3, 4)
GEN_RETRIES = 400


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _bbox(coords: list[Coord] | np.ndarray) -> BBox:
    arr = np.asarray(coords, dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _expand_bbox(box: BBox, pad: int) -> BBox:
    r0, c0, r1, c1 = box
    return r0 - pad, c0 - pad, r1 + pad, c1 + pad


def _boxes_intersect(a: BBox, b: BBox) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _translate_bbox(box: BBox, dr: int, dc: int) -> BBox:
    r0, c0, r1, c1 = box
    return r0 + dr, c0 + dc, r1 + dr, c1 + dc


def _components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[Coord] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _choose_anchor(g: np.ndarray) -> Coord:
    anchors = [(int(r), int(c)) for r, c in np.argwhere(g == RED)]
    assert anchors, "expected at least one red anchor candidate"

    scored: list[tuple[int, int, int]] = []
    for r, c in anchors:
        touch = sum(
            1
            for nr, nc in _neighbors4(r, c)
            if _in_bounds(g.shape, nr, nc) and int(g[nr, nc]) == BLUE
        )
        scored.append((touch, r, c))

    best_touch = max(touch for touch, _r, _c in scored)
    best = [(r, c) for touch, r, c in scored if touch == best_touch]
    return min(best)


def _extract_copy_specs(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    anchor = _choose_anchor(g)
    blue_coords = [(int(r), int(c)) for r, c in np.argwhere(g == BLUE)]
    assert blue_coords, "expected a blue source sprite"
    offsets = sorted((r - anchor[0], c - anchor[1]) for r, c in blue_coords)

    targets: list[tuple[int, int, int]] = []
    for color in sorted(set(map(int, np.unique(g))) - {bg, BLUE, RED}):
        coords = np.argwhere(g == color)
        for r, c in coords:
            targets.append((int(r), int(c), int(color)))
    targets.sort()
    assert 3 <= len(targets) <= 4, "expected 3-4 target controls"
    return bg, offsets, targets


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the blue source sprite at every control pixel, then grow a same-color halo around each stamped copy.

    Input structure:
    - A black background contains one irregular blue source sprite and one red anchor pixel adjacent to it.
    - Three or four isolated target control pixels use distinct colors chosen from green, yellow, pink, and orange.
    - The target pixels are spaced so the copied blue stamps fit on the grid and do not overlap each other.

    Rule:
    - Pick the red pixel whose 4-neighborhood touches the most blue pixels as the anchor.
    - Record all blue source coordinates relative to that anchor, erase the original scene, and stamp those blue offsets at every target control pixel.
    - Keep each target pixel's own color embedded inside its stamped copy, count how many pixels of that color appear inside that copy, and expand the copy's bounding box by that count on all four sides.
    - Fill each expanded rectangle with the target color, but immediately restore that copy's blue pixels and embedded marker; later halos overwrite earlier ones if the expanded rectangles overlap.

    Key insight:
    - The control pixel serves two roles at once: it chooses where the source sprite is copied, and its embedded color becomes the parameter that determines the halo color and thickness.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, offsets, targets = _extract_copy_specs(g)
    out = np.full_like(g, bg)
    copies: list[tuple[int, int, int, BBox, int, dict[Coord, int]]] = []

    for tr, tc, color in targets:
        body: dict[Coord, int] = {}
        for dr, dc in offsets:
            rr = tr + dr
            cc = tc + dc
            assert _in_bounds(g.shape, rr, cc), "stamped sprite must stay in bounds"
            body[(rr, cc)] = BLUE
        body[(tr, tc)] = color

        coords = list(body.keys())
        box = _bbox(coords)
        marker_count = sum(1 for value in body.values() if value == color)
        assert marker_count >= 1
        copies.append((tr, tc, color, box, marker_count, body))

        for (rr, cc), value in body.items():
            out[rr, cc] = value

    for tr, tc, color, box, marker_count, body in copies:
        er0, ec0, er1, ec1 = _expand_bbox(box, marker_count)
        assert 0 <= er0 <= er1 < g.shape[0]
        assert 0 <= ec0 <= ec1 < g.shape[1]
        out[er0 : er1 + 1, ec0 : ec1 + 1] = color
        for (rr, cc), value in body.items():
            out[rr, cc] = value

    return out


def _random_connected_shape() -> tuple[list[Coord], Coord] | None:
    for _ in range(200):
        h = random.choice(SOURCE_BBOX_CHOICES)
        w = random.choice(SOURCE_BBOX_CHOICES)
        count = random.randint(SOURCE_SIZE[0], min(SOURCE_SIZE[1], h * w))

        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        while len(cells) < count:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))
        if len(cells) != count:
            continue

        box = _bbox(list(cells))
        bh = box[2] - box[0] + 1
        bw = box[3] - box[1] + 1
        if bh > 4 or bw > 4:
            continue
        if len(cells) == bh * bw:
            continue

        inner_candidates: list[tuple[int, Coord]] = []
        outer_candidates: list[tuple[int, Coord]] = []
        for ar in range(-1, h + 1):
            for ac in range(-1, w + 1):
                if (ar, ac) in cells:
                    continue
                touch = sum((nr, nc) in cells for nr, nc in _neighbors4(ar, ac))
                if touch < 2:
                    continue
                bucket = inner_candidates if 0 <= ar < h and 0 <= ac < w else outer_candidates
                bucket.append((touch, (ar, ac)))

        if not inner_candidates and not outer_candidates:
            continue

        use_inner = inner_candidates and (not outer_candidates or random.random() < 0.5)
        pool = inner_candidates if use_inner else outer_candidates
        best_touch = max(touch for touch, _pos in pool)
        anchors = [pos for touch, pos in pool if touch == best_touch]
        anchor = random.choice(anchors)

        offsets = [(r - anchor[0], c - anchor[1]) for r, c in cells]
        offsets = _random_transform(offsets)
        return sorted(offsets), (0, 0)
    return None


def _random_transform(offsets: list[Coord]) -> list[Coord]:
    def tf(coord: Coord, mode: int) -> Coord:
        r, c = coord
        if mode == 0:
            return r, c
        if mode == 1:
            return r, -c
        if mode == 2:
            return -r, c
        if mode == 3:
            return -r, -c
        if mode == 4:
            return c, r
        if mode == 5:
            return c, -r
        if mode == 6:
            return -c, r
        return -c, -r

    mode = random.randrange(8)
    pts = [tf(coord, mode) for coord in offsets]
    return pts


def _copy_body(offsets: list[Coord]) -> dict[Coord, int]:
    body = {coord: BLUE for coord in offsets}
    body[(0, 0)] = -1
    return body


def _target_candidates(shape: tuple[int, int], rel_body: list[Coord]) -> list[Coord]:
    box = _bbox(rel_body)
    r0, c0, r1, c1 = box
    h, w = shape
    rows = range(3 - r0, h - 3 - r1)
    cols = range(3 - c0, w - 3 - c1)
    return [(r, c) for r in rows for c in cols]


def _place_targets(shape: tuple[int, int], offsets: list[Coord], count: int) -> list[tuple[int, int]] | None:
    rel_body = list(_copy_body(offsets).keys())
    rel_box = _bbox(rel_body)
    rel_exp = _expand_bbox(rel_box, 1)
    candidates = _target_candidates(shape, rel_body)
    random.shuffle(candidates)
    want_overlap = random.random() < 0.35

    best: list[tuple[int, int]] | None = None
    for _ in range(120):
        placed: list[tuple[int, int]] = []
        body_boxes: list[BBox] = []
        exp_boxes: list[BBox] = []
        pool = candidates[:]
        random.shuffle(pool)
        for tr, tc in pool:
            body_box = _translate_bbox(rel_box, tr, tc)
            exp_box = _translate_bbox(rel_exp, tr, tc)
            ok = True
            for other_body, other_exp in zip(body_boxes, exp_boxes):
                if _boxes_intersect(body_box, other_body):
                    ok = False
                    break
                if _boxes_intersect(exp_box, other_body) or _boxes_intersect(other_exp, body_box):
                    ok = False
                    break
            if not ok:
                continue
            placed.append((tr, tc))
            body_boxes.append(body_box)
            exp_boxes.append(exp_box)
            if len(placed) == count:
                break
        if len(placed) != count:
            continue

        overlap_pairs = 0
        for i in range(len(exp_boxes)):
            for j in range(i + 1, len(exp_boxes)):
                if _boxes_intersect(exp_boxes[i], exp_boxes[j]):
                    overlap_pairs += 1
        if want_overlap and overlap_pairs == 0:
            continue
        best = placed
        break
    return best


def _place_source(
    shape: tuple[int, int],
    offsets: list[Coord],
    target_positions: list[Coord],
) -> tuple[Coord, list[Coord]] | None:
    rel_source = list(offsets) + [(0, 0)]
    src_box = _bbox(rel_source)
    target_boxes = []
    rel_body = list(_copy_body(offsets).keys())
    rel_exp = _expand_bbox(_bbox(rel_body), 1)
    for tr, tc in target_positions:
        target_boxes.append(_translate_bbox(rel_exp, tr, tc))

    h, w = shape
    r0, c0, r1, c1 = src_box
    candidates = [
        (ar, ac)
        for ar in range(-r0, h - r1)
        for ac in range(-c0, w - c1)
    ]
    random.shuffle(candidates)
    for ar, ac in candidates:
        abs_box = _translate_bbox(src_box, ar, ac)
        padded = _expand_bbox(abs_box, 1)
        if padded[0] < 0 or padded[1] < 0 or padded[2] >= h or padded[3] >= w:
            continue
        if any(_boxes_intersect(padded, box) for box in target_boxes):
            continue
        blue = [(ar + dr, ac + dc) for dr, dc in offsets]
        if any((ar, ac) == pos for pos in target_positions):
            continue
        if any(cell in target_positions for cell in blue):
            continue
        return (ar, ac), blue
    return None


def _build_input(
    shape: tuple[int, int],
    offsets: list[Coord],
    targets: list[tuple[int, int, int]],
    anchor: Coord,
    blue: list[Coord],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for r, c in blue:
        grid[r, c] = BLUE
    grid[anchor] = RED
    for r, c, color in targets:
        grid[r, c] = color
    return grid


def _valid_source(grid: np.ndarray, anchor: Coord) -> bool:
    blue_mask = grid == BLUE
    comps = _components(blue_mask)
    if len(comps) != 1:
        return False
    if len(comps[0]) < SOURCE_SIZE[0] or len(comps[0]) > SOURCE_SIZE[1]:
        return False
    box = _bbox(comps[0])
    if box[2] - box[0] + 1 > 4 or box[3] - box[1] + 1 > 4:
        return False
    touch = sum(
        1
        for nr, nc in _neighbors4(*anchor)
        if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == BLUE
    )
    return touch >= 2


def _attempt() -> np.ndarray | None:
    shape = (random.randint(*GRID_ROWS), random.randint(*GRID_COLS))
    source = _random_connected_shape()
    if source is None:
        return None
    offsets, _origin = source

    target_count = random.choice(TARGET_COUNT_CHOICES)
    positions = _place_targets(shape, offsets, target_count)
    if positions is None:
        return None

    colors = random.sample(TARGET_COLORS, target_count)
    targets = sorted((r, c, color) for (r, c), color in zip(positions, colors))

    placement = _place_source(shape, offsets, [(r, c) for r, c, _color in targets])
    if placement is None:
        return None
    anchor, blue = placement

    grid = _build_input(shape, offsets, targets, anchor, blue)
    if not _valid_source(grid, anchor):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    non_bg_out = set(map(int, np.unique(out))) - {BLACK}
    if len(non_bg_out) < 3:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
