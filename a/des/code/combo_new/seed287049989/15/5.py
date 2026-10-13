import random
from collections import defaultdict, deque
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


BLACK = 0
GREY = 5
RING_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
SIDES = ("top", "bottom", "left", "right")
DIR_BY_SIDE = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
SIDE_ORDER = {side: idx for idx, side in enumerate(SIDES)}

GRID_H_RANGE = (20, 30)
GRID_W_RANGE = (20, 30)
N_SQUARE_CHOICES = [1, 2, 2, 3, 3, 4]
TARGET_RING_RETRIES = 300
MISS_RING_RETRIES = 200
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> List[List[Tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: List[List[Tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: List[Tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: Sequence[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_size(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return r1 - r0 + 1, c1 - c0 + 1


def _bboxes_too_close(
    a: Tuple[int, int, int, int], b: Tuple[int, int, int, int], margin: int = 1
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 < br0 - margin
        or br1 < ar0 - margin
        or ac1 < bc0 - margin
        or bc1 < ac0 - margin
    )


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _ring_cells(ring: Dict[str, object]) -> List[Tuple[int, int]]:
    r0, c0, r1, c1 = ring["bbox"]  # type: ignore[index]
    side = ring["side"]  # type: ignore[index]
    gap_start = ring["gap_start"]  # type: ignore[index]
    cells: List[Tuple[int, int]] = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if r not in (r0, r1) and c not in (c0, c1):
                continue
            if side in ("top", "bottom"):
                gap_row = r0 if side == "top" else r1
                if r == gap_row and gap_start <= c - c0 <= gap_start + 1:
                    continue
            else:
                gap_col = c0 if side == "left" else c1
                if c == gap_col and gap_start <= r - r0 <= gap_start + 1:
                    continue
            cells.append((r, c))
    return cells


def _full_ring_border(h: int, w: int) -> np.ndarray:
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return border


def _identify_ring(submask: np.ndarray) -> Optional[Tuple[str, int]]:
    h, w = submask.shape
    if not (5 <= h <= 7 and 5 <= w <= 7):
        return None
    border = _full_ring_border(h, w)
    if np.any(submask & ~border):
        return None
    missing = border & ~submask
    if int(missing.sum()) != 2:
        return None

    miss = np.argwhere(missing)
    rows = miss[:, 0]
    cols = miss[:, 1]
    if np.all(rows == 0):
        side = "top"
        values = sorted(int(c) for c in cols)
    elif np.all(rows == h - 1):
        side = "bottom"
        values = sorted(int(c) for c in cols)
    elif np.all(cols == 0):
        side = "left"
        values = sorted(int(r) for r in rows)
    elif np.all(cols == w - 1):
        side = "right"
        values = sorted(int(r) for r in rows)
    else:
        return None

    if values[1] != values[0] + 1:
        return None
    return side, values[0]


def _parse_squares(grid: np.ndarray) -> List[Dict[str, object]]:
    squares: List[Dict[str, object]] = []
    for coords in _components(grid == GREY):
        bbox = _bbox(coords)
        h, w = _bbox_size(bbox)
        if h != w or not (3 <= h <= 5) or len(coords) != h * w:
            raise AssertionError("grey components must be solid 3x3 to 5x5 squares")
        r0, c0, r1, c1 = bbox
        if not np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == GREY):
            raise AssertionError("grey square must be solid")
        squares.append({"bbox": bbox, "size": h})
    squares.sort(key=lambda sq: sq["bbox"])  # type: ignore[index]
    return squares


def _parse_rings(grid: np.ndarray) -> List[Dict[str, object]]:
    mask = (grid != BLACK) & (grid != GREY)
    rings: List[Dict[str, object]] = []
    seen_colors = set()
    for coords in _components(mask):
        bbox = _bbox(coords)
        rows, cols = zip(*coords)
        colors = np.unique(grid[np.array(rows), np.array(cols)])
        if colors.size != 1:
            raise AssertionError("ring component must be monochrome")
        color = int(colors[0])
        r0, c0, r1, c1 = bbox
        submask = grid[r0 : r1 + 1, c0 : c1 + 1] == color
        ring_info = _identify_ring(submask)
        if ring_info is None:
            raise AssertionError("non-grey component must be a gapped rectangular ring")
        side, gap_start = ring_info
        if color in seen_colors:
            raise AssertionError("ring colors must be distinct")
        seen_colors.add(color)
        rings.append(
            {
                "bbox": bbox,
                "color": color,
                "side": side,
                "gap_start": gap_start,
            }
        )
    rings.sort(
        key=lambda ring: (
            ring["bbox"][0],  # type: ignore[index]
            ring["bbox"][1],  # type: ignore[index]
            SIDE_ORDER[ring["side"]],  # type: ignore[index]
        )
    )
    return rings


def _ring_gap_interval(ring: Dict[str, object]) -> Tuple[int, int]:
    r0, c0, _r1, _c1 = ring["bbox"]  # type: ignore[index]
    gap_start = ring["gap_start"]  # type: ignore[index]
    if ring["side"] in ("top", "bottom"):  # type: ignore[index]
        return c0 + gap_start, c0 + gap_start + 1
    return r0 + gap_start, r0 + gap_start + 1


def _first_target_for_ring(
    ring: Dict[str, object], squares: Sequence[Dict[str, object]]
) -> Tuple[Optional[int], Optional[int]]:
    r0, c0, r1, c1 = ring["bbox"]  # type: ignore[index]
    side = ring["side"]  # type: ignore[index]
    ray0, ray1 = _ring_gap_interval(ring)

    best: Optional[Tuple[int, int, int, int, int]] = None
    for idx, square in enumerate(squares):
        sr0, sc0, sr1, sc1 = square["bbox"]  # type: ignore[index]
        if side == "top":
            if sr1 >= r0 or not _intervals_overlap(ray0, ray1, sc0, sc1):
                continue
            dist = r0 - sr1 - 1
        elif side == "bottom":
            if sr0 <= r1 or not _intervals_overlap(ray0, ray1, sc0, sc1):
                continue
            dist = sr0 - r1 - 1
        elif side == "left":
            if sc1 >= c0 or not _intervals_overlap(ray0, ray1, sr0, sr1):
                continue
            dist = c0 - sc1 - 1
        else:
            if sc0 <= c1 or not _intervals_overlap(ray0, ray1, sr0, sr1):
                continue
            dist = sc0 - c1 - 1

        candidate = (dist, idx, sr0, sc0, square["size"])  # type: ignore[index]
        if best is None or candidate < best:
            best = candidate

    if best is None:
        return None, None
    return best[1], best[0]


def _paint_square_layer(
    out: np.ndarray, bbox: Tuple[int, int, int, int], depth: int, color: int
) -> bool:
    r0, c0, r1, c1 = bbox
    r0 += depth
    c0 += depth
    r1 -= depth
    c1 -= depth
    if r0 > r1 or c0 > c1:
        return False
    out[r0, c0 : c1 + 1] = color
    out[r1, c0 : c1 + 1] = color
    out[r0 : r1 + 1, c0] = color
    out[r0 : r1 + 1, c1] = color
    return True


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    squares = _parse_squares(g)
    rings = _parse_rings(g)

    out = g.copy()
    for ring in rings:
        rr, cc = zip(*_ring_cells(ring))
        out[np.array(rr), np.array(cc)] = BLACK

    by_target: Dict[int, List[Tuple[int, Dict[str, object]]]] = defaultdict(list)
    for ring in rings:
        target_idx, dist = _first_target_for_ring(ring, squares)
        if target_idx is not None and dist is not None:
            by_target[target_idx].append((dist, ring))

    for target_idx, assigned in by_target.items():
        assigned.sort(
            key=lambda item: (
                item[0],
                item[1]["bbox"][0],  # type: ignore[index]
                item[1]["bbox"][1],  # type: ignore[index]
                SIDE_ORDER[item[1]["side"]],  # type: ignore[index]
            )
        )
        bbox = squares[target_idx]["bbox"]  # type: ignore[index]
        for depth, (_dist, ring) in enumerate(assigned):
            if not _paint_square_layer(out, bbox, depth, ring["color"]):  # type: ignore[index]
                break

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Clear every colored gapped ring and transfer its color into the grey square it points toward.

    Input structure:
    - The background is black.
    - Foreground objects are distinct-colored one-pixel-thick rectangular rings with a 2-cell gap on exactly one side, plus solid grey squares of size 3x3 to 5x5.
    - Rings and grey squares are separated from each other and do not overlap.

    Rule:
    - Interpret each ring's gap side as a firing direction and scan straight outward from the gap corridor to the first grey square on that line.
    - For every grey square, sort all pointing rings by gap-to-square distance from near to far.
    - Recolor the square from the outside inward: the nearest ring colors the outer layer, the next ring colors the next inner layer, extra rings are ignored once no inner layer remains, and all rings are erased to black.

    Key insight:
    - The gap is not damage to repair; it is a directional emitter that chooses a target square and an ordering for layered color transfer.
    """
    return _solve(grid)


def _layer_capacity(size: int) -> int:
    return (size + 1) // 2


def _render_square(grid: np.ndarray, square: Dict[str, object]) -> None:
    r0, c0, r1, c1 = square["bbox"]  # type: ignore[index]
    grid[r0 : r1 + 1, c0 : c1 + 1] = GREY


def _render_ring(grid: np.ndarray, ring: Dict[str, object]) -> None:
    for r, c in _ring_cells(ring):
        grid[r, c] = ring["color"]  # type: ignore[index]


def _plan_counts(
    rng: random.Random, squares: Sequence[Dict[str, object]]
) -> Tuple[List[int], int, bool]:
    n = len(squares)
    has_idle_square = n > 1 and rng.random() < 0.5
    has_multi = rng.random() < 0.7
    has_overflow = has_multi and rng.random() < 0.4
    has_miss_ring = rng.random() < 0.5

    max_active = n - 1 if has_idle_square else n
    active_count = rng.randint(1, max_active)
    active = rng.sample(range(n), active_count)
    counts = [0] * n
    for idx in active:
        counts[idx] = 1

    if has_multi:
        multi_idx = rng.choice(active)
        cap = _layer_capacity(squares[multi_idx]["size"])  # type: ignore[index]
        if has_overflow:
            target_count = cap + 1
        else:
            target_count = rng.randint(2, cap)
        counts[multi_idx] = max(counts[multi_idx], target_count)

    total_targeted = sum(counts)
    miss_count = 0
    if has_miss_ring:
        miss_count = rng.randint(1, 2)

    while total_targeted + miss_count > len(RING_COLORS):
        if miss_count > 0:
            miss_count -= 1
        else:
            richest = max(range(n), key=lambda idx: counts[idx])
            if counts[richest] > 1:
                counts[richest] -= 1
                total_targeted -= 1
            else:
                break

    return counts, miss_count, has_overflow


def _place_squares(
    rng: random.Random, grid_shape: Tuple[int, int]
) -> Optional[List[Dict[str, object]]]:
    h, w = grid_shape
    n_squares = rng.choice(N_SQUARE_CHOICES)
    squares: List[Dict[str, object]] = []
    for _ in range(n_squares):
        placed = False
        for _ in range(120):
            size = rng.randint(3, 5)
            r0 = rng.randint(0, h - size)
            c0 = rng.randint(0, w - size)
            bbox = (r0, c0, r0 + size - 1, c0 + size - 1)
            if any(_bboxes_too_close(bbox, sq["bbox"]) for sq in squares):  # type: ignore[index]
                continue
            squares.append({"bbox": bbox, "size": size})
            placed = True
            break
        if not placed:
            return None
    squares.sort(key=lambda sq: sq["bbox"])  # type: ignore[index]
    return squares


def _make_target_ring(
    side: str,
    bbox: Tuple[int, int, int, int],
    gap_start: int,
    color: int,
    target_idx: int,
    distance: int,
) -> Dict[str, object]:
    return {
        "side": side,
        "bbox": bbox,
        "gap_start": gap_start,
        "color": color,
        "target_idx": target_idx,
        "distance": distance,
    }


def _target_ring_candidates(
    square_idx: int,
    squares: Sequence[Dict[str, object]],
    occupied: Sequence[Tuple[int, int, int, int]],
    grid_shape: Tuple[int, int],
    used_distances: Sequence[int],
    color: int,
) -> List[Dict[str, object]]:
    h, w = grid_shape
    sr0, sc0, sr1, sc1 = squares[square_idx]["bbox"]  # type: ignore[index]
    used = set(used_distances)
    cands: List[Dict[str, object]] = []

    for side in SIDES:
        for ring_h in range(5, 8):
            for ring_w in range(5, 8):
                if side in ("top", "bottom"):
                    gap_positions = range(1, ring_w - 2)
                    align_values = range(sc0, sc1)
                else:
                    gap_positions = range(1, ring_h - 2)
                    align_values = range(sr0, sr1)

                for gap_start in gap_positions:
                    for align in align_values:
                        if side == "top":
                            c0 = align - gap_start
                            max_dist = h - ring_h - sr1 - 1
                            if c0 < 0 or c0 + ring_w > w or max_dist < 1:
                                continue
                            distances = range(1, max_dist + 1)
                        elif side == "bottom":
                            c0 = align - gap_start
                            max_dist = sr0 - ring_h
                            if c0 < 0 or c0 + ring_w > w or max_dist < 1:
                                continue
                            distances = range(1, max_dist + 1)
                        elif side == "left":
                            r0 = align - gap_start
                            max_dist = w - ring_w - sc1 - 1
                            if r0 < 0 or r0 + ring_h > h or max_dist < 1:
                                continue
                            distances = range(1, max_dist + 1)
                        else:
                            r0 = align - gap_start
                            max_dist = sc0 - ring_w
                            if r0 < 0 or r0 + ring_h > h or max_dist < 1:
                                continue
                            distances = range(1, max_dist + 1)

                        for distance in distances:
                            if distance in used:
                                continue
                            if side == "top":
                                r0 = sr1 + 1 + distance
                                bbox = (r0, c0, r0 + ring_h - 1, c0 + ring_w - 1)
                            elif side == "bottom":
                                r0 = sr0 - distance - ring_h
                                bbox = (r0, c0, r0 + ring_h - 1, c0 + ring_w - 1)
                            elif side == "left":
                                c0 = sc1 + 1 + distance
                                bbox = (r0, c0, r0 + ring_h - 1, c0 + ring_w - 1)
                            else:
                                c0 = sc0 - distance - ring_w
                                bbox = (r0, c0, r0 + ring_h - 1, c0 + ring_w - 1)

                            if any(_bboxes_too_close(bbox, other) for other in occupied):
                                continue
                            ring = _make_target_ring(side, bbox, gap_start, color, square_idx, distance)
                            actual_target, actual_dist = _first_target_for_ring(ring, squares)
                            if actual_target == square_idx and actual_dist == distance:
                                cands.append(ring)
    return cands


def _place_target_rings(
    rng: random.Random,
    squares: Sequence[Dict[str, object]],
    counts: Sequence[int],
    colors: Sequence[int],
    grid_shape: Tuple[int, int],
) -> Optional[List[Dict[str, object]]]:
    occupied = [sq["bbox"] for sq in squares]  # type: ignore[index]
    rings: List[Dict[str, object]] = []
    color_iter = iter(colors)

    order = sorted([idx for idx, count in enumerate(counts) if count > 0], key=lambda idx: -counts[idx])
    for square_idx in order:
        used_distances: List[int] = []
        for _ in range(counts[square_idx]):
            color = next(color_iter)
            cands = _target_ring_candidates(
                square_idx, squares, occupied, grid_shape, used_distances, color
            )
            if not cands:
                return None
            ring = rng.choice(cands)
            rings.append(ring)
            occupied.append(ring["bbox"])  # type: ignore[arg-type]
            used_distances.append(ring["distance"])  # type: ignore[arg-type]
    return rings


def _place_miss_rings(
    rng: random.Random,
    squares: Sequence[Dict[str, object]],
    occupied: List[Tuple[int, int, int, int]],
    colors: Sequence[int],
    grid_shape: Tuple[int, int],
) -> Optional[List[Dict[str, object]]]:
    h, w = grid_shape
    rings: List[Dict[str, object]] = []
    for color in colors:
        placed = False
        for _ in range(MISS_RING_RETRIES):
            side = rng.choice(SIDES)
            ring_h = rng.randint(5, 7)
            ring_w = rng.randint(5, 7)
            if side in ("top", "bottom"):
                gap_start = rng.randint(1, ring_w - 3)
                c0 = rng.randint(0, w - ring_w)
                r0 = 0 if side == "top" else h - ring_h
            else:
                gap_start = rng.randint(1, ring_h - 3)
                r0 = rng.randint(0, h - ring_h)
                c0 = 0 if side == "left" else w - ring_w
            bbox = (r0, c0, r0 + ring_h - 1, c0 + ring_w - 1)
            if any(_bboxes_too_close(bbox, other) for other in occupied):
                continue
            ring = {
                "side": side,
                "bbox": bbox,
                "gap_start": gap_start,
                "color": color,
                "target_idx": None,
                "distance": None,
            }
            target_idx, _dist = _first_target_for_ring(ring, squares)
            if target_idx is not None:
                continue
            rings.append(ring)
            occupied.append(bbox)
            placed = True
            break
        if not placed:
            return None
    return rings


def _expected_output(
    grid: np.ndarray, squares: Sequence[Dict[str, object]], rings: Sequence[Dict[str, object]]
) -> np.ndarray:
    out = grid.copy()
    for ring in rings:
        for r, c in _ring_cells(ring):
            out[r, c] = BLACK

    by_target: Dict[int, List[Tuple[int, Dict[str, object]]]] = defaultdict(list)
    for ring in rings:
        target_idx = ring["target_idx"]  # type: ignore[index]
        distance = ring["distance"]  # type: ignore[index]
        if target_idx is None or distance is None:
            continue
        by_target[target_idx].append((distance, ring))

    for target_idx, assigned in by_target.items():
        assigned.sort(
            key=lambda item: (
                item[0],
                item[1]["bbox"][0],  # type: ignore[index]
                item[1]["bbox"][1],  # type: ignore[index]
                SIDE_ORDER[item[1]["side"]],  # type: ignore[index]
            )
        )
        bbox = squares[target_idx]["bbox"]  # type: ignore[index]
        for depth, (_distance, ring) in enumerate(assigned):
            if not _paint_square_layer(out, bbox, depth, ring["color"]):  # type: ignore[index]
                break
    return out


def _gen_attempt() -> Optional[np.ndarray]:
    rng = random.Random()
    grid_h = rng.randint(*GRID_H_RANGE)
    grid_w = rng.randint(*GRID_W_RANGE)
    squares = _place_squares(rng, (grid_h, grid_w))
    if squares is None:
        return None

    counts, miss_count, has_overflow = _plan_counts(rng, squares)
    total_rings = sum(counts) + miss_count
    if total_rings <= 0 or total_rings > len(RING_COLORS):
        return None

    colors = RING_COLORS[:]
    rng.shuffle(colors)
    target_colors = colors[: sum(counts)]
    miss_colors = colors[sum(counts) : total_rings]

    target_rings = _place_target_rings(rng, squares, counts, target_colors, (grid_h, grid_w))
    if target_rings is None:
        return None

    occupied = [sq["bbox"] for sq in squares] + [ring["bbox"] for ring in target_rings]  # type: ignore[index]
    miss_rings = _place_miss_rings(rng, squares, occupied, miss_colors, (grid_h, grid_w))
    if miss_rings is None:
        return None

    all_rings = target_rings + miss_rings
    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    for square in squares:
        _render_square(grid, square)
    for ring in all_rings:
        _render_ring(grid, ring)

    expected = _expected_output(grid, squares, all_rings)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    if has_overflow:
        overflow_ok = any(
            counts[idx] > _layer_capacity(square["size"]) for idx, square in enumerate(squares)  # type: ignore[index]
        )
        if not overflow_ok:
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
