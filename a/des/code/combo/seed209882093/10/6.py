from __future__ import annotations

import random

import numpy as np


AZURE = 8
GREY = 5
PALETTE = [0, 1, 2, 3, 4, 6, 7, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_HEIGHT_RANGE = (18, 28)
GRID_WIDTH_RANGE = (18, 28)
BIG_OBJECT_COUNT_RANGE = (2, 4)
SMALL_OBJECT_COUNT_RANGE = (4, 7)
OBJECT_GAP = 1
MAX_GENERATION_ATTEMPTS = 400


def _in_bounds(grid: np.ndarray, r: int, c: int) -> bool:
    return 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]


def _cross_pixels(r: int, c: int, length: int) -> list[tuple[int, int]]:
    pixels = [(r, c)]
    for step in range(1, length + 1):
        pixels.extend(
            [
                (r - step, c),
                (r + step, c),
                (r, c - step),
                (r, c + step),
            ]
        )
    return pixels


def _run_length(grid: np.ndarray, r: int, c: int, dr: int, dc: int, color: int) -> int:
    steps = 0
    rr, cc = r + dr, c + dc
    while _in_bounds(grid, rr, cc) and int(grid[rr, cc]) == int(color):
        steps += 1
        rr += dr
        cc += dc
    return steps


def _find_crosses(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    crosses: list[dict] = []
    for r, c in np.argwhere(g == GREY):
        r = int(r)
        c = int(c)
        neighbors = []
        ok = True
        for dr, dc in DIR4:
            rr, cc = r + dr, c + dc
            if not _in_bounds(g, rr, cc):
                ok = False
                break
            neighbors.append(int(g[rr, cc]))
        if not ok or len(set(neighbors)) != 1:
            continue

        color = int(neighbors[0])
        if color in (AZURE, GREY):
            continue

        lengths = [_run_length(g, r, c, dr, dc, color) for dr, dc in DIR4]
        if len(set(lengths)) != 1:
            continue

        length = int(lengths[0])
        if not 2 <= length <= 4:
            continue

        exact = True
        for dr, dc in DIR4:
            rr = r + dr * (length + 1)
            cc = c + dc * (length + 1)
            if _in_bounds(g, rr, cc) and int(g[rr, cc]) == color:
                exact = False
                break
        if not exact:
            continue

        pixels = _cross_pixels(r, c, length)
        crosses.append(
            {
                "center": (r, c),
                "color": color,
                "length": length,
                "pixels": pixels,
                "pixel_set": frozenset(pixels),
                "square": (r - length, c - length, r + length, c + length),
            }
        )
    return crosses


def _cross_sort_key(cross: dict) -> tuple[int, int]:
    r, c = cross["center"]
    return int(c), int(r)


def _apply_crosses(grid: np.ndarray, crosses: list[dict]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for cross in sorted(crosses, key=_cross_sort_key):
        r0, c0, r1, c1 = cross["square"]
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[(patch != AZURE) & (patch != GREY)] = int(cross["color"])
        for rr, cc in cross["pixels"]:
            out[rr, cc] = AZURE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each cross-centered square region, then erase the crosses."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    crosses = _find_crosses(g)
    assert 2 <= len(crosses) <= 4
    return _apply_crosses(g, crosses)


def _normalize(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return sorted((r - min_r, c - min_c) for r, c in cells)


def _flip_cells(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    shape = set(cells)
    if random.random() < 0.5:
        max_c = max(c for _, c in shape)
        shape = {(r, max_c - c) for r, c in shape}
    if random.random() < 0.5:
        max_r = max(r for r, _ in shape)
        shape = {(max_r - r, c) for r, c in shape}
    if random.random() < 0.35:
        shape = {(c, r) for r, c in shape}
    return _normalize(shape)


def _random_connected_shape(area: int, height: int, width: int) -> list[tuple[int, int]] | None:
    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    frontier = set()
    for dr, dc in DIR4:
        rr, cc = start[0] + dr, start[1] + dc
        if 0 <= rr < height and 0 <= cc < width:
            frontier.add((rr, cc))
    while len(cells) < area and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        if cell in cells:
            continue
        cells.add(cell)
        for dr, dc in DIR4:
            rr, cc = cell[0] + dr, cell[1] + dc
            if 0 <= rr < height and 0 <= cc < width and (rr, cc) not in cells:
                frontier.add((rr, cc))
    if len(cells) != area:
        return None
    return _normalize(cells)


def _random_big_object() -> list[tuple[int, int]]:
    mode = random.choice(["rect", "dense", "notched"])
    if mode == "rect":
        h = random.randint(2, 4)
        w = random.randint(3, 6)
        return [(r, c) for r in range(h) for c in range(w)]

    if mode == "notched":
        h = random.randint(3, 5)
        w = random.randint(3, 6)
        cells = {(r, c) for r in range(h) for c in range(w)}
        corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
        cells.remove(random.choice(corners))
        return _flip_cells(_normalize(cells))

    h = random.randint(3, 5)
    w = random.randint(3, 6)
    lo = max(6, (h * w) // 2)
    hi = max(lo, h * w - 1)
    for _ in range(30):
        area = random.randint(lo, hi)
        cells = _random_connected_shape(area, h, w)
        if cells is not None:
            return _flip_cells(cells)
    return [(r, c) for r in range(3) for c in range(3)]


def _random_small_object() -> list[tuple[int, int]]:
    mode = random.choice(["line_h", "line_v", "rect", "el", "tee", "poly"])
    if mode == "line_h":
        length = random.randint(2, 5)
        return [(0, c) for c in range(length)]
    if mode == "line_v":
        length = random.randint(2, 5)
        return [(r, 0) for r in range(length)]
    if mode == "rect":
        h = random.randint(2, 3)
        w = random.randint(2, 4)
        return _flip_cells([(r, c) for r in range(h) for c in range(w)])
    if mode == "el":
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        cells = {(r, 0) for r in range(h)} | {(h - 1, c) for c in range(w)}
        return _flip_cells(_normalize(cells))
    if mode == "tee":
        w = random.randint(3, 5)
        mid = w // 2
        cells = {(0, c) for c in range(w)} | {(1, mid), (2, mid)}
        return _flip_cells(_normalize(cells))

    h = random.randint(2, 4)
    w = random.randint(2, 5)
    lo = max(3, (h * w) // 2)
    hi = max(lo, min(h * w, 7))
    for _ in range(20):
        area = random.randint(lo, hi)
        cells = _random_connected_shape(area, h, w)
        if cells is not None:
            return _flip_cells(cells)
    return _flip_cells([(0, 0), (0, 1), (1, 0)])


def _bbox_of(cells: list[tuple[int, int]]) -> tuple[int, int]:
    return max(r for r, _ in cells) + 1, max(c for _, c in cells) + 1


def _place_object(
    grid: np.ndarray,
    occupied: np.ndarray,
    cells: list[tuple[int, int]],
    color: int,
) -> bool:
    height, width = _bbox_of(cells)
    H, W = grid.shape
    positions = []
    for r0 in range(0, H - height + 1):
        for c0 in range(0, W - width + 1):
            rr0 = max(0, r0 - OBJECT_GAP)
            cc0 = max(0, c0 - OBJECT_GAP)
            rr1 = min(H, r0 + height + OBJECT_GAP)
            cc1 = min(W, c0 + width + OBJECT_GAP)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            positions.append((r0, c0))
    if not positions:
        return False

    r0, c0 = random.choice(positions)
    for dr, dc in cells:
        grid[r0 + dr, c0 + dc] = color

    rr0 = max(0, r0 - OBJECT_GAP)
    cc0 = max(0, c0 - OBJECT_GAP)
    rr1 = min(H, r0 + height + OBJECT_GAP)
    cc1 = min(W, c0 + width + OBJECT_GAP)
    occupied[rr0:rr1, cc0:cc1] = True
    return True


def _overlap_bbox(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return None
    return r0, c0, r1, c1


def _enumerate_cross_candidates(
    base: np.ndarray,
    cross_occupied: np.ndarray,
    placed_crosses: list[dict],
    length: int,
) -> list[dict]:
    H, W = base.shape
    object_mask = base != AZURE
    candidates = []
    for r in range(length, H - length):
        for c in range(length, W - length):
            pixels = _cross_pixels(r, c, length)
            if any(cross_occupied[rr, cc] for rr, cc in pixels):
                continue

            pixel_set = frozenset(pixels)
            square = (r - length, c - length, r + length, c + length)
            target_count = 0
            arm_hit = False
            for rr in range(square[0], square[2] + 1):
                for cc in range(square[1], square[3] + 1):
                    if not object_mask[rr, cc]:
                        continue
                    if (rr, cc) in pixel_set:
                        arm_hit = True
                        continue
                    target_count += 1

            overlap_pairs = 0
            overlap_targets = 0
            for prev in placed_crosses:
                shared = _overlap_bbox(square, prev["square"])
                if shared is None:
                    continue
                overlap_pairs += 1
                pr0, pc0, pr1, pc1 = shared
                for rr in range(pr0, pr1 + 1):
                    for cc in range(pc0, pc1 + 1):
                        if not object_mask[rr, cc]:
                            continue
                        if (rr, cc) in pixel_set or (rr, cc) in prev["pixel_set"]:
                            continue
                        overlap_targets += 1

            candidates.append(
                {
                    "center": (r, c),
                    "length": length,
                    "pixels": pixels,
                    "pixel_set": pixel_set,
                    "square": square,
                    "target_count": target_count,
                    "arm_hit": arm_hit,
                    "overlap_pairs": overlap_pairs,
                    "overlap_targets": overlap_targets,
                }
            )
    return candidates


def _choose_cross_candidate(
    candidates: list[dict],
    require_overlap: bool,
    require_arm_hit: bool,
) -> dict | None:
    usable = [cand for cand in candidates if cand["target_count"] >= 2]
    if not usable:
        usable = [cand for cand in candidates if cand["target_count"] >= 1]
    if require_overlap:
        usable = [cand for cand in usable if cand["overlap_pairs"] > 0 and cand["overlap_targets"] > 0]
    if require_arm_hit:
        hit = [cand for cand in usable if cand["arm_hit"]]
        if hit:
            usable = hit
    if not usable:
        return None
    usable.sort(
        key=lambda cand: (cand["overlap_targets"], cand["target_count"], int(cand["arm_hit"])),
        reverse=True,
    )
    return random.choice(usable[: min(12, len(usable))])


def _place_crosses(base: np.ndarray, colors: list[int], want_overlap: bool) -> list[dict] | None:
    grid = np.array(base, copy=True)
    cross_occupied = np.zeros_like(grid, dtype=bool)
    crosses: list[dict] = []
    overlap_index = random.randint(1, len(colors) - 1) if want_overlap else -1
    for index, color in enumerate(colors):
        length = random.randint(2, 4)
        require_overlap = bool(index == overlap_index)
        require_arm_hit = random.random() < 0.55
        candidates = _enumerate_cross_candidates(base, cross_occupied, crosses, length)
        chosen = _choose_cross_candidate(candidates, require_overlap=require_overlap, require_arm_hit=require_arm_hit)
        if chosen is None:
            return None

        cross = {
            "center": chosen["center"],
            "color": int(color),
            "length": int(length),
            "pixels": chosen["pixels"],
            "pixel_set": chosen["pixel_set"],
            "square": chosen["square"],
        }
        for rr, cc in cross["pixels"]:
            grid[rr, cc] = color
            cross_occupied[rr, cc] = True
        cr, cc = cross["center"]
        grid[cr, cc] = GREY
        crosses.append(cross)
    return crosses


def _build_input() -> np.ndarray | None:
    H = random.randint(*GRID_HEIGHT_RANGE)
    W = random.randint(*GRID_WIDTH_RANGE)
    n_crosses = random.randint(2, 4)
    palette = PALETTE[:]
    random.shuffle(palette)
    cross_colors = palette[:n_crosses]
    object_colors = palette[n_crosses:]
    if len(object_colors) < 3:
        return None

    grid = np.full((H, W), AZURE, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    for _ in range(random.randint(*BIG_OBJECT_COUNT_RANGE)):
        if not _place_object(grid, occupied, _random_big_object(), random.choice(object_colors)):
            return None
    for _ in range(random.randint(*SMALL_OBJECT_COUNT_RANGE)):
        if not _place_object(grid, occupied, _random_small_object(), random.choice(object_colors)):
            return None

    base = np.array(grid, copy=True)
    want_overlap = n_crosses >= 2 and random.random() < 0.45
    crosses = _place_crosses(base, cross_colors, want_overlap=want_overlap)
    if crosses is None:
        return None

    for cross in crosses:
        for rr, cc in cross["pixels"]:
            grid[rr, cc] = cross["color"]
        cr, cc = cross["center"]
        grid[cr, cc] = GREY

    detected = _find_crosses(grid)
    expected_keys = sorted((c["center"], c["length"], c["color"]) for c in crosses)
    detected_keys = sorted((c["center"], c["length"], c["color"]) for c in detected)
    if detected_keys != expected_keys:
        return None

    out = _apply_crosses(grid, crosses)
    if not np.array_equal(main(grid), out):
        return None

    overlap_count = 0
    ordered = sorted(crosses, key=_cross_sort_key)
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            if _overlap_bbox(ordered[i]["square"], ordered[j]["square"]) is not None:
                overlap_count += 1
    if want_overlap and overlap_count == 0:
        return None
    if want_overlap:
        reverse_out = _apply_crosses(grid, list(reversed(ordered)))
        if np.array_equal(out, reverse_out):
            return None

    marker_mask = np.zeros_like(grid, dtype=bool)
    for cross in crosses:
        for rr, cc in cross["pixels"]:
            marker_mask[rr, cc] = True
    visible_recolors = int(((out != grid) & (~marker_mask)).sum())
    if visible_recolors < max(1, len(crosses) // 2):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
