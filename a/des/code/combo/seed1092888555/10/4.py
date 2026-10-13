import random
from collections import Counter, deque

import numpy as np


BG = 4  # yellow
PALETTE = [c for c in range(10) if c != BG]
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_ATTEMPTS = 500


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            norm = {(rr - r0, cc - c0) for rr, cc in coords}
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "shape": frozenset(norm),
                    "height": r1 - r0 + 1,
                    "width": c1 - c0 + 1,
                }
            )
    return comps


def _mirror_shape(shape: frozenset[tuple[int, int]], width: int) -> frozenset[tuple[int, int]]:
    return frozenset((r, width - 1 - c) for r, c in shape)


def _is_mirror_pair(left: dict, right: dict) -> bool:
    if left["color"] != right["color"]:
        return False
    if left["height"] != right["height"] or left["width"] != right["width"]:
        return False
    lr0, lc0, lr1, lc1 = left["bbox"]
    rr0, rc0, rr1, rc1 = right["bbox"]
    if (lr0, lr1) != (rr0, rr1):
        return False
    if lc1 >= rc0:
        return False
    return _mirror_shape(left["shape"], left["width"]) == right["shape"]


def _candidate_pairs(components: list[dict]) -> list[tuple[int, int]]:
    by_color: dict[int, list[tuple[int, dict]]] = {}
    for idx, comp in enumerate(components):
        by_color.setdefault(int(comp["color"]), []).append((idx, comp))

    pairs: list[tuple[int, int]] = []
    for items in by_color.values():
        items.sort(key=lambda item: (item[1]["bbox"][0], item[1]["bbox"][1]))
        for pos_i, (idx_i, comp_i) in enumerate(items):
            ci0 = comp_i["bbox"][1]
            ci1 = comp_i["bbox"][3]
            best: tuple[int, int, int] | None = None
            for idx_j, comp_j in items[pos_i + 1 :]:
                if not _is_mirror_pair(comp_i, comp_j):
                    continue
                gap = comp_j["bbox"][1] - ci1 - 1
                if gap < 0:
                    continue
                if best is None or gap < best[0] or (gap == best[0] and comp_j["bbox"][1] < best[1]):
                    best = (gap, comp_j["bbox"][1], idx_j)
            if best is not None:
                pairs.append((idx_i, best[2]))
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find same-color object pairs that are horizontal mirrors and merge each pair.

    Background is yellow. Foreground objects are 4-connected monochromatic
    components. Whenever two objects of the same color occupy the same row band
    and one is the horizontal mirror of the other, remove both and redraw a
    tightly joined left-right symmetric composite centered between them.
    Unmatched objects stay unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = _background_color(g)

    comps = _find_components(g, bg)
    out = g.copy()
    used: set[int] = set()

    for left_idx, right_idx in _candidate_pairs(comps):
        if left_idx in used or right_idx in used:
            continue
        left = comps[left_idx]
        right = comps[right_idx]
        if not _is_mirror_pair(left, right):
            continue

        used.add(left_idx)
        used.add(right_idx)
        color = int(left["color"])
        r0, c0, r1, _ = left["bbox"]
        _, _, _, rc1 = right["bbox"]
        union_w = rc1 - c0 + 1
        merged_w = left["width"] + right["width"]
        start_c = c0 + (union_w - merged_w) // 2

        for rr, cc in left["coords"]:
            out[rr, cc] = bg
        for rr, cc in right["coords"]:
            out[rr, cc] = bg

        for dr, dc in left["shape"]:
            out[r0 + dr, start_c + dc] = color
        for dr, dc in right["shape"]:
            out[r0 + dr, start_c + left["width"] + dc] = color

    return out


def _shape_from_cells(cells: set[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, c0 = min(rows), min(cols)
    return frozenset((r - r0, c - c0) for r, c in cells)


def _random_connected_shape(height: int, width: int, area: int) -> frozenset[tuple[int, int]]:
    # Build a connected skeleton that touches all four bbox sides, then expand.
    cells = {(r, 0) for r in range(height)}
    cells.update((height - 1, c) for c in range(width))
    while len(cells) < area:
        r, c = random.choice(tuple(cells))
        options = []
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                options.append((nr, nc))
        if options:
            cells.add(random.choice(options))
    return _shape_from_cells(cells)


def _shape_key(shape: frozenset[tuple[int, int]], height: int, width: int) -> tuple:
    return (height, width, tuple(sorted(shape)))


def _place_shape(grid: np.ndarray, occupied: np.ndarray, top: int, left: int, shape: frozenset[tuple[int, int]], color: int) -> None:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    h = max(rows) + 1
    w = max(cols) + 1
    for dr, dc in shape:
        grid[top + dr, left + dc] = color
    r0 = max(0, top - 1)
    r1 = min(grid.shape[0], top + h + 1)
    c0 = max(0, left - 1)
    c1 = min(grid.shape[1], left + w + 1)
    occupied[r0:r1, c0:c1] = True


def _free(occupied: np.ndarray, top: int, left: int, height: int, width: int) -> bool:
    h, w = occupied.shape
    if top < 0 or left < 0 or top + height > h or left + width > w:
        return False
    r0 = max(0, top - 1)
    r1 = min(h, top + height + 1)
    c0 = max(0, left - 1)
    c1 = min(w, left + width + 1)
    return not occupied[r0:r1, c0:c1].any()


def _sample_pair_shape(forbidden: set[tuple]) -> tuple[frozenset[tuple[int, int]], int, int] | None:
    for _ in range(80):
        height = random.randint(2, 5)
        width = random.randint(2, 4)
        area = random.randint(max(height + width - 1, 3), min(height * width - 1, height * width))
        shape = _random_connected_shape(height, width, area)
        key = _shape_key(shape, height, width)
        mirror_key = _shape_key(_mirror_shape(shape, width), height, width)
        if key not in forbidden and mirror_key not in forbidden:
            return shape, height, width
    return None


def _sample_distractor_shape(forbidden: set[tuple]) -> tuple[frozenset[tuple[int, int]], int, int] | None:
    for _ in range(120):
        height = random.randint(2, 5)
        width = random.randint(2, 5)
        area = random.randint(max(height + width - 1, 3), min(height * width - 1, height * width))
        shape = _random_connected_shape(height, width, area)
        mirror = _mirror_shape(shape, width)
        if shape == mirror:
            continue
        key = _shape_key(shape, height, width)
        mirror_key = _shape_key(mirror, height, width)
        if key not in forbidden and mirror_key not in forbidden:
            return shape, height, width
    return None


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(16, 26)
    w = random.randint(16, 28)
    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    pair_count = random.randint(1, 3)
    distractor_count = random.randint(1, 3)
    total_colors = pair_count + distractor_count
    if total_colors > len(PALETTE):
        return None
    colors = random.sample(PALETTE, total_colors)
    pair_colors = colors[:pair_count]
    distractor_colors = colors[pair_count:]

    used_shape_keys: set[tuple] = set()
    pair_specs: list[tuple[int, int, int, frozenset[tuple[int, int]], int, int]] = []

    for color in pair_colors:
        sampled = _sample_pair_shape(used_shape_keys)
        if sampled is None:
            return None
        shape, box_h, box_w = sampled
        mirror = _mirror_shape(shape, box_w)
        used_shape_keys.add(_shape_key(shape, box_h, box_w))
        used_shape_keys.add(_shape_key(mirror, box_h, box_w))

        placed = False
        for _ in range(120):
            gap = random.randint(3, 8)
            total_w = 2 * box_w + gap
            if total_w > w - 4 or box_h > h - 2:
                continue
            top = random.randint(1, h - box_h - 1)
            left = random.randint(1, w - total_w - 1)
            right = left + box_w + gap
            if not _free(occupied, top, left, box_h, box_w):
                continue
            if not _free(occupied, top, right, box_h, box_w):
                continue
            _place_shape(grid, occupied, top, left, shape, color)
            _place_shape(grid, occupied, top, right, mirror, color)
            pair_specs.append((color, top, left, shape, box_h, box_w))
            placed = True
            break
        if not placed:
            return None

    for idx, color in enumerate(distractor_colors):
        use_pair_color = pair_colors and random.random() < 0.5
        actual_color = random.choice(pair_colors) if use_pair_color else color
        sampled = _sample_distractor_shape(used_shape_keys)
        if sampled is None:
            return None
        shape, box_h, box_w = sampled
        used_shape_keys.add(_shape_key(shape, box_h, box_w))
        used_shape_keys.add(_shape_key(_mirror_shape(shape, box_w), box_h, box_w))

        placed = False
        for _ in range(120):
            top = random.randint(1, h - box_h - 1)
            left = random.randint(1, w - box_w - 1)
            if not _free(occupied, top, left, box_h, box_w):
                continue
            # Avoid sharing a row band with same-color pair objects; that keeps matching unambiguous.
            bad_band = False
            for pair_color, pair_top, _, _, pair_h, _ in pair_specs:
                if pair_color == actual_color and pair_top == top and pair_h == box_h:
                    bad_band = True
                    break
            if bad_band:
                continue
            _place_shape(grid, occupied, top, left, shape, actual_color)
            placed = True
            break
        if not placed:
            return None

    out = main(grid)
    in_counts = Counter(map(int, grid.flat))
    out_counts = Counter(map(int, out.flat))
    if in_counts != out_counts:
        return None

    comps = _find_components(grid, BG)
    pair_total = 0
    for a, b in _candidate_pairs(comps):
        if _is_mirror_pair(comps[a], comps[b]):
            pair_total += 1
    if pair_total < pair_count:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
