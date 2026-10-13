import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
COLOR_ORDER = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8}
RECT_DIMS_BY_AREA = {
    4: [(2, 2)],
    6: [(2, 3)],
    8: [(2, 4)],
    9: [(3, 3)],
    10: [(2, 5)],
    12: [(2, 6), (3, 4)],
}
AREAS = tuple(sorted(RECT_DIMS_BY_AREA))
PAIR_COUNT_RANGE = (2, 4)
GRID_H_RANGE = (15, 26)
GRID_W_RANGE = (16, 30)
MAX_PAIR_ATTEMPTS = 200
MAX_LAYOUT_ATTEMPTS = 200
MAX_GENERATION_ATTEMPTS = 400

_RNG = np.random.default_rng()


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return np.array(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    seen = np.zeros_like(mask, dtype=bool)
    stack = [tuple(map(int, coords[0]))]
    seen[stack[0]] = True
    count = 0
    while stack:
        r, c = stack.pop()
        count += 1
        for dr, dc in DIR4:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                stack.append((rr, cc))
    return count == int(mask.sum())


def _extract_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    seen = np.zeros_like(g, dtype=bool)
    comps = []
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            color = int(g[r, c])
            if color == 0 or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if (
                        0 <= nr < g.shape[0]
                        and 0 <= nc < g.shape[1]
                        and not seen[nr, nc]
                        and int(g[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True
            comps.append(
                {
                    "color": color,
                    "area": int(len(coords)),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "mask": mask,
                }
            )
    comps.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["color"]))
    return comps


def _factor_pairs(area: int) -> list[tuple[int, int]]:
    pairs = []
    for h in range(1, int(np.sqrt(area)) + 1):
        if area % h != 0:
            continue
        w = area // h
        pairs.append((h, w))
        if h != w:
            pairs.append((w, h))
    return pairs


def _pattern_key(pattern: np.ndarray) -> tuple:
    return (
        int(pattern.shape[0]),
        int(pattern.shape[1]),
        tuple(int(x) for x in pattern.ravel()),
    )


def _rectangle_sort_key(rect: np.ndarray) -> tuple:
    colors = sorted({int(x) for x in np.unique(rect) if int(x) != 0}, key=lambda c: COLOR_ORDER[c])
    return (
        int(rect.size),
        int(COLOR_ORDER[colors[0]]),
        int(COLOR_ORDER[colors[1]]),
        int(rect.shape[0]),
        int(rect.shape[1]),
        tuple(int(x) for x in rect.ravel()),
    )


def _pair_rectangles(comp_a: dict, comp_b: dict) -> list[np.ndarray]:
    if int(comp_a["color"]) == int(comp_b["color"]):
        return []
    mask_a = np.asarray(comp_a["mask"], dtype=bool)
    mask_b = np.asarray(comp_b["mask"], dtype=bool)
    ha, wa = mask_a.shape
    hb, wb = mask_b.shape
    coords_a = np.argwhere(mask_a)
    coords_b = np.argwhere(mask_b)
    total = int(mask_a.sum() + mask_b.sum())
    patterns = {}

    for h, w in _factor_pairs(total):
        if ha > h or wa > w or hb > h or wb > w:
            continue
        placements_a = []
        for r0 in range(h - ha + 1):
            for c0 in range(w - wa + 1):
                occ = np.zeros((h, w), dtype=bool)
                rr = coords_a[:, 0] + r0
                cc = coords_a[:, 1] + c0
                occ[rr, cc] = True
                placements_a.append((occ, rr, cc))
        for occ_a, rr_a, cc_a in placements_a:
            for r0 in range(h - hb + 1):
                for c0 in range(w - wb + 1):
                    rr_b = coords_b[:, 0] + r0
                    cc_b = coords_b[:, 1] + c0
                    if np.any(occ_a[rr_b, cc_b]):
                        continue
                    board = np.zeros((h, w), dtype=int)
                    board[rr_a, cc_a] = int(comp_a["color"])
                    board[rr_b, cc_b] = int(comp_b["color"])
                    patterns.setdefault(board.tobytes(), board)

    return [patterns[key] for key in sorted(patterns, key=lambda k: _pattern_key(patterns[k]))]


def _build_pair_map(comps: list[dict]) -> dict[tuple[int, int], np.ndarray]:
    pair_map = {}
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            rects = _pair_rectangles(comps[i], comps[j])
            if rects:
                pair_map[(i, j)] = rects[0]
    return pair_map


def _choose_matching(pair_map: dict[tuple[int, int], np.ndarray], n: int) -> list[tuple[int, int]]:
    def search(used: set[int], chosen: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
        if len(used) == n:
            return list(chosen)

        best_i = None
        best_options = None
        for i in range(n):
            if i in used:
                continue
            options = []
            for j in range(n):
                if j == i or j in used:
                    continue
                key = (min(i, j), max(i, j))
                if key not in pair_map:
                    continue
                rect = pair_map[key]
                options.append((j, _rectangle_sort_key(rect), _pattern_key(rect)))
            if best_options is None or len(options) < len(best_options):
                best_i = i
                best_options = options
        if not best_options:
            return None

        for j, _sort_key, _pat_key in sorted(best_options, key=lambda item: (item[1], item[2], item[0])):
            used.add(best_i)
            used.add(j)
            chosen.append((min(best_i, j), max(best_i, j)))
            result = search(used, chosen)
            if result is not None:
                return result
            chosen.pop()
            used.remove(best_i)
            used.remove(j)
        return None

    result = search(set(), [])
    if result is None:
        raise ValueError("no valid pairing")
    return result


def _count_matchings(pair_map: dict[tuple[int, int], np.ndarray], n: int, limit: int = 2) -> int:
    def search(used: set[int]) -> int:
        if len(used) == n:
            return 1

        best_i = None
        best_options = None
        for i in range(n):
            if i in used:
                continue
            options = []
            for j in range(n):
                if j == i or j in used:
                    continue
                if (min(i, j), max(i, j)) in pair_map:
                    options.append(j)
            if best_options is None or len(options) < len(best_options):
                best_i = i
                best_options = options
        if not best_options:
            return 0

        total = 0
        for j in best_options:
            used.add(best_i)
            used.add(j)
            total += search(used)
            used.remove(best_i)
            used.remove(j)
            if total >= limit:
                return limit
        return total

    return search(set())


def _pack_rectangles(rects: list[np.ndarray]) -> np.ndarray:
    ordered = sorted(rects, key=_rectangle_sort_key)
    height = max(int(rect.shape[0]) for rect in ordered)
    width = sum(int(rect.shape[1]) for rect in ordered) + len(ordered) - 1
    out = np.zeros((height, width), dtype=int)
    c0 = 0
    for rect in ordered:
        h, w = rect.shape
        out[:h, c0 : c0 + w] = rect
        c0 += w + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair scattered monochrome fragments by translational rectangle complementation,
    rebuild each pair into its filled rectangle, then sort the rectangles by area.
    """
    comps = _extract_components(np.asarray(grid, dtype=int))
    if len(comps) == 0 or len(comps) % 2 != 0:
        raise ValueError("expected a non-empty even number of components")

    pair_map = _build_pair_map(comps)
    matching = _choose_matching(pair_map, len(comps))
    rects = [pair_map[(i, j)] for i, j in matching]
    return _pack_rectangles(rects)


def _random_connected_subset(h: int, w: int, area: int) -> np.ndarray | None:
    mask = np.zeros((h, w), dtype=bool)
    start = tuple(_RNG.integers([0, 0], [h, w]))
    mask[start] = True
    while int(mask.sum()) < area:
        frontier = []
        for r, c in np.argwhere(mask):
            for dr, dc in DIR4:
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < h and 0 <= cc < w and not mask[rr, cc]:
                    frontier.append((rr, cc))
        if not frontier:
            return None
        rr, cc = frontier[int(_RNG.integers(0, len(frontier)))]
        mask[rr, cc] = True
    return mask


def _dihedral_masks(mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    board = np.zeros(mask_a.shape, dtype=int)
    board[mask_a] = 1
    board[mask_b] = 2
    if bool(_RNG.integers(0, 2)):
        board = np.fliplr(board)
    board = np.rot90(board, int(_RNG.integers(0, 4)))
    return _trim_mask(board == 1), _trim_mask(board == 2)


def _is_full_rectangle(mask: np.ndarray) -> bool:
    return int(mask.sum()) == int(mask.shape[0] * mask.shape[1])


def _make_pair(area: int, color_a: int, color_b: int) -> dict | None:
    dims = RECT_DIMS_BY_AREA[area]
    h, w = dims[int(_RNG.integers(0, len(dims)))]
    for _ in range(MAX_PAIR_ATTEMPTS):
        split_a = int(_RNG.integers(max(2, area - 6), min(6, area - 2) + 1))
        mask_a = _random_connected_subset(h, w, split_a)
        if mask_a is None:
            continue
        mask_b = ~mask_a
        if not _is_connected(mask_b):
            continue
        if _is_full_rectangle(mask_a) and _is_full_rectangle(mask_b):
            continue

        mask_a, mask_b = _dihedral_masks(mask_a, mask_b)
        comp_a = {"color": int(color_a), "mask": mask_a}
        comp_b = {"color": int(color_b), "mask": mask_b}
        rects = _pair_rectangles(comp_a, comp_b)
        if len(rects) != 1:
            continue
        return {
            "a": {"color": int(color_a), "mask": mask_a},
            "b": {"color": int(color_b), "mask": mask_b},
            "rect": rects[0],
            "area": int(area),
        }
    return None


def _choose_areas(pair_count: int, tie_branch: bool) -> list[int]:
    if not tie_branch:
        picks = _RNG.choice(np.array(AREAS), size=pair_count, replace=False)
        return [int(x) for x in picks]

    tie_area = int(AREAS[int(_RNG.integers(0, len(AREAS)))])
    remaining = [area for area in AREAS if area != tie_area]
    other_count = pair_count - 2
    others = []
    if other_count > 0:
        picks = _RNG.choice(np.array(remaining), size=other_count, replace=False)
        others = [int(x) for x in picks]
    areas = [tie_area, tie_area] + others
    _RNG.shuffle(areas)
    return areas


def _can_place(occupied: np.ndarray, mask: np.ndarray, top: int, left: int) -> bool:
    h, w = mask.shape
    coords = np.argwhere(mask)
    for rr, cc in coords:
        r = int(top + rr)
        c = int(left + cc)
        r0 = max(0, r - 1)
        r1 = min(occupied.shape[0], r + 2)
        c0 = max(0, c - 1)
        c1 = min(occupied.shape[1], c + 2)
        if occupied[r0:r1, c0:c1].any():
            return False
    return True


def _paint(grid: np.ndarray, occupied: np.ndarray, mask: np.ndarray, color: int, top: int, left: int) -> None:
    for rr, cc in np.argwhere(mask):
        r = int(top + rr)
        c = int(left + cc)
        grid[r, c] = int(color)
        occupied[r, c] = True


def _layout_shapes(shapes: list[dict]) -> np.ndarray | None:
    total_box = sum((shape["mask"].shape[0] + 2) * (shape["mask"].shape[1] + 2) for shape in shapes)
    min_side = int(np.ceil(np.sqrt(total_box))) + 2
    h_lo = max(GRID_H_RANGE[0], min_side)
    w_lo = max(GRID_W_RANGE[0], min_side + 1)
    if h_lo > GRID_H_RANGE[1] or w_lo > GRID_W_RANGE[1]:
        return None

    for _ in range(MAX_LAYOUT_ATTEMPTS):
        h = int(_RNG.integers(h_lo, GRID_H_RANGE[1] + 1))
        w = int(_RNG.integers(w_lo, GRID_W_RANGE[1] + 1))
        grid = np.zeros((h, w), dtype=int)
        occupied = np.zeros((h, w), dtype=bool)
        order = list(range(len(shapes)))
        _RNG.shuffle(order)
        ok = True
        for idx in order:
            shape = shapes[idx]
            mh, mw = shape["mask"].shape
            placed = False
            for _ in range(120):
                top = int(_RNG.integers(0, h - mh + 1))
                left = int(_RNG.integers(0, w - mw + 1))
                if not _can_place(occupied, shape["mask"], top, left):
                    continue
                _paint(grid, occupied, shape["mask"], shape["color"], top, left)
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            return grid
    return None


def _gen_attempt() -> np.ndarray | None:
    pair_count = int(_RNG.integers(PAIR_COUNT_RANGE[0], PAIR_COUNT_RANGE[1] + 1))
    tie_branch = bool(_RNG.integers(0, 2))
    areas = _choose_areas(pair_count, tie_branch)

    colors = [int(x) for x in _RNG.choice(np.arange(1, 10), size=2 * pair_count, replace=False)]
    _RNG.shuffle(colors)

    pairs = []
    for idx, area in enumerate(areas):
        pair = _make_pair(area, colors[2 * idx], colors[2 * idx + 1])
        if pair is None:
            return None
        pairs.append(pair)

    shapes = []
    expected_rects = []
    for pair in pairs:
        shapes.append(pair["a"])
        shapes.append(pair["b"])
        expected_rects.append(pair["rect"])

    grid = _layout_shapes(shapes)
    if grid is None:
        return None

    comps = _extract_components(grid)
    if len(comps) != 2 * pair_count:
        return None
    if any(not (2 <= comp["area"] <= 6) for comp in comps):
        return None

    pair_map = _build_pair_map(comps)
    if _count_matchings(pair_map, len(comps), limit=2) != 1:
        return None

    expected = _pack_rectangles(expected_rects)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid complementary-rectangle sample")
