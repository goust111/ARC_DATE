import numpy as np
from collections import Counter, deque
from typing import Dict, List, Optional, Sequence, Set, Tuple


def main(grid: np.ndarray) -> np.ndarray:
    """Pair complementary fragments into full rectangles and sort them by original gap.

    Input structure:
    - The background color is black (0).
    - Non-black pixels form several monochrome 4-connected components, all with distinct colors.
    - For each valid pair of components, translating them without rotation or flip makes a complete filled axis-aligned rectangle.

    Rule:
    - Extract all non-black components and find every pair that can be translated so their masks tile one solid rectangle.
    - Keep each component's original color when placing it into that rectangle.
    - For each found pair, compute the horizontal distance between component bbox centers from the input and sort pairs by that distance ascending.
    - Output the translated rectangles from left to right with one background column between consecutive rectangles.

    Key insight:
    - Pairing is recovered from geometric complementarity (shape + rectangle completion), while the output order is independently encoded by the original bbox-center distance signal.
    """

    comps = _extract_components(grid)
    if len(comps) < 2:
        return np.array(grid, copy=True)

    pair_candidates = _find_all_pair_candidates(comps)
    if not pair_candidates:
        return np.array(grid, copy=True)

    paired = _select_pairs(pair_candidates, len(comps))
    if not paired:
        return np.array(grid, copy=True)

    heights = [p["h"] for p in paired]
    widths = [p["w"] for p in paired]

    # Keep a consistent size; inputs generated for this task should already satisfy this.
    common_h = Counter(heights).most_common(1)[0][0]
    common_w = Counter(widths).most_common(1)[0][0]
    paired = [p for p in paired if p["h"] == common_h and p["w"] == common_w]
    if not paired:
        return np.array(grid, copy=True)

    paired.sort(key=lambda p: (p["distance2"], p["i"], p["j"]))
    h = common_h
    w = common_w

    out_width = len(paired) * w + max(0, len(paired) - 1)
    out = np.zeros((h, out_width), dtype=grid.dtype)

    col0 = 0
    for p in paired:
        for r, c in p["coords_a"]:
            out[r, col0 + c] = comps[p["i"]]["color"]
        for r, c in p["coords_b"]:
            out[r, col0 + c] = comps[p["j"]]["color"]
        col0 += w + 1

    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    h = int(rng.integers(5, 31))
    w = int(rng.integers(5, 31))
    grid = np.zeros((h, w), dtype=np.int64)
    for color in range(1, int(rng.integers(2, 6))):
        for _ in range(int(rng.integers(1, 4))):
            grid[int(rng.integers(0, h)), int(rng.integers(0, w))] = color
    return grid


def _fallback_input() -> np.ndarray:
    H, W = 20, 22
    grid = np.zeros((H, W), dtype=np.int64)
    rng = np.random.default_rng(0)
    rect_h = 4
    rect_w = 6

    def to_coords(mask: Set[Tuple[int, int]], top: int, left: int, color: int):
        for r, c in mask:
            grid[r + top, c + left] = color

    def split_pair():
        a = set()
        b = set()
        k = rect_w // 2
        for r in range(rect_h):
            for c in range(rect_w):
                if c < k:
                    a.add((r, c))
                else:
                    b.add((r, c))
        return a, b

    a1, b1 = split_pair()
    a2, b2 = _make_border_pair(rect_h, rect_w)
    for r, c in a1:
        grid[r + 2, c + 1] = 1
        grid[r + 7, c + 14] = 3
    for r, c in b1:
        grid[r + 2, c + 12] = 2
        grid[r + 7, c + 1] = 4
    for r, c in a2:
        grid[r + 12, c + 3] = 5
    for r, c in b2:
        grid[r + 12, c + 12] = 6
    return grid


def _extract_components(grid: np.ndarray) -> List[Dict]:
    H, W = grid.shape
    seen = np.zeros((H, W), dtype=bool)
    comps: List[Dict] = []
    for r0 in range(H):
        for c0 in range(W):
            color = int(grid[r0, c0])
            if color == 0 or seen[r0, c0]:
                continue

            q = deque([(r0, c0)])
            seen[r0, c0] = True
            pixels: List[Tuple[int, int]] = []
            min_r = max_r = r0
            min_c = max_c = c0

            while q:
                r, c = q.popleft()
                pixels.append((r, c))
                if r < min_r:
                    min_r = r
                if r > max_r:
                    max_r = r
                if c < min_c:
                    min_c = c
                if c > max_c:
                    max_c = c

                if r > 0 and not seen[r - 1, c] and int(grid[r - 1, c]) == color:
                    seen[r - 1, c] = True
                    q.append((r - 1, c))
                if r + 1 < H and not seen[r + 1, c] and int(grid[r + 1, c]) == color:
                    seen[r + 1, c] = True
                    q.append((r + 1, c))
                if c > 0 and not seen[r, c - 1] and int(grid[r, c - 1]) == color:
                    seen[r, c - 1] = True
                    q.append((r, c - 1))
                if c + 1 < W and not seen[r, c + 1] and int(grid[r, c + 1]) == color:
                    seen[r, c + 1] = True
                    q.append((r, c + 1))

            comp_set = set(pixels)
            comps.append(
                {
                    "color": color,
                    "pixels": pixels,
                    "set": comp_set,
                    "r0": min_r,
                    "r1": max_r,
                    "c0": min_c,
                    "c1": max_c,
                    "h": max_r - min_r + 1,
                    "w": max_c - min_c + 1,
                    "center2c": min_c + max_c,
                }
            )

    return comps


def _find_all_pair_candidates(comps: List[Dict]) -> List[Dict]:
    n = len(comps)
    candidates: List[Dict] = []
    for i in range(n):
        for j in range(i + 1, n):
            rel = _find_tile_relation(comps[i], comps[j])
            if rel is None:
                continue
            rel = dict(rel)
            rel["i"] = i
            rel["j"] = j
            rel["distance2"] = abs(comps[i]["center2c"] - comps[j]["center2c"])
            candidates.append(rel)
    return candidates


def _find_tile_relation(comp_a: Dict, comp_b: Dict) -> Optional[Dict]:
    a_set = comp_a["set"]
    b_set = comp_b["set"]

    if comp_a["h"] == comp_a["w"] == 1 and comp_b["h"] == comp_b["w"] == 1:
        return None

    min_ra, min_ca, max_ra, max_ca = comp_a["r0"], comp_a["c0"], comp_a["r1"], comp_a["c1"]
    min_rb, min_cb, max_rb, max_cb = comp_b["r0"], comp_b["c0"], comp_b["r1"], comp_b["c1"]

    len_a = len(a_set)
    len_b = len(b_set)

    best: Optional[Dict] = None
    for dr in range(-comp_b["h"], comp_a["h"] + 1):
        shifted_b_min_r = min_rb + dr
        shifted_b_max_r = max_rb + dr
        min_r = min(min_ra, shifted_b_min_r)
        max_r = max(max_ra, shifted_b_max_r)

        for dc in range(-comp_b["w"], comp_a["w"] + 1):
            shifted_b_min_c = min_cb + dc
            shifted_b_max_c = max_cb + dc
            min_c = min(min_ca, shifted_b_min_c)
            max_c = max(max_ca, shifted_b_max_c)

            h = max_r - min_r + 1
            w = max_c - min_c + 1

            if h * w != len_a + len_b:
                continue

            # Ensure no overlap and exact rectangle fill.
            valid = True
            for r_b, c_b in comp_b["pixels"]:
                rr = r_b + dr
                cc = c_b + dc
                if (rr, cc) in a_set:
                    valid = False
                    break
            if not valid:
                continue

            # Additional quick test (implicit in area check): union of two disjoint sets
            # must cover the whole bbox.
            candidate = {
                "h": h,
                "w": w,
                "coords_a": [(r - min_r, c - min_c) for (r, c) in comp_a["pixels"]],
                "coords_b": [(r + dr - min_r, c + dc - min_c) for (r, c) in comp_b["pixels"]],
            }

            if best is None:
                best = candidate
            else:
                key = (candidate["h"], candidate["w"], min_r, min_c, dr, dc)
                best_key = (best["h"], best["w"], min_r, min_c, dr, dc)
                if key < best_key:
                    best = candidate

    return best


def _select_pairs(candidates: List[Dict], n: int) -> List[Dict]:
    by_i = [[] for _ in range(n)]
    for c in candidates:
        by_i[c["i"]].append(c)

    for bucket in by_i:
        bucket.sort(key=lambda d: (d["h"], d["w"], d["distance2"], d["i"], d["j"]))

    best: List[Dict] = []
    used = [False] * n

    def signature(pairs: Sequence[Dict]) -> Tuple:
        return tuple((p["distance2"], p["i"], p["j"]) for p in sorted(pairs, key=lambda x: (x["distance2"], x["i"], x["j"])) )

    def backtrack(current: List[Dict]) -> None:
        nonlocal best

        remaining = sum(1 for u in used if not u)
        upper_bound = len(current) + remaining // 2
        if upper_bound <= len(best):
            return

        try:
            i = next(idx for idx, u in enumerate(used) if not u)
        except StopIteration:
            if len(current) > len(best) or (len(current) == len(best) and signature(current) < signature(best)):
                best = list(current)
            return

        used[i] = True

        # Option to leave this component unmatched.
        backtrack(current)

        for c in by_i[i]:
            j = c["j"]
            if used[j]:
                continue
            used[j] = True
            current.append(c)
            backtrack(current)
            current.pop()
            used[j] = False

        used[i] = False

    backtrack([])
    return best


def _make_shape_pair(rect_h: int, rect_w: int, rng: np.random.Generator) -> Optional[Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]]:
    methods = [
        _make_random_connected_pair,
        _make_split_pair,
        _make_border_pair,
    ]

    for _ in range(40):
        method = int(rng.integers(0, len(methods)))
        a, b = methods[method](rect_h, rect_w, rng)
        if a is None or b is None:
            continue
        if not a or not b:
            continue
        a = set(a)
        b = set(b)
        # Keep a and b connected and complementary by construction.
        if _shape_is_connected(a, rect_h, rect_w) and _shape_is_connected(b, rect_h, rect_w):
            a, b = _random_flip_pair(a, b, rect_h, rect_w, rng)
            return a, b

    return None


def _make_split_pair(rect_h: int, rect_w: int, rng: np.random.Generator):
    if rect_w >= 2 and (rect_h < 2 or rng.random() < 0.5):
        cut = int(rng.integers(1, rect_w))
        a = set()
        b = set()
        for r in range(rect_h):
            for c in range(rect_w):
                (a if c < cut else b).add((r, c))
    elif rect_h >= 2:
        cut = int(rng.integers(1, rect_h))
        a = set()
        b = set()
        for r in range(rect_h):
            for c in range(rect_w):
                (a if r < cut else b).add((r, c))
    else:
        return None, None

    return a, b


def _make_border_pair(rect_h: int, rect_w: int, rng: np.random.Generator = None):
    del rng
    if rect_h < 3 or rect_w < 3:
        return None, None
    a = set()
    b = set()
    for r in range(rect_h):
        for c in range(rect_w):
            if r == 0 or c == 0 or r == rect_h - 1 or c == rect_w - 1:
                a.add((r, c))
            else:
                b.add((r, c))
    return a, b


def _make_random_connected_pair(rect_h: int, rect_w: int, rng: np.random.Generator):
    total = rect_h * rect_w
    if total < 2:
        return None, None

    min_area = max(1, total // 4)
    max_area = max(1, (3 * total) // 4)

    all_cells = [(r, c) for r in range(rect_h) for c in range(rect_w)]

    for _ in range(80):
        target_area = int(rng.integers(min_area, max_area + 1))
        start = all_cells[int(rng.integers(len(all_cells)))]
        shape_a = {start}
        frontier = {start}

        for _ in range(1, target_area):
            candidates = set()
            for r0, c0 in list(frontier):
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    r = r0 + dr
                    c = c0 + dc
                    if 0 <= r < rect_h and 0 <= c < rect_w and (r, c) not in shape_a:
                        candidates.add((r, c))
            if not candidates:
                shape_a = None
                break
            next_cell = tuple(candidates.pop() if len(candidates) == 1 else candidates.pop())
            shape_a.add(next_cell)
            frontier.add(next_cell)

        if shape_a is None:
            continue

        shape_b = {(r, c) for r in range(rect_h) for c in range(rect_w) if (r, c) not in shape_a}
        if not shape_a or not shape_b:
            continue
        if _shape_is_connected(shape_b, rect_h, rect_w):
            return shape_a, shape_b

    return None, None


def _random_flip_pair(a: Set[Tuple[int, int]], b: Set[Tuple[int, int]], h: int, w: int, rng: np.random.Generator) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
    if rng.random() < 0.5:
        a = {(r, w - 1 - c) for r, c in a}
        b = {(r, w - 1 - c) for r, c in b}
    if rng.random() < 0.5:
        a = {(h - 1 - r, c) for r, c in a}
        b = {(h - 1 - r, c) for r, c in b}
    return a, b


def _shape_is_connected(shape: Set[Tuple[int, int]], h: int, w: int) -> bool:
    if not shape:
        return False
    q = deque([next(iter(shape))])
    seen = {q[0]}
    area = 1
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) in shape and (rr, cc) not in seen:
                seen.add((rr, cc))
                q.append((rr, cc))
                area += 1
    return area == len(shape)


def _can_place(shape: Set[Tuple[int, int]], used: np.ndarray, top: int, left: int) -> bool:
    H, W = used.shape
    for r, c in shape:
        rr = r + top
        cc = c + left
        if rr < 0 or rr >= H or cc < 0 or cc >= W:
            return False
        if used[rr, cc]:
            return False
    # Add 1-cell spacing to avoid accidental merge by 4-connectivity.
    for r, c in shape:
        rr = r + top
        cc = c + left
        if rr > 0 and used[rr - 1, cc]:
            return False
        if rr + 1 < H and used[rr + 1, cc]:
            return False
        if cc > 0 and used[rr, cc - 1]:
            return False
        if cc + 1 < W and used[rr, cc + 1]:
            return False
    return True


def _bbox_shape_size(shape: Set[Tuple[int, int]]) -> Tuple[int, int]:
    rs = [r for r, _ in shape]
    cs = [c for _, c in shape]
    return max(rs) - min(rs) + 1, max(cs) - min(cs) + 1


def _bbox_col2(shape: Set[Tuple[int, int]], left: int) -> Tuple[int, int]:
    cols = [left + c for _, c in shape]
    return min(cols), max(cols)
