from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
COLORS = tuple(range(1, 10))
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _min_block_distance(point: tuple[int, int], block_cells: list[tuple[int, int]]) -> int:
    r, c = point
    return min(abs(r - br) + abs(c - bc) for br, bc in block_cells)


def _extract_patterns(grid: np.ndarray) -> list[dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    colors = [int(color) for color in np.unique(g) if int(color) != BACKGROUND]
    assert 2 <= len(colors) <= 3, "expected 2-3 distinct primitive colors"

    patterns: list[dict[str, int]] = []
    for color in colors:
        comps = _components(g == color)
        anchors: list[list[tuple[int, int]]] = []
        singles: list[tuple[int, int]] = []
        for comp in comps:
            r0, c0, r1, c1 = _bbox(comp)
            if len(comp) == 4 and (r1 - r0 + 1, c1 - c0 + 1) == (2, 2):
                anchors.append(comp)
            else:
                assert len(comp) == 1, "non-anchor same-color objects must be isolated single pixels"
                singles.append(comp[0])

        assert len(anchors) == 1, "each color must have exactly one 2x2 primitive block"
        anchor = anchors[0]
        r0, c0, r1, c1 = _bbox(anchor)
        assert (r1 - r0 + 1, c1 - c0 + 1) == (2, 2)

        nearby = [pt for pt in singles if _min_block_distance(pt, anchor) <= 3]
        assert len(nearby) == len(singles), "all same-color singleton counters must be near their primitive"
        k = len(nearby)
        assert 1 <= k <= 3, "each primitive must have 1-3 counter pixels"

        patterns.append({"color": color, "r": r0, "c": c0, "k": k})

    counts = [pattern["k"] for pattern in patterns]
    assert len(counts) == len(set(counts)), "counter values must be distinct"
    return patterns


def _priority_order(patterns: list[dict[str, int]]) -> list[dict[str, int]]:
    return sorted(patterns, key=lambda p: (-p["k"], p["r"], p["c"]))


def _simulate_output(patterns: list[dict[str, int]], shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    out = np.zeros((h, w), dtype=int)
    for pattern in _priority_order(patterns):
        r = pattern["r"]
        c = pattern["c"]
        span = 2 * (pattern["k"] + 1)
        assert r + 2 <= h and c + span <= w, "tiled primitive must fit inside the grid"
        for rr in range(r, r + 2):
            for cc in range(c, c + span):
                if out[rr, cc] == BACKGROUND:
                    out[rr, cc] = pattern["color"]
    return out


def _footprint(pattern: dict[str, int]) -> tuple[int, int, int, int]:
    r0 = pattern["r"]
    c0 = pattern["c"]
    return r0, c0, r0 + 1, c0 + 2 * (pattern["k"] + 1) - 1


def _overlap_area(a: dict[str, int], b: dict[str, int]) -> int:
    ar0, ac0, ar1, ac1 = _footprint(a)
    br0, bc0, br1, bc1 = _footprint(b)
    ih = min(ar1, br1) - max(ar0, br0) + 1
    iw = min(ac1, bc1) - max(ac0, bc0) + 1
    if ih <= 0 or iw <= 0:
        return 0
    return ih * iw


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count scattered same-color singleton pixels to decide how many rightward copies each 2x2 primitive should make.

    Input structure:
    - The grid has a black background and 2-3 distinct non-black colors.
    - Each color appears as one filled 2x2 primitive block plus 1-3 isolated same-color single pixels nearby.
    - The singleton pixels are not 4-connected to their primitive block and serve only as counters.

    Rule:
    - For each color, find its 2x2 primitive and count the nearby isolated same-color singleton pixels; call that count K.
    - Erase all singleton counter pixels and replace the primitive by a rightward tiling of K extra adjacent 2x2 copies, so the color occupies a solid 2 x (2*(K+1)) strip.
    - If strips from different primitives overlap, the strip with larger K keeps the contested cells; ties break by earlier primitive position in reading order.

    Key insight:
    - The scattered single pixels are not separate objects: their count simultaneously encodes the tiling length and the overlap priority, so the most-copied primitive also wins collisions.
    """
    patterns = _extract_patterns(np.asarray(grid, dtype=int))
    return _simulate_output(patterns, tuple(np.asarray(grid).shape))


def _block_cells(r: int, c: int) -> list[tuple[int, int]]:
    return [(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)]


def _cell_has_neighbor(cell: tuple[int, int], occupied: set[tuple[int, int]]) -> bool:
    r, c = cell
    for rr in range(r - 1, r + 2):
        for cc in range(c - 1, c + 2):
            if (rr, cc) in occupied:
                return True
    return False


def _singleton_candidates(
    shape: tuple[int, int],
    block_r: int,
    block_c: int,
    occupied: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    h, w = shape
    block = _block_cells(block_r, block_c)
    candidates: list[tuple[int, int]] = []
    for r in range(max(0, block_r - 3), min(h, block_r + 5)):
        for c in range(max(0, block_c - 3), min(w, block_c + 5)):
            if (r, c) in occupied:
                continue
            d = _min_block_distance((r, c), block)
            if d not in {2, 3}:
                continue
            if _cell_has_neighbor((r, c), occupied):
                continue
            candidates.append((r, c))
    random.shuffle(candidates)
    return candidates


def _place_singletons(
    shape: tuple[int, int],
    patterns: list[dict[str, int]],
) -> list[dict[str, int | list[tuple[int, int]]]] | None:
    occupied: set[tuple[int, int]] = set()
    enriched: list[dict[str, int | list[tuple[int, int]]]] = []
    for pattern in patterns:
        for cell in _block_cells(pattern["r"], pattern["c"]):
            occupied.add(cell)
        enriched.append({**pattern, "singletons": []})

    for pattern in sorted(enriched, key=lambda p: -int(p["k"])):
        candidates = _singleton_candidates(shape, int(pattern["r"]), int(pattern["c"]), occupied)
        needed = int(pattern["k"])
        if len(candidates) < needed:
            return None
        chosen: list[tuple[int, int]] = []
        for candidate in candidates:
            if _cell_has_neighbor(candidate, occupied):
                continue
            chosen.append(candidate)
            occupied.add(candidate)
            if len(chosen) == needed:
                break
        if len(chosen) != needed:
            return None
        pattern["singletons"] = chosen
    return enriched


def _pair_layout() -> tuple[int, int]:
    return random.choice([(0, 0), (1, -1), (1, 1)])


def _choose_counts(n_patterns: int) -> list[int]:
    if n_patterns == 2:
        counts = random.sample([1, 2, 3], 2)
    else:
        counts = [1, 2, 3]
    random.shuffle(counts)
    return counts


def _anchor_gap_delta(left_k: int) -> int:
    max_overlap_width = min(3, 2 * left_k - 1)
    overlap_width = random.randint(1, max_overlap_width)
    return 2 * (left_k + 1) - overlap_width


def _attempt_generate() -> np.ndarray | None:
    n_patterns = 2 if random.random() < 0.55 else 3
    colors = random.sample(COLORS, n_patterns)
    counts = _choose_counts(n_patterns)

    if n_patterns == 2:
        pair_counts = counts
        extra_count = None
    else:
        pair_counts = counts[:2]
        extra_count = counts[2]

    h = random.randint(12, 16)
    left_k, right_k = pair_counts
    delta = _anchor_gap_delta(left_k)
    len_left = 2 * (left_k + 1)
    len_right = 2 * (right_k + 1)
    min_w = delta + len_right + 4
    if min_w > 18:
        return None
    w = random.randint(max(13, min_w), 18)

    max_left_c = min(4, w - len_left - 2, w - delta - len_right - 2)
    if max_left_c < 2:
        return None
    left_c = random.randint(2, max_left_c)
    right_c = left_c + delta

    same_or_offset, direction = _pair_layout()
    if same_or_offset == 0:
        left_r = right_r = random.randint(2, h - 4)
    else:
        base_r = random.randint(2, h - 5)
        if direction < 0:
            left_r = base_r + 1
            right_r = base_r
        else:
            left_r = base_r
            right_r = base_r + 1

    pair = [
        {"color": colors[0], "r": left_r, "c": left_c, "k": left_k},
        {"color": colors[1], "r": right_r, "c": right_c, "k": right_k},
    ]

    patterns = pair[:]
    if n_patterns == 3 and extra_count is not None:
        extra_len = 2 * (extra_count + 1)
        possible_rows = [
            r
            for r in range(2, h - 3)
            if all(abs(r - existing["r"]) >= 3 for existing in pair)
        ]
        if not possible_rows:
            return None
        extra_r = random.choice(possible_rows)
        max_extra_c = w - extra_len - 2
        if max_extra_c < 2:
            return None
        extra_c_choices = list(range(2, max_extra_c + 1))
        random.shuffle(extra_c_choices)
        extra_c = None
        for candidate_c in extra_c_choices:
            if any(
                not (
                    candidate_c + 1 < existing["c"] - 1
                    or existing["c"] + 1 < candidate_c - 1
                    or abs(extra_r - existing["r"]) >= 2
                )
                for existing in pair
            ):
                continue
            extra_c = candidate_c
            break
        if extra_c is None:
            return None
        patterns.append({"color": colors[2], "r": extra_r, "c": extra_c, "k": extra_count})

    if not any(_overlap_area(patterns[i], patterns[j]) > 0 for i in range(len(patterns)) for j in range(i + 1, len(patterns))):
        return None

    if max(_overlap_area(patterns[i], patterns[j]) for i in range(len(patterns)) for j in range(i + 1, len(patterns))) > 6:
        return None

    with_singletons = _place_singletons((h, w), patterns)
    if with_singletons is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    for pattern in with_singletons:
        color = int(pattern["color"])
        r = int(pattern["r"])
        c = int(pattern["c"])
        grid[r : r + 2, c : c + 2] = color
        for sr, sc in pattern["singletons"]:
            grid[sr, sc] = color

    expected = _simulate_output(patterns, (h, w))
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid puzzle after {GEN_RETRIES} attempts")
