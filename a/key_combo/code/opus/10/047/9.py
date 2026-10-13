from __future__ import annotations

import random
import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7

LINE_COLORS = (BLUE, GREEN)
FILL_BY_TOUCH = {
    frozenset({BLUE}): YELLOW,
    frozenset({GREEN}): RED,
    frozenset({BLUE, GREEN}): ORANGE,
}

GEN_ATTEMPTS = 400
GEN_LINE_ATTEMPTS = 500
GEN_SHAPES = [(12, 13), (13, 13), (13, 14), (14, 15), (15, 15)]
GEN_PROFILES = ("mixed_only", "blue_and_mixed", "green_and_mixed")
GEN_PROFILE_WEIGHTS = (1, 2, 2)


def _sign(x: int) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def _diag_aligned(p: tuple[int, int], q: tuple[int, int]) -> bool:
    return p != q and abs(int(p[0]) - int(q[0])) == abs(int(p[1]) - int(q[1]))


def _sorted_pair(p: tuple[int, int], q: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return (p, q) if p <= q else (q, p)


def _pair_same_color(points: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    pts = sorted((int(r), int(c)) for r, c in points)
    if len(pts) % 2 != 0:
        raise ValueError("same-color points must be even")
    candidates = []
    for i, p in enumerate(pts):
        for j in range(i + 1, len(pts)):
            q = pts[j]
            if not _diag_aligned(p, q):
                continue
            a, b = _sorted_pair(p, q)
            candidates.append((abs(a[0] - b[0]), a[0], a[1], b[0], b[1], i, j))
    candidates.sort()

    used = [False] * len(pts)
    pairs = []
    while not all(used):
        choice = None
        for cand in candidates:
            _, _, _, _, _, i, j = cand
            if not used[i] and not used[j]:
                choice = cand
                break
        if choice is None:
            raise ValueError("could not produce an unambiguous diagonal pairing")
        _, _, _, _, _, i, j = choice
        used[i] = True
        used[j] = True
        pairs.append(_sorted_pair(pts[i], pts[j]))
    return pairs


def _line_cells(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
    if not _diag_aligned(p, q):
        raise ValueError("segment endpoints must lie on a 45-degree diagonal")
    dr = _sign(q[0] - p[0])
    dc = _sign(q[1] - p[1])
    steps = abs(int(q[0]) - int(p[0]))
    return [(int(p[0]) + k * dr, int(p[1]) + k * dc) for k in range(steps + 1)]


def _segment_key(segment: tuple[int, tuple[int, int], tuple[int, int]]) -> tuple[int, int, int, int, int]:
    color, p, q = segment
    a, b = _sorted_pair(p, q)
    return int(a[0]), int(a[1]), int(b[0]), int(b[1]), int(color)


def _infer_segments(grid: np.ndarray) -> list[tuple[int, tuple[int, int], tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    segments = []
    for color in LINE_COLORS:
        coords = [tuple(map(int, rc)) for rc in np.argwhere(g == color)]
        if not coords:
            continue
        for p, q in _pair_same_color(coords):
            segments.append((int(color), p, q))
    segments.sort(key=_segment_key)
    return segments


def _render_segments(
    shape: tuple[int, int],
    segments: list[tuple[int, tuple[int, int], tuple[int, int]]],
    base: np.ndarray | None = None,
) -> np.ndarray:
    out = np.zeros(shape, dtype=int) if base is None else np.array(base, dtype=int, copy=True)
    for color, p, q in sorted(segments, key=_segment_key):
        for r, c in _line_cells(p, q):
            if out[r, c] == BLACK:
                out[r, c] = int(color)
    return out


def _zero_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(comp)
    return components


def _touch_colors(region: list[tuple[int, int]], drawn: np.ndarray) -> frozenset[int]:
    h, w = drawn.shape
    region_set = set(region)
    touches: set[int] = set()
    for r, c in region:
        is_boundary = False
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or (nr, nc) not in region_set:
                is_boundary = True
                break
        if not is_boundary:
            continue
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and int(drawn[nr, nc]) in LINE_COLORS:
                    touches.add(int(drawn[nr, nc]))
    return frozenset(touches)


def _analyze_drawn(drawn: np.ndarray) -> tuple[list[dict], list[dict]]:
    h, w = drawn.shape
    zero_mask = drawn == BLACK
    corner_cells = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    large_regions = []
    small_regions = []

    for comp in _zero_components(zero_mask):
        comp_set = set(comp)
        if any(corner in comp_set for corner in corner_cells):
            continue
        touches = _touch_colors(comp, drawn)
        info = {
            "coords": comp,
            "area": int(len(comp)),
            "touches": touches,
            "fill": FILL_BY_TOUCH.get(touches),
        }
        if len(comp) >= 3 and info["fill"] is not None:
            large_regions.append(info)
        else:
            small_regions.append(info)
    return large_regions, small_regions


def _solve_from_segments(shape: tuple[int, int], segments: list[tuple[int, tuple[int, int], tuple[int, int]]]) -> tuple[np.ndarray, np.ndarray, list[dict], list[dict]]:
    drawn = _render_segments(shape, segments)
    large_regions, small_regions = _analyze_drawn(drawn)
    out = drawn.copy()
    for region in large_regions:
        fill = int(region["fill"])
        for r, c in region["coords"]:
            out[r, c] = fill
    return drawn, out, large_regions, small_regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect paired same-color diagonal endpoints, then fill the enclosed black regions by which line colors touch them.

    Input structure:
    - The grid has a black background with scattered single blue and green pixels.
    - Each color appears an even number of times, and valid pairs lie on shared 45-degree diagonals.
    - The input contains only endpoints; the diagonal line segments themselves are absent.

    Rule:
    - Pair same-color pixels by nearest diagonal distance, breaking ties by the top-left pair first.
    - Draw every pair as a full 45-degree segment in its own color, keeping line pixels unchanged thereafter.
    - After all segments are drawn, find black 4-connected regions closed off by the segments and/or the grid boundary; ignore regions smaller than 3 cells.
    - Fill a closed region yellow if it touches only blue line pixels, red if it touches only green line pixels, and orange if it touches both.

    Key insight:
    - The regions do not exist in the input; they emerge only after all diagonal pair-connections are completed, so pairing, drawing, enclosure detection, and color-mapping must happen in that order.
    """
    g = np.asarray(grid, dtype=int)
    segments = _infer_segments(g)
    drawn = _render_segments(g.shape, segments, base=g)
    large_regions, _small_regions = _analyze_drawn(drawn)
    out = drawn.copy()
    for region in large_regions:
        fill = int(region["fill"])
        for r, c in region["coords"]:
            out[r, c] = fill
    return out


def _full_backslash(h: int, w: int, delta: int) -> list[tuple[int, int]]:
    cells = []
    for r in range(max(0, delta), min(h - 1, w - 1 + delta) + 1):
        c = r - delta
        if 0 <= c < w:
            cells.append((int(r), int(c)))
    return cells


def _full_slash(h: int, w: int, sigma: int) -> list[tuple[int, int]]:
    cells = []
    for r in range(max(0, sigma - (w - 1)), min(h - 1, sigma) + 1):
        c = sigma - r
        if 0 <= c < w:
            cells.append((int(r), int(c)))
    return cells


def _apply_transform(grid: np.ndarray, name: str) -> np.ndarray:
    if name == "identity":
        return grid.copy()
    if name == "flipud":
        return np.flipud(grid)
    if name == "fliplr":
        return np.fliplr(grid)
    if name == "rot180":
        return np.rot90(grid, 2)
    if name == "transpose":
        return grid.T.copy()
    if name == "rot90":
        return np.rot90(grid, 1)
    if name == "rot270":
        return np.rot90(grid, 3)
    raise ValueError(f"unknown transform: {name}")


def _segment_cells(segment: tuple[int, tuple[int, int], tuple[int, int]]) -> list[tuple[int, int]]:
    return _line_cells(segment[1], segment[2])


def _segments_to_input(shape: tuple[int, int], segments: list[tuple[int, tuple[int, int], tuple[int, int]]]) -> np.ndarray | None:
    h, w = shape
    corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    endpoints: list[tuple[int, tuple[int, int]]] = []
    endpoint_cells = set()
    for color, p, q in segments:
        if p in corners or q in corners or p == q:
            return None
        if p in endpoint_cells or q in endpoint_cells:
            return None
        endpoint_cells.add(p)
        endpoint_cells.add(q)
        endpoints.append((int(color), p))
        endpoints.append((int(color), q))

    for segment in segments:
        for cell in _segment_cells(segment)[1:-1]:
            if cell in endpoint_cells:
                return None

    inp = np.zeros(shape, dtype=int)
    for color, cell in endpoints:
        inp[cell] = int(color)

    intended = sorted((_segment_key(seg), seg) for seg in segments)
    try:
        inferred = _infer_segments(inp)
    except ValueError:
        return None
    if len(inferred) != len(segments):
        return None
    if [_segment_key(seg) for seg in inferred] != [_segment_key(seg) for _, seg in intended]:
        return None
    return inp


def _candidate_deltas(h: int, w: int) -> list[int]:
    candidates = []
    for delta in range(-(w - 2), h - 1):
        cells = _full_backslash(h, w, delta)
        if len(cells) < 6:
            continue
        if cells[0][0] in (0, h - 1) and cells[0][1] in (0, w - 1):
            continue
        if cells[-1][0] in (0, h - 1) and cells[-1][1] in (0, w - 1):
            continue
        candidates.append(int(delta))
    return candidates


def _candidate_sigmas(h: int, w: int) -> list[int]:
    candidates = []
    for sigma in range(1, h + w - 3):
        cells = _full_slash(h, w, sigma)
        if len(cells) < 6:
            continue
        if cells[0][0] in (0, h - 1) and cells[0][1] in (0, w - 1):
            continue
        if cells[-1][0] in (0, h - 1) and cells[-1][1] in (0, w - 1):
            continue
        candidates.append(int(sigma))
    return candidates


def _profile_ok(profile: str, regions: list[dict]) -> bool:
    touches = {region["touches"] for region in regions}
    if not (2 <= len(regions) <= 4):
        return False
    if profile == "mixed_only":
        return touches == {frozenset({BLUE, GREEN})}
    if profile == "blue_and_mixed":
        return frozenset({BLUE}) in touches and frozenset({BLUE, GREEN}) in touches and frozenset({GREEN}) not in touches
    if profile == "green_and_mixed":
        return frozenset({GREEN}) in touches and frozenset({BLUE, GREEN}) in touches and frozenset({BLUE}) not in touches
    return False


def _try_generate_profile(h: int, w: int, profile: str, rng: random.Random) -> np.ndarray | None:
    deltas = _candidate_deltas(h, w)
    sigmas = _candidate_sigmas(h, w)
    if len(deltas) < 3 or len(sigmas) < 3:
        return None

    delta_pairs = [(a, b) for a in deltas for b in deltas if a < b and 2 <= b - a <= 5]
    sigma_pairs = [(a, b) for a in sigmas for b in sigmas if a < b and 2 <= b - a <= 5]
    if not delta_pairs or not sigma_pairs:
        return None

    rng.shuffle(delta_pairs)
    rng.shuffle(sigma_pairs)
    transform_names = ["identity", "flipud", "fliplr", "rot180", "transpose", "rot90", "rot270"]

    for _ in range(GEN_LINE_ATTEMPTS):
        d1, d2 = delta_pairs[rng.randrange(len(delta_pairs))]
        s1, s2 = sigma_pairs[rng.randrange(len(sigma_pairs))]
        segments = [
            (BLUE, _full_backslash(h, w, d1)[0], _full_backslash(h, w, d1)[-1]),
            (BLUE, _full_backslash(h, w, d2)[0], _full_backslash(h, w, d2)[-1]),
            (GREEN, _full_slash(h, w, s1)[0], _full_slash(h, w, s1)[-1]),
            (GREEN, _full_slash(h, w, s2)[0], _full_slash(h, w, s2)[-1]),
        ]

        if profile == "blue_and_mixed":
            s3 = rng.choice(sigmas)
            extra_cells = _full_slash(h, w, s3)
            segments.append((BLUE, extra_cells[0], extra_cells[-1]))
        elif profile == "green_and_mixed":
            d3 = rng.choice(deltas)
            extra_cells = _full_backslash(h, w, d3)
            segments.append((GREEN, extra_cells[0], extra_cells[-1]))

        inp = _segments_to_input((h, w), segments)
        if inp is None:
            continue

        _drawn, expected, large_regions, small_regions = _solve_from_segments((h, w), segments)
        if not _profile_ok(profile, large_regions):
            continue
        if not small_regions:
            continue
        if not any(region["area"] >= 5 for region in large_regions):
            continue
        if len({tuple(sorted(region["touches"])) for region in large_regions}) == 1 and profile != "mixed_only":
            continue

        transform_name = rng.choice(transform_names)
        transformed = _apply_transform(inp, transform_name)
        expected_transformed = _apply_transform(expected, transform_name)
        if not np.array_equal(main(transformed), expected_transformed):
            continue
        return transformed
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        h, w = rng.choice(GEN_SHAPES)
        profile = rng.choices(GEN_PROFILES, weights=GEN_PROFILE_WEIGHTS, k=1)[0]
        grid = _try_generate_profile(h, w, profile, rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
