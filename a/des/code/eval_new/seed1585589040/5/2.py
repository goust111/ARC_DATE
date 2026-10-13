from __future__ import annotations

from collections import Counter, deque

import numpy as np


_DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_DIR8 = _DIR4 + ((1, 1), (1, -1), (-1, 1), (-1, -1))

_BG = 5
_NON_BG_COLORS = tuple(c for c in range(10) if c != _BG)
_PATTERNS = {
    2: ((0, 1, 0, 1),),
    3: ((0, 1, 2, 1), (0, 1, 0, 2), (0, 1, 2, 0)),
    4: ((0, 1, 2, 3),),
}
_RNG = np.random.default_rng()


def _corner_bg(grid: np.ndarray) -> int:
    corners = [int(grid[0, 0]), int(grid[0, -1]), int(grid[-1, 0]), int(grid[-1, -1])]
    return int(Counter(corners).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in _DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _touches_border(coords: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _read_legend(grid: np.ndarray, bg: int) -> list[int]:
    legend_block = grid[-2:, :]
    best_row = None
    best_cols: list[int] = []
    for rr in range(2):
        cols = [c for c in range(legend_block.shape[1]) if int(legend_block[rr, c]) != bg]
        if not cols:
            continue
        if best_row is None or len(cols) > len(best_cols):
            best_row = rr
            best_cols = cols
    assert best_row is not None, "legend row must exist in the bottom two rows"
    colors = [int(legend_block[best_row, c]) for c in best_cols]
    assert len(colors) == len(set(colors)), "legend colors must be distinct"
    return colors


def _ordered_boundary_colors(body: np.ndarray, bg: int, region: list[tuple[int, int]]) -> list[int]:
    h, w = body.shape
    region_mask = np.zeros((h, w), dtype=bool)
    for r, c in region:
        region_mask[r, c] = True

    boundary = set()
    for r, c in region:
        for dr, dc in _DIR8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not bool(region_mask[nr, nc]) and int(body[nr, nc]) != bg:
                boundary.add((nr, nc))

    counts: Counter[int] = Counter()
    first_pos: dict[int, tuple[int, int]] = {}
    for r, c in sorted(boundary):
        color = int(body[r, c])
        counts[color] += 1
        first_pos.setdefault(color, (r, c))

    ordered = sorted(counts, key=lambda color: (-counts[color], first_pos[color][0], first_pos[color][1]))
    return [int(color) for color in ordered]


def _fill_region_layers(out: np.ndarray, region: list[tuple[int, int]], fill_colors: list[int]) -> None:
    if not fill_colors:
        return

    h, w = out.shape
    region_mask = np.zeros((h, w), dtype=bool)
    for r, c in region:
        region_mask[r, c] = True

    dist = -np.ones((h, w), dtype=int)
    q = deque()
    for r, c in region:
        if any(
            not (0 <= r + dr < h and 0 <= c + dc < w) or not bool(region_mask[r + dr, c + dc])
            for dr, dc in _DIR4
        ):
            dist[r, c] = 0
            q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in _DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and bool(region_mask[nr, nc]) and int(dist[nr, nc]) == -1:
                dist[nr, nc] = int(dist[r, c] + 1)
                q.append((nr, nc))

    for r, c in region:
        layer = int(dist[r, c])
        if 0 <= layer < len(fill_colors):
            out[r, c] = int(fill_colors[layer])


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 3, "expected a 2D grid with a two-row legend"

    bg = _corner_bg(g)
    legend = _read_legend(g, bg)
    body = g[:-2, :].copy()
    out = body.copy()

    for region in _components(body == bg):
        if _touches_border(region, body.shape):
            continue
        ordered = _ordered_boundary_colors(body, bg, region)
        usable = min(len(ordered), len(legend))
        _fill_region_layers(out, region, legend[:usable])

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed grey region with legend colors ordered by its boundary color-length ranking.

    Input structure:
    - Grey is the background color. Above the legend there are one or more closed outlines made from 2-4 colored straight boundary segments.
    - Each outline surrounds a grey interior, and different outlines are separated by background.
    - The bottom two rows form a legend area; one of those rows contains distinct colored pixels read from left to right.

    Rule:
    - Remove the bottom two legend rows in the output.
    - For every enclosed grey region, count how many boundary pixels belong to each surrounding color and sort those colors from most frequent to least frequent.
    - Match rank 1, rank 2, ... to legend positions 1, 2, ... and then fill the interior from the boundary inward in successive 4-connected layers using those legend colors. If the region is deeper than the available ranked colors, the remaining center stays grey.

    Key insight:
    - The boundary colors are not copied directly: their length ranking selects the legend order, and that ordered legend drives the inward layer-by-layer fill.
    """
    return _transform(grid)


def _perimeter_segments(h: int, w: int) -> tuple[list[tuple[int, int]], tuple[int, int, int, int]]:
    path = [(0, c) for c in range(w)]
    path += [(r, w - 1) for r in range(1, h)]
    path += [(h - 1, c) for c in range(w - 2, -1, -1)]
    path += [(r, 0) for r in range(h - 2, 0, -1)]
    lengths = (w, h - 1, w - 1, h - 2)
    return path, lengths


def _sample_inner_small(k: int, max_small: int) -> int | None:
    modes: list[np.ndarray] = []
    shallow_hi = min(max_small, 2 * k - 2)
    if 1 <= shallow_hi:
        modes.append(np.arange(1, shallow_hi + 1))
    exact_lo, exact_hi = 2 * k - 1, min(max_small, 2 * k)
    if exact_lo <= exact_hi:
        modes.append(np.arange(exact_lo, exact_hi + 1))
    deep_lo = 2 * k + 1
    if deep_lo <= max_small:
        modes.append(np.arange(deep_lo, max_small + 1))
    if not modes:
        return None
    choices = modes[int(_RNG.integers(0, len(modes)))]
    return int(choices[int(_RNG.integers(0, len(choices)))])


def _sample_shape_spec(legend_len: int, max_outer_h: int, max_outer_w: int) -> dict[str, object] | None:
    if max_outer_h < 3 or max_outer_w < 3:
        return None
    max_small = min(max_outer_h - 2, max_outer_w - 2)
    if max_small < 1:
        return None

    for _ in range(200):
        k = int(_RNG.integers(2, min(4, legend_len) + 1))
        inner_small = _sample_inner_small(k, max_small)
        if inner_small is None:
            continue

        if bool(_RNG.integers(0, 2)):
            inner_h = inner_small
            if max_outer_w - 2 < inner_small:
                continue
            inner_w = int(_RNG.integers(inner_small, max_outer_w - 1))
            if inner_h > max_outer_h - 2:
                continue
        else:
            inner_w = inner_small
            if max_outer_h - 2 < inner_small:
                continue
            inner_h = int(_RNG.integers(inner_small, max_outer_h - 1))
            if inner_w > max_outer_w - 2:
                continue

        h, w = inner_h + 2, inner_w + 2
        pattern = _PATTERNS[k][int(_RNG.integers(0, len(_PATTERNS[k])))]
        colors = _RNG.choice(_NON_BG_COLORS, size=k, replace=False).tolist()
        seg_colors = [int(colors[idx]) for idx in pattern]
        _, seg_lengths = _perimeter_segments(h, w)

        totals: Counter[int] = Counter()
        for color, length in zip(seg_colors, seg_lengths):
            totals[int(color)] += int(length)
        if len(totals) != k or len(set(totals.values())) != k:
            continue

        return {
            "h": h,
            "w": w,
            "segment_colors": seg_colors,
        }
    return None


def _bbox_far_enough(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 2) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0


def _paint_shape(grid: np.ndarray, r0: int, c0: int, spec: dict[str, object]) -> None:
    h = int(spec["h"])
    w = int(spec["w"])
    path, seg_lengths = _perimeter_segments(h, w)
    seg_colors = [int(color) for color in spec["segment_colors"]]
    idx = 0
    for color, seg_len in zip(seg_colors, seg_lengths):
        for _ in range(seg_len):
            rr, cc = path[idx]
            grid[r0 + rr, c0 + cc] = color
            idx += 1


def _enclosed_regions(body: np.ndarray) -> list[list[tuple[int, int]]]:
    return [comp for comp in _components(body == _BG) if not _touches_border(comp, body.shape)]


def _gen_attempt() -> np.ndarray | None:
    legend_len = int(_RNG.integers(2, 5))
    object_count = int(_RNG.choice(np.array([1, 2, 3])))

    min_h = {1: 8, 2: 12, 3: 16}[object_count]
    min_w = {1: 10, 2: 16, 3: 20}[object_count]
    work_h = int(_RNG.integers(min_h, 29))
    work_w = int(_RNG.integers(max(min_w, legend_len + 4), 31))

    grid = np.full((work_h + 2, work_w), _BG, dtype=int)
    size_cap = {1: 12, 2: 10, 3: 8}[object_count]
    placed_bboxes: list[tuple[int, int, int, int]] = []

    for _ in range(object_count):
        placed = False
        for _ in range(120):
            spec = _sample_shape_spec(
                legend_len=legend_len,
                max_outer_h=min(size_cap, work_h - 2),
                max_outer_w=min(size_cap, work_w - 2),
            )
            if spec is None:
                return None

            h = int(spec["h"])
            w = int(spec["w"])
            if work_h - h <= 1 or work_w - w <= 1:
                continue
            r0 = int(_RNG.integers(1, work_h - h))
            c0 = int(_RNG.integers(1, work_w - w))
            bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
            if any(not _bbox_far_enough(bbox, other, pad=2) for other in placed_bboxes):
                continue

            _paint_shape(grid, r0, c0, spec)
            placed_bboxes.append(bbox)
            placed = True
            break
        if not placed:
            return None

    legend_colors = _RNG.choice(_NON_BG_COLORS, size=legend_len, replace=False)
    legend_row = work_h + int(_RNG.integers(0, 2))
    if work_w - legend_len <= 1:
        return None
    start_c = int(_RNG.integers(1, work_w - legend_len))
    grid[legend_row, start_c : start_c + legend_len] = legend_colors

    body = grid[:-2, :]
    enclosed = _enclosed_regions(body)
    if len(enclosed) != object_count:
        return None

    out = _transform(grid)
    if out.shape != body.shape:
        return None
    if np.array_equal(out, body):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after 300 attempts")
