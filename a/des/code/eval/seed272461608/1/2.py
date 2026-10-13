from __future__ import annotations

import numpy as np


LIGHT_COLOR = 4
WALL_COLOR = 5
OVERLAP_COLOR = 9
_OPS = ("identity", "flip_lr", "rot90", "rot270")


def _background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[np.argmax(counts)])


def _apply_op(grid: np.ndarray, op: str) -> np.ndarray:
    if op == "identity":
        return np.array(grid, dtype=int, copy=True)
    if op == "flip_lr":
        return np.fliplr(grid)
    if op == "rot90":
        return np.rot90(grid, 1)
    if op == "rot270":
        return np.rot90(grid, 3)
    raise ValueError(op)


def _inverse_op(op: str) -> str:
    if op == "identity":
        return "identity"
    if op == "flip_lr":
        return "flip_lr"
    if op == "rot90":
        return "rot270"
    if op == "rot270":
        return "rot90"
    raise ValueError(op)


def _line_signature(coords: np.ndarray) -> tuple[str, int]:
    rows = np.unique(coords[:, 0])
    cols = np.unique(coords[:, 1])
    if len(rows) == 1:
        return "horizontal", int(rows[0])
    if len(cols) == 1:
        return "vertical", int(cols[0])
    raise ValueError("expected a straight line")


def _canonicalizing_op(grid: np.ndarray) -> str:
    light = np.argwhere(np.asarray(grid, dtype=int) == LIGHT_COLOR)
    wall = np.argwhere(np.asarray(grid, dtype=int) == WALL_COLOR)
    if light.size == 0 or wall.size == 0:
        raise ValueError("missing light line or wall line")

    light_orient, light_pos = _line_signature(light)
    wall_orient, wall_pos = _line_signature(wall)
    if light_orient != wall_orient:
        raise ValueError("light line and wall must be parallel")

    if light_orient == "vertical":
        return "identity" if light_pos < wall_pos else "flip_lr"
    return "rot90" if light_pos < wall_pos else "rot270"


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _fragment_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if visited[r, c] or color == bg or color in {LIGHT_COLOR, WALL_COLOR}:
                continue

            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(g[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "area": len(coords),
                }
            )

    return comps


def _solve_canonical_right(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    wall_cols = np.unique(np.argwhere(g == WALL_COLOR)[:, 1])
    if len(wall_cols) != 1:
        raise ValueError("canonical scene must have one vertical wall")
    wall_col = int(wall_cols[0])

    pieces = _fragment_components(g, bg)
    pieces = sorted(
        pieces,
        key=lambda comp: (
            wall_col - int(comp["bbox"][3]) - 1,
            int(comp["bbox"][0]),
            int(comp["bbox"][1]),
            int(comp["color"]),
        ),
    )

    out = np.array(g, copy=True)
    fragment_mask = (g != bg) & (g != LIGHT_COLOR) & (g != WALL_COLOR)
    out[fragment_mask] = bg

    h, w = g.shape
    paint_count = np.zeros((h, w), dtype=int)
    paint_color = np.full((h, w), bg, dtype=int)
    depth = np.zeros(h, dtype=int)

    for comp in pieces:
        r0, c0, r1, c1 = [int(v) for v in comp["bbox"]]
        color = int(comp["color"])
        width = c1 - c0 + 1

        front_depth = int(depth[r0 : r1 + 1].min()) + 1
        right = wall_col - front_depth
        left = right - width + 1

        for rr in range(r0, r1 + 1):
            for cc in range(left, right + 1):
                if paint_count[rr, cc] == 0:
                    paint_color[rr, cc] = color
                else:
                    paint_color[rr, cc] = OVERLAP_COLOR
                paint_count[rr, cc] += 1

        depth[r0 : r1 + 1] = np.maximum(depth[r0 : r1 + 1], front_depth + width - 1)

    out[paint_count > 0] = paint_color[paint_count > 0]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stack colored rectangles against the wall opposite the light line.

    Pieces are processed from nearest to farthest relative to the wall. In each covered row/column,
    the current stack depth determines where the front edge stops; if a later piece spans both stacked
    and empty rows/columns, it can partially overlap earlier pieces, and those cells become maroon.
    """
    g = np.asarray(grid, dtype=int)
    op = _canonicalizing_op(g)
    canonical = _apply_op(g, op)
    solved = _solve_canonical_right(canonical)
    return _apply_op(solved, _inverse_op(op))


def _random_gaps(total: int, parts: int, rng: np.random.Generator) -> list[int]:
    extra = rng.multinomial(total, np.full(parts, 1.0 / parts))
    return [int(v) for v in extra]


def _sample_widths(depth: int, count: int, rng: np.random.Generator) -> list[int] | None:
    budget = depth - (count + 1)
    if budget < count:
        return None
    for _ in range(200):
        widths = rng.integers(1, 5, size=count)
        if int(widths.sum()) <= budget:
            return [int(v) for v in widths]
    return None


def _place_disjoint_rows(height: int, sizes: list[int], rng: np.random.Generator) -> list[int] | None:
    starts = [0] * len(sizes)
    blocked = np.zeros(height, dtype=bool)
    for idx in rng.permutation(len(sizes)):
        size = int(sizes[idx])
        candidates: list[int] = []
        for start in range(0, height - size + 1):
            lo = max(0, start - 1)
            hi = min(height, start + size + 1)
            if not blocked[lo:hi].any():
                candidates.append(start)
        if not candidates:
            return None
        start = int(rng.choice(candidates))
        starts[idx] = start
        lo = max(0, start - 1)
        hi = min(height, start + size + 1)
        blocked[lo:hi] = True
    return starts


def _place_overlap_rows(height: int, sizes: list[int], rng: np.random.Generator) -> list[int] | None:
    if len(sizes) < 2:
        return None

    s0 = int(sizes[0])
    s1 = int(sizes[1])
    pair_candidates: list[tuple[int, int]] = []
    for a in range(0, height - s0 + 1):
        a1 = a + s0 - 1
        for b in range(0, height - s1 + 1):
            b1 = b + s1 - 1
            overlaps = not (a1 < b or b1 < a)
            b_has_outside = not (a <= b and b1 <= a1)
            if overlaps and b_has_outside:
                pair_candidates.append((a, b))
    if not pair_candidates:
        return None

    start0, start1 = pair_candidates[int(rng.integers(len(pair_candidates)))]
    starts = [start0, start1] + [0] * (len(sizes) - 2)

    blocked = np.zeros(height, dtype=bool)
    union_lo = min(start0, start1)
    union_hi = max(start0 + s0 - 1, start1 + s1 - 1)
    blocked[max(0, union_lo - 1) : min(height, union_hi + 2)] = True

    for idx in range(2, len(sizes)):
        size = int(sizes[idx])
        candidates: list[int] = []
        for start in range(0, height - size + 1):
            lo = max(0, start - 1)
            hi = min(height, start + size + 1)
            if not blocked[lo:hi].any():
                candidates.append(start)
        if not candidates:
            return None
        start = int(rng.choice(candidates))
        starts[idx] = start
        lo = max(0, start - 1)
        hi = min(height, start + size + 1)
        blocked[lo:hi] = True

    return starts


def _canonical_scene(rng: np.random.Generator, want_overlap: bool) -> np.ndarray | None:
    height = int(rng.integers(10, 19))
    width = int(rng.integers(14, 23))
    bg = int(rng.choice([0, 1, 2, 3, 6, 7, 8]))

    light_col = int(rng.integers(1, 4))
    wall_col = int(rng.integers(max(light_col + 8, width - 5), width - 1))
    band_depth = wall_col - light_col - 1
    if band_depth < 6:
        return None

    max_count = min(4, max(2, (band_depth - 1) // 2))
    if max_count < 2:
        return None
    count = int(rng.integers(2, max_count + 1))

    widths = _sample_widths(band_depth, count, rng)
    if widths is None:
        return None
    heights = [int(v) for v in rng.integers(1, 5, size=count)]

    if want_overlap:
        row_starts = _place_overlap_rows(height, heights, rng)
    else:
        row_starts = _place_disjoint_rows(height, heights, rng)
    if row_starts is None:
        return None

    min_used = sum(widths) + (count + 1)
    extra = band_depth - min_used
    if extra < 0:
        return None
    gaps = [v + 1 for v in _random_gaps(extra, count + 1, rng)]

    front_gaps: list[int] = []
    cursor = gaps[0]
    for idx, width_i in enumerate(widths):
        front_gaps.append(cursor)
        cursor += width_i + gaps[idx + 1]

    palette = [c for c in range(10) if c not in {bg, LIGHT_COLOR, WALL_COLOR}]
    if len(palette) < count:
        return None
    colors = [int(v) for v in rng.choice(palette, size=count, replace=False)]

    grid = np.full((height, width), bg, dtype=int)
    grid[:, light_col] = LIGHT_COLOR
    grid[:, wall_col] = WALL_COLOR

    for start_r, frag_h, frag_w, gap, color in zip(row_starts, heights, widths, front_gaps, colors):
        right = wall_col - gap - 1
        left = right - frag_w + 1
        if left <= light_col or right >= wall_col:
            return None
        grid[start_r : start_r + frag_h, left : right + 1] = color

    return grid


def _choose_output_op(rng: np.random.Generator) -> str:
    return _OPS[int(rng.integers(len(_OPS)))]


def _fragment_distance_signature(grid: np.ndarray) -> list[int]:
    canonical = _apply_op(np.asarray(grid, dtype=int), _canonicalizing_op(np.asarray(grid, dtype=int)))
    bg = _background_color(canonical)
    wall_col = int(np.unique(np.argwhere(canonical == WALL_COLOR)[:, 1])[0])
    return sorted(
        wall_col - int(comp["bbox"][3]) - 1
        for comp in _fragment_components(canonical, bg)
    )


def _valid_scene(grid: np.ndarray, want_overlap: bool) -> bool:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    if int((g == bg).sum()) <= g.size // 2:
        return False

    try:
        out = main(g)
    except Exception:
        return False

    if np.array_equal(out, g):
        return False

    overlap_present = bool(np.any(out == OVERLAP_COLOR))
    if overlap_present != want_overlap:
        return False

    try:
        distances = _fragment_distance_signature(g)
    except Exception:
        return False
    if len(distances) < 2 or len(set(distances)) != len(distances):
        return False

    canonical = _apply_op(g, _canonicalizing_op(g))
    bg_c = _background_color(canonical)
    for comp in _fragment_components(canonical, bg_c):
        r0, c0, r1, c1 = [int(v) for v in comp["bbox"]]
        if comp["area"] != (r1 - r0 + 1) * (c1 - c0 + 1):
            return False
        if r1 - r0 + 1 > 4 or c1 - c0 + 1 > 4:
            return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    want_overlap = bool(rng.integers(2))
    canonical = _canonical_scene(rng, want_overlap)
    if canonical is None:
        return None
    transformed = _apply_op(canonical, _choose_output_op(rng))
    if not _valid_scene(transformed, want_overlap):
        return None
    return transformed


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after 400 attempts")
