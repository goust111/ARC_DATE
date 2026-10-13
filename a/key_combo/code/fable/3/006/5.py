from __future__ import annotations

import random

import numpy as np


def _mode(values: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(values, dtype=int), return_counts=True)
    top = int(counts.max())
    assert int(np.count_nonzero(counts == top)) == 1, "mode must be unique"
    return int(vals[int(np.argmax(counts))])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size, "expected a non-empty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _distance_to_rect_boundary(h: int, w: int) -> np.ndarray:
    rows = np.arange(int(h))
    cols = np.arange(int(w))
    dr = np.minimum(rows, int(h) - 1 - rows)[:, None]
    dc = np.minimum(cols, int(w) - 1 - cols)[None, :]
    return np.minimum(dr, dc).astype(int)


def _wall_lines(grid: np.ndarray) -> tuple[int, list[int], list[int]]:
    """Find the color whose cells are exactly full-height/full-width separator lines."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[int, list[int], list[int]]] = []

    for color in map(int, np.unique(g)):
        rows = [r for r in range(h) if bool(np.all(g[r, :] == color))]
        cols = [c for c in range(w) if bool(np.all(g[:, c] == color))]
        if not rows and not cols:
            continue

        line_mask = np.zeros((h, w), dtype=bool)
        if rows:
            line_mask[rows, :] = True
        if cols:
            line_mask[:, cols] = True

        if bool(np.all((g != color) | line_mask)):
            candidates.append((color, rows, cols))

    assert len(candidates) == 1, "expected one unambiguous separator-line color"
    return candidates[0]


def _spans_between_lines(size: int, lines: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    prev = -1
    for line in sorted(map(int, lines)):
        assert line - prev > 1, "separator lines must leave non-empty regions"
        spans.append((prev + 1, line - 1))
        prev = line
    assert int(size) - prev > 1, "separator lines must leave non-empty regions"
    spans.append((prev + 1, int(size) - 1))
    return spans


def _partition_slices(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    wall, wall_rows, wall_cols = _wall_lines(grid)
    assert wall_rows or wall_cols, "expected at least one separator line"
    h, w = grid.shape
    row_spans = _spans_between_lines(h, wall_rows)
    col_spans = _spans_between_lines(w, wall_cols)
    return wall, row_spans, col_spans


def _read_sprite_blueprint(region: np.ndarray) -> tuple[int, list[tuple[int, int]]]:
    """Return the region base color and the sprite's outer-to-inner (color, thickness) list."""
    base = _mode(region)
    mask = region != base
    assert bool(mask.any()), "each partition must contain one blueprint sprite"

    r0, c0, r1, c1 = _bbox(mask)
    patch = region[r0 : r1 + 1, c0 : c1 + 1]
    side_h, side_w = patch.shape
    assert side_h == side_w, "blueprint sprite must have a square bounding box"
    assert bool(np.all(patch != base)), "sprite bounding box must be filled by sprite colors"

    outside = mask.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    assert not bool(outside.any()), "expected exactly one sprite in each partition"

    dist = _distance_to_rect_boundary(side_h, side_w)
    ring_colors: list[int] = []
    for d in range(int(dist.max()) + 1):
        vals = np.unique(patch[dist == d])
        assert len(vals) == 1, "each sprite ring must be monochrome"
        ring_colors.append(int(vals[0]))

    seq: list[tuple[int, int]] = []
    for color in ring_colors:
        if seq and seq[-1][0] == color:
            prev_color, prev_thick = seq[-1]
            seq[-1] = (prev_color, prev_thick + 1)
        else:
            seq.append((int(color), 1))

    assert seq, "expected a non-empty layer sequence"
    layer_colors = [color for color, _thick in seq]
    assert len(set(layer_colors)) == len(layer_colors), "sprite layer colors must be distinct"
    assert base not in layer_colors, "sprite colors must differ from the partition base"
    return int(base), seq


def _expanded_cycle(seq: list[tuple[int, int]]) -> list[int]:
    cycle: list[int] = []
    for color, thick in seq:
        assert int(thick) > 0, "layer thickness must be positive"
        cycle.extend([int(color)] * int(thick))
    assert cycle, "layer cycle must not be empty"
    return cycle


def _paint_region(h: int, w: int, seq: list[tuple[int, int]]) -> np.ndarray:
    cycle = _expanded_cycle(seq)
    dist = _distance_to_rect_boundary(h, w)
    out = np.empty((int(h), int(w)), dtype=int)
    period = len(cycle)
    for d in range(int(dist.max()) + 1):
        out[dist == d] = cycle[d % period]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each partition's small concentric square blueprint into full-room concentric bands.

    Input structure:
    - Full-height and full-width separator lines of one color divide the grid into rectangular partitions.
    - Every partition has a single dominant base color and one filled square concentric sprite.
    - The sprite's monochrome rings, read from outside to inside, give layer colors and ring thicknesses.

    Rule:
    - Keep all separator-line cells unchanged.
    - For each partition, read the sprite as an outer-to-inner sequence of (color, thickness).
    - Repaint the whole partition by Chebyshev distance from that partition's own boundary, cycling the sequence when its thicknesses are exhausted.

    Key insight:
    - The small sprite is a movable decorating blueprint; its position disappears, while its layer order and layer thicknesses scale up to fill the entire partition.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must be at most 30x30"

    wall, row_spans, col_spans = _partition_slices(g)
    out = g.copy()

    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            region = g[r0 : r1 + 1, c0 : c1 + 1]
            _base, seq = _read_sprite_blueprint(region)
            out[r0 : r1 + 1, c0 : c1 + 1] = _paint_region(region.shape[0], region.shape[1], seq)

    assert bool(np.all(out[g == wall] == wall)), "separator lines must be preserved"
    return out


# ---------------- generator ----------------

_GEN_MAX_ATTEMPTS = 500
_COLORS = list(range(10))
_LAYOUTS = [(1, 2), (2, 1), (2, 2), (2, 3), (3, 2)]
_THICKNESS_PATTERNS = [
    (1, 2),
    (2, 1),
    (1, 1, 1),
    (1, 2, 1),
    (2, 1, 1),
    (1, 1, 2),
    (2, 2),
    (1, 1, 1, 1),
]


def _sample_axis_sizes(count: int, rng: random.Random) -> list[int] | None:
    for _ in range(100):
        sizes = [rng.randint(7, 12) for _ in range(int(count))]
        if sum(sizes) + int(count) - 1 <= 30:
            return sizes
    return None


def _starts_from_sizes(sizes: list[int]) -> tuple[list[tuple[int, int]], list[int]]:
    spans: list[tuple[int, int]] = []
    lines: list[int] = []
    pos = 0
    for i, size in enumerate(sizes):
        spans.append((pos, pos + int(size) - 1))
        pos += int(size)
        if i != len(sizes) - 1:
            lines.append(pos)
            pos += 1
    return spans, lines


def _color_for_uncycled_distance(d: int, seq: list[tuple[int, int]]) -> int:
    offset = int(d)
    for color, thick in seq:
        if offset < int(thick):
            return int(color)
        offset -= int(thick)
    raise AssertionError("distance exceeds uncycled sprite sequence")


def _make_sprite(seq: list[tuple[int, int]]) -> np.ndarray:
    total = sum(int(thick) for _color, thick in seq)
    side = 2 * total - 1
    dist = _distance_to_rect_boundary(side, side)
    patch = np.empty((side, side), dtype=int)
    for d in range(total):
        patch[dist == d] = _color_for_uncycled_distance(d, seq)
    return patch


def _sequence_for_region(h: int, w: int, rng: random.Random) -> list[tuple[int, int]]:
    max_depth = (min(int(h), int(w)) - 1) // 2
    choices = [p for p in _THICKNESS_PATTERNS if sum(p) <= max_depth]
    assert choices, "region too small for a cycling blueprint"
    thicks = list(rng.choice(choices))
    return [(0, int(t)) for t in thicks]


def _sample_palette(wall: int, base: int, layer_count: int, rng: random.Random) -> list[int]:
    pool = [color for color in _COLORS if color not in {int(wall), int(base)}]
    return rng.sample(pool, int(layer_count))


def _paste_sprite(
    grid: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    patch: np.ndarray,
    rng: random.Random,
    force_off_center: bool,
) -> tuple[int, int]:
    ph, pw = patch.shape
    top_min = int(r0) + 1
    left_min = int(c0) + 1
    top_max = int(r1) - ph
    left_max = int(c1) - pw
    assert top_min <= top_max and left_min <= left_max, "sprite must fit with a base-color margin"

    candidates = [(r, c) for r in range(top_min, top_max + 1) for c in range(left_min, left_max + 1)]
    if force_off_center and len(candidates) > 1:
        center_r = (int(r0) + int(r1) - ph + 1) / 2
        center_c = (int(c0) + int(c1) - pw + 1) / 2
        off = [(r, c) for r, c in candidates if (r, c) != (round(center_r), round(center_c))]
        if off:
            candidates = off
    sr, sc = rng.choice(candidates)
    grid[sr : sr + ph, sc : sc + pw] = patch
    return int(sr), int(sc)


def _expected_from_meta(shape: tuple[int, int], wall: int, meta: list[dict[str, object]]) -> np.ndarray:
    out = np.full(shape, int(wall), dtype=int)
    for item in meta:
        r0, r1, c0, c1 = item["bounds"]  # type: ignore[misc]
        seq = item["seq"]  # type: ignore[assignment]
        out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = _paint_region(
            int(r1) - int(r0) + 1,
            int(c1) - int(c0) + 1,
            seq,  # type: ignore[arg-type]
        )
    return out


def _naive_without_thickness(shape: tuple[int, int], wall: int, meta: list[dict[str, object]]) -> np.ndarray:
    out = np.full(shape, int(wall), dtype=int)
    for item in meta:
        r0, r1, c0, c1 = item["bounds"]  # type: ignore[misc]
        seq = [(color, 1) for color, _thick in item["seq"]]  # type: ignore[index]
        out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = _paint_region(
            int(r1) - int(r0) + 1,
            int(c1) - int(c0) + 1,
            seq,
        )
    return out


def _paint_no_cycle(h: int, w: int, seq: list[tuple[int, int]]) -> np.ndarray:
    cycle = _expanded_cycle(seq)
    dist = _distance_to_rect_boundary(h, w)
    out = np.empty((int(h), int(w)), dtype=int)
    last = len(cycle) - 1
    for d in range(int(dist.max()) + 1):
        out[dist == d] = cycle[min(d, last)]
    return out


def _naive_without_cycling(shape: tuple[int, int], wall: int, meta: list[dict[str, object]]) -> np.ndarray:
    out = np.full(shape, int(wall), dtype=int)
    for item in meta:
        r0, r1, c0, c1 = item["bounds"]  # type: ignore[misc]
        seq = item["seq"]  # type: ignore[assignment]
        out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = _paint_no_cycle(
            int(r1) - int(r0) + 1,
            int(c1) - int(c0) + 1,
            seq,  # type: ignore[arg-type]
        )
    return out


def _semantic_guard(grid: np.ndarray, expected: np.ndarray, wall: int, meta: list[dict[str, object]]) -> bool:
    if grid.ndim != 2 or expected.ndim != 2 or max(grid.shape) > 30 or max(expected.shape) > 30:
        return False
    if len(meta) < 2:
        return False
    if np.array_equal(grid, expected):
        return False

    has_thick_layer = False
    has_cycle = False
    has_off_center_sprite = False
    used_sequences: set[tuple[tuple[int, int], ...]] = set()

    for item in meta:
        r0, r1, c0, c1 = item["bounds"]  # type: ignore[misc]
        sr, sc = item["sprite"]  # type: ignore[misc]
        ph, pw = item["sprite_shape"]  # type: ignore[misc]
        seq = item["seq"]  # type: ignore[assignment]
        cycle_len = sum(thick for _color, thick in seq)  # type: ignore[union-attr]
        max_depth = (min(int(r1) - int(r0) + 1, int(c1) - int(c0) + 1) - 1) // 2
        has_cycle |= max_depth >= cycle_len
        has_thick_layer |= any(thick > 1 for _color, thick in seq)  # type: ignore[union-attr]
        used_sequences.add(tuple(seq))  # type: ignore[arg-type]

        centered_r = (int(r0) + int(r1) - int(ph) + 1) / 2
        centered_c = (int(c0) + int(c1) - int(pw) + 1) / 2
        has_off_center_sprite |= (int(sr), int(sc)) != (round(centered_r), round(centered_c))

    if not has_thick_layer or not has_cycle or not has_off_center_sprite:
        return False
    if len(used_sequences) < 2:
        return False
    if np.array_equal(expected, _naive_without_thickness(grid.shape, wall, meta)):
        return False
    if np.array_equal(expected, _naive_without_cycling(grid.shape, wall, meta)):
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_row_parts, n_col_parts = rng.choice(_LAYOUTS)
    row_sizes = _sample_axis_sizes(n_row_parts, rng)
    col_sizes = _sample_axis_sizes(n_col_parts, rng)
    if row_sizes is None or col_sizes is None:
        return None

    row_spans, wall_rows = _starts_from_sizes(row_sizes)
    col_spans, wall_cols = _starts_from_sizes(col_sizes)
    height = sum(row_sizes) + len(wall_rows)
    width = sum(col_sizes) + len(wall_cols)
    if max(height, width) > 30:
        return None

    wall = rng.choice(_COLORS)
    grid = np.full((height, width), int(wall), dtype=int)
    meta: list[dict[str, object]] = []
    force_off_center_at = rng.randrange(len(row_spans) * len(col_spans))
    idx = 0

    for rr, (r0, r1) in enumerate(row_spans):
        for cc, (c0, c1) in enumerate(col_spans):
            h = int(r1) - int(r0) + 1
            w = int(c1) - int(c0) + 1
            base_pool = [color for color in _COLORS if color != wall]
            base = rng.choice(base_pool)
            grid[r0 : r1 + 1, c0 : c1 + 1] = int(base)

            seq_template = _sequence_for_region(h, w, rng)
            colors = _sample_palette(wall, base, len(seq_template), rng)
            seq = [(colors[i], thick) for i, (_dummy, thick) in enumerate(seq_template)]
            patch = _make_sprite(seq)
            sr, sc = _paste_sprite(
                grid,
                r0,
                c0,
                r1,
                c1,
                patch,
                rng,
                force_off_center=(idx == force_off_center_at),
            )
            meta.append(
                {
                    "bounds": (int(r0), int(r1), int(c0), int(c1)),
                    "base": int(base),
                    "seq": seq,
                    "sprite": (sr, sc),
                    "sprite_shape": tuple(map(int, patch.shape)),
                    "slot": (rr, cc),
                }
            )
            idx += 1

    # The full separator rows and columns are already wall color because the canvas began as wall.
    for r in wall_rows:
        grid[r, :] = wall
    for c in wall_cols:
        grid[:, c] = wall

    expected = _expected_from_meta(grid.shape, wall, meta)
    if not _semantic_guard(grid, expected, wall, meta):
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid micro-renovation blueprint input")
