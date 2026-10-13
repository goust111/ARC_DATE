from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
WALL = 5
OBJECT = 7
SEGMENT_COLORS = (1, 2, 3, 4)
GENERATE_ATTEMPTS = 500


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _monochrome_components(grid: np.ndarray, skip_colors: set[int]) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color in skip_colors:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": len(coords),
                }
            )
    return components


def _classify_input(grid: np.ndarray) -> tuple[int, int, list[tuple[int, int, int, int]], list[tuple[int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 6 and w >= 6

    bg = Counter(map(int, g.flat)).most_common(1)[0][0]
    wall = int(g[0, 0])
    assert np.all(g[0] == wall), "top edge must be a solid wall bar"
    assert wall != bg

    segments: list[tuple[int, int, int, int]] = []
    objects: list[tuple[int, int, int]] = []
    for comp in _monochrome_components(g, {bg, wall}):
        r0, c0, r1, c1 = comp["bbox"]
        if comp["area"] == 1:
            objects.append((r0, c0, int(comp["color"])))
            continue
        assert r0 == r1, "all non-wall multi-cell objects must be horizontal segments"
        assert comp["area"] == c1 - c0 + 1, "segments must be solid 1-cell-thick bars"
        segments.append((r0, c0, c1, int(comp["color"])))

    assert 3 <= len(segments) <= 4
    assert 3 <= len(objects) <= 5
    assert len({r for r, _, _, _ in segments}) == len(segments)
    assert len({c for _, c, _ in objects}) == len(objects)
    return bg, wall, segments, objects


def _render_base(shape: tuple[int, int], bg: int, wall: int, segments: list[tuple[int, int, int, int]]) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), bg, dtype=int)
    out[0, :] = wall

    for row, c0, c1, color in segments:
        out[row, c0 : c1 + 1] = color

    for row, c0, c1, color in sorted(segments, key=lambda item: item[0], reverse=True):
        if row > 1:
            out[1:row, c0 : c1 + 1] = color
    return out


def _settle_objects(
    base: np.ndarray,
    bg: int,
    wall: int,
    objects: list[tuple[int, int, int]],
) -> tuple[np.ndarray, list[tuple[int, int, int, int, int]]]:
    out = base.copy()
    h, _ = out.shape
    occupied = np.zeros_like(out, dtype=bool)
    settled: list[tuple[int, int, int, int, int]] = []

    for start_r, col, start_color in sorted(objects, key=lambda item: item[0], reverse=True):
        row = start_r
        while row + 1 < h and not occupied[row + 1, col]:
            row += 1

        final_color = start_color
        for rr in range(row - 1, -1, -1):
            value = int(base[rr, col])
            if value != bg and value != wall:
                final_color = value
                break

        out[row, col] = final_color
        occupied[row, col] = True
        settled.append((start_r, col, row, col, final_color))

    return out, settled


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project upward color bands from horizontal segments, then recolor falling objects by the deepest band in their landing columns.

    Input structure:
    - The background is black.
    - The top row is a solid gray edge bar.
    - Several colored horizontal segments sit lower in the grid, and several isolated single-pixel objects sit near the top.

    Rule:
    - Each horizontal segment projects a same-color rectangular band upward from the row just above the segment to the row just below the gray bar.
    - Farther segments project first and nearer segments overwrite overlapping band cells.
    - Each single-pixel object then falls straight downward until it hits the bottom edge or a previously settled object, ignoring the colored segments.
    - After landing, an object is recolored by the first non-background, non-wall band/segment cell found directly above it in the same column; if none exists, it keeps its original color.

    Key insight:
    - The final object color is determined only after combining two independent spatial processes: upward projection fixes per-column color ownership, and downward gravity chooses which column each landed object samples from below.
    """
    bg, wall, segments, objects = _classify_input(grid)
    base = _render_base(np.asarray(grid).shape, bg, wall, segments)
    out, _ = _settle_objects(base, bg, wall, objects)
    return out


def _column_semantics(
    h: int,
    w: int,
    segments: list[tuple[int, int, int, int]],
) -> tuple[np.ndarray, list[int | None], list[int | None], list[int]]:
    base = _render_base((h, w), BG, WALL, segments)
    top_visible: list[int | None] = []
    deepest_visible: list[int | None] = []
    cover_counts: list[int] = []

    for c in range(w):
        top = int(base[1, c]) if h > 1 and int(base[1, c]) not in (BG, WALL) else None
        deep = None
        for r in range(h - 1, -1, -1):
            value = int(base[r, c])
            if value not in (BG, WALL):
                deep = value
                break
        top_visible.append(top)
        deepest_visible.append(deep)
        cover_counts.append(sum(c0 <= c <= c1 for _row, c0, c1, _color in segments))

    return base, top_visible, deepest_visible, cover_counts


def _has_overlap_pair(segments: list[tuple[int, int, int, int]]) -> bool:
    for i, (_r0, a0, a1, _ca) in enumerate(segments):
        for _r1, b0, b1, _cb in segments[i + 1 :]:
            if max(a0, b0) <= min(a1, b1):
                return True
    return False


def _pick_columns(
    rng: random.Random,
    overlap_cols: list[int],
    single_cols: list[int],
    empty_cols: list[int],
    total: int,
) -> tuple[list[int], dict[int, str]] | None:
    all_candidates = sorted(set(overlap_cols) | set(single_cols) | set(empty_cols))
    by_kind = [("overlap", overlap_cols), ("single", single_cols), ("empty", empty_cols)]

    for min_gap in (2, 1, 0):
        chosen: list[int] = []
        labels: dict[int, str] = {}
        ok = True
        for kind, cols in sorted(by_kind, key=lambda item: len(item[1])):
            options = [c for c in cols if all(abs(c - prev) > min_gap for prev in chosen)]
            if not options:
                ok = False
                break
            col = rng.choice(options)
            chosen.append(col)
            labels[col] = kind
        if not ok:
            continue

        while len(chosen) < total:
            pool = [c for c in all_candidates if c not in chosen and all(abs(c - prev) > min_gap for prev in chosen)]
            if not pool:
                break
            col = rng.choice(pool)
            chosen.append(col)
            if col in overlap_cols:
                labels[col] = "overlap"
            elif col in single_cols:
                labels[col] = "single"
            else:
                labels[col] = "empty"

        if len(chosen) == total:
            rng.shuffle(chosen)
            return chosen, labels
    return None


def _gen_segments(rng: random.Random, h: int, w: int) -> list[tuple[int, int, int, int]] | None:
    n_segments = rng.randint(3, 4)
    rows = sorted(rng.sample(range(5, h - 2), n_segments))
    colors = rng.sample(list(SEGMENT_COLORS), n_segments)

    widths = [rng.randint(3, 5) for _ in range(n_segments)]
    starts: list[int] = [0] * n_segments

    anchor_width = widths[0]
    starts[0] = rng.randint(1, w - anchor_width - 2)

    shifts = [shift for shift in (-2, -1, 1, 2) if 0 <= starts[0] + shift <= w - widths[1]]
    if not shifts:
        return None
    starts[1] = starts[0] + rng.choice(shifts)
    if max(starts[0], starts[1]) > min(starts[0] + widths[0] - 1, starts[1] + widths[1] - 1):
        return None
    if starts[0] == starts[1] and widths[0] == widths[1]:
        return None

    for idx in range(2, n_segments):
        starts[idx] = rng.randint(0, w - widths[idx])

    segments = [(rows[i], starts[i], starts[i] + widths[i] - 1, colors[i]) for i in range(n_segments)]
    if not _has_overlap_pair(segments):
        return None
    return segments


def _semantic_checks(
    inp: np.ndarray,
    segments: list[tuple[int, int, int, int]],
    source_objects: list[tuple[int, int, int]],
    object_labels: dict[int, str],
) -> bool:
    h, w = inp.shape
    if not (max(h, w) <= 30):
        return False
    if not _has_overlap_pair(segments):
        return False

    base, top_visible, deepest_visible, cover_counts = _column_semantics(h, w, segments)
    pred = main(inp)

    expected_by_col: dict[int, int] = {}
    for _r, c, color in source_objects:
        expected_by_col[c] = deepest_visible[c] if deepest_visible[c] is not None else color

    if len(set(expected_by_col.values())) < 2:
        return False
    if OBJECT not in expected_by_col.values():
        return False
    if all(color == OBJECT for color in expected_by_col.values()):
        return False

    overlap_cols = [c for c, kind in object_labels.items() if kind == "overlap"]
    if not overlap_cols:
        return False
    for c in overlap_cols:
        if cover_counts[c] < 2 or deepest_visible[c] is None or top_visible[c] is None:
            return False
        if deepest_visible[c] == top_visible[c]:
            return False

    for src_r, c, src_color in source_objects:
        if int(pred[h - 1, c]) != expected_by_col[c]:
            return False
        if int(pred[src_r, c]) == src_color and src_r != h - 1:
            return False

    if np.array_equal(pred, inp):
        return False
    if np.count_nonzero(base != BG) >= (h * w * 3) // 5:
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(13, 18)
    w = rng.randint(13, 18)

    segments = _gen_segments(rng, h, w)
    if segments is None:
        return None

    _base, top_visible, deepest_visible, cover_counts = _column_semantics(h, w, segments)
    overlap_cols = [c for c in range(w) if cover_counts[c] >= 2 and top_visible[c] is not None and deepest_visible[c] != top_visible[c]]
    single_cols = [c for c in range(w) if cover_counts[c] == 1]
    empty_cols = [c for c in range(w) if deepest_visible[c] is None]
    if not overlap_cols or not single_cols or not empty_cols:
        return None

    n_objects = rng.randint(3, 5)
    picked = _pick_columns(rng, overlap_cols, single_cols, empty_cols, n_objects)
    if picked is None:
        return None
    object_cols, object_labels = picked

    object_rows = [rng.randint(1, 3) for _ in range(n_objects)]
    if len(set(object_rows)) == 1 and n_objects > 1:
        object_rows[0] = 1
        object_rows[-1] = min(3, object_rows[-1] + 1)

    source_objects = list(zip(object_rows, object_cols, [OBJECT] * n_objects))

    grid = np.full((h, w), BG, dtype=int)
    grid[0, :] = WALL
    for row, c0, c1, color in segments:
        grid[row, c0 : c1 + 1] = color
    for row, col, color in source_objects:
        if grid[row, col] != BG:
            return None
        grid[row, col] = color

    try:
        bg, wall, parsed_segments, parsed_objects = _classify_input(grid)
    except AssertionError:
        return None
    if bg != BG or wall != WALL:
        return None
    if sorted(parsed_segments) != sorted(segments):
        return None
    if sorted(parsed_objects) != sorted(source_objects):
        return None

    if not _semantic_checks(grid, segments, source_objects, object_labels):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATE_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
