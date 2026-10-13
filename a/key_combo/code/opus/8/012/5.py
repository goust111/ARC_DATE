from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_ROOT = Path("/home/lixinyu/snap/new")
if _ARC_ROOT.exists():
    sys.path.append(str(_ARC_ROOT))

try:
    from solvers.arc_utils import Color, get_bg
except Exception:
    class Color:
        BLACK = 0
        BLUE = 1
        GREEN = 3
        YELLOW = 4
        PINK = 6
        ORANGE = 7
        MAROON = 9

    def get_bg(grid: np.ndarray) -> int:
        vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
        return int(vals[np.argmax(counts)])


PATH_COLORS = (Color.GREEN, Color.YELLOW, Color.ORANGE, Color.PINK)
MAX_ATTEMPTS = 500


def _parse_source_run(grid: np.ndarray) -> list[dict[str, object]]:
    row = np.asarray(grid[0], dtype=int)
    cols = [int(c) for c in np.flatnonzero(row != Color.BLACK)]
    assert cols, "top row must contain a source run"
    start, end = cols[0], cols[-1]
    assert cols == list(range(start, end + 1)), "source pixels must occupy one contiguous run"
    assert 2 <= len(cols) <= 4, "source run length must be 2..4"
    colors = [int(row[c]) for c in cols]
    assert len(set(colors)) == len(colors), "source colors must be distinct"
    assert set(colors).issubset(PATH_COLORS), "unexpected source color"
    assert np.all(row[:start] == Color.BLACK)
    assert np.all(row[end + 1 :] == Color.BLACK)
    return [{"color": color, "source_col": col} for color, col in zip(colors, cols)]


def _parse_input(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = map(int, g.shape)
    assert h >= 5 and w >= 5
    assert get_bg(g) == Color.BLACK, "background must be black"

    sources = _parse_source_run(g)
    source_colors = {int(src["color"]) for src in sources}
    allowed = {Color.BLACK, Color.MAROON, *source_colors}
    assert set(map(int, np.unique(g))).issubset(allowed), "unexpected colors in grid"

    markers = np.argwhere(g == Color.MAROON)
    assert markers.shape == (2, 2), "expected exactly two maroon markers"
    marker_rows = sorted(int(r) for r in markers[:, 0])
    marker_cols = sorted(int(c) for c in markers[:, 1])
    assert marker_rows[0] > 0, "markers must not be on the top row"
    assert marker_rows[0] != marker_rows[1], "markers must be on different rows"
    assert marker_cols[0] != marker_cols[1], "markers must be on different columns"

    used_source_cells = {(0, int(src["source_col"])) for src in sources}
    for src in sources:
        color = int(src["color"])
        source_col = int(src["source_col"])
        coords = [tuple(map(int, coord)) for coord in np.argwhere(g == color)]
        assert 1 <= len(coords) <= 2, "each source color may appear only as source plus optional endpoint"
        assert (0, source_col) in coords, "missing source pixel on top row"
        extras = [coord for coord in coords if coord != (0, source_col)]
        if extras:
            endpoint = extras[0]
            er, ec = endpoint
            assert er > 0, "endpoint must be below the top row"
            assert ec != source_col, "endpoint cannot share the source column"
            src["endpoint"] = endpoint
        else:
            src["endpoint"] = None

    for r, c in markers:
        assert (int(r), int(c)) not in used_source_cells

    return {
        "grid": g,
        "shape": (h, w),
        "sources": sources,
        "marker_rows": tuple(marker_rows),
        "marker_cols": tuple(marker_cols),
    }


def _paint_paths(shape: tuple[int, int], sources: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    h, w = map(int, shape)
    owners = np.full((h, w), -1, dtype=int)
    colors = np.zeros((h, w), dtype=int)
    for src in sources:
        color = int(src["color"])
        c0 = int(src["source_col"])
        endpoint = src["endpoint"]
        cells: list[tuple[int, int]] = []
        if endpoint is None:
            cells.extend((r, c0) for r in range(h))
        else:
            er, ec = map(int, endpoint)
            cells.extend((r, c0) for r in range(er + 1))
            step = 1 if ec > c0 else -1
            cells.extend((er, c) for c in range(c0, ec + step, step))
        for r, c in cells:
            prev = int(owners[r, c])
            assert prev in (-1, color), "path overlap between different colors is ambiguous"
            owners[r, c] = color
            colors[r, c] = color
    return owners != -1, colors


def _compose(parsed: dict[str, object]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    h, w = parsed["shape"]
    r0, r1 = parsed["marker_rows"]
    c0, c1 = parsed["marker_cols"]
    out = np.zeros((h, w), dtype=int)

    blue_mask = np.zeros((h, w), dtype=bool)
    if r1 - r0 > 1 and c1 - c0 > 1:
        blue_mask[r0 + 1 : r1, c0 + 1 : c1] = True
        out[blue_mask] = Color.BLUE

    path_mask, path_colors = _paint_paths((h, w), parsed["sources"])
    out[path_mask] = path_colors[path_mask]

    cross_mask = np.zeros((h, w), dtype=bool)
    cross_mask[[r0, r1], :] = True
    cross_mask[:, [c0, c1]] = True
    out[cross_mask] = Color.MAROON

    return out, {
        "blue_mask": blue_mask,
        "path_mask": path_mask,
        "path_colors": path_colors,
        "cross_mask": cross_mask,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combine top-edge ray paths with a maroon-marked rectangle using fixed drawing priority.

    Input structure:
    - The grid has a black background and a single contiguous top-row run of 2 to 4 distinct source pixels.
    - Source colors come from green, yellow, orange, and pink; each source color may have one matching endpoint pixel elsewhere.
    - Exactly two maroon pixels lie away from the top row and define opposite corners of a rectangle.

    Rule:
    - For each top-row source, draw a 1-pixel-wide vertical path downward in its source column.
    - If that source color has a matching endpoint, stop the vertical segment on the endpoint row and extend horizontally on that row to the endpoint column; otherwise continue to the bottom edge.
    - Independently draw full-width maroon lines through both marker rows, full-height maroon lines through both marker columns, and fill the strict interior of the marker rectangle blue.
    - Merge the two constructions with priority maroon cross lines over source-colored paths over blue fill over black background.

    Key insight:
    - The puzzle is not a pipeline: the path system and the rectangle system are inferred separately, then resolved on one shared canvas by an explicit priority order.
    """
    parsed = _parse_input(np.asarray(grid, dtype=int))
    out, _layers = _compose(parsed)
    return out


def _endpoint_patterns(n_sources: int) -> list[tuple[bool, bool]]:
    if n_sources == 2:
        return [(False, False), (True, False), (False, True), (True, True)]
    return [(False, False), (True, False), (False, True), (True, True)]


def _pick_rect_columns(
    width: int, source_cols: list[int], conflict_col: int, rng: random.Random
) -> tuple[int, int] | None:
    pairs = []
    source_set = set(source_cols)
    for c0 in range(conflict_col):
        for c1 in range(conflict_col + 1, width):
            interior = set(range(c0 + 1, c1))
            if len(interior) < 2:
                continue
            if conflict_col not in interior:
                continue
            if len(interior - source_set) < 1:
                continue
            pairs.append((c0, c1))
    if not pairs:
        return None
    return rng.choice(pairs)


def _pick_rect_rows(height: int, rng: random.Random) -> tuple[int, int] | None:
    pairs = []
    for r0 in range(2, height - 3):
        for r1 in range(r0 + 3, height - 1):
            pairs.append((r0, r1))
    if not pairs:
        return None
    return rng.choice(pairs)


def _pick_endpoint(
    side: str,
    source_col: int,
    shape: tuple[int, int],
    marker_rows: tuple[int, int],
    marker_cols: tuple[int, int],
    used: set[tuple[int, int]],
    rng: random.Random,
    *,
    force_below: bool,
) -> tuple[int, int] | None:
    h, w = shape
    r0, r1 = marker_rows
    bad_rows = {0, r0, r1}
    bad_cols = set(marker_cols)
    candidate_rows: list[int] = []
    if force_below:
        candidate_rows.extend(r for r in range(r1 + 1, h) if r not in bad_rows)
    else:
        upper = [r for r in range(1, r0) if r not in bad_rows]
        lower = [r for r in range(r1 + 1, h) if r not in bad_rows]
        if upper and lower:
            candidate_rows.extend(rng.sample([upper, lower], 2)[0])
            candidate_rows.extend(r for r in (upper + lower) if r not in candidate_rows)
        else:
            candidate_rows.extend(upper + lower)
    if side == "left":
        candidate_cols = [c for c in range(0, source_col) if c not in bad_cols]
    else:
        candidate_cols = [c for c in range(source_col + 1, w) if c not in bad_cols]
    rng.shuffle(candidate_rows)
    rng.shuffle(candidate_cols)
    for r in candidate_rows:
        for c in candidate_cols:
            if (r, c) not in used:
                return (r, c)
    return None


def _build_candidate(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(13, 18)
    w = rng.randint(13, 18)
    n_sources = rng.choices([2, 3, 4], weights=[3, 4, 3])[0]
    if w < n_sources + 6:
        return None

    start_col = rng.randint(2, w - n_sources - 3)
    source_cols = list(range(start_col, start_col + n_sources))
    source_colors = rng.sample(list(PATH_COLORS), n_sources)

    endpoint_flags = [False] * n_sources
    left_flag, right_flag = rng.choice(_endpoint_patterns(n_sources))
    endpoint_flags[0] = left_flag
    endpoint_flags[-1] = right_flag

    conflict_candidates = [idx for idx in range(1, n_sources - 1)]
    if not conflict_candidates:
        conflict_candidates = [idx for idx, has_endpoint in enumerate(endpoint_flags) if not has_endpoint]
    if not conflict_candidates:
        conflict_candidates = [0, n_sources - 1]
    conflict_idx = rng.choice(conflict_candidates)
    conflict_col = source_cols[conflict_idx]

    rect_cols = _pick_rect_columns(w, source_cols, conflict_col, rng)
    rect_rows = _pick_rect_rows(h, rng)
    if rect_cols is None or rect_rows is None:
        return None
    marker_cols = tuple(sorted(rect_cols))
    marker_rows = tuple(sorted(rect_rows))

    grid = np.zeros((h, w), dtype=int)
    for color, col in zip(source_colors, source_cols):
        grid[0, col] = color

    used = {(0, col) for col in source_cols}
    marker_a = (marker_rows[0], marker_cols[0])
    marker_b = (marker_rows[1], marker_cols[1])
    grid[marker_a] = Color.MAROON
    grid[marker_b] = Color.MAROON
    used.update([marker_a, marker_b])

    if endpoint_flags[0]:
        ep = _pick_endpoint(
            "left",
            source_cols[0],
            (h, w),
            marker_rows,
            marker_cols,
            used,
            rng,
            force_below=(conflict_idx == 0),
        )
        if ep is None:
            return None
        grid[ep] = source_colors[0]
        used.add(ep)

    if endpoint_flags[-1]:
        ep = _pick_endpoint(
            "right",
            source_cols[-1],
            (h, w),
            marker_rows,
            marker_cols,
            used,
            rng,
            force_below=(conflict_idx == n_sources - 1),
        )
        if ep is None:
            return None
        grid[ep] = source_colors[-1]
        used.add(ep)

    return grid


def _semantic_check(grid: np.ndarray) -> None:
    parsed = _parse_input(grid)
    out, layers = _compose(parsed)
    blue_mask = layers["blue_mask"]
    path_mask = layers["path_mask"]
    path_colors = layers["path_colors"]
    cross_mask = layers["cross_mask"]

    assert blue_mask.any(), "rectangle interior must be non-empty"
    assert (blue_mask & path_mask).any(), "at least one path must conflict with blue fill"
    assert (cross_mask & path_mask).any(), "at least one path must conflict with the maroon cross"
    assert (blue_mask & ~path_mask & ~cross_mask).any(), "blue fill must remain visible"

    expected = np.zeros_like(out)
    expected[blue_mask] = Color.BLUE
    expected[path_mask] = path_colors[path_mask]
    expected[cross_mask] = Color.MAROON
    assert np.array_equal(out, expected)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = _build_candidate(rng)
        if grid is None:
            continue
        try:
            _semantic_check(grid)
            _ = main(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
