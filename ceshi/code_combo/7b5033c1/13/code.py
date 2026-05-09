# Combo variant of 7b5033c1 (index 13)
# combo_spec: D5/2+D8/3
# Name: 按分割点切段+分行输出

from __future__ import annotations

from collections import Counter
import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import get_bg


# Design analysis for the combo variant
# - Concretization and trace:
#   A concrete traversal sequence can look like
#       [2,2,4,4,9,3,3,1,1,7,6,6,2,2]
#   where 9 and 7 are special split markers. The original task would output a
#   14x1 column containing every color. The D5-only variant would repack maximal
#   same-color runs into rows. This combo instead:
#   1. traces the snake exactly as before;
#   2. detects internal singleton colors [9, 7] as split points;
#   3. removes those markers and cuts the path into
#          [2,2,4,4], [3,3,1,1], [6,6,2,2];
#   4. writes each full subpath as one row, left-aligned with bg padding.
#   Output becomes
#       [[2,2,4,4],
#        [3,3,1,1],
#        [6,6,2,2]]
#   The changed pipeline is therefore "trace -> detect markers -> cut ->
#   delete markers -> pack subpaths as rows", not merely "trace -> reshape".
# - Complexity calibration:
#   The generator forces 3-4 subpaths, hence 2-3 split markers, and every
#   subpath contains at least two color runs of length >= 2. This prevents two
#   degenerate cases: collapse back to the original Nx1 strip, and collapse to a
#   D5-only run table where each row is monochrome.
# - Implicit-assumption audit:
#   `_trace_path_cells` still relies on the old invariant that all non-bg cells
#   form one simple 4-connected chain with exactly two endpoints. That remains
#   true because markers only recolor path cells. The old final line
#   `reshape((-1, 1))` no longer holds: not every non-bg cell survives to output,
#   because split markers are deleted. `_find_marker_indices` introduces a new
#   assumption: internal path colors with total frequency 1 are markers. The
#   generator guarantees this by giving content colors run length >= 2 and by
#   never placing markers at endpoints or adjacent to each other.
# - Placement legality:
#   Path growth reuses a self-avoiding extension routine that rejects
#   out-of-bounds moves, revisits, and side-touching cells. Coloring never adds
#   extra objects; it only paints the already-legal path, so overlap and branch
#   creation cannot occur at the coloring stage.
# - Visual observability:
#   In the input, humans can see a single snake plus a few isolated singleton
#   colors embedded inside it. In the output, those singleton colors disappear
#   and the snake is broken into several left-aligned rows. Because each row is
#   multicolor, the user can infer that rows represent whole subpaths rather than
#   monochrome color runs.
# - Visual cleanliness:
#   There is only one object family in the input: one snake. Split markers are
#   single cells on that snake, so there is no clutter from overlapping regions
#   or extra decorations. The output is a small compact rectangle with only tail
#   padding in bg.
# - Color-change inference:
#   No arbitrary recoloring happens. Non-marker colors are preserved exactly and
#   marker colors simply vanish. The cue for vanishing is structural
#   (internal singleton usage), not tied to a fixed numeric palette value.


def _trace_path_cells(g: np.ndarray, bg: int) -> list[tuple[int, int]]:
    """Return snake cells in traversal order from the top-left endpoint."""
    mask = g != int(bg)
    pts = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    if not pts:
        return []

    pset = set(pts)
    deg: dict[tuple[int, int], int] = {}
    for r, c in pts:
        d = 0
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            if (r + dr, c + dc) in pset:
                d += 1
        deg[(r, c)] = int(d)

    ends = [p for p in pts if int(deg[p]) == 1]
    assert len(ends) == 2
    start = min(ends)

    order: list[tuple[int, int]] = []
    prev: tuple[int, int] | None = None
    cur = start
    for _ in range(len(pts)):
        order.append(cur)
        r, c = cur
        nbrs = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            q = (r + dr, c + dc)
            if q in pset and q != prev:
                nbrs.append(q)
        if not nbrs:
            break
        assert len(nbrs) == 1
        prev, cur = cur, nbrs[0]

    assert len(order) == len(pts)
    return order


def _extract_color_sequence(g: np.ndarray) -> tuple[int, list[int]]:
    bg = int(get_bg(g))
    order = _trace_path_cells(g, bg)
    cols = [int(g[r, c]) for r, c in order]
    return bg, cols


def _find_marker_indices(cols: list[int]) -> list[int]:
    """
    Recover split points as internal singleton colors.

    New solver assumption:
    - Any internal color that appears exactly once on the traced snake is a split
      marker and should be removed.
    - Endpoints are never markers.
    """
    counts = Counter(int(col) for col in cols)
    markers: list[int] = []
    for idx, col in enumerate(cols):
        if 0 < idx < len(cols) - 1 and counts[int(col)] == 1:
            markers.append(int(idx))
    return markers


def _split_at_markers(cols: list[int], marker_indices: list[int]) -> list[list[int]]:
    """
    Remove marker cells and split the traversal sequence into subpaths.

    Generator guarantees marker indices are strictly internal and non-adjacent, so
    every recovered subpath is non-empty.
    """
    if not cols:
        return []

    segments: list[list[int]] = []
    start = 0
    for idx in marker_indices:
        if idx > start:
            segments.append([int(v) for v in cols[start:idx]])
        start = idx + 1
    if start < len(cols):
        segments.append([int(v) for v in cols[start:]])
    return segments


def _pack_segments(segments: list[list[int]], bg: int) -> np.ndarray:
    """Pack variable-length subpath sequences into a left-aligned table."""
    if not segments:
        return np.zeros((0, 0), dtype=int)

    out_h = int(len(segments))
    out_w = int(max(len(seg) for seg in segments))
    out = np.full((out_h, out_w), int(bg), dtype=int)
    for r, seg in enumerate(segments):
        out[int(r), : len(seg)] = np.array(seg, dtype=int)
    return out


def _analyze_grid(grid: np.ndarray) -> dict[str, int]:
    """
    Inspect a generated example.

    This is used for post-generation validation and later behavior statistics.
    """
    g = np.array(grid, dtype=int)
    bg, cols = _extract_color_sequence(g)
    marker_indices = _find_marker_indices(cols)
    segments = _split_at_markers(cols, marker_indices)
    multicolor_rows = 0
    for seg in segments:
        if len(set(int(v) for v in seg)) >= 2:
            multicolor_rows += 1
    out = _pack_segments(segments, bg)
    return {
        "marker_count": int(len(marker_indices)),
        "segment_count": int(len(segments)),
        "output_h": int(out.shape[0]),
        "output_w": int(out.shape[1]) if out.size else 0,
        "multicolor_rows": int(multicolor_rows),
        "input_non_bg": int(len(cols)),
        "output_non_bg": int(sum(len(seg) for seg in segments)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the snake, cut it at internal singleton markers, and emit one row per
    remaining subpath.

    Loop-by-loop assumption audit:
    - Path tracing assumes one simple chain with two endpoints.
    - Marker detection assumes content colors are never singletons and markers are
      never endpoints.
    - Segment packing assumes bg padding remains distinguishable from true cells.
    The generator enforces all three conditions.
    """
    g = np.array(grid, dtype=int)
    bg, cols = _extract_color_sequence(g)
    marker_indices = _find_marker_indices(cols)
    segments = _split_at_markers(cols, marker_indices)
    return _pack_segments(segments, bg)


def _try_extend(
    path_set: set[tuple[int, int]],
    last: tuple[int, int],
    dr: int,
    dc: int,
    count: int,
    H: int,
    W: int,
) -> list[tuple[int, int]]:
    """Try to extend the path while rejecting revisits and side-touching cells."""
    segment: list[tuple[int, int]] = []
    r, c = last
    for i in range(1, count + 1):
        nr, nc = int(r + dr * i), int(c + dc * i)
        if nr < 1 or nr >= H - 1 or nc < 1 or nc >= W - 1:
            break
        if (nr, nc) in path_set:
            break

        pred = (int(r + dr * (i - 1)), int(c + dc * (i - 1)))
        bad = False
        for ddr, ddc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nb = (int(nr + ddr), int(nc + ddc))
            if nb != pred and nb in path_set:
                bad = True
                break
        if bad:
            break

        segment.append((nr, nc))
        path_set.add((nr, nc))
    return segment


def _random_partition(
    total: int,
    num_parts: int,
    min_len: int,
    max_len: int,
    *,
    require_variety: bool = False,
) -> list[int] | None:
    """Partition `total` into bounded positive integers."""
    if num_parts * min_len > total or num_parts * max_len < total:
        return None

    for _ in range(100):
        parts = [int(min_len)] * int(num_parts)
        remaining = int(total - sum(parts))
        for _ in range(remaining):
            eligible = [i for i in range(num_parts) if parts[i] < max_len]
            if not eligible:
                break
            parts[random.choice(eligible)] += 1
        if sum(parts) != total:
            continue
        if require_variety and len(set(parts)) < 2:
            continue
        random.shuffle(parts)
        return [int(p) for p in parts]
    return None


def _build_path(target_len: int, H: int, W: int) -> list[tuple[int, int]] | None:
    """Grow a snake of exactly `target_len` cells."""
    for _ in range(120):
        start_r = random.randint(1, 4)
        start_c = random.randint(1, W - 2)

        path = [(int(start_r), int(start_c))]
        path_set = {(int(start_r), int(start_c))}
        turns = 0
        is_vertical = True
        h_dir = random.choice([-1, 1])

        while len(path) < target_len:
            run_len = random.randint(1, 5)
            run_len = min(int(run_len), int(target_len - len(path)))
            if run_len <= 0:
                break

            last = path[-1]
            segment: list[tuple[int, int]] = []

            if is_vertical:
                v_dir = 1 if random.random() < 0.85 else -1
                segment = _try_extend(path_set, last, v_dir, 0, run_len, H, W)
                if not segment:
                    segment = _try_extend(path_set, last, -v_dir, 0, run_len, H, W)
            else:
                segment = _try_extend(path_set, last, 0, h_dir, run_len, H, W)
                if not segment:
                    segment = _try_extend(path_set, last, 0, -h_dir, run_len, H, W)
                    if segment:
                        h_dir = -h_dir
                h_dir = -h_dir

            if not segment:
                break

            path.extend(segment)
            turns += 1
            is_vertical = not is_vertical

        if len(path) == target_len and 7 <= turns <= 12:
            return path
    return None


def _build_segment_sequence(seg_len: int, palette: list[int]) -> list[int] | None:
    """Construct one kept subpath with at least two visible color runs."""
    max_runs = min(3, seg_len // 2)
    if max_runs < 2:
        return None

    for _ in range(80):
        num_runs = random.randint(2, max_runs)
        max_run_len = max(4, seg_len - 2 * (num_runs - 1))
        run_lengths = _random_partition(
            seg_len,
            num_runs,
            min_len=2,
            max_len=max_run_len,
            require_variety=(num_runs > 1),
        )
        if run_lengths is None:
            continue

        run_colors: list[int] = []
        for _idx in range(num_runs):
            if not run_colors:
                run_colors.append(int(random.choice(palette)))
            else:
                choices = [c for c in palette if c != run_colors[-1]]
                run_colors.append(int(random.choice(choices)))
        if len(set(run_colors)) < 2:
            continue

        seq: list[int] = []
        for run_len, color in zip(run_lengths, run_colors):
            seq.extend([int(color)] * int(run_len))
        return seq
    return None


def _build_sequence(total_len: int, bg: int) -> tuple[list[int], list[list[int]]] | None:
    """
    Build a traversal color sequence with internal singleton split markers.

    Solver-facing guarantees:
    - marker count is 2 or 3;
    - marker colors appear exactly once and only at internal positions;
    - every kept segment has length >= 5 and at least two color runs, so rows in
      the output are visibly multicolor.
    """
    usable = [c for c in range(10) if c != int(bg)]
    for _ in range(120):
        segment_count = random.randint(3, 4)
        marker_count = segment_count - 1
        content_len = total_len - marker_count
        segment_lengths = _random_partition(
            content_len,
            segment_count,
            min_len=5,
            max_len=9,
            require_variety=True,
        )
        if segment_lengths is None:
            continue

        palette_size = random.randint(2, 3)
        if palette_size + marker_count > len(usable):
            continue
        content_palette = random.sample(usable, palette_size)
        marker_pool = [c for c in usable if c not in content_palette]
        marker_colors = random.sample(marker_pool, marker_count)

        segments: list[list[int]] = []
        for seg_len in segment_lengths:
            seg = _build_segment_sequence(int(seg_len), content_palette)
            if seg is None:
                segments = []
                break
            segments.append(seg)
        if len(segments) != segment_count:
            continue

        seq: list[int] = []
        for idx, seg in enumerate(segments):
            seq.extend(seg)
            if idx < marker_count:
                seq.append(int(marker_colors[idx]))

        counts = Counter(seq)
        marker_ok = all(counts[int(marker)] == 1 for marker in marker_colors)
        content_ok = all(counts[int(color)] >= 2 for color in content_palette)
        if not marker_ok or not content_ok:
            continue
        return seq, segments
    return None


def _build_expected_output(segments: list[list[int]], bg: int) -> np.ndarray:
    return _pack_segments(segments, bg)


def _gen_attempt() -> np.ndarray | None:
    H, W = 16, 16
    bg = random.randint(0, 9)
    total_len = random.randint(24, 30)

    path = _build_path(total_len, H, W)
    if path is None:
        return None

    built = _build_sequence(total_len, bg)
    if built is None:
        return None
    colors, segments = built

    grid = np.full((H, W), int(bg), dtype=int)
    for (r, c), col in zip(path, colors):
        grid[int(r), int(c)] = int(col)

    expected_out = _build_expected_output(segments, bg)
    actual_out = main(grid)
    if not np.array_equal(actual_out, expected_out):
        return None

    stats = _analyze_grid(grid)
    if stats["marker_count"] < 2:
        return None
    if stats["segment_count"] < 3:
        return None
    if stats["output_w"] <= 1:
        return None
    if stats["multicolor_rows"] != stats["segment_count"]:
        return None
    if stats["input_non_bg"] - stats["output_non_bg"] != stats["marker_count"]:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate input after 400 attempts")
