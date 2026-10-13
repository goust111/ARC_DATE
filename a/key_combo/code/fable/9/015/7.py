from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


YELLOW = 4
BACKGROUND = 0
MARKER_CHOICES = (1, 2, 3, 5, 6, 7, 8, 9)
GEN_ATTEMPTS = 500


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).flatten())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.asarray(grid).flatten()))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _segment_markers(seg: dict) -> tuple[tuple[int, int], tuple[int, int]]:
    if seg["axis"] == "h":
        return (int(seg["r"]), int(seg["c0"])), (int(seg["r"]), int(seg["c1"]))
    return (int(seg["r0"]), int(seg["c"])), (int(seg["r1"]), int(seg["c"]))


def _segment_cells(seg: dict) -> list[tuple[int, int]]:
    if seg["axis"] == "h":
        r = int(seg["r"])
        c0 = int(seg["c0"])
        c1 = int(seg["c1"])
        return [(r, c) for c in range(c0, c1 + 1)]
    c = int(seg["c"])
    r0 = int(seg["r0"])
    r1 = int(seg["r1"])
    return [(r, c) for r in range(r0, r1 + 1)]


def _segments_to_fence(shape: tuple[int, int], segments: list[dict]) -> np.ndarray:
    fence = np.zeros(shape, dtype=bool)
    for seg in segments:
        if seg["axis"] == "h":
            fence[int(seg["r"]), int(seg["c0"]) : int(seg["c1"]) + 1] = True
        else:
            fence[int(seg["r0"]) : int(seg["r1"]) + 1, int(seg["c"])] = True
    return fence


def _enclosed_mask_from_fence(fence: np.ndarray) -> np.ndarray:
    h, w = map(int, fence.shape)
    passable = ~fence
    outside = np.zeros_like(fence, dtype=bool)
    q: deque[tuple[int, int]] = deque()

    def push(r: int, c: int) -> None:
        if 0 <= r < h and 0 <= c < w and passable[r, c] and not outside[r, c]:
            outside[r, c] = True
            q.append((r, c))

    for r in range(h):
        push(r, 0)
        push(r, w - 1)
    for c in range(w):
        push(0, c)
        push(h - 1, c)

    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    return passable & ~outside


def _fence_mask_from_grid(grid: np.ndarray, bg: int, marker: int) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    fence = np.zeros((h, w), dtype=bool)

    for r in range(h):
        cols = np.flatnonzero(g[r] == marker)
        assert len(cols) <= 2, "each row may contain at most two markers"
        if len(cols) == 2:
            c0, c1 = map(int, cols)
            fence[r, c0 : c1 + 1] = True

    for c in range(w):
        rows = np.flatnonzero(g[:, c] == marker)
        assert len(rows) <= 2, "each column may contain at most two markers"
        if len(rows) == 2:
            r0, r1 = map(int, rows)
            fence[r0 : r1 + 1, c] = True

    assert np.all((g == marker) <= fence), "all markers must belong to the inferred fence"
    assert np.all((g == bg) | (g == marker)), "input may only contain background and marker"
    return fence


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build invisible fences between aligned marker pairs, fill every fence-enclosed background region yellow, then erase the fences.

    Input structure:
    - The background occupies most cells, and there is exactly one non-background marker color.
    - Each row and each column contains zero, one, or two isolated marker pixels.
    - Yellow does not appear in the input.

    Rule:
    - Every row with exactly two markers defines a horizontal fence segment spanning between them, inclusive; every column with exactly two markers defines a vertical fence segment the same way.
    - Treat all fence cells, including the marker cells themselves, as solid barriers. Any background region that cannot reach the grid boundary by 4-neighbor movement is filled yellow.
    - Finally erase every fence cell back to background, so the output keeps only yellow lakes and the original background.

    Key insight:
    - The visible markers are only fence endpoints; the decisive line segments never survive into the output.
    - Isolated fence pieces can exist as decoys, but only fence segments that collaborate to enclose background create yellow regions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = get_bg(g)
    non_bg = colors_present(g, exclude={bg})
    assert len(non_bg) == 1, "expected exactly one marker color"
    marker = int(non_bg[0])
    assert bg != YELLOW and marker != YELLOW, "yellow must be absent from the input"

    fence = _fence_mask_from_grid(g, bg, marker)
    enclosed = _enclosed_mask_from_fence(fence)

    out = np.full_like(g, bg)
    out[enclosed] = YELLOW
    return out


def _rows_and_cols_used(segments: list[dict]) -> tuple[set[int], set[int]]:
    rows: set[int] = set()
    cols: set[int] = set()
    for seg in segments:
        (r0, c0), (r1, c1) = _segment_markers(seg)
        rows.add(r0)
        rows.add(r1)
        cols.add(c0)
        cols.add(c1)
    return rows, cols


def _marker_set(segments: list[dict]) -> set[tuple[int, int]]:
    markers: set[tuple[int, int]] = set()
    for seg in segments:
        markers.update(_segment_markers(seg))
    return markers


def _markers_are_isolated(markers: set[tuple[int, int]]) -> bool:
    pts = list(markers)
    for i, (r0, c0) in enumerate(pts):
        for r1, c1 in pts[i + 1 :]:
            if max(abs(r0 - r1), abs(c0 - c1)) <= 1:
                return False
    return True


def _valid_marker_counts(shape: tuple[int, int], markers: set[tuple[int, int]]) -> bool:
    h, w = map(int, shape)
    row_counts = [0] * h
    col_counts = [0] * w
    for r, c in markers:
        row_counts[r] += 1
        col_counts[c] += 1
    return max(row_counts, default=0) <= 2 and max(col_counts, default=0) <= 2


def _build_input(shape: tuple[int, int], marker: int, segments: list[dict]) -> np.ndarray:
    grid = np.full(shape, BACKGROUND, dtype=int)
    for r, c in _marker_set(segments):
        grid[r, c] = marker
    return grid


def _rect_module(shape: tuple[int, int]) -> tuple[list[dict], dict] | None:
    h, w = map(int, shape)
    mode = random.choices(
        population=[(0, 0), (1, 0), (0, 1), (1, 1)],
        weights=[1, 2, 2, 2],
        k=1,
    )[0]
    has_vdiv, has_hdiv = mode
    if h < 12 or w < 12:
        return None

    r0 = random.randint(2, h - 8)
    r1 = random.randint(r0 + 4, h - 3)
    c0 = random.randint(2, w - 8)
    c1 = random.randint(c0 + 4, w - 3)

    segments = [
        {"axis": "h", "r": r0, "c0": c0, "c1": c1, "role": "perimeter"},
        {"axis": "h", "r": r1, "c0": c0, "c1": c1, "role": "perimeter"},
        {"axis": "v", "c": c0, "r0": r0, "r1": r1, "role": "perimeter"},
        {"axis": "v", "c": c1, "r0": r0, "r1": r1, "role": "perimeter"},
    ]

    if has_vdiv:
        div_c = random.randint(c0 + 1, c1 - 1)
        segments.append({"axis": "v", "c": div_c, "r0": r0 - 2, "r1": r1 + 2, "role": "divider"})
    if has_hdiv:
        div_r = random.randint(r0 + 1, r1 - 1)
        segments.append({"axis": "h", "r": div_r, "c0": c0 - 2, "c1": c1 + 2, "role": "divider"})

    return segments, {"type": "rect", "feature": bool(has_vdiv or has_hdiv)}


def _hex_module(shape: tuple[int, int]) -> tuple[list[dict], dict] | None:
    h, w = map(int, shape)
    if h < 9 or w < 9:
        return None

    r0 = random.randint(1, h - 7)
    g1 = random.randint(2, min(4, h - r0 - 4))
    r1 = r0 + g1
    g2 = random.randint(2, min(4, h - r1 - 2))
    r2 = r1 + g2

    c0 = random.randint(1, w - 7)
    g1c = random.randint(2, min(4, w - c0 - 4))
    c1 = c0 + g1c
    g2c = random.randint(2, min(4, w - c1 - 2))
    c2 = c1 + g2c

    flip_r = random.random() < 0.5
    flip_c = random.random() < 0.5

    def tr(pt: tuple[int, int]) -> tuple[int, int]:
        r, c = pt
        if flip_r:
            r = r0 + r2 - r
        if flip_c:
            c = c0 + c2 - c
        return int(r), int(c)

    base = [(r0, c0), (r0, c2), (r1, c2), (r1, c1), (r2, c1), (r2, c0)]
    vertices = [tr(pt) for pt in base]
    segments: list[dict] = []
    for a, b in zip(vertices, vertices[1:] + vertices[:1]):
        if a[0] == b[0]:
            segments.append(
                {"axis": "h", "r": a[0], "c0": min(a[1], b[1]), "c1": max(a[1], b[1]), "role": "perimeter"}
            )
        else:
            segments.append(
                {"axis": "v", "c": a[1], "r0": min(a[0], b[0]), "r1": max(a[0], b[0]), "role": "perimeter"}
            )

    return segments, {"type": "hex", "feature": True}


def _open_segment(shape: tuple[int, int]) -> dict:
    h, w = map(int, shape)
    if random.random() < 0.5:
        length = random.randint(4, max(4, min(8, w - 2)))
        c0 = random.randint(1, w - length - 1)
        return {"axis": "h", "r": random.randint(1, h - 2), "c0": c0, "c1": c0 + length, "role": "open"}
    length = random.randint(4, max(4, min(8, h - 2)))
    r0 = random.randint(1, h - length - 1)
    return {"axis": "v", "c": random.randint(1, w - 2), "r0": r0, "r1": r0 + length, "role": "open"}


def _can_add(global_segments: list[dict], new_segments: list[dict], shape: tuple[int, int]) -> bool:
    old_fence = _segments_to_fence(shape, global_segments)
    new_fence = _segments_to_fence(shape, new_segments)
    if np.any(old_fence & new_fence):
        return False

    markers = _marker_set(global_segments) | _marker_set(new_segments)
    if not _markers_are_isolated(markers):
        return False
    if not _valid_marker_counts(shape, markers):
        return False
    return True


def _sample_module(shape: tuple[int, int], want_feature_bias: bool) -> tuple[list[dict], dict] | None:
    builders = [_rect_module, _hex_module]
    weights = [2, 3] if want_feature_bias else [4, 2]
    builder = random.choices(builders, weights=weights, k=1)[0]
    return builder(shape)


def _expected_output_from_segments(shape: tuple[int, int], segments: list[dict]) -> np.ndarray:
    fence = _segments_to_fence(shape, segments)
    enclosed = _enclosed_mask_from_fence(fence)
    out = np.full(shape, BACKGROUND, dtype=int)
    out[enclosed] = YELLOW
    return out


def _validate_semantics(shape: tuple[int, int], segments: list[dict], marker: int) -> np.ndarray | None:
    markers = _marker_set(segments)
    if not markers or not _markers_are_isolated(markers) or not _valid_marker_counts(shape, markers):
        return None

    inp = _build_input(shape, marker, segments)
    intended_fence = _segments_to_fence(shape, segments)
    inferred_fence = _fence_mask_from_grid(inp, BACKGROUND, marker)
    if not np.array_equal(intended_fence, inferred_fence):
        return None

    out = _expected_output_from_segments(shape, segments)
    if int((out == YELLOW).sum()) < 3:
        return None
    if np.array_equal(out, np.full(shape, BACKGROUND, dtype=int)):
        return None
    if not np.array_equal(main(inp), out):
        return None

    for idx, seg in enumerate(segments):
        reduced = segments[:idx] + segments[idx + 1 :]
        altered = _expected_output_from_segments(shape, reduced)
        if seg["role"] == "open":
            if not np.array_equal(altered, out):
                return None
        else:
            if np.array_equal(altered, out):
                return None

    return inp


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(14, 24)
    w = random.randint(14, 24)
    shape = (h, w)
    marker = random.choice(MARKER_CHOICES)

    segments: list[dict] = []
    featureful = False
    module_count = random.choices([1, 2], weights=[3, 2], k=1)[0]

    for idx in range(module_count):
        placed = False
        for _ in range(80):
            built = _sample_module(shape, want_feature_bias=(idx == 0 and not featureful))
            if built is None:
                continue
            new_segments, meta = built
            if not _can_add(segments, new_segments, shape):
                continue
            segments.extend(new_segments)
            featureful = featureful or bool(meta["feature"])
            placed = True
            break
        if not placed:
            return None

    open_count = random.choices([0, 1], weights=[2, 3], k=1)[0]
    for _ in range(open_count):
        placed = False
        for _ in range(80):
            seg = _open_segment(shape)
            if _can_add(segments, [seg], shape):
                segments.append(seg)
                featureful = True
                placed = True
                break
        if not placed:
            return None

    if not featureful:
        return None
    return _validate_semantics(shape, segments, marker)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid invisible-fence input")
