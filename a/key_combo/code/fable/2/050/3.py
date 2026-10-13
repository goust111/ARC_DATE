from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
BAR_COLOR = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
SIDES = ("left", "right", "top", "bottom")
GEN_ATTEMPTS = 400
COUNTEREXAMPLE_PROB = 0.8
SEGMENT_COUNT_CHOICES = [2, 3, 4]
SEGMENT_COUNT_WEIGHTS = [3, 4, 3]
COUNT_POOL = [1, 2, 3, 4]
H_RANGE = (14, 22)
W_RANGE = (14, 20)
BAR_THICKNESS_CHOICES = (1, 2)


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _detect_edge_bar(grid: np.ndarray, bg: int) -> tuple[str, int, int, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[str, int, int]] = []

    def add_candidate(side: str, thickness: int, color: int, expected_area: int) -> None:
        if thickness <= 0 or color == bg:
            return
        if int(np.count_nonzero(g == color)) == expected_area:
            candidates.append((side, thickness, color))

    top_color = int(g[0, 0])
    if top_color != bg and np.all(g[0, :] == top_color):
        t = 0
        while t < h and np.all(g[t, :] == top_color):
            t += 1
        add_candidate("top", t, top_color, t * w)

    bottom_color = int(g[-1, 0])
    if bottom_color != bg and np.all(g[-1, :] == bottom_color):
        t = 0
        while t < h and np.all(g[h - 1 - t, :] == bottom_color):
            t += 1
        add_candidate("bottom", t, bottom_color, t * w)

    left_color = int(g[0, 0])
    if left_color != bg and np.all(g[:, 0] == left_color):
        t = 0
        while t < w and np.all(g[:, t] == left_color):
            t += 1
        add_candidate("left", t, left_color, t * h)

    right_color = int(g[0, -1])
    if right_color != bg and np.all(g[:, -1] == right_color):
        t = 0
        while t < w and np.all(g[:, w - 1 - t] == right_color):
            t += 1
        add_candidate("right", t, right_color, t * h)

    assert len(candidates) == 1, f"expected exactly one edge bar, got {candidates}"
    side, thickness, color = candidates[0]
    mask = np.zeros_like(g, dtype=bool)
    if side == "top":
        mask[:thickness, :] = True
    elif side == "bottom":
        mask[h - thickness :, :] = True
    elif side == "left":
        mask[:, :thickness] = True
    else:
        mask[:, w - thickness :] = True
    return side, thickness, color, mask


def _parse_instance(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = get_bg(g)
    side, thickness, bar_color, bar_mask = _detect_edge_bar(g, bg)
    h, w = g.shape
    parallel = "horizontal" if side in ("top", "bottom") else "vertical"
    comps = _components((g != bg) & (~bar_mask))
    assert len(comps) >= 2, "expected multiple marked segments"

    color_presence: Counter[int] = Counter()
    raw_infos = []
    for coords in comps:
        colors = sorted({int(g[r, c]) for r, c in coords.tolist()})
        assert bg not in colors and bar_color not in colors
        assert len(colors) == 2, f"segment must contain one main color and one marker color: {colors}"
        for color in colors:
            color_presence[color] += 1
        raw_infos.append({"coords": coords, "bbox": bbox_coords(coords), "colors": colors})

    marker_candidates = [color for color, count in color_presence.items() if count == len(comps)]
    assert len(marker_candidates) == 1, f"expected one shared marker color, got {marker_candidates}"
    marker_color = marker_candidates[0]

    segments = []
    for info in raw_infos:
        r0, c0, r1, c1 = info["bbox"]
        main_colors = [color for color in info["colors"] if color != marker_color]
        assert len(main_colors) == 1
        main_color = main_colors[0]

        if parallel == "vertical":
            assert c0 == c1, f"expected vertical segment bbox, got {info['bbox']}"
            line = g[r0 : r1 + 1, c0]
            assert np.all((line == main_color) | (line == marker_color))
            marker_count = int(np.count_nonzero(line == marker_color))
            assert 1 <= marker_count < line.size
            if side == "left":
                rect_bbox = (r0, thickness, r1, c0)
                distance = c0 - thickness
            else:
                rect_bbox = (r0, c0, r1, w - thickness - 1)
                distance = (w - thickness - 1) - c0
            span = (r0, r1)
        else:
            assert r0 == r1, f"expected horizontal segment bbox, got {info['bbox']}"
            line = g[r0, c0 : c1 + 1]
            assert np.all((line == main_color) | (line == marker_color))
            marker_count = int(np.count_nonzero(line == marker_color))
            assert 1 <= marker_count < line.size
            if side == "top":
                rect_bbox = (thickness, c0, r0, c1)
                distance = r0 - thickness
            else:
                rect_bbox = (r0, c0, h - thickness - 1, c1)
                distance = (h - thickness - 1) - r0
            span = (c0, c1)

        assert distance >= 1, "segment must be separated from the edge bar"
        segments.append(
            {
                "bbox": info["bbox"],
                "rect_bbox": rect_bbox,
                "main_color": main_color,
                "marker_count": marker_count,
                "distance": distance,
                "span": span,
            }
        )

    counts = [segment["marker_count"] for segment in segments]
    assert len(set(counts)) == len(counts), "marker counts must be unique for unambiguous layering"
    segments.sort(key=lambda segment: segment["marker_count"])
    return {
        "grid": g,
        "bg": bg,
        "side": side,
        "bar_color": bar_color,
        "bar_thickness": thickness,
        "bar_mask": bar_mask,
        "marker_color": marker_color,
        "parallel": parallel,
        "segments": segments,
    }


def _render(parsed: dict) -> np.ndarray:
    out = parsed["grid"].copy()
    for segment in parsed["segments"]:
        r0, c0, r1, c1 = segment["rect_bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = segment["main_color"]
    return out


def _spans_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(a[0], b[0]) <= min(a[1], b[1])


def _has_distance_counterexample(parsed: dict) -> bool:
    segments = parsed["segments"]
    for i, first in enumerate(segments):
        for second in segments[i + 1 :]:
            if not _spans_overlap(first["span"], second["span"]):
                continue
            later = second if second["marker_count"] > first["marker_count"] else first
            nearer = second if second["distance"] < first["distance"] else first
            if later is not nearer:
                return True
    return False


def _apply_side_transform(grid: np.ndarray, side: str) -> np.ndarray:
    if side == "left":
        return grid.copy()
    if side == "right":
        return np.fliplr(grid).copy()
    if side == "top":
        return grid.T.copy()
    if side == "bottom":
        return np.flipud(grid.T).copy()
    raise ValueError(f"unknown side {side}")


def _choose_columns(width: int, bar_thickness: int, count: int) -> list[int] | None:
    limit = min(width - 2, bar_thickness + 10)
    candidates = list(range(bar_thickness + 2, limit + 1))
    random.shuffle(candidates)
    cols: list[int] = []
    for col in candidates:
        if all(abs(col - other) >= 2 for other in cols):
            cols.append(col)
            if len(cols) == count:
                return sorted(cols)
    return None


def _choose_overlap_start(anchor_start: int, anchor_len: int, length: int, height: int, min_overlap: int) -> int | None:
    low = max(1, anchor_start - length + min_overlap)
    high = min(height - length - 1, anchor_start + anchor_len - min_overlap)
    if low > high:
        return None
    return random.randint(low, high)


def _segment_cells(start: int, length: int, col: int) -> list[tuple[int, int]]:
    return [(start + offset, col) for offset in range(length)]


def _validate_generated_instance(grid: np.ndarray, require_counterexample: bool) -> None:
    parsed = _parse_instance(grid)
    out = _render(parsed)
    assert out.shape == grid.shape
    assert max(out.shape) <= 30
    assert np.array_equal(out[parsed["bar_mask"]], grid[parsed["bar_mask"]])
    assert not np.any(out == parsed["marker_color"])

    overlap_pairs = 0
    for i, first in enumerate(parsed["segments"]):
        r0, c0, r1, c1 = first["rect_bbox"]
        assert np.all(out[r0 : r1 + 1, c0 : c1 + 1] >= 0)
        for second in parsed["segments"][i + 1 :]:
            if _spans_overlap(first["span"], second["span"]):
                overlap_pairs += 1
    assert overlap_pairs >= 1, "generator must produce at least one overlapping projection pair"
    if require_counterexample:
        assert _has_distance_counterexample(parsed), "generator must reject distance-order-only examples"


def main(grid: np.ndarray) -> np.ndarray:
    """Project each marked parallel segment toward the solid edge bar, and let higher vote counts overwrite lower ones.

    Input structure:
    - A black background contains exactly one solid edge bar and several one-cell-thick line segments parallel to that bar.
    - Each segment has its own main color, while some cells on every segment are replaced by one shared marker color.

    Rule:
    - Count the marker cells on each segment, restore the whole segment to its main color, and extend it straight toward the edge bar into a same-span rectangle that stops before the bar.
    - Draw those rectangles in ascending marker-count order so segments with more markers are painted later and overwrite earlier overlaps.
    - Keep the edge bar unchanged, and leave no marker pixels in the output.

    Key insight:
    - The shared marker color is only a vote for draw order; overlap resolution depends on those counts rather than on which segment is nearer to the bar.
    """
    parsed = _parse_instance(np.asarray(grid, dtype=int))
    return _render(parsed)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        n_segments = random.choices(SEGMENT_COUNT_CHOICES, weights=SEGMENT_COUNT_WEIGHTS, k=1)[0]
        require_counterexample = random.random() < COUNTEREXAMPLE_PROB

        counts = random.sample(COUNT_POOL, n_segments)
        if require_counterexample and counts[0] > counts[1]:
            counts[0], counts[1] = counts[1], counts[0]
        if not require_counterexample and counts[0] < counts[1]:
            counts[0], counts[1] = counts[1], counts[0]

        h = random.randint(max(H_RANGE[0], max(counts) + 6), H_RANGE[1])
        w = random.randint(W_RANGE[0], W_RANGE[1])
        bar_thickness = random.choice(BAR_THICKNESS_CHOICES)
        cols = _choose_columns(w, bar_thickness, n_segments)
        if cols is None:
            continue

        lengths = [random.randint(count + 2, min(count + 5, h - 2)) for count in counts]
        start0 = random.randint(1, h - lengths[0] - 1)
        start1 = _choose_overlap_start(start0, lengths[0], lengths[1], h, min_overlap=2)
        if start1 is None:
            continue
        starts = [start0, start1]
        for i in range(2, n_segments):
            if random.random() < 0.55:
                anchor_idx = random.choice([0, 1])
                start = _choose_overlap_start(starts[anchor_idx], lengths[anchor_idx], lengths[i], h, min_overlap=1)
                if start is None:
                    start = random.randint(1, h - lengths[i] - 1)
            else:
                start = random.randint(1, h - lengths[i] - 1)
            starts.append(start)

        marker_color = random.choice(PALETTE)
        main_colors = random.sample([color for color in PALETTE if color != marker_color], n_segments)

        grid = np.full((h, w), BG, dtype=int)
        grid[:, :bar_thickness] = BAR_COLOR
        for start, length, col, count, main_color in zip(starts, lengths, cols, counts, main_colors):
            cells = _segment_cells(start, length, col)
            for r, c in cells:
                grid[r, c] = main_color
            marker_rows = random.sample(range(length), count)
            for row_offset in marker_rows:
                grid[start + row_offset, col] = marker_color

        side = random.choice(SIDES)
        transformed = _apply_side_transform(grid, side)
        try:
            _validate_generated_instance(transformed, require_counterexample=require_counterexample)
        except AssertionError:
            continue
        return transformed

    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
