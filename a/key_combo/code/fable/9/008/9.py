from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
GRAY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]

GEN_ATTEMPTS = 400
SHAPE_ATTEMPTS = 200
PLACEMENT_ATTEMPTS = 300


def _neighbors(connectivity: int) -> list[tuple[int, int]]:
    if int(connectivity) == 8:
        return [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
    return [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _components_of_color(grid: np.ndarray, color: int, connectivity: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    steps = _neighbors(connectivity)
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in steps:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(comp)
    return parts


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_mask(coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask, (r0, c0, r1, c1)


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in _neighbors(4):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _top_profile(mask: np.ndarray) -> tuple[int, ...]:
    h, w = mask.shape
    tops = []
    for c in range(w):
        rows = np.flatnonzero(mask[:, c])
        assert len(rows) > 0
        tops.append(int(rows[0]))
    return tuple(tops)


def _bottom_profile(mask: np.ndarray) -> tuple[int, ...]:
    h, w = mask.shape
    bottoms = []
    for c in range(w):
        rows = np.flatnonzero(mask[:, c])
        assert len(rows) > 0
        bottoms.append(int(rows[-1]))
    return tuple(bottoms)


def _sample_and_markers(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int, int]]]:
    colored_components: list[tuple[int, list[tuple[int, int]]]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in (BG, GRAY)):
        for comp in _components_of_color(grid, color, connectivity=4):
            colored_components.append((color, comp))

    sample_candidates = [(color, comp) for color, comp in colored_components if len(comp) > 1]
    assert len(sample_candidates) == 1, "expected exactly one multi-cell colored sample"
    sample_color, sample_coords = sample_candidates[0]
    sample_mask, _ = _crop_mask(sample_coords)
    assert _is_connected(sample_mask)

    markers = []
    for color, comp in colored_components:
        if len(comp) == 1:
            r, c = comp[0]
            markers.append((int(r), int(c), int(color)))
    assert markers, "expected colored marker pixels"
    return sample_mask, sample_coords, markers


def _gray_trace_infos(grid: np.ndarray, expected_width: int) -> list[dict]:
    infos = []
    for comp in _components_of_color(grid, GRAY, connectivity=8):
        rows_by_col: dict[int, list[int]] = defaultdict(list)
        for r, c in comp:
            rows_by_col[int(c)].append(int(r))
        cols = sorted(rows_by_col)
        assert cols, "empty gray component"
        assert len(cols) == expected_width, "trace width must match sample width"
        assert cols == list(range(cols[0], cols[0] + expected_width)), "trace columns must be consecutive"
        ordered_rows = []
        for col in cols:
            assert len(rows_by_col[col]) == 1, "trace must have exactly one gray cell per column"
            ordered_rows.append(rows_by_col[col][0])
        infos.append(
            {
                "coords": comp,
                "left_col": cols[0],
                "right_col": cols[-1],
                "rows": tuple(ordered_rows),
            }
        )
    return infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the sample sprite back above each gray bottom-trace and recolor it to the nearby marker color.

    Input structure:
    - A black background contains one complete multi-cell monochrome sprite sample.
    - Elsewhere, several gray traces appear; each trace occupies one cell in every column
      of a consecutive width equal to the sample bbox width.
    - One colored marker pixel sits exactly one cell below the leftmost gray cell of each trace.

    Rule:
    - Read the sample sprite's occupied mask and, for each sample column, its lowest occupied row.
    - For each gray trace, align the sample bbox to the same column interval and place the sample
      so that each column's lowest sample cell lands exactly one cell above that trace column's gray cell.
    - Paint each stamped copy entirely in that trace's marker color, erase the marker pixel, keep
      the gray trace unchanged, and leave the original sample in place with its original color.

    Key insight:
    - The gray polyline is not a partial sprite; it is the one-dimensional projection of the sample's
      per-column bottom contour, so the whole sprite position is recovered only by inverting that contour.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    sample_mask, _sample_coords, markers = _sample_and_markers(g)
    sh, sw = sample_mask.shape
    sample_bottom = _bottom_profile(sample_mask)
    out = g.copy()

    trace_infos = _gray_trace_infos(g, sw)
    assert len(trace_infos) == len(markers), "each trace must have exactly one marker"
    used: set[int] = set()

    sample_local_cells = [(int(r), int(c)) for r, c in np.argwhere(sample_mask)]
    h, w = g.shape
    for marker_r, marker_c, marker_color in markers:
        matches = [
            (idx, info)
            for idx, info in enumerate(trace_infos)
            if idx not in used and info["left_col"] == marker_c and info["rows"][0] == marker_r - 1
        ]
        assert len(matches) == 1, "marker must identify a unique left-aligned trace"
        idx, info = matches[0]
        used.add(idx)

        top_rows = [info["rows"][dc] - 1 - sample_bottom[dc] for dc in range(sw)]
        assert len(set(top_rows)) == 1, "trace must invert to one consistent placement"
        top = top_rows[0]
        assert 0 <= top and top + sh <= h
        left = marker_c

        for dr, dc in sample_local_cells:
            out[top + dr, left + dc] = marker_color
        out[marker_r, marker_c] = BG

    assert len(used) == len(trace_infos)
    return out


def _make_sample_mask() -> np.ndarray:
    for _ in range(SHAPE_ATTEMPTS):
        sh = random.randint(4, 7)
        sw = random.randint(3, 6)

        bottom = [random.randint(1, sh - 1)]
        for _c in range(1, sw):
            lo = max(1, bottom[-1] - 1)
            hi = min(sh - 1, bottom[-1] + 1)
            bottom.append(random.randint(lo, hi))
        lift = (sh - 1) - max(bottom)
        bottom = [b + lift for b in bottom]
        if len(set(bottom)) < 2:
            continue

        top = [random.randint(0, min(1, bottom[0]))]
        for c in range(1, sw):
            lo = max(0, top[-1] - 1)
            hi = min(bottom[c], top[-1] + 1)
            top.append(random.randint(lo, hi))
        shift = min(top)
        top = [t - shift for t in top]
        if max(top) == 0:
            continue
        if any(t > b for t, b in zip(top, bottom)):
            continue

        mask = np.zeros((sh, sw), dtype=bool)
        for c in range(sw):
            mask[top[c] : bottom[c] + 1, c] = True

        if mask.all():
            continue
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if not _is_connected(mask):
            continue
        if len(set(_bottom_profile(mask))) < 2:
            continue
        if max(_top_profile(mask)) == 0:
            continue
        return mask
    raise RuntimeError("failed to build a non-degenerate sample sprite")


def _rect_overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar, ac, ah, aw = a
    br, bc, bh, bw = b
    return not (
        ar + ah + pad <= br
        or br + bh + pad <= ar
        or ac + aw + pad <= bc
        or bc + bw + pad <= ac
    )


def _place_blocks(sizes: list[tuple[int, int]], h: int, w: int) -> list[tuple[int, int]] | None:
    order = sorted(range(len(sizes)), key=lambda i: sizes[i][0] * sizes[i][1], reverse=True)
    placed: list[tuple[int, int, int, int] | None] = [None] * len(sizes)
    for idx in order:
        bh, bw = sizes[idx]
        options = [(r, c) for r in range(h - bh + 1) for c in range(w - bw + 1)]
        random.shuffle(options)
        for r, c in options:
            rect = (r, c, bh, bw)
            if all(existing is None or not _rect_overlaps(rect, existing, pad=1) for existing in placed):
                placed[idx] = rect
                break
        if placed[idx] is None:
            return None
    return [(rect[0], rect[1]) for rect in placed if rect is not None]


def _render_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    for dr, dc in np.argwhere(mask):
        grid[top + int(dr), left + int(dc)] = int(color)


def _extract_trace_semantics(grid: np.ndarray) -> dict:
    sample_mask, sample_coords, markers = _sample_and_markers(grid)
    sh, sw = sample_mask.shape
    sample_bbox = _bbox(sample_coords)
    sample_top, sample_left, _, _ = sample_bbox
    sample_bottom = _bottom_profile(sample_mask)
    sample_top_profile = _top_profile(sample_mask)

    assert len(set(sample_bottom)) >= 2, "bottom trace must vary"
    assert max(sample_top_profile) > 0, "sample top must also vary to block rectangle shortcuts"

    traces = _gray_trace_infos(grid, sw)
    assert len(traces) == len(markers)

    windows = [(sample_top, sample_left, sh, sw)]
    placements = []
    used: set[int] = set()
    for marker_r, marker_c, marker_color in markers:
        matches = [
            (idx, info)
            for idx, info in enumerate(traces)
            if idx not in used and info["left_col"] == marker_c and info["rows"][0] == marker_r - 1
        ]
        assert len(matches) == 1
        idx, info = matches[0]
        used.add(idx)

        tops = [info["rows"][dc] - 1 - sample_bottom[dc] for dc in range(sw)]
        assert len(set(tops)) == 1
        top = tops[0]
        assert 0 <= top <= grid.shape[0] - sh
        placement = (top, marker_c, sh, sw)
        for old in windows:
            assert not _rect_overlaps(placement, old, pad=0), "stamped sprite windows must stay separate"
        windows.append(placement)
        placements.append(
            {
                "top": top,
                "left": marker_c,
                "color": marker_color,
                "rows": info["rows"],
                "marker": (marker_r, marker_c),
            }
        )

    return {
        "sample_mask": sample_mask,
        "sample_bbox": sample_bbox,
        "sample_top": sample_top,
        "sample_left": sample_left,
        "sample_bottom": sample_bottom,
        "placements": placements,
    }


def _make_case() -> tuple[np.ndarray, np.ndarray]:
    sample_mask = _make_sample_mask()
    sh, sw = sample_mask.shape
    trace_count = random.randint(2, 4)
    sample_color, *marker_colors = random.sample(PALETTE, trace_count + 1)

    canvas_h = random.randint(max(12, sh + 5), 24)
    canvas_w = random.randint(max(12, sw + 5), 24)

    sample_size = (sh, sw)
    trace_sizes = [(sh + 2, sw) for _ in range(trace_count)]
    placed = _place_blocks([sample_size, *trace_sizes], canvas_h, canvas_w)
    if placed is None:
        raise RuntimeError("failed to place non-overlapping sample/trace blocks")

    inp = np.full((canvas_h, canvas_w), BG, dtype=int)
    out = np.full((canvas_h, canvas_w), BG, dtype=int)

    sample_top, sample_left = placed[0]
    _render_mask(inp, sample_mask, sample_top, sample_left, sample_color)
    _render_mask(out, sample_mask, sample_top, sample_left, sample_color)

    bottom = _bottom_profile(sample_mask)
    for (trace_top, trace_left), marker_color in zip(placed[1:], marker_colors):
        _render_mask(out, sample_mask, trace_top, trace_left, marker_color)
        for dc, low in enumerate(bottom):
            gray_r = trace_top + low + 1
            gray_c = trace_left + dc
            inp[gray_r, gray_c] = GRAY
            out[gray_r, gray_c] = GRAY
        marker_r = trace_top + bottom[0] + 2
        inp[marker_r, trace_left] = marker_color

    return inp, out


def _verify_case(inp: np.ndarray, expected: np.ndarray) -> None:
    assert inp.ndim == 2 and expected.ndim == 2
    assert max(inp.shape) <= 30 and max(expected.shape) <= 30

    semantics = _extract_trace_semantics(inp)
    placements = semantics["placements"]
    assert 2 <= len(placements) <= 4

    actual = main(inp)
    assert np.array_equal(actual, expected), "solver output mismatch"

    sample_mask = semantics["sample_mask"]
    sample_top = semantics["sample_top"]
    sample_left = semantics["sample_left"]
    sh, sw = sample_mask.shape

    occupied = np.zeros(inp.shape, dtype=bool)
    sample_cells = [(sample_top + int(r), sample_left + int(c)) for r, c in np.argwhere(sample_mask)]
    for r, c in sample_cells:
        occupied[r, c] = True

    for item in placements:
        top = item["top"]
        left = item["left"]
        marker_r, marker_c = item["marker"]
        assert int(inp[marker_r, marker_c]) == int(item["color"])
        assert int(expected[marker_r, marker_c]) == BG
        assert item["rows"][0] == marker_r - 1
        for dc, row in enumerate(item["rows"]):
            assert int(inp[row, left + dc]) == GRAY
            assert int(expected[row, left + dc]) == GRAY
        for dr, dc in np.argwhere(sample_mask):
            rr = top + int(dr)
            cc = left + int(dc)
            assert not occupied[rr, cc], "sample and stamped copies must not overlap"
            occupied[rr, cc] = True
            assert int(expected[rr, cc]) == int(item["color"])


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        try:
            inp, expected = _make_case()
            _verify_case(inp, expected)
            return inp
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid bottom-trace inversion stamp puzzle")
