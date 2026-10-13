from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
FRAME = 9
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8]

GRID_H_RANGE = (18, 29)
GRID_W_RANGE = (18, 29)
FRAME_THICKNESS_CHOICES = (2, 3)
FRAME_MARGIN = 1
FRAME_H_PAD = 6
FRAME_W_PAD = 6
GAP_COUNT_RANGE = (2, 6)
GAP_LEN_RANGE = (2, 6)
GAP_SEPARATION = 1
GEN_RETRIES = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(comp)
    return parts


def _runs(mask_1d: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, flag in enumerate(mask_1d.tolist()):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask_1d) - 1))
    return runs


def _largest_non_bg_color(grid: np.ndarray, bg: int) -> int:
    colors = [int(c) for c in np.unique(grid) if int(c) != bg]
    assert colors, "expected non-background colors"
    return max(colors, key=lambda color: int(np.count_nonzero(grid == color)))


def _infer_thickness(grid: np.ndarray, frame_color: int, bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    limit = min(sub.shape) // 2

    def corner_depth(which: str) -> int:
        depth = 0
        for k in range(1, limit + 1):
            if which == "tl":
                block = sub[:k, :k]
            elif which == "tr":
                block = sub[:k, -k:]
            elif which == "bl":
                block = sub[-k:, :k]
            else:
                block = sub[-k:, -k:]
            if np.all(block == frame_color):
                depth = k
            else:
                break
        return depth

    depths = [corner_depth(name) for name in ("tl", "tr", "bl", "br")]
    thickness = min(depths)
    assert thickness >= 1, f"invalid frame thickness: {depths}"
    return int(thickness)


def _collect_gaps(
    grid: np.ndarray,
    bg: int,
    bbox: tuple[int, int, int, int],
    thickness: int,
) -> dict[str, list[tuple[int, int]]]:
    r0, c0, r1, c1 = bbox
    top_band = np.all(grid[r0 : r0 + thickness, c0 + thickness : c1 - thickness + 1] == bg, axis=0)
    bottom_band = np.all(
        grid[r1 - thickness + 1 : r1 + 1, c0 + thickness : c1 - thickness + 1] == bg,
        axis=0,
    )
    left_band = np.all(grid[r0 + thickness : r1 - thickness + 1, c0 : c0 + thickness] == bg, axis=1)
    right_band = np.all(
        grid[r0 + thickness : r1 - thickness + 1, c1 - thickness + 1 : c1 + 1] == bg,
        axis=1,
    )

    return {
        "top": [(c0 + thickness + a, c0 + thickness + b) for a, b in _runs(top_band)],
        "bottom": [(c0 + thickness + a, c0 + thickness + b) for a, b in _runs(bottom_band)],
        "left": [(r0 + thickness + a, r0 + thickness + b) for a, b in _runs(left_band)],
        "right": [(r0 + thickness + a, r0 + thickness + b) for a, b in _runs(right_band)],
    }


def _gap_rect(
    side: str,
    span: tuple[int, int],
    bbox: tuple[int, int, int, int],
    thickness: int,
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    a, b = span
    if side == "top":
        return (r0, a, r0 + thickness - 1, b)
    if side == "bottom":
        return (r1 - thickness + 1, a, r1, b)
    if side == "left":
        return (a, c0, b, c0 + thickness - 1)
    if side == "right":
        return (a, c1 - thickness + 1, b, c1)
    raise AssertionError(f"unknown side: {side}")


def _rect_to_coords(rect: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _find_marker_target(
    marker: tuple[int, int],
    gaps: dict[str, list[tuple[int, int]]],
    bbox: tuple[int, int, int, int],
    thickness: int,
) -> tuple[str, tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    mr, mc = marker

    if mr == r0 - 1 and c0 + thickness <= mc <= c1 - thickness:
        side = "top"
        coord = mc
    elif mr == r1 + 1 and c0 + thickness <= mc <= c1 - thickness:
        side = "bottom"
        coord = mc
    elif mc == c0 - 1 and r0 + thickness <= mr <= r1 - thickness:
        side = "left"
        coord = mr
    elif mc == c1 + 1 and r0 + thickness <= mr <= r1 - thickness:
        side = "right"
        coord = mr
    else:
        raise AssertionError(f"marker is not aligned with the frame: {marker}")

    hits = [span for span in gaps[side] if span[0] <= coord <= span[1]]
    assert len(hits) == 1, f"marker {marker} must identify exactly one gap on {side}"
    return side, hits[0]


def _inside_bbox(cell: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r, c = cell
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def _rect_clear(mask: np.ndarray, rect: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = rect
    return not bool(mask[r0 : r1 + 1, c0 : c1 + 1].any())


def _paint_rect(grid: np.ndarray, rect: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = rect
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _frame_source_rect(
    side: str,
    span: tuple[int, int],
    shift: int,
    bbox: tuple[int, int, int, int],
    thickness: int,
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    a, b = span
    if side == "top":
        return (r0 + shift, a, r0 + shift + thickness - 1, b)
    if side == "bottom":
        return (r1 - shift - thickness + 1, a, r1 - shift, b)
    if side == "left":
        return (a, c0 + shift, b, c0 + shift + thickness - 1)
    if side == "right":
        return (a, c1 - shift - thickness + 1, b, c1 - shift)
    raise AssertionError(f"unknown side: {side}")


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    frame_color = _largest_non_bg_color(g, bg)
    frame_cells = np.argwhere(g == frame_color)
    assert frame_cells.size > 0, "missing frame"
    bbox = _bbox(frame_cells)
    thickness = _infer_thickness(g, frame_color, bbox)
    gaps = _collect_gaps(g, bg, bbox, thickness)

    out = g.copy()
    colors = [int(c) for c in np.unique(g) if int(c) not in (bg, frame_color)]
    for color in colors:
        coords = [tuple(map(int, cell)) for cell in np.argwhere(g == color)]
        assert coords, f"missing color cells for {color}"

        outside = [cell for cell in coords if not _inside_bbox(cell, bbox)]
        inside = [cell for cell in coords if _inside_bbox(cell, bbox)]
        assert len(outside) == 1, f"color {color} must have exactly one outside marker"
        assert inside, f"color {color} must have an interior fragment"

        inside_mask = np.zeros(g.shape, dtype=bool)
        for r, c in inside:
            inside_mask[r, c] = True
        comps = _components(inside_mask)
        assert len(comps) == 1, f"color {color} fragment must be 4-connected"

        side, span = _find_marker_target(outside[0], gaps, bbox, thickness)
        target_rect = _gap_rect(side, span, bbox, thickness)
        target_shape = _rect_to_coords((0, 0, target_rect[2] - target_rect[0], target_rect[3] - target_rect[1]))

        frag_bbox = _bbox(np.array(inside, dtype=int))
        frag_shape = _rect_to_coords((0, 0, frag_bbox[2] - frag_bbox[0], frag_bbox[3] - frag_bbox[1]))
        frag_coords = {(r - frag_bbox[0], c - frag_bbox[1]) for r, c in inside}
        assert frag_coords == frag_shape == target_shape, f"fragment color {color} must match the gap rectangle"

        if side in ("top", "bottom"):
            assert frag_bbox[1] == span[0] and frag_bbox[3] == span[1], (
                f"horizontal fragment {color} is not aligned with its gap"
            )
        else:
            assert frag_bbox[0] == span[0] and frag_bbox[2] == span[1], (
                f"vertical fragment {color} is not aligned with its gap"
            )

        for r, c in outside + inside:
            out[r, c] = bg
        _paint_rect(out, target_rect, frame_color)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pull each colored fragment straight into the same-colored frame notch, then merge it into the frame.

    Input structure:
    - A thick maroon rectangular frame sits inside the canvas.
    - Each side of the frame may contain one or more black rectangular gaps.
    - A single colored marker pixel outside a gap identifies the matching colored fragment inside the frame.

    Rule:
    - Match each fragment to the unique outside marker with the same color.
    - Use the marker position to determine which side gap the fragment belongs to.
    - Move the fragment only along the inward normal of that side until it fills the gap.
    - Recolor the embedded fragment to maroon, erase the marker, and clear the fragment's original position to black.
    """
    return _transform(grid)


def _draw_full_frame(
    shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
    thickness: int,
) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    r0, c0, r1, c1 = bbox
    grid[r0 : r0 + thickness, c0 : c1 + 1] = FRAME
    grid[r1 - thickness + 1 : r1 + 1, c0 : c1 + 1] = FRAME
    grid[r0 : r1 + 1, c0 : c0 + thickness] = FRAME
    grid[r0 : r1 + 1, c1 - thickness + 1 : c1 + 1] = FRAME
    return grid


def _max_interval_count(length: int) -> int:
    return max(0, (length + GAP_SEPARATION) // (GAP_LEN_RANGE[0] + GAP_SEPARATION))


def _sample_intervals(length: int, count: int) -> list[tuple[int, int]] | None:
    max_len = min(GAP_LEN_RANGE[1], length)
    if count == 0:
        return []
    if max_len < GAP_LEN_RANGE[0]:
        return None

    candidates = [
        (start, end)
        for start in range(length)
        for end in range(start + GAP_LEN_RANGE[0] - 1, min(length - 1, start + max_len - 1) + 1)
    ]
    random.shuffle(candidates)

    def fits(span: tuple[int, int], chosen: list[tuple[int, int]]) -> bool:
        s0, s1 = span
        return all(s1 + GAP_SEPARATION < c0 or s0 > c1 + GAP_SEPARATION for c0, c1 in chosen)

    def search(chosen: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
        if len(chosen) == count:
            return sorted(chosen)
        pool = candidates[:]
        random.shuffle(pool)
        for span in pool:
            if fits(span, chosen):
                chosen.append(span)
                found = search(chosen)
                if found is not None:
                    return found
                chosen.pop()
        return None

    return search([])


def _choose_side_counts(capacity: dict[str, int]) -> Counter[str] | None:
    feasible = [side for side, cap in capacity.items() if cap > 0]
    if len(feasible) < 2:
        return None

    total_cap = sum(capacity.values())
    hi = min(GAP_COUNT_RANGE[1], total_cap, len(PALETTE))
    lo = min(GAP_COUNT_RANGE[0], hi)
    if lo < 2:
        return None

    for _ in range(200):
        count = random.randint(lo, hi)
        picks = [random.choice(feasible) for _ in range(count)]
        side_counts: Counter[str] = Counter(picks)
        if len(side_counts) < 2:
            continue
        if all(side_counts[side] <= capacity[side] for side in side_counts):
            return side_counts
    return None


def _marker_cell(side: str, span: tuple[int, int], bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    coord = random.randint(span[0], span[1])
    if side == "top":
        return (r0 - 1, coord)
    if side == "bottom":
        return (r1 + 1, coord)
    if side == "left":
        return (coord, c0 - 1)
    if side == "right":
        return (coord, c1 + 1)
    raise AssertionError(f"unknown side: {side}")


def _place_sources(
    shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
    thickness: int,
    specs: list[dict[str, object]],
) -> dict[int, tuple[int, int, int, int]] | None:
    h, w = shape
    occupied = np.zeros((h, w), dtype=bool)
    placement: dict[int, tuple[int, int, int, int]] = {}

    frame_h = bbox[2] - bbox[0] + 1
    frame_w = bbox[3] - bbox[1] + 1
    order = sorted(
        range(len(specs)),
        key=lambda idx: (
            -(int(specs[idx]["span"][1]) - int(specs[idx]["span"][0]) + 1),
            specs[idx]["side"],
        ),
    )

    def candidates_for(spec: dict[str, object]) -> list[int]:
        side = str(spec["side"])
        max_shift = frame_h - 2 * thickness if side in ("top", "bottom") else frame_w - 2 * thickness
        if max_shift < thickness:
            return []
        lo = thickness + 1 if max_shift >= thickness + 1 else thickness
        values = list(range(lo, max_shift + 1))
        random.shuffle(values)
        return values

    def search(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        spec = specs[idx]
        side = str(spec["side"])
        span = tuple(spec["span"])
        for shift in candidates_for(spec):
            rect = _frame_source_rect(side, span, shift, bbox, thickness)
            r0, c0, r1, c1 = rect
            if not (0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w):
                continue
            if not _rect_clear(occupied, rect):
                continue
            occupied[r0 : r1 + 1, c0 : c1 + 1] = True
            placement[idx] = rect
            if search(pos + 1):
                return True
            occupied[r0 : r1 + 1, c0 : c1 + 1] = False
            placement.pop(idx, None)
        return False

    return placement if search(0) else None


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    thickness = random.choice(FRAME_THICKNESS_CHOICES)

    min_frame_h = 4 * thickness + FRAME_H_PAD
    min_frame_w = 4 * thickness + FRAME_W_PAD
    max_frame_h = h - 2 * FRAME_MARGIN
    max_frame_w = w - 2 * FRAME_MARGIN
    if min_frame_h > max_frame_h or min_frame_w > max_frame_w:
        return None

    frame_h = random.randint(min_frame_h, max_frame_h)
    frame_w = random.randint(min_frame_w, max_frame_w)
    r0 = random.randint(FRAME_MARGIN, h - frame_h - FRAME_MARGIN)
    c0 = random.randint(FRAME_MARGIN, w - frame_w - FRAME_MARGIN)
    bbox = (r0, c0, r0 + frame_h - 1, c0 + frame_w - 1)

    inner_h = frame_h - 2 * thickness
    inner_w = frame_w - 2 * thickness
    if inner_h < thickness or inner_w < thickness:
        return None

    capacity = {
        "top": _max_interval_count(inner_w),
        "bottom": _max_interval_count(inner_w),
        "left": _max_interval_count(inner_h),
        "right": _max_interval_count(inner_h),
    }
    side_counts = _choose_side_counts(capacity)
    if side_counts is None:
        return None

    side_spans: dict[str, list[tuple[int, int]]] = {}
    for side, count in side_counts.items():
        axis_len = inner_w if side in ("top", "bottom") else inner_h
        spans = _sample_intervals(axis_len, count)
        if spans is None:
            return None
        side_spans[side] = spans

    specs: list[dict[str, object]] = []
    color_pool = random.sample(PALETTE, sum(side_counts.values()))
    color_idx = 0
    for side in ("top", "bottom", "left", "right"):
        for local_span in side_spans.get(side, []):
            if side in ("top", "bottom"):
                span = (c0 + thickness + local_span[0], c0 + thickness + local_span[1])
            else:
                span = (r0 + thickness + local_span[0], r0 + thickness + local_span[1])
            specs.append({"side": side, "span": span, "color": color_pool[color_idx]})
            color_idx += 1

    random.shuffle(specs)
    placements = _place_sources((h, w), bbox, thickness, specs)
    if placements is None:
        return None

    expected = _draw_full_frame((h, w), bbox, thickness)
    inp = expected.copy()

    for spec in specs:
        side = str(spec["side"])
        span = tuple(spec["span"])
        _paint_rect(inp, _gap_rect(side, span, bbox, thickness), BG)

    marker_cells: set[tuple[int, int]] = set()
    for idx, spec in enumerate(specs):
        side = str(spec["side"])
        span = tuple(spec["span"])
        color = int(spec["color"])
        rect = placements[idx]
        marker = _marker_cell(side, span, bbox)
        if marker in marker_cells or inp[marker] != BG:
            return None
        marker_cells.add(marker)
        _paint_rect(inp, rect, color)
        inp[marker] = color

    solved = _transform(inp)
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid magnetic-notch input")
