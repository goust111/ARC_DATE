from __future__ import annotations

import random

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))

# Generator configuration.
GENERATION_RETRIES = 300
RING_COUNT_RANGE = (2, 5)
RING_SHRINK_STEPS = (4, 6, 8)
INPUT_MODES = ("final", "shifted", "skewed")
PALETTE_COLUMN = 0
MIN_RING_LEFT = 2


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_hw(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return r1 - r0 + 1, c1 - c0 + 1


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 < ir0 and oc0 < ic0 and or1 > ir1 and oc1 > ic1


def _find_components(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BG or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": len(coords),
                }
            )
    return comps


def _is_rect_ring(comp: dict[str, object]) -> bool:
    r0, c0, r1, c1 = comp["bbox"]  # type: ignore[index]
    h, w = _bbox_hw((r0, c0, r1, c1))
    area = int(comp["area"])  # type: ignore[arg-type]
    if h < 3 or w < 3 or area != 2 * (h + w) - 4:
        return False

    pixels = set(comp["coords"])  # type: ignore[arg-type]
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r == r0 or r == r1 or c == c0 or c == c1
            if on_border != ((r, c) in pixels):
                return False
    return True


def _extract_palette(singletons: list[dict[str, object]], count: int) -> list[int]:
    by_col: dict[int, list[tuple[int, int]]] = {}
    for comp in singletons:
        (r, c) = comp["coords"][0]  # type: ignore[index]
        by_col.setdefault(c, []).append((r, int(comp["color"])))

    best: tuple[int, int, list[int]] | None = None
    for c, entries in by_col.items():
        entries.sort()
        for start in range(len(entries) - count + 1):
            chunk = entries[start : start + count]
            rows = [r for r, _ in chunk]
            if rows == list(range(rows[0], rows[0] + count)):
                palette = [color for _, color in chunk]
                candidate = (c, rows[0], palette)
                if best is None or candidate[:2] < best[:2]:
                    best = candidate
                break

    if best is None:
        candidates = [(c, sorted(entries)) for c, entries in by_col.items() if len(entries) >= count]
        assert candidates, "could not find a vertical palette"
        c, entries = min(candidates, key=lambda item: (item[0], item[1][0][0]))
        best = (c, entries[0][0], [color for _, color in entries[:count]])

    return best[2]


def _draw_ring(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _centered_bbox(height: int, width: int, box_h: int, box_w: int) -> tuple[int, int, int, int]:
    r0 = (height - box_h) // 2
    c0 = (width - box_w) // 2
    return r0, c0, r0 + box_h - 1, c0 + box_w - 1


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape

    comps = _find_components(g)
    rings = [comp for comp in comps if _is_rect_ring(comp)]
    assert rings, "expected at least one rectangular ring"

    palette = _extract_palette([comp for comp in comps if int(comp["area"]) == 1], len(rings))
    out = np.zeros_like(g)

    for ring in sorted(rings, key=lambda comp: _bbox_hw(comp["bbox"]), reverse=True):  # type: ignore[index]
        bbox = ring["bbox"]  # type: ignore[index]
        depth = 1 + sum(_contains(other["bbox"], bbox) for other in rings if other is not ring)  # type: ignore[index]
        assert 1 <= depth <= len(palette)
        ring_h, ring_w = _bbox_hw(bbox)
        _draw_ring(out, _centered_bbox(height, width, ring_h, ring_w), palette[depth - 1])

    return out


def _sample_palette_and_ring_colors(count: int) -> tuple[list[int], list[int]]:
    if 2 * count <= len(COLORS) and random.random() < 0.75:
        chosen = random.sample(COLORS, 2 * count)
        return chosen[:count], chosen[count:]

    palette = random.sample(COLORS, count)
    for _ in range(100):
        rings = random.sample(COLORS, count)
        if all(rings[i] != palette[i] for i in range(count)):
            return palette, rings
    raise RuntimeError("failed to sample non-degenerate colors")


def _sample_size_chain(outer_size: int, count: int) -> list[int] | None:
    sizes = [outer_size]
    current = outer_size
    for depth in range(1, count):
        remaining = count - depth - 1
        min_next = 3 + 4 * remaining
        steps = [step for step in RING_SHRINK_STEPS if current - step >= min_next]
        if not steps:
            return None
        current -= random.choice(steps)
        sizes.append(current)
    return sizes


def _valid_outer_positions(height: int, width: int, box_h: int, box_w: int) -> list[tuple[int, int]]:
    return [
        (r0, c0)
        for r0 in range(1, height - box_h)
        for c0 in range(MIN_RING_LEFT, width - box_w)
    ]


def _place_concentric_chain(
    outer_top: int,
    outer_left: int,
    heights: list[int],
    widths: list[int],
) -> list[tuple[int, int, int, int]]:
    bboxes: list[tuple[int, int, int, int]] = []
    r0 = outer_top
    c0 = outer_left
    for idx, (box_h, box_w) in enumerate(zip(heights, widths)):
        if idx > 0:
            prev_h = heights[idx - 1]
            prev_w = widths[idx - 1]
            r0 += (prev_h - box_h) // 2
            c0 += (prev_w - box_w) // 2
        bboxes.append((r0, c0, r0 + box_h - 1, c0 + box_w - 1))
    return bboxes


def _gap_pairs(total_gap: int) -> list[tuple[int, int]]:
    return [(a, total_gap - a) for a in range(2, total_gap - 1) if total_gap - a >= 2]


def _place_skewed_chain(
    height: int,
    width: int,
    heights: list[int],
    widths: list[int],
) -> list[tuple[int, int, int, int]] | None:
    outer_positions = _valid_outer_positions(height, width, heights[0], widths[0])
    if not outer_positions:
        return None

    asym_levels = [
        idx
        for idx in range(1, len(heights))
        if heights[idx - 1] - heights[idx] > 4 or widths[idx - 1] - widths[idx] > 4
    ]
    if not asym_levels:
        return None

    forced_level = random.choice(asym_levels)
    outer_top, outer_left = random.choice(outer_positions)
    bboxes = [(outer_top, outer_left, outer_top + heights[0] - 1, outer_left + widths[0] - 1)]
    used_asymmetry = False

    for idx in range(1, len(heights)):
        pr0, pc0, pr1, pc1 = bboxes[-1]
        parent_h = pr1 - pr0 + 1
        parent_w = pc1 - pc0 + 1
        box_h = heights[idx]
        box_w = widths[idx]
        row_pairs = _gap_pairs(parent_h - box_h)
        col_pairs = _gap_pairs(parent_w - box_w)
        if not row_pairs or not col_pairs:
            return None

        if idx == forced_level:
            asym_rows = [pair for pair in row_pairs if pair[0] != pair[1]]
            asym_cols = [pair for pair in col_pairs if pair[0] != pair[1]]
            if asym_rows and (not asym_cols or random.random() < 0.5):
                top_gap, bottom_gap = random.choice(asym_rows)
                left_gap, right_gap = random.choice(col_pairs)
                used_asymmetry = True
            elif asym_cols:
                top_gap, bottom_gap = random.choice(row_pairs)
                left_gap, right_gap = random.choice(asym_cols)
                used_asymmetry = True
            else:
                return None
        else:
            top_gap, bottom_gap = random.choice(row_pairs)
            left_gap, right_gap = random.choice(col_pairs)

        nr0 = pr0 + top_gap
        nc0 = pc0 + left_gap
        nr1 = pr1 - bottom_gap
        nc1 = pc1 - right_gap
        bboxes.append((nr0, nc0, nr1, nc1))

    return bboxes if used_asymmetry else None


def _build_expected_output(
    height: int,
    width: int,
    heights: list[int],
    widths: list[int],
    palette: list[int],
) -> np.ndarray:
    out = np.zeros((height, width), dtype=int)
    for depth, (box_h, box_w) in enumerate(zip(heights, widths), start=1):
        _draw_ring(out, _centered_bbox(height, width, box_h, box_w), palette[depth - 1])
    return out


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(*RING_COUNT_RANGE)
    min_outer = 3 + 4 * (count - 1)

    height = random.randint(max(12, min_outer + 2), min_outer + 10)
    width = random.randint(max(14, min_outer + 4), min_outer + 12)

    outer_height_choices = [v for v in range(min_outer, height - 1) if v % 2 == height % 2]
    outer_width_choices = [v for v in range(min_outer, width - 3) if v % 2 == width % 2]
    if not outer_height_choices or not outer_width_choices:
        return None

    outer_h = random.choice(outer_height_choices)
    outer_w = random.choice(outer_width_choices)
    heights = _sample_size_chain(outer_h, count)
    widths = _sample_size_chain(outer_w, count)
    if heights is None or widths is None:
        return None

    palette, ring_colors = _sample_palette_and_ring_colors(count)
    mode = random.choice(INPUT_MODES)

    centered_outer = _centered_bbox(height, width, heights[0], widths[0])
    centered_boxes = _place_concentric_chain(centered_outer[0], centered_outer[1], heights, widths)

    if mode == "final":
        bboxes = centered_boxes
    elif mode == "shifted":
        positions = [pos for pos in _valid_outer_positions(height, width, heights[0], widths[0]) if pos != centered_outer[:2]]
        if not positions:
            return None
        outer_top, outer_left = random.choice(positions)
        bboxes = _place_concentric_chain(outer_top, outer_left, heights, widths)
    else:
        bboxes = _place_skewed_chain(height, width, heights, widths)
        if bboxes is None:
            positions = [pos for pos in _valid_outer_positions(height, width, heights[0], widths[0]) if pos != centered_outer[:2]]
            if not positions:
                return None
            outer_top, outer_left = random.choice(positions)
            bboxes = _place_concentric_chain(outer_top, outer_left, heights, widths)

    grid = np.zeros((height, width), dtype=int)
    for depth, color in enumerate(palette):
        grid[depth, PALETTE_COLUMN] = color
    for bbox, color in zip(bboxes, ring_colors):
        _draw_ring(grid, bbox, color)

    comps = _find_components(grid)
    rings = [comp for comp in comps if _is_rect_ring(comp)]
    if len(rings) != count:
        return None
    palette_found = _extract_palette([comp for comp in comps if int(comp["area"]) == 1], count)
    if palette_found != palette:
        return None

    expected = _build_expected_output(height, width, heights, widths, palette)
    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_RETRIES} attempts")
