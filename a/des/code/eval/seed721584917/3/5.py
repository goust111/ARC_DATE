from __future__ import annotations

import random

import numpy as np


MAGENTA = 6

# Generator configuration.
GRID_MAX = 30
PAIR_COUNT_RANGE = (3, 5)
RECT_HEIGHT_RANGE = (4, 6)
RECT_WIDTH_RANGE = (4, 7)
GAP_RANGE = (3, 5)
MARGIN_RANGE = (1, 3)
PATTERN_SIZE_RANGE = (4, 10)
GEN_RETRIES = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _pair_signature(points: list[tuple[int, int]]) -> tuple[tuple[int, int, int, int], bool, bool]:
    (r0, c0), (r1, c1) = points
    assert r0 != r1 and c0 != c1, "anchor pairs must define a non-degenerate rectangle"

    top, bottom = (points[0], points[1]) if points[0][0] < points[1][0] else (points[1], points[0])
    left, right = (points[0], points[1]) if points[0][1] < points[1][1] else (points[1], points[0])

    bbox = (min(r0, r1), min(c0, c1), max(r0, r1), max(c0, c1))
    upper_is_left = bool(top[1] < bottom[1])
    left_is_top = bool(left[0] < right[0])
    return bbox, upper_is_left, left_is_top


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the magenta pattern from the source anchor pair to every target anchor pair.

    The source rectangle is the only same-color anchor pair whose rectangle contains
    magenta pixels. Each target rectangle has the same size as the source rectangle;
    the target pair's diagonal orientation decides whether the pattern is mirrored
    horizontally and/or vertically before recoloring it to the target anchor color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    pair_infos: list[dict[str, object]] = []

    for color in sorted(int(c) for c in np.unique(g) if int(c) not in (bg, MAGENTA)):
        pts = [tuple(map(int, pt)) for pt in np.argwhere(g == color)]
        assert len(pts) == 2, f"color {color} must appear exactly twice as an anchor pair"
        bbox, upper_is_left, left_is_top = _pair_signature(pts)
        r0, c0, r1, c1 = bbox
        region = g[r0 : r1 + 1, c0 : c1 + 1]
        pair_infos.append(
            {
                "color": color,
                "bbox": bbox,
                "upper_is_left": upper_is_left,
                "left_is_top": left_is_top,
                "has_pattern": bool(np.any(region == MAGENTA)),
            }
        )

    sources = [info for info in pair_infos if bool(info["has_pattern"])]
    assert len(sources) == 1, "expected exactly one source anchor pair"
    source = sources[0]

    sr0, sc0, sr1, sc1 = source["bbox"]
    template = g[sr0 : sr1 + 1, sc0 : sc1 + 1] == MAGENTA
    assert bool(template.any()), "source rectangle must contain a magenta pattern"
    source_shape = template.shape

    out = g.copy()
    for info in pair_infos:
        if info is source:
            continue

        r0, c0, r1, c1 = info["bbox"]
        assert (r1 - r0 + 1, c1 - c0 + 1) == source_shape, "target rectangles must match the source size"

        mask = template
        if bool(info["upper_is_left"]) != bool(source["upper_is_left"]):
            mask = np.fliplr(mask)
        if bool(info["left_is_top"]) != bool(source["left_is_top"]):
            mask = np.flipud(mask)

        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        color = int(info["color"])
        allowed = (patch == bg) | (patch == color)
        assert bool(np.all((~mask) | allowed)), "target rectangles should only contain background and anchors"
        patch[mask & (patch == bg)] = color

    return out


def _grow_pattern(height: int, width: int) -> np.ndarray | None:
    corners = {(0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)}
    cells = [(r, c) for r in range(height) for c in range(width) if (r, c) not in corners]
    if len(cells) < PATTERN_SIZE_RANGE[0]:
        return None

    max_size = min(PATTERN_SIZE_RANGE[1], len(cells))
    target_size = random.randint(PATTERN_SIZE_RANGE[0], max_size)

    for _ in range(120):
        start = random.choice(cells)
        chosen = {start}

        while len(chosen) < target_size:
            frontier: set[tuple[int, int]] = set()
            for r, c in chosen:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in corners and (nr, nc) not in chosen:
                        frontier.add((nr, nc))
            if not frontier:
                break
            if random.random() < 0.8:
                chosen.add(random.choice(tuple(frontier)))
            else:
                chosen.add(random.choice(cells))

        if len(chosen) != target_size:
            continue

        mask = np.zeros((height, width), dtype=bool)
        for r, c in chosen:
            mask[r, c] = True

        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        if rows.size < 2 or cols.size < 2:
            continue
        if not mask[:, : width // 2].any() or not mask[:, (width + 1) // 2 :].any():
            continue
        if not mask[: height // 2, :].any() or not mask[(height + 1) // 2 :, :].any():
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        if np.array_equal(mask, np.flipud(mask)):
            continue
        if np.array_equal(mask, np.flipud(np.fliplr(mask))):
            continue
        return mask

    return None


def _layout(pair_count: int, rect_height: int, rect_width: int) -> tuple[int, int, list[tuple[int, int]]] | None:
    ncols_options = [n for n in (2, 3) if n <= pair_count]
    random.shuffle(ncols_options)

    for ncols in ncols_options:
        nrows = (pair_count + ncols - 1) // ncols
        gap_r = random.randint(*GAP_RANGE)
        gap_c = random.randint(*GAP_RANGE)
        jitter_r_max = gap_r - 2
        jitter_c_max = gap_c - 2
        top = random.randint(*MARGIN_RANGE)
        left = random.randint(*MARGIN_RANGE)
        bottom = random.randint(*MARGIN_RANGE)
        right = random.randint(*MARGIN_RANGE)

        rows = top + bottom + nrows * rect_height + (nrows - 1) * gap_r + jitter_r_max
        cols = left + right + ncols * rect_width + (ncols - 1) * gap_c + jitter_c_max
        if rows > GRID_MAX or cols > GRID_MAX:
            continue

        slots = [(slot_r, slot_c) for slot_r in range(nrows) for slot_c in range(ncols)]
        chosen_slots = random.sample(slots, pair_count)
        placements: list[tuple[int, int]] = []
        for slot_r, slot_c in chosen_slots:
            base_r = top + slot_r * (rect_height + gap_r)
            base_c = left + slot_c * (rect_width + gap_c)
            jitter_r = random.randint(0, jitter_r_max)
            jitter_c = random.randint(0, jitter_c_max)
            placements.append((base_r + jitter_r, base_c + jitter_c))
        return rows, cols, placements

    return None


def _paint_pair(grid: np.ndarray, color: int, top: int, left: int, height: int, width: int, orientation: int) -> None:
    if orientation == 0:
        anchors = ((top, left), (top + height - 1, left + width - 1))
    else:
        anchors = ((top, left + width - 1), (top + height - 1, left))
    for r, c in anchors:
        grid[r, c] = int(color)


def _build_instance() -> np.ndarray | None:
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    rect_height = random.randint(*RECT_HEIGHT_RANGE)
    rect_width = random.randint(*RECT_WIDTH_RANGE)

    layout = _layout(pair_count, rect_height, rect_width)
    if layout is None:
        return None
    rows, cols, placements = layout

    pattern = _grow_pattern(rect_height, rect_width)
    if pattern is None:
        return None

    bg = random.choice([c for c in range(10) if c != MAGENTA])
    pair_colors = random.sample([c for c in range(10) if c not in (bg, MAGENTA)], pair_count)

    grid = np.full((rows, cols), int(bg), dtype=int)
    source_index = random.randrange(pair_count)
    source_orientation = random.randint(0, 1)

    target_orientations = [source_orientation, 1 - source_orientation]
    while len(target_orientations) < pair_count - 1:
        target_orientations.append(random.randint(0, 1))
    random.shuffle(target_orientations)

    target_cursor = 0
    for i, ((top, left), color) in enumerate(zip(placements, pair_colors, strict=False)):
        if i == source_index:
            orientation = source_orientation
        else:
            orientation = target_orientations[target_cursor]
            target_cursor += 1

        _paint_pair(grid, color, top, left, rect_height, rect_width, orientation)
        if i == source_index:
            patch = grid[top : top + rect_height, left : left + rect_width]
            patch[pattern] = MAGENTA

    assert _most_common_color(grid) == bg
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_instance()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
