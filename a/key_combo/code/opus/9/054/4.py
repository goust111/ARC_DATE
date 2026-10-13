from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
RED = 2
YELLOW = 4
GREY = 5

INPUT_FILL_COLORS = [1, 3, 6, 7, 8, 9]
INPUT_MARKER_COLORS = [1, 3, 6, 7, 8, 9]

GRID_RANGES = {
    2: ((18, 22), (18, 24)),
    3: ((19, 24), (24, 29)),
    4: ((20, 26), (29, 30)),
}
GEN_RETRY_LIMIT = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _label_mask(mask: np.ndarray) -> tuple[np.ndarray, int]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    next_id = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            next_id += 1
            queue = deque([(r, c)])
            labels[r, c] = next_id
            while queue:
                rr, cc = queue.popleft()
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = next_id
                        queue.append((nr, nc))
    return labels, next_id


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    labels, count = _label_mask(mask)
    comps: list[list[tuple[int, int]]] = []
    for comp_id in range(1, count + 1):
        coords = np.argwhere(labels == comp_id)
        comps.append([(int(r), int(c)) for r, c in coords])
    return comps


def _x_mask(height: int, width: int) -> np.ndarray:
    assert height > 0 and width > 0
    if height > width:
        return _x_mask(width, height).T

    mask = np.zeros((height, width), dtype=bool)
    center_rows = {(height - 1) // 2, height // 2}
    for r in range(height):
        d = min(r, height - 1 - r)
        left = d
        right = width - 1 - d
        if r in center_rows:
            mask[r, left : right + 1] = True
        else:
            mask[r, left] = True
            mask[r, right] = True
    return mask


def _extract_rectangles(grid: np.ndarray) -> list[dict]:
    fg_mask = (grid != BLACK) & (grid != RED)
    rectangles = []
    for coords in _components(fg_mask):
        r0, c0, r1, c1 = _bbox(coords)
        block = grid[r0 : r1 + 1, c0 : c1 + 1]
        height, width = block.shape
        assert height >= 3 and width >= 3, "rectangle needs a strict interior"
        assert len(coords) == height * width, "foreground component must fill its bbox"
        assert np.all(block != BLACK) and np.all(block != RED)

        border = np.concatenate((block[0, :], block[-1, :], block[1:-1, 0], block[1:-1, -1]))
        border_colors = np.unique(border)
        assert border_colors.size == 1, "rectangle border must be monochrome"
        fill_color = int(border_colors[0])

        colors = [int(v) for v in np.unique(block)]
        assert len(colors) == 2, "rectangle must contain one fill color and one marker color"
        marker_color = next(color for color in colors if color != fill_color)
        marker_coords = np.argwhere(block == marker_color)
        assert marker_coords.shape[0] == 1, "marker color must appear exactly once"
        mr, mc = map(int, marker_coords[0])
        assert 0 < mr < height - 1 and 0 < mc < width - 1, "marker must be strictly inside"

        rectangles.append(
            {
                "bbox": (r0, c0, r1, c1),
                "fill_color": fill_color,
                "marker_color": marker_color,
                "marker_pos": (r0 + mr, c0 + mc),
            }
        )
    assert rectangles, "expected at least one marked rectangle"
    return rectangles


def _region_parity(base: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    assert base.ndim == 2
    h, w = base.shape
    assert int(base[h - 1, 0]) == BLACK, "lower-left anchor cell must be background"

    region_labels, region_count = _label_mask(base == BLACK)
    assert region_count >= 2, "walls must split the background into multiple regions"
    start_region = int(region_labels[h - 1, 0])
    assert start_region > 0

    graph = {region_id: set() for region_id in range(1, region_count + 1)}
    for r, c in np.argwhere(base == RED):
        neighbors = {
            int(region_labels[nr, nc])
            for nr, nc in _neighbors4(int(r), int(c))
            if 0 <= nr < h and 0 <= nc < w and region_labels[nr, nc] > 0
        }
        neighbors = {region_id for region_id in neighbors if region_id > 0}
        neighbor_list = sorted(neighbors)
        for i, a in enumerate(neighbor_list):
            for b in neighbor_list[i + 1 :]:
                graph[a].add(b)
                graph[b].add(a)

    parity = {start_region: 0}
    queue = deque([start_region])
    while queue:
        region = queue.popleft()
        for neighbor in graph[region]:
            expected = 1 - parity[region]
            if neighbor in parity:
                assert parity[neighbor] == expected, "region graph must be bipartite"
            else:
                parity[neighbor] = expected
                queue.append(neighbor)

    assert set(parity) == set(graph), "all background regions must be connected through wall adjacencies"
    return region_labels, parity


def _render_output_from_rectangles(base: np.ndarray, rectangles: list[dict], region_labels: np.ndarray, parity: dict[int, int]) -> np.ndarray:
    out = base.copy()
    for rect in rectangles:
        r0, c0, r1, c1 = rect["bbox"]
        region_ids = set(np.unique(region_labels[r0 : r1 + 1, c0 : c1 + 1]).tolist())
        assert region_ids and 0 not in region_ids and len(region_ids) == 1, "rectangle must lie inside one background region"
        region_id = next(iter(region_ids))
        frame_color = GREY if parity[region_id] == 0 else YELLOW
        marker_color = int(rect["marker_color"])

        out[r0, c0 : c1 + 1] = frame_color
        out[r1, c0 : c1 + 1] = frame_color
        out[r0 : r1 + 1, c0] = frame_color
        out[r0 : r1 + 1, c1] = frame_color

        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[_x_mask(r1 - r0 - 1, c1 - c0 - 1)] = marker_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reframe each wall-contained solid rectangle with the color of its bipartite wall region and draw an X in the marker color.

    Input structure:
    - A black background is partitioned by one-cell-thick red wall segments into 4-connected background regions.
    - Some regions contain exactly one solid non-red rectangle, and each such rectangle has one interior marker pixel of a different color.
    - The lower-left background region contains no rectangle and anchors the region-class assignment.

    Rule:
    - Keep every red wall pixel unchanged and erase every non-wall rectangle back to black before redrawing.
    - Label the black regions separated by the walls, build wall-side adjacencies, and 2-color the region graph starting from the lower-left region as class A.
    - Class A regions give their rectangles grey frames, class B regions give their rectangles yellow frames; inside each reframed rectangle, draw the canonical X using the rectangle's marker color.

    Key insight:
    - The rectangle's output frame color is not local to the rectangle at all; it is determined by the global wall-topology parity of the background region that contains it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rectangles = _extract_rectangles(g)
    base = g.copy()
    base[(base != BLACK) & (base != RED)] = BLACK
    region_labels, parity = _region_parity(base)
    return _render_output_from_rectangles(base, rectangles, region_labels, parity)


def _build_wall_path(height: int, anchor: int, rng: random.Random) -> list[tuple[int, int]]:
    if height < 9:
        raise ValueError("height too small for wall generation")
    segment_count = rng.choices([1, 2, 3], weights=[2, 4, 3], k=1)[0]
    split_rows = sorted(rng.sample(range(3, height - 3), segment_count - 1))
    segment_ends = split_rows + [height - 1]

    allowed = [anchor]
    if anchor > 1:
        allowed.append(anchor - 1)
    allowed.append(anchor + 1)
    allowed = sorted(set(allowed))

    columns = [rng.choice(allowed)]
    while len(columns) < segment_count:
        options = [col for col in allowed if abs(col - columns[-1]) == 1]
        columns.append(rng.choice(options))

    path: list[tuple[int, int]] = [(0, columns[0])]
    row = 0
    col = columns[0]
    for seg_index, row_end in enumerate(segment_ends):
        while row < row_end:
            row += 1
            path.append((row, col))
        if seg_index + 1 < segment_count:
            next_col = columns[seg_index + 1]
            step = 1 if next_col > col else -1
            while col != next_col:
                col += step
                path.append((row, col))
    return path


def _build_wall_grid(height: int, width: int, n_rectangles: int, rng: random.Random) -> np.ndarray | None:
    n_walls = n_rectangles
    region_widths = [rng.randint(5, 6) for _ in range(n_rectangles + 1)]
    min_width = sum(region_widths) + n_walls
    lo_w, hi_w = GRID_RANGES[n_rectangles][1]
    if min_width > hi_w:
        return None
    width = max(width, min_width)
    extra = width - min_width
    for _ in range(extra):
        region_widths[rng.randrange(len(region_widths))] += 1

    anchors = []
    cursor = region_widths[0]
    for wall_index in range(n_walls):
        anchors.append(cursor)
        if wall_index + 1 < len(region_widths):
            cursor += 1 + region_widths[wall_index + 1]

    grid = np.zeros((height, width), dtype=int)
    for anchor in anchors:
        path = _build_wall_path(height, anchor, rng)
        for r, c in path:
            if not (0 <= r < height and 0 <= c < width):
                return None
            if grid[r, c] == RED:
                return None
            grid[r, c] = RED

    base = grid.copy()
    labels, parity = _region_parity(base)
    region_count = int(labels.max())
    if region_count != n_rectangles + 1:
        return None
    degrees = {region_id: 0 for region_id in range(1, region_count + 1)}
    for r, c in np.argwhere(base == RED):
        regions = sorted(
            {
                int(labels[nr, nc])
                for nr, nc in _neighbors4(int(r), int(c))
                if 0 <= nr < height and 0 <= nc < width and labels[nr, nc] > 0
            }
        )
        for i, a in enumerate(regions):
            for b in regions[i + 1 :]:
                degrees[a] += 1
                degrees[b] += 1
    if any(degree == 0 for degree in degrees.values()):
        return None
    if parity[int(labels[height - 1, 0])] != 0:
        return None
    return grid


def _rectangle_candidates(region_labels: np.ndarray, region_id: int) -> list[tuple[int, int, int, int]]:
    h, w = region_labels.shape
    candidates: list[tuple[int, int, int, int]] = []
    for rect_h in range(4, min(h, 9)):
        for rect_w in range(4, min(w, 8)):
            for r0 in range(0, h - rect_h + 1):
                r1 = r0 + rect_h - 1
                for c0 in range(0, w - rect_w + 1):
                    c1 = c0 + rect_w - 1
                    block = region_labels[r0 : r1 + 1, c0 : c1 + 1]
                    if np.all(block == region_id):
                        candidates.append((r0, c0, r1, c1))
    candidates.sort(key=lambda box: ((box[2] - box[0] + 1) * (box[3] - box[1] + 1), box[0], box[1], box[3]))
    return candidates


def _choose_rectangles(base: np.ndarray, rng: random.Random) -> list[dict] | None:
    region_labels, parity = _region_parity(base)
    start_region = int(region_labels[base.shape[0] - 1, 0])
    region_ids = sorted(int(v) for v in np.unique(region_labels) if int(v) > 0)
    occupied_region_ids = [region_id for region_id in region_ids if region_id != start_region]
    candidate_map = {region_id: _rectangle_candidates(region_labels, region_id) for region_id in occupied_region_ids}
    if any(not boxes for boxes in candidate_map.values()):
        return None

    rectangles = []
    parity_signatures = set()
    for region_id in occupied_region_ids:
        candidates = candidate_map[region_id][:]
        rng.shuffle(candidates)
        preferred = [
            box
            for box in candidates
            if (box[2] - box[0] - 1) % 2 == 0 and (box[3] - box[1] - 1) % 2 == 0
        ]
        if len(rectangles) == 0 and preferred:
            chosen = rng.choice(preferred)
        else:
            chosen = candidates[0]
        parity_signatures.add(((chosen[2] - chosen[0] - 1) % 2, (chosen[3] - chosen[1] - 1) % 2))
        rectangles.append({"bbox": chosen, "region_id": region_id, "region_parity": parity[region_id]})

    if not any(sig == (0, 0) for sig in parity_signatures):
        return None
    if len(parity_signatures) < 2:
        return None
    if {rect["region_parity"] for rect in rectangles} != {0, 1}:
        return None
    return rectangles


def _assign_input_colors(rectangles: list[dict], rng: random.Random) -> None:
    by_parity = {
        0: [rect for rect in rectangles if rect["region_parity"] == 0],
        1: [rect for rect in rectangles if rect["region_parity"] == 1],
    }
    duplicate_fill = rng.choice(INPUT_FILL_COLORS)
    a_rect = rng.choice(by_parity[0])
    b_rect = rng.choice(by_parity[1])
    a_rect["fill_color"] = duplicate_fill
    b_rect["fill_color"] = duplicate_fill

    remaining_fills = [color for color in INPUT_FILL_COLORS if color != duplicate_fill]
    rng.shuffle(remaining_fills)
    for rect in rectangles:
        if "fill_color" not in rect:
            rect["fill_color"] = remaining_fills.pop() if remaining_fills else rng.choice(INPUT_FILL_COLORS)

    marker_choices = INPUT_MARKER_COLORS[:]
    rng.shuffle(marker_choices)
    for rect in rectangles:
        valid = [color for color in INPUT_MARKER_COLORS if color != rect["fill_color"]]
        rect["marker_color"] = marker_choices.pop() if marker_choices and marker_choices[-1] in valid else rng.choice(valid)


def _place_markers(rectangles: list[dict], rng: random.Random) -> None:
    for rect in rectangles:
        r0, c0, r1, c1 = rect["bbox"]
        marker_r = rng.randint(r0 + 1, r1 - 1)
        marker_c = rng.randint(c0 + 1, c1 - 1)
        rect["marker_pos"] = (marker_r, marker_c)


def _render_input(base: np.ndarray, rectangles: list[dict]) -> np.ndarray:
    grid = base.copy()
    for rect in rectangles:
        r0, c0, r1, c1 = rect["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(rect["fill_color"])
        mr, mc = rect["marker_pos"]
        grid[mr, mc] = int(rect["marker_color"])
    return grid


def _wrong_fill_frame_output(base: np.ndarray, rectangles: list[dict]) -> np.ndarray:
    out = base.copy()
    for rect in rectangles:
        r0, c0, r1, c1 = rect["bbox"]
        out[r0, c0 : c1 + 1] = int(rect["fill_color"])
        out[r1, c0 : c1 + 1] = int(rect["fill_color"])
        out[r0 : r1 + 1, c0] = int(rect["fill_color"])
        out[r0 : r1 + 1, c1] = int(rect["fill_color"])
        out[r0 + 1 : r1, c0 + 1 : c1][_x_mask(r1 - r0 - 1, c1 - c0 - 1)] = int(rect["marker_color"])
    return out


def _semantic_checks(inp: np.ndarray, rectangles: list[dict]) -> bool:
    try:
        extracted = _extract_rectangles(inp)
    except AssertionError:
        return False
    if len(extracted) != len(rectangles):
        return False

    base = inp.copy()
    base[(base != BLACK) & (base != RED)] = BLACK
    try:
        region_labels, parity = _region_parity(base)
    except AssertionError:
        return False

    start_region = int(region_labels[inp.shape[0] - 1, 0])
    if any(int(region_labels[rect["marker_pos"]]) == start_region for rect in rectangles):
        return False
    if {parity[int(region_labels[rect["marker_pos"]])] for rect in rectangles} != {0, 1}:
        return False

    expected = _render_output_from_rectangles(base, rectangles, region_labels, parity)
    wrong_fill = _wrong_fill_frame_output(base, rectangles)
    if np.array_equal(expected, wrong_fill):
        return False
    if GREY not in expected or YELLOW not in expected:
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    rng = random.Random()
    n_rectangles = rng.choice([2, 3, 4])
    (lo_h, hi_h), (lo_w, hi_w) = GRID_RANGES[n_rectangles]
    height = rng.randint(lo_h, hi_h)
    width = rng.randint(lo_w, hi_w)

    base = _build_wall_grid(height, width, n_rectangles, rng)
    if base is None:
        return None

    rectangles = _choose_rectangles(base, rng)
    if rectangles is None:
        return None
    _assign_input_colors(rectangles, rng)
    _place_markers(rectangles, rng)

    inp = _render_input(base, rectangles)
    if not _semantic_checks(inp, rectangles):
        return None

    solved = main(inp)
    base_again = inp.copy()
    base_again[(base_again != BLACK) & (base_again != RED)] = BLACK
    region_labels, parity = _region_parity(base_again)
    expected = _render_output_from_rectangles(base_again, rectangles, region_labels, parity)
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid wall-region rectangle puzzle")
