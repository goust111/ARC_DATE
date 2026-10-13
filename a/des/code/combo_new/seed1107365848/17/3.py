from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _frame_bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return None
    expected = 2 * height + 2 * width - 4
    if len(coords) != expected:
        return None
    if any(r not in (r0, r1) and c not in (c0, c1) for r, c in coords):
        return None
    return r0, c0, r1, c1


def _find_frames(grid: np.ndarray, bg: int) -> list[dict[str, int | tuple[int, int, int, int]]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    frames: list[dict[str, int | tuple[int, int, int, int]]] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            bbox = _frame_bbox(coords)
            if bbox is not None:
                frames.append({"color": color, "bbox": bbox})

    return frames


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 + 2 <= ir0 and oc0 + 2 <= ic0 and ir1 <= or1 - 2 and ic1 <= oc1 - 2


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each innermost nested frame cavity using the outermost or innermost frame color according to nesting parity.

    Input structure:
    - The dominant background color fills most of the grid.
    - One or more disjoint groups of axis-aligned hollow rectangular frames appear on the grid.
    - Each frame is one pixel thick, monochromatic, and different layers in the same group use different colors.
    - Frames in a group are strictly nested, leaving background gaps between consecutive layers, and the innermost cavity is background.

    Rule:
    - Detect every rectangular frame and recover nesting chains by rectangle containment.
    - For each leaf chain, count its number of frame layers K.
    - If K is odd, fill the innermost cavity with the outermost frame color.
    - If K is even, fill the innermost cavity with the innermost frame color.
    - Keep all frame pixels and all background gaps between frames unchanged.

    Key insight:
    - The task is not simple region filling: you must first reconstruct nested frame depth, then use the odd/even depth to choose whether the fill color comes from the chain's outside or inside end.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    frames = _find_frames(g, bg)
    out = g.copy()
    if not frames:
        return out

    n = len(frames)
    parents: list[int | None] = [None] * n
    children: list[list[int]] = [[] for _ in range(n)]
    areas = [_bbox_area(frame["bbox"]) for frame in frames]

    for i, frame in enumerate(frames):
        bbox_i = frame["bbox"]
        parent = None
        parent_area = None
        for j, other in enumerate(frames):
            if i == j:
                continue
            bbox_j = other["bbox"]
            if not _contains(bbox_j, bbox_i):
                continue
            area_j = areas[j]
            if parent is None or area_j < parent_area:
                parent = j
                parent_area = area_j
        parents[i] = parent
        if parent is not None:
            children[parent].append(i)

    leaves = [i for i in range(n) if not children[i]]
    for leaf in leaves:
        chain = []
        cur: int | None = leaf
        while cur is not None:
            chain.append(cur)
            cur = parents[cur]
        chain.reverse()

        outermost = frames[chain[0]]
        innermost = frames[chain[-1]]
        fill_color = int(outermost["color"]) if len(chain) % 2 == 1 else int(innermost["color"])
        r0, c0, r1, c1 = innermost["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = fill_color

    return out


MAX_GENERATION_ATTEMPTS = 400
ARC_COLORS = tuple(range(10))
GROUP_COUNT_OPTIONS = (1, 2, 3, 4)
LAYER_OPTIONS = (1, 2, 3, 4)
INNER_SIZE_RANGE = (1, 5)
GAP_RANGE = (1, 2)
ROW_MARGIN_RANGE = (1, 3)
INTER_GROUP_GAP_RANGE = (1, 3)
INTER_ROW_GAP_RANGE = (1, 3)
GRID_EXTRA_SLACK_MAX = 4
SIZE_CAP_BY_GROUP_COUNT = {1: 24, 2: 18, 3: 14, 4: 12}


def _random_composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    values = []
    last = -1
    for cut in cuts + [total + parts - 1]:
        values.append(cut - last - 1)
        last = cut
    return values


def _sample_layer_counts(num_groups: int) -> list[int]:
    if num_groups == 1:
        return [random.choice(LAYER_OPTIONS)]
    counts = [random.choice((1, 3)), random.choice((2, 4))]
    while len(counts) < num_groups:
        counts.append(random.choice(LAYER_OPTIONS))
    random.shuffle(counts)
    return counts


def _sample_group_spec(group_id: int, layer_count: int, bg: int, num_groups: int) -> dict | None:
    palette = [color for color in ARC_COLORS if color != bg]
    size_cap = SIZE_CAP_BY_GROUP_COUNT[num_groups]

    for _ in range(80):
        max_inner = INNER_SIZE_RANGE[1]
        if layer_count >= 3:
            max_inner = min(max_inner, 4)
        if layer_count >= 4 or num_groups >= 3:
            max_inner = min(max_inner, 3)
        inner_h = random.randint(INNER_SIZE_RANGE[0], max_inner)
        inner_w = random.randint(INNER_SIZE_RANGE[0], max_inner)

        gaps: list[tuple[int, int]] = []
        for _layer in range(layer_count - 1):
            gap_hi = GAP_RANGE[1]
            if num_groups == 4 and layer_count >= 3:
                gap_hi = 1
            gaps.append((random.randint(GAP_RANGE[0], gap_hi), random.randint(GAP_RANGE[0], gap_hi)))

        total_margin_h = sum(gap_h + 1 for gap_h, _ in gaps)
        total_margin_w = sum(gap_w + 1 for _, gap_w in gaps)
        outer_h = inner_h + 2 + 2 * total_margin_h
        outer_w = inner_w + 2 + 2 * total_margin_w
        if outer_h > size_cap or outer_w > size_cap:
            continue

        colors = random.sample(palette, layer_count)
        r0, c0, r1, c1 = 0, 0, outer_h - 1, outer_w - 1
        frames = []
        frames.append((r0, c0, r1, c1))
        for gap_h, gap_w in gaps:
            r0 += gap_h + 1
            c0 += gap_w + 1
            r1 -= gap_h + 1
            c1 -= gap_w + 1
            frames.append((r0, c0, r1, c1))

        return {
            "id": group_id,
            "layers": layer_count,
            "colors": colors,
            "frames": frames,
            "height": outer_h,
            "width": outer_w,
        }

    return None


def _choose_rows(specs: list[dict]) -> list[list[dict]]:
    shuffled = specs[:]
    random.shuffle(shuffled)
    n = len(shuffled)
    if n == 1:
        return [shuffled]
    if random.random() < 0.4:
        return [shuffled]
    cut = random.randint(1, n - 1)
    return [shuffled[:cut], shuffled[cut:]]


def _arrange_groups(specs: list[dict]) -> tuple[int, int, dict[int, tuple[int, int]]] | None:
    for _ in range(80):
        rows = _choose_rows(specs)
        row_data = []
        for row in rows:
            left_margin = random.randint(*ROW_MARGIN_RANGE)
            right_margin = random.randint(*ROW_MARGIN_RANGE)
            gaps = [random.randint(*INTER_GROUP_GAP_RANGE) for _ in range(max(0, len(row) - 1))]
            row_height = max(int(spec["height"]) for spec in row)
            row_width = left_margin + right_margin + sum(int(spec["width"]) for spec in row) + sum(gaps)
            row_data.append(
                {
                    "specs": row,
                    "left_margin": left_margin,
                    "right_margin": right_margin,
                    "gaps": gaps,
                    "height": row_height,
                    "width": row_width,
                }
            )

        top_margin = random.randint(*ROW_MARGIN_RANGE)
        bottom_margin = random.randint(*ROW_MARGIN_RANGE)
        row_gaps = [random.randint(*INTER_ROW_GAP_RANGE) for _ in range(max(0, len(row_data) - 1))]
        min_h = top_margin + bottom_margin + sum(data["height"] for data in row_data) + sum(row_gaps)
        min_w = max(data["width"] for data in row_data)
        if min_h > 30 or min_w > 30:
            continue

        grid_h = min_h + random.randint(0, min(GRID_EXTRA_SLACK_MAX, 30 - min_h))
        grid_w = min_w + random.randint(0, min(GRID_EXTRA_SLACK_MAX, 30 - min_w))

        placements: dict[int, tuple[int, int]] = {}
        extra_vertical = _random_composition(grid_h - min_h, len(row_data) + 1)
        y = top_margin + extra_vertical[0]
        for row_index, data in enumerate(row_data):
            row_width = int(data["width"])
            row_left = random.randint(0, grid_w - row_width)
            x = row_left + int(data["left_margin"])
            for spec_index, spec in enumerate(data["specs"]):
                band_height = int(data["height"])
                top = y + random.randint(0, band_height - int(spec["height"]))
                placements[int(spec["id"])] = (top, x)
                x += int(spec["width"])
                if spec_index < len(data["gaps"]):
                    x += int(data["gaps"][spec_index])
            y += int(data["height"])
            if row_index < len(row_gaps):
                y += int(row_gaps[row_index]) + int(extra_vertical[row_index + 1])

        if len(placements) == len(specs):
            return grid_h, grid_w, placements

    return None


def _draw_frame(grid: np.ndarray, top: int, left: int, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[top + r0, left + c0 : left + c1 + 1] = color
    grid[top + r1, left + c0 : left + c1 + 1] = color
    grid[top + r0 : top + r1 + 1, left + c0] = color
    grid[top + r0 : top + r1 + 1, left + c1] = color


def _build_pair() -> tuple[np.ndarray, np.ndarray] | None:
    num_groups = random.choice(GROUP_COUNT_OPTIONS)
    bg = random.choice(ARC_COLORS)
    layer_counts = _sample_layer_counts(num_groups)

    specs = []
    for group_id, layer_count in enumerate(layer_counts):
        spec = _sample_group_spec(group_id, layer_count, bg, num_groups)
        if spec is None:
            return None
        specs.append(spec)

    arranged = _arrange_groups(specs)
    if arranged is None:
        return None
    grid_h, grid_w, placements = arranged

    inp = np.full((grid_h, grid_w), bg, dtype=int)
    out = inp.copy()
    for spec in specs:
        top, left = placements[int(spec["id"])]
        for bbox, color in zip(spec["frames"], spec["colors"]):
            _draw_frame(inp, top, left, bbox, int(color))
            _draw_frame(out, top, left, bbox, int(color))
        inner_r0, inner_c0, inner_r1, inner_c1 = spec["frames"][-1]
        fill_color = int(spec["colors"][0]) if int(spec["layers"]) % 2 == 1 else int(spec["colors"][-1])
        out[top + inner_r0 + 1 : top + inner_r1, left + inner_c0 + 1 : left + inner_c1] = fill_color

    counts = Counter(map(int, inp.reshape(-1)))
    bg_count = counts[bg]
    if any(color != bg and count >= bg_count for color, count in counts.items()):
        return None
    if np.array_equal(inp, out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        pair = _build_pair()
        if pair is None:
            continue
        inp, out = pair
        try:
            if np.array_equal(main(inp), out):
                return inp
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid nested-frame puzzle")
