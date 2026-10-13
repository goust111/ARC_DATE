from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_ATTEMPTS = 400
GEN_HEIGHT_RANGE = (14, 22)
GEN_WIDTH_RANGE = (14, 22)
GEN_PAIR_COUNT_RANGE = (2, 4)
GEN_FRAGMENT_SIZE_RANGE = (2, 5)
GEN_GAP_RANGE = (2, 6)
GEN_MARGIN = 1
GEN_PLACEMENT_ATTEMPTS = 160
GEN_FILL_COLOR = 9
GEN_PALETTE = tuple(range(10))


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _find_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == int(bg):
                continue
            color = int(g[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            r0 = r1 = r
            c0 = c1 = c
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                r0 = min(r0, cr)
                r1 = max(r1, cr)
                c0 = min(c0, cc)
                c1 = max(c1, cc)
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            comps.append(
                {
                    "color": color,
                    "coords": np.array(coords, dtype=int),
                    "bbox": (r0, c0, r1, c1),
                }
            )
    return comps


def _bbox_gap(a0: int, a1: int, b0: int, b1: int) -> int:
    if a1 < b0:
        return int(b0 - a1 - 1)
    if b1 < a0:
        return int(a0 - b1 - 1)
    return 0


def _center_distance(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> tuple[float, float]:
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b
    return abs((ar0 + ar1) / 2.0 - (br0 + br1) / 2.0), abs((ac0 + ac1) / 2.0 - (bc0 + bc1) / 2.0)


def _pair_axis(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> str:
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b
    horizontal_gap = _bbox_gap(ac0, ac1, bc0, bc1)
    vertical_gap = _bbox_gap(ar0, ar1, br0, br1)
    if horizontal_gap != vertical_gap:
        return "h" if horizontal_gap > vertical_gap else "v"
    dr, dc = _center_distance(box_a, box_b)
    return "h" if dc >= dr else "v"


def _claim(claims: dict[tuple[int, int], int], coord: tuple[int, int], color: int) -> None:
    prev = claims.get(coord)
    assert prev is None, f"overlapping output claims at {coord}"
    claims[coord] = int(color)


def _find_corner_marker(grid: np.ndarray, bg: int) -> tuple[tuple[int, int], int]:
    h, w = grid.shape
    corners = ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1))
    hits = [(r, c) for r, c in corners if int(grid[r, c]) != int(bg)]
    assert len(hits) == 1, "expected a unique non-background corner marker"
    marker = hits[0]
    return marker, int(grid[marker])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-colored fragments, slide each pair together, and fill the merged bbox with the corner-marker color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _most_common_color(g)
    marker_coord, fill_color = _find_corner_marker(g, bg)

    work = g.copy()
    work[marker_coord] = bg
    components = _find_components(work, bg)
    assert components, "expected at least one fragment pair"

    by_color: dict[int, list[dict[str, object]]] = defaultdict(list)
    for comp in components:
        by_color[int(comp["color"])].append(comp)

    out_claims: dict[tuple[int, int], int] = {}
    for color, pair in by_color.items():
        assert len(pair) == 2, f"expected exactly two fragments for color {color}"
        comp_a, comp_b = pair
        box_a = tuple(int(v) for v in comp_a["bbox"])
        box_b = tuple(int(v) for v in comp_b["bbox"])
        axis = _pair_axis(box_a, box_b)

        if axis == "h":
            first, second = sorted((comp_a, comp_b), key=lambda comp: int(comp["bbox"][1]))
            fb1 = int(first["bbox"][3])
            sa0 = int(second["bbox"][1])
            gap = int(sa0 - fb1 - 1)
            shift_first = (0, gap // 2)
            shift_second = (0, -(gap - gap // 2))
        else:
            first, second = sorted((comp_a, comp_b), key=lambda comp: int(comp["bbox"][0]))
            fb1 = int(first["bbox"][2])
            sa0 = int(second["bbox"][0])
            gap = int(sa0 - fb1 - 1)
            shift_first = (gap // 2, 0)
            shift_second = (-(gap - gap // 2), 0)

        assert gap >= 0, "paired fragments must be separated or already adjacent"

        moved_cells: set[tuple[int, int]] = set()
        moved_bbox: list[int] | None = None
        for comp, (dr, dc) in ((first, shift_first), (second, shift_second)):
            for r_raw, c_raw in comp["coords"]:
                r = int(r_raw) + int(dr)
                c = int(c_raw) + int(dc)
                moved_cells.add((r, c))
                if moved_bbox is None:
                    moved_bbox = [r, c, r, c]
                else:
                    moved_bbox[0] = min(moved_bbox[0], r)
                    moved_bbox[1] = min(moved_bbox[1], c)
                    moved_bbox[2] = max(moved_bbox[2], r)
                    moved_bbox[3] = max(moved_bbox[3], c)
                _claim(out_claims, (r, c), color)

        assert moved_bbox is not None
        r0, c0, r1, c1 = moved_bbox
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if (r, c) not in moved_cells:
                    _claim(out_claims, (r, c), fill_color)

    out = np.full_like(g, bg)
    for (r, c), color in out_claims.items():
        out[r, c] = int(color)
    return out


def _expand_rect(rect: tuple[int, int, int, int], margin: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = rect
    return (
        max(0, r0 - margin),
        max(0, c0 - margin),
        min(h - 1, r1 + margin),
        min(w - 1, c1 + margin),
    )


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _random_component_mask(height: int, width: int) -> np.ndarray:
    area = height * width
    pivot = (random.randint(0, height - 1), random.randint(0, width - 1))
    cells = {(0, 0)}
    current = (0, 0)

    for target in (pivot, (height - 1, width - 1)):
        while current != target:
            r, c = current
            options = []
            if r < target[0]:
                options.append((r + 1, c))
            if c < target[1]:
                options.append((r, c + 1))
            current = random.choice(options)
            cells.add(current)

    min_target = max(len(cells), (area + 1) // 2)
    max_target = max(min_target, area - 1)
    target = random.randint(min_target, max_target)
    while len(cells) < target:
        frontier = []
        for r, c in cells:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(random.choice(frontier))

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    if random.random() < 0.5:
        mask = np.flip(mask, axis=0)
    if random.random() < 0.5:
        mask = np.flip(mask, axis=1)
    return mask


def _make_pair_record(color: int, orientation: str) -> dict[str, object] | None:
    h1 = random.randint(*GEN_FRAGMENT_SIZE_RANGE)
    w1 = random.randint(*GEN_FRAGMENT_SIZE_RANGE)
    h2 = random.randint(*GEN_FRAGMENT_SIZE_RANGE)
    w2 = random.randint(*GEN_FRAGMENT_SIZE_RANGE)
    gap = random.randint(*GEN_GAP_RANGE)
    move_first = gap // 2

    mask1 = _random_component_mask(h1, w1)
    mask2 = _random_component_mask(h2, w2)

    if orientation == "h":
        overlap = random.randint(1, min(h1, h2))
        top1 = 0
        top2 = h1 - overlap
        env_h = h1 + h2 - overlap
        env_w = w1 + gap + w2

        input_cells = {(top1 + r, c) for r, c in np.argwhere(mask1)}
        input_cells |= {(top2 + r, w1 + gap + c) for r, c in np.argwhere(mask2)}

        output_cells = {(top1 + r, move_first + c) for r, c in np.argwhere(mask1)}
        output_cells |= {(top2 + r, move_first + w1 + c) for r, c in np.argwhere(mask2)}

        output_bbox = (0, move_first, env_h - 1, move_first + w1 + w2 - 1)
    else:
        overlap = random.randint(1, min(w1, w2))
        left1 = 0
        left2 = w1 - overlap
        env_h = h1 + gap + h2
        env_w = w1 + w2 - overlap

        input_cells = {(r, left1 + c) for r, c in np.argwhere(mask1)}
        input_cells |= {(h1 + gap + r, left2 + c) for r, c in np.argwhere(mask2)}

        output_cells = {(move_first + r, left1 + c) for r, c in np.argwhere(mask1)}
        output_cells |= {(move_first + h1 + r, left2 + c) for r, c in np.argwhere(mask2)}

        output_bbox = (move_first, 0, move_first + h1 + h2 - 1, env_w - 1)

    r0, c0, r1, c1 = output_bbox
    fill_cells = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if (r, c) not in output_cells
    }
    if not fill_cells:
        return None

    return {
        "color": int(color),
        "input_shape": (env_h, env_w),
        "input_cells": input_cells,
        "output_cells": output_cells,
        "fill_cells": fill_cells,
        "output_bbox": output_bbox,
    }


def _shift_cells(cells: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in cells}


def _translate_rect(rect: tuple[int, int, int, int], dr: int, dc: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = rect
    return (r0 + dr, c0 + dc, r1 + dr, c1 + dc)


def _balanced_orientations(count: int) -> list[str]:
    horizontal_count = count // 2
    if count % 2 == 1 and random.random() < 0.5:
        horizontal_count += 1
    orientations = ["h"] * horizontal_count + ["v"] * (count - horizontal_count)
    random.shuffle(orientations)
    return orientations


def _place_pairs(
    height: int,
    width: int,
    colors: list[int],
) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray] | None:
    records: list[dict[str, object]] = []
    input_reserved: list[tuple[int, int, int, int]] = []
    output_reserved: list[tuple[int, int, int, int]] = []
    inp = np.full((height, width), -1, dtype=int)
    out = np.full((height, width), -1, dtype=int)

    for color, orientation in zip(colors, _balanced_orientations(len(colors))):
        placed = False
        for _ in range(GEN_PLACEMENT_ATTEMPTS):
            record = _make_pair_record(int(color), orientation)
            if record is None:
                continue

            env_h, env_w = record["input_shape"]
            if env_h + 2 * GEN_MARGIN > height or env_w + 2 * GEN_MARGIN > width:
                continue

            r_start = random.randint(GEN_MARGIN, height - env_h - GEN_MARGIN)
            c_start = random.randint(GEN_MARGIN, width - env_w - GEN_MARGIN)

            input_rect = (r_start, c_start, r_start + env_h - 1, c_start + env_w - 1)
            output_rect = _translate_rect(tuple(record["output_bbox"]), r_start, c_start)
            input_guard = _expand_rect(input_rect, 1, (height, width))
            output_guard = _expand_rect(output_rect, 1, (height, width))
            if any(_rects_overlap(input_guard, rect) for rect in input_reserved):
                continue
            if any(_rects_overlap(output_guard, rect) for rect in output_reserved):
                continue

            shifted_input = _shift_cells(record["input_cells"], r_start, c_start)
            shifted_output = _shift_cells(record["output_cells"], r_start, c_start)
            shifted_fill = _shift_cells(record["fill_cells"], r_start, c_start)

            if any(inp[r, c] != -1 for r, c in shifted_input):
                continue
            if any(out[r, c] != -1 for r, c in shifted_output | shifted_fill):
                continue

            for r, c in shifted_input:
                inp[r, c] = int(color)
            for r, c in shifted_fill:
                out[r, c] = GEN_FILL_COLOR
            for r, c in shifted_output:
                out[r, c] = int(color)

            records.append(
                {
                    "color": int(color),
                    "input_cells": shifted_input,
                    "output_cells": shifted_output,
                    "fill_cells": shifted_fill,
                }
            )
            input_reserved.append(input_guard)
            output_reserved.append(output_guard)
            placed = True
            break

        if not placed:
            return None

    return records, inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        height = random.randint(*GEN_HEIGHT_RANGE)
        width = random.randint(*GEN_WIDTH_RANGE)
        pair_count = random.randint(*GEN_PAIR_COUNT_RANGE)

        bg_choices = [color for color in GEN_PALETTE if color != GEN_FILL_COLOR]
        bg = random.choice(bg_choices)
        pair_colors = [color for color in GEN_PALETTE if color not in (bg, GEN_FILL_COLOR)]
        random.shuffle(pair_colors)
        pair_colors = pair_colors[:pair_count]

        placed = _place_pairs(height, width, pair_colors)
        if placed is None:
            continue
        _records, raw_inp, raw_out = placed

        inp = np.full((height, width), bg, dtype=int)
        out = np.full((height, width), bg, dtype=int)
        inp[raw_inp != -1] = raw_inp[raw_inp != -1]
        out[raw_out != -1] = raw_out[raw_out != -1]

        corners = ((0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1))
        marker = random.choice(corners)
        if int(inp[marker]) != bg:
            continue
        inp[marker] = GEN_FILL_COLOR

        values, counts = np.unique(inp, return_counts=True)
        if int(values[int(np.argmax(counts))]) != int(bg):
            continue
        if len(set(np.unique(inp))) < 4:
            continue
        if not np.array_equal(main(inp), out):
            continue
        return inp

    raise RuntimeError("failed to generate a valid input")
