import random
from typing import Dict, List, Sequence, Tuple

import numpy as np


BG = 0
FRAME = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 400
LEGEND_GAP = 1
AREA_GAP = 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey-framed region by tiling the legend template keyed by its interior marker color.

    Input structure:
    - The background is black, and several grey rectangular frames enclose interior regions.
    - Each frame interior contains mostly background plus one or two colored marker pixels.
    - A legend sits either to the right or below the framed regions; each legend block has an index pixel at its top-left corner and a 2x2 or 3x3 template in its lower-right sub-square.

    Rule:
    - Detect every framed rectangle and read the non-background marker colors inside it, ignoring black background and the grey border.
    - Parse the legend into a color-to-template mapping by using each block's top-left pixel as the key and its lower-right template square as the value.
    - If a region has one marker color, use that template directly; if it has two, sort the markers from left to right and overlay the second template onto the first wherever the second template is non-black.
    - Tile the resulting template from the interior's top-left corner to fill the entire framed interior without changing the grey borders or the legend.

    Key insight:
    - The interior markers are lookup keys rather than fill colors, and two markers mean "compose templates in spatial order, then tile the composite."
    """
    g = np.asarray(grid, dtype=int)
    frames = _find_frames(g)
    assert frames, "expected at least one grey frame"

    legend = _parse_legend(g, frames)
    assert legend, "expected a non-empty legend"

    out = g.copy()
    for r0, c0, r1, c1 in frames:
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        markers = [
            (int(rr), int(cc), int(inner[rr, cc]))
            for rr, cc in np.argwhere((inner != BG) & (inner != FRAME))
        ]
        if not markers:
            continue

        markers.sort(key=lambda item: (item[1], item[0]))
        templates = [legend[color] for _, _, color in markers if color in legend]
        assert templates, "every marker color should have a legend entry"

        composite = templates[0].copy()
        for template in templates[1:]:
            composite = _overlay_template(composite, template)

        filled = _tile_template(composite, inner.shape)
        out[r0 + 1 : r1, c0 + 1 : c1] = filled

    return out


def _find_frames(grid: np.ndarray) -> List[Tuple[int, int, int, int]]:
    h, w = grid.shape
    frames: List[Tuple[int, int, int, int]] = []
    for r in range(h - 2):
        for c in range(w - 2):
            if int(grid[r, c]) != FRAME:
                continue
            if r > 0 and int(grid[r - 1, c]) == FRAME:
                continue
            if c > 0 and int(grid[r, c - 1]) == FRAME:
                continue
            if int(grid[r + 1, c]) != FRAME or int(grid[r, c + 1]) != FRAME:
                continue

            c1 = c
            while c1 + 1 < w and int(grid[r, c1 + 1]) == FRAME:
                c1 += 1
            r1 = r
            while r1 + 1 < h and int(grid[r1 + 1, c]) == FRAME:
                r1 += 1

            if r1 - r < 2 or c1 - c < 2:
                continue
            if not np.all(grid[r, c : c1 + 1] == FRAME):
                continue
            if not np.all(grid[r1, c : c1 + 1] == FRAME):
                continue
            if not np.all(grid[r : r1 + 1, c] == FRAME):
                continue
            if not np.all(grid[r : r1 + 1, c1] == FRAME):
                continue
            if np.any(grid[r + 1 : r1, c + 1 : c1] == FRAME):
                continue

            frames.append((r, c, r1, c1))

    frames.sort()
    return frames


def _point_inside_any_frame(point: Tuple[int, int], frames: Sequence[Tuple[int, int, int, int]]) -> bool:
    r, c = point
    for r0, c0, r1, c1 in frames:
        if r0 <= r <= r1 and c0 <= c <= c1:
            return True
    return False


def _parse_legend(grid: np.ndarray, frames: Sequence[Tuple[int, int, int, int]]) -> Dict[int, np.ndarray]:
    legend_points = [
        (int(r), int(c))
        for r, c in np.argwhere(grid != BG)
        if int(grid[r, c]) != FRAME and not _point_inside_any_frame((int(r), int(c)), frames)
    ]
    assert legend_points, "legend pixels must exist outside frames"

    rows = [r for r, _ in legend_points]
    cols = [c for _, c in legend_points]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    bbox_h = r1 - r0 + 1
    bbox_w = c1 - c0 + 1

    candidates = []
    for block_size in (3, 4):
        if bbox_w == block_size and (bbox_h + LEGEND_GAP) % (block_size + LEGEND_GAP) == 0:
            count = (bbox_h + LEGEND_GAP) // (block_size + LEGEND_GAP)
            if count >= 2:
                candidates.append(("right", block_size, count))
        if bbox_h == block_size and (bbox_w + LEGEND_GAP) % (block_size + LEGEND_GAP) == 0:
            count = (bbox_w + LEGEND_GAP) // (block_size + LEGEND_GAP)
            if count >= 2:
                candidates.append(("bottom", block_size, count))

    assert len(candidates) == 1, f"legend layout should be uniquely identifiable, got {candidates}"
    orientation, block_size, count = candidates[0]

    mapping: Dict[int, np.ndarray] = {}
    for idx in range(count):
        br = r0 + idx * (block_size + LEGEND_GAP) if orientation == "right" else r0
        bc = c0 if orientation == "right" else c0 + idx * (block_size + LEGEND_GAP)
        key = int(grid[br, bc])
        template = np.array(grid[br + 1 : br + block_size, bc + 1 : bc + block_size], dtype=int)
        assert key != BG and key != FRAME
        mapping[key] = template

    return mapping


def _overlay_template(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    h = max(first.shape[0], second.shape[0])
    w = max(first.shape[1], second.shape[1])
    out = np.full((h, w), BG, dtype=int)
    out[: first.shape[0], : first.shape[1]] = first
    second_patch = out[: second.shape[0], : second.shape[1]]
    mask = second != BG
    second_patch[mask] = second[mask]
    return out


def _tile_template(template: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    th, tw = template.shape
    out = np.empty((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            out[r, c] = int(template[r % th, c % tw])
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    tile_size = rng.choice([2, 3])
    block_size = tile_size + 1
    orientation = rng.choice(["right", "bottom"])
    region_count = rng.randint(2, 4)
    marker_counts = _marker_count_pattern(region_count, rng)
    used_color_count = rng.randint(2, min(4, len(COLOR_POOL)))
    legend_count = rng.randint(used_color_count, min(4, len(COLOR_POOL)))

    legend_colors = rng.sample(COLOR_POOL, legend_count)
    used_colors = legend_colors[:used_color_count]
    rng.shuffle(used_colors)

    templates = {
        color: _random_template(tile_size, color, rng, legend_colors)
        for color in legend_colors
    }

    legend_span = legend_count * block_size + (legend_count - 1) * LEGEND_GAP
    if orientation == "right":
        max_main_w = 30 - AREA_GAP - block_size
        if max_main_w < 12:
            return None
        height = rng.randint(max(12, legend_span), 30)
        main_w = rng.randint(12, max_main_w)
        grid_h, grid_w = height, main_w + AREA_GAP + block_size
        main_h = grid_h
    else:
        max_main_h = 30 - AREA_GAP - block_size
        if max_main_h < 12:
            return None
        width = rng.randint(max(12, legend_span), 30)
        main_h = rng.randint(12, max_main_h)
        grid_h, grid_w = main_h + AREA_GAP + block_size, width
        main_w = grid_w

    grid = np.full((grid_h, grid_w), BG, dtype=int)

    frames = _place_frames(main_h, main_w, marker_counts, rng)
    if frames is None:
        return None

    region_assignments = _assign_region_colors(marker_counts, used_colors, rng)
    for (r0, c0, r1, c1), colors in zip(frames, region_assignments):
        grid[r0 : r1 + 1, c0] = FRAME
        grid[r0 : r1 + 1, c1] = FRAME
        grid[r0, c0 : c1 + 1] = FRAME
        grid[r1, c0 : c1 + 1] = FRAME
        _place_markers(grid, (r0, c0, r1, c1), colors, rng)

    if orientation == "right":
        legend_r0 = rng.randint(0, grid_h - legend_span)
        legend_c0 = main_w + AREA_GAP
        for idx, color in enumerate(legend_colors):
            br = legend_r0 + idx * (block_size + LEGEND_GAP)
            _paint_legend_block(grid, br, legend_c0, color, templates[color])
    else:
        legend_r0 = main_h + AREA_GAP
        legend_c0 = rng.randint(0, grid_w - legend_span)
        for idx, color in enumerate(legend_colors):
            bc = legend_c0 + idx * (block_size + LEGEND_GAP)
            _paint_legend_block(grid, legend_r0, bc, color, templates[color])

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def _marker_count_pattern(region_count: int, rng: random.Random) -> List[int]:
    if region_count == 1:
        return [rng.choice([1, 2])]
    counts = [1, 2]
    while len(counts) < region_count:
        counts.append(1 if len(counts) % 2 == 0 else 2)
    rng.shuffle(counts)
    return counts


def _random_template(
    tile_size: int,
    index_color: int,
    rng: random.Random,
    legend_colors: Sequence[int],
) -> np.ndarray:
    choices = [c for c in COLOR_POOL if c != FRAME and c != index_color]
    extra_choices = [c for c in legend_colors if c != index_color]
    for _ in range(200):
        template = np.full((tile_size, tile_size), BG, dtype=int)
        fill_palette = rng.sample(choices, rng.randint(2, min(3, len(choices))))
        if extra_choices and rng.random() < 0.5:
            fill_palette[0] = rng.choice(extra_choices)
        for r in range(tile_size):
            for c in range(tile_size):
                if rng.random() < 0.35:
                    template[r, c] = BG
                else:
                    template[r, c] = rng.choice(fill_palette)
        if np.count_nonzero(template != BG) < tile_size:
            continue
        if len({int(v) for v in template.flatten() if int(v) != BG}) < 2:
            continue
        if any(np.all(template[r, :] == BG) for r in range(tile_size)):
            continue
        if any(np.all(template[:, c] == BG) for c in range(tile_size)):
            continue
        return template
    raise RuntimeError("failed to sample a valid template")


def _place_frames(
    main_h: int,
    main_w: int,
    marker_counts: Sequence[int],
    rng: random.Random,
) -> List[Tuple[int, int, int, int]] | None:
    blocked = np.zeros((main_h, main_w), dtype=bool)
    sizes = []
    for marker_count in marker_counts:
        min_inner_w = 3 if marker_count == 1 else 4
        inner_h = rng.randint(3, min(8, main_h - 4))
        inner_w = rng.randint(min_inner_w, min(8, main_w - 4))
        sizes.append((inner_h + 2, inner_w + 2))
    sizes = sorted(sizes, key=lambda size: size[0] * size[1], reverse=True)

    frames: List[Tuple[int, int, int, int]] = []
    for outer_h, outer_w in sizes:
        placed = False
        if outer_h + 2 >= main_h or outer_w + 2 >= main_w:
            return None
        for _ in range(200):
            r0 = rng.randint(1, main_h - outer_h - 1)
            c0 = rng.randint(1, main_w - outer_w - 1)
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(main_h, r1 + 2)
            cc1 = min(main_w, c1 + 2)
            if blocked[rr0:rr1, cc0:cc1].any():
                continue
            blocked[rr0:rr1, cc0:cc1] = True
            frames.append((r0, c0, r1, c1))
            placed = True
            break
        if not placed:
            return None

    frames.sort()
    return frames


def _assign_region_colors(
    marker_counts: Sequence[int],
    used_colors: Sequence[int],
    rng: random.Random,
) -> List[List[int]]:
    assignments: List[List[int]] = []
    for marker_count in marker_counts:
        if marker_count == 1:
            assignments.append([rng.choice(used_colors)])
        else:
            assignments.append(rng.sample(list(used_colors), 2))
    return assignments


def _place_markers(
    grid: np.ndarray,
    frame: Tuple[int, int, int, int],
    colors: Sequence[int],
    rng: random.Random,
) -> None:
    r0, c0, r1, c1 = frame
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    rows = list(range(r0 + 1, r1))
    cols = list(range(c0 + 1, c1))
    if len(colors) == 1:
        grid[rng.choice(rows), rng.choice(cols)] = int(colors[0])
        return

    marker_cols = sorted(rng.sample(cols, 2))
    marker_rows = [rng.choice(rows), rng.choice(rows)]
    if inner_h > 1 and marker_rows[0] == marker_rows[1] and rng.random() < 0.5:
        marker_rows[1] = rows[(rows.index(marker_rows[1]) + 1) % len(rows)]
    grid[marker_rows[0], marker_cols[0]] = int(colors[0])
    grid[marker_rows[1], marker_cols[1]] = int(colors[1])


def _paint_legend_block(
    grid: np.ndarray,
    top: int,
    left: int,
    index_color: int,
    template: np.ndarray,
) -> None:
    grid[top, left] = int(index_color)
    h, w = template.shape
    grid[top + 1 : top + 1 + h, left + 1 : left + 1 + w] = template

