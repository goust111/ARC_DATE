import random
from collections import Counter

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
COLOR_POOL = list(range(1, 10))
MAX_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_of_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def spiral_coords(height: int, width: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    top, bottom = 0, height - 1
    left, right = 0, width - 1
    while top <= bottom and left <= right:
        for c in range(left, right + 1):
            coords.append((top, c))
        top += 1
        for r in range(top, bottom + 1):
            coords.append((r, right))
        right -= 1
        if top <= bottom:
            for c in range(right, left - 1, -1):
                coords.append((bottom, c))
            bottom -= 1
        if left <= right:
            for r in range(bottom, top - 1, -1):
                coords.append((r, left))
            left += 1
    return coords


def container_distance(cell: tuple[int, int], bbox: tuple[int, int, int, int]) -> int:
    r, c = cell
    r0, c0, r1, c1 = bbox
    dr = 0 if r0 <= r <= r1 else min(abs(r - r0), abs(r - r1))
    dc = 0 if c0 <= c <= c1 else min(abs(c - c0), abs(c - c1))
    return int(dr + dc)


def find_containers(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    containers = []
    for color in sorted(int(c) for c in np.unique(g) if int(c) != bg):
        for coords in components_of_mask(g == color):
            r0, c0, r1, c1 = bbox_coords(coords)
            h, w = r1 - r0 + 1, c1 - c0 + 1
            if h < 3 or w < 3:
                continue
            local = np.zeros((h, w), dtype=bool)
            local[coords[:, 0] - r0, coords[:, 1] - c0] = True
            border = np.zeros((h, w), dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if not np.array_equal(local, border):
                continue
            interior = g[r0 + 1 : r1, c0 + 1 : c1]
            marker_coords = np.argwhere(interior != bg)
            if marker_coords.shape[0] != 1:
                continue
            mr, mc = map(int, marker_coords[0])
            marker_pos = (r0 + 1 + mr, c0 + 1 + mc)
            marker_color = int(g[marker_pos])
            if marker_color in (bg, color):
                continue
            containers.append(
                {
                    "frame_color": int(color),
                    "bbox": (r0, c0, r1, c1),
                    "marker_pos": marker_pos,
                    "marker_color": marker_color,
                    "interior_shape": (h - 2, w - 2),
                }
            )
    containers.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return containers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Absorb each marker-colored scatter set into its matching hollow rectangle and redraw the amount as a clockwise spiral level.

    Input structure:
    - The background is the most common color.
    - Several separated monochrome hollow rectangular frames appear on the grid.
    - Each frame interior contains exactly one non-background marker pixel, and marker colors are pairwise distinct.
    - Outside the frames, isolated single-pixel scatters include some pixels whose colors match frame markers and some distractor colors that match no marker.

    Rule:
    - For each frame, read the color C of its interior marker pixel.
    - Count how many C-colored pixels lie outside that frame interior, erase those exterior C pixels to background, and clear the whole interior to background.
    - Starting from the interior top-left cell, fill exactly that count of cells with color C along a clockwise spiral; keep the frame and all non-matching distractor scatters unchanged.

    Key insight:
    - The interior marker is only a color label. The actual fill amount comes from a global count of same-colored exterior material, not from local position, distance, or the marker itself.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)

    containers = find_containers(g)
    assert containers, "expected at least one hollow rectangle container"

    marker_colors = [container["marker_color"] for container in containers]
    assert len(marker_colors) == len(set(marker_colors)), "marker colors must be unique"

    out = g.copy()
    fill_counts: dict[int, int] = {}
    for container in containers:
        marker_color = container["marker_color"]
        marker_r, marker_c = container["marker_pos"]
        fill_counts[marker_color] = int(np.count_nonzero(g == marker_color)) - 1
        assert fill_counts[marker_color] >= 0
        marker_mask = np.zeros_like(g, dtype=bool)
        marker_mask[marker_r, marker_c] = True
        out[(g == marker_color) & ~marker_mask] = bg

    for container in containers:
        r0, c0, r1, c1 = container["bbox"]
        inner_h, inner_w = container["interior_shape"]
        marker_color = container["marker_color"]
        n = fill_counts[marker_color]
        out[r0 + 1 : r1, c0 + 1 : c1] = bg
        path = spiral_coords(inner_h, inner_w)
        assert n <= len(path)
        for rr, cc in path[:n]:
            out[r0 + 1 + rr, c0 + 1 + cc] = marker_color
    return out


def sample_layout(count: int) -> list[int]:
    options = {
        2: [[2], [1, 1]],
        3: [[3], [2, 1], [1, 2]],
        4: [[4], [2, 2], [3, 1], [1, 3]],
    }
    return random.choice(options[count]).copy()


def sample_container_sizes(count: int) -> list[tuple[int, int]]:
    sizes = []
    for _ in range(count):
        outer_h = random.randint(4, 8)
        outer_w = random.randint(4, 9)
        sizes.append((outer_h, outer_w))
    return sizes


def place_containers(count: int) -> tuple[int, int, list[dict]] | None:
    row_groups = sample_layout(count)
    sizes = sample_container_sizes(count)
    random.shuffle(sizes)
    rows: list[list[tuple[int, int]]] = []
    idx = 0
    for group_size in row_groups:
        rows.append(sizes[idx : idx + group_size])
        idx += group_size

    horizontal_gaps = [[random.randint(2, 4) for _ in range(max(0, len(row) - 1))] for row in rows]
    vertical_gaps = [random.randint(2, 5) for _ in range(max(0, len(rows) - 1))]
    top_margin = random.randint(1, 3)
    bottom_margin = random.randint(1, 3)
    left_margin = random.randint(1, 3)
    right_margin = random.randint(1, 3)

    row_widths = []
    row_heights = []
    for row, gaps in zip(rows, horizontal_gaps):
        row_widths.append(sum(w for _, w in row) + sum(gaps))
        row_heights.append(max(h for h, _ in row))

    height = top_margin + sum(row_heights) + sum(vertical_gaps) + bottom_margin
    width = left_margin + max(row_widths) + right_margin
    if height < 18:
        bottom_margin += 18 - height
        height = 18
    if width < 18:
        right_margin += 18 - width
        width = 18
    if height > 30 or width > 30:
        return None

    specs = []
    y = top_margin
    for row, gaps, row_h, row_w in zip(rows, horizontal_gaps, row_heights, row_widths):
        extra = width - left_margin - right_margin - row_w
        row_x = left_margin + random.randint(0, max(0, extra))
        x = row_x
        for outer_h, outer_w in row:
            specs.append({"bbox": (y, x, y + outer_h - 1, x + outer_w - 1), "size": (outer_h, outer_w)})
            x += outer_w
            if gaps:
                x += gaps.pop(0)
        y += row_h
        if vertical_gaps:
            y += vertical_gaps.pop(0)

    random.shuffle(specs)
    return height, width, specs


def sample_fill_counts(capacities: list[int]) -> list[int]:
    count = len(capacities)
    indices = list(range(count))
    random.shuffle(indices)
    fills = [0] * count

    partial_idx = indices[0]
    fills[partial_idx] = random.randint(1, capacities[partial_idx] - 1)

    if count >= 2:
        second_idx = indices[1]
        if random.random() < 0.5:
            fills[second_idx] = 0
        else:
            fills[second_idx] = random.randint(max(1, capacities[second_idx] // 2), capacities[second_idx])

    if count >= 3:
        third_idx = indices[2]
        if random.random() < 0.5:
            fills[third_idx] = capacities[third_idx]
        else:
            fills[third_idx] = random.randint(1, capacities[third_idx])

    for idx in indices[3:]:
        fills[idx] = random.randint(0, capacities[idx])

    if not any(value > 0 for value in fills):
        fills[partial_idx] = random.randint(1, capacities[partial_idx] - 1)
    if len(set(fills)) == 1:
        tweak_idx = indices[-1]
        fills[tweak_idx] = capacities[tweak_idx] if fills[tweak_idx] != capacities[tweak_idx] else 0
    if not any(0 < value < capacities[idx] for idx, value in enumerate(fills)):
        fills[partial_idx] = max(1, capacities[partial_idx] // 2)
    return fills


def eligible_scatter_cells(grid: np.ndarray, container_bboxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int]]:
    h, w = grid.shape
    inside_any_bbox = np.zeros_like(grid, dtype=bool)
    for r0, c0, r1, c1 in container_bboxes:
        inside_any_bbox[r0 : r1 + 1, c0 : c1 + 1] = True

    cells = []
    for r in range(h):
        for c in range(w):
            if inside_any_bbox[r, c] or grid[r, c] != 0:
                continue
            ok = True
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] != 0:
                    ok = False
                    break
            if ok:
                cells.append((r, c))
    random.shuffle(cells)
    return cells


def choose_lure_cell(
    available: set[tuple[int, int]],
    owner_bbox: tuple[int, int, int, int],
    other_bboxes: list[tuple[int, int, int, int]],
) -> tuple[int, int] | None:
    lure_cells = []
    for cell in available:
        own_dist = container_distance(cell, owner_bbox)
        other_dist = min(container_distance(cell, bbox) for bbox in other_bboxes)
        if other_dist < own_dist:
            lure_cells.append(cell)
    return random.choice(lure_cells) if lure_cells else None


def place_isolated_pixels(
    grid: np.ndarray,
    placements: list[tuple[int, int, int]],
    available: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    for r, c, color in placements:
        if (r, c) not in available:
            return False
        grid[r, c] = color
        blocked = {(r, c)}
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                blocked.add((nr, nc))
        available.difference_update(blocked)
    return True


def expected_from_specs(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    bg = get_bg(grid)
    out = grid.copy()
    for spec in specs:
        marker_color = spec["marker_color"]
        out[out == marker_color] = bg
        r0, c0, r1, c1 = spec["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = bg
        path = spiral_coords(r1 - r0 - 1, c1 - c0 - 1)
        for rr, cc in path[: spec["fill_count"]]:
            out[r0 + 1 + rr, c0 + 1 + cc] = marker_color
    return out


def validate_instance(grid: np.ndarray, specs: list[dict], distractor_colors: set[int], lure_required: bool) -> bool:
    bg = get_bg(grid)
    if not np.array_equal(np.asarray(grid, dtype=int), grid):
        return False
    color_counts = Counter(map(int, grid.flat))
    if any(color_counts[bg] <= count for color, count in color_counts.items() if color != bg):
        return False

    containers = find_containers(grid)
    if len(containers) != len(specs):
        return False

    by_marker = {spec["marker_color"]: spec for spec in specs}
    marker_colors = {spec["marker_color"] for spec in specs}
    if len(marker_colors) != len(specs):
        return False

    lure_seen = not lure_required
    positive_containers = 0
    all_bboxes = [spec["bbox"] for spec in specs]
    for container in containers:
        marker_color = container["marker_color"]
        if marker_color not in by_marker:
            return False
        spec = by_marker[marker_color]
        if container["bbox"] != spec["bbox"]:
            return False
        r0, c0, r1, c1 = spec["bbox"]
        capacity = (r1 - r0 - 1) * (c1 - c0 - 1)
        if spec["fill_count"] > capacity:
            return False
        same_color_outside = int(np.count_nonzero(grid == marker_color)) - 1
        if same_color_outside != spec["fill_count"]:
            return False
        if same_color_outside > 0:
            positive_containers += 1
            if not lure_seen:
                owner_bbox = spec["bbox"]
                other_bboxes = [bbox for bbox in all_bboxes if bbox != owner_bbox]
                for cell in spec["scatter_cells"]:
                    if other_bboxes and container_distance(cell, owner_bbox) > min(container_distance(cell, bbox) for bbox in other_bboxes):
                        lure_seen = True
                        break

    if positive_containers == 0 or not lure_seen:
        return False

    outside_bbox = np.ones_like(grid, dtype=bool)
    for r0, c0, r1, c1 in all_bboxes:
        outside_bbox[r0 : r1 + 1, c0 : c1 + 1] = False
    outside_fg = np.argwhere((grid != bg) & outside_bbox)
    for r, c in outside_fg:
        neigh = 0
        for dr, dc in DIRS4:
            nr, nc = int(r + dr), int(c + dc)
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and grid[nr, nc] != bg:
                neigh += 1
        if neigh != 0:
            return False

    distractor_present = any(int(grid[tuple(cell)]) in distractor_colors for cell in outside_fg)
    if not distractor_present:
        return False

    expected = expected_from_specs(grid, specs)
    actual = main(grid)
    if np.array_equal(actual, grid):
        return False
    if not np.array_equal(actual, expected):
        return False
    return True


def build_attempt() -> np.ndarray | None:
    container_count = random.randint(2, 4)
    layout = place_containers(container_count)
    if layout is None:
        return None
    height, width, specs = layout

    marker_colors = random.sample(COLOR_POOL, container_count)
    remaining = [color for color in COLOR_POOL if color not in marker_colors]
    frame_palette_size = random.randint(1, min(3, len(remaining) - 1))
    frame_palette = random.sample(remaining, frame_palette_size)
    distractor_pool = [color for color in remaining if color not in frame_palette]
    if not distractor_pool:
        return None
    distractor_count = random.randint(1, min(2, len(distractor_pool)))
    distractor_colors = set(random.sample(distractor_pool, distractor_count))

    capacities = []
    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        capacities.append((r1 - r0 - 1) * (c1 - c0 - 1))
    fill_counts = sample_fill_counts(capacities)

    grid = np.zeros((height, width), dtype=int)
    for idx, spec in enumerate(specs):
        frame_color = random.choice(frame_palette)
        marker_color = marker_colors[idx]
        r0, c0, r1, c1 = spec["bbox"]
        grid[r0, c0 : c1 + 1] = frame_color
        grid[r1, c0 : c1 + 1] = frame_color
        grid[r0 : r1 + 1, c0] = frame_color
        grid[r0 : r1 + 1, c1] = frame_color
        marker_r = random.randint(r0 + 1, r1 - 1)
        marker_c = random.randint(c0 + 1, c1 - 1)
        grid[marker_r, marker_c] = marker_color
        spec["frame_color"] = frame_color
        spec["marker_color"] = marker_color
        spec["marker_pos"] = (marker_r, marker_c)
        spec["fill_count"] = fill_counts[idx]
        spec["scatter_cells"] = []

    available = set(eligible_scatter_cells(grid, [spec["bbox"] for spec in specs]))
    distractor_total = random.randint(2, 7)
    required = sum(fill_counts) + distractor_total
    if len(available) < required:
        return None

    lure_required = container_count >= 2 and any(count > 0 for count in fill_counts) and random.random() < 0.7
    lure_placements: list[tuple[int, int, int]] = []
    remaining_marker_counts = {spec["marker_color"]: spec["fill_count"] for spec in specs}
    if lure_required:
        owners = [spec for spec in specs if spec["fill_count"] > 0]
        random.shuffle(owners)
        placed_lure = False
        for owner in owners:
            other_bboxes = [spec["bbox"] for spec in specs if spec is not owner]
            if not other_bboxes:
                continue
            lure = choose_lure_cell(available, owner["bbox"], other_bboxes)
            if lure is None:
                continue
            owner["scatter_cells"].append(lure)
            remaining_marker_counts[owner["marker_color"]] -= 1
            lure_placements.append((lure[0], lure[1], owner["marker_color"]))
            placed_lure = True
            break
        if not placed_lure:
            return None

    remaining_placements: list[tuple[int, int, int]] = []
    color_multiset = []
    for spec in specs:
        color_multiset.extend([spec["marker_color"]] * remaining_marker_counts[spec["marker_color"]])
    distractor_colors_list = list(distractor_colors)
    for _ in range(distractor_total):
        color_multiset.append(random.choice(distractor_colors_list))
    random.shuffle(color_multiset)

    if not place_isolated_pixels(grid, lure_placements, available):
        return None

    for color in color_multiset:
        if not available:
            return None
        cell = random.choice(tuple(available))
        grid[cell[0], cell[1]] = color
        remaining_placements.append((cell[0], cell[1], color))
        blocked = {cell}
        for dr, dc in DIRS4:
            nr, nc = cell[0] + dr, cell[1] + dc
            if 0 <= nr < height and 0 <= nc < width:
                blocked.add((nr, nc))
        available.difference_update(blocked)

    for r, c, color in remaining_placements:
        if color in marker_colors:
            for spec in specs:
                if spec["marker_color"] == color:
                    spec["scatter_cells"].append((r, c))
                    break

    if validate_instance(grid, specs, distractor_colors, lure_required):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid container-spiral task")
