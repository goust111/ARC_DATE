import random
from collections import Counter

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
TASK_COLORS = (BLUE, RED, GREEN)

LEGEND_BOX_WIDTH = 9
GEN_RETRY_LIMIT = 400


def get_bg(grid: np.ndarray) -> int:
    values = Counter(map(int, grid.ravel()))
    return int(values.most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return bbox_coords(coords)


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _find_legend_box(grid: np.ndarray) -> tuple[int, int, int, int]:
    grey_mask = grid == GREY
    box = bbox(grey_mask)
    assert box is not None, "expected a grey legend frame"
    r0, c0, r1, c1 = box

    border = np.zeros_like(grey_mask)
    border[r0, c0 : c1 + 1] = True
    border[r1, c0 : c1 + 1] = True
    border[r0 : r1 + 1, c0] = True
    border[r0 : r1 + 1, c1] = True
    assert np.array_equal(grey_mask, border), "grey pixels must form one rectangular border"
    assert c1 - c0 + 1 >= 7 and r1 - r0 + 1 >= 5
    return r0, c0, r1, c1


def _outside_box_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.ones(shape, dtype=bool)
    r0, c0, r1, c1 = box
    mask[r0 : r1 + 1, c0 : c1 + 1] = False
    return mask


def _legend_keys(grid: np.ndarray, box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, _c1 = box
    key_c = c0 + 1
    keys = []
    for r in range(r0 + 1, r1):
        color = int(grid[r, key_c])
        if color not in (BG, GREY):
            keys.append((r, color))
    assert 2 <= len(keys) <= 3
    assert len({color for _, color in keys}) == len(keys)
    for (ra, _), (rb, _) in zip(keys, keys[1:]):
        assert rb - ra >= 2
    return keys


def _outside_components(grid: np.ndarray, box: tuple[int, int, int, int]) -> list[dict]:
    outside = _outside_box_mask(grid.shape, box)
    components: list[dict] = []
    for color in TASK_COLORS:
        comps = _connected_components((grid == color) & outside)
        for coords in comps:
            components.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return components


def _slide_order(component: dict) -> tuple[int, int, int, int, int]:
    r0, c0, r1, c1 = component["bbox"]
    return (-c1, -c0, r0, r1, -int(component["area"]))


def _slide_objects(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    h, w = grid.shape
    out = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    r0, c0, r1, c1 = box
    occupied[r0, c0 : c1 + 1] = True
    occupied[r1, c0 : c1 + 1] = True
    occupied[r0 : r1 + 1, c0] = True
    occupied[r0 : r1 + 1, c1] = True
    out[occupied] = GREY

    for component in sorted(_outside_components(grid, box), key=_slide_order):
        color = int(component["color"])
        coords = np.asarray(component["coords"], dtype=int)
        rows = coords[:, 0]
        cols = coords[:, 1]
        shift = 0
        while True:
            next_cols = cols + shift + 1
            if np.any(next_cols >= w):
                break
            if np.any(occupied[rows, next_cols]):
                break
            shift += 1

        dest_cols = cols + shift
        assert not np.any(occupied[rows, dest_cols])
        out[rows, dest_cols] = color
        occupied[rows, dest_cols] = True

    return out


def _count_components_by_color(
    grid: np.ndarray, box: tuple[int, int, int, int], colors: list[int]
) -> dict[int, int]:
    outside = _outside_box_mask(grid.shape, box)
    counts: dict[int, int] = {}
    for color in colors:
        counts[int(color)] = len(_connected_components((grid == int(color)) & outside))
    return counts


def _render_legend(
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    keys: list[tuple[int, int]],
    counts: dict[int, int],
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    r0, c0, r1, c1 = box
    key_c = c0 + 1

    out[r0 : r1 + 1, c0 : c1 + 1] = BG
    out[r0, c0 : c1 + 1] = GREY
    out[r1, c0 : c1 + 1] = GREY
    out[r0 : r1 + 1, c0] = GREY
    out[r0 : r1 + 1, c1] = GREY

    for row, color in keys:
        out[row, key_c] = int(color)
        for i in range(counts[int(color)]):
            dot_c = key_c + 2 * (i + 1)
            assert dot_c < c1, "legend is too narrow for the tally dots"
            out[row, dot_c] = int(color)

    return out


def _analyze_instance(grid: np.ndarray) -> dict:
    box = _find_legend_box(grid)
    keys = _legend_keys(grid, box)
    key_colors = [color for _, color in keys]
    initial_counts = _count_components_by_color(grid, box, key_colors)
    slidden = _slide_objects(grid, box)
    final_counts = _count_components_by_color(slidden, box, key_colors)
    return {
        "box": box,
        "keys": keys,
        "initial_counts": initial_counts,
        "slidden": slidden,
        "final_counts": final_counts,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Slide the left-side colored objects rightward, merge same-color contacts, and write the merged tallies into the legend box.

    Input structure:
    - A black background contains several blue, red, and/or green monochromatic 4-connected objects in the left portion of the grid.
    - On the right sits a grey one-pixel rectangular legend frame whose leftmost interior column holds one key pixel for each used object color.

    Rule:
    - Keep the legend frame fixed, clear the movable objects, and process those outside the legend from right to left by original bounding-box right edge.
    - Each object slides horizontally to the right until its next step would leave the grid, hit an already settled object, or hit a grey legend-border pixel.
    - After all objects settle, recount 4-connected components of each keyed color outside the legend, treating newly touching same-color objects as one merged object.
    - Repaint the legend interior so each key row keeps its key pixel and gains one same-color dot every other column to the right, with one dot per final component count.

    Key insight:
    - The legend counts the post-slide connected components, not the initial objects, so the rightward gravity simulation must be done before tallying.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    assert get_bg(g) == BG

    analysis = _analyze_instance(g)
    out = _render_legend(analysis["slidden"], analysis["box"], analysis["keys"], analysis["final_counts"])
    return out


def _sample_counts(n_colors: int) -> list[int]:
    if n_colors == 2:
        return list(random.choice(((3, 3), (4, 2), (4, 3))))
    return list(random.choice(((3, 2, 2), (4, 2, 2), (3, 3, 2), (4, 3, 2))))


def _place_bar(grid: np.ndarray, row: int, start: int, length: int, color: int) -> None:
    grid[row, start : start + length] = int(color)


def _sample_key_rows(height: int, count: int) -> list[int]:
    candidates = list(range(1, height - 1, 2))
    assert len(candidates) >= count
    return sorted(random.sample(candidates, count))


def _gen_attempt() -> np.ndarray | None:
    n_colors = random.choice((2, 3))
    colors = random.sample(list(TASK_COLORS), n_colors)
    counts = _sample_counts(n_colors)
    primary_count = counts[0]
    total_objects = sum(counts)
    lane_count = total_objects - 1

    min_h = max(12, 2 * lane_count + 1)
    max_h = min(18, min_h + 3)
    if min_h > max_h:
        return None

    height = random.randint(min_h, max_h)
    width = random.randint(17, 20)
    legend_left = width - LEGEND_BOX_WIDTH
    if legend_left < 8:
        return None

    grid = np.zeros((height, width), dtype=int)
    box = (0, legend_left, height - 1, width - 1)
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY

    key_rows = _sample_key_rows(height, n_colors)
    key_colors = colors[:]
    random.shuffle(key_colors)
    for row, color in zip(key_rows, key_colors):
        grid[row, c0 + 1] = int(color)

    lane_rows = sorted(random.sample(list(range(1, height - 1, 2)), lane_count))
    merge_row = random.choice(lane_rows)
    lane_rows.remove(merge_row)

    right_len = random.randint(2, min(5, legend_left - 2))
    left_len = random.randint(2, min(5, legend_left - right_len - 1))
    if left_len < 2 or right_len < 2:
        return None

    max_right_start = legend_left - right_len
    min_right_start = left_len + 2
    if min_right_start > max_right_start:
        return None
    right_start = random.randint(min_right_start, max_right_start)
    max_left_start = right_start - left_len - 1
    if max_left_start < 0:
        return None
    left_start = random.randint(0, max_left_start)

    primary_color = colors[0]
    _place_bar(grid, merge_row, left_start, left_len, primary_color)
    _place_bar(grid, merge_row, right_start, right_len, primary_color)

    object_specs: list[tuple[int, int]] = []
    remaining_primary = primary_count - 2
    object_specs.extend((primary_color, random.randint(2, 5)) for _ in range(remaining_primary))
    for color, count in zip(colors[1:], counts[1:]):
        object_specs.extend((color, random.randint(2, 5)) for _ in range(count))
    random.shuffle(object_specs)

    if len(object_specs) != len(lane_rows):
        return None

    for row, (color, length) in zip(lane_rows, object_specs):
        if length >= legend_left:
            return None
        start = random.randint(0, legend_left - length)
        _place_bar(grid, row, start, length, color)

    analysis = _analyze_instance(grid)
    final_counts = analysis["final_counts"]
    initial_counts = analysis["initial_counts"]

    if sorted(color for _, color in analysis["keys"]) != sorted(colors):
        return None
    if sum(initial_counts.values()) != total_objects:
        return None
    if not (initial_counts[primary_color] == primary_count and final_counts[primary_color] == primary_count - 1):
        return None
    if final_counts[primary_color] < 2:
        return None
    if any(final_counts[color] > 3 for color in final_counts):
        return None
    if sum(final_counts.values()) >= sum(initial_counts.values()):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
