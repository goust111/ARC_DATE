from __future__ import annotations

from collections import deque
import random

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
COLORS = tuple(range(10))


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(mask: np.ndarray) -> list[dict]:
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components = []

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue

            queue = deque([(r, c)])
            seen[r, c] = True
            coords = []

            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            components.append(
                {
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(arr.shape[0]),
                }
            )

    return components


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.array(grid[r0 : r1 + 1, c0 : c1 + 1], copy=True)


def _split_panels(grid: np.ndarray) -> tuple[str, int, np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    candidates = []

    if height % 2 == 1:
        divider_r = height // 2
        if np.all(g[divider_r, :] == g[divider_r, 0]):
            top = g[:divider_r, :]
            bottom = g[divider_r + 1 :, :]
            if top.size and top.shape == bottom.shape:
                candidates.append(("horizontal", divider_r, top, bottom))

    if width % 2 == 1:
        divider_c = width // 2
        if np.all(g[:, divider_c] == g[0, divider_c]):
            left = g[:, :divider_c]
            right = g[:, divider_c + 1 :]
            if left.size and left.shape == right.shape:
                candidates.append(("vertical", divider_c, left, right))

    assert len(candidates) == 1, "expected exactly one centered divider"
    return candidates[0]


def _stack_panel(panel: np.ndarray, bg: int) -> np.ndarray:
    objects = _find_components(panel != bg)
    assert objects, "selected panel must contain objects"
    objects.sort(key=lambda obj: (-obj["area"], obj["bbox"][0], obj["bbox"][1]))

    widths = [obj["bbox"][3] - obj["bbox"][1] + 1 for obj in objects]
    heights = [obj["bbox"][2] - obj["bbox"][0] + 1 for obj in objects]
    out_h = int(sum(heights))
    out_w = int(max(widths))
    out = np.full((out_h, out_w), bg, dtype=int)

    bottom = out_h
    for obj in objects:
        patch = _crop_bbox(panel, obj["bbox"])
        mask = patch != bg
        ph, pw = patch.shape
        bottom -= ph
        left = (out_w - pw) // 2
        out_slice = out[bottom : bottom + ph, left : left + pw]
        out_slice[mask] = patch[mask]

    return out


def _analyze_input(grid: np.ndarray) -> tuple[str, int, int]:
    bg = get_bg(grid)
    orientation, _index, first, second = _split_panels(grid)
    return orientation, len(_find_components(first != bg)), len(_find_components(second != bg))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the panel with fewer objects and stack its objects into a centered pyramid.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = get_bg(g)

    _orientation, _index, first, second = _split_panels(g)
    first_count = len(_find_components(first != bg))
    second_count = len(_find_components(second != bg))
    chosen = first if first_count <= second_count else second
    return _stack_panel(chosen, bg)


# --------------- generator ---------------

MAX_GENERATION_ATTEMPTS = 400
PANEL_ATTEMPTS = 80
OBJECT_ATTEMPTS = 80
POSITION_ATTEMPTS = 100

VERTICAL_HEIGHTS = (10, 12, 14)
VERTICAL_PANEL_WIDTHS = tuple(range(6, 11))
HORIZONTAL_PANEL_HEIGHTS = tuple(range(5, 9))
HORIZONTAL_WIDTHS = (10, 12, 14)


def _normalize_cells(cells: set[tuple[int, int]]) -> tuple[list[tuple[int, int]], int, int]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    norm = {(r - min_r, c - min_c) for r, c in cells}
    height = max(r for r, _ in norm) + 1
    width = max(c for _, c in norm) + 1
    return sorted(norm), int(height), int(width)


def _random_shape(panel_h: int, panel_w: int, object_count: int) -> tuple[list[tuple[int, int]], int, int] | None:
    shape_h_cap = 5 if object_count <= 2 else 4
    shape_w_cap = 6 if object_count <= 2 else 5
    max_h = min(shape_h_cap, panel_h - 2)
    max_w = min(shape_w_cap, panel_w - 2)
    if max_h < 1 or max_w < 1:
        return None

    for _ in range(OBJECT_ATTEMPTS):
        box_h = random.randint(1, max_h)
        box_w = random.randint(1, max_w)
        max_area = min(box_h * box_w, 10 if object_count <= 2 else 8)
        if max_area <= 1:
            target_area = 1
        else:
            min_area = 1 if random.random() < 0.15 else 2
            target_area = random.randint(min_area, max_area)

        cells = {(random.randrange(box_h), random.randrange(box_w))}
        while len(cells) < target_area:
            frontier = set()
            for r, c in cells:
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))

        if len(cells) == target_area:
            return _normalize_cells(cells)

    return None


def _block_cells(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    height, width = blocked.shape
    for r, c in cells:
        for rr in range(max(0, r - 1), min(height, r + 2)):
            for cc in range(max(0, c - 1), min(width, c + 2)):
                blocked[rr, cc] = True


def _make_panel(panel_h: int, panel_w: int, bg: int, object_colors: list[int]) -> np.ndarray | None:
    for _ in range(PANEL_ATTEMPTS):
        panel = np.full((panel_h, panel_w), bg, dtype=int)
        blocked = np.zeros((panel_h, panel_w), dtype=bool)

        for color in object_colors:
            shape = _random_shape(panel_h, panel_w, len(object_colors))
            if shape is None:
                break
            cells, shape_h, shape_w = shape
            min_r = 1
            min_c = 1
            max_r = panel_h - shape_h - 1
            max_c = panel_w - shape_w - 1
            if max_r < min_r or max_c < min_c:
                break

            placed = False
            for _ in range(POSITION_ATTEMPTS):
                r0 = random.randint(min_r, max_r)
                c0 = random.randint(min_c, max_c)
                placed_cells = [(r0 + r, c0 + c) for r, c in cells]
                if any(blocked[r, c] for r, c in placed_cells):
                    continue
                for r, c in placed_cells:
                    panel[r, c] = color
                _block_cells(blocked, placed_cells)
                placed = True
                break

            if not placed:
                break
        else:
            if len(_find_components(panel != bg)) == len(object_colors):
                return panel

    return None


def _choose_counts(branch: str) -> tuple[int, int]:
    if branch == "tie":
        count = random.randint(1, 4)
        return count, count
    if branch == "first_fewer":
        first = random.randint(1, 3)
        second = random.randint(first + 1, 4)
        return first, second
    second = random.randint(1, 3)
    first = random.randint(second + 1, 4)
    return first, second


def _sample_distinct_colors(count: int, excluded: set[int]) -> list[int]:
    choices = [color for color in COLORS if color not in excluded]
    return random.sample(choices, count)


def _gen_attempt() -> np.ndarray | None:
    orientation = random.choice(("vertical", "horizontal"))
    branch = random.choice(("first_fewer", "second_fewer", "tie"))
    first_count, second_count = _choose_counts(branch)

    bg = random.choice(COLORS)
    divider_color = random.choice([color for color in COLORS if color != bg])

    if orientation == "vertical":
        height = random.choice(VERTICAL_HEIGHTS)
        panel_w = random.choice(VERTICAL_PANEL_WIDTHS)
        width = 2 * panel_w + 1
        first_shape = (height, panel_w)
        second_shape = (height, panel_w)
    else:
        panel_h = random.choice(HORIZONTAL_PANEL_HEIGHTS)
        width = random.choice(HORIZONTAL_WIDTHS)
        height = 2 * panel_h + 1
        first_shape = (panel_h, width)
        second_shape = (panel_h, width)

    first_colors = _sample_distinct_colors(first_count, {bg, divider_color})
    second_colors = _sample_distinct_colors(second_count, {bg, divider_color})
    first_panel = _make_panel(first_shape[0], first_shape[1], bg, first_colors)
    second_panel = _make_panel(second_shape[0], second_shape[1], bg, second_colors)
    if first_panel is None or second_panel is None:
        return None

    grid = np.full((height, width), bg, dtype=int)
    if orientation == "vertical":
        divider_c = first_shape[1]
        grid[:, divider_c] = divider_color
        grid[:, :divider_c] = first_panel
        grid[:, divider_c + 1 :] = second_panel
    else:
        divider_r = first_shape[0]
        grid[divider_r, :] = divider_color
        grid[:divider_r, :] = first_panel
        grid[divider_r + 1 :, :] = second_panel

    if get_bg(grid) != bg:
        return None

    found_orientation, found_first, found_second = _analyze_input(grid)
    if found_orientation != orientation or found_first != first_count or found_second != second_count:
        return None

    chosen_panel = first_panel if first_count <= second_count else second_panel
    chosen_areas = [obj["area"] for obj in _find_components(chosen_panel != bg)]
    if len(chosen_areas) > 1 and len(set(chosen_areas)) != len(chosen_areas):
        return None

    out = main(grid)
    chosen_width = max(obj["bbox"][3] - obj["bbox"][1] + 1 for obj in _find_components(chosen_panel != bg))
    chosen_height = sum(obj["bbox"][2] - obj["bbox"][0] + 1 for obj in _find_components(chosen_panel != bg))
    if out.shape != (chosen_height, chosen_width):
        return None
    if int(np.count_nonzero(out != bg)) != int(np.count_nonzero(chosen_panel != bg)):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
