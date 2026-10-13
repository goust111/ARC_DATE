from __future__ import annotations

import math
import random

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _background_color(grid: np.ndarray) -> int:
    flat = np.asarray(grid, dtype=int).ravel()
    assert flat.size > 0, "expected a non-empty grid"
    counts = np.bincount(flat, minlength=int(flat.max()) + 1)
    return int(np.argmax(counts))


def _find_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                }
            )

    return components


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each foreground object with a line segment aligned to the longer side of its bbox.

    Rule:
    - If bbox width >= bbox height, draw a horizontal segment across the bbox center row.
    - Otherwise, draw a vertical segment across the bbox center column.
    - Keep the object's color and erase all other original pixels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    components = _find_components(g, bg)
    assert components, "expected at least one foreground object"

    out = np.full(g.shape, bg, dtype=int)
    for comp in components:
        color = int(comp["color"])
        r0, c0, r1, c1 = comp["bbox"]
        height = r1 - r0 + 1
        width = c1 - c0 + 1

        if width >= height:
            row = (r0 + r1) // 2
            out[row, c0 : c1 + 1] = color
        else:
            col = (c0 + c1) // 2
            out[r0 : r1 + 1, col] = color

    return out


# --------------- generator ---------------

BACKGROUND_COLOR = 7  # orange
FOREGROUND_COLORS = tuple(color for color in range(10) if color != BACKGROUND_COLOR)

GRID_MIN_SIZE = 14
GRID_MAX_SIZE = 28
OBJECT_COUNT_RANGE = (2, 5)
OUTER_MARGIN = 1
BBOX_PADDING = 1
ODD_SIDES = (3, 5, 7, 9)
SQUARE_PROBABILITY = 0.2
MIN_FILL_RATIO = 0.38
MAX_FILL_RATIO = 0.74
PLACEMENT_TRIES = 80
SHAPE_TRIES = 80
GENERATION_TRIES = 300


def _random_path(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = start
    rr, cc = end
    path = [(r, c)]
    while (r, c) != (rr, cc):
        options: list[tuple[int, int]] = []
        if r < rr:
            options.append((r + 1, c))
        elif r > rr:
            options.append((r - 1, c))
        if c < cc:
            options.append((r, c + 1))
        elif c > cc:
            options.append((r, c - 1))
        r, c = random.choice(options)
        path.append((r, c))
    return path


def _frontier(cells: set[tuple[int, int]], height: int, width: int) -> list[tuple[int, int]]:
    border = set()
    for r, c in cells:
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                border.add((nr, nc))
    return list(border)


def _line_mask(height: int, width: int, horizontal: bool) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    if horizontal:
        mask[height // 2, :] = True
    else:
        mask[:, width // 2] = True
    return mask


def _random_component_mask(height: int, width: int) -> np.ndarray | None:
    area = height * width
    target_line_area = max(height, width)
    for _ in range(SHAPE_TRIES):
        center = (height // 2, width // 2)
        anchors = [
            center,
            (0, random.randrange(width)),
            (height - 1, random.randrange(width)),
            (random.randrange(height), 0),
            (random.randrange(height), width - 1),
        ]

        cells = {center}
        for anchor in anchors[1:]:
            cells.update(_random_path(anchor, center))

        low = max(len(cells), target_line_area + 1, math.ceil(area * MIN_FILL_RATIO))
        high = min(area - 1, math.floor(area * MAX_FILL_RATIO))
        if high < low:
            high = area - 1
        if high < low:
            continue

        target_area = random.randint(low, high)
        while len(cells) < target_area:
            border = _frontier(cells, height, width)
            if not border:
                break
            boundary_cells = [cell for cell in border if 0 in cell or cell[0] == height - 1 or cell[1] == width - 1]
            pool = boundary_cells if boundary_cells and random.random() < 0.3 else border
            cells.add(random.choice(pool))

        mask = np.zeros((height, width), dtype=bool)
        rr, cc = np.array(sorted(cells), dtype=int).T
        mask[rr, cc] = True

        if mask.all():
            continue
        if not mask[0, :].any() or not mask[-1, :].any() or not mask[:, 0].any() or not mask[:, -1].any():
            continue
        if np.array_equal(mask, _line_mask(height, width, horizontal=width >= height)):
            continue
        return mask

    return None


def _sample_bbox(kind: str, grid_h: int, grid_w: int) -> tuple[int, int] | None:
    max_h = grid_h - 2 * OUTER_MARGIN
    max_w = grid_w - 2 * OUTER_MARGIN
    side_h = [value for value in ODD_SIDES if value <= max_h]
    side_w = [value for value in ODD_SIDES if value <= max_w]
    if not side_h or not side_w:
        return None

    if kind == "wide":
        if random.random() < SQUARE_PROBABILITY:
            square_choices = [value for value in side_h if value in side_w]
            if square_choices:
                side = random.choice(square_choices)
                return side, side
        choices = [(h, w) for h in side_h for w in side_w if w > h]
    else:
        choices = [(h, w) for h in side_h for w in side_w if h > w]

    if not choices:
        return None
    return random.choice(choices)


def _branch_kinds(count: int) -> list[str]:
    assert count >= 2
    kinds = ["wide", "tall"]
    while len(kinds) < count:
        kinds.append(random.choice(("wide", "tall")))
    random.shuffle(kinds)
    return kinds


def _place_specs(grid_h: int, grid_w: int, specs: list[dict[str, object]]) -> bool:
    reserved = np.zeros((grid_h, grid_w), dtype=bool)
    specs.sort(key=lambda spec: int(spec["height"]) * int(spec["width"]), reverse=True)

    for spec in specs:
        height = int(spec["height"])
        width = int(spec["width"])
        r_lo = OUTER_MARGIN
        c_lo = OUTER_MARGIN
        r_hi = grid_h - height - OUTER_MARGIN
        c_hi = grid_w - width - OUTER_MARGIN
        if r_hi < r_lo or c_hi < c_lo:
            return False

        placed = False
        for _ in range(PLACEMENT_TRIES):
            top = random.randint(r_lo, r_hi)
            left = random.randint(c_lo, c_hi)
            rr0 = max(0, top - BBOX_PADDING)
            cc0 = max(0, left - BBOX_PADDING)
            rr1 = min(grid_h, top + height + BBOX_PADDING)
            cc1 = min(grid_w, left + width + BBOX_PADDING)
            if reserved[rr0:rr1, cc0:cc1].any():
                continue
            reserved[rr0:rr1, cc0:cc1] = True
            spec["top"] = top
            spec["left"] = left
            placed = True
            break

        if not placed:
            return False

    return True


def _render_input(grid_h: int, grid_w: int, specs: list[dict[str, object]]) -> np.ndarray:
    grid = np.full((grid_h, grid_w), BACKGROUND_COLOR, dtype=int)
    for spec in specs:
        mask = spec["mask"]
        color = int(spec["color"])
        top = int(spec["top"])
        left = int(spec["left"])
        rr, cc = np.argwhere(mask).T
        grid[top + rr, left + cc] = color
    return grid


def _render_expected(grid_h: int, grid_w: int, specs: list[dict[str, object]]) -> np.ndarray:
    out = np.full((grid_h, grid_w), BACKGROUND_COLOR, dtype=int)
    for spec in specs:
        color = int(spec["color"])
        top = int(spec["top"])
        left = int(spec["left"])
        height = int(spec["height"])
        width = int(spec["width"])
        if width >= height:
            out[top + height // 2, left : left + width] = color
        else:
            out[top : top + height, left + width // 2] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    object_count = random.randint(*OBJECT_COUNT_RANGE)
    min_side = max(GRID_MIN_SIZE, 10 + 2 * object_count)
    grid_h = random.randint(min_side, GRID_MAX_SIZE)
    grid_w = random.randint(min_side, GRID_MAX_SIZE)

    kinds = _branch_kinds(object_count)
    colors = random.sample(FOREGROUND_COLORS, object_count)
    specs: list[dict[str, object]] = []

    for kind, color in zip(kinds, colors):
        sampled = _sample_bbox(kind, grid_h, grid_w)
        if sampled is None:
            return None
        height, width = sampled
        mask = _random_component_mask(height, width)
        if mask is None:
            return None
        specs.append(
            {
                "kind": kind,
                "color": color,
                "height": height,
                "width": width,
                "mask": mask,
            }
        )

    if not _place_specs(grid_h, grid_w, specs):
        return None

    inp = _render_input(grid_h, grid_w, specs)
    expected = _render_expected(grid_h, grid_w, specs)

    if _background_color(inp) != BACKGROUND_COLOR:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_TRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
