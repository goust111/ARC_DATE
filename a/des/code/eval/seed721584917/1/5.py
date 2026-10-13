import random
from typing import Iterable

import numpy as np


BG = 0
GREY = 5
MAROON = 9
SPECIAL_COLORS = {BG, GREY, MAROON}
FRAME_COLORS = [1, 2, 3, 4, 6, 7, 8]
MAX_GENERATION_ATTEMPTS = 400


TEMPLATE_POOL = [
    np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=bool),
    np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),
]


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(comp)
    return parts


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _tile_mask(template: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    reps_r = (shape[0] + template.shape[0] - 1) // template.shape[0]
    reps_c = (shape[1] + template.shape[1] - 1) // template.shape[1]
    return np.tile(template, (reps_r, reps_c))[: shape[0], : shape[1]]


def _extract_templates(grid: np.ndarray) -> list[np.ndarray]:
    grey_coords = np.argwhere(grid == GREY)
    assert grey_coords.size > 0, "missing grey template library"
    r0, c0 = grey_coords.min(axis=0)
    r1, c1 = grey_coords.max(axis=0)
    r0 = int(r0)
    c0 = int(c0)
    r1 = int(r1)
    c1 = int(c1)
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert sub.shape[0] == 5, "template library must be 5 cells tall including border"
    assert np.all(sub[0, :] == GREY) and np.all(sub[-1, :] == GREY), "bad library border rows"
    divider_cols = [c for c in range(sub.shape[1]) if np.all(sub[:, c] == GREY)]
    assert divider_cols[0] == 0 and divider_cols[-1] == sub.shape[1] - 1, "library border columns missing"
    assert len(divider_cols) >= 4, "expected at least three templates"

    templates: list[np.ndarray] = []
    for left, right in zip(divider_cols, divider_cols[1:]):
        assert right - left == 4, "template cells must be 3 columns wide"
        patch = sub[1:-1, left + 1 : right]
        assert patch.shape == (3, 3), "template patch must be 3x3"
        assert np.all((patch == BG) | (patch == MAROON)), "template patch contains unexpected colors"
        template = patch == MAROON
        assert np.any(template), "template cannot be empty"
        templates.append(template)
    return templates


def _find_rectangular_frames(grid: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    frames: list[tuple[int, int, int, int, int]] = []
    for color_value in np.unique(grid):
        color = int(color_value)
        if color in SPECIAL_COLORS:
            continue
        for comp in _components(grid == color):
            r0, c0, r1, c1 = _bbox(comp)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 3 or w < 3:
                continue
            border = np.zeros((h, w), dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            sub = grid[r0 : r1 + 1, c0 : c1 + 1]
            if np.all(sub[border] == color) and np.any(sub[1:-1, 1:-1] != color):
                frames.append((r0, c0, r1, c1, color))
    frames.sort()
    return frames


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    templates = _extract_templates(g)
    out = g.copy()
    frames = _find_rectangular_frames(g)
    assert frames, "missing colored rectangular regions"

    for r0, c0, r1, c1, color in frames:
        region_in = g[r0 + 1 : r1, c0 + 1 : c1]
        count = int(np.count_nonzero(region_in == color))
        assert 1 <= count <= len(templates), "marker count must pick an existing template"
        fill = out[r0 + 1 : r1, c0 + 1 : c1]
        fill[:, :] = BG
        tiled = _tile_mask(templates[count - 1], fill.shape)
        fill[tiled] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    return _transform(grid)


def _draw_library(grid: np.ndarray, top: int, left: int, templates: list[np.ndarray]) -> tuple[int, int, int, int]:
    width = 4 * len(templates) + 1
    bottom = top + 4
    right = left + width - 1
    grid[top, left : right + 1] = GREY
    grid[bottom, left : right + 1] = GREY
    for idx in range(len(templates) + 1):
        c = left + 4 * idx
        grid[top : bottom + 1, c] = GREY
    for idx, template in enumerate(templates):
        patch = grid[top + 1 : top + 4, left + 1 + 4 * idx : left + 4 + 4 * idx]
        patch[template] = MAROON
    return top, left, bottom, right


def _reserve(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 0) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(mask.shape[0] - 1, r1 + pad)
    cc1 = min(mask.shape[1] - 1, c1 + pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _is_free(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 0) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(mask.shape[0] - 1, r1 + pad)
    cc1 = min(mask.shape[1] - 1, c1 + pad)
    return not mask[rr0 : rr1 + 1, cc0 : cc1 + 1].any()


def _build_expected(
    grid: np.ndarray,
    region_specs: list[dict[str, object]],
    templates: list[np.ndarray],
) -> np.ndarray:
    out = grid.copy()
    for spec in region_specs:
        r0, c0, r1, c1 = spec["bbox"]
        color = int(spec["color"])
        count = int(spec["count"])
        fill = out[r0 + 1 : r1, c0 + 1 : c1]
        fill[:, :] = BG
        tiled = _tile_mask(templates[count - 1], fill.shape)
        fill[tiled] = color
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    template_count = rng.randint(3, 5)
    templates = [template.copy() for template in rng.sample(TEMPLATE_POOL, template_count)]

    region_count = rng.randint(2, min(5, len(FRAME_COLORS)))
    counts = [rng.randint(1, template_count) for _ in range(region_count)]
    if region_count > 1 and template_count > 1 and len(set(counts)) == 1:
        counts[-1] = 1 + (counts[-1] % template_count)

    lib_width = 4 * template_count + 1
    min_height = 17
    min_width = max(18, lib_width + 4)
    grid_h = rng.randint(min_height, min_height + 10)
    grid_w = rng.randint(min_width, min_width + 10)
    grid = np.zeros((grid_h, grid_w), dtype=int)

    lib_top = rng.randint(1, 2)
    lib_left = rng.randint(1, grid_w - lib_width - 1)
    lib_bbox = _draw_library(grid, lib_top, lib_left, templates)

    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    _reserve(occupied, *lib_bbox, pad=1)

    region_colors = rng.sample(FRAME_COLORS, region_count)
    region_specs: list[dict[str, object]] = []
    top_min = lib_bbox[2] + 2
    if top_min + 4 >= grid_h:
        return None

    for idx in range(region_count):
        color = region_colors[idx]
        count = counts[idx]
        placed = False
        for _ in range(200):
            inner_h = rng.randint(3, 8)
            inner_w = rng.randint(3, 8)
            if inner_h * inner_w <= count:
                continue
            if idx == 0 and inner_h % 3 == 0 and inner_w % 3 == 0:
                continue
            outer_h = inner_h + 2
            outer_w = inner_w + 2
            if top_min > grid_h - outer_h - 1 or 1 > grid_w - outer_w - 1:
                return None
            r0 = rng.randint(top_min, grid_h - outer_h - 1)
            c0 = rng.randint(1, grid_w - outer_w - 1)
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1
            if not _is_free(occupied, r0, c0, r1, c1, pad=1):
                continue

            grid[r0, c0 : c1 + 1] = color
            grid[r1, c0 : c1 + 1] = color
            grid[r0 : r1 + 1, c0] = color
            grid[r0 : r1 + 1, c1] = color

            cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
            for mr, mc in rng.sample(cells, count):
                grid[mr, mc] = color

            _reserve(occupied, r0, c0, r1, c1, pad=1)
            region_specs.append({"bbox": (r0, c0, r1, c1), "color": color, "count": count})
            placed = True
            break
        if not placed:
            return None

    expected = _build_expected(grid, region_specs, templates)
    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None
    if np.array_equal(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to create a valid sample")
