from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.reshape(-1)))
    return int(counts.most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue

            queue = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []

            while queue:
                rr, cc = queue.popleft()
                cells.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            comps.append(np.array(cells, dtype=int))

    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _mask_from_coords(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (r0, c0, r1, c1)


def _is_hollow_rectangle(mask: np.ndarray) -> bool:
    h, w = map(int, mask.shape)
    if h < 3 or w < 3:
        return False
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.array_equal(mask, border))


def _find_frames(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for color0 in np.unique(grid):
        color = int(color0)
        if color == bg:
            continue
        for coords in _components(grid == color):
            mask, box = _mask_from_coords(coords)
            if _is_hollow_rectangle(mask):
                frames.append({"color": color, "bbox": box})
    assert frames, "expected at least one hollow rectangular frame"
    return frames


def _template_map(grid: np.ndarray, bg: int, frames: list[dict[str, object]]) -> dict[int, np.ndarray]:
    outside_frame_boxes = np.ones(grid.shape, dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        outside_frame_boxes[r0 : r1 + 1, c0 : c1 + 1] = False

    templates: dict[int, np.ndarray] = {}
    template_pixels = (grid != bg) & outside_frame_boxes
    for coords in _components(template_pixels):
        mask, _box = _mask_from_coords(coords)
        area = int(mask.sum())
        assert area not in templates, "template areas must be unique"
        templates[area] = mask

    assert templates, "expected a template library outside the frames"
    return templates


def _stamp_centered(
    out: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    shape: np.ndarray,
    color: int,
    bg: int,
) -> None:
    r0, c0, r1, c1 = map(int, frame_bbox)
    inner_r0, inner_c0 = r0 + 1, c0 + 1
    inner_r1, inner_c1 = r1 - 1, c1 - 1
    out[inner_r0 : inner_r1 + 1, inner_c0 : inner_c1 + 1] = bg

    anchor_r = (inner_r0 + inner_r1) // 2
    anchor_c = (inner_c0 + inner_c1) // 2
    sh, sw = map(int, shape.shape)
    top = anchor_r - sh // 2
    left = anchor_c - sw // 2

    assert inner_r0 <= top and top + sh - 1 <= inner_r1, "template must fit vertically"
    assert inner_c0 <= left and left + sw - 1 <= inner_c1, "template must fit horizontally"

    patch = out[top : top + sh, left : left + sw]
    patch[shape] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each frame's seed pixels with the centered template whose area matches the seed count.

    Input structure:
    - Black background surrounds several disjoint 1-pixel-thick hollow rectangular frames.
    - Each frame interior contains one or more scattered seed pixels, all in one non-background color for that frame.
    - A corner library contains several separated solid template components on the background.

    Rule:
    - Detect every hollow frame and count the non-background seed pixels inside it.
    - Find the unique library template whose pixel area equals that count.
    - Clear the frame interior and redraw that template in the seed color, aligning the template bounding-box center to the frame interior center.
    - Keep the frame outlines and the template library unchanged.

    Key insight:
    - The seed pixels carry only two pieces of information: their count selects the template shape, and their color is the color used for the centered redraw.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _get_bg(g)
    frames = _find_frames(g, bg)
    templates = _template_map(g, bg, frames)

    out = g.copy()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        seed_values = interior[interior != bg]
        assert seed_values.size > 0, "each frame must contain seeds"
        seed_colors = set(map(int, seed_values.tolist()))
        assert len(seed_colors) == 1, "each frame must contain a single seed color"
        seed_color = seed_colors.pop()

        count = int(seed_values.size)
        assert count in templates, "seed count must match one template area"
        _stamp_centered(out, (r0, c0, r1, c1), templates[count], seed_color, bg)

    return out


_TEMPLATE_BANK: dict[int, np.ndarray] = {
    1: np.array([[1]], dtype=bool),
    3: np.array([[1, 1, 1]], dtype=bool),
    5: np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    6: np.array(
        [
            [1, 1, 1],
            [1, 1, 0],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    7: np.array(
        [
            [1, 1, 1],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    8: np.array(
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 0],
        ],
        dtype=bool,
    ),
    9: np.array(
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
    11: np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    13: np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
        ],
        dtype=bool,
    ),
}

_GRID_SIZE_RANGE = (24, 30)
_FRAME_COUNT_RANGE = (2, 4)
_LIBRARY_COUNT_RANGE = (3, 5)
_INNER_EXTRA_RANGE = (0, 3)
_OBJECT_GAP = 1
_EDGE_MARGIN = 1
_GEN_RETRIES = 300


def _transform_mask(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out = np.rot90(out, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    return out.copy()


def _library_layout(templates: list[tuple[int, np.ndarray]]) -> tuple[int, int, list[tuple[int, np.ndarray, int, int]]]:
    count = len(templates)
    lib_cols = 1 if count <= 3 else random.choice([1, 2])
    lib_rows = (count + lib_cols - 1) // lib_cols
    cell_h = max(int(mask.shape[0]) for _area, mask in templates)
    cell_w = max(int(mask.shape[1]) for _area, mask in templates)

    lib_h = lib_rows * cell_h + (lib_rows - 1)
    lib_w = lib_cols * cell_w + (lib_cols - 1)
    placed: list[tuple[int, np.ndarray, int, int]] = []

    for idx, (area, mask) in enumerate(templates):
        rr = idx // lib_cols
        cc = idx % lib_cols
        cell_r0 = rr * (cell_h + 1)
        cell_c0 = cc * (cell_w + 1)
        top = cell_r0 + (cell_h - int(mask.shape[0])) // 2
        left = cell_c0 + (cell_w - int(mask.shape[1])) // 2
        placed.append((area, mask, int(top), int(left)))

    return int(lib_h), int(lib_w), placed


def _corner_origin(height: int, width: int, box_h: int, box_w: int, corner: str) -> tuple[int, int]:
    if corner == "tl":
        return _EDGE_MARGIN, _EDGE_MARGIN
    if corner == "tr":
        return _EDGE_MARGIN, width - _EDGE_MARGIN - box_w
    if corner == "bl":
        return height - _EDGE_MARGIN - box_h, _EDGE_MARGIN
    if corner == "br":
        return height - _EDGE_MARGIN - box_h, width - _EDGE_MARGIN - box_w
    raise AssertionError("unknown corner")


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    h_total, w_total = map(int, occupied.shape)
    if r0 < _EDGE_MARGIN or c0 < _EDGE_MARGIN:
        return False
    if r0 + h > h_total - _EDGE_MARGIN or c0 + w > w_total - _EDGE_MARGIN:
        return False

    rr0 = max(0, r0 - _OBJECT_GAP)
    cc0 = max(0, c0 - _OBJECT_GAP)
    rr1 = min(h_total, r0 + h + _OBJECT_GAP)
    cc1 = min(w_total, c0 + w + _OBJECT_GAP)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _mark_rect(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    occupied[r0 : r0 + h, c0 : c0 + w] = True


def _choose_interior_size(shape: np.ndarray, area: int) -> tuple[int, int]:
    sh, sw = map(int, shape.shape)
    for _ in range(50):
        extra_h = random.randint(*_INNER_EXTRA_RANGE)
        extra_w = random.randint(*_INNER_EXTRA_RANGE)
        if sh + extra_h <= 0 or sw + extra_w <= 0:
            continue
        if (sh + extra_h) * (sw + extra_w) > area:
            return sh + extra_h, sw + extra_w
    return sh + 1, sw


def _pick_seed_positions(
    inner_h: int,
    inner_w: int,
    count: int,
    avoid_exact: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    assert len(cells) > count
    for _ in range(60):
        picked = random.sample(cells, count)
        if set(picked) != avoid_exact:
            return picked
    raise AssertionError("failed to sample non-degenerate seed positions")


def _frame_target_cells(inner_h: int, inner_w: int, shape: np.ndarray) -> set[tuple[int, int]]:
    anchor_r = (inner_h - 1) // 2
    anchor_c = (inner_w - 1) // 2
    sh, sw = map(int, shape.shape)
    top = anchor_r - sh // 2
    left = anchor_c - sw // 2
    cells = set()
    for r, c in np.argwhere(shape):
        cells.add((int(top + r), int(left + c)))
    return cells


def _render_pair(
    height: int,
    width: int,
    frame_color: int,
    template_color: int,
    library_origin: tuple[int, int],
    library_shapes: list[tuple[int, np.ndarray, int, int]],
    frame_specs: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.zeros((height, width), dtype=int)
    out = np.zeros((height, width), dtype=int)

    lib_r0, lib_c0 = library_origin
    for _area, mask, dr, dc in library_shapes:
        sh, sw = map(int, mask.shape)
        patch_in = inp[lib_r0 + dr : lib_r0 + dr + sh, lib_c0 + dc : lib_c0 + dc + sw]
        patch_out = out[lib_r0 + dr : lib_r0 + dr + sh, lib_c0 + dc : lib_c0 + dc + sw]
        patch_in[mask] = template_color
        patch_out[mask] = template_color

    for spec in frame_specs:
        r0, c0 = spec["origin"]
        inner_h = int(spec["inner_h"])
        inner_w = int(spec["inner_w"])
        seed_color = int(spec["seed_color"])
        shape = spec["shape"]
        seeds = spec["seeds"]
        area = int(spec["area"])

        r1 = r0 + inner_h + 1
        c1 = c0 + inner_w + 1
        inp[r0, c0 : c1 + 1] = frame_color
        inp[r1, c0 : c1 + 1] = frame_color
        inp[r0 : r1 + 1, c0] = frame_color
        inp[r0 : r1 + 1, c1] = frame_color

        out[r0, c0 : c1 + 1] = frame_color
        out[r1, c0 : c1 + 1] = frame_color
        out[r0 : r1 + 1, c0] = frame_color
        out[r0 : r1 + 1, c1] = frame_color

        for sr, sc in seeds:
            inp[r0 + 1 + sr, c0 + 1 + sc] = seed_color

        assert len(seeds) == area
        _stamp_centered(out, (r0, c0, r1, c1), shape, seed_color, 0)

    return inp, out


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(*_GRID_SIZE_RANGE)
    width = random.randint(*_GRID_SIZE_RANGE)
    if min(height, width) < 10:
        return None

    frame_count = random.randint(*_FRAME_COUNT_RANGE)
    library_count = random.randint(max(_LIBRARY_COUNT_RANGE[0], 2), _LIBRARY_COUNT_RANGE[1])
    if library_count > len(_TEMPLATE_BANK):
        return None

    available_areas = random.sample(list(_TEMPLATE_BANK), library_count)
    library_templates = [(area, _transform_mask(_TEMPLATE_BANK[area])) for area in available_areas]
    random.shuffle(library_templates)
    lib_h, lib_w, laid_out_library = _library_layout(library_templates)
    if lib_h > height - 2 * _EDGE_MARGIN or lib_w > width - 2 * _EDGE_MARGIN:
        return None

    colors = list(range(1, 10))
    random.shuffle(colors)
    frame_color = colors.pop()
    template_color = colors.pop()
    seed_colors = random.sample(colors, frame_count)

    chosen_areas = [random.choice(available_areas) for _ in range(frame_count)]
    if frame_count >= 2 and len(set(chosen_areas)) == 1 and len(available_areas) > 1:
        chosen_areas[0], chosen_areas[1] = random.sample(available_areas, 2)

    frame_specs: list[dict[str, object]] = []
    for area, seed_color in zip(chosen_areas, seed_colors):
        shape = dict(library_templates)[area]
        inner_h, inner_w = _choose_interior_size(shape, area)
        target_cells = _frame_target_cells(inner_h, inner_w, shape)
        seeds = _pick_seed_positions(inner_h, inner_w, area, target_cells)
        frame_specs.append(
            {
                "area": area,
                "shape": shape,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "seed_color": seed_color,
                "seeds": seeds,
            }
        )

    frame_specs.sort(
        key=lambda spec: (int(spec["inner_h"]) + 2) * (int(spec["inner_w"]) + 2),
        reverse=True,
    )

    occupied = np.zeros((height, width), dtype=bool)
    corner = random.choice(["tl", "tr", "bl", "br"])
    lib_origin = _corner_origin(height, width, lib_h, lib_w, corner)
    if not _can_place(occupied, lib_origin[0], lib_origin[1], lib_h, lib_w):
        return None
    _mark_rect(occupied, lib_origin[0], lib_origin[1], lib_h, lib_w)

    for spec in frame_specs:
        box_h = int(spec["inner_h"]) + 2
        box_w = int(spec["inner_w"]) + 2
        placed = False
        for _ in range(200):
            r0 = random.randint(_EDGE_MARGIN, height - _EDGE_MARGIN - box_h)
            c0 = random.randint(_EDGE_MARGIN, width - _EDGE_MARGIN - box_w)
            if _can_place(occupied, r0, c0, box_h, box_w):
                spec["origin"] = (r0, c0)
                _mark_rect(occupied, r0, c0, box_h, box_w)
                placed = True
                break
        if not placed:
            return None

    inp, expected = _render_pair(
        height,
        width,
        frame_color,
        template_color,
        lib_origin,
        laid_out_library,
        frame_specs,
    )

    if np.array_equal(main(inp), expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
