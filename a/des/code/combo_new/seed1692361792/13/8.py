import random

import numpy as np


BG = 0
RED = 2
GREY = 5
PATTERN_COLORS = (1, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 300


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for row in range(height):
        for col in range(width):
            if not bool(mask[row, col]) or bool(seen[row, col]):
                continue
            stack = [(row, col)]
            seen[row, col] = True
            comp: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                comp.append((int(r), int(c)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = int(r + dr)
                    nc = int(c + dc)
                    if 0 <= nr < height and 0 <= nc < width and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _parse_library(grid: np.ndarray, points: list[tuple[int, int]]) -> dict | None:
    r0, c0, r1, c1 = _bbox(points)
    local = grid[r0 : r1 + 1, c0 : c1 + 1] == GREY
    height, width = local.shape
    if height < 3 or width < 5:
        return None
    if not bool(local[0].all()) or not bool(local[-1].all()):
        return None

    full_cols = [col for col in range(width) if bool(local[:, col].all())]
    if len(full_cols) < 3:
        return None

    gaps = [full_cols[i + 1] - full_cols[i] for i in range(len(full_cols) - 1)]
    if len(set(gaps)) != 1 or gaps[0] < 2:
        return None

    allowed = np.zeros_like(local, dtype=bool)
    allowed[0, :] = True
    allowed[-1, :] = True
    allowed[:, full_cols] = True
    if not np.array_equal(local, allowed):
        return None

    cells = []
    for left, right in zip(full_cols, full_cols[1:], strict=False):
        cell = grid[r0 + 1 : r1, c0 + left + 1 : c0 + right].copy()
        if cell.size == 0:
            return None
        cells.append(cell)
    return {"bbox": (r0, c0, r1, c1), "cells": cells}


def _find_library(grid: np.ndarray) -> dict:
    candidates = []
    for comp in _components(grid == GREY):
        parsed = _parse_library(grid, comp)
        if parsed is not None:
            candidates.append(parsed)
    assert candidates, "expected one pattern library"
    return max(
        candidates,
        key=lambda item: (
            len(item["cells"]),
            item["bbox"][2] - item["bbox"][0],
            item["bbox"][3] - item["bbox"][1],
        ),
    )


def _is_target_rectangle(grid: np.ndarray, points: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(points)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    local = grid[r0 : r1 + 1, c0 : c1 + 1] == GREY
    border = np.zeros_like(local, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.array_equal(local, border))


def _resize_nearest(pattern: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    src_h, src_w = pattern.shape
    row_idx = np.minimum(src_h - 1, (np.arange(out_h) * src_h) // out_h)
    col_idx = np.minimum(src_w - 1, (np.arange(out_w) * src_w) // out_w)
    return pattern[row_idx[:, None], col_idx[None, :]]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the library pattern indexed by each rectangle's red markers and scale it into the rectangle interior.

    Input structure:
    - The background is black.
    - Several one-pixel-thick grey rectangular frames appear on the grid, and a
      separate grey-framed library region sits in one corner.
    - The library is split by thin grey vertical separators into equal cells laid
      out from left to right; each cell contains a small colored pattern.
    - Every non-library rectangle contains 1 to 4 red marker pixels inside its border.

    Rule:
    - Locate the library and read its cells from left to right as pattern 1,
      pattern 2, pattern 3, and so on.
    - For each other grey rectangle, count the red markers inside its border.
    - Take the corresponding library pattern and resize it with nearest-neighbor
      scaling so it exactly matches that rectangle's interior size.
    - Replace the rectangle interior with the scaled pattern, covering the red markers.

    Key insight:
    - The red pixels are only an index into the corner library; the visible output
      comes from copying and scaling the referenced stored pattern, not from
      preserving the marker pixels themselves.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    library = _find_library(g)
    patterns = library["cells"]
    out = g.copy()

    for comp in _components(g == GREY):
        if not _is_target_rectangle(g, comp):
            continue
        rect_bbox = _bbox(comp)
        if rect_bbox == library["bbox"]:
            continue
        r0, c0, r1, c1 = rect_bbox
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        marker_count = int(np.count_nonzero(interior == RED))
        if marker_count == 0:
            continue
        assert 1 <= marker_count <= len(patterns), "marker count must select a library cell"
        out[r0 + 1 : r1, c0 + 1 : c1] = _resize_nearest(patterns[marker_count - 1], r1 - r0 - 1, c1 - c0 - 1)

    return out


def _mask_plus(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[height // 2, :] = True
    mask[:, width // 2] = True
    return mask


def _mask_x(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    denom = max(1, height - 1)
    for row in range(height):
        col_a = int(round(row * (width - 1) / denom))
        col_b = int(round((height - 1 - row) * (width - 1) / denom))
        mask[row, col_a] = True
        mask[row, col_b] = True
    return mask


def _mask_hstripe(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[::2, :] = True
    return mask


def _mask_vstripe(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[:, ::2] = True
    return mask


def _mask_ring(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _mask_checker(height: int, width: int) -> np.ndarray:
    rows, cols = np.indices((height, width))
    return (rows + cols) % 2 == 0


PATTERN_BUILDERS = (
    _mask_plus,
    _mask_x,
    _mask_hstripe,
    _mask_vstripe,
    _mask_ring,
    _mask_checker,
)


def _build_patterns() -> list[np.ndarray]:
    cell_h = random.randint(3, 4)
    cell_w = random.randint(3, 5)
    builders = random.sample(PATTERN_BUILDERS, 4)
    colors = random.sample(list(PATTERN_COLORS), 4)
    patterns = []
    for builder, color in zip(builders, colors, strict=False):
        cell = np.full((cell_h, cell_w), BG, dtype=int)
        cell[builder(cell_h, cell_w)] = int(color)
        patterns.append(cell)
    return patterns


def _build_library_patch(patterns: list[np.ndarray]) -> np.ndarray:
    cell_h, cell_w = patterns[0].shape
    count = len(patterns)
    width = count * (cell_w + 1) + 1
    patch = np.full((cell_h + 2, width), BG, dtype=int)
    patch[0, :] = GREY
    patch[-1, :] = GREY
    for idx in range(count + 1):
        col = idx * (cell_w + 1)
        patch[:, col] = GREY
    for idx, pattern in enumerate(patterns):
        col = idx * (cell_w + 1) + 1
        patch[1 : 1 + cell_h, col : col + cell_w] = pattern
    return patch


def _corner_origin(height: int, width: int, patch_h: int, patch_w: int, corner: str) -> tuple[int, int]:
    row_margin = random.randint(0, 1)
    col_margin = random.randint(0, 1)
    if "top" in corner:
        row = row_margin
    else:
        row = height - patch_h - row_margin
    if "left" in corner:
        col = col_margin
    else:
        col = width - patch_w - col_margin
    return int(row), int(col)


def _reserve(occupied: np.ndarray, row: int, col: int, height: int, width: int, pad: int = 1) -> None:
    h, w = occupied.shape
    r0 = max(0, row - pad)
    c0 = max(0, col - pad)
    r1 = min(h, row + height + pad)
    c1 = min(w, col + width + pad)
    occupied[r0:r1, c0:c1] = True


def _find_position(occupied: np.ndarray, box_h: int, box_w: int) -> tuple[int, int] | None:
    height, width = occupied.shape
    candidates = [
        (row, col)
        for row in range(1, height - box_h)
        for col in range(1, width - box_w)
        if not bool(occupied[row - 1 : row + box_h + 1, col - 1 : col + box_w + 1].any())
    ]
    if not candidates:
        return None
    return random.choice(candidates)


def _rectangle_specs(pattern_h: int, pattern_w: int, count: int) -> list[tuple[int, int]]:
    specs = [
        (pattern_h, pattern_w),
        (min(7, pattern_h + random.randint(1, 3)), min(7, pattern_w + random.randint(1, 3))),
        (min(7, pattern_h + random.randint(0, 2)), min(7, pattern_w + random.randint(2, 3))),
        (min(7, pattern_h + random.randint(2, 3)), min(7, pattern_w + random.randint(0, 2))),
    ]
    while len(specs) < count:
        specs.append((random.randint(3, 7), random.randint(3, 7)))
    random.shuffle(specs)
    return specs[:count]


def _draw_library(grid: np.ndarray, patch: np.ndarray, row: int, col: int) -> None:
    grid[row : row + patch.shape[0], col : col + patch.shape[1]] = patch


def _draw_frame(grid: np.ndarray, row: int, col: int, inner_h: int, inner_w: int) -> None:
    grid[row, col : col + inner_w + 2] = GREY
    grid[row + inner_h + 1, col : col + inner_w + 2] = GREY
    grid[row : row + inner_h + 2, col] = GREY
    grid[row : row + inner_h + 2, col + inner_w + 1] = GREY


def _scatter_markers(grid: np.ndarray, row: int, col: int, inner_h: int, inner_w: int, count: int) -> None:
    cells = [(r, c) for r in range(row + 1, row + 1 + inner_h) for c in range(col + 1, col + 1 + inner_w)]
    for r, c in random.sample(cells, count):
        grid[r, c] = RED


def _paint_scaled_pattern(grid: np.ndarray, row: int, col: int, inner_h: int, inner_w: int, pattern: np.ndarray) -> None:
    grid[row + 1 : row + 1 + inner_h, col + 1 : col + 1 + inner_w] = _resize_nearest(pattern, inner_h, inner_w)


def _gen_attempt() -> np.ndarray | None:
    patterns = _build_patterns()
    library_patch = _build_library_patch(patterns)
    lib_h, lib_w = library_patch.shape
    height = random.randint(max(20, lib_h + 10), 30)
    width = random.randint(max(22, lib_w + 2), 30)

    inp = np.full((height, width), BG, dtype=int)
    out = inp.copy()
    occupied = np.zeros((height, width), dtype=bool)

    corner = random.choice(("top_left", "top_right", "bottom_left", "bottom_right"))
    lib_row, lib_col = _corner_origin(height, width, lib_h, lib_w, corner)
    _draw_library(inp, library_patch, lib_row, lib_col)
    _draw_library(out, library_patch, lib_row, lib_col)
    _reserve(occupied, lib_row, lib_col, lib_h, lib_w)

    marker_counts = [1, 2, 3, 4]
    if random.random() < 0.5:
        marker_counts.append(random.randint(1, 4))
    inner_sizes = _rectangle_specs(patterns[0].shape[0], patterns[0].shape[1], len(marker_counts))
    rects = [
        {"markers": markers, "inner_h": inner_h, "inner_w": inner_w}
        for markers, (inner_h, inner_w) in zip(marker_counts, inner_sizes, strict=False)
    ]
    order = sorted(range(len(rects)), key=lambda idx: rects[idx]["inner_h"] * rects[idx]["inner_w"], reverse=True)

    placements: dict[int, tuple[int, int]] = {}
    for idx in order:
        box_h = rects[idx]["inner_h"] + 2
        box_w = rects[idx]["inner_w"] + 2
        pos = _find_position(occupied, box_h, box_w)
        if pos is None:
            return None
        placements[idx] = pos
        _reserve(occupied, pos[0], pos[1], box_h, box_w)

    for idx, spec in enumerate(rects):
        row, col = placements[idx]
        inner_h = int(spec["inner_h"])
        inner_w = int(spec["inner_w"])
        markers = int(spec["markers"])
        _draw_frame(inp, row, col, inner_h, inner_w)
        _draw_frame(out, row, col, inner_h, inner_w)
        _scatter_markers(inp, row, col, inner_h, inner_w, markers)
        _paint_scaled_pattern(out, row, col, inner_h, inner_w, patterns[markers - 1])

    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
