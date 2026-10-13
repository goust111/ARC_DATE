import math
import random
from typing import Iterable

import numpy as np


BG = 0
GREY = 5
FRAME_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

# Generator configuration.
GEN_RETRIES = 400
GRID_MIN = 20
GRID_MAX = 30
FRAME_COUNT_RANGE = (2, 4)
USED_TEMPLATE_COUNT_RANGE = (2, 4)
EXTRA_TEMPLATE_COUNT_RANGE = (0, 2)
CELL_DIMS = (1, 2, 3, 4)
ROOM_COUNT_CHOICES = (2, 3)
DICT_GAP = 1
PLACEMENT_PAD = 1
PLACEMENT_TRIES = 200


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
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
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(coords)
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _frame_spec_from_component(
    grid: np.ndarray, color: int, coords: list[tuple[int, int]]
) -> dict[str, object] | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    local = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        local[r - r0, c - c0] = True

    full_rows = [r for r in range(h) if bool(np.all(local[r, :]))]
    full_cols = [c for c in range(w) if bool(np.all(local[:, c]))]
    if len(full_rows) < 2 or len(full_cols) < 2:
        return None
    if full_rows[0] != 0 or full_rows[-1] != h - 1:
        return None
    if full_cols[0] != 0 or full_cols[-1] != w - 1:
        return None

    row_steps = [b - a for a, b in zip(full_rows, full_rows[1:])]
    col_steps = [b - a for a, b in zip(full_cols, full_cols[1:])]
    if not row_steps or not col_steps:
        return None
    if len(set(row_steps)) != 1 or len(set(col_steps)) != 1:
        return None

    cell_h = row_steps[0] - 1
    cell_w = col_steps[0] - 1
    if cell_h <= 0 or cell_w <= 0:
        return None

    expected = np.zeros_like(local)
    expected[full_rows, :] = True
    expected[:, full_cols] = True
    if not np.array_equal(local, expected):
        return None

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    if np.any((patch != BG) & (patch != color)):
        return None

    return {
        "color": color,
        "bbox": (r0, c0, r1, c1),
        "full_rows": full_rows,
        "full_cols": full_cols,
        "cell_shape": (cell_h, cell_w),
    }


def _extract_templates(grid: np.ndarray) -> tuple[dict[tuple[int, int], np.ndarray], np.ndarray]:
    grey_components = _components(grid == GREY)
    templates: dict[tuple[int, int], np.ndarray] = {}
    dict_mask = np.zeros_like(grid, dtype=bool)
    total_grey = int(np.count_nonzero(grid == GREY))
    seen_grey = 0

    for coords in grey_components:
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if int(np.count_nonzero(patch == GREY)) != int(np.count_nonzero(border)):
            continue
        if not np.all(patch[border] == GREY):
            continue
        if np.any(patch[1:-1, 1:-1] == GREY):
            continue
        size = (h - 2, w - 2)
        if size in templates:
            raise AssertionError("template sizes must be unique")
        templates[size] = patch[1:-1, 1:-1].copy()
        dict_mask[r0 : r1 + 1, c0 : c1 + 1] = True
        seen_grey += int(np.count_nonzero(border))

    assert templates, "expected at least one grey template block"
    assert seen_grey == total_grey, "all grey pixels must belong to template borders"
    return templates, dict_mask


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    templates, dict_mask = _extract_templates(g)
    out = np.zeros_like(g)
    frames: list[dict[str, object]] = []

    for color in np.unique(g):
        color = int(color)
        if color in (BG, GREY):
            continue
        mask = (g == color) & ~dict_mask
        for coords in _components(mask):
            spec = _frame_spec_from_component(g, color, coords)
            assert spec is not None, "non-template colored components must be valid partitioned frames"
            frames.append(spec)

    assert frames, "expected at least one frame"

    for spec in frames:
        color = int(spec["color"])
        r0, c0, r1, c1 = spec["bbox"]  # type: ignore[misc]
        full_rows = spec["full_rows"]  # type: ignore[assignment]
        full_cols = spec["full_cols"]  # type: ignore[assignment]
        cell_shape = spec["cell_shape"]  # type: ignore[assignment]
        assert cell_shape in templates, "missing template for frame cell size"
        pattern = templates[cell_shape]
        mask = pattern != BG

        local = out[r0 : r1 + 1, c0 : c1 + 1]
        local[full_rows, :] = color
        local[:, full_cols] = color

        row_spans = [(full_rows[i] + 1, full_rows[i + 1] - 1) for i in range(len(full_rows) - 1)]
        col_spans = [(full_cols[i] + 1, full_cols[i + 1] - 1) for i in range(len(full_cols) - 1)]
        for rr0, rr1 in row_spans:
            for cc0, cc1 in col_spans:
                cell = local[rr0 : rr1 + 1, cc0 : cc1 + 1]
                assert cell.shape == pattern.shape
                cell[mask] = color

    return out


def _frame_patch(color: int, cell_h: int, cell_w: int, room_rows: int, room_cols: int) -> np.ndarray:
    h = room_rows * cell_h + room_rows + 1
    w = room_cols * cell_w + room_cols + 1
    patch = np.zeros((h, w), dtype=int)

    row_lines = [i * (cell_h + 1) for i in range(room_rows + 1)]
    col_lines = [i * (cell_w + 1) for i in range(room_cols + 1)]
    patch[row_lines, :] = color
    patch[:, col_lines] = color
    return patch


def _pattern_mask(h: int, w: int) -> np.ndarray:
    rows, cols = np.indices((h, w))
    candidates = [
        (rows + cols) % 2 == random.randint(0, 1),
        rows % 2 == random.randint(0, 1),
        cols % 2 == random.randint(0, 1),
        (rows == cols) | (rows + cols == w - 1),
        (rows < (h + 1) // 2) ^ (cols < (w + 1) // 2),
        rows <= cols * max(1, h) // max(1, w),
        np.random.random((h, w)) < random.uniform(0.35, 0.65),
    ]
    random.shuffle(candidates)
    for cand in candidates:
        mask = np.asarray(cand, dtype=bool)
        if 0 < int(mask.sum()) < h * w:
            return mask
    mask = np.zeros((h, w), dtype=bool)
    mask.flat[0] = True
    return mask


def _template_patch(size: tuple[int, int], color: int) -> np.ndarray:
    h, w = size
    patch = np.zeros((h + 2, w + 2), dtype=int)
    patch[0, :] = GREY
    patch[-1, :] = GREY
    patch[:, 0] = GREY
    patch[:, -1] = GREY
    patch[1:-1, 1:-1][_pattern_mask(h, w)] = color
    return patch


def _compose_dictionary(blocks: list[np.ndarray], orientation: str) -> np.ndarray:
    if orientation == "horizontal":
        height = max(block.shape[0] for block in blocks)
        width = sum(block.shape[1] for block in blocks) + DICT_GAP * (len(blocks) - 1)
        patch = np.zeros((height, width), dtype=int)
        c = 0
        for block in blocks:
            r = random.randint(0, height - block.shape[0])
            patch[r : r + block.shape[0], c : c + block.shape[1]] = block
            c += block.shape[1] + DICT_GAP
        return patch

    width = max(block.shape[1] for block in blocks)
    height = sum(block.shape[0] for block in blocks) + DICT_GAP * (len(blocks) - 1)
    patch = np.zeros((height, width), dtype=int)
    r = 0
    for block in blocks:
        c = random.randint(0, width - block.shape[1])
        patch[r : r + block.shape[0], c : c + block.shape[1]] = block
        r += block.shape[0] + DICT_GAP
    return patch


def _reserve(occ: np.ndarray, r: int, c: int, h: int, w: int, pad: int = PLACEMENT_PAD) -> None:
    rr0 = max(0, r - pad)
    cc0 = max(0, c - pad)
    rr1 = min(occ.shape[0], r + h + pad)
    cc1 = min(occ.shape[1], c + w + pad)
    occ[rr0:rr1, cc0:cc1] = True


def _fits(occ: np.ndarray, r: int, c: int, h: int, w: int, pad: int = PLACEMENT_PAD) -> bool:
    rr0 = max(0, r - pad)
    cc0 = max(0, c - pad)
    rr1 = min(occ.shape[0], r + h + pad)
    cc1 = min(occ.shape[1], c + w + pad)
    return not bool(np.any(occ[rr0:rr1, cc0:cc1]))


def _paste(grid: np.ndarray, patch: np.ndarray, r: int, c: int) -> None:
    h, w = patch.shape
    grid[r : r + h, c : c + w] = patch


def _place_randomly(
    grid: np.ndarray, occ: np.ndarray, patch: np.ndarray, fixed_side: tuple[int, int] | None = None
) -> bool:
    h, w = patch.shape
    gh, gw = grid.shape
    if h > gh - 2 or w > gw - 2:
        return False
    if fixed_side is not None:
        r, c = fixed_side
        if not _fits(occ, r, c, h, w):
            return False
        _paste(grid, patch, r, c)
        _reserve(occ, r, c, h, w)
        return True

    for _ in range(PLACEMENT_TRIES):
        r = random.randint(1, gh - h - 1)
        c = random.randint(1, gw - w - 1)
        if not _fits(occ, r, c, h, w):
            continue
        _paste(grid, patch, r, c)
        _reserve(occ, r, c, h, w)
        return True
    return False


def _choose_template_sizes() -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    sizes = [(h, w) for h in CELL_DIMS for w in CELL_DIMS if h * w >= 2]
    random.shuffle(sizes)
    used_count = random.randint(*USED_TEMPLATE_COUNT_RANGE)
    used = sizes[:used_count]
    rest = sizes[used_count:]
    max_extra = min(EXTRA_TEMPLATE_COUNT_RANGE[1], len(rest))
    extra_count = random.randint(EXTRA_TEMPLATE_COUNT_RANGE[0], max_extra)
    extra = rest[:extra_count]
    return used, extra


def _sample_frame_specs(used_sizes: list[tuple[int, int]]) -> list[dict[str, object]]:
    frame_count = random.randint(*FRAME_COUNT_RANGE)
    colors = list(FRAME_COLORS)
    random.shuffle(colors)
    specs: list[dict[str, object]] = []
    modes = ["horizontal", "vertical", "both"]
    for idx in range(frame_count):
        cell_h, cell_w = random.choice(used_sizes)
        mode = random.choice(modes)
        if mode == "horizontal":
            room_rows = random.choice(ROOM_COUNT_CHOICES)
            room_cols = 1
        elif mode == "vertical":
            room_rows = 1
            room_cols = random.choice(ROOM_COUNT_CHOICES)
        else:
            room_rows = random.choice(ROOM_COUNT_CHOICES)
            room_cols = random.choice(ROOM_COUNT_CHOICES)
        color = colors[idx]
        patch = _frame_patch(color, cell_h, cell_w, room_rows, room_cols)
        specs.append(
            {
                "color": color,
                "cell_shape": (cell_h, cell_w),
                "mode": mode,
                "room_rows": room_rows,
                "room_cols": room_cols,
                "patch": patch,
            }
        )
    return specs


def _build_grid() -> np.ndarray | None:
    used_sizes, extra_sizes = _choose_template_sizes()
    frame_specs = _sample_frame_specs(used_sizes)
    template_sizes = used_sizes + extra_sizes

    pattern_colors = [c for c in FRAME_COLORS if c != GREY]
    blocks = [_template_patch(size, random.choice(pattern_colors)) for size in template_sizes]
    random.shuffle(blocks)
    dict_orientation = random.choice(("horizontal", "vertical"))
    dict_patch = _compose_dictionary(blocks, dict_orientation)

    frame_patches = [spec["patch"] for spec in frame_specs]
    max_frame_h = max(p.shape[0] for p in frame_patches)
    max_frame_w = max(p.shape[1] for p in frame_patches)
    total_area = dict_patch.size + sum(p.size for p in frame_patches)
    base = max(GRID_MIN, math.ceil(math.sqrt(total_area * 2.5)))

    side = random.choice(("top", "bottom", "left", "right"))
    if side in ("top", "bottom"):
        min_h = max(base, dict_patch.shape[0] + max_frame_h + 4)
        min_w = max(base, dict_patch.shape[1] + 2, max_frame_w + 2)
    else:
        min_h = max(base, dict_patch.shape[0] + 2, max_frame_h + 2)
        min_w = max(base, dict_patch.shape[1] + max_frame_w + 4)

    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None

    h = random.randint(min_h, GRID_MAX)
    w = random.randint(min_w, GRID_MAX)
    grid = np.zeros((h, w), dtype=int)
    occ = np.zeros((h, w), dtype=bool)

    if side == "top":
        dict_r = 1
        dict_c = random.randint(1, w - dict_patch.shape[1] - 1)
    elif side == "bottom":
        dict_r = h - dict_patch.shape[0] - 1
        dict_c = random.randint(1, w - dict_patch.shape[1] - 1)
    elif side == "left":
        dict_r = random.randint(1, h - dict_patch.shape[0] - 1)
        dict_c = 1
    else:
        dict_r = random.randint(1, h - dict_patch.shape[0] - 1)
        dict_c = w - dict_patch.shape[1] - 1

    if not _place_randomly(grid, occ, dict_patch, (dict_r, dict_c)):
        return None

    for patch in sorted(frame_patches, key=lambda arr: arr.size, reverse=True):
        if not _place_randomly(grid, occ, patch):
            return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if np.any(out == GREY):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_grid()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
