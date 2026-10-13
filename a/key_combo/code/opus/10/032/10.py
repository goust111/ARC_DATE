import random
from collections import deque

import numpy as np


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components of a boolean mask as coord arrays."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _parse_legend(grid: np.ndarray) -> dict[int, tuple[int, int]]:
    h, _ = grid.shape
    legend: dict[int, tuple[int, int]] = {}
    used = np.zeros(h, dtype=bool)
    r = 0
    while r < h - 1:
        block = grid[r : r + 2, :2]
        left = block[:, 0]
        right = block[:, 1]
        if (
            np.all(left != 0)
            and np.all(right != 0)
            and left[0] == left[1]
            and right[0] != right[1]
        ):
            key = int(left[0])
            assert key not in legend, "legend keys must be unique"
            legend[key] = (int(right[0]), int(right[1]))
            used[r : r + 2] = True
            r += 2
            continue
        r += 1

    assert len(legend) == 2, "expected exactly two legend entries"
    legend_noise = grid[~used, :2]
    assert not np.any(legend_noise), "legend columns must contain only the two 2x2 entries"
    return legend


def _main_bbox(grid: np.ndarray, keys: set[int]) -> tuple[int, int, int, int]:
    mask = np.isin(grid, list(keys))
    mask[:, :2] = False
    coords = np.argwhere(mask)
    assert coords.size > 0, "main mask must contain legend key colors"
    return _bbox(coords)


def _checkerboard_fill(mask_grid: np.ndarray, mapping: dict[int, tuple[int, int]], offset_parity: int = 0) -> np.ndarray:
    h, w = mask_grid.shape
    rows, cols = np.indices((h, w))
    even = ((rows + cols + offset_parity) % 2) == 0
    out = np.zeros((h, w), dtype=int)
    for key, (c1, c2) in mapping.items():
        region = mask_grid == key
        out[region & even] = c1
        out[region & ~even] = c2
    return out


def _component_anchored_fill(mask_grid: np.ndarray, mapping: dict[int, tuple[int, int]]) -> np.ndarray:
    out = np.zeros_like(mask_grid)
    for key, (c1, c2) in mapping.items():
        for comp in _components(mask_grid == key):
            r0, c0, _, _ = _bbox(comp)
            for r, c in comp:
                out[r, c] = c1 if ((int(r) - r0 + int(c) - c0) % 2 == 0) else c2
    return out


def _flat_recolor(mask_grid: np.ndarray, mapping: dict[int, tuple[int, int]]) -> np.ndarray:
    out = np.zeros_like(mask_grid)
    for key, (c1, _c2) in mapping.items():
        out[mask_grid == key] = c1
    return out


def _extract_main_region(grid: np.ndarray) -> tuple[np.ndarray, dict[int, tuple[int, int]], tuple[int, int, int, int]]:
    legend = _parse_legend(grid)
    keys = set(legend)
    bbox = _main_bbox(grid, keys)
    r0, c0, r1, c1 = bbox
    main_region = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
    assert np.all(np.isin(main_region, [0, *keys])), "main region may only contain background and legend keys"
    assert np.any(main_region[0] != 0) and np.any(main_region[-1] != 0), "main region must touch top and bottom"
    assert np.any(main_region[:, 0] != 0) and np.any(main_region[:, -1] != 0), "main region must touch left and right"
    return main_region, legend, bbox


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode left-side legend entries into two checkerboard palettes and paint each mask color with its own anchored checkerboard.

    Input structure:
    - The leftmost two columns contain exactly two stacked 2x2 legend entries on a black background.
    - In each legend entry, the left column is a repeated non-black key color and the right column holds two different non-black palette colors.
    - Away from the legend, the main rectangular region contains only black plus the two legend key colors as a binary mask.

    Rule:
    - Read each legend entry as key K -> (C1, C2), where C1 is the upper right color and C2 is the lower right color.
    - Crop the right-side main region and, for every cell whose mask color is K, paint a checkerboard anchored at the main region's top-left: even (r+c) gets C1 and odd gets C2.
    - Keep black background cells black and discard the legend from the output.

    Key insight:
    - The legend does not map one color to one replacement color; it maps each mask color to a two-color checkerboard phase, and the shared anchor is the cropped main region rather than each component.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    main_region, legend, _bbox_main = _extract_main_region(g)
    return _checkerboard_fill(main_region, legend)


_BG = 0
_GEN_RETRIES = 500
_MIN_SIDE = 8
_MAX_SIDE = 15


def _pick_parity_compatible(start: int, end: int, required_parity: int) -> int | None:
    choices = [v for v in range(start, end + 1) if v % 2 == required_parity]
    return random.choice(choices) if choices else None


def _touches_other_color(main: np.ndarray, cells: np.ndarray, color: int) -> bool:
    h, w = main.shape
    for r, c in cells:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = int(r) + dr, int(c) + dc
            if 0 <= nr < h and 0 <= nc < w and main[nr, nc] not in (_BG, color):
                return True
    return False


def _try_add_patch(main: np.ndarray, color: int) -> bool:
    h, w = main.shape
    patch_h = random.randint(1, min(3, h - 1))
    patch_w = random.randint(1, min(3, w - 1))
    r0 = random.randint(0, h - patch_h)
    c0 = random.randint(0, w - patch_w)
    cells = np.array([(r, c) for r in range(r0, r0 + patch_h) for c in range(c0, c0 + patch_w)], dtype=int)
    if np.any(main[cells[:, 0], cells[:, 1]] != _BG):
        return False
    if _touches_other_color(main, cells, color):
        return False
    main[cells[:, 0], cells[:, 1]] = color
    return True


def _build_main_mask(h: int, w: int, key_a: int, key_b: int) -> np.ndarray | None:
    main = np.zeros((h, w), dtype=int)
    vert_w = random.randint(1, 2)
    hori_h = random.randint(1, 2)

    gap_row = _pick_parity_compatible(1, h - hori_h - 1, 1 - (hori_h % 2))
    gap_col = _pick_parity_compatible(1, w - vert_w - 1, 1 - (vert_w % 2))
    if gap_row is None or gap_col is None:
        return None

    vertical_color, horizontal_color = random.sample([key_a, key_b], 2)
    main[:gap_row, gap_col : gap_col + vert_w] = vertical_color
    main[gap_row + hori_h :, gap_col : gap_col + vert_w] = vertical_color
    main[gap_row : gap_row + hori_h, :gap_col] = horizontal_color
    main[gap_row : gap_row + hori_h, gap_col + vert_w :] = horizontal_color

    for _ in range(random.randint(1, 4)):
        _try_add_patch(main, random.choice([key_a, key_b]))

    return main


def _place_legend(grid: np.ndarray, entries: list[tuple[int, tuple[int, int]]]) -> None:
    h, _ = grid.shape
    top = random.randint(0, h - 5)
    row = top
    for i, (key, (c1, c2)) in enumerate(entries):
        grid[row : row + 2, 0] = key
        grid[row, 1] = c1
        grid[row + 1, 1] = c2
        row += 3 if i == 0 else 2


def _semantic_checks(inp: np.ndarray) -> bool:
    try:
        main_region, legend, bbox_main = _extract_main_region(inp)
        expected = main(inp)
    except AssertionError:
        return False

    keys = list(legend)
    if any(np.count_nonzero(main_region == key) < 4 for key in keys):
        return False
    if not np.any(main_region == 0):
        return False

    rows, cols = np.indices(main_region.shape)
    parity = (rows + cols) % 2
    for key in keys:
        region = main_region == key
        if not np.any(region & (parity == 0)) or not np.any(region & (parity == 1)):
            return False
        comps = _components(region)
        if len(comps) < 2:
            return False
        top_left_parities = {(_bbox(comp)[0] + _bbox(comp)[1]) % 2 for comp in comps}
        if len(top_left_parities) < 2:
            return False

    r0, c0, _r1, _c1 = bbox_main
    if np.array_equal(expected, _component_anchored_fill(main_region, legend)):
        return False
    if np.array_equal(expected, _flat_recolor(main_region, legend)):
        return False
    if np.array_equal(expected, _checkerboard_fill(main_region, legend, offset_parity=(r0 + c0) % 2)):
        return False
    return True


def _generate_candidate() -> np.ndarray | None:
    h = random.randint(_MIN_SIDE, _MAX_SIDE)
    w = random.randint(_MIN_SIDE, _MAX_SIDE)
    palette = random.sample(range(1, 10), 6)
    key1, key2, c11, c12, c21, c22 = palette

    main_mask = _build_main_mask(h, w, key1, key2)
    if main_mask is None:
        return None

    grid = np.zeros((h, w + 3), dtype=int)
    entries = [(key1, (c11, c12)), (key2, (c21, c22))]
    random.shuffle(entries)
    _place_legend(grid, entries)
    grid[:, 3:] = main_mask

    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 10")
