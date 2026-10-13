from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
PATTERN_COLORS = (1, 2, 3, 4, 6, 7, 8)

ROW_COUNT_RANGE = (2, 4)
COL_COUNT_RANGE = (2, 4)
MIN_CELL_SIZE = 4
MAX_CELL_SIZE = 6
PATTERN_SIZE_CHOICES = (2, 3)
GEN_ATTEMPTS = 400

CORNER_OFFSETS = {
    "tl": (0, 0),
    "tr": (0, -1),
    "bl": (-1, 0),
    "br": (-1, -1),
}


def _separator_spans(length: int, separators: np.ndarray) -> list[tuple[int, int]]:
    bounds = [-1] + sorted({int(idx) for idx in np.ravel(separators)}) + [int(length)]
    spans: list[tuple[int, int]] = []
    for left, right in zip(bounds, bounds[1:]):
        start = left + 1
        end = right
        if start < end:
            spans.append((start, end))
    return spans


def _region_layout(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    sep_rows = np.flatnonzero(np.all(grid == GREY, axis=1))
    sep_cols = np.flatnonzero(np.all(grid == GREY, axis=0))
    row_spans = _separator_spans(grid.shape[0], sep_rows)
    col_spans = _separator_spans(grid.shape[1], sep_cols)
    assert row_spans and col_spans
    assert len({r1 - r0 for r0, r1 in row_spans}) == 1
    assert len({c1 - c0 for c0, c1 in col_spans}) == 1
    return row_spans, col_spans


def _marker_corner(region: np.ndarray) -> str:
    corners = {
        "tl": int(region[0, 0]) == MAROON,
        "tr": int(region[0, -1]) == MAROON,
        "bl": int(region[-1, 0]) == MAROON,
        "br": int(region[-1, -1]) == MAROON,
    }
    matches = [name for name, ok in corners.items() if ok]
    assert len(matches) == 1
    return matches[0]


def _crop_pattern(region: np.ndarray) -> np.ndarray:
    mask = (region != BLACK) & (region != GREY) & (region != MAROON)
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0) + 1
    return region[r0:r1, c0:c1].copy()


def _transform_patch(patch: np.ndarray, corner: str) -> np.ndarray:
    if corner == "tl":
        return patch.copy()
    if corner == "tr":
        return np.fliplr(patch).copy()
    if corner == "bl":
        return np.flipud(patch).copy()
    return np.flipud(np.fliplr(patch)).copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flip each cell's colored micro-pattern according to its maroon corner marker and recenter it inside the cell.

    Input structure:
    - The background is black.
    - One-cell-thick grey full rows and full columns partition the grid into equal rectangular subregions.
    - Each subregion contains a small non-black, non-grey colored pattern and exactly one maroon marker pixel in one corner; the remaining subregion cells are black.

    Rule:
    - Detect the grey separator rows and columns to recover all subregions.
    - In each subregion, read which corner contains the maroon marker.
    - Transform the pattern inside its own bounding box: top-left marker keeps it unchanged, top-right flips it left-right, bottom-left flips it top-bottom, and bottom-right flips both axes.
    - Remove the maroon marker, place the transformed pattern centered in the same subregion, and keep every grey separator pixel unchanged.

    Key insight:
    - The single maroon corner pixel is a hidden control signal; its corner location selects which local reflection of the nearby pattern should be drawn.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    row_spans, col_spans = _region_layout(g)

    out = np.full_like(g, BLACK)
    out[g == GREY] = GREY

    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            region = g[r0:r1, c0:c1]
            corner = _marker_corner(region)
            patch = _transform_patch(_crop_pattern(region), corner)
            ph, pw = patch.shape
            cell_h, cell_w = r1 - r0, c1 - c0
            assert ph <= cell_h and pw <= cell_w
            rr = r0 + (cell_h - ph) // 2
            cc = c0 + (cell_w - pw) // 2
            out[rr : rr + ph, cc : cc + pw] = patch

    return out


def _is_connected(mask: np.ndarray) -> bool:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if not coords:
        return False
    seen = {coords[0]}
    queue = deque([coords[0]])
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == len(coords)


def _random_pattern_mask() -> np.ndarray | None:
    h = random.choice(PATTERN_SIZE_CHOICES)
    w = random.choice(PATTERN_SIZE_CHOICES)
    for _ in range(80):
        mask = np.array([[random.random() < 0.55 for _ in range(w)] for _ in range(h)], dtype=bool)
        if int(mask.sum()) < 2 or int(mask.sum()) == h * w:
            continue
        if not mask[0].any() or not mask[-1].any() or not mask[:, 0].any() or not mask[:, -1].any():
            continue
        if not _is_connected(mask):
            continue
        variants = {
            mask.tobytes(),
            np.fliplr(mask).tobytes(),
            np.flipud(mask).tobytes(),
            np.flipud(np.fliplr(mask)).tobytes(),
        }
        if len(variants) != 4:
            continue
        return mask
    return None


def _paint_separators(grid: np.ndarray, row_spans: list[tuple[int, int]], col_spans: list[tuple[int, int]]) -> None:
    for r0, r1 in row_spans[:-1]:
        grid[r1, :] = GREY
    for c0, c1 in col_spans[:-1]:
        grid[:, c1] = GREY


def _corner_cell(cell_h: int, cell_w: int, corner: str) -> tuple[int, int]:
    rr = 0 if corner[0] == "t" else cell_h - 1
    cc = 0 if corner[1] == "l" else cell_w - 1
    return rr, cc


def _placement_options(cell_h: int, cell_w: int, patch: np.ndarray, corner: str) -> list[tuple[int, int]]:
    ph, pw = patch.shape
    mr, mc = _corner_cell(cell_h, cell_w, corner)
    placements: list[tuple[int, int]] = []
    for r in range(cell_h - ph + 1):
        for c in range(cell_w - pw + 1):
            if r <= mr < r + ph and c <= mc < c + pw:
                continue
            placements.append((r, c))
    return placements


def _expected_from_specs(
    shape: tuple[int, int],
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    specs: list[tuple[int, int, str, np.ndarray]],
) -> np.ndarray:
    out = np.full(shape, BLACK, dtype=int)
    _paint_separators(out, row_spans, col_spans)
    for row_idx, col_idx, corner, patch in specs:
        r0, r1 = row_spans[row_idx]
        c0, c1 = col_spans[col_idx]
        transformed = _transform_patch(patch, corner)
        ph, pw = transformed.shape
        rr = r0 + (r1 - r0 - ph) // 2
        cc = c0 + (c1 - c0 - pw) // 2
        out[rr : rr + ph, cc : cc + pw] = transformed
    return out


def _choose_grid_spec() -> tuple[int, int, int, int] | None:
    n_rows = random.randint(*ROW_COUNT_RANGE)
    n_cols = random.randint(*COL_COUNT_RANGE)
    max_cell_h = min(MAX_CELL_SIZE, (30 - (n_rows - 1)) // n_rows)
    max_cell_w = min(MAX_CELL_SIZE, (30 - (n_cols - 1)) // n_cols)
    if max_cell_h < MIN_CELL_SIZE or max_cell_w < MIN_CELL_SIZE:
        return None
    cell_h = random.randint(MIN_CELL_SIZE, max_cell_h)
    cell_w = random.randint(MIN_CELL_SIZE, max_cell_w)
    return n_rows, n_cols, cell_h, cell_w


def _build_spans(count: int, cell_size: int) -> list[tuple[int, int]]:
    spans = []
    for idx in range(count):
        start = idx * (cell_size + 1)
        spans.append((start, start + cell_size))
    return spans


def _gen_attempt() -> np.ndarray | None:
    spec = _choose_grid_spec()
    if spec is None:
        return None
    n_rows, n_cols, cell_h, cell_w = spec
    row_spans = _build_spans(n_rows, cell_h)
    col_spans = _build_spans(n_cols, cell_w)
    height = row_spans[-1][1]
    width = col_spans[-1][1]
    grid = np.full((height, width), BLACK, dtype=int)
    _paint_separators(grid, row_spans, col_spans)

    corners = (["tl", "tr", "bl", "br"] * ((n_rows * n_cols + 3) // 4))[: n_rows * n_cols]
    random.shuffle(corners)

    specs: list[tuple[int, int, str, np.ndarray]] = []
    cell_index = 0
    for row_idx, (r0, r1) in enumerate(row_spans):
        for col_idx, (c0, c1) in enumerate(col_spans):
            corner = corners[cell_index]
            cell_index += 1
            mask = _random_pattern_mask()
            if mask is None:
                return None
            color = random.choice(PATTERN_COLORS)
            patch = np.where(mask, color, BLACK).astype(int)
            placements = _placement_options(cell_h, cell_w, patch, corner)
            if not placements:
                return None
            pr, pc = random.choice(placements)
            region = grid[r0:r1, c0:c1]
            region[pr : pr + patch.shape[0], pc : pc + patch.shape[1]] = patch
            mr, mc = _corner_cell(cell_h, cell_w, corner)
            region[mr, mc] = MAROON
            specs.append((row_idx, col_idx, corner, patch.copy()))

    expected = _expected_from_specs(grid.shape, row_spans, col_spans, specs)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
