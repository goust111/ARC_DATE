from __future__ import annotations

import numpy as np


DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
RNG = np.random.default_rng()

# --------------- solver ---------------


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue

            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            coord_arr = np.array(coords, dtype=int)
            r0, c0 = coord_arr.min(axis=0)
            r1, c1 = coord_arr.max(axis=0)
            components.append(
                {
                    "color": color,
                    "coords": coord_arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )

    components.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["color"]))
    return components


def main(grid: np.ndarray) -> np.ndarray:
    """
    Horizontally flip every 4-connected monochrome sprite in place within its own bbox.

    Input structure:
    - The background is dominant.
    - The foreground consists of separated monochrome 4-connected sprites.

    Rule:
    - For each sprite, reflect its occupied cells across the vertical midline of its bbox.
    - Keep each reflected sprite in the same bbox position.
    - Preserve colors and background.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _get_bg(g)
    out = g.copy()

    for comp in _find_components(g, bg):
        coords = comp["coords"]
        color = int(comp["color"])
        _r0, c0, _r1, c1 = comp["bbox"]

        out[coords[:, 0], coords[:, 1]] = bg
        mirrored_cols = c1 - (coords[:, 1] - c0)
        out[coords[:, 0], mirrored_cols] = color

    return out


# --------------- generator ---------------

BG_COLOR = 8
GRID_H_RANGE = (10, 19)
GRID_W_RANGE = (10, 19)
OBJECT_COUNT_RANGE = (2, 5)
OBJECT_AREA_RANGE = (4, 8)
MAX_BBOX_HEIGHT = 4
MAX_BBOX_WIDTH = 5
MIN_DENSITY_X10 = 4
BOX_PADDING = 1
MAX_GENERATION_ATTEMPTS = 400
COLOR_POOL = [0, 1, 2, 3, 4, 5, 6, 7, 9]


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return mask[rows][:, cols]


def _grow_mask(height: int, width: int, area: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    start = (int(RNG.integers(height)), int(RNG.integers(width)))
    cells = {start}

    while len(cells) < area:
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for dr, dc in DIRS4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        pick = tuple(frontier)[int(RNG.integers(len(frontier)))]
        cells.add(pick)

    for r, c in cells:
        mask[r, c] = True
    return _crop_mask(mask)


def _random_asymmetric_mask() -> np.ndarray | None:
    area = int(RNG.integers(OBJECT_AREA_RANGE[0], OBJECT_AREA_RANGE[1] + 1))
    dims: list[tuple[int, int]] = []
    for height in range(2, MAX_BBOX_HEIGHT + 1):
        for width in range(2, MAX_BBOX_WIDTH + 1):
            if area > height * width:
                continue
            if area * 10 < MIN_DENSITY_X10 * height * width:
                continue
            dims.append((height, width))

    if not dims:
        return None

    for _ in range(80):
        height, width = dims[int(RNG.integers(len(dims)))]
        mask = _grow_mask(height, width, area)
        if mask.shape[1] <= 1:
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        return mask
    return None


def _bbox_is_free(blocked: np.ndarray, top: int, left: int, height: int, width: int) -> bool:
    h, w = blocked.shape
    r0 = max(0, top - BOX_PADDING)
    c0 = max(0, left - BOX_PADDING)
    r1 = min(h, top + height + BOX_PADDING)
    c1 = min(w, left + width + BOX_PADDING)
    return not bool(blocked[r0:r1, c0:c1].any())


def _mark_bbox(blocked: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    h, w = blocked.shape
    r0 = max(0, top - BOX_PADDING)
    c0 = max(0, left - BOX_PADDING)
    r1 = min(h, top + height + BOX_PADDING)
    c1 = min(w, left + width + BOX_PADDING)
    blocked[r0:r1, c0:c1] = True


def _place_objects(grid: np.ndarray, masks: list[np.ndarray], colors: list[int]) -> bool:
    h, w = grid.shape
    blocked = np.zeros((h, w), dtype=bool)

    order = sorted(range(len(masks)), key=lambda idx: masks[idx].shape[0] * masks[idx].shape[1], reverse=True)
    for idx in order:
        mask = masks[idx]
        color = colors[idx]
        mh, mw = mask.shape

        positions = [
            (top, left)
            for top in range(h - mh + 1)
            for left in range(w - mw + 1)
            if _bbox_is_free(blocked, top, left, mh, mw)
        ]
        if not positions:
            return False

        top, left = positions[int(RNG.integers(len(positions)))]
        patch = np.full((mh, mw), BG_COLOR, dtype=int)
        patch[mask] = color
        grid[top : top + mh, left : left + mw][mask] = color
        _mark_bbox(blocked, top, left, mh, mw)

    return True


def _component_shapes(grid: np.ndarray) -> list[np.ndarray]:
    bg = _get_bg(grid)
    shapes = []
    for comp in _find_components(grid, bg):
        coords = comp["coords"]
        r0, c0, r1, c1 = comp["bbox"]
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
        shapes.append(mask)
    return shapes


def _gen_attempt() -> np.ndarray | None:
    height = int(RNG.integers(GRID_H_RANGE[0], GRID_H_RANGE[1]))
    width = int(RNG.integers(GRID_W_RANGE[0], GRID_W_RANGE[1]))
    object_count = int(RNG.integers(OBJECT_COUNT_RANGE[0], OBJECT_COUNT_RANGE[1] + 1))

    masks: list[np.ndarray] = []
    for _ in range(object_count):
        mask = _random_asymmetric_mask()
        if mask is None:
            return None
        if bool(RNG.integers(0, 2)):
            mask = np.fliplr(mask)
        masks.append(mask)

    colors = list(RNG.choice(COLOR_POOL, size=object_count, replace=False))
    grid = np.full((height, width), BG_COLOR, dtype=int)
    if not _place_objects(grid, masks, colors):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    in_shapes = _component_shapes(grid)
    out_shapes = _component_shapes(out)
    if len(in_shapes) != object_count or len(out_shapes) != object_count:
        return None

    for mask in in_shapes:
        if mask.sum() < OBJECT_AREA_RANGE[0] or mask.sum() > OBJECT_AREA_RANGE[1]:
            return None
        if np.array_equal(mask, np.fliplr(mask)):
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
