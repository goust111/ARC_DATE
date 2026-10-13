from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
SPECIAL_COLORS = {BLACK, GREY, MAROON}
NON_SPECIAL_COLORS = [1, 2, 3, 4, 6, 7, 8]

LEGEND_ROWS = 3
LEGEND_ROW_HEIGHT = 2
LEGEND_HEIGHT = LEGEND_ROWS * LEGEND_ROW_HEIGHT + LEGEND_ROWS + 1
LEGEND_WIDTH = 10
PATTERN_LIBRARY = [
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 0, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 1]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
]
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return _bbox_from_coords(coords)


def _crop(arr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return arr[r0 : r1 + 1, c0 : c1 + 1]


def _coords_to_mask(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _strictly_inside(
    inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]
) -> bool:
    ir0, ic0, ir1, ic1 = inner
    or0, oc0, or1, oc1 = outer
    return or0 < ir0 and oc0 < ic0 and ir1 < or1 and ic1 < oc1


def _tile_mask(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    mh, mw = map(int, mask.shape)
    reps_r = (int(h) + mh - 1) // mh
    reps_c = (int(w) + mw - 1) // mw
    return np.tile(mask, (reps_r, reps_c))[:h, :w]


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    return len(_components(mask)) == 1


def _touches_own_bbox(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _random_connected_mask(h: int, w: int) -> np.ndarray:
    if h == 1 or w == 1:
        return np.ones((h, w), dtype=bool)
    for _ in range(200):
        density = random.uniform(0.4, 0.8)
        mask = np.random.random((h, w)) < density
        if not mask.any():
            continue
        if not _touches_own_bbox(mask):
            continue
        if _is_connected(mask):
            return mask
    return np.ones((h, w), dtype=bool)


def _gap_positions(length: int) -> tuple[int, int]:
    interior = length - 2
    gap_len = random.randint(2, min(3, interior))
    start = random.randint(1, length - gap_len - 1)
    return int(start), int(gap_len)


def _draw_fence(arr: np.ndarray, color: int, side: str) -> None:
    h, w = map(int, arr.shape)
    arr[0, :] = color
    arr[-1, :] = color
    arr[:, 0] = color
    arr[:, -1] = color
    if side in {"up", "down"}:
        start, gap_len = _gap_positions(w)
        row = 0 if side == "up" else h - 1
        arr[row, start : start + gap_len] = BLACK
    else:
        start, gap_len = _gap_positions(h)
        col = 0 if side == "left" else w - 1
        arr[start : start + gap_len, col] = BLACK


def _classify_fence(mask: np.ndarray) -> str | None:
    h, w = map(int, mask.shape)
    if h < 3 or w < 3:
        return None

    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if np.any(mask & ~border):
        return None

    if not (mask[0, 0] and mask[0, -1] and mask[-1, 0] and mask[-1, -1]):
        return None

    sides = {
        "up": np.flatnonzero(~mask[0, 1:-1]) + 1,
        "down": np.flatnonzero(~mask[-1, 1:-1]) + 1,
        "left": np.flatnonzero(~mask[1:-1, 0]) + 1,
        "right": np.flatnonzero(~mask[1:-1, -1]) + 1,
    }

    candidates = []
    for side, idx in sides.items():
        if idx.size == 0:
            continue
        if idx.size < 2:
            return None
        if not np.array_equal(idx, np.arange(int(idx[0]), int(idx[-1]) + 1)):
            return None
        candidates.append(side)

    if len(candidates) != 1:
        return None

    side = candidates[0]
    if side != "up" and np.any(~mask[0, 1:-1]):
        return None
    if side != "down" and np.any(~mask[-1, 1:-1]):
        return None
    if side != "left" and np.any(~mask[1:-1, 0]):
        return None
    if side != "right" and np.any(~mask[1:-1, -1]):
        return None
    return side


def _legend_mapping_from_subgrid(sub: np.ndarray) -> dict[int, np.ndarray] | None:
    h, w = map(int, sub.shape)
    if h < 3 or w < 3:
        return None
    if not np.all(sub[:, 0] == GREY) or not np.all(sub[:, -1] == GREY):
        return None

    full_rows = [r for r in range(h) if np.all(sub[r, :] == GREY)]
    if len(full_rows) < 2 or full_rows[0] != 0 or full_rows[-1] != h - 1:
        return None

    mapping: dict[int, np.ndarray] = {}
    for a, b in zip(full_rows, full_rows[1:], strict=False):
        if b - a <= 1:
            return None
        row_patch = sub[a + 1 : b, 1:-1]
        maroon_runs = []
        for rr in range(row_patch.shape[0]):
            cols = np.flatnonzero(row_patch[rr] == MAROON)
            if cols.size == 0:
                continue
            if not np.array_equal(cols, np.arange(int(cols[0]), int(cols[-1]) + 1)):
                return None
            maroon_runs.append(int(cols.size))
        if len(maroon_runs) != 1:
            return None
        length = maroon_runs[0]

        sample_mask = ~np.isin(row_patch, list(SPECIAL_COLORS))
        sample_bbox = _bbox_from_mask(sample_mask)
        if sample_bbox is None:
            return None
        pattern = _crop(sample_mask, sample_bbox)
        mapping[length] = pattern.copy()

    lengths = sorted(mapping)
    if lengths != list(range(1, len(lengths) + 1)):
        return None
    return mapping


def _find_legend(grid: np.ndarray) -> tuple[tuple[int, int, int, int], dict[int, np.ndarray]]:
    h, w = map(int, grid.shape)
    grey_comps = _components(grid == GREY)
    candidates = []
    for coords in grey_comps:
        bbox = _bbox_from_coords(coords)
        r0, c0, r1, c1 = bbox
        if r1 != h - 1 and c1 != w - 1:
            continue
        sub = _crop(grid, bbox)
        mapping = _legend_mapping_from_subgrid(sub)
        if mapping is None:
            continue
        candidates.append((coords.shape[0], bbox, mapping))

    assert candidates, "expected one legend component"
    _area, bbox, mapping = max(candidates, key=lambda item: item[0])
    return bbox, mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each enclosed sprite with the legend texture selected by its projection span.

    Input structure:
    - Black background.
    - A grey bordered legend on the right or bottom, split into rows.
    - Each legend row has a maroon horizontal bar on the left and a small texture sample.
    - Main-area sprites are monochrome 4-connected components inside same-color rectangular fences.
    - Each fence has exactly one contiguous gap on one side.

    Rule:
    - The gap side chooses the projection direction.
    - Left/right gaps use the sprite width; up/down gaps use the sprite height.
    - Match that span to the legend row whose maroon bar has the same length.
    - Remove the fence and sprite, then tile the matched texture over the sprite bbox using the sprite color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    legend_bbox, mapping = _find_legend(g)
    lr0, lc0, lr1, lc1 = legend_bbox
    legend_mask = np.zeros_like(g, dtype=bool)
    legend_mask[lr0 : lr1 + 1, lc0 : lc1 + 1] = True

    for color in sorted(int(v) for v in np.unique(g)):
        if color in SPECIAL_COLORS:
            continue
        mask = (g == color) & (~legend_mask)
        if not mask.any():
            continue

        fences = []
        sprites = []
        for coords in _components(mask):
            bbox = _bbox_from_coords(coords)
            local_mask = _coords_to_mask(coords, bbox)
            gap_side = _classify_fence(local_mask)
            item = {"coords": coords, "bbox": bbox}
            if gap_side is None:
                sprites.append(item)
            else:
                item["gap_side"] = gap_side
                fences.append(item)

        assert fences, "each sprite color needs a fence"
        used_sprites: set[int] = set()
        for fence in fences:
            matches = [
                (idx, sprite)
                for idx, sprite in enumerate(sprites)
                if idx not in used_sprites and _strictly_inside(sprite["bbox"], fence["bbox"])
            ]
            assert len(matches) == 1, "each fence must contain exactly one sprite"
            idx, sprite = matches[0]
            used_sprites.add(idx)

            sr0, sc0, sr1, sc1 = sprite["bbox"]
            sprite_h = sr1 - sr0 + 1
            sprite_w = sc1 - sc0 + 1
            span = sprite_w if fence["gap_side"] in {"left", "right"} else sprite_h
            assert span in mapping, "sprite projection span must appear in legend"

            pattern = mapping[span]
            tiled = _tile_mask(pattern, sprite_h, sprite_w)
            out[fence["coords"][:, 0], fence["coords"][:, 1]] = BLACK
            out[sr0 : sr1 + 1, sc0 : sc1 + 1] = BLACK
            out[sr0 : sr1 + 1, sc0 : sc1 + 1][tiled] = color

    return out


def _sample_patterns() -> dict[int, np.ndarray]:
    patterns: dict[int, np.ndarray] = {}
    chosen = random.sample(PATTERN_LIBRARY, LEGEND_ROWS)
    for length, base in enumerate(chosen, start=1):
        mask = base.copy()
        if random.choice([False, True]):
            mask = np.fliplr(mask)
        if mask.shape[0] > 1 and random.choice([False, True]):
            mask = np.flipud(mask)
        patterns[length] = mask
    return patterns


def _legend_patch(patterns: dict[int, np.ndarray]) -> np.ndarray:
    patch = np.full((LEGEND_HEIGHT, LEGEND_WIDTH), BLACK, dtype=int)
    patch[0, :] = GREY
    patch[-1, :] = GREY
    patch[:, 0] = GREY
    patch[:, -1] = GREY

    sample_colors = random.choices(NON_SPECIAL_COLORS, k=LEGEND_ROWS)
    for row_idx in range(LEGEND_ROWS):
        sep = row_idx * (LEGEND_ROW_HEIGHT + 1)
        patch[sep, :] = GREY
        start_r = sep + 1
        end_r = start_r + LEGEND_ROW_HEIGHT
        length = row_idx + 1
        patch[start_r, 1 : 1 + length] = MAROON

        pattern = patterns[length]
        ph, pw = map(int, pattern.shape)
        top = start_r + random.randint(0, LEGEND_ROW_HEIGHT - ph)
        left = 6
        color = int(sample_colors[row_idx])
        patch[top : top + ph, left : left + pw][pattern] = color
        patch[end_r, :] = GREY

    return patch


def _positive_split(total: int) -> tuple[int, int]:
    left = random.randint(1, total - 1)
    return int(left), int(total - left)


def _make_object_patch(cell_h: int, cell_w: int, direction: str, span: int, color: int) -> np.ndarray | None:
    if direction in {"left", "right"}:
        sprite_w = span
        max_h = min(3, cell_h - 4)
        if max_h < 1:
            return None
        sprite_h = random.randint(1, max_h)
    else:
        sprite_h = span
        max_w = min(3, cell_w - 4)
        if max_w < 1:
            return None
        sprite_w = random.randint(1, max_w)

    min_fence_h = sprite_h + 4
    min_fence_w = sprite_w + 4
    if min_fence_h > cell_h or min_fence_w > cell_w:
        return None

    fence_h = random.randint(min_fence_h, cell_h)
    fence_w = random.randint(min_fence_w, cell_w)

    top_gap = fence_h - sprite_h - 2
    left_gap = fence_w - sprite_w - 2
    pad_t, pad_b = _positive_split(top_gap)
    pad_l, pad_r = _positive_split(left_gap)

    patch = np.full((cell_h, cell_w), BLACK, dtype=int)
    off_r = random.randint(0, cell_h - fence_h)
    off_c = random.randint(0, cell_w - fence_w)

    fence = np.full((fence_h, fence_w), BLACK, dtype=int)
    _draw_fence(fence, color, direction)
    patch[off_r : off_r + fence_h, off_c : off_c + fence_w] = np.maximum(
        patch[off_r : off_r + fence_h, off_c : off_c + fence_w], fence
    )

    sprite_mask = _random_connected_mask(sprite_h, sprite_w)
    sr = off_r + 1 + pad_t
    sc = off_c + 1 + pad_l
    patch[sr : sr + sprite_h, sc : sc + sprite_w][sprite_mask] = color
    return patch


def _cell_boxes(height: int, width: int) -> list[tuple[int, int, int, int]]:
    row_cut = height // 2
    col_cut = width // 2
    cells = [
        (0, 0, row_cut - 1, col_cut - 1),
        (0, col_cut + 1, row_cut - 1, width - 1),
        (row_cut + 1, 0, height - 1, col_cut - 1),
        (row_cut + 1, col_cut + 1, height - 1, width - 1),
    ]
    return cells


def _embed(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = map(int, patch.shape)
    grid[top : top + h, left : left + w] = patch


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(25, 30)
    width = random.randint(25, 30)
    orientation = random.choice(["right", "bottom"])

    patterns = _sample_patterns()
    legend = _legend_patch(patterns)
    grid = np.full((height, width), BLACK, dtype=int)

    if orientation == "right":
        legend_r0 = random.randint(0, height - LEGEND_HEIGHT)
        legend_c0 = width - LEGEND_WIDTH
        _embed(grid, legend, legend_r0, legend_c0)
        main_r0, main_c0 = 0, 0
        main_h, main_w = height, legend_c0
    else:
        legend_r0 = height - LEGEND_HEIGHT
        legend_c0 = random.randint(0, width - LEGEND_WIDTH)
        _embed(grid, legend, legend_r0, legend_c0)
        main_r0, main_c0 = 0, 0
        main_h, main_w = legend_r0, width

    if main_h < 15 or main_w < 15:
        return None

    cells = _cell_boxes(main_h, main_w)
    if any(r1 - r0 + 1 < 7 or c1 - c0 + 1 < 7 for r0, c0, r1, c1 in cells):
        return None

    directions = random.sample(["up", "down", "left", "right"], 4)
    spans = [1, 2, 3, random.randint(1, 3)]
    random.shuffle(spans)
    colors = random.sample(NON_SPECIAL_COLORS, 4)

    for (r0, c0, r1, c1), direction, span, color in zip(cells, directions, spans, colors, strict=False):
        cell_h = r1 - r0 + 1
        cell_w = c1 - c0 + 1
        patch = _make_object_patch(cell_h, cell_w, direction, span, color)
        if patch is None:
            return None
        _embed(grid, patch, main_r0 + r0, main_c0 + c0)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
