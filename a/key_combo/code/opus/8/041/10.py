from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
RED = 2
SPRITE_COLORS = (1, 3, 4, 6, 7, 8, 9)
MAX_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _scale(sprite: np.ndarray, factor: int) -> np.ndarray:
    return np.repeat(np.repeat(sprite, factor, axis=0), factor, axis=1)


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    marker_rows = []
    for r in range(h):
        row = g[r]
        non_bg = set(map(int, row[row != BACKGROUND]))
        red_cols = np.flatnonzero(row == RED)
        if len(red_cols) == 0:
            continue
        assert non_bg == {RED}, "marker rows may only contain red markers"
        assert len(red_cols) == 2, "each marker row must contain exactly two red cells"
        c0, c1 = map(int, red_cols)
        assert c0 + c1 == w - 1, "marker pair must be mirrored about the vertical axis"
        gap = c1 - c0 - 1
        marker_rows.append((gap, r, c0, c1))

    assert 3 <= len(marker_rows) <= 5, "expected 3-5 marker rows"
    marker_rows.sort()
    gaps = [gap for gap, _r, _c0, _c1 in marker_rows]
    min_gap = min(gaps)
    assert gaps.count(min_gap) == 1, "minimum gap must be unique"

    marker_half_limit = h // 2
    assert all(r < marker_half_limit for _gap, r, _c0, _c1 in marker_rows), "markers must live in the upper half"

    sprite_mask = (g != BACKGROUND) & (g != RED)
    assert sprite_mask.any(), "expected a non-red sprite"
    comps = _components4(sprite_mask)
    assert len(comps) == 1, "sprite must be a single 4-connected component"
    sprite_bbox = _bbox(sprite_mask)
    assert sprite_bbox is not None
    r0, c0, r1, c1 = sprite_bbox
    assert r0 >= marker_half_limit, "sprite must live in the lower half"
    sprite = _crop(g, sprite_bbox)

    return {
        "shape": (h, w),
        "rows": marker_rows,
        "min_gap": min_gap,
        "chosen_row": next(item for item in marker_rows if item[0] == min_gap),
        "sprite_bbox": sprite_bbox,
        "sprite": sprite,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Enlarge the lower sprite by the unique smallest horizontal gap among mirrored red marker pairs above it.

    Input structure:
    - The grid has black background, 3-5 upper rows each containing exactly two red pixels mirrored across the vertical center, and one lower multicolor sprite that uses non-black non-red colors.
    - Marker rows contain only red pixels and background; the sprite is a single 4-connected object whose bounding box may include internal black holes.

    Rule:
    - Compute the horizontal background gap between the two red markers on every marker row.
    - Find the unique row with the smallest such gap and connect that row's two markers with a red horizontal segment.
    - Crop the lower sprite to its tight bounding box, preserving any black cells inside the crop.
    - Scale the crop uniformly by the smallest gap value, replacing each source cell with a same-color square of that size, and output only the scaled sprite.

    Key insight:
    - The upper red markers are not themselves the output object: their closest pair encodes a number, and that number becomes the scale factor for the separate sprite below.
    """
    info = _analyze(grid)
    g = np.asarray(grid, dtype=int).copy()
    _gap, r, c0, c1 = info["chosen_row"]
    g[r, c0 : c1 + 1] = RED
    return _scale(_crop(g, info["sprite_bbox"]), info["min_gap"])


def _touches_all_bbox_sides(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _build_mask_library() -> dict[tuple[int, int], list[np.ndarray]]:
    library: dict[tuple[int, int], list[np.ndarray]] = {}
    for h in (2, 3):
        for w in (2, 3):
            masks: list[np.ndarray] = []
            total = h * w
            for bits in range(1 << total):
                cells = bits.bit_count()
                if cells < 3 or cells >= total:
                    continue
                mask = np.zeros((h, w), dtype=bool)
                for idx in range(total):
                    if bits & (1 << idx):
                        mask[idx // w, idx % w] = True
                if not _touches_all_bbox_sides(mask):
                    continue
                if len(_components4(mask)) != 1:
                    continue
                masks.append(mask)
            assert masks
            library[(h, w)] = masks
    return library


MASK_LIBRARY = _build_mask_library()


def _sample_sprite() -> np.ndarray:
    h, w = random.choice(list(MASK_LIBRARY))
    mask = random.choice(MASK_LIBRARY[(h, w)])
    area = int(mask.sum())
    color_count = 2 if area <= 4 else random.choice((2, 3))
    palette = random.sample(SPRITE_COLORS, color_count)
    sprite = np.zeros((h, w), dtype=int)

    cells = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    random.shuffle(cells)

    # Use every chosen color at least once so the sprite is visibly multicolor.
    for idx, (r, c) in enumerate(cells):
        sprite[r, c] = palette[idx] if idx < color_count else random.choice(palette)
    return sprite


def _feasible_gaps(width: int) -> list[int]:
    parity = width % 2
    return [gap for gap in range(2, 7) if gap <= width - 2 and gap % 2 == parity]


def _cols_for_gap(width: int, gap: int) -> tuple[int, int]:
    c0 = (width - gap - 2) // 2
    c1 = width - 1 - c0
    return int(c0), int(c1)


def _make_case() -> np.ndarray | None:
    width = random.randint(9, 18)
    gaps = _feasible_gaps(width)
    if len(gaps) < 2:
        return None

    height = random.randint(14, 18)
    sprite = _sample_sprite()
    sh, sw = sprite.shape
    chosen_gap_levels = sorted(random.sample(gaps, k=random.randint(2, len(gaps))))
    min_gap = random.choice(chosen_gap_levels[:-1])
    higher_gaps = [gap for gap in chosen_gap_levels if gap > min_gap]
    if not higher_gaps:
        return None

    marker_rows = random.randint(3, 5)
    top_rows = list(range(0, height // 2 - 1))
    if len(top_rows) < marker_rows:
        return None
    chosen_rows = sorted(random.sample(top_rows, marker_rows))
    row_gaps = [min_gap] + [random.choice(higher_gaps) for _ in range(marker_rows - 1)]
    random.shuffle(row_gaps)
    if row_gaps.count(min_gap) != 1:
        return None

    sprite_top_min = height // 2
    sprite_top_max = height - sh
    if sprite_top_min > sprite_top_max:
        return None
    sprite_r0 = random.randint(sprite_top_min, sprite_top_max)
    sprite_c0 = random.randint(0, width - sw)

    grid = np.zeros((height, width), dtype=int)
    for r, gap in zip(chosen_rows, row_gaps):
        c0, c1 = _cols_for_gap(width, gap)
        grid[r, c0] = RED
        grid[r, c1] = RED
    grid[sprite_r0 : sprite_r0 + sh, sprite_c0 : sprite_c0 + sw] = sprite

    info = _analyze(grid)
    out = main(grid)
    expected = _scale(info["sprite"], info["min_gap"])
    if not np.array_equal(out, expected):
        return None
    if max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _make_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 10")
