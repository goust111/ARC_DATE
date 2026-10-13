from __future__ import annotations

import itertools
from collections import defaultdict

import numpy as np


BG = 0
FILL = 1
FG = 2
MASK_SIZE = 3
CANVAS_SIZE = 15
CENTRAL_SLICE = (slice(3, 12), slice(3, 12))
_RNG = np.random.default_rng()


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _paint_pattern(canvas: np.ndarray, coords: np.ndarray, top: int, left: int, color: int) -> None:
    rows = top + coords[:, 0]
    cols = left + coords[:, 1]
    canvas[rows, cols] = int(color)


def _expand_mask(mask: np.ndarray, color: int, *, use_blocks: bool = True, use_stamps: bool = True) -> np.ndarray:
    out = np.full((CANVAS_SIZE, CANVAS_SIZE), BG, dtype=int)
    coords = np.argwhere(mask)

    for r_raw, c_raw in coords:
        r = int(r_raw)
        c = int(c_raw)
        block_r = MASK_SIZE * (r + 1)
        block_c = MASK_SIZE * (c + 1)

        if use_blocks:
            out[block_r : block_r + MASK_SIZE, block_c : block_c + MASK_SIZE] = int(color)

        if not use_stamps:
            continue
        if r == 0:
            _paint_pattern(out, coords, block_r - MASK_SIZE, block_c, color)
        elif r == MASK_SIZE - 1:
            _paint_pattern(out, coords, block_r + MASK_SIZE, block_c, color)
        if c == 0:
            _paint_pattern(out, coords, block_r, block_c - MASK_SIZE, color)
        elif c == MASK_SIZE - 1:
            _paint_pattern(out, coords, block_r, block_c + MASK_SIZE, color)

    return out


def _enclosed_regions(grid: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    enclosed: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != BG or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            touches_edge = False

            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                if rr in (0, h - 1) or cc in (0, w - 1):
                    touches_edge = True
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == BG:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            if not touches_edge:
                enclosed.append(np.array(coords, dtype=int))

    return enclosed


def _solve_mask(mask: np.ndarray, color: int) -> np.ndarray:
    out = _expand_mask(mask, color, use_blocks=True, use_stamps=True)
    for region in _enclosed_regions(out):
        out[region[:, 0], region[:, 1]] = FILL
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a 3x3 mask into wall blocks and edge stamps, then flood-fill the enclosed voids.

    Input structure:
    - The input is a 3x3 black-background grid with exactly one foreground color.
    - The foreground occupies 3 to 6 cells and defines a binary mask.
    - Foreground cells may lie on edges or corners, which controls outward stamp directions.

    Rule:
    - Create a 15x15 black canvas and map each foreground input cell to a 3x3 solid block in the centered 3x3 block lattice.
    - For every foreground cell on the input border, stamp the original 3x3 foreground mask once just outside that block on each touched border side.
    - Treat all foreground pixels as walls, find every 4-connected black region, and recolor only the regions that do not touch the canvas border to blue.

    Key insight:
    - The same 3x3 mask defines both the large wall skeleton and the outward stamp extensions, and only their combined topology determines which black pockets become fillable enclosures.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (MASK_SIZE, MASK_SIZE), "expected a 3x3 input grid"

    colors = sorted(int(x) for x in np.unique(g) if int(x) != BG)
    assert len(colors) == 1, "expected exactly one foreground color"
    color = colors[0]
    assert color != FILL, "foreground color must differ from the fill color"

    return _solve_mask(g == color, color)


def _mask_from_bits(bits: int) -> np.ndarray:
    flat = np.array([(bits >> i) & 1 for i in range(MASK_SIZE * MASK_SIZE)], dtype=bool)
    return flat.reshape(MASK_SIZE, MASK_SIZE)


def _profile_mask(mask: np.ndarray) -> dict | None:
    fg_count = int(mask.sum())
    if not 3 <= fg_count <= 6:
        return None

    full = _expand_mask(mask, FG, use_blocks=True, use_stamps=True)
    blocks_only = _expand_mask(mask, FG, use_blocks=True, use_stamps=False)
    stamps_only = _expand_mask(mask, FG, use_blocks=False, use_stamps=True)

    full_regions = _enclosed_regions(full)
    if not full_regions:
        return None
    if _enclosed_regions(blocks_only):
        return None
    if _enclosed_regions(stamps_only):
        return None

    enclosed_sizes = tuple(sorted(int(len(region)) for region in full_regions))
    total_fill = int(sum(enclosed_sizes))
    if len(enclosed_sizes) > 4 or total_fill < 2 or total_fill > 16:
        return None

    outer = full.copy()
    outer[CENTRAL_SLICE] = BG
    outer_pixels = int(np.count_nonzero(outer == FG))
    if outer_pixels == 0:
        return None

    edge_count = int(mask[0, :].sum() + mask[2, :].sum() + mask[1, 0] + mask[1, 2])
    corner_count = int(mask[0, 0] + mask[0, 2] + mask[2, 0] + mask[2, 2])
    profile = {
        "mask": mask.copy(),
        "fg_count": fg_count,
        "has_center": bool(mask[1, 1]),
        "edge_count": edge_count,
        "corner_count": corner_count,
        "region_count": len(enclosed_sizes),
        "enclosed_sizes": enclosed_sizes,
        "total_fill": total_fill,
    }

    grid = np.full((MASK_SIZE, MASK_SIZE), BG, dtype=int)
    grid[mask] = FG
    expected = _solve_mask(mask, FG)
    if not np.array_equal(main(grid), expected):
        return None

    blue_pixels = int(np.count_nonzero(expected == FILL))
    if blue_pixels != total_fill:
        return None
    return profile


def _build_candidate_profiles() -> list[dict]:
    profiles: list[dict] = []
    for fg_count in range(3, 7):
        for cells in itertools.combinations(range(MASK_SIZE * MASK_SIZE), fg_count):
            bits = 0
            for idx in cells:
                bits |= 1 << idx
            profile = _profile_mask(_mask_from_bits(bits))
            if profile is not None:
                profiles.append(profile)

    assert len(profiles) >= 20, "need a sufficiently diverse generator pool"
    return profiles


_CANDIDATE_PROFILES = _build_candidate_profiles()
_PROFILE_BUCKETS: dict[tuple[int, bool, int, int], list[dict]] = defaultdict(list)
for _profile in _CANDIDATE_PROFILES:
    key = (
        int(_profile["fg_count"]),
        bool(_profile["has_center"]),
        int(_profile["region_count"]),
        int(_profile["corner_count"]),
    )
    _PROFILE_BUCKETS[key].append(_profile)
_BUCKET_KEYS = tuple(_PROFILE_BUCKETS)


def _validate_generated(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.shape != (MASK_SIZE, MASK_SIZE):
        return False
    if not set(map(int, np.unique(grid))).issubset({BG, FG}):
        return False

    mask = grid == FG
    profile = _profile_mask(mask)
    if profile is None:
        return False

    out = main(grid)
    if out.shape != (CANVAS_SIZE, CANVAS_SIZE):
        return False
    if not np.any(out == FILL):
        return False
    return True


def generate_input() -> np.ndarray:
    bucket_key = _BUCKET_KEYS[int(_RNG.integers(len(_BUCKET_KEYS)))]
    bucket = _PROFILE_BUCKETS[bucket_key]
    profile = bucket[int(_RNG.integers(len(bucket)))]

    grid = np.full((MASK_SIZE, MASK_SIZE), BG, dtype=int)
    grid[profile["mask"]] = FG
    assert _validate_generated(grid)
    return grid
