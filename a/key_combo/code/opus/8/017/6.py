from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

try:
    from arc_utils import Color, bbox, components_mask
except ModuleNotFoundError:
    helper_dir = Path("/home/lixinyu/snap/new/solvers")
    if str(helper_dir) not in sys.path:
        sys.path.append(str(helper_dir))
    from arc_utils import Color, bbox, components_mask


BG = Color.BLACK
SEP = Color.GREY
MAIN_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW)
NOISE_COLORS = (Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON)
GEN_ATTEMPTS = 400
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _find_separator_cross(grid: np.ndarray) -> tuple[int, int, int]:
    h, w = grid.shape
    solid_rows = [(r, int(grid[r, 0])) for r in range(h) if np.all(grid[r, :] == grid[r, 0])]
    solid_cols = [(c, int(grid[0, c])) for c in range(w) if np.all(grid[:, c] == grid[0, c])]
    matches = [
        (r, c, row_color)
        for r, row_color in solid_rows
        for c, col_color in solid_cols
        if row_color == col_color
    ]
    assert len(matches) == 1, "expected exactly one solid separator cross"
    r, c, color = matches[0]
    assert 0 < r < h - 1 and 0 < c < w - 1, "cross must split grid into four nonempty quadrants"
    return r, c, color


def _quadrants(shape: tuple[int, int], cross_r: int, cross_c: int) -> dict[str, tuple[slice, slice]]:
    h, w = shape
    return {
        "UL": (slice(0, cross_r), slice(0, cross_c)),
        "UR": (slice(0, cross_r), slice(cross_c + 1, w)),
        "LL": (slice(cross_r + 1, h), slice(0, cross_c)),
        "LR": (slice(cross_r + 1, h), slice(cross_c + 1, w)),
    }


def _dominant_color(region: np.ndarray) -> int:
    values = [int(v) for v in region.ravel() if int(v) not in (BG, SEP)]
    counts = Counter(values)
    assert counts, "quadrant must contain foreground"
    ranked = counts.most_common()
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "quadrant dominant color must be unique"
    return int(ranked[0][0])


def _mirror_mask(mask: np.ndarray, axis: str) -> np.ndarray:
    b = bbox(mask)
    assert b is not None, "expected a non-empty mask"
    r0, c0, r1, c1 = b
    coords = np.argwhere(mask)
    out = np.zeros_like(mask, dtype=bool)
    if axis == "horizontal":
        rr = coords[:, 0]
        cc = c0 + c1 - coords[:, 1]
    elif axis == "vertical":
        rr = r0 + r1 - coords[:, 0]
        cc = coords[:, 1]
    else:
        raise ValueError(f"unknown axis {axis}")
    out[rr, cc] = True
    return out


def _is_connected(mask: np.ndarray) -> bool:
    return len(components_mask(mask, connectivity=4)) == 1


def main(grid: np.ndarray) -> np.ndarray:
    """Denoise each cross-defined quadrant, then mirror diagonal quadrant pairs on matched axes.

    Input structure:
    - A black background contains exactly one full grey row and one full grey column,
      forming a separator cross that splits the canvas into four nonempty quadrants.
    - Each quadrant contains one unique dominant foreground color plus a few off-color
      noise pixels, and the dominant color forms a single irregular 4-connected shape.

    Rule:
    - Find the unique solid grey row and solid grey column, and use them to define the
      UL, UR, LL, and LR quadrants.
    - In each quadrant, keep only the most frequent non-black, non-grey color and erase
      every other foreground pixel to black.
    - Mirror the cleaned dominant shape inside its own bounding box: UL and LR flip
      horizontally, while UR and LL flip vertically.
    - Keep the grey separator cross unchanged and leave all other cells black.

    Key insight:
    - The axis is determined by the quadrant's diagonal pairing, but the reflection itself
      is anchored to the dominant shape's local bounding box rather than to the whole quadrant.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    cross_r, cross_c, sep = _find_separator_cross(g)
    assert sep == SEP, "separator color must be grey"

    out = np.full_like(g, BG)
    out[cross_r, :] = sep
    out[:, cross_c] = sep

    axes = {"UL": "horizontal", "UR": "vertical", "LL": "vertical", "LR": "horizontal"}
    for name, (rs, cs) in _quadrants(g.shape, cross_r, cross_c).items():
        region = g[rs, cs]
        dominant = _dominant_color(region)
        clean_mask = region == dominant
        assert _is_connected(clean_mask), "dominant shape must be 4-connected"
        mirrored = _mirror_mask(clean_mask, axes[name])
        out_region = out[rs, cs]
        out_region[mirrored] = dominant

    return out


def _grow_connected_mask(h: int, w: int, area: int) -> np.ndarray:
    cells = {(random.randrange(h), random.randrange(w))}
    while len(cells) < area:
        frontier = {
            (r + dr, c + dc)
            for r, c in cells
            for dr, dc in NEIGHBORS4
            if 0 <= r + dr < h and 0 <= c + dc < w and (r + dr, c + dc) not in cells
        }
        cells.add(random.choice(tuple(frontier)))

    mask = np.zeros((h, w), dtype=bool)
    coords = np.array(sorted(cells), dtype=int)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    b = bbox(mask)
    assert b is not None
    r0, c0, r1, c1 = b
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _random_shape(max_h: int, max_w: int) -> np.ndarray | None:
    dims = [(h, w) for h in range(3, max_h + 1) for w in range(3, max_w + 1)]
    random.shuffle(dims)
    for h, w in dims:
        max_area = min(8, h * w - 1)
        if max_area < 5:
            continue
        for _ in range(80):
            area = random.randint(5, max_area)
            mask = _grow_connected_mask(h, w, area)
            mask = _trim_mask(mask)
            if mask.shape[0] < 2 or mask.shape[1] < 2:
                continue
            if not _is_connected(mask):
                continue
            if np.array_equal(mask, np.flipud(mask)) or np.array_equal(mask, np.fliplr(mask)):
                continue
            if np.array_equal(np.flipud(mask), np.fliplr(mask)):
                continue
            return mask
    return None


def _shape_with_noise(qh: int, qw: int, color: int) -> np.ndarray | None:
    max_h = min(4, qh - 2)
    max_w = min(4, qw - 2)
    if max_h < 3 or max_w < 3:
        return None

    shape = _random_shape(max_h, max_w)
    if shape is None:
        return None

    sh, sw = shape.shape
    top = random.randint(1, qh - sh - 1)
    left = random.randint(1, qw - sw - 1)

    region = np.zeros((qh, qw), dtype=int)
    shape_coords = np.argwhere(shape)
    global_shape = {(top + int(r), left + int(c)) for r, c in shape_coords}
    for r, c in global_shape:
        region[r, c] = color

    forbidden = set(global_shape)
    for r, c in global_shape:
        for dr, dc in NEIGHBORS4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < qh and 0 <= cc < qw:
                forbidden.add((rr, cc))

    candidates = [(r, c) for r in range(qh) for c in range(qw) if (r, c) not in forbidden]
    if len(candidates) < 1:
        return None

    noise_count = random.randint(1, min(3, len(candidates)))
    noise_cells = random.sample(candidates, noise_count)
    palette = random.sample(NOISE_COLORS, random.randint(1, min(3, len(NOISE_COLORS))))
    for r, c in noise_cells:
        region[r, c] = random.choice(palette)

    return region


def _expected_axis(name: str) -> str:
    return "horizontal" if name in {"UL", "LR"} else "vertical"


def _nonzero_count(region: np.ndarray) -> int:
    return int(np.count_nonzero(region != BG))


def _validate_instance(grid: np.ndarray) -> bool:
    try:
        cross_r, cross_c, sep = _find_separator_cross(grid)
        if sep != SEP:
            return False

        out = main(grid)
        if out.shape != grid.shape:
            return False

        expected_quads = _quadrants(grid.shape, cross_r, cross_c)
        if not np.all(out[cross_r, :] == SEP) or not np.all(out[:, cross_c] == SEP):
            return False
        if np.any(out[(grid == SEP) == 0] == SEP):
            return False

        for name, (rs, cs) in expected_quads.items():
            region = grid[rs, cs]
            out_region = out[rs, cs]
            dominant = _dominant_color(region)
            counts = Counter(int(v) for v in region.ravel() if int(v) not in (BG, SEP))
            if dominant not in MAIN_COLORS:
                return False
            if len([c for c in counts if c in MAIN_COLORS]) != 1:
                return False

            clean = region == dominant
            if len(components_mask(clean, connectivity=4)) != 1:
                return False
            if np.array_equal(clean, np.flipud(clean)) or np.array_equal(clean, np.fliplr(clean)):
                return False

            noise = (region != BG) & (region != SEP) & (region != dominant)
            if not (1 <= int(noise.sum()) <= 3):
                return False

            for r, c in np.argwhere(noise):
                for dr, dc in NEIGHBORS4:
                    rr, cc = int(r) + dr, int(c) + dc
                    if 0 <= rr < region.shape[0] and 0 <= cc < region.shape[1] and clean[rr, cc]:
                        return False

            axis = _expected_axis(name)
            mirrored = _mirror_mask(clean, axis)
            wrong_axis = _mirror_mask(clean, "vertical" if axis == "horizontal" else "horizontal")
            whole_region = np.fliplr(clean) if axis == "horizontal" else np.flipud(clean)

            if not np.array_equal(out_region == dominant, mirrored):
                return False
            if np.array_equal(mirrored, wrong_axis):
                return False
            if np.array_equal(mirrored, clean):
                return False
            if np.array_equal(out_region == dominant, whole_region):
                return False
            if _nonzero_count(out_region) >= _nonzero_count(region):
                return False
            if np.any((out_region != BG) & (out_region != dominant)):
                return False

            dom_bbox = bbox(clean)
            if dom_bbox is None:
                return False
            r0, c0, r1, c1 = dom_bbox
            if r0 == 0 or c0 == 0 or r1 == region.shape[0] - 1 or c1 == region.shape[1] - 1:
                return False

        return not np.array_equal(grid, out)
    except AssertionError:
        return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(14, 20)
    w = random.randint(14, 20)
    cross_r = random.randint(5, h - 6)
    cross_c = random.randint(5, w - 6)
    grid = np.zeros((h, w), dtype=int)
    grid[cross_r, :] = SEP
    grid[:, cross_c] = SEP

    colors = random.sample(MAIN_COLORS, 4)
    quadrants = list(_quadrants(grid.shape, cross_r, cross_c).items())
    for (name, (rs, cs)), color in zip(quadrants, colors):
        qh = rs.stop - rs.start
        qw = cs.stop - cs.start
        region = _shape_with_noise(qh, qw, color)
        if region is None:
            return None
        grid[rs, cs] = region

    return grid if _validate_instance(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
