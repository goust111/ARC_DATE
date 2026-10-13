"""
ARC unified helper library (single file, canonical APIs).

This module is the single shared helper surface that *all* solver examples should
use (ARC-2 and migrated ARC-1 seeds), so they can appear together in one prompt
without API confusion.

## Conventions (non-negotiable)
- Grid shape: `grid.shape == (H, W)` meaning `(rows, cols)`.
- Indexing: always `grid[r, c]`.
- Coords: always `(r, c)`.
- BBox: `(r0, c0, r1, c1)` inclusive.
"""

from __future__ import annotations

import math
import random
import itertools
import warnings
from collections import Counter
from typing import Iterable, Iterator, Sequence

import numpy as np
from scipy.ndimage import label
from scipy import ndimage

# =============================================================================
# Colors
# =============================================================================


class Color:
    """ARC palette color ids (ints, not Enum)."""

    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    GRAY = 5
    PINK = 6
    ORANGE = 7
    TEAL = 8
    MAROON = 9

    WHITE = 10  # debug / visualization
    PAD = 11

    BACKGROUND = 0

    ALL_COLORS = [BLACK, BLUE, RED, GREEN, YELLOW, GREY, PINK, ORANGE, TEAL, MAROON]
    NOT_BLACK = [BLUE, RED, GREEN, YELLOW, GREY, PINK, ORANGE, TEAL, MAROON]


ARC_RGB = np.array(
    [
        (0x00, 0x00, 0x00),  # 0 black
        (0x00, 0x74, 0xD9),  # 1 blue
        (0xFF, 0x41, 0x36),  # 2 red
        (0x2E, 0xCC, 0x40),  # 3 green
        (0xFF, 0xDC, 0x00),  # 4 yellow
        (0xA0, 0xA0, 0xA0),  # 5 grey
        (0xF0, 0x12, 0xBE),  # 6 magenta/pink
        (0xFF, 0x85, 0x1B),  # 7 orange
        (0x7F, 0xDB, 0xFF),  # 8 teal/azure
        (0x87, 0x0C, 0x25),  # 9 maroon
        (0xFF, 0xFF, 0xFF),  # 10 white
        (0x00, 0x00, 0x00),  # 11 pad fallback
    ],
    dtype=np.uint8,
)


# =============================================================================
# Types + basics
# =============================================================================

BBox = tuple[int, int, int, int]  # (r0,c0,r1,c1) inclusive
Coord = tuple[int, int]  # (r,c)


def get_bg(grid: np.ndarray) -> int:
    """Background color = most common value in the grid."""
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude: Iterable[int] | None = None) -> list[int]:
    """
    Return the sorted unique colors present in `grid`.

    Args:
        exclude: optional iterable of colors to exclude (e.g. `{bg}`)
    """
    cols = set(map(int, grid.flatten()))
    if exclude is not None:
        cols -= set(map(int, exclude))
    return sorted(cols)


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    """Return True iff (r,c) is within a grid of shape (H,W)."""
    H, W = map(int, shape)
    return 0 <= int(r) < H and 0 <= int(c) < W


def neighbors4(r: int, c: int) -> Iterator[Coord]:
    """4-neighborhood of (r,c)."""
    r = int(r)
    c = int(c)
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def neighbors8(r: int, c: int) -> Iterator[Coord]:
    """8-neighborhood of (r,c)."""
    r = int(r)
    c = int(c)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc






# =============================================================================
# Scaling
# =============================================================================


def scale_sprite(sprite: np.ndarray, factor: int) -> np.ndarray:
    """
    Scale a sprite by an integer factor using nearest-neighbor replication.

    Returns an array of shape (h*factor, w*factor).
    """
    k = int(factor)
    if k <= 0:
        raise ValueError("scale_sprite: factor must be >= 1")
    return np.kron(sprite, np.ones((k, k), dtype=sprite.dtype))


# =============================================================================
# Internal helpers (for legacy keyword aliases during migration)
# =============================================================================


def _consume_alias(kwargs: dict, primary: str, aliases: Sequence[str]):
    """
    Consume a keyword argument by primary name or any alias, ensuring no conflicts.

    Returns:
        (value, used_name) or (None, None) if not found.
    """
    used = None
    if primary in kwargs:
        used = primary
    else:
        for a in aliases:
            if a in kwargs:
                used = a
                break
    if used is None:
        return None, None
    val = kwargs.pop(used)
    names = [primary, *aliases]
    for n in names:
        if n != used and n in kwargs:
            raise TypeError(f"got multiple values for {primary}: {used} and {n}")
    return val, used


# =============================================================================
# Masks / coords
# =============================================================================


def coords_where(mask: np.ndarray) -> np.ndarray:
    """Return coords where mask is True as an (N,2) array of (r,c)."""
    return np.argwhere(mask)




# =============================================================================
# BBox
# =============================================================================


def bbox_coords(coords: np.ndarray) -> BBox:
    """BBox of an (N,2) coords array (r,c). Returns (r0,c0,r1,c1) inclusive."""
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def bbox(mask: np.ndarray) -> BBox | None:
    """
    Bounding box (bbox) of a boolean mask.

    Args:
        mask: boolean array where True indicates foreground.

    Returns:
        (r0,c0,r1,c1) inclusive, or None if the mask has no True values.
    """
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return bbox_coords(coords)


def bbox_mask(mask: np.ndarray) -> np.ndarray:
    """
    Return a boolean mask that is True exactly inside the bbox of `mask`.

    This corresponds to `bounding_box_mask(...)` in the legacy seeds API.
    If `mask` is empty, returns an all-False mask.
    """
    out = np.zeros_like(mask, dtype=bool)
    b = bbox(mask)
    if b is None:
        return out
    r0, c0, r1, c1 = b
    out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = True
    return out


def bbox_union(bboxes: Iterable[BBox]) -> BBox | None:
    """Union of bboxes, or None for empty input."""
    bbs = list(bboxes)
    if not bbs:
        return None
    r0 = min(b[0] for b in bbs)
    c0 = min(b[1] for b in bbs)
    r1 = max(b[2] for b in bbs)
    c1 = max(b[3] for b in bbs)
    return int(r0), int(c0), int(r1), int(c1)


def bbox_h(b: BBox) -> int:
    r0, _c0, r1, _c1 = b
    return int(r1 - r0 + 1)


def bbox_w(b: BBox) -> int:
    _r0, c0, _r1, c1 = b
    return int(c1 - c0 + 1)


def bbox_anchor(b: BBox, *, anchor: str = "upper left") -> Coord:
    """
    Return an anchor point (r,c) for a bbox.

    anchor:
        upper left/right, lower left/right,
        center,
        upper center, lower center,
        left center, right center
    """
    anchor = anchor.lower().replace(" ", "").replace("top", "upper").replace("bottom", "lower")
    r0, c0, r1, c1 = map(int, b)
    h = bbox_h(b)
    w = bbox_w(b)

    if anchor == "upperleft":
        rr, cc = r0, c0
    elif anchor == "upperright":
        rr, cc = r0, c1
    elif anchor == "lowerleft":
        rr, cc = r1, c0
    elif anchor == "lowerright":
        rr, cc = r1, c1
    elif anchor == "center":
        rr, cc = r0 + (h - 1) / 2, c0 + (w - 1) / 2
    elif anchor == "uppercenter":
        rr, cc = r0, c0 + (w - 1) / 2
    elif anchor == "lowercenter":
        rr, cc = r1, c0 + (w - 1) / 2
    elif anchor == "leftcenter":
        rr, cc = r0 + (h - 1) / 2, c0
    elif anchor == "rightcenter":
        rr, cc = r0 + (h - 1) / 2, c1
    else:
        raise ValueError("bbox_anchor: invalid anchor")

    return int(round(rr)), int(round(cc))


def crop_bbox(grid: np.ndarray, bbox: BBox) -> np.ndarray:
    """Crop a grid to bbox (inclusive)."""
    r0, c0, r1, c1 = map(int, bbox)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


# =============================================================================
# Paste (single canonical placement primitive)
# =============================================================================


def paste(
    dst: np.ndarray,
    patch: np.ndarray,
    r0: int,
    c0: int,
    *,
    transparent: int | None = None,
    mask: np.ndarray | None = None,
    overwrite: bool = True,
    clip: bool = False,
) -> np.ndarray:
    """
    Paste `patch` into `dst` at top-left (r0,c0).

    This is the ONE placement primitive solver examples should use.

    - Opaque paste: `paste(dst, patch, r0, c0)`
    - Transparent overlay: `paste(dst, patch, r0, c0, transparent=bg)`
    - Masked paste: `paste(dst, patch, r0, c0, mask=m)`
    - Non-overwriting overlay: `paste(dst, patch, r0, c0, transparent=bg, overwrite=False)`
    - Strict vs clip: default `clip=False` (fail fast); set `clip=True` to clip out-of-bounds.
    """
    H, W = dst.shape
    r0 = int(r0)
    c0 = int(c0)
    h, w = map(int, patch.shape)

    if not clip:
        if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
            raise ValueError("paste: patch out of bounds (set clip=True to allow clipping)")
        rr0, cc0 = r0, c0
        rr1, cc1 = r0 + h, c0 + w
        pr0, pc0 = 0, 0
    else:
        rr0 = max(0, r0)
        cc0 = max(0, c0)
        rr1 = min(H, r0 + h)
        cc1 = min(W, c0 + w)
        if rr0 >= rr1 or cc0 >= cc1:
            return dst
        pr0 = rr0 - r0
        pc0 = cc0 - c0

    pr1 = pr0 + (rr1 - rr0)
    pc1 = pc0 + (cc1 - cc0)

    dst_view = dst[rr0:rr1, cc0:cc1]
    patch_view = patch[pr0:pr1, pc0:pc1]

    m = np.ones(patch_view.shape, dtype=bool)
    if mask is not None:
        m &= mask[pr0:pr1, pc0:pc1].astype(bool)
    if transparent is not None:
        m &= (patch_view != int(transparent))
        if not overwrite:
            m &= (dst_view == int(transparent))
    else:
        if not overwrite:
            raise ValueError("paste: overwrite=False requires transparent to be set")

    dst_view[m] = patch_view[m]
    return dst


def paint_coords(out: np.ndarray, coords: Iterable[Coord], color: int) -> None:
    """Set out[r,c]=color for each coord (in-bounds only)."""
    H, W = out.shape
    v = int(color)
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if 0 <= rr < H and 0 <= cc < W:
            out[rr, cc] = v


# =============================================================================
# Connected components (single unified entrypoint: find_components)
# =============================================================================

STRUCT4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
STRUCT8 = np.ones((3, 3), dtype=int)


def label_mask(mask: np.ndarray, connectivity: int = 4) -> tuple[np.ndarray, int]:
    """Connected-components labeling for a boolean mask."""
    struct = STRUCT4 if int(connectivity) == 4 else STRUCT8
    lab, n = label(mask.astype(bool), structure=struct)
    return lab, int(n)


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    """List of components from a boolean mask, each as an (N,2) coords array."""
    lab, n = label_mask(mask, connectivity=connectivity)
    return [np.argwhere(lab == k) for k in range(1, n + 1)]


def find_components(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    connectivity: int = 4,
    monochromatic: bool = True,
    colors: Sequence[int] | None = None,
    min_area: int = 1,
    return_format: str = "dict",
) -> list[dict] | list[np.ndarray]:
    """
    Find connected components in a grid (unified API for ARC-2 + migrated seeds).

    return_format:
      - \"dict\": list of dicts {color, coords, bbox, area}
      - \"objects\": list of full-grid arrays containing exactly one component
    """
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    return_format = str(return_format).lower().strip()
    if return_format not in {"dict", "objects"}:
        raise ValueError("find_components: return_format must be 'dict' or 'objects'")

    if colors is None:
        colors_set = None
        colors_iter = sorted(set(map(int, grid.flatten())) - {bg})
    else:
        colors_set = {int(c) for c in colors}
        colors_iter = [int(c) for c in colors_set if int(c) != bg]

    # Monochromatic = label per-color
    if monochromatic:
        if return_format == "dict":
            out: list[dict] = []
            for col in colors_iter:
                mask = (grid == col)
                if not mask.any():
                    continue
                comps = components_mask(mask, connectivity=connectivity)
                for coords in comps:
                    if int(coords.shape[0]) < int(min_area):
                        continue
                    out.append(
                        {
                            "color": int(col),
                            "coords": coords,
                            "bbox": bbox_coords(coords),
                            "area": int(coords.shape[0]),
                        }
                    )
            return out

        objs: list[np.ndarray] = []
        struct = STRUCT4 if int(connectivity) == 4 else STRUCT8
        for col in colors_iter:
            lab, n = label(grid == int(col), structure=struct)
            for k in range(1, int(n) + 1):
                m = (lab == k)
                if int(m.sum()) < int(min_area):
                    continue
                obj = np.full_like(grid, bg)
                obj[m] = int(col)
                objs.append(obj)
        return objs

    # Multicolor components = label foreground as a whole
    struct = STRUCT4 if int(connectivity) == 4 else STRUCT8
    if colors_set is None:
        fg_mask = (grid != bg)
    else:
        fg_mask = (grid != bg) & np.isin(grid, list(colors_set))

    lab, n = label(fg_mask, structure=struct)
    if return_format == "dict":
        out: list[dict] = []
        for k in range(1, int(n) + 1):
            coords = np.argwhere(lab == k)
            if coords.size == 0 or int(coords.shape[0]) < int(min_area):
                continue
            out.append(
                {
                    "color": None,
                    "coords": coords,
                    "bbox": bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
        return out

    objs: list[np.ndarray] = []
    for k in range(1, int(n) + 1):
        m = (lab == k)
        if int(m.sum()) < int(min_area):
            continue
        obj = np.full_like(grid, bg)
        obj[m] = grid[m]
        objs.append(obj)
    return objs


# =============================================================================
# Geometry
# =============================================================================


def flood_fill(grid: np.ndarray, r: int, c: int, color: int, *, connectivity: int = 4) -> None:
    """Flood fill region containing (r,c) with `color` (in-place)."""
    H, W = grid.shape
    r0 = int(r)
    c0 = int(c)
    if not (0 <= r0 < H and 0 <= c0 < W):
        return
    new_col = int(color)
    old_col = int(grid[r0, c0])
    if old_col == new_col:
        return
    if int(connectivity) == 4:
        def nbrs(rr: int, cc: int):
            yield rr - 1, cc
            yield rr + 1, cc
            yield rr, cc - 1
            yield rr, cc + 1
    else:
        def nbrs(rr: int, cc: int):
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    yield rr + dr, cc + dc
    stack: list[Coord] = [(r0, c0)]
    while stack:
        rr, cc = stack.pop()
        if not (0 <= rr < H and 0 <= cc < W):
            continue
        if int(grid[rr, cc]) != old_col:
            continue
        grid[rr, cc] = new_col
        for nr, nc in nbrs(rr, cc):
            stack.append((nr, nc))


def draw_line(
    grid: np.ndarray,
    r: int,
    c: int,
    *,
    end_r: int | None = None,
    end_c: int | None = None,
    length: int | None = None,
    direction: tuple[int, int] | None = None,
    color: int,
    stop_at: set[int] | None = None,
    clip: bool = False,
) -> Coord:
    """
    Draw a line starting at (r,c) either to an endpoint or by (length,direction).
    Modifies grid in-place. Returns (r_end,c_end) last drawn.

    Compatibility note (seed helpers):
      - If you provide `direction` and leave `length=None`, the line extends until
        it would leave the grid (implemented by using a sufficiently large length).
    """
    rr = int(r)
    cc = int(c)
    H, W = grid.shape
    col = int(color)
    stop_at = set() if stop_at is None else set(map(int, stop_at))

    if (end_r is None) != (end_c is None):
        raise ValueError("draw_line: either provide both end_r and end_c, or neither")

    if end_r is not None and end_c is not None:
        er = int(end_r)
        ec = int(end_c)
        dr = er - rr
        dc = ec - cc
        length = max(abs(dr), abs(dc)) + 1
        dr = 0 if dr == 0 else int(dr / abs(dr))
        dc = 0 if dc == 0 else int(dc / abs(dc))
    else:
        if direction is None:
            raise ValueError("draw_line: must provide endpoint or direction")
        dr, dc = map(int, direction)
        dr = 0 if dr == 0 else int(dr / abs(dr))
        dc = 0 if dc == 0 else int(dc / abs(dc))
        if length is None:
            # Extend until the edge (seed-style behavior). This uses a bound that is
            # certainly long enough even if the start is out-of-bounds and clip=True.
            length = max(H, W) * 2
        length = int(length)

    last_r, last_c = rr, cc
    for i in range(int(length)):
        r2 = rr + i * dr
        c2 = cc + i * dc
        if not (0 <= r2 < H and 0 <= c2 < W):
            if bool(clip):
                continue
            break
        if int(grid[r2, c2]) in stop_at:
            break
        grid[r2, c2] = col
        last_r, last_c = r2, c2
    return int(last_r), int(last_c)


# =============================================================================
# Tiling
# =============================================================================


def tile_fill(shape: tuple[int, int], tile: np.ndarray, *, offset: Coord = (0, 0)) -> np.ndarray:
    """Return a new grid of `shape` filled by repeating `tile` with a phase offset."""
    H, W = map(int, shape)
    h, w = tile.shape
    r_off, c_off = map(int, offset)
    out = np.empty((H, W), dtype=tile.dtype)
    for r in range(H):
        tr = (r - r_off) % h
        for c in range(W):
            tc = (c - c_off) % w
            out[r, c] = tile[tr, tc]
    return out




# =============================================================================
# Motion / interaction (translate, collision/contact)
# =============================================================================


def translate(obj: np.ndarray, dr: int, dc: int, *, background: int = Color.BLACK) -> np.ndarray:
    """Translate a full-grid object by (dr,dc), filling empty with background."""
    dr = int(dr)
    dc = int(dc)
    bg = int(background)
    H, W = obj.shape
    out = np.full_like(obj, bg)
    rr, cc = np.nonzero(obj != bg)
    rr2 = rr + dr
    cc2 = cc + dc
    inb = (0 <= rr2) & (rr2 < H) & (0 <= cc2) & (cc2 < W)
    out[rr2[inb], cc2[inb]] = obj[rr[inb], cc[inb]]
    return out


def collision(
    *,
    object1: np.ndarray,
    object2: np.ndarray,
    r1: int = 0,
    c1: int = 0,
    r2: int = 0,
    c2: int = 0,
    background: int = Color.BLACK,
) -> bool:
    """Return True iff non-background pixels overlap with object2 placed at (r2,c2) relative to object1 at (r1,c1)."""
    bg = int(background)
    a = object1
    b = object2
    H1, W1 = a.shape
    H2, W2 = b.shape

    dr = int(r2) - int(r1)
    dc = int(c2) - int(c1)

    ra0 = max(0, dr)
    ca0 = max(0, dc)
    ra1 = min(H1, dr + H2)
    ca1 = min(W1, dc + W2)
    if ra0 >= ra1 or ca0 >= ca1:
        return False

    rb0 = ra0 - dr
    cb0 = ca0 - dc
    rb1 = rb0 + (ra1 - ra0)
    cb1 = cb0 + (ca1 - ca0)
    return bool(((a[ra0:ra1, ca0:ca1] != bg) & (b[rb0:rb1, cb0:cb1] != bg)).any())


def contact(
    *,
    object1: np.ndarray,
    object2: np.ndarray,
    r1: int = 0,
    c1: int = 0,
    r2: int = 0,
    c2: int = 0,
    background: int = Color.BLACK,
    connectivity: int = 4,
) -> bool:
    """Return True iff objects touch (share edge/diagonal) or overlap."""
    bg = int(background)
    struct = STRUCT4 if int(connectivity) == 4 else STRUCT8
    m2 = (object2 != bg)
    m2d = ndimage.binary_dilation(m2, structure=struct)
    return collision(
        object1=(object1 != bg).astype(int),
        object2=m2d.astype(int),
        r1=r1,
        c1=c1,
        r2=r2,
        c2=c2,
        background=0,
    )


# =============================================================================
# Generation helpers
# =============================================================================

def first_free_location(
    canvas: np.ndarray,
    patch: np.ndarray,
    *,
    transparent: int,
    border: int = 0,
    padding: int = 0,
    padding_connectivity: int = 8,
) -> Coord | None:
    """
    Find the first (top-left) location where `patch` fits without overlap.

    Args:
        transparent: color treated as transparent in both `canvas` and `patch`
        border: keep at least this many empty cells to the canvas border
        padding: if > 0, treat `canvas` occupancy as dilated by `padding` pixels.
                 (useful for “leave a gap” placement constraints)
        padding_connectivity: 4 or 8 for the dilation neighborhood

    Returns:
        (r0,c0) or None if no placement exists.
    """
    H, W = canvas.shape
    h, w = patch.shape
    border = int(border)
    padding = int(padding)
    bg = int(transparent)

    if h <= 0 or w <= 0:
        return None
    if H - h + 1 - border <= border or W - w + 1 - border <= border:
        return None

    # Occupancy of the canvas (optionally dilated to enforce padding).
    occ = (canvas != bg)
    if padding > 0:
        if int(padding_connectivity) == 4:
            struct = STRUCT4
        elif int(padding_connectivity) == 8:
            struct = STRUCT8
        else:
            raise ValueError("first_free_location: padding_connectivity must be 4 or 8")
        occ = ndimage.binary_dilation(occ, iterations=padding, structure=struct)

    patch_mask = (patch != bg)
    if not bool(patch_mask.any()):
        # Empty patch: any in-bounds location works; choose first.
        return int(border), int(border)

    for r0 in range(border, H - h + 1 - border):
        for c0 in range(border, W - w + 1 - border):
            if not bool((occ[r0 : r0 + h, c0 : c0 + w] & patch_mask).any()):
                return int(r0), int(c0)
    return None


def random_sprite(
    m,
    n,
    *,
    density: float = 0.5,
    symmetry: str | None = None,
    color_palette: list[int] | None = None,
    connectivity: int = 4,
    background: int = Color.BLACK,
    no_repeated_colors: bool = False,
) -> np.ndarray:
    """Generate a contiguous sprite of shape (m,n) with random symmetry and palette (for puzzle generation)."""
    if isinstance(m, range):
        m = list(m)
    if isinstance(n, range):
        n = list(n)
    if not isinstance(m, list):
        m = [m]
    if not isinstance(n, list):
        n = [n]

    m_original, n_original = m, n
    density_original, symmetry_original = density, symmetry
    palette_original, conn_original, bg_original = color_palette, connectivity, background

    can_be_square = any(m_ == n_ for m_ in m for n_ in n)
    symmetry_types = ["horizontal", "vertical", "not_symmetric", "mirror"]
    if can_be_square:
        symmetry_types += ["diagonal", "radial"]
    symmetry = symmetry or random.choice(symmetry_types)

    has_to_be_square = symmetry in ("diagonal", "radial")
    if has_to_be_square:
        m, n = random.choice([(m_, n_) for m_ in m_original for n_ in n_original if m_ == n_])
    else:
        m = random.choice(m)
        n = random.choice(n)

    if m == 1 or n == 1:
        density = 1.0
    elif m == 2 or n == 2:
        density = max(float(density), 0.6)
    elif m == 3 or n == 3:
        density = max(float(density), 0.5)
    else:
        density = max(0.4, min(0.95, random.gauss(float(density), 0.1)))

    sprite = generate_sprite(
        n=int(m),
        m=int(n),
        symmetry_type=symmetry,
        color_palate=color_palette,
        fill_percentage=density,
        connectivity=connectivity,
        background=background,
        no_repeated_colors=no_repeated_colors,
    )
    assert is_contiguous(sprite, connectivity=connectivity, background=background), "Generated sprite is not contiguous."

    bg = int(background)
    if (
        np.sum(sprite[0, :] != bg) > 0
        and np.sum(sprite[-1, :] != bg) > 0
        and np.sum(sprite[:, 0] != bg) > 0
        and np.sum(sprite[:, -1] != bg) > 0
    ):
        return sprite

    return random_sprite(
        m_original,
        n_original,
        density=density_original,
        symmetry=symmetry_original,
        color_palette=palette_original,
        connectivity=conn_original,
        background=bg_original,
        no_repeated_colors=no_repeated_colors,
    )


def random_free_location(
    canvas: np.ndarray,
    patch: np.ndarray,
    *,
    transparent: int,
    border: int = 0,
    padding: int = 0,
    padding_connectivity: int = 8,
) -> Coord:
    """
    Return a random (r,c) top-left location where `patch` fits without overlap.

    This is the generator-friendly counterpart of `first_free_location` and mirrors the
    legacy seeds' `random_free_location_for_sprite` behavior.

    Raises:
        ValueError: if no free location exists.
    """
    H, W = canvas.shape
    h, w = patch.shape
    border = int(border)
    padding = int(padding)
    pad_conn = int(padding_connectivity)
    bg = int(transparent)

    if h <= 0 or w <= 0:
        raise ValueError("random_free_location: patch must be non-empty")
    if H <= 0 or W <= 0:
        raise ValueError("random_free_location: canvas must be non-empty")

    if border < 0:
        raise ValueError("random_free_location: border must be >= 0")
    if padding < 0:
        raise ValueError("random_free_location: padding must be >= 0")
    if pad_conn not in (4, 8):
        raise ValueError("random_free_location: padding_connectivity must be 4 or 8")

    r_min = border
    c_min = border
    r_max = H - border - h
    c_max = W - border - w
    if r_max < r_min or c_max < c_min:
        raise ValueError("random_free_location: patch cannot fit (border too large or patch too big)")

    occ = (canvas != bg)
    if padding > 0:
        struct = STRUCT4 if pad_conn == 4 else STRUCT8
        occ = ndimage.binary_dilation(occ, structure=struct, iterations=padding)

    m = (patch != bg)

    candidates: list[Coord] = []
    for r0 in range(r_min, r_max + 1):
        for c0 in range(c_min, c_max + 1):
            if not bool((occ[r0 : r0 + h, c0 : c0 + w] & m).any()):
                candidates.append((int(r0), int(c0)))

    if not candidates:
        raise ValueError("random_free_location: no free location")
    return random.choice(candidates)


def randomly_scatter_points(
    grid: np.ndarray,
    color: int,
    *,
    density: float = 0.5,
    background: int = Color.BLACK,
) -> np.ndarray:
    """Randomly scatter points of `color` into `background` cells until reaching density."""
    bg = int(background)
    col = int(color)
    colored = 0
    H, W = grid.shape
    target = int(round(float(density) * H * W))
    while colored < target:
        r = np.random.randint(0, H)
        c = np.random.randint(0, W)
        if int(grid[r, c]) == bg:
            grid[r, c] = col
            colored += 1
    return grid


def randomly_spaced_indices(max_len: int, n_indices: int, border_size: int = 1, padding: int = 1) -> np.ndarray:
    """Generate randomly-spaced indices guaranteed to not be adjacent."""
    max_len = int(max_len)
    n_indices = int(n_indices)
    border_size = int(border_size)
    padding = int(padding)

    if border_size > 0:
        return (
            randomly_spaced_indices(max_len - border_size - 2, n_indices, border_size=0, padding=padding) + border_size
        )

    indices = [0 for _ in range(max_len)]
    while sum(indices) < n_indices:
        possible = [
            i
            for i in range(max_len)
            if sum(indices[max(0, i - padding) : min(i + 1 + padding, max_len)]) == 0
        ]
        if not possible:
            raise ValueError("randomly_spaced_indices: no feasible placement")
        indices[random.choice(possible)] = 1
    return np.argwhere(indices).flatten()

# =============================================================================
# Misc 
# =============================================================================
# These are useful for some seed-style solvers, but they are
# still part of the *canonical* API surface (no duplicate synonyms like blit vs paste).

def check_between_objects(
    obj1: np.ndarray,
    obj2: np.ndarray,
    r: int | None = None,
    c: int | None = None,
    padding: int = 0,
    background: int = Color.BLACK,
    **kwargs,
) -> bool:
    """Check if a pixel (r,c) lies between two objects, horizontally or vertically."""
    xr, used_x = _consume_alias(kwargs, "r", ["x"])
    yc, used_y = _consume_alias(kwargs, "c", ["y"])
    if r is None and xr is not None:
        if used_x == "x":
            warnings.warn("check_between_objects: keyword 'x' is deprecated; use 'r'", DeprecationWarning, stacklevel=2)
        r = xr
    if c is None and yc is not None:
        if used_y == "y":
            warnings.warn("check_between_objects: keyword 'y' is deprecated; use 'c'", DeprecationWarning, stacklevel=2)
        c = yc
    if r is None or c is None:
        raise TypeError("check_between_objects: must provide r,c (or x,y)")
    if kwargs:
        raise TypeError(f"check_between_objects: unexpected keyword arguments: {sorted(kwargs.keys())}")

    r = int(r)
    c = int(c)
    pad = int(padding)
    bg = int(background)

    b1 = bbox(obj1 != bg)
    b2 = bbox(obj2 != bg)
    if b1 is None or b2 is None:
        return False
    r1, c1, r1b, c1b = b1
    r2, c2, r2b, c2b = b2

    # Horizontal between: objects separated along columns, and row overlap is sufficient.
    left, right = ((r1, c1, r1b, c1b), (r2, c2, r2b, c2b)) if c1 <= c2 else ((r2, c2, r2b, c2b), (r1, c1, r1b, c1b))
    lr0, _lc0, lr1, lc1 = left
    rr0, rc0, rr1, _rc1 = right
    if lc1 < c < rc0:
        top = max(lr0, rr0) + pad
        bot = min(lr1, rr1) - pad
        if top <= r <= bot:
            return True

    # Vertical between: objects separated along rows, and col overlap is sufficient.
    top_obj, bot_obj = ((r1, c1, r1b, c1b), (r2, c2, r2b, c2b)) if r1 <= r2 else ((r2, c2, r2b, c2b), (r1, c1, r1b, c1b))
    tr0, tc0, tr1, tc1 = top_obj
    br0, bc0, br1, bc1 = bot_obj
    if tr1 < r < br0:
        leftc = max(tc0, bc0) + pad
        rightc = min(tc1, bc1) - pad
        if leftc <= c <= rightc:
            return True

    return False


def object_interior(grid: np.ndarray, background: int = Color.BLACK) -> np.ndarray:
    """Return a boolean mask of the object's interior (including boundary)."""
    bg = int(background)
    mask = 1 * (grid != bg)
    H, W = grid.shape

    # Flood fill exterior (mark as 42) starting from any background pixel on the border.
    for r in range(H):
        if int(grid[r, 0]) == bg:
            flood_fill(mask, r, 0, 42)
        if int(grid[r, W - 1]) == bg:
            flood_fill(mask, r, W - 1, 42)
    for c in range(W):
        if int(grid[0, c]) == bg:
            flood_fill(mask, 0, c, 42)
        if int(grid[H - 1, c]) == bg:
            flood_fill(mask, H - 1, c, 42)

    return mask != 42


def object_boundary(grid: np.ndarray, background: int = Color.BLACK) -> np.ndarray:
    """Return a boolean mask of the object's boundary (excluding interior holes)."""
    bg = int(background)
    exterior = ~object_interior(grid, background=bg)

    canvas_boundary = np.zeros_like(grid, dtype=bool)
    canvas_boundary[0, :] = True
    canvas_boundary[-1, :] = True
    canvas_boundary[:, 0] = True
    canvas_boundary[:, -1] = True

    adjacent_to_exterior = ndimage.binary_dilation(exterior, iterations=1)
    return (grid != bg) & (adjacent_to_exterior | canvas_boundary)


def object_neighbors(grid: np.ndarray, background: int = Color.BLACK, connectivity: int = 4) -> np.ndarray:
    """Mask of pixels that neighbor the object (but are not part of it)."""
    bg = int(background)
    boundary = object_boundary(grid, background=bg)
    if int(connectivity) == 4:
        struct = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    elif int(connectivity) == 8:
        struct = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    else:
        raise ValueError("Connectivity must be 4 or 8.")

    neigh = ndimage.binary_dilation(boundary, structure=struct)
    return neigh & (grid == bg)


# =============================================================================
# Symmetry + sprite generation (ported from seeds/common.py, but canonical r/c semantics)
# =============================================================================


class Symmetry:
    """
    Symmetry transformations.

    Subclasses implement:
        apply(r,c,iters=1) -> (r',c')
        _iter_range(shape) -> (start, stop) range tuple, used by orbit()
    """

    def apply(self, r, c, iters: int = 1):
        raise NotImplementedError

    def _iter_range(self, grid_shape: tuple[int, int]):
        raise NotImplementedError


def orbit(grid: np.ndarray, r: int, c: int, symmetries: Sequence[Symmetry]) -> list[tuple[int, int]]:
    """Compute the orbit of (r,c) under the provided symmetries."""
    all_possible: list[tuple[int, int]] = []
    possible_iterations = itertools.product(*[list(range(*s._iter_range(grid.shape))) for s in symmetries])
    for iters in possible_iterations:
        rr, cc = int(r), int(c)
        for sym, i in zip(symmetries, iters):
            rr, cc = sym.apply(rr, cc, i)
            if not (0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]):
                break
        else:
            all_possible.append((int(rr), int(cc)))
    return list(set(all_possible))


class TranslationalSymmetry(Symmetry):
    """Translation symmetry by a fixed vector (dr,dc)."""

    def __init__(self, dr: int | None = None, dc: int | None = None, **kwargs):
        # Legacy keyword aliases (deprecated): translate_x->dr, translate_y->dc
        tx, used_tx = _consume_alias(kwargs, "dr", ["translate_x"])
        ty, used_ty = _consume_alias(kwargs, "dc", ["translate_y"])
        if dr is None and tx is not None:
            if used_tx == "translate_x":
                warnings.warn("TranslationalSymmetry: 'translate_x' is deprecated; use 'dr'", DeprecationWarning, stacklevel=2)
            dr = tx
        if dc is None and ty is not None:
            if used_ty == "translate_y":
                warnings.warn("TranslationalSymmetry: 'translate_y' is deprecated; use 'dc'", DeprecationWarning, stacklevel=2)
            dc = ty
        if dr is None or dc is None:
            raise TypeError("TranslationalSymmetry: must provide dr,dc")
        if kwargs:
            raise TypeError(f"TranslationalSymmetry: unexpected keyword arguments: {sorted(kwargs.keys())}")
        self.dr, self.dc = int(dr), int(dc)

    def apply(self, r, c, iters: int = 1):
        r = r + int(iters) * self.dr
        c = c + int(iters) * self.dc
        if isinstance(r, np.ndarray):
            r = r.astype(int)
        if isinstance(c, np.ndarray):
            c = c.astype(int)
        if isinstance(r, float):
            r = int(round(r))
        if isinstance(c, float):
            c = int(round(c))
        return r, c

    def __repr__(self):
        return f"TranslationalSymmetry(dr={self.dr}, dc={self.dc})"

    def __str__(self):
        return self.__repr__()

    def _iter_range(self, grid_shape: tuple[int, int]):
        top = 0
        if self.dr != 0:
            top = math.ceil(grid_shape[0] / abs(self.dr))
        if self.dc != 0:
            top = max(top, math.ceil(grid_shape[1] / abs(self.dc)))
        return (-top, top + 1)


def detect_translational_symmetry(
    grid: np.ndarray,
    ignore_colors: list[int] | None = None,
    background: int | None = None,
) -> list[Symmetry]:
    """Detect translational symmetries using a scoring heuristic."""
    if ignore_colors is None:
        ignore_colors = [Color.BLACK]
    ignore_colors = [int(c) for c in ignore_colors]

    n, m = grid.shape
    r_poss = [TranslationalSymmetry(dr=tr, dc=0) for tr in range(1, n)]
    r_poss += [TranslationalSymmetry(dr=-tr, dc=0) for tr in range(1, n)]
    c_poss = [TranslationalSymmetry(dr=0, dc=tc) for tc in range(1, m)]
    c_poss += [TranslationalSymmetry(dr=0, dc=-tc) for tc in range(1, m)]
    rc_poss = [TranslationalSymmetry(dr=tr, dc=tc) for tr in range(1, n) for tc in range(1, m)]

    def score(sym: Symmetry) -> float:
        perfectly_preserved, outside_canvas, conflict = _score_symmetry(grid, sym, ignore_colors, background=background)
        return perfectly_preserved - 0.01 * outside_canvas - 100000 * conflict

    r_scores = [score(s) for s in r_poss]
    c_scores = [score(s) for s in c_poss]
    rc_scores = [score(s) for s in rc_poss]

    r_poss = [(r_poss[i], r_scores[i]) for i in range(len(r_poss)) if r_scores[i] > 0]
    c_poss = [(c_poss[i], c_scores[i]) for i in range(len(c_poss)) if c_scores[i] > 0]
    rc_poss = [(rc_poss[i], rc_scores[i]) for i in range(len(rc_poss)) if rc_scores[i] > 0]

    detections: list[Symmetry] = []
    if r_poss:
        detections.append(max(r_poss, key=lambda t: (t[1], -abs(t[0].dr)))[0])
    if c_poss:
        detections.append(max(c_poss, key=lambda t: (t[1], -abs(t[0].dc)))[0])
    if not detections and rc_poss:
        detections.append(max(rc_poss, key=lambda t: (t[1], -abs(t[0].dr) - abs(t[0].dc)))[0])
    return detections


class MirrorSymmetry(Symmetry):
    """Mirror symmetry across horizontal and/or vertical axis (in pixel coordinates)."""

    def __init__(self, mirror_r: float | None, mirror_c: float | None, **kwargs):
        # Legacy keyword aliases (deprecated): mirror_x->mirror_r, mirror_y->mirror_c
        mx, used_mx = _consume_alias(kwargs, "mirror_r", ["mirror_x"])
        my, used_my = _consume_alias(kwargs, "mirror_c", ["mirror_y"])
        if mirror_r is None and mx is not None:
            if used_mx == "mirror_x":
                warnings.warn("MirrorSymmetry: 'mirror_x' is deprecated; use 'mirror_r'", DeprecationWarning, stacklevel=2)
            mirror_r = mx
        if mirror_c is None and my is not None:
            if used_my == "mirror_y":
                warnings.warn("MirrorSymmetry: 'mirror_y' is deprecated; use 'mirror_c'", DeprecationWarning, stacklevel=2)
            mirror_c = my
        if kwargs:
            raise TypeError(f"MirrorSymmetry: unexpected keyword arguments: {sorted(kwargs.keys())}")
        self.mirror_r, self.mirror_c = mirror_r, mirror_c

    def apply(self, r, c, iters: int = 1):
        if int(iters) % 2 == 0:
            return r, c
        if self.mirror_r is not None:
            r = 2 * self.mirror_r - r
        if self.mirror_c is not None:
            c = 2 * self.mirror_c - c
        if isinstance(r, np.ndarray):
            r = r.astype(int)
        if isinstance(c, np.ndarray):
            c = c.astype(int)
        if isinstance(r, float):
            r = int(round(r))
        if isinstance(c, float):
            c = int(round(c))
        return r, c

    def __repr__(self):
        return f"MirrorSymmetry(mirror_r={self.mirror_r}, mirror_c={self.mirror_c})"

    def __str__(self):
        return self.__repr__()

    def _iter_range(self, grid_shape: tuple[int, int]):
        return (0, 2)


def detect_mirror_symmetry(
    grid: np.ndarray,
    ignore_colors: list[int] | None = None,
    background: int | None = None,
) -> list[Symmetry]:
    """Detect mirror symmetries. Returns a list (possibly empty)."""
    if ignore_colors is None:
        ignore_colors = [Color.BLACK]
    ignore_colors = [int(c) for c in ignore_colors]

    n, m = grid.shape
    rc_poss = [MirrorSymmetry(mirror_r=rc + z, mirror_c=cc + z) for rc in range(n) for cc in range(m) for z in (0, 0.5)]
    r_poss = [MirrorSymmetry(mirror_r=rc + z, mirror_c=None) for rc in range(n) for z in (0, 0.5)]
    c_poss = [MirrorSymmetry(mirror_r=None, mirror_c=cc + z) for cc in range(m) for z in (0, 0.5)]

    best, best_score = [], 0.0
    for sym in r_poss + c_poss + rc_poss:
        perfectly_preserved, outside_canvas, conflict = _score_symmetry(grid, sym, ignore_colors, background=background)
        score = perfectly_preserved - 0.01 * outside_canvas - 10000 * conflict
        if conflict > 0 or perfectly_preserved == 0:
            continue
        if score > best_score:
            best = [sym]
            best_score = score
        elif score == best_score:
            best.append(sym)
    return best


def detect_rotational_symmetry(
    grid: np.ndarray,
    ignore_colors: list[int] | None = None,
    background: int | None = None,
) -> Symmetry | None:
    """Detect best rotational symmetry (90-degree steps). Returns a Symmetry or None."""
    if ignore_colors is None:
        ignore_colors = [Color.BLACK]
    ignore_colors = [int(c) for c in ignore_colors]

    class RotationalSymmetry(Symmetry):
        def __init__(self, center_r: float, center_c: float):
            self.center_r, self.center_c = center_r, center_c

        def apply(self, r, c, iters: int = 1):
            r, c = r - self.center_r, c - self.center_c
            for _ in range(abs(int(iters))):
                if int(iters) >= 0:
                    r, c = c, -r
                else:
                    r, c = -c, r
            r, c = r + self.center_r, c + self.center_c
            if isinstance(r, np.ndarray):
                r = r.astype(int)
            if isinstance(c, np.ndarray):
                c = c.astype(int)
            if isinstance(r, float):
                r = int(round(r))
            if isinstance(c, float):
                c = int(round(c))
            return r, c

        def _iter_range(self, grid_shape: tuple[int, int]):
            return (0, 4)

    n, m = grid.shape
    possibilities = [RotationalSymmetry(center_r=rc + z, center_c=cc + z) for rc in range(n) for cc in range(m) for z in (0, 0.5)]

    best_sym, best_score = None, 0.0
    for sym in possibilities:
        perfectly_preserved, outside_canvas, conflict = _score_symmetry(grid, sym, ignore_colors, background=background)
        score = perfectly_preserved - 5 * outside_canvas - 1000 * conflict
        if score > best_score:
            best_sym = sym
            best_score = score
    return best_sym


def _score_symmetry(
    grid: np.ndarray,
    symmetry: Symmetry,
    ignore_colors: list[int],
    background: int | None = None,
) -> tuple[int, int, int]:
    """
    Score how well `grid` satisfies the symmetry.

    Returns:
        (perfect_mapping, off_canvas, bad_mapping)
    """
    n, m = grid.shape
    if background is None:
        occupied = np.argwhere(~np.isin(grid, ignore_colors))
    else:
        occupied = np.argwhere((~np.isin(grid, ignore_colors)) & (grid != int(background)))

    if occupied.size == 0:
        return 0, 0, 0

    tr, tc = symmetry.apply(occupied[:, 0], occupied[:, 1])
    in_canvas = (tr >= 0) & (tr < n) & (tc >= 0) & (tc < m)
    off_canvas = int(np.sum(~in_canvas))

    tr = tr[in_canvas]
    tc = tc[in_canvas]
    occupied = occupied[in_canvas]

    original = grid[occupied[:, 0], occupied[:, 1]]
    transformed = grid[tr, tc]

    bad = int(np.sum((original != transformed) & (~np.isin(transformed, ignore_colors))))
    perfect = int(np.sum(original == transformed))
    return perfect, off_canvas, bad


def apply_symmetry(sprite: np.ndarray, symmetry_type: str, background: int = Color.BLACK) -> np.ndarray:
    """Apply horizontal/vertical mirror symmetry within sprite bounds."""
    bg = int(background)
    n, m = sprite.shape
    if symmetry_type == "horizontal":
        for c in range(m):
            for r in range(n // 2):
                sprite[r, c] = sprite[n - 1 - r, c] = sprite[r, c] if sprite[r, c] != bg else sprite[n - 1 - r, c]
    elif symmetry_type == "vertical":
        for r in range(n):
            for c in range(m // 2):
                sprite[r, c] = sprite[r, m - 1 - c] = sprite[r, c] if sprite[r, c] != bg else sprite[r, m - 1 - c]
    else:
        raise ValueError(f"Invalid symmetry type {symmetry_type}.")
    return sprite


def apply_diagonal_symmetry(sprite: np.ndarray, background: int = Color.BLACK) -> np.ndarray:
    """Apply diagonal symmetry within bounds (square sprite only)."""
    bg = int(background)
    n, m = sprite.shape
    if n != m:
        raise ValueError("Diagonal symmetry requires a square sprite.")
    for r in range(n):
        for c in range(r + 1, m):
            col = bg
            if sprite[c, r] != bg:
                col = int(sprite[c, r])
            if sprite[r, c] != bg:
                col = int(sprite[r, c])
            sprite[r, c] = sprite[c, r] = col
    return sprite


def is_contiguous(bitmask: np.ndarray, background: int = Color.BLACK, connectivity: int = 4) -> bool:
    """Return True iff the mask contains exactly one connected component."""
    struct = STRUCT4 if int(connectivity) == 4 else STRUCT8
    _lab, n = label(bitmask != int(background), structure=struct)
    return int(n) == 1


def generate_sprite(
    n: int,
    m: int,
    symmetry_type: str,
    *,
    fill_percentage: float = 0.5,
    max_colors: int = 9,
    color_palate: list[int] | None = None,
    connectivity: int = 4,
    background: int = Color.BLACK,
    no_repeated_colors: bool = False,
) -> np.ndarray:
    """Generate a random contiguous sprite with optional symmetry (for puzzle generation)."""
    bg = int(background)
    n = int(n)
    m = int(m)
    max_colors = int(max_colors)

    if color_palate is None:
        n_colors = 1
        while n_colors < max_colors and random.random() < 0.3:
            n_colors += 1
        color_palate = random.sample([c for c in Color.ALL_COLORS if c != bg], n_colors)
    else:
        n_colors = len(color_palate)

    grid = np.full((n, m), bg)

    if symmetry_type == "not_symmetric":
        r, c = random.randint(0, n - 1), random.randint(0, m - 1)
        diagonal_orientation = True
    elif symmetry_type == "horizontal":
        r, c = random.randint(0, n - 1), m // 2
        diagonal_orientation = True
    elif symmetry_type == "vertical":
        r, c = n // 2, random.randint(0, m - 1)
        diagonal_orientation = True
    elif symmetry_type == "diagonal":
        diagonal_orientation = random.choice([True, False])
        r = random.randint(0, n - 1)
        c = r if diagonal_orientation else n - 1 - r
    elif symmetry_type == "mirror":
        original_n, original_m = n, m
        n, m = int(n / 2 + 0.5), int(m / 2 + 0.5)
        r, c = random.randint(0, n - 1), random.randint(0, m - 1)
        grid = np.full((n, m), bg)
        diagonal_orientation = True
    elif symmetry_type == "radial":
        if n != m:
            raise ValueError("Radial symmetry requires a square grid.")
        original_length = n
        n, m = int(n / 2 + 0.5), int(m / 2 + 0.5)
        r, c = n - 1, m - 1
        diagonal_orientation = True
    else:
        raise ValueError(f"Invalid symmetry type {symmetry_type}.")

    if int(connectivity) == 4:
        moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    elif int(connectivity) == 8:
        moves = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    else:
        raise ValueError("Connectivity must be 4 or 8.")

    color_index = 0
    while int(np.sum(grid != bg)) < float(fill_percentage) * n * m:
        if no_repeated_colors:
            if (grid[r, c] == bg):
                grid[r, c] = color_palate[color_index]
                color_index = (color_index + 1) % n_colors
        else:
            grid[r, c] = color_palate[color_index]
            if random.random() < 0.33:
                color_index = random.choice(range(n_colors))
        dr, dc = random.choice(moves)
        r2, c2 = r + dr, c + dc
        if 0 <= r2 < n and 0 <= c2 < m:
            r, c = r2, c2

    if symmetry_type in ("horizontal", "vertical"):
        grid = apply_symmetry(grid, symmetry_type, background=bg)
    elif symmetry_type == "radial":
        output = np.full((original_length, original_length), bg)
        paste(output, grid, 0, 0, transparent=bg)
        for _ in range(3):
            paste(output, np.rot90(output), 0, 0, transparent=bg)
        grid = output
    elif symmetry_type == "mirror":
        output = np.full((original_n, original_m), bg)
        output[:n, :m] = grid
        dx = 0 if original_n % 2 == 0 else -1
        dy = 0 if original_m % 2 == 0 else -1
        output[n + dx :, :m] = np.flipud(grid)
        output[:n, m + dy :] = np.fliplr(grid)
        output[n + dx :, m + dy :] = np.flipud(np.fliplr(grid))
        grid = output
        if not is_contiguous(grid, background=bg, connectivity=connectivity):
            return generate_sprite(
                n=original_n,
                m=original_m,
                symmetry_type=symmetry_type,
                fill_percentage=fill_percentage,
                color_palate=color_palate,
                connectivity=connectivity,
                background=bg,
                no_repeated_colors=no_repeated_colors,
            )
    elif symmetry_type == "diagonal":
        if diagonal_orientation:
            grid = np.flipud(grid)
            grid = apply_diagonal_symmetry(grid, background=bg)
            grid = np.flipud(grid)
        else:
            grid = apply_diagonal_symmetry(grid, background=bg)

    return grid


def detect_objects(
    grid: np.ndarray,
    _=None,
    predicate=None,
    background: int = Color.BLACK,
    monochromatic: bool = False,
    connectivity: int | None = None,
    allowed_dimensions: list[tuple[int, int]] | None = None,
    colors: list[int] | None = None,
    can_overlap: bool = False,
) -> list[np.ndarray]:
    """
    Seeds-compatible object detector returning full-grid objects.

    Prefer `find_components(..., return_format="dict")` for new solver code.
    """
    bg = int(background)
    objects: list[np.ndarray] = []

    if connectivity is not None:
        objects.extend(
            find_components(
                grid,
                bg=bg,
                connectivity=connectivity,
                monochromatic=monochromatic,
                return_format="objects",
            )
        )
        if colors is not None:
            allowed = set(map(int, colors)) | {bg}
            objects = [obj for obj in objects if all(int(v) in allowed for v in obj.flatten())]
        if predicate is not None:
            def _cropped(o: np.ndarray) -> np.ndarray:
                b = bbox(o != bg)
                return crop_bbox(o, b) if b is not None else o[0:0, 0:0]

            objects = [obj for obj in objects if predicate(_cropped(obj))]

    if allowed_dimensions is not None:
        allowed_dimensions = [(int(h), int(w)) for h, w in allowed_dimensions]
        objects = [obj for obj in objects if obj.shape in allowed_dimensions]

        scan_objects: list[np.ndarray] = []
        H, W = grid.shape
        for h, w in allowed_dimensions:
            for r in range(H - h + 1):
                for c in range(W - w + 1):
                    cand = grid[r : r + h, c : c + w]
                    if not np.any(cand != bg):
                        continue
                    if colors is not None:
                        allowed = set(map(int, colors)) | {bg}
                        if not all(int(v) in allowed for v in cand.flatten()):
                            continue
                    if predicate is not None and not predicate(cand):
                        continue
                    obj = np.full_like(grid, bg)
                    obj[r : r + h, c : c + w] = cand
                    if not any(np.array_equal(obj, o) for o in objects):
                        scan_objects.append(obj)
        objects.extend(scan_objects)

    if not can_overlap and objects:
        def size_key(obj: np.ndarray):
            b = bbox(obj != bg)
            if b is None:
                return (0, 0)
            spr = crop_bbox(obj, b)
            return (spr.shape[0] * spr.shape[1], int(np.sum(obj != bg)))

        objects.sort(key=size_key, reverse=True)
        masks = [obj != bg for obj in objects]

        n = len(objects)
        overlap = [[False] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                overlap[i][j] = overlap[j][i] = bool(np.any(masks[i] & masks[j]))

        keep = [i for i in range(n) if not any(overlap[i][j] for j in range(n) if j != i)]
        remaining = [i for i in range(n) if i not in keep]

        if remaining:
            best_possible = np.zeros_like(grid, dtype=bool)
            for i in remaining:
                best_possible |= masks[i]
            best_score = int(np.sum(best_possible))

            def dfs(idx: int, chosen: list[int], cur_mask: np.ndarray):
                if idx == len(remaining):
                    return chosen, int(np.sum(cur_mask))
                i = remaining[idx]
                if any(overlap[i][j] for j in chosen):
                    return dfs(idx + 1, chosen, cur_mask)
                with_chosen, with_score = dfs(idx + 1, chosen + [i], cur_mask | masks[i])
                if with_score == best_score:
                    return with_chosen, with_score
                without_chosen, without_score = dfs(idx + 1, chosen, cur_mask)
                return (with_chosen, with_score) if with_score > without_score else (without_chosen, without_score)

            chosen, _ = dfs(0, keep, np.zeros_like(grid, dtype=bool))
            objects = [objects[i] for i in chosen]

    return objects

