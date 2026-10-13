from __future__ import annotations

import numpy as np

from arc_utils import find_components, get_bg, paste
import itertools
import random


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(int(a0), int(b0)) <= min(int(a1), int(b1))


def _is_solid_2x2(comp: dict, g: np.ndarray) -> bool:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    return (
        int(comp["area"]) == 4
        and int(r1 - r0 + 1) == 2
        and int(c1 - c0 + 1) == 2
        and bool((g[r0 : r1 + 1, c0 : c1 + 1] == int(comp["color"])).all())
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide the non-square sprite to the square anchor and replace the anchor with the sprite's mirror image.

    Input structure:
    - The background is the most common color.
    - There is one 8-connected non-background sprite that is not a solid 2x2 square.
    - There is one solid 2x2 square anchor in another non-background color.

    Rule:
    - Move the sprite straight toward the anchor along their unique cardinal separation until it touches the anchor's near edge.
    - Reflect the moved sprite across that near edge and draw the reflected copy in the anchor color.
    - Clear the original locations; the output keeps only the moved sprite and reflected anchor-color copy.

    Key insight:
    - The 2x2 square marks the mirror line: its side facing the sprite becomes the contact edge for a two-color reflected pair.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = map(int, g.shape)

    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True)
    assert len(comps) == 2

    anchors = [comp for comp in comps if _is_solid_2x2(comp, g)]
    assert len(anchors) == 1
    anchor = anchors[0]

    sprites = [comp for comp in comps if comp is not anchor]
    assert len(sprites) == 1
    sprite = sprites[0]
    assert int(sprite["color"]) != int(anchor["color"])

    sr0, sc0, sr1, sc1 = map(int, sprite["bbox"])
    ar0, ac0, ar1, ac1 = map(int, anchor["bbox"])

    candidates: list[tuple[str, int, int]] = []
    if sc1 < ac0 and _intervals_overlap(sr0, sr1, ar0, ar1):
        candidates.append(("left", 0, ac0 - sc1 - 1))
    if sc0 > ac1 and _intervals_overlap(sr0, sr1, ar0, ar1):
        candidates.append(("right", 0, ac1 + 1 - sc0))
    if sr1 < ar0 and _intervals_overlap(sc0, sc1, ac0, ac1):
        candidates.append(("above", ar0 - sr1 - 1, 0))
    if sr0 > ar1 and _intervals_overlap(sc0, sc1, ac0, ac1):
        candidates.append(("below", ar1 + 1 - sr0, 0))
    assert len(candidates) == 1

    side, dr, dc = candidates[0]
    moved: list[tuple[int, int]] = []
    mirrored: list[tuple[int, int]] = []
    for r, c in sprite["coords"]:
        rr = int(r) + int(dr)
        cc = int(c) + int(dc)
        moved.append((rr, cc))
        if side == "left":
            mirrored.append((rr, 2 * ac0 - 1 - cc))
        elif side == "right":
            mirrored.append((rr, 2 * ac1 + 1 - cc))
        elif side == "above":
            mirrored.append((2 * ar0 - 1 - rr, cc))
        else:
            mirrored.append((2 * ar1 + 1 - rr, cc))

    assert all(0 <= r < H and 0 <= c < W for r, c in moved)
    assert all(0 <= r < H and 0 <= c < W for r, c in mirrored)
    assert not (set(moved) & set(mirrored))

    out = np.full_like(g, bg)
    for r, c in moved:
        out[r, c] = int(sprite["color"])
    for r, c in mirrored:
        out[r, c] = int(anchor["color"])
    return out

# --------------- generator (o5x) ---------------

from arc_utils import is_contiguous

# ── Config ──────────────────────────────────────────────────────────────
GRID_MIN, GRID_MAX = 10, 20        # canvas size range (rows and cols)
SPRITE_AREA = 4                     # number of filled cells per sprite
SPRITE_BBOX_SHAPES = [(2, 3), (3, 2), (3, 3)]  # allowed sprite bbox dims
GAP_MIN, GAP_MAX = 2, 4            # empty-cell gap between sprite and anchor
ANCHOR_SIZE = 2                     # anchor is always a 2x2 solid square
NUM_COLORS = 10                     # ARC color palette size
SIDES = ["left", "right", "above", "below"]
RETRY_LIMIT = 300                   # max attempts before raising


def _has_full_bbox(mask: np.ndarray) -> bool:
    """True if mask has at least one filled cell on each edge of its bounding box."""
    return (mask[0].any() and mask[-1].any() and
            mask[:, 0].any() and mask[:, -1].any())


def _sprite_masks(axis: str) -> list[np.ndarray]:
    """Enumerate all valid 4-cell sprite masks for a given mirror axis.

    A valid mask is 8-connected, spans its full bbox, and is NOT symmetric
    along the given axis (lr or ud).
    """
    flip = np.fliplr if axis == "lr" else np.flipud
    out = []
    for h, w in SPRITE_BBOX_SHAPES:
        for cells in itertools.combinations(range(h * w), SPRITE_AREA):
            mask = np.zeros((h, w), dtype=bool)
            for cell in cells:
                mask[cell // w, cell % w] = True
            if not _has_full_bbox(mask):
                continue
            if not is_contiguous(mask, connectivity=8):
                continue
            if np.array_equal(mask, flip(mask)):
                continue
            out.append(mask)
    return out


_MASKS = {"lr": _sprite_masks("lr"), "ud": _sprite_masks("ud")}


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int,
                color: int, bg: int) -> None:
    """Stamp a boolean mask onto grid at (r0, c0) using the given color."""
    patch = np.full(mask.shape, bg, dtype=int)
    patch[mask] = color
    paste(grid, patch, r0, c0, transparent=bg)


def _rand_pair_origin(H: int, W: int, h: int, w: int, side: str,
                      gap: int) -> tuple[int, int] | None:
    """Pick a random origin for the touching sprite+mirror pair.

    Returns (row, col) of the output pair's top-left corner, or None if the
    pair cannot fit on the grid with enough room for the input-side gap.
    """
    if side in {"left", "right"}:
        r_lo, r_hi = 0, H - h
        total_w = 2 * w + (gap if side == "right" else 0)
        c_lo = gap if side == "left" else 0
        c_hi = W - total_w if side == "right" else W - 2 * w
    else:
        c_lo, c_hi = 0, W - w
        total_h = 2 * h + (gap if side == "below" else 0)
        r_lo = gap if side == "above" else 0
        r_hi = H - total_h if side == "below" else H - 2 * h
    if r_lo > r_hi or c_lo > c_hi:
        return None
    return random.randint(r_lo, r_hi), random.randint(c_lo, c_hi)


def _layout_for_side(side: str, pr: int, pc: int, h: int, w: int,
                     gap: int, sprite: np.ndarray):
    """Compute output and input placement params for a given side.

    Returns (sprite_r, sprite_c, mirror_r, mirror_c, flipped_sprite,
             anchor_r, anchor_c, input_r, input_c).
    """
    flip = np.fliplr if side in {"left", "right"} else np.flipud
    flipped = flip(sprite)

    if side == "left":
        sr, sc = pr, pc                     # sprite at pair origin
        mr, mc = sr, sc + w                  # mirror right of sprite
        anchor_r = random.randint(sr, sr + h - ANCHOR_SIZE)
        anchor_c = sc + w
        input_r, input_c = sr, sc - gap
    elif side == "right":
        sr, sc = pr, pc + w                  # sprite right half
        mr, mc = pr, pc                      # mirror left half
        anchor_r = random.randint(sr, sr + h - ANCHOR_SIZE)
        anchor_c = sc - ANCHOR_SIZE
        input_r, input_c = sr, sc + gap
    elif side == "above":
        sr, sc = pr, pc                      # sprite at top
        mr, mc = sr + h, sc                  # mirror below
        anchor_r = sr + h
        anchor_c = random.randint(sc, sc + w - ANCHOR_SIZE)
        input_r, input_c = sr - gap, sc
    else:  # below
        sr, sc = pr + h, pc                  # sprite at bottom
        mr, mc = pr, pc                      # mirror above
        anchor_r = sr - ANCHOR_SIZE
        anchor_c = random.randint(sc, sc + w - ANCHOR_SIZE)
        input_r, input_c = sr + gap, sc

    return sr, sc, mr, mc, flipped, anchor_r, anchor_c, input_r, input_c


def _gen_attempt() -> np.ndarray | None:
    """Try to build one valid (input, output) pair; return input or None."""
    H = random.randint(GRID_MIN, GRID_MAX)
    W = random.randint(GRID_MIN, GRID_MAX)
    bg = random.randrange(NUM_COLORS)
    sprite_color, anchor_color = random.sample(
        [c for c in range(NUM_COLORS) if c != bg], 2)

    side = random.choice(SIDES)
    axis = "lr" if side in {"left", "right"} else "ud"
    sprite = random.choice(_MASKS[axis]).copy()
    h, w = map(int, sprite.shape)
    gap = random.randint(GAP_MIN, GAP_MAX)

    origin = _rand_pair_origin(H, W, h, w, side, gap)
    if origin is None:
        return None
    pr, pc = origin

    (sr, sc, mr, mc, flipped,
     anchor_r, anchor_c, input_r, input_c) = _layout_for_side(
        side, pr, pc, h, w, gap, sprite)

    # Build expected output
    out = np.full((H, W), bg, dtype=int)
    _paint_mask(out, sprite, sr, sc, sprite_color, bg)
    _paint_mask(out, flipped, mr, mc, anchor_color, bg)

    # Build input
    inp = np.full((H, W), bg, dtype=int)
    _paint_mask(inp, sprite, input_r, input_c, sprite_color, bg)
    paste(inp, np.full((ANCHOR_SIZE, ANCHOR_SIZE), anchor_color, dtype=int),
          anchor_r, anchor_c)

    # Oracle validation
    comps = find_components(inp, bg=bg, connectivity=8, monochromatic=True)
    if len(comps) != 2:
        return None
    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 11dc524f input")
