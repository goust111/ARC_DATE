from __future__ import annotations

import math
from typing import Iterable

import numpy as np


BLACK = 0
GREY = 5
FG_PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 600


def _infer_sprite_shape(shape: tuple[int, int]) -> tuple[int, int]:
    rows, cols = map(int, shape)
    h = math.isqrt(rows)
    w = math.isqrt(cols)
    assert h * h == rows and w * w == cols, "input must be h^2 by w^2"
    return h, w


def _iter_block_positions(h: int, w: int) -> Iterable[tuple[int, int]]:
    for r in range(h):
        for c in range(w):
            yield r, c


def _block_view(grid: np.ndarray, br: int, bc: int, h: int, w: int) -> np.ndarray:
    return grid[br * h : (br + 1) * h, bc * w : (bc + 1) * w]


def _mask_to_sprite(mask: np.ndarray, fg: int) -> np.ndarray:
    return np.where(mask, int(fg), BLACK).astype(int)


def _stamp_self(mask: np.ndarray, fg: int) -> np.ndarray:
    h, w = map(int, mask.shape)
    tile = _mask_to_sprite(mask, fg)
    out = np.zeros((h * h, w * w), dtype=int)
    for br, bc in _iter_block_positions(h, w):
        if bool(mask[br, bc]):
            _block_view(out, br, bc, h, w)[:] = tile
    return out


def _flat_index(r: int, c: int, w: int) -> int:
    return int(r) * int(w) + int(c)


def _unflatten(idx: int, w: int) -> tuple[int, int]:
    return int(idx) // int(w), int(idx) % int(w)


def _extract_fg_color(grid: np.ndarray) -> int:
    colors = sorted(set(map(int, np.unique(grid))) - {BLACK, GREY})
    assert len(colors) == 1, "expected exactly one visible non-black non-grey color"
    return int(colors[0])


def _analyze_constraints(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    h, w = _infer_sprite_shape(g.shape)
    fg = _extract_fg_color(g)
    count = h * w
    forced = np.zeros(count, dtype=bool)
    conflict = np.zeros((count, count), dtype=bool)
    color_hits = 0

    for br, bc in _iter_block_positions(h, w):
        p = _flat_index(br, bc, w)
        block = _block_view(g, br, bc, h, w)
        for sr, sc in _iter_block_positions(h, w):
            q = _flat_index(sr, sc, w)
            value = int(block[sr, sc])
            if value == fg:
                forced[p] = True
                forced[q] = True
                color_hits += 1
            elif value == BLACK:
                conflict[p, q] = True
                conflict[q, p] = True
            else:
                assert value == GREY, "unexpected color in occluded stamp grid"

    assert color_hits > 0, "at least one visible foreground cell is required"

    forced_ids = np.flatnonzero(forced)
    if forced_ids.size:
        assert not bool(conflict[np.ix_(forced_ids, forced_ids)].any()), "contradictory visible evidence"

    banned = np.diag(conflict).copy()
    if forced_ids.size:
        banned |= conflict[:, forced].any(axis=1)
    unique = bool(np.all(forced | banned))
    mask = forced.reshape(h, w)

    return {
        "h": h,
        "w": w,
        "fg": fg,
        "forced": forced,
        "conflict": conflict,
        "banned": banned,
        "unique": unique,
        "mask": mask,
    }


def _verify_candidate(grid: np.ndarray, mask: np.ndarray, fg: int) -> None:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, mask.shape)
    sprite = _mask_to_sprite(mask, fg)
    for br, bc in _iter_block_positions(h, w):
        block = _block_view(g, br, bc, h, w)
        visible = block != GREY
        if bool(mask[br, bc]):
            assert np.array_equal(block[visible], sprite[visible]), "active block disagrees with recovered sprite"
        else:
            assert np.all(block[visible] == BLACK), "inactive block cannot show visible foreground"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the unseen monochrome sprite from its self-stamped large grid after grey rectangular occlusion.

    Input structure:
    - The input is an h^2 by w^2 grid partitioned into h by w blocks of size h by w.
    - Before occlusion, each non-black cell of one small monochrome sprite selected one block position,
      and that whole sprite was stamped into the selected block; black sprite cells left their blocks empty.
    - Several axis-aligned grey rectangles then covered parts of the large stamped image, possibly hiding
      only part of a block or an entire block.

    Rule:
    - Infer h and w from the square roots of the input dimensions.
    - Use all non-grey evidence across the stamped blocks to recover which sprite cells are foreground
      and which are black.
    - Enforce the self-reference that the macro block layout and the micro stamped content are the same sprite.
    - Output the recovered h by w sprite.

    Key insight:
    - Every visible foreground cell simultaneously identifies one active block and one foreground sprite cell,
      while every visible black cell forbids that block-position/sprite-cell pair from both being foreground.
      Because the block layout equals the sprite itself, information propagates between macro positions and
      micro pixels until the sprite is uniquely determined.
    """
    analysis = _analyze_constraints(np.asarray(grid, dtype=int))
    assert analysis["unique"], "visible evidence does not uniquely determine the sprite"
    mask = analysis["mask"]
    fg = int(analysis["fg"])
    _verify_candidate(np.asarray(grid, dtype=int), mask, fg)
    return _mask_to_sprite(mask, fg)


def _random_sprite_mask(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    count = h * w
    for _ in range(200):
        density = float(rng.uniform(0.35, 0.65))
        mask = rng.random((h, w)) < density
        ones = int(mask.sum())
        if not (3 <= ones <= count - 2):
            continue
        if np.count_nonzero(mask.any(axis=1)) < 2 or np.count_nonzero(mask.any(axis=0)) < 2:
            continue
        return mask
    raise RuntimeError("failed to sample a non-degenerate sprite mask")


def _paint_rect(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    mask[r0:r1, c0:c1] = True


def _sample_disjoint_strips(rng: np.random.Generator, h: int, w: int) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    if rng.random() < 0.5:
        top_h = int(rng.integers(1, h))
        bottom_h = int(rng.integers(1, h - top_h + 1))
        while top_h + bottom_h > h:
            top_h = int(rng.integers(1, h))
            bottom_h = int(rng.integers(1, h - top_h + 1))
        return (0, 0, top_h, w), (h - bottom_h, 0, h, w)
    left_w = int(rng.integers(1, w))
    right_w = int(rng.integers(1, w - left_w + 1))
    while left_w + right_w > w:
        left_w = int(rng.integers(1, w))
        right_w = int(rng.integers(1, w - left_w + 1))
    return (0, 0, h, left_w), (0, w - right_w, h, w)


def _build_occlusion(
    rng: np.random.Generator, mask: np.ndarray
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int], tuple[int, int]]:
    h, w = map(int, mask.shape)
    active = [tuple(map(int, pos)) for pos in np.argwhere(mask)]
    hidden = active[int(rng.integers(len(active)))]
    others = [pos for pos in active if pos != hidden]
    donor1 = others[int(rng.integers(len(others)))]
    donor2_choices = [pos for pos in others if pos != donor1]
    donor2 = donor2_choices[int(rng.integers(len(donor2_choices)))]

    grey = np.zeros((h * h, w * w), dtype=bool)

    _paint_rect(grey, hidden[0] * h, hidden[1] * w, (hidden[0] + 1) * h, (hidden[1] + 1) * w)

    strip1, strip2 = _sample_disjoint_strips(rng, h, w)
    for (br, bc), rect in ((donor1, strip1), (donor2, strip2)):
        r0, c0, r1, c1 = rect
        _paint_rect(grey, br * h + r0, bc * w + c0, br * h + r1, bc * w + c1)

    for br, bc in active:
        block = _block_view(grey, br, bc, h, w)
        if block.any():
            continue
        rr = int(rng.integers(0, h))
        cc = int(rng.integers(0, w))
        block[rr, cc] = True

    if rng.random() < 0.55:
        extra_h = int(rng.integers(1, h + 1))
        extra_w = int(rng.integers(1, w + 1))
        top = int(rng.integers(0, h * h - extra_h + 1))
        left = int(rng.integers(0, w * w - extra_w + 1))
        _paint_rect(grey, top, left, top + extra_h, left + extra_w)

    return grey, hidden, donor1, donor2


def _block_statuses(mask: np.ndarray, grey: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    partial_active: list[tuple[int, int]] = []
    full_active: list[tuple[int, int]] = []
    untouched_active: list[tuple[int, int]] = []
    for br, bc in _iter_block_positions(h, w):
        if not bool(mask[br, bc]):
            continue
        block = _block_view(grey, br, bc, h, w)
        if block.all():
            full_active.append((br, bc))
        elif block.any():
            partial_active.append((br, bc))
        else:
            untouched_active.append((br, bc))
    return partial_active, full_active, untouched_active


def _semantic_checks(
    inp: np.ndarray, mask: np.ndarray, fg: int, hidden: tuple[int, int], donor1: tuple[int, int], donor2: tuple[int, int]
) -> bool:
    h, w = map(int, mask.shape)
    analysis = _analyze_constraints(inp)
    if not analysis["unique"]:
        return False
    if int(analysis["fg"]) != int(fg):
        return False
    if not np.array_equal(analysis["mask"], mask):
        return False
    try:
        _verify_candidate(inp, mask, fg)
    except AssertionError:
        return False

    grey = inp == GREY
    partial_active, full_active, untouched_active = _block_statuses(mask, grey)
    if hidden not in full_active:
        return False
    if untouched_active:
        return False
    if donor1 not in partial_active or donor2 not in partial_active:
        return False

    donor_visible_union = (~_block_view(grey, donor1[0], donor1[1], h, w)) | (~_block_view(grey, donor2[0], donor2[1], h, w))
    if not bool(np.all(donor_visible_union)):
        return False

    hidden_idx = _flat_index(hidden[0], hidden[1], w)
    if not bool(analysis["forced"][hidden_idx]):
        return False
    if np.any(_block_view(inp, hidden[0], hidden[1], h, w) != GREY):
        return False

    zero_cells = np.argwhere(~mask)
    if zero_cells.size == 0:
        return False
    zero_block_evidence = False
    for zr, zc in zero_cells:
        if int(_block_view(inp, donor1[0], donor1[1], h, w)[int(zr), int(zc)]) == BLACK:
            zero_block_evidence = True
            break
        if int(_block_view(inp, donor2[0], donor2[1], h, w)[int(zr), int(zc)]) == BLACK:
            zero_block_evidence = True
            break
    if not zero_block_evidence:
        return False

    complement_needed = False
    for br, bc in (donor1, donor2):
        block = _block_view(inp, br, bc, h, w)
        hidden_positions = np.argwhere(block == GREY)
        for sr, sc in hidden_positions:
            other = donor2 if (br, bc) == donor1 else donor1
            other_block = _block_view(inp, other[0], other[1], h, w)
            if int(other_block[int(sr), int(sc)]) != GREY:
                complement_needed = True
                break
        if complement_needed:
            break
    return complement_needed


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(3, 6))
    w = int(rng.integers(3, 6))
    fg = int(rng.choice(FG_PALETTE))
    mask = _random_sprite_mask(rng, h, w)
    base = _stamp_self(mask, fg)
    grey, hidden, donor1, donor2 = _build_occlusion(rng, mask)
    if np.all(grey):
        return None
    inp = base.copy()
    inp[grey] = GREY
    if not np.any(inp == fg):
        return None
    try:
        out = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(out, _mask_to_sprite(mask, fg)):
        return None
    if not _semantic_checks(inp, mask, fg, hidden, donor1, donor2):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid self-stamp inversion input")
