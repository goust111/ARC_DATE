import itertools
import random

import numpy as np


BG = 0
SPRITE_RED = 2
H_MIRROR_GREEN = 3
V_MIRROR_YELLOW = 4
HV_MIRROR_ORANGE = 7

MIN_SIDE = 18
MAX_SIDE = 28
GEN_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    box = _bbox(mask)
    assert box is not None
    r0, c0, r1, c1 = box
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, mask.shape)), mask.astype(np.uint8).tobytes()


def _all_flip_variants(mask: np.ndarray) -> list[np.ndarray]:
    return [
        mask.copy(),
        np.fliplr(mask),
        np.flipud(mask),
        np.flipud(np.fliplr(mask)),
    ]


def _is_valid_sprite_mask(mask: np.ndarray) -> bool:
    if mask.dtype != bool:
        mask = mask.astype(bool)
    if int(mask.sum()) < 4 or int(mask.sum()) > 6:
        return False
    comps = _components4(mask)
    if len(comps) != 1:
        return False
    box = _bbox(mask)
    assert box is not None
    r0, c0, r1, c1 = box
    if r0 != 0 or c0 != 0:
        return False
    if r1 + 1 > 3 or c1 + 1 > 3:
        return False
    variants = _all_flip_variants(mask)
    if len({_mask_signature(v) for v in variants}) != 4:
        return False
    return True


def _build_sprite_library() -> list[np.ndarray]:
    sprites: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}
    cells = [(r, c) for r in range(3) for c in range(3)]
    for n in range(4, 7):
        for picks in itertools.combinations(cells, n):
            mask = np.zeros((3, 3), dtype=bool)
            for r, c in picks:
                mask[r, c] = True
            if len(_components4(mask)) != 1:
                continue
            cropped = _crop_mask(mask)
            if not _is_valid_sprite_mask(cropped):
                continue
            sprites[_mask_signature(cropped)] = cropped
    assert sprites
    return list(sprites.values())


SPRITE_LIBRARY = _build_sprite_library()


def _single_component(grid: np.ndarray, color: int) -> np.ndarray:
    comps = _components4(grid == color)
    assert len(comps) == 1
    return comps[0]


def _is_consecutive(values: np.ndarray) -> bool:
    vals = sorted(int(v) for v in values)
    return vals == list(range(vals[0], vals[-1] + 1))


def _parse_instance(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int], int, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert int(g.min()) >= 0 and int(g.max()) <= 9
    assert set(np.unique(g)).issubset({BG, SPRITE_RED, H_MIRROR_GREEN, V_MIRROR_YELLOW})

    sprite_coords = _single_component(g, SPRITE_RED)
    vbar_coords = _single_component(g, H_MIRROR_GREEN)
    hbar_coords = _single_component(g, V_MIRROR_YELLOW)

    sprite_mask = np.zeros_like(g, dtype=bool)
    sprite_mask[sprite_coords[:, 0], sprite_coords[:, 1]] = True
    sprite_box = _bbox(sprite_mask)
    assert sprite_box is not None
    r0, c0, r1, c1 = sprite_box
    sprite_patch = sprite_mask[r0 : r1 + 1, c0 : c1 + 1].copy()
    sh, sw = map(int, sprite_patch.shape)

    assert 4 <= int(sprite_patch.sum()) <= 6
    assert sh <= 3 and sw <= 3
    assert len({_mask_signature(v) for v in _all_flip_variants(sprite_patch)}) == 4

    v_rows = vbar_coords[:, 0]
    v_cols = vbar_coords[:, 1]
    assert len(set(map(int, v_cols))) == 1
    assert _is_consecutive(v_rows)
    p_v = int(v_rows.max() - v_rows.min() + 1)
    assert 3 <= p_v <= 5

    h_rows = hbar_coords[:, 0]
    h_cols = hbar_coords[:, 1]
    assert len(set(map(int, h_rows))) == 1
    assert _is_consecutive(h_cols)
    p_h = int(h_cols.max() - h_cols.min() + 1)
    assert 3 <= p_h <= 5

    # Fix the control-bar anchoring so the input is uniquely decodable.
    assert int(v_cols[0]) == c1 + 2
    assert int(v_rows.min()) == r0
    assert int(h_rows[0]) == r1 + 2
    assert int(h_cols.min()) == c0
    assert p_v >= sh + 1
    assert p_h >= sw + 1

    return sprite_patch, (r0, c0), p_v, p_h


def _copy_color(i: int, j: int) -> int:
    if i % 2 == 0 and j % 2 == 0:
        return SPRITE_RED
    if i % 2 == 0 and j % 2 == 1:
        return H_MIRROR_GREEN
    if i % 2 == 1 and j % 2 == 0:
        return V_MIRROR_YELLOW
    return HV_MIRROR_ORANGE


def _transform_patch(sprite_patch: np.ndarray, i: int, j: int) -> np.ndarray:
    patch = sprite_patch
    if i % 2 == 1:
        patch = np.flipud(patch)
    if j % 2 == 1:
        patch = np.fliplr(patch)
    return patch


def _tile_output(
    canvas_shape: tuple[int, int],
    sprite_patch: np.ndarray,
    origin: tuple[int, int],
    p_v: int,
    p_h: int,
    *,
    apply_mirror: bool,
    apply_recolor: bool,
) -> np.ndarray:
    out = np.zeros(canvas_shape, dtype=int)
    h, w = canvas_shape
    r0, c0 = origin
    sh, sw = sprite_patch.shape
    max_i = (h - r0 - sh) // p_v
    max_j = (w - c0 - sw) // p_h

    for i in range(max_i + 1):
        for j in range(max_j + 1):
            patch = _transform_patch(sprite_patch, i, j) if apply_mirror else sprite_patch
            color = _copy_color(i, j) if apply_recolor else SPRITE_RED
            rr = r0 + i * p_v
            cc = c0 + j * p_h
            view = out[rr : rr + sh, cc : cc + sw]
            assert not np.any(view[patch])
            view[patch] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Tile one asymmetric sprite by bar-encoded periods, mirror by parity, and recolor by mirror type.

    Input structure:
    - Black is the background.
    - A single small red 4-connected sprite sits near the upper-left; its bbox is at most 3x3 and the sprite is
      asymmetric under the horizontal/vertical flips used by the task.
    - A top-aligned green vertical bar stands one column to the right of the sprite bbox, and a left-aligned yellow
      horizontal bar stands one row below the sprite bbox.

    Rule:
    - Read the yellow bar length as the horizontal period and the green bar length as the vertical period.
    - Use the sprite bbox top-left as the anchor of copy (0,0); place copy (i,j) at anchor + (i*P_v, j*P_h) while
      the full bbox still fits inside the grid.
    - Mirror copies by parity: even/even unchanged, even/odd left-right flipped, odd/even up-down flipped,
      odd/odd flipped both ways; recolor those four cases red, green, yellow, and orange respectively.
    - Remove both control bars from the output.

    Key insight:
    - The bars are length encoders rather than drawable templates: their only job is to define the tiling periods,
      after which row/column parity determines both the mirror operation and the final color.
    """
    g = np.asarray(grid, dtype=int)
    sprite_patch, origin, p_v, p_h = _parse_instance(g)
    return _tile_output(
        g.shape,
        sprite_patch,
        origin,
        p_v,
        p_h,
        apply_mirror=True,
        apply_recolor=True,
    )


def _count_options(origin: int, sprite_size: int, period: int) -> list[tuple[int, int, int]]:
    options = []
    for copies in range(3, 8):
        lo = origin + (copies - 1) * period + sprite_size
        hi = lo + period - 1
        lo = max(lo, MIN_SIDE)
        hi = min(hi, MAX_SIDE)
        if lo <= hi:
            options.append((copies, lo, hi))
    return options


def _make_candidate() -> np.ndarray | None:
    sprite_patch = random.choice(SPRITE_LIBRARY)
    sh, sw = map(int, sprite_patch.shape)
    r0 = random.randint(0, 3)
    c0 = random.randint(0, 3)

    period_pairs = []
    for p_v in range(3, 6):
        if p_v < sh + 1:
            continue
        for p_h in range(3, 6):
            if p_h < sw + 1:
                continue
            if p_v >= sh + 2 and p_h >= sw + 2:
                continue
            period_pairs.append((p_v, p_h))
    if not period_pairs:
        return None
    p_v, p_h = random.choice(period_pairs)

    v_options = _count_options(r0, sh, p_v)
    h_options = _count_options(c0, sw, p_h)
    if not v_options or not h_options:
        return None

    _, h_lo, h_hi = random.choice(v_options)
    _, w_lo, w_hi = random.choice(h_options)
    h = random.randint(h_lo, h_hi)
    w = random.randint(w_lo, w_hi)

    grid = np.zeros((h, w), dtype=int)
    sprite_view = grid[r0 : r0 + sh, c0 : c0 + sw]
    sprite_view[sprite_patch] = SPRITE_RED

    v_col = c0 + sw + 1
    h_row = r0 + sh + 1
    if r0 <= h_row < r0 + p_v and c0 <= v_col < c0 + p_h:
        return None
    grid[r0 : r0 + p_v, v_col] = H_MIRROR_GREEN
    grid[h_row, c0 : c0 + p_h] = V_MIRROR_YELLOW
    return grid


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        sprite_patch, origin, p_v, p_h = _parse_instance(grid)
    except Exception:
        return False

    out = main(grid)
    r0, c0 = origin
    sh, sw = map(int, sprite_patch.shape)
    copies_v = (grid.shape[0] - r0 - sh) // p_v + 1
    copies_h = (grid.shape[1] - c0 - sw) // p_h + 1

    if copies_v < 3 or copies_h < 3:
        return False
    if len({_mask_signature(v) for v in _all_flip_variants(sprite_patch)}) != 4:
        return False
    if not {SPRITE_RED, H_MIRROR_GREEN, V_MIRROR_YELLOW, HV_MIRROR_ORANGE}.issubset(set(map(int, np.unique(out)))):
        return False

    no_mirror = _tile_output(
        grid.shape,
        sprite_patch,
        origin,
        p_v,
        p_h,
        apply_mirror=False,
        apply_recolor=True,
    )
    no_recolor = _tile_output(
        grid.shape,
        sprite_patch,
        origin,
        p_v,
        p_h,
        apply_mirror=True,
        apply_recolor=False,
    )
    if np.array_equal(out, no_mirror):
        return False
    if np.array_equal(out, no_recolor):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _make_candidate()
        if grid is None:
            continue
        if not _semantic_checks(grid):
            continue
        return grid
    raise RuntimeError("failed to generate a valid periodic-mirror tiling input")
