from __future__ import annotations

import numpy as np


TEAL = 8
PALETTE = tuple(range(10))
RNG = np.random.default_rng()
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(counts.argmax())


def _layout_candidates(shape: tuple[int, int]) -> list[tuple[int, int]]:
    height, width = map(int, shape)
    if height != width:
        return []

    candidates: list[tuple[int, int]] = []
    for k in range(2, height + 1):
        rem = height - k * k
        if rem < k - 1 or rem % (k - 1) != 0:
            continue
        gap = rem // (k - 1)
        if gap >= 1 and width == k * k + (k - 1) * gap:
            candidates.append((k, gap))
    return candidates


def _starts(k: int, gap: int) -> list[int]:
    return [i * (k + gap) for i in range(k)]


def _is_uniform(block: np.ndarray) -> bool:
    return bool(np.all(block == int(block[0, 0])))


def _is_solid_rectangle(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    view = mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    return bool(np.all(view) and int(view.size) == int(mask.sum()))


def _decode_with_layout(grid: np.ndarray, bg: int, k: int, gap: int) -> np.ndarray | None:
    height, width = map(int, grid.shape)
    starts = _starts(k, gap)
    if starts[-1] + k != height or width != height:
        return None

    for i in range(k - 1):
        row_gap = grid[starts[i] + k : starts[i + 1], :]
        col_gap = grid[:, starts[i] + k : starts[i + 1]]
        if row_gap.size == 0 or col_gap.size == 0:
            return None
        if not np.all(row_gap == bg) or not np.all(col_gap == bg):
            return None

    coarse = np.full((k, k), bg, dtype=int)
    detail_pos: tuple[int, int] | None = None
    detail_block: np.ndarray | None = None

    for br, rs in enumerate(starts):
        for bc, cs in enumerate(starts):
            block = grid[rs : rs + k, cs : cs + k]
            if _is_uniform(block):
                coarse[br, bc] = int(block[0, 0])
            else:
                if detail_pos is not None:
                    return None
                detail_pos = (br, bc)
                detail_block = block.copy()

    if detail_pos is None or detail_block is None:
        return None

    dr, dc = detail_pos
    expected = coarse.copy()
    expected[dr, dc] = TEAL

    mismatch = detail_block != expected
    if not _is_solid_rectangle(mismatch):
        return None

    occluder_colors = np.unique(detail_block[mismatch])
    if occluder_colors.size != 1:
        return None
    if int(occluder_colors[0]) in set(map(int, np.unique(expected))):
        return None

    visible_teal = np.argwhere(detail_block == TEAL)
    if visible_teal.size:
        if visible_teal.shape[0] != 1:
            return None
        if tuple(map(int, visible_teal[0])) != (dr, dc):
            return None

    return expected


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the occluded fine sprite by reading the surrounding slot array as its enlarged color map.

    Input structure:
    - The grid is a square k-by-k array of k-by-k slots separated by background divider bands.
    - Every slot except one is either completely empty or filled solidly with a single color.
    - The remaining slot contains a k-by-k detailed pattern whose visible cells match the coarse slot colors, except that one rectangular region is covered by a distinct occluder color and the cell at the slot's own coarse coordinate is teal.

    Rule:
    - Infer k and the divider width from the overall square layout.
    - Collapse each solid or empty slot to one coarse cell, and locate the unique non-uniform slot as the damaged detailed pattern.
    - Reconstruct the full k-by-k detail from the coarse slot map, placing teal at the cell whose local coordinate matches the detailed slot's slot position, and output only that repaired k-by-k pattern.

    Key insight:
    - The whole coarse array is a blown-up copy of the missing fine picture, so the repair direction is reversed: the large-scale slot arrangement determines the hidden cells inside the small damaged slot.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    outputs = []
    for k, gap in _layout_candidates(g.shape):
        out = _decode_with_layout(g, bg, k, gap)
        if out is not None:
            outputs.append(out)

    assert outputs, "failed to decode any valid slot layout"
    unique = {(tuple(out.shape), out.tobytes()): out for out in outputs}
    assert len(unique) == 1, "expected a unique repaired detail"
    return next(iter(unique.values()))


def _sample_slot_mask(k: int, detail_pos: tuple[int, int]) -> np.ndarray | None:
    for _ in range(GEN_ATTEMPTS):
        fill_prob = float(RNG.uniform(0.35, 0.58))
        mask = RNG.random((k, k)) < fill_prob
        mask[detail_pos] = False
        if int(mask.sum()) < k + 1:
            continue
        if int((~mask).sum()) < 2:
            continue
        if np.any(mask.sum(axis=0) == 0) or np.any(mask.sum(axis=1) == 0):
            continue
        return mask
    return None


def _assign_colors(mask: np.ndarray, bg: int, occluder: int) -> np.ndarray:
    k = int(mask.shape[0])
    coords = np.argwhere(mask)
    available = [c for c in PALETTE if c not in {bg, TEAL, occluder}]
    n_fill = len(coords)
    n_colors = int(RNG.integers(2, min(4, n_fill) + 1))
    chosen = [int(c) for c in RNG.choice(np.array(available, dtype=int), size=n_colors, replace=False)]

    coarse = np.full((k, k), bg, dtype=int)
    perm = RNG.permutation(n_fill)
    for idx, coord_idx in enumerate(perm):
        r, c = map(int, coords[coord_idx])
        if idx < n_colors:
            coarse[r, c] = chosen[idx]
        else:
            coarse[r, c] = int(chosen[int(RNG.integers(0, n_colors))])
    return coarse


def _sample_occlusion(
    target: np.ndarray,
    bg: int,
    occluder: int,
    detail_pos: tuple[int, int],
    cover_self: bool,
) -> np.ndarray | None:
    k = int(target.shape[0])
    sr, sc = detail_pos
    for _ in range(GEN_ATTEMPTS):
        h = int(RNG.integers(1, k + 1))
        w = int(RNG.integers(1, k + 1))
        area = h * w
        if area > k * k - 2:
            continue
        r0 = int(RNG.integers(0, k - h + 1))
        c0 = int(RNG.integers(0, k - w + 1))
        hits_self = r0 <= sr < r0 + h and c0 <= sc < c0 + w
        if hits_self != cover_self:
            continue

        block = target.copy()
        block[r0 : r0 + h, c0 : c0 + w] = occluder

        visible = target.copy()
        visible[r0 : r0 + h, c0 : c0 + w] = -1
        if not np.any(visible == TEAL) and not cover_self:
            continue
        if not np.any((visible >= 0) & (visible != bg) & (visible != TEAL)):
            continue
        if np.unique(block).size < 3:
            continue
        return block
    return None


def _render_case(
    k: int,
    gap: int,
    bg: int,
    occluder: int,
    detail_pos: tuple[int, int],
    coarse: np.ndarray,
    detail_block: np.ndarray,
) -> np.ndarray:
    size = k * k + (k - 1) * gap
    grid = np.full((size, size), bg, dtype=int)
    starts = _starts(k, gap)
    for br, rs in enumerate(starts):
        for bc, cs in enumerate(starts):
            slot = grid[rs : rs + k, cs : cs + k]
            if (br, bc) == detail_pos:
                slot[:] = detail_block
            elif int(coarse[br, bc]) != bg:
                slot[:] = int(coarse[br, bc])
    return grid


def _gen_attempt() -> np.ndarray | None:
    k = int(RNG.choice(np.array([3, 4, 5], dtype=int)))
    gap_options = [gap for gap in (1, 2, 3) if k * k + (k - 1) * gap <= 30]
    gap = int(RNG.choice(np.array(gap_options, dtype=int)))

    bg_choices = [c for c in PALETTE if c != TEAL]
    bg = int(RNG.choice(np.array(bg_choices, dtype=int)))
    occluder_choices = [c for c in PALETTE if c not in {bg, TEAL}]
    occluder = int(RNG.choice(np.array(occluder_choices, dtype=int)))
    detail_pos = (int(RNG.integers(0, k)), int(RNG.integers(0, k)))

    mask = _sample_slot_mask(k, detail_pos)
    if mask is None:
        return None
    coarse = _assign_colors(mask, bg, occluder)

    target = coarse.copy()
    target[detail_pos] = TEAL

    cover_self = bool(RNG.integers(0, 2))
    detail_block = _sample_occlusion(target, bg, occluder, detail_pos, cover_self)
    if detail_block is None:
        return None

    grid = _render_case(k, gap, bg, occluder, detail_pos, coarse, detail_block)
    counts = np.bincount(grid.ravel(), minlength=10)
    if int(counts[bg]) <= int(np.max(np.delete(counts, bg))):
        return None
    if np.array_equal(main(grid), target):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
