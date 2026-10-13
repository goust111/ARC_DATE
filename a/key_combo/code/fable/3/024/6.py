from __future__ import annotations

import random

import numpy as np


def _colors_present(g: np.ndarray) -> list[int]:
    return [int(v) for v in np.unique(g)]


def _relation_options(n: int) -> list[tuple[int, int, list[int]]]:
    options = []
    for k in range(2, int(np.sqrt(n)) + 1):
        spare = n - k * k
        if spare < 0 or spare % (k - 1):
            continue
        gap = spare // (k - 1)
        starts = [i * (k + gap) for i in range(k)]
        if starts[-1] + k == n:
            options.append((k, gap, starts))
    return options


def _strip_mask(n: int, k: int, starts: list[int], axis: int, hidden: int) -> np.ndarray:
    mask = np.zeros((n, n), dtype=bool)
    s = starts[hidden]
    if axis == 0:
        mask[s : s + k, :] = True
    else:
        mask[:, s : s + k] = True
    return mask


def _slot_area_mask(n: int, k: int, starts: list[int]) -> np.ndarray:
    mask = np.zeros((n, n), dtype=bool)
    for rs in starts:
        for cs in starts:
            mask[rs : rs + k, cs : cs + k] = True
    return mask


def _is_hidden(axis: int, hidden: int, br: int, bc: int) -> bool:
    return (axis == 0 and br == hidden) or (axis == 1 and bc == hidden)


def _try_solve_candidate(
    g: np.ndarray,
    *,
    k: int,
    starts: list[int],
    bg: int,
    sprite_color: int,
    occluder_color: int,
    axis: int,
    hidden: int,
) -> np.ndarray | None:
    n = g.shape[0]
    occ_mask = _strip_mask(n, k, starts, axis, hidden)
    if not np.all(g[occ_mask] == occluder_color):
        return None
    if np.any(g[~occ_mask] == occluder_color):
        return None

    separator = ~_slot_area_mask(n, k, starts)
    if np.any((separator & ~occ_mask) & (g != bg)):
        return None

    block_masks: dict[tuple[int, int], np.ndarray] = {}
    visible_positions: list[tuple[int, int]] = []
    union = np.zeros((k, k), dtype=bool)

    for br, rs in enumerate(starts):
        for bc, cs in enumerate(starts):
            if _is_hidden(axis, hidden, br, bc):
                continue
            block = g[rs : rs + k, cs : cs + k]
            if np.any((block != bg) & (block != sprite_color)):
                return None
            mask = block == sprite_color
            block_masks[(br, bc)] = mask
            if np.any(mask):
                visible_positions.append((br, bc))
                union |= mask

    if len(visible_positions) < 2:
        return None
    if int(union.sum()) < 3:
        return None

    signatures: set[bytes] = set()
    for (br, bc), mask in block_masks.items():
        if union[br, bc]:
            expected = union.copy()
            expected[br, bc] = False
            if not np.array_equal(mask, expected):
                return None
            missing = union & ~mask
            if int(missing.sum()) != 1 or not bool(missing[br, bc]):
                return None
            if np.array_equal(mask, union):
                return None
            signatures.add(mask.tobytes())
        elif np.any(mask):
            return None

    if len(signatures) != len(visible_positions):
        return None

    hidden_required = [
        (br, bc)
        for br in range(k)
        for bc in range(k)
        if _is_hidden(axis, hidden, br, bc) and union[br, bc]
    ]
    if not hidden_required:
        return None

    out = g.copy()
    out[occ_mask] = bg
    for br, bc in hidden_required:
        rs = starts[br]
        cs = starts[bc]
        block = out[rs : rs + k, cs : cs + k]
        block[:, :] = bg
        block[union] = sprite_color
        block[br, bc] = bg

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the occluded row or column of coordinate-holed copies in a self-similar sprite lattice.

    Input structure:
    - A square canvas is partitioned into a k by k lattice of k by k slots, with
      optional uniform background gaps between slots.
    - One foreground color forms a solid bar covering exactly one whole slot row
      or one whole slot column.
    - A second foreground color appears as sprite copies in the other slots; the
      occupied slot coordinates are the same mask as the sprite itself.
    - Every visible copy is missing exactly the local cell equal to that copy's
      coarse lattice coordinate, so no complete sprite is shown.

    Rule:
    - Align all visible copies by slot coordinates and take their union to recover
      the complete sprite mask.
    - Read that complete mask as the coarse lattice map to decide which covered
      slots should contain copies.
    - Erase the occluding bar, then draw the missing copies only in those covered
      slots, again leaving out each copy's own coordinate cell.

    Key insight:
    - The small holes are coordinates, not noise: the copies jointly reconstruct a
      sprite that is also the map of where all copies belong.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1], "expected a square 2D grid"
    n = int(g.shape[0])

    colors = _colors_present(g)
    assert len(colors) == 3, "expected background, sprite color, and occluder color"

    outputs: list[np.ndarray] = []
    for k, _gap, starts in _relation_options(n):
        for bg in colors:
            others = [c for c in colors if c != bg]
            for occluder_color in others:
                sprite_color = next(c for c in others if c != occluder_color)
                for axis in (0, 1):
                    for hidden in range(k):
                        solved = _try_solve_candidate(
                            g,
                            k=k,
                            starts=starts,
                            bg=bg,
                            sprite_color=sprite_color,
                            occluder_color=occluder_color,
                            axis=axis,
                            hidden=hidden,
                        )
                        if solved is not None:
                            outputs.append(solved)

    unique: list[np.ndarray] = []
    for out in outputs:
        if not any(np.array_equal(out, old) for old in unique):
            unique.append(out)

    assert len(unique) == 1, f"expected a unique interpretation, got {len(unique)}"
    return unique[0]


# --------------- generator ---------------

_K_CHOICES = (3, 4, 5)
_MAX_ATTEMPTS = 600
_COLOR_POOL = tuple(range(10))
_GEN_CALL_INDEX = 0


def _neighbors4_in(k: int, r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < k and 0 <= cc < k:
            out.append((rr, cc))
    return out


def _random_connected_mask(k: int) -> np.ndarray:
    total = k * k
    min_cells = max(4, k + 1)
    max_cells = min(total - 2, max(min_cells, int(total * 0.68)))
    target = random.randint(min_cells, max_cells)

    start = (random.randrange(k), random.randrange(k))
    cells = {start}
    frontier = set(_neighbors4_in(k, *start))
    while len(cells) < target:
        if not frontier:
            return _random_connected_mask(k)
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        for nb in _neighbors4_in(k, *cell):
            if nb not in cells:
                frontier.add(nb)

    mask = np.zeros((k, k), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _valid_hidden_indices(sprite_mask: np.ndarray, axis: int) -> list[int]:
    k = sprite_mask.shape[0]
    total = int(sprite_mask.sum())
    valid = []
    for i in range(k):
        line = sprite_mask[i, :] if axis == 0 else sprite_mask[:, i]
        hidden_count = int(line.sum())
        visible_count = total - hidden_count
        if 1 <= hidden_count < k and visible_count >= 2:
            valid.append(i)
    return valid


def _starts(k: int, gap: int) -> list[int]:
    return [i * (k + gap) for i in range(k)]


def _render_answer(
    sprite_mask: np.ndarray,
    starts: list[int],
    bg: int,
    sprite_color: int,
) -> np.ndarray:
    k = sprite_mask.shape[0]
    n = starts[-1] + k
    out = np.full((n, n), bg, dtype=int)
    for br, rs in enumerate(starts):
        for bc, cs in enumerate(starts):
            if not sprite_mask[br, bc]:
                continue
            block = out[rs : rs + k, cs : cs + k]
            block[sprite_mask] = sprite_color
            block[br, bc] = bg
    return out


def _apply_occluder(
    solved: np.ndarray,
    k: int,
    starts: list[int],
    axis: int,
    hidden: int,
    occluder_color: int,
) -> np.ndarray:
    inp = solved.copy()
    s = starts[hidden]
    if axis == 0:
        inp[s : s + k, :] = occluder_color
    else:
        inp[:, s : s + k] = occluder_color
    return inp


def _semantic_checks(
    inp: np.ndarray,
    expected: np.ndarray,
    sprite_mask: np.ndarray,
    starts: list[int],
    axis: int,
    hidden: int,
    bg: int,
    sprite_color: int,
    occluder_color: int,
) -> bool:
    k = sprite_mask.shape[0]
    n = inp.shape[0]
    occ_mask = _strip_mask(n, k, starts, axis, hidden)
    if not np.all(inp[occ_mask] == occluder_color):
        return False
    if np.any(inp[~occ_mask] == occluder_color):
        return False
    if np.any(expected == occluder_color):
        return False

    line = sprite_mask[hidden, :] if axis == 0 else sprite_mask[:, hidden]
    if not (1 <= int(line.sum()) < k):
        return False

    union = np.zeros((k, k), dtype=bool)
    signatures = set()
    visible_count = 0
    for br, rs in enumerate(starts):
        for bc, cs in enumerate(starts):
            if _is_hidden(axis, hidden, br, bc):
                continue
            actual = inp[rs : rs + k, cs : cs + k] == sprite_color
            if sprite_mask[br, bc]:
                visible_count += 1
                want = sprite_mask.copy()
                want[br, bc] = False
                if not np.array_equal(actual, want):
                    return False
                if np.array_equal(actual, sprite_mask):
                    return False
                union |= actual
                signatures.add(actual.tobytes())
            elif np.any(actual):
                return False

    if visible_count < 2 or len(signatures) != visible_count:
        return False
    if not np.array_equal(union, sprite_mask):
        return False
    if np.array_equal(inp, expected):
        return False
    return np.array_equal(expected, _render_answer(sprite_mask, starts, bg, sprite_color))


def _gen_attempt(axis: int) -> np.ndarray | None:
    k = random.choice(_K_CHOICES)
    max_gap = (30 - k * k) // (k - 1)
    if max_gap < 1:
        return None
    gap = random.randint(1, min(3, max_gap))
    starts = _starts(k, gap)

    sprite_mask = _random_connected_mask(k)
    hidden_options = _valid_hidden_indices(sprite_mask, axis)
    if not hidden_options:
        return None
    hidden = random.choice(hidden_options)

    bg, sprite_color, occluder_color = random.sample(_COLOR_POOL, 3)
    expected = _render_answer(sprite_mask, starts, bg, sprite_color)
    inp = _apply_occluder(expected, k, starts, axis, hidden, occluder_color)

    if not _semantic_checks(
        inp,
        expected,
        sprite_mask,
        starts,
        axis,
        hidden,
        bg,
        sprite_color,
        occluder_color,
    ):
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    global _GEN_CALL_INDEX

    preferred_axis = _GEN_CALL_INDEX % 2
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt(preferred_axis)
        if grid is not None:
            _GEN_CALL_INDEX += 1
            return grid

    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt(random.randrange(2))
        if grid is not None:
            _GEN_CALL_INDEX += 1
            return grid

    raise RuntimeError("failed to generate a valid coordinate-hole lattice input")
