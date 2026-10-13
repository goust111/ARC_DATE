from __future__ import annotations

import itertools

import numpy as np

from arc_utils import Color, colors_present, get_bg


def _active_runs(active: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(active):
        if bool(value) and start is None:
            start = idx
        elif not bool(value) and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(active) - 1))
    return runs


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask)


def _candidate_key(candidate: tuple[int, int, int]) -> tuple[int, int, int]:
    i, j, k = candidate
    a, b = sorted((i, j))
    return a, b, k


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the two sprite masks whose pixelwise union equals a third sprite mask.

    Input structure
    - The grid has black background and red foreground arranged as six 4x4
      sprite regions in a 3-by-2 array, separated by blank rows and columns.
    - Each sprite region contains a non-empty red mask.
    - Exactly one sprite is the pixelwise OR of two other sprite masks, except
      that duplicate operand copies are resolved by row-major position.

    Rule
    - Find two operand sprites whose foreground masks each contribute at least
      one unique pixel and whose union exactly matches a destination sprite.
    - Recolor the operand sprite pixels teal and recolor the destination sprite
      pixels green; leave all other pixels unchanged.

    Key insight
    - The three changed sprites are identified by a mask-union relation; position
      matters only when identical operand copies create the same relation.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    assert bg == Color.BLACK, "expected black background"
    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 1, "expected one foreground color"
    fg = fg_colors[0]
    assert fg == Color.RED, "expected red input sprites"

    row_runs = _active_runs(np.any(g != bg, axis=1))
    col_runs = _active_runs(np.any(g != bg, axis=0))
    assert len(row_runs) == 3 and len(col_runs) == 2, "expected a 3-by-2 sprite array"
    assert all(r1 - r0 + 1 == 4 for r0, r1 in row_runs), "sprite rows must be 4 high"
    assert all(c1 - c0 + 1 == 4 for c0, c1 in col_runs), "sprite columns must be 4 wide"

    windows: list[tuple[int, int, int, int]] = []
    masks: list[np.ndarray] = []
    for r0, r1 in row_runs:
        for c0, c1 in col_runs:
            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            assert patch.shape == (4, 4), "expected 4x4 sprite window"
            assert np.all((patch == bg) | (patch == fg)), (
                "sprite windows must use bg and foreground only"
            )
            mask = patch != bg
            assert np.any(mask), "sprite window must not be empty"
            windows.append((r0, r1, c0, c1))
            masks.append(mask)

    candidates: list[tuple[int, int, int]] = []
    for dst in range(len(masks)):
        source_indexes = [idx for idx in range(len(masks)) if idx != dst]
        for src_a, src_b in itertools.combinations(source_indexes, 2):
            mask_a = masks[src_a]
            mask_b = masks[src_b]
            if not np.array_equal(mask_a | mask_b, masks[dst]):
                continue
            if not (np.any(mask_a & ~mask_b) and np.any(mask_b & ~mask_a)):
                continue
            candidates.append((src_a, src_b, dst))

    assert candidates, "expected a two-sprite union relation"
    if len(candidates) > 1:
        # Ambiguity is allowed only when duplicate operand copies make the same relation.
        assert len({dst for _, _, dst in candidates}) == 1, "ambiguous union destination"
        pair_shape_keys = {
            tuple(sorted((_mask_key(masks[src_a]), _mask_key(masks[src_b]))))
            for src_a, src_b, _ in candidates
        }
        assert len(pair_shape_keys) == 1, "ambiguous operand masks"

    best_key = min(_candidate_key(candidate) for candidate in candidates)
    selected = [candidate for candidate in candidates if _candidate_key(candidate) == best_key]
    assert len(selected) == 1, "expected a unique row-major union relation"
    src_a, src_b, dst = selected[0]

    out = g.copy()
    for idx, color in ((src_a, Color.TEAL), (src_b, Color.TEAL), (dst, Color.GREEN)):
        r0, r1, c0, c1 = windows[idx]
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[masks[idx]] = int(color)
    return out

# --------------- generator (o5x) ---------------

# Canvas and fixed 3x2 cell layout.
_GEN_GRID_SHAPE = (16, 11)
_GEN_CELL_SIZE = 4
_GEN_CELL_AREA = _GEN_CELL_SIZE * _GEN_CELL_SIZE
_GEN_CELL_ORIGINS = ((1, 1), (1, 6), (6, 1), (6, 6), (11, 1), (11, 6))
_GEN_SPRITE_COUNT = len(_GEN_CELL_ORIGINS)

# Sprite density and operand-union constraints.
_GEN_MIN_PIXELS, _GEN_MAX_PIXELS = 7, 14
_GEN_UNION_MIN_PIXELS = 8
_GEN_OVERLAP_MIN_PIXELS, _GEN_OVERLAP_MAX_PIXELS = 4, 12
_GEN_MIN_UNIQUE_PIXELS = 1

# Role order is operands, destination, then three distractors.
_GEN_OPERAND_A_IDX, _GEN_OPERAND_B_IDX, _GEN_DESTINATION_IDX = 0, 1, 2
_GEN_DISTRACTOR_COUNT = 3
_GEN_EXPECTED_UNION_CANDIDATES = [(_GEN_OPERAND_A_IDX, _GEN_OPERAND_B_IDX, _GEN_DESTINATION_IDX)]
_GEN_OUTPUT_COLORS = (int(Color.TEAL), int(Color.TEAL), int(Color.GREEN)) + (int(Color.RED),) * 3
_GEN_BG_COLOR, _GEN_INPUT_COLOR = int(Color.BLACK), int(Color.RED)

# Retry budget for rejecting ambiguous random masks.
_GEN_RETRY_LIMIT = 300


def _gen_mask_from_positions(positions) -> np.ndarray:
    mask = np.zeros((_GEN_CELL_SIZE, _GEN_CELL_SIZE), dtype=bool)
    mask.flat[np.asarray(positions, dtype=int)] = True
    return mask


def _gen_mask_key(mask: np.ndarray) -> tuple[tuple[bool, ...], ...]:
    return tuple(tuple(bool(v) for v in row) for row in mask)


def _gen_positions_without(positions: np.ndarray, excluded: np.ndarray) -> np.ndarray:
    excluded_set = set(excluded)
    return np.array([p for p in positions if p not in excluded_set], dtype=int)


def _gen_operand_unique_choices(overlap: int) -> list[tuple[int, int, int]]:
    choices = []
    min_unique = max(_GEN_MIN_UNIQUE_PIXELS, _GEN_MIN_PIXELS - overlap)
    max_unique = _GEN_MAX_PIXELS - overlap
    for a_unique in range(min_unique, max_unique + 1):
        for b_unique in range(min_unique, max_unique + 1):
            union_count = overlap + a_unique + b_unique
            if _GEN_UNION_MIN_PIXELS <= union_count <= _GEN_MAX_PIXELS:
                choices.append((a_unique, b_unique, union_count))
    return choices


def _gen_operand_masks() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    overlap = int(np.random.randint(_GEN_OVERLAP_MIN_PIXELS, _GEN_OVERLAP_MAX_PIXELS + 1))
    choices = _gen_operand_unique_choices(overlap)
    a_unique, b_unique, union_count = choices[int(np.random.randint(len(choices)))]
    union_pos = np.random.choice(_GEN_CELL_AREA, size=union_count, replace=False)
    overlap_pos = np.random.choice(union_pos, size=overlap, replace=False)
    remaining = _gen_positions_without(union_pos, overlap_pos)
    a_only = np.random.choice(remaining, size=a_unique, replace=False)
    b_only = _gen_positions_without(remaining, a_only)

    a = _gen_mask_from_positions(np.concatenate([overlap_pos, a_only]))
    b = _gen_mask_from_positions(np.concatenate([overlap_pos, b_only]))
    d = a | b
    return a, b, d


def _gen_random_sprite_mask() -> np.ndarray:
    count = int(np.random.randint(_GEN_MIN_PIXELS, _GEN_MAX_PIXELS + 1))
    return _gen_mask_from_positions(np.random.choice(_GEN_CELL_AREA, size=count, replace=False))


def _gen_union_candidates(masks: list[np.ndarray]) -> list[tuple[int, int, int]]:
    candidates = []
    for dst, dst_mask in enumerate(masks):
        for i in range(len(masks)):
            if i == dst:
                continue
            for j in range(i + 1, len(masks)):
                if j == dst:
                    continue
                mask_i = masks[i]
                mask_j = masks[j]
                if not np.array_equal(mask_i | mask_j, dst_mask):
                    continue
                if np.any(mask_i & ~mask_j) and np.any(mask_j & ~mask_i):
                    candidates.append((i, j, dst))
    return candidates


def _gen_masks_are_valid(masks: list[np.ndarray]) -> bool:
    counts = [int(np.sum(mask)) for mask in masks]
    if any(count < _GEN_MIN_PIXELS or count > _GEN_MAX_PIXELS for count in counts):
        return False
    if len({_gen_mask_key(mask) for mask in masks}) != len(masks):
        return False
    return _gen_union_candidates(masks) == _GEN_EXPECTED_UNION_CANDIDATES


def _gen_build_output(masks: list[np.ndarray]) -> np.ndarray:
    out = np.zeros(_GEN_GRID_SHAPE, dtype=int)
    for pos_idx, role_idx in enumerate(np.random.permutation(_GEN_SPRITE_COUNT)):
        r0, c0 = _GEN_CELL_ORIGINS[pos_idx]
        role_idx = int(role_idx)
        patch = out[r0 : r0 + _GEN_CELL_SIZE, c0 : c0 + _GEN_CELL_SIZE]
        patch[masks[role_idx]] = _GEN_OUTPUT_COLORS[role_idx]
    return out


def _gen_attempt() -> np.ndarray | None:
    a, b, d = _gen_operand_masks()
    masks = [a, b, d] + [_gen_random_sprite_mask() for _ in range(_GEN_DISTRACTOR_COUNT)]
    if not _gen_masks_are_valid(masks):
        return None

    out = _gen_build_output(masks)
    inp = np.where(out == _GEN_BG_COLOR, _GEN_BG_COLOR, _GEN_INPUT_COLOR).astype(int)
    try:
        if not np.array_equal(main(inp), out):
            return None
    except AssertionError:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid fea12743 input")
