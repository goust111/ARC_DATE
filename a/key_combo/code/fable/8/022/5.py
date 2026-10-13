from __future__ import annotations

import itertools
from collections import Counter

import numpy as np


BLUE = 1
TRANSFORM_COUNT = 4
MAX_GENERATION_ATTEMPTS = 400
LAYOUTS = ((2, 2), (2, 3), (3, 2))
INPUT_COLORS = (0, 2, 3, 4, 5, 6, 7, 8, 9)


def _apply_transform(tile: np.ndarray, transform_id: int) -> np.ndarray:
    transform_id = int(transform_id)
    if transform_id == 0:
        return tile.copy()
    if transform_id == 1:
        return np.fliplr(tile)
    if transform_id == 2:
        return np.flipud(tile)
    if transform_id == 3:
        return np.flipud(np.fliplr(tile))
    raise ValueError("unknown transform")


def _find_separator_color(grid: np.ndarray) -> int:
    row_colors = [int(grid[r, 0]) for r in range(grid.shape[0]) if np.all(grid[r, :] == grid[r, 0])]
    col_colors = [int(grid[0, c]) for c in range(grid.shape[1]) if np.all(grid[:, c] == grid[0, c])]
    candidates = set(row_colors) & set(col_colors)
    assert candidates, "expected a divider color with full rows and columns"
    counts = Counter(row_colors + col_colors)
    best = max(candidates, key=lambda color: (counts[color], color))
    return int(best)


def _segments(length: int, separators: list[int]) -> list[tuple[int, int]]:
    parts: list[tuple[int, int]] = []
    start = 0
    for idx in separators + [length]:
        if idx > start:
            parts.append((start, idx))
        start = idx + 1
    return parts


def _parse_blocks(grid: np.ndarray) -> tuple[list[np.ndarray], tuple[int, int], int]:
    sep = _find_separator_color(grid)
    row_seps = [int(r) for r in range(grid.shape[0]) if np.all(grid[r, :] == sep)]
    col_seps = [int(c) for c in range(grid.shape[1]) if np.all(grid[:, c] == sep)]
    assert row_seps and col_seps, "expected both row and column separators"

    row_parts = _segments(grid.shape[0], row_seps)
    col_parts = _segments(grid.shape[1], col_seps)
    assert len(row_parts) >= 2 and len(col_parts) >= 2, "expected a block grid"

    block_hs = {r1 - r0 for r0, r1 in row_parts}
    block_ws = {c1 - c0 for c0, c1 in col_parts}
    assert len(block_hs) == 1 and len(block_ws) == 1, "blocks must have equal size"
    block_h = next(iter(block_hs))
    block_w = next(iter(block_ws))

    blocks = [
        grid[r0:r1, c0:c1].copy()
        for r0, r1 in row_parts
        for c0, c1 in col_parts
    ]
    return blocks, (block_h, block_w), sep


def _majority_consensus(aligned: np.ndarray) -> tuple[np.ndarray, int]:
    colors = np.unique(aligned)
    counts = np.stack([(aligned == color).sum(axis=0) for color in colors], axis=0)
    best_idx = counts.argmax(axis=0)
    consensus = colors[best_idx].astype(int)
    score = int(counts.max(axis=0).sum())
    return consensus, score


def _candidate_from_assignment(
    blocks: list[np.ndarray],
    variants: np.ndarray,
    assignment: tuple[int, ...],
    bg: int,
) -> dict | None:
    aligned = np.stack([variants[i, assignment[i]] for i in range(len(blocks))], axis=0)
    consensus, score = _majority_consensus(aligned)

    damaged: list[tuple[int, np.ndarray]] = []
    for idx in range(len(blocks)):
        mismatch = aligned[idx] != consensus
        hard = mismatch & (((aligned[idx] != bg) & (consensus != bg)) | ((aligned[idx] != bg) & (consensus == bg)))
        if np.any(hard):
            return None
        soft = mismatch & (aligned[idx] == bg) & (consensus != bg)
        if np.any(soft):
            damaged.append((idx, soft))

    if len(damaged) != 1:
        return None

    damaged_index, repaired_mask_aligned = damaged[0]
    revert_transform = int(assignment[damaged_index])
    restored = _apply_transform(consensus, revert_transform)
    repaired_mask = _apply_transform(repaired_mask_aligned.astype(int), revert_transform).astype(bool)
    restored[repaired_mask] = BLUE

    return {
        "score": score,
        "output": restored,
        "consensus": consensus,
        "assignment": assignment,
        "damaged_index": damaged_index,
        "repaired_mask_aligned": repaired_mask_aligned,
        "repaired_mask": repaired_mask,
    }


def _solve(grid: np.ndarray, *, with_debug: bool = False):
    blocks, block_shape, sep = _parse_blocks(grid)
    del block_shape, sep

    separator = _find_separator_color(grid)
    non_sep = grid[grid != separator]
    bg = int(Counter(map(int, non_sep.ravel())).most_common(1)[0][0])

    variants = np.stack(
        [[_apply_transform(block, transform_id) for transform_id in range(TRANSFORM_COUNT)] for block in blocks],
        axis=0,
    )

    best_score = None
    best_candidates: list[dict] = []
    for tail in itertools.product(range(TRANSFORM_COUNT), repeat=len(blocks) - 1):
        assignment = (0, *tail)
        candidate = _candidate_from_assignment(blocks, variants, assignment, bg)
        if candidate is None:
            continue
        score = int(candidate["score"])
        if best_score is None or score > best_score:
            best_score = score
            best_candidates = [candidate]
        elif score == best_score:
            best_candidates.append(candidate)

    assert best_candidates, "expected at least one valid orientation assignment"

    output_map: dict[bytes, dict] = {}
    for candidate in best_candidates:
        key = candidate["output"].tobytes()
        output_map.setdefault(key, candidate)
    assert len(output_map) == 1, "best assignments must imply a unique repaired block"

    winner = next(iter(output_map.values()))
    if with_debug:
        return winner["output"], {
            "bg": bg,
            "best_score": int(best_score),
            "best_assignment_count": len(best_candidates),
            "unique_output_count": len(output_map),
            "damaged_index": int(winner["damaged_index"]),
            "assignment": tuple(int(x) for x in winner["assignment"]),
            "consensus": winner["consensus"].copy(),
            "repaired_mask": winner["repaired_mask"].copy(),
            "blocks": [block.copy() for block in blocks],
        }
    return winner["output"]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the uniquely damaged mirrored tile by aligning all blocks to a common pose, voting a consensus, then returning the fix in the damaged block's own pose.

    Input structure:
    - Full separator rows and columns split the canvas into equal rectangular blocks.
    - Every block shows the same multicolor pattern under one of four reflection-equivalent poses: identity, left-right mirror, up-down mirror, or 180-degree rotation.
    - Exactly one block is damaged because some non-background pattern pixels were erased back to the pattern background; blue never appears in the input.

    Rule:
    - Choose one pose for each block so the aligned blocks agree as much as possible.
    - Take a per-cell majority vote over the aligned blocks to recover the consensus pattern.
    - Find the unique block whose disagreements are only missing foreground pixels, transform the consensus back into that block's original pose, and recolor only the repaired cells blue.

    Key insight:
    - The redundancy is hidden behind mirrors: the blocks must be pose-normalized before voting, and the voted answer must then be put back into the damaged block's original pose.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    return _solve(g)


def _random_pattern(rng: np.random.Generator, h: int, w: int, bg: int, colors: list[int]) -> np.ndarray | None:
    area = h * w
    color_count = int(rng.integers(2, min(4, len(colors)) + 1))
    fg_colors = [int(x) for x in rng.choice(colors, size=color_count, replace=False)]
    fg_total = int(rng.integers(max(6, area // 5), min(area // 2, area - 2) + 1))
    pattern = np.full((h, w), bg, dtype=int)

    coords = [(r, c) for r in range(h) for c in range(w)]
    rng.shuffle(coords)
    selected = coords[:fg_total]
    for i, (r, c) in enumerate(selected):
        if i < color_count:
            pattern[r, c] = fg_colors[i]
        else:
            pattern[r, c] = fg_colors[int(rng.integers(color_count))]

    if np.count_nonzero(pattern != bg) != fg_total:
        return None
    if len({int(x) for x in pattern.ravel()}) < 3:
        return None

    diffs = [int(np.count_nonzero(pattern != _apply_transform(pattern, t))) for t in range(1, TRANSFORM_COUNT)]
    if min(diffs) < max(4, fg_total // 3):
        return None

    if not any(pattern[0, :] != bg) and not any(pattern[-1, :] != bg) and not any(pattern[:, 0] != bg) and not any(pattern[:, -1] != bg):
        return None
    return pattern


def _build_grid(
    pattern: np.ndarray,
    layout: tuple[int, int],
    orientations: list[int],
    damaged_index: int,
    removed_coords: list[tuple[int, int]],
    sep: int,
    bg: int,
) -> tuple[np.ndarray, np.ndarray]:
    block_h, block_w = pattern.shape
    rows, cols = layout
    grid_h = rows * block_h + (rows - 1)
    grid_w = cols * block_w + (cols - 1)
    grid = np.full((grid_h, grid_w), sep, dtype=int)
    full_blocks: list[np.ndarray] = []

    for idx, orient in enumerate(orientations):
        block = _apply_transform(pattern, orient)
        full_blocks.append(block.copy())
        rr = idx // cols
        cc = idx % cols
        r0 = rr * (block_h + 1)
        c0 = cc * (block_w + 1)
        grid[r0:r0 + block_h, c0:c0 + block_w] = block

    damaged_block = full_blocks[damaged_index].copy()
    for r, c in removed_coords:
        damaged_block[r, c] = bg

    rr = damaged_index // cols
    cc = damaged_index % cols
    r0 = rr * (block_h + 1)
    c0 = cc * (block_w + 1)
    grid[r0:r0 + block_h, c0:c0 + block_w] = damaged_block

    output = full_blocks[damaged_index].copy()
    for r, c in removed_coords:
        output[r, c] = BLUE
    return grid, output


def _unaligned_vote_output(grid: np.ndarray, bg: int) -> np.ndarray:
    blocks, _, _ = _parse_blocks(grid)
    raw = np.stack(blocks, axis=0)
    consensus, _ = _majority_consensus(raw)
    differing = []
    for idx, block in enumerate(blocks):
        mismatch = block != consensus
        soft = mismatch & (block == bg) & (consensus != bg)
        hard = mismatch & ~soft
        if not np.any(hard) and np.any(soft):
            differing.append((idx, soft))
    if len(differing) != 1:
        return consensus
    idx, repaired = differing[0]
    out = consensus.copy()
    out[repaired] = BLUE
    del idx
    return out


def _semantic_checks(
    grid: np.ndarray,
    expected: np.ndarray,
    *,
    damaged_index: int,
    layout: tuple[int, int],
    orientations: list[int],
) -> bool:
    solved, debug = _solve(grid, with_debug=True)
    if not np.array_equal(solved, expected):
        return False
    if debug["damaged_index"] != damaged_index:
        return False
    if debug["unique_output_count"] != 1:
        return False
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False
    if BLUE in set(map(int, grid.ravel())):
        return False
    if len(set(orientations)) < 2:
        return False
    if orientations[damaged_index] == 0:
        return False

    rows, cols = layout
    if rows * cols != len(orientations):
        return False

    non_sep = grid[grid != _find_separator_color(grid)]
    bg = int(Counter(map(int, non_sep.ravel())).most_common(1)[0][0])
    naive = _unaligned_vote_output(grid, bg)
    if np.array_equal(naive, expected):
        return False

    canonical = debug["consensus"]
    if np.array_equal(canonical, expected):
        return False

    blue_count = int(np.count_nonzero(expected == BLUE))
    if blue_count == 0:
        return False

    blocks, _, _ = _parse_blocks(grid)
    differing_blocks = 0
    for idx, block in enumerate(blocks):
        full = _apply_transform(canonical, debug["assignment"][idx])
        mismatch = full != block
        if np.any(mismatch):
            if idx != damaged_index:
                return False
            if not np.all((block == bg)[mismatch] & (full != bg)[mismatch]):
                return False
            differing_blocks += 1
    return differing_blocks == 1


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    rows, cols = LAYOUTS[int(rng.integers(len(LAYOUTS)))]
    max_h = min(8, (30 - (rows - 1)) // rows)
    max_w = min(8, (30 - (cols - 1)) // cols)
    block_h = int(rng.integers(5, max_h + 1))
    block_w = int(rng.integers(5, max_w + 1))

    bg = int(rng.choice(INPUT_COLORS))
    remaining = [color for color in INPUT_COLORS if color != bg]
    sep = int(rng.choice(remaining))
    paint_colors = [color for color in remaining if color != sep]

    pattern = _random_pattern(rng, block_h, block_w, bg, paint_colors)
    if pattern is None:
        return None

    block_count = rows * cols
    orientations = [int(x) for x in rng.integers(0, TRANSFORM_COUNT, size=block_count)]
    if len(set(orientations)) < 2:
        return None

    damaged_index = int(rng.integers(block_count))
    if orientations[damaged_index] == 0:
        return None

    fg_coords = [tuple(map(int, coord)) for coord in np.argwhere(_apply_transform(pattern, orientations[damaged_index]) != bg)]
    if len(fg_coords) < 3:
        return None
    remove_count = int(rng.integers(1, min(len(fg_coords) // 2, 6) + 1))
    chosen = rng.choice(len(fg_coords), size=remove_count, replace=False)
    removed_coords = [fg_coords[int(i)] for i in np.atleast_1d(chosen)]

    grid, expected = _build_grid(pattern, (rows, cols), orientations, damaged_index, removed_coords, sep, bg)
    if not _semantic_checks(grid, expected, damaged_index=damaged_index, layout=(rows, cols), orientations=orientations):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-consensus repair input")
