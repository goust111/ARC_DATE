import numpy as np


BACKGROUND = 0
RED = 2
GREY = 5
TEAL = 8

MIN_BLOCKS = 5
MAX_BLOCKS = 7
GENERATION_ATTEMPTS = 500


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components of a boolean mask as (N, 2) coord arrays."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _find_red_blocks(top: np.ndarray) -> list[tuple[int, int]]:
    """Find exact 2x2 red components in the upper half and return their top-left coords."""
    blocks = []
    for coords in _components(top == RED):
        if coords.shape[0] != 4:
            continue
        r0 = int(coords[:, 0].min())
        c0 = int(coords[:, 1].min())
        r1 = int(coords[:, 0].max())
        c1 = int(coords[:, 1].max())
        if r1 - r0 != 1 or c1 - c0 != 1:
            continue
        if np.all(top[r0 : r0 + 2, c0 : c0 + 2] == RED):
            blocks.append((r0, c0))
    return blocks


def main(grid: np.ndarray) -> np.ndarray:
    """Use the XOR difference between aligned grey half-masks to choose which upper 2x2 red blocks join a right-to-left alternating recolor.

    Input structure:
    The grid has even height and is split horizontally into equal top and bottom halves on a black background. Both halves contain irregular grey masks at aligned coordinates, and only the top half also contains several disjoint 2x2 red blocks with distinct left columns.

    Rule:
    Align the two halves by row offset and column. Build an implicit activation mask at positions where exactly one aligned half-cell is grey. Among the top-half red 2x2 blocks, keep only those whose top-left probe cell lies on an activated position. Sort these activated blocks from right to left, recolor ranks 0, 2, 4, ... to teal, leave odd-ranked activated blocks red, and leave every non-activated block red. Keep the bottom half unchanged.

    Key insight:
    The lower grey mask is not decorative: after aligned XOR filtering, it defines the subset of red blocks that participate in the familiar right-to-left alternating recolor pattern.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or g.shape[0] % 2 != 0:
        raise ValueError("grid must be a 2D array with even height")

    half = g.shape[0] // 2
    top = g[:half, :]
    bottom = g[half:, :]
    out = g.copy()

    active_blocks = []
    for r0, c0 in _find_red_blocks(top):
        top_active = bool(top[r0, c0] == GREY)
        bottom_active = bool(bottom[r0, c0] == GREY)
        if top_active ^ bottom_active:
            active_blocks.append((r0, c0))

    for rank, (r0, c0) in enumerate(sorted(active_blocks, key=lambda rc: rc[1], reverse=True)):
        if rank % 2 == 0:
            out[r0 : r0 + 2, c0 : c0 + 2] = TEAL
    return out


def _neighbor_sum(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    padded = np.pad(mask.astype(int), 1)
    total = np.zeros((h, w), dtype=int)
    for dr in range(3):
        for dc in range(3):
            total += padded[dr : dr + h, dc : dc + w]
    return total


def _smooth_blob(height: int, width: int, density: float, rng: np.random.Generator) -> np.ndarray:
    mask = rng.random((height, width)) < density
    for _ in range(int(rng.integers(2, 5))):
        counts = _neighbor_sum(mask)
        mask = np.where(mask, counts >= 4, counts >= 5)
    if not mask.any():
        mask[int(rng.integers(height)), int(rng.integers(width))] = True
    return mask


def _blocks_conflict(a: tuple[int, int], b: tuple[int, int]) -> bool:
    ar, ac = a
    br, bc = b
    dc = abs(ac - bc)
    if dc <= 1:
        return True
    if dc == 2 and not (ar + 1 < br or br + 1 < ar):
        return True
    return False


def _pick_block_positions(
    half: int,
    width: int,
    count: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]] | None:
    candidates = [(r, c) for r in range(half - 1) for c in range(width - 1)]
    for _ in range(100):
        order = rng.permutation(len(candidates))
        chosen: list[tuple[int, int]] = []
        for idx in order:
            pos = candidates[int(idx)]
            if any(_blocks_conflict(pos, other) for other in chosen):
                continue
            chosen.append(pos)
            if len(chosen) == count:
                return chosen
    return None


def _make_bottom_mask(
    half: int,
    width: int,
    blocks: list[tuple[int, int]],
    active_idx: set[int],
    rng: np.random.Generator,
) -> np.ndarray | None:
    on_cells = [blocks[i] for i in sorted(active_idx)]
    off_cells = [blocks[i] for i in range(len(blocks)) if i not in active_idx]
    for _ in range(60):
        bottom = _smooth_blob(half, width, float(rng.uniform(0.30, 0.42)), rng)
        if rng.random() < 0.5:
            bottom |= _smooth_blob(half, width, float(rng.uniform(0.04, 0.10)), rng)
        for r0, c0 in on_cells:
            bottom[r0, c0] = True
        for r0, c0 in off_cells:
            bottom[r0, c0] = False
        if 0.30 <= float(bottom.mean()) <= 0.50:
            return bottom
    return None


def _make_top_mask(
    half: int,
    width: int,
    bottom: np.ndarray,
    blocks: list[tuple[int, int]],
    rng: np.random.Generator,
) -> np.ndarray | None:
    area = half * width
    for _ in range(80):
        top = _smooth_blob(half, width, float(rng.uniform(0.30, 0.44)), rng)
        if rng.random() < 0.5:
            top |= _smooth_blob(half, width, float(rng.uniform(0.04, 0.12)), rng)
        if len(blocks) >= 2:
            top[blocks[0][0], blocks[0][1]] = True
            r1, c1 = blocks[1]
            top[r1 : r1 + 2, c1 : c1 + 2] = False

        top_density = float(top.mean())
        overlap = int(np.count_nonzero(top & bottom))
        top_only = int(np.count_nonzero(top & ~bottom))
        bottom_only = int(np.count_nonzero(bottom & ~top))
        overlap_flags = [bool(top[r0 : r0 + 2, c0 : c0 + 2].any()) for r0, c0 in blocks]
        if not (0.30 <= top_density <= 0.50):
            continue
        if not (any(overlap_flags) and not all(overlap_flags)):
            continue
        if overlap < max(6, area // 10):
            continue
        if top_only < 4 or bottom_only < 4:
            continue
        if top_only + bottom_only < max(10, area // 6):
            continue
        return top
    return None


def _render_input(top_mask: np.ndarray, bottom_mask: np.ndarray, blocks: list[tuple[int, int]]) -> np.ndarray:
    half, width = top_mask.shape
    grid = np.zeros((half * 2, width), dtype=int)
    grid[:half, :][top_mask] = GREY
    grid[half:, :][bottom_mask] = GREY
    for r0, c0 in blocks:
        grid[r0 : r0 + 2, c0 : c0 + 2] = RED
    return grid


def _expected_output(inp: np.ndarray, blocks: list[tuple[int, int]]) -> np.ndarray:
    out = inp.copy()
    half = inp.shape[0] // 2
    active_blocks = []
    for r0, c0 in blocks:
        if bool(inp[r0, c0] == GREY) ^ bool(inp[half + r0, c0] == GREY):
            active_blocks.append((r0, c0))
    for rank, (r0, c0) in enumerate(sorted(active_blocks, key=lambda rc: rc[1], reverse=True)):
        if rank % 2 == 0:
            out[r0 : r0 + 2, c0 : c0 + 2] = TEAL
    return out


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    half = int(rng.integers(6, 10))
    width = int(rng.integers(12, 18))
    block_count = int(rng.integers(MIN_BLOCKS, MAX_BLOCKS + 1))
    blocks = _pick_block_positions(half, width, block_count, rng)
    if blocks is None:
        return None

    active_count = int(rng.integers(3, block_count))
    active_indices = set(map(int, rng.choice(block_count, size=active_count, replace=False)))

    bottom_mask = _make_bottom_mask(half, width, blocks, active_indices, rng)
    if bottom_mask is None:
        return None
    top_mask = _make_top_mask(half, width, bottom_mask, blocks, rng)
    if top_mask is None:
        return None

    inp = _render_input(top_mask, bottom_mask, blocks)
    visible_top_density = float(np.mean(inp[:half, :] == GREY))
    if not (0.18 <= visible_top_density <= 0.45):
        return None

    detected = _find_red_blocks(inp[:half, :])
    if len(detected) != block_count or len({c for _, c in detected}) != block_count:
        return None

    observed_active_count = sum(bool(inp[half + r0, c0] == GREY) for r0, c0 in blocks)
    if observed_active_count != active_count:
        return None

    expected = _expected_output(inp, blocks)
    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
