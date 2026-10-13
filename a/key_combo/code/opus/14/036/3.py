from __future__ import annotations

import hashlib
import random

import numpy as np


BG = 0
FG_PALETTE = (1, 2, 3, 4)
MAX_ATTEMPTS = 800
_SEEN_OUTPUTS: set[str] = set()


def _grid_signature(grid: np.ndarray) -> str:
    g = np.asarray(grid, dtype=int)
    return hashlib.md5(g.tobytes() + str(tuple(map(int, g.shape))).encode()).hexdigest()


def _run_starts(grid: np.ndarray, axis: int) -> list[int]:
    length = int(grid.shape[axis])
    starts = [0]
    for i in range(1, length):
        prev = grid[i - 1, :] if axis == 0 else grid[:, i - 1]
        cur = grid[i, :] if axis == 0 else grid[:, i]
        if not np.array_equal(cur, prev):
            starts.append(i)
    return starts


def _compress(grid: np.ndarray) -> np.ndarray:
    row_starts = _run_starts(grid, axis=0)
    col_starts = _run_starts(grid, axis=1)
    return np.asarray(grid, dtype=int)[np.ix_(row_starts, col_starts)].copy()


def _apply_variant(block: np.ndarray, code: str) -> np.ndarray:
    if code == "id":
        return block.copy()
    if code == "h":
        return np.fliplr(block)
    if code == "v":
        return np.flipud(block)
    if code == "r":
        return np.rot90(block, 2)
    raise ValueError(f"unknown variant code: {code}")


def _matches_base(base: np.ndarray, block: np.ndarray) -> bool:
    return any(
        np.array_equal(block, candidate)
        for candidate in (
            base,
            np.fliplr(base),
            np.flipud(base),
            np.rot90(base, 2),
        )
    )


def _smallest_period_width(grid: np.ndarray) -> int:
    g = np.asarray(grid, dtype=int)
    _, width = g.shape
    for period in range(1, width + 1):
        if width % period != 0:
            continue
        base = g[:, :period]
        ok = True
        for start in range(0, width, period):
            block = g[:, start : start + period]
            if not _matches_base(base, block):
                ok = False
                break
        if ok:
            return period
    raise AssertionError("expected at least the full width to be a valid period")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress consecutive duplicate rows/columns, then extract the smallest leftmost block whose reflected copies tile the compressed grid.

    Input structure:
    - The input is a black-background grid whose rows and columns have been run-length expanded,
      so many adjacent rows and columns are exact duplicates.
    - After collapsing those duplicate runs, the remaining grid is made of equal-width horizontal
      blocks built from the same multicolor sprite.
    - Each later block matches the first block directly, or after a left-right flip, an up-down
      flip, or a 180-degree rotation.

    Rule:
    - Keep the first row from each consecutive run of identical rows, and the first column from
      each consecutive run of identical columns.
    - On that compressed grid, test candidate block widths from smallest to largest, considering
      only widths that divide the compressed width.
    - Return the leftmost block for the first width whose equal-width slices are all either the
      same as that block, its horizontal mirror, its vertical mirror, or its 180-degree rotation.

    Key insight:
    - The visible input is a stretched version of a repeated reflected tiling, so the solver must
      first undo the stretch and only then search for the true smallest mirrored period.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "input must be a non-empty 2-D grid"

    compressed = _compress(g)
    period = _smallest_period_width(compressed)
    return compressed[:, :period].copy()


def _sample_output_block() -> np.ndarray | None:
    height = random.randint(3, 5)
    width = random.randint(2, 4)
    color_count = random.randint(2, 3)
    palette = random.sample(FG_PALETTE, color_count)

    for _ in range(200):
        block = np.zeros((height, width), dtype=int)
        probs = [0.30] + [0.70 / color_count] * color_count
        choices = [BG] + palette
        flat = random.choices(choices, weights=probs, k=height * width)
        block[:, :] = np.array(flat, dtype=int).reshape(height, width)

        used = sorted(set(int(v) for v in block.flat if int(v) != BG))
        if len(used) != color_count:
            continue
        if int(np.count_nonzero(block)) < max(4, height):
            continue
        if int(np.count_nonzero(block == BG)) == 0:
            continue
        if any(np.array_equal(block[r], block[r + 1]) for r in range(height - 1)):
            continue
        if any(np.array_equal(block[:, c], block[:, c + 1]) for c in range(width - 1)):
            continue
        if np.array_equal(block, np.fliplr(block)):
            continue
        if np.array_equal(block, np.flipud(block)):
            continue
        if np.array_equal(block, np.rot90(block, 2)):
            continue
        if _smallest_period_width(block) != width:
            continue
        return block
    return None


def _build_compressed_grid(block: np.ndarray) -> tuple[np.ndarray, list[str]] | None:
    width = int(block.shape[1])
    tile_count = random.randint(3, 4)
    scenario = random.choice(
        [
            ["h"],
            ["v"],
            ["r"],
            ["h", "v"],
            ["h", "r"],
            ["v", "r"],
            ["h", "v", "r"],
        ]
    )

    variants = ["id"]
    if tile_count - 1 >= len(scenario):
        extra = scenario[:]
        while len(extra) < tile_count - 1:
            extra.append(random.choice(scenario))
        random.shuffle(extra)
        variants.extend(extra[: tile_count - 1])
    else:
        chosen = random.sample(scenario, tile_count - 1)
        variants.extend(chosen)

    blocks = [_apply_variant(block, code) for code in variants]
    compressed = np.concatenate(blocks, axis=1)

    if not any(not np.array_equal(blocks[0], patch) for patch in blocks[1:]):
        return None
    if _smallest_period_width(compressed) != width:
        return None
    if len(_run_starts(compressed, axis=0)) != compressed.shape[0]:
        return None
    if len(_run_starts(compressed, axis=1)) != compressed.shape[1]:
        return None
    return compressed, variants


def _sample_repeats(length: int, *, min_total: int, max_total: int) -> list[int] | None:
    for _ in range(300):
        repeats = [random.randint(1, 3) for _ in range(length)]
        if max(repeats) == 1:
            continue
        total = sum(repeats)
        if min_total <= total <= max_total:
            return repeats
    return None


def _expand_grid(grid: np.ndarray, row_repeats: list[int], col_repeats: list[int]) -> np.ndarray:
    expanded = np.repeat(np.asarray(grid, dtype=int), row_repeats, axis=0)
    expanded = np.repeat(expanded, col_repeats, axis=1)
    return expanded


def _validate_scene(
    output_block: np.ndarray,
    compressed: np.ndarray,
    expanded: np.ndarray,
    variants: list[str],
) -> bool:
    if max(expanded.shape) > 30 or max(output_block.shape) > 30:
        return False
    if variants[0] != "id" or all(code == "id" for code in variants[1:]):
        return False
    if _grid_signature(main(expanded)) != _grid_signature(output_block):
        return False
    if not np.array_equal(_compress(expanded), compressed):
        return False
    if _smallest_period_width(compressed) != output_block.shape[1]:
        return False
    if _smallest_period_width(expanded) <= output_block.shape[1]:
        return False
    if compressed.shape[1] <= output_block.shape[1]:
        return False
    if compressed.shape == expanded.shape:
        return False
    if len(_run_starts(expanded, axis=0)) == expanded.shape[0]:
        return False
    if len(_run_starts(expanded, axis=1)) == expanded.shape[1]:
        return False
    if len(set(int(v) for v in output_block.flat if int(v) != BG)) < 2:
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    output_block = _sample_output_block()
    if output_block is None:
        return None

    output_sig = _grid_signature(output_block)
    if output_sig in _SEEN_OUTPUTS:
        return None

    built = _build_compressed_grid(output_block)
    if built is None:
        return None
    compressed, variants = built

    row_repeats = _sample_repeats(compressed.shape[0], min_total=10, max_total=24)
    col_repeats = _sample_repeats(compressed.shape[1], min_total=14, max_total=30)
    if row_repeats is None or col_repeats is None:
        return None

    expanded = _expand_grid(compressed, row_repeats, col_repeats)
    if not _validate_scene(output_block, compressed, expanded, variants):
        return None

    _SEEN_OUTPUTS.add(output_sig)
    return expanded


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirrored-period compression sample")
