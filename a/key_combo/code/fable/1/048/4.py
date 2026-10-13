from __future__ import annotations

import random
import sys
from math import isqrt
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists():
    sys.path.append(str(_ARC_UTILS_DIR))

try:
    from arc_utils import bbox, bbox_h, bbox_w, colors_present
except Exception:
    def colors_present(grid: np.ndarray, *, exclude=None) -> list[int]:
        cols = set(map(int, np.asarray(grid).flatten()))
        if exclude is not None:
            cols -= set(map(int, exclude))
        return sorted(cols)

    def bbox(mask: np.ndarray):
        coords = np.argwhere(mask)
        if coords.size == 0:
            return None
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return int(r0), int(c0), int(r1), int(c1)

    def bbox_h(box) -> int:
        return int(box[2] - box[0] + 1)

    def bbox_w(box) -> int:
        return int(box[3] - box[1] + 1)


GEN_RETRIES = 400
PALETTE = tuple(range(10))
BLOCK_SIZES = (3, 4, 5)
CORRUPTION_TYPES = ("recolor_filled", "erase_filled", "occupy_empty")


def _infer_block_size(grid: np.ndarray) -> int:
    h, w = map(int, grid.shape)
    assert h == w, "grid must be square"
    n = isqrt(h)
    assert n * n == h, "side length must be a perfect square"
    return n


def _split_blocks(grid: np.ndarray, n: int) -> np.ndarray:
    return np.asarray(grid).reshape(n, n, n, n).transpose(0, 2, 1, 3)


def _candidate_model(grid: np.ndarray, bg: int) -> dict | None:
    n = _infer_block_size(grid)
    blocks = _split_blocks(grid, n)
    patterns: dict[bytes, np.ndarray] = {}
    counts: dict[bytes, int] = {}

    for br in range(n):
        for bc in range(n):
            block = np.array(blocks[br, bc], copy=True)
            if np.any(block != bg):
                key = block.tobytes()
                patterns.setdefault(key, block)
                counts[key] = counts.get(key, 0) + 1

    if not counts:
        return None

    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    majority_key, support = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0
    if support <= runner_up:
        return None

    seed_block = np.array(patterns[majority_key], copy=True)
    seed_mask = seed_block != bg
    filled = int(seed_mask.sum())
    if filled == 0 or filled == n * n:
        return None

    fg_colors = colors_present(seed_block, exclude=[bg])
    if len(fg_colors) != 1:
        return None

    layout = np.any(blocks != bg, axis=(2, 3))
    layout_diff = int(np.count_nonzero(layout != seed_mask))
    if layout_diff > 1:
        return None

    blank = np.full((n, n), bg, dtype=int)
    mismatches: list[tuple[int, int]] = []
    for br in range(n):
        for bc in range(n):
            expected = seed_block if bool(seed_mask[br, bc]) else blank
            if not np.array_equal(blocks[br, bc], expected):
                mismatches.append((br, bc))

    if len(mismatches) != 1:
        return None

    return {
        "bg": int(bg),
        "n": n,
        "seed_block": seed_block,
        "seed_mask": seed_mask,
        "bad_block": mismatches[0],
        "support": int(support),
        "support_margin": int(support - runner_up),
        "layout_diff": int(layout_diff),
    }


def _infer_model(grid: np.ndarray) -> dict:
    candidates = []
    for bg in colors_present(grid):
        model = _candidate_model(grid, int(bg))
        if model is not None:
            candidates.append(model)

    assert candidates, "failed to infer a consistent fractal model"
    candidates.sort(
        key=lambda model: (
            int(model["support"]),
            int(model["support_margin"]),
            -int(model["layout_diff"]),
        ),
        reverse=True,
    )
    return candidates[0]


def _repair_grid(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    model = _infer_model(g)
    n = int(model["n"])
    bg = int(model["bg"])
    br, bc = model["bad_block"]
    seed_block = np.asarray(model["seed_block"], dtype=int)
    seed_mask = np.asarray(model["seed_mask"], dtype=bool)
    expected = seed_block if bool(seed_mask[br, bc]) else np.full((n, n), bg, dtype=int)

    out = g.copy()
    r0 = br * n
    c0 = bc * n
    out[r0 : r0 + n, c0 : c0 + n] = expected
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the single corrupted block in a second-order self-similar fractal.

    Input structure:
    - The grid is square with side length N^2 and is partitioned into an N x N array of N x N blocks.
    - There exists a monochrome N x N seed pattern: every foreground seed cell expands to a full copy of the seed pattern inside its block, while every background seed cell expands to an all-background block.
    - Exactly one block is corrupted: a filled block may be recolored or erased, or an empty block may contain stray foreground pixels.

    Rule:
    - Enumerate possible background colors and, for each one, collect all non-empty blocks.
    - The majority exact block pattern among those non-empty blocks is the seed block template; its occupied cells also define the correct block-level occupancy layout.
    - Find the unique block whose actual contents disagree with the template implied by that two-scale seed, then redraw that block as either a clean seed copy or a blank block.

    Key insight:
    - The image redundantly describes the same seed twice: once by which top-level blocks are occupied and once by the pixel pattern inside every clean occupied block.
    """
    return _repair_grid(np.asarray(grid, dtype=int))


def _render_seed_block(seed_mask: np.ndarray, bg: int, fg: int) -> np.ndarray:
    return np.where(seed_mask, fg, bg).astype(int)


def _render_fractal(seed_mask: np.ndarray, bg: int, fg: int) -> np.ndarray:
    fractal = np.kron(seed_mask.astype(int), seed_mask.astype(int))
    return np.where(fractal != 0, fg, bg).astype(int)


def _seed_is_good(seed_mask: np.ndarray) -> bool:
    n = int(seed_mask.shape[0])
    filled = int(seed_mask.sum())
    if filled < 3 or filled > n * n - 2:
        return False
    box = bbox(seed_mask)
    if box is None or bbox_h(box) < 2 or bbox_w(box) < 2:
        return False
    if np.unique(np.argwhere(seed_mask)[:, 0]).size < 2:
        return False
    if np.unique(np.argwhere(seed_mask)[:, 1]).size < 2:
        return False
    return True


def _grow_seed(n: int, target: int) -> np.ndarray:
    cells = {(random.randrange(n), random.randrange(n))}
    while len(cells) < target:
        if random.random() < 0.75:
            r, c = random.choice(tuple(cells))
            nbrs = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
            random.shuffle(nbrs)
            placed = False
            for nr, nc in nbrs:
                if 0 <= nr < n and 0 <= nc < n and (nr, nc) not in cells:
                    cells.add((nr, nc))
                    placed = True
                    break
            if placed:
                continue
        cells.add((random.randrange(n), random.randrange(n)))

    seed = np.zeros((n, n), dtype=bool)
    for r, c in cells:
        seed[r, c] = True
    return seed


def _sample_seed(n: int) -> np.ndarray:
    for _ in range(200):
        target = random.randint(3, n * n - 2)
        if random.random() < 0.6:
            seed = _grow_seed(n, target)
        else:
            coords = random.sample([(r, c) for r in range(n) for c in range(n)], target)
            seed = np.zeros((n, n), dtype=bool)
            for r, c in coords:
                seed[r, c] = True
        if _seed_is_good(seed):
            return seed
    raise RuntimeError("failed to sample a valid seed pattern")


def _block_slice(br: int, bc: int, n: int) -> tuple[slice, slice]:
    return slice(br * n, (br + 1) * n), slice(bc * n, (bc + 1) * n)


def _corrupt_filled_block(
    grid: np.ndarray, seed_mask: np.ndarray, pos: tuple[int, int], fg: int, noise: int
) -> np.ndarray:
    n = int(seed_mask.shape[0])
    br, bc = pos
    rs, cs = _block_slice(br, bc, n)
    patch = grid[rs, cs].copy()
    original = patch.copy()
    fg_coords = [tuple(map(int, rc)) for rc in np.argwhere(seed_mask)]
    changed = random.sample(fg_coords, random.randint(1, max(1, len(fg_coords) // 2)))
    for r, c in changed:
        patch[r, c] = noise
    assert not np.array_equal(patch, original)
    grid[rs, cs] = patch
    return grid


def _corrupt_empty_block(
    grid: np.ndarray, seed_mask: np.ndarray, pos: tuple[int, int], fg: int, noise: int
) -> np.ndarray:
    n = int(seed_mask.shape[0])
    br, bc = pos
    rs, cs = _block_slice(br, bc, n)
    patch = grid[rs, cs].copy()
    mode = random.choice(("scatter", "partial_seed", "wrong_full"))

    if mode == "wrong_full":
        patch[:] = np.where(seed_mask, noise if random.random() < 0.7 else fg, patch)
    elif mode == "partial_seed":
        coords = [tuple(map(int, rc)) for rc in np.argwhere(seed_mask)]
        use = random.sample(coords, random.randint(1, len(coords)))
        for r, c in use:
            patch[r, c] = noise if random.random() < 0.5 else fg
    else:
        all_coords = [(r, c) for r in range(n) for c in range(n)]
        use = random.sample(all_coords, random.randint(1, max(2, n)))
        for r, c in use:
            patch[r, c] = noise if random.random() < 0.5 else fg

    assert np.any(patch != grid[rs, cs])
    grid[rs, cs] = patch
    return grid


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(BLOCK_SIZES)
    bg, fg, noise = random.sample(PALETTE, 3)
    seed_mask = _sample_seed(n)
    seed_block = _render_seed_block(seed_mask, bg, fg)
    clean = _render_fractal(seed_mask, bg, fg)

    filled_blocks = [tuple(map(int, rc)) for rc in np.argwhere(seed_mask)]
    empty_blocks = [tuple(map(int, rc)) for rc in np.argwhere(~seed_mask)]
    if len(filled_blocks) < 3 or not empty_blocks:
        return None

    branch = random.choice(CORRUPTION_TYPES)
    if branch in {"recolor_filled", "erase_filled"}:
        bad = random.choice(filled_blocks)
    else:
        bad = random.choice(empty_blocks)

    inp = clean.copy()
    rs, cs = _block_slice(bad[0], bad[1], n)
    if branch == "recolor_filled":
        inp = _corrupt_filled_block(inp, seed_mask, bad, fg, noise)
    elif branch == "erase_filled":
        inp[rs, cs] = bg
    else:
        inp = _corrupt_empty_block(inp, seed_mask, bad, fg, noise)

    if np.array_equal(inp, clean):
        return None

    try:
        inferred = _infer_model(inp)
        repaired = _repair_grid(inp)
    except AssertionError:
        return None

    if tuple(map(int, inferred["bad_block"])) != tuple(map(int, bad)):
        return None
    if not np.array_equal(inferred["seed_block"], seed_block):
        return None
    if not np.array_equal(repaired, clean):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid self-similar fractal repair input")
