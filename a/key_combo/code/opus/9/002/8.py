import importlib.util
from pathlib import Path

import numpy as np


def _load_arc_utils():
    try:
        from arc_utils import Color, scale_sprite

        return Color, scale_sprite
    except ModuleNotFoundError:
        util_path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
        spec = importlib.util.spec_from_file_location("arc_utils_fallback", util_path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)
        return module.Color, module.scale_sprite


Color, scale_sprite = _load_arc_utils()

BLACK = Color.BLACK
BLUE = Color.BLUE
RED = Color.RED
GREY = Color.GREY

GRID_SIDE_CHOICES = {
    6: (3, 2),
    8: (4, 2),
    9: (3, 3),
    10: (5, 2),
    12: ((3, 4), (4, 3)),
    15: (5, 3),
    16: (4, 4),
    20: (5, 4),
}
GEN_RETRIES = 400


def _recover_candidates(grid: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    side = int(grid.shape[0])
    candidates = []
    for k in range(2, 5):
        if side % k != 0:
            continue
        n = side // k
        if n < 3 or n * n > 30:
            continue

        seed = np.zeros((n, n), dtype=int)
        valid = True
        for br in range(n):
            for bc in range(n):
                block = grid[br * k : (br + 1) * k, bc * k : (bc + 1) * k]
                grey_count = int(np.count_nonzero(block == GREY))
                if grey_count > k - 1:
                    valid = False
                    break
                visible = block[block != GREY]
                if visible.size == 0:
                    valid = False
                    break
                colors = set(map(int, np.unique(visible)))
                if not colors <= {BLACK, BLUE} or len(colors) != 1:
                    valid = False
                    break
                seed[br, bc] = int(visible[0])
            if not valid:
                break
        if not valid:
            continue

        blue_count = int(np.count_nonzero(seed == BLUE))
        if not (3 <= blue_count <= n * n - 3):
            continue
        candidates.append((n, k, seed))
    return candidates


def _infer_seed_and_scale(grid: np.ndarray) -> tuple[np.ndarray, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1], "expected a square 2D grid"
    assert set(map(int, np.unique(g))) <= {BLACK, BLUE, GREY}, "unexpected input colors"

    candidates = _recover_candidates(g)
    assert len(candidates) == 1, "expected a unique block scale and seed"
    _n, k, seed = candidates[0]
    return seed, int(k)


def _stamp(seed: np.ndarray) -> np.ndarray:
    n = int(seed.shape[0])
    out = np.full((n * n, n * n), BLACK, dtype=int)
    block_index = 0
    for br in range(n):
        for bc in range(n):
            if int(seed[br, bc]) != BLUE:
                continue
            patch = seed.copy()
            if block_index % 3 == 0:
                patch[patch == BLUE] = RED
            out[br * n : (br + 1) * n, bc * n : (bc + 1) * n] = patch
            block_index += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover a noisy scaled seed, stamp that seed into its own foreground blocks, and recolor every third stamp red.

    Input structure:
    - The input is a square black-background grid whose side length is `N*K`.
    - It is partitioned into an implicit `N x N` lattice of `K x K` tiles, where
      each tile is blue or black except for a few grey noise pixels.
    - Ignoring grey, each tile is monochrome, and those tile colors encode a
      hidden binary seed pattern.

    Rule:
    - Infer the unique tile size `K` in `2..4` and the corresponding `N x N`
      seed by collapsing each `K x K` tile after ignoring grey noise.
    - Create an `(N*N) x (N*N)` black output split into `N x N` blocks of size
      `N x N`.
    - For every blue cell of the recovered seed, copy the whole seed into the
      matching output block; black seed cells leave their output blocks empty.
    - Number the copied blocks in row-major order over blue seed cells only, and
      recolor all blue pixels inside blocks numbered `0, 3, 6, ...` to red.

    Key insight:
    - The noisy enlargement is only an intermediate encoding: denoising reveals a
      seed that simultaneously specifies where copies go and what each copy
      contains, while the final red/blue distinction depends on the stamp order.
    """
    seed, _k = _infer_seed_and_scale(np.asarray(grid, dtype=int))
    return _stamp(seed)


def _has_axis_changes(seed: np.ndarray) -> bool:
    return bool(np.any(seed[:, 1:] != seed[:, :-1])) and bool(np.any(seed[1:, :] != seed[:-1, :]))


def _seed_is_diverse(seed: np.ndarray) -> bool:
    blue_count = int(np.count_nonzero(seed == BLUE))
    n = int(seed.shape[0])
    if not (4 <= blue_count <= n * n - 4):
        return False
    if len(np.unique(seed.sum(axis=0))) == 1 and len(np.unique(seed.sum(axis=1))) == 1:
        return False
    if np.count_nonzero(seed.sum(axis=0)) < 2 or np.count_nonzero(seed.sum(axis=1)) < 2:
        return False
    return _has_axis_changes(seed)


def _render_noisy_input(seed: np.ndarray, k: int) -> np.ndarray:
    clean = scale_sprite(seed, k).astype(int)
    grid = clean.copy()
    n = int(seed.shape[0])
    blue_noisy = 0
    black_noisy = 0

    for br in range(n):
        for bc in range(n):
            max_noise = k - 1
            if np.random.rand() < 0.7:
                noise_count = int(np.random.randint(1, max_noise + 1))
            else:
                noise_count = 0
            if noise_count == 0:
                continue

            cells = np.random.choice(k * k, size=noise_count, replace=False)
            rr = br * k + cells // k
            cc = bc * k + cells % k
            grid[rr, cc] = GREY
            if int(seed[br, bc]) == BLUE:
                blue_noisy += 1
            else:
                black_noisy += 1

    if blue_noisy == 0 or black_noisy == 0:
        return np.empty((0, 0), dtype=int)
    return grid


def _pick_pair() -> tuple[int, int]:
    side = int(np.random.choice(sorted(GRID_SIDE_CHOICES)))
    options = GRID_SIDE_CHOICES[side]
    if isinstance(options[0], tuple):
        return options[int(np.random.randint(len(options)))]
    return options


def _sample_seed(n: int) -> np.ndarray:
    for _ in range(200):
        density = float(np.random.uniform(0.3, 0.7))
        seed = (np.random.rand(n, n) < density).astype(int)
        if _seed_is_diverse(seed):
            return seed
    raise RuntimeError("failed to sample a diverse seed")


def _validate_instance(grid: np.ndarray, seed: np.ndarray, k: int) -> bool:
    if grid.shape[0] == 0:
        return False
    if max(grid.shape) > 30:
        return False

    candidates = _recover_candidates(grid)
    if len(candidates) != 1:
        return False
    n2, k2, recovered = candidates[0]
    if k2 != k or n2 != seed.shape[0] or not np.array_equal(recovered, seed):
        return False

    out = main(grid)
    expected = _stamp(seed)
    if not np.array_equal(out, expected):
        return False
    if set(map(int, np.unique(out))) != {BLACK, BLUE, RED}:
        return False

    plain = np.full_like(out, BLACK)
    n = int(seed.shape[0])
    for br in range(n):
        for bc in range(n):
            if int(seed[br, bc]) == BLUE:
                plain[br * n : (br + 1) * n, bc * n : (bc + 1) * n] = seed
    if np.array_equal(out, plain):
        return False

    blue_positions = np.argwhere(seed == BLUE)
    if blue_positions.shape[0] < 4:
        return False
    red_blocks = sum(1 for idx in range(int(blue_positions.shape[0])) if idx % 3 == 0)
    blue_blocks = int(blue_positions.shape[0]) - red_blocks
    if red_blocks == 0 or blue_blocks == 0:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        n, k = _pick_pair()
        seed = _sample_seed(n)
        grid = _render_noisy_input(seed, k)
        if _validate_instance(grid, seed, k):
            return grid
    raise RuntimeError("failed to generate a valid noisy seed-stamp puzzle input")
