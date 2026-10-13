from __future__ import annotations

import math
import random

import numpy as np


BLACK = 0
PALETTE = tuple(range(1, 10))
MAX_GEN_ATTEMPTS = 600


def _block_lattice_shape(grid: np.ndarray) -> tuple[int, int]:
    h, w = map(int, grid.shape)
    n = math.isqrt(h)
    m = math.isqrt(w)
    assert n * n == h and m * m == w, "grid must be an n^2 by m^2 block mosaic"
    return n, m


def _split_blocks(grid: np.ndarray, n: int, m: int) -> np.ndarray:
    return grid.reshape(n, n, m, m).transpose(0, 2, 1, 3)


def _parse_block(block: np.ndarray) -> tuple[int, np.ndarray] | None:
    colors = [int(color) for color in np.unique(block) if int(color) != BLACK]
    if len(colors) != 1:
        return None
    color = colors[0]
    mask = block == color
    if not bool(mask.any()):
        return None
    return color, mask


def _render_block_from_mask(mask: np.ndarray, color: int) -> np.ndarray:
    block = np.zeros(mask.shape, dtype=int)
    block[mask] = int(color)
    return block


def _canonical_blocks(seed: np.ndarray) -> dict[int, np.ndarray]:
    blocks: dict[int, np.ndarray] = {}
    for color in np.unique(seed):
        color = int(color)
        blocks[color] = _render_block_from_mask(seed == color, color)
    return blocks


def _render_mosaic(seed: np.ndarray) -> np.ndarray:
    seed = np.asarray(seed, dtype=int)
    n, m = map(int, seed.shape)
    out = np.zeros((n * n, m * m), dtype=int)
    blocks = _canonical_blocks(seed)
    for r in range(n):
        for c in range(m):
            out[r * n : (r + 1) * n, c * m : (c + 1) * m] = blocks[int(seed[r, c])]
    return out


def _candidate_repair(
    blocks: np.ndarray,
    parsed: list[list[tuple[int, np.ndarray] | None]],
    bad_rc: tuple[int, int],
) -> dict[str, object] | None:
    n, m = map(int, blocks.shape[:2])
    bad_r, bad_c = map(int, bad_rc)
    known = np.ones((n, m), dtype=bool)
    known[bad_r, bad_c] = False

    observed = np.full((n, m), -1, dtype=int)
    masks_by_color: dict[int, np.ndarray] = {}

    for r in range(n):
        for c in range(m):
            if (r, c) == (bad_r, bad_c):
                continue
            sig = parsed[r][c]
            if sig is None:
                return None
            color, mask = sig
            observed[r, c] = int(color)
            prev = masks_by_color.get(int(color))
            if prev is None:
                masks_by_color[int(color)] = mask
            elif not np.array_equal(prev, mask):
                return None

    if not masks_by_color:
        return None

    covering_colors: list[int] = []
    for color, mask in masks_by_color.items():
        if not np.array_equal(mask[known], (observed == color)[known]):
            return None
        if bool(mask[bad_r, bad_c]):
            covering_colors.append(int(color))

    if len(covering_colors) != 1:
        return None

    true_color = int(covering_colors[0])
    seed = observed.copy()
    seed[bad_r, bad_c] = true_color
    expected_block = _render_block_from_mask(masks_by_color[true_color], true_color)
    if np.array_equal(blocks[bad_r, bad_c], expected_block):
        return None

    repaired = _render_mosaic(seed)
    return {
        "bad_rc": (bad_r, bad_c),
        "seed": seed,
        "true_color": true_color,
        "expected_block": expected_block,
        "output": repaired,
    }


def _find_repairs(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    n, m = _block_lattice_shape(g)
    blocks = _split_blocks(g, n, m)
    parsed = [[_parse_block(blocks[r, c]) for c in range(m)] for r in range(n)]

    repairs: list[dict[str, object]] = []
    for r in range(n):
        for c in range(m):
            if (candidate := _candidate_repair(blocks, parsed, (r, c))) is not None:
                repairs.append(candidate)
    return repairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the unique corrupted block in a self-referential block mosaic by reconstructing the hidden seed color map.

    Input structure:
    - The whole grid has shape (n*n, m*m), split into an n-by-m array of blocks.
    - Each block itself also has shape (n, m) and normally uses black plus one non-black color.
    - In an intact block at block position (r, c), the non-black color is the hidden seed color at (r, c), and its painted cells are exactly all seed positions having that same color.

    Rule:
    - Assume exactly one block has been replaced by a pattern that breaks the self-reference rule.
    - Test each block position as the possible bad one; for the remaining blocks, require that every color reports one consistent global mask and that each mask matches the set of same-colored block positions outside the hypothesized bad position.
    - The unique hypothesis that makes all other blocks mutually consistent reveals the hidden seed color at the bad position; redraw that block with its color's global mask and return the fully repaired mosaic.

    Key insight:
    - Every intact block names the seed twice at once: its color gives one seed cell value, and its painted mask gives the full occurrence set of that color in the seed, so all intact blocks except one determine the missing block uniquely.
    """
    g = np.asarray(grid, dtype=int)
    repairs = _find_repairs(g)
    assert len(repairs) == 1, "expected exactly one bad block reconstruction"
    return np.asarray(repairs[0]["output"], dtype=int)


def _sample_seed_shape() -> tuple[int, int]:
    n = random.randint(3, 5)
    m = random.randint(3, 5)
    return n, m


def _sample_counts(total: int) -> list[int]:
    max_colors = min(5, total // 2)
    assert max_colors >= 2
    while True:
        n_colors = random.randint(2, max_colors)
        extras = np.random.multinomial(total - 2 * n_colors, [1.0 / n_colors] * n_colors)
        counts = [2 + int(extra) for extra in extras]
        if max(counts) >= total - 1:
            continue
        if max(counts) > int(math.ceil(total * 0.6)):
            continue
        if sum(count > 2 for count in counts) == 0:
            continue
        return counts


def _sample_seed() -> np.ndarray:
    n, m = _sample_seed_shape()
    total = n * m
    counts = _sample_counts(total)
    colors = random.sample(PALETTE, len(counts))
    flat = []
    for color, count in zip(colors, counts):
        flat.extend([int(color)] * int(count))
    random.shuffle(flat)
    seed = np.array(flat, dtype=int).reshape(n, m)
    assert seed.shape == (n, m)
    return seed


def _random_mask(shape: tuple[int, int], forbid: set[bytes]) -> np.ndarray:
    n, m = map(int, shape)
    total = n * m
    while True:
        count = random.randint(1, total - 1)
        idx = random.sample(range(total), count)
        mask = np.zeros((n, m), dtype=bool)
        mask.reshape(-1)[idx] = True
        if mask.tobytes() not in forbid:
            return mask


def _corrupt_block(
    seed: np.ndarray,
    canonical: dict[int, np.ndarray],
    bad_rc: tuple[int, int],
) -> np.ndarray:
    true_color = int(seed[bad_rc])
    colors = sorted(int(color) for color in canonical)
    other_colors = [color for color in colors if color != true_color]
    masks_by_color = {color: canonical[color] == color for color in colors}
    forbid_masks = {mask.tobytes() for mask in masks_by_color.values()}

    style = random.choices(
        population=[
            "wrong_pair",
            "wrong_mask_same_color",
            "wrong_color_same_mask",
            "random_mono",
            "multi_color_noise",
        ],
        weights=[3, 3, 2, 2, 2],
        k=1,
    )[0]

    if style == "wrong_pair" and other_colors:
        borrowed = random.choice(other_colors)
        return canonical[borrowed].copy()

    if style == "wrong_mask_same_color" and other_colors:
        borrowed = random.choice(other_colors)
        return _render_block_from_mask(masks_by_color[borrowed], true_color)

    if style == "wrong_color_same_mask":
        paint_color = random.choice(other_colors) if other_colors and random.random() < 0.7 else random.choice(PALETTE)
        if int(paint_color) == true_color:
            paint_color = random.choice(other_colors or [color for color in PALETTE if color != true_color])
        return _render_block_from_mask(masks_by_color[true_color], int(paint_color))

    if style == "random_mono":
        paint_color = random.choice(PALETTE)
        mask = _random_mask(seed.shape, forbid_masks)
        if int(paint_color) == true_color and np.array_equal(mask, masks_by_color[true_color]):
            paint_color = random.choice([color for color in PALETTE if color != true_color])
        return _render_block_from_mask(mask, int(paint_color))

    paint_colors = random.sample(PALETTE, 2)
    block = np.zeros(seed.shape, dtype=int)
    total = seed.size
    cells = random.sample(range(total), random.randint(2, max(2, total - 1)))
    split = random.randint(1, len(cells) - 1)
    first = cells[:split]
    second = cells[split:]
    flat = block.reshape(-1)
    for idx in first:
        flat[idx] = int(paint_colors[0])
    for idx in second:
        flat[idx] = int(paint_colors[1])
    return block


def _validate_instance(inp: np.ndarray, seed: np.ndarray, bad_rc: tuple[int, int]) -> bool:
    n, m = map(int, seed.shape)
    expected = _render_mosaic(seed)
    repairs = _find_repairs(inp)
    if len(repairs) != 1:
        return False

    repair = repairs[0]
    if tuple(repair["bad_rc"]) != tuple(bad_rc):
        return False
    if not np.array_equal(repair["seed"], seed):
        return False
    if not np.array_equal(repair["output"], expected):
        return False
    if not np.array_equal(main(inp), expected):
        return False

    blocks_in = _split_blocks(inp, n, m)
    blocks_expected = _split_blocks(expected, n, m)
    diffs = []
    for r in range(n):
        for c in range(m):
            same = np.array_equal(blocks_in[r, c], blocks_expected[r, c])
            if (r, c) == bad_rc:
                if same:
                    return False
                diffs.append(int(np.count_nonzero(blocks_in[r, c] != blocks_expected[r, c])))
            elif not same:
                return False

    if len(diffs) != 1 or diffs[0] < 2:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    seed = _sample_seed()
    canonical = _canonical_blocks(seed)
    clean = _render_mosaic(seed)
    n, m = map(int, seed.shape)
    bad_rc = (random.randrange(n), random.randrange(m))
    corrupted = _corrupt_block(seed, canonical, bad_rc)

    true_block = canonical[int(seed[bad_rc])]
    if np.array_equal(corrupted, true_block):
        return None

    inp = clean.copy()
    br, bc = bad_rc
    inp[br * n : (br + 1) * n, bc * m : (bc + 1) * m] = corrupted
    if not _validate_instance(inp, seed, bad_rc):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        if (grid := _gen_attempt()) is not None:
            return grid
    raise RuntimeError("failed to generate a valid self-referential mosaic repair input")
