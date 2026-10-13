from __future__ import annotations

import random

import numpy as np


BG = 0
STENCIL = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
SAMPLE_SIZE = 3
STENCIL_SPAN = 4
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_sample_stencil(grid: np.ndarray) -> tuple[int, int]:
    candidates: list[tuple[int, int]] = []
    h, w = grid.shape
    for r in range(h - SAMPLE_SIZE):
        for c in range(w - SAMPLE_SIZE):
            if np.all(grid[r : r + SAMPLE_SIZE, c + SAMPLE_SIZE] == STENCIL) and np.all(
                grid[r + SAMPLE_SIZE, c : c + STENCIL_SPAN] == STENCIL
            ):
                candidates.append((r, c))
    assert len(candidates) == 1, "expected exactly one grey L stencil"
    return candidates[0]


def _halo_is_bg(grid: np.ndarray, r0: int, c0: int, h: int, w: int, bg: int) -> bool:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(grid.shape[0], r0 + h + 1)
    cc1 = min(grid.shape[1], c0 + w + 1)
    halo = grid[rr0:rr1, cc0:cc1].copy()
    halo[r0 - rr0 : r0 - rr0 + h, c0 - cc0 : c0 - cc0 + w] = bg
    return bool(np.all(halo == bg))


def _find_isolated_blocks(grid: np.ndarray, bg: int, source: tuple[int, int]) -> list[tuple[int, int, np.ndarray]]:
    blocks: list[tuple[int, int, np.ndarray]] = []
    h, w = grid.shape
    for r in range(h - SAMPLE_SIZE + 1):
        for c in range(w - SAMPLE_SIZE + 1):
            if (r, c) == source:
                continue
            patch = grid[r : r + SAMPLE_SIZE, c : c + SAMPLE_SIZE]
            if np.any((patch == bg) | (patch == STENCIL)):
                continue
            if _halo_is_bg(grid, r, c, SAMPLE_SIZE, SAMPLE_SIZE, bg):
                blocks.append((r, c, patch.copy()))
    return blocks


def _scene_oracle(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[tuple[int, int], int]], int, int]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    source = _find_sample_stencil(g)
    sr, sc = source
    sample = g[sr : sr + SAMPLE_SIZE, sc : sc + SAMPLE_SIZE].copy()
    assert np.all((sample != bg) & (sample != STENCIL)), "sample must be a full non-background 3x3 patch"

    one_offs: list[tuple[tuple[int, int], int]] = []
    exact_count = 0
    distractor_count = 0
    used_positions: set[tuple[int, int]] = set()
    for _r, _c, patch in _find_isolated_blocks(g, bg, source):
        diff = np.argwhere(patch != sample)
        if len(diff) == 0:
            exact_count += 1
            continue
        if len(diff) == 1:
            dr, dc = map(int, diff[0])
            assert (dr, dc) not in used_positions, "one-off difference positions must be unique"
            used_positions.add((dr, dc))
            one_offs.append(((dr, dc), int(patch[dr, dc])))
            continue
        distractor_count += 1

    out = np.zeros((SAMPLE_SIZE, SAMPLE_SIZE), dtype=int)
    for (dr, dc), color in one_offs:
        out[dr, dc] = int(color)
    assert one_offs, "expected at least one one-off copy"
    return out, one_offs, exact_count, distractor_count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the grey-L-marked 3x3 sample and aggregate the error cells from all one-off copies onto a 3x3 canvas.

    Input structure:
    - One 3x3 sample patch is marked by grey pixels immediately on its right
      edge and below it, forming an L stencil.
    - Elsewhere, several isolated 3x3 color patches appear on the black
      background; each is either an exact copy of the sample, a copy with
      exactly one changed cell, or a distractor with multiple differences.
    - The changed-cell positions among the one-off copies are all distinct in
      sample-relative coordinates.

    Rule:
    - Use the grey L to read the unique 3x3 sample.
    - Inspect every other isolated 3x3 patch and keep only those whose Hamming
      distance from the sample is exactly one; ignore exact copies.
    - Produce a 3x3 black output and, for each one-off copy, write the wrong
      color into the differing relative cell position.

    Key insight:
    - The answer never appears as a whole patch in the input; it is encoded by
      collecting the locations and colors of the single-cell errors scattered
      across near-matching copies.
    """
    out, _one_offs, _exact_count, _distractor_count = _scene_oracle(grid)
    return out


def _expand8(mask: np.ndarray) -> np.ndarray:
    expanded = mask.copy()
    expanded[1:, :] |= mask[:-1, :]
    expanded[:-1, :] |= mask[1:, :]
    expanded[:, 1:] |= mask[:, :-1]
    expanded[:, :-1] |= mask[:, 1:]
    expanded[1:, 1:] |= mask[:-1, :-1]
    expanded[:-1, :-1] |= mask[1:, 1:]
    expanded[1:, :-1] |= mask[:-1, 1:]
    expanded[:-1, 1:] |= mask[1:, :-1]
    return expanded


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _sample_cells(sr: int, sc: int) -> list[tuple[int, int]]:
    return [(sr + r, sc + c) for r in range(SAMPLE_SIZE) for c in range(SAMPLE_SIZE)]


def _stencil_cells(sr: int, sc: int) -> list[tuple[int, int]]:
    cells = [(sr + r, sc + SAMPLE_SIZE) for r in range(SAMPLE_SIZE)]
    cells.extend((sr + SAMPLE_SIZE, sc + c) for c in range(STENCIL_SPAN))
    return cells


def _random_sample() -> np.ndarray:
    while True:
        distinct = random.randint(3, 6)
        colors = random.sample(PALETTE, distinct)
        flat = colors.copy()
        flat.extend(random.choice(colors) for _ in range(SAMPLE_SIZE * SAMPLE_SIZE - distinct))
        random.shuffle(flat)
        sample = np.array(flat, dtype=int).reshape(SAMPLE_SIZE, SAMPLE_SIZE)
        if len(set(map(int, sample.ravel()))) >= 3:
            return sample


def _mutate_patch(sample: np.ndarray, positions: list[tuple[int, int]]) -> np.ndarray:
    patch = sample.copy()
    for r, c in positions:
        choices = [color for color in PALETTE if color != int(sample[r, c])]
        patch[r, c] = random.choice(choices)
    return patch


def _make_distractor(sample: np.ndarray) -> np.ndarray:
    diff_count = random.choices([2, 3, 4, 5], weights=[5, 3, 2, 1])[0]
    positions = random.sample([(r, c) for r in range(SAMPLE_SIZE) for c in range(SAMPLE_SIZE)], diff_count)
    return _mutate_patch(sample, positions)


def _can_place_patch(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    blocked = _expand8(occupied)
    return not bool(blocked[r0 : r0 + h, c0 : c0 + w].any())


def _place_patch(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    grid[r0 : r0 + h, c0 : c0 + w] = patch
    occupied[r0 : r0 + h, c0 : c0 + w] = True


def _all_patch_positions(height: int, width: int) -> list[tuple[int, int]]:
    positions = [(r, c) for r in range(height - SAMPLE_SIZE + 1) for c in range(width - SAMPLE_SIZE + 1)]
    random.shuffle(positions)
    return positions


def _place_source(grid: np.ndarray, occupied: np.ndarray, sample: np.ndarray) -> tuple[int, int] | None:
    height, width = grid.shape
    positions = [(r, c) for r in range(height - STENCIL_SPAN + 1) for c in range(width - STENCIL_SPAN + 1)]
    random.shuffle(positions)
    for sr, sc in positions:
        cells = _sample_cells(sr, sc) + _stencil_cells(sr, sc)
        trial = np.zeros_like(occupied)
        for r, c in cells:
            trial[r, c] = True
        if _expand8(occupied)[trial].any():
            continue
        _place_patch(grid, occupied, sample, sr, sc)
        _paint_cells(grid, _stencil_cells(sr, sc), STENCIL)
        occupied[np.array([r for r, _c in _stencil_cells(sr, sc)]), np.array([c for _r, c in _stencil_cells(sr, sc)])] = True
        return sr, sc
    return None


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    height = random.randint(16, 28)
    width = random.randint(16, 28)
    grid = np.full((height, width), BG, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    sample = _random_sample()
    source = _place_source(grid, occupied, sample)
    if source is None:
        raise RuntimeError("failed to place sample source")

    one_off_count = random.randint(2, 6)
    positions = random.sample([(r, c) for r in range(SAMPLE_SIZE) for c in range(SAMPLE_SIZE)], one_off_count)
    expected = np.zeros((SAMPLE_SIZE, SAMPLE_SIZE), dtype=int)
    payloads: list[np.ndarray] = []
    for pos in positions:
        patch = _mutate_patch(sample, [pos])
        expected[pos] = int(patch[pos])
        payloads.append(patch)

    if random.random() < 0.6:
        payloads.append(sample.copy())

    for _ in range(random.randint(1, 3)):
        payloads.append(_make_distractor(sample))

    random.shuffle(payloads)
    for patch in payloads:
        for r0, c0 in _all_patch_positions(height, width):
            if _can_place_patch(occupied, r0, c0, SAMPLE_SIZE, SAMPLE_SIZE):
                _place_patch(grid, occupied, patch, r0, c0)
                break
        else:
            raise RuntimeError("failed to place all 3x3 patches")

    oracle_out, one_offs, exact_count, distractor_count = _scene_oracle(grid)
    assert np.array_equal(oracle_out, expected), "semantic oracle mismatch"
    assert len(one_offs) == one_off_count, "wrong number of one-off copies"
    assert exact_count in (0, 1), "expected at most one exact copy decoy"
    assert distractor_count >= 1, "generator should include distractor coverage"
    assert np.array_equal(main(grid), expected), "main must match the generated target"
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        try:
            grid, _expected = _generate_case()
        except (AssertionError, RuntimeError):
            continue
        if grid.ndim == 2 and max(grid.shape) <= 30:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
