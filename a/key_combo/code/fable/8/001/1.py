from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, in_bounds


DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
PALETTE = tuple(range(1, 10))


def _full_in_bounds_spiral(
    center: tuple[int, int], shape: tuple[int, int]
) -> list[tuple[int, int]]:
    coords = [center]
    r, c = center
    leg_len = 1
    dir_idx = 0
    while True:
        for _ in range(2):
            dr, dc = DIRS[dir_idx % 4]
            for _step in range(leg_len):
                r += dr
                c += dc
                if not in_bounds(shape, r, c):
                    return coords
                coords.append((r, c))
            dir_idx += 1
        leg_len += 1


def _spiral_prefix(
    center: tuple[int, int], shape: tuple[int, int], end_idx: int
) -> list[tuple[int, int]] | None:
    coords = [center]
    if end_idx == 0:
        return coords

    r, c = center
    leg_len = 1
    dir_idx = 0
    while len(coords) <= end_idx:
        for _ in range(2):
            dr, dc = DIRS[dir_idx % 4]
            for _step in range(leg_len):
                r += dr
                c += dc
                if not in_bounds(shape, r, c):
                    return None
                coords.append((r, c))
                if len(coords) > end_idx:
                    return coords
            dir_idx += 1
        leg_len += 1
    return coords


def _indices_for_targets(
    center: tuple[int, int],
    shape: tuple[int, int],
    targets: set[tuple[int, int]],
) -> dict[tuple[int, int], int] | None:
    if not in_bounds(shape, center[0], center[1]):
        return None

    found: dict[tuple[int, int], int] = {}
    remaining = set(targets)
    if center in remaining:
        found[center] = 0
        remaining.remove(center)
        if not remaining:
            return found

    r, c = center
    leg_len = 1
    dir_idx = 0
    step_idx = 0
    while remaining:
        for _ in range(2):
            dr, dc = DIRS[dir_idx % 4]
            for _step in range(leg_len):
                r += dr
                c += dc
                step_idx += 1
                if not in_bounds(shape, r, c):
                    return None
                if (r, c) in remaining:
                    found[(r, c)] = step_idx
                    remaining.remove((r, c))
                    if not remaining:
                        return found
            dir_idx += 1
        leg_len += 1
    return found


def _infer_period(sample_pairs: list[tuple[int, int]]) -> tuple[int, list[int]]:
    assert sample_pairs, "expected at least one colored sample"
    max_idx = max(idx for idx, _color in sample_pairs)
    sample_count = len(sample_pairs)
    for period in range(1, min(max_idx + 1, sample_count) + 1):
        residue_to_color: dict[int, int] = {}
        seen = set()
        ok = True
        for idx, color in sample_pairs:
            residue = idx % period
            seen.add(residue)
            prev = residue_to_color.get(residue)
            if prev is None:
                residue_to_color[residue] = color
            elif prev != color:
                ok = False
                break
        if ok and len(seen) == period:
            return period, [residue_to_color[r] for r in range(period)]
    raise AssertionError("no complete minimal period fits the sampled spiral colors")


def _find_candidates(
    grid: np.ndarray,
) -> list[tuple[tuple[int, int], dict[tuple[int, int], int], int, list[int]]]:
    samples = {
        (int(r), int(c)): int(grid[r, c])
        for r, c in np.argwhere(grid != Color.BLACK)
    }
    assert samples, "input must contain colored samples"

    h, w = grid.shape
    candidates = []
    target_coords = set(samples)
    for r in range(h):
        for c in range(w):
            index_map = _indices_for_targets((r, c), grid.shape, target_coords)
            if index_map is None:
                continue
            pairs = sorted((index_map[coord], color) for coord, color in samples.items())
            try:
                period, pattern = _infer_period(pairs)
            except AssertionError:
                continue
            candidates.append(((r, c), index_map, period, pattern))
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the hidden square spiral and repaint its full visible prefix with the inferred periodic color band.

    Input structure:
    - A black grid contains several isolated non-black sample pixels.
    - Every sample lies on one hidden square spiral that starts from an unmarked center cell
      and walks up, right, down, left with leg lengths 1, 1, 2, 2, 3, 3, ...
    - Reading colors along that spiral gives a repeating color sequence, and every position
      inside one period is sampled at least once somewhere among the visible pixels.

    Rule:
    - Find the unique center whose in-bounds spiral path contains all colored sample pixels.
    - Unroll the spiral from that center up to the farthest sampled pixel and infer the
      smallest period consistent with the sampled colors.
    - Repaint every cell on that spiral prefix by the recovered periodic sequence, keeping
      all other cells black.

    Key insight:
    - The puzzle combines two dependent reconstructions: first recover the hidden 2D spiral
      geometry, then collapse that path to 1D so periodic pattern repair becomes well-defined.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    candidates = _find_candidates(g)
    assert len(candidates) == 1, f"expected one valid spiral center, got {len(candidates)}"
    center, index_map, period, pattern = candidates[0]
    farthest = max(index_map.values())
    prefix = _spiral_prefix(center, g.shape, farthest)
    assert prefix is not None

    out = np.zeros_like(g)
    for idx, (r, c) in enumerate(prefix):
        out[r, c] = pattern[idx % period]

    for (r, c), idx in index_map.items():
        assert out[r, c] == g[r, c], "reconstructed spiral must preserve all visible samples"
    return out


def _primitive_period(seq: list[int]) -> int:
    for period in range(1, len(seq) + 1):
        if len(seq) % period != 0:
            continue
        if all(seq[i] == seq[i % period] for i in range(len(seq))):
            return period
    raise AssertionError("sequence must have a period")


def _choose_pattern(period: int) -> list[int]:
    for _ in range(200):
        color_count = random.randint(2, min(4, period))
        colors = random.sample(PALETTE, color_count)
        seq = [random.choice(colors) for _ in range(period)]
        if len(set(seq)) < 2:
            continue
        if _primitive_period(seq) == period:
            return seq
    raise RuntimeError("failed to sample a primitive color period")


def _too_close(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1


def _choose_sample_indices(
    prefix: list[tuple[int, int]], period: int, farthest: int
) -> list[int] | None:
    by_residue = {
        residue: [idx for idx in range(1, farthest + 1) if idx % period == residue]
        for residue in range(period)
    }
    if any(not indices for indices in by_residue.values()):
        return None

    extra_residues = random.sample(
        range(period),
        k=random.randint(max(1, period // 2), min(period, 3)),
    )
    need = {residue: 1 for residue in range(period)}
    for residue in extra_residues:
        need[residue] += 1

    requests: list[int | tuple[str, int]] = [farthest]
    need[farthest % period] -= 1
    residue_order = list(range(period))
    random.shuffle(residue_order)
    for residue in residue_order:
        for _ in range(need[residue]):
            requests.append(("residue", residue))

    def search(pos: int, chosen: list[int], used: set[int]) -> list[int] | None:
        if pos == len(requests):
            return sorted(chosen)

        request = requests[pos]
        if isinstance(request, int):
            idx_options = [request]
        else:
            residue = request[1]
            idx_options = by_residue[residue][:]
            random.shuffle(idx_options)
            idx_options.sort(key=lambda idx: (-idx // period, -idx))

        for idx in idx_options:
            if idx in used:
                continue
            coord = prefix[idx]
            if any(_too_close(coord, prefix[other]) for other in chosen):
                continue
            chosen.append(idx)
            used.add(idx)
            result = search(pos + 1, chosen, used)
            if result is not None:
                return result
            used.remove(idx)
            chosen.pop()
        return None

    chosen = search(0, [], set())
    if chosen is None:
        return None

    spare = [idx for idx in range(1, farthest) if idx not in chosen]
    random.shuffle(spare)
    bonus_target = random.randint(0, min(4, len(spare)))
    for idx in spare:
        if bonus_target == 0:
            break
        coord = prefix[idx]
        if any(_too_close(coord, prefix[other]) for other in chosen):
            continue
        chosen.append(idx)
        bonus_target -= 1
    return sorted(chosen)


def _render_output(
    shape: tuple[int, int],
    prefix: list[tuple[int, int]],
    pattern: list[int],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    period = len(pattern)
    for idx, (r, c) in enumerate(prefix):
        out[r, c] = pattern[idx % period]
    return out


def _render_input(
    shape: tuple[int, int],
    prefix: list[tuple[int, int]],
    pattern: list[int],
    sample_indices: list[int],
) -> np.ndarray:
    inp = np.zeros(shape, dtype=int)
    period = len(pattern)
    for idx in sample_indices:
        r, c = prefix[idx]
        inp[r, c] = pattern[idx % period]
    return inp


def _valid_instance(
    inp: np.ndarray,
    out: np.ndarray,
    period: int,
    pattern: list[int],
    sample_indices: list[int],
) -> bool:
    colored = [tuple(map(int, rc)) for rc in np.argwhere(inp != Color.BLACK)]
    if len(colored) != len(sample_indices):
        return False
    if any(_too_close(a, b) for i, a in enumerate(colored) for b in colored[i + 1 :]):
        return False
    if len(sample_indices) < period + 2:
        return False
    if len({idx % period for idx in sample_indices}) != period:
        return False
    if len(out.nonzero()[0]) - len(sample_indices) < period + 2:
        return False
    if _primitive_period(pattern) != period:
        return False

    candidates = _find_candidates(inp)
    if len(candidates) != 1:
        return False
    center, index_map, got_period, got_pattern = candidates[0]
    if inp[center] != Color.BLACK:
        return False
    if got_period != period or got_pattern != pattern:
        return False
    if sorted(index_map.values()) != sample_indices:
        return False
    try:
        solved = main(inp)
    except Exception:
        return False
    return np.array_equal(solved, out)


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(15, 30)
    w = random.randint(15, 30)
    shape = (h, w)

    for _ in range(80):
        center = (random.randrange(h), random.randrange(w))
        full_prefix = _full_in_bounds_spiral(center, shape)
        max_idx = len(full_prefix) - 1
        if max_idx < 18:
            continue

        max_period = min(6, max_idx // 3)
        if max_period < 2:
            continue
        period = random.randint(2, max_period)
        min_farthest = max(3 * period, period + 6)
        if max_idx < min_farthest:
            continue

        farthest = random.randint(min_farthest, max_idx)
        prefix = full_prefix[: farthest + 1]
        pattern = _choose_pattern(period)
        sample_indices = _choose_sample_indices(prefix, period, farthest)
        if sample_indices is None:
            continue
        residues = {idx % period for idx in sample_indices}
        if len(residues) != period:
            continue
        if max(sample_indices) != farthest:
            continue
        if sum(sample_indices.count(idx) for idx in sample_indices) != len(sample_indices):
            continue
        if len(sample_indices) >= farthest - 1:
            continue
        if len(sample_indices) < period + 2:
            continue

        inp = _render_input(shape, prefix, pattern, sample_indices)
        out = _render_output(shape, prefix, pattern)
        if not _valid_instance(inp, out, period, pattern, sample_indices):
            continue
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-spiral periodic-repair input")
