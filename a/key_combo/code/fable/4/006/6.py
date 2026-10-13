from __future__ import annotations

import math
import random

import numpy as np


GREY = 5
PAINT_COLORS = (0, 1, 2, 3, 4, 6, 7, 8, 9)
K_CHOICES = (3, 4, 5)
GEN_RETRIES = 500


def _infer_k(grid: np.ndarray) -> int:
    assert grid.ndim == 2, "expected a 2D grid"
    h, w = map(int, grid.shape)
    assert h == w, "expected a square grid"
    k = int(math.isqrt(h))
    assert k >= 2 and k * k == h, "expected k by k blocks, each k by k pixels"
    return k


def _block(grid: np.ndarray, k: int, br: int, bc: int) -> np.ndarray:
    return grid[br * k : (br + 1) * k, bc * k : (bc + 1) * k]


def _is_pure(block: np.ndarray) -> bool:
    return bool(np.all(block == block[0, 0]))


def _pure_color(block: np.ndarray) -> int:
    assert _is_pure(block), "expected a solid-color block"
    return int(block[0, 0])


def _analyze(grid: np.ndarray) -> tuple[int, tuple[int, int], np.ndarray, tuple[int, int], int]:
    k = _infer_k(grid)

    thumbnail_positions: list[tuple[int, int]] = []
    for br in range(k):
        for bc in range(k):
            if not _is_pure(_block(grid, k, br, bc)):
                thumbnail_positions.append((br, bc))
    assert len(thumbnail_positions) == 1, "expected exactly one non-solid thumbnail block"

    map_pos = thumbnail_positions[0]
    mr, mc = map_pos
    thumbnail = _block(grid, k, mr, mc).copy()
    grey_cells = np.argwhere(thumbnail == GREY)
    assert grey_cells.shape == (1, 2), "expected exactly one grey self-reference cell"
    assert tuple(map(int, grey_cells[0])) == map_pos, "grey self-reference must name the thumbnail block"

    mismatches: list[tuple[int, int, int]] = []
    for br in range(k):
        for bc in range(k):
            if (br, bc) == map_pos:
                continue
            claimed = int(thumbnail[br, bc])
            assert claimed != GREY, "grey is reserved for the thumbnail's self-reference cell"
            actual = _pure_color(_block(grid, k, br, bc))
            if actual != claimed:
                mismatches.append((br, bc, claimed))

    assert len(mismatches) == 1, "expected exactly one block to disagree with the thumbnail"
    er, ec, target_color = mismatches[0]
    return k, map_pos, thumbnail, (er, ec), int(target_color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Correct the one solid block that disagrees with its self-describing thumbnail.

    Input structure:
    - The square grid is exactly partitioned into k by k large blocks, each block also
      measuring k by k pixels.
    - All large blocks are solid-color except one multicolor k by k block, which is a
      one-pixel-per-block thumbnail of the whole block layout.
    - The thumbnail's cell at its own block coordinate is grey and is the only grey
      self-reference marker.

    Rule:
    - Locate the unique multicolor thumbnail block and verify that its grey cell names
      that block's coordinate.
    - Compare every non-grey thumbnail cell with the color of the corresponding large
      solid block.
    - Exactly one large block disagrees; repaint that entire large block to the color
      claimed by the thumbnail.
    - Leave every other block, including the thumbnail block itself, unchanged.

    Key insight:
    - The small multicolor block is an authoritative map of the big block layout, so
      the fix changes the physical large block rather than editing the map cell.
    """
    g = np.asarray(grid, dtype=int)
    k, _map_pos, _thumbnail, error_pos, target_color = _analyze(g)
    er, ec = error_pos

    out = g.copy()
    out[er * k : (er + 1) * k, ec * k : (ec + 1) * k] = int(target_color)
    return out


def _positions(k: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(k) for c in range(k)]


def _sample_claims(k: int, map_pos: tuple[int, int]) -> np.ndarray:
    non_map = [pos for pos in _positions(k) if pos != map_pos]
    palette_size = random.randint(3, min(5, len(non_map)))
    palette = random.sample(list(PAINT_COLORS), palette_size)

    values = palette[:]
    while len(values) < len(non_map):
        values.append(random.choice(palette))
    random.shuffle(values)

    claims = np.empty((k, k), dtype=int)
    claims[map_pos] = GREY
    for pos, color in zip(non_map, values):
        claims[pos] = int(color)
    return claims


def _choose_error(claims: np.ndarray, map_pos: tuple[int, int]) -> tuple[tuple[int, int], int]:
    k = int(claims.shape[0])
    non_map = [pos for pos in _positions(k) if pos != map_pos]
    counts: dict[int, int] = {}
    for pos in non_map:
        color = int(claims[pos])
        counts[color] = counts.get(color, 0) + 1

    target_positions = [pos for pos in non_map if counts[int(claims[pos])] >= 2]
    if not target_positions:
        raise ValueError("need a repeated claimed color for the wrong block")

    error_pos = random.choice(target_positions)
    claimed_color = int(claims[error_pos])
    wrong_choices = [color for color, count in counts.items() if color != claimed_color and count >= 1]
    if not wrong_choices:
        raise ValueError("need a visible alternate color for the wrong block")
    return error_pos, int(random.choice(wrong_choices))


def _build_grid(k: int, map_pos: tuple[int, int], claims: np.ndarray, error_pos: tuple[int, int], wrong_color: int) -> np.ndarray:
    n = k * k
    grid = np.empty((n, n), dtype=int)
    for br in range(k):
        for bc in range(k):
            if (br, bc) == map_pos:
                grid[br * k : (br + 1) * k, bc * k : (bc + 1) * k] = claims
            else:
                color = int(wrong_color) if (br, bc) == error_pos else int(claims[br, bc])
                grid[br * k : (br + 1) * k, bc * k : (bc + 1) * k] = color
    return grid


def _expected_from_plan(grid: np.ndarray, k: int, error_pos: tuple[int, int], target_color: int) -> np.ndarray:
    out = grid.copy()
    er, ec = error_pos
    out[er * k : (er + 1) * k, ec * k : (ec + 1) * k] = int(target_color)
    return out


def _changed_blocks(before: np.ndarray, after: np.ndarray, k: int) -> list[tuple[int, int]]:
    changed = []
    for br in range(k):
        for bc in range(k):
            if not np.array_equal(_block(before, k, br, bc), _block(after, k, br, bc)):
                changed.append((br, bc))
    return changed


def _is_non_degenerate(claims: np.ndarray, map_pos: tuple[int, int], error_pos: tuple[int, int], wrong_color: int) -> bool:
    non_map = [pos for pos in _positions(int(claims.shape[0])) if pos != map_pos]
    claimed_color = int(claims[error_pos])
    other_claimed = sum(1 for pos in non_map if pos != error_pos and int(claims[pos]) == claimed_color)
    other_wrong = sum(1 for pos in non_map if pos != error_pos and int(claims[pos]) == int(wrong_color))
    varied_thumbnail = len({int(claims[pos]) for pos in non_map}) >= 3
    return bool(other_claimed >= 1 and other_wrong >= 1 and varied_thumbnail)


def _gen_attempt() -> np.ndarray | None:
    k = random.choice(K_CHOICES)
    map_pos = random.choice(_positions(k))
    claims = _sample_claims(k, map_pos)

    try:
        error_pos, wrong_color = _choose_error(claims, map_pos)
    except ValueError:
        return None

    if not _is_non_degenerate(claims, map_pos, error_pos, wrong_color):
        return None

    grid = _build_grid(k, map_pos, claims, error_pos, wrong_color)
    expected = _expected_from_plan(grid, k, error_pos, int(claims[error_pos]))

    try:
        found_k, found_map, thumbnail, found_error, found_color = _analyze(grid)
        solved = main(grid)
    except Exception:
        return None

    if found_k != k or found_map != map_pos or found_error != error_pos:
        return None
    if found_color != int(claims[error_pos]):
        return None
    if not np.array_equal(thumbnail, claims):
        return None
    if not np.array_equal(solved, expected):
        return None
    if _changed_blocks(grid, solved, k) != [error_pos]:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid thumbnail self-correction input")
