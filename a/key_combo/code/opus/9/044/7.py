from __future__ import annotations

from collections import deque
import hashlib

import numpy as np


BLACK = 0
ORANGE = 7
TILE_SIZE = 3
OUTPUT_SIZE = 5
ALLOWED_COLORS = (1, 2, 3, 4)

_RNG = np.random.default_rng()
_SEEN_OUTPUTS: set[str] = set()
_GEN_CALL_COUNT = 0
_BRANCH_PROFILES = (
    {"horizontal": True, "tile_count": 3, "winner_pos": 1, "prefer_diag": True},
    {"horizontal": False, "tile_count": 4, "winner_pos": 2, "prefer_diag": False},
    {"horizontal": True, "tile_count": 4, "winner_pos": 0, "prefer_diag": False},
    {"horizontal": False, "tile_count": 3, "winner_pos": 1, "prefer_diag": True},
)


def _neighbors(shape: tuple[int, int], r: int, c: int, connectivity: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    deltas = ((-1, 0), (1, 0), (0, -1), (0, 1))
    if connectivity == 8:
        deltas = deltas + ((-1, -1), (-1, 1), (1, -1), (1, 1))
    h, w = shape
    for dr, dc in deltas:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _is_connected(mask: np.ndarray, connectivity: int) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    queue: deque[tuple[int, int]] = deque([start])
    while queue:
        r, c = queue.popleft()
        for rr, cc in _neighbors(mask.shape, r, c, connectivity):
            if mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                queue.append((rr, cc))
    return len(seen) == int(mask.sum())


def _strip_tiles(grid: np.ndarray) -> list[np.ndarray]:
    h, w = grid.shape
    horizontal = h == TILE_SIZE and w % TILE_SIZE == 0
    vertical = w == TILE_SIZE and h % TILE_SIZE == 0
    assert horizontal != vertical, "expected a horizontal or vertical 3x3 tile strip"
    tile_count = (w if horizontal else h) // TILE_SIZE
    assert 3 <= tile_count <= 4, "expected 3 or 4 tiles"
    axis = 1 if horizontal else 0
    return [
        grid[:, c : c + TILE_SIZE].copy() if axis == 1 else grid[r : r + TILE_SIZE, :].copy()
        for r, c in (
            [(0, i * TILE_SIZE) for i in range(tile_count)]
            if axis == 1
            else [(i * TILE_SIZE, 0) for i in range(tile_count)]
        )
    ]


def _tile_color(tile: np.ndarray) -> int:
    colors = sorted(set(map(int, tile.flat)) - {BLACK})
    assert len(colors) == 1, "each tile must contain exactly one foreground color"
    return colors[0]


def _completed_tile(tile: np.ndarray, color: int) -> np.ndarray:
    completed = np.full((TILE_SIZE, TILE_SIZE), ORANGE, dtype=int)
    completed[tile == color] = color
    return completed


def _render_output(tile: np.ndarray) -> np.ndarray:
    color = _tile_color(tile)
    out = np.full((OUTPUT_SIZE, OUTPUT_SIZE), color, dtype=int)
    out[1:4, 1:4] = _completed_tile(tile, color)
    return out


def _validate_input_semantics(grid: np.ndarray) -> tuple[list[np.ndarray], list[int], int]:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2 and max(grid.shape) <= 30
    tiles = _strip_tiles(np.asarray(grid, dtype=int))
    counts: list[int] = []
    colors: list[int] = []
    for tile in tiles:
        assert tile.shape == (TILE_SIZE, TILE_SIZE)
        color = _tile_color(tile)
        assert color in ALLOWED_COLORS, "tile colors must come from {1,2,3,4}"
        colors.append(color)
        mask = tile == color
        assert _is_connected(mask, connectivity=8), "each tile must be one 8-connected sprite"
        counts.append(int(mask.sum()))
    assert len(set(colors)) == len(colors), "tile colors must be distinct"
    assert len(set(counts)) == len(counts), "foreground counts must be pairwise distinct"
    winner = int(np.argmin(counts))
    assert counts[winner] >= 2, "winner must have at least two foreground cells"
    assert counts[winner] <= 7, "winner must leave at least two black gaps to fill"
    assert np.count_nonzero(tiles[winner] == BLACK) >= 2
    return tiles, counts, winner


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the unique least-filled sprite tile, complete its missing cells in orange, and wrap it in its original-color outline.

    Input structure:
    - The input is a horizontal `3 x 3N` strip or vertical `3N x 3` strip of 3x3 tiles on black background.
    - Each tile contains exactly one non-black sprite color chosen from blue, red, green, or yellow, and each tile uses a different color.
    - Inside every tile, that color forms one 8-connected sprite; the tiles have distinct foreground-pixel counts.

    Rule:
    - Split the strip into its 3x3 tiles and count the non-black pixels in each tile.
    - Choose the unique tile with the smallest foreground count, and record its original sprite color `C`.
    - Fill every black cell of that winning 3x3 tile with orange while keeping its original `C` cells unchanged.
    - Output a 5x5 grid whose outer one-cell border is color `C` and whose centered 3x3 interior is the completed two-color tile.

    Key insight:
    - The decisive feature is the unique minimum foreground area, not position or color; orange marks the missing part, while the original sprite color returns as the surrounding outline.
    """
    g = np.asarray(grid, dtype=int)
    tiles, counts, winner = _validate_input_semantics(g)
    assert counts.count(min(counts)) == 1
    return _render_output(tiles[winner])


def _hash_grid(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes() + str(grid.shape).encode()).hexdigest()


def _enumerate_masks() -> dict[int, dict[str, list[np.ndarray]]]:
    masks: dict[int, dict[str, list[np.ndarray]]] = {}
    for count in range(2, 9):
        all_masks: list[np.ndarray] = []
        diag_masks: list[np.ndarray] = []
        for bits in range(1 << 9):
            if bits.bit_count() != count:
                continue
            mask = np.zeros((TILE_SIZE, TILE_SIZE), dtype=bool)
            for idx in range(9):
                if (bits >> idx) & 1:
                    mask[idx // TILE_SIZE, idx % TILE_SIZE] = True
            if not _is_connected(mask, connectivity=8):
                continue
            all_masks.append(mask.copy())
            if not _is_connected(mask, connectivity=4):
                diag_masks.append(mask.copy())
        masks[count] = {"all": all_masks, "diag": diag_masks}
    return masks


_MASKS_BY_COUNT = _enumerate_masks()


def _profile_for_call() -> dict[str, object] | None:
    if _GEN_CALL_COUNT < len(_BRANCH_PROFILES):
        return dict(_BRANCH_PROFILES[_GEN_CALL_COUNT])
    return None


def _sample_counts(tile_count: int, prefer_diag: bool) -> list[int]:
    choices = list(range(2, 9))
    for _ in range(50):
        counts = sorted(int(v) for v in _RNG.choice(choices, size=tile_count, replace=False))
        if counts[0] <= 7 and (not prefer_diag or _MASKS_BY_COUNT[counts[0]]["diag"]):
            return counts
    raise RuntimeError("failed to sample distinct counts")


def _sample_mask(count: int, prefer_diag: bool) -> np.ndarray:
    pool = _MASKS_BY_COUNT[count]["diag"] if prefer_diag and _MASKS_BY_COUNT[count]["diag"] else _MASKS_BY_COUNT[count]["all"]
    idx = int(_RNG.integers(0, len(pool)))
    return pool[idx].copy()


def _make_tile(mask: np.ndarray, color: int) -> np.ndarray:
    tile = np.full((TILE_SIZE, TILE_SIZE), BLACK, dtype=int)
    tile[mask] = color
    return tile


def _arrange_tiles(
    horizontal: bool,
    tile_specs: list[tuple[int, np.ndarray]],
) -> np.ndarray:
    tiles = [_make_tile(mask, color) for color, mask in tile_specs]
    return np.concatenate(tiles, axis=1 if horizontal else 0)


def _semantic_counterexample_checks(grid: np.ndarray) -> np.ndarray:
    tiles, counts, winner = _validate_input_semantics(grid)
    assert winner != int(np.argmax(counts)), "winner and maximum tile must differ"
    assert _tile_color(tiles[winner]) != ORANGE
    out = main(grid)
    expected = _render_output(tiles[winner])
    assert np.array_equal(out, expected)
    center = out[1:4, 1:4]
    assert not np.any(center == BLACK), "completed center must have no black cells"
    assert ORANGE in set(map(int, center.flat)), "orange fill must be visible"
    border_mask = np.ones((OUTPUT_SIZE, OUTPUT_SIZE), dtype=bool)
    border_mask[1:4, 1:4] = False
    winner_color = _tile_color(tiles[winner])
    assert np.all(out[border_mask] == winner_color), "outline must use the winner's original color"
    return out


def _gen_attempt() -> np.ndarray | None:
    global _GEN_CALL_COUNT

    profile = _profile_for_call()
    horizontal = bool(profile["horizontal"]) if profile is not None else bool(_RNG.integers(0, 2))
    tile_count = int(profile["tile_count"]) if profile is not None else int(_RNG.integers(3, 5))
    prefer_diag = bool(profile["prefer_diag"]) if profile is not None else bool(_RNG.integers(0, 2))
    winner_pos = int(profile["winner_pos"]) if profile is not None else int(_RNG.integers(0, tile_count))

    counts = _sample_counts(tile_count, prefer_diag=prefer_diag)
    other_counts = counts[1:]
    _RNG.shuffle(other_counts)
    assigned_counts = other_counts[:winner_pos] + [counts[0]] + other_counts[winner_pos:]

    colors = list(map(int, _RNG.choice(np.array(ALLOWED_COLORS), size=tile_count, replace=False)))
    _RNG.shuffle(colors)

    tile_specs: list[tuple[int, np.ndarray]] = []
    for idx, (color, count) in enumerate(zip(colors, assigned_counts)):
        mask = _sample_mask(count, prefer_diag=prefer_diag and idx == winner_pos)
        tile_specs.append((color, mask))

    grid = _arrange_tiles(horizontal, tile_specs)
    out = _semantic_counterexample_checks(grid)
    output_sig = _hash_grid(out)
    if output_sig in _SEEN_OUTPUTS:
        return None
    _SEEN_OUTPUTS.add(output_sig)
    _GEN_CALL_COUNT += 1
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid non-degenerate input")
