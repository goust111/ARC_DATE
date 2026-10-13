from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import Color
except ModuleNotFoundError:
    helper_dir = Path("/home/lixinyu/snap/new/solvers")
    if str(helper_dir) not in sys.path:
        sys.path.append(str(helper_dir))
    from arc_utils import Color


def _tile_fill(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    tile_h, tile_w = map(int, tile.shape)
    rows = np.arange(int(shape[0]))[:, None] % tile_h
    cols = np.arange(int(shape[1]))[None, :] % tile_w
    return tile[rows, cols]


def _minimal_tiles(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    known = g != Color.BLACK
    coords = np.argwhere(known)
    if coords.size == 0:
        return []

    height, width = map(int, g.shape)
    candidates: list[np.ndarray] = []
    for tile_h in range(1, height + 1):
        for tile_w in range(1, width + 1):
            tile = np.full((tile_h, tile_w), -1, dtype=int)
            valid = True
            for r, c in coords:
                tr = int(r) % tile_h
                tc = int(c) % tile_w
                value = int(g[r, c])
                seen = int(tile[tr, tc])
                if seen == -1:
                    tile[tr, tc] = value
                elif seen != value:
                    valid = False
                    break
            if valid and np.all(tile != -1):
                candidates.append(tile)

    if not candidates:
        return []

    best_area = min(int(tile.shape[0] * tile.shape[1]) for tile in candidates)
    return [tile for tile in candidates if int(tile.shape[0] * tile.shape[1]) == best_area]


def _infer_unique_minimal_tile(grid: np.ndarray) -> np.ndarray:
    candidates = _minimal_tiles(grid)
    assert candidates, "expected at least one fully observed repeating tile candidate"
    assert len(candidates) == 1, "expected a unique minimal repeating tile"
    return candidates[0]


def _separator_candidates(grid: np.ndarray) -> list[int]:
    values, counts = np.unique(grid, return_counts=True)
    count_map = {int(v): int(c) for v, c in zip(values, counts)}

    candidates = []
    width = int(grid.shape[1])
    for r in range(int(grid.shape[0])):
        value = int(grid[r, 0])
        if value == Color.BLACK:
            continue
        if np.all(grid[r] == value) and count_map.get(value, 0) == width:
            candidates.append(r)
    return candidates


def _find_separator_row(grid: np.ndarray) -> int:
    candidates = _separator_candidates(grid)
    assert len(candidates) == 1, "expected a unique separator row color that appears only on that row"
    sep = int(candidates[0])
    assert 0 < sep < grid.shape[0] - 1, "separator must split the grid into nonempty top and bottom regions"
    return sep


def _restore_visible_fragments(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    restored = np.asarray(top, dtype=int).copy()
    bottom = np.asarray(bottom, dtype=int)
    width = int(restored.shape[1])

    for c in range(width):
        hole_rows = np.where(restored[:, c] == Color.BLACK)[0]
        fragments = bottom[:, c][bottom[:, c] != Color.BLACK]
        if fragments.size == 0:
            continue
        assert hole_rows.size >= fragments.size, "fragment stack cannot exceed the number of holes in its column"
        restored[hole_rows[-fragments.size :], c] = fragments
    return restored


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reverse each column's fallen scraps into the wallpaper holes, then use the restored periodic tile to finish the remaining gaps.

    Input structure:
    - A unique full-row separator color splits the grid into an upper wallpaper zone and a lower fragment zone.
    - The upper zone is a periodic non-black wallpaper with some cells replaced by black holes.
    - The lower zone is black except for same-column fragment stacks resting on the bottom edge.

    Rule:
    - Identify the separator row and process each column independently.
    - Refill bottommost visible fragments into that column's holes from bottom to top; in a full stack column, any higher unmatched holes stay black.
    - Infer the unique smallest repeating tile from the restored non-black wallpaper cells and use it to fill every remaining hole.
    - Keep the separator row unchanged and clear the entire lower zone to black.

    Key insight:
    - The fragment pile is not noise: it is the gravitational consequence of the missing wallpaper pixels, and only after reversing that fall does the wallpaper's minimal period become uniquely determined.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep = _find_separator_row(g)
    top = g[:sep, :]
    bottom = g[sep + 1 :, :]
    restored = _restore_visible_fragments(top, bottom)
    tile = _infer_unique_minimal_tile(restored)

    full_top = _tile_fill(tile, restored.shape)
    known = restored != Color.BLACK
    assert np.all(full_top[known] == restored[known]), "inferred tile must preserve every restored non-black pixel"

    out = np.full_like(g, Color.BLACK)
    out[:sep, :] = full_top
    out[sep, :] = g[sep, :]
    return out


# --------------- generator (o3) ---------------

_RNG = np.random.default_rng()
_GEN_RETRIES = 600
_TILE_SHAPES = ((2, 3), (2, 4), (3, 2), (3, 3), (3, 4), (4, 2), (4, 3))
_PALETTE = tuple(int(c) for c in Color.NOT_BLACK)


def _choice(seq):
    return seq[int(_RNG.integers(0, len(seq)))]


def _full_period(tile: np.ndarray) -> bool:
    candidates = _minimal_tiles(tile)
    return len(candidates) == 1 and candidates[0].shape == tile.shape and np.array_equal(candidates[0], tile)


def _generate_tile() -> tuple[np.ndarray, int, int, int, int] | None:
    tile_h, tile_w = _choice(_TILE_SHAPES)
    hidden_r = int(_RNG.integers(tile_h))
    hidden_c = int(_RNG.integers(tile_w))
    mix_r = _choice([r for r in range(tile_h) if r != hidden_r])

    color_count = int(_RNG.integers(4, 6))
    colors = [int(c) for c in _RNG.choice(np.array(_PALETTE), size=color_count, replace=False)]
    hidden_color = int(colors[0])
    fill_palette = np.array(colors[1:], dtype=int)

    tile = _RNG.choice(fill_palette, size=(tile_h, tile_w)).astype(int)
    tile[hidden_r, hidden_c] = hidden_color
    tile[mix_r, hidden_c] = int(_choice([int(c) for c in fill_palette]))

    if np.count_nonzero(tile == hidden_color) != 1:
        return None
    if int(tile[mix_r, hidden_c]) == hidden_color:
        return None
    if len(np.unique(tile)) < 3:
        return None
    if not _full_period(tile):
        return None
    return tile, hidden_r, hidden_c, mix_r, hidden_color


def _phase_rows(height: int, residue: int, period: int) -> list[int]:
    return [r for r in range(int(height)) if r % int(period) == int(residue)]


def _phase_cols(width: int, residue: int, period: int) -> list[int]:
    return [c for c in range(int(width)) if c % int(period) == int(residue)]


def _visible_stack_bottom_to_top(top: np.ndarray, holes: np.ndarray, col: int, bottom_h: int) -> list[int]:
    hole_rows = np.where(holes[:, col])[0]
    colors = [int(top[r, col]) for r in hole_rows[::-1]]
    return colors[:bottom_h]


def _choose_small_rows(top: np.ndarray, holes: np.ndarray, col: int, count: int) -> list[int] | None:
    rows = [r for r in range(top.shape[0]) if not holes[r, col]]
    if len(rows) < count:
        return None
    if count == 1:
        return [int(_choice(rows))]

    scored: list[tuple[tuple[int, int], list[int]]] = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            pair = [int(rows[i]), int(rows[j])]
            color_score = len({int(top[r, col]) for r in pair})
            spread = abs(int(pair[1]) - int(pair[0]))
            scored.append(((color_score, spread), pair))
    if not scored:
        return None

    best_score = max(score for score, _ in scored)
    best_pairs = [pair for score, pair in scored if score == best_score]
    return best_pairs[int(_RNG.integers(0, len(best_pairs)))]


def _build_hole_mask(
    top: np.ndarray,
    hidden_r: int,
    hidden_c: int,
    mix_r: int,
    bottom_h: int,
) -> tuple[np.ndarray, int, int] | None:
    tile_h, tile_w = map(int, _infer_unique_minimal_tile(top).shape)
    height, width = map(int, top.shape)

    holes = np.zeros((height, width), dtype=bool)
    hidden_rows = _phase_rows(height, hidden_r, tile_h)
    hidden_cols = _phase_cols(width, hidden_c, tile_w)
    for c in hidden_cols:
        holes[hidden_rows, c] = True

    mix_col = int(_choice(hidden_cols))
    mix_rows = _phase_rows(height, mix_r, tile_h)
    success = False
    for r in reversed(mix_rows):
        if holes[r, mix_col]:
            continue
        holes[r, mix_col] = True
        visible = _visible_stack_bottom_to_top(top, holes, mix_col, bottom_h)
        if len(visible) >= 2 and len(set(visible)) >= 2:
            success = True
            break
    if not success:
        return None

    small_candidates = [c for c in range(width) if c != mix_col and c % tile_w != hidden_c]
    if not small_candidates:
        return None
    small_col = int(_choice(small_candidates))

    max_small = min(bottom_h - 1, 2)
    small_count = 1 if max_small == 1 else int(_RNG.integers(1, max_small + 1))
    small_rows = _choose_small_rows(top, holes, small_col, small_count)
    if small_rows is None:
        return None
    for r in small_rows:
        holes[int(r), small_col] = True

    return holes, mix_col, small_col


def _simulate_fall(top: np.ndarray, holes: np.ndarray, separator_color: int, bottom_h: int) -> np.ndarray:
    height, width = map(int, top.shape)
    inp_top = top.copy()
    inp_top[holes] = Color.BLACK

    bottom = np.full((bottom_h, width), Color.BLACK, dtype=int)
    for c in range(width):
        hole_rows = np.where(holes[:, c])[0]
        visible = [int(top[r, c]) for r in hole_rows[::-1][:bottom_h]]
        if visible:
            bottom[bottom_h - len(visible) :, c] = visible[::-1]

    separator = np.full((1, width), int(separator_color), dtype=int)
    return np.vstack([inp_top, separator, bottom])


def _wrong_topdown_restore(top: np.ndarray, bottom: np.ndarray, mix_col: int) -> np.ndarray:
    restored = _restore_visible_fragments(top, bottom)
    hole_rows = np.where(top[:, mix_col] == Color.BLACK)[0]
    fragments = bottom[:, mix_col][bottom[:, mix_col] != Color.BLACK]
    if fragments.size == 0:
        return restored

    count = int(fragments.size)
    restored[hole_rows[-count:], mix_col] = Color.BLACK
    restored[hole_rows[:count], mix_col] = fragments
    return restored


def _attempt_case() -> np.ndarray | None:
    tile_data = _generate_tile()
    if tile_data is None:
        return None
    tile, hidden_r, hidden_c, mix_r, hidden_color = tile_data
    tile_h, tile_w = map(int, tile.shape)

    top_h = int(_RNG.integers(max(8, tile_h * 3), min(16, tile_h * 5) + 1))
    top_w = int(_RNG.integers(max(8, tile_w * 3), min(16, tile_w * 5) + 1))
    full_top = _tile_fill(tile, (top_h, top_w))

    hidden_rows = _phase_rows(top_h, hidden_r, tile_h)
    if len(hidden_rows) < 3:
        return None
    bottom_h = int(_RNG.integers(2, min(4, len(hidden_rows) - 1) + 1))

    holes_data = _build_hole_mask(full_top, hidden_r, hidden_c, mix_r, bottom_h)
    if holes_data is None:
        return None
    holes, mix_col, small_col = holes_data

    separator_choices = [color for color in _PALETTE if color not in set(map(int, np.unique(tile)))]
    if not separator_choices:
        return None
    separator_color = int(_choice(separator_choices))

    inp = _simulate_fall(full_top, holes, separator_color, bottom_h)
    expected = np.vstack(
        [
            full_top,
            np.full((1, top_w), separator_color, dtype=int),
            np.full((bottom_h, top_w), Color.BLACK, dtype=int),
        ]
    )

    if inp.shape[0] > 30 or inp.shape[1] > 30:
        return None
    if expected.shape[0] > 30 or expected.shape[1] > 30:
        return None
    if _separator_candidates(inp) != [top_h]:
        return None

    top = inp[:top_h, :]
    bottom = inp[top_h + 1 :, :]
    restored = _restore_visible_fragments(top, bottom)

    if hidden_color in set(map(int, np.unique(top[top != Color.BLACK]))):
        return None
    if hidden_color not in set(map(int, np.unique(bottom[bottom != Color.BLACK]))):
        return None

    hole_counts = [int(np.count_nonzero(holes[:, c])) for c in range(top_w)]
    if not any(count > bottom_h for count in hole_counts):
        return None
    if not any(0 < count < bottom_h for count in hole_counts):
        return None
    if not np.any(restored == Color.BLACK):
        return None

    before_candidates = _minimal_tiles(top)
    if len(before_candidates) == 1:
        return None

    after_candidates = _minimal_tiles(restored)
    if len(after_candidates) != 1:
        return None
    if after_candidates[0].shape != tile.shape or not np.array_equal(after_candidates[0], tile):
        return None

    if np.count_nonzero(bottom[:, mix_col] != Color.BLACK) != bottom_h:
        return None
    mix_visible = bottom[:, mix_col][bottom[:, mix_col] != Color.BLACK]
    if len(set(map(int, mix_visible))) < 2:
        return None

    wrong = _wrong_topdown_restore(top, bottom, mix_col)
    wrong_candidates = _minimal_tiles(wrong)
    if len(wrong_candidates) == 1 and wrong_candidates[0].shape == tile.shape and np.array_equal(wrong_candidates[0], tile):
        return None

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _attempt_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid falling-wallpaper input")
