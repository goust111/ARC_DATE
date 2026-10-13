import numpy as np


_RNG = np.random.default_rng()
_PALETTE = np.arange(1, 10, dtype=int)
_GEN_ATTEMPTS = 400


def _split_tiles(grid: np.ndarray) -> np.ndarray:
    """Return the input as a (3, 3, tile_h, tile_w) tile array."""
    h, w = grid.shape
    assert h % 3 == 0 and w % 3 == 0, "input must be a 3x3 tiling"
    tile_h, tile_w = h // 3, w // 3
    return grid.reshape(3, tile_h, 3, tile_w).swapaxes(1, 2).copy()


def _join_tiles(tiles: np.ndarray) -> np.ndarray:
    """Inverse of _split_tiles for a (3, 3, tile_h, tile_w) tile array."""
    return tiles.swapaxes(1, 2).reshape(
        tiles.shape[0] * tiles.shape[2],
        tiles.shape[1] * tiles.shape[3],
    )


def _majority_color(values: np.ndarray) -> int:
    counts = np.bincount(values.astype(int), minlength=max(10, int(values.max()) + 1))
    winners = np.flatnonzero(counts == counts.max())
    assert len(winners) == 1, "tile votes must have a unique majority"
    return int(winners[0])


def _reconstruct_prototype(tiles: np.ndarray) -> np.ndarray:
    flat_tiles = tiles.reshape(9, tiles.shape[2], tiles.shape[3])
    tile_h, tile_w = flat_tiles.shape[1:]
    prototype = np.zeros((tile_h, tile_w), dtype=int)
    for r in range(tile_h):
        for c in range(tile_w):
            prototype[r, c] = _majority_color(flat_tiles[:, r, c])
    return prototype


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the hidden prototype by majority vote across nine tiles and summarize which tiles contain a single corruption.

    Input structure
    - The whole grid is exactly a 3x3 tiling of same-sized rectangular tiles.
    - All nine tiles are copies of one unseen prototype pattern.
    - Each tile is either perfect or has exactly one pixel changed to a non-background error color.

    Rule
    - Split the grid into its nine equal tiles.
    - For every within-tile coordinate, take the majority color across the nine tiles to recover the prototype.
    - Compare each tile against that recovered prototype and emit a 3x3 summary grid:
      intact tiles become 0, while a damaged tile becomes the error color of its unique mismatching pixel.

    Key insight
    - The answer lives at tile level, not pixel level: majority vote reconstructs a prototype that is never shown directly, then each tile's one-pixel deviation collapses to one color in a 3x3 error map.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    tiles = _split_tiles(g)
    prototype = _reconstruct_prototype(tiles)

    out = np.zeros((3, 3), dtype=int)
    for tile_r in range(3):
        for tile_c in range(3):
            tile = tiles[tile_r, tile_c]
            diff = np.argwhere(tile != prototype)
            if diff.size == 0:
                continue
            assert diff.shape[0] == 1, "each tile may differ from the prototype in at most one cell"
            r, c = map(int, diff[0])
            error_color = int(tile[r, c])
            assert error_color != 0, "corrupted pixels must be non-background"
            out[tile_r, tile_c] = error_color
    return out


def _draw_rect(canvas: np.ndarray, color: int) -> None:
    h, w = canvas.shape
    rect_h = int(_RNG.integers(1, min(h, max(2, (2 * h) // 3)) + 1))
    rect_w = int(_RNG.integers(1, min(w, max(2, (2 * w) // 3)) + 1))
    r0 = int(_RNG.integers(0, h - rect_h + 1))
    c0 = int(_RNG.integers(0, w - rect_w + 1))
    canvas[r0 : r0 + rect_h, c0 : c0 + rect_w] = color


def _draw_frame(canvas: np.ndarray, color: int) -> None:
    h, w = canvas.shape
    if h < 3 or w < 3:
        _draw_rect(canvas, color)
        return
    frame_h = int(_RNG.integers(3, h + 1))
    frame_w = int(_RNG.integers(3, w + 1))
    r0 = int(_RNG.integers(0, h - frame_h + 1))
    c0 = int(_RNG.integers(0, w - frame_w + 1))
    r1, c1 = r0 + frame_h - 1, c0 + frame_w - 1
    canvas[r0, c0 : c1 + 1] = color
    canvas[r1, c0 : c1 + 1] = color
    canvas[r0 : r1 + 1, c0] = color
    canvas[r0 : r1 + 1, c1] = color


def _draw_line(canvas: np.ndarray, color: int) -> None:
    h, w = canvas.shape
    if _RNG.random() < 0.5:
        r = int(_RNG.integers(0, h))
        c0 = int(_RNG.integers(0, w))
        c1 = int(_RNG.integers(c0 + 1, w + 1))
        canvas[r, c0:c1] = color
    else:
        c = int(_RNG.integers(0, w))
        r0 = int(_RNG.integers(0, h))
        r1 = int(_RNG.integers(r0 + 1, h + 1))
        canvas[r0:r1, c] = color


def _draw_l_shape(canvas: np.ndarray, color: int) -> None:
    h, w = canvas.shape
    corner_r = int(_RNG.integers(0, h))
    corner_c = int(_RNG.integers(0, w))
    down = int(_RNG.integers(1, h - corner_r + 1))
    right = int(_RNG.integers(1, w - corner_c + 1))
    canvas[corner_r : corner_r + down, corner_c] = color
    canvas[corner_r, corner_c : corner_c + right] = color


def _draw_cross(canvas: np.ndarray, color: int) -> None:
    h, w = canvas.shape
    center_r = int(_RNG.integers(0, h))
    center_c = int(_RNG.integers(0, w))
    arm = int(_RNG.integers(0, min(h, w) // 2 + 1))
    r0 = max(0, center_r - arm)
    r1 = min(h - 1, center_r + arm)
    c0 = max(0, center_c - arm)
    c1 = min(w - 1, center_c + arm)
    canvas[r0 : r1 + 1, center_c] = color
    canvas[center_r, c0 : c1 + 1] = color


def _generate_prototype() -> np.ndarray:
    drawers = (_draw_rect, _draw_frame, _draw_line, _draw_l_shape, _draw_cross)
    for _ in range(_GEN_ATTEMPTS):
        tile_h = int(_RNG.integers(4, 11))
        tile_w = int(_RNG.integers(4, 11))
        prototype = np.zeros((tile_h, tile_w), dtype=int)

        color_count = int(_RNG.integers(2, 5))
        palette = _RNG.choice(_PALETTE, size=color_count, replace=False).astype(int).tolist()
        op_count = int(_RNG.integers(color_count + 1, color_count + 4))
        op_colors = palette + _RNG.choice(palette, size=op_count - color_count, replace=True).astype(int).tolist()
        _RNG.shuffle(op_colors)

        for color in op_colors:
            drawer = drawers[int(_RNG.integers(0, len(drawers)))]
            drawer(prototype, int(color))

        fg = int(np.count_nonzero(prototype))
        colors = set(map(int, np.unique(prototype))) - {0}
        if not colors or len(colors) < 2:
            continue
        if not np.any(prototype == 0):
            continue
        if not (0.18 * prototype.size <= fg <= 0.72 * prototype.size):
            continue
        return prototype
    raise RuntimeError("failed to generate a prototype with enough structure")


def _damage_plan(prototype: np.ndarray, damaged_tile_count: int) -> list[tuple[tuple[int, int], int]]:
    bg_coords = [(int(r), int(c)) for r, c in np.argwhere(prototype == 0)]
    fg_coords = [(int(r), int(c)) for r, c in np.argwhere(prototype != 0)]
    _RNG.shuffle(bg_coords)
    _RNG.shuffle(fg_coords)

    kinds: list[str] = []
    if damaged_tile_count >= 2 and bg_coords and fg_coords:
        if _RNG.random() < 0.8:
            kinds.extend(["bg", "fg"])
    while len(kinds) < damaged_tile_count:
        choices = []
        if bg_coords and kinds.count("bg") < len(bg_coords):
            choices.append("bg")
        if fg_coords and kinds.count("fg") < len(fg_coords):
            choices.append("fg")
        assert choices, "not enough prototype cells to place damage"
        kinds.append(str(_RNG.choice(choices)))
    _RNG.shuffle(kinds)

    plan = []
    for kind in kinds:
        coord = bg_coords.pop() if kind == "bg" else fg_coords.pop()
        original = int(prototype[coord])
        error_choices = [color for color in _PALETTE.tolist() if color != original]
        error_color = int(_RNG.choice(error_choices))
        plan.append((coord, error_color))
    return plan


def _make_example() -> tuple[np.ndarray, np.ndarray]:
    prototype = _generate_prototype()
    tiles = np.repeat(prototype[None, None, :, :], 3, axis=0)
    tiles = np.repeat(tiles, 3, axis=1)

    tile_slots = [(r, c) for r in range(3) for c in range(3)]
    _RNG.shuffle(tile_slots)
    damaged_tile_count = int(_RNG.integers(2, 8))
    damaged_tiles = tile_slots[:damaged_tile_count]
    damage_plan = _damage_plan(prototype, damaged_tile_count)

    expected = np.zeros((3, 3), dtype=int)
    for (tile_r, tile_c), ((r, c), error_color) in zip(damaged_tiles, damage_plan):
        tiles[tile_r, tile_c, r, c] = error_color
        expected[tile_r, tile_c] = error_color

    inp = _join_tiles(tiles)
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        inp, expected = _make_example()
        if max(inp.shape) > 30:
            continue
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid tiled-majority ARC input")
