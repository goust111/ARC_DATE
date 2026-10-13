from __future__ import annotations

import numpy as np


BG = 0
BOUNDARY = 5
BAR_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    starts = np.argwhere(mask)
    for r0, c0 in starts:
        r0 = int(r0)
        c0 = int(c0)
        if seen[r0, c0]:
            continue
        stack = [(r0, c0)]
        seen[r0, c0] = True
        coords: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            coords.append((r, c))
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        parts.append(np.array(coords, dtype=int))
    return parts


def _bar_candidate(grid: np.ndarray, coords: np.ndarray, color: int) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h != 1 or w < 3 or int(coords.shape[0]) != w:
        return None
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    if not np.all(patch == color):
        return None
    cells = {(int(r), int(c)) for r, c in coords}
    gh, gw = grid.shape
    for r, c in cells:
        for nr, nc in _neighbors4(r, c):
            if not (0 <= nr < gh and 0 <= nc < gw):
                continue
            if (nr, nc) in cells:
                continue
            if int(grid[nr, nc]) != BG:
                return None
    return {"color": int(color), "row": int(r0), "c0": int(c0), "c1": int(c1), "length": int(w)}


def _frame_candidate(grid: np.ndarray, coords: np.ndarray, color: int) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h != w or h < 4:
        return None
    if int(coords.shape[0]) != 4 * h - 4:
        return None
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    perimeter = np.zeros_like(patch, dtype=bool)
    perimeter[0, :] = True
    perimeter[-1, :] = True
    perimeter[:, 0] = True
    perimeter[:, -1] = True
    inner = patch[1:-1, 1:-1]
    if inner.size == 0:
        return None
    if not np.all(patch[perimeter] == color):
        return None
    if np.any(inner == BG) or np.any(inner == color):
        return None
    if len(np.unique(inner)) < 2:
        return None
    return {
        "color": int(color),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "tile": inner.copy(),
        "top": int(r0),
        "left": int(c0),
    }


def _extract_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert not np.any(g == BOUNDARY), "grey is reserved for the output boundary"
    bars: list[dict] = []
    tiles: dict[int, dict] = {}

    for color in sorted(int(x) for x in np.unique(g) if int(x) != BG):
        comps = _components(g == color)
        bar_hits: list[dict] = []
        frame_hits: list[dict] = []
        for coords in comps:
            bar = _bar_candidate(g, coords, color)
            if bar is not None:
                bar_hits.append(bar)
                continue
            frame = _frame_candidate(g, coords, color)
            if frame is not None:
                frame_hits.append(frame)
        if bar_hits or frame_hits:
            assert len(bar_hits) == 1, f"expected exactly one bar for color {color}"
            assert len(frame_hits) == 1, f"expected exactly one framed tile for color {color}"
            bars.append(bar_hits[0])
            tiles[int(color)] = frame_hits[0]

    assert 2 <= len(bars) <= 3, "expected 2-3 colored bars"
    bar_colors = [int(bar["color"]) for bar in bars]
    bar_lengths = [int(bar["length"]) for bar in bars]
    assert len(set(bar_colors)) == len(bar_colors), "bar colors must be distinct"
    assert len(set(bar_lengths)) == len(bar_lengths), "bar lengths must be distinct"
    assert set(bar_colors) == set(tiles), "every bar color must match one framed tile"

    bars.sort(key=lambda item: (int(item["row"]), int(item["c0"])))
    return {"grid": g, "bars": bars, "tiles": tiles}


def _diamond_mask(shape: tuple[int, int], row: int, c0: int, c1: int) -> np.ndarray:
    h, w = shape
    length = int(c1 - c0 + 1)
    radius = (length - 1) // 2
    mask = np.zeros((h, w), dtype=bool)
    for offset in range(radius + 1):
        left = int(c0 + offset)
        right = int(c1 - offset)
        if left > right:
            continue
        for rr in (row - offset, row + offset):
            if 0 <= rr < h:
                mask[rr, left : right + 1] = True
    return mask


def _erode4(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, constant_values=False)
    return (
        mask
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )


def _paint_tiled_region(out: np.ndarray, mask: np.ndarray, tile: np.ndarray) -> None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return
    r0, c0 = coords.min(axis=0)
    th, tw = tile.shape
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        out[rr, cc] = int(tile[(rr - int(r0)) % th, (cc - int(c0)) % tw])


def _apply_bar(out: np.ndarray, bar: dict, tile: np.ndarray) -> np.ndarray:
    mask = _diamond_mask(out.shape, int(bar["row"]), int(bar["c0"]), int(bar["c1"]))
    interior = _erode4(mask)
    boundary = mask & ~interior
    out[boundary] = BOUNDARY
    _paint_tiled_region(out, interior, tile)
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each horizontal bar by a grey-rimmed diamond filled with its matching tile pattern.

    Input structure:
    - The background is black.
    - There are 2-3 solid one-pixel-thick horizontal bars, each with a distinct color and a distinct length.
    - For every bar color there is one small square tile enclosed by a one-pixel border of the same color.

    Rule:
    - For each bar, find the framed tile whose border color matches the bar color, then remove the border and keep only the tile content.
    - Use the bar length as the diamond's center-row width; each row above or below shrinks by one cell on both ends until the width reaches one or zero.
    - Paint the outermost diamond layer grey and fill the strict interior by repeating the matched tile from the interior bbox's top-left corner.

    Key insight:
    - Each bar carries two independent pieces of information at once: its color selects which tile to repeat, and its length selects where that repeated pattern is allowed to appear.
    """
    scene = _extract_scene(grid)
    out = np.array(scene["grid"], copy=True)
    for bar in scene["bars"]:
        tile = scene["tiles"][int(bar["color"])]["tile"]
        _apply_bar(out, bar, tile)
    return out


def _has_nontrivial_period(tile: np.ndarray) -> bool:
    h, w = tile.shape
    for dr in range(h):
        for dc in range(w):
            if dr == 0 and dc == 0:
                continue
            rolled = np.roll(np.roll(tile, dr, axis=0), dc, axis=1)
            if np.array_equal(tile, rolled):
                return True
    return False


def _make_tile(size: int, forbidden: set[int], rng: np.random.Generator) -> np.ndarray:
    palette = [c for c in BAR_COLORS if c not in forbidden]
    assert len(palette) >= 3
    for _ in range(200):
        use_n = int(rng.integers(2, min(4, len(palette)) + 1))
        chosen = [int(x) for x in rng.choice(palette, size=use_n, replace=False)]
        tile = rng.choice(chosen, size=(size, size), replace=True).astype(int)
        if len(np.unique(tile)) < 2:
            continue
        if np.array_equal(tile, np.rot90(tile, 1)) or np.array_equal(tile, np.rot90(tile, 2)):
            continue
        if _has_nontrivial_period(tile):
            continue
        return tile
    raise RuntimeError("failed to build a non-degenerate tile")


def _corner_anchor(corner: str, outer: int, h: int, w: int) -> tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, w - outer
    if corner == "bl":
        return h - outer, 0
    if corner == "br":
        return h - outer, w - outer
    raise ValueError(f"unknown corner {corner}")


def _render_input(shape: tuple[int, int], bars: list[dict], tiles: dict[int, dict]) -> np.ndarray:
    h, w = shape
    grid = np.full((h, w), BG, dtype=int)
    for color, tile_spec in tiles.items():
        tile = tile_spec["tile"]
        outer = int(tile.shape[0] + 2)
        top = int(tile_spec["top"])
        left = int(tile_spec["left"])
        grid[top : top + outer, left : left + outer] = int(color)
        grid[top + 1 : top + outer - 1, left + 1 : left + outer - 1] = tile
    for bar in bars:
        row = int(bar["row"])
        c0 = int(bar["c0"])
        c1 = int(bar["c1"])
        grid[row, c0 : c1 + 1] = int(bar["color"])
    return grid


def _render_output(shape: tuple[int, int], bars: list[dict], tiles: dict[int, dict]) -> np.ndarray:
    out = _render_input(shape, bars, tiles)
    for bar in sorted(bars, key=lambda item: (int(item["row"]), int(item["c0"]))):
        _apply_bar(out, bar, tiles[int(bar["color"])]["tile"])
    return out


def _choose_lengths(count: int, rng: np.random.Generator) -> list[int]:
    if count == 3:
        lengths = [5, 7, 9]
    else:
        options = ([5, 9], [7, 9], [5, 11], [7, 11])
        lengths = list(options[int(rng.integers(0, len(options)))])
    rng.shuffle(lengths)
    return [int(x) for x in lengths]


def _scene_attempt(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    count = 3 if float(rng.random()) < 0.6 else 2
    lengths = _choose_lengths(count, rng)
    max_len = max(lengths)
    max_radius = (max_len - 1) // 2

    tile_sizes = [int(rng.integers(2, 4)) for _ in range(count)]
    outer_sizes = [size + 2 for size in tile_sizes]
    corners = ["tl", "tr", "bl", "br"]
    rng.shuffle(corners)
    corner_choices = corners[:count]

    left_outer = max((outer_sizes[i] for i, corner in enumerate(corner_choices) if corner in {"tl", "bl"}), default=0)
    right_outer = max((outer_sizes[i] for i, corner in enumerate(corner_choices) if corner in {"tr", "br"}), default=0)
    min_w = max(21, left_outer + max_len + right_outer + 3)
    if min_w > 30:
        return None

    sep = max_len
    min_h = max(21, count * sep)
    if min_h > 30:
        return None

    h = int(rng.integers(min_h, 31))
    w = int(rng.integers(min_w, 31))
    top_shift = int(rng.integers(0, h - (count * sep) + 1))
    rows = [int(max_radius + top_shift + idx * sep) for idx in range(count)]

    colors = [int(x) for x in rng.choice(BAR_COLORS, size=count, replace=False)]
    bars: list[dict] = []
    tiles: dict[int, dict] = {}

    col_min = left_outer + 1
    for idx, (color, length, row, size, corner) in enumerate(zip(colors, lengths, rows, tile_sizes, corner_choices)):
        outer = size + 2
        top, left = _corner_anchor(corner, outer, h, w)
        tiles[int(color)] = {
            "color": int(color),
            "top": int(top),
            "left": int(left),
            "corner": corner,
            "tile": _make_tile(size, {BG, BOUNDARY, *colors}, rng),
        }
        col_max = w - right_outer - length - 1
        if col_min > col_max:
            return None
        c0 = int(rng.integers(col_min, col_max + 1))
        bars.append({"color": int(color), "row": int(row), "c0": int(c0), "c1": int(c0 + length - 1), "length": int(length)})

    inp = _render_input((h, w), bars, tiles)
    out = _render_output((h, w), bars, tiles)
    return inp, out


def _semantic_validate(inp: np.ndarray, out: np.ndarray, expected_scene: dict) -> None:
    parsed = _extract_scene(inp)
    parsed_bars = {int(bar["color"]): bar for bar in parsed["bars"]}
    claimed = np.zeros(inp.shape, dtype=bool)
    patch_signatures: set[tuple[tuple[int, int], bytes]] = set()

    for color, tile_spec in expected_scene["tiles"].items():
        parsed_tile = parsed["tiles"][int(color)]["tile"]
        assert np.array_equal(parsed_tile, tile_spec["tile"]), "tile parsing changed tile content"
        top = int(tile_spec["top"])
        left = int(tile_spec["left"])
        outer = int(tile_spec["tile"].shape[0] + 2)
        assert np.array_equal(out[top : top + outer, left : left + outer], inp[top : top + outer, left : left + outer]), "tile template must stay unchanged"

    for bar in expected_scene["bars"]:
        parsed_bar = parsed_bars[int(bar["color"])]
        assert parsed_bar["row"] == bar["row"] and parsed_bar["c0"] == bar["c0"] and parsed_bar["length"] == bar["length"]
        tile = expected_scene["tiles"][int(bar["color"])]["tile"]
        mask = _diamond_mask(inp.shape, int(bar["row"]), int(bar["c0"]), int(bar["c1"]))
        assert not np.any(claimed & mask), "diamonds must not overlap in generated scenes"
        claimed |= mask
        interior = _erode4(mask)
        boundary = mask & ~interior
        assert np.all(out[boundary] == BOUNDARY), "diamond boundary must be grey"
        coords = np.argwhere(interior)
        if coords.size > 0:
            r0, c0 = coords.min(axis=0)
            for r, c in coords:
                rr = int(r)
                cc = int(c)
                expected = int(tile[(rr - int(r0)) % tile.shape[0], (cc - int(c0)) % tile.shape[1]])
                assert int(out[rr, cc]) == expected, "diamond interior must be tiled from interior top-left"
        r0 = int(bar["row"] - (bar["length"] - 1) // 2)
        r1 = int(bar["row"] + (bar["length"] - 1) // 2)
        c0b = int(bar["c0"])
        c1b = int(bar["c1"])
        patch = out[r0 : r1 + 1, c0b : c1b + 1]
        patch_signatures.add((patch.shape, patch.tobytes()))

    assert len(patch_signatures) == len(expected_scene["bars"]), "different bars must yield distinguishable diamond patches"
    assert np.array_equal(out[~claimed], inp[~claimed]), "only diamond regions may change"


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    built = _scene_attempt(rng)
    if built is None:
        return None
    inp, expected = built
    scene = _extract_scene(inp)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None
    try:
        _semantic_validate(inp, actual, scene)
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
