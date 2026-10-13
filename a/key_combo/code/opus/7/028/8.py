from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
ORANGE = 7
TEAL = 8

PORT_COLORS = (BLUE, RED, GREEN)
TILE_SIZE = 5
GAP = 1
TILE_STRIDE = TILE_SIZE + GAP
TILE_COUNT = 3
GRID_SIZE = TILE_COUNT * TILE_SIZE + (TILE_COUNT - 1) * GAP
GEN_RETRIES = 300

_PATTERN_TEMPLATES = (
    ((1, 2), (2, 2), (3, 2)),
    ((0, 2), (1, 2), (2, 2), (3, 2)),
    ((1, 2), (2, 1), (2, 2), (2, 3)),
    ((1, 2), (2, 2), (3, 2), (2, 1), (2, 3)),
    ((1, 1), (1, 2), (1, 3), (2, 2), (3, 2)),
    ((1, 2), (2, 2), (3, 2), (3, 1), (3, 3)),
    ((0, 2), (1, 2), (2, 1), (2, 2), (2, 3)),
    ((1, 2), (2, 2), (3, 2), (4, 2), (2, 1), (2, 3)),
    ((1, 1), (1, 2), (1, 3), (2, 2), (3, 2), (4, 2)),
)


def _runs(active: list[bool]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for i, value in enumerate(active):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(active) - 1))
    return runs


def _tile_runs(g: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    row_runs = _runs([not np.all(g[r, :] == BLACK) for r in range(g.shape[0])])
    col_runs = _runs([not np.all(g[:, c] == BLACK) for c in range(g.shape[1])])
    assert len(row_runs) == TILE_COUNT and len(col_runs) == TILE_COUNT, "expected a 3x3 tile lattice"
    assert all(r1 - r0 + 1 == TILE_SIZE for r0, r1 in row_runs), "expected 5x5 tiles"
    assert all(c1 - c0 + 1 == TILE_SIZE for c0, c1 in col_runs), "expected 5x5 tiles"
    return row_runs, col_runs


def _tile_origin(tile_r: int, tile_c: int) -> tuple[int, int]:
    return tile_r * TILE_STRIDE, tile_c * TILE_STRIDE


def _tile_view(g: np.ndarray, tile_r: int, tile_c: int) -> np.ndarray:
    r0, c0 = _tile_origin(tile_r, tile_c)
    return g[r0 : r0 + TILE_SIZE, c0 : c0 + TILE_SIZE]


def _complete_tile(tile: np.ndarray) -> np.ndarray:
    out = tile.copy()
    if not np.any(tile == TEAL):
        return out

    cols = [c for c in range(TILE_SIZE) if np.any(tile[:, c] == TEAL)]
    assert cols, "occluder columns must be non-empty"
    assert cols == list(range(cols[0], cols[-1] + 1)), "occluder must be a solid vertical strip"
    assert cols[0] in (0, TILE_SIZE - len(cols)), "occluder must cover the left or right side"

    src = tile.copy()
    for r in range(TILE_SIZE):
        for c in cols:
            mc = TILE_SIZE - 1 - c
            mirrored = int(src[r, mc])
            if mirrored == TEAL:
                mirrored = ORANGE
            out[r, c] = mirrored
    return out


def _complete_scene(g: np.ndarray) -> np.ndarray:
    out = np.asarray(g, dtype=int).copy()
    row_runs, col_runs = _tile_runs(out)
    for tile_r, (r0, r1) in enumerate(row_runs):
        for tile_c, (c0, c1) in enumerate(col_runs):
            tile = out[r0 : r1 + 1, c0 : c1 + 1]
            out[r0 : r1 + 1, c0 : c1 + 1] = _complete_tile(tile)
    return out


def _port_entry(
    g: np.ndarray,
    r: int,
    c: int,
) -> tuple[int, int] | None:
    h, w = g.shape
    if r == 0 and r + 1 < h and int(g[r + 1, c]) != BLACK:
        return None
    if r == h - 1 and r - 1 >= 0 and int(g[r - 1, c]) != BLACK:
        return None
    if c == 0 and c + 1 < w and int(g[r, c + 1]) != BLACK:
        return None
    if c == w - 1 and c - 1 >= 0 and int(g[r, c - 1]) != BLACK:
        return None

    candidates = []
    if r - 1 >= 0 and int(g[r - 1, c]) == BLACK:
        candidates.append((r - 1, c))
    if r + 1 < h and int(g[r + 1, c]) == BLACK:
        candidates.append((r + 1, c))
    if c - 1 >= 0 and int(g[r, c - 1]) == BLACK:
        candidates.append((r, c - 1))
    if c + 1 < w and int(g[r, c + 1]) == BLACK:
        candidates.append((r, c + 1))
    if len(candidates) != 1:
        return None
    return candidates[0]


def _collect_ports(g: np.ndarray) -> dict[int, list[dict[str, tuple[int, int] | int]]]:
    row_runs, col_runs = _tile_runs(g)
    ports: dict[int, list[dict[str, tuple[int, int] | int]]] = defaultdict(list)

    for tile_r, (r0, r1) in enumerate(row_runs):
        for tile_c, (c0, c1) in enumerate(col_runs):
            seen: set[tuple[int, int]] = set()
            for rr in range(r0, r1 + 1):
                for cc in range(c0, c1 + 1):
                    if rr not in (r0, r1) and cc not in (c0, c1):
                        continue
                    color = int(g[rr, cc])
                    if color not in PORT_COLORS:
                        continue
                    entry = _port_entry(g, rr, cc)
                    if entry is None or (rr, cc) in seen:
                        continue
                    ports[color].append(
                        {
                            "tile": (tile_r, tile_c),
                            "port": (rr, cc),
                            "entry": entry,
                        }
                    )
                    seen.add((rr, cc))
    return dict(ports)


def _inclusive_range(a: int, b: int) -> range:
    step = 1 if b >= a else -1
    return range(a, b + step, step)


def _path_for_pair(pair: list[dict[str, tuple[int, int] | int]]) -> list[tuple[int, int]]:
    assert len(pair) == 2, "expected exactly two detected ports per color"
    a, b = pair
    ta_r, ta_c = a["tile"]
    tb_r, tb_c = b["tile"]
    ea_r, ea_c = a["entry"]
    eb_r, eb_c = b["entry"]

    cells: list[tuple[int, int]] = []
    if ta_r == tb_r:
        start, end = (a, b) if ta_c < tb_c else (b, a)
        start_tile_r, start_tile_c = start["tile"]
        sr, sc = start["entry"]
        er, ec = end["entry"]
        if sr == er:
            for c in _inclusive_range(sc, ec):
                cells.append((sr, c))
        else:
            turn_c = (start_tile_c + 1) * TILE_STRIDE - 1
            for c in _inclusive_range(sc, turn_c):
                cells.append((sr, c))
            for r in _inclusive_range(sr, er):
                cells.append((r, turn_c))
            for c in _inclusive_range(turn_c, ec):
                cells.append((er, c))
    else:
        assert ta_c == tb_c, "same-color tiles must share a tile row or tile column"
        start, end = (a, b) if ta_r < tb_r else (b, a)
        start_tile_r, start_tile_c = start["tile"]
        sr, sc = start["entry"]
        er, ec = end["entry"]
        if sc == ec:
            for r in _inclusive_range(sr, er):
                cells.append((r, sc))
        else:
            turn_r = (start_tile_r + 1) * TILE_STRIDE - 1
            for r in _inclusive_range(sr, turn_r):
                cells.append((r, sc))
            for c in _inclusive_range(sc, ec):
                cells.append((turn_r, c))
            for r in _inclusive_range(turn_r, er):
                cells.append((r, ec))

    deduped: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for cell in cells:
        if cell not in seen:
            deduped.append(cell)
            seen.add(cell)
    return deduped


def _draw_connections(scene: np.ndarray, ports: dict[int, list[dict[str, tuple[int, int] | int]]]) -> np.ndarray:
    out = scene.copy()
    planned: dict[tuple[int, int], int] = {}

    for color, pair in ports.items():
        for r, c in _path_for_pair(pair):
            assert int(scene[r, c]) == BLACK, "paths may only traverse black channels"
            planned[(r, c)] = color if (r, c) not in planned else min(planned[(r, c)], color)

    for (r, c), color in planned.items():
        out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Mirror-complete occluded tiles, reveal channel-side ports, then connect matched colors through the black corridor lattice.

    Input structure:
    - The grid is a 3x3 lattice of 5x5 orange tiles separated by 1-cell black channel rows and columns.
    - Each tile contains a small grey 4-connected motif that is left-right symmetric, and some tiles have a teal occluder covering the left or right side.
    - Red, blue, and green edge pixels act as ports only when they sit on a tile edge that directly touches a black channel.

    Rule:
    - For every occluded tile, mirror the visible half across the tile's vertical axis into the teal strip to restore the hidden half.
    - After restoration, collect every channel-adjacent edge port of colors blue, red, or green and pair the two tiles of each color.
    - If a pair shares a tile row, draw its line through black channels with horizontal-first routing; if it shares a tile column, use vertical-first routing. When paths overlap, keep the smaller color id.

    Key insight:
    - The useful ports are not just colored edge pixels: some are hidden until tile-local mirror repair, and only the channel-adjacent repaired ports determine the corridor routes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape == (GRID_SIZE, GRID_SIZE), "expected a 17x17 grid"

    repaired = _complete_scene(g)
    ports = _collect_ports(repaired)
    assert ports, "expected at least one port color"
    assert all(len(pair) == 2 for pair in ports.values()), "each port color must produce exactly two usable ports"
    return _draw_connections(repaired, ports)


def _blank_scene() -> np.ndarray:
    scene = np.full((GRID_SIZE, GRID_SIZE), ORANGE, dtype=int)
    for k in range(1, TILE_COUNT):
        idx = k * TILE_STRIDE - 1
        scene[idx, :] = BLACK
        scene[:, idx] = BLACK
    return scene


def _paint_pattern(scene: np.ndarray, tile_r: int, tile_c: int, coords: tuple[tuple[int, int], ...]) -> None:
    tile = _tile_view(scene, tile_r, tile_c)
    for r, c in coords:
        tile[r, c] = GREY


def _place_top_center(scene: np.ndarray, tile_r: int, tile_c: int, color: int) -> None:
    _tile_view(scene, tile_r, tile_c)[0, 2] = color


def _place_bottom_center(scene: np.ndarray, tile_r: int, tile_c: int, color: int) -> None:
    _tile_view(scene, tile_r, tile_c)[4, 2] = color


def _place_left_right_pair(scene: np.ndarray, tile_r: int, tile_c: int, color: int, row: int) -> None:
    tile = _tile_view(scene, tile_r, tile_c)
    tile[row, 0] = color
    tile[row, 4] = color


def _apply_side_occluder(scene: np.ndarray, tile_r: int, tile_c: int, side: str, width: int = 2) -> None:
    assert width in (2, 3)
    tile = _tile_view(scene, tile_r, tile_c)
    if side == "left":
        tile[:, :width] = TEAL
    else:
        assert side == "right"
        tile[:, TILE_SIZE - width :] = TEAL


def _maybe_flip(grid: np.ndarray, flip_lr: bool, flip_ud: bool) -> np.ndarray:
    out = grid
    if flip_lr:
        out = np.fliplr(out)
    if flip_ud:
        out = np.flipud(out)
    return out.copy()


def _base_recovered_scene() -> tuple[np.ndarray, dict[str, object]]:
    scene = _blank_scene()

    for tile_r in range(TILE_COUNT):
        for tile_c in range(TILE_COUNT):
            _paint_pattern(scene, tile_r, tile_c, random.choice(_PATTERN_TEMPLATES))

    blue_color = BLUE
    red_color = RED
    green_color = GREEN

    blue_variant = random.choice(("lt_rb", "lb_rt"))
    if blue_variant == "lt_rb":
        _place_top_center(scene, 1, 0, blue_color)
        _place_bottom_center(scene, 1, 2, blue_color)
    else:
        _place_bottom_center(scene, 1, 0, blue_color)
        _place_top_center(scene, 1, 2, blue_color)

    _place_top_center(scene, 2, 1, green_color)
    _place_top_center(scene, 2, 2, green_color)

    red_top_row = random.randint(1, 3)
    red_bottom_row = random.randint(1, 3)
    _place_left_right_pair(scene, 0, 2, red_color, red_top_row)
    _place_left_right_pair(scene, 2, 2, red_color, red_bottom_row)

    meta = {
        "blue_variant": blue_variant,
        "red_top_row": red_top_row,
        "red_bottom_row": red_bottom_row,
    }
    return scene, meta


def _occluded_input_from_recovered(scene: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    inp = scene.copy()

    # Mandatory hidden-port tile: right outer column, inward left edge hidden.
    _apply_side_occluder(inp, 0, 2, "left", width=2)
    hidden_port = (0, 2)

    extra_tiles = [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (1, 2),
        (2, 0),
        (2, 1),
    ]
    random.shuffle(extra_tiles)
    extra_count = random.randint(2, 3)
    occluded_tiles = [hidden_port]

    for tile_r, tile_c in extra_tiles[:extra_count]:
        side = random.choice(("left", "right"))
        width = 2
        if (tile_r, tile_c) in {(1, 0), (1, 2), (2, 1)}:
            width = 2
        _apply_side_occluder(inp, tile_r, tile_c, side, width=width)
        occluded_tiles.append((tile_r, tile_c))

    return inp, {"occluded_tiles": tuple(occluded_tiles)}


def _semantic_checks(inp: np.ndarray, out: np.ndarray) -> None:
    repaired = _complete_scene(inp)
    ports = _collect_ports(repaired)
    assert set(ports) == set(PORT_COLORS), "generator must expose all three connection colors"
    assert all(len(pair) == 2 for pair in ports.values()), "each color must have two usable ports"

    hidden_revealed = False
    for pair in ports.values():
        for port in pair:
            r, c = port["port"]
            if int(inp[r, c]) == TEAL and int(repaired[r, c]) in PORT_COLORS:
                hidden_revealed = True
    assert hidden_revealed, "at least one usable port must be revealed only after mirror completion"

    turned = False
    crossings: dict[tuple[int, int], list[int]] = defaultdict(list)
    for color, pair in ports.items():
        path = _path_for_pair(pair)
        rows = {r for r, _ in path}
        cols = {c for _, c in path}
        if len(rows) > 1 and len(cols) > 1:
            turned = True
        for cell in path:
            crossings[cell].append(color)
    assert turned, "at least one connection must turn at a channel intersection"

    overlap_cells = {cell: colors for cell, colors in crossings.items() if len(set(colors)) > 1}
    assert overlap_cells, "at least one crossing is required to exercise path priority"
    for cell, colors in overlap_cells.items():
        r, c = cell
        assert int(out[r, c]) == min(colors), "crossings must keep the smallest color id"

    assert np.array_equal(out, _draw_connections(repaired, ports)), "semantic reconstruction must match the solver output"


def _attempt_generate() -> np.ndarray:
    recovered, _meta = _base_recovered_scene()
    inp, _occ_meta = _occluded_input_from_recovered(recovered)

    flip_lr = bool(random.getrandbits(1))
    flip_ud = bool(random.getrandbits(1))
    inp = _maybe_flip(inp, flip_lr, flip_ud)

    out = main(inp)
    _semantic_checks(inp, out)
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            return _attempt_generate()
        except AssertionError:
            continue
    raise RuntimeError(f"failed to generate a valid sample after {GEN_RETRIES} attempts")
