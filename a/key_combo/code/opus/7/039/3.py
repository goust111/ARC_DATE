from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6

PATH_COLORS = (GREEN, YELLOW, PINK)
TILE_COUNT = 3
TILE_SIZE_CHOICES = (5, 7)
MAX_GEN_ATTEMPTS = 400


def _spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for sep in separators:
        assert start < sep, "separator lines must leave nonempty regions"
        spans.append((start, sep - 1))
        start = sep + 1
    assert start < length, "separator lines must leave a final nonempty region"
    spans.append((start, length - 1))
    return spans


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _separator_layout(grid: np.ndarray) -> tuple[list[int], list[int], list[tuple[int, int]], list[tuple[int, int]]]:
    sep_allowed = {GREY, *PATH_COLORS}
    sep_rows = [int(r) for r in range(grid.shape[0]) if set(map(int, grid[r, :])).issubset(sep_allowed)]
    sep_cols = [int(c) for c in range(grid.shape[1]) if set(map(int, grid[:, c])).issubset(sep_allowed)]
    assert len(sep_rows) == TILE_COUNT - 1 and len(sep_cols) == TILE_COUNT - 1, "expected a 3x3 separator grid"
    row_spans = _spans(int(grid.shape[0]), sep_rows)
    col_spans = _spans(int(grid.shape[1]), sep_cols)
    assert len(row_spans) == TILE_COUNT and len(col_spans) == TILE_COUNT
    heights = {r1 - r0 + 1 for r0, r1 in row_spans}
    widths = {c1 - c0 + 1 for c0, c1 in col_spans}
    assert len(heights) == 1 and len(widths) == 1, "expected equal-sized tiles"
    return sep_rows, sep_cols, row_spans, col_spans


def _swap_red_blue(sprite: np.ndarray) -> np.ndarray:
    out = sprite.copy()
    out[sprite == BLUE] = RED
    out[sprite == RED] = BLUE
    return out


def _transform_sprite(sprite: np.ndarray, color: int) -> np.ndarray:
    if color == GREEN:
        return sprite.copy()
    if color == YELLOW:
        return np.fliplr(sprite)
    if color == PINK:
        return _swap_red_blue(sprite)
    raise AssertionError(f"unexpected path color: {color}")


def _neighbors(tile: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = tile
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < TILE_COUNT and 0 <= cc < TILE_COUNT:
            out.append((rr, cc))
    return out


def _source_and_sprite(
    grid: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
) -> tuple[tuple[int, int], np.ndarray]:
    source_tiles: list[tuple[int, int]] = []
    sprites: list[np.ndarray] = []
    for tr, (r0, r1) in enumerate(row_spans):
        for tc, (c0, c1) in enumerate(col_spans):
            tile = grid[r0 : r1 + 1, c0 : c1 + 1]
            if np.any((tile == BLUE) | (tile == RED)):
                assert {int(x) for x in np.unique(tile)}.issubset({BG, BLUE, RED}), "source tile should contain only the sprite"
                bb = _bbox((tile == BLUE) | (tile == RED))
                assert bb is not None
                sr0, sc0, sr1, sc1 = bb
                sprite = tile[sr0 : sr1 + 1, sc0 : sc1 + 1].copy()
                assert sprite.shape == (3, 3), "expected a centered 3x3 sprite"
                assert BLUE in sprite and RED in sprite, "sprite must use both colors"
                source_tiles.append((tr, tc))
                sprites.append(sprite)
    assert len(source_tiles) == 1, "expected exactly one source tile"
    return source_tiles[0], sprites[0]


def _edge_graph_for_color(
    grid: np.ndarray,
    color: int,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    sep_rows: list[int],
    sep_cols: list[int],
) -> dict[tuple[int, int], set[tuple[int, int]]]:
    graph: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for tr in range(TILE_COUNT - 1):
        sep_r = sep_rows[tr]
        for tc, (c0, c1) in enumerate(col_spans):
            if np.any(grid[sep_r, c0 : c1 + 1] == color):
                a = (tr, tc)
                b = (tr + 1, tc)
                graph.setdefault(a, set()).add(b)
                graph.setdefault(b, set()).add(a)
    for tr, (r0, r1) in enumerate(row_spans):
        for tc in range(TILE_COUNT - 1):
            sep_c = sep_cols[tc]
            if np.any(grid[r0 : r1 + 1, sep_c] == color):
                a = (tr, tc)
                b = (tr, tc + 1)
                graph.setdefault(a, set()).add(b)
                graph.setdefault(b, set()).add(a)
    return graph


def _target_tile(
    grid: np.ndarray,
    source: tuple[int, int],
    color: int,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    sep_rows: list[int],
    sep_cols: list[int],
) -> tuple[int, int]:
    graph = _edge_graph_for_color(grid, color, row_spans, col_spans, sep_rows, sep_cols)
    assert source in graph, "path must leave the source tile"
    seen = {source}
    stack = [source]
    while stack:
        node = stack.pop()
        for nxt in graph[node]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    degrees = {node: len(graph[node] & seen) for node in seen}
    endpoints = [node for node, deg in degrees.items() if deg == 1]
    assert len(seen) >= 2 and len(endpoints) == 2, "each color path must form a simple tile path"
    target = next(node for node in endpoints if node != source)
    return target


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the template sprite into the tiles reached by colored guide paths.

    Input structure:
    - A black canvas is partitioned into a 3x3 grid of equal square tiles by full
      grey separator rows and columns.
    - One non-corner tile contains a 3x3 red/blue sprite, while 2-3 colored guide
      paths lie on the separator lines using green, yellow, or pink pixels.

    Rule:
    - Keep the grey separators and the source sprite unchanged.
    - For each guide-path color, follow the same-color separator path leaving the
      source tile; the boundary segments it occupies identify one unique target tile.
    - Clear all guide pixels, then stamp the source sprite into that target tile:
      green copies it as-is, yellow applies a horizontal flip, and pink swaps red
      with blue.

    Key insight:
    - The path color encodes both which tile is reachable from the source and which
      transformation should be applied to the same template sprite at that endpoint.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    sep_rows, sep_cols, row_spans, col_spans = _separator_layout(g)
    source, sprite = _source_and_sprite(g, row_spans, col_spans)

    out = g.copy()
    path_mask = np.isin(out, PATH_COLORS)
    out[path_mask] = GREY

    tile_h = row_spans[0][1] - row_spans[0][0] + 1
    tile_w = col_spans[0][1] - col_spans[0][0] + 1
    assert tile_h >= 5 and tile_w >= 5 and tile_h % 2 == 1 and tile_w % 2 == 1

    for color in PATH_COLORS:
        if not np.any(g == color):
            continue
        target = _target_tile(g, source, color, row_spans, col_spans, sep_rows, sep_cols)
        r0, r1 = row_spans[target[0]]
        c0, c1 = col_spans[target[1]]
        stamp = _transform_sprite(sprite, color)
        top = r0 + (tile_h - stamp.shape[0]) // 2
        left = c0 + (tile_w - stamp.shape[1]) // 2
        out[top : top + stamp.shape[0], left : left + stamp.shape[1]] = stamp

    return out


def _segment_nodes(segment: tuple[str, int, int], tile_size: int) -> list[tuple[int, int]]:
    kind, a, b = segment
    sep_rows = [tile_size, 2 * tile_size + 1]
    sep_cols = [tile_size, 2 * tile_size + 1]
    parity = sep_rows[0] % 2
    if kind == "h":
        sep_r = sep_rows[a]
        c0 = b * (tile_size + 1)
        c1 = c0 + tile_size - 1
        return [(sep_r, c) for c in range(c0, c1 + 1) if c % 2 == parity]
    sep_c = sep_cols[b]
    r0 = a * (tile_size + 1)
    r1 = r0 + tile_size - 1
    return [(r, sep_c) for r in range(r0, r1 + 1) if r % 2 == parity]


def _build_nodes_for_route(
    route: dict[str, object],
    tile_size: int,
) -> list[tuple[int, int]]:
    segments = route["segments"]
    if len(segments) == 1:
        return _segment_nodes(segments[0], tile_size)

    first = _segment_nodes(segments[0], tile_size)
    second = _segment_nodes(segments[1], tile_size)
    if route["first_toward_low"]:
        first = list(reversed(first))
    if route["second_toward_low"]:
        second = list(reversed(second))

    sep_rows = [tile_size, 2 * tile_size + 1]
    sep_cols = [tile_size, 2 * tile_size + 1]
    turn_r = sep_rows[route["turn_sep_row"]]
    turn_c = sep_cols[route["turn_sep_col"]]
    return first + [(turn_r, turn_c)] + second


def _route_catalog() -> list[dict[str, object]]:
    return [
        {"target": (0, 1), "segments": [("h", 0, 1)], "is_corner": False, "intersections": set()},
        {"target": (2, 1), "segments": [("h", 1, 1)], "is_corner": False, "intersections": set()},
        {"target": (1, 0), "segments": [("v", 1, 0)], "is_corner": False, "intersections": set()},
        {"target": (1, 2), "segments": [("v", 1, 1)], "is_corner": False, "intersections": set()},
        {
            "target": (0, 0),
            "segments": [("h", 0, 1), ("v", 0, 0)],
            "turn_sep_row": 0,
            "turn_sep_col": 0,
            "first_toward_low": True,
            "second_toward_low": True,
            "is_corner": True,
            "intersections": {(0, 0)},
        },
        {
            "target": (0, 0),
            "segments": [("v", 1, 0), ("h", 0, 0)],
            "turn_sep_row": 0,
            "turn_sep_col": 0,
            "first_toward_low": True,
            "second_toward_low": True,
            "is_corner": True,
            "intersections": {(0, 0)},
        },
        {
            "target": (0, 2),
            "segments": [("h", 0, 1), ("v", 0, 1)],
            "turn_sep_row": 0,
            "turn_sep_col": 1,
            "first_toward_low": False,
            "second_toward_low": True,
            "is_corner": True,
            "intersections": {(0, 1)},
        },
        {
            "target": (0, 2),
            "segments": [("v", 1, 1), ("h", 0, 2)],
            "turn_sep_row": 0,
            "turn_sep_col": 1,
            "first_toward_low": True,
            "second_toward_low": False,
            "is_corner": True,
            "intersections": {(0, 1)},
        },
        {
            "target": (2, 0),
            "segments": [("h", 1, 1), ("v", 2, 0)],
            "turn_sep_row": 1,
            "turn_sep_col": 0,
            "first_toward_low": True,
            "second_toward_low": False,
            "is_corner": True,
            "intersections": {(1, 0)},
        },
        {
            "target": (2, 0),
            "segments": [("v", 1, 0), ("h", 1, 0)],
            "turn_sep_row": 1,
            "turn_sep_col": 0,
            "first_toward_low": False,
            "second_toward_low": True,
            "is_corner": True,
            "intersections": {(1, 0)},
        },
        {
            "target": (2, 2),
            "segments": [("h", 1, 1), ("v", 2, 1)],
            "turn_sep_row": 1,
            "turn_sep_col": 1,
            "first_toward_low": False,
            "second_toward_low": False,
            "is_corner": True,
            "intersections": {(1, 1)},
        },
        {
            "target": (2, 2),
            "segments": [("v", 1, 1), ("h", 1, 2)],
            "turn_sep_row": 1,
            "turn_sep_col": 1,
            "first_toward_low": False,
            "second_toward_low": False,
            "is_corner": True,
            "intersections": {(1, 1)},
        },
    ]


def _random_sprite() -> np.ndarray:
    while True:
        sprite = np.random.choice([BLUE, RED], size=(3, 3))
        if BLUE not in sprite or RED not in sprite:
            continue
        flipped = np.fliplr(sprite)
        swapped = _swap_red_blue(sprite)
        if np.array_equal(sprite, flipped):
            continue
        if np.array_equal(sprite, swapped):
            continue
        if np.array_equal(flipped, swapped):
            continue
        return sprite.astype(int)


def _choose_routes(count: int, require_corner: bool) -> list[dict[str, object]] | None:
    routes = _route_catalog()
    random.shuffle(routes)
    chosen: list[dict[str, object]] = []
    used_targets: set[tuple[int, int]] = set()
    used_segments: set[tuple[str, int, int]] = set()
    used_intersections: set[tuple[int, int]] = set()

    def backtrack(idx: int) -> bool:
        if len(chosen) == count:
            if require_corner and not any(bool(route["is_corner"]) for route in chosen):
                return False
            return True
        if idx >= len(routes):
            return False

        remaining = len(routes) - idx
        need = count - len(chosen)
        if remaining < need:
            return False

        route = routes[idx]
        segments = set(route["segments"])
        intersections = set(route["intersections"])
        target = route["target"]
        if (
            target not in used_targets
            and segments.isdisjoint(used_segments)
            and intersections.isdisjoint(used_intersections)
        ):
            chosen.append(route)
            used_targets.add(target)
            used_segments.update(segments)
            used_intersections.update(intersections)
            if backtrack(idx + 1):
                return True
            chosen.pop()
            used_targets.remove(target)
            used_segments.difference_update(segments)
            used_intersections.difference_update(intersections)

        return backtrack(idx + 1)

    if backtrack(0):
        return chosen
    return None


def _render_case() -> tuple[np.ndarray, np.ndarray] | None:
    tile_size = random.choice(TILE_SIZE_CHOICES)
    side = TILE_COUNT * tile_size + (TILE_COUNT - 1)
    grid = np.full((side, side), BG, dtype=int)

    sep_rows = [tile_size, 2 * tile_size + 1]
    sep_cols = [tile_size, 2 * tile_size + 1]
    grid[sep_rows, :] = GREY
    grid[:, sep_cols] = GREY

    source_tile = (1, 1)
    row0 = source_tile[0] * (tile_size + 1)
    col0 = source_tile[1] * (tile_size + 1)
    sprite = _random_sprite()
    top = row0 + (tile_size - 3) // 2
    left = col0 + (tile_size - 3) // 2
    grid[top : top + 3, left : left + 3] = sprite

    path_count = 3 if random.random() < 0.65 else 2
    require_corner = random.random() < 0.8
    routes = _choose_routes(path_count, require_corner)
    if routes is None:
        return None

    colors = list(PATH_COLORS)
    random.shuffle(colors)
    for color, route in zip(colors[:path_count], routes):
        for r, c in _build_nodes_for_route(route, tile_size):
            assert 0 <= r < side and 0 <= c < side
            assert grid[r, c] in {BG, GREY}, "paths must only occupy separator/background cells before coloring"
            grid[r, c] = color

    expected = main(grid)
    if np.array_equal(expected, grid):
        return None
    if not np.all(expected[sep_rows, :] == GREY):
        return None
    if not np.all(expected[:, sep_cols] == GREY):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        pair = _render_case()
        if pair is None:
            continue
        grid, expected = pair
        try:
            actual = main(grid)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid puzzle-3 input")
