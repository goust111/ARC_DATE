from __future__ import annotations

import random
from collections import Counter

import numpy as np


Coord = tuple[int, int]

GRID_SHAPE = (11, 11)
TILE_SIZE = 3
BG = 0
SEP = 5
PATH_COLORS = (1, 2, 3, 4, 7)
SLOT_ORIGINS = (0, 4, 8)
INPUT_SLOTS = [(r, c) for r in SLOT_ORIGINS for c in SLOT_ORIGINS]
OUTPUT_SLOTS = [(r, c) for r in reversed(SLOT_ORIGINS) for c in SLOT_ORIGINS]
MAX_GENERATION_ATTEMPTS = 400


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _separator_segments(length: int, dividers: list[int]) -> list[tuple[int, int]]:
    segments = []
    start = 0
    for divider in dividers + [length]:
        segments.append((start, divider))
        start = divider + 1
    return segments


def _tile_segments(g: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    sep_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == SEP)]
    sep_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == SEP)]
    assert sep_rows == [3, 7], "expected grey separator rows at 3 and 7"
    assert sep_cols == [3, 7], "expected grey separator cols at 3 and 7"
    row_segments = _separator_segments(g.shape[0], sep_rows)
    col_segments = _separator_segments(g.shape[1], sep_cols)
    assert row_segments == [(0, 3), (4, 7), (8, 11)]
    assert col_segments == [(0, 3), (4, 7), (8, 11)]
    return row_segments, col_segments


def _endpoint_key(cell: Coord) -> tuple[int, int]:
    r, c = cell
    return abs(r) + abs(c), r


def _ordered_path_from_mask(mask: np.ndarray) -> tuple[list[Coord], bool]:
    coords = np.argwhere(mask)
    cells = {tuple(map(int, coord)) for coord in coords}
    assert len(cells) == len(coords), "path cells must be unique"
    assert 2 <= len(cells) <= 7, "valid 3x3 path lengths are 2..7"

    degrees = {
        cell: sum((nbr in cells) for nbr in neighbors4(*cell))
        for cell in cells
    }
    endpoints = sorted(cell for cell, degree in degrees.items() if degree == 1)
    assert len(endpoints) == 2, "path must have exactly two endpoints"
    assert all(degree in (1, 2) for degree in degrees.values()), "path must not branch"

    d0 = abs(endpoints[0][0]) + abs(endpoints[0][1])
    d1 = abs(endpoints[1][0]) + abs(endpoints[1][1])
    tie = d0 == d1
    start = min(endpoints, key=_endpoint_key)

    path = []
    prev = None
    cur = start
    while True:
        path.append(cur)
        nxt = [nbr for nbr in neighbors4(*cur) if nbr in cells and nbr != prev]
        if not nxt:
            break
        assert len(nxt) == 1, "path traversal must be unique"
        prev, cur = cur, nxt[0]

    assert len(path) == len(cells), "path traversal must cover the whole path"
    return path, tie


def _path_info(tile: np.ndarray) -> dict:
    assert tile.shape == (TILE_SIZE, TILE_SIZE)
    assert set(map(int, tile.ravel())) <= {BG, *PATH_COLORS}
    mask = tile != BG
    assert np.any(mask), "tile must contain a foreground path"

    # Enforce a single multicolor component.
    coords = {tuple(map(int, coord)) for coord in np.argwhere(mask)}
    seen = set()
    stack = [next(iter(coords))]
    while stack:
        cell = stack.pop()
        if cell in seen:
            continue
        seen.add(cell)
        for nbr in neighbors4(*cell):
            if nbr in coords and nbr not in seen:
                stack.append(nbr)
    assert seen == coords, "tile must contain exactly one path component"

    path, tie = _ordered_path_from_mask(mask)
    colors = [int(tile[r, c]) for r, c in path]
    assert len(colors) == len(path)
    assert all(colors[i] != colors[i + 1] for i in range(len(colors) - 1)), "adjacent path colors must differ"

    return {
        "tile": tile.copy(),
        "path": path,
        "colors": tuple(colors),
        "length": len(path),
        "start": path[0],
        "start_tie": tie,
        "sort_key": (len(path), path[0][0], path[0][1], tuple(path)),
    }


def _reverse_tile_colors(info: dict) -> np.ndarray:
    out = np.zeros((TILE_SIZE, TILE_SIZE), dtype=int)
    for (r, c), color in zip(info["path"], reversed(info["colors"])):
        out[r, c] = int(color)
    return out


def _base_canvas() -> np.ndarray:
    g = np.full(GRID_SHAPE, BG, dtype=int)
    g[3, :] = SEP
    g[7, :] = SEP
    g[:, 3] = SEP
    g[:, 7] = SEP
    return g


def _extract_tiles(g: np.ndarray) -> list[np.ndarray]:
    row_segments, col_segments = _tile_segments(g)
    tiles = []
    for r0, r1 in row_segments:
        for c0, c1 in col_segments:
            tile = g[r0:r1, c0:c1]
            assert tile.shape == (TILE_SIZE, TILE_SIZE)
            tiles.append(tile.copy())
    assert len(tiles) == 9
    return tiles


def _extract_tiles_by_slots(g: np.ndarray, slots: list[tuple[int, int]]) -> list[np.ndarray]:
    tiles = []
    for r0, c0 in slots:
        tiles.append(g[r0 : r0 + TILE_SIZE, c0 : c0 + TILE_SIZE].copy())
    return tiles


def _place_tiles(tiles: list[np.ndarray], slots: list[tuple[int, int]]) -> np.ndarray:
    out = _base_canvas()
    for tile, (r0, c0) in zip(tiles, slots):
        out[r0 : r0 + TILE_SIZE, c0 : c0 + TILE_SIZE] = tile
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort the separated path tiles by traversal length and reverse each path's color sequence when repacking them.

    Input structure:
    - The grid is an 11x11 black canvas split by full grey rows and columns at indices 3 and 7 into nine 3x3 tiles.
    - Each tile contains exactly one 4-connected one-pixel-wide multicolor path on black background.
    - Every path has exactly two endpoints and no branches; adjacent path pixels always have different colors.

    Rule:
    - In each tile, choose the endpoint nearest to the tile's top-left corner by Manhattan distance; if tied, choose the smaller row.
    - Traverse the path from that endpoint to the other endpoint, recording both the ordered color sequence and the path length.
    - Sort the tiles by increasing path length; if lengths tie, break them by the chosen local start cell and then by the traversed local coordinate sequence so the order stays deterministic.
    - Repack the sorted tiles into destination slots from bottom-left to top-right, and within each moved tile write the recorded colors back in reverse order along the same path geometry.
    - Keep the grey separator lines unchanged.

    Key insight:
    - A single endpoint-to-endpoint traversal serves two roles at once: it provides the ordering key for tile rearrangement and the direction needed to reverse the color string on the unchanged path shape.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == GRID_SHAPE, "expected an 11x11 grid"

    infos = [_path_info(tile) for tile in _extract_tiles(g)]
    ordered_infos = sorted(infos, key=lambda info: info["sort_key"])
    transformed_tiles = [_reverse_tile_colors(info) for info in ordered_infos]
    return _place_tiles(transformed_tiles, OUTPUT_SLOTS)


def _enumerate_simple_paths() -> list[dict]:
    all_paths = set()

    def dfs(path: list[Coord], used: set[Coord]) -> None:
        if len(path) >= 2:
            canon = min(tuple(path), tuple(reversed(path)))
            all_paths.add(canon)
        if len(path) == 9:
            return
        r, c = path[-1]
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < TILE_SIZE and 0 <= nc < TILE_SIZE and (nr, nc) not in used:
                used.add((nr, nc))
                path.append((nr, nc))
                dfs(path, used)
                path.pop()
                used.remove((nr, nc))

    for start in [(r, c) for r in range(TILE_SIZE) for c in range(TILE_SIZE)]:
        dfs([start], {start})

    library = []
    for coords in sorted(all_paths):
        tile = np.zeros((TILE_SIZE, TILE_SIZE), dtype=int)
        for idx, (r, c) in enumerate(coords):
            tile[r, c] = PATH_COLORS[idx % 2]
        try:
            info = _path_info(tile)
        except AssertionError:
            continue
        library.append(
            {
                "coords": tuple(info["path"]),
                "length": info["length"],
                "start": info["start"],
                "start_tie": info["start_tie"],
                "sort_key": info["sort_key"],
            }
        )
    return library


PATH_LIBRARY = _enumerate_simple_paths()
PATHS_BY_LENGTH: dict[int, list[dict]] = {}
for entry in PATH_LIBRARY:
    PATHS_BY_LENGTH.setdefault(entry["length"], []).append(entry)


def _sample_color_sequence(length: int) -> tuple[int, ...]:
    for _ in range(200):
        palette = random.sample(PATH_COLORS, random.randint(2, min(len(PATH_COLORS), max(2, length))))
        seq = [random.choice(palette)]
        for _idx in range(1, length):
            choices = [color for color in palette if color != seq[-1]]
            seq.append(random.choice(choices))
        if seq != list(reversed(seq)):
            return tuple(seq)
    raise RuntimeError("failed to generate a non-palindromic color sequence")


def _paint_path(coords: tuple[Coord, ...], colors: tuple[int, ...]) -> np.ndarray:
    tile = np.zeros((TILE_SIZE, TILE_SIZE), dtype=int)
    for (r, c), color in zip(coords, colors):
        tile[r, c] = int(color)
    return tile


def _pick_path_set() -> list[dict] | None:
    duplicate_lengths = [
        length
        for length, entries in PATHS_BY_LENGTH.items()
        if len({entry["start"] for entry in entries}) >= 2
    ]
    tie_lengths = [
        length
        for length, entries in PATHS_BY_LENGTH.items()
        if any(entry["start_tie"] for entry in entries)
    ]
    if len(duplicate_lengths) < 3 or not tie_lengths:
        return None

    tie_length = random.choice(tie_lengths)
    dup_lengths = random.sample(duplicate_lengths, 3)
    if tie_length not in dup_lengths:
        dup_lengths[0] = tie_length

    selected = []
    for length in range(2, 8):
        entries = PATHS_BY_LENGTH[length]
        need = 2 if length in dup_lengths else 1
        chosen: list[dict] = []

        if length == tie_length:
            chosen.append(random.choice([entry for entry in entries if entry["start_tie"]]))

        while len(chosen) < need:
            candidate = random.choice(entries)
            if any(candidate["sort_key"] == entry["sort_key"] for entry in chosen):
                continue
            if need == 2 and chosen and candidate["start"] == chosen[0]["start"]:
                continue
            chosen.append(candidate)
        selected.extend(chosen)

    random.shuffle(selected)

    if len(selected) != 9:
        return None
    if len({entry["sort_key"] for entry in selected}) != 9:
        return None
    if not any(entry["start_tie"] for entry in selected):
        return None
    return selected


def _semantic_signature(tile: np.ndarray) -> tuple[int, ...]:
    return tuple(map(int, tile.ravel()))


def _validate_semantics(inp: np.ndarray, out: np.ndarray) -> bool:
    try:
        in_infos = [_path_info(tile) for tile in _extract_tiles(inp)]
        out_tiles_in_output_order = _extract_tiles_by_slots(out, OUTPUT_SLOTS)
        out_infos = [_path_info(tile) for tile in out_tiles_in_output_order]
    except AssertionError:
        return False

    lengths = [info["length"] for info in in_infos]
    counts = Counter(lengths)
    if sorted(counts.values()) != [1, 1, 1, 2, 2, 2]:
        return False
    if not any(info["start_tie"] for info in in_infos):
        return False
    if any(info["colors"] == tuple(reversed(info["colors"])) for info in in_infos):
        return False
    if len({_semantic_signature(info["tile"]) for info in in_infos}) != 9:
        return False

    ordered = sorted(in_infos, key=lambda info: info["sort_key"])
    expected_out_tiles = [_reverse_tile_colors(info) for info in ordered]
    actual_out_tiles = out_tiles_in_output_order
    for expected, actual in zip(expected_out_tiles, actual_out_tiles):
        if not np.array_equal(expected, actual):
            return False
    if np.array_equal(inp, out):
        return False
    if all(np.array_equal(tile, expected) for tile, expected in zip(_extract_tiles(inp), expected_out_tiles)):
        return False

    # Geometry must stay fixed within each sorted tile.
    for source_info, dest_info in zip(ordered, out_infos):
        if tuple(source_info["path"]) != tuple(dest_info["path"]):
            return False
        if tuple(reversed(source_info["colors"])) != dest_info["colors"]:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    shapes = _pick_path_set()
    if shapes is None:
        return None

    tiles = []
    reversed_signatures = set()
    for shape in shapes:
        colors = _sample_color_sequence(shape["length"])
        tile = _paint_path(shape["coords"], colors)
        info = _path_info(tile)
        reversed_tile = _reverse_tile_colors(info)
        tile_sig = _semantic_signature(tile)
        reversed_sig = _semantic_signature(reversed_tile)
        if tile_sig in reversed_signatures or reversed_sig in reversed_signatures:
            return None
        reversed_signatures.add(tile_sig)
        reversed_signatures.add(reversed_sig)
        tiles.append(tile)

    infos = [_path_info(tile) for tile in tiles]
    ordered_tiles = [_reverse_tile_colors(info) for info in sorted(infos, key=lambda info: info["sort_key"])]

    perm = list(range(9))
    random.shuffle(perm)
    if perm == list(range(9)):
        return None
    input_tiles = [tiles[idx] for idx in perm]
    inp = _place_tiles(input_tiles, INPUT_SLOTS)
    out = main(inp)

    if not _validate_semantics(inp, out):
        return None

    # Additional independent check: destination slot order really follows the sorted keys.
    extracted_in = _extract_tiles(inp)
    extracted_out = _extract_tiles(out)
    sorted_from_input = [_reverse_tile_colors(info) for info in sorted((_path_info(tile) for tile in extracted_in), key=lambda info: info["sort_key"])]
    extracted_out_in_output_order = _extract_tiles_by_slots(out, OUTPUT_SLOTS)
    if any(not np.array_equal(a, b) for a, b in zip(sorted_from_input, extracted_out_in_output_order)):
        return None
    if len({_semantic_signature(tile) for tile in ordered_tiles}) != 9:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle 7 input")
