import random
from collections import Counter

import numpy as np


BLACK = 0
GREY = 5
TILE_SIZE = 3
SEP = 1
NON_FIXED_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
CORNER_TO_COORD = {
    "tl": (0, 0),
    "tr": (0, 2),
    "br": (2, 2),
    "bl": (2, 0),
}
CORNER_TO_ROT = {
    "tl": 0,
    "tr": 3,  # np.rot90 uses counter-clockwise quarter turns.
    "br": 2,
    "bl": 1,
}
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GEN_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _tile_spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    assert separators == sorted(separators)
    assert all(b - a > 1 for a, b in zip(separators, separators[1:])), "separator lines must be width 1"
    spans = []
    prev = -1
    for sep_idx in separators + [length]:
        start = prev + 1
        end = sep_idx - 1
        assert start <= end, "separator layout must leave non-empty tiles"
        spans.append((start, end))
        prev = sep_idx
    return spans


def _pattern_key(tile: np.ndarray) -> tuple[int, ...]:
    return tuple(int(x) for x in tile.ravel())


def _rotationally_asymmetric(tile: np.ndarray) -> bool:
    return all(not np.array_equal(tile, np.rot90(tile, k)) for k in (1, 2, 3))


def _parse_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) > 0
    h, w = g.shape

    sep_rows = [r for r in range(h) if np.all(g[r, :] == GREY)]
    sep_cols = [c for c in range(w) if np.all(g[:, c] == GREY)]
    assert sep_rows and sep_cols, "expected full grey separator lines"

    row_spans = _tile_spans(h, sep_rows)
    col_spans = _tile_spans(w, sep_cols)
    assert all(r1 - r0 + 1 == TILE_SIZE for r0, r1 in row_spans), "tile height must be 3"
    assert all(c1 - c0 + 1 == TILE_SIZE for c0, c1 in col_spans), "tile width must be 3"

    sprites = []
    markers = []
    empties = []
    for tr, (r0, r1) in enumerate(row_spans):
        for tc, (c0, c1) in enumerate(col_spans):
            tile = g[r0 : r1 + 1, c0 : c1 + 1].copy()
            fg_mask = (tile != BLACK) & (tile != GREY)
            if np.any(fg_mask):
                assert not np.any(tile == GREY), "sprite tiles may not contain grey marker pixels"
                coords = np.argwhere(fg_mask)
                assert 2 <= coords.shape[0] <= 4, "sprite must have 2-4 foreground cells"
                assert len(_components(fg_mask)) == 1, "sprite must be 4-connected"
                assert 1 <= len(set(map(int, tile[fg_mask]))) <= 2, "sprite must use 1-2 colors"
                sprites.append(
                    {
                        "pos": (tr, tc),
                        "bounds": (r0, r1, c0, c1),
                        "tile": tile,
                        "key": _pattern_key(tile),
                    }
                )
            elif np.any(tile == GREY):
                marker_coords = np.argwhere(tile == GREY)
                assert marker_coords.shape[0] == 1, "marked blank tile must contain exactly one grey pixel"
                rr, cc = map(int, marker_coords[0])
                corner = next((name for name, coord in CORNER_TO_COORD.items() if coord == (rr, cc)), None)
                assert corner is not None, "marker must sit in a 3x3 corner"
                assert np.count_nonzero(tile == BLACK) == 8, "marked tile must otherwise be black"
                markers.append(
                    {
                        "pos": (tr, tc),
                        "bounds": (r0, r1, c0, c1),
                        "corner": corner,
                        "tile": tile,
                    }
                )
            else:
                empties.append({"pos": (tr, tc), "bounds": (r0, r1, c0, c1), "tile": tile})

    assert len(sprites) >= 3, "need repeated template plus distractors"
    assert markers, "expected at least one marked tile"

    counts = Counter(sprite["key"] for sprite in sprites)
    duplicate_keys = [key for key, count in counts.items() if count == 2]
    assert len(duplicate_keys) == 1, "expected exactly one duplicated sprite pattern"
    assert all(count == 1 or count == 2 for count in counts.values()), "all non-template sprites must be unique"
    template_key = duplicate_keys[0]
    template_records = [sprite for sprite in sprites if sprite["key"] == template_key]
    template_tile = template_records[0]["tile"]

    return {
        "grid": g,
        "row_spans": row_spans,
        "col_spans": col_spans,
        "sprites": sprites,
        "markers": markers,
        "empties": empties,
        "template_key": template_key,
        "template_tile": template_tile,
        "template_positions": [sprite["pos"] for sprite in template_records],
    }


def _expected_output(parsed: dict) -> np.ndarray:
    out = parsed["grid"].copy()
    template = parsed["template_tile"]
    for sprite in parsed["sprites"]:
        r0, r1, c0, c1 = sprite["bounds"]
        if sprite["key"] == parsed["template_key"]:
            out[r0 : r1 + 1, c0 : c1 + 1] = BLACK
    for marker in parsed["markers"]:
        r0, r1, c0, c1 = marker["bounds"]
        out[r0 : r1 + 1, c0 : c1 + 1] = np.rot90(template, CORNER_TO_ROT[marker["corner"]])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify the only duplicated 3x3 sprite tile, erase its two source copies, and stamp rotated copies into grey-corner marker tiles.

    Input structure:
    - Grey full rows and columns partition the canvas into equal 3x3 inner tiles.
    - Sprite tiles contain one 4-connected 2-4-cell pattern in 1-2 non-black, non-grey colors.
    - Exactly one full 3x3 sprite pattern appears in two different tiles; all other sprite patterns appear once.
    - Marked blank tiles are otherwise black and contain exactly one grey pixel in one of the four inner corners.

    Rule:
    - Compare the full 3x3 contents of all sprite tiles and select the unique pattern that occurs exactly twice as the template.
    - Decode each marked blank tile's grey corner as a rotation: top-left=0, top-right=clockwise 90, bottom-right=180, bottom-left=clockwise 270.
    - Replace each marked blank tile by the correspondingly rotated template.
    - Clear the two original template tiles to black, while leaving separator lines, unmarked blank tiles, and non-template sprite tiles unchanged.

    Key insight:
    - Three independent cues must be combined: repetition chooses which tile is the stamp, marker presence chooses where to stamp, and marker corner position chooses how to rotate it.
    """
    parsed = _parse_grid(grid)
    return _expected_output(parsed)


def _random_connected_mask(rng: np.random.Generator, cells: int) -> np.ndarray:
    mask = np.zeros((TILE_SIZE, TILE_SIZE), dtype=bool)
    start = tuple(int(x) for x in rng.integers(0, TILE_SIZE, size=2))
    mask[start] = True
    while int(mask.sum()) < cells:
        frontier = []
        for r, c in np.argwhere(mask):
            for dr, dc in NEIGHBORS4:
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < TILE_SIZE and 0 <= nc < TILE_SIZE and not mask[nr, nc]:
                    frontier.append((nr, nc))
        nr, nc = frontier[int(rng.integers(len(frontier)))]
        mask[nr, nc] = True
    return mask


def _random_pattern(
    rng: np.random.Generator,
    palette: list[int],
    forbidden: set[tuple[int, ...]],
    *,
    cell_count: int | None = None,
    asymmetric: bool = False,
) -> np.ndarray:
    for _ in range(300):
        cells = int(cell_count if cell_count is not None else rng.integers(2, 5))
        mask = _random_connected_mask(rng, cells)
        color_count = int(rng.integers(1, 3))
        colors = [int(x) for x in rng.choice(palette, size=color_count, replace=False)]
        tile = np.zeros((TILE_SIZE, TILE_SIZE), dtype=int)
        coords = np.argwhere(mask)
        if color_count == 1:
            tile[mask] = colors[0]
        else:
            assignments = rng.integers(0, color_count, size=len(coords))
            if len(set(map(int, assignments))) != color_count:
                continue
            for idx, (r, c) in enumerate(coords):
                tile[int(r), int(c)] = colors[int(assignments[idx])]
        key = _pattern_key(tile)
        if key in forbidden:
            continue
        if asymmetric and not _rotationally_asymmetric(tile):
            continue
        return tile
    raise RuntimeError("failed to generate a unique sprite tile")


def _marker_corners(rng: np.random.Generator, count: int) -> list[str]:
    corners = list(CORNER_TO_COORD)
    base = corners.copy()
    while len(base) < count:
        base.append(corners[int(rng.integers(0, 4))])
    rng.shuffle(base)
    return base


def _tile_origin(tr: int, tc: int) -> tuple[int, int]:
    step = TILE_SIZE + SEP
    return tr * step, tc * step


def _build_grid(
    tile_rows: int,
    tile_cols: int,
    template: np.ndarray,
    template_positions: list[tuple[int, int]],
    marker_assignments: dict[tuple[int, int], str],
    distractor_tiles: dict[tuple[int, int], np.ndarray],
) -> np.ndarray:
    h = tile_rows * TILE_SIZE + (tile_rows - 1) * SEP
    w = tile_cols * TILE_SIZE + (tile_cols - 1) * SEP
    grid = np.full((h, w), BLACK, dtype=int)

    for tr in range(1, tile_rows):
        grid[tr * (TILE_SIZE + SEP) - 1, :] = GREY
    for tc in range(1, tile_cols):
        grid[:, tc * (TILE_SIZE + SEP) - 1] = GREY

    for tr, tc in template_positions:
        r0, c0 = _tile_origin(tr, tc)
        grid[r0 : r0 + TILE_SIZE, c0 : c0 + TILE_SIZE] = template
    for (tr, tc), tile in distractor_tiles.items():
        r0, c0 = _tile_origin(tr, tc)
        grid[r0 : r0 + TILE_SIZE, c0 : c0 + TILE_SIZE] = tile
    for (tr, tc), corner in marker_assignments.items():
        r0, c0 = _tile_origin(tr, tc)
        rr, cc = CORNER_TO_COORD[corner]
        grid[r0 + rr, c0 + cc] = GREY
    return grid


def _semantic_check(grid: np.ndarray) -> None:
    parsed = _parse_grid(grid)
    template = parsed["template_tile"]
    assert _rotationally_asymmetric(template), "template must change under every non-zero quarter turn"
    assert len(parsed["template_positions"]) == 2, "template must appear exactly twice"
    assert len(parsed["markers"]) >= 4, "generation must expose all four rotation branches"
    marker_corners = [marker["corner"] for marker in parsed["markers"]]
    assert set(CORNER_TO_COORD).issubset(set(marker_corners)), "all four corner-coded rotations must appear"
    assert len(parsed["empties"]) >= 1, "need at least one untouched empty tile"

    sprite_keys = [sprite["key"] for sprite in parsed["sprites"]]
    template_key = parsed["template_key"]
    assert sum(key == template_key for key in sprite_keys) == 2

    rotated_template_keys = {_pattern_key(np.rot90(template, k)) for k in (1, 2, 3)}
    distractor_keys = {sprite["key"] for sprite in parsed["sprites"] if sprite["key"] != template_key}
    assert rotated_template_keys & distractor_keys, "need a distractor that matches a rotated template"

    expected = _expected_output(parsed)
    actual = main(grid)
    assert np.array_equal(actual, expected), "solver output must match independent semantic transform"

    for sprite in parsed["sprites"]:
        r0, r1, c0, c1 = sprite["bounds"]
        out_tile = actual[r0 : r1 + 1, c0 : c1 + 1]
        if sprite["key"] == template_key:
            assert np.all(out_tile == BLACK), "template sources must be erased"
        else:
            assert np.array_equal(out_tile, sprite["tile"]), "non-template sprite tiles must stay unchanged"
    for marker in parsed["markers"]:
        r0, r1, c0, c1 = marker["bounds"]
        out_tile = actual[r0 : r1 + 1, c0 : c1 + 1]
        want = np.rot90(template, CORNER_TO_ROT[marker["corner"]])
        assert np.array_equal(out_tile, want), "marker tiles must receive the correctly rotated template"


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GEN_ATTEMPTS):
        tile_rows = int(rng.integers(3, 6))
        tile_cols = int(rng.integers(3, 6))
        total_tiles = tile_rows * tile_cols

        max_markers = min(6, total_tiles - 5)
        marker_count = int(rng.integers(4, max_markers + 1))
        remaining = total_tiles - 2 - marker_count
        distractor_max = min(5, remaining - 1)
        if distractor_max < 2:
            continue
        distractor_count = int(rng.integers(2, distractor_max + 1))

        palette_size = int(rng.integers(3, 6))
        palette = [int(x) for x in rng.choice(NON_FIXED_COLORS, size=palette_size, replace=False)]

        forbidden: set[tuple[int, ...]] = set()
        template = _random_pattern(rng, palette, forbidden, asymmetric=True)
        forbidden.add(_pattern_key(template))

        distractors: list[np.ndarray] = []
        rotated_distractor = np.rot90(template, int(rng.integers(1, 4)))
        forbidden.add(_pattern_key(rotated_distractor))
        distractors.append(rotated_distractor)
        while len(distractors) < distractor_count:
            same_count = int(np.count_nonzero(template)) if len(distractors) == 1 else None
            tile = _random_pattern(rng, palette, forbidden, cell_count=same_count)
            forbidden.add(_pattern_key(tile))
            distractors.append(tile)

        positions = [(tr, tc) for tr in range(tile_rows) for tc in range(tile_cols)]
        random.shuffle(positions)
        template_positions = [positions.pop(), positions.pop()]

        marker_positions = [positions.pop() for _ in range(marker_count)]
        corners = _marker_corners(rng, marker_count)
        marker_assignments = {pos: corner for pos, corner in zip(marker_positions, corners)}

        distractor_positions = [positions.pop() for _ in range(distractor_count)]
        distractor_tiles = {pos: tile for pos, tile in zip(distractor_positions, distractors)}

        grid = _build_grid(
            tile_rows=tile_rows,
            tile_cols=tile_cols,
            template=template,
            template_positions=template_positions,
            marker_assignments=marker_assignments,
            distractor_tiles=distractor_tiles,
        )

        try:
            _semantic_check(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle-8 input")
