from __future__ import annotations

from collections import Counter
import hashlib
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

DIR_UP = (-1, 0)
DIR_DOWN = (1, 0)
DIR_LEFT = (0, -1)
DIR_RIGHT = (0, 1)
CARDINALS = (DIR_UP, DIR_DOWN, DIR_LEFT, DIR_RIGHT)

BODY_OFFSETS = {
    DIR_UP: ((0, -1), (0, 1), (1, -1), (1, 0), (1, 1), (2, 0)),
    DIR_DOWN: ((0, -1), (0, 1), (-1, -1), (-1, 0), (-1, 1), (-2, 0)),
    DIR_LEFT: ((-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1), (0, 2)),
    DIR_RIGHT: ((-1, 0), (1, 0), (-1, -1), (0, -1), (1, -1), (0, -2)),
}

BODY_EXTENTS = {}
for _direction, _offsets in BODY_OFFSETS.items():
    all_cells = [(0, 0), *_offsets]
    BODY_EXTENTS[_direction] = (
        min(dr for dr, _dc in all_cells),
        max(dr for dr, _dc in all_cells),
        min(dc for _dr, dc in all_cells),
        max(dc for _dr, dc in all_cells),
    )

SOLID_COLORS = (BLUE, RED, GREEN)
MARKER_COLORS = (YELLOW, PINK, ORANGE, TEAL)
BODY_COLOR = MAROON
RECENT_OUTPUTS: set[str] = set()


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r - 1, c - 1),
        (r - 1, c),
        (r - 1, c + 1),
        (r, c - 1),
        (r, c + 1),
        (r + 1, c - 1),
        (r + 1, c),
        (r + 1, c + 1),
    ]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []
    height, width = mask.shape
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors8(rr, cc):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bands_from_separators(start: int, end: int, separators: list[int]) -> list[tuple[int, int]]:
    bands = []
    cur = int(start)
    for cut in separators:
        bands.append((cur, int(cut) - 1))
        cur = int(cut) + 1
    bands.append((cur, int(end)))
    return bands


def _locate_lattice(grid: np.ndarray) -> dict:
    height, width = grid.shape
    row_grey = np.count_nonzero(grid == GREY, axis=1)
    col_grey = np.count_nonzero(grid == GREY, axis=0)
    max_row_grey = int(row_grey.max())
    max_col_grey = int(col_grey.max())
    sep_rows = [r for r in range(height) if int(row_grey[r]) == max_row_grey and max_row_grey > 0]
    sep_cols = [c for c in range(width) if int(col_grey[c]) == max_col_grey and max_col_grey > 0]
    assert len(sep_rows) >= 3 and len(sep_cols) >= 3, "expected an internal separator lattice"

    row_gaps = [b - a for a, b in zip(sep_rows, sep_rows[1:])]
    col_gaps = [b - a for a, b in zip(sep_cols, sep_cols[1:])]
    assert len(set(row_gaps)) == 1 and len(set(col_gaps)) == 1, "separator spacing must be regular"
    cell_h = row_gaps[0] - 1
    cell_w = col_gaps[0] - 1
    assert cell_h >= 1 and cell_w >= 1

    top = sep_rows[0] - cell_h
    bottom = sep_rows[-1] + cell_h
    left = sep_cols[0] - cell_w
    right = sep_cols[-1] + cell_w
    assert 0 <= top < bottom < height and 0 <= left < right < width

    for r in sep_rows:
        assert np.all(grid[r, left : right + 1] == GREY)
    for c in sep_cols:
        assert np.all(grid[top : bottom + 1, c] == GREY)

    row_bands = _bands_from_separators(top, bottom, sep_rows)
    col_bands = _bands_from_separators(left, right, sep_cols)
    assert all(r1 - r0 + 1 == cell_h for r0, r1 in row_bands)
    assert all(c1 - c0 + 1 == cell_w for c0, c1 in col_bands)

    return {
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
        "cell_h": cell_h,
        "cell_w": cell_w,
        "row_bands": row_bands,
        "col_bands": col_bands,
    }


def _extract_solid_block(grid: np.ndarray, lattice: dict) -> tuple[np.ndarray, dict]:
    solid: dict[tuple[int, int], int] = {}
    for ri, (r0, r1) in enumerate(lattice["row_bands"]):
        for ci, (c0, c1) in enumerate(lattice["col_bands"]):
            cell = grid[r0 : r1 + 1, c0 : c1 + 1]
            vals = set(map(int, cell.flat))
            if len(vals) == 1:
                color = next(iter(vals))
                if color not in (BLACK, GREY):
                    solid[(ri, ci)] = color

    assert solid, "expected at least one solid cell"
    solid_rows = sorted({ri for ri, _ci in solid})
    solid_cols = sorted({ci for _ri, ci in solid})
    row0, row1 = solid_rows[0], solid_rows[-1]
    col0, col1 = solid_cols[0], solid_cols[-1]
    for ri in range(row0, row1 + 1):
        for ci in range(col0, col1 + 1):
            assert (ri, ci) in solid, "solid cells must form one rectangle"
    assert len(solid) == (row1 - row0 + 1) * (col1 - col0 + 1), "no extra solid cells allowed"

    out = np.empty((row1 - row0 + 1, col1 - col0 + 1), dtype=int)
    for ri in range(row0, row1 + 1):
        for ci in range(col0, col1 + 1):
            out[ri - row0, ci - col0] = solid[(ri, ci)]

    meta = {
        "solid_row0": row0,
        "solid_col0": col0,
        "solid_h": out.shape[0],
        "solid_w": out.shape[1],
    }
    return out, meta


def _direction_from_centroid(body: np.ndarray, marker: tuple[int, int]) -> tuple[int, int]:
    my, mx = marker
    cy = float(body[:, 0].mean())
    cx = float(body[:, 1].mean())
    dy = cy - my
    dx = cx - mx
    assert abs(abs(dy) - abs(dx)) > 1e-6 or min(abs(dy), abs(dx)) < 1e-6

    if abs(dy) > abs(dx):
        assert abs(dx) < 1e-6
        return DIR_UP if dy > 0 else DIR_DOWN
    assert abs(dy) < 1e-6
    return DIR_LEFT if dx > 0 else DIR_RIGHT


def _find_band_index(value: int, bands: list[tuple[int, int]]) -> int:
    hits = [i for i, (a, b) in enumerate(bands) if a <= value <= b]
    assert len(hits) == 1, "ray must pass through exactly one cell band"
    return hits[0]


def _gap_is_clear(grid: np.ndarray, lattice: dict, marker: tuple[int, int], direction: tuple[int, int]) -> None:
    my, mx = marker
    if direction == DIR_RIGHT:
        assert np.all(grid[my, mx + 1 : lattice["left"]] == BLACK)
    elif direction == DIR_LEFT:
        assert np.all(grid[my, lattice["right"] + 1 : mx] == BLACK)
    elif direction == DIR_DOWN:
        assert np.all(grid[my + 1 : lattice["top"], mx] == BLACK)
    else:
        assert np.all(grid[lattice["bottom"] + 1 : my, mx] == BLACK)


def _decode_sprites(grid: np.ndarray, lattice: dict) -> list[dict]:
    outside = np.ones(grid.shape, dtype=bool)
    outside[lattice["top"] : lattice["bottom"] + 1, lattice["left"] : lattice["right"] + 1] = False
    mask = (grid != BLACK) & outside
    comps = _components(mask)
    assert 2 <= len(comps) <= 4, "expected 2-4 sprites outside the lattice"

    sprites = []
    for coords in comps:
        colors = [int(grid[r, c]) for r, c in coords]
        counts = Counter(colors)
        marker_candidates = [color for color, count in counts.items() if count == 1]
        body_candidates = [color for color, count in counts.items() if 5 <= count <= 7]
        assert len(marker_candidates) == 1 and len(body_candidates) == 1
        marker_color = marker_candidates[0]
        body_color = body_candidates[0]
        assert marker_color != body_color

        marker_pos = tuple(map(int, coords[[i for i, color in enumerate(colors) if color == marker_color][0]]))
        body = np.asarray([xy for xy, color in zip(coords, colors) if color == body_color], dtype=int)
        direction = _direction_from_centroid(body, marker_pos)
        _gap_is_clear(grid, lattice, marker_pos, direction)
        sprites.append(
            {
                "marker_color": marker_color,
                "marker": marker_pos,
                "direction": direction,
            }
        )

    return sprites


def _apply_rays(extracted: np.ndarray, lattice: dict, solid_meta: dict, sprites: list[dict]) -> np.ndarray:
    out = extracted.copy()
    solid_row0 = solid_meta["solid_row0"]
    solid_col0 = solid_meta["solid_col0"]
    solid_h = solid_meta["solid_h"]
    solid_w = solid_meta["solid_w"]
    targeted: set[tuple[int, int]] = set()

    for sprite in sprites:
        my, mx = sprite["marker"]
        marker_color = sprite["marker_color"]
        direction = sprite["direction"]
        if direction in (DIR_LEFT, DIR_RIGHT):
            row_idx = _find_band_index(my, lattice["row_bands"])
            assert solid_row0 <= row_idx < solid_row0 + solid_h
            rr = row_idx - solid_row0
            cc = 0 if direction == DIR_RIGHT else solid_w - 1
        else:
            col_idx = _find_band_index(mx, lattice["col_bands"])
            assert solid_col0 <= col_idx < solid_col0 + solid_w
            rr = 0 if direction == DIR_DOWN else solid_h - 1
            cc = col_idx - solid_col0
        assert (rr, cc) not in targeted, "sprites must hit distinct reduced-grid cells"
        targeted.add((rr, cc))
        out[rr, cc] = marker_color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the solid-cell subgrid hidden in a separator lattice, then recolor the ray-hit edge cells using the surrounding directional sprites.

    Input structure:
    - A black background contains one internal grey separator lattice that partitions a rectangular area into equal cells.
    - Exactly one contiguous rectangular block of lattice cells is solid monochrome; other cells are multicolor distractor patterns.
    - Outside the lattice, 2-4 small 8-connected sprites sit in the black margins; each sprite has one repeated body color and one singleton marker color.

    Rule:
    - Recover the lattice geometry from the grey separator rows and columns, then keep only the monochrome non-black, non-grey cells and crop them to their occupied rectangle to form the reduced color grid.
    - For each outside sprite, compute the centroid of its body-color pixels relative to the marker pixel; that centroid points backward, so the ray direction is the opposite cardinal direction.
    - Map the marker's row or column to the corresponding lattice band. A horizontal ray recolors the first reduced-grid cell in that row from the entry side; a vertical ray does the analogous operation for the corresponding reduced-grid column.

    Key insight:
    - The separator lattice is only an intermediate coordinate system: you must first extract the solid-cell rectangle, then use the sprite body bias to decide which reduced-grid edge cell each ray strikes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    lattice = _locate_lattice(g)
    extracted, solid_meta = _extract_solid_block(g, lattice)
    sprites = _decode_sprites(g, lattice)
    return _apply_rays(extracted, lattice, solid_meta, sprites)


def _pattern_cell(cell_h: int, cell_w: int, ri: int, ci: int) -> np.ndarray:
    palette = np.asarray(SOLID_COLORS, dtype=int)
    rr, cc = np.indices((cell_h, cell_w))
    pattern = (rr + 2 * cc + ri + 2 * ci) % len(palette)
    cell = palette[pattern]
    assert len(set(map(int, cell.flat))) >= 2
    return cell


def _place_sprite(
    grid: np.ndarray,
    lattice: dict,
    direction: tuple[int, int],
    target_row: int | None,
    target_col: int | None,
    marker_color: int,
    top_margin: int,
    bottom_margin: int,
    left_margin: int,
    right_margin: int,
    rng: random.Random,
) -> dict:
    row_bands = lattice["row_bands"]
    col_bands = lattice["col_bands"]
    min_dr, max_dr, min_dc, max_dc = BODY_EXTENTS[direction]

    if direction == DIR_RIGHT:
        band = row_bands[target_row]
        marker_r = rng.randint(band[0], band[1])
        max_gap = left_margin + min_dc - 1
        gap = rng.randint(1, max_gap)
        marker_c = lattice["left"] - gap - 1
    elif direction == DIR_LEFT:
        band = row_bands[target_row]
        marker_r = rng.randint(band[0], band[1])
        max_gap = right_margin - max_dc - 1
        gap = rng.randint(1, max_gap)
        marker_c = lattice["right"] + gap + 1
    elif direction == DIR_DOWN:
        band = col_bands[target_col]
        marker_c = rng.randint(band[0], band[1])
        max_gap = top_margin + min_dr - 1
        gap = rng.randint(1, max_gap)
        marker_r = lattice["top"] - gap - 1
    else:
        band = col_bands[target_col]
        marker_c = rng.randint(band[0], band[1])
        max_gap = bottom_margin - max_dr - 1
        gap = rng.randint(1, max_gap)
        marker_r = lattice["bottom"] + gap + 1

    coords = [(marker_r, marker_c, marker_color)]
    for dr, dc in BODY_OFFSETS[direction]:
        coords.append((marker_r + dr, marker_c + dc, BODY_COLOR))

    for r, c, color in coords:
        assert 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]
        assert grid[r, c] == BLACK
        grid[r, c] = color

    return {
        "direction": direction,
        "marker": (marker_r, marker_c),
        "marker_color": marker_color,
    }


def _build_candidate() -> tuple[np.ndarray, np.ndarray] | None:
    rng = random.Random()

    while True:
        lattice_rows = rng.randint(4, 6)
        lattice_cols = rng.randint(4, 6)
        cell_h = rng.randint(2, 3)
        cell_w = rng.randint(2, 3)
        lattice_h = lattice_rows * cell_h + (lattice_rows - 1)
        lattice_w = lattice_cols * cell_w + (lattice_cols - 1)
        if lattice_h <= 22 and lattice_w <= 22:
            break

    solid_h = rng.randint(3, min(4, lattice_rows))
    solid_w = rng.randint(3, min(4, lattice_cols))
    solid_row0 = rng.randint(0, lattice_rows - solid_h)
    solid_col0 = rng.randint(0, lattice_cols - solid_w)

    top_margin = 4 + rng.randint(0, 30 - lattice_h - 8)
    bottom_margin = 30 - lattice_h - top_margin
    left_margin = 4 + rng.randint(0, 30 - lattice_w - 8)
    right_margin = 30 - lattice_w - left_margin
    height = top_margin + lattice_h + bottom_margin
    width = left_margin + lattice_w + right_margin

    grid = np.zeros((height, width), dtype=int)
    lattice = {
        "top": top_margin,
        "bottom": top_margin + lattice_h - 1,
        "left": left_margin,
        "right": left_margin + lattice_w - 1,
        "cell_h": cell_h,
        "cell_w": cell_w,
        "row_bands": [],
        "col_bands": [],
    }

    sep_rows = [top_margin + cell_h + i * (cell_h + 1) for i in range(lattice_rows - 1)]
    sep_cols = [left_margin + cell_w + i * (cell_w + 1) for i in range(lattice_cols - 1)]
    lattice["row_bands"] = _bands_from_separators(lattice["top"], lattice["bottom"], sep_rows)
    lattice["col_bands"] = _bands_from_separators(lattice["left"], lattice["right"], sep_cols)

    grid[sep_rows, lattice["left"] : lattice["right"] + 1] = GREY
    grid[lattice["top"] : lattice["bottom"] + 1, sep_cols] = GREY

    solid = np.asarray(rng.choices(SOLID_COLORS, k=solid_h * solid_w), dtype=int).reshape(solid_h, solid_w)
    if len(set(map(int, solid.flat))) == 1:
        alt = [c for c in SOLID_COLORS if c != int(solid[0, 0])]
        solid[rng.randrange(solid_h), rng.randrange(solid_w)] = rng.choice(alt)

    for ri, (r0, r1) in enumerate(lattice["row_bands"]):
        for ci, (c0, c1) in enumerate(lattice["col_bands"]):
            if solid_row0 <= ri < solid_row0 + solid_h and solid_col0 <= ci < solid_col0 + solid_w:
                color = int(solid[ri - solid_row0, ci - solid_col0])
                grid[r0 : r1 + 1, c0 : c1 + 1] = color
            else:
                grid[r0 : r1 + 1, c0 : c1 + 1] = _pattern_cell(cell_h, cell_w, ri, ci)

    sprite_count = rng.randint(2, 4)
    while True:
        directions = rng.sample(list(CARDINALS), sprite_count)
        if sprite_count >= 3:
            break
        if any(d in (DIR_LEFT, DIR_RIGHT) for d in directions) and any(d in (DIR_UP, DIR_DOWN) for d in directions):
            break

    marker_colors = rng.sample(list(MARKER_COLORS), sprite_count)
    used_targets: set[tuple[int, int]] = set()
    row_choices = list(range(solid_h))
    col_choices = list(range(solid_w))
    rng.shuffle(row_choices)
    rng.shuffle(col_choices)
    row_ptr = 0
    col_ptr = 0

    sprites = []
    expected = solid.copy()
    for direction, marker_color in zip(directions, marker_colors):
        if direction in (DIR_LEFT, DIR_RIGHT):
            if row_ptr >= len(row_choices):
                return None
            rr = row_choices[row_ptr]
            row_ptr += 1
            cc = 0 if direction == DIR_RIGHT else solid_w - 1
            if (rr, cc) in used_targets:
                return None
            used_targets.add((rr, cc))
            expected[rr, cc] = marker_color
            sprite = _place_sprite(
                grid,
                lattice,
                direction,
                solid_row0 + rr,
                None,
                marker_color,
                top_margin,
                bottom_margin,
                left_margin,
                right_margin,
                rng,
            )
        else:
            if col_ptr >= len(col_choices):
                return None
            cc = col_choices[col_ptr]
            col_ptr += 1
            rr = 0 if direction == DIR_DOWN else solid_h - 1
            if (rr, cc) in used_targets:
                return None
            used_targets.add((rr, cc))
            expected[rr, cc] = marker_color
            sprite = _place_sprite(
                grid,
                lattice,
                direction,
                None,
                solid_col0 + cc,
                marker_color,
                top_margin,
                bottom_margin,
                left_margin,
                right_margin,
                rng,
            )
        sprites.append(sprite)

    lattice2 = _locate_lattice(grid)
    extracted, meta = _extract_solid_block(grid, lattice2)
    if not np.array_equal(extracted, solid):
        return None
    decoded_sprites = _decode_sprites(grid, lattice2)
    out = _apply_rays(extracted, lattice2, meta, decoded_sprites)
    if not np.array_equal(out, expected):
        return None
    if int(np.sum(out != solid)) != sprite_count:
        return None
    if len(set(map(int, out.flat))) < 3:
        return None

    out_sig = hashlib.md5(out.tobytes() + str(out.shape).encode()).hexdigest()
    if out_sig in RECENT_OUTPUTS:
        return None
    RECENT_OUTPUTS.add(out_sig)
    if len(RECENT_OUTPUTS) > 512:
        RECENT_OUTPUTS.clear()
        RECENT_OUTPUTS.add(out_sig)

    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(400):
        built = _build_candidate()
        if built is None:
            continue
        grid, expected = built
        try:
            solved = main(grid)
        except AssertionError:
            continue
        if np.array_equal(solved, expected):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
