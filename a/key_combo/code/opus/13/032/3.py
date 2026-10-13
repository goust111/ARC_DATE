from __future__ import annotations

import math
import random
from typing import Iterable

import numpy as np


BACKGROUND = 0
PALETTE = (1, 2, 3, 4)
KEY_COLOR = 4
SEPARATOR_COLOR = 5
LATIN_SIZE = 4
GEN_MAX_ATTEMPTS = 400


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(np.asarray(coords, dtype=int))
    return components


def _extract_patch(grid: np.ndarray, coords: np.ndarray, color: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    rr = coords[:, 0] - r0
    cc = coords[:, 1] - c0
    patch[rr, cc] = int(color)
    return patch, (r0, c0, r1, c1)


def _validate_partial_latin(grid: np.ndarray) -> None:
    for row in grid:
        seen = [int(v) for v in row if int(v) != BACKGROUND]
        assert len(seen) == len(set(seen)), "rows may not repeat a filled Latin color"
    for col in grid.T:
        seen = [int(v) for v in col if int(v) != BACKGROUND]
        assert len(seen) == len(set(seen)), "columns may not repeat a filled Latin color"


def _solve_latin_unique(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    _validate_partial_latin(out)
    solutions: list[np.ndarray] = []
    palette_set = set(PALETTE)

    def domain(r: int, c: int) -> list[int]:
        row_used = {int(v) for v in out[r, :] if int(v) != BACKGROUND}
        col_used = {int(v) for v in out[:, c] if int(v) != BACKGROUND}
        return [color for color in PALETTE if color not in row_used and color not in col_used]

    def search() -> None:
        if len(solutions) > 1:
            return
        blanks = np.argwhere(out == BACKGROUND)
        if len(blanks) == 0:
            if all({int(v) for v in row} == palette_set for row in out) and all(
                {int(v) for v in col} == palette_set for col in out.T
            ):
                solutions.append(out.copy())
            return

        choices: list[tuple[int, int, int, list[int]]] = []
        for r_raw, c_raw in blanks:
            r, c = int(r_raw), int(c_raw)
            values = domain(r, c)
            if not values:
                return
            choices.append((len(values), r, c, values))
        choices.sort(key=lambda item: (item[0], item[1], item[2]))
        _, r, c, values = choices[0]
        for value in values:
            out[r, c] = value
            search()
            out[r, c] = BACKGROUND

    search()
    assert len(solutions) == 1, "Latin clues must admit exactly one completion"
    return solutions[0]


def _find_separator_row(grid: np.ndarray) -> int:
    rows = [
        r
        for r in range(grid.shape[0])
        if np.all(grid[r, :] == SEPARATOR_COLOR)
    ]
    assert len(rows) == 1, "expected one full-width separator row"
    return rows[0]


def _find_latin_window(upper: np.ndarray) -> np.ndarray:
    h, w = upper.shape
    candidates: list[np.ndarray] = []
    for r0 in range(h - LATIN_SIZE + 1):
        for c0 in range(w - LATIN_SIZE + 1):
            patch = upper[r0 : r0 + LATIN_SIZE, c0 : c0 + LATIN_SIZE]
            values = set(map(int, np.unique(patch)))
            if not values.issubset({BACKGROUND, *PALETTE}):
                continue
            if np.count_nonzero(patch) < 8:
                continue
            if any(np.count_nonzero(patch[r, :]) == 0 for r in range(LATIN_SIZE)):
                continue
            if any(np.count_nonzero(patch[:, c]) == 0 for c in range(LATIN_SIZE)):
                continue
            if set(map(int, np.unique(patch))) - {BACKGROUND} != set(PALETTE):
                continue
            try:
                completed = _solve_latin_unique(patch)
            except AssertionError:
                continue
            candidates.append(completed)
    assert len(candidates) == 1, "expected one uniquely completable 4x4 Latin window"
    return candidates[0]


def _yellow_row_per_column(latin: np.ndarray) -> list[int]:
    rows: list[int] = []
    for c in range(LATIN_SIZE):
        matches = np.where(latin[:, c] == KEY_COLOR)[0]
        assert len(matches) == 1, "each column must contain yellow exactly once"
        rows.append(int(matches[0]))
    return rows


def _rotated_top_left(r0: int, c0: int, h: int, w: int, steps_cw: int) -> tuple[int, int]:
    steps = steps_cw % 4
    new_h, new_w = (h, w) if steps % 2 == 0 else (w, h)
    center_r = r0 + (h - 1) / 2.0
    center_c = c0 + (w - 1) / 2.0
    new_r0 = math.floor(center_r - (new_h - 1) / 2.0)
    new_c0 = math.floor(center_c - (new_w - 1) / 2.0)
    return int(new_r0), int(new_c0)


def _place_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    assert 0 <= r0 and 0 <= c0 and r0 + h <= grid.shape[0] and c0 + w <= grid.shape[1], "patch must fit"
    for rr, cc in np.argwhere(patch != BACKGROUND):
        tr = r0 + int(rr)
        tc = c0 + int(cc)
        assert grid[tr, tc] == BACKGROUND, "sprites may not overlap"
        grid[tr, tc] = int(patch[rr, cc])


def _rotate_component_in_place(grid: np.ndarray, coords: np.ndarray, color: int, steps_cw: int) -> np.ndarray:
    patch, (r0, c0, _r1, _c1) = _extract_patch(grid, coords, color)
    rotated = np.rot90(patch, k=(-steps_cw) % 4)
    nr0, nc0 = _rotated_top_left(r0, c0, patch.shape[0], patch.shape[1], steps_cw)

    out = grid.copy()
    out[coords[:, 0], coords[:, 1]] = BACKGROUND
    h, w = rotated.shape
    assert 0 <= nr0 and 0 <= nc0 and nr0 + h <= out.shape[0] and nc0 + w <= out.shape[1], "rotated sprite must fit"
    for rr, cc in np.argwhere(rotated != BACKGROUND):
        tr = nr0 + int(rr)
        tc = nc0 + int(cc)
        assert out[tr, tc] == BACKGROUND, "rotated sprites may not collide"
        out[tr, tc] = int(color)
    return out


def _lower_sprites(lower: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    sprites: list[tuple[int, int, np.ndarray]] = []
    for color in PALETTE:
        coords_list = _component_coords(lower == color)
        assert len(coords_list) == 1, "each palette color must form one sprite"
        coords = coords_list[0]
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert 3 <= h <= 4 and 3 <= w <= 4, "sprites must be 3x3 to 4x4"
        sprites.append((c0, r0, coords))
    sprites.sort()
    return sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the partial Latin square, read the yellow row in each column, and rotate the four lower sprites by those quarter-turn counts.

    Input structure:
    - A black background contains one full-width gray separator row that splits the grid into an upper panel and a lower panel.
    - The upper panel has exactly one 4x4 partially filled Latin-square window over colors {1,2,3,4}, where black cells are blanks and filled cells already satisfy row/column uniqueness.
    - The lower panel contains four non-overlapping monochrome sprites colored 1, 2, 3, and 4; each sprite is a single 4-connected component with a 3x3 to 4x4 tight bbox.

    Rule:
    - Find the unique completion of the 4x4 Latin square so every row and column contains colors {1,2,3,4} exactly once.
    - For each Latin column from left to right, locate the row index of yellow (color 4); that row index is the clockwise quarter-turn count for the correspondingly ordered lower sprite.
    - Rotate each lower sprite in place around its bbox center, using the upper-left cell when the center falls on a half-pixel, and output only the transformed lower panel.

    Key insight:
    - The Latin square is not the final object to render: its completed yellow positions encode four rotation parameters that must be transferred column-by-column to the lower sprites.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_r = _find_separator_row(g)
    latin = _find_latin_window(g[:sep_r, :])
    rotations = _yellow_row_per_column(latin)

    lower = g[sep_r + 1 :, :].copy()
    sprites = _lower_sprites(lower)
    out = lower.copy()
    for steps, (_c0, _r0, coords) in zip(rotations, sprites):
        color = int(out[coords[0, 0], coords[0, 1]])
        out = _rotate_component_in_place(out, coords, color, steps)
    return out


def _latin_square() -> np.ndarray:
    base = np.fromfunction(lambda r, c: (r + c) % LATIN_SIZE, (LATIN_SIZE, LATIN_SIZE), dtype=int)
    row_perm = np.array(random.sample(range(LATIN_SIZE), LATIN_SIZE), dtype=int)
    col_perm = np.array(random.sample(range(LATIN_SIZE), LATIN_SIZE), dtype=int)
    color_perm = np.array(random.sample(PALETTE, LATIN_SIZE), dtype=int)
    return color_perm[base[row_perm][:, col_perm]]


def _blank_partial_latin(full: np.ndarray) -> np.ndarray | None:
    yellow_cells = [(r, c) for r in range(LATIN_SIZE) for c in range(LATIN_SIZE) if int(full[r, c]) == KEY_COLOR]
    other_cells = [(r, c) for r in range(LATIN_SIZE) for c in range(LATIN_SIZE) if int(full[r, c]) != KEY_COLOR]
    blank_count = random.choice((6, 7))
    yellow_blank_count = random.choice((2, 3))
    for _ in range(80):
        blanks = set(random.sample(yellow_cells, yellow_blank_count))
        blanks.update(random.sample(other_cells, blank_count - yellow_blank_count))
        partial = full.copy()
        for r, c in blanks:
            partial[r, c] = BACKGROUND
        if any(np.count_nonzero(partial[r, :]) == 0 for r in range(LATIN_SIZE)):
            continue
        if any(np.count_nonzero(partial[:, c]) == 0 for c in range(LATIN_SIZE)):
            continue
        if set(map(int, np.unique(partial))) - {BACKGROUND} != set(PALETTE):
            continue
        try:
            solved = _solve_latin_unique(partial)
        except AssertionError:
            continue
        if not np.array_equal(solved, full):
            continue
        return partial
    return None


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _is_connected(mask: np.ndarray) -> bool:
    return len(_component_coords(mask)) == 1


def _rotationally_distinct(mask: np.ndarray) -> bool:
    return all(not np.array_equal(mask, np.rot90(mask, k=k)) for k in (1, 2, 3))


def _mask_orbit(mask: np.ndarray) -> tuple[bytes, ...]:
    variants = [np.rot90(mask, k=k) for k in range(4)]
    return tuple(sorted(variant.astype(np.uint8).tobytes() for variant in variants))


def _random_sprite_mask(size: int) -> np.ndarray | None:
    cells = [(r, c) for r in range(size) for c in range(size)]
    target_area = random.randint(size + 1, size * size - 1)
    start = random.choice(cells)
    chosen = {start}
    frontier = set()

    def extend_frontier(r: int, c: int) -> None:
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < size and 0 <= nc < size and (nr, nc) not in chosen:
                frontier.add((nr, nc))

    extend_frontier(*start)
    while len(chosen) < target_area and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        chosen.add(cell)
        extend_frontier(*cell)

    if len(chosen) != target_area:
        return None

    mask = np.zeros((size, size), dtype=bool)
    for r, c in chosen:
        mask[r, c] = True
    if not _touches_all_sides(mask):
        return None
    if not _is_connected(mask):
        return None
    if not _rotationally_distinct(mask):
        return None
    return mask


def _sample_sprite_masks() -> list[np.ndarray] | None:
    masks: list[np.ndarray] = []
    orbits: set[tuple[bytes, ...]] = set()
    for _ in range(4):
        for _ in range(120):
            size = random.choice((3, 4))
            mask = _random_sprite_mask(size)
            if mask is None:
                continue
            orbit = _mask_orbit(mask)
            if orbit in orbits:
                continue
            masks.append(mask)
            orbits.add(orbit)
            break
        else:
            return None
    return masks


def _paint_mask(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _layout_sprites(patches: Iterable[np.ndarray], panel_h: int, panel_w: int) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    patches = list(patches)
    widths = [patch.shape[1] for patch in patches]
    total_w = sum(widths)
    gap_choices = [1, 2, 3]
    gaps = [random.choice(gap_choices) for _ in range(len(patches) - 1)]
    side_pad_left = random.choice((0, 1, 2))
    side_pad_right = random.choice((0, 1, 2))
    needed_w = total_w + sum(gaps) + side_pad_left + side_pad_right
    if needed_w > panel_w:
        return None

    slack = panel_w - needed_w
    extra = [0] * (len(patches) + 1)
    for _ in range(slack):
        extra[random.randrange(len(extra))] += 1

    grid = np.zeros((panel_h, panel_w), dtype=int)
    positions: list[tuple[int, int]] = []
    col = side_pad_left + extra[0]
    for idx, patch in enumerate(patches):
        h, w = patch.shape
        max_r0 = panel_h - h
        if max_r0 < 0:
            return None
        r0 = random.randint(0, max_r0)
        _place_patch(grid, patch, r0, col)
        positions.append((r0, col))
        col += w
        if idx < len(patches) - 1:
            col += gaps[idx] + extra[idx + 1]
    assert col + side_pad_right + extra[-1] <= panel_w
    return grid, positions


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    latin_full = _latin_square()
    latin_partial = _blank_partial_latin(latin_full)
    if latin_partial is None:
        return None

    masks = _sample_sprite_masks()
    if masks is None:
        return None

    color_order = random.sample(PALETTE, len(PALETTE))
    output_patches = [_paint_mask(mask, color) for mask, color in zip(masks, color_order)]
    lower_h = random.randint(5, 8)
    lower_w = random.randint(18, 24)
    layout = _layout_sprites(output_patches, lower_h, lower_w)
    if layout is None:
        return None
    output_lower, positions = layout

    input_lower = np.zeros_like(output_lower)
    rotations = _yellow_row_per_column(latin_full)
    for steps, color, patch, (r0, c0) in zip(rotations, color_order, output_patches, positions):
        inverse_patch = np.rot90(patch, k=steps % 4)
        if inverse_patch.shape != patch.shape:
            return None
        _place_patch(input_lower, inverse_patch, r0, c0)

    upper_h = random.randint(4, 7)
    total_h = upper_h + 1 + lower_h
    width = lower_w
    grid = np.zeros((total_h, width), dtype=int)
    sep_r = upper_h
    grid[sep_r, :] = SEPARATOR_COLOR

    latin_r0 = random.randint(0, upper_h - LATIN_SIZE)
    latin_c0 = random.randint(0, width - LATIN_SIZE)
    grid[latin_r0 : latin_r0 + LATIN_SIZE, latin_c0 : latin_c0 + LATIN_SIZE] = latin_partial
    grid[sep_r + 1 :, :] = input_lower

    if len(_yellow_row_per_column(_find_latin_window(grid[:sep_r, :]))) != 4:
        return None
    if not np.array_equal(main(grid), output_lower):
        return None
    if np.array_equal(input_lower, output_lower):
        return None
    return grid, output_lower


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        built = _build_case()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid Latin-rotation ARC instance")
