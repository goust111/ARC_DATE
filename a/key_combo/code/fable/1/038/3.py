from __future__ import annotations

from importlib.machinery import SourceFileLoader

import numpy as np


_ARC = SourceFileLoader(
    "arc_utils_038_3",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()

Color = _ARC.Color
bbox = _ARC.bbox
components_mask = _ARC.components_mask
crop_bbox = _ARC.crop_bbox
paste = _ARC.paste


BLACK = Color.BLACK
GREY = Color.GREY
SPRITE_PALETTE = [c for c in Color.NOT_BLACK if c != GREY]
GEN_RETRIES = 400
_RNG = np.random.default_rng()


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((r + dr, c + dc))
    return out


def _component_mask(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in coords:
        mask[int(r), int(c)] = True
    return mask


def _touches_grey(coords: np.ndarray, grey_mask: np.ndarray) -> bool:
    h, w = grey_mask.shape
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        for nr, nc in _neighbors8(rr, cc):
            if 0 <= nr < h and 0 <= nc < w and grey_mask[nr, nc]:
                return True
    return False


def _extract_legend(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    colored_mask = (grid != BLACK) & (grid != GREY)
    comps = components_mask(colored_mask, connectivity=8)
    assert comps, "expected at least one colored component"

    grey_mask = (grid == GREY)
    detached = [coords for coords in comps if not _touches_grey(coords, grey_mask)]
    pool = detached if detached else comps
    max_area = max(int(coords.shape[0]) for coords in pool)
    largest = [coords for coords in pool if int(coords.shape[0]) == max_area]
    assert len(largest) == 1, "expected a unique legend component"

    legend_mask = _component_mask(grid.shape, largest[0])
    legend_bbox = bbox(legend_mask)
    assert legend_bbox is not None
    legend = crop_bbox(grid, legend_bbox)
    return legend, legend_mask


def _find_copy_candidates(
    grid: np.ndarray,
    sprite: np.ndarray,
    wall_mask: np.ndarray,
    residue_mask: np.ndarray,
) -> list[tuple[int, int]]:
    h, w = grid.shape
    sh, sw = sprite.shape
    sprite_cells = np.argwhere(sprite != BLACK)
    residue_count = int(residue_mask.sum())
    candidates: list[tuple[int, int]] = []

    for r0 in range(h - sh + 1):
        for c0 in range(w - sw + 1):
            matched = 0
            ok = True
            for sr, sc in sprite_cells:
                rr = r0 + int(sr)
                cc = c0 + int(sc)
                if not wall_mask[rr, cc]:
                    continue
                if int(grid[rr, cc]) != int(sprite[int(sr), int(sc)]):
                    ok = False
                    break
                matched += 1
            if ok and matched == residue_count:
                candidates.append((r0, c0))

    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the full hidden legend copy from the colored residue left on parallel grey walls.

    Input structure:
    - Black is the background.
    - Grey cells form several parallel one-cell-thick wall segments.
    - Away from the walls there is one complete multicolor legend sprite.
    - Additional non-grey colored cells appear only on wall cells; they are the
      visible intersection residue of another translated copy of the sprite.

    Rule:
    - Identify the complete legend sprite and treat every other colored wall cell
      as residue.
    - Find the unique in-bounds translation of the legend sprite whose non-black
      cells intersect the wall mask exactly at those residue cells with matching
      colors.
    - Draw the entire sprite at that translated position, overwriting black and
      grey cells there while leaving the original legend and unrelated walls fixed.

    Key insight:
    - The colored wall cells are not standalone markers; together they are the
      exact wall-cross-section of one unseen translated sprite, so matching that
      cross-section against the complete legend recovers the only valid offset.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sprite, legend_mask = _extract_legend(g)
    colored_mask = (g != BLACK) & (g != GREY)
    residue_mask = colored_mask & ~legend_mask
    assert residue_mask.any(), "expected at least one residue cell"

    wall_mask = (g == GREY) | residue_mask
    candidates = _find_copy_candidates(g, sprite, wall_mask, residue_mask)
    assert len(candidates) == 1, "expected residues to pin a unique translation"

    out = g.copy()
    paste(out, sprite, candidates[0][0], candidates[0][1], transparent=BLACK)
    return out


def _randint(rng: np.random.Generator, low: int, high: int) -> int:
    return int(rng.integers(int(low), int(high) + 1))


def _grow_connected_mask(
    rng: np.random.Generator,
    height: int,
    width: int,
    area: int,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    start = (_randint(rng, 0, height - 1), _randint(rng, 0, width - 1))
    mask[start] = True
    frontier = {
        nbr
        for nbr in _neighbors8(*start)
        if 0 <= nbr[0] < height and 0 <= nbr[1] < width
    }

    while int(mask.sum()) < area:
        choices = list(frontier)
        pick = choices[_randint(rng, 0, len(choices) - 1)]
        frontier.remove(pick)
        mask[pick] = True
        for nbr in _neighbors8(*pick):
            nr, nc = nbr
            if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                frontier.add(nbr)

    return mask


def _make_sprite(rng: np.random.Generator) -> np.ndarray:
    for _ in range(100):
        height = _randint(rng, 4, 7)
        width = _randint(rng, 4, 7)
        area_lo = max(7, (height * width) // 3)
        area_hi = min(height * width - 1, 15)
        if area_lo > area_hi:
            continue
        area = _randint(rng, area_lo, area_hi)
        mask = _grow_connected_mask(rng, height, width, area)
        if np.count_nonzero(mask.any(axis=1)) < 2 or np.count_nonzero(mask.any(axis=0)) < 2:
            continue

        color_count = _randint(rng, 2, min(4, area))
        palette = [int(c) for c in rng.choice(SPRITE_PALETTE, size=color_count, replace=False)]
        coords = np.argwhere(mask)
        rng.shuffle(coords)

        sprite = np.zeros((height, width), dtype=int)
        for idx, (r, c) in enumerate(coords):
            if idx < color_count:
                color = palette[idx]
            else:
                color = int(palette[_randint(rng, 0, color_count - 1)])
            sprite[int(r), int(c)] = color

        if len(set(map(int, np.unique(sprite))) - {BLACK}) >= 2:
            return sprite

    raise RuntimeError("failed to build sprite")


def _interval_with_length(
    rng: np.random.Generator,
    start_min: int,
    end_max: int,
    max_len: int,
) -> tuple[int, int] | None:
    if start_min > end_max:
        return None
    avail = end_max - start_min + 1
    if avail < 2:
        return None
    seg_len = _randint(rng, 2, min(max_len, avail))
    seg_start = _randint(rng, start_min, end_max - seg_len + 1)
    return seg_start, seg_start + seg_len - 1


def _paint_horizontal_walls(
    grid: np.ndarray,
    sprite: np.ndarray,
    copy_r: int,
    copy_c: int,
    hit_rows: list[int],
    rng: np.random.Generator,
) -> None:
    width = grid.shape[1]
    for rel_r in hit_rows:
        cols = np.flatnonzero(sprite[rel_r] != BLACK)
        left = max(0, copy_c + int(cols.min()) - _randint(rng, 1, 3))
        right = min(width - 1, copy_c + int(cols.max()) + _randint(rng, 1, 3))
        grid[copy_r + rel_r, left : right + 1] = GREY

        if rng.random() < 0.5:
            side_intervals = []
            if left >= 3:
                side_intervals.append((0, left - 2))
            if right <= width - 4:
                side_intervals.append((right + 2, width - 1))
            if side_intervals:
                s0, s1 = side_intervals[_randint(rng, 0, len(side_intervals) - 1)]
                extra = _interval_with_length(rng, s0, s1, 5)
                if extra is not None:
                    grid[copy_r + rel_r, extra[0] : extra[1] + 1] = GREY


def _paint_vertical_walls(
    grid: np.ndarray,
    sprite: np.ndarray,
    copy_r: int,
    copy_c: int,
    hit_cols: list[int],
    rng: np.random.Generator,
) -> None:
    height = grid.shape[0]
    for rel_c in hit_cols:
        rows = np.flatnonzero(sprite[:, rel_c] != BLACK)
        top = max(0, copy_r + int(rows.min()) - _randint(rng, 1, 3))
        bottom = min(height - 1, copy_r + int(rows.max()) + _randint(rng, 1, 3))
        grid[top : bottom + 1, copy_c + rel_c] = GREY

        if rng.random() < 0.5:
            side_intervals = []
            if top >= 3:
                side_intervals.append((0, top - 2))
            if bottom <= height - 4:
                side_intervals.append((bottom + 2, height - 1))
            if side_intervals:
                s0, s1 = side_intervals[_randint(rng, 0, len(side_intervals) - 1)]
                extra = _interval_with_length(rng, s0, s1, 5)
                if extra is not None:
                    grid[extra[0] : extra[1] + 1, copy_c + rel_c] = GREY


def _place_legend_horizontally(
    rng: np.random.Generator,
    grid_shape: tuple[int, int],
    sprite_shape: tuple[int, int],
    wall_rows: list[int],
) -> tuple[int, int] | None:
    height, width = grid_shape
    sh, sw = sprite_shape
    options: list[tuple[int, int]] = []
    top_high = min(wall_rows) - sh - 1
    if top_high >= 1:
        options.append((1, top_high))
    bottom_low = max(wall_rows) + 2
    bottom_high = height - sh - 1
    if bottom_low <= bottom_high:
        options.append((bottom_low, bottom_high))
    if not options:
        return None
    r_low, r_high = options[_randint(rng, 0, len(options) - 1)]
    return _randint(rng, r_low, r_high), _randint(rng, 1, width - sw - 1)


def _place_legend_vertically(
    rng: np.random.Generator,
    grid_shape: tuple[int, int],
    sprite_shape: tuple[int, int],
    wall_cols: list[int],
) -> tuple[int, int] | None:
    height, width = grid_shape
    sh, sw = sprite_shape
    options: list[tuple[int, int]] = []
    left_high = min(wall_cols) - sw - 1
    if left_high >= 1:
        options.append((1, left_high))
    right_low = max(wall_cols) + 2
    right_high = width - sw - 1
    if right_low <= right_high:
        options.append((right_low, right_high))
    if not options:
        return None
    c_low, c_high = options[_randint(rng, 0, len(options) - 1)]
    return _randint(rng, 1, height - sh - 1), _randint(rng, c_low, c_high)


def _overlay_residue(
    grid: np.ndarray,
    sprite: np.ndarray,
    copy_r: int,
    copy_c: int,
) -> np.ndarray:
    residue_mask = np.zeros_like(grid, dtype=bool)
    for sr, sc in np.argwhere(sprite != BLACK):
        rr = copy_r + int(sr)
        cc = copy_c + int(sc)
        if grid[rr, cc] == GREY:
            grid[rr, cc] = int(sprite[int(sr), int(sc)])
            residue_mask[rr, cc] = True
    return residue_mask


def _expected_output(grid: np.ndarray, sprite: np.ndarray, copy_r: int, copy_c: int) -> np.ndarray:
    out = grid.copy()
    paste(out, sprite, copy_r, copy_c, transparent=BLACK)
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    sprite = _make_sprite(rng)
    sh, sw = sprite.shape
    active_rows = [int(r) for r in np.flatnonzero((sprite != BLACK).any(axis=1))]
    active_cols = [int(c) for c in np.flatnonzero((sprite != BLACK).any(axis=0))]
    if len(active_rows) < 2 or len(active_cols) < 2:
        return None

    height = _randint(rng, max(16, 2 * sh + 4), 30)
    width = _randint(rng, max(16, 2 * sw + 4), 30)
    grid = np.zeros((height, width), dtype=int)
    orientation = "h" if rng.random() < 0.5 else "v"

    for _ in range(80):
        copy_r = _randint(rng, 1, height - sh - 1)
        copy_c = _randint(rng, 1, width - sw - 1)

        if orientation == "h":
            hit_count = _randint(rng, 2, min(4, len(active_rows)))
            hit_rows = sorted(int(x) for x in rng.choice(active_rows, size=hit_count, replace=False))
            wall_rows = [copy_r + r for r in hit_rows]
            legend_pos = _place_legend_horizontally(rng, grid.shape, sprite.shape, wall_rows)
            if legend_pos is None:
                continue
            grid.fill(BLACK)
            _paint_horizontal_walls(grid, sprite, copy_r, copy_c, hit_rows, rng)
        else:
            hit_count = _randint(rng, 2, min(4, len(active_cols)))
            hit_cols = sorted(int(x) for x in rng.choice(active_cols, size=hit_count, replace=False))
            wall_cols = [copy_c + c for c in hit_cols]
            legend_pos = _place_legend_vertically(rng, grid.shape, sprite.shape, wall_cols)
            if legend_pos is None:
                continue
            grid.fill(BLACK)
            _paint_vertical_walls(grid, sprite, copy_r, copy_c, hit_cols, rng)

        paste(grid, sprite, legend_pos[0], legend_pos[1], transparent=BLACK)
        residue_mask = _overlay_residue(grid, sprite, copy_r, copy_c)
        residue_count = int(residue_mask.sum())
        sprite_area = int(np.count_nonzero(sprite))
        if residue_count < 3 or residue_count >= sprite_area:
            continue

        if orientation == "h" and len(np.unique(np.argwhere(residue_mask)[:, 0])) < 2:
            continue
        if orientation == "v" and len(np.unique(np.argwhere(residue_mask)[:, 1])) < 2:
            continue

        wall_mask = (grid == GREY) | residue_mask
        candidates = _find_copy_candidates(grid, sprite, wall_mask, residue_mask)
        if candidates != [(copy_r, copy_c)]:
            continue

        expected = _expected_output(grid, sprite, copy_r, copy_c)
        try:
            solved = main(grid)
        except AssertionError:
            continue
        if not np.array_equal(solved, expected):
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-copy wall residue puzzle")
