from __future__ import annotations

from collections import defaultdict
import random

import numpy as np


BLACK = 0
PALETTE = tuple(range(1, 10))
GEN_RETRIES = 400
MASK_RETRIES = 80
PLACEMENT_PAD = 1


def _alternating_pair(line: np.ndarray) -> tuple[int, int] | None:
    if line.size < 2:
        return None
    first = int(line[0])
    second = int(line[1])
    if first == second:
        return None
    for idx, value in enumerate(line):
        expected = first if idx % 2 == 0 else second
        if int(value) != expected:
            return None
    return first, second


def _find_mirror(grid: np.ndarray) -> tuple[str, int, int, int]:
    row_hits = []
    for r in range(grid.shape[0]):
        pair = _alternating_pair(grid[r, :])
        if pair is not None:
            row_hits.append(("row", r, pair[0], pair[1]))

    col_hits = []
    for c in range(grid.shape[1]):
        pair = _alternating_pair(grid[:, c])
        if pair is not None:
            col_hits.append(("col", c, pair[0], pair[1]))

    hits = row_hits + col_hits
    assert len(hits) == 1, "expected exactly one alternating mirror stripe"
    return hits[0]


def _neighbors4(r: int, c: int, height: int, width: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < height and 0 <= cc < width:
            yield rr, cc


def _find_objects(
    grid: np.ndarray,
    axis_kind: str,
    axis_idx: int,
    stripe_colors: set[int],
) -> list[dict[str, np.ndarray | int | str]]:
    h, w = grid.shape
    active = grid != BLACK
    if axis_kind == "row":
        active[axis_idx, :] = False
    else:
        active[:, axis_idx] = False

    seen = np.zeros_like(active, dtype=bool)
    objects: list[dict[str, np.ndarray | int | str]] = []

    for r in range(h):
        for c in range(w):
            if not active[r, c] or seen[r, c]:
                continue

            color = int(grid[r, c])
            assert color not in stripe_colors, "object colors must differ from stripe colors"

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            side = None

            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                if axis_kind == "row":
                    assert abs(rr - axis_idx) >= 2, "objects must not touch the mirror row"
                    here_side = "upper" if rr < axis_idx else "lower"
                else:
                    assert abs(cc - axis_idx) >= 2, "objects must not touch the mirror column"
                    here_side = "left" if cc < axis_idx else "right"
                if side is None:
                    side = here_side
                else:
                    assert side == here_side, "each object must stay on one side of the mirror"

                for nr, nc in _neighbors4(rr, cc, h, w):
                    if active[nr, nc] and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            objects.append(
                {
                    "color": color,
                    "coords": np.array(coords, dtype=int),
                    "side": side,
                }
            )

    return objects


def _reflect_coords(coords: np.ndarray, axis_kind: str, axis_idx: int) -> np.ndarray:
    reflected = coords.copy()
    if axis_kind == "row":
        reflected[:, 0] = 2 * axis_idx - reflected[:, 0]
    else:
        reflected[:, 1] = 2 * axis_idx - reflected[:, 1]
    return reflected


def _solve_grid(grid: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    axis_kind, axis_idx, overlap_color, stripe_other = _find_mirror(g)
    stripe_colors = {int(overlap_color), int(stripe_other)}

    objects = _find_objects(g, axis_kind, axis_idx, stripe_colors)
    assert objects, "expected at least one foreground object"
    sides = {str(obj["side"]) for obj in objects}
    if axis_kind == "row":
        assert sides == {"upper", "lower"}, "expected objects on both sides of the mirror row"
    else:
        assert sides == {"left", "right"}, "expected objects on both sides of the mirror column"

    original_mask = np.zeros_like(g, dtype=bool)
    for obj in objects:
        coords = np.asarray(obj["coords"], dtype=int)
        original_mask[coords[:, 0], coords[:, 1]] = True

    projected: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
    for obj in objects:
        color = int(obj["color"])
        reflected = _reflect_coords(np.asarray(obj["coords"], dtype=int), axis_kind, axis_idx)
        assert np.all((0 <= reflected[:, 0]) & (reflected[:, 0] < h))
        assert np.all((0 <= reflected[:, 1]) & (reflected[:, 1] < w))
        if axis_kind == "row":
            assert np.all(reflected[:, 0] != axis_idx)
        else:
            assert np.all(reflected[:, 1] != axis_idx)
        for rr, cc in reflected:
            projected[(int(rr), int(cc))].append(color)

    out = g.copy()
    collision_pixels = 0
    clean_copy_pixels = 0
    for (r, c), colors in projected.items():
        if original_mask[r, c] or len(colors) > 1:
            out[r, c] = int(overlap_color)
            collision_pixels += 1
        else:
            out[r, c] = int(colors[0])
            clean_copy_pixels += 1

    return out, {
        "collision_pixels": collision_pixels,
        "clean_copy_pixels": clean_copy_pixels,
        "object_count": len(objects),
        "axis_kind": axis_kind,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect every foreground object across the hidden alternating stripe and recolor overlap pixels with the stripe's leading color.

    Input structure:
    - The background is black.
    - Exactly one full row or one full column is a two-color alternating stripe; this stripe is the hidden mirror line.
    - Away from that stripe, the remaining non-black cells form monochrome 4-connected foreground objects on both sides, and those object colors differ from the stripe colors.

    Rule:
    - Identify the alternating full row or column as the reflection axis.
    - Reflect each foreground object across that axis and draw the reflected copy in the object's original color.
    - If a reflected pixel lands on an original object pixel, or if multiple reflected pixels land on the same cell, recolor that output cell to the stripe's first color instead.
    - Keep the original stripe and all original object pixels in place; only add the reflected copies on top.

    Key insight:
    - The alternating line is not an ordinary object but a mirror, and the reflected copy's color is controlled by whether the landing cell collides with existing structure.
    """
    return _solve_grid(grid)[0]


def _random_mask(rng: random.Random) -> np.ndarray:
    for _ in range(MASK_RETRIES):
        h = rng.randint(2, 4)
        w = rng.randint(2, 5)
        target = rng.randint(2, min(7, h * w))

        seed = (rng.randrange(h), rng.randrange(w))
        cells = {seed}
        frontier = {nb for nb in _neighbors4(seed[0], seed[1], h, w)}

        while len(cells) < target:
            if not frontier:
                break
            cell = rng.choice(tuple(frontier))
            frontier.remove(cell)
            if cell in cells:
                continue
            cells.add(cell)
            for nb in _neighbors4(cell[0], cell[1], h, w):
                if nb not in cells:
                    frontier.add(nb)

        if len(cells) != target:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        rows = np.flatnonzero(mask.any(axis=1))
        cols = np.flatnonzero(mask.any(axis=0))
        mask = mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        return mask

    raise RuntimeError("failed to generate a connected mask")


def _mask_coords(mask: np.ndarray, top: int, left: int) -> np.ndarray:
    return np.argwhere(mask).astype(int) + np.array([top, left], dtype=int)


def _mark_with_padding(occupied: np.ndarray, top: int, left: int, mask: np.ndarray, pad: int) -> None:
    h, w = mask.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(occupied.shape[0], top + h + pad)
    c1 = min(occupied.shape[1], left + w + pad)
    occupied[r0:r1, c0:c1] = True


def _can_place(occupied: np.ndarray, top: int, left: int, mask: np.ndarray, pad: int) -> bool:
    h, w = mask.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(occupied.shape[0], top + h + pad)
    c1 = min(occupied.shape[1], left + w + pad)
    return not occupied[r0:r1, c0:c1].any()


def _placement_positions(
    shape: tuple[int, int],
    axis_row: int,
    side: str,
    mask: np.ndarray,
    occupied: np.ndarray,
) -> list[tuple[int, int]]:
    h, w = shape
    mh, mw = mask.shape
    span = min(axis_row, h - 1 - axis_row)
    if side == "upper":
        row_min = axis_row - span
        row_max = axis_row - mh - 1
    else:
        row_min = axis_row + 2
        row_max = axis_row + span - mh + 1
    if row_min > row_max:
        return []

    positions = []
    for top in range(row_min, row_max + 1):
        for left in range(0, w - mw + 1):
            if _can_place(occupied, top, left, mask, PLACEMENT_PAD):
                positions.append((top, left))
    return positions


def _reflect_top_for_row(top: int, height: int, axis_row: int) -> int:
    return 2 * axis_row - (top + height - 1)


def _place_object(
    grid: np.ndarray,
    exact: np.ndarray,
    occupied: np.ndarray,
    mask: np.ndarray,
    top: int,
    left: int,
    color: int,
) -> np.ndarray:
    coords = _mask_coords(mask, top, left)
    grid[coords[:, 0], coords[:, 1]] = color
    exact[coords[:, 0], coords[:, 1]] = True
    _mark_with_padding(occupied, top, left, mask, PLACEMENT_PAD)
    return coords


def _collision_partner(
    rng: random.Random,
    shape: tuple[int, int],
    axis_row: int,
    occupied: np.ndarray,
    ref_mask: np.ndarray,
    side: str,
) -> tuple[np.ndarray, int, int] | None:
    best_partial: list[tuple[np.ndarray, int, int]] = []
    best_exact: list[tuple[np.ndarray, int, int]] = []

    for _ in range(20):
        mask = _random_mask(rng)
        positions = _placement_positions(shape, axis_row, side, mask, occupied)
        for top, left in positions:
            coords = _mask_coords(mask, top, left)
            overlap = int(np.count_nonzero(ref_mask[coords[:, 0], coords[:, 1]]))
            if overlap == 0:
                continue
            if overlap < min(int(mask.sum()), int(ref_mask.sum())):
                best_partial.append((mask, top, left))
            else:
                best_exact.append((mask, top, left))

    pool = best_partial if best_partial else best_exact
    if not pool:
        return None
    return rng.choice(pool)


def _isolated_object(
    rng: random.Random,
    shape: tuple[int, int],
    axis_row: int,
    exact: np.ndarray,
    occupied: np.ndarray,
) -> tuple[np.ndarray, int, int] | None:
    side_order = ["upper", "lower"]
    rng.shuffle(side_order)

    for _ in range(30):
        mask = _random_mask(rng)
        for side in side_order:
            positions = _placement_positions(shape, axis_row, side, mask, occupied)
            rng.shuffle(positions)
            for top, left in positions:
                coords = _mask_coords(mask, top, left)
                reflected = _reflect_coords(coords, "row", axis_row)
                if exact[reflected[:, 0], reflected[:, 1]].any():
                    continue
                return mask, top, left
    return None


def _make_candidate(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(14, 24)
    width = rng.randint(14, 24)
    axis_row = rng.randint(4, height - 5)
    span = min(axis_row, height - 1 - axis_row)
    if span < 3:
        return None

    stripe_first, stripe_second = rng.sample(PALETTE, 2)
    object_count = rng.randint(3, 5)
    object_colors = rng.sample([c for c in PALETTE if c not in {stripe_first, stripe_second}], object_count)

    grid = np.zeros((height, width), dtype=int)
    grid[axis_row, :] = np.array(
        [stripe_first if c % 2 == 0 else stripe_second for c in range(width)],
        dtype=int,
    )

    exact = np.zeros((height, width), dtype=bool)
    occupied = np.zeros((height, width), dtype=bool)

    side_a = rng.choice(["upper", "lower"])
    side_b = "lower" if side_a == "upper" else "upper"

    mask_a = _random_mask(rng)
    positions_a = _placement_positions(grid.shape, axis_row, side_a, mask_a, occupied)
    if not positions_a:
        return None
    top_a, left_a = rng.choice(positions_a)
    coords_a = _place_object(grid, exact, occupied, mask_a, top_a, left_a, object_colors[0])

    ref_a = _reflect_coords(coords_a, "row", axis_row)
    ref_a_mask = np.zeros_like(exact)
    ref_a_mask[ref_a[:, 0], ref_a[:, 1]] = True

    partner = _collision_partner(rng, grid.shape, axis_row, occupied, ref_a_mask, side_b)
    if partner is None:
        top_b = _reflect_top_for_row(top_a, mask_a.shape[0], axis_row)
        left_b = left_a
        if not _can_place(occupied, top_b, left_b, mask_a, PLACEMENT_PAD):
            return None
        mask_b = mask_a.copy()
    else:
        mask_b, top_b, left_b = partner
    _place_object(grid, exact, occupied, mask_b, top_b, left_b, object_colors[1])

    for color in object_colors[2:]:
        extra = _isolated_object(rng, grid.shape, axis_row, exact, occupied)
        if extra is None:
            return None
        mask_c, top_c, left_c = extra
        _place_object(grid, exact, occupied, mask_c, top_c, left_c, color)

    if rng.random() < 0.5:
        grid = np.ascontiguousarray(grid.T)

    try:
        out, info = _solve_grid(grid)
    except AssertionError:
        return None

    if int(info["collision_pixels"]) == 0 or int(info["clean_copy_pixels"]) == 0:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _make_candidate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-projection sample")
