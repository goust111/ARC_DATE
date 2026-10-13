from __future__ import annotations

import random

import numpy as np


BG = 0
NON_BG_COLORS = tuple(range(1, 10))
QUADRANTS = ("NW", "NE", "SW", "SE")
PALETTE_CORNERS = ("TL", "TR", "BL", "BR")
GENERATION_ATTEMPTS = 400

_GEN_COUNTER = 0


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _corner_cells(height: int, width: int, corner: str) -> list[tuple[int, int]]:
    if corner == "TL":
        r0, c0 = 0, 0
    elif corner == "TR":
        r0, c0 = 0, width - 2
    elif corner == "BL":
        r0, c0 = height - 2, 0
    elif corner == "BR":
        r0, c0 = height - 2, width - 2
    else:
        raise ValueError(f"unknown corner {corner}")
    return [(r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1)]


def _find_palette(g: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    height, width = map(int, g.shape)
    candidates: list[tuple[np.ndarray, dict[str, int]]] = []
    for corner in PALETTE_CORNERS:
        cells = _corner_cells(height, width, corner)
        vals = [int(g[r, c]) for r, c in cells]
        if all(v != BG for v in vals) and len(set(vals)) == 4:
            mask = np.zeros(g.shape, dtype=bool)
            for r, c in cells:
                mask[r, c] = True
            candidates.append(
                (
                    mask,
                    {
                        "NW": vals[0],
                        "NE": vals[1],
                        "SW": vals[2],
                        "SE": vals[3],
                    },
                )
            )
    assert len(candidates) == 1, "expected one distinct 2x2 palette in a grid corner"
    return candidates[0]


def _rotate_cell(
    r: int,
    c: int,
    center_r2: int,
    center_c2: int,
    turns: int,
) -> tuple[int, int]:
    rr2 = 2 * int(r)
    cc2 = 2 * int(c)
    for _ in range(int(turns) % 4):
        dr2 = rr2 - center_r2
        dc2 = cc2 - center_c2
        rr2 = center_r2 + dc2
        cc2 = center_c2 - dr2
    assert rr2 % 2 == 0 and cc2 % 2 == 0, "rotation must land on cells"
    return rr2 // 2, cc2 // 2


def _quadrant(r: int, c: int, center_r2: int, center_c2: int) -> str:
    rr2 = 2 * int(r)
    cc2 = 2 * int(c)
    assert rr2 != center_r2 and cc2 != center_c2, "repair cells must lie in a quadrant"
    if rr2 < center_r2:
        return "NW" if cc2 < center_c2 else "NE"
    return "SW" if cc2 < center_c2 else "SE"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the broken quadrant of a fourfold rotational figure using three-to-one voting and palette-colored fills.

    Input structure:
    - The background is black, and a single-color foreground figure sits away from the corners.
    - One grid corner contains a distinct 2x2 palette whose top-left, top-right,
      bottom-left, and bottom-right cells name the figure's NW, NE, SW, and SE quadrants.
    - After ignoring the palette, the foreground figure has a square bounding box
      and is almost fourfold rotationally symmetric, with errors only in one quadrant.

    Rule:
    - Use the foreground figure's square bounding box to locate its rotation center.
    - For each cell in that box, compare the three cells reached by 90, 180, and
      270 degree rotations around the center; their majority foreground/background
      value is the true value for the current cell.
    - Clear cells that are extra foreground, and fill missing foreground cells
      with the palette color for the missing cell's own quadrant.
    - Leave the 2x2 palette and all already-correct cells unchanged.

    Key insight:
    - The rotational majority decides where repairs happen, while the corner
      palette independently supplies the signature color for missing cells.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    palette_mask, quadrant_colors = _find_palette(g)

    body = g.copy()
    body[palette_mask] = BG
    body_colors = sorted(set(map(int, body.flatten())) - {BG})
    assert len(body_colors) == 1, "expected exactly one foreground figure color outside the palette"
    figure_color = int(body_colors[0])

    fig_box = _bbox(body == figure_color)
    assert fig_box is not None, "expected a foreground figure"
    r0, c0, r1, c1 = fig_box
    assert r1 - r0 == c1 - c0, "figure bbox must be square for quarter-turn voting"
    center_r2 = r0 + r1
    center_c2 = c0 + c1

    out = g.copy()
    height, width = map(int, g.shape)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            rotated = [
                _rotate_cell(r, c, center_r2, center_c2, turns)
                for turns in (1, 2, 3)
            ]
            assert all(0 <= rr < height and 0 <= cc < width for rr, cc in rotated)

            votes = sum(int(g[rr, cc]) == figure_color for rr, cc in rotated)
            truth_is_foreground = votes >= 2
            current_is_foreground = int(g[r, c]) == figure_color

            if truth_is_foreground == current_is_foreground:
                continue
            if truth_is_foreground:
                out[r, c] = quadrant_colors[_quadrant(r, c, center_r2, center_c2)]
            else:
                out[r, c] = BG

    return out


def _local_orbit(r: int, c: int, size: int) -> tuple[tuple[int, int], ...]:
    center2 = size - 1
    cells = {
        _rotate_cell(int(r), int(c), center2, center2, turns)
        for turns in range(4)
    }
    return tuple(sorted(cells))


def _has_neighbor(mask: np.ndarray, cells: tuple[tuple[int, int], ...]) -> bool:
    height, width = map(int, mask.shape)
    for r, c in cells:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < height and 0 <= cc < width and bool(mask[rr, cc]):
                return True
    return False


def _make_symmetric_mask(size: int) -> np.ndarray:
    mid = size // 2
    mask = np.zeros((size, size), dtype=bool)
    mask[mid, :] = True
    mask[:, mid] = True

    min_target = min(size * size, 2 * size + 8)
    max_target = max(min_target, int(size * size * random.uniform(0.32, 0.48)))
    target_area = random.randint(min_target, max_target)

    for _ in range(size * size * 8):
        if int(mask.sum()) >= target_area:
            break

        candidates: list[tuple[int, int, tuple[tuple[int, int], ...]]] = []
        for r in range(size):
            for c in range(size):
                if r == mid or c == mid or bool(mask[r, c]):
                    continue
                orbit = _local_orbit(r, c, size)
                if any(bool(mask[rr, cc]) for rr, cc in orbit):
                    continue
                if _has_neighbor(mask, orbit):
                    candidates.append((r, c, orbit))

        if not candidates:
            break

        r, c, orbit = random.choice(candidates)
        for rr, cc in orbit:
            mask[rr, cc] = True

    return mask


def _local_quadrant_cells(size: int, quadrant: str) -> list[tuple[int, int]]:
    mid = size // 2
    if quadrant == "NW":
        rows = range(0, mid)
        cols = range(0, mid)
    elif quadrant == "NE":
        rows = range(0, mid)
        cols = range(mid + 1, size)
    elif quadrant == "SW":
        rows = range(mid + 1, size)
        cols = range(0, mid)
    elif quadrant == "SE":
        rows = range(mid + 1, size)
        cols = range(mid + 1, size)
    else:
        raise ValueError(f"unknown quadrant {quadrant}")
    return [(r, c) for r in rows for c in cols]


def _choose_layout() -> tuple[int, int, int, int, int, str]:
    height = random.randint(21, 30)
    width = random.randint(21, 30)
    max_size = min(17, height - 8, width - 8)
    odd_sizes = [s for s in range(9, max_size + 1, 2)]
    if not odd_sizes:
        raise ValueError("grid too small for figure")
    size = random.choice(odd_sizes)
    margin = 4
    r0 = random.randint(margin, height - size - margin)
    c0 = random.randint(margin, width - size - margin)
    palette_corner = random.choice(PALETTE_CORNERS)
    return height, width, size, r0, c0, palette_corner


def _paint_palette(
    grid: np.ndarray,
    corner: str,
    colors: list[int],
) -> dict[str, int]:
    cells = _corner_cells(*grid.shape, corner)
    mapping = {"NW": colors[0], "NE": colors[1], "SW": colors[2], "SE": colors[3]}
    for (r, c), color in zip(cells, colors):
        grid[r, c] = int(color)
    return mapping


def _render_attempt(forced_quadrant: str) -> np.ndarray | None:
    height, width, size, top, left, palette_corner = _choose_layout()
    mask = _make_symmetric_mask(size)

    quadrant_cells = _local_quadrant_cells(size, forced_quadrant)
    present = [cell for cell in quadrant_cells if bool(mask[cell])]
    absent = [cell for cell in quadrant_cells if not bool(mask[cell])]
    if not present or not absent:
        return None

    missing_count = random.randint(1, min(3, len(present)))
    extra_count = random.randint(1, min(3, len(absent)))
    missing = random.sample(present, missing_count)
    extras = random.sample(absent, extra_count)

    colors = random.sample(NON_BG_COLORS, 5)
    figure_color = colors[0]
    palette_colors = colors[1:]

    truth = np.full((height, width), BG, dtype=int)
    quadrant_color_map = _paint_palette(truth, palette_corner, palette_colors)
    rr, cc = np.where(mask)
    truth[top + rr, left + cc] = figure_color

    inp = truth.copy()
    for r, c in missing:
        inp[top + r, left + c] = BG
    for r, c in extras:
        inp[top + r, left + c] = figure_color

    expected = inp.copy()
    for r, c in missing:
        expected[top + r, left + c] = quadrant_color_map[forced_quadrant]
    for r, c in extras:
        expected[top + r, left + c] = BG

    if _semantically_valid(
        inp,
        expected,
        truth,
        figure_color,
        top,
        left,
        size,
        forced_quadrant,
        missing,
        extras,
    ):
        return inp
    return None


def _semantically_valid(
    inp: np.ndarray,
    expected: np.ndarray,
    truth: np.ndarray,
    figure_color: int,
    top: int,
    left: int,
    size: int,
    quadrant: str,
    missing: list[tuple[int, int]],
    extras: list[tuple[int, int]],
) -> bool:
    if not missing or not extras:
        return False

    try:
        got = main(inp)
    except AssertionError:
        return False
    if not np.array_equal(got, expected):
        return False

    palette_mask, quadrant_colors = _find_palette(inp)
    if not np.array_equal(inp[palette_mask], expected[palette_mask]):
        return False

    center_r2 = 2 * top + size - 1
    center_c2 = 2 * left + size - 1
    truth_mask = truth == int(figure_color)
    input_mask = inp == int(figure_color)
    changed = np.argwhere(truth_mask != input_mask)
    if int(changed.shape[0]) != len(missing) + len(extras):
        return False
    changed_quadrants = {
        _quadrant(int(r), int(c), center_r2, center_c2)
        for r, c in changed
    }
    if changed_quadrants != {quadrant}:
        return False

    fill_color = quadrant_colors[quadrant]
    for r, c in missing:
        ar, ac = top + r, left + c
        if int(inp[ar, ac]) != BG or int(truth[ar, ac]) != int(figure_color):
            return False
        if int(expected[ar, ac]) != int(fill_color):
            return False
    for r, c in extras:
        ar, ac = top + r, left + c
        if int(inp[ar, ac]) != int(figure_color) or int(truth[ar, ac]) != BG:
            return False
        if int(expected[ar, ac]) != BG:
            return False

    mono_repaired = truth.copy()
    if np.array_equal(expected, mono_repaired):
        return False
    if np.array_equal(inp, expected):
        return False

    return True


def generate_input() -> np.ndarray:
    global _GEN_COUNTER
    forced_quadrant = QUADRANTS[_GEN_COUNTER % len(QUADRANTS)]
    _GEN_COUNTER += 1

    for _ in range(GENERATION_ATTEMPTS):
        grid = _render_attempt(forced_quadrant)
        if grid is not None:
            assert grid.ndim == 2 and max(grid.shape) <= 30
            return grid
    raise RuntimeError("failed to generate a valid rotation-vote repair input")
