from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
QUADRANTS = ("UL", "UR", "LL", "LR")

# Generator configuration.
TOP_HEIGHT_RANGE = (6, 10)
BOTTOM_HEIGHT_RANGE = (6, 10)
LEFT_WIDTH_RANGE = (7, 12)
RIGHT_WIDTH_RANGE = (7, 12)
BLOCK_COUNT_RANGE = (1, 4)
PLACEMENT_ATTEMPTS = 120
GENERATION_ATTEMPTS = 400


def _dominant_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _detect_cross(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int], tuple[int, int]]:
    h, w = grid.shape
    candidates: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    for raw_color in np.unique(grid):
        color = int(raw_color)
        if color == bg:
            continue
        full_rows = [r for r in range(h) if np.all(grid[r, :] == color)]
        full_cols = [c for c in range(w) if np.all(grid[:, c] == color)]
        if len(full_rows) != 2 or len(full_cols) != 2:
            continue
        if full_rows[1] != full_rows[0] + 1 or full_cols[1] != full_cols[0] + 1:
            continue
        candidates.append((color, (full_rows[0], full_rows[1]), (full_cols[0], full_cols[1])))
    assert len(candidates) == 1, "expected exactly one 2x2-centered full cross"
    return candidates[0]


def _find_components(grid: np.ndarray, blocked: set[int]) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in blocked or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(arr.shape[0]),
                }
            )

    return comps


def _quadrant_of(bbox: tuple[int, int, int, int], cross_rows: tuple[int, int], cross_cols: tuple[int, int]) -> str:
    r0, c0, r1, c1 = bbox
    if r1 < cross_rows[0] and c1 < cross_cols[0]:
        return "UL"
    if r1 < cross_rows[0] and c0 > cross_cols[1]:
        return "UR"
    if r0 > cross_rows[1] and c1 < cross_cols[0]:
        return "LL"
    if r0 > cross_rows[1] and c0 > cross_cols[1]:
        return "LR"
    raise AssertionError("foreground block must lie entirely inside one quadrant")


def _storage_params(quadrant: str, cross_rows: tuple[int, int], cross_cols: tuple[int, int]) -> tuple[int, int, int, int]:
    if quadrant[0] == "U":
        near_row = cross_rows[0] - 1
        far_row = cross_rows[0] - 2
    else:
        near_row = cross_rows[1] + 1
        far_row = cross_rows[1] + 2

    if quadrant[1] == "L":
        start_col = cross_cols[0] - 1
        step = -1
    else:
        start_col = cross_cols[1] + 1
        step = 1
    return near_row, far_row, start_col, step


def _shape_of(comp: dict) -> str:
    r0, c0, r1, c1 = comp["bbox"]
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    assert comp["area"] == 2, "expected each foreground block to have area 2"
    assert (height, width) in ((2, 1), (1, 2)), "expected only 2x1 or 1x2 blocks"
    return "v" if (height, width) == (2, 1) else "h"


def _center2_and_dist2(
    comp: dict,
    cross_rows: tuple[int, int],
    cross_cols: tuple[int, int],
) -> tuple[int, int, int]:
    r0, c0, r1, c1 = comp["bbox"]
    center_r2 = r0 + r1
    center_c2 = c0 + c1
    cross_r2 = cross_rows[0] + cross_rows[1]
    cross_c2 = cross_cols[0] + cross_cols[1]
    dist2 = max(abs(center_r2 - cross_r2), abs(center_c2 - cross_c2))
    return center_r2, center_c2, dist2


def _pack_quadrant(
    out: np.ndarray,
    bg: int,
    quadrant: str,
    blocks: list[dict],
    cross_rows: tuple[int, int],
    cross_cols: tuple[int, int],
) -> None:
    if not blocks:
        return

    near_row, far_row, cursor, step = _storage_params(quadrant, cross_rows, cross_cols)
    for comp in blocks:
        color = int(comp["color"])
        shape = _shape_of(comp)
        if shape == "v":
            targets = [(far_row, cursor), (near_row, cursor)]
            cursor += step
        else:
            targets = [(near_row, cursor), (near_row, cursor + step)]
            cursor += 2 * step

        for r, c in targets:
            assert 0 <= r < out.shape[0] and 0 <= c < out.shape[1], "storage placement must stay in bounds"
            assert int(out[r, c]) == bg, "storage cells must be empty before packing"
            out[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Gather each quadrant's scattered domino blocks, sort them by visual distance to the
    central cross, and store them along the inner strip next to the center.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _dominant_color(g)
    cross_color, cross_rows, cross_cols = _detect_cross(g, bg)

    comps = _find_components(g, blocked={bg, cross_color})
    assert comps, "expected at least one foreground block"

    by_quadrant: dict[str, list[dict]] = {quadrant: [] for quadrant in QUADRANTS}
    out = g.copy()

    for comp in comps:
        shape = _shape_of(comp)
        quadrant = _quadrant_of(comp["bbox"], cross_rows, cross_cols)
        center_r2, center_c2, dist2 = _center2_and_dist2(comp, cross_rows, cross_cols)
        comp["shape"] = shape
        comp["quadrant"] = quadrant
        comp["center_r2"] = center_r2
        comp["center_c2"] = center_c2
        comp["dist2"] = dist2
        by_quadrant[quadrant].append(comp)
        for r, c in comp["coords"]:
            out[int(r), int(c)] = bg

    for quadrant, blocks in by_quadrant.items():
        blocks.sort(key=lambda comp: (int(comp["dist2"]), int(comp["center_r2"]), int(comp["center_c2"])))
        _pack_quadrant(out, bg, quadrant, blocks, cross_rows, cross_cols)

    return out


def _blank_canvas(top_h: int, bottom_h: int, left_w: int, right_w: int) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    rows = top_h + 2 + bottom_h
    cols = left_w + 2 + right_w
    grid = np.full((rows, cols), BG, dtype=int)
    cross_rows = (top_h, top_h + 1)
    cross_cols = (left_w, left_w + 1)
    grid[cross_rows[0] : cross_rows[1] + 1, :] = GREY
    grid[:, cross_cols[0] : cross_cols[1] + 1] = GREY
    return grid, cross_rows, cross_cols


def _scatter_bounds(
    grid_shape: tuple[int, int],
    cross_rows: tuple[int, int],
    cross_cols: tuple[int, int],
    quadrant: str,
) -> tuple[int, int, int, int]:
    rows, cols = grid_shape
    if quadrant == "UL":
        return 0, cross_rows[0] - 3, 0, cross_cols[0] - 1
    if quadrant == "UR":
        return 0, cross_rows[0] - 3, cross_cols[1] + 1, cols - 1
    if quadrant == "LL":
        return cross_rows[1] + 3, rows - 1, 0, cross_cols[0] - 1
    if quadrant == "LR":
        return cross_rows[1] + 3, rows - 1, cross_cols[1] + 1, cols - 1
    raise AssertionError("unknown quadrant")


def _storage_width(orientations: list[str]) -> int:
    return sum(1 if orientation == "v" else 2 for orientation in orientations)


def _choose_orientations(side_width: int) -> list[str] | None:
    for _ in range(80):
        count = random.randint(BLOCK_COUNT_RANGE[0], BLOCK_COUNT_RANGE[1])
        orientations = [random.choice(("v", "h")) for _ in range(count)]
        if _storage_width(orientations) <= side_width:
            return orientations
    return None


def _can_place(mask: np.ndarray, r0: int, c0: int, height: int, width: int) -> bool:
    rows, cols = mask.shape
    if r0 < 0 or c0 < 0 or r0 + height > rows or c0 + width > cols:
        return False
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(rows, r0 + height + 1)
    cc1 = min(cols, c0 + width + 1)
    return not bool(mask[rr0:rr1, cc0:cc1].any())


def _place_quadrant_blocks(
    grid: np.ndarray,
    cross_rows: tuple[int, int],
    cross_cols: tuple[int, int],
    quadrant: str,
    orientations: list[str],
) -> list[dict] | None:
    r_lo, r_hi, c_lo, c_hi = _scatter_bounds(grid.shape, cross_rows, cross_cols, quadrant)
    avail_h = r_hi - r_lo + 1
    avail_w = c_hi - c_lo + 1
    if avail_h < 2 or avail_w < 1:
        return None

    colors = random.sample(PALETTE, len(orientations))
    mask = np.zeros((avail_h, avail_w), dtype=bool)
    cross_r2 = cross_rows[0] + cross_rows[1]
    cross_c2 = cross_cols[0] + cross_cols[1]
    used_dist2: set[int] = set()
    specs: list[dict] = []

    for color, orientation in zip(colors, orientations):
        height, width = ((2, 1) if orientation == "v" else (1, 2))
        placements = [
            (r, c)
            for r in range(r_lo, r_hi - height + 2)
            for c in range(c_lo, c_hi - width + 2)
            if _can_place(mask, r - r_lo, c - c_lo, height, width)
        ]
        random.shuffle(placements)

        placed = False
        for r0, c0 in placements[:PLACEMENT_ATTEMPTS]:
            r1 = r0 + height - 1
            c1 = c0 + width - 1
            center_r2 = r0 + r1
            center_c2 = c0 + c1
            dist2 = max(abs(center_r2 - cross_r2), abs(center_c2 - cross_c2))
            if dist2 in used_dist2:
                continue

            mask[r0 - r_lo : r0 - r_lo + height, c0 - c_lo : c0 - c_lo + width] = True
            grid[r0 : r1 + 1, c0 : c1 + 1] = color
            used_dist2.add(dist2)
            specs.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "area": 2,
                    "shape": orientation,
                    "quadrant": quadrant,
                    "center_r2": center_r2,
                    "center_c2": center_c2,
                    "dist2": dist2,
                }
            )
            placed = True
            break

        if not placed:
            return None

    return specs


def _expected_from_specs(
    inp: np.ndarray,
    specs: list[dict],
    cross_rows: tuple[int, int],
    cross_cols: tuple[int, int],
) -> np.ndarray:
    out = inp.copy()
    by_quadrant: dict[str, list[dict]] = {quadrant: [] for quadrant in QUADRANTS}

    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = BG
        by_quadrant[str(spec["quadrant"])].append(spec)

    for quadrant in QUADRANTS:
        blocks = sorted(
            by_quadrant[quadrant],
            key=lambda spec: (int(spec["dist2"]), int(spec["center_r2"]), int(spec["center_c2"])),
        )
        near_row, far_row, cursor, step = _storage_params(quadrant, cross_rows, cross_cols)
        for spec in blocks:
            color = int(spec["color"])
            if spec["shape"] == "v":
                targets = [(far_row, cursor), (near_row, cursor)]
                cursor += step
            else:
                targets = [(near_row, cursor), (near_row, cursor + step)]
                cursor += 2 * step
            for r, c in targets:
                assert out[r, c] == BG
                out[r, c] = color

    return out


def _gen_attempt() -> np.ndarray | None:
    top_h = random.randint(*TOP_HEIGHT_RANGE)
    bottom_h = random.randint(*BOTTOM_HEIGHT_RANGE)
    left_w = random.randint(*LEFT_WIDTH_RANGE)
    right_w = random.randint(*RIGHT_WIDTH_RANGE)
    grid, cross_rows, cross_cols = _blank_canvas(top_h, bottom_h, left_w, right_w)

    planned_orientations = {
        "UL": _choose_orientations(left_w),
        "UR": _choose_orientations(right_w),
        "LL": _choose_orientations(left_w),
        "LR": _choose_orientations(right_w),
    }
    if any(orientations is None for orientations in planned_orientations.values()):
        return None

    specs: list[dict] = []
    for quadrant in QUADRANTS:
        placed = _place_quadrant_blocks(
            grid,
            cross_rows,
            cross_cols,
            quadrant,
            planned_orientations[quadrant],
        )
        if placed is None:
            return None
        specs.extend(placed)

    expected = _expected_from_specs(grid, specs, cross_rows, cross_cols)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
