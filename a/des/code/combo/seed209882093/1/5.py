import itertools
import random

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))

GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (22, 36)
SHAPE_COUNT_RANGE = (2, 5)
INNER_H_RANGE = (3, 8)
INNER_W_RANGE = (3, 10)
MAX_LINES_PER_AXIS = 3
PLACEMENT_PAD = 2
PLACEMENT_RETRIES = 250
PALETTE_GAP_RANGE = (1, 3)
GEN_RETRIES = 500

DELTAS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DELTAS8 = DELTAS4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    deltas = DELTAS8 if connectivity == 8 else DELTAS4
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            cells: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in deltas:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _palette_row(grid: np.ndarray) -> int:
    rows = np.flatnonzero(np.any(grid != BG, axis=1))
    assert rows.size > 0, "expected non-empty grid"
    return int(rows[-1])


def _parse_palette(row: np.ndarray) -> list[int]:
    colors: list[int] = []
    w = int(row.shape[0])
    c = 0
    while c < w:
        if int(row[c]) == BG:
            c += 1
            continue
        color = int(row[c])
        start = c
        c += 1
        while c < w and int(row[c]) == color:
            c += 1
        assert c == start + 1, "palette entries must be single pixels"
        if c < w:
            assert int(row[c]) == BG, "palette entries must be separated by black"
        colors.append(color)
    assert colors, "palette must be non-empty"
    assert len(set(colors)) == len(colors), "palette colors must be distinct"
    return colors


def _local_mask(coords: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _shape_regions_and_contour(mask: np.ndarray) -> tuple[int, np.ndarray]:
    bg_comps = _components(~mask, connectivity=4)
    outside = np.zeros_like(mask, dtype=bool)
    inside_count = 0
    h, w = mask.shape

    for comp in bg_comps:
        touches_edge = bool(
            np.any(comp[:, 0] == 0)
            or np.any(comp[:, 0] == h - 1)
            or np.any(comp[:, 1] == 0)
            or np.any(comp[:, 1] == w - 1)
        )
        if touches_edge:
            outside[comp[:, 0], comp[:, 1]] = True
        else:
            inside_count += 1

    assert inside_count >= 1, "closed shape must contain at least one interior region"

    contour = np.zeros_like(mask, dtype=bool)
    for r, c in np.argwhere(mask):
        if r == 0 or c == 0 or r == h - 1 or c == w - 1:
            contour[r, c] = True
            continue
        if (
            outside[r - 1, c]
            or outside[r + 1, c]
            or outside[r, c - 1]
            or outside[r, c + 1]
        ):
            contour[r, c] = True

    assert bool(np.any(contour)), "shape must have an outer contour"
    return int(inside_count), contour


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each closed shape's outer contour by ranking shapes by the number of
    black subregions inside them after internal cut lines split the interior.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected 2D grid"

    palette_r = _palette_row(g)
    palette = _parse_palette(g[palette_r, :])

    scene = g.copy()
    scene[palette_r, :] = BG

    shapes = []
    for coords in _components(scene != BG, connectivity=8):
        box = _bbox(coords)
        local = _local_mask(coords, box)
        region_count, contour_local = _shape_regions_and_contour(local)
        r0, c0, _r1, _c1 = box
        contour = np.argwhere(contour_local) + np.array([r0, c0], dtype=int)
        shapes.append(
            {
                "bbox": box,
                "regions": int(region_count),
                "contour": contour,
            }
        )

    assert len(shapes) == len(palette), "palette size must match shape count"
    shapes.sort(key=lambda item: (item["regions"], item["bbox"][1], item["bbox"][0]))

    out = scene.copy()
    for color, shape in zip(palette, shapes):
        out[shape["contour"][:, 0], shape["contour"][:, 1]] = int(color)

    return out


def _sample_spaced_positions(length: int, count: int) -> list[int]:
    if count == 0:
        return []
    choices = [
        combo
        for combo in itertools.combinations(range(length), count)
        if all(b - a >= 2 for a, b in zip(combo, combo[1:]))
    ]
    assert choices, "requested line count must fit"
    return list(random.choice(choices))


def _choose_shape_colors(palette: list[int]) -> tuple[int, int]:
    pool = [c for c in COLORS if c not in palette]
    if len(pool) >= 2:
        contour, cut = random.sample(pool, 2)
    else:
        contour, cut = random.sample(COLORS, 2)
    return int(contour), int(cut)


def _contour_coords(h: int, w: int) -> np.ndarray:
    cells = [(0, c) for c in range(w)]
    cells.extend((h - 1, c) for c in range(w))
    cells.extend((r, 0) for r in range(1, h - 1))
    cells.extend((r, w - 1) for r in range(1, h - 1))
    return np.asarray(cells, dtype=int)


def _build_shape(palette: list[int]) -> dict[str, object]:
    inner_h = random.randint(*INNER_H_RANGE)
    inner_w = random.randint(*INNER_W_RANGE)

    max_h_lines = min(MAX_LINES_PER_AXIS, (inner_h + 1) // 2)
    max_v_lines = min(MAX_LINES_PER_AXIS, (inner_w + 1) // 2)
    h_lines = random.randint(0, max_h_lines)
    v_lines = random.randint(0, max_v_lines)
    h_pos = _sample_spaced_positions(inner_h, h_lines)
    v_pos = _sample_spaced_positions(inner_w, v_lines)

    contour_color, cut_color = _choose_shape_colors(palette)
    patch = np.zeros((inner_h + 2, inner_w + 2), dtype=int)
    patch[0, :] = contour_color
    patch[-1, :] = contour_color
    patch[:, 0] = contour_color
    patch[:, -1] = contour_color
    for rr in h_pos:
        patch[rr + 1, 1:-1] = cut_color
    for cc in v_pos:
        patch[1:-1, cc + 1] = cut_color

    return {
        "patch": patch,
        "contour_local": _contour_coords(inner_h + 2, inner_w + 2),
        "regions": int((h_lines + 1) * (v_lines + 1)),
        "contour_color": int(contour_color),
    }


def _place_shapes(h: int, w: int, specs: list[dict[str, object]]) -> list[dict[str, object]] | None:
    used = np.zeros((h, w), dtype=bool)
    used_lefts: set[int] = set()
    placed: list[dict[str, object]] = []

    order = sorted(specs, key=lambda item: -int(np.prod(item["patch"].shape)))
    for spec in order:
        patch = np.asarray(spec["patch"], dtype=int)
        ph, pw = map(int, patch.shape)
        max_top = h - ph - 2
        max_left = w - pw - 1
        if max_top < 1 or max_left < 1:
            return None

        candidates = [
            (top, left)
            for top in range(1, max_top + 1)
            for left in range(1, max_left + 1)
            if left not in used_lefts
        ]
        random.shuffle(candidates)

        chosen: tuple[int, int] | None = None
        for top, left in candidates[:PLACEMENT_RETRIES]:
            r0 = max(0, top - PLACEMENT_PAD)
            c0 = max(0, left - PLACEMENT_PAD)
            r1 = min(h, top + ph + PLACEMENT_PAD)
            c1 = min(w, left + pw + PLACEMENT_PAD)
            if not np.any(used[r0:r1, c0:c1]):
                chosen = (top, left)
                break
        if chosen is None:
            return None

        top, left = chosen
        r0 = max(0, top - PLACEMENT_PAD)
        c0 = max(0, left - PLACEMENT_PAD)
        r1 = min(h, top + ph + PLACEMENT_PAD)
        c1 = min(w, left + pw + PLACEMENT_PAD)
        used[r0:r1, c0:c1] = True
        used_lefts.add(int(left))

        placed.append(
            {
                "patch": patch,
                "top": int(top),
                "left": int(left),
                "regions": int(spec["regions"]),
                "contour_color": int(spec["contour_color"]),
                "contour": np.asarray(spec["contour_local"], dtype=int)
                + np.array([top, left], dtype=int),
            }
        )

    return placed


def _palette_columns(width: int, count: int) -> list[int] | None:
    gaps = [random.randint(*PALETTE_GAP_RANGE) for _ in range(count - 1)]
    total = count + sum(gaps)
    if total > width:
        return None
    start = random.randint(0, width - total)
    cols: list[int] = []
    cur = start
    for i in range(count):
        cols.append(int(cur))
        if i < count - 1:
            cur += 1 + gaps[i]
    return cols


def _expected_output(grid: np.ndarray, placed: list[dict[str, object]], palette: list[int]) -> np.ndarray:
    out = grid.copy()
    out[-1, :] = BG
    ordered = sorted(placed, key=lambda item: (item["regions"], item["left"], item["top"]))
    for color, shape in zip(palette, ordered):
        contour = np.asarray(shape["contour"], dtype=int)
        out[contour[:, 0], contour[:, 1]] = int(color)
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    n_shapes = random.randint(*SHAPE_COUNT_RANGE)
    palette = random.sample(COLORS, n_shapes)

    specs = [_build_shape(palette) for _ in range(n_shapes)]
    if max(int(spec["regions"]) for spec in specs) == 1:
        return None

    placed = _place_shapes(h, w, specs)
    if placed is None:
        return None

    palette_cols = _palette_columns(w, n_shapes)
    if palette_cols is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    for shape in placed:
        patch = np.asarray(shape["patch"], dtype=int)
        top = int(shape["top"])
        left = int(shape["left"])
        ph, pw = patch.shape
        grid[top : top + ph, left : left + pw] = np.where(
            patch != BG,
            patch,
            grid[top : top + ph, left : left + pw],
        )

    for col, color in zip(palette_cols, palette):
        grid[h - 1, col] = int(color)

    expected = _expected_output(grid, placed, palette)
    changed = any(int(shape["contour_color"]) != int(color) for shape, color in zip(
        sorted(placed, key=lambda item: (item["regions"], item["left"], item["top"])),
        palette,
    ))
    if not changed:
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
