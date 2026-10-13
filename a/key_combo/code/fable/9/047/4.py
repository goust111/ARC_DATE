from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


GRAY = 5
RED = 2
PALETTE = tuple(color for color in range(10) if color not in {GRAY, RED})

GEN_HEIGHT_RANGE = (12, 24)
GEN_WIDTH_RANGE = (12, 24)
GEN_TILE_HEIGHTS = (2, 3, 4)
GEN_TILE_WIDTHS = (2, 3, 4)
GEN_RECT_SIDE_MAX = 5
GEN_RECT_COUNT_MIN = 2
GEN_RECT_COUNT_MAX = 5
GEN_COVERAGE_RANGE = (0.05, 0.24)
GEN_ATTEMPTS = 400
GEN_PAIR_SCAN_LIMIT = 80


def _tile_grid(tile: np.ndarray, height: int, width: int) -> np.ndarray:
    rows = np.arange(height)[:, None] % tile.shape[0]
    cols = np.arange(width)[None, :] % tile.shape[1]
    return tile[rows, cols].astype(int)


def _is_full_period(tile: np.ndarray) -> bool:
    height, width = tile.shape
    for row_period in range(1, height + 1):
        for col_period in range(1, width + 1):
            if row_period == height and col_period == width:
                continue
            if all(
                int(tile[r, c]) == int(tile[r % row_period, c % col_period])
                for r in range(height)
                for c in range(width)
            ):
                return False
    return True


def _consistent_tile(
    visible_rows: np.ndarray,
    visible_cols: np.ndarray,
    visible_values: np.ndarray,
    row_period: int,
    col_period: int,
) -> np.ndarray | None:
    tile = np.full((row_period, col_period), -1, dtype=int)
    for r, c, value in zip(visible_rows, visible_cols, visible_values):
        rr = int(r) % row_period
        cc = int(c) % col_period
        value = int(value)
        if tile[rr, cc] == -1:
            tile[rr, cc] = value
        elif tile[rr, cc] != value:
            return None
    if np.any(tile == -1):
        return None
    return tile


def _infer_unique_tile(grid: np.ndarray, missing_color: int = GRAY) -> tuple[int, int, np.ndarray]:
    height, width = grid.shape
    visible_rows, visible_cols = np.nonzero(grid != missing_color)
    visible_values = grid[visible_rows, visible_cols]

    best_area: int | None = None
    best: list[tuple[int, int, np.ndarray]] = []
    for row_period in range(1, height + 1):
        if best_area is not None and row_period > best_area:
            break
        for col_period in range(1, width + 1):
            area = row_period * col_period
            if best_area is not None and area > best_area:
                break
            tile = _consistent_tile(
                visible_rows,
                visible_cols,
                visible_values,
                row_period,
                col_period,
            )
            if tile is None:
                continue
            if best_area is None or area < best_area:
                best_area = area
                best = [(row_period, col_period, tile)]
            else:
                best.append((row_period, col_period, tile))

    assert best, "visible cells must determine a periodic tile"
    assert len(best) == 1, "minimum-area periodic tile must be unique"
    return best[0]


def _patch_from_tile(
    tile: np.ndarray, row_period: int, col_period: int, r0: int, c0: int, r1: int, c1: int
) -> np.ndarray:
    rows = np.arange(r0, r1 + 1) % row_period
    cols = np.arange(c0, c1 + 1) % col_period
    return tile[np.ix_(rows, cols)].astype(int)


def _find_color_rectangles(grid: np.ndarray, color: int) -> list[tuple[int, int, int, int]]:
    height, width = grid.shape
    seen = np.zeros((height, width), dtype=bool)
    rectangles: list[tuple[int, int, int, int]] = []

    for r in range(height):
        for c in range(width):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if (
                        0 <= nr < height
                        and 0 <= nc < width
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [rr for rr, _cc in coords]
            cols = [cc for _rr, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            assert np.all(patch == color), "gray component must be a solid rectangle"
            rectangles.append((r0, c0, r1, c1))

    return rectangles


def main(grid: np.ndarray) -> np.ndarray:
    """Restore a gray-occluded binary wallpaper, except fully single-phase rectangles become solid red.

    Input structure:
    - The grid contains exactly two non-gray pattern colors tiled from one unique minimum rectangular period.
    - One or more solid gray rectangles occlude parts of that wallpaper; red does not appear in the input.
    - The visible non-gray cells are sufficient to recover both the tile and its global phase.

    Rule:
    - Infer the unique minimum-area repeating tile from the visible non-gray cells.
    - For each gray rectangle, inspect the tile colors that should lie under every covered cell.
    - If the whole rectangle lies on one tile color, repaint the entire rectangle red; otherwise restore each cell from the inferred tile.

    Key insight:
    - All gray rectangles look identical at the visible layer; their fate is decided only by the hidden periodic phase underneath, so the wallpaper must be reconstructed first.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    colors = set(map(int, np.unique(g)))
    assert GRAY in colors, "expected gray occluder rectangles"
    assert RED not in colors, "red is output-only"
    visible_colors = colors - {GRAY}
    assert len(visible_colors) == 2, "expected exactly two pattern colors besides gray"

    row_period, col_period, tile = _infer_unique_tile(g, missing_color=GRAY)
    assert set(map(int, tile.ravel())) == visible_colors

    out = g.copy()
    rectangles = _find_color_rectangles(g, GRAY)
    assert rectangles, "expected at least one gray rectangle"
    for r0, c0, r1, c1 in rectangles:
        patch = _patch_from_tile(tile, row_period, col_period, r0, c0, r1, c1)
        if np.all(patch == int(patch[0, 0])):
            out[r0 : r1 + 1, c0 : c1 + 1] = RED
        else:
            out[r0 : r1 + 1, c0 : c1 + 1] = patch
    return out


def _boxes_separated(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], padding: int = 1
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + padding < br0 or br1 + padding < ar0 or ac1 + padding < bc0 or bc1 + padding < ac0


def _visible_every_residue(mask: np.ndarray, tile_shape: tuple[int, int]) -> bool:
    tile_h, tile_w = tile_shape
    for rr in range(tile_h):
        for cc in range(tile_w):
            if not np.any(mask[rr::tile_h, cc::tile_w]):
                return False
    return True


def _scan_rectangles(base: np.ndarray) -> dict[tuple[int, int], dict[str, list[tuple[int, int, int, int]]]]:
    height, width = base.shape
    classes: dict[tuple[int, int], dict[str, list[tuple[int, int, int, int]]]] = defaultdict(
        lambda: {"uniform": [], "mixed": []}
    )
    for rect_h in range(1, min(GEN_RECT_SIDE_MAX, height) + 1):
        for rect_w in range(1, min(GEN_RECT_SIDE_MAX, width) + 1):
            if rect_h * rect_w < 2:
                continue
            size_key = (rect_h, rect_w)
            for r0 in range(height - rect_h + 1):
                for c0 in range(width - rect_w + 1):
                    patch = base[r0 : r0 + rect_h, c0 : c0 + rect_w]
                    bbox = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
                    key = "uniform" if np.all(patch == int(patch[0, 0])) else "mixed"
                    classes[size_key][key].append(bbox)
    return classes


def _random_tile(rng: np.random.Generator) -> np.ndarray | None:
    tile_h = int(rng.choice(GEN_TILE_HEIGHTS))
    tile_w = int(rng.choice(GEN_TILE_WIDTHS))
    c1, c2 = map(int, rng.choice(PALETTE, size=2, replace=False))
    tile = rng.choice(np.array([c1, c2], dtype=int), size=(tile_h, tile_w))
    if len(set(map(int, tile.ravel()))) != 2:
        return None
    if not _is_full_period(tile):
        return None
    return tile.astype(int)


def _pick_rectangles(
    rng: np.random.Generator,
    classes: dict[tuple[int, int], dict[str, list[tuple[int, int, int, int]]]],
    grid_shape: tuple[int, int],
    tile_shape: tuple[int, int],
) -> list[tuple[tuple[int, int, int, int], str]] | None:
    shared_sizes = [
        size
        for size, groups in classes.items()
        if groups["uniform"] and groups["mixed"]
    ]
    if not shared_sizes:
        return None

    size_order = shared_sizes[:]
    rng.shuffle(size_order)
    size_order.sort(key=lambda size: size[0] * size[1], reverse=True)

    total = grid_shape[0] * grid_shape[1]
    target_min_cover = GEN_COVERAGE_RANGE[0] * total
    target_max_cover = GEN_COVERAGE_RANGE[1] * total

    for anchor_size in size_order:
        uniform_candidates = classes[anchor_size]["uniform"][:]
        mixed_candidates = classes[anchor_size]["mixed"][:]
        rng.shuffle(uniform_candidates)
        rng.shuffle(mixed_candidates)
        uniform_candidates = uniform_candidates[:GEN_PAIR_SCAN_LIMIT]
        mixed_candidates = mixed_candidates[:GEN_PAIR_SCAN_LIMIT]

        for box_u in uniform_candidates:
            for box_m in mixed_candidates:
                if not _boxes_separated(box_u, box_m):
                    continue

                selected: list[tuple[tuple[int, int, int, int], str]] = [
                    (box_u, "uniform"),
                    (box_m, "mixed"),
                ]
                covered = (box_u[2] - box_u[0] + 1) * (box_u[3] - box_u[1] + 1)
                covered += (box_m[2] - box_m[0] + 1) * (box_m[3] - box_m[1] + 1)

                for size in size_order:
                    if len(selected) >= GEN_RECT_COUNT_MAX:
                        break
                    if covered >= target_min_cover and len(selected) >= GEN_RECT_COUNT_MIN:
                        break
                    kinds = ["uniform", "mixed"]
                    rng.shuffle(kinds)
                    for kind in kinds:
                        candidates = classes[size][kind][:]
                        rng.shuffle(candidates)
                        for box in candidates:
                            if any(box == existing_box for existing_box, _existing_kind in selected):
                                continue
                            if not all(
                                _boxes_separated(box, existing_box)
                                for existing_box, _existing_kind in selected
                            ):
                                continue
                            box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
                            if covered + box_area > target_max_cover:
                                continue
                            selected.append((box, kind))
                            covered += box_area
                            break
                        if len(selected) >= GEN_RECT_COUNT_MAX or (
                            covered >= target_min_cover and len(selected) >= GEN_RECT_COUNT_MIN
                        ):
                            break

                if not (target_min_cover <= covered <= target_max_cover):
                    continue

                visible_mask = np.ones(grid_shape, dtype=bool)
                for (r0, c0, r1, c1), _kind in selected:
                    visible_mask[r0 : r1 + 1, c0 : c1 + 1] = False
                if _visible_every_residue(visible_mask, tile_shape):
                    return selected
    return None


def _expected_from_base(
    base: np.ndarray, rectangles: list[tuple[tuple[int, int, int, int], str]]
) -> tuple[np.ndarray, np.ndarray]:
    inp = base.copy()
    out = base.copy()
    for (r0, c0, r1, c1), kind in rectangles:
        inp[r0 : r1 + 1, c0 : c1 + 1] = GRAY
        if kind == "uniform":
            out[r0 : r1 + 1, c0 : c1 + 1] = RED
    return inp, out


def _validate_semantics(
    inp: np.ndarray,
    out: np.ndarray,
    base: np.ndarray,
    tile: np.ndarray,
    rectangles: list[tuple[tuple[int, int, int, int], str]],
) -> bool:
    if not rectangles:
        return False
    kinds = [kind for _box, kind in rectangles]
    if "uniform" not in kinds or "mixed" not in kinds:
        return False
    if RED in set(map(int, np.unique(inp))):
        return False

    for i, (box_a, _kind_a) in enumerate(rectangles):
        if not np.all(inp[box_a[0] : box_a[2] + 1, box_a[1] : box_a[3] + 1] == GRAY):
            return False
        for box_b, _kind_b in rectangles[i + 1 :]:
            if not _boxes_separated(box_a, box_b):
                return False

    for (r0, c0, r1, c1), kind in rectangles:
        patch = base[r0 : r1 + 1, c0 : c1 + 1]
        if kind == "uniform":
            if not np.all(patch == int(patch[0, 0])):
                return False
            if not np.all(out[r0 : r1 + 1, c0 : c1 + 1] == RED):
                return False
        else:
            if np.all(patch == int(patch[0, 0])):
                return False
            if not np.array_equal(out[r0 : r1 + 1, c0 : c1 + 1], patch):
                return False

    visible_mask = inp != GRAY
    if not _visible_every_residue(visible_mask, tile.shape):
        return False

    try:
        row_period, col_period, inferred_tile = _infer_unique_tile(inp, missing_color=GRAY)
    except AssertionError:
        return False
    if (row_period, col_period) != tile.shape:
        return False
    if not np.array_equal(inferred_tile, tile):
        return False
    if not np.array_equal(main(inp), out):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    tile = _random_tile(rng)
    if tile is None:
        return None

    height = int(rng.integers(GEN_HEIGHT_RANGE[0], GEN_HEIGHT_RANGE[1] + 1))
    width = int(rng.integers(GEN_WIDTH_RANGE[0], GEN_WIDTH_RANGE[1] + 1))
    base = _tile_grid(tile, height, width)

    classes = _scan_rectangles(base)
    rectangles = _pick_rectangles(rng, classes, base.shape, tile.shape)
    if rectangles is None:
        return None

    inp, out = _expected_from_base(base, rectangles)
    if not _validate_semantics(inp, out, base, tile, rectangles):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
