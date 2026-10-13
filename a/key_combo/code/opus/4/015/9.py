from __future__ import annotations

from importlib.machinery import SourceFileLoader
import random

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_arc_utils_shared",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()

Color = _ARC_UTILS.Color
bbox_coords = _ARC_UTILS.bbox_coords
colors_present = _ARC_UTILS.colors_present
components_mask = _ARC_UTILS.components_mask
crop_bbox = _ARC_UTILS.crop_bbox
get_bg = _ARC_UTILS.get_bg
paste = _ARC_UTILS.paste


BG_COLOR = Color.BLACK
MARKER_COLOR = Color.GREY
SPRITE_COLOR = Color.BLUE
GRID_LIMIT = 30
GEN_RETRY_LIMIT = 300

SHAPE_CATALOG = (
    np.array([[1, 0], [1, 1]], dtype=int),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
)


def _rotate_patch_cw(patch: np.ndarray, quarter_turns: int) -> np.ndarray:
    turns = int(quarter_turns) % 4
    if turns == 0:
        return patch.copy()
    return np.rot90(patch, k=-turns).copy()


def _patch_signature(patch: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in patch.tolist())


def _center_top_left(anchor_r: int, anchor_c: int, pitch: int, patch: np.ndarray) -> tuple[int, int]:
    h, w = map(int, patch.shape)
    return int(anchor_r + (pitch - h) // 2), int(anchor_c + (pitch - w) // 2)


def _find_lattice(grid: np.ndarray) -> tuple[list[int], list[int], int, int]:
    marker_coords = np.argwhere(grid == MARKER_COLOR)
    assert len(marker_coords) > 0, "expected gray marker lattice"

    rows = sorted({int(r) for r, _c in marker_coords})
    cols = sorted({int(c) for _r, c in marker_coords})
    n = len(rows)
    assert n == len(cols) and n in (4, 5), "expected a 4x4 or 5x5 lattice"
    assert len(marker_coords) == n * n, "expected a complete Cartesian lattice"
    assert all(int(grid[r, c]) == MARKER_COLOR for r in rows for c in cols)

    row_diffs = np.diff(rows)
    col_diffs = np.diff(cols)
    assert len(set(map(int, row_diffs))) == 1, "expected equal row spacing"
    assert len(set(map(int, col_diffs))) == 1, "expected equal column spacing"
    pitch = int(row_diffs[0])
    assert pitch == int(col_diffs[0]) and pitch >= 4, "expected square lattice cells with pitch >= 4"
    return rows, cols, pitch, n


def _cell_index(rows: list[int], cols: list[int], pitch: int, bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, bbox)
    row_matches = [idx for idx, row in enumerate(rows) if row < r0 and r1 < row + pitch]
    col_matches = [idx for idx, col in enumerate(cols) if col < c0 and c1 < col + pitch]
    assert len(row_matches) == 1 and len(col_matches) == 1, "sprite must belong to one lattice cell"
    return int(row_matches[0]), int(col_matches[0])


def _collect_sprites(
    grid: np.ndarray,
) -> tuple[list[int], list[int], int, int, int, np.ndarray, list[tuple[int, int]]]:
    rows, cols, pitch, n = _find_lattice(grid)
    bg = int(get_bg(grid))
    sprite_colors = colors_present(grid, exclude={bg, MARKER_COLOR})
    assert sprite_colors == [SPRITE_COLOR], "expected blue sprites and gray markers"

    components = components_mask(grid == SPRITE_COLOR, connectivity=4)
    assert len(components) == n * n, "expected one sprite in every lattice cell"

    template_patch: np.ndarray | None = None
    used_cells: set[tuple[int, int]] = set()
    cell_order: list[tuple[int, int]] = []
    for coords in components:
        assert 3 <= int(coords.shape[0]) <= 4, "expected a small 3-4 pixel sprite"
        patch = crop_bbox(grid, bbox_coords(coords))
        if template_patch is None:
            template_patch = patch
            variants = {_patch_signature(_rotate_patch_cw(template_patch, k)) for k in range(4)}
            assert len(variants) == 4, "sprite must visibly change under every quarter-turn rotation"
        else:
            assert np.array_equal(patch, template_patch), "all input sprites must share one shape and orientation"

        cell = _cell_index(rows, cols, pitch, bbox_coords(coords))
        assert cell not in used_cells, "expected at most one sprite per cell"
        used_cells.add(cell)
        cell_order.append(cell)

    assert template_patch is not None
    assert len(used_cells) == n * n, "expected all lattice cells to be occupied"
    return rows, cols, pitch, n, bg, template_patch, cell_order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Center each repeated sprite inside its lattice cell, then rotate it by the cell's concentric-ring depth.

    Input structure:
    - The background is black.
    - Gray pixels form a complete 4x4 or 5x5 regular Cartesian marker lattice with one sprite-carrying cell per marker.
    - Every cell contains the same off-center blue sprite, and the sprite is asymmetric under 90-degree rotation.

    Rule:
    - Keep the gray marker lattice fixed.
    - Identify the blue sprite in each lattice cell, clear its original off-center placement, and move the sprite to the centered slot of that cell.
    - Compute each cell's concentric depth inside the NxN cell array as `min(i, j, N-1-i, N-1-j)`.
    - Rotate the centered sprite clockwise by `depth * 90` degrees, so the outer ring stays unchanged, the next ring turns 90 degrees, and so on.

    Key insight:
    - The output orientation is not tied to absolute row or column position; it is determined only by each cell's global concentric-ring depth in the marker lattice.
    """
    g = np.asarray(grid, dtype=int)
    rows, cols, pitch, n, bg, template_patch, cell_order = _collect_sprites(g)

    out = g.copy()
    out[out == SPRITE_COLOR] = bg

    for i, j in cell_order:
        depth = min(i, j, n - 1 - i, n - 1 - j)
        rotated = _rotate_patch_cw(template_patch, depth)
        target_r, target_c = _center_top_left(rows[i], cols[j], pitch, rotated)
        target_mask = rotated != bg
        target = out[target_r : target_r + rotated.shape[0], target_c : target_c + rotated.shape[1]]
        assert np.all(target[target_mask] == bg), "centered target slot must be empty"
        paste(out, rotated, target_r, target_c, transparent=bg)

    return out


def _sample_layout() -> tuple[int, int, int, int, int, int]:
    n = 5 if random.random() < 0.6 else 4
    pitch_choices = [5] if n == 5 else [5, 7]
    pitch = random.choice(pitch_choices)

    top = random.randint(0, min(2, GRID_LIMIT - n * pitch))
    left = random.randint(0, min(2, GRID_LIMIT - n * pitch))
    bottom = random.randint(0, min(2, GRID_LIMIT - (top + n * pitch)))
    right = random.randint(0, min(2, GRID_LIMIT - (left + n * pitch)))

    height = top + n * pitch + bottom
    width = left + n * pitch + right
    assert height <= GRID_LIMIT and width <= GRID_LIMIT
    return n, pitch, top, left, height, width


def _valid_offsets(pitch: int, patch: np.ndarray) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    h, w = map(int, patch.shape)
    centered = ((pitch - h) // 2, (pitch - w) // 2)
    offsets = [
        (dr, dc)
        for dr in range(1, pitch - h + 1)
        for dc in range(1, pitch - w + 1)
        if (dr, dc) != centered
    ]
    assert offsets, "need at least one non-centered placement"
    return offsets, centered


def _place_markers(shape: tuple[int, int], rows: list[int], cols: list[int]) -> np.ndarray:
    grid = np.full(shape, BG_COLOR, dtype=int)
    grid[np.ix_(rows, cols)] = MARKER_COLOR
    return grid


def _build_expected_output(
    shape: tuple[int, int],
    rows: list[int],
    cols: list[int],
    pitch: int,
    input_patch: np.ndarray,
) -> np.ndarray:
    n = len(rows)
    out = _place_markers(shape, rows, cols)
    for i in range(n):
        for j in range(n):
            depth = min(i, j, n - 1 - i, n - 1 - j)
            rotated = _rotate_patch_cw(input_patch, depth)
            target_r, target_c = _center_top_left(rows[i], cols[j], pitch, rotated)
            paste(out, rotated, target_r, target_c, transparent=BG_COLOR)
    return out


def _semantic_checks(
    inp: np.ndarray,
    out: np.ndarray,
    rows: list[int],
    cols: list[int],
    pitch: int,
    input_patch: np.ndarray,
) -> bool:
    n = len(rows)
    if not np.all(inp[np.ix_(rows, cols)] == MARKER_COLOR):
        return False
    if not np.all(out[np.ix_(rows, cols)] == MARKER_COLOR):
        return False

    seen_orientations: set[tuple[tuple[int, ...], ...]] = set()
    for i in range(n):
        for j in range(n):
            depth = min(i, j, n - 1 - i, n - 1 - j)
            rotated = _rotate_patch_cw(input_patch, depth)
            target_r, target_c = _center_top_left(rows[i], cols[j], pitch, rotated)
            view = out[target_r : target_r + rotated.shape[0], target_c : target_c + rotated.shape[1]]
            if not np.array_equal(view, rotated):
                return False
            seen_orientations.add(_patch_signature(rotated))

    return len(seen_orientations) >= 2


def _gen_attempt() -> np.ndarray | None:
    n, pitch, top, left, height, width = _sample_layout()
    rows = [top + i * pitch for i in range(n)]
    cols = [left + j * pitch for j in range(n)]

    base_patch = random.choice(SHAPE_CATALOG)
    input_patch = _rotate_patch_cw(base_patch, random.randrange(4))
    offsets, centered = _valid_offsets(pitch, input_patch)

    inp = _place_markers((height, width), rows, cols)
    chosen_offsets: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(n):
            dr, dc = random.choice(offsets)
            chosen_offsets.append((dr, dc))
            paste(inp, input_patch, rows[i] + dr, cols[j] + dc, transparent=BG_COLOR)

    if len(set(chosen_offsets)) < 2:
        return None

    out = _build_expected_output(inp.shape, rows, cols, pitch, input_patch)
    if centered in chosen_offsets:
        return None
    if not _semantic_checks(inp, out, rows, cols, pitch, input_patch):
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid concentric-depth lattice puzzle")
