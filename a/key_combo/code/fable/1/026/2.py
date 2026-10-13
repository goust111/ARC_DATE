from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import colors_present, paste


_BG = 0
_BLOCKER = 5
_DIVIDER_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
_SPRITE_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
_MAX_ATTEMPTS = 400

_SHAPES_BY_SIZE: dict[int, list[np.ndarray]] = {
    1: [np.array([[1]], dtype=int)],
    2: [np.array([[1, 1]], dtype=int)],
    3: [
        np.array([[1, 1, 1]], dtype=int),
        np.array([[1, 0], [1, 1]], dtype=int),
    ],
    4: [
        np.array([[1, 1], [1, 1]], dtype=int),
        np.array([[1, 1, 1], [1, 0, 0]], dtype=int),
        np.array([[1, 1, 1, 1]], dtype=int),
    ],
    5: [
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
        np.array([[1, 0, 0], [1, 1, 1], [1, 0, 0]], dtype=int),
        np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=int),
    ],
    6: [
        np.array([[1, 1, 1], [1, 1, 1]], dtype=int),
        np.array([[1, 0, 0], [1, 1, 1], [1, 0, 1]], dtype=int),
        np.array([[1, 1, 1], [0, 1, 0], [1, 1, 0]], dtype=int),
    ],
}


def _line_groups(mask: np.ndarray) -> list[tuple[int, int]]:
    idxs = [int(i) for i in np.flatnonzero(mask)]
    if not idxs:
        return []
    groups: list[tuple[int, int]] = []
    start = idxs[0]
    prev = idxs[0]
    for idx in idxs[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        groups.append((start, prev))
        start = idx
        prev = idx
    groups.append((start, prev))
    return groups


def _segments(length: int, groups: list[tuple[int, int]]) -> list[tuple[int, int]]:
    pieces: list[tuple[int, int]] = []
    cursor = 0
    for start, end in groups:
        if cursor < start:
            pieces.append((cursor, start - 1))
        cursor = end + 1
    if cursor < length:
        pieces.append((cursor, length - 1))
    return pieces


def _find_divider_layout(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = map(int, grid.shape)
    candidates: list[tuple[int, list[tuple[int, int]], list[tuple[int, int]]]] = []
    for color in colors_present(grid):
        row_groups = _line_groups(np.all(grid == int(color), axis=1))
        col_groups = _line_groups(np.all(grid == int(color), axis=0))
        if len(row_groups) < 1 or len(col_groups) < 1:
            continue
        row_segments = _segments(h, row_groups)
        col_segments = _segments(w, col_groups)
        if len(row_segments) < 2 or len(col_segments) < 2:
            continue
        if len({r1 - r0 + 1 for r0, r1 in row_segments}) != 1:
            continue
        if len({c1 - c0 + 1 for c0, c1 in col_segments}) != 1:
            continue
        candidates.append((int(color), row_segments, col_segments))

    assert len(candidates) == 1, "expected exactly one regular divider grid"
    return candidates[0]


def _snake_cells(n_rows: int, n_cols: int) -> list[tuple[int, int]]:
    order: list[tuple[int, int]] = []
    for row in range(n_rows):
        cols = range(n_cols) if row % 2 == 0 else range(n_cols - 1, -1, -1)
        for col in cols:
            order.append((row, col))
    return order


def _rowmajor_cells(n_rows: int, n_cols: int) -> list[tuple[int, int]]:
    return [(row, col) for row in range(n_rows) for col in range(n_cols)]


def _infer_bg(
    grid: np.ndarray,
    row_segments: list[tuple[int, int]],
    col_segments: list[tuple[int, int]],
) -> int:
    counts: Counter[int] = Counter()
    for r0, r1 in row_segments:
        for c0, c1 in col_segments:
            cell = grid[r0 : r1 + 1, c0 : c1 + 1]
            if np.all(cell == _BLOCKER):
                continue
            counts.update(set(map(int, np.unique(cell))) - {_BLOCKER})
    assert counts, "expected at least one non-blocker cell"
    bg, _ = counts.most_common(1)[0]
    return int(bg)


def _analyze_grid(grid: np.ndarray) -> dict:
    divider_color, row_segments, col_segments = _find_divider_layout(grid)
    bg = _infer_bg(grid, row_segments, col_segments)
    blockers: set[tuple[int, int]] = set()
    sprites: list[dict] = []

    for row, (r0, r1) in enumerate(row_segments):
        for col, (c0, c1) in enumerate(col_segments):
            cell = grid[r0 : r1 + 1, c0 : c1 + 1]
            if np.all(cell == _BLOCKER):
                blockers.add((row, col))
                continue

            unique = set(map(int, np.unique(cell)))
            assert divider_color not in unique
            remaining = unique - {bg}
            if not remaining:
                continue

            assert _BLOCKER not in remaining
            assert len(remaining) == 1, "sprite cells must contain one monochrome sprite"
            sprite_color = remaining.pop()
            assert unique <= {bg, sprite_color}
            sprites.append(
                {
                    "cell": (row, col),
                    "color": int(sprite_color),
                    "patch": cell.copy(),
                    "pixels": int(np.sum(cell == sprite_color)),
                }
            )

    return {
        "divider": int(divider_color),
        "bg": int(bg),
        "row_segments": row_segments,
        "col_segments": col_segments,
        "blockers": blockers,
        "sprites": sprites,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each sprite forward on the divider grid's snake-ordered conveyor by its own pixel count.

    Input structure:
    - Full divider-color rows and columns partition the board into equal-sized cells.
    - A few cells contain a single monochrome sprite on a plain background; some cells are solid gray blockers.
    - All remaining cells are empty, and divider lines stay outside the cell interiors.

    Rule:
    - Read the cells in boustrophedon order: top row left-to-right, next row right-to-left, and so on.
    - Remove every solid gray blocker cell from that order to get the effective conveyor.
    - Each sprite advances forward by n effective cells, where n is its own number of colored pixels, while keeping the same patch inside its destination cell.
    - Clear the original sprite cells; keep divider lines and blocker cells unchanged.

    Key insight:
    - The motion is controlled by a one-dimensional snake walk over cells with blockers skipped, and each sprite's step size is encoded by its visible pixel count.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    info = _analyze_grid(g)
    row_segments = info["row_segments"]
    col_segments = info["col_segments"]
    blockers = info["blockers"]
    sprites = info["sprites"]
    bg = info["bg"]

    n_rows = len(row_segments)
    n_cols = len(col_segments)
    raw_order = _snake_cells(n_rows, n_cols)
    valid_order = [cell for cell in raw_order if cell not in blockers]
    valid_index = {cell: idx for idx, cell in enumerate(valid_order)}

    out = g.copy()
    for row, (r0, r1) in enumerate(row_segments):
        for col, (c0, c1) in enumerate(col_segments):
            if (row, col) not in blockers:
                out[r0 : r1 + 1, c0 : c1 + 1] = bg

    destinations: dict[tuple[int, int], dict] = {}
    for sprite in sprites:
        start = sprite["cell"]
        assert start in valid_index
        dest_idx = valid_index[start] + int(sprite["pixels"])
        assert dest_idx < len(valid_order), "sprite movement must stay on the conveyor"
        dest = valid_order[dest_idx]
        assert dest not in destinations, "sprites must land in distinct cells"
        destinations[dest] = sprite

    for (row, col), sprite in destinations.items():
        r0, r1 = row_segments[row]
        c0, c1 = col_segments[col]
        assert sprite["patch"].shape == (r1 - r0 + 1, c1 - c0 + 1)
        paste(out, sprite["patch"], r0, c0)

    return out


def _cell_top_left(row: int, col: int, cell_h: int, cell_w: int) -> tuple[int, int]:
    return 1 + row * (cell_h + 1), 1 + col * (cell_w + 1)


def _base_grid(n_rows: int, n_cols: int, cell_h: int, cell_w: int, divider: int) -> np.ndarray:
    h = n_rows * cell_h + (n_rows + 1)
    w = n_cols * cell_w + (n_cols + 1)
    grid = np.full((h, w), _BG, dtype=int)
    for r in range(0, h, cell_h + 1):
        grid[r, :] = divider
    for c in range(0, w, cell_w + 1):
        grid[:, c] = divider
    return grid


def _all_transforms(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[bytes] = set()
    for flip in (False, True):
        arr = np.fliplr(mask) if flip else mask
        for k in range(4):
            rotated = np.rot90(arr, k)
            key = rotated.tobytes()
            if key in seen:
                continue
            seen.add(key)
            variants.append(rotated.copy())
    return variants


def _make_sprite_patch(
    rng: np.random.Generator, cell_h: int, cell_w: int, color: int, pixels: int
) -> np.ndarray | None:
    masks: list[np.ndarray] = []
    for base in _SHAPES_BY_SIZE[pixels]:
        for variant in _all_transforms(base):
            h, w = map(int, variant.shape)
            if h <= cell_h and w <= cell_w:
                masks.append(variant)
    if not masks:
        return None

    mask = masks[int(rng.integers(0, len(masks)))]
    assert int(mask.sum()) == int(pixels)
    mh, mw = map(int, mask.shape)
    off_r = int(rng.integers(0, cell_h - mh + 1))
    off_c = int(rng.integers(0, cell_w - mw + 1))

    patch = np.full((cell_h, cell_w), _BG, dtype=int)
    sprite = np.where(mask.astype(bool), int(color), _BG)
    paste(patch, sprite, off_r, off_c)
    return patch


def _render_scene(
    n_rows: int,
    n_cols: int,
    cell_h: int,
    cell_w: int,
    divider: int,
    blockers: set[tuple[int, int]],
    sprites: list[dict],
    *,
    use_destinations: bool,
) -> np.ndarray:
    grid = _base_grid(n_rows, n_cols, cell_h, cell_w, divider)
    for row, col in blockers:
        r0, c0 = _cell_top_left(row, col, cell_h, cell_w)
        grid[r0 : r0 + cell_h, c0 : c0 + cell_w] = _BLOCKER
    for sprite in sprites:
        row, col = sprite["dest"] if use_destinations else sprite["start"]
        r0, c0 = _cell_top_left(row, col, cell_h, cell_w)
        paste(grid, sprite["patch"], r0, c0)
    return grid


def _sample_layout(rng: np.random.Generator) -> tuple[int, int, int, int] | None:
    for _ in range(40):
        n_rows = int(rng.integers(3, 5))
        n_cols = int(rng.integers(4, 6))
        max_h = min(5, (30 - (n_rows + 1)) // n_rows)
        max_w = min(5, (30 - (n_cols + 1)) // n_cols)
        if max_h < 4 or max_w < 4:
            continue
        cell_h = int(rng.integers(4, max_h + 1))
        cell_w = int(rng.integers(4, max_w + 1))
        return n_rows, n_cols, cell_h, cell_w
    return None


def _sample_blockers(
    rng: np.random.Generator, n_rows: int, n_cols: int
) -> set[tuple[int, int]] | None:
    cells = [(row, col) for row in range(n_rows) for col in range(n_cols)]
    total = len(cells)
    max_blockers = min(4, total - 5)
    if max_blockers < 1:
        return None

    for _ in range(80):
        count = int(rng.integers(1, max_blockers + 1))
        chosen = rng.choice(total, size=count, replace=False)
        blockers = {cells[int(idx)] for idx in np.atleast_1d(chosen)}
        if all(any((row, col) not in blockers for col in range(n_cols)) for row in range(n_rows)):
            return blockers
    return None


def _move_candidates(
    n_rows: int, n_cols: int, blockers: set[tuple[int, int]], max_step: int
) -> tuple[list[dict], list[tuple[int, int]]]:
    raw_snake = _snake_cells(n_rows, n_cols)
    valid_snake = [cell for cell in raw_snake if cell not in blockers]
    rowmajor_valid = [cell for cell in _rowmajor_cells(n_rows, n_cols) if cell not in blockers]

    raw_index = {cell: idx for idx, cell in enumerate(raw_snake)}
    rowmajor_index = {cell: idx for idx, cell in enumerate(rowmajor_valid)}

    candidates: list[dict] = []
    for start_idx, start in enumerate(valid_snake):
        for step in range(1, max_step + 1):
            dest_idx = start_idx + step
            if dest_idx >= len(valid_snake):
                continue
            dest = valid_snake[dest_idx]

            alt_rowmajor = None
            alt_idx = rowmajor_index[start] + step
            if alt_idx < len(rowmajor_valid):
                alt_rowmajor = rowmajor_valid[alt_idx]

            alt_raw = None
            raw_dest_idx = raw_index[start] + step
            if raw_dest_idx < len(raw_snake):
                alt_raw = raw_snake[raw_dest_idx]

            candidates.append(
                {
                    "start": start,
                    "dest": dest,
                    "step": int(step),
                    "snake_diff": alt_rowmajor is not None and dest != alt_rowmajor,
                    "blocker_diff": alt_raw is not None and dest != alt_raw,
                }
            )
    return candidates, valid_snake


def _select_moves(
    rng: np.random.Generator, n_rows: int, n_cols: int, blockers: set[tuple[int, int]], max_step: int
) -> list[dict] | None:
    candidates, valid_snake = _move_candidates(n_rows, n_cols, blockers, max_step)
    max_sprites = min(4, len(valid_snake) // 2)
    if max_sprites < 2 or not candidates:
        return None

    for _ in range(500):
        target = int(rng.integers(2, max_sprites + 1))
        chosen: list[dict] = []
        used_cells: set[tuple[int, int]] = set()
        has_snake_counterexample = False
        has_blocker_counterexample = False

        for idx in rng.permutation(len(candidates)):
            cand = candidates[int(idx)]
            if cand["start"] in used_cells or cand["dest"] in used_cells:
                continue
            chosen.append(cand)
            used_cells.add(cand["start"])
            used_cells.add(cand["dest"])
            has_snake_counterexample |= bool(cand["snake_diff"])
            has_blocker_counterexample |= bool(cand["blocker_diff"])
            if len(chosen) == target:
                break

        step_set = {int(c["step"]) for c in chosen}
        if (
            len(chosen) == target
            and has_snake_counterexample
            and has_blocker_counterexample
            and max(step_set) >= 3
            and len(step_set) >= min(2, target)
        ):
            return chosen
    return None


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    layout = _sample_layout(rng)
    if layout is None:
        return None
    n_rows, n_cols, cell_h, cell_w = layout

    blockers = _sample_blockers(rng, n_rows, n_cols)
    if blockers is None:
        return None

    max_step = min(max(_SHAPES_BY_SIZE), cell_h * cell_w - 1)
    moves = _select_moves(rng, n_rows, n_cols, blockers, max_step)
    if moves is None:
        return None

    divider = int(rng.choice(_DIVIDER_COLORS))
    sprite_colors = [color for color in _SPRITE_COLOR_POOL if color != divider]
    if len(sprite_colors) < len(moves):
        return None
    color_ids = rng.choice(sprite_colors, size=len(moves), replace=False)

    sprites: list[dict] = []
    for move, color in zip(moves, color_ids):
        patch = _make_sprite_patch(rng, cell_h, cell_w, int(color), int(move["step"]))
        if patch is None:
            return None
        sprites.append(
            {
                "start": move["start"],
                "dest": move["dest"],
                "step": int(move["step"]),
                "patch": patch,
                "color": int(color),
            }
        )

    inp = _render_scene(n_rows, n_cols, cell_h, cell_w, divider, blockers, sprites, use_destinations=False)
    expected = _render_scene(n_rows, n_cols, cell_h, cell_w, divider, blockers, sprites, use_destinations=True)

    if not any(move["snake_diff"] for move in moves):
        return None
    if not any(move["blocker_diff"] for move in moves):
        return None
    if np.array_equal(main(inp), expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid snake-conveyor input")
