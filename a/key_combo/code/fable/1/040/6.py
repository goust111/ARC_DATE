from __future__ import annotations

import importlib.util
import random
from collections import Counter
from pathlib import Path

import numpy as np


def _load_arc_utils():
    path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("_arc_utils_fable_040_6", path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ARC_UTILS = _load_arc_utils()
if _ARC_UTILS is not None:
    get_bg = _ARC_UTILS.get_bg
    crop_bbox = _ARC_UTILS.crop_bbox
else:
    def get_bg(grid: np.ndarray) -> int:
        return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])

    def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        r0, c0, r1, c1 = map(int, bbox)
        return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


BACKGROUND = 0
PALETTE = tuple(range(1, 10))
DIRECTIONS = ("up", "right", "down", "left")
ROT_TO_UP = {"up": 0, "right": 1, "down": 2, "left": 3}
ROT_FROM_UP = {"up": 0, "right": 3, "down": 2, "left": 1}
GAP_AT = {
    "up": lambda n: (0, n // 2),
    "right": lambda n: (n // 2, n - 1),
    "down": lambda n: (n - 1, n // 2),
    "left": lambda n: (n // 2, 0),
}

PAD = 1
SEP = 1
FRAME_SIZE_CHOICES = (5, 5, 7)
MAX_GROUP_SIZE = 16
MAX_ATTEMPTS = 500


def _runs(bits: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, value in enumerate(map(bool, bits)):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(bits) - 1))
    return runs


def _frame_origin(cell: tuple[int, int], frame_size: int) -> tuple[int, int]:
    return PAD + cell[0] * (frame_size + SEP), PAD + cell[1] * (frame_size + SEP)


def _canvas_shape(rows: int, cols: int, frame_size: int) -> tuple[int, int]:
    return (
        2 * PAD + rows * frame_size + (rows - 1) * SEP,
        2 * PAD + cols * frame_size + (cols - 1) * SEP,
    )


def _gap_direction(block: np.ndarray, color: int, bg: int) -> str:
    n = int(block.shape[0])
    assert block.shape == (n, n)
    assert n >= 5 and n % 2 == 1

    border = np.zeros((n, n), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    gap_coords = np.argwhere(border & (block == bg))
    assert gap_coords.shape[0] == 1, "expected exactly one border gap"
    gr, gc = map(int, gap_coords[0])

    for direction, locator in GAP_AT.items():
        if (gr, gc) == locator(n):
            filled_border = border.copy()
            filled_border[gr, gc] = False
            assert np.all(block[filled_border] == color), "frame border must be solid except for the gap"
            return direction
    raise AssertionError("gap must be centered on one side")


def _paint_frame(
    grid: np.ndarray,
    r0: int,
    c0: int,
    frame_size: int,
    color: int,
    direction: str,
    interior_mask: np.ndarray,
    bg: int = BACKGROUND,
) -> None:
    block = np.full((frame_size, frame_size), int(bg), dtype=int)
    block[0, :] = color
    block[-1, :] = color
    block[:, 0] = color
    block[:, -1] = color
    gap_r, gap_c = GAP_AT[direction](frame_size)
    block[gap_r, gap_c] = bg
    block[1:-1, 1:-1][interior_mask] = color
    grid[r0 : r0 + frame_size, c0 : c0 + frame_size] = block


def _parse_frames(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    row_runs = _runs(np.any(g != bg, axis=1))
    col_runs = _runs(np.any(g != bg, axis=0))
    assert row_runs and col_runs, "expected a framed lattice"

    frame_hs = {r1 - r0 + 1 for r0, r1 in row_runs}
    frame_ws = {c1 - c0 + 1 for c0, c1 in col_runs}
    assert len(frame_hs) == 1 and len(frame_ws) == 1
    frame_size = frame_hs.pop()
    assert frame_size == frame_ws.pop()
    assert frame_size >= 5 and frame_size % 2 == 1

    frames: list[dict] = []
    for r0, r1 in row_runs:
        for c0, c1 in col_runs:
            block = crop_bbox(g, (r0, c0, r1, c1))
            colors = set(map(int, np.unique(block[block != bg])))
            assert len(colors) == 1, "each frame block must use one non-background color"
            color = colors.pop()
            direction = _gap_direction(block, color, bg)
            frames.append(
                {
                    "r0": int(r0),
                    "c0": int(c0),
                    "color": int(color),
                    "direction": direction,
                    "interior": (block[1:-1, 1:-1] == color),
                }
            )

    return bg, frame_size, frames


def main(grid: np.ndarray) -> np.ndarray:
    """Complete each same-colored gapped frame with the union of its motif in the gap-up pose.

    Input structure:
    - The grid is a lattice of equal-size square frames separated by full background rows and columns.
    - Every frame uses one non-background color, has a one-cell-thick border, and has exactly one centered one-cell gap on one side.
    - Inside each frame there may be zero or more pixels of the frame color, and same-colored frames may face different directions.

    Rule:
    - Group frames by color.
    - For each frame, rotate its interior so that its border gap would point upward.
    - Take the union of those rotated interior masks within each color group.
    - Rotate that union back to each frame's own direction and repaint only the interior; keep the border gap unchanged.

    Key insight:
    - The missing border cell is an orientation marker, so same-colored motifs must be aligned by direction before sharing pixels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg, frame_size, frames = _parse_frames(g)
    interior_size = frame_size - 2

    unions: dict[int, np.ndarray] = {}
    for frame in frames:
        canonical = np.rot90(frame["interior"], ROT_TO_UP[frame["direction"]])
        color = frame["color"]
        if color not in unions:
            unions[color] = np.zeros((interior_size, interior_size), dtype=bool)
        unions[color] |= canonical

    out = g.copy()
    for frame in frames:
        color = frame["color"]
        direction = frame["direction"]
        rotated = np.rot90(unions[color], ROT_FROM_UP[direction])
        _paint_frame(out, frame["r0"], frame["c0"], frame_size, color, direction, rotated, bg)
    return out


def _naive_absolute_union_output(grid: np.ndarray) -> np.ndarray:
    bg, frame_size, frames = _parse_frames(grid)
    interior_size = frame_size - 2

    unions: dict[int, np.ndarray] = {}
    for frame in frames:
        color = frame["color"]
        if color not in unions:
            unions[color] = np.zeros((interior_size, interior_size), dtype=bool)
        unions[color] |= frame["interior"]

    out = np.asarray(grid, dtype=int).copy()
    for frame in frames:
        _paint_frame(
            out,
            frame["r0"],
            frame["c0"],
            frame_size,
            frame["color"],
            frame["direction"],
            unions[frame["color"]],
            bg,
        )
    return out


def _all_layouts(frame_size: int) -> list[tuple[int, int]]:
    layouts = []
    for rows in range(2, 5):
        for cols in range(2, 5):
            h, w = _canvas_shape(rows, cols, frame_size)
            if h <= 30 and w <= 30:
                layouts.append((rows, cols))
    return layouts


def _choose_group_sizes(total_frames: int) -> list[int]:
    n_colors = random.randint(2, min(5, total_frames - 1))
    sizes = [1] * n_colors
    remaining = total_frames - n_colors

    min_repeat = 2 if remaining >= 2 and n_colors >= 3 and total_frames >= 6 else 1
    max_repeat = min(n_colors, remaining)
    repeat_groups = random.randint(min_repeat, max_repeat)
    repeated = random.sample(range(n_colors), repeat_groups)
    for idx in repeated:
        sizes[idx] += 1
    remaining -= repeat_groups

    while remaining:
        expandable = [idx for idx, size in enumerate(sizes) if size < MAX_GROUP_SIZE]
        idx = random.choice(expandable)
        sizes[idx] += 1
        remaining -= 1

    random.shuffle(sizes)
    return sizes


def _is_rotation_asymmetric(mask: np.ndarray) -> bool:
    return all(not np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3))


def _random_motif(interior_size: int, *, allow_empty: bool, force_asymmetric: bool) -> np.ndarray:
    total = interior_size * interior_size
    low = 0 if allow_empty else 2
    high = min(total - 1, max(2, total // 2 + 1))

    for _ in range(200):
        fill = random.randint(low, high)
        motif = np.zeros((interior_size, interior_size), dtype=bool)
        if fill:
            motif.flat[random.sample(range(total), fill)] = True
        if force_asymmetric and not _is_rotation_asymmetric(motif):
            continue
        return motif
    raise RuntimeError("failed to sample a valid motif")


def _assign_cells(rows: int, cols: int, colors: list[int], sizes: list[int]):
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    random.shuffle(cells)

    frames_by_color: dict[int, list[tuple[int, int]]] = {}
    assignment: dict[tuple[int, int], int] = {}
    cursor = 0
    for color, size in zip(colors, sizes):
        picked = cells[cursor : cursor + size]
        cursor += size
        frames_by_color[color] = picked
        for cell in picked:
            assignment[cell] = color
    return frames_by_color, assignment


def _assign_orientations(
    assignment: dict[tuple[int, int], int],
    frames_by_color: dict[int, list[tuple[int, int]]],
    critical_color: int,
) -> dict[tuple[int, int], str]:
    orientations: dict[tuple[int, int], str] = {}
    all_cells = list(assignment)

    diverse_cells = random.sample(all_cells, k=min(len(all_cells), 4))
    diverse_dirs = random.sample(DIRECTIONS, k=len(diverse_cells))
    for cell, direction in zip(diverse_cells, diverse_dirs):
        orientations[cell] = direction

    critical_cells = frames_by_color[critical_color]
    critical_dirs = random.sample(DIRECTIONS, k=min(len(critical_cells), 4))
    for cell, direction in zip(critical_cells, critical_dirs):
        orientations[cell] = direction

    if len(critical_cells) > 4:
        for cell in critical_cells[4:]:
            orientations[cell] = random.choice(DIRECTIONS)

    for cell in all_cells:
        orientations.setdefault(cell, random.choice(DIRECTIONS))
    return orientations


def _split_motif(full_mask: np.ndarray, copies: int) -> list[np.ndarray]:
    parts = [np.zeros_like(full_mask, dtype=bool) for _ in range(copies)]
    coords = [tuple(map(int, rc)) for rc in np.argwhere(full_mask)]
    if not coords:
        return parts

    for coord in coords:
        cover = 1 if copies == 1 else random.randint(1, min(2, copies))
        for idx in random.sample(range(copies), cover):
            parts[idx][coord] = True

    if copies > 1 and len(coords) > 1 and all(np.array_equal(part, full_mask) for part in parts):
        removable = [
            coord for coord in coords if sum(int(part[coord]) for part in parts) > 1
        ]
        if removable:
            idx = random.randrange(copies)
            parts[idx][removable[0]] = False
    return parts


def _build_grids(
    rows: int,
    cols: int,
    frame_size: int,
    assignment: dict[tuple[int, int], int],
    frames_by_color: dict[int, list[tuple[int, int]]],
    orientations: dict[tuple[int, int], str],
    canonical_motifs: dict[int, np.ndarray],
):
    h, w = _canvas_shape(rows, cols, frame_size)
    inp = np.full((h, w), BACKGROUND, dtype=int)
    out = np.full((h, w), BACKGROUND, dtype=int)

    subsets_by_cell: dict[tuple[int, int], np.ndarray] = {}
    for color, cells in frames_by_color.items():
        parts = _split_motif(canonical_motifs[color], len(cells))
        for cell, part in zip(cells, parts):
            subsets_by_cell[cell] = part

    for cell, color in assignment.items():
        direction = orientations[cell]
        r0, c0 = _frame_origin(cell, frame_size)
        in_mask = np.rot90(subsets_by_cell[cell], ROT_FROM_UP[direction])
        out_mask = np.rot90(canonical_motifs[color], ROT_FROM_UP[direction])
        _paint_frame(inp, r0, c0, frame_size, color, direction, in_mask, BACKGROUND)
        _paint_frame(out, r0, c0, frame_size, color, direction, out_mask, BACKGROUND)

    return inp, out


def _gen_attempt():
    frame_size = random.choice(FRAME_SIZE_CHOICES)
    rows, cols = random.choice(_all_layouts(frame_size))
    total_frames = rows * cols

    sizes = _choose_group_sizes(total_frames)
    colors = random.sample(PALETTE, len(sizes))
    frames_by_color, assignment = _assign_cells(rows, cols, colors, sizes)

    repeated_colors = [color for color, cells in frames_by_color.items() if len(cells) > 1]
    assert repeated_colors
    critical_color = random.choice(repeated_colors)

    interior_size = frame_size - 2
    canonical_motifs: dict[int, np.ndarray] = {}
    for color in colors:
        repeated = len(frames_by_color[color]) > 1
        canonical_motifs[color] = _random_motif(
            interior_size,
            allow_empty=not repeated and random.random() < 0.35,
            force_asymmetric=(color == critical_color) or (repeated and random.random() < 0.7),
        )

    orientations = _assign_orientations(assignment, frames_by_color, critical_color)
    inp, out = _build_grids(
        rows,
        cols,
        frame_size,
        assignment,
        frames_by_color,
        orientations,
        canonical_motifs,
    )

    if get_bg(inp) != BACKGROUND:
        return None
    if np.array_equal(inp, out):
        return None
    if len({orientations[cell] for cell in frames_by_color[critical_color]}) < 2:
        return None
    if not np.array_equal(main(inp), out):
        return None
    if np.array_equal(_naive_absolute_union_output(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {MAX_ATTEMPTS} attempts")
