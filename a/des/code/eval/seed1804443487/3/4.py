from __future__ import annotations

import random

import numpy as np


_SIDES = ("up", "right", "down", "left")
_OPPOSITE = {"up": "down", "right": "left", "down": "up", "left": "right"}
_ROT_K = {"up": 0, "right": 3, "down": 2, "left": 1}

_GEN_BG = 0
_GEN_BORDER = 1
_BOX_CHOICES = (5, 7, 9)
_REGION_COUNT_MIN = 2
_REGION_COUNT_MAX = 4
_PATTERN_COLORS = (2, 3, 4, 5, 6, 7, 8, 9)
_PATTERN_SHAPES = (
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[0, 1], [1, 1], [1, 0]], dtype=bool),
)
_LAYOUT_RETRIES = 200
_GEN_RETRIES = 300


def _majority_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _detect_layout(grid: np.ndarray, bg: int) -> tuple[int, int, int, int, int]:
    h, w = grid.shape
    best: tuple[tuple[int, int, int], int, int, int] | None = None

    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        row_hits = [
            r
            for r in range(h)
            if np.all((grid[r] == color) | (grid[r] == bg))
            and int(np.count_nonzero(grid[r] == color)) > w // 2
        ]
        col_hits = [
            c
            for c in range(w)
            if np.all((grid[:, c] == color) | (grid[:, c] == bg))
            and int(np.count_nonzero(grid[:, c] == color)) > h // 2
        ]
        if len(row_hits) < 4 or len(col_hits) < 4:
            continue

        box_h = int(max(np.diff(row_hits))) + 1
        box_w = int(max(np.diff(col_hits))) + 1
        if box_h < 5 or box_w < 5 or box_h != box_w:
            continue
        if h % box_h != 0 or w % box_w != 0:
            continue

        expected_rows = [r for r in range(h) if r % box_h in (0, box_h - 1)]
        expected_cols = [c for c in range(w) if c % box_w in (0, box_w - 1)]
        if row_hits != expected_rows or col_hits != expected_cols:
            continue

        score = (len(row_hits), len(col_hits), int(np.count_nonzero(grid == color)))
        if best is None or score > best[0]:
            best = (score, color, box_h, box_w)

    assert best is not None, "failed to detect the partition grid"
    _, border, box_h, box_w = best
    return int(border), int(box_h), int(box_w), h // box_h, w // box_w


def _region_bounds(row: int, col: int, box: int) -> tuple[int, int, int, int]:
    top = row * box
    left = col * box
    return top, left, top + box - 1, left + box - 1


def _gap_side(grid: np.ndarray, top: int, left: int, box: int, bg: int) -> str:
    bottom = top + box - 1
    right = left + box - 1
    mid = box // 2
    checks = {
        "up": (top, left + mid),
        "right": (top + mid, right),
        "down": (bottom, left + mid),
        "left": (top + mid, left),
    }
    missing = [side for side, (r, c) in checks.items() if int(grid[r, c]) == bg]
    assert len(missing) == 1, "every region must expose exactly one gap"
    return missing[0]


def _rotate_patch(patch: np.ndarray, side: str) -> np.ndarray:
    return np.rot90(patch, _ROT_K[side])


def _edge_is_open(side_a: str, side_b: str, dr: int, dc: int) -> bool:
    if dr == -1 and dc == 0:
        return side_a == "up" or side_b == "down"
    if dr == 1 and dc == 0:
        return side_a == "down" or side_b == "up"
    if dr == 0 and dc == -1:
        return side_a == "left" or side_b == "right"
    if dr == 0 and dc == 1:
        return side_a == "right" or side_b == "left"
    raise ValueError((dr, dc))


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _majority_color(g)
    border, box_h, box_w, n_rows, n_cols = _detect_layout(g, bg)
    assert box_h == box_w, "regions must be square to support quarter-turn rotation"
    box = box_h
    inner = box - 2

    gap_dirs: list[list[str]] = [[""] * n_cols for _ in range(n_rows)]
    rotated_originals: list[list[np.ndarray | None]] = [[None] * n_cols for _ in range(n_rows)]
    is_original = np.zeros((n_rows, n_cols), dtype=bool)

    for r in range(n_rows):
        for c in range(n_cols):
            top, left, bottom, right = _region_bounds(r, c, box)
            patch = g[top + 1 : bottom, left + 1 : right].copy()
            patch[patch == border] = bg
            gap = _gap_side(g, top, left, box, bg)
            gap_dirs[r][c] = gap
            if np.any(patch != bg):
                is_original[r, c] = True
                rotated_originals[r][c] = _rotate_patch(patch, gap)

    out = g.copy()
    for r in range(n_rows):
        for c in range(n_cols):
            top, left, bottom, right = _region_bounds(r, c, box)
            out[top + 1 : bottom, left + 1 : right] = bg

    for r in range(n_rows):
        for c in range(n_cols):
            top, left, bottom, right = _region_bounds(r, c, box)
            if is_original[r, c]:
                out[top + 1 : bottom, left + 1 : right] = rotated_originals[r][c]
                continue

            combined = np.full((inner, inner), bg, dtype=int)
            sources: list[tuple[int, int, np.ndarray]] = []
            for dr, dc in ((0, -1), (-1, 0), (1, 0), (0, 1)):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < n_rows and 0 <= nc < n_cols):
                    continue
                if not is_original[nr, nc]:
                    continue
                if not _edge_is_open(gap_dirs[r][c], gap_dirs[nr][nc], dr, dc):
                    continue
                sources.append((nr, nc, rotated_originals[nr][nc]))

            # The prompt specifies left-to-right, then top-to-bottom overlay order.
            for _, _, src in sorted(sources, key=lambda item: (item[1], item[0])):
                mask = src != bg
                combined[mask] = src[mask]

            if sources:
                combined = _rotate_patch(combined, gap_dirs[r][c])
            out[top + 1 : bottom, left + 1 : right] = combined

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each region's seed pattern from its border gap, then copy those
    rotated patterns across gapped shared edges into empty neighbors and rotate
    them again by the target region's own gap direction.
    """
    return _solve(grid)


def _make_pattern_patch(inner: int, bg: int, color: int, rng: random.Random) -> np.ndarray:
    shape = rng.choice(_PATTERN_SHAPES)
    sh, sw = shape.shape
    r0 = rng.randint(0, inner - sh)
    c0 = rng.randint(0, inner - sw)
    patch = np.full((inner, inner), bg, dtype=int)
    patch[r0 : r0 + sh, c0 : c0 + sw][shape] = color
    return patch


def _count_incoming(
    gap_dirs: list[list[str]], has_pattern: np.ndarray, row: int, col: int
) -> list[tuple[int, int]]:
    n_rows, n_cols = has_pattern.shape
    incoming: list[tuple[int, int]] = []
    for dr, dc in ((0, -1), (-1, 0), (1, 0), (0, 1)):
        nr, nc = row + dr, col + dc
        if not (0 <= nr < n_rows and 0 <= nc < n_cols):
            continue
        if not has_pattern[nr, nc]:
            continue
        if _edge_is_open(gap_dirs[row][col], gap_dirs[nr][nc], dr, dc):
            incoming.append((nr, nc))
    return incoming


def _render_input(
    box: int,
    gap_dirs: list[list[str]],
    originals: list[list[np.ndarray | None]],
    bg: int,
    border: int,
) -> np.ndarray:
    n_rows = len(gap_dirs)
    n_cols = len(gap_dirs[0])
    grid = np.full((n_rows * box, n_cols * box), bg, dtype=int)
    mid = box // 2

    for r in range(n_rows):
        for c in range(n_cols):
            top, left, bottom, right = _region_bounds(r, c, box)
            grid[top, left : right + 1] = border
            grid[bottom, left : right + 1] = border
            grid[top : bottom + 1, left] = border
            grid[top : bottom + 1, right] = border

            side = gap_dirs[r][c]
            if side == "up":
                grid[top, left + mid] = bg
            elif side == "right":
                grid[top + mid, right] = bg
            elif side == "down":
                grid[bottom, left + mid] = bg
            else:
                grid[top + mid, left] = bg

            patch = originals[r][c]
            if patch is not None:
                grid[top + 1 : bottom, left + 1 : right] = patch

    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    box = rng.choice(_BOX_CHOICES)
    inner = box - 2
    n_rows = rng.randint(_REGION_COUNT_MIN, _REGION_COUNT_MAX)
    n_cols = rng.randint(_REGION_COUNT_MIN, _REGION_COUNT_MAX)
    want_multi = rng.random() < 0.5

    for _ in range(_LAYOUT_RETRIES):
        gap_dirs = [
            [rng.choice(_SIDES) for _ in range(n_cols)]
            for _ in range(n_rows)
        ]

        density = rng.uniform(0.35, 0.6)
        has_pattern = np.array(
            [[rng.random() < density for _ in range(n_cols)] for _ in range(n_rows)],
            dtype=bool,
        )
        if not has_pattern.any() or has_pattern.all():
            continue

        single_targets = 0
        multi_targets = 0
        receiving_targets = 0
        for r in range(n_rows):
            for c in range(n_cols):
                if has_pattern[r, c]:
                    continue
                incoming = _count_incoming(gap_dirs, has_pattern, r, c)
                if incoming:
                    receiving_targets += 1
                if len(incoming) == 1:
                    single_targets += 1
                if len(incoming) >= 2:
                    multi_targets += 1

        if receiving_targets == 0:
            continue
        if want_multi and multi_targets == 0:
            continue
        if not want_multi and single_targets == 0:
            continue

        palette = list(_PATTERN_COLORS)
        rng.shuffle(palette)
        n_palette = rng.randint(2, min(4, len(palette)))
        palette = palette[:n_palette]

        originals: list[list[np.ndarray | None]] = [[None] * n_cols for _ in range(n_rows)]
        for r in range(n_rows):
            for c in range(n_cols):
                if not has_pattern[r, c]:
                    continue
                color = rng.choice(palette)
                originals[r][c] = _make_pattern_patch(inner, _GEN_BG, color, rng)

        grid = _render_input(box, gap_dirs, originals, _GEN_BG, _GEN_BORDER)
        try:
            out = _solve(grid)
        except AssertionError:
            continue

        if np.array_equal(out, grid):
            continue
        if int(np.count_nonzero(out != grid)) < 3:
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    """
    Create a regular framed-region grid with one gap per region and a mix of
    seeded and empty cells. The generated layouts always admit at least one
    non-trivial propagation from a patterned region into an empty region.
    """
    rng = random.Random()
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
