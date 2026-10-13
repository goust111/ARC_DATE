from __future__ import annotations

import random

import numpy as np


SIDE_TO_VECTOR = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
SIDES = tuple(SIDE_TO_VECTOR)

BG = 0
GRID_MIN = 14
GRID_MAX = 30
FRAME_COUNT_CHOICES = (1, 2, 3, 4)
FRAME_COUNT_WEIGHTS = (2, 3, 2, 1)
INTERIOR_H_RANGE = (2, 5)
INTERIOR_W_RANGE = (2, 5)
MAX_SEEDS = 8
PLACE_RETRIES = 120
GEN_RETRIES = 400


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _boundary_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _detect_frame(
    grid: np.ndarray,
    cells: list[tuple[int, int]],
    color: int,
    bg: int,
) -> dict[str, object] | None:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return None

    boundary = _boundary_cells(r0, c0, r1, c1)
    cell_set = set(cells)
    if len(cell_set) != len(boundary) - 1:
        return None
    if not cell_set.issubset(boundary):
        return None

    missing = list(boundary - cell_set)
    if len(missing) != 1:
        return None
    gap_r, gap_c = missing[0]
    if int(grid[gap_r, gap_c]) != bg:
        return None
    if (gap_r in (r0, r1)) and (gap_c in (c0, c1)):
        return None

    if gap_r == r0:
        side = "top"
    elif gap_r == r1:
        side = "bottom"
    elif gap_c == c0:
        side = "left"
    else:
        side = "right"

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    seed_local = np.argwhere(interior != bg)
    if seed_local.size == 0:
        return None
    seed_colors = {int(interior[r, c]) for r, c in seed_local}
    if len(seed_colors) != 1:
        return None
    seed_color = next(iter(seed_colors))
    if seed_color == color:
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "gap": (gap_r, gap_c),
        "side": side,
        "seed_color": seed_color,
        "seed_count": int(seed_local.shape[0]),
    }


def _ray_cells(
    gap: tuple[int, int],
    side: str,
    length: int,
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = shape
    dr, dc = SIDE_TO_VECTOR[side]
    r, c = gap
    cells: list[tuple[int, int]] = []
    for _ in range(length):
        if not (0 <= r < h and 0 <= c < w):
            break
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shoot outward rays through rectangular frame gaps, with length set by interior seed count.

    Input structure:
    - The grid has a dominant background color and one or more 1-pixel-thick rectangular frames.
    - Each frame is monochrome, has exactly one missing border cell on one side, and encloses a background interior.
    - Inside each frame, all non-background seed pixels share one color that differs from the frame color.

    Rule:
    - Detect every monochrome component whose cells form a rectangle border with exactly one missing side cell.
    - For each such frame, locate the gap side, count the enclosed seed pixels, and read their shared color.
    - Starting at the gap cell itself, draw a straight outward ray in the gap direction for exactly that many cells, truncated by the grid boundary.

    Key insight:
    - The transformation couples two separate cues from the same frame: the missing border pixel encodes direction, while the number of enclosed seed pixels encodes ray length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _dominant_color(g)
    out = g.copy()

    frames: list[dict[str, object]] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        for comp in _components(g == color):
            frame = _detect_frame(g, comp, color, bg)
            if frame is not None:
                frames.append(frame)

    assert frames, "expected at least one gapped rectangular frame"
    frames.sort(key=lambda item: tuple(int(v) for v in item["bbox"]))  # deterministic but not semantically important

    for frame in frames:
        gap = frame["gap"]
        side = str(frame["side"])
        seed_count = int(frame["seed_count"])
        seed_color = int(frame["seed_color"])
        for r, c in _ray_cells(gap, side, seed_count, g.shape):
            out[r, c] = seed_color

    return out


def _available_from_gap(gap: tuple[int, int], side: str, shape: tuple[int, int]) -> int:
    h, w = shape
    r, c = gap
    if side == "top":
        return r + 1
    if side == "bottom":
        return h - r
    if side == "left":
        return c + 1
    return w - c


def _box_buffer_cells(
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    shape: tuple[int, int],
    pad: int = 1,
) -> set[tuple[int, int]]:
    h, w = shape
    cells: set[tuple[int, int]] = set()
    for r in range(max(0, r0 - pad), min(h - 1, r1 + pad) + 1):
        for c in range(max(0, c0 - pad), min(w - 1, c1 + pad) + 1):
            cells.add((r, c))
    return cells


def _point_buffer_cells(
    points: list[tuple[int, int]],
    shape: tuple[int, int],
    pad: int = 1,
) -> set[tuple[int, int]]:
    h, w = shape
    cells: set[tuple[int, int]] = set()
    for r, c in points:
        for rr in range(max(0, r - pad), min(h - 1, r + pad) + 1):
            for cc in range(max(0, c - pad), min(w - 1, c + pad) + 1):
                cells.add((rr, cc))
    return cells


def _render_input(shape: tuple[int, int], border_color: int, frames: list[dict[str, object]]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        gap = tuple(frame["gap"])
        seed_color = int(frame["seed_color"])
        seed_positions = list(frame["seed_positions"])
        for r, c in _boundary_cells(r0, c0, r1, c1):
            if (r, c) != gap:
                grid[r, c] = border_color
        for r, c in seed_positions:
            grid[r, c] = seed_color
    return grid


def _render_expected(inp: np.ndarray, frames: list[dict[str, object]]) -> np.ndarray:
    out = inp.copy()
    for frame in frames:
        gap = tuple(frame["gap"])
        side = str(frame["side"])
        seed_color = int(frame["seed_color"])
        seed_count = len(frame["seed_positions"])
        for r, c in _ray_cells(gap, side, seed_count, inp.shape):
            out[r, c] = seed_color
    return out


def _truncation_flags(count: int) -> list[bool]:
    if count == 1:
        return [bool(random.getrandbits(1))]
    flags = [False] * (count // 2) + [True] * (count // 2)
    if count % 2:
        flags.append(bool(random.getrandbits(1)))
    random.shuffle(flags)
    return flags


def _sample_position(
    side: str,
    truncated: bool,
    seed_count: int,
    box_h: int,
    box_w: int,
    h: int,
    w: int,
) -> tuple[int, int] | None:
    max_r0 = h - box_h
    max_c0 = w - box_w
    if max_r0 < 0 or max_c0 < 0:
        return None

    if side == "top":
        row_lo = 0 if truncated else max(0, seed_count - 1)
        row_hi = min(max_r0, seed_count - 2) if truncated else max_r0
        if row_lo > row_hi:
            return None
        return random.randint(row_lo, row_hi), random.randint(0, max_c0)

    if side == "bottom":
        row_lo = max(0, max_r0 - seed_count + 2) if truncated else 0
        row_hi = max_r0 if truncated else min(max_r0, max_r0 - seed_count + 1)
        if row_lo > row_hi:
            return None
        return random.randint(row_lo, row_hi), random.randint(0, max_c0)

    if side == "left":
        col_lo = 0 if truncated else max(0, seed_count - 1)
        col_hi = min(max_c0, seed_count - 2) if truncated else max_c0
        if col_lo > col_hi:
            return None
        return random.randint(0, max_r0), random.randint(col_lo, col_hi)

    col_lo = max(0, max_c0 - seed_count + 2) if truncated else 0
    col_hi = max_c0 if truncated else min(max_c0, max_c0 - seed_count + 1)
    if col_lo > col_hi:
        return None
    return random.randint(0, max_r0), random.randint(col_lo, col_hi)


def _place_frame(
    shape: tuple[int, int],
    border_color: int,
    seed_palette: list[int],
    reserved: np.ndarray,
    side: str,
    truncated: bool,
) -> dict[str, object] | None:
    h, w = shape
    del border_color  # kept explicit in the signature because it is a generation role.

    for _ in range(PLACE_RETRIES):
        inner_h = random.randint(*INTERIOR_H_RANGE)
        inner_w = random.randint(*INTERIOR_W_RANGE)
        box_h = inner_h + 2
        box_w = inner_w + 2
        max_seed_count = min(inner_h * inner_w, MAX_SEEDS)
        if max_seed_count < 1:
            continue
        seed_count = random.randint(1, max_seed_count)
        pos = _sample_position(side, truncated, seed_count, box_h, box_w, h, w)
        if pos is None:
            continue
        r0, c0 = pos
        r1, c1 = r0 + box_h - 1, c0 + box_w - 1

        if side in ("top", "bottom"):
            gap = (r0 if side == "top" else r1, random.randint(c0 + 1, c1 - 1))
        else:
            gap = (random.randint(r0 + 1, r1 - 1), c0 if side == "left" else c1)

        available = _available_from_gap(gap, side, shape)
        if truncated != (seed_count > available):
            continue

        ray_cells = _ray_cells(gap, side, min(seed_count, available), shape)
        candidate_buffer = _box_buffer_cells(r0, c0, r1, c1, shape) | _point_buffer_cells(ray_cells, shape)
        if any(bool(reserved[r, c]) for r, c in candidate_buffer):
            continue

        seed_positions = random.sample(
            [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)],
            seed_count,
        )
        seed_color = random.choice(seed_palette)

        for r, c in candidate_buffer:
            reserved[r, c] = True

        return {
            "bbox": (r0, c0, r1, c1),
            "gap": gap,
            "side": side,
            "seed_color": seed_color,
            "seed_positions": seed_positions,
        }

    return None


def _generate_once() -> np.ndarray | None:
    frame_count = random.choices(FRAME_COUNT_CHOICES, weights=FRAME_COUNT_WEIGHTS, k=1)[0]
    min_dim = min(GRID_MAX, GRID_MIN + 3 * frame_count)
    h = random.randint(min_dim, GRID_MAX)
    w = random.randint(min_dim, GRID_MAX)
    border_color = random.randint(1, 9)
    seed_palette = [color for color in range(1, 10) if color != border_color]

    reserved = np.zeros((h, w), dtype=bool)
    sides = random.sample(SIDES, frame_count)
    trunc_flags = _truncation_flags(frame_count)
    frames: list[dict[str, object]] = []

    for side, truncated in zip(sides, trunc_flags):
        frame = _place_frame((h, w), border_color, seed_palette, reserved, side, truncated)
        if frame is None:
            return None
        frames.append(frame)

    inp = _render_input((h, w), border_color, frames)
    counts = np.bincount(inp.ravel(), minlength=10)
    if counts[BG] <= max(int(counts[color]) for color in range(1, 10)):
        return None

    expected = _render_expected(inp, frames)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
