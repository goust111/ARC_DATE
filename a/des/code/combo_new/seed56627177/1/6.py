import random
from collections import Counter, deque

import numpy as np


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _describe_frame(grid: np.ndarray, bg: int, comp: list[tuple[int, int]]) -> dict[str, int] | None:
    coords = np.asarray(comp, dtype=int)
    rs = coords[:, 0]
    cs = coords[:, 1]
    r0, r1 = int(rs.min()), int(rs.max())
    c0, c1 = int(cs.min()), int(cs.max())
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return None

    perimeter_area = 2 * (height + width) - 4
    if len(comp) != perimeter_area:
        return None
    if any(r not in (r0, r1) and c not in (c0, c1) for r, c in comp):
        return None

    for c in range(c0, c1 + 1):
        if int(grid[r0, c]) == bg or int(grid[r1, c]) == bg:
            return None
    for r in range(r0, r1 + 1):
        if int(grid[r, c0]) == bg or int(grid[r, c1]) == bg:
            return None

    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    if not np.any(interior != bg):
        return None

    perimeter_colors = Counter(int(grid[r, c]) for r, c in comp)
    if len(perimeter_colors) != 2:
        return None
    marker_color, marker_count = min(perimeter_colors.items(), key=lambda item: item[1])
    border_color, border_count = max(perimeter_colors.items(), key=lambda item: item[1])
    if marker_color == border_color or marker_count not in (1, 2, 3) or border_count <= marker_count:
        return None

    return {
        "r0": r0,
        "c0": c0,
        "r1": r1,
        "c1": c1,
        "k": int(marker_count),
    }


def _find_frames(grid: np.ndarray, bg: int) -> list[dict[str, int]]:
    frames: list[dict[str, int]] = []
    for comp in components_mask(grid != bg):
        frame = _describe_frame(grid, bg, comp)
        if frame is not None:
            frames.append(frame)
    frames.sort(key=lambda info: (info["r0"], info["c0"]))
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each enclosed inner pattern clockwise by the number of marker pixels on its surrounding frame.

    Input structure:
    - The background is black, with several disjoint one-pixel rectangular frames.
    - Each frame perimeter uses one main border color plus 1-3 marker pixels in a second color placed on the border.
    - Inside every frame sits a smaller asymmetric pattern separated from the border by background cells.

    Rule:
    - Detect each rectangular frame as a connected perimeter component enclosing a non-empty interior.
    - Count the minority-color border pixels on that frame; this count K is always 1, 2, or 3.
    - Keep the frame unchanged, but rotate the entire interior subgrid clockwise by K quarter-turns in place.

    Key insight:
    - The marker pixels encode only a count, not a direction; the count directly chooses how many 90-degree clockwise rotations to apply to the enclosed pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    out = g.copy()
    for frame in _find_frames(g, bg):
        r0, c0, r1, c1, k = frame["r0"], frame["c0"], frame["r1"], frame["c1"], frame["k"]
        inner = out[r0 + 1 : r1, c0 + 1 : c1].copy()
        assert inner.shape[0] == inner.shape[1], "90-degree in-place rotation requires a square frame interior"
        out[r0 + 1 : r1, c0 + 1 : c1] = np.rot90(inner, -k)
    return out


# --------------- generator ---------------

BG_COLOR = 0
GRID_SIDE_RANGE = (12, 30)
FRAME_COUNT_RANGE = (1, 4)
INNER_SIDE_RANGE = (5, 8)
COLOR_POOL = tuple(range(1, 10))
MAX_PATTERN_CELLS = 7
PATTERN_RETRIES = 120
GENERATION_RETRIES = 400


def _marker_counts_for_sample(rng: random.Random, n_frames: int) -> list[int]:
    counts = [rng.randint(1, 3) for _ in range(n_frames)]
    if n_frames >= 3 and rng.random() < 0.7:
        counts[:3] = [1, 2, 3]
        rng.shuffle(counts)
    return counts


def _random_connected_mask(side: int, n_cells: int, rng: random.Random) -> np.ndarray | None:
    start = (rng.randrange(side), rng.randrange(side))
    cells = {start}
    while len(cells) < n_cells:
        frontier = []
        for r, c in cells:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < side and 0 <= nc < side and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(rng.choice(frontier))

    mask = np.zeros((side, side), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _make_pattern_patch(inner_side: int, color: int, rng: random.Random) -> np.ndarray | None:
    core_side = inner_side - 2
    max_cells = min(MAX_PATTERN_CELLS, core_side * core_side - 1)
    min_cells = 3 if core_side <= 3 else 4
    if max_cells < min_cells:
        return None

    for _ in range(PATTERN_RETRIES):
        n_cells = rng.randint(min_cells, max_cells)
        mask = _random_connected_mask(core_side, n_cells, rng)
        if mask is None:
            continue
        if any(np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3)):
            continue
        mask = np.rot90(mask, rng.randrange(4))
        patch = np.zeros((inner_side, inner_side), dtype=int)
        core = patch[1:-1, 1:-1]
        core[mask] = int(color)
        return patch
    return None


def _frame_candidates(outer_side: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(1, outer_side - 1):
        cells.append((0, c))
        cells.append((outer_side - 1, c))
    for r in range(1, outer_side - 1):
        cells.append((r, 0))
        cells.append((r, outer_side - 1))
    return cells


def _place_frames(h: int, w: int, outer_sides: list[int], rng: random.Random) -> list[tuple[int, int]] | None:
    occupied = np.zeros((h, w), dtype=bool)
    placements: list[tuple[int, int]] = []
    for outer_side in outer_sides:
        candidates: list[tuple[int, int]] = []
        for r0 in range(1, h - outer_side):
            for c0 in range(1, w - outer_side):
                r1 = r0 + outer_side - 1
                c1 = c0 + outer_side - 1
                rr0 = max(0, r0 - 1)
                cc0 = max(0, c0 - 1)
                rr1 = min(h, r1 + 2)
                cc1 = min(w, c1 + 2)
                if not occupied[rr0:rr1, cc0:cc1].any():
                    candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = rng.choice(candidates)
        placements.append((r0, c0))
        r1 = r0 + outer_side - 1
        c1 = c0 + outer_side - 1
        occupied[max(0, r0 - 1) : min(h, r1 + 2), max(0, c0 - 1) : min(w, c1 + 2)] = True
    return placements


def _draw_frame(grid: np.ndarray, r0: int, c0: int, outer_side: int, border_color: int) -> None:
    r1 = r0 + outer_side - 1
    c1 = c0 + outer_side - 1
    grid[r0, c0 : c1 + 1] = int(border_color)
    grid[r1, c0 : c1 + 1] = int(border_color)
    grid[r0 : r1 + 1, c0] = int(border_color)
    grid[r0 : r1 + 1, c1] = int(border_color)


def _paste_pattern(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    view = grid[r0 : r0 + h, c0 : c0 + w]
    mask = patch != BG_COLOR
    view[mask] = patch[mask]


def _expected_output(grid: np.ndarray, frames: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    for frame in frames:
        r0 = int(frame["r0"])
        c0 = int(frame["c0"])
        inner_side = int(frame["inner_side"])
        k = int(frame["k"])
        patch = out[r0 : r0 + inner_side, c0 : c0 + inner_side].copy()
        out[r0 : r0 + inner_side, c0 : c0 + inner_side] = np.rot90(patch, -k)
    return out


def _passes_oracle(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        actual = main(inp)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_SIDE_RANGE)
    w = rng.randint(*GRID_SIDE_RANGE)

    area = h * w
    max_frames = min(FRAME_COUNT_RANGE[1], 1 + (area >= 180) + (area >= 320) + (area >= 500))
    n_frames = rng.randint(FRAME_COUNT_RANGE[0], max_frames)

    border_color, marker_color, pattern_color = rng.sample(COLOR_POOL, 3)
    marker_counts = _marker_counts_for_sample(rng, n_frames)

    specs = []
    for k in marker_counts:
        inner_side = rng.randint(*INNER_SIDE_RANGE)
        pattern = _make_pattern_patch(inner_side, pattern_color, rng)
        if pattern is None:
            return None
        specs.append({"inner_side": inner_side, "outer_side": inner_side + 2, "k": k, "pattern": pattern})

    specs.sort(key=lambda item: int(item["outer_side"]), reverse=True)
    placements = _place_frames(h, w, [int(spec["outer_side"]) for spec in specs], rng)
    if placements is None:
        return None

    inp = np.full((h, w), BG_COLOR, dtype=int)
    frame_records: list[dict[str, object]] = []
    for spec, (r0, c0) in zip(specs, placements):
        inner_side = int(spec["inner_side"])
        outer_side = int(spec["outer_side"])
        k = int(spec["k"])
        pattern = np.asarray(spec["pattern"], dtype=int)
        _draw_frame(inp, r0, c0, outer_side, border_color)

        marker_rel = rng.sample(_frame_candidates(outer_side), k)
        for dr, dc in marker_rel:
            inp[r0 + dr, c0 + dc] = int(marker_color)

        _paste_pattern(inp, pattern, r0 + 1, c0 + 1)
        frame_records.append({"r0": r0 + 1, "c0": c0 + 1, "inner_side": inner_side, "k": k})

    expected = _expected_output(inp, frame_records)
    if not _passes_oracle(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
