from __future__ import annotations

import random

import numpy as np


BG = 0
GUIDE = 2
SOURCE_COLORS = (1, 4, 7)


def _guide_candidates(grid: np.ndarray) -> list[int]:
    h, w = grid.shape
    candidates: list[int] = []
    for r in range(h):
        row = grid[r]
        red = int(np.count_nonzero(row == GUIDE))
        if red < w - 2:
            continue
        if np.any((row != BG) & (row != GUIDE)):
            continue
        candidates.append(int(r))
    return candidates


def _find_guide_row(grid: np.ndarray) -> int:
    candidates = _guide_candidates(grid)
    if not candidates:
        raise ValueError("expected a horizontal red guide row")
    h = grid.shape[0]
    return max(candidates, key=lambda r: (int(np.count_nonzero(grid[r] == GUIDE)), -abs(r - h // 2)))


def _source_pixels(grid: np.ndarray) -> list[tuple[int, int, int]]:
    return [(int(r), int(c), int(grid[r, c])) for r, c in np.argwhere((grid != BG) & (grid != GUIDE))]


def _block_cells(shape: tuple[int, int], r: int, c: int) -> list[tuple[int, int]]:
    h, w = shape
    cells: list[tuple[int, int]] = []
    for dr in (0, 1):
        for dc in (0, 1):
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < h and 0 <= cc < w:
                cells.append((rr, cc))
    return cells


def _classify_source(c: int, gap_cols: set[int], width: int) -> str:
    block_cols = {int(c)}
    if int(c) + 1 < width:
        block_cols.add(int(c) + 1)
    if int(c) in gap_cols:
        return "direct"
    if block_cols & gap_cols:
        return "emergent"
    return "safe"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each scattered source pixel into a 2x2 block, then erase the blocks whose expanded column span covers a guide-bar gap.

    Input structure:
    - A black background contains one horizontal red guide bar spanning the full row, with 1-2 single-column black gaps.
    - Several isolated non-black, non-red source pixels sit at least two rows above the guide.
    - Source colors are drawn from a small visible palette and may repeat.

    Rule:
    - Keep the red guide row unchanged and treat every black cell on that row as a gap column.
    - For each source pixel, form a same-color 2x2 block anchored at that pixel's top-left position, clipping any part that falls outside the grid.
    - If the expanded block's in-bounds column range intersects any guide-gap column, erase that whole block; otherwise keep it.
    - The output contains only the unchanged guide bar and the surviving expanded blocks.

    Key insight:
    - Survival is decided after expansion, so a pixel just left of a gap can die only because its new right-hand column reaches into the gap.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("grid must be 2D")

    h, w = g.shape
    guide_row = _find_guide_row(g)
    gap_cols = set(map(int, np.flatnonzero(g[guide_row] == BG)))

    out = np.full((h, w), BG, dtype=int)
    out[guide_row] = g[guide_row]

    for r, c, color in _source_pixels(g):
        if _classify_source(c, gap_cols, w) != "safe":
            continue
        for rr, cc in _block_cells(g.shape, r, c):
            out[rr, cc] = int(color)

    out[guide_row] = g[guide_row]
    return out


def _can_place(used: list[tuple[int, int]], r: int, c: int) -> bool:
    return all(max(abs(int(r) - ur), abs(int(c) - uc)) >= 2 for ur, uc in used)


def _place_sources(
    guide_row: int,
    width: int,
    items: list[str],
    cols_by_kind: dict[str, list[int]],
) -> list[tuple[str, int, int]] | None:
    rows = list(range(guide_row - 1))
    order = sorted(items, key=lambda kind: (len(cols_by_kind[kind]), kind))

    for _ in range(120):
        used: list[tuple[int, int]] = []
        placed: list[tuple[str, int, int]] = []
        ok = True
        for kind in order:
            coords = [(r, c) for r in rows for c in cols_by_kind[kind]]
            random.shuffle(coords)
            chosen = None
            for r, c in coords:
                if _can_place(used, r, c):
                    chosen = (kind, int(r), int(c))
                    used.append((int(r), int(c)))
                    placed.append(chosen)
                    break
            if chosen is None:
                ok = False
                break
        if ok:
            random.shuffle(placed)
            return placed
    return None


def _is_isolated_source(grid: np.ndarray, r: int, c: int) -> bool:
    h, w = grid.shape
    r0 = max(0, int(r) - 1)
    r1 = min(h, int(r) + 2)
    c0 = max(0, int(c) - 1)
    c1 = min(w, int(c) + 2)
    neighborhood = (grid[r0:r1, c0:c1] != BG) & (grid[r0:r1, c0:c1] != GUIDE)
    return int(np.count_nonzero(neighborhood)) == 1


def _validate_example(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    h, w = grid.shape
    if h > 30 or w > 30:
        return False

    guide_rows = _guide_candidates(grid)
    if len(guide_rows) != 1:
        return False
    guide_row = guide_rows[0]
    gap_cols = set(map(int, np.flatnonzero(grid[guide_row] == BG)))
    if not (1 <= len(gap_cols) <= 2):
        return False
    if any(c in (0, w - 1) for c in gap_cols):
        return False
    if int(np.count_nonzero(grid == GUIDE)) != w - len(gap_cols):
        return False

    sources = _source_pixels(grid)
    if not (4 <= len(sources) <= 6):
        return False
    kinds: list[str] = []
    for r, c, color in sources:
        if color not in SOURCE_COLORS:
            return False
        if r > guide_row - 2:
            return False
        if not _is_isolated_source(grid, r, c):
            return False
        kinds.append(_classify_source(c, gap_cols, w))

    if "direct" not in kinds or "emergent" not in kinds or "safe" not in kinds:
        return False

    expected = np.zeros_like(grid)
    expected[guide_row] = grid[guide_row]
    for r, c, color in sources:
        if _classify_source(c, gap_cols, w) != "safe":
            continue
        for rr, cc in _block_cells(grid.shape, r, c):
            expected[rr, cc] = int(color)
    expected[guide_row] = grid[guide_row]
    return np.array_equal(main(grid), expected)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(11, 15)
    w = random.randint(11, 16)
    guide_row = random.randint(max(4, h // 2 - 1), min(h - 3, h // 2 + 1))

    gap_count = 2 if random.random() < 0.65 else 1
    gap_pool = list(range(1, w - 1))
    random.shuffle(gap_pool)
    gaps: list[int] = []
    for c in gap_pool:
        if all(abs(int(c) - prev) >= 2 for prev in gaps):
            gaps.append(int(c))
            if len(gaps) == gap_count:
                break
    if len(gaps) != gap_count:
        return None
    gaps.sort()

    safe_cols = [c for c in range(w - 1) if c not in gaps and c + 1 not in gaps]
    emergent_cols = [c - 1 for c in gaps if c - 1 >= 0]
    direct_cols = list(gaps)
    if not safe_cols or not emergent_cols or not direct_cols:
        return None

    n_sources = random.randint(4, 6)
    items = ["direct", "emergent", "safe"]
    if w >= 12 and n_sources >= 5 and random.random() < 0.35:
        items.append("edge")
    while len(items) < n_sources:
        items.append(random.choices(["direct", "emergent", "safe"], weights=[2, 2, 3], k=1)[0])

    cols_by_kind = {
        "direct": direct_cols,
        "emergent": emergent_cols,
        "safe": safe_cols,
        "edge": [w - 1],
    }
    placed = _place_sources(guide_row, w, items, cols_by_kind)
    if placed is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    grid[guide_row, :] = GUIDE
    for gap in gaps:
        grid[guide_row, gap] = BG

    colors = [random.choice(SOURCE_COLORS) for _ in placed]
    if len(colors) > 1 and len(set(colors)) == 1:
        idx = random.randrange(len(colors))
        colors[idx] = random.choice([c for c in SOURCE_COLORS if c != colors[idx]])

    for (_, r, c), color in zip(placed, colors):
        grid[r, c] = int(color)

    if not _validate_example(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
