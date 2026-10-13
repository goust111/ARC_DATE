import numpy as np


BG = 0
FRAME = 5
PATTERN_COLORS = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
GEN_RETRIES = 300


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _frame_border() -> set[tuple[int, int]]:
    return {
        (r, c)
        for r in range(5)
        for c in range(5)
        if r in (0, 4) or c in (0, 4)
    }


_BASE_BORDER = _frame_border()
_CLOCK_TEMPLATES = {
    "up": {(r + 2, c) for (r, c) in _BASE_BORDER} | {(0, 2), (1, 2)},
    "down": _BASE_BORDER | {(5, 2), (6, 2)},
    "left": {(r, c + 2) for (r, c) in _BASE_BORDER} | {(2, 0), (2, 1)},
    "right": _BASE_BORDER | {(2, 5), (2, 6)},
}
_FRAME_OFFSETS = {
    "up": (2, 0),
    "down": (0, 0),
    "left": (0, 2),
    "right": (0, 0),
}
_LAYOUTS = {
    "A": {"up": (0, 0), "right": (0, 1), "left": (1, 0), "down": (1, 1)},
    "B": {"left": (0, 0), "up": (0, 1), "down": (1, 0), "right": (1, 1)},
}


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    mask = grid == color
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _normalize(coords: list[tuple[int, int]]) -> set[tuple[int, int]]:
    r0 = min(r for r, _ in coords)
    c0 = min(c for _, c in coords)
    return {(r - r0, c - c0) for r, c in coords}


def _detect_clocks(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)
    clocks: list[dict[str, object]] = []

    for color in map(int, np.unique(g)):
        if color == bg:
            continue
        for comp in _components_of_color(g, color):
            if len(comp) != 18:
                continue
            shape = _normalize(comp)
            direction = None
            for name, template in _CLOCK_TEMPLATES.items():
                if shape == template:
                    direction = name
                    break
            if direction is None:
                continue

            comp_r0 = min(r for r, _ in comp)
            comp_c0 = min(c for _, c in comp)
            dr, dc = _FRAME_OFFSETS[direction]
            frame_r0 = comp_r0 + dr
            frame_c0 = comp_c0 + dc
            assert 0 <= frame_r0 <= h - 5
            assert 0 <= frame_c0 <= w - 5

            pattern = g[frame_r0 + 1 : frame_r0 + 4, frame_c0 + 1 : frame_c0 + 4].copy()
            clocks.append(
                {
                    "direction": direction,
                    "frame_r": frame_r0,
                    "frame_c": frame_c0,
                    "pattern": pattern,
                }
            )

    assert len(clocks) == 4, "expected exactly four clock objects"
    assert {clock["direction"] for clock in clocks} == {"up", "down", "left", "right"}
    return clocks


def _valid_layout(layout: dict[str, tuple[int, int]], by_dir: dict[str, dict[str, object]]) -> bool:
    for row in (0, 1):
        row_dirs = [direction for direction, (rr, _cc) in layout.items() if rr == row]
        row_dirs.sort(key=lambda direction: layout[direction][1])
        left_dir, right_dir = row_dirs
        if int(by_dir[left_dir]["frame_c"]) >= int(by_dir[right_dir]["frame_c"]):
            return False
    return True


def _choose_layout(clocks: list[dict[str, object]]) -> tuple[str, dict[str, tuple[int, int]]]:
    by_dir = {str(clock["direction"]): clock for clock in clocks}
    valid = [
        (name, layout)
        for name, layout in _LAYOUTS.items()
        if _valid_layout(layout, by_dir)
    ]
    assert len(valid) == 1, "clock columns must select exactly one legal 2x2 arrangement"
    return valid[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the 3x3 pattern inside each clock frame and place it into a 2x2 panel.

    Every clock contributes one directional constraint:
    - up/down decide which output row the pattern belongs to;
    - left/right decide which output column the pattern belongs to.

    Those constraints allow exactly two global layouts. The intended one is the unique
    layout whose two rows preserve the original left-to-right order of the contributing
    clocks' frame positions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)

    clocks = _detect_clocks(g)
    _layout_name, layout = _choose_layout(clocks)
    by_dir = {str(clock["direction"]): clock for clock in clocks}

    out = np.full((7, 7), bg, dtype=int)
    for direction, (panel_r, panel_c) in layout.items():
        pattern = np.asarray(by_dir[direction]["pattern"], dtype=int)
        r0 = panel_r * 4
        c0 = panel_c * 4
        out[r0 : r0 + 3, c0 : c0 + 3] = pattern
    return out


def _clock_cells(frame_r0: int, frame_c0: int, direction: str) -> set[tuple[int, int]]:
    cells = {(frame_r0 + r, frame_c0 + c) for r in range(5) for c in range(5)}
    if direction == "up":
        cells |= {(frame_r0 - 1, frame_c0 + 2), (frame_r0 - 2, frame_c0 + 2)}
    elif direction == "down":
        cells |= {(frame_r0 + 5, frame_c0 + 2), (frame_r0 + 6, frame_c0 + 2)}
    elif direction == "left":
        cells |= {(frame_r0 + 2, frame_c0 - 1), (frame_r0 + 2, frame_c0 - 2)}
    else:
        cells |= {(frame_r0 + 2, frame_c0 + 5), (frame_r0 + 2, frame_c0 + 6)}
    return cells


def _inflate(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    halo: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    halo.add((rr, cc))
    return halo


def _row_candidates(h: int, direction: str) -> np.ndarray:
    if direction == "up":
        return np.arange(2, h - 4, dtype=int)
    if direction == "down":
        return np.arange(0, h - 6, dtype=int)
    return np.arange(0, h - 4, dtype=int)


def _sample_pattern(rng: np.random.Generator, used: set[tuple[int, ...]]) -> np.ndarray:
    for _ in range(100):
        mode = int(rng.integers(4))
        if mode == 0:
            color = int(rng.choice(PATTERN_COLORS))
            pattern = np.full((3, 3), color, dtype=int)
        elif mode == 1:
            colors = rng.choice(PATTERN_COLORS, size=2, replace=False)
            pattern = np.empty((3, 3), dtype=int)
            axis = int(rng.integers(2))
            labels = rng.integers(0, 2, size=3)
            if len(set(map(int, labels))) == 1:
                labels[1] = 1 - labels[1]
            if axis == 0:
                for r in range(3):
                    pattern[r, :] = int(colors[labels[r]])
            else:
                for c in range(3):
                    pattern[:, c] = int(colors[labels[c]])
        elif mode == 2:
            colors = rng.choice(PATTERN_COLORS, size=2, replace=False)
            pattern = np.empty((3, 3), dtype=int)
            for r in range(3):
                for c in range(3):
                    pattern[r, c] = int(colors[(r + c + int(rng.integers(2))) % 2])
        else:
            color_count = int(rng.integers(2, 4))
            colors = rng.choice(PATTERN_COLORS, size=color_count, replace=False)
            indices = rng.integers(0, color_count, size=(3, 3))
            pattern = colors[indices].astype(int)
            if len(np.unique(pattern)) < 2:
                continue

        sig = tuple(map(int, pattern.ravel()))
        if sig not in used:
            used.add(sig)
            return pattern
    raise RuntimeError("failed to sample a distinct 3x3 pattern")


def _sample_columns(rng: np.random.Generator, width: int, variant: str) -> dict[str, int]:
    safe = np.arange(2, width - 6, dtype=int)
    picks = sorted(map(int, rng.choice(safe, size=4, replace=False)))
    s0, s1, s2, s3 = picks

    if variant == "A":
        if rng.random() < 0.5:
            return {"up": s0, "left": s1, "right": s2, "down": s3}
        return {"left": s0, "up": s1, "right": s2, "down": s3}

    if rng.random() < 0.5:
        return {"left": s0, "down": s1, "right": s2, "up": s3}
    return {"down": s0, "left": s1, "right": s2, "up": s3}


def _sample_positions(
    rng: np.random.Generator, height: int, width: int, variant: str
) -> dict[str, tuple[int, int]] | None:
    frame_cols = _sample_columns(rng, width, variant)
    blocked: set[tuple[int, int]] = set()
    placements: dict[str, tuple[int, int]] = {}

    directions = list(frame_cols)
    rng.shuffle(directions)
    for direction in directions:
        c0 = frame_cols[direction]
        rows = _row_candidates(height, direction).copy()
        rng.shuffle(rows)
        placed = False
        for r0 in map(int, rows):
            cells = _clock_cells(r0, c0, direction)
            if any(not (0 <= r < height and 0 <= c < width) for r, c in cells):
                continue
            if not blocked.isdisjoint(cells):
                continue
            placements[direction] = (r0, c0)
            blocked |= _inflate(cells, height, width)
            placed = True
            break
        if not placed:
            return None
    return placements


def _draw_clock(
    grid: np.ndarray, frame_r0: int, frame_c0: int, direction: str, pattern: np.ndarray
) -> None:
    grid[frame_r0 : frame_r0 + 5, frame_c0 : frame_c0 + 5] = BG
    for r, c in _BASE_BORDER:
        grid[frame_r0 + r, frame_c0 + c] = FRAME
    grid[frame_r0 + 1 : frame_r0 + 4, frame_c0 + 1 : frame_c0 + 4] = pattern

    if direction == "up":
        grid[frame_r0 - 2 : frame_r0, frame_c0 + 2] = FRAME
    elif direction == "down":
        grid[frame_r0 + 5 : frame_r0 + 7, frame_c0 + 2] = FRAME
    elif direction == "left":
        grid[frame_r0 + 2, frame_c0 - 2 : frame_c0] = FRAME
    else:
        grid[frame_r0 + 2, frame_c0 + 5 : frame_c0 + 7] = FRAME


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(18, 27))
    width = int(rng.integers(18, 27))
    variant = "A" if rng.random() < 0.5 else "B"

    placements = _sample_positions(rng, height, width, variant)
    if placements is None:
        return None

    grid = np.full((height, width), BG, dtype=int)
    used_patterns: set[tuple[int, ...]] = set()
    for direction, (frame_r0, frame_c0) in placements.items():
        pattern = _sample_pattern(rng, used_patterns)
        _draw_clock(grid, frame_r0, frame_c0, direction, pattern)

    try:
        clocks = _detect_clocks(grid)
        layout_name, _layout = _choose_layout(clocks)
        if layout_name != variant:
            return None
        _ = main(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid clock-arrangement input")
