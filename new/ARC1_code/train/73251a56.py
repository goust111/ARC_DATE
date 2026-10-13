import numpy as np
import random


def _palette_cycle(g: np.ndarray) -> list[int]:
    h, w = g.shape
    assert h == w and h >= 3, "grid must be a square large enough to show the header"
    assert np.all(g[0] != 0), "top row must expose the complete nonzero color cycle"

    diag_color = int(g[0, 0])
    cycle: list[int] = []
    for c in range(2, w, 2):
        color = int(g[0, c])
        if color == diag_color and cycle:
            break
        assert color not in cycle, "top-row cycle colors must be unique until they repeat"
        cycle.append(color)

    colors = {int(color) for color in g.ravel() if color != 0}
    assert len(cycle) >= 2 and cycle[0] == diag_color, "cycle must start with the diagonal color"
    assert set(cycle) == colors, "top-row cycle must contain every nonzero color"
    assert int(g[0, 1]) == cycle[-1], "column 1 must show the previous color in the cycle"
    return cycle


def _table_value(r: int, c: int, cycle: list[int]) -> int:
    near = min(r, c)
    dist = abs(r - c)
    if dist == 0:
        return cycle[0]
    if dist <= near + 1:
        return cycle[-1]
    return cycle[((dist - (near + 2)) // (near + 2)) % len(cycle)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill black gaps by reconstructing the cyclic symmetric coordinate table visible in the non-black cells.

    Input structure:
    - The grid is square, with black pixels marking missing cells in an otherwise complete symmetric table.
    - The first row is complete: it shows the diagonal color, the previous cycle color, then paired runs of the palette cycle.
    - Every non-black input pixel already agrees with the hidden table.

    Rule:
    - Read the palette cycle from every other first-row cell starting at column 2, stopping when the diagonal color repeats.
    - For a cell, let near = min(row, column) and dist = abs(row - column).
    - Use the diagonal color when dist is 0, the previous cycle color when dist <= near + 1, otherwise cycle through blocks of width near + 2.

    Key insight:
    The missing pixels are not locally interpolated; they are entries of a global symmetric table indexed by distance from the main diagonal.
    """
    g = np.asarray(grid, dtype=int)
    cycle = _palette_cycle(g)

    out = np.empty_like(g)
    h, w = g.shape
    for r in range(h):
        for c in range(w):
            out[r, c] = _table_value(r, c, cycle)

    assert np.all((g == 0) | (g == out)), "nonzero input cells must already match the reconstructed table"
    return out

# --------------- generator (o5x) ---------------

# Canvas and palette sampling.
GRID_SIZE = 21
FIRST_MASK_ROW = 1
MIN_COLORS = 6
MAX_COLORS = 9
FIRST_COLOR = 1

# Black region constraints.
MIN_COMPONENTS = 2
MAX_COMPONENTS = 5
MIN_COMPONENT_AREA = 6
MAX_COMPONENT_AREA = 32
MIN_TOTAL_BLACK = 40
MAX_TOTAL_BLACK = 57

# Component shape bank.
RECT_MIN_H = 2
RECT_MAX_H = 9
RECT_MIN_W = 2
RECT_MAX_W = 8
L_MIN_H = 3
L_MAX_H = 9
L_MIN_W = 3
L_MAX_W = 8
L_LEG_MIN = 2
L_SHAPE_PROB = 0.25
MAX_GENERATION_ATTEMPTS = 300

_GEN_COMPONENT_BANK = None


def _gen_table_value(r: int, c: int, cycle: list[int]) -> int:
    near = min(r, c)
    dist = abs(r - c)
    if dist == 0:
        return cycle[0]
    if dist <= near + 1:
        return cycle[-1]
    return cycle[((dist - (near + 2)) // (near + 2)) % len(cycle)]


def _gen_build_table(cycle: list[int]) -> np.ndarray:
    out = np.empty((GRID_SIZE, GRID_SIZE), dtype=int)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            out[r, c] = _gen_table_value(r, c, cycle)
    return out


def _gen_add_mask(bank: dict[int, list[tuple[np.ndarray, str]]], mask: np.ndarray, kind: str) -> None:
    area = int(mask.sum())
    if MIN_COMPONENT_AREA <= area <= MAX_COMPONENT_AREA:
        bank.setdefault(area, []).append((mask.copy(), kind))


def _gen_component_bank() -> dict[int, list[tuple[np.ndarray, str]]]:
    """Return cached component masks keyed by exact black-cell area."""
    global _GEN_COMPONENT_BANK
    if _GEN_COMPONENT_BANK is not None:
        return _GEN_COMPONENT_BANK

    bank: dict[int, list[tuple[np.ndarray, str]]] = {}
    for h in range(RECT_MIN_H, RECT_MAX_H + 1):
        for w in range(RECT_MIN_W, RECT_MAX_W + 1):
            area = h * w
            if MIN_COMPONENT_AREA <= area <= MAX_COMPONENT_AREA:
                _gen_add_mask(bank, np.ones((h, w), dtype=bool), "rect")

    for h in range(L_MIN_H, L_MAX_H + 1):
        for w in range(L_MIN_W, L_MAX_W + 1):
            for leg_w in range(L_LEG_MIN, w):
                for leg_h in range(L_LEG_MIN, h):
                    base = np.zeros((h, w), dtype=bool)
                    base[:, :leg_w] = True
                    base[:leg_h, :] = True
                    if base.sum() > MAX_COMPONENT_AREA:
                        continue
                    variants = (
                        base,
                        np.flipud(base),
                        np.fliplr(base),
                        np.flipud(np.fliplr(base)),
                    )
                    for variant in variants:
                        _gen_add_mask(bank, variant, "l")

    _GEN_COMPONENT_BANK = bank
    return bank


def _gen_can_sum(areas: tuple[int, ...], slots: int, total: int, memo: dict[tuple[int, int], bool]) -> bool:
    key = (slots, total)
    if key in memo:
        return memo[key]
    if slots == 0:
        memo[key] = total == 0
        return memo[key]
    if total < MIN_COMPONENT_AREA * slots or total > MAX_COMPONENT_AREA * slots:
        memo[key] = False
        return False
    memo[key] = any(_gen_can_sum(areas, slots - 1, total - area, memo) for area in areas)
    return memo[key]


def _gen_choose_components(count: int, total: int) -> list[np.ndarray] | None:
    """Sample component masks whose areas sum exactly to the requested total."""
    bank = _gen_component_bank()
    areas = tuple(sorted(bank))
    memo: dict[tuple[int, int], bool] = {}
    if not _gen_can_sum(areas, count, total, memo):
        return None

    masks: list[np.ndarray] = []
    remaining = total
    for slots in range(count, 0, -1):
        choices = [
            area
            for area in areas
            if _gen_can_sum(areas, slots - 1, remaining - area, memo)
        ]
        if not choices:
            return None
        area = random.choice(choices)
        options = bank[area]
        rects = [mask for mask, kind in options if kind == "rect"]
        l_shapes = [mask for mask, kind in options if kind == "l"]
        if rects and l_shapes:
            pool = l_shapes if random.random() < L_SHAPE_PROB else rects
        else:
            pool = rects or l_shapes
        masks.append(random.choice(pool).copy())
        remaining -= area

    random.shuffle(masks)
    return masks


def _gen_fits(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    h, w = patch.shape
    rows, cols = np.nonzero(patch)
    for dr, dc in zip(rows, cols):
        r = r0 + int(dr)
        c = c0 + int(dc)
        if canvas[r, c]:
            return False
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and canvas[nr, nc]:
                return False
    return r0 >= FIRST_MASK_ROW and r0 + h <= GRID_SIZE and c0 >= 0 and c0 + w <= GRID_SIZE


def _gen_place_components(masks: list[np.ndarray]) -> np.ndarray | None:
    """Place sampled masks without overlap or 4-neighbor contact."""
    canvas = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    for patch in sorted(masks, key=lambda m: int(m.sum()), reverse=True):
        h, w = patch.shape
        locations = [
            (r, c)
            for r in range(FIRST_MASK_ROW, GRID_SIZE + 1 - h)
            for c in range(0, GRID_SIZE + 1 - w)
            if _gen_fits(canvas, patch, r, c)
        ]
        if not locations:
            return None
        r0, c0 = random.choice(locations)
        canvas[r0 : r0 + h, c0 : c0 + w] |= patch
    return canvas


def _gen_attempt() -> np.ndarray | None:
    n_colors = random.randint(MIN_COLORS, MAX_COLORS)
    start = random.randint(FIRST_COLOR, n_colors)
    cycle = list(range(start, n_colors + 1)) + list(range(FIRST_COLOR, start))
    output = _gen_build_table(cycle)

    component_count = random.randint(MIN_COMPONENTS, MAX_COMPONENTS)
    total_black = random.randint(MIN_TOTAL_BLACK, MAX_TOTAL_BLACK)
    masks = _gen_choose_components(component_count, total_black)
    if masks is None:
        return None
    black = _gen_place_components(masks)
    if black is None:
        return None

    inp = output.copy()
    inp[black] = 0
    if not np.array_equal(main(inp), output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid input")
