import random
from collections import Counter, deque

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))

# --------------- solver ---------------

QUADRANTS = ("ul", "ur", "ll", "lr")


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _segment_from_component(comp: list[tuple[int, int]]) -> dict[str, int | str]:
    rows = sorted({r for r, _ in comp})
    cols = sorted({c for _, c in comp})
    if len(rows) == 1:
        row = rows[0]
        c0 = min(cols)
        c1 = max(cols)
        assert len(comp) == c1 - c0 + 1
        return {"orientation": "h", "row": int(row), "c0": int(c0), "c1": int(c1)}
    if len(cols) == 1:
        col = cols[0]
        r0 = min(rows)
        r1 = max(rows)
        assert len(comp) == r1 - r0 + 1
        return {"orientation": "v", "col": int(col), "r0": int(r0), "r1": int(r1)}
    raise AssertionError("component must be a straight segment")


def _find_palette(grid: np.ndarray) -> dict[str, int]:
    counts = Counter(map(int, grid.ravel()))
    h, w = grid.shape
    candidates: list[tuple[int, int]] = []
    for r in range(h - 1):
        for c in range(w - 1):
            vals = [
                int(grid[r, c]),
                int(grid[r, c + 1]),
                int(grid[r + 1, c]),
                int(grid[r + 1, c + 1]),
            ]
            if BG in vals or len(set(vals)) != 4:
                continue
            if any(counts[v] != 1 for v in vals):
                continue
            candidates.append((r, c))
    assert len(candidates) == 1, "expected a unique 2x2 palette"
    r, c = candidates[0]
    return {
        "ul": int(grid[r, c]),
        "ur": int(grid[r, c + 1]),
        "ll": int(grid[r + 1, c]),
        "lr": int(grid[r + 1, c + 1]),
    }


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    palette = _find_palette(g)
    palette_colors = set(palette.values())

    for color in sorted(int(v) for v in np.unique(g) if int(v) not in palette_colors | {BG}):
        comps = _components(g == color)
        assert len(comps) == 2, "each line color must form exactly two segments"
        segs = [_segment_from_component(comp) for comp in comps]
        horizontals = [seg for seg in segs if seg["orientation"] == "h"]
        verticals = [seg for seg in segs if seg["orientation"] == "v"]
        assert len(horizontals) == 1 and len(verticals) == 1

        hseg = horizontals[0]
        vseg = verticals[0]
        inter_r = int(hseg["row"])
        inter_c = int(vseg["col"])

        if inter_r < int(vseg["r0"]):
            vertical_side = "top"
            far_r = int(vseg["r1"])
        else:
            assert inter_r > int(vseg["r1"])
            vertical_side = "bottom"
            far_r = int(vseg["r0"])

        if inter_c < int(hseg["c0"]):
            horizontal_side = "left"
            far_c = int(hseg["c1"])
        else:
            assert inter_c > int(hseg["c1"])
            horizontal_side = "right"
            far_c = int(hseg["c0"])

        if vertical_side == "top" and horizontal_side == "left":
            quad = "ul"
        elif vertical_side == "top" and horizontal_side == "right":
            quad = "ur"
        elif vertical_side == "bottom" and horizontal_side == "left":
            quad = "ll"
        else:
            quad = "lr"

        rr0, rr1 = sorted((int(inter_r), int(far_r)))
        cc0, cc1 = sorted((int(inter_c), int(far_c)))
        if rr1 - rr0 >= 2 and cc1 - cc0 >= 2:
            out[rr0 + 1 : rr1, cc0 + 1 : cc1] = int(palette[quad])

    return out


# --------------- generator ---------------

PAIR_COUNT_CHOICES = [2, 3, 4, 5]
PAIR_COUNT_WEIGHTS = [2, 4, 3, 1]
GRID_MIN_BY_PAIRS = {2: 16, 3: 18, 4: 22, 5: 24}
GRID_MAX = 30
RECT_SIZE_RANGE = (4, 10)
PLACEMENT_TRIES = 80
GEN_RETRIES = 400
OBJECT_PAD = 1


def _bbox_clear(blocked: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = OBJECT_PAD) -> bool:
    h, w = blocked.shape
    if r0 < 1 or c0 < 1 or r1 >= h - 1 or c1 >= w - 1:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    return not bool(blocked[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _reserve(blocked: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = OBJECT_PAD) -> None:
    h, w = blocked.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _draw_pair(
    grid: np.ndarray,
    color: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    quad: str,
    gap_r: int,
    gap_c: int,
) -> None:
    if quad == "ul":
        grid[r0, c0 + gap_c : c1 + 1] = color
        grid[r0 + gap_r : r1 + 1, c0] = color
    elif quad == "ur":
        grid[r0, c0 : c1 - gap_c + 1] = color
        grid[r0 + gap_r : r1 + 1, c1] = color
    elif quad == "ll":
        grid[r1, c0 + gap_c : c1 + 1] = color
        grid[r0 : r1 - gap_r + 1, c0] = color
    else:
        assert quad == "lr"
        grid[r1, c0 : c1 - gap_c + 1] = color
        grid[r0 : r1 - gap_r + 1, c1] = color


def _expected_output(grid: np.ndarray, specs: list[dict[str, int | str]], palette: dict[str, int]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        r0 = int(spec["r0"])
        c0 = int(spec["c0"])
        r1 = int(spec["r1"])
        c1 = int(spec["c1"])
        out[r0 + 1 : r1, c0 + 1 : c1] = int(palette[str(spec["quad"])])
    return out


def _place_palette(grid: np.ndarray, blocked: np.ndarray, palette_colors: list[int]) -> dict[str, int] | None:
    h, w = grid.shape
    spots: list[tuple[int, int]] = []
    for r in range(1, h - 2):
        for c in range(1, w - 2):
            if _bbox_clear(blocked, r, c, r + 1, c + 1):
                spots.append((r, c))
    if not spots:
        return None
    r, c = random.choice(spots)
    grid[r, c] = palette_colors[0]
    grid[r, c + 1] = palette_colors[1]
    grid[r + 1, c] = palette_colors[2]
    grid[r + 1, c + 1] = palette_colors[3]
    _reserve(blocked, r, c, r + 1, c + 1)
    return {"ul": palette_colors[0], "ur": palette_colors[1], "ll": palette_colors[2], "lr": palette_colors[3]}


def _place_pair(grid: np.ndarray, blocked: np.ndarray, color: int) -> dict[str, int | str] | None:
    h, w = grid.shape
    for _ in range(PLACEMENT_TRIES):
        rect_h = random.randint(*RECT_SIZE_RANGE)
        rect_w = random.randint(*RECT_SIZE_RANGE)
        if rect_h > h - 2 or rect_w > w - 2:
            continue
        r0 = random.randint(1, h - rect_h - 1)
        c0 = random.randint(1, w - rect_w - 1)
        r1 = r0 + rect_h - 1
        c1 = c0 + rect_w - 1
        if not _bbox_clear(blocked, r0, c0, r1, c1):
            continue

        gap_r = random.randint(1, rect_h - 3)
        gap_c = random.randint(1, rect_w - 3)
        quad = random.choice(QUADRANTS)
        _draw_pair(grid, color, r0, c0, r1, c1, quad, gap_r, gap_c)
        _reserve(blocked, r0, c0, r1, c1)
        return {"color": color, "r0": r0, "c0": c0, "r1": r1, "c1": c1, "quad": quad}
    return None


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choices(PAIR_COUNT_CHOICES, weights=PAIR_COUNT_WEIGHTS, k=1)[0]
    min_dim = GRID_MIN_BY_PAIRS[pair_count]
    height = random.randint(min_dim, GRID_MAX)
    width = random.randint(min_dim, GRID_MAX)
    grid = np.full((height, width), BG, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)

    sampled = random.sample(COLORS, pair_count + 4)
    palette_colors = sampled[:4]
    line_colors = sampled[4:]
    random.shuffle(palette_colors)

    palette = _place_palette(grid, blocked, palette_colors)
    if palette is None:
        return None

    specs: list[dict[str, int | str]] = []
    for color in line_colors:
        spec = _place_pair(grid, blocked, color)
        if spec is None:
            return None
        specs.append(spec)

    expected = _expected_output(grid, specs, palette)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
