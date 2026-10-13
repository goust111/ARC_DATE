import numpy as np
from collections import deque


DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

BG = 0
FENCE = 1
COLOR_POOL = tuple(range(2, 10))

GRID_H_RANGE = (18, 31)
GRID_W_RANGE = (18, 31)
FENCE_COUNT_RETRIES = 140
PLACEMENT_RETRIES = 120
MAX_GENERATE_ATTEMPTS = 400

SEED_TEMPLATES = [
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0], [1, 1], [0, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 1]], dtype=bool),
    np.array([[1, 1, 1, 0], [0, 1, 0, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
]


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIRS.values():
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _color_components(grid: np.ndarray, bg: int) -> list[dict]:
    comps: list[dict] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for coords in _components(grid == color):
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _is_frame_component(grid: np.ndarray, bg: int, comp: dict) -> bool:
    color = int(comp["color"])
    r0, c0, r1, c1 = comp["bbox"]
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros_like(patch, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if int(comp["area"]) != int(border.sum()):
        return False
    if not np.all(patch[border] == color):
        return False
    if np.any(patch[~border] == color):
        return False
    inner = patch[1:-1, 1:-1]
    return bool(np.any((inner != bg) & (inner != color)))


def _ordered_stripe_cells(
    grid: np.ndarray, bg: int, frame_color: int, bbox: tuple[int, int, int, int], side: str
) -> list[tuple[int, int]] | None:
    h, w = grid.shape
    r0, c0, r1, c1 = bbox
    cells: list[tuple[int, int]] = []

    if side == "up":
        if r0 == 0:
            return None
        hits = [c for c in range(c0, c1 + 1) if int(grid[r0 - 1, c]) not in {bg, frame_color}]
        if len(hits) != 1:
            return None
        c = hits[0]
        rr = r0 - 1
        while rr >= 0 and int(grid[rr, c]) not in {bg, frame_color}:
            cells.append((rr, c))
            rr -= 1
    elif side == "down":
        if r1 == h - 1:
            return None
        hits = [c for c in range(c0, c1 + 1) if int(grid[r1 + 1, c]) not in {bg, frame_color}]
        if len(hits) != 1:
            return None
        c = hits[0]
        rr = r1 + 1
        while rr < h and int(grid[rr, c]) not in {bg, frame_color}:
            cells.append((rr, c))
            rr += 1
    elif side == "left":
        if c0 == 0:
            return None
        hits = [r for r in range(r0, r1 + 1) if int(grid[r, c0 - 1]) not in {bg, frame_color}]
        if len(hits) != 1:
            return None
        r = hits[0]
        cc = c0 - 1
        while cc >= 0 and int(grid[r, cc]) not in {bg, frame_color}:
            cells.append((r, cc))
            cc -= 1
    else:
        assert side == "right"
        if c1 == w - 1:
            return None
        hits = [r for r in range(r0, r1 + 1) if int(grid[r, c1 + 1]) not in {bg, frame_color}]
        if len(hits) != 1:
            return None
        r = hits[0]
        cc = c1 + 1
        while cc < w and int(grid[r, cc]) not in {bg, frame_color}:
            cells.append((r, cc))
            cc += 1

    if not (2 <= len(cells) <= 4):
        return None
    return cells


def _frame_palette(
    grid: np.ndarray, bg: int, frame_color: int, bbox: tuple[int, int, int, int]
) -> tuple[int, int]:
    candidates = []
    for side in ("up", "down", "left", "right"):
        cells = _ordered_stripe_cells(grid, bg, frame_color, bbox, side)
        if cells is not None:
            candidates.append(cells)
    assert len(candidates) == 1, "each frame must have exactly one attached stripe"
    colors = [int(grid[r, c]) for r, c in candidates[0]]
    assert len(set(colors)) == 2, "stripe must use exactly two colors"
    color_a, color_b = colors[0], colors[1]
    assert color_a != color_b
    for i, color in enumerate(colors):
        assert color == (color_a if i % 2 == 0 else color_b), "stripe must alternate"
    return color_a, color_b


def _frame_seed(
    grid: np.ndarray, bg: int, frame_color: int, bbox: tuple[int, int, int, int]
) -> tuple[np.ndarray, int]:
    r0, c0, r1, c1 = bbox
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    mask = (inner != bg) & (inner != frame_color)
    comps = _components(mask)
    assert len(comps) == 1, "each frame must contain exactly one seed component"
    coords = comps[0] + np.array([r0 + 1, c0 + 1])
    seed_colors = {int(grid[r, c]) for r, c in coords}
    assert len(seed_colors) == 1, "seed must be monochrome"
    return coords, next(iter(seed_colors))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer a growth direction from the seed's off-center position inside each hollow fence,
    then extend translated frontier layers with alternating stripe colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)

    frame_components = [
        comp for comp in _color_components(g, bg) if _is_frame_component(g, bg, comp)
    ]
    assert frame_components, "expected at least one hollow fence"
    frame_components.sort(key=lambda comp: comp["bbox"])

    out = g.copy()
    for frame in frame_components:
        frame_color = int(frame["color"])
        r0, c0, r1, c1 = frame["bbox"]
        color_a, color_b = _frame_palette(g, bg, frame_color, frame["bbox"])
        seed_coords, _seed_color = _frame_seed(g, bg, frame_color, frame["bbox"])

        sr0, sc0, sr1, sc1 = _bbox(seed_coords)
        inner_r0, inner_c0 = r0 + 1, c0 + 1
        inner_r1, inner_c1 = r1 - 1, c1 - 1
        gaps = {
            "up": sr0 - inner_r0,
            "down": inner_r1 - sr1,
            "left": sc0 - inner_c0,
            "right": inner_c1 - sc1,
        }
        max_gap = max(gaps.values())
        growth_dirs = [name for name, gap in gaps.items() if gap == max_gap]
        assert len(growth_dirs) == 1, "growth direction must be uniquely determined"
        direction = growth_dirs[0]
        dr, dc = DIRS[direction]

        occupied = np.zeros_like(g, dtype=bool)
        occupied[seed_coords[:, 0], seed_coords[:, 1]] = True
        for step in range(1, max_gap + 1):
            translated = np.zeros_like(g, dtype=bool)
            rr = seed_coords[:, 0] + dr * step
            cc = seed_coords[:, 1] + dc * step
            assert np.all((inner_r0 <= rr) & (rr <= inner_r1))
            assert np.all((inner_c0 <= cc) & (cc <= inner_c1))
            translated[rr, cc] = True
            new_cells = translated & ~occupied
            out[new_cells] = color_a if step % 2 == 1 else color_b
            occupied |= translated

    return out


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _random_seed_mask(rng: np.random.Generator) -> np.ndarray:
    mask = SEED_TEMPLATES[int(rng.integers(0, len(SEED_TEMPLATES)))].copy()
    if rng.random() < 0.5:
        mask = np.fliplr(mask)
    mask = np.rot90(mask, k=int(rng.integers(0, 4)))
    return _normalize_mask(mask.astype(bool))


def _build_fence_spec(rng: np.random.Generator) -> dict:
    seed_mask = _random_seed_mask(rng)
    sh, sw = seed_mask.shape

    growth_dir = str(rng.choice(np.array(list(DIRS.keys()), dtype=object)))
    lead_gap = int(rng.integers(2, 5))
    other_gap = lambda: int(rng.integers(1, lead_gap))

    gaps = {
        "up": other_gap(),
        "down": other_gap(),
        "left": other_gap(),
        "right": other_gap(),
    }
    gaps[growth_dir] = lead_gap

    top_gap = gaps["up"]
    bottom_gap = gaps["down"]
    left_gap = gaps["left"]
    right_gap = gaps["right"]

    outer_h = sh + top_gap + bottom_gap + 2
    outer_w = sw + left_gap + right_gap + 2

    stripe_side = str(rng.choice(np.array(list(DIRS.keys()), dtype=object)))
    stripe_len = int(rng.integers(2, 5))
    seed_color, color_a, color_b = map(
        int, rng.choice(np.array(COLOR_POOL), size=3, replace=False)
    )

    if stripe_side in {"up", "down"}:
        stripe_anchor = int(rng.integers(1, outer_w - 1))
    else:
        stripe_anchor = int(rng.integers(1, outer_h - 1))

    return {
        "seed_mask": seed_mask,
        "seed_color": seed_color,
        "color_a": color_a,
        "color_b": color_b,
        "growth_dir": growth_dir,
        "gaps": gaps,
        "outer_h": outer_h,
        "outer_w": outer_w,
        "stripe_side": stripe_side,
        "stripe_len": stripe_len,
        "stripe_anchor": stripe_anchor,
    }


def _envelope(spec: dict, r0: int, c0: int) -> tuple[int, int, int, int]:
    top_ext = spec["stripe_len"] if spec["stripe_side"] == "up" else 0
    bottom_ext = spec["stripe_len"] if spec["stripe_side"] == "down" else 0
    left_ext = spec["stripe_len"] if spec["stripe_side"] == "left" else 0
    right_ext = spec["stripe_len"] if spec["stripe_side"] == "right" else 0
    return (
        r0 - top_ext,
        c0 - left_ext,
        r0 + spec["outer_h"] - 1 + bottom_ext,
        c0 + spec["outer_w"] - 1 + right_ext,
    )


def _place_specs(
    rng: np.random.Generator, h: int, w: int, specs: list[dict]
) -> list[dict] | None:
    occupied = np.zeros((h, w), dtype=bool)
    placed: list[dict] = []

    for spec in specs:
        top_ext = spec["stripe_len"] if spec["stripe_side"] == "up" else 0
        bottom_ext = spec["stripe_len"] if spec["stripe_side"] == "down" else 0
        left_ext = spec["stripe_len"] if spec["stripe_side"] == "left" else 0
        right_ext = spec["stripe_len"] if spec["stripe_side"] == "right" else 0

        row_lo = top_ext
        row_hi = h - spec["outer_h"] - bottom_ext
        col_lo = left_ext
        col_hi = w - spec["outer_w"] - right_ext
        if row_lo > row_hi or col_lo > col_hi:
            return None

        success = False
        for _ in range(PLACEMENT_RETRIES):
            r0 = int(rng.integers(row_lo, row_hi + 1))
            c0 = int(rng.integers(col_lo, col_hi + 1))
            er0, ec0, er1, ec1 = _envelope(spec, r0, c0)
            cr0 = max(0, er0 - 1)
            cc0 = max(0, ec0 - 1)
            cr1 = min(h - 1, er1 + 1)
            cc1 = min(w - 1, ec1 + 1)
            if np.any(occupied[cr0 : cr1 + 1, cc0 : cc1 + 1]):
                continue
            occupied[er0 : er1 + 1, ec0 : ec1 + 1] = True
            placed.append({**spec, "r0": r0, "c0": c0})
            success = True
            break
        if not success:
            return None

    return placed


def _draw_spec(grid: np.ndarray, spec: dict) -> None:
    r0, c0 = spec["r0"], spec["c0"]
    outer_h, outer_w = spec["outer_h"], spec["outer_w"]

    grid[r0, c0 : c0 + outer_w] = FENCE
    grid[r0 + outer_h - 1, c0 : c0 + outer_w] = FENCE
    grid[r0 : r0 + outer_h, c0] = FENCE
    grid[r0 : r0 + outer_h, c0 + outer_w - 1] = FENCE

    seed_mask = spec["seed_mask"]
    top_gap = spec["gaps"]["up"]
    left_gap = spec["gaps"]["left"]
    sr0 = r0 + 1 + top_gap
    sc0 = c0 + 1 + left_gap
    sh, sw = seed_mask.shape
    patch = grid[sr0 : sr0 + sh, sc0 : sc0 + sw]
    patch[seed_mask] = spec["seed_color"]

    side = spec["stripe_side"]
    length = spec["stripe_len"]
    anchor = spec["stripe_anchor"]
    colors = [spec["color_a"] if i % 2 == 0 else spec["color_b"] for i in range(length)]

    if side == "up":
        col = c0 + anchor
        for i, color in enumerate(colors):
            grid[r0 - 1 - i, col] = color
    elif side == "down":
        col = c0 + anchor
        for i, color in enumerate(colors):
            grid[r0 + outer_h + i, col] = color
    elif side == "left":
        row = r0 + anchor
        for i, color in enumerate(colors):
            grid[row, c0 - 1 - i] = color
    else:
        row = r0 + anchor
        for i, color in enumerate(colors):
            grid[row, c0 + outer_w + i] = color


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(*GRID_H_RANGE))
    w = int(rng.integers(*GRID_W_RANGE))

    max_fences = 1
    if min(h, w) >= 22:
        max_fences = 2
    if min(h, w) >= 26:
        max_fences = 3
    count = int(rng.integers(1, max_fences + 1))

    for _ in range(FENCE_COUNT_RETRIES):
        specs = [_build_fence_spec(rng) for _ in range(count)]
        placed = _place_specs(rng, h, w, specs)
        if placed is None:
            continue

        grid = np.full((h, w), BG, dtype=int)
        for spec in placed:
            _draw_spec(grid, spec)

        try:
            out = main(grid)
        except AssertionError:
            continue
        if np.array_equal(out, grid):
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATE_ATTEMPTS} attempts")
