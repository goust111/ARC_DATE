import numpy as np


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or g[r, c] == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and g[nr, nc] != bg:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            components.append({"coords": arr, "bbox": (int(r0), int(c0), int(r1), int(c1))})
    return components


def _marker_color(values: np.ndarray) -> int:
    colors, counts = np.unique(values, return_counts=True)
    assert len(colors) == 2, "each object must use one body color and one marker color"
    order = np.argsort(counts)
    assert counts[order[0]] < counts[order[1]], "marker color must be strictly rarer than body color"
    return int(colors[order[0]])


def _paint(out: np.ndarray, r: int, c: int, color: int, bg: int) -> None:
    assert 0 <= r < out.shape[0] and 0 <= c < out.shape[1], "shifted pixel must stay inside the grid"
    prev = int(out[r, c])
    assert prev in (bg, color), "shifted objects must not overlap"
    out[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove each fracture marker line and push the surrounding object pixels one cell away from it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    out = np.full_like(g, bg)
    components = find_components(g, bg)
    assert components, "expected at least one fractured object"

    for comp in components:
        coords = comp["coords"]
        r0, c0, r1, c1 = comp["bbox"]
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        assert height != width, "object bbox must determine a unique orientation"

        values = g[coords[:, 0], coords[:, 1]]
        marker = _marker_color(values)
        marker_coords = coords[values == marker]
        body_coords = coords[values != marker]
        assert len(marker_coords) > 0 and len(body_coords) > 0

        if width > height:
            per_row: dict[int, int] = {}
            for r, c in marker_coords:
                rr = int(r)
                assert rr not in per_row, "wide object must have exactly one marker per row"
                per_row[rr] = int(c)
            assert set(per_row) == set(range(r0, r1 + 1)), "wide object marker must span all rows"

            for r, c in body_coords:
                rr = int(r)
                cc = int(c)
                split = per_row[rr]
                assert cc != split
                shifted_c = cc - 1 if cc < split else cc + 1
                _paint(out, rr, shifted_c, int(g[rr, cc]), bg)
        else:
            per_col: dict[int, int] = {}
            for r, c in marker_coords:
                cc = int(c)
                assert cc not in per_col, "tall object must have exactly one marker per column"
                per_col[cc] = int(r)
            assert set(per_col) == set(range(c0, c1 + 1)), "tall object marker must span all columns"

            for r, c in body_coords:
                rr = int(r)
                cc = int(c)
                split = per_col[cc]
                assert rr != split
                shifted_r = rr - 1 if rr < split else rr + 1
                _paint(out, shifted_r, cc, int(g[rr, cc]), bg)

    return out


BACKGROUND_COLORS = tuple(range(10))
MIN_GRID_SIDE = 12
MAX_GRID_SIDE = 20
MAX_OBJECTS = 5
MAX_RETRIES = 300
MAX_NOTCH = 2
SAFETY_PAD = 2


def _random_walk(length: int, start: int, lo: int, hi: int, rng: np.random.Generator) -> list[int]:
    seq = [start]
    for _ in range(length - 1):
        cur = seq[-1]
        step_choices = [step for step in (-1, 0, 1) if lo <= cur + step <= hi]
        seq.append(int(cur + rng.choice(step_choices)))
    return seq


def _wide_patch(height: int, width: int, body: int, marker: int, rng: np.random.Generator) -> np.ndarray:
    patch = np.full((height, width), -1, dtype=int)
    split_lo = 1
    split_hi = width - 2
    splits = _random_walk(height, int(rng.integers(split_lo, split_hi + 1)), split_lo, split_hi, rng)

    left_caps = [min(MAX_NOTCH, split - 1) for split in splits]
    right_caps = [min(MAX_NOTCH, width - split - 2) for split in splits]
    lefts = [int(rng.integers(0, cap + 1)) for cap in left_caps]
    rights = [int(rng.integers(0, cap + 1)) for cap in right_caps]
    lefts[int(rng.integers(0, height))] = 0
    rights[int(rng.integers(0, height))] = 0

    for r, split in enumerate(splits):
        left_start = lefts[r]
        right_end = width - 1 - rights[r]
        assert left_start <= split - 1
        assert split + 1 <= right_end
        patch[r, left_start:split] = body
        patch[r, split] = marker
        patch[r, split + 1 : right_end + 1] = body

    return patch


def _make_patch(orientation: str, body: int, marker: int, rng: np.random.Generator) -> np.ndarray:
    short_side = int(rng.integers(2, 6))
    long_side = int(rng.integers(short_side + 1, min(short_side + 7, 11)))
    base = _wide_patch(short_side, long_side, body, marker, rng)
    return base if orientation == "wide" else base.T.copy()


def _can_place(mask: np.ndarray, top: int, left: int, shape: tuple[int, int]) -> bool:
    h, w = shape
    return not mask[top - SAFETY_PAD : top + h + SAFETY_PAD, left - SAFETY_PAD : left + w + SAFETY_PAD].any()


def _place_patch(grid: np.ndarray, mask: np.ndarray, patch: np.ndarray, bg: int, rng: np.random.Generator) -> bool:
    gh, gw = grid.shape
    ph, pw = patch.shape
    if ph + 2 * SAFETY_PAD > gh or pw + 2 * SAFETY_PAD > gw:
        return False

    candidates = [
        (r, c)
        for r in range(SAFETY_PAD, gh - ph - SAFETY_PAD + 1)
        for c in range(SAFETY_PAD, gw - pw - SAFETY_PAD + 1)
        if _can_place(mask, r, c, patch.shape)
    ]
    if not candidates:
        return False

    top, left = candidates[int(rng.integers(0, len(candidates)))]
    sub = grid[top : top + ph, left : left + pw]
    visible = patch != -1
    assert np.all(sub[visible] == bg)
    sub[visible] = patch[visible]
    mask[top : top + ph, left : left + pw] |= visible
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(MIN_GRID_SIDE, MAX_GRID_SIDE + 1))
    width = int(rng.integers(MIN_GRID_SIDE, MAX_GRID_SIDE + 1))
    bg = int(rng.choice(BACKGROUND_COLORS))
    grid = np.full((height, width), bg, dtype=int)
    mask = np.zeros((height, width), dtype=bool)

    palette = [color for color in range(10) if color != bg]
    object_count = int(rng.integers(1, MAX_OBJECTS + 1))
    orientations = rng.choice(["wide", "tall"], size=object_count, replace=True)

    patches = []
    for orientation in orientations:
        body, marker = rng.choice(palette, size=2, replace=False)
        patch = _make_patch(str(orientation), int(body), int(marker), rng)
        patches.append(patch)

    for patch in sorted(patches, key=lambda x: x.shape[0] * x.shape[1], reverse=True):
        if not _place_patch(grid, mask, patch, bg, rng):
            return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fractured-object grid")
