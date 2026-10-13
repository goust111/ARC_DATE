import numpy as np
from collections import Counter, deque


BG = 0
INDICATOR = 5
MAX_ATTEMPTS = 400
FULL_CANVAS = 30


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    arr = np.asarray(coords, dtype=int)
    return (
        int(arr[:, 0].min()),
        int(arr[:, 1].min()),
        int(arr[:, 0].max()),
        int(arr[:, 1].max()),
    )


def _ring_order(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    order: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        order.append((r0, c))
    for r in range(r0 + 1, r1 + 1):
        order.append((r, c1))
    for c in range(c1 - 1, c0 - 1, -1):
        order.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        order.append((r, c0))
    return order


def _mode(values: list[int]) -> int:
    counts = Counter(int(v) for v in values)
    return max(counts.items(), key=lambda item: (item[1], -item[0]))[0]


def _parse_indicator(coords: list[tuple[int, int]]) -> tuple[bool, int]:
    coord_set = set(coords)
    endpoints: list[tuple[int, int]] = []
    elbow: tuple[int, int] | None = None

    for cell in coords:
        r, c = cell
        nbrs = [
            (r + dr, c + dc)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
            if (r + dr, c + dc) in coord_set
        ]
        if len(nbrs) == 1:
            endpoints.append(cell)
        elif len(nbrs) == 2:
            (r1, c1), (r2, c2) = nbrs
            if r1 != r2 and c1 != c2:
                elbow = cell

    if elbow is None:
        raise AssertionError("expected an L-shaped indicator")
    assert len(endpoints) == 2, "indicator should have two endpoints"

    horiz_ep = next(ep for ep in endpoints if ep[0] == elbow[0])
    vert_ep = next(ep for ep in endpoints if ep[1] == elbow[1])
    h_sign = 1 if horiz_ep[1] > elbow[1] else -1
    v_sign = 1 if vert_ep[0] > elbow[0] else -1
    steps = max(abs(horiz_ep[1] - elbow[1]) + 1, abs(vert_ep[0] - elbow[0]) + 1)
    clockwise = h_sign == v_sign
    return clockwise, steps


def _shift_markers(
    order: list[tuple[int, int]],
    markers: list[tuple[int, int]],
    clockwise: bool,
    steps: int,
) -> list[tuple[tuple[int, int], int]]:
    shift = steps % len(order)
    if not clockwise:
        shift = -shift
    moved: list[tuple[tuple[int, int], int]] = []
    for idx, color in markers:
        moved.append((order[(idx + shift) % len(order)], int(color)))
    return moved


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move every marker around its rectangular ring according to the adjacent gray L-indicator.

    Input structure:
    - The background is black.
    - Each object is a one-pixel-thick rectangular ring of one base color, with a few ring cells replaced by marker colors.
    - A gray L-shaped indicator sits directly outside one side of each ring and touches it by side-adjacency.

    Rule:
    - Separate gray indicators from the colored ring components.
    - For each ring, recover the clockwise perimeter order of its rectangular border cells.
    - Read the adjacent gray L: its elbow orientation gives clockwise vs. counterclockwise motion, and the longer arm length gives the number of steps.
    - Restore every marker's old cell to the ring base color, then place the marker that many perimeter steps away along the ring.

    Key insight:
    The ring itself is the motion path, while the nearby gray L encodes two parameters at once: turn orientation selects the direction and arm length selects the cyclic shift size.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    ring_mask = (g != BG) & (g != INDICATOR)
    ring_comps = _components(ring_mask)
    indicator_comps = _components(g == INDICATOR)
    unused_indicators = [list(comp) for comp in indicator_comps]

    for ring in ring_comps:
        bbox = _bbox(ring)
        r0, c0, r1, c1 = bbox
        assert r1 - r0 + 1 >= 3 and c1 - c0 + 1 >= 3, "ring must enclose an interior"
        order = _ring_order(bbox)
        assert set(order) == set(ring), "foreground component must be a rectangular ring"

        values = [int(g[r, c]) for r, c in order]
        base = _mode(values)
        markers = [(idx, val) for idx, val in enumerate(values) if val != base]

        ring_set = set(ring)
        indicator_idx: int | None = None
        for idx, comp in enumerate(unused_indicators):
            if any(
                (r + dr, c + dc) in ring_set
                for r, c in comp
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
            ):
                indicator_idx = idx
                break
        assert indicator_idx is not None, "each ring needs one adjacent indicator"
        indicator = unused_indicators.pop(indicator_idx)

        clockwise, steps = _parse_indicator(indicator)
        for r, c in order:
            out[r, c] = base
        for (nr, nc), color in _shift_markers(order, markers, clockwise, steps):
            out[nr, nc] = color

    return out


def _indicator_coords(
    ring_h: int,
    ring_w: int,
    side: str,
    clockwise: bool,
    near_len: int,
    away_len: int,
    offset: int,
) -> list[tuple[int, int]]:
    if side == "top":
        near = [(-1, offset + i) for i in range(near_len)]
        elbow = near[-1] if clockwise else near[0]
        away = [(elbow[0] - i, elbow[1]) for i in range(1, away_len)]
    elif side == "bottom":
        near = [(ring_h, offset + i) for i in range(near_len)]
        elbow = near[0] if clockwise else near[-1]
        away = [(elbow[0] + i, elbow[1]) for i in range(1, away_len)]
    elif side == "left":
        near = [(offset + i, -1) for i in range(near_len)]
        elbow = near[-1] if clockwise else near[0]
        away = [(elbow[0], elbow[1] - i) for i in range(1, away_len)]
    elif side == "right":
        near = [(offset + i, ring_w) for i in range(near_len)]
        elbow = near[0] if clockwise else near[-1]
        away = [(elbow[0], elbow[1] + i) for i in range(1, away_len)]
    else:
        raise ValueError(f"unknown side: {side}")
    return near + away


def _make_object(rng: np.random.Generator, track_color: int) -> tuple[np.ndarray, np.ndarray] | None:
    ring_h = int(rng.integers(4, 8))
    ring_w = int(rng.integers(4, 10))
    side = str(rng.choice(np.array(["top", "bottom", "left", "right"], dtype=object)))
    side_span = ring_w if side in ("top", "bottom") else ring_h

    near_cap = min(5, side_span)
    if near_cap < 2:
        return None
    near_len = int(rng.integers(2, near_cap + 1))
    away_len = int(rng.integers(2, 6))
    if near_len == away_len:
        away_len = away_len + 1 if away_len < 5 else away_len - 1

    clockwise = bool(rng.integers(2))
    offset = int(rng.integers(0, side_span - near_len + 1))
    indicator = _indicator_coords(ring_h, ring_w, side, clockwise, near_len, away_len, offset)

    order = _ring_order((0, 0, ring_h - 1, ring_w - 1))
    perimeter = len(order)
    marker_count = int(rng.integers(1, min(5, perimeter // 3 + 1)))
    marker_idx = [int(x) for x in rng.choice(perimeter, size=marker_count, replace=False)]
    marker_palette = [c for c in range(1, 10) if c not in (INDICATOR, track_color)]
    marker_colors = [int(rng.choice(marker_palette)) for _ in range(marker_count)]

    all_coords = order + indicator
    min_r = min(r for r, _ in all_coords)
    min_c = min(c for _, c in all_coords)
    max_r = max(r for r, _ in all_coords)
    max_c = max(c for _, c in all_coords)

    shift_r = 1 - min_r
    shift_c = 1 - min_c
    patch_h = max_r - min_r + 3
    patch_w = max_c - min_c + 3

    inp = np.zeros((patch_h, patch_w), dtype=int)
    out = np.zeros((patch_h, patch_w), dtype=int)
    shifted_order = [(r + shift_r, c + shift_c) for r, c in order]
    shifted_indicator = [(r + shift_r, c + shift_c) for r, c in indicator]

    for canvas in (inp, out):
        for r, c in shifted_order:
            canvas[r, c] = track_color
        for r, c in shifted_indicator:
            canvas[r, c] = INDICATOR

    markers = list(zip(marker_idx, marker_colors))
    for idx, color in markers:
        r, c = shifted_order[idx]
        inp[r, c] = color

    steps = max(near_len, away_len)
    for (r, c), color in _shift_markers(shifted_order, markers, clockwise, steps):
        out[r, c] = color

    return inp, out


def _place_objects(
    rng: np.random.Generator,
    objects: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray] | None:
    inp = np.zeros((FULL_CANVAS, FULL_CANVAS), dtype=int)
    out = np.zeros((FULL_CANVAS, FULL_CANVAS), dtype=int)
    reserved = np.zeros((FULL_CANVAS, FULL_CANVAS), dtype=bool)

    order = sorted(
        range(len(objects)),
        key=lambda idx: objects[idx][0].shape[0] * objects[idx][0].shape[1],
        reverse=True,
    )

    for idx in order:
        patch_in, patch_out = objects[idx]
        ph, pw = patch_in.shape
        if ph > FULL_CANVAS - 2 or pw > FULL_CANVAS - 2:
            return None

        row_min = 1
        row_max = FULL_CANVAS - ph - 1
        col_min = 1
        col_max = FULL_CANVAS - pw - 1
        if row_max < row_min or col_max < col_min:
            return None

        placed = False
        for _ in range(300):
            r0 = int(rng.integers(row_min, row_max + 1))
            c0 = int(rng.integers(col_min, col_max + 1))
            if reserved[r0 : r0 + ph, c0 : c0 + pw].any():
                continue
            block_in = inp[r0 : r0 + ph, c0 : c0 + pw]
            block_out = out[r0 : r0 + ph, c0 : c0 + pw]
            mask_in = patch_in != BG
            mask_out = patch_out != BG
            block_in[mask_in] = patch_in[mask_in]
            block_out[mask_out] = patch_out[mask_out]
            reserved[r0 : r0 + ph, c0 : c0 + pw] = True
            placed = True
            break

        if not placed:
            return None

    used = inp != BG
    rows, cols = np.where(used)
    if len(rows) == 0:
        return None
    r0 = max(0, int(rows.min()) - int(rng.integers(0, 3)))
    c0 = max(0, int(cols.min()) - int(rng.integers(0, 3)))
    r1 = min(FULL_CANVAS - 1, int(rows.max()) + int(rng.integers(0, 3)))
    c1 = min(FULL_CANVAS - 1, int(cols.max()) + int(rng.integers(0, 3)))
    return inp[r0 : r1 + 1, c0 : c1 + 1], out[r0 : r1 + 1, c0 : c1 + 1]


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    n_objects = int(rng.integers(1, 4))
    track_palette = [1, 2, 3, 4, 6, 7, 8, 9]
    track_colors = [int(x) for x in rng.choice(track_palette, size=n_objects, replace=False)]

    objects: list[tuple[np.ndarray, np.ndarray]] = []
    for color in track_colors:
        built = _make_object(rng, color)
        if built is None:
            return None
        objects.append(built)

    placed = _place_objects(rng, objects)
    if placed is None:
        return None
    inp, expected = placed

    if inp.ndim != 2 or max(inp.shape) > 30:
        return None
    if expected.ndim != 2 or max(expected.shape) > 30:
        return None
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
