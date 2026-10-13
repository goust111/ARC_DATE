import numpy as np

from collections import Counter, deque


BG = 0
MAX_FRAGMENT_CELLS = 4

LEGEND_RING_SIZE = 4
LEGEND_ROW_H = 4
LEGEND_ROW_GAP = 1
LEGEND_ROWS = 3
LEGEND_SAMPLE_TOP = 1
LEGEND_SAMPLE_LEFT = 8
LEGEND_W = 10
LEGEND_H = LEGEND_ROWS * LEGEND_ROW_H + (LEGEND_ROWS - 1) * LEGEND_ROW_GAP

GRID_H_RANGE = (24, 31)
GRID_W_RANGE = (24, 31)
RING_COUNT_CHOICES = (3, 4)
RING_SIDE_MIN = 6
RING_SIDE_MAX = 10
RING_RECT_GAP = 2
FRAGMENT_PAD = 1
GEN_RETRIES = 400

RNG = np.random.default_rng()


def _get_bg(grid: np.ndarray) -> int:
    values = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(values))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rect_border_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _canonical_shape(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    best = None
    for k in range(4):
        rot = _normalize_mask(np.rot90(mask, k))
        key = tuple(map(tuple, np.argwhere(rot)))
        if best is None or key < best:
            best = key
    assert best is not None
    return best


def _legend_origins(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    return [
        (0, 0),
        (0, w - LEGEND_W),
        (h - LEGEND_H, 0),
        (h - LEGEND_H, w - LEGEND_W),
    ]


def _small_ring_color(patch: np.ndarray, bg: int) -> int | None:
    if patch.shape != (LEGEND_RING_SIZE, LEGEND_RING_SIZE):
        return None
    border = _rect_border_mask(*patch.shape)
    colors = {int(v) for v in patch[border].tolist() if int(v) != bg}
    if len(colors) != 1:
        return None
    color = int(next(iter(colors)))
    if not np.all(patch[border] == color):
        return None
    if np.any(patch[~border] != bg):
        return None
    return color


def _decode_pattern(sample: np.ndarray, ring_color: int, bg: int) -> tuple[str, int | bool]:
    vals = {int(v) for v in sample.ravel().tolist()}
    if len(vals) == 1:
        value = int(sample[0, 0])
        return ("solid_ring" if value == ring_color else "solid_const", value)
    if sample[0, 0] == sample[0, 1] and sample[1, 0] == sample[1, 1] and sample[0, 0] != sample[1, 0]:
        assert {int(sample[0, 0]), int(sample[1, 0])} <= {bg, ring_color}
        return ("hstripe", bool(sample[0, 0] == ring_color))
    if sample[0, 0] == sample[1, 1] and sample[0, 1] == sample[1, 0] and sample[0, 0] != sample[0, 1]:
        assert {int(sample[0, 0]), int(sample[0, 1])} <= {bg, ring_color}
        return ("checker", bool(sample[0, 0] == ring_color))
    raise AssertionError("unrecognized legend sample")


def _marker_count(window: np.ndarray, row_start: int, ring_color: int) -> int:
    marker = window[row_start : row_start + LEGEND_ROW_H, 4:8] == ring_color
    return len(_components(marker))


def _find_legend_bbox(grid: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    h, w = grid.shape
    assert h >= LEGEND_H and w >= LEGEND_W
    found = None
    for r0, c0 in _legend_origins(grid.shape):
        window = grid[r0 : r0 + LEGEND_H, c0 : c0 + LEGEND_W]
        counts = []
        ok = True
        for row_idx in range(LEGEND_ROWS):
            row_start = row_idx * (LEGEND_ROW_H + LEGEND_ROW_GAP)
            color = _small_ring_color(window[row_start : row_start + LEGEND_ROW_H, :LEGEND_RING_SIZE], bg)
            if color is None:
                ok = False
                break
            count = _marker_count(window, row_start, color)
            sample = window[
                row_start + LEGEND_SAMPLE_TOP : row_start + LEGEND_SAMPLE_TOP + 2,
                LEGEND_SAMPLE_LEFT : LEGEND_SAMPLE_LEFT + 2,
            ]
            try:
                _decode_pattern(sample, color, bg)
            except AssertionError:
                ok = False
                break
            counts.append(count)
        if ok and sorted(counts) == [0, 1, 2]:
            assert found is None
            found = (int(r0), int(c0), int(r0 + LEGEND_H - 1), int(c0 + LEGEND_W - 1))
    assert found is not None, "legend not found"
    return found


def _parse_legend(grid: np.ndarray, legend_bbox: tuple[int, int, int, int], bg: int) -> dict[int, tuple[str, int | bool]]:
    r0, c0, r1, c1 = legend_bbox
    window = grid[r0 : r1 + 1, c0 : c1 + 1]
    mapping: dict[int, tuple[str, int | bool]] = {}
    for row_idx in range(LEGEND_ROWS):
        row_start = row_idx * (LEGEND_ROW_H + LEGEND_ROW_GAP)
        ring = window[row_start : row_start + LEGEND_ROW_H, :LEGEND_RING_SIZE]
        ring_color = int(_small_ring_color(ring, bg))
        count = _marker_count(window, row_start, ring_color)
        sample = window[
            row_start + LEGEND_SAMPLE_TOP : row_start + LEGEND_SAMPLE_TOP + 2,
            LEGEND_SAMPLE_LEFT : LEGEND_SAMPLE_LEFT + 2,
        ]
        mapping[count] = _decode_pattern(sample, ring_color, bg)
    assert sorted(mapping) == [0, 1, 2]
    return mapping


def _apply_pattern(
    out: np.ndarray,
    bbox: tuple[int, int, int, int],
    ring_color: int,
    pattern: tuple[str, int | bool],
    bg: int,
) -> None:
    r0, c0, r1, c1 = bbox
    kind, param = pattern
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if out[r, c] != bg:
                continue
            rr = r - (r0 + 1)
            cc = c - (c0 + 1)
            if kind == "solid_const":
                out[r, c] = int(param)
            elif kind == "solid_ring":
                out[r, c] = int(ring_color)
            elif kind == "hstripe":
                use_ring = (rr % 2 == 0) if bool(param) else (rr % 2 == 1)
                out[r, c] = int(ring_color) if use_ring else int(bg)
            elif kind == "checker":
                use_ring = ((rr + cc) % 2 == 0) if bool(param) else ((rr + cc) % 2 == 1)
                out[r, c] = int(ring_color) if use_ring else int(bg)
            else:
                raise AssertionError("unknown pattern type")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair missing ring segments using same-colored fragments, then fill each
    ring interior according to the legend entry for its original gap count.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    legend_bbox = _find_legend_bbox(g, bg)
    legend_map = _parse_legend(g, legend_bbox, bg)

    out = g.copy()
    legend_mask = np.zeros_like(g, dtype=bool)
    lr0, lc0, lr1, lc1 = legend_bbox
    legend_mask[lr0 : lr1 + 1, lc0 : lc1 + 1] = True

    for color in sorted({int(v) for v in np.unique(g[~legend_mask]).tolist() if int(v) != bg}):
        color_mask = (g == color) & (~legend_mask)
        comps = _components(color_mask)
        large = [comp for comp in comps if int(comp.shape[0]) > MAX_FRAGMENT_CELLS]
        if not large:
            continue
        ring_coords = np.concatenate(large, axis=0)
        bbox = _bbox(ring_coords)
        r0, c0, r1, c1 = bbox
        h = r1 - r0 + 1
        w = c1 - c0 + 1

        border = _rect_border_mask(h, w)
        present = np.zeros((h, w), dtype=bool)
        present[ring_coords[:, 0] - r0, ring_coords[:, 1] - c0] = True
        missing = border & (~present)
        gaps = _components(missing)
        gap_count = len(gaps)
        assert gap_count in legend_map

        fragments = [comp for comp in comps if int(comp.shape[0]) <= MAX_FRAGMENT_CELLS]
        assert len(fragments) == gap_count
        gap_shapes = Counter(_canonical_shape(comp) for comp in gaps)
        fragment_shapes = Counter(_canonical_shape(comp) for comp in fragments)
        assert gap_shapes == fragment_shapes

        for comp in fragments:
            out[comp[:, 0], comp[:, 1]] = bg

        for rr in range(h):
            for cc in range(w):
                if border[rr, cc]:
                    out[r0 + rr, c0 + cc] = color

        _apply_pattern(out, bbox, color, legend_map[gap_count], bg)

    return out


def _draw_border(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int, gaps: list[np.ndarray]) -> None:
    r0, c0, r1, c1 = bbox
    mask = _rect_border_mask(r1 - r0 + 1, c1 - c0 + 1)
    for gap in gaps:
        mask[gap[:, 0], gap[:, 1]] = False
    for rr, cc in np.argwhere(mask):
        grid[r0 + rr, c0 + cc] = color


def _shape_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _random_line_gap(h: int, w: int) -> np.ndarray:
    options = []
    for length in range(2, 5):
        if w - 2 >= length:
            for side in ("top", "bottom"):
                for start in range(1, w - length):
                    options.append((side, start, length))
        if h - 2 >= length:
            for side in ("left", "right"):
                for start in range(1, h - length):
                    options.append((side, start, length))
    assert options
    side, start, length = options[int(RNG.integers(len(options)))]
    coords = []
    if side == "top":
        coords = [(0, start + i) for i in range(length)]
    elif side == "bottom":
        coords = [(h - 1, start + i) for i in range(length)]
    elif side == "left":
        coords = [(start + i, 0) for i in range(length)]
    else:
        coords = [(start + i, w - 1) for i in range(length)]
    return np.array(coords, dtype=int)


def _random_l_gap(h: int, w: int) -> np.ndarray:
    options = []
    for vrun, hrun in ((2, 2), (2, 3), (3, 2)):
        if h >= vrun and w >= hrun:
            options.append(("tl", vrun, hrun))
            options.append(("tr", vrun, hrun))
            options.append(("bl", vrun, hrun))
            options.append(("br", vrun, hrun))
    assert options
    corner, vrun, hrun = options[int(RNG.integers(len(options)))]
    cells = set()
    if corner == "tl":
        for rr in range(vrun):
            cells.add((rr, 0))
        for cc in range(hrun):
            cells.add((0, cc))
    elif corner == "tr":
        for rr in range(vrun):
            cells.add((rr, w - 1))
        for cc in range(hrun):
            cells.add((0, w - 1 - cc))
    elif corner == "bl":
        for rr in range(vrun):
            cells.add((h - 1 - rr, 0))
        for cc in range(hrun):
            cells.add((h - 1, cc))
    else:
        for rr in range(vrun):
            cells.add((h - 1 - rr, w - 1))
        for cc in range(hrun):
            cells.add((h - 1, w - 1 - cc))
    return np.array(sorted(cells), dtype=int)


def _touches(a: np.ndarray, b: np.ndarray) -> bool:
    aset = {tuple(x) for x in a.tolist()}
    for rr, cc in b.tolist():
        for dr, dc in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
            if (rr + dr, cc + dc) in aset:
                return True
    return False


def _sample_gaps(h: int, w: int, gap_count: int) -> list[np.ndarray]:
    if gap_count == 0:
        return []
    for _ in range(200):
        gaps = []
        used = []
        ok = True
        for _idx in range(gap_count):
            gap = _random_line_gap(h, w) if float(RNG.random()) < 0.5 else _random_l_gap(h, w)
            if any(_touches(gap, prev) for prev in used):
                ok = False
                break
            used.append(gap)
            gaps.append(gap)
        if not ok:
            continue
        border = _rect_border_mask(h, w)
        for gap in gaps:
            border[gap[:, 0], gap[:, 1]] = False
        comps = _components(border)
        if not comps:
            continue
        if min(int(comp.shape[0]) for comp in comps) <= MAX_FRAGMENT_CELLS:
            continue
        missing = np.zeros((h, w), dtype=bool)
        for gap in gaps:
            missing[gap[:, 0], gap[:, 1]] = True
        if len(_components(missing)) != gap_count:
            continue
        return gaps
    raise AssertionError("failed to sample gaps")


def _pattern_sample(spec: tuple[str, int | bool], ring_color: int) -> np.ndarray:
    kind, param = spec
    sample = np.full((2, 2), BG, dtype=int)
    if kind == "solid_const":
        sample[:, :] = int(param)
    elif kind == "solid_ring":
        sample[:, :] = int(ring_color)
    elif kind == "hstripe":
        if bool(param):
            sample[0, :] = ring_color
        else:
            sample[1, :] = ring_color
    elif kind == "checker":
        for r in range(2):
            for c in range(2):
                use_ring = ((r + c) % 2 == 0) if bool(param) else ((r + c) % 2 == 1)
                sample[r, c] = int(ring_color) if use_ring else BG
    else:
        raise AssertionError("bad pattern spec")
    return sample


def _draw_legend(
    grid: np.ndarray,
    corner: int,
    row_colors: list[int],
    legend_patterns: dict[int, tuple[str, int | bool]],
) -> tuple[int, int, int, int]:
    h, w = grid.shape
    if corner == 0:
        r0, c0 = 0, 0
    elif corner == 1:
        r0, c0 = 0, w - LEGEND_W
    elif corner == 2:
        r0, c0 = h - LEGEND_H, 0
    else:
        r0, c0 = h - LEGEND_H, w - LEGEND_W

    for gap_count in range(LEGEND_ROWS):
        row_start = r0 + gap_count * (LEGEND_ROW_H + LEGEND_ROW_GAP)
        color = int(row_colors[gap_count])
        grid[row_start : row_start + 4, c0 : c0 + 4] = np.where(
            _rect_border_mask(4, 4),
            color,
            BG,
        )
        if gap_count >= 1:
            grid[row_start + 1, c0 + 5 : c0 + 7] = color
        if gap_count == 2:
            grid[row_start + 0, c0 + 5 : c0 + 7] = color
            grid[row_start + 2, c0 + 5 : c0 + 7] = color
            grid[row_start + 1, c0 + 5 : c0 + 7] = BG
        sample = _pattern_sample(legend_patterns[gap_count], color)
        grid[row_start + LEGEND_SAMPLE_TOP : row_start + LEGEND_SAMPLE_TOP + 2, c0 + LEGEND_SAMPLE_LEFT : c0 + LEGEND_SAMPLE_LEFT + 2] = sample
    return int(r0), int(c0), int(r0 + LEGEND_H - 1), int(c0 + LEGEND_W - 1)


def _expand_bbox(bbox: tuple[int, int, int, int], pad: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _place_ring_bboxes(
    shape: tuple[int, int],
    legend_bbox: tuple[int, int, int, int],
    sizes: list[tuple[int, int]],
) -> list[tuple[int, int, int, int]] | None:
    h, w = shape
    placed = [_expand_bbox(legend_bbox, 1, shape)]
    out = []
    for rh, rw in sizes:
        success = None
        for _ in range(300):
            r0 = int(RNG.integers(0, h - rh + 1))
            c0 = int(RNG.integers(0, w - rw + 1))
            bbox = (r0, c0, r0 + rh - 1, c0 + rw - 1)
            padded = _expand_bbox(bbox, RING_RECT_GAP, shape)
            if any(_intersects(padded, prev) for prev in placed):
                continue
            success = bbox
            placed.append(padded)
            out.append(bbox)
            break
        if success is None:
            return None
    return out


def _can_place_mask(blocked: np.ndarray, mask: np.ndarray, top: int, left: int, pad: int) -> bool:
    h, w = blocked.shape
    mh, mw = mask.shape
    if top < 0 or left < 0 or top + mh > h or left + mw > w:
        return False
    for rr, cc in np.argwhere(mask):
        r = top + int(rr)
        c = left + int(cc)
        r0 = max(0, r - pad)
        c0 = max(0, c - pad)
        r1 = min(h - 1, r + pad)
        c1 = min(w - 1, c + pad)
        if np.any(blocked[r0 : r1 + 1, c0 : c1 + 1]):
            return False
    return True


def _mark_mask(blocked: np.ndarray, mask: np.ndarray, top: int, left: int, pad: int) -> None:
    h, w = blocked.shape
    for rr, cc in np.argwhere(mask):
        r = top + int(rr)
        c = left + int(cc)
        r0 = max(0, r - pad)
        c0 = max(0, c - pad)
        r1 = min(h - 1, r + pad)
        c1 = min(w - 1, c + pad)
        blocked[r0 : r1 + 1, c0 : c1 + 1] = True


def _place_fragments(
    shape: tuple[int, int],
    blocked: np.ndarray,
    ring_color: int,
    gaps: list[np.ndarray],
) -> list[np.ndarray] | None:
    placed = []
    for gap in gaps:
        mask = _shape_mask(gap)
        rot = np.rot90(mask, int(RNG.integers(4)))
        success = None
        for _ in range(400):
            top = int(RNG.integers(0, shape[0] - rot.shape[0] + 1))
            left = int(RNG.integers(0, shape[1] - rot.shape[1] + 1))
            if not _can_place_mask(blocked, rot, top, left, FRAGMENT_PAD):
                continue
            success = (top, left, rot)
            break
        if success is None:
            return None
        top, left, rot = success
        coords = np.argwhere(rot) + np.array([top, left], dtype=int)
        placed.append(coords.astype(int))
        _mark_mask(blocked, rot, top, left, FRAGMENT_PAD)
    return placed


def _build_expected(
    inp: np.ndarray,
    rings: list[dict],
    legend_patterns: dict[int, tuple[str, int | bool]],
) -> np.ndarray:
    out = inp.copy()
    for ring in rings:
        color = int(ring["color"])
        bbox = tuple(ring["bbox"])
        for frag in ring["fragments"]:
            out[frag[:, 0], frag[:, 1]] = BG
        r0, c0, r1, c1 = bbox
        border = _rect_border_mask(r1 - r0 + 1, c1 - c0 + 1)
        for rr, cc in np.argwhere(border):
            out[r0 + rr, c0 + cc] = color
        _apply_pattern(out, bbox, color, legend_patterns[int(ring["gap_count"])], BG)
    return out


def _gen_attempt() -> np.ndarray | None:
    h = int(RNG.integers(*GRID_H_RANGE))
    w = int(RNG.integers(*GRID_W_RANGE))
    if h < LEGEND_H or w < LEGEND_W:
        return None
    grid = np.zeros((h, w), dtype=int)

    corner = int(RNG.integers(4))
    row_colors = [int(x) for x in RNG.choice(np.arange(1, 10), size=3, replace=False)]
    families = ["solid_const", "hstripe", "checker"]
    perm = [families[i] for i in RNG.permutation(len(families))]
    legend_patterns: dict[int, tuple[str, int | bool]] = {}
    for gap_count, family in enumerate(perm):
        if family == "solid_const":
            solid_choices = [c for c in range(1, 10) if c != row_colors[gap_count]]
            legend_patterns[gap_count] = (family, int(RNG.choice(solid_choices)))
        elif family == "hstripe":
            legend_patterns[gap_count] = (family, bool(RNG.integers(2)))
        else:
            legend_patterns[gap_count] = (family, bool(RNG.integers(2)))
    legend_bbox = _draw_legend(grid, corner, row_colors, legend_patterns)

    ring_count = int(RNG.choice(RING_COUNT_CHOICES))
    gap_counts = [0, 1, 2]
    while len(gap_counts) < ring_count:
        gap_counts.append(int(RNG.integers(3)))
    RNG.shuffle(gap_counts)

    ring_colors = [int(x) for x in RNG.choice(np.arange(1, 10), size=ring_count, replace=False)]
    sizes = [(int(RNG.integers(RING_SIDE_MIN, RING_SIDE_MAX)), int(RNG.integers(RING_SIDE_MIN, RING_SIDE_MAX))) for _ in range(ring_count)]
    boxes = _place_ring_bboxes(grid.shape, legend_bbox, sizes)
    if boxes is None:
        return None

    rings = []
    blocked = np.zeros_like(grid, dtype=bool)
    lr0, lc0, lr1, lc1 = _expand_bbox(legend_bbox, 1, grid.shape)
    blocked[lr0 : lr1 + 1, lc0 : lc1 + 1] = True

    for color, gap_count, bbox in zip(ring_colors, gap_counts, boxes):
        r0, c0, r1, c1 = bbox
        gaps = _sample_gaps(r1 - r0 + 1, c1 - c0 + 1, int(gap_count))
        _draw_border(grid, bbox, int(color), gaps)
        br0, bc0, br1, bc1 = _expand_bbox(bbox, 1, grid.shape)
        blocked[br0 : br1 + 1, bc0 : bc1 + 1] = True
        rings.append(
            {
                "color": int(color),
                "gap_count": int(gap_count),
                "bbox": tuple(int(x) for x in bbox),
                "gaps": gaps,
                "fragments": [],
            }
        )

    for ring in rings:
        frags = _place_fragments(grid.shape, blocked, int(ring["color"]), ring["gaps"])
        if frags is None:
            return None
        ring["fragments"] = frags
        for frag in frags:
            grid[frag[:, 0], frag[:, 1]] = int(ring["color"])

    expected = _build_expected(grid, rings, legend_patterns)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
