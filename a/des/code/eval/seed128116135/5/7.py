import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = tuple(c for c in range(10) if c not in (BLACK, GREY))
GEN_ATTEMPTS = 300


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _label_components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]

    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in directions:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comp = np.zeros((h, w), dtype=bool)
            for rr, cc in cells:
                comp[rr, cc] = True
            components.append(comp)
    return components


def _find_frame(grid: np.ndarray) -> tuple[int, int, int, int, int]:
    grey_components = _label_components(grid == GREY, connectivity=4)
    candidates = []
    for comp in grey_components:
        coords = np.argwhere(comp)
        if coords.size == 0:
            continue
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        local = comp[r0 : r1 + 1, c0 : c1 + 1]
        h, w = local.shape
        if h < 5 or w < 3:
            continue
        divider = None
        for rr in range(1, h - 1):
            if not local[rr].all():
                continue
            top_h = rr - 1
            bottom_h = h - rr - 2
            if top_h <= 0 or top_h != bottom_h:
                continue
            expected = np.zeros_like(local, dtype=bool)
            expected[0, :] = True
            expected[-1, :] = True
            expected[:, 0] = True
            expected[:, -1] = True
            expected[rr, :] = True
            if np.array_equal(local, expected):
                divider = r0 + rr
                break
        if divider is not None:
            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            candidates.append((area, r0, c0, r1, c1, divider))
    assert candidates, "failed to locate frame"
    candidates.sort(reverse=True)
    _, r0, c0, r1, c1, divider = candidates[0]
    return r0, c0, r1, c1, divider


def _learn_mapping(top: np.ndarray, bottom: np.ndarray) -> dict[int, int]:
    mapping = {BLACK: BLACK}
    for color in np.unique(top):
        positions = top == int(color)
        targets = np.unique(bottom[positions])
        assert len(targets) == 1, "sample mapping must be pointwise consistent"
        mapping[int(color)] = int(targets[0])
    assert len(set(mapping.values())) == len(mapping), "mapping must be one-to-one"
    return mapping


def _classify_symmetry(arr: np.ndarray) -> str:
    horiz = np.array_equal(arr, np.fliplr(arr))
    vert = np.array_equal(arr, np.flipud(arr))
    rot = np.array_equal(arr, np.rot90(arr, 2))
    if horiz and not vert and not rot:
        return "horizontal"
    if vert and not horiz and not rot:
        return "vertical"
    if rot and not horiz and not vert:
        return "rotation"
    if not horiz and not vert and not rot:
        return "none"
    return "ambiguous"


def _apply_mapping(arr: np.ndarray, kind: str, mapping: dict[int, int]) -> np.ndarray:
    out = arr.copy()
    h, w = arr.shape
    if kind == "horizontal":
        rs = slice(0, h)
        cs = slice(w // 2, w)
    elif kind == "vertical":
        rs = slice(h // 2, h)
        cs = slice(0, w)
    elif kind == "rotation":
        rs = slice(h // 2, h)
        cs = slice(w // 2, w)
    else:
        rs = slice(0, h)
        cs = slice(0, w)

    sub = out[rs, cs]
    original = sub.copy()
    for color, target in mapping.items():
        sub[original == int(color)] = int(target)
    out[rs, cs] = sub
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn a pixelwise color mapping from the framed sample pair, then apply it
    to each outside pattern on a symmetry-dependent subregion.
    """
    g = np.asarray(grid, dtype=int)
    r0, c0, r1, c1, divider = _find_frame(g)
    top = g[r0 + 1 : divider, c0 + 1 : c1]
    bottom = g[divider + 1 : r1, c0 + 1 : c1]
    mapping = _learn_mapping(top, bottom)

    out = g.copy()
    frame_box = np.zeros_like(g, dtype=bool)
    frame_box[r0 : r1 + 1, c0 : c1 + 1] = True
    outside = (g != BLACK) & (~frame_box)

    for comp in _label_components(outside, connectivity=8):
        coords = np.argwhere(comp)
        if coords.size == 0:
            continue
        br0, bc0, br1, bc1 = _bbox_from_coords(coords)
        local_mask = comp[br0 : br1 + 1, bc0 : bc1 + 1]
        arr = g[br0 : br1 + 1, bc0 : bc1 + 1].copy()
        arr[~local_mask] = BLACK
        kind = _classify_symmetry(arr)
        if kind == "ambiguous":
            kind = "none"
        transformed = _apply_mapping(arr, kind, mapping)
        patch = out[br0 : br1 + 1, bc0 : bc1 + 1]
        patch[local_mask] = transformed[local_mask]
        out[br0 : br1 + 1, bc0 : bc1 + 1] = patch
    return out


def _random_derangement(colors: list[int], rng: np.random.Generator) -> dict[int, int]:
    perm = colors[:]
    for _ in range(100):
        rng.shuffle(perm)
        if all(a != b for a, b in zip(colors, perm)):
            return {BLACK: BLACK, **{int(a): int(b) for a, b in zip(colors, perm)}}
    raise AssertionError("failed to sample derangement")


def _make_sample_pattern(h: int, w: int, colors: list[int], rng: np.random.Generator) -> np.ndarray:
    values = [BLACK] + colors
    probs = np.array([0.25] + [0.75 / len(colors)] * len(colors))
    arr = rng.choice(values, size=(h, w), p=probs).astype(int)
    for color in colors:
        rr = int(rng.integers(0, h))
        cc = int(rng.integers(0, w))
        arr[rr, cc] = int(color)
    return arr


def _random_color_block(shape: tuple[int, int], colors: list[int], rng: np.random.Generator) -> np.ndarray:
    return rng.choice(colors, size=shape).astype(int)


def _make_pattern(kind: str, colors: list[int], rng: np.random.Generator) -> np.ndarray:
    for _ in range(200):
        if kind == "none":
            h = int(rng.integers(3, 6))
            w = int(rng.integers(3, 6))
            arr = _random_color_block((h, w), colors, rng)
        elif kind == "horizontal":
            h = int(rng.integers(3, 6))
            half = int(rng.integers(2, 4))
            left = _random_color_block((h, half), colors, rng)
            arr = np.concatenate([left, np.fliplr(left)], axis=1)
        elif kind == "vertical":
            half = int(rng.integers(2, 4))
            w = int(rng.integers(3, 6))
            top = _random_color_block((half, w), colors, rng)
            arr = np.concatenate([top, np.flipud(top)], axis=0)
        else:
            h = int(rng.integers(4, 7))
            w = int(rng.integers(4, 7))
            if h % 2 == 1:
                h += 1
            if w % 2 == 1:
                w += 1
            arr = np.zeros((h, w), dtype=int)
            filled = np.zeros((h, w), dtype=bool)
            for r in range(h):
                for c in range(w):
                    if filled[r, c]:
                        continue
                    rr = h - 1 - r
                    cc = w - 1 - c
                    color = int(rng.choice(colors))
                    arr[r, c] = color
                    arr[rr, cc] = color
                    filled[r, c] = True
                    filled[rr, cc] = True
        if _classify_symmetry(arr) == kind:
            return arr
    raise AssertionError(f"failed to create {kind} pattern")


def _place_pattern(
    grid: np.ndarray,
    pattern: np.ndarray,
    row_range: tuple[int, int],
    col_range: tuple[int, int],
    gap: int,
    rng: np.random.Generator,
) -> bool:
    ph, pw = pattern.shape
    rmin, rmax = row_range
    cmin, cmax = col_range
    if rmin > rmax or cmin > cmax:
        return False
    for _ in range(80):
        r0 = int(rng.integers(rmin, rmax + 1))
        c0 = int(rng.integers(cmin, cmax + 1))
        rr0 = max(0, r0 - gap)
        cc0 = max(0, c0 - gap)
        rr1 = min(grid.shape[0], r0 + ph + gap)
        cc1 = min(grid.shape[1], c0 + pw + gap)
        if np.any(grid[rr0:rr1, cc0:cc1] != BLACK):
            continue
        grid[r0 : r0 + ph, c0 : c0 + pw] = pattern
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    sample_h = int(rng.integers(3, 5))
    sample_w = int(rng.integers(4, 7))
    colors = list(map(int, rng.choice(COLOR_POOL, size=int(rng.integers(3, 6)), replace=False)))
    mapping = _random_derangement(colors, rng)

    top_sample = _make_sample_pattern(sample_h, sample_w, colors, rng)
    bottom_sample = top_sample.copy()
    for color, target in mapping.items():
        bottom_sample[top_sample == int(color)] = int(target)

    kinds = ["none", "horizontal", "vertical", "rotation"]
    rng.shuffle(kinds)
    patterns = {kind: _make_pattern(kind, colors, rng) for kind in kinds}

    tl_kind, tr_kind, bl_kind, br_kind = kinds
    tl = patterns[tl_kind]
    tr = patterns[tr_kind]
    bl = patterns[bl_kind]
    br = patterns[br_kind]

    sep = 1
    top_space = max(tl.shape[0], tr.shape[0]) + sep + 1 + int(rng.integers(0, 2))
    bottom_space = max(bl.shape[0], br.shape[0]) + sep + 1 + int(rng.integers(0, 2))
    left_space = max(tl.shape[1], bl.shape[1]) + sep + 1 + int(rng.integers(0, 2))
    right_space = max(tr.shape[1], br.shape[1]) + sep + 1 + int(rng.integers(0, 2))

    frame_h = 2 * sample_h + 3
    frame_w = sample_w + 2
    h = top_space + frame_h + bottom_space
    w = left_space + frame_w + right_space
    if h > 30 or w > 30:
        return None

    grid = np.zeros((h, w), dtype=int)
    fr0 = top_space
    fc0 = left_space
    fr1 = fr0 + frame_h - 1
    fc1 = fc0 + frame_w - 1
    divider = fr0 + sample_h + 1

    grid[fr0, fc0 : fc1 + 1] = GREY
    grid[fr1, fc0 : fc1 + 1] = GREY
    grid[fr0 : fr1 + 1, fc0] = GREY
    grid[fr0 : fr1 + 1, fc1] = GREY
    grid[divider, fc0 : fc1 + 1] = GREY
    grid[fr0 + 1 : divider, fc0 + 1 : fc1] = top_sample
    grid[divider + 1 : fr1, fc0 + 1 : fc1] = bottom_sample

    placements = [
        (tl, (1, fr0 - sep - tl.shape[0]), (1, fc0 - sep - tl.shape[1])),
        (tr, (1, fr0 - sep - tr.shape[0]), (fc1 + sep + 1, w - tr.shape[1] - 1)),
        (bl, (fr1 + sep + 1, h - bl.shape[0] - 1), (1, fc0 - sep - bl.shape[1])),
        (br, (fr1 + sep + 1, h - br.shape[0] - 1), (fc1 + sep + 1, w - br.shape[1] - 1)),
    ]

    for pattern, rows, cols in placements:
        if rows[0] > rows[1] or cols[0] > cols[1]:
            return None
        if not _place_pattern(grid, pattern, rows, cols, gap=0, rng=rng):
            return None

    try:
        solved = main(grid)
    except Exception:
        return None
    if solved.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
