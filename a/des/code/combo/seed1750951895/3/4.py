import numpy as np


AZURE = 8
_RNG = np.random.default_rng()
_DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_FG_COLORS = tuple(c for c in range(10) if c != AZURE)
_MAX_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in _DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                }
            )
    return comps


def _symmetry_type(mask: np.ndarray) -> str:
    horiz = bool(np.array_equal(mask, np.flipud(mask)))
    vert = bool(np.array_equal(mask, np.fliplr(mask)))
    rot = bool(np.array_equal(mask, np.flipud(np.fliplr(mask))))
    if horiz and vert:
        return "full"
    if horiz:
        return "h_only"
    if vert:
        return "v_only"
    if rot:
        return "rot_only"
    raise AssertionError("object does not match any allowed symmetry class")


def _draw_symbol(out: np.ndarray, bbox: tuple[int, int, int, int], color: int, sym_type: str) -> None:
    r0, c0, r1, c1 = bbox
    cr = (r0 + r1) // 2
    cc = (c0 + c1) // 2
    if sym_type == "h_only":
        out[cr, c0 : c1 + 1] = color
        return
    if sym_type == "v_only":
        out[r0 : r1 + 1, cc] = color
        return
    if sym_type == "rot_only":
        out[cr, cc] = color
        return
    if sym_type == "full":
        for rr, cc2 in ((cr, cc), (cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
            if 0 <= rr < out.shape[0] and 0 <= cc2 < out.shape[1]:
                out[rr, cc2] = color
        return
    raise AssertionError("unknown symmetry type")


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = get_bg(g)
    comps = _components(g, bg)
    assert comps

    out = np.full_like(g, bg)
    for comp in comps:
        sym_type = _symmetry_type(comp["mask"])
        _draw_symbol(out, comp["bbox"], int(comp["color"]), sym_type)
    return out


def _mask_from_coords(coords: set[tuple[int, int]]) -> np.ndarray:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _is_connected(mask: np.ndarray) -> bool:
    points = np.argwhere(mask)
    if len(points) == 0:
        return False
    start = tuple(map(int, points[0]))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for dr, dc in _DIRS4:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1]):
                continue
            if not mask[nr, nc] or (nr, nc) in seen:
                continue
            seen.add((nr, nc))
            stack.append((nr, nc))
    return len(seen) == len(points)


def _build_h_only() -> np.ndarray:
    for _ in range(40):
        h = int(_RNG.choice([3, 5, 7]))
        w = int(_RNG.choice([3, 5, 7]))
        top_h = h // 2
        mid = top_h
        coords = {(mid, c) for c in range(w)}
        left_span = int(_RNG.integers(1, w + 1))
        right_span = int(_RNG.integers(1, w + 1))
        coords.update((mid - 1, c) for c in range(left_span))
        coords.update((mid - 1, c) for c in range(w - right_span, w))
        if top_h >= 2 and _RNG.random() < 0.6:
            anchor_col = int(_RNG.integers(0, w))
            coords.add((0, anchor_col))
            coords.add((1, anchor_col))
        mirrored = set(coords)
        for r, c in list(coords):
            mirrored.add((h - 1 - r, c))
        mask = _mask_from_coords(mirrored)
        if _is_connected(mask) and _symmetry_type(mask) == "h_only":
            return mask
    return np.array(
        [
            [1, 0, 0],
            [1, 1, 1],
            [1, 0, 0],
        ],
        dtype=bool,
    )


def _build_v_only() -> np.ndarray:
    return _build_h_only().T


def _build_rot_only() -> np.ndarray:
    for _ in range(40):
        h = int(_RNG.choice([3, 5, 7]))
        w = int(_RNG.choice([3, 5, 7]))
        cr = h // 2
        cc = w // 2
        arm = int(_RNG.integers(1, min(cr, cc) + 1))
        coords = {(cr, c) for c in range(cc - arm, cc + arm + 1)}
        nub = int(_RNG.integers(1, min(cr, cc) + 1))
        coords.add((cr - nub, cc - arm))
        coords.add((cr + nub, cc + arm))
        if _RNG.random() < 0.5 and arm + 1 <= cc and nub + 1 <= cr:
            coords.add((cr, cc - arm - 1))
            coords.add((cr, cc + arm + 1))
        mask = _mask_from_coords(coords)
        if mask.shape != (h, w):
            continue
        if _is_connected(mask) and _symmetry_type(mask) == "rot_only":
            return mask
    return np.array(
        [
            [1, 1, 0],
            [0, 1, 0],
            [0, 1, 1],
        ],
        dtype=bool,
    )


def _build_full() -> np.ndarray:
    h = int(_RNG.choice([3, 5, 7]))
    w = int(_RNG.choice([3, 5, 7]))
    cr = h // 2
    cc = w // 2
    vr = int(_RNG.integers(1, cr + 1))
    hr = int(_RNG.integers(1, cc + 1))
    mask = np.zeros((h, w), dtype=bool)
    mask[cr - vr : cr + vr + 1, cc] = True
    mask[cr, cc - hr : cc + hr + 1] = True
    if h >= 5 and w >= 5 and _RNG.random() < 0.5:
        pad_r = int(_RNG.integers(1, cr + 1))
        pad_c = int(_RNG.integers(1, cc + 1))
        mask[cr - pad_r : cr + pad_r + 1, cc - pad_c : cc + pad_c + 1] |= np.array(
            [[False]], dtype=bool
        )
    assert _symmetry_type(mask) == "full"
    return mask


def _sample_mask(kind: str) -> np.ndarray:
    if kind == "h_only":
        return _build_h_only()
    if kind == "v_only":
        return _build_v_only()
    if kind == "rot_only":
        return _build_rot_only()
    if kind == "full":
        return _build_full()
    raise AssertionError("unknown generator kind")


def _can_place(grid: np.ndarray, top: int, left: int, mask: np.ndarray, bg: int) -> bool:
    h, w = mask.shape
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(grid.shape[0], top + h + 1)
    c1 = min(grid.shape[1], left + w + 1)
    return bool(np.all(grid[r0:r1, c0:c1] == bg))


def _place_mask(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    rows, cols = np.where(mask)
    grid[top + rows, left + cols] = color


def _gen_attempt() -> np.ndarray | None:
    height = int(_RNG.integers(14, 23))
    width = int(_RNG.integers(14, 23))
    grid = np.full((height, width), AZURE, dtype=int)

    kinds = ["h_only", "v_only", "rot_only", "full"]
    extra_count = int(_RNG.integers(0, 3))
    for _ in range(extra_count):
        kinds.append(str(_RNG.choice(kinds)))
    _RNG.shuffle(kinds)

    if len(kinds) > len(_FG_COLORS):
        return None
    colors = list(_RNG.choice(_FG_COLORS, size=len(kinds), replace=False))

    for kind, color in zip(kinds, colors):
        placed = False
        for _ in range(80):
            mask = _sample_mask(kind)
            mh, mw = mask.shape
            if mh > height - 2 or mw > width - 2:
                continue
            top = int(_RNG.integers(1, height - mh))
            left = int(_RNG.integers(1, width - mw))
            if not _can_place(grid, top, left, mask, AZURE):
                continue
            _place_mask(grid, top, left, mask, int(color))
            placed = True
            break
        if not placed:
            return None

    comps = _components(grid, AZURE)
    if len(comps) != len(kinds):
        return None
    observed = sorted(_symmetry_type(comp["mask"]) for comp in comps)
    if observed != sorted(kinds):
        return None
    try:
        main(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
