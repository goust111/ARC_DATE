import numpy as np

from collections import Counter


BLACK = 0
GREY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
RNG = np.random.default_rng()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _canonical_pattern(pattern: tuple[bool, ...]) -> tuple[bool, ...]:
    rev = pattern[::-1]
    return pattern if pattern <= rev else rev


def _side_inner_positions(
    r0: int, c0: int, r1: int, c1: int, side: str
) -> list[tuple[int, int]]:
    if side == "top":
        return [(r0, c) for c in range(c0 + 1, c1)]
    if side == "bottom":
        return [(r1, c) for c in range(c0 + 1, c1)]
    if side == "left":
        return [(r, c0) for r in range(r0 + 1, r1)]
    return [(r, c1) for r in range(r0 + 1, r1)]


def _parse_lock(grid: np.ndarray, coords: np.ndarray) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or h > 8 or w < 5 or w > 8:
        return None

    cellset = {tuple(map(int, pt)) for pt in coords}
    perimeter = set()
    for c in range(c0, c1 + 1):
        perimeter.add((r0, c))
        perimeter.add((r1, c))
    for r in range(r0, r1 + 1):
        perimeter.add((r, c0))
        perimeter.add((r, c1))
    if any(pt not in perimeter for pt in cellset):
        return None

    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    if not corners.issubset(cellset):
        return None

    candidates = []
    for side in ("top", "bottom", "left", "right"):
        ok = True
        for other in ("top", "bottom", "left", "right"):
            positions = _side_inner_positions(r0, c0, r1, c1, other)
            if other == side:
                continue
            if any(pos not in cellset for pos in positions):
                ok = False
                break
        if not ok:
            continue
        pattern = tuple(pos not in cellset for pos in _side_inner_positions(r0, c0, r1, c1, side))
        if 0 < sum(pattern) < len(pattern):
            candidates.append((side, pattern))
    if len(candidates) != 1:
        return None

    side, pattern = candidates[0]
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    inner_vals = [int(v) for v in inner.ravel() if int(v) not in (BLACK, GREY)]
    if not inner_vals:
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "side": side,
        "pattern": pattern,
        "inner_color": int(Counter(inner_vals).most_common(1)[0][0]),
    }


def _parse_key(grid: np.ndarray, coords: np.ndarray) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    mask = np.zeros((h, w), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    vals = {int(grid[r, c]) for r, c in coords}
    if len(vals) != 1:
        return None
    color = int(next(iter(vals)))

    if h == 2 and w >= 5:
        row_counts = mask.sum(axis=1)
        full_rows = [i for i, cnt in enumerate(row_counts) if int(cnt) == w]
        if len(full_rows) != 1:
            return None
        stem_row = full_rows[0]
        tooth_row = 1 - stem_row
        if mask[tooth_row, 0] or mask[tooth_row, -1]:
            return None
        pattern = tuple(bool(x) for x in mask[tooth_row, 1:-1])
        if not (0 < sum(pattern) < len(pattern)):
            return None
        return {
            "coords": [(int(r), int(c)) for r, c in coords],
            "color": color,
            "pattern": pattern,
        }

    if w == 2 and h >= 5:
        col_counts = mask.sum(axis=0)
        full_cols = [i for i, cnt in enumerate(col_counts) if int(cnt) == h]
        if len(full_cols) != 1:
            return None
        stem_col = full_cols[0]
        tooth_col = 1 - stem_col
        if mask[0, tooth_col] or mask[-1, tooth_col]:
            return None
        pattern = tuple(bool(x) for x in mask[1:-1, tooth_col])
        if not (0 < sum(pattern) < len(pattern)):
            return None
        return {
            "coords": [(int(r), int(c)) for r, c in coords],
            "color": color,
            "pattern": pattern,
        }

    return None


def _match_keys_to_locks(keys: list[dict], locks: list[dict]) -> list[tuple[int, bool]] | None:
    options: list[list[tuple[int, bool]]] = []
    for key in keys:
        cand = []
        for li, lock in enumerate(locks):
            if len(key["pattern"]) != len(lock["pattern"]):
                continue
            if key["pattern"] == lock["pattern"]:
                cand.append((li, False))
            elif key["pattern"] == lock["pattern"][::-1]:
                cand.append((li, True))
        if not cand:
            return None
        options.append(cand)

    order = sorted(range(len(keys)), key=lambda i: len(options[i]))
    assignment: list[tuple[int, bool] | None] = [None] * len(keys)
    used: set[int] = set()

    def dfs(idx: int) -> bool:
        if idx == len(order):
            return True
        ki = order[idx]
        for li, mirrored in options[ki]:
            if li in used:
                continue
            used.add(li)
            assignment[ki] = (li, mirrored)
            if dfs(idx + 1):
                return True
            assignment[ki] = None
            used.remove(li)
        return False

    if not dfs(0):
        return None
    return [(int(li), bool(mirrored)) for li, mirrored in assignment]


def _analyze_scene(grid: np.ndarray) -> dict | None:
    g = np.asarray(grid, dtype=int)
    locks = []
    for comp in _components(g == GREY):
        parsed = _parse_lock(g, comp)
        if parsed is not None:
            locks.append(parsed)
    if not locks:
        return None

    owner = np.full(g.shape, -1, dtype=int)
    for i, lock in enumerate(locks):
        r0, c0, r1, c1 = lock["bbox"]
        if r1 - r0 < 2 or c1 - c0 < 2:
            return None
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if owner[r, c] != -1:
                    return None
                owner[r, c] = i

    keys = []
    for comp in _components((g != BLACK) & (g != GREY)):
        owners = {int(owner[r, c]) for r, c in comp}
        if len(owners) == 1 and next(iter(owners)) >= 0:
            continue
        if owners != {-1}:
            return None
        parsed = _parse_key(g, comp)
        if parsed is None:
            return None
        keys.append(parsed)

    if len(keys) != len(locks):
        return None

    matches = _match_keys_to_locks(keys, locks)
    if matches is None:
        return None

    return {"locks": locks, "keys": keys, "matches": matches}


def _draw_inserted_key(out: np.ndarray, lock: dict, color: int) -> None:
    r0, c0, r1, c1 = lock["bbox"]
    pattern = lock["pattern"]
    side = lock["side"]
    if side == "top":
        out[r0 - 1, c0 : c1 + 1] = color
        for i, has_tooth in enumerate(pattern, start=1):
            if has_tooth:
                out[r0, c0 + i] = color
        return
    if side == "bottom":
        out[r1 + 1, c0 : c1 + 1] = color
        for i, has_tooth in enumerate(pattern, start=1):
            if has_tooth:
                out[r1, c0 + i] = color
        return
    if side == "left":
        out[r0 : r1 + 1, c0 - 1] = color
        for i, has_tooth in enumerate(pattern, start=1):
            if has_tooth:
                out[r0 + i, c0] = color
        return
    out[r0 : r1 + 1, c1 + 1] = color
    for i, has_tooth in enumerate(pattern, start=1):
        if has_tooth:
            out[r0 + i, c1] = color


def _flip_lock_interior(out: np.ndarray, original: np.ndarray, lock: dict, key_color: int) -> None:
    r0, c0, r1, c1 = lock["bbox"]
    inner = original[r0 + 1 : r1, c0 + 1 : c1]
    fg = (inner != BLACK) & (inner != GREY)
    if not fg.any():
        return

    base_color = lock["inner_color"]
    new_inner = inner.copy()
    new_inner[fg] = key_color

    adj_black = np.zeros_like(fg, dtype=bool)
    for rr, cc in np.argwhere(fg):
        for dr, dc in DIR4:
            nr, nc = int(rr + dr), int(cc + dc)
            if 0 <= nr < inner.shape[0] and 0 <= nc < inner.shape[1] and inner[nr, nc] == BLACK:
                adj_black[nr, nc] = True
    new_inner[adj_black] = base_color
    out[r0 + 1 : r1, c0 + 1 : c1] = new_inner


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match colored comb-keys to grey framed locks by complementary tooth positions,
    insert each key into its lock, erase the source key, and flip the lock's inner pattern.
    """
    g = np.asarray(grid, dtype=int)
    scene = _analyze_scene(g)
    assert scene is not None

    out = g.copy()
    for key in scene["keys"]:
        for r, c in key["coords"]:
            out[r, c] = BLACK

    for key, (lock_idx, _mirrored) in zip(scene["keys"], scene["matches"]):
        lock = scene["locks"][lock_idx]
        _draw_inserted_key(out, lock, key["color"])
        _flip_lock_interior(out, g, lock, key["color"])

    return out


def _random_connected_mask(h: int, w: int, area: int) -> np.ndarray:
    area = max(1, min(area, h * w))
    mask = np.zeros((h, w), dtype=bool)
    start = (int(RNG.integers(0, h)), int(RNG.integers(0, w)))
    mask[start] = True
    cells = [start]
    frontier = {start}
    while len(cells) < area:
        fr, fc = list(frontier)[int(RNG.integers(0, len(frontier)))]
        neighbors = []
        for dr, dc in DIR4:
            nr, nc = fr + dr, fc + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                neighbors.append((nr, nc))
        if not neighbors:
            frontier.remove((fr, fc))
            if not frontier:
                frontier = set(cells)
            continue
        nr, nc = neighbors[int(RNG.integers(0, len(neighbors)))]
        mask[nr, nc] = True
        cells.append((nr, nc))
        frontier.add((nr, nc))
    return mask


def _build_key_mask(
    pattern: tuple[bool, ...], orientation: str, tooth_side: str
) -> np.ndarray:
    span = len(pattern) + 2
    if orientation == "horizontal":
        mask = np.zeros((2, span), dtype=bool)
        if tooth_side == "top":
            stem_row, tooth_row = 1, 0
        else:
            stem_row, tooth_row = 0, 1
        mask[stem_row, :] = True
        for i, bit in enumerate(pattern, start=1):
            if bit:
                mask[tooth_row, i] = True
        return mask

    mask = np.zeros((span, 2), dtype=bool)
    if tooth_side == "left":
        stem_col, tooth_col = 1, 0
    else:
        stem_col, tooth_col = 0, 1
    mask[:, stem_col] = True
    for i, bit in enumerate(pattern, start=1):
        if bit:
            mask[i, tooth_col] = True
    return mask


def _clear_with_pad(occ: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occ.shape[0] - 1, r1 + pad)
    cc1 = min(occ.shape[1] - 1, c1 + pad)
    return not occ[rr0 : rr1 + 1, cc0 : cc1 + 1].any()


def _mark_occ(occ: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occ.shape[0] - 1, r1 + pad)
    cc1 = min(occ.shape[1] - 1, c1 + pad)
    occ[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _sample_lock_specs(count: int) -> list[dict] | None:
    specs = []
    seen = set()
    for _ in range(count):
        found = None
        for _attempt in range(200):
            side = str(RNG.choice(["top", "bottom", "left", "right"]))
            h = int(RNG.integers(5, 9))
            w = int(RNG.integers(5, 9))
            plen = (w - 2) if side in ("top", "bottom") else (h - 2)
            bits = tuple(bool(x) for x in RNG.integers(0, 2, size=plen))
            if not (0 < sum(bits) < plen):
                continue
            if bits == bits[::-1]:
                continue
            canon = (plen, _canonical_pattern(bits))
            if canon in seen:
                continue
            found = {"side": side, "h": h, "w": w, "pattern": bits}
            seen.add(canon)
            break
        if found is None:
            return None
        specs.append(found)
    return specs


def _draw_lock(
    grid: np.ndarray,
    r0: int,
    c0: int,
    spec: dict,
    inner_color: int,
    inner_mask: np.ndarray,
) -> None:
    h = spec["h"]
    w = spec["w"]
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY

    for i, has_tooth in enumerate(spec["pattern"], start=1):
        if not has_tooth:
            continue
        if spec["side"] == "top":
            grid[r0, c0 + i] = BLACK
        elif spec["side"] == "bottom":
            grid[r1, c0 + i] = BLACK
        elif spec["side"] == "left":
            grid[r0 + i, c0] = BLACK
        else:
            grid[r0 + i, c1] = BLACK

    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    inner[inner_mask] = inner_color


def _draw_key(grid: np.ndarray, r0: int, c0: int, mask: np.ndarray, color: int) -> None:
    h, w = mask.shape
    patch = grid[r0 : r0 + h, c0 : c0 + w]
    patch[mask] = color


def _gen_attempt() -> np.ndarray | None:
    pair_count = int(RNG.integers(1, 4))
    h = int(RNG.integers(20, 31))
    w = int(RNG.integers(20, 31))
    grid = np.full((h, w), BLACK, dtype=int)
    occ = np.zeros((h, w), dtype=bool)

    specs = _sample_lock_specs(pair_count)
    if specs is None:
        return None

    available = COLOR_POOL.copy()
    RNG.shuffle(available)
    if len(available) < 2 * pair_count:
        return None

    lock_colors = available[:pair_count]
    key_colors = available[pair_count : 2 * pair_count]

    lock_records = []
    for i, spec in enumerate(specs):
        lh, lw = spec["h"], spec["w"]
        placed = None
        for _attempt in range(200):
            r0 = int(RNG.integers(1, h - lh))
            c0 = int(RNG.integers(1, w - lw))
            r1 = r0 + lh - 1
            c1 = c0 + lw - 1
            if not _clear_with_pad(occ, r0, c0, r1, c1, pad=2):
                continue
            placed = (r0, c0)
            break
        if placed is None:
            return None

        inner_h = lh - 2
        inner_w = lw - 2
        max_area = inner_h * inner_w - 1
        if max_area < 1:
            return None
        area = int(RNG.integers(1, min(max_area, 5) + 1))
        inner_mask = _random_connected_mask(inner_h, inner_w, area)

        r0, c0 = placed
        _draw_lock(grid, r0, c0, spec, int(lock_colors[i]), inner_mask)
        _mark_occ(occ, r0, c0, r0 + lh - 1, c0 + lw - 1, pad=2)
        lock_records.append(spec)

    for i, spec in enumerate(lock_records):
        pattern = spec["pattern"]
        mirrored = bool(RNG.integers(0, 2))
        shown = pattern[::-1] if mirrored else pattern
        orientation = str(RNG.choice(["horizontal", "vertical"]))
        if orientation == "horizontal":
            tooth_side = str(RNG.choice(["top", "bottom"]))
        else:
            tooth_side = str(RNG.choice(["left", "right"]))
        mask = _build_key_mask(shown, orientation, tooth_side)
        kh, kw = map(int, mask.shape)
        placed = None
        for _attempt in range(300):
            r0 = int(RNG.integers(0, h - kh + 1))
            c0 = int(RNG.integers(0, w - kw + 1))
            r1 = r0 + kh - 1
            c1 = c0 + kw - 1
            if not _clear_with_pad(occ, r0, c0, r1, c1, pad=1):
                continue
            placed = (r0, c0)
            break
        if placed is None:
            return None
        _draw_key(grid, placed[0], placed[1], mask, int(key_colors[i]))
        _mark_occ(occ, placed[0], placed[1], placed[0] + kh - 1, placed[1] + kw - 1, pad=1)

    scene = _analyze_scene(grid)
    if scene is None:
        return None
    if len(scene["locks"]) != pair_count or len(scene["keys"]) != pair_count:
        return None
    _ = main(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
