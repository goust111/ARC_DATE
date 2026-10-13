import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components_for_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _frame_from_component(coords: np.ndarray) -> dict | None:
    rs = coords[:, 0]
    cs = coords[:, 1]
    r0, c0 = int(rs.min()), int(cs.min())
    r1, c1 = int(rs.max()), int(cs.max())
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or w < 5:
        return None

    coord_set = {tuple(map(int, pt)) for pt in coords}
    perimeter = set()
    for c in range(c0, c1 + 1):
        perimeter.add((r0, c))
        perimeter.add((r1, c))
    for r in range(r0 + 1, r1):
        perimeter.add((r, c0))
        perimeter.add((r, c1))

    if not coord_set <= perimeter:
        return None

    missing = list(perimeter - coord_set)
    if len(missing) != 1 or len(coord_set) != len(perimeter) - 1:
        return None

    gap_r, gap_c = missing[0]
    if (gap_r in (r0, r1)) and (gap_c in (c0, c1)):
        return None

    if gap_r == r0:
        side = "top"
    elif gap_r == r1:
        side = "bottom"
    elif gap_c == c0:
        side = "left"
    else:
        side = "right"

    return {
        "bbox": (r0, c0, r1, c1),
        "gap": (int(gap_r), int(gap_c)),
        "side": side,
    }


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    assert np.any(rows) and np.any(cols)
    return mask[np.ix_(rows, cols)]


def _seed_info(grid: np.ndarray, frame: dict, bg: int, border_color: int) -> dict:
    r0, c0, r1, c1 = frame["bbox"]
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    inner_mask = (interior != bg) & (interior != border_color)
    assert np.any(inner_mask), "each frame must contain a seed"

    colors = set(map(int, np.unique(interior[inner_mask])))
    assert len(colors) == 1, "seed must use exactly one color inside each frame"
    seed_color = colors.pop()

    seed_coords = np.argwhere(interior == seed_color)
    sr0, sc0 = seed_coords.min(axis=0)
    sr1, sc1 = seed_coords.max(axis=0)
    local = interior[sr0 : sr1 + 1, sc0 : sc1 + 1] == seed_color

    h_sym = np.array_equal(local, local[::-1, :])
    v_sym = np.array_equal(local, local[:, ::-1])
    assert h_sym or v_sym, "seed must have at least one reflection symmetry"

    if h_sym and v_sym:
        kind = "hv"
    elif h_sym:
        kind = "h"
    else:
        kind = "v"

    top = r0 + 1 + int(sr0)
    left = c0 + 1 + int(sc0)
    return {
        "color": int(seed_color),
        "mask": local,
        "kind": kind,
        "top": int(top),
        "left": int(left),
    }


def _aligned_positions(base: int, step: int, low: int, high: int) -> list[int]:
    if low > high:
        return []
    start = base
    while start < low:
        start += step
    while start - step >= low:
        start -= step
    out = []
    cur = start
    while cur <= high:
        out.append(int(cur))
        cur += step
    return out


def _paint_seed(out: np.ndarray, top: int, left: int, mask: np.ndarray, color: int, bg: int) -> None:
    patch = out[top : top + mask.shape[0], left : left + mask.shape[1]]
    fill = mask & (patch == bg)
    patch[fill] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each interior seed pattern inside a gapped rectangular frame.

    Input structure:
    - Background is the dominant color.
    - One or more 1-pixel-thick monochrome rectangular borders each have exactly
      one missing border cell on one side.
    - Inside each border sits a smaller monochrome seed pattern with horizontal
      symmetry, vertical symmetry, or both.

    Rule:
    - Horizontal-only seeds repeat left/right.
    - Vertical-only seeds repeat up/down.
    - Seeds with both symmetries repeat in both directions.
    - The frame gap fixes one tiling phase: top/bottom gaps fix the seed center
      column, left/right gaps fix the seed center row. The other phase comes
      from the observed seed position.
    - Only background cells inside the frame are painted; the border, the gap,
      and everything outside the frame stay unchanged.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()

    frames: list[tuple[int, dict, int]] = []
    for color in sorted(set(map(int, np.unique(g))) - {bg}):
        for comp in _components_for_color(g, color):
            frame = _frame_from_component(comp)
            if frame is not None:
                frames.append((frame["bbox"][0], frame, int(color)))

    frames.sort(key=lambda item: item[0])
    assert frames, "expected at least one gapped frame"

    for _, frame, border_color in frames:
        seed = _seed_info(g, frame, bg, border_color)
        mask = seed["mask"]
        sh, sw = mask.shape
        cr, cc = sh // 2, sw // 2
        seed_center_r = seed["top"] + cr
        seed_center_c = seed["left"] + cc
        gap_r, gap_c = frame["gap"]
        r0, c0, r1, c1 = frame["bbox"]

        anchor_row = seed_center_r
        anchor_col = seed_center_c
        if frame["side"] in ("top", "bottom"):
            anchor_col = gap_c
        else:
            anchor_row = gap_r

        base_top = anchor_row - cr
        base_left = anchor_col - cc
        row_low, row_high = r0 + 1, r1 - sh
        col_low, col_high = c0 + 1, c1 - sw

        if seed["kind"] == "h":
            row_positions = [int(base_top)]
            col_positions = _aligned_positions(base_left, sw, col_low, col_high)
        elif seed["kind"] == "v":
            row_positions = _aligned_positions(base_top, sh, row_low, row_high)
            col_positions = [int(base_left)]
        else:
            row_positions = _aligned_positions(base_top, sh, row_low, row_high)
            col_positions = _aligned_positions(base_left, sw, col_low, col_high)

        for top in row_positions:
            for left in col_positions:
                if not (row_low <= top <= row_high and col_low <= left <= col_high):
                    continue
                _paint_seed(out, top, left, mask, seed["color"], bg)

    return out


GEN_ATTEMPTS = 300
OBJECT_BUILD_ATTEMPTS = 200
PLACEMENT_ATTEMPTS = 250
MASK_ATTEMPTS = 200
GRID_MIN = 24
GRID_MAX = 40
OBJ_COUNT_RANGE = (2, 4)
NOISE_MAX = 6


def _mask_components(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            count += 1
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                cr, cc = stack.pop()
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
    return count


def _orbits(kind: str, h: int, w: int) -> list[list[tuple[int, int]]]:
    orbits: list[list[tuple[int, int]]] = []
    seen: set[tuple[int, int]] = set()
    for r in range(h):
        for c in range(w):
            if (r, c) in seen:
                continue
            pts = {(r, c)}
            if kind in ("h", "hv"):
                pts.add((h - 1 - r, c))
            if kind in ("v", "hv"):
                pts.add((r, w - 1 - c))
            if kind == "hv":
                pts.add((h - 1 - r, w - 1 - c))
            orbit = sorted(pts)
            for pt in orbit:
                seen.add(pt)
            orbits.append(orbit)
    return orbits


def _orbit_adjacent(orbit: list[tuple[int, int]], mask: np.ndarray) -> bool:
    if not np.any(mask):
        return True
    for r, c in orbit:
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and bool(mask[nr, nc]):
                return True
    return False


def _make_seed_mask(kind: str, rng: np.random.Generator) -> np.ndarray:
    dims = [(3, 3), (3, 5), (5, 3), (5, 5)]
    if kind == "hv":
        dims = [(3, 3), (3, 5), (5, 3), (5, 5), (5, 5)]
    for _ in range(MASK_ATTEMPTS):
        h, w = dims[int(rng.integers(len(dims)))]
        orbit_list = _orbits(kind, h, w)
        center = (h // 2, w // 2)
        mask = np.zeros((h, w), dtype=bool)

        center_idx = next(i for i, orb in enumerate(orbit_list) if center in orb)
        chosen = {center_idx}
        for r, c in orbit_list[center_idx]:
            mask[r, c] = True

        target = int(rng.integers(3, len(orbit_list) + 1))
        while len(chosen) < target:
            frontier = [i for i, orb in enumerate(orbit_list) if i not in chosen and _orbit_adjacent(orb, mask)]
            if not frontier:
                break
            idx = int(frontier[int(rng.integers(len(frontier)))])
            chosen.add(idx)
            for r, c in orbit_list[idx]:
                mask[r, c] = True

        mask = _trim_mask(mask)
        if min(mask.shape) < 3:
            continue
        if _mask_components(mask) != 1:
            continue
        h_sym = np.array_equal(mask, mask[::-1, :])
        v_sym = np.array_equal(mask, mask[:, ::-1])
        if kind == "h" and h_sym and not v_sym:
            return mask
        if kind == "v" and v_sym and not h_sym:
            return mask
        if kind == "hv" and h_sym and v_sym and int(mask.sum()) < mask.size:
            return mask
    raise RuntimeError(f"failed to make {kind} seed")


def _odd_choices(lo: int, hi: int) -> list[int]:
    start = lo if lo % 2 == 1 else lo + 1
    return list(range(start, hi + 1, 2))


def _positions_for_center(center: int, size: int, outer_len: int) -> list[int]:
    low = 1
    high = outer_len - size - 1
    return _aligned_positions(center - size // 2, size, low, high)


def _valid_centers(size: int, outer_len: int, need: int) -> list[int]:
    low = 1 + size // 2
    high = outer_len - 2 - size // 2
    return [center for center in range(low, high + 1) if len(_positions_for_center(center, size, outer_len)) >= need]


def _draw_frame(patch: np.ndarray, color: int, side: str) -> None:
    h, w = patch.shape
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    if side == "top":
        patch[0, w // 2] = 0
    elif side == "bottom":
        patch[-1, w // 2] = 0
    elif side == "left":
        patch[h // 2, 0] = 0
    else:
        patch[h // 2, -1] = 0


def _paint_local_seed(patch: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    sub = patch[top : top + mask.shape[0], left : left + mask.shape[1]]
    sub[mask] = int(color)


def _build_object(kind: str, border_color: int, seed_color: int, side: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(OBJECT_BUILD_ATTEMPTS):
        mask = _make_seed_mask(kind, rng)
        sh, sw = mask.shape

        min_ih = sh
        min_iw = sw
        if kind in ("v", "hv"):
            min_ih = max(min_ih, 2 * sh)
        if kind in ("h", "hv"):
            min_iw = max(min_iw, 2 * sw)
        if side in ("top", "bottom") and kind in ("h", "hv"):
            min_iw = max(min_iw, 3 * sw)
        if side in ("left", "right") and kind in ("v", "hv"):
            min_ih = max(min_ih, 3 * sh)

        ih_choices = _odd_choices(min_ih, min_ih + max(4, 2 * sh))
        iw_choices = _odd_choices(min_iw, min_iw + max(4, 2 * sw))
        if not ih_choices or not iw_choices:
            continue

        ih = int(ih_choices[int(rng.integers(len(ih_choices)))])
        iw = int(iw_choices[int(rng.integers(len(iw_choices)))])
        outer_h, outer_w = ih + 2, iw + 2

        need_rows = 2 if kind in ("v", "hv") else 1
        need_cols = 2 if kind in ("h", "hv") else 1

        if side in ("left", "right"):
            center_rows = [outer_h // 2]
            if len(_positions_for_center(center_rows[0], sh, outer_h)) < need_rows:
                continue
        else:
            center_rows = _valid_centers(sh, outer_h, need_rows)
            if not center_rows:
                continue

        if side in ("top", "bottom"):
            center_cols = [outer_w // 2]
            if len(_positions_for_center(center_cols[0], sw, outer_w)) < need_cols:
                continue
        else:
            center_cols = _valid_centers(sw, outer_w, need_cols)
            if not center_cols:
                continue

        center_r = int(center_rows[int(rng.integers(len(center_rows)))])
        center_c = int(center_cols[int(rng.integers(len(center_cols)))])
        row_positions = _positions_for_center(center_r, sh, outer_h)
        col_positions = _positions_for_center(center_c, sw, outer_w)

        if kind == "h":
            row_positions = [center_r - sh // 2]
        elif kind == "v":
            col_positions = [center_c - sw // 2]

        if len(row_positions) < need_rows or len(col_positions) < need_cols:
            continue

        tiles = [(int(r), int(c)) for r in row_positions for c in col_positions]
        seed_top, seed_left = tiles[int(rng.integers(len(tiles)))]

        inp = np.zeros((outer_h, outer_w), dtype=int)
        out = np.zeros((outer_h, outer_w), dtype=int)
        _draw_frame(inp, border_color, side)
        _draw_frame(out, border_color, side)
        _paint_local_seed(inp, seed_top, seed_left, mask, seed_color)
        for top, left in tiles:
            _paint_local_seed(out, top, left, mask, seed_color)
        if np.array_equal(inp, out):
            continue
        return inp, out
    raise RuntimeError("failed to build object")


def _choose_object_types(n: int, rng: np.random.Generator) -> list[str]:
    base = ["h", "v", "hv"]
    perm = [base[i] for i in rng.permutation(len(base))]
    out = []
    while len(out) < n:
        out.extend(perm)
        perm = [base[i] for i in rng.permutation(len(base))]
    return out[:n]


def _gen_attempt(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    n_obj = int(rng.integers(OBJ_COUNT_RANGE[0], OBJ_COUNT_RANGE[1] + 1))
    kinds = _choose_object_types(n_obj, rng)
    sides = [side for side in ("top", "bottom", "left", "right")]
    side_order = [sides[i] for i in rng.permutation(len(sides))]
    while len(side_order) < n_obj:
        side_order.extend([sides[i] for i in rng.permutation(len(sides))])

    palette = list(rng.permutation(np.arange(1, 10)))
    if 2 * n_obj > len(palette):
        return None

    objects: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_obj):
        border_color = int(palette[2 * i])
        seed_color = int(palette[2 * i + 1])
        objects.append(_build_object(kinds[i], border_color, seed_color, side_order[i], rng))

    max_h = max(inp.shape[0] for inp, _ in objects)
    max_w = max(inp.shape[1] for inp, _ in objects)
    total_area = sum(inp.shape[0] * inp.shape[1] for inp, _ in objects)
    min_h = max(GRID_MIN, max_h + 4, int(np.sqrt(total_area * 2.4)))
    min_w = max(GRID_MIN, max_w + 4, int(np.sqrt(total_area * 2.4)))
    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None

    grid_h = int(rng.integers(min_h, GRID_MAX + 1))
    grid_w = int(rng.integers(min_w, GRID_MAX + 1))
    inp_grid = np.zeros((grid_h, grid_w), dtype=int)
    out_grid = np.zeros((grid_h, grid_w), dtype=int)
    blocked = np.zeros((grid_h, grid_w), dtype=bool)

    placements: list[tuple[int, int, int, int]] = []
    order = [int(i) for i in rng.permutation(len(objects))]
    for obj_idx in order:
        inp_obj, out_obj = objects[obj_idx]
        oh, ow = inp_obj.shape
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = int(rng.integers(1, grid_h - oh))
            c0 = int(rng.integers(1, grid_w - ow))
            rr0 = max(0, r0 - 2)
            cc0 = max(0, c0 - 2)
            rr1 = min(grid_h, r0 + oh + 2)
            cc1 = min(grid_w, c0 + ow + 2)
            if np.any(blocked[rr0:rr1, cc0:cc1]):
                continue

            inp_grid[r0 : r0 + oh, c0 : c0 + ow] = inp_obj
            out_grid[r0 : r0 + oh, c0 : c0 + ow] = out_obj
            blocked[rr0:rr1, cc0:cc1] = True
            placements.append((r0, c0, oh, ow))
            placed = True
            break
        if not placed:
            return None

    noise_colors = list(range(1, 10))
    free = np.argwhere(~blocked & (inp_grid == 0))
    if len(free):
        rng.shuffle(free, axis=0)
        n_noise = int(rng.integers(0, min(NOISE_MAX, len(free)) + 1))
        for cell in free[:n_noise]:
            r, c = map(int, cell)
            color = int(noise_colors[int(rng.integers(len(noise_colors)))])
            inp_grid[r, c] = color
            out_grid[r, c] = color

    if not np.array_equal(main(inp_grid), out_grid):
        return None
    return inp_grid, out_grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        result = _gen_attempt(rng)
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate a valid seed-tiling puzzle")
