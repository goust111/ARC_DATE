import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PAIR_COLORS = [2, 1, 4, 6, 7, 8, 9, 3, 5]
RNG = np.random.default_rng()

MIN_PAIRS = 2
MAX_PAIRS = 4
GRID_H_RANGE = (20, 31)
GRID_W_RANGE = (20, 31)
PAIR_SHAPE_RETRIES = 100
PLACEMENT_RETRIES = 8
GEN_RETRIES = 400


def _background_color(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(values, minlength=10).argmax())


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _coords_to_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[rows - r0, cols - c0] = True
    return mask, (r0, c0, r1, c1)


def _row_signature(mask: np.ndarray) -> dict | None:
    h = mask.shape[0]
    starts: list[int] = []
    ends: list[int] = []
    for r in range(h):
        cols = np.flatnonzero(mask[r])
        if cols.size == 0 or int(cols[-1] - cols[0] + 1) != int(cols.size):
            return None
        starts.append(int(cols[0]))
        ends.append(int(cols[-1]))

    left_const = len(set(starts)) == 1
    right_const = len(set(ends)) == 1
    if left_const == right_const:
        return None

    if left_const:
        contour = tuple(int(max(ends) - x) for x in ends)
        side = "right"
    else:
        contour = tuple(int(x - min(starts)) for x in starts)
        side = "left"
    return {"axis": "v", "side": side, "contour": contour}


def _col_signature(mask: np.ndarray) -> dict | None:
    w = mask.shape[1]
    starts: list[int] = []
    ends: list[int] = []
    for c in range(w):
        rows = np.flatnonzero(mask[:, c])
        if rows.size == 0 or int(rows[-1] - rows[0] + 1) != int(rows.size):
            return None
        starts.append(int(rows[0]))
        ends.append(int(rows[-1]))

    top_const = len(set(starts)) == 1
    bottom_const = len(set(ends)) == 1
    if top_const == bottom_const:
        return None

    if top_const:
        contour = tuple(int(max(ends) - x) for x in ends)
        side = "bottom"
    else:
        contour = tuple(int(x - min(starts)) for x in starts)
        side = "top"
    return {"axis": "h", "side": side, "contour": contour}


def _analyze_piece(mask: np.ndarray) -> dict:
    row_sig = _row_signature(mask)
    col_sig = _col_signature(mask)
    assert (row_sig is None) ^ (col_sig is None), "piece must have exactly one jagged axis"

    info = row_sig if row_sig is not None else col_sig
    contour = tuple(int(x) for x in info["contour"])
    assert contour and max(contour) > 0 and min(contour) == 0, "piece must be non-rectangular"

    depth = int(max(contour))
    comp = tuple(int(depth - x) for x in contour)
    pair_key = (info["axis"], tuple(sorted((contour, comp))))

    return {
        "axis": info["axis"],
        "side": info["side"],
        "contour": contour,
        "depth": depth,
        "pair_key": pair_key,
    }


def _extract_pieces(grid: np.ndarray, bg: int) -> list[dict]:
    comps = _find_components(grid != bg)
    pieces: list[dict] = []
    for coords in comps:
        mask, bbox = _coords_to_mask(coords)
        info = _analyze_piece(mask)
        pieces.append(
            {
                "coords": coords,
                "bbox": bbox,
                "mask": mask,
                **info,
            }
        )
    return pieces


def _pieces_match(a: dict, b: dict) -> bool:
    if a["axis"] != b["axis"]:
        return False
    if len(a["contour"]) != len(b["contour"]):
        return False
    if a["axis"] == "v" and {a["side"], b["side"]} != {"left", "right"}:
        return False
    if a["axis"] == "h" and {a["side"], b["side"]} != {"top", "bottom"}:
        return False

    depth = int(max(a["contour"]))
    if depth != int(max(b["contour"])):
        return False
    return all(int(x + y) == depth for x, y in zip(a["contour"], b["contour"]))


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = 0
    assert np.any(g == bg), "expected black background"
    fg_colors = [int(c) for c in np.unique(g) if int(c) != bg]
    assert len(fg_colors) == 1, "expected a single foreground color"

    pieces = _extract_pieces(g, bg)
    assert pieces and len(pieces) % 2 == 0

    groups: dict[tuple, list[int]] = {}
    for idx, piece in enumerate(pieces):
        groups.setdefault(piece["pair_key"], []).append(idx)

    pairs: list[tuple[int, int]] = []
    for idxs in groups.values():
        assert len(idxs) == 2, "jagged shapes must pair uniquely"
        i, j = idxs
        assert _pieces_match(pieces[i], pieces[j]), "paired shapes must be complementary"
        pairs.append((i, j))

    order = sorted(range(len(pieces)), key=lambda i: (pieces[i]["bbox"][0], pieces[i]["bbox"][1]))
    order_rank = {idx: rank for rank, idx in enumerate(order)}
    pairs.sort(key=lambda ij: min(order_rank[ij[0]], order_rank[ij[1]]))

    palette = [c for c in PAIR_COLORS if c not in {bg, fg_colors[0]}]
    assert len(palette) >= len(pairs), "not enough output colors"

    out = np.full_like(g, bg)
    for color, (i, j) in zip(palette, pairs):
        for piece_idx in (i, j):
            coords = pieces[piece_idx]["coords"]
            out[coords[:, 0], coords[:, 1]] = int(color)
    return out


def _random_seam(length: int, depth: int) -> np.ndarray:
    for _ in range(100):
        segs = int(RNG.integers(3, min(length, 6) + 1))
        cut_count = segs - 1
        if cut_count:
            cuts = sorted(int(x) for x in RNG.choice(np.arange(1, length), size=cut_count, replace=False))
        else:
            cuts = []

        values = [int(RNG.integers(0, depth + 1)) for _ in range(segs)]
        i0, i1 = RNG.choice(segs, size=2, replace=False)
        values[int(i0)] = 0
        values[int(i1)] = depth
        if not any(values[k] != values[k + 1] for k in range(segs - 1)):
            continue

        seam = np.empty(length, dtype=int)
        bounds = [0] + cuts + [length]
        for idx in range(segs):
            seam[bounds[idx] : bounds[idx + 1]] = values[idx]

        if 0 in seam and depth in seam and np.any(seam != seam[0]):
            return seam
    raise RuntimeError("failed to sample a seam")


def _make_vertical_pair() -> tuple[np.ndarray, np.ndarray]:
    for _ in range(PAIR_SHAPE_RETRIES):
        height = int(RNG.integers(5, 10))
        left_base = int(RNG.integers(2, 6))
        right_base = int(RNG.integers(2, 6))
        depth = int(RNG.integers(1, 4))
        seam = _random_seam(height, depth)

        left = np.zeros((height, left_base + depth), dtype=bool)
        right = np.zeros((height, right_base + depth), dtype=bool)
        for r, s in enumerate(seam):
            left[r, : left_base + int(s)] = True
            right[r, int(s) :] = True

        if RNG.random() < 0.5:
            left = np.flipud(left)
            right = np.flipud(right)
        if RNG.random() < 0.5:
            left = np.fliplr(left)
            right = np.fliplr(right)

        left = left.copy()
        right = right.copy()
        try:
            info_left = _analyze_piece(left)
            info_right = _analyze_piece(right)
        except AssertionError:
            continue
        if info_left["axis"] == "v" and info_right["axis"] == "v" and _pieces_match(info_left, info_right):
            return left, right
    raise RuntimeError("failed to build a vertical pair")


def _make_horizontal_pair() -> tuple[np.ndarray, np.ndarray]:
    for _ in range(PAIR_SHAPE_RETRIES):
        width = int(RNG.integers(5, 10))
        top_base = int(RNG.integers(2, 6))
        bottom_base = int(RNG.integers(2, 6))
        depth = int(RNG.integers(1, 4))
        seam = _random_seam(width, depth)

        top = np.zeros((top_base + depth, width), dtype=bool)
        bottom = np.zeros((bottom_base + depth, width), dtype=bool)
        for c, s in enumerate(seam):
            top[: top_base + int(s), c] = True
            bottom[int(s) :, c] = True

        if RNG.random() < 0.5:
            top = np.fliplr(top)
            bottom = np.fliplr(bottom)
        if RNG.random() < 0.5:
            top = np.flipud(top)
            bottom = np.flipud(bottom)

        top = top.copy()
        bottom = bottom.copy()
        try:
            info_top = _analyze_piece(top)
            info_bottom = _analyze_piece(bottom)
        except AssertionError:
            continue
        if info_top["axis"] == "h" and info_bottom["axis"] == "h" and _pieces_match(info_top, info_bottom):
            return top, bottom
    raise RuntimeError("failed to build a horizontal pair")


def _sample_pair() -> tuple[np.ndarray, np.ndarray, tuple]:
    for _ in range(PAIR_SHAPE_RETRIES):
        if RNG.random() < 0.5:
            a, b = _make_vertical_pair()
        else:
            a, b = _make_horizontal_pair()
        try:
            key_a = _analyze_piece(a)["pair_key"]
            key_b = _analyze_piece(b)["pair_key"]
        except AssertionError:
            continue
        if key_a == key_b:
            return a, b, key_a
    raise RuntimeError("failed to sample a unique pair")


def _place_masks(masks: list[np.ndarray], shape: tuple[int, int]) -> list[tuple[int, int]] | None:
    h, w = shape
    blocked = np.zeros((h, w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(masks)
    coords_cache = [np.argwhere(mask) for mask in masks]

    order = sorted(range(len(masks)), key=lambda i: int(coords_cache[i].shape[0]), reverse=True)
    for idx in order:
        mask = masks[idx]
        mh, mw = mask.shape
        positions = [(r, c) for r in range(h - mh + 1) for c in range(w - mw + 1)]
        RNG.shuffle(positions)
        placed = False
        coords = coords_cache[idx]
        for r0, c0 in positions:
            rr = coords[:, 0] + r0
            cc = coords[:, 1] + c0
            if np.any(blocked[rr, cc]):
                continue
            placements[idx] = (r0, c0)
            for r, c in zip(rr, cc):
                r_lo = max(0, int(r) - 1)
                r_hi = min(h, int(r) + 2)
                c_lo = max(0, int(c) - 1)
                c_hi = min(w, int(c) + 2)
                blocked[r_lo:r_hi, c_lo:c_hi] = True
            placed = True
            break
        if not placed:
            return None
    return [(int(r), int(c)) for r, c in placements if r is not None and c is not None]


def _build_grid(masks: list[np.ndarray], color: int) -> np.ndarray | None:
    max_h = max(mask.shape[0] for mask in masks)
    max_w = max(mask.shape[1] for mask in masks)
    total_cells = sum(int(mask.sum()) for mask in masks)
    density_target = float(RNG.uniform(2.0, 2.6))

    min_h = max(GRID_H_RANGE[0], max_h + 2, int(np.ceil(np.sqrt(total_cells * density_target))))
    min_w = max(GRID_W_RANGE[0], max_w + 2, int(np.ceil(np.sqrt(total_cells * density_target))))
    if min_h >= GRID_H_RANGE[1] or min_w >= GRID_W_RANGE[1]:
        return None

    for _ in range(PLACEMENT_RETRIES):
        h = int(RNG.integers(min_h, GRID_H_RANGE[1]))
        w = int(RNG.integers(min_w, GRID_W_RANGE[1]))
        placements = _place_masks(masks, (h, w))
        if placements is None:
            continue

        grid = np.zeros((h, w), dtype=int)
        for mask, (r0, c0) in zip(masks, placements):
            coords = np.argwhere(mask)
            grid[coords[:, 0] + r0, coords[:, 1] + c0] = int(color)
        return grid
    return None


def _gen_attempt() -> np.ndarray | None:
    pair_count = int(RNG.integers(MIN_PAIRS, MAX_PAIRS + 1))
    fg = int(RNG.integers(1, 10))

    masks: list[np.ndarray] = []
    used_keys: set[tuple] = set()
    for _ in range(pair_count):
        for _ in range(PAIR_SHAPE_RETRIES):
            a, b, key = _sample_pair()
            if key in used_keys:
                continue
            used_keys.add(key)
            masks.extend([a, b])
            break
        else:
            return None

    grid = _build_grid(masks, fg)
    if grid is None:
        return None

    try:
        pieces = _extract_pieces(grid, 0)
        if len(pieces) != 2 * pair_count:
            return None
        groups: dict[tuple, list[dict]] = {}
        for piece in pieces:
            groups.setdefault(piece["pair_key"], []).append(piece)
        if len(groups) != pair_count or any(len(group) != 2 for group in groups.values()):
            return None
        out = main(grid)
    except AssertionError:
        return None

    out_colors = [int(c) for c in np.unique(out) if int(c) != 0]
    if len(out_colors) != pair_count:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
