from __future__ import annotations

import numpy as np


BG = 0
WORK_H = 30
WORK_W = 30
GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 250
SIDES = ("top", "bottom", "left", "right")


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> None:
    h, w = map(int, patch.shape)
    if transparent is None:
        dst[r0 : r0 + h, c0 : c0 + w] = patch
        return
    sub = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != int(transparent)
    sub[mask] = patch[mask]


def _components(mask: np.ndarray, *, connectivity: int = 4) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    if connectivity == 4:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]

    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in offsets:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(pts, dtype=int))
    return out


def _normalize_positions(positions: list[int]) -> tuple[int, ...]:
    pos = sorted(int(p) for p in positions)
    start = pos[0]
    return tuple(int(p - start) for p in pos)


def _detect_lock(coords: np.ndarray) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 5 or w < 5:
        return None

    mask = np.zeros((h, w), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True

    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    if np.any(mask & ~border):
        return None
    if not (mask[0, 0] and mask[0, -1] and mask[-1, 0] and mask[-1, -1]):
        return None

    missing = border & ~mask
    if not bool(missing.any()):
        return None

    side_hits: list[tuple[str, list[int]]] = []
    top = np.where(missing[0, 1:-1])[0].tolist()
    bottom = np.where(missing[-1, 1:-1])[0].tolist()
    left = np.where(missing[1:-1, 0])[0].tolist()
    right = np.where(missing[1:-1, -1])[0].tolist()
    if top:
        side_hits.append(("top", top))
    if bottom:
        side_hits.append(("bottom", bottom))
    if left:
        side_hits.append(("left", left))
    if right:
        side_hits.append(("right", right))
    if len(side_hits) != 1:
        return None

    side, positions = side_hits[0]
    side_len = int((w - 2) if side in ("top", "bottom") else (h - 2))
    if len(positions) >= side_len:
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "side": side,
        "signature": _normalize_positions(positions),
    }


def _detect_key(coords: np.ndarray) -> dict | None:
    pts = {(int(r), int(c)) for r, c in coords}
    rows = sorted({int(r) for r, _c in pts})
    cols = sorted({int(c) for _r, c in pts})
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    candidates: list[dict] = []

    for handle_r in rows:
        row_cols = sorted(c for r, c in pts if r == handle_r)
        if row_cols != list(range(min_c, max_c + 1)):
            continue
        if len(row_cols) < 3:
            continue
        other = [(r, c) for r, c in pts if r != handle_r]
        if len(other) < 2:
            continue
        tooth_rows = {r for r, _c in other}
        if len(tooth_rows) != 1:
            continue
        tooth_r = next(iter(tooth_rows))
        if abs(tooth_r - handle_r) != 1:
            continue
        tooth_cols = sorted(c for _r, c in other)
        if any(c < min_c or c > max_c for c in tooth_cols):
            continue
        if len(tooth_cols) != len(set(tooth_cols)):
            continue
        left_end = min_c in tooth_cols
        right_end = max_c in tooth_cols
        if left_end == right_end:
            continue
        if left_end:
            positions = tuple(sorted(int(c - min_c) for c in tooth_cols))
        else:
            positions = tuple(sorted(int(max_c - c) for c in tooth_cols))
        candidates.append({"orientation": "h", "signature": positions})

    for handle_c in cols:
        col_rows = sorted(r for r, c in pts if c == handle_c)
        if col_rows != list(range(min_r, max_r + 1)):
            continue
        if len(col_rows) < 3:
            continue
        other = [(r, c) for r, c in pts if c != handle_c]
        if len(other) < 2:
            continue
        tooth_cols = {c for _r, c in other}
        if len(tooth_cols) != 1:
            continue
        tooth_c = next(iter(tooth_cols))
        if abs(tooth_c - handle_c) != 1:
            continue
        tooth_rows = sorted(r for r, _c in other)
        if any(r < min_r or r > max_r for r in tooth_rows):
            continue
        if len(tooth_rows) != len(set(tooth_rows)):
            continue
        top_end = min_r in tooth_rows
        bottom_end = max_r in tooth_rows
        if top_end == bottom_end:
            continue
        if top_end:
            positions = tuple(sorted(int(r - min_r) for r in tooth_rows))
        else:
            positions = tuple(sorted(int(max_r - r) for r in tooth_rows))
        candidates.append({"orientation": "v", "signature": positions})

    unique = {(cand["orientation"], cand["signature"]) for cand in candidates}
    if len(unique) != 1:
        return None
    orientation, signature = next(iter(unique))
    return {"orientation": orientation, "signature": signature}


def _extract_pattern(grid: np.ndarray, lock: dict, bg: int) -> dict | None:
    r0, c0, r1, c1 = lock["bbox"]
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    colors = [int(v) for v in np.unique(interior) if int(v) != int(bg)]
    if len(colors) != 1:
        return None
    color = int(colors[0])
    mask = interior == color
    if (
        bool(mask[0, :].any())
        or bool(mask[-1, :].any())
        or bool(mask[:, 0].any())
        or bool(mask[:, -1].any())
    ):
        return None

    coords = np.argwhere(mask)
    pr0, pc0, pr1, pc1 = _bbox(coords)
    submask = mask[pr0 : pr1 + 1, pc0 : pc1 + 1]
    patch = np.full(submask.shape, bg, dtype=int)
    patch[submask] = color
    return {"color": color, "patch": patch}


def _inside_any_lock(r: int, c: int, locks: list[dict]) -> bool:
    for lock in locks:
        r0, c0, r1, c1 = lock["bbox"]
        if r0 <= r <= r1 and c0 <= c <= c1:
            return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Release each locked inner pattern to its same-color target marker once the matching key is identified.

    Input structure:
    - The background is black.
    - Several locks appear as colored rectangular borders of thickness 1, each with a row of missing border pixels on exactly one side and a monochrome pattern fully inside the border.
    - Several same-color keys appear elsewhere; each key is a thin horizontal or vertical handle with short unit teeth on one side, and the handle color matches one lock border.
    - Single-pixel target markers use the same color as a lock's inner pattern.

    Rule:
    - Pair each lock with the key that shares its border color.
    - Read the lock's notch spacing pattern from the missing border pixels and the key's tooth spacing pattern from the protruding teeth; only matching signatures count as valid pairs.
    - For every valid pair, crop the inner pattern from the lock and stamp it with its top-left corner anchored at the same-color target marker.
    - Remove locks and keys from the scene; keep the target markers and the released patterns.

    Key insight:
    - Color links keys to locks, but the decisive confirmation comes from matching relative spacing of teeth and notches before moving the hidden inner sprite to its same-color marker.
    """
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    bg = _get_bg(g)

    components_by_color: dict[int, list[np.ndarray]] = {}
    for color in np.unique(g):
        color = int(color)
        if color == int(bg):
            continue
        comps = _components(g == color, connectivity=4)
        if comps:
            components_by_color[color] = comps

    locks: list[dict] = []
    keys_by_color: dict[int, dict] = {}
    border_colors: set[int] = set()

    for color, comps in components_by_color.items():
        lock_candidates = []
        key_candidates = []
        for comp in comps:
            lock = _detect_lock(comp)
            if lock is not None:
                lock_candidates.append(lock)
                continue
            key = _detect_key(comp)
            if key is not None:
                key_candidates.append(key)
        if len(lock_candidates) == 1 and key_candidates:
            lock = lock_candidates[0]
            lock["color"] = int(color)
            locks.append(lock)
            border_colors.add(int(color))
            keys_by_color[int(color)] = key_candidates[0]

    out = np.full((h, w), bg, dtype=int)

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == int(bg) or color in border_colors:
                continue
            if _inside_any_lock(int(r), int(c), locks):
                continue
            out[r, c] = color

    for lock in locks:
        border_color = int(lock["color"])
        key = keys_by_color.get(border_color)
        if key is None or tuple(key["signature"]) != tuple(lock["signature"]):
            continue

        pattern = _extract_pattern(g, lock, bg)
        if pattern is None:
            continue
        pat_color = int(pattern["color"])

        marker_candidates = [
            (int(r), int(c))
            for r, c in np.argwhere(g == pat_color)
            if not _inside_any_lock(int(r), int(c), locks)
        ]
        if len(marker_candidates) != 1:
            continue

        mr, mc = marker_candidates[0]
        _paste(out, pattern["patch"], mr, mc, transparent=bg)

    return out


def _randint(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(int(lo), int(hi) + 1))


def _choice(rng: np.random.Generator, seq):
    return seq[int(rng.integers(0, len(seq)))]


def _grow_mask(rng: np.random.Generator, h: int, w: int, area: int) -> np.ndarray:
    seed = (int(rng.integers(0, h)), int(rng.integers(0, w)))
    cells = {seed}
    while len(cells) < area:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return _grow_mask(rng, h, w, area)
        pick = frontier[int(rng.integers(0, len(frontier)))]
        cells.add(pick)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _make_pattern_patch(rng: np.random.Generator, color: int, bg: int) -> np.ndarray:
    for _ in range(50):
        h = _randint(rng, 2, 4)
        w = _randint(rng, 2, 5)
        max_area = max(2, h * w - 1)
        area = _randint(rng, 2, max_area)
        mask = _grow_mask(rng, h, w, area)
        if mask.shape[0] <= 4 and mask.shape[1] <= 5:
            patch = np.full(mask.shape, bg, dtype=int)
            patch[mask] = color
            return patch
    raise RuntimeError("failed to build pattern patch")


def _make_lock_patch(
    rng: np.random.Generator,
    border_color: int,
    pattern_patch: np.ndarray,
    side: str,
    bg: int,
) -> tuple[np.ndarray, tuple[int, ...]]:
    ph, pw = map(int, pattern_patch.shape)

    tooth_count = _randint(rng, 2, 4)
    feasible_steps = [step for step in (1, 2, 3) if 1 + step * (tooth_count - 1) <= 7]
    step = int(_choice(rng, feasible_steps))
    signature = tuple(int(i * step) for i in range(tooth_count))

    interior_h = ph + 2 + _randint(rng, 0, 2)
    interior_w = pw + 2 + _randint(rng, 0, 2)
    if side in ("top", "bottom"):
        interior_w = max(interior_w, int(signature[-1]) + 2 + _randint(rng, 0, 2))
    else:
        interior_h = max(interior_h, int(signature[-1]) + 2 + _randint(rng, 0, 2))

    outer_h = interior_h + 2
    outer_w = interior_w + 2
    patch = np.full((outer_h, outer_w), bg, dtype=int)
    patch[0, :] = border_color
    patch[-1, :] = border_color
    patch[:, 0] = border_color
    patch[:, -1] = border_color

    side_len = int(interior_w if side in ("top", "bottom") else interior_h)
    offset = _randint(rng, 0, side_len - (int(signature[-1]) + 1))
    abs_positions = [int(offset + p) for p in signature]
    if side == "top":
        for pos in abs_positions:
            patch[0, 1 + pos] = bg
    elif side == "bottom":
        for pos in abs_positions:
            patch[-1, 1 + pos] = bg
    elif side == "left":
        for pos in abs_positions:
            patch[1 + pos, 0] = bg
    else:
        for pos in abs_positions:
            patch[1 + pos, -1] = bg

    inner_r = _randint(rng, 1, interior_h - ph - 1)
    inner_c = _randint(rng, 1, interior_w - pw - 1)
    _paste(patch, pattern_patch, 1 + inner_r, 1 + inner_c, transparent=bg)
    return patch, signature


def _make_key_patch(
    rng: np.random.Generator,
    color: int,
    signature: tuple[int, ...],
    orientation: str,
    bg: int,
) -> np.ndarray:
    handle_len = max(3, int(signature[-1]) + 1 + _randint(rng, 2, 4))
    origin_at_min = bool(rng.integers(0, 2))
    if orientation == "h":
        teeth_on_top = bool(rng.integers(0, 2))
        patch = np.full((2, handle_len), bg, dtype=int)
        handle_r = 1 if teeth_on_top else 0
        tooth_r = 0 if teeth_on_top else 1
        patch[handle_r, :] = color
        cols = [int(p) for p in signature] if origin_at_min else [int(handle_len - 1 - p) for p in signature]
        patch[tooth_r, cols] = color
        return patch

    teeth_on_left = bool(rng.integers(0, 2))
    patch = np.full((handle_len, 2), bg, dtype=int)
    handle_c = 1 if teeth_on_left else 0
    tooth_c = 0 if teeth_on_left else 1
    patch[:, handle_c] = color
    rows = [int(p) for p in signature] if origin_at_min else [int(handle_len - 1 - p) for p in signature]
    patch[rows, tooth_c] = color
    return patch


def _can_place_box(occ: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> bool:
    rr0 = max(0, int(r0 - pad))
    cc0 = max(0, int(c0 - pad))
    rr1 = min(int(occ.shape[0]), int(r0 + h + pad))
    cc1 = min(int(occ.shape[1]), int(c0 + w + pad))
    return not bool(occ[rr0:rr1, cc0:cc1].any())


def _mark_box(occ: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> None:
    rr0 = max(0, int(r0 - pad))
    cc0 = max(0, int(c0 - pad))
    rr1 = min(int(occ.shape[0]), int(r0 + h + pad))
    cc1 = min(int(occ.shape[1]), int(c0 + w + pad))
    occ[rr0:rr1, cc0:cc1] = True


def _find_box_spot(
    rng: np.random.Generator,
    occ: np.ndarray,
    h: int,
    w: int,
    *,
    pad: int,
) -> tuple[int, int] | None:
    max_r = int(occ.shape[0] - h)
    max_c = int(occ.shape[1] - w)
    if max_r < 0 or max_c < 0:
        return None
    for _ in range(PLACE_ATTEMPTS):
        r0 = _randint(rng, 0, max_r)
        c0 = _randint(rng, 0, max_c)
        if _can_place_box(occ, r0, c0, h, w, pad):
            return int(r0), int(c0)
    return None


def _crop_pair(inp: np.ndarray, out: np.ndarray, rng: np.random.Generator, bg: int) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere((inp != bg) | (out != bg))
    r0, c0, r1, c1 = _bbox(coords)
    extra_top = _randint(rng, 0, min(3, r0))
    extra_left = _randint(rng, 0, min(3, c0))
    extra_bottom = _randint(rng, 0, min(3, inp.shape[0] - 1 - r1))
    extra_right = _randint(rng, 0, min(3, inp.shape[1] - 1 - c1))
    rr0 = r0 - extra_top
    cc0 = c0 - extra_left
    rr1 = r1 + extra_bottom
    cc1 = c1 + extra_right
    return inp[rr0 : rr1 + 1, cc0 : cc1 + 1].copy(), out[rr0 : rr1 + 1, cc0 : cc1 + 1].copy()


def _generate_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    pair_count = _choice(rng, [2, 3])
    colors = rng.choice(np.arange(1, 10), size=2 * pair_count, replace=False)
    border_colors = [int(v) for v in colors[:pair_count]]
    pattern_colors = [int(v) for v in colors[pair_count:]]
    side_order = [SIDES[int(i)] for i in rng.permutation(len(SIDES))[:pair_count]]

    specs = []
    for border_color, pattern_color, side in zip(border_colors, pattern_colors, side_order):
        pattern_patch = _make_pattern_patch(rng, pattern_color, BG)
        lock_patch, signature = _make_lock_patch(rng, border_color, pattern_patch, side, BG)
        orientation = "h" if side in ("top", "bottom") else "v"
        key_patch = _make_key_patch(rng, border_color, signature, orientation, BG)
        specs.append(
            {
                "pattern_color": pattern_color,
                "pattern_patch": pattern_patch,
                "lock_patch": lock_patch,
                "key_patch": key_patch,
            }
        )

    specs.sort(key=lambda item: int(item["lock_patch"].size), reverse=True)

    inp = np.full((WORK_H, WORK_W), BG, dtype=int)
    out = np.full((WORK_H, WORK_W), BG, dtype=int)
    input_occ = np.zeros((WORK_H, WORK_W), dtype=bool)
    output_occ = np.zeros((WORK_H, WORK_W), dtype=bool)

    for spec in specs:
        lh, lw = map(int, spec["lock_patch"].shape)
        kh, kw = map(int, spec["key_patch"].shape)

        lock_pos = _find_box_spot(rng, input_occ, lh, lw, pad=1)
        if lock_pos is None:
            return None
        key_pos = _find_box_spot(rng, input_occ, kh, kw, pad=1)
        if key_pos is None:
            return None

        lr, lc = lock_pos
        kr, kc = key_pos
        _paste(inp, spec["lock_patch"], lr, lc, transparent=BG)
        _paste(inp, spec["key_patch"], kr, kc, transparent=BG)
        _mark_box(input_occ, lr, lc, lh, lw, 1)
        _mark_box(input_occ, kr, kc, kh, kw, 1)

    specs.sort(key=lambda item: int(item["pattern_patch"].shape[0] * item["pattern_patch"].shape[1]), reverse=True)
    for spec in specs:
        ph, pw = map(int, spec["pattern_patch"].shape)
        marker_pos = None
        for _ in range(PLACE_ATTEMPTS):
            mr = _randint(rng, 0, WORK_H - ph)
            mc = _randint(rng, 0, WORK_W - pw)
            if not _can_place_box(input_occ, mr, mc, 1, 1, 1):
                continue
            if not _can_place_box(output_occ, mr, mc, ph, pw, 1):
                continue
            marker_pos = (mr, mc)
            break
        if marker_pos is None:
            return None

        mr, mc = marker_pos
        inp[mr, mc] = int(spec["pattern_color"])
        out[mr, mc] = int(spec["pattern_color"])
        _paste(out, spec["pattern_patch"], mr, mc, transparent=BG)
        _mark_box(input_occ, mr, mc, 1, 1, 1)
        _mark_box(output_occ, mr, mc, ph, pw, 1)

    cropped_inp, cropped_out = _crop_pair(inp, out, rng, BG)
    if cropped_inp.ndim != 2 or cropped_out.ndim != 2:
        return None
    if max(cropped_inp.shape) > 30 or max(cropped_out.shape) > 30:
        return None
    if not np.array_equal(main(cropped_inp), cropped_out):
        return None
    return cropped_inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
