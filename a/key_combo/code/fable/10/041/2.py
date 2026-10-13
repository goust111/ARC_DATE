from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors(r: int, c: int, connectivity: int) -> list[tuple[int, int]]:
    if connectivity == 4:
        return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((r + dr, c + dc))
    return out


def _components_from_mask(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
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
                for nr, nc in _neighbors(rr, cc, connectivity):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _normalize(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = _bbox(coords)
    shifted = coords - np.array([r0, c0], dtype=int)
    return tuple(sorted((int(r), int(c)) for r, c in shifted))


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    sig = _normalize(coords)
    h = max(r for r, _ in sig) + 1
    w = max(c for _, c in sig) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in sig:
        mask[r, c] = True
    return mask


def _d4_signatures(coords: np.ndarray) -> set[tuple[tuple[int, int], ...]]:
    mask = _coords_to_mask(coords)
    variants = set()
    cur = mask
    for _ in range(4):
        for candidate in (cur, np.fliplr(cur)):
            pts = np.argwhere(candidate)
            variants.add(_normalize(pts))
        cur = np.rot90(cur)
    return variants


def _paint_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, color: int) -> None:
    for r, c in patch:
        grid[r0 + int(r), c0 + int(c)] = int(color)


def _mirrored_positions(r: int, c: int, a2: int, b2: int) -> list[tuple[int, int]]:
    return [
        (r, a2 - c),
        (b2 - r, c),
        (b2 - r, a2 - c),
    ]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the missing region of a biaxially symmetric mosaic, then use that hole shape to keep and recolor only matching outside objects.

    Input structure:
    - The background is black.
    - One side of the grid contains a large multicolor rectangular mosaic that is mirror-symmetric across both a vertical and a horizontal axis.
    - Inside that mosaic, exactly one 8-connected black hole lies strictly away from the mosaic border.
    - Outside the mosaic, several separated monochrome connected objects sit in the empty background.

    Rule:
    - Identify the large mosaic as the largest 8-connected non-background component and infer its two reflection axes from its bounding box center.
    - Within the mosaic bounding box, locate the unique enclosed black component; for each hole cell, restore its color from the non-black mirror counterparts across the vertical axis, horizontal axis, or both, using majority color if several candidates are visible.
    - Normalize the full hole shape by translation and compare every outside object against it up to rotation and reflection.
    - Erase every non-matching outside object, keep every matching one, and recolor all kept objects to the color written back most often while repairing the hole.

    Key insight:
    - The missing pixels play two roles at once: their mirrored counterparts reveal the repair colors, and their combined silhouette is the template that decides which scattered objects survive.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    assert bg == BG

    fg_components = _components_from_mask(g != bg, connectivity=8)
    assert fg_components
    mosaic = max(fg_components, key=len)
    mosaic_set = {tuple(map(int, rc)) for rc in mosaic}
    r0, c0, r1, c1 = _bbox(mosaic)
    a2 = c0 + c1
    b2 = r0 + r1

    sub = g[r0 : r1 + 1, c0 : c1 + 1]
    zero_components = _components_from_mask(sub == bg, connectivity=8)
    hole_local = [
        comp
        for comp in zero_components
        if np.all(comp[:, 0] > 0)
        and np.all(comp[:, 0] < sub.shape[0] - 1)
        and np.all(comp[:, 1] > 0)
        and np.all(comp[:, 1] < sub.shape[1] - 1)
    ]
    assert len(hole_local) == 1
    hole = hole_local[0] + np.array([r0, c0], dtype=int)
    hole_set = {tuple(map(int, rc)) for rc in hole}

    out = g.copy()
    repair_counter: Counter[int] = Counter()
    pending = {tuple(map(int, rc)) for rc in hole}
    while pending:
        progressed = False
        for r, c in list(pending):
            colors = []
            for rr, cc in _mirrored_positions(r, c, a2, b2):
                if r0 <= rr <= r1 and c0 <= cc <= c1 and (rr, cc) not in hole_set:
                    val = int(out[rr, cc])
                    if val != bg:
                        colors.append(val)
            if not colors:
                continue
            chosen = sorted(Counter(colors).items(), key=lambda item: (-item[1], item[0]))[0][0]
            out[r, c] = chosen
            repair_counter[chosen] += 1
            pending.remove((r, c))
            progressed = True
        assert progressed, "hole could not be fully repaired from mirrored evidence"
    assert repair_counter
    winner = sorted(repair_counter.items(), key=lambda item: (-item[1], item[0]))[0][0]

    hole_variants = _d4_signatures(hole)
    for comp in fg_components:
        comp_set = {tuple(map(int, rc)) for rc in comp}
        if comp_set == mosaic_set:
            continue
        if _normalize(comp) in hole_variants:
            out[comp[:, 0], comp[:, 1]] = winner
        else:
            out[comp[:, 0], comp[:, 1]] = bg

    return out


def _paint_orbit_rect(
    patch: np.ndarray,
    rr: int,
    cc: int,
    rh: int,
    rw: int,
    color: int,
) -> None:
    h, w = patch.shape
    a2 = w - 1
    b2 = h - 1
    for r in range(rr, rr + rh):
        for c in range(cc, cc + rw):
            orbit = {
                (r, c),
                (r, a2 - c),
                (b2 - r, c),
                (b2 - r, a2 - c),
            }
            for pr, pc in orbit:
                if 0 <= pr < h and 0 <= pc < w:
                    patch[pr, pc] = int(color)


def _make_symmetric_patch(h: int, w: int, rng: random.Random) -> np.ndarray:
    colors = rng.sample(PALETTE, rng.randint(3, 5))
    base = colors[0]
    accents = colors[1:]
    patch = np.full((h, w), base, dtype=int)

    rep_h = (h + 1) // 2
    rep_w = (w + 1) // 2
    for _ in range(rng.randint(5, 9)):
        rh = rng.randint(1, min(3, rep_h))
        rw = rng.randint(1, min(3, rep_w))
        rr = rng.randint(0, rep_h - rh)
        cc = rng.randint(0, rep_w - rw)
        color = rng.choice(accents)
        _paint_orbit_rect(patch, rr, cc, rh, rw, color)

    used = set(map(int, patch.ravel()))
    if len(used) < 3:
        _paint_orbit_rect(patch, 1, 1, 2, 1, accents[0])
        _paint_orbit_rect(patch, max(0, rep_h - 2), max(0, rep_w - 2), 1, 2, accents[-1])
    return patch


def _random_connected_shape(
    rng: random.Random,
    *,
    min_area: int = 4,
    max_area: int = 8,
    target_area: int | None = None,
    target_dims: tuple[int, int] | None = None,
    require_d4_variety: bool = False,
    forbid_variants: set[tuple[tuple[int, int], ...]] | None = None,
) -> np.ndarray:
    for _ in range(500):
        h = rng.randint(3, 5)
        w = rng.randint(3, 5)
        area_hi = min(max_area, h * w - 1)
        if target_area is None and area_hi < min_area:
            continue
        area = target_area if target_area is not None else rng.randint(min_area, area_hi)
        start = (rng.randrange(h), rng.randrange(w))
        cells = {start}
        frontier = set()
        frontier.update(
            (nr, nc)
            for nr, nc in _neighbors(start[0], start[1], 8)
            if 0 <= nr < h and 0 <= nc < w
        )
        while len(cells) < area and frontier:
            cell = rng.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            frontier.update(
                (nr, nc)
                for nr, nc in _neighbors(cell[0], cell[1], 8)
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells
            )
        if len(cells) != area:
            continue
        coords = np.array(sorted(cells), dtype=int)
        coords = np.array(_normalize(coords), dtype=int)
        sig = _normalize(coords)
        variants = _d4_signatures(coords)
        dims = _coords_to_mask(coords).shape
        if target_dims is not None and dims != target_dims:
            continue
        if require_d4_variety and len(variants) < 4:
            continue
        if forbid_variants is not None and sig in forbid_variants:
            continue
        return coords
    raise RuntimeError("failed to build connected shape")


def _hole_mode_ok(coords: np.ndarray, a2: int, b2: int, mode: str) -> bool:
    r2 = 2 * coords[:, 0]
    c2 = 2 * coords[:, 1]
    if mode == "quadrant":
        return (
            (np.all(r2 < b2) or np.all(r2 > b2))
            and (np.all(c2 < a2) or np.all(c2 > a2))
        )
    if mode == "vertical":
        crosses_v = np.min(c2) <= a2 <= np.max(c2)
        one_h_side = np.all(r2 < b2) or np.all(r2 > b2)
        return crosses_v and one_h_side
    if mode == "horizontal":
        crosses_h = np.min(r2) <= b2 <= np.max(r2)
        one_v_side = np.all(c2 < a2) or np.all(c2 > a2)
        return crosses_h and one_v_side
    return False


def _choose_hole(
    patch: np.ndarray,
    shape: np.ndarray,
    rng: random.Random,
) -> tuple[np.ndarray, Counter[int], str]:
    h, w = patch.shape
    hs, ws = _coords_to_mask(shape).shape
    a2 = w - 1
    b2 = h - 1
    modes = ["quadrant", "vertical", "horizontal"]

    for _ in range(800):
        mode = rng.choice(modes)
        if mode == "quadrant":
            r0 = rng.randint(1, h - hs - 1)
            c0 = rng.randint(1, w - ws - 1)
        elif mode == "vertical":
            center_c = w // 2 - ws // 2 + rng.randint(-1, 1)
            c0 = max(1, min(w - ws - 1, center_c))
            r0 = rng.randint(1, h - hs - 1)
        else:
            center_r = h // 2 - hs // 2 + rng.randint(-1, 1)
            r0 = max(1, min(h - hs - 1, center_r))
            c0 = rng.randint(1, w - ws - 1)

        placed = shape + np.array([r0, c0], dtype=int)
        if not _hole_mode_ok(placed, a2, b2, mode):
            continue

        cell_set = {tuple(map(int, rc)) for rc in placed}
        ok = True
        for r, c in cell_set:
            if r <= 0 or c <= 0 or r >= h - 1 or c >= w - 1:
                ok = False
                break
            mirrors = _mirrored_positions(r, c, a2, b2)
            if not any((rr, cc) not in cell_set for rr, cc in mirrors):
                ok = False
                break
        if not ok:
            continue

        colors = Counter(int(patch[r, c]) for r, c in cell_set)
        if len(colors) < 2:
            continue
        ordered = colors.most_common()
        if len(ordered) > 1 and ordered[0][1] == ordered[1][1]:
            continue

        visible = patch.copy()
        for r, c in cell_set:
            visible[r, c] = BG
        visible_counter = Counter(int(v) for v in visible.ravel() if int(v) != BG)
        if not visible_counter:
            continue
        if visible_counter.most_common(1)[0][0] == ordered[0][0]:
            continue

        return placed, colors, mode

    raise RuntimeError("failed to place hole")


def _shape_patch(shape: np.ndarray, color: int) -> np.ndarray:
    mask = _coords_to_mask(shape)
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _place_object(grid: np.ndarray, shape: np.ndarray, color: int, allowed_cols: tuple[int, int], rng: random.Random) -> np.ndarray:
    patch = _shape_patch(shape, color)
    h, w = patch.shape
    rows, cols = grid.shape
    cmin, cmax = allowed_cols
    for _ in range(500):
        r0 = rng.randint(1, rows - h - 1)
        c0 = rng.randint(cmin, cmax - w + 1)
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(rows, r0 + h + 1)
        cc1 = min(cols, c0 + w + 1)
        if np.any(grid[rr0:rr1, cc0:cc1] != BG):
            continue
        mask = patch != BG
        grid[r0 : r0 + h, c0 : c0 + w][mask] = patch[mask]
        return shape + np.array([r0, c0], dtype=int)
    raise RuntimeError("failed to place object")


def _make_distractor(
    hole_shape: np.ndarray,
    hole_variants: set[tuple[tuple[int, int], ...]],
    rng: random.Random,
    *,
    same_area: bool,
    same_dims: bool,
) -> np.ndarray:
    dims = _coords_to_mask(hole_shape).shape if same_dims else None
    area = len(hole_shape) if same_area else None
    for _ in range(500):
        shape = _random_connected_shape(
            rng,
            target_area=area,
            target_dims=dims,
            forbid_variants=hole_variants,
        )
        if _normalize(shape) not in hole_variants:
            return shape
    raise RuntimeError("failed to build distractor")


def _build_example(rng: random.Random) -> tuple[np.ndarray, np.ndarray, dict]:
    for _ in range(500):
        try:
            H = rng.randint(18, 27)
            W = rng.randint(22, 30)
            patch_h = rng.randint(10, min(16, H - 4))
            patch_w = rng.randint(10, min(15, W - 8))
            side = rng.choice(["left", "right"])
            field_width = W - patch_w - 4
            if field_width < 7:
                continue

            patch = _make_symmetric_patch(patch_h, patch_w, rng)
            hole_shape = _random_connected_shape(rng, require_d4_variety=True)
            hole_coords, repair_counts, hole_mode = _choose_hole(patch, hole_shape, rng)
            hole_winner = repair_counts.most_common(1)[0][0]
            hole_variants = _d4_signatures(hole_coords)

            grid_in = np.zeros((H, W), dtype=int)
            grid_out = np.zeros((H, W), dtype=int)
            top = rng.randint(1, H - patch_h - 1)
            left = 1 if side == "left" else W - patch_w - 2
            grid_in[top : top + patch_h, left : left + patch_w] = patch
            grid_out[top : top + patch_h, left : left + patch_w] = patch
            for r, c in hole_coords:
                grid_in[top + r, left + c] = BG

            if side == "left":
                allowed_cols = (left + patch_w + 2, W - 2)
            else:
                allowed_cols = (1, left - 2)
            if allowed_cols[1] - allowed_cols[0] + 1 < 4:
                continue

            variant_list = list(hole_variants)
            base_sig = _normalize(hole_shape)
            alt_variants = [sig for sig in variant_list if sig != base_sig]
            match_shapes = []
            match_shapes.append(np.array(base_sig, dtype=int))
            if alt_variants:
                match_shapes.append(np.array(rng.choice(alt_variants), dtype=int))
            if rng.random() < 0.5:
                match_shapes.append(np.array(rng.choice(variant_list), dtype=int))
            match_shapes = match_shapes[: rng.randint(1, min(3, len(match_shapes)))]

            distractors = [
                _make_distractor(hole_shape, hole_variants, rng, same_area=True, same_dims=True),
                _make_distractor(hole_shape, hole_variants, rng, same_area=True, same_dims=False),
            ]
            for _ in range(rng.randint(1, 2)):
                distractors.append(_make_distractor(hole_shape, hole_variants, rng, same_area=False, same_dims=False))

            object_shapes = [(True, shp) for shp in match_shapes] + [(False, shp) for shp in distractors]
            rng.shuffle(object_shapes)
            object_count = len(object_shapes)

            obj_colors = rng.sample(list(PALETTE), object_count)
            if hole_winner in obj_colors and all(is_match for is_match, _ in object_shapes):
                continue
            if hole_winner not in obj_colors and any(not is_match for is_match, _ in object_shapes) and rng.random() < 0.5:
                idx = next(i for i, (is_match, _) in enumerate(object_shapes) if not is_match)
                other = [c for c in PALETTE if c not in obj_colors or c == obj_colors[idx]]
                if hole_winner in other:
                    obj_colors[idx] = hole_winner

            placed = []
            for (is_match, shape), color in zip(object_shapes, obj_colors):
                coords = _place_object(grid_in, shape, color, allowed_cols, rng)
                placed.append((is_match, coords, color))
                if is_match:
                    grid_out[coords[:, 0], coords[:, 1]] = hole_winner
        except RuntimeError:
            continue

        for r, c in hole_coords:
            grid_out[top + r, left + c] = patch[r, c]

        meta = {
            "side": side,
            "hole_mode": hole_mode,
            "winner": hole_winner,
            "match_count": sum(1 for is_match, _, _ in placed if is_match),
            "noise_count": sum(1 for is_match, _, _ in placed if not is_match),
            "has_alt_match": any(_normalize(coords - coords.min(axis=0)) != base_sig for is_match, coords, _ in placed if is_match),
        }
        return grid_in, grid_out, meta
    raise RuntimeError("failed to build example")


def _validate_generated(inp: np.ndarray, expected: np.ndarray, meta: dict) -> bool:
    if inp.ndim != 2 or expected.ndim != 2 or max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if meta["match_count"] < 1 or meta["noise_count"] < 1:
        return False
    out = main(inp)
    if not np.array_equal(out, expected):
        return False

    fg = _components_from_mask(inp != BG, connectivity=8)
    if len(fg) < 3:
        return False
    mosaic = max(fg, key=len)
    r0, c0, r1, c1 = _bbox(mosaic)
    sub = inp[r0 : r1 + 1, c0 : c1 + 1]
    zero_components = _components_from_mask(sub == BG, connectivity=8)
    inner = [
        comp
        for comp in zero_components
        if np.all(comp[:, 0] > 0)
        and np.all(comp[:, 0] < sub.shape[0] - 1)
        and np.all(comp[:, 1] > 0)
        and np.all(comp[:, 1] < sub.shape[1] - 1)
    ]
    if len(inner) != 1:
        return False

    hole = inner[0]
    hole_patch = expected[r0 : r1 + 1, c0 : c1 + 1]
    repaired_colors = Counter(int(hole_patch[r, c]) for r, c in hole)
    ordered = repaired_colors.most_common()
    if len(ordered) < 2 or ordered[0][1] == ordered[1][1]:
        return False
    visible_colors = Counter(int(v) for v in sub.ravel() if int(v) != BG)
    if visible_colors.most_common(1)[0][0] == ordered[0][0]:
        return False

    hole_variants = _d4_signatures(hole)
    object_components = [comp for comp in fg if comp is not mosaic]
    matched = 0
    erased = 0
    winner = ordered[0][0]
    for comp in object_components:
        sig = _normalize(comp)
        out_vals = set(int(expected[r, c]) for r, c in comp)
        if sig in hole_variants:
            matched += 1
            if out_vals != {winner}:
                return False
        else:
            erased += 1
            if out_vals != {BG}:
                return False
    if matched < 1 or erased < 1:
        return False
    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(800):
        inp, expected, meta = _build_example(rng)
        if _validate_generated(inp, expected, meta):
            return inp
    raise RuntimeError("failed to generate a valid puzzle input")
