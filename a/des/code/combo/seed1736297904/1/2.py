import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAROON = 9
GEN_ATTEMPTS = 400
RNG = np.random.default_rng()


def _arr(rows) -> np.ndarray:
    return np.array(rows, dtype=bool)


SHAPE_LIBRARY = [
    _arr([[1]]),
    _arr([[1, 1]]),
    _arr([[1, 1, 1]]),
    _arr([[1, 0], [1, 1]]),
    _arr([[1, 1], [1, 1]]),
    _arr([[1, 1, 1, 1]]),
    _arr([[1, 1, 1], [0, 1, 0]]),
    _arr([[1, 0], [1, 0], [1, 1]]),
    _arr([[0, 1, 1], [1, 1, 0]]),
    _arr([[1, 1, 1, 1, 1]]),
    _arr([[0, 1, 0], [1, 1, 1], [0, 1, 0]]),
]


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    starts = np.argwhere(mask)
    for sr, sc in starts:
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        stack = [(sr, sc)]
        seen[sr, sc] = True
        coords = []
        while stack:
            r, c = stack.pop()
            coords.append((r, c))
            for dr, dc in DIR4:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _coords_to_mask(coords: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return (r0, c0, r1, c1), mask


def _rotations(mask: np.ndarray) -> list[np.ndarray]:
    rots: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rot = np.rot90(mask, k)
        key = (tuple(map(int, rot.shape)), rot.tobytes())
        if key not in seen:
            seen.add(key)
            rots.append(rot)
    return rots


def _canonical_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    keys = [(tuple(map(int, rot.shape)), rot.tobytes()) for rot in _rotations(mask)]
    return min(keys)


def _contact_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    count = 0
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        for dr, dc in DIR4:
            rr = r + dr
            cc = c + dc
            if not (0 <= rr < h and 0 <= cc < w and mask[rr, cc]):
                count += 1
    return count


def _render_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    rows, cols = np.argwhere(mask).T
    grid[top + rows, left + cols] = int(color)


def _bbox_free(occupied: np.ndarray, top: int, left: int, height: int, width: int, pad: int) -> bool:
    h, w = occupied.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    return not bool(np.any(occupied[r0:r1, c0:c1]))


def _mark_bbox(occupied: np.ndarray, top: int, left: int, height: int, width: int, pad: int) -> None:
    h, w = occupied.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    occupied[r0:r1, c0:c1] = True


def _try_place_masks(
    canvas_shape: tuple[int, int],
    masks: list[np.ndarray],
    pad: int,
    occupied: np.ndarray | None = None,
) -> list[tuple[int, int]] | None:
    h, w = canvas_shape
    used = np.zeros((h, w), dtype=bool) if occupied is None else occupied.copy()
    order = sorted(
        range(len(masks)),
        key=lambda i: (
            -int(masks[i].shape[0] * masks[i].shape[1]),
            -int(masks[i].sum()),
        ),
    )
    pos: list[tuple[int, int] | None] = [None] * len(masks)
    for idx in order:
        mask = masks[idx]
        mh, mw = map(int, mask.shape)
        candidates = []
        for top in range(h - mh + 1):
            for left in range(w - mw + 1):
                if _bbox_free(used, top, left, mh, mw, pad):
                    candidates.append((top, left))
        if not candidates:
            return None
        choice = candidates[int(RNG.integers(len(candidates)))]
        pos[idx] = choice
        _mark_bbox(used, choice[0], choice[1], mh, mw, pad)
    return [(int(r), int(c)) for r, c in pos]


def _pick_n_distinct(pool: list[int], n: int) -> list[int]:
    idx = RNG.choice(len(pool), size=n, replace=False)
    return [int(pool[i]) for i in idx]


def _legend_data(grid: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    best_rows: list[tuple[int, int, np.ndarray, tuple[int, int]]] = []
    best_color = None
    for color in map(int, np.unique(g)):
        if color == bg:
            continue
        rows = []
        for coords in _components(g == color):
            r0, c0, r1, c1 = _bbox(coords)
            if r0 != r1:
                continue
            if int(coords.shape[0]) != c1 - c0 + 1:
                continue
            right_c = c1 + 1
            if right_c >= w:
                continue
            fill = int(g[r0, right_c])
            if fill in (bg, color):
                continue
            ok = True
            for dr, dc in DIR4:
                rr = r0 + dr
                cc = right_c + dc
                if not (0 <= rr < h and 0 <= cc < w):
                    continue
                if dr == 0 and dc == -1:
                    continue
                if int(g[rr, cc]) != bg:
                    ok = False
                    break
            if not ok:
                continue
            rows.append((int(coords.shape[0]), fill, coords, (int(r0), int(right_c))))
        if len(rows) > len(best_rows):
            best_rows = rows
            best_color = color
    assert best_color is not None and len(best_rows) >= 2, "legend not found"

    mapping: dict[int, int] = {}
    mask = np.zeros_like(g, dtype=bool)
    for dots, fill, coords, (rr, cc) in best_rows:
        assert dots not in mapping or mapping[dots] == fill
        mapping[dots] = fill
        mask[coords[:, 0], coords[:, 1]] = True
        mask[rr, cc] = True
    assert mapping, "legend mapping must be non-empty"
    return mapping, mask


def _find_frame(grid: np.ndarray, bg: int, ignore_mask: np.ndarray) -> tuple[int, tuple[int, int, int, int], np.ndarray]:
    work = np.asarray(grid, dtype=int).copy()
    work[ignore_mask] = int(bg)
    best = None
    for color in map(int, np.unique(work)):
        if color == bg:
            continue
        for coords in _components(work == color):
            area = int(coords.shape[0])
            if best is None or area > best[0]:
                best = (area, color, coords)
    assert best is not None, "frame not found"
    _, color, coords = best
    frame_mask = np.zeros_like(work, dtype=bool)
    frame_mask[coords[:, 0], coords[:, 1]] = True
    return color, _bbox(coords), frame_mask


def _hole_specs(subgrid: np.ndarray, frame_color: int, bg: int) -> list[dict]:
    h, w = subgrid.shape
    holes = []
    for coords in _components(subgrid == bg):
        if np.any(coords[:, 0] == 0) or np.any(coords[:, 0] == h - 1) or np.any(coords[:, 1] == 0) or np.any(coords[:, 1] == w - 1):
            continue
        (r0, c0, r1, c1), mask = _coords_to_mask(coords)
        contact = 0
        for rr, cc in coords:
            rr = int(rr)
            cc = int(cc)
            for dr, dc in DIR4:
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and int(subgrid[nr, nc]) == int(frame_color):
                    contact += 1
        holes.append(
            {
                "bbox": (r0, c0, r1, c1),
                "mask": mask,
                "contact": int(contact),
                "key": _canonical_key(mask),
            }
        )
    assert holes, "at least one hole is required"
    return holes


def _sprite_keys(grid: np.ndarray, bg: int, frame_bbox: tuple[int, int, int, int], legend_mask: np.ndarray) -> dict[tuple[tuple[int, int], bytes], int]:
    work = np.asarray(grid, dtype=int).copy()
    work[legend_mask] = int(bg)
    r0, c0, r1, c1 = frame_bbox
    work[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    counts: dict[tuple[tuple[int, int], bytes], int] = {}
    for coords in _components(work != bg):
        vals = np.unique(work[coords[:, 0], coords[:, 1]])
        assert len(vals) == 1, "sprites must be monochrome"
        _bbox_value, mask = _coords_to_mask(coords)
        key = _canonical_key(mask)
        counts[key] = counts.get(key, 0) + 1
    return counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match outer sprites to holes in a central frame, then recolor holes by a contact-count legend.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)

    legend_map, legend_mask = _legend_data(g, bg)
    frame_color, frame_bbox, _frame_mask = _find_frame(g, bg, legend_mask)
    r0, c0, r1, c1 = frame_bbox

    sub = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    holes = _hole_specs(sub, frame_color, bg)
    sprites = _sprite_keys(g, bg, frame_bbox, legend_mask)

    out = sub.copy()
    for hole in holes:
        key = hole["key"]
        assert sprites.get(key, 0) > 0, "every hole must have a matching sprite"
        sprites[key] -= 1
        contact = int(hole["contact"])
        assert contact in legend_map, "missing legend entry for contact count"
        fill = int(legend_map[contact])
        hr0, hc0, hr1, hc1 = hole["bbox"]
        patch = out[hr0 : hr1 + 1, hc0 : hc1 + 1]
        patch[hole["mask"]] = fill

    return out


def _build_shape_buckets() -> dict[int, list[np.ndarray]]:
    buckets: dict[int, list[np.ndarray]] = {}
    for mask in SHAPE_LIBRARY:
        buckets.setdefault(_contact_count(mask), []).append(mask)
    return buckets


SHAPE_BUCKETS = _build_shape_buckets()


def _random_rot(mask: np.ndarray) -> np.ndarray:
    return np.rot90(mask, int(RNG.integers(4)))


def _make_legend(count_to_color: dict[int, int], bg: int) -> np.ndarray:
    rows = list(count_to_color.items())
    RNG.shuffle(rows)
    legend_h = 2 * len(rows) - 1
    legend_w = max(count for count, _ in rows) + 1
    legend = np.full((legend_h, legend_w), bg, dtype=int)
    for i, (count, color) in enumerate(rows):
        row = 2 * i
        legend[row, :count] = MAROON
        legend[row, count] = int(color)
    return legend


def _corner_position(grid_shape: tuple[int, int], obj_shape: tuple[int, int], corner: int) -> tuple[int, int]:
    gh, gw = grid_shape
    oh, ow = obj_shape
    if corner == 0:
        return 1, 1
    if corner == 1:
        return 1, gw - ow - 1
    if corner == 2:
        return gh - oh - 1, 1
    return gh - oh - 1, gw - ow - 1


def _frame_position(grid_shape: tuple[int, int], frame_shape: tuple[int, int], blocked: np.ndarray) -> tuple[int, int] | None:
    gh, gw = grid_shape
    fh, fw = frame_shape
    candidates = []
    for top in range(1, gh - fh):
        for left in range(1, gw - fw):
            if _bbox_free(blocked, top, left, fh, fw, pad=1):
                center_r = top + fh / 2.0
                center_c = left + fw / 2.0
                dist = abs(center_r - gh / 2.0) + abs(center_c - gw / 2.0)
                candidates.append((dist, top, left))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    keep = candidates[: max(1, len(candidates) // 4)]
    _, top, left = keep[int(RNG.integers(len(keep)))]
    return int(top), int(left)


def _gen_attempt() -> np.ndarray | None:
    n_holes = int(RNG.integers(3, 5))
    available_counts = sorted(SHAPE_BUCKETS)
    chosen_counts = [int(x) for x in RNG.choice(available_counts, size=n_holes, replace=False)]
    chosen_masks = [_random_rot(SHAPE_BUCKETS[count][int(RNG.integers(len(SHAPE_BUCKETS[count])))]).copy() for count in chosen_counts]

    interior_h = int(RNG.integers(9, 13))
    interior_w = int(RNG.integers(11, 16))
    hole_pos = _try_place_masks((interior_h, interior_w), chosen_masks, pad=1)
    if hole_pos is None:
        return None

    frame_h = interior_h + 2
    frame_w = interior_w + 2

    non_maroon = [c for c in range(10) if c != MAROON]
    bg = int(non_maroon[int(RNG.integers(len(non_maroon)))])
    frame_pool = [c for c in non_maroon if c != bg]
    frame_color = int(frame_pool[int(RNG.integers(len(frame_pool)))])

    fill_pool = [c for c in range(10) if c not in (bg, frame_color, MAROON)]
    if len(fill_pool) < n_holes:
        return None
    fill_colors = _pick_n_distinct(fill_pool, n_holes)
    sprite_pool = [c for c in range(10) if c not in (bg, frame_color, MAROON)]
    sprite_colors = _pick_n_distinct(sprite_pool, n_holes)
    count_to_color = {int(count): int(color) for count, color in zip(chosen_counts, fill_colors)}

    frame = np.full((frame_h, frame_w), frame_color, dtype=int)
    for mask, (top, left) in zip(chosen_masks, hole_pos):
        _render_mask(frame, mask, top + 1, left + 1, bg)

    legend = _make_legend(count_to_color, bg)
    gh = int(RNG.integers(max(24, frame_h + 8), 31))
    gw = int(RNG.integers(max(24, max(frame_w, legend.shape[1]) + 8), 31))
    if gh <= frame_h + 2 or gw <= frame_w + 2:
        return None
    grid = np.full((gh, gw), bg, dtype=int)

    corner = int(RNG.integers(4))
    leg_top, leg_left = _corner_position(grid.shape, legend.shape, corner)
    _render_mask(grid, np.ones_like(legend, dtype=bool), leg_top, leg_left, bg)
    legend_patch = grid[leg_top : leg_top + legend.shape[0], leg_left : leg_left + legend.shape[1]]
    legend_patch[:, :] = legend

    occupied = np.zeros_like(grid, dtype=bool)
    _mark_bbox(occupied, leg_top, leg_left, legend.shape[0], legend.shape[1], pad=1)

    frame_pos = _frame_position(grid.shape, frame.shape, occupied)
    if frame_pos is None:
        return None
    frame_top, frame_left = frame_pos
    grid[frame_top : frame_top + frame_h, frame_left : frame_left + frame_w] = frame
    _mark_bbox(occupied, frame_top, frame_left, frame_h, frame_w, pad=1)

    sprite_masks = [_random_rot(mask.copy()) for mask in chosen_masks]
    sprite_pos = _try_place_masks(grid.shape, sprite_masks, pad=1, occupied=occupied)
    if sprite_pos is None:
        return None
    for mask, (top, left), color in zip(sprite_masks, sprite_pos, sprite_colors):
        _render_mask(grid, mask, top, left, color)

    if _bg_color(grid) != bg:
        return None

    legend_map, legend_mask = _legend_data(grid, bg)
    if legend_map != count_to_color:
        return None

    frame_color_found, frame_bbox, _ = _find_frame(grid, bg, legend_mask)
    if frame_color_found != frame_color:
        return None
    fr0, fc0, fr1, fc1 = frame_bbox
    if (fr1 - fr0 + 1, fc1 - fc0 + 1) != frame.shape:
        return None

    holes = _hole_specs(grid[fr0 : fr1 + 1, fc0 : fc1 + 1], frame_color, bg)
    if len(holes) != n_holes:
        return None
    if sorted(hole["contact"] for hole in holes) != sorted(chosen_counts):
        return None

    sprites = _sprite_keys(grid, bg, frame_bbox, legend_mask)
    hole_keys = sorted(hole["key"] for hole in holes)
    sprite_keys = sorted([key for key, count in sprites.items() for _ in range(count)])
    if hole_keys != sprite_keys:
        return None

    out = main(grid)
    if out.shape != frame.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
