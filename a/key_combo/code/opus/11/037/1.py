import numpy as np


UNKNOWN = -1
BG = 0
COLORS = (0, 1, 2)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _runs_from_mask(mask: np.ndarray):
    runs = []
    i = 0
    while i < len(mask):
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < len(mask) and mask[j]:
            j += 1
        runs.append((i, j))
        i = j
    return runs


def _dense_runs(counts: np.ndarray):
    counts = np.asarray(counts, dtype=int)
    thresholds = sorted({int(v) for v in counts if int(v) > 0})
    best = None
    for threshold in thresholds:
        runs = _runs_from_mask(counts >= threshold)
        if len(runs) < 2:
            continue
        lengths = {b - a for a, b in runs}
        gaps = {runs[i + 1][0] - runs[i][1] for i in range(len(runs) - 1)}
        if len(lengths) != 1 or len(gaps) != 1:
            continue
        support = sum(b - a for a, b in runs)
        signal = int(counts[counts >= threshold].sum())
        cand = (support, signal, threshold)
        if best is None or cand > best[0]:
            best = (cand, runs)
    assert best is not None, f"failed to detect dense runs from counts={counts.tolist()}"
    return best[1]


def _vote_blocks(blocks):
    n_slots = len(blocks)
    slot_h, slot_w = blocks[0].shape
    counts = np.zeros((slot_h, slot_w, 3), dtype=int)
    for block in blocks:
        for color in COLORS:
            counts[:, :, color] += (block == color)

    voted = np.full((slot_h, slot_w), UNKNOWN, dtype=int)
    for r in range(slot_h):
        for c in range(slot_w):
            best = int(counts[r, c].max())
            winners = np.flatnonzero(counts[r, c] == best)
            if best > n_slots // 2 and len(winners) == 1:
                voted[r, c] = int(winners[0])
    return voted, counts


def _bbox_from_known_foreground(sprite):
    coords = np.argwhere(sprite > 0)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _infer_vertical_axis(voted):
    r0, c0, r1, c1 = _bbox_from_known_foreground(voted)
    patch = voted[r0 : r1 + 1, c0 : c1 + 1]
    h, w = patch.shape
    best = []
    best_support = -1
    for axis_s in range(2 * w - 1):
        support = 0
        ok = True
        for r in range(h):
            for c in range(w):
                val = int(patch[r, c])
                if val <= 0:
                    continue
                mc = axis_s - c
                if not (0 <= mc < w):
                    ok = False
                    break
                mval = int(patch[r, mc])
                if mval == UNKNOWN:
                    continue
                if mval != val:
                    ok = False
                    break
                support += 1
            if not ok:
                break
        if ok:
            if support > best_support:
                best_support = support
                best = [axis_s]
            elif support == best_support:
                best.append(axis_s)
    assert best_support > 0
    assert len(best) == 1, f"ambiguous axis candidates: {best}"
    return c0 + best[0], (r0, c0, r1, c1)


def _fill_unknowns(voted, counts, axis_s):
    out = voted.copy()
    mirror_fills = []
    fallback_pairs = []
    seen_pairs = set()
    for r, c in np.argwhere(out == UNKNOWN):
        r = int(r)
        c = int(c)
        if out[r, c] != UNKNOWN:
            continue
        mc = int(axis_s - c)
        if 0 <= mc < out.shape[1] and out[r, mc] != UNKNOWN:
            out[r, c] = int(out[r, mc])
            mirror_fills.append((r, c))
            continue
        if 0 <= mc < out.shape[1]:
            key = (r, min(c, mc), max(c, mc))
            if key in seen_pairs:
                continue
            pair_counts = counts[r, c].copy()
            if mc != c:
                pair_counts += counts[r, mc]
            best = np.flatnonzero(pair_counts == pair_counts.max())
            assert len(best) == 1
            color = int(best[0])
            out[r, c] = color
            out[r, mc] = color
            fallback_pairs.append(key)
            seen_pairs.add(key)
            continue

        best = np.flatnonzero(counts[r, c] == counts[r, c].max())
        assert len(best) == 1
        out[r, c] = int(best[0])
    assert not np.any(out == UNKNOWN)
    return out, mirror_fills, fallback_pairs


def _solve_details(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    fg = g != BG
    row_runs = _dense_runs(fg.sum(axis=1))
    col_runs = _dense_runs(fg.sum(axis=0))
    slot_h = row_runs[0][1] - row_runs[0][0]
    slot_w = col_runs[0][1] - col_runs[0][0]
    assert slot_h > 1 and slot_w > 1

    blocks = []
    slot_origins = []
    for r0, r1 in row_runs:
        for c0, c1 in col_runs:
            block = g[r0:r1, c0:c1]
            assert block.shape == (slot_h, slot_w)
            blocks.append(block)
            slot_origins.append((r0, c0))

    voted, counts = _vote_blocks(blocks)
    axis_s, bbox = _infer_vertical_axis(voted)
    restored, mirror_fills, fallback_pairs = _fill_unknowns(voted, counts, axis_s)

    out = np.zeros_like(g)
    for r0, r1 in row_runs:
        for c0, c1 in col_runs:
            out[r0:r1, c0:c1] = restored

    return {
        "row_runs": row_runs,
        "col_runs": col_runs,
        "slot_origins": slot_origins,
        "voted": voted,
        "counts": counts,
        "bbox": bbox,
        "axis_s": axis_s,
        "restored": restored,
        "mirror_fills": mirror_fills,
        "fallback_pairs": fallback_pairs,
        "out": out,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Recover a repeated damaged symmetric sprite by majority vote first and mirror completion second.

    Input structure
    The background is black. Nonblack cells form dense repeated row bands and column bands whose intersections are equal-sized slots, plus sparse red/blue noise outside the slots. Each slot contains a corrupted copy of the same sparse two-color sprite, and some per-pixel slot votes are unresolved because no color has a strict majority.

    Rule
    Detect the regular slot array from the dense row and column bands. For each relative slot coordinate, take the strict majority color across all slots; coordinates with no strict majority remain unknown. Infer the sprite's vertical reflection axis from the already recovered foreground evidence, then fill each unknown cell from its mirror partner when that partner is known, or from the unique strongest color support over the whole mirror pair when both sides are still unknown. Write the completed clean sprite back into every slot and erase everything outside the slot array to black.

    Key insight
    Majority vote suppresses random slot corruption but leaves systematic ties unresolved; the missing information is recovered only after the partially restored sprite reveals its own reflection axis.
    """
    return _solve_details(grid)["out"]


def _is_connected(mask):
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    start = tuple(map(int, pts[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _build_orbits(h, w):
    orbits = []
    cell_to_orbit = {}
    for r in range(h):
        for c in range((w + 1) // 2):
            mc = w - 1 - c
            coords = tuple(sorted({(r, c), (r, mc)}))
            idx = len(orbits)
            orbits.append(coords)
            for cell in coords:
                cell_to_orbit[cell] = idx
    adj = [set() for _ in orbits]
    for idx, coords in enumerate(orbits):
        for r, c in coords:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w:
                    j = cell_to_orbit[(nr, nc)]
                    if j != idx:
                        adj[idx].add(j)
    return orbits, adj


def _axis_symmetric(sprite):
    return np.array_equal(sprite, sprite[:, ::-1])


def _make_sprite(rng):
    dims = [(4, 4), (4, 5), (5, 5)]
    for _ in range(300):
        h, w = dims[int(rng.integers(len(dims)))]
        target = int(rng.integers(9, 13))
        orbits, adj = _build_orbits(h, w)
        center = np.array([(sum(r for r, _ in orb) / len(orb), sum(c for _, c in orb) / len(orb)) for orb in orbits], dtype=float)
        center[:, 0] = np.abs(center[:, 0] - (h - 1) / 2)
        center[:, 1] = np.abs(center[:, 1] - (w - 1) / 2)
        seed_order = np.argsort(center[:, 0] + center[:, 1])
        active = {int(seed_order[int(rng.integers(min(3, len(seed_order))))])}
        area = sum(len(orbits[i]) for i in active)

        while True:
            mask = np.zeros((h, w), dtype=bool)
            for idx in active:
                for r, c in orbits[idx]:
                    mask[r, c] = True
            row_ok = bool(np.all(mask.sum(axis=1) > 0))
            col_ok = bool(np.all(mask.sum(axis=0) > 0))
            pair_orbits = [i for i in active if len(orbits[i]) == 2]
            if 9 <= area <= 12 and row_ok and col_ok and len(pair_orbits) >= 2 and _is_connected(mask):
                break

            frontier = sorted({j for i in active for j in adj[i] if j not in active})
            if not frontier:
                break
            choices = []
            weights = []
            for j in frontier:
                orbit_area = len(orbits[j])
                if area + orbit_area > 12:
                    continue
                cells = np.array(orbits[j], dtype=int)
                edge_bias = 1.0
                if np.any(cells[:, 0] == 0) or np.any(cells[:, 0] == h - 1) or np.any(cells[:, 1] == 0) or np.any(cells[:, 1] == w - 1):
                    edge_bias = 1.5
                choices.append(j)
                weights.append(edge_bias)
            if not choices:
                break
            pick = int(rng.choice(len(choices), p=np.array(weights) / np.sum(weights)))
            active.add(choices[pick])
            area = sum(len(orbits[i]) for i in active)

        mask = np.zeros((h, w), dtype=bool)
        for idx in active:
            for r, c in orbits[idx]:
                mask[r, c] = True
        if not (9 <= int(mask.sum()) <= 12):
            continue
        if not np.all(mask.sum(axis=1) > 0) or not np.all(mask.sum(axis=0) > 0):
            continue
        if not _is_connected(mask):
            continue

        fg_orbits = [orbits[i] for i in active]
        sprite = np.zeros((h, w), dtype=int)
        orbit_colors = []
        for orb in fg_orbits:
            color = int(1 + rng.integers(2))
            orbit_colors.append(color)
            for r, c in orb:
                sprite[r, c] = color
        if len(set(orbit_colors)) < 2:
            continue
        if not _axis_symmetric(sprite):
            continue

        try:
            axis_s, bbox = _infer_vertical_axis(sprite.copy())
        except Exception:
            continue
        if axis_s != w - 1 or bbox != (0, 0, h - 1, w - 1):
            continue

        pair_orbits = [orb for orb in fg_orbits if len(orb) == 2 and 0 < orb[0][1] < w - 1]
        if len(pair_orbits) < 2:
            continue
        return sprite, pair_orbits
    return None, None


def _tie_pattern(n_slots, true_color, mode):
    other = 1 if true_color == 2 else 2
    half = n_slots // 2
    if n_slots % 2 == 0:
        if mode == "other":
            return {true_color: half, other: half, 0: 0}
        return {true_color: half, other: 0, 0: half}
    if mode == "other":
        return {true_color: half, other: half, 0: 1}
    return {true_color: half, other: 1, 0: half}


def _majority_pattern(n_slots, true_color):
    other = 1 if true_color == 2 else 2
    maj = n_slots // 2 + 1
    rem = n_slots - maj
    return {true_color: maj, other: rem // 2, 0: rem - rem // 2}


def _assign_corruptions(sprite, cell_patterns, n_slots, rng):
    events = []
    for cell, counts in cell_patterns.items():
        true_color = int(sprite[cell])
        for color, count in counts.items():
            if color == true_color:
                continue
            for _ in range(int(count)):
                events.append((cell, int(color)))
    rng.shuffle(events)
    events.sort(key=lambda item: item[0])

    slot_assign = [dict() for _ in range(n_slots)]
    slot_mods = [0] * n_slots
    for cell, color in events:
        order = np.argsort(slot_mods)
        placed = False
        for idx in order:
            i = int(idx)
            if slot_mods[i] >= 3 or cell in slot_assign[i]:
                continue
            slot_assign[i][cell] = color
            slot_mods[i] += 1
            placed = True
            break
        if not placed:
            return None

    total_mods_by_cell = {cell: n_slots - int(counts[int(sprite[cell])]) for cell, counts in cell_patterns.items()}
    safe_cells = [(r, c) for r in range(sprite.shape[0]) for c in range(sprite.shape[1]) if (r, c) not in cell_patterns]
    max_wrong = (n_slots - 1) // 2
    for i in range(n_slots):
        if slot_mods[i] >= 1:
            continue
        rng.shuffle(safe_cells)
        placed = False
        for cell in safe_cells:
            used = total_mods_by_cell.get(cell, 0)
            if used >= max_wrong or cell in slot_assign[i]:
                continue
            true_color = int(sprite[cell])
            if true_color == 0:
                new_color = int(1 + rng.integers(2))
            else:
                new_color = int(rng.choice([0, 3 - true_color]))
            slot_assign[i][cell] = new_color
            slot_mods[i] += 1
            total_mods_by_cell[cell] = used + 1
            placed = True
            break
        if not placed:
            return None

    if not all(1 <= m <= 3 for m in slot_mods):
        return None

    blocks = []
    for i in range(n_slots):
        block = sprite.copy()
        for (r, c), color in slot_assign[i].items():
            block[r, c] = color
        blocks.append(block)
    return blocks, slot_mods


def _render_slots(shape, sprite, row_starts, col_starts):
    out = np.zeros(shape, dtype=int)
    h, w = sprite.shape
    for r0 in row_starts:
        for c0 in col_starts:
            out[r0 : r0 + h, c0 : c0 + w] = sprite
    return out


def _scatter_noise(grid, row_starts, col_starts, slot_h, slot_w, rng):
    H, W = grid.shape
    in_slot = np.zeros((H, W), dtype=bool)
    for r0 in row_starts:
        for c0 in col_starts:
            in_slot[r0 : r0 + slot_h, c0 : c0 + slot_w] = True

    rows_cap = np.zeros(H, dtype=int)
    cols_cap = np.zeros(W, dtype=int)
    count = int(rng.integers(6, 13))
    candidates = [(r, c) for r in range(H) for c in range(W) if not in_slot[r, c]]
    rng.shuffle(candidates)
    made = 0
    for r, c in candidates:
        if rows_cap[r] >= 2 or cols_cap[c] >= 2:
            continue
        grid[r, c] = int(1 + rng.integers(2))
        rows_cap[r] += 1
        cols_cap[c] += 1
        made += 1
        if made == count:
            return True
    return False


def _layout_options():
    options = []
    for slot_rows, slot_cols in ((3, 3), (3, 4), (4, 3)):
        for slot_h, slot_w in ((4, 4), (4, 5), (5, 5)):
            for row_gap in (1, 2):
                for col_gap in (1, 2):
                    for top in (1, 2):
                        for left in (1, 2):
                            for bottom in (2, 3, 4):
                                for right in (2, 3, 4):
                                    H = top + slot_rows * slot_h + (slot_rows - 1) * row_gap + bottom
                                    W = left + slot_cols * slot_w + (slot_cols - 1) * col_gap + right
                                    if 18 <= H <= 30 and 18 <= W <= 30:
                                        options.append((slot_rows, slot_cols, slot_h, slot_w, row_gap, col_gap, top, left, H, W))
    return options


LAYOUTS = _layout_options()


def _gen_attempt():
    rng = np.random.default_rng()
    layout = LAYOUTS[int(rng.integers(len(LAYOUTS)))]
    slot_rows, slot_cols, slot_h, slot_w, row_gap, col_gap, top, left, H, W = layout

    sprite, pair_orbits = _make_sprite(rng)
    if sprite is None or sprite.shape != (slot_h, slot_w):
        return None

    n_slots = slot_rows * slot_cols
    copy_orbit, fallback_orbit = rng.choice(len(pair_orbits), size=2, replace=False)
    copy_orbit = pair_orbits[int(copy_orbit)]
    fallback_orbit = pair_orbits[int(fallback_orbit)]

    unknown_cell = copy_orbit[int(rng.integers(2))]
    known_cell = copy_orbit[1 - copy_orbit.index(unknown_cell)]
    left_cell, right_cell = sorted(fallback_orbit, key=lambda cell: cell[1])

    cell_patterns = {
        unknown_cell: _tie_pattern(n_slots, int(sprite[unknown_cell]), "other"),
        known_cell: _majority_pattern(n_slots, int(sprite[known_cell])),
        left_cell: _tie_pattern(n_slots, int(sprite[left_cell]), "other"),
        right_cell: _tie_pattern(n_slots, int(sprite[right_cell]), "bg"),
    }
    assigned = _assign_corruptions(sprite, cell_patterns, n_slots, rng)
    if assigned is None:
        return None
    blocks, slot_mods = assigned

    row_starts = [top + i * (slot_h + row_gap) for i in range(slot_rows)]
    col_starts = [left + i * (slot_w + col_gap) for i in range(slot_cols)]

    out = _render_slots((H, W), sprite, row_starts, col_starts)
    inp = np.zeros((H, W), dtype=int)
    idx = 0
    for r0 in row_starts:
        for c0 in col_starts:
            inp[r0 : r0 + slot_h, c0 : c0 + slot_w] = blocks[idx]
            idx += 1
    if not _scatter_noise(inp, row_starts, col_starts, slot_h, slot_w, rng):
        return None

    try:
        details = _solve_details(inp)
    except Exception:
        return None
    if not np.array_equal(details["out"], out):
        return None

    voted = details["voted"]
    counts = details["counts"]
    if voted.shape != sprite.shape or details["restored"].shape != sprite.shape:
        return None
    unknown_mask = voted == UNKNOWN
    true_fg = sprite > 0
    if int(np.sum(unknown_mask & true_fg)) != 3:
        return None
    if int(np.sum(unknown_mask & true_fg)) > int(true_fg.sum()) // 3:
        return None
    if details["axis_s"] != slot_w - 1:
        return None
    if details["bbox"] != (0, 0, slot_h - 1, slot_w - 1):
        return None
    if len(details["mirror_fills"]) < 1 or len(details["fallback_pairs"]) < 1:
        return None
    if not np.array_equal(details["restored"], sprite):
        return None
    if np.array_equal(voted, sprite):
        return None
    if slot_mods and not all(1 <= m <= 3 for m in slot_mods):
        return None

    for cell in (unknown_cell, left_cell, right_cell):
        r, c = cell
        best = np.flatnonzero(counts[r, c] == counts[r, c].max())
        if len(best) == 1:
            return None
    pair_counts = counts[left_cell[0], left_cell[1]] + counts[right_cell[0], right_cell[1]]
    if len(np.flatnonzero(pair_counts == pair_counts.max())) != 1:
        return None
    if int(np.argmax(pair_counts)) != int(sprite[left_cell]):
        return None
    if int(voted[known_cell]) != int(sprite[known_cell]):
        return None
    if int(np.count_nonzero(inp)) <= int(np.count_nonzero(out)):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
