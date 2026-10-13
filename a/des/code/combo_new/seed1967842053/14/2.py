import numpy as np


def _neighbors(r, c, h, w):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(grid):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] == 0 or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors(cr, cc, h, w):
                    if grid[nr, nc] != 0 and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _crop_mask(mask):
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0) + 1
    return mask[r0:r1, c0:c1].copy(), int(r0), int(c0)


def _is_connected(mask):
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    h, w = mask.shape
    start = tuple(int(x) for x in coords[0])
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors(r, c, h, w):
            if mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(coords)


def _parse_piece(grid, coords):
    values = np.array([grid[r, c] for r, c in coords])
    colors, counts = np.unique(values, return_counts=True)
    singleton = np.where(counts == 1)[0]
    if len(singleton) == 1:
        marker = int(colors[int(singleton[0])])
        body = int(colors[int(np.argmax(np.where(colors == marker, -1, counts)))])
    else:
        marker = int(colors[int(np.argmin(counts))])
        body = int(colors[int(np.argmax(counts))])

    marker_cell = None
    for r, c in coords:
        if int(grid[r, c]) == marker:
            marker_cell = (int(r), int(c))
            break

    arr_coords = np.array(coords, dtype=int)
    r0, c0 = arr_coords.min(axis=0)
    r1, c1 = arr_coords.max(axis=0) + 1
    mask = np.zeros((r1 - r0, c1 - c0), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return {
        "mask": mask,
        "color": body,
        "marker": marker,
        "marker_pos": marker_cell,
    }


def _packing_solutions(pieces, limit=1):
    if not pieces:
        return []
    areas = [int(np.sum(p["mask"])) for p in pieces]
    total = int(sum(areas))
    max_h = max(int(p["mask"].shape[0]) for p in pieces)
    max_w = max(int(p["mask"].shape[1]) for p in pieces)
    results = []

    for h in range(1, total + 1):
        if total % h != 0:
            continue
        w = total // h
        if h < max_h or w < max_w or h > 30 or w > 30:
            continue

        options = []
        for p in pieces:
            mask = p["mask"]
            mh, mw = mask.shape
            mcoords = np.argwhere(mask)
            piece_opts = []
            for r in range(h - mh + 1):
                for c in range(w - mw + 1):
                    flats = ((mcoords[:, 0] + r) * w + (mcoords[:, 1] + c)).astype(int)
                    piece_opts.append((flats, int(r), int(c)))
            options.append(piece_opts)
        if any(len(opts) == 0 for opts in options):
            continue

        occ = np.zeros(h * w, dtype=bool)
        unused = set(range(len(pieces)))
        placed = [None] * len(pieces)

        def backtrack():
            if len(results) >= limit:
                return
            if not unused:
                results.append((h, w, placed.copy()))
                return
            empty = int(np.flatnonzero(~occ)[0])
            choices = []
            for i in unused:
                for opt_idx, (flats, r, c) in enumerate(options[i]):
                    if empty in flats and not np.any(occ[flats]):
                        choices.append((-areas[i], i, r, c, opt_idx))
            choices.sort()
            for _, i, _, _, opt_idx in choices:
                flats, r, c = options[i][opt_idx]
                unused.remove(i)
                occ[flats] = True
                placed[i] = (r, c)
                backtrack()
                placed[i] = None
                occ[flats] = False
                unused.add(i)
                if len(results) >= limit:
                    return

        backtrack()
        if len(results) >= limit:
            return results
    return results


def _pack_rectangle(pieces, dtype):
    sols = _packing_solutions(pieces, limit=1)
    if not sols:
        raise ValueError("pieces do not tile a rectangle")
    h, w, placements = sols[0]
    out = np.zeros((h, w), dtype=dtype)
    for i, p in enumerate(pieces):
        r, c = placements[i]
        coords = np.argwhere(p["mask"])
        out[coords[:, 0] + r, coords[:, 1] + c] = p["color"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Group marked fragments and pack each marker group into a gapless rectangle.

    Input structure:
    The background is black. Each non-black 4-connected component is a small
    fragment with one body color and exactly one differently colored marker
    pixel. The marker colors identify two or three groups, and fragments from
    one group are scattered independently in the input.

    Rule:
    Find each fragment, recover its body-colored mask by treating its marker
    pixel as body color, and group fragments by marker color. For each marker
    group, translate its fragments without rotation or reflection until they
    form one solid rectangle. Place the resulting rectangles in a new black
    canvas from left to right by the first input column where each marker color
    appears, with one black column between adjacent rectangles.

    Key insight:
    The marker color is only a grouping clue; the final geometry comes from
    solving the independent no-gap rectangle packing within each group.
    """
    pieces_by_marker = {}
    first_pos = {}
    for comp in _components(grid):
        piece = _parse_piece(grid, comp)
        marker = piece["marker"]
        pieces_by_marker.setdefault(marker, []).append(piece)
        mr, mc = piece["marker_pos"]
        pos = (int(mc), int(mr))
        if marker not in first_pos or pos < first_pos[marker]:
            first_pos[marker] = pos

    ordered_markers = sorted(pieces_by_marker, key=lambda m: first_pos[m])
    rects = [_pack_rectangle(pieces_by_marker[m], grid.dtype) for m in ordered_markers]
    if not rects:
        return np.zeros((1, 1), dtype=grid.dtype)

    out_h = max(rect.shape[0] for rect in rects)
    out_w = sum(rect.shape[1] for rect in rects) + max(0, len(rects) - 1)
    out = np.zeros((out_h, out_w), dtype=grid.dtype)
    col = 0
    for rect in rects:
        h, w = rect.shape
        out[:h, col:col + w] = rect
        col += w + 1
    return out


def _random_partition(h, w, k, rng):
    labels = -np.ones((h, w), dtype=int)
    cells = [(r, c) for r in range(h) for c in range(w)]
    seeds = rng.choice(len(cells), size=k, replace=False)
    for label, idx in enumerate(seeds):
        r, c = cells[int(idx)]
        labels[r, c] = label

    remaining = h * w - k
    while remaining:
        frontier = []
        for r, c in cells:
            if labels[r, c] != -1:
                continue
            around = [labels[nr, nc] for nr, nc in _neighbors(r, c, h, w) if labels[nr, nc] != -1]
            if around:
                frontier.append((r, c, around))
        r, c, around = frontier[int(rng.integers(len(frontier)))]
        labels[r, c] = int(around[int(rng.integers(len(around)))])
        remaining -= 1
    return labels


def _make_group_masks(rng, k):
    for _ in range(500):
        h = int(rng.integers(3, 6))
        w = int(rng.integers(3, 6))
        if h * w < k * 3:
            continue
        labels = _random_partition(h, w, k, rng)
        masks = []
        ok = True
        for label in range(k):
            full = labels == label
            if int(np.sum(full)) < 3 or not _is_connected(full):
                ok = False
                break
            cropped, _, _ = _crop_mask(full)
            masks.append(cropped)
        if not ok:
            continue
        dummy = [{"mask": mask, "color": 1} for mask in masks]
        sols = _packing_solutions(dummy, limit=2)
        if len(sols) == 1 and sols[0][0] == h and sols[0][1] == w:
            return masks

    if k == 2:
        labels = np.array([[0, 0, 1, 1], [0, 1, 1, 1], [0, 0, 0, 1]], dtype=int)
    elif k == 3:
        labels = np.array([[0, 0, 1, 1], [0, 2, 2, 1], [0, 0, 2, 1], [0, 2, 2, 1]], dtype=int)
    else:
        labels = np.array(
            [
                [0, 0, 1, 1, 1],
                [0, 2, 2, 1, 3],
                [0, 0, 2, 3, 3],
                [0, 2, 2, 2, 3],
                [0, 0, 0, 3, 3],
            ],
            dtype=int,
        )
    masks = []
    for label in range(k):
        cropped, _, _ = _crop_mask(labels == label)
        masks.append(cropped)
    return masks


def _choose_marker_cell(mask, rng):
    coords = [tuple(int(x) for x in rc) for rc in np.argwhere(mask)]
    candidates = []
    preferred = []
    h, w = mask.shape
    for r, c in coords:
        test = mask.copy()
        test[r, c] = False
        if not _is_connected(test):
            continue
        degree = sum(1 for nr, nc in _neighbors(r, c, h, w) if mask[nr, nc])
        candidates.append((r, c))
        if degree >= 2:
            preferred.append((r, c))
    pool = preferred or candidates or coords
    return pool[int(rng.integers(len(pool)))]


def _can_place(grid, mask, top, left):
    h, w = grid.shape
    for rr, cc in np.argwhere(mask):
        r = int(top + rr)
        c = int(left + cc)
        if grid[r, c] != 0:
            return False
        r0 = max(0, r - 1)
        r1 = min(h, r + 2)
        c0 = max(0, c - 1)
        c1 = min(w, c + 2)
        if np.any(grid[r0:r1, c0:c1] != 0):
            return False
    return True


def _draw_fragment(grid, frag, top, left):
    coords = np.argwhere(frag["mask"])
    grid[coords[:, 0] + top, coords[:, 1] + left] = frag["body"]
    mr, mc = frag["mark_cell"]
    grid[top + mr, left + mc] = frag["marker"]
    return int(top + mr), int(left + mc)


def _place_randomly(fragments, rng):
    n = len(fragments)
    low = min(30, max(14, 12 + n))
    for _ in range(80):
        h = int(rng.integers(low, 31))
        w = int(rng.integers(low, 31))
        grid = np.zeros((h, w), dtype=int)
        marker_positions = {}
        ok = True
        for idx in rng.permutation(n):
            frag = fragments[int(idx)]
            mh, mw = frag["mask"].shape
            placed = False
            for _attempt in range(900):
                top = int(rng.integers(0, h - mh + 1))
                left = int(rng.integers(0, w - mw + 1))
                if _can_place(grid, frag["mask"], top, left):
                    mr, mc = _draw_fragment(grid, frag, top, left)
                    marker_positions.setdefault(frag["marker"], []).append((mr, mc))
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if ok:
            first_cols = [min(c for _, c in pos) for pos in marker_positions.values()]
            if len(first_cols) == len(set(first_cols)):
                return grid
    return None


def _place_in_rows(fragments, rng):
    grid = np.zeros((30, 30), dtype=int)
    row = int(rng.integers(0, 2))
    col = int(rng.integers(0, 2))
    row_h = 0
    marker_positions = {}
    for idx in rng.permutation(len(fragments)):
        frag = fragments[int(idx)]
        mh, mw = frag["mask"].shape
        if col + mw > 29:
            row += row_h + 2
            col = int(rng.integers(0, 2))
            row_h = 0
        if row + mh > 30:
            return None
        mr, mc = _draw_fragment(grid, frag, row, col)
        marker_positions.setdefault(frag["marker"], []).append((mr, mc))
        col += mw + 2
        row_h = max(row_h, mh)
    first_cols = [min(c for _, c in pos) for pos in marker_positions.values()]
    if len(first_cols) != len(set(first_cols)):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    colors = list(range(1, 10))

    for _ in range(200):
        group_count = int(rng.integers(2, 4))
        marker_colors = [int(c) for c in rng.choice(colors, size=group_count, replace=False)]
        body_palette = [c for c in colors if c not in marker_colors]
        fragments = []

        for marker in marker_colors:
            piece_count = int(rng.integers(2, 5))
            masks = _make_group_masks(rng, piece_count)
            for mask in masks:
                fragments.append(
                    {
                        "mask": mask,
                        "body": int(body_palette[int(rng.integers(len(body_palette)))]),
                        "marker": int(marker),
                        "mark_cell": _choose_marker_cell(mask, rng),
                    }
                )

        grid = _place_randomly(fragments, rng)
        if grid is None:
            grid = _place_in_rows(fragments, rng)
        if grid is not None and max(grid.shape) <= 30:
            return grid

    marker_colors = [1, 2]
    body_palette = [3, 4, 5, 6]
    fragments = []
    for marker, count in zip(marker_colors, [2, 3]):
        for mask in _make_group_masks(rng, count):
            fragments.append(
                {
                    "mask": mask,
                    "body": int(body_palette[int(rng.integers(len(body_palette)))]),
                    "marker": marker,
                    "mark_cell": _choose_marker_cell(mask, rng),
                }
            )
    grid = _place_in_rows(fragments, rng)
    if grid is None:
        grid = np.zeros((30, 30), dtype=int)
    return grid
