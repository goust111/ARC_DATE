import random
from collections import Counter, deque

import numpy as np


PALETTE = tuple(range(10))
MAX_GENERATION_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) != bg:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
            comps.append(
                {
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "anchor": (r0, c0),
                    "patch": patch,
                }
            )
    return comps


def _patch_signature(patch: np.ndarray) -> tuple[tuple[int, int], bytes]:
    p = np.asarray(patch, dtype=int)
    return tuple(map(int, p.shape)), p.tobytes()


def _paste_clip(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    h, w = patch.shape
    rr0 = max(0, int(r0))
    cc0 = max(0, int(c0))
    rr1 = min(dst.shape[0], int(r0) + h)
    cc1 = min(dst.shape[1], int(c0) + w)
    if rr0 >= rr1 or cc0 >= cc1:
        return
    pr0 = rr0 - int(r0)
    pc0 = cc0 - int(c0)
    view = patch[pr0 : pr0 + (rr1 - rr0), pc0 : pc0 + (cc1 - cc0)]
    mask = view != bg
    dst_view = dst[rr0:rr1, cc0:cc1]
    dst_view[mask] = view[mask]


def _render_from_rows_cols(
    shape: tuple[int, int],
    bg: int,
    patch: np.ndarray,
    rows: list[int],
    cols: list[int],
    shift: tuple[int, int],
) -> np.ndarray:
    h, w = patch.shape
    period_r = rows[1] - rows[0]
    period_c = cols[1] - cols[0]
    dr, dc = shift
    all_rows = list(rows)
    all_cols = list(cols)

    prev_r = rows[0] - period_r
    while prev_r + dr + h > 0:
        all_rows.insert(0, prev_r)
        prev_r -= period_r
    next_r = rows[-1] + period_r
    while next_r + dr < shape[0]:
        all_rows.append(next_r)
        next_r += period_r

    prev_c = cols[0] - period_c
    while prev_c + dc + w > 0:
        all_cols.insert(0, prev_c)
        prev_c -= period_c
    next_c = cols[-1] + period_c
    while next_c + dc < shape[1]:
        all_cols.append(next_c)
        next_c += period_c

    out = np.full(shape, bg, dtype=int)
    for r0 in all_rows:
        for c0 in all_cols:
            _paste_clip(out, patch, r0 + dr, c0 + dc, bg)
    return out


def _infer_axis(values: list[int]) -> tuple[list[int], int, int | None]:
    counts = Counter(int(v) for v in values)
    candidates = [None] + [value for value, count in counts.items() if count == 1]
    solutions = []

    for removed in candidates:
        kept = sorted(value for value in counts if value != removed)
        if len(kept) < 3:
            continue
        diffs = [b - a for a, b in zip(kept, kept[1:])]
        if len(set(diffs)) != 1 or diffs[0] <= 0:
            continue
        repeat = max(counts[v] for v in kept)
        if repeat < 2:
            continue
        if removed is None:
            if any(counts[v] != repeat for v in kept):
                continue
            penalty = 0
        else:
            if counts[removed] != 1:
                continue
            if any(counts[v] not in {repeat, repeat - 1} for v in kept):
                continue
            if sum(counts[v] == repeat - 1 for v in kept) != 1:
                continue
            penalty = 1
        solutions.append((penalty, -repeat, -len(kept), kept, diffs[0], removed))

    assert solutions, "failed to infer a periodic axis"
    _, _, _, kept, period, removed = min(solutions)
    return kept, int(period), removed


def _infer_task(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    comps = _components(g, bg)
    assert len(comps) >= 9, "expected a full translation lattice with many instances"

    sig_counts = Counter(_patch_signature(comp["patch"]) for comp in comps)
    signature, count = sig_counts.most_common(1)[0]
    assert count == len(comps), "expected all visible instances to share one sprite"
    patch_shape, patch_bytes = signature
    patch = np.frombuffer(patch_bytes, dtype=np.asarray(g).dtype).reshape(patch_shape).copy()

    rows = [comp["anchor"][0] for comp in comps]
    cols = [comp["anchor"][1] for comp in comps]
    regular_rows, period_r, removed_row = _infer_axis(rows)
    regular_cols, period_c, removed_col = _infer_axis(cols)

    actual = {tuple(map(int, comp["anchor"])) for comp in comps}
    expected = {(r, c) for r in regular_rows for c in regular_cols}
    unexpected = sorted(actual - expected)
    missing = sorted(expected - actual)
    assert len(unexpected) == 1, "expected one misplaced instance"
    assert len(missing) == 1, "expected one missing lattice slot"

    actual_anchor = tuple(map(int, unexpected[0]))
    expected_anchor = tuple(map(int, missing[0]))
    shift = (actual_anchor[0] - expected_anchor[0], actual_anchor[1] - expected_anchor[1])
    assert shift != (0, 0), "expected a non-zero displacement"
    assert abs(shift[0]) < period_r and abs(shift[1]) < period_c, "misplacement must be smaller than the period"
    assert removed_row is not None or shift[0] == 0, "row outlier should agree with vertical shift"
    assert removed_col is not None or shift[1] == 0, "column outlier should agree with horizontal shift"

    return {
        "bg": bg,
        "patch": patch,
        "rows": regular_rows,
        "cols": regular_cols,
        "shift": shift,
        "expected_anchor": expected_anchor,
        "actual_anchor": actual_anchor,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the uniquely displaced tile instance, treat its offset as a global phase instruction, and retile the canvas.

    Input structure:
    - The grid has one background color and one repeated connected foreground sprite.
    - The sprite appears on a complete row-by-column translation lattice with constant row and column periods.
    - Exactly one full sprite instance is shifted away from its lattice anchor by a smaller-than-period offset.

    Rule:
    - Recover the regular lattice from the repeated sprite anchors and identify the one missing anchor together with the one unexpected anchor.
    - Use the unexpected-anchor minus missing-anchor vector as the defect displacement, which first repairs the ideal tiling.
    - Re-render the ideal infinite tiling with its global phase shifted by that same vector, clipped back to the original canvas size.

    Key insight:
    - The defect is not noise to erase; its displacement is the instruction for how the whole periodic world must be re-phased, so repair-only misses the border-consistent shifted tiling.
    """
    info = _infer_task(np.asarray(grid, dtype=int))
    return _render_from_rows_cols(
        shape=grid.shape,
        bg=info["bg"],
        patch=info["patch"],
        rows=info["rows"],
        cols=info["cols"],
        shift=info["shift"],
    )


def _random_mask(h: int, w: int) -> np.ndarray:
    min_area = max(3, (h * w + 1) // 2)
    max_area = h * w - 1 if h * w > 3 else h * w
    for _ in range(200):
        target = random.randint(min_area, max_area)
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        frontier = [start]
        while len(cells) < target and frontier:
            rr, cc = random.choice(frontier)
            options = []
            for nr, nc in _neighbors4(rr, cc):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    options.append((nr, nc))
            if not options:
                frontier.remove((rr, cc))
                continue
            nxt = random.choice(options)
            cells.add(nxt)
            frontier.append(nxt)
            if random.random() < 0.25 and len(frontier) > 3:
                frontier.pop(random.randrange(len(frontier)))

        rows = {r for r, _ in cells}
        cols = {c for _, c in cells}
        if 0 not in rows or h - 1 not in rows or 0 not in cols or w - 1 not in cols:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        return mask
    raise RuntimeError("failed to make a connected sprite mask")


def _random_patch(bg: int) -> np.ndarray:
    for _ in range(200):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        mask = _random_mask(h, w)
        fg_count = random.choice([1, 2])
        colors = random.sample([c for c in PALETTE if c != bg], fg_count)
        patch = np.full((h, w), bg, dtype=int)
        cells = list(zip(*np.where(mask)))
        for r, c in cells:
            patch[r, c] = random.choice(colors)
        if fg_count == 2:
            patch[cells[0]] = colors[0]
            patch[cells[-1]] = colors[1]
        if len(np.unique(patch[mask])) == fg_count:
            return patch
    raise RuntimeError("failed to make a colored sprite")


def _erase_patch(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    h, w = patch.shape
    mask = patch != bg
    view = dst[r0 : r0 + h, c0 : c0 + w]
    view[mask] = bg


def _translate_canvas(grid: np.ndarray, dr: int, dc: int, bg: int) -> np.ndarray:
    out = np.full_like(grid, bg)
    h, w = grid.shape
    rr0 = max(0, dr)
    rr1 = min(h, h + dr)
    cc0 = max(0, dc)
    cc1 = min(w, w + dc)
    src_r0 = rr0 - dr
    src_r1 = rr1 - dr
    src_c0 = cc0 - dc
    src_c1 = cc1 - dc
    out[rr0:rr1, cc0:cc1] = grid[src_r0:src_r1, src_c0:src_c1]
    return out


def _make_case() -> tuple[np.ndarray, np.ndarray]:
    bg = random.choice(PALETTE)
    patch = _random_patch(bg)
    ph, pw = patch.shape

    period_r = random.randint(ph + 2, min(9, ph + 6))
    period_c = random.randint(pw + 2, min(9, pw + 6))

    max_nr = min(5, 1 + (30 - ph) // period_r)
    max_nc = min(5, 1 + (30 - pw) // period_c)
    if max_nr < 3 or max_nc < 3:
        raise RuntimeError("period too large for a valid lattice")
    nr = random.randint(3, max_nr)
    nc = random.randint(3, max_nc)

    top_margin = random.randint(0, period_r - 1)
    left_margin = random.randint(0, period_c - 1)
    bottom_margin = random.randint(0, period_r - 1)
    right_margin = random.randint(0, period_c - 1)
    height = top_margin + (nr - 1) * period_r + ph + bottom_margin
    width = left_margin + (nc - 1) * period_c + pw + right_margin
    if height > 30 or width > 30:
        raise RuntimeError("canvas too large")

    rows = [top_margin + i * period_r for i in range(nr)]
    cols = [left_margin + j * period_c for j in range(nc)]

    max_dr = period_r - ph - 1
    max_dc = period_c - pw - 1
    if max_dr < 0 or max_dc < 0:
        raise RuntimeError("sprite leaves no movement slack")

    shift_kind = random.choices(["row", "col", "diag"], weights=[2, 2, 3], k=1)[0]
    if shift_kind == "row":
        dr = random.choice([d for d in range(-max_dr, max_dr + 1) if d != 0])
        dc = 0
    elif shift_kind == "col":
        dr = 0
        dc = random.choice([d for d in range(-max_dc, max_dc + 1) if d != 0])
    else:
        dr = random.choice([d for d in range(-max_dr, max_dr + 1) if d != 0] or [0])
        dc = random.choice([d for d in range(-max_dc, max_dc + 1) if d != 0] or [0])
        if dr == 0 and dc == 0:
            raise RuntimeError("diagonal branch failed to pick a non-zero shift")

    valid_sites = []
    for r0 in rows:
        for c0 in cols:
            rr = r0 + dr
            cc = c0 + dc
            if 0 <= rr <= height - ph and 0 <= cc <= width - pw:
                valid_sites.append((r0, c0))
    if not valid_sites:
        raise RuntimeError("no site can host the displaced sprite")
    defect_anchor = random.choice(valid_sites)

    ideal = np.full((height, width), bg, dtype=int)
    for r0 in rows:
        for c0 in cols:
            _paste_clip(ideal, patch, r0, c0, bg)

    inp = ideal.copy()
    _erase_patch(inp, patch, defect_anchor[0], defect_anchor[1], bg)
    _paste_clip(inp, patch, defect_anchor[0] + dr, defect_anchor[1] + dc, bg)

    expected = _render_from_rows_cols(inp.shape, bg, patch, rows, cols, (dr, dc))
    return inp, expected


def _semantic_checks(inp: np.ndarray, expected: np.ndarray) -> bool:
    info = _infer_task(inp)
    solved = main(inp)
    if not np.array_equal(solved, expected):
        return False

    ideal = _render_from_rows_cols(inp.shape, info["bg"], info["patch"], info["rows"], info["cols"], (0, 0))
    if np.array_equal(expected, ideal):
        return False

    dr, dc = info["shift"]
    if np.array_equal(expected, _translate_canvas(inp, dr, dc, info["bg"])):
        return False

    border_changed = not np.array_equal(expected[[0, -1], :], ideal[[0, -1], :]) or not np.array_equal(
        expected[:, [0, -1]], ideal[:, [0, -1]]
    )
    if not border_changed:
        shifted_patch_hits_edge = False
        ph, pw = info["patch"].shape
        for r0 in info["rows"]:
            sr = r0 + dr
            if sr < 0 or sr + ph > inp.shape[0]:
                shifted_patch_hits_edge = True
                break
        for c0 in info["cols"]:
            sc = c0 + dc
            if sc < 0 or sc + pw > inp.shape[1]:
                shifted_patch_hits_edge = True
                break
        if not shifted_patch_hits_edge:
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            inp, expected = _make_case()
        except (AssertionError, RuntimeError, ValueError):
            continue
        if _semantic_checks(inp, expected):
            return inp
    raise RuntimeError("failed to generate a valid input")
