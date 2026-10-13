from collections import Counter, deque

import numpy as np


BG_COLOR = 0
PATH_COLOR = 5
FRAME_COLOR = 9
SOURCE_COLORS = (1, 2, 3, 4, 6, 7, 8)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
TEMP_SIZE = 24
GEN_ATTEMPTS = 300

_GEN_RNG = np.random.default_rng()
_GEN_COUNTER = 0


def _mode_color(grid: np.ndarray) -> int:
    vals = np.asarray(grid, dtype=int).ravel()
    return int(Counter(map(int, vals)).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(pts)
    return parts


def _bbox(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_to_union(a: np.ndarray, b: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray]:
    mask = (a != bg) | (b != bg)
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return (
        a[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy(),
        b[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy(),
    )


def _tight_patch_from_component(grid: np.ndarray, coords: list[tuple[int, int]], bg: int) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c in coords:
        patch[r - r0, c - c0] = int(grid[r, c])
    return patch


def _frame_info(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        return None
    if len(coords) != 2 * (h + w) - 4:
        return None
    border = set(coords)
    for cc in range(c0, c1 + 1):
        if (r0, cc) not in border or (r1, cc) not in border:
            return None
    for rr in range(r0, r1 + 1):
        if (rr, c0) not in border or (rr, c1) not in border:
            return None
    return {
        "bbox": (r0, c0, r1, c1),
        "coords": list(coords),
        "inner": (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
    }


def _ordered_path(path_cells: list[tuple[int, int]], source_touch: list[tuple[int, int]]) -> list[tuple[int, int]]:
    path_set = set(path_cells)
    neigh = {
        p: [q for q in ((p[0] - 1, p[1]), (p[0] + 1, p[1]), (p[0], p[1] - 1), (p[0], p[1] + 1)) if q in path_set]
        for p in path_cells
    }
    if len(path_cells) == 1:
        return [path_cells[0]]

    endpoints = [p for p, qs in neigh.items() if len(qs) == 1]
    assert len(endpoints) == 2, "path must be a simple open polyline"
    start = next((p for p in source_touch if p in endpoints), None)
    assert start is not None, "source-side endpoint not found"

    ordered: list[tuple[int, int]] = []
    prev = None
    curr = start
    while True:
        ordered.append(curr)
        nxts = [p for p in neigh[curr] if p != prev]
        if not nxts:
            break
        prev, curr = curr, nxts[0]
    assert len(ordered) == len(path_cells), "path traversal must visit every cell once"
    return ordered


def _turn_count(ordered: list[tuple[int, int]]) -> int:
    if len(ordered) < 3:
        return 0
    dirs = [
        (ordered[i + 1][0] - ordered[i][0], ordered[i + 1][1] - ordered[i][1])
        for i in range(len(ordered) - 1)
    ]
    turns = 0
    for i in range(1, len(dirs)):
        if dirs[i] != dirs[i - 1]:
            turns += 1
    return turns


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route each source object through its gray path into the matching maroon slot, rotating by the number of bends.

    Input structure:
    - The background is the dominant color.
    - Small monochrome source objects in unique colors sit away from any slot.
    - Hollow rectangular slot frames are drawn in maroon, with empty interiors.
    - Each source touches exactly one 1-cell-wide gray path, and that path reaches exactly one slot frame without crossing other paths.

    Rule:
    - Detect every gray path component and use adjacency to pair it with one source object and one maroon slot.
    - Trace the path from the source side to the slot side and count how many times its direction changes by 90 degrees.
    - Remove the source object and the whole gray path from the canvas.
    - Rotate the source object clockwise by 90 degrees per bend count, then place it centered inside the matched slot while keeping its original color.

    Key insight:
    - The gray connector is not only a pairing cue: its rectilinear geometry encodes the required rotation through its bend count.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    h, w = g.shape

    frame_comps = _components(g == FRAME_COLOR)
    frames: list[dict] = []
    frame_lookup: dict[tuple[int, int], int] = {}
    for comp in frame_comps:
        info = _frame_info(g, comp)
        if info is None:
            continue
        idx = len(frames)
        frames.append(info)
        for cell in info["coords"]:
            frame_lookup[cell] = idx

    source_mask = (g != bg) & (g != PATH_COLOR) & (g != FRAME_COLOR)
    source_comps = _components(source_mask)
    sources: list[dict] = []
    source_lookup: dict[tuple[int, int], int] = {}
    for comp in source_comps:
        colors = {int(g[r, c]) for r, c in comp}
        assert len(colors) == 1, "each source object must be monochrome"
        idx = len(sources)
        sources.append(
            {
                "coords": list(comp),
                "color": next(iter(colors)),
                "patch": _tight_patch_from_component(g, comp, bg),
            }
        )
        for cell in comp:
            source_lookup[cell] = idx

    out = g.copy()
    out[g == PATH_COLOR] = bg
    for src in sources:
        for r, c in src["coords"]:
            out[r, c] = bg

    for comp in _components(g == PATH_COLOR):
        source_ids: set[int] = set()
        frame_ids: set[int] = set()
        source_touch: list[tuple[int, int]] = []
        frame_touch: list[tuple[int, int]] = []

        for r, c in comp:
            hit_source = False
            hit_frame = False
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if (nr, nc) in source_lookup:
                    source_ids.add(source_lookup[(nr, nc)])
                    hit_source = True
                if (nr, nc) in frame_lookup:
                    frame_ids.add(frame_lookup[(nr, nc)])
                    hit_frame = True
            if hit_source:
                source_touch.append((r, c))
            if hit_frame:
                frame_touch.append((r, c))

        assert len(source_ids) == 1, "each path must touch exactly one source"
        assert len(frame_ids) == 1, "each path must touch exactly one slot"
        assert len(set(source_touch)) == 1, "path should meet its source at one endpoint"
        assert len(set(frame_touch)) == 1, "path should meet its slot at one endpoint"

        ordered = _ordered_path(comp, list(set(source_touch)))
        turns = _turn_count(ordered) % 4

        src = sources[next(iter(source_ids))]
        frame = frames[next(iter(frame_ids))]
        rotated = np.rot90(src["patch"], -turns)
        inner_r0, inner_c0, inner_r1, inner_c1 = frame["inner"]
        inner_h = inner_r1 - inner_r0 + 1
        inner_w = inner_c1 - inner_c0 + 1
        ph, pw = rotated.shape
        assert ph <= inner_h and pw <= inner_w, "rotated source must fit inside its slot"
        dst_r0 = inner_r0 + (inner_h - ph) // 2
        dst_c0 = inner_c0 + (inner_w - pw) // 2
        mask = rotated != bg
        out[dst_r0 : dst_r0 + ph, dst_c0 : dst_c0 + pw][mask] = rotated[mask]

    return out


# --------------- generator ---------------


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask.astype(int))


def _random_shape_mask(rng: np.random.Generator) -> np.ndarray:
    for _ in range(200):
        hh = int(rng.integers(3, 5))
        ww = int(rng.integers(3, 5))
        area_lo = max(4, (hh * ww) // 2)
        area = int(rng.integers(area_lo, hh * ww))
        start = (int(rng.integers(0, hh)), int(rng.integers(0, ww)))
        cells = {start}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < hh and 0 <= nc < ww and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(frontier[int(rng.integers(0, len(frontier)))])
        if len(cells) != area:
            continue

        mask = np.zeros((hh, ww), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        rows, cols = np.where(mask)
        if rows.min() != 0 or rows.max() != hh - 1 or cols.min() != 0 or cols.max() != ww - 1:
            continue
        if bool(mask.all()):
            continue
        rots = {_mask_key(np.rot90(mask, k)) for k in range(4)}
        if len(rots) < 4:
            continue
        if not np.any(mask[:, -1]):
            continue
        return mask
    raise RuntimeError("failed to build a suitable source shape")


def _draw_frame(canvas: np.ndarray, top: int, left: int, inner_h: int, inner_w: int) -> None:
    bottom = top + inner_h + 1
    right = left + inner_w + 1
    canvas[top, left : right + 1] = FRAME_COLOR
    canvas[bottom, left : right + 1] = FRAME_COLOR
    canvas[top : bottom + 1, left] = FRAME_COLOR
    canvas[top : bottom + 1, right] = FRAME_COLOR


def _ordered_segment(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    assert r0 == r1 or c0 == c1
    out: list[tuple[int, int]] = []
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        for cc in range(c0, c1 + step, step):
            out.append((r0, cc))
    else:
        step = 1 if r1 >= r0 else -1
        for rr in range(r0, r1 + step, step):
            out.append((rr, c0))
    return out


def _polyline_cells(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for i in range(len(points) - 1):
        seg = _ordered_segment(points[i], points[i + 1])
        if i:
            seg = seg[1:]
        cells.extend(seg)
    return cells


def _build_pair_patch(turns: int, color: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(200):
        source_mask = _random_shape_mask(rng)
        sh, sw = source_mask.shape
        edge_rows = np.flatnonzero(source_mask[:, -1])
        anchor_row = int(rng.choice(edge_rows))
        y0 = int(rng.integers(8, 15))
        sr = y0 - anchor_row
        sc = 2
        if sr < 1 or sr + sh >= TEMP_SIZE - 1:
            continue

        min_inner = max(sh, sw) + 1
        max_inner = min_inner + 2
        inner_h = int(rng.integers(min_inner, max_inner + 1))
        inner_w = int(rng.integers(min_inner, max_inner + 1))
        fh = inner_h + 2
        fw = inner_w + 2
        x0 = sc + sw

        frame_top = frame_left = None
        points: list[tuple[int, int]] | None = None

        if turns == 0:
            x_end = int(rng.integers(x0 + 2, TEMP_SIZE - fw))
            attach = int(rng.integers(0, inner_h))
            frame_top = y0 - 1 - attach
            frame_left = x_end + 1
            points = [(y0, x0), (y0, x_end)]
        elif turns == 2:
            x1 = int(rng.integers(x0 + 2, TEMP_SIZE - fw - 2))
            x_end = int(rng.integers(x1 + 2, TEMP_SIZE - fw))
            deltas = [d for d in (-5, -4, -3, -2, 2, 3, 4, 5) if 1 <= y0 + d < TEMP_SIZE - 1]
            if not deltas:
                continue
            y1 = y0 + int(rng.choice(deltas))
            attach = int(rng.integers(0, inner_h))
            frame_top = y1 - 1 - attach
            frame_left = x_end + 1
            points = [(y0, x0), (y0, x1), (y1, x1), (y1, x_end)]
        elif turns == 1:
            below = bool(int(rng.integers(0, 2)))
            if below:
                top_lo = y0 + 3
                top_hi = TEMP_SIZE - fh
                if top_lo > top_hi:
                    continue
                frame_top = int(rng.integers(top_lo, top_hi + 1))
                y_end = frame_top - 1
            else:
                top_lo = 1
                top_hi = y0 - fh - 2
                if top_lo > top_hi:
                    continue
                frame_top = int(rng.integers(top_lo, top_hi + 1))
                y_end = frame_top + fh
            attach = int(rng.integers(0, inner_w))
            left_lo = x0 + 2 - (1 + attach)
            left_lo = max(left_lo, 1)
            left_hi = TEMP_SIZE - fw
            if left_lo > left_hi:
                continue
            frame_left = int(rng.integers(left_lo, left_hi + 1))
            x_end = frame_left + 1 + attach
            points = [(y0, x0), (y0, x_end), (y_end, x_end)]
        else:
            below = bool(int(rng.integers(0, 2)))
            if below:
                top_lo = y0 + 3
                top_hi = TEMP_SIZE - fh
                if top_lo > top_hi:
                    continue
                frame_top = int(rng.integers(top_lo, top_hi + 1))
                y_end = frame_top - 1
                y1_choices = [y for y in range(1, y0 - 1)]
            else:
                top_lo = 1
                top_hi = y0 - fh - 2
                if top_lo > top_hi:
                    continue
                frame_top = int(rng.integers(top_lo, top_hi + 1))
                y_end = frame_top + fh
                y1_choices = [y for y in range(y0 + 2, TEMP_SIZE - 1)]
            if not y1_choices:
                continue
            attach = int(rng.integers(0, inner_w))
            left_lo = x0 + 4 - (1 + attach)
            left_lo = max(left_lo, 1)
            left_hi = TEMP_SIZE - fw
            if left_lo > left_hi:
                continue
            frame_left = int(rng.integers(left_lo, left_hi + 1))
            x_end = frame_left + 1 + attach
            if x_end <= x0 + 3:
                continue
            x1 = int(rng.integers(x0 + 2, x_end - 1))
            y1 = int(rng.choice(y1_choices))
            points = [(y0, x0), (y0, x1), (y1, x1), (y1, x_end), (y_end, x_end)]

        assert points is not None and frame_top is not None and frame_left is not None
        if frame_top < 0 or frame_left < 0 or frame_top + fh > TEMP_SIZE or frame_left + fw > TEMP_SIZE:
            continue

        path_cells = _polyline_cells(points)
        path_set = set(path_cells)
        if len(path_set) != len(path_cells):
            continue

        temp_in = np.full((TEMP_SIZE, TEMP_SIZE), BG_COLOR, dtype=int)
        source_patch = np.full((sh, sw), BG_COLOR, dtype=int)
        source_patch[source_mask] = color
        temp_in[sr : sr + sh, sc : sc + sw] = source_patch

        for r, c in path_cells:
            if temp_in[r, c] != BG_COLOR:
                break
            temp_in[r, c] = PATH_COLOR
        else:
            frame_mask = np.zeros_like(temp_in, dtype=bool)
            _draw_frame(temp_in, frame_top, frame_left, inner_h, inner_w)
            frame_mask[frame_top : frame_top + fh, frame_left : frame_left + fw] = (
                temp_in[frame_top : frame_top + fh, frame_left : frame_left + fw] == FRAME_COLOR
            )

            if np.any((temp_in == PATH_COLOR) & frame_mask):
                continue
            src_cells = {
                (sr + r, sc + c)
                for r in range(sh)
                for c in range(sw)
                if bool(source_mask[r, c])
            }
            start_touch = [p for p in path_cells if sum((p[0] + dr, p[1] + dc) in src_cells for dr, dc in DIR4) > 0]
            frame_cells = {
                (r, c)
                for r in range(frame_top, frame_top + fh)
                for c in range(frame_left, frame_left + fw)
                if temp_in[r, c] == FRAME_COLOR
            }
            end_touch = [p for p in path_cells if sum((p[0] + dr, p[1] + dc) in frame_cells for dr, dc in DIR4) > 0]
            if len(start_touch) != 1 or len(end_touch) != 1:
                continue

            temp_out = np.full_like(temp_in, BG_COLOR)
            _draw_frame(temp_out, frame_top, frame_left, inner_h, inner_w)
            rotated_source = np.rot90(source_patch, -turns)
            rh, rw = rotated_source.shape
            dst_r0 = frame_top + 1 + (inner_h - rh) // 2
            dst_c0 = frame_left + 1 + (inner_w - rw) // 2
            mask = rotated_source != BG_COLOR
            temp_out[dst_r0 : dst_r0 + rh, dst_c0 : dst_c0 + rw][mask] = rotated_source[mask]

            rot = int(rng.integers(0, 4))
            rot_in = np.rot90(temp_in, rot)
            rot_out = np.rot90(temp_out, rot)
            pair_in, pair_out = _crop_to_union(rot_in, rot_out, BG_COLOR)
            return pair_in, pair_out

    raise RuntimeError("failed to build pair patch")


def _turn_schedule(num_pairs: int, rng: np.random.Generator) -> list[int]:
    global _GEN_COUNTER
    start = _GEN_COUNTER % 4
    _GEN_COUNTER += 1
    turns = [int((start + i) % 4) for i in range(num_pairs)]
    rng.shuffle(turns)
    return turns


def _pack_positions(
    patches: list[np.ndarray], rng: np.random.Generator
) -> tuple[tuple[int, int], list[tuple[int, int]], list[int]] | None:
    n = len(patches)
    candidates = []
    if n == 1:
        candidates = [1]
    elif n == 2:
        candidates = [1, 2]
    else:
        candidates = [2, 1]

    for cols in candidates:
        order = list(map(int, rng.permutation(n)))
        groups = [order[i : i + cols] for i in range(0, n, cols)]
        gap_x = int(rng.integers(1, 4))
        gap_y = int(rng.integers(1, 4))
        outer = int(rng.integers(1, 3))
        row_heights = [max(int(patches[i].shape[0]) for i in group) for group in groups]
        row_widths = [
            sum(int(patches[i].shape[1]) for i in group) + gap_x * (len(group) - 1)
            for group in groups
        ]
        base_h = outer * 2 + sum(row_heights) + gap_y * (len(groups) - 1)
        base_w = outer * 2 + max(row_widths)
        if base_h > 30 or base_w > 30:
            continue

        extra_h = int(rng.integers(0, min(4, 31 - base_h)))
        extra_w = int(rng.integers(0, min(5, 31 - base_w)))
        H = base_h + extra_h
        W = base_w + extra_w
        top_extra = int(rng.integers(0, extra_h + 1))
        y = outer + top_extra
        positions: list[tuple[int, int] | None] = [None] * n

        for row_h, row_w, group in zip(row_heights, row_widths, groups):
            x_slack = W - 2 * outer - row_w
            x = outer + int(rng.integers(0, x_slack + 1))
            for idx in group:
                patch = patches[idx]
                ph, pw = patch.shape
                positions[idx] = (y + (row_h - ph) // 2, x)
                x += pw + gap_x
            y += row_h + gap_y

        assert all(pos is not None for pos in positions)
        return (H, W), [pos for pos in positions if pos is not None], order

    return None


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    num_pairs = int(rng.choice([1, 2, 3], p=[0.2, 0.45, 0.35]))
    colors = list(rng.choice(SOURCE_COLORS, size=num_pairs, replace=False))
    turns = _turn_schedule(num_pairs, rng)

    patches_in: list[np.ndarray] = []
    patches_out: list[np.ndarray] = []
    for turn, color in zip(turns, colors):
        try:
            pin, pout = _build_pair_patch(turn, int(color), rng)
        except RuntimeError:
            return None
        patches_in.append(pin)
        patches_out.append(pout)

    packed = _pack_positions(patches_in, rng)
    if packed is None:
        return None
    (H, W), positions, _order = packed

    grid_in = np.full((H, W), BG_COLOR, dtype=int)
    grid_out = np.full((H, W), BG_COLOR, dtype=int)
    for idx, (r0, c0) in enumerate(positions):
        pin = patches_in[idx]
        pout = patches_out[idx]
        ph, pw = pin.shape
        grid_in[r0 : r0 + ph, c0 : c0 + pw] = pin
        grid_out[r0 : r0 + ph, c0 : c0 + pw] = pout

    try:
        produced = main(grid_in)
    except Exception:
        return None
    if not np.array_equal(produced, grid_out):
        return None
    return grid_in


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(_GEN_RNG)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {GEN_ATTEMPTS} attempts")
