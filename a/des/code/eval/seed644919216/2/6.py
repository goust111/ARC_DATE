from collections import defaultdict, deque
import random

import numpy as np


BG_COLOR = 0
MARKER_COLOR = 5
LIB_TILE_W = 4
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _most_common_color(grid: np.ndarray) -> int:
    flat = np.asarray(grid, dtype=int).ravel()
    counts = np.bincount(flat, minlength=max(10, int(flat.max()) + 1))
    return int(np.argmax(counts))


def _components(mask: np.ndarray) -> list[tuple[list[tuple[int, int]], bool]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[tuple[list[tuple[int, int]], bool]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            touches_border = False
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                if cr == 0 or cr == h - 1 or cc == 0 or cc == w - 1:
                    touches_border = True
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append((cells, touches_border))
    return out


def _find_separator_col(grid: np.ndarray, bg: int) -> int:
    non_bg_cols = np.where(np.any(grid != bg, axis=0))[0]
    assert non_bg_cols.size > 0, "grid must contain non-background pixels"
    last_non_bg = int(non_bg_cols[-1])
    for c in range(last_non_bg - 1, -1, -1):
        if bool(np.all(grid[:, c] == bg)):
            return c
    raise AssertionError("failed to locate library separator column")


def _row_groups(mask_1d: np.ndarray) -> list[tuple[int, int]]:
    groups: list[tuple[int, int]] = []
    r = 0
    while r < len(mask_1d):
        if not bool(mask_1d[r]):
            r += 1
            continue
        r0 = r
        while r + 1 < len(mask_1d) and bool(mask_1d[r + 1]):
            r += 1
        groups.append((r0, r))
        r += 1
    return groups


def _parse_library(lib: np.ndarray, bg: int) -> dict[int, np.ndarray]:
    mapping: dict[int, np.ndarray] = {}
    row_mask = np.any(lib != bg, axis=1)
    for r0, r1 in _row_groups(row_mask):
        block = lib[r0 : r1 + 1]
        coords = np.argwhere(block != bg)
        assert coords.size > 0, "sample block must contain non-background pixels"
        c0 = int(coords[:, 1].min())
        c1 = int(coords[:, 1].max())
        sample = block[:, c0 : c1 + 1]
        marker_col = sample[:, 0]
        marker_rows = np.where(marker_col != bg)[0]
        assert marker_rows.size > 0 and int(marker_rows[0]) == 0, "marker must start at sample top"
        marker_color = int(marker_col[0])
        count = 0
        while count < sample.shape[0] and int(sample[count, 0]) == marker_color:
            count += 1
        pattern_mask = sample != bg
        pattern_mask[:, 0] = False
        pattern_coords = np.argwhere(pattern_mask)
        assert pattern_coords.size > 0, "sample must contain a non-marker pattern"
        pr0 = int(pattern_coords[:, 0].min())
        pr1 = int(pattern_coords[:, 0].max())
        pc0 = int(pattern_coords[:, 1].min())
        pc1 = int(pattern_coords[:, 1].max())
        mapping[count] = sample[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()
    assert mapping, "library must contain at least one sample"
    return mapping


def _norm_edge(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return (a, b) if a <= b else (b, a)


def _count_region_sides(cells: list[tuple[int, int]]) -> int:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    pts = {(r - r0, c - c0) for r, c in cells}

    edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for r, c in pts:
        if (r - 1, c) not in pts:
            edges.add(((r, c), (r, c + 1)))
        if (r + 1, c) not in pts:
            edges.add(((r + 1, c), (r + 1, c + 1)))
        if (r, c - 1) not in pts:
            edges.add(((r, c), (r + 1, c)))
        if (r, c + 1) not in pts:
            edges.add(((r, c + 1), (r + 1, c + 1)))

    unused = {_norm_edge(a, b) for a, b in edges}
    adj: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for a, b in unused:
        adj[a].append(b)
        adj[b].append(a)

    sides = 0
    while unused:
        start, nxt = min(unused)
        dirs = [(nxt[0] - start[0], nxt[1] - start[1])]
        unused.remove(_norm_edge(start, nxt))
        prev = start
        cur = nxt
        while True:
            nbrs = adj[cur]
            assert len(nbrs) == 2, "boundary graph must be a collection of cycles"
            candidate = nbrs[0] if nbrs[0] != prev else nbrs[1]
            edge = _norm_edge(cur, candidate)
            if edge in unused:
                unused.remove(edge)
            dirs.append((candidate[0] - cur[0], candidate[1] - cur[1]))
            prev, cur = cur, candidate
            if cur == start:
                break
        sides += sum(dirs[i] != dirs[i - 1] for i in range(len(dirs)))
    return sides


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    sep = _find_separator_col(g, bg)

    main_area = g[:, :sep]
    library = g[:, sep + 1 :]
    samples = _parse_library(library, bg)

    out = g.copy()
    for cells, touches_border in _components(main_area == bg):
        if touches_border:
            continue
        sides = _count_region_sides(cells)
        assert sides in samples, f"missing library sample for {sides} sides"
        tile = samples[sides]
        th, tw = tile.shape
        r0 = min(r for r, _ in cells)
        c0 = min(c for _, c in cells)
        for r, c in cells:
            out[r, c] = int(tile[(r - r0) % th, (c - c0) % tw])
    return out


def _draw_polygon(vertices: list[tuple[int, int]], line_color: int) -> np.ndarray:
    h = max(r for r, _ in vertices) + 1
    w = max(c for _, c in vertices) + 1
    patch = np.full((h, w), BG_COLOR, dtype=int)
    pts = vertices + [vertices[0]]
    for (r0, c0), (r1, c1) in zip(pts, pts[1:]):
        if r0 == r1:
            lo, hi = sorted((c0, c1))
            patch[r0, lo : hi + 1] = line_color
        else:
            lo, hi = sorted((r0, r1))
            patch[lo : hi + 1, c0] = line_color
    return patch


def _interior_mask(patch: np.ndarray) -> np.ndarray:
    comps = [cells for cells, touches in _components(patch == BG_COLOR) if not touches]
    assert len(comps) == 1, "polygon patch must enclose exactly one region"
    mask = np.zeros_like(patch, dtype=bool)
    for r, c in comps[0]:
        mask[r, c] = True
    return mask


def _rotate_mask(mask: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(mask.astype(int), k).astype(bool)


def _make_rect_patch(line_color: int) -> tuple[np.ndarray, np.ndarray, int]:
    h = random.randint(3, 7)
    w = random.randint(3, 7)
    patch = _draw_polygon([(0, 0), (0, w), (h, w), (h, 0)], line_color)
    interior = _interior_mask(patch)
    return patch, interior, 4


def _make_l_patch(line_color: int) -> tuple[np.ndarray, np.ndarray, int]:
    a = random.randint(2, 4)
    b = random.randint(2, 4)
    c = random.randint(2, 4)
    patch = _draw_polygon([(0, 0), (0, a + b), (a, a + b), (a, a), (a + c, a), (a + c, 0)], line_color)
    k = random.randrange(4)
    patch = np.rot90(patch, k)
    interior = _rotate_mask(_interior_mask(_draw_polygon([(0, 0), (0, a + b), (a, a + b), (a, a), (a + c, a), (a + c, 0)], line_color)), k)
    return patch.copy(), interior.copy(), 6


def _make_notch_patch(line_color: int) -> tuple[np.ndarray, np.ndarray, int]:
    s1 = random.randint(2, 4)
    s2 = random.randint(2, 4)
    s3 = random.randint(2, 4)
    total_h = s1 + s2 + s3
    total_w = random.randint(6, 9)
    depth = random.randint(2, total_w - 2)
    base = _draw_polygon(
        [
            (0, 0),
            (0, total_w),
            (s1, total_w),
            (s1, total_w - depth),
            (s1 + s2, total_w - depth),
            (s1 + s2, total_w),
            (total_h, total_w),
            (total_h, 0),
        ],
        line_color,
    )
    k = random.randrange(4)
    patch = np.rot90(base, k)
    interior = _rotate_mask(_interior_mask(base), k)
    return patch.copy(), interior.copy(), 8


def _make_region_patch(side_count: int, line_color: int) -> tuple[np.ndarray, np.ndarray, int]:
    if side_count == 4:
        return _make_rect_patch(line_color)
    if side_count == 6:
        return _make_l_patch(line_color)
    if side_count == 8:
        return _make_notch_patch(line_color)
    raise AssertionError(f"unsupported side count {side_count}")


def _make_pattern(available_colors: list[int], seen: set[bytes]) -> np.ndarray:
    for _ in range(60):
        h = random.randint(2, 4)
        w = LIB_TILE_W
        kind = random.choice(["solid", "checker", "vstripe", "hstripe", "diag", "blocks"])
        cols = random.sample(available_colors, random.randint(1, min(3, len(available_colors))))
        tile = np.zeros((h, w), dtype=int)
        if kind == "solid":
            tile[:, :] = cols[0]
        elif kind == "checker":
            if len(cols) == 1:
                cols = cols * 2
            for r in range(h):
                for c in range(w):
                    tile[r, c] = cols[(r + c) % 2]
        elif kind == "vstripe":
            if len(cols) == 1:
                cols = cols * 2
            for c in range(w):
                tile[:, c] = cols[c % len(cols)]
        elif kind == "hstripe":
            if len(cols) == 1:
                cols = cols * 2
            for r in range(h):
                tile[r, :] = cols[r % len(cols)]
        elif kind == "diag":
            if len(cols) == 1:
                cols = cols * 2
            for r in range(h):
                for c in range(w):
                    tile[r, c] = cols[(2 * r + c) % len(cols)]
        else:
            if len(cols) == 1:
                cols = cols * 2
            split = random.randint(1, w - 1)
            tile[:, :split] = cols[0]
            tile[:, split:] = cols[1]
            if len(cols) > 2 and h > 2:
                tile[h // 2 :, split:] = cols[2]
        sig = tile.tobytes()
        if sig not in seen:
            seen.add(sig)
            return tile
    raise RuntimeError("failed to generate a distinct pattern tile")


def _place_patch(
    canvas: np.ndarray,
    reserved: np.ndarray,
    patch: np.ndarray,
) -> tuple[int, int] | None:
    h, w = canvas.shape
    ph, pw = patch.shape
    if ph + 2 > h or pw + 2 > w:
        return None
    for _ in range(80):
        r0 = random.randint(1, h - ph - 1)
        c0 = random.randint(1, w - pw - 1)
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(h, r0 + ph + 1)
        cc1 = min(w, c0 + pw + 1)
        if bool(np.any(reserved[rr0:rr1, cc0:cc1])):
            continue
        line_mask = patch != BG_COLOR
        canvas[r0 : r0 + ph, c0 : c0 + pw][line_mask] = patch[line_mask]
        reserved[rr0:rr1, cc0:cc1] = True
        return r0, c0
    return None


def _add_stub(
    canvas: np.ndarray,
    top: int,
    left: int,
    patch: np.ndarray,
    line_color: int,
) -> bool:
    ph, pw = patch.shape
    choices = []
    top_cols = np.where(patch[0] == line_color)[0]
    if top_cols.size > 0:
        choices.append(("top", top_cols))
    bottom_cols = np.where(patch[-1] == line_color)[0]
    if bottom_cols.size > 0:
        choices.append(("bottom", bottom_cols))
    left_rows = np.where(patch[:, 0] == line_color)[0]
    if left_rows.size > 0:
        choices.append(("left", left_rows))
    random.shuffle(choices)
    h, _ = canvas.shape
    for direction, coords in choices:
        picks = list(map(int, coords))
        random.shuffle(picks)
        for idx in picks:
            if direction == "top":
                gc = left + idx
                if bool(np.all(canvas[:top, gc] == BG_COLOR)):
                    canvas[:top, gc] = line_color
                    return True
            elif direction == "bottom":
                gc = left + idx
                if bool(np.all(canvas[top + ph :, gc] == BG_COLOR)):
                    canvas[top + ph :, gc] = line_color
                    return True
            else:
                gr = top + idx
                if bool(np.all(canvas[gr, :left] == BG_COLOR)):
                    canvas[gr, :left] = line_color
                    return True
    return False


def _fill_region(out: np.ndarray, cells: list[tuple[int, int]], tile: np.ndarray) -> None:
    th, tw = tile.shape
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    for r, c in cells:
        out[r, c] = int(tile[(r - r0) % th, (c - c0) % tw])


def _build_library(samples: dict[int, np.ndarray], height: int) -> np.ndarray | None:
    order = list(samples.items())
    random.shuffle(order)
    block_heights = [max(side_count, tile.shape[0]) for side_count, tile in order]
    total_h = sum(block_heights) + max(0, len(block_heights) - 1)
    if total_h > height:
        return None
    top_margin = random.randint(0, height - total_h)
    lib = np.full((height, LIB_TILE_W + 1), BG_COLOR, dtype=int)
    r = top_margin
    for (side_count, tile), block_h in zip(order, block_heights):
        lib[r : r + side_count, 0] = MARKER_COLOR
        lib[r : r + tile.shape[0], 1 : 1 + tile.shape[1]] = tile
        r += block_h + 1
    return lib


def _generate_once() -> tuple[np.ndarray, np.ndarray] | None:
    height = random.randint(26, 30)
    main_w = random.randint(15, 21)
    line_color = random.choice([1, 2, 3, 4, 6, 7, 8, 9])
    pattern_colors = [c for c in range(10) if c not in (BG_COLOR, MARKER_COLOR, line_color)]
    num_regions = random.randint(2, 4)
    num_unique = random.randint(2, min(3, num_regions))
    unique_counts = random.sample([4, 6, 8], num_unique)
    counts = unique_counts.copy()
    while len(counts) < num_regions:
        counts.append(random.choice(unique_counts))
    random.shuffle(counts)

    samples: dict[int, np.ndarray] = {}
    seen_tiles: set[bytes] = set()
    for side_count in unique_counts:
        samples[side_count] = _make_pattern(pattern_colors, seen_tiles)

    library = _build_library(samples, height)
    if library is None:
        return None

    main_area = np.full((height, main_w), BG_COLOR, dtype=int)
    reserved = np.zeros_like(main_area, dtype=bool)
    regions: list[tuple[list[tuple[int, int]], int]] = []
    placed: list[tuple[int, int, np.ndarray]] = []

    for side_count in counts:
        patch, interior_mask, _ = _make_region_patch(side_count, line_color)
        pos = _place_patch(main_area, reserved, patch)
        if pos is None:
            return None
        top, left = pos
        placed.append((top, left, patch))
        local_cells = [tuple(map(int, rc)) for rc in np.argwhere(interior_mask)]
        global_cells = [(top + r, left + c) for r, c in local_cells]
        measured = _count_region_sides(global_cells)
        if measured != side_count:
            return None
        regions.append((global_cells, side_count))

    if random.random() < 0.75:
        stub_order = list(range(len(placed)))
        random.shuffle(stub_order)
        stub_budget = random.randint(1, len(placed))
        added = 0
        for idx in stub_order:
            top, left, patch = placed[idx]
            if _add_stub(main_area, top, left, patch, line_color):
                added += 1
            if added >= stub_budget:
                break

    grid = np.full((height, main_w + 1 + library.shape[1]), BG_COLOR, dtype=int)
    grid[:, :main_w] = main_area
    grid[:, main_w + 1 :] = library

    expected = grid.copy()
    for cells, side_count in regions:
        _fill_region(expected, cells, samples[side_count])

    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        generated = _generate_once()
        if generated is None:
            continue
        inp, expected = generated
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError(f"failed to generate a valid puzzle after {GEN_ATTEMPTS} attempts")
