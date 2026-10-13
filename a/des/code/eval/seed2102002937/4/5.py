from __future__ import annotations

from collections import defaultdict, deque

import numpy as np


BG = 0
OUTER_MIN = 5
OUTER_MAX = 8
GRID_MIN = 20
GRID_MAX = 34
EDGE_MARGIN = 1
RECT_PADDING = 2
MAX_RETRIES = 500
PAIR_BUILD_TRIES = 200
PATTERN_TRIES = 80
PAIR_COUNT_CHOICES = (1, 2, 3)

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PERP = {
    (-1, 0): [(0, -1), (0, 1)],
    (1, 0): [(0, -1), (0, 1)],
    (0, -1): [(-1, 0), (1, 0)],
    (0, 1): [(-1, 0), (1, 0)],
}
OPPOSITE = {d: (-d[0], -d[1]) for d in DIRS}
SIDE_FROM_DIR = {
    (-1, 0): "top",
    (1, 0): "bottom",
    (0, -1): "left",
    (0, 1): "right",
}


def _rand_choice(rng: np.random.Generator, seq):
    return seq[int(rng.integers(0, len(seq)))]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _ring_coords(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    coords = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if r in (r0, r1) or c in (c0, c1):
                coords.append((r, c))
    return coords


def _frame_sets(top: tuple[int, int], h: int, w: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    r0, c0 = top
    ring: set[tuple[int, int]] = set()
    inner: set[tuple[int, int]] = set()
    for r in range(r0, r0 + h):
        for c in range(c0, c0 + w):
            if r in (r0, r0 + h - 1) or c in (c0, c0 + w - 1):
                ring.add((r, c))
            else:
                inner.add((r, c))
    return ring, inner


def _crop_non_bg(arr: np.ndarray, bg: int) -> np.ndarray | None:
    coords = np.argwhere(arr != bg)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    frames: list[dict] = []
    colors = [int(v) for v in np.unique(g) if int(v) != bg]
    for color in colors:
        for fh in range(OUTER_MIN, OUTER_MAX + 1):
            for fw in range(OUTER_MIN, OUTER_MAX + 1):
                need = 2 * fh + 2 * fw - 4
                for r0 in range(h - fh + 1):
                    r1 = r0 + fh - 1
                    for c0 in range(w - fw + 1):
                        c1 = c0 + fw - 1
                        sub = g[r0 : r1 + 1, c0 : c1 + 1]
                        if int(np.count_nonzero(sub == color)) != need:
                            continue
                        if not np.all(sub[0, :] == color):
                            continue
                        if not np.all(sub[-1, :] == color):
                            continue
                        if not np.all(sub[:, 0] == color):
                            continue
                        if not np.all(sub[:, -1] == color):
                            continue
                        if np.any(sub[1:-1, 1:-1] == color):
                            continue
                        frames.append(
                            {
                                "color": color,
                                "bbox": (r0, c0, r1, c1),
                                "inner_bbox": (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
                            }
                        )
    return frames


def _attachment_candidates(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> list[tuple[int, int]]:
    h, w = grid.shape
    r0, c0, r1, c1 = bbox
    out = []
    for r, c in _ring_coords(bbox):
        touched = False
        for nr, nc in _neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if r0 <= nr <= r1 and c0 <= nc <= c1:
                continue
            if int(grid[nr, nc]) == color:
                touched = True
                break
        if touched:
            out.append((r, c))
    return out


def _extract_content_patch(grid: np.ndarray, frame: dict, bg: int) -> np.ndarray | None:
    r0, c0, r1, c1 = frame["inner_bbox"]
    return _crop_non_bg(grid[r0 : r1 + 1, c0 : c1 + 1], bg)


def _trace_path(
    grid: np.ndarray,
    color: int,
    src_frame: dict,
    tgt_frame: dict,
    src_attach: tuple[int, int],
    tgt_attach: tuple[int, int],
) -> list[tuple[int, int]]:
    src_ring = set(_ring_coords(src_frame["bbox"]))
    tgt_ring = set(_ring_coords(tgt_frame["bbox"]))
    allowed: set[tuple[int, int]] = set()
    for r, c in np.argwhere(grid == color):
        pt = (int(r), int(c))
        if pt in (src_attach, tgt_attach) or (pt not in src_ring and pt not in tgt_ring):
            allowed.add(pt)

    parent: dict[tuple[int, int], tuple[int, int] | None] = {src_attach: None}
    q = deque([src_attach])
    while q and tgt_attach not in parent:
        cur = q.popleft()
        for nxt in _neighbors4(*cur):
            if nxt not in allowed or nxt in parent:
                continue
            parent[nxt] = cur
            q.append(nxt)

    assert tgt_attach in parent, "channel path must connect the two frames"
    path = []
    cur = tgt_attach
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def _count_turns(path: list[tuple[int, int]]) -> int:
    dirs = []
    for i in range(len(path) - 1):
        r0, c0 = path[i]
        r1, c1 = path[i + 1]
        dirs.append((r1 - r0, c1 - c0))
    turns = 0
    for i in range(1, len(dirs)):
        if dirs[i] != dirs[i - 1]:
            turns += 1
    return turns


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-color hollow frames, count channel turns, and copy rotated contents into the empty mate.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    frames = _find_frames(g, bg)
    usable = []
    for frame in frames:
        attachments = _attachment_candidates(g, frame["bbox"], int(frame["color"]))
        if len(attachments) == 1:
            frame = dict(frame)
            frame["attachment"] = attachments[0]
            usable.append(frame)

    by_color: dict[int, list[dict]] = defaultdict(list)
    for frame in usable:
        by_color[int(frame["color"])].append(frame)

    out = g.copy()
    for color, pair in by_color.items():
        assert len(pair) == 2, f"expected exactly two frames for color {color}"
        a, b = pair
        a_patch = _extract_content_patch(g, a, bg)
        b_patch = _extract_content_patch(g, b, bg)
        assert (a_patch is None) != (b_patch is None), "exactly one frame in a pair must contain content"

        if a_patch is None:
            src_frame, tgt_frame, src_patch = b, a, b_patch
        else:
            src_frame, tgt_frame, src_patch = a, b, a_patch

        path = _trace_path(
            g,
            int(color),
            src_frame,
            tgt_frame,
            tuple(src_frame["attachment"]),
            tuple(tgt_frame["attachment"]),
        )
        turns = _count_turns(path)
        rot = np.rot90(src_patch, -int(turns % 4))

        tr0, tc0, tr1, tc1 = tgt_frame["inner_bbox"]
        assert rot.shape[0] <= (tr1 - tr0 + 1)
        assert rot.shape[1] <= (tc1 - tc0 + 1)
        out[tr0 : tr0 + rot.shape[0], tc0 : tc0 + rot.shape[1]] = rot

    return out


def _attachment_local(rng: np.random.Generator, h: int, w: int, side: str) -> tuple[int, int]:
    if side == "top":
        return 0, int(rng.integers(1, w - 1))
    if side == "bottom":
        return h - 1, int(rng.integers(1, w - 1))
    if side == "left":
        return int(rng.integers(1, h - 1)), 0
    return int(rng.integers(1, h - 1)), w - 1


def _build_dirs(rng: np.random.Generator, turns: int) -> list[tuple[int, int]]:
    dirs = [_rand_choice(rng, DIRS)]
    for _ in range(turns):
        dirs.append(_rand_choice(rng, PERP[dirs[-1]]))
    return dirs


def _build_path(dirs: list[tuple[int, int]], lengths: list[int]) -> list[tuple[int, int]]:
    path = [(0, 0)]
    for (dr, dc), length in zip(dirs, lengths, strict=True):
        for _ in range(length):
            r, c = path[-1]
            path.append((r + dr, c + dc))
    return path


def _grow_connected_cells(
    rng: np.random.Generator, h: int, w: int, area: int
) -> set[tuple[int, int]] | None:
    seed = (int(rng.integers(0, h)), int(rng.integers(0, w)))
    cells = {seed}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(frontier[int(rng.integers(0, len(frontier)))])
    return cells


def _make_pattern(
    rng: np.random.Generator, max_h: int, max_w: int, turns: int, palette: list[int]
) -> np.ndarray | None:
    for _ in range(PATTERN_TRIES):
        h = int(rng.integers(2, max_h + 1))
        w = int(rng.integers(2, max_w + 1))
        area = int(rng.integers(3, h * w + 1))
        cells = _grow_connected_cells(rng, h, w, area)
        if cells is None:
            continue

        patch = np.full((h, w), BG, dtype=int)
        n_colors = int(rng.integers(1, min(3, len(palette)) + 1))
        colors = [int(x) for x in rng.choice(np.array(palette, dtype=int), size=n_colors, replace=False)]
        for r, c in cells:
            patch[r, c] = colors[int(rng.integers(0, len(colors)))]
        patch = _crop_non_bg(patch, BG)
        if patch is None:
            continue
        if patch.shape[0] > max_h or patch.shape[1] > max_w:
            continue
        if int(np.count_nonzero(patch)) < 3:
            continue
        if int(np.count_nonzero(patch)) == patch.size and len(np.unique(patch)) == 1:
            continue
        if turns % 4:
            rotated = np.rot90(patch, -turns)
            if patch.shape == rotated.shape and np.array_equal(patch, rotated):
                continue
        return patch
    return None


def _rect_conflict(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    ar, ac, ah, aw = a
    br, bc, bh, bw = b
    return not (
        ar + ah + pad <= br
        or br + bh + pad <= ar
        or ac + aw + pad <= bc
        or bc + bw + pad <= ac
    )


def _place_rectangles(
    rng: np.random.Generator, sizes: list[tuple[int, int]], h: int, w: int
) -> list[tuple[int, int]] | None:
    order = sorted(range(len(sizes)), key=lambda i: sizes[i][0] * sizes[i][1], reverse=True)
    rects: list[tuple[int, int, int, int] | None] = [None] * len(sizes)
    positions: list[tuple[int, int] | None] = [None] * len(sizes)

    for idx in order:
        ph, pw = sizes[idx]
        if ph + 2 * EDGE_MARGIN > h or pw + 2 * EDGE_MARGIN > w:
            return None
        candidates = []
        for r0 in range(EDGE_MARGIN, h - ph - EDGE_MARGIN + 1):
            for c0 in range(EDGE_MARGIN, w - pw - EDGE_MARGIN + 1):
                rect = (r0, c0, ph, pw)
                if any(
                    prev is not None and _rect_conflict(rect, prev, RECT_PADDING)
                    for prev in rects
                ):
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return None
        pos = _rand_choice(rng, candidates)
        rects[idx] = (pos[0], pos[1], ph, pw)
        positions[idx] = pos

    return [tuple(pos) for pos in positions if pos is not None]


def _build_pair_patch(
    rng: np.random.Generator, frame_color: int, content_palette: list[int], turns: int
) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(PAIR_BUILD_TRIES):
        sh = int(rng.integers(OUTER_MIN, OUTER_MAX + 1))
        sw = int(rng.integers(OUTER_MIN, OUTER_MAX + 1))
        th = int(rng.integers(OUTER_MIN, OUTER_MAX + 1))
        tw = int(rng.integers(OUTER_MIN, OUTER_MAX + 1))

        dirs = _build_dirs(rng, turns)
        lengths = []
        for i in range(turns + 1):
            if turns == 0:
                lo, hi = 4, 7
            elif i in (0, turns):
                lo, hi = 3, 6
            else:
                lo, hi = 2, 5
            lengths.append(int(rng.integers(lo, hi + 1)))

        path = _build_path(dirs, lengths)
        if len(set(path)) != len(path):
            continue

        src_side = SIDE_FROM_DIR[dirs[0]]
        tgt_side = SIDE_FROM_DIR[OPPOSITE[dirs[-1]]]
        src_attach_local = _attachment_local(rng, sh, sw, src_side)
        tgt_attach_local = _attachment_local(rng, th, tw, tgt_side)

        src_top = (path[0][0] - src_attach_local[0], path[0][1] - src_attach_local[1])
        tgt_top = (path[-1][0] - tgt_attach_local[0], path[-1][1] - tgt_attach_local[1])

        src_ring, src_inner = _frame_sets(src_top, sh, sw)
        tgt_ring, tgt_inner = _frame_sets(tgt_top, th, tw)
        src_frame = src_ring | src_inner
        tgt_frame = tgt_ring | tgt_inner
        path_mid = set(path[1:-1])

        if src_frame & tgt_frame:
            continue
        if path_mid & (src_frame | tgt_frame):
            continue

        bad = False
        for cell in path_mid:
            for nb in _neighbors4(*cell):
                if nb in src_inner or nb in tgt_inner:
                    bad = True
                    break
                if (nb in src_ring and nb != path[0]) or (nb in tgt_ring and nb != path[-1]):
                    bad = True
                    break
            if bad:
                break
        if bad:
            continue

        src_inner_h = sh - 2
        src_inner_w = sw - 2
        tgt_inner_h = th - 2
        tgt_inner_w = tw - 2
        if turns % 2 == 0:
            max_ph = min(src_inner_h, tgt_inner_h, 4)
            max_pw = min(src_inner_w, tgt_inner_w, 4)
        else:
            max_ph = min(src_inner_h, tgt_inner_w, 4)
            max_pw = min(src_inner_w, tgt_inner_h, 4)
        if max_ph < 2 or max_pw < 2:
            continue

        pattern = _make_pattern(rng, max_ph, max_pw, turns, content_palette)
        if pattern is None:
            continue

        ph, pw = map(int, pattern.shape)
        src_inner_top = (src_top[0] + 1, src_top[1] + 1)
        pat_r = src_inner_top[0] + int(rng.integers(0, src_inner_h - ph + 1))
        pat_c = src_inner_top[1] + int(rng.integers(0, src_inner_w - pw + 1))

        all_pts = list(src_frame | tgt_frame | set(path))
        min_r = min(r for r, _ in all_pts)
        min_c = min(c for _, c in all_pts)
        max_r = max(r for r, _ in all_pts)
        max_c = max(c for _, c in all_pts)
        patch_h = max_r - min_r + 1
        patch_w = max_c - min_c + 1
        if patch_h > 18 or patch_w > 18:
            continue

        dr = -min_r
        dc = -min_c
        inp = np.full((patch_h, patch_w), BG, dtype=int)
        for r, c in src_ring | tgt_ring | set(path):
            inp[r + dr, c + dc] = frame_color
        rr = pat_r + dr
        cc = pat_c + dc
        inp[rr : rr + ph, cc : cc + pw] = pattern

        out = inp.copy()
        rotated = np.rot90(pattern, -turns)
        tr = tgt_top[0] + 1 + dr
        tc = tgt_top[1] + 1 + dc
        out[tr : tr + rotated.shape[0], tc : tc + rotated.shape[1]] = rotated
        return inp, out

    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        n_pairs = int(_rand_choice(rng, PAIR_COUNT_CHOICES))
        frame_colors = [int(x) for x in rng.choice(np.arange(1, 10), size=n_pairs, replace=False)]
        content_palette = [c for c in range(1, 10) if c not in frame_colors]

        patches: list[tuple[np.ndarray, np.ndarray]] = []
        ok = True
        for color in frame_colors:
            turns = int(rng.integers(0, 4))
            built = _build_pair_patch(rng, color, content_palette, turns)
            if built is None:
                ok = False
                break
            patches.append(built)
        if not ok:
            continue

        sizes = [tuple(map(int, inp.shape)) for inp, _ in patches]
        total_area = sum(h * w for h, w in sizes)
        max_h = max(h for h, _ in sizes)
        max_w = max(w for _, w in sizes)

        layout = None
        for _ in range(80):
            low_h = max(GRID_MIN, max_h + 2 * EDGE_MARGIN)
            low_w = max(GRID_MIN, max_w + 2 * EDGE_MARGIN)
            high_h = max(low_h, GRID_MAX)
            high_w = max(low_w, GRID_MAX)
            h = int(rng.integers(low_h, high_h + 1))
            w = int(rng.integers(low_w, high_w + 1))
            if h * w < total_area * 2 + 80:
                continue
            positions = _place_rectangles(rng, sizes, h, w)
            if positions is not None:
                layout = (h, w, positions)
                break
        if layout is None:
            continue

        h, w, positions = layout
        inp = np.full((h, w), BG, dtype=int)
        out = np.full((h, w), BG, dtype=int)
        for (local_in, local_out), (r0, c0) in zip(patches, positions, strict=True):
            ph, pw = local_in.shape
            inp[r0 : r0 + ph, c0 : c0 + pw] = local_in
            out[r0 : r0 + ph, c0 : c0 + pw] = local_out

        if _most_common_color(inp) != BG or _most_common_color(out) != BG:
            continue
        if not np.array_equal(main(inp), out):
            continue
        return inp

    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
