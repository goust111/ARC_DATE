from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


DIR4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]
DIR8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _label_mask(mask: np.ndarray) -> tuple[np.ndarray, int]:
    labels = np.zeros(mask.shape, dtype=int)
    for idx, comp in enumerate(_components(mask), start=1):
        for r, c in comp:
            labels[r, c] = idx
    return labels, int(labels.max())


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for dr, dc in DIR8:
        src_r0 = max(0, -dr)
        src_r1 = min(h, h - dr)
        src_c0 = max(0, -dc)
        src_c1 = min(w, w - dc)
        dst_r0 = max(0, dr)
        dst_r1 = min(h, h + dr)
        dst_c0 = max(0, dc)
        dst_c1 = min(w, w + dc)
        out[dst_r0:dst_r1, dst_c0:dst_c1] |= mask[src_r0:src_r1, src_c0:src_c1]
    return out


def _span_lengths(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = mask.shape
    hspan = np.zeros((h, w), dtype=int)
    vspan = np.zeros((h, w), dtype=int)

    for r in range(h):
        c = 0
        while c < w:
            if not mask[r, c]:
                c += 1
                continue
            c0 = c
            while c < w and mask[r, c]:
                c += 1
            hspan[r, c0:c] = c - c0

    for c in range(w):
        r = 0
        while r < h:
            if not mask[r, c]:
                r += 1
                continue
            r0 = r
            while r < h and mask[r, c]:
                r += 1
            vspan[r0:r, c] = r - r0

    return hspan, vspan


def _branch_regions(comp_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[int, int], dict[int, dict[int, set[int]]]]:
    hspan, vspan = _span_lengths(comp_mask)
    h, w = comp_mask.shape

    has_h = np.zeros((h, w), dtype=bool)
    has_v = np.zeros((h, w), dtype=bool)
    for r, c in np.argwhere(comp_mask):
        r = int(r)
        c = int(c)
        has_h[r, c] = (c > 0 and comp_mask[r, c - 1]) or (c + 1 < w and comp_mask[r, c + 1])
        has_v[r, c] = (r > 0 and comp_mask[r - 1, c]) or (r + 1 < h and comp_mask[r + 1, c])

    junction_mask = comp_mask & has_h & has_v & (np.abs(hspan - vspan) <= 1)
    seg_mask = comp_mask & (~junction_mask)

    seg_ids, seg_n = _label_mask(seg_mask)
    jun_ids, jun_n = _label_mask(junction_mask)

    if seg_n == 0:
        seg_ids, seg_n = _label_mask(comp_mask)
        jun_ids = np.zeros_like(comp_mask, dtype=int)
        jun_n = 0
        junction_mask[:] = False

    widths: dict[int, int] = {}
    local_width = np.minimum(hspan, vspan)
    for sid in range(1, seg_n + 1):
        vals = [int(local_width[r, c]) for r, c in np.argwhere(seg_ids == sid)]
        widths[sid] = int(Counter(vals).most_common(1)[0][0])

    adj: dict[int, set[int]] = {jid: set() for jid in range(1, jun_n + 1)}
    for r in range(h):
        for c in range(w):
            jid = int(jun_ids[r, c])
            if jid == 0:
                continue
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    sid = int(seg_ids[nr, nc])
                    if sid:
                        adj[jid].add(sid)

    allowed: dict[int, dict[int, set[int]]] = {}
    for jid, neigh in adj.items():
        allowed[jid] = {}
        neigh_list = sorted(neigh)
        for incoming in [0] + neigh_list:
            cands = [sid for sid in neigh_list if sid != incoming]
            if not cands:
                allowed[jid][incoming] = set()
                continue
            max_w = max(widths[sid] for sid in cands)
            allowed[jid][incoming] = {sid for sid in cands if widths[sid] == max_w}

    return seg_ids, jun_ids, widths, allowed


def _seed_distances(
    comp_mask: np.ndarray,
    seg_ids: np.ndarray,
    jun_ids: np.ndarray,
    allowed: dict[int, dict[int, set[int]]],
    start: tuple[int, int],
) -> np.ndarray:
    h, w = comp_mask.shape
    dist = np.full((h, w), -1, dtype=int)

    sr, sc = start
    start_seg = int(seg_ids[sr, sc])
    start_last = start_seg if start_seg else 0

    q = deque([(sr, sc, start_last)])
    seen = {(sr, sc, start_last): 0}
    dist[sr, sc] = 0

    while q:
        r, c, last_seg = q.popleft()
        cur_d = seen[(r, c, last_seg)]
        sid_here = int(seg_ids[r, c])
        jid_here = int(jun_ids[r, c])

        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or not comp_mask[nr, nc]:
                continue

            sid_next = int(seg_ids[nr, nc])
            jid_next = int(jun_ids[nr, nc])
            state: tuple[int, int, int] | None = None

            if sid_here:
                if sid_next == sid_here:
                    state = (nr, nc, sid_here)
                elif jid_next:
                    state = (nr, nc, sid_here)
            elif jid_here:
                if jid_next == jid_here:
                    state = (nr, nc, last_seg)
                elif sid_next:
                    allowed_out = allowed.get(jid_here, {}).get(last_seg, set())
                    if sid_next in allowed_out:
                        state = (nr, nc, sid_next)

            if state is None or state in seen:
                continue

            seen[state] = cur_d + 1
            if dist[nr, nc] == -1 or cur_d + 1 < dist[nr, nc]:
                dist[nr, nc] = cur_d + 1
            q.append(state)

    return dist


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed pipe interior from its colored endpoint marker, following only the widest branches.

    Input structure:
    - Background is the dominant color, with one wall color forming enclosed orthogonal pipe channels.
    - Pipe interiors are background-colored cavities fully surrounded by walls, optionally connected by junction boxes.
    - Single-pixel colored markers sit inside dead-end pipe cells; marker colors are unique and differ from the wall.

    Rule:
    - Treat every non-wall enclosed cavity as a pipe interior and flood from each marker through that interior.
    - When a flood reaches a junction, it may continue only into the widest outgoing branch; equal-width widest branches all continue.
    - If two markers can reach the same pipe cells, color each cell by the marker with the shorter pipe-path distance; exact distance ties stay background.
    - Wall pixels and the exterior background remain unchanged.

    Key insight:
    - The task is a constrained multi-source fill on pipe topology: first separate enclosed pipe interiors from the outer background, then resolve branch choices by local corridor width before comparing simultaneous distances from all markers.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    h, w = g.shape

    counts = Counter(int(v) for v in g.ravel())
    bg = int(counts.most_common(1)[0][0])

    non_bg = [color for color, _ in counts.most_common() if color != bg]
    assert non_bg, "expected a wall color"
    wall = int(non_bg[0])

    marker_cells: dict[tuple[int, int], int] = {}
    for color, cnt in counts.items():
        if color in (bg, wall) or cnt != 1:
            continue
        r, c = map(int, np.argwhere(g == color)[0])
        marker_cells[(r, c)] = int(color)

    passable = g != wall
    for comp in _components(passable):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue

        comp_mask = np.zeros((h, w), dtype=bool)
        for r, c in comp:
            comp_mask[r, c] = True

        seeds = [(pos, color) for pos, color in marker_cells.items() if comp_mask[pos]]
        if not seeds:
            continue

        seg_ids, jun_ids, _widths, allowed = _branch_regions(comp_mask)
        seed_dists = [(color, _seed_distances(comp_mask, seg_ids, jun_ids, allowed, pos)) for pos, color in seeds]

        for r, c in comp:
            if int(g[r, c]) == wall:
                continue
            best_d = None
            best_color = bg
            tie = False
            for color, dist in seed_dists:
                d = int(dist[r, c])
                if d < 0:
                    continue
                if best_d is None or d < best_d:
                    best_d = d
                    best_color = int(color)
                    tie = False
                elif d == best_d and color != best_color:
                    tie = True
            if best_d is None:
                continue
            out[r, c] = bg if tie else best_color

    return out


WALL_COLORS = [5, 6]
MARKER_COLORS = [1, 2, 3, 4, 7, 8, 9]
COMPONENT_TYPES = ["line", "branch_single_tie", "branch_single_wide", "branch_double"]
GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 80


def _tight_crop_with_margin(arr: np.ndarray, margin: int = 1) -> np.ndarray:
    pts = np.argwhere(arr != 0)
    r0 = max(0, int(pts[:, 0].min()) - margin)
    c0 = max(0, int(pts[:, 1].min()) - margin)
    r1 = min(arr.shape[0], int(pts[:, 0].max()) + margin + 1)
    c1 = min(arr.shape[1], int(pts[:, 1].max()) + margin + 1)
    return arr[r0:r1, c0:c1].copy()


def _render_component(interior: np.ndarray, markers: dict[tuple[int, int], int], wall_color: int) -> np.ndarray:
    base = np.pad(interior.astype(bool), 2)
    wall = _dilate8(base) & (~base)
    grid = np.zeros(base.shape, dtype=int)
    grid[wall] = int(wall_color)
    for (r, c), color in markers.items():
        grid[r + 2, c + 2] = int(color)
    return _tight_crop_with_margin(grid, margin=1)


def _line_component(wall_color: int, colors: list[int]) -> tuple[np.ndarray, int]:
    width = random.randint(1, 3)
    length = random.randint(width + 4, width + 11)
    interior = np.ones((width, length), dtype=bool)
    row = width // 2
    markers = {(row, 0): colors[0], (row, length - 1): colors[1]}
    grid = _render_component(interior, markers, wall_color)
    return np.rot90(grid, random.randrange(4)), 2


def _branch_component(
    wall_color: int,
    colors: list[int],
    tie: bool,
    double_marker: bool,
) -> tuple[np.ndarray, int]:
    w_stem = random.randint(1, 3)
    if tie:
        w_up = w_down = random.randint(1, 3)
    else:
        pair = [(1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2)]
        w_up, w_down = random.choice(pair)

    j = max(w_stem, w_up, w_down)
    len_left = random.randint(j + 2, j + 7)
    len_up = random.randint(j + 2, j + 7)
    len_down = random.randint(j + 2, j + 7)

    height = len_up + j + len_down
    width = len_left + j
    interior = np.zeros((height, width), dtype=bool)

    box_r0 = len_up
    box_c0 = len_left
    interior[box_r0 : box_r0 + j, box_c0 : box_c0 + j] = True

    stem_r0 = box_r0 + (j - w_stem) // 2
    interior[stem_r0 : stem_r0 + w_stem, : box_c0 + j] = True

    up_c0 = box_c0 + (j - w_up) // 2
    interior[: box_r0 + j, up_c0 : up_c0 + w_up] = True

    down_c0 = box_c0 + (j - w_down) // 2
    interior[box_r0:, down_c0 : down_c0 + w_down] = True

    leaves = {
        "stem": (stem_r0 + w_stem // 2, 0),
        "up": (0, up_c0 + w_up // 2),
        "down": (height - 1, down_c0 + w_down // 2),
    }
    chosen = ["stem"] if not double_marker else list(random.choice([("stem", "up"), ("stem", "down"), ("up", "down")]))

    markers: dict[tuple[int, int], int] = {}
    for idx, name in enumerate(chosen):
        markers[leaves[name]] = colors[idx]
    marker_count = len(chosen)

    grid = _render_component(interior, markers, wall_color)
    return np.rot90(grid, random.randrange(4)), marker_count


def _sample_component(wall_color: int, colors: list[int]) -> tuple[np.ndarray, int]:
    kind = random.choice(COMPONENT_TYPES)
    if kind == "line":
        return _line_component(wall_color, colors)
    if kind == "branch_single_tie":
        return _branch_component(wall_color, colors, tie=True, double_marker=False)
    if kind == "branch_single_wide":
        return _branch_component(wall_color, colors, tie=False, double_marker=False)
    return _branch_component(wall_color, colors, tie=random.choice([False, True]), double_marker=True)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        wall_color = random.choice(WALL_COLORS)
        marker_pool = [c for c in MARKER_COLORS if c != wall_color]
        random.shuffle(marker_pool)

        component_count = random.randint(1, 3)
        components: list[np.ndarray] = []
        used_colors = 0
        ok = True
        for _ in range(component_count):
            if used_colors + 2 > len(marker_pool):
                ok = False
                break
            comp, n_markers = _sample_component(wall_color, marker_pool[used_colors : used_colors + 2])
            used_colors += n_markers
            components.append(comp)
        if not ok:
            continue

        min_h = max(comp.shape[0] for comp in components) + 2
        min_w = max(comp.shape[1] for comp in components) + 2
        h = random.randint(max(12, min_h), 30)
        w = random.randint(max(12, min_w), 30)
        grid = np.zeros((h, w), dtype=int)
        occupied = np.zeros((h, w), dtype=bool)

        placed = True
        for comp in sorted(components, key=lambda arr: arr.shape[0] * arr.shape[1], reverse=True):
            ch, cw = comp.shape
            nz = comp != 0
            done = False
            for _ in range(PLACE_ATTEMPTS):
                r0 = random.randint(0, h - ch)
                c0 = random.randint(0, w - cw)
                blocked = _dilate8(occupied)
                if np.any(blocked[r0 : r0 + ch, c0 : c0 + cw] & nz):
                    continue
                patch = grid[r0 : r0 + ch, c0 : c0 + cw]
                patch[nz] = comp[nz]
                grid[r0 : r0 + ch, c0 : c0 + cw] = patch
                occupied[r0 : r0 + ch, c0 : c0 + cw] |= nz
                done = True
                break
            if not done:
                placed = False
                break

        if not placed:
            continue

        counts = Counter(int(v) for v in grid.ravel())
        if counts[0] <= max(counts.get(wall_color, 0), 0):
            continue
        return grid

    raise RuntimeError("failed to generate a valid pipe puzzle")
