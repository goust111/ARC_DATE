from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np


MOVER_COLOR = 2
TARGET_COLOR = 1
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GEN_BG_CHOICES = [0, 3, 5, 6]
GEN_RIVER_CHOICES = [4, 7, 8, 9]
GEN_ANCHOR_CHOICES = [0, 3, 4, 5, 6, 7, 8, 9]
GEN_RETRIES = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    mask = grid == int(color)
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(grid.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _find_solid_square_bbox(grid: np.ndarray, color: int) -> tuple[int, int, int, int]:
    coords = [(int(r), int(c)) for r, c in np.argwhere(grid == int(color))]
    assert coords
    r0, c0, r1, c1 = _bbox(coords)
    side_h = r1 - r0 + 1
    side_w = c1 - c0 + 1
    assert side_h == side_w
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert bool((patch == int(color)).all())
    assert len(coords) == side_h * side_w
    return r0, c0, r1, c1


def _find_anchor(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    colors = [int(c) for c in np.unique(grid) if int(c) not in {int(bg), MOVER_COLOR, TARGET_COLOR}]
    matches: list[tuple[int, tuple[int, int, int, int]]] = []
    for color in colors:
        comps = _components_of_color(grid, color)
        solid_boxes: list[tuple[int, int, int, int]] = []
        for comp in comps:
            r0, c0, r1, c1 = _bbox(comp)
            if r1 - r0 + 1 != 3 or c1 - c0 + 1 != 3:
                continue
            if len(comp) != 9:
                continue
            if bool((grid[r0 : r1 + 1, c0 : c1 + 1] == int(color)).all()):
                solid_boxes.append((r0, c0, r1, c1))
        if len(solid_boxes) == 1 and int((grid == int(color)).sum()) > 9:
            matches.append((int(color), solid_boxes[0]))
    assert len(matches) == 1
    return matches[0]


def _detect_river_color(grid: np.ndarray, bg: int, anchor_color: int) -> int:
    colors = [int(c) for c in np.unique(grid) if int(c) not in {int(bg), int(anchor_color), MOVER_COLOR, TARGET_COLOR}]
    assert len(colors) == 1
    return colors[0]


def _ray_terminal(
    coord: tuple[int, int],
    anchor: tuple[int, int, int, int],
    shape: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    r, c = coord
    r0, c0, r1, c1 = anchor
    dr = 1 if r < r0 else -1 if r > r1 else 0
    dc = 1 if c < c0 else -1 if c > c1 else 0
    assert (dr, dc) != (0, 0)
    if dr != 0 and dc != 0:
        assert abs(r - (r0 if r < r0 else r1)) == abs(c - (c0 if c < c0 else c1))
    cr, cc = r, c
    for _ in range(max(shape) * 3):
        nr, nc = cr + dr, cc + dc
        if r0 <= nr <= r1 and c0 <= nc <= c1:
            return (cr, cc), (dr, dc)
        assert _in_bounds(shape, nr, nc)
        cr, cc = nr, nc
    raise AssertionError("ray does not reach anchor")


def _line_from_terminal(
    terminal: tuple[int, int],
    step_to_anchor: tuple[int, int],
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    rr, cc = terminal
    dr, dc = step_to_anchor
    while _in_bounds(shape, rr, cc):
        out.append((rr, cc))
        rr -= dr
        cc -= dc
    return out


def _enumerate_anchor_rays(
    anchor: tuple[int, int, int, int],
    shape: tuple[int, int],
) -> list[tuple[tuple[int, int], tuple[int, int], str]]:
    h, w = shape
    r0, c0, r1, c1 = anchor
    rays: list[tuple[tuple[int, int], tuple[int, int], str]] = []
    for r in range(r0, r1 + 1):
        if c0 - 1 >= 0:
            rays.append(((r, c0 - 1), (0, 1), "cardinal"))
        if c1 + 1 < w:
            rays.append(((r, c1 + 1), (0, -1), "cardinal"))
    for c in range(c0, c1 + 1):
        if r0 - 1 >= 0:
            rays.append(((r0 - 1, c), (1, 0), "cardinal"))
        if r1 + 1 < h:
            rays.append(((r1 + 1, c), (-1, 0), "cardinal"))
    if r0 - 1 >= 0 and c0 - 1 >= 0:
        rays.append(((r0 - 1, c0 - 1), (1, 1), "diag"))
    if r0 - 1 >= 0 and c1 + 1 < w:
        rays.append(((r0 - 1, c1 + 1), (1, -1), "diag"))
    if r1 + 1 < h and c0 - 1 >= 0:
        rays.append(((r1 + 1, c0 - 1), (-1, 1), "diag"))
    if r1 + 1 < h and c1 + 1 < w:
        rays.append(((r1 + 1, c1 + 1), (-1, -1), "diag"))
    return rays


def _underlay_is_river(grid: np.ndarray, r: int, c: int, riverish: set[int]) -> bool:
    horizontal = (
        c - 1 >= 0
        and c + 1 < grid.shape[1]
        and int(grid[r, c - 1]) in riverish
        and int(grid[r, c + 1]) in riverish
    )
    vertical = (
        r - 1 >= 0
        and r + 1 < grid.shape[0]
        and int(grid[r - 1, c]) in riverish
        and int(grid[r + 1, c]) in riverish
    )
    return bool(horizontal or vertical)


def _cleanup_infer(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    anchor_color: int | None = None,
    anchor_box: tuple[int, int, int, int] | None = None,
    river_color: int | None = None,
) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = _get_bg(g) if bg is None else int(bg)
    if anchor_color is None or anchor_box is None:
        anchor_color, anchor_box = _find_anchor(g, bg)
    if river_color is None:
        river_color = _detect_river_color(g, bg, anchor_color)

    r0, c0, r1, c1 = anchor_box
    out = g.copy()
    riverish = {int(river_color), MOVER_COLOR, TARGET_COLOR}
    rays: dict[tuple[tuple[int, int], tuple[int, int]], list[tuple[int, int, int]]] = defaultdict(list)
    occupied = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}

    for rr, cc in np.argwhere(g == int(anchor_color)):
        r, c = int(rr), int(cc)
        if r0 <= r <= r1 and c0 <= c <= c1:
            continue
        base_color = int(river_color) if _underlay_is_river(g, r, c, riverish) else int(bg)
        out[r, c] = base_color
        terminal, step = _ray_terminal((r, c), anchor_box, g.shape)
        dist = max(abs(r - terminal[0]), abs(c - terminal[1]))
        rays[(terminal, step)].append((dist, r, c))

    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            out[r, c] = int(anchor_color)

    for (terminal, step), cells in rays.items():
        dr, dc = step
        tr, tc = terminal
        for i, _cell in enumerate(sorted(cells)):
            dest = (tr - i * dr, tc - i * dc)
            assert _in_bounds(g.shape, *dest)
            assert dest not in occupied
            occupied.add(dest)
            out[dest] = int(anchor_color)

    return out


def _slide_result(
    grid: np.ndarray,
    *,
    river_color: int,
) -> tuple[np.ndarray, tuple[int, int], bool, np.ndarray]:
    g = np.array(grid, dtype=int)
    mover = _find_solid_square_bbox(g, MOVER_COLOR)
    target = _find_solid_square_bbox(g, TARGET_COLOR)
    n = mover[2] - mover[0] + 1
    assert target[2] - target[0] + 1 == n and target[3] - target[1] + 1 == n
    sr, sc = mover[0], mover[1]
    tr, tc = target[0], target[1]

    traversable = np.isin(g, [int(river_color), MOVER_COLOR, TARGET_COLOR])
    h2 = g.shape[0] - n + 1
    w2 = g.shape[1] - n + 1
    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            if bool(traversable[r : r + n, c : c + n].all()):
                feasible[r, c] = True
    assert bool(feasible[sr, sc])

    dist = -np.ones((h2, w2), dtype=int)
    q = deque([(sr, sc)])
    dist[sr, sc] = 0
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h2 and 0 <= nc < w2 and feasible[nr, nc] and dist[nr, nc] < 0:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))

    reached = bool(feasible[tr, tc] and dist[tr, tc] >= 0)
    if reached:
        fr, fc = tr, tc
    else:
        best = -1
        best_score: tuple[int, int, int] | None = None
        fr, fc = sr, sc
        for r in range(h2):
            for c in range(w2):
                if dist[r, c] < 0:
                    continue
                score = (int(dist[r, c]), -abs(r - tr) - abs(c - tc), -r * g.shape[1] - c)
                if score > (best_score if best_score is not None else (-1, -10**9, -10**9)):
                    best = int(dist[r, c])
                    best_score = score
                    fr, fc = r, c
        assert best >= 0

    out = g.copy()
    out[sr : sr + n, sc : sc + n] = int(river_color)
    out[fr : fr + n, fc : fc + n] = MOVER_COLOR
    return out, (fr, fc), reached, dist


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress anchor-colored ray pixels back toward their 3x3 anchor, then slide the moving square through the river.

    Input structure:
    - The most common color is the background, and one other color forms a winding river corridor.
    - A red solid N×N mover square and a blue solid N×N target square sit on the river.
    - A fourth color has one detached 3×3 anchor block outside the river plus loose pixels on row, column, or diagonal sightlines aimed at that anchor.

    Rule:
    - First identify each loose anchor-colored ray and pack its pixels into consecutive cells pressed against the anchor along the same ray.
    - Every vacated ray cell reverts to its underlying terrain: river cells become river color again, while off-river cells become background.
    - Then slide the red N×N square through positions whose full footprint is river, replacing the blue target if reachable; otherwise stop at the farthest reachable position and refill the start with river.

    Key insight:
    - The visible river width is misleading until the anchor-color debris is compressed away, and that compression can also create a new obstruction next to the anchor.
    """
    g = np.array(grid, dtype=int)
    bg = _get_bg(g)
    anchor_color, anchor_box = _find_anchor(g, bg)
    river_color = _detect_river_color(g, bg, anchor_color)
    cleaned = _cleanup_infer(g, bg=bg, anchor_color=anchor_color, anchor_box=anchor_box, river_color=river_color)
    out, _final, _reached, _dist = _slide_result(cleaned, river_color=river_color)
    return out


def _bfs_placement(feasible: np.ndarray, sr: int, sc: int) -> np.ndarray:
    dist = -np.ones(feasible.shape, dtype=int)
    q = deque([(sr, sc)])
    dist[sr, sc] = 0
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < feasible.shape[0] and 0 <= nc < feasible.shape[1] and feasible[nr, nc] and dist[nr, nc] < 0:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))
    return dist


def _hamiltonian_walk(
    rows: int,
    cols: int,
    start: tuple[int, int],
    end: tuple[int, int],
) -> list[tuple[int, int]] | None:
    total = rows * cols
    best_path = None
    best_len = 0
    for _ in range(80):
        visited = {start}
        path = [start]
        cur = start
        while True:
            if cur == end and len(path) >= max(5, int(total * 0.6)):
                if len(path) > best_len:
                    best_len = len(path)
                    best_path = list(path)
                break
            neighbors = []
            for dr, dc in DIR4:
                nr, nc = cur[0] + dr, cur[1] + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                    neighbors.append((nr, nc))
            if not neighbors:
                break
            valid = []
            for nb in neighbors:
                if nb == end:
                    valid.append(nb)
                    continue
                test_visited = visited | {nb}
                seen = {end}
                q = deque([end])
                while q:
                    rr, cc = q.popleft()
                    for dr2, dc2 in DIR4:
                        nr2, nc2 = rr + dr2, cc + dc2
                        if 0 <= nr2 < rows and 0 <= nc2 < cols and (nr2, nc2) not in test_visited and (nr2, nc2) not in seen:
                            seen.add((nr2, nc2))
                            q.append((nr2, nc2))
                for dr2, dc2 in DIR4:
                    ar, ac = nb[0] + dr2, nb[1] + dc2
                    if 0 <= ar < rows and 0 <= ac < cols and (ar, ac) not in test_visited and (ar, ac) in seen:
                        valid.append(nb)
                        break
            if not valid:
                break
            random.shuffle(valid)
            nxt = valid[0]
            visited.add(nxt)
            path.append(nxt)
            cur = nxt
        if best_path is not None and best_len >= int(total * 0.75):
            break
    return best_path


def _compute_offsets(size: int, n: int, num: int) -> list[int] | None:
    if num == 1:
        return [1]
    first = 1
    last = size - n - 1
    if last < first:
        return None
    step = (last - first) / (num - 1)
    return [int(round(first + i * step)) for i in range(num)]


def _render_channel(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    n: int,
    offsets_r: list[int],
    offsets_c: list[int],
    river_color: int,
) -> list[tuple[int, int]]:
    pixel_pos: list[tuple[int, int]] = []
    for cr, cc in path:
        pr, pc = offsets_r[cr], offsets_c[cc]
        pixel_pos.append((pr, pc))
        grid[pr : pr + n, pc : pc + n] = int(river_color)
    for i in range(len(path) - 1):
        pr1, pc1 = pixel_pos[i]
        pr2, pc2 = pixel_pos[i + 1]
        if pr1 == pr2:
            c_lo, c_hi = min(pc1, pc2), max(pc1, pc2) + n
            grid[pr1 : pr1 + n, c_lo:c_hi] = int(river_color)
        else:
            r_lo, r_hi = min(pr1, pr2), max(pr1, pr2) + n
            grid[r_lo:r_hi, pc1 : pc1 + n] = int(river_color)
    return pixel_pos


def _gen_base_river() -> tuple[np.ndarray, int, int, int] | None:
    n = random.choice([2, 3, 4])
    h = random.randint(max(13, 4 * n + 3), 26)
    w = random.randint(max(13, 4 * n + 3), 26)
    coarse_rows = random.randint(3, min(6, max(3, (h - 2) // (n + 1))))
    coarse_cols = random.randint(3, min(6, max(3, (w - 2) // (n + 1))))

    offsets_r = _compute_offsets(h, n, coarse_rows)
    offsets_c = _compute_offsets(w, n, coarse_cols)
    if offsets_r is None or offsets_c is None:
        return None
    if any(offsets_r[i + 1] - offsets_r[i] < n for i in range(len(offsets_r) - 1)):
        return None
    if any(offsets_c[i + 1] - offsets_c[i] < n for i in range(len(offsets_c) - 1)):
        return None

    edge_cells = set()
    for c in range(coarse_cols):
        edge_cells.add((0, c))
        edge_cells.add((coarse_rows - 1, c))
    for r in range(1, coarse_rows - 1):
        edge_cells.add((r, 0))
        edge_cells.add((r, coarse_cols - 1))
    edge_cells = list(edge_cells)
    random.shuffle(edge_cells)
    start = edge_cells[0]
    far = sorted(edge_cells, key=lambda cell: -(abs(cell[0] - start[0]) + abs(cell[1] - start[1])))
    end = far[0] if far[0] != start else far[1]

    walk = _hamiltonian_walk(coarse_rows, coarse_cols, start, end)
    if walk is None or len(walk) < 5:
        return None
    if len(set(r for r, _ in walk)) < min(3, coarse_rows) or len(set(c for _, c in walk)) < min(3, coarse_cols):
        return None

    bg = random.choice(GEN_BG_CHOICES)
    river_choices = [c for c in GEN_RIVER_CHOICES if c != bg]
    river_color = random.choice(river_choices)
    grid = np.full((h, w), bg, dtype=int)
    pixel_pos = _render_channel(grid, walk, n, offsets_r, offsets_c, river_color)

    mover_r, mover_c = pixel_pos[0]
    target_r, target_c = pixel_pos[-1]
    grid[mover_r : mover_r + n, mover_c : mover_c + n] = MOVER_COLOR
    grid[target_r : target_r + n, target_c : target_c + n] = TARGET_COLOR

    if int((grid == bg).sum()) <= h * w // 2:
        return None
    out, _final, reached, _dist = _slide_result(grid, river_color=river_color)
    if not reached:
        return None
    if np.array_equal(out, grid):
        return None
    return grid, bg, river_color, n


def _ray_kind(base: np.ndarray, river_color: int, cell: tuple[int, int]) -> str:
    value = int(base[cell])
    if value == int(river_color):
        return "R"
    if value == MOVER_COLOR or value == TARGET_COLOR:
        return "X"
    return "B"


def _find_clear_option(
    base: np.ndarray,
    river_color: int,
    terminal: tuple[int, int],
    step: tuple[int, int],
    n: int,
) -> dict | None:
    cells = _line_from_terminal(terminal, step, base.shape)
    if len(cells) < 2 * n:
        return None
    if any(_ray_kind(base, river_color, cell) != "B" for cell in cells[:n]):
        return None
    idx = n
    while idx + n <= len(cells):
        if all(_ray_kind(base, river_color, cell) == "R" for cell in cells[idx : idx + n]):
            if all(_ray_kind(base, river_color, cell) == "B" for cell in cells[n:idx]):
                return {
                    "mode": "clear",
                    "terminal": terminal,
                    "step": step,
                    "originals": list(cells[idx : idx + n]),
                    "dests": list(cells[:n]),
                }
        idx += 1
    return None


def _find_block_option(
    base: np.ndarray,
    river_color: int,
    terminal: tuple[int, int],
    step: tuple[int, int],
    n: int,
) -> dict | None:
    cells = _line_from_terminal(terminal, step, base.shape)
    if len(cells) < 2 * n:
        return None
    if any(_ray_kind(base, river_color, cell) != "R" for cell in cells[:n]):
        return None
    idx = n
    while idx + n <= len(cells):
        if all(_ray_kind(base, river_color, cell) == "B" for cell in cells[idx : idx + n]):
            if all(_ray_kind(base, river_color, cell) == "R" for cell in cells[n:idx]):
                return {
                    "mode": "block",
                    "terminal": terminal,
                    "step": step,
                    "originals": list(cells[idx : idx + n]),
                    "dests": list(cells[:n]),
                }
        idx += 1
    return None


def _find_bg_diag_option(
    base: np.ndarray,
    river_color: int,
    terminal: tuple[int, int],
    step: tuple[int, int],
) -> dict | None:
    cells = _line_from_terminal(terminal, step, base.shape)
    if len(cells) < 3:
        return None
    if any(_ray_kind(base, river_color, cell) != "B" for cell in cells[:2]):
        return None
    far_cells = [cell for cell in cells[2:] if _ray_kind(base, river_color, cell) == "B"]
    if len(far_cells) < 1:
        return None
    count = random.choice([1, 2]) if len(far_cells) >= 2 else 1
    return {
        "mode": "decor",
        "terminal": terminal,
        "step": step,
        "originals": list(far_cells[:count]),
        "dests": list(cells[:count]),
    }


def _options_for_anchor(
    base: np.ndarray,
    river_color: int,
    anchor: tuple[int, int, int, int],
    n: int,
) -> dict:
    clear_opts = []
    block_opts = []
    decor_opts = []
    for terminal, step, kind in _enumerate_anchor_rays(anchor, base.shape):
        if kind == "cardinal":
            clear = _find_clear_option(base, river_color, terminal, step, n)
            if clear is not None:
                clear_opts.append(clear)
            block = _find_block_option(base, river_color, terminal, step, n)
            if block is not None:
                block_opts.append(block)
        else:
            decor = _find_bg_diag_option(base, river_color, terminal, step)
            if decor is not None:
                decor_opts.append(decor)
    return {"anchor": anchor, "clear": clear_opts, "block": block_opts, "decor": decor_opts}


def _build_input_and_cleaned(
    base: np.ndarray,
    bg: int,
    river_color: int,
    anchor_color: int,
    anchor: tuple[int, int, int, int],
    specs: list[dict],
) -> tuple[np.ndarray, np.ndarray] | None:
    inp = np.array(base, dtype=int)
    cleaned = np.array(base, dtype=int)
    r0, c0, r1, c1 = anchor
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            inp[r, c] = int(anchor_color)
            cleaned[r, c] = int(anchor_color)

    used_originals: set[tuple[int, int]] = set()
    used_dests: set[tuple[int, int]] = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}

    for spec in specs:
        for cell in spec["originals"]:
            if cell in used_originals or cell in used_dests:
                return None
            used_originals.add(cell)
            inp[cell] = int(anchor_color)
        for cell in spec["dests"]:
            if cell in used_dests:
                return None
            used_dests.add(cell)
            cleaned[cell] = int(anchor_color)

    inferred = _cleanup_infer(inp, bg=bg, anchor_color=anchor_color, anchor_box=anchor, river_color=river_color)
    if not np.array_equal(inferred, cleaned):
        return None
    return inp, cleaned


def _anchor_candidates(base: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = base.shape
    out = []
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            patch = base[r0 : r0 + 3, c0 : c0 + 3]
            if bool((patch == int(_get_bg(base))).all()):
                out.append((r0, c0, r0 + 2, c0 + 2))
    random.shuffle(out)
    return out


def _distance_at(dist: np.ndarray, pos: tuple[int, int]) -> int:
    return int(dist[pos[0], pos[1]])


def _gen_attempt() -> np.ndarray | None:
    branch = "reach" if random.random() < 0.5 else "stuck"
    base_info = _gen_base_river()
    if base_info is None:
        return None
    base, bg, river_color, n = base_info
    anchor_color_choices = [c for c in GEN_ANCHOR_CHOICES if c not in {bg, river_color, MOVER_COLOR, TARGET_COLOR}]
    anchor_color = random.choice(anchor_color_choices)

    candidates = []
    for anchor in _anchor_candidates(base):
        opts = _options_for_anchor(base, river_color, anchor, n)
        if opts["clear"] and (branch == "reach" or opts["block"]):
            candidates.append(opts)
    random.shuffle(candidates)

    for opts in candidates[:40]:
        clear_list = list(opts["clear"])
        block_list = list(opts["block"])
        decor_list = list(opts["decor"])
        random.shuffle(clear_list)
        random.shuffle(block_list)
        random.shuffle(decor_list)

        for clear_spec in clear_list[:8]:
            spec_sets = [[clear_spec]]
            if branch == "stuck":
                for block_spec in block_list[:8]:
                    if block_spec["terminal"] == clear_spec["terminal"]:
                        continue
                    spec_sets.append([clear_spec, block_spec])
            elif decor_list and random.random() < 0.6:
                decor = decor_list[0]
                if decor["terminal"] != clear_spec["terminal"]:
                    spec_sets.append([clear_spec, decor])

            for specs in spec_sets:
                built = _build_input_and_cleaned(base, bg, river_color, anchor_color, opts["anchor"], specs)
                if built is None:
                    continue
                inp, cleaned = built
                raw_out, raw_final, raw_reached, raw_dist = _slide_result(inp, river_color=river_color)
                out, final, reached, dist = _slide_result(cleaned, river_color=river_color)
                if raw_reached:
                    continue
                start = (_find_solid_square_bbox(inp, MOVER_COLOR)[0], _find_solid_square_bbox(inp, MOVER_COLOR)[1])
                target = (_find_solid_square_bbox(inp, TARGET_COLOR)[0], _find_solid_square_bbox(inp, TARGET_COLOR)[1])
                if final == start:
                    continue
                if np.array_equal(raw_out, out):
                    continue
                if _distance_at(dist, final) <= _distance_at(raw_dist, raw_final):
                    continue
                if branch == "reach":
                    if not reached:
                        continue
                    if final != target:
                        continue
                else:
                    if reached:
                        continue
                    if not any(spec["mode"] == "block" and any(base[cell] == river_color for cell in spec["dests"]) for spec in specs):
                        continue
                if not np.array_equal(main(inp), out):
                    continue
                return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid river-anchor puzzle")
