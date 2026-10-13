from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
PIPE = 5
SEED_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in DIR4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _most_common_color(g: np.ndarray) -> int:
    return int(Counter(int(x) for x in g.flat).most_common(1)[0][0])


def _infer_bg_and_pipe(g: np.ndarray) -> tuple[int, int]:
    bg = _most_common_color(g)
    counts = Counter(int(x) for x in g.flat if int(x) != bg)
    assert counts, "expected at least one foreground color"
    pipe = int(counts.most_common(1)[0][0])
    return bg, pipe


def _orth_degree(mask: np.ndarray) -> np.ndarray:
    deg = np.zeros(mask.shape, dtype=int)
    deg[1:, :] += mask[:-1, :]
    deg[:-1, :] += mask[1:, :]
    deg[:, 1:] += mask[:, :-1]
    deg[:, :-1] += mask[:, 1:]
    return deg


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(int(cr), int(cc), h, w):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate seed colors through a gray pipe network and resolve crossings by local neighbor majority.

    Input structure:
    - The background is black.
    - Gray pixels form a one-cell-wide orthogonal pipe network with straight runs, bends,
      and some four-way crossings.
    - Non-gray seed pixels sit at pipe endpoints; each ordinary pipe segment is associated
      with exactly one seed when crossings are removed.

    Rule:
    - Infer the gray pipe color and treat four-way junction cells as unresolved crossings.
    - First recolor every non-crossing gray pipe segment with the unique seed color adjacent
      to that segment, while seed pixels themselves stay unchanged.
    - Then resolve each crossing by counting the colored orthogonal neighbors around it:
      the unique majority color wins; if the largest count is tied, the crossing stays gray.
    - Any gray pipe cells not uniquely attributable to a seed remain gray.

    Key insight:
    - The transformation is two-stage: remove the ambiguous crossing cells to identify which
      seed colors which pipe segment, then use only the already-colored local arms to decide
      the final crossing color.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg, pipe = _infer_bg_and_pipe(g)

    pipe_mask = g == int(pipe)
    deg = _orth_degree(pipe_mask)
    cross_mask = pipe_mask & (deg >= 3)
    segment_mask = pipe_mask & (~cross_mask)
    seed_mask = (g != int(bg)) & (g != int(pipe))

    out = g.copy()

    for comp in _components(segment_mask):
        adj_colors: set[int] = set()
        for r, c in comp:
            for nr, nc in _neighbors4(r, c, h, w):
                if bool(seed_mask[nr, nc]):
                    adj_colors.add(int(g[nr, nc]))
        if len(adj_colors) == 1:
            fill = next(iter(adj_colors))
            for r, c in comp:
                out[r, c] = int(fill)

    for r, c in np.argwhere(cross_mask):
        counts = Counter()
        for nr, nc in _neighbors4(int(r), int(c), h, w):
            col = int(out[nr, nc])
            if col != int(bg) and col != int(pipe):
                counts[int(col)] += 1
        if counts:
            top = max(counts.values())
            winners = [col for col, cnt in counts.items() if int(cnt) == int(top)]
            if len(winners) == 1:
                out[int(r), int(c)] = int(winners[0])
            else:
                out[int(r), int(c)] = int(pipe)

    return out


def _mark_blocked(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                blocked[rr, cc] = True


def _can_place(blocked: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    h, w = blocked.shape
    uniq = set(cells)
    if len(uniq) != len(cells):
        return False
    for r, c in uniq:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if bool(blocked[r, c]):
            return False
    return True


def _place_candidate(grid: np.ndarray, blocked: np.ndarray, cell_colors: dict[tuple[int, int], int]) -> bool:
    cells = [(int(r), int(c)) for r, c in cell_colors]
    if not _can_place(blocked, cells):
        return False
    for (r, c), col in cell_colors.items():
        grid[int(r), int(c)] = int(col)
    _mark_blocked(blocked, cells)
    return True


def _sample_cross_types(n_cross: int) -> list[str]:
    if n_cross == 1:
        return [random.choice(["tie", "majority"])]
    types = ["tie"] * (n_cross // 2) + ["majority"] * (n_cross // 2)
    if n_cross % 2 == 1:
        types.append(random.choice(["tie", "majority"]))
    random.shuffle(types)
    return types


def _cross_candidate(h: int, w: int, mode: str) -> dict[tuple[int, int], int] | None:
    lengths = {
        "N": random.randint(2, 6),
        "S": random.randint(2, 6),
        "W": random.randint(2, 6),
        "E": random.randint(2, 6),
    }
    r_min = lengths["N"] + 1
    r_max = h - lengths["S"] - 2
    c_min = lengths["W"] + 1
    c_max = w - lengths["E"] - 2
    if r_min > r_max or c_min > c_max:
        return None

    cr = random.randint(r_min, r_max)
    cc = random.randint(c_min, c_max)
    arms = ["N", "S", "W", "E"]

    if mode == "tie":
        a, b = random.sample(SEED_COLORS, 2)
        arm_colors = [a, a, b, b]
    else:
        pattern = random.choice(["3-1", "2-1-1"])
        if pattern == "3-1":
            a, b = random.sample(SEED_COLORS, 2)
            arm_colors = [a, a, a, b]
        else:
            a, b, c = random.sample(SEED_COLORS, 3)
            arm_colors = [a, a, b, c]
    random.shuffle(arm_colors)
    color_by_arm = dict(zip(arms, arm_colors))

    cell_colors: dict[tuple[int, int], int] = {(int(cr), int(cc)): PIPE}
    deltas = {"N": (-1, 0), "S": (1, 0), "W": (0, -1), "E": (0, 1)}
    for arm in arms:
        dr, dc = deltas[arm]
        length = int(lengths[arm])
        for step in range(1, length):
            cell_colors[(int(cr + dr * step), int(cc + dc * step))] = PIPE
        cell_colors[(int(cr + dr * length), int(cc + dc * length))] = int(color_by_arm[arm])
    return cell_colors


def _relative_polyline() -> list[tuple[int, int]] | None:
    n_seg = random.choices([1, 2, 3], weights=[3, 4, 3], k=1)[0]
    dirs: list[tuple[int, int]] = [random.choice(DIR4)]
    for _ in range(1, n_seg):
        prev = dirs[-1]
        turns = [d for d in DIR4 if d != prev and d != (-prev[0], -prev[1]) and d[0] != prev[0] and d[1] != prev[1]]
        dirs.append(random.choice(turns))
    lengths = [random.randint(2, 6) for _ in range(n_seg)]

    coords = [(0, 0)]
    used = {(0, 0)}
    cur = (0, 0)
    for direction, steps in zip(dirs, lengths):
        dr, dc = direction
        for _ in range(steps):
            nxt = (int(cur[0] + dr), int(cur[1] + dc))
            if nxt in used:
                return None
            for sdr, sdc in DIR4:
                nr, nc = int(nxt[0] + sdr), int(nxt[1] + sdc)
                if (nr, nc) in used and (nr, nc) != cur:
                    return None
            coords.append(nxt)
            used.add(nxt)
            cur = nxt

    if random.random() < 0.5:
        coords = list(reversed(coords))
    return coords


def _standalone_candidate(h: int, w: int) -> dict[tuple[int, int], int] | None:
    rel = _relative_polyline()
    if rel is None:
        return None
    rs = [r for r, _ in rel]
    cs = [c for _, c in rel]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    height = r1 - r0 + 1
    width = c1 - c0 + 1

    if height + 2 > h or width + 2 > w:
        return None

    off_r = random.randint(1 - r0, h - 2 - r1)
    off_c = random.randint(1 - c0, w - 2 - c1)
    cells = [(int(r + off_r), int(c + off_c)) for r, c in rel]

    seed = random.choice(SEED_COLORS)
    cell_colors: dict[tuple[int, int], int] = {}
    for i, (r, c) in enumerate(cells):
        cell_colors[(int(r), int(c))] = int(seed if i == 0 else PIPE)
    return cell_colors


def _scene_is_valid(grid: np.ndarray) -> bool:
    try:
        bg, pipe = _infer_bg_and_pipe(grid)
    except Exception:
        return False

    pipe_mask = grid == int(pipe)
    if not bool(pipe_mask.any()):
        return False

    deg = _orth_degree(pipe_mask)
    cross_mask = pipe_mask & (deg >= 3)
    if np.any(cross_mask & (deg != 4)):
        return False

    seed_mask = (grid != int(bg)) & (grid != int(pipe))
    for r, c in np.argwhere(seed_mask):
        pipe_neighbors = 0
        for nr, nc in _neighbors4(int(r), int(c), grid.shape[0], grid.shape[1]):
            if bool(pipe_mask[nr, nc]):
                pipe_neighbors += 1
        if pipe_neighbors != 1:
            return False

    segment_mask = pipe_mask & (~cross_mask)
    comps = _components(segment_mask)
    if not comps:
        return False
    for comp in comps:
        adj_colors: set[int] = set()
        for r, c in comp:
            for nr, nc in _neighbors4(r, c, grid.shape[0], grid.shape[1]):
                if bool(seed_mask[nr, nc]):
                    adj_colors.add(int(grid[nr, nc]))
        if len(adj_colors) != 1:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(15, 30)
    w = random.randint(15, 30)
    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    n_cross = random.randint(1, 3)
    n_standalone = random.randint(0, 4)

    for mode in _sample_cross_types(n_cross):
        placed = False
        for _ in range(120):
            cand = _cross_candidate(h, w, mode)
            if cand is None:
                continue
            if _place_candidate(grid, blocked, cand):
                placed = True
                break
        if not placed:
            return None

    for _ in range(n_standalone):
        placed = False
        for _ in range(120):
            cand = _standalone_candidate(h, w)
            if cand is None:
                continue
            if _place_candidate(grid, blocked, cand):
                placed = True
                break
        if not placed:
            return None

    if not _scene_is_valid(grid):
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pipe-propagation input")
