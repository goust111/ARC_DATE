from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists():
    sys.path.append(str(_ARC_UTILS_DIR))

try:
    from arc_utils import components_mask, get_bg, neighbors4
except Exception:
    def get_bg(grid: np.ndarray) -> int:
        return int(Counter(map(int, np.asarray(grid, dtype=int).ravel())).most_common(1)[0][0])

    def neighbors4(r: int, c: int):
        yield int(r) - 1, int(c)
        yield int(r) + 1, int(c)
        yield int(r), int(c) - 1
        yield int(r), int(c) + 1

    def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
        assert int(connectivity) == 4
        m = np.asarray(mask, dtype=bool)
        seen = np.zeros_like(m, dtype=bool)
        out: list[np.ndarray] = []
        h, w = m.shape
        for r in range(h):
            for c in range(w):
                if not m[r, c] or seen[r, c]:
                    continue
                stack = [(r, c)]
                seen[r, c] = True
                coords = []
                while stack:
                    rr, cc = stack.pop()
                    coords.append((rr, cc))
                    for nr, nc in neighbors4(rr, cc):
                        if 0 <= nr < h and 0 <= nc < w and m[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
                out.append(np.array(coords, dtype=int))
        return out


Coord = tuple[int, int]
GEN_MAX_ATTEMPTS = 500
GEN_COLOR_POOL = tuple(range(10))


def _coords_to_mask(coords: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _path_degrees(mask: np.ndarray, coords: np.ndarray) -> dict[Coord, int]:
    degrees: dict[Coord, int] = {}
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        deg = 0
        for nr, nc in neighbors4(rr, cc):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and bool(mask[nr, nc]):
                deg += 1
        degrees[(rr, cc)] = deg
    return degrees


def _simple_path_component(grid: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray]:
    candidates = []
    for coords in components_mask(grid != int(bg), connectivity=4):
        if int(coords.shape[0]) < 4:
            continue
        mask = _coords_to_mask(coords, grid.shape)
        degrees = _path_degrees(mask, coords)
        deg_values = list(degrees.values())
        if deg_values.count(1) != 2:
            continue
        if any(deg not in (1, 2) for deg in deg_values):
            continue
        candidates.append((int(coords.shape[0]), coords, mask))
    assert candidates, "expected one snake-path foreground component"
    _size, coords, mask = max(candidates, key=lambda item: item[0])
    return coords, mask


def _ordered_path(mask: np.ndarray, coords: np.ndarray) -> list[Coord]:
    degrees = _path_degrees(mask, coords)
    endpoints = [coord for coord, deg in degrees.items() if deg == 1]
    assert len(endpoints) == 2, "snake path must have two endpoints"

    ordered = [endpoints[0]]
    prev: Coord | None = None
    cur = endpoints[0]
    while True:
        nxts = []
        for nr, nc in neighbors4(*cur):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and bool(mask[nr, nc]):
                if prev is None or (nr, nc) != prev:
                    nxts.append((int(nr), int(nc)))
        if not nxts:
            break
        assert len(nxts) == 1, "snake path traversal must be unambiguous"
        prev, cur = cur, nxts[0]
        ordered.append(cur)

    assert len(ordered) == int(coords.shape[0]), "path traversal missed cells"
    return ordered


def _path_specials(grid: np.ndarray, path: list[Coord]) -> tuple[int, int, tuple[int, int], int]:
    counts = Counter(int(grid[r, c]) for r, c in path)
    marker_colors = [color for color, cnt in counts.items() if int(cnt) == 2]
    wall_colors = [color for color, cnt in counts.items() if int(cnt) == 1]
    assert len(marker_colors) == 1, "expected one marker color appearing twice on the path"
    assert len(wall_colors) == 1, "expected one wall color appearing once on the path"

    marker_color = int(marker_colors[0])
    wall_color = int(wall_colors[0])
    marker_idx = [i for i, (r, c) in enumerate(path) if int(grid[r, c]) == marker_color]
    wall_idx = [i for i, (r, c) in enumerate(path) if int(grid[r, c]) == wall_color]
    assert len(marker_idx) == 2 and len(wall_idx) == 1
    return marker_color, wall_color, (int(marker_idx[0]), int(marker_idx[1])), int(wall_idx[0])


def _periodic_fill_indices(path_len: int, marker_pair: tuple[int, int], wall_idx: int) -> set[int]:
    lo, hi = sorted(map(int, marker_pair))
    k = int(hi - lo)
    assert k >= 1

    keep: set[int] = set()
    aligned = (int(wall_idx - lo) % k) == 0
    for idx in range(path_len):
        if (int(idx - lo) % k) != 0:
            continue
        if aligned:
            if wall_idx < lo and idx <= wall_idx:
                continue
            if wall_idx > hi and idx >= wall_idx:
                continue
        if idx == wall_idx:
            continue
        keep.add(int(idx))
    return keep


def _render_output(
    grid: np.ndarray,
    path: list[Coord],
    marker_color: int,
    marker_pair: tuple[int, int],
    wall_idx: int,
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for idx in _periodic_fill_indices(len(path), marker_pair, wall_idx):
        r, c = path[idx]
        out[int(r), int(c)] = int(marker_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill a snake path with a path-length period set by two marker dots, with an aligned third dot acting as a one-sided wall.

    Input structure:
    - A background contains one thin 4-connected snake path with two endpoints and several bends.
    - Two same-colored marker pixels lie on that path, and one third-color pixel also lies on the path.
    - Extra off-path pixels may share the marker color, but they are isolated distractors and do not belong to the snake component.

    Rule:
    - Linearize the snake by walking from one endpoint to the other and measure arc-length distance along that ordered path.
    - Let k be the path distance between the two on-path marker pixels; recolor every path cell whose arc-length from either marker is a multiple of k to the marker color.
    - If the third-color path pixel falls on that same modulo-k phase, it blocks propagation only on its side: the wall cell itself stays unchanged and no later beat past it on that side is painted.
    - Leave the wall pixel, the off-path distractors, and every non-beat background/path cell unchanged.

    Key insight:
    - The period lives in the path's intrinsic 1D arc-length coordinate, not in straight row/column spacing, so both the fill and the wall test depend on positions along the ordered snake rather than geometric distance in the image plane.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    coords, mask = _simple_path_component(g, bg)
    path = _ordered_path(mask, coords)
    marker_color, _wall_color, marker_pair, wall_idx = _path_specials(g, path)
    return _render_output(g, path, marker_color, marker_pair, wall_idx)


def _valid_coarse_step(shape: tuple[int, int], nxt: Coord, prev: Coord, occupied: set[Coord]) -> bool:
    h, w = shape
    r, c = nxt
    if not (0 <= r < h and 0 <= c < w):
        return False
    if nxt in occupied:
        return False
    for nr, nc in neighbors4(r, c):
        if (int(nr), int(nc)) in occupied and (int(nr), int(nc)) != prev:
            return False
    return True


def _coarse_path(shape: tuple[int, int], target_nodes: int, min_bends: int) -> list[Coord] | None:
    h, w = shape
    start = (random.randrange(h), random.randrange(w))
    path = [start]
    occupied = {start}

    def dfs(last_dir: Coord | None, bends: int) -> bool:
        if len(path) == target_nodes:
            return bends >= min_bends
        cur = path[-1]
        candidates = []
        for nr, nc in neighbors4(*cur):
            nxt = (int(nr), int(nc))
            if not _valid_coarse_step(shape, nxt, cur, occupied):
                continue
            step = (int(nxt[0] - cur[0]), int(nxt[1] - cur[1]))
            turn = int(last_dir is not None and step != last_dir)
            urgency = int(bends + turn < min_bends)
            candidates.append((urgency, random.random(), nxt, step, turn))
        candidates.sort(key=lambda item: (item[0], item[1]))
        for _urgency, _noise, nxt, step, turn in candidates:
            next_bends = bends + turn
            remaining = target_nodes - len(path) - 1
            if next_bends + remaining < min_bends:
                continue
            occupied.add(nxt)
            path.append(nxt)
            if dfs(step, next_bends):
                return True
            path.pop()
            occupied.remove(nxt)
        return False

    if dfs(None, 0):
        return path
    return None


def _expand_path(coarse: list[Coord], top: int, left: int) -> list[Coord]:
    fine = [(int(top + 2 * coarse[0][0]), int(left + 2 * coarse[0][1]))]
    for (r0, c0), (r1, c1) in zip(coarse, coarse[1:]):
        fine.append((int(top + r0 + r1), int(left + c0 + c1)))
        fine.append((int(top + 2 * r1), int(left + 2 * c1)))
    return fine


def _choose_periodic_spec(path_len: int, want_aligned_wall: bool) -> tuple[tuple[int, int], int] | None:
    k_values = list(range(2, min(6, path_len - 1) + 1))
    random.shuffle(k_values)
    for k in k_values:
        residues = list(range(k))
        random.shuffle(residues)
        for residue in residues:
            beats = list(range(residue, path_len, k))
            if want_aligned_wall and len(beats) < 5:
                continue
            if not want_aligned_wall and len(beats) < 4:
                continue

            if want_aligned_wall:
                options = []
                for j in range(len(beats) - 1):
                    if j >= 2:
                        options.append(((beats[j], beats[j + 1]), beats[j - 1]))
                    if j + 3 < len(beats):
                        options.append(((beats[j], beats[j + 1]), beats[j + 2]))
                if options:
                    return random.choice(options)
                continue

            pairs = []
            for j in range(len(beats) - 1):
                marker_pair = (beats[j], beats[j + 1])
                wall_choices = [idx for idx in range(path_len) if idx not in beats]
                if wall_choices:
                    pairs.append((marker_pair, random.choice(wall_choices)))
            if pairs:
                return random.choice(pairs)
    return None


def _isolated_bg_cells(grid: np.ndarray, forbidden: set[Coord]) -> list[Coord]:
    cells = []
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != int(get_bg(grid)):
                continue
            pos = (r, c)
            if pos in forbidden:
                continue
            ok = True
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (int(nr), int(nc)) in forbidden:
                    ok = False
                    break
            if ok:
                cells.append(pos)
    return cells


def _semantic_expected(
    inp: np.ndarray,
    ordered_path: list[Coord],
    marker_color: int,
    marker_pair: tuple[int, int],
    wall_idx: int,
) -> np.ndarray:
    return _render_output(inp, ordered_path, marker_color, marker_pair, wall_idx)


def _gen_attempt() -> np.ndarray | None:
    bg, path_color, marker_color, wall_color = random.sample(GEN_COLOR_POOL, 4)
    coarse_h = random.randint(4, 7)
    coarse_w = random.randint(4, 7)
    node_count = random.randint(5, min(11, coarse_h * coarse_w))
    min_bends = random.randint(2, min(5, node_count - 2))

    coarse = None
    for _ in range(80):
        coarse = _coarse_path((coarse_h, coarse_w), node_count, min_bends)
        if coarse is not None:
            break
    if coarse is None:
        return None

    base_h = 2 * coarse_h - 1
    base_w = 2 * coarse_w - 1
    top = random.randint(1, 4)
    left = random.randint(1, 4)
    bottom = random.randint(1, 4)
    right = random.randint(1, 4)
    height = base_h + top + bottom
    width = base_w + left + right
    if max(height, width) > 30:
        return None

    ordered_path = _expand_path(coarse, top, left)
    if len(ordered_path) < 9:
        return None

    want_aligned_wall = bool(random.getrandbits(1))
    spec = _choose_periodic_spec(len(ordered_path), want_aligned_wall)
    if spec is None:
        return None
    marker_pair, wall_idx = spec

    inp = np.full((height, width), int(bg), dtype=int)
    for r, c in ordered_path:
        inp[int(r), int(c)] = int(path_color)

    m0, m1 = marker_pair
    mr0, mc0 = ordered_path[m0]
    mr1, mc1 = ordered_path[m1]
    wr, wc = ordered_path[wall_idx]
    inp[int(mr0), int(mc0)] = int(marker_color)
    inp[int(mr1), int(mc1)] = int(marker_color)
    inp[int(wr), int(wc)] = int(wall_color)

    path_set = {tuple(cell) for cell in ordered_path}
    distractor_count = random.randint(1, 4)
    occupied = set(path_set)
    choices = _isolated_bg_cells(inp, occupied)
    random.shuffle(choices)
    if len(choices) < distractor_count:
        return None
    for r, c in choices[:distractor_count]:
        inp[int(r), int(c)] = int(marker_color)
        occupied.add((int(r), int(c)))

    expected = _semantic_expected(inp, ordered_path, marker_color, marker_pair, wall_idx)
    if np.array_equal(inp, expected):
        return None

    aligned = ((wall_idx - min(marker_pair)) % abs(marker_pair[1] - marker_pair[0])) == 0
    if want_aligned_wall != aligned:
        return None

    if want_aligned_wall:
        blocked_side_changed = False
        for idx in range(len(ordered_path)):
            if (idx - min(marker_pair)) % abs(marker_pair[1] - marker_pair[0]) != 0:
                continue
            if wall_idx < min(marker_pair) and idx < wall_idx:
                r, c = ordered_path[idx]
                if int(expected[r, c]) == int(path_color):
                    blocked_side_changed = True
                    break
            if wall_idx > max(marker_pair) and idx > wall_idx:
                r, c = ordered_path[idx]
                if int(expected[r, c]) == int(path_color):
                    blocked_side_changed = True
                    break
        if not blocked_side_changed:
            return None

    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid snake-period input")
