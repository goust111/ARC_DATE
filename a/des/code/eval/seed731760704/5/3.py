from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BLACK = 0
MAROON = 9
PATH_COLORS = (1, 2, 3, 4, 5, 6, 7, 8)
DIRS = {
    "N": (-1, 0),
    "S": (1, 0),
    "W": (0, -1),
    "E": (0, 1),
}
ORTHOGONAL = {
    "N": ("W", "E"),
    "S": ("E", "W"),
    "W": ("S", "N"),
    "E": ("N", "S"),
}
GEN_RETRIES = 300


@dataclass(frozen=True)
class Fragment:
    input_cells: tuple[tuple[int, int, int], ...]
    output_cells: tuple[tuple[int, int, int], ...]
    h: int
    w: int


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        comp: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            comp.append((rr, cc))
            for nr, nc in _neighbors4(rr, cc):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _bbox(points: set[tuple[int, int]] | list[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(points)
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    return min(rs), min(cs), max(rs), max(cs)


def _dir_from_delta(dr: int, dc: int) -> str:
    for name, (rr, cc) in DIRS.items():
        if (dr, dc) == (rr, cc):
            return name
    raise ValueError(f"unknown direction for delta {(dr, dc)}")


def _chebyshev_to_bbox(r: int, c: int, bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    dr = 0 if r0 <= r <= r1 else (r0 - r if r < r0 else r - r1)
    dc = 0 if c0 <= c <= c1 else (c0 - c if c < c0 else c - c1)
    return max(dr, dc)


def _manhattan_to_bbox(r: int, c: int, bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    dr = 0 if r0 <= r <= r1 else (r0 - r if r < r0 else r - r1)
    dc = 0 if c0 <= c <= c1 else (c0 - c if c < c0 else c - c1)
    return dr + dc


def _marker_side(square_bbox: tuple[int, int, int, int], marker: tuple[int, int]) -> str:
    r0, c0, r1, c1 = square_bbox
    mr, mc = marker
    if mr == r0 - 1 and c0 <= mc <= c1:
        return "N"
    if mr == r1 + 1 and c0 <= mc <= c1:
        return "S"
    if mc == c0 - 1 and r0 <= mr <= r1:
        return "W"
    if mc == c1 + 1 and r0 <= mr <= r1:
        return "E"
    raise ValueError("marker is not side-adjacent to square")


def _path_component(grid: np.ndarray, color: int) -> set[tuple[int, int]]:
    comps = _components(grid == color)
    assert comps, f"missing component for color {color}"
    comps.sort(key=len, reverse=True)
    path = set(comps[0])
    assert len(path) >= 2
    return path


def _square_instructions(grid: np.ndarray) -> list[dict]:
    squares: list[dict] = []
    for comp in _components(grid == MAROON):
        r0, c0, r1, c1 = _bbox(comp)
        n = r1 - r0 + 1
        if n != c1 - c0 + 1 or not (2 <= n <= 4):
            continue
        if not np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == MAROON):
            continue
        markers: set[tuple[int, int]] = set()
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                        val = int(grid[nr, nc])
                        if val not in (BLACK, MAROON):
                            markers.add((nr, nc))
        assert len(markers) == 1
        marker = next(iter(markers))
        squares.append(
            {
                "bbox": (r0, c0, r1, c1),
                "size": n,
                "marker": marker,
                "color": int(grid[marker]),
            }
        )
    return squares


def _split_path(
    path: set[tuple[int, int]], cut: tuple[int, int]
) -> tuple[list[set[tuple[int, int]]], dict[str, set[tuple[int, int]]]]:
    neighbors = [p for p in _neighbors4(*cut) if p in path]
    assert len(neighbors) == 2
    remaining = path - {cut}
    comps: list[set[tuple[int, int]]] = []
    seen: set[tuple[int, int]] = set()
    for start in neighbors:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        comp: set[tuple[int, int]] = set()
        while stack:
            cur = stack.pop()
            comp.add(cur)
            for nxt in _neighbors4(*cur):
                if nxt in remaining and nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        comps.append(comp)
    assert len(comps) == 2
    comp_by_dir: dict[str, set[tuple[int, int]]] = {}
    for comp in comps:
        touch = [p for p in neighbors if p in comp]
        assert len(touch) == 1
        nr, nc = touch[0]
        comp_by_dir[_dir_from_delta(nr - cut[0], nc - cut[1])] = comp
    return comps, comp_by_dir


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cut each target path at the closest point to its maroon square and translate
    the shorter side by the square's side length. If both sides are equally long,
    move the side indicated by the marker's side of the square.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    instructions = _square_instructions(g)
    transformed: dict[int, set[tuple[int, int]]] = {}
    markers_to_clear: set[tuple[int, int]] = set()

    for info in instructions:
        color = int(info["color"])
        n = int(info["size"])
        marker = tuple(info["marker"])
        square_bbox = tuple(info["bbox"])
        markers_to_clear.add(marker)

        path = _path_component(g, color)
        cut = min(
            path,
            key=lambda rc: (
                _chebyshev_to_bbox(rc[0], rc[1], square_bbox),
                _manhattan_to_bbox(rc[0], rc[1], square_bbox),
                rc[0],
                rc[1],
            ),
        )
        comps, comp_by_dir = _split_path(path, cut)

        if len(comps[0]) != len(comps[1]):
            moved = min(comps, key=len)
        else:
            marker_dir = _marker_side(square_bbox, marker)
            if marker_dir in comp_by_dir:
                moved = comp_by_dir[marker_dir]
            else:
                moved = min(
                    comps,
                    key=lambda comp: (
                        min(abs(r - marker[0]) + abs(c - marker[1]) for r, c in comp),
                        sorted(comp)[0],
                    ),
                )
        touch = [p for p in _neighbors4(*cut) if p in moved]
        assert len(touch) == 1
        move_dir = _dir_from_delta(touch[0][0] - cut[0], touch[0][1] - cut[1])
        dr, dc = DIRS[move_dir]

        stationary = path - moved
        shifted = {(r + n * dr, c + n * dc) for r, c in moved}
        transformed[color] = stationary | shifted

    out[g == MAROON] = BLACK
    for marker in markers_to_clear:
        out[marker] = BLACK
    for color in transformed:
        out[out == color] = BLACK
    for color, pts in transformed.items():
        for r, c in pts:
            out[r, c] = color
    return out


def _trace_arm(init_dir: str, total_len: int, turn_after: int, turn_dir: str | None) -> list[tuple[int, int]]:
    dr0, dc0 = DIRS[init_dir]
    r = 0
    c = 0
    pts: list[tuple[int, int]] = []
    for _ in range(turn_after):
        r += dr0
        c += dc0
        pts.append((r, c))
    if turn_dir is not None:
        dr1, dc1 = DIRS[turn_dir]
        for _ in range(total_len - turn_after):
            r += dr1
            c += dc1
            pts.append((r, c))
    return pts


def _random_arm(rng: random.Random, init_dir: str, total_len: int) -> list[tuple[int, int]]:
    if total_len == 1 or rng.random() < 0.45:
        return _trace_arm(init_dir, total_len, total_len, None)
    turn_after = rng.randint(1, total_len - 1)
    turn_dir = rng.choice(ORTHOGONAL[init_dir])
    return _trace_arm(init_dir, total_len, turn_after, turn_dir)


def _build_square(n: int, vertical_dir: str, horizontal_dir: str) -> tuple[set[tuple[int, int]], tuple[int, int, int, int]]:
    row_sign = 1 if vertical_dir == "N" else -1
    col_sign = 1 if horizontal_dir == "W" else -1
    rows = list(range(1, n + 1)) if row_sign > 0 else list(range(-n, 0))
    cols = list(range(1, n + 1)) if col_sign > 0 else list(range(-n, 0))
    pts = {(r, c) for r in rows for c in cols}
    return pts, _bbox(pts)


def _side_marker(
    rng: random.Random, square_bbox: tuple[int, int, int, int], marker_dir: str
) -> tuple[int, int]:
    r0, c0, r1, c1 = square_bbox
    if marker_dir in ("N", "S"):
        cols = list(range(c0, c1 + 1))
        bad = min(cols, key=lambda x: (abs(x), x))
        cols = [c for c in cols if c != bad]
        col = rng.choice(cols)
        row = r0 - 1 if marker_dir == "N" else r1 + 1
        return row, col
    rows = list(range(r0, r1 + 1))
    bad = min(rows, key=lambda x: (abs(x), x))
    rows = [r for r in rows if r != bad]
    row = rng.choice(rows)
    col = c0 - 1 if marker_dir == "W" else c1 + 1
    return row, col


def _is_simple_path(path: set[tuple[int, int]]) -> bool:
    if not path:
        return False
    degrees = {pt: 0 for pt in path}
    for r, c in path:
        deg = sum((nr, nc) in path for nr, nc in _neighbors4(r, c))
        if deg == 0 or deg > 2:
            return False
        degrees[(r, c)] = deg
    if sum(deg == 1 for deg in degrees.values()) != 2:
        return False
    stack = [next(iter(path))]
    seen = {stack[0]}
    while stack:
        cur = stack.pop()
        for nxt in _neighbors4(*cur):
            if nxt in path and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(path)


def _fragment_ok(
    vertical_arm: list[tuple[int, int]],
    horizontal_arm: list[tuple[int, int]],
    cut: tuple[int, int],
    square_bbox: tuple[int, int, int, int],
    marker: tuple[int, int],
    moved_dir: str,
    n: int,
) -> bool:
    vert_set = set(vertical_arm)
    hori_set = set(horizontal_arm)
    if len(vert_set) != len(vertical_arm) or len(hori_set) != len(horizontal_arm):
        return False
    if vert_set & hori_set:
        return False

    path = vert_set | hori_set | {cut}
    if not _is_simple_path(path):
        return False

    for r, c in path - {cut}:
        if _chebyshev_to_bbox(r, c, square_bbox) < 2:
            return False
    for r, c in path:
        if abs(r - marker[0]) + abs(c - marker[1]) <= 1:
            return False

    moved_arm = vert_set if moved_dir in ("N", "S") else hori_set
    stationary = path - moved_arm
    dr, dc = DIRS[moved_dir]
    shifted = {(r + n * dr, c + n * dc) for r, c in moved_arm}

    if shifted & stationary:
        return False
    square_cells = {
        (r, c)
        for r in range(square_bbox[0], square_bbox[2] + 1)
        for c in range(square_bbox[1], square_bbox[3] + 1)
    }
    if shifted & square_cells:
        return False
    if marker in shifted:
        return False
    for r, c in shifted:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in stationary:
                return False
    return True


def _make_fragment(rng: random.Random, color: int) -> Fragment | None:
    for _ in range(400):
        n = rng.randint(2, 4)
        vertical_dir = rng.choice(("N", "S"))
        horizontal_dir = rng.choice(("W", "E"))
        tie_case = rng.random() < 0.5

        if tie_case:
            arm_len = rng.randint(3, 5)
            len_v = arm_len
            len_h = arm_len
            moved_dir = rng.choice((vertical_dir, horizontal_dir))
        else:
            short_len = rng.randint(2, 4)
            long_len = short_len + rng.randint(1, 2)
            if rng.random() < 0.5:
                len_v, len_h = short_len, long_len
                moved_dir = vertical_dir
            else:
                len_v, len_h = long_len, short_len
                moved_dir = horizontal_dir

        marker_dir = moved_dir if tie_case else rng.choice((vertical_dir, horizontal_dir))
        square_cells, square_bbox = _build_square(n, vertical_dir, horizontal_dir)
        marker = _side_marker(rng, square_bbox, marker_dir)

        for _ in range(200):
            vertical_arm = _random_arm(rng, vertical_dir, len_v)
            horizontal_arm = _random_arm(rng, horizontal_dir, len_h)
            if not _fragment_ok(
                vertical_arm,
                horizontal_arm,
                (0, 0),
                square_bbox,
                marker,
                moved_dir,
                n,
            ):
                continue

            path = set(vertical_arm) | set(horizontal_arm) | {(0, 0)}
            moved_arm = set(vertical_arm) if moved_dir in ("N", "S") else set(horizontal_arm)
            stationary = path - moved_arm
            dr, dc = DIRS[moved_dir]
            shifted = {(r + n * dr, c + n * dc) for r, c in moved_arm}

            input_map: dict[tuple[int, int], int] = {pt: color for pt in path}
            input_map.update({pt: MAROON for pt in square_cells})
            input_map[marker] = color
            output_map: dict[tuple[int, int], int] = {pt: color for pt in stationary | shifted}

            union_pts = set(input_map) | set(output_map)
            r0, c0, r1, c1 = _bbox(union_pts)

            def norm_map(cells: dict[tuple[int, int], int]) -> tuple[tuple[int, int, int], ...]:
                return tuple(sorted((r - r0, c - c0, val) for (r, c), val in cells.items()))

            frag = Fragment(
                input_cells=norm_map(input_map),
                output_cells=norm_map(output_map),
                h=r1 - r0 + 1,
                w=c1 - c0 + 1,
            )
            return frag
    return None


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _place_fragments(
    rng: random.Random, fragments: list[Fragment], h: int, w: int
) -> list[tuple[int, int]] | None:
    order = sorted(range(len(fragments)), key=lambda i: fragments[i].h * fragments[i].w, reverse=True)
    positions: list[tuple[int, int] | None] = [None] * len(fragments)
    rects: list[tuple[int, int, int, int]] = []
    for idx in order:
        frag = fragments[idx]
        placed = False
        for _ in range(250):
            r0 = rng.randint(0, h - frag.h)
            c0 = rng.randint(0, w - frag.w)
            rect = (r0, c0, r0 + frag.h - 1, c0 + frag.w - 1)
            if any(_rects_overlap(rect, other, pad=1) for other in rects):
                continue
            positions[idx] = (r0, c0)
            rects.append(rect)
            placed = True
            break
        if not placed:
            return None
    return [pos for pos in positions if pos is not None]


def _assemble(
    fragments: list[Fragment], positions: list[tuple[int, int]], h: int, w: int
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.zeros((h, w), dtype=int)
    out = np.zeros((h, w), dtype=int)
    for frag, (r0, c0) in zip(fragments, positions, strict=False):
        for rr, cc, val in frag.input_cells:
            inp[r0 + rr, c0 + cc] = val
        for rr, cc, val in frag.output_cells:
            out[r0 + rr, c0 + cc] = val
    return inp, out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    count = rng.choices([1, 2, 3], weights=[2, 3, 2], k=1)[0]
    colors = rng.sample(PATH_COLORS, count)
    fragments: list[Fragment] = []
    for color in colors:
        frag = _make_fragment(rng, color)
        if frag is None:
            return None
        fragments.append(frag)

    min_h = max(18, max(f.h for f in fragments) + 2)
    min_w = max(18, max(f.w for f in fragments) + 2)
    for _ in range(120):
        h = rng.randint(min_h, 30)
        w = rng.randint(min_w, 30)
        positions = _place_fragments(rng, fragments, h, w)
        if positions is None:
            continue
        inp, expected = _assemble(fragments, positions, h, w)
        try:
            actual = main(inp)
        except Exception:
            continue
        if not np.array_equal(actual, expected):
            continue
        return inp
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
