import random
from collections import Counter, defaultdict, deque
from dataclasses import dataclass

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
GEN_ATTEMPTS = 500

GRID_H_RANGE = (20, 31)
GRID_W_RANGE = (20, 31)
OUTER_SIZE_RANGE = (3, 7)  # exclusive upper bound
WIRE_LEN_RANGE = (2, 6)    # exclusive upper bound
CORNER_GAP_RANGE = (2, 7)  # exclusive upper bound
PAIR_COUNT_CHOICES = (1, 2)
SINGLE_COUNT_CHOICES = (0, 1, 2)


@dataclass(frozen=True)
class Fragment:
    shell: int
    chip: int
    outer_r0: int
    outer_c0: int
    outer_h: int
    outer_w: int
    wire_dir: str
    wire_len: int
    attach: int

    @property
    def outer_r1(self) -> int:
        return self.outer_r0 + self.outer_h - 1

    @property
    def outer_c1(self) -> int:
        return self.outer_c0 + self.outer_w - 1

    @property
    def area(self) -> int:
        return self.outer_h * self.outer_w

    @property
    def orientation(self) -> str:
        return "h" if self.wire_dir in ("left", "right") else "v"

    @property
    def chip_bbox(self) -> tuple[int, int, int, int]:
        return (
            self.outer_r0 + 1,
            self.outer_c0 + 1,
            self.outer_r1 - 1,
            self.outer_c1 - 1,
        )

    @property
    def endpoint(self) -> tuple[int, int]:
        if self.wire_dir == "right":
            return self.outer_r0 + self.attach, self.outer_c1 + self.wire_len
        if self.wire_dir == "left":
            return self.outer_r0 + self.attach, self.outer_c0 - self.wire_len
        if self.wire_dir == "down":
            return self.outer_r1 + self.wire_len, self.outer_c0 + self.attach
        return self.outer_r0 - self.wire_len, self.outer_c0 + self.attach

    def border_cells(self) -> list[tuple[int, int]]:
        cells: list[tuple[int, int]] = []
        for c in range(self.outer_c0, self.outer_c1 + 1):
            cells.append((self.outer_r0, c))
            cells.append((self.outer_r1, c))
        for r in range(self.outer_r0 + 1, self.outer_r1):
            cells.append((r, self.outer_c0))
            cells.append((r, self.outer_c1))
        return cells

    def chip_cells(self) -> list[tuple[int, int]]:
        r0, c0, r1, c1 = self.chip_bbox
        return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]

    def wire_cells(self) -> list[tuple[int, int]]:
        if self.wire_dir == "right":
            r = self.outer_r0 + self.attach
            return [(r, c) for c in range(self.outer_c1 + 1, self.outer_c1 + self.wire_len + 1)]
        if self.wire_dir == "left":
            r = self.outer_r0 + self.attach
            return [(r, c) for c in range(self.outer_c0 - self.wire_len, self.outer_c0)]
        if self.wire_dir == "down":
            c = self.outer_c0 + self.attach
            return [(r, c) for r in range(self.outer_r1 + 1, self.outer_r1 + self.wire_len + 1)]
        c = self.outer_c0 + self.attach
        return [(r, c) for r in range(self.outer_r0 - self.wire_len, self.outer_r0)]

    def input_cells(self) -> set[tuple[int, int]]:
        return set(self.border_cells()) | set(self.chip_cells()) | set(self.wire_cells())

    def paint_input(self, grid: np.ndarray) -> None:
        for r, c in self.border_cells():
            grid[r, c] = self.shell
        for r, c in self.chip_cells():
            grid[r, c] = self.chip
        for r, c in self.wire_cells():
            grid[r, c] = self.shell


def _bg_color(grid: np.ndarray) -> int:
    return int(Counter(int(v) for v in grid.ravel()).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) != color:
                continue
            visited[r, c] = True
            comp: list[tuple[int, int]] = []
            q = deque([(r, c)])
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _component_from(grid: np.ndarray, start: tuple[int, int], color: int) -> set[tuple[int, int]]:
    h, w = grid.shape
    sr, sc = start
    if not (0 <= sr < h and 0 <= sc < w) or int(grid[sr, sc]) != color:
        return set()
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and int(grid[nr, nc]) == color:
                seen.add((nr, nc))
                q.append((nr, nc))
    return seen


def _bbox(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _rectangle_border_set(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _parse_wire(
    outside: set[tuple[int, int]],
    outer_r0: int,
    outer_c0: int,
    outer_r1: int,
    outer_c1: int,
) -> tuple[str, int, int] | None:
    if len(outside) < 2 or len(outside) > 5:
        return None

    rows = sorted({r for r, _ in outside})
    cols = sorted({c for _, c in outside})

    if all(r < outer_r0 for r, _ in outside):
        if len(cols) != 1:
            return None
        col = cols[0]
        if not (outer_c0 <= col <= outer_c1):
            return None
        ordered_rows = sorted(r for r, _ in outside)
        if ordered_rows != list(range(outer_r0 - len(outside), outer_r0)):
            return None
        return "up", len(outside), col - outer_c0

    if all(r > outer_r1 for r, _ in outside):
        if len(cols) != 1:
            return None
        col = cols[0]
        if not (outer_c0 <= col <= outer_c1):
            return None
        ordered_rows = sorted(r for r, _ in outside)
        if ordered_rows != list(range(outer_r1 + 1, outer_r1 + len(outside) + 1)):
            return None
        return "down", len(outside), col - outer_c0

    if all(c < outer_c0 for _, c in outside):
        if len(rows) != 1:
            return None
        row = rows[0]
        if not (outer_r0 <= row <= outer_r1):
            return None
        ordered_cols = sorted(c for _, c in outside)
        if ordered_cols != list(range(outer_c0 - len(outside), outer_c0)):
            return None
        return "left", len(outside), row - outer_r0

    if all(c > outer_c1 for _, c in outside):
        if len(rows) != 1:
            return None
        row = rows[0]
        if not (outer_r0 <= row <= outer_r1):
            return None
        ordered_cols = sorted(c for _, c in outside)
        if ordered_cols != list(range(outer_c1 + 1, outer_c1 + len(outside) + 1)):
            return None
        return "right", len(outside), row - outer_r0

    return None


def _find_fragments(grid: np.ndarray) -> list[Fragment]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)
    fragments: list[Fragment] = []
    seen_keys: set[tuple[int, int, int, int, int]] = set()

    for chip in sorted({int(v) for v in np.unique(g)} - {bg}):
        for comp in _components_of_color(g, chip):
            r0, c0, r1, c1 = _bbox(comp)
            comp_area = len(comp)
            if comp_area != (r1 - r0 + 1) * (c1 - c0 + 1):
                continue
            if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
                continue

            outer_r0 = r0 - 1
            outer_c0 = c0 - 1
            outer_r1 = r1 + 1
            outer_c1 = c1 + 1
            border = _rectangle_border_set(outer_r0, outer_c0, outer_r1, outer_c1)
            shell_colors = {int(g[r, c]) for r, c in border}
            if len(shell_colors) != 1:
                continue
            shell = next(iter(shell_colors))
            if shell in (bg, chip):
                continue

            shell_comp = _component_from(g, (outer_r0, outer_c0), shell)
            if not border.issubset(shell_comp):
                continue

            outside = shell_comp - border
            parsed = _parse_wire(outside, outer_r0, outer_c0, outer_r1, outer_c1)
            if parsed is None:
                continue
            wire_dir, wire_len, attach = parsed
            key = (chip, outer_r0, outer_c0, outer_r1, outer_c1)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            fragments.append(
                Fragment(
                    shell=shell,
                    chip=chip,
                    outer_r0=outer_r0,
                    outer_c0=outer_c0,
                    outer_h=outer_r1 - outer_r0 + 1,
                    outer_w=outer_c1 - outer_c0 + 1,
                    wire_dir=wire_dir,
                    wire_len=wire_len,
                    attach=attach,
                )
            )

    return fragments


def _pair_fragments(fragments: list[Fragment]) -> list[tuple[Fragment, Fragment]]:
    by_shell: dict[int, list[Fragment]] = defaultdict(list)
    for frag in fragments:
        by_shell[frag.shell].append(frag)

    pairs: list[tuple[Fragment, Fragment]] = []
    for group in by_shell.values():
        if len(group) != 2:
            continue
        a, b = group
        if a.orientation == b.orientation:
            continue
        pairs.append((a, b))
    return pairs


def _line_cells(p0: tuple[int, int], p1: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = p0
    r1, c1 = p1
    if r0 == r1:
        start, end = sorted((c0, c1))
        return [(r0, c) for c in range(start, end + 1)]
    if c0 == c1:
        start, end = sorted((r0, r1))
        return [(r, c0) for r in range(start, end + 1)]
    raise AssertionError("expected orthogonal line")


def _marker_cells(
    corner: tuple[int, int],
    h_endpoint: tuple[int, int],
    v_endpoint: tuple[int, int],
) -> list[tuple[int, int]]:
    r, c = corner
    rows = [r - 1, r] if v_endpoint[0] < r else [r, r + 1]
    cols = [c - 1, c] if h_endpoint[1] < c else [c, c + 1]
    return [(rr, cc) for rr in rows for cc in cols]


def _marker_color(a: Fragment, b: Fragment) -> int:
    if a.area != b.area:
        return a.chip if a.area < b.area else b.chip
    if a.outer_r0 != b.outer_r0:
        return a.chip if a.outer_r0 < b.outer_r0 else b.chip
    return a.chip if a.outer_c0 <= b.outer_c0 else b.chip


def _apply_pair(out: np.ndarray, a: Fragment, b: Fragment) -> None:
    if a.orientation == "h":
        h_frag, v_frag = a, b
    else:
        h_frag, v_frag = b, a

    corner = (h_frag.endpoint[0], v_frag.endpoint[1])
    for r, c in _line_cells(h_frag.endpoint, corner):
        out[r, c] = h_frag.shell
    for r, c in _line_cells(v_frag.endpoint, corner):
        out[r, c] = v_frag.shell

    color = _marker_color(a, b)
    for r, c in _marker_cells(corner, h_frag.endpoint, v_frag.endpoint):
        out[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-shell fragments whose wires point along perpendicular axes, extend
    their wires until the two extension rays meet, and place a 2x2 corner mark
    colored by the smaller fragment's chip color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for a, b in _pair_fragments(_find_fragments(g)):
        _apply_pair(out, a, b)
    return out


def _assign_unique_chip_colors(shell_by_fragment: list[int]) -> list[int] | None:
    fragments = list(range(len(shell_by_fragment)))
    for _ in range(200):
        chips = random.sample(PALETTE, len(fragments))
        if all(chips[i] != shell_by_fragment[i] for i in fragments):
            return chips
    return None


def _buffered(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _choose_outer_dims(equal_area: bool) -> tuple[tuple[int, int], tuple[int, int]]:
    if equal_area:
        h = random.randrange(*OUTER_SIZE_RANGE)
        w = random.randrange(*OUTER_SIZE_RANGE)
        if random.random() < 0.5:
            return (h, w), (w, h)
        return (h, w), (h, w)

    while True:
        a = (random.randrange(*OUTER_SIZE_RANGE), random.randrange(*OUTER_SIZE_RANGE))
        b = (random.randrange(*OUTER_SIZE_RANGE), random.randrange(*OUTER_SIZE_RANGE))
        if a[0] * a[1] != b[0] * b[1]:
            return a, b


def _build_pair_candidate(
    h: int,
    w: int,
    shell: int,
    chips: tuple[int, int],
    equal_area: bool,
    occupied_buffer: set[tuple[int, int]],
) -> tuple[tuple[Fragment, Fragment], set[tuple[int, int]]] | None:
    for _ in range(300):
        (h_h, h_w), (v_h, v_w) = _choose_outer_dims(equal_area)
        h_dir = random.choice(("left", "right"))
        v_dir = random.choice(("up", "down"))
        h_attach = random.randint(1, h_h - 2)
        v_attach = random.randint(1, v_w - 2)
        h_wire = random.randrange(*WIRE_LEN_RANGE)
        v_wire = random.randrange(*WIRE_LEN_RANGE)
        h_gap = random.randrange(*CORNER_GAP_RANGE)
        v_gap = random.randrange(*CORNER_GAP_RANGE)

        corner_r = random.randint(2, h - 3)
        corner_c = random.randint(2, w - 3)

        if h_dir == "right":
            h_r0 = corner_r - h_attach
            h_c0 = corner_c - h_gap - h_wire - (h_w - 1)
        else:
            h_r0 = corner_r - h_attach
            h_c0 = corner_c + h_gap + h_wire

        if v_dir == "down":
            v_r0 = corner_r - v_gap - v_wire - (v_h - 1)
            v_c0 = corner_c - v_attach
        else:
            v_r0 = corner_r + v_gap + v_wire
            v_c0 = corner_c - v_attach

        frag_h = Fragment(shell, chips[0], h_r0, h_c0, h_h, h_w, h_dir, h_wire, h_attach)
        frag_v = Fragment(shell, chips[1], v_r0, v_c0, v_h, v_w, v_dir, v_wire, v_attach)

        if equal_area and frag_h.outer_r0 == frag_v.outer_r0:
            continue

        input_cells = frag_h.input_cells() | frag_v.input_cells()
        if len(input_cells) != len(frag_h.input_cells()) + len(frag_v.input_cells()):
            continue
        if any(not (0 <= r < h and 0 <= c < w) for r, c in input_cells):
            continue

        pair_cells = set(input_cells)
        corner = (frag_h.endpoint[0], frag_v.endpoint[1])
        pair_cells.update(_line_cells(frag_h.endpoint, corner))
        pair_cells.update(_line_cells(frag_v.endpoint, corner))
        marker = _marker_cells(corner, frag_h.endpoint, frag_v.endpoint)
        if any(not (0 <= r < h and 0 <= c < w) for r, c in marker):
            continue
        pair_cells.update(marker)

        if pair_cells & occupied_buffer:
            continue
        return (frag_h, frag_v), pair_cells
    return None


def _build_single_candidate(
    h: int,
    w: int,
    shell: int,
    chip: int,
    occupied_buffer: set[tuple[int, int]],
) -> tuple[Fragment, set[tuple[int, int]]] | None:
    for _ in range(250):
        outer_h = random.randrange(*OUTER_SIZE_RANGE)
        outer_w = random.randrange(*OUTER_SIZE_RANGE)
        wire_dir = random.choice(("up", "down", "left", "right"))
        wire_len = random.randrange(*WIRE_LEN_RANGE)
        attach = random.randint(1, outer_h - 2) if wire_dir in ("left", "right") else random.randint(1, outer_w - 2)

        if wire_dir == "up":
            r0 = random.randint(wire_len, h - outer_h - 1)
            c0 = random.randint(1, w - outer_w - 1)
        elif wire_dir == "down":
            r0 = random.randint(1, h - outer_h - wire_len - 1)
            c0 = random.randint(1, w - outer_w - 1)
        elif wire_dir == "left":
            r0 = random.randint(1, h - outer_h - 1)
            c0 = random.randint(wire_len, w - outer_w - 1)
        else:
            r0 = random.randint(1, h - outer_h - 1)
            c0 = random.randint(1, w - outer_w - wire_len - 1)

        frag = Fragment(shell, chip, r0, c0, outer_h, outer_w, wire_dir, wire_len, attach)
        cells = frag.input_cells()
        if any(not (0 <= r < h and 0 <= c < w) for r, c in cells):
            continue
        if cells & occupied_buffer:
            continue
        return frag, cells
    return None


def _render_input(h: int, w: int, fragments: list[Fragment]) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for frag in fragments:
        frag.paint_input(grid)
    return grid


def _render_expected(inp: np.ndarray, fragments: list[Fragment]) -> np.ndarray:
    out = inp.copy()
    for a, b in _pair_fragments(fragments):
        _apply_pair(out, a, b)
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    num_pairs = random.choice(PAIR_COUNT_CHOICES)
    num_singles = random.choice(SINGLE_COUNT_CHOICES)
    total_fragments = num_pairs * 2 + num_singles
    total_groups = num_pairs + num_singles

    shell_colors = random.sample(PALETTE, total_groups)
    shell_by_fragment: list[int] = []
    for idx in range(num_pairs):
        shell_by_fragment.extend([shell_colors[idx], shell_colors[idx]])
    for idx in range(num_singles):
        shell_by_fragment.append(shell_colors[num_pairs + idx])

    chips = _assign_unique_chip_colors(shell_by_fragment)
    if chips is None:
        return None

    fragments: list[Fragment] = []
    pair_flags = [random.random() < 0.5 for _ in range(num_pairs)]
    occupied_cells: set[tuple[int, int]] = set()
    occupied_buffer: set[tuple[int, int]] = set()

    chip_idx = 0
    for pair_idx in range(num_pairs):
        candidate = _build_pair_candidate(
            h=h,
            w=w,
            shell=shell_colors[pair_idx],
            chips=(chips[chip_idx], chips[chip_idx + 1]),
            equal_area=pair_flags[pair_idx],
            occupied_buffer=occupied_buffer,
        )
        chip_idx += 2
        if candidate is None:
            return None
        (frag_a, frag_b), pair_cells = candidate
        fragments.extend([frag_a, frag_b])
        occupied_cells |= pair_cells
        occupied_buffer = _buffered(occupied_cells, h, w)

    for single_idx in range(num_singles):
        candidate = _build_single_candidate(
            h=h,
            w=w,
            shell=shell_colors[num_pairs + single_idx],
            chip=chips[chip_idx],
            occupied_buffer=occupied_buffer,
        )
        chip_idx += 1
        if candidate is None:
            return None
        frag, cells = candidate
        fragments.append(frag)
        occupied_cells |= cells
        occupied_buffer = _buffered(occupied_cells, h, w)

    if len(fragments) != total_fragments:
        return None

    inp = _render_input(h, w, fragments)
    expected = _render_expected(inp, fragments)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
