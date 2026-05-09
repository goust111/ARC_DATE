# Variant of 1818057f
# Dimension: D6 — '+'字的嵌入约束 (meta)
# Change: '+'字总是嵌入在更大的前景连通分量中（size 6-61），从不以孤立5像素十字出现... → 只重着色孤立的'+'，不处理嵌入的

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import get_bg


# Design analysis
# a) Concrete behavior and trace:
#    - Input still uses one bg color and one fg color. The visual distinction is now:
#      some fg components are exact 5-cell '+' crosses, while others contain the same
#      local '+' center but have extra fg cells attached along one or more arms.
#    - Example trace:
#        exact component {(r,c),(r+-1,c),(r,c+-1)} size=5:
#          original solver: recolors it because the center has 4 orthogonal fg neighbors
#          variant solver: recolors it because the component size is exactly 5
#        embedded component = exact plus plus one extra cell at (r,c+2), size=6:
#          original solver: also recolors the central 5 cells
#          variant solver: skips it because the plus is not isolated
#      The new behavior therefore changes the filtering stage, not the plus detector itself.
# b) Complexity calibration:
#    - If the generator emitted only isolated pluses, the variant would look identical to
#      the original task. To avoid that degeneration, every sample contains both isolated
#      and embedded pluses, plus non-plus distractor components.
#    - Components are separated by a 4-neighbor moat so the size cue is visually legible.
# c) Old implicit assumptions that break:
#    - The original `if all(neighbors are fg): recolor` assumes every local '+' motif
#      should be highlighted. That is no longer valid because the connected-component
#      size now matters.
#    - Recoloring directly during the scan is safe only when component context is irrelevant.
#      Here we must compute component membership first, then decide whether to recolor.
# d) Placement legality:
#    - Every component is placed fully in-bounds.
#    - Every new component must occupy bg cells and keep a 4-neighbor gap from prior fg,
#      preventing accidental merges.
#    - Embedded pluses are built by extending one or two arms beyond the 5-cell cross,
#      guaranteeing component size > 5 while preserving a visible plus center.
# e) Visual observability:
#    - Humans can compare two nearly identical motifs: a small standalone '+' turns teal,
#      a larger cross-containing component does not. This is inferable from 2-3 examples.
# f) Visual cleanliness:
#    - A moat between components avoids clutter and prevents the grid from turning into
#      an undifferentiated blob field.
# g) Color-change inference:
#    - Only one recolor is used: fg -> teal. The deciding feature is structural
#      (isolated 5-cell component versus larger component), not color id.
# 9) New solver assumptions and generator guarantees:
#    - `main()` assumes a target isolated plus forms a 4-connected component of size 5.
#      `generate_input()` guarantees that by placing exact 5-cell crosses with no
#      orthogonally adjacent fg cells.
#    - `main()` ignores embedded pluses by component size. `generate_input()` guarantees
#      at least one embedded plus component with a valid local plus center and size > 5.


DIRS4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]
PLUS_OFFSETS = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]

DISTRACTOR_SHAPES = [
    [(0, 0)],
    [(0, 0), (0, 1)],
    [(0, 0), (1, 0)],
    [(0, 0), (0, 1), (0, 2)],
    [(0, 0), (1, 0), (2, 0)],
    [(0, 0), (1, 0), (1, 1)],
    [(0, 1), (1, 0), (1, 1)],
    [(0, 0), (0, 1), (1, 0), (1, 1)],
    [(0, 0), (0, 1), (0, 2), (0, 3)],
    [(0, 0), (1, 0), (2, 0), (3, 0)],
    [(0, 0), (0, 1), (1, 1), (1, 2)],
    [(0, 1), (1, 0), (1, 1), (2, 0)],
    [(0, 0), (1, 0), (2, 0), (2, 1)],
    [(0, 1), (1, 1), (2, 0), (2, 1)],
    [(0, 0), (0, 1), (0, 2), (1, 0)],
    [(0, 0), (0, 1), (0, 2), (1, 2)],
    [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],
    [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],
    [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
    [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)],
]


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _plus_center_in_component(comp: list[tuple[int, int]]) -> tuple[int, int] | None:
    if len(comp) != 5:
        return None
    cells = set(comp)
    for r, c in comp:
        if all((r + dr, c + dc) in cells for dr, dc in DIRS4):
            return r, c
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect isolated thin '+' shapes in a single foreground color and recolor them to teal.

    Input structure:
    - The background is the most frequent color.
    - One foreground color forms multiple disconnected components, including exact 5-cell '+'
      crosses and larger components that may also contain a local '+' center.

    Rule:
    - Find every 4-connected foreground component of size exactly 5.
    - If that component has a center cell whose 4 orthogonal neighbors are the other 4 cells,
      recolor the whole 5-cell component to teal (8).
    - Leave embedded '+' motifs inside larger foreground components unchanged.

    Key insight: A local neighbor test is no longer enough; the solver must combine plus-shape
    detection with component-size analysis so that only standalone crosses are recolored.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    colors = sorted(set(map(int, g.flatten())) - {bg})
    if not colors:
        return out
    fg = int(colors[0])

    for comp in _connected_components(g == fg):
        if _plus_center_in_component(comp) is None:
            continue
        for r, c in comp:
            out[r, c] = 8
    return out


def _plus_cells(cr: int, cc: int) -> list[tuple[int, int]]:
    return [(cr + dr, cc + dc) for dr, dc in PLUS_OFFSETS]


def _translated(shape: list[tuple[int, int]], r0: int, c0: int) -> list[tuple[int, int]]:
    return [(r0 + dr, c0 + dc) for dr, dc in shape]


def _can_place_component(grid: np.ndarray, cells: list[tuple[int, int]], bg: int) -> bool:
    H, W = grid.shape
    cell_set = set(cells)
    for r, c in cells:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if int(grid[r, c]) != bg:
            return False
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in cell_set and int(grid[nr, nc]) != bg:
                return False
    return True


def _place_component(grid: np.ndarray, cells: list[tuple[int, int]], fg: int) -> None:
    for r, c in cells:
        grid[r, c] = fg


def _build_embedded_plus_shape(rng: np.random.Generator) -> list[tuple[int, int]]:
    cells = set(PLUS_OFFSETS)
    dirs = rng.permutation(len(DIRS4))
    n_ext = int(rng.integers(1, 3))
    for idx in dirs[:n_ext]:
        dr, dc = DIRS4[int(idx)]
        ext_len = int(rng.integers(1, 4))
        for step in range(2, ext_len + 2):
            cells.add((dr * step, dc * step))
    return list(cells)


def _count_plus_centers(grid: np.ndarray, fg: int) -> tuple[int, int]:
    comps = _connected_components(grid == fg)
    owner: dict[tuple[int, int], int] = {}
    for idx, comp in enumerate(comps):
        for cell in comp:
            owner[cell] = idx

    H, W = grid.shape
    isolated = 0
    embedded = 0
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if int(grid[r, c]) != fg:
                continue
            if all(int(grid[r + dr, c + dc]) == fg for dr, dc in DIRS4):
                if len(comps[owner[(r, c)]]) == 5:
                    isolated += 1
                else:
                    embedded += 1
    return isolated, embedded


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))
    colors = sorted(set(map(int, g.flatten())) - {bg})
    if not colors:
        return out
    fg = int(colors[0])
    H, W = g.shape
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if int(g[r, c]) != fg:
                continue
            if all(int(g[r + dr, c + dc]) == fg for dr, dc in DIRS4):
                for dr, dc in PLUS_OFFSETS:
                    out[r + dr, c + dc] = 8
    return out


def _place_isolated_pluses(grid: np.ndarray, rng: np.random.Generator, count: int, bg: int, fg: int) -> bool:
    H, W = grid.shape
    for _ in range(count):
        placed = False
        for _attempt in range(120):
            cr = int(rng.integers(2, H - 2))
            cc = int(rng.integers(2, W - 2))
            cells = _plus_cells(cr, cc)
            if _can_place_component(grid, cells, bg):
                _place_component(grid, cells, fg)
                placed = True
                break
        if not placed:
            return False
    return True


def _place_embedded_pluses(grid: np.ndarray, rng: np.random.Generator, count: int, bg: int, fg: int) -> bool:
    H, W = grid.shape
    for _ in range(count):
        placed = False
        for _attempt in range(200):
            shape = _build_embedded_plus_shape(rng)
            rs = [dr for dr, _dc in shape]
            cs = [dc for _dr, dc in shape]
            r_min, r_max = min(rs), max(rs)
            c_min, c_max = min(cs), max(cs)
            r0 = int(rng.integers(-r_min, H - r_max))
            c0 = int(rng.integers(-c_min, W - c_max))
            cells = _translated(shape, r0, c0)
            if _can_place_component(grid, cells, bg):
                _place_component(grid, cells, fg)
                placed = True
                break
        if not placed:
            return False
    return True


def _fill_with_distractors(grid: np.ndarray, rng: np.random.Generator, bg: int, fg: int, target_fg: int) -> bool:
    H, W = grid.shape
    current_fg = int(np.sum(grid == fg))
    stall = 0
    while current_fg < target_fg and stall < 700:
        remaining = target_fg - current_fg
        candidates = [shape for shape in DISTRACTOR_SHAPES if len(shape) <= max(1, remaining + 1)]
        shape = candidates[int(rng.integers(0, len(candidates)))]
        rs = [dr for dr, _dc in shape]
        cs = [dc for _dr, dc in shape]
        r_max = max(rs)
        c_max = max(cs)
        r0 = int(rng.integers(0, H - r_max))
        c0 = int(rng.integers(0, W - c_max))
        cells = _translated(shape, r0, c0)
        if _can_place_component(grid, cells, bg):
            _place_component(grid, cells, fg)
            current_fg += len(cells)
            stall = 0
        else:
            stall += 1
    return current_fg >= target_fg


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    H = int(rng.integers(10, 23))
    W = int(rng.integers(10, 23))
    bg, fg = 2, 4
    grid = np.full((H, W), bg, dtype=int)

    area = H * W
    total_pluses = int(rng.integers(2, min(8, max(4, area // 35 + 1))))
    isolated_count = int(rng.integers(1, total_pluses))
    embedded_count = total_pluses - isolated_count

    if not _place_isolated_pluses(grid, rng, isolated_count, bg, fg):
        return None
    if not _place_embedded_pluses(grid, rng, embedded_count, bg, fg):
        return None

    target_density = float(rng.uniform(0.24, 0.36))
    target_fg = int(area * target_density)
    if not _fill_with_distractors(grid, rng, bg, fg, target_fg):
        return None

    if int(np.sum(grid == fg)) >= area // 2:
        return None

    isolated_plus_centers, embedded_plus_centers = _count_plus_centers(grid, fg)
    if isolated_plus_centers < 1 or embedded_plus_centers < 1:
        return None

    new_out = main(grid)
    old_out = _legacy_main(grid)
    if not np.any(new_out == 8):
        return None
    if np.array_equal(new_out, old_out):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 400 attempts")
