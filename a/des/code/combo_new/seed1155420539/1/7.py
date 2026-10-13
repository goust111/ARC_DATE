from collections import deque
import random

import numpy as np


QUADRANTS = ("TL", "TR", "BL", "BR")


def _background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _corner_cells(h: int, w: int) -> dict[str, list[tuple[int, int]]]:
    return {
        "TL": [(0, 0), (0, 1), (1, 0), (1, 1)],
        "TR": [(0, w - 2), (0, w - 1), (1, w - 2), (1, w - 1)],
        "BL": [(h - 2, 0), (h - 2, 1), (h - 1, 0), (h - 1, 1)],
        "BR": [(h - 2, w - 2), (h - 2, w - 1), (h - 1, w - 2), (h - 1, w - 1)],
    }


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(cells)
    return out


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _c in cells]
    cs = [c for _r, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _is_shell(cells: list[tuple[int, int]], box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    expected = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }
    return set(cells) == expected


def _is_solid_rect(cells: list[tuple[int, int]], box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    return len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _inside_with_gap(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> bool:
    ir0, ic0, ir1, ic1 = inner
    or0, oc0, or1, oc1 = outer
    return or0 + 1 < ir0 and ir1 < or1 - 1 and oc0 + 1 < ic0 and ic1 < oc1 - 1


def _quadrant(box: tuple[int, int, int, int], h: int, w: int) -> str:
    r0, c0, r1, c1 = box
    assert r0 + r1 != h, "object center must not lie on the horizontal midline"
    assert c0 + c1 != w, "object center must not lie on the vertical midline"
    vert = "T" if r0 + r1 < h else "B"
    horiz = "L" if c0 + c1 < w else "R"
    return vert + horiz


def _target_origin(quadrant: str, shape: tuple[int, int], grid_shape: tuple[int, int]) -> tuple[int, int]:
    sh, sw = shape
    h, w = grid_shape
    if quadrant == "TL":
        return 0, 0
    if quadrant == "TR":
        return 0, w - sw
    if quadrant == "BL":
        return h - sh, 0
    return h - sh, w - sw


def _paint_shell(grid: np.ndarray, r0: int, c0: int, sh: int, sw: int, color: int) -> None:
    grid[r0, c0 : c0 + sw] = color
    grid[r0 + sh - 1, c0 : c0 + sw] = color
    grid[r0 : r0 + sh, c0] = color
    grid[r0 : r0 + sh, c0 + sw - 1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """Separate nested shell-core objects by sending each shell to its quadrant corner.

    Input structure:
    - The background is the dominant gray color, and each grid corner contains a
      uniform 2x2 color marker; the four marker colors are all different.
    - Away from the corner markers are one or more nested objects. Each nested
      object has a one-cell-thick hollow rectangular shell and one smaller solid
      rectangular core fully inside it with at least one background-cell gap.
    - Each nested object lies in exactly one quadrant of the grid, with at most
      one object per quadrant.

    Rule:
    - Determine the quadrant of each shell-core object from the shell position.
    - Erase the shell at its original location, clear the matching corner area,
      and redraw that shell with its outer edges aligned to the two grid edges of
      the corresponding corner.
    - Leave the core pixels in place, but recolor them to the original color of
      that corner's 2x2 marker.

    Key insight:
    The shell movement is spatial, while the core color comes from the target
    corner marker rather than from the shell or the core itself.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 6 and w >= 6

    bg = _background_color(g)
    corners = _corner_cells(h, w)
    corner_colors: dict[str, int] = {}
    ignored: set[tuple[int, int]] = set()
    for name, cells in corners.items():
        colors = {int(g[r, c]) for r, c in cells}
        assert len(colors) == 1, "each corner marker must be a solid 2x2 square"
        color = colors.pop()
        assert color != bg
        corner_colors[name] = color
        ignored.update(cells)
    assert len(set(corner_colors.values())) == 4

    mask = g != bg
    for r, c in ignored:
        mask[r, c] = False

    shells = []
    cores = []
    for cells in _components(mask):
        colors = {int(g[r, c]) for r, c in cells}
        assert len(colors) == 1, "object components must be monochromatic"
        color = colors.pop()
        box = _bbox(cells)
        entry = {"cells": cells, "color": color, "bbox": box}
        if _is_shell(cells, box):
            shells.append(entry)
        elif _is_solid_rect(cells, box):
            cores.append(entry)
        else:
            raise AssertionError("foreground components must be shells or solid cores")

    assert shells, "expected at least one nested object"
    used_cores: set[int] = set()
    moves = []
    for shell in shells:
        candidates = [
            i
            for i, core in enumerate(cores)
            if i not in used_cores
            and core["color"] != shell["color"]
            and _inside_with_gap(core["bbox"], shell["bbox"])
        ]
        assert len(candidates) == 1, "each shell must contain exactly one solid core"
        core_idx = candidates[0]
        used_cores.add(core_idx)
        core = cores[core_idx]

        sr0, sc0, sr1, sc1 = shell["bbox"]
        sh, sw = sr1 - sr0 + 1, sc1 - sc0 + 1
        quadrant = _quadrant(shell["bbox"], h, w)
        tr, tc = _target_origin(quadrant, (sh, sw), (h, w))
        moves.append((shell, core, quadrant, tr, tc, sh, sw))

    assert len(used_cores) == len(cores), "every solid core must be paired with a shell"
    assert len({m[2] for m in moves}) == len(moves), "each quadrant may contain at most one object"

    out = g.copy()

    for _shell, _core, _quad, tr, tc, sh, sw in moves:
        out[tr : tr + sh, tc : tc + sw] = bg

    for shell, _core, _quad, _tr, _tc, _sh, _sw in moves:
        for r, c in shell["cells"]:
            out[r, c] = bg

    for shell, core, quadrant, tr, tc, sh, sw in moves:
        for r, c in core["cells"]:
            out[r, c] = corner_colors[quadrant]
        _paint_shell(out, tr, tc, sh, sw, shell["color"])

    return out


GEN_BG = 5
GEN_PALETTE = [0, 1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 300


def _side_range(total: int, size: int, low_side: bool) -> tuple[int, int]:
    half = total // 2
    if low_side:
        return size + 1, half - size - 1
    return half + 1, total - 2 * size - 1


def _max_shell_size(total: int, low_side: bool) -> int:
    section = total // 2 if low_side else total - total // 2
    return min(6, (section - 2) // 2)


def _draw_solid(grid: np.ndarray, r0: int, c0: int, height: int, width: int, color: int) -> None:
    grid[r0 : r0 + height, c0 : c0 + width] = color


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(28, 30)
    w = random.randint(28, 30)
    grid = np.full((h, w), GEN_BG, dtype=int)

    marker_colors = random.sample(GEN_PALETTE, 4)
    marker_by_quadrant = dict(zip(QUADRANTS, marker_colors))
    for quadrant, cells in _corner_cells(h, w).items():
        color = marker_by_quadrant[quadrant]
        for r, c in cells:
            grid[r, c] = color

    object_count = random.randint(1, 4)
    chosen_quadrants = random.sample(list(QUADRANTS), object_count)
    shell_color_choices = [c for c in GEN_PALETTE if c not in marker_colors]
    if not shell_color_choices:
        return None

    for quadrant in chosen_quadrants:
        top = quadrant[0] == "T"
        left = quadrant[1] == "L"
        max_sh = _max_shell_size(h, top)
        max_sw = _max_shell_size(w, left)
        if max_sh < 5 or max_sw < 5:
            return None
        sh = random.randint(5, max_sh)
        sw = random.randint(5, max_sw)

        rlo, rhi = _side_range(h, sh, top)
        clo, chi = _side_range(w, sw, left)
        if rlo > rhi or clo > chi:
            return None
        r0 = random.randint(rlo, rhi)
        c0 = random.randint(clo, chi)

        shell_color = random.choice(shell_color_choices)
        core_choices = [
            c
            for c in GEN_PALETTE
            if c != shell_color and c != marker_by_quadrant[quadrant]
        ]
        core_color = random.choice(core_choices)

        _paint_shell(grid, r0, c0, sh, sw, shell_color)

        max_ch = sh - 4
        max_cw = sw - 4
        core_h = random.randint(1, max_ch)
        core_w = random.randint(1, max_cw)
        cr0 = random.randint(r0 + 2, r0 + sh - core_h - 2)
        cc0 = random.randint(c0 + 2, c0 + sw - core_w - 2)
        _draw_solid(grid, cr0, cc0, core_h, core_w, core_color)

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid seed1155420539 puzzle 7 input")
