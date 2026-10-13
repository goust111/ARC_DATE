import random
from collections import deque

import numpy as np


GREY = 5
ORTHO = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _background_color(g: np.ndarray) -> int:
    vals, counts = np.unique(g[g != GREY], return_counts=True)
    assert len(vals) > 0, "expected at least one non-grey color"
    return int(vals[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or not mask[sr, sc]:
                continue
            q = deque([(sr, sc)])
            seen[sr, sc] = True
            comp: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                comp.append((r, c))
                for dr, dc in ORTHO:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _grey_frames(g: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for comp in _components(g == GREY):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        height, width = r1 - r0 + 1, c1 - c0 + 1
        if height < 4 or width < 4:
            continue
        expected = 2 * height + 2 * width - 4
        if len(comp) != expected:
            continue
        border = np.zeros(g.shape, dtype=bool)
        border[r0 : r1 + 1, c0] = True
        border[r0 : r1 + 1, c1] = True
        border[r0, c0 : c1 + 1] = True
        border[r1, c0 : c1 + 1] = True
        if all(border[r, c] for r, c in comp) and np.all(g[border] == GREY):
            frames.append((r0, c0, r1, c1))
    assert frames, "expected at least one complete grey rectangular frame"
    return frames


def _corner_score(shape: tuple[int, int], cells: tuple[tuple[int, int], ...]) -> int:
    h, w = shape
    corners = ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1))
    return min(max(abs(r - cr) + abs(c - cc) for r, c in cells) for cr, cc in corners)


def _legend_colors(g: np.ndarray, bg: int) -> tuple[int, int]:
    h, w = g.shape
    blocked = {bg, GREY}
    horizontal: list[tuple[int, tuple[tuple[int, int], ...]]] = []
    vertical: list[tuple[int, tuple[tuple[int, int], ...]]] = []

    for r in range(h):
        for c in range(w - 2):
            frag = int(g[r, c])
            target = int(g[r, c + 2])
            if frag == int(g[r, c + 1]) and frag not in blocked and target not in blocked and target != frag:
                horizontal.append((target, ((r, c), (r, c + 1), (r, c + 2))))

    for r in range(h - 2):
        for c in range(w):
            frag = int(g[r, c])
            target = int(g[r + 2, c])
            if frag == int(g[r + 1, c]) and frag not in blocked and target not in blocked and target != frag:
                vertical.append((target, ((r, c), (r + 1, c), (r + 2, c))))

    assert horizontal, "expected a horizontal legend entry"
    assert vertical, "expected a vertical legend entry"

    h_color, _ = min(horizontal, key=lambda item: _corner_score(g.shape, item[1]))
    v_color, _ = min(vertical, key=lambda item: _corner_score(g.shape, item[1]))
    assert h_color != v_color, "legend output colors must be distinct"
    return int(h_color), int(v_color)


def _fragment_orientation(g: np.ndarray, box: tuple[int, int, int, int], bg: int) -> str:
    r0, c0, r1, c1 = box
    cells: list[tuple[int, int, int]] = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            color = int(g[r, c])
            if color not in (bg, GREY):
                cells.append((r, c, color))

    assert len(cells) == 2, "each frame interior must contain exactly one two-pixel fragment"
    (r_a, c_a, col_a), (r_b, c_b, col_b) = cells
    assert col_a == col_b, "fragment pixels must have the same color"
    if r_a == r_b and abs(c_a - c_b) == 1:
        return "horizontal"
    if c_a == c_b and abs(r_a - r_b) == 1:
        return "vertical"
    raise AssertionError("fragment pixels must be orthogonally adjacent")


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each grey rectangular frame according to the orientation of its interior fragment.

    Input structure:
    - The grid has a dominant background color and several one-pixel-thick grey rectangular frames.
    - Each grey frame encloses background cells plus exactly one two-pixel same-color fragment.
    - A compact corner legend contains a horizontal two-pixel fragment followed on the right by color A, and a vertical two-pixel fragment followed below by color B.

    Rule:
    - Read the legend so horizontal fragments map to color A and vertical fragments map to color B.
    - For every complete grey rectangular frame, inspect the enclosed two-pixel fragment.
    - Recolor the whole grey border to color A for horizontal fragments, or color B for vertical fragments; leave fragments and background unchanged.

    Key insight:
    The fragment color is a distractor; its spatial orientation selects one of the two legend colors used to repaint the enclosing border.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must be at most 30x30"

    bg = _background_color(g)
    horizontal_color, vertical_color = _legend_colors(g, bg)

    out = g.copy()
    for r0, c0, r1, c1 in _grey_frames(g):
        orientation = _fragment_orientation(g, (r0, c0, r1, c1), bg)
        color = horizontal_color if orientation == "horizontal" else vertical_color
        out[r0 : r1 + 1, c0] = color
        out[r0 : r1 + 1, c1] = color
        out[r0, c0 : c1 + 1] = color
        out[r1, c0 : c1 + 1] = color
    return out


def _draw_legend(grid: np.ndarray, corner: str, h_frag: int, v_frag: int, h_color: int, v_color: int, bg: int) -> tuple[int, int, int, int]:
    patch = np.full((5, 4), bg, dtype=int)
    patch[0, 0:2] = h_frag
    patch[0, 2] = h_color
    patch[2:4, 0] = v_frag
    patch[4, 0] = v_color

    h, w = grid.shape
    ph, pw = patch.shape
    if corner == "tl":
        r0, c0 = 0, 0
    elif corner == "tr":
        r0, c0 = 0, w - pw
    elif corner == "bl":
        r0, c0 = h - ph, 0
    else:
        r0, c0 = h - ph, w - pw
    grid[r0 : r0 + ph, c0 : c0 + pw] = patch
    return r0, c0, r0 + ph - 1, c0 + pw - 1


def _reserve_box(occ: np.ndarray, box: tuple[int, int, int, int], pad: int = 1) -> None:
    h, w = occ.shape
    r0, c0, r1, c1 = box
    occ[max(0, r0 - pad) : min(h, r1 + pad + 1), max(0, c0 - pad) : min(w, c1 + pad + 1)] = True


def _fits(occ: np.ndarray, r0: int, c0: int, fh: int, fw: int, pad: int = 1) -> bool:
    h, w = occ.shape
    r1, c1 = r0 + fh - 1, c0 + fw - 1
    if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
        return False
    return not bool(occ[max(0, r0 - pad) : min(h, r1 + pad + 1), max(0, c0 - pad) : min(w, c1 + pad + 1)].any())


def _draw_frame(grid: np.ndarray, r0: int, c0: int, fh: int, fw: int) -> None:
    r1, c1 = r0 + fh - 1, c0 + fw - 1
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY


def _place_fragment(grid: np.ndarray, box: tuple[int, int, int, int], orientation: str, color: int) -> None:
    r0, c0, r1, c1 = box
    if orientation == "horizontal":
        r = random.randint(r0 + 1, r1 - 1)
        c = random.randint(c0 + 1, c1 - 2)
        grid[r, c] = color
        grid[r, c + 1] = color
    else:
        r = random.randint(r0 + 1, r1 - 2)
        c = random.randint(c0 + 1, c1 - 1)
        grid[r, c] = color
        grid[r + 1, c] = color


def _expected_output(inp: np.ndarray, boxes: list[tuple[int, int, int, int, str]], h_color: int, v_color: int) -> np.ndarray:
    out = inp.copy()
    for r0, c0, r1, c1, orientation in boxes:
        color = h_color if orientation == "horizontal" else v_color
        out[r0 : r1 + 1, c0] = color
        out[r0 : r1 + 1, c1] = color
        out[r0, c0 : c1 + 1] = color
        out[r1, c0 : c1 + 1] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(16, 30)
    w = random.randint(16, 30)
    bg = random.choice([0, 1, 2, 3, 4, 6, 7, 8, 9])
    colors = [c for c in range(10) if c not in (bg, GREY)]
    h_frag, v_frag, h_color, v_color = random.sample(colors, 4)
    fragment_pool = [c for c in colors if c not in (h_color, v_color)]

    grid = np.full((h, w), bg, dtype=int)
    occ = np.zeros((h, w), dtype=bool)
    legend_box = _draw_legend(grid, random.choice(["tl", "tr", "bl", "br"]), h_frag, v_frag, h_color, v_color, bg)
    _reserve_box(occ, legend_box, pad=1)

    n_frames = random.randint(2, 5)
    orientations = ["horizontal", "vertical"]
    orientations.extend(random.choice(["horizontal", "vertical"]) for _ in range(n_frames - 2))
    random.shuffle(orientations)

    placed: list[tuple[int, int, int, int, str]] = []
    for orientation in orientations:
        for _ in range(250):
            fh = random.randint(5, min(9, h - 2))
            fw = random.randint(5, min(10, w - 2))
            r0 = random.randint(1, h - fh - 1)
            c0 = random.randint(1, w - fw - 1)
            if not _fits(occ, r0, c0, fh, fw, pad=1):
                continue
            _draw_frame(grid, r0, c0, fh, fw)
            _place_fragment(grid, (r0, c0, r0 + fh - 1, c0 + fw - 1), orientation, random.choice(fragment_pool))
            _reserve_box(occ, (r0, c0, r0 + fh - 1, c0 + fw - 1), pad=1)
            placed.append((r0, c0, r0 + fh - 1, c0 + fw - 1, orientation))
            break
        else:
            return None

    expected = _expected_output(grid, placed, h_color, v_color)
    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
