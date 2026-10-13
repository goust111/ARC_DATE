import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
COLORS = tuple(c for c in range(10) if c not in (BLACK, GREY))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        q = deque([(sr, sc)])
        pts: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            pts.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]) and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
        comps.append(pts)
    return comps


def _bbox(pts: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    return min(rs), min(cs), max(rs), max(cs)


def _find_frame(g: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(g == GREY)
    assert len(pts) > 0, "a grey template frame is required"
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must enclose at least one cavity"
    assert np.all(g[r0, c0 : c1 + 1] == GREY)
    assert np.all(g[r1, c0 : c1 + 1] == GREY)
    assert np.all(g[r0 : r1 + 1, c0] == GREY)
    assert np.all(g[r0 : r1 + 1, c1] == GREY)
    return r0, c0, r1, c1


def _black_strip(g: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> bool:
    if r0 >= r1 or c0 >= c1:
        return True
    return bool(np.all(g[r0:r1, c0:c1] == BLACK))


def _legend_mapping(g: np.ndarray, frame_top: int) -> dict[int, np.ndarray]:
    h, w = g.shape
    candidates: list[tuple[int, int, int, int, np.ndarray]] = []
    for s in (3, 2):
        for r in range(0, max(0, frame_top - s + 1)):
            for c in range(0, w - 2 * s + 1):
                color = int(g[r, c])
                if color in (BLACK, GREY):
                    continue
                square = g[r : r + s, c : c + s]
                pattern = g[r : r + s, c + s : c + 2 * s]
                if not bool(np.all(square == color)):
                    continue
                if not bool(np.isin(pattern, [BLACK, color]).all()):
                    continue
                if not bool((pattern == color).any()):
                    continue
                if c > 0 and not _black_strip(g, r, r + s, c - 1, c):
                    continue
                if c + 2 * s < w and not _black_strip(g, r, r + s, c + 2 * s, c + 2 * s + 1):
                    continue
                if r > 0 and not _black_strip(g, r - 1, r, c, c + 2 * s):
                    continue
                if r + s < h and not _black_strip(g, r + s, r + s + 1, c, c + 2 * s):
                    continue
                candidates.append((r, c, s, color, pattern == color))

    used = np.zeros_like(g, dtype=bool)
    mapping: dict[int, np.ndarray] = {}
    for r, c, s, color, mask in sorted(candidates, key=lambda x: (x[0], x[1], -x[2])):
        if bool(used[r : r + s, c : c + 2 * s].any()):
            continue
        assert color not in mapping, "legend colors must be unique"
        mapping[color] = mask.copy()
        used[r : r + s, c : c + 2 * s] = True
    assert mapping, "at least one color legend is required"
    return mapping


def _cavities(g: np.ndarray, frame: tuple[int, int, int, int]) -> list[tuple[int, int, int, int, int]]:
    r0, c0, r1, c1 = frame
    inside = g[r0 + 1 : r1, c0 + 1 : c1]
    comps = _components(inside != GREY)
    out: list[tuple[int, int, int, int, int]] = []
    for pts in comps:
        lr0, lc0, lr1, lc1 = _bbox(pts)
        submask = inside[lr0 : lr1 + 1, lc0 : lc1 + 1] != GREY
        assert bool(submask.all()), "each cavity must be rectangular"
        gr0, gc0 = r0 + 1 + lr0, c0 + 1 + lc0
        gr1, gc1 = r0 + 1 + lr1, c0 + 1 + lc1
        patch = g[gr0 : gr1 + 1, gc0 : gc1 + 1]
        markers = np.argwhere((patch != BLACK) & (patch != GREY))
        assert len(markers) == 1, "each cavity must contain exactly one marker"
        mr, mc = map(int, markers[0])
        color = int(patch[mr, mc])
        out.append((gr0, gc0, gr1, gc1, color))
    assert out, "template must contain at least one cavity"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey-template cavity with the scaled legend pattern keyed by its marker color.

    Input structure:
    - The background is black.
    - Above the template is a horizontal legend row. Each legend entry is a solid 2x2 or
      3x3 color square immediately followed by a same-sized black/color mask.
    - Below it is one grey rectangular template whose grey border and grey walls split the
      interior into rectangular cavities.
    - Each cavity contains exactly one single-pixel non-black, non-grey marker whose color
      matches one legend square; all other cavity cells are black.

    Rule:
    - Read the color marker in every cavity.
    - Look up that color's mask in the upper legend.
    - Block-scale the mask by integer row and column factors so it exactly covers the
      cavity rectangle.
    - Clear the cavity to black, then paint the scaled foreground mask with the marker
      color while leaving all grey border and wall pixels unchanged.

    Key insight: The single marker pixel is only an index into the legend; the cavity size
    supplies the block-scaling factors for the looked-up pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frame = _find_frame(g)
    mapping = _legend_mapping(g, frame[0])
    out = g.copy()

    for r0, c0, r1, c1, color in _cavities(g, frame):
        assert color in mapping, "every marker color must appear in the legend"
        mask = mapping[color]
        mh, mw = mask.shape
        ch, cw = r1 - r0 + 1, c1 - c0 + 1
        assert ch % mh == 0 and cw % mw == 0, "cavity size must be an integer mask scale"
        scaled = np.kron(mask, np.ones((ch // mh, cw // mw), dtype=bool))
        assert scaled.shape == (ch, cw)
        out[r0 : r1 + 1, c0 : c1 + 1] = BLACK
        out[r0 : r1 + 1, c0 : c1 + 1][scaled] = color

    return out


def _random_mask(size: int, rng: random.Random) -> np.ndarray:
    total = size * size
    count = rng.randint(1, total - 1)
    cells = rng.sample(range(total), count)
    mask = np.zeros((size, size), dtype=bool)
    for idx in cells:
        mask[idx // size, idx % size] = True
    return mask


def _draw_frame(
    grid: np.ndarray,
    top: int,
    left: int,
    row_heights: list[int],
    col_widths: list[int],
) -> list[tuple[int, int, int, int]]:
    frame_h = sum(row_heights) + len(row_heights) + 1
    frame_w = sum(col_widths) + len(col_widths) + 1

    rr = 0
    grid[top + rr, left : left + frame_w] = GREY
    for ch in row_heights:
        rr += ch + 1
        grid[top + rr, left : left + frame_w] = GREY

    cc = 0
    grid[top : top + frame_h, left + cc] = GREY
    for cw in col_widths:
        cc += cw + 1
        grid[top : top + frame_h, left + cc] = GREY

    cavities: list[tuple[int, int, int, int]] = []
    local_r = 1
    for ch in row_heights:
        local_c = 1
        for cw in col_widths:
            r0 = top + local_r
            c0 = left + local_c
            cavities.append((r0, c0, r0 + ch - 1, c0 + cw - 1))
            local_c += cw + 1
        local_r += ch + 1
    return cavities


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    size = rng.choice([2, 3])
    row_bands = rng.randint(1, 3)
    col_bands = rng.randint(2, 4)
    cavity_count = row_bands * col_bands
    n_entries = rng.randint(2, min(4, cavity_count, len(COLORS)))

    max_scale = 4 if size == 2 else 3
    row_scales = [rng.randint(1, max_scale) for _ in range(row_bands)]
    col_scales = [rng.randint(1, max_scale) for _ in range(col_bands)]
    row_heights = [size * k for k in row_scales]
    col_widths = [size * k for k in col_scales]
    frame_h = sum(row_heights) + row_bands + 1
    frame_w = sum(col_widths) + col_bands + 1

    gaps = [rng.randint(1, 2) for _ in range(n_entries - 1)]
    legend_w = 2 * size * n_entries + sum(gaps)
    legend_top = rng.randint(0, 1)
    vertical_gap = rng.randint(2, 5)
    min_h = legend_top + size + vertical_gap + frame_h
    min_w = max(frame_w, legend_w)
    if min_h > 30 or min_w > 30:
        return None

    extra_h = rng.randint(0, 30 - min_h)
    extra_between = rng.randint(0, extra_h)
    bottom_extra = extra_h - extra_between
    h = min_h + extra_h
    w = rng.randint(min_w, 30)
    frame_top = legend_top + size + vertical_gap + extra_between
    frame_left = rng.randint(0, w - frame_w)
    legend_left = rng.randint(0, w - legend_w)

    grid = np.full((h, w), BLACK, dtype=int)
    colors = rng.sample(list(COLORS), n_entries)
    patterns = [_random_mask(size, rng) for _ in colors]

    c = legend_left
    for i, (color, mask) in enumerate(zip(colors, patterns)):
        grid[legend_top : legend_top + size, c : c + size] = color
        patch = np.where(mask, color, BLACK)
        grid[legend_top : legend_top + size, c + size : c + 2 * size] = patch
        c += 2 * size
        if i < len(gaps):
            c += gaps[i]

    cavities = _draw_frame(grid, frame_top, frame_left, row_heights, col_widths)
    assigned = colors[:]
    while len(assigned) < len(cavities):
        assigned.append(rng.choice(colors))
    rng.shuffle(assigned)

    for (r0, c0, r1, c1), color in zip(cavities, assigned):
        mr = rng.randint(r0, r1)
        mc = rng.randint(c0, c1)
        grid[mr, mc] = color

    if bottom_extra < 0:
        return None
    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape or max(out.shape) > 30:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(500):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to generate a valid input")
