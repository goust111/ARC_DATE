import numpy as np
from collections import Counter

from arc_utils import DIR4_URDL as _DIRS
from arc_utils import bbox_pts as _bbox
from arc_utils import cc4 as _cc4
from arc_utils import most_common as _most_common


def _parse_legend(grid: np.ndarray, legend_h: int) -> dict[int, str]:
    """
    Legend is a top region with a background color (typically 5) and 4 vertical bars.
    Each bar's horizontal position inside its cell encodes turn direction:
      - bar on left  -> 'L' (turn counterclockwise)
      - bar on right -> 'R' (turn clockwise)
    """
    legend = grid[:legend_h, :]
    if legend.size == 0:
        return {}
    bg = _most_common(legend)

    # Separator columns are solid bg throughout the legend.
    sep_cols = [c for c in range(legend.shape[1]) if np.all(legend[:, c] == bg)]
    if len(sep_cols) < 2:
        return {}

    turn: dict[int, str] = {}
    for a, b in zip(sep_cols, sep_cols[1:], strict=False):
        if b - a <= 2:
            continue
        interior = legend[1 : legend_h - 1, a + 1 : b]
        colors = set(map(int, np.unique(interior)))
        colors.discard(bg)
        colors.discard(0)
        if len(colors) != 1:
            continue
        col = int(next(iter(colors)))
        xs = np.where(interior == col)[1]
        if xs.size == 0:
            continue
        x = int(xs[0])  # relative to interior
        # Decide left/right by proximity to borders (interior is typically 4 cells wide).
        turn[col] = "L" if x <= (interior.shape[1] - 1) // 2 else "R"
    return turn


def _find_teal_square(g: np.ndarray, teal: int = 8) -> tuple[int, int, int] | None:
    """Return (r0,c0,size) for a solid square teal component."""
    h, w = g.shape
    best = None
    best_key = None  # (area, r0, c0)
    for cells in _cc4(g == teal):
        r0, c0, r1, c1 = _bbox(cells)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if hh != ww:
            continue
        size = hh
        if int(np.sum(g[r0 : r1 + 1, c0 : c1 + 1] == teal)) != size * size:
            continue
        key = (int(size * size), int(r0), int(c0))
        if best_key is None or key[0] > best_key[0] or (key[0] == best_key[0] and key[1:] < best_key[1:]):
            best_key = key
            best = (int(r0), int(c0), int(size))
    return best


def _rot_ccw(d: int) -> int:
    return (d - 1) % 4


def _rot_cw(d: int) -> int:
    return (d + 1) % 4


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Emit four teal beams from a square, turning at colored obstacles per a legend.

    Input structure:
    - Top band: grey (5) legend with colored vertical bars in cells; bar position (left or right) encodes turn direction.
    - Below: main grid with bg, a solid teal (8) square, and colored obstacle blocks.

    Rule:
    - Remove the legend from the output.
    - From the teal square, shoot a beam in each cardinal direction (as a sliding copy of the square).
    - Each beam paints teal over bg cells as it moves.
    - On hitting an obstacle of color C, consult the legend: bar on left → turn counterclockwise, bar on right → turn clockwise. Continue in the new direction.
    - Beams stop at grid edges or unrecognized obstacles.

    Key insight: The legend's bar position (left vs right within its cell) is the encoding for turn direction at each obstacle color.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    # Legend height: crop everything above and including the last row containing legend bg (usually 5).
    # In this dataset, the legend uses a dedicated color (commonly 5) that doesn't appear below.
    # Robustly: treat 5 as the legend bg if present near the top; otherwise use the top-row mode.
    legend_bg = 5 if 5 in set(map(int, np.unique(g[: min(h, 8), :]))) else _most_common(g[0:1, :])
    rows_with_legend = np.where(np.any(g == legend_bg, axis=1))[0]
    legend_h = int(rows_with_legend.max() + 1) if rows_with_legend.size else 0

    legend_turn = _parse_legend(g, legend_h)

    base = g[legend_h:, :].copy()
    if base.size == 0:
        return base

    bg = _most_common(base)
    teal = 8
    src = _find_teal_square(base, teal=teal)
    if src is None:
        return base
    sr, sc, sz = src

    out = base.copy()

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r and 0 <= c and r + sz - 1 < out.shape[0] and c + sz - 1 < out.shape[1]

    def paint_block(r: int, c: int) -> None:
        blk = out[r : r + sz, c : c + sz]
        blk[blk == bg] = teal
        out[r : r + sz, c : c + sz] = blk

    def obstacle_color_at_next(nr: int, nc: int) -> int | None:
        blk = out[nr : nr + sz, nc : nc + sz]
        cols = set(map(int, np.unique(blk)))
        cols.discard(bg)
        cols.discard(teal)
        if not cols:
            return None
        if len(cols) == 1:
            return int(next(iter(cols)))
        return -1  # ambiguous

    # Four beams, each is a moving 2x2 teal block.
    for d0 in range(4):
        r, c, d = sr, sc, d0
        seen: set[tuple[int, int, int]] = set()
        for _step in range(out.shape[0] * out.shape[1] * 10):
            state = (r, c, d)
            if state in seen:
                break
            seen.add(state)

            dr, dc = _DIRS[d]
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc):
                break

            oc = obstacle_color_at_next(nr, nc)
            if oc is None:
                r, c = nr, nc
                paint_block(r, c)
                continue
            if oc == -1:
                break

            turn = legend_turn.get(oc)
            if turn is None:
                break
            d = _rot_ccw(d) if turn == "L" else _rot_cw(d)

    return out
