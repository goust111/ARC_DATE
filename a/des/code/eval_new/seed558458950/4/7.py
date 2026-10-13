import numpy as np
import random


RING_OFFSETS = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0)]
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
BG_COLOR = 0


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and mask[nr, nc]
                        and not visited[nr, nc]
                    ):
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _coords_to_mask(
    coords: list[tuple[int, int]], h: int, w: int
) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    return mask


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _outer_border_mask(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    rows, cols = np.where(mask)
    r0, c0 = int(rows.min()), int(cols.min())
    r1, c1 = int(rows.max()), int(cols.max())
    rr = np.arange(mask.shape[0])[:, None]
    cc = np.arange(mask.shape[1])[None, :]
    return mask & (
        (rr == r0)
        | (rr == r1)
        | (cc == c0)
        | (cc == c1)
    )


def _find_legend(grid: np.ndarray, bg: int) -> tuple[list[int], set[tuple[int, int]]]:
    h, w = grid.shape
    candidates: list[tuple[list[int], set[tuple[int, int]]]] = []

    if h < 3 or w < 3:
        raise ValueError("grid too small to contain cycle legend")

    corners = [
        (0, 0),
        (0, w - 3),
        (h - 3, 0),
        (h - 3, w - 3),
    ]

    for r0, c0 in corners:
        if r0 < 0 or c0 < 0 or r0 + 2 >= h or c0 + 2 >= w:
            continue
        if int(grid[r0 + 1, c0 + 1]) != int(bg):
            continue
        cycle = []
        cells = set[tuple[int, int]]()
        valid = True
        for dr, dc in RING_OFFSETS:
            v = int(grid[r0 + dr, c0 + dc])
            if v == bg:
                valid = False
                break
            cycle.append(v)
            cells.add((r0 + dr, c0 + dc))
        if valid and len(set(cycle)) == 8:
            candidates.append((cycle, cells))

    if not candidates:
        # Fallback: search all 3x3 rings that match the same pattern.
        for r0 in range(h - 2):
            for c0 in range(w - 2):
                if int(grid[r0 + 1, c0 + 1]) != int(bg):
                    continue
                cycle = []
                cells = set[tuple[int, int]]()
                valid = True
                for dr, dc in RING_OFFSETS:
                    v = int(grid[r0 + dr, c0 + dc])
                    if v == bg:
                        valid = False
                        break
                    cycle.append(v)
                    cells.add((r0 + dr, c0 + dc))
                if valid and len(set(cycle)) == 8:
                    candidates.append((cycle, cells))
        if not candidates:
            raise ValueError("legend not found")

    cycle, cells = candidates[0]
    if len(set(cycle)) != len(cycle):
        raise ValueError("legend cycle colors must be distinct")
    return cycle, cells


def _find_onion_mask(
    grid: np.ndarray, bg: int, legend_cells: set[tuple[int, int]]
) -> np.ndarray:
    h, w = grid.shape
    non_bg = grid != bg
    for r, c in legend_cells:
        non_bg[r, c] = False

    comps = _components(non_bg)
    if not comps:
        return np.zeros((h, w), dtype=bool)

    # Prefer the largest non-border component.
    best: np.ndarray | None = None
    for comp in sorted(comps, key=lambda c: len(c), reverse=True):
        if len(comp) <= 8:
            continue
        r0, c0, r1, c1 = _bbox(comp)
        touches_edge = r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1
        comp_mask = _coords_to_mask(comp, h, w)
        if not touches_edge:
            return comp_mask
        if best is None:
            best = comp_mask
    if best is None:
        best = _coords_to_mask(max(comps, key=len), h, w)
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """
    Peeling one outer frame at a time updates the next inner frame by cycling colors.

    Input structure:
    - bg is the dominant color. A multi-layer 1-pixel-thick concentric rectangle onion
      sits in the grid, with a solid inner core.
    - A 3×3 cycle legend in one corner stores colors on its 8-cycle border around a bg center.

    Rule:
    - Repeatedly take the current outermost layer of the onion, look up its color's
      clockwise successor on the cycle legend, remove that outer layer to bg, and recolor the
      newly exposed next layer with that successor color.
    - Continue until the onion has no removable outer frame (only the final solid core remains).
    - Return the grid with only the final peeled result: bg background, untouched legend,
      and the inner core recolored by the chain of color-successor operations.

    Key insight:
    - Each peel both consumes one layer and shifts the next remaining layer along the color
      cycle, so the final core color depends on all previous layers through iterative successor
      chaining.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    bg = _bg_color(g)
    cycle, legend_cells = _find_legend(g, bg)
    cycle_next = {cycle[i]: cycle[(i + 1) % len(cycle)] for i in range(len(cycle))}

    onion_mask = _find_onion_mask(g, bg, legend_cells)
    if onion_mask.sum() == 0:
        return g.copy()

    out = g.copy()
    current = onion_mask.copy()

    while True:
        outer = _outer_border_mask(current)
        outer_count = int(outer.sum())
        if outer_count == 0 or outer_count == int(current.sum()):
            break

        outer_colors, outer_counts = np.unique(out[outer], return_counts=True)
        curr_color = int(outer_colors[int(np.argmax(outer_counts))])
        next_color = cycle_next.get(curr_color, curr_color)

        inner = current & ~outer
        if not inner.any():
            break
        inner_outer = _outer_border_mask(inner)
        if inner_outer.sum() == 0:
            break

        out[outer] = bg
        out[inner_outer] = next_color
        current = inner

    return out


def _paint_cycle_legend(
    grid: np.ndarray, corner: int, cycle: list[int]
) -> set[tuple[int, int]]:
    h, w = grid.shape
    if corner == 0:
        r0, c0 = 0, 0
    elif corner == 1:
        r0, c0 = 0, w - 3
    elif corner == 2:
        r0, c0 = h - 3, 0
    else:
        r0, c0 = h - 3, w - 3

    cells = set()
    for (dr, dc), color in zip(RING_OFFSETS, cycle):
        rr = r0 + dr
        cc = c0 + dc
        grid[rr, cc] = color
        cells.add((rr, cc))
    grid[r0 + 1, c0 + 1] = BG_COLOR
    cells.discard((r0 + 1, c0 + 1))
    return cells


def _draw_frame(
    grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int
) -> None:
    grid[r0, c0:c1 + 1] = color
    grid[r1, c0:c1 + 1] = color
    grid[r0:r1 + 1, c0] = color
    grid[r0:r1 + 1, c1] = color


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(1000):
        h = rng.randint(14, 30)
        w = rng.randint(14, 30)
        layer_count = rng.randint(3, 5)

        if rng.random() < 0.45:
            core_h, core_w = 1, 1
        else:
            core_h = rng.randint(2, 4)
            core_w = rng.randint(2, 4)

        outer_h = core_h + 2 * (layer_count - 1)
        outer_w = core_w + 2 * (layer_count - 1)
        if outer_h + 4 >= h or outer_w + 4 >= w:
            continue

        grid = np.full((h, w), BG_COLOR, dtype=int)

        legend_corner = rng.randrange(4)
        cycle = rng.sample(COLOR_POOL, 8)
        rot = rng.randint(0, 7)
        cycle = cycle[rot:] + cycle[:rot]

        # keep layers distinct and always valid against the cycle mapping
        try:
            layer_colors = rng.sample(cycle, layer_count)
        except ValueError:
            continue
        if len(set(layer_colors)) != layer_count:
            continue

        legend_cells = _paint_cycle_legend(grid, legend_corner, cycle)

        # Keep onion away from borders and away from the legend marker.
        top_min, left_min = 1, 1
        top_max = h - outer_h - 1
        left_max = w - outer_w - 1
        if top_max < top_min or left_max < left_min:
            continue

        placed = False
        for _ in range(80):
            r0 = rng.randint(top_min, top_max)
            c0 = rng.randint(left_min, left_max)
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1

            overlap = False
            for rr in range(r0, r1 + 1):
                for cc in range(c0, c1 + 1):
                    if (rr, cc) in legend_cells:
                        overlap = True
                        break
                if overlap:
                    break
            if overlap:
                continue

            for i in range(layer_count):
                cr0 = r0 + i
                cc0 = c0 + i
                cr1 = r0 + outer_h - 1 - i
                cc1 = c0 + outer_w - 1 - i
                if i == layer_count - 1:
                    grid[cr0 : cr1 + 1, cc0 : cc1 + 1] = layer_colors[i]
                else:
                    _draw_frame(grid, cr0, cc0, cr1, cc1, layer_colors[i])

            placed = True
            break

        if not placed:
            continue

        # quick validity check: one connected onion body distinct from legend.
        onion_mask = _find_onion_mask(grid, BG_COLOR, legend_cells)
        if not onion_mask.any():
            continue

        bg = _bg_color(grid)
        if bg != BG_COLOR:
            continue
        return grid

    raise RuntimeError("failed to generate a valid input for the new task")
