from __future__ import annotations

from collections import deque

import numpy as np


BG = 0


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _binary_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _monochrome_components(grid: np.ndarray, bg: int = BG) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            r0 = r1 = r
            c0 = c1 = c
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                r0 = min(r0, rr)
                r1 = max(r1, rr)
                c0 = min(c0, cc)
                c1 = max(c1, cc)
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "area": len(coords),
                }
            )
    return comps


def _is_rectangular_frame(grid: np.ndarray, comp: dict) -> bool:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if int(comp["area"]) != 2 * h + 2 * w - 4:
        return False

    sub = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return np.array_equal(sub == int(comp["color"]), border)


def _find_legend_mapping(grid: np.ndarray) -> dict[int, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    comps = _monochrome_components(g, bg=BG)

    bars = []
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if r0 == r1 and int(comp["area"]) == c1 - c0 + 1:
            bars.append(comp)

    def collect(top: bool, left: bool) -> dict[int, int] | None:
        row_to_comp: dict[int, dict] = {}
        for comp in bars:
            r0, c0, _r1, c1 = map(int, comp["bbox"])
            if left and c0 != 0:
                continue
            if not left and c1 != w - 1:
                continue
            if r0 in row_to_comp:
                return None
            row_to_comp[r0] = comp

        rows: list[dict] = []
        if top:
            r = 0
            while r in row_to_comp:
                rows.append(row_to_comp[r])
                r += 1
        else:
            r = h - 1
            while r in row_to_comp:
                rows.append(row_to_comp[r])
                r -= 1

        if len(rows) < 2:
            return None

        mapping: dict[int, int] = {}
        seen_colors: set[int] = set()
        for comp in rows:
            _r0, c0, _r1, c1 = map(int, comp["bbox"])
            length = c1 - c0 + 1
            color = int(comp["color"])
            if length in mapping or color in seen_colors:
                return None
            mapping[length] = color
            seen_colors.add(color)

        if set(mapping) != set(range(1, len(rows) + 1)):
            return None
        return mapping

    candidates = []
    for top in (True, False):
        for left in (True, False):
            mapping = collect(top=top, left=left)
            if mapping is not None:
                candidates.append(mapping)

    assert candidates, "legend must be detectable in one corner"
    max_len = max(len(mapping) for mapping in candidates)
    winners = [mapping for mapping in candidates if len(mapping) == max_len]
    assert len(winners) == 1, "legend corner must be structurally unique"
    return winners[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each framed region with the legend color indexed by its island count.

    The input contains:
    - black background;
    - one-pixel rectangular frames, all in the same border color;
    - a corner legend made of horizontal bars of lengths 1..N;
    - several small interior islands inside each frame.

    For every frame, count 4-connected foreground islands inside it. If the
    count is K, find the legend bar of length K and fill only the black pixels
    inside that frame with the legend bar's color.
    """
    g = np.asarray(grid, dtype=int)
    legend = _find_legend_mapping(g)

    frames = [comp for comp in _monochrome_components(g, bg=BG) if _is_rectangular_frame(g, comp)]
    assert frames, "at least one frame is required"

    out = g.copy()
    for comp in frames:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        island_count = len(_binary_components(interior != BG))
        assert island_count in legend, "each island count must exist in the legend"
        fill_color = int(legend[island_count])
        patch = out[r0 + 1 : r1, c0 + 1 : c1]
        patch[interior == BG] = fill_color

    return out


# --------------- generator ---------------

GEN_RETRIES = 400
GRID_H_RANGE = (18, 27)
GRID_W_RANGE = (18, 31)
LEGEND_N_RANGE = (2, 5)
FRAME_COUNT_RANGE = (1, 4)
FRAME_H_RANGE = (5, 10)
FRAME_W_RANGE = (5, 11)
FRAME_PADDING = 1
CORNERS = ("tl", "tr", "bl", "br")

ISLAND_SHAPES = (
    ((0, 0),),
    ((0, 0),),
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1)),
)


def _randint(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(lo, hi + 1))


def _mark_blocked(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int) -> None:
    h, w = mask.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _draw_legend(
    grid: np.ndarray,
    legend_colors: list[int],
    corner: str,
    rng: np.random.Generator,
) -> tuple[dict[int, int], tuple[int, int, int, int]]:
    h, w = grid.shape
    n = len(legend_colors)
    lengths = list(range(1, n + 1))
    if rng.random() < 0.5:
        lengths.reverse()

    if corner[0] == "t":
        rows = list(range(n))
    else:
        rows = list(range(h - n, h))

    for row, length in zip(rows, lengths):
        color = int(legend_colors[length - 1])
        if corner[1] == "l":
            grid[row, :length] = color
        else:
            grid[row, w - length :] = color

    if corner[1] == "l":
        bbox = (rows[0], 0, rows[-1], n - 1)
    else:
        bbox = (rows[0], w - n, rows[-1], w - 1)
    mapping = {idx + 1: int(color) for idx, color in enumerate(legend_colors)}
    return mapping, bbox


def _place_islands(
    ih: int,
    iw: int,
    count: int,
    colors: list[int],
    rng: np.random.Generator,
) -> np.ndarray | None:
    for _ in range(80):
        patch = np.zeros((ih, iw), dtype=int)
        occupied = np.zeros((ih, iw), dtype=bool)
        success = True
        for idx in range(count):
            placed = False
            for _ in range(250):
                if count - idx >= 3 and rng.random() < 0.7:
                    shape = ISLAND_SHAPES[_randint(rng, 0, 4)]
                else:
                    shape = ISLAND_SHAPES[_randint(rng, 0, len(ISLAND_SHAPES) - 1)]
                sh = max(r for r, _c in shape) + 1
                sw = max(c for _r, c in shape) + 1
                if sh > ih or sw > iw:
                    continue
                r0 = _randint(rng, 0, ih - sh)
                c0 = _randint(rng, 0, iw - sw)
                cells = [(r0 + dr, c0 + dc) for dr, dc in shape]
                cell_set = set(cells)

                valid = True
                for r, c in cells:
                    if occupied[r, c]:
                        valid = False
                        break
                    for rr in range(max(0, r - 1), min(ih, r + 2)):
                        for cc in range(max(0, c - 1), min(iw, c + 2)):
                            if occupied[rr, cc] and (rr, cc) not in cell_set:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break

                if not valid:
                    continue

                color = int(colors[_randint(rng, 0, len(colors) - 1)])
                for r, c in cells:
                    patch[r, c] = color
                    occupied[r, c] = True
                placed = True
                break

            if not placed:
                success = False
                break

        if success and np.count_nonzero(patch == BG) > 0:
            assert len(_binary_components(patch != BG)) == count
            return patch
    return None


def _frame_capacity(ih: int, iw: int) -> int:
    return ((ih + 1) // 2) * ((iw + 1) // 2)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = _randint(rng, *GRID_H_RANGE)
    w = _randint(rng, *GRID_W_RANGE)
    n_legend = _randint(rng, *LEGEND_N_RANGE)
    n_frames = _randint(rng, *FRAME_COUNT_RANGE)
    corner = CORNERS[_randint(rng, 0, len(CORNERS) - 1)]

    palette = list(map(int, rng.permutation(np.arange(1, 10))))
    border_color = palette[0]
    legend_colors = palette[1 : 1 + n_legend]
    island_colors = palette[1 + n_legend :]
    if len(island_colors) < 2:
        return None

    grid = np.zeros((h, w), dtype=int)
    legend, legend_bbox = _draw_legend(grid, legend_colors, corner, rng)

    blocked = np.zeros((h, w), dtype=bool)
    _mark_blocked(blocked, *legend_bbox, pad=1)

    frame_records: list[tuple[int, int, int, int, int]] = []
    for _frame_idx in range(n_frames):
        placed = False
        for _ in range(200):
            fh = _randint(rng, *FRAME_H_RANGE)
            fw = _randint(rng, *FRAME_W_RANGE)
            if fh >= h - 1 or fw >= w - 1:
                continue

            ih = fh - 2
            iw = fw - 2
            max_islands = min(n_legend, _frame_capacity(ih, iw))
            if max_islands < 1:
                continue
            if rng.random() < 0.3:
                island_count = max_islands
            else:
                island_count = _randint(rng, 1, max_islands)

            interior = _place_islands(ih, iw, island_count, island_colors, rng)
            if interior is None:
                continue

            r_lo = 1
            c_lo = 1
            r_hi = h - fh - 1
            c_hi = w - fw - 1
            if r_hi < r_lo or c_hi < c_lo:
                continue

            r0 = _randint(rng, r_lo, r_hi)
            c0 = _randint(rng, c_lo, c_hi)
            r1 = r0 + fh - 1
            c1 = c0 + fw - 1

            rr0 = max(0, r0 - FRAME_PADDING)
            cc0 = max(0, c0 - FRAME_PADDING)
            rr1 = min(h - 1, r1 + FRAME_PADDING)
            cc1 = min(w - 1, c1 + FRAME_PADDING)
            if np.any(blocked[rr0 : rr1 + 1, cc0 : cc1 + 1]):
                continue

            grid[r0, c0 : c1 + 1] = border_color
            grid[r1, c0 : c1 + 1] = border_color
            grid[r0 : r1 + 1, c0] = border_color
            grid[r0 : r1 + 1, c1] = border_color
            grid[r0 + 1 : r1, c0 + 1 : c1] = interior

            _mark_blocked(blocked, r0, c0, r1, c1, pad=FRAME_PADDING)
            frame_records.append((r0, c0, r1, c1, island_count))
            placed = True
            break

        if not placed:
            return None

    expected = grid.copy()
    for r0, c0, r1, c1, island_count in frame_records:
        patch = expected[r0 + 1 : r1, c0 + 1 : c1]
        src = grid[r0 + 1 : r1, c0 + 1 : c1]
        patch[src == BG] = int(legend[island_count])

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
