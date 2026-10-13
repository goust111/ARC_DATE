from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

try:
    _UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
    if _UTILS_DIR.is_dir() and str(_UTILS_DIR) not in sys.path:
        sys.path.append(str(_UTILS_DIR))
    from arc_utils import Color, in_bounds  # type: ignore
except Exception:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        YELLOW = 4
        GREY = 5
        PINK = 6

    def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
        return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


BG = int(Color.BLACK)
BARRIER = int(Color.GREY)
CENTER = int(Color.PINK)
MARKER_COLORS = (int(Color.BLUE), int(Color.RED), int(Color.GREEN), int(Color.YELLOW))

_GEN_SHAPES = (
    (1, 1),
    (1, 2),
    (2, 1),
    (2, 2),
)
_GEN_SHAPE_WEIGHTS = (8, 5, 5, 1)
_GEN_RETRIES = 400


def _gray_runs(line: np.ndarray) -> list[tuple[int, int]]:
    cols = [int(i) for i in np.flatnonzero(line == BARRIER)]
    if not cols:
        return []
    runs: list[tuple[int, int]] = []
    start = end = cols[0]
    for c in cols[1:]:
        if c == end + 1:
            end = c
        else:
            runs.append((start, end))
            start = end = c
    runs.append((start, end))
    return runs


def _marker_cells(grid: np.ndarray) -> list[tuple[int, int]]:
    coords = np.argwhere((grid != BG) & (grid != BARRIER))
    return [tuple(map(int, rc)) for rc in coords]


def _collect_pairs(grid: np.ndarray) -> tuple[list[dict], set[tuple[int, int]], set[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    markers = set(_marker_cells(g))
    pairs: list[dict] = []

    for r in range(H):
        runs = _gray_runs(g[r, :])
        if len(runs) != 1:
            continue
        c0, c1 = runs[0]
        left = [(r, c) for c in range(c0) if (r, c) in markers]
        right = [(r, c) for c in range(c1 + 1, W) if (r, c) in markers]
        if len(left) == 1 and len(right) == 1:
            (lr, lc), (rr, rc) = left[0], right[0]
            pairs.append(
                {
                    "axis": "row",
                    "left": (lr, lc),
                    "right": (rr, rc),
                    "left_color": int(g[lr, lc]),
                    "right_color": int(g[rr, rc]),
                    "barrier": (r, c0, c1),
                }
            )

    for c in range(W):
        runs = _gray_runs(g[:, c])
        if len(runs) != 1:
            continue
        r0, r1 = runs[0]
        top = [(r, c) for r in range(r0) if (r, c) in markers]
        bottom = [(r, c) for r in range(r1 + 1, H) if (r, c) in markers]
        if len(top) == 1 and len(bottom) == 1:
            (tr, tc), (br, bc) = top[0], bottom[0]
            pairs.append(
                {
                    "axis": "col",
                    "top": (tr, tc),
                    "bottom": (br, bc),
                    "top_color": int(g[tr, tc]),
                    "bottom_color": int(g[br, bc]),
                    "barrier": (r0, r1, c),
                }
            )

    usage: dict[tuple[int, int], int] = {}
    for pair in pairs:
        pts = [pair["left"], pair["right"]] if pair["axis"] == "row" else [pair["top"], pair["bottom"]]
        for pt in pts:
            usage[pt] = usage.get(pt, 0) + 1
    assert all(count == 1 for count in usage.values()), "every paired marker must belong to exactly one pair"

    paired = set(usage)
    isolated = markers - paired
    return pairs, paired, isolated


def _emitters_from_pairs(pairs: list[dict]) -> list[dict]:
    emitters: list[dict] = []
    for pair in pairs:
        if pair["axis"] == "row":
            r, c0, c1 = pair["barrier"]
            left_dst = (int(r), int(c1) + 1)
            right_dst = (int(r), int(c0) - 1)
            assert left_dst[1] > c1 and right_dst[1] < c0
            emitters.append({"pos": left_dst, "ray_color": int(pair["right_color"])})
            emitters.append({"pos": right_dst, "ray_color": int(pair["left_color"])})
        else:
            r0, r1, c = pair["barrier"]
            top_dst = (int(r1) + 1, int(c))
            bottom_dst = (int(r0) - 1, int(c))
            assert top_dst[0] > r1 and bottom_dst[0] < r0
            emitters.append({"pos": top_dst, "ray_color": int(pair["bottom_color"])})
            emitters.append({"pos": bottom_dst, "ray_color": int(pair["top_color"])})

    centers = [tuple(e["pos"]) for e in emitters]
    assert len(centers) == len(set(centers)), "swapped marker centers must be distinct"
    return emitters


def _nearest_axis_color(emitters: list[dict], fixed: int, moving: int, axis: str) -> int | None:
    if not emitters:
        return None
    if axis == "row":
        best = min(
            emitters,
            key=lambda e: (abs(int(moving) - int(e["pos"][1])), int(e["pos"][1])),
        )
    else:
        best = min(
            emitters,
            key=lambda e: (abs(int(moving) - int(e["pos"][0])), int(e["pos"][0])),
        )
    return int(best["ray_color"])


def _render_from_pairs(shape: tuple[int, int], barrier_mask: np.ndarray, pairs: list[dict]) -> np.ndarray:
    H, W = map(int, shape)
    out = np.full((H, W), BG, dtype=int)
    out[barrier_mask] = BARRIER

    emitters = _emitters_from_pairs(pairs)
    centers = {tuple(e["pos"]) for e in emitters}
    row_emitters: dict[int, list[dict]] = {}
    col_emitters: dict[int, list[dict]] = {}
    for emitter in emitters:
        r, c = map(int, emitter["pos"])
        row_emitters.setdefault(r, []).append(emitter)
        col_emitters.setdefault(c, []).append(emitter)

    for r, row_list in row_emitters.items():
        for c in range(W):
            if barrier_mask[r, c] or (r, c) in centers:
                continue
            color = _nearest_axis_color(row_list, r, c, "row")
            if color is not None:
                out[r, c] = int(color)

    for c, col_list in col_emitters.items():
        for r in range(H):
            if barrier_mask[r, c] or (r, c) in centers:
                continue
            color = _nearest_axis_color(col_list, c, r, "col")
            if color is not None:
                out[r, c] = int(color)

    for r, c in centers:
        out[r, c] = CENTER
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap each across-barrier marker pair and draw partner-colored cross rays from the swapped positions.

    Input structure:
    - A black background contains several disjoint gray horizontal or vertical barrier segments.
    - Non-gray non-black single pixels are colored markers.
    - A valid pair is formed when one whole row or one whole column contains exactly one gray segment and exactly one marker on each side of that segment.

    Rule:
    - Detect every valid row pair and column pair.
    - Erase all original markers; markers that do not belong to any pair disappear completely.
    - Move each paired marker to the first cell immediately across the barrier on the partner side.
    - From each swapped position, extend horizontal and vertical rays to the grid edge using the paired partner's original color; gray barriers stay gray, and swapped centers become magenta.
    - When a horizontal ray and a vertical ray both reach the same non-barrier cell, keep the vertical ray color there.

    Key insight:
    - The swapped location comes from the marker itself, but the ray color comes from its original partner, so the output only becomes determined after combining the swap relation with the pairing relation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    pairs, _paired, _isolated = _collect_pairs(g)
    barrier_mask = g == BARRIER
    return _render_from_pairs(g.shape, barrier_mask, pairs)


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
    ]


def _cells_with_halo(cells: list[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    H, W = shape
    halo = set()
    for r, c in cells:
        halo.add((r, c))
        for rr, cc in _neighbors8(r, c):
            if 0 <= rr < H and 0 <= cc < W:
                halo.add((rr, cc))
    return halo


def _pick_shape() -> tuple[int, int]:
    return random.choices(_GEN_SHAPES, weights=_GEN_SHAPE_WEIGHTS, k=1)[0]


def _pick_pair_colors(n_pairs: int) -> list[tuple[int, int]]:
    tokens = [color for color in MARKER_COLORS for _ in range(2)]
    need = 2 * int(n_pairs)
    for _ in range(80):
        picked = random.sample(tokens, need)
        pairs = list(zip(picked[::2], picked[1::2]))
        if all(a != b for a, b in pairs):
            random.shuffle(pairs)
            return pairs
    raise RuntimeError("failed to sample distinct colors for all pairs")


def _horizontal_cells(r: int, c0: int, c1: int) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for c in range(int(c0), int(c1) + 1)]


def _vertical_cells(c: int, r0: int, r1: int) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r in range(int(r0), int(r1) + 1)]


def _segment_clear(cells: list[tuple[int, int]], halo_used: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    H, W = shape
    for r, c in cells:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if (r, c) in halo_used:
            return False
    return True


def _place_row_pairs(
    shape: tuple[int, int],
    n_row_pairs: int,
    barrier: np.ndarray,
    halo_used: set[tuple[int, int]],
) -> list[dict] | None:
    H, W = shape
    pairs: list[dict] = []
    center_cols: set[int] = set()
    for _ in range(n_row_pairs):
        placed = False
        for _attempt in range(120):
            r = random.randint(2, H - 3)
            length = random.randint(3, min(6, W - 4))
            c0 = random.randint(2, W - length - 2)
            c1 = c0 + length - 1
            left_center = c0 - 1
            right_center = c1 + 1
            if left_center in center_cols or right_center in center_cols:
                continue
            cells = _horizontal_cells(r, c0, c1)
            if not _segment_clear(cells, halo_used, shape):
                continue
            barrier[r, c0 : c1 + 1] = BARRIER
            halo_used |= _cells_with_halo(cells, shape)
            center_cols.update((left_center, right_center))
            pairs.append({"row": r, "c0": c0, "c1": c1})
            placed = True
            break
        if not placed:
            return None
    return pairs


def _place_col_pairs(
    shape: tuple[int, int],
    n_col_pairs: int,
    barrier: np.ndarray,
    halo_used: set[tuple[int, int]],
    row_pair_rows: set[int],
) -> list[dict] | None:
    H, W = shape
    pairs: list[dict] = []
    center_rows: set[int] = set()
    blocked_cols = {int(c) for c in np.flatnonzero(barrier.any(axis=0))}
    for _ in range(n_col_pairs):
        placed = False
        for _attempt in range(160):
            c = random.randint(2, W - 3)
            if c in blocked_cols:
                continue
            length = random.randint(3, min(6, H - 4))
            r0 = random.randint(2, H - length - 2)
            r1 = r0 + length - 1
            if any(r0 <= rr <= r1 for rr in row_pair_rows):
                continue
            top_center = r0 - 1
            bottom_center = r1 + 1
            if top_center in center_rows or bottom_center in center_rows:
                continue
            cells = _vertical_cells(c, r0, r1)
            if not _segment_clear(cells, halo_used, shape):
                continue
            barrier[r0 : r1 + 1, c] = BARRIER
            halo_used |= _cells_with_halo(cells, shape)
            center_rows.update((top_center, bottom_center))
            blocked_cols.add(c)
            pairs.append({"col": c, "r0": r0, "r1": r1})
            placed = True
            break
        if not placed:
            return None
    return pairs


def _place_extra_h(
    shape: tuple[int, int],
    count: int,
    barrier: np.ndarray,
    halo_used: set[tuple[int, int]],
) -> bool:
    H, W = shape
    blocked_rows = {int(r) for r in np.flatnonzero(barrier.any(axis=1))}
    for _ in range(count):
        placed = False
        for _attempt in range(120):
            r = random.randint(1, H - 2)
            if r in blocked_rows:
                continue
            length = random.randint(3, min(6, W - 2))
            c0 = random.randint(1, W - length - 1)
            c1 = c0 + length - 1
            cells = _horizontal_cells(r, c0, c1)
            if not _segment_clear(cells, halo_used, shape):
                continue
            barrier[r, c0 : c1 + 1] = BARRIER
            halo_used |= _cells_with_halo(cells, shape)
            blocked_rows.add(r)
            placed = True
            break
        if not placed:
            return False
    return True


def _place_extra_v(
    shape: tuple[int, int],
    count: int,
    barrier: np.ndarray,
    halo_used: set[tuple[int, int]],
) -> bool:
    H, W = shape
    blocked_cols = {int(c) for c in np.flatnonzero(barrier.any(axis=0))}
    blocked_rows = {int(r) for r in np.flatnonzero(barrier.any(axis=1))}
    for _ in range(count):
        placed = False
        for _attempt in range(120):
            c = random.randint(1, W - 2)
            if c in blocked_cols:
                continue
            length = random.randint(3, min(6, H - 2))
            r0 = random.randint(1, H - length - 1)
            r1 = r0 + length - 1
            if any(r in blocked_rows for r in range(r0, r1 + 1)):
                continue
            cells = _vertical_cells(c, r0, r1)
            if not _segment_clear(cells, halo_used, shape):
                continue
            barrier[r0 : r1 + 1, c] = BARRIER
            halo_used |= _cells_with_halo(cells, shape)
            blocked_cols.add(c)
            placed = True
            break
        if not placed:
            return False
    return True


def _free_rows_cols(barrier: np.ndarray) -> tuple[list[int], list[int]]:
    safe_rows = [int(r) for r in range(barrier.shape[0]) if not barrier[r, :].any()]
    safe_cols = [int(c) for c in range(barrier.shape[1]) if not barrier[:, c].any()]
    return safe_rows, safe_cols


def _place_pair_markers(
    grid: np.ndarray,
    row_pairs: list[dict],
    col_pairs: list[dict],
    pair_colors: list[tuple[int, int]],
) -> tuple[list[dict], dict[int, int]] | None:
    barrier = grid == BARRIER
    safe_rows, safe_cols = _free_rows_cols(barrier)
    placed_pairs: list[dict] = []
    counts: dict[int, int] = {color: 0 for color in MARKER_COLORS}
    color_iter = iter(pair_colors)

    for pair in row_pairs:
        color_left, color_right = next(color_iter)
        r, c0, c1 = int(pair["row"]), int(pair["c0"]), int(pair["c1"])
        left_choices = [c for c in safe_cols if 1 <= c <= c0 - 2 and grid[r, c] == BG]
        right_choices = [c for c in safe_cols if c1 + 2 <= c <= grid.shape[1] - 2 and grid[r, c] == BG]
        if not left_choices or not right_choices:
            return None
        lc = random.choice(left_choices)
        rc = random.choice(right_choices)
        grid[r, lc] = color_left
        grid[r, rc] = color_right
        counts[color_left] += 1
        counts[color_right] += 1
        placed_pairs.append(
            {
                "axis": "row",
                "barrier": (r, c0, c1),
                "markers": [((r, lc), color_left), ((r, rc), color_right)],
            }
        )

    for pair in col_pairs:
        color_top, color_bottom = next(color_iter)
        c, r0, r1 = int(pair["col"]), int(pair["r0"]), int(pair["r1"])
        top_choices = [r for r in safe_rows if 1 <= r <= r0 - 2 and grid[r, c] == BG]
        bottom_choices = [r for r in safe_rows if r1 + 2 <= r <= grid.shape[0] - 2 and grid[r, c] == BG]
        if not top_choices or not bottom_choices:
            return None
        tr = random.choice(top_choices)
        br = random.choice(bottom_choices)
        grid[tr, c] = color_top
        grid[br, c] = color_bottom
        counts[color_top] += 1
        counts[color_bottom] += 1
        placed_pairs.append(
            {
                "axis": "col",
                "barrier": (r0, r1, c),
                "markers": [((tr, c), color_top), ((br, c), color_bottom)],
            }
        )

    return placed_pairs, counts


def _place_isolates(grid: np.ndarray, counts: dict[int, int]) -> int:
    barrier = grid == BARRIER
    safe_rows, safe_cols = _free_rows_cols(barrier)
    cells = [(r, c) for r in safe_rows for c in safe_cols if grid[r, c] == BG]
    random.shuffle(cells)
    extra_colors = [color for color, used in counts.items() for _ in range(2 - used)]
    random.shuffle(extra_colors)
    max_iso = min(2, len(cells), len(extra_colors))
    if max_iso == 0:
        return 0
    n_iso = random.choices([0, 1, 2][: max_iso + 1], weights=[2, 5, 3][: max_iso + 1], k=1)[0]
    placed = 0
    for color, (r, c) in zip(extra_colors, cells):
        if placed >= n_iso:
            break
        grid[r, c] = int(color)
        counts[int(color)] += 1
        placed += 1
    return placed


def _pair_signature(pairs: list[dict]) -> set[tuple]:
    out = set()
    for pair in pairs:
        if pair["axis"] == "row":
            pts = tuple(sorted((pair["left"], pair["right"])))
            out.add(("row", pair["barrier"], pts))
        else:
            pts = tuple(sorted((pair["top"], pair["bottom"])))
            out.add(("col", pair["barrier"], pts))
    return out


def _semantic_checks(grid: np.ndarray, planned_pairs: list[dict], expected_isolates: int) -> bool:
    try:
        pairs, _paired, isolated = _collect_pairs(grid)
    except AssertionError:
        return False
    if len(isolated) != expected_isolates:
        return False

    wanted = set()
    for pair in planned_pairs:
        if pair["axis"] == "row":
            wanted.add(("row", pair["barrier"], tuple(sorted(pt for pt, _ in pair["markers"]))))
        else:
            wanted.add(("col", pair["barrier"], tuple(sorted(pt for pt, _ in pair["markers"]))))
    if _pair_signature(pairs) != wanted:
        return False

    out = main(grid)
    if np.any(out[grid == BARRIER] != BARRIER):
        return False

    emitters = _emitters_from_pairs(pairs)
    centers = {tuple(e["pos"]) for e in emitters}
    for emitter in emitters:
        if out[tuple(emitter["pos"])] != CENTER:
            return False
        er, ec = map(int, emitter["pos"])
        color = int(emitter["ray_color"])
        visible = False
        for cc in range(out.shape[1]):
            if (er, cc) in centers or grid[er, cc] == BARRIER:
                continue
            if out[er, cc] == color:
                visible = True
                break
        if not visible:
            for rr in range(out.shape[0]):
                if (rr, ec) in centers or grid[rr, ec] == BARRIER:
                    continue
                if out[rr, ec] == color:
                    visible = True
                    break
        if not visible:
            return False

    mixed_rows = any(pair["axis"] == "row" for pair in pairs)
    mixed_cols = any(pair["axis"] == "col" for pair in pairs)
    if mixed_rows and mixed_cols:
        row_emitters = {}
        col_emitters = {}
        for emitter in emitters:
            er, ec = emitter["pos"]
            row_emitters.setdefault(er, []).append(emitter)
            col_emitters.setdefault(ec, []).append(emitter)
        visible_override = False
        for r, row_list in row_emitters.items():
            for c, col_list in col_emitters.items():
                if (r, c) in centers or grid[r, c] == BARRIER:
                    continue
                h_col = _nearest_axis_color(row_list, r, c, "row")
                v_col = _nearest_axis_color(col_list, c, r, "col")
                if h_col is None or v_col is None or h_col == v_col:
                    continue
                if out[r, c] == v_col:
                    visible_override = True
                    break
            if visible_override:
                break
        if not visible_override:
            return False

    return True


def generate_input() -> np.ndarray:
    for _attempt in range(_GEN_RETRIES):
        H = random.randint(16, 24)
        W = random.randint(16, 24)
        grid = np.full((H, W), BG, dtype=int)
        barrier = np.zeros((H, W), dtype=int)
        halo_used: set[tuple[int, int]] = set()

        n_row_pairs, n_col_pairs = _pick_shape()
        row_pairs = _place_row_pairs((H, W), n_row_pairs, barrier, halo_used)
        if row_pairs is None:
            continue
        col_pairs = _place_col_pairs((H, W), n_col_pairs, barrier, halo_used, {pair["row"] for pair in row_pairs})
        if col_pairs is None:
            continue

        grid[barrier == BARRIER] = BARRIER
        pair_colors = _pick_pair_colors(n_row_pairs + n_col_pairs)
        placed = _place_pair_markers(grid, row_pairs, col_pairs, pair_colors)
        if placed is None:
            continue
        planned_pairs, used_counts = placed
        counts = used_counts
        n_isolates = _place_isolates(grid, counts)
        if any(count > 2 for count in counts.values()):
            continue
        if len(np.unique(grid)) < 4:
            continue
        if not _semantic_checks(grid, planned_pairs, n_isolates):
            continue
        return grid

    raise RuntimeError("failed to generate a valid input")
