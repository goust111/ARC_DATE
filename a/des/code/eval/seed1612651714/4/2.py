from __future__ import annotations

import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _outside_is_bg(grid: np.ndarray, bg: int, r0: int, c0: int, r1: int, c1: int) -> bool:
    h, w = grid.shape
    if r0 > 0 and np.any(grid[r0 - 1, c0 : c1 + 1] != bg):
        return False
    if r1 + 1 < h and np.any(grid[r1 + 1, c0 : c1 + 1] != bg):
        return False
    if c0 > 0 and np.any(grid[r0 : r1 + 1, c0 - 1] != bg):
        return False
    if c1 + 1 < w and np.any(grid[r0 : r1 + 1, c1 + 1] != bg):
        return False
    return True


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _detect_rectangles(grid: np.ndarray) -> list[dict[str, int | str]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    candidates: list[dict[str, int | str]] = []

    for pts in _components(g != bg):
        rows = [r for r, _ in pts]
        cols = [c for _, c in pts]
        r0 = min(rows)
        c0 = min(cols)
        r1 = max(rows)
        c1 = max(cols)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        if not _outside_is_bg(g, bg, r0, c0, r1, c1):
            continue

        fence = int(g[r0, c0])
        if fence == bg:
            continue
        if int(g[r0, c1]) != fence or int(g[r1, c0]) != fence or int(g[r1, c1]) != fence:
            continue

        mismatches: list[tuple[int, int]] = []
        for c in range(c0 + 1, c1):
            if int(g[r0, c]) != fence:
                mismatches.append((r0, c))
            if int(g[r1, c]) != fence:
                mismatches.append((r1, c))
        for r in range(r0 + 1, r1):
            if int(g[r, c0]) != fence:
                mismatches.append((r, c0))
            if int(g[r, c1]) != fence:
                mismatches.append((r, c1))
        if len(mismatches) != 1:
            continue

        mr, mc = mismatches[0]
        marker = int(g[mr, mc])
        if marker in {bg, fence}:
            continue

        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        if interior.size == 0:
            continue
        if np.any(interior == fence):
            continue
        if not np.any(interior != bg):
            continue

        if mr == r0:
            side = "top"
        elif mr == r1:
            side = "bottom"
        elif mc == c0:
            side = "left"
        else:
            side = "right"

        candidates.append(
            {
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "mr": mr,
                "mc": mc,
                "side": side,
                "fence": fence,
                "area": (r1 - r0 + 1) * (c1 - c0 + 1),
            }
        )

    candidates.sort(key=lambda item: int(item["area"]), reverse=True)
    chosen: list[dict[str, int | str]] = []
    for cand in candidates:
        r0 = int(cand["r0"])
        c0 = int(cand["c0"])
        r1 = int(cand["r1"])
        c1 = int(cand["c1"])
        overlaps = False
        for other in chosen:
            or0 = int(other["r0"])
            oc0 = int(other["c0"])
            or1 = int(other["r1"])
            oc1 = int(other["c1"])
            if not (r1 < or0 or or1 < r0 or c1 < oc0 or oc1 < c0):
                overlaps = True
                break
        if not overlaps:
            chosen.append(cand)

    chosen.sort(key=lambda item: (int(item["r0"]), int(item["c0"])))
    return chosen


def _apply_single_fold(
    out: np.ndarray,
    source: np.ndarray,
    rect: dict[str, int | str],
    bg: int,
) -> None:
    h, w = source.shape
    r0 = int(rect["r0"])
    c0 = int(rect["c0"])
    r1 = int(rect["r1"])
    c1 = int(rect["c1"])
    mr = int(rect["mr"])
    mc = int(rect["mc"])
    fence = int(rect["fence"])
    side = str(rect["side"])

    if side in {"top", "bottom"}:
        for r in range(r0 + 1, r1):
            color = int(source[r, mc])
            if color in {bg, fence}:
                continue
            dst_r = 2 * r0 - r if side == "top" else 2 * r1 - r
            if 0 <= dst_r < h:
                out[dst_r, mc] = color
    else:
        for c in range(c0 + 1, c1):
            color = int(source[mr, c])
            if color in {bg, fence}:
                continue
            dst_c = 2 * c0 - c if side == "left" else 2 * c1 - c
            if 0 <= dst_c < w:
                out[mr, dst_c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """Fold the marker-selected interior strip outside its rectangular fence."""
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()
    for rect in _detect_rectangles(g):
        _apply_single_fold(out, g, rect, bg)
    return out


GRID_MIN = 14
GRID_MAX = 26
OBJECT_COUNT_CHOICES = (1, 2, 3)
SIDE_CHOICES = ("top", "bottom", "left", "right")
PATTERN_MODES = ("noise", "row_bands", "col_bands", "checker", "blocks")
MAX_GENERATION_ATTEMPTS = 400


def _choose_shape() -> tuple[int, int]:
    h = random.randint(GRID_MIN, GRID_MAX)
    if random.random() < 0.65:
        delta = random.choice((-3, -2, -1, 0, 1, 2, 3))
        w = min(GRID_MAX, max(GRID_MIN, h + delta))
    else:
        w = random.randint(GRID_MIN, GRID_MAX)
    return h, w


def _size_bounds(n_objects: int) -> tuple[tuple[int, int], tuple[int, int]]:
    if n_objects == 1:
        return (3, 7), (3, 7)
    if n_objects == 2:
        return (2, 6), (2, 6)
    return (2, 5), (2, 5)


def _make_pattern(
    height: int,
    width: int,
    palette: list[int],
    side: str,
    strip_index: int,
) -> np.ndarray:
    mode = random.choice(PATTERN_MODES)
    arr = np.empty((height, width), dtype=int)

    if mode == "noise":
        arr[:] = np.random.choice(palette, size=(height, width))
    elif mode == "row_bands":
        last = None
        for r in range(height):
            options = [c for c in palette if c != last] or palette
            color = random.choice(options)
            arr[r, :] = color
            last = color
    elif mode == "col_bands":
        last = None
        for c in range(width):
            options = [v for v in palette if v != last] or palette
            color = random.choice(options)
            arr[:, c] = color
            last = color
    elif mode == "checker":
        colors = np.asarray(palette, dtype=int)
        rr, cc = np.indices((height, width))
        phase = random.randrange(len(palette))
        row_step = random.choice((1, 2))
        col_step = random.choice((1, 2))
        arr[:] = colors[(phase + row_step * rr + col_step * cc) % len(colors)]
    else:
        base = random.choice(palette)
        arr[:] = base
        patch_count = random.randint(2, 5)
        for _ in range(patch_count):
            color = random.choice([c for c in palette if c != base] or palette)
            ph = random.randint(1, max(1, min(3, height)))
            pw = random.randint(1, max(1, min(3, width)))
            r0 = random.randint(0, height - ph)
            c0 = random.randint(0, width - pw)
            arr[r0 : r0 + ph, c0 : c0 + pw] = color

    if np.unique(arr).size == 1 and len(palette) > 1:
        rr = random.randrange(height)
        cc = random.randrange(width)
        current = int(arr[rr, cc])
        arr[rr, cc] = random.choice([c for c in palette if c != current])

    if side in {"top", "bottom"} and height > 1 and len(palette) > 1:
        values = np.unique(arr[:, strip_index])
        if values.size == 1:
            row = random.randrange(height)
            current = int(arr[row, strip_index])
            arr[row, strip_index] = random.choice([c for c in palette if c != current])
    if side in {"left", "right"} and width > 1 and len(palette) > 1:
        values = np.unique(arr[strip_index, :])
        if values.size == 1:
            col = random.randrange(width)
            current = int(arr[strip_index, col])
            arr[strip_index, col] = random.choice([c for c in palette if c != current])

    return arr


def _visible_fold_cells(
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    side: str,
    marker_offset: int,
    shape: tuple[int, int],
) -> set[tuple[int, int]]:
    h, w = shape
    cells: set[tuple[int, int]] = set()
    if side in {"top", "bottom"}:
        marker_c = c0 + 1 + marker_offset
        for r in range(r0 + 1, r1):
            dst_r = 2 * r0 - r if side == "top" else 2 * r1 - r
            if 0 <= dst_r < h:
                cells.add((dst_r, marker_c))
    else:
        marker_r = r0 + 1 + marker_offset
        for c in range(c0 + 1, c1):
            dst_c = 2 * c0 - c if side == "left" else 2 * c1 - c
            if 0 <= dst_c < w:
                cells.add((marker_r, dst_c))
    return cells


def _footprint(
    shape: tuple[int, int],
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    fold_cells: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    h, w = shape
    base = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
    }
    base |= fold_cells

    reserved: set[tuple[int, int]] = set()
    for r, c in base:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    reserved.add((nr, nc))
    return reserved


def _place_objects(
    shape: tuple[int, int],
    n_objects: int,
) -> list[dict[str, int | str | np.ndarray]] | None:
    h, w = shape
    occupied: set[tuple[int, int]] = set()
    objects: list[dict[str, int | str | np.ndarray]] = []
    (h_lo, h_hi), (w_lo, w_hi) = _size_bounds(n_objects)

    for _ in range(n_objects):
        placed = False
        for _attempt in range(300):
            inner_h = random.randint(h_lo, h_hi)
            inner_w = random.randint(w_lo, w_hi)
            outer_h = inner_h + 2
            outer_w = inner_w + 2
            if outer_h + 2 > h or outer_w + 2 > w:
                continue

            r0 = random.randint(1, h - outer_h - 1)
            c0 = random.randint(1, w - outer_w - 1)
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1
            side = random.choice(SIDE_CHOICES)
            marker_offset = random.randint(0, inner_w - 1) if side in {"top", "bottom"} else random.randint(0, inner_h - 1)
            fold_cells = _visible_fold_cells(r0, c0, r1, c1, side, marker_offset, shape)
            if not fold_cells:
                continue

            reserved = _footprint(shape, r0, c0, r1, c1, fold_cells)
            if reserved & occupied:
                continue

            objects.append(
                {
                    "r0": r0,
                    "c0": c0,
                    "r1": r1,
                    "c1": c1,
                    "side": side,
                    "marker_offset": marker_offset,
                }
            )
            occupied |= reserved
            placed = True
            break
        if not placed:
            return None

    return objects


def _render_input(
    shape: tuple[int, int],
    bg: int,
    fence: int,
    marker: int,
    palette: list[int],
    specs: list[dict[str, int | str | np.ndarray]],
) -> tuple[np.ndarray, list[dict[str, int | str]]]:
    grid = np.full(shape, bg, dtype=int)
    solver_specs: list[dict[str, int | str]] = []

    for spec in specs:
        r0 = int(spec["r0"])
        c0 = int(spec["c0"])
        r1 = int(spec["r1"])
        c1 = int(spec["c1"])
        side = str(spec["side"])
        marker_offset = int(spec["marker_offset"])
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1

        strip_index = marker_offset
        interior_palette = random.sample(palette, random.randint(2, min(4, len(palette))))
        interior = _make_pattern(inner_h, inner_w, interior_palette, side, strip_index)
        grid[r0 + 1 : r1, c0 + 1 : c1] = interior
        grid[r0, c0 : c1 + 1] = fence
        grid[r1, c0 : c1 + 1] = fence
        grid[r0 : r1 + 1, c0] = fence
        grid[r0 : r1 + 1, c1] = fence

        if side == "top":
            mr, mc = r0, c0 + 1 + marker_offset
        elif side == "bottom":
            mr, mc = r1, c0 + 1 + marker_offset
        elif side == "left":
            mr, mc = r0 + 1 + marker_offset, c0
        else:
            mr, mc = r0 + 1 + marker_offset, c1
        grid[mr, mc] = marker

        solver_specs.append(
            {
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "mr": mr,
                "mc": mc,
                "side": side,
                "fence": fence,
            }
        )

    return grid, solver_specs


def _expected_output(grid: np.ndarray, specs: list[dict[str, int | str]]) -> np.ndarray:
    out = grid.copy()
    bg = _get_bg(grid)
    for spec in specs:
        _apply_single_fold(out, grid, spec, bg)
    return out


def _gen_attempt() -> np.ndarray | None:
    shape = _choose_shape()
    n_objects = random.choice(OBJECT_COUNT_CHOICES)
    placed = _place_objects(shape, n_objects)
    if placed is None:
        return None

    colors = random.sample(list(range(10)), 7)
    bg = colors[0]
    fence = colors[1]
    marker = colors[2]
    palette = colors[3:]

    grid, specs = _render_input(shape, bg, fence, marker, palette, placed)
    if _get_bg(grid) != bg:
        return None

    expected = _expected_output(grid, specs)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if len(_detect_rectangles(grid)) != n_objects:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid folding-fence input")
