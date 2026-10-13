from __future__ import annotations

from collections import Counter, deque

import numpy as np


BG = 0
MAROON = 9
_RNG = np.random.default_rng()

GENERATE_RETRIES = 300
PATCH_RETRIES = 120
DEFECT_RETRIES = 200


Cell = tuple[int, int]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _mirror_cell(cell: Cell, axis_kind: str, axis2: int) -> Cell:
    r, c = cell
    if axis_kind == "vertical":
        return r, axis2 - c
    return axis2 - r, c


def _detect_marker_side(grid: np.ndarray) -> str:
    if np.all(grid[0] == MAROON):
        return "top"
    if np.all(grid[-1] == MAROON):
        return "bottom"
    if np.all(grid[:, 0] == MAROON):
        return "left"
    if np.all(grid[:, -1] == MAROON):
        return "right"
    raise AssertionError("expected one full maroon edge line")


def _score_axis(colors: dict[Cell, int], shape: tuple[int, int], axis_kind: str, axis2: int) -> tuple[int, int]:
    seen: set[tuple[Cell, Cell]] = set()
    mismatches = 0
    agreements = 0

    for cell, value in colors.items():
        mirror = _mirror_cell(cell, axis_kind, axis2)
        if not _in_bounds(shape, *mirror):
            mismatches += 1
            continue
        pair = (cell, mirror) if cell <= mirror else (mirror, cell)
        if pair in seen:
            continue
        seen.add(pair)
        if value == colors.get(mirror, BG):
            agreements += 1
        else:
            mismatches += 1
    return mismatches, agreements


def _best_axis(colors: dict[Cell, int], shape: tuple[int, int], axis_kind: str) -> int:
    if axis_kind == "vertical":
        center_ref = min(c for _, c in colors) + max(c for _, c in colors)
        candidates = range(0, 2 * shape[1] - 1)
    else:
        center_ref = min(r for r, _ in colors) + max(r for r, _ in colors)
        candidates = range(0, 2 * shape[0] - 1)

    best_axis = None
    best_key = None
    for axis2 in candidates:
        mismatches, agreements = _score_axis(colors, shape, axis_kind, axis2)
        key = (mismatches, -agreements, abs(axis2 - center_ref))
        if best_key is None or key < best_key:
            best_key = key
            best_axis = axis2
    assert best_axis is not None
    return int(best_axis)


def _repair_by_symmetry(colors: dict[Cell, int], shape: tuple[int, int], axis_kind: str, axis2: int) -> dict[Cell, int]:
    counts = Counter(colors.values())
    candidate_cells = set(colors)
    for cell in colors:
        mirror = _mirror_cell(cell, axis_kind, axis2)
        if _in_bounds(shape, *mirror):
            candidate_cells.add(mirror)

    repaired: dict[Cell, int] = {}
    seen: set[tuple[Cell, Cell]] = set()
    for cell in sorted(candidate_cells):
        mirror = _mirror_cell(cell, axis_kind, axis2)
        if not _in_bounds(shape, *mirror):
            continue
        pair = (cell, mirror) if cell <= mirror else (mirror, cell)
        if pair in seen:
            continue
        seen.add(pair)

        a, b = pair
        va = colors.get(a, BG)
        vb = colors.get(b, BG)

        if va == BG and vb == BG:
            continue
        if va == BG:
            chosen = vb
        elif vb == BG:
            chosen = va
        elif va == vb:
            chosen = va
        else:
            ca = counts[va]
            cb = counts[vb]
            chosen = va if ca >= cb else vb

        repaired[a] = chosen
        repaired[b] = chosen
    return repaired


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if _in_bounds(mask.shape, rr, cc) and mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                q.append((rr, cc))
    return len(seen) == int(mask.sum())


def _symmetry_pairs(mask: np.ndarray, axis_kind: str, axis2: int) -> list[tuple[Cell, Cell]]:
    pairs: list[tuple[Cell, Cell]] = []
    for r0, c0 in np.argwhere(mask):
        cell = (int(r0), int(c0))
        mirror = _mirror_cell(cell, axis_kind, axis2)
        if not _in_bounds(mask.shape, *mirror) or not mask[mirror]:
            continue
        if cell < mirror:
            pairs.append((cell, mirror))
    return pairs


def _mismatch_count(arr: np.ndarray, axis_kind: str, axis2: int) -> int:
    colors = {
        (int(r), int(c)): int(arr[r, c])
        for r, c in np.argwhere(arr != BG)
    }
    return _score_axis(colors, arr.shape, axis_kind, axis2)[0]


def _width_profile(height: int, half_width: int) -> np.ndarray:
    widths = np.empty(height, dtype=int)
    widths[0] = int(_RNG.integers(max(2, half_width // 2), half_width + 1))
    for r in range(1, height):
        lo = max(1, widths[r - 1] - 2)
        hi = min(half_width, widths[r - 1] + 2)
        widths[r] = int(_RNG.integers(lo, hi + 1))
    if widths.max() < min(half_width, 3):
        widths[int(_RNG.integers(0, height))] = min(half_width, 3)
    return widths


def _build_vertical_patch() -> tuple[np.ndarray, int]:
    for _ in range(PATCH_RETRIES):
        height = int(_RNG.integers(6, 13))
        half_width = int(_RNG.integers(3, 8))
        odd_axis = bool(_RNG.integers(0, 2))
        widths = _width_profile(height, half_width)

        width = 2 * half_width + (1 if odd_axis else 0)
        patch = np.zeros((height, width), dtype=int)

        palette = [c for c in range(1, 9)]
        base, outer, band = map(int, _RNG.choice(palette, size=3, replace=False))

        for r, row_width in enumerate(widths):
            row_len = 2 * row_width + (1 if odd_axis else 0)
            start = (width - row_len) // 2
            stop = start + row_len
            patch[r, start:stop] = base
            patch[r, start] = outer
            patch[r, stop - 1] = outer

        eligible = [r for r in range(height) if int(np.count_nonzero(patch[r])) >= 5]
        if eligible:
            band_start = int(_RNG.choice(eligible))
            band_len = int(_RNG.integers(1, min(4, height - band_start) + 1))
            for r in range(band_start, band_start + band_len):
                cols = np.flatnonzero(patch[r])
                if cols.size > 2:
                    patch[r, cols[1:-1]] = band

        counts = Counter(int(v) for v in patch[patch != BG])
        if len(counts) < 2:
            continue
        if counts.most_common(1)[0][0] != base:
            continue
        if not _is_connected(patch != BG):
            continue

        axis2 = 2 * half_width if odd_axis else 2 * half_width - 1
        if len(_symmetry_pairs(patch != BG, "vertical", axis2)) < 4:
            continue
        return patch, axis2

    raise RuntimeError("failed to build a valid symmetric patch")


def _damage_patch(clean: np.ndarray, axis_kind: str, axis2: int) -> np.ndarray | None:
    damaged = clean.copy()
    used_colors = {int(v) for v in clean[clean != BG]}
    spare_colors = [c for c in range(1, 9) if c not in used_colors]
    pairs = _symmetry_pairs(clean != BG, axis_kind, axis2)
    if len(pairs) < 4 or not spare_colors:
        return None

    defect_count = min(len(pairs), int(_RNG.integers(1, 5)))
    if defect_count == 1:
        defect_types = [str(_RNG.choice(["missing", "recolor"]))]
    else:
        extras = [str(_RNG.choice(["missing", "recolor"])) for _ in range(defect_count - 2)]
        defect_types = ["missing", "recolor"] + extras
        _RNG.shuffle(defect_types)

    used_pairs: set[tuple[Cell, Cell]] = set()
    for defect_type in defect_types:
        placed = False
        for _ in range(DEFECT_RETRIES):
            pair = pairs[int(_RNG.integers(0, len(pairs)))]
            if pair in used_pairs:
                continue
            cell = pair[int(_RNG.integers(0, 2))]
            other = pair[0] if cell == pair[1] else pair[1]
            trial = damaged.copy()

            if defect_type == "missing":
                trial[cell] = BG
            else:
                forbidden = {int(damaged[cell]), int(damaged[other])}
                choices = [c for c in spare_colors if c not in forbidden]
                if not choices:
                    continue
                trial[cell] = int(_RNG.choice(choices))

            if not _is_connected(trial != BG):
                continue

            damaged = trial
            used_pairs.add(pair)
            placed = True
            break

        if not placed:
            return None

    if _mismatch_count(damaged, axis_kind, axis2) != defect_count:
        return None
    return damaged


def _apply_side_line(grid: np.ndarray, side: str) -> None:
    if side == "top":
        grid[0, :] = MAROON
    elif side == "bottom":
        grid[-1, :] = MAROON
    elif side == "left":
        grid[:, 0] = MAROON
    else:
        grid[:, -1] = MAROON


def _choose_position(limit: int, size: int, side: str) -> int:
    max_start = limit - size - 1
    assert max_start >= 1
    if side in ("left", "top"):
        lo = max(2, (limit - size) // 2)
        hi = max_start
    else:
        lo = 1
        hi = min(max_start - 1, (limit - size) // 2)
    if hi < lo:
        lo = 2
        hi = max(lo, max_start)
    return int(_RNG.integers(lo, hi + 1))


def _place_case(clean: np.ndarray, damaged: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray]:
    ph, pw = clean.shape
    if side in ("left", "right"):
        H = int(_RNG.integers(ph + 3, min(31, ph + 12)))
        W = int(_RNG.integers(pw + 5, min(31, pw + 16)))
        top = int(_RNG.integers(1, H - ph))
        left = _choose_position(W, pw, side)
        dest_top = top
        dest_left = 1 if side == "left" else W - pw - 1
    else:
        H = int(_RNG.integers(ph + 5, min(31, ph + 16)))
        W = int(_RNG.integers(pw + 3, min(31, pw + 12)))
        top = _choose_position(H, ph, side)
        left = int(_RNG.integers(1, W - pw))
        dest_top = 1 if side == "top" else H - ph - 1
        dest_left = left

    inp = np.zeros((H, W), dtype=int)
    out = np.zeros((H, W), dtype=int)
    _apply_side_line(inp, side)
    _apply_side_line(out, side)

    in_view = inp[top : top + ph, left : left + pw]
    in_view[damaged != BG] = damaged[damaged != BG]

    out_view = out[dest_top : dest_top + ph, dest_left : dest_left + pw]
    out_view[clean != BG] = clean[clean != BG]
    return inp, out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair a nearly symmetric multicolor object from its mirror side and slide it next to the maroon edge line.

    Input structure:
    - bg = black.
    - One large 4-connected multicolor object occupies the interior and is almost symmetric about a horizontal or vertical axis.
    - A full 1-pixel maroon line lies on exactly one grid edge and is parallel to the true symmetry axis.

    Rule:
    - Use the maroon edge line to determine whether the hidden symmetry axis is horizontal or vertical.
    - Search that orientation for the axis with the fewest mirror mismatches, then repair every broken mirror pair by copying the non-background mirror color; if both sides are colored but disagree, keep the color that appears more often in the whole object.
    - Translate the repaired object perpendicular to the symmetry axis until it sits exactly one cell inside the maroon line, clear the old position, and keep the maroon line unchanged.

    Key insight:
    - The edge marker fixes the axis direction and the translation target, while the object's near-symmetry reveals the exact axis position and supplies the missing colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    side = _detect_marker_side(g)
    axis_kind = "vertical" if side in ("left", "right") else "horizontal"
    colors = {
        (int(r), int(c)): int(g[r, c])
        for r, c in np.argwhere((g != BG) & (g != MAROON))
    }
    assert colors, "expected one interior object"

    axis2 = _best_axis(colors, g.shape, axis_kind)
    repaired = _repair_by_symmetry(colors, g.shape, axis_kind, axis2)

    rows = [r for r, _ in repaired]
    cols = [c for _, c in repaired]
    if side == "left":
        dr, dc = 0, 1 - min(cols)
    elif side == "right":
        dr, dc = 0, g.shape[1] - 2 - max(cols)
    elif side == "top":
        dr, dc = 1 - min(rows), 0
    else:
        dr, dc = g.shape[0] - 2 - max(rows), 0

    out = np.zeros_like(g)
    out[g == MAROON] = MAROON
    for (r, c), color in repaired.items():
        rr = r + dr
        cc = c + dc
        assert _in_bounds(g.shape, rr, cc)
        out[rr, cc] = color
    return out


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        clean, axis2 = _build_vertical_patch()
        damaged = _damage_patch(clean, "vertical", axis2)
        if damaged is None:
            continue

        if bool(_RNG.integers(0, 2)):
            clean_case = clean
            damaged_case = damaged
            side = str(_RNG.choice(["left", "right"]))
        else:
            clean_case = clean.T.copy()
            damaged_case = damaged.T.copy()
            side = str(_RNG.choice(["top", "bottom"]))

        inp, expected = _place_case(clean_case, damaged_case, side)
        if np.array_equal(main(inp), expected):
            return inp
    raise RuntimeError("failed to generate a valid input")
