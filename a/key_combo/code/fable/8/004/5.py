from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np


def get_bg(grid: np.ndarray) -> int:
    """Return the most common color in the grid."""
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    """Return the inclusive bbox of an (N,2) coords array."""
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Crop a grid to an inclusive bbox."""
    r0, c0, r1, c1 = map(int, box)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def neighbors8(r: int, c: int) -> Iterable[tuple[int, int]]:
    """Yield 8-neighbors of a cell using the shared ARC (r,c) convention."""
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def find_components_mask(mask: np.ndarray, *, connectivity: int = 8) -> list[np.ndarray]:
    """Return connected components of a boolean mask as coord arrays."""
    height, width = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    if connectivity == 8:
        deltas = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
    elif connectivity == 4:
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        raise ValueError("connectivity must be 4 or 8")

    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        stack = [(sr, sc)]
        seen[sr, sc] = True
        coords: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            coords.append((r, c))
            for dr, dc in deltas:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < height and 0 <= cc < width and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        components.append(np.array(coords, dtype=int))
    return components


def _swap_template_colors(template: np.ndarray, body_color: int, marker_color: int) -> np.ndarray:
    swapped = template.copy()
    body_mask = swapped == body_color
    marker_mask = swapped == marker_color
    swapped[body_mask] = marker_color
    swapped[marker_mask] = body_color
    return swapped


def _copy_fits(
    shape: tuple[int, int],
    top_left: tuple[int, int],
    patch_shape: tuple[int, int],
) -> bool:
    rows, cols = map(int, shape)
    r0, c0 = map(int, top_left)
    h, w = map(int, patch_shape)
    return 0 <= r0 and 0 <= c0 and r0 + h <= rows and c0 + w <= cols


def _analyze_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    fg_mask = g != bg
    components = find_components_mask(fg_mask, connectivity=8)
    assert len(components) == 1, "expected exactly one foreground sprite"

    coords = components[0]
    box = bbox_coords(coords)
    patch = crop_bbox(g, box)
    height, width = map(int, patch.shape)
    assert height % 2 == 1 and width % 2 == 1, "bbox dimensions must both be odd"

    colors = [int(color) for color in np.unique(patch) if int(color) != bg]
    assert len(colors) == 2, "expected exactly one body color and one marker color"
    counts = {color: int(np.sum(patch == color)) for color in colors}
    marker_candidates = [color for color, count in counts.items() if count == 1]
    assert len(marker_candidates) == 1, "expected a unique marker pixel color"
    marker_color = int(marker_candidates[0])
    body_color = int(next(color for color in colors if color != marker_color))
    assert counts[body_color] >= 2, "sprite body must contain multiple cells"

    marker_rel = tuple(map(int, np.argwhere(patch == marker_color)[0]))
    center_rel = (height // 2, width // 2)
    dr = marker_rel[0] - center_rel[0]
    dc = marker_rel[1] - center_rel[1]
    assert (dr, dc) != (0, 0), "marker may not lie at the bbox center"

    return {
        "grid": g,
        "bg": int(bg),
        "bbox": tuple(map(int, box)),
        "template": patch,
        "body_color": body_color,
        "marker_color": marker_color,
        "marker_rel": marker_rel,
        "step": (int(dr), int(dc)),
    }


def _render_output(info: dict) -> np.ndarray:
    g = info["grid"]
    bg = int(info["bg"])
    r0, c0, _r1, _c1 = info["bbox"]
    template = info["template"]
    body_color = int(info["body_color"])
    marker_color = int(info["marker_color"])
    dr, dc = info["step"]
    h, w = map(int, template.shape)

    out = g.copy()
    template_mask = template != bg
    copy_index = 1
    while True:
        nr0 = r0 + copy_index * dr
        nc0 = c0 + copy_index * dc
        if not _copy_fits(out.shape, (nr0, nc0), (h, w)):
            break

        stamped = template if copy_index % 2 == 0 else _swap_template_colors(template, body_color, marker_color)
        window = out[nr0 : nr0 + h, nc0 : nc0 + w]
        window[template_mask] = stamped[template_mask]
        out[nr0 : nr0 + h, nc0 : nc0 + w] = window
        copy_index += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Repeat the unique marked sprite along its self-encoded offset, swapping body and marker colors on odd copies.

    Input structure:
    - A mostly empty grid contains exactly one 8-connected foreground sprite made of one body color plus one unique marker-color pixel.
    - The sprite's tight bbox has odd height and odd width, and the marker pixel is not at that bbox center.
    - Aside from the sprite, all remaining cells are background.

    Rule:
    - Compute the translation vector as `marker_position - bbox_center`, using bbox-local `(row, col)` coordinates.
    - Keep the original sprite unchanged, then place translated copies at offsets `1*v`, `2*v`, `3*v`, ... until the next whole copy would leave the grid.
    - Odd-numbered copies swap the body color and marker color before stamping; even-numbered copies keep the original coloring.

    Key insight:
    - The lone off-center marker is both part of the sprite and the encoded displacement that generates the whole periodic stamp sequence.
    """
    return _render_output(_analyze_input(grid))


def _line8(a: tuple[int, int], b: tuple[int, int], rng: np.random.Generator) -> list[tuple[int, int]]:
    """Connect two cells with an 8-connected path that always stays inside the bbox."""
    r, c = map(int, a)
    tr, tc = map(int, b)
    cells = [(r, c)]
    while (r, c) != (tr, tc):
        dr = int(np.sign(tr - r))
        dc = int(np.sign(tc - c))
        options: list[tuple[int, int]] = []
        if dr != 0 and dc != 0:
            options = [(r + dr, c + dc), (r + dr, c), (r, c + dc)]
        elif dr != 0:
            options = [(r + dr, c)]
        else:
            options = [(r, c + dc)]
        r, c = options[int(rng.integers(len(options)))]
        cells.append((r, c))
    return cells


def _mask_is_connected(mask: np.ndarray) -> bool:
    comps = find_components_mask(mask, connectivity=8)
    return len(comps) == 1


def _no_shift_overlap(mask: np.ndarray, dr: int, dc: int) -> bool:
    coords = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    span = max(mask.shape)
    for k in range(1, span + 1):
        if not any(
            0 <= r + k * dr < mask.shape[0] and 0 <= c + k * dc < mask.shape[1]
            for r, c in coords
        ):
            continue
        for r, c in coords:
            if (r + k * dr, c + k * dc) in coords:
                return False
    return True


def _choose_step(h: int, w: int, rng: np.random.Generator) -> tuple[int, int]:
    hr = h // 2
    hc = w // 2
    categories = ["row", "col", "diag"]
    mode = categories[int(rng.integers(len(categories)))]

    def choose_mag(limit: int) -> int:
        mags = np.arange(1, limit + 1)
        weights = 1 / mags
        weights = weights / weights.sum()
        mag = int(rng.choice(mags, p=weights))
        return mag if rng.random() < 0.5 else -mag

    if mode == "row":
        return choose_mag(hr), 0
    if mode == "col":
        return 0, choose_mag(hc)
    return choose_mag(hr), choose_mag(hc)


def _choose_geometry_and_step(rng: np.random.Generator) -> tuple[int, int, int, int]:
    odd_sizes = [3, 5, 7, 9]

    def choose_mag(limit: int) -> int:
        mags = np.arange(1, min(limit, 2) + 1)
        weights = 1 / mags
        weights = weights / weights.sum()
        mag = int(rng.choice(mags, p=weights))
        return mag if rng.random() < 0.5 else -mag

    mode = ["row", "col", "diag"][int(rng.integers(3))]
    if mode == "row":
        h = int(rng.choice([3, 5, 7]))
        w = int(rng.choice([size for size in odd_sizes if size >= h]))
        return h, w, choose_mag(h // 2), 0
    if mode == "col":
        w = int(rng.choice([3, 5, 7]))
        h = int(rng.choice([size for size in odd_sizes if size >= w]))
        return h, w, 0, choose_mag(w // 2)
    h = int(rng.choice([3, 5, 7]))
    w = int(rng.choice([size for size in odd_sizes if size >= h]))
    return h, w, choose_mag(h // 2), choose_mag(w // 2)


def _is_nonzero_multiple(drow: int, dcol: int, dr: int, dc: int) -> bool:
    if drow == 0 and dcol == 0:
        return False
    if dr == 0:
        return drow == 0 and dc != 0 and dcol % dc == 0 and dcol // dc != 0
    if dc == 0:
        return dcol == 0 and dr != 0 and drow % dr == 0 and drow // dr != 0
    if drow % dr != 0 or dcol % dc != 0:
        return False
    k_row = drow // dr
    k_col = dcol // dc
    return k_row == k_col and k_row != 0


def _cell_compatible(mask: np.ndarray, r: int, c: int, dr: int, dc: int) -> bool:
    for rr, cc in np.argwhere(mask):
        if _is_nonzero_multiple(int(r) - int(rr), int(c) - int(cc), dr, dc):
            return False
    return True


def _build_row_chain(
    h: int,
    w: int,
    marker_rel: tuple[int, int],
    dr: int,
    dc: int,
    rng: np.random.Generator,
) -> np.ndarray | None:
    marker_r, marker_c = marker_rel
    fixed_patterns = [
        {0: 0, h - 1: w - 1},
        {0: w - 1, h - 1: 0},
    ]
    if marker_r not in (0, h - 1):
        fixed_patterns.extend(
            [
                {marker_r - 1: 0, marker_r + 1: w - 1} if 0 < marker_r < h - 1 else {},
                {marker_r - 1: w - 1, marker_r + 1: 0} if 0 < marker_r < h - 1 else {},
            ]
        )

    for edge_fixed in fixed_patterns:
        fixed = {marker_r: marker_c}
        valid = True
        for row, col in edge_fixed.items():
            if not (0 <= row < h):
                valid = False
                break
            if row in fixed and fixed[row] != col:
                valid = False
                break
            fixed[row] = col
        if not valid:
            continue

        cols = [-1] * h

        def reachable(row: int, col: int) -> bool:
            for future_row, future_col in fixed.items():
                if future_row <= row:
                    continue
                if abs(future_col - col) > future_row - row:
                    return False
            return True

        def backtrack(row: int) -> bool:
            if row == h:
                return True
            if row in fixed:
                candidates = [fixed[row]]
            else:
                candidates = list(range(w))
                rng.shuffle(candidates)
            prev = cols[row - 1] if row > 0 else None
            for col in candidates:
                if prev is not None and abs(col - prev) > 1:
                    continue
                ok = True
                for rr in range(row):
                    if cols[rr] == -1:
                        continue
                    if _is_nonzero_multiple(row - rr, col - cols[rr], dr, dc):
                        ok = False
                        break
                if not ok or not reachable(row, col):
                    continue
                cols[row] = col
                if backtrack(row + 1):
                    return True
                cols[row] = -1
            return False

        if backtrack(0):
            mask = np.zeros((h, w), dtype=bool)
            for row, col in enumerate(cols):
                mask[row, col] = True
            return mask
    return None


def _build_col_chain(
    h: int,
    w: int,
    marker_rel: tuple[int, int],
    dr: int,
    dc: int,
    rng: np.random.Generator,
) -> np.ndarray | None:
    marker_r, marker_c = marker_rel
    fixed_patterns = [
        {0: 0, w - 1: h - 1},
        {0: h - 1, w - 1: 0},
    ]
    if marker_c not in (0, w - 1):
        fixed_patterns.extend(
            [
                {marker_c - 1: 0, marker_c + 1: h - 1} if 0 < marker_c < w - 1 else {},
                {marker_c - 1: h - 1, marker_c + 1: 0} if 0 < marker_c < w - 1 else {},
            ]
        )

    for edge_fixed in fixed_patterns:
        fixed = {marker_c: marker_r}
        valid = True
        for col, row in edge_fixed.items():
            if not (0 <= col < w):
                valid = False
                break
            if col in fixed and fixed[col] != row:
                valid = False
                break
            fixed[col] = row
        if not valid:
            continue

        rows = [-1] * w

        def reachable(col: int, row: int) -> bool:
            for future_col, future_row in fixed.items():
                if future_col <= col:
                    continue
                if abs(future_row - row) > future_col - col:
                    return False
            return True

        def backtrack(col: int) -> bool:
            if col == w:
                return True
            if col in fixed:
                candidates = [fixed[col]]
            else:
                candidates = list(range(h))
                rng.shuffle(candidates)
            prev = rows[col - 1] if col > 0 else None
            for row in candidates:
                if prev is not None and abs(row - prev) > 1:
                    continue
                ok = True
                for cc in range(col):
                    if rows[cc] == -1:
                        continue
                    if _is_nonzero_multiple(row - rows[cc], col - cc, dr, dc):
                        ok = False
                        break
                if not ok or not reachable(col, row):
                    continue
                rows[col] = row
                if backtrack(col + 1):
                    return True
                rows[col] = -1
            return False

        if backtrack(0):
            mask = np.zeros((h, w), dtype=bool)
            for col, row in enumerate(rows):
                mask[row, col] = True
            return mask
    return None


def _generate_mask(
    h: int,
    w: int,
    marker_rel: tuple[int, int],
    dr: int,
    dc: int,
    rng: np.random.Generator,
) -> np.ndarray | None:
    if dr != 0:
        mask = _build_row_chain(h, w, marker_rel, dr, dc, rng)
    else:
        mask = _build_col_chain(h, w, marker_rel, dr, dc, rng)
    if mask is None:
        return None

    extra_attempts = int(rng.integers(0, max(2, min(h, w))))
    for _ in range(extra_attempts):
        frontier = []
        for r, c in np.argwhere(mask):
            for rr, cc in neighbors8(int(r), int(c)):
                if 0 <= rr < h and 0 <= cc < w and not mask[rr, cc] and _cell_compatible(mask, rr, cc, dr, dc):
                    frontier.append((rr, cc))
        if not frontier:
            break
        rr, cc = frontier[int(rng.integers(len(frontier)))]
        mask[rr, cc] = True

    if not mask[marker_rel]:
        return None
    if not _mask_is_connected(mask):
        return None
    if mask.all():
        return None
    if not mask[0, :].any() or not mask[-1, :].any() or not mask[:, 0].any() or not mask[:, -1].any():
        return None
    if not _no_shift_overlap(mask, dr, dc):
        return None
    return mask


def _max_copy_count(h: int, w: int, dr: int, dc: int) -> int:
    limits = []
    if dr != 0:
        limits.append((30 - h) // abs(dr))
    if dc != 0:
        limits.append((30 - w) // abs(dc))
    return max(0, min(limits)) if limits else 0


def _choose_copy_count(max_copies: int, rng: np.random.Generator) -> int:
    if max_copies <= 1:
        return 1
    choices = np.arange(1, min(max_copies, 5) + 1)
    weights = np.ones(len(choices), dtype=float)
    if len(choices) >= 2:
        weights[1:] *= 2.5
    weights = weights / weights.sum()
    return int(rng.choice(choices, p=weights))


def _dimension_with_exact_copies(
    size: int,
    step: int,
    copies: int,
    rng: np.random.Generator,
) -> tuple[int, int]:
    base = size + copies * abs(step)
    slack = 30 - base
    if slack < 0:
        raise ValueError("dimension exceeds ARC limit")
    if step > 0:
        max_leading = min(2, slack)
        leading = int(rng.integers(0, max_leading + 1))
        max_trailing = min(step - 1, slack - leading)
        trailing = int(rng.integers(0, max_trailing + 1))
        total = leading + size + copies * step + trailing
        return total, leading
    if step < 0:
        step = -step
        max_leading = min(step - 1, slack)
        leading = int(rng.integers(0, max_leading + 1))
        max_trailing = min(2, slack - leading)
        trailing = int(rng.integers(0, max_trailing + 1))
        total = leading + size + copies * step + trailing
        return total, leading + copies * step
    max_leading = min(2, slack)
    leading = int(rng.integers(0, max_leading + 1))
    max_trailing = min(2, slack - leading)
    trailing = int(rng.integers(0, max_trailing + 1))
    total = leading + size + trailing
    return total, leading


def _build_grid(
    mask: np.ndarray,
    marker_rel: tuple[int, int],
    colors: tuple[int, int, int],
    copies: int,
    step: tuple[int, int],
    rng: np.random.Generator,
) -> np.ndarray:
    bg, body_color, marker_color = map(int, colors)
    h, w = map(int, mask.shape)
    dr, dc = map(int, step)
    rows, r0 = _dimension_with_exact_copies(h, dr, copies, rng)
    cols, c0 = _dimension_with_exact_copies(w, dc, copies, rng)
    assert rows <= 30 and cols <= 30

    grid = np.full((rows, cols), bg, dtype=int)
    rr = slice(r0, r0 + h)
    cc = slice(c0, c0 + w)
    patch = grid[rr, cc]
    patch[mask] = body_color
    patch[marker_rel] = marker_color
    grid[rr, cc] = patch
    return grid


def _generate_base_mask(rng: np.random.Generator) -> np.ndarray:
    if rng.random() < 0.5:
        h, w = [(5, 3), (7, 3), (7, 5), (9, 5), (9, 7)][int(rng.integers(5))]
        left_to_right = bool(rng.integers(2))
        start = 0 if left_to_right else w - 1
        end = w - 1 if left_to_right else 0
        cols = [start]
        for row in range(1, h - 1):
            prev = cols[-1]
            remaining = h - 1 - row
            lo = max(0, prev - 1, end - remaining)
            hi = min(w - 1, prev + 1, end + remaining)
            cols.append(int(rng.integers(lo, hi + 1)))
        cols.append(end)
        mask = np.zeros((h, w), dtype=bool)
        for row, col in enumerate(cols):
            mask[row, col] = True
        return mask

    h, w = [(3, 5), (3, 7), (5, 7), (5, 9), (7, 9)][int(rng.integers(5))]
    top_to_bottom = bool(rng.integers(2))
    start = 0 if top_to_bottom else h - 1
    end = h - 1 if top_to_bottom else 0
    rows = [start]
    for col in range(1, w - 1):
        prev = rows[-1]
        remaining = w - 1 - col
        lo = max(0, prev - 1, end - remaining)
        hi = min(h - 1, prev + 1, end + remaining)
        rows.append(int(rng.integers(lo, hi + 1)))
    rows.append(end)
    mask = np.zeros((h, w), dtype=bool)
    for col, row in enumerate(rows):
        mask[row, col] = True
    return mask


def _marker_candidates(mask: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    h, w = map(int, mask.shape)
    center = (h // 2, w // 2)
    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        marker_rel = (int(r), int(c))
        step = (marker_rel[0] - center[0], marker_rel[1] - center[1])
        if step == (0, 0):
            continue
        if _no_shift_overlap(mask, step[0], step[1]) and _max_copy_count(h, w, step[0], step[1]) >= 1:
            candidates.append((marker_rel, step))
    return candidates


def _semantic_check(grid: np.ndarray) -> bool:
    info = _analyze_input(grid)
    g = info["grid"]
    out = _render_output(info)
    bg = int(info["bg"])
    r0, c0, _r1, _c1 = info["bbox"]
    template = info["template"]
    mask = template != bg
    h, w = map(int, template.shape)
    dr, dc = info["step"]
    body_color = int(info["body_color"])
    marker_color = int(info["marker_color"])

    expected = g.copy()
    seen_copy = False
    copy_index = 1
    while True:
        nr0 = r0 + copy_index * dr
        nc0 = c0 + copy_index * dc
        if not _copy_fits(expected.shape, (nr0, nc0), (h, w)):
            break
        seen_copy = True
        patch = template if copy_index % 2 == 0 else _swap_template_colors(template, body_color, marker_color)
        expected[nr0 : nr0 + h, nc0 : nc0 + w][mask] = patch[mask]
        copy_index += 1

    if not seen_copy:
        return False
    if not np.array_equal(expected, out):
        return False

    body_mask = template == body_color
    if int(np.sum(body_mask)) < 2:
        return False
    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(500):
        mask = _generate_base_mask(rng)
        candidates = _marker_candidates(mask)
        if not candidates:
            continue

        grouped: dict[str, list[tuple[tuple[int, int], tuple[int, int]]]] = {"row": [], "col": [], "diag": []}
        for marker_rel, step in candidates:
            dr, dc = step
            if dr == 0:
                grouped["row"].append((marker_rel, step))
            elif dc == 0:
                grouped["col"].append((marker_rel, step))
            else:
                grouped["diag"].append((marker_rel, step))

        nonempty = [name for name, group in grouped.items() if group]
        weights = np.array([2.0 if name == "diag" else 1.0 for name in nonempty], dtype=float)
        weights = weights / weights.sum()
        chosen_group = nonempty[int(rng.choice(len(nonempty), p=weights))]
        marker_rel, (dr, dc) = grouped[chosen_group][int(rng.integers(len(grouped[chosen_group])))]

        h, w = map(int, mask.shape)
        max_copies = _max_copy_count(h, w, dr, dc)
        if max_copies < 1:
            continue
        copies = _choose_copy_count(max_copies, rng)

        colors = rng.choice(np.arange(10), size=3, replace=False)
        grid = _build_grid(mask, marker_rel, tuple(map(int, colors)), copies, (dr, dc), rng)
        if _semantic_check(grid):
            return grid
    raise RuntimeError("failed to generate a valid self-offset stamp input")
