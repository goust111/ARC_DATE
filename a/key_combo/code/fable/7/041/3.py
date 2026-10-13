from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from itertools import permutations

import numpy as np


GENERATION_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, bg: int) -> list[int]:
    return sorted(int(x) for x in np.unique(grid) if int(x) != bg)


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _normalize_coords(coords: np.ndarray) -> frozenset[tuple[int, int]]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    return frozenset((int(r - r0), int(c - c0)) for r, c in coords)


def _shape_extent(shape: frozenset[tuple[int, int]]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _scan_anchors(
    grid: np.ndarray,
    shape: frozenset[tuple[int, int]],
    own_color: int,
    overlap_color: int,
) -> list[tuple[int, int]]:
    h, w = _shape_extent(shape)
    coords = list(shape)
    out = []
    for r0 in range(grid.shape[0] - h + 1):
        for c0 in range(grid.shape[1] - w + 1):
            has_own = False
            ok = True
            for dr, dc in coords:
                val = int(grid[r0 + dr, c0 + dc])
                if val == own_color:
                    has_own = True
                elif val != overlap_color:
                    ok = False
                    break
            if ok and has_own:
                out.append((r0, c0))
    return out


def _gcd_nonzero(values: list[int]) -> int:
    vals = [abs(int(v)) for v in values if int(v) != 0]
    if not vals:
        return 0
    g = vals[0]
    for v in vals[1:]:
        g = math.gcd(g, v)
    return g


def _infer_lattice(positions: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    if len(positions) < 2:
        return None
    rows = [r for r, _ in positions]
    cols = [c for _, c in positions]
    drs = [rows[j] - rows[i] for i in range(len(rows)) for j in range(i + 1, len(rows))]
    dcs = [cols[j] - cols[i] for i in range(len(cols)) for j in range(i + 1, len(cols))]
    step_r = _gcd_nonzero(drs)
    step_c = _gcd_nonzero(dcs)
    if step_r <= 0 or step_c <= 0:
        return None
    mod_r = rows[0] % step_r
    mod_c = cols[0] % step_c
    if any(r % step_r != mod_r for r in rows):
        return None
    if any(c % step_c != mod_c for c in cols):
        return None
    return step_r, step_c, mod_r, mod_c


def _on_lattice(pos: tuple[int, int], lattice: tuple[int, int, int, int]) -> bool:
    r, c = pos
    step_r, step_c, mod_r, mod_c = lattice
    return r % step_r == mod_r and c % step_c == mod_c


def _render_true_stamps(
    shape: frozenset[tuple[int, int]],
    base_anchors: list[tuple[int, int]],
    odd_anchor: tuple[int, int],
    base_color: int,
    odd_color: int,
    overlap_color: int,
    bg: int,
    canvas_shape: tuple[int, int],
) -> np.ndarray:
    h, w = canvas_shape
    counts = np.zeros((h, w), dtype=int)
    odd_counts = np.zeros((h, w), dtype=int)
    for r0, c0 in base_anchors:
        for dr, dc in shape:
            counts[r0 + dr, c0 + dc] += 1
    for dr, dc in shape:
        counts[odd_anchor[0] + dr, odd_anchor[1] + dc] += 1
        odd_counts[odd_anchor[0] + dr, odd_anchor[1] + dc] += 1

    expected = np.full((h, w), bg, dtype=int)
    expected[counts >= 2] = overlap_color
    expected[(counts == 1) & (odd_counts == 1)] = odd_color
    expected[(counts == 1) & (odd_counts == 0)] = base_color
    return expected


def _validate_candidate(
    grid: np.ndarray,
    shape: frozenset[tuple[int, int]],
    base_color: int,
    odd_color: int,
    overlap_color: int,
    base_anchors: list[tuple[int, int]],
    odd_anchor: tuple[int, int],
    lattice: tuple[int, int, int, int],
    bg: int,
) -> bool:
    expected = _render_true_stamps(
        shape,
        base_anchors,
        odd_anchor,
        base_color,
        odd_color,
        overlap_color,
        bg,
        grid.shape,
    )

    if np.any((expected == odd_color) != (grid == odd_color)):
        return False
    if np.any((expected == overlap_color) != (grid == overlap_color)):
        return False
    if np.any((expected == base_color) & (grid != base_color)):
        return False
    if np.any((expected == bg) & ~(np.isin(grid, [bg, base_color]))):
        return False

    fake_mask = (grid == base_color) & (expected == bg)
    if not fake_mask.any():
        return False
    fake_components = _find_components(fake_mask)
    if len(fake_components) != 1:
        return False
    fake_shape = _normalize_coords(fake_components[0])
    if fake_shape == shape:
        return False
    fake_top = (
        int(fake_components[0][:, 0].min()),
        int(fake_components[0][:, 1].min()),
    )
    if _on_lattice(fake_top, lattice):
        return False
    return True


def _solve_details(grid: np.ndarray) -> tuple[list[tuple[int, int]], tuple[int, int], int, int, tuple[int, int, int, int]]:
    bg = _get_bg(grid)
    colors = _colors_present(grid, bg)
    if len(colors) != 3:
        raise ValueError("expected exactly three foreground colors")

    shape_groups: dict[frozenset[tuple[int, int]], list[dict[str, object]]] = defaultdict(list)
    for color in colors:
        for coords in _find_components(grid == color):
            shape = _normalize_coords(coords)
            record = {
                "color": color,
                "shape": shape,
                "top": (int(coords[:, 0].min()), int(coords[:, 1].min())),
            }
            shape_groups[shape].append(record)

    candidate_shapes = [shape for shape, items in shape_groups.items() if len(shape) >= 3 and len(items) >= 2]
    if not candidate_shapes:
        raise ValueError("could not infer template shape")
    candidate_shapes.sort(key=lambda shape: (len(shape), len(shape_groups[shape])), reverse=True)

    best = None
    for shape in candidate_shapes:
        for base_color, odd_color, overlap_color in permutations(colors, 3):
            full_positions = [
                item["top"]
                for item in shape_groups[shape]
                if item["color"] in (base_color, odd_color)
            ]
            if len(full_positions) < 2:
                continue
            lattice = _infer_lattice(full_positions)
            if lattice is None:
                continue

            base_anchors = [pos for pos in _scan_anchors(grid, shape, base_color, overlap_color) if _on_lattice(pos, lattice)]
            odd_anchors = [pos for pos in _scan_anchors(grid, shape, odd_color, overlap_color) if _on_lattice(pos, lattice)]
            if len(odd_anchors) != 1 or not base_anchors:
                continue

            odd_anchor = odd_anchors[0]
            base_anchors = sorted(set(base_anchors) - {odd_anchor})
            if len(base_anchors) + 1 < len(full_positions):
                continue
            if not _validate_candidate(
                grid,
                shape,
                base_color,
                odd_color,
                overlap_color,
                base_anchors,
                odd_anchor,
                lattice,
                bg,
            ):
                continue

            score = (len(base_anchors) + 1, len(shape))
            if best is None or score > best[0]:
                best = (
                    score,
                    base_anchors,
                    odd_anchor,
                    base_color,
                    odd_color,
                    lattice,
                )
    if best is None:
        raise ValueError("no valid reconstruction found")
    return best[1], best[2], best[3], best[4], best[5]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Invert a repeated stamp image into its coarse lattice layout while discarding the off-lattice fake copy.

    Input structure:
    The grid has a dominant background plus many copies of one unknown connected template. Most true copies use one color, exactly one true copy uses a second color, and every pixel shared by two true copies is recolored into a fixed overlap color. There is also exactly one extra connected shape in the main template color whose shape is different and whose top-left is off the true lattice.

    Rule:
    Recover the repeated template from the intact same-shape copies, infer the row and column spacing of the lattice from their top-left positions, and treat overlap-color pixels as belonging to the true copies whose template cells cover them. Keep only template placements whose top-left lies on that lattice, mark the unique odd-colored placement with its own color, mark all other true placements with the template color, ignore the fake copy, and crop the result to the minimal lattice-aligned bounding box covering the true placements.

    Key insight:
    The large picture is a stamp-upscaled encoding of a much smaller occupancy grid: repeated intact copies reveal the hidden template, the shared lattice explains which partial/overlapped placements are real, and the only off-lattice same-color component is the fake that must be removed before downsampling.
    """
    base_anchors, odd_anchor, base_color, odd_color, lattice = _solve_details(grid)
    all_anchors = base_anchors + [odd_anchor]
    rows = [r for r, _ in all_anchors]
    cols = [c for _, c in all_anchors]
    r0 = min(rows)
    c0 = min(cols)
    step_r, step_c, _, _ = lattice
    out_h = (max(rows) - r0) // step_r + 1
    out_w = (max(cols) - c0) // step_c + 1
    bg = _get_bg(grid)
    out = np.full((out_h, out_w), bg, dtype=int)
    for r, c in base_anchors:
        out[(r - r0) // step_r, (c - c0) // step_c] = base_color
    out[(odd_anchor[0] - r0) // step_r, (odd_anchor[1] - c0) // step_c] = odd_color
    return out


def _grow_connected_shape(h: int, w: int, target_cells: int) -> frozenset[tuple[int, int]]:
    cells = {(0, 0)}
    while len(cells) < target_cells:
        frontier = []
        for r, c in cells:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(random.choice(frontier))
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0 = min(rows)
    c0 = min(cols)
    norm = frozenset((r - r0, c - c0) for r, c in cells)
    if (0, 0) not in norm:
        return frozenset()
    return norm


def _shift_overlap_size(shape: frozenset[tuple[int, int]], dr: int, dc: int) -> int:
    shifted = {(r + dr, c + dc) for r, c in shape}
    return len(set(shape) & shifted)


def _random_template() -> frozenset[tuple[int, int]]:
    for _ in range(200):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        target = random.randint(max(4, (h * w + 1) // 2), h * w - (1 if h * w >= 5 else 0))
        shape = _grow_connected_shape(h, w, target)
        if not shape:
            continue
        rows = {r for r, _ in shape}
        cols = {c for _, c in shape}
        if len(rows) < 2 or len(cols) < 2:
            continue
        if len(shape) == h * w:
            continue
        return shape
    raise RuntimeError("failed to build template")


def _shape_to_mask(shape: frozenset[tuple[int, int]]) -> np.ndarray:
    h, w = _shape_extent(shape)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in shape:
        mask[r, c] = True
    return mask


def _random_fake_shape(template: frozenset[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    th, tw = _shape_extent(template)
    for _ in range(300):
        h = random.randint(2, min(4, max(2, th + 1)))
        w = random.randint(2, min(4, max(2, tw + 1)))
        target = random.randint(3, max(3, min(h * w - 1, len(template) + 1)))
        shape = _grow_connected_shape(h, w, target)
        if not shape or shape == template:
            continue
        fake_cells = set(shape)
        contains_template = False
        for r_shift in range(-th, h + 1):
            for c_shift in range(-tw, w + 1):
                shifted = {(r + r_shift, c + c_shift) for r, c in template}
                if all((r, c) in fake_cells for r, c in shifted):
                    contains_template = True
                    break
            if contains_template:
                break
        if not contains_template:
            return shape
    raise RuntimeError("failed to build fake shape")


def _paint_stamp_grid(
    shape: frozenset[tuple[int, int]],
    true_anchors: list[tuple[int, int]],
    odd_anchor: tuple[int, int],
    fake_anchor: tuple[int, int],
    fake_shape: frozenset[tuple[int, int]],
    bg: int,
    base_color: int,
    odd_color: int,
    overlap_color: int,
    canvas_shape: tuple[int, int],
) -> np.ndarray:
    h, w = canvas_shape
    owners: dict[tuple[int, int], list[int]] = defaultdict(list)
    for anchor in true_anchors:
        color = odd_color if anchor == odd_anchor else base_color
        for dr, dc in shape:
            owners[(anchor[0] + dr, anchor[1] + dc)].append(color)

    grid = np.full((h, w), bg, dtype=int)
    for (r, c), colors in owners.items():
        if len(colors) >= 2:
            grid[r, c] = overlap_color
        else:
            grid[r, c] = colors[0]
    for dr, dc in fake_shape:
        grid[fake_anchor[0] + dr, fake_anchor[1] + dc] = base_color
    return grid


def _touches(mask: np.ndarray, anchor: tuple[int, int], shape: frozenset[tuple[int, int]]) -> bool:
    ah, aw = mask.shape
    occupied = {(anchor[0] + dr, anchor[1] + dc) for dr, dc in shape}
    for r, c in occupied:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if 0 <= nr < ah and 0 <= nc < aw and mask[nr, nc]:
                    if (nr, nc) not in occupied:
                        return True
    return False


def _build_output(
    indices: list[tuple[int, int]],
    odd_index: tuple[int, int],
    bg: int,
    base_color: int,
    odd_color: int,
) -> np.ndarray:
    rows = [r for r, _ in indices]
    cols = [c for _, c in indices]
    r0 = min(rows)
    c0 = min(cols)
    out = np.full((max(rows) - r0 + 1, max(cols) - c0 + 1), bg, dtype=int)
    for r, c in indices:
        out[r - r0, c - c0] = odd_color if (r, c) == odd_index else base_color
    return out


def _layout_specs(mode: str) -> tuple[list[tuple[int, int]], int, int]:
    if mode == "horizontal":
        patterns = [
            [(0, 0), (0, 2), (2, 0), (2, 3), (3, 1), (1, 1), (1, 2)],
            [(0, 1), (0, 3), (2, 0), (3, 2), (3, 0), (1, 1), (1, 2)],
        ]
        return list(random.choice(patterns)), 4, 4
    patterns = [
        [(0, 0), (2, 0), (0, 2), (3, 2), (1, 3), (1, 1), (2, 1)],
        [(1, 0), (3, 0), (0, 2), (2, 3), (0, 1), (1, 1), (2, 1)],
    ]
    return list(random.choice(patterns)), 4, 4


def _gen_attempt() -> np.ndarray | None:
    bg, base_color, odd_color, overlap_color = random.sample(range(10), 4)
    template = _random_template()
    th, tw = _shape_extent(template)
    mode = random.choice(["horizontal", "vertical"])

    if mode == "horizontal":
        sr = th + random.randint(1, 2)
        candidates = [sc for sc in range(max(1, (tw + 1) // 2), tw) if 0 < _shift_overlap_size(template, 0, sc) < len(template)]
        if not candidates:
            return None
        sc = random.choice(candidates)
    else:
        sc = tw + random.randint(1, 2)
        candidates = [sr for sr in range(max(1, (th + 1) // 2), th) if 0 < _shift_overlap_size(template, sr, 0) < len(template)]
        if not candidates:
            return None
        sr = random.choice(candidates)

    index_list, rows_n, cols_n = _layout_specs(mode)
    true_indices = list(index_list)
    odd_index = random.choice(true_indices)

    margin_top = random.randint(1, 3)
    margin_left = random.randint(1, 3)
    h = margin_top * 2 + (rows_n - 1) * sr + th
    w = margin_left * 2 + (cols_n - 1) * sc + tw
    if h > 30 or w > 30:
        return None

    true_anchors = [(margin_top + r * sr, margin_left + c * sc) for r, c in true_indices]
    odd_anchor = (margin_top + odd_index[0] * sr, margin_left + odd_index[1] * sc)

    occupied = np.zeros((h, w), dtype=bool)
    for anchor in true_anchors:
        for dr, dc in template:
            rr = anchor[0] + dr
            cc = anchor[1] + dc
            occupied[rr, cc] = True

    fake_shape = _random_fake_shape(template)
    fh, fw = _shape_extent(fake_shape)
    fake_positions = []
    mod_r = margin_top % sr
    mod_c = margin_left % sc
    for r0 in range(0, h - fh + 1):
        for c0 in range(0, w - fw + 1):
            if r0 % sr == mod_r and c0 % sc == mod_c:
                continue
            cells = [(r0 + dr, c0 + dc) for dr, dc in fake_shape]
            if any(occupied[r, c] for r, c in cells):
                continue
            if _touches(occupied, (r0, c0), fake_shape):
                continue
            fake_positions.append((r0, c0))
    if not fake_positions:
        return None
    fake_anchor = random.choice(fake_positions)

    grid = _paint_stamp_grid(
        template,
        true_anchors,
        odd_anchor,
        fake_anchor,
        fake_shape,
        bg,
        base_color,
        odd_color,
        overlap_color,
        (h, w),
    )

    expected = _build_output(true_indices, odd_index, bg, base_color, odd_color)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
