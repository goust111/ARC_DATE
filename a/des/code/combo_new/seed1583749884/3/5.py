import random

import numpy as np


BG = 0
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GEN_RETRIES = 400
OBJECT_RETRIES = 80

HOLE_LAYOUTS = {
    1: [
        [(0.50, 0.50)],
        [(0.35, 0.50)],
        [(0.65, 0.50)],
        [(0.50, 0.35)],
        [(0.50, 0.65)],
    ],
    2: [
        [(0.28, 0.50), (0.68, 0.50)],
        [(0.50, 0.25), (0.50, 0.68)],
        [(0.28, 0.28), (0.68, 0.68)],
        [(0.28, 0.68), (0.68, 0.28)],
    ],
    3: [
        [(0.28, 0.22), (0.28, 0.62), (0.62, 0.42)],
        [(0.22, 0.28), (0.62, 0.28), (0.42, 0.62)],
        [(0.50, 0.16), (0.50, 0.46), (0.50, 0.74)],
        [(0.16, 0.50), (0.46, 0.50), (0.74, 0.50)],
    ],
}

CORE_RANGES = {
    0: ((5, 8), (5, 8)),
    1: ((6, 9), (6, 9)),
    2: ((7, 9), (7, 10)),
    3: ((8, 10), (8, 10)),
}


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIR4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _components_from_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _monochrome_components(grid: np.ndarray, valid_mask: np.ndarray) -> list[tuple[int, list[tuple[int, int]]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[tuple[int, list[tuple[int, int]]]] = []
    for r in range(h):
        for c in range(w):
            if not valid_mask[r, c] or seen[r, c]:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if valid_mask[nr, nc] and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append((color, comp))
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _touches_border(coords: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return mask.copy()
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _boundary_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    boundary = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            for dr, dc in DIR4:
                nr = r + dr
                nc = c + dc
                if not (0 <= nr < h and 0 <= nc < w) or not mask[nr, nc]:
                    boundary[r, c] = True
                    break
    return boundary


def _hole_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    holes = 0
    for comp in _components_from_mask(~mask):
        if not _touches_border(comp, h, w):
            holes += 1
    return holes


def _interior_count(mask: np.ndarray) -> int:
    boundary = _boundary_mask(mask)
    return int(np.sum(mask & ~boundary))


def _is_connected(mask: np.ndarray) -> bool:
    comps = _components_from_mask(mask)
    return len(comps) == 1


def _legend_mapping(grid: np.ndarray) -> tuple[dict[int, int], np.ndarray]:
    h, w = grid.shape
    candidates = []
    fg_comps = _components_from_mask(grid != BG)
    for comp in fg_comps:
        if not _touches_border(comp, h, w):
            continue
        r0, c0, r1, c1 = _bbox(comp)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(sub == BG):
            continue
        row_colors: list[int] = []
        ok = True
        for row in sub:
            if not np.all(row == row[0]):
                ok = False
                break
            row_colors.append(int(row[0]))
        if not ok:
            continue
        runs: list[tuple[int, int]] = []
        for color in row_colors:
            if not runs or runs[-1][0] != color:
                runs.append((color, 1))
            else:
                runs[-1] = (color, runs[-1][1] + 1)
        if len(runs) != 4 or len({color for color, _ in runs}) != 4:
            continue
        legend_mask = np.zeros_like(grid, dtype=bool)
        for rr, cc in comp:
            legend_mask[rr, cc] = True
        mapping = {idx: color for idx, (color, _) in enumerate(runs)}
        score = (sub.size, sub.shape[1], sub.shape[0], r0, c0)
        candidates.append((score, mapping, legend_mask))
    if not candidates:
        raise AssertionError("expected one edge legend formed by four vertical color blocks")
    candidates.sort(key=lambda item: item[0])
    _, mapping, legend_mask = candidates[0]
    return mapping, legend_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Preserve each object's boundary and recolor its interior from a hole-count legend.

    Input structure:
    - The background is black.
    - Away from the border legend, the grid contains several 4-connected monochrome foreground objects with distinct colors and varied shapes.
    - Some objects contain 4-connected black holes fully enclosed by the object; hole counts range from 0 to 3.
    - A border-touching legend is a vertical stack of four solid color blocks whose top-to-bottom order encodes the fill colors for hole counts 0, 1, 2, and 3.

    Rule:
    - Detect the legend and read its four colors from top to bottom as the mapping for hole counts 0..3.
    - For each non-legend foreground object, keep all boundary pixels in the object's original color. A boundary pixel has at least one 4-neighbor outside the object, including black holes and other colors.
    - Count the object's enclosed black holes using 4-connectivity inside its own bounding box.
    - Replace every non-boundary object pixel with the legend color indexed by that hole count, while leaving the holes themselves black.

    Key insight:
    - The relevant object feature is topological rather than geometric: the number of enclosed holes selects a legend color, and only the object's true interior gets recolored.
    """
    g = np.asarray(grid, dtype=int)
    mapping, legend_mask = _legend_mapping(g)
    valid = (g != BG) & ~legend_mask
    out = g.copy()

    for _color, comp in _monochrome_components(g, valid):
        r0, c0, r1, c1 = _bbox(comp)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in comp:
            mask[r - r0, c - c0] = True
        holes = _hole_count(mask)
        if holes not in mapping:
            raise AssertionError(f"legend missing mapping for hole count {holes}")
        boundary = _boundary_mask(mask)
        fill_color = int(mapping[holes])
        interior = mask & ~boundary
        rr, cc = np.where(interior)
        out[r0 + rr, c0 + cc] = fill_color

    return out


def _rects_separated(rects: list[tuple[int, int, int, int]]) -> bool:
    for i, (r0, c0, h0, w0) in enumerate(rects):
        for r1, c1, h1, w1 in rects[i + 1 :]:
            if not (
                r0 + h0 + 1 <= r1
                or r1 + h1 + 1 <= r0
                or c0 + w0 + 1 <= c1
                or c1 + w1 + 1 <= c0
            ):
                return False
    return True


def _hole_rects(core_h: int, core_w: int, hole_count: int, rng: random.Random):
    if hole_count == 0:
        return []
    layouts = HOLE_LAYOUTS[hole_count]
    for _ in range(60):
        rects = []
        layout = rng.choice(layouts)
        for frac_r, frac_c in layout:
            tall_ok = core_h >= 8 and hole_count <= 2 and rng.random() < 0.35
            wide_ok = core_w >= 8 and hole_count <= 2 and rng.random() < 0.35
            hole_h = 1 + int(tall_ok)
            hole_w = 1 + int(wide_ok)
            if hole_count == 3 and rng.random() < 0.7:
                hole_h = 1
            if hole_count == 3 and rng.random() < 0.7:
                hole_w = 1

            max_r = core_h - hole_h - 1
            max_c = core_w - hole_w - 1
            if max_r < 1 or max_c < 1:
                rects = None
                break
            base_r = 1 + int(round(frac_r * (max_r - 1)))
            base_c = 1 + int(round(frac_c * (max_c - 1)))
            top = min(max(1, base_r + rng.randint(-1, 1)), max_r)
            left = min(max(1, base_c + rng.randint(-1, 1)), max_c)
            rects.append((top, left, hole_h, hole_w))
        if rects is not None and _rects_separated(rects):
            return rects
    return None


def _candidate_bites(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return []
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    candidates: list[list[tuple[int, int]]] = []

    singles = [
        (r0, c0),
        (r0, c1),
        (r1, c0),
        (r1, c1),
        (r0, (c0 + c1) // 2),
        (r1, (c0 + c1) // 2),
        ((r0 + r1) // 2, c0),
        ((r0 + r1) // 2, c1),
    ]
    for cell in singles:
        if mask[cell]:
            candidates.append([cell])

    if c1 - c0 >= 3:
        mc = (c0 + c1) // 2
        top_pair = [(r0, mc), (r0, min(c1, mc + 1))]
        bottom_pair = [(r1, mc), (r1, min(c1, mc + 1))]
        if all(mask[cell] for cell in top_pair):
            candidates.append(top_pair)
        if all(mask[cell] for cell in bottom_pair):
            candidates.append(bottom_pair)
    if r1 - r0 >= 3:
        mr = (r0 + r1) // 2
        left_pair = [(mr, c0), (min(r1, mr + 1), c0)]
        right_pair = [(mr, c1), (min(r1, mr + 1), c1)]
        if all(mask[cell] for cell in left_pair):
            candidates.append(left_pair)
        if all(mask[cell] for cell in right_pair):
            candidates.append(right_pair)

    seen = set()
    unique: list[list[tuple[int, int]]] = []
    for cells in candidates:
        key = tuple(sorted(cells))
        if key not in seen:
            seen.add(key)
            unique.append(cells)
    return unique


def _validated_variant(mask: np.ndarray, hole_count: int, min_interior: int) -> np.ndarray | None:
    cropped = _crop_mask(mask)
    if np.sum(cropped) < 8:
        return None
    if not _is_connected(cropped):
        return None
    if _hole_count(cropped) != hole_count:
        return None
    if _interior_count(cropped) < min_interior:
        return None
    return cropped


def _random_transform(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    k = rng.randint(0, 3)
    out = np.rot90(mask, k)
    if rng.random() < 0.5:
        out = np.fliplr(out)
    if rng.random() < 0.5:
        out = np.flipud(out)
    return out.copy()


def _make_object_mask(hole_count: int, rng: random.Random) -> np.ndarray | None:
    min_interior = 2
    (h_lo, h_hi), (w_lo, w_hi) = CORE_RANGES[hole_count]
    for _ in range(OBJECT_RETRIES):
        core_h = rng.randint(h_lo, h_hi)
        core_w = rng.randint(w_lo, w_hi)
        canvas = np.zeros((core_h + 4, core_w + 4), dtype=bool)
        top, left = 2, 2
        canvas[top : top + core_h, left : left + core_w] = True

        sides = ["top", "bottom", "left", "right"]
        rng.shuffle(sides)
        for side in sides[: rng.randint(0, 2)]:
            if side in ("top", "bottom") and core_w >= 4:
                length = rng.randint(1, min(3, core_w - 2))
                c0 = left + rng.randint(1, core_w - length - 1)
                row = top - 1 if side == "top" else top + core_h
                canvas[row, c0 : c0 + length] = True
            if side in ("left", "right") and core_h >= 4:
                length = rng.randint(1, min(3, core_h - 2))
                r0 = top + rng.randint(1, core_h - length - 1)
                col = left - 1 if side == "left" else left + core_w
                canvas[r0 : r0 + length, col] = True

        rects = _hole_rects(core_h, core_w, hole_count, rng)
        if rects is None:
            continue
        for hr, hc, hh, hw in rects:
            canvas[top + hr : top + hr + hh, left + hc : left + hc + hw] = False

        base = _validated_variant(canvas, hole_count, min_interior)
        if base is None:
            continue

        mutated = base
        if rng.random() < 0.7:
            bite_budget = rng.randint(0, 2)
            applied = 0
            while applied < bite_budget:
                bites = _candidate_bites(mutated)
                if not bites:
                    break
                rng.shuffle(bites)
                changed = False
                for cells in bites:
                    trial = mutated.copy()
                    for rr, cc in cells:
                        if not (0 <= rr < trial.shape[0] and 0 <= cc < trial.shape[1]):
                            continue
                        trial[rr, cc] = False
                    checked = _validated_variant(trial, hole_count, min_interior)
                    if checked is not None:
                        mutated = checked
                        applied += 1
                        changed = True
                        break
                if not changed:
                    break

        mutated = _random_transform(mutated, rng)
        checked = _validated_variant(mutated, hole_count, min_interior)
        if checked is not None:
            return checked
    return None


def _build_legend(colors: list[int], band_h: int, width: int) -> np.ndarray:
    legend = np.zeros((4 * band_h, width), dtype=int)
    for idx, color in enumerate(colors):
        legend[idx * band_h : (idx + 1) * band_h, :] = int(color)
    return legend


def _place_legend(
    grid: np.ndarray,
    occupied: np.ndarray,
    legend: np.ndarray,
    side: str,
    rng: random.Random,
) -> tuple[int, int]:
    h, w = grid.shape
    lh, lw = legend.shape
    if side == "left":
        r0 = rng.randint(0, h - lh)
        c0 = 0
    elif side == "right":
        r0 = rng.randint(0, h - lh)
        c0 = w - lw
    elif side == "top":
        r0 = 0
        c0 = rng.randint(0, w - lw)
    else:
        r0 = h - lh
        c0 = rng.randint(0, w - lw)
    grid[r0 : r0 + lh, c0 : c0 + lw] = legend
    occupied[max(0, r0 - 1) : min(h, r0 + lh + 1), max(0, c0 - 1) : min(w, c0 + lw + 1)] = True
    return r0, c0


def _object_positions(h: int, w: int, occupied: np.ndarray, mask: np.ndarray) -> list[tuple[int, int]]:
    mh, mw = mask.shape
    positions = []
    for r0 in range(1, h - mh):
        for c0 in range(1, w - mw):
            if occupied[r0 - 1 : r0 + mh + 1, c0 - 1 : c0 + mw + 1].any():
                continue
            positions.append((r0, c0))
    return positions


def _paint_object(
    inp: np.ndarray,
    out: np.ndarray,
    mask: np.ndarray,
    r0: int,
    c0: int,
    obj_color: int,
    fill_color: int,
) -> None:
    mh, mw = mask.shape
    boundary = _boundary_mask(mask)
    interior = mask & ~boundary
    in_patch = inp[r0 : r0 + mh, c0 : c0 + mw]
    out_patch = out[r0 : r0 + mh, c0 : c0 + mw]
    in_patch[mask] = obj_color
    out_patch[mask] = obj_color
    out_patch[interior] = fill_color


def _sample_dimensions(
    objects: list[np.ndarray], legend_h: int, legend_w: int, rng: random.Random
) -> tuple[int, int] | None:
    max_h = max(obj.shape[0] for obj in objects)
    max_w = max(obj.shape[1] for obj in objects)
    padded_area = sum((obj.shape[0] + 2) * (obj.shape[1] + 2) for obj in objects)
    choices = []
    for h in range(max(20, legend_h + 2, max_h + 2), 31):
        for w in range(max(20, legend_w + 2, max_w + 2), 31):
            if h * w >= padded_area + 50:
                choices.append((h, w))
    if not choices:
        return None
    return rng.choice(choices)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    num_objects = rng.choice([4, 5])
    hole_counts = [0, 1, 2, 3]
    if num_objects == 5:
        hole_counts.append(rng.randint(0, 3))
    rng.shuffle(hole_counts)

    objects: list[tuple[np.ndarray, int]] = []
    for hole_count in hole_counts:
        mask = _make_object_mask(hole_count, rng)
        if mask is None:
            return None
        objects.append((mask, hole_count))

    band_h = rng.randint(1, 2)
    legend_w = rng.randint(1, 2)
    legend_colors = random.sample(range(1, 10), 4)
    remaining_colors = [c for c in range(1, 10) if c not in legend_colors]
    rng.shuffle(remaining_colors)
    object_colors = remaining_colors[:num_objects]

    dims = _sample_dimensions([mask for mask, _ in objects], 4 * band_h, legend_w, rng)
    if dims is None:
        return None
    h, w = dims

    inp = np.zeros((h, w), dtype=int)
    out = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    legend = _build_legend(legend_colors, band_h, legend_w)
    side = rng.choice(["left", "right", "top", "bottom"])
    _place_legend(inp, occupied, legend, side, rng)
    out[:, :] = inp

    indexed = list(zip(objects, object_colors))
    indexed.sort(key=lambda item: item[0][0].shape[0] * item[0][0].shape[1], reverse=True)
    for (mask, hole_count), obj_color in indexed:
        positions = _object_positions(h, w, occupied, mask)
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        _paint_object(inp, out, mask, r0, c0, obj_color, legend_colors[hole_count])
        mh, mw = mask.shape
        occupied[max(0, r0 - 1) : min(h, r0 + mh + 1), max(0, c0 - 1) : min(w, c0 + mw + 1)] = True

    if np.array_equal(inp, out):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
