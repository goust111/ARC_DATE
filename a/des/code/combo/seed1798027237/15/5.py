import random
from collections import Counter

import numpy as np


BG = 0
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GENERATE_RETRIES = 600


def _find_monochrome_components(grid: np.ndarray, ignore: np.ndarray | None = None) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    if ignore is None:
        ignore = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or ignore[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = [(r, c)]
            coords = []
            while queue:
                rr, cc = queue.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or ignore[nr, nc]:
                        continue
                    if int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))

            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            comps.append(
                {
                    "color": color,
                    "coords": set(coords),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "area": len(coords),
                }
            )
    return comps


def _mirror(coord: tuple[int, int], orient: str, axis: int) -> tuple[int, int]:
    r, c = coord
    if orient == "v":
        return (r, 2 * axis - c)
    return (2 * axis - r, c)


def _side(coord: tuple[int, int], orient: str, axis: int) -> int:
    r, c = coord
    value = c if orient == "v" else r
    if value < axis:
        return -1
    if value > axis:
        return 1
    return 0


def _axis_candidates(coords: set[tuple[int, int]]) -> list[dict]:
    arr = np.array(sorted(coords), dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    out: list[dict] = []

    for orient, axes in (("v", range(c0, c1 + 1)), ("h", range(r0, r1 + 1))):
        for axis in axes:
            matched = 0
            neg_only = 0
            pos_only = 0
            for coord in coords:
                side = _side(coord, orient, axis)
                if _mirror(coord, orient, axis) in coords:
                    matched += 1
                elif side < 0:
                    neg_only += 1
                elif side > 0:
                    pos_only += 1
            out.append(
                {
                    "orient": orient,
                    "axis": axis,
                    "matched": matched,
                    "neg_only": neg_only,
                    "pos_only": pos_only,
                    "score": (matched, abs(pos_only - neg_only)),
                }
            )
    return out


def _best_axis(coords: set[tuple[int, int]]) -> tuple[dict, int]:
    candidates = _axis_candidates(coords)
    best = max(candidates, key=lambda item: item["score"])
    ties = sum(1 for item in candidates if item["score"] == best["score"])
    return best, ties


def _detect_legend(grid: np.ndarray) -> tuple[dict[int, int], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    counts = Counter(int(x) for x in g.flat if int(x) != BG)
    candidates = []
    corner_specs = [
        (0, 1, list(range(w))),
        (0, 1, list(range(w - 1, -1, -1))),
        (h - 2, h - 1, list(range(w))),
        (h - 2, h - 1, list(range(w - 1, -1, -1))),
    ]

    for top_row, bottom_row, scan_cols in corner_specs:
        run_cols = []
        for c in scan_cols:
            upper = int(g[top_row, c])
            lower = int(g[bottom_row, c])
            if upper != BG and lower != BG:
                run_cols.append(c)
            else:
                break
        if not run_cols:
            continue

        upper_colors = [int(g[top_row, c]) for c in run_cols]
        lower_colors = [int(g[bottom_row, c]) for c in run_cols]
        if len(set(upper_colors)) != len(upper_colors):
            continue
        if len(set(lower_colors)) != len(lower_colors):
            continue
        if set(upper_colors) & set(lower_colors):
            continue
        if not all(counts[color] > 1 for color in upper_colors):
            continue
        if not all(counts[color] == 1 for color in lower_colors):
            continue

        mask = np.zeros_like(g, dtype=bool)
        for c in run_cols:
            mask[top_row, c] = True
            mask[bottom_row, c] = True
        mapping = {int(g[top_row, c]): int(g[bottom_row, c]) for c in run_cols}
        score = (len(run_cols), sum(counts[color] for color in upper_colors))
        candidates.append((score, mapping, mask))

    assert candidates, "legend not found"
    _, mapping, mask = max(candidates, key=lambda item: item[0])
    return mapping, mask


def _fold_component(
    out: np.ndarray,
    coords: set[tuple[int, int]],
    color: int,
    overlap_color: int,
    axis_info: dict,
) -> None:
    orient = axis_info["orient"]
    axis = int(axis_info["axis"])
    move_positive = axis_info["pos_only"] > axis_info["neg_only"]
    assert axis_info["pos_only"] != axis_info["neg_only"], "fold direction is ambiguous"

    stationary: set[tuple[int, int]] = set()
    moving: list[tuple[int, int]] = []
    for coord in coords:
        side = _side(coord, orient, axis)
        if side == 0 or ((side > 0) != move_positive):
            stationary.add(coord)
        else:
            moving.append(coord)

    for r, c in coords:
        out[r, c] = BG
    for r, c in stationary:
        out[r, c] = color
    for coord in moving:
        target = _mirror(coord, orient, axis)
        out[target] = overlap_color if target in stationary else color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold each near-symmetric monochrome object across its best horizontal or
    vertical axis, moving the more protruding side inward. The corner legend
    maps original object colors to the recolor used where the folded side
    overlaps an existing pixel on the stationary side.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    mapping, legend_mask = _detect_legend(g)

    components = _find_monochrome_components(g, ignore=legend_mask)
    colors = [comp["color"] for comp in components]
    assert len(colors) == len(set(colors)), "objects must have distinct colors"
    assert all(color in mapping for color in colors), "every object color must appear in the legend"

    out = g.copy()
    for comp in components:
        axis_info, _ties = _best_axis(comp["coords"])
        _fold_component(out, comp["coords"], comp["color"], mapping[comp["color"]], axis_info)
    return out


def _coords_to_mask(coords: set[tuple[int, int]]) -> np.ndarray:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    h = max(rows) - min(rows) + 1
    w = max(cols) - min(cols) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[r - min(rows), c - min(cols)] = True
    return mask


def _build_object() -> tuple[np.ndarray, str, int] | None:
    for _ in range(200):
        orient = random.choice(["v", "h"])
        side = random.choice([-1, 1])
        height = random.randint(4, 8)
        base_max = random.randint(1, 3)

        arms = [random.randint(0, base_max) for _ in range(height)]
        if not any(length > 0 for length in arms):
            arms[random.randrange(height)] = 1
        if len(set(arms)) == 1:
            idx = random.randrange(height)
            if arms[idx] < base_max:
                arms[idx] += 1
            else:
                arms[idx] -= 1

        extras = [0] * height
        extra_rows = random.sample(range(height), random.randint(1, max(1, height // 2)))
        for row in extra_rows:
            extras[row] = random.randint(1, 2)

        coords: set[tuple[int, int]] = set()
        for r in range(height):
            coords.add((r, 0))
            for d in range(1, arms[r] + 1):
                coords.add((r, -d))
                coords.add((r, d))
            for d in range(arms[r] + 1, arms[r] + extras[r] + 1):
                coords.add((r, side * d))

        if orient == "h":
            coords = {(c, r) for r, c in coords}

        min_r = min(r for r, _ in coords)
        min_c = min(c for _, c in coords)
        coords = {(r - min_r, c - min_c) for r, c in coords}

        axis_info, ties = _best_axis(coords)
        predicted_side = 1 if axis_info["pos_only"] > axis_info["neg_only"] else -1
        if ties != 1:
            continue
        if axis_info["orient"] != orient:
            continue
        if predicted_side != side:
            continue
        if axis_info["matched"] == len(coords):
            continue
        mask = _coords_to_mask(coords)
        if min(mask.shape) < 3:
            continue
        return mask, orient, side
    return None


def _legend_positions(h: int, w: int, count: int, corner: str) -> list[tuple[int, int, int]]:
    if corner == "tl":
        cols = list(range(count))
        return [(0, 1, c) for c in cols]
    if corner == "tr":
        cols = list(range(w - count, w))
        return [(0, 1, c) for c in cols]
    if corner == "bl":
        cols = list(range(count))
        return [(h - 2, h - 1, c) for c in cols]
    cols = list(range(w - count, w))
    return [(h - 2, h - 1, c) for c in cols]


def _can_place(blocked: np.ndarray, top: int, left: int, shape: tuple[int, int]) -> bool:
    h, w = blocked.shape
    obj_h, obj_w = shape
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(h, top + obj_h + 1)
    c1 = min(w, left + obj_w + 1)
    return not blocked[r0:r1, c0:c1].any()


def _mark_blocked(blocked: np.ndarray, top: int, left: int, shape: tuple[int, int]) -> None:
    h, w = blocked.shape
    obj_h, obj_w = shape
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(h, top + obj_h + 1)
    c1 = min(w, left + obj_w + 1)
    blocked[r0:r1, c0:c1] = True


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(2, 4)
    palette = random.sample(list(range(1, 10)), 2 * count)
    object_colors = palette[:count]
    overlap_colors = palette[count:]

    objects = []
    for color in object_colors:
        built = _build_object()
        if built is None:
            return None
        mask, orient, side = built
        objects.append({"color": color, "mask": mask, "orient": orient, "side": side, "area": int(mask.sum())})

    max_h = max(obj["mask"].shape[0] for obj in objects)
    max_w = max(obj["mask"].shape[1] for obj in objects)
    min_h = max(18, max_h + 8, (2 * max_h + 7) if count >= 3 else 0)
    min_w = max(18, max_w + 8, (2 * max_w + 7) if count >= 3 else 0, count + 6)
    if min_h > 30 or min_w > 30:
        return None

    h = random.randint(min_h, 30)
    w = random.randint(min_w, 30)
    grid = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    corner = random.choice(["tl", "tr", "bl", "br"])
    pairs = list(zip(object_colors, overlap_colors))
    random.shuffle(pairs)
    legend_positions = _legend_positions(h, w, count, corner)
    for (orig_color, overlap_color), (top_row, bottom_row, legend_col) in zip(pairs, legend_positions):
        grid[top_row, legend_col] = orig_color
        grid[bottom_row, legend_col] = overlap_color

    for top_row, bottom_row, legend_col in legend_positions:
        blocked[max(0, top_row - 1) : min(h, bottom_row + 2), max(0, legend_col - 1) : min(w, legend_col + 2)] = True

    objects.sort(key=lambda item: item["area"], reverse=True)
    for obj in objects:
        mask = obj["mask"]
        obj_h, obj_w = mask.shape
        row_choices = list(range(3, h - obj_h - 2))
        col_choices = list(range(3, w - obj_w - 2))
        if not row_choices or not col_choices:
            return None

        placed = False
        for _ in range(250):
            top = random.choice(row_choices)
            left = random.choice(col_choices)
            if not _can_place(blocked, top, left, mask.shape):
                continue
            rr, cc = np.where(mask)
            grid[top + rr, left + cc] = obj["color"]
            _mark_blocked(blocked, top, left, mask.shape)
            placed = True
            break
        if not placed:
            return None

    mapping, legend_mask = _detect_legend(grid)
    components = _find_monochrome_components(grid, ignore=legend_mask)
    if len(components) != count:
        return None
    for comp in components:
        axis_info, ties = _best_axis(comp["coords"])
        if ties != 1 or axis_info["neg_only"] == axis_info["pos_only"]:
            return None
        if mapping.get(comp["color"]) is None:
            return None
    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
