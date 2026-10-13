import random
from collections import Counter

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
LEGEND_COLOR_POOL = (BLUE, RED, GREEN, YELLOW, MAGENTA)

NEIGH4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGH8 = NEIGH4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]

MATCH_AREA_OPTIONS = {
    1: [8, 9, 10, 12],
    2: [14, 15, 16, 18],
    3: [18, 20, 21, 24],
}
DISTRACTOR_AREA_OPTIONS = {
    1: [11],
    2: [13, 17],
    3: [19, 23],
}
FRAME_AREA_POOL = [6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 24]


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    neigh = NEIGH4 if connectivity == 4 else NEIGH8
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
                for dr, dc in neigh:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _points_to_mask(points: set[tuple[int, int]]) -> np.ndarray:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    r0, c0 = min(rows), min(cols)
    r1, c1 = max(rows), max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in points:
        mask[r - r0, c - c0] = True
    return mask


def _hole_count(mask: np.ndarray) -> int:
    if mask.size == 0:
        return 0
    h, w = mask.shape
    holes = 0
    for comp in _components(~mask, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        holes += 1
    return holes


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _extract_component_mask(points: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(points)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in points:
        mask[r - r0, c - c0] = True
    return mask


def _detect_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for pts in _components(grid == ORANGE, connectivity=4):
        r0, c0, r1, c1 = _bbox(pts)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        sh, sw = sub.shape
        if sh < 4 or sw < 4:
            continue
        border_mask = np.zeros_like(sub, dtype=bool)
        border_mask[0, :] = True
        border_mask[-1, :] = True
        border_mask[:, 0] = True
        border_mask[:, -1] = True
        if not np.all(sub[border_mask] == ORANGE):
            continue
        if np.any(sub[1:-1, 1:-1] == ORANGE):
            continue
        if len(pts) != int(border_mask.sum()):
            continue
        ih, iw = sh - 2, sw - 2
        frames.append(
            {
                "area": int(ih * iw),
                "inner": (r0 + 1, c0 + 1, ih, iw),
            }
        )
    return frames


def _legend_mapping(grid: np.ndarray, bg: int) -> dict[int, int]:
    mapping: dict[int, int] = {}
    special = {bg, GREY, ORANGE}
    for color in np.unique(grid):
        color = int(color)
        if color in special:
            continue
        for pts in _components(grid == color, connectivity=8):
            mask = _extract_component_mask(pts)
            holes = _hole_count(mask)
            if holes > 0:
                mapping[holes] = color
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    out = g.copy()

    frames = _detect_frames(g)
    area_to_frames: dict[int, list[dict]] = {}
    for frame in frames:
        area_to_frames.setdefault(int(frame["area"]), []).append(frame)

    legend = _legend_mapping(g, bg)
    used_areas: set[int] = set()

    for pts in _components(g == GREY, connectivity=8):
        for r, c in pts:
            out[r, c] = bg
        area = len(pts)
        mask = _extract_component_mask(pts)
        holes = _hole_count(mask)
        frames_for_area = area_to_frames.get(int(area), [])
        if len(frames_for_area) != 1 or area in used_areas or holes not in legend:
            continue
        used_areas.add(int(area))
        inner_r, inner_c, inner_h, inner_w = frames_for_area[0]["inner"]
        out[inner_r : inner_r + inner_h, inner_c : inner_c + inner_w] = int(legend[holes])

    return out


def _base_holey_mask(hole_count: int) -> np.ndarray:
    width = 2 * hole_count + 1
    mask = np.ones((3, width), dtype=bool)
    for idx in range(hole_count):
        mask[1, 1 + 2 * idx] = False
    return mask


def _random_transform(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    out = np.array(mask, copy=True)
    if rng.random() < 0.5:
        out = np.fliplr(out)
    if rng.random() < 0.5:
        out = np.flipud(out)
    out = np.rot90(out, rng.randrange(4))
    return np.array(out, copy=True)


def _grow_holey_shape(hole_count: int, target_area: int, rng: random.Random) -> np.ndarray:
    base = _base_holey_mask(hole_count)
    base_area = int(base.sum())
    if target_area < base_area:
        raise ValueError("target_area too small")
    for _ in range(50):
        points = {tuple(map(int, p)) for p in np.argwhere(base)}
        while len(points) < target_area:
            candidates: set[tuple[int, int]] = set()
            for r, c in points:
                for dr, dc in NEIGH4:
                    cand = (r + dr, c + dc)
                    if cand not in points:
                        candidates.add(cand)
            cand_list = list(candidates)
            rng.shuffle(cand_list)
            placed = False
            for cand in cand_list:
                trial = set(points)
                trial.add(cand)
                trial_mask = _points_to_mask(trial)
                if _hole_count(trial_mask) != hole_count:
                    continue
                if max(trial_mask.shape) > 10:
                    continue
                points = trial
                placed = True
                break
            if not placed:
                break
        if len(points) != target_area:
            continue
        mask = _random_transform(_points_to_mask(points), rng)
        if int(mask.sum()) == target_area and _hole_count(mask) == hole_count:
            return mask
    raise ValueError("failed to grow shape")


def _colorize(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = int(color)
    return patch


def _factor_pairs(area: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for h in range(2, int(area**0.5) + 1):
        if area % h != 0:
            continue
        w = area // h
        if w < 2 or max(h, w) > 7:
            continue
        pairs.append((h, w))
    return pairs


def _choose_match_areas(hole_counts: list[int], rng: random.Random) -> dict[int, int] | None:
    order = hole_counts[:]
    rng.shuffle(order)
    chosen: dict[int, int] = {}
    used: set[int] = set()
    for hole_count in order:
        options = MATCH_AREA_OPTIONS[hole_count][:]
        rng.shuffle(options)
        area = next((value for value in options if value not in used and _factor_pairs(value)), None)
        if area is None:
            return None
        chosen[hole_count] = int(area)
        used.add(int(area))
    return chosen


def _frame_patch(inner_h: int, inner_w: int) -> np.ndarray:
    patch = np.full((inner_h + 2, inner_w + 2), BG, dtype=int)
    patch[0, :] = ORANGE
    patch[-1, :] = ORANGE
    patch[:, 0] = ORANGE
    patch[:, -1] = ORANGE
    return patch


def _place_patch(
    grid: np.ndarray,
    reserved: np.ndarray,
    patch: np.ndarray,
    rng: random.Random,
    row_lo: int,
    row_hi: int,
    col_lo: int,
    col_hi: int,
    pad: int = 1,
    tries: int = 300,
) -> tuple[int, int] | None:
    gh, gw = grid.shape
    ph, pw = patch.shape
    row_lo = max(0, row_lo)
    col_lo = max(0, col_lo)
    row_hi = min(row_hi, gh - ph)
    col_hi = min(col_hi, gw - pw)
    if row_hi < row_lo or col_hi < col_lo:
        return None
    for _ in range(tries):
        top = rng.randint(row_lo, row_hi)
        left = rng.randint(col_lo, col_hi)
        rr0 = max(0, top - pad)
        cc0 = max(0, left - pad)
        rr1 = min(gh, top + ph + pad)
        cc1 = min(gw, left + pw + pad)
        if reserved[rr0:rr1, cc0:cc1].any():
            continue
        region = grid[top : top + ph, left : left + pw]
        if np.any((patch != BG) & (region != BG)):
            continue
        region[patch != BG] = patch[patch != BG]
        reserved[rr0:rr1, cc0:cc1] = True
        return top, left
    return None


def _sample_unmatched_areas(count: int, blocked: set[int], rng: random.Random) -> list[int] | None:
    options = [area for area in FRAME_AREA_POOL if area not in blocked and _factor_pairs(area)]
    if len(options) < count:
        return None
    rng.shuffle(options)
    return [int(v) for v in options[:count]]


def _expected_output(
    grid: np.ndarray,
    frame_records: dict[int, dict],
    matched_sprites: list[dict],
    all_sprite_cells: list[list[tuple[int, int]]],
    legend_colors: dict[int, int],
) -> np.ndarray:
    out = np.array(grid, copy=True)
    for cells in all_sprite_cells:
        for r, c in cells:
            out[r, c] = BG
    for spec in matched_sprites:
        frame = frame_records[spec["area"]]
        r0, c0, ih, iw = frame["inner"]
        out[r0 : r0 + ih, c0 : c0 + iw] = int(legend_colors[spec["holes"]])
    return out


def _absolute_cells(mask: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    return [(top + int(r), left + int(c)) for r, c in np.argwhere(mask)]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    frame_count = rng.choice([3, 4])
    matched_count = rng.choice([2, 3])
    if matched_count > frame_count:
        matched_count = frame_count

    min_size = 20 if frame_count == 3 and matched_count == 2 else 22
    h = rng.randint(min_size, 26)
    w = rng.randint(min_size, 26)
    grid = np.full((h, w), BG, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    matched_holes = rng.sample([1, 2, 3], matched_count)
    match_areas = _choose_match_areas(matched_holes, rng)
    if match_areas is None:
        return None

    legend_colors = {holes: color for holes, color in zip(matched_holes, rng.sample(LEGEND_COLOR_POOL, matched_count))}

    distractor_area = None
    distractor_spec = None
    if rng.random() < 0.5:
        distractor_holes = rng.choice([1, 2, 3])
        distractor_options = DISTRACTOR_AREA_OPTIONS[distractor_holes][:]
        rng.shuffle(distractor_options)
        distractor_area = int(distractor_options[0])
        try:
            distractor_mask = _grow_holey_shape(distractor_holes, distractor_area, rng)
        except ValueError:
            return None
        distractor_spec = {
            "holes": int(distractor_holes),
            "area": int(distractor_area),
            "mask": distractor_mask,
        }

    blocked_areas = set(match_areas.values())
    if distractor_area is not None:
        blocked_areas.add(int(distractor_area))
    unmatched_areas = _sample_unmatched_areas(frame_count - matched_count, blocked_areas, rng)
    if unmatched_areas is None:
        return None

    legend_items = list(legend_colors.items())
    rng.shuffle(legend_items)
    for holes, color in legend_items:
        legend_target_area = int(_base_holey_mask(holes).sum()) + rng.randint(0, 2)
        try:
            legend_mask = _grow_holey_shape(holes, legend_target_area, rng)
        except ValueError:
            return None
        patch = _colorize(legend_mask, color)
        row_hi = min(2, h - patch.shape[0])
        pos = _place_patch(grid, reserved, patch, rng, 0, row_hi, 0, w - patch.shape[1], pad=1, tries=300)
        if pos is None:
            return None

    frame_areas = list(match_areas.values()) + unmatched_areas
    rng.shuffle(frame_areas)
    frame_records: dict[int, dict] = {}
    frame_row_lo = max(h // 2 - 1, 8)
    for area in frame_areas:
        pairs = _factor_pairs(int(area))
        if not pairs:
            return None
        inner_h, inner_w = rng.choice(pairs)
        if rng.random() < 0.5:
            inner_h, inner_w = inner_w, inner_h
        patch = _frame_patch(inner_h, inner_w)
        pos = _place_patch(
            grid,
            reserved,
            patch,
            rng,
            frame_row_lo,
            h - patch.shape[0],
            0,
            w - patch.shape[1],
            pad=1,
            tries=400,
        )
        if pos is None:
            return None
        top, left = pos
        frame_records[int(area)] = {"inner": (top + 1, left + 1, inner_h, inner_w)}

    matched_sprites: list[dict] = []
    for holes in matched_holes:
        area = int(match_areas[holes])
        try:
            sprite_mask = _grow_holey_shape(holes, area, rng)
        except ValueError:
            return None
        matched_sprites.append({"holes": int(holes), "area": area, "mask": sprite_mask})
    rng.shuffle(matched_sprites)

    all_sprite_cells: list[list[tuple[int, int]]] = []
    sprite_specs = matched_sprites[:]
    if distractor_spec is not None:
        sprite_specs.append(distractor_spec)
    rng.shuffle(sprite_specs)

    for spec in sprite_specs:
        patch = _colorize(spec["mask"], GREY)
        pos = _place_patch(grid, reserved, patch, rng, 3, h - patch.shape[0], 0, w - patch.shape[1], pad=1, tries=500)
        if pos is None:
            return None
        top, left = pos
        cells = _absolute_cells(spec["mask"], top, left)
        all_sprite_cells.append(cells)

    expected = _expected_output(grid, frame_records, matched_sprites, all_sprite_cells, legend_colors)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    sig = Counter(grid.flatten())
    if int(sig[BG]) <= max(int(sig[GREY]), int(sig[ORANGE])):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(500):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
