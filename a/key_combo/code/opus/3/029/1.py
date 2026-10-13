from __future__ import annotations

from dataclasses import dataclass
import itertools

import numpy as np


BG = 0
WATER = 1
HOLE = 9
BASIN_COLORS = (1, 3, 7)


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class Basin:
    color: int
    top: int
    left: int
    mask: np.ndarray
    left_profile: tuple[int, ...]
    holes: tuple[Coord, ...]

    @property
    def height(self) -> int:
        return int(self.mask.shape[0])

    @property
    def width(self) -> int:
        return int(self.mask.shape[1])

    @property
    def bbox(self) -> BBox:
        return (
            int(self.top),
            int(self.left),
            int(self.top + self.height - 1),
            int(self.left + self.width - 1),
        )


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(int(r - 1), int(c)), (int(r + 1), int(c)), (int(r), int(c - 1)), (int(r), int(c + 1))]


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.array(coords, dtype=int))
    return comps


def _axis_candidates(coords: np.ndarray) -> list[int]:
    coord_set = {tuple(map(int, rc)) for rc in coords}
    _r0, c0, _r1, c1 = _bbox_coords(coords)
    valid: list[int] = []
    for axis_sum in range(2 * int(c0), 2 * int(c1) + 1):
        ok = True
        for r, c in coord_set:
            mc = int(axis_sum - c)
            if (r, mc) not in coord_set:
                ok = False
                break
        if ok:
            valid.append(int(axis_sum))
    return valid


def _split_basins_and_water(grid: np.ndarray) -> tuple[list[np.ndarray], list[Coord]]:
    comps = _components(grid != BG)
    basins: list[np.ndarray] = []
    waters: list[Coord] = []
    for coords in comps:
        vals = [int(grid[r, c]) for r, c in coords]
        if coords.shape[0] == 1 and vals[0] == WATER:
            waters.append((int(coords[0, 0]), int(coords[0, 1])))
        else:
            basins.append(coords)
    basins.sort(key=lambda pts: _bbox_coords(pts))
    waters.sort()
    return basins, waters


def _repair_walls(grid: np.ndarray) -> tuple[np.ndarray, list[Coord]]:
    g = np.asarray(grid, dtype=int)
    basins, waters = _split_basins_and_water(g)
    walls = g.copy()
    for r, c in waters:
        walls[r, c] = BG

    for coords in basins:
        visible = {int(walls[r, c]) for r, c in coords if int(walls[r, c]) not in (BG, HOLE)}
        assert len(visible) == 1
        basin_color = next(iter(visible))
        r0, c0, r1, c1 = _bbox_coords(coords)
        axis_sums = _axis_candidates(coords)
        assert axis_sums == [c0 + c1]
        axis_sum = axis_sums[0]

        for r, c in coords:
            if int(walls[r, c]) != HOLE:
                continue
            rr = int(r0 + r1 - r)
            mc = int(axis_sum - c)
            orbit = [
                (int(r), int(mc)),
                (int(rr), int(c)),
                (int(rr), int(mc)),
            ]
            colors = {
                int(walls[rr2, cc2])
                for rr2, cc2 in orbit
                if _in_bounds(walls.shape, rr2, cc2) and int(walls[rr2, cc2]) not in (BG, HOLE)
            }
            if not colors:
                colors = {int(basin_color)}
            assert colors == {int(basin_color)}
            walls[r, c] = int(basin_color)

    return walls, waters


def _touch_colors(walls: np.ndarray, pos: Coord) -> set[int]:
    r, c = map(int, pos)
    return {
        int(walls[nr, nc])
        for nr, nc in _neighbors4(r, c)
        if _in_bounds(walls.shape, nr, nc) and int(walls[nr, nc]) != BG
    }


def _touch_color(walls: np.ndarray, pos: Coord) -> int:
    colors = _touch_colors(walls, pos)
    assert len(colors) == 1
    return next(iter(colors))


def _is_empty(walls: np.ndarray, water_mask: np.ndarray, r: int, c: int) -> bool:
    return _in_bounds(walls.shape, r, c) and int(walls[r, c]) == BG and not bool(water_mask[r, c])


def _settle_one(walls: np.ndarray, water_mask: np.ndarray, start: Coord) -> tuple[Coord, bool]:
    h, w = walls.shape
    r, c = map(int, start)
    used_diag = False
    for _ in range(h * w):
        if r + 1 < h and _is_empty(walls, water_mask, r + 1, c):
            r += 1
            continue

        options: list[Coord] = []
        if r + 1 < h and c - 1 >= 0 and _is_empty(walls, water_mask, r + 1, c - 1):
            options.append((int(r + 1), int(c - 1)))
        if r + 1 < h and c + 1 < w and _is_empty(walls, water_mask, r + 1, c + 1):
            options.append((int(r + 1), int(c + 1)))
        if len(options) == 1:
            r, c = options[0]
            used_diag = True
            continue
        break
    else:
        raise AssertionError("water did not settle")
    return (int(r), int(c)), bool(used_diag)


def _simulate_water(walls: np.ndarray, starts: list[Coord]) -> tuple[list[Coord], list[bool]]:
    positions = [tuple(map(int, p)) for p in starts]
    water_mask = np.zeros(walls.shape, dtype=bool)
    for r, c in positions:
        assert _is_empty(walls, water_mask, r, c)
        water_mask[r, c] = True

    used_diag = [False] * len(positions)
    h, w = walls.shape
    for _ in range(h * w):
        moved = False
        order = sorted(range(len(positions)), key=lambda idx: (-positions[idx][0], positions[idx][1]))
        for idx in order:
            r, c = positions[idx]
            if not bool(water_mask[r, c]):
                continue
            water_mask[r, c] = False
            nxt, diag = _settle_one(walls, water_mask, (r, c))
            positions[idx] = nxt
            used_diag[idx] = bool(used_diag[idx] or diag)
            water_mask[nxt[0], nxt[1]] = True
            moved = bool(moved or nxt != (r, c))
        if not moved:
            break
    else:
        raise AssertionError("water simulation did not converge")

    return positions, used_diag


def main(grid: np.ndarray) -> np.ndarray:
    """Repair maroon wall gaps by basin symmetry, then let the water settle and inherit its basin color.

    Input structure:
    - Black background contains 2-3 large 4-connected basin-wall components and 1-3 isolated blue water pixels.
    - Each basin wall is monochrome except for maroon gap pixels and is vertically symmetric inside its own bbox.
    - The repaired basins are open upward containers; water starts in empty space above them.

    Rule:
    - Treat maroon as damaged wall material, not final content.
    - For each basin component, replace every maroon pixel with the unique visible color in its vertical-mirror / 180-degree orbit inside that basin.
    - After repair, iteratively move each water pixel: fall straight down through background, or if blocked below slide exactly one step to the unique open lower diagonal; otherwise stop. Use bottom-to-top, then left-to-right update order, and let water pixels block each other.
    - Recolor every resting water pixel to the unique basin-wall color touching it in the 4-neighborhood.

    Key insight:
    - The maroon pixels are physical wall breaches. Symmetry repair changes the collision topology first; only then does gravity reveal which basin each droplet belongs to.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    walls, starts = _repair_walls(g)
    final_positions, _diag = _simulate_water(walls, starts)
    out = walls.copy()
    for pos in final_positions:
        out[pos] = int(_touch_color(walls, pos))
    return out


def _sample_left_profile(height: int, width: int, rng: np.random.Generator, *, force_taper: bool) -> tuple[int, ...]:
    assert width % 2 == 1 and width >= 7
    max_inset = min(3, (width - 3) // 2)
    if force_taper:
        top_inset = int(rng.integers(1, max_inset + 1))
    else:
        top_inset = int(rng.integers(0, max_inset + 1))

    if top_inset == 0:
        return tuple(0 for _ in range(height - 1))

    drop_rows = sorted(rng.choice(np.arange(1, height - 1), size=top_inset, replace=False).tolist())
    cur = int(top_inset)
    profile: list[int] = []
    drop_ptr = 0
    for r in range(height - 1):
        profile.append(int(cur))
        if drop_ptr < len(drop_rows) and r + 1 == drop_rows[drop_ptr]:
            cur -= 1
            drop_ptr += 1
    profile[-1] = 0
    for r in range(len(profile) - 2, -1, -1):
        profile[r] = max(profile[r], profile[r + 1])
    return tuple(map(int, profile))


def _draw_basin_mask(height: int, width: int, left_profile: tuple[int, ...]) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    right_profile = [int(width - 1 - c) for c in left_profile]

    mask[0, int(left_profile[0])] = True
    mask[0, int(right_profile[0])] = True
    for r in range(height - 2):
        lc0 = int(left_profile[r])
        lc1 = int(left_profile[r + 1])
        rc0 = int(right_profile[r])
        rc1 = int(right_profile[r + 1])

        for c in range(min(lc0, lc1), max(lc0, lc1) + 1):
            mask[r + 1, c] = True
        for c in range(min(rc0, rc1), max(rc0, rc1) + 1):
            mask[r + 1, c] = True
        mask[r, lc0] = True
        mask[r, rc0] = True

    base_left = int(left_profile[-1])
    base_right = int(width - 1 - base_left)
    mask[height - 1, base_left : base_right + 1] = True
    return mask


def _candidate_holes(mask: np.ndarray, left_profile: tuple[int, ...]) -> tuple[list[Coord], list[Coord]]:
    height, width = mask.shape
    left_cells: list[Coord] = []
    right_cells: list[Coord] = []
    for r in range(1, height - 1):
        lc = int(left_profile[r if r < height - 1 else height - 2])
        rc = int(width - 1 - lc)
        if bool(mask[r, lc]):
            left_cells.append((int(r), int(lc)))
        if bool(mask[r, rc]):
            right_cells.append((int(r), int(rc)))
    return left_cells, right_cells


def _make_basin(color: int, top: int, left: int, height: int, width: int, rng: np.random.Generator, *,
                force_taper: bool, allow_pair: bool, force_floor_hole: bool, force_ledges: bool) -> Basin:
    left_profile = _sample_left_profile(height, width, rng, force_taper=force_taper)
    mask = _draw_basin_mask(height, width, left_profile)
    if force_ledges:
        top_left = int(left_profile[0])
        top_right = int(width - 1 - top_left)
        if 1 < top_right - top_left:
            mask[1, top_left + 1] = True
            mask[1, top_right - 1] = True
    left_cells, right_cells = _candidate_holes(mask, left_profile)
    assert left_cells and right_cells

    if force_floor_hole:
        base_row = int(height - 1)
        base_left = int(left_profile[-1])
        base_right = int(width - 1 - left_profile[-1])
        floor_cols = [c for c in range(base_left + 1, base_right) if c != width // 2]
        if not floor_cols:
            floor_cols = list(range(base_left, base_right + 1))
        hole_col = int(rng.choice(floor_cols))
        holes = [(base_row, hole_col)]
        if rng.random() < 0.5:
            side_col = hole_col - 1 if hole_col > base_left + 1 else hole_col + 1
            if base_left <= side_col <= base_right:
                holes.append((base_row, int(side_col)))
        holes_local = tuple(dict.fromkeys(holes))
        holes_global = tuple((int(top + r), int(left + c)) for r, c in holes_local)
        return Basin(
            color=int(color),
            top=int(top),
            left=int(left),
            mask=mask,
            left_profile=tuple(map(int, left_profile)),
            holes=holes_global,
        )

    preferred_rows = [
        r
        for r in range(1, height - 1)
        if left_profile[r - 1] != left_profile[r]
        or (r + 1 < height - 1 and left_profile[r] != left_profile[r + 1])
    ]
    upper_limit = max(2, height // 2)
    preferred_rows = [r for r in preferred_rows if r <= upper_limit]
    if not preferred_rows:
        preferred_rows = [r for r in range(1, height - 1) if r <= upper_limit]

    use_pair = bool(allow_pair and rng.random() < 0.55 and len(left_cells) >= 2)
    if use_pair:
        row_choices = preferred_rows
        hole_row = int(rng.choice(row_choices))
        lc = int(left_profile[hole_row])
        rc = int(width - 1 - lc)
        holes_local = ((hole_row, lc), (hole_row, rc))
    else:
        side_cells = left_cells if bool(rng.integers(0, 2)) else right_cells
        side_lookup = {r: c for r, c in side_cells}
        row_choices = [r for r in preferred_rows if r in side_lookup]
        if not row_choices:
            row_choices = [r for r, _c in side_cells]
        hole_row = int(rng.choice(row_choices))
        hole_col = int(side_lookup[hole_row])
        holes = [(int(hole_row), int(hole_col))]
        if rng.random() < 0.65:
            for rr, cc in side_cells:
                if abs(int(rr) - int(hole_row)) == 1:
                    holes.append((int(rr), int(cc)))
                    break
        holes_local = tuple(dict.fromkeys(holes))

    holes_global = tuple((int(top + r), int(left + c)) for r, c in holes_local)
    return Basin(
        color=int(color),
        top=int(top),
        left=int(left),
        mask=mask,
        left_profile=tuple(map(int, left_profile)),
        holes=holes_global,
    )


def _render_walls(shape: tuple[int, int], basins: list[Basin], *, show_holes: bool) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for basin in basins:
        h, w = basin.mask.shape
        view = grid[basin.top : basin.top + h, basin.left : basin.left + w]
        view[basin.mask] = int(basin.color)
        grid[basin.top : basin.top + h, basin.left : basin.left + w] = view
    if show_holes:
        for basin in basins:
            for r, c in basin.holes:
                grid[r, c] = HOLE
    return grid


def _isolated_bg_cells(grid: np.ndarray, basin: Basin) -> list[Coord]:
    h, w = grid.shape
    r0, c0, _r1, c1 = basin.bbox
    starts: list[Coord] = []
    row_hi = max(0, int(r0 - 1))
    for r in range(row_hi):
        for c in range(max(0, c0 - 2), min(w, c1 + 3)):
            if int(grid[r, c]) != BG:
                continue
            nbr_vals = [
                int(grid[nr, nc])
                for nr, nc in _neighbors4(r, c)
                if _in_bounds(grid.shape, nr, nc)
            ]
            if any(v != BG for v in nbr_vals):
                continue
            starts.append((int(r), int(c)))
    return starts


def _single_drop_candidate(walls: np.ndarray, hole_walls: np.ndarray, basin: Basin, start: Coord) -> dict | None:
    finals, diags = _simulate_water(walls, [start])
    final = finals[0]
    diag = bool(diags[0])
    r0, c0, r1, c1 = basin.bbox
    if not (r0 < final[0] < r1 and c0 < final[1] < c1):
        return None
    colors = _touch_colors(walls, final)
    if colors != {int(basin.color)}:
        return None
    broken_final = _simulate_water(hole_walls, [start])[0][0]
    return {
        "start": tuple(map(int, start)),
        "final": tuple(map(int, final)),
        "diag": bool(diag),
        "repair_needed": tuple(map(int, final)) != tuple(map(int, broken_final)),
    }


def _choose_water_starts(
    walls: np.ndarray,
    hole_walls: np.ndarray,
    input_walls: np.ndarray,
    basins: list[Basin],
    rng: np.random.Generator,
) -> list[Coord] | None:
    per_basin: list[list[dict]] = []
    for basin in basins:
        cands = []
        for start in _isolated_bg_cells(input_walls, basin):
            cand = _single_drop_candidate(walls, hole_walls, basin, start)
            if cand is not None:
                cands.append(cand)
        if not cands:
            return None
        rng.shuffle(cands)
        per_basin.append(cands[:16])

    for combo in itertools.product(*per_basin):
        starts = [item["start"] for item in combo]
        if len(set(starts)) != len(starts):
            continue
        finals, diags = _simulate_water(walls, starts)
        broken_finals, _ = _simulate_water(hole_walls, starts)
        if not any(a != b for a, b in zip(finals, broken_finals, strict=True)):
            continue
        if not any(diags):
            continue
        final_colors = [_touch_color(walls, pos) for pos in finals]
        if not any(color != WATER for color in final_colors):
            continue
        return [tuple(map(int, p)) for p in starts]
    return None


def _build_scene(rng: np.random.Generator) -> np.ndarray | None:
    basin_count = 3 if rng.random() < 0.62 else 2
    colors = [1, 3, 7]
    rng.shuffle(colors)
    if basin_count == 2:
        colors = [1, colors[0] if colors[0] != 1 else colors[1]]
    else:
        colors = colors[:3]

    if basin_count == 3:
        width_choices = [7, 7, 9]
        gap_hi = 3
        margin_hi = 2
    else:
        width_choices = [7, 9, 11]
        gap_hi = 4
        margin_hi = 3

    widths = [int(rng.choice(width_choices)) for _ in range(basin_count)]
    heights = [int(rng.integers(5, 9)) for _ in range(basin_count)]
    gaps = [int(rng.integers(2, gap_hi + 1)) for _ in range(max(0, basin_count - 1))]
    left_margin = int(rng.integers(1, margin_hi + 1))
    right_margin = int(rng.integers(1, margin_hi + 1))
    total_w = left_margin + sum(widths) + sum(gaps) + right_margin
    if total_w > 30:
        return None

    height_min = max(16, max(heights) + 7)
    height_max = min(28, max(heights) + 12)
    grid_h = int(rng.integers(height_min, height_max + 1))
    bottom = int(grid_h - rng.integers(2, 4))
    if bottom - max(heights) < 5:
        return None

    basins: list[Basin] = []
    cursor = int(left_margin)
    pair_idx = int(rng.integers(0, basin_count)) if rng.random() < 0.45 else -1
    taper_idx = int(rng.integers(0, basin_count))
    leak_idx = int(rng.integers(0, basin_count))
    diag_idx = int(rng.integers(0, basin_count))
    if basin_count > 1 and diag_idx == leak_idx:
        diag_idx = int((diag_idx + 1) % basin_count)
    for idx, color in enumerate(colors):
        top = int(bottom - heights[idx] + 1)
        basin = _make_basin(
            int(color),
            int(top),
            int(cursor),
            int(heights[idx]),
            int(widths[idx]),
            rng,
            force_taper=(idx == taper_idx or idx == diag_idx),
            allow_pair=(idx == pair_idx),
            force_floor_hole=(idx == leak_idx),
            force_ledges=(idx == diag_idx),
        )
        basins.append(basin)
        if idx < basin_count - 1:
            cursor += int(widths[idx] + gaps[idx])

    repaired_walls = _render_walls((grid_h, total_w), basins, show_holes=False)
    broken_walls = _render_walls((grid_h, total_w), basins, show_holes=False)
    input_walls = _render_walls((grid_h, total_w), basins, show_holes=True)

    for basin in basins:
        for r, c in basin.holes:
            broken_walls[r, c] = BG

    starts = _choose_water_starts(repaired_walls, broken_walls, input_walls, basins, rng)
    if starts is None:
        return None

    inp = input_walls.copy()
    for r, c in starts:
        inp[r, c] = WATER
    return inp


def _validate_scene(inp: np.ndarray) -> bool:
    if inp is None or inp.ndim != 2 or max(inp.shape) > 30:
        return False
    try:
        out = main(inp)
    except Exception:
        return False

    if not np.any(inp == HOLE):
        return False
    if np.any(out == HOLE):
        return False

    try:
        repaired_walls, starts = _repair_walls(inp)
        hole_walls = repaired_walls.copy()
        hole_walls[inp == HOLE] = BG
        fixed_positions, diags = _simulate_water(repaired_walls, starts)
        broken_positions, _ = _simulate_water(hole_walls, starts)
        if not any(a != b for a, b in zip(fixed_positions, broken_positions, strict=True)):
            return False
        if not any(diags):
            return False
        final_colors = [_touch_color(repaired_walls, pos) for pos in fixed_positions]
        if not any(color != WATER for color in final_colors):
            return False

        basins, waters = _split_basins_and_water(inp)
        if len(basins) not in (2, 3) or len(waters) not in (2, 3):
            return False
        if any(_axis_candidates(coords) != [_bbox_coords(coords)[1] + _bbox_coords(coords)[3]] for coords in basins):
            return False

        expected = repaired_walls.copy()
        for pos, color in zip(fixed_positions, final_colors, strict=True):
            expected[pos] = int(color)
        if not np.array_equal(out, expected):
            return False
        return True
    except Exception:
        return False


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        inp = _build_scene(rng)
        if inp is not None and _validate_scene(inp):
            return inp
    raise RuntimeError("failed to generate a valid basin-repair gravity input")
