import random
from collections import defaultdict

import numpy as np


BG = 0
GREY = 5
OBJECT_COLORS = (1, 2, 3, 4)
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    for dr, dc in NEIGHBORS4:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _tile_clean(grid: np.ndarray) -> np.ndarray:
    out = np.tile(grid, (2, 2))
    out[out == GREY] = BG
    return out


def _expand_axis(lo: int, hi: int, size: int, wrap: bool) -> list[int]:
    if wrap:
        return sorted({idx % size for idx in range(lo, hi + 1)})
    start = max(0, lo)
    end = min(size - 1, hi)
    if start > end:
        return []
    return list(range(start, end + 1))


def _extract_objects(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    specs: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if v not in (BG, GREY)):
        comps = _components(grid == color)
        assert len(comps) == 1, "each foreground color must form exactly one object"
        coords = comps[0]
        r0, c0, r1, c1 = _bbox(coords)
        specs.append(
            {
                "color": int(color),
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "area": int(coords.shape[0]),
                "touch_right": bool(np.any(coords[:, 1] == w - 1)),
                "touch_bottom": bool(np.any(coords[:, 0] == h - 1)),
                "touch_left": bool(np.any(coords[:, 1] == 0)),
                "touch_top": bool(np.any(coords[:, 0] == 0)),
            }
        )
    return specs


def _logical_component_claims(grid: np.ndarray) -> tuple[np.ndarray, dict[tuple[int, int], list[tuple[int, int]]], list[dict]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    tiled = _tile_clean(g)
    claims: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    specs = _extract_objects(g)

    for spec in specs:
        touch_right = spec["touch_right"]
        touch_bottom = spec["touch_bottom"]
        if not touch_right and not touch_bottom:
            continue

        r0, c0, r1, c1 = spec["bbox"]
        rows = _expand_axis(r0 - 1, r1 + 1, h, wrap=touch_bottom)
        cols = _expand_axis(c0 - 1, c1 + 1, w, wrap=touch_right)
        color = int(spec["color"])
        area = int(spec["area"])

        if touch_right and touch_bottom:
            groups = [([0, 1], [0, 1], 4 * area)]
        elif touch_right:
            groups = [([rb], [0, 1], 2 * area) for rb in (0, 1)]
        else:
            groups = [([0, 1], [cb], 2 * area) for cb in (0, 1)]

        for row_blocks, col_blocks, priority_area in groups:
            for rb in row_blocks:
                for cb in col_blocks:
                    for rr in rows:
                        for cc in cols:
                            tr = rr + rb * h
                            tc = cc + cb * w
                            if tiled[tr, tc] == BG:
                                claims[(tr, tc)].append((priority_area, color))

    return tiled, claims, specs


def _solve(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[tuple[int, int], list[tuple[int, int]]], list[dict]]:
    tiled, claims, specs = _logical_component_claims(grid)
    out = tiled.copy()
    for (r, c), contenders in claims.items():
        priority_area, color = min(contenders, key=lambda item: (item[0], item[1]))
        _ = priority_area
        out[r, c] = color
    return out, tiled, claims, specs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile the grid into four quadrants, erase grey noise, and enlarge only the colors whose tiled copies merge across quadrant boundaries.

    Input structure:
    - The grid has black background, one interior grey distractor pixel, and 2 to 4 small monochrome 4-connected objects in distinct colors.
    - At least one object lies safely inside the grid, while at least one object reaches the right edge, the bottom edge, or both.
    - Each foreground color appears in exactly one object, so edge contact is read per-color without ambiguity.

    Rule:
    - Build a 2x2 tiled output and replace every grey pixel with black.
    - For each non-grey foreground color, treat touching the right edge and/or bottom edge as causing that color's tiled copies to merge across the corresponding quadrant seams.
    - Colors whose tiled copy-groups stay split into four components remain unchanged.
    - For every merged logical component, expand its bounding box by one cell in the merged tiling topology and fill only black cells in that expanded region with the component's color.
    - If two colors try to fill the same black cell, the component with fewer pixels wins that conflict.

    Key insight:
    - The same 2x2 tiling both replicates the scene and reveals which colors should grow: interior objects stay as four independent copies, while edge-touching objects merge across seams and therefore trigger the bbox expansion.
    """
    out, _, _, _ = _solve(np.asarray(grid, dtype=int))
    return out


def _normalize(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return sorted((r - min_r, c - min_c) for r, c in cells)


def _transform_shape(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    pts = list(cells)
    if random.random() < 0.5:
        pts = [(r, -c) for r, c in pts]
    for _ in range(random.randint(0, 3)):
        pts = [(c, -r) for r, c in pts]
    return _normalize(set(pts))


def _random_shape(area: int) -> list[tuple[int, int]]:
    for _ in range(80):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) not in cells:
                        frontier.append((nr, nc))
            random.shuffle(frontier)
            cells.add(frontier[0])
        pts = _transform_shape(_normalize(cells))
        h = max(r for r, _ in pts) + 1
        w = max(c for _, c in pts) + 1
        if h <= 3 and w <= 3:
            return pts
    raise RuntimeError("failed to sample a small connected shape")


def _profile_order(profile: str) -> int:
    return {"both": 0, "right": 1, "bottom": 2, "internal": 3}[profile]


def _fits_profile(cells: list[tuple[int, int]], h: int, w: int, profile: str) -> bool:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    if profile == "internal":
        return min(rows) >= 2 and max(rows) <= h - 3 and min(cols) >= 2 and max(cols) <= w - 3
    if profile == "right":
        return max(cols) == w - 1 and max(rows) < h - 1 and min(rows) >= 1 and min(cols) >= 1
    if profile == "bottom":
        return max(rows) == h - 1 and max(cols) < w - 1 and min(cols) >= 1 and min(rows) >= 1
    if profile == "both":
        return max(rows) == h - 1 and max(cols) == w - 1 and min(rows) >= 1 and min(cols) >= 1
    raise ValueError(profile)


def _can_place(grid: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    h, w = grid.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if grid[r, c] != BG:
            return False
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] != BG and (nr, nc) not in cells:
                return False
    return True


def _place_object(grid: np.ndarray, profile: str, color: int, area: int, *, force_row_one: bool = False, force_col_one: bool = False) -> dict | None:
    h, w = grid.shape
    for _ in range(120):
        shape = _random_shape(area)
        sh = max(r for r, _ in shape) + 1
        sw = max(c for _, c in shape) + 1

        if profile == "internal":
            row_choices = list(range(2, h - sh - 1))
            col_choices = list(range(2, w - sw - 1))
        elif profile == "right":
            row_choices = [1] if force_row_one else list(range(1, h - sh))
            col_choices = [w - sw]
        elif profile == "bottom":
            row_choices = [h - sh]
            col_choices = [1] if force_col_one else list(range(1, w - sw))
        elif profile == "both":
            row_choices = [h - sh]
            col_choices = [w - sw]
        else:
            raise ValueError(profile)

        if not row_choices or not col_choices:
            continue

        r0 = random.choice(row_choices)
        c0 = random.choice(col_choices)
        cells = [(r0 + r, c0 + c) for r, c in shape]
        if not _fits_profile(cells, h, w, profile):
            continue
        if not _can_place(grid, cells):
            continue

        for r, c in cells:
            grid[r, c] = color
        coords = np.array(cells, dtype=int)
        return {
            "color": color,
            "profile": profile,
            "area": area,
            "coords": coords,
            "bbox": _bbox(coords),
        }
    return None


def _choose_profiles(n_objects: int) -> tuple[list[str], bool]:
    conflict = n_objects >= 3 and random.random() < 0.55
    profiles = ["internal"]
    if conflict:
        profiles.extend(["right", "bottom"])
    else:
        profiles.append(random.choices(["right", "bottom", "both"], weights=[3, 3, 2])[0])

    while len(profiles) < n_objects:
        profiles.append(random.choices(["internal", "right", "bottom", "both"], weights=[2, 2, 2, 1])[0])
    random.shuffle(profiles)
    return profiles, conflict


def _place_grey(grid: np.ndarray) -> bool:
    h, w = grid.shape
    choices = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    random.shuffle(choices)
    for r, c in choices:
        if grid[r, c] != BG:
            continue
        if any(0 <= nr < h and 0 <= nc < w and grid[nr, nc] != BG for nr, nc in _neighbors4(r, c)):
            continue
        grid[r, c] = GREY
        return True
    return False


def _validate_semantics(grid: np.ndarray, expect_conflict: bool) -> bool:
    h, w = grid.shape
    if max(h, w) > 30:
        return False
    if np.count_nonzero(grid == GREY) != 1:
        return False

    specs = _extract_objects(grid)
    if not (2 <= len(specs) <= 4):
        return False

    has_internal = False
    has_merged = False
    has_both = False
    for spec in specs:
        if not (2 <= spec["area"] <= 4):
            return False
        if spec["touch_left"] or spec["touch_top"]:
            return False
        if spec["touch_right"] or spec["touch_bottom"]:
            has_merged = True
            has_both = has_both or (spec["touch_right"] and spec["touch_bottom"])
        else:
            r0, c0, r1, c1 = spec["bbox"]
            if min(r0, c0, h - 1 - r1, w - 1 - c1) < 2:
                return False
            has_internal = True
    if not has_internal or not has_merged:
        return False

    out, tiled, claims, _ = _solve(grid)
    if out.shape != (2 * h, 2 * w):
        return False
    if np.any(out[tiled != BG] != tiled[tiled != BG]):
        return False

    changed = (out != tiled)
    if not np.any(changed):
        return False

    internal_colors = {spec["color"] for spec in specs if not spec["touch_right"] and not spec["touch_bottom"]}
    for color in internal_colors:
        if np.count_nonzero(out == color) != 4 * np.count_nonzero(grid == color):
            return False

    if expect_conflict:
        conflict_ok = False
        for contenders in claims.values():
            if len(contenders) < 2:
                continue
            areas = sorted(a for a, _ in contenders)
            if areas[0] < areas[1]:
                conflict_ok = True
                break
        if not conflict_ok:
            return False

    if has_both:
        both_specs = [spec for spec in specs if spec["touch_right"] and spec["touch_bottom"]]
        if not both_specs:
            return False

    added_ratio = np.count_nonzero(changed) / float(out.size)
    if added_ratio > 0.38:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n_objects = random.choices([2, 3, 4], weights=[1, 3, 2])[0]
    profiles, expect_conflict = _choose_profiles(n_objects)
    h = random.randint(7, 9)
    w = random.randint(7, 9)
    grid = np.zeros((h, w), dtype=int)
    colors = random.sample(OBJECT_COLORS, len(profiles))

    metas: list[dict] = []
    profile_indices = list(range(len(profiles)))
    profile_indices.sort(key=lambda idx: _profile_order(profiles[idx]))

    if expect_conflict:
        area_choices = [2, 4]
        random.shuffle(area_choices)
        forced_areas = {"right": area_choices[0], "bottom": area_choices[1]}
    else:
        forced_areas = {}

    right_forced = False
    bottom_forced = False
    for idx in profile_indices:
        profile = profiles[idx]
        color = colors[idx]
        if profile in forced_areas and not (profile == "right" and right_forced) and not (profile == "bottom" and bottom_forced):
            area = forced_areas[profile]
        else:
            area = random.choice([2, 3, 4])

        meta = _place_object(
            grid,
            profile,
            color,
            area,
            force_row_one=expect_conflict and profile == "right" and not right_forced,
            force_col_one=expect_conflict and profile == "bottom" and not bottom_forced,
        )
        if meta is None:
            return None
        metas.append(meta)
        if expect_conflict and profile == "right" and not right_forced:
            right_forced = True
        if expect_conflict and profile == "bottom" and not bottom_forced:
            bottom_forced = True

    _ = metas
    if not _place_grey(grid):
        return None
    if not _validate_semantics(grid, expect_conflict):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
