import numpy as np
from collections import deque


BLACK = 0
GREY = 5
MAROON = 9
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8]
SIDES = ("top", "bottom", "left", "right")

# The template's top-left cell is reserved for the wall-color key.
# The remaining eight cells encode relative offsets around the seed.
NON_LABEL_OFFSETS = [
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 0),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
VISIBLE_GROWTH_OFFSETS = [offset for offset in NON_LABEL_OFFSETS if offset != (0, 0)]
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rs = coords[:, 0]
    cs = coords[:, 1]
    return int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max())


def _extract_templates(grid: np.ndarray) -> dict[int, list[tuple[int, int]]]:
    assert grid.shape[0] >= 3
    band = grid[-3:, :]
    mapping: dict[int, list[tuple[int, int]]] = {}

    c = 0
    while c <= grid.shape[1] - 3:
        anchor = int(band[0, c])
        if anchor not in (BLACK, GREY, MAROON):
            patch = band[:, c : c + 3]
            assert patch.shape == (3, 3), "every template must be a 3x3 patch"
            assert int(patch[0, 0]) == anchor, "template anchor must be its top-left pixel"
            assert np.all(np.isin(patch, [BLACK, MAROON, anchor])), "template patch contains invalid colors"

            offsets = [(int(rr) - 1, int(cc) - 1) for rr, cc in np.argwhere(patch == MAROON)]
            assert offsets, "template must encode at least one maroon cell"
            assert any(offset != (0, 0) for offset in offsets), "template must encode visible growth"
            assert anchor not in mapping, "template colors must be unique"
            mapping[anchor] = offsets
            c += 3
        else:
            c += 1

    assert mapping, "at least one bottom template is required"
    return mapping


def _wall_side_color(sub: np.ndarray) -> int:
    assert sub.shape[0] >= 3 and sub.shape[1] >= 3
    corners = [int(sub[0, 0]), int(sub[0, -1]), int(sub[-1, 0]), int(sub[-1, -1])]
    assert all(color == GREY for color in corners), "rectangle corners must stay grey"

    segments = {
        "top": sub[0, 1:-1],
        "bottom": sub[-1, 1:-1],
        "left": sub[1:-1, 0],
        "right": sub[1:-1, -1],
    }

    colored: list[int] = []
    for segment in segments.values():
        values = set(map(int, np.unique(segment)))
        if values == {GREY}:
            continue
        assert len(values) == 1 and GREY not in values, "exactly one side segment may be recolored"
        colored.append(int(next(iter(values))))

    assert len(colored) == 1, "each wall must have exactly one recolored side"
    return colored[0]


def _find_regions(grid: np.ndarray) -> list[dict[str, int]]:
    wall_area = grid[:-3, :]
    comps = _components(wall_area != BLACK)
    regions: list[dict[str, int]] = []

    for coords in comps:
        r0, c0, r1, c1 = _bbox(coords)
        if r1 - r0 < 4 or c1 - c0 < 4:
            continue

        cell_set = {tuple(map(int, pt)) for pt in coords}
        perimeter: set[tuple[int, int]] = set()
        for cc in range(c0, c1 + 1):
            perimeter.add((r0, cc))
            perimeter.add((r1, cc))
        for rr in range(r0 + 1, r1):
            perimeter.add((rr, c0))
            perimeter.add((rr, c1))

        is_ring = True
        for rr, cc in perimeter:
            if (rr, cc) not in cell_set:
                is_ring = False
                break
        if not is_ring:
            continue

        sub = wall_area[r0 : r1 + 1, c0 : c1 + 1]
        side_color = _wall_side_color(sub)

        interior = wall_area[r0 + 1 : r1, c0 + 1 : c1]
        seed_coords = np.argwhere(interior != BLACK)
        assert seed_coords.shape[0] == 1, "each region must contain exactly one colored seed"
        sr, sc = map(int, seed_coords[0])
        seed_color = int(interior[sr, sc])
        assert seed_color not in (BLACK, GREY, MAROON, side_color), "seed color must stay distinct from wall colors"

        extras = cell_set - perimeter
        if extras not in (set(), {(r0 + 1 + sr, c0 + 1 + sc)}):
            continue

        regions.append(
            {
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "seed_r": r0 + 1 + sr,
                "seed_c": c0 + 1 + sc,
                "seed_color": seed_color,
                "side_color": side_color,
            }
        )

    assert regions, "expected at least one closed rectangle region"
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each interior seed using the bottom template selected by the rectangle wall's colored side.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 8 and g.shape[1] >= 8

    templates = _extract_templates(g)
    regions = _find_regions(g)
    out = g.copy()

    for region in regions:
        offsets = templates.get(region["side_color"])
        assert offsets is not None, "every colored wall must have a matching template"

        for dr, dc in offsets:
            rr = region["seed_r"] + dr
            cc = region["seed_c"] + dc
            if region["r0"] < rr < region["r1"] and region["c0"] < cc < region["c1"]:
                out[rr, cc] = region["seed_color"]

    return out


def _random_pattern(rng: np.random.Generator) -> list[tuple[int, int]]:
    noncenter = [offset for offset in NON_LABEL_OFFSETS if offset != (0, 0)]
    count = int(rng.integers(1, len(noncenter) + 1))
    chosen_idx = rng.choice(len(noncenter), size=count, replace=False)
    pattern = [noncenter[int(idx)] for idx in chosen_idx]
    if rng.random() < 0.35:
        pattern.append((0, 0))
    return pattern


def _seed_candidates(inner_h: int, inner_w: int, offsets: list[tuple[int, int]], require_clip: bool) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for sr in range(inner_h):
        for sc in range(inner_w):
            valid_growth = 0
            all_valid = True
            any_invalid = False
            for dr, dc in offsets:
                rr = sr + dr
                cc = sc + dc
                inside = 0 <= rr < inner_h and 0 <= cc < inner_w
                if not inside:
                    any_invalid = True
                    all_valid = False
                elif (dr, dc) != (0, 0):
                    valid_growth += 1
            if valid_growth == 0:
                continue
            if require_clip and any_invalid:
                candidates.append((sr, sc))
            if not require_clip and all_valid:
                candidates.append((sr, sc))
    return candidates


def _choose_seed(inner_h: int, inner_w: int, offsets: list[tuple[int, int]], clip_mode: bool, rng: np.random.Generator) -> tuple[tuple[int, int], bool] | None:
    primary = _seed_candidates(inner_h, inner_w, offsets, require_clip=clip_mode)
    if primary:
        choice = primary[int(rng.integers(0, len(primary)))]
        return choice, clip_mode

    fallback = _seed_candidates(inner_h, inner_w, offsets, require_clip=not clip_mode)
    if fallback:
        choice = fallback[int(rng.integers(0, len(fallback)))]
        return choice, not clip_mode

    return None


def _draw_template(grid: np.ndarray, color: int, offsets: list[tuple[int, int]], row0: int, col0: int) -> None:
    patch = np.zeros((3, 3), dtype=int)
    patch[0, 0] = color
    for dr, dc in offsets:
        patch[dr + 1, dc + 1] = MAROON
    grid[row0 : row0 + 3, col0 : col0 + 3] = patch


def _draw_region(
    grid: np.ndarray,
    r0: int,
    c0: int,
    inner_h: int,
    inner_w: int,
    side: str,
    wall_color: int,
    seed: tuple[int, int],
    seed_color: int,
) -> None:
    r1 = r0 + inner_h + 1
    c1 = c0 + inner_w + 1

    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY

    if side == "top":
        grid[r0, c0 + 1 : c1] = wall_color
    elif side == "bottom":
        grid[r1, c0 + 1 : c1] = wall_color
    elif side == "left":
        grid[r0 + 1 : r1, c0] = wall_color
    else:
        grid[r0 + 1 : r1, c1] = wall_color

    sr, sc = seed
    grid[r0 + 1 + sr, c0 + 1 + sc] = seed_color


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    region_count = int(rng.integers(1, 4))
    extra_template = 1 if region_count <= 2 and rng.random() < 0.45 else 0
    template_count = region_count + extra_template

    height = int(rng.integers(16, 27))
    width = int(rng.integers(max(18, 5 * template_count + 4), 31))
    if width - 3 * template_count < template_count + 1:
        return None

    grid = np.zeros((height, width), dtype=int)
    template_row = height - 3
    if template_row < 8:
        return None

    template_colors = list(map(int, rng.choice(COLOR_POOL, size=template_count, replace=False)))
    patterns = {color: _random_pattern(rng) for color in template_colors}

    order = template_colors[:]
    rng.shuffle(order)
    gap_total = width - 3 * template_count
    gaps = [1] * (template_count + 1)
    for _ in range(gap_total - len(gaps)):
        gaps[int(rng.integers(0, len(gaps)))] += 1

    col = gaps[0]
    for idx, color in enumerate(order):
        _draw_template(grid, color, patterns[color], template_row, col)
        col += 3 + gaps[idx + 1]

    used_wall_colors = template_colors[:]
    rng.shuffle(used_wall_colors)
    used_wall_colors = used_wall_colors[:region_count]

    occ = np.zeros((height, width), dtype=bool)
    occ[template_row - 1 :, :] = True
    clip_modes: list[bool] = []

    for wall_color in used_wall_colors:
        offsets = patterns[wall_color]
        placed = False

        for _ in range(250):
            inner_h = int(rng.integers(3, 8))
            inner_w = int(rng.integers(3, 10))
            side = SIDES[int(rng.integers(0, len(SIDES)))]
            want_clip = bool(rng.integers(0, 2))
            seed_choice = _choose_seed(inner_h, inner_w, offsets, want_clip, rng)
            if seed_choice is None:
                continue

            (seed_r, seed_c), actual_clip = seed_choice
            outer_h = inner_h + 2
            outer_w = inner_w + 2
            max_r0 = template_row - outer_h - 2
            max_c0 = width - outer_w - 1
            if max_r0 < 1 or max_c0 < 1:
                continue

            r0 = int(rng.integers(1, max_r0 + 1))
            c0 = int(rng.integers(1, max_c0 + 1))
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1

            pad_r0 = max(0, r0 - 1)
            pad_c0 = max(0, c0 - 1)
            pad_r1 = min(height, r1 + 2)
            pad_c1 = min(width, c1 + 2)
            if np.any(occ[pad_r0:pad_r1, pad_c0:pad_c1]):
                continue

            seed_choices = [color for color in COLOR_POOL if color != wall_color]
            seed_color = int(rng.choice(seed_choices))
            _draw_region(grid, r0, c0, inner_h, inner_w, side, wall_color, (seed_r, seed_c), seed_color)
            occ[r0 : r1 + 1, c0 : c1 + 1] = True
            clip_modes.append(actual_clip)
            placed = True
            break

        if not placed:
            return None

    if not any(clip_modes) and region_count > 1:
        return None
    if not any(not clip for clip in clip_modes):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    """
    Randomly generate a valid puzzle instance with multiple rectangle/template pairings.
    """
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
