import random
from collections import Counter, deque

import numpy as np


GEN_RETRIES = 300
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
ALL_COLORS = tuple(range(10))


def _mode_color(arr: np.ndarray) -> int:
    values, counts = np.unique(arr, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_rectangular_ring(coords: list[tuple[int, int]]) -> bool:
    if len(coords) < 8:
        return False
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    expected = 2 * (h + w) - 4
    if len(coords) != expected:
        return False
    pts = set(coords)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            if on_border != ((r, c) in pts):
                return False
    return True


def _find_walls(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    wall_comps: list[dict] = []
    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        for comp in _components(g == color):
            if not _is_rectangular_ring(comp):
                continue
            r0, c0, r1, c1 = _bbox(comp)
            inner = g[r0 + 1 : r1, c0 + 1 : c1]
            seed_coords = [(r0 + 1 + rr, c0 + 1 + cc) for rr, cc in np.argwhere(inner != bg)]
            assert len(seed_coords) == 1, "each wall interior must contain exactly one seed"
            sr, sc = seed_coords[0]
            seed_color = int(g[sr, sc])
            assert seed_color != color, "seed color must differ from wall color"
            wall_comps.append(
                {
                    "wall_color": color,
                    "bbox": (r0, c0, r1, c1),
                    "coords": comp,
                    "seed": (sr, sc, seed_color),
                }
            )
    assert wall_comps, "expected at least one rectangular wall"
    return wall_comps


def _legend_mapping(grid: np.ndarray, bg: int, walls: list[dict]) -> dict[int, list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    legend_mask = g != bg
    for wall in walls:
        for r, c in wall["coords"]:
            legend_mask[r, c] = False
        sr, sc, _ = wall["seed"]
        legend_mask[sr, sc] = False

    legend_coords = np.argwhere(legend_mask)
    assert len(legend_coords) > 0, "expected legend pixels outside walls"

    legend_colors = [int(g[r, c]) for r, c in legend_coords]
    maroon = Counter(legend_colors).most_common(1)[0][0]

    rows_with_legend = sorted({int(r) for r, _ in legend_coords})
    bands: list[tuple[int, int]] = []
    start = rows_with_legend[0]
    prev = start
    for row in rows_with_legend[1:]:
        if row == prev + 1:
            prev = row
            continue
        bands.append((start, prev))
        start = prev = row
    bands.append((start, prev))

    mapping: dict[int, list[tuple[int, int]]] = {}
    for r0, r1 in bands:
        band = g[r0 : r1 + 1]
        maroon_pts = [(r0 + rr, cc) for rr, cc in np.argwhere(band == maroon)]
        assert maroon_pts, "each legend band must contain template pixels"
        swatches = [(r0 + rr, cc) for rr, cc in np.argwhere((band != bg) & (band != maroon))]
        assert len(swatches) == 1, "each legend band must contain exactly one color swatch"

        sr, sc = swatches[0]
        seed_color = int(g[sr, sc])
        tr0, tc0, tr1, tc1 = _bbox(maroon_pts)
        assert (tr1 - tr0 + 1) % 2 == 1 and (tc1 - tc0 + 1) % 2 == 1
        center_r = (tr0 + tr1) // 2
        center_c = (tc0 + tc1) // 2
        assert tr0 <= sr <= tr1, "swatch row should align with the template band"
        assert sc < tc0, "swatch must sit to the left of its template"

        offsets = sorted((r - center_r, c - center_c) for r, c in maroon_pts)
        assert seed_color not in mapping, "legend seed colors must be unique"
        mapping[seed_color] = offsets

    assert mapping, "legend mapping must be non-empty"
    return mapping


def _fold_index(idx: int, lo: int, hi: int) -> int:
    if lo == hi:
        return lo
    span = hi - lo
    period = 2 * span
    x = (idx - lo) % period
    if x > span:
        x = period - x
    return lo + x


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each seed according to its legend template, reflecting overflow back
    into the rectangular walled interior.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = _mode_color(g)
    walls = _find_walls(g, bg)
    mapping = _legend_mapping(g, bg, walls)

    out = g.copy()
    for wall in walls:
        r0, c0, r1, c1 = wall["bbox"]
        lo_r, hi_r = r0 + 1, r1 - 1
        lo_c, hi_c = c0 + 1, c1 - 1
        sr, sc, seed_color = wall["seed"]
        assert seed_color in mapping, "every seed color must appear in the legend"
        for dr, dc in mapping[seed_color]:
            rr = _fold_index(sr + dr, lo_r, hi_r)
            cc = _fold_index(sc + dc, lo_c, hi_c)
            out[rr, cc] = seed_color

    return out


def _template_offsets(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cr, cc = h // 2, w // 2
    return [(r - cr, c - cc) for r, c in np.argwhere(mask)]


def _make_template(rng: random.Random) -> np.ndarray:
    for _ in range(200):
        h = rng.choice((3, 5))
        w = rng.choice((3, 5))
        cr, cc = h // 2, w // 2
        mask = np.zeros((h, w), dtype=bool)
        mask[cr, cc] = True

        anchors = [
            (0, rng.randrange(w)),
            (h - 1, rng.randrange(w)),
            (rng.randrange(h), 0),
            (rng.randrange(h), w - 1),
        ]
        for ar, ac in anchors:
            r, c = cr, cc
            while r != ar:
                r += 1 if ar > r else -1
                mask[r, c] = True
            while c != ac:
                c += 1 if ac > c else -1
                mask[r, c] = True

        extra_steps = rng.randint(1, max(2, h + w - 2))
        frontier = [(r, c) for r, c in np.argwhere(mask)]
        for _ in range(extra_steps):
            br, bc = frontier[rng.randrange(len(frontier))]
            options = [(br + dr, bc + dc) for dr, dc in DIRS4 if 0 <= br + dr < h and 0 <= bc + dc < w]
            nr, nc = options[rng.randrange(len(options))]
            mask[nr, nc] = True
            frontier = [(r, c) for r, c in np.argwhere(mask)]

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows[0] or not rows[-1] or not cols[0] or not cols[-1]:
            continue
        if _is_rectangular_ring([(int(r), int(c)) for r, c in np.argwhere(mask)]):
            continue
        if not mask[cr, cc]:
            continue
        if mask.sum() < 5:
            continue
        return mask
    raise RuntimeError("failed to build template")


def _rand_inclusive(rng: random.Random, lo: int, hi: int) -> int | None:
    if lo > hi:
        return None
    return rng.randint(lo, hi)


def _choose_seed_position(
    rng: random.Random,
    offsets: list[tuple[int, int]],
    inner_h: int,
    inner_w: int,
    mode: str,
) -> tuple[int, int]:
    min_dr = min(dr for dr, _ in offsets)
    max_dr = max(dr for dr, _ in offsets)
    min_dc = min(dc for _, dc in offsets)
    max_dc = max(dc for _, dc in offsets)

    safe_r_lo = max(0, -min_dr)
    safe_r_hi = min(inner_h - 1, inner_h - 1 - max_dr)
    safe_c_lo = max(0, -min_dc)
    safe_c_hi = min(inner_w - 1, inner_w - 1 - max_dc)

    if mode == "safe":
        sr = _rand_inclusive(rng, safe_r_lo, safe_r_hi)
        sc = _rand_inclusive(rng, safe_c_lo, safe_c_hi)
        if sr is None or sc is None:
            return rng.randrange(inner_h), rng.randrange(inner_w)
        return sr, sc

    top_hi = min(inner_h - 1, -min_dr - 1)
    bottom_lo = max(0, inner_h - max_dr)
    left_hi = min(inner_w - 1, -min_dc - 1)
    right_lo = max(0, inner_w - max_dc)

    vertical_choices = []
    if top_hi >= 0:
        vertical_choices.append(("top", 0, top_hi))
    if bottom_lo <= inner_h - 1:
        vertical_choices.append(("bottom", bottom_lo, inner_h - 1))
    horizontal_choices = []
    if left_hi >= 0:
        horizontal_choices.append(("left", 0, left_hi))
    if right_lo <= inner_w - 1:
        horizontal_choices.append(("right", right_lo, inner_w - 1))

    choose_vertical = rng.random() < 0.75 and vertical_choices
    choose_horizontal = rng.random() < 0.75 and horizontal_choices
    if not choose_vertical and not choose_horizontal:
        if vertical_choices and horizontal_choices:
            if rng.random() < 0.5:
                choose_vertical = True
            else:
                choose_horizontal = True
        elif vertical_choices:
            choose_vertical = True
        elif horizontal_choices:
            choose_horizontal = True

    if choose_vertical:
        _name, r_lo, r_hi = rng.choice(vertical_choices)
        sr = rng.randint(r_lo, r_hi)
    else:
        sr = _rand_inclusive(rng, safe_r_lo, safe_r_hi)
        if sr is None:
            sr = rng.randrange(inner_h)

    if choose_horizontal:
        _name, c_lo, c_hi = rng.choice(horizontal_choices)
        sc = rng.randint(c_lo, c_hi)
    else:
        sc = _rand_inclusive(rng, safe_c_lo, safe_c_hi)
        if sc is None:
            sc = rng.randrange(inner_w)

    return sr, sc


def _room_overflows(offsets: list[tuple[int, int]], sr: int, sc: int, inner_h: int, inner_w: int) -> bool:
    for dr, dc in offsets:
        rr = sr + dr
        cc = sc + dc
        if rr < 0 or rr >= inner_h or cc < 0 or cc >= inner_w:
            return True
    return False


def _rooms_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0)


def _place_rooms(
    rng: random.Random,
    play_h: int,
    play_w: int,
    room_sizes: list[tuple[int, int]],
) -> list[tuple[int, int, int, int]] | None:
    placed: list[tuple[int, int, int, int]] = []
    for outer_h, outer_w in room_sizes:
        margin_r = 1 if play_h - outer_h >= 3 else 0
        margin_c = 1 if play_w - outer_w >= 3 else 0
        if play_h - outer_h < 0 or play_w - outer_w < 0:
            return None
        options = []
        for r0 in range(margin_r, play_h - outer_h - margin_r + 1):
            for c0 in range(margin_c, play_w - outer_w - margin_c + 1):
                rect = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
                if all(not _rooms_overlap(rect, prev) for prev in placed):
                    options.append(rect)
        if not options:
            return None
        placed.append(options[rng.randrange(len(options))])
    return placed


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_templates = rng.randint(2, 4)
    orientation = rng.choice(("bottom", "right"))

    role_colors = rng.sample(ALL_COLORS, n_templates + 3)
    bg = role_colors[0]
    wall_color = role_colors[1]
    template_color = role_colors[2]
    seed_colors = role_colors[3:]

    entries = []
    for seed_color in seed_colors:
        template = _make_template(rng)
        entries.append({"seed_color": seed_color, "template": template, "offsets": _template_offsets(template)})

    n_rooms = rng.randint(max(2, n_templates), n_templates + 2)
    room_entries = entries.copy()
    while len(room_entries) < n_rooms:
        room_entries.append(rng.choice(entries))
    rng.shuffle(room_entries)

    modes = ["overflow", "safe"]
    while len(modes) < n_rooms:
        modes.append(rng.choice(("overflow", "safe")))
    rng.shuffle(modes)

    room_specs = []
    for entry, mode in zip(room_entries, modes):
        tpl_h, tpl_w = entry["template"].shape
        if mode == "safe":
            inner_h = rng.randint(max(5, tpl_h), max(6, tpl_h + 2))
            inner_w = rng.randint(max(5, tpl_w), max(6, tpl_w + 2))
        else:
            inner_h = rng.randint(3, max(5, tpl_h + 1))
            inner_w = rng.randint(3, max(5, tpl_w + 1))
        room_specs.append(
            {
                "entry": entry,
                "mode": mode,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "outer_h": inner_h + 2,
                "outer_w": inner_w + 2,
            }
        )

    legend_rows = []
    for entry in entries:
        h = entry["template"].shape[0]
        legend_rows.append(h)
    legend_h = sum(legend_rows) + (len(legend_rows) - 1)
    legend_w = 1 + max(entry["template"].shape[1] for entry in entries)

    area_need = sum(spec["outer_h"] * spec["outer_w"] for spec in room_specs)
    play_h = max(12, rng.randint(12, 18))
    play_w = max(12, rng.randint(14, 22))
    while play_h * play_w < area_need * 3:
        if play_h <= play_w:
            play_h += 1
        else:
            play_w += 1
        if play_h > 22 or play_w > 28:
            break

    if orientation == "bottom":
        gap = rng.randint(1, 2)
        bottom_pad = rng.randint(0, 1)
        total_h = play_h + gap + legend_h + bottom_pad
        total_w = play_w + rng.randint(0, 3)
        if total_w < legend_w + 2:
            total_w = legend_w + 2
        legend_top = play_h + gap
        legend_left = rng.randint(0, total_w - legend_w)
    else:
        gap = rng.randint(1, 2)
        right_pad = rng.randint(0, 1)
        total_h = max(play_h + rng.randint(0, 2), legend_h + rng.randint(0, 3))
        total_w = play_w + gap + legend_w + right_pad
        legend_top = rng.randint(0, total_h - legend_h)
        legend_left = play_w + gap

    if total_h > 30 or total_w > 30:
        return None

    if orientation == "bottom":
        room_rects = _place_rooms(rng, play_h, total_w, [(s["outer_h"], s["outer_w"]) for s in room_specs])
    else:
        room_rects = _place_rooms(rng, total_h, play_w, [(s["outer_h"], s["outer_w"]) for s in room_specs])
    if room_rects is None:
        return None

    grid = np.full((total_h, total_w), bg, dtype=int)
    overflow_count = 0
    safe_count = 0

    for spec, (r0, c0, r1, c1) in zip(room_specs, room_rects):
        grid[r0 : r1 + 1, c0] = wall_color
        grid[r0 : r1 + 1, c1] = wall_color
        grid[r0, c0 : c1 + 1] = wall_color
        grid[r1, c0 : c1 + 1] = wall_color

        sr_local, sc_local = _choose_seed_position(
            rng,
            spec["entry"]["offsets"],
            spec["inner_h"],
            spec["inner_w"],
            spec["mode"],
        )
        sr = r0 + 1 + sr_local
        sc = c0 + 1 + sc_local
        grid[sr, sc] = spec["entry"]["seed_color"]

        if _room_overflows(spec["entry"]["offsets"], sr_local, sc_local, spec["inner_h"], spec["inner_w"]):
            overflow_count += 1
        else:
            safe_count += 1

    ordered_entries = entries[:]
    rng.shuffle(ordered_entries)
    cursor = legend_top
    for entry in ordered_entries:
        template = entry["template"]
        h, w = template.shape
        grid[cursor + h // 2, legend_left] = entry["seed_color"]
        for rr, cc in np.argwhere(template):
            grid[cursor + rr, legend_left + 1 + cc] = template_color
        cursor += h + 1

    if overflow_count == 0 or safe_count == 0:
        return None
    if _mode_color(grid) != bg:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
