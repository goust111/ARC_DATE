from __future__ import annotations

import random

import numpy as np


MAX_GENERATION_ATTEMPTS = 400
SCENARIOS = ("nested", "overlap", "mixed")


def _get_bg(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _box_h(box: tuple[int, int, int, int]) -> int:
    return int(box[2] - box[0] + 1)


def _box_w(box: tuple[int, int, int, int]) -> int:
    return int(box[3] - box[1] + 1)


def _expand_box(
    box: tuple[int, int, int, int],
    top: int,
    left: int,
    bottom: int,
    right: int,
) -> tuple[int, int, int, int]:
    return (
        int(box[0] - top),
        int(box[1] - left),
        int(box[2] + bottom),
        int(box[3] + right),
    )


def _shift_box(box: tuple[int, int, int, int], dr: int, dc: int) -> tuple[int, int, int, int]:
    return (
        int(box[0] + dr),
        int(box[1] + dc),
        int(box[2] + dr),
        int(box[3] + dc),
    )


def _boxes_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _box_contains(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return bool(a[0] <= b[0] and a[1] <= b[1] and a[2] >= b[2] and a[3] >= b[3])


def _intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    r0 = max(int(a[0]), int(b[0]))
    c0 = max(int(a[1]), int(b[1]))
    r1 = min(int(a[2]), int(b[2]))
    c1 = min(int(a[3]), int(b[3]))
    if r0 > r1 or c0 > c1:
        return 0
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _ring_mask(height: int, width: int, thickness: int) -> np.ndarray:
    mask = np.ones((height, width), dtype=bool)
    mask[thickness : height - thickness, thickness : width - thickness] = False
    return mask


def _make_spec(
    color: int,
    fill: int,
    thickness: int,
    inner: tuple[int, int, int, int],
) -> dict:
    outer = _expand_box(inner, thickness, thickness, thickness, thickness)
    return {
        "color": int(color),
        "fill": int(fill),
        "t": int(thickness),
        "inner": tuple(map(int, inner)),
        "outer": tuple(map(int, outer)),
    }


def _shift_spec(spec: dict, dr: int, dc: int) -> dict:
    return {
        "color": int(spec["color"]),
        "fill": int(spec["fill"]),
        "t": int(spec["t"]),
        "inner": _shift_box(spec["inner"], dr, dc),
        "outer": _shift_box(spec["outer"], dr, dc),
    }


def _ring_cells(spec: dict) -> set[tuple[int, int]]:
    outer = spec["outer"]
    inner = spec["inner"]
    cells: set[tuple[int, int]] = set()
    for r in range(int(outer[0]), int(outer[2]) + 1):
        for c in range(int(outer[1]), int(outer[3]) + 1):
            if int(inner[0]) <= r <= int(inner[2]) and int(inner[1]) <= c <= int(inner[3]):
                continue
            cells.add((int(r), int(c)))
    return cells


def _rings_disjoint(specs: list[dict]) -> bool:
    occupied: set[tuple[int, int]] = set()
    for spec in specs:
        ring = _ring_cells(spec)
        if occupied & ring:
            return False
        occupied |= ring
    return True


def _cover_count(specs: list[dict], height: int, width: int) -> np.ndarray:
    cover = np.zeros((height, width), dtype=int)
    for spec in specs:
        r0, c0, r1, c1 = spec["inner"]
        cover[r0 : r1 + 1, c0 : c1 + 1] += 1
    return cover


def _render_from_specs(grid: np.ndarray, specs: list[dict], bg: int) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    main = out[:-1].copy()
    border_mask = np.asarray(grid[:-1] != bg, dtype=bool)
    for spec in sorted(specs, key=lambda s: (-int(s["t"]), int(s["outer"][0]), int(s["outer"][1]))):
        r0, c0, r1, c1 = spec["inner"]
        window = main[r0 : r1 + 1, c0 : c1 + 1]
        window[~border_mask[r0 : r1 + 1, c0 : c1 + 1]] = int(spec["fill"])
    out[:-1] = main
    return out


def _parse_palette(grid: np.ndarray, bg: int) -> dict[int, int]:
    row = np.asarray(grid[-1], dtype=int)
    idxs = [int(c) for c in range(row.shape[0]) if int(row[c]) != bg]
    assert len(idxs) >= 2 and len(idxs) % 2 == 0, "palette row must contain color pairs"
    mapping: dict[int, int] = {}
    for i in range(0, len(idxs), 2):
        c0 = idxs[i]
        c1 = idxs[i + 1]
        assert c1 == c0 + 1, "palette pairs must be adjacent"
        key = int(row[c0])
        val = int(row[c1])
        mapping[key] = val
    return mapping


def _detect_rectangles(main_area: np.ndarray, mapping: dict[int, int]) -> list[dict]:
    rectangles: list[dict] = []
    for color, fill in mapping.items():
        pts = np.argwhere(main_area == int(color))
        assert pts.size != 0, f"missing rectangle for key color {color}"
        r0, c0 = map(int, pts.min(axis=0))
        r1, c1 = map(int, pts.max(axis=0))
        sub = main_area[r0 : r1 + 1, c0 : c1 + 1]
        mask = sub == int(color)
        found_t = None
        for thickness in (1, 2, 3):
            if sub.shape[0] <= 2 * thickness or sub.shape[1] <= 2 * thickness:
                continue
            if np.array_equal(mask, _ring_mask(sub.shape[0], sub.shape[1], thickness)):
                found_t = int(thickness)
                break
        assert found_t is not None, f"color {color} does not form a supported hollow rectangle"
        rectangles.append(
            {
                "color": int(color),
                "fill": int(fill),
                "t": int(found_t),
                "inner": (int(r0 + found_t), int(c0 + found_t), int(r1 - found_t), int(c1 - found_t)),
                "outer": (int(r0), int(c0), int(r1), int(c1)),
            }
        )
    return rectangles


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill hollow rectangles using the bottom-row palette, painting thicker borders first.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2

    bg = _get_bg(g)
    mapping = _parse_palette(g, bg)

    main_area = g[:-1].copy()
    border_mask = np.asarray(main_area != bg, dtype=bool)
    rectangles = _detect_rectangles(main_area, mapping)

    rectangles.sort(key=lambda rect: (-int(rect["t"]), int(rect["outer"][0]), int(rect["outer"][1])))
    for rect in rectangles:
        r0, c0, r1, c1 = rect["inner"]
        region = main_area[r0 : r1 + 1, c0 : c1 + 1]
        region[~border_mask[r0 : r1 + 1, c0 : c1 + 1]] = int(rect["fill"])

    out = g.copy()
    out[:-1] = main_area
    return out


def _shift_specs_into_grid(rng: random.Random, specs: list[dict], height: int, width: int) -> list[dict] | None:
    min_r = min(int(spec["outer"][0]) for spec in specs)
    min_c = min(int(spec["outer"][1]) for spec in specs)
    max_r = max(int(spec["outer"][2]) for spec in specs)
    max_c = max(int(spec["outer"][3]) for spec in specs)

    max_top = int(height - 3 - max_r)
    max_left = int(width - 1 - max_c)
    min_top = int(-min_r)
    min_left = int(-min_c)
    if min_top > max_top or min_left > max_left:
        return None

    dr = rng.randint(min_top, max_top)
    dc = rng.randint(min_left, max_left)
    shifted = [_shift_spec(spec, dr, dc) for spec in specs]
    if any(int(spec["outer"][2]) > height - 3 for spec in shifted):
        return None
    return shifted


def _gen_nested_core(
    rng: random.Random,
    height: int,
    width: int,
    colors: list[int],
    fills: list[int],
) -> list[dict] | None:
    for _ in range(250):
        level_ts = rng.sample([1, 2, 3], 3)
        if level_ts == [1, 2, 3] or level_ts == [3, 2, 1]:
            continue

        inner_h = rng.randint(2, 4)
        inner_w = rng.randint(2, 5)
        inner_box = (0, 0, inner_h - 1, inner_w - 1)
        inner_spec = _make_spec(colors[2], fills[2], level_ts[2], inner_box)

        mid_gaps = [rng.randint(1, 4) for _ in range(4)]
        mid_box = _expand_box(inner_spec["outer"], mid_gaps[0], mid_gaps[1], mid_gaps[2], mid_gaps[3])
        mid_spec = _make_spec(colors[1], fills[1], level_ts[1], mid_box)

        outer_gaps = [rng.randint(1, 4) for _ in range(4)]
        outer_box = _expand_box(mid_spec["outer"], outer_gaps[0], outer_gaps[1], outer_gaps[2], outer_gaps[3])
        outer_spec = _make_spec(colors[0], fills[0], level_ts[0], outer_box)

        specs = [outer_spec, mid_spec, inner_spec]
        shifted = _shift_specs_into_grid(rng, specs, height, width)
        if shifted is None:
            continue
        return shifted
    return None


def _gen_overlap_core(
    rng: random.Random,
    height: int,
    width: int,
    colors: list[int],
    fills: list[int],
) -> list[dict] | None:
    for _ in range(700):
        shared_h = rng.randint(2, 4)
        shared_w = rng.randint(2, 5)
        specs: list[dict] = []
        for color, fill, thickness in zip(colors, fills, [1, 2, 3]):
            up = rng.randint(0, 6)
            left = rng.randint(0, 7)
            down = rng.randint(0, 6)
            right = rng.randint(0, 7)
            inner = (-up, -left, shared_h - 1 + down, shared_w - 1 + right)
            specs.append(_make_spec(color, fill, thickness, inner))

        pairwise_contains = False
        for i in range(3):
            for j in range(i + 1, 3):
                if _box_contains(specs[i]["inner"], specs[j]["inner"]) or _box_contains(specs[j]["inner"], specs[i]["inner"]):
                    pairwise_contains = True
                    break
            if pairwise_contains:
                break
        if pairwise_contains or not _rings_disjoint(specs):
            continue

        shifted = _shift_specs_into_grid(rng, specs, height, width)
        if shifted is None:
            continue

        cover = _cover_count(shifted, height - 1, width)
        if int(cover.max()) < 2:
            continue
        return shifted
    return None


def _gen_mixed_core(
    rng: random.Random,
    height: int,
    width: int,
    colors: list[int],
    fills: list[int],
) -> list[dict] | None:
    for _ in range(700):
        role_ts = rng.sample([1, 2, 3], 3)
        shared_h = rng.randint(2, 3)
        shared_w = rng.randint(2, 4)
        shared = (0, 0, shared_h - 1, shared_w - 1)

        inner_gaps = [rng.randint(1, 3) for _ in range(4)]
        mid_box = _expand_box(shared, inner_gaps[0], inner_gaps[1], inner_gaps[2], inner_gaps[3])
        mid_spec = _make_spec(colors[1], fills[1], role_ts[1], mid_box)

        outer_gaps = [rng.randint(1, 4) for _ in range(4)]
        outer_box = _expand_box(mid_spec["outer"], outer_gaps[0], outer_gaps[1], outer_gaps[2], outer_gaps[3])
        outer_spec = _make_spec(colors[0], fills[0], role_ts[0], outer_box)

        third_gaps = [rng.randint(0, 6) for _ in range(4)]
        third_box = _expand_box(shared, third_gaps[0], third_gaps[1], third_gaps[2], third_gaps[3])
        third_spec = _make_spec(colors[2], fills[2], role_ts[2], third_box)

        if not _boxes_intersect(mid_spec["inner"], third_spec["inner"]):
            continue
        if _box_contains(mid_spec["inner"], third_spec["inner"]) or _box_contains(third_spec["inner"], mid_spec["inner"]):
            continue
        if not _rings_disjoint([outer_spec, mid_spec, third_spec]):
            continue

        shifted = _shift_specs_into_grid(rng, [outer_spec, mid_spec, third_spec], height, width)
        if shifted is None:
            continue

        contain_pairs = 0
        for i in range(3):
            for j in range(3):
                if i != j and _box_contains(shifted[i]["inner"], shifted[j]["inner"]):
                    contain_pairs += 1
        if contain_pairs == 0:
            continue
        cover = _cover_count(shifted, height - 1, width)
        if int(cover.max()) < 2:
            continue
        return shifted
    return None


def _place_extra_rect(
    rng: random.Random,
    height: int,
    width: int,
    color: int,
    fill: int,
    existing: list[dict],
) -> dict | None:
    expanded_existing = [_expand_box(spec["outer"], 1, 1, 1, 1) for spec in existing]
    for _ in range(300):
        thickness = rng.choice([1, 2, 3])
        inner_h = rng.randint(2, 7)
        inner_w = rng.randint(2, 8)
        outer_h = inner_h + 2 * thickness
        outer_w = inner_w + 2 * thickness
        if outer_h > height - 2 or outer_w > width:
            continue

        outer_r0 = rng.randint(0, height - 2 - outer_h)
        outer_c0 = rng.randint(0, width - outer_w)
        inner = (
            int(outer_r0 + thickness),
            int(outer_c0 + thickness),
            int(outer_r0 + thickness + inner_h - 1),
            int(outer_c0 + thickness + inner_w - 1),
        )
        spec = _make_spec(color, fill, thickness, inner)

        if any(_boxes_intersect(_expand_box(spec["outer"], 1, 1, 1, 1), other) for other in expanded_existing):
            continue
        if any(_boxes_intersect(spec["inner"], other["inner"]) for other in existing):
            continue
        if not _rings_disjoint(existing + [spec]):
            continue
        return spec
    return None


def _random_partition(rng: random.Random, total: int, parts: int) -> list[int]:
    if parts == 1:
        return [int(total)]
    cuts = sorted(rng.sample(range(total + parts - 1), parts - 1))
    vals: list[int] = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        vals.append(int(cut - prev - 1))
        prev = cut
    return vals


def _place_palette_row(rng: random.Random, width: int, specs: list[dict], bg: int) -> np.ndarray:
    palette = np.full(width, int(bg), dtype=int)
    entries = [(int(spec["color"]), int(spec["fill"])) for spec in specs]
    rng.shuffle(entries)
    base = 2 * len(entries) + max(0, len(entries) - 1)
    extra = int(width - base)
    assert extra >= 0
    extras = _random_partition(rng, extra, len(entries) + 1)
    left = extras[0]
    right = extras[-1]
    gaps = [1 + extras[i + 1] for i in range(len(entries) - 1)]

    c = left
    for i, (key, fill) in enumerate(entries):
        palette[c] = int(key)
        palette[c + 1] = int(fill)
        c += 2
        if i < len(gaps):
            c += int(gaps[i])
    assert c + right == width
    return palette


def _build_grid(height: int, width: int, specs: list[dict], bg: int, rng: random.Random) -> np.ndarray:
    grid = np.full((height, width), int(bg), dtype=int)
    for spec in specs:
        for r, c in _ring_cells(spec):
            grid[r, c] = int(spec["color"])
    grid[-1] = _place_palette_row(rng, width, specs, bg)
    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    rect_count = 3 if rng.random() < 0.55 else 4
    scenario = SCENARIOS[int(rng.random() * len(SCENARIOS))]

    bg = rng.randint(0, 9)
    available = [c for c in range(10) if c != bg]
    rng.shuffle(available)
    border_colors = available[:rect_count]
    fill_pool = [c for c in available if c not in border_colors]
    if len(fill_pool) < rect_count:
        return None
    fill_colors = fill_pool[:rect_count]

    min_width = max(18, 2 * rect_count + max(0, rect_count - 1) + 2)
    height = rng.randint(18, 28)
    width = rng.randint(min_width, 30)

    core_colors = border_colors[:3]
    core_fills = fill_colors[:3]
    if scenario == "nested":
        core_specs = _gen_nested_core(rng, height, width, core_colors, core_fills)
    elif scenario == "overlap":
        core_specs = _gen_overlap_core(rng, height, width, core_colors, core_fills)
    else:
        core_specs = _gen_mixed_core(rng, height, width, core_colors, core_fills)
    if core_specs is None:
        return None

    specs = list(core_specs)
    if rect_count == 4:
        extra = _place_extra_rect(rng, height, width, border_colors[3], fill_colors[3], specs)
        if extra is None:
            return None
        specs.append(extra)

    grid = _build_grid(height, width, specs, bg, rng)

    counts = np.bincount(grid.ravel(), minlength=10)
    if int(np.argmax(counts)) != bg:
        return None

    core_cover = _cover_count(core_specs, height - 1, width)
    if int(core_cover.max()) < 2:
        return None

    expected = _render_from_specs(grid, specs, bg)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        rng = random.Random()
        result = _gen_attempt(rng)
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid input")
