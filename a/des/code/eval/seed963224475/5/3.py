from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

GEN_RETRIES = 400
PLACE_RETRIES = 250
MAIN_H_RANGE = (6, 13)
LEGEND_COUNT_RANGE = (2, 4)
LEGEND_GAP_RANGE = (1, 3)
LEGEND_MARGIN_RANGE = (0, 3)
EXTRA_WIDTH_RANGE = (0, 5)


def _parse_legend(legend: np.ndarray, sep_color: int) -> list[dict[str, object]]:
    colors = [int(v) for v in np.unique(legend) if int(v) not in (BLACK, sep_color)]
    entries: list[dict[str, object]] = []
    for color in colors:
        rows, cols = np.where(legend == color)
        assert len(rows) > 0, "every legend color must appear at least once"

        r0 = int(rows.min())
        r1 = int(rows.max())
        c0 = int(cols.min())
        c1 = int(cols.max())
        assert (r0 + r1) % 2 == 0 and (c0 + c1) % 2 == 0, "template bbox must have a pixel center"

        cr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        assert int(legend[cr, cc]) == color, "template center pixel must be occupied"

        offsets = tuple(sorted((int(r - cr), int(c - cc)) for r, c in zip(rows, cols)))
        entries.append(
            {
                "color": color,
                "left": c0,
                "right": c1,
                "offsets": offsets,
            }
        )

    entries.sort(key=lambda entry: int(entry["left"]))
    for left, right in zip(entries, entries[1:]):
        assert int(left["right"]) + 1 < int(right["left"]), "legend templates must be separated by black columns"
    return entries


def _stamp(seed: tuple[int, int], offsets: tuple[tuple[int, int], ...]) -> set[tuple[int, int]]:
    sr, sc = seed
    return {(sr + dr, sc + dc) for dr, dc in offsets}


def _solve(main_area: np.ndarray, entries: list[dict[str, object]]) -> np.ndarray:
    seed_colors = [int(v) for v in np.unique(main_area) if int(v) != BLACK]
    legend_colors = [int(entry["color"]) for entry in entries]
    assert set(seed_colors) == set(legend_colors), "main-area seed colors must match legend colors exactly"

    seeds: dict[int, tuple[int, int]] = {}
    for color in legend_colors:
        coords = np.argwhere(main_area == color)
        assert coords.shape == (1, 2), "each legend color must have exactly one seed pixel"
        seeds[color] = (int(coords[0, 0]), int(coords[0, 1]))

    out = np.full(main_area.shape, BLACK, dtype=int)
    h, w = main_area.shape
    for entry in entries:
        color = int(entry["color"])
        for rr, cc in _stamp(seeds[color], entry["offsets"]):
            assert 0 <= rr < h and 0 <= cc < w, "grown template must stay inside the main area"
            out[rr, cc] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Grow each seed by the hidden legend template of the same color, using legend order as overwrite priority."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, _w = g.shape
    assert h >= 4, "grid must contain a main area, separator, and two legend rows"

    sep_r = h - 3
    sep_color = int(g[sep_r, 0])
    assert sep_color != BLACK, "separator must be visible"
    assert np.all(g[sep_r, :] == sep_color), "separator must be a full-width uniform row"

    main_area = g[:sep_r, :]
    legend = g[sep_r + 1 :, :]
    assert legend.shape[0] == 2, "legend area must have exactly two rows"

    entries = _parse_legend(legend, sep_color)
    assert entries, "legend must contain at least one template"
    return _solve(main_area, entries)


def _random_template(rng: random.Random) -> tuple[int, ...]:
    radius = rng.randint(1, 3)
    dxs = {-radius, 0, radius}
    mids = [dx for dx in range(-radius + 1, radius) if dx != 0]
    for dx in mids:
        if rng.random() < 0.45:
            dxs.add(dx)
    return tuple(sorted(dxs))


def _template_options(rng: random.Random, count: int) -> list[tuple[int, ...]] | None:
    templates: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for _ in range(count):
        for _ in range(80):
            tpl = _random_template(rng)
            if tpl not in seen:
                templates.append(tpl)
                seen.add(tpl)
                break
        else:
            return None
    return templates


def _overlap_deltas(a: tuple[int, ...], b: tuple[int, ...]) -> list[int]:
    a_set = set(a)
    deltas: list[int] = []
    for delta in sorted({da - db for da in a for db in b if abs(da - db) >= 2}):
        if a_set & {delta + db for db in b}:
            deltas.append(int(delta))
    return deltas


def _seed_bounds(h: int, w: int, offsets: tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    drs = [dr for dr, _dc in offsets]
    dcs = [dc for _dr, dc in offsets]
    return (
        -min(drs),
        h - 1 - max(drs),
        -min(dcs),
        w - 1 - max(dcs),
    )


def _place_seed(
    rng: random.Random,
    h: int,
    w: int,
    offsets: tuple[tuple[int, int], ...],
    seeds: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
) -> tuple[int, int] | None:
    r_lo, r_hi, c_lo, c_hi = _seed_bounds(h, w, offsets)
    if r_lo > r_hi or c_lo > c_hi:
        return None

    for _ in range(PLACE_RETRIES):
        seed = (rng.randint(r_lo, r_hi), rng.randint(c_lo, c_hi))
        if any(max(abs(seed[0] - r), abs(seed[1] - c)) <= 1 for r, c in seeds):
            continue
        cells = _stamp(seed, offsets)
        if cells & occupied:
            continue
        return seed
    return None


def _render_overlap_count(entries: list[dict[str, object]], seeds: dict[int, tuple[int, int]]) -> int:
    seen: set[tuple[int, int]] = set()
    overlaps = 0
    for entry in entries:
        color = int(entry["color"])
        for cell in _stamp(seeds[color], entry["offsets"]):
            if cell in seen:
                overlaps += 1
            else:
                seen.add(cell)
    return overlaps


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    count = rng.randint(*LEGEND_COUNT_RANGE)
    colors = rng.sample(COLOR_POOL, count)
    templates_1d = _template_options(rng, count)
    if templates_1d is None:
        return None

    rows = [rng.randint(0, 1) for _ in range(count)]
    if count > 1 and len(set(rows)) == 1 and rng.random() < 0.5:
        rows[rng.randrange(count)] = 1 - rows[0]
    widths = [template[-1] - template[0] + 1 for template in templates_1d]
    gaps = [rng.randint(*LEGEND_GAP_RANGE) for _ in range(count - 1)]
    x = rng.randint(*LEGEND_MARGIN_RANGE)
    starts: list[int] = []
    for i, width in enumerate(widths):
        starts.append(x)
        x += width
        if i < count - 1:
            x += gaps[i]
    x += rng.randint(*LEGEND_MARGIN_RANGE) + rng.randint(*EXTRA_WIDTH_RANGE)

    w = max(x, rng.randint(10, 18))
    main_h = rng.randint(*MAIN_H_RANGE)

    entries: list[dict[str, object]] = []
    for color, template, row, left, width in zip(colors, templates_1d, rows, starts, widths):
        offsets = tuple((0, dx) for dx in template)
        entries.append(
            {
                "color": int(color),
                "row": int(row),
                "left": int(left),
                "right": int(left + width - 1),
                "offsets": offsets,
                "dxs": template,
            }
        )

    overlap_mode = rng.random() < 0.5
    seeds: dict[int, tuple[int, int]] = {}
    occupied: set[tuple[int, int]] = set()
    placed_seed_list: list[tuple[int, int]] = []

    if overlap_mode:
        pair_options: list[tuple[int, int, list[int]]] = []
        for i in range(count):
            for j in range(i + 1, count):
                deltas = _overlap_deltas(entries[i]["dxs"], entries[j]["dxs"])
                if deltas:
                    pair_options.append((i, j, deltas))
        if not pair_options:
            return None

        i, j, deltas = rng.choice(pair_options)
        delta = rng.choice(deltas)
        offsets_i = entries[i]["offsets"]
        offsets_j = entries[j]["offsets"]
        r_lo_i, r_hi_i, c_lo_i, c_hi_i = _seed_bounds(main_h, w, offsets_i)
        _r_lo_j, _r_hi_j, c_lo_j, c_hi_j = _seed_bounds(main_h, w, offsets_j)
        c_lo = max(c_lo_i, c_lo_j - delta)
        c_hi = min(c_hi_i, c_hi_j - delta)
        if c_lo > c_hi:
            return None

        placed = False
        for _ in range(PLACE_RETRIES):
            row = rng.randint(r_lo_i, r_hi_i)
            col_i = rng.randint(c_lo, c_hi)
            seed_i = (row, col_i)
            seed_j = (row, col_i + delta)
            if max(abs(seed_i[0] - seed_j[0]), abs(seed_i[1] - seed_j[1])) <= 1:
                continue
            cells_i = _stamp(seed_i, offsets_i)
            cells_j = _stamp(seed_j, offsets_j)
            if not (cells_i & cells_j):
                continue
            seeds[int(entries[i]["color"])] = seed_i
            seeds[int(entries[j]["color"])] = seed_j
            placed_seed_list.extend([seed_i, seed_j])
            occupied |= cells_i | cells_j
            placed = True
            break
        if not placed:
            return None

    remaining = [idx for idx in range(count) if int(entries[idx]["color"]) not in seeds]
    remaining.sort(key=lambda idx: -len(entries[idx]["offsets"]))
    for idx in remaining:
        seed = _place_seed(rng, main_h, w, entries[idx]["offsets"], placed_seed_list, occupied)
        if seed is None:
            return None
        seeds[int(entries[idx]["color"])] = seed
        placed_seed_list.append(seed)
        occupied |= _stamp(seed, entries[idx]["offsets"])

    overlap_count = _render_overlap_count(entries, seeds)
    if overlap_mode and overlap_count == 0:
        return None
    if (not overlap_mode) and overlap_count != 0:
        return None

    grid = np.full((main_h + 3, w), BLACK, dtype=int)
    grid[main_h, :] = GREY
    for entry in entries:
        row = main_h + 1 + int(entry["row"])
        center = int(entry["left"]) - int(entry["dxs"][0])
        for dx in entry["dxs"]:
            grid[row, center + dx] = int(entry["color"])

    for color, seed in seeds.items():
        grid[seed[0], seed[1]] = int(color)

    out = main(grid)
    if out.shape != (main_h, w):
        return None
    if np.count_nonzero(out) <= count:
        return None
    if np.count_nonzero(out) > (main_h * w) // 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
