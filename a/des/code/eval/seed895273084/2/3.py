from __future__ import annotations

import random

import numpy as np


BG = 0
SATELLITE = 4
SEPARATOR = 5
PALETTE = [1, 2, 3, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 400


TEMPLATE_LIBRARY = (
    frozenset(
        (dr, dc)
        for dr in range(-1, 2)
        for dc in range(-1, 2)
        if (dr, dc) != (0, 0)
    ),
    frozenset(
        (dr, dc)
        for dr in range(-2, 3)
        for dc in range(-2, 3)
        if abs(dr) + abs(dc) == 2
    ),
    frozenset(
        (dr, dc)
        for dr in range(-2, 3)
        for dc in range(-2, 3)
        if max(abs(dr), abs(dc)) == 2
    ),
    frozenset(
        (dr, dc)
        for dr in range(-2, 3)
        for dc in range(-2, 3)
        if (abs(dr) == 2 and abs(dc) <= 1)
        or (abs(dc) == 2 and abs(dr) <= 1)
        or (abs(dr) == 1 and abs(dc) == 1)
    ),
    frozenset(
        (dr, dc)
        for dr in range(-2, 3)
        for dc in range(-2, 3)
        if abs(dr) + abs(dc) == 2 or (abs(dr) == abs(dc) and (dr, dc) != (0, 0))
    ),
)


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _is_connected8(coords: list[tuple[int, int]]) -> bool:
    if not coords:
        return False
    remaining = set(coords)
    stack = [coords[0]]
    seen = set()
    while stack:
        cell = stack.pop()
        if cell in seen:
            continue
        seen.add(cell)
        for nxt in _neighbors8(*cell):
            if nxt in remaining and nxt not in seen:
                stack.append(nxt)
    return len(seen) == len(remaining)


def _find_separator_row(g: np.ndarray) -> tuple[int, int]:
    rows = [
        int(r)
        for r in range(g.shape[0])
        if np.all(g[r, :] == g[r, 0]) and int(g[r, 0]) != BG
    ]
    assert len(rows) == 1, "expected one non-background horizontal separator row"
    row = rows[0]
    return row, int(g[row, 0])


def _extract_templates(top: np.ndarray, separator_color: int) -> dict[int, frozenset[tuple[int, int]]]:
    colors = [int(color) for color in np.unique(top) if int(color) not in {BG, separator_color}]
    assert colors, "expected at least one template color"

    templates: dict[int, frozenset[tuple[int, int]]] = {}
    for color in colors:
        coords = [tuple(map(int, rc)) for rc in np.argwhere(top == color)]
        assert _is_connected8(coords), "each template color must form one 8-connected object"

        rows = [r for r, _c in coords]
        cols = [c for _r, c in coords]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        assert (r1 - r0) % 2 == 0 and (c1 - c0) % 2 == 0, "templates need an odd-sized bbox"

        center_r = (r0 + r1) // 2
        center_c = (c0 + c1) // 2
        offsets = frozenset((r - center_r, c - center_c) for r, c in coords)
        assert (0, 0) not in offsets, "template center must stay black"
        templates[color] = offsets

    return templates


def _count_satellites(g: np.ndarray, r: int, c: int) -> int:
    r0 = max(0, r - 3)
    r1 = min(g.shape[0], r + 4)
    c0 = max(0, c - 3)
    c1 = min(g.shape[1], c + 4)
    return int(np.count_nonzero(g[r0:r1, c0:c1] == SATELLITE))


def _seed_positions(g: np.ndarray, sep_row: int) -> list[tuple[int, int]]:
    lower = g[sep_row + 1 :]
    coords = np.argwhere((lower != BG) & (lower != SATELLITE))
    return [(sep_row + 1 + int(r), int(c)) for r, c in coords]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each seed color to its top template, count nearby yellow satellites,
    then stamp scaled copies of the template around the seed for layers 1..N.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    sep_row, sep_color = _find_separator_row(g)
    templates = _extract_templates(g[:sep_row, :], sep_color)

    out = g.copy()
    lower = out[sep_row + 1 :]
    lower[lower == SATELLITE] = BG

    for r, c in _seed_positions(g, sep_row):
        color = int(g[r, c])
        assert color in templates, "every seed color must have a matching template"
        layer_count = _count_satellites(g, r, c)
        for scale in range(1, layer_count + 1):
            for dr, dc in templates[color]:
                rr = r + scale * dr
                cc = c + scale * dc
                assert sep_row + 1 <= rr < g.shape[0] and 0 <= cc < g.shape[1], "growth must stay in-bounds"
                assert int(out[rr, cc]) in {BG, color}, "growth regions must not conflict"
                out[rr, cc] = color

    return out


def _render_patch(offsets: frozenset[tuple[int, int]], color: int) -> np.ndarray:
    rows = [r for r, _c in offsets]
    cols = [c for _r, c in offsets]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    for r, c in offsets:
        patch[r - r0, c - c0] = color
    return patch


def _random_partition(total: int, parts: int) -> list[int]:
    bins = [0] * parts
    for _ in range(total):
        bins[random.randrange(parts)] += 1
    return bins


def _pack_templates(
    offsets_by_color: dict[int, frozenset[tuple[int, int]]],
) -> tuple[np.ndarray, dict[int, frozenset[tuple[int, int]]]] | None:
    items = list(offsets_by_color.items())
    random.shuffle(items)

    if len(items) == 1:
        rows = [[items[0]]]
    else:
        row_count = 2 if len(items) >= 4 and random.random() < 0.5 else 1
        if row_count == 1:
            rows = [items]
        else:
            cut = random.randint(1, len(items) - 1)
            rows = [items[:cut], items[cut:]]

    row_specs = []
    max_row_width = 0
    for row_items in rows:
        patches = [(color, _render_patch(offsets, color)) for color, offsets in row_items]
        gaps = [random.randint(2, 4) for _ in range(max(0, len(patches) - 1))]
        row_height = max(patch.shape[0] for _color, patch in patches)
        row_width = sum(patch.shape[1] for _color, patch in patches) + sum(gaps)
        row_specs.append(
            {
                "patches": patches,
                "gaps": gaps,
                "row_height": row_height,
                "row_width": row_width,
            }
        )
        max_row_width = max(max_row_width, row_width)

    side_margin = 1
    extra_width = random.randint(0, 6)
    row_gap = random.randint(1, 2)
    top_width = max_row_width + 2 * side_margin + extra_width
    core_height = sum(int(spec["row_height"]) for spec in row_specs) + row_gap * (len(row_specs) - 1)
    extra_height = random.randint(0, 2)
    top_height = core_height + 2 * side_margin + extra_height
    top = np.full((top_height, top_width), BG, dtype=int)

    row_slack = _random_partition(extra_height, len(row_specs) + 1)
    current_r = side_margin + row_slack[0]
    for row_index, spec in enumerate(row_specs):
        patches = spec["patches"]
        gaps = spec["gaps"]
        row_height = int(spec["row_height"])
        row_width = int(spec["row_width"])
        band_top = current_r
        start_c = side_margin + random.randint(0, top_width - 2 * side_margin - row_width)
        c = start_c
        for idx, (color, patch) in enumerate(patches):
            patch_r = band_top + random.randint(0, row_height - patch.shape[0])
            patch_h, patch_w = patch.shape
            top[patch_r : patch_r + patch_h, c : c + patch_w] = patch
            c += patch_w
            if idx < len(gaps):
                c += gaps[idx]
        current_r += row_height
        if row_index < len(row_specs) - 1:
            current_r += row_gap + row_slack[row_index + 1]

    return top, offsets_by_color


def _growth_offsets(template_offsets: frozenset[tuple[int, int]], layers: int) -> set[tuple[int, int]]:
    cells = set()
    for scale in range(1, layers + 1):
        for dr, dc in template_offsets:
            cells.add((scale * dr, scale * dc))
    return cells


def _choose_satellites(blocked: set[tuple[int, int]], count: int) -> list[tuple[int, int]] | None:
    candidates = [
        (dr, dc)
        for dr in range(-3, 4)
        for dc in range(-3, 4)
        if max(abs(dr), abs(dc)) <= 3 and (dr, dc) != (0, 0) and (dr, dc) not in blocked
    ]
    random.shuffle(candidates)

    chosen: list[tuple[int, int]] = []
    for dr, dc in candidates:
        if all(max(abs(dr - r0), abs(dc - c0)) >= 2 for r0, c0 in chosen):
            chosen.append((dr, dc))
            if len(chosen) == count:
                return chosen
    if len(candidates) >= count:
        return candidates[:count]
    return None


def _place_seed_jobs(jobs: list[dict[str, object]], height: int, width: int) -> bool:
    occupied = np.zeros((height, width), dtype=bool)
    order = sorted(range(len(jobs)), key=lambda idx: int(jobs[idx]["reserve"]), reverse=True)
    for idx in order:
        reserve = int(jobs[idx]["reserve"])
        candidates = []
        for r in range(reserve, height - reserve):
            for c in range(reserve, width - reserve):
                if np.any(occupied[r - reserve : r + reserve + 1, c - reserve : c + reserve + 1]):
                    continue
                candidates.append((r, c))
        if not candidates:
            return False
        center = random.choice(candidates)
        jobs[idx]["center"] = center
        r, c = center
        occupied[r - reserve : r + reserve + 1, c - reserve : c + reserve + 1] = True
    return True


def _build_expected(
    grid: np.ndarray,
    sep_row: int,
    jobs: list[dict[str, object]],
) -> np.ndarray:
    out = grid.copy()
    lower = out[sep_row + 1 :]
    lower[lower == SATELLITE] = BG
    for job in jobs:
        r, c = job["abs_center"]
        color = int(job["color"])
        for dr, dc in job["growth_cells"]:
            out[r + dr, c + dc] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    template_count = random.randint(3, 5)
    template_shapes = random.sample(list(TEMPLATE_LIBRARY), template_count)
    separator_color = SEPARATOR
    template_colors = random.sample(PALETTE, template_count)
    offsets_by_color = {
        color: shape
        for color, shape in zip(template_colors, template_shapes, strict=True)
    }
    top_pack = _pack_templates(offsets_by_color)
    if top_pack is None:
        return None
    top, _ = top_pack

    seed_count = random.randint(2, 3)
    seed_colors = [random.choice(template_colors) for _ in range(seed_count)]
    layer_counts = [random.randint(1, 4) for _ in range(seed_count)]

    jobs: list[dict[str, object]] = []
    for color, layers in zip(seed_colors, layer_counts, strict=True):
        growth_cells = _growth_offsets(offsets_by_color[color], layers)
        blocked = set(growth_cells)
        blocked.add((0, 0))
        satellites = _choose_satellites(blocked, layers)
        if satellites is None:
            return None
        extent = max(3, max(max(abs(dr), abs(dc)) for dr, dc in blocked))
        jobs.append(
            {
                "color": color,
                "layers": layers,
                "growth_cells": growth_cells,
                "satellites": satellites,
                "reserve": extent + 1,
            }
        )

    max_reserve = max(int(job["reserve"]) for job in jobs)
    bottom_height = random.randint(max(16, 2 * max_reserve + 1), max(20, 2 * max_reserve + 7))
    bottom_width = random.randint(max(16, 2 * max_reserve + 1), max(22, 2 * max_reserve + 9))
    if not _place_seed_jobs(jobs, bottom_height, bottom_width):
        return None

    bottom = np.full((bottom_height, bottom_width), BG, dtype=int)
    for job in jobs:
        center = tuple(job["center"])
        job["abs_center"] = (top.shape[0] + 1 + center[0], center[1])
        r, c = center
        color = int(job["color"])
        bottom[r, c] = color
        for dr, dc in job["satellites"]:
            rr = r + dr
            cc = c + dc
            if bottom[rr, cc] != BG:
                return None
            bottom[rr, cc] = SATELLITE

    separator = np.full((1, max(top.shape[1], bottom.shape[1])), separator_color, dtype=int)
    width = separator.shape[1]
    padded_top = np.full((top.shape[0], width), BG, dtype=int)
    padded_bottom = np.full((bottom.shape[0], width), BG, dtype=int)

    top_c0 = random.randint(0, width - top.shape[1])
    bottom_c0 = random.randint(0, width - bottom.shape[1])
    padded_top[:, top_c0 : top_c0 + top.shape[1]] = top
    padded_bottom[:, bottom_c0 : bottom_c0 + bottom.shape[1]] = bottom

    for job in jobs:
        abs_r, abs_c = job["abs_center"]
        job["abs_center"] = (abs_r, abs_c + bottom_c0)

    grid = np.vstack([padded_top, separator, padded_bottom])
    expected = _build_expected(grid, padded_top.shape[0], jobs)

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
