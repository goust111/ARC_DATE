import random
from collections import defaultdict

import numpy as np


DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            top = int(arr[:, 0].min())
            left = int(arr[:, 1].min())
            objects.append(
                {
                    "color": color,
                    "coords": arr,
                    "area": int(arr.shape[0]),
                    "anchor": (top, left),
                }
            )
    return objects


def _find_separator_row(grid: np.ndarray, bg: int) -> int:
    h, _w = grid.shape
    candidates = [
        r
        for r in range(1, h - 1)
        if int(grid[r, 0]) != bg
        and np.all(grid[r] == grid[r, 0])
        and np.any(grid[:r] != bg)
        and np.any(grid[r + 1 :] != bg)
    ]
    assert len(candidates) == 1, "expected one solid separator row between legend and body"
    return int(candidates[0])


def _read_legend_pairs(legend: np.ndarray, bg: int) -> list[tuple[int, int]]:
    comps = _components_mask(legend != bg)
    assert comps, "legend must contain at least one color pair"
    ordered = sorted(comps, key=lambda arr: (int(arr[:, 0].min()), int(arr[:, 1].min())))
    pairs: list[tuple[int, int]] = []
    for coords in ordered:
        assert coords.shape[0] == 2, "each legend entry must be a 1x2 pair"
        rows = {int(r) for r, _c in coords}
        cols = sorted(int(c) for _r, c in coords)
        assert len(rows) == 1 and cols[1] == cols[0] + 1, "legend pair must be horizontal"
        row = next(iter(rows))
        a = int(legend[row, cols[0]])
        b = int(legend[row, cols[1]])
        assert a != b, "legend colors must differ"
        pairs.append((a, b))
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _mode_color(g)
    sep_r = _find_separator_row(g, bg)
    legend = g[:sep_r]
    body = g[sep_r + 1 :]

    legend_pairs = _read_legend_pairs(legend, bg)
    objects = _monochrome_components(body, bg)

    out = g.copy()
    used: set[int] = set()
    for a, b in legend_pairs:
        a_by_area: dict[int, list[int]] = defaultdict(list)
        b_by_area: dict[int, list[int]] = defaultdict(list)
        for idx, obj in enumerate(objects):
            if idx in used:
                continue
            if obj["color"] == a:
                a_by_area[obj["area"]].append(idx)
            elif obj["color"] == b:
                b_by_area[obj["area"]].append(idx)

        for area in sorted(set(a_by_area) & set(b_by_area)):
            left = sorted(a_by_area[area], key=lambda idx: objects[idx]["anchor"])
            right = sorted(b_by_area[area], key=lambda idx: objects[idx]["anchor"])
            for idx_a, idx_b in zip(left, right):
                used.add(idx_a)
                used.add(idx_b)
                coords_a = objects[idx_a]["coords"]
                coords_b = objects[idx_b]["coords"]
                out[sep_r + 1 + coords_a[:, 0], coords_a[:, 1]] = b
                out[sep_r + 1 + coords_b[:, 0], coords_b[:, 1]] = a

    return out


BG = 8
SEP = 5
COLOR_POOL = [0, 1, 2, 3, 4, 6, 7, 9]
AREA_CHOICES = list(range(1, 8))
PAIR_COUNT_RANGE = (1, 3)
LEGEND_HEIGHT_RANGE = (1, 3)
BODY_HEIGHT_RANGE = (8, 14)
BODY_WIDTH_RANGE = (14, 22)
PLACEMENT_ATTEMPTS = 250
GENERATION_ATTEMPTS = 400


def _random_connected_mask(area: int) -> np.ndarray:
    dims: list[tuple[int, int]] = []
    for h in range(1, 6):
        for w in range(1, 6):
            if area <= h * w and area / (h * w) >= 0.3:
                dims.append((h, w))
    while True:
        h, w = random.choice(dims)
        cells = {(random.randrange(h), random.randrange(w))}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in DIRS4:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                        frontier.append((rr, cc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != area:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = mask[np.any(mask, axis=1)][:, np.any(mask, axis=0)]
        if random.random() < 0.5:
            mask = np.flipud(mask)
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        if random.random() < 0.5:
            mask = np.rot90(mask, random.randrange(4))
        return mask.copy()


def _can_place(occupied: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> bool:
    h, w = occupied.shape
    for dr, dc in np.argwhere(mask):
        rr = int(r0 + dr)
        cc = int(c0 + dc)
        r1 = max(0, rr - 1)
        r2 = min(h, rr + 2)
        c1 = max(0, cc - 1)
        c2 = min(w, cc + 2)
        if np.any(occupied[r1:r2, c1:c2]):
            return False
    return True


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    rows, cols = np.argwhere(mask).T
    grid[r0 + rows, c0 + cols] = color


def _place_objects(body_shape: tuple[int, int], specs: list[dict]) -> list[tuple[int, int]] | None:
    occupied = np.zeros(body_shape, dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(specs)
    order = sorted(range(len(specs)), key=lambda idx: (-specs[idx]["area"], idx))
    for idx in order:
        mask = specs[idx]["mask"]
        mh, mw = mask.shape
        h, w = body_shape
        positions = [
            (r0, c0)
            for r0 in range(h - mh + 1)
            for c0 in range(w - mw + 1)
            if _can_place(occupied, mask, r0, c0)
        ]
        if not positions:
            return None
        r0, c0 = random.choice(positions)
        placements[idx] = (r0, c0)
        _paint_mask(occupied, mask, r0, c0, True)
    if any(pos is None for pos in placements):
        return None
    return [(int(pos[0]), int(pos[1])) for pos in placements]


def _new_spec(area: int, input_color: int, output_color: int, role: str) -> dict:
    return {
        "area": area,
        "mask": _random_connected_mask(area),
        "input_color": int(input_color),
        "output_color": int(output_color),
        "role": role,
    }


def _legend_modes(count: int) -> list[str]:
    if count == 1:
        return ["both"]
    modes = [random.choice(["swap", "mismatch", "both"]) for _ in range(count)]
    if not any(mode in {"swap", "both"} for mode in modes):
        modes[random.randrange(count)] = "both"
    if not any(mode in {"mismatch", "both"} for mode in modes):
        modes[random.randrange(count)] = "both"
    return modes


def _build_specs(pairs: list[tuple[int, int]], remaining_colors: list[int]) -> tuple[list[dict], bool, bool, bool]:
    specs: list[dict] = []
    modes = _legend_modes(len(pairs))
    has_swap = False
    has_mismatch = False
    for (a, b), mode in zip(pairs, modes):
        used_areas: set[int] = set()
        if mode in {"swap", "both"}:
            area = random.choice(AREA_CHOICES)
            used_areas.add(area)
            specs.append(_new_spec(area, a, b, "swap"))
            specs.append(_new_spec(area, b, a, "swap"))
            has_swap = True
        if mode in {"mismatch", "both"}:
            mismatch_areas = random.sample([area for area in AREA_CHOICES if area not in used_areas], 2)
            specs.append(_new_spec(mismatch_areas[0], a, a, "mismatch"))
            specs.append(_new_spec(mismatch_areas[1], b, b, "mismatch"))
            has_mismatch = True

    has_unlisted_equal = False
    if len(remaining_colors) >= 2:
        area = random.choice(AREA_CHOICES)
        c1, c2 = random.sample(remaining_colors, 2)
        specs.append(_new_spec(area, c1, c1, "unlisted_equal"))
        specs.append(_new_spec(area, c2, c2, "unlisted_equal"))
        has_unlisted_equal = True

    if remaining_colors and random.random() < 0.7:
        specs.append(
            _new_spec(
                random.choice(AREA_CHOICES),
                random.choice(remaining_colors),
                random.choice(remaining_colors),
                "filler",
            )
        )
        specs[-1]["output_color"] = specs[-1]["input_color"]

    return specs, has_swap, has_mismatch, has_unlisted_equal


def _render_pair_row(grid: np.ndarray, pairs: list[tuple[int, int]], row: int, left: int) -> None:
    c = left
    for a, b in pairs:
        grid[row, c] = a
        grid[row, c + 1] = b
        c += 3


def _make_pair_colors() -> tuple[list[tuple[int, int]], list[int]]:
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    chosen = random.sample(COLOR_POOL, pair_count * 2)
    pairs = [(int(chosen[i]), int(chosen[i + 1])) for i in range(0, len(chosen), 2)]
    remaining = [color for color in COLOR_POOL if color not in chosen]
    return pairs, remaining


def _build_grids(
    legend_h: int,
    body_h: int,
    width: int,
    pairs: list[tuple[int, int]],
    specs: list[dict],
    placements: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    height = legend_h + 1 + body_h
    inp = np.full((height, width), BG, dtype=int)
    out = np.full((height, width), BG, dtype=int)

    legend_row = random.randrange(legend_h)
    legend_w = 3 * len(pairs) - 1
    left = random.randint(0, width - legend_w)
    drawn_pairs = [(b, a) if random.random() < 0.5 else (a, b) for a, b in pairs]
    _render_pair_row(inp, drawn_pairs, legend_row, left)
    _render_pair_row(out, drawn_pairs, legend_row, left)
    inp[legend_h, :] = SEP
    out[legend_h, :] = SEP

    body_r0 = legend_h + 1
    for spec, (r0, c0) in zip(specs, placements):
        _paint_mask(inp, spec["mask"], body_r0 + r0, c0, spec["input_color"])
        _paint_mask(out, spec["mask"], body_r0 + r0, c0, spec["output_color"])
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    pairs, remaining_colors = _make_pair_colors()
    specs, has_swap, has_mismatch, has_unlisted_equal = _build_specs(pairs, remaining_colors)
    if not (has_swap and has_mismatch and has_unlisted_equal):
        return None

    legend_w = 3 * len(pairs) - 1
    for _ in range(PLACEMENT_ATTEMPTS):
        legend_h = random.randint(*LEGEND_HEIGHT_RANGE)
        body_h = random.randint(*BODY_HEIGHT_RANGE)
        min_width = max(BODY_WIDTH_RANGE[0], legend_w)
        width = random.randint(min_width, BODY_WIDTH_RANGE[1])
        placements = _place_objects((body_h, width), specs)
        if placements is None:
            continue
        inp, expected = _build_grids(legend_h, body_h, width, pairs, specs, placements)
        if np.array_equal(inp, expected):
            continue
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if not np.array_equal(actual, expected):
            continue
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
