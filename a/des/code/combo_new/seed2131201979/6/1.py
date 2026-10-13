import random
from collections import Counter
from itertools import combinations

import numpy as np


BG = 0
FRAME_COLOR = 5
TEMPLATE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400
VALID_USED_H_SETS = {
    2: list(combinations((1, 2, 3, 4), 2)),
    3: list(combinations((1, 2, 3, 4), 3)),
}


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [p[0] for p in points]
    cols = [p[1] for p in points]
    return min(rows), min(cols), max(rows), max(cols)


def _is_rect_outline(points: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(points)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    expected = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0 + 1, r1):
        expected.add((r, c0))
        expected.add((r, c1))
    return set(points) == expected


def _find_frame(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    best = None
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for comp in _components(grid == color):
            if not _is_rect_outline(comp):
                continue
            r0, c0, r1, c1 = _bbox(comp)
            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            score = (area, r0, c0)
            if best is None or score > best[0]:
                best = (score, color, (r0, c0, r1, c1))
    assert best is not None, "expected one rectangular frame"
    return best[1], best[2]


def _extract_templates(
    grid: np.ndarray,
    bg: int,
    frame_color: int,
    frame_bbox: tuple[int, int, int, int],
) -> dict[int, dict[str, int]]:
    frame_top = frame_bbox[0]
    templates: dict[int, dict[str, int]] = {}
    for color in np.unique(grid):
        color = int(color)
        if color in (bg, frame_color):
            continue
        coords = np.argwhere((grid == color) & (np.arange(grid.shape[0])[:, None] < frame_top))
        if coords.size == 0:
            continue
        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        if not np.all(grid[r0, c0 : c1 + 1] == color):
            continue
        if not np.all(grid[r1, c0 : c1 + 1] == color):
            continue
        if not np.all(grid[r0 : r1 + 1, c0] == color):
            continue
        if not np.all(grid[r0 : r1 + 1, c1] == color):
            continue
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        holes = int(np.count_nonzero(interior == color))
        assert holes >= 1, "template must contain at least one hole marker"
        templates[color] = {
            "color": color,
            "top": r0,
            "left": c0,
            "height": r1 - r0 + 1,
            "width": c1 - c0 + 1,
            "holes": holes,
        }
    assert templates, "expected at least one upper template"
    return templates


def _scaled_outline(height: int, width: int, scale: int, color: int) -> np.ndarray:
    base = np.ones((height, width), dtype=bool)
    base[1:-1, 1:-1] = False
    mask = np.kron(base, np.ones((scale, scale), dtype=bool))
    out = np.zeros(mask.shape, dtype=int)
    out[mask] = color
    return out


def _compose_output(entries: list[dict[str, int]], bg: int) -> np.ndarray:
    assert entries, "expected at least one referenced template"
    items = []
    for idx, entry in enumerate(entries):
        scale = int(entry["holes"])
        tile = _scaled_outline(int(entry["height"]), int(entry["width"]), scale, int(entry["color"]))
        items.append(
            {
                "scale": scale,
                "top": int(entry.get("top", 0)),
                "left": int(entry.get("left", 0)),
                "order": idx,
                "tile": tile,
            }
        )
    items.sort(key=lambda item: (-item["scale"], item["top"], item["left"], item["order"]))
    out_h = max(int(item["tile"].shape[0]) for item in items)
    out_w = sum(int(item["tile"].shape[1]) for item in items) + len(items) - 1
    out = np.full((out_h, out_w), bg, dtype=int)
    cursor = 0
    for item in items:
        tile = item["tile"]
        th, tw = tile.shape
        out[:th, cursor : cursor + tw] = np.where(tile != bg, tile, out[:th, cursor : cursor + tw])
        cursor += tw + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match lower seed colors to upper hollow templates, then output the templates' scaled outlines sorted by hole count.

    Input structure:
    - The background is black.
    - The upper area contains several single-color hollow rectangular templates; each template has a 1-pixel outline and 1 to 4 same-color marker pixels inside the outline.
    - The lower area contains one larger grey rectangular frame whose interior holds colored seed pixels matching some template colors.

    Rule:
    - Find the large lower frame and read every non-background seed color inside it.
    - For each seed color, locate the upper template with the same color and count how many inner marker pixels that template contains.
    - Scale that template by the marker count, but keep only the scaled outline pixels and discard the inner markers.
    - Place the scaled outlines left to right in descending marker-count order with one black column between them, top-aligned.

    Key insight:
    - The lower seeds are only a color-based lookup clue; the template's internal marker count simultaneously chooses the scale factor and the output ordering.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    frame_color, frame_bbox = _find_frame(g, bg)
    templates = _extract_templates(g, bg, frame_color, frame_bbox)

    r0, c0, r1, c1 = frame_bbox
    referenced: list[dict[str, int]] = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            color = int(g[r, c])
            if color in (bg, frame_color):
                continue
            assert color in templates, "seed color must match an upper template"
            referenced.append(templates[color])

    return _compose_output(referenced, bg)


def _dim_candidates(hole_count: int) -> list[tuple[int, int]]:
    candidates = []
    for height in range(3, 8):
        for width in range(3, 8):
            if (height - 2) * (width - 2) > hole_count:
                candidates.append((height, width))
    return candidates


DIM_CANDIDATES = {holes: _dim_candidates(holes) for holes in range(1, 5)}


def _random_holes(rng: random.Random, height: int, width: int, hole_count: int) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1)]
    holes = rng.sample(cells, hole_count)
    holes.sort()
    return holes


def _sample_used_templates(rng: random.Random) -> list[dict[str, object]] | None:
    used_count = rng.choice((2, 3))
    hole_counts = list(rng.choice(VALID_USED_H_SETS[used_count]))
    rng.shuffle(hole_counts)

    for _ in range(200):
        specs = []
        for hole_count in hole_counts:
            height, width = rng.choice(DIM_CANDIDATES[hole_count])
            holes = _random_holes(rng, height, width, hole_count)
            specs.append(
                {
                    "height": height,
                    "width": width,
                    "holes_count": hole_count,
                    "holes": holes,
                    "used": True,
                }
            )
        out_w = sum(int(spec["holes_count"]) * int(spec["width"]) for spec in specs) + used_count - 1
        out_h = max(int(spec["holes_count"]) * int(spec["height"]) for spec in specs)
        if out_w > 30 or out_h > 30:
            continue

        for _ in range(80):
            seed_positions = [rng.choice(spec["holes"]) for spec in specs]
            if len(set(seed_positions)) != len(seed_positions):
                continue
            for spec, seed_pos in zip(specs, seed_positions):
                spec["seed_pos"] = seed_pos
            return specs
    return None


def _add_extra_templates(
    rng: random.Random, used_specs: list[dict[str, object]]
) -> list[dict[str, object]]:
    max_extra = 4 - len(used_specs)
    extra_count = rng.randint(0, max_extra)
    specs = list(used_specs)
    for _ in range(extra_count):
        hole_count = rng.randint(1, 4)
        height, width = rng.choice(DIM_CANDIDATES[hole_count])
        specs.append(
            {
                "height": height,
                "width": width,
                "holes_count": hole_count,
                "holes": _random_holes(rng, height, width, hole_count),
                "used": False,
            }
        )
    return specs


def _build_expected(used_specs: list[dict[str, object]]) -> np.ndarray:
    entries = []
    for order, spec in enumerate(used_specs):
        entries.append(
            {
                "color": int(spec["color"]),
                "height": int(spec["height"]),
                "width": int(spec["width"]),
                "holes": int(spec["holes_count"]),
                "top": order,
                "left": 0,
            }
        )
    return _compose_output(entries, BG)


def _draw_template(grid: np.ndarray, top: int, left: int, spec: dict[str, object]) -> None:
    height = int(spec["height"])
    width = int(spec["width"])
    color = int(spec["color"])
    grid[top, left : left + width] = color
    grid[top + height - 1, left : left + width] = color
    grid[top : top + height, left] = color
    grid[top : top + height, left + width - 1] = color
    for hole_r, hole_c in spec["holes"]:
        grid[top + hole_r, left + hole_c] = color


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    used_specs = _sample_used_templates(rng)
    if used_specs is None:
        return None

    specs = _add_extra_templates(rng, used_specs)
    if len(specs) > len(TEMPLATE_COLORS):
        return None

    colors = rng.sample(TEMPLATE_COLORS, len(specs))
    for spec, color in zip(specs, colors):
        spec["color"] = color

    template_order = list(range(len(specs)))
    rng.shuffle(template_order)
    ordered_specs = [specs[i] for i in template_order]
    gaps = [rng.randint(1, 3) for _ in range(max(0, len(ordered_specs) - 1))]
    top_width = sum(int(spec["width"]) for spec in ordered_specs) + sum(gaps)
    top_height = max(int(spec["height"]) for spec in ordered_specs) + rng.randint(2, 4)

    max_used_height = max(int(spec["height"]) for spec in used_specs)
    max_used_width = max(int(spec["width"]) for spec in used_specs)
    max_seed_r = max(int(spec["seed_pos"][0]) for spec in used_specs)
    max_seed_c = max(int(spec["seed_pos"][1]) for spec in used_specs)
    inner_h = max(6, max_used_height, max_seed_r + 2) + rng.randint(0, 2)
    inner_w = max(7, max_used_width, max_seed_c + 2) + rng.randint(1, 4)
    frame_h = inner_h + 2
    frame_w = inner_w + 2

    largest_template_area = max(int(spec["height"]) * int(spec["width"]) for spec in specs)
    if frame_h * frame_w <= largest_template_area:
        inner_w += 2
        frame_w = inner_w + 2

    vertical_gap = rng.randint(2, 4)
    bottom_margin = rng.randint(1, 2)
    grid_h = top_height + vertical_gap + frame_h + bottom_margin

    min_grid_w = max(top_width + 2, frame_w + 2)
    if grid_h > 30 or min_grid_w > 30:
        return None
    grid_w = min_grid_w + rng.randint(0, min(3, 30 - min_grid_w))

    grid = np.full((grid_h, grid_w), BG, dtype=int)

    top_start = rng.randint(0, grid_w - top_width)
    cursor = top_start
    for idx, spec in enumerate(ordered_specs):
        max_top = top_height - int(spec["height"])
        top = rng.randint(1, max_top) if max_top >= 1 else 0
        _draw_template(grid, top, cursor, spec)
        cursor += int(spec["width"])
        if idx < len(gaps):
            cursor += gaps[idx]

    frame_top = top_height + vertical_gap
    frame_left = rng.randint(0, grid_w - frame_w)
    grid[frame_top, frame_left : frame_left + frame_w] = FRAME_COLOR
    grid[frame_top + frame_h - 1, frame_left : frame_left + frame_w] = FRAME_COLOR
    grid[frame_top : frame_top + frame_h, frame_left] = FRAME_COLOR
    grid[frame_top : frame_top + frame_h, frame_left + frame_w - 1] = FRAME_COLOR

    for spec in used_specs:
        seed_r, seed_c = spec["seed_pos"]
        grid[frame_top + 1 + seed_r, frame_left + 1 + seed_c] = int(spec["color"])

    expected = _build_expected(used_specs)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
