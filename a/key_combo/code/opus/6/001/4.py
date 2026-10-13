from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


ALLOWED_COLORS = (1, 2, 3, 4, 5, 6)
RING_CELLS = (
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 2),
    (2, 0),
    (2, 1),
    (2, 2),
)
CORNER_ORDER = ((0, 0), (0, 2), (2, 2), (2, 0))
SIDE_ORDER = ((0, 1), (1, 2), (2, 1), (1, 0))
GENERATION_RETRIES = 600


def _get_bg(grid: np.ndarray) -> int:
    counts = Counter(map(int, np.asarray(grid).ravel()))
    return int(counts.most_common(1)[0][0])


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_non_bg_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    mask = grid != int(bg)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(
                {
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                    "area": len(coords),
                }
            )
    return comps


def _is_legend(grid: np.ndarray, bg: int, comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if w != 2 or not (2 <= h <= 4):
        return False
    if comp["area"] != h * w:
        return False
    return bool(np.all(grid[r0 : r1 + 1, c0 : c1 + 1] != int(bg)))


def _is_ring(grid: np.ndarray, bg: int, comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    if (r1 - r0 + 1, c1 - c0 + 1) != (3, 3):
        return False
    if comp["area"] != 8:
        return False
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    if int(patch[1, 1]) != int(bg):
        return False
    for dr, dc in RING_CELLS:
        if int(patch[dr, dc]) == int(bg):
            return False
    return True


def _parse_grid(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    comps = _find_non_bg_components(g, bg)

    legends: list[dict] = []
    rings: list[dict] = []
    for comp in comps:
        if _is_legend(g, bg, comp):
            legends.append(comp)
        elif _is_ring(g, bg, comp):
            rings.append(comp)
        else:
            raise AssertionError("unexpected non-background component")

    assert len(legends) == 1, "expected exactly one 2-column legend"
    assert len(rings) in (2, 3), "expected 2 or 3 rings"

    legend = legends[0]
    r0, c0, r1, _c1 = legend["bbox"]
    rules = [(int(g[r, c0]), int(g[r, c0 + 1])) for r in range(r0, r1 + 1)]
    ring_tops = sorted((int(comp["bbox"][0]), int(comp["bbox"][1])) for comp in rings)
    return bg, rules, ring_tops


def _replace_color(color: int, rules: list[tuple[int, int]]) -> tuple[int, int]:
    current = int(color)
    changes = 0
    for left, right in rules:
        if current == int(left):
            current = int(right)
            changes += 1
    return current, changes


def _apply_chain_to_patch(patch: np.ndarray, rules: list[tuple[int, int]]) -> tuple[np.ndarray, int]:
    out = patch.copy()
    max_changes = 0
    for dr, dc in RING_CELLS:
        out[dr, dc], changes = _replace_color(int(out[dr, dc]), rules)
        max_changes = max(max_changes, changes)
    return out, max_changes


def _cycle_patch(patch: np.ndarray) -> np.ndarray:
    out = patch.copy()

    corners = [int(patch[r, c]) for r, c in CORNER_ORDER]
    new_corners = [corners[3], corners[0], corners[1], corners[2]]
    for (r, c), value in zip(CORNER_ORDER, new_corners):
        out[r, c] = value

    sides = [int(patch[r, c]) for r, c in SIDE_ORDER]
    new_sides = [sides[1], sides[2], sides[3], sides[0]]
    for (r, c), value in zip(SIDE_ORDER, new_sides):
        out[r, c] = value

    return out


def _solve_variant(grid: np.ndarray, mode: str) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg, rules, ring_tops = _parse_grid(g)
    out = g.copy()

    for r0, c0 in ring_tops:
        patch = g[r0 : r0 + 3, c0 : c0 + 3].copy()
        if mode == "replace_then_cycle":
            recolored, _ = _apply_chain_to_patch(patch, rules)
            final = _cycle_patch(recolored)
        elif mode == "replace_only":
            final, _ = _apply_chain_to_patch(patch, rules)
        elif mode == "cycle_only":
            final = _cycle_patch(patch)
        elif mode == "cycle_then_replace":
            cycled = _cycle_patch(patch)
            final, _ = _apply_chain_to_patch(cycled, rules)
        else:
            raise ValueError(f"unknown mode: {mode}")
        final[1, 1] = bg
        out[r0 : r0 + 3, c0 : c0 + 3] = final

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Apply the legend's chained recoloring to each ring, then cycle corner and side colors in opposite directions.

    Input structure:
    - The background is black and all non-background pixels belong either to 2 or 3 separated 3x3 rings or to one separated 2-column vertical legend.
    - Each ring is a complete 3x3 frame of eight colored pixels around a black center.
    - Each legend row contains a left color and a right color, read from top to bottom as chained replacement rules.

    Rule:
    - Ignore the background and the legend itself when replacing colors.
    - For every ring pixel, scan the legend rows from top to bottom; whenever the current color matches the row's left color, replace it with the row's right color and continue scanning later rows on the updated color.
    - After all ring pixels have been chain-replaced, rotate the four corner colors one step clockwise and the four side-center colors one step counterclockwise, leaving the legend and all black centers unchanged.

    Key insight:
    - The cycle acts on the post-replacement colors, so chained recoloring changes which visible colors are being moved around each ring.
    """
    return _solve_variant(np.asarray(grid, dtype=int), "replace_then_cycle")


def _rect_far(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 2) -> bool:
    return (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _render_ring(values: list[int]) -> np.ndarray:
    ring = np.zeros((3, 3), dtype=int)
    for (dr, dc), value in zip(RING_CELLS, values):
        ring[dr, dc] = int(value)
    return ring


def _place_layout(
    h: int, w: int, ring_count: int, legend_rows: int, side: str
) -> tuple[tuple[int, int, int, int], list[tuple[int, int]]] | None:
    if side == "left":
        legend_c0 = random.choice([0, 1])
    else:
        legend_c0 = random.choice([w - 2, w - 3])
    legend_r0 = random.randint(0, h - legend_rows)
    legend_box = (legend_r0, legend_c0, legend_r0 + legend_rows - 1, legend_c0 + 1)

    candidates = [(r, c) for r in range(h - 2) for c in range(w - 2)]
    random.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    boxes = [legend_box]
    for r0, c0 in candidates:
        box = (r0, c0, r0 + 2, c0 + 2)
        if any(not _rect_far(box, other, pad=2) for other in boxes):
            continue
        chosen.append((r0, c0))
        boxes.append(box)
        if len(chosen) == ring_count:
            return legend_box, chosen
    return None


def _sample_rules(rule_count: int) -> list[tuple[int, int]]:
    chain_len = random.randint(2, min(3, rule_count))
    chain_colors = random.sample(ALLOWED_COLORS, chain_len + 1)
    rules = [(chain_colors[i], chain_colors[i + 1]) for i in range(chain_len)]

    used_src = {src for src, _dst in rules}
    while len(rules) < rule_count:
        src_choices = [color for color in ALLOWED_COLORS if color not in used_src]
        if not src_choices:
            src_choices = list(ALLOWED_COLORS)
        src = random.choice(src_choices)
        dst = random.choice([color for color in ALLOWED_COLORS if color != src])
        rules.append((src, dst))
        used_src.add(src)
    return rules


def _random_ring_values(pool: list[int]) -> list[int]:
    palette_size = random.randint(2, min(4, len(pool)))
    palette = random.sample(pool, palette_size)
    values = [random.choice(palette) for _ in range(8)]
    if len(set(values)) < 2:
        values[random.randrange(8)] = random.choice([c for c in palette if c != values[0]])
    return values


def _sample_ring_values(ring_count: int, rules: list[tuple[int, int]]) -> list[list[int]] | None:
    legend_colors = sorted({color for pair in rules for color in pair})
    extras = [color for color in ALLOWED_COLORS if color not in legend_colors]
    pool = legend_colors[:]
    if extras and random.random() < 0.6:
        pool.append(random.choice(extras))

    for _ in range(200):
        rings = [_random_ring_values(pool) for _ in range(ring_count)]
        flat = [value for ring in rings for value in ring]
        missing = [color for color in legend_colors if color not in flat]
        if missing:
            slots = [(ri, pi) for ri in range(ring_count) for pi in range(8)]
            random.shuffle(slots)
            for color, (ri, pi) in zip(missing, slots):
                rings[ri][pi] = color
        if all(len(set(ring)) >= 2 for ring in rings):
            return rings
    return None


def _build_grid(
    h: int,
    w: int,
    legend_box: tuple[int, int, int, int],
    ring_positions: list[tuple[int, int]],
    rules: list[tuple[int, int]],
    ring_values: list[list[int]],
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    lr0, lc0, lr1, _lc1 = legend_box
    for idx, r in enumerate(range(lr0, lr1 + 1)):
        left, right = rules[idx]
        grid[r, lc0] = int(left)
        grid[r, lc0 + 1] = int(right)
    for (r0, c0), values in zip(ring_positions, ring_values):
        grid[r0 : r0 + 3, c0 : c0 + 3] = _render_ring(values)
    return grid


def _ring_patch_values(grid: np.ndarray, ring_tops: list[tuple[int, int]]) -> list[list[int]]:
    vals: list[list[int]] = []
    for r0, c0 in ring_tops:
        patch = grid[r0 : r0 + 3, c0 : c0 + 3]
        vals.append([int(patch[dr, dc]) for dr, dc in RING_CELLS])
    return vals


def _semantic_ok(grid: np.ndarray) -> bool:
    try:
        bg, rules, ring_tops = _parse_grid(grid)
    except AssertionError:
        return False

    if bg != 0:
        return False

    ring_values = _ring_patch_values(grid, ring_tops)
    legend_colors = {color for pair in rules for color in pair}
    flat_values = [value for ring in ring_values for value in ring]
    if not legend_colors.issubset(set(flat_values)):
        return False

    max_chain_changes = 0
    recolored_unique_counts = []
    for r0, c0 in ring_tops:
        patch = grid[r0 : r0 + 3, c0 : c0 + 3]
        recolored, changes = _apply_chain_to_patch(patch, rules)
        max_chain_changes = max(max_chain_changes, changes)
        recolored_unique_counts.append(len({int(recolored[dr, dc]) for dr, dc in RING_CELLS}))

    if max_chain_changes < 2:
        return False
    if any(count < 2 for count in recolored_unique_counts):
        return False

    full = _solve_variant(grid, "replace_then_cycle")
    replace_only = _solve_variant(grid, "replace_only")
    cycle_only = _solve_variant(grid, "cycle_only")

    if np.array_equal(full, grid):
        return False
    if np.array_equal(full, replace_only):
        return False
    if np.array_equal(full, cycle_only):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(13, 18)
    w = random.randint(13, 18)
    ring_count = random.choice([2, 3])
    legend_rows = random.randint(2, 4)
    side = random.choice(["left", "right"])

    layout = _place_layout(h, w, ring_count, legend_rows, side)
    if layout is None:
        return None
    legend_box, ring_positions = layout

    rules = _sample_rules(legend_rows)
    for _ in range(150):
        ring_values = _sample_ring_values(ring_count, rules)
        if ring_values is None:
            continue
        grid = _build_grid(h, w, legend_box, ring_positions, rules, ring_values)
        if _semantic_ok(grid):
            break
    else:
        return None

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, _solve_variant(grid, "replace_then_cycle")):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid chained-ring example")
