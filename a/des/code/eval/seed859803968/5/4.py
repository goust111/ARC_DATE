from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GREEN = 3
GREY = 5
MAROON = 9
RESERVED = {BG, GREEN, GREY, MAROON}
KEY_COLOR_POOL = [1, 2, 4, 6, 7, 8]
NEIGHBORS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]
DIR_STEPS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

_BASE_TEMPLATE_ROWS = [
    ("010", "111", "010"),
    ("111", "010", "010"),
    ("010", "010", "111"),
    ("100", "111", "100"),
    ("001", "111", "001"),
    ("110", "011", "001"),
    ("011", "110", "100"),
    ("111", "100", "111"),
    ("111", "001", "111"),
    ("100", "110", "011"),
    ("001", "011", "110"),
    ("111", "101", "111"),
]


def _mask_from_rows(rows: tuple[str, str, str] | list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


def _template_variants() -> list[np.ndarray]:
    seen: dict[tuple[int, ...], np.ndarray] = {}
    for rows in _BASE_TEMPLATE_ROWS:
        base = _mask_from_rows(rows)
        variants = [base]
        variants.append(np.fliplr(base))
        variants.append(np.flipud(base))
        variants.append(np.flipud(np.fliplr(base)))
        for arr in variants:
            for k in range(4):
                cur = np.rot90(arr, k)
                sig = tuple(int(x) for x in cur.reshape(-1))
                seen.setdefault(sig, cur.copy())
    return list(seen.values())


TEMPLATE_VARIANTS = _template_variants()


def _find_separator(g: np.ndarray) -> int:
    rows = [int(r) for r in range(g.shape[0]) if np.all(g[r] == GREY)]
    assert len(rows) == 1, "expected exactly one grey separator row"
    return rows[0]


def _parse_palette(g: np.ndarray, sep: int) -> dict[int, np.ndarray]:
    h, w = g.shape
    mapping: dict[int, np.ndarray] = {}
    for rr, cc in np.argwhere(~np.isin(g[sep + 1 :], list(RESERVED))):
        r = int(rr + sep + 1)
        c = int(cc)
        if r - 1 < sep + 1 or r + 1 >= h or c + 3 >= w:
            continue
        box = g[r - 1 : r + 2, c + 1 : c + 4]
        if not np.any(box == MAROON):
            continue
        if not np.all(np.isin(box, [BG, MAROON])):
            continue
        color = int(g[r, c])
        mask = (box == MAROON)
        prev = mapping.get(color)
        if prev is None:
            mapping[color] = mask.copy()
        else:
            assert np.array_equal(prev, mask), "palette key color cannot map to multiple templates"
    assert mapping, "failed to parse palette entries"
    return mapping


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
                comp.append((int(cr), int(cc)))
                for dr, dc in NEIGHBORS8:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _satellite_count(g: np.ndarray, sep: int, sr: int, sc: int) -> int:
    count = 0
    for dr, dc in NEIGHBORS8:
        rr = sr + dr
        cc = sc + dc
        if 0 <= rr < sep and 0 <= cc < g.shape[1] and int(g[rr, cc]) == GREY:
            count += 1
    return count


def _match_arrow(seed: tuple[int, int], green_comps: list[list[tuple[int, int]]]) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    sr, sc = seed
    matches: list[tuple[tuple[int, int], list[tuple[int, int]]]] = []
    for comp in green_comps:
        if len(comp) != 3:
            continue
        near: list[tuple[int, int]] = []
        for r, c in comp:
            dr = int(r - sr)
            dc = int(c - sc)
            if abs(dr) + abs(dc) == 2 and (dr == 0 or dc == 0):
                near.append((dr, dc))
        if len(near) != 1:
            continue
        dr2, dc2 = near[0]
        step = (0 if dr2 == 0 else dr2 // 2, 0 if dc2 == 0 else dc2 // 2)
        ok = True
        for r, c in comp:
            dr = int(r - sr)
            dc = int(c - sc)
            if abs(dr) + abs(dc) < 2:
                ok = False
                break
            if step[0] != 0 and dr * step[0] < 2:
                ok = False
                break
            if step[1] != 0 and dc * step[1] < 2:
                ok = False
                break
        if ok:
            matches.append((step, comp))
    assert len(matches) == 1, "expected each seed to match one green arrow"
    return matches[0]


def _layer_origin(sr: int, sc: int, step: tuple[int, int], layer: int) -> tuple[int, int]:
    dr, dc = step
    if (dr, dc) == (-1, 0):
        return sr - 3 * layer, sc - 1
    if (dr, dc) == (1, 0):
        return sr + 3 * layer - 2, sc - 1
    if (dr, dc) == (0, -1):
        return sr - 1, sc - 3 * layer
    if (dr, dc) == (0, 1):
        return sr - 1, sc + 3 * layer - 2
    raise AssertionError(f"unsupported direction {step}")


def _relative_stamp_cells(template: np.ndarray, direction: str, layers: int) -> set[tuple[int, int]]:
    step = DIR_STEPS[direction]
    cells = set()
    for layer in range(1, layers + 1):
        r0, c0 = _layer_origin(0, 0, step, layer)
        for rr, cc in zip(*np.where(template), strict=False):
            cells.add((int(r0 + rr), int(c0 + cc)))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use palette lookup + arrow direction + grey-neighbor count to grow a 3x3 template.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    sep = _find_separator(g)
    mapping = _parse_palette(g, sep)
    key_colors = list(mapping)

    green_comps = _components(g[:sep] == GREEN)
    seeds = [tuple(map(int, rc)) for rc in np.argwhere(np.isin(g[:sep], key_colors))]
    seeds.sort()
    assert seeds, "expected at least one seed"

    out = g.copy()
    canvas = out[:sep]
    canvas[(canvas == GREEN) | (canvas == GREY)] = BG

    for sr, sc in seeds:
        color = int(g[sr, sc])
        template = mapping[color]
        layers = _satellite_count(g, sep, sr, sc)
        step, _arrow = _match_arrow((sr, sc), green_comps)
        for layer in range(1, layers + 1):
            r0, c0 = _layer_origin(sr, sc, step, layer)
            assert 0 <= r0 and 0 <= c0 and r0 + 3 <= sep and c0 + 3 <= g.shape[1], "growth must stay in canvas"
            patch = out[r0 : r0 + 3, c0 : c0 + 3]
            patch[template] = color
        out[sr, sc] = color

    return out


def _arrow_offsets(direction: str, bend: int) -> list[tuple[int, int]]:
    if direction == "up":
        return [(-2, 0), (-3, 0), (-3, bend)]
    if direction == "down":
        return [(2, 0), (3, 0), (3, bend)]
    if direction == "left":
        return [(0, -2), (0, -3), (bend, -3)]
    if direction == "right":
        return [(0, 2), (0, 3), (bend, 3)]
    raise AssertionError(f"unknown direction {direction}")


def _guarded(coords: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    guard: set[tuple[int, int]] = set()
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    guard.add((rr, cc))
    return guard


def _all_inside(coords: set[tuple[int, int]], h: int, w: int) -> bool:
    return all(0 <= r < h and 0 <= c < w for r, c in coords)


def _sample_templates(count: int) -> list[np.ndarray]:
    masks = random.sample(TEMPLATE_VARIANTS, count)
    return [mask.copy() for mask in masks]


def _group_colors(num_entries: int, group_count: int) -> list[int]:
    colors = random.sample(KEY_COLOR_POOL, num_entries)
    chosen = colors.copy()
    while len(chosen) < group_count:
        chosen.append(random.choice(colors))
    random.shuffle(chosen)
    return chosen


def _sample_group_specs(num_entries: int) -> tuple[dict[int, np.ndarray], list[dict[str, object]]]:
    templates = _sample_templates(num_entries)
    key_colors = random.sample(KEY_COLOR_POOL, num_entries)
    mapping = {color: template for color, template in zip(key_colors, templates, strict=True)}

    group_count = random.randint(max(2, num_entries), min(5, num_entries + 2))
    colors = key_colors.copy()
    while len(colors) < group_count:
        colors.append(random.choice(key_colors))
    random.shuffle(colors)

    directions = [random.choice(tuple(DIR_STEPS)) for _ in range(group_count)]
    if group_count >= 2 and len(set(directions)) == 1:
        directions[random.randrange(group_count)] = random.choice([d for d in DIR_STEPS if d != directions[0]])

    layers = [random.randint(0, 4) for _ in range(group_count)]
    if max(layers) == 0:
        layers[random.randrange(group_count)] = random.randint(1, 4)

    specs: list[dict[str, object]] = []
    for color, direction, layer_count in zip(colors, directions, layers, strict=True):
        specs.append(
            {
                "color": int(color),
                "direction": direction,
                "layers": int(layer_count),
                "template": mapping[int(color)],
            }
        )
    return mapping, specs


def _place_groups(canvas_h: int, canvas_w: int, specs: list[dict[str, object]]) -> list[dict[str, object]] | None:
    placed: list[dict[str, object]] = []
    blocked: set[tuple[int, int]] = set()

    for spec in specs:
        direction = str(spec["direction"])
        layers = int(spec["layers"])
        template = np.asarray(spec["template"], dtype=bool)
        cardinal_gap = DIR_STEPS[direction]
        sat_choices = [delta for delta in NEIGHBORS8 if delta != cardinal_gap]

        placed_here = False
        for _ in range(250):
            bend = random.choice([-1, 1])
            satellites = random.sample(sat_choices, layers)
            arrow = _arrow_offsets(direction, bend)
            input_rel = {(0, 0), *satellites, *arrow}
            output_rel = {(0, 0), *_relative_stamp_cells(template, direction, layers)}
            visual_rel = input_rel | output_rel

            rel_rows = [r for r, _ in visual_rel]
            rel_cols = [c for _, c in visual_rel]
            sr_lo = -min(rel_rows)
            sr_hi = canvas_h - 1 - max(rel_rows)
            sc_lo = -min(rel_cols)
            sc_hi = canvas_w - 1 - max(rel_cols)
            if sr_lo > sr_hi or sc_lo > sc_hi:
                continue

            sr = random.randint(sr_lo, sr_hi)
            sc = random.randint(sc_lo, sc_hi)
            visual_abs = {(sr + r, sc + c) for r, c in visual_rel}
            if not _all_inside(visual_abs, canvas_h, canvas_w):
                continue

            guard = _guarded(visual_abs, canvas_h, canvas_w)
            if guard & blocked:
                continue

            accepted = dict(spec)
            accepted["seed"] = (sr, sc)
            accepted["satellites"] = [(sr + r, sc + c) for r, c in satellites]
            accepted["arrow"] = [(sr + r, sc + c) for r, c in arrow]
            placed.append(accepted)
            blocked |= guard
            placed_here = True
            break

        if not placed_here:
            return None

    return placed


def _draw_palette(grid: np.ndarray, sep: int, mapping: dict[int, np.ndarray]) -> None:
    for idx, color in enumerate(sorted(mapping)):
        center_r = sep + 2 + 4 * idx
        grid[center_r, 0] = int(color)
        patch = grid[center_r - 1 : center_r + 2, 1:4]
        patch[mapping[color]] = MAROON


def _render_input(canvas_h: int, canvas_w: int, mapping: dict[int, np.ndarray], placed: list[dict[str, object]]) -> np.ndarray:
    palette_h = 4 * len(mapping) - 1
    total_h = canvas_h + 1 + palette_h
    grid = np.full((total_h, canvas_w), BG, dtype=int)
    sep = canvas_h
    grid[sep, :] = GREY
    _draw_palette(grid, sep, mapping)

    for item in placed:
        sr, sc = item["seed"]
        color = int(item["color"])
        grid[sr, sc] = color
        for rr, cc in item["satellites"]:
            grid[rr, cc] = GREY
        for rr, cc in item["arrow"]:
            grid[rr, cc] = GREEN
    return grid


def _generate_attempt() -> np.ndarray | None:
    num_entries = random.randint(2, 4)
    mapping, specs = _sample_group_specs(num_entries)
    max_layers = max(int(spec["layers"]) for spec in specs)

    canvas_h_lo = max(16, 8 + 3 * max_layers)
    canvas_h_hi = max(canvas_h_lo + 2, canvas_h_lo + 8)
    canvas_w_lo = max(18, 10 + 3 * max_layers)
    canvas_w_hi = max(canvas_w_lo + 2, canvas_w_lo + 10)
    canvas_h = random.randint(canvas_h_lo, canvas_h_hi)
    canvas_w = random.randint(canvas_w_lo, canvas_w_hi)

    placed = _place_groups(canvas_h, canvas_w, specs)
    if placed is None:
        return None

    grid = _render_input(canvas_h, canvas_w, mapping, placed)
    try:
        _ = main(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
