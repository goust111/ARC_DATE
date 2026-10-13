from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
SEP = 5
CONTROLLER_POOL = [1, 2, 3, 4, 6, 7, 8, 9]


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _panel_ranges_from_sep_cols(width: int, sep_cols: list[int]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    prev = -1
    for col in sorted(set(int(c) for c in sep_cols)) + [width]:
        if col - prev > 1:
            ranges.append((prev + 1, col - 1))
        prev = col
    return ranges


def _find_divider(grid: np.ndarray) -> tuple[int, int, list[int], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[int, int, list[int], list[tuple[int, int]]]] = []

    for r in range(h - 1):
        row = g[r]
        if not bool(np.all(row == row[0])):
            continue
        sep = int(row[0])
        lower = g[r + 1 :, :]
        if lower.size == 0:
            continue
        sep_cols = [c for c in range(w) if bool(np.all(lower[:, c] == sep))]
        panel_ranges = _panel_ranges_from_sep_cols(w, sep_cols)
        if len(panel_ranges) >= 2:
            candidates.append((int(r), sep, sep_cols, panel_ranges))

    assert len(candidates) == 1, "expected exactly one divider row"
    return candidates[0]


def _count_bends(coords: np.ndarray) -> int:
    pts = {tuple(map(int, p)) for p in coords}
    endpoints = 0
    bends = 0
    for r, c in pts:
        nbrs = []
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in pts:
                nbrs.append((nr, nc))
        deg = len(nbrs)
        assert deg in (1, 2), "polyline must be a simple path"
        if deg == 1:
            endpoints += 1
            continue
        (r1, c1), (r2, c2) = nbrs
        if r1 != r2 and c1 != c2:
            bends += 1
    assert endpoints == 2, "polyline must have two endpoints"
    return bends


def _extract_controllers(top: np.ndarray, sep: int) -> dict[int, dict[str, int | tuple[int, int, int, int]]]:
    bg = _most_common_color(top)
    colors = sorted(set(map(int, top.ravel())) - {bg, int(sep)})
    controllers: dict[int, dict[str, int | tuple[int, int, int, int]]] = {}
    for color in colors:
        coords = np.argwhere(top == color)
        assert coords.size > 0
        r0, c0, r1, c1 = _bbox(coords)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        assert height != width, "controller extent axis must be unique"
        bends = _count_bends(coords)
        assert 1 <= bends <= 3, "expected one to three bends"
        axis = 1 if width > height else 0
        controllers[int(color)] = {
            "axis": int(axis),
            "shift": int(bends),
            "bbox": (r0, c0, r1, c1),
        }
    assert controllers, "expected at least one controller"
    return controllers


def _parse_scene(
    grid: np.ndarray,
) -> tuple[int, int, list[tuple[int, int]], dict[int, dict[str, int | tuple[int, int, int, int]]]]:
    g = np.asarray(grid, dtype=int)
    divider_r, sep, _sep_cols, panel_ranges = _find_divider(g)
    top = g[:divider_r, :]
    controllers = _extract_controllers(top, sep)
    assert len(controllers) == len(panel_ranges), "panel count must match controller count"
    return divider_r, sep, panel_ranges, controllers


def _transform_with_claims(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    divider_r, _sep, panel_ranges, controllers = _parse_scene(g)
    out = g.copy()
    controller_colors = set(controllers)
    claimed: set[int] = set()

    for c0, c1 in panel_ranges:
        panel = g[divider_r + 1 :, c0 : c1 + 1]
        claim = sorted(set(map(int, np.unique(panel))) & controller_colors)
        assert len(claim) == 1, "each panel must contain exactly one controller color"
        color = int(claim[0])
        assert color not in claimed, "panels must claim distinct controllers"
        claimed.add(color)
        spec = controllers[color]
        axis = int(spec["axis"])
        shift = int(spec["shift"])
        out[divider_r + 1 :, c0 : c1 + 1] = np.roll(panel, shift=shift, axis=axis)

    assert claimed == controller_colors, "controller claims must be bijective"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Cyclically shift each lower panel using the color-claimed bent polyline in the upper control band.

    Input structure:
    - A single-color horizontal divider splits the grid into an upper control band and a lower panel band.
    - The upper band contains several one-cell-wide monochrome bent polylines of distinct colors; each polyline has one to three bends and a clear horizontal or vertical extent.
    - The lower band is partitioned by full separator columns into the same number of rectangular panels, and each panel's multicolor pattern contains exactly one controller color.

    Rule:
    - Match every panel to the polyline whose color appears inside that panel, not to the polyline above it by position.
    - Count the matched polyline's bends k.
    - If the polyline is wider than tall, cyclically roll that panel right by k cells; if it is taller than wide, cyclically roll that panel down by k cells.

    Key insight:
    - The binding signal is shared color, while the motion parameters are hidden in the polyline's topology and extent axis; positional alignment is a decoy.
    """
    return _transform_with_claims(np.asarray(grid, dtype=int))


def _trace_vertices(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = [tuple(map(int, vertices[0]))]
    for (r0, c0), (r1, c1) in zip(vertices, vertices[1:]):
        dr = 0 if r1 == r0 else (1 if r1 > r0 else -1)
        dc = 0 if c1 == c0 else (1 if c1 > c0 else -1)
        steps = max(abs(r1 - r0), abs(c1 - c0))
        for step in range(1, steps + 1):
            cells.append((int(r0 + dr * step), int(c0 + dc * step)))
    return cells


def _polyline_patch(axis_name: str, bends: int, color: int) -> np.ndarray:
    for _ in range(200):
        r = 0
        c = 0
        vertices = [(r, c)]
        sign = random.choice([-1, 1])
        for idx in range(bends + 1):
            if axis_name == "h":
                if idx % 2 == 0:
                    c += random.randint(2, 3)
                else:
                    if idx > 1 and random.random() < 0.6:
                        sign *= -1
                    r += sign * random.randint(1, 2)
            else:
                if idx % 2 == 0:
                    r += random.randint(2, 3)
                else:
                    if idx > 1 and random.random() < 0.6:
                        sign *= -1
                    c += sign * random.randint(1, 2)
            vertices.append((r, c))

        cells = _trace_vertices(vertices)
        if len(cells) != len(set(cells)):
            continue
        rows = [rr for rr, _ in cells]
        cols = [cc for _, cc in cells]
        r_off = -min(rows)
        c_off = -min(cols)
        shifted = [(rr + r_off, cc + c_off) for rr, cc in cells]
        height = max(rr for rr, _ in shifted) + 1
        width = max(cc for _, cc in shifted) + 1
        patch = np.full((height, width), BG, dtype=int)
        for rr, cc in shifted:
            patch[rr, cc] = int(color)
        coords = np.argwhere(patch == color)
        if _count_bends(coords) != bends:
            continue
        if axis_name == "h" and not (width > height):
            continue
        if axis_name == "v" and not (height > width):
            continue
        return patch
    raise ValueError("failed to generate controller polyline")


def _derangement(items: list[int]) -> list[int]:
    order = list(items)
    for _ in range(200):
        perm = order[:]
        random.shuffle(perm)
        if all(a != b for a, b in zip(order, perm)):
            return perm
    raise ValueError("failed to build derangement")


def _panel_semantics_ok(
    panel: np.ndarray,
    claim_color: int,
    controller_specs: dict[int, dict[str, int | str]],
) -> bool:
    true_spec = controller_specs[claim_color]
    target = np.roll(panel, shift=int(true_spec["shift"]), axis=int(true_spec["axis"]))
    if np.array_equal(target, panel):
        return False
    for color, spec in controller_specs.items():
        rolled = np.roll(panel, shift=int(spec["shift"]), axis=int(spec["axis"]))
        if color != claim_color and np.array_equal(rolled, target):
            return False
    return True


def _panel_pattern(
    shape: tuple[int, int],
    claim_color: int,
    extra_colors: list[int],
    controller_specs: dict[int, dict[str, int | str]],
) -> np.ndarray:
    h, w = shape
    cells = [(r, c) for r in range(h) for c in range(w)]
    for _ in range(300):
        base = random.sample(extra_colors, 3)
        panel = np.empty((h, w), dtype=int)
        phase_a = random.randint(0, 4)
        phase_b = random.randint(0, 4)
        for r in range(h):
            for c in range(w):
                if (r + 2 * c + phase_a) % 3 == 0:
                    panel[r, c] = base[0]
                elif (2 * r + c + phase_b) % 2 == 0:
                    panel[r, c] = base[1]
                else:
                    panel[r, c] = base[2]

        claim_n = random.randint(4, min(7, h * w - 2))
        claim_positions = random.sample(cells, claim_n)
        for r, c in claim_positions:
            panel[r, c] = int(claim_color)

        accent_pool = [pos for pos in cells if pos not in claim_positions]
        accent_n = random.randint(2, min(4, len(accent_pool)))
        for r, c in random.sample(accent_pool, accent_n):
            panel[r, c] = int(random.choice(base))

        colors_present = set(map(int, np.unique(panel)))
        if claim_color not in colors_present:
            continue
        if len(colors_present) < 3:
            continue
        if not _panel_semantics_ok(panel, claim_color, controller_specs):
            continue
        return panel
    raise ValueError("failed to generate panel pattern")


def _build_position_aligned_output(
    grid: np.ndarray,
    ordered_controller_colors: list[int],
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    divider_r, _sep, panel_ranges, controllers = _parse_scene(g)
    out = g.copy()
    for idx, (c0, c1) in enumerate(panel_ranges):
        panel = g[divider_r + 1 :, c0 : c1 + 1]
        spec = controllers[int(ordered_controller_colors[idx])]
        out[divider_r + 1 :, c0 : c1 + 1] = np.roll(
            panel,
            shift=int(spec["shift"]),
            axis=int(spec["axis"]),
        )
    return out


def _scene_valid(
    grid: np.ndarray,
    claim_order: list[int],
    controller_order: list[int],
    controller_specs: dict[int, dict[str, int | str]],
) -> bool:
    try:
        divider_r, sep, panel_ranges, parsed = _parse_scene(grid)
        if sep != SEP:
            return False
        if sorted(controller_order) != sorted(parsed):
            return False
        if len(panel_ranges) != len(claim_order):
            return False
        if {int(spec["axis"]) for spec in parsed.values()} != {0, 1}:
            return False
        if {int(spec["shift"]) for spec in parsed.values()} != {1, 2, 3}:
            return False
        if divider_r <= 0 or divider_r >= grid.shape[0] - 1:
            return False

        for idx, (c0, c1) in enumerate(panel_ranges):
            panel = grid[divider_r + 1 :, c0 : c1 + 1]
            claim = sorted(set(map(int, np.unique(panel))) & set(parsed))
            if claim != [int(claim_order[idx])]:
                return False
            if not _panel_semantics_ok(panel, int(claim_order[idx]), controller_specs):
                return False

        true_out = main(grid)
        if np.array_equal(true_out, grid):
            return False
        positional_out = _build_position_aligned_output(grid, controller_order)
        if np.array_equal(positional_out, true_out):
            return False
        return True
    except Exception:
        return False


def generate_input() -> np.ndarray:
    for _ in range(400):
        panel_count = random.choice([3, 4])
        controller_colors = random.sample(CONTROLLER_POOL, panel_count)

        bend_values = [1, 2, 3]
        if panel_count == 4:
            bend_values.append(random.choice([1, 2, 3]))
        axis_values = ["h", "v"] + [random.choice(["h", "v"]) for _ in range(panel_count - 2)]
        random.shuffle(bend_values)
        random.shuffle(axis_values)

        controller_specs: dict[int, dict[str, int | str]] = {}
        patches: list[tuple[np.ndarray, int]] = []
        try:
            for color, bends, axis_name in zip(controller_colors, bend_values, axis_values):
                patch = _polyline_patch(axis_name, bends, color)
                controller_specs[int(color)] = {
                    "axis": 1 if axis_name == "h" else 0,
                    "shift": int(bends),
                    "axis_name": axis_name,
                }
                patches.append((patch, int(color)))
        except ValueError:
            continue

        random.shuffle(patches)
        gaps = [random.randint(1, 2) for _ in range(panel_count - 1)]
        top_needed = sum(patch.shape[1] for patch, _color in patches) + sum(gaps)

        widths: list[int] | None = None
        for _inner in range(200):
            candidate = [random.randint(5, 7) for _ in range(panel_count)]
            total_width = sum(candidate) + (panel_count - 1)
            if top_needed <= total_width <= 30:
                widths = candidate
                break
        if widths is None:
            continue

        lower_h = random.randint(5, 7)
        top_h = max(patch.shape[0] for patch, _color in patches) + random.randint(1, 3)
        height = top_h + 1 + lower_h
        width = sum(widths) + (panel_count - 1)
        if height > 30 or width > 30:
            continue

        grid = np.full((height, width), BG, dtype=int)
        grid[top_h, :] = SEP

        claim_order = _derangement([color for _patch, color in patches])
        extra_pool = [c for c in CONTROLLER_POOL if c not in controller_colors]

        c = 0
        for idx, panel_w in enumerate(widths):
            claim_color = int(claim_order[idx])
            try:
                panel = _panel_pattern((lower_h, panel_w), claim_color, extra_pool, controller_specs)
            except ValueError:
                break
            grid[top_h + 1 :, c : c + panel_w] = panel
            c += panel_w
            if idx < panel_count - 1:
                grid[top_h + 1 :, c] = SEP
                c += 1
        else:
            x = random.randint(0, width - top_needed)
            ordered_controller_colors: list[int] = []
            for idx, (patch, color) in enumerate(patches):
                ph, pw = patch.shape
                y = random.randint(0, top_h - ph)
                mask = patch != BG
                view = grid[y : y + ph, x : x + pw]
                if bool(np.any(view[mask] != BG)):
                    break
                view[mask] = patch[mask]
                ordered_controller_colors.append(int(color))
                x += pw
                if idx < len(gaps):
                    x += gaps[idx]
            else:
                if _scene_valid(grid, claim_order, ordered_controller_colors, controller_specs):
                    return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
