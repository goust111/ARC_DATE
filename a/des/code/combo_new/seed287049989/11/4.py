from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
SIDE_TO_ARROW_DIR = {
    "left": (0, 1),
    "right": (0, -1),
    "top": (1, 0),
    "bottom": (-1, 0),
}
SIDE_TO_DELTA = {
    "left": (0, -1),
    "right": (0, 1),
    "top": (-1, 0),
    "bottom": (1, 0),
}
GEN_ATTEMPTS = 400
MODULE_ATTEMPTS = 120
SIDE_ATTEMPTS = 100
CANVAS_ATTEMPTS = 120
MIN_STRIP_LEN = 3
MAX_STRIP_LEN = 7
MIN_SHAFT_LEN = 2
MAX_SHAFT_LEN = 4
MIN_GAP = 1
MAX_GAP = 4


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            queue = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append({"color": color, "coords": coords})
    return comps


def _classify_strip(coords: list[tuple[int, int]]) -> dict | None:
    rows = sorted({r for r, _ in coords})
    cols = sorted({c for _, c in coords})
    if len(rows) == 1:
        r = rows[0]
        ordered = sorted(c for _, c in coords)
        if ordered == list(range(ordered[0], ordered[-1] + 1)):
            cells = {(r, c) for c in ordered}
            return {
                "orientation": "h",
                "cells": cells,
                "row": r,
                "start": ordered[0],
                "end": ordered[-1],
            }
    if len(cols) == 1:
        c = cols[0]
        ordered = sorted(r for r, _ in coords)
        if ordered == list(range(ordered[0], ordered[-1] + 1)):
            cells = {(r, c) for r in ordered}
            return {
                "orientation": "v",
                "cells": cells,
                "col": c,
                "start": ordered[0],
                "end": ordered[-1],
            }
    return None


def _perp_deltas(direction: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    dr, dc = direction
    if dr == 0:
        return (-1, 0), (1, 0)
    return (0, -1), (0, 1)


def _classify_arrow(coords: list[tuple[int, int]]) -> dict | None:
    cells = set(coords)
    if not (4 <= len(cells) <= 6):
        return None

    degrees = {}
    for r, c in cells:
        degrees[(r, c)] = sum((nr, nc) in cells for nr, nc in _neighbors4(r, c))

    if any(deg not in (1, 2, 3) for deg in degrees.values()):
        return None
    tails = [cell for cell, deg in degrees.items() if deg == 3]
    tips = [cell for cell, deg in degrees.items() if deg == 1]
    mids = [cell for cell, deg in degrees.items() if deg == 2]
    if len(tails) != 1 or len(tips) != 3 or len(mids) != len(cells) - 4:
        return None

    tail = tails[0]
    tail_r, tail_c = tail
    neighbor_offsets = {
        (nr - tail_r, nc - tail_c)
        for nr, nc in _neighbors4(tail_r, tail_c)
        if (nr, nc) in cells
    }
    for direction in DIRS4:
        dr, dc = direction
        perp1, perp2 = _perp_deltas(direction)
        if neighbor_offsets != {direction, perp1, perp2}:
            continue

        shaft = [tail]
        step = 1
        while (tail_r + step * dr, tail_c + step * dc) in cells:
            shaft.append((tail_r + step * dr, tail_c + step * dc))
            step += 1
        shaft_len = len(shaft)
        if not (MIN_SHAFT_LEN <= shaft_len <= MAX_SHAFT_LEN):
            continue

        expected = {tail}
        expected.add((tail_r + perp1[0], tail_c + perp1[1]))
        expected.add((tail_r + perp2[0], tail_c + perp2[1]))
        for k in range(1, shaft_len):
            expected.add((tail_r + k * dr, tail_c + k * dc))
        if expected != cells:
            continue

        return {
            "cells": cells,
            "tail": tail,
            "tip": shaft[-1],
            "dir": direction,
            "shaft_len": shaft_len,
        }
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase each T-shaped arrow and expand the matched color band from the side the arrow came from.

    Input structure:
    - The background is the dominant color.
    - Foreground objects are one-cell-wide monochrome horizontal or vertical bands, plus
      monochrome T-shaped arrows made from a straight shaft and a 3-pixel tail bar.
    - Each arrow has the same color as exactly one target band, and a clear background ray
      from its tip reaches that band in the arrow's pointing direction.

    Rule:
    - Detect the arrow direction from the T shape: the crossbar marks the tail and the opposite
      endpoint is the tip.
    - Remove every arrow completely.
    - For each arrow, extend the matched band toward the arrow by the shaft length. If several
      arrows hit the same side of the same band, add their shaft lengths together.
    - Leave all other band cells unchanged.

    Key insight:
    - The T shape tells both which endpoint is the tip and which side of the same-colored band
      must be expanded; the expansion amount is the shaft length, not the gap to the band.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    h, w = g.shape

    components = _find_components(g, bg)
    strips = []
    arrows = []
    for comp in components:
        arrow = _classify_arrow(comp["coords"])
        if arrow is not None:
            arrow["color"] = int(comp["color"])
            arrows.append(arrow)
            continue
        strip = _classify_strip(comp["coords"])
        if strip is not None:
            strip["color"] = int(comp["color"])
            strips.append(strip)
            continue
        raise AssertionError("unrecognized foreground component")

    strip_at = {}
    for strip_id, strip in enumerate(strips):
        for cell in strip["cells"]:
            strip_at[cell] = strip_id

    out = g.copy()
    extensions: dict[tuple[int, tuple[int, int]], int] = defaultdict(int)

    for arrow in arrows:
        for r, c in arrow["cells"]:
            out[r, c] = bg

        dr, dc = arrow["dir"]
        rr, cc = arrow["tip"][0] + dr, arrow["tip"][1] + dc
        while 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == bg:
            rr += dr
            cc += dc

        assert 0 <= rr < h and 0 <= cc < w, "arrow ray must hit a strip"
        strip_id = strip_at.get((rr, cc))
        assert strip_id is not None, "arrow must hit a strip component"
        strip = strips[strip_id]
        assert strip["color"] == arrow["color"], "arrow and strip colors must match"

        delta = (-dr, -dc)
        extensions[(strip_id, delta)] += int(arrow["shaft_len"])

    for (strip_id, delta), amount in extensions.items():
        dr, dc = delta
        strip = strips[strip_id]
        color = int(strip["color"])
        for k in range(1, amount + 1):
            for r, c in strip["cells"]:
                rr, cc = r + dr * k, c + dc * k
                assert 0 <= rr < h and 0 <= cc < w, "band extension must stay in-bounds"
                out[rr, cc] = color

    return out


def _component_is_separate(new_cells: set[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    if new_cells & occupied:
        return False
    for r, c in new_cells:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in occupied:
                return False
    return True


def _strip_cells(orientation: str, length: int) -> set[tuple[int, int]]:
    if orientation == "h":
        return {(0, c) for c in range(length)}
    return {(r, 0) for r in range(length)}


def _boundary_targets(orientation: str, length: int, side: str) -> list[tuple[int, int]]:
    if orientation == "h":
        if side == "left":
            return [(0, 0)]
        if side == "right":
            return [(0, length - 1)]
        return [(0, c) for c in range(length)]
    if side == "top":
        return [(0, 0)]
    if side == "bottom":
        return [(length - 1, 0)]
    return [(r, 0) for r in range(length)]


def _make_arrow_cells(
    target: tuple[int, int],
    side: str,
    shaft_len: int,
    gap: int,
) -> set[tuple[int, int]]:
    tr, tc = target
    dr, dc = SIDE_TO_ARROW_DIR[side]
    tip = (tr - dr * (gap + 1), tc - dc * (gap + 1))
    tail = (tip[0] - dr * (shaft_len - 1), tip[1] - dc * (shaft_len - 1))
    perp1, perp2 = _perp_deltas((dr, dc))

    cells = {(tail[0] + dr * k, tail[1] + dc * k) for k in range(shaft_len)}
    cells.add((tail[0] + perp1[0], tail[1] + perp1[1]))
    cells.add((tail[0] + perp2[0], tail[1] + perp2[1]))
    return cells


def _side_capacity(orientation: str, length: int, side: str) -> int:
    broadside = (orientation == "h" and side in ("top", "bottom")) or (
        orientation == "v" and side in ("left", "right")
    )
    if broadside and length >= 5:
        return 2
    return 1


def _sample_side_counts(orientation: str, length: int) -> dict[str, int] | None:
    for _ in range(60):
        counts = {}
        total = 0
        for side in SIDE_TO_ARROW_DIR:
            cap = _side_capacity(orientation, length, side)
            if cap == 2:
                count = random.choice([0, 1, 2])
            else:
                count = random.choice([0, 1])
            counts[side] = count
            total += count
        if 1 <= total <= 5:
            return counts
    return None


def _sample_arrows_for_side(
    orientation: str,
    length: int,
    side: str,
    count: int,
    occupied: set[tuple[int, int]],
) -> list[dict] | None:
    targets = _boundary_targets(orientation, length, side)
    if count > len(targets):
        return None

    for _ in range(SIDE_ATTEMPTS):
        local_occupied = set(occupied)
        used_targets = set()
        arrows = []
        ok = True
        for _arrow_index in range(count):
            placed = False
            for _choice in range(80):
                choices = [target for target in targets if target not in used_targets]
                if not choices:
                    break
                target = random.choice(choices)
                shaft_len = random.randint(MIN_SHAFT_LEN, MAX_SHAFT_LEN)
                gap = random.randint(MIN_GAP, MAX_GAP)
                cells = _make_arrow_cells(target, side, shaft_len, gap)
                if _component_is_separate(cells, local_occupied):
                    arrows.append(
                        {
                            "cells": cells,
                            "shaft_len": shaft_len,
                            "side": side,
                        }
                    )
                    local_occupied |= cells
                    used_targets.add(target)
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if ok:
            return arrows
    return None


def _normalize_module(
    color: int,
    input_cells: set[tuple[int, int]],
    output_cells: set[tuple[int, int]],
) -> dict:
    all_cells = input_cells | output_cells
    min_r = min(r for r, _ in all_cells)
    max_r = max(r for r, _ in all_cells)
    min_c = min(c for _, c in all_cells)
    max_c = max(c for _, c in all_cells)

    def shift(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
        return {(r - min_r, c - min_c) for r, c in cells}

    return {
        "color": int(color),
        "input_cells": shift(input_cells),
        "output_cells": shift(output_cells),
        "h": max_r - min_r + 1,
        "w": max_c - min_c + 1,
    }


def _sample_module(color: int) -> dict | None:
    for _ in range(MODULE_ATTEMPTS):
        orientation = random.choice(["h", "v"])
        length = random.randint(MIN_STRIP_LEN, MAX_STRIP_LEN)
        side_counts = _sample_side_counts(orientation, length)
        if side_counts is None:
            continue

        strip = _strip_cells(orientation, length)
        occupied = set(strip)
        arrows = []
        ext_totals = Counter()

        side_order = sorted(side_counts, key=lambda side: side_counts[side], reverse=True)
        ok = True
        for side in side_order:
            count = side_counts[side]
            if count == 0:
                continue
            built = _sample_arrows_for_side(orientation, length, side, count, occupied)
            if built is None:
                ok = False
                break
            for arrow in built:
                arrows.append(arrow)
                occupied |= arrow["cells"]
                ext_totals[side] += int(arrow["shaft_len"])
        if not ok:
            continue

        input_cells = set(strip)
        for arrow in arrows:
            input_cells |= arrow["cells"]

        output_cells = set(strip)
        for side, amount in ext_totals.items():
            dr, dc = SIDE_TO_DELTA[side]
            for k in range(1, amount + 1):
                for r, c in strip:
                    output_cells.add((r + dr * k, c + dc * k))

        module = _normalize_module(color, input_cells, output_cells)
        if module["h"] <= 20 and module["w"] <= 20:
            return module
    return None


def _boxes_conflict(a: tuple[int, int, int, int], b: tuple[int, int, int, int], spacing: int) -> bool:
    ar0, ac0, ah, aw = a
    br0, bc0, bh, bw = b
    ar1, ac1 = ar0 + ah - 1, ac0 + aw - 1
    br1, bc1 = br0 + bh - 1, bc0 + bw - 1
    separated = (
        ar1 + spacing < br0
        or br1 + spacing < ar0
        or ac1 + spacing < bc0
        or bc1 + spacing < ac0
    )
    return not separated


def _place_modules(modules: list[dict], h: int, w: int, spacing: int = 1) -> dict[int, tuple[int, int]] | None:
    order = sorted(range(len(modules)), key=lambda i: modules[i]["h"] * modules[i]["w"], reverse=True)
    placed_boxes: list[tuple[int, int, int, int]] = []
    placements: dict[int, tuple[int, int]] = {}

    for idx in order:
        mod = modules[idx]
        candidates = []
        for r0 in range(h - mod["h"] + 1):
            for c0 in range(w - mod["w"] + 1):
                box = (r0, c0, mod["h"], mod["w"])
                if all(not _boxes_conflict(box, other, spacing) for other in placed_boxes):
                    candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        placements[idx] = (r0, c0)
        placed_boxes.append((r0, c0, mod["h"], mod["w"]))
    return placements


def _compose_scene(modules: list[dict], bg: int) -> tuple[np.ndarray, np.ndarray] | None:
    fg_area = sum(len(module["input_cells"]) for module in modules)
    max_h = max(module["h"] for module in modules)
    max_w = max(module["w"] for module in modules)
    min_area = max(2 * fg_area + 1, 100)

    for _ in range(CANVAS_ATTEMPTS):
        h = random.randint(max(max_h, 10), 30)
        w_lo = max(max_w, (min_area + h - 1) // h)
        if w_lo > 30:
            continue
        w = random.randint(max(w_lo, 10), 30)
        placements = _place_modules(modules, h, w, spacing=1)
        if placements is None:
            continue

        inp = np.full((h, w), int(bg), dtype=int)
        out = np.full_like(inp, int(bg))
        for idx, module in enumerate(modules):
            r0, c0 = placements[idx]
            color = int(module["color"])
            for r, c in module["input_cells"]:
                inp[r0 + r, c0 + c] = color
            for r, c in module["output_cells"]:
                out[r0 + r, c0 + c] = color

        if int(np.count_nonzero(inp == bg)) <= inp.size // 2:
            continue
        return inp, out
    return None


def _gen_attempt() -> np.ndarray | None:
    n_modules = random.randint(1, 3)
    colors = list(range(10))
    bg = random.choice(colors)
    fg_colors = random.sample([color for color in colors if color != bg], n_modules)

    modules = []
    for color in fg_colors:
        module = _sample_module(color)
        if module is None:
            return None
        modules.append(module)

    scene = _compose_scene(modules, bg)
    if scene is None:
        return None
    inp, expected = scene

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
