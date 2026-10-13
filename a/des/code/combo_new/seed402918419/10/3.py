import random
from collections import deque

import numpy as np


BG = 0
FRAME = 3
USABLE_COLORS = [1, 2, 4, 5, 6, 7, 8, 9]
CARDINALS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE_SIDES = ["top", "bottom", "left", "right"]
BOX_SIZES = [5, 7]
COUNT_CHOICES = [1, 2, 3, 4]
GEN_RETRIES = 500


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in CARDINALS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_palettes(grid: np.ndarray) -> tuple[dict[int, int], list[tuple[int, int, int, int]]]:
    h, w = grid.shape
    palettes: list[tuple[int, int, int, int]] = []
    for r0 in range(h - 1):
        for c0 in range(w - 1):
            if not (r0 == 0 or r0 == h - 2 or c0 == 0 or c0 == w - 2):
                continue
            block = grid[r0 : r0 + 2, c0 : c0 + 2]
            left = int(block[0, 0])
            right = int(block[0, 1])
            if left == BG or right == BG or left == right:
                continue
            if not bool(np.all(block[:, 0] == left) and np.all(block[:, 1] == right)):
                continue
            palettes.append((r0, c0, left, right))

    mapping: dict[int, int] = {}
    for _r0, _c0, left, right in palettes:
        mapping[left] = right
        mapping[right] = left
    return mapping, palettes


def _find_containers(grid: np.ndarray) -> list[dict[str, int | tuple[int, int]]]:
    containers: list[dict[str, int | tuple[int, int]]] = []
    for comp in _components_of_color(grid, FRAME):
        rows = [r for r, _c in comp]
        cols = [c for _r, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        box_h = r1 - r0 + 1
        box_w = c1 - c0 + 1
        if box_h not in BOX_SIZES or box_w not in BOX_SIZES:
            continue
        if len(comp) != 2 * box_h + 2 * box_w - 4:
            continue

        ok = True
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                on_border = r in (r0, r1) or c in (c0, c1)
                val = int(grid[r, c])
                if on_border and val != FRAME:
                    ok = False
                if not on_border and val == FRAME:
                    ok = False
        if not ok:
            continue

        cr = r0 + box_h // 2
        cc = c0 + box_w // 2
        probe = int(grid[cr, cc])
        if probe in (BG, FRAME):
            continue

        direction = None
        for dr, dc in CARDINALS:
            nr, nc = cr + dr, cc + dc
            if int(grid[nr, nc]) == probe:
                if direction is not None:
                    direction = None
                    break
                direction = (dr, dc)
        if direction is None:
            continue

        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        vals = {int(v) for v in np.unique(interior)}
        if not vals.issubset({BG, probe}):
            continue
        if int(np.count_nonzero(interior == probe)) != 2:
            continue

        containers.append(
            {
                "r0": r0,
                "c0": c0,
                "h": box_h,
                "w": box_w,
                "cr": cr,
                "cc": cc,
                "probe": probe,
                "dr": direction[0],
                "dc": direction[1],
            }
        )

    containers.sort(key=lambda item: (int(item["r0"]), int(item["c0"])))
    return containers


def _fill_container_half(out: np.ndarray, info: dict[str, int | tuple[int, int]], fill_color: int) -> None:
    r0 = int(info["r0"])
    c0 = int(info["c0"])
    box_h = int(info["h"])
    box_w = int(info["w"])
    cr = int(info["cr"])
    cc = int(info["cc"])
    dr = int(info["dr"])
    dc = int(info["dc"])

    for r in range(r0 + 1, r0 + box_h - 1):
        for c in range(c0 + 1, c0 + box_w - 1):
            if r == cr and c == cc:
                continue
            if dr == -1 and r <= cr:
                out[r, c] = fill_color
            elif dr == 1 and r >= cr:
                out[r, c] = fill_color
            elif dc == -1 and c <= cc:
                out[r, c] = fill_color
            elif dc == 1 and c >= cc:
                out[r, c] = fill_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the pointed half of each green container using the partner color from a border palette.

    Input structure:
    - Background is black.
    - Several hollow green rectangles sit away from the grid edge, each with an interior center probe.
    - Every probe is a 2-pixel indicator: a colored center pixel plus one same-colored neighbor pointing up, down, left, or right.
    - Along one grid edge there are 2x2 palette blocks whose left column is one color and right column is another color.

    Rule:
    - For each container, match its probe color to one of the colors appearing in a palette block.
    - The other color in that same palette block is the fill color for the container.
    - Use the probe direction to choose the corresponding interior half of the container and fill that half with the fill color.
    - Keep the green border, palette blocks, and the center probe pixel unchanged.

    Key insight:
    - The probe color is a lookup key into an external two-color palette, while the probe direction selects which half of the hollow rectangle receives the looked-up fill color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    palette_map, _palettes = _find_palettes(g)
    containers = _find_containers(g)

    out = g.copy()
    for info in containers:
        probe = int(info["probe"])
        assert probe in palette_map
        _fill_container_half(out, info, int(palette_map[probe]))
    return out


def _choose_spaced_starts(limit: int, count: int) -> list[int] | None:
    starts = list(range(limit - 1))
    for _ in range(200):
        random.shuffle(starts)
        chosen: list[int] = []
        for start in starts:
            if all(abs(start - prev) >= 3 for prev in chosen):
                chosen.append(start)
                if len(chosen) == count:
                    return sorted(chosen)
    return None


def _draw_palette(grid: np.ndarray, side: str, start: int, left: int, right: int) -> tuple[int, int]:
    if side == "top":
        r0, c0 = 0, start
    elif side == "bottom":
        r0, c0 = grid.shape[0] - 2, start
    elif side == "left":
        r0, c0 = start, 0
    else:
        r0, c0 = start, grid.shape[1] - 2

    grid[r0 : r0 + 2, c0] = left
    grid[r0 : r0 + 2, c0 + 1] = right
    return r0, c0


def _bbox_is_free(occupied: np.ndarray, r0: int, c0: int, box_h: int, box_w: int, pad: int = 1) -> bool:
    h, w = occupied.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h, r0 + box_h + pad)
    cc1 = min(w, c0 + box_w + pad)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _mark_bbox(occupied: np.ndarray, r0: int, c0: int, box_h: int, box_w: int) -> None:
    occupied[r0 : r0 + box_h, c0 : c0 + box_w] = True


def _render_container(grid: np.ndarray, spec: dict[str, int | tuple[int, int]]) -> None:
    r0 = int(spec["r0"])
    c0 = int(spec["c0"])
    box_h = int(spec["h"])
    box_w = int(spec["w"])
    probe = int(spec["probe"])
    dr = int(spec["dr"])
    dc = int(spec["dc"])

    grid[r0, c0 : c0 + box_w] = FRAME
    grid[r0 + box_h - 1, c0 : c0 + box_w] = FRAME
    grid[r0 : r0 + box_h, c0] = FRAME
    grid[r0 : r0 + box_h, c0 + box_w - 1] = FRAME

    cr = r0 + box_h // 2
    cc = c0 + box_w // 2
    grid[cr, cc] = probe
    grid[cr + dr, cc + dc] = probe


def _expected_from_specs(inp: np.ndarray, specs: list[dict[str, int | tuple[int, int]]]) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        _fill_container_half(out, spec, int(spec["fill"]))
    return out


def _grid_bounds_for_side(h: int, w: int, side: str, box_h: int, box_w: int) -> tuple[int, int, int, int] | None:
    min_r = 2
    max_r = h - box_h - 2
    min_c = 2
    max_c = w - box_w - 2

    if side == "top":
        min_r = 4
    elif side == "bottom":
        max_r = h - box_h - 4
    elif side == "left":
        min_c = 4
    else:
        max_c = w - box_w - 4

    if min_r > max_r or min_c > max_c:
        return None
    return min_r, max_r, min_c, max_c


def _place_containers(h: int, w: int, side: str, base_specs: list[dict[str, int]]) -> list[dict[str, int]] | None:
    occupied = np.zeros((h, w), dtype=bool)
    specs: list[dict[str, int] | None] = [None] * len(base_specs)
    order = sorted(range(len(base_specs)), key=lambda idx: base_specs[idx]["h"] * base_specs[idx]["w"], reverse=True)

    for idx in order:
        base = base_specs[idx]
        bounds = _grid_bounds_for_side(h, w, side, base["h"], base["w"])
        if bounds is None:
            return None
        min_r, max_r, min_c, max_c = bounds
        candidates = [
            (r0, c0)
            for r0 in range(min_r, max_r + 1)
            for c0 in range(min_c, max_c + 1)
            if _bbox_is_free(occupied, r0, c0, base["h"], base["w"], pad=1)
        ]
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        placed = dict(base)
        placed["r0"] = r0
        placed["c0"] = c0
        placed["cr"] = r0 + base["h"] // 2
        placed["cc"] = c0 + base["w"] // 2
        specs[idx] = placed
        _mark_bbox(occupied, r0, c0, base["h"], base["w"])

    return [spec for spec in specs if spec is not None]


def _gen_attempt() -> np.ndarray | None:
    count = random.choice(COUNT_CHOICES)
    min_side = {1: 15, 2: 16, 3: 18, 4: 20}[count]
    h = random.randint(min_side, 30)
    w = random.randint(min_side, 30)
    side = random.choice(PALETTE_SIDES)

    palette_starts = _choose_spaced_starts(w if side in ("top", "bottom") else h, count)
    if palette_starts is None:
        return None

    color_pool = random.sample(USABLE_COLORS, 2 * count)
    direction_pool = random.sample(CARDINALS, count)
    base_specs: list[dict[str, int]] = []
    palette_specs: list[tuple[int, int, int]] = []

    for idx in range(count):
        color_a = color_pool[2 * idx]
        color_b = color_pool[2 * idx + 1]
        if random.random() < 0.5:
            probe, fill = color_a, color_b
        else:
            probe, fill = color_b, color_a
        left, right = (probe, fill) if random.random() < 0.5 else (fill, probe)
        dr, dc = direction_pool[idx]
        base_specs.append(
            {
                "h": random.choice(BOX_SIZES),
                "w": random.choice(BOX_SIZES),
                "probe": probe,
                "fill": fill,
                "dr": dr,
                "dc": dc,
            }
        )
        palette_specs.append((palette_starts[idx], left, right))

    specs = _place_containers(h, w, side, base_specs)
    if specs is None:
        return None

    inp = np.zeros((h, w), dtype=int)
    for start, left, right in palette_specs:
        _draw_palette(inp, side, start, left, right)
    for spec in specs:
        _render_container(inp, spec)

    expected = _expected_from_specs(inp, specs)
    palette_map, palettes = _find_palettes(inp)
    containers = _find_containers(inp)
    if len(palettes) != count or len(containers) != count:
        return None
    if any(int(spec["probe"]) not in palette_map for spec in specs):
        return None
    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
