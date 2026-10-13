import random

import numpy as np


BG = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
CORNERS = ("tl", "tr", "bl", "br")
MAX_RETRIES = 500


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _expanded_rect(r0: int, c0: int, r1: int, c1: int, h: int, w: int, gap: int) -> tuple[int, int, int, int]:
    return max(0, r0 - gap), max(0, c0 - gap), min(h - 1, r1 + gap), min(w - 1, c1 + gap)


def _rects_touch(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0)


def _stamp_origin(tip_r: int, tip_c: int, shape: tuple[int, int], corner: str) -> tuple[int, int]:
    h, w = map(int, shape)
    if corner == "tl":
        return int(tip_r - h + 1), int(tip_c - w + 1)
    if corner == "tr":
        return int(tip_r - h + 1), int(tip_c)
    if corner == "bl":
        return int(tip_r), int(tip_c - w + 1)
    assert corner == "br"
    return int(tip_r), int(tip_c)


def _find_legends(g: np.ndarray, bg: int) -> tuple[list[dict], np.ndarray]:
    h, w = g.shape
    mask = (g != bg) & (g != GREY)
    candidates: list[dict] = []

    for size in (3, 2):
        for r0 in range(1, h - size + 1):
            for c0 in range(w - size + 1):
                if not bool(mask[r0 : r0 + size, c0 : c0 + size].all()):
                    continue
                label_cells = [(r0 - 1, c) for c in range(c0, c0 + size) if mask[r0 - 1, c]]
                if len(label_cells) != 1:
                    continue

                rr0, cc0, rr1, cc1 = _expanded_rect(r0 - 1, c0, r0 + size - 1, c0 + size - 1, h, w, gap=1)
                allowed = {(label_cells[0][0], label_cells[0][1])}
                for rr in range(r0, r0 + size):
                    for cc in range(c0, c0 + size):
                        allowed.add((rr, cc))

                ok = True
                for rr in range(rr0, rr1 + 1):
                    for cc in range(cc0, cc1 + 1):
                        if (rr, cc) in allowed:
                            continue
                        if int(g[rr, cc]) != bg:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue

                center_r = r0 + (size - 1) / 2.0
                center_c = c0 + (size - 1) / 2.0
                corner = ("t" if center_r < (h - 1) / 2.0 else "b") + ("l" if center_c < (w - 1) / 2.0 else "r")
                label_r, label_c = label_cells[0]
                candidates.append(
                    {
                        "corner": corner,
                        "bbox": (int(r0), int(c0), int(r0 + size - 1), int(c0 + size - 1)),
                        "pattern": g[r0 : r0 + size, c0 : c0 + size].copy(),
                        "label": (int(label_r), int(label_c)),
                        "label_color": int(g[label_r, label_c]),
                    }
                )

    by_corner: dict[str, dict] = {}
    for cand in candidates:
        corner = cand["corner"]
        cur = by_corner.get(corner)
        if cur is None or cand["pattern"].shape[0] > cur["pattern"].shape[0]:
            by_corner[corner] = cand

    legends = [by_corner[key] for key in sorted(by_corner)]
    assert legends, "expected at least one corner legend"
    label_colors = [item["label_color"] for item in legends]
    assert len(label_colors) == len(set(label_colors)), "legend label colors must be unique"

    legend_mask = np.zeros_like(g, dtype=bool)
    for legend in legends:
        r0, c0, r1, c1 = legend["bbox"]
        legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True
        lr, lc = legend["label"]
        legend_mask[lr, lc] = True

    return legends, legend_mask


def _find_arrows(g: np.ndarray, bg: int, blocked: np.ndarray) -> list[dict]:
    mask = (g != bg) & (~blocked)
    arrows: list[dict] = []
    for comp in _components(mask):
        colors = {int(g[r, c]) for r, c in comp}
        if GREY not in colors:
            continue
        tip_cells = [(r, c) for r, c in comp if int(g[r, c]) != GREY]
        shaft_cells = [(r, c) for r, c in comp if int(g[r, c]) == GREY]
        if len(tip_cells) != 1 or len(shaft_cells) not in (1, 2, 3):
            continue

        rows = sorted({r for r, _c in comp})
        cols = sorted({c for _r, c in comp})
        tip_r, tip_c = tip_cells[0]
        direction: str | None = None
        if len(rows) == 1:
            row = rows[0]
            span = list(range(min(cols), max(cols) + 1))
            if cols != span:
                continue
            if tip_c == min(cols):
                direction = "left"
            elif tip_c == max(cols):
                direction = "right"
            else:
                continue
            if any(r != row for r, _c in shaft_cells):
                continue
        elif len(cols) == 1:
            col = cols[0]
            span = list(range(min(rows), max(rows) + 1))
            if rows != span:
                continue
            if tip_r == min(rows):
                direction = "up"
            elif tip_r == max(rows):
                direction = "down"
            else:
                continue
            if any(c != col for _r, c in shaft_cells):
                continue
        else:
            continue

        arrows.append(
            {
                "tip": (int(tip_r), int(tip_c)),
                "tip_color": int(g[tip_r, tip_c]),
                "direction": direction,
                "scale": int(len(shaft_cells)),
            }
        )

    arrows.sort(key=lambda item: (item["tip"][0], item["tip"][1]))
    assert arrows, "expected at least one arrow"
    return arrows


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read each grey arrow as a color-and-scale instruction, then stamp the matching corner pattern at the arrow tip.

    Input structure:
    - The background is black.
    - Several small full 2x2 or 3x3 multi-color patterns sit near the grid corners, and each has one single-pixel label directly one row above the pattern.
    - The middle area contains thin horizontal or vertical arrows made from 1 to 3 grey cells plus a single colored tip cell at one end.

    Rule:
    - Match an arrow to the corner pattern whose label color equals the arrow-tip color.
    - The number of grey shaft cells is the integer scale factor; enlarge the selected pattern by replacing each source cell with an N x N block.
    - Copy the enlarged pattern onto the original canvas, keeping the legend orientation, and align the legend-facing inner corner of that copy to the arrow tip.
    - Existing arrow cells may be covered by the pasted pattern; untouched cells remain unchanged.

    Key insight:
    - The arrow encodes two independent instructions at once: tip color says which corner legend to use, while grey length says how large the copied pattern must become.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    legends, legend_mask = _find_legends(g, bg)
    legend_by_color = {legend["label_color"]: legend for legend in legends}
    arrows = _find_arrows(g, bg, legend_mask)

    out = g.copy()
    for arrow in arrows:
        assert arrow["tip_color"] in legend_by_color, "arrow tip color must match a legend label"
        legend = legend_by_color[arrow["tip_color"]]
        scale = arrow["scale"]
        pattern = legend["pattern"]
        scaled = np.kron(pattern, np.ones((scale, scale), dtype=int))
        tip_r, tip_c = arrow["tip"]
        r0, c0 = _stamp_origin(tip_r, tip_c, scaled.shape, legend["corner"])
        r1 = r0 + scaled.shape[0]
        c1 = c0 + scaled.shape[1]
        assert 0 <= r0 and 0 <= c0 and r1 <= g.shape[0] and c1 <= g.shape[1], "scaled stamp must fit"
        out[r0:r1, c0:c1] = scaled

    return out


def _random_pattern(rng: random.Random, size: int, label_color: int, extra_palette: list[int]) -> np.ndarray:
    num_colors = rng.randint(2, min(4, 1 + len(extra_palette)))
    others = rng.sample(extra_palette, num_colors - 1)
    palette = [label_color] + others
    cells = size * size
    values = palette.copy()
    while len(values) < cells:
        values.append(rng.choice(palette))
    rng.shuffle(values)
    return np.array(values, dtype=int).reshape(size, size)


def _corner_position(rng: random.Random, h: int, w: int, size: int, corner: str) -> tuple[int, int]:
    row_margin = rng.randint(1, 2)
    col_margin = rng.randint(1, 2)
    if corner[0] == "t":
        r0 = row_margin
    else:
        r0 = h - size - row_margin
    if corner[1] == "l":
        c0 = col_margin
    else:
        c0 = w - size - col_margin
    return int(r0), int(c0)


def _stamp_bbox(tip: tuple[int, int], shape: tuple[int, int], corner: str) -> tuple[int, int, int, int]:
    r0, c0 = _stamp_origin(tip[0], tip[1], shape, corner)
    h, w = map(int, shape)
    return int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)


def _paint_legend(grid: np.ndarray, legend: dict) -> None:
    r0, c0 = legend["pos"]
    size = legend["size"]
    grid[r0 : r0 + size, c0 : c0 + size] = legend["pattern"]
    grid[r0 - 1, c0 + legend["label_offset"]] = legend["label_color"]


def _sample_legends(rng: random.Random, h: int, w: int) -> list[dict]:
    num_legends = rng.randint(2, 4)
    chosen_corners = rng.sample(list(CORNERS), num_legends)
    label_colors = rng.sample(PALETTE, num_legends)
    extra_palette = [color for color in PALETTE if color not in label_colors]

    legends: list[dict] = []
    for corner, label_color in zip(chosen_corners, label_colors):
        size = rng.choice([2, 3])
        r0, c0 = _corner_position(rng, h, w, size, corner)
        pattern = _random_pattern(rng, size, label_color, extra_palette)
        legends.append(
            {
                "corner": corner,
                "label_color": int(label_color),
                "size": int(size),
                "pos": (int(r0), int(c0)),
                "pattern": pattern,
                "label_offset": int(rng.randrange(size)),
            }
        )
    return legends


def _central_tip_bounds(h: int, w: int) -> tuple[tuple[int, int], tuple[int, int]]:
    row_lo = max(4, h // 4)
    row_hi = min(h - 5, h - h // 4 - 1)
    col_lo = max(4, w // 4)
    col_hi = min(w - 5, w - w // 4 - 1)
    return (int(row_lo), int(max(row_lo, row_hi))), (int(col_lo), int(max(col_lo, col_hi)))


def _choose_arrow_specs(rng: random.Random, legends: list[dict]) -> list[dict]:
    num_arrows = rng.randint(2, 4)
    lengths = [1, 2, 3]
    rng.shuffle(lengths)
    directions = list(DIRS)
    rng.shuffle(directions)
    specs: list[dict] = []
    for idx in range(num_arrows):
        legend = legends[idx] if idx < len(legends) and rng.random() < 0.65 else rng.choice(legends)
        specs.append(
            {
                "legend": legend,
                "scale": int(lengths[idx % len(lengths)] if idx < 3 else rng.choice([1, 2, 3])),
                "direction": directions[idx % len(directions)] if idx < 4 else rng.choice(list(DIRS)),
            }
        )
    return specs


def _place_arrows(rng: random.Random, grid: np.ndarray, legends: list[dict], specs: list[dict]) -> bool:
    h, w = grid.shape
    occupied_input = grid != BG
    reserved_output: list[tuple[int, int, int, int]] = []
    for legend in legends:
        r0, c0 = legend["pos"]
        size = legend["size"]
        lr = r0 - 1
        lc = c0 + legend["label_offset"]
        reserved_output.append((lr, lc, lr, lc))
        reserved_output.append((r0, c0, r0 + size - 1, c0 + size - 1))

    row_bounds, col_bounds = _central_tip_bounds(h, w)

    for spec in specs:
        legend = spec["legend"]
        scale = spec["scale"]
        direction = spec["direction"]
        scaled_shape = (legend["size"] * scale, legend["size"] * scale)
        dr, dc = DIRS[direction]

        placed = False
        for _ in range(300):
            tip_r = rng.randint(*row_bounds)
            tip_c = rng.randint(*col_bounds)
            stamp = _stamp_bbox((tip_r, tip_c), scaled_shape, legend["corner"])
            sr0, sc0, sr1, sc1 = stamp
            if sr0 < 0 or sc0 < 0 or sr1 >= h or sc1 >= w:
                continue
            if any(_rects_touch(stamp, other, gap=1) for other in reserved_output):
                continue

            cells = [(tip_r, tip_c)]
            ok = True
            for step in range(1, scale + 1):
                rr = tip_r - dr * step
                cc = tip_c - dc * step
                if not (0 <= rr < h and 0 <= cc < w):
                    ok = False
                    break
                cells.append((rr, cc))
            if not ok:
                continue

            for rr, cc in cells:
                for dr2, dc2 in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr2
                    nc = cc + dc2
                    if 0 <= nr < h and 0 <= nc < w and occupied_input[nr, nc]:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue

            rr0, cc0, rr1, cc1 = _expanded_rect(sr0, sc0, sr1, sc1, h, w, gap=0)
            shaft_has_space = 0 <= rr0 <= rr1 < h and 0 <= cc0 <= cc1 < w
            if not shaft_has_space:
                continue

            for step, (rr, cc) in enumerate(cells):
                grid[rr, cc] = legend["label_color"] if step == 0 else GREY
                occupied_input[rr, cc] = True

            reserved_output.append(stamp)
            placed = True
            break

        if not placed:
            return False

    return True


def _valid_scene(inp: np.ndarray) -> bool:
    try:
        out = main(inp)
    except Exception:
        return False
    if inp.ndim != 2 or out.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        rng = random.Random()
        h = rng.randint(20, 29)
        w = rng.randint(20, 29)
        grid = np.full((h, w), BG, dtype=int)

        legends = _sample_legends(rng, h, w)
        for legend in legends:
            _paint_legend(grid, legend)

        specs = _choose_arrow_specs(rng, legends)
        if not _place_arrows(rng, grid, legends, specs):
            continue
        if _valid_scene(grid):
            return grid

    raise RuntimeError("failed to generate valid arrow-scale legend scene")
