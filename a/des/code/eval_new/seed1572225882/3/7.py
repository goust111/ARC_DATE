import random

import numpy as np


BG = 0
MARK = 4


def _same_color_components(grid, color, seen):
    rows, cols = grid.shape
    comps = []
    for r in range(rows):
        for c in range(cols):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < rows and 0 <= nc < cols and not seen[nr, nc]:
                        if int(grid[nr, nc]) == color:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
            comps.append(cells)
    return comps


def _find_borders(grid):
    rects = []
    seen = np.zeros(grid.shape, dtype=bool)
    for raw in np.unique(grid):
        color = int(raw)
        if color in (BG, MARK):
            continue
        for cells in _same_color_components(grid, color, seen):
            rs = [p[0] for p in cells]
            cs = [p[1] for p in cells]
            r0, r1 = min(rs), max(rs)
            c0, c1 = min(cs), max(cs)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 3 or w < 3:
                continue
            perimeter = 2 * h + 2 * w - 4
            if len(cells) != perimeter:
                continue
            ok = True
            for rr, cc in cells:
                if rr not in (r0, r1) and cc not in (c0, c1):
                    ok = False
                    break
            if ok:
                rects.append({"color": color, "r0": r0, "r1": r1, "c0": c0, "c1": c1})
    return rects


def main(grid: np.ndarray) -> np.ndarray:
    """Match each stamp to its same-colored envelope and print its 2x enlarged interior pattern.

    Input structure:
    The background is black.  Objects are rectangular colored borders.  Small borders are stamps
    with a non-black interior drawing, while larger matching borders are envelopes whose interior
    contains one yellow single-pixel anchor.

    Rule:
    Detect rectangular borders, split them by whether their interior contains a yellow anchor, and
    pair each envelope with the stamp that has the same border color.  Remove every stamp and every
    yellow anchor.  Copy the stamp interior pattern, enlarge it by replacing each source cell with a
    2x2 block, and place only its non-black cells into the matching envelope with the anchor as the
    enlarged pattern's top-left corner.

    Key insight:
    The border color gives the pairing, the stamp interior gives the motif, and the yellow pixel gives
    the placement anchor after a fixed 2x scale-up.
    """
    grid = np.asarray(grid)
    out = grid.copy()
    stamps = {}
    envelopes = {}

    for rect in _find_borders(grid):
        r0, r1, c0, c1 = rect["r0"], rect["r1"], rect["c0"], rect["c1"]
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        markers = np.argwhere(interior == MARK)
        color = rect["color"]
        if len(markers):
            marker = (r0 + 1 + int(markers[0, 0]), c0 + 1 + int(markers[0, 1]))
            envelopes[color] = {**rect, "marker": marker}
        elif np.any(interior != BG):
            stamps[color] = {**rect, "pattern": interior.copy()}

    for stamp in stamps.values():
        out[stamp["r0"] : stamp["r1"] + 1, stamp["c0"] : stamp["c1"] + 1] = BG

    for env in envelopes.values():
        r0, r1, c0, c1 = env["r0"], env["r1"], env["c0"], env["c1"]
        inside = out[r0 + 1 : r1, c0 + 1 : c1]
        inside[inside == MARK] = BG

    rows, cols = out.shape
    for color, env in envelopes.items():
        stamp = stamps.get(color)
        if stamp is None:
            continue
        enlarged = np.repeat(np.repeat(stamp["pattern"], 2, axis=0), 2, axis=1)
        ar, ac = env["marker"]
        for rr in range(enlarged.shape[0]):
            for cc in range(enlarged.shape[1]):
                val = int(enlarged[rr, cc])
                if val == BG:
                    continue
                tr, tc = ar + rr, ac + cc
                if 0 <= tr < rows and 0 <= tc < cols:
                    out[tr, tc] = val
    return out


def _draw_border(grid, r, c, h, w, color):
    grid[r, c : c + w] = color
    grid[r + h - 1, c : c + w] = color
    grid[r : r + h, c] = color
    grid[r : r + h, c + w - 1] = color


def _place(used, h, w, gap=1):
    rows, cols = used.shape
    if h > rows or w > cols:
        return None
    for _ in range(900):
        r = random.randint(0, rows - h)
        c = random.randint(0, cols - w)
        r0 = max(0, r - gap)
        c0 = max(0, c - gap)
        r1 = min(rows, r + h + gap)
        c1 = min(cols, c + w + gap)
        if not used[r0:r1, c0:c1].any():
            used[r : r + h, c : c + w] = True
            return r, c
    return None


def _asymmetric(mask):
    if mask.size <= 2:
        return True
    if mask.shape[0] > 1 and np.array_equal(mask, mask[::-1, :]):
        return False
    if mask.shape[1] > 1 and np.array_equal(mask, mask[:, ::-1]):
        return False
    if mask.shape[0] > 1 and mask.shape[1] > 1 and np.array_equal(mask, mask[::-1, ::-1]):
        return False
    return True


def _pattern(h, w, color):
    cells = h * w
    for _ in range(80):
        mask = np.zeros((h, w), dtype=bool)
        target = random.randint(1, max(1, cells - 1))
        spots = random.sample(range(cells), target)
        for pos in spots:
            mask[pos // w, pos % w] = True
        if _asymmetric(mask):
            pat = np.zeros((h, w), dtype=int)
            pat[mask] = color
            return pat
    pat = np.zeros((h, w), dtype=int)
    pat[0, 0] = color
    if h > 1 and w > 1:
        pat[h - 1, 0] = color
    return pat


def generate_input() -> np.ndarray:
    colors = [1, 2, 3, 5, 6, 7, 8, 9]
    for _ in range(300):
        count = random.choice([1, 2, 3])
        min_side = 14 + 4 * count
        rows = random.randint(min_side, 30)
        cols = random.randint(min_side, 30)
        grid = np.zeros((rows, cols), dtype=int)
        used = np.zeros((rows, cols), dtype=bool)

        border_colors = random.sample(colors, count)
        pattern_colors = [c for c in colors if c not in border_colors]
        random.shuffle(pattern_colors)
        specs = []
        objects = []
        for i, border in enumerate(border_colors):
            sh = random.randint(3, 5)
            sw = random.randint(3, 5)
            ph, pw = sh - 2, sw - 2
            eh = random.randint(max(8, 2 * ph + 2), 12)
            ew = random.randint(max(8, 2 * pw + 2), 12)
            pat = _pattern(ph, pw, pattern_colors[i % len(pattern_colors)])
            spec = {"border": border, "stamp_h": sh, "stamp_w": sw, "env_h": eh, "env_w": ew, "pattern": pat}
            specs.append(spec)
            objects.append(("env", i, eh, ew))
            objects.append(("stamp", i, sh, sw))

        random.shuffle(objects)
        objects.sort(key=lambda obj: obj[2] * obj[3], reverse=True)
        placed = {}
        failed = False
        for kind, idx, h, w in objects:
            loc = _place(used, h, w, gap=1)
            if loc is None:
                failed = True
                break
            placed[(kind, idx)] = loc
        if failed:
            continue

        for i, spec in enumerate(specs):
            r, c = placed[("stamp", i)]
            _draw_border(grid, r, c, spec["stamp_h"], spec["stamp_w"], spec["border"])
            grid[r + 1 : r + spec["stamp_h"] - 1, c + 1 : c + spec["stamp_w"] - 1] = spec["pattern"]

            r, c = placed[("env", i)]
            _draw_border(grid, r, c, spec["env_h"], spec["env_w"], spec["border"])
            scaled_h = spec["pattern"].shape[0] * 2
            scaled_w = spec["pattern"].shape[1] * 2
            ar_min, ac_min = r + 1, c + 1
            ar_max = r + spec["env_h"] - 1 - scaled_h
            ac_max = c + spec["env_w"] - 1 - scaled_w
            grid[random.randint(ar_min, ar_max), random.randint(ac_min, ac_max)] = MARK
        return grid

    return generate_input()
