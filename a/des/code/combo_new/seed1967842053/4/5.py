import numpy as np


GREY = 5


def _most_common_color(grid):
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_for_color(grid, color):
    rows, cols = grid.shape
    seen = np.zeros((rows, cols), dtype=bool)
    comps = []
    for r in range(rows):
        for c in range(cols):
            if seen[r, c] or grid[r, c] != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < rows and 0 <= nc < cols and not seen[nr, nc] and grid[nr, nc] == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(cells)
    return comps


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), max(rs), min(cs), max(cs)


def _is_solid_rect(grid, color, box):
    r0, r1, c0, c1 = box
    return bool(np.all(grid[r0:r1 + 1, c0:c1 + 1] == color))


def _is_frame_rect(grid, color, box):
    r0, r1, c0, c1 = box
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    sub = grid[r0:r1 + 1, c0:c1 + 1] == color
    if not (np.all(sub[0, :]) and np.all(sub[-1, :]) and np.all(sub[:, 0]) and np.all(sub[:, -1])):
        return False
    return not np.any(sub[1:-1, 1:-1])


def main(grid: np.ndarray) -> np.ndarray:
    """Cut each solid color object through the matching same-color mold frame.

    Input structure:
    The background is the most common color.  Each active color appears as one
    one-pixel rectangular frame and as one separate solid rectangle of the same
    color.  The frame interior has the same size as that solid rectangle and is
    filled only with background pixels and grey mask pixels.

    Rule:
    Match each solid object to the rectangular frame with the same color.  Clear
    the original solid object to background.  Inside the frame, replace grey
    mask pixels with the object's color and replace masked background positions
    with background, while preserving the frame border.

    Key insight:
    The same-color pair identifies which object belongs to which mold, and the
    grey pixels inside the mold are the cut-through template for the object.
    """
    g = np.array(grid, copy=True)
    out = np.array(grid, copy=True)
    bg = _most_common_color(g)

    for color in [int(v) for v in np.unique(g) if int(v) not in (bg, GREY)]:
        frames = []
        solids = []
        for cells in _components_for_color(g, color):
            box = _bbox(cells)
            if _is_frame_rect(g, color, box):
                frames.append(box)
            elif _is_solid_rect(g, color, box):
                solids.append((box, cells))

        for frame in frames:
            r0, r1, c0, c1 = frame
            inner_h = r1 - r0 - 1
            inner_w = c1 - c0 - 1
            match = None
            for box, cells in solids:
                sr0, sr1, sc0, sc1 = box
                if sr1 - sr0 + 1 == inner_h and sc1 - sc0 + 1 == inner_w:
                    match = (box, cells)
                    break
            if match is None:
                continue

            for rr, cc in match[1]:
                out[rr, cc] = bg
            mask = g[r0 + 1:r1, c0 + 1:c1]
            out[r0 + 1:r1, c0 + 1:c1] = np.where(mask == GREY, color, bg)

    return out


def _place_rect(occupied, height, width, rng):
    rows, cols = occupied.shape
    candidates = []
    for r in range(rows - height + 1):
        for c in range(cols - width + 1):
            r0 = max(0, r - 1)
            r1 = min(rows, r + height + 1)
            c0 = max(0, c - 1)
            c1 = min(cols, c + width + 1)
            if not np.any(occupied[r0:r1, c0:c1]):
                candidates.append((r, c))
    if not candidates:
        return None
    r, c = candidates[int(rng.integers(len(candidates)))]
    occupied[r:r + height, c:c + width] = True
    return r, c


def _mask_pattern(height, width, rng):
    rr, cc = np.indices((height, width))
    mode = int(rng.integers(6))
    if mode == 0:
        mask = rng.random((height, width)) < rng.uniform(0.25, 0.65)
    elif mode == 1:
        period = int(rng.integers(2, 4))
        mask = (cc + int(rng.integers(period))) % period == 0
    elif mode == 2:
        period = int(rng.integers(2, 4))
        mask = (rr + int(rng.integers(period))) % period == 0
    elif mode == 3:
        period = int(rng.integers(2, 4))
        mask = (rr + cc + int(rng.integers(period))) % period == 0
    elif mode == 4:
        mask = np.zeros((height, width), dtype=bool)
        h2 = int(rng.integers(1, height + 1))
        w2 = int(rng.integers(1, width + 1))
        r = int(rng.integers(height - h2 + 1))
        c = int(rng.integers(width - w2 + 1))
        mask[r:r + h2, c:c + w2] = True
    else:
        mask = (rr == int(rng.integers(height))) | (cc == int(rng.integers(width)))

    if np.all(mask):
        mask[int(rng.integers(height)), int(rng.integers(width))] = False
    if not np.any(mask):
        mask[int(rng.integers(height)), int(rng.integers(width))] = True
    if not np.any(~mask):
        mask[0, 0] = False
    return mask


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    palette = list(range(10))
    for _ in range(1000):
        rows = int(rng.integers(18, 31))
        cols = int(rng.integers(18, 31))
        bg_choices = [c for c in palette if c != GREY]
        bg = int(rng.choice(bg_choices))
        color_choices = [c for c in palette if c not in (bg, GREY)]
        rng.shuffle(color_choices)
        pair_count = int(rng.integers(1, min(5, len(color_choices) + 1)))
        colors = color_choices[:pair_count]

        grid = np.full((rows, cols), bg, dtype=int)
        occupied = np.zeros((rows, cols), dtype=bool)
        ok = True

        specs = []
        for color in colors:
            inner_h = int(rng.integers(2, 7))
            inner_w = int(rng.integers(2, 7))
            specs.append((color, inner_h, inner_w))
        rng.shuffle(specs)

        for color, inner_h, inner_w in specs:
            frame_h = inner_h + 2
            frame_w = inner_w + 2
            frame_pos = _place_rect(occupied, frame_h, frame_w, rng)
            obj_pos = _place_rect(occupied, inner_h, inner_w, rng)
            if frame_pos is None or obj_pos is None:
                ok = False
                break

            fr, fc = frame_pos
            grid[fr:fr + frame_h, fc:fc + frame_w] = color
            grid[fr + 1:fr + frame_h - 1, fc + 1:fc + frame_w - 1] = bg
            mask = _mask_pattern(inner_h, inner_w, rng)
            grid[fr + 1:fr + frame_h - 1, fc + 1:fc + frame_w - 1][mask] = GREY

            orow, ocol = obj_pos
            grid[orow:orow + inner_h, ocol:ocol + inner_w] = color

        if not ok:
            continue
        counts = np.bincount(grid.ravel(), minlength=10)
        if counts[bg] > max(counts[c] for c in palette if c != bg):
            return grid

    return generate_input()

