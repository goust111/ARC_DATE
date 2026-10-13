import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
ARC_COLORS = tuple(range(10))
SEED_COLORS = tuple(c for c in ARC_COLORS if c not in (BLACK, GREY))


def _neighbors4(r, c):
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(pts):
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    return min(rs), min(cs), max(rs), max(cs)


def _rect_border_is_grey(g, r0, c0, r1, c1):
    if r1 - r0 < 4 or c1 - c0 < 4:
        return False
    return bool(
        np.all(g[r0, c0 : c1 + 1] == GREY)
        and np.all(g[r1, c0 : c1 + 1] == GREY)
        and np.all(g[r0 : r1 + 1, c0] == GREY)
        and np.all(g[r0 : r1 + 1, c1] == GREY)
    )


def _find_legend_bbox(g):
    candidates = []
    for comp in _components(g == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        if r0 > 2 or c0 > 2:
            continue
        if not _rect_border_is_grey(g, r0, c0, r1, c1):
            continue
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        if inner_w < 3 or inner_h < 3 or (inner_h + 1) % 4 != 0:
            continue
        candidates.append((r0 + c0, r0, c0, r1, c1))
    assert candidates, "a grey-bordered legend must sit in the upper-left corner"
    _, r0, c0, r1, c1 = min(candidates)
    return r0, c0, r1, c1


def _legend_patterns(g, bbox):
    r0, c0, r1, c1 = bbox
    inner_h = r1 - r0 - 1
    n = (inner_h + 1) // 4
    mapping = {}
    for i in range(n):
        pr = r0 + 1 + 4 * i
        pc = c0 + 1
        patch = g[pr : pr + 3, pc : pc + 3].copy()
        vals = set(map(int, patch.reshape(-1)))
        colors = vals - {BLACK, GREY}
        assert len(colors) == 1, "each legend sample must have one foreground color"
        color = next(iter(colors))
        assert vals <= {BLACK, color}, "legend samples may contain only black and their foreground color"
        mapping[int(color)] = patch
        if i + 1 < n:
            sep = g[pr + 3, pc : pc + 3]
            assert np.all(sep == BLACK), "legend samples must be separated by black rows"
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill grey-fenced regions with seed-anchored textures read from a corner legend.

    Input structure:
    - Black is the background and grey pixels form one-pixel closed fences plus a grey-bordered legend in the upper-left corner.
    - The legend contains vertically stacked 3x3 texture samples, separated by black rows; each sample uses black and one seed color.
    - Every non-legend grey fence encloses one connected black interior containing exactly one non-black, non-grey seed pixel.

    Rule:
    - Parse the upper-left legend as a mapping from each sample's foreground color to its full 3x3 black/foreground texture.
    - For each closed non-grey component inside a grey fence, read the unique seed color and look up that color's texture.
    - Tile the 3x3 texture over the enclosed black cells so the seed position corresponds to the texture center cell.
    - Do not overwrite grey fence pixels or the seed pixel itself; black texture cells remain black.

    Key insight:
    - The corner legend supplies the color-to-texture lookup, while the seed position fixes the phase of the repeated texture inside its own enclosure.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    legend_bbox = _find_legend_bbox(g)
    patterns = _legend_patterns(g, legend_bbox)

    out = g.copy()
    lr0, lc0, lr1, lc1 = legend_bbox

    for comp in _components(g != GREY):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        if any(lr0 <= r <= lr1 and lc0 <= c <= lc1 for r, c in comp):
            continue

        seeds = [(r, c, int(g[r, c])) for r, c in comp if int(g[r, c]) not in (BLACK, GREY)]
        if len(seeds) != 1:
            continue
        sr, sc, seed_color = seeds[0]
        assert seed_color in patterns, "every seed color must appear in the legend"
        pattern = patterns[seed_color]

        for r, c in comp:
            if int(g[r, c]) != BLACK:
                continue
            pr = (r - sr + 1) % 3
            pc = (c - sc + 1) % 3
            out[r, c] = int(pattern[pr, pc])

    return out


def _pattern_masks():
    masks = []
    for bits in range(1 << 9):
        if not (bits & (1 << 4)):
            continue
        count = int(bits.bit_count())
        if count < 2 or count > 6:
            continue
        flat = np.array([(bits >> k) & 1 for k in range(9)], dtype=bool)
        mask = flat.reshape(3, 3)
        if not (mask.any(axis=0).all() and mask.any(axis=1).all()):
            continue
        masks.append(mask)
    return masks


def _is_connected(mask):
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    start = tuple(map(int, pts[0]))
    seen = {start}
    q = deque([start])
    h, w = mask.shape
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(pts)


def _make_interior_mask():
    ih = random.randint(4, 8)
    iw = random.randint(4, 8)
    mask = np.ones((ih, iw), dtype=bool)
    edge = [
        (r, c)
        for r in range(ih)
        for c in range(iw)
        if r in (0, ih - 1) or c in (0, iw - 1)
    ]
    random.shuffle(edge)
    target_removals = random.randint(1, min(7, len(edge) // 3))
    removed = 0
    for r, c in edge:
        if removed >= target_removals:
            break
        trial = mask.copy()
        trial[r, c] = False
        if int(trial.sum()) < 8:
            continue
        if not (trial[0].any() and trial[-1].any() and trial[:, 0].any() and trial[:, -1].any()):
            continue
        if _is_connected(trial):
            mask = trial
            removed += 1
    return mask


def _ring_from_interior(inner):
    ph, pw = inner.shape[0] + 2, inner.shape[1] + 2
    interior = np.zeros((ph, pw), dtype=bool)
    interior[1:-1, 1:-1] = inner
    dilated = np.zeros_like(interior)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            src_r0 = max(0, -dr)
            src_r1 = min(ph, ph - dr)
            src_c0 = max(0, -dc)
            src_c1 = min(pw, pw - dc)
            dst_r0 = max(0, dr)
            dst_r1 = min(ph, ph + dr)
            dst_c0 = max(0, dc)
            dst_c1 = min(pw, pw + dc)
            dilated[dst_r0:dst_r1, dst_c0:dst_c1] |= interior[src_r0:src_r1, src_c0:src_c1]
    ring = dilated & ~interior
    return interior, ring


def _make_region_patch():
    for _ in range(60):
        inner_mask = _make_interior_mask()
        interior, ring = _ring_from_interior(inner_mask)
        if _is_connected(interior) and _is_connected(ring):
            return interior, ring
    inner_mask = np.ones((5, 5), dtype=bool)
    return _ring_from_interior(inner_mask)


def _draw_legend(grid, colors, masks):
    legend_h = 4 * len(colors) + 1
    legend_w = 5
    grid[:legend_h, :legend_w] = BLACK
    grid[0, :legend_w] = GREY
    grid[legend_h - 1, :legend_w] = GREY
    grid[:legend_h, 0] = GREY
    grid[:legend_h, legend_w - 1] = GREY
    for i, (color, mask) in enumerate(zip(colors, masks)):
        r = 1 + 4 * i
        sample = np.where(mask, int(color), BLACK)
        grid[r : r + 3, 1:4] = sample
    return legend_h, legend_w


def _can_place(occupied, r0, c0, ph, pw):
    h, w = occupied.shape
    if r0 < 1 or c0 < 1 or r0 + ph >= h or c0 + pw >= w:
        return False
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h, r0 + ph + 1)
    cc1 = min(w, c0 + pw + 1)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _mark_occupied(occupied, r0, c0, ph, pw):
    h, w = occupied.shape
    occupied[max(0, r0 - 1) : min(h, r0 + ph + 1), max(0, c0 - 1) : min(w, c0 + pw + 1)] = True


def _place_region(grid, occupied, color):
    h, w = grid.shape
    for _ in range(160):
        interior, ring = _make_region_patch()
        ph, pw = ring.shape
        if h - ph - 1 < 1 or w - pw - 1 < 1:
            continue
        r0 = random.randint(1, h - ph - 1)
        c0 = random.randint(1, w - pw - 1)
        if not _can_place(occupied, r0, c0, ph, pw):
            continue

        grid[r0 : r0 + ph, c0 : c0 + pw][ring] = GREY
        choices = np.argwhere(interior)
        sr, sc = map(int, choices[random.randrange(len(choices))])
        grid[r0 + sr, c0 + sc] = int(color)
        _mark_occupied(occupied, r0, c0, ph, pw)
        return True
    return False


def _gen_attempt():
    n_colors = random.randint(2, 4)
    legend_h = 4 * n_colors + 1
    height = random.randint(max(18, legend_h + 7), 30)
    width = random.randint(18, 30)

    colors = random.sample(list(SEED_COLORS), n_colors)
    masks = random.sample(_pattern_masks(), n_colors)

    grid = np.full((height, width), BLACK, dtype=int)
    legend_h, legend_w = _draw_legend(grid, colors, masks)

    occupied = np.zeros((height, width), dtype=bool)
    occupied[: min(height, legend_h + 1), : min(width, legend_w + 1)] = True

    region_count = random.randint(n_colors, min(5, n_colors + 2))
    region_colors = colors[:]
    while len(region_colors) < region_count:
        region_colors.append(random.choice(colors))
    random.shuffle(region_colors)

    for color in region_colors:
        if not _place_region(grid, occupied, color):
            return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
