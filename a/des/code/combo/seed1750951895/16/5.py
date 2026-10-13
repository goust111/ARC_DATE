import random
from collections import Counter

import numpy as np


BG = 0
GEN_RETRIES = 400


def get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(pts)
    return out


def bbox(pts: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    return min(rs), min(cs), max(rs), max(cs)


def find_frame(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    candidates: list[tuple[int, tuple[int, int, int, int], int]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for pts in components(grid == color):
            r0, c0, r1, c1 = bbox(pts)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            ring = np.zeros_like(grid, dtype=bool)
            ring[r0, c0 : c1 + 1] = True
            ring[r1, c0 : c1 + 1] = True
            ring[r0 : r1 + 1, c0] = True
            ring[r0 : r1 + 1, c1] = True
            if len(pts) != int(ring.sum()):
                continue
            if not np.all(grid[ring] == color):
                continue
            candidates.append((int(ring.sum()), (r0, c0, r1, c1), color))
    assert candidates, "no frame found"
    _area, frame_bbox, frame_color = max(candidates, key=lambda x: x[0])
    return frame_color, frame_bbox


def extract_legend(
    grid: np.ndarray,
    bg: int,
    frame_bbox: tuple[int, int, int, int],
    placeholder_colors: set[int],
) -> dict[int, int]:
    outside = np.ones_like(grid, dtype=bool)
    fr0, fc0, fr1, fc1 = frame_bbox
    outside[fr0 : fr1 + 1, fc0 : fc1 + 1] = False
    mask = outside & (grid != bg)
    candidates: list[dict[int, int]] = []
    for pts in components(mask):
        r0, c0, r1, c1 = bbox(pts)
        if c1 - c0 + 1 != 2:
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(sub == bg):
            continue
        left = [int(x) for x in sub[:, 0]]
        right = [int(x) for x in sub[:, 1]]
        if len(set(left)) != len(left):
            continue
        if set(left) != placeholder_colors:
            continue
        if any(l == r for l, r in zip(left, right)):
            continue
        candidates.append({l: r for l, r in zip(left, right)})
    assert len(candidates) == 1, "legend must be unique"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    frame_color, (fr0, fc0, fr1, fc1) = find_frame(g, bg)
    crop = g[fr0 : fr1 + 1, fc0 : fc1 + 1].copy()
    interior = crop[1:-1, 1:-1]

    placeholder_colors = {
        int(g[fr0 + r, fc0 + c])
        for r in range(1, crop.shape[0] - 1)
        for c in range(1, crop.shape[1] - 1)
        if int(g[fr0 + r, fc0 + c]) not in (bg, frame_color)
    }
    assert placeholder_colors, "no placeholders found"

    placeholders: dict[int, tuple[int, int]] = {}
    for color in placeholder_colors:
        pts = list(zip(*np.where(interior == color)))
        assert len(pts) == 1, "placeholders must be single pixels"
        placeholders[color] = (int(pts[0][0]), int(pts[0][1]))

    legend = extract_legend(g, bg, (fr0, fc0, fr1, fc1), placeholder_colors)

    outside = np.ones_like(g, dtype=bool)
    outside[fr0 : fr1 + 1, fc0 : fc1 + 1] = False
    sprites: dict[int, tuple[list[tuple[int, int]], tuple[int, int]]] = {}
    for color in sorted(placeholder_colors):
        comps = components(outside & (g == color))
        candidates = []
        for pts in comps:
            r0, c0, r1, c1 = bbox(pts)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if len(pts) <= 1:
                continue
            if h % 2 == 0 or w % 2 == 0:
                continue
            local = {(r - r0, c - c0) for r, c in pts}
            center = (h // 2, w // 2)
            if center not in local:
                continue
            candidates.append((len(pts), sorted(local), (h, w)))
        assert len(candidates) == 1, f"sprite for color {color} must be unique"
        _area, local_pts, shape = candidates[0]
        sprites[color] = (local_pts, shape)

    out = crop.copy()
    out[1:-1, 1:-1] = bg

    ih, iw = interior.shape
    for color, (pr, pc) in placeholders.items():
        local_pts, (h, w) = sprites[color]
        out_color = legend[color]
        off_r = pr - h // 2 + 1
        off_c = pc - w // 2 + 1
        for dr, dc in local_pts:
            rr = off_r + dr
            cc = off_c + dc
            assert 1 <= rr < ih + 1 and 1 <= cc < iw + 1
            out[rr, cc] = out_color
    return out


def random_shape(rng: random.Random, h: int, w: int) -> np.ndarray:
    center = (h // 2, w // 2)
    cells = {center}
    target = rng.randint(max(3, (h * w) // 3), min(h * w - 1, (2 * h * w) // 3))
    frontier = [center]
    while len(cells) < target:
        base = frontier[rng.randrange(len(frontier))]
        choices = []
        for nr, nc in neighbors4(*base):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                choices.append((nr, nc))
        if choices:
            nxt = choices[rng.randrange(len(choices))]
            cells.add(nxt)
            frontier.append(nxt)
        else:
            frontier.pop(rng.randrange(len(frontier)))
            if not frontier:
                frontier = list(cells)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    # Force the visible bbox to match the intended odd-sized patch.
    edge_targets = [(0, center[1]), (h - 1, center[1]), (center[0], 0), (center[0], w - 1)]
    for tr, tc in edge_targets:
        r, c = center
        while (r, c) != (tr, tc):
            if r < tr:
                r += 1
            elif r > tr:
                r -= 1
            elif c < tc:
                c += 1
            else:
                c -= 1
            mask[r, c] = True
    return mask


def can_place(occ: np.ndarray, top: int, left: int, h: int, w: int, pad: int = 1) -> bool:
    H, W = occ.shape
    if top < 0 or left < 0 or top + h > H or left + w > W:
        return False
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(H, top + h + pad)
    c1 = min(W, left + w + pad)
    return not occ[r0:r1, c0:c1].any()


def stamp_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int):
    h, w = patch.shape
    mask = patch != BG
    grid[top : top + h, left : left + w][mask] = patch[mask]


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        try:
            return _gen_attempt(rng)
        except AssertionError:
            continue
    raise RuntimeError("failed to generate valid input")


def _gen_attempt(rng: random.Random) -> np.ndarray:
    grid_h = rng.randint(18, 28)
    grid_w = rng.randint(18, 30)
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    occ = np.zeros((grid_h, grid_w), dtype=bool)

    count = rng.randint(2, 4)
    palette = list(range(1, 10))
    frame_color = rng.choice(palette)
    palette.remove(frame_color)
    placeholder_colors = rng.sample(palette, count)
    for c in placeholder_colors:
        palette.remove(c)
    output_colors = rng.sample(palette, count)

    sprite_specs = []
    max_h = max_w = 0
    for color in placeholder_colors:
        h = rng.choice([3, 5])
        w = rng.choice([3, 5])
        mask = random_shape(rng, h, w)
        sprite_specs.append((color, mask))
        max_h = max(max_h, h)
        max_w = max(max_w, w)

    inner_h = rng.randint(max(8, max_h + 4), 14)
    inner_w = rng.randint(max(8, max_w + 6), 18)
    frame_h = inner_h + 2
    frame_w = inner_w + 2
    assert frame_h + 2 <= grid_h and frame_w + 2 <= grid_w
    frame_top = rng.randint(1, grid_h - frame_h - 1)
    frame_left = rng.randint(1, grid_w - frame_w - 1)

    grid[frame_top, frame_left : frame_left + frame_w] = frame_color
    grid[frame_top + frame_h - 1, frame_left : frame_left + frame_w] = frame_color
    grid[frame_top : frame_top + frame_h, frame_left] = frame_color
    grid[frame_top : frame_top + frame_h, frame_left + frame_w - 1] = frame_color
    occ[frame_top : frame_top + frame_h, frame_left : frame_left + frame_w] = True

    placed_centers: list[tuple[int, int, int, int]] = []
    placeholders: dict[int, tuple[int, int]] = {}
    for color, mask in sprite_specs:
        h, w = mask.shape
        for _ in range(200):
            pr = rng.randint(h // 2, inner_h - 1 - h // 2)
            pc = rng.randint(w // 2, inner_w - 1 - w // 2)
            r0 = pr - h // 2
            c0 = pc - w // 2
            r1 = pr + h // 2
            c1 = pc + w // 2
            ok = True
            for or0, oc0, or1, oc1 in placed_centers:
                if not (r1 + 1 < or0 or or1 + 1 < r0 or c1 + 1 < oc0 or oc1 + 1 < c0):
                    ok = False
                    break
            if ok:
                placed_centers.append((r0, c0, r1, c1))
                placeholders[color] = (pr, pc)
                grid[frame_top + 1 + pr, frame_left + 1 + pc] = color
                break
        assert color in placeholders

    legend_rows = count
    legend_patch = np.zeros((legend_rows, 2), dtype=int)
    row_order = placeholder_colors[:]
    rng.shuffle(row_order)
    color_map = {src: dst for src, dst in zip(placeholder_colors, output_colors)}
    for i, src in enumerate(row_order):
        legend_patch[i, 0] = src
        legend_patch[i, 1] = color_map[src]

    for _ in range(200):
        lt = rng.randint(0, grid_h - legend_rows)
        ll = rng.randint(0, grid_w - 2)
        if can_place(occ, lt, ll, legend_rows, 2):
            stamp_patch(grid, legend_patch, lt, ll)
            occ[lt:lt + legend_rows, ll:ll + 2] = True
            break
    else:
        raise AssertionError("legend placement failed")

    for color, mask in sprite_specs:
        h, w = mask.shape
        patch = np.where(mask, color, BG)
        for _ in range(200):
            top = rng.randint(0, grid_h - h)
            left = rng.randint(0, grid_w - w)
            if can_place(occ, top, left, h, w):
                stamp_patch(grid, patch, top, left)
                occ[top:top + h, left:left + w] = True
                break
        else:
            raise AssertionError("sprite placement failed")

    assert len(set(placeholders)) == count
    return grid
