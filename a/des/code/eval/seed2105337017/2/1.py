import math
import random
from collections import Counter, deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
QUADRANTS = ("TL", "TR", "BL", "BR")
GEN_RETRIES = 300


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        q = deque([(r, c)])
        pts: list[tuple[int, int]] = []
        while q:
            rr, cc = q.popleft()
            pts.append((rr, cc))
            for dr, dc in DIR4:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(np.asarray(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bool(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    assert len(pts) > 0
    r0, c0, r1, c1 = _bbox(pts)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _transform(mask: np.ndarray, quadrant: str) -> np.ndarray:
    if quadrant == "TL":
        return mask.copy()
    if quadrant == "TR":
        return np.fliplr(mask)
    if quadrant == "BL":
        return np.flipud(mask)
    if quadrant == "BR":
        return np.rot90(mask, 2)
    raise ValueError(quadrant)


def _scale_mask(mask: np.ndarray, scale: int) -> np.ndarray:
    return np.kron(mask.astype(np.uint8), np.ones((scale, scale), dtype=np.uint8)).astype(bool)


def _corner_offset(quadrant: str, box_h: int, box_w: int, shape_h: int, shape_w: int) -> tuple[int, int]:
    top = 0 if quadrant in ("TL", "TR") else box_h - shape_h
    left = 0 if quadrant in ("TL", "BL") else box_w - shape_w
    return top, left


def _seed_quadrant(seed_r: int, seed_c: int, r0: int, c0: int, r1: int, c1: int) -> str:
    top = 2 * (seed_r - r0) < (r1 - r0)
    left = 2 * (seed_c - c0) < (c1 - c0)
    if top and left:
        return "TL"
    if top and not left:
        return "TR"
    if not top and left:
        return "BL"
    return "BR"


def _find_legend(grid: np.ndarray, bg: int) -> tuple[tuple[int, int, int, int], int, int, np.ndarray]:
    candidates: list[tuple[int, int, int, int, int, int, int, int]] = []
    for frame_color in map(int, np.unique(grid)):
        if frame_color == bg:
            continue
        for comp in _components(grid == frame_color):
            r0, c0, r1, c1 = _bbox(comp)
            hh = r1 - r0 + 1
            ww = c1 - c0 + 1
            if hh < 4 or ww < 4:
                continue
            local = np.zeros((hh, ww), dtype=bool)
            local[comp[:, 0] - r0, comp[:, 1] - c0] = True
            border = np.zeros((hh, ww), dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if not np.array_equal(local, border):
                continue
            inside = grid[r0 + 1 : r1, c0 + 1 : c1]
            counts = Counter(int(v) for v in inside.ravel() if int(v) not in (bg, frame_color))
            if not counts:
                continue
            pattern_color, count = max(counts.items(), key=lambda item: item[1])
            if count < 2:
                continue
            candidates.append((count, hh * ww, r0, c0, r1, c1, frame_color, pattern_color))
    assert candidates, "legend frame not found"
    _, _, r0, c0, r1, c1, frame_color, pattern_color = max(candidates)
    template = _crop_bool(grid[r0 + 1 : r1, c0 + 1 : c1] == pattern_color)
    return (r0, c0, r1, c1), int(frame_color), int(pattern_color), template


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow the legend template inside each enclosed region.

    The template comes from the legend box. Each seed's quadrant inside its region
    chooses a transform and the corresponding corner anchor.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    (lr0, lc0, lr1, lc1), _, seed_color, template = _find_legend(g, bg)

    passable = (g == bg) | (g == seed_color)
    passable[lr0 : lr1 + 1, lc0 : lc1 + 1] = False

    out = g.copy()
    for comp in _components(passable):
        if np.any(comp[:, 0] == 0) or np.any(comp[:, 0] == g.shape[0] - 1):
            continue
        if np.any(comp[:, 1] == 0) or np.any(comp[:, 1] == g.shape[1] - 1):
            continue
        seeds = [(int(r), int(c)) for r, c in comp if int(g[r, c]) == seed_color]
        if len(seeds) != 1:
            continue
        seed_r, seed_c = seeds[0]
        r0, c0, r1, c1 = _bbox(comp)
        local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        local[comp[:, 0] - r0, comp[:, 1] - c0] = True
        quadrant = _seed_quadrant(seed_r, seed_c, r0, c0, r1, c1)
        transformed = _transform(template, quadrant)

        max_scale = min(local.shape[0] // transformed.shape[0], local.shape[1] // transformed.shape[1])
        assert max_scale >= 1, "template must fit at least at scale 1"

        placed = False
        for scale in range(max_scale, 0, -1):
            scaled = _scale_mask(transformed, scale)
            sh, sw = scaled.shape
            off_r, off_c = _corner_offset(quadrant, local.shape[0], local.shape[1], sh, sw)
            window = local[off_r : off_r + sh, off_c : off_c + sw]
            if window.shape != scaled.shape or np.any(scaled & ~window):
                continue
            patch = out[r0 + off_r : r0 + off_r + sh, c0 + off_c : c0 + off_c + sw]
            patch[scaled] = seed_color
            placed = True
            break
        assert placed, "scaled template must fit inside each enclosed region"

    return out


def _is_connected(mask: np.ndarray) -> bool:
    return mask.any() and len(_components(mask)) == 1


def _random_template(rng: random.Random) -> np.ndarray:
    for _ in range(1000):
        h = rng.randint(3, 5)
        w = rng.randint(3, 5)
        p = rng.uniform(0.35, 0.65)
        mask = np.array([[rng.random() < p for _ in range(w)] for _ in range(h)], dtype=bool)
        if mask.sum() < max(4, (h * w) // 3):
            continue
        if mask.sum() > h * w - 2:
            continue
        if not mask[0].any() or not mask[-1].any() or not mask[:, 0].any() or not mask[:, -1].any():
            continue
        if not _is_connected(mask):
            continue
        variants = [_transform(mask, q) for q in QUADRANTS]
        if len({variant.tobytes() for variant in variants}) != 4:
            continue
        return mask
    raise RuntimeError("failed to build an asymmetric template")


def _valid_interior(mask: np.ndarray, required: np.ndarray) -> bool:
    if np.any(required & ~mask):
        return False
    if not mask[0].any() or not mask[-1].any() or not mask[:, 0].any() or not mask[:, -1].any():
        return False
    if mask.sum() < required.sum() + 3:
        return False
    return _is_connected(mask)


def _make_irregular_interior(h: int, w: int, required: np.ndarray, rng: random.Random) -> np.ndarray:
    mask = np.ones((h, w), dtype=bool)
    carved = 0
    target = rng.randint(1, 4)
    attempts = 0
    while attempts < 80 and carved < target:
        attempts += 1
        cand = mask.copy()
        side = rng.choice(("top", "bottom", "left", "right"))
        if side in ("top", "bottom"):
            if w < 3:
                continue
            depth = rng.randint(1, max(1, min(h - 1, h // 3 + 1)))
            length = rng.randint(1, w - 1)
            start = rng.randint(0, w - length)
            rows = slice(0, depth) if side == "top" else slice(h - depth, h)
            cand[rows, start : start + length] = False
        else:
            if h < 3:
                continue
            depth = rng.randint(1, max(1, min(w - 1, w // 3 + 1)))
            length = rng.randint(1, h - 1)
            start = rng.randint(0, h - length)
            cols = slice(0, depth) if side == "left" else slice(w - depth, w)
            cand[start : start + length, cols] = False
        if not _valid_interior(cand, required):
            continue
        mask = cand
        carved += 1
    return mask


def _choose_seed(mask: np.ndarray, quadrant: str, rng: random.Random) -> tuple[int, int] | None:
    h, w = mask.shape
    top = quadrant in ("TL", "TR")
    left = quadrant in ("TL", "BL")
    row_lo, row_hi = (0, h // 2) if top else ((h + 1) // 2, h)
    col_lo, col_hi = (0, w // 2) if left else ((w + 1) // 2, w)
    coords = [
        (int(r), int(c))
        for r, c in np.argwhere(mask)
        if row_lo <= int(r) < row_hi and col_lo <= int(c) < col_hi
    ]
    if not coords:
        return None
    corner = {
        "TL": (0, 0),
        "TR": (0, w - 1),
        "BL": (h - 1, 0),
        "BR": (h - 1, w - 1),
    }[quadrant]
    coords.sort(key=lambda rc: abs(rc[0] - corner[0]) + abs(rc[1] - corner[1]))
    pool = coords[: min(len(coords), 8)]
    return pool[rng.randrange(len(pool))]


def _render_region_patch(interior: np.ndarray, wall_color: int, seed_color: int, seed: tuple[int, int]) -> np.ndarray:
    h, w = interior.shape
    patch = np.zeros((h + 2, w + 2), dtype=int)
    inner = np.zeros_like(patch, dtype=bool)
    inner[1 : h + 1, 1 : w + 1] = interior
    wall = np.zeros_like(patch, dtype=bool)
    for r, c in np.argwhere(inner):
        r = int(r)
        c = int(c)
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h + 2 and 0 <= nc < w + 2 and not inner[nr, nc]:
                wall[nr, nc] = True
    patch[wall] = wall_color
    sr, sc = seed
    patch[1 + sr, 1 + sc] = seed_color
    return patch


def _build_region_patch(
    template: np.ndarray,
    quadrant: str,
    wall_color: int,
    seed_color: int,
    max_inner_dim: int,
    rng: random.Random,
) -> np.ndarray | None:
    transformed = _transform(template, quadrant)
    th, tw = transformed.shape
    max_scale = min(3, max_inner_dim // th, max_inner_dim // tw)
    if max_scale < 1:
        return None
    scale = rng.randint(1, max_scale)
    inner_h = rng.randint(th * scale, min(max_inner_dim, th * (scale + 1) - 1))
    inner_w = rng.randint(tw * scale, min(max_inner_dim, tw * (scale + 1) - 1))
    scaled = _scale_mask(transformed, scale)
    sh, sw = scaled.shape
    off_r, off_c = _corner_offset(quadrant, inner_h, inner_w, sh, sw)
    required = np.zeros((inner_h, inner_w), dtype=bool)
    required[off_r : off_r + sh, off_c : off_c + sw] = scaled
    interior = _make_irregular_interior(inner_h, inner_w, required, rng)
    seed = _choose_seed(interior, quadrant, rng)
    if seed is None:
        return None
    return _render_region_patch(interior, wall_color, seed_color, seed)


def _legend_patch(template: np.ndarray, legend_color: int, seed_color: int) -> np.ndarray:
    h, w = template.shape
    patch = np.zeros((h + 4, w + 4), dtype=int)
    patch[0, :] = legend_color
    patch[-1, :] = legend_color
    patch[:, 0] = legend_color
    patch[:, -1] = legend_color
    patch[2 : 2 + h, 2 : 2 + w][template] = seed_color
    return patch


def _corner_origin(corner: str, h: int, w: int, ph: int, pw: int) -> tuple[int, int]:
    if corner == "TL":
        return 0, 0
    if corner == "TR":
        return 0, w - pw
    if corner == "BL":
        return h - ph, 0
    if corner == "BR":
        return h - ph, w - pw
    raise ValueError(corner)


def _rects_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _paste(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    grid[r0 : r0 + h, c0 : c0 + w] = patch


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = 0
    colors = list(range(1, 10))
    rng.shuffle(colors)
    wall_color, legend_color, seed_color = colors[:3]

    template = _random_template(rng)
    legend = _legend_patch(template, legend_color, seed_color)
    legend_corner = rng.choice(QUADRANTS)

    n_regions = rng.choice((2, 3, 4))
    quadrants = rng.sample(list(QUADRANTS), n_regions)
    max_inner_dim = 10 if n_regions == 4 else 12
    region_patches: list[np.ndarray] = []
    for quadrant in quadrants:
        patch = _build_region_patch(template, quadrant, wall_color, seed_color, max_inner_dim, rng)
        if patch is None:
            return None
        region_patches.append(patch)

    total_area = legend.size + sum(patch.size for patch in region_patches)
    base = max(
        22,
        int(math.ceil(math.sqrt(total_area * 1.65))),
        max(legend.shape[0], legend.shape[1]) + 4,
        max(max(patch.shape) for patch in region_patches) + 4,
    )
    base = min(base, 36)

    for _ in range(20):
        grid_h = base + rng.randint(0, 2)
        grid_w = base + rng.randint(0, 2)
        if grid_h > 36 or grid_w > 36:
            continue
        grid = np.full((grid_h, grid_w), bg, dtype=int)
        rects: list[tuple[int, int, int, int]] = []

        lr0, lc0 = _corner_origin(legend_corner, grid_h, grid_w, legend.shape[0], legend.shape[1])
        _paste(grid, legend, lr0, lc0)
        rects.append((lr0, lc0, lr0 + legend.shape[0] - 1, lc0 + legend.shape[1] - 1))

        ok = True
        for patch in sorted(region_patches, key=lambda item: item.shape[0] * item.shape[1], reverse=True):
            ph, pw = patch.shape
            if ph + 2 > grid_h or pw + 2 > grid_w:
                ok = False
                break
            placed = False
            for _ in range(250):
                r0 = rng.randint(1, grid_h - ph - 1)
                c0 = rng.randint(1, grid_w - pw - 1)
                rect = (r0, c0, r0 + ph - 1, c0 + pw - 1)
                if any(_rects_close(rect, other, 1) for other in rects):
                    continue
                _paste(grid, patch, r0, c0)
                rects.append(rect)
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            return grid
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
