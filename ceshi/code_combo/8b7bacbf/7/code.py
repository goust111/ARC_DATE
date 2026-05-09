# combo_spec: D4/0+D14/0
# task_id: 8b7bacbf
# name: 非边缘种子+4连通连接判定

import random
from collections import Counter

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8

"""
Fusion design: D4/0 + D14/0

- D4/0: seeds are globally unique non-background singleton pixels placed in the
  interior, not on the border.
- D14/0: a cavity fills only when its boundary has an orthogonal (4-neighbor)
  contact with a traced line-color pixel. Corner-only touches remain unfilled.

Generator obligations:
- every generated grid has at least one interior seed
- every seed gets at least one orthogonally attached ring that should fill
- the overall puzzle also contains at least one diagonal-only attached ring on a
  traced component, so the stricter 4-connectivity rule is observable
"""


def _components_mask(mask: np.ndarray, conn8: bool) -> list[list[tuple[int, int]]]:
    comps = components_mask(mask, connectivity=8 if conn8 else 4)
    return [[(int(r), int(c)) for r, c in comp] for comp in comps]


def _seed_pixels(g: np.ndarray, bg: int) -> list[tuple[int, int]]:
    vals, cnts = np.unique(g, return_counts=True)
    seeds: list[tuple[int, int]] = []
    for v, cnt in zip(vals, cnts):
        v = int(v)
        if v == bg or int(cnt) != 1:
            continue
        r, c = map(int, np.argwhere(g == v)[0])
        seeds.append((r, c))
    return seeds


def _nonbg_component8_from(g: np.ndarray, start: tuple[int, int], bg: int) -> set[tuple[int, int]]:
    h, w = g.shape
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and int(g[nr, nc]) != bg:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return seen


def _extract_cavities(
    g: np.ndarray, bg: int
) -> list[tuple[set[tuple[int, int]], set[tuple[int, int]], set[int]]]:
    h, w = g.shape
    regions = []
    for pts in _components_mask(g == bg, conn8=False):
        region = set(pts)
        boundary = set()
        for r, c in region:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in region and int(g[nr, nc]) != bg:
                    boundary.add((nr, nc))
        regions.append((region, boundary))

    corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    outside = set()
    for i, (region, _) in enumerate(regions):
        if any(p in region for p in corners if int(g[p]) == bg):
            outside.add(i)

    cavities = []
    for i, (region, boundary) in enumerate(regions):
        if i in outside or not boundary:
            continue
        boundary_colors = {int(g[p]) for p in boundary}
        cavities.append((region, boundary, boundary_colors))
    return cavities


def _line_contact_mode(
    g: np.ndarray,
    boundary: set[tuple[int, int]],
    traced: set[tuple[int, int]],
    line_col: int,
) -> str | None:
    h, w = g.shape
    has_any8 = False
    has_any4 = False
    for r, c in boundary:
        for nr, nc in neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) in traced and int(g[nr, nc]) == line_col:
                has_any8 = True
                break
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) in traced and int(g[nr, nc]) == line_col:
                has_any4 = True
                break
    if has_any4:
        return "ortho"
    if has_any8:
        return "diag_only"
    return None


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()
    cavities = _extract_cavities(g, bg)

    for sr, sc in _seed_pixels(g, bg):
        seed_col = int(g[sr, sc])
        neigh_cols = []
        for nr, nc in neighbors4(sr, sc):
            if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]:
                v = int(g[nr, nc])
                if v != bg:
                    neigh_cols.append(v)
        if not neigh_cols:
            continue
        line_col = Counter(neigh_cols).most_common(1)[0][0]
        traced = _nonbg_component8_from(g, (sr, sc), bg)

        for cavity, boundary, boundary_colors in cavities:
            if len(boundary_colors) != 1:
                continue
            if _line_contact_mode(g, boundary, traced, line_col) != "ortho":
                continue
            for r, c in cavity:
                if int(out[r, c]) == bg:
                    out[r, c] = seed_col

    return out


# --------------- generator ---------------

GRID_H_RANGE = (10, 26)
GRID_W_RANGE = (10, 26)
BG_COLORS = [0, 5]

SEED_COUNT_WEIGHTS = [1, 1, 2]
SEED_MARGIN = 2
SEED_PLACEMENT_TRIES = 100
MIN_LINE_LEN = 5
LINE_LEN_RANGE = (9, 28)

FRAME_COLORS_MAX = 4
CAVITY_SIZE_WEIGHTS = [2, 2, 2, 3]
DISTRACTOR_CAVITY_WEIGHTS = [2, 2, 3]
FRAME_PLACEMENT_PAD = 1
ORTHO_ATTACH_TRIES = 60
DIAG_ATTACH_TRIES = 120
DISTRACTOR_FRAME_TRIES = 40

DIAG_WEIGHT_ALIGNED = 5
DIAG_WEIGHT_OTHER = 1
ORTHO_WEIGHT_ALIGNED = 2
ORTHO_WEIGHT_OTHER = 1
BIAS_FLIP_PROB = 0.3
BORDER_AVOIDANCE_PROB = 0.35

GEN_RETRIES = 400

DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _frame_ring_coords(r0: int, c0: int, ch: int, cw: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    height, width = ch + 2, cw + 2
    ring = []
    for dr in range(height):
        for dc in range(width):
            on_border = dr == 0 or dr == height - 1 or dc == 0 or dc == width - 1
            is_corner = (dr in (0, height - 1)) and (dc in (0, width - 1))
            if on_border and not is_corner:
                ring.append((r0 + dr, c0 + dc))
    cavity = [(r0 + 1 + dr, c0 + 1 + dc) for dr in range(ch) for dc in range(cw)]
    return ring, cavity


def _ring_contact_mode(canvas: np.ndarray, ring: list[tuple[int, int]], line_col: int) -> str | None:
    h, w = canvas.shape
    has_any8 = False
    has_any4 = False
    for rr, cc in ring:
        for nr, nc in neighbors8(rr, cc):
            if _in_bounds(h, w, nr, nc) and canvas[nr, nc] == line_col:
                has_any8 = True
                break
        for nr, nc in neighbors4(rr, cc):
            if _in_bounds(h, w, nr, nc) and canvas[nr, nc] == line_col:
                has_any4 = True
                break
    if has_any4:
        return "ortho"
    if has_any8:
        return "diag_only"
    return None


def _validate_frame_placement(
    canvas: np.ndarray,
    ring: list[tuple[int, int]],
    cavity: list[tuple[int, int]],
    line_col: int,
    require_contact: str,
) -> bool:
    h, w = canvas.shape
    for rr, cc in ring:
        if canvas[rr, cc] != -1:
            return False
    for rr, cc in cavity:
        if canvas[rr, cc] != -1:
            return False

    for cr, cc in cavity:
        for nr, nc in neighbors8(cr, cc):
            if _in_bounds(h, w, nr, nc) and canvas[nr, nc] == line_col:
                return False

    ring_set = set(ring)
    for rr, cc in ring:
        for nr, nc in neighbors8(rr, cc):
            if not _in_bounds(h, w, nr, nc) or (nr, nc) in ring_set:
                continue
            v = canvas[nr, nc]
            if v != -1 and v != line_col:
                return False

    mode = _ring_contact_mode(canvas, ring, line_col)
    if require_contact == "ortho":
        return mode == "ortho"
    if require_contact == "diag_only":
        return mode == "diag_only"
    raise ValueError(require_contact)


def _grow_line(
    grid: np.ndarray,
    start: tuple[int, int],
    line_col: int,
    target_len: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    h, w = grid.shape
    mid_r = (h - 1) / 2
    mid_c = (w - 1) / 2
    dr_side = -1 if start[0] < mid_r else 1
    dc_side = -1 if start[1] < mid_c else 1
    dr_bias = dr_side
    dc_bias = dc_side
    cur = start
    path = [start]

    while len(path) < target_len:
        dirs = []
        for ddr, ddc in DIAG_DIRS:
            aligned = ddr == dr_bias or ddc == dc_bias
            weight = DIAG_WEIGHT_ALIGNED if aligned else DIAG_WEIGHT_OTHER
            dirs.extend([(ddr, ddc)] * weight)
        for ddr, ddc in ORTHO_DIRS:
            aligned = (ddr == dr_bias and ddc == 0) or (ddc == dc_bias and ddr == 0)
            weight = ORTHO_WEIGHT_ALIGNED if aligned else ORTHO_WEIGHT_OTHER
            dirs.extend([(ddr, ddc)] * weight)
        rng.shuffle(dirs)

        placed = False
        for ddr, ddc in dirs:
            nr, nc = cur[0] + ddr, cur[1] + ddc
            if not _in_bounds(h, w, nr, nc) or grid[nr, nc] != -1:
                continue
            if any(
                _in_bounds(h, w, ar, ac) and grid[ar, ac] != -1 and (ar, ac) != cur
                for ar, ac in neighbors8(nr, nc)
            ):
                continue
            if (nr == 0 or nc == 0 or nr == h - 1 or nc == w - 1) and rng.random() < BORDER_AVOIDANCE_PROB:
                continue
            cur = (nr, nc)
            path.append(cur)
            grid[nr, nc] = line_col
            placed = True
            break

        if not placed:
            if rng.random() < BIAS_FLIP_PROB:
                dr_bias *= -1
            if rng.random() < BIAS_FLIP_PROB:
                dc_bias *= -1
            else:
                break
    return path


def _halo_offsets(fh: int, fw: int) -> list[tuple[int, int]]:
    return [
        (dr, dc)
        for dr in range(-1, fh + 1)
        for dc in range(-1, fw + 1)
        if not (0 <= dr < fh and 0 <= dc < fw)
    ]


def _try_attach_frame(
    canvas: np.ndarray,
    line_path: list[tuple[int, int]],
    line_col: int,
    frame_col: int,
    rng: random.Random,
    require_contact: str,
    max_tries: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    h, w = canvas.shape
    pad = FRAME_PLACEMENT_PAD
    for _ in range(max_tries):
        ch = rng.choice(CAVITY_SIZE_WEIGHTS)
        cw = rng.choice(CAVITY_SIZE_WEIGHTS)
        fh, fw = ch + 2, cw + 2
        anchor = rng.choice(line_path)
        offsets = _halo_offsets(fh, fw)
        rng.shuffle(offsets)
        for dr, dc in offsets:
            r0 = anchor[0] - dr
            c0 = anchor[1] - dc
            if r0 < pad or c0 < pad or r0 + fh > h - pad or c0 + fw > w - pad:
                continue
            ring, cavity = _frame_ring_coords(r0, c0, ch, cw)
            if not _validate_frame_placement(canvas, ring, cavity, line_col, require_contact):
                continue
            for rr, cc in ring:
                canvas[rr, cc] = frame_col
            return ring, cavity
    return None


def _try_attach_distractor_frame(canvas: np.ndarray, frame_col: int, rng: random.Random) -> bool:
    h, w = canvas.shape
    for _ in range(DISTRACTOR_FRAME_TRIES):
        ch = rng.choice(DISTRACTOR_CAVITY_WEIGHTS)
        cw = rng.choice(DISTRACTOR_CAVITY_WEIGHTS)
        fh, fw = ch + 2, cw + 2
        r0 = rng.randint(1, h - fh - 1)
        c0 = rng.randint(1, w - fw - 1)
        ok = True
        for rr in range(r0 - 1, r0 + fh + 1):
            for cc in range(c0 - 1, c0 + fw + 1):
                if _in_bounds(h, w, rr, cc) and canvas[rr, cc] != -1:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        ring, _ = _frame_ring_coords(r0, c0, ch, cw)
        for rr, cc in ring:
            canvas[rr, cc] = frame_col
        return True
    return False


def _pick_seed_and_first(
    canvas: np.ndarray, rng: random.Random
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    h, w = canvas.shape
    mid_r = (h - 1) / 2
    mid_c = (w - 1) / 2
    candidates = []
    for r in range(SEED_MARGIN, h - SEED_MARGIN):
        for c in range(SEED_MARGIN, w - SEED_MARGIN):
            if canvas[r, c] != -1:
                continue
            if any(canvas[rr, cc] != -1 for rr in range(r - 1, r + 2) for cc in range(c - 1, c + 2)):
                continue
            centrality = abs(r - mid_r) + abs(c - mid_c)
            border_dist = min(r, c, h - 1 - r, w - 1 - c)
            candidates.append((centrality, -border_dist, (r, c)))
    if not candidates:
        return None

    candidates.sort()
    for _, _, seed_rc in candidates[: min(16, len(candidates))]:
        seed_dist = abs(seed_rc[0] - mid_r) + abs(seed_rc[1] - mid_c)
        first_choices = []
        for nr, nc in neighbors4(*seed_rc):
            if not _in_bounds(h, w, nr, nc) or canvas[nr, nc] != -1:
                continue
            next_dist = abs(nr - mid_r) + abs(nc - mid_c)
            outward_rank = 0 if next_dist > seed_dist else 1
            first_choices.append((outward_rank, rng.random(), (nr, nc)))
        if first_choices:
            first_choices.sort()
            return seed_rc, first_choices[0][2]
    return None


def _distribute_loop_budget(num_seeds: int, total: int, rng: random.Random) -> list[int]:
    order = list(range(num_seeds))
    rng.shuffle(order)
    out = [0] * num_seeds
    remain = total
    while remain > 0 and any(v == 0 for v in out):
        for i in order:
            if out[i] == 0 and remain > 0:
                out[i] += 1
                remain -= 1
    while remain > 0:
        out[rng.choice(order)] += 1
        remain -= 1
    return out


def _attachment_stats(g: np.ndarray) -> dict[str, int]:
    bg = get_bg(g)
    cavities = _extract_cavities(g, bg)
    stats = {"ortho": 0, "diag_only": 0}
    for sr, sc in _seed_pixels(g, bg):
        neigh_cols = []
        for nr, nc in neighbors4(sr, sc):
            if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]:
                v = int(g[nr, nc])
                if v != bg:
                    neigh_cols.append(v)
        if not neigh_cols:
            continue
        line_col = Counter(neigh_cols).most_common(1)[0][0]
        traced = _nonbg_component8_from(g, (sr, sc), bg)
        for _, boundary, boundary_colors in cavities:
            if len(boundary_colors) != 1:
                continue
            mode = _line_contact_mode(g, boundary, traced, line_col)
            if mode in stats:
                stats[mode] += 1
    return stats


def _all_seeds_internal(g: np.ndarray) -> bool:
    bg = get_bg(g)
    h, w = g.shape
    seeds = _seed_pixels(g, bg)
    if not seeds:
        return False
    return all(0 < r < h - 1 and 0 < c < w - 1 for r, c in seeds)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bg = rng.choice(BG_COLORS)
    palette = [c for c in range(10) if c != bg]
    rng.shuffle(palette)

    num_seeds = rng.choice(SEED_COUNT_WEIGHTS)
    if num_seeds * 2 > len(palette):
        return None

    canvas = -np.ones((h, w), dtype=int)
    seed_colors = [palette.pop() for _ in range(num_seeds)]
    line_colors = [palette.pop() for _ in range(num_seeds)]
    frame_colors = [palette.pop() for _ in range(rng.randint(1, min(FRAME_COLORS_MAX, len(palette))))]

    lines_info = []
    for i in range(num_seeds):
        picked = _pick_seed_and_first(canvas, rng)
        if picked is None:
            return None
        seed_rc, first_rc = picked
        canvas[seed_rc] = seed_colors[i]
        canvas[first_rc] = line_colors[i]
        target_len = rng.randint(LINE_LEN_RANGE[0], min(LINE_LEN_RANGE[1], h + w))
        path = _grow_line(canvas, first_rc, line_colors[i], target_len, rng)
        if len(path) < MIN_LINE_LEN:
            return None
        lines_info.append((seed_colors[i], seed_rc, line_colors[i], path))

    total_ortho = rng.randint(num_seeds, num_seeds + 2)
    per_seed_ortho = _distribute_loop_budget(num_seeds, total_ortho, rng)
    per_seed_diag = [0] * num_seeds
    for _ in range(rng.randint(1, max(1, num_seeds))):
        per_seed_diag[rng.randrange(num_seeds)] += 1

    for i, (_, _, line_col, path) in enumerate(lines_info):
        for _ in range(per_seed_ortho[i]):
            frame_col = rng.choice(frame_colors)
            placed = _try_attach_frame(
                canvas,
                path,
                line_col,
                frame_col,
                rng,
                require_contact="ortho",
                max_tries=ORTHO_ATTACH_TRIES,
            )
            if placed is None:
                return None

    for i, (_, _, line_col, path) in enumerate(lines_info):
        for _ in range(per_seed_diag[i]):
            frame_col = rng.choice(frame_colors)
            placed = _try_attach_frame(
                canvas,
                path,
                line_col,
                frame_col,
                rng,
                require_contact="diag_only",
                max_tries=DIAG_ATTACH_TRIES,
            )
            if placed is None:
                return None

    if rng.random() < 0.5:
        _try_attach_distractor_frame(canvas, rng.choice(frame_colors), rng)

    grid = np.where(canvas == -1, bg, canvas).astype(int)

    vals, cnts = np.unique(grid, return_counts=True)
    if int(vals[np.argmax(cnts)]) != bg:
        return None
    if not _all_seeds_internal(grid):
        return None
    for seed_col, _, _, _ in lines_info:
        if int((grid == seed_col).sum()) != 1:
            return None

    stats = _attachment_stats(grid)
    if stats["ortho"] < total_ortho or stats["diag_only"] < sum(per_seed_diag):
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("generate_input: exhausted attempts")
