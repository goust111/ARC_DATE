# Variant of 8b7bacbf
# Dimension: D13 — 线路颜色与种子颜色的关系 (meta)
# Change: 每个种子有独立的线路颜色，种子色唯一出现且与线路颜色不同... → 多条线路共享同一颜色
#
# Design analysis:
# variant_spec:
#   concrete_change:
#     - 输入仍然有 2 个边界单像素 seed，seed 颜色彼此不同且各自只出现一次。
#     - 两个 seed 各自连到一条 8 连通折线路径，但这两条路径现在使用同一个 shared_line_col。
#     - 若干单色框环仍旧贴在线路外侧；被某个 seed 所在网络连接到的环内腔体，要用该 seed 的颜色填满。
#   behavior_trace:
#     - 原题里，seed A 旁边常见的是线路色 x，seed B 旁边常见的是线路色 y；人和程序都能用颜色把两条路径直接区分开。
#     - 现在两个 seed 的邻接线路色都相同。solver 必须先从 seed 出发追踪它自己的 8 连通非背景组件，
#       再判断某个 cavity 的边界是否接触“这个组件内部的线路像素”，而不是接触“某个全局唯一线路颜色”。
#   complexity_calibration:
#     - 为了让 D13 在 2-4 个样例里肉眼可见，seed 数固定为 2；否则单 seed 样例无法展示“共享线路色”。
#     - 其余自由度基本保留：背景仍是最常见色，框环颜色池仍可复用，distractor 仍可存在。
#   old_assumption_audit:
#     - 原 `main()` 先从 seed 的 4 邻域读出 `line_col`，再用该颜色验证 cavity 是否贴着 seed 的那条线。
#     - 这在“每条线颜色不同”时无害；但在线路共享颜色后，若不同网络在 8 邻域上靠得太近，
#       仅凭颜色会把别的 seed 的线路也当成当前 seed 的线路。
#     - 新 `main()` 因此把判定收紧为：边界必须接触 `traced_line`，即“当前 seed 所追踪组件中的线路像素”。
#   placement_legality:
#     - generator 新增 `owners` 辅助网格，只在生成时使用，用来标记每个非背景像素属于哪个网络。
#     - 线段生长时禁止 8 邻接到 foreign owner；框环放置时只允许外侧接触所属网络的 line path，不能碰到别的网络或旧框环。
#   observability:
#     - 每个样例都包含 2 个不同 seed 色，但它们邻接到的线路色完全相同，解题者无法再靠线路颜色分配路径。
#   cleanliness:
#     - cavity 周围在放置前必须全空，避免共享线路色后出现视觉上像“穿腔而过”的干扰。
#   color_inference:
#     - 颜色规则被简化成“多个 seed 共用一个线路色”，没有引入新的颜色映射表；可观察线索来自连通性而不是颜色数值。

import random
from collections import Counter

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


def _components_mask(mask: np.ndarray, conn8: bool) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8 if conn8 else 4)]


def _touches_border(pts: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)


def _seed_pixels(g: np.ndarray, bg: int) -> list[tuple[int, int]]:
    h, w = g.shape
    vals, cnts = np.unique(g, return_counts=True)
    seeds: list[tuple[int, int]] = []
    for v, cnt in zip(vals, cnts):
        v = int(v)
        if v == bg or int(cnt) != 1:
            continue
        r, c = map(int, np.argwhere(g == v)[0])
        if r == 0 or c == 0 or r == h - 1 or c == w - 1:
            seeds.append((r, c))
    return seeds


def _nonbg_component8_from(g: np.ndarray, start: tuple[int, int], bg: int) -> set[tuple[int, int]]:
    h, w = g.shape
    st = [start]
    vis = {start}
    while st:
        r, c = st.pop()
        for nr, nc in neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in vis and int(g[nr, nc]) != bg:
                vis.add((nr, nc))
                st.append((nr, nc))
    return vis


def _seed_line_color(g: np.ndarray, seed: tuple[int, int], bg: int) -> int:
    h, w = g.shape
    sr, sc = seed
    neigh_cols = []
    for nr, nc in neighbors4(sr, sc):
        if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) != bg:
            neigh_cols.append(int(g[nr, nc]))
    assert neigh_cols, "seed pixel must have at least one non-bg neighbor"
    line_col = Counter(neigh_cols).most_common(1)[0][0]
    assert all(v == line_col for v in neigh_cols), "seed must touch exactly one shared line color"
    return int(line_col)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill enclosed regions along a line seeded by unique border pixels.

    Variant rule:
    - Border seed pixels still provide the fill color.
    - Multiple seeded line networks may now share the same line color.
    - Therefore the solver must separate networks by connectivity, not by line-color identity.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    out = g.copy()

    # Precompute background regions (4-connected). Exclude those containing background corners as "outside".
    bg_comps = _components_mask(g == bg, conn8=False)
    regions = []  # list of (set_pts, boundary_set, touches_border, size)
    for pts in bg_comps:
        region = set(pts)
        touches = _touches_border(pts, h, w)
        boundary = set()
        for r, c in region:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in region and int(g[nr, nc]) != bg:
                    boundary.add((nr, nc))
        regions.append((region, boundary, touches, len(region)))

    corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    outside_idxs = set()
    for i, (reg, _, _, _) in enumerate(regions):
        if any(p in reg for p in corners if int(g[p]) == bg):
            outside_idxs.add(i)

    cavities = []
    for i, (reg, bnd, _, _) in enumerate(regions):
        if i in outside_idxs or not bnd:
            continue
        bcols = {int(g[p]) for p in bnd}
        cavities.append((reg, bnd, bcols))

    for seed in _seed_pixels(g, bg):
        sr, sc = seed
        seed_col = int(g[sr, sc])
        line_col = _seed_line_color(g, seed, bg)

        # Critical D13 change: isolate the seed's own network by connectivity, then
        # look for shared-color line pixels only inside that traced component.
        traced = _nonbg_component8_from(g, seed, bg)
        traced_line = {p for p in traced if int(g[p]) == line_col}
        assert traced_line, "traced component must contain shared line pixels"

        for cav, boundary, bcols in cavities:
            if len(bcols) != 1:
                continue
            boundary_in_traced = boundary & traced
            if not boundary_in_traced:
                continue

            ok = False
            for r, c in boundary_in_traced:
                for nr, nc in neighbors8(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) in traced_line:
                        ok = True
                        break
                if ok:
                    break
            if not ok:
                continue

            for r, c in cav:
                if int(out[r, c]) == bg:
                    out[r, c] = seed_col

    return out


# --------------- generator (shared_line_color_variant) ---------------

# ========================== CFG ==========================
# Canvas / grid
GRID_H_RANGE = (8, 30)             # grid height bounds
GRID_W_RANGE = (8, 30)             # grid width bounds
BG_COLORS = [0, 5]                 # background color choices

# Seeds
SEED_COUNT = 2                     # fixed to keep the D13 change visible
MIN_LINE_LEN = 4                   # discard attempt if line shorter than this
LINE_LEN_RANGE = (8, 35)           # target line length bounds (clamped by h+w)

# Frame loops
FRAME_COLORS_MAX = 4               # max distinct frame colors
TOTAL_LOOPS_MAX = 9                # max total loops per grid
CAVITY_SIZE_WEIGHTS = [2, 2, 2, 2, 2, 3]        # cavity h/w choices (attached)
DISTRACTOR_CAVITY_WEIGHTS = [2, 2, 2, 3]        # cavity h/w choices (distractor)
FRAME_PLACEMENT_PAD = 1            # min distance from frame bbox to grid edge
FRAME_ATTACH_TRIES = 40            # retries per frame attachment attempt
DISTRACTOR_FRAME_TRIES = 40        # retries per distractor frame placement

# Line growth
DIAG_WEIGHT_ALIGNED = 5            # weight for bias-aligned diagonal steps
DIAG_WEIGHT_OTHER = 1              # weight for non-aligned diagonal steps
ORTHO_WEIGHT_ALIGNED = 2           # weight for bias-aligned orthogonal steps
ORTHO_WEIGHT_OTHER = 1             # weight for non-aligned orthogonal steps
BORDER_AVOIDANCE_PROB = 0.85       # probability of skipping border cells
BIAS_FLIP_PROB = 0.3               # probability of flipping bias when stuck

# Distractor network
DISTRACTOR_NETWORK_PROB = 0.6      # chance of line+frame network vs solo frames
DISTRACTOR_NETWORK_TRIES = 50      # retries for placing distractor network
DISTRACTOR_LINE_LEN = (5, 15)      # distractor line length range
DISTRACTOR_SOLO_MAX = 2            # max solo distractor frames
DISTRACTOR_NETWORK_FRAMES = (1, 3) # frames per distractor network

# Generation
GEN_RETRIES = 300                  # outer retry budget in generate_input()
# =========================================================

DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _in_bounds(h, w, r, c):
    return 0 <= r < h and 0 <= c < w


def _start_cell_clear(canvas: np.ndarray, first_rc: tuple[int, int], allowed_neighbor: tuple[int, int]) -> bool:
    h, w = canvas.shape
    r, c = first_rc
    if canvas[r, c] != -1:
        return False
    for nr, nc in neighbors8(r, c):
        if not _in_bounds(h, w, nr, nc):
            continue
        if canvas[nr, nc] != -1 and (nr, nc) != allowed_neighbor:
            return False
    return True


# ---------------------------------------------------------------------------
# Frame geometry
# ---------------------------------------------------------------------------

def _frame_ring_coords(r0, c0, ch, cw):
    """Return (ring_coords, cavity_coords) for a rounded-rect frame."""
    H, W = ch + 2, cw + 2
    ring = []
    for dr in range(H):
        for dc in range(W):
            on_bdr = (dr == 0 or dr == H - 1 or dc == 0 or dc == W - 1)
            corner = (dr in (0, H - 1)) and (dc in (0, W - 1))
            if on_bdr and not corner:
                ring.append((r0 + dr, c0 + dc))
    cavity = [(r0 + 1 + dr, c0 + 1 + dc) for dr in range(ch) for dc in range(cw)]
    return ring, cavity


def _validate_frame_placement(canvas, owners, ring, cavity, line_path, owner_id, h, w):
    """
    Check that ring+cavity can be placed without merging networks.

    New D13-specific assumption:
    - Different networks may share a line color, so ownership rather than color
      must guard against 8-neighborhood contact during generation.
    """
    ring_set = set(ring)
    line_path_set = set(line_path)

    for rr, cc in ring:
        if canvas[rr, cc] != -1:
            return False
    for rr, cc in cavity:
        if canvas[rr, cc] != -1:
            return False

    # Before the ring exists, the cavity must be completely free of occupied neighbors.
    for cr, cc in cavity:
        for nr, nc in neighbors8(cr, cc):
            if _in_bounds(h, w, nr, nc) and canvas[nr, nc] != -1:
                return False

    touches_line = False
    for rr, cc in ring:
        for nr, nc in neighbors8(rr, cc):
            if not _in_bounds(h, w, nr, nc) or (nr, nc) in ring_set:
                continue
            if canvas[nr, nc] == -1:
                continue
            if int(owners[nr, nc]) != owner_id:
                return False
            if (nr, nc) in line_path_set:
                touches_line = True
            else:
                return False
    return touches_line


# ---------------------------------------------------------------------------
# Line growth
# ---------------------------------------------------------------------------

def _grow_line(grid, owners, start, line_col, owner_id, target_len, rng):
    """Grow a meandering 8-connected 1px-wide path while staying owner-isolated."""
    h, w = grid.shape
    path = [start]
    dr_bias = 1 if start[0] < h // 2 else -1
    dc_bias = 1 if start[1] < w // 2 else -1
    cur = start
    path_set = {start}
    while len(path) < target_len:
        dirs = []
        for ddr, ddc in DIAG_DIRS:
            aligned = (ddr == dr_bias or ddc == dc_bias)
            w_d = DIAG_WEIGHT_ALIGNED if aligned else DIAG_WEIGHT_OTHER
            dirs.extend([(ddr, ddc)] * w_d)
        for ddr, ddc in ORTHO_DIRS:
            aligned = (ddr == dr_bias and ddc == 0) or (ddc == dc_bias and ddr == 0)
            w_d = ORTHO_WEIGHT_ALIGNED if aligned else ORTHO_WEIGHT_OTHER
            dirs.extend([(ddr, ddc)] * w_d)
        rng.shuffle(dirs)
        placed = False
        for ddr, ddc in dirs:
            nr, nc = cur[0] + ddr, cur[1] + ddc
            if not _in_bounds(h, w, nr, nc) or grid[nr, nc] != -1:
                continue

            blocked = False
            for nnr, nnc in neighbors8(nr, nc):
                if not _in_bounds(h, w, nnr, nnc):
                    continue
                if (nnr, nnc) == cur:
                    continue
                if (nnr, nnc) in path_set:
                    blocked = True
                    break
                if grid[nnr, nnc] != -1 and int(owners[nnr, nnc]) != owner_id:
                    blocked = True
                    break
            if blocked:
                continue

            if (nr == 0 or nc == 0 or nr == h - 1 or nc == w - 1) and rng.random() < BORDER_AVOIDANCE_PROB:
                continue

            cur = (nr, nc)
            path.append(cur)
            path_set.add(cur)
            grid[nr, nc] = line_col
            owners[nr, nc] = owner_id
            placed = True
            break
        if not placed:
            if rng.random() < BIAS_FLIP_PROB:
                dr_bias *= -1
            if rng.random() < BIAS_FLIP_PROB:
                dc_bias *= -1
            break
    return path


# ---------------------------------------------------------------------------
# Frame attachment
# ---------------------------------------------------------------------------

def _halo_offsets(fh, fw):
    """Compute offsets around a frame bbox that lie outside it (the halo)."""
    return [(dr, dc) for dr in range(-1, fh + 1) for dc in range(-1, fw + 1)
            if not (0 <= dr < fh and 0 <= dc < fw)]


def _try_attach_frame(canvas, owners, line_path, owner_id, frame_col, rng, max_tries=FRAME_ATTACH_TRIES):
    """Try to place a frame loop 8-adjacent to some line pixel of the same owner."""
    h, w = canvas.shape
    pad = FRAME_PLACEMENT_PAD
    for _ in range(max_tries):
        ch = rng.choice(CAVITY_SIZE_WEIGHTS)
        cw = rng.choice(CAVITY_SIZE_WEIGHTS)
        lp = rng.choice(line_path)
        fh, fw = ch + 2, cw + 2
        offsets = _halo_offsets(fh, fw)
        rng.shuffle(offsets)
        for dr, dc in offsets:
            r0 = lp[0] - dr
            c0 = lp[1] - dc
            if r0 < pad or c0 < pad or r0 + fh > h - pad or c0 + fw > w - pad:
                continue
            ring, cavity = _frame_ring_coords(r0, c0, ch, cw)
            if not _validate_frame_placement(canvas, owners, ring, cavity, line_path, owner_id, h, w):
                continue
            for rr, cc in ring:
                canvas[rr, cc] = frame_col
                owners[rr, cc] = owner_id
            return ring, cavity
    return None


def _try_attach_distractor_frame(canvas, owners, frame_col, owner_id, rng, max_tries=DISTRACTOR_FRAME_TRIES):
    """Place an isolated distractor loop in empty space."""
    h, w = canvas.shape
    for _ in range(max_tries):
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
            owners[rr, cc] = owner_id
        return True
    return None


def _clear_owner_pixels(canvas: np.ndarray, owners: np.ndarray, owner_id: int) -> None:
    pts = np.argwhere(owners == owner_id)
    for r, c in pts:
        canvas[int(r), int(c)] = -1
        owners[int(r), int(c)] = -1


def _try_place_distractor_network(canvas, owners, line_col, frame_colors_pool, rng, owner_id):
    """Place a disconnected line+frame distractor (no seed)."""
    h, w = canvas.shape
    for _ in range(DISTRACTOR_NETWORK_TRIES):
        r = rng.randint(2, h - 3)
        c = rng.randint(2, w - 3)
        if any(canvas[r + dr, c + dc] != -1 for dr in range(-1, 2) for dc in range(-1, 2)):
            continue
        canvas[r, c] = line_col
        owners[r, c] = owner_id
        target_len = rng.randint(*DISTRACTOR_LINE_LEN)
        path = _grow_line(canvas, owners, (r, c), line_col, owner_id, target_len, rng)
        if len(path) < 3:
            _clear_owner_pixels(canvas, owners, owner_id)
            continue
        attached = 0
        for _ in range(rng.randint(*DISTRACTOR_NETWORK_FRAMES)):
            fc = rng.choice(frame_colors_pool)
            if _try_attach_frame(canvas, owners, path, owner_id, fc, rng) is not None:
                attached += 1
        if attached > 0:
            return True
        _clear_owner_pixels(canvas, owners, owner_id)
    return False


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def _corner_config(corner_kind, h, w, rng):
    """Return (seed_rc, first_line_rc) for corner_kind in {TL,TR,BL,BR}."""
    if corner_kind == "TL":
        seed = (0, 0)
        first = rng.choice([(0, 1), (1, 0)])
    elif corner_kind == "TR":
        seed = (0, w - 1)
        first = rng.choice([(0, w - 2), (1, w - 1)])
    elif corner_kind == "BL":
        seed = (h - 1, 0)
        first = rng.choice([(h - 1, 1), (h - 2, 0)])
    elif corner_kind == "BR":
        seed = (h - 1, w - 1)
        first = rng.choice([(h - 1, w - 2), (h - 2, w - 1)])
    else:
        raise ValueError(corner_kind)
    return seed, first


def _distribute_loop_budget(num_seeds, num_attached_target, rng):
    """Distribute attached-loop budget across seeds, guaranteeing >= 1 each."""
    seed_order = list(range(num_seeds))
    rng.shuffle(seed_order)
    per_seed = [0] * num_seeds
    remaining = num_attached_target
    while remaining > 0 and any(per_seed[i] == 0 for i in range(num_seeds)):
        for i in seed_order:
            if per_seed[i] == 0 and remaining > 0:
                per_seed[i] += 1
                remaining -= 1
    while remaining > 0:
        per_seed[rng.choice(seed_order)] += 1
        remaining -= 1
    return per_seed


def _shared_line_color_for_seeds(grid: np.ndarray, bg: int) -> set[int]:
    cols = set()
    for seed in _seed_pixels(grid, bg):
        cols.add(_seed_line_color(grid, seed, bg))
    return cols


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bg = rng.choice(BG_COLORS)
    palette = [c for c in range(10) if c != bg]
    rng.shuffle(palette)

    num_seeds = SEED_COUNT
    if num_seeds + 1 > len(palette):
        return None

    canvas = -np.ones((h, w), dtype=int)
    owners = -np.ones((h, w), dtype=int)

    seed_colors = [palette.pop() for _ in range(num_seeds)]
    shared_line_col = palette.pop()

    all_corners = ["TL", "TR", "BL", "BR"]
    rng.shuffle(all_corners)
    corners_used = all_corners[:num_seeds]

    num_frame_cols = rng.randint(1, min(FRAME_COLORS_MAX, len(palette)))
    frame_colors_pool = [palette.pop() for _ in range(num_frame_cols)]

    lines_info = []
    for i in range(num_seeds):
        seed_rc, first_rc = _corner_config(corners_used[i], h, w, rng)
        if canvas[seed_rc] != -1 or not _start_cell_clear(canvas, first_rc, seed_rc):
            return None
        canvas[seed_rc] = seed_colors[i]
        owners[seed_rc] = i
        canvas[first_rc] = shared_line_col
        owners[first_rc] = i
        target_len = rng.randint(LINE_LEN_RANGE[0], min(LINE_LEN_RANGE[1], h + w))
        path = _grow_line(canvas, owners, first_rc, shared_line_col, i, target_len, rng)
        if len(path) < MIN_LINE_LEN:
            return None
        lines_info.append((seed_colors[i], seed_rc, shared_line_col, path, i))

    total_loops_target = rng.randint(num_seeds, TOTAL_LOOPS_MAX)
    num_distractors_target = rng.randint(0, max(0, total_loops_target - num_seeds))
    num_attached_target = total_loops_target - num_distractors_target
    per_seed = _distribute_loop_budget(num_seeds, num_attached_target, rng)

    for i, (_, _, _, path, owner_id) in enumerate(lines_info):
        attached = 0
        for _ in range(per_seed[i]):
            fc = rng.choice(frame_colors_pool)
            if _try_attach_frame(canvas, owners, path, owner_id, fc, rng) is not None:
                attached += 1
        if attached == 0:
            return None

    next_owner = num_seeds
    if rng.random() < DISTRACTOR_NETWORK_PROB:
        _try_place_distractor_network(canvas, owners, shared_line_col, frame_colors_pool, rng, next_owner)
    else:
        for j in range(rng.randint(0, DISTRACTOR_SOLO_MAX)):
            _try_attach_distractor_frame(canvas, owners, rng.choice(frame_colors_pool), next_owner + j, rng)

    grid = np.where(canvas == -1, bg, canvas).astype(int)

    vals, cnts = np.unique(grid, return_counts=True)
    if int(vals[np.argmax(cnts)]) != bg:
        return None

    for seed_col, _, _, _, _ in lines_info:
        if int((grid == seed_col).sum()) != 1:
            return None

    shared_cols = _shared_line_color_for_seeds(grid, bg)
    if len(shared_cols) != 1:
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
