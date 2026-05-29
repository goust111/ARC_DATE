# Variant of d35bdbdc
# Dimension: D13 — 属性转移的范围 (transform)
# Change: 仅转移中心色（单像素或小区域的颜色替换）... → 转移源精灵的完整图案

import numpy as np

from arc_utils import neighbors4, neighbors8, components_mask, paste
import random
from collections import deque


# Design analysis
# ---
# variant_spec:
#   goal:
#     keep:
#       - 灰色单笔路径选出两个 destination sprite
#       - destination 的中心像素颜色仍作为 lookup key，去匹配 source 的 border color
#       - 非 destination 精灵全部删除，灰路径保留
#     change:
#       - 原题只把 source 的中心色写到 destination 中心
#       - 现在把 source 的整个内部 3x3 图案复制到 destination 的内部 3x3 区域
#   sprite_geometry:
#     original_issue:
#       - 原题精灵是 3x3，内部只有一个中心像素；如果直接照搬，D13 变化不可见
#     adaptation:
#       - 所有精灵统一升级为 5x5 方框边框，border 厚度 1
#       - 内部是完整的 3x3 图案区，便于“整块图案转移”
#       - 这是为 D13 可观察性做的最小必要适配；路径机制和颜色匹配机制保持不变
#   concrete_behavior_trace:
#     example:
#       - 某个 source 的 border=7，内部 3x3 为:
#           2 4 4
#           2 2 4
#           4 2 4
#       - 某个 destination 的中心像素颜色=7，因此它指向这个 source
#       - 原逻辑只会把 destination 中心 1 个像素改成 source.center_color
#       - 新逻辑会把 destination 整个内部 3x3 改写成上面的图案，位置和朝向不变
#   complexity_calibration:
#     - source 图案从一个小型、非对称的 3x3 模板库中采样，避免退化成“只改中心”或“纯色块”
#     - destination 输入态内部保持简单：8 个填充色 + 1 个中心 lookup key，便于人眼看出 key 在中心
#     - 模板库数量有限，且都保持原朝向，不引入旋转/翻转分支，避免 2-3 个样例无法归纳
#   implicit_assumption_audit:
#     original_main_dependencies:
#       - _detect_bordered_sprites 假设每个 sprite 恰好只有两种颜色且中心色只出现一次
#       - _ports_for_sprite 假设 port 距中心曼哈顿距离为 2，中间那一格就是 border
#       - 最终写出逻辑只做 out[center] = src.center
#     variant_updates:
#       - 新检测改为“组件 bbox 必须是 5x5，外围一圈颜色统一，内部 3x3 任意非 bg/grey”
#       - 新 port 规则改为 endpoint 距中心 3 格；离中心 2 格的位置必须是 border
#       - 最终写出逻辑改为整块覆盖 destination 的内部 3x3，并且使用输入中预先记录的 source.interior
#   placement_legality:
#     - 15x15 画布上使用 5 个安全锚点：四角 + 中央
#     - 任意两个被选锚点都不会 4 邻接重叠；最多可放 5 个 sprite
#     - 生成灰路径时先保留所有 sprite 的未来脚印，避免路径穿过之后要放 source/distractor 的位置
#   observability:
#     - 目标内部输入态是“简单 key 图案”，输出态是“拷贝来的非对称 3x3 图案”
#     - 因此人眼可以直接看到：不是只改中心，而是整块内部花纹被替换
#   cleanliness:
#     - 每个 sprite 占一个独立 5x5 区域，路径只走 sprite 之外的空白
#     - 不叠层，不共享内部区域，输出只剩两个 destination + 灰路径
#   new_main_assumptions_generator_must_satisfy:
#     - 每个 sprite 都是完整未裁剪的 5x5 方框
#     - destination 中心色必须唯一对应某个 source.border
#     - source 内部图案不能退化成单中心差异；生成器用非对称模板库保证这一点


SPRITE_SIZE = 5
INNER_SIZE = 3
CENTER_OFFSET = 2
PORT_DIST = 3
PORT_BORDER_DIST = 2


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _find_trail_endpoints(
    g: np.ndarray, grey: int, sprites: list[dict],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Find the two grey endpoints of the one-stroke trail under 8-connectivity."""
    h, w = g.shape
    cells = set((int(r), int(c)) for r, c in zip(*np.where(g == grey), strict=False))
    adj: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in cells:
        adj[(r, c)] = [(nr, nc) for nr, nc in neighbors8(r, c) if (nr, nc) in cells]

    deg1 = [c for c, nbs in adj.items() if len(nbs) == 1]
    if len(deg1) >= 2:
        return deg1[0], deg1[1]

    port_cells: set[tuple[int, int]] = set()
    for sp in sprites:
        cr, cc = sp["center_pos"]
        b = int(sp["border"])
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            br, bc = cr + PORT_BORDER_DIST * dr, cc + PORT_BORDER_DIST * dc
            er, ec = cr + PORT_DIST * dr, cc + PORT_DIST * dc
            if (er, ec) in cells and 0 <= br < h and 0 <= bc < w and int(g[br, bc]) == b:
                port_cells.add((er, ec))

    start = deg1[0] if deg1 else next(iter(cells))
    n = len(cells)

    def dfs(cur: tuple[int, int], visited: frozenset) -> list | None:
        if len(visited) == n:
            return [cur] if cur in port_cells else None
        for nb in adj[cur]:
            if nb not in visited:
                result = dfs(nb, visited | {nb})
                if result is not None:
                    return [cur] + result
        return None

    path = dfs(start, frozenset({start}))
    assert path, "could not trace grey cells as one-stroke trail"
    return path[0], path[-1]


def _detect_bordered_sprites(g: np.ndarray, bg: int, grey: int) -> list[dict]:
    """
    Bordered sprite = a 5x5 4-connected component whose outer ring is one border color and
    whose inner 3x3 area is a non-grey, non-bg pattern. The center pixel still serves as
    the lookup key for destination matching.
    """
    mask = (g != bg) & (g != grey)
    comps = components_mask(mask, connectivity=4)
    sprites: list[dict] = []
    ring = np.ones((SPRITE_SIZE, SPRITE_SIZE), dtype=bool)
    ring[1:-1, 1:-1] = False
    for pts in comps:
        if len(pts) != SPRITE_SIZE * SPRITE_SIZE:
            continue
        r0 = int(pts[:, 0].min())
        r1 = int(pts[:, 0].max())
        c0 = int(pts[:, 1].min())
        c1 = int(pts[:, 1].max())
        if (r1 - r0 + 1) != SPRITE_SIZE or (c1 - c0 + 1) != SPRITE_SIZE:
            continue

        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        border_vals = patch[ring]
        uniq_border = np.unique(border_vals)
        if len(uniq_border) != 1:
            continue
        border_color = int(uniq_border[0])
        if border_color in (bg, grey):
            continue

        interior = patch[1:-1, 1:-1].copy()
        if np.any(interior == bg) or np.any(interior == grey):
            continue

        center_pos = (r0 + CENTER_OFFSET, c0 + CENTER_OFFSET)
        sprites.append(
            {
                "pts": pts,
                "border": border_color,
                "center": int(g[center_pos]),
                "center_pos": center_pos,
                "top_left": (r0, c0),
                "interior": interior.copy(),
            }
        )
    return sprites


def _ports_for_sprite(sprite: dict, g: np.ndarray, eps: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    A grey endpoint is a port for this sprite if:
      - it is exactly 3 steps away from the sprite center in a cardinal direction,
      - the cell 2 steps away from the center in that same direction is the border color.
    """
    cr, cc = sprite["center_pos"]
    b = int(sprite["border"])
    ports: list[tuple[int, int]] = []
    for ir, ic in neighbors4(cr, cc):
        dr, dc = ir - cr, ic - cc
        br, bc = cr + PORT_BORDER_DIST * dr, cc + PORT_BORDER_DIST * dc
        er, ec = cr + PORT_DIST * dr, cc + PORT_DIST * dc
        if (er, ec) not in eps:
            continue
        if 0 <= br < g.shape[0] and 0 <= bc < g.shape[1] and int(g[br, bc]) == b:
            ports.append((er, ec))
    return ports


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer whole interior patterns between framed sprites matched by color, keeping only the path-linked ones.

    Input structure:
    - Background, a grey (5) one-stroke trail, and several 5x5 framed sprites.
    - Each sprite has a uniform border color and a full 3x3 interior pattern; the interior center pixel is the lookup key.
    - The two grey trail endpoints touch exactly two destination sprites via ports one cell beyond the frame.

    Rule:
    - Identify the two destination sprites from the grey trail endpoints.
    - For each destination, use its original center pixel color to find the source sprite whose border color matches that key.
    - Copy the source sprite's entire 3x3 interior pattern into the destination interior, preserving the pattern's orientation and layout.
    - Delete all non-destination sprites; keep the grey trail and the two updated destinations.

    Key insight: The path still chooses which sprites survive, but the matched source now donates a full interior stamp rather than a single center color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    grey = 5

    sprites = _detect_bordered_sprites(g, bg, grey)
    assert sprites, "expected at least one bordered sprite in the grid"

    te1, te2 = _find_trail_endpoints(g, grey, sprites)
    trail_eps = {te1, te2}

    eps_pair = [te1, te2]
    opts = {ep: [sp for sp in sprites if ep in _ports_for_sprite(sp, g, trail_eps)] for ep in eps_pair}

    dest: list[dict] = [None, None]  # type: ignore[list-item]
    order = [0, 1] if len(opts[te1]) <= len(opts[te2]) else [1, 0]
    used: set[int] = set()
    for i in order:
        ep = eps_pair[i]
        other = eps_pair[1 - i]
        cands = [sp for sp in opts[ep] if id(sp) not in used]
        best = max(
            cands,
            key=lambda sp: abs(sp["center_pos"][0] - other[0]) + abs(sp["center_pos"][1] - other[1]),
        )
        dest[i] = best
        used.add(id(best))
    assert all(d is not None for d in dest), "expected exactly 2 dest sprites from trail endpoints"
    dest_ids = {id(sp) for sp in dest}

    by_border: dict[int, list[dict]] = {}
    for sp in sprites:
        by_border.setdefault(int(sp["border"]), []).append(sp)

    out = g.copy()

    for sp in sprites:
        if id(sp) in dest_ids:
            continue
        for r, c in sp["pts"]:
            out[r, c] = bg

    keep = np.zeros(out.shape, dtype=bool)
    q: list[tuple[int, int]] = []
    for d in dest:
        for r, c in d["pts"]:
            if out[r, c] != bg and out[r, c] != grey and not keep[r, c]:
                keep[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.pop()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1]:
                if keep[nr, nc]:
                    continue
                if out[nr, nc] == bg or out[nr, nc] == grey:
                    continue
                keep[nr, nc] = True
                q.append((nr, nc))
    drop = (out != bg) & (out != grey) & (~keep)
    out[drop] = bg

    for d in dest:
        want_border = int(d["center"])
        srcs = by_border.get(want_border, [])
        if not srcs:
            continue
        assert len(srcs) == 1, "each destination key should match at most one source border"
        src = srcs[0]
        r0, c0 = d["top_left"]
        out[r0 + 1 : r0 + 4, c0 + 1 : c0 + 4] = src["interior"]

    return out


# --------------- generator (D13) ---------------

GRID_H, GRID_W = 15, 15
BG, GREY = 0, 5
SPRITE_COUNTS = [4, 5]
SOURCE_COUNTS = [1, 2]
PATH_LEN_MIN, PATH_LEN_MAX = 12, 30
MIN_PORT_DIST = 6
MAX_THICKNESS = 3
PATH_WANDER_AWAY_P = 0.5
PATH_WANDER_DIAG_P = 0.6
PATH_MODERATE_RANDOM_P = 0.3
GEN_RETRIES = 400

ANCHORS = [(0, 0), (0, 10), (10, 0), (10, 10), (5, 5)]
PATTERN_MASKS = [
    np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=bool),
    np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
]


def _make_sprite_patch(border_color: int, interior: np.ndarray) -> np.ndarray:
    p = np.full((SPRITE_SIZE, SPRITE_SIZE), border_color, dtype=int)
    p[1:-1, 1:-1] = np.asarray(interior, dtype=int)
    return p


def _place_sprite(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    paste(grid, patch, r0, c0)


def _sprite_footprint(r0: int, c0: int) -> tuple[slice, slice]:
    return slice(r0, r0 + SPRITE_SIZE), slice(c0, c0 + SPRITE_SIZE)


def _paint_reserved(mask: np.ndarray, r0: int, c0: int) -> None:
    rs, cs = _sprite_footprint(r0, c0)
    mask[rs, cs] = True


def _too_thick(r, c, occupied):
    """Check if adding (r,c) would push any cell's 8-degree above MAX_THICKNESS."""
    nbs = [(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr or dc) and (r + dr, c + dc) in occupied]
    if len(nbs) > MAX_THICKNESS:
        return True
    for nr, nc in nbs:
        deg = sum(1 for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr or dc) and (nr + dr, nc + dc) in occupied)
        if deg + 1 > MAX_THICKNESS:
            return True
    return False


def _gen_winding_path(grid, start, end, target_len):
    """Generate a winding 8-connected grey path."""
    path = [start]
    visited = {start}
    cr, cc = start

    def score(m):
        return abs(m[0] - end[0]) + abs(m[1] - end[1])

    for _ in range(target_len + 40):
        if max(abs(cr - end[0]), abs(cc - end[1])) <= 1:
            break

        moves = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = cr + dr, cc + dc
                if (
                    0 <= nr < GRID_H
                    and 0 <= nc < GRID_W
                    and (nr, nc) not in visited
                    and grid[nr, nc] == BG
                    and not _too_thick(nr, nc, visited)
                ):
                    moves.append((nr, nc, dr != 0 and dc != 0))

        if not moves:
            return None

        dist = abs(cr - end[0]) + abs(cc - end[1])
        budget = target_len - len(path)
        diag = [m for m in moves if m[2]]
        away = [m for m in moves if score(m) > dist]

        if budget > dist + 4:
            if away and random.random() < PATH_WANDER_AWAY_P:
                pool = away
            elif diag and random.random() < PATH_WANDER_DIAG_P:
                pool = diag
            else:
                pool = moves
            chosen = random.choice(pool)
        elif budget > dist + 1:
            pool = diag if diag and random.random() < PATH_WANDER_DIAG_P else moves
            if random.random() < PATH_MODERATE_RANDOM_P:
                chosen = random.choice(pool)
            else:
                chosen = min(pool, key=score)
        else:
            best = sorted(moves, key=score)
            top = [m for m in best if score(m) <= score(best[0]) + 1]
            dtop = [m for m in top if m[2]]
            chosen = random.choice(dtop) if dtop else best[0]

        cr, cc = chosen[0], chosen[1]
        path.append((cr, cc))
        visited.add((cr, cc))

    if end not in visited:
        if not _too_thick(end[0], end[1], visited):
            path.append(end)
        else:
            return None

    return path


def _is_8_connected(cells):
    """Check 8-connectivity of a set of cells via BFS."""
    if len(cells) <= 1:
        return True
    cell_set = set(cells)
    start = next(iter(cell_set))
    visited = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nb = (r + dr, c + dc)
                if nb in cell_set and nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
    return len(visited) == len(cell_set)


def _endpoint_deg8(path_cells: set[tuple[int, int]], cell: tuple[int, int]) -> int:
    r, c = cell
    return sum(1 for nr, nc in neighbors8(r, c) if (nr, nc) in path_cells)


def _center_of(top_left: tuple[int, int]) -> tuple[int, int]:
    return top_left[0] + CENTER_OFFSET, top_left[1] + CENTER_OFFSET


def _port_options(top_left: tuple[int, int], reserved: np.ndarray) -> list[tuple[int, int]]:
    cr, cc = _center_of(top_left)
    ports: list[tuple[int, int]] = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        er, ec = cr + PORT_DIST * dr, cc + PORT_DIST * dc
        if 0 <= er < GRID_H and 0 <= ec < GRID_W and not reserved[er, ec]:
            ports.append((er, ec))
    return ports


def _choose_two_distinct_ports(
    dest_positions: list[tuple[int, int]], reserved: np.ndarray
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    opts0 = _port_options(dest_positions[0], reserved)
    opts1 = _port_options(dest_positions[1], reserved)
    if not opts0 or not opts1:
        return None
    pairs = [(p0, p1) for p0 in opts0 for p1 in opts1 if p0 != p1]
    pairs = [pair for pair in pairs if abs(pair[0][0] - pair[1][0]) + abs(pair[0][1] - pair[1][1]) >= MIN_PORT_DIST]
    if not pairs:
        return None
    return random.choice(pairs)


def _make_mask_pattern(mask: np.ndarray, accent: int, fill: int) -> np.ndarray:
    pat = np.full((INNER_SIZE, INNER_SIZE), fill, dtype=int)
    pat[mask] = accent
    return pat


def _simple_dest_pattern(key_color: int, fill: int) -> np.ndarray:
    pat = np.full((INNER_SIZE, INNER_SIZE), fill, dtype=int)
    pat[1, 1] = key_color
    return pat


def _pick_color(exclude: set[int]) -> int:
    candidates = [c for c in range(1, 10) if c not in exclude]
    if not candidates:
        raise ValueError("no colors available")
    return random.choice(candidates)


def _assign_payloads(border_colors, num_sprites):
    remaining = list(range(2, num_sprites))
    if not remaining:
        return None

    max_sources = min(2, len(remaining))
    num_sources = random.choice([n for n in SOURCE_COUNTS if n <= max_sources])
    source_indices = remaining[:num_sources]
    distractor_indices = remaining[num_sources:]

    if num_sources == 1:
        dest_center_colors = [border_colors[source_indices[0]]] * 2
        source_for_dest = [source_indices[0], source_indices[0]]
    else:
        dest_center_colors = [border_colors[si] for si in source_indices]
        source_for_dest = source_indices[:]

    source_interiors: dict[int, np.ndarray] = {}
    for si in source_indices:
        mask = random.choice(PATTERN_MASKS)
        accent = _pick_color({BG, GREY, border_colors[si]})
        fill = _pick_color({BG, GREY, border_colors[si], accent})
        source_interiors[si] = _make_mask_pattern(mask, accent, fill)

    distractor_interiors: dict[int, np.ndarray] = {}
    for di in distractor_indices:
        if random.random() < 0.5:
            mask = random.choice(PATTERN_MASKS)
            accent = _pick_color({BG, GREY, border_colors[di]})
            fill = _pick_color({BG, GREY, border_colors[di], accent})
            distractor_interiors[di] = _make_mask_pattern(mask, accent, fill)
        else:
            key = _pick_color({BG, GREY, border_colors[di]})
            fill = _pick_color({BG, GREY, border_colors[di], key})
            distractor_interiors[di] = _simple_dest_pattern(key, fill)

    dest_input_interiors = []
    dest_output_interiors = []
    for di, want_border in enumerate(dest_center_colors):
        fill = _pick_color({BG, GREY, border_colors[di], want_border})
        dest_input_interiors.append(_simple_dest_pattern(want_border, fill))
        dest_output_interiors.append(source_interiors[source_for_dest[di]].copy())

    return dict(
        source_indices=source_indices,
        distractor_indices=distractor_indices,
        dest_input_interiors=dest_input_interiors,
        dest_output_interiors=dest_output_interiors,
        source_interiors=source_interiors,
        distractor_interiors=distractor_interiors,
    )


def _gen_attempt():
    num_sprites = random.choice(SPRITE_COUNTS)
    chosen_positions = random.sample(ANCHORS, num_sprites)
    border_colors = random.sample([c for c in range(1, 10) if c != GREY], num_sprites)

    payloads = _assign_payloads(border_colors, num_sprites)
    if payloads is None:
        return None

    reserved = np.zeros((GRID_H, GRID_W), dtype=bool)
    for r0, c0 in chosen_positions:
        _paint_reserved(reserved, r0, c0)

    dest_positions = chosen_positions[:2]
    ports = _choose_two_distinct_ports(dest_positions, reserved)
    if ports is None:
        return None

    path_blockers = np.zeros((GRID_H, GRID_W), dtype=int)
    for i, (r0, c0) in enumerate(chosen_positions):
        patch = None
        if i < 2:
            patch = _make_sprite_patch(border_colors[i], payloads["dest_output_interiors"][i])
        else:
            patch = np.full((SPRITE_SIZE, SPRITE_SIZE), 9, dtype=int)
        _place_sprite(path_blockers, patch, r0, c0)

    p0, p1 = ports
    cheb = max(abs(p0[0] - p1[0]), abs(p0[1] - p1[1]))
    lo = max(PATH_LEN_MIN, cheb + 1)
    if lo > PATH_LEN_MAX:
        return None
    target_len = random.randint(lo, PATH_LEN_MAX)
    path_cells = _gen_winding_path(path_blockers, p0, p1, target_len)
    if path_cells is None or not (PATH_LEN_MIN <= len(path_cells) <= PATH_LEN_MAX):
        return None
    if not _is_8_connected(path_cells):
        return None
    path_set = set(path_cells)
    if _endpoint_deg8(path_set, p0) != 1 or _endpoint_deg8(path_set, p1) != 1:
        return None

    output_grid = np.zeros((GRID_H, GRID_W), dtype=int)
    for i in range(2):
        r0, c0 = dest_positions[i]
        patch = _make_sprite_patch(border_colors[i], payloads["dest_output_interiors"][i])
        _place_sprite(output_grid, patch, r0, c0)

    for r, c in path_cells:
        if output_grid[r, c] != BG:
            return None
        output_grid[r, c] = GREY

    input_grid = output_grid.copy()
    for i in range(2):
        r0, c0 = dest_positions[i]
        input_grid[r0 + 1 : r0 + 4, c0 + 1 : c0 + 4] = payloads["dest_input_interiors"][i]

    for j, si in enumerate(payloads["source_indices"]):
        idx = 2 + j
        r0, c0 = chosen_positions[idx]
        patch = _make_sprite_patch(border_colors[idx], payloads["source_interiors"][si])
        _place_sprite(input_grid, patch, r0, c0)

    offset = 2 + len(payloads["source_indices"])
    for j, di in enumerate(payloads["distractor_indices"]):
        idx = offset + j
        r0, c0 = chosen_positions[idx]
        patch = _make_sprite_patch(border_colors[idx], payloads["distractor_interiors"][di])
        _place_sprite(input_grid, patch, r0, c0)

    try:
        out = main(input_grid)
        if np.array_equal(out, output_grid):
            return input_grid
    except Exception:
        pass
    return None


def generate_input():
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {GEN_RETRIES} attempts")
