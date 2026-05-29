# Variant of 195c6913
# Dimension: D13 — 模板和终止块的擦除行为 (transform)
# Change: 输出中不再把模板 2x2 块全部擦成背景；终止块仍然擦除，
# 但每个模板块会改涂成其对应标记轨迹最后一次在 path 上写下的循环色。
#
# Design analysis:
# - Concretization / behavior trace:
#   输入仍然是左上角模板块 + 左侧 col-0 标记 + 一条厚 path + 右下终止块。
#   这里把模板块和标记做成一一对应：两个模板块、两个标记、两个互异循环色。
#   小例子：模板为 [A,B]，上方标记颜色 A、下方标记颜色 B。若 A 标记沿 path
#   最后一次落笔写成 B，则输出里 A 对应的 2x2 模板块整体变成 B；B 标记同理。
#   原逻辑是在模拟前直接把模板块擦成 bg；新逻辑改为先保留模板块、跑完整条轨迹、
#   记录每个标记最后一次 path 落笔色，再回填到对应模板块。
# - Complexity calibration:
#   仅改 D13 的可见结果，不改交替转向、碰撞着色、终止块擦除这些核心机制。
#   为避免“哪个模板块对应哪个标记”不清楚，生成器固定为两个唯一模板色对应两个标记。
#   这样输出左上角会稳定显示两个终态色，视觉上明显区别于原题“整块消失”。
# - Hidden assumptions checked against main():
#   1) 模板色必须唯一，否则按颜色匹配模板块会有歧义；生成器强制两个不同 cycle 色。
#   2) 每个模板色都必须恰好有一个对应 marker；生成器固定两个 marker，颜色分别取两块模板色。
#   3) “最终色”指最后一次写到 path 的循环色，而不是碰撞时写到拐角的 end_color；
#      因此需要单独维护 last_cycle_color，不能直接复用 out[nr,nc]。
#   4) 终止块仍需唯一且与模板色不同，否则 end block 检测会混淆；生成器保持此约束。
# - Placement legality:
#   模板仍放在左上背景区的两块 2x2（cols 1-2, 4-5），marker 仍嵌在 col 0 path 段，
#   两 marker 之间保留可 carve 的间距；终止块仍放右下背景区并与 path 保持至少 1 格隔离。
# - Visual observability / cleanliness:
#   输出仍有整条轨迹和碰撞角点，解题者可以把每个 marker 的输入颜色和模板块颜色对上，
#   再从轨迹末端看到该标记最后写出的循环色，左上角模板块就成了这个“终态摘要”。
#   终止块继续擦除，避免右下角和左上角同时保留大色块造成视觉噪声。
# - Color inference:
#   只使用两种模板/marker 色，且颜色变化由同色 marker 的轨迹直接提示，不依赖颜色 id 排序。

from __future__ import annotations
from collections import Counter
import sys
import numpy as np

if "/home/lixinyu/snap/new/solvers" not in sys.path:
    sys.path.append("/home/lixinyu/snap/new/solvers")

def _assert_path_single_component(g: np.ndarray, path_color: int) -> None:
    """Eval invariant: path_color forms a single 4-connected component.
    Catches generators that build two disconnected bands as a topology shortcut."""
    from arc_utils import label_mask
    _, n_path = label_mask(g == path_color, connectivity=4)
    assert n_path == 1, f"path_color must be a single 4-connected component, got {n_path}"


def _assert_marker_embedded_in_path(
    g: np.ndarray, markers: list[tuple[int, int, int]], path_color: int
) -> None:
    """Eval invariant: each marker cell's vertical neighbors at col 0 are path
    (marker is embedded in a col-0 path segment, not floating on bg).
    All 9 eval markers satisfy this. Catches generators that place markers on
    bg cells with the band only touching col 1."""
    H, _ = g.shape
    for r, c, _ in markers:
        if c != 0:
            continue
        if r > 0:
            assert int(g[r - 1, c]) == path_color, (
                f"marker at ({r},{c}) up-neighbor ({r-1},{c})={int(g[r-1,c])} "
                f"must be path_color {path_color}"
            )
        if r < H - 1:
            assert int(g[r + 1, c]) == path_color, (
                f"marker at ({r},{c}) down-neighbor ({r+1},{c})={int(g[r+1,c])} "
                f"must be path_color {path_color}"
            )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate color-cycling particles on a meandering path and write their terminal cycle state back into the template.

    Input structure:
    - Top-left: a row of 2x2 monochrome blocks encodes the repeating cycle, and each block color also names one marker.
    - Left edge: isolated single-pixel markers sit inside a col-0 path segment and start from the template color they match.
    - Elsewhere: one distinct 2x2 end block marks the collision color; the path itself is the dominant non-background band.

    Rule:
    - Each marker advances straight along the path, repainting visited path cells by stepping through the cycle.
    - On every non-path collision, the collision cell gets the end color, then the particle alternates turn direction CCW, CW, CCW, ...
    - If the scheduled turn is not a path cell, that particle stops.
    - In the output, the end block is erased, while each template block is recolored to the last cycle color written by its matching marker.

    Key insight: The trajectory logic is unchanged; the new information channel is that the template no longer disappears, it summarizes each marker's final cycle phase.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    template_bg = int(g[0, 0])

    # Path is the most frequent color != template_bg.
    counts = Counter(map(int, g.flatten()))
    path_color = int(max([(cnt, int(col)) for col, cnt in counts.items() if int(col) != int(template_bg)])[1])
    _assert_path_single_component(g, path_color)

    # Detect the repeating cycle from 2x2 blocks in the top-left region.
    pattern: list[int] = []
    template_blocks: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int]] = set()
    for r in range(min(6, H - 1)):
        for c in range(min(25, W - 1)):
            if (int(r), int(c)) in seen:
                continue
            v = int(g[int(r), int(c)])
            if v == int(template_bg) or v == int(path_color):
                continue
            if int(g[int(r + 1), int(c)]) == v and int(g[int(r), int(c + 1)]) == v and int(g[int(r + 1), int(c + 1)]) == v:
                pattern.append(int(v))
                template_blocks.append((int(r), int(c), int(v)))
                for rr, cc in [(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)]:
                    seen.add((int(rr), int(cc)))

    # Detect end color block (2x2).
    end_color = None
    end_pos = None
    pat_set = set(map(int, pattern))
    for r in range(H - 1):
        for c in range(W - 1):
            v = int(g[int(r), int(c)])
            if v == int(template_bg) or v == int(path_color) or v in pat_set:
                continue
            if int(g[int(r + 1), int(c)]) == v and int(g[int(r), int(c + 1)]) == v and int(g[int(r + 1), int(c + 1)]) == v:
                end_color = int(v)
                end_pos = (int(r), int(c))
                break
        if end_pos is not None:
            break

    # Detect start markers: isolated pattern-color pixels adjacent to the path.
    markers: list[tuple[int, int, int]] = []
    for r in range(H):
        for c in range(W):
            v = int(g[int(r), int(c)])
            if v not in pat_set:
                continue
            same = 0
            has_path = False
            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < H and 0 <= cc < W:
                    if int(g[rr, cc]) == v:
                        same += 1
                    if int(g[rr, cc]) == int(path_color):
                        has_path = True
            if same == 0 and has_path:
                markers.append((int(r), int(c), int(v)))

    _assert_marker_embedded_in_path(g, markers, int(path_color))

    out = g.copy()
    # End block is still erased in this variant; template blocks are recolored later.
    if end_pos is not None:
        er, ec = end_pos
        out[int(er) : int(er + 2), int(ec) : int(ec + 2)] = int(template_bg)

    # Initial direction preference (matches examples).
    dir_order = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # R, D, L, U

    per_marker_cells: list[set[tuple[int, int]]] = []
    final_by_color: dict[int, int] = {}
    for sr, sc, sv in markers:
        init = None
        for dr, dc in dir_order:
            rr = int(sr + dr)
            cc = int(sc + dc)
            if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(path_color):
                init = (int(dr), int(dc))
                break
        if init is None or not pattern:
            per_marker_cells.append(set())
            continue

        r, c = int(sr), int(sc)
        dr, dc = init
        idx = int(pattern.index(int(sv))) if int(sv) in pattern else 0
        last_cycle_color = int(sv)
        hits = 0  # collision count so far
        visited: set[tuple[int, int, int, int, int, int]] = set()
        my_cells: set[tuple[int, int]] = set()

        for _ in range(H * W * 4):
            state = (int(r), int(c), int(dr), int(dc), int(idx), int(hits & 1))
            if state in visited:
                break
            visited.add(state)

            nr = int(r + dr)
            nc = int(c + dc)
            if not (0 <= nr < H and 0 <= nc < W):
                break

            if int(g[nr, nc]) == int(path_color):
                idx = int((idx + 1) % len(pattern))
                last_cycle_color = int(pattern[idx])
                out[nr, nc] = int(last_cycle_color)
                my_cells.add((int(nr), int(nc)))
                r, c = nr, nc
                continue

            # Collision with non-path.
            if int(g[nr, nc]) == int(template_bg) and end_color is not None:
                out[nr, nc] = int(end_color)

            # Strict alternating turn: no fallback if blocked.
            left = (-int(dc), int(dr))  # CCW
            right = (int(dc), -int(dr))  # CW
            tdr, tdc = left if (int(hits) % 2 == 0) else right
            rr = int(r + tdr)
            cc = int(c + tdc)
            if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(path_color):
                dr, dc = int(tdr), int(tdc)
                hits += 1
                continue
            break

        per_marker_cells.append(my_cells)
        final_by_color[int(sv)] = int(last_cycle_color)

    # Trajectories must be pairwise disjoint (bg islands separate them).
    for i in range(len(per_marker_cells)):
        for j in range(i + 1, len(per_marker_cells)):
            overlap = per_marker_cells[i] & per_marker_cells[j]
            assert not overlap, (
                f"marker trajectories {i},{j} overlap on {len(overlap)} cell(s)"
            )

    for r, c, v in template_blocks:
        fill = int(final_by_color.get(int(v), int(v)))
        out[int(r) : int(r + 2), int(c) : int(c + 2)] = fill

    return out

# --------------- generator (markers-first SAv4_mine: weighted sampling + fjord→island, 58% n=2) ---------------
# Markers-first construction: pick markers, build band with lobe to embed them,
# optionally carve fjord→island to create INTERIOR separator (25%), place template
# + end_block, run solver as oracle.

import random
from arc_utils import label_mask


def _build_band(H: int, W: int, lobe_rows: list[int] | None = None) -> np.ndarray:
    """Return a boolean mask for a thick meandering anti-diagonal band.

    Row 0's path span is in the rightmost cols (c_left[0] >= W-6, c_right[0] = W-1).
    Row H-1's path span is in the leftmost cols (c_left[H-1] = 0, c_right[H-1] <= 5).
    The band widens toward the middle and is 4-connected row-by-row.

    If `lobe_rows` is given, the band extends to col 0 across the CONTIGUOUS
    span [min(lobe_rows)-1, max(lobe_rows)+1] (inclusive), forming a single
    lobe that covers each marker's col-0 neighborhood. Outside that span,
    c_left gradients up at 2 cols/row. Baked into the parametric centerline
    so jitter/smoothing treat the lobe as part of the band's natural shape.

    Template area (rows 1-3) stays clear (c_left >= 12) so 2x2 blocks fit.
    """
    # Parametric centerline (anti-diagonal) and halfwidth profile.
    # Right-edge variation: bottom endpoint of centerline can shift right from
    # 0 up to W//4, which pushes c_right at lower rows rightward (band's right
    # edge stays higher longer instead of always collapsing to the left corner).
    # c_left is protected by marker-lobe + anchor, so this only expands right
    # edge — matches user's "only shift right, not left" requirement.
    bot_center = random.randint(0, W // 4)
    centers = np.linspace(W - 1, bot_center, H).astype(float)
    half_w = np.array([
        0.5 * (3 + (1.0 - abs(2.0 * r / max(1, H - 1) - 1.0)) * (W - 6))
        for r in range(H)
    ])

    c_left = np.zeros(H, dtype=int)
    c_right = np.zeros(H, dtype=int)

    # Smooth jitter: kernel=3 moving-average keeps adjacent-row jitter correlated
    # (independent ±3 jitter otherwise creates 1-cell bg notches in flat regions).
    cc_jit = np.convolve(
        np.array([random.randint(-3, 3) for _ in range(H)], dtype=float),
        np.ones(3) / 3, mode='same')
    w_jit = np.convolve(
        np.array([random.randint(-3, 3) for _ in range(H)], dtype=float),
        np.ones(3) / 3, mode='same')

    for r in range(H):
        c_left_param = centers[r] - half_w[r]
        if c_left_param <= 0:
            # "Flat left edge" row: band reaches/passes col 0 (inside lobe span
            # or natural band bottom). No left-edge jitter to avoid 1-cell notches.
            c_left[r] = 0
            c_right[r] = int(round(centers[r] + cc_jit[r] + half_w[r] + w_jit[r] / 2))
        else:
            # Slope row: legitimate c_left > 0; symmetric jitter.
            w = max(2, min(W, int(round(2 * half_w[r] + w_jit[r]))))
            cc = centers[r] + cc_jit[r]
            c_left[r] = int(round(cc - w / 2))
            c_right[r] = c_left[r] + w - 1

    # Lobe constraint: c_left = 0 from above the upper marker all the way to
    # H-1 (band reaches col 0 across the lower portion). lmax is set to H-1
    # rather than max(lobe_rows)+1 so the gradient between max(lobe_rows)+1 and
    # the bottom anchor doesn't form a spurious V-shaped bg notch — anchor
    # already forces c_left[H-1]=0, so the lobe should extend to meet it
    # naturally. Fjord carve creates the separator inside this lobe band.
    if lobe_rows:
        lmin = min(lobe_rows) - 1
        lmax = H - 1
        for r in range(H):
            d_outside = max(0, lmin - r, r - lmax)
            c_left[r] = min(c_left[r], 2 * d_outside)

    # Template area (rows 1-3): force c_left >= 12 so 2x2 blocks at cols {1,4,7,10}
    # have bg to sit on. Post-jitter because jitter could otherwise push c_left <12.
    for r in range(1, min(4, H)):
        c_left[r] = max(c_left[r], 12)
        if c_right[r] < c_left[r]:
            c_right[r] = c_left[r]

    # Anchor top to right corner; bottom c_left must be 0 but c_right takes
    # the shifted parametric (bot_center-dependent), clamped to grid only.
    c_right[0] = W - 1
    c_left[0] = max(c_left[0], W - 6)
    c_left[H - 1] = 0
    c_right[H - 1] = max(5, min(W - 1, c_right[H - 1]))

    # Remove 1-row c_left spikes in slope region (local maxima sandwiched by
    # lower neighbors). Flat rows (c_left=0) aren't affected since they're
    # already minimal.
    for _ in range(2):
        for r in range(1, H - 1):
            if c_left[r] > c_left[r - 1] and c_left[r] > c_left[r + 1]:
                c_left[r] = max(c_left[r - 1], c_left[r + 1])

    # Enforce row-to-row 4-connectivity + mostly-monotone leftward motion.
    for r in range(1, H):
        c_left[r] = min(c_left[r], c_left[r - 1] + 2)
        c_right[r] = min(c_right[r], c_right[r - 1] + 2)
        if c_left[r] > c_right[r - 1]:
            c_left[r] = c_right[r - 1]
        if c_right[r] < c_left[r - 1]:
            c_right[r] = c_left[r - 1]
        if c_right[r] < c_left[r]:
            c_right[r] = c_left[r]

    c_left = np.clip(c_left, 0, W - 1)
    c_right = np.clip(c_right, 0, W - 1)
    c_right[0] = W - 1
    c_left[H - 1] = 0

    mask = np.zeros((H, W), dtype=bool)
    for r in range(H):
        mask[r, c_left[r]:c_right[r] + 1] = True
    return mask


def _carve_fjord(mask: np.ndarray, r1: int, r2: int) -> bool:
    """Carve a bg 'fjord' biting into the band from col 0, between marker rows.

    The fjord is 2-3 rows tall at col 0 (strictly between r1 and r2), and
    extends rightward. As we go right, the fjord's top row decreases (slopes
    up-right, parallel to the band direction). The fjord stops before breaking
    the band's 4-connectivity.

    Mutates `mask` in place. Returns True if a valid carve was made; False otherwise
    (caller must use a pre-carve copy to restore on failure).
    """
    H, W = mask.shape
    sep_h = random.randint(2, 3)
    # Start rows in col 0 strictly between markers.
    sr_min = r1 + 2
    sr_max = r2 - 1 - sep_h
    if sr_max < sr_min:
        return False
    sr = random.randint(sr_min, sr_max)

    # Determine max reach: stop before disconnecting band. We probe one col at a time.
    cur_top = sr
    last_good = mask.copy()
    made_carve = False
    carved_cols = 0
    for k in range(W - 1):
        # Try to carve [cur_top, cur_top + sep_h - 1] at column k.
        trial = mask.copy()
        for dr in range(sep_h):
            rr = cur_top + dr
            if 0 <= rr < H:
                trial[rr, k] = False
        _, n_trial = label_mask(trial, connectivity=4)
        if n_trial != 1:
            break
        # Accept this column.
        np.copyto(mask, trial)
        np.copyto(last_good, trial)
        made_carve = True
        carved_cols += 1
        # Move top upward occasionally (slope up-right, parallel to anti-diagonal band).
        # Lower probability = fjord extends further horizontally before climbing.
        if random.random() < 0.3 and cur_top > 0:
            cur_top -= 1
    return made_carve


def _gen_attempt() -> np.ndarray | None:
    H = W = random.randint(20, 30)
    # This variant needs an unambiguous marker↔template correspondence, so we
    # always use two unique template colors and two matching markers.
    n_markers = 2

    palette = list(range(10))
    random.shuffle(palette)
    bg = palette[0]
    path_color = palette[1]
    end_color = palette[2]
    n_blocks = 2
    cycle_colors = palette[3:3 + n_blocks]

    # Markers-first: pick marker rows BEFORE building band. Band is built with
    # a lobe extending to col 0 across the marker-row neighborhood, baking
    # markers into the band's parametric curve.
    # Keep the original two-marker topology because it gives disjoint channels
    # and makes both template recolorings visible.
    r1 = random.randint(H // 3, H // 2)
    r2 = random.randint(max(r1 + 5, 2 * H // 3), H - 4)
    markers = [r1, r2]

    mask = _build_band(H, W, lobe_rows=markers)
    # g[0,0] must be bg (template_bg per solver). Band should not touch top-left corner.
    if mask[0, 0]:
        return None
    _, n_path = label_mask(mask, connectivity=4)
    if n_path != 1:
        return None
    if not mask[0, :].any() or not mask[:, W - 1].any():
        return None
    if not mask[H - 1, :].any() or not mask[:, 0].any():
        return None
    # Marker embedding should be guaranteed by forced_left_rows, but verify.
    for r in markers:
        if not (mask[r, 0] and mask[r - 1, 0] and mask[r + 1, 0]):
            return None

    trial = mask.copy()
    if not _carve_fjord(trial, r1, r2):
        return None
    # Keep both left-edge fjords and interior islands so the old topology
    # variety remains, while the transform change stays the only new concept.
    use_interior = (random.random() < 0.25)
    if use_interior:
        K = random.randint(2, 4)
        pre_carve_cells = mask & ~trial
        for c in range(min(K, W)):
            trial[:, c] |= pre_carve_cells[:, c]
    _, nc = label_mask(trial, connectivity=4)
    if nc != 1:
        return None
    if not trial[0, :].any() or not trial[:, W - 1].any():
        return None
    if not trial[H - 1, :].any() or not trial[:, 0].any():
        return None
    for r in markers:
        if not (trial[r, 0] and trial[r - 1, 0] and trial[r + 1, 0]):
            return None
    mask = trial

    _, n_bg = label_mask(~mask, connectivity=4)
    if n_bg < 2:
        return None

    # Template placement stays in the original top-left lane, but with exactly
    # two blocks so each one maps to one marker by color.
    template_row = random.choice([1, 2])
    block_cols = [1, 4]
    if block_cols[-1] + 1 >= W:
        return None
    for bc in block_cols:
        for dr in range(2):
            for dc in range(2):
                rr, cc = template_row + dr, bc + dc
                if rr >= H or cc >= W or mask[rr, cc]:
                    return None

    g = np.full((H, W), bg, dtype=int)
    g[mask] = path_color

    for i, bc in enumerate(block_cols):
        g[template_row:template_row + 2, bc:bc + 2] = cycle_colors[i]

    # End block: 2x2 in bottom-right bg region, isolated from path by >=1 cell.
    end_candidates = []
    for r in range(H // 2, H - 1):
        for c in range(W // 2, W - 1):
            if not (g[r:r + 2, c:c + 2] == bg).all():
                continue
            r0, r1_ = max(0, r - 1), min(H, r + 3)
            c0, c1_ = max(0, c - 1), min(W, c + 3)
            if (g[r0:r1_, c0:c1_] == path_color).any():
                continue
            end_candidates.append((r, c))
    if not end_candidates:
        return None
    er, ec = random.choice(end_candidates)
    g[er:er + 2, ec:ec + 2] = end_color

    for r, color in zip(markers, cycle_colors):
        g[r, 0] = int(color)

    # Path must remain single 4-conn after marker overwrite.
    _, n_path_after = label_mask(g == path_color, connectivity=4)
    if n_path_after != 1:
        return None

    # g[0,0] must be bg (== template_bg in solver).
    if g[0, 0] != bg:
        return None

    # path_color must be most-frequent non-bg color (solver's detection).
    # NOTE: solver uses template_bg = g[0,0], NOT most-common. So bg does NOT
    # need to be globally majority — only path must be max non-bg.
    cnt = Counter(map(int, g.flatten()))
    non_bg = sorted([(c, col) for col, c in cnt.items() if col != bg], reverse=True)
    if not non_bg or non_bg[0][1] != path_color:
        return None

    # Oracle: solver enforces marker embedding & disjoint trajectories.
    try:
        out = main(g)
    except Exception:
        return None

    # D13 should be visible: at least one template block must actually change.
    changed_blocks = 0
    for bc, color in zip(block_cols, cycle_colors):
        if not np.all(out[template_row:template_row + 2, bc:bc + 2] == int(color)):
            changed_blocks += 1
    if changed_blocks == 0:
        return None

    return g


def generate_input() -> np.ndarray:
    for _ in range(300):
        try:
            g = _gen_attempt()
        except Exception:
            g = None
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 300 attempts")
