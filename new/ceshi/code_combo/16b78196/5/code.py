# Combo variant of 16b78196 (index 5)
# combo_spec: D2/2+D12/0
# Name: 中空框架+部分腔体填充
#
# Design analysis
# ---------------
# a) 具体化与行为追踪
# - 原题 piece 基本是实心连通块。放回输出后，后一块需要把前一块 bbox 里的黑色条带空腔全部补满。
# - 本融合版把 piece 改成“中空框架”风格：它们不是实心块，而是沿固定条带生长的单色 hollow/U-frame。
#   视觉上每块都有自带内腔。后一块只需要覆盖前一块“结构性内腔”的至少 50%，然后继续朝远离 bar
#   的方向生长；不要求补满。
# - 关键差异是“什么算前一块的腔体”。这里不能再把 bbox 里的所有背景都算进去，否则会把框架 opening
#   外侧的条带背景误判成应被填充的 cavity。solver 改为：把 head 的局部空白按“相对 bar 一侧的边界可达性”
#   分成两类，只把不能从 bar-facing 边界到达的那部分视为框架内腔。
# - 一个具体例子（宽 4、向上生长）：
#   root = 1001 / 1001 / 1111。其 bbox 内黑格有 4 个，它们都不能从底边到达，因此都是 frame cavity。
#   next = 1111 / 1001 / 0110，向上平移 2 行后只覆盖 root cavity 的上半 2 格，覆盖率 2/4=50%，
#   同时又伸到更外侧。输出中 root 的另外 2 格洞仍保留，可肉眼看出“部分填充”。
#
# b) 复杂度校准
# - 如果只放宽成“>=50%”而 piece 仍常常是实心块，视觉上会退化回原题；如果 piece 是闭合方框，
#   下一块又根本进不去前一块的洞里。这里采用 open-away-from-bar 的 hollow/U-frame，既保留“中空”
#   视觉，又让下一块能插进前一块 cavity。
# - 为了减少回溯歧义，两个 chain 固定使用不同 strip width（4 与 5）。这样 piece 宽度本身就是强约束，
#   其他链的 piece 无法误接到错误 notch 上。
#
# c) main() 隐式假设审查
# - 原代码 `cavity = bbox 内所有 bg`：在融合后失效，因为框架 opening 外的背景不应被当成内腔。
# - 原代码 continuation 只检查 `touches + must_cover=全部 cavity`：放宽到 50% 后，如果不加“向外延伸”
#   剪枝，会出现 piece 只在原 bbox 里游走的假解，因此保留 outward check。
# - 原代码默认 piece 没有自带洞；现在 head 自身的局部空白会继承到下一步，因此必须从 piece 几何中推断
#   结构性 cavity，而不是从当前画布做全 bbox 扫描。
#
# d) 放置合法性
# - generator 先构造 output，再把 piece 打散成 input。
# - 所有 piece 都保持单色、4 连通、不旋转于 input 中；输出中的 chain 仅靠平移复原。
# - 两条 chain 分居 bar 两侧，used notch 从 root 与 bar 的重叠部分反推，decoy notch 额外从 bar 边界雕刻。
#
# e) 视觉可观察性
# - 每个样例都同时出现：
#   1. hollow frame / U-frame source piece；
#   2. continuation 只补前一块 cavity 的一半，留下明显残洞。
# - 这些信号都纯几何可见，不依赖颜色数值。
#
# f) 可视化整洁度
# - 只保留 bar、两条 chain、若干 decoy notch。两条 chain 放在 bar 的相对两侧，避免叠加。
# - piece 颜色各异，但颜色不参与匹配规则，作用只是分隔对象。
#
# g) 变化的可推断性
# - 人类可从少量样例看出：散落 piece 都像 hollow frame；输出里后一块并没有补满前一块的洞，
#   但仍被视为后继并继续朝外延伸。
#
# h) 量化验证指标
# - `coverage_ratio = covered_structural_cavity / structural_cavity_size`。本生成器固定让每条 chain 的
#   continuation ratio = 0.5。
# - `frame_cavity_size > 0` 对全部 piece 成立，确保“中空框架”不是口头设定。
# - 额外对比原 solver：若原 solver 在这些输入上常与本 solver 输出不同，则说明融合行为没有退化回原题。
#
# i) 新假设与 generator 保证
# - solver 假设 chain piece 的结构性 cavity 可通过“空白是否能从 bar-facing 边界到达”推断。
#   generator 专门使用这类 open-away-from-bar 的 frame 模板，因此该假设稳定成立。
# - solver 假设错误宽度的 piece 不能接入某条链；generator 用固定 strip width 4/5 的双链结构保证这一点。

from __future__ import annotations

import random
import sys
from collections import deque
from itertools import combinations
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
_SOLVERS = _ROOT / "solvers"
if str(_SOLVERS) not in sys.path:
    sys.path.append(str(_SOLVERS))

from arc_utils import components_mask, find_components, get_bg

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _normalize(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    seen = {next(iter(cells))}
    stack = [next(iter(cells))]
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            nb = (r + dr, c + dc)
            if nb in cells and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(cells)


def _bbox_of(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _shape_from_cells(cells: set[tuple[int, int]]) -> dict[str, object]:
    r0, c0, r1, c1 = _bbox_of(cells)
    rel = sorted((r - r0, c - c0) for r, c in cells)
    return {"rel": rel, "rel_set": set(rel), "h": r1 - r0 + 1, "w": c1 - c0 + 1}


def _pixels(shape: dict[str, object], top: int, left: int) -> set[tuple[int, int]]:
    return {(top + lr, left + lc) for lr, lc in shape["rel"]}


def _bar_notches(bar_mask: np.ndarray, bar_r0: int, bar_c0: int) -> list[dict[str, object]]:
    empty = ~bar_mask
    h, w = empty.shape
    notches: list[dict[str, object]] = []
    for pts in components_mask(empty, connectivity=4):
        rr = pts[:, 0]
        cc = pts[:, 1]
        top = int((rr == 0).sum())
        bottom = int((rr == h - 1).sum())
        left = int((cc == 0).sum())
        right = int((cc == w - 1).sum())
        if top + bottom + left + right == 0:
            continue
        hole_gl = {(bar_r0 + int(r), bar_c0 + int(c)) for r, c in pts}
        side, mouth_n = max(
            [(name, cnt) for name, cnt in [("top", top), ("bottom", bottom), ("left", left), ("right", right)] if cnt > 0],
            key=lambda x: x[1],
        )
        axis = "cols" if side in ("top", "bottom") else "rows"
        notches.append({"axis": axis, "hole": hole_gl, "mouth_n": mouth_n, "side": side})
    return notches


def _touches(pts: set[tuple[int, int]], other: set[tuple[int, int]]) -> bool:
    for r, c in pts:
        for dr, dc in DIR4:
            if (r + dr, c + dc) in other:
                return True
    return False


def _oriented_base_cells(width: int, kind: str) -> tuple[set[tuple[int, int]], int, int]:
    if width == 4 and kind == "root":
        rows = ["1001", "1001", "1111"]
    elif width == 4 and kind == "follow":
        rows = ["1111", "0101", "0110"]
    elif width == 5 and kind == "root":
        rows = ["10001", "10001", "11111"]
    elif width == 5 and kind == "follow":
        rows = ["11111", "01001", "01110"]
    else:
        raise ValueError("unsupported template")
    cells = {(r, c) for r, row in enumerate(rows) for c, ch in enumerate(row) if ch == "1"}
    return cells, len(rows), len(rows[0])


def _rotate_ccw(cells: set[tuple[int, int]], h: int, w: int) -> tuple[set[tuple[int, int]], int, int]:
    return {(w - 1 - c, r) for r, c in cells}, w, h


def _rotate_cw(cells: set[tuple[int, int]], h: int, w: int) -> tuple[set[tuple[int, int]], int, int]:
    return {(c, h - 1 - r) for r, c in cells}, w, h


def _flip_ud(cells: set[tuple[int, int]], h: int, w: int) -> tuple[set[tuple[int, int]], int, int]:
    return {(h - 1 - r, c) for r, c in cells}, h, w


def _orient_template(width: int, kind: str, axis: str, sign: int) -> dict[str, object]:
    cells, h, w = _oriented_base_cells(width, kind)
    if axis == "cols" and sign < 0:
        pass
    elif axis == "cols" and sign > 0:
        cells, h, w = _flip_ud(cells, h, w)
    elif axis == "rows" and sign < 0:
        cells, h, w = _rotate_ccw(cells, h, w)
    else:
        cells, h, w = _rotate_cw(cells, h, w)
    return _shape_from_cells(cells)


def _bar_facing_boundary(shape: dict[str, object], axis: str, sign: int) -> list[tuple[int, int]]:
    h = int(shape["h"])
    w = int(shape["w"])
    if axis == "cols":
        r = h - 1 if sign < 0 else 0
        return [(r, c) for c in range(w)]
    c = w - 1 if sign < 0 else 0
    return [(r, c) for r in range(h)]


def _structural_cavity_local(shape: dict[str, object], axis: str, sign: int) -> set[tuple[int, int]]:
    h = int(shape["h"])
    w = int(shape["w"])
    occ = set(shape["rel"])
    empty = {(r, c) for r in range(h) for c in range(w) if (r, c) not in occ}
    if not empty:
        return set()
    q = deque([rc for rc in _bar_facing_boundary(shape, axis, sign) if rc in empty])
    reachable = set(q)
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nb = (r + dr, c + dc)
            if 0 <= nb[0] < h and 0 <= nb[1] < w and nb in empty and nb not in reachable:
                reachable.add(nb)
                q.append(nb)
    return empty - reachable


def _structural_cavity_global(shape: dict[str, object], top: int, left: int, axis: str, sign: int) -> set[tuple[int, int]]:
    return {(top + r, left + c) for r, c in _structural_cavity_local(shape, axis, sign)}


def _placement_pts(
    shape: dict[str, object],
    top: int,
    left: int,
    occ: set[tuple[int, int]],
    H: int,
    W: int,
    must_cover: set[tuple[int, int]] | None = None,
    strip: tuple[str, int, int] | None = None,
) -> set[tuple[int, int]] | None:
    if top < 0 or left < 0 or top + int(shape["h"]) > H or left + int(shape["w"]) > W:
        return None
    if strip is not None:
        axis, A, B = strip
        if axis == "cols":
            if left != A or int(shape["w"]) != B - A + 1:
                return None
        else:
            if top != A or int(shape["h"]) != B - A + 1:
                return None
    pts = _pixels(shape, top, left)
    if pts & occ:
        return None
    if must_cover is not None and not (must_cover <= pts):
        return None
    return pts


def _aligned_positions(axis: str, A: int, cover: set[tuple[int, int]], shape: dict[str, object]):
    if axis == "cols":
        left = A
        tops = set()
        for rN, cN in cover:
            for lr, lc in shape["rel"]:
                if cN == left + lc:
                    tops.add(rN - lr)
        for top in tops:
            yield top, left
    else:
        top = A
        lefts = set()
        for rN, cN in cover:
            for lr, lc in shape["rel"]:
                if rN == top + lr:
                    lefts.add(cN - lc)
        for left in lefts:
            yield top, left


def _hole_positions(hole: set[tuple[int, int]], shape: dict[str, object]) -> set[tuple[int, int]]:
    out = set()
    for rH, cH in hole:
        for lr, lc in shape["rel"]:
            out.add((rH - lr, cH - lc))
    return out


def _grow_sign_for_root(root_pts: set[tuple[int, int]], axis: str, bar_r0: int, bar_c0: int) -> int:
    if axis == "cols":
        return -1 if min(r for r, _ in root_pts) < bar_r0 else 1
    return -1 if min(c for _, c in root_pts) < bar_c0 else 1


def _extends_outward(chain: dict[str, object], pts: set[tuple[int, int]]) -> bool:
    head = chain["head"]
    sign = int(chain["grow_sign"])
    if chain["axis"] == "cols":
        head_min = int(head["top"])
        head_max = head_min + int(head["shape"]["h"]) - 1
        new_min = min(r for r, _ in pts)
        new_max = max(r for r, _ in pts)
        return new_min < head_min if sign < 0 else new_max > head_max
    head_min = int(head["left"])
    head_max = head_min + int(head["shape"]["w"]) - 1
    new_min = min(c for _, c in pts)
    new_max = max(c for _, c in pts)
    return new_min < head_min if sign < 0 else new_max > head_max


def _outward_gain(chain: dict[str, object], pts: set[tuple[int, int]]) -> int:
    head = chain["head"]
    sign = int(chain["grow_sign"])
    if chain["axis"] == "cols":
        head_min = int(head["top"])
        head_max = head_min + int(head["shape"]["h"]) - 1
        new_min = min(r for r, _ in pts)
        new_max = max(r for r, _ in pts)
        return (head_min - new_min) if sign < 0 else (new_max - head_max)
    head_min = int(head["left"])
    head_max = head_min + int(head["shape"]["w"]) - 1
    new_min = min(c for _, c in pts)
    new_max = max(c for _, c in pts)
    return (head_min - new_min) if sign < 0 else (new_max - head_max)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = get_bg(g)

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    comps.sort(key=lambda d: -d["area"])
    assert comps

    bar = comps[0]
    pieces_raw = comps[1:]

    out = np.full((H, W), int(bg), dtype=int)
    bar_r0, bar_c0, bar_r1, bar_c1 = map(int, bar["bbox"])
    bar_h, bar_w = bar_r1 - bar_r0 + 1, bar_c1 - bar_c0 + 1
    bar_color = int(bar["color"])
    bar_occ = {(int(r), int(c)) for r, c in bar["coords"]}
    bar_mask = np.zeros((bar_h, bar_w), dtype=bool)
    for r, c in bar_occ:
        bar_mask[r - bar_r0, c - bar_c0] = True
        out[r, c] = bar_color

    pieces: list[dict[str, object]] = []
    for p in pieces_raw:
        r0, c0 = map(int, p["bbox"][:2])
        shape = _shape_from_cells({(int(r), int(c)) for r, c in p["coords"]})
        pieces.append({"color": int(p["color"]), "shape": shape, "orig": (r0, c0)})

    notches = _bar_notches(bar_mask, bar_r0, bar_c0)

    root_cands = []
    for ni, notch in enumerate(notches):
        axis = str(notch["axis"])
        for pi, piece in enumerate(pieces):
            shape = piece["shape"]
            for top, left in _hole_positions(notch["hole"], shape):
                pts = _placement_pts(shape, top, left, bar_occ, H, W, must_cover=notch["hole"])
                if pts is None:
                    continue
                sign = _grow_sign_for_root(pts, axis, bar_r0, bar_c0)
                cavity = _structural_cavity_local(shape, axis, sign)
                if not cavity:
                    continue
                if axis == "cols":
                    A, B = int(left), int(left + int(shape["w"]) - 1)
                else:
                    A, B = int(top), int(top + int(shape["h"]) - 1)
                mv = abs(int(top) - int(piece["orig"][0])) + abs(int(left) - int(piece["orig"][1]))
                root_cands.append(
                    (
                        (int(mv), -int(notch["mouth_n"]), -len(cavity)),
                        {
                            "ni": ni,
                            "pi": pi,
                            "top": int(top),
                            "left": int(left),
                            "axis": axis,
                            "A": A,
                            "B": B,
                            "shape": shape,
                            "grow_sign": sign,
                        },
                    )
                )

    root_cands.sort(key=lambda x: x[0])
    cand_roots = [pair[1] for pair in root_cands[:24]]

    best_out = out.copy()
    best_key = None

    def search(out_grid: np.ndarray, occ_set: set[tuple[int, int]], chains: list[dict[str, object]], remaining: set[int]) -> bool:
        nonlocal best_out, best_key
        placed_cnt = len(pieces) - len(remaining)
        key = (placed_cnt, -len(remaining), -len(occ_set))
        if best_key is None or key > best_key:
            best_key = key
            best_out = out_grid.copy()
        if not remaining:
            return True

        def gen_extend_moves(chain: dict[str, object], remaining_set: set[int]):
            axis = str(chain["axis"])
            A, B = int(chain["A"]), int(chain["B"])
            head = chain["head"]
            sign = int(chain["grow_sign"])
            head_pts = _pixels(head["shape"], int(head["top"]), int(head["left"]))
            cavity = _structural_cavity_global(head["shape"], int(head["top"]), int(head["left"]), axis, sign)
            cavity = {rc for rc in cavity if out_grid[rc[0], rc[1]] == bg}
            if not cavity:
                return []
            need = (len(cavity) + 1) // 2

            moves = []
            for pi in remaining_set:
                shape = pieces[pi]["shape"]
                seen = set()
                for top, left in _aligned_positions(axis, A, cavity, shape):
                    if (top, left) in seen:
                        continue
                    seen.add((top, left))
                    pts = _placement_pts(shape, top, left, occ_set, H, W, strip=(axis, A, B))
                    if pts is None:
                        continue
                    covered = len(pts & cavity)
                    if covered < need:
                        continue
                    if not _touches(pts, head_pts):
                        continue
                    if not _extends_outward(chain, pts):
                        continue
                    mv = abs(int(top) - int(pieces[pi]["orig"][0])) + abs(int(left) - int(pieces[pi]["orig"][1]))
                    score = (int(covered == len(cavity)), -covered, -_outward_gain(chain, pts), mv)
                    moves.append((score, pi, int(top), int(left), pts, shape))
            moves.sort(key=lambda item: item[0])
            return moves

        for chain in chains:
            moves = gen_extend_moves(chain, remaining)
            if not moves:
                continue
            for _score, pi, top, left, pts, shape in moves[:10]:
                for r, c in pts:
                    out_grid[r, c] = int(pieces[pi]["color"])
                old_head = chain["head"]
                chain["head"] = {"top": int(top), "left": int(left), "shape": shape}
                occ2 = set(occ_set) | set(pts)
                rem2 = set(remaining)
                rem2.remove(pi)
                if search(out_grid, occ2, chains, rem2):
                    return True
                chain["head"] = old_head
                for r, c in pts:
                    out_grid[r, c] = int(bg)
                for r, c in bar_occ:
                    out_grid[r, c] = bar_color
        return False

    for k in range(1, min(3, len(cand_roots)) + 1):
        for roots_sel in combinations(range(len(cand_roots)), k):
            used_p = set()
            used_n = set()
            occ0 = set(bar_occ)
            out0 = np.full((H, W), int(bg), dtype=int)
            for r, c in bar_occ:
                out0[r, c] = bar_color
            chains = []
            ok = True
            for idx in roots_sel:
                root = cand_roots[idx]
                if root["pi"] in used_p or root["ni"] in used_n:
                    ok = False
                    break
                pts = _pixels(root["shape"], int(root["top"]), int(root["left"]))
                if pts & occ0:
                    ok = False
                    break
                used_p.add(int(root["pi"]))
                used_n.add(int(root["ni"]))
                for r, c in pts:
                    out0[r, c] = int(pieces[int(root["pi"])]["color"])
                occ0 |= pts
                chains.append(
                    {
                        "axis": root["axis"],
                        "A": root["A"],
                        "B": root["B"],
                        "grow_sign": root["grow_sign"],
                        "head": {"top": root["top"], "left": root["left"], "shape": root["shape"]},
                    }
                )
            if not ok:
                continue
            remaining = set(range(len(pieces))) - used_p
            if search(out0, occ0, chains, remaining):
                return best_out

    return best_out


def _holes_too_close_1d(start_a: int, size_a: int, start_b: int, size_b: int, min_gap: int) -> bool:
    end_a = start_a + size_a - 1
    end_b = start_b + size_b - 1
    if end_a < start_b:
        gap = start_b - end_a - 1
    elif end_b < start_a:
        gap = start_a - end_b - 1
    else:
        gap = -1
    return gap < min_gap


def _global_from_shape(shape: dict[str, object], top: int, left: int) -> set[tuple[int, int]]:
    return {(top + r, left + c) for r, c in shape["rel"]}


def _paint_cells(grid: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _derive_notch_from_root(root_cells: set[tuple[int, int]], bar_bounds: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    br0, bc0, br1, bc1 = bar_bounds
    return {(r, c) for r, c in root_cells if br0 <= r <= br1 and bc0 <= c <= bc1}


def _random_decoy_notches(
    rng: random.Random,
    bar_cells: set[tuple[int, int]],
    bar_bounds: tuple[int, int, int, int],
    orientation: str,
    used_specs: list[tuple[str, int, int]],
    want: int,
) -> list[set[tuple[int, int]]]:
    br0, bc0, br1, bc1 = bar_bounds
    decoys: list[set[tuple[int, int]]] = []
    used = list(used_specs)
    min_gap = 3
    for _ in range(300):
        if len(decoys) >= want:
            break
        if orientation == "horizontal":
            side = rng.choice(["top", "bottom"])
            width = rng.randint(1, 3)
            start = rng.randint(1, (bc1 - bc0 + 1) - width - 1)
            if any(side == oside and _holes_too_close_1d(start, width, ostart, ow, min_gap) for oside, ostart, ow in used):
                continue
            row = br0 if side == "top" else br1
            notch = {(row, bc0 + start + dx) for dx in range(width)}
        else:
            side = rng.choice(["left", "right"])
            width = rng.randint(1, 3)
            start = rng.randint(1, (br1 - br0 + 1) - width - 1)
            if any(side == oside and _holes_too_close_1d(start, width, ostart, ow, min_gap) for oside, ostart, ow in used):
                continue
            col = bc0 if side == "left" else bc1
            notch = {(br0 + start + dy, col) for dy in range(width)}
        if not notch <= bar_cells:
            continue
        new_bar = bar_cells - notch
        if not _connected(new_bar):
            continue
        bar_cells -= notch
        decoys.append(notch)
        used.append((side, start, width))
    return decoys


def _scatter_piece(
    rng: random.Random,
    input_grid: np.ndarray,
    placed_occ: set[tuple[int, int]],
    shape: dict[str, object],
    color: int,
    orientation: str,
    bar_bounds: tuple[int, int, int, int],
) -> bool:
    H, W = input_grid.shape
    br0, bc0, br1, bc1 = bar_bounds
    p_h = int(shape["h"])
    p_w = int(shape["w"])
    rel = list(shape["rel"])
    for _ in range(500):
        gap = rng.randint(2, 7)
        if orientation == "horizontal":
            top = (br0 - gap - p_h) if rng.random() < 0.5 else (br1 + gap + 1)
            left = rng.randint(0, W - p_w)
        else:
            left = (bc0 - gap - p_w) if rng.random() < 0.5 else (bc1 + gap + 1)
            top = rng.randint(0, H - p_h)
        if top < 0 or left < 0 or top + p_h > H or left + p_w > W:
            continue
        pts = {(top + dr, left + dc) for dr, dc in rel}
        if pts & placed_occ:
            continue
        if any((r + dr, c + dc) in placed_occ for r, c in pts for dr, dc in DIR4):
            continue
        _paint_cells(input_grid, pts, color)
        placed_occ |= pts
        return True
    return False


def _chain_layout(orientation: str, side: str) -> tuple[str, int]:
    if orientation == "horizontal":
        return "cols", (-1 if side == "top" else 1)
    return "rows", (-1 if side == "left" else 1)


def _root_follow_positions(bar_bounds: tuple[int, int, int, int], axis: str, sign: int, A: int, width: int) -> tuple[tuple[int, int], tuple[int, int]]:
    br0, bc0, br1, bc1 = bar_bounds
    if axis == "cols" and sign < 0:
        return (br0 - 2, A), (br0 - 4, A)
    if axis == "cols" and sign > 0:
        return (br1, A), (br1 + 2, A)
    if axis == "rows" and sign < 0:
        return (A, bc0 - 2), (A, bc0 - 4)
    return (A, bc1), (A, bc1 + 2)


def _gen_attempt(return_meta: bool = False):
    rng = random.Random()
    H = W = 30
    bg = 0

    orientation = rng.choice(["horizontal", "vertical"])
    thickness = rng.randint(5, 7)
    bar_color = rng.choice(list(range(1, 10)))

    if orientation == "horizontal":
        center = int(rng.uniform(0.35, 0.53) * H)
        bar_r0 = max(0, center - thickness // 2)
        bar_r1 = min(H - 1, bar_r0 + thickness - 1)
        bar_r0 = bar_r1 - thickness + 1
        bar_c0, bar_c1 = 0, W - 1
        side_order = ["top", "bottom"]
        lanes = [(4, rng.randint(2, W - 6)), (5, rng.randint(2, W - 7))]
    else:
        center = int(rng.uniform(0.35, 0.53) * W)
        bar_c0 = max(0, center - thickness // 2)
        bar_c1 = min(W - 1, bar_c0 + thickness - 1)
        bar_c0 = bar_c1 - thickness + 1
        bar_r0, bar_r1 = 0, H - 1
        side_order = ["left", "right"]
        lanes = [(4, rng.randint(2, H - 6)), (5, rng.randint(2, H - 7))]

    rng.shuffle(side_order)
    bar_bounds = (bar_r0, bar_c0, bar_r1, bar_c1)
    bar_cells = {(r, c) for r in range(bar_r0, bar_r1 + 1) for c in range(bar_c0, bar_c1 + 1)}

    chains_meta = []
    pieces_out: list[dict[str, object]] = []
    used_specs: list[tuple[str, int, int]] = []
    colors = [c for c in range(1, 10) if c != bar_color]
    rng.shuffle(colors)
    color_idx = 0

    for (width, lane_start), side in zip(lanes, side_order):
        axis, sign = _chain_layout(orientation, side)
        root_shape = _orient_template(width, "root", axis, sign)
        follow_shape = _orient_template(width, "follow", axis, sign)
        if axis == "cols":
            A = lane_start
            B = A + width - 1
            if B >= W - 1:
                return None
            used_specs.append((side, A, width))
        else:
            A = lane_start
            B = A + width - 1
            if B >= H - 1:
                return None
            used_specs.append((side, A, width))

        (rtop, rleft), (ftop, fleft) = _root_follow_positions(bar_bounds, axis, sign, A, width)
        root_cells = _global_from_shape(root_shape, rtop, rleft)
        follow_cells = _global_from_shape(follow_shape, ftop, fleft)
        if any(not (0 <= r < H and 0 <= c < W) for r, c in root_cells | follow_cells):
            return None
        if root_cells & follow_cells:
            return None

        notch = _derive_notch_from_root(root_cells, bar_bounds)
        if not notch:
            return None
        if not notch <= bar_cells:
            return None

        root_cavity = _structural_cavity_global(root_shape, rtop, rleft, axis, sign)
        covered = len(root_cavity & follow_cells)
        if covered * 2 < len(root_cavity):
            return None
        if covered == len(root_cavity):
            return None

        bar_cells -= notch

        root_color = colors[color_idx]
        follow_color = colors[color_idx + 1]
        color_idx += 2
        pieces_out.append({"color": root_color, "shape": root_shape, "coords": root_cells, "axis": axis, "sign": sign})
        pieces_out.append({"color": follow_color, "shape": follow_shape, "coords": follow_cells, "axis": axis, "sign": sign})
        chains_meta.append(
            {
                "side": side,
                "width": width,
                "root_cavity": len(root_cavity),
                "covered": covered,
                "ratio": covered / len(root_cavity),
            }
        )

    decoys = _random_decoy_notches(rng, bar_cells, bar_bounds, orientation, used_specs, want=rng.randint(4, 6))
    if len(decoys) < 4:
        return None

    bar_mask = np.zeros((bar_r1 - bar_r0 + 1, bar_c1 - bar_c0 + 1), dtype=bool)
    for r, c in bar_cells:
        bar_mask[r - bar_r0, c - bar_c0] = True
    for pts in components_mask(~bar_mask, connectivity=4):
        rr = pts[:, 0]
        cc = pts[:, 1]
        if not ((rr == 0).any() or (rr == bar_mask.shape[0] - 1).any() or (cc == 0).any() or (cc == bar_mask.shape[1] - 1).any()):
            return None

    output = np.full((H, W), bg, dtype=int)
    _paint_cells(output, bar_cells, bar_color)
    for piece in pieces_out:
        _paint_cells(output, piece["coords"], int(piece["color"]))

    input_grid = np.full((H, W), bg, dtype=int)
    _paint_cells(input_grid, bar_cells, bar_color)
    placed_occ = set(bar_cells)

    pieces_in = list(pieces_out)
    rng.shuffle(pieces_in)
    for piece in pieces_in:
        if not _scatter_piece(rng, input_grid, placed_occ, piece["shape"], int(piece["color"]), orientation, bar_bounds):
            return None

    meta = {
        "chains": chains_meta,
        "decoys": len(decoys),
        "piece_cavities": [len(_structural_cavity_local(piece["shape"], str(piece["axis"]), int(piece["sign"]))) for piece in pieces_out],
    }
    if return_meta:
        return input_grid, output, meta
    return input_grid, output


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is None:
            continue
        inp, out = result
        if np.array_equal(main(inp), out):
            return inp
    raise RuntimeError("generate_input: failed after 400 attempts")
