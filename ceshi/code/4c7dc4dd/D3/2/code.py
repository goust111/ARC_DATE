# Variant of 4c7dc4dd
# Dimension: D3 — 源对象结构 (src)
# Change: 两个训练精灵（s1.s, s1.d）位于框架内部，各使用 1-2 种非背景色，像素数 3-32... → 连通分量作为独立子精灵

import numpy as np
import random
from arc_utils import colors_present, paste

# Design analysis
# design:
#   visualization:
#     - 原任务里，训练源图通常作为一个整体来推断规则；这里改成 6x6 内部由两个彼此 4-连通分离的小组件组成。
#     - 每个组件都落在 3x3 对角槽位里，因此人眼能直接看到“这是两个分开的子精灵”。
#     - 规则仍从 {endpoints, fill, sym_fill, comp_checker, invert0, sym} 中选，但规则不再对整个 6x6 一次性生效，而是对每个组件自己的紧致 bbox 单独生效，再贴回原位。
#   behavior_trace:
#     - 原逻辑：若训练源图里有两个分开的 L 形，global fill 会跨整个 6x6 的行列范围补点，可能把两个对象混在一起。
#     - 变体逻辑：先切出两个连通分量；对每个分量各自的 bbox 做 fill；两个 bbox 彼此独立，所以不会互相“借行/借列”。
#     - 差异点正好落在 infer/apply template 这一环：frame 检测和 pair 检测可保留，模板学习与应用必须改成 component-wise。
#   complexity_calibration:
#     - 为了让“多组件”在 2-3 组样例里肉眼可见，内框固定为 6x6，训练/测试源图都强制恰好 2 个分量。
#     - 每个局部组件来自一组经过验证的 3x3 模板，避免出现规则歧义（例如某些连通 fill 骨架会被 sym_fill 误判）。
#     - 规则种类仍保留原题六类，没有引入颜色排序、颜色编号计算等不可见机制。
#   implicit_assumptions_audit:
#     - 原 main() 中 `infer_template(src, dst)` 默认 `src` 是一个整体对象；变体下这个前提失效，因此新增 `infer_componentwise(...)`。
#     - 原 `apply_template(kind, params, test_inner)` 会直接对整个测试图作用；变体下必须改成对测试图的每个 4-连通分量分别作用。
#     - 原 fill/endpoints 等分支默认“零背景都属于同一个工作画布”；变体下这个画布必须收缩到组件 bbox，否则会把独立组件错误耦合。
#     - 为了让 `dst` 仍能由 `src` 的 bbox 局部决定，generator 保证每个局部变换都不越出自己的 bbox。
#   placement_legality:
#     - 每个 6x6 内部使用两个 3x3 对角槽位；槽位只允许放一个连通局部组件，因此不会重叠、不会越界。
#     - 即便局部 bbox 小于 3x3，也只在各自槽位内随机偏移，仍与另一槽位保持 4-连通分离。
#   observability_and_cleanliness:
#     - 背景、框架布局和 pair 识别方式维持原题风格；真正变化只在框内对象结构。
#     - 两个局部组件使用同一规则但可取不同朝向，既能体现“同规则重复出现”，又不会把画面挤得过乱。


# ------- CFG -------
H = W = 30
FRAME_SIZE_CHOICES = (8,)  # fixed so the 6x6 interior can cleanly host two 3x3 component slots
INNER_SIZE = FRAME_SIZE_CHOICES[0] - 2
LOCAL_SLOT = 3
SLOT_PAIRS = (
    ((0, 0), (3, 3)),
    ((0, 3), (3, 0)),
)

SPRITE_COLORS = [1, 2, 3, 4, 5, 6, 8]
ALL_COLORS = list(range(10))
BG_PERIOD_CHOICES = (4, 5)
BG_PERTURB_FRACTION = (0.30, 0.50)
BG_EXTRA_NOISE = (0,)
QUAD_LOW_MIN, QUAD_LOW_MAX = 1, 8
QUAD_BOTTOM_MARGIN = 8

RULES = ["endpoints", "fill", "sym_fill", "comp_checker", "invert0", "sym"]
SPRITE_NZ_MIN, SPRITE_NZ_MAX = 3, 32
GEN_MAX_ATTEMPTS = 250
LOCAL_BUILD_RETRIES = 64

FILL_BASE = np.array(
    [
        [1, 1, 1],
        [1, 0, 1],
        [1, 0, 0],
    ],
    dtype=int,
)
ENDPOINTS_BASE = np.array([[1, 1, 1]], dtype=int)
COMP_CHECKER_BASE = np.array(
    [
        [1, 2, 1],
        [2, 1, 0],
        [0, 2, 0],
    ],
    dtype=int,
)
INVERT0_BASE = np.array(
    [
        [1, 1, 1],
        [1, 2, 0],
        [1, 1, 1],
    ],
    dtype=int,
)
SYM_FILL_BASE_PATTERN = np.array(
    [
        [0, 0, 1],
        [0, 0, 2],
        [1, 2, 1],
    ],
    dtype=int,
)
SYM_BASE = np.array(
    [
        [1, 2, 0],
        [0, 1, 0],
        [0, 1, 1],
    ],
    dtype=int,
)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover a local sprite rule from a framed training pair, then use it on a test sprite.

    Input structure:
    - The grid contains several hollow square frames on a noisy background; one frame is blank.
    - Two non-blank frames form the training pair and share the same interior palette, while a third non-blank frame is the test source.
    - In this variant, each non-blank source interior is itself made of multiple disconnected foreground components.

    Rule:
    - Find the training pair exactly as before, but treat each disconnected source component as its own sub-sprite.
    - Infer one shared local transformation by comparing each source component with the destination pixels inside that component's bbox.
    - Apply that same local transformation independently to every disconnected component of the test sprite, and return the transformed interior.

    Key insight:
    - The analogy is no longer about one monolithic source object. The solver must first segment the source into independent pieces, notice that the same rule repeats per piece, and only then generalize it to the test frame.
    """
    g = np.asarray(grid, dtype=int)
    H0, W0 = g.shape

    def find_square_frames():
        cands = []
        for size in range(4, min(H0, W0) + 1):
            for r0 in range(0, H0 - size + 1):
                r1 = r0 + size - 1
                for c0 in range(0, W0 - size + 1):
                    c1 = c0 + size - 1
                    col = int(g[r0, c0])
                    if int(g[r0, c1]) != col or int(g[r1, c0]) != col or int(g[r1, c1]) != col:
                        continue
                    if not (
                        np.all(g[r0, c0 : c1 + 1] == col)
                        and np.all(g[r1, c0 : c1 + 1] == col)
                        and np.all(g[r0 : r1 + 1, c0] == col)
                        and np.all(g[r0 : r1 + 1, c1] == col)
                    ):
                        continue
                    cands.append((size, col, r0, c0, r1, c1))
        cands.sort(key=lambda t: (-t[0], t[2], t[3]))
        picked = []
        for size, col, r0, c0, r1, c1 in cands:
            overlaps = False
            for _, _, pr0, pc0, pr1, pc1 in picked:
                if not (r1 < pr0 or pr1 < r0 or c1 < pc0 or pc1 < c0):
                    overlaps = True
                    break
            if not overlaps:
                picked.append((size, col, r0, c0, r1, c1))
            if len(picked) >= 6:
                break
        return picked

    def stable_colors(src, dst):
        st = set()
        for c in set(int(x) for x in np.unique(src)):
            if c == 0:
                continue
            if np.array_equal(src == c, dst == c):
                st.add(c)
        return st

    def pick_var_color(x, st):
        vals = [int(v) for v in np.unique(x) if int(v) != 0 and int(v) not in st]
        if not vals:
            return None
        best = None
        for v in vals:
            cnt = int(np.sum(x == v))
            if best is None or cnt > best[0]:
                best = (cnt, v)
        return best[1]

    def symmetries(arr):
        outs = []
        for k in range(4):
            r = np.rot90(arr, k)
            outs.append(r)
            outs.append(np.fliplr(r))
        return outs

    def sym_fill_base(arr):
        y = np.zeros_like(arr)
        for t in (lambda z: z, np.fliplr, np.flipud, lambda z: np.flipud(np.fliplr(z))):
            zz = t(arr)
            rr, cc = np.where(zz != 0)
            for r, c in zip(rr.tolist(), cc.tolist()):
                val = int(zz[r, c])
                if y[r, c] != 0 and int(y[r, c]) != val:
                    return None
                y[r, c] = val
        return y

    def infer_local_template(src, dst):
        st = stable_colors(src, dst)
        v = pick_var_color(dst, st) or pick_var_color(src, st)

        if np.array_equal(dst == 0, src != 0):
            nz = np.argwhere(dst != 0)
            if nz.size > 0:
                even_cols = set()
                odd_cols = set()
                for r, c in nz:
                    col = int(dst[int(r), int(c)])
                    if ((int(r) + int(c)) & 1) == 0:
                        even_cols.add(col)
                    else:
                        odd_cols.add(col)
                if len(even_cols) == 1 and len(odd_cols) == 1:
                    return ("comp_checker", {"stable": st})

        sym_src = sym_fill_base(src)
        if sym_src is not None:
            if np.all(dst[sym_src != 0] == sym_src[sym_src != 0]):
                fill_cols = sorted(set(int(x) for x in dst[sym_src == 0].ravel() if int(x) != 0))
                if len(fill_cols) <= 1:
                    fill_col = fill_cols[0] if fill_cols else None
                    ok = True
                    if fill_col is not None:
                        bad = (sym_src == 0) & (dst != 0) & (dst != fill_col)
                        if np.any(bad):
                            ok = False
                    if ok:
                        keep0 = (sym_src == 0) & (dst == 0)
                        return ("sym_fill", {"stable": st, "keep0": keep0})

        if v is not None:
            pred = src.copy()
            m0 = src == 0
            mv = src == v
            pred[m0] = v
            pred[mv] = 0
            for c in st:
                pred[dst == c] = c
            if np.array_equal(pred, dst):
                return ("invert0", {"stable": st})

        for idx, var in enumerate(symmetries(src)):
            if np.array_equal(var, dst):
                return ("sym", {"idx": idx})

        if v is not None:
            coords = np.argwhere(src == v)
            s = set((int(r), int(c)) for r, c in coords)
            keep = np.zeros_like(src, dtype=bool)
            for r, c in s:
                deg = 0
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        if (r + dr, c + dc) in s:
                            deg += 1
                if deg <= 1:
                    keep[r, c] = True
            pred = src.copy()
            pred[src == v] = 0
            pred[keep] = v
            if np.array_equal(pred, dst):
                return ("endpoints", {"stable": st})

            pred = src.copy()
            h, w = pred.shape
            stable_mask = np.zeros_like(pred, dtype=bool)
            for c in st:
                stable_mask |= pred == c
            for r in range(h):
                nz = np.where(pred[r, :] != 0)[0]
                if nz.size >= 2:
                    cmin = int(nz.min())
                    cmax = int(nz.max())
                    for cc in range(cmin, cmax + 1):
                        if pred[r, cc] == 0 and not stable_mask[r, cc]:
                            pred[r, cc] = v
            for c in range(w):
                nz = np.where(pred[:, c] != 0)[0]
                if nz.size >= 2:
                    rmin = int(nz.min())
                    rmax = int(nz.max())
                    for rr in range(rmin, rmax + 1):
                        if pred[rr, c] == 0 and not stable_mask[rr, c]:
                            pred[rr, c] = v
            if np.array_equal(pred, dst):
                return ("fill", {"stable": st})

        return None

    def apply_local_template(kind, params, x):
        if kind == "sym":
            return symmetries(x)[params["idx"]].copy()

        st = set(params.get("stable", set()))
        v = pick_var_color(x, st)

        if kind == "comp_checker":
            nz = np.argwhere(x != 0)
            even_counts = {}
            odd_counts = {}
            for r, c in nz:
                col = int(x[int(r), int(c)])
                if ((int(r) + int(c)) & 1) == 0:
                    even_counts[col] = even_counts.get(col, 0) + 1
                else:
                    odd_counts[col] = odd_counts.get(col, 0) + 1
            if even_counts:
                c_even = max(even_counts.items(), key=lambda t: t[1])[0]
            elif odd_counts:
                c_even = max(odd_counts.items(), key=lambda t: t[1])[0]
            else:
                return x.copy()
            c_odd = max(odd_counts.items(), key=lambda t: t[1])[0] if odd_counts else c_even
            h, w = x.shape
            y = np.zeros_like(x)
            for r in range(h):
                for c in range(w):
                    if x[r, c] != 0:
                        y[r, c] = 0
                    else:
                        y[r, c] = c_even if ((r + c) & 1) == 0 else c_odd
            return y

        if kind == "sym_fill":
            h, w = x.shape
            nz = np.argwhere(x != 0)
            if nz.size == 0:
                return x.copy()
            best = None
            counts = {}
            for r, c in nz:
                col = int(x[int(r), int(c)])
                d = min(int(r), int(c), h - 1 - int(r), w - 1 - int(c))
                counts[col] = counts.get(col, 0) + 1
                if best is None or d > best[0] or (d == best[0] and counts[col] > best[1]):
                    best = (d, counts[col], col)
            fill_col = best[2]
            y = sym_fill_base(x)
            if y is None:
                return x.copy()
            keep0 = params.get("keep0", None)
            if keep0 is None or keep0.shape != y.shape:
                keep0 = np.zeros_like(y, dtype=bool)
            y[(y == 0) & ~keep0] = fill_col
            return y

        if v is None:
            return x.copy()

        if kind == "invert0":
            y = x.copy()
            m0 = x == 0
            mv = x == v
            y[m0] = v
            y[mv] = 0
            return y

        if kind == "endpoints":
            coords = np.argwhere(x == v)
            s = set((int(r), int(c)) for r, c in coords)
            keep = np.zeros_like(x, dtype=bool)
            for r, c in s:
                deg = 0
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        if (r + dr, c + dc) in s:
                            deg += 1
                if deg <= 1:
                    keep[r, c] = True
            y = x.copy()
            y[x == v] = 0
            y[keep] = v
            return y

        if kind == "fill":
            y = x.copy()
            h, w = y.shape
            stable_mask = np.zeros_like(y, dtype=bool)
            for c in st:
                stable_mask |= y == c
            for r in range(h):
                nz = np.where(y[r, :] != 0)[0]
                if nz.size >= 2:
                    cmin = int(nz.min())
                    cmax = int(nz.max())
                    for cc in range(cmin, cmax + 1):
                        if y[r, cc] == 0 and not stable_mask[r, cc]:
                            y[r, cc] = v
            for c in range(w):
                nz = np.where(y[:, c] != 0)[0]
                if nz.size >= 2:
                    rmin = int(nz.min())
                    rmax = int(nz.max())
                    for rr in range(rmin, rmax + 1):
                        if y[rr, c] == 0 and not stable_mask[rr, c]:
                            y[rr, c] = v
            return y

        return x.copy()

    def component_bboxes4(arr):
        mask = arr != 0
        h, w = arr.shape
        seen = np.zeros_like(mask, dtype=bool)
        out = []
        for r in range(h):
            for c in range(w):
                if not mask[r, c] or seen[r, c]:
                    continue
                stack = [(r, c)]
                seen[r, c] = True
                cells = []
                while stack:
                    rr, cc = stack.pop()
                    cells.append((rr, cc))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
                rows = [p[0] for p in cells]
                cols = [p[1] for p in cells]
                out.append((min(rows), min(cols), max(rows), max(cols)))
        out.sort()
        return out

    def freeze_value(v):
        if isinstance(v, np.ndarray):
            return tuple(tuple(int(x) for x in row) for row in v.astype(int).tolist())
        if isinstance(v, set):
            return tuple(sorted(int(x) for x in v))
        return v

    def param_signature(kind, params):
        return (kind, tuple(sorted((k, freeze_value(v)) for k, v in params.items())))

    def infer_componentwise(src, dst):
        bboxes = component_bboxes4(src)
        if len(bboxes) < 2:
            return None
        covered = np.zeros_like(src, dtype=bool)
        first_kind = None
        first_params = None
        first_sig = None
        for r0, c0, r1, c1 in bboxes:
            covered[r0 : r1 + 1, c0 : c1 + 1] = True
            local_src = src[r0 : r1 + 1, c0 : c1 + 1]
            local_dst = dst[r0 : r1 + 1, c0 : c1 + 1]
            tpl = infer_local_template(local_src, local_dst)
            if tpl is None:
                return None
            kind, params = tpl
            sig = param_signature(kind, params)
            if first_kind is None:
                first_kind = kind
                first_params = params
                first_sig = sig
            elif sig != first_sig:
                return None
        if np.any((dst != 0) & ~covered):
            return None
        return first_kind, first_params

    def apply_componentwise(kind, params, x):
        bboxes = component_bboxes4(x)
        if not bboxes:
            return x.copy()
        y = np.zeros_like(x)
        for r0, c0, r1, c1 in bboxes:
            local = x[r0 : r1 + 1, c0 : c1 + 1]
            transformed = apply_local_template(kind, params, local)
            y[r0 : r1 + 1, c0 : c1 + 1] = transformed
        return y

    frames = find_square_frames()
    if len(frames) < 3:
        return g.copy()

    inners = []
    for _, _, r0, c0, r1, c1 in frames:
        inners.append(g[r0 + 1 : r1, c0 + 1 : c1])

    blank_idx = None
    for i, inner in enumerate(inners):
        if inner.size > 0 and np.all(inner == 0):
            blank_idx = i
            break
    if blank_idx is None:
        blank_idx = int(np.argmin([np.count_nonzero(inner) for inner in inners]))

    nonblank = [i for i in range(len(inners)) if i != blank_idx]
    sigs = []
    for i in nonblank:
        inner = inners[i]
        nz = sorted(set(int(x) for x in inner.ravel() if int(x) != 0))
        sigs.append((i, tuple(nz)))

    pair = None
    for a in range(len(sigs)):
        for b in range(a + 1, len(sigs)):
            if sigs[a][1] == sigs[b][1]:
                pair = (sigs[a][0], sigs[b][0])
                break
        if pair is not None:
            break
    if pair is None:
        counts = [(i, int(np.count_nonzero(inners[i]))) for i in nonblank]
        counts.sort(key=lambda t: t[1])
        pair = (counts[0][0], counts[1][0])

    remaining = [i for i in nonblank if i not in pair]
    if not remaining:
        return inners[blank_idx].copy()
    test_idx = remaining[0]

    src_a = inners[pair[0]]
    src_b = inners[pair[1]]
    test_inner = inners[test_idx]

    best = None
    for src, dst in ((src_a, src_b), (src_b, src_a)):
        component_tpl = infer_componentwise(src, dst)
        if component_tpl is not None:
            kind, params = component_tpl
            mode = "components"
        else:
            whole_tpl = infer_local_template(src, dst)
            if whole_tpl is None:
                continue
            kind, params = whole_tpl
            mode = "whole"

        st = set(params.get("stable", set()))
        v_src = pick_var_color(src, st)
        v_dst = pick_var_color(dst, st)
        score = 1
        if kind == "fill" and v_src is not None and v_dst is not None:
            if int(np.sum(dst == v_dst)) >= int(np.sum(src == v_src)):
                score += 2
        if kind == "endpoints" and v_src is not None and v_dst is not None:
            if int(np.sum(dst == v_dst)) <= int(np.sum(src == v_src)):
                score += 2
        if mode == "components":
            score += 20
        src_n = int(np.sum((src != 0) & ~np.isin(src, list(st))))
        test_n = int(np.sum((test_inner != 0) & ~np.isin(test_inner, list(st))))
        score2 = score * 100 - abs(src_n - test_n)
        if best is None or score2 > best[0]:
            best = (score2, mode, kind, params)

    if best is None:
        return test_inner.copy()

    _, mode, kind, params = best
    if mode == "components":
        return apply_componentwise(kind, params, test_inner)
    return apply_local_template(kind, params, test_inner)


def find_square_frames(grid: np.ndarray) -> list[tuple[int, int, int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    cands = []
    for size in range(4, min(h, w) + 1):
        for r0 in range(0, h - size + 1):
            r1 = r0 + size - 1
            for c0 in range(0, w - size + 1):
                c1 = c0 + size - 1
                col = int(g[r0, c0])
                if int(g[r0, c1]) != col or int(g[r1, c0]) != col or int(g[r1, c1]) != col:
                    continue
                if not (
                    np.all(g[r0, c0 : c1 + 1] == col)
                    and np.all(g[r1, c0 : c1 + 1] == col)
                    and np.all(g[r0 : r1 + 1, c0] == col)
                    and np.all(g[r0 : r1 + 1, c1] == col)
                ):
                    continue
                cands.append((size, col, r0, c0, r1, c1))
    cands.sort(key=lambda t: (-t[0], t[2], t[3]))
    picked = []
    for cand in cands:
        _, _, r0, c0, r1, c1 = cand
        overlaps = False
        for _, _, pr0, pc0, pr1, pc1 in picked:
            if not (r1 < pr0 or pr1 < r0 or c1 < pc0 or pc1 < c0):
                overlaps = True
                break
        if not overlaps:
            picked.append(cand)
    return picked


def stable_colors(src: np.ndarray, dst: np.ndarray) -> set[int]:
    st = set()
    for c in set(int(x) for x in np.unique(src)):
        if c == 0:
            continue
        if np.array_equal(src == c, dst == c):
            st.add(c)
    return st


def pick_var_color(x: np.ndarray, st: set[int]) -> int | None:
    vals = [int(v) for v in np.unique(x) if int(v) != 0 and int(v) not in st]
    if not vals:
        return None
    best = None
    for v in vals:
        cnt = int(np.sum(x == v))
        if best is None or cnt > best[0]:
            best = (cnt, v)
    return best[1]


def symmetries(arr: np.ndarray) -> list[np.ndarray]:
    outs = []
    for k in range(4):
        r = np.rot90(arr, k)
        outs.append(r)
        outs.append(np.fliplr(r))
    return outs


def sym_fill_base(arr: np.ndarray) -> np.ndarray | None:
    y = np.zeros_like(arr)
    for t in (lambda z: z, np.fliplr, np.flipud, lambda z: np.flipud(np.fliplr(z))):
        zz = t(arr)
        rr, cc = np.where(zz != 0)
        for r, c in zip(rr.tolist(), cc.tolist()):
            val = int(zz[r, c])
            if y[r, c] != 0 and int(y[r, c]) != val:
                return None
            y[r, c] = val
    return y


def infer_local_template(src: np.ndarray, dst: np.ndarray):
    st = stable_colors(src, dst)
    v = pick_var_color(dst, st) or pick_var_color(src, st)

    if np.array_equal(dst == 0, src != 0):
        nz = np.argwhere(dst != 0)
        if nz.size > 0:
            even_cols = set()
            odd_cols = set()
            for r, c in nz:
                col = int(dst[int(r), int(c)])
                if ((int(r) + int(c)) & 1) == 0:
                    even_cols.add(col)
                else:
                    odd_cols.add(col)
            if len(even_cols) == 1 and len(odd_cols) == 1:
                return ("comp_checker", {"stable": st})

    sym_src = sym_fill_base(src)
    if sym_src is not None:
        if np.all(dst[sym_src != 0] == sym_src[sym_src != 0]):
            fill_cols = sorted(set(int(x) for x in dst[sym_src == 0].ravel() if int(x) != 0))
            if len(fill_cols) <= 1:
                fill_col = fill_cols[0] if fill_cols else None
                ok = True
                if fill_col is not None:
                    bad = (sym_src == 0) & (dst != 0) & (dst != fill_col)
                    if np.any(bad):
                        ok = False
                if ok:
                    keep0 = (sym_src == 0) & (dst == 0)
                    return ("sym_fill", {"stable": st, "keep0": keep0})

    if v is not None:
        pred = src.copy()
        m0 = src == 0
        mv = src == v
        pred[m0] = v
        pred[mv] = 0
        for c in st:
            pred[dst == c] = c
        if np.array_equal(pred, dst):
            return ("invert0", {"stable": st})

    for idx, var in enumerate(symmetries(src)):
        if np.array_equal(var, dst):
            return ("sym", {"idx": idx})

    if v is not None:
        coords = np.argwhere(src == v)
        s = set((int(r), int(c)) for r, c in coords)
        keep = np.zeros_like(src, dtype=bool)
        for r, c in s:
            deg = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    if (r + dr, c + dc) in s:
                        deg += 1
            if deg <= 1:
                keep[r, c] = True
        pred = src.copy()
        pred[src == v] = 0
        pred[keep] = v
        if np.array_equal(pred, dst):
            return ("endpoints", {"stable": st})

        pred = src.copy()
        h, w = pred.shape
        stable_mask = np.zeros_like(pred, dtype=bool)
        for c in st:
            stable_mask |= pred == c
        for r in range(h):
            nz = np.where(pred[r, :] != 0)[0]
            if nz.size >= 2:
                cmin = int(nz.min())
                cmax = int(nz.max())
                for cc in range(cmin, cmax + 1):
                    if pred[r, cc] == 0 and not stable_mask[r, cc]:
                        pred[r, cc] = v
        for c in range(w):
            nz = np.where(pred[:, c] != 0)[0]
            if nz.size >= 2:
                rmin = int(nz.min())
                rmax = int(nz.max())
                for rr in range(rmin, rmax + 1):
                    if pred[rr, c] == 0 and not stable_mask[rr, c]:
                        pred[rr, c] = v
        if np.array_equal(pred, dst):
            return ("fill", {"stable": st})

    return None


def apply_local_template(kind: str, params: dict, x: np.ndarray) -> np.ndarray:
    if kind == "sym":
        return symmetries(x)[params["idx"]].copy()

    st = set(params.get("stable", set()))
    v = pick_var_color(x, st)

    if kind == "comp_checker":
        rr, cc = np.indices(x.shape)
        parity = (rr + cc) & 1
        even_mask = (x != 0) & (parity == 0)
        odd_mask = (x != 0) & (parity == 1)
        even_counts = _bincount_colors(x[even_mask])
        odd_counts = _bincount_colors(x[odd_mask])
        if even_counts:
            c_even = max(even_counts.items(), key=lambda t: t[1])[0]
        elif odd_counts:
            c_even = max(odd_counts.items(), key=lambda t: t[1])[0]
        else:
            return x.copy()
        c_odd = max(odd_counts.items(), key=lambda t: t[1])[0] if odd_counts else c_even
        y = np.where(parity == 0, c_even, c_odd).astype(x.dtype)
        y[x != 0] = 0
        return y

    if kind == "sym_fill":
        h, w = x.shape
        nz = np.argwhere(x != 0)
        if nz.size == 0:
            return x.copy()
        best = None
        counts = {}
        for r, c in nz:
            col = int(x[int(r), int(c)])
            d = min(int(r), int(c), h - 1 - int(r), w - 1 - int(c))
            counts[col] = counts.get(col, 0) + 1
            if best is None or d > best[0] or (d == best[0] and counts[col] > best[1]):
                best = (d, counts[col], col)
        fill_col = best[2]
        y = sym_fill_base(x)
        if y is None:
            return x.copy()
        keep0 = params.get("keep0", None)
        if keep0 is None or keep0.shape != y.shape:
            keep0 = np.zeros_like(y, dtype=bool)
        y[(y == 0) & ~keep0] = fill_col
        return y

    if v is None:
        return x.copy()

    if kind == "invert0":
        y = x.copy()
        m0 = x == 0
        mv = x == v
        y[m0] = v
        y[mv] = 0
        return y

    if kind == "endpoints":
        coords = np.argwhere(x == v)
        s = set((int(r), int(c)) for r, c in coords)
        keep = np.zeros_like(x, dtype=bool)
        for r, c in s:
            deg = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    if (r + dr, c + dc) in s:
                        deg += 1
            if deg <= 1:
                keep[r, c] = True
        y = x.copy()
        y[x == v] = 0
        y[keep] = v
        return y

    if kind == "fill":
        y = x.copy()
        h, w = y.shape
        stable_mask = np.zeros_like(y, dtype=bool)
        for c in st:
            stable_mask |= y == c
        for r in range(h):
            nz = np.where(y[r, :] != 0)[0]
            if nz.size >= 2:
                cmin = int(nz.min())
                cmax = int(nz.max())
                for cc in range(cmin, cmax + 1):
                    if y[r, cc] == 0 and not stable_mask[r, cc]:
                        y[r, cc] = v
        for c in range(w):
            nz = np.where(y[:, c] != 0)[0]
            if nz.size >= 2:
                rmin = int(nz.min())
                rmax = int(nz.max())
                for rr in range(rmin, rmax + 1):
                    if y[rr, c] == 0 and not stable_mask[rr, c]:
                        y[rr, c] = v
        return y

    return x.copy()


def component_bboxes4(arr: np.ndarray) -> list[tuple[int, int, int, int]]:
    mask = arr != 0
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [p[0] for p in cells]
            cols = [p[1] for p in cells]
            out.append((min(rows), min(cols), max(rows), max(cols)))
    out.sort()
    return out


def _bincount_colors(values: np.ndarray) -> dict[int, int]:
    out: dict[int, int] = {}
    for v in values.tolist():
        if v == 0:
            continue
        out[v] = out.get(v, 0) + 1
    return out


def crop_to_content(arr: np.ndarray) -> np.ndarray:
    nz = np.argwhere(arr != 0)
    if nz.size == 0:
        return arr.copy()
    r0, c0 = nz.min(axis=0)
    r1, c1 = nz.max(axis=0)
    return arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def colorize_blueprint(bp: np.ndarray, colors: list[int]) -> np.ndarray:
    y = np.zeros_like(bp)
    if 1 in bp:
        y[bp == 1] = colors[0]
    if 2 in bp:
        y[bp == 2] = colors[1] if len(colors) >= 2 else colors[0]
    return y


def choose_palette(rng: random.Random, count: int, forbidden: set[int]) -> list[int]:
    pool = [c for c in SPRITE_COLORS if c not in forbidden]
    if len(pool) < count:
        raise ValueError("not enough sprite colors")
    return rng.sample(pool, count)


def choose_rule_setup(rng: random.Random) -> tuple[str, dict, int]:
    kind = rng.choice(RULES)
    params: dict = {}
    if kind in {"invert0", "sym_fill"}:
        color_count = 2
    elif kind == "sym":
        color_count = rng.choice([1, 2])
        params["idx"] = rng.randint(1, 7)
    elif kind == "comp_checker":
        color_count = rng.choice([1, 2])
    else:
        color_count = 1
    return kind, params, color_count


def random_oriented(base: np.ndarray, rng: random.Random) -> np.ndarray:
    arr = symmetries(base)[rng.randrange(8)]
    return crop_to_content(arr)


def build_local_component(kind: str, setup_params: dict, colors: list[int], rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(LOCAL_BUILD_RETRIES):
        if kind == "fill":
            src = colorize_blueprint(random_oriented(FILL_BASE, rng), colors)
            dst = apply_local_template("fill", {"stable": set()}, src)
        elif kind == "endpoints":
            src = colorize_blueprint(random_oriented(ENDPOINTS_BASE, rng), colors)
            dst = apply_local_template("endpoints", {"stable": set()}, src)
        elif kind == "comp_checker":
            src = colorize_blueprint(random_oriented(COMP_CHECKER_BASE, rng), colors)
            dst = apply_local_template("comp_checker", {"stable": set()}, src)
        elif kind == "invert0":
            src = colorize_blueprint(random_oriented(INVERT0_BASE, rng), colors)
            dst = apply_local_template("invert0", {"stable": {colors[1]}}, src)
        elif kind == "sym_fill":
            src = colorize_blueprint(random_oriented(SYM_FILL_BASE_PATTERN, rng), colors)
            keep0 = np.zeros(src.shape, dtype=bool)
            dst = apply_local_template("sym_fill", {"keep0": keep0}, src)
        elif kind == "sym":
            src = colorize_blueprint(random_oriented(SYM_BASE, rng), colors)
            dst = symmetries(src)[setup_params["idx"]]
        else:
            raise ValueError(kind)

        if np.array_equal(src, dst):
            continue
        inferred = infer_local_template(src, dst)
        if inferred is None or inferred[0] != kind:
            continue
        if kind == "sym" and inferred[1].get("idx") != setup_params["idx"]:
            continue
        return src, dst
    raise ValueError(f"failed local build for {kind}")


def place_local(canvas: np.ndarray, local: np.ndarray, slot_anchor: tuple[int, int], rng: random.Random) -> tuple[int, int]:
    r_base, c_base = slot_anchor
    h, w = local.shape
    dr = rng.randint(0, LOCAL_SLOT - h)
    dc = rng.randint(0, LOCAL_SLOT - w)
    paste(canvas, local, r_base + dr, c_base + dc, transparent=0)
    return r_base + dr, c_base + dc


def build_multi_component_pair(kind: str, setup_params: dict, colors: list[int], rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    src = np.zeros((INNER_SIZE, INNER_SIZE), dtype=int)
    dst = np.zeros_like(src)
    slots = rng.choice(SLOT_PAIRS)
    for slot_anchor in slots:
        local_src, local_dst = build_local_component(kind, setup_params, colors, rng)
        r0, c0 = place_local(src, local_src, slot_anchor, rng)
        paste(dst, local_dst, r0, c0, transparent=0)
    return src, dst


def quadrant_positions(s: int, rng: random.Random) -> list[tuple[int, int]]:
    low, high = QUAD_LOW_MIN, QUAD_LOW_MAX
    bottom_low = H - s - QUAD_BOTTOM_MARGIN
    bottom_high = H - s - 1
    return [
        (rng.randint(low, high), rng.randint(low, high)),
        (rng.randint(low, high), rng.randint(bottom_low, bottom_high)),
        (rng.randint(bottom_low, bottom_high), rng.randint(low, high)),
        (rng.randint(bottom_low, bottom_high), rng.randint(bottom_low, bottom_high)),
    ]


def render_background(
    frame_boxes: list[tuple[int, int, int, int]],
    bg_palette: list[int],
    rng: random.Random,
) -> np.ndarray:
    period = len(bg_palette)
    direction = rng.choice([1, -1])
    rr, cc = np.indices((H, W))
    idx = (rr + direction * cc) % period
    grid = np.asarray(bg_palette, dtype=int)[idx]

    reserved = np.zeros((H, W), dtype=bool)
    for r0, c0, r1, c1 in frame_boxes:
        reserved[r0 : r1 + 1, c0 : c1 + 1] = True
    bg_cells = [(r, c) for r, c in zip(*np.where(~reserved))]
    lo, hi = BG_PERTURB_FRACTION
    perturb_n = int(round(len(bg_cells) * rng.uniform(lo, hi)))
    noise_choices = list(bg_palette) + [c for c in BG_EXTRA_NOISE if c not in bg_palette]
    for r, c in rng.sample(bg_cells, perturb_n):
        grid[r, c] = rng.choice(noise_choices)
    return grid


def draw_frame(grid: np.ndarray, r0: int, c0: int, s: int, frame_color: int, sprite: np.ndarray) -> None:
    r1, c1 = r0 + s - 1, c0 + s - 1
    grid[r0 : r1 + 1, c0 : c1 + 1] = 0
    grid[r0, c0 : c1 + 1] = frame_color
    grid[r1, c0 : c1 + 1] = frame_color
    grid[r0 : r1 + 1, c0] = frame_color
    grid[r0 : r1 + 1, c1] = frame_color
    paste(grid, sprite, r0 + 1, c0 + 1, transparent=0)


def component_count(arr: np.ndarray) -> int:
    return len(component_bboxes4(arr))


def validate_scene(grid: np.ndarray, expected_output: np.ndarray) -> bool:
    frames = find_square_frames(grid)
    if len(frames) != 4:
        return False
    sizes = {size for size, _, *_ in frames}
    if sizes != {FRAME_SIZE_CHOICES[0]}:
        return False
    inners = [grid[r0 + 1 : r1, c0 + 1 : c1] for _, _, r0, c0, r1, c1 in frames]
    if sum(int(np.all(inner == 0)) for inner in inners) != 1:
        return False
    nonblank_sigs = [tuple(colors_present(inner, exclude={0})) for inner in inners]
    nonblank_sigs = [s for s in nonblank_sigs if s]
    dup_count = sum(
        1
        for i in range(len(nonblank_sigs))
        for j in range(i + 1, len(nonblank_sigs))
        if nonblank_sigs[i] == nonblank_sigs[j]
    )
    if dup_count != 1:
        return False
    return np.array_equal(main(grid), expected_output)


def _valid_nz(arr: np.ndarray) -> bool:
    return SPRITE_NZ_MIN <= int(np.count_nonzero(arr)) <= SPRITE_NZ_MAX


def _choose_frame_and_bg(used: set[int], rng: random.Random) -> tuple[int, list[int]]:
    frame_color = rng.choice([c for c in ALL_COLORS if c not in used])
    bg_pool = [c for c in ALL_COLORS if c not in used and c != frame_color]
    bg_palette = rng.sample(bg_pool, rng.choice(BG_PERIOD_CHOICES))
    return frame_color, bg_palette


def _assign_slots(positions: list[tuple[int, int]], rng: random.Random) -> tuple[list[int], int]:
    order = sorted(range(4), key=lambda i: positions[i])
    train_slots = sorted(rng.sample(order, 2), key=order.index)
    remaining = [i for i in order if i not in train_slots]
    test_slot = rng.choice(remaining)
    return train_slots, test_slot


def _render_scene(
    s: int,
    positions: list[tuple[int, int]],
    sprites: list[np.ndarray],
    frame_color: int,
    bg_palette: list[int],
    rng: random.Random,
) -> np.ndarray:
    frame_boxes = [(r0, c0, r0 + s - 1, c0 + s - 1) for r0, c0 in positions]
    grid = render_background(frame_boxes, bg_palette, rng)
    for (r0, c0), sprite in zip(positions, sprites):
        draw_frame(grid, r0, c0, s, frame_color, sprite)
    return grid


def _build_sprites(kind: str, setup_params: dict, p1: list[int], p2: list[int], rng: random.Random):
    try:
        train_src, train_dst = build_multi_component_pair(kind, setup_params, p1, rng)
    except ValueError:
        return None
    comp_inferred = None
    if component_count(train_src) >= 2:
        first = None
        ok = True
        for r0, c0, r1, c1 in component_bboxes4(train_src):
            local_src = train_src[r0 : r1 + 1, c0 : c1 + 1]
            local_dst = train_dst[r0 : r1 + 1, c0 : c1 + 1]
            tpl = infer_local_template(local_src, local_dst)
            if tpl is None or tpl[0] != kind:
                ok = False
                break
            if first is None:
                first = tpl
            else:
                sig0 = (first[0], tuple(sorted((k, str(v.tolist()) if isinstance(v, np.ndarray) else tuple(sorted(v)) if isinstance(v, set) else v) for k, v in first[1].items())))
                sig1 = (tpl[0], tuple(sorted((k, str(v.tolist()) if isinstance(v, np.ndarray) else tuple(sorted(v)) if isinstance(v, set) else v) for k, v in tpl[1].items())))
                if sig0 != sig1:
                    ok = False
                    break
        if ok:
            comp_inferred = first
    if comp_inferred is None:
        return None

    try:
        test_src, test_dst = build_multi_component_pair(kind, setup_params, p2, rng)
    except ValueError:
        return None

    if np.array_equal(test_dst, test_src):
        return None
    for arr in (train_src, train_dst, test_src, test_dst):
        if not _valid_nz(arr):
            return None
    if component_count(train_src) != 2 or component_count(test_src) != 2:
        return None
    train_sig = tuple(colors_present(train_src, exclude={0}))
    if train_sig != tuple(colors_present(train_dst, exclude={0})):
        return None
    if tuple(colors_present(test_src, exclude={0})) == train_sig:
        return None
    return train_src, train_dst, test_src, test_dst


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    s = FRAME_SIZE_CHOICES[0]

    kind, setup_params, color_count = choose_rule_setup(rng)
    p1 = choose_palette(rng, color_count, set())
    p2 = choose_palette(rng, color_count, set(p1))

    built = _build_sprites(kind, setup_params, p1, p2, rng)
    if built is None:
        return None
    train_src, train_dst, test_src, test_dst = built

    frame_color, bg_palette = _choose_frame_and_bg(set(p1) | set(p2), rng)
    positions = quadrant_positions(s, rng)
    train_slots, test_slot = _assign_slots(positions, rng)

    sprites = [np.zeros((INNER_SIZE, INNER_SIZE), dtype=int) for _ in range(4)]
    sprites[train_slots[0]] = train_src
    sprites[train_slots[1]] = train_dst
    sprites[test_slot] = test_src

    grid = _render_scene(s, positions, sprites, frame_color, bg_palette, rng)
    if not validate_scene(grid, test_dst):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_MAX_ATTEMPTS} attempts")
