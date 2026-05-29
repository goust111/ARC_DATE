# Variant of 4c7dc4dd
# Dimension: D19 — 变换验证机制 (meta)
# Change: 空白框架仅作为干扰/区分标志，不参与变换逻辑 → 空白框架包含提示
#
# Design analysis
# ---------------
# a) 具体化与行为追踪
#    - 原题四个同尺寸方框里有一个全黑空框。solver 先把它排除，再从其余三个里找训练对和测试精灵。
#    - 本变体把“空框”改成“提示框”：它的内部用保留色 7 画一支 8 向箭头，箭头表示训练对的有向关系，
#      即“哪一帧是 s1.s，哪一帧是 s1.d”。这样提示框从被动占位变成主动提供元信息。
#    - 具体小例子：若两个同配色训练框分别位于左上和右下，提示框中的箭头若指向右下，
#      则 solver 必须把左上当源、右下当目标；若箭头反向，则应交换两者。原始 solver 会双向尝试再打分，
#      变体 solver 则先读箭头，再只按箭头指定方向推断模板并作用到测试框。
# b) 复杂度校准
#    - 只引入一个新的提示机制，不改四框布局和“按训练对泛化到测试框”的主干。
#    - 为了保证提示确实影响答案，生成器只采样方向敏感的 90°/270° 旋转对称规则。若采样自反规则
#      （如镜像、自反反色），即使有箭头也不会改变测试输出，会退化回原题。
# c) 隐式假设审查
#    - 原 `blank_idx` 逻辑依赖“有且仅有一个 interior 全 0”，在本变体中失效，必须改成识别提示色 7。
#    - 原 `nonblank` / `pair` / `remaining` 逻辑把“非空的剩余帧”默认当测试帧；现在应先剔除提示框，
#      再在剩余三帧中找唯一同配色训练对，最后剩下的才是测试帧。
#    - 原 `best` 循环双向尝试训练对方向，默认方向可猜；本变体里方向来自提示框，所以不能再靠启发式反推。
# d) 放置合法性
#    - 仍使用 30x30 画布、4 个 6-8 的同尺寸方框、象限式摆放，确保不重叠不越界。
#    - 提示箭头只画在提示框内部，并且保留色 7 不出现在训练/测试精灵、边框和背景中，避免识别冲突。
# e) 视觉可观察性
#    - 人类看到样例时，可以先注意到唯一的保留色箭头，再观察它指向哪一个训练框。
#    - 训练规则限定为 90°/270° 旋转，方向差异在 2-3 组样例上可见，不依赖颜色 id 或隐藏排序。
# f) 可视化整洁度
#    - 箭头只占提示框内部，不叠加到其他对象上；背景和边框颜色与提示色、精灵色相互避让。
#    - 精灵仍是小型稀疏图案，避免内部过满导致旋转前后难以区分。
# g) 颜色变化可推断性
#    - 提示机制不做颜色映射；颜色只承担“训练对同配色”“测试框不同配色”“提示色固定为 7”这三种可见角色。

import numpy as np
import random
from arc_utils import colors_present, draw_line, paste


H = W = 30
SPRITE_COLORS = [1, 2, 3, 4, 5, 6, 8]
ALL_COLORS = list(range(10))
HINT_COLOR = 7

BG_PERIOD_CHOICES = (4, 5)
BG_PERTURB_FRACTION = (0.30, 0.50)
BG_EXTRA_NOISE = (0,)

FRAME_SIZE_CHOICES = (6, 7, 8)
QUAD_LOW_MIN, QUAD_LOW_MAX = 1, 8
QUAD_BOTTOM_MARGIN = 8

RULES = ["sym"]
ROTATION_IDXS = (2, 6)  # rot90 / rot270 only; direction matters.
SPRITE_NZ_MIN, SPRITE_NZ_MAX = 3, 32
SPARSE_COUNT_MAX = 10
GEN_MAX_ATTEMPTS = 400

DIRECTIONS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def sign(x: int) -> int:
    if x < 0:
        return -1
    if x > 0:
        return 1
    return 0


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
        if len(picked) >= 6:
            break
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


def infer_template(src: np.ndarray, dst: np.ndarray):
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

    def sym_fill_base(a: np.ndarray) -> np.ndarray | None:
        y = np.zeros_like(a)
        for t in (lambda z: z, np.fliplr, np.flipud, lambda z: np.flipud(np.fliplr(z))):
            zz = t(a)
            rr, cc = np.where(zz != 0)
            for r, c in zip(rr.tolist(), cc.tolist()):
                val = int(zz[r, c])
                if y[r, c] != 0 and int(y[r, c]) != val:
                    return None
                y[r, c] = val
        return y

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


def apply_template(kind: str, params: dict, x: np.ndarray) -> np.ndarray:
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
                y[r, c] = 0 if x[r, c] != 0 else (c_even if ((r + c) & 1) == 0 else c_odd)
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

        y = np.zeros_like(x)
        for t in (lambda z: z, np.fliplr, np.flipud, lambda z: np.flipud(np.fliplr(z))):
            zz = t(x)
            rr, cc = np.where(zz != 0)
            for r, c in zip(rr.tolist(), cc.tolist()):
                val = int(zz[r, c])
                if y[r, c] != 0 and int(y[r, c]) != val:
                    return x.copy()
                y[r, c] = val

        keep0 = params.get("keep0", None)
        if keep0 is None or keep0.shape != y.shape:
            keep0 = np.zeros_like(y, dtype=bool)
        for r in range(h):
            for c in range(w):
                if y[r, c] == 0 and not bool(keep0[r, c]):
                    y[r, c] = fill_col
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


def arrow_mask(n: int, direction: tuple[int, int]) -> np.ndarray:
    dr, dc = direction
    mask = np.zeros((n, n), dtype=int)
    mid = n // 2

    tail_r = 0 if dr > 0 else (n - 1 if dr < 0 else mid)
    tail_c = 0 if dc > 0 else (n - 1 if dc < 0 else mid)
    tip_r = n - 1 if dr > 0 else (0 if dr < 0 else mid)
    tip_c = n - 1 if dc > 0 else (0 if dc < 0 else mid)

    draw_line(mask, tail_r, tail_c, end_r=tip_r, end_c=tip_c, color=1)

    perp1 = (-dc, dr)
    perp2 = (dc, -dr)
    base_r = tip_r - dr
    base_c = tip_c - dc
    for wr, wc in ((base_r + perp1[0], base_c + perp1[1]), (base_r + perp2[0], base_c + perp2[1])):
        if 0 <= wr < n and 0 <= wc < n:
            mask[wr, wc] = 1
    mask[tip_r, tip_c] = 1
    return mask


def render_hint_sprite(n: int, direction: tuple[int, int]) -> np.ndarray:
    return arrow_mask(n, direction) * HINT_COLOR


def decode_hint_direction(inner: np.ndarray) -> tuple[int, int] | None:
    if inner.size == 0 or np.any((inner != 0) & (inner != HINT_COLOR)):
        return None
    mask = (inner == HINT_COLOR).astype(int)
    for direction in DIRECTIONS:
        if np.array_equal(mask, arrow_mask(inner.shape[0], direction)):
            return direction
    return None


def box_direction(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> tuple[int, int]:
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b
    ca_r = ar0 + ar1
    ca_c = ac0 + ac1
    cb_r = br0 + br1
    cb_c = bc0 + bc1
    return sign(cb_r - ca_r), sign(cb_c - ca_c)


def identify_training_pair(inners: list[tuple[tuple[int, int, int, int], np.ndarray]], ignore_idx: int):
    candidates = [i for i in range(len(inners)) if i != ignore_idx]
    sigs = []
    for i in candidates:
        inner = inners[i][1]
        nz = tuple(sorted(set(int(x) for x in inner.ravel() if int(x) != 0)))
        sigs.append((i, nz))
    pair = None
    for a in range(len(sigs)):
        for b in range(a + 1, len(sigs)):
            if sigs[a][1] and sigs[a][1] == sigs[b][1]:
                pair = (sigs[a][0], sigs[b][0])
                break
        if pair is not None:
            break
    return pair


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover a framed-sprite transformation and apply it to the test sprite, using the hint frame to orient the example pair.

    Input structure:
    - The grid contains four non-overlapping hollow square frames of the same size on a noisy background.
    - Two frames form the training example because their interiors use the same non-zero color set; a third frame is the test sprite.
    - The remaining frame is no longer empty: it contains a color-7 arrow that points from the training source frame toward the training destination frame.

    Rule:
    - Detect the hint frame by its reserved arrow color, decode the arrow direction, and use that direction to order the two matched training frames.
    - Infer the sprite transform from the ordered pair (rotation/reflection, fill, endpoints, inversion, etc.) exactly as before.
    - Apply that learned transform to the test sprite and return only the transformed test interior.

    Key insight:
    - The extra frame is now active metadata: it tells you how to read the analogy, not what to draw directly. The puzzle is “arrow says A -> B, so apply the same rule to C”.
    """
    g = np.asarray(grid, dtype=int)
    frames = find_square_frames(g)
    if len(frames) < 4:
        return g.copy()

    inners = []
    for _, _, r0, c0, r1, c1 in frames:
        inners.append(((r0, c0, r1, c1), g[r0 + 1 : r1, c0 + 1 : c1]))

    hint_candidates = [i for i, (_, inner) in enumerate(inners) if HINT_COLOR in inner]
    if len(hint_candidates) != 1:
        return g.copy()
    hint_idx = hint_candidates[0]
    hint_direction = decode_hint_direction(inners[hint_idx][1])
    if hint_direction is None:
        return g.copy()

    pair = identify_training_pair(inners, hint_idx)
    if pair is None:
        return g.copy()

    remaining = [i for i in range(len(inners)) if i not in pair and i != hint_idx]
    if len(remaining) != 1:
        return g.copy()
    test_idx = remaining[0]

    pair_dir = box_direction(inners[pair[0]][0], inners[pair[1]][0])
    if pair_dir == hint_direction:
        src_idx, dst_idx = pair
    elif (-pair_dir[0], -pair_dir[1]) == hint_direction:
        src_idx, dst_idx = pair[1], pair[0]
    else:
        return g.copy()

    src = inners[src_idx][1]
    dst = inners[dst_idx][1]
    test_inner = inners[test_idx][1]

    tpl = infer_template(src, dst)
    if tpl is None:
        return test_inner.copy()
    kind, params = tpl
    return apply_template(kind, params, test_inner)


def inverse_symmetry(arr: np.ndarray, idx: int) -> np.ndarray:
    if idx == 0:
        return arr.copy()
    if idx == 1:
        return np.fliplr(arr)
    if idx == 2:
        return np.rot90(arr, -1)
    if idx == 3:
        return np.rot90(np.fliplr(arr), -1)
    if idx == 4:
        return np.rot90(arr, 2)
    if idx == 5:
        return np.flipud(arr)
    if idx == 6:
        return np.rot90(arr, 1)
    if idx == 7:
        return np.rot90(np.fliplr(arr), 1)
    raise ValueError(idx)


def choose_palette(rng: random.Random, count: int, forbidden: set[int]) -> list[int]:
    pool = [c for c in SPRITE_COLORS if c not in forbidden]
    if len(pool) < count:
        raise ValueError("not enough sprite colors")
    return rng.sample(pool, count)


def random_sparse_pattern(n: int, colors: list[int], rng: random.Random) -> np.ndarray:
    arr = np.zeros((n, n), dtype=int)
    count = rng.randint(3, min(SPARSE_COUNT_MAX, n * n - 1))
    cells = rng.sample([(r, c) for r in range(n) for c in range(n)], count)
    used = [colors[0]] * count
    if len(colors) == 2:
        for i in range(count):
            used[i] = colors[rng.randint(0, 1)]
        used[0] = colors[0]
        used[1] = colors[1]
    for (r, c), col in zip(cells, used):
        arr[r, c] = col
    return arr


def build_sym_pair(n: int, colors: list[int], rng: random.Random, idx: int) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(60):
        dst = random_sparse_pattern(n, colors, rng)
        src = inverse_symmetry(dst, idx)
        if np.array_equal(src, dst):
            continue
        inferred = infer_template(src, dst)
        reverse = infer_template(dst, src)
        if inferred is None or inferred[0] != "sym" or inferred[1]["idx"] != idx:
            continue
        if reverse is None or reverse[0] != "sym":
            continue
        if reverse[1]["idx"] == idx:
            continue
        return src, dst
    raise ValueError("failed to build directional sym pair")


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


def _valid_nz(arr: np.ndarray) -> bool:
    return SPRITE_NZ_MIN <= int(np.count_nonzero(arr)) <= SPRITE_NZ_MAX


def choose_rule_setup(n: int, rng: random.Random) -> tuple[str, dict, int]:
    return "sym", {"idx": rng.choice(ROTATION_IDXS)}, rng.choice([1, 2])


def build_training_pair(kind: str, params: dict, n: int, colors: list[int], rng: random.Random):
    if kind != "sym":
        raise ValueError(kind)
    return build_sym_pair(n, colors, rng, params["idx"])


def _build_sprites(kind: str, setup_params: dict, n: int, p1: list[int], p2: list[int], rng: random.Random):
    train_src, train_dst = build_training_pair(kind, setup_params, n, p1, rng)
    inferred = infer_template(train_src, train_dst)
    reverse = infer_template(train_dst, train_src)
    if inferred is None or inferred[0] != "sym":
        return None
    if reverse is None or reverse[0] != "sym":
        return None

    test_src = random_sparse_pattern(n, p2, rng)
    test_dst = apply_template(inferred[0], inferred[1], test_src)
    reverse_dst = apply_template(reverse[0], reverse[1], test_src)

    if np.array_equal(test_dst, test_src) or np.array_equal(test_dst, reverse_dst):
        return None
    for arr in (train_src, train_dst, test_src, test_dst):
        if not _valid_nz(arr):
            return None
    train_sig = tuple(colors_present(train_src, exclude={0}))
    if train_sig != tuple(colors_present(train_dst, exclude={0})):
        return None
    if tuple(colors_present(test_src, exclude={0})) == train_sig:
        return None
    return train_src, train_dst, test_src, test_dst


def _choose_frame_and_bg(used: set[int], rng: random.Random) -> tuple[int, list[int]]:
    frame_color = rng.choice([c for c in ALL_COLORS if c not in used and c != HINT_COLOR])
    bg_pool = [c for c in ALL_COLORS if c not in used and c not in {frame_color, HINT_COLOR}]
    feasible_periods = [p for p in BG_PERIOD_CHOICES if p <= len(bg_pool)]
    bg_palette = rng.sample(bg_pool, rng.choice(feasible_periods))
    return frame_color, bg_palette


def _assign_slots(rng: random.Random) -> tuple[int, int, int, int]:
    src_slot, dst_slot = rng.sample(range(4), 2)
    remaining = [i for i in range(4) if i not in {src_slot, dst_slot}]
    test_slot = rng.choice(remaining)
    hint_slot = remaining[0] if remaining[0] != test_slot else remaining[1]
    return src_slot, dst_slot, test_slot, hint_slot


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


def validate_scene(grid: np.ndarray, expected_output: np.ndarray) -> bool:
    frames = find_square_frames(grid)
    if len(frames) != 4:
        return False
    sizes = {size for size, _, *_ in frames}
    if len(sizes) != 1 or next(iter(sizes)) not in set(FRAME_SIZE_CHOICES):
        return False

    inners = [grid[r0 + 1 : r1, c0 + 1 : c1] for _, _, r0, c0, r1, c1 in frames]
    hint_idxs = [i for i, inner in enumerate(inners) if HINT_COLOR in inner]
    if len(hint_idxs) != 1:
        return False
    if decode_hint_direction(inners[hint_idxs[0]]) is None:
        return False

    sigs = []
    for i, inner in enumerate(inners):
        if i == hint_idxs[0]:
            continue
        sigs.append(tuple(colors_present(inner, exclude={0})))
    dup_count = sum(1 for i in range(len(sigs)) for j in range(i + 1, len(sigs)) if sigs[i] and sigs[i] == sigs[j])
    if dup_count != 1:
        return False
    return np.array_equal(main(grid), expected_output)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    s = rng.choice(FRAME_SIZE_CHOICES)
    n = s - 2

    kind, setup_params, color_count = choose_rule_setup(n, rng)
    p1 = choose_palette(rng, color_count, set())
    p2 = choose_palette(rng, color_count, set(p1))

    built = _build_sprites(kind, setup_params, n, p1, p2, rng)
    if built is None:
        return None
    train_src, train_dst, test_src, test_dst = built

    frame_color, bg_palette = _choose_frame_and_bg(set(p1) | set(p2), rng)
    positions = quadrant_positions(s, rng)
    src_slot, dst_slot, test_slot, hint_slot = _assign_slots(rng)
    hint_dir = box_direction(
        (positions[src_slot][0], positions[src_slot][1], positions[src_slot][0] + s - 1, positions[src_slot][1] + s - 1),
        (positions[dst_slot][0], positions[dst_slot][1], positions[dst_slot][0] + s - 1, positions[dst_slot][1] + s - 1),
    )

    sprites = [np.zeros((n, n), dtype=int) for _ in range(4)]
    sprites[src_slot] = train_src
    sprites[dst_slot] = train_dst
    sprites[test_slot] = test_src
    sprites[hint_slot] = render_hint_sprite(n, hint_dir)

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
