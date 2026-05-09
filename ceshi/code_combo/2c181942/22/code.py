# Combo variant of 2c181942 (index 22)
# combo_spec: D3/3+D5/0
# Name: 缩放变换+矩形hub

from __future__ import annotations

import random

import numpy as np
from scipy.ndimage import label as _label

from arc_utils import bbox_coords, get_bg, label_mask, paste, random_free_location, scale_sprite

# Design analysis for the combo variant
# -------------------------------------
# spec:
#   preserved_rule:
#     - 仍然是“中央 hub + 按颜色匹配的单色 sprite”。
#     - 输出仍然只保留 hub 和贴附后的匹配 sprite，其他内容清空为背景色。
#   fused_change:
#     - D5/0: hub 从 4x4 空心正方形改为固定 5x6 空心矩形。
#       因此 top/bot 的目标接触长度是 4，left/right 的目标接触长度是 3。
#     - D3/3: 不再尝试 rot90。每个 sprite 在输入里已经是对应 side 的朝向，
#       solver 要测量 sprite 朝向 hub 的接触边长度，再按目标边长做整数缩放。
#   concrete_behavior_trace:
#     - 一个具体小例子：
#       top sprite 的原始底边接触段长为 2，而 hub 顶边长度为 4，所以输出里该 sprite 按 x2 缩放后贴到顶边。
#       left sprite 的原始右边接触段长为 1，而 hub 左边长度为 3，所以输出里该 sprite 按 x3 缩放后贴到左边。
#     - 原题是在“颜色配对后旋转到唯一合法姿态”；本融合版是在“颜色配对后保持朝向不变，按 side 的边长比例缩放”。
#     - 因为 hub 是矩形，所以水平边和垂直边天然有不同目标长度；再配合输入 sprite 的不同原始接触边长度，
#       样例中会稳定出现两档缩放因子：水平边 x2、垂直边 x3。
#   complexity_calibration:
#     - hub 尺寸固定成 5x6，而不是开放很多矩形档位，避免搜索空间过大。
#     - 所有样例都包含四个 side，对解题者来说会反复看到“上/下都放大 2 倍，左/右都放大 3 倍”的重复证据。
#     - 输入 sprite 只在平移位置上随机，不再额外旋转；否则“旋转”和“缩放”两套解释会互相干扰。
#   implicit_assumptions_review:
#     - 原 `_find_hub()` 默认 hub 是 `n x n`；这里改成固定 5x6。
#     - 原 solver 依赖“尝试 4 种旋转后找到唯一合法接触边”；这里该假设失效，改为直接读取当前朝向上的接触边。
#     - 原 `target = n - 2` 把四边长度绑定成同一个值；这里拆成 horizontal=4, vertical=3。
#     - 新 solver 额外假设：目标边长能被输入 sprite 的原始接触边长整除，从而得到整数缩放因子。
#       generator 固定水平 sprite 接触边长为 2、垂直 sprite 接触边长为 1，保证该条件成立。
#     - 新 solver 还假设每种边色在 hub 外只有一个主 8 连通分量；generator 不会制造同色竞争主体。
#   placement_legality:
#     - generator 继续采用 output-first：先构造缩放后的正确输出，再把未缩放 sprite 散落回输入。
#     - attached 阶段逐个检查越界和重叠；scatter 阶段使用 `random_free_location(..., padding=2)` 保证 sprite 彼此和 hub 留白。
#   visual_observability:
#     - 人类从 2-4 个样例中可以直接看到：hub 不是方形，且贴到上下的 sprite 明显放大到更宽，
#       贴到左右的 sprite 明显放大到更高；输入与输出的朝向保持一致，变化集中在尺寸。
#     - 缩放只依赖可见的几何长度，不依赖颜色数值大小或其他不可见属性。
#   visual_cleanliness:
#     - 只使用单色 sprite、单个中央 hub、无额外图例；对象间有固定 padding，避免挤在一起。
#     - 不加 distractor，避免缩放线索被无关色块稀释。
#   generator_checks_for_new_solver_assumptions:
#     - 每个 base mask 都会验证：8 连通、目标侧接触边长度正确、并且缩放后仍然是清晰单体。
#     - `main(generate_input()) == output` 的 oracle 检查保证 solver 与 generator 的几何解释一致。

HUB_H = 5
HUB_W = 6
HORIZ_TARGET = HUB_W - 2
VERT_TARGET = HUB_H - 2
SIDES = ("top", "bot", "left", "right")

_TOP_CANONICAL = [
    np.array(
        [
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 0],
            [0, 1, 1],
            [0, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
        ],
        dtype=bool,
    ),
]

_LEFT_CANONICAL = [
    np.array(
        [
            [1, 0],
            [1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 0],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 0, 0],
            [1, 1, 0],
        ],
        dtype=bool,
    ),
]


def _crop_to_mask(arr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(pts)
    return arr[r0 : r1 + 1, c0 : c1 + 1], mask[r0 : r1 + 1, c0 : c1 + 1]


def _edge_segment(mask: np.ndarray, side: str):
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None
    if side == "top":
        rr = int(pts[:, 0].min())
        cols = sorted(int(x) for x in pts[pts[:, 0] == rr][:, 1])
        return ("h", rr, cols)
    if side == "bot":
        rr = int(pts[:, 0].max())
        cols = sorted(int(x) for x in pts[pts[:, 0] == rr][:, 1])
        return ("h", rr, cols)
    if side == "left":
        cc = int(pts[:, 1].min())
        rows = sorted(int(x) for x in pts[pts[:, 1] == cc][:, 0])
        return ("v", cc, rows)
    if side == "right":
        cc = int(pts[:, 1].max())
        rows = sorted(int(x) for x in pts[pts[:, 1] == cc][:, 0])
        return ("v", cc, rows)
    return None


def _is_contig(seq: list[int]) -> bool:
    return bool(seq) and all(int(seq[i]) + 1 == int(seq[i + 1]) for i in range(len(seq) - 1))


def _target_len_for_side(side: str) -> int:
    return HORIZ_TARGET if side in ("top", "bot") else VERT_TARGET


def _need_edge_for_side(side: str) -> str:
    if side == "top":
        return "bot"
    if side == "bot":
        return "top"
    if side == "left":
        return "right"
    return "left"


def _contact_len(mask: np.ndarray, side: str) -> int:
    seg = _edge_segment(mask, side)
    if seg is None:
        return -1
    seq = seg[2]
    if not _is_contig(seq):
        return -1
    return int(len(seq))


def _scale_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    return scale_sprite(mask.astype(int), int(factor)).astype(bool)


def _scale_arr(arr: np.ndarray, factor: int) -> np.ndarray:
    return scale_sprite(arr, int(factor)).astype(int)


def _validate_mask(mask: np.ndarray, need_edge: str, expected_len: int) -> None:
    labeled, num = _label(mask.astype(int), structure=np.ones((3, 3), dtype=int))
    assert int(num) == 1
    assert _contact_len(mask, need_edge) == int(expected_len)
    _ = labeled


def _make_sprite_mask(side: str, rng: random.Random) -> np.ndarray:
    if side in ("top", "bot"):
        base = rng.choice(_TOP_CANONICAL).copy()
        mask = base if side == "top" else np.flipud(base)
    else:
        base = rng.choice(_LEFT_CANONICAL).copy()
        mask = base if side == "left" else np.fliplr(base)

    _validate_mask(mask, _need_edge_for_side(side), 2 if side in ("top", "bot") else 1)
    return mask


def _build_hub(out: np.ndarray, hub_r: int, hub_c: int, side_colors: dict[str, int], bg: int) -> None:
    out[hub_r : hub_r + HUB_H, hub_c : hub_c + HUB_W] = int(bg)
    out[hub_r, hub_c + 1 : hub_c + HUB_W - 1] = int(side_colors["top"])
    out[hub_r + HUB_H - 1, hub_c + 1 : hub_c + HUB_W - 1] = int(side_colors["bot"])
    out[hub_r + 1 : hub_r + HUB_H - 1, hub_c] = int(side_colors["left"])
    out[hub_r + 1 : hub_r + HUB_H - 1, hub_c + HUB_W - 1] = int(side_colors["right"])


def _attachment_offset(mask: np.ndarray, side: str, hub_r: int, hub_c: int) -> tuple[int, int]:
    if side == "top":
        _typ, edge_r, cols = _edge_segment(mask, "bot")
        return int(hub_r - edge_r - 1), int(hub_c + 1 - cols[0])
    if side == "bot":
        _typ, edge_r, cols = _edge_segment(mask, "top")
        return int(hub_r + HUB_H - edge_r), int(hub_c + 1 - cols[0])
    if side == "left":
        _typ, edge_c, rows = _edge_segment(mask, "right")
        return int(hub_r + 1 - rows[0]), int(hub_c - edge_c - 1)
    _typ, edge_c, rows = _edge_segment(mask, "left")
    return int(hub_r + 1 - rows[0]), int(hub_c + HUB_W - edge_c)


def _place_patch(out: np.ndarray, arr: np.ndarray, mask: np.ndarray, sr: int, sc: int, bg: int) -> bool:
    h, w = mask.shape
    H, W = out.shape
    if sr < 0 or sc < 0 or sr + h > H or sc + w > W:
        return False
    for rr in range(h):
        for cc in range(w):
            if bool(mask[rr, cc]) and int(out[sr + rr, sc + cc]) != int(bg):
                return False
    paste(out, arr, int(sr), int(sc), transparent=int(bg))
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Attach monochrome sprites to the matching sides of a hollow 5x6 rectangle.

    Fused rule:
    - The hub is a hollow rectangle, not a square.
    - Sprites are already in the correct orientation for their target side.
    - For each edge color, measure the sprite's contact edge on the side facing the hub,
      compute `target_len / source_len`, scale the whole sprite by that integer factor,
      then paste it flush to the matching hub side.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    def _find_hub():
        best = None  # (score, (r,c), (top,bot,left,right))
        for r in range(0, H - HUB_H + 1):
            for c in range(0, W - HUB_W + 1):
                if (
                    int(g[r, c]) != int(bg)
                    or int(g[r, c + HUB_W - 1]) != int(bg)
                    or int(g[r + HUB_H - 1, c]) != int(bg)
                    or int(g[r + HUB_H - 1, c + HUB_W - 1]) != int(bg)
                ):
                    continue

                inner = g[r + 1 : r + HUB_H - 1, c + 1 : c + HUB_W - 1]
                if not bool((inner == int(bg)).all()):
                    continue

                top = g[r, c + 1 : c + HUB_W - 1]
                bot = g[r + HUB_H - 1, c + 1 : c + HUB_W - 1]
                left = g[r + 1 : r + HUB_H - 1, c]
                right = g[r + 1 : r + HUB_H - 1, c + HUB_W - 1]

                if (
                    bool((top == int(bg)).any())
                    or bool((bot == int(bg)).any())
                    or bool((left == int(bg)).any())
                    or bool((right == int(bg)).any())
                ):
                    continue

                if (
                    len(set(map(int, top))) != 1
                    or len(set(map(int, bot))) != 1
                    or len(set(map(int, left))) != 1
                    or len(set(map(int, right))) != 1
                ):
                    continue

                side_cols = (int(top[0]), int(bot[0]), int(left[0]), int(right[0]))
                if len(set(side_cols)) != 4:
                    continue

                cr = float(r + (HUB_H - 1) / 2)
                cc = float(c + (HUB_W - 1) / 2)
                score = abs(float(cr - (H - 1) / 2)) + abs(float(cc - (W - 1) / 2))
                cand = (float(score), (int(r), int(c)), side_cols)
                if best is None or float(cand[0]) < float(best[0]):
                    best = cand
        return best

    hub = _find_hub()
    assert hub is not None

    _score, (r, c), (top_col, bot_col, left_col, right_col) = hub
    hub_mask = np.zeros((H, W), dtype=bool)
    hub_mask[r : r + HUB_H, c : c + HUB_W] = True

    out = np.full_like(g, int(bg))
    out[r : r + HUB_H, c : c + HUB_W] = g[r : r + HUB_H, c : c + HUB_W]

    segs = {
        "top": (int(top_col), ("bot", int(r - 1), list(range(int(c + 1), int(c + HUB_W - 1))), HORIZ_TARGET)),
        "bot": (int(bot_col), ("top", int(r + HUB_H), list(range(int(c + 1), int(c + HUB_W - 1))), HORIZ_TARGET)),
        "left": (int(left_col), ("right", int(c - 1), list(range(int(r + 1), int(r + HUB_H - 1))), VERT_TARGET)),
        "right": (int(right_col), ("left", int(c + HUB_W), list(range(int(r + 1), int(r + HUB_H - 1))), VERT_TARGET)),
    }

    for _side, (col, (need_edge, fixed, span, target)) in segs.items():
        m = (g == int(col)) & (~hub_mask)
        if not bool(m.any()):
            continue

        labc, nc = label_mask(m, connectivity=8)
        best_id = None
        best_sz = -1
        for cid in range(1, int(nc) + 1):
            sz = int((labc == int(cid)).sum())
            if sz > best_sz:
                best_sz = sz
                best_id = int(cid)
        if best_id is None or best_sz <= 0:
            continue

        spr_mask = labc == int(best_id)
        spr_arr = g.copy()
        spr_arr[~spr_mask] = int(bg)
        spr_arr, spr_mask = _crop_to_mask(spr_arr, spr_mask)

        source_len = _contact_len(spr_mask, need_edge)
        if source_len <= 0 or int(target) % int(source_len) != 0:
            continue

        factor = int(target) // int(source_len)
        scaled_arr = _scale_arr(spr_arr, factor)
        scaled_mask = _scale_mask(spr_mask, factor)
        scaled_arr, scaled_mask = _crop_to_mask(scaled_arr, scaled_mask)

        seg = _edge_segment(scaled_mask, need_edge)
        if seg is None:
            continue
        seq = seg[2]
        if len(seq) != int(target) or not _is_contig(seq):
            continue

        if need_edge in ("left", "right"):
            _typ, edge_c, rows = seg
            dr = int(span[0] - rows[0])
            dc = int(fixed - edge_c)
        else:
            _typ, edge_r, cols = seg
            dr = int(fixed - edge_r)
            dc = int(span[0] - cols[0])

        placed = {(int(rr + dr), int(cc2 + dc)) for rr, cc2 in np.argwhere(scaled_mask)}
        if any(not (0 <= rr < H and 0 <= cc2 < W) for rr, cc2 in placed):
            continue

        ok = True
        for rr, cc2 in placed:
            val = int(scaled_arr[int(rr - dr), int(cc2 - dc)])
            if val == int(bg):
                continue
            if int(out[int(rr), int(cc2)]) != int(bg) and int(out[int(rr), int(cc2)]) != val:
                ok = False
                break
        if not ok:
            continue

        paste(out, scaled_arr, int(dr), int(dc), mask=scaled_mask, overwrite=True, clip=True)

    return out


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 8

    H = rng.randint(23, 28)
    W = rng.randint(26, 32)

    all_colors = [c for c in range(10) if c != bg]
    rng.shuffle(all_colors)
    top_col, bot_col, left_col, right_col = all_colors[:4]
    side_colors = {"top": top_col, "bot": bot_col, "left": left_col, "right": right_col}
    sprite_sides = list(SIDES)

    input_masks = {side: _make_sprite_mask(side, rng) for side in sprite_sides}
    scaled_masks = {}
    for side, mask in input_masks.items():
        source_len = _contact_len(mask, _need_edge_for_side(side))
        target_len = _target_len_for_side(side)
        if source_len <= 0 or target_len % source_len != 0:
            return None
        factor = target_len // source_len
        scaled_masks[side] = _scale_mask(mask, factor)

    max_extent = max(max(mask.shape) for mask in scaled_masks.values())
    margin = max_extent + 2

    hub_r_min = margin
    hub_r_max = H - HUB_H - margin
    hub_c_min = margin
    hub_c_max = W - HUB_W - margin
    if hub_r_min > hub_r_max or hub_c_min > hub_c_max:
        return None

    hub_r = rng.randint(hub_r_min, hub_r_max)
    hub_c = rng.randint(hub_c_min, hub_c_max)

    out = np.full((H, W), int(bg), dtype=int)
    _build_hub(out, hub_r, hub_c, side_colors, bg)

    attached_sprites: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for side in sprite_sides:
        mask = scaled_masks[side]
        arr = np.where(mask, int(side_colors[side]), int(bg)).astype(int)
        sr, sc = _attachment_offset(mask, side, hub_r, hub_c)
        if not _place_patch(out, arr, mask, int(sr), int(sc), bg):
            return None
        attached_sprites[side] = (arr, mask)

    inp = np.full((H, W), int(bg), dtype=int)
    _build_hub(inp, hub_r, hub_c, side_colors, bg)
    canvas = inp.copy()

    for side in sprite_sides:
        mask = input_masks[side]
        arr = np.where(mask, int(side_colors[side]), int(bg)).astype(int)
        try:
            pr, pc = random_free_location(canvas, arr, transparent=int(bg), padding=2, padding_connectivity=8)
        except ValueError:
            return None
        paste(canvas, arr, int(pr), int(pc), transparent=int(bg))
        paste(inp, arr, int(pr), int(pc), transparent=int(bg))

    result = main(inp)
    if not np.array_equal(result, out):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 300 attempts")
