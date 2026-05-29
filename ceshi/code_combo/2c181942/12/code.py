# Combo variant of 2c181942 (index 12)
# combo_spec: D2/1+D5/0
# Name: 尺寸排序匹配+矩形hub

from __future__ import annotations

import random

import numpy as np
from scipy.ndimage import label as _label

from arc_utils import bbox_coords, get_bg, label_mask, paste, random_free_location

# Design analysis for the combo variant
# -------------------------------------
# spec:
#   preserved_rule:
#     - 输入仍然包含一个中心附近的空心 hub 和若干散落的单色 sprite。
#     - 输出仍然只保留 hub 与贴附后的 sprite，其余位置清空为背景色。
#     - sprite 的唯一运动方式仍是“旋转后贴到某条 hub 边上”，不做镜像。
#   fused_change:
#     - D5/0: hub 从 4x4 方形改成固定 7x8 的矩形外框，内部保持全背景色。
#     - D2/1: 不再按颜色配对；改为测量 sprite 接触边长度与 hub 边长度，
#       将两者都按长度从大到小排序后建立对应关系。
#   concreteization:
#     - 为了让“排序”真实参与推理，hub 四条边的有效彩色段长度做成四个不同档位：
#       6, 5, 4, 3。这样不会出现矩形天然带来的上下同长/左右同长并列歧义。
#     - 具体做法是：hub 外接框固定为 7x8，但每条边只占据本边上的一段连续彩色 run，
#       run 长度分别为 6/5/4/3，且四条边颜色互不相同。
#     - sprite 颜色与 hub 颜色彻底脱钩：generator 让二者使用两组互不重叠的颜色。
#       因而原题“颜色=边”的匹配规则在本变体中失效，解题者必须观察长度关系。
#   behavior_trace:
#     - 原题：先看 hub 某边的颜色，再去找同色 sprite，随后只验证旋转是否满足固定接触长度。
#     - 本变体：先识别 hub 四边的四个长度档位，再识别每个 sprite 在某个旋转下能提供的
#       接触长度，按 6>5>4>3 排序后一一对应，最后把对应 sprite 贴到该边。
#     - 例如：如果输入里最长的 sprite 接触边是 6，则它必须贴到 hub 的 6 格边；即使它的颜色
#       与该边完全不同，也照样贴过去。第二长贴 5，第三长贴 4，最短贴 3。
#   complexity_calibration:
#     - 为避免规则退化，generator 总是放 4 个目标 sprite，对应四个不同长度档位。
#     - 颜色总数被控制在 8 个非背景色：4 个 hub 色 + 4 个 sprite 色，不再引入额外颜色逻辑。
#     - hub 外框尺寸固定，只让“边长档位分配”和“sprite 摆放旋转”随机化；这样视觉上既明显不同，
#       又不会因为自由度太多导致样例难以归纳。
#   implicit_assumptions_review:
#     - 原 `_find_hub()` 假设 hub 是 NxN 方形并且四边目标长度统一为 `n-2`；这里两点都失效。
#     - 原 solver 按边颜色逐边找 sprite；这里颜色不再是匹配键，因此必须先收集所有 sprite，
#       再按“可匹配的接触长度”排序后统一配对。
#     - 原 generator 允许缺边；这里若缺边则排序信息不完整，故强制四边全都出现。
#     - 原矩形单变体只区分水平/垂直两种长度；这里继续细化成 6/5/4/3 四档，避免并列。
#   placement_legality:
#     - output-first：先构造 hub+贴附 sprite 的真输出，再把 sprite 旋转散落回输入。
#     - 水平 sprite 只向上/下伸展，垂直 sprite 只向左/右伸展，因此 attached 阶段天然不互相覆盖。
#     - scatter 阶段使用 `random_free_location(..., padding=2, padding_connectivity=8)` 保证与 hub、
#       彼此之间留出清晰间隙。
#   visual_observability:
#     - 人类在 2-4 组样例中能直接看到：hub 四边长度各不相同；sprite 颜色与 hub 色不相等；
#       输出却总是按长度档位重新贴附。这给出了足够强的结构线索。
#     - 排序仅依赖可见的连续像素长度，不依赖颜色值大小、生成顺序或其它不可视因素。
#   cleanliness:
#     - 不放额外干扰色块，避免网格被大量无关对象污染。
#     - hub 固定居中附近，sprite 散落时保持间距，输出结构紧凑清晰。
#   new_solver_assumptions_checked_against_generator:
#     - solver 假设唯一 hub 是中心附近的 7x8 空心矩形，且四边 run 长度集合恰为 {6,5,4,3}；
#       generator 严格按此构造。
#     - solver 假设每个目标 sprite 是单色 8 连通组件，并且只对应一个长度档位；
#       generator 仅从经过该条件验证的模板池中取样。
#     - solver 假设每个长度档位恰好有一个目标 sprite；generator 总是生成四个目标 sprite。

HUB_H = 7
HUB_W = 8
INNER_H = HUB_H - 2
INNER_W = HUB_W - 2
SIDES = ("top", "bot", "left", "right")
SIDE_NEED_EDGE = {"top": "bot", "bot": "top", "left": "right", "right": "left"}

TARGET_LENGTHS = (6, 5, 4, 3)
HORIZONTAL_LENGTHS = (6, 5)
VERTICAL_LENGTHS = (4, 3)

_CANONICAL_MASKS = {
    6: [
        np.array(
            [
                [0, 0, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
            ],
            dtype=bool,
        )
    ],
    5: [
        np.array(
            [
                [0, 0, 1, 0, 0],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
            ],
            dtype=bool,
        )
    ],
    4: [
        np.array(
            [
                [0, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
                [0, 1, 1],
            ],
            dtype=bool,
        ),
        np.array(
            [
                [0, 0, 1],
                [1, 1, 1],
                [1, 1, 1],
                [0, 0, 1],
            ],
            dtype=bool,
        ),
    ],
    3: [
        np.array(
            [
                [0, 1, 1],
                [1, 1, 1],
                [0, 1, 1],
            ],
            dtype=bool,
        ),
        np.array(
            [
                [0, 0, 1],
                [1, 1, 1],
                [0, 0, 1],
            ],
            dtype=bool,
        ),
    ],
}


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


def _matches_attachment(mask: np.ndarray, need_edge: str, target_len: int) -> bool:
    if need_edge in ("left", "right"):
        if not bool(np.array_equal(mask, np.flipud(mask))):
            return False
    else:
        if not bool(np.array_equal(mask, np.fliplr(mask))):
            return False

    seg = _edge_segment(mask, need_edge)
    if seg is None:
        return False

    if need_edge in ("left", "right"):
        _typ, _edge_c, rows = seg
        return len(rows) == int(target_len) and _is_contig(rows)

    _typ, _edge_r, cols = seg
    return len(cols) == int(target_len) and _is_contig(cols)


def _count_valid_rotations(mask: np.ndarray, need_edge: str, target_len: int) -> int:
    count = 0
    arr = mask.astype(int)
    for krot in range(4):
        rot_arr = np.rot90(arr, krot)
        rot_mask = np.rot90(mask, krot)
        rot_arr, rot_mask = _crop_to_mask(rot_arr, rot_mask)
        if _matches_attachment(rot_mask, need_edge, target_len):
            count += 1
    return count


def _possible_target_lengths(mask: np.ndarray) -> set[int]:
    found: set[int] = set()
    arr = mask.astype(int)
    for krot in range(4):
        rot_arr = np.rot90(arr, krot)
        rot_mask = np.rot90(mask, krot)
        rot_arr, rot_mask = _crop_to_mask(rot_arr, rot_mask)
        for target_len in HORIZONTAL_LENGTHS:
            if _matches_attachment(rot_mask, "top", int(target_len)) or _matches_attachment(rot_mask, "bot", int(target_len)):
                found.add(int(target_len))
        for target_len in VERTICAL_LENGTHS:
            if _matches_attachment(rot_mask, "left", int(target_len)) or _matches_attachment(rot_mask, "right", int(target_len)):
                found.add(int(target_len))
    return found


def _line_run(values: np.ndarray, bg: int) -> tuple[int, int, int, list[int]] | None:
    seq = [int(v) for v in values]
    nz = [i for i, v in enumerate(seq) if int(v) != int(bg)]
    if not nz:
        return None
    start = int(nz[0])
    end = int(nz[-1])
    if any(int(v) != int(bg) for v in seq[:start]):
        return None
    if any(int(v) != int(bg) for v in seq[end + 1 :]):
        return None
    color = int(seq[start])
    if color == int(bg):
        return None
    if any(int(v) != color for v in seq[start : end + 1]):
        return None
    idxs = list(range(int(start), int(end) + 1))
    return int(start), int(end), int(color), idxs


def _span_positions(side: str, start: int, length: int) -> list[int]:
    if side in ("top", "bot"):
        return [int(1 + start + i) for i in range(int(length))]
    return [int(1 + start + i) for i in range(int(length))]


def _orient_mask(mask: np.ndarray, side: str) -> np.ndarray:
    if side == "top":
        return mask.copy()
    if side == "bot":
        return np.flipud(mask)
    if side == "left":
        return mask.copy()
    return np.fliplr(mask)


def _make_sprite_mask(side: str, target_len: int, rng: random.Random) -> np.ndarray:
    base = rng.choice(_CANONICAL_MASKS[int(target_len)]).copy()
    mask = _orient_mask(base, side)
    need_edge = SIDE_NEED_EDGE[side]
    assert _count_valid_rotations(mask, need_edge, int(target_len)) == 1
    assert _possible_target_lengths(mask) == {int(target_len)}
    labeled, num = _label(mask, structure=np.ones((3, 3), dtype=int))
    assert int(num) == 1
    return mask


def _build_hub(
    out: np.ndarray,
    hub_r: int,
    hub_c: int,
    side_lengths: dict[str, int],
    side_colors: dict[str, int],
    side_starts: dict[str, int],
    bg: int,
) -> dict[str, dict[str, object]]:
    out[hub_r : hub_r + HUB_H, hub_c : hub_c + HUB_W] = int(bg)
    specs: dict[str, dict[str, object]] = {}
    for side in SIDES:
        length = int(side_lengths[side])
        start = int(side_starts[side])
        positions = _span_positions(side, start, length)
        color = int(side_colors[side])
        if side == "top":
            out[hub_r, hub_c + 1 + start : hub_c + 1 + start + length] = color
            abs_span = [int(hub_c + p) for p in positions]
            fixed = int(hub_r - 1)
        elif side == "bot":
            out[hub_r + HUB_H - 1, hub_c + 1 + start : hub_c + 1 + start + length] = color
            abs_span = [int(hub_c + p) for p in positions]
            fixed = int(hub_r + HUB_H)
        elif side == "left":
            out[hub_r + 1 + start : hub_r + 1 + start + length, hub_c] = color
            abs_span = [int(hub_r + p) for p in positions]
            fixed = int(hub_c - 1)
        else:
            out[hub_r + 1 + start : hub_r + 1 + start + length, hub_c + HUB_W - 1] = color
            abs_span = [int(hub_r + p) for p in positions]
            fixed = int(hub_c + HUB_W)
        specs[side] = {
            "side": side,
            "color": color,
            "target_len": length,
            "need_edge": SIDE_NEED_EDGE[side],
            "fixed": fixed,
            "span": abs_span,
        }
    return specs


def _attachment_offset(mask: np.ndarray, side_spec: dict[str, object]) -> tuple[int, int]:
    side = str(side_spec["side"])
    if side == "top":
        _typ, edge_r, cols = _edge_segment(mask, "bot")
        return int(side_spec["fixed"]) - int(edge_r), int(side_spec["span"][0]) - int(cols[0])
    if side == "bot":
        _typ, edge_r, cols = _edge_segment(mask, "top")
        return int(side_spec["fixed"]) - int(edge_r), int(side_spec["span"][0]) - int(cols[0])
    if side == "left":
        _typ, edge_c, rows = _edge_segment(mask, "right")
        return int(side_spec["span"][0]) - int(rows[0]), int(side_spec["fixed"]) - int(edge_c)
    _typ, edge_c, rows = _edge_segment(mask, "left")
    return int(side_spec["span"][0]) - int(rows[0]), int(side_spec["fixed"]) - int(edge_c)


def _scatter_patch(mask: np.ndarray, color: int, bg: int, rng: random.Random) -> np.ndarray:
    arr = np.where(mask, int(color), int(bg)).astype(int)
    krot = rng.choice([1, 2, 3])
    rot_arr = np.rot90(arr, krot)
    rot_mask = np.rot90(mask, krot)
    rows_any = rot_mask.any(axis=1)
    cols_any = rot_mask.any(axis=0)
    r0, r1 = np.where(rows_any)[0][[0, -1]]
    c0, c1 = np.where(cols_any)[0][[0, -1]]
    return rot_arr[r0 : r1 + 1, c0 : c1 + 1]


def _find_hub(g: np.ndarray, bg: int):
    H, W = g.shape
    best = None  # (score, (r,c), side_specs)
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

            top_run = _line_run(g[r, c + 1 : c + HUB_W - 1], int(bg))
            bot_run = _line_run(g[r + HUB_H - 1, c + 1 : c + HUB_W - 1], int(bg))
            left_run = _line_run(g[r + 1 : r + HUB_H - 1, c], int(bg))
            right_run = _line_run(g[r + 1 : r + HUB_H - 1, c + HUB_W - 1], int(bg))
            if any(run is None for run in (top_run, bot_run, left_run, right_run)):
                continue

            top_start, _top_end, top_col, top_idxs = top_run
            bot_start, _bot_end, bot_col, bot_idxs = bot_run
            left_start, _left_end, left_col, left_idxs = left_run
            right_start, _right_end, right_col, right_idxs = right_run

            side_lengths = {
                "top": len(top_idxs),
                "bot": len(bot_idxs),
                "left": len(left_idxs),
                "right": len(right_idxs),
            }
            if set(side_lengths.values()) != set(TARGET_LENGTHS):
                continue

            side_cols = (int(top_col), int(bot_col), int(left_col), int(right_col))
            if len(set(side_cols)) != 4:
                continue

            side_specs = {
                "top": {
                    "side": "top",
                    "color": int(top_col),
                    "target_len": int(len(top_idxs)),
                    "need_edge": "bot",
                    "fixed": int(r - 1),
                    "span": [int(c + 1 + idx) for idx in top_idxs],
                    "start": int(top_start),
                },
                "bot": {
                    "side": "bot",
                    "color": int(bot_col),
                    "target_len": int(len(bot_idxs)),
                    "need_edge": "top",
                    "fixed": int(r + HUB_H),
                    "span": [int(c + 1 + idx) for idx in bot_idxs],
                    "start": int(bot_start),
                },
                "left": {
                    "side": "left",
                    "color": int(left_col),
                    "target_len": int(len(left_idxs)),
                    "need_edge": "right",
                    "fixed": int(c - 1),
                    "span": [int(r + 1 + idx) for idx in left_idxs],
                    "start": int(left_start),
                },
                "right": {
                    "side": "right",
                    "color": int(right_col),
                    "target_len": int(len(right_idxs)),
                    "need_edge": "left",
                    "fixed": int(c + HUB_W),
                    "span": [int(r + 1 + idx) for idx in right_idxs],
                    "start": int(right_start),
                },
            }

            cr = float(r + (HUB_H - 1) / 2)
            cc = float(c + (HUB_W - 1) / 2)
            score = abs(float(cr - (H - 1) / 2)) + abs(float(cc - (W - 1) / 2))
            cand = (float(score), (int(r), int(c)), side_specs)
            if best is None or float(cand[0]) < float(best[0]):
                best = cand
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """
    Attach sprites to the hollow rectangular hub by descending contact-edge length.

    Variant rule:
    - The hub is a fixed 7x8 hollow rectangle-like frame with four distinct edge-run lengths: 6, 5, 4, 3.
    - Sprite colors are not matching keys. Instead, each sprite is assigned to the hub side whose required
      contact length occupies the same rank when both sides and sprites are sorted from longest to shortest.
    - Rotations are still resolved by the original symmetry constraint:
      horizontal attachments require fliplr symmetry, vertical attachments require flipud symmetry.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    hub = _find_hub(g, int(bg))
    assert hub is not None

    _score, (r, c), side_specs = hub
    hub_mask = np.zeros((H, W), dtype=bool)
    hub_mask[r : r + HUB_H, c : c + HUB_W] = True

    out = np.full_like(g, int(bg))
    out[r : r + HUB_H, c : c + HUB_W] = g[r : r + HUB_H, c : c + HUB_W]

    sprite_candidates: list[dict[str, object]] = []
    comp_labels, comp_count = label_mask((g != int(bg)) & (~hub_mask), connectivity=8)
    for cid in range(1, int(comp_count) + 1):
        comp_mask = comp_labels == int(cid)
        if not bool(comp_mask.any()):
            continue
        comp_arr = g.copy()
        comp_arr[~comp_mask] = int(bg)
        comp_arr, comp_mask = _crop_to_mask(comp_arr, comp_mask)

        vals = {int(v) for v in comp_arr[comp_mask]}
        if len(vals) != 1:
            continue

        valid: list[dict[str, object]] = []
        for side_spec in side_specs.values():
            need_edge = str(side_spec["need_edge"])
            target_len = int(side_spec["target_len"])
            for krot in range(4):
                a = np.rot90(comp_arr, krot)
                msk = np.rot90(comp_mask, krot)
                a, msk = _crop_to_mask(a, msk)
                if not _matches_attachment(msk, need_edge, int(target_len)):
                    continue

                dr, dc = _attachment_offset(msk, side_spec)
                placed = {(int(rr + dr), int(cc2 + dc)) for rr, cc2 in np.argwhere(msk)}
                if any(not (0 <= rr < H and 0 <= cc2 < W) for rr, cc2 in placed):
                    continue

                ok = True
                for rr, cc2 in placed:
                    val = int(a[int(rr - dr), int(cc2 - dc)])
                    if val == int(bg):
                        continue
                    if int(out[int(rr), int(cc2)]) != int(bg) and int(out[int(rr), int(cc2)]) != val:
                        ok = False
                        break
                if not ok:
                    continue

                valid.append(
                    {
                        "side": str(side_spec["side"]),
                        "target_len": int(target_len),
                        "arr": a,
                        "mask": msk,
                        "dr": int(dr),
                        "dc": int(dc),
                    }
                )

        valid_lengths = sorted({int(item["target_len"]) for item in valid}, reverse=True)
        if len(valid_lengths) != 1:
            continue
        target_len = int(valid_lengths[0])
        chosen = next(item for item in valid if int(item["target_len"]) == int(target_len))
        sprite_candidates.append(
            {
                "target_len": int(target_len),
                "candidate": chosen,
            }
        )

    sorted_sides = sorted(side_specs.values(), key=lambda spec: -int(spec["target_len"]))
    sorted_sprites = sorted(sprite_candidates, key=lambda item: -int(item["target_len"]))

    for side_spec, sprite_info in zip(sorted_sides, sorted_sprites):
        if int(side_spec["target_len"]) != int(sprite_info["target_len"]):
            continue
        cand = sprite_info["candidate"]
        paste(out, cand["arr"], int(cand["dr"]), int(cand["dc"]), mask=cand["mask"], overwrite=True, clip=True)

    return out


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 8

    H = rng.randint(20, 26)
    W = rng.randint(22, 28)

    palette = [c for c in range(10) if c != bg]
    rng.shuffle(palette)
    hub_colors = palette[:4]
    sprite_colors = palette[4:8]

    side_colors = {
        "top": int(hub_colors[0]),
        "bot": int(hub_colors[1]),
        "left": int(hub_colors[2]),
        "right": int(hub_colors[3]),
    }
    sprite_color_map = {
        "top": int(sprite_colors[0]),
        "bot": int(sprite_colors[1]),
        "left": int(sprite_colors[2]),
        "right": int(sprite_colors[3]),
    }

    horiz_perm = list(HORIZONTAL_LENGTHS)
    vert_perm = list(VERTICAL_LENGTHS)
    rng.shuffle(horiz_perm)
    rng.shuffle(vert_perm)
    side_lengths = {
        "top": int(horiz_perm[0]),
        "bot": int(horiz_perm[1]),
        "left": int(vert_perm[0]),
        "right": int(vert_perm[1]),
    }

    side_starts = {
        "top": rng.randint(0, INNER_W - int(side_lengths["top"])),
        "bot": rng.randint(0, INNER_W - int(side_lengths["bot"])),
        "left": rng.randint(0, INNER_H - int(side_lengths["left"])),
        "right": rng.randint(0, INNER_H - int(side_lengths["right"])),
    }

    sprite_masks = {side: _make_sprite_mask(side, int(side_lengths[side]), rng) for side in SIDES}

    top_margin = max(sprite_masks["top"].shape[0], 2) + 2
    bot_margin = max(sprite_masks["bot"].shape[0], 2) + 2
    left_margin = max(sprite_masks["left"].shape[1], 2) + 2
    right_margin = max(sprite_masks["right"].shape[1], 2) + 2

    hub_r_min = top_margin
    hub_r_max = H - HUB_H - bot_margin
    hub_c_min = left_margin
    hub_c_max = W - HUB_W - right_margin
    if hub_r_min > hub_r_max or hub_c_min > hub_c_max:
        return None

    hub_r = rng.randint(hub_r_min, hub_r_max)
    hub_c = rng.randint(hub_c_min, hub_c_max)

    out = np.full((H, W), int(bg), dtype=int)
    side_specs = _build_hub(out, hub_r, hub_c, side_lengths, side_colors, side_starts, int(bg))

    attached_sprites: dict[str, tuple[np.ndarray, np.ndarray, int, int]] = {}
    for side in SIDES:
        mask = sprite_masks[side]
        arr = np.where(mask, int(sprite_color_map[side]), int(bg)).astype(int)
        sr, sc = _attachment_offset(mask, side_specs[side])
        sh, sw = mask.shape

        if sr < 0 or sc < 0 or sr + sh > H or sc + sw > W:
            return None

        for rr in range(sh):
            for cc in range(sw):
                if bool(mask[rr, cc]) and int(out[sr + rr, sc + cc]) != int(bg):
                    return None

        paste(out, arr, int(sr), int(sc), transparent=int(bg))
        attached_sprites[side] = (arr, mask, int(sr), int(sc))

    inp = np.full((H, W), int(bg), dtype=int)
    _build_hub(inp, hub_r, hub_c, side_lengths, side_colors, side_starts, int(bg))
    canvas = inp.copy()

    for side in SIDES:
        _arr, mask, _sr, _sc = attached_sprites[side]
        patch = _scatter_patch(mask, int(sprite_color_map[side]), int(bg), rng)
        try:
            pr, pc = random_free_location(canvas, patch, transparent=int(bg), padding=2, padding_connectivity=8)
        except ValueError:
            return None
        paste(canvas, patch, int(pr), int(pc), transparent=int(bg))
        paste(inp, patch, int(pr), int(pc), transparent=int(bg))

    if not np.array_equal(main(inp), out):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 300 attempts")
