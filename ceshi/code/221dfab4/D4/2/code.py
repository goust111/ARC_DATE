# Variant of 221dfab4
# Dimension: D4 — 绿色条纹的全行/列重着色范围 (trajectory)
# Change: 绿色条纹将全行/列的所有非背景像素重着色为绿色（全局 flood 效果）
#   → 绿色条纹仅重着色跨度两侧外的像素；跨度内只把背景单元补成绿色条纹，
#   已有前景像素则保留原色。
#
# Design analysis for this variant
# - Concrete behavior:
#   原题的绿色步可以概括成“整行/整列非背景 -> 绿，再把跨度内整段刷绿”。
#   如果直接把“全行/列 flood”改成“只处理跨度外”，但仍然无条件把跨度内整段刷绿，
#   那么最终输出与原题在像素级上是等价的：跨度内仍会全绿，跨度外的非背景也仍会变绿。
#   这种字面改法不会形成可观察变体。
# - Minimal observable adaptation:
#   为了让 D4 真正可见，同时尽量不动其它维度，这里把绿色步具体化为：
#   1) 跨度外：只有已有非背景像素会被重着色为绿色；
#   2) 跨度内：只对背景单元补出绿色条纹，已有前景像素保留原色。
#   小例子（水平传播，跨度列 8..11，绿色条纹在第 4 行）：
#   输入第 4 行若是 `..ff|ff..`（`|` 表示跨度边界），
#   原题会得到 `..gg|gg..`，而这里得到 `..gg|ffgg` 中跨度内原有 `f` 被保留，
#   只有跨度外的对象部分被重着色。
# - Complexity calibration:
#   若生成器仍随机撒 blob，很多样本不会有对象穿过绿色条纹的跨度，视觉上又会退化回原题。
#   所以生成器显式放置至少一个跨越跨度边界、且其中心落在绿色条纹上的 blob，
#   让“跨度内保留 / 跨度外变绿”的互补行为稳定出现。
# - Implicit assumptions reviewed:
#   1) 仍假设边界上存在单一主黄色线段；生成器只放一个黄色边段，且其它对象不用黄色。
#   2) 仍假设线段是水平或垂直直段；生成器只生成这种情况。
#   3) 新 solver 假设绿色条纹上若存在跨度内前景，应该保留；生成器保证这类命中经常存在。
#   4) gap 仍只在跨度内清空为背景；生成器允许 blob 穿过 gap/green/yellow 行列，使差异更明显。
# - Placement legality:
#   定向 bridge blob 先放，再用 `random_free_location` 放其它 blob，避免重叠和越界。
#   bridge blob 只放在离边界至少 1 格的位置，且不同绿色条纹相隔 6 格，3xN / Nx3 blob 不会互撞。
# - Visual observability:
#   人类从样例中能看到：黄色条纹仍是整段覆盖；绿色条纹则像“走廊补绿 + 两侧命中才变绿”，
#   穿过跨度边界的 blob 在同一绿色条纹上会被分裂成“内侧保原色、外侧变绿”的互补图案。

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "arc2_solver_select" / "arc_utils.py").exists():
            return parent
    raise RuntimeError("Could not locate arc2_solver_select/arc_utils.py from code.py")


REPO_ROOT = _find_repo_root()
ARC_UTILS_DIR = REPO_ROOT / "arc2_solver_select"
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(ARC_UTILS_DIR))

from arc_utils import get_bg, label_mask, random_free_location, random_sprite


Y = 4
G = 3


def _segment_info(g: np.ndarray):
    H, W = g.shape
    border = np.zeros((H, W), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    smask = (g == Y) & border
    if not bool(smask.any()):
        return None

    lab, n = label_mask(smask, connectivity=4)
    if int(n) <= 0:
        return None

    best_id = max(range(1, int(n) + 1), key=lambda cid: int((lab == int(cid)).sum()))
    pts = np.argwhere(lab == int(best_id))
    rs = pts[:, 0]
    cs = pts[:, 1]
    r0, r1 = int(rs.min()), int(rs.max())
    c0, c1 = int(cs.min()), int(cs.max())

    if r0 == r1:
        start_pos = int(r0)
        dir1 = 1 if start_pos == 0 else -1
        return {
            "orient": "h",
            "start_pos": start_pos,
            "dir1": dir1,
            "limit": H,
            "span0": int(c0),
            "span1": int(c1),
        }
    if c0 == c1:
        start_pos = int(c0)
        dir1 = 1 if start_pos == 0 else -1
        return {
            "orient": "v",
            "start_pos": start_pos,
            "dir1": dir1,
            "limit": W,
            "span0": int(r0),
            "span1": int(r1),
        }
    return None


def _green_positions(start_pos: int, dir1: int, limit: int) -> list[int]:
    pos_list: list[int] = []
    step = 0
    while True:
        pos = int(start_pos) + int(dir1) * (2 * int(step))
        if not (0 <= pos < int(limit)):
            break
        if step % 3 == 2:
            pos_list.append(int(pos))
        step += 1
    return pos_list


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    info = _segment_info(g)
    if info is None:
        return {"inside_hits": 0, "bridge_lines": 0, "green_lines": 0}

    orient = info["orient"]
    span0 = info["span0"]
    span1 = info["span1"]
    green_positions = _green_positions(info["start_pos"], info["dir1"], info["limit"])

    inside_hits = 0
    bridge_lines = 0
    for pos in green_positions:
        if orient == "h":
            line = g[int(pos), :]
        else:
            line = g[:, int(pos)]
        inside = line[int(span0) : int(span1) + 1] != bg
        outside_left = line[: int(span0)] != bg
        outside_right = line[int(span1) + 1 :] != bg
        inside_hits += int(inside.sum())
        if bool(inside.any()) and (bool(outside_left.any()) or bool(outside_right.any())):
            bridge_lines += 1

    return {
        "inside_hits": int(inside_hits),
        "bridge_lines": int(bridge_lines),
        "green_lines": int(len(green_positions)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate Y/Y/G stripes from the yellow edge segment toward the opposite edge.

    Yellow stripes still overwrite the full span.
    Green stripes now act on complementary regions:
    - outside the span: recolor existing non-background pixels to green;
    - inside the span: only background cells are filled green, existing objects survive.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    out = g.copy()
    H, W = g.shape

    info = _segment_info(g)
    if info is None:
        return out

    orient = info["orient"]
    start_pos = int(info["start_pos"])
    dir1 = int(info["dir1"])
    limit = int(info["limit"])
    span0 = int(info["span0"])
    span1 = int(info["span1"])

    cycle = [Y, Y, G]
    step = 0
    while True:
        pos = int(start_pos) + int(dir1) * (2 * int(step))
        if not (0 <= pos < limit):
            break

        col = int(cycle[int(step) % 3])
        if orient == "h":
            if col == G:
                if span0 > 0:
                    out[pos, :span0] = np.where(out[pos, :span0] != bg, G, out[pos, :span0])
                if span1 + 1 < W:
                    out[pos, span1 + 1 :] = np.where(
                        out[pos, span1 + 1 :] != bg, G, out[pos, span1 + 1 :]
                    )
                mid = out[pos, span0 : span1 + 1]
                out[pos, span0 : span1 + 1] = np.where(mid == bg, G, mid)
            else:
                out[pos, span0 : span1 + 1] = col

            gap = int(start_pos) + int(dir1) * (2 * int(step) + 1)
            if 0 <= gap < H:
                out[int(gap), span0 : span1 + 1] = bg
        else:
            if col == G:
                if span0 > 0:
                    out[:span0, pos] = np.where(out[:span0, pos] != bg, G, out[:span0, pos])
                if span1 + 1 < H:
                    out[span1 + 1 :, pos] = np.where(
                        out[span1 + 1 :, pos] != bg, G, out[span1 + 1 :, pos]
                    )
                mid = out[span0 : span1 + 1, pos]
                out[span0 : span1 + 1, pos] = np.where(mid == bg, G, mid)
            else:
                out[span0 : span1 + 1, pos] = col

            gap = int(start_pos) + int(dir1) * (2 * int(step) + 1)
            if 0 <= gap < W:
                out[span0 : span1 + 1, int(gap)] = bg

        step += 1

    return out


def _make_bridge_sprite(length_along_line: int, thickness: int, fg: int, bg: int, orient: str) -> np.ndarray:
    if orient == "h":
        h, w = thickness, length_along_line
    else:
        h, w = length_along_line, thickness
    sprite = np.full((h, w), bg, dtype=int)
    sprite[:, :] = fg
    if orient == "h":
        sprite[0, 0] = bg
        sprite[-1, -1] = bg
        if w >= 5:
            sprite[0, -2] = bg
    else:
        sprite[0, 0] = bg
        sprite[-1, -1] = bg
        if h >= 5:
            sprite[-2, 0] = bg
    return sprite


def _can_paste(canvas: np.ndarray, sprite: np.ndarray, r0: int, c0: int, bg: int) -> bool:
    H, W = canvas.shape
    h, w = sprite.shape
    if r0 < 1 or c0 < 1 or r0 + h > H - 1 or c0 + w > W - 1:
        return False
    window = canvas[r0 : r0 + h, c0 : c0 + w]
    return not bool(((window != bg) & (sprite != bg)).any())


def _paste(canvas: np.ndarray, sprite: np.ndarray, r0: int, c0: int, bg: int) -> None:
    h, w = sprite.shape
    mask = sprite != bg
    canvas[r0 : r0 + h, c0 : c0 + w][mask] = sprite[mask]


def _place_bridge_blob(
    canvas: np.ndarray,
    rng: random.Random,
    orient: str,
    span0: int,
    span1: int,
    line_pos: int,
    fg: int,
    bg: int,
) -> bool:
    H, W = canvas.shape
    span_len = span1 - span0 + 1
    inside_len = min(span_len, rng.randint(2, 4))
    thickness = 3

    choices: list[tuple[str, int]] = []
    left_room = span0 - 1
    right_room = (W - 2 - span1) if orient == "h" else (H - 2 - span1)
    if left_room >= 2:
        choices.append(("left", min(left_room, rng.randint(2, 4))))
    if right_room >= 2:
        choices.append(("right", min(right_room, rng.randint(2, 4))))
    if not choices:
        return False

    rng.shuffle(choices)
    for side, outside_len in choices:
        total_len = inside_len + outside_len
        sprite = _make_bridge_sprite(total_len, thickness, fg, bg, orient)
        if orient == "h":
            r0 = int(line_pos) - 1
            if side == "left":
                c0 = span0 - outside_len
            else:
                c0 = span1 - inside_len + 1
        else:
            c0 = int(line_pos) - 1
            if side == "left":
                r0 = span0 - outside_len
            else:
                r0 = span1 - inside_len + 1

        if _can_paste(canvas, sprite, r0, c0, bg):
            _paste(canvas, sprite, r0, c0, bg)
            return True
    return False


def _gen_attempt():
    rng = random.Random()

    H = rng.randint(25, 30)
    W = rng.randint(22, 30)

    available = [c for c in range(10) if c not in (G, Y)]
    bg, fg = rng.sample(available, 2)

    edge = rng.choice(["top", "bottom", "left", "right"])
    seg_len = rng.randint(3, 8)

    if edge in ("top", "bottom"):
        max_start = W - 1 - seg_len
        if max_start < 1:
            return None
        seg_start = rng.randint(1, max_start)
        orient = "h"
        edge_row = 0 if edge == "top" else H - 1
        span0, span1 = seg_start, seg_start + seg_len - 1
        start_pos = edge_row
        limit = H
    else:
        max_start = H - 1 - seg_len
        if max_start < 1:
            return None
        seg_start = rng.randint(1, max_start)
        orient = "v"
        edge_col = 0 if edge == "left" else W - 1
        span0, span1 = seg_start, seg_start + seg_len - 1
        start_pos = edge_col
        limit = W

    dir1 = 1 if start_pos == 0 else -1
    green_positions = _green_positions(start_pos, dir1, limit)
    if len(green_positions) < 2:
        return None

    grid = np.full((H, W), bg, dtype=int)
    if orient == "h":
        grid[start_pos, span0 : span1 + 1] = Y
    else:
        grid[span0 : span1 + 1, start_pos] = Y

    canvas = grid.copy()

    bridge_targets = green_positions[:]
    rng.shuffle(bridge_targets)
    bridge_ok = 0
    for line_pos in bridge_targets[:2]:
        if _place_bridge_blob(canvas, rng, orient, span0, span1, int(line_pos), fg, bg):
            bridge_ok += 1
    if bridge_ok == 0:
        return None

    extra_blobs = rng.randint(3, 6)
    for _ in range(extra_blobs):
        bh = rng.randint(2, 7)
        bw = rng.randint(2, 8)
        density = rng.uniform(0.45, 0.85)
        try:
            sprite = random_sprite(
                bh,
                bw,
                density=density,
                symmetry="not_symmetric",
                color_palette=[fg],
                connectivity=4,
                background=bg,
            )
        except (AssertionError, RecursionError):
            continue

        mask = sprite != bg
        rows_any = np.any(mask, axis=1)
        cols_any = np.any(mask, axis=0)
        if not rows_any.any():
            continue
        r0, r1 = np.where(rows_any)[0][[0, -1]]
        c0, c1 = np.where(cols_any)[0][[0, -1]]
        sprite = sprite[r0 : r1 + 1, c0 : c1 + 1]
        pix = int(np.sum(sprite != bg))
        if pix < 5 or pix > 45:
            continue

        try:
            pr, pc = random_free_location(canvas, sprite, transparent=bg, border=1, padding=1)
        except ValueError:
            continue
        _paste(canvas, sprite, pr, pc, bg)

    fg_mask = (canvas != bg) & (canvas != Y)
    _, n_placed = label_mask(fg_mask, connectivity=4)
    if int(n_placed) < 4:
        return None

    metrics = _variant_metrics(canvas)
    if metrics["inside_hits"] < 2 or metrics["bridge_lines"] < 1:
        return None

    if int(get_bg(canvas)) != bg:
        return None

    out = main(canvas)
    if not np.array_equal(main(canvas), out):
        return None

    return canvas


def generate_input():
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 400 attempts")
