# Combo variant of c7f57c3e (index 22)
# combo_spec: D1/0+D2/2+D4/1
# Name: 形状分组 + 洞数配对 + 空心框架
#
# Design analysis
# ---------------
# a) 具体化与行为追踪
#    - 原题把每个对象当成“整只彩色原型”的拷贝：先识别它更像 proto_A 还是 proto_B，再整只替换成另一只。
#    - 本融合版把对象改成空心框架。每个框架都含有 color-1 锚 spine、边框色 frame、内部填充色 fill，
#      并且内部背景孔洞数量可见且稳定。
#    - 现在最小对象不再只给出 2 个原型，而是给出 2 个外轮廓家族 × 2 个洞数子类 = 4 个字典框架：
#      shape A / 1-hole, shape A / 2-hole, shape B / 1-hole, shape B / 2-hole。
#    - 例子：若某个大对象是 shape A 且内部有 2 个洞，原题会把它整只换成“另一个原型”；单独 D1 会只看 shape，
#      让它借用 shape B 的配色；单独 D2 会只看洞数，在同洞数的候选里换对象。
#      融合后真正执行的是：
#      1. 用填洞后的外轮廓判断它属于 shape A 还是 shape B。
#      2. 用原始空心结构数出它有 1 个还是 2 个洞。
#      3. 去对侧 shape 组里找“同洞数”的 palette。
#      4. 只重涂这只对象自身的框架与填充，不改变它的外轮廓、洞布局、bbox、尺度。
#    - 差异点在于：shape 决定跨到哪一组，hole_count 决定在对侧组里拿哪一套 frame/fill 配色，
#      空心框架则让这两个判断都可以从像素结构直接观察出来。
# b) 复杂度校准
#    - 只保留 2 个 shape 家族与 2 档洞数 {1, 2}，避免组合爆炸。
#    - 每个样例固定放入 4 个 scale=1 字典对象各 1 个，再加 1 个放大对象；这样 shape 与 hole_count 都有重复支撑。
#    - palette 结构控制为“每个 shape 组固定一类 frame 色，hole_count 决定 fill 色”，总变量色保持在 6 个，
#      既能看出 hole-sensitive 配对，又不会让颜色线索过载。
# c) 隐式假设审查
#    - 原 main() 假设“最小对象只给出 2 个原型”，这里必须改成恰好 4 个。
#    - 原 main() 用精确彩色图案作为分组身份；融合后这会把同 shape 不同洞数错误拆开，因此要改成
#      “对非背景掩码先填充封闭洞，再用填洞后的 silhouette 作为 shape key”。
#    - 原 main() 把“替换目标 = 另一只完整原型”；融合后若仍替整只对象，会让 D1 退化成换轮廓，
#      与“按形状分组但保留原轮廓”矛盾。因此输出必须是同一 canonical frame 的重涂版本。
#    - 原 main() 的匹配主要依赖锚点对齐；这里对象 bbox 不变，所以锚 spine 主要承担角色识别与结构稳定性，
#      缩放直接由统一 base bbox 推出。
#    - 新 solver 额外假设每个对象恰有两个变量色，且只有 frame 色接触 bbox 边界。generator 会显式保证。
# d) 放置合法性
#    - 因为输入输出都保持相同 bbox 与占用区域，生成器无需为“替换后外形漂移”预留并集 bbox，只需保留原 bbox。
#    - 所有对象都用 full-bbox marker 做留白，padding >= 2，border >= 1，避免接触、重叠和连通域误合并。
#    - 放大仅用整数最近邻，因此洞数、轮廓和角色分区都稳定保真。
# e) 视觉可观察性
#    - 人类能先看出有两类外轮廓，再看出每类里各有单洞/双洞两种空心框架。
#    - 输出中外轮廓与洞布局完全保留，只是 frame/fill 的颜色借给了对侧 shape 组里相同洞数的对象，
#      因而“shape 决定跨组、hole_count 决定取哪套颜色”的链条能从少量样例中肉眼归纳。
#    - 规则不依赖颜色编号大小、生成顺序等不可视因素。
# f) 可视化整洁度
#    - 四个基础模板共用 6x6 bbox，布局紧凑且内部孔洞较大；对象间距固定，避免彼此干扰。
#    - 背景孔洞、锚 spine、frame、fill 各自角色稳定，网格看上去像一组规整的“空心徽章”而不是碎片拼贴。
# g) 变化的可推断性
#    - 颜色变化始终成对发生：frame 与 fill 一起借自对侧组，但 hole_count 会改变借到的 fill 子色。
#    - 若把 hole_count 去掉，shape A 的 1-hole 与 2-hole 输出会错误地涂成同一套颜色；因此洞数不是装饰，
#      而是实际控制变量。
# 9) 新 main() 的额外假设与 generator 保证
#    - 假设最小面积对象完整暴露 4 个字典框架：generator 固定放入 4 个 scale=1 基础对象。
#    - 假设每个 shape 家族恰有两档洞数且外轮廓相同：generator 为每个 family 预定义一对共 silhouette 的 1-hole / 2-hole 模板。
#    - 假设所有模板共享同一 base bbox 尺寸：generator 固定使用 6x6 模板，solver 因而可安全用 bbox 推尺度。
#    - 假设填充封闭洞后的 silhouette 能唯一恢复 family：generator 让两个 family 的外轮廓明显不同，并允许整体做同构变换以增加变化。

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

ROOT = None
for _parent in Path(__file__).resolve().parents:
    if (_parent / "solvers" / "arc_utils.py").exists():
        ROOT = _parent
        break
assert ROOT is not None, "could not locate project root containing solvers/arc_utils.py"
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, paste, random_free_location


HOLE_CONNECTIVITY = 4


def _scale_nn(arr: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return arr.copy()
    return np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)


def _count_enclosed_holes(arr: np.ndarray, bg: int) -> int:
    hole_mask = arr == bg
    lab, n = label_mask(hole_mask, connectivity=HOLE_CONNECTIVITY)
    h, w = arr.shape
    count = 0
    for k in range(1, n + 1):
        pts = np.argwhere(lab == k)
        if np.any(pts[:, 0] == 0) or np.any(pts[:, 0] == h - 1) or np.any(pts[:, 1] == 0) or np.any(pts[:, 1] == w - 1):
            continue
        count += 1
    return int(count)


def _frame_fill_colors(arr: np.ndarray, bg: int) -> tuple[int, int]:
    colors = [int(v) for v in np.unique(arr) if v not in (bg, 1)]
    assert len(colors) == 2, "each frame sprite must have exactly two variable colors"

    border_vals = set(map(int, arr[0, :].tolist() + arr[-1, :].tolist() + arr[:, 0].tolist() + arr[:, -1].tolist()))
    frame_candidates = [c for c in colors if c in border_vals]
    assert len(frame_candidates) == 1, "exactly one variable color must touch the bbox boundary"

    frame = int(frame_candidates[0])
    fill = int(colors[0] if colors[1] == frame else colors[1])
    return frame, fill


def _canonicalize_frame(arr: np.ndarray, bg: int) -> np.ndarray:
    frame, fill = _frame_fill_colors(arr, bg)
    out = np.zeros(arr.shape, dtype=int)
    out[arr == 1] = 1
    out[arr == frame] = 2
    out[arr == fill] = 3
    return out


def _paint_from_canonical(labels: np.ndarray, bg: int, palette: tuple[int, int]) -> np.ndarray:
    out = np.full(labels.shape, bg, dtype=int)
    out[labels == 1] = 1
    out[labels == 2] = int(palette[0])
    out[labels == 3] = int(palette[1])
    return out


def _filled_silhouette(arr: np.ndarray, bg: int) -> np.ndarray:
    filled = (arr != bg).copy()
    hole_mask = arr == bg
    lab, n = label_mask(hole_mask, connectivity=HOLE_CONNECTIVITY)
    h, w = arr.shape
    for k in range(1, n + 1):
        pts = np.argwhere(lab == k)
        if np.any(pts[:, 0] == 0) or np.any(pts[:, 0] == h - 1) or np.any(pts[:, 1] == 0) or np.any(pts[:, 1] == w - 1):
            continue
        filled[lab == k] = True
    return filled.astype(int)


def _shape_key(arr: np.ndarray, bg: int) -> tuple[int, ...]:
    return tuple(_filled_silhouette(arr, bg).ravel().tolist())


def _extract_sprites(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    comps = components_mask(grid != bg, connectivity=8)
    sprites: list[dict[str, object]] = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        sprites.append({"pts": pts, "bbox": (r0, c0, r1, c1), "arr": arr})
    return sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint each hollow-frame sprite with the opposite shape group's palette, matched by hole count.

    Input structure:
    - bg is the most common color. Every non-bg component is a connected hollow-frame sprite with a color-1 anchor spine, one frame color, one fill color, and either one or two enclosed bg holes.
    - The smallest sprites expose four base templates: two outer-silhouette families, each appearing once with 1 hole and once with 2 holes. Larger sprites are scaled copies of those templates.

    Rule:
    - Recover the four base templates from the smallest sprites, then group them by filled outer silhouette rather than by literal colors.
    - For each sprite, infer its shape family from that silhouette and count its enclosed holes from the original hollow structure.
    - Keep the sprite's own canonical frame layout fixed, but repaint its frame/fill roles with the palette taken from the other shape family at the same hole count.

    Key insight: the swap no longer changes the object's outline; shape only tells you which opposite family to consult, while the visible hole count selects which palette inside that family should be borrowed.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    sprites = _extract_sprites(g, bg)
    assert len(sprites) >= 5, "grid must contain 4 base hollow frames plus at least one extra sprite"

    areas = [int((s["bbox"][2] - s["bbox"][0] + 1) * (s["bbox"][3] - s["bbox"][1] + 1)) for s in sprites]
    min_area = min(areas)
    base = [s for s, a in zip(sprites, areas, strict=False) if a == min_area]

    uniq: dict[tuple[int, ...], dict[str, object]] = {}
    for s in base:
        arr = np.asarray(s["arr"], dtype=int)
        canon = _canonicalize_frame(arr, bg)
        key = tuple(canon.ravel().tolist())
        if key not in uniq:
            uniq[key] = {
                "canon": canon,
                "palette": _frame_fill_colors(arr, bg),
                "shape_key": _shape_key(arr, bg),
                "shape_mask": _filled_silhouette(arr, bg),
                "holes": _count_enclosed_holes(arr, bg),
            }
    assert len(uniq) == 4, "smallest-area sprites must yield exactly 4 distinct base hollow frames"

    protos = list(uniq.values())
    base_h, base_w = protos[0]["canon"].shape
    for proto in protos[1:]:
        assert proto["canon"].shape == (base_h, base_w), "all base templates must share one bbox size"

    family_groups: dict[tuple[int, ...], dict[int, dict[str, object]]] = {}
    for proto in protos:
        hole_count = int(proto["holes"])
        group = family_groups.setdefault(proto["shape_key"], {})
        assert hole_count not in group, "each shape family should expose one prototype per hole count"
        group[hole_count] = proto

    assert len(family_groups) == 2, "base prototypes must form exactly 2 outer-silhouette families"
    families = list(family_groups.values())
    for family in families:
        assert set(family.keys()) == {1, 2}, "each shape family must provide both 1-hole and 2-hole variants"

    out = g.copy()

    for s in sprites:
        r0, c0, r1, c1 = s["bbox"]
        arr = np.asarray(s["arr"], dtype=int)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        assert h % base_h == 0 and w % base_w == 0, "sprite bbox must be an integer scale of the base template"
        scale_h = h // base_h
        scale_w = w // base_w
        assert scale_h == scale_w, "scaling must be isotropic"
        scale = int(scale_h)

        filled_arr = _filled_silhouette(arr, bg)
        family_idx = None
        best_score = -1
        for idx, family in enumerate(families):
            ref_shape = next(iter(family.values()))["shape_mask"]
            scaled_shape = _scale_nn(np.asarray(ref_shape, dtype=int), scale)
            if scaled_shape.shape != filled_arr.shape:
                continue
            score = int(np.sum(scaled_shape == filled_arr))
            if np.array_equal(scaled_shape, filled_arr):
                score += int(filled_arr.size * 10)
            if score > best_score:
                best_score = score
                family_idx = idx
        assert family_idx is not None, "could not identify the sprite's shape family"

        holes = _count_enclosed_holes(arr, bg)
        assert holes in (1, 2), "every sprite must have 1 or 2 enclosed holes"

        target_family = families[1 - family_idx]
        assert holes in target_family, "opposite family must contain the same hole-count palette"
        target_palette = tuple(target_family[holes]["palette"])

        canon_arr = _canonicalize_frame(arr, bg)
        replacement = _paint_from_canonical(canon_arr, bg, target_palette)

        for rr, cc in s["pts"]:
            out[rr, cc] = bg
        paste(out, replacement, r0, c0, transparent=bg)

    return out


# --------------- generator (shape-grouped hollow-frame palette swap) ---------------

# --- CFG: canvas ---
GRID_SIZE_MIN = 26
GRID_SIZE_MAX = 36
GRID_MARGIN_FOR_SPRITES = 8

# --- CFG: prototypes ---
BG_CHOICES = [0, 2, 3, 4, 5, 6, 7, 8, 9]
BIG_SCALES = [2, 3]

# --- CFG: placement ---
PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2
PLACEMENT_CONNECTIVITY = 8

# --- CFG: generation retries ---
MAX_RETRIES = 400


# Labels: 0 = bg hole/outside, 1 = anchor spine, 2 = frame role, 3 = fill role.
_FAMILY_A = {
    1: np.array(
        [
            [2, 1, 1, 1, 1, 2],
            [2, 0, 0, 1, 0, 2],
            [2, 0, 3, 3, 0, 2],
            [2, 0, 0, 0, 0, 2],
            [2, 0, 0, 0, 0, 2],
            [2, 2, 2, 2, 2, 2],
        ],
        dtype=int,
    ),
    2: np.array(
        [
            [2, 1, 1, 1, 1, 2],
            [2, 0, 0, 1, 0, 2],
            [2, 0, 3, 1, 0, 2],
            [2, 0, 0, 1, 0, 2],
            [2, 0, 0, 1, 0, 2],
            [2, 2, 2, 2, 2, 2],
        ],
        dtype=int,
    ),
}

_FAMILY_B = {
    1: np.array(
        [
            [0, 2, 1, 1, 2, 0],
            [2, 0, 0, 1, 0, 2],
            [2, 0, 3, 3, 0, 2],
            [2, 0, 0, 0, 0, 2],
            [2, 0, 0, 0, 0, 2],
            [2, 2, 2, 2, 2, 2],
        ],
        dtype=int,
    ),
    2: np.array(
        [
            [0, 2, 1, 1, 2, 0],
            [2, 0, 0, 1, 0, 2],
            [2, 0, 3, 1, 0, 2],
            [2, 0, 0, 1, 0, 2],
            [2, 0, 0, 1, 0, 2],
            [2, 2, 2, 2, 2, 2],
        ],
        dtype=int,
    ),
}


def _transform_labels(arr: np.ndarray, transform: str) -> np.ndarray:
    if transform == "id":
        return arr.copy()
    if transform == "flipud":
        return np.flipud(arr).copy()
    if transform == "fliplr":
        return np.fliplr(arr).copy()
    if transform == "rot90":
        return np.rot90(arr).copy()
    raise ValueError(f"unknown transform: {transform}")


def _make_template_records() -> list[dict[str, object]]:
    transform = random.choice(["id", "flipud", "fliplr", "rot90"])
    records: list[dict[str, object]] = []
    for family_idx, family in enumerate((_FAMILY_A, _FAMILY_B)):
        for hole_count in (1, 2):
            labels = _transform_labels(family[hole_count], transform)
            records.append(
                {
                    "family": family_idx,
                    "holes": hole_count,
                    "labels": labels,
                }
            )
    return records


def _pick_palettes(bg: int) -> dict[tuple[int, int], tuple[int, int]]:
    pool = [c for c in range(10) if c not in (bg, 1)]
    frame_a, frame_b, fill_a1, fill_a2, fill_b1, fill_b2 = random.sample(pool, 6)
    return {
        (0, 1): (frame_a, fill_a1),
        (0, 2): (frame_a, fill_a2),
        (1, 1): (frame_b, fill_b1),
        (1, 2): (frame_b, fill_b2),
    }


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_CHOICES)
    records = _make_template_records()
    palettes = _pick_palettes(bg)

    for rec in records:
        family = int(rec["family"])
        holes = int(rec["holes"])
        labels = np.asarray(rec["labels"], dtype=int)
        rec["input_proto"] = _paint_from_canonical(labels, bg, palettes[(family, holes)])
        rec["output_proto"] = _paint_from_canonical(labels, bg, palettes[(1 - family, holes)])

    instances = [(idx, 1) for idx in range(len(records))]
    instances.append((random.randrange(len(records)), random.choice(BIG_SCALES)))
    random.shuffle(instances)
    instances.sort(key=lambda item: item[1], reverse=True)

    base_h, base_w = np.asarray(records[0]["labels"]).shape
    max_dim = max(max(base_h, base_w) * scale for _idx, scale in instances)
    min_n = max(GRID_SIZE_MIN, max_dim + GRID_MARGIN_FOR_SPRITES)
    if min_n > GRID_SIZE_MAX:
        return None
    n = random.randint(min_n, GRID_SIZE_MAX)

    inp = np.full((n, n), bg, dtype=int)
    out = np.full((n, n), bg, dtype=int)
    reserve = np.full((n, n), bg, dtype=int)

    for proto_idx, scale in instances:
        input_patch = _scale_nn(np.asarray(records[proto_idx]["input_proto"], dtype=int), scale)
        output_patch = _scale_nn(np.asarray(records[proto_idx]["output_proto"], dtype=int), scale)
        marker = np.full(input_patch.shape, 1, dtype=int)
        try:
            top, left = random_free_location(
                reserve,
                marker,
                transparent=bg,
                border=PLACEMENT_BORDER,
                padding=PLACEMENT_PADDING,
                padding_connectivity=PLACEMENT_CONNECTIVITY,
            )
        except ValueError:
            return None

        paste(reserve, marker, top, left, transparent=bg)
        paste(inp, input_patch, top, left, transparent=bg)
        paste(out, output_patch, top, left, transparent=bg)

    try:
        solved = main(inp)
    except Exception:
        return None
    if np.array_equal(solved, out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid c7f57c3e combo input")
