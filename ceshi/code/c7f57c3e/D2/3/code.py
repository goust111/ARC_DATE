# Variant of c7f57c3e
# Dimension: D2 — 配对匹配策略 (match_cond)
# Change: 跨组配对时要求尺度相同（s.scale == d.scale）且位置重叠最大化... → 按对称性匹配

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the D2 variant
# ----------------------------------
# variant_spec:
#   concrete_behavior:
#     - 每个非背景连通块仍然是一个缩放后的多色精灵，且都包含固定的颜色 1 锚点。
#     - 但输入不再只含“两个全局原型反复出现”。现在每个样例包含 2 对精灵；
#       每一对内部的两只精灵在形状上互为镜像（水平 / 垂直 / 主对角线 / 副对角线），
#       同时各自携带不同的两色调色板。
#     - 输出时，每只精灵保留自己的位置、尺度和几何形状，只把自己的两种非锚点颜色
#       换成镜像配对对象的两种颜色；因此真正交换的是“颜色对”，不是整个形状。
#   behavior_trace:
#     - 原逻辑：从最小精灵提取两种原型 -> 识别某精灵属于哪种原型与尺度 -> 直接替换成另一原型。
#     - 新逻辑：先对每个精灵按锚点恢复尺度 -> 压回未缩放标签图 -> 用镜像闭包求规范键；
#       规范键相同的两个精灵视为一对 -> 仅交换这一对的两种非锚点颜色。
#     - 差异：原题的“替换目标”由全局另一原型决定；变体的“替换颜色来源”由局部镜像配对决定。
#   complexity_calibration:
#     - 为了避免规则退化成“全局统一换色”，每对精灵都使用独立的颜色对，且样例内共有 4 个
#       不同的有序颜色对；这样不做配对就无法推出正确输出。
#     - 为了让镜像匹配成为必要条件，两对精灵使用不同的不对称模板，且至少有 1 对跨尺度
#       （一只 scale=1，另一只 scale in {2,3}），从而明确打破旧的“同尺度优先”思路。
#   implicit_assumption_audit:
#     - 原代码依赖“最小面积精灵恰好给出两个原型”以及“替换对象只由原型类别决定”；这些前提
#       在变体里都失效，因此不能再从最小精灵抽原型，也不能按原型二选一替换。
#     - 新 solver 改为依赖以下输入性质：
#       1. 每个精灵恰有 1 个锚点块（颜色 1），其缩放后仍是正方形块，可据此恢复 scale。
#       2. 两种非锚点角色的像素数固定为 2 和 4，缩放后面积仍严格有别，可据此恢复角色顺序。
#       3. 每个镜像家族在同一网格中恰好出现 2 次，可唯一成对。
#     - generator 会显式构造并验证这些性质。
#   placement_legality:
#     - 精灵按 bbox 留白放置，边界至少 1 格，bbox 之间至少间隔 2 格，避免 8 邻域粘连。
#     - 输出只在原 bbox 内重着色，不改变占用区域，因此不会新增越界或重叠风险。
#   visual_observability:
#     - 模板全部是非对称形状，镜像关系肉眼可见；而输出只交换颜色、不换形，能让人看到
#       “左边这只拿到了右边镜像体的颜色”。
#     - 至少一对跨尺度，能避免误读为“同尺寸精灵互换”。
#   cleanliness_and_color_inference:
#     - 每个精灵只有背景、锚点 1 和两种可变颜色，总颜色负担低。
#     - 两个可变区域的角色由几何面积区分，而非颜色 id 排序；这满足 ARC 中“可见特征驱动”
#       的要求，也让颜色交换规则可从少量样例推断。


# --------------- shared helpers ---------------

GRID_SIZE_MIN = 18
GRID_SIZE_MAX = 28
GRID_MARGIN_FOR_SPRITES = 6

BG_CHOICES = [0, 2, 3, 4, 5, 6, 7, 8, 9]
PAIR_COUNT = 2
BIG_SCALES = [2, 3]

PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2
PLACEMENT_CONNECTIVITY = 8

MAX_RETRIES = 400

MIRROR_KINDS = ("h", "v", "d", "ad")


def _anti_diag(arr: np.ndarray) -> np.ndarray:
    return np.fliplr(np.flipud(arr)).T


def _mirror(arr: np.ndarray, kind: str) -> np.ndarray:
    if kind == "id":
        return arr.copy()
    if kind == "h":
        return np.flipud(arr)
    if kind == "v":
        return np.fliplr(arr)
    if kind == "d":
        return arr.T
    if kind == "ad":
        return _anti_diag(arr)
    raise ValueError(f"unknown mirror kind: {kind}")


def _all_symmetries(arr: np.ndarray) -> list[np.ndarray]:
    return [
        arr.copy(),
        np.rot90(arr, 1),
        np.rot90(arr, 2),
        np.rot90(arr, 3),
        np.flipud(arr),
        np.fliplr(arr),
        arr.T,
        _anti_diag(arr),
    ]


def _scale_nn(arr: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return arr.copy()
    return np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)


def _tight_crop(arr: np.ndarray) -> np.ndarray:
    ys, xs = np.where(arr != 0)
    assert len(ys) > 0, "template must contain foreground cells"
    return arr[int(ys.min()) : int(ys.max()) + 1, int(xs.min()) : int(xs.max()) + 1].copy()


def _pick_anchor_bbox(arr: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(arr == 1)
    assert len(ys) > 0, "sprite must contain anchor color 1"
    return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())


def _infer_scale(arr: np.ndarray) -> int:
    r0, c0, r1, c1 = _pick_anchor_bbox(arr)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert h == w, "scaled anchor must remain a square block"
    return int(h)


def _compress_scaled(arr: np.ndarray, scale: int) -> np.ndarray:
    h, w = arr.shape
    assert h % scale == 0 and w % scale == 0, "bbox must be divisible by scale"
    out = np.zeros((h // scale, w // scale), dtype=int)
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            block = arr[r * scale : (r + 1) * scale, c * scale : (c + 1) * scale]
            vals = np.unique(block)
            assert len(vals) == 1, "sprite is not a nearest-neighbor scaled template"
            out[r, c] = int(vals[0])
    return out


def _normalize_roles(arr: np.ndarray, bg: int) -> tuple[np.ndarray, tuple[int, int]]:
    colors = [int(c) for c in np.unique(arr) if c not in (bg, 1)]
    assert len(colors) == 2, "each sprite must contain exactly two non-anchor colors"
    counts = sorted((int(np.sum(arr == c)), c) for c in colors)
    assert counts[0][0] != counts[1][0], "role areas must be distinct"
    role2_color = counts[0][1]
    role3_color = counts[1][1]

    norm = np.zeros(arr.shape, dtype=int)
    norm[arr == 1] = 1
    norm[arr == role2_color] = 2
    norm[arr == role3_color] = 3
    return norm, (role2_color, role3_color)


def _family_key(norm: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    candidates = []
    for variant in _all_symmetries(norm):
        candidates.append((tuple(map(int, variant.shape)), tuple(int(x) for x in variant.ravel())))
    return min(candidates)


def _extract_sprite_record(grid: np.ndarray, pts: np.ndarray, bg: int) -> dict:
    r0, c0, r1, c1 = bbox_coords(pts)
    arr = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    scale = _infer_scale(arr)
    base = _compress_scaled(arr, scale)
    norm, role_colors = _normalize_roles(base, bg)
    return {
        "pts": pts,
        "bbox": (r0, c0, r1, c1),
        "arr": arr,
        "scale": scale,
        "base": base,
        "norm": norm,
        "role_colors": role_colors,
        "key": _family_key(norm),
    }


def _recolor_sprite(arr: np.ndarray, bg: int, src_colors: tuple[int, int], dst_colors: tuple[int, int]) -> np.ndarray:
    out = arr.copy()
    out[arr == src_colors[0]] = dst_colors[0]
    out[arr == src_colors[1]] = dst_colors[1]
    out[arr == 1] = 1
    out[arr == bg] = bg
    return out


def _detect_pair_transform(a: np.ndarray, b: np.ndarray) -> str:
    for kind in MIRROR_KINDS:
        if np.array_equal(_mirror(a, kind), b):
            return kind
    if np.array_equal(a, b):
        return "id"
    raise AssertionError("paired sprites are not related by a supported mirror")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap non-anchor colors between mirror-matched sprites while keeping every sprite fixed in place.

    Input structure:
    - bg is the most frequent color. Each foreground component is one scaled multicolor sprite with a color-1 anchor block and exactly two other colors.
    - The grid contains two mirror families; within each family, the two sprites have shapes related by one horizontal, vertical, or diagonal reflection, possibly at different scales.

    Rule:
    - Recover each sprite's integer scale from its anchor block, compress it back to an unscaled label pattern, and normalize the two non-anchor roles by their visible region sizes.
    - Group sprites by mirror-family equivalence, then for each matched pair exchange only the two non-anchor colors; each sprite keeps its own geometry, scale, and position.

    Key insight:
    - Mirror matching must happen on a scale-free, color-normalized representation. Once the reflected mate is found, the output is just a recoloring of the existing pixels, not a shape replacement.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    comps = components_mask(g != bg, connectivity=8)
    assert len(comps) >= 2, "grid must contain at least two sprites"

    sprites = [_extract_sprite_record(g, pts, bg) for pts in comps]

    groups: dict[tuple[tuple[int, int], tuple[int, ...]], list[dict]] = {}
    for sprite in sprites:
        groups.setdefault(sprite["key"], []).append(sprite)

    assert len(groups) == PAIR_COUNT, "the grid must contain exactly two mirror families"

    out = g.copy()
    for family in groups.values():
        assert len(family) == 2, "each mirror family must contain exactly two sprites"
        left, right = family
        _detect_pair_transform(left["norm"], right["norm"])

        for sprite, partner in ((left, right), (right, left)):
            for rr, cc in sprite["pts"]:
                out[rr, cc] = bg
            recolored = _recolor_sprite(sprite["arr"], bg, sprite["role_colors"], partner["role_colors"])
            r0, c0, _r1, _c1 = sprite["bbox"]
            paste(out, recolored, r0, c0, transparent=bg)

    return out


# --------------- generator ---------------

PROTO_TEMPLATES = (
    np.array(
        [
            [0, 1, 0, 0],
            [2, 2, 3, 0],
            [0, 3, 3, 0],
            [0, 0, 3, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 1, 0, 0, 0],
            [0, 2, 0, 3, 0],
            [0, 2, 3, 3, 0],
            [0, 0, 0, 3, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 0, 1, 0],
            [0, 2, 2, 0],
            [3, 0, 0, 0],
            [0, 3, 3, 0],
            [0, 0, 3, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 1, 0, 0, 0],
            [0, 2, 2, 0, 0],
            [0, 0, 3, 3, 0],
            [0, 0, 0, 3, 3],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 2, 2, 0, 0],
            [0, 0, 3, 0, 0],
            [0, 0, 3, 3, 0],
            [0, 0, 0, 3, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 1, 0, 0],
            [0, 2, 0, 0],
            [0, 2, 3, 0],
            [0, 0, 3, 3],
            [0, 0, 0, 3],
        ],
        dtype=int,
    ),
)


def _colorize(labels: np.ndarray, palette: tuple[int, int], bg: int) -> np.ndarray:
    out = np.full(labels.shape, bg, dtype=int)
    out[labels == 1] = 1
    out[labels == 2] = palette[0]
    out[labels == 3] = palette[1]
    return out


def _pair_specs(bg: int) -> list[dict]:
    templates = random.sample(list(PROTO_TEMPLATES), PAIR_COUNT)
    mirror_kinds = random.sample(list(MIRROR_KINDS), PAIR_COUNT)

    pool = [c for c in range(10) if c not in (bg, 1)]
    palettes = random.sample([(a, b) for a in pool for b in pool if a != b], 2 * PAIR_COUNT)

    big_pair_idx = random.randrange(PAIR_COUNT)
    big_scale = random.choice(BIG_SCALES)

    specs = []
    palette_idx = 0
    for idx in range(PAIR_COUNT):
        template = _tight_crop(templates[idx])
        mirror_kind = mirror_kinds[idx]
        pal_a = palettes[palette_idx]
        pal_b = palettes[palette_idx + 1]
        palette_idx += 2

        scales = [1, 1]
        if idx == big_pair_idx:
            scales[random.randrange(2)] = big_scale

        left_labels = template
        right_labels = _tight_crop(_mirror(template, mirror_kind))
        if random.random() < 0.5:
            left_labels, right_labels = right_labels, left_labels

        specs.append(
            {
                "mirror_kind": mirror_kind,
                "labels": (left_labels, right_labels),
                "palettes": (pal_a, pal_b),
                "scales": tuple(scales),
            }
        )
    return specs


def _build_patches(bg: int) -> tuple[list[dict], list[dict]]:
    input_sprites = []
    output_sprites = []

    for spec in _pair_specs(bg):
        labels_a, labels_b = spec["labels"]
        pal_a, pal_b = spec["palettes"]
        scale_a, scale_b = spec["scales"]

        in_a = _scale_nn(_colorize(labels_a, pal_a, bg), scale_a)
        in_b = _scale_nn(_colorize(labels_b, pal_b, bg), scale_b)
        out_a = _scale_nn(_colorize(labels_a, pal_b, bg), scale_a)
        out_b = _scale_nn(_colorize(labels_b, pal_a, bg), scale_b)

        input_sprites.extend(
            [
                {"patch": in_a, "pair": len(input_sprites) // 2, "scale": scale_a},
                {"patch": in_b, "pair": len(input_sprites) // 2, "scale": scale_b},
            ]
        )
        output_sprites.extend(
            [
                {"patch": out_a, "pair": len(output_sprites) // 2, "scale": scale_a},
                {"patch": out_b, "pair": len(output_sprites) // 2, "scale": scale_b},
            ]
        )

    return input_sprites, output_sprites


def _place_sprites(inp_sprites: list[dict], out_sprites: list[dict], bg: int) -> tuple[np.ndarray, np.ndarray] | None:
    max_dim = max(max(sprite["patch"].shape) for sprite in inp_sprites)
    n = random.randint(max(GRID_SIZE_MIN, max_dim + GRID_MARGIN_FOR_SPRITES), GRID_SIZE_MAX)

    inp = np.full((n, n), bg, dtype=int)
    out = np.full((n, n), bg, dtype=int)
    reserve = np.full((n, n), bg, dtype=int)

    order = sorted(range(len(inp_sprites)), key=lambda idx: max(inp_sprites[idx]["patch"].shape), reverse=True)
    for idx in order:
        in_patch = inp_sprites[idx]["patch"]
        out_patch = out_sprites[idx]["patch"]
        marker = np.full(in_patch.shape, 1, dtype=int)
        try:
            r0, c0 = random_free_location(
                reserve,
                marker,
                transparent=bg,
                border=PLACEMENT_BORDER,
                padding=PLACEMENT_PADDING,
                padding_connectivity=PLACEMENT_CONNECTIVITY,
            )
        except ValueError:
            return None

        paste(reserve, marker, r0, c0)
        paste(inp, in_patch, r0, c0, transparent=bg)
        paste(out, out_patch, r0, c0, transparent=bg)

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_CHOICES)
    inp_sprites, out_sprites = _build_patches(bg)
    placed = _place_sprites(inp_sprites, out_sprites, bg)
    if placed is None:
        return None
    inp, expected = placed

    try:
        solved = main(inp)
    except Exception:
        return None
    if np.array_equal(solved, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid c7f57c3e D2 input")
