# Combo variant of 269e22fb (index 26)
# combo_spec: D5/0+D6/0+D15/1
# Name: 对称图案旋转群50%密度

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


for parent in Path(__file__).resolve().parents:
    candidate = parent / "solvers"
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))
        break

from arc_utils import Color, colors_present


# Design analysis for the combo variant
# -------------------------------------
# a) 具体化与行为追踪
# - 原题是“固定 20x20 位图 + 8 个二面体变体 + 两色矩形裁剪”。solver 只要枚举前景极性、
#   朝向和偏移，找到唯一窗口即可。
# - 这里把固定位图改成“由隐藏 5x5 seed 程序化生成”的家族，但不直接复用 D5 的四份同值复制，
#   因为那会让总前景数永远是 16 的倍数，无法得到恰好 200/400 的 50% 密度。
# - 融合后的 10x10 基元改为四块 5x5 拷贝：
#     [[T0(seed),      T1(seed)],
#      [~T4(seed),    ~T5(seed)]]
#   其中 T0/T1/T4/T5 与 D5 相同，只是下排两块整体取反。再把该 10x10 基元按
#     [[Q, rot270(Q)],
#      [rot90(Q), rot180(Q)]]
#   拼成 20x20。这样整张图保持 C4 旋转对称，同时每个 seed 标签恰好以“2 次原值 + 2 次反值”
#   出现在 10x10 基元里，所以无论 seed 如何，整张 20x20 永远正好 200 个前景像素。
# - 小例子：若某个裁剪窗来自 seed=S 的输出，solver 把某颜色当作前景时，会把窗口内每个像素
#   通过 (可见位 xor 该位置是否取反) 还原回对应 seed 标签。若颜色极性选对，就恢复出 S；
#   若颜色极性反过来，就恢复出 ~S。两者都会生成合法二值图案，但用各自对应的 fg/bg 重绘后，
#   20x20 彩色结果完全相同，因此 solver 需要从“唯一原始匹配”改成“唯一彩色输出”。
# - 与原题差异：
#   1. 不再匹配一张硬编码位图，而是从重复标签约束中重建隐藏 seed。
#   2. 允许的群从 D4 缩到“仅旋转”，而由于整图本身 C4 对称，四个旋转进一步塌缩成同一类。
#   3. 50% 密度不再只是统计性质，而是与“补色 seed 也合法”一起制造稳定的双极性原始匹配。
# b) 复杂度校准
# - 自由度只放在 5x5 seed 上，共 25 个隐藏位；完整输出仍是单个双色大图，视觉上干净。
# - 继续使用 D5 风格的重复标签布局，使任意 H>=10 且 W>=10 的合法裁剪都能覆盖全部 25 个标签，
#   避免“需要全图才可解”的过强自由度。
# - 生成器额外过滤掉 seed 的自旋转/自镜像/补对称退化，防止错误偏移也偶然变成合法答案。
# c) 隐式假设审查（针对原 main）
# - 原来的 `base = _pattern_mask()` 假设存在一张固定模板；现在失效，必须改成标签图 + 取反图。
# - 原来的 `_dihedral_variants(base)` 假设要枚举 D4；这里 D6/0 只保留旋转，而整图又是 C4，
#   所以不再有多朝向循环，solver 只枚举偏移和两种极性。
# - 原来的 `len(matches) == 1` 依赖前景极性唯一；D15/1 叠加后稳定出现两条原始匹配
#   `(seed, fg=a)` 与 `(~seed, fg=b)`，因此必须改成“去重后的彩色 20x20 唯一”。
# - 原来的匹配逻辑默认裁剪直接等于模板子窗；现在每个位置还依赖该位置对应的是 seed 位
#   还是其补位，所以需要显式追踪 `INVERT_MAP`。
# d) 放置合法性
# - 仍是 output-first：先采样 seed、颜色和裁剪尺寸，再从合法 20x20 输出中裁剪输入。
# - 没有多对象碰撞问题；合法性只取决于裁剪不越界，以及该裁剪经 solver 推回的彩色输出唯一。
# e) 视觉可观察性
# - 多个样例会显示：完整输出不再是同一张固定图，而是一族明显具有 90 度旋转对称的双色纹样。
# - 与原题相比，输出会明显更“均衡”：前景/背景各占一半；裁剪窗来自同一类旋转对称纹样家族。
# - 由于不再使用翻转类，样例中不会出现必须区分镜像才能完成任务的情形；这与原题形成可见差异。
# f) 可视化整洁度
# - 画面中仍只有一个矩形裁剪窗和一个双色整图，没有额外标记层；规律全由几何重复和补色拼接承载。
# - 取反只发生在内部结构，不引入第三种颜色，也不会造成对象重叠难以辨认。
# g) 变化的可推断性
# - 颜色规则仍然简单：输出只复用输入中的两种颜色。
# - 人类可从输出直接看到“完整图是旋转对称的大纹样”，并从多样例对比看出不同题目实例的纹样不同；
#   颜色本身没有隐藏排序或数值依赖。
# i) 新 solver 假设与 generator 对齐
# - `main()` 假设输入恰好两色：generator 只采样两种 ARC 颜色。
# - `main()` 假设裁剪覆盖全部 25 个标签：generator 把尺寸限制在 H in [10,16], W in [10,12]。
# - `main()` 假设合法输入会产生恰好 2 个原始匹配但 1 个彩色输出：generator 用同一匹配函数自检。


GRID_SIZE = 20
SEED_SIZE = 5

# Reuse the D5 geometric layout, but invert the lower row of copies so the global
# family is exactly 50% dense for every seed.
COPY_SPECS: tuple[tuple[int, bool], ...] = (
    (0, False),
    (1, False),
    (4, True),
    (5, True),
)

SEED_LABELS = np.arange(SEED_SIZE * SEED_SIZE, dtype=int).reshape(SEED_SIZE, SEED_SIZE)


def _apply_transform(arr: np.ndarray, variant: int) -> np.ndarray:
    if variant < 4:
        return np.rot90(arr, variant)
    return np.fliplr(np.rot90(arr, variant - 4))


def _build_quadrant_maps() -> tuple[np.ndarray, np.ndarray]:
    label_rows: list[np.ndarray] = []
    invert_rows: list[np.ndarray] = []
    for row in ((0, 1), (2, 3)):
        label_blocks: list[np.ndarray] = []
        invert_blocks: list[np.ndarray] = []
        for idx in row:
            variant, invert = COPY_SPECS[idx]
            label_blocks.append(_apply_transform(SEED_LABELS, variant))
            invert_blocks.append(np.full((SEED_SIZE, SEED_SIZE), invert, dtype=bool))
        label_rows.append(np.hstack(label_blocks))
        invert_rows.append(np.hstack(invert_blocks))
    return np.vstack(label_rows), np.vstack(invert_rows)


def _build_full_from_quadrant(quadrant: np.ndarray) -> np.ndarray:
    return np.block(
        [
            [quadrant, np.rot90(quadrant, 3)],
            [np.rot90(quadrant, 1), np.rot90(quadrant, 2)],
        ]
    )


def _build_label_and_invert_maps() -> tuple[np.ndarray, np.ndarray]:
    quadrant_labels, quadrant_inverts = _build_quadrant_maps()
    return _build_full_from_quadrant(quadrant_labels), _build_full_from_quadrant(quadrant_inverts)


LABEL_MAP, INVERT_MAP = _build_label_and_invert_maps()
REFLECTED_LABEL_MAP = np.fliplr(LABEL_MAP)
REFLECTED_INVERT_MAP = np.fliplr(INVERT_MAP)

assert LABEL_MAP.shape == (GRID_SIZE, GRID_SIZE)
assert INVERT_MAP.shape == (GRID_SIZE, GRID_SIZE)
assert np.array_equal(LABEL_MAP, np.rot90(LABEL_MAP, 1))
assert np.array_equal(INVERT_MAP, np.rot90(INVERT_MAP, 1))

for _H in range(10, 17):
    for _W in range(10, 13):
        for _r in range(GRID_SIZE - _H + 1):
            for _c in range(GRID_SIZE - _W + 1):
                assert len(set(map(int, LABEL_MAP[_r : _r + _H, _c : _c + _W].reshape(-1)))) == 25


def _other_color(colors: list[int], fg: int) -> int:
    return int(colors[0] if int(colors[1]) == int(fg) else colors[1])


def _render(pattern: np.ndarray, fg: int, bg: int) -> np.ndarray:
    return np.where(pattern, int(fg), int(bg)).astype(int)


def _dedupe_grids(grids: list[np.ndarray]) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    seen: set[bytes] = set()
    for grid in grids:
        key = np.ascontiguousarray(grid).tobytes()
        if key not in seen:
            seen.add(key)
            unique.append(grid)
    return unique


def _pattern_from_seed(seed: np.ndarray) -> np.ndarray:
    pattern = seed.reshape(-1)[LABEL_MAP] ^ INVERT_MAP
    assert pattern.shape == (GRID_SIZE, GRID_SIZE)
    assert int(pattern.sum()) == 200
    assert np.array_equal(pattern, np.rot90(pattern, 1))
    return pattern


def _dihedral_transforms(arr: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    for k in range(4):
        rotated = np.rot90(arr, k)
        variants.append(rotated)
        variants.append(np.fliplr(rotated))
    return variants


def _seed_is_valid(seed: np.ndarray) -> bool:
    variants = _dihedral_transforms(seed)
    for candidate in variants[1:]:
        if np.array_equal(seed, candidate):
            return False
        if np.array_equal(seed, ~candidate):
            return False

    pattern = _pattern_from_seed(seed)
    for k in range(4):
        if np.array_equal(pattern, np.fliplr(np.rot90(pattern, k))):
            return False
    return True


def _reconstruct_seed(label_window: np.ndarray, invert_window: np.ndarray, visible: np.ndarray) -> np.ndarray | None:
    assignments = np.full(SEED_SIZE * SEED_SIZE, -1, dtype=np.int8)
    for label, invert, bit in zip(label_window.reshape(-1), invert_window.reshape(-1), visible.reshape(-1), strict=True):
        idx = int(label)
        seed_bit = int(bool(bit) ^ bool(invert))
        prev = int(assignments[idx])
        if prev != -1 and prev != seed_bit:
            return None
        assignments[idx] = seed_bit

    if np.any(assignments < 0):
        return None

    seed = assignments.reshape(SEED_SIZE, SEED_SIZE).astype(bool)
    return seed if _seed_is_valid(seed) else None


def _raw_seed_matches_for_maps(
    visible: np.ndarray,
    label_map: np.ndarray,
    invert_map: np.ndarray,
) -> tuple[list[np.ndarray], int]:
    H, W = map(int, visible.shape)
    seeds: list[np.ndarray] = []
    for r in range(GRID_SIZE - H + 1):
        for c in range(GRID_SIZE - W + 1):
            seed = _reconstruct_seed(
                label_map[r : r + H, c : c + W],
                invert_map[r : r + H, c : c + W],
                visible,
            )
            if seed is not None:
                seeds.append(seed)
    return seeds, len(seeds)


def _candidate_outputs(grid: np.ndarray) -> tuple[list[np.ndarray], int]:
    g = np.asarray(grid, dtype=int)
    colors = colors_present(g)

    rendered: list[np.ndarray] = []
    raw_match_count = 0
    for fg in colors:
        bg = _other_color(colors, int(fg))
        visible = g == int(fg)
        seeds, match_count = _raw_seed_matches_for_maps(visible, LABEL_MAP, INVERT_MAP)
        raw_match_count += match_count
        for seed in seeds:
            rendered.append(_render(_pattern_from_seed(seed), int(fg), int(bg)))
    return _dedupe_grids(rendered), raw_match_count


def _reflected_raw_match_count(grid: np.ndarray) -> int:
    g = np.asarray(grid, dtype=int)
    colors = colors_present(g)
    match_count = 0
    for fg in colors:
        visible = g == int(fg)
        _, count = _raw_seed_matches_for_maps(visible, REFLECTED_LABEL_MAP, REFLECTED_INVERT_MAP)
        match_count += count
    return match_count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the full 20x20 balanced rotational motif from a two-color crop of one legal sample.

    Input structure:
    - The grid is a rectangular crop from a 20x20 binary motif generated from a hidden 5x5 seed.
    - The full motif has quarter-turn symmetry and is drawn with exactly two colors whose roles are not directly known.

    Rule:
    - Try each observed color as the logical foreground and slide the crop across every possible offset of the shared rotational label map.
    - At each offset, undo the local direct-vs-inverted copy pattern to recover the 25 hidden seed bits.
    - Regenerate the full 20x20 motif from any consistent seed, repaint it with the tested foreground/background colors, and return the unique colored reconstruction after deduplication.

    Key insight:
    - Because the family is exactly 50% dense and closed under seed complementation, the wrong foreground choice can still produce a valid raw seed match; the reliable invariant is the final colored 20x20 image, not the raw polarity match count.
    """
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    assert H <= GRID_SIZE and W <= GRID_SIZE

    colors = colors_present(g)
    assert len(colors) == 2

    outputs, _ = _candidate_outputs(g)
    assert len(outputs) == 1
    return outputs[0]


def _random_seed(rng: random.Random) -> np.ndarray:
    while True:
        seed = np.array([[rng.random() < 0.5 for _ in range(SEED_SIZE)] for _ in range(SEED_SIZE)], dtype=bool)
        if _seed_is_valid(seed):
            return seed


def _random_colors(rng: random.Random) -> tuple[int, int]:
    fg, bg = rng.sample(Color.ALL_COLORS, 2)
    return int(fg), int(bg)


def _random_example(rng: random.Random) -> tuple[np.ndarray, np.ndarray, int]:
    for _ in range(200):
        seed = _random_seed(rng)
        pattern = _pattern_from_seed(seed)
        fg, bg = _random_colors(rng)
        output = _render(pattern, fg, bg)

        for _ in range(50):
            H = rng.randint(10, 16)
            W = rng.randint(10, 12)
            r0 = rng.randint(0, GRID_SIZE - H)
            c0 = rng.randint(0, GRID_SIZE - W)
            input_grid = output[r0 : r0 + H, c0 : c0 + W].copy()

            outputs, raw_match_count = _candidate_outputs(input_grid)
            if raw_match_count != 2:
                continue
            if _reflected_raw_match_count(input_grid) != 0:
                continue
            if len(outputs) != 1:
                continue
            if not np.array_equal(outputs[0], output):
                continue
            return input_grid, output, raw_match_count

    raise RuntimeError("failed to sample a crop with unique rotation-only reconstruction")


def generate_input() -> np.ndarray:
    """
    Output-first generator for the combo variant.

    Every legal crop covers all 25 seed labels, and every accepted sample exhibits
    the intended fusion behavior: two raw polarity matches collapsing to one output,
    while the reflected family is explicitly rejected so D6/0 stays observable.
    """
    rng = random.Random()
    for _ in range(200):
        input_grid, output, _ = _random_example(rng)
        if np.array_equal(main(input_grid), output):
            return input_grid
    raise RuntimeError("failed to generate a valid combo sample after 200 attempts")
