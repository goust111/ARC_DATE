# Variant of 409aa875
# Dimension: D16 — 投射物起始偏移 (path)
# Change: 投射物从顶点出发，落在顶点+5*方向处（含从顶点本身计数）... → 从精灵非顶点像素（臂端）发射

import random
from collections import Counter, defaultdict

import numpy as np

from arc_utils import components_mask, get_bg as _bg_color, get_bg, neighbors4, neighbors8

# Design analysis for the D16 variant
# a) 具体化与行为追踪
#    - 输入仍然是 16x16、背景色 7、单色 3 像素三角精灵；几何分类不变，仍然只有 L 形和 V 形。
#    - 变化只发生在“从哪里发射”。原题每个精灵只有 1 枚投射物：先从形状推出 `(apex, dir)`，再算
#      `apex + 5*dir`。现在方向 `dir` 保持不变，但发射点改成两个非顶点像素，也就是两条臂端；
#      同一个精灵会得到两个落点：`arm1 + 5*dir` 与 `arm2 + 5*dir`。
#    - 小例子：
#      原 L 形 `{(5,5),(6,5),(5,6)}`，corner/apex=(5,5)，dir=(-1,-1)。
#      原逻辑落点只有 `(0,0)`；新逻辑改成从 `(6,5)` 和 `(5,6)` 发射，落点变成 `(1,0)` 与 `(0,1)`。
#      因而原题是一枚单点标记，变体是两个错开的标记，且更容易和别的精灵/投射物发生碰撞。
# b) 复杂度校准
#    - 方向判定、命中着色、蓝色碰撞格都不改，只把“单发”升级成“双发”。这样视觉上差异明显，
#      但规则仍然只依赖少量稳定类别：L/V 两种形状、8 个方向、两枚投射物。
#    - 若继续沿用旧 generator 的“每个精灵只绑定一个 target”，会把新自由度压扁成旧题近似物。
#      因此 generator 改为按 placement 自带的双落点筛样。
# c) 隐式假设审查
#    - 旧 `main()` 的 `dest_to_shooters` 循环默认“每个 component 只贡献一个 dest”；D16 下已失效，
#      必须改成每个 component 贡献两个 dest。
#    - 旧命中统计 `if dest in cell_to_comp` 只看落点位置，这个前提在变体中仍成立；命中精灵时仍然
#      整体染成 9，不需要区分命中了哪个像素。
#    - 旧代码对落在背景的颜色处理依赖“同一背景格的计数”。这个计数逻辑仍可直接复用，只是计数源
#      从 N 枚投射物变成 2N 枚。
# d) 放置合法性
#    - 生成器预计算所有 `(apex, dir)` placement，要求精灵本体与两个 arm-based 落点都在界内。
#    - 精灵之间仍维持 Chebyshev 距离至少 2，防止 8 连通粘连，保证 solver 的 component 分割稳定。
#    - 若特意种一个“背景碰撞”事件，就把该背景格加入 `reserved_bg`，后续禁止任何精灵占据该格，
#      以免碰撞事件在最终输入中退化成“命中精灵”。
# e) 视觉可观察性
#    - 这个变体最重要的可见证据是：单个精灵会在输出里造成两个近邻结果，而不是一个 apex 对齐的点。
#    - 为防止样本退化成旧题，我在 generator 中显式拒绝 `new_output == legacy_output` 的输入。
#      这样训练样例上一定能看见“双发射”的后果。
# f) 可视化整洁度
#    - 仍限制精灵数量在原范围内，并拒绝过多背景标记的样本，避免双发射把画面塞满。
#    - 输出只有三种视觉元素：原色精灵、栗色命中/单落点、蓝色多重碰撞，层次和原题一致。
# g) 颜色变化可推断性
#    - 颜色语义保持原题：命中精灵或单枚落空背景 -> 9；多枚同落背景 -> 1。
#      变体不引入新的颜色映射，只改变“哪些格被击中”，因此人类仍可从少量样例归纳规则。

# ---- canvas / palette ----
H = W = 16
BG = 7
HIT_COLOR = 9
COLLISION_COLOR = 1
SPRITE_COLORS = [0, 1, 2, 3, 4, 5, 6, 8, 9]

# ---- geometry ----
PROJECTILE_DIST = 5
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
AXIS_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_DIRS = DIAG_DIRS + AXIS_DIRS

# ---- structural counts per grid ----
SPRITE_COUNT_RANGE = (4, 9)
MAX_BG_MARKERS = 10
MAX_ATTEMPTS = 500


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _triangle_info(
    comp: list[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int], tuple[tuple[int, int], tuple[int, int]]] | None:
    """Return (apex, dir, arm_pixels) for a valid 3-pixel triangle sprite."""
    if len(comp) != 3:
        return None
    s = set(comp)

    deg4 = {}
    deg8 = {}
    for r, c in comp:
        deg4[(r, c)] = sum(1 for n in neighbors4(r, c) if n in s)
        deg8[(r, c)] = sum(1 for n in neighbors8(r, c) if n in s)

    corners = [p for p, d in deg4.items() if d == 2]
    if len(corners) == 1:
        apex = corners[0]
        ar, ac = apex
        arms: list[tuple[int, int]] = []
        for r, c in comp:
            if (r, c) == apex:
                continue
            if abs(r - ar) + abs(c - ac) != 1:
                return None
            arms.append((r, c))
        if len(arms) != 2:
            return None
        dr_sum = sum(r - ar for r, _ in arms)
        dc_sum = sum(c - ac for _, c in arms)
        dr = 0 if dr_sum == 0 else (-1 if dr_sum > 0 else 1)
        dc = 0 if dc_sum == 0 else (-1 if dc_sum > 0 else 1)
        return apex, (dr, dc), tuple(sorted(arms))

    apexes = [p for p, d in deg8.items() if d == 2]
    if len(apexes) != 1:
        return None
    apex = apexes[0]
    ar, ac = apex
    arms = [p for p in comp if p != apex]
    if len(arms) != 2:
        return None
    deltas = [(r - ar, c - ac) for r, c in arms]
    if not all(abs(dr) == 1 and abs(dc) == 1 for dr, dc in deltas):
        return None
    dr = -((deltas[0][0] + deltas[1][0]) // 2)
    dc = -((deltas[0][1] + deltas[1][1]) // 2)
    if abs(dr) + abs(dc) != 1:
        return None
    return apex, (dr, dc), tuple(sorted(arms))


def _arm_destinations(comp: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    info = _triangle_info(comp)
    if info is None:
        return None
    _apex, (dr, dc), arms = info
    return [(r + PROJECTILE_DIST * dr, c + PROJECTILE_DIST * dc) for r, c in arms]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Resolve dual projectiles fired by 3-pixel triangle sprites and mark their impacts.

    Input structure:
    - The grid uses one dominant background color; every foreground component is a monochrome 3-pixel triangle, either an L-shape or a diagonal V-shape.

    Rule:
    - Each triangle still defines a single firing direction from its geometry, but it no longer shoots from the apex.
    - The two non-apex arm pixels each fire one projectile, and each projectile lands at `arm + 5 * direction`.
    - Any projectile that lands on a sprite recolors that whole sprite to maroon (9).
    - Empty background landing cells become blue (1) if reached by at least two projectiles; otherwise a lone background landing becomes maroon (9).

    Key insight:
    - Shape orientation still determines direction, but the visible path change is that every sprite now contributes two shifted landing cells, so doubled shots create denser overlaps than the apex-fired original.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)

    comps: list[list[tuple[int, int]]] = []
    cell_to_comp: dict[tuple[int, int], int] = {}
    for col in sorted(set(int(v) for v in np.unique(g)) - {bg}):
        for comp in _cc8(g == col):
            cid = len(comps)
            comps.append(comp)
            for rc in comp:
                cell_to_comp[rc] = cid

    dest_to_shooters: dict[tuple[int, int], list[int]] = defaultdict(list)
    for cid, comp in enumerate(comps):
        dests = _arm_destinations(comp)
        if dests is None:
            continue
        for rr, cc in dests:
            if 0 <= rr < h and 0 <= cc < w:
                dest_to_shooters[(rr, cc)].append(cid)

    hit_components: set[int] = set()
    for dest in dest_to_shooters:
        if dest in cell_to_comp:
            hit_components.add(cell_to_comp[dest])

    out = g.copy()
    for cid in hit_components:
        for r, c in comps[cid]:
            out[r, c] = HIT_COLOR

    for (r, c), shooters in dest_to_shooters.items():
        if (r, c) in cell_to_comp:
            continue
        out[r, c] = COLLISION_COLOR if len(shooters) >= 2 else HIT_COLOR

    return out


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    """Original single-projectile behavior, kept only to reject degenerate generator samples."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)

    comps: list[list[tuple[int, int]]] = []
    cell_to_comp: dict[tuple[int, int], int] = {}
    for col in sorted(set(int(v) for v in np.unique(g)) - {bg}):
        for comp in _cc8(g == col):
            cid = len(comps)
            comps.append(comp)
            for rc in comp:
                cell_to_comp[rc] = cid

    dest_to_shooters: dict[tuple[int, int], list[int]] = defaultdict(list)
    for cid, comp in enumerate(comps):
        info = _triangle_info(comp)
        if info is None:
            continue
        (ar, ac), (dr, dc), _arms = info
        rr = ar + PROJECTILE_DIST * dr
        cc = ac + PROJECTILE_DIST * dc
        if 0 <= rr < h and 0 <= cc < w:
            dest_to_shooters[(rr, cc)].append(cid)

    hit_components: set[int] = set()
    for dest in dest_to_shooters:
        if dest in cell_to_comp:
            hit_components.add(cell_to_comp[dest])

    out = g.copy()
    for cid in hit_components:
        for r, c in comps[cid]:
            out[r, c] = HIT_COLOR

    for (r, c), shooters in dest_to_shooters.items():
        if (r, c) in cell_to_comp:
            continue
        out[r, c] = COLLISION_COLOR if len(shooters) >= 2 else HIT_COLOR

    return out


def _sprite_cells(apex: tuple[int, int], direction: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    ar, ac = apex
    dr, dc = direction
    if abs(dr) + abs(dc) == 2:
        return ((ar, ac), (ar - dr, ac), (ar, ac - dc))
    if dr != 0:
        return ((ar, ac), (ar - dr, ac - 1), (ar - dr, ac + 1))
    return ((ar, ac), (ar - 1, ac - dc), (ar + 1, ac - dc))


def _in_bounds_cells(cells: tuple[tuple[int, int], ...]) -> bool:
    return all(0 <= r < H and 0 <= c < W for r, c in cells)


def _halo(cells: tuple[tuple[int, int], ...]) -> frozenset[tuple[int, int]]:
    return frozenset(
        (rr, cc)
        for r, c in cells
        for rr in range(max(0, r - 1), min(H, r + 2))
        for cc in range(max(0, c - 1), min(W, c + 2))
    )


def _build_placements() -> tuple[list[dict], dict[tuple[int, int], list[dict]]]:
    placements: list[dict] = []
    by_any_dest: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for ar in range(H):
        for ac in range(W):
            for dr, dc in ALL_DIRS:
                cells = _sprite_cells((ar, ac), (dr, dc))
                if not _in_bounds_cells(cells):
                    continue
                arms = (cells[1], cells[2])
                dests = tuple((r + PROJECTILE_DIST * dr, c + PROJECTILE_DIST * dc) for r, c in arms)
                if not _in_bounds_cells(dests):
                    continue
                placement = {
                    "apex": (ar, ac),
                    "dir": (dr, dc),
                    "cells": cells,
                    "cell_set": frozenset(cells),
                    "halo": _halo(cells),
                    "arms": arms,
                    "dests": dests,
                    "legacy_dest": (ar + PROJECTILE_DIST * dr, ac + PROJECTILE_DIST * dc),
                }
                placements.append(placement)
                for dest in dests:
                    by_any_dest[dest].append(placement)
    return placements, by_any_dest


PLACEMENTS, PLACEMENTS_BY_ANY_DEST = _build_placements()


def _fits(placement: dict, occupied: set[tuple[int, int]], reserved_bg: set[tuple[int, int]]) -> bool:
    return not (placement["halo"] & occupied) and not (placement["cell_set"] & reserved_bg)


def _paint_sprite(grid: np.ndarray, placement: dict, color: int) -> None:
    for r, c in placement["cells"]:
        grid[r, c] = color


def _event_data(placements: list[dict]) -> tuple[dict[tuple[int, int], int], set[int], Counter]:
    cell_to_sprite: dict[tuple[int, int], int] = {}
    for idx, placement in enumerate(placements):
        for cell in placement["cells"]:
            cell_to_sprite[cell] = idx

    hit_sprites: set[int] = set()
    bg_counts: Counter = Counter()
    for placement in placements:
        for dest in placement["dests"]:
            if dest in cell_to_sprite:
                hit_sprites.add(cell_to_sprite[dest])
            else:
                bg_counts[dest] += 1
    return cell_to_sprite, hit_sprites, bg_counts


def _render_output(inp: np.ndarray, placements: list[dict]) -> np.ndarray:
    out = inp.copy()
    _cell_to_sprite, hit_sprites, bg_counts = _event_data(placements)
    for idx in hit_sprites:
        _paint_sprite(out, placements[idx], HIT_COLOR)
    for dest, count in bg_counts.items():
        out[dest] = COLLISION_COLOR if count >= 2 else HIT_COLOR
    return out


def _stats_for_placements(placements: list[dict], out: np.ndarray, legacy_out: np.ndarray) -> dict[str, int]:
    _cell_to_sprite, hit_sprites, bg_counts = _event_data(placements)
    return {
        "projectiles": 2 * len(placements),
        "hit_sprites": len(hit_sprites),
        "collision_cells": sum(1 for count in bg_counts.values() if count >= 2),
        "lone_bg": sum(1 for count in bg_counts.values() if count == 1),
        "bg_markers": len(bg_counts),
        "distinct_dests": len({dest for placement in placements for dest in placement["dests"]}),
        "legacy_diff_cells": int(np.count_nonzero(out != legacy_out)),
    }


def _validate_input(grid: np.ndarray, n_sprites: int) -> bool:
    if grid.shape != (H, W):
        return False
    if get_bg(grid) != BG:
        return False
    comps = components_mask(grid != BG, connectivity=8)
    if len(comps) != n_sprites:
        return False
    return all(comp.shape == (3, 2) for comp in comps)


def _choose_collision_seed(rng: random.Random) -> tuple[list[dict], set[tuple[int, int]]] | None:
    dests = list(PLACEMENTS_BY_ANY_DEST)
    rng.shuffle(dests)
    for dest in dests:
        candidates = PLACEMENTS_BY_ANY_DEST[dest][:]
        rng.shuffle(candidates)
        for i, first in enumerate(candidates):
            occ1 = set(first["cells"])
            for second in candidates[i + 1 :]:
                if second["cell_set"] & first["cell_set"]:
                    continue
                if second["halo"] & occ1:
                    continue
                return [first, second], {dest}
    return None


def _choose_hit_seed(rng: random.Random) -> tuple[list[dict], set[tuple[int, int]]] | None:
    bases = PLACEMENTS[:]
    rng.shuffle(bases)
    for base in bases:
        occupied = set(base["cells"])
        target_cells = list(base["cells"])
        rng.shuffle(target_cells)
        for cell in target_cells:
            candidates = PLACEMENTS_BY_ANY_DEST[cell][:]
            rng.shuffle(candidates)
            for shooter in candidates:
                if shooter["cell_set"] & base["cell_set"]:
                    continue
                if shooter["halo"] & occupied:
                    continue
                return [base, shooter], set()
    return None


def _fill_random(
    rng: random.Random,
    placements: list[dict],
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    target_n: int,
) -> bool:
    candidates = PLACEMENTS[:]
    rng.shuffle(candidates)
    for placement in candidates:
        if len(placements) >= target_n:
            return True
        if _fits(placement, occupied, reserved_bg):
            placements.append(placement)
            occupied.update(placement["cells"])
    return len(placements) >= target_n


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    sprite_color = rng.choice(SPRITE_COLORS)
    n_sprites = rng.randint(*SPRITE_COUNT_RANGE)

    seed_mode = rng.choice(["collision", "hit"])
    if seed_mode == "collision":
        seed = _choose_collision_seed(rng)
    else:
        seed = _choose_hit_seed(rng)
    if seed is None:
        return None

    placements, reserved_bg = seed
    placements = placements[:]
    occupied: set[tuple[int, int]] = set()
    for placement in placements:
        occupied.update(placement["cells"])

    if len(placements) > n_sprites:
        return None
    if not _fill_random(rng, placements, occupied, reserved_bg, n_sprites):
        return None

    inp = np.full((H, W), BG, dtype=int)
    for placement in placements:
        _paint_sprite(inp, placement, sprite_color)

    if not _validate_input(inp, n_sprites):
        return None

    out = _render_output(inp, placements)
    legacy_out = _legacy_main(inp)
    stats = _stats_for_placements(placements, out, legacy_out)

    if stats["hit_sprites"] + stats["collision_cells"] < 1:
        return None
    if stats["bg_markers"] < 2:
        return None
    if stats["bg_markers"] > MAX_BG_MARKERS:
        return None
    if stats["legacy_diff_cells"] < 2:
        return None
    if np.count_nonzero(out != BG) <= np.count_nonzero(inp != BG):
        return None
    if not np.any(out == HIT_COLOR):
        return None
    if np.array_equal(inp, out):
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {MAX_ATTEMPTS} attempts")
