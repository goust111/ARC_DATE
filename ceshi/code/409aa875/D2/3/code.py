# Variant of 409aa875
# Dimension: D2 — 投射距离 (path)
# Change: 固定距离5格（apex + 5 * direction）... → 距离等于同方向上到最近精灵的距离减1
#
# design_analysis:
#   variant_core:
#     original_path: 每个三像素三角精灵都把落点固定在 apex + 5 * dir。
#     new_path: 每个精灵沿自己的发射方向逐格搜索，遇到射线上的第一个“别的精灵像素”时，
#       投射物停在该像素前一格；若整条射线没有别的精灵，则这个精灵不产生可见落点。
#   a_concretization_and_trace:
#     input_objects: 仍然只有两类 3 像素精灵。L 形精灵沿对角方向发射，V 形精灵沿轴向发射。
#     changed_stage: 变化只发生在“由 apex/dir 推出落点”的阶段，shape→direction 的识别不变。
#     worked_example:
#       original: 若某个右射 V 形 apex 在 (8,2)，旧规则直接落到 (8,7)。
#       variant: 若该行最先遇到的别的精灵 apex 在 (8,9)，则投射物会落到 (8,8)；
#         若最近阻挡精灵改成在 (8,6)，则落点变成 (8,5)。同一朝向会因间距不同而给出不同输出。
#   b_complexity_calibration:
#     visual_signal: 样例里显式构造不同间距的“射手-阻挡者”链，并至少包含一次双射手同落点碰撞，
#       保证人类能从不同 gap 推断“不是固定 5 格”。
#     control: 仍只保留两种结果颜色（9 与 1），避免把复杂度扩散到颜色映射。
#   c_old_assumptions_to_remove:
#     - main() 中 `rr = ar + 5 * dr`, `cc = ac + 5 * dc` 依赖距离常量，变体下必须改成逐格 ray cast。
#     - generator 的 placement 预计算按固定 dest 建索引，这在动态距离下失效，必须改成显式摆放 blocker。
#     - 原先可直接把“目标格是否命中精灵”作为事件类型预先指定；新规则里目标由最近邻决定，
#       若中途出现更近精灵会改变结果，因此生成时必须检查射线无遮挡。
#   d_placement_legality:
#     - 所有精灵仍需是互不 8 邻接的 3 像素组件，避免组件合并导致 solver 误判。
#     - generator 通过 motif 构造：一个活跃 shooter 对应同方向、在前方的 blocker；
#       并验证 shooter 到 blocker 之前的射线格全为空。
#     - 样例中允许 blocker 自己没有前向邻居，此时它只是障碍物，不产生落点；solver 也按 no-op 处理。
#   e_visual_observability:
#     - 每个样例至少有一次蓝色碰撞点，并强制动态 solver 输出不同于旧 fixed-5 输出。
#     - 生成时先控制 shooter apex 与 blocker 的基础间距，再允许射线先撞到 blocker 的腿部像素；
#       因此实际距离会出现 1、2、3、... 的多档变化，视觉上明显不是常量 5。
#     - 额外要求动态 solver 输出与旧 fixed-5 solver 输出不同，确保变体特征真的出现在样例里。
#   f_cleanliness:
#     - 一个网格只放 1 个碰撞 motif，外加少量唯一落点 motif 与可选被动障碍，避免输出过密。
#     - 落点只用单像素标记，原精灵保持单色，结构清楚。
#   g_color_reasoning:
#     - 颜色规则完全沿用原题：单射击背景标 9，双射击背景标 1，不引入新的隐式颜色条件。
#   h_new_solver_assumptions_checked_against_generator:
#     - solver 假设精灵可被 `_triangle_apex_and_dir` 唯一识别；generator 只使用规范 L/V 形。
#     - solver 假设若 step == 1 则前一格会落回 apex，这种输入被视为无效近贴；generator 用 halo 间隔避免它。
#     - solver 允许“没有前向精灵则无投射物”；generator 允许少量纯 blocker 精灵，但至少保证两次有效射击。

import random
from collections import Counter, defaultdict

import numpy as np

from arc_utils import components_mask, get_bg as _bg_color, neighbors4, neighbors8


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _triangle_apex_and_dir(comp: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    # Returns (apex_rc, dir_rc). Direction is one of 4-neigh (axis) or diagonal.
    if len(comp) != 3:
        return None
    s = set(comp)

    deg4 = {}
    deg8 = {}
    for r, c in comp:
        deg4[(r, c)] = sum(1 for n in neighbors4(r, c) if n in s)
        deg8[(r, c)] = sum(1 for n in neighbors8(r, c) if n in s)

    corners = [p for p, d in deg4.items() if d == 2]
    if corners:
        corner = corners[0]
        cr, cc = corner
        arms = []
        for r, c in comp:
            if (r, c) == corner:
                continue
            dr, dc = r - cr, c - cc
            if abs(dr) + abs(dc) == 1:
                arms.append((dr, dc))
        if len(arms) != 2:
            return None
        dr_sum = arms[0][0] + arms[1][0]
        dc_sum = arms[0][1] + arms[1][1]
        dr = 0 if dr_sum == 0 else (-1 if dr_sum > 0 else 1)
        dc = 0 if dc_sum == 0 else (-1 if dc_sum > 0 else 1)
        return corner, (dr, dc)

    apexes = [p for p, d in deg8.items() if d == 2]
    if len(apexes) != 1:
        return None
    apex = apexes[0]
    ar, ac = apex
    others = [p for p in comp if p != apex]
    d1 = (others[0][0] - ar, others[0][1] - ac)
    d2 = (others[1][0] - ar, others[1][1] - ac)
    dr_sum = d1[0] + d2[0]
    dc_sum = d1[1] + d2[1]
    dr = -(dr_sum // 2)
    dc = -(dc_sum // 2)
    if abs(dr) + abs(dc) != 1:
        return None
    return apex, (dr, dc)


def _collect_components(g: np.ndarray, bg: int) -> tuple[list[list[tuple[int, int]]], dict[tuple[int, int], int]]:
    comps: list[list[tuple[int, int]]] = []
    cell_to_comp: dict[tuple[int, int], int] = {}
    for col in sorted(set(int(v) for v in np.unique(g)) - {bg}):
        for comp in _cc8(g == col):
            cid = len(comps)
            comps.append(comp)
            for rc in comp:
                cell_to_comp[rc] = cid
    return comps, cell_to_comp


def _dynamic_shots(
    comps: list[list[tuple[int, int]]],
    cell_to_comp: dict[tuple[int, int], int],
    shape: tuple[int, int],
) -> list[dict]:
    h, w = shape
    shots: list[dict] = []
    for cid, comp in enumerate(comps):
        ad = _triangle_apex_and_dir(comp)
        if ad is None:
            continue
        (ar, ac), (dr, dc) = ad
        step = 1
        while True:
            rr = ar + step * dr
            cc = ac + step * dc
            if not (0 <= rr < h and 0 <= cc < w):
                break
            other = cell_to_comp.get((rr, cc))
            if other is not None and other != cid:
                if step >= 2:
                    shots.append(
                        {
                            "cid": cid,
                            "apex": (ar, ac),
                            "dir": (dr, dc),
                            "dest": (ar + (step - 1) * dr, ac + (step - 1) * dc),
                            "blocker": other,
                            "distance": step - 1,
                        }
                    )
                break
            step += 1
    return shots


def _render_from_shots(
    g: np.ndarray,
    comps: list[list[tuple[int, int]]],
    cell_to_comp: dict[tuple[int, int], int],
    shots: list[dict],
) -> np.ndarray:
    out = g.copy()
    dest_to_shooters: dict[tuple[int, int], list[int]] = {}
    for shot in shots:
        dest_to_shooters.setdefault(shot["dest"], []).append(shot["cid"])

    hit_components: set[int] = set()
    for dest in dest_to_shooters:
        if dest in cell_to_comp:
            hit_components.add(cell_to_comp[dest])

    for cid in hit_components:
        for r, c in comps[cid]:
            out[r, c] = 9

    for dest, shooters in dest_to_shooters.items():
        if dest in cell_to_comp:
            continue
        out[dest] = 1 if len(shooters) >= 2 else 9

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire projectiles from 3-pixel triangle sprites; mark the stopping cells with maroon or blue.

    Input structure:
    - bg = most common color. Non-bg 3-pixel sprites are monochrome L-shapes or diagonal V-shapes, exactly as in the base task.

    Rule:
    - Each triangle still determines its apex and emission direction from its shape: L-shapes fire diagonally away from the missing 2x2 corner, V-shapes fire along the axis opposite the opening.
    - The projectile no longer travels a fixed 5 cells. Instead, walk cell-by-cell along the ray until the first pixel of a different sprite is encountered, then stop one cell earlier.
    - If multiple projectiles stop on the same empty background cell, that cell becomes blue (1); otherwise a single stopping cell on background becomes maroon (9). The hit-sprite recolor rule is kept for robustness, though valid generated instances stop before the blocker.

    Key insight: orientation still gives the direction, but the visible mark is controlled by the nearest forward neighbor on that ray, so the crucial inference is a per-sprite nearest-obstacle search rather than a constant offset.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    comps, cell_to_comp = _collect_components(g, bg)
    shots = _dynamic_shots(comps, cell_to_comp, g.shape)
    return _render_from_shots(g, comps, cell_to_comp, shots)


# ---- canvas / palette ----
H = W = 16
BG = 7
HIT_COLOR = 9
COLLISION_COLOR = 1
SPRITE_COLORS = [0, 1, 2, 3, 4, 5, 6, 8, 9]

# ---- geometry ----
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
AXIS_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_DIRS = DIAG_DIRS + AXIS_DIRS
DIST_CHOICES = [2, 3, 4, 6, 7]

# ---- generation retries ----
MAX_ATTEMPTS = 400


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


def _make_placement(apex: tuple[int, int], direction: tuple[int, int]) -> dict | None:
    cells = _sprite_cells(apex, direction)
    if not _in_bounds_cells(cells):
        return None
    return {
        "apex": apex,
        "dir": direction,
        "cells": cells,
        "cell_set": frozenset(cells),
        "halo": _halo(cells),
    }


def _build_placements() -> tuple[list[dict], dict[tuple[tuple[int, int], tuple[int, int]], dict]]:
    placements: list[dict] = []
    by_key: dict[tuple[tuple[int, int], tuple[int, int]], dict] = {}
    for ar in range(H):
        for ac in range(W):
            for direction in ALL_DIRS:
                placement = _make_placement((ar, ac), direction)
                if placement is None:
                    continue
                placements.append(placement)
                by_key[((ar, ac), direction)] = placement
    return placements, by_key


PLACEMENTS, PLACEMENT_BY_KEY = _build_placements()
ALL_DESTS = [(r, c) for r in range(H) for c in range(W)]


def _build_placements_by_cell() -> dict[tuple[int, int], list[dict]]:
    by_cell: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for placement in PLACEMENTS:
        for cell in placement["cells"]:
            by_cell[cell].append(placement)
    return by_cell


PLACEMENTS_BY_CELL = _build_placements_by_cell()


def _lookup(apex: tuple[int, int], direction: tuple[int, int]) -> dict | None:
    return PLACEMENT_BY_KEY.get((apex, direction))


def _fits(placement: dict | None, occupied: set[tuple[int, int]]) -> bool:
    return placement is not None and not (placement["halo"] & occupied)


def _paint_sprite(grid: np.ndarray, placement: dict, color: int) -> None:
    for r, c in placement["cells"]:
        grid[r, c] = color


def _ray_clear(
    shooter: dict,
    stop_cell: tuple[int, int],
    occupied: set[tuple[int, int]],
    ignore: set[tuple[int, int]],
) -> bool:
    ar, ac = shooter["apex"]
    dr, dc = shooter["dir"]
    rr, cc = ar + dr, ac + dc
    while (rr, cc) != stop_cell:
        if (rr, cc) in occupied and (rr, cc) not in ignore:
            return False
        rr += dr
        cc += dc
    return True


def _build_pair_candidates() -> tuple[list[dict], dict[tuple[int, int], list[dict]]]:
    pairs: list[dict] = []
    by_dest: dict[tuple[int, int], list[dict]] = defaultdict(list)
    seen: set[tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]] = set()
    for shooter in PLACEMENTS:
        ar, ac = shooter["apex"]
        dr, dc = shooter["dir"]
        step = 2
        while True:
            hit = (ar + step * dr, ac + step * dc)
            if not (0 <= hit[0] < H and 0 <= hit[1] < W):
                break
            dest = (ar + (step - 1) * dr, ac + (step - 1) * dc)
            for blocker in PLACEMENTS_BY_CELL.get(hit, []):
                if blocker is shooter:
                    continue
                key = (shooter["apex"], shooter["dir"], blocker["apex"], blocker["dir"])
                if key in seen:
                    continue
                if dest in shooter["cell_set"] or dest in blocker["cell_set"]:
                    continue
                occupied = set()
                if not _fits(shooter, occupied):
                    continue
                occupied.update(shooter["cells"])
                if not _fits(blocker, occupied):
                    continue
                rr, cc = ar + dr, ac + dc
                blocked_earlier = False
                while (rr, cc) != hit:
                    if (rr, cc) in blocker["cell_set"]:
                        blocked_earlier = True
                        break
                    rr += dr
                    cc += dc
                if blocked_earlier:
                    continue
                pair_data = {
                    "dest": dest,
                    "dir": shooter["dir"],
                    "distance": step - 1,
                    "shooter": shooter,
                    "blocker": blocker,
                    "stop_cell": hit,
                    "placements": (shooter, blocker),
                    "own_cells": set(shooter["cells"]) | set(blocker["cells"]),
                }
                pairs.append(pair_data)
                by_dest[dest].append(pair_data)
                seen.add(key)
            step += 1
    return pairs, by_dest


PAIR_CANDIDATES, PAIRS_BY_DEST = _build_pair_candidates()


def _try_add_collision_motif(
    rng: random.Random,
    sprites: list[dict],
    occupied: set[tuple[int, int]],
    reserved_dests: set[tuple[int, int]],
) -> bool:
    dests = list(PAIRS_BY_DEST)
    rng.shuffle(dests)
    for dest in dests:
        if dest in reserved_dests or dest in occupied:
            continue
        candidates = PAIRS_BY_DEST[dest][:]
        rng.shuffle(candidates)
        for i, first in enumerate(candidates):
            temp_occupied = set(occupied)
            ok = True
            for placement in first["placements"]:
                if not _fits(placement, temp_occupied):
                    ok = False
                    break
                temp_occupied.update(placement["cells"])
            if not ok:
                continue
            for second in candidates[i + 1 :]:
                if second["dir"] == first["dir"] or second["distance"] == first["distance"]:
                    continue
                temp2 = set(temp_occupied)
                ok2 = True
                for placement in second["placements"]:
                    if not _fits(placement, temp2):
                        ok2 = False
                        break
                    temp2.update(placement["cells"])
                if not ok2:
                    continue
                if not _ray_clear(first["shooter"], first["stop_cell"], temp2, first["own_cells"]):
                    continue
                if not _ray_clear(second["shooter"], second["stop_cell"], temp2, second["own_cells"]):
                    continue
                sprites.extend(first["placements"])
                sprites.extend(second["placements"])
                occupied.update(first["own_cells"])
                occupied.update(second["own_cells"])
                reserved_dests.add(dest)
                return True
    return False


def _try_add_unique_motif(
    rng: random.Random,
    sprites: list[dict],
    occupied: set[tuple[int, int]],
    reserved_dests: set[tuple[int, int]],
) -> bool:
    dests = list(PAIRS_BY_DEST)
    rng.shuffle(dests)
    for dest in dests:
        if dest in reserved_dests or dest in occupied:
            continue
        candidates = PAIRS_BY_DEST[dest][:]
        rng.shuffle(candidates)
        for pair in candidates:
            temp_occupied = set(occupied)
            ok = True
            for placement in pair["placements"]:
                if not _fits(placement, temp_occupied):
                    ok = False
                    break
                temp_occupied.update(placement["cells"])
            if not ok:
                continue
            if not _ray_clear(pair["shooter"], pair["stop_cell"], temp_occupied, pair["own_cells"]):
                continue
            sprites.extend(pair["placements"])
            occupied.update(pair["own_cells"])
            reserved_dests.add(dest)
            return True
    return False


def _try_add_passive(
    rng: random.Random,
    sprites: list[dict],
    occupied: set[tuple[int, int]],
) -> bool:
    candidates = PLACEMENTS[:]
    rng.shuffle(candidates)
    for placement in candidates:
        if _fits(placement, occupied):
            sprites.append(placement)
            occupied.update(placement["cells"])
            return True
    return False


def _validate_input(grid: np.ndarray, n_sprites: int) -> bool:
    if grid.shape != (H, W):
        return False
    if _bg_color(grid) != BG:
        return False
    comps = components_mask(grid != BG, connectivity=8)
    if len(comps) != n_sprites:
        return False
    return all(comp.shape == (3, 2) for comp in comps)


def _fixed5_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    comps, cell_to_comp = _collect_components(g, bg)
    shots = []
    for cid, comp in enumerate(comps):
        ad = _triangle_apex_and_dir(comp)
        if ad is None:
            continue
        (ar, ac), (dr, dc) = ad
        rr = ar + 5 * dr
        cc = ac + 5 * dc
        if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]:
            shots.append({"cid": cid, "dest": (rr, cc)})
    return _render_from_shots(g, comps, cell_to_comp, shots)


def _scene_metrics(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    comps, cell_to_comp = _collect_components(g, bg)
    shots = _dynamic_shots(comps, cell_to_comp, g.shape)
    dest_counts = Counter(shot["dest"] for shot in shots if shot["dest"] not in cell_to_comp)
    return {
        "shots": shots,
        "distances": [shot["distance"] for shot in shots],
        "active_shots": len(shots),
        "bg_collision_count": sum(1 for count in dest_counts.values() if count >= 2),
        "bg_unique_count": sum(1 for count in dest_counts.values() if count == 1),
        "hit_count": sum(1 for shot in shots if shot["dest"] in cell_to_comp),
    }


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    sprite_color = rng.choice(SPRITE_COLORS)
    unique_count, passive_count = rng.choice(
        [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
            (2, 0),
            (2, 1),
        ]
    )
    target_sprites = 4 + 2 * unique_count + passive_count

    sprites: list[dict] = []
    occupied: set[tuple[int, int]] = set()
    reserved_dests: set[tuple[int, int]] = set()

    if not _try_add_collision_motif(rng, sprites, occupied, reserved_dests):
        return None
    for _ in range(unique_count):
        if not _try_add_unique_motif(rng, sprites, occupied, reserved_dests):
            return None
    for _ in range(passive_count):
        if not _try_add_passive(rng, sprites, occupied):
            return None

    if len(sprites) != target_sprites:
        return None

    inp = np.full((H, W), BG, dtype=int)
    for placement in sprites:
        _paint_sprite(inp, placement, sprite_color)

    if not _validate_input(inp, target_sprites):
        return None

    metrics = _scene_metrics(inp)
    if metrics["active_shots"] < 2:
        return None
    if metrics["bg_collision_count"] < 1:
        return None
    if metrics["hit_count"] != 0:
        return None
    if not metrics["distances"] or any(dist <= 0 for dist in metrics["distances"]):
        return None
    if not any(dist != 5 for dist in metrics["distances"]):
        return None
    if len(set(metrics["distances"])) < 2:
        return None

    out = main(inp)
    fixed5 = _fixed5_output(inp)
    if np.array_equal(out, inp):
        return None
    if np.array_equal(out, fixed5):
        return None
    if not np.any(out == COLLISION_COLOR):
        return None
    if not np.any(out == HIT_COLOR):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {MAX_ATTEMPTS} attempts")
