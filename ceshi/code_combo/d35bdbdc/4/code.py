# Combo variant of d35bdbdc (index 4)
# combo_spec: D2/2+D4/2
# Name: 空间邻近配对+面积判定目标
#
# Design analysis
# spec:
#   concrete_visual:
#     - 移除原题的灰色路径，画面只剩背景 0 和若干带边框精灵。
#     - 仍沿用“单中心像素 + 同色边框”的两色 box 精灵语法，避免引入新的识别对象。
#     - 两个目标精灵固定为完整 3x3 方框，面积 9；两个源精灵固定为边界裁剪后的 2x3 方框，面积 6。
#     - 目标与其对应源在空间上靠近，但两组之间拉开距离，因此人眼可以先看出“大框留下”，再看出“每个大框吸收最近小框的中心色”。
#   behavior_trace:
#     - 原题：灰路径端点决定哪两个精灵保留，再用目标中心色去查某个 source 的边框色。
#     - 本融合版：先完全忽略路径，因为输入里不再有路径；先按面积挑出两个大精灵作为目标。
#     - 然后只在剩余小精灵中做最近距离的一对一配对：每个目标接收离自己最近且尚未配对的小精灵的中心色。
#     - 差异核心：角色判定来自面积，配对关系来自几何距离，颜色链不再承担匹配功能。
#   complexity_calibration:
#     - 精灵总数固定为 4：2 个大目标 + 2 个小源。这样每题都同时展示“面积判定”和“两组最近配对”两个步骤，不会被多余 distractor 稀释。
#     - 面积档位只保留 9 和 6，两大两小，目标集合无歧义。
#     - 距离布局使用少量模板，并强制同组距离显著小于跨组距离，避免最近配对出现平局或需要不可视 tie-break。
#     - 目标输入中心色全部选自“非任何边框色”的颜色集合，防止退化回原题的 border-lookup 解释。
#   implicit_assumption_audit:
#     - 原 solver 依赖 grey path、端点、port 和 border->center 查找；这些前提在本变体中全部失效，因此整段路径逻辑必须删除。
#     - 新 solver 的隐式假设 1：恰好存在两个最大面积精灵。生成器用两个完整 3x3 和两个裁剪 2x3 保证。
#     - 新 solver 的隐式假设 2：最近配对唯一。生成器把同组中心距控制在 4-5，而跨组中心距通常在 8 以上。
#     - 新 solver 的隐式假设 3：删除小精灵后，保留对象仍是独立 component。生成器继续禁止重叠和 4 邻接。
#   placement_legality:
#     - 所有 patch 都从合法模板位点中放置；完整目标只放在完全可见位置，小源只放在单边裁剪位置。
#     - `_overlaps()` 同时禁止重叠与 4 邻接，确保组件检测稳定、不会把两个精灵误并成一个 component。
#     - 生成器采用少量经过验证的布局模板，直接保证目标与源之间留有至少一格空隙。
#   visual_observability:
#     - 输入到输出最显眼的现象是：两个大框保留，两个小框消失；这让“按面积选目标”肉眼可见。
#     - 两个大框各自变成邻近小框的中心色，且原始大框中心色并不对应任何边框色，因此观察者会被迫放弃原题那种颜色查表解释。
#   cleanliness_and_inference:
#     - 没有灰路径后，画面层次更单纯，不会出现“路径穿过对象”造成的阅读噪声。
#     - 颜色总量受控：4 个边框色 + 4 个非边框中心色，且输出只发生两个中心像素变化，足以让少量样例支持规则归纳。

import itertools
import random

import numpy as np

from arc_utils import components_mask, neighbors4, paste


GRID_H, GRID_W = 10, 10
BG, GREY = 0, 5
GEN_RETRIES = 400


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _detect_bordered_sprites(g: np.ndarray, bg: int, grey: int) -> list[dict]:
    """
    Bordered sprite = a 4-connected component (excluding bg/grey) with exactly 2 colors:
      - border color appears multiple times
      - center color appears exactly once (its location is the 'center')
    This also handles clipped/partial borders near the grid boundary.
    """
    mask = (g != bg) & (g != grey)
    comps = components_mask(mask, connectivity=4)
    sprites: list[dict] = []
    for pts in comps:
        cols = [int(g[r, c]) for r, c in pts]
        uniq = sorted(set(cols))
        if len(uniq) != 2:
            continue
        counts = {c: cols.count(c) for c in uniq}
        center_color = min(counts, key=lambda c: counts[c])
        if counts[center_color] != 1:
            continue
        border_color = [c for c in uniq if c != center_color][0]
        if border_color in (bg, grey) or center_color in (bg, grey):
            continue
        center_pos = None
        for r, c in pts:
            if int(g[r, c]) == center_color:
                center_pos = (int(r), int(c))
                break
        if center_pos is None:
            continue
        sprites.append(
            {
                "pts": pts,
                "border": int(border_color),
                "center": int(center_color),
                "center_pos": center_pos,
                "area": len(pts),
            }
        )
    return sprites


def _largest_destinations(sprites: list[dict], k: int = 2) -> list[dict]:
    ranked = sorted(
        sprites,
        key=lambda sp: (-int(sp["area"]), int(sp["center_pos"][0]), int(sp["center_pos"][1])),
    )
    assert len(ranked) >= k, f"expected at least {k} bordered sprites"
    return ranked[:k]


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _pair_by_proximity(destinations: list[dict], sources: list[dict]) -> list[tuple[dict, dict]]:
    """
    Match each destination to a distinct nearby source by minimum total Manhattan distance.
    The generator guarantees a unique optimum; tie-breakers only stabilize arbitrary inputs.
    """
    assert len(destinations) == 2, "expected exactly two destinations"
    assert len(sources) >= 2, "expected at least two source candidates"

    best_pairs: list[tuple[dict, dict]] | None = None
    best_key: tuple | None = None
    for subset in itertools.combinations(sources, len(destinations)):
        for perm in itertools.permutations(subset):
            dists = tuple(_manhattan(d["center_pos"], s["center_pos"]) for d, s in zip(destinations, perm))
            source_pos = tuple((int(s["center_pos"][0]), int(s["center_pos"][1])) for s in perm)
            key = (sum(dists), max(dists), dists, source_pos)
            if best_key is None or key < best_key:
                best_key = key
                best_pairs = list(zip(destinations, perm))
    assert best_pairs is not None, "failed to form destination-source pairs"
    return best_pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer colors from nearby small sprites into the largest bordered sprites, deleting the rest.

    Input structure:
    - The grid contains several two-color bordered box sprites and no grey path structure.
    - Exactly two sprites are fully visible 3x3 boxes, while the remaining sprites are smaller clipped boxes near the border.

    Rule:
    - Rank all bordered sprites by visible area and treat the two largest sprites as destinations.
    - Among the remaining smaller sprites, pair each destination with a distinct source using nearest-center Manhattan distance.
    - Copy each paired source's center color into its destination center.
    - Delete every non-destination sprite.

    Key insight:
    - The task is solved by combining two geometric cues: area identifies which sprites survive, and spatial proximity decides which disappearing sprite donates its center color to each survivor.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    grey = GREY

    sprites = _detect_bordered_sprites(g, bg, grey)
    assert len(sprites) >= 4, "expected at least four bordered sprites"

    dest = _largest_destinations(sprites, k=2)
    dest_ids = {id(sp) for sp in dest}
    sources = [sp for sp in sprites if id(sp) not in dest_ids]
    pairs = _pair_by_proximity(dest, sources)

    out = g.copy()
    for sp in sprites:
        if id(sp) in dest_ids:
            continue
        for r, c in sp["pts"]:
            out[r, c] = bg

    keep = np.zeros(out.shape, dtype=bool)
    q: list[tuple[int, int]] = []
    for d in dest:
        for r, c in d["pts"]:
            if out[r, c] != bg and not keep[r, c]:
                keep[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.pop()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1]:
                if keep[nr, nc] or out[nr, nc] == bg:
                    continue
                keep[nr, nc] = True
                q.append((nr, nc))
    out[(out != bg) & (~keep)] = bg

    for d, src in pairs:
        dr, dc = d["center_pos"]
        out[dr, dc] = int(src["center"])

    return out


def _make_box_patch(border_color: int, center_color: int) -> np.ndarray:
    patch = np.full((3, 3), border_color, dtype=int)
    patch[1, 1] = center_color
    return patch


def _place_sprite(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    paste(grid, patch, r0, c0, transparent=BG, clip=True)


def _erase_sprite(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = grid.shape
    for dr in range(3):
        for dc in range(3):
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < h and 0 <= cc < w and patch[dr, dc] != BG:
                grid[rr, cc] = BG


def _overlaps(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    """Reject overlap and 4-neighbor touching between sprites."""
    h, w = grid.shape
    patch_cells = set()
    for dr in range(3):
        for dc in range(3):
            if patch[dr, dc] == BG:
                continue
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < h and 0 <= cc < w:
                if grid[rr, cc] != BG:
                    return True
                patch_cells.add((rr, cc))
    for rr, cc in patch_cells:
        for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
            if (nr, nc) not in patch_cells and 0 <= nr < h and 0 <= nc < w:
                if int(grid[nr, nc]) != BG:
                    return True
    return False


def _visible_area(patch: np.ndarray, r0: int, c0: int) -> int:
    area = 0
    for dr in range(3):
        for dc in range(3):
            if patch[dr, dc] == BG:
                continue
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < GRID_H and 0 <= cc < GRID_W:
                area += 1
    return area


def _layout_specs() -> list[dict]:
    return [
        {
            "dests": [(3, 1), (4, 6)],
            "sources": [(-1, 1), (8, 6)],
        },
        {
            "dests": [(4, 1), (3, 6)],
            "sources": [(8, 1), (-1, 6)],
        },
        {
            "dests": [(1, 3), (6, 4)],
            "sources": [(1, -1), (6, 8)],
        },
        {
            "dests": [(1, 4), (6, 3)],
            "sources": [(1, 8), (6, -1)],
        },
        {
            "dests": [(3, 2), (4, 5)],
            "sources": [(-1, 2), (8, 5)],
        },
        {
            "dests": [(2, 3), (5, 4)],
            "sources": [(2, -1), (5, 8)],
        },
    ]


def _sample_role_colors() -> dict:
    border_colors = random.sample([c for c in range(1, 10) if c != GREY], 4)
    non_border = [c for c in range(1, 10) if c not in border_colors and c != GREY]
    assert len(non_border) == 4
    random.shuffle(non_border)
    target_centers = non_border[:2]
    source_centers = non_border[2:]
    random.shuffle(source_centers)
    return {
        "dest_borders": border_colors[:2],
        "source_borders": border_colors[2:],
        "target_centers": target_centers,
        "source_centers": source_centers,
    }


def _pairing_margin(dests: list[dict], sources: list[dict]) -> int:
    pairs = _pair_by_proximity(dests, sources)
    paired_dist = max(_manhattan(d["center_pos"], s["center_pos"]) for d, s in pairs)
    pair_ids = {(id(d), id(s)) for d, s in pairs}
    cross_nonpair = [
        _manhattan(d["center_pos"], s["center_pos"])
        for d in dests
        for s in sources
        if (id(d), id(s)) not in pair_ids
    ]
    return min(cross_nonpair) - paired_dist


def _build_example(layout: dict, colors: dict) -> tuple[np.ndarray, np.ndarray]:
    output_grid = np.zeros((GRID_H, GRID_W), dtype=int)
    input_grid = np.zeros((GRID_H, GRID_W), dtype=int)

    full_patch_specs = []
    source_patch_specs = []
    for i, (r0, c0) in enumerate(layout["dests"]):
        out_patch = _make_box_patch(colors["dest_borders"][i], colors["source_centers"][i])
        in_patch = _make_box_patch(colors["dest_borders"][i], colors["target_centers"][i])
        assert _visible_area(out_patch, r0, c0) == 9
        full_patch_specs.append((r0, c0, out_patch, in_patch))
    for i, (r0, c0) in enumerate(layout["sources"]):
        patch = _make_box_patch(colors["source_borders"][i], colors["source_centers"][i])
        assert _visible_area(patch, r0, c0) == 6
        source_patch_specs.append((r0, c0, patch))

    planning = np.zeros((GRID_H, GRID_W), dtype=int)
    for r0, c0, out_patch, _in_patch in full_patch_specs:
        if _overlaps(planning, out_patch, r0, c0):
            raise AssertionError("invalid full destination placement")
        _place_sprite(planning, out_patch, r0, c0)
    for r0, c0, patch in source_patch_specs:
        if _overlaps(planning, patch, r0, c0):
            raise AssertionError("invalid source placement")
        _place_sprite(planning, patch, r0, c0)

    for r0, c0, out_patch, in_patch in full_patch_specs:
        _place_sprite(output_grid, out_patch, r0, c0)
        _place_sprite(input_grid, in_patch, r0, c0)
    for r0, c0, patch in source_patch_specs:
        _place_sprite(input_grid, patch, r0, c0)

    return input_grid, output_grid


def _gen_attempt() -> np.ndarray | None:
    layout = random.choice(_layout_specs())
    colors = _sample_role_colors()
    try:
        input_grid, output_grid = _build_example(layout, colors)
    except AssertionError:
        return None

    sprites = _detect_bordered_sprites(input_grid, BG, GREY)
    if len(sprites) != 4:
        return None
    areas = sorted(int(sp["area"]) for sp in sprites)
    if areas != [6, 6, 9, 9]:
        return None

    dests = _largest_destinations(sprites, k=2)
    sources = [sp for sp in sprites if id(sp) not in {id(d) for d in dests}]
    if _pairing_margin(dests, sources) <= 0:
        return None
    if any(int(d["center"]) in {int(s["border"]) for s in sources} for d in dests):
        return None

    try:
        out = main(input_grid)
    except Exception:
        return None
    if np.array_equal(out, output_grid):
        return input_grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {GEN_RETRIES} attempts")
