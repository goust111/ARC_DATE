# Variant of 7ed72f31
# Dimension: D16 — 精灵与标记之间的基数关系 (meta)
# Change: 一对一关系：每个标记对应一个精灵（或一组8邻域精灵），每个精灵有且只有一个标记... → 一对多：一个精灵被多个标记反射

from __future__ import annotations

import itertools
import random

import numpy as np
from scipy import ndimage

from arc_utils import STRUCT8, get_bg, label_mask


"""
design_analysis:
  concrete_behavior:
    original:
      - 红线通常嵌在精灵组件内部，或单个红点单独挨着一个精灵。
      - 一个标记只驱动一个精灵；生成器也只会给每个精灵安排一个标记。
    variant:
      - 保留同样的三类轴语义：水平线、垂直线、单点。
      - 但为了让"同一精灵属于多个标记"在4连通分量里可见，红标记改为与彩色精灵分离的纯红组件。
      - 一个彩色精灵的紧包围盒外侧可以同时放多个红标记：
        - 顶/底一整条水平红线，和精灵之间固定隔着1格背景，跨度等于精灵bbox宽度。
        - 左/右一整条垂直红线，和精灵之间固定隔着1格背景，跨度等于精灵bbox高度。
        - bbox外一格的单个红点（生成器固定放在角点，并要求与真实彩色像素对角接触）。
      - 输出中，同一精灵会针对每个邻近红标记各反射一次，因此一个输入精灵可生成2-3份镜像副本。
    trace_example:
      - 假设一个绿色L形精灵bbox为3x3，顶部一格有长度3的红横线，左侧一格有长度3的红竖线。
      - 原逻辑：这两个红线既不在精灵内部，也不是孤立单点，因此不会驱动任何反射。
      - 新逻辑：顶部红线把整个L形向上镜像一次；左侧红线再把同一L形向左镜像一次。
      - 差异核心不在反射公式，而在"同一彩色组件会被多个分离红组件分别认领"。
  complexity_calibration:
    - 不引入颜色映射；颜色仍只表示所属精灵，避免把复杂度转移到不可见规则。
    - 标记类型只保留三档：row/col/point，和原题一致。
    - 每个样例含3-4个精灵，其中至少1个精灵有2-3个标记；其余精灵多为单标记，给解题者提供对照。
    - 多标记精灵至少包含一个线标记，保证与原solver在同一输入上行为不同，而不是退化成原本就能处理的多红点情况。
  implicit_assumptions_review:
    original_main_risks:
      - `label_mask(g != bg, connectivity=4)` 后，原代码把"含红组件"视为反射主体，默认红轴与payload在同一4连通块里。
      - 原孤立红点分支只收 `len(comp["coords"]) == 1` 的红组件，因此纯红线组件会被完全忽略。
      - 原嵌入式分支通过全图扫描同色像素，默认颜色和局部bbox足以唯一锁定payload；一旦一个颜色在附近重复，这种写法会有误吸附风险。
    variant_response:
      - 新solver把三类组件拆开处理：混合组件沿用原题思路；纯红组件单独分类；纯彩色组件单独作为可被多个标记命中的精灵。
      - 对混合组件，只反射该组件内部的非红像素，去掉对"附近同色像素不会混入"的隐式假设。
      - 对纯红线组件，新增bbox外侧一格匹配规则；这是D16真正需要的新代码路径。
  placement_legality:
    - 生成器先构造完整输出patch：原精灵 + 全部镜像 + 红标记，再整体放置，保证不会越界。
    - 同一bundle内部会拒绝以下情况：
      - 任意两个红标记4连通接触，导致合并成L形/十字形歧义组件。
      - 不同镜像副本彼此重叠。
      - 镜像覆盖原精灵或红标记。
    - 全局放置时按输出footprint做带padding的空位采样，避免不同bundle互相碰撞，保证solver匹配唯一。
  visual_observability:
    - 输入里能直接看到"一个精灵旁边围着两个或三个红标记"；输出里会对应出现多个方向的副本。
    - 2-3组样例足以归纳：红线/红点都像独立辐射源，同一精灵若邻近多个，就复制多次。
  visual_cleanliness:
    - 红点只放在角外一格，且要求对应角上有真实payload像素，避免悬空红点。
    - 每个bundle最多3个标记，避免单区块过于拥挤。
    - 颜色数维持3-4种，bundle之间分开摆放。
  color_inference:
    - 无颜色变换；所有新增像素保留原精灵颜色。
  new_solver_assumptions_to_audit:
    - 纯红水平线/垂直线必须是连续直线；生成器严格构造连续线。
      - 纯红线与目标精灵bbox之间固定隔着1格背景，因此轴位置是bbox外第二行/列；生成器按该几何关系放置。
    - 纯红点位于精灵bbox一格halo中，且不会同时落入两个精灵的可匹配halo；生成器通过bundle级独立放置与全局padding保证唯一匹配。
"""


MARK = 2

# Canvas
GRID_MIN, GRID_MAX = 18, 28

# Bundle / marker counts
BUNDLES_MIN, BUNDLES_MAX = 3, 4
MARKERS_TOTAL_MIN, MARKERS_TOTAL_MAX = 4, 7
MARKERS_PER_BUNDLE_MAX = 3

# Payload shape
PAYLOAD_SIDE_MIN, PAYLOAD_SIDE_MAX = 2, 5
PAYLOAD_PX_MIN, PAYLOAD_PX_MAX = 3, 8
PAYLOAD_DENSITY_LO, PAYLOAD_DENSITY_HI = 0.45, 0.78

# Color palette
PALETTE_COLORS_MIN, PALETTE_COLORS_MAX = 3, 4

# Layout
PLACEMENT_PADDING = 1
BG_COVERAGE_MIN = 0.88

# Retries
MASK_RETRIES = 50
BUNDLE_RETRIES = 80
GEN_RETRIES = 400

_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_MARKER_KINDS = ("top", "bottom", "left", "right", "tl", "tr", "bl", "br")


def _inb(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= int(r) < int(h) and 0 <= int(c) < int(w)


def _refl_point(p: tuple[int, int], r: int, c: int) -> tuple[int, int]:
    pr, pc = map(int, p)
    return int(2 * pr - r), int(2 * pc - c)


def _refl_row(r0: int, r: int, c: int) -> tuple[int, int]:
    return int(2 * int(r0) - int(r)), int(c)


def _refl_col(c0: int, r: int, c: int) -> tuple[int, int]:
    return int(r), int(2 * int(c0) - int(c))


def _classify_red_component(coords: list[tuple[int, int]]) -> tuple[str, object] | None:
    reds = sorted((int(r), int(c)) for r, c in coords)
    if not reds:
        return None
    if len(reds) == 1:
        return ("point", reds[0])
    rs = {r for r, _ in reds}
    cs = {c for _, c in reds}
    if len(rs) == 1:
        row = reds[0][0]
        cols = sorted(c for _, c in reds)
        if cols == list(range(cols[0], cols[-1] + 1)):
            return ("row", (int(row), int(cols[0]), int(cols[-1])))
    if len(cs) == 1:
        col = reds[0][1]
        rows = sorted(r for r, _ in reds)
        if rows == list(range(rows[0], rows[-1] + 1)):
            return ("col", (int(col), int(rows[0]), int(rows[-1])))
    return None


def _marker_matches_sprite(
    marker: tuple[str, object],
    sprite: dict,
) -> bool:
    bbox = sprite["bbox"]
    coordset = set((int(r), int(c)) for r, c in sprite["coords"])
    r0, c0, r1, c1 = map(int, bbox)
    kind, data = marker
    if kind == "row":
        row, mc0, mc1 = map(int, data)
        return int(mc0) == int(c0) and int(mc1) == int(c1) and int(row) in (int(r0 - 2), int(r1 + 2))
    if kind == "col":
        col, mr0, mr1 = map(int, data)
        return int(mr0) == int(r0) and int(mr1) == int(r1) and int(col) in (int(c0 - 2), int(c1 + 2))
    if kind == "point":
        pr, pc = map(int, data)
        point_matches = {
            (int(r0 - 1), int(c0 - 1)): (int(r0), int(c0)),
            (int(r0 - 1), int(c1 + 1)): (int(r0), int(c1)),
            (int(r1 + 1), int(c0 - 1)): (int(r1), int(c0)),
            (int(r1 + 1), int(c1 + 1)): (int(r1), int(c1)),
        }
        return (int(pr), int(pc)) in point_matches and point_matches[(int(pr), int(pc))] in coordset
    return False


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    h, w = g.shape

    lab, n = label_mask(g != int(bg), connectivity=4)
    mixed_components: list[dict] = []
    pure_red_markers: list[tuple[str, object]] = []
    standalone_sprites: list[dict] = []

    for k in range(1, int(n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab == int(k))]
        if not coords:
            continue
        vals = [int(g[r, c]) for r, c in coords]
        r0 = min(r for r, _ in coords)
        c0 = min(c for _, c in coords)
        r1 = max(r for r, _ in coords)
        c1 = max(c for _, c in coords)
        comp = {"coords": coords, "vals": vals, "bbox": (int(r0), int(c0), int(r1), int(c1))}
        only_red = all(int(v) == int(MARK) for v in vals)
        has_red = any(int(v) == int(MARK) for v in vals)
        if only_red:
            marker = _classify_red_component(coords)
            if marker is not None:
                pure_red_markers.append(marker)
        elif has_red:
            mixed_components.append(comp)
        else:
            standalone_sprites.append(comp)

    out = g.copy()

    # 1) Preserve original embedded behavior, but only reflect pixels belonging to the same mixed component.
    for comp in mixed_components:
        reds = [(int(r), int(c)) for (r, c), v in zip(comp["coords"], comp["vals"]) if int(v) == int(MARK)]
        marker = _classify_red_component(reds)
        if marker is None:
            continue
        for (r, c), v in zip(comp["coords"], comp["vals"]):
            v = int(v)
            if v == int(MARK):
                continue
            if marker[0] == "row":
                rr, cc = _refl_row(int(marker[1][0]), int(r), int(c))
            elif marker[0] == "col":
                rr, cc = _refl_col(int(marker[1][0]), int(r), int(c))
            else:
                rr, cc = _refl_point(marker[1], int(r), int(c))
            if _inb(h, w, rr, cc):
                out[int(rr), int(cc)] = int(v)

    # 2) Variant path: one standalone sprite can be claimed by multiple detached red markers.
    for spr in standalone_sprites:
        matches = [marker for marker in pure_red_markers if _marker_matches_sprite(marker, spr)]
        if not matches:
            continue
        for marker in matches:
            for r, c in spr["coords"]:
                v = int(g[int(r), int(c)])
                if marker[0] == "row":
                    rr, cc = _refl_row(int(marker[1][0]), int(r), int(c))
                elif marker[0] == "col":
                    rr, cc = _refl_col(int(marker[1][0]), int(r), int(c))
                else:
                    rr, cc = _refl_point(marker[1], int(r), int(c))
                if _inb(h, w, rr, cc):
                    out[int(rr), int(cc)] = int(v)

    return out


def _random_connected_mask(h: int, w: int, target_pixels: int, *, rng: random.Random) -> np.ndarray | None:
    if target_pixels < 1 or target_pixels > h * w:
        return None
    mask = np.zeros((h, w), dtype=bool)
    mask[rng.randrange(h), rng.randrange(w)] = True
    count = 1
    while count < target_pixels:
        frontier = []
        for r in range(h):
            for c in range(w):
                if mask[r, c]:
                    continue
                for dr, dc in _DIRS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                        frontier.append((r, c))
                        break
        if not frontier:
            return None
        r, c = rng.choice(frontier)
        mask[r, c] = True
        count += 1
    if not mask.any(axis=1).all() or not mask.any(axis=0).all():
        return None
    return mask


def _marker_cells(kind: str, ph: int, pw: int) -> list[tuple[int, int]]:
    if kind == "top":
        return [(-2, c) for c in range(pw)]
    if kind == "bottom":
        return [(ph + 1, c) for c in range(pw)]
    if kind == "left":
        return [(r, -2) for r in range(ph)]
    if kind == "right":
        return [(r, pw + 1) for r in range(ph)]
    if kind == "tl":
        return [(-1, -1)]
    if kind == "tr":
        return [(-1, pw)]
    if kind == "bl":
        return [(ph, -1)]
    if kind == "br":
        return [(ph, pw)]
    raise ValueError(f"unknown marker kind: {kind}")


def _mirror_coords_for_marker(
    kind: str,
    payload_coords: list[tuple[int, int]],
    ph: int,
    pw: int,
) -> list[tuple[int, int]]:
    if kind == "top":
        return [_refl_row(-2, r, c) for r, c in payload_coords]
    if kind == "bottom":
        return [_refl_row(ph + 1, r, c) for r, c in payload_coords]
    if kind == "left":
        return [_refl_col(-2, r, c) for r, c in payload_coords]
    if kind == "right":
        return [_refl_col(pw + 1, r, c) for r, c in payload_coords]
    if kind == "tl":
        return [_refl_point((-1, -1), r, c) for r, c in payload_coords]
    if kind == "tr":
        return [_refl_point((-1, pw), r, c) for r, c in payload_coords]
    if kind == "bl":
        return [_refl_point((ph, -1), r, c) for r, c in payload_coords]
    if kind == "br":
        return [_refl_point((ph, pw), r, c) for r, c in payload_coords]
    raise ValueError(f"unknown marker kind: {kind}")


def _point_marker_supported(kind: str, pmask: np.ndarray) -> bool:
    if kind == "tl":
        return bool(pmask[0, 0])
    if kind == "tr":
        return bool(pmask[0, -1])
    if kind == "bl":
        return bool(pmask[-1, 0])
    if kind == "br":
        return bool(pmask[-1, -1])
    return True


def _markers_stay_separate(marker_kinds: tuple[str, ...], ph: int, pw: int) -> bool:
    markers = [set(_marker_cells(kind, ph, pw)) for kind in marker_kinds]
    for i in range(len(markers)):
        for j in range(i + 1, len(markers)):
            for r1, c1 in markers[i]:
                for r2, c2 in markers[j]:
                    if abs(int(r1) - int(r2)) + abs(int(c1) - int(c2)) <= 1:
                        return False
    return True


def _build_bundle(marker_count: int, require_line: bool, rng: random.Random) -> dict | None:
    for _ in range(BUNDLE_RETRIES):
        ph = rng.randint(PAYLOAD_SIDE_MIN, PAYLOAD_SIDE_MAX)
        pw = rng.randint(PAYLOAD_SIDE_MIN, PAYLOAD_SIDE_MAX)
        max_px = min(PAYLOAD_PX_MAX, int(PAYLOAD_DENSITY_HI * ph * pw))
        min_px = max(PAYLOAD_PX_MIN, int(np.ceil(PAYLOAD_DENSITY_LO * ph * pw)))
        if min_px > max_px:
            continue
        pmask = _random_connected_mask(ph, pw, rng.randint(min_px, max_px), rng=rng)
        if pmask is None:
            continue
        if np.array_equal(pmask, pmask[::-1]) and np.array_equal(pmask, pmask[:, ::-1]):
            continue

        payload_coords = [(int(r), int(c)) for r, c in np.argwhere(pmask)]
        marker_subsets = list(itertools.combinations(_MARKER_KINDS, marker_count))
        rng.shuffle(marker_subsets)

        for marker_kinds in marker_subsets:
            if require_line and not any(kind in {"top", "bottom", "left", "right"} for kind in marker_kinds):
                continue
            if any(not _point_marker_supported(kind, pmask) for kind in marker_kinds):
                continue
            if not _markers_stay_separate(marker_kinds, ph, pw):
                continue

            orig = set(payload_coords)
            reds = set()
            mirrors = set()
            ok = True
            for kind in marker_kinds:
                cells = set(_marker_cells(kind, ph, pw))
                mirr = set(_mirror_coords_for_marker(kind, payload_coords, ph, pw))
                if orig & cells or reds & cells or orig & mirr or reds & mirr or mirrors & mirr:
                    ok = False
                    break
                reds |= cells
                mirrors |= mirr
            if not ok:
                continue

            all_pts = list(orig | reds | mirrors)
            minr = min(r for r, _ in all_pts)
            minc = min(c for _, c in all_pts)
            maxr = max(r for r, _ in all_pts)
            maxc = max(c for _, c in all_pts)
            h = maxr - minr + 1
            w = maxc - minc + 1
            if h > 15 or w > 15:
                continue

            payload = np.zeros((h, w), dtype=bool)
            mirror = np.zeros((h, w), dtype=bool)
            red_pixels: list[tuple[int, int]] = []
            for r, c in orig:
                payload[r - minr, c - minc] = True
            for r, c in mirrors:
                mirror[r - minr, c - minc] = True
            for r, c in sorted(reds):
                red_pixels.append((int(r - minr), int(c - minc)))
            return {
                "payload": payload,
                "mirror": mirror,
                "red_pixels": red_pixels,
                "marker_count": int(marker_count),
                "marker_kinds": tuple(marker_kinds),
            }
    return None


def _random_free_location(
    canvas: np.ndarray,
    patch_mask: np.ndarray,
    bg: int,
    rng: random.Random,
    *,
    padding: int = PLACEMENT_PADDING,
):
    h, w = patch_mask.shape
    H, W = canvas.shape
    if h > H or w > W:
        return None
    occ = canvas != int(bg)
    if padding > 0:
        occ = ndimage.binary_dilation(occ, structure=STRUCT8, iterations=padding)
    cands = []
    for r0 in range(H - h + 1):
        for c0 in range(W - w + 1):
            if not (occ[r0 : r0 + h, c0 : c0 + w] & patch_mask).any():
                cands.append((r0, c0))
    if not cands:
        return None
    return rng.choice(cands)


def _stamp_sprite(canvas: np.ndarray, spr: dict, r0: int, c0: int) -> None:
    h, w = spr["payload"].shape
    colored = spr["payload"] | spr["mirror"]
    canvas[r0 : r0 + h, c0 : c0 + w][colored] = int(spr["color"])
    for rr, cc in spr["red_pixels"]:
        canvas[r0 + rr, c0 + cc] = int(MARK)


def _erase_mirror(canvas: np.ndarray, spr: dict, bg: int) -> None:
    r0, c0 = int(spr["r0"]), int(spr["c0"])
    h, w = spr["mirror"].shape
    canvas[r0 : r0 + h, c0 : c0 + w][spr["mirror"]] = int(bg)


def _marker_count_plan(bundle_count: int, rng: random.Random) -> list[int] | None:
    counts = [1] * bundle_count
    multi_indices = [rng.randrange(bundle_count)]
    if bundle_count >= 4 and rng.random() < 0.5:
        extra_idx = rng.choice([i for i in range(bundle_count) if i not in multi_indices])
        multi_indices.append(extra_idx)
    for idx in multi_indices:
        counts[idx] += 1

    target_total = rng.randint(max(MARKERS_TOTAL_MIN, sum(counts)), min(MARKERS_TOTAL_MAX, bundle_count * MARKERS_PER_BUNDLE_MAX))
    while sum(counts) < target_total:
        cands = [i for i in range(bundle_count) if counts[i] < MARKERS_PER_BUNDLE_MAX]
        if not cands:
            break
        counts[rng.choice(cands)] += 1
    if sum(counts) < MARKERS_TOTAL_MIN or sum(counts) > MARKERS_TOTAL_MAX:
        return None
    if max(counts) < 2:
        return None
    return counts


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    N = rng.randint(GRID_MIN, GRID_MAX)
    bg = rng.choice([c for c in range(10) if c != MARK])
    palette = [c for c in range(10) if c != MARK and c != bg]

    bundle_count = rng.randint(BUNDLES_MIN, BUNDLES_MAX)
    counts = _marker_count_plan(bundle_count, rng)
    if counts is None:
        return None

    chosen_colors = rng.sample(palette, bundle_count)

    sprites: list[dict] = []
    for marker_count in counts:
        spr = _build_bundle(marker_count, require_line=marker_count >= 2, rng=rng)
        if spr is None:
            return None
        sprites.append(spr)

    rng.shuffle(chosen_colors)
    for spr, color in zip(sprites, chosen_colors):
        spr["color"] = int(color)

    output = np.full((N, N), bg, dtype=int)
    rng.shuffle(sprites)
    for spr in sprites:
        footprint = spr["payload"] | spr["mirror"]
        for r, c in spr["red_pixels"]:
            footprint[r, c] = True
        loc = _random_free_location(output, footprint, bg, rng)
        if loc is None:
            return None
        r0, c0 = loc
        spr["r0"], spr["c0"] = int(r0), int(c0)
        _stamp_sprite(output, spr, r0, c0)

    inp = output.copy()
    for spr in sprites:
        _erase_mirror(inp, spr, bg)

    if (inp == int(bg)).sum() / float(N * N) < BG_COVERAGE_MIN:
        return None

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
