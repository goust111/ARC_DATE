from __future__ import annotations

# Variant of 7ed72f31
# Dimension: D9 — 原始像素的保留与删除 (transform)
# Change: 反射后原始payload像素和红色标记都保留在输出中... → 删除原始payload，仅保留镜像
#
# Design analysis
# ===============
# variant_spec:
#   core_change:
#     original_behavior: 对每个红色标记组，把 payload 反射到另一侧，同时保留输入中的原 payload 和红色标记。
#     new_behavior: 对每个红色标记组，把 payload 反射到另一侧后，擦除原 payload；红色标记保留不动。
#   concrete_grid_effect:
#     row_axis:
#       input: 单色 payload 全在红色水平线的一侧。
#       output: 水平线保留，payload 只出现在另一侧；原侧被擦成背景。
#     col_axis:
#       input: 单色 payload 全在红色垂直线的一侧。
#       output: 垂直线保留，payload 只出现在另一侧；原侧被擦成背景。
#     isolated_point:
#       input: 单个红点与一个 8 连通彩色组件对角相邻。
#       output: 红点保留，整个彩色组件只出现在中心对称后的对侧；原组件被擦除。
#   behavior_trace:
#     example:
#       input: "红线下方有 3 格绿色 L 形。"
#       original_solver: "在红线上方画出绿色 L 形，但下方原 L 形仍存在。"
#       new_solver: "先确定这 3 个绿色源像素，再把它们逐个映射到红线上方，最后把下方这 3 个源像素擦掉。"
#       delta: "copy 语义变成 move 语义；视觉上对象跳过红轴。"
#   complexity_calibration:
#     keep: 保持原题的三类标记几何（水平线、垂直线、孤立红点）和单色 payload。
#     reduce: 生成器不再放置原 spec 中可选的、会被 bbox 误捕获的小散点。
#     reason: deletion 语义下，如果 solver 仍按宽松 bbox 抓取同色散点，会把无关像素也删除，导致题意被噪声主导。
#   implicit_assumption_review:
#     original_main_risk_1:
#       code_pattern: "out = g.copy(); 反射时只写镜像，不处理源像素。"
#       old_assumption: 保留源像素是正确的。
#       new_status: 失效，必须显式擦除源 payload。
#     original_main_risk_2:
#       code_pattern: "对 row/col 组按整张网格扫描同色像素，并用 bbox 判 eligible。"
#       old_assumption: 即使多抓到附近同色散点，也只是额外复制，通常无害。
#       new_status: 失效；deletion 下多抓到的同色像素会被误删，因此改为只处理该 marker 组件内的 payload。
#     original_main_risk_3:
#       code_pattern: "边反射边写 out。"
#       old_assumption: 新写入的镜像不会影响旧像素是否保留。
#       new_status: deletion 下需要分离 source collection / erase / paint，避免写入顺序造成自我覆盖。
#     isolated_case_note:
#       assumption: 孤立红点周围恰有一个相邻 8 连通彩色组件。
#       generator_guarantee: 生成器通过 1 格缓冲带隔离各组，并让孤立红点只与一个组件对角接触。
#   placement_legality:
#     strategy: 先为每组生成局部模板（原 payload、红标记、镜像 payload 的并集），再把整个并集连同 1 格缓冲带放到大网格中。
#     guarantees:
#       - 原 payload、红标记、镜像 payload 都不越界
#       - 不同组在输入和输出里都不会重叠
#       - 孤立红点的 8 邻域不会碰到其他组
#   visual_observability:
#     cues:
#       - 每张图至少含 row / col / isolated 三类中的多个实例，且通常覆盖三类。
#       - payload 形状用小型不规则连通块，避免“镜像前后看不出来”。
#       - 输出中原侧清空、对侧出现副本，和原题的“双边同时存在”形成稳定可见差异。
#   cleanliness:
#     layout: 组与组之间保留缓冲，不叠层，不共享颜色。
#   color_inference:
#     rule: 颜色本身不变，只改变位置与是否保留原像素，因此无需引入额外颜色映射。
#   new_main_assumptions:
#     - marker-containing 组件内的非红像素就是唯一应被移动的 payload。
#     - payload 的镜像位置落在背景上，不与其他组冲突。
#     - 孤立红点只对应其 8 邻域触发到的那一个组件。

import random
import sys
from pathlib import Path

import numpy as np

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import get_bg, label_mask


Coord = tuple[int, int]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect payloads using red markers, but delete the original payload afterward.

    Kept from the source task:
    - Horizontal red line => reflect across that row.
    - Vertical red line => reflect across that column.
    - Single red point inside a mixed component => point reflection of that component payload.
    - Isolated red point => point reflection of the adjacent 8-connected color component.

    D9 change:
    - Red markers stay.
    - Source payload pixels are erased.
    - Only the reflected copy remains in the output.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    mark = 2
    h, w = g.shape

    def inb(r: int, c: int) -> bool:
        return 0 <= int(r) < h and 0 <= int(c) < w

    def refl_point(center: Coord, r: int, c: int) -> Coord:
        cr, cc = center
        return int(2 * cr - r), int(2 * cc - c)

    def refl_row(r0: int, r: int, c: int) -> Coord:
        return int(2 * r0 - r), int(c)

    def refl_col(c0: int, r: int, c: int) -> Coord:
        return int(r), int(2 * c0 - c)

    out = g.copy()
    erase: set[Coord] = set()
    paint: list[tuple[int, int, int]] = []

    # 4-connected components among all non-background pixels.
    lab, n = label_mask(g != bg, connectivity=4)
    comps: list[dict[str, object]] = []
    for k in range(1, int(n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab == int(k))]
        if not coords:
            continue
        vals = [int(g[r, c]) for r, c in coords]
        comps.append({"coords": coords, "vals": vals})

    # 1) Marker-bearing mixed components: move only the payload inside that component.
    for comp in comps:
        coords = comp["coords"]
        vals = comp["vals"]
        reds = [(int(r), int(c)) for (r, c), v in zip(coords, vals) if int(v) == mark]
        if not reds:
            continue

        payload = [(int(r), int(c), int(v)) for (r, c), v in zip(coords, vals) if int(v) != mark]
        if not payload:
            continue

        rows = {int(r) for r, _ in reds}
        cols = {int(c) for _, c in reds}
        if len(reds) == 1:
            axis = ("point", (int(reds[0][0]), int(reds[0][1])))
        elif len(rows) == 1:
            axis = ("row", int(next(iter(rows))))
        elif len(cols) == 1:
            axis = ("col", int(next(iter(cols))))
        else:
            continue

        for r, c, v in payload:
            if axis[0] == "row":
                rr, cc = refl_row(int(axis[1]), int(r), int(c))
            elif axis[0] == "col":
                rr, cc = refl_col(int(axis[1]), int(r), int(c))
            else:
                rr, cc = refl_point(axis[1], int(r), int(c))

            erase.add((int(r), int(c)))
            if inb(rr, cc):
                paint.append((int(rr), int(cc), int(v)))

    # 2) Isolated red pixels: move the unique adjacent 8-connected component.
    isolated_markers: list[Coord] = []
    for comp in comps:
        coords = comp["coords"]
        vals = comp["vals"]
        if len(coords) == 1 and int(vals[0]) == mark:
            isolated_markers.append((int(coords[0][0]), int(coords[0][1])))

    for pr, pc in isolated_markers:
        neighbor_pixels: list[tuple[int, int, int]] = []
        neighbor_colors: set[int] = set()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = int(pr + dr), int(pc + dc)
                if not inb(rr, cc):
                    continue
                v = int(g[rr, cc])
                if v == bg or v == mark:
                    continue
                neighbor_pixels.append((int(rr), int(cc), int(v)))
                neighbor_colors.add(int(v))

        for color in neighbor_colors:
            labv, _ = label_mask(g == int(color), connectivity=8)
            labels = {
                int(labv[rr, cc])
                for rr, cc, vv in neighbor_pixels
                if int(vv) == int(color) and int(labv[rr, cc]) != 0
            }
            for lbl in labels:
                for r, c in np.argwhere(labv == int(lbl)):
                    src = (int(r), int(c))
                    dst = refl_point((int(pr), int(pc)), int(r), int(c))
                    erase.add(src)
                    if inb(*dst):
                        paint.append((int(dst[0]), int(dst[1]), int(color)))

    for r, c in erase:
        if int(out[r, c]) != mark:
            out[r, c] = bg
    for r, c, v in paint:
        out[int(r), int(c)] = int(v)
    return out


def _grow_connected(
    height: int,
    width: int,
    size: int,
    *,
    connectivity: int,
    seed: Coord,
    rng: random.Random,
) -> set[Coord]:
    """Random connected set inside a box using growth from a required seed."""
    if not (0 <= seed[0] < height and 0 <= seed[1] < width):
        raise ValueError("seed outside box")
    if size < 1 or size > height * width:
        raise ValueError("invalid size")

    if connectivity == 4:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        nbrs = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]

    cells = {seed}
    frontier: set[Coord] = set()

    def push_neighbors(r: int, c: int) -> None:
        for dr, dc in nbrs:
            rr, cc = int(r + dr), int(c + dc)
            if 0 <= rr < height and 0 <= cc < width and (rr, cc) not in cells:
                frontier.add((rr, cc))

    push_neighbors(*seed)
    while len(cells) < size:
        if not frontier:
            raise ValueError("growth got stuck")
        nxt = rng.choice(tuple(frontier))
        frontier.remove(nxt)
        cells.add(nxt)
        push_neighbors(*nxt)
    return cells


def _make_row_group(color: int, rng: random.Random) -> dict[str, object]:
    red_len = rng.randint(2, 6)
    payload_h = rng.randint(2, 4)
    max_area = payload_h * red_len
    size = rng.randint(3, min(11, max_area))
    side = rng.choice(["above", "below"])
    seed = (payload_h - 1, rng.randrange(red_len)) if side == "above" else (0, rng.randrange(red_len))
    box = _grow_connected(payload_h, red_len, size, connectivity=4, seed=seed, rng=rng)

    axis_r = payload_h
    reds = [(axis_r, c) for c in range(red_len)]
    if side == "above":
        src = [(r, c, color) for r, c in box]
        dst = [(2 * axis_r - r, c, color) for r, c in box]
    else:
        src = [(r + payload_h + 1, c, color) for r, c in box]
        dst = [(2 * axis_r - (r + payload_h + 1), c, color) for r, c in box]

    height = payload_h * 2 + 1
    width = red_len
    return {"height": height, "width": width, "reds": reds, "src": src, "dst": dst, "type": "row"}


def _make_col_group(color: int, rng: random.Random) -> dict[str, object]:
    red_len = rng.randint(2, 6)
    payload_w = rng.randint(2, 4)
    max_area = red_len * payload_w
    size = rng.randint(3, min(11, max_area))
    side = rng.choice(["left", "right"])
    seed = (rng.randrange(red_len), payload_w - 1) if side == "left" else (rng.randrange(red_len), 0)
    box = _grow_connected(red_len, payload_w, size, connectivity=4, seed=seed, rng=rng)

    axis_c = payload_w
    reds = [(r, axis_c) for r in range(red_len)]
    if side == "left":
        src = [(r, c, color) for r, c in box]
        dst = [(r, 2 * axis_c - c, color) for r, c in box]
    else:
        src = [(r, c + payload_w + 1, color) for r, c in box]
        dst = [(r, 2 * axis_c - (c + payload_w + 1), color) for r, c in box]

    height = red_len
    width = payload_w * 2 + 1
    return {"height": height, "width": width, "reds": reds, "src": src, "dst": dst, "type": "col"}


def _make_isolated_group(color: int, rng: random.Random) -> dict[str, object]:
    quad_h = rng.randint(2, 3)
    quad_w = rng.randint(2, 3)
    size = rng.randint(1, min(8, quad_h * quad_w))
    box = _grow_connected(quad_h, quad_w, size, connectivity=8, seed=(0, 0), rng=rng)
    sr = rng.choice([-1, 1])
    sc = rng.choice([-1, 1])

    marker = (quad_h, quad_w)
    reds = [marker]
    src = []
    dst = []
    for r, c in box:
        dr, dc = int(r + 1), int(c + 1)
        src.append((int(marker[0] + sr * dr), int(marker[1] + sc * dc), color))
        dst.append((int(marker[0] - sr * dr), int(marker[1] - sc * dc), color))

    height = quad_h * 2 + 1
    width = quad_w * 2 + 1
    return {"height": height, "width": width, "reds": reds, "src": src, "dst": dst, "type": "isolated"}


def _expand(coords: set[Coord], height: int, width: int, pad: int = 1) -> set[Coord]:
    out: set[Coord] = set()
    for r, c in coords:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                rr, cc = int(r + dr), int(c + dc)
                if 0 <= rr < height and 0 <= cc < width:
                    out.add((rr, cc))
    return out


def _place_group(
    inp: np.ndarray,
    out: np.ndarray,
    blocked: np.ndarray,
    group: dict[str, object],
    rng: random.Random,
) -> bool:
    h, w = inp.shape
    gh = int(group["height"])
    gw = int(group["width"])
    reds = [(int(r), int(c)) for r, c in group["reds"]]
    src = [(int(r), int(c), int(v)) for r, c, v in group["src"]]
    dst = [(int(r), int(c), int(v)) for r, c, v in group["dst"]]
    footprint = {(r, c) for r, c in reds} | {(r, c) for r, c, _ in src} | {(r, c) for r, c, _ in dst}

    for _ in range(200):
        r0 = rng.randint(0, h - gh)
        c0 = rng.randint(0, w - gw)
        placed = {(r0 + r, c0 + c) for r, c in footprint}
        halo = _expand(placed, h, w, pad=1)
        if any(blocked[r, c] for r, c in halo):
            continue

        for r, c in halo:
            blocked[r, c] = True
        for r, c in reds:
            inp[r0 + r, c0 + c] = 2
            out[r0 + r, c0 + c] = 2
        for r, c, v in src:
            inp[r0 + r, c0 + c] = int(v)
        for r, c, v in dst:
            out[r0 + r, c0 + c] = int(v)
        return True
    return False


def _generate_once(rng: random.Random) -> np.ndarray:
    h = rng.randint(18, 28)
    w = rng.randint(18, 28)
    bg = rng.choice([c for c in range(10) if c != 2])
    inp = np.full((h, w), bg, dtype=int)
    expected = np.full((h, w), bg, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    n_groups = rng.randint(3, 6)
    base_types = ["row", "col", "isolated"]
    types = base_types[: min(n_groups, len(base_types))]
    while len(types) < n_groups:
        types.append(rng.choice(base_types))
    rng.shuffle(types)

    palette = [c for c in range(10) if c not in (bg, 2)]
    colors = rng.sample(palette, len(types))

    builders = {"row": _make_row_group, "col": _make_col_group, "isolated": _make_isolated_group}
    for typ, color in zip(types, colors):
        group = builders[typ](int(color), rng)
        if not _place_group(inp, expected, blocked, group, rng):
            raise ValueError("failed to place group")

    solved = main(inp)
    if not np.array_equal(solved, expected):
        raise AssertionError("oracle mismatch")
    return inp


def generate_input() -> np.ndarray:
    """Generate a valid input grid for the D9 variant."""
    rng = random.Random()
    last_error: Exception | None = None
    for _ in range(200):
        try:
            return _generate_once(rng)
        except Exception as exc:  # pragma: no cover - retry logic
            last_error = exc
    raise RuntimeError(f"generate_input failed after retries: {last_error}")
