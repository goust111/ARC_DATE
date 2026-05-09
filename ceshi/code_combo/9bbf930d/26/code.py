# Combo variant of 9bbf930d (index 26)
# combo_spec: D2/1+D4/3+D5/3
# Name: 三重颜色语义：粒子匹配走廊+障碍色转向+走廊色轨迹

from __future__ import annotations

import random

import numpy as np

"""
Design analysis for the combo variant
=====================================

a) 具体化与行为追踪
   - 网格仍是原题的条纹布局：偶数行是彩色条，奇数行是水平通道，竖直转接靠
     `c_v-1` 这一列上的缺口完成。
   - 融合后的三层颜色语义分工如下：
       1. 源颜色：col 0 的粒子不再统一为 6，而是使用若干“走廊色”。它只决定
          该行粒子能否进入最左侧第一条水平走廊；只有粒子色 == 该行走廊色时，
          这一行才会启动。
       2. 障碍颜色：一旦粒子已经进入迷宫，后续所有拐弯都只看正前方挡块颜色。
          颜色 2 表示逆时针转 90 度，颜色 9 表示顺时针转 90 度。
       3. 轨迹颜色：输出中不只保留终点；粒子走过的每个通行格都会被重涂为
          “当前所在走廊”的颜色，因此跨到另一条水平走廊后，轨迹颜色也会切换。
   - 一个典型小例子：
       * 某行源颜色是红色，左边第一条真实水平走廊也是红色，因此该粒子可以启动。
       * 它向右走到一个颜色 2 的挡块，虽然上/下两边都被几何上打开，但颜色 2
         强制它从右转上。
       * 它沿竖直通道上行，遇到颜色 9 的挡块，再被强制从上转右，进入一条蓝色
         水平走廊。
       * 输出时，前半段轨迹是红色，后半段轨迹变成蓝色，终点再写回原来的红色粒子。
   - 与原题差异：
       * 原题入口不看粒子色；现在入口先受源颜色约束。
       * 原题转弯看“哪一侧唯一可走”；现在故意把两侧都打开，再由 2/9 决定方向。
       * 原题没有轨迹；现在输出直接暴露出“走过了哪些走廊颜色”。

b) 复杂度校准
   - 若把 D5 扩展成“每次转入新走廊都必须和粒子原色匹配”，那么 D4/3 的多色
     轨迹就会被直接消掉。因此这里把 D5 的限制严格放在它所属的 `src` 维度：
     只约束起始入口是否可进入，不约束后续走廊换色。
   - 转向提示色只保留两种：2/9。这样 2-4 个训练样例就能重复支撑映射，不会变成
     需要记很多颜色表的任务。
   - 真实路径里强制至少出现一条跨色 U-turn：同一粒子会先走一种走廊色，再被
     cue-color 拐进另一种走廊色，确保三项融合都可见。

c) 隐式假设审查
   - 原 `PARTICLE = 6` 的假设失效：col 0 的奇数行源现在是多色的，solver 不能再
     用常量 6 识别活动源。
   - 原 `_trace()` 的“碰到彩色格后检查两个垂直方向且至多一个合法”也失效：
     这里故意制造双分支，若继续沿用旧逻辑会停在拐点。
   - 原 solver 不需要保存中间状态；D4/3 后必须把“当前轨迹色”与“完整 trail”
     从运动逻辑里显式带出来，而且绘制必须写到输入副本 `out` 上，不能边走边改
     读图。
   - 原入口过滤只要求某行存在同色条对；现在还要附加“源颜色 == 第一条真实走廊色”。
   - 新冲突检查：若让入口匹配和后续转入走廊匹配都同时依赖源颜色，就无法出现
     多色轨迹；因此 generator 与 solver 一致地只把颜色匹配放在入口阶段。

d) 放置合法性
   - 竖直通道仍然通过预留 3 列窗口分配，避免多个转弯结构互相覆盖。
   - U-turn 的目标行入口会被打坏，保证该目标行自身不会再从 col 0 独立出发并与
     上层路径重叠。
   - 源颜色只写在 col 0，col 1 固定为背景，不会破坏条纹几何。
   - 所有 builder 都只在既有的 bar/channel 模板上开缺口或放 cue 块，不会越界。

e) 视觉可观察性
   - 人类可以从输入直接读出：左侧源颜色不同，只有与该行第一条走廊同色的粒子会动。
   - 路线上每次转弯点前面都放 2 或 9，且训练样例里都要求至少出现一条真正被访问到
     的双分支 U-turn，因此颜色到方向的映射可以从输出终点和轨迹走向反推出来。
   - 输出轨迹的颜色切换发生在转入新水平走廊时，旁边正好有对应颜色的条对作为结构线索，
     不需要任何数值推理。

f) 可视化整洁度
   - 颜色语义按空间分层：col 0 看源颜色、拐点前一格看 cue 颜色、路径本身看走廊色。
     三者不在同一像素上竞争。
   - 轨迹只复用已有走廊色，不额外引入第三套标记色，因此网格依然是“条纹 + 少量提示色”
     的整洁外观。

g) 变化的可推断性
   - cue 映射固定为 2=逆时针、9=顺时针，不在样例间重排。
   - 源颜色直接取自走廊调色板，而不是额外的隐藏标签色；“能否出发”可以靠肉眼比较
     左边源色与该行第一条真实走廊色是否一致来判断。
   - 轨迹颜色变化有明确结构伴随：每次换色都对应进入了另一条由同色条对包围的水平走廊。

h) 量化退化检查
   - `mismatched_candidates`：几何上可进入但因源颜色不匹配而静止的候选行数，用来验证
     D5 真的在起作用。
   - `ambiguous_events` 与 `legacy_differences`：活动路径上有多少次“两边都能转但只有
     cue-color 才能决定方向”，以及与旧的“唯一合法转弯”规则相比有多少条路径终点不同，
     用来验证 D2 真的在起作用。
   - `multicolor_paths`、`color_transitions`、`trail_cells`：用来验证 D4 不是退化成单色
     涂抹或无轨迹。

i) 新 solver 假设与 generator 保证
   - 假设：活动源只出现在奇数行，且 col 1 始终为空。generator 固定这样构造。
   - 假设：真正被访问到的挡块颜色只会是 2 或 9；否则粒子会停下。generator 只在真实
     转弯点放 2/9，并避免活动路径撞上普通条色。
   - 假设：轨迹换色发生在进入新的真实水平走廊时。generator 至少放置一条跨色 U-turn，
     保证这个现象在训练样例里反复出现。
"""

BG = 7
TURN_CCW = 2
TURN_CW = 9
_DELTAS = {"right": (0, 1), "left": (0, -1), "up": (-1, 0), "down": (1, 0)}
_PERPENDICULAR = {
    "right": ["up", "down"],
    "left": ["up", "down"],
    "up": ["left", "right"],
    "down": ["left", "right"],
}
_CCW_TURN = {"right": "up", "up": "left", "left": "down", "down": "right"}
_CW_TURN = {"right": "down", "down": "left", "left": "up", "up": "right"}
_TURN_CLASS = {TURN_CCW: "ccw", TURN_CW: "cw"}
_PALETTE_POOL = [0, 1, 3, 4, 5, 8]


def _corridor_color(grid: np.ndarray, r: int, c: int, direction: str) -> int | None:
    """Return the corridor-pair color visible at (r, c), else None."""
    H, W = grid.shape
    if not (0 <= r < H and 0 <= c < W):
        return None
    if int(grid[r, c]) != BG:
        return None
    if direction in {"left", "right"}:
        if r == 0 or r == H - 1:
            return None
        a, b = int(grid[r - 1, c]), int(grid[r + 1, c])
    else:
        if c == 0 or c == W - 1:
            return None
        a, b = int(grid[r, c - 1]), int(grid[r, c + 1])
    if a == b and a not in {BG, TURN_CCW, TURN_CW}:
        return a
    return None


def _turn_from_blocker(color: int, direction: str) -> str | None:
    """Map cue color 2/9 to a relative turn."""
    cls = _TURN_CLASS.get(color)
    if cls == "ccw":
        return _CCW_TURN[direction]
    if cls == "cw":
        return _CW_TURN[direction]
    return None


def _cue_color(from_dir: str, to_dir: str) -> int:
    if _CCW_TURN[from_dir] == to_dir:
        return TURN_CCW
    if _CW_TURN[from_dir] == to_dir:
        return TURN_CW
    raise ValueError(f"invalid turn {from_dir}->{to_dir}")


def _find_entrance_col(grid: np.ndarray, r: int, particle_color: int) -> int | None:
    """First horizontal corridor on row r that matches the source color."""
    _, W = grid.shape
    for c in range(2, W):
        color = _corridor_color(grid, r, c, "right")
        if color != particle_color:
            continue
        bar_above = next((cc for cc in range(W) if int(grid[r - 1, cc]) == color), W)
        bar_below = next((cc for cc in range(W) if int(grid[r + 1, cc]) == color), W)
        if bar_above == bar_below and bar_above <= 3:
            return c
    return None


def _first_real_corridor_color(grid: np.ndarray, r: int) -> int | None:
    """Geometry-only version of the entrance scan, ignoring the source color."""
    _, W = grid.shape
    for c in range(2, W):
        color = _corridor_color(grid, r, c, "right")
        if color is None:
            continue
        bar_above = next((cc for cc in range(W) if int(grid[r - 1, cc]) == color), W)
        bar_below = next((cc for cc in range(W) if int(grid[r + 1, cc]) == color), W)
        if bar_above == bar_below and bar_above <= 3:
            return color
    return None


def _trace_path(
    grid: np.ndarray, start_r: int, start_c: int
) -> tuple[int, int, list[tuple[int, int, int]], list[dict[str, object]]]:
    """Trace one admitted particle and record both its trail and cue events."""
    H, W = grid.shape
    particle_color = int(grid[start_r, start_c])
    r, c = start_r, start_c
    direction = "right"
    active_color: int | None = None
    trail: list[tuple[int, int, int]] = []
    events: list[dict[str, object]] = []

    for _ in range(700):
        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= H or nc < 0 or nc >= W:
            break
        nxt = int(grid[nr, nc])
        if nxt == BG:
            step_color = _corridor_color(grid, nr, nc, direction)
            if step_color is not None:
                active_color = step_color
            if active_color is not None:
                trail.append((nr, nc, active_color))
            r, c = nr, nc
            continue

        valid_dirs = []
        valid_colors = {}
        for turn_dir in _PERPENDICULAR[direction]:
            tdr, tdc = _DELTAS[turn_dir]
            tr, tc = r + tdr, c + tdc
            turn_color = _corridor_color(grid, tr, tc, turn_dir)
            if turn_color is not None:
                valid_dirs.append(turn_dir)
                valid_colors[turn_dir] = turn_color
        turn_dir = _turn_from_blocker(nxt, direction)
        turn_color = valid_colors.get(turn_dir) if turn_dir is not None else None
        events.append(
            {
                "from_dir": direction,
                "blocker_color": nxt,
                "valid_dirs": tuple(valid_dirs),
                "chosen": turn_dir,
                "turn_color": turn_color,
                "particle_color": particle_color,
            }
        )
        if turn_dir is None or turn_color is None:
            break
        tdr, tdc = _DELTAS[turn_dir]
        r, c = r + tdr, c + tdc
        direction = turn_dir
        active_color = turn_color
        trail.append((r, c, active_color))

    return r, c, trail, events


def _legacy_trace(grid: np.ndarray, start_r: int, start_c: int) -> tuple[int, int]:
    """Ablation: keep source matching, but remove cue-colored tie-breaking."""
    H, W = grid.shape
    r, c = start_r, start_c
    direction = "right"
    for _ in range(700):
        dr, dc = _DELTAS[direction]
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= H or nc < 0 or nc >= W:
            break
        if int(grid[nr, nc]) == BG:
            r, c = nr, nc
            continue
        options = []
        for turn_dir in _PERPENDICULAR[direction]:
            tdr, tdc = _DELTAS[turn_dir]
            tr, tc = r + tdr, c + tdc
            if _corridor_color(grid, tr, tc, turn_dir) is not None:
                options.append((tr, tc, turn_dir))
        if len(options) != 1:
            break
        r, c, direction = options[0]
    return r, c


def _collect_paths(grid: np.ndarray) -> list[dict[str, object]]:
    """Collect all solver-visible source paths and trail statistics."""
    g = np.array(grid, dtype=int)
    H, _ = g.shape
    paths = []
    for r in range(1, H - 1, 2):
        particle_color = int(g[r, 0])
        if particle_color == BG:
            continue
        entrance_col = _find_entrance_col(g, r, particle_color)
        if entrance_col is None:
            paths.append(
                {
                    "row": r,
                    "start": (r, 0),
                    "end": (r, 0),
                    "trail": [],
                    "events": [],
                    "moved": False,
                    "turns": 0,
                    "distinct_colors": 0,
                    "color_transitions": 0,
                    "particle_color": particle_color,
                    "legacy_end": (r, 0),
                }
            )
            continue
        end_r, end_c, trail, events = _trace_path(g, r, 0)
        colors = [color for _, _, color in trail]
        paths.append(
            {
                "row": r,
                "start": (r, 0),
                "end": (end_r, end_c),
                "trail": trail,
                "events": events,
                "moved": (end_r, end_c) != (r, 0),
                "turns": len(events),
                "distinct_colors": len(set(colors)),
                "color_transitions": sum(
                    1 for i in range(1, len(colors)) if colors[i] != colors[i - 1]
                ),
                "particle_color": particle_color,
                "legacy_end": _legacy_trace(g, r, 0),
            }
        )
    return paths


def _candidate_source_rows(grid: np.ndarray) -> list[tuple[int, int]]:
    """Rows whose geometry exposes a real left entrance, ignoring source colors."""
    H, _ = grid.shape
    rows = []
    for r in range(1, H - 1, 2):
        color = _first_real_corridor_color(grid, r)
        if color is not None:
            rows.append((r, color))
    return rows


def _fusion_metrics(grid: np.ndarray) -> dict[str, int]:
    """Quantify whether the three fused behaviors are actually active."""
    paths = _collect_paths(grid)
    candidates = _candidate_source_rows(grid)
    candidate_color = {r: color for r, color in candidates}
    moved_paths = [p for p in paths if p["moved"]]
    cue_counts = {TURN_CCW: 0, TURN_CW: 0}
    ambiguous_events = 0
    vertical_turns = 0
    legacy_differences = 0
    for path in moved_paths:
        if path["end"] != path["legacy_end"]:
            legacy_differences += 1
        for ev in path["events"]:
            blocker = int(ev["blocker_color"])
            if blocker in cue_counts:
                cue_counts[blocker] += 1
            if len(ev["valid_dirs"]) == 2:
                ambiguous_events += 1
            if ev["from_dir"] in {"up", "down"} and ev["chosen"] is not None:
                vertical_turns += 1
    mismatched_candidates = sum(
        1
        for r, color in candidates
        if int(grid[r, 0]) != color
    )
    return {
        "candidate_rows": len(candidates),
        "moved": len(moved_paths),
        "turning_paths": sum(1 for p in moved_paths if p["turns"] > 0),
        "multicolor_paths": sum(1 for p in moved_paths if p["distinct_colors"] > 1),
        "trail_cells": sum(len(p["trail"]) for p in moved_paths),
        "color_transitions": sum(int(p["color_transitions"]) for p in moved_paths),
        "cue_ccw": cue_counts[TURN_CCW],
        "cue_cw": cue_counts[TURN_CW],
        "ambiguous_events": ambiguous_events,
        "vertical_turns": vertical_turns,
        "legacy_differences": legacy_differences,
        "mismatched_candidates": mismatched_candidates,
        "matched_candidate_colors": len(
            {candidate_color[p["row"]] for p in moved_paths if p["row"] in candidate_color}
        ),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Move source-colored particles through cue-directed corridors and paint corridor trails.

    Input structure:
    - bg=7. Odd rows in col 0 contain source particles colored from the corridor palette.
    - Even rows are colored bars. Real horizontal corridors are BG cells between same-color bars.
    - Cue colors 2 and 9 appear only as blockers that encode relative turn direction.

    Rule:
    - A source may start only if the first real horizontal corridor on its row has the same color.
    - After entering, the particle always goes straight through BG. When blocked by 2/9, it turns
      90 degrees according to the cue color, provided that the chosen side exposes a corridor cell.
    - The output paints every visited travel cell with the current corridor color; the final cell is
      overwritten back to the particle's original source color.

    Key insight:
    - Color has three independent semantics: source color gates admission, blocker color chooses turn,
      and corridor color paints the trail, so the output trail records the sequence of corridor segments.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()

    for path in _collect_paths(g):
        if not path["moved"]:
            continue
        start_r, start_c = path["start"]
        end_r, end_c = path["end"]
        particle_color = int(path["particle_color"])
        out[start_r, start_c] = BG
        for tr, tc, color in path["trail"]:
            out[tr, tc] = color
        out[end_r, end_c] = particle_color

    return out


# --------------- generator ---------------

_HEIGHT_OPTIONS = [17, 19, 21, 23]
_WIDTH_RANGE = (14, 21)
_BAR_START_OPTIONS = [2, 3]
_N_EXTRA_COLORS = 4
_N_REAL_OPTIONS = [4, 5, 6]
_N_REAL_WEIGHTS = [4, 3, 2]
_MIN_PALETTE_DIVERSITY = 3
_STRAIGHT_PROB = 0.15
_L_INTERNAL_PROB = 0.35
_U_INTERNAL_STOP_PROB = 0.65
_VCOL_LEFT_PAD = 4
_VCOL_RIGHT_PAD = 2
_VCOL_WINDOW = (-1, 0, 1)
_FAKE_DECOR_RANGE = (0, 2)
_MIN_MOVED = 3
_MAX_MOVED = 6
_MIN_DISTINCT_COLORS = 6
_MIN_MULTICOLOR_PATHS = 1
_MIN_COLOR_TRANSITIONS = 1
_MIN_TRAIL_CELLS = 18
_MIN_MISMATCHED_CANDIDATES = 1
_MAX_RETRIES = 500


def _allocate_vcol(bar_start: int, W: int, used_vcols: set[int]) -> int | None:
    c_min = bar_start + _VCOL_LEFT_PAD
    c_max = W - _VCOL_RIGHT_PAD
    candidates = [
        c
        for c in range(c_min, c_max + 1)
        if all((c + d) not in used_vcols for d in _VCOL_WINDOW)
    ]
    if not candidates:
        return None
    c_v = random.choice(candidates)
    used_vcols.update({c_v + d for d in _VCOL_WINDOW})
    return c_v


def _other_palette_color(colors: list[int], forbidden: int) -> int | None:
    options = [c for c in colors if c != forbidden]
    return random.choice(options) if options else None


def _paint_base_grid(H: int, W: int, bar_start: int, channel_spec: list[str], palette: list[int]):
    g = np.full((H, W), BG, dtype=int)
    bar_rows = list(range(0, H, 2))
    bar_color = {bar_rows[0]: random.choice(palette)}
    for i, ctype in enumerate(channel_spec):
        top, bot = bar_rows[i], bar_rows[i + 1]
        if ctype == "real":
            bar_color[bot] = bar_color[top]
        else:
            alt = _other_palette_color(palette, bar_color[top])
            if alt is None:
                alt = bar_color[top]
            bar_color[bot] = alt
    for br in bar_rows:
        g[br, bar_start:W] = bar_color[br]
    return g, bar_rows, bar_color


def _open_source_split(g: np.ndarray, top: int, bot: int, c_v: int, bar_color: dict[int, int]) -> None:
    """Open both vertical choices at the first blocker so cue-color must decide."""
    g[top, c_v - 1] = BG
    g[bot, c_v - 1] = BG
    g[top, c_v - 2] = bar_color[top]
    g[top, c_v] = bar_color[top]
    g[bot, c_v - 2] = bar_color[bot]
    g[bot, c_v] = bar_color[bot]


def _build_l_turn(
    g: np.ndarray,
    H: int,
    top: int,
    bot: int,
    row: int,
    c_v: int,
    direction: str,
    bar_color: dict[int, int],
    stop_at: int | None,
) -> None:
    g[row, c_v] = _cue_color("right", direction)
    _open_source_split(g, top, bot, c_v, bar_color)
    if direction == "up":
        for r_br in range(0, top, 2):
            if stop_at is not None and r_br <= stop_at:
                continue
            g[r_br, c_v - 1] = BG
    else:
        for r_br in range(bot + 2, H, 2):
            if stop_at is not None and r_br >= stop_at:
                continue
            g[r_br, c_v - 1] = BG


def _build_u_turn(
    g: np.ndarray,
    W: int,
    top: int,
    bot: int,
    row: int,
    c_v: int,
    direction: str,
    reentry: str,
    j_top: int,
    j_bot: int,
    bar_color: dict[int, int],
    bar_start: int,
    palette: list[int],
    internal_stop: bool,
) -> None:
    """Build a color-cued U-turn from one real row into another real row."""
    g[row, c_v] = _cue_color("right", direction)
    _open_source_split(g, top, bot, c_v, bar_color)

    if direction == "up":
        for r_br in range(j_bot, top + 1, 2):
            g[r_br, c_v - 1] = BG
        j_row = j_top + 1
        block_r = j_top
    else:
        for r_br in range(bot, j_top + 1, 2):
            g[r_br, c_v - 1] = BG
        j_row = j_top + 1
        block_r = j_bot

    g[block_r, c_v - 1] = _cue_color(direction, reentry)
    g[j_row, c_v - 2] = BG
    g[j_row, c_v] = BG

    alt = _other_palette_color(palette, bar_color[j_top])
    if alt is not None:
        g[j_top, bar_start] = alt

    if internal_stop:
        if reentry == "left":
            lo, hi = bar_start + 2, c_v - 3
        else:
            lo, hi = c_v + 2, W - 1
        if lo <= hi:
            g[j_row, random.randint(lo, hi)] = bar_color[j_top]


def _plan_forced_u_turn(
    real_indices: list[int],
    bar_rows: list[int],
    bar_color: dict[int, int],
    bar_start: int,
    W: int,
    used_vcols: set[int],
) -> tuple[dict[int, dict[str, object]] | None, set[int]]:
    """Create one guaranteed cross-color U-turn that shows both cue colors."""
    pairs = []
    for src in real_indices:
        for dst in real_indices:
            if src == dst:
                continue
            if bar_color[bar_rows[src]] == bar_color[bar_rows[dst]]:
                continue
            pairs.append((src, dst))
    if not pairs:
        return None, set()
    random.shuffle(pairs)
    for src, dst in pairs:
        c_v = _allocate_vcol(bar_start, W, used_vcols)
        if c_v is None:
            break
        direction = "up" if dst < src else "down"
        plan = {
            src: {
                "kind": "u",
                "target": dst,
                "direction": direction,
                "reentry": "right",
                "c_v": c_v,
                "internal_stop": random.random() < _U_INTERNAL_STOP_PROB,
                "forced": True,
            },
            dst: {"kind": "u_target", "forced": False},
        }
        return plan, {src}
    return None, set()


def _plan_remaining(
    real_indices: list[int],
    bar_start: int,
    W: int,
    used_vcols: set[int],
    plans: dict[int, dict[str, object]],
) -> None:
    for i in real_indices:
        if i in plans:
            continue
        if random.random() < _STRAIGHT_PROB:
            plans[i] = {"kind": "straight"}
            continue
        c_v = _allocate_vcol(bar_start, W, used_vcols)
        if c_v is None:
            plans[i] = {"kind": "straight"}
            continue
        direction = random.choice(["up", "down"])
        kind = "l_internal" if random.random() < _L_INTERNAL_PROB else "l_edge"
        plans[i] = {"kind": kind, "direction": direction, "c_v": c_v}


def _execute_plans(
    g: np.ndarray,
    H: int,
    W: int,
    plans: dict[int, dict[str, object]],
    bar_rows: list[int],
    bar_color: dict[int, int],
    bar_start: int,
    palette: list[int],
) -> None:
    for i, plan in plans.items():
        if plan["kind"] in {"straight", "u_target"}:
            continue
        top, bot = bar_rows[i], bar_rows[i + 1]
        row = top + 1
        c_v = int(plan["c_v"])
        if plan["kind"] in {"l_edge", "l_internal"}:
            stop_at = None
            direction = str(plan["direction"])
            if plan["kind"] == "l_internal":
                if direction == "up":
                    cands = list(range(2, top, 2))
                else:
                    cands = list(range(bot + 2, H - 1, 2))
                stop_at = random.choice(cands) if cands else None
            _build_l_turn(g, H, top, bot, row, c_v, direction, bar_color, stop_at)
        elif plan["kind"] == "u":
            target = int(plan["target"])
            j_top, j_bot = bar_rows[target], bar_rows[target + 1]
            _build_u_turn(
                g,
                W,
                top,
                bot,
                row,
                c_v,
                str(plan["direction"]),
                str(plan["reentry"]),
                j_top,
                j_bot,
                bar_color,
                bar_start,
                palette,
                bool(plan.get("internal_stop", False)),
            )


def _add_fake_decor(
    g: np.ndarray,
    channel_spec: list[str],
    bar_rows: list[int],
    bar_color: dict[int, int],
    bar_start: int,
) -> None:
    _, W = g.shape
    for i, ctype in enumerate(channel_spec):
        if ctype != "fake":
            continue
        top, bot = bar_rows[i], bar_rows[i + 1]
        r = top + 1
        n_decor = random.randint(*_FAKE_DECOR_RANGE)
        if n_decor == 0:
            continue
        cols = random.sample(range(bar_start + 1, W), min(n_decor, W - bar_start - 1))
        for c_decor in cols:
            if g[r, c_decor] == BG:
                g[r, c_decor] = random.choice([bar_color[top], bar_color[bot]])


def _assign_source_colors(
    g: np.ndarray,
    palette: list[int],
    required_match_rows: set[int],
) -> tuple[dict[int, int], dict[str, int]] | tuple[None, None]:
    candidates = _candidate_source_rows(g)
    candidate_rows = [r for r, _ in candidates]
    if len(candidate_rows) < _MIN_MOVED + _MIN_MISMATCHED_CANDIDATES:
        return None, None
    candidate_color = {r: color for r, color in candidates}
    real_colors = sorted(set(candidate_color.values()))
    if len(real_colors) < 2:
        return None, None
    if not required_match_rows.issubset(set(candidate_rows)):
        return None, None

    max_match = min(_MAX_MOVED, len(candidate_rows) - _MIN_MISMATCHED_CANDIDATES)
    min_match = max(_MIN_MOVED, len(required_match_rows))
    if max_match < min_match:
        return None, None
    n_match = random.randint(min_match, max_match)
    extra_rows = [r for r in candidate_rows if r not in required_match_rows]
    matched_rows = set(required_match_rows)
    if len(matched_rows) < n_match:
        matched_rows.update(random.sample(extra_rows, n_match - len(matched_rows)))

    source_color: dict[int, int] = {}
    H, _ = g.shape
    for r in range(1, H - 1, 2):
        corridor_color = candidate_color.get(r) or _first_real_corridor_color(g, r)
        if corridor_color is None:
            source_color[r] = random.choice(real_colors)
            continue
        if r in matched_rows:
            source_color[r] = corridor_color
        else:
            alt = _other_palette_color(real_colors, corridor_color)
            if alt is None:
                return None, None
            source_color[r] = alt

    stats = {
        "candidate_rows": len(candidate_rows),
        "matched_rows": len(matched_rows),
        "mismatched_candidates": sum(
            1 for r in candidate_rows if source_color[r] != candidate_color[r]
        ),
        "matched_colors": len({source_color[r] for r in matched_rows}),
    }
    if stats["mismatched_candidates"] < _MIN_MISMATCHED_CANDIDATES:
        return None, None
    if stats["matched_colors"] < 2:
        return None, None
    return source_color, stats


def _paint_sources(g: np.ndarray, source_color: dict[int, int]) -> None:
    H, _ = g.shape
    g[:, 0] = BG
    g[:, 1] = BG
    for r in range(1, H - 1, 2):
        g[r, 0] = source_color[r]


def _count_moved_sources(inp: np.ndarray, out: np.ndarray) -> int:
    H = inp.shape[0]
    return sum(
        1
        for r in range(1, H - 1, 2)
        if int(inp[r, 0]) != BG and int(out[r, 0]) == BG
    )


def _gen_attempt() -> np.ndarray | None:
    H = random.choice(_HEIGHT_OPTIONS)
    W = random.randint(*_WIDTH_RANGE)
    palette = random.sample(_PALETTE_POOL, _N_EXTRA_COLORS)
    bar_rows = list(range(0, H, 2))
    n_channels = len(bar_rows) - 1

    n_real = min(random.choices(_N_REAL_OPTIONS, weights=_N_REAL_WEIGHTS, k=1)[0], n_channels)
    real_set = set(random.sample(range(n_channels), n_real))
    channel_spec = ["real" if i in real_set else "fake" for i in range(n_channels)]
    bar_start = random.choice(_BAR_START_OPTIONS)

    g, bar_rows, bar_color = _paint_base_grid(H, W, bar_start, channel_spec, palette)
    if len(set(bar_color.values())) < _MIN_PALETTE_DIVERSITY:
        return None

    real_indices = [i for i, s in enumerate(channel_spec) if s == "real"]
    used_vcols: set[int] = set()
    plans: dict[int, dict[str, object]] = {}

    forced_plan, required_sources = _plan_forced_u_turn(
        real_indices, bar_rows, bar_color, bar_start, W, used_vcols
    )
    if forced_plan is None:
        return None
    plans.update(forced_plan)
    _plan_remaining(real_indices, bar_start, W, used_vcols, plans)
    _execute_plans(g, H, W, plans, bar_rows, bar_color, bar_start, palette)
    _add_fake_decor(g, channel_spec, bar_rows, bar_color, bar_start)

    required_match_rows = {bar_rows[i] + 1 for i in required_sources}
    source_color, src_stats = _assign_source_colors(g, palette, required_match_rows)
    if source_color is None or src_stats is None:
        return None
    _paint_sources(g, source_color)

    try:
        out = main(g)
    except Exception:
        return None

    metrics = _fusion_metrics(g)
    n_moved = _count_moved_sources(g, out)
    if not (_MIN_MOVED <= n_moved <= _MAX_MOVED):
        return None
    if n_moved != metrics["moved"]:
        return None
    if metrics["moved"] != src_stats["matched_rows"]:
        return None
    if metrics["cue_ccw"] < 1 or metrics["cue_cw"] < 1:
        return None
    if metrics["ambiguous_events"] < 2:
        return None
    if metrics["legacy_differences"] < 1:
        return None
    if metrics["vertical_turns"] < 1:
        return None
    if metrics["multicolor_paths"] < _MIN_MULTICOLOR_PATHS:
        return None
    if metrics["color_transitions"] < _MIN_COLOR_TRANSITIONS:
        return None
    if metrics["trail_cells"] < _MIN_TRAIL_CELLS:
        return None
    if metrics["mismatched_candidates"] < _MIN_MISMATCHED_CANDIDATES:
        return None
    if metrics["matched_candidate_colors"] < 2:
        return None
    if len(set(np.unique(g).tolist())) < _MIN_DISTINCT_COLORS:
        return None
    if np.array_equal(out, g):
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {_MAX_RETRIES} attempts")
