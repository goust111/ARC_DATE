# Variant of 6e453dd6
# Dimension: D10 — 多步处理顺序 (multi_step)
# Change: 三步序列：Step1平移精灵→Step2检测封闭洞→Step3发射射线。各精灵独立处理，步骤间无交互... → 先发射射线再平移
#
# Design analysis:
# spec:
#   concretization:
#     - 画面仍然是左宽右窄的双分区，灰色整列分隔线不变，黑色精灵全部在左侧，右侧初始为空背景。
#     - 输出的左侧最终状态仍然与原题一致：所有黑色精灵都会右移到紧贴分隔线。
#     - 变体只改步骤顺序：红色射线不是由“平移后的贴线洞”决定，而是由“输入时刻已经贴着 sep-2 的封闭洞”决定。
#   behavior_trace:
#     - 原题里，一个发射器即使起初离分隔线较远，只要它右移后洞会落到 sep-2，就会在对应行发射红线。
#     - 本变体里，同一个发射器如果起初左移了 1-3 格，那么在发射判定发生时它的洞还没到 sep-2，因此不会发射；之后再平移也不会补发射。
#     - 只有输入中本来就 flush 的发射器会留下红线，而其他发射器只会在左侧移动，不会在右侧留下痕迹。
#   complexity_calibration:
#     - 保持原题的精灵类型、带状排布、>=60% 发射器比例和窄右半区，避免引入新的识别任务。
#     - 为了让 D10 差异稳定可见，生成器强制“至少一个发射器 flush、至少一个发射器 non-flush”，这样同一题里同时出现会发射和不会发射的封闭洞。
#   implicit_assumptions_from_original_main:
#     - 原 `main()` 先清空旧黑像素再重画，这默认“射线只依赖平移后的 left”。变体中如果照搬，会错误地按新位置触发射线。
#     - 原 `trigger_col = sep - 2` 依赖所有发射判定都发生在平移后；在变体里这个列仍然正确，但必须在修改 left 之前检查。
#     - 原代码用整块左半区找洞，这在变体里仍成立，因为精灵的纵向 band 互不重叠，外部背景始终与边界连通，真正的洞仍是内部 bg 连通块。
#   placement_legality:
#     - 每个精灵占据独立竖向 band，因此无论向左回退多少列都不会与其他精灵重叠。
#     - 精灵最终 flush 位置仍完全落在左半区内，输入中的左移量也被限制在 [0, c0]，不会越界。
#     - 右半区始终只允许背景或最终的红射线，solver 的断言因此保持有效。
#   visual_observability:
#     - 人类能直接看到：有些“洞形发射器”输入时已经贴着分隔线，所以输出有红线；另一些形状相似但离分隔线更左，输出只移动不发光。
#     - 规则不依赖颜色 id、计数顺序或隐藏排序，只依赖肉眼可见的“是否先贴线再判定”。
#   visual_cleanliness:
#     - 红线只出现在右侧，黑精灵只在左侧，层次与原题一致，不增加额外标记。
#     - band 之间至少隔一行背景，避免多个候选发射器在视觉上粘连。
#   color_inference:
#     - 仍只使用背景色、黑色、灰色分隔线和红色射线四种角色色；红色只由“输入时已贴线的封闭洞”触发，映射关系稳定。

import random

import numpy as np

from arc_utils import components_mask, paste


def _components4(mask):
    return components_mask(mask, connectivity=4)


def _find_separator_col(g: np.ndarray, sep_color: int = 5) -> int:
    _, w = g.shape
    cols = [c for c in range(w) if np.all(g[:, c] == sep_color)]
    assert len(cols) == 1, f"expected a unique full separator column, got {cols}"
    return int(cols[0])


def _background_color(g: np.ndarray, sep_col: int, sep_color: int = 5) -> int:
    h, w = g.shape
    border = []
    for c in range(w):
        if c != sep_col and g[0, c] != sep_color:
            border.append(int(g[0, c]))
        if c != sep_col and g[h - 1, c] != sep_color:
            border.append(int(g[h - 1, c]))
    for r in range(h):
        if g[r, 0] != sep_color:
            border.append(int(g[r, 0]))
        if g[r, w - 1] != sep_color:
            border.append(int(g[r, w - 1]))
    assert border, "expected non-separator border pixels to infer the background"
    vals, counts = np.unique(np.array(border, dtype=int), return_counts=True)
    winners = vals[counts == counts.max()]
    assert len(winners) == 1, f"background color must be unique on the border, got {winners.tolist()}"
    return int(winners[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire rays from already-aligned holes, then slide the black sprites into the separator.

    Input structure:
    - A single full-height grey column splits the board into a left sprite pane and a narrow right pane.
    - Black connected sprites sit on the left over one background color, while the right pane starts empty.

    Rule:
    - First inspect the left pane exactly as given and find enclosed background holes.
    - Any hole that already reaches the column immediately left of the separator emits a red ray across the right pane on each of its rows.
    - Only after those rays are fixed do all black sprites translate rightward until their outer edge touches the separator.

    Key insight: A hole that would become a nozzle after sliding is too late in this variant; emission depends on whether the hole is separator-adjacent before the move.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    sep_color = 5
    black = 0
    red = 2

    sep = _find_separator_col(g, sep_color=sep_color)
    bg = _background_color(g, sep_col=sep, sep_color=sep_color)

    out = g.copy()
    left = out[:, :sep].copy()
    left_mask = left == black
    comps = _components4(left_mask)

    # Step 1: on the original left pane, find holes that are already adjacent to the separator.
    holes = []
    for pts in _components4(left == bg):
        if any(r == 0 or r == h - 1 or c == 0 or c == sep - 1 for r, c in pts):
            continue
        holes.append(pts)

    trigger_col = sep - 2
    trigger_rows = []
    for pts in holes:
        if not any(c == trigger_col for _, c in pts):
            continue
        trigger_rows.extend(sorted(set(r for r, _ in pts)))
    assert len(trigger_rows) == len(set(trigger_rows)), f"trigger rows must be unique, got {trigger_rows}"

    for r in trigger_rows:
        assert np.all(out[r, sep + 1 :] == bg), f"right half of row {r} must be background before ray emission"
        out[r, sep + 1 :] = red

    # Step 2: afterwards, slide every black component rightward until it is flush with the separator.
    moved_left = left.copy()
    moved_left[moved_left == black] = bg

    target_right = sep - 1
    for comp in comps:
        max_c = max(c for _, c in comp)
        delta = target_right - max_c
        assert delta >= 0, "left-side components must stay left of the separator"
        for r, c in comp:
            nc = c + delta
            assert 0 <= nc < sep, f"shifted pixel {(r, c)} left the pane"
            moved_left[r, nc] = black

    out[:, :sep] = moved_left
    return out


# --------------- generator ---------------

# ── Colors ──
BG = 6
BLACK = 0
RED = 2
SEP_COLOR = 5

# ── Grid dimensions ──
H_MIN, H_MAX = 16, 25
W_MIN, W_MAX = 10, 16
SEP_MIN, SEP_MAX = 5, 11
RIGHT_W_MIN, RIGHT_W_MAX = 2, 4

# ── Sprite constraints ──
N_SPRITES_MIN, N_SPRITES_MAX = 2, 5
SPRITE_H_MAX = 8
SPRITE_W_MAX = 6
MIN_BAND_GAP = 1

# ── Probabilities ──
SPLIT_PROB = 0.4
EXTRA_EMITTERS_PROB = 0.25
DISTRACTOR_FLUSH_PROB = 0.35

# ── Retry budget ──
MAX_RETRIES = 300


def _scatter(total, slots):
    """Distribute *total* units randomly into *slots* bins."""
    vals = [0] * slots
    for _ in range(total):
        vals[random.randrange(slots)] += 1
    return vals


def _min_emitters_for(n):
    """Minimum emitter count to satisfy >=60% emitter fraction for *n* sprites."""
    return (3 * n + 4) // 5


def _min_rows_needed(n_emitters, n_total):
    """Minimum grid rows to fit *n_total* sprites with *n_emitters* emitters."""
    return n_emitters * 3 + (n_total - n_emitters) * 2 + (n_total - 1) + 2


def _sprite_bands(h, is_emitter):
    """Partition grid rows into disjoint vertical bands for each sprite."""
    n = len(is_emitter)
    min_heights = [3 if e else 2 for e in is_emitter]
    base_gaps = [MIN_BAND_GAP] * (n + 1)
    height_cap = h - sum(base_gaps)
    if sum(min_heights) > height_cap:
        return None

    heights = min_heights[:]
    extra_cap = sum(SPRITE_H_MAX - x for x in heights)
    extra = random.randint(0, min(extra_cap, height_cap - sum(heights)))
    for _ in range(extra):
        choices = [i for i, x in enumerate(heights) if x < SPRITE_H_MAX]
        heights[random.choice(choices)] += 1

    gaps = [a + b for a, b in zip(base_gaps, _scatter(h - sum(heights) - sum(base_gaps), n + 1))]
    bands = []
    r = gaps[0]
    for i, sh in enumerate(heights):
        bands.append((r, sh))
        r += sh + gaps[i + 1]
    return bands


def _emitter_patch(sh, max_sw):
    """Build an emitter sprite patch: a closed shell with an interior hole at the right edge."""
    hole_w = random.choice([1, 2]) if max_sw >= 4 else 1
    sw = random.randint(hole_w + 2, max_sw)
    patch = np.full((sh, sw), BG, dtype=int)

    right_wall = sw - 1
    left_wall = sw - hole_w - 2

    right_shift = random.randint(1, max(1, sw - 3))
    mini_w = sw - right_shift
    can_split = sh >= 5 and mini_w >= 3 and right_shift >= 1
    if can_split and random.random() < SPLIT_PROB:
        _build_split_emitter(patch, sh, sw, left_wall, right_wall, mini_w)
    else:
        patch[0, :] = BLACK
        patch[-1, :] = BLACK
        patch[1:-1, left_wall] = BLACK
        patch[1:-1, right_wall] = BLACK
    return patch


def _build_split_emitter(patch, sh, sw, left_wall, right_wall, mini_w):
    """Fill *patch* in-place with a split emitter (mini section + main section)."""
    place_top = random.random() < 0.5
    mini_h = random.randint(3, min(4, sh - 2))
    main_h = sh - mini_h + 1
    if place_top:
        patch[0, :mini_w] = BLACK
        for r in range(mini_h):
            patch[r, 0] = BLACK
            patch[r, mini_w - 1] = BLACK
        patch[mini_h - 1, :] = BLACK
        patch[-1, :] = BLACK
        for r in range(mini_h, sh):
            patch[r, left_wall] = BLACK
            patch[r, right_wall] = BLACK
    else:
        patch[0, :] = BLACK
        for r in range(main_h - 1):
            patch[r, left_wall] = BLACK
            patch[r, right_wall] = BLACK
        patch[main_h - 1, :] = BLACK
        patch[-1, :mini_w] = BLACK
        for r in range(main_h, sh):
            patch[r, 0] = BLACK
            patch[r, mini_w - 1] = BLACK


def _distractor_patch(sh, max_sw):
    """Build a non-emitter distractor sprite (C-shape, L-shape, or solid block)."""
    choices = ["l_shape"]
    if sh >= 3 and max_sw >= 3:
        choices.extend(["c_left", "c_right"])
    if sh <= 3:
        choices.append("solid")
    kind = random.choice(choices)

    if kind == "solid":
        sw = random.randint(2, min(3, max_sw))
        return np.full((sh, sw), BLACK, dtype=int)

    min_sw = 3 if kind.startswith("c_") or sh == 2 else 2
    sw = random.randint(min_sw, max_sw)
    patch = np.full((sh, sw), BG, dtype=int)
    if kind == "c_left":
        patch[0, :] = BLACK
        patch[-1, :] = BLACK
        patch[:, -1] = BLACK
    elif kind == "c_right":
        patch[0, :] = BLACK
        patch[-1, :] = BLACK
        patch[:, 0] = BLACK
    else:
        patch[:, -1] = BLACK
        if random.random() < 0.5:
            patch[0, :] = BLACK
        else:
            patch[-1, :] = BLACK
    return patch


def _choose_grid_dims():
    """Pick grid height, width, and separator column satisfying all constraints."""
    h = random.randint(H_MIN, H_MAX)
    right_w = random.randint(RIGHT_W_MIN, RIGHT_W_MAX)
    valid_seps = [s for s in range(SEP_MIN, SEP_MAX + 1) if W_MIN <= s + 1 + right_w <= W_MAX]
    sep = random.choice(valid_seps)
    w = sep + 1 + right_w
    return h, w, sep


def _choose_emitter_flags(h):
    """Decide sprite count and which sprites are emitters."""
    valid_counts = [n for n in range(N_SPRITES_MIN, N_SPRITES_MAX + 1)
                    if _min_rows_needed(_min_emitters_for(n), n) <= h]
    n = random.choice(valid_counts)

    min_em = _min_emitters_for(n)
    max_em = min_em
    for k in range(min_em + 1, n + 1):
        if _min_rows_needed(k, n) <= h:
            max_em = k
    if max_em > min_em and random.random() < EXTRA_EMITTERS_PROB:
        n_emitters = random.randint(min_em + 1, max_em)
    else:
        n_emitters = min_em

    flags = [True] * n_emitters + [False] * (n - n_emitters)
    random.shuffle(flags)
    return flags


def _emitter_rows(patch):
    """Rows whose hole touches the inner face of the patch's right wall."""
    sh, sw = patch.shape
    assert sw >= 2
    rows = [r for r in range(sh) if patch[r, sw - 1] == BLACK and patch[r, sw - 2] == BG]
    assert rows, "emitter patch must expose at least one right-adjacent hole row"
    return rows


def _render_slid_state(h, w, sep, bands, flags):
    """Stamp all sprites flush against the separator and return the left-only final state."""
    out = np.full((h, w), BG, dtype=int)
    out[:, sep] = SEP_COLOR

    sprites = []
    max_sw = min(SPRITE_W_MAX, sep - 1)
    for (r0, sh), is_emitter in zip(bands, flags):
        patch = _emitter_patch(sh, max_sw) if is_emitter else _distractor_patch(sh, max_sw)
        sw = patch.shape[1]
        c0 = sep - sw
        paste(out, patch, r0, c0, transparent=BG)
        sprites.append({
            "patch": patch,
            "r0": r0,
            "c0": c0,
            "is_emitter": bool(is_emitter),
            "trigger_rows": _emitter_rows(patch) if is_emitter else [],
        })
    return out, sprites


def _choose_input_deltas(sprites):
    """Choose left shifts so some emitters are flush now and some only after the later slide."""
    emitter_ids = [i for i, sprite in enumerate(sprites) if sprite["is_emitter"]]
    assert emitter_ids, "generator requires at least one emitter"

    if len(emitter_ids) == 1:
        flush_emitters = {emitter_ids[0]}
    else:
        n_flush = random.randint(1, len(emitter_ids) - 1)
        flush_emitters = set(random.sample(emitter_ids, n_flush))

    deltas = []
    for i, sprite in enumerate(sprites):
        c0 = sprite["c0"]
        assert c0 >= 1, "flush sprite must leave at least one column to unslide"
        if sprite["is_emitter"]:
            delta = 0 if i in flush_emitters else random.randint(1, c0)
        else:
            delta = 0 if random.random() < DISTRACTOR_FLUSH_PROB else random.randint(1, c0)
        deltas.append(delta)

    assert any(sprites[i]["is_emitter"] and deltas[i] == 0 for i in range(len(sprites)))
    assert any(sprites[i]["is_emitter"] and deltas[i] > 0 for i in range(len(sprites)))
    return deltas


def _build_input(h, w, sep, sprites, deltas):
    """Apply the chosen left shifts to obtain the input grid."""
    inp = np.full((h, w), BG, dtype=int)
    inp[:, sep] = SEP_COLOR
    for sprite, delta in zip(sprites, deltas):
        paste(inp, sprite["patch"], sprite["r0"], sprite["c0"] - delta, transparent=BG)
    return inp


def _expected_output(slid_state, sep, sprites, deltas):
    """Build the independent oracle output for the reversed step order."""
    out = slid_state.copy()
    for sprite, delta in zip(sprites, deltas):
        if not sprite["is_emitter"] or delta != 0:
            continue
        for lr in sprite["trigger_rows"]:
            out[sprite["r0"] + lr, sep + 1 :] = RED
    return out


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt: build input, derive expected output, verify with oracle."""
    h, w, sep = _choose_grid_dims()
    flags = _choose_emitter_flags(h)
    bands = _sprite_bands(h, flags)
    if bands is None:
        return None

    slid_state, sprites = _render_slid_state(h, w, sep, bands, flags)
    deltas = _choose_input_deltas(sprites)
    inp = _build_input(h, w, sep, sprites, deltas)
    expected = _expected_output(slid_state, sep, sprites, deltas)

    try:
        if not np.array_equal(main(inp), expected):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 6e453dd6 input")
