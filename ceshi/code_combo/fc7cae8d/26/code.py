# Combo variant of fc7cae8d (index 26)
# combo_spec: D4/4+D9/1
# Name: 条件旋转+长度编码旋转
#
# Design analysis for this combo variant
# a) Concrete behavior and trace:
#    The frame still carries one horizontal control edge and one vertical control edge. The interior
#    object is still a filled rectangle plus two 1px guides on the two control-facing sides, but the
#    guides no longer copy the control colors. Instead they use two fixed marker colors:
#    horizontal-guide marker = 6, vertical-guide marker = 8. The two marker segments have the same
#    visible length L and are anchored at the shared guide corner. Length encodes a base rotation:
#    L=2/3/4/5 -> rot90 count 0/1/2/3. The solver first undoes an unknown D4 transform by searching
#    for the orientation where marker 6 lies on the frame's horizontal control side and marker 8 lies
#    on the vertical control side. It then crops away those two guide borders, exactly as in the
#    original task, and finally rotates the cropped core by
#    total_k = base_k_from_length + [ctrl_h > ctrl_v] (mod 4).
#    Small trace: suppose the true aligned object should face top+left, the guide segments both have
#    length 4, so base_k=2. If the frame colors are top=7 and left=3 then ctrl_h>ctrl_v, so the
#    extra term is 1 and the cropped core is rotated 3 quarter-turns total. Under the original task,
#    the same sample would stop immediately after alignment + crop.
# b) Complexity calibration:
#    The combo is intentionally two-stage but still low-arity. The length code has only four values,
#    repeated on both guide sides, so humans get redundant evidence instead of a large lookup table.
#    The conditional branch is binary and tied to a simple observable comparison between the two frame
#    colors. The generator forces non-square, asymmetric cores so every non-zero rotation is visible.
# c) Implicit assumptions review:
#    - The original guide_edges() routine assumed guide colors match frame colors and occupy nearly an
#      entire edge. That assumption is broken here, so the solver replaces it with explicit detection
#      of anchored marker segments on candidate border sides.
#    - The original return-after-crop assumed crop is the terminal action. That is false here because
#      the cropped core still needs a length-coded rotation and maybe one extra conditional turn.
#    - extract_center_object() still assumes the target object is the largest interior component; the
#      generator preserves that by placing exactly one interior guided rectangle away from the frame.
#    - Control detection still assumes one dominant horizontal border color and one dominant vertical
#      border color; the frame painter preserves that, with only moderate grey noise on the unused
#      edges.
# d) Placement legality:
#    The generator constructs the aligned guided object first, applies an inverse D4 transform, and
#    uses random_free_location(..., transparent=0, border=1). Non-zero object cells never overlap the
#    frame controls or extend out of bounds, and the object remains a single interior component.
# e) Visual observability:
#    A human sees two fixed-color guide segments whose common length changes across examples, so the
#    length->rotation code is inferable from repeated structure. Separately, examples with ctrl_h >
#    ctrl_v versus ctrl_h < ctrl_v differ by one extra quarter-turn after the same guide-length code.
# f) Visual cleanliness:
#    There is only one interior object. The frame controls stay on the outer border, the object stays
#    separated by at least one background cell, and the guide markers occupy only thin side segments.
#    This keeps the three role types (frame controls, guide markers, core pattern) visually distinct.
# g) Color-change inferability:
#    No hidden recoloring is used. Frame control colors vary only to trigger the simple greater-than
#    condition; guide markers keep fixed colors and therefore read as structural symbols, not as a
#    free-form color mapping problem.
# h) New solver assumptions and generator guarantees:
#    - main() now assumes the horizontal and vertical guide markers share one of the four legal
#      lengths and are anchored next to their shared corner. generate_input() enforces that exactly.
#    - main() assumes guide marker colors 6 and 8 are absent from the core/fill/frame controls except
#      at the guides. generate_input() excludes them from all other sampled colors.
#    - main() assumes the core is not rotationally ambiguous enough to hide the final orientation.
#      generate_input() enforces non-square shape and asymmetric deterministic motifs.

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/arc2_solver_select")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg, paste, random_free_location

BG = 0
NOISE = 5
H_MARK = 6
V_MARK = 8

LEN_TO_BASE_K = {2: 0, 3: 1, 4: 2, 5: 3}
BASE_K_TO_LEN = {v: k for k, v in LEN_TO_BASE_K.items()}
CTRL_COLORS = [1, 2, 3, 4, 7, 9]


def _detect_controls_frame(arr: np.ndarray, bg: int) -> list[tuple[str, int]]:
    H, W = arr.shape
    sides = {
        "top": arr[0, 1:-1],
        "bottom": arr[-1, 1:-1],
        "left": arr[1:-1, 0],
        "right": arr[1:-1, -1],
    }
    cand: list[tuple[str, int, int, int]] = []
    for side, values in sides.items():
        vals = list(map(int, values))
        non_bg = [x for x in vals if x != int(bg)]
        if not non_bg:
            continue
        color, count = Counter(non_bg).most_common(1)[0]
        if int(count) >= len(vals) - 1:
            cand.append((side, int(color), int(count), int(len(vals))))
    cand.sort(key=lambda t: (t[2] / t[3], t[2]), reverse=True)
    return [(side, color) for side, color, _count, _length in cand[:2]]


def _extract_center_object(arr: np.ndarray, bg: int) -> np.ndarray:
    H, W = arr.shape
    comps = find_components(arr, bg=int(bg), connectivity=4, monochromatic=False, return_format="dict")
    best = None
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if bool(r0 == 0 or c0 == 0 or r1 == H - 1 or c1 == W - 1):
            continue
        area = int(comp["area"])
        if best is None or area > best[0]:
            best = (area, (r0, c0, r1, c1))
    if best is None:
        return arr.copy()
    return crop_bbox(arr, best[1])


def _control_axes(desired: dict[str, int]) -> tuple[str, str, int, int]:
    horiz_side = None
    vert_side = None
    ctrl_h = None
    ctrl_v = None
    for side, color in desired.items():
        if side in ("top", "bottom"):
            horiz_side = side
            ctrl_h = int(color)
        else:
            vert_side = side
            ctrl_v = int(color)
    if horiz_side is None or vert_side is None or ctrl_h is None or ctrl_v is None:
        raise ValueError("expected one horizontal and one vertical control")
    return str(horiz_side), str(vert_side), int(ctrl_h), int(ctrl_v)


def _side_values(arr: np.ndarray, side: str) -> np.ndarray:
    if side == "top":
        return arr[0, :]
    if side == "bottom":
        return arr[-1, :]
    if side == "left":
        return arr[:, 0]
    if side == "right":
        return arr[:, -1]
    raise ValueError(f"unknown side: {side}")


def _anchored_run_length(values: np.ndarray, color: int, anchor: str, bg: int) -> int:
    vals = [int(v) for v in values]
    positions = [idx for idx, v in enumerate(vals) if int(v) == int(color)]
    if not positions:
        return 0
    if any(int(v) not in {int(bg), int(color)} for v in vals):
        return 0
    if positions != list(range(positions[0], positions[-1] + 1)):
        return 0
    run_len = len(positions)
    if anchor == "start":
        expected = list(range(1, 1 + run_len))
    elif anchor == "end":
        expected = list(range(len(vals) - 1 - run_len, len(vals) - 1))
    else:
        raise ValueError(f"unknown anchor: {anchor}")
    if positions != expected:
        return 0
    return int(run_len)


def _aligned_code_length(arr: np.ndarray, horiz_side: str, vert_side: str, bg: int) -> int:
    h_anchor = "start" if vert_side == "left" else "end"
    v_anchor = "start" if horiz_side == "top" else "end"
    h_len = _anchored_run_length(_side_values(arr, horiz_side), H_MARK, h_anchor, bg)
    v_len = _anchored_run_length(_side_values(arr, vert_side), V_MARK, v_anchor, bg)
    if h_len != v_len or h_len not in LEN_TO_BASE_K:
        return 0
    return int(h_len)


def _crop_guides(arr: np.ndarray, horiz_side: str, vert_side: str) -> np.ndarray:
    top = 1 if horiz_side == "top" else 0
    bottom = int(arr.shape[0]) - (1 if horiz_side == "bottom" else 0)
    left = 1 if vert_side == "left" else 0
    right = int(arr.shape[1]) - (1 if vert_side == "right" else 0)
    return arr[int(top) : int(bottom), int(left) : int(right)]


def _conditional_extra(ctrl_h: int, ctrl_v: int) -> int:
    return 1 if int(ctrl_h) > int(ctrl_v) else 0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the canonical guide orientation, crop the guide borders, then rotate the core by a
    two-part rule: base quarter-turns encoded by the guide-segment length plus one extra turn when
    the horizontal frame-control color is numerically larger than the vertical one.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    ctrls = _detect_controls_frame(g, bg)
    desired = {side: color for side, color in ctrls}
    horiz_side, vert_side, ctrl_h, ctrl_v = _control_axes(desired)
    obj = _extract_center_object(g, bg)

    for k in range(4):
        base = np.rot90(obj, int(k))
        for flip in (False, True):
            t = np.fliplr(base) if flip else base
            code_len = _aligned_code_length(t, horiz_side, vert_side, bg)
            if not code_len:
                continue
            core = _crop_guides(t, horiz_side, vert_side)
            total_k = (LEN_TO_BASE_K[int(code_len)] + _conditional_extra(ctrl_h, ctrl_v)) % 4
            return np.rot90(core, int(total_k))

    return obj


def _transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(arr, int(k))
    return np.fliplr(out) if flip else out


def _inverse_transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.fliplr(arr) if flip else arr
    return np.rot90(out, -int(k))


def _build_core(h: int, w: int, fill: int, pattern_colors: list[int]) -> np.ndarray:
    core = np.full((h, w), int(fill), dtype=int)
    colors = list(pattern_colors)

    # Deterministic asymmetric anchors make all quarter-turn outcomes visually distinct.
    core[0, 1 : min(w, 1 + max(2, w // 2))] = int(colors[0])
    core[min(h - 1, 1) : min(h, 1 + max(2, h // 2)), w - 2] = int(colors[1])

    elbow_r = max(1, h - 3)
    core[elbow_r : elbow_r + 2, 0] = int(colors[2])
    core[h - 1, 0 : min(w, 3)] = int(colors[2])

    if len(colors) > 3:
        core[h // 2, max(1, w // 3)] = int(colors[3])

    area = h * w
    max_nonfill = min(area - 1, max(len(colors) + 6, int(area * 0.28)))
    target_nonfill = random.randint(len(colors) + 6, max_nonfill)
    painted = int(np.count_nonzero(core != int(fill)))

    while painted < target_nonfill:
        color = int(random.choice(colors))
        shape_kind = random.choice(["dot", "line", "rect"])
        if shape_kind == "dot":
            cells = [(random.randrange(h), random.randrange(w))]
        elif shape_kind == "line":
            line_len = random.randint(2, min(5, max(h, w)))
            if random.random() < 0.5:
                rr = random.randrange(h)
                c0 = random.randrange(w - line_len + 1)
                cells = [(rr, c0 + dc) for dc in range(line_len)]
            else:
                r0 = random.randrange(h - line_len + 1)
                cc = random.randrange(w)
                cells = [(r0 + dr, cc) for dr in range(line_len)]
        else:
            mh = random.randint(1, min(3, h))
            mw = random.randint(1, min(3, w))
            r0 = random.randrange(h - mh + 1)
            c0 = random.randrange(w - mw + 1)
            cells = [(r0 + dr, c0 + dc) for dr in range(mh) for dc in range(mw)]

        remaining = target_nonfill - painted
        if remaining <= 0:
            break
        free_cells = [(r, c) for r, c in cells if int(core[r, c]) == int(fill)]
        if len(free_cells) > remaining:
            free_cells = random.sample(free_cells, remaining)
        for r, c in free_cells:
            core[r, c] = int(color)
        painted += len(free_cells)

    return core


def _attach_length_guides(core: np.ndarray, horiz_side: str, vert_side: str, code_len: int) -> np.ndarray:
    h, w = core.shape
    has_top = horiz_side == "top"
    has_left = vert_side == "left"
    out = np.zeros((h + 1, w + 1), dtype=int)
    r0 = 1 if has_top else 0
    c0 = 1 if has_left else 0
    paste(out, core, int(r0), int(c0))

    if horiz_side == "top":
        row = 0
    else:
        row = int(out.shape[0] - 1)
    if vert_side == "left":
        col = 0
    else:
        col = int(out.shape[1] - 1)

    if vert_side == "left":
        h_start = 1
        h_end = 1 + int(code_len)
    else:
        h_start = int(out.shape[1] - 1 - code_len)
        h_end = int(out.shape[1] - 1)
    out[int(row), int(h_start) : int(h_end)] = int(H_MARK)

    if horiz_side == "top":
        v_start = 1
        v_end = 1 + int(code_len)
    else:
        v_start = int(out.shape[0] - 1 - code_len)
        v_end = int(out.shape[0] - 1)
    out[int(v_start) : int(v_end), int(col)] = int(V_MARK)

    out[int(row), int(col)] = int(BG)
    return out


def _paint_frame(grid: np.ndarray, horiz_side: str, vert_side: str, ctrl_h: int, ctrl_v: int) -> None:
    n = grid.shape[0]
    if horiz_side == "top":
        grid[0, :] = int(ctrl_h)
    else:
        grid[-1, :] = int(ctrl_h)
    if vert_side == "left":
        grid[:, 0] = int(ctrl_v)
    else:
        grid[:, -1] = int(ctrl_v)

    for side in ("top", "bottom", "left", "right"):
        if side in (horiz_side, vert_side):
            continue
        if side in ("top", "bottom"):
            r = 0 if side == "top" else n - 1
            coords = [(r, c) for c in range(1, n - 1)]
        else:
            c = 0 if side == "left" else n - 1
            coords = [(r, c) for r in range(1, n - 1)]
        min_count = int(np.ceil(0.35 * len(coords)))
        max_count = int(np.floor(0.60 * len(coords)))
        noise_count = random.randint(min_count, max_count)
        for r, c in random.sample(coords, noise_count):
            grid[r, c] = int(NOISE)


def _choose_controls() -> tuple[int, int]:
    trigger = bool(random.randrange(2))
    if trigger:
        hi = random.choice(CTRL_COLORS[1:])
        lo_choices = [c for c in CTRL_COLORS if c < hi]
        return int(hi), int(random.choice(lo_choices))
    lo = random.choice(CTRL_COLORS[:-1])
    hi_choices = [c for c in CTRL_COLORS if c > lo]
    return int(lo), int(random.choice(hi_choices))


def _expected_output(core: np.ndarray, code_len: int, ctrl_h: int, ctrl_v: int) -> np.ndarray:
    total_k = (LEN_TO_BASE_K[int(code_len)] + _conditional_extra(ctrl_h, ctrl_v)) % 4
    return np.rot90(core, int(total_k))


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(22, 29)
    horiz_side, vert_side = random.choice(
        [("top", "left"), ("top", "right"), ("bottom", "left"), ("bottom", "right")]
    )

    ctrl_h, ctrl_v = _choose_controls()
    code_len = random.choice(sorted(LEN_TO_BASE_K))

    remaining = [c for c in range(1, 10) if c not in {NOISE, H_MARK, V_MARK, ctrl_h, ctrl_v}]
    fill = random.choice(remaining)
    remaining = [c for c in remaining if c != fill]
    pattern_colors = random.sample(remaining, random.randint(3, min(4, len(remaining))))

    max_core_h = min(13, n - 4)
    max_core_w = min(15, n - 4)
    min_core_dim = max(6, int(code_len) + 1)
    if max_core_h < min_core_dim or max_core_w < min_core_dim + 1:
        return None

    core_h = random.randint(min_core_dim, max_core_h)
    core_w_choices = [w for w in range(min_core_dim, max_core_w + 1) if w != core_h]
    if not core_w_choices:
        return None
    core_w = random.choice(core_w_choices)

    core = _build_core(core_h, core_w, fill, pattern_colors)
    expected = _expected_output(core, code_len, ctrl_h, ctrl_v)

    aligned_object = _attach_length_guides(core, horiz_side, vert_side, code_len)
    k = random.randrange(4)
    flip = bool(random.randrange(2))
    input_object = _inverse_transform(aligned_object, k, flip)

    grid = np.zeros((n, n), dtype=int)
    _paint_frame(grid, horiz_side, vert_side, ctrl_h, ctrl_v)

    try:
        r0, c0 = random_free_location(grid, input_object, transparent=0, border=1)
    except ValueError:
        return None
    paste(grid, input_object, int(r0), int(c0), transparent=0)

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for fc7cae8d combo 26")
