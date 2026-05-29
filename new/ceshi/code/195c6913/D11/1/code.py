# Variant of 195c6913
# Dimension: D11 — 初始方向确定规则 (path)
# Change: 初始方向按优先级R,D,L,U检查第一个相邻path邻居确定（对col 0标记总是向右）→ 初始方向由模板中额外符号指定

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import label_mask


# Design analysis for the D11 variant
# a) Concrete change + trace:
#    The original top-left template only stored the recolor cycle via 2x2 blocks.
#    Here the first empty "template slot" immediately after the last 2x2 block
#    stores a two-pixel direction glyph:
#      - anchor at (template_row, 1 + 3*n_blocks) with pattern[0]
#      - indicator either to its right (launch Right) with pattern[-1]
#        or below it (launch Down) with pattern[-1]
#    Small trace: under the old rule, a col-0 marker with a path neighbor on the
#    right always launched right. Under the new rule, the exact same local path can
#    still launch down if the glyph's indicator sits below the anchor.
# b) Complexity calibration:
#    Only two headings are used: Right and Down. Left is impossible for col-0
#    markers, and Up would usually dead-end at the first collision because the
#    mandatory CCW turn would point out of bounds from the left edge. Keeping two
#    visible headings makes the symbolic change learnable without exploding cases.
# c) Hidden assumptions in the old solver:
#    The old `dir_order = [R,D,L,U]` relied on the marker neighborhood encoding the
#    launch direction. That is no longer valid, so `main()` must never inspect
#    neighboring path cells to decide the initial heading. The generator therefore
#    guarantees the glyph exists and is readable before marker simulation begins.
# d) Placement legality:
#    The generator reserves the next template slot for the glyph, keeps rows 1..3
#    clear through column 14 so the symbol never overlaps the path, and verifies
#    that template blocks, glyph cells, markers, and end block all lie on
#    background cells before painting them.
# e) Visual observability:
#    The glyph changes the trajectory immediately near the marker. `Down` samples are
#    constructed so the marker still has a valid right-path neighbor, which means
#    the new output differs from the original priority-based solver on the same
#    input. The cue is therefore visible both in the template and in the path trace.
# f) Visual cleanliness:
#    The glyph sits in a dedicated vacant template slot instead of overlapping the
#    2x2 cycle blocks, so the template remains readable as "cycle blocks + small
#    direction hint" rather than a noisy blob of mixed colors.
# g) Color inferability:
#    No color-ID arithmetic is introduced. The glyph reuses the first and last cycle
#    colors, and its meaning comes purely from relative position (right/below),
#    which is directly visible in 2-3 examples.
# h) Main()'s new assumptions:
#    - cycle blocks are still a single row of 2x2 monochrome blocks in the top-left
#    - the next slot after the last block contains the anchor/indicator glyph
#    - indicator is either right of or below the anchor, never both
#    - `Down` samples keep `mask[r, 1]` true for every marker row so the old rule
#      would have chosen Right, making the variant non-degenerate
# i) Generator guarantees for those assumptions:
#    The template row is still 1 or 2, the glyph uses the reserved next slot, cycle
#    colors are sampled without repetition, and every accepted sample is rechecked
#    by `main()` as an oracle before being returned.


def _assert_path_single_component(g: np.ndarray, path_color: int) -> None:
    """Eval invariant: path_color forms a single 4-connected component."""
    _, n_path = label_mask(g == path_color, connectivity=4)
    assert n_path == 1, f"path_color must be a single 4-connected component, got {n_path}"


def _assert_marker_embedded_in_path(
    g: np.ndarray, markers: list[tuple[int, int, int]], path_color: int
) -> None:
    """Eval invariant: col-0 markers stay embedded in a vertical path segment."""
    H, _ = g.shape
    for r, c, _ in markers:
        if c != 0:
            continue
        if r > 0:
            assert int(g[r - 1, c]) == path_color, (
                f"marker at ({r},{c}) up-neighbor ({r-1},{c})={int(g[r-1,c])} "
                f"must be path_color {path_color}"
            )
        if r < H - 1:
            assert int(g[r + 1, c]) == path_color, (
                f"marker at ({r},{c}) down-neighbor ({r+1},{c})={int(g[r+1,c])} "
                f"must be path_color {path_color}"
            )


def _detect_pattern_blocks(
    g: np.ndarray, template_bg: int, path_color: int
) -> tuple[list[int], list[tuple[int, int]]]:
    H, W = g.shape
    pattern: list[int] = []
    pat_cells: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for r in range(min(6, H - 1)):
        for c in range(min(25, W - 1)):
            if (int(r), int(c)) in seen:
                continue
            v = int(g[int(r), int(c)])
            if v == int(template_bg) or v == int(path_color):
                continue
            if (
                int(g[int(r + 1), int(c)]) == v
                and int(g[int(r), int(c + 1)]) == v
                and int(g[int(r + 1), int(c + 1)]) == v
            ):
                pattern.append(int(v))
                for rr, cc in [(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)]:
                    seen.add((int(rr), int(cc)))
                    pat_cells.append((int(rr), int(cc)))
    return pattern, pat_cells


def _decode_direction_symbol(
    g: np.ndarray,
    pattern: list[int],
    pat_cells: list[tuple[int, int]],
    template_bg: int,
) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    assert pattern and pat_cells, "template pattern must exist before reading direction glyph"
    H, W = g.shape
    template_row = min(r for r, _ in pat_cells)
    inst_c = 1 + 3 * len(pattern)
    anchor = (int(template_row), int(inst_c))
    right = (int(template_row), int(inst_c + 1))
    down = (int(template_row + 1), int(inst_c))
    assert 0 <= anchor[0] < H and 0 <= right[1] < W and 0 <= down[0] < H, (
        "direction glyph must fit inside the reserved post-template slot"
    )
    assert int(g[anchor]) == int(pattern[0]), (
        f"direction anchor at {anchor} must use the first cycle color {pattern[0]}"
    )
    right_is_indicator = int(g[right]) == int(pattern[-1]) and int(g[down]) == int(template_bg)
    down_is_indicator = int(g[down]) == int(pattern[-1]) and int(g[right]) == int(template_bg)
    assert right_is_indicator ^ down_is_indicator, (
        "direction glyph must place the indicator either right of or below the anchor"
    )
    if right_is_indicator:
        return (0, 1), [anchor, right]
    return (1, 0), [anchor, down]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate recoloring particles on a path, with the launch heading read from a template glyph.

    Input structure:
    - Top-left: a row of 2x2 monochrome blocks defines the repeating recolor cycle.
    - The next empty template slot contains a two-pixel glyph: first-cycle-color anchor plus a last-cycle-color indicator.
    - Isolated first-cycle-color pixels on col 0 are start markers embedded in the path, and a distinct 2x2 block marks the end color.

    Rule:
    - Decode the glyph: indicator to the right means launch Right; indicator below means launch Down.
    - Each marker starts in that decoded direction, advances through path cells, and repaints them by cycling through the template colors.
    - On each collision, paint the collision cell with the end color only when it is background, then apply the strict CCW/CW alternating turn schedule; if the scheduled turn is blocked, stop.
    - Erase the template blocks, the direction glyph, and the end block from the output.

    Key insight:
    - The template now carries both the color cycle and the initial heading, so the solver must parse a symbolic instruction instead of inferring the start direction from local path adjacency.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    template_bg = int(g[0, 0])

    counts = Counter(map(int, g.flatten()))
    path_color = int(
        max([(cnt, int(col)) for col, cnt in counts.items() if int(col) != int(template_bg)])[1]
    )
    _assert_path_single_component(g, path_color)

    pattern, pat_cells = _detect_pattern_blocks(g, int(template_bg), int(path_color))
    init_dir, symbol_cells = _decode_direction_symbol(g, pattern, pat_cells, int(template_bg))

    end_color = None
    end_pos = None
    pat_set = set(map(int, pattern))
    for r in range(H - 1):
        for c in range(W - 1):
            v = int(g[int(r), int(c)])
            if v == int(template_bg) or v == int(path_color) or v in pat_set:
                continue
            if (
                int(g[int(r + 1), int(c)]) == v
                and int(g[int(r), int(c + 1)]) == v
                and int(g[int(r + 1), int(c + 1)]) == v
            ):
                end_color = int(v)
                end_pos = (int(r), int(c))
                break
        if end_pos is not None:
            break

    markers: list[tuple[int, int, int]] = []
    for r in range(H):
        for c in range(W):
            v = int(g[int(r), int(c)])
            if v not in pat_set:
                continue
            same = 0
            has_path = False
            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < H and 0 <= cc < W:
                    if int(g[rr, cc]) == v:
                        same += 1
                    if int(g[rr, cc]) == int(path_color):
                        has_path = True
            if same == 0 and has_path:
                markers.append((int(r), int(c), int(v)))

    _assert_marker_embedded_in_path(g, markers, int(path_color))

    out = g.copy()
    for r, c in pat_cells + symbol_cells:
        out[int(r), int(c)] = int(template_bg)
    if end_pos is not None:
        er, ec = end_pos
        out[int(er) : int(er + 2), int(ec) : int(ec + 2)] = int(template_bg)

    per_marker_cells: list[set[tuple[int, int]]] = []
    for sr, sc, sv in markers:
        if not pattern:
            per_marker_cells.append(set())
            continue

        r, c = int(sr), int(sc)
        dr, dc = init_dir
        idx = int(pattern.index(int(sv))) if int(sv) in pattern else 0
        hits = 0
        visited: set[tuple[int, int, int, int, int, int]] = set()
        my_cells: set[tuple[int, int]] = set()

        for _ in range(H * W * 4):
            state = (int(r), int(c), int(dr), int(dc), int(idx), int(hits & 1))
            if state in visited:
                break
            visited.add(state)

            nr = int(r + dr)
            nc = int(c + dc)
            if not (0 <= nr < H and 0 <= nc < W):
                break

            if int(g[nr, nc]) == int(path_color):
                idx = int((idx + 1) % len(pattern))
                out[nr, nc] = int(pattern[idx])
                my_cells.add((int(nr), int(nc)))
                r, c = nr, nc
                continue

            if int(g[nr, nc]) == int(template_bg) and end_color is not None:
                out[nr, nc] = int(end_color)

            left = (-int(dc), int(dr))
            right = (int(dc), -int(dr))
            tdr, tdc = left if (int(hits) % 2 == 0) else right
            rr = int(r + tdr)
            cc = int(c + tdc)
            if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(path_color):
                dr, dc = int(tdr), int(tdc)
                hits += 1
                continue
            break

        per_marker_cells.append(my_cells)

    for i in range(len(per_marker_cells)):
        for j in range(i + 1, len(per_marker_cells)):
            overlap = per_marker_cells[i] & per_marker_cells[j]
            assert not overlap, (
                f"marker trajectories {i},{j} overlap on {len(overlap)} cell(s)"
            )

    return out


def _build_band(H: int, W: int, lobe_rows: list[int] | None = None, lobe_tail: int = 4) -> np.ndarray:
    """Return a thick anti-diagonal band with a finite left-edge lobe for markers."""
    bot_center = random.randint(0, W // 4)
    centers = np.linspace(W - 1, bot_center, H).astype(float)
    half_w = np.array(
        [
            0.5 * (3 + (1.0 - abs(2.0 * r / max(1, H - 1) - 1.0)) * (W - 6))
            for r in range(H)
        ]
    )

    c_left = np.zeros(H, dtype=int)
    c_right = np.zeros(H, dtype=int)

    cc_jit = np.convolve(
        np.array([random.randint(-3, 3) for _ in range(H)], dtype=float),
        np.ones(3) / 3,
        mode="same",
    )
    w_jit = np.convolve(
        np.array([random.randint(-3, 3) for _ in range(H)], dtype=float),
        np.ones(3) / 3,
        mode="same",
    )

    for r in range(H):
        c_left_param = centers[r] - half_w[r]
        if c_left_param <= 0:
            c_left[r] = 0
            c_right[r] = int(round(centers[r] + cc_jit[r] + half_w[r] + w_jit[r] / 2))
        else:
            w = max(2, min(W, int(round(2 * half_w[r] + w_jit[r]))))
            cc = centers[r] + cc_jit[r]
            c_left[r] = int(round(cc - w / 2))
            c_right[r] = c_left[r] + w - 1

    if lobe_rows:
        lmin = max(0, min(lobe_rows) - 1)
        lmax = min(H - 2, max(lobe_rows) + lobe_tail)
        for r in range(H):
            if lmin <= r <= lmax:
                c_left[r] = 0
            else:
                d_outside = max(0, lmin - r, r - lmax)
                c_left[r] = min(c_left[r], 2 * d_outside)

    for r in range(1, min(4, H)):
        c_left[r] = max(c_left[r], 15)
        if c_right[r] < c_left[r]:
            c_right[r] = c_left[r]

    c_right[0] = W - 1
    c_left[0] = max(c_left[0], W - 6)
    c_right[H - 1] = max(3, min(W - 1, c_right[H - 1]))

    for _ in range(2):
        for r in range(1, H - 1):
            if c_left[r] > c_left[r - 1] and c_left[r] > c_left[r + 1]:
                c_left[r] = max(c_left[r - 1], c_left[r + 1])

    for r in range(1, H):
        c_left[r] = min(c_left[r], c_left[r - 1] + 2)
        c_right[r] = min(c_right[r], c_right[r - 1] + 2)
        if c_left[r] > c_right[r - 1]:
            c_left[r] = c_right[r - 1]
        if c_right[r] < c_left[r - 1]:
            c_right[r] = c_left[r - 1]
        if c_right[r] < c_left[r]:
            c_right[r] = c_left[r]

    c_left = np.clip(c_left, 0, W - 1)
    c_right = np.clip(c_right, 0, W - 1)
    c_right[0] = W - 1

    mask = np.zeros((H, W), dtype=bool)
    for r in range(H):
        mask[r, c_left[r] : c_right[r] + 1] = True
    return mask


def _carve_fjord(mask: np.ndarray, r1: int, r2: int) -> bool:
    """Carve a bg strip from col 0 between two marker rows without disconnecting the path."""
    H, W = mask.shape
    sep_h = random.randint(2, 3)
    sr_min = r1 + 2
    sr_max = r2 - 1 - sep_h
    if sr_max < sr_min:
        return False
    sr = random.randint(sr_min, sr_max)

    cur_top = sr
    made_carve = False
    for k in range(W - 1):
        trial = mask.copy()
        for dr in range(sep_h):
            rr = cur_top + dr
            if 0 <= rr < H:
                trial[rr, k] = False
        _, n_trial = label_mask(trial, connectivity=4)
        if n_trial != 1:
            break
        np.copyto(mask, trial)
        made_carve = True
        if random.random() < 0.3 and cur_top > 0:
            cur_top -= 1
    return made_carve


def _legacy_priority_direction(g: np.ndarray, sr: int, sc: int, path_color: int) -> tuple[int, int] | None:
    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        rr = int(sr + dr)
        cc = int(sc + dc)
        if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1] and int(g[rr, cc]) == int(path_color):
            return (int(dr), int(dc))
    return None


def _gen_attempt() -> np.ndarray | None:
    H = W = random.randint(20, 30)
    n_markers = 2 if random.random() < 0.7 else 1

    palette = list(range(10))
    random.shuffle(palette)
    bg = palette[0]
    path_color = palette[1]
    end_color = palette[2]
    n_blocks = random.randint(2, 4)
    cycle_colors = random.sample(palette[3:], n_blocks)

    init_dir = (1, 0) if random.random() < 0.6 else (0, 1)
    lobe_tail = random.randint(4, 7) if init_dir == (1, 0) else random.randint(2, 5)

    if n_markers == 1:
        r1 = random.randint(H // 2, H - 6)
        markers = [r1]
    else:
        r1 = random.randint(H // 3, H // 2)
        r2 = random.randint(max(r1 + 6, 2 * H // 3), H - 5)
        markers = [r1, r2]

    mask = _build_band(H, W, lobe_rows=markers, lobe_tail=lobe_tail)
    if mask[0, 0]:
        return None
    _, n_path = label_mask(mask, connectivity=4)
    if n_path != 1:
        return None
    if not mask[0, :].any() or not mask[:, W - 1].any() or not mask[H - 1, :].any():
        return None
    if not mask[:, 0].any():
        return None
    for r in markers:
        if not (mask[r, 0] and mask[r - 1, 0] and mask[r + 1, 0]):
            return None
        if not (0 <= 1 < W and mask[r, 1]):
            return None

    if init_dir == (1, 0):
        lobe_end = max(r for r in range(H) if mask[r, 0])
        if lobe_end >= H - 1 or not mask[lobe_end, min(1, W - 1)]:
            return None

    if n_markers == 2:
        trial = mask.copy()
        if not _carve_fjord(trial, r1, r2):
            return None
        use_interior = random.random() < 0.25
        if use_interior:
            K = random.randint(2, 4)
            pre_carve_cells = mask & ~trial
            for c in range(min(K, W)):
                trial[:, c] |= pre_carve_cells[:, c]
        _, nc = label_mask(trial, connectivity=4)
        if nc != 1:
            return None
        if not trial[0, :].any() or not trial[:, W - 1].any() or not trial[H - 1, :].any():
            return None
        if not trial[:, 0].any():
            return None
        for r in markers:
            if not (trial[r, 0] and trial[r - 1, 0] and trial[r + 1, 0] and trial[r, 1]):
                return None
        if init_dir == (1, 0):
            lobe_end = max(r for r in range(H) if trial[r, 0])
            if lobe_end >= H - 1 or not trial[lobe_end, 1]:
                return None
        mask = trial

    _, n_bg = label_mask(~mask, connectivity=4)
    if n_bg < 2:
        return None

    template_row = random.choice([1, 2])
    block_cols = [1, 4, 7, 10][:n_blocks]
    inst_c = 1 + 3 * n_blocks
    if inst_c + 1 >= W:
        return None

    g = np.full((H, W), bg, dtype=int)
    g[mask] = path_color

    for bc in block_cols:
        for dr in range(2):
            for dc in range(2):
                rr, cc = template_row + dr, bc + dc
                if rr >= H or cc >= W or g[rr, cc] != bg:
                    return None

    anchor = (template_row, inst_c)
    indicator = (template_row, inst_c + 1) if init_dir == (0, 1) else (template_row + 1, inst_c)
    symbol_cells = [anchor, indicator]
    for rr, cc in symbol_cells:
        if rr >= H or cc >= W or g[rr, cc] != bg:
            return None

    for i, bc in enumerate(block_cols):
        g[template_row : template_row + 2, bc : bc + 2] = cycle_colors[i]

    g[anchor] = cycle_colors[0]
    g[indicator] = cycle_colors[-1]

    end_candidates = []
    for r in range(H // 2, H - 1):
        for c in range(W // 2, W - 1):
            if not (g[r : r + 2, c : c + 2] == bg).all():
                continue
            r0, r1_ = max(0, r - 1), min(H, r + 3)
            c0, c1_ = max(0, c - 1), min(W, c + 3)
            if (g[r0:r1_, c0:c1_] == path_color).any():
                continue
            end_candidates.append((r, c))
    if not end_candidates:
        return None
    er, ec = random.choice(end_candidates)
    g[er : er + 2, ec : ec + 2] = end_color

    for r in markers:
        g[r, 0] = cycle_colors[0]

    _, n_path_after = label_mask(g == path_color, connectivity=4)
    if n_path_after != 1:
        return None
    if g[0, 0] != bg:
        return None

    cnt = Counter(map(int, g.flatten()))
    non_bg = sorted([(c, col) for col, c in cnt.items() if col != bg], reverse=True)
    if not non_bg or non_bg[0][1] != path_color:
        return None

    for r in markers:
        if _legacy_priority_direction(g, r, 0, path_color) != (0, 1):
            return None
    if init_dir == (1, 0):
        assert any(
            _legacy_priority_direction(g, r, 0, path_color) != init_dir for r in markers
        )

    try:
        _ = main(g)
    except Exception:
        return None

    return g


def generate_input() -> np.ndarray:
    for _ in range(300):
        try:
            g = _gen_attempt()
        except Exception:
            g = None
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 300 attempts")
