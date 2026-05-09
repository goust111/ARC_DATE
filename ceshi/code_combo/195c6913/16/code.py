# Combo variant of 195c6913 (index 16)
# combo_spec: D2/2+D7/0
# Name: 符号指令转向+颜色条纹模板

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import label_mask

# Design analysis
# ---------------
# a) Concrete form and behavior trace
#    - The original task had two separate ideas in the top-left legend:
#      fat 2x2 color blocks encoded the paint cycle, while collision turns were
#      fixed globally by parity (L,R,L,R,...).
#    - This combo compresses both channels into one compact legend. The cycle is
#      now a single ROW of 1x1 color cells, read left->right. Each color cell has
#      a tiny 2-pixel side bar attached on its left or right:
#      left bar => CCW turn, right bar => CW turn.
#    - Example with three legend cells at cols [2,6,10]:
#      colors [red, blue, yellow], side bars [left, right, right].
#      The particle paints red->blue->yellow->red... exactly as before, but its
#      first three successful collision turns are CCW, then CW, then CW. Under
#      the old task, those same collisions would have forced CCW, then CW, then
#      CCW. The third bend therefore diverges and the output changes visibly.
# b) Complexity calibration
#    - The legend still uses only 2-4 repeated slots, so the rule is inferable
#      from a few examples. Each slot carries one color and one binary turn cue.
#    - Slots are spaced by 4 columns. That is deliberate: it keeps the row easy
#      to parse and prevents a right-bar followed by a left-bar from merging into
#      an accidental 2x2 symbol block that could be mistaken for the end marker.
#    - Cycle colors are sampled without repetition. Repeated colors are legal in
#      the original family, but on a 1x1 stripe they weaken the visibility of the
#      template change and make marker/template disambiguation less clean.
# c) Hidden assumptions that break from the original main()
#    - Old template detection assumed every cycle entry is a monochrome 2x2 block.
#      That fails here; we instead scan for a top-left single-row sequence of
#      color cells whose neighboring side bars define turn instructions.
#    - Old loop detection only needed `hits & 1` because parity fully defined the
#      turn rule. Here the next turn depends on `hits % len(turn_seq)`.
#    - Old template erasure cleared only cycle cells. Here both stripe cells and
#      their symbol bars must be erased from the output.
#    - Old end-block detection could skip only cycle colors. Here the symbol color
#      must also be excluded so a 2x1 instruction bar is never misread as semantic
#      content for the end marker search.
# d) Placement legality
#    - The generator reserves a clean top-left pocket through the last legend slot.
#      All stripe cells and side bars sit on background and are not path-adjacent.
#    - Markers remain isolated pixels on col 0 embedded inside the path band, just
#      as in the original task, so initial-direction behavior stays stable.
#    - The end block remains a 2x2 monochrome block in the bottom-right background
#      region with a 1-cell moat from the path.
# e) Visual observability
#    - A human can see that the cycle is no longer carried by 2x2 blocks: instead
#      the top-left legend is a slim row of colored cells with tiny left/right bars.
#    - The new turn rule is visible because bars repeat per slot; generator samples
#      non-alternating sequences and rejects cases whose output would match the old
#      parity-based rule, so the symbol channel is functional rather than cosmetic.
# f) Visual cleanliness
#    - Roles remain separated: legend in the top-left pocket, path as the large
#      diagonal band, markers on the left edge, end block in the opposite pocket.
#    - The legend uses one uniform symbol color and one row of color cells, so the
#      viewer sees a compact two-channel pattern instead of a noisy mixed region.
# g) Color-change inferability
#    - Colors still only communicate the paint cycle and the terminal collision
#      color. Turn direction comes from side placement, not from hidden color order.
# h) Fusion metric against degeneration
#    - We compare the combo simulation against an "old alternating-turn" simulation
#      on the same generated input. A valid sample must differ, and the generator
#      also enforces enough total turns to make the difference observable.
# i) New solver assumptions and generator guarantees
#    - There is exactly one top-left legend row with 2-4 color cells spaced by 4
#      columns, each having exactly one 2x1 side bar.
#    - All instruction bars share one symbol color and stay within the legend row's
#      two-row footprint.
#    - No other non-bg/non-path objects appear in that footprint, so the legend can
#      be extracted before markers are searched.


def _assert_path_single_component(g: np.ndarray, path_color: int) -> None:
    """Eval invariant: path_color forms a single 4-connected component."""
    _, n_path = label_mask(g == path_color, connectivity=4)
    assert n_path == 1, f"path_color must be a single 4-connected component, got {n_path}"


def _assert_marker_embedded_in_path(
    g: np.ndarray, markers: list[tuple[int, int, int]], path_color: int
) -> None:
    """Eval invariant: each col-0 marker remains embedded in a vertical path segment."""
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


def _detect_stripe_legend(
    g: np.ndarray, template_bg: int, path_color: int
) -> tuple[list[int], list[tuple[int, int]], list[tuple[int, int]], list[str], int]:
    """Return (pattern, stripe_cells, instr_cells, turn_seq, symbol_color)."""
    H, W = g.shape
    row_limit = min(5, H - 1)
    col_limit = min(18, W - 1)
    candidates: list[tuple[int, int, list[int], list[tuple[int, int]], list[tuple[int, int]], list[str], int]] = []

    for r in range(1, row_limit):
        colors: list[int] = []
        stripe_cells: list[tuple[int, int]] = []
        instr_cells: list[tuple[int, int]] = []
        turn_seq: list[str] = []
        symbol_cols: list[int] = []
        symbol_colors: list[int] = []

        for c in range(1, col_limit):
            v = int(g[r, c])
            if v == int(template_bg) or v == int(path_color):
                continue
            if int(g[r + 1, c]) == v:
                continue

            left_bar = (
                c - 1 >= 0
                and int(g[r, c - 1]) == int(g[r + 1, c - 1])
                and int(g[r, c - 1]) not in (int(template_bg), int(path_color), int(v))
            )
            right_bar = (
                c + 1 < W
                and int(g[r, c + 1]) == int(g[r + 1, c + 1])
                and int(g[r, c + 1]) not in (int(template_bg), int(path_color), int(v))
            )
            if left_bar == right_bar:
                continue

            bar_col = int(c - 1) if left_bar else int(c + 1)
            stripe_cells.append((int(r), int(c)))
            colors.append(int(v))
            turn_seq.append("L" if left_bar else "R")
            instr_cells.extend([(int(r), int(bar_col)), (int(r + 1), int(bar_col))])
            symbol_cols.append(int(bar_col))
            symbol_colors.append(int(g[r, bar_col]))

        if not (2 <= len(stripe_cells) <= 4):
            continue

        stripe_cells = sorted(stripe_cells, key=lambda rc: rc[1])
        colors = [int(g[rr, cc]) for rr, cc in stripe_cells]
        turn_seq = [
            "L"
            if int(g[r, c - 1]) == int(g[r + 1, c - 1]) and int(g[r, c - 1]) not in (int(template_bg), int(path_color), int(g[r, c]))
            else "R"
            for _, c in stripe_cells
        ]
        symbol_cols = [int(c - 1) if t == "L" else int(c + 1) for (_, c), t in zip(stripe_cells, turn_seq)]
        symbol_colors = [int(g[r, bc]) for bc in symbol_cols]
        instr_cells = [(int(r), int(bc)) for bc in symbol_cols] + [(int(r + 1), int(bc)) for bc in symbol_cols]

        cols = [c for _, c in stripe_cells]
        if any(cols[i + 1] - cols[i] != 4 for i in range(len(cols) - 1)):
            continue
        if len(set(colors)) != len(colors):
            continue
        if len(set(symbol_colors)) != 1:
            continue

        legend_cells = set(stripe_cells) | set(instr_cells)
        c0 = min(min(cols), min(symbol_cols))
        c1 = max(max(cols), max(symbol_cols))
        valid = True
        for rr in range(r, r + 2):
            for cc in range(max(0, c0), min(W, c1 + 1)):
                v = int(g[rr, cc])
                if (rr, cc) in legend_cells:
                    if v == int(template_bg) or v == int(path_color):
                        valid = False
                        break
                elif v != int(template_bg):
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue

        for rr, cc in legend_cells:
            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nr = int(rr + dr)
                nc = int(cc + dc)
                if 0 <= nr < H and 0 <= nc < W and int(g[nr, nc]) == int(path_color):
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue

        candidates.append(
            (
                int(r),
                int(cols[0]),
                colors,
                stripe_cells,
                instr_cells,
                turn_seq,
                int(symbol_colors[0]),
            )
        )

    assert candidates, "stripe legend not found"
    candidates.sort(key=lambda item: (item[0], item[1]))
    _, _, pattern, stripe_cells, instr_cells, turn_seq, symbol_color = candidates[0]
    return pattern, stripe_cells, instr_cells, turn_seq, symbol_color


def _extract_scene(grid: np.ndarray) -> dict[str, object]:
    g = np.array(grid, dtype=int)
    H, W = g.shape

    template_bg = int(g[0, 0])

    counts = Counter(map(int, g.flatten()))
    path_color = int(
        max([(cnt, int(col)) for col, cnt in counts.items() if int(col) != int(template_bg)])[1]
    )
    _assert_path_single_component(g, path_color)

    pattern, pat_cells, instr_cells, turn_seq, symbol_color = _detect_stripe_legend(
        g, template_bg, path_color
    )

    end_color = None
    end_pos = None
    pat_set = set(map(int, pattern))
    for r in range(H - 1):
        for c in range(W - 1):
            v = int(g[r, c])
            if v in (int(template_bg), int(path_color), int(symbol_color)) or v in pat_set:
                continue
            if (
                int(g[r + 1, c]) == v
                and int(g[r, c + 1]) == v
                and int(g[r + 1, c + 1]) == v
            ):
                end_color = int(v)
                end_pos = (int(r), int(c))
                break
        if end_pos is not None:
            break

    markers: list[tuple[int, int, int]] = []
    for r in range(H):
        for c in range(W):
            v = int(g[r, c])
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

    return {
        "g": g,
        "H": H,
        "W": W,
        "template_bg": template_bg,
        "path_color": path_color,
        "pattern": pattern,
        "pat_cells": pat_cells,
        "instr_cells": instr_cells,
        "turn_seq": turn_seq,
        "symbol_color": symbol_color,
        "end_color": end_color,
        "end_pos": end_pos,
        "markers": markers,
    }


def _simulate_scene(scene: dict[str, object], mode: str) -> tuple[np.ndarray, dict[str, object]]:
    g = np.array(scene["g"], dtype=int)
    H = int(scene["H"])
    W = int(scene["W"])
    template_bg = int(scene["template_bg"])
    path_color = int(scene["path_color"])
    pattern = [int(v) for v in scene["pattern"]]
    pat_cells = [(int(r), int(c)) for r, c in scene["pat_cells"]]
    instr_cells = [(int(r), int(c)) for r, c in scene["instr_cells"]]
    turn_seq = [str(v) for v in scene["turn_seq"]]
    end_color = None if scene["end_color"] is None else int(scene["end_color"])
    end_pos = None if scene["end_pos"] is None else tuple(scene["end_pos"])
    markers = [(int(r), int(c), int(v)) for r, c, v in scene["markers"]]

    out = g.copy()
    for r, c in pat_cells + instr_cells:
        out[int(r), int(c)] = int(template_bg)
    if end_pos is not None:
        er, ec = end_pos
        out[int(er) : int(er + 2), int(ec) : int(ec + 2)] = int(template_bg)

    dir_order = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # R, D, L, U
    phase_mod = len(turn_seq) if mode == "legend" else 2

    per_marker_cells: list[set[tuple[int, int]]] = []
    turn_counts: list[int] = []
    used_turns: list[list[str]] = []

    for sr, sc, sv in markers:
        init = None
        for dr, dc in dir_order:
            rr = int(sr + dr)
            cc = int(sc + dc)
            if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(path_color):
                init = (int(dr), int(dc))
                break
        if init is None or not pattern:
            per_marker_cells.append(set())
            turn_counts.append(0)
            used_turns.append([])
            continue

        r, c = int(sr), int(sc)
        dr, dc = init
        idx = int(pattern.index(int(sv))) if int(sv) in pattern else 0
        hits = 0
        visited: set[tuple[int, int, int, int, int, int]] = set()
        my_cells: set[tuple[int, int]] = set()
        my_turns: list[str] = []

        for _ in range(H * W * 4):
            phase = int(hits % phase_mod) if phase_mod else 0
            state = (int(r), int(c), int(dr), int(dc), int(idx), phase)
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
            if mode == "legend":
                assert turn_seq, "turn legend must be non-empty"
                turn = turn_seq[hits % len(turn_seq)]
            else:
                turn = "L" if (hits % 2 == 0) else "R"
            tdr, tdc = left if turn == "L" else right

            rr = int(r + tdr)
            cc = int(c + tdc)
            if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(path_color):
                dr, dc = int(tdr), int(tdc)
                hits += 1
                my_turns.append(turn)
                continue
            break

        per_marker_cells.append(my_cells)
        turn_counts.append(hits)
        used_turns.append(my_turns)

    for i in range(len(per_marker_cells)):
        for j in range(i + 1, len(per_marker_cells)):
            overlap = per_marker_cells[i] & per_marker_cells[j]
            assert not overlap, f"marker trajectories {i},{j} overlap on {len(overlap)} cell(s)"

    return out, {
        "turn_counts": turn_counts,
        "used_turns": used_turns,
        "turn_seq": turn_seq,
        "n_markers": len(markers),
    }


def _fusion_metrics(grid: np.ndarray) -> dict[str, object]:
    """Helper for audit scripts: quantify whether the fused channels matter."""
    scene = _extract_scene(grid)
    legend_out, legend_stats = _simulate_scene(scene, mode="legend")
    alt_out, _ = _simulate_scene(scene, mode="alternate")
    pattern_cols = [int(c) for _, c in scene["pat_cells"]]
    return {
        "pattern_len": len(scene["pattern"]),
        "turn_seq": "".join(scene["turn_seq"]),
        "legend_differs_from_alternating": int(not np.array_equal(legend_out, alt_out)),
        "total_turns": int(sum(int(v) for v in legend_stats["turn_counts"])),
        "max_turns_one_marker": int(max([0] + [int(v) for v in legend_stats["turn_counts"]])),
        "single_row_span": (int(min(pattern_cols)), int(max(pattern_cols))) if pattern_cols else None,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate path-tracing particles whose paint cycle and turn commands come from one compact stripe legend.

    Input structure:
    - Top-left: a single row of 2-4 color cells defines the repeating paint cycle from left to right.
    - Each color cell has a 2-pixel side bar: bar on the left means CCW, bar on the right means CW.
    - Isolated cycle-colored pixels on col 0 are start markers embedded in the path band.
    - A separate 2x2 monochrome block gives the terminal collision color.

    Rule:
    - Each marker moves straight through path cells, repainting them by advancing through the stripe colors cyclically.
    - When forward motion hits a non-path cell, that collision cell is painted with the terminal color if it is background.
    - The particle then reads the next legend instruction in stripe order: left bar => turn CCW, right bar => turn CW.
    - The turn sequence repeats cyclically across collisions; if the requested turn does not lead to path, the particle stops.

    Key insight:
    - The top-left template is now a fused legend: the same compact stripe tells you both which colors repeat and which way each collision should bend.
    """
    scene = _extract_scene(grid)
    out, _ = _simulate_scene(scene, mode="legend")
    return out


def _build_band(
    H: int,
    W: int,
    lobe_rows: list[int] | None = None,
    template_clear_col: int = 16,
) -> np.ndarray:
    """Return a boolean mask for a thick meandering anti-diagonal band."""
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
        lmin = min(lobe_rows) - 1
        lmax = H - 1
        for r in range(H):
            d_outside = max(0, lmin - r, r - lmax)
            c_left[r] = min(c_left[r], 2 * d_outside)

    for r in range(1, min(5, H)):
        c_left[r] = max(c_left[r], template_clear_col)
        if c_right[r] < c_left[r]:
            c_right[r] = c_left[r]

    c_right[0] = W - 1
    c_left[0] = max(c_left[0], W - 6, max(0, template_clear_col - 2))
    c_left[H - 1] = 0
    c_right[H - 1] = max(5, min(W - 1, c_right[H - 1]))

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
    c_left[H - 1] = 0

    mask = np.zeros((H, W), dtype=bool)
    for r in range(H):
        mask[r, c_left[r] : c_right[r] + 1] = True
    return mask


def _carve_fjord(mask: np.ndarray, r1: int, r2: int) -> bool:
    """Carve a bg separator into the left side of the band without disconnecting it."""
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


def _sample_turn_seq(n_slots: int) -> list[str]:
    # Keep both turn directions visible in the legend. Pure-L legends are valid
    # mechanically but under-cover the CW branch and make the symbol channel look cosmetic.
    curated = {
        3: ["LLR", "LRR", "RLL"],
        4: ["LLRL", "LRRL", "LLRR", "LRLL", "LRRR", "RRLL"],
    }
    if n_slots in curated:
        return list(random.choice(curated[n_slots]))

    alt_prefix = ["L" if (i % 2 == 0) else "R" for i in range(n_slots)]
    candidates: list[list[str]] = []
    for bits in range(1 << n_slots):
        seq = ["R" if ((bits >> i) & 1) else "L" for i in range(n_slots)]
        if seq == alt_prefix:
            continue
        if "L" not in seq or "R" not in seq:
            continue
        if any(seq[i] == seq[i - 1] for i in range(1, n_slots)):
            candidates.append(seq)
    assert candidates
    return random.choice(candidates)


def _place_stripe_legend(
    g: np.ndarray,
    mask: np.ndarray,
    cycle_colors: list[int],
    turn_seq: list[str],
    symbol_color: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Paint the fused stripe legend in the reserved top-left pocket."""
    H, W = g.shape
    stripe_row = random.choice([1, 2])
    color_cols = [2 + 4 * i for i in range(len(cycle_colors))]
    if color_cols[-1] + 1 >= W or stripe_row + 1 >= H:
        raise AssertionError("legend does not fit")

    stripe_cells: list[tuple[int, int]] = []
    instr_cells: list[tuple[int, int]] = []
    bg = int(g[0, 0])

    for c, turn in zip(color_cols, turn_seq):
        bar_col = int(c - 1) if turn == "L" else int(c + 1)
        cells = [(stripe_row, c), (stripe_row, bar_col), (stripe_row + 1, bar_col)]
        for rr, cc in cells:
            if not (0 <= rr < H and 0 <= cc < W):
                raise AssertionError("legend out of bounds")
            if mask[rr, cc] or int(g[rr, cc]) != bg:
                raise AssertionError("legend overlaps non-bg")
        stripe_cells.append((int(stripe_row), int(c)))
        instr_cells.extend([(int(stripe_row), int(bar_col)), (int(stripe_row + 1), int(bar_col))])

    for rr, cc in stripe_cells + instr_cells:
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nr = int(rr + dr)
            nc = int(cc + dc)
            if 0 <= nr < H and 0 <= nc < W and mask[nr, nc]:
                raise AssertionError("legend touches path")

    for (rr, cc), col in zip(stripe_cells, cycle_colors):
        g[rr, cc] = int(col)
    for rr, cc in instr_cells:
        g[rr, cc] = int(symbol_color)

    return stripe_cells, instr_cells


def _gen_attempt() -> np.ndarray | None:
    H = W = random.randint(20, 30)
    n_markers = 2 if random.random() < 0.5 else 1

    palette = list(range(10))
    random.shuffle(palette)
    bg = palette[0]
    path_color = palette[1]
    end_color = palette[2]
    symbol_color = palette[3]
    n_slots = random.choice([3, 4])
    cycle_colors = random.sample(palette[4:], n_slots)
    turn_seq = _sample_turn_seq(n_slots)

    if n_markers == 1:
        r1 = random.randint(H // 2, H - 4)
        markers = [r1]
    else:
        r1 = random.randint(H // 3, H // 2)
        r2 = random.randint(max(r1 + 5, 2 * H // 3), H - 4)
        markers = [r1, r2]

    color_cols = [2 + 4 * i for i in range(n_slots)]
    template_clear_col = color_cols[-1] + 2
    mask = _build_band(H, W, lobe_rows=markers, template_clear_col=template_clear_col)
    if mask[0, 0]:
        return None
    _, n_path = label_mask(mask, connectivity=4)
    if n_path != 1:
        return None
    if not mask[0, :].any() or not mask[:, W - 1].any():
        return None
    if not mask[H - 1, :].any() or not mask[:, 0].any():
        return None
    for r in markers:
        if not (mask[r, 0] and mask[r - 1, 0] and mask[r + 1, 0]):
            return None

    if n_markers == 2:
        trial = mask.copy()
        if not _carve_fjord(trial, r1, r2):
            return None
        if random.random() < 0.25:
            K = random.randint(2, 4)
            pre_carve_cells = mask & ~trial
            for c in range(min(K, W)):
                trial[:, c] |= pre_carve_cells[:, c]
        _, nc = label_mask(trial, connectivity=4)
        if nc != 1:
            return None
        if not trial[0, :].any() or not trial[:, W - 1].any():
            return None
        if not trial[H - 1, :].any() or not trial[:, 0].any():
            return None
        for r in markers:
            if not (trial[r, 0] and trial[r - 1, 0] and trial[r + 1, 0]):
                return None
        mask = trial

    _, n_bg = label_mask(~mask, connectivity=4)
    if n_bg < 2:
        return None

    g = np.full((H, W), bg, dtype=int)
    g[mask] = path_color

    try:
        _place_stripe_legend(g, mask, cycle_colors, turn_seq, symbol_color)
    except AssertionError:
        return None

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

    try:
        scene = _extract_scene(g)
        legend_out, legend_stats = _simulate_scene(scene, mode="legend")
        alt_out, _ = _simulate_scene(scene, mode="alternate")
        if np.array_equal(legend_out, alt_out):
            return None
        if sum(int(v) for v in legend_stats["turn_counts"]) < 3:
            return None
        if max([0] + [int(v) for v in legend_stats["turn_counts"]]) < 2:
            return None
        used_turns = "".join("".join(turns) for turns in legend_stats["used_turns"])
        if "L" not in used_turns or "R" not in used_turns:
            return None
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
