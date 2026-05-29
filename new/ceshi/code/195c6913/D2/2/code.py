# Variant of 195c6913
# Dimension: D2 — 碰撞转弯规则：严格交替左右转 (path)
# Change: 碰撞时第1/3/5次左转(CCW)，第2/4/6次右转(CW)，严格交替；若计划转向方向不是path则停止（无后备方案）... → 符号指令指定转向（图例编码）

from __future__ import annotations

from collections import Counter
import random
import numpy as np

from arc_utils import label_mask

# Design analysis for the D2 variant
# ----------------------------------
# a) Concrete encoding and behavior trace:
#    - Keep the original top-left 2x2 monochrome color blocks as the paint cycle.
#    - Add one 2x1 vertical "instruction bar" next to every block:
#      bar on the left  => CCW turn, bar on the right => CW turn.
#    - The particle still paints path cells by advancing through the block colors,
#      but collision turns now read the side-bar sequence cyclically in block order.
#    - Example: three blocks [A,B,C] with bars [left,right,right] mean color cycle
#      A->B->C->A... and turn sequence CCW,CW,CW,CCW,...
#      Original logic at the second collision would force CW only because of parity;
#      the variant instead obeys the second legend bar, so repeated turns or any
#      later departure from strict alternation visibly alter the trajectory.
# b) Complexity calibration:
#    - The symbol is positional rather than color-based, so the human-visible cue is
#      local and repeated 2-4 times. The task remains one legend row plus one path.
#    - The turn sequence is low-cardinality (only CCW/CW), preventing overfitting to
#      arbitrary colors, while still allowing non-alternating patterns that differ
#      clearly from the original behavior.
# c) Old assumptions that break under the variant:
#    - Old `hits & 1` state only tracked alternating parity; now loop detection must
#      track `hits % len(turn_seq)`.
#    - Old template erasure removed only 2x2 blocks; now symbol-bar cells must also
#      be erased or the output would leak legend instructions.
#    - Old template placement stride 3 left no room for side symbols; generator must
#      widen the layout while keeping the rest of the task unchanged.
# d) Placement legality:
#    - Blocks use start columns {1,6,11,16}; each occupies 2 cols, and its left/right
#      bar occupies one adjacent separator column, so objects never overlap.
#    - The path builder reserves rows 1-3 through the last legend column so bars and
#      blocks always sit on background, never on path.
#    - End block, markers, fjord/island separator, and single-component path keep the
#      original constraints and are still validated by the solver oracle.
# e) Visual observability:
#    - The turn instruction is directly visible next to each cycle block; no hidden
#      use of color ids or counts is needed.
#    - Generator rejects samples whose output matches the old alternating rule, so the
#      new legend really controls the observed behavior instead of being decorative.
# f) Visual cleanliness:
#    - Only one additional narrow bar is attached to each template block; the rest of
#      the canvas remains the original broad path + isolated markers composition.
#    - Template and path stay spatially separated by the reserved top-left clearance.
# g) Color-change inferability:
#    - Colors still signal only the paint cycle and the end marker, exactly as before.
#      The new turn rule is encoded by left/right placement, not by hidden color maps.
# h) New solver assumptions and generator guarantees:
#    - Every template block must have exactly one adjacent instruction bar.
#    - All bars lie on the same two rows as their block, and no other 2x2 blocks appear
#      in the top-left legend region.
#    - Generator constructs and validates those conditions before calling `main()`.


def _assert_path_single_component(g: np.ndarray, path_color: int) -> None:
    """Eval invariant: path_color forms a single 4-connected component."""
    _, n_path = label_mask(g == path_color, connectivity=4)
    assert n_path == 1, f"path_color must be a single 4-connected component, got {n_path}"


def _assert_marker_embedded_in_path(
    g: np.ndarray, markers: list[tuple[int, int, int]], path_color: int
) -> None:
    """Eval invariant: each col-0 marker remains embedded inside the path band."""
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


def _extract_scene(grid: np.ndarray) -> dict[str, object]:
    g = np.array(grid, dtype=int)
    H, W = g.shape

    template_bg = int(g[0, 0])

    counts = Counter(map(int, g.flatten()))
    path_color = int(
        max(
            [(cnt, int(col)) for col, cnt in counts.items() if int(col) != int(template_bg)]
        )[1]
    )
    _assert_path_single_component(g, path_color)

    pattern: list[int] = []
    pat_cells: list[tuple[int, int]] = []
    instr_cells: list[tuple[int, int]] = []
    turn_seq: list[str] = []
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

                left_bar = (
                    c - 1 >= 0
                    and int(g[int(r), int(c - 1)]) == int(g[int(r + 1), int(c - 1)])
                    and int(g[int(r), int(c - 1)]) not in (int(template_bg), int(path_color), int(v))
                )
                right_bar = (
                    c + 2 < W
                    and int(g[int(r), int(c + 2)]) == int(g[int(r + 1), int(c + 2)])
                    and int(g[int(r), int(c + 2)]) not in (int(template_bg), int(path_color), int(v))
                )
                assert left_bar != right_bar, f"template block at {(r, c)} must have exactly one side bar"
                if left_bar:
                    turn_seq.append("L")
                    instr_cells.extend([(int(r), int(c - 1)), (int(r + 1), int(c - 1))])
                else:
                    turn_seq.append("R")
                    instr_cells.extend([(int(r), int(c + 2)), (int(r + 1), int(c + 2))])

    assert pattern, "template legend must contain at least one 2x2 cycle block"
    assert len(turn_seq) == len(pattern), "every cycle block must contribute one turn instruction"

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
        my_used_turns: list[str] = []

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
                tdr, tdc = left if turn == "L" else right
            else:
                turn = "L" if (hits % 2 == 0) else "R"
                tdr, tdc = left if turn == "L" else right

            rr = int(r + tdr)
            cc = int(c + tdc)
            if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(path_color):
                dr, dc = int(tdr), int(tdc)
                hits += 1
                my_used_turns.append(turn)
                continue
            break

        per_marker_cells.append(my_cells)
        turn_counts.append(hits)
        used_turns.append(my_used_turns)

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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace cycle-colored particles across a path using legend-specified turns.

    Input structure:
    - Top-left: a row of 2x2 monochrome blocks gives the paint cycle, and each block has a vertical side bar that encodes a turn instruction.
    - Isolated single pixels in a cycle color act as start markers embedded on the left edge of the path band.
    - A separate 2x2 monochrome block provides the terminal collision color.

    Rule:
    - Each marker moves straight through adjacent path cells, repainting them by advancing through the block colors cyclically.
    - When the forward step is blocked, paint that collision cell with the terminal color if it is background, then read the next side-bar instruction.
    - A bar on the block's left means turn CCW; a bar on the block's right means turn CW.
    - The turn instructions are read cyclically in template order, and if the requested turn is not path the particle stops immediately.

    Key insight:
    - The template is now a two-channel legend: block colors define the paint cycle, while left/right side bars independently define the collision-turn sequence.
    """
    scene = _extract_scene(grid)
    out, _ = _simulate_scene(scene, mode="legend")
    return out


# --------------- generator (legend-driven turns + non-degenerate oracle filter) ---------------


def _build_band(
    H: int,
    W: int,
    lobe_rows: list[int] | None = None,
    template_clear_col: int = 12,
) -> np.ndarray:
    """Return a boolean mask for a thick meandering anti-diagonal band."""
    bot_center = random.randint(0, W // 4)
    centers = np.linspace(W - 1, bot_center, H).astype(float)
    half_w = np.array([
        0.5 * (3 + (1.0 - abs(2.0 * r / max(1, H - 1) - 1.0)) * (W - 6))
        for r in range(H)
    ])

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

    for r in range(1, min(4, H)):
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
    """Carve a bg separator biting into the band from col 0."""
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


def _sample_turn_seq(n_blocks: int) -> list[str]:
    alt_prefix = ["L" if (i % 2 == 0) else "R" for i in range(n_blocks)]
    candidates: list[list[str]] = []
    for bits in range(1 << max(0, n_blocks - 1)):
        seq = ["L"] + ["R" if ((bits >> i) & 1) else "L" for i in range(n_blocks - 1)]
        if seq == alt_prefix:
            continue
        if any(seq[i] == seq[i - 1] for i in range(1, n_blocks)):
            candidates.append(seq)
    assert candidates
    return random.choice(candidates)


def _gen_attempt() -> np.ndarray | None:
    H = W = random.randint(20, 30)
    n_markers = 2 if random.random() < 0.7 else 1

    palette = list(range(10))
    random.shuffle(palette)
    bg = palette[0]
    path_color = palette[1]
    end_color = palette[2]
    symbol_color = palette[3]
    n_blocks = random.randint(2, 4)
    cycle_pool = palette[4:]
    cycle_colors = [random.choice(cycle_pool) for _ in range(n_blocks)]
    turn_seq = _sample_turn_seq(n_blocks)

    if n_markers == 1:
        r1 = random.randint(H // 2, H - 4)
        markers = [r1]
    else:
        r1 = random.randint(H // 3, H // 2)
        r2 = random.randint(max(r1 + 5, 2 * H // 3), H - 4)
        markers = [r1, r2]

    block_cols = [1 + 5 * i for i in range(n_blocks)]
    template_clear_col = block_cols[-1] + 3
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

    template_row = random.choice([1, 2])
    for bc, turn in zip(block_cols, turn_seq):
        bar_col = bc - 1 if turn == "L" else bc + 2
        for rr in range(template_row, template_row + 2):
            for cc in [bc, bc + 1, bar_col]:
                if not (0 <= rr < H and 0 <= cc < W) or mask[rr, cc]:
                    return None

    g = np.full((H, W), bg, dtype=int)
    g[mask] = path_color

    for i, (bc, turn) in enumerate(zip(block_cols, turn_seq)):
        g[template_row : template_row + 2, bc : bc + 2] = cycle_colors[i]
        bar_col = bc - 1 if turn == "L" else bc + 2
        g[template_row : template_row + 2, bar_col] = symbol_color

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
