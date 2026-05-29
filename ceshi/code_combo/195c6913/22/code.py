# Combo variant of 195c6913 (index 22)
# combo_spec: D2/2+D4/4+D11/1
# Name: 符号转向指令+箭头标记+模板方向符号

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import label_mask

# Design analysis for the combo variant
# -------------------------------------
# a) Concrete fused encoding and behavior trace
# - Keep the original anti-diagonal path band and the top-left row of 2x2 color
#   blocks. Those blocks still define the recolor cycle.
# - Add one vertical 2x1 side bar beside every 2x2 block:
#   left side => CCW turn, right side => CW turn. Collision turns read these bars
#   cyclically in block order, so the template now visibly stores a turn program.
# - Replace the old single-pixel sources with 3-cell arrowheads in the first cycle
#   color. The arrow tip is the launch cell.
# - Add one extra template glyph after the last block: an anchor pixel plus one
#   same-colored indicator pixel in one of the four cardinal directions. This
#   encodes a global launch heading.
# - Fusion choice for the D4/D11 tension: every arrowhead must point in the same
#   direction as the template glyph. The solver decodes both and requires them to
#   agree before simulating. Example: if the glyph points Down and both markers are
#   `v` arrowheads, motion starts downward; if the turn bars read [L,R,R], later
#   collisions use CCW,CW,CW,CCW,... rather than alternating by parity.
# - Relative to the original task, three assumptions change together: the source is
#   no longer a single pixel, the initial heading is no longer inferred from local
#   path adjacency, and collision turns are no longer parity-driven.
# b) Complexity calibration
# - The added symbols all use low-cardinality, human-visible geometry: arrow shape,
#   left/right bar placement, and one-cardinal-step glyph offset. There is no
#   hidden ordering by color id or pixel count.
# - The generator keeps only 2-4 cycle blocks and 1-2 markers, so the scene stays
#   readable even though both source and template now carry directional cues.
# - All markers in one grid share the same glyph direction. This avoids exploding
#   the hypothesis space while still leaving multiple examples across the dataset
#   to show that the shared heading can vary between samples.
# c) Hidden-assumption audit against the original solver
# - Original marker detection assumed isolated size-1 cells on col 0. That is
#   replaced by 8-connected size-3 arrowhead decoding.
# - Original launch direction used neighbor priority R,D,L,U. That would erase the
#   new symbolic cues, so the solver now decodes the arrowheads and template glyph
#   first, then asserts consistency.
# - Original turn state tracked only `hits & 1`. Because the turn program is now a
#   legend sequence, loop detection must track `hits % len(turn_seq)` instead.
# - Original template erasure removed only the 2x2 blocks. The combo must also
#   erase side bars and the global direction glyph.
# d) Placement legality
# - Blocks start at columns 1,6,11,16. Each occupies 2 columns; its side bar uses
#   either the column just left or just right of that block. The global glyph sits
#   4 columns after the final block start, so it never overlaps the last side bar.
# - The path generator reserves rows 1..3 through the glyph area. That leaves clean
#   background for both side bars and the up/down/left/right indicator positions.
# - Arrowheads are placed only on ambient path cells, with the tip's forward cell
#   also path and with Chebyshev separation between markers, so components stay
#   decodable and disjoint.
# e) Visual observability
# - A human can read three independent visual cues from the input: block colors are
#   the paint cycle, side bars are the turn sequence, and the arrow/glyph direction
#   explains the initial launch heading.
# - The generator rejects cases where the legend turns collapse to the old strict
#   alternation or where the symbolic launch heading matches the old R,D,L,U
#   priority. That keeps the fused behavior visible in the output rather than merely
#   decorative.
# f) Visual cleanliness
# - All extra symbols live in the original semantic zones: directional bars and the
#   glyph stay in the top-left template region, while arrowheads stay inside the
#   left-side path lobe. No object needs to overlap another object's footprint.
# - Using one distinct symbol color for all template instructions makes the template
#   read as "color blocks + instruction marks" rather than as many unrelated colors.
# g) Inferability and new solver assumptions
# - The first cycle color still marks the starting phase; no extra color mapping is
#   introduced beyond "instruction marks share one symbol color".
# - New assumptions: each block has exactly one side bar, the post-template glyph
#   has exactly one indicator, the glyph direction matches every arrow marker, and
#   all placed arrows are non-degenerate against legacy neighbor-priority launch.
# - The generator constructs and validates those assumptions before returning.

DIR4 = {
    "R": (0, 1),
    "D": (1, 0),
    "L": (0, -1),
    "U": (-1, 0),
}
DIR_ORDER = ["R", "D", "L", "U"]
ARROW_SHAPES = {
    "R": ((0, 0), (-1, -1), (1, -1)),
    "L": ((0, 0), (-1, 1), (1, 1)),
    "U": ((0, 0), (1, -1), (1, 1)),
    "D": ((0, 0), (-1, -1), (-1, 1)),
}


def _assert_path_single_component(g: np.ndarray, path_color: int) -> None:
    """Eval invariant: path_color forms a single 4-connected component."""
    _, n_path = label_mask(g == path_color, connectivity=4)
    assert n_path == 1, f"path_color must be a single 4-connected component, got {n_path}"


def _arrow_cells(tip_r: int, tip_c: int, direction: str) -> list[tuple[int, int]]:
    return [
        (int(tip_r + dr), int(tip_c + dc))
        for dr, dc in ARROW_SHAPES[str(direction)]
    ]


def _decode_arrow_component(coords: np.ndarray) -> tuple[int, int, str] | None:
    if int(coords.shape[0]) != 3:
        return None
    comp = {(int(r), int(c)) for r, c in coords}
    matches: list[tuple[int, int, str]] = []
    for direction in DIR_ORDER:
        for tip_r, tip_c in comp:
            if set(_arrow_cells(int(tip_r), int(tip_c), direction)) == comp:
                matches.append((int(tip_r), int(tip_c), direction))
    if len(matches) != 1:
        return None
    return matches[0]


def _priority_dir_grid(g: np.ndarray, r: int, c: int, path_color: int) -> str | None:
    H, W = g.shape
    for direction in DIR_ORDER:
        dr, dc = DIR4[direction]
        rr = int(r + dr)
        cc = int(c + dc)
        if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(path_color):
            return direction
    return None


def _priority_dir_mask(mask: np.ndarray, r: int, c: int) -> str | None:
    H, W = mask.shape
    for direction in DIR_ORDER:
        dr, dc = DIR4[direction]
        rr = int(r + dr)
        cc = int(c + dc)
        if 0 <= rr < H and 0 <= cc < W and bool(mask[rr, cc]):
            return direction
    return None


def _glyph_indicator(anchor: tuple[int, int], direction: str) -> tuple[int, int]:
    ar, ac = anchor
    dr, dc = DIR4[direction]
    return (int(ar + dr), int(ac + dc))


def _joint_symbolic_dir(marker_dir: str, global_dir: str) -> str:
    assert str(marker_dir) == str(global_dir), (
        f"arrow direction {marker_dir} must agree with template glyph {global_dir}"
    )
    return str(marker_dir)


def _find_arrow_markers(
    g: np.ndarray, marker_color: int, path_color: int
) -> list[tuple[int, int, int, str, tuple[tuple[int, int], ...]]]:
    H, W = g.shape
    markers: list[tuple[int, int, int, str, tuple[tuple[int, int], ...]]] = []
    lab, n_comp = label_mask(g == int(marker_color), connectivity=8)
    for k in range(1, int(n_comp) + 1):
        coords = np.argwhere(lab == k)
        decoded = _decode_arrow_component(coords)
        if decoded is None:
            continue
        tip_r, tip_c, direction = decoded
        dr, dc = DIR4[direction]
        nr, nc = int(tip_r + dr), int(tip_c + dc)
        assert 0 <= nr < H and 0 <= nc < W, (
            f"marker tip at ({tip_r},{tip_c}) points {direction} out of bounds"
        )
        assert int(g[nr, nc]) == int(path_color), (
            f"marker tip at ({tip_r},{tip_c}) points to ({nr},{nc})={int(g[nr, nc])}, "
            f"expected path_color {path_color}"
        )
        cells = tuple(sorted((int(r), int(c)) for r, c in coords.tolist()))
        markers.append((int(tip_r), int(tip_c), int(marker_color), direction, cells))
    markers.sort(key=lambda x: (x[0], x[1], x[3]))
    return markers


def _assert_arrow_markers_valid(
    g: np.ndarray,
    markers: list[tuple[int, int, int, str, tuple[tuple[int, int], ...]]],
    path_color: int,
    global_dir: str,
) -> None:
    H, W = g.shape
    assert 1 <= len(markers) <= 2, f"expected 1-2 arrow markers, got {len(markers)}"
    used: set[tuple[int, int]] = set()
    for tip_r, tip_c, color, direction, cells in markers:
        assert direction == str(global_dir), (
            f"marker at ({tip_r},{tip_c}) points {direction}, expected {global_dir}"
        )
        assert tuple(sorted(_arrow_cells(tip_r, tip_c, direction))) == cells
        dr, dc = DIR4[direction]
        nr, nc = int(tip_r + dr), int(tip_c + dc)
        assert 0 <= nr < H and 0 <= nc < W and int(g[nr, nc]) == int(path_color)
        for r, c in cells:
            assert 0 <= r < H and 0 <= c < W
            assert int(g[r, c]) == int(color)
            assert (int(r), int(c)) not in used, "marker cells must be disjoint"
            used.add((int(r), int(c)))


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
    block_origins: list[tuple[int, int]] = []
    pat_cells: list[tuple[int, int]] = []
    instr_cells: list[tuple[int, int]] = []
    turn_seq: list[str] = []
    symbol_color: int | None = None
    seen: set[tuple[int, int]] = set()

    for r in range(min(6, H - 1)):
        for c in range(min(26, W - 1)):
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
                block_origins.append((int(r), int(c)))
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
                bar_col = int(c - 1) if left_bar else int(c + 2)
                bar_color = int(g[int(r), bar_col])
                if symbol_color is None:
                    symbol_color = bar_color
                assert int(symbol_color) == bar_color, "all instruction symbols must share one color"
                turn_seq.append("L" if left_bar else "R")
                instr_cells.extend([(int(r), bar_col), (int(r + 1), bar_col)])

    assert pattern, "template legend must contain at least one 2x2 cycle block"
    assert len(turn_seq) == len(pattern), "every cycle block must contribute one turn instruction"
    assert symbol_color is not None, "instruction bars must exist"

    template_row = min(r for r, _ in block_origins)
    anchor_c = max(c for _, c in block_origins) + 4
    anchor = (int(template_row), int(anchor_c))
    assert 0 <= anchor[0] < H and 0 <= anchor[1] < W, "direction glyph anchor out of bounds"
    assert int(g[anchor]) == int(symbol_color), "direction glyph anchor must use the instruction color"

    glyph_cells = [anchor]
    glyph_dirs: list[str] = []
    for direction in DIR_ORDER:
        rr, cc = _glyph_indicator(anchor, direction)
        if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(symbol_color):
            glyph_dirs.append(direction)
            glyph_cells.append((int(rr), int(cc)))
    assert len(glyph_dirs) == 1, "direction glyph must have exactly one indicator"
    global_dir = glyph_dirs[0]

    end_color = None
    end_pos = None
    pat_set = set(map(int, pattern))
    for r in range(H - 1):
        for c in range(W - 1):
            v = int(g[int(r), int(c)])
            if v in {int(template_bg), int(path_color), int(symbol_color)} or v in pat_set:
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

    markers = _find_arrow_markers(g, int(pattern[0]), int(path_color))
    _assert_arrow_markers_valid(g, markers, int(path_color), str(global_dir))

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
        "glyph_cells": glyph_cells,
        "global_dir": global_dir,
        "end_color": end_color,
        "end_pos": end_pos,
        "markers": markers,
    }


def _simulate_scene(
    scene: dict[str, object], turn_mode: str, init_mode: str
) -> tuple[np.ndarray, dict[str, object]]:
    g = np.array(scene["g"], dtype=int)
    H = int(scene["H"])
    W = int(scene["W"])
    template_bg = int(scene["template_bg"])
    path_color = int(scene["path_color"])
    pattern = [int(v) for v in scene["pattern"]]
    pat_cells = [(int(r), int(c)) for r, c in scene["pat_cells"]]
    instr_cells = [(int(r), int(c)) for r, c in scene["instr_cells"]]
    glyph_cells = [(int(r), int(c)) for r, c in scene["glyph_cells"]]
    turn_seq = [str(v) for v in scene["turn_seq"]]
    global_dir = str(scene["global_dir"])
    end_color = None if scene["end_color"] is None else int(scene["end_color"])
    end_pos = None if scene["end_pos"] is None else tuple(scene["end_pos"])
    markers = [
        (int(r), int(c), int(v), str(d), tuple(cells))
        for r, c, v, d, cells in scene["markers"]
    ]

    out = g.copy()
    for r, c in pat_cells + instr_cells + glyph_cells:
        out[int(r), int(c)] = int(template_bg)
    if end_pos is not None:
        er, ec = end_pos
        out[int(er) : int(er + 2), int(ec) : int(ec + 2)] = int(template_bg)

    per_marker_cells: list[set[tuple[int, int]]] = []
    turn_counts: list[int] = []
    interesting_markers = 0

    for sr, sc, sv, marker_dir, _cells in markers:
        priority_dir = _priority_dir_grid(g, int(sr), int(sc), int(path_color))
        if init_mode == "symbolic":
            init_dir = _joint_symbolic_dir(str(marker_dir), str(global_dir))
        else:
            init_dir = priority_dir
        if priority_dir is not None and priority_dir != marker_dir:
            interesting_markers += 1
        if init_dir is None or not pattern:
            per_marker_cells.append(set())
            turn_counts.append(0)
            continue

        r, c = int(sr), int(sc)
        dr, dc = DIR4[str(init_dir)]
        idx = int(pattern.index(int(sv))) if int(sv) in pattern else 0
        hits = 0
        phase_mod = len(turn_seq) if turn_mode == "legend" else 2
        visited: set[tuple[int, int, int, int, int, int]] = set()
        my_cells: set[tuple[int, int]] = set()

        for _ in range(H * W * 4):
            phase = int(hits % phase_mod) if phase_mod else 0
            state = (int(r), int(c), int(dr), int(dc), int(idx), int(phase))
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
            if turn_mode == "legend":
                turn = turn_seq[hits % len(turn_seq)]
            else:
                turn = "L" if (hits % 2 == 0) else "R"
            tdr, tdc = left if turn == "L" else right
            rr = int(r + tdr)
            cc = int(c + tdc)
            if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(path_color):
                dr, dc = int(tdr), int(tdc)
                hits += 1
                continue
            break

        per_marker_cells.append(my_cells)
        turn_counts.append(int(hits))

    for i in range(len(per_marker_cells)):
        for j in range(i + 1, len(per_marker_cells)):
            overlap = per_marker_cells[i] & per_marker_cells[j]
            assert not overlap, f"marker trajectories {i},{j} overlap on {len(overlap)} cell(s)"

    return out, {
        "turn_counts": turn_counts,
        "interesting_markers": interesting_markers,
        "global_dir": global_dir,
        "turn_seq": turn_seq,
        "n_markers": len(markers),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate recoloring particles on a path using arrow-shaped sources plus template direction symbols.

    Input structure:
    - Top-left: a row of 2x2 monochrome blocks gives the repeating paint cycle, and each block has a vertical side bar on its left or right that encodes a turn instruction.
    - After the last block, one extra two-pixel glyph in the same symbol color encodes a shared launch direction for all particles.
    - Start markers are 3-cell arrowheads in the first cycle color, embedded inside the path band, and a separate 2x2 block provides the terminal collision color.

    Rule:
    - Decode the cycle from the 2x2 blocks, decode the turn sequence from the side bars, and decode the launch direction from the template glyph.
    - Decode each arrow marker and require its pointing direction to agree with the template glyph; each particle starts from the arrow tip in that shared direction.
    - While the next cell is path, advance and repaint path cells by cycling through the template colors; on a blocked forward step, mark the collision cell with the end color only when it is background, then execute the next legend turn (CCW for a left bar, CW for a right bar).
    - If the requested legend turn is not path, stop immediately; erase all template blocks, side bars, the direction glyph, and the end block from the output.

    Key insight:
    - The template now carries two independent symbolic programs at once: block colors specify the repaint cycle, and instruction marks specify both the shared launch heading and the collision-turn schedule, while the arrowheads localize the actual start tips and confirm the heading geometrically.
    """
    scene = _extract_scene(grid)
    out, _ = _simulate_scene(scene, turn_mode="legend", init_mode="symbolic")
    return out


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
            width = max(2, min(W, int(round(2 * half_w[r] + w_jit[r]))))
            cc = centers[r] + cc_jit[r]
            c_left[r] = int(round(cc - width / 2))
            c_right[r] = c_left[r] + width - 1

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
    H, W = mask.shape
    sep_h = random.randint(2, 3)
    sr_min = r1 + 2
    sr_max = r2 - 1 - sep_h
    if sr_max < sr_min:
        return False
    cur_top = random.randint(sr_min, sr_max)
    made_carve = False
    for k in range(W - 1):
        trial = mask.copy()
        for dr in range(sep_h):
            rr = int(cur_top + dr)
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
    for bits in range(1 << n_blocks):
        seq = ["R" if ((bits >> i) & 1) else "L" for i in range(n_blocks)]
        if seq == alt_prefix:
            continue
        if any(seq[i] == seq[i - 1] for i in range(1, n_blocks)):
            candidates.append(seq)
    assert candidates
    return random.choice(candidates)


def _chebyshev_separated(
    cells_a: tuple[tuple[int, int], ...], cells_b: tuple[tuple[int, int], ...]
) -> bool:
    for r1, c1 in cells_a:
        for r2, c2 in cells_b:
            if max(abs(int(r1) - int(r2)), abs(int(c1) - int(c2))) <= 1:
                return False
    return True


def _collect_arrow_candidates(mask: np.ndarray, target_row: int) -> list[dict[str, object]]:
    H, W = mask.shape
    col_hi = max(6, W // 3)
    row_ranges = [
        range(max(1, target_row - 3), min(H - 1, target_row + 4)),
        range(1, H - 1),
    ]
    for row_range in row_ranges:
        candidates: list[dict[str, object]] = []
        for direction in DIR_ORDER:
            dr, dc = DIR4[direction]
            for r in row_range:
                for c in range(1, min(W - 1, col_hi + 1)):
                    cells = _arrow_cells(int(r), int(c), direction)
                    if any(not (0 <= rr < H and 0 <= cc < W) for rr, cc in cells):
                        continue
                    if not all(bool(mask[rr, cc]) for rr, cc in cells):
                        continue
                    nr, nc = int(r + dr), int(c + dc)
                    if not (0 <= nr < H and 0 <= nc < W and bool(mask[nr, nc])):
                        continue
                    priority = _priority_dir_mask(mask, int(r), int(c))
                    if priority is None or priority == direction:
                        continue
                    local = mask[
                        max(0, r - 1) : min(H, r + 2),
                        max(0, c - 1) : min(W, c + 2),
                    ]
                    if int(local.sum()) < 5:
                        continue
                    candidates.append(
                        {
                            "tip": (int(r), int(c)),
                            "direction": direction,
                            "cells": tuple(sorted(cells)),
                            "row_gap": abs(int(r) - int(target_row)),
                            "col": int(c),
                        }
                    )
        if candidates:
            random.shuffle(candidates)
            candidates.sort(key=lambda cand: (int(cand["row_gap"]), int(cand["col"])))
            return candidates[:80]
    return []


def _place_arrow_markers(
    g: np.ndarray, mask: np.ndarray, target_rows: list[int], marker_color: int
) -> list[dict[str, object]] | None:
    cand_lists = [_collect_arrow_candidates(mask, int(row)) for row in target_rows]
    if any(not cands for cands in cand_lists):
        return None

    if len(cand_lists) == 1:
        direction_buckets: dict[str, list[dict[str, object]]] = {d: [] for d in DIR_ORDER}
        for cand in cand_lists[0]:
            direction_buckets[str(cand["direction"])].append(cand)
        nonempty = [d for d in DIR_ORDER if direction_buckets[d]]
        if not nonempty:
            return None
        direction = random.choice(nonempty)
        pool = direction_buckets[direction][: min(12, len(direction_buckets[direction]))]
        choice = random.choice(pool)
        for r, c in choice["cells"]:
            g[int(r), int(c)] = int(marker_color)
        return [choice]

    pairs: list[dict[str, object]] = []
    for cand_a in cand_lists[0]:
        for cand_b in cand_lists[1]:
            if str(cand_a["direction"]) != str(cand_b["direction"]):
                continue
            if not _chebyshev_separated(cand_a["cells"], cand_b["cells"]):
                continue
            pairs.append(
                {
                    "a": cand_a,
                    "b": cand_b,
                    "direction": str(cand_a["direction"]),
                    "row_gap": int(cand_a["row_gap"]) + int(cand_b["row_gap"]),
                    "col_sum": int(cand_a["col"]) + int(cand_b["col"]),
                }
            )
    if not pairs:
        return None
    by_direction: dict[str, list[dict[str, object]]] = {d: [] for d in DIR_ORDER}
    for pair in pairs:
        by_direction[str(pair["direction"])].append(pair)
    available_dirs = [d for d in DIR_ORDER if by_direction[d]]
    direction = random.choice(available_dirs)
    pool = by_direction[direction]
    random.shuffle(pool)
    pool.sort(key=lambda item: (int(item["row_gap"]), int(item["col_sum"])))
    pick = random.choice(pool[: min(16, len(pool))])
    placed = [pick["a"], pick["b"]]
    for cand in placed:
        for r, c in cand["cells"]:
            g[int(r), int(c)] = int(marker_color)
    return placed


def _gen_attempt() -> np.ndarray | None:
    H = W = random.randint(20, 30)
    n_markers = 2 if random.random() < 0.7 else 1

    max_blocks = min(4, (W - 2) // 5)
    if max_blocks < 2:
        return None
    n_blocks = random.randint(2, max_blocks)

    colors = random.sample(list(range(10)), 4 + n_blocks)
    bg = colors[0]
    path_color = colors[1]
    end_color = colors[2]
    symbol_color = colors[3]
    cycle_colors = colors[4:]
    turn_seq = _sample_turn_seq(n_blocks)

    if n_markers == 1:
        r1 = random.randint(H // 2, H - 4)
        target_rows = [r1]
    else:
        r1 = random.randint(H // 3, H // 2)
        r2 = random.randint(max(r1 + 5, 2 * H // 3), H - 4)
        target_rows = [r1, r2]

    block_cols = [1 + 5 * i for i in range(n_blocks)]
    template_row = 2
    anchor = (template_row, block_cols[-1] + 4)
    template_clear_col = anchor[1] + 2
    mask = _build_band(H, W, lobe_rows=target_rows, template_clear_col=template_clear_col)
    if mask[0, 0]:
        return None
    _, n_path = label_mask(mask, connectivity=4)
    if n_path != 1:
        return None
    if not mask[0, :].any() or not mask[:, W - 1].any():
        return None
    if not mask[H - 1, :].any() or not mask[:, 0].any():
        return None

    if n_markers == 2:
        trial = mask.copy()
        if not _carve_fjord(trial, int(target_rows[0]), int(target_rows[1])):
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
        mask = trial

    _, n_bg = label_mask(~mask, connectivity=4)
    if n_bg < 2:
        return None

    g = np.full((H, W), bg, dtype=int)
    g[mask] = path_color

    for i, (bc, turn) in enumerate(zip(block_cols, turn_seq)):
        bar_col = bc - 1 if turn == "L" else bc + 2
        for rr in range(template_row, template_row + 2):
            for cc in [bc, bc + 1, bar_col]:
                if not (0 <= rr < H and 0 <= cc < W) or int(g[rr, cc]) != int(bg):
                    return None
        g[template_row : template_row + 2, bc : bc + 2] = int(cycle_colors[i])
        g[template_row : template_row + 2, bar_col] = int(symbol_color)

    end_candidates = []
    for r in range(H // 2, H - 1):
        for c in range(W // 2, W - 1):
            if not (g[r : r + 2, c : c + 2] == bg).all():
                continue
            r0, r1_ = max(0, r - 1), min(H, r + 3)
            c0, c1_ = max(0, c - 1), min(W, c + 3)
            if (g[r0:r1_, c0:c1_] == path_color).any():
                continue
            end_candidates.append((int(r), int(c)))
    if not end_candidates:
        return None
    er, ec = random.choice(end_candidates)
    g[er : er + 2, ec : ec + 2] = int(end_color)

    placed = _place_arrow_markers(g, mask, target_rows, int(cycle_colors[0]))
    if placed is None:
        return None
    global_dir = str(placed[0]["direction"])
    if any(str(cand["direction"]) != global_dir for cand in placed):
        return None

    indicator = _glyph_indicator(anchor, global_dir)
    glyph_cells = [anchor, indicator]
    for rr, cc in glyph_cells:
        if not (0 <= rr < H and 0 <= cc < W):
            return None
        if int(g[rr, cc]) != int(bg):
            return None
    g[anchor] = int(symbol_color)
    g[indicator] = int(symbol_color)

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
        combo_out, combo_stats = _simulate_scene(scene, turn_mode="legend", init_mode="symbolic")
        alt_out, _ = _simulate_scene(scene, turn_mode="alternate", init_mode="symbolic")
        priority_out, priority_stats = _simulate_scene(scene, turn_mode="legend", init_mode="priority")
        if np.array_equal(combo_out, alt_out):
            return None
        if np.array_equal(combo_out, priority_out):
            return None
        if int(combo_stats["interesting_markers"]) != int(combo_stats["n_markers"]):
            return None
        if sum(int(v) for v in combo_stats["turn_counts"]) < 3:
            return None
        if max([0] + [int(v) for v in combo_stats["turn_counts"]]) < 2:
            return None
        if str(combo_stats["global_dir"]) == "R":
            return None
        if priority_stats["interesting_markers"] != combo_stats["interesting_markers"]:
            return None
    except Exception:
        return None

    return g


def generate_input() -> np.ndarray:
    for _ in range(400):
        try:
            g = _gen_attempt()
        except Exception:
            g = None
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 400 attempts")
