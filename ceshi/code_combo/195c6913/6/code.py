# Combo variant of 195c6913 (index 6)
# combo_spec: D7/2+D13/0
# Name: 模板移至右下角+保留模板不擦除
#
# Design analysis
# - concrete_behavior:
#   - Base task: a row of 2x2 cycle blocks lives in the upper-left pocket, a
#     distinct 2x2 end block lives in the lower-right pocket, and both are
#     erased in the output after the path-walk painting is done.
#   - Combo task: these two anchors swap pockets, so the cycle template is now a
#     bottom-right horizontal row of 2x2 blocks and the end block moves to the
#     top-left background pocket. The particle dynamics stay unchanged.
#   - New output difference: the solver must still use the bottom-right template
#     as its instruction source, but unlike the base task it leaves those 2x2
#     blocks visible in the output. Only the top-left end block is erased.
#   - Example trace: on a valid input, the old solver would search the wrong
#     corner for the cycle and would also remove any detected template cells. The
#     combo solver instead reads the lower-right blocks left-to-right, finds the
#     top-left end color block, paints trajectories, stamps collision pixels with
#     the end color, erases only the end block, and keeps the template intact.
# - complexity_calibration:
#   - The cycle encoding remains the original 2-4 monochrome 2x2 blocks with
#     1-cell background separators, so the instruction system is still easy to
#     infer from a few samples.
#   - The corner inversion is made visually strong by right-aligning the template
#     near the lower edge and reserving a clean upper-left pocket for the end
#     block. Keeping the template visible in the output makes the combo effect
#     obvious instead of subtle.
# - implicit_assumptions_review:
#   - `g[0,0]` is still treated as background, so the swapped-in top-left end
#     block cannot cover the corner; the generator samples it strictly inside the
#     pocket at rows/cols >= 1.
#   - Path detection still assumes path_color is the most frequent non-bg color.
#     The generator preserves the original color budget so the wide path band
#     dominates every non-bg count.
#   - Marker detection assumes cycle-colored singletons adjacent to the path. The
#     bottom-right template is placed inside isolated background, so its 2x2
#     blocks cannot be mistaken for markers.
#   - Template parsing assumes one ordered row of 2x2 blocks in the lower-right
#     scan window. The generator keeps that geometry fixed and keeps the upper-left
#     end block away from the template colors.
#   - Output preservation creates a new assumption: template cells must not be on
#     the traversed path or collision frontier. The generator enforces a full bg
#     footprint plus a path-free halo around the lower-right pocket.
# - placement_legality:
#   - The path band is built first and checked for single-component 4-connectivity
#     and col-0 marker embedding.
#   - The template footprint is required to be all background in the lower-right
#     region before painting; the end block is sampled from top-left 2x2 bg
#     candidates whose surrounding 1-cell halo contains no path.
#   - Markers overwrite only col-0 path cells and are revalidated so the path
#     remains one 4-connected component after overwrite.
# - observability_and_tidiness:
#   - Across samples, humans can consistently read "instructions at bottom-right,
#     terminal block at top-left" because the two objects occupy opposite empty
#     pockets with no clutter.
#   - Preserving the template in the output avoids visual ambiguity: the solver is
#     clearly not transforming that object, only using it as a reference while
#     removing the separate end block.
# - inferability_of_change:
#   - The combo does not introduce hidden color ordering or numeric-color tricks.
#     The only changed cues are spatial: where the cycle is read from and whether
#     it survives into the output. Those cues are directly visible in 2-3 examples.

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

_SOLVERS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVERS_DIR) not in sys.path:
    sys.path.append(str(_SOLVERS_DIR))

from arc_utils import label_mask


def _assert_path_single_component(g: np.ndarray, path_color: int) -> None:
    """Eval invariant: path_color forms a single 4-connected component."""
    _, n_path = label_mask(g == path_color, connectivity=4)
    assert n_path == 1, f"path_color must be a single 4-connected component, got {n_path}"


def _assert_marker_embedded_in_path(
    g: np.ndarray, markers: list[tuple[int, int, int]], path_color: int
) -> None:
    """Eval invariant: each marker remains embedded in a col-0 path segment."""
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


def _find_template_blocks(
    g: np.ndarray, template_bg: int, path_color: int
) -> tuple[list[int], list[tuple[int, int]]]:
    """Find the bottom-right horizontal row of 2x2 monochrome template blocks."""
    H, W = g.shape
    row_blocks: dict[int, list[tuple[int, int, int]]] = {}
    row_lo = max(0, H - 6)
    col_lo = max(0, W - 25)
    for r in range(row_lo, H - 1):
        for c in range(col_lo, W - 1):
            v = int(g[int(r), int(c)])
            if v == int(template_bg) or v == int(path_color):
                continue
            if (
                int(g[int(r + 1), int(c)]) == v
                and int(g[int(r), int(c + 1)]) == v
                and int(g[int(r + 1), int(c + 1)]) == v
            ):
                row_blocks.setdefault(int(r), []).append((int(c), int(r), int(v)))

    if not row_blocks:
        return [], []

    template_row = max(row_blocks, key=lambda r: (len(row_blocks[r]), r))
    blocks = sorted(row_blocks[template_row], key=lambda item: item[0])
    pattern = [v for _, _, v in blocks]
    pat_cells: list[tuple[int, int]] = []
    for c, r, _ in blocks:
        pat_cells.extend([(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)])
    return pattern, pat_cells


def _find_end_block(
    g: np.ndarray, template_bg: int, path_color: int, pat_set: set[int]
) -> tuple[int | None, tuple[int, int] | None]:
    """Find the swapped-in top-left 2x2 end block."""
    H, W = g.shape
    search_limits = [
        (range(0, min(H - 1, H // 2 + 4)), range(0, min(W - 1, W // 2 + 4))),
        (range(0, H - 1), range(0, W - 1)),
    ]
    for rows, cols in search_limits:
        for r in rows:
            for c in cols:
                v = int(g[int(r), int(c)])
                if v == int(template_bg) or v == int(path_color) or v in pat_set:
                    continue
                if (
                    int(g[int(r + 1), int(c)]) == v
                    and int(g[int(r), int(c + 1)]) == v
                    and int(g[int(r + 1), int(c + 1)]) == v
                ):
                    return int(v), (int(r), int(c))
    return None, None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate alternating-turn particles while reading a preserved template from the opposite corner.

    Input structure:
    - Bottom-right: a horizontal row of 2x2 monochrome blocks defines the repeating color cycle.
    - Left edge: isolated single pixels of a cycle color mark path start points embedded in the col-0 band.
    - Top-left: one distinct 2x2 monochrome block provides the collision/terminal color.

    Rule:
    - Each marker follows the adjacent path, advancing one step at a time and repainting visited path cells by cycling through the template colors.
    - When the next straight step is not path, that collision cell is painted with the end color, then the walker tries the scheduled 90-degree turn.
    - Turn parity is strict and alternating: first turn left, second right, third left, and so on; if the scheduled turn is not path, the walker stops.

    Key insight:
    - The motion logic is the same alternating-turn mechanism as the base task, but now the usable instruction template lives in the lower-right corner and remains visible in the output, so the separate top-left 2x2 block is the only anchor that gets erased.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    template_bg = int(g[0, 0])

    counts = Counter(map(int, g.flatten()))
    path_color = int(
        max((cnt, int(col)) for col, cnt in counts.items() if int(col) != int(template_bg))[1]
    )
    _assert_path_single_component(g, path_color)

    pattern, _pat_cells = _find_template_blocks(g, int(template_bg), int(path_color))
    pat_set = set(map(int, pattern))
    end_color, end_pos = _find_end_block(g, int(template_bg), int(path_color), pat_set)

    markers: list[tuple[int, int, int]] = []
    for r in range(H):
        for c in range(W):
            v = int(g[int(r), int(c)])
            if v not in pat_set:
                continue
            same = 0
            has_path = False
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
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
    if end_pos is not None:
        er, ec = end_pos
        out[int(er) : int(er + 2), int(ec) : int(ec + 2)] = int(template_bg)

    dir_order = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # R, D, L, U

    per_marker_cells: list[set[tuple[int, int]]] = []
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
            continue

        r, c = int(sr), int(sc)
        dr, dc = init
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
            assert not overlap, f"marker trajectories {i},{j} overlap on {len(overlap)} cell(s)"

    return out


def _build_band(H: int, W: int, lobe_rows: list[int] | None = None) -> np.ndarray:
    """Return a thick meandering anti-diagonal path band."""
    bot_center = random.randint(0, W // 4)
    centers = np.linspace(W - 1, bot_center, H).astype(float)
    half_w = np.array(
        [0.5 * (3 + (1.0 - abs(2.0 * r / max(1, H - 1) - 1.0)) * (W - 6)) for r in range(H)]
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

    # Clear a stable upper-left pocket for the swapped-in end block.
    for r in range(1, min(4, H)):
        c_left[r] = max(c_left[r], 12)
        if c_right[r] < c_left[r]:
            c_right[r] = c_left[r]

    c_right[0] = W - 1
    c_left[0] = max(c_left[0], W - 6)
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
    """Carve a background separator from col 0 between two marker rows."""
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


def _gen_attempt() -> np.ndarray | None:
    H = W = random.randint(20, 30)
    n_markers = 2 if random.random() < 0.55 else 1

    palette = list(range(10))
    random.shuffle(palette)
    bg = palette[0]
    path_color = palette[1]
    end_color = palette[2]
    n_blocks = random.randint(2, 4)
    cycle_pool = palette[3:]
    cycle_colors = random.sample(cycle_pool, n_blocks)

    if n_markers == 1:
        r1 = random.randint(H // 2, H - 4)
        markers = [r1]
    else:
        r1 = random.randint(H // 3, H // 2)
        r2 = random.randint(max(r1 + 5, 2 * H // 3), H - 4)
        markers = [r1, r2]

    mask = _build_band(H, W, lobe_rows=markers)
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
            keep_cols = random.randint(2, 4)
            pre_carve_cells = mask & ~trial
            for c in range(min(keep_cols, W)):
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

    # Right-aligned bottom template: same geometry as the base task, swapped pocket.
    template_row = random.choice([H - 4, H - 3])
    start_col = W - 3 * n_blocks
    span_w = 3 * n_blocks - 1
    if template_row < 0 or start_col < 1:
        return None
    if (g[template_row : template_row + 2, start_col : start_col + span_w] != bg).any():
        return None
    r0 = max(0, template_row - 1)
    r1_ = min(H, template_row + 3)
    c0 = max(0, start_col - 1)
    c1_ = min(W, start_col + span_w + 1)
    if (g[r0:r1_, c0:c1_] == path_color).any():
        return None
    block_cols = [start_col + 3 * i for i in range(n_blocks)]
    for i, bc in enumerate(block_cols):
        g[template_row : template_row + 2, bc : bc + 2] = cycle_colors[i]

    # Swapped end block: isolated 2x2 in the top-left pocket, not on (0,0).
    end_candidates = []
    for r in range(1, min(H // 2 + 1, H - 1)):
        for c in range(1, min(W // 2 + 1, W - 1)):
            if not (g[r : r + 2, c : c + 2] == bg).all():
                continue
            r0 = max(0, r - 1)
            r1_ = min(H, r + 3)
            c0 = max(0, c - 1)
            c1_ = min(W, c + 3)
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
    non_bg = sorted(((c, col) for col, c in cnt.items() if col != bg), reverse=True)
    if not non_bg or non_bg[0][1] != path_color:
        return None

    try:
        out = main(g)
    except Exception:
        return None

    # Non-degeneracy for the combo: template must survive and end block must be removed.
    if not np.array_equal(
        out[template_row : template_row + 2, start_col : start_col + span_w],
        g[template_row : template_row + 2, start_col : start_col + span_w],
    ):
        return None
    if not np.all(out[er : er + 2, ec : ec + 2] == bg):
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
