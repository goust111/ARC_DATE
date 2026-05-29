# Variant of 195c6913
# Dimension: D7 — 模板查找机制 (symbol_instructions)
# Change: 左上角一排2x2单色块定义颜色循环顺序（从左到右），标记的起始循环位由其颜色在循环中的位置决定。... → 模板位于右下角（与终止块位置互换）
#
# Design analysis
# - concrete_behavior:
#   - The color-cycle template is still a single horizontal row of 2x2 monochrome
#     blocks read left-to-right, but it now sits in the bottom-right background
#     pocket. The lone 2x2 end block moves to the top-left background pocket.
#   - Markers still live on col 0 inside the path lobe, and the path-walk /
#     alternating-turn dynamics are unchanged. The visible change is therefore
#     entirely about where the solver must read the instruction blocks from.
# - example_trace:
#   - Original code scans only the upper-left window for 2x2 blocks, so on a
#     swapped-layout input it would collect an empty cycle and either do nothing
#     or mis-read the top-left end block.
#   - The variant instead scans the lower-right window for the ordered template,
#     then looks for the distinct 2x2 end block in the upper-left pocket. After
#     that, the walking logic is identical.
# - complexity_calibration:
#   - The template keeps the original 2-4 block count, 2x2 size, 1-cell gaps,
#     and horizontal ordering so the rule remains inferable from a few examples.
#   - The generator right-aligns the template near the lower edge so the swap is
#     visually obvious, not a barely-shifted local perturbation.
# - implicit_assumptions_review:
#   - `g[0,0]` is still used as background, so the swapped-in end block must not
#     cover the corner; the generator places it strictly inside the top-left bg
#     pocket.
#   - Path-color detection still relies on "most frequent non-bg"; the generator
#     preserves the original color budget so this remains true.
#   - Marker detection still assumes markers are isolated cycle-colored singles
#     adjacent to the path; template blocks are far from the path so they do not
#     get mistaken for markers.
# - placement_legality:
#   - The template footprint is checked as an all-bg lower-right rectangle before
#     painting the 2x2 blocks, guaranteeing no overlap with path cells.
#   - The end block is sampled from 2x2 top-left bg candidates with a 1-cell
#     path-free halo so it stays isolated and solver-readable.
# - observability_and_tidiness:
#   - Every valid sample has template rows in the lower-right half and an end
#     block in the upper-left half, making the inversion visible at a glance.
#   - No invisible ordering tricks are introduced: the cycle is still read from
#     spatial left-to-right order, only from a different corner.

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
    """Find the bottom-right row of 2x2 monochrome template blocks."""
    H, W = g.shape
    pattern: list[int] = []
    pat_cells: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    row_lo = max(0, H - 6)
    col_lo = max(0, W - 25)
    for r in range(row_lo, H - 1):
        for c in range(col_lo, W - 1):
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
    Simulate particles on a path while reading their color cycle from a swapped template corner.

    Input structure:
    - Bottom-right: a horizontal row of 2x2 monochrome blocks defines the repeating cycle.
    - Left edge: isolated single pixels of a cycle color mark start points embedded in the path.
    - Top-left: one distinct 2x2 monochrome block supplies the terminal collision color.

    Rule:
    - Each marker moves straight along adjacent path cells and paints the cycle one step at a time.
    - When the next step is not path, that collision cell is painted with the terminal color.
    - After each collision the turn direction alternates left, right, left, right; if the scheduled turn is blocked, movement stops.

    Key insight:
    - The motion rule is unchanged from the base task; the twist is that the instruction template must be read from the opposite corner, while the upper-left 2x2 block is no longer part of the cycle at all.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    template_bg = int(g[0, 0])

    counts = Counter(map(int, g.flatten()))
    path_color = int(
        max([(cnt, int(col)) for col, cnt in counts.items() if int(col) != int(template_bg)])[1]
    )
    _assert_path_single_component(g, path_color)

    pattern, pat_cells = _find_template_blocks(g, int(template_bg), int(path_color))

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
    for r, c in pat_cells:
        out[int(r), int(c)] = int(template_bg)
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
    """Return a thick meandering anti-diagonal path band.

    The upper-left pocket remains clear for the swapped-in end block. The path
    still fans from the upper-right toward the lower-left, with an optional col-0
    lobe to embed the markers.
    """
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

    # Keep a stable top-left bg pocket for the swapped-in end block.
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
    """Carve a bg separator from col 0 between the two marker rows."""
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
    n_markers = 2 if random.random() < 0.7 else 1

    palette = list(range(10))
    random.shuffle(palette)
    bg = palette[0]
    path_color = palette[1]
    end_color = palette[2]
    n_blocks = random.randint(2, 4)
    cycle_pool = palette[3:]
    cycle_colors = [random.choice(cycle_pool) for _ in range(n_blocks)]

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
        use_interior = random.random() < 0.25
        if use_interior:
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

    # Right-aligned bottom template: same 2x2+gap geometry, opposite corner.
    template_row = random.choice([H - 4, H - 3])
    start_col = W - 3 * n_blocks
    span_w = 3 * n_blocks - 1
    if template_row < 0 or start_col < 1:
        return None
    if (g[template_row : template_row + 2, start_col : start_col + span_w] != bg).any():
        return None
    block_cols = [start_col + 3 * i for i in range(n_blocks)]
    for i, bc in enumerate(block_cols):
        g[template_row : template_row + 2, bc : bc + 2] = cycle_colors[i]

    # Swapped end block: isolated 2x2 in the top-left bg pocket, not on (0,0).
    end_candidates = []
    for r in range(1, min(H // 2 + 1, H - 1)):
        for c in range(1, min(W // 2 + 1, W - 1)):
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
