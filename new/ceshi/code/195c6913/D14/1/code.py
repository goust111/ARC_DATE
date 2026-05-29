# Variant of 195c6913
# Dimension: D14 — 终止块结构 (meta)
# Change: 终止块为单个2x2单色块（end_color），位于右下bg区域，所有碰撞角标记都使用此颜色。... → 多个终止块——每个标记对应一个

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import label_mask

# Design analysis for the D14 variant
# a) Concrete behavior:
#    - The top-left template is unchanged: a left-to-right row of 2x2 blocks
#      defines the paint cycle.
#    - The left-edge markers are unchanged in shape: isolated single pixels
#      embedded in the path.
#    - The change is in the lower-right bg region: instead of one 2x2 terminal
#      block, place one 2x2 block per marker, each with a distinct color.
#      Markers and end blocks are paired by vertical order: top marker uses the
#      top end block's color, bottom marker uses the bottom end block's color.
#    - Small example trace:
#      old rule: markers M1/M2 both paint collision corners with shared color E.
#      new rule: end blocks are [E1, E2], sorted top-to-bottom. M1 paints all of
#      its corner hits with E1, M2 paints all of its corner hits with E2.
# b) Complexity calibration:
#    - Keep end blocks as the same simple 2x2 monochrome objects; only multiplicity
#      and pairing change. This makes the difference visible without introducing
#      extra shape logic.
#    - End colors are unique per grid so the pairing is visually legible.
# c) Hidden-assumption audit on the original main():
#    - "first 2x2 non-template block == the end block": no longer valid.
#    - "one erase rectangle for the end block": no longer valid.
#    - "all collision marks use one end_color": no longer valid.
#    - Path detection, template detection, marker detection, turn alternation,
#      and trajectory disjointness remain valid and are preserved.
# d) Placement legality:
#    - Generator places all end blocks on 2x2 bg squares in the bottom-right
#      region, with a one-cell moat from the path and no overlap between blocks.
#    - For two markers, chosen blocks have strictly increasing rows so the pairing
#      by vertical order is well-defined.
# e) Visual observability:
#    - The variant is visible only when samples have 2 markers and 2 end
#      blocks, so the generator fixes n_markers == 2.
#    - The pairing cue is spatial and visible: upper marker ↔ upper end block,
#      lower marker ↔ lower end block.
# f) Visual cleanliness:
#    - End blocks stay grouped in the lower-right bg region, away from template
#      and away from the path, so they do not clutter the active trajectory area.
# g) Color inferability:
#    - The end-color mapping is not based on numeric color ids. It is determined
#      only by the observable top-to-bottom correspondence between markers and
#      end blocks.
# h) New main() assumptions guaranteed by generate_input():
#    - Number of detected end blocks equals number of markers.
#    - End blocks are 2x2 monochrome blocks with distinct colors.
#    - Sorting markers and end blocks by row yields the intended pairing.


def _assert_path_single_component(g: np.ndarray, path_color: int) -> None:
    """Eval invariant: path_color forms a single 4-connected component."""
    _, n_path = label_mask(g == path_color, connectivity=4)
    assert n_path == 1, f"path_color must be a single 4-connected component, got {n_path}"


def _assert_marker_embedded_in_path(
    g: np.ndarray, markers: list[tuple[int, int, int]], path_color: int
) -> None:
    """Eval invariant: each col-0 marker sits inside a vertical path segment."""
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


def _find_pattern_blocks(
    g: np.ndarray, template_bg: int, path_color: int
) -> tuple[list[int], list[tuple[int, int]]]:
    pattern: list[int] = []
    pat_cells: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    H, W = g.shape
    for r in range(min(6, H - 1)):
        for c in range(min(25, W - 1)):
            if (int(r), int(c)) in seen:
                continue
            v = int(g[r, c])
            if v == int(template_bg) or v == int(path_color):
                continue
            if (
                int(g[r + 1, c]) == v
                and int(g[r, c + 1]) == v
                and int(g[r + 1, c + 1]) == v
            ):
                pattern.append(int(v))
                for rr, cc in ((r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)):
                    seen.add((int(rr), int(cc)))
                    pat_cells.append((int(rr), int(cc)))
    return pattern, pat_cells


def _find_markers(
    g: np.ndarray, path_color: int, pattern: list[int]
) -> list[tuple[int, int, int]]:
    H, W = g.shape
    pat_set = set(map(int, pattern))
    markers: list[tuple[int, int, int]] = []
    for r in range(H):
        for c in range(W):
            v = int(g[r, c])
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
    return markers


def _find_end_blocks(
    g: np.ndarray, template_bg: int, path_color: int, pat_set: set[int]
) -> list[tuple[int, int, int]]:
    H, W = g.shape
    end_blocks: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int]] = set()
    for r in range(H - 1):
        for c in range(W - 1):
            if (int(r), int(c)) in seen:
                continue
            v = int(g[r, c])
            if v == int(template_bg) or v == int(path_color) or v in pat_set:
                continue
            if (
                int(g[r + 1, c]) == v
                and int(g[r, c + 1]) == v
                and int(g[r + 1, c + 1]) == v
            ):
                end_blocks.append((int(r), int(c), int(v)))
                for rr in range(r, r + 2):
                    for cc in range(c, c + 2):
                        seen.add((int(rr), int(cc)))
    return end_blocks


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate particles along a path with alternating turns and marker-specific terminal colors.

    Input structure:
    - Top-left: a row of 2x2 monochrome blocks defines the repeating color cycle.
    - Left edge: 1 or 2 isolated marker pixels of a cycle color are embedded in the path.
    - Lower-right bg region: one 2x2 monochrome terminal block per marker, each with a distinct color.

    Rule:
    - Each marker launches a particle that advances along the path and paints cycle colors in order.
    - When a forward step hits a non-path cell, the collision cell is painted with that marker's paired terminal color.
    - Turn direction still alternates strictly CCW, CW, CCW, CW...; if the scheduled turn is not path, the particle stops.
    - All template blocks and all terminal blocks are erased to background in the output.

    Key insight:
    - The motion logic is unchanged; the only new information to recover is the pairing between markers and terminal colors, which is encoded by matching their top-to-bottom order.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    template_bg = int(g[0, 0])
    counts = Counter(map(int, g.flatten()))
    path_color = int(
        max((cnt, int(col)) for col, cnt in counts.items() if int(col) != int(template_bg))[1]
    )
    _assert_path_single_component(g, path_color)

    pattern, pat_cells = _find_pattern_blocks(g, template_bg, path_color)
    pat_set = set(map(int, pattern))
    markers = _find_markers(g, path_color, pattern)
    _assert_marker_embedded_in_path(g, markers, int(path_color))

    end_blocks = _find_end_blocks(g, template_bg, path_color, pat_set)
    assert len(end_blocks) == len(markers), (
        f"expected one end block per marker, got {len(end_blocks)} blocks for {len(markers)} markers"
    )
    end_colors = [col for _, _, col in end_blocks]
    assert len(set(end_colors)) == len(end_colors), "end block colors must be distinct"

    markers = sorted(markers, key=lambda x: (int(x[0]), int(x[1])))
    end_blocks = sorted(end_blocks, key=lambda x: (int(x[0]), int(x[1])))
    marker_to_end_color = {
        (int(sr), int(sc)): int(color)
        for (sr, sc, _), (_, _, color) in zip(markers, end_blocks)
    }

    out = g.copy()
    for r, c in pat_cells:
        out[int(r), int(c)] = int(template_bg)
    for er, ec, _ in end_blocks:
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

        my_end_color = marker_to_end_color[(int(sr), int(sc))]
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

            if int(g[nr, nc]) == int(template_bg):
                out[nr, nc] = int(my_end_color)

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
    """Return a boolean mask for a thick meandering anti-diagonal band."""
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
    """Carve a bg separator from the left edge without disconnecting the band."""
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


def _candidate_end_blocks(g: np.ndarray, bg: int, path_color: int) -> list[tuple[int, int]]:
    H, W = g.shape
    end_candidates: list[tuple[int, int]] = []
    for r in range(H // 2, H - 1):
        for c in range(W // 2, W - 1):
            if not (g[r : r + 2, c : c + 2] == bg).all():
                continue
            r0, r1 = max(0, r - 1), min(H, r + 3)
            c0, c1 = max(0, c - 1), min(W, c + 3)
            if (g[r0:r1, c0:c1] == path_color).any():
                continue
            end_candidates.append((int(r), int(c)))
    return end_candidates


def _choose_end_positions(
    g: np.ndarray, bg: int, path_color: int, n_markers: int
) -> list[tuple[int, int]] | None:
    candidates = _candidate_end_blocks(g, bg, path_color)
    if n_markers == 1:
        if not candidates:
            return None
        return [random.choice(candidates)]

    best_pairs: list[tuple[int, int, int]] = []
    for i, (r1, c1) in enumerate(candidates):
        for j in range(i + 1, len(candidates)):
            r2, c2 = candidates[j]
            if r2 <= r1 + 2:
                continue
            if max(abs(r1 - r2), abs(c1 - c2)) <= 2:
                continue
            score = abs(c1 - c2) + max(0, 4 - (r2 - r1))
            best_pairs.append((score, i, j))
    if not best_pairs:
        return None

    best_pairs.sort(key=lambda x: x[0])
    top = best_pairs[: min(12, len(best_pairs))]
    _, i1, i2 = random.choice(top)
    p1 = candidates[i1]
    p2 = candidates[i2]
    return [p1, p2] if p1[0] <= p2[0] else [p2, p1]


def _gen_attempt() -> np.ndarray | None:
    H = W = random.randint(20, 30)
    n_markers = 2

    palette = list(range(10))
    random.shuffle(palette)
    bg = palette[0]
    path_color = palette[1]
    n_blocks = random.randint(2, 4)

    needed = n_blocks + n_markers
    if needed > len(palette) - 2:
        return None
    picked = palette[2 : 2 + needed]
    cycle_colors = picked[:n_blocks]
    end_colors = picked[n_blocks:]

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
    block_cols = [1, 4, 7, 10][:n_blocks]
    if block_cols[-1] + 1 >= W:
        return None
    for bc in block_cols:
        for dr in range(2):
            for dc in range(2):
                rr, cc = template_row + dr, bc + dc
                if rr >= H or cc >= W or mask[rr, cc]:
                    return None

    g = np.full((H, W), bg, dtype=int)
    g[mask] = path_color

    for i, bc in enumerate(block_cols):
        g[template_row : template_row + 2, bc : bc + 2] = cycle_colors[i]

    end_positions = _choose_end_positions(g, bg, path_color, n_markers)
    if end_positions is None:
        return None
    for (er, ec), end_color in zip(end_positions, end_colors):
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

    pat_set = set(cycle_colors)
    found_end_blocks = _find_end_blocks(g, bg, path_color, pat_set)
    if len(found_end_blocks) != n_markers:
        return None
    if len({col for _, _, col in found_end_blocks}) != n_markers:
        return None

    if n_markers == 2:
        collision_colors = {
            int(out[r, c])
            for r in range(H)
            for c in range(W)
            if int(g[r, c]) == bg and int(out[r, c]) != bg
        }
        if len(collision_colors) < 2:
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
