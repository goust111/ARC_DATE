# Variant of 195c6913
# Dimension: D4 — 源对象：标记像素定义 (src)
# Change: 1-2个孤立的单像素标记，颜色=循环色[0]，位于col 0（左网格边缘），嵌入在path区段中；初始方向优先级为R,D... → 标记颜色各不相同（非全部=循环色[0]）
#
# Design analysis
# a) Concrete behavior and trace:
#    - Input still has three object families: a thick anti-diagonal path band,
#      a top-left row of 2x2 template blocks encoding a cycle, and 1-2 single
#      marker pixels on col 0 embedded inside the path lobe.
#    - The D4 change is only in marker semantics: a marker may now match any
#      template block color, not always the first one. Its own color says
#      which cycle phase it starts from.
#    - Example: template colors [A, B, C], marker color B. Original generator
#      would have forced marker=A, so the first painted path cell became B.
#      In the variant, marker=B means the local cycle index starts at B, so
#      the first painted path cell becomes C. Turns and stop conditions stay
#      identical; only the paint phase shifts.
# b) Complexity calibration:
#    - To keep the mapping visually learnable, template colors are forced
#      unique. Repeated template colors would make "marker color -> start slot"
#      ambiguous, so the generator removes that degree of freedom.
#    - Marker count stays 1-2. When there are 2 markers their colors are
#      distinct, giving side-by-side evidence that different start phases
#      produce offset color traces on otherwise similar motion rules.
# c) Hidden-assumption audit of the original main():
#    - Marker detection itself was already general: it finds isolated pixels of
#      any template color adjacent to the path. No change needed there.
#    - The original generator silently made `pattern.index(sv)` degenerate by
#      always setting `sv == pattern[0]`. The solver line is valid, but only if
#      each template color is unique. The variant therefore adds an explicit
#      uniqueness assertion for template colors and for multi-marker colors.
#    - Erasing template/end blocks, path detection by most-frequent non-bg, and
#      collision/turn logic do not rely on the old "all markers share color"
#      property, so they remain unchanged.
# d) Placement legality:
#    - Markers remain on col 0 and must overwrite path cells whose vertical
#      neighbors are also path, preserving the embedded-marker invariant.
#    - Template placement, end-block placement, fjord/island separator carving,
#      and path 4-connectivity rules are unchanged; the only extra legality
#      rule is that marker colors must come from distinct template slots.
# e) Visual observability:
#    - Every generated sample has at least one marker whose start index is not
#      0, so the variant does not collapse back to the original visual behavior.
#    - In 2-marker samples, the markers usually paint different first-step
#      colors because they start from different template slots.
# f) Visual cleanliness:
#    - No new symbols are introduced. The same single-pixel markers are reused,
#      so the grid stays readable. Disjoint trajectory constraints still prevent
#      color interleaving from becoming cluttered.
# g) Color-rule inferability:
#    - The only color dependency is local and explicit: match each marker's
#      color to the identical 2x2 template block, then advance one step ahead in
#      that cycle while walking. Keeping 2-4 unique template colors makes this
#      inferable from a few examples.
# 9) New solver assumptions and generator guarantees:
#    - main() now assumes template colors are unique and each marker color maps
#      to exactly one template slot.
#    - generate_input() guarantees this by sampling template colors without
#      replacement and assigning distinct start slots to different markers.

from __future__ import annotations

from collections import Counter
import random

import numpy as np
from arc_utils import label_mask


def _assert_path_single_component(g: np.ndarray, path_color: int) -> None:
    """Eval invariant: path_color forms a single 4-connected component."""
    _, n_path = label_mask(g == path_color, connectivity=4)
    assert n_path == 1, f"path_color must be a single 4-connected component, got {n_path}"


def _assert_marker_embedded_in_path(
    g: np.ndarray, markers: list[tuple[int, int, int]], path_color: int
) -> None:
    """Each col-0 marker must replace a path cell inside a vertical path run."""
    H, _ = g.shape
    for r, c, _ in markers:
        assert c == 0, f"marker must lie on col 0, got ({r},{c})"
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


def _parse_scene(
    g: np.ndarray,
) -> tuple[
    int,
    int,
    list[int],
    list[tuple[int, int]],
    int | None,
    tuple[int, int] | None,
    list[tuple[int, int, int]],
]:
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
    seen: set[tuple[int, int]] = set()
    for r in range(min(6, H - 1)):
        for c in range(min(25, W - 1)):
            if (int(r), int(c)) in seen:
                continue
            v = int(g[int(r), int(c)])
            if v == template_bg or v == path_color:
                continue
            if (
                int(g[int(r + 1), int(c)]) == v
                and int(g[int(r), int(c + 1)]) == v
                and int(g[int(r + 1), int(c + 1)]) == v
            ):
                pattern.append(v)
                for rr, cc in [(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)]:
                    seen.add((int(rr), int(cc)))
                    pat_cells.append((int(rr), int(cc)))

    assert pattern, "template cycle not found"
    assert len(pattern) == len(set(pattern)), "variant requires unique template colors"
    pat_set = set(pattern)

    end_color = None
    end_pos = None
    for r in range(H - 1):
        for c in range(W - 1):
            v = int(g[int(r), int(c)])
            if v == template_bg or v == path_color or v in pat_set:
                continue
            if (
                int(g[int(r + 1), int(c)]) == v
                and int(g[int(r), int(c + 1)]) == v
                and int(g[int(r + 1), int(c + 1)]) == v
            ):
                end_color = v
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
                    if int(g[rr, cc]) == path_color:
                        has_path = True
            if same == 0 and has_path:
                markers.append((int(r), int(c), v))

    _assert_marker_embedded_in_path(g, markers, path_color)
    assert 1 <= len(markers) <= 2, f"expected 1-2 markers, got {len(markers)}"
    marker_colors = [sv for _, _, sv in markers]
    assert len(marker_colors) == len(set(marker_colors)), "marker colors must be distinct"

    return template_bg, path_color, pattern, pat_cells, end_color, end_pos, markers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace particles along a path while their paint phase is chosen by marker color.

    Input structure:
    - Top-left: a row of 2x2 solid blocks gives the repeating cycle in left-to-right order.
    - Left edge: 1-2 isolated marker pixels sit on col 0 inside the path lobe, and each marker color matches one template block.
    - Elsewhere: one distinct 2x2 block marks the collision color used at path exits.

    Rule:
    - Each marker launches from its own cell using the first adjacent path cell in R,D,L,U priority order.
    - The marker's color selects its starting slot in the template cycle; every forward step paints the next color in that cycle.
    - On a non-path collision, paint that collision cell with the end color, then alternate turn direction left, right, left, right.
    - If the required turn does not lead into path, that marker stops immediately.

    Key insight:
    - The geometry is solved exactly as before; the new deduction is that a marker does not imply "cycle starts at block 0".
      Its own color is the phase anchor, so different markers can walk the same motion rule with offset paint sequences.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    (
        template_bg,
        path_color,
        pattern,
        pat_cells,
        end_color,
        end_pos,
        markers,
    ) = _parse_scene(g)
    color_to_idx = {int(v): i for i, v in enumerate(pattern)}

    out = g.copy()
    for r, c in pat_cells:
        out[int(r), int(c)] = template_bg
    if end_pos is not None:
        er, ec = end_pos
        out[int(er) : int(er + 2), int(ec) : int(ec + 2)] = template_bg

    dir_order = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # R, D, L, U

    per_marker_cells: list[set[tuple[int, int]]] = []
    for sr, sc, sv in markers:
        init = None
        for dr, dc in dir_order:
            rr = int(sr + dr)
            cc = int(sc + dc)
            if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == path_color:
                init = (int(dr), int(dc))
                break
        if init is None:
            per_marker_cells.append(set())
            continue

        r, c = int(sr), int(sc)
        dr, dc = init
        idx = int(color_to_idx[int(sv)])
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

            if int(g[nr, nc]) == path_color:
                idx = int((idx + 1) % len(pattern))
                out[nr, nc] = int(pattern[idx])
                my_cells.add((int(nr), int(nc)))
                r, c = nr, nc
                continue

            if int(g[nr, nc]) == template_bg and end_color is not None:
                out[nr, nc] = int(end_color)

            left = (-int(dc), int(dr))
            right = (int(dc), -int(dr))
            tdr, tdc = left if (int(hits) % 2 == 0) else right
            rr = int(r + tdr)
            cc = int(c + tdc)
            if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == path_color:
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
    """Return a thick meandering anti-diagonal band."""
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
    """Carve a bg separator strip from col 0 between two marker rows."""
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


def _choose_marker_start_slots(n_markers: int, n_blocks: int) -> list[int]:
    """Pick marker cycle slots, forcing the variant to be visible."""
    assert n_blocks >= 2
    if n_markers == 1:
        return [random.randint(1, n_blocks - 1)]
    slots = random.sample(list(range(n_blocks)), n_markers)
    return [int(x) for x in slots]


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

    marker_slots = _choose_marker_start_slots(n_markers, n_blocks)
    assert any(slot != 0 for slot in marker_slots), "variant must exercise nonzero start slot"
    for r, slot in zip(markers, marker_slots):
        g[r, 0] = cycle_colors[int(slot)]

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
