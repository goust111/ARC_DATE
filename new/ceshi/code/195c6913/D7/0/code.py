# Variant of 195c6913
# Dimension: D7 — 模板查找机制 (symbol_instructions)
# Change: 左上角一排2x2单色块定义颜色循环顺序（从左到右），标记的起始循环位由其颜色在循环中的位置决定。... → 模板为颜色条纹（1xN或Nx1）

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import label_mask

# Design analysis
# ---------------
# a) Concrete form and behavior trace
#    - The template is no longer a row of separated 2x2 monochrome blocks.
#      Instead, the top-left region contains one compact connected stripe:
#      either 1xN (horizontal) or Nx1 (vertical), with N in [2,4].
#    - Each stripe cell contributes one color to the cycle in reading order.
#      Horizontal stripes are read left->right; vertical stripes top->bottom.
#    - The rest of the task stays the same: left-edge marker pixels launch
#      particles, the path walk still alternates CCW/CW turns on collisions,
#      and the end marker is still a 2x2 block in the opposite bg region.
#    - Small trace:
#      old input template: [2x2 red] gap [2x2 blue] gap [2x2 yellow]
#      old solver cycle: [red, blue, yellow]
#      new input template: one row [red, blue, yellow]
#      new solver cycle: [red, blue, yellow]
#      difference: the cycle information now comes from a 1-cell-thick linear
#      component rather than repeated 2x2 blobs, so template detection must use
#      connected-line geometry instead of per-color 2x2 checks.
# b) Complexity calibration
#    - To keep the change visible, the generator shrinks the reserved top-left
#      clearance from the wide old block area to a smaller stripe area.
#    - Stripe length remains 2-4 so the cycle still has repeated support across
#      samples without adding too many cases.
#    - Cycle colors are sampled without repetition. Repeated colors are legal in
#      the original family, but on a 1-cell-thick stripe they make the template
#      visually ambiguous and reduce the observability of the D7 change.
# c) Hidden assumptions in the original main()
#    - Old pattern detection assumed every template color appears inside a 2x2
#      monochrome square. This fails immediately for a 1xN / Nx1 stripe.
#    - Old template erasure cleared 2x2 footprints. That would erase wrong
#      cells under the new variant, so we must track exact stripe cells.
#    - Marker detection uses membership in the cycle-color set and requires the
#      candidate pixel to be isolated from same-colored neighbors. This still
#      works, but only if template cells are not mistaken for markers. The new
#      generator guarantees the stripe is not path-adjacent, and the solver
#      locates the stripe as a connected linear component before scanning markers.
#    - End-block detection assumes it is the unique 2x2 monochrome component
#      among non-bg/non-path colors. This remains valid because the stripe is
#      1-cell thick and uses colors disjoint from the end color.
# d) Placement legality
#    - Horizontal stripes live near rows 1-2 and cols 1-2; vertical stripes
#      live near rows 1-2 and cols 1-2 as well. The path builder reserves a
#      small clear rectangle so the stripe never overlaps the path.
#    - The end block stays in the bottom-right bg region and remains separated
#      from the path by at least one cell, exactly as before.
#    - Markers remain embedded in a col-0 path segment, preserving the solver's
#      path-topology assertions and initial-direction behavior.
# e) Visual observability
#    - Across examples, the human-visible difference is that the color cycle is
#      encoded by a compact colored line rather than several fat 2x2 blocks.
#    - Both horizontal and vertical stripe orientations are generated so the
#      solver must identify a linear template, not memorize one fixed row shape.
# f) Visual cleanliness
#    - The stripe is isolated in the top-left bg pocket, the path remains a
#      large diagonal band, markers stay on col 0, and the end block stays in
#      the opposite bg pocket. These roles remain spatially separated.
# g) Color-change inferability
#    - Color changes along the path are still explained by the visible template
#      order. The stripe's reading direction is structural (left->right or
#      top->bottom), not based on color ids or hidden ranking.
# h) Variant metric to check against degeneration
#    - After implementation, we will measure the stripe orientation distribution
#      and compare the new solver against the original 2x2-block solver on the
#      same generated inputs. A low old-solver match rate confirms the new
#      template mechanism is actually controlling the samples.
# i) New solver assumptions that the generator must satisfy
#    - There is exactly one top-left connected non-path/non-bg linear component
#      of size 2-4, and it is the template stripe.
#    - The stripe is fully inside the reserved top-left zone, not adjacent to
#      the path, and disjoint from the end block and markers.


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


def _detect_template_stripe(
    g: np.ndarray, template_bg: int, path_color: int
) -> tuple[list[int], list[tuple[int, int]]]:
    """Find the top-left 1xN / Nx1 cycle stripe and return (pattern, cells)."""
    mask = (g != int(template_bg)) & (g != int(path_color))
    labels, n_comp = label_mask(mask, connectivity=4)
    candidates: list[tuple[tuple[int, int, int, int], list[tuple[int, int]]]] = []

    for k in range(1, n_comp + 1):
        coords = np.argwhere(labels == k)
        if coords.size == 0:
            continue
        cells = [(int(r), int(c)) for r, c in coords]
        if not (2 <= len(cells) <= 4):
            continue
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if not ((h == 1 and w == len(cells)) or (w == 1 and h == len(cells))):
            continue
        if r0 < 1 or c0 < 1 or r1 > 5 or c1 > 5:
            continue
        touches_path = False
        for r, c in cells:
            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1] and int(g[rr, cc]) == int(path_color):
                    touches_path = True
                    break
            if touches_path:
                break
        if touches_path:
            continue
        candidates.append(((r0, c0, r1, c1), cells))

    assert candidates, "template stripe not found"
    candidates.sort(key=lambda item: (item[0][0] + item[0][1], item[0][0], item[0][1]))
    r0, c0, r1, c1 = candidates[0][0]
    cells = candidates[0][1]
    if r0 == r1:
        cells = sorted(cells, key=lambda rc: rc[1])
    else:
        cells = sorted(cells, key=lambda rc: rc[0])
    pattern = [int(g[r, c]) for r, c in cells]
    return pattern, cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace color-cycling particles through a path while reading the cycle from a compact stripe template.

    Input structure:
    - Top-left: one connected 1xN or Nx1 color stripe defines the repeating cycle in reading order.
    - Isolated single pixels of a cycle color: start markers embedded in the left-edge path segment.
    - A distinct 2x2 monochrome block: terminal color marker in the opposite bg region.

    Rule:
    - Each marker launches a particle along the path, recoloring visited path cells by advancing through the stripe cycle.
    - When the particle tries to step off the path, that collision cell is painted with the terminal color and the particle attempts a 90° turn.
    - Turn direction alternates strictly: first collision turns left (CCW), second turns right (CW), then left again, and so on.
    - If the scheduled turn does not point to another path cell, the particle stops immediately.

    Key insight:
    - The path dynamics are unchanged; the new challenge is to identify that the cycle is encoded by a single linear stripe rather than by several 2x2 blocks.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    template_bg = int(g[0, 0])

    counts = Counter(map(int, g.flatten()))
    path_color = int(
        max([(cnt, int(col)) for col, cnt in counts.items() if int(col) != int(template_bg)])[1]
    )
    _assert_path_single_component(g, path_color)

    pattern, pat_cells = _detect_template_stripe(g, template_bg, path_color)

    end_color = None
    end_pos = None
    pat_set = set(map(int, pattern))
    for r in range(H - 1):
        for c in range(W - 1):
            v = int(g[r, c])
            if v == int(template_bg) or v == int(path_color) or v in pat_set:
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

    out = g.copy()
    for r, c in pat_cells:
        out[int(r), int(c)] = int(template_bg)
    if end_pos is not None:
        er, ec = end_pos
        out[int(er):int(er + 2), int(ec):int(ec + 2)] = int(template_bg)

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

    # Compact stripe only needs a small clear top-left pocket.
    for r in range(1, min(6, H)):
        c_left[r] = max(c_left[r], 6)
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
        mask[r, c_left[r]:c_right[r] + 1] = True
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


def _place_template_stripe(
    g: np.ndarray,
    mask: np.ndarray,
    cycle_colors: list[int],
) -> tuple[list[tuple[int, int]], str]:
    """Paint a horizontal or vertical stripe template in the reserved top-left bg pocket."""
    H, W = g.shape
    n = len(cycle_colors)
    orientations = ["h", "v"]
    random.shuffle(orientations)

    for orient in orientations:
        if orient == "h":
            candidates: list[list[tuple[int, int]]] = []
            for r in [1, 2]:
                for c in [1, 2]:
                    if c + n - 1 >= W:
                        continue
                    cells = [(r, c + k) for k in range(n)]
                    if all(not mask[rr, cc] and int(g[rr, cc]) == int(g[0, 0]) for rr, cc in cells):
                        candidates.append(cells)
            if candidates:
                cells = random.choice(candidates)
                for (rr, cc), col in zip(cells, cycle_colors):
                    g[rr, cc] = int(col)
                return cells, orient
        else:
            candidates = []
            for r in [1, 2]:
                for c in [1, 2]:
                    if r + n - 1 >= H:
                        continue
                    cells = [(r + k, c) for k in range(n)]
                    if all(not mask[rr, cc] and int(g[rr, cc]) == int(g[0, 0]) for rr, cc in cells):
                        candidates.append(cells)
            if candidates:
                cells = random.choice(candidates)
                for (rr, cc), col in zip(cells, cycle_colors):
                    g[rr, cc] = int(col)
                return cells, orient

    raise AssertionError("failed to place template stripe")


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

    g = np.full((H, W), bg, dtype=int)
    g[mask] = path_color

    try:
        _template_cells, _template_orient = _place_template_stripe(g, mask, cycle_colors)
    except AssertionError:
        return None

    end_candidates = []
    for r in range(H // 2, H - 1):
        for c in range(W // 2, W - 1):
            if not (g[r:r + 2, c:c + 2] == bg).all():
                continue
            r0, r1_ = max(0, r - 1), min(H, r + 3)
            c0, c1_ = max(0, c - 1), min(W, c + 3)
            if (g[r0:r1_, c0:c1_] == path_color).any():
                continue
            end_candidates.append((r, c))
    if not end_candidates:
        return None
    er, ec = random.choice(end_candidates)
    g[er:er + 2, ec:ec + 2] = end_color

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
