# Variant of 195c6913
# Dimension: D17 — bg分隔条/峡湾形态 (obstacle)
# Change: n_markers=2时，两个标记的轨迹由bg分隔条分开：~75%为从col 0向右上延伸的峡湾，~25%为path内部... → 分隔条为path色而非bg色

"""
Design analysis for the D17 variant.

a) Concrete form and behavior trace
- The global layout stays the same: top-left cycle template, one end-color 2x2
  block in the opposite bg region, and one or two markers on col 0 embedded in
  a thick anti-diagonal band.
- The D17 change is local but structural: when there are two markers, the thing
  that separates their channels is no longer a bg gap. It is an elongated,
  single-color strip placed *inside* the band. Visually it looks like a grey
  wall embedded in the corridor.
- Traversable cells are still exactly the dominant path color. The new strip is
  non-traversable even though it is not bg.
- Small trace:
  Original: a marker walks right until the next cell is the carved bg fjord,
  stamps that bg cell with end_color, then alternates its turn.
  Variant: the same geometric separator cell is obstacle_color instead of bg,
  so the marker now collides with a colored wall, stamps that wall cell with
  end_color, and then alternates its turn. The movement logic changes only in
  what counts as "wall", but the input becomes visibly different.

b) Complexity calibration
- Only one extra structural color is introduced: obstacle_color.
- The obstacle still has just two shape modes, matching the old separator
  distribution: about 75% left-edge fjord, about 25% enclosed interior island.
- Keeping the old geometry but changing the material/color makes the variant
  easy to perceive without adding ambiguous extra freedoms.

c) Hidden-assumption audit
- Original end-block detection scanned for the first 2x2 monochrome patch not
  equal to bg/path/template colors. That becomes invalid because a long colored
  obstacle naturally contains many 2x2 monochrome subpatches.
- Therefore the solver must identify the end block as a full same-color
  connected component whose area is exactly 4 and whose bbox is exactly 2x2.
- Marker detection and path following can still use the dominant path color, but
  collision painting must now handle obstacle_color as well as bg.

d) Placement legality
- The generator first builds the original wide band, then carves the separator
- from traversable path cells so the remaining path_color stays one 4-connected
  component. The carved cells are finally recolored as obstacle_color.
- Template blocks are placed only on true bg, never on traversable path or on
  the obstacle strip.
- The end block is placed in a bottom-right bg pocket with a one-cell bg margin
  from any structural cell.

e) Visual observability
- In n_markers=2 inputs, humans can now see an internal colored wall inside the
  band instead of empty bg biting into it. In outputs, at least one collision is
  required to land on that wall so the wall actively participates in the rule.

f) Visual cleanliness
- The obstacle is kept as one elongated component rather than several fragments.
- It stays away from the template zone and end block, so objects remain legible.

g) Color-change inferability
- No sorting or logic depends on numeric color ids.
- The only new color role is "internal wall"; it is supported by repeated shape
  evidence inside each n_markers=2 sample and by output hit marks on the wall.
"""

from __future__ import annotations

from collections import Counter
import os
import random
import sys

import numpy as np

_SOLVERS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "solvers")
)
if _SOLVERS_DIR not in sys.path:
    sys.path.append(_SOLVERS_DIR)

from arc_utils import label_mask


def _assert_path_single_component(g: np.ndarray, path_color: int) -> None:
    """Eval invariant: traversable path_color stays one 4-connected component."""
    _, n_path = label_mask(g == path_color, connectivity=4)
    assert n_path == 1, f"path_color must be a single 4-connected component, got {n_path}"


def _assert_marker_embedded_in_path(
    g: np.ndarray, markers: list[tuple[int, int, int]], path_color: int
) -> None:
    """Eval invariant: every col-0 marker is embedded inside a vertical path run."""
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


def _find_pattern(
    g: np.ndarray, template_bg: int, path_color: int
) -> tuple[list[int], list[tuple[int, int]]]:
    H, W = g.shape
    best_pattern: list[int] = []
    best_cells: list[tuple[int, int]] = []
    for base_r in [1, 2]:
        if base_r + 1 >= H:
            continue
        row_pattern: list[int] = []
        row_cells: list[tuple[int, int]] = []
        for base_c in [1, 4, 7, 10]:
            if base_c + 1 >= W:
                break
            v = int(g[int(base_r), int(base_c)])
            if v == int(template_bg) or v == int(path_color):
                break
            if not (
                int(g[int(base_r + 1), int(base_c)]) == v
                and int(g[int(base_r), int(base_c + 1)]) == v
                and int(g[int(base_r + 1), int(base_c + 1)]) == v
            ):
                break
            row_pattern.append(int(v))
            row_cells.extend(
                [
                    (int(base_r), int(base_c)),
                    (int(base_r + 1), int(base_c)),
                    (int(base_r), int(base_c + 1)),
                    (int(base_r + 1), int(base_c + 1)),
                ]
            )
        if len(row_pattern) > len(best_pattern):
            best_pattern = row_pattern
            best_cells = row_cells
    return best_pattern, best_cells


def _find_end_block(
    g: np.ndarray, template_bg: int, path_color: int, pat_set: set[int]
) -> tuple[int | None, tuple[int, int] | None]:
    """Find the unique 2x2 connected component used as the end block.

    This intentionally ignores obstacle stripes that may contain 2x2 same-color
    subwindows, because those belong to components larger than area 4.
    """
    H, W = g.shape
    candidates: list[tuple[bool, int, int, int]] = []
    for color in sorted(set(map(int, g.flatten()))):
        if color in {int(template_bg), int(path_color)} or color in pat_set:
            continue
        lab, n = label_mask(g == int(color), connectivity=4)
        for k in range(1, n + 1):
            coords = np.argwhere(lab == k)
            if len(coords) != 4:
                continue
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            if int(r1 - r0) != 1 or int(c1 - c0) != 1:
                continue
            if not np.all(g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == int(color)):
                continue
            in_br = int(r0) >= H // 2 and int(c0) >= W // 2
            candidates.append((bool(in_br), int(r0), int(c0), int(color)))
    if not candidates:
        return None, None
    candidates.sort(key=lambda x: (x[0], x[1] + x[2], x[1], x[2]), reverse=True)
    _in_br, r0, c0, color = candidates[0]
    return int(color), (int(r0), int(c0))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate cycle-painting particles inside a band that may contain a colored internal wall.

    Input structure:
    - Top-left 2x2 monochrome blocks encode the repeating paint cycle.
    - Isolated cycle-colored pixels on col 0 are start markers embedded in the main path.
    - A separate 2x2 block gives the terminal collision color.
    - With two markers, the band can also include a thin obstacle strip of a distinct structural color.

    Rule:
    - Only the dominant path color is traversable; the internal strip behaves like a wall.
    - Each marker travels straight through traversable cells, repainting them by advancing through the cycle.
    - Stepping into bg or into the obstacle strip counts as a collision: recolor that hit cell with the end color, then alternate turns left, right, left, right.
    - If the scheduled turn does not land on traversable path, the particle stops; template blocks and the end block are erased from the output.

    Key insight:
    - The crucial distinction is between "same region visually" and "same movement class": a colored stripe can sit inside the band but still be non-traversable, so collision logic must follow the dominant path color rather than treating every non-bg cell as walkable.
    """
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

    pattern, pat_cells = _find_pattern(g, int(template_bg), int(path_color))
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

            if end_color is not None:
                out[nr, nc] = int(end_color)

            left = (-int(dc), int(dr))  # CCW
            right = (int(dc), -int(dr))  # CW
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


def _build_band(H: int, W: int, lobe_rows: list[int] | None = None) -> np.ndarray:
    """Return a thick anti-diagonal band used for the traversable path."""
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
    """Carve a separator shape from the traversable path mask."""
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


def _make_obstacle_separator(
    base_mask: np.ndarray, r1: int, r2: int
) -> tuple[np.ndarray, np.ndarray, bool] | None:
    """Return (traversable_path_mask, obstacle_mask, use_interior)."""
    trial = base_mask.copy()
    if not _carve_fjord(trial, r1, r2):
        return None

    use_interior = random.random() < 0.25
    if use_interior:
        K = random.randint(2, 4)
        pre_carve_cells = base_mask & ~trial
        for c in range(min(K, base_mask.shape[1])):
            trial[:, c] |= pre_carve_cells[:, c]

    obstacle_mask = base_mask & ~trial
    if obstacle_mask.sum() < 6:
        return None
    _, n_path = label_mask(trial, connectivity=4)
    _, n_obs = label_mask(obstacle_mask, connectivity=4)
    if n_path != 1 or n_obs != 1:
        return None
    if use_interior and obstacle_mask[:, 0].any():
        return None
    if not use_interior and not obstacle_mask[:, 0].any():
        return None
    return trial, obstacle_mask, use_interior


def _gen_attempt() -> np.ndarray | None:
    H = W = random.randint(20, 30)
    n_markers = 2 if random.random() < 0.7 else 1

    palette = list(range(10))
    random.shuffle(palette)
    bg = palette[0]
    path_color = palette[1]
    end_color = palette[2]
    obstacle_color = palette[3]
    n_blocks = random.randint(2, 4)
    cycle_pool = palette[4:]
    cycle_colors = [random.choice(cycle_pool) for _ in range(n_blocks)]

    if n_markers == 1:
        r1 = random.randint(H // 2, H - 4)
        markers = [r1]
    else:
        r1 = random.randint(H // 3, H // 2)
        r2 = random.randint(max(r1 + 5, 2 * H // 3), H - 4)
        markers = [r1, r2]

    base_mask = _build_band(H, W, lobe_rows=markers)
    if base_mask[0, 0]:
        return None
    _, n_base = label_mask(base_mask, connectivity=4)
    if n_base != 1:
        return None
    if not base_mask[0, :].any() or not base_mask[:, W - 1].any():
        return None
    if not base_mask[H - 1, :].any() or not base_mask[:, 0].any():
        return None
    for r in markers:
        if not (base_mask[r, 0] and base_mask[r - 1, 0] and base_mask[r + 1, 0]):
            return None

    path_mask = base_mask.copy()
    obstacle_mask = np.zeros_like(base_mask, dtype=bool)
    if n_markers == 2:
        sep = _make_obstacle_separator(base_mask, r1, r2)
        if sep is None:
            return None
        path_mask, obstacle_mask, _use_interior = sep
        for r in markers:
            if not (path_mask[r, 0] and path_mask[r - 1, 0] and path_mask[r + 1, 0]):
                return None

    structural_mask = path_mask | obstacle_mask
    _, n_bg = label_mask(~structural_mask, connectivity=4)
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
                if rr >= H or cc >= W or structural_mask[rr, cc]:
                    return None

    g = np.full((H, W), bg, dtype=int)
    g[path_mask] = path_color
    g[obstacle_mask] = obstacle_color

    for i, bc in enumerate(block_cols):
        g[template_row : template_row + 2, bc : bc + 2] = cycle_colors[i]

    end_candidates = []
    for r in range(H // 2, H - 1):
        for c in range(W // 2, W - 1):
            if not (g[r : r + 2, c : c + 2] == bg).all():
                continue
            r0, r1_ = max(0, r - 1), min(H, r + 3)
            c0, c1_ = max(0, c - 1), min(W, c + 3)
            if (g[r0:r1_, c0:c1_] != bg).any():
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
        out = main(g)
    except Exception:
        return None

    if n_markers == 2:
        if obstacle_mask.sum() == 0:
            return None
        if not np.any(obstacle_mask & (out == end_color)):
            return None
        if not np.any(obstacle_mask & (out == obstacle_color)):
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
