# Variant of 195c6913
# Dimension: D9 — 标记数量与基数关系 (meta)
# Change: 1或2个标记（n_markers==2出现率>=50%），各标记独立模拟，轨迹必须不相交。... → 标记之间有依赖顺序

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import label_mask


# Design analysis
# ----------------
# a) Concretization and behavior trace
#    - Objects stay the same: a left-to-right 2x2 template cycle at top-left,
#      two single-pixel markers on col 0, one 2x2 end block in the opposite
#      background region, and one connected path color region.
#    - The D9 change is made visible by ordering markers by row, bottom-to-top.
#      When there are 2 markers, the lower marker is simulated first on the
#      original path. Its painted trail is then frozen and treated as solid
#      terrain by the upper marker.
#    - Generator builds a shared elbow corridor:
#        lower marker: right along its row -> turn up at column x -> turn right
#        upper marker: right along its own row toward the same column x
#      In the original task the upper marker would pass through x because every
#      marker reads the original path. In this variant the cell at (upper_row, x)
#      has already been recolored by the lower marker, so the upper marker hits a
#      non-path obstacle and stops there. This is the intended multi-step
#      dependency.
# b) Complexity calibration
#    - The only new control variable is row order; it is directly visible.
#    - The alternating-turn rule, template-cycle recoloring, and end-color corner
#      marks are kept unchanged, so the task is still recognizably the same
#      family rather than a new puzzle.
#    - The generator does not permit arbitrary dependency shapes. It uses one
#      shared column x and one fixed row order, which is enough to make the
#      change observable without introducing too many hidden cases.
# c) Old implicit assumptions that must change
#    - Original solver tested passability against immutable `g`, so later
#      markers ignored earlier paint. That is invalid here.
#    - A naive mutation of the occupancy grid at every step would also be wrong:
#      the current marker would block itself on its own freshly painted cells.
#      The correct behavior is: each marker walks on a board that already
#      contains previous markers' trails, but not its own current trail.
#    - Original pairwise-disjoint assertion is removed. Shared intended route
#      cells are now legal in the input because only execution order decides
#      whether they stay traversable.
# d) Placement legality
#    - Both markers remain on col 0 and keep path-colored vertical neighbors.
#    - The template rows 1-3 / cols 1-11 are reserved as background, so the
#      2x2 cycle blocks always fit.
#    - The shared dependency column x is placed well to the right of the
#      template, and the top-right connector uses the last column to touch the
#      top edge without entering the template zone.
#    - The end block is searched in the bottom-right background region with a
#      one-cell moat from the path, as in the original task style.
# e) Visual observability
#    - With two markers, one trail visibly appears first and the other visibly
#      truncates against it. This difference can be inferred from 2-3 examples
#      because the blocking relation always follows row order and a shared
#      corridor, never color id.
# f) Visual cleanliness
#    - The path is a tidy orthogonal route with one shared junction column.
#      Template, markers, path, and end block stay spatially separated.
# g) Color-change inferability
#    - Color cycling still comes only from the top-left template. No extra
#      color-dependent rule is added.
# h) Generator-side variant validation
#    - For n=2, generator requires at least one attempted move into a previous
#      trail cell (`dependency_hits >= 1`) and also requires the sequential
#      output to differ from an independent counterfactual simulation.
# i) New solver assumptions audited against generator
#    - Markers are sorted by row descending; generator always uses distinct rows.
#    - Later-marker dependency only matters if the earlier trail reaches the
#      shared column on the later marker's row; generator explicitly builds that.
#    - Current-marker motion assumes its own trail is not an obstacle; solver
#      enforces this by committing painted cells to the occupancy board only
#      after that marker finishes.


def _assert_path_single_component(g: np.ndarray, path_color: int) -> None:
    """Eval invariant: path_color forms one 4-connected component."""
    _, n_path = label_mask(g == path_color, connectivity=4)
    assert n_path == 1, f"path_color must be a single 4-connected component, got {n_path}"


def _assert_marker_embedded_in_path(
    g: np.ndarray, markers: list[tuple[int, int, int]], path_color: int
) -> None:
    """Each marker must overwrite a col-0 path cell with path-colored vertical neighbors."""
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


def _detect_pattern(
    g: np.ndarray, template_bg: int, path_color: int
) -> tuple[list[int], list[tuple[int, int]]]:
    pattern: list[int] = []
    pat_cells: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    H, W = g.shape
    for r in range(min(6, H - 1)):
        for c in range(min(25, W - 1)):
            if (r, c) in seen:
                continue
            v = int(g[r, c])
            if v == template_bg or v == path_color:
                continue
            if (
                int(g[r + 1, c]) == v
                and int(g[r, c + 1]) == v
                and int(g[r + 1, c + 1]) == v
            ):
                pattern.append(v)
                for rr, cc in [(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)]:
                    seen.add((rr, cc))
                    pat_cells.append((rr, cc))
    return pattern, pat_cells


def _detect_end_block(
    g: np.ndarray, template_bg: int, path_color: int, pat_set: set[int]
) -> tuple[int | None, tuple[int, int] | None]:
    H, W = g.shape
    for r in range(H - 1):
        for c in range(W - 1):
            v = int(g[r, c])
            if v == template_bg or v == path_color or v in pat_set:
                continue
            if (
                int(g[r + 1, c]) == v
                and int(g[r, c + 1]) == v
                and int(g[r + 1, c + 1]) == v
            ):
                return v, (r, c)
    return None, None


def _detect_markers(g: np.ndarray, pat_set: set[int], path_color: int) -> list[tuple[int, int, int]]:
    H, W = g.shape
    markers: list[tuple[int, int, int]] = []
    for r in range(H):
        for c in range(W):
            v = int(g[r, c])
            if v not in pat_set:
                continue
            same = 0
            has_path = False
            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < H and 0 <= cc < W:
                    if int(g[rr, cc]) == v:
                        same += 1
                    if int(g[rr, cc]) == path_color:
                        has_path = True
            if same == 0 and has_path:
                markers.append((r, c, v))
    return markers


def _initial_direction(board: np.ndarray, sr: int, sc: int, path_color: int) -> tuple[int, int] | None:
    H, W = board.shape
    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # R, D, L, U
        rr = sr + dr
        cc = sc + dc
        if 0 <= rr < H and 0 <= cc < W and int(board[rr, cc]) == path_color:
            return dr, dc
    return None


def _simulate_markers(
    g: np.ndarray,
    out: np.ndarray,
    pattern: list[int],
    path_color: int,
    template_bg: int,
    end_color: int | None,
    markers: list[tuple[int, int, int]],
    *,
    sequential: bool,
) -> dict[str, object]:
    H, W = g.shape
    marker_order = sorted(markers, key=lambda x: (int(x[0]), int(x[1])), reverse=True)
    occupied = g.copy()
    previous_trail: set[tuple[int, int]] = set()
    trace_sets: list[set[tuple[int, int]]] = []
    dependency_hits = 0
    dependency_markers = 0

    for sr, sc, sv in marker_order:
        board = occupied if sequential else g
        init = _initial_direction(board, int(sr), int(sc), int(path_color))
        if init is None or not pattern:
            trace_sets.append(set())
            continue

        r, c = int(sr), int(sc)
        dr, dc = int(init[0]), int(init[1])
        idx = int(pattern.index(int(sv))) if int(sv) in pattern else 0
        hits = 0
        visited: set[tuple[int, int, int, int, int, int]] = set()
        my_cells: list[tuple[int, int, int]] = []
        my_set: set[tuple[int, int]] = set()
        hit_previous = False

        for _ in range(H * W * 4):
            state = (r, c, dr, dc, idx, hits & 1)
            if state in visited:
                break
            visited.add(state)

            nr = r + dr
            nc = c + dc
            if not (0 <= nr < H and 0 <= nc < W):
                break

            cell = int(board[nr, nc])
            if cell == int(path_color):
                idx = int((idx + 1) % len(pattern))
                color = int(pattern[idx])
                out[nr, nc] = color
                my_cells.append((nr, nc, color))
                my_set.add((nr, nc))
                r, c = nr, nc
                continue

            if sequential and (nr, nc) in previous_trail and not hit_previous:
                hit_previous = True
                dependency_markers += 1
            if sequential and (nr, nc) in previous_trail:
                dependency_hits += 1

            if cell == int(template_bg) and end_color is not None:
                out[nr, nc] = int(end_color)

            left = (-dc, dr)
            right = (dc, -dr)
            tdr, tdc = left if (hits % 2 == 0) else right
            rr = r + tdr
            cc = c + tdc
            if 0 <= rr < H and 0 <= cc < W and int(board[rr, cc]) == int(path_color):
                dr, dc = int(tdr), int(tdc)
                hits += 1
                continue
            break

        if sequential:
            for rr, cc, color in my_cells:
                occupied[rr, cc] = int(color)
                previous_trail.add((rr, cc))
        trace_sets.append(my_set)

    return {
        "marker_order": marker_order,
        "trace_sets": trace_sets,
        "dependency_hits": int(dependency_hits),
        "dependency_markers": int(dependency_markers),
    }


def _solve_with_mode(grid: np.ndarray, *, sequential: bool) -> tuple[np.ndarray, dict[str, object]]:
    g = np.array(grid, dtype=int)
    H, W = g.shape

    template_bg = int(g[0, 0])
    counts = Counter(map(int, g.flatten()))
    path_color = int(
        max((cnt, int(col)) for col, cnt in counts.items() if int(col) != int(template_bg))[1]
    )
    _assert_path_single_component(g, path_color)

    pattern, pat_cells = _detect_pattern(g, template_bg, path_color)
    pat_set = set(map(int, pattern))
    end_color, end_pos = _detect_end_block(g, template_bg, path_color, pat_set)
    markers = _detect_markers(g, pat_set, path_color)
    _assert_marker_embedded_in_path(g, markers, path_color)

    out = g.copy()
    for r, c in pat_cells:
        out[r, c] = int(template_bg)
    if end_pos is not None:
        er, ec = end_pos
        out[er : er + 2, ec : ec + 2] = int(template_bg)

    stats = _simulate_markers(
        g,
        out,
        pattern,
        path_color,
        template_bg,
        end_color,
        markers,
        sequential=sequential,
    )
    stats.update(
        {
            "pattern": pattern,
            "path_color": int(path_color),
            "template_bg": int(template_bg),
            "end_color": end_color,
            "end_pos": end_pos,
            "markers": markers,
            "shape": (H, W),
        }
    )
    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate ordered particles along a path where earlier trails become solid obstacles for later markers.

    Input structure:
    - Top-left: 2x2 monochrome blocks encode the repeating paint cycle from left to right.
    - One or two isolated cycle-colored pixels sit on col 0 as start markers, each embedded in the path.
    - A separate 2x2 monochrome block marks the collision color used on background impacts.

    Rule:
    - Markers are processed in row order from bottom to top, not independently.
    - Each marker moves straight on path cells, advancing through the cycle colors as it paints.
    - On hitting non-path, it follows the same strict CCW/CW alternating turn schedule; only a background impact gets recolored to the end color.
    - After one marker finishes, every path cell it painted becomes blocked terrain for the next marker.

    Key insight: The alternating-turn mechanic still controls each route locally, but the global behavior now depends on visible marker order because later markers must react to trails left by earlier ones.
    """
    out, _ = _solve_with_mode(grid, sequential=True)
    return out


def _build_path_mask(H: int, W: int, marker_rows: list[int]) -> np.ndarray:
    mask = np.zeros((H, W), dtype=bool)
    marker_rows = sorted(marker_rows)

    if len(marker_rows) == 1:
        rb = int(marker_rows[0])
        rt = random.randint(4, max(4, rb - 3))
        x = random.randint(max(13, W // 2), W - 5)

        mask[rb - 1 : H, 1] = True
        mask[rb, 1 : x + 1] = True
        mask[rt : rb + 1, x] = True
        mask[rt, x:W] = True
        mask[0 : rt + 1, W - 1] = True
        mask[rb - 1 : rb + 2, 0:2] = True
    else:
        ru, rb = int(marker_rows[0]), int(marker_rows[1])
        rt = random.randint(4, max(4, ru - 2))
        x = random.randint(max(13, W // 2), W - 5)
        upper_end = random.randint(min(W - 3, x + 2), min(W - 3, x + 6))

        mask[ru - 1 : H, 1] = True
        mask[ru, 1 : upper_end + 1] = True
        mask[rb, 1 : x + 1] = True
        mask[rt : rb + 1, x] = True
        mask[rt, x:W] = True
        mask[0 : rt + 1, W - 1] = True
        for r in [ru, rb]:
            mask[r - 1 : r + 2, 0:2] = True

    return mask


def _place_end_block(g: np.ndarray, bg: int, path_color: int) -> tuple[int, int] | None:
    H, W = g.shape
    candidates: list[tuple[int, int]] = []
    for r in range(H // 2, H - 1):
        for c in range(W // 2, W - 1):
            if not (g[r : r + 2, c : c + 2] == bg).all():
                continue
            r0, r1 = max(0, r - 1), min(H, r + 3)
            c0, c1 = max(0, c - 1), min(W, c + 3)
            if (g[r0:r1, c0:c1] == path_color).any():
                continue
            candidates.append((r, c))
    if not candidates:
        return None
    return random.choice(candidates)


def _gen_attempt() -> np.ndarray | None:
    H = W = random.randint(20, 30)
    n_markers = 2

    palette = list(range(10))
    random.shuffle(palette)
    bg = palette[0]
    path_color = palette[1]
    end_color = palette[2]
    n_blocks = random.randint(2, 4)
    cycle_colors = random.sample(palette[3:], n_blocks)

    if n_markers == 1:
        marker_rows = [random.randint(H // 2, H - 5)]
    else:
        ru = random.randint(6, H // 2 - 1)
        rb = random.randint(max(ru + 5, H // 2 + 1), H - 5)
        marker_rows = [ru, rb]

    mask = _build_path_mask(H, W, marker_rows)
    if mask[0, 0]:
        return None
    if mask[:, 0].sum() < (2 * len(marker_rows)):
        return None

    _, n_path = label_mask(mask, connectivity=4)
    if n_path != 1:
        return None
    if not mask[0, :].any() or not mask[:, W - 1].any():
        return None
    if not mask[H - 1, :].any() or not mask[:, 0].any():
        return None

    for r in marker_rows:
        if not (1 <= r < H - 1):
            return None
        if not (mask[r, 0] and mask[r - 1, 0] and mask[r + 1, 0]):
            return None
        if not mask[r, 1]:
            return None

    template_row = random.choice([1, 2])
    block_cols = [1, 4, 7, 10][:n_blocks]
    if block_cols[-1] + 1 >= W:
        return None
    for bc in block_cols:
        if mask[template_row : template_row + 2, bc : bc + 2].any():
            return None

    g = np.full((H, W), bg, dtype=int)
    g[mask] = path_color

    for i, bc in enumerate(block_cols):
        g[template_row : template_row + 2, bc : bc + 2] = int(cycle_colors[i])

    end_pos = _place_end_block(g, bg, path_color)
    if end_pos is None:
        return None
    er, ec = end_pos
    g[er : er + 2, ec : ec + 2] = int(end_color)

    for r in marker_rows:
        g[r, 0] = int(cycle_colors[0])

    _, n_path_after = label_mask(g == path_color, connectivity=4)
    if n_path_after != 1:
        return None
    if int(g[0, 0]) != int(bg):
        return None

    cnt = Counter(map(int, g.flatten()))
    non_bg = sorted([(c, col) for col, c in cnt.items() if int(col) != int(bg)], reverse=True)
    if not non_bg or int(non_bg[0][1]) != int(path_color):
        return None

    try:
        seq_out, seq_stats = _solve_with_mode(g, sequential=True)
        _ = seq_out
    except Exception:
        return None

    if n_markers == 2:
        try:
            indep_out, _ = _solve_with_mode(g, sequential=False)
        except Exception:
            return None
        if int(seq_stats["dependency_hits"]) < 1:
            return None
        if np.array_equal(seq_out, indep_out):
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
