# Combo variant of 195c6913 (index 8)
# combo_spec: D9/2+D15/2
# Name: 标记间有依赖顺序+按从上到下顺序模拟

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import label_mask


# Design analysis
# ----------------
# a) Concretization and behavior trace
#    - The visible objects stay close to the original task: a top-left row of
#      2x2 template blocks, two col-0 start markers using the first cycle
#      color, one 2x2 end block in the opposite background region, and one
#      connected path-color structure.
#    - The fused change is implemented as a fixed top-to-bottom execution order.
#      If the upper marker is at row ru and the lower marker at row rb, the
#      upper marker is always simulated first. Its painted trail is committed to
#      the occupancy board only after its own run finishes.
#    - Generator uses a dependency corridor:
#        upper marker: Down on col 0 -> Right on row rb-1 -> Down at col x
#                      -> Right on row rb
#        lower marker: Right on row rb toward the same col x
#      In the original task the lower marker would continue through row rb
#      because row-rb path cells stay traversable. In this combo variant the
#      upper marker has already recolored cell (rb, x), so the lower marker
#      collides with a fresh obstacle and stops because its scheduled turn cell
#      is also inside the earlier trail.
# b) Complexity calibration
#    - Only one new dependency axis is introduced: row order from top to bottom.
#      That is easy to observe and repeats across samples.
#    - The original alternating-turn mechanic, template-driven color cycle, and
#      end-color collision marks are preserved, so the puzzle remains in the
#      same family instead of becoming a different task.
#    - The generator uses one shared dependency column x and a small number of
#      orthogonal corridors. This keeps the rule legible and avoids hidden
#      branches that would be hard to infer from 2-4 examples.
# c) Old implicit assumptions that must change
#    - Original `main()` checked passability against immutable `g`; that makes
#      markers independent and is invalid here.
#    - Committing paint immediately at every step would also be wrong because a
#      marker would block itself. The correct board for one marker is:
#      original path + all earlier finished trails, but excluding its own
#      current trail until it stops.
#    - Original disjoint-trajectory assumption is removed. Shared intended route
#      cells are now legal in the input because ordering decides who gets them.
# d) Placement legality
#    - Both markers remain at col 0 with path-colored vertical neighbors.
#    - The template zone stays in rows 1-3 and cols 1-11, while the dependency
#      corridor is placed lower and farther right, so template blocks never
#      overlap the path.
#    - The end block is placed in a background patch below the lower route with
#      a one-cell moat from the path, matching the original object grammar.
# e) Visual observability
#    - The crucial difference is visible: the upper trail exists first, and the
#      lower marker truncates exactly where it meets that trail. Since ordering
#      is purely geometric (higher row first), the rule can be inferred from a
#      few examples without relying on hidden metadata or color ids.
# f) Visual cleanliness
#    - The path is an orthogonal scaffold with clear separation among template,
#      markers, dependency corridor, and end block. Overlap happens only where
#      the rule intends it: later motion meeting earlier paint.
# g) Color-change inferability
#    - No extra color mapping is introduced. Cycle colors still come only from
#      the template, and the end color still only marks background impacts.
# h) Generator-side variant validation
#    - Every sample is checked twice: sequential top-to-bottom and independent.
#      We require at least one collision into a previous trail and require the
#      two outputs to differ, which rejects degenerate inputs that behave like
#      the original task.
# i) New solver assumptions audited against generator
#    - Markers are ordered strictly by row ascending; generator always uses two
#      distinct rows with ample spacing.
#    - The lower marker's first blocked attempt must target a cell painted by
#      the upper marker; generator explicitly builds the shared column x.
#    - The turn cell after that collision must also be blocked by the same
#      earlier trail so the lower marker visibly stops; generator guarantees
#      that by letting the upper marker paint row rb-1 before row rb.


def _assert_path_single_component(g: np.ndarray, path_color: int) -> None:
    """Path color must remain one 4-connected component."""
    _, n_path = label_mask(g == path_color, connectivity=4)
    assert n_path == 1, f"path_color must be a single 4-connected component, got {n_path}"


def _assert_marker_embedded_in_path(
    g: np.ndarray, markers: list[tuple[int, int, int]], path_color: int
) -> None:
    """Each col-0 marker must overwrite a path cell with path neighbors above/below."""
    h, _ = g.shape
    for r, c, _ in markers:
        if c != 0:
            continue
        if r > 0:
            assert int(g[r - 1, c]) == path_color, (
                f"marker at ({r},{c}) up-neighbor ({r-1},{c})={int(g[r-1,c])} "
                f"must be path_color {path_color}"
            )
        if r < h - 1:
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
    h, w = g.shape
    for r in range(min(6, h - 1)):
        for c in range(min(25, w - 1)):
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
                for rr, cc in ((r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)):
                    seen.add((rr, cc))
                    pat_cells.append((rr, cc))
    return pattern, pat_cells


def _detect_end_block(
    g: np.ndarray, template_bg: int, path_color: int, pat_set: set[int]
) -> tuple[int | None, tuple[int, int] | None]:
    h, w = g.shape
    for r in range(h - 1):
        for c in range(w - 1):
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
    h, w = g.shape
    markers: list[tuple[int, int, int]] = []
    for r in range(h):
        for c in range(w):
            v = int(g[r, c])
            if v not in pat_set:
                continue
            same = 0
            has_path = False
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    if int(g[rr, cc]) == v:
                        same += 1
                    if int(g[rr, cc]) == path_color:
                        has_path = True
            if same == 0 and has_path:
                markers.append((r, c, v))
    return markers


def _initial_direction(board: np.ndarray, sr: int, sc: int, path_color: int) -> tuple[int, int] | None:
    h, w = board.shape
    for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):  # R, D, L, U
        rr = sr + dr
        cc = sc + dc
        if 0 <= rr < h and 0 <= cc < w and int(board[rr, cc]) == path_color:
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
    h, w = g.shape
    marker_order = sorted(markers, key=lambda x: (int(x[0]), int(x[1])))
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

        for _ in range(h * w * 4):
            state = (r, c, dr, dc, idx, hits & 1)
            if state in visited:
                break
            visited.add(state)

            nr = r + dr
            nc = c + dc
            if not (0 <= nr < h and 0 <= nc < w):
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
            if 0 <= rr < h and 0 <= cc < w and int(board[rr, cc]) == int(path_color):
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
    h, w = g.shape

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
            "shape": (h, w),
        }
    )
    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate ordered path-following markers where earlier paint becomes a real obstacle for later markers.

    Input structure:
    - Top-left 2x2 monochrome blocks encode a repeating color cycle from left to right.
    - Two isolated cycle-colored pixels on col 0 act as start markers, each embedded inside the path region.
    - A separate 2x2 monochrome block gives the color used when a marker hits background.

    Rule:
    - Markers are not independent: process them by row from top to bottom.
    - While one marker moves, it still follows the original straight-step and strict alternating CCW/CW turn schedule, painting path cells by advancing through the template cycle.
    - After a marker finishes, every cell it painted becomes blocked terrain for all later markers; only background collision cells receive the end color.
    - If the scheduled turn cell is not path on that updated board, the current marker stops immediately.

    Key insight: The local motion rule is unchanged, but the global outcome now depends on visible vertical order because the upper trail freezes first and reshapes what the lower marker can still traverse.
    """
    out, _ = _solve_with_mode(grid, sequential=True)
    return out


def _build_path_mask(h: int, w: int, marker_rows: list[int]) -> tuple[np.ndarray, int]:
    ru, rb = sorted(int(r) for r in marker_rows)
    x_lo = max(13, w // 2)
    x_hi = w - 5
    if x_lo > x_hi:
        raise ValueError("grid too narrow for dependency corridor")
    x = random.randint(x_lo, x_hi)

    mask = np.zeros((h, w), dtype=bool)

    # Lower-left trunk to the bottom edge.
    mask[ru:h, 0] = True

    # Upper connector around the top marker gap.
    mask[ru - 1, 0 : w] = True
    mask[0 : ru, w - 1] = True

    # Dependency elbow used by the upper marker.
    mask[rb - 1, 0 : x + 1] = True
    mask[ru - 1 : rb + 1, x] = True

    # Lower marker corridor and bypass around its marker cell.
    mask[rb - 1 : rb + 2, 0 : 2] = True
    mask[rb, 1:w] = True

    return mask, x


def _place_end_block(g: np.ndarray, bg: int, path_color: int, rb: int) -> tuple[int, int] | None:
    h, w = g.shape
    candidates: list[tuple[int, int]] = []
    row_start = max(rb + 2, h // 2)
    for r in range(row_start, h - 1):
        for c in range(w // 2, w - 1):
            if not (g[r : r + 2, c : c + 2] == bg).all():
                continue
            r0, r1 = max(0, r - 1), min(h, r + 3)
            c0, c1 = max(0, c - 1), min(w, c + 3)
            if (g[r0:r1, c0:c1] == path_color).any():
                continue
            candidates.append((r, c))
    if not candidates:
        return None
    return random.choice(candidates)


def _gen_attempt() -> np.ndarray | None:
    h = w = random.randint(20, 30)

    palette = list(range(10))
    random.shuffle(palette)
    bg = int(palette[0])
    path_color = int(palette[1])
    end_color = int(palette[2])
    n_blocks = random.randint(2, 4)
    cycle_colors = random.sample(palette[3:], n_blocks)

    ru = random.randint(6, max(6, h // 2 - 2))
    rb = random.randint(max(ru + 5, h // 2 + 1), h - 4)
    marker_rows = [ru, rb]

    try:
        mask, _x = _build_path_mask(h, w, marker_rows)
    except ValueError:
        return None

    if mask[0, 0]:
        return None

    _, n_path = label_mask(mask, connectivity=4)
    if n_path != 1:
        return None
    if not mask[0, :].any() or not mask[:, w - 1].any():
        return None
    if not mask[h - 1, :].any() or not mask[:, 0].any():
        return None

    for r in marker_rows:
        if not (1 <= r < h - 1):
            return None
        if not (mask[r, 0] and mask[r - 1, 0] and mask[r + 1, 0]):
            return None

    template_row = random.choice([1, 2])
    block_cols = [1, 4, 7, 10][:n_blocks]
    if block_cols[-1] + 1 >= w:
        return None
    for bc in block_cols:
        if mask[template_row : template_row + 2, bc : bc + 2].any():
            return None

    g = np.full((h, w), bg, dtype=int)
    g[mask] = path_color

    for i, bc in enumerate(block_cols):
        g[template_row : template_row + 2, bc : bc + 2] = int(cycle_colors[i])

    end_pos = _place_end_block(g, bg, path_color, rb)
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
    non_bg = sorted(((c, col) for col, c in cnt.items() if int(col) != int(bg)), reverse=True)
    if not non_bg or int(non_bg[0][1]) != int(path_color):
        return None

    try:
        seq_out, seq_stats = _solve_with_mode(g, sequential=True)
        indep_out, _ = _solve_with_mode(g, sequential=False)
    except Exception:
        return None

    if int(seq_stats["dependency_hits"]) < 1:
        return None
    if int(seq_stats["dependency_markers"]) < 1:
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
