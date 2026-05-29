# Variant of 195c6913
# Dimension: D11 — 初始方向确定规则 (path)
# Change: 初始方向按优先级R,D,L,U检查第一个相邻path邻居确定（对col 0标记总是向右）
#   → 初始方向由标记颜色编码

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import label_mask


# Design analysis
# a) Concrete behavior:
#    The grid still has a top-left row of 2x2 template blocks, a thick path band,
#    isolated markers on col 0, and one 2x2 end block. The visible change is that
#    marker color now carries launch direction. This variant uses two recurring
#    marker colors only: blue(1) means Down, red(2) means Right. Every generated
#    grid contains both colors, with the blue marker above the red marker. The
#    upper blue marker first descends inside the col-0 path channel, then turns
#    into the band at the fjord mouth; the lower red marker still starts rightward.
# b) Behavior trace:
#    In the original task, a marker at (r,0) with path at both (r,1) and (r+1,0)
#    always starts Right because R is checked before D. In this variant the same
#    geometry launches differently: a blue marker ignores the right neighbor and
#    starts Down, while a red marker still starts Right. This isolates D11 cleanly:
#    one marker behaves like the old rule, the other visibly differs on the same grid.
# c) Complexity calibration:
#    Only two mapped colors are used so the color→direction rule is inferable from
#    a few samples. Every sample contains both mapped colors, avoiding task-level
#    coverage gaps where a test marker color never appeared in training.
# d) Hidden assumptions from the original code:
#    The original `dir_order` scan assumed local geometry alone selected the launch
#    direction, and the generator exploited that by painting all markers with one
#    fixed template color. That is no longer valid: `main()` must not inspect R/D/L/U
#    to choose a start heading, and the generator must deliberately emit multiple
#    marker colors. Repeated template colors are also avoided so a marker's cycle
#    phase stays visually unambiguous.
# e) Placement legality:
#    Markers remain embedded in the col-0 path segment. The generator additionally
#    guarantees that the blue marker has both a right-path neighbor and a down-path
#    neighbor, so the new solver's Down launch differs from the old solver's Right
#    launch on the same input. The red marker also has a right-path neighbor.
# f) Visual observability / cleanliness:
#    Two markers, two colors, two start headings: one control, one changed case.
#    This is easier to read than introducing 3-4 direction colors at once. To keep
#    the blue downward start visible and safe, 2-marker samples always use the
#    left-edge fjord separator rather than an enclosed interior island.
# g) New solver assumptions guaranteed by the generator:
#    Only blue and red appear as isolated markers; both colors are present in the
#    template; the blue marker is placed above the fjord so its downward launch
#    produces a short vertical prefix before the alternating-turn logic takes over.

DIR_BY_MARKER_COLOR: dict[int, tuple[int, int]] = {
    1: (1, 0),  # blue -> Down
    2: (0, 1),  # red -> Right
}


def _assert_path_single_component(g: np.ndarray, path_color: int) -> None:
    """Eval invariant: path_color forms a single 4-connected component."""
    _, n_path = label_mask(g == path_color, connectivity=4)
    assert n_path == 1, f"path_color must be a single 4-connected component, got {n_path}"


def _assert_marker_embedded_in_path(
    g: np.ndarray, markers: list[tuple[int, int, int]], path_color: int
) -> None:
    """Eval invariant: each col-0 marker stays embedded inside a vertical path run."""
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


def _priority_initial_direction(
    g: np.ndarray, r: int, c: int, path_color: int
) -> tuple[int, int] | None:
    """Original-task reference rule: first path neighbor in R,D,L,U order."""
    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        rr = int(r + dr)
        cc = int(c + dc)
        if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1] and int(g[rr, cc]) == int(path_color):
            return int(dr), int(dc)
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate color-phased walkers whose launch direction is encoded by marker color.

    Input structure:
    - Top-left: 2x2 monochrome blocks define the repeating paint cycle.
    - Isolated cycle-colored pixels mark particle starts; their color also encodes heading.
    - A separate 2x2 monochrome block provides the collision/terminal color.

    Rule:
    - Each marker starts with the cycle phase named by its own color, then launches in the
      direction assigned to that color (here blue→Down, red→Right).
    - While moving on the path, the walker paints successive cycle colors.
    - On each collision with non-path, it paints that collision cell with the end color and
      alternates turn direction CCW, then CW, then CCW...; if the scheduled turn is blocked,
      the walker stops.

    Key insight:
    - The path geometry no longer decides the first step. Two markers can sit in the same
      kind of col-0 path pocket yet diverge immediately because their colors encode different
      initial directions.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    template_bg = int(g[0, 0])

    counts = Counter(map(int, g.flatten()))
    path_color = int(
        max((cnt, int(col)) for col, cnt in counts.items() if int(col) != int(template_bg))[1]
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

    out = g.copy()
    for r, c in pat_cells:
        out[int(r), int(c)] = int(template_bg)
    if end_pos is not None:
        er, ec = end_pos
        out[int(er) : int(er + 2), int(ec) : int(ec + 2)] = int(template_bg)

    per_marker_cells: list[set[tuple[int, int]]] = []
    for sr, sc, sv in markers:
        init = DIR_BY_MARKER_COLOR.get(int(sv))
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
    """Return a thick anti-diagonal path band with a col-0 lobe for the markers."""
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
    """Carve a left-edge background fjord between the two marker rows."""
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

    # Every sample contains both mapped marker colors so the direction code is observable.
    reserved_cycle = [1, 2]
    free_colors = [c for c in range(10) if c not in reserved_cycle]
    random.shuffle(free_colors)
    bg, path_color, end_color = free_colors[:3]
    extra_pool = free_colors[3:]

    n_blocks = random.randint(2, 4)
    extra_count = n_blocks - len(reserved_cycle)
    cycle_colors = reserved_cycle + random.sample(extra_pool, extra_count)
    random.shuffle(cycle_colors)

    r1 = random.randint(H // 3, H // 2)
    r2 = random.randint(max(r1 + 5, 2 * H // 3), H - 4)
    marker_rows = [r1, r2]

    mask = _build_band(H, W, lobe_rows=marker_rows)
    if mask[0, 0]:
        return None
    _, n_path = label_mask(mask, connectivity=4)
    if n_path != 1:
        return None
    if not mask[0, :].any() or not mask[:, W - 1].any():
        return None
    if not mask[H - 1, :].any() or not mask[:, 0].any():
        return None
    for r in marker_rows:
        if not (mask[r, 0] and mask[r - 1, 0] and mask[r + 1, 0]):
            return None
        if W < 2 or not mask[r, 1]:
            return None

    trial = mask.copy()
    if not _carve_fjord(trial, r1, r2):
        return None
    _, nc = label_mask(trial, connectivity=4)
    if nc != 1:
        return None
    if not trial[0, :].any() or not trial[:, W - 1].any():
        return None
    if not trial[H - 1, :].any() or not trial[:, 0].any():
        return None
    for r in marker_rows:
        if not (trial[r, 0] and trial[r - 1, 0] and trial[r + 1, 0]):
            return None
        if not trial[r, 1]:
            return None
    mask = trial

    _, n_bg = label_mask(~mask, connectivity=4)
    if n_bg < 3:
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

    g[r1, 0] = 1
    g[r2, 0] = 2

    _, n_path_after = label_mask(g == path_color, connectivity=4)
    if n_path_after != 1:
        return None
    if g[0, 0] != bg:
        return None

    cnt = Counter(map(int, g.flatten()))
    non_bg = sorted([(c, col) for col, c in cnt.items() if col != bg], reverse=True)
    if not non_bg or non_bg[0][1] != path_color:
        return None

    upper_old = _priority_initial_direction(g, r1, 0, path_color)
    lower_old = _priority_initial_direction(g, r2, 0, path_color)
    if upper_old != (0, 1) or lower_old != (0, 1):
        return None
    if DIR_BY_MARKER_COLOR[1] == upper_old or DIR_BY_MARKER_COLOR[2] != lower_old:
        return None

    for r, color in [(r1, 1), (r2, 2)]:
        dr, dc = DIR_BY_MARKER_COLOR[color]
        rr = r + dr
        cc = 0 + dc
        if not (0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(path_color)):
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
