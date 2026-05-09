# Combo variant of 195c6913 (index 21)
# combo_spec: D8/2+D9/0
# Name: 每标记独立子网格输出+3-4个标记

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import label_mask


class Scene:
    def __init__(
        self,
        bg: int,
        path_color: int,
        pattern: tuple[int, ...],
        pat_cells: tuple[tuple[int, int], ...],
        end_color: int,
        end_pos: tuple[int, int],
        markers: tuple[tuple[int, int, int], ...],
    ) -> None:
        self.bg = bg
        self.path_color = path_color
        self.pattern = pattern
        self.pat_cells = pat_cells
        self.end_color = end_color
        self.end_pos = end_pos
        self.markers = markers


class SimResult:
    def __init__(
        self,
        trail_cells: set[tuple[int, int]],
        painted: dict[tuple[int, int], int],
        corner_cells: set[tuple[int, int]],
        turns: int,
    ) -> None:
        self.trail_cells = trail_cells
        self.painted = painted
        self.corner_cells = corner_cells
        self.turns = turns


def _assert_path_single_component(g: np.ndarray, path_color: int) -> None:
    _, n_path = label_mask(g == path_color, connectivity=4)
    assert n_path == 1, f"path_color must be a single 4-connected component, got {n_path}"


def _assert_marker_embedded_in_path(
    g: np.ndarray, markers: list[tuple[int, int, int]], path_color: int
) -> None:
    h, _ = g.shape
    for r, c, _ in markers:
        assert c == 0, f"marker must stay on col 0, got ({r},{c})"
        if r > 0:
            assert int(g[r - 1, c]) == path_color
        if r < h - 1:
            assert int(g[r + 1, c]) == path_color


def _extract_scene(grid: np.ndarray) -> Scene:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = int(g[0, 0])

    counts = Counter(map(int, g.ravel()))
    non_bg = [(cnt, col) for col, cnt in counts.items() if int(col) != bg]
    assert non_bg, "grid must contain a non-bg path color"
    path_color = int(max(non_bg)[1])
    _assert_path_single_component(g, path_color)

    pattern: list[int] = []
    pat_cells: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for r in range(min(6, h - 1)):
        for c in range(min(25, w - 1)):
            if (r, c) in seen:
                continue
            v = int(g[r, c])
            if v in (bg, path_color):
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
    assert 2 <= len(pattern) <= 4, f"expected 2-4 template blocks, got {len(pattern)}"

    pat_set = set(pattern)
    end_color = None
    end_pos = None
    for r in range(h - 1):
        for c in range(w - 1):
            v = int(g[r, c])
            if v == bg or v == path_color or v in pat_set:
                continue
            if (
                int(g[r + 1, c]) == v
                and int(g[r, c + 1]) == v
                and int(g[r + 1, c + 1]) == v
            ):
                end_color = v
                end_pos = (r, c)
                break
        if end_pos is not None:
            break
    assert end_color is not None and end_pos is not None, "missing end block"

    markers: list[tuple[int, int, int]] = []
    for r in range(h):
        for c in range(w):
            v = int(g[r, c])
            if v not in pat_set:
                continue
            same = 0
            has_path = False
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    if int(g[rr, cc]) == v:
                        same += 1
                    if int(g[rr, cc]) == path_color:
                        has_path = True
            if same == 0 and has_path:
                markers.append((r, c, v))

    markers.sort()
    assert 3 <= len(markers) <= 4, f"expected 3-4 markers, got {len(markers)}"
    _assert_marker_embedded_in_path(g, markers, path_color)
    return Scene(
        bg=bg,
        path_color=path_color,
        pattern=tuple(pattern),
        pat_cells=tuple(pat_cells),
        end_color=int(end_color),
        end_pos=end_pos,
        markers=tuple(markers),
    )


def _initial_direction(g: np.ndarray, path_color: int, sr: int, sc: int) -> tuple[int, int] | None:
    for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        rr, cc = sr + dr, sc + dc
        if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1] and int(g[rr, cc]) == path_color:
            return dr, dc
    return None


def _simulate_marker(g: np.ndarray, scene: Scene, marker: tuple[int, int, int]) -> SimResult:
    sr, sc, sv = marker
    init = _initial_direction(g, scene.path_color, sr, sc)
    assert init is not None, f"marker at {(sr, sc)} is not adjacent to path"

    r, c = sr, sc
    dr, dc = init
    idx = int(scene.pattern.index(int(sv)))
    hits = 0
    visited: set[tuple[int, int, int, int, int, int]] = set()
    painted: dict[tuple[int, int], int] = {}
    trail_cells: set[tuple[int, int]] = set()
    corner_cells: set[tuple[int, int]] = set()
    h, w = g.shape

    for _ in range(h * w * 4):
        state = (r, c, dr, dc, idx, hits & 1)
        if state in visited:
            break
        visited.add(state)

        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            break

        if int(g[nr, nc]) == scene.path_color:
            idx = (idx + 1) % len(scene.pattern)
            painted[(nr, nc)] = int(scene.pattern[idx])
            trail_cells.add((nr, nc))
            r, c = nr, nc
            continue

        if int(g[nr, nc]) == scene.bg:
            corner_cells.add((nr, nc))

        left = (-dc, dr)
        right = (dc, -dr)
        tdr, tdc = left if (hits % 2 == 0) else right
        rr, cc = r + tdr, c + tdc
        if 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == scene.path_color:
            dr, dc = tdr, tdc
            hits += 1
            continue
        break

    return SimResult(
        trail_cells=trail_cells,
        painted=painted,
        corner_cells=corner_cells,
        turns=hits,
    )


def _render_base(g: np.ndarray, scene: Scene) -> np.ndarray:
    out = np.array(g, copy=True, dtype=int)
    for r, c in scene.pat_cells:
        out[r, c] = scene.bg
    er, ec = scene.end_pos
    out[er : er + 2, ec : ec + 2] = scene.bg
    for r, c, _ in scene.markers:
        out[r, c] = scene.path_color
    return out


def _render_marker_frame(
    g: np.ndarray,
    scene: Scene,
    marker: tuple[int, int, int],
    sim: SimResult,
) -> np.ndarray:
    frame = _render_base(g, scene)
    sr, sc, sv = marker
    frame[sr, sc] = sv
    for (r, c), color in sim.painted.items():
        frame[r, c] = color
    for r, c in sim.corner_cells:
        frame[r, c] = scene.end_color
    return frame


def _concat_frames(frames: list[np.ndarray], bg: int, gap: int = 2) -> np.ndarray:
    assert frames
    h = frames[0].shape[0]
    pieces: list[np.ndarray] = []
    for i, frame in enumerate(frames):
        assert frame.shape == frames[0].shape
        if i:
            pieces.append(np.full((h, gap), bg, dtype=int))
        pieces.append(frame)
    return np.hstack(pieces)


# Design analysis for the combo variant.
# a) Concrete behavior:
#    Original 195c6913 paints all marker trajectories back onto one shared board.
#    Here the input still contains one shared board, but with 3-4 col-0 markers.
#    Example: if markers are at rows 9, 15, 22, each marker is simulated on the
#    same path using the same alternating CCW/CW turn rule, yet the output becomes
#    frame(row 9) | gap | frame(row 15) | gap | frame(row 22). Inside each frame,
#    only that marker remains visible; the other marker cells are restored to the
#    underlying path color. This keeps the "independent simulation" observable.
# b) Complexity calibration:
#    The variant is visibly different from the source task because one input now
#    yields multiple sibling outputs and the marker count is always 3 or 4. Panel
#    order is only the visible top-to-bottom marker order; no color-id sorting is
#    used. The path keeps separators so trajectories remain distinct rather than
#    collapsing into the same painted region.
# c) Hidden assumptions audited:
#    The original solver assumed a single output canvas, 1-2 markers, and a marker
#    set whose union could be painted in place. The new main() removes the single-
#    canvas assumption by rendering one cleaned base per marker, then concatenating
#    those bases. Marker detection, template detection, end-block detection, and
#    strict turn scheduling are kept, but main() now asserts the generator supplied
#    3-4 markers and sorts them spatially before concatenation.
# d) Placement legality:
#    The generator now builds explicit non-overlapping staircase channels, one per
#    marker, then connects those channels with a shared vertical spine that sits on
#    cells no particle is forced to enter. That keeps the path 4-connected while the
#    traversed routes remain separate. Template blocks stay in the clear top-left
#    zone; the end block stays in isolated bottom-right background; markers overwrite
#    only embedded col-0 path cells. Oracle simulation rejects overlaps or trivial
#    paths.
# e) Visual observability:
#    Human solvers can see that all markers share the same kind of start pixel on
#    the left edge, then see the output duplicated into equally sized panels with
#    blank gutters. Each panel contains the same cleaned board geometry but a
#    different single trajectory, making the split rule inferable from 2-3 examples.
# f) Visual cleanliness:
#    By restoring non-selected markers to path_color inside each panel, each frame
#    contains one start pixel, one trail, and a few end-color corner marks only.
#    This avoids multi-trajectory overpainting inside the same region.
# g) Inferability of change:
#    Color cycling still comes only from the top-left 2x2 template row, while panel
#    order comes from marker row order. The learner does not need hidden numeric
#    properties of colors; the repeated board geometry plus one-trail-per-panel is
#    the structural clue.
def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate several left-edge markers independently and lay the answers out as side-by-side panels.

    Input structure:
    - Top-left: 2x2 monochrome blocks define the repeating color cycle from left to right.
    - Along column 0: exactly 3 or 4 isolated marker pixels, each embedded in the path and colored with one of the cycle colors.
    - Elsewhere: one 2x2 end block and one thick path-color band crossing the board diagonally.

    Rule:
    - Treat every marker as its own particle on the same input board, using the usual straight motion and strict alternating turn schedule (CCW, then CW, then CCW, ...).
    - For one marker at a time, erase the template blocks and end block, restore the other markers back to path color, then paint only that marker's visited path cells and collision corners.
    - Sort markers from top to bottom and concatenate those per-marker boards horizontally, leaving a bg gutter between neighboring boards.

    Key insight:
    - The path simulation itself is unchanged; the new challenge is to separate the shared input into several independent output panels, one trajectory per marker.
    """
    g = np.asarray(grid, dtype=int)
    scene = _extract_scene(g)

    sims = [_simulate_marker(g, scene, marker) for marker in scene.markers]
    for i in range(len(sims)):
        for j in range(i + 1, len(sims)):
            overlap = sims[i].trail_cells & sims[j].trail_cells
            assert not overlap, f"marker trajectories {i},{j} overlap on {len(overlap)} cells"

    frames = [
        _render_marker_frame(g, scene, marker, sim)
        for marker, sim in zip(scene.markers, sims)
    ]
    return _concat_frames(frames, scene.bg, gap=2)

def _sample_marker_rows(h: int, n_markers: int) -> list[int] | None:
    min_gap = 5
    start = h // 3
    stop = h - 4
    anchors = np.linspace(start, stop, n_markers)
    rows: list[int] = []
    prev = -99
    for i, anchor in enumerate(anchors):
        remaining = n_markers - i - 1
        lo = max(start, int(round(anchor)) - 2, prev + min_gap)
        hi = min(stop - remaining * min_gap, int(round(anchor)) + 2)
        if lo > hi:
            return None
        row = random.randint(lo, hi)
        rows.append(row)
        prev = row
    return rows


def _assign_marker_colors(cycle_colors: list[int], n_markers: int) -> list[int]:
    assert cycle_colors
    if n_markers <= len(cycle_colors):
        colors = random.sample(cycle_colors, n_markers)
    else:
        colors = list(cycle_colors)
        while len(colors) < n_markers:
            colors.append(random.choice(cycle_colors))
        random.shuffle(colors)
    assert len(set(colors)) >= 2, "marker colors must expose multiple cycle phases"
    return colors


def _build_route_mask(h: int, w: int, rows: list[int]) -> np.ndarray | None:
    n_markers = len(rows)
    top_rows: list[int] = []
    for i, row in enumerate(rows):
        lo = 6 if i == 0 else rows[i - 1] + 2
        hi = row - 3
        if lo > hi:
            return None
        top_rows.append(random.randint(lo, hi))

    x_spine = random.randint(w // 2, w - 8)
    anchors = np.linspace(min(x_spine - 3, w // 2), 5, n_markers)
    turn_cols: list[int] = []
    prev_hi = x_spine - 3
    for anchor in anchors:
        lo = max(4, int(round(anchor)) - 1)
        hi = min(prev_hi, int(round(anchor)) + 1)
        if lo > hi:
            lo = hi = max(4, prev_hi)
        if hi < 4:
            return None
        turn_col = random.randint(lo, hi)
        turn_cols.append(turn_col)
        prev_hi = turn_col - 1

    end_cols: list[int] = []
    for _ in rows:
        lo = x_spine + 2
        hi = w - 3
        if lo > hi:
            return None
        end_cols.append(random.randint(lo, hi))

    mask = np.zeros((h, w), dtype=bool)
    for row, top_row, turn_col, end_col in zip(rows, top_rows, turn_cols, end_cols):
        mask[row, : turn_col + 1] = True
        mask[top_row : row + 1, turn_col] = True
        mask[top_row, turn_col : end_col + 1] = True
        mask[max(0, row - 1) : min(h, row + 2), :2] = True

    mask[: top_rows[-1] + 1, x_spine] = True
    mask[max(0, top_rows[0] - 2), x_spine:] = True
    mask[rows[-1] - 1 :, 0] = True

    _, n_path = label_mask(mask, connectivity=4)
    if n_path != 1:
        return None
    return mask


def _find_end_candidates(g: np.ndarray, bg: int, path_color: int) -> list[tuple[int, int]]:
    h, w = g.shape
    end_candidates: list[tuple[int, int]] = []
    for r in range(h // 2, h - 1):
        for c in range(w // 2, w - 1):
            if not (g[r : r + 2, c : c + 2] == bg).all():
                continue
            r0, r1 = max(0, r - 1), min(h, r + 3)
            c0, c1 = max(0, c - 1), min(w, c + 3)
            if (g[r0:r1, c0:c1] == path_color).any():
                continue
            end_candidates.append((r, c))
    return end_candidates


def _oracle_metrics(g: np.ndarray) -> tuple[int, int, list[int]]:
    scene = _extract_scene(g)
    sims = [_simulate_marker(g, scene, marker) for marker in scene.markers]
    for i in range(len(sims)):
        for j in range(i + 1, len(sims)):
            assert not (sims[i].trail_cells & sims[j].trail_cells)
    trail_lengths = [len(sim.trail_cells) for sim in sims]
    total_turns = sum(sim.turns for sim in sims)
    return len(scene.markers), total_turns, trail_lengths


def _gen_attempt() -> np.ndarray | None:
    h = w = random.randint(24, 30)
    # Four-marker layouts are filtered out more often by the geometry checks, so
    # bias the proposal slightly toward 4 to keep accepted samples balanced.
    n_markers = 3 if random.random() < 0.24 else 4
    rows = _sample_marker_rows(h, n_markers)
    if rows is None:
        return None

    palette = list(range(10))
    random.shuffle(palette)
    bg = palette[0]
    path_color = palette[1]
    end_color = palette[2]
    n_blocks = random.randint(3, 4)
    cycle_colors = palette[3 : 3 + n_blocks]
    marker_colors = _assign_marker_colors(cycle_colors, n_markers)

    mask = _build_route_mask(h, w, rows)
    if mask is None:
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
    for r in rows:
        if not (mask[r, 0] and mask[r - 1, 0] and mask[r + 1, 0]):
            return None

    template_row = random.choice([1, 2])
    block_cols = [1, 4, 7, 10][:n_blocks]
    for bc in block_cols:
        for dr in range(2):
            for dc in range(2):
                rr, cc = template_row + dr, bc + dc
                if rr >= h or cc >= w or mask[rr, cc]:
                    return None

    g = np.full((h, w), bg, dtype=int)
    g[mask] = path_color
    for color, bc in zip(cycle_colors, block_cols):
        g[template_row : template_row + 2, bc : bc + 2] = color

    end_candidates = _find_end_candidates(g, bg, path_color)
    if not end_candidates:
        return None
    er, ec = random.choice(end_candidates)
    g[er : er + 2, ec : ec + 2] = end_color

    for r, color in zip(rows, marker_colors):
        g[r, 0] = color

    _, n_path_final = label_mask(g == path_color, connectivity=4)
    if n_path_final != 1:
        return None
    if g[0, 0] != bg:
        return None

    cnt = Counter(map(int, g.ravel()))
    non_bg = sorted([(c, col) for col, c in cnt.items() if col != bg], reverse=True)
    if not non_bg or non_bg[0][1] != path_color:
        return None

    try:
        marker_count, total_turns, trail_lengths = _oracle_metrics(g)
        out = main(g)
    except Exception:
        return None

    expected_w = marker_count * w + (marker_count - 1) * 2
    if out.shape != (h, expected_w):
        return None
    if min(trail_lengths) < 8:
        return None
    if total_turns < marker_count:
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(500):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 500 attempts")
