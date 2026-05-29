# Variant of 195c6913
# Dimension: D9 — 标记数量与基数关系 (meta)
# Change: 1或2个标记（n_markers==2出现率>=50%），各标记独立模拟，轨迹必须不相交。... → 单标记发射多轮（碰撞后重新出发）

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import label_mask


# Design analysis for the D9 variant
# a) Concrete behavior + trace
#    - The input still contains three visible object families:
#      1) a top-left row of 2x2 monochrome template blocks that define the cycle,
#      2) one isolated marker pixel on col 0 colored with template[0],
#      3) one isolated 2x2 end-color block in a background pocket.
#    - The core change is not "another marker somewhere else", but "the same
#      marker launches again after it gets stuck". To make this visible, the
#      generator builds one connected path object with several fresh branches
#      attached in sequence. A launch paints one branch, consumes it, and stops
#      at a branch tip. The next launch starts from that tip and enters a
#      different unused branch according to the cyclic initial-direction order
#      R,D,L,U.
#    - Small trace:
#      original: marker M at col 0 launches Right, turns by CCW/CW alternation,
#      then stops forever when the scheduled turn is blocked.
#      variant: M launches Right and stops at P1. From P1, the path already
#      used by round 1 is no longer traversable, so the next usable launch
#      direction is Down. That second round paints a new branch and stops at P2.
#      Then the same logic continues with Left, then Up, until no adjacent unused
#      path cell remains in any of the four launch directions.
#    - Therefore the pipeline change sits inside the solver's state update:
#      the path must become a consumable resource, not a static backdrop reused
#      independently by several markers.
# b) Complexity calibration
#    - A trivial version would make each round a straight ray; that would remove
#      the original alternating-turn mechanism. This variant keeps 3-segment
#      corridor rounds, so each launch still includes multiple collision-driven
#      turns before it stops.
#    - A too-broad band would make relaunch directions ambiguous after a stop.
#      To keep the new rule human-readable, the generator narrows the path into
#      a clean corridor network with one fresh outgoing branch at each stop.
# c) Implicit assumptions reviewed against the original main()
#    - Original code iterates over all detected markers and assumes each marker
#      can read the untouched input path. That is no longer valid; repeated
#      launches from the same source must see already painted trail cells as
#      unavailable, otherwise the particle can simply relaunch backward and the
#      D9 change becomes invisible.
#    - Original disjointness checking compares different markers' visited sets.
#      Here that concept moves inside one source: each round must consume path
#      cells so later rounds are automatically disjoint except at the restart cell.
#    - Original initial-direction logic is a one-shot choice. Here we promote it
#      to a cyclic preference cursor: after a successful R launch, the next
#      search starts from D and can skip blocked directions until some unused
#      path is found or all four are blocked.
# d) Placement legality
#    - The generator keeps the original object vocabulary and non-overlap rules:
#      template blocks stay in the top-left background zone, the path is one
#      4-connected component, the end block is a 2x2 background block in the
#      bottom-right region with a one-cell buffer from the path, and the marker
#      sits on col 0 with path-colored vertical neighbors.
#    - A 3x2 start pad is used so overwriting the central marker pixel does not
#      disconnect the path component.
# e) Visual observability
#    - Humans can see the variant because there is only one visible marker in the
#      input, yet the output contains several separated colored runs with multiple
#      end-color collision dots. Across examples the launch order is spatially
#      consistent: first a right-going run, then a lower run from its tip, then
#      a left-going run, then an upper run.
# f) Visual cleanliness
#    - Branches only meet at restart tips, so the output does not devolve into a
#      dense overpainted blob. Template, marker, path, and end block remain easy
#      to distinguish in both input and output.
# g) Color inference
#    - Colors still follow the original visible cue: the ordered 2x2 template
#      blocks define the repeating cycle, and every launch restarts from the same
#      marker color. No hidden ordering uses color ids.
# New main() assumptions that the generator guarantees
#    - Exactly one isolated marker is present.
#    - The path contains at least one usable launch and typically 4 launches.
#    - The planned relaunch tips have at most one fresh outgoing branch in the
#      intended cyclic direction order, so the multi-round behavior is stable.


DIR_ORDER = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # R, D, L, U


def _assert_path_single_component(g: np.ndarray, path_color: int) -> None:
    _, n_path = label_mask(g == path_color, connectivity=4)
    assert n_path == 1, f"path_color must be a single 4-connected component, got {n_path}"


def _assert_marker_embedded_in_path(
    g: np.ndarray, markers: list[tuple[int, int, int]], path_color: int
) -> None:
    H, _ = g.shape
    for r, c, _ in markers:
        assert c == 0, f"marker must be on col 0, got ({r},{c})"
        if r > 0:
            assert int(g[r - 1, c]) == path_color, (
                f"marker at ({r},{c}) up-neighbor must be path_color {path_color}"
            )
        if r < H - 1:
            assert int(g[r + 1, c]) == path_color, (
                f"marker at ({r},{c}) down-neighbor must be path_color {path_color}"
            )


def _extract_scene(grid: np.ndarray) -> dict[str, object]:
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

    pat_set = set(map(int, pattern))
    end_color = None
    end_pos = None
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
            for dr, dc in DIR_ORDER:
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

    return {
        "grid": g,
        "H": H,
        "W": W,
        "template_bg": template_bg,
        "path_color": path_color,
        "pattern": pattern,
        "pat_cells": pat_cells,
        "end_color": end_color,
        "end_pos": end_pos,
        "markers": markers,
    }


def _simulate_single_source(
    scene: dict[str, object],
    out: np.ndarray,
) -> tuple[np.ndarray, int, list[set[tuple[int, int]]]]:
    g = scene["grid"]
    H = int(scene["H"])
    W = int(scene["W"])
    template_bg = int(scene["template_bg"])
    path_color = int(scene["path_color"])
    pattern = list(scene["pattern"])
    end_color = scene["end_color"]
    markers = list(scene["markers"])

    assert len(markers) == 1, f"variant requires exactly one marker, got {len(markers)}"
    if not pattern:
        return out, 0, []

    avail = g == path_color
    sr, sc, sv = markers[0]
    current = (int(sr), int(sc))
    base_idx = int(pattern.index(int(sv))) if int(sv) in pattern else 0
    next_pref = 0
    rounds = 0
    per_round_cells: list[set[tuple[int, int]]] = []

    for _ in range(H * W):
        launch = None
        for off in range(4):
            k = int((next_pref + off) % 4)
            dr, dc = DIR_ORDER[k]
            nr = int(current[0] + dr)
            nc = int(current[1] + dc)
            if 0 <= nr < H and 0 <= nc < W and bool(avail[nr, nc]):
                launch = (k, int(dr), int(dc))
                break
        if launch is None:
            break

        launch_k, dr, dc = launch
        next_pref = int((launch_k + 1) % 4)
        r, c = current
        idx = int(base_idx)
        hits = 0
        my_cells: set[tuple[int, int]] = set()

        for _step in range(H * W * 4):
            nr = int(r + dr)
            nc = int(c + dc)
            if not (0 <= nr < H and 0 <= nc < W):
                current = (int(r), int(c))
                per_round_cells.append(my_cells)
                rounds += 1
                break

            if bool(avail[nr, nc]):
                avail[nr, nc] = False
                idx = int((idx + 1) % len(pattern))
                out[nr, nc] = int(pattern[idx])
                my_cells.add((int(nr), int(nc)))
                r, c = int(nr), int(nc)
                continue

            if int(g[nr, nc]) == int(template_bg) and end_color is not None:
                out[nr, nc] = int(end_color)

            left = (-int(dc), int(dr))
            right = (int(dc), -int(dr))
            tdr, tdc = left if (int(hits) % 2 == 0) else right
            rr = int(r + tdr)
            cc = int(c + tdc)
            if 0 <= rr < H and 0 <= cc < W and bool(avail[rr, cc]):
                dr, dc = int(tdr), int(tdc)
                hits += 1
                continue

            current = (int(r), int(c))
            per_round_cells.append(my_cells)
            rounds += 1
            break
        else:
            current = (int(r), int(c))
            per_round_cells.append(my_cells)
            rounds += 1

    return out, rounds, per_round_cells


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    scene = _extract_scene(grid)
    out = np.array(grid, dtype=int).copy()
    for r, c in scene["pat_cells"]:
        out[int(r), int(c)] = int(scene["template_bg"])
    if scene["end_pos"] is not None:
        er, ec = scene["end_pos"]
        out[int(er) : int(er + 2), int(ec) : int(ec + 2)] = int(scene["template_bg"])
    _, rounds, per_round = _simulate_single_source(scene, out)
    return {
        "n_markers": len(scene["markers"]),
        "rounds": int(rounds),
        "used_cells": int(sum(len(s) for s in per_round)),
        "nonempty_rounds": int(sum(1 for s in per_round if s)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repeatedly relaunch one particle from the same source until no fresh exit remains.

    Input structure:
    - Top-left 2x2 monochrome blocks encode the repeating paint cycle.
    - Exactly one isolated cycle-colored marker sits on col 0 inside the path.
    - A distinct 2x2 monochrome block provides the collision/terminal color.

    Rule:
    - Remove the template blocks and the 2x2 terminal block in the output.
    - Launch from the lone marker with initial direction preference R,D,L,U.
    - During one launch, move straight on unused path cells, paint by advancing
      through the cycle, and turn strictly CCW,CW,CCW,CW... on collisions.
    - When a launch cannot take its scheduled turn, stop at that cell, then
      relaunch from there using the next cyclic initial direction whose adjacent
      cell is still unused path. Stop only when all four directions are blocked.

    Key insight:
    - The particle is not duplicated; the already painted trail becomes state.
      Consuming visited path cells turns a multi-marker separation problem into
      a single-source branch-exploration problem.
    """
    scene = _extract_scene(grid)
    assert len(scene["markers"]) == 1, "variant expects a single visible marker"

    out = np.array(scene["grid"], dtype=int).copy()
    for r, c in scene["pat_cells"]:
        out[int(r), int(c)] = int(scene["template_bg"])
    if scene["end_pos"] is not None:
        er, ec = scene["end_pos"]
        out[int(er) : int(er + 2), int(ec) : int(ec + 2)] = int(scene["template_bg"])

    out, _rounds, _cells = _simulate_single_source(scene, out)
    return out


def _add_segment(cells: set[tuple[int, int]], a: tuple[int, int], b: tuple[int, int]) -> None:
    r1, c1 = a
    r2, c2 = b
    if r1 == r2:
        lo, hi = sorted((c1, c2))
        for c in range(lo, hi + 1):
            cells.add((int(r1), int(c)))
    else:
        assert c1 == c2
        lo, hi = sorted((r1, r2))
        for r in range(lo, hi + 1):
            cells.add((int(r), int(c1)))


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    _add_segment(cells, a, b)
    return cells


def _build_path_cells(H: int, W: int, template_row: int) -> set[tuple[int, int]] | None:
    for _ in range(80):
        r_top = int(template_row + 3 + random.randint(0, 1))
        rs = int(H - random.randint(8, 10))
        r2 = int(H // 2 + random.randint(0, 2))
        r3 = int(H - random.randint(5, 6))
        r4 = int(H - random.randint(2, 3))
        r5 = int(H // 2 - random.randint(1, 2))
        r6 = int(r_top + random.randint(0, 2))

        c1 = int(random.randint(3, max(3, W // 4)))
        c2_lo = max(c1 + 4, W // 2 - 1)
        c2_hi = min(W - 9, W // 2 + 2)
        if c2_lo > c2_hi:
            continue
        c2 = int(random.randint(c2_lo, c2_hi))
        c3_lo = max(c2 + 4, W - 6)
        c3_hi = W - 4
        if c3_lo > c3_hi:
            continue
        c3 = int(random.randint(c3_lo, c3_hi))
        c4_lo = max(c2 + 1, W // 2)
        c4_hi = c3 - 3
        if c4_lo > c4_hi:
            continue
        c4 = int(random.randint(c4_lo, c4_hi))
        c5_lo = max(4, c4 - 6)
        c5_hi = c4 - 2
        if c5_lo > c5_hi:
            continue
        c5 = int(random.randint(c5_lo, c5_hi))
        c6_lo = max(2, c5 - 4)
        c6_hi = c5 - 2
        if c6_lo > c6_hi:
            continue
        c6 = int(random.randint(c6_lo, c6_hi))

        if not (0 < r_top < r6 < r5 < r3 < r4 < H):
            continue
        if not (0 < c1 < c2 < c3 < W and 0 < c6 < c5 < c4 < c3):
            continue
        if rs <= r_top + 3 or rs >= H - 1:
            continue

        cells: set[tuple[int, int]] = set()
        for rr in range(rs - 1, rs + 2):
            for cc in range(0, 2):
                cells.add((int(rr), int(cc)))

        path_points = [
            ((rs, 1), (rs, c1)),
            ((rs, c1), (r_top, c1)),
            ((r_top, c1), (r_top, c2)),
            ((r_top, c2), (r2, c2)),
            ((r2, c2), (r2, c3)),
            ((r2, c3), (r3, c3)),
            ((r3, c3), (r3, c4)),
            ((r3, c4), (r4, c4)),
            ((r4, c4), (r4, c5)),
            ((r4, c5), (r5, c5)),
            ((r5, c5), (r5, c6)),
            ((r5, c6), (r6, c6)),
        ]
        seg_cells = [_segment_cells(a, b) for a, b in path_points]
        bad_overlap = False
        for i in range(len(seg_cells)):
            for j in range(i + 1, len(seg_cells)):
                overlap = seg_cells[i] & seg_cells[j]
                allowed: set[tuple[int, int]] = set()
                if j == i + 1:
                    allowed = {path_points[i][1]}
                if overlap != allowed:
                    bad_overlap = True
                    break
            if bad_overlap:
                break
        if bad_overlap:
            continue

        for seg in seg_cells:
            cells |= seg

        if any(not (0 <= r < H and 0 <= c < W) for r, c in cells):
            continue
        if any(r <= template_row + 1 and c <= 11 for r, c in cells):
            continue
        return cells
    return None


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
            candidates.append((int(r), int(c)))
    if not candidates:
        return None
    return random.choice(candidates)


def _gen_attempt() -> np.ndarray | None:
    H = W = random.randint(20, 30)
    palette = list(range(10))
    random.shuffle(palette)
    bg = int(palette[0])
    path_color = int(palette[1])
    end_color = int(palette[2])
    n_blocks = random.randint(2, 4)
    cycle_pool = palette[3:]
    cycle_colors = [int(random.choice(cycle_pool)) for _ in range(n_blocks)]
    template_row = random.choice([1, 2])
    block_cols = [1, 4, 7, 10][:n_blocks]
    if block_cols[-1] + 1 >= W:
        return None

    cells = _build_path_cells(H, W, template_row)
    if cells is None:
        return None

    g = np.full((H, W), bg, dtype=int)
    for r, c in cells:
        g[int(r), int(c)] = int(path_color)

    for i, bc in enumerate(block_cols):
        if (g[template_row : template_row + 2, bc : bc + 2] == path_color).any():
            return None
        g[template_row : template_row + 2, bc : bc + 2] = int(cycle_colors[i])

    end_pos = _place_end_block(g, bg, path_color)
    if end_pos is None:
        return None
    er, ec = end_pos
    g[int(er) : int(er + 2), int(ec) : int(ec + 2)] = int(end_color)

    marker_rows = [r for r, c in cells if c == 0]
    if not marker_rows:
        return None
    sr = int(sorted(marker_rows)[len(marker_rows) // 2])
    if sr <= 0 or sr >= H - 1:
        return None
    if not (
        int(g[sr - 1, 0]) == int(path_color)
        and int(g[sr + 1, 0]) == int(path_color)
        and int(g[sr, 1]) == int(path_color)
    ):
        return None
    g[int(sr), 0] = int(cycle_colors[0])

    _, n_path_after = label_mask(g == path_color, connectivity=4)
    if n_path_after != 1:
        return None
    if int(g[0, 0]) != int(bg):
        return None

    cnt = Counter(map(int, g.flatten()))
    non_bg = sorted([(c, col) for col, c in cnt.items() if col != bg], reverse=True)
    if not non_bg or int(non_bg[0][1]) != int(path_color):
        return None

    try:
        out = main(g)
        metrics = _variant_metrics(g)
    except Exception:
        return None

    if metrics["n_markers"] != 1:
        return None
    if metrics["rounds"] < 3 or metrics["nonempty_rounds"] < 3:
        return None
    if np.array_equal(out, g):
        return None

    return g


def generate_input() -> np.ndarray:
    for _ in range(300):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 300 attempts")
