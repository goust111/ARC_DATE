# Combo variant of 1818057f (index 3)
# combo_spec: D1/1+D5/0
# Name: 密集重叠十字旋转为X

from __future__ import annotations

import numpy as np

from arc_utils import get_bg


# Design analysis for the combo variant
# a) Concrete behavior and trace
#    - Original 1818057f detects every foreground '+' center and recolors the same
#      5 cells. D1/1 changes the write step: each detected '+' rotates into an 'X'.
#    - D5/0 removes the old spacing rule, so two intended centers may now be adjacent.
#      Example with centers at (r,c) and (r,c+1):
#        the right arm of the left '+' is also the center of the right '+'.
#      Under the combo rule, both centers stay yellow, shared arm cells are cleared,
#      and the diagonals around both centers are painted teal. This produces a dense
#      lattice of X-like marks rather than independent isolated rotations.
#    - The crucial difference from the original task is therefore twofold:
#      detect on the unchanged input, then render all rotations simultaneously.
#
# b) Complexity calibration
#    - Full unrestricted overlap would quickly degenerate into solid blobs where a
#      human could not tell which '+' caused which output. The generator therefore
#      uses self-avoiding chains of adjacent centers: overlap is guaranteed, but the
#      center graph stays sparse enough to read.
#    - Extra foreground noise is still added, so the solver must identify local '+'
#      centers inside mixed components rather than operate on pre-cut tokens.
#
# c) Implicit-assumption review
#    - The original solver mutates `out` locally per center; that was safe when each
#      motif wrote only onto its own cells. It is no longer safe because one cell can
#      simultaneously be a center of one '+' and an arm of another.
#    - The old generator reserved non-overlapping 3x3 neighborhoods. That assumption
#      must be removed, otherwise D5/0 would never be visible.
#    - The new solver therefore first collects centers from the input, then applies a
#      priority merge: center survives as fg, any diagonal landing site becomes teal,
#      and only remaining arm cells are cleared to background.
#
# d) Placement legality
#    - Centers stay at least 2 cells from the border so the input '+' and output 'X'
#      both fit, and short extensions can still be added.
#    - Overlap is allowed only inside a chain cluster; separate clusters keep a gap so
#      unrelated structures do not merge into unreadable masses.
#    - Every later-added distractor shape is accepted only if the detected center set
#      remains exactly the intended one.
#
# e) Visual observability
#    - A chain of adjacent '+' centers makes the overlap rule obvious to a human:
#      neighboring centers remain yellow while teal diagonals appear on both sides,
#      which cannot be explained by simple recoloring.
#    - Because at least one adjacent-center pair is forced in every sample, the combo
#      behavior is visible across just a few examples.
#
# f) Visual cleanliness
#    - The generator prefers 1- or 2-cluster layouts and short attachments, which keeps
#      the dense interaction local instead of turning the whole board into one blob.
#    - Distractors are placed away from overlap clusters when possible and are rejected
#      if they introduce new accidental centers.
#
# g) Color-change inference
#    - The only added color is teal. Yellow still marks original centers, while red
#      reappears where rotated arms are cleared. This structure is enough for a human
#      to infer both the geometric move and the priority of overlapping writes.
#
# h) Quantitative anti-degeneration target
#    - Each generated input must contain at least one orthogonally adjacent pair of
#      centers. This ensures D5/0 is active instead of silently falling back to the
#      separated-center regime.
#
# i) New solver assumptions to honor in generation
#    - `main()` assumes every relevant transformation can be derived solely from the
#      set of detected centers in the original grid.
#    - `generate_input()` enforces that exact center set by validating after every
#      extension and distractor placement.


BG = 2
FG = 4
ROT = 8
ORTHO = [(1, 0), (-1, 0), (0, 1), (0, -1)]
DIAG = [(1, 1), (1, -1), (-1, 1), (-1, -1)]


def _is_plus_center(grid: np.ndarray, r: int, c: int, fg: int) -> bool:
    if int(grid[r, c]) != int(fg):
        return False
    return all(int(grid[r + dr, c + dc]) == int(fg) for dr, dc in ORTHO)


def _find_plus_centers(grid: np.ndarray, fg: int) -> list[tuple[int, int]]:
    H, W = grid.shape
    centers: list[tuple[int, int]] = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if _is_plus_center(grid, r, c, fg):
                centers.append((r, c))
    return centers


def _count_adjacent_center_pairs(centers: list[tuple[int, int]]) -> int:
    center_set = set(centers)
    pairs = 0
    for r, c in centers:
        for dr, dc in [(1, 0), (0, 1)]:
            if (r + dr, c + dc) in center_set:
                pairs += 1
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect overlapping foreground '+' centers and rotate each one into a merged 'X' pattern.

    Input structure:
    - bg is the most common color.
    - Exactly one non-bg foreground color forms mixed components on the bg, including
      multiple '+' centers that may now overlap or sit immediately next to each other.

    Rule:
    - A '+' center is any foreground pixel whose 4 orthogonal neighbors are also foreground.
    - For every detected center, keep the center foreground, move its 4 arms to the 4
      diagonal positions as teal (8), and restore its orthogonal arm cells to bg.
    - When multiple '+' motifs overlap, combine all rotations simultaneously with priority
      center > teal diagonal landing site > cleared arm cell.

    Key insight: Overlap means a single pixel can play different roles for different '+'
    motifs, so the solver must detect all centers from the original grid first and only
    then merge their effects with a fixed write priority.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    colors = sorted(set(map(int, g.flatten())) - {bg})
    if not colors:
        return out
    fg = int(colors[0])

    centers = _find_plus_centers(g, fg)
    if not centers:
        return out

    center_set = set(centers)
    arm_cells: set[tuple[int, int]] = set()
    diag_cells: set[tuple[int, int]] = set()
    for r, c in centers:
        for dr, dc in ORTHO:
            arm_cells.add((r + dr, c + dc))
        for dr, dc in DIAG:
            diag_cells.add((r + dr, c + dc))

    for r, c in arm_cells:
        if (r, c) not in center_set:
            out[r, c] = bg
    for r, c in diag_cells:
        if (r, c) not in center_set:
            out[r, c] = ROT
    for r, c in center_set:
        out[r, c] = fg
    return out


def _paint_plus_union(grid: np.ndarray, centers: list[tuple[int, int]]) -> None:
    for r, c in centers:
        grid[r, c] = FG
        for dr, dc in ORTHO:
            grid[r + dr, c + dc] = FG


def _interior_center_cells(H: int, W: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(2, H - 2) for c in range(2, W - 2)]


def _far_from_existing(r: int, c: int, occupied: set[tuple[int, int]]) -> bool:
    return all(max(abs(r - rr), abs(c - cc)) >= 3 for rr, cc in occupied)


def _make_center_chain(
    rng: np.random.Generator,
    H: int,
    W: int,
    occupied: set[tuple[int, int]],
    length: int,
) -> list[tuple[int, int]] | None:
    starts = _interior_center_cells(H, W)
    rng.shuffle(starts)

    for start in starts:
        if not _far_from_existing(start[0], start[1], occupied):
            continue

        path = [start]
        path_set = {start}
        while len(path) < length:
            options: list[tuple[int, tuple[int, int]]] = []
            for end_idx, (er, ec) in ((0, path[0]), (-1, path[-1])):
                for dr, dc in ORTHO:
                    nr, nc = er + dr, ec + dc
                    cand = (nr, nc)
                    if nr < 2 or nr >= H - 2 or nc < 2 or nc >= W - 2:
                        continue
                    if cand in occupied or cand in path_set:
                        continue
                    adj = sum(abs(nr - pr) + abs(nc - pc) == 1 for pr, pc in path)
                    if adj != 1:
                        continue
                    options.append((end_idx, cand))
            if not options:
                break

            end_idx, cand = options[int(rng.integers(len(options)))]
            if end_idx == 0:
                path.insert(0, cand)
            else:
                path.append(cand)
            path_set.add(cand)

        if len(path) == length:
            return path
    return None


def _valid_added_cells(grid: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    H, W = grid.shape
    seen: set[tuple[int, int]] = set()
    for r, c in cells:
        if (r, c) in seen:
            continue
        seen.add((r, c))
        if r < 0 or r >= H or c < 0 or c >= W:
            return False
        if int(grid[r, c]) != BG:
            return False
    return True


def _apply_if_center_set_preserved(
    grid: np.ndarray,
    cells: list[tuple[int, int]],
    intended_centers: set[tuple[int, int]],
) -> int:
    if not cells or not _valid_added_cells(grid, cells):
        return 0
    temp = grid.copy()
    unique_cells = list(dict.fromkeys(cells))
    for r, c in unique_cells:
        temp[r, c] = FG
    if set(_find_plus_centers(temp, FG)) != intended_centers:
        return 0
    for r, c in unique_cells:
        grid[r, c] = FG
    return len(unique_cells)


def _attach_extensions(
    rng: np.random.Generator,
    grid: np.ndarray,
    clusters: list[list[tuple[int, int]]],
    intended_centers: set[tuple[int, int]],
) -> bool:
    for cluster in clusters:
        anchors = cluster.copy()
        rng.shuffle(anchors)
        placed = False
        for r, c in anchors:
            dirs = ORTHO.copy()
            rng.shuffle(dirs)
            for dr, dc in dirs:
                trail: list[tuple[int, int]] = []
                for dist in range(2, 5):
                    nr, nc = r + dr * dist, c + dc * dist
                    if nr < 0 or nr >= grid.shape[0] or nc < 0 or nc >= grid.shape[1]:
                        break
                    if int(grid[nr, nc]) != BG:
                        break
                    trail.append((nr, nc))
                if not trail:
                    continue
                take = int(rng.integers(1, min(3, len(trail)) + 1))
                if _apply_if_center_set_preserved(grid, trail[:take], intended_centers):
                    placed = True
                    break
            if placed:
                break
        if not placed:
            return False
    return True


def _grow_distractors(
    rng: np.random.Generator,
    grid: np.ndarray,
    target_fg: int,
    intended_centers: set[tuple[int, int]],
) -> None:
    H, W = grid.shape
    current_fg = int(np.sum(grid != BG))
    attempts = 0
    while current_fg < target_fg and attempts < H * W * 30:
        attempts += 1
        r = int(rng.integers(0, H))
        c = int(rng.integers(0, W))
        mode = int(rng.integers(0, 4))
        if mode == 0:
            shape = [(r, c)]
        else:
            dr, dc = ORTHO[int(rng.integers(0, 4))]
            length = int(rng.integers(2, 5))
            shape = [(r + dr * step, c + dc * step) for step in range(length)]
        added = _apply_if_center_set_preserved(grid, shape, intended_centers)
        current_fg += added


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    H = int(rng.integers(10, 23))
    W = int(rng.integers(10, 23))
    grid = np.full((H, W), BG, dtype=int)

    max_k = min(9, max(2, (H * W) // 24))
    total_centers = int(rng.integers(2, max_k + 1))

    occupied_centers: set[tuple[int, int]] = set()
    clusters: list[list[tuple[int, int]]] = []

    first_len = int(rng.integers(2, min(5, total_centers) + 1))
    first_cluster = _make_center_chain(rng, H, W, occupied_centers, first_len)
    if first_cluster is None:
        return None
    clusters.append(first_cluster)
    occupied_centers.update(first_cluster)

    remaining = total_centers - first_len
    while remaining > 0:
        if remaining >= 2 and rng.random() < 0.35:
            length = int(rng.integers(2, min(4, remaining) + 1))
        else:
            length = 1

        if length == 1:
            starts = _interior_center_cells(H, W)
            rng.shuffle(starts)
            placed = None
            for r, c in starts:
                if _far_from_existing(r, c, occupied_centers):
                    placed = [(r, c)]
                    break
        else:
            placed = _make_center_chain(rng, H, W, occupied_centers, length)

        if placed is None:
            return None

        clusters.append(placed)
        occupied_centers.update(placed)
        remaining -= len(placed)

    centers = [cell for cluster in clusters for cell in cluster]
    intended_centers = set(centers)
    if _count_adjacent_center_pairs(centers) < 1:
        return None

    _paint_plus_union(grid, centers)
    if set(_find_plus_centers(grid, FG)) != intended_centers:
        return None

    if not _attach_extensions(rng, grid, clusters, intended_centers):
        return None

    total = H * W
    min_fg = int(np.ceil(total * 0.26))
    max_fg = int(np.floor(total * 0.41))
    target_fg = int(rng.integers(min_fg, max_fg + 1))
    _grow_distractors(rng, grid, target_fg, intended_centers)

    fg_count = int(np.sum(grid != BG))
    if fg_count < min_fg or fg_count > max_fg:
        return None
    if fg_count >= total // 2:
        return None

    found = set(_find_plus_centers(grid, FG))
    if found != intended_centers:
        return None

    out = main(grid)
    if not np.any(out == ROT):
        return None
    if _count_adjacent_center_pairs(sorted(found)) < 1:
        return None

    # Anti-degeneration: the combo must not collapse to the original recolor output.
    orig_like = grid.copy()
    for r, c in found:
        for dr, dc in [(0, 0), *ORTHO]:
            orig_like[r + dr, c + dc] = ROT
    if np.array_equal(out, orig_like):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(600):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 600 attempts")
