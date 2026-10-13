from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
FILL = 2
GREY = 5

DIR_BY_COLOR = {
    3: (0, -1),   # green -> left
    4: (1, 0),    # yellow -> down
    6: (-1, 0),   # magenta -> up
    8: (0, 1),    # teal -> right
}


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def boundary_from_region(region: np.ndarray) -> np.ndarray:
    h, w = region.shape
    out = np.zeros_like(region, dtype=bool)
    cells = np.argwhere(region)
    for r0, c0 in cells:
        r = int(r0)
        c = int(c0)
        for nr, nc in neighbors4(r, c):
            if not in_bounds(region.shape, nr, nc) or not bool(region[nr, nc]):
                out[r, c] = True
                break
    return out


def connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r0, c0 in np.argwhere(mask):
        r = int(r0)
        c = int(c0)
        if seen[r, c]:
            continue
        seen[r, c] = True
        q = deque([(r, c)])
        comp: list[tuple[int, int]] = []
        while q:
            rr, cc = q.popleft()
            comp.append((rr, cc))
            for nr, nc in neighbors4(rr, cc):
                if in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def outline_runs(row_mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    c = 0
    w = row_mask.shape[0]
    while c < w:
        if not bool(row_mask[c]):
            c += 1
            continue
        start = c
        while c + 1 < w and bool(row_mask[c + 1]):
            c += 1
        runs.append((start, c))
        c += 1
    return runs


def row_scan_fill(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    outline = out != BG
    h, w = out.shape
    for r in range(h):
        runs = outline_runs(outline[r])
        for i in range(0, len(runs) - 1, 2):
            left = runs[i][1]
            right = runs[i + 1][0]
            if right - left <= 1:
                continue
            span = out[r, left + 1:right]
            span[span == BG] = FILL
    return out


def exterior_background(grid: np.ndarray) -> np.ndarray:
    outline = grid != BG
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if not bool(outline[r, c]) and not bool(seen[r, c]):
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not bool(outline[r, c]) and not bool(seen[r, c]):
                seen[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if in_bounds(grid.shape, nr, nc) and not bool(outline[nr, nc]) and not bool(seen[nr, nc]):
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def slide_plugs(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    grey = g == GREY
    plugs = [(int(r), int(c), int(g[r, c])) for r, c in np.argwhere((g != BG) & (g != GREY))]
    moved: list[dict] = []
    for r, c, color in plugs:
        assert color in DIR_BY_COLOR, f"unsupported plug color {color}"
        out[r, c] = BG
    used_targets: set[tuple[int, int]] = set()
    for r, c, color in plugs:
        dr, dc = DIR_BY_COLOR[color]
        rr, cc = r, c
        while True:
            nr, nc = rr + dr, cc + dc
            if not in_bounds(g.shape, nr, nc) or bool(grey[nr, nc]):
                break
            rr, cc = nr, nc
        target = (rr, cc)
        assert target not in used_targets, "two plugs landed on the same cell"
        used_targets.add(target)
        out[rr, cc] = color
        moved.append(
            {
                "start": (r, c),
                "target": target,
                "color": color,
                "dir": (dr, dc),
                "distance": abs(rr - r) + abs(cc - c),
            }
        )
    return out, moved


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide the color-coded plug pixels to close the open outline, then fill the enclosed interior by row scan.

    Input structure:
    - The background is black.
    - Gray pixels form a nearly closed one-pixel outline with a few missing boundary cells.
    - Non-background, non-gray singleton pixels are plugs; their colors encode slide directions.

    Rule:
    - Treat every non-black, non-gray pixel as a plug.
    - Slide each plug in its color-coded direction until the next cell would be gray or leave the grid; clear the original cell and keep the plug color at the landing cell.
    - After all plugs move, treat gray and plug cells together as the closed outline.
    - Scan each row from left to right, pair consecutive outline runs, and fill black cells strictly between each pair with red.
    - Preserve all outline pixels in their original colors.

    Key insight:
    - The plug colors do not say what to paint; they say how to repair the missing boundary so the later row-wise fill becomes well-defined.
    """
    moved_grid, _moved = slide_plugs(np.asarray(grid, dtype=int))
    return row_scan_fill(moved_grid)


def build_region(h: int, w: int) -> np.ndarray | None:
    region = np.zeros((h, w), dtype=bool)
    top = random.randint(0, 1)
    bottom = h - 1 - random.randint(0, 1)
    left_band = sorted(random.sample(range(1, h - 1), 2))
    right_band = sorted(random.sample(range(1, h - 1), 2))
    left_inset = random.choice([1, 1, 2])
    right_inset = random.choice([0, 1, 2])
    if right_inset == 0 and random.random() < 0.6:
        right_inset = 1

    row_left = []
    row_right = []
    for r in range(h):
        left = 0
        right = w - 1
        if left_band[0] <= r <= left_band[1]:
            left += left_inset
        if right_band[0] <= r <= right_band[1]:
            right -= right_inset
        if r == top:
            left += random.choice([0, 0, 1])
            right -= random.choice([0, 0, 1])
        if r == bottom:
            left += random.choice([0, 0, 1])
            right -= random.choice([0, 0, 1])
        if right - left + 1 < 5:
            return None
        row_left.append(left)
        row_right.append(right)
        region[r, left:right + 1] = True

    if all(l == row_left[0] for l in row_left) and all(r == row_right[0] for r in row_right):
        return None
    return region


def outline_candidates(region: np.ndarray, outline: np.ndarray) -> list[dict]:
    h, w = region.shape
    cands: list[dict] = []
    for r0, c0 in np.argwhere(outline):
        r = int(r0)
        c = int(c0)
        for color, (dr, dc) in DIR_BY_COLOR.items():
            fr, fc = r + dr, c + dc
            br, bc = r - dr, c - dc
            if not in_bounds((h, w), fr, fc) or not bool(outline[fr, fc]):
                continue
            if not in_bounds((h, w), br, bc) or bool(region[br, bc]):
                continue
            max_dist = 0
            cells: list[tuple[int, int]] = []
            for dist in range(1, 5):
                sr = r - dr * dist
                sc = c - dc * dist
                if not in_bounds((h, w), sr, sc) or bool(region[sr, sc]) or bool(outline[sr, sc]):
                    break
                max_dist = dist
                cells.append((sr, sc))
            if max_dist == 0:
                continue
            cands.append(
                {
                    "target": (r, c),
                    "color": color,
                    "dir": (dr, dc),
                    "forward": (fr, fc),
                    "starts": tuple(cells),
                }
            )
    return cands


def choose_plugs(cands: list[dict]) -> list[dict] | None:
    variants: list[dict] = []
    for cand in cands:
        dr, dc = cand["dir"]
        for start in cand["starts"]:
            sr, sc = start
            tr, tc = cand["target"]
            path = []
            rr, cc = sr, sc
            while (rr, cc) != (tr, tc):
                path.append((rr, cc))
                rr += dr
                cc += dc
            path.append((tr, tc))
            variants.append(
                {
                    "target": cand["target"],
                    "color": cand["color"],
                    "dir": cand["dir"],
                    "forward": cand["forward"],
                    "start": start,
                    "path": tuple(path),
                }
            )

    for _ in range(250):
        goal = random.choice([2, 3, 3])
        pool = variants[:]
        random.shuffle(pool)
        chosen: list[dict] = []
        used_targets: set[tuple[int, int]] = set()
        removed_targets: set[tuple[int, int]] = set()
        blocked_cells: set[tuple[int, int]] = set()
        used_colors: set[int] = set()

        for prefer_new_dir in (True, False):
            for item in pool:
                if len(chosen) >= goal:
                    break
                if prefer_new_dir and item["color"] in used_colors:
                    continue
                target = item["target"]
                forward = item["forward"]
                path_cells = set(item["path"])
                if target in removed_targets or forward in removed_targets:
                    continue
                if target in blocked_cells or forward in path_cells:
                    continue
                if path_cells & blocked_cells:
                    continue
                if any(cell in removed_targets for cell in path_cells if cell != target):
                    continue
                chosen.append(item)
                used_targets.add(target)
                removed_targets.add(target)
                blocked_cells |= path_cells
                used_colors.add(item["color"])
            if len(chosen) >= goal:
                break

        if len(chosen) < goal:
            continue
        if len({item["color"] for item in chosen}) < min(2, goal):
            continue
        return chosen
    return None


def translate_mask(mask: np.ndarray, top: int, left: int, shape: tuple[int, int]) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros(shape, dtype=bool)
    out[top:top + h, left:left + w] = mask
    return out


def build_closed_instance() -> tuple[np.ndarray, np.ndarray, list[dict]] | None:
    gh = random.randint(10, 15)
    gw = random.randint(10, 15)
    ph = random.randint(6, min(9, gh - 3))
    pw = random.randint(7, min(10, gw - 3))
    local = build_region(ph, pw)
    if local is None:
        return None
    boundary = boundary_from_region(local)
    boundary_count = int(boundary.sum())
    if not (18 <= boundary_count <= 28):
        return None

    inner = local & (~boundary)
    if int(inner.sum()) < 6:
        return None
    widths = [int(np.count_nonzero(inner[r])) for r in range(ph) if int(np.count_nonzero(inner[r])) > 0]
    if len(set(widths)) < 2:
        return None

    top = random.randint(1, gh - ph - 1)
    left = random.randint(1, gw - pw - 1)
    full_region = translate_mask(local, top, left, (gh, gw))
    full_boundary = translate_mask(boundary, top, left, (gh, gw))
    cands = outline_candidates(full_region, full_boundary)
    if len(cands) < 2:
        return None
    chosen = choose_plugs(cands)
    if chosen is None:
        return None

    closed = np.zeros((gh, gw), dtype=int)
    closed[full_boundary] = GREY
    return closed, full_region, chosen


def semantic_check(inp: np.ndarray, closed: np.ndarray, region: np.ndarray, chosen: list[dict]) -> bool:
    try:
        moved, moves = slide_plugs(inp)
        out = main(inp)
    except Exception:
        return False

    expected_targets = {tuple(item["target"]) for item in chosen}
    actual_targets = {tuple(item["target"]) for item in moves}
    if expected_targets != actual_targets:
        return False
    if any(item["distance"] <= 0 for item in moves):
        return False

    if not np.array_equal((moved == GREY) | np.isin(moved, list(DIR_BY_COLOR)), closed != BG):
        return False

    raw_fill = row_scan_fill(inp)
    if np.array_equal(raw_fill, out):
        return False

    exterior = exterior_background(moved)
    interior = (~exterior) & (moved == BG)
    if int(interior.sum()) == 0:
        return False
    if not np.array_equal(out == FILL, interior):
        return False

    comps = connected_components(interior)
    if len(comps) != 1:
        return False

    target_rows = [int(np.count_nonzero(out[r] == FILL)) for r in range(out.shape[0]) if int(np.count_nonzero(out[r] == FILL)) > 0]
    if len(set(target_rows)) < 2:
        return False

    if not np.array_equal(interior, region & (~(closed != BG))):
        return False

    for item in chosen:
        altered = inp.copy()
        sr, sc = item["start"]
        altered[sr, sc] = BG
        try:
            alt_out = main(altered)
        except Exception:
            continue
        if np.array_equal(alt_out, out):
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(400):
        built = build_closed_instance()
        if built is None:
            continue
        closed, region, chosen = built
        inp = closed.copy()
        for item in chosen:
            tr, tc = item["target"]
            sr, sc = item["start"]
            inp[tr, tc] = BG
            inp[sr, sc] = item["color"]
        if semantic_check(inp, closed, region, chosen):
            return inp
    raise RuntimeError("failed to generate a valid sliding-outline input")
