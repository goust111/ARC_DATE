# Variant of 5961cc34
# Dimension: D5 — 色块端口结构 (src)
# Change: 每个色块恰好一条边有1-2个相邻绿色像素作为端口，端口方向决定射线发射方向... → 多端口色块

from __future__ import annotations

from collections import Counter
import importlib.util
from itertools import combinations
from pathlib import Path
import random

import numpy as np

from arc_utils import colors_present, components_mask, get_bg, paint_coords, paste


_ORIGINAL_MAIN = None
_ORIGINAL_SOLVER = Path("/home/lixinyu/snap/new/arc2_solver_select/5961cc34.py")


def _original_main():
    global _ORIGINAL_MAIN
    if _ORIGINAL_MAIN is None:
        spec = importlib.util.spec_from_file_location("orig_5961cc34_d5", _ORIGINAL_SOLVER)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _ORIGINAL_MAIN = module.main
    return _ORIGINAL_MAIN


# Design analysis for the D5 variant
# - Concrete form:
#   The blob silhouette stays the same diamond/octagon family, but each blob now has
#   2 or 3 distinct green edge bars instead of exactly 1. Every green edge bar emits
#   its own red ray after the blob is activated, so one hit can split into several
#   outgoing directions. The start marker, colors, blob shapes, and erase-to-bg output
#   convention stay unchanged.
# - Behavior trace:
#   Original: start ray hits blob A -> A has one port -> one thick ray continues ->
#   maybe hits blob B -> linear chain.
#   Variant: start ray hits root blob A -> A has 2-3 ports -> multiple rays leave A
#   at once -> at least two of them hit different child blobs -> each child then emits
#   all of its own ports to the boundary. The visible difference is a branching red
#   network rather than a single corridor.
# - Complexity calibration:
#   The generator always creates multi-port blobs and forces the root blob to activate
#   at least two child blobs, so the branch is visible in a small number of examples.
#   Port counts are restricted to 2 or 3 and directions stay axis-aligned to keep the
#   rule learnable.
# - Hidden assumptions from the original solver:
#   The original code extracted blobs from blue pixels alone. With several green bars,
#   blue-only connectivity can become fragile on small silhouettes, so the new solver
#   groups blobs using the combined blue+green footprint and stores blue/green subsets
#   inside each blob. The BFS emission loop itself already generalizes cleanly to
#   multiple ports, but it now relies on 2-3 distinct directions per blob.
# - Placement legality:
#   The generator remains output-first. It places the root from the start ray, then
#   places two child blobs on two different outgoing root ports, then extends all
#   remaining ports to the boundary. Blob footprints are kept disjoint with 8-neighbor
#   padding, and rays are rejected if they would cross existing rays or blob cells
#   outside their own source blob.
# - Visual observability:
#   The branch is structural, not color-id based. A human can see that one activated
#   blob has several green edges and that the output turns each of those exits into a
#   red branch. Root branching is mandatory so the variant does not collapse back to a
#   visually linear example.
# - Visual cleanliness / color inference:
#   Colors are unchanged from the source task. Overlap is minimized by forbidding ray
#   crossings away from the source blob, keeping the output legible.
# - New main() assumptions audited against generate_input():
#   1. Blue is still the dominant non-bg/start color.
#   2. Green is the unique non-bg/start color 4-adjacent to blue.
#   3. Every blue+green blob component is separated from others by at least one cell.
#   4. Every blob has 2 or 3 distinct port directions, each bar length 1 or 2.
#   5. Incoming rays always hit a blue entry side, never a green side.
#   6. No ray crosses an existing ray except within the emitting blob footprint.
#   The generator explicitly enforces all six conditions.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct a red branching ray tree from blue blobs with several green exits.

    Input structure:
    - Background is the most common color. Large blue blobs have 2-3 green edge bars,
      with each green side indicating one outgoing ray direction for that blob.
    - The start marker is still a yellow cap on top of a short red segment at the bottom.

    Rule:
    - Fire a thin upward ray from the start until it reaches the first blob and activate it.
    - Every activated blob turns red and emits a thick ray through each of its green ports.
    - Rays keep traveling until they leave the grid or hit new blobs; newly hit blobs repeat
      the process until no additional blobs are activated.

    Key insight: The green markings no longer encode a single continuation; they encode a set
    of branching exits, so the final red structure is a BFS-style ray tree rooted at the start.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    start_candidates: list[tuple[int, int, int, list[tuple[int, int]]]] = []
    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=8):
            if coords.size == 0:
                continue
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
            if c0 != c1 or (r1 - r0 + 1) < 2:
                continue
            if r1 != H - 1:
                continue
            if any(int(g[rr, c0]) != int(col) for rr in range(r0, r1 + 1)):
                continue
            if r0 - 1 < 0:
                continue
            cap = int(g[r0 - 1, c0])
            if cap == bg or cap == int(col):
                continue
            start_candidates.append(
                (int(col), int(cap), int(c0), [(int(rr), int(c0)) for rr in range(r0, r1 + 1)])
            )

    assert len(start_candidates) == 1
    red_col, yellow_col, start_col, start_seg = start_candidates[0]

    counts = Counter(map(int, g.flatten()))
    other_cols = [(int(cnt), int(col)) for col, cnt in counts.items() if int(col) not in (bg, red_col, yellow_col)]
    assert other_cols
    max_count = max(cnt for cnt, _col in other_cols)
    blue_candidates = [int(col) for cnt, col in other_cols if cnt == max_count]
    assert len(blue_candidates) == 1
    blue_col = int(blue_candidates[0])

    green_candidates: list[int] = []
    for col in colors_present(g, exclude={bg, red_col, yellow_col, blue_col}):
        pts = np.argwhere(g == int(col))
        if pts.size == 0:
            continue
        ok = False
        for r, c in pts:
            r = int(r)
            c = int(c)
            for dr, dc in DIRS:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == blue_col:
                    ok = True
                    break
            if ok:
                break
        if ok:
            green_candidates.append(int(col))

    assert len(green_candidates) == 1
    green_col = int(green_candidates[0])

    out = np.full_like(g, bg)

    blob_id = np.full((H, W), -1, dtype=int)
    blobs: list[dict] = []
    combo_mask = (g == blue_col) | (g == green_col)
    for coords in components_mask(combo_mask, connectivity=8):
        full = [(int(r), int(c)) for r, c in coords]
        blue_pts = [(r, c) for r, c in full if int(g[r, c]) == blue_col]
        green_pts = [(r, c) for r, c in full if int(g[r, c]) == green_col]
        assert blue_pts and green_pts
        bi = len(blobs)
        for r, c in blue_pts:
            blob_id[r, c] = bi
        blobs.append({"full": full, "blue": blue_pts, "green": green_pts, "ports": []})

    for blob in blobs:
        full = blob["full"]
        blue_set = set(blob["blue"])
        green_pts = blob["green"]
        min_r = min(r for r, _c in full)
        max_r = max(r for r, _c in full)
        min_c = min(c for _r, c in full)
        max_c = max(c for _r, c in full)
        edge_greens = {
            UP: [(r, c) for r, c in green_pts if r == min_r],
            DOWN: [(r, c) for r, c in green_pts if r == max_r],
            LEFT: [(r, c) for r, c in green_pts if c == min_c],
            RIGHT: [(r, c) for r, c in green_pts if c == max_c],
        }
        for d in DIRS:
            greens = edge_greens[d]
            if not greens:
                continue
            port_pixels = []
            for gr, gc in greens:
                br = gr - d[0]
                bc = gc - d[1]
                assert (br, bc) in blue_set
                port_pixels.append((int(br), int(bc)))
            port_pixels = _sort_on_ray(port_pixels, d)
            assert 1 <= len(port_pixels) <= 2
            blob["ports"].append({"dir": d, "pixels": port_pixels})
        assert 2 <= len(blob["ports"]) <= 3

    activated: set[int] = set()
    queue: list[int] = []

    def activate_blob_at(r: int, c: int) -> None:
        bi = int(blob_id[int(r), int(c)])
        assert bi >= 0
        if bi in activated:
            return
        activated.add(bi)
        queue.append(bi)
        paint_coords(out, blobs[bi]["blue"], red_col)

    for r, c in start_seg:
        out[r, c] = red_col
    top_r = min(int(r) for r, _c in start_seg)
    assert top_r > 0 and int(g[top_r - 1, start_col]) == yellow_col
    out[top_r - 1, start_col] = red_col
    for r in range(top_r - 1, -1, -1):
        out[r, start_col] = red_col
        if int(g[r, start_col]) == blue_col:
            activate_blob_at(r, start_col)
            break

    def extend_thick(port_pixels: list[tuple[int, int]], d: tuple[int, int]) -> None:
        dr, dc = d
        for pr, pc in port_pixels:
            r, c = int(pr), int(pc)
            while True:
                r += dr
                c += dc
                if not (0 <= r < H and 0 <= c < W):
                    break
                out[r, c] = red_col
                if int(g[r, c]) == blue_col:
                    activate_blob_at(r, c)
                    break

    while queue:
        bi = int(queue.pop(0))
        for port in blobs[bi]["ports"]:
            extend_thick(port["pixels"], port["dir"])

    assert len(blobs) - len(activated) <= 1, "at most 1 distractor blob"
    return out


# --- Colors ---
BG = 8
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

# --- Directions ---
UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRS = (UP, DOWN, LEFT, RIGHT)
OPP = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}

# --- Diamond/octagon shapes (h, w) ---
SHAPES = ((3, 4), (4, 4), (5, 4), (5, 6), (6, 6), (8, 8))
_SHAPE_WIDTHS = {
    (3, 4): (2, 4, 2),
    (4, 4): (2, 4, 4, 2),
    (5, 4): (2, 4, 4, 4, 2),
    (5, 6): (2, 4, 6, 4, 2),
    (6, 6): (2, 4, 6, 6, 4, 2),
    (8, 8): (2, 4, 6, 8, 8, 6, 4, 2),
}

# --- Canvas bounds ---
H_MIN, H_MAX = 13, 27
W_MIN, W_MAX = 15, 27
START_COL_PAD = 2

# --- Start segment ---
START_H_MIN, START_H_MAX = 2, 3

# --- Branching / distractor ---
ROOT_PORT_COUNT_CHOICES = (2, 3)
CHILD_BLOB_COUNT = 2
MAX_DISTRACTORS = 1

# --- Placement ---
MIN_GAP_FIRST = 4
MIN_GAP_BRANCH = 3

# --- Retry ---
GEN_RETRIES = 400


def _shape_mask(h: int, w: int) -> np.ndarray:
    widths = _SHAPE_WIDTHS[(h, w)]
    mask = np.zeros((h, w), dtype=bool)
    for r, ww in enumerate(widths):
        c0 = (w - ww) // 2
        mask[r, c0 : c0 + ww] = True
    return mask


def _perp_value(rc: tuple[int, int], d: tuple[int, int]) -> int:
    return rc[1] if d[0] else rc[0]


def _along_value(rc: tuple[int, int], d: tuple[int, int]) -> int:
    return rc[0] if d[0] else rc[1]


def _sort_on_ray(coords: list[tuple[int, int]], d: tuple[int, int]) -> list[tuple[int, int]]:
    return sorted(coords, key=lambda p: _perp_value(p, d))


def _edge_offsets(mask: np.ndarray, d: tuple[int, int]) -> list[tuple[int, int]]:
    pts = np.argwhere(mask)
    axis = 0 if d in (UP, DOWN) else 1
    extreme = pts[:, axis].min() if d in (UP, LEFT) else pts[:, axis].max()
    keep = pts[:, axis] == extreme
    return _sort_on_ray([(int(r), int(c)) for r, c in pts[keep]], d)


def _center_port_window(edge: list[tuple[int, int]], d: tuple[int, int]) -> list[tuple[int, int]]:
    width = min(2, len(edge))
    start = (len(edge) - width) // 2
    return _sort_on_ray(edge[start : start + width], d)


def _blob_spec(size: tuple[int, int], port_dirs: tuple[tuple[int, int], ...]) -> dict | None:
    h, w = size
    mask = _shape_mask(h, w)

    ports_local: dict[tuple[int, int], list[tuple[int, int]]] = {}
    green_set: set[tuple[int, int]] = set()
    for d in port_dirs:
        edge = _edge_offsets(mask, d)
        green = _center_port_window(edge, d)
        if not (1 <= len(green) <= 2):
            return None
        if green_set & set(green):
            return None
        green_set |= set(green)
        port_pixels: list[tuple[int, int]] = []
        for gr, gc in green:
            br = gr - d[0]
            bc = gc - d[1]
            if not (0 <= br < h and 0 <= bc < w):
                return None
            if not mask[br, bc] or (br, bc) in green_set:
                return None
            port_pixels.append((int(br), int(bc)))
        if len(set(port_pixels)) != len(port_pixels):
            return None
        ports_local[d] = _sort_on_ray(port_pixels, d)

    blue_mask = mask.copy()
    for r, c in green_set:
        blue_mask[r, c] = False
    if len(components_mask(blue_mask, connectivity=8)) != 1:
        return None

    patch = np.full((h, w), BG, dtype=int)
    patch[mask] = BLUE
    for r, c in green_set:
        patch[r, c] = GREEN

    blue = [(int(r), int(c)) for r, c in np.argwhere(blue_mask)]
    full = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    return {
        "size": (h, w),
        "mask": mask,
        "patch": patch,
        "blue": blue,
        "full": full,
        "port_dirs": tuple(port_dirs),
        "ports_by_dir": ports_local,
    }


def _build_all_specs() -> list[dict]:
    specs: list[dict] = []
    for size in SHAPES:
        for k in (2, 3):
            for dirs in combinations(DIRS, k):
                spec = _blob_spec(size, tuple(dirs))
                if spec is not None:
                    specs.append(spec)
    return specs


ALL_SPECS = _build_all_specs()


def _dilate8(mask: np.ndarray) -> np.ndarray:
    H, W = mask.shape
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r0, r1 = max(0, dr), H + min(0, dr)
            c0, c1 = max(0, dc), W + min(0, dc)
            out[r0:r1, c0:c1] |= mask[r0 - dr : r1 - dr, c0 - dc : c1 - dc]
    return out


def _offset(offsets: list[tuple[int, int]], r0: int, c0: int) -> list[tuple[int, int]]:
    return [(r0 + r, c0 + c) for r, c in offsets]


def _in_bounds(coords: list[tuple[int, int]], H: int, W: int) -> bool:
    return all(0 <= r < H and 0 <= c < W for r, c in coords)


def _path_to_entries(
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    entries: list[tuple[int, int]],
    include_origin: bool,
    H: int,
    W: int,
) -> list[tuple[int, int]] | None:
    dr, dc = d
    cells: list[tuple[int, int]] = []
    for (r, c), (er, ec) in zip(starts, entries):
        if include_origin:
            cells.append((r, c))
        guard = H + W + 10
        while (r, c) != (er, ec):
            r += dr
            c += dc
            guard -= 1
            if guard < 0 or not (0 <= r < H and 0 <= c < W):
                return None
            cells.append((r, c))
    return cells


def _path_to_boundary(
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    include_origin: bool,
    H: int,
    W: int,
) -> list[tuple[int, int]]:
    dr, dc = d
    cells: list[tuple[int, int]] = []
    for r, c in starts:
        if include_origin:
            cells.append((r, c))
        while True:
            r += dr
            c += dc
            if not (0 <= r < H and 0 <= c < W):
                break
            cells.append((r, c))
    return cells


def _mark_ray(out: np.ndarray, ray_mask: np.ndarray, cells: list[tuple[int, int]]) -> None:
    paint_coords(out, cells, RED)
    paint_coords(ray_mask, cells, True)


def _mark_mask(mask: np.ndarray, cells: list[tuple[int, int]]) -> None:
    paint_coords(mask, cells, True)


def _path_is_clear(
    path: list[tuple[int, int]],
    occupied: np.ndarray,
    ray_mask: np.ndarray,
    allowed: set[tuple[int, int]],
) -> bool:
    for r, c in path:
        if (r, c) in allowed:
            continue
        if occupied[r, c] or ray_mask[r, c]:
            return False
    return True


def _terminal_paths(
    blob: dict,
    occupied: np.ndarray,
    ray_mask: np.ndarray,
    H: int,
    W: int,
) -> list[tuple[tuple[int, int], list[tuple[int, int]]]] | None:
    full_set = set(blob["full"])
    tmp_ray = ray_mask.copy()
    out: list[tuple[tuple[int, int], list[tuple[int, int]]]] = []
    dirs = list(blob["ports_by_dir"].keys())
    random.shuffle(dirs)
    for d in dirs:
        path = _path_to_boundary(blob["ports_by_dir"][d], d, False, H, W)
        if not _path_is_clear(path, occupied, tmp_ray, full_set):
            return None
        _mark_mask(tmp_ray, path)
        out.append((d, path))
    return out


def _try_place_along_axis(
    spec: dict,
    window: list[tuple[int, int]],
    d: tuple[int, int],
    lane_vals: list[int],
    starts: list[tuple[int, int]],
    include_origin: bool,
    min_gap: int,
    occ_pad: np.ndarray,
    blue_mask: np.ndarray,
    ray_mask: np.ndarray,
    H: int,
    W: int,
) -> list[dict]:
    h, w = spec["mask"].shape
    win_vals = [_perp_value(p, d) for p in window]
    edge_line = _along_value(window[0], d)
    start_line = _along_value(starts[0], d)
    is_vertical = bool(d[0])

    fixed = lane_vals[0] - win_vals[0]
    extent = H - h if is_vertical else W - w
    low = edge_line
    high = extent + edge_line
    if d in (UP, LEFT):
        high = min(high, start_line - min_gap)
    else:
        low = max(low, start_line + min_gap)

    entry_lines = list(range(low, high + 1))
    random.shuffle(entry_lines)
    candidates: list[dict] = []

    for entry_line in entry_lines:
        origin_along = entry_line - edge_line
        r0 = origin_along if is_vertical else fixed
        c0 = fixed if is_vertical else origin_along
        if is_vertical:
            entry_cells = [(entry_line, c0 + c) for _, c in window]
        else:
            entry_cells = [(r0 + r, entry_line) for r, _ in window]

        full = _offset(spec["full"], r0, c0)
        if not _in_bounds(full, H, W):
            continue
        if any(occ_pad[r, c] or ray_mask[r, c] for r, c in full):
            continue

        path = _path_to_entries(starts, d, entry_cells, include_origin, H, W)
        if path is None:
            continue
        entry_set = set(entry_cells)
        if any(blue_mask[r, c] for r, c in path if (r, c) not in entry_set):
            continue

        candidates.append(
            {
                "spec": spec,
                "r0": r0,
                "c0": c0,
                "entry": entry_cells,
                "path": path,
                "blue": _offset(spec["blue"], r0, c0),
                "full": full,
                "ports_by_dir": {
                    dd: _sort_on_ray(_offset(local, r0, c0), dd)
                    for dd, local in spec["ports_by_dir"].items()
                },
            }
        )

    return candidates


def _candidate_blob_placements(
    H: int,
    W: int,
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    include_origin: bool,
    occupied: np.ndarray,
    blue_mask: np.ndarray,
    ray_mask: np.ndarray,
) -> list[dict]:
    starts = _sort_on_ray(starts, d)
    thickness = len(starts)
    lane_vals = [_perp_value(p, d) for p in starts]
    lane_delta = [v - lane_vals[0] for v in lane_vals]
    entry_side = OPP[d]
    occ_pad = _dilate8(occupied)
    min_gap = MIN_GAP_FIRST if include_origin else MIN_GAP_BRANCH
    candidates: list[dict] = []

    specs = ALL_SPECS.copy()
    random.shuffle(specs)
    for spec in specs:
        if entry_side in spec["port_dirs"]:
            continue
        edge = _edge_offsets(spec["mask"], entry_side)
        if len(edge) < thickness:
            continue
        for k in range(len(edge) - thickness + 1):
            window = edge[k : k + thickness]
            win_delta = [_perp_value(p, d) - _perp_value(window[0], d) for p in window]
            if win_delta != lane_delta:
                continue
            candidates.extend(
                _try_place_along_axis(
                    spec,
                    window,
                    d,
                    lane_vals,
                    starts,
                    include_origin,
                    min_gap,
                    occ_pad,
                    blue_mask,
                    ray_mask,
                    H,
                    W,
                )
            )

    random.shuffle(candidates)
    return candidates


def _occupy_blob_masks(occupied: np.ndarray, blue_mask: np.ndarray, blob: dict) -> None:
    for r, c in blob["full"]:
        occupied[r, c] = True
    for r, c in blob["blue"]:
        blue_mask[r, c] = True


def _place_blob(inp: np.ndarray, occupied: np.ndarray, blue_mask: np.ndarray, blob: dict) -> None:
    paste(inp, blob["spec"]["patch"], blob["r0"], blob["c0"], transparent=BG)
    _occupy_blob_masks(occupied, blue_mask, blob)


def _commit_blob(
    inp: np.ndarray,
    out: np.ndarray,
    occupied: np.ndarray,
    blue_mask: np.ndarray,
    ray_mask: np.ndarray,
    blob: dict,
    inbound_path: list[tuple[int, int]],
    terminal_paths: list[tuple[tuple[int, int], list[tuple[int, int]]]],
) -> None:
    _mark_ray(out, ray_mask, inbound_path)
    _place_blob(inp, occupied, blue_mask, blob)
    paint_coords(out, blob["blue"], RED)
    for _d, path in terminal_paths:
        _mark_ray(out, ray_mask, path)


def _choose_terminal_child(
    H: int,
    W: int,
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    occupied: np.ndarray,
    blue_mask: np.ndarray,
    ray_mask: np.ndarray,
    parent_full: set[tuple[int, int]],
) -> dict | None:
    for cand in _candidate_blob_placements(H, W, starts, d, False, occupied, blue_mask, ray_mask):
        if not _path_is_clear(cand["path"], occupied, ray_mask, parent_full):
            continue
        tmp_occ = occupied.copy()
        tmp_blue = blue_mask.copy()
        tmp_ray = ray_mask.copy()
        _mark_mask(tmp_ray, cand["path"])
        _occupy_blob_masks(tmp_occ, tmp_blue, cand)
        terminal = _terminal_paths(cand, tmp_occ, tmp_ray, H, W)
        if terminal is None:
            continue
        cand["terminal_paths"] = terminal
        return cand
    return None


def _place_root_tree(
    H: int,
    W: int,
    cap_r: int,
    start_col: int,
    inp: np.ndarray,
    out: np.ndarray,
    occupied: np.ndarray,
    blue_mask: np.ndarray,
    ray_mask: np.ndarray,
) -> bool:
    root_candidates = _candidate_blob_placements(
        H, W, [(cap_r, start_col)], UP, True, occupied, blue_mask, ray_mask
    )
    start_allow = {(cap_r, start_col)}

    for root in root_candidates:
        if len(root["ports_by_dir"]) not in ROOT_PORT_COUNT_CHOICES:
            continue
        if not _path_is_clear(root["path"], occupied, ray_mask, start_allow):
            continue

        base_inp = inp.copy()
        base_out = out.copy()
        base_occ = occupied.copy()
        base_blue = blue_mask.copy()
        base_ray = ray_mask.copy()
        _mark_ray(base_out, base_ray, root["path"])
        _place_blob(base_inp, base_occ, base_blue, root)
        paint_coords(base_out, root["blue"], RED)

        port_dirs = list(root["ports_by_dir"].keys())
        combos = list(combinations(port_dirs, CHILD_BLOB_COUNT))
        random.shuffle(combos)
        root_full = set(root["full"])

        for child_dirs in combos:
            trial_inp = base_inp.copy()
            trial_out = base_out.copy()
            trial_occ = base_occ.copy()
            trial_blue = base_blue.copy()
            trial_ray = base_ray.copy()
            ok = True

            for d in child_dirs:
                child = _choose_terminal_child(
                    H,
                    W,
                    root["ports_by_dir"][d],
                    d,
                    trial_occ,
                    trial_blue,
                    trial_ray,
                    root_full,
                )
                if child is None:
                    ok = False
                    break
                _commit_blob(
                    trial_inp,
                    trial_out,
                    trial_occ,
                    trial_blue,
                    trial_ray,
                    child,
                    child["path"],
                    child["terminal_paths"],
                )

            if not ok:
                continue

            root_terminals: list[tuple[tuple[int, int], list[tuple[int, int]]]] = []
            for d in port_dirs:
                if d in child_dirs:
                    continue
                path = _path_to_boundary(root["ports_by_dir"][d], d, False, H, W)
                if not _path_is_clear(path, trial_occ, trial_ray, root_full):
                    ok = False
                    break
                root_terminals.append((d, path))
            if not ok:
                continue

            for _d, path in root_terminals:
                _mark_ray(trial_out, trial_ray, path)

            inp[:, :] = trial_inp
            out[:, :] = trial_out
            occupied[:, :] = trial_occ
            blue_mask[:, :] = trial_blue
            ray_mask[:, :] = trial_ray
            return True

    return False


def _place_distractor(
    inp: np.ndarray,
    occupied: np.ndarray,
    blue_mask: np.ndarray,
    ray_mask: np.ndarray,
) -> bool:
    H, W = inp.shape
    occ_pad = _dilate8(occupied)
    specs = ALL_SPECS.copy()
    random.shuffle(specs)
    candidates: list[dict] = []
    for spec in specs:
        h, w = spec["mask"].shape
        for r0 in range(H - h + 1):
            for c0 in range(W - w + 1):
                full = _offset(spec["full"], r0, c0)
                if any(occ_pad[r, c] or ray_mask[r, c] for r, c in full):
                    continue
                candidates.append(
                    {
                        "spec": spec,
                        "r0": r0,
                        "c0": c0,
                        "full": full,
                        "blue": _offset(spec["blue"], r0, c0),
                    }
                )
    if not candidates:
        return False
    _place_blob(inp, occupied, blue_mask, random.choice(candidates))
    return True


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(H_MIN, H_MAX)
    W = random.randint(W_MIN, W_MAX)
    inp = np.full((H, W), BG, dtype=int)
    out = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    blue_mask = np.zeros((H, W), dtype=bool)
    ray_mask = np.zeros((H, W), dtype=bool)

    start_h = random.randint(START_H_MIN, START_H_MAX)
    cap_r = H - start_h - 1
    start_col = random.randint(START_COL_PAD, W - START_COL_PAD - 1)
    inp[cap_r, start_col] = YELLOW
    out[cap_r, start_col] = RED
    ray_mask[cap_r, start_col] = True
    occupied[cap_r, start_col] = True
    for r in range(cap_r + 1, H):
        inp[r, start_col] = RED
        out[r, start_col] = RED
        occupied[r, start_col] = True

    if not _place_root_tree(H, W, cap_r, start_col, inp, out, occupied, blue_mask, ray_mask):
        return None

    if random.random() < 0.6:
        if not _place_distractor(inp, occupied, blue_mask, ray_mask):
            return None

    if not np.array_equal(main(inp), out):
        return None
    if np.array_equal(_original_main()(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 5961cc34 D5 variant input")
