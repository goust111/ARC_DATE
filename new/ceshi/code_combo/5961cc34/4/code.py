# Combo variant of 5961cc34 (index 4)
# combo_spec: D5/0+D6/0
# Name: 多端口色块+穿透激活

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
        spec = importlib.util.spec_from_file_location("orig_5961cc34_combo4", _ORIGINAL_SOLVER)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _ORIGINAL_MAIN = module.main
    return _ORIGINAL_MAIN


# ---------------------------------------------------------------------------
# Design analysis for combo D5/0 + D6/0
#
# a) concretization + behavior trace:
#    - Concrete scene: blue diamond/octagon blobs still sit on an azure background,
#      but every blob now carries 2 green edge bars, not 1. The bottom start marker
#      is unchanged: one yellow cap above a short red post.
#    - Original task trace:
#      start ray goes up -> first blob hit activates -> that blob emits one outgoing
#      ray -> each ray stops at the first next blob. The visible network is basically
#      a single bendy chain.
#    - Combo trace:
#      start ray goes up through the whole column -> it activates blob A and keeps
#      going to activate blob B (sometimes blob C). Each activated blob then emits
#      both of its side ports, so one penetrative trunk hit turns into several red
#      branches. The key visible difference is "one vertical ray lights multiple
#      multi-port blobs", not merely "first hit plus one continuation".
#    - Small mental example:
#      two stacked blobs share the same launch column. In the input both blobs show
#      left+right green exits. In the output the start column becomes a continuous
#      red line through both blobs, and both blobs also send red rays left and right.
#
# b) complexity calibration:
#    - The generator forces at least 2 trunk blobs on the start column, so D6 is
#      always visible and never collapses back to first-hit behavior.
#    - Activated blobs always have exactly 2 distinct ports (left and right). This
#      keeps the rule branchy enough to show D5, but avoids excessive clutter from
#      3-way branching plus penetration at the same time.
#    - Optional distractors remain off every ray and do not introduce another axis
#      of variation.
#
# c) implicit assumption review:
#    - Original solver assumed blue-only components were sufficient. With several
#      green sides, parsing by blue+green footprint is safer, so blobs are recovered
#      from the combined mask and then split into blue/green subsets.
#    - Original solver relied on two "break on first blob" stops: once on the start
#      ray and once on port rays. Both breaks must disappear for D6.
#    - Port parsing can no longer assume exactly one direction. The new solver must
#      collect every green-bearing edge and emit from all of them.
#    - Activating a blob still depends on crossing a blue pixel. The generator
#      therefore aligns the penetrative trunk with blue entry/exit sides, never with
#      a green side.
#
# d) placement legality:
#    - The generator uses the original output-first spirit: it first decides a valid
#      trunk of 2-3 stacked multi-port blobs, then derives the input by restoring
#      their blue/green bodies and erasing rays.
#    - Trunk blobs are vertically separated by at least one background row and are
#      aligned so the start column crosses blue cells on both the bottom and top
#      edges, guaranteeing a legal penetrative scan.
#    - Distractors are placed with 8-neighbor padding and are forbidden to overlap
#      any red ray trajectory.
#
# e) visual observability:
#    - A human can infer D6 by comparing that the same upward red line turns multiple
#      blobs red, not just the first one.
#    - A human can infer D5 because each active blob visibly has two green exits and
#      the output converts both exits into red side rays.
#    - No rule depends on hidden color ordering or non-visual metadata.
#
# f) cleanliness:
#    - To keep the board readable, activated blobs in the main trunk only emit left
#      and right rays. This avoids ambiguous overlap with the vertical trunk.
#    - Parallel horizontal branches at different heights are legible, and distractors
#      remain isolated from the ray system.
#
# g) inferability of color change:
#    - Only the same original palette is used: blue/green blobs become red exactly
#      when they lie on the propagated network; untouched distractors disappear in
#      the output because the output is a "red network only" rendering.
#    - The color change is fully explained by structural reachability, which is
#      visible from the start post, the stacked trunk hits, and the dual green ports.
#
# New main() assumptions audited against generate_input():
#    1. There is exactly one bottom red post with a non-red cap.
#    2. Blue is the dominant non-bg/start color and green is the unique color
#       4-adjacent to blue.
#    3. Every blob is a single 8-connected blue+green component with 2 ports.
#    4. Port directions are axis-aligned and each port width is 1 or 2 cells.
#    5. Start-ray-trunk entry/exit sides are blue, so penetration always crosses
#       blue body cells rather than grazing only green markers.
#    6. At most one distractor blob exists and it is disjoint from every ray path.
#    The generator explicitly enforces all six assumptions.
# ---------------------------------------------------------------------------


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild a red branching network where rays scan through several multi-port blobs.

    Input structure:
    - The background is the dominant color. Each large blue blob has 2 green edge bars
      on different sides, and each bar marks one outgoing emission direction.
    - The launch marker is a yellow cap above a short red post that touches the bottom.

    Rule:
    - Repaint the post and cap red, then sweep a thin red ray upward all the way to the
      top; every blob crossed by that scan activates, and the scan does not stop at the
      first hit.
    - Every activated blob turns red and emits a thick ray through each of its green
      ports. Those thick rays also keep travelling after crossing blobs, activating every
      new blob on their line until they leave the grid.

    Key insight: The green markers now describe several exits per blob, while the rays
    themselves are no longer first-hit blockers, so the output is a penetrative BFS over
    a branching relay graph rather than a single linear chain.
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
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
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
            port_pixels: list[tuple[int, int]] = []
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

    while queue:
        bi = int(queue.pop(0))
        for port in blobs[bi]["ports"]:
            extend_thick(port["pixels"], port["dir"])

    assert len(blobs) - len(activated) <= 1, "at most 1 distractor blob"
    return out


BG = 8
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRS = (UP, DOWN, LEFT, RIGHT)

SHAPES = ((3, 4), (4, 4), (5, 4), (5, 6), (6, 6), (8, 8))
_SHAPE_WIDTHS = {
    (3, 4): (2, 4, 2),
    (4, 4): (2, 4, 4, 2),
    (5, 4): (2, 4, 4, 4, 2),
    (5, 6): (2, 4, 6, 4, 2),
    (6, 6): (2, 4, 6, 6, 4, 2),
    (8, 8): (2, 4, 6, 8, 8, 6, 4, 2),
}

H_MIN, H_MAX = 15, 27
W_MIN, W_MAX = 15, 27
START_COL_PAD = 2
START_H_MIN, START_H_MAX = 2, 3

START_BG_GAP = 2
TRUNK_BG_GAP = 1
MAX_BLOB_COUNT = 4
MAX_DISTRACTORS = 1
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
        green_set |= set(green)
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
TRUNK_SPECS = [spec for spec in ALL_SPECS if spec["port_dirs"] == (LEFT, RIGHT)]


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


def _trace_to_boundary(
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


def _place_blob(inp: np.ndarray, occupied: np.ndarray, blue_mask: np.ndarray, blob: dict) -> None:
    paste(inp, blob["spec"]["patch"], blob["r0"], blob["c0"], transparent=BG)
    for r, c in blob["full"]:
        occupied[r, c] = True
    for r, c in blob["blue"]:
        blue_mask[r, c] = True


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


def _find_trunk_layout(
    H: int,
    W: int,
    start_col: int,
    cap_r: int,
    active_count: int,
    occupied: np.ndarray,
) -> list[dict] | None:
    initial_limit = cap_r - START_BG_GAP - 1
    if initial_limit < 0:
        return None

    def dfs(entry_limit: int, remaining: int, occ: np.ndarray) -> list[dict] | None:
        if remaining == 0:
            return []
        occ_pad = _dilate8(occ)
        candidates: list[dict] = []

        for spec in TRUNK_SPECS:
            h, w = spec["mask"].shape
            down_edge = _edge_offsets(spec["mask"], DOWN)
            up_edge = _edge_offsets(spec["mask"], UP)
            assert len(down_edge) == len(up_edge)
            for idx, entry_local in enumerate(down_edge):
                c0 = start_col - int(entry_local[1])
                if not (0 <= c0 <= W - w):
                    continue
                min_entry_row = int(entry_local[0])
                if min_entry_row > entry_limit:
                    continue
                rows = list(range(min_entry_row, entry_limit + 1))
                random.shuffle(rows)
                for entry_row in rows:
                    r0 = int(entry_row - entry_local[0])
                    full = _offset(spec["full"], r0, c0)
                    if any(occ_pad[r, c] for r, c in full):
                        continue
                    candidates.append(
                        {
                            "spec": spec,
                            "r0": r0,
                            "c0": c0,
                            "full": full,
                            "blue": _offset(spec["blue"], r0, c0),
                            "ports_by_dir": {
                                d: _sort_on_ray(_offset(local, r0, c0), d)
                                for d, local in spec["ports_by_dir"].items()
                            },
                            "entry": (entry_row, start_col),
                            "exit": (r0 + int(up_edge[idx][0]), c0 + int(up_edge[idx][1])),
                        }
                    )

        random.shuffle(candidates)
        for cand in candidates:
            occ2 = occ.copy()
            for r, c in cand["full"]:
                occ2[r, c] = True
            next_limit = cand["r0"] - TRUNK_BG_GAP - 1
            rest = dfs(next_limit, remaining - 1, occ2)
            if rest is not None:
                return [cand] + rest
        return None

    return dfs(initial_limit, active_count, occupied.copy())


def _active_count_for(cap_r: int) -> int:
    if cap_r >= 11:
        return random.choices([2, 3], weights=[4, 1])[0]
    return 2


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
    occupied[cap_r, start_col] = True
    ray_mask[cap_r, start_col] = True
    for r in range(cap_r + 1, H):
        inp[r, start_col] = RED
        out[r, start_col] = RED
        occupied[r, start_col] = True

    active_count = _active_count_for(cap_r)
    if active_count > MAX_BLOB_COUNT:
        return None

    trunk = _find_trunk_layout(H, W, start_col, cap_r, active_count, occupied)
    if trunk is None:
        return None

    for blob in trunk:
        _place_blob(inp, occupied, blue_mask, blob)

    start_path = _trace_to_boundary([(cap_r, start_col)], UP, True, H, W)
    _mark_ray(out, ray_mask, start_path)
    for blob in trunk:
        paint_coords(out, blob["blue"], RED)
    for blob in trunk:
        for d in (LEFT, RIGHT):
            side_path = _trace_to_boundary(blob["ports_by_dir"][d], d, False, H, W)
            _mark_ray(out, ray_mask, side_path)

    max_distractors = min(MAX_DISTRACTORS, MAX_BLOB_COUNT - active_count)
    num_distractors = random.randint(0, max_distractors)
    for _ in range(num_distractors):
        if not _place_distractor(inp, occupied, blue_mask, ray_mask):
            return None

    combo_out = main(inp)
    if not np.array_equal(combo_out, out):
        return None
    try:
        orig_out = _original_main()(inp)
    except Exception:
        orig_out = None
    if orig_out is not None and np.array_equal(orig_out, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 5961cc34 combo input")
