# Variant of 20270e3b
# Dimension: D10 — 颜色体系 (meta)
# Change: 固定三色体系：bg=1（蓝），obj=4（黄），ctrl=7（橙）... → 连接器颜色作为匹配线索
#
# Design analysis
# - Concrete behavior:
#   Each sprite now has two horizontal connector bars instead of one. All connector bars in a
#   sample have the same length, so geometry alone does not identify the docking pair. Exactly
#   one connector color is shared by base and plug; that shared-color pair is the true docking
#   pair. The other connector on each sprite uses a different unshared color and is ignored.
# - Trace on the critical step:
#   Original task: base has one orange bar, plug has one orange bar, so solver can match by
#   "the only ctrl-colored bar" and by equal shape. Variant: base might have {red, green},
#   plug might have {red, maroon}. All four bars have the same length. The solver must first
#   find the unique shared color (red), then align only those red bars; shape is now only a
#   consistency check, not the key selector.
# - Complexity calibration:
#   Using exactly two connectors per sprite keeps the visual scene readable while making color
#   genuinely necessary. More connectors would add clutter; one connector would make the color
#   change degenerate because the old positional logic would still solve the sample.
# - Hidden assumptions from the original main():
#   1. `ctrl=7` fixed: no longer valid. We now treat every non-{bg,obj} color as a connector
#      color and recolor all such pixels to obj in the output.
#   2. "One connector per sprite": no longer valid. We now extract all horizontal same-color
#      runs and select the active pair by unique shared color.
#   3. "Equal connector shape identifies the pair": weakened. Equal shape is retained as a
#      sanity check after color matching, but generator deliberately makes distractors share
#      the same length so shape cannot decide the match.
#   4. Overlap test `cur == ctrl or val == ctrl`: no longer valid. Any overlap must involve at
#      least one connector-colored cell, because connector colors are now variable.
# - Placement legality:
#   Base still starts at the grid origin. Plug is still placed elsewhere without overlap.
#   Generator constructs the plug in docked coordinates first, verifies that only the active
#   docking seam overlaps base, then detaches and scatters the plug. This preserves the
#   original physical rule while adding extra colored bars safely.
# - Visual observability:
#   Across samples the shared connector color changes, while body color stays yellow. In each
#   input, the only repeated connector color across the two sprites predicts which bars line up
#   in the output. Because distractor bars remain visible in the input and disappear into yellow
#   after merging, the color-matching rule is directly inspectable from a few examples.
# - Visual cleanliness:
#   Exactly two connectors per sprite, all horizontal, and unique non-shared distractor colors
#   keep the scene legible. Generator rejects cases where connectors overlap improperly or where
#   the plug would extend above/left of the base after docking.
# - Color inferability:
#   Only three connector colors appear in a sample: one shared match color and two distractor
#   colors. This is enough to make the rule visible without creating a large arbitrary color map.
# - New main() assumptions and generator guarantees:
#   1. There are exactly two 8-connected foreground components. Generator enforces this.
#   2. Each sprite has exactly one connector run in the shared color. Generator enforces unique
#      connector colors within each sprite.
#   3. The shared-color runs have equal normalized shape and valid orientation. Generator uses a
#      single chosen connector length for all bars and only recolors valid boundary segments.
#   4. Docking overlap occurs only along the active seam. Generator validates the assembled state
#      before placing the sprites on the input canvas.

from __future__ import annotations

import random
import numpy as np

from arc_utils import bbox_coords, is_contiguous, label_mask, paste, random_free_location


def _crop_nonbg(arr: np.ndarray, bg: int) -> np.ndarray:
    nz = np.argwhere(arr != int(bg))
    if nz.size == 0:
        return np.full((0, 0), int(bg), dtype=int)
    r0, c0, r1, c1 = bbox_coords(nz)
    return arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _extract_components(g: np.ndarray, bg: int) -> list[dict]:
    lab, n = label_mask(g != int(bg), connectivity=8)
    comps: list[dict] = []
    for k in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(k))
        r0, c0, r1, c1 = bbox_coords(pts)
        sprite = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=int)
        for r, c in pts:
            sprite[int(r - r0), int(c - c0)] = int(g[int(r), int(c)])
        comps.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "sprite": sprite,
            }
        )
    comps.sort(key=lambda item: int(item["bbox"][0]) + int(item["bbox"][1]))
    return comps


def _connector_runs(sprite: np.ndarray, bg: int, obj: int) -> list[dict]:
    h, w = sprite.shape
    runs: list[dict] = []
    for r in range(h):
        c = 0
        while c < w:
            col = int(sprite[int(r), int(c)])
            if col in (int(bg), int(obj)):
                c += 1
                continue
            c0 = c
            while c + 1 < w and int(sprite[int(r), int(c + 1)]) == col:
                c += 1
            c1 = c
            runs.append(
                {
                    "color": int(col),
                    "row": int(r),
                    "c0": int(c0),
                    "c1": int(c1),
                    "pts": {(int(r), int(cc)) for cc in range(int(c0), int(c1) + 1)},
                }
            )
            c += 1
    return runs


def _run_shape(run: dict) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    pts = set(run["pts"])
    r0 = min(r for r, _c in pts)
    c0 = min(c for _r, c in pts)
    shape = {(int(r - r0), int(c - c0)) for r, c in pts}
    return shape, (int(r0), int(c0))


def _unique_run_by_color(runs: list[dict], color: int) -> dict:
    matches = [run for run in runs if int(run["color"]) == int(color)]
    assert len(matches) == 1
    return matches[0]


def _assemble_from_sprites(
    base_sprite: np.ndarray,
    plug_sprite: np.ndarray,
    *,
    bg: int,
    obj: int,
    match_color: int,
) -> np.ndarray:
    runs_base = _connector_runs(base_sprite, bg, obj)
    runs_plug = _connector_runs(plug_sprite, bg, obj)
    conn_colors = {int(run["color"]) for run in runs_base + runs_plug}

    run_base = _unique_run_by_color(runs_base, match_color)
    run_plug = _unique_run_by_color(runs_plug, match_color)

    shp_base, (br0, bc0) = _run_shape(run_base)
    shp_plug, (pr0, pc0) = _run_shape(run_plug)
    assert shp_base == shp_plug

    # Active base connector: body above, free below.
    for _r, c in run_base["pts"]:
        assert int(base_sprite[int(br0), int(c)]) == int(match_color)
        assert int(br0) > 0 and int(base_sprite[int(br0 - 1), int(c)]) == int(obj)
        assert int(br0 + 1) >= base_sprite.shape[0] or int(base_sprite[int(br0 + 1), int(c)]) == int(bg)

    # Active plug connector: free above, body below.
    for _r, c in run_plug["pts"]:
        assert int(plug_sprite[int(pr0), int(c)]) == int(match_color)
        assert int(pr0) == 0 or int(plug_sprite[int(pr0 - 1), int(c)]) == int(bg)
        assert int(pr0 + 1) < plug_sprite.shape[0] and int(plug_sprite[int(pr0 + 1), int(c)]) == int(obj)

    tr = int((br0 - 1) - pr0)
    tc = int(bc0 - pc0)

    base_pts = {(int(r), int(c)) for r, c in np.argwhere(base_sprite != int(bg))}
    plug_pts = {(int(r + tr), int(c + tc)) for r, c in np.argwhere(plug_sprite != int(bg))}
    pts_all = base_pts | plug_pts
    coords_all = np.array(list(pts_all), dtype=int)
    rr0, cc0, rr1, cc1 = bbox_coords(coords_all)
    rr0, cc0, rr1, cc1 = map(int, (rr0, cc0, rr1, cc1))
    assert rr0 == 0 and cc0 == 0

    out = np.full((int(rr1 + 1), int(cc1 + 1)), int(bg), dtype=int)

    for r, c in base_pts:
        out[int(r), int(c)] = int(base_sprite[int(r), int(c)])

    for r2, c2 in plug_pts:
        src = int(plug_sprite[int(r2 - tr), int(c2 - tc)])
        cur = int(out[int(r2), int(c2)])
        if cur != int(bg):
            assert cur in conn_colors or src in conn_colors
        out[int(r2), int(c2)] = int(src)

    out[(out != int(bg)) & (out != int(obj))] = int(obj)
    return _crop_nonbg(out, bg)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg, obj = 1, 4

    comps = _extract_components(g, bg)
    assert len(comps) == 2
    assert int(comps[0]["bbox"][0] + comps[0]["bbox"][1]) == 0 or int(comps[1]["bbox"][0] + comps[1]["bbox"][1]) == 0

    base_sprite = comps[0]["sprite"]
    plug_sprite = comps[1]["sprite"]
    if int(comps[1]["bbox"][0] + comps[1]["bbox"][1]) < int(comps[0]["bbox"][0] + comps[0]["bbox"][1]):
        base_sprite, plug_sprite = plug_sprite, base_sprite

    runs_base = _connector_runs(base_sprite, bg, obj)
    runs_plug = _connector_runs(plug_sprite, bg, obj)
    base_colors = {int(run["color"]) for run in runs_base}
    plug_colors = {int(run["color"]) for run in runs_plug}
    shared = sorted(base_colors & plug_colors)
    assert len(shared) == 1

    return _assemble_from_sprites(base_sprite, plug_sprite, bg=bg, obj=obj, match_color=int(shared[0]))


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

import random as _rng


def _grow_shape(mask: np.ndarray, target: int, frontier: list[tuple[int, int]]) -> None:
    count = int(mask.sum())
    while count < int(target) and frontier:
        idx = _rng.randint(0, len(frontier) - 1)
        r, c = frontier[idx]
        h, w = mask.shape
        nbrs = [
            (int(r + dr), int(c + dc))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            if 0 <= int(r + dr) < h and 0 <= int(c + dc) < w and not mask[int(r + dr), int(c + dc)]
        ]
        if not nbrs:
            frontier[idx] = frontier[-1]
            frontier.pop()
            continue
        nr, nc = _rng.choice(nbrs)
        mask[int(nr), int(nc)] = True
        count += 1
        frontier.append((int(nr), int(nc)))


def _thin(mask: np.ndarray, keep: set[tuple[int, int]] | None = None) -> None:
    h, w = mask.shape
    keep = keep or set()
    pixels = list(zip(*np.where(mask)))
    _rng.shuffle(pixels)
    for r, c in pixels:
        if (int(r), int(c)) in keep:
            continue
        adj_bg = any(
            not (0 <= int(r + dr) < h and 0 <= int(c + dc) < w) or not mask[int(r + dr), int(c + dc)]
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        )
        if adj_bg:
            continue
        mask[int(r), int(c)] = False
        if not is_contiguous(mask, background=0, connectivity=8):
            mask[int(r), int(c)] = True


def _valid_segments(mask: np.ndarray, mode: str, conn_len: int) -> list[dict]:
    h, w = mask.shape
    segs: list[dict] = []
    for r in range(h):
        for c0 in range(w - int(conn_len) + 1):
            cols = list(range(int(c0), int(c0 + conn_len)))
            if not all(bool(mask[int(r), int(c)]) for c in cols):
                continue
            if mode == "base":
                if int(r) == 0:
                    continue
                if not all(bool(mask[int(r - 1), int(c)]) for c in cols):
                    continue
                if not all(int(r + 1) >= h or not bool(mask[int(r + 1), int(c)]) for c in cols):
                    continue
            else:
                if int(r + 1) >= h:
                    continue
                if not all(bool(mask[int(r + 1), int(c)]) for c in cols):
                    continue
                if not all(int(r) == 0 or not bool(mask[int(r - 1), int(c)]) for c in cols):
                    continue
            segs.append(
                {
                    "row": int(r),
                    "c0": int(c0),
                    "c1": int(c0 + conn_len - 1),
                    "pts": {(int(r), int(c)) for c in cols},
                }
            )
    return segs


def _segments_compatible(a: dict, b: dict) -> bool:
    if set(a["pts"]) & set(b["pts"]):
        return False
    if int(a["row"]) == int(b["row"]):
        left = min(int(a["c1"]), int(b["c1"]))
        right = max(int(a["c0"]), int(b["c0"]))
        if right <= left + 1:
            return False
    return True


def _choose_two_segments(segs: list[dict]) -> tuple[dict, dict] | None:
    pairs = []
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            if _segments_compatible(segs[i], segs[j]):
                pairs.append((segs[i], segs[j]))
    if not pairs:
        return None
    return _rng.choice(pairs)


def _paint_segment(arr: np.ndarray, seg: dict, color: int) -> None:
    for r, c in seg["pts"]:
        arr[int(r), int(c)] = int(color)


def _build_base(conn_len: int, match_color: int, extra_color: int) -> tuple[np.ndarray, dict]:
    base_h = _rng.randint(3, 7)
    base_w = _rng.randint(max(3, conn_len + 1), 10)
    target = _rng.randint(6, min(29, base_h * base_w))
    mask = np.zeros((base_h, base_w), dtype=bool)
    mask[0, 0] = True
    _grow_shape(mask, target, [(0, 0)])
    if not is_contiguous(mask, background=0, connectivity=8):
        raise ValueError("base-not-contiguous")
    _thin(mask)

    segs = _valid_segments(mask, "base", conn_len)
    choice = _choose_two_segments(segs)
    if choice is None:
        raise ValueError("base-no-two-segments")
    active, distractor = choice

    bg, obj = 1, 4
    base_grid = np.full(mask.shape, bg, dtype=int)
    base_grid[mask] = obj
    _paint_segment(base_grid, active, match_color)
    _paint_segment(base_grid, distractor, extra_color)
    return base_grid, active


def _build_plug(
    conn_len: int,
    match_color: int,
    extra_color: int,
    active_base: dict,
    base_grid: np.ndarray,
) -> np.ndarray:
    bg, obj = 1, 4
    base_h, base_w = base_grid.shape
    plug_h = _rng.randint(3, 7)
    plug_w = _rng.randint(max(2, conn_len + 1), min(10, base_w + 4))

    max_conn_c0 = min(int(active_base["c0"]), int(plug_w - conn_len))
    if max_conn_c0 < 0:
        raise ValueError("plug-no-horizontal-room")
    active_c0 = _rng.randint(0, max_conn_c0)
    dc_offset = int(active_base["c0"] - active_c0)

    mask = np.zeros((plug_h, plug_w), dtype=bool)
    for c in range(active_c0, active_c0 + int(conn_len)):
        mask[0, int(c)] = True
        mask[1, int(c)] = True

    plug_max = min(29, plug_h * plug_w)
    if plug_max < 6:
        raise ValueError("plug-too-small")
    target = _rng.randint(6, plug_max)
    frontier = list(zip(*np.where(mask)))
    _grow_shape(mask, target, [(int(r), int(c)) for r, c in frontier])
    if not is_contiguous(mask, background=0, connectivity=8):
        raise ValueError("plug-not-contiguous")

    keep = {(0, int(c)) for c in range(active_c0, active_c0 + int(conn_len))}
    _thin(mask, keep=keep)

    segs = _valid_segments(mask, "plug", conn_len)
    active = None
    extra_candidates = []
    for seg in segs:
        if seg["row"] == 0 and seg["c0"] == int(active_c0) and seg["c1"] == int(active_c0 + conn_len - 1):
            active = seg
        else:
            extra_candidates.append(seg)
    if active is None:
        raise ValueError("plug-active-lost")

    extra_candidates = [seg for seg in extra_candidates if _segments_compatible(active, seg)]
    if not extra_candidates:
        raise ValueError("plug-no-extra-segment")
    distractor = _rng.choice(extra_candidates)

    # Validate docked overlap before coloring.
    tr = int(active_base["row"] - 1)
    for r in range(plug_h):
        for c in range(plug_w):
            if not mask[int(r), int(c)]:
                continue
            dr = int(r + tr)
            dc = int(c + dc_offset)
            if 0 <= dr < base_h and 0 <= dc < base_w and int(base_grid[int(dr), int(dc)]) != int(bg):
                active_zone = (
                    ((int(r), int(c)) in active["pts"] and int(base_grid[int(dr), int(dc)]) == int(obj))
                    or ((int(dr), int(dc)) in active_base["pts"])
                )
                if not active_zone:
                    raise ValueError("plug-overlaps-base-away-from-seam")

    plug_grid = np.full(mask.shape, bg, dtype=int)
    plug_grid[mask] = obj
    _paint_segment(plug_grid, active, match_color)
    _paint_segment(plug_grid, distractor, extra_color)
    return plug_grid


def _gen_attempt() -> tuple[np.ndarray, dict] | None:
    bg, obj = 1, 4
    palette = [2, 3, 6, 7, 8, 9]
    match_color, base_extra_color, plug_extra_color = _rng.sample(palette, 3)
    conn_len = _rng.randint(1, 3)

    try:
        base_grid, active_base = _build_base(conn_len, match_color, base_extra_color)
        plug_grid = _build_plug(conn_len, match_color, plug_extra_color, active_base, base_grid)
    except ValueError:
        return None

    base_sprite = _crop_nonbg(base_grid, bg)
    plug_sprite = _crop_nonbg(plug_grid, bg)
    if base_sprite.size == 0 or plug_sprite.size == 0:
        return None
    if int(np.sum(base_sprite != bg)) < 5 or int(np.sum(plug_sprite != bg)) < 5:
        return None

    try:
        expected = _assemble_from_sprites(base_sprite, plug_sprite, bg=bg, obj=obj, match_color=match_color)
    except Exception:
        return None
    if expected.size == 0 or not np.any(expected != bg):
        return None

    bh, bw = base_sprite.shape
    ph, pw = plug_sprite.shape
    canvas_h = _rng.randint(max(7, max(bh, ph) + 2), 11)
    canvas_w = _rng.randint(max(4, max(bw, pw) + 1), 13)
    if bh > canvas_h or bw > canvas_w or ph > canvas_h or pw > canvas_w:
        return None

    canvas = np.full((canvas_h, canvas_w), bg, dtype=int)
    paste(canvas, base_sprite, 0, 0, transparent=bg)
    try:
        pr, pc = random_free_location(canvas, plug_sprite, transparent=bg, padding=0)
    except ValueError:
        return None
    if int(pr + pc) == 0:
        return None
    paste(canvas, plug_sprite, int(pr), int(pc), transparent=bg)

    if np.sum(canvas == bg) <= canvas.size // 2:
        return None

    try:
        actual = main(canvas)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    info = {
        "match_color": int(match_color),
        "base_colors": sorted(int(x) for x in set(np.unique(base_sprite)) - {bg, obj}),
        "plug_colors": sorted(int(x) for x in set(np.unique(plug_sprite)) - {bg, obj}),
        "conn_len": int(conn_len),
    }
    return canvas, info


def generate_input() -> np.ndarray:
    for _ in range(1500):
        out = _gen_attempt()
        if out is not None:
            return out[0]
    raise RuntimeError("generate_input: failed after 1500 attempts")
