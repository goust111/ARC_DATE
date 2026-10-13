from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
FRAME = 1
MARKER = 2
TRAIL = 6
SPRITE_COLORS = (3, 4, 5, 7, 8, 9)

DIRS4 = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

OPPOSITE = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}

SIDE_TO_TRAIL = {
    "top": "up",
    "bottom": "down",
    "left": "left",
    "right": "right",
}

SIDE_TO_PERP = {
    "top": ("left", "right"),
    "bottom": ("left", "right"),
    "left": ("up", "down"),
    "right": ("up", "down"),
}

CAVITY_DIMS = (
    (2, 2),
    (2, 3),
    (2, 4),
    (3, 2),
    (3, 3),
    (3, 4),
)

GEN_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _bbox(coords: list[tuple[int, int]] | np.ndarray) -> tuple[int, int, int, int]:
    arr = np.asarray(coords, dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, mask: np.ndarray) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    view[mask] = patch[mask]


def _largest_non_bg_component(grid: np.ndarray, bg: int) -> dict:
    best: dict | None = None
    for color in sorted(set(map(int, grid.flat)) - {bg}):
        for comp in _components(grid == color):
            cand = {
                "color": int(color),
                "coords": comp,
                "bbox": _bbox(comp),
                "area": len(comp),
            }
            if best is None or cand["area"] > best["area"]:
                best = cand
    assert best is not None
    return best


def _find_cavities(
    grid: np.ndarray, bg: int, frame_color: int, frame_bbox: tuple[int, int, int, int]
) -> tuple[list[dict], np.ndarray]:
    r0, c0, r1, c1 = frame_bbox
    local = grid[r0 : r1 + 1, c0 : c1 + 1]
    h, w = local.shape
    cavity_ids = np.full((h, w), -1, dtype=int)
    cavities: list[dict] = []
    for comp in _components(local == bg):
        lr0, lc0, lr1, lc1 = _bbox(comp)
        if lr0 == 0 or lc0 == 0 or lr1 == h - 1 or lc1 == w - 1:
            continue
        ch = lr1 - lr0 + 1
        cw = lc1 - lc0 + 1
        assert len(comp) == ch * cw, "expected rectangular cavity"
        cavity_id = len(cavities)
        for rr, cc in comp:
            cavity_ids[rr, cc] = cavity_id
        cavities.append(
            {
                "id": cavity_id,
                "bbox": (r0 + lr0, c0 + lc0, r0 + lr1, c0 + lc1),
                "size": (ch, cw),
            }
        )
    return cavities, cavity_ids


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _get_bg(g)

    frame = _largest_non_bg_component(g, bg)
    frame_color = int(frame["color"])
    frame_bbox = tuple(map(int, frame["bbox"]))
    cavities, cavity_ids = _find_cavities(g, bg, frame_color, frame_bbox)

    sprite_mask = ~np.isin(g, [bg, frame_color, MARKER, TRAIL])
    sprite_comps = _components(sprite_mask)
    actors: list[dict] = []
    r0, c0, r1, c1 = frame_bbox

    for comp in sprite_comps:
        sb = _bbox(comp)
        sr0, sc0, sr1, sc1 = sb
        ph = sr1 - sr0 + 1
        pw = sc1 - sc0 + 1
        patch = g[sr0 : sr1 + 1, sc0 : sc1 + 1].copy()
        mask = np.zeros((ph, pw), dtype=bool)
        for rr, cc in comp:
            mask[rr - sr0, cc - sc0] = True

        markers: set[tuple[int, int]] = set()
        for rr, cc in comp:
            for nr, nc in _neighbors4(rr, cc):
                if _in_bounds((h, w), nr, nc) and g[nr, nc] == MARKER:
                    markers.add((nr, nc))
        assert len(markers) == 1, "each sprite must have exactly one adjacent marker"
        mr, mc = next(iter(markers))

        trail_dirs = []
        for name, (dr, dc) in DIRS4.items():
            cells = [(mr + dr * k, mc + dc * k) for k in range(1, 4)]
            if all(_in_bounds((h, w), rr, cc) and g[rr, cc] == TRAIL for rr, cc in cells):
                trail_dirs.append(name)
        assert len(trail_dirs) == 1, "each marker must have exactly one straight trail"
        trail_dir = trail_dirs[0]
        aim_dir = OPPOSITE[trail_dir]
        adr, adc = DIRS4[aim_dir]

        saw_frame = False
        target_id: int | None = None
        ray_cells: list[tuple[int, int]] = []
        rr, cc = mr + adr, mc + adc
        while _in_bounds((h, w), rr, cc):
            ray_cells.append((rr, cc))
            if r0 <= rr <= r1 and c0 <= cc <= c1:
                local_id = int(cavity_ids[rr - r0, cc - c0])
                if local_id != -1:
                    assert saw_frame, "ray reached cavity before entering the frame"
                    target_id = local_id
                    break
                if g[rr, cc] == frame_color:
                    saw_frame = True
            rr += adr
            cc += adc
        assert target_id is not None, "ray must target exactly one cavity"

        actors.append(
            {
                "bbox": sb,
                "size": (ph, pw),
                "coords": comp,
                "patch": patch,
                "mask": mask,
                "marker": (mr, mc),
                "trail_dir": trail_dir,
                "aim_dir": aim_dir,
                "target_id": target_id,
                "trail_cells": [(mr + DIRS4[trail_dir][0] * k, mc + DIRS4[trail_dir][1] * k) for k in range(1, 4)],
                "ray_cells": ray_cells,
            }
        )

    actors.sort(key=lambda a: (a["marker"][0], a["marker"][1]))
    return {
        "grid": g,
        "bg": bg,
        "frame_color": frame_color,
        "frame_bbox": frame_bbox,
        "cavities": cavities,
        "actors": actors,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Aim each trailed sprite at a cavity and erase the sprite if its size does not fit.

    Input structure:
    - A black background contains one large solid template component with several internal rectangular black cavities.
    - Outside the template sit colored 4-connected sprites, each adjacent to one red marker pixel and a straight three-cell pink trail.
    - The trail extends away from the template, so the opposite cardinal direction from the trail points back toward the template.

    Rule:
    - Identify the template and its internal rectangular cavities.
    - For each sprite, read the trail direction, reverse it, and ray-cast from the red marker toward the template.
    - The first cavity reached by that ray is the sprite's target cavity; compare the sprite's bounding-box height and width with the cavity's height and width.
    - If the sizes match exactly, erase the sprite, marker, and trail from their original location and paste the sprite into the cavity with top-left alignment; otherwise erase the sprite, marker, and trail and leave the cavity empty.

    Key insight:
    - The pink trail is not painted into the answer; it is a directional address that selects a specific cavity, and only then does exact size matching decide whether the sprite is embedded or deleted.
    """
    info = _parse_scene(np.asarray(grid, dtype=int))
    out = info["grid"].copy()
    bg = int(info["bg"])

    for actor in info["actors"]:
        for rr, cc in actor["coords"]:
            out[rr, cc] = bg
        mr, mc = actor["marker"]
        out[mr, mc] = bg
        for rr, cc in actor["trail_cells"]:
            out[rr, cc] = bg

    for actor in info["actors"]:
        cavity = info["cavities"][actor["target_id"]]
        if tuple(actor["size"]) == tuple(cavity["size"]):
            cr0, cc0, _cr1, _cc1 = cavity["bbox"]
            _paste(out, actor["patch"], cr0, cc0, actor["mask"])

    return out


def _line_candidates(cavities: dict[str, dict]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    all_items = list(cavities.items())
    for side, cavity in all_items:
        r0, c0, r1, c1 = cavity["local_bbox"]
        vals: list[int] = []
        if side == "top":
            for c in range(c0, c1 + 1):
                if all(other["local_bbox"][0] >= r0 or not (other["local_bbox"][1] <= c <= other["local_bbox"][3])
                       for other_side, other in all_items if other_side != side):
                    vals.append(c)
        elif side == "bottom":
            for c in range(c0, c1 + 1):
                if all(other["local_bbox"][2] <= r1 or not (other["local_bbox"][1] <= c <= other["local_bbox"][3])
                       for other_side, other in all_items if other_side != side):
                    vals.append(c)
        elif side == "left":
            for r in range(r0, r1 + 1):
                if all(other["local_bbox"][1] >= c0 or not (other["local_bbox"][0] <= r <= other["local_bbox"][2])
                       for other_side, other in all_items if other_side != side):
                    vals.append(r)
        else:
            for r in range(r0, r1 + 1):
                if all(other["local_bbox"][3] <= c1 or not (other["local_bbox"][0] <= r <= other["local_bbox"][2])
                       for other_side, other in all_items if other_side != side):
                    vals.append(r)
        out[side] = vals
    return out


def _rects_touch_or_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _place_cavities(frame_h: int, frame_w: int) -> dict[str, dict] | None:
    dims = random.sample(list(CAVITY_DIMS), 4)
    random.shuffle(dims)
    side_to_dim = dict(zip(("top", "bottom", "left", "right"), dims))
    for _ in range(120):
        cavities: dict[str, dict] = {}
        ok = True
        for side in ("top", "bottom", "left", "right"):
            ch, cw = side_to_dim[side]
            row_lo = 1
            row_hi = frame_h - ch - 1
            col_lo = 1
            col_hi = frame_w - cw - 1
            if row_hi < row_lo or col_hi < col_lo:
                return None

            if side == "top":
                row_hi = min(row_hi, max(row_lo, frame_h // 3 - ch))
            elif side == "bottom":
                row_lo = max(row_lo, min(row_hi, (2 * frame_h) // 3 - ch))
            elif side == "left":
                col_hi = min(col_hi, max(col_lo, frame_w // 3 - cw))
            else:
                col_lo = max(col_lo, min(col_hi, (2 * frame_w) // 3 - cw))

            if row_lo > row_hi or col_lo > col_hi:
                ok = False
                break
            rr = random.randint(row_lo, row_hi)
            cc = random.randint(col_lo, col_hi)
            bbox = (rr, cc, rr + ch - 1, cc + cw - 1)
            if any(_rects_touch_or_overlap(bbox, prev["local_bbox"], gap=1) for prev in cavities.values()):
                ok = False
                break
            cavities[side] = {
                "side": side,
                "local_bbox": bbox,
                "size": (ch, cw),
            }

        if not ok:
            continue
        candidates = _line_candidates(cavities)
        if all(candidates[side] for side in cavities):
            top_cols = candidates["top"]
            bottom_cols = candidates["bottom"]
            left_rows = candidates["left"]
            right_rows = candidates["right"]
            if any(tc != bc for tc in top_cols for bc in bottom_cols) and any(lr != rr for lr in left_rows for rr in right_rows):
                return cavities
    return None


def _make_connected_sprite(h: int, w: int) -> np.ndarray:
    area = h * w
    min_cells = max(4, area // 2)
    for _ in range(120):
        target = random.randint(min_cells, area)
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        while len(cells) < target:
            frontier = []
            for rr, cc in cells:
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != target:
            continue
        rows = {r for r, _ in cells}
        cols = {c for _, c in cells}
        if 0 not in rows or h - 1 not in rows or 0 not in cols or w - 1 not in cols:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for rr, cc in cells:
            mask[rr, cc] = True
        return mask
    return np.ones((h, w), dtype=bool)


def _candidate_actor_cells(
    shape: tuple[int, int],
    frame_bbox: tuple[int, int, int, int],
    side: str,
    line: int,
    sprite_mask: np.ndarray,
    gap: int,
    placement_side: str,
) -> tuple[list[tuple[int, int]], tuple[int, int], list[tuple[int, int]], tuple[int, int], np.ndarray] | None:
    grid_h, grid_w = shape
    sprite_h, sprite_w = sprite_mask.shape
    fr0, fc0, fr1, fc1 = frame_bbox
    trail_name = SIDE_TO_TRAIL[side]
    aim_name = OPPOSITE[trail_name]
    tdr, tdc = DIRS4[trail_name]

    if side == "top":
        marker = (fr0 - gap, fc0 + line)
    elif side == "bottom":
        marker = (fr1 + gap, fc0 + line)
    elif side == "left":
        marker = (fr0 + line, fc0 - gap)
    else:
        marker = (fr0 + line, fc1 + gap)

    trail_cells = [(marker[0] + tdr * k, marker[1] + tdc * k) for k in range(1, 4)]
    if not all(_in_bounds(shape, rr, cc) for rr, cc in [marker, *trail_cells]):
        return None

    if placement_side == "left":
        boundary = [
            (rr, cc) for rr in range(sprite_h) for cc in range(sprite_w) if sprite_mask[rr, cc] and cc == sprite_w - 1
        ]
    elif placement_side == "right":
        boundary = [
            (rr, cc) for rr in range(sprite_h) for cc in range(sprite_w) if sprite_mask[rr, cc] and cc == 0
        ]
    elif placement_side == "up":
        boundary = [
            (rr, cc) for rr in range(sprite_h) for cc in range(sprite_w) if sprite_mask[rr, cc] and rr == sprite_h - 1
        ]
    else:
        boundary = [
            (rr, cc) for rr in range(sprite_h) for cc in range(sprite_w) if sprite_mask[rr, cc] and rr == 0
        ]
    random.shuffle(boundary)
    for anchor_r, anchor_c in boundary:
        if placement_side == "left":
            sr0 = marker[0] - anchor_r
            sc0 = marker[1] - sprite_w
        elif placement_side == "right":
            sr0 = marker[0] - anchor_r
            sc0 = marker[1] + 1
        elif placement_side == "up":
            sr0 = marker[0] - sprite_h
            sc0 = marker[1] - anchor_c
        else:
            sr0 = marker[0] + 1
            sc0 = marker[1] - anchor_c

        sprite_cells = []
        ok = True
        for rr in range(sprite_h):
            for cc in range(sprite_w):
                if not sprite_mask[rr, cc]:
                    continue
                gr, gc = sr0 + rr, sc0 + cc
                if not (0 <= gr < grid_h and 0 <= gc < grid_w):
                    ok = False
                    break
                sprite_cells.append((gr, gc))
            if not ok:
                break
        if not ok:
            continue

        adr, adc = DIRS4[aim_name]
        ray_probe = (marker[0] + adr, marker[1] + adc)
        if not _in_bounds(shape, *ray_probe):
            continue
        return sprite_cells, marker, trail_cells, (sr0, sc0), sprite_mask
    return None


def _fits_with_padding(cells: list[tuple[int, int]], occupied: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    cand = set(cells)
    if len(cand) != len(cells):
        return False
    for rr, cc in cand:
        if not _in_bounds(shape, rr, cc):
            return False
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = rr + dr, cc + dc
                if (nr, nc) in occupied and (nr, nc) not in cand:
                    return False
    return True


def _semantic_checks(inp: np.ndarray) -> None:
    info = _parse_scene(inp)
    assert inp.ndim == 2 and max(inp.shape) <= 30
    assert len(info["cavities"]) == 4
    assert len(info["actors"]) == 4
    cavity_sizes = [tuple(c["size"]) for c in info["cavities"]]
    assert len(set(cavity_sizes)) == 4, "cavity sizes must be pairwise distinct"
    target_ids = [int(a["target_id"]) for a in info["actors"]]
    assert len(set(target_ids)) == 4, "each cavity must be targeted by at most one sprite"
    matches = [
        tuple(info["cavities"][a["target_id"]]["size"]) == tuple(a["size"])
        for a in info["actors"]
    ]
    assert sum(matches) == 3, "need exactly three size matches and one mismatch"
    sprite_sizes = [tuple(a["size"]) for a in info["actors"]]
    assert len(set(sprite_sizes)) == 3, "one mismatching sprite should duplicate another sprite size to make rays necessary"
    out = main(inp)
    assert out.shape == inp.shape
    assert not np.array_equal(inp, out)

    filled = 0
    for cavity in info["cavities"]:
        r0, c0, r1, c1 = cavity["bbox"]
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(patch != BG):
            filled += 1
    assert filled == 3


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(22, 28)
    w = random.randint(22, 28)
    frame_h = random.randint(10, 13)
    frame_w = random.randint(12, 16)

    top_margin_max = min(8, h - frame_h - 6)
    left_margin_max = min(8, w - frame_w - 6)
    if top_margin_max < 6 or left_margin_max < 6:
        return None

    fr0 = random.randint(6, top_margin_max)
    fc0 = random.randint(6, left_margin_max)
    fr1 = fr0 + frame_h - 1
    fc1 = fc0 + frame_w - 1
    frame_bbox = (fr0, fc0, fr1, fc1)

    cavities = _place_cavities(frame_h, frame_w)
    if cavities is None:
        return None
    candidates = _line_candidates(cavities)

    top_options = [(t, b) for t in candidates["top"] for b in candidates["bottom"] if t != b]
    side_options = [(l, r) for l in candidates["left"] for r in candidates["right"] if l != r]
    if not top_options or not side_options:
        return None
    top_line, bottom_line = random.choice(top_options)
    left_line, right_line = random.choice(side_options)
    chosen_lines = {
        "top": top_line,
        "bottom": bottom_line,
        "left": left_line,
        "right": right_line,
    }

    grid = np.zeros((h, w), dtype=int)
    grid[fr0 : fr1 + 1, fc0 : fc1 + 1] = FRAME
    for cavity in cavities.values():
        lr0, lc0, lr1, lc1 = cavity["local_bbox"]
        grid[fr0 + lr0 : fr0 + lr1 + 1, fc0 + lc0 : fc0 + lc1 + 1] = BG
        cavity["bbox"] = (fr0 + lr0, fc0 + lc0, fr0 + lr1, fc0 + lc1)

    mismatch_side = random.choice(["top", "bottom", "left", "right"])
    donor_side = random.choice([s for s in ("top", "bottom", "left", "right") if s != mismatch_side])
    sprite_dims = {side: cavities[side]["size"] for side in cavities}
    sprite_dims[mismatch_side] = cavities[donor_side]["size"]

    colors = random.sample(list(SPRITE_COLORS), 4)
    occupied = {(r, c) for r in range(h) for c in range(w) if grid[r, c] != BG}

    for color, side in zip(colors, random.sample(["top", "bottom", "left", "right"], 4)):
        sh, sw = sprite_dims[side]
        sprite_mask = _make_connected_sprite(sh, sw)
        placements = list(SIDE_TO_PERP[side])
        random.shuffle(placements)
        random.shuffle(placements)
        placed = False
        for _ in range(20):
            if placed:
                break
            gap = random.randint(2, 4)
            line = chosen_lines[side]
            random.shuffle(placements)
            for placement_side in placements:
                candidate = _candidate_actor_cells(
                    (h, w), frame_bbox, side, line, sprite_mask, gap, placement_side
                )
                if candidate is None:
                    continue
                sprite_cells, marker, trail_cells, (sr0, sc0), mask = candidate
                all_cells = sprite_cells + [marker] + trail_cells
                if not _fits_with_padding(all_cells, occupied, (h, w)):
                    continue
                for rr, cc in sprite_cells:
                    grid[rr, cc] = color
                grid[marker] = MARKER
                for rr, cc in trail_cells:
                    grid[rr, cc] = TRAIL
                occupied.update(all_cells)
                placed = True
                break
            if not placed:
                sprite_mask = _make_connected_sprite(sh, sw)
        if not placed:
            return None

    try:
        _semantic_checks(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate valid input")
