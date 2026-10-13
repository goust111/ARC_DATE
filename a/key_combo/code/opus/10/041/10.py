from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
BAR = 5
BLUE = 1
RED = 2
PIECE_COLORS = [3, 4, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def find_components(grid: np.ndarray, bg: int | None = None) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    if bg is None:
        bg = get_bg(g)
    out = []
    for color in sorted(set(map(int, g.ravel())) - {int(bg)}):
        for coords in components_mask(g == int(color)):
            out.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return out


def local_mask(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[int(r) - r0, int(c) - c0] = True
    return mask


def mask_cells(mask: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    pts = []
    for r, c in np.argwhere(mask):
        pts.append((top + int(r), left + int(c)))
    return pts


def rotate_side(mask: np.ndarray, side: str) -> np.ndarray:
    if side == "top":
        return mask.copy()
    if side == "bottom":
        return np.flipud(mask)
    if side == "left":
        return np.rot90(mask, 1)
    if side == "right":
        return np.rot90(mask, -1)
    raise ValueError(side)


def open_frame_cells(bbox: tuple[int, int, int, int], open_side: str, gap0: int, gap1: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    pts = []
    for c in range(c0, c1 + 1):
        if not (open_side == "top" and gap0 <= c <= gap1):
            pts.append((r0, c))
        if not (open_side == "bottom" and gap0 <= c <= gap1):
            pts.append((r1, c))
    for r in range(r0 + 1, r1):
        if not (open_side == "left" and gap0 <= r <= gap1):
            pts.append((r, c0))
        if not (open_side == "right" and gap0 <= r <= gap1):
            pts.append((r, c1))
    return pts


def classify_open_frame(comp: dict) -> dict | None:
    r0, c0, r1, c1 = comp["bbox"]
    pts = {(int(r), int(c)) for r, c in comp["coords"]}
    border = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0, r1 + 1):
        border.add((r, c0))
        border.add((r, c1))
    if pts - border:
        return None
    missing = sorted(border - pts)
    if not missing:
        return None
    sides = set()
    for r, c in missing:
        if r == r0:
            sides.add("top")
        elif r == r1:
            sides.add("bottom")
        elif c == c0:
            sides.add("left")
        elif c == c1:
            sides.add("right")
    if len(sides) != 1:
        return None
    side = next(iter(sides))
    if side in ("top", "bottom"):
        vals = sorted(c for r, c in missing)
    else:
        vals = sorted(r for r, c in missing)
    if vals != list(range(vals[0], vals[-1] + 1)):
        return None
    return {"bbox": comp["bbox"], "color": int(comp["color"]), "open_side": side, "gap": (int(vals[0]), int(vals[-1]))}


def detect_bar_and_notches(grid: np.ndarray) -> tuple[dict, list[dict]]:
    bg = get_bg(grid)
    comps = find_components(grid, bg=bg)
    comps.sort(key=lambda item: -item["area"])
    bar = comps[0]
    br0, bc0, br1, bc1 = bar["bbox"]
    orient = "horizontal" if (br1 - br0 + 1) == 2 else "vertical"
    box = grid[br0 : br1 + 1, bc0 : bc1 + 1]
    hole_mask = box == bg
    notches = []
    for coords in components_mask(hole_mask):
        rr = coords[:, 0]
        cc = coords[:, 1]
        side = None
        if np.all(rr == 0):
            side = "top" if orient == "horizontal" else "left"
        elif np.all(rr == box.shape[0] - 1):
            side = "bottom" if orient == "horizontal" else "right"
        elif np.all(cc == 0):
            side = "left"
        elif np.all(cc == box.shape[1] - 1):
            side = "right"
        if side is None:
            continue
        hole = {(br0 + int(r), bc0 + int(c)) for r, c in coords}
        r0, c0, r1, c1 = bbox_coords(np.array(sorted(hole), dtype=int))
        notches.append({"side": side, "hole": hole, "bbox": (r0, c0, r1, c1), "orient": orient})
    notches.sort(key=lambda item: (item["side"], item["bbox"]))
    return bar, notches


def strict_depth(notch: dict, bar: dict, frames: list[dict]) -> int:
    br0, bc0, br1, bc1 = bar["bbox"]
    nr0, nc0, nr1, nc1 = notch["bbox"]
    center_r = (nr0 + nr1) // 2
    center_c = (nc0 + nc1) // 2
    side = notch["side"]
    needed = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}[side]
    depth = 0
    for frame in frames:
        if frame["open_side"] != needed:
            continue
        r0, c0, r1, c1 = frame["bbox"]
        g0, g1 = frame["gap"]
        if side == "top" and r1 >= br0:
            continue
        if side == "bottom" and r0 <= br1:
            continue
        if side == "left" and c1 >= bc0:
            continue
        if side == "right" and c0 <= bc1:
            continue
        if side in ("top", "bottom"):
            if not (g0 <= center_c <= g1):
                continue
        else:
            if not (g0 <= center_r <= g1):
                continue
        depth += 1
    return depth


def strip_bounds(notch: dict, width: int) -> tuple[int, int]:
    nr0, nc0, nr1, nc1 = notch["bbox"]
    if notch["side"] in ("top", "bottom"):
        start = (nc0 + nc1 - width + 1) // 2
    else:
        start = (nr0 + nr1 - width + 1) // 2
    return int(start), int(start + width - 1)


def inside_bar_cells(cells: list[tuple[int, int]], bar_bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bar_bbox
    return {(r, c) for r, c in cells if r0 <= r <= r1 and c0 <= c <= c1}


def within_strip(cells: list[tuple[int, int]], side: str, strip: tuple[int, int]) -> bool:
    a, b = strip
    if side in ("top", "bottom"):
        return all(a <= c <= b for _, c in cells)
    return all(a <= r <= b for r, _ in cells)


def extends_outward(cells: list[tuple[int, int]], bbox: tuple[int, int, int, int], side: str) -> bool:
    r0, c0, r1, c1 = bbox
    if side == "top":
        return min(r for r, _ in cells) < r0
    if side == "bottom":
        return max(r for r, _ in cells) > r1
    if side == "left":
        return min(c for _, c in cells) < c0
    return max(c for _, c in cells) > c1


def extra_cells_outward(cells: list[tuple[int, int]], bbox: tuple[int, int, int, int], side: str) -> bool:
    r0, c0, r1, c1 = bbox
    extras = []
    for r, c in cells:
        inside = r0 <= r <= r1 and c0 <= c <= c1
        if not inside:
            extras.append((r, c))
    if not extras:
        return False
    if side == "top":
        return all(r < r0 for r, _ in extras)
    if side == "bottom":
        return all(r > r1 for r, _ in extras)
    if side == "left":
        return all(c < c0 for _, c in extras)
    return all(c > c1 for _, c in extras)


def piece_variants(grid: np.ndarray, bar: dict, frames: list[dict]) -> list[dict]:
    bg = get_bg(grid)
    skip = {int(bar["color"]), BLUE, RED}
    out = []
    for comp in find_components(grid, bg=bg):
        if int(comp["color"]) in skip:
            continue
        out.append(
            {
                "color": int(comp["color"]),
                "bbox": comp["bbox"],
                "mask": local_mask(comp["coords"], comp["bbox"]),
                "coords": [(int(r), int(c)) for r, c in np.argwhere(local_mask(comp["coords"], comp["bbox"]))],
            }
        )
    return out


def enumerate_root_placements(piece: dict, notch: dict, bar: dict, strip: tuple[int, int], occ: np.ndarray):
    h, w = occ.shape
    hole = notch["hole"]
    bar_bbox = bar["bbox"]
    piece_local = piece["coords"]
    seen = set()
    for hr, hc in hole:
        for lr, lc in piece_local:
            top = hr - lr
            left = hc - lc
            key = (top, left)
            if key in seen:
                continue
            seen.add(key)
            if top < 0 or left < 0 or top + piece["mask"].shape[0] > h or left + piece["mask"].shape[1] > w:
                continue
            cells = mask_cells(piece["mask"], top, left)
            if not within_strip(cells, notch["side"], strip):
                continue
            if any(occ[r, c] for r, c in cells):
                continue
            if inside_bar_cells(cells, bar_bbox) != hole:
                continue
            if not extends_outward(cells, bar_bbox, notch["side"]):
                continue
            yield {"top": top, "left": left, "cells": cells, "bbox": (top, left, top + piece["mask"].shape[0] - 1, left + piece["mask"].shape[1] - 1)}


def enumerate_next_placements(piece: dict, side: str, strip: tuple[int, int], head_bbox: tuple[int, int, int, int], cavity: set[tuple[int, int]], occ: np.ndarray):
    h, w = occ.shape
    piece_local = piece["coords"]
    seen = set()
    for cr, cc in cavity:
        for lr, lc in piece_local:
            top = cr - lr
            left = cc - lc
            key = (top, left)
            if key in seen:
                continue
            seen.add(key)
            if top < 0 or left < 0 or top + piece["mask"].shape[0] > h or left + piece["mask"].shape[1] > w:
                continue
            cells = mask_cells(piece["mask"], top, left)
            if not within_strip(cells, side, strip):
                continue
            if any(occ[r, c] for r, c in cells):
                continue
            inside = {(r, c) for r, c in cells if head_bbox[0] <= r <= head_bbox[2] and head_bbox[1] <= c <= head_bbox[3]}
            if inside != cavity:
                continue
            if not extra_cells_outward(cells, head_bbox, side):
                continue
            yield {"top": top, "left": left, "cells": cells, "bbox": (top, left, top + piece["mask"].shape[0] - 1, left + piece["mask"].shape[1] - 1)}


def cavity_cells(occ: np.ndarray, bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    out = set()
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if not occ[r, c]:
                out.add((r, c))
    return out


def solve_chain(notch: dict, piece_ids: list[int], pieces: list[dict], occ: np.ndarray) -> list[dict]:
    width = strict_depth(notch, notch["bar"], notch["frames"]) + 1
    strip = strip_bounds(notch, width)
    candidates = []

    def extend(head_bbox, color, remaining_ids, occ_now: np.ndarray):
        cav = cavity_cells(occ_now, head_bbox)
        if not cav:
            return [[]]
        out = []
        for idx in remaining_ids:
            piece = pieces[idx]
            if piece["color"] != color:
                continue
            for place in enumerate_next_placements(piece, notch["side"], strip, head_bbox, cav, occ_now):
                occ2 = occ_now.copy()
                for r, c in place["cells"]:
                    occ2[r, c] = True
                tails = extend(place["bbox"], color, [x for x in remaining_ids if x != idx], occ2)
                for tail in tails:
                    out.append([{"idx": idx, "cells": place["cells"], "color": color, "bbox": place["bbox"]}] + tail)
        return out

    for idx in piece_ids:
        piece = pieces[idx]
        for root in enumerate_root_placements(piece, notch, notch["bar"], strip, occ):
            occ2 = occ.copy()
            for r, c in root["cells"]:
                occ2[r, c] = True
            chains = extend(root["bbox"], piece["color"], [x for x in piece_ids if x != idx], occ2)
            if not chains:
                if not cavity_cells(occ2, root["bbox"]):
                    chains = [[]]
                else:
                    continue
            for tail in chains:
                placed = [{"idx": idx, "cells": root["cells"], "color": piece["color"], "bbox": root["bbox"]}] + tail
                score = (len(placed), sum(len(step["cells"]) for step in placed))
                candidates.append((score, placed))
    if not candidates:
        return []
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each notch-matched chain with a strip width determined by the notch's containment depth.

    Input structure:
    - The background is black. A central grey solid bar of thickness 2 has 2-3 boundary notches.
    - Around some notches sit one or two open rectangular frames whose opening faces the bar.
    - Several monochrome sprites are scattered away from the bar; only some of them are used.

    Rule:
    - Find every bar notch and count how many aligned open frames strictly contain that notch's channel: depth 0, 1, or 2.
    - Convert depth to strip width with `width = depth + 1`.
    - For each notch, move the unique root sprite whose in-bar cells exactly fill the notch, then keep extending the same-color chain outward: each next sprite must fill exactly the background cavity inside the previous sprite's bbox while staying inside the notch's strip.
    - Process deeper notches first, keep the bar and frames unchanged, and erase every unused scattered sprite.

    Key insight:
    - The local chain-fill step is the same as the bar-notch assembly puzzle, but the strip width is not read from the root sprite: it is read from the notch's global containment depth inside the nested open frames.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    bar, notches = detect_bar_and_notches(g)
    frame_comps = [classify_open_frame(comp) for comp in find_components(g, bg=bg) if int(comp["color"]) in (BLUE, RED)]
    frames = [frame for frame in frame_comps if frame is not None]
    for notch in notches:
        notch["bar"] = bar
        notch["frames"] = frames
        notch["depth"] = strict_depth(notch, bar, frames)
    notches.sort(key=lambda item: (-item["depth"], item["side"], item["bbox"]))

    pieces = piece_variants(g, bar, frames)
    remaining = list(range(len(pieces)))
    occ = np.zeros((h, w), dtype=bool)
    fixed = np.zeros((h, w), dtype=int)
    for r, c in bar["coords"]:
        occ[int(r), int(c)] = True
        fixed[int(r), int(c)] = int(bar["color"])
    for frame in frames:
        for r, c in open_frame_cells(frame["bbox"], frame["open_side"], frame["gap"][0], frame["gap"][1]):
            occ[r, c] = True
            fixed[r, c] = int(frame["color"])

    placed_steps: list[dict] = []
    for notch in notches:
        chain = solve_chain(notch, remaining, pieces, occ)
        if not chain:
            continue
        for step in chain:
            placed_steps.append(step)
            remaining.remove(step["idx"])
            for r, c in step["cells"]:
                occ[r, c] = True

    out = fixed.copy()
    out[out == 0] = bg
    for step in placed_steps:
        for r, c in step["cells"]:
            out[r, c] = int(step["color"])
    return out


def canonical_root(width: int) -> np.ndarray:
    if width == 1:
        return np.array([[1], [1]], dtype=bool)
    if width == 2:
        return np.array([[1, 0], [1, 0], [1, 1]], dtype=bool)
    return np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=bool)


def canonical_tail(width: int) -> np.ndarray | None:
    if width == 1:
        return None
    if width == 2:
        return np.array([[1], [1], [1]], dtype=bool)
    return np.array([[1], [1], [1], [1]], dtype=bool)


def chain_output_positions(side: str, bar_bbox: tuple[int, int, int, int], strip: tuple[int, int], width: int) -> list[tuple[np.ndarray, int, int]]:
    br0, bc0, br1, bc1 = bar_bbox
    root = rotate_side(canonical_root(width), side)
    tail0 = canonical_tail(width)
    tail = None if tail0 is None else rotate_side(tail0, side)
    if side == "top":
        rt = br0 - root.shape[0] + 1
        rl = strip[0]
        out = [(root, rt, rl)]
        if tail is not None:
            out.append((tail, rt - 1, rl + width // 2))
        return out
    if side == "bottom":
        rt = br1
        rl = strip[0]
        out = [(root, rt, rl)]
        if tail is not None:
            out.append((tail, rt + root.shape[0] - 2, rl + width // 2))
        return out
    if side == "left":
        rt = strip[0]
        rl = bc0 - root.shape[1] + 1
        out = [(root, rt, rl)]
        if tail is not None:
            out.append((tail, rt + width // 2, rl - 1))
        return out
    rt = strip[0]
    rl = bc1
    out = [(root, rt, rl)]
    if tail is not None:
        out.append((tail, rt + width // 2, rl + root.shape[1] - 2))
    return out


def paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> list[tuple[int, int]]:
    cells = []
    for r, c in np.argwhere(mask):
        rr = top + int(r)
        cc = left + int(c)
        grid[rr, cc] = int(color)
        cells.append((rr, cc))
    return cells


def fits_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, bg: int = BG, pad: int = 1) -> bool:
    h, w = grid.shape
    mh, mw = mask.shape
    if top < 0 or left < 0 or top + mh > h or left + mw > w:
        return False
    for r, c in np.argwhere(mask):
        rr = top + int(r)
        cc = left + int(c)
        if grid[rr, cc] != bg:
            return False
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] != bg and (dr != 0 or dc != 0):
                    return False
    return True


def random_place_mask(grid: np.ndarray, mask: np.ndarray, rng: random.Random, bg: int = BG, pad: int = 1) -> tuple[int, int] | None:
    h, w = grid.shape
    mh, mw = mask.shape
    spots = [(r, c) for r in range(h - mh + 1) for c in range(w - mw + 1)]
    rng.shuffle(spots)
    for top, left in spots:
        if fits_mask(grid, mask, top, left, bg=bg, pad=pad):
            return top, left
    return None


def add_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], open_side: str, gap: tuple[int, int], color: int) -> None:
    for r, c in open_frame_cells(bbox, open_side, gap[0], gap[1]):
        grid[r, c] = int(color)


def frame_specs_for_notch(side: str, strip: tuple[int, int], depth: int, bar_bbox: tuple[int, int, int, int]) -> list[tuple[tuple[int, int, int, int], str, tuple[int, int], int]]:
    br0, bc0, br1, bc1 = bar_bbox
    a, b = strip
    frames = []
    if depth >= 1:
        if side == "top":
            frames.append(((br0 - 4, a - 1, br0 - 1, b + 1), "bottom", (a, b), BLUE))
        elif side == "bottom":
            frames.append(((br1 + 1, a - 1, br1 + 4, b + 1), "top", (a, b), BLUE))
        elif side == "left":
            frames.append(((a - 1, bc0 - 4, b + 1, bc0 - 1), "right", (a, b), BLUE))
        else:
            frames.append(((a - 1, bc1 + 1, b + 1, bc1 + 4), "left", (a, b), BLUE))
    if depth >= 2:
        if side == "top":
            frames.append(((br0 - 6, a - 1, br0 - 3, b + 1), "bottom", (a, b), RED))
        elif side == "bottom":
            frames.append(((br1 + 3, a - 1, br1 + 6, b + 1), "top", (a, b), RED))
        elif side == "left":
            frames.append(((a - 1, bc0 - 6, b + 1, bc0 - 3), "right", (a, b), RED))
        else:
            frames.append(((a - 1, bc1 + 3, b + 1, bc1 + 6), "left", (a, b), RED))
    return frames


def random_decoy_mask(rng: random.Random) -> np.ndarray:
    shapes = [
        np.array([[1, 1], [1, 0]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
        np.array([[1, 0], [1, 1], [0, 1]], dtype=bool),
        np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    ]
    return shapes[rng.randrange(len(shapes))].copy()


def _gen_attempt(rng: random.Random) -> tuple[np.ndarray, np.ndarray, list[int]] | None:
    orient = rng.choice(["horizontal", "vertical"])
    n_notches = rng.choice([2, 3])
    depths = [0, 1, 2]
    rng.shuffle(depths)
    depths = depths[:n_notches]
    if n_notches == 2 and rng.random() < 0.35:
        depths[1] = depths[0]
    if orient == "horizontal":
        h = rng.randint(20, 26)
        w = rng.randint(20, 28)
        length = rng.randint(12, 14)
        br0 = h // 2 - 1
        bc0 = (w - length) // 2
        bar_bbox = (br0, bc0, br0 + 1, bc0 + length - 1)
        sides = [rng.choice(["top", "bottom"]) for _ in range(n_notches)]
        if n_notches >= 2 and len(set(sides)) == 1 and rng.random() < 0.7:
            sides[0] = "top"
            sides[1] = "bottom"
        available = list(range(bc0 + 1, bc0 + length - 2))
    else:
        h = rng.randint(20, 28)
        w = rng.randint(20, 26)
        length = rng.randint(12, 14)
        bc0 = w // 2 - 1
        br0 = (h - length) // 2
        bar_bbox = (br0, bc0, br0 + length - 1, bc0 + 1)
        sides = [rng.choice(["left", "right"]) for _ in range(n_notches)]
        if n_notches >= 2 and len(set(sides)) == 1 and rng.random() < 0.7:
            sides[0] = "left"
            sides[1] = "right"
        available = list(range(br0 + 1, br0 + length - 2))

    notch_specs = []
    used_ranges: dict[str, list[tuple[int, int]]] = {"top": [], "bottom": [], "left": [], "right": []}
    for side, depth in zip(sides, depths):
        width = depth + 1
        choices = available[:]
        rng.shuffle(choices)
        chosen = None
        for start in choices:
            end = start + width - 1
            if orient == "horizontal" and end > bar_bbox[3]:
                continue
            if orient == "vertical" and end > bar_bbox[2]:
                continue
            pad = 2
            lo = start - pad
            hi = end + pad
            if any(not (hi < a or lo > b) for a, b in used_ranges[side]):
                continue
            chosen = (start, end)
            break
        if chosen is None:
            return None
        used_ranges[side].append(chosen)
        notch_specs.append({"side": side, "depth": depth, "width": width, "strip": chosen})

    out = np.full((h, w), BG, dtype=int)
    br0, bc0, br1, bc1 = bar_bbox
    out[br0 : br1 + 1, bc0 : bc1 + 1] = BAR
    piece_patches: list[tuple[np.ndarray, int]] = []
    semantic_depths = []

    for idx, spec in enumerate(notch_specs):
        side = spec["side"]
        depth = spec["depth"]
        width = spec["width"]
        strip = spec["strip"]
        color = PIECE_COLORS[idx]
        semantic_depths.append(depth)
        if side == "top":
            out[br0, strip[0] : strip[1] + 1] = BG
        elif side == "bottom":
            out[br1, strip[0] : strip[1] + 1] = BG
        elif side == "left":
            out[strip[0] : strip[1] + 1, bc0] = BG
        else:
            out[strip[0] : strip[1] + 1, bc1] = BG
        for bbox, open_side, gap, frame_color in frame_specs_for_notch(side, strip, depth, bar_bbox):
            r0, c0, r1, c1 = bbox
            if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
                return None
            add_frame(out, bbox, open_side, gap, frame_color)
        for patch_mask, top, left in chain_output_positions(side, bar_bbox, strip, width):
            paint_mask(out, patch_mask, top, left, color)
            piece_patches.append((patch_mask.astype(int) * color, color))

    inp = np.full((h, w), BG, dtype=int)
    inp[br0 : br1 + 1, bc0 : bc1 + 1] = BAR
    for spec in notch_specs:
        strip = spec["strip"]
        side = spec["side"]
        if side == "top":
            inp[br0, strip[0] : strip[1] + 1] = BG
        elif side == "bottom":
            inp[br1, strip[0] : strip[1] + 1] = BG
        elif side == "left":
            inp[strip[0] : strip[1] + 1, bc0] = BG
        else:
            inp[strip[0] : strip[1] + 1, bc1] = BG
        for bbox, open_side, gap, frame_color in frame_specs_for_notch(side, strip, spec["depth"], bar_bbox):
            add_frame(inp, bbox, open_side, gap, frame_color)

    scatter = inp.copy()
    rng.shuffle(piece_patches)
    for patch, _color in piece_patches:
        pos = random_place_mask(scatter, patch != BG, rng, pad=1)
        if pos is None:
            return None
        paint_mask(scatter, patch != BG, pos[0], pos[1], int(patch.max()))

    for _ in range(rng.randint(1, 3)):
        decoy_mask = random_decoy_mask(rng)
        decoy_color = rng.choice(PIECE_COLORS)
        pos = random_place_mask(scatter, decoy_mask, rng, pad=1)
        if pos is None:
            continue
        paint_mask(scatter, decoy_mask, pos[0], pos[1], decoy_color)

    bar, notches = detect_bar_and_notches(scatter)
    frame_comps = [classify_open_frame(comp) for comp in find_components(scatter, bg=BG) if int(comp["color"]) in (BLUE, RED)]
    frames = [frame for frame in frame_comps if frame is not None]
    depths_found = sorted(strict_depth(notch, bar, frames) for notch in notches)
    if depths_found != sorted(semantic_depths):
        return None
    solved = main(scatter)
    if not np.array_equal(solved, out):
        return None
    return scatter, out, semantic_depths


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        result = _gen_attempt(rng)
        if result is None:
            continue
        return result[0]
    raise RuntimeError("failed to generate valid chained-containment input")
