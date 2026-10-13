from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
FRAME = 5
FRAGMENT = 7
INNER_COLORS = (1, 2, 3, 4, 6, 8, 9)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 400


def neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def most_common_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(cnts)])


def components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            seen[r0, c0] = True
            stack = [(r0, c0)]
            comp: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                comp.append((r, c))
                for nr, nc in neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(mask)
    assert len(pts) > 0
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_component(coords: list[tuple[int, int]]) -> tuple[tuple[int, int, int, int], np.ndarray]:
    r0, c0, r1, c1 = bbox_from_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return (r0, c0, r1, c1), mask


def all_orientations(mask: np.ndarray) -> list[np.ndarray]:
    seen: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}
    for flipped in (mask, np.fliplr(mask)):
        for k in range(4):
            cur = np.rot90(flipped, k)
            key = (tuple(map(int, cur.shape)), cur.tobytes())
            if key not in seen:
                seen[key] = cur.copy()
    return list(seen.values())


def canonical_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min((tuple(map(int, cur.shape)), cur.tobytes()) for cur in all_orientations(mask))


def perimeter_cycle(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    pts: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        pts.append((r0, c))
    for r in range(r0 + 1, r1):
        pts.append((r, c1))
    for c in range(c1, c0 - 1, -1):
        pts.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        pts.append((r, c0))
    return pts


def side_contact_groups(
    coords: list[tuple[int, int]], frame_bbox: tuple[int, int, int, int]
) -> tuple[dict[str, list[int]], list[tuple[int, int]]]:
    r0, c0, r1, c1 = frame_bbox
    groups = {"top": [], "bottom": [], "left": [], "right": []}
    contacts: list[tuple[int, int]] = []
    for r, c in coords:
        if r == r0 and c0 < c < c1:
            groups["top"].append(c)
            contacts.append((r + 1, c))
        elif r == r1 and c0 < c < c1:
            groups["bottom"].append(c)
            contacts.append((r - 1, c))
        elif c == c0 and r0 < r < r1:
            groups["left"].append(r)
            contacts.append((r, c + 1))
        elif c == c1 and r0 < r < r1:
            groups["right"].append(r)
            contacts.append((r, c - 1))
    out = {side: sorted(vals) for side, vals in groups.items() if vals}
    return out, contacts


def rect_cells(
    frame_bbox: tuple[int, int, int, int], side: str, start: int, length: int, depth: int
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame_bbox
    cells: list[tuple[int, int]] = []
    if side == "top":
        for r in range(r0 + 1, r0 + 1 + depth):
            for c in range(start, start + length):
                cells.append((r, c))
    elif side == "bottom":
        for r in range(r1 - depth, r1):
            for c in range(start, start + length):
                cells.append((r, c))
    elif side == "left":
        for r in range(start, start + length):
            for c in range(c0 + 1, c0 + 1 + depth):
                cells.append((r, c))
    else:
        for r in range(start, start + length):
            for c in range(c1 - depth, c1):
                cells.append((r, c))
    return cells


def placement_options(
    frame_bbox: tuple[int, int, int, int],
    groups: dict[str, list[int]],
    used_inside: set[tuple[int, int]],
    contact_len: int,
) -> list[dict[str, object]]:
    r0, c0, r1, c1 = frame_bbox
    max_depth = {
        "top": r1 - r0 - 1,
        "bottom": r1 - r0 - 1,
        "left": c1 - c0 - 1,
        "right": c1 - c0 - 1,
    }
    out: list[dict[str, object]] = []
    for side, positions in groups.items():
        if len(positions) < contact_len:
            continue
        for i in range(len(positions) - contact_len + 1):
            segment = positions[i : i + contact_len]
            if segment[-1] - segment[0] != contact_len - 1:
                continue
            for depth in (1, 2, 3):
                area = contact_len * depth
                if not (2 <= area <= 6):
                    continue
                if depth > max_depth[side]:
                    continue
                cells = rect_cells(frame_bbox, side, segment[0], contact_len, depth)
                if any(cell in used_inside for cell in cells):
                    continue
                out.append(
                    {
                        "side": side,
                        "contact_keys": {(side, pos) for pos in segment},
                        "cells": cells,
                    }
                )
    return out


def expand_cells(cells: set[tuple[int, int]], h: int, w: int, pad: int) -> set[tuple[int, int]]:
    out = set(cells)
    for r, c in list(cells):
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def build_notch_candidates(frame_bbox: tuple[int, int, int, int]) -> list[dict[str, object]]:
    r0, c0, r1, c1 = frame_bbox
    perim = perimeter_cycle(r0, c0, r1, c1)
    plen = len(perim)
    out: list[dict[str, object]] = []
    for length in range(2, 7):
        for start in range(plen):
            idxs = [(start + i) % plen for i in range(length)]
            coords = [perim[idx] for idx in idxs]
            dirs = [
                (coords[i + 1][0] - coords[i][0], coords[i + 1][1] - coords[i][1])
                for i in range(len(coords) - 1)
            ]
            if any(abs(dr) + abs(dc) != 1 for dr, dc in dirs):
                continue
            turns = sum(dirs[i] != dirs[i - 1] for i in range(1, len(dirs)))
            if turns > 1:
                continue
            groups, contacts = side_contact_groups(coords, frame_bbox)
            if not contacts:
                continue
            bbox_local, mask = crop_component(coords)
            out.append(
                {
                    "indices": tuple(idxs),
                    "coords": coords,
                    "bbox": bbox_local,
                    "mask": mask,
                    "key": canonical_key(mask),
                    "groups": groups,
                    "contact_count": len(contacts),
                }
            )
    return out


def supports_multi(candidate: dict[str, object]) -> bool:
    groups = candidate["groups"]
    lengths = sorted((len(v) for v in groups.values()), reverse=True)
    total = sum(lengths)
    if total < 3:
        return False
    return bool(lengths and (lengths[0] >= 3 or (lengths[0] >= 2 and total - lengths[0] >= 1)))


def select_branches(rng: random.Random) -> list[str]:
    n = rng.choices([2, 3, 4], weights=[1, 3, 2])[0]
    if n == 2:
        branches = list(rng.choice([("none", "single"), ("none", "multi"), ("single", "multi")]))
    elif n == 3:
        branches = ["none", "single", "multi"]
    else:
        branches = ["none", "single", "multi", rng.choice(["none", "single", "multi"])]
    rng.shuffle(branches)
    return branches


def select_notches(
    rng: random.Random, frame_bbox: tuple[int, int, int, int], branches: list[str]
) -> list[dict[str, object]] | None:
    candidates = build_notch_candidates(frame_bbox)
    plen = 2 * ((frame_bbox[2] - frame_bbox[0] + 1) + (frame_bbox[3] - frame_bbox[1] + 1)) - 4
    blocked: set[int] = set()
    used_keys: set[tuple[tuple[int, int], bytes]] = set()
    selected: list[dict[str, object]] = []
    order = sorted(branches, key=lambda x: {"multi": 0, "single": 1, "none": 2}[x])
    for branch in order:
        def suitable(cand: dict[str, object]) -> bool:
            if any(idx in blocked for idx in cand["indices"]):
                return False
            if branch == "multi":
                return supports_multi(cand)
            return True

        pool = [cand for cand in candidates if cand["key"] not in used_keys and suitable(cand)]
        if not pool:
            pool = [cand for cand in candidates if suitable(cand)]
        if not pool:
            return None
        rng.shuffle(pool)
        pool.sort(key=lambda cand: (-int(cand["contact_count"]), len(cand["indices"])))
        cand = rng.choice(pool[: min(10, len(pool))])
        chosen = dict(cand)
        chosen["branch"] = branch
        selected.append(chosen)
        used_keys.add(cand["key"])
        for idx in cand["indices"]:
            for delta in range(-2, 3):
                blocked.add((idx + delta) % plen)
    rng.shuffle(selected)
    return selected


def paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def place_target_blocks(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    notch: dict[str, object],
    used_inside: set[tuple[int, int]],
    rng: random.Random,
) -> bool:
    branch = notch["branch"]
    groups = notch["groups"]
    if branch == "none":
        return True

    colors = list(INNER_COLORS)
    rng.shuffle(colors)

    if branch == "single":
        options: list[dict[str, object]] = []
        for contact_len in (3, 2, 1):
            options.extend(placement_options(frame_bbox, groups, used_inside, contact_len))
        if not options:
            return False
        choice = rng.choice(options)
        paint_cells(grid, choice["cells"], colors[0])
        used_inside.update(choice["cells"])
        return True

    major_opts = placement_options(frame_bbox, groups, used_inside, 3)
    major_opts += placement_options(frame_bbox, groups, used_inside, 2)
    minor_opts = placement_options(frame_bbox, groups, used_inside, 1)
    rng.shuffle(major_opts)
    rng.shuffle(minor_opts)
    for major in major_opts:
        major_cells = set(major["cells"])
        for minor in minor_opts:
            if major["contact_keys"] & minor["contact_keys"]:
                continue
            if major_cells & set(minor["cells"]):
                continue
            paint_cells(grid, major["cells"], colors[0])
            paint_cells(grid, minor["cells"], colors[1])
            used_inside.update(major["cells"])
            used_inside.update(minor["cells"])
            return True
    return False


def add_extra_blocks(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    used_inside: set[tuple[int, int]],
    rng: random.Random,
) -> None:
    h, w = grid.shape
    r0, c0, r1, c1 = frame_bbox
    dims = [(1, 2), (2, 1), (1, 3), (3, 1), (2, 2), (1, 4), (4, 1), (2, 3), (3, 2)]
    blocked = expand_cells(set(used_inside), h, w, 1)
    extra_count = rng.randint(0, 2)
    for _ in range(extra_count):
        rng.shuffle(dims)
        placed = False
        for hh, ww in dims:
            if r1 - r0 - 3 < hh or c1 - c0 - 3 < ww:
                continue
            for _ in range(30):
                top = rng.randint(r0 + 2, r1 - hh)
                left = rng.randint(c0 + 2, c1 - ww)
                cells = [(r, c) for r in range(top, top + hh) for c in range(left, left + ww)]
                if any(cell in blocked for cell in cells):
                    continue
                color = rng.choice(INNER_COLORS)
                paint_cells(grid, cells, color)
                used_inside.update(cells)
                blocked = expand_cells(set(used_inside), h, w, 1)
                placed = True
                break
            if placed:
                break


def mask_cells(mask: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    pts = np.argwhere(mask)
    return [(int(top + r), int(left + c)) for r, c in pts]


def can_place_fragment(
    grid: np.ndarray,
    cells: list[tuple[int, int]],
    frame_bbox: tuple[int, int, int, int],
) -> bool:
    h, w = grid.shape
    r0, c0, r1, c1 = frame_bbox
    cell_set = set(cells)
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if r0 - 1 <= r <= r1 + 1 and c0 - 1 <= c <= c1 + 1:
            return False
        if grid[r, c] != BG:
            return False
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if not (0 <= rr < h and 0 <= cc < w):
                    continue
                if (rr, cc) in cell_set:
                    continue
                if grid[rr, cc] != BG:
                    return False
    return True


def place_fragments(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    notches: list[dict[str, object]],
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    for notch in notches:
        base_mask = notch["mask"]
        variants = all_orientations(base_mask)
        rng.shuffle(variants)
        if len(variants) > 1 and rng.random() < 0.7:
            variant = rng.choice(variants[1:])
        else:
            variant = rng.choice(variants)
        ph, pw = map(int, variant.shape)
        placed = False
        for _ in range(200):
            top = rng.randint(0, h - ph)
            left = rng.randint(0, w - pw)
            cells = mask_cells(variant, top, left)
            if not can_place_fragment(grid, cells, frame_bbox):
                continue
            paint_cells(grid, cells, FRAGMENT)
            placed = True
            break
        if not placed:
            return False
    return True


def draw_frame(grid: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = frame_bbox
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME


def color_vote(
    grid: np.ndarray,
    notch_cells: list[tuple[int, int]],
    frame_bbox: tuple[int, int, int, int],
    bg: int,
) -> int:
    r0, c0, r1, c1 = frame_bbox
    counts: Counter[int] = Counter()
    for r, c in notch_cells:
        for nr, nc in neighbors4(r, c):
            if not (r0 < nr < r1 and c0 < nc < c1):
                continue
            color = int(grid[nr, nc])
            if color in (bg, FRAME, FRAGMENT):
                continue
            counts[color] += 1
    if not counts:
        return FRAME
    return max(counts.items(), key=lambda item: (item[1], -item[0]))[0]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = most_common_color(g)

    frame_mask = g == FRAME
    assert frame_mask.any()
    frame_bbox = bbox_from_mask(frame_mask)
    r0, c0, r1, c1 = frame_bbox

    border = np.zeros_like(frame_mask)
    border[r0, c0 : c1 + 1] = True
    border[r1, c0 : c1 + 1] = True
    border[r0 : r1 + 1, c0] = True
    border[r0 : r1 + 1, c1] = True

    notch_components = components(border & (g == bg))

    fragment_components = components(g == FRAGMENT)
    fragment_counts: Counter[tuple[tuple[int, int], bytes]] = Counter()
    for comp in fragment_components:
        _, mask = crop_component(comp)
        fragment_counts[canonical_key(mask)] += 1

    out = g.copy()
    out[out == FRAGMENT] = bg

    for comp in notch_components:
        _, mask = crop_component(comp)
        key = canonical_key(mask)
        assert fragment_counts[key] > 0
        fragment_counts[key] -= 1
        fill = color_vote(g, comp, frame_bbox, bg)
        for r, c in comp:
            out[r, c] = fill

    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    frame_h = rng.randint(10, 16)
    frame_w = rng.randint(10, 18)
    top_margin = rng.randint(2, 5)
    bottom_margin = rng.randint(2, 5)
    left_margin = rng.randint(2, 5)
    right_margin = rng.randint(2, 5)
    h = frame_h + top_margin + bottom_margin
    w = frame_w + left_margin + right_margin
    r0 = top_margin
    c0 = left_margin
    frame_bbox = (r0, c0, r0 + frame_h - 1, c0 + frame_w - 1)

    branches = select_branches(rng)
    notches = select_notches(rng, frame_bbox, branches)
    if not notches:
        return None

    grid = np.full((h, w), BG, dtype=int)
    draw_frame(grid, frame_bbox)

    for notch in notches:
        for r, c in notch["coords"]:
            grid[r, c] = BG

    used_inside: set[tuple[int, int]] = set()
    for notch in notches:
        if not place_target_blocks(grid, frame_bbox, notch, used_inside, rng):
            return None

    add_extra_blocks(grid, frame_bbox, used_inside, rng)

    if not place_fragments(grid, frame_bbox, notches, rng):
        return None

    out = main(grid)
    if not isinstance(out, np.ndarray):
        return None
    if np.any(out == FRAGMENT):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
