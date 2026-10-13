from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_ATTEMPTS = 600


def _background(grid: np.ndarray) -> int:
    return int(Counter(int(v) for v in grid.ravel()).most_common(1)[0][0])


def _non_bg_components(grid: np.ndarray, bg: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == int(bg):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    rr, cc2 = int(cr + dr), int(cc + dc)
                    if 0 <= rr < h and 0 <= cc2 < w and not seen[rr, cc2]:
                        if int(grid[rr, cc2]) != int(bg):
                            seen[rr, cc2] = True
                            q.append((rr, cc2))
            comps.append(comp)
    return comps


def _bbox(cells: set[tuple[int, int]] | list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _c in cells]
    cs = [c for _r, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp each colored object onto its matching isolated markers using the object's odd pixel as the alignment point.

    Input structure:
    - bg = most common color.
    - Several medium 4-connected stamp objects are present; each stamp is one body color plus exactly one differently colored pixel inside it.
    - Elsewhere, isolated singleton marker pixels have the same color as one stamp body.

    Rule:
    - For every stamp, record all cells in its component and the odd-colored pixel's offset within the stamp bbox.
    - Erase the original stamp components to background.
    - For each isolated marker of the stamp's body color, translate the stamp so the odd-pixel offset lands on that marker.
    - Paint every translated stamp cell, including the alignment cell, with the stamp body color.
    - Apply stamps in decreasing stamp area order, so later smaller stamps cover earlier larger stamps when placements overlap.

    Key insight: The unique off-color pixel is not decoration; it is the anchor that fixes the exact translation, and stamp area controls layering.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _background(g)

    singleton_by_color: dict[int, list[tuple[int, int]]] = {}
    stamps: list[dict] = []

    for comp in _non_bg_components(g, bg):
        if len(comp) == 1:
            r, c = comp[0]
            singleton_by_color.setdefault(int(g[r, c]), []).append((r, c))
            continue

        colors = Counter(int(g[r, c]) for r, c in comp)
        if len(colors) != 2:
            continue
        anchor_colors = [col for col, n in colors.items() if int(n) == 1]
        body_colors = [col for col, n in colors.items() if int(n) == len(comp) - 1]
        if len(anchor_colors) != 1 or len(body_colors) != 1:
            continue

        anchor_color = int(anchor_colors[0])
        body = int(body_colors[0])
        r0, c0, _r1, _c1 = _bbox(comp)
        anchor_abs = next((r, c) for r, c in comp if int(g[r, c]) == anchor_color)
        cells = tuple(sorted((int(r - r0), int(c - c0)) for r, c in comp))
        stamps.append(
            {
                "body": body,
                "area": len(comp),
                "cells": cells,
                "anchor": (int(anchor_abs[0] - r0), int(anchor_abs[1] - c0)),
                "origin": (int(r0), int(c0)),
            }
        )

    assert stamps, "expected at least one anchored stamp object"

    out = np.full((h, w), int(bg), dtype=int)
    stamps.sort(key=lambda s: (-int(s["area"]), int(s["origin"][0]), int(s["origin"][1])))

    for stamp in stamps:
        ar, ac = stamp["anchor"]
        for mr, mc in singleton_by_color.get(int(stamp["body"]), []):
            top, left = int(mr - ar), int(mc - ac)
            for rr, cc in stamp["cells"]:
                r, c = int(top + rr), int(left + cc)
                if 0 <= r < h and 0 <= c < w:
                    out[r, c] = int(stamp["body"])
    return out


def _connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            nb = (r + dr, c + dc)
            if nb in cells and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(cells)


def _normalize(cells: set[tuple[int, int]], anchor: tuple[int, int]):
    r0, c0, _r1, _c1 = _bbox(cells)
    out = {(int(r - r0), int(c - c0)) for r, c in cells}
    return out, (int(anchor[0] - r0), int(anchor[1] - c0))


def _valid_shape(cells: set[tuple[int, int]], anchor: tuple[int, int]) -> bool:
    if anchor not in cells or len(cells) < 7:
        return False
    cells, anchor = _normalize(cells, anchor)
    r0, c0, r1, c1 = _bbox(cells)
    if r0 != 0 or c0 != 0 or r1 < 2 or c1 < 2:
        return False
    ar, ac = anchor
    if ar in (0, r1) or ac in (0, c1):
        return False
    body = set(cells)
    body.remove(anchor)
    if not _connected(body):
        return False
    neigh = sum((ar + dr, ac + dc) in body for dr, dc in DIR4)
    return neigh >= 2


def _transform_shape(cells: set[tuple[int, int]], anchor: tuple[int, int], height: int, width: int):
    cells = set(cells)
    ar, ac = anchor
    h, w = int(height), int(width)

    for _ in range(random.randrange(4)):
        cells = {(c, h - 1 - r) for r, c in cells}
        ar, ac = ac, h - 1 - ar
        h, w = w, h
    if random.random() < 0.5:
        cells = {(r, w - 1 - c) for r, c in cells}
        ac = w - 1 - ac
    if random.random() < 0.5:
        cells = {(h - 1 - r, c) for r, c in cells}
        ar = h - 1 - ar
    return _normalize(cells, (ar, ac))


def _make_shape(used_areas: set[int]) -> dict | None:
    for _ in range(250):
        height = random.randint(3, 5)
        width = random.randint(3, 6)
        cells = {(r, c) for r in range(height) for c in range(width)}
        anchor = (random.randint(1, height - 2), random.randint(1, width - 2))
        lo = max(7, height + width - 1)
        hi = min(height * width, 16)
        choices = [a for a in range(lo, hi + 1) if a not in used_areas]
        if not choices:
            continue
        target_area = random.choice(choices)

        while len(cells) > target_area:
            candidates = list(cells - {anchor})
            random.shuffle(candidates)
            removed = False
            for cell in candidates:
                trial = set(cells)
                trial.remove(cell)
                r0, c0, r1, c1 = _bbox(trial)
                if (r0, c0, r1, c1) != (0, 0, height - 1, width - 1):
                    continue
                if _valid_shape(trial, anchor):
                    cells = trial
                    removed = True
                    break
            if not removed:
                break
        if len(cells) != target_area:
            continue

        cells, anchor2 = _transform_shape(cells, anchor, height, width)
        if len(cells) in used_areas or not _valid_shape(cells, anchor2):
            continue
        r0, c0, r1, c1 = _bbox(cells)
        return {
            "cells": tuple(sorted(cells)),
            "anchor": anchor2,
            "area": len(cells),
            "h": int(r1 - r0 + 1),
            "w": int(c1 - c0 + 1),
        }
    return None


def _make_specs(count: int) -> list[dict] | None:
    used_areas: set[int] = set()
    specs = []
    for _ in range(count):
        spec = _make_shape(used_areas)
        if spec is None:
            return None
        used_areas.add(int(spec["area"]))
        specs.append(spec)
    return specs


def _abs_cells(spec: dict, top: int, left: int) -> set[tuple[int, int]]:
    return {(int(top + r), int(left + c)) for r, c in spec["cells"]}


def _marker_pos(spec: dict, top: int, left: int) -> tuple[int, int]:
    ar, ac = spec["anchor"]
    return int(top + ar), int(left + ac)


def _marker_clear(marker: tuple[int, int], markers: list[tuple[int, int]], h: int, w: int) -> bool:
    r, c = marker
    if not (0 < r < h - 1 and 0 < c < w - 1):
        return False
    return all(max(abs(r - rr), abs(c - cc)) > 1 for rr, cc in markers)


def _placement(specs: list[dict], idx: int, top: int, left: int, h: int, w: int) -> dict | None:
    spec = specs[idx]
    if top < 0 or left < 0 or top + int(spec["h"]) > h or left + int(spec["w"]) > w:
        return None
    marker = _marker_pos(spec, top, left)
    cells = _abs_cells(spec, top, left)
    return {"idx": int(idx), "top": int(top), "left": int(left), "marker": marker, "cells": cells}


def _random_top_left(spec: dict, h: int, w: int) -> tuple[int, int]:
    return random.randint(0, h - int(spec["h"])), random.randint(0, w - int(spec["w"]))


def _different_stamp_overlap(placements: list[dict]) -> bool:
    owner: dict[tuple[int, int], int] = {}
    for placement in placements:
        idx = int(placement["idx"])
        for cell in placement["cells"]:
            if cell in owner and owner[cell] != idx:
                return True
            owner[cell] = idx
    return False


def _any_overlap(placements: list[dict]) -> bool:
    seen: set[tuple[int, int]] = set()
    for placement in placements:
        for cell in placement["cells"]:
            if cell in seen:
                return True
            seen.add(cell)
    return False


def _add_overlap_pair(
    specs: list[dict],
    h: int,
    w: int,
    counts: list[int],
    placements: list[dict],
    markers: list[tuple[int, int]],
    output_taken: set[tuple[int, int]],
) -> bool:
    ordered = sorted(range(len(specs)), key=lambda i: -int(specs[i]["area"]))
    pairs = [(ordered[i], ordered[j]) for i in range(len(ordered)) for j in range(i + 1, len(ordered))]
    random.shuffle(pairs)
    for big, small in pairs:
        if counts[big] <= 0 or counts[small] <= 0:
            continue
        a, b = specs[big], specs[small]
        offsets = [(dr, dc) for dr in range(-int(b["h"]) + 1, int(a["h"])) for dc in range(-int(b["w"]) + 1, int(a["w"]))]
        random.shuffle(offsets)
        for _ in range(140):
            top_a, left_a = _random_top_left(a, h, w)
            pa = _placement(specs, big, top_a, left_a, h, w)
            if pa is None:
                continue
            for dr, dc in offsets:
                top_b, left_b = top_a + dr, left_a + dc
                pb = _placement(specs, small, top_b, left_b, h, w)
                if pb is None:
                    continue
                if not (pa["cells"] & pb["cells"]):
                    continue
                ma, mb = pa["marker"], pb["marker"]
                if not _marker_clear(ma, markers, h, w):
                    continue
                if not _marker_clear(mb, markers + [ma], h, w):
                    continue
                placements.extend([pa, pb])
                markers.extend([ma, mb])
                output_taken.update(pa["cells"])
                output_taken.update(pb["cells"])
                counts[big] -= 1
                counts[small] -= 1
                return True
    return False


def _fill_marker_layout(specs: list[dict], h: int, w: int, counts: list[int], want_overlap: bool) -> list[dict] | None:
    placements: list[dict] = []
    markers: list[tuple[int, int]] = []
    output_taken: set[tuple[int, int]] = set()

    if want_overlap and not _add_overlap_pair(specs, h, w, counts, placements, markers, output_taken):
        return None

    tasks = [idx for idx, n in enumerate(counts) for _ in range(int(n))]
    random.shuffle(tasks)
    for idx in tasks:
        spec = specs[idx]
        placed = False
        for _ in range(350):
            top, left = _random_top_left(spec, h, w)
            p = _placement(specs, idx, top, left, h, w)
            if p is None:
                continue
            if p["cells"] & output_taken:
                continue
            if not _marker_clear(p["marker"], markers, h, w):
                continue
            placements.append(p)
            markers.append(p["marker"])
            output_taken.update(p["cells"])
            placed = True
            break
        if not placed:
            return None

    if want_overlap:
        return placements if _different_stamp_overlap(placements) else None
    return None if _any_overlap(placements) else placements


def _expected_grid(bg: int, shape: tuple[int, int], specs: list[dict], placements: list[dict]) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    by_idx: dict[int, list[dict]] = {i: [] for i in range(len(specs))}
    for placement in placements:
        by_idx[int(placement["idx"])].append(placement)
    order = sorted(range(len(specs)), key=lambda i: -int(specs[i]["area"]))
    for idx in order:
        color = int(specs[idx]["body"])
        for placement in by_idx[idx]:
            for r, c in placement["cells"]:
                out[int(r), int(c)] = color
    return out


def _cells_clear(cells: set[tuple[int, int]], occupied: set[tuple[int, int]], h: int, w: int) -> bool:
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                if (rr, cc) in occupied:
                    return False
    return True


def _place_original_stamps(grid: np.ndarray, specs: list[dict], placements: list[dict]) -> bool:
    h, w = grid.shape
    target_cover = set()
    occupied = set()
    for p in placements:
        target_cover.update(p["cells"])
        occupied.add(p["marker"])

    order = list(range(len(specs)))
    random.shuffle(order)
    for idx in order:
        spec = specs[idx]
        candidates = [
            (r, c)
            for r in range(1, h - int(spec["h"]))
            for c in range(1, w - int(spec["w"]))
        ]
        random.shuffle(candidates)
        done = False
        for top, left in candidates:
            cells = _abs_cells(spec, top, left)
            if cells & target_cover:
                continue
            if not _cells_clear(cells, occupied, h, w):
                continue
            ar, ac = spec["anchor"]
            for rr, cc in spec["cells"]:
                r, c = int(top + rr), int(left + cc)
                if (rr, cc) == (ar, ac):
                    grid[r, c] = int(spec["anchor_color"])
                else:
                    grid[r, c] = int(spec["body"])
            occupied.update(cells)
            done = True
            break
        if not done:
            return False
    return True


def _marker_counts(count: int) -> list[int]:
    counts = [1 + int(random.random() < 0.55) for _ in range(count)]
    if random.random() < 0.25:
        counts[random.randrange(count)] += 1
    while sum(counts) > 7:
        i = random.randrange(count)
        if counts[i] > 1:
            counts[i] -= 1
    return counts


def _gen_attempt() -> np.ndarray | None:
    stamp_count = random.randint(2, 4)
    min_side = 18 if stamp_count <= 3 else 22
    h = random.randint(min_side, 30)
    w = random.randint(min_side, 30)

    specs = _make_specs(stamp_count)
    if specs is None:
        return None

    colors = random.sample(range(10), 1 + 2 * stamp_count)
    bg = int(colors[0])
    bodies = colors[1 : 1 + stamp_count]
    anchors = colors[1 + stamp_count :]
    for spec, body, anchor_color in zip(specs, bodies, anchors):
        spec["body"] = int(body)
        spec["anchor_color"] = int(anchor_color)

    want_overlap = random.random() < 0.5
    counts = _marker_counts(stamp_count)
    placements = _fill_marker_layout(specs, h, w, counts, want_overlap)
    if placements is None:
        return None

    grid = np.full((h, w), int(bg), dtype=int)
    for p in placements:
        mr, mc = p["marker"]
        grid[mr, mc] = int(specs[int(p["idx"])]["body"])

    if not _place_original_stamps(grid, specs, placements):
        return None

    expected = _expected_grid(bg, (h, w), specs, placements)
    if _background(grid) != bg:
        return None
    try:
        got = main(grid)
    except Exception:
        return None
    if not np.array_equal(got, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid anchored-stamp input")
