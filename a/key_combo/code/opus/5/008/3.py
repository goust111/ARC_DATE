from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from importlib.machinery import SourceFileLoader
from itertools import combinations
import random

import numpy as np

try:
    from arc_utils import components_mask, find_components, get_bg
except Exception:
    _ARC_UTILS = SourceFileLoader(
        "_arc_utils_fallback", "/home/lixinyu/snap/new/solvers/arc_utils.py"
    ).load_module()
    components_mask = _ARC_UTILS.components_mask
    find_components = _ARC_UTILS.find_components
    get_bg = _ARC_UTILS.get_bg


BG = 0
BAR = 5
DIRS = ("up", "down", "left", "right")
D4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


@dataclass(frozen=True)
class Shape:
    cells: tuple[tuple[int, int], ...]
    h: int
    w: int
    direction: str
    front_lanes: tuple[int, ...]
    gated: bool


def _shape_sig(cells: set[tuple[int, int]] | list[tuple[int, int]] | np.ndarray) -> tuple[tuple[int, int], ...]:
    if isinstance(cells, np.ndarray):
        pts = [(int(r), int(c)) for r, c in cells]
    else:
        pts = [(int(r), int(c)) for r, c in cells]
    r0 = min(r for r, _ in pts)
    c0 = min(c for _, c in pts)
    return tuple(sorted((r - r0, c - c0) for r, c in pts))


def _shape_bbox(sig: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in sig) + 1, max(c for _, c in sig) + 1


def _connected(cells: set[tuple[int, int]] | list[tuple[int, int]]) -> bool:
    pts = set(cells)
    if not pts:
        return False
    stack = [next(iter(pts))]
    seen = {stack[0]}
    while stack:
        r, c = stack.pop()
        for dr, dc in D4:
            nb = (r + dr, c + dc)
            if nb in pts and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(pts)


def _centroid_direction(sig: tuple[tuple[int, int], ...], h: int, w: int) -> str:
    rs = [r for r, _ in sig]
    cs = [c for _, c in sig]
    dr = float(sum(rs)) / len(rs) - (h - 1) / 2.0
    dc = float(sum(cs)) / len(cs) - (w - 1) / 2.0
    if max(abs(dr), abs(dc)) < 1e-9 or abs(abs(dr) - abs(dc)) < 1e-9:
        raise ValueError("shape direction is ambiguous")
    if abs(dc) > abs(dr):
        return "right" if dc > 0 else "left"
    return "down" if dr > 0 else "up"


def _front_lanes(sig: tuple[tuple[int, int], ...], direction: str) -> tuple[int, ...]:
    if direction == "up":
        edge = min(r for r, _ in sig)
        return tuple(sorted(c for r, c in sig if r == edge))
    if direction == "down":
        edge = max(r for r, _ in sig)
        return tuple(sorted(c for r, c in sig if r == edge))
    if direction == "left":
        edge = min(c for _, c in sig)
        return tuple(sorted(r for r, c in sig if c == edge))
    edge = max(c for _, c in sig)
    return tuple(sorted(r for r, c in sig if c == edge))


def _build_shape_library() -> dict[str, list[Shape]]:
    library = {d: [] for d in DIRS}
    for w in range(3, 5):
        universe = [(r, c) for r in range(2) for c in range(w)]
        max_area = min(7, 2 * w)
        for area in range(4, max_area + 1):
            for comb in combinations(universe, area):
                cells = set(comb)
                if not _connected(cells):
                    continue
                sig = _shape_sig(cells)
                h, ww = _shape_bbox(sig)
                if h != 2 or ww != w:
                    continue
                try:
                    direction = _centroid_direction(sig, h, ww)
                except ValueError:
                    continue
                lanes = _front_lanes(sig, direction)
                span = ww if direction in ("up", "down") else h
                library[direction].append(
                    Shape(
                        cells=sig,
                        h=h,
                        w=ww,
                        direction=direction,
                        front_lanes=lanes,
                        gated=len(lanes) < span,
                    )
                )
    for direction in DIRS:
        uniq: dict[tuple[tuple[int, int], ...], Shape] = {}
        for shape in library[direction]:
            uniq[shape.cells] = shape
        library[direction] = sorted(
            uniq.values(),
            key=lambda s: (not s.gated, s.w, len(s.cells), s.cells),
        )
    return library


SHAPES = _build_shape_library()


def _translate(sig: tuple[tuple[int, int], ...], top: int, left: int) -> set[tuple[int, int]]:
    return {(top + r, left + c) for r, c in sig}


def _bbox_of_cells(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _find_bar_and_notches(grid: np.ndarray) -> tuple[int, dict, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    assert comps, "no components found"
    bar = max(comps, key=lambda comp: int(comp["area"]))
    bar_cells = {(int(r), int(c)) for r, c in bar["coords"]}
    br0, bc0, br1, bc1 = map(int, bar["bbox"])
    bh, bw = br1 - br0 + 1, bc1 - bc0 + 1
    bar_mask = np.zeros((bh, bw), dtype=bool)
    for r, c in bar_cells:
        bar_mask[r - br0, c - bc0] = True

    notches: list[dict] = []
    for pts in components_mask(~bar_mask, connectivity=4):
        rs = pts[:, 0]
        cs = pts[:, 1]
        touches_top = bool(np.any(rs == 0))
        touches_bottom = bool(np.any(rs == bh - 1))
        if not (touches_top or touches_bottom):
            continue
        assert not (touches_top and touches_bottom), "notch cannot touch both bar sides"
        side = "top" if touches_top else "bottom"
        hole = {(br0 + int(r), bc0 + int(c)) for r, c in pts}
        sig = _shape_sig(pts)
        hr0, hc0, hr1, hc1 = _bbox_of_cells(hole)
        notches.append(
            {
                "side": side,
                "cells": hole,
                "sig": sig,
                "bbox": (hr0, hc0, hr1, hc1),
            }
        )
    notches.sort(key=lambda n: (n["bbox"][0], n["bbox"][1]))
    return bg, {"cells": bar_cells, "bbox": (br0, bc0, br1, bc1), "color": int(bar["color"])}, notches


def _current_front(cells: set[tuple[int, int]], direction: str) -> tuple[int, tuple[int, ...]]:
    if direction == "up":
        coord = min(r for r, _ in cells)
        lanes = tuple(sorted(c for r, c in cells if r == coord))
        return coord, lanes
    if direction == "down":
        coord = max(r for r, _ in cells)
        lanes = tuple(sorted(c for r, c in cells if r == coord))
        return coord, lanes
    if direction == "left":
        coord = min(c for _, c in cells)
        lanes = tuple(sorted(r for r, c in cells if c == coord))
        return coord, lanes
    coord = max(c for _, c in cells)
    lanes = tuple(sorted(r for r, c in cells if c == coord))
    return coord, lanes


def _search_marker(
    remaining: set[tuple[int, int]],
    direction: str,
    front_coord: int,
    lanes: tuple[int, ...],
    strip: tuple[int, int],
) -> tuple[int, int] | None:
    if not remaining or not lanes:
        return None
    center2 = strip[0] + strip[1]
    candidates: list[tuple[tuple[int, int, int, int], tuple[int, int]]] = []
    for r, c in remaining:
        if direction == "up":
            if r >= front_coord or c not in lanes:
                continue
            key = (front_coord - r, abs(2 * c - center2), c, r)
        elif direction == "down":
            if r <= front_coord or c not in lanes:
                continue
            key = (r - front_coord, abs(2 * c - center2), c, r)
        elif direction == "left":
            if c >= front_coord or r not in lanes:
                continue
            key = (front_coord - c, abs(2 * r - center2), r, c)
        else:
            if c <= front_coord or r not in lanes:
                continue
            key = (c - front_coord, abs(2 * r - center2), r, c)
        candidates.append((key, (int(r), int(c))))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _segment_to_marker(marker: tuple[int, int], direction: str, front_coord: int, strip: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = marker
    a, b = strip
    if direction == "up":
        return {(rr, cc) for rr in range(r, front_coord) for cc in range(a, b + 1)}
    if direction == "down":
        return {(rr, cc) for rr in range(front_coord + 1, r + 1) for cc in range(a, b + 1)}
    if direction == "left":
        return {(rr, cc) for rr in range(a, b + 1) for cc in range(c, front_coord)}
    return {(rr, cc) for rr in range(a, b + 1) for cc in range(front_coord + 1, c + 1)}


def _solve_scene(grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg, bar, notches = _find_bar_and_notches(g)
    assert bg == BG
    assert bar["color"] == BAR
    assert 2 <= len(notches) <= 3

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    sprites: list[dict] = []
    markers_by_color: dict[int, set[tuple[int, int]]] = {}
    for comp in comps:
        color = int(comp["color"])
        if color == BAR:
            continue
        area = int(comp["area"])
        cells = {(int(r), int(c)) for r, c in comp["coords"]}
        if area == 1:
            markers_by_color.setdefault(color, set()).update(cells)
        else:
            sig = _shape_sig(comp["coords"])
            hh, ww = _shape_bbox(sig)
            sprites.append(
                {
                    "color": color,
                    "sig": sig,
                    "bbox": tuple(map(int, comp["bbox"])),
                    "h": hh,
                    "w": ww,
                }
            )

    assert len(sprites) == len(notches), "sprite count must match notch count"
    assert len({sprite["sig"] for sprite in sprites}) == len(sprites), "sprite shapes must be unique"
    assert len({notch["sig"] for notch in notches}) == len(notches), "notch shapes must be unique"

    sprite_by_sig = {sprite["sig"]: sprite for sprite in sprites}
    roots: list[dict] = []
    for notch in notches:
        assert notch["sig"] in sprite_by_sig, "every notch must have a matching sprite"
        sprite = sprite_by_sig[notch["sig"]]
        direction = _centroid_direction(sprite["sig"], sprite["h"], sprite["w"])
        r0, c0, r1, c1 = notch["bbox"]
        if direction in ("up", "down"):
            strip = (c0, c1)
        else:
            strip = (r0, r1)
        roots.append(
            {
                "color": int(sprite["color"]),
                "direction": direction,
                "root_cells": set(notch["cells"]),
                "root_bbox": (r0, c0, r1, c1),
                "strip": strip,
                "sprite_sig": sprite["sig"],
            }
        )

    roots.sort(key=lambda item: item["color"])

    out = np.full((h, w), bg, dtype=int)
    root_cells_all: dict[tuple[int, int], int] = {}
    for r, c in bar["cells"]:
        out[r, c] = BAR
    for root in roots:
        for r, c in root["root_cells"]:
            out[r, c] = int(root["color"])
            root_cells_all[(r, c)] = int(root["color"])

    chain_info: list[dict] = []
    for root in roots:
        color = int(root["color"])
        remaining = set(markers_by_color.get(color, set()))
        current = set(root["root_cells"])
        consumed: list[tuple[int, int]] = []
        segments: list[set[tuple[int, int]]] = []
        while True:
            front_coord, lanes = _current_front(current, root["direction"])
            marker = _search_marker(remaining, root["direction"], front_coord, lanes, root["strip"])
            if marker is None:
                break
            remaining.remove(marker)
            consumed.append(marker)
            seg = _segment_to_marker(marker, root["direction"], front_coord, root["strip"])
            current |= seg
            segments.append(seg)
        chain_info.append(
            {
                "color": color,
                "direction": root["direction"],
                "strip": root["strip"],
                "root_cells": set(root["root_cells"]),
                "segments": segments,
                "consumed": consumed,
                "ignored_markers": remaining,
            }
        )

    written_by_chain: dict[tuple[int, int], int] = {}
    for chain in sorted(chain_info, key=lambda item: item["color"]):
        color = int(chain["color"])
        for seg in chain["segments"]:
            for cell in seg:
                if cell in root_cells_all and root_cells_all[cell] != color:
                    continue
                prev = written_by_chain.get(cell)
                if prev is not None and prev != color:
                    continue
                written_by_chain[cell] = color
                out[cell] = color

    return out, chain_info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Plug asymmetric sprites into a gray bar's notches, then grow fixed-width chains along the root piece's facing side.

    Input structure:
    - Background is black. The largest monochrome component is a horizontal gray bar whose top or bottom edge has 2-3 background notches.
    - Each non-gray multi-cell component is a monochrome sprite whose exact shape matches one notch, and each sprite shape is asymmetric inside its own bbox.
    - Each sprite color also has several isolated same-color singleton markers scattered away from the sprites.

    Rule:
    - Match each notch with the unique sprite of the same normalized shape and place that sprite into the notch as the chain root.
    - Read the root direction from the root shape's centroid offset relative to its bbox center: larger horizontal offset means left/right, larger vertical offset means up/down.
    - The root bbox fixes a strip width perpendicular to that direction. Repeatedly search forward from the root or current chain front: only singleton markers lying on the current front face's active lanes are eligible, and the nearest eligible one is consumed.
    - For each consumed marker, extend a full fixed-width strip segment from the current front to that marker, then continue from the new front. Keep the bar and filled roots, erase the original scattered sprites, and resolve strip-strip overlaps in favor of the smaller color id.

    Key insight:
    - The same root shape carries both pieces of information: its contour selects the notch, and its centroid skew selects the chain direction and the initial search lanes that decide which markers can start the chain.
    """
    out, _ = _solve_scene(np.asarray(grid, dtype=int))
    return out


def _choose_shape(rng: random.Random, direction: str, prefer_gated: bool) -> Shape:
    choices = SHAPES[direction]
    if prefer_gated:
        gated = [shape for shape in choices if shape.gated]
        if gated:
            choices = gated
    return rng.choice(choices)


def _abs_front_lanes(root_top: int, root_left: int, shape: Shape, direction: str) -> tuple[int, ...]:
    if direction in ("up", "down"):
        return tuple(root_left + lane for lane in shape.front_lanes)
    return tuple(root_top + lane for lane in shape.front_lanes)


def _pick_notch_positions(rng: random.Random, widths: list[int], bar_c0: int, bar_w: int) -> list[int] | None:
    order = sorted(range(len(widths)), key=lambda i: -widths[i])
    starts = [None] * len(widths)
    intervals: list[tuple[int, int]] = []
    for idx in order:
        width = widths[idx]
        candidates = []
        for start in range(bar_c0 + 1, bar_c0 + bar_w - width - 1):
            end = start + width - 1
            if all(end + 2 < a or start - 2 > b for a, b in intervals):
                candidates.append(start)
        if not candidates:
            return None
        start = rng.choice(candidates)
        starts[idx] = start
        intervals.append((start, start + width - 1))
    return [int(v) for v in starts]


def _make_chain_plan(
    rng: random.Random,
    h: int,
    w: int,
    bar_bbox: tuple[int, int, int, int],
    side: str,
    color: int,
    shape: Shape,
    notch_left: int,
) -> dict | None:
    br0, bc0, br1, bc1 = bar_bbox
    root_top = br0 if side == "top" else br1 - 1
    root_left = notch_left
    root_bbox = (root_top, root_left, root_top + shape.h - 1, root_left + shape.w - 1)
    root_cells = _translate(shape.cells, root_top, root_left)
    direction = shape.direction
    if direction == "up" and side != "top":
        return None
    if direction == "down" and side != "bottom":
        return None
    if direction in ("up", "down"):
        strip = (root_left, root_left + shape.w - 1)
    else:
        strip = (root_top, root_top + shape.h - 1)

    markers_reachable: list[tuple[int, int]] = []
    markers_decoy: list[tuple[int, int]] = []
    front_lanes = _abs_front_lanes(root_top, root_left, shape, direction)
    full_rows = list(range(root_top, root_top + shape.h))
    full_cols = list(range(root_left, root_left + shape.w))

    if direction == "up":
        max_count = min(4, root_top - 1)
        if max_count < 2:
            return None
        count = rng.randint(2, max_count)
        current = root_top
        for _ in range(count):
            gap = rng.randint(2, 3)
            row = current - gap
            if row < 1:
                break
            col = rng.choice(full_cols)
            markers_reachable.append((row, col))
            current = row
        if len(markers_reachable) < 2:
            return None
        if rng.random() < 0.8:
            decoy_col = full_cols[0] - 1 if full_cols[0] > 0 else full_cols[-1] + 1
            decoy_row = markers_reachable[0][0] + 1
            if 0 <= decoy_col < w and 0 < decoy_row < root_top:
                markers_decoy.append((decoy_row, decoy_col))
    elif direction == "down":
        max_count = min(4, h - (root_top + shape.h) - 1)
        if max_count < 2:
            return None
        count = rng.randint(2, max_count)
        current = root_top + shape.h - 1
        for _ in range(count):
            gap = rng.randint(2, 3)
            row = current + gap
            if row > h - 2:
                break
            col = rng.choice(full_cols)
            markers_reachable.append((row, col))
            current = row
        if len(markers_reachable) < 2:
            return None
        if rng.random() < 0.8:
            decoy_col = full_cols[-1] + 1 if full_cols[-1] + 1 < w else full_cols[0] - 1
            decoy_row = markers_reachable[0][0] - 1
            if 0 <= decoy_col < w and root_top + shape.h - 1 < decoy_row < h - 1:
                markers_decoy.append((decoy_row, decoy_col))
    elif direction == "right":
        count = rng.randint(2, 4)
        if bc1 + 2 >= w - 1:
            return None
        current = bc1
        first_row = rng.choice(list(front_lanes))
        other_rows = [row for row in full_rows if row not in front_lanes]
        first_col = bc1 + rng.randint(2, 3)
        if first_col >= w - 1:
            return None
        markers_reachable.append((first_row, first_col))
        current = first_col
        for _ in range(count - 1):
            gap = rng.randint(2, 4)
            col = current + gap
            if col > w - 2:
                break
            row = rng.choice(full_rows)
            markers_reachable.append((row, col))
            current = col
        if len(markers_reachable) < 2:
            return None
        if other_rows:
            decoy_col = bc1 + 1
            if decoy_col < markers_reachable[0][1]:
                markers_decoy.append((rng.choice(other_rows), decoy_col))
    else:
        count = rng.randint(2, 4)
        if bc0 - 2 <= 0:
            return None
        current = bc0
        first_row = rng.choice(list(front_lanes))
        other_rows = [row for row in full_rows if row not in front_lanes]
        first_col = bc0 - rng.randint(2, 3)
        if first_col <= 0:
            return None
        markers_reachable.append((first_row, first_col))
        current = first_col
        for _ in range(count - 1):
            gap = rng.randint(2, 4)
            col = current - gap
            if col < 1:
                break
            row = rng.choice(full_rows)
            markers_reachable.append((row, col))
            current = col
        if len(markers_reachable) < 2:
            return None
        if other_rows:
            decoy_col = bc0 - 1
            if decoy_col > markers_reachable[0][1]:
                markers_decoy.append((rng.choice(other_rows), decoy_col))

    markers_all = list(dict.fromkeys(markers_reachable + markers_decoy))
    return {
        "color": color,
        "shape": shape,
        "side": side,
        "direction": direction,
        "root_bbox": root_bbox,
        "root_cells": root_cells,
        "strip": strip,
        "reachable_markers": markers_reachable,
        "decoy_markers": markers_decoy,
        "markers": markers_all,
    }


def _can_place(cells: set[tuple[int, int]], occupied: set[tuple[int, int]], h: int, w: int, gap: bool = True) -> bool:
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if (r, c) in occupied:
            return False
        if gap:
            for dr, dc in D4:
                if (r + dr, c + dc) in occupied:
                    return False
    return True


def _place_sprites(
    rng: random.Random,
    h: int,
    w: int,
    bar_cells: set[tuple[int, int]],
    marker_cells: set[tuple[int, int]],
    chains: list[dict],
) -> list[dict] | None:
    occupied = set(bar_cells) | set(marker_cells)
    placed: list[dict] = []
    for chain in rng.sample(chains, len(chains)):
        shape = chain["shape"]
        sig = shape.cells
        ph, pw = shape.h, shape.w
        placed_ok = False
        for _ in range(250):
            zone = rng.choice(["top", "bottom", "left", "right"])
            if zone == "top":
                top_hi = min(h - ph - 2, chain["root_bbox"][0] - 4)
                left_lo, left_hi = 1, w - pw - 2
                if top_hi < 1 or left_hi < left_lo:
                    continue
                top = rng.randint(1, top_hi)
                left = rng.randint(left_lo, left_hi)
            elif zone == "bottom":
                top_lo, top_hi = max(1, chain["root_bbox"][2] + 4), h - ph - 2
                left_lo, left_hi = 1, w - pw - 2
                if top_hi < top_lo or left_hi < left_lo:
                    continue
                top = rng.randint(top_lo, top_hi)
                left = rng.randint(left_lo, left_hi)
            elif zone == "left":
                top_lo, top_hi = 1, h - ph - 2
                left_lo, left_hi = 1, chain["root_bbox"][1] - pw - 4
                if top_hi < top_lo or left_hi < left_lo:
                    continue
                top = rng.randint(top_lo, top_hi)
                left = rng.randint(left_lo, left_hi)
            else:
                top_lo, top_hi = 1, h - ph - 2
                left_lo, left_hi = chain["root_bbox"][3] + 4, w - pw - 2
                if top_hi < top_lo or left_hi < left_lo:
                    continue
                top = rng.randint(top_lo, top_hi)
                left = rng.randint(left_lo, left_hi)
            cells = _translate(sig, top, left)
            if not _can_place(cells, occupied, h, w, gap=True):
                continue
            occupied |= cells
            placed.append({"color": chain["color"], "cells": cells})
            placed_ok = True
            break
        if not placed_ok:
            return None
    return placed


def _render_expected(h: int, w: int, bar_cells: set[tuple[int, int]], chains: list[dict]) -> np.ndarray:
    out = np.full((h, w), BG, dtype=int)
    for r, c in bar_cells:
        out[r, c] = BAR
    root_map: dict[tuple[int, int], int] = {}
    for chain in chains:
        for cell in chain["root_cells"]:
            root_map[cell] = chain["color"]
            out[cell] = chain["color"]
    for chain in sorted(chains, key=lambda item: item["color"]):
        color = chain["color"]
        current = set(chain["root_cells"])
        for marker in chain["reachable_markers"]:
            front_coord, _ = _current_front(current, chain["direction"])
            seg = _segment_to_marker(marker, chain["direction"], front_coord, chain["strip"])
            current |= seg
            for cell in seg:
                if cell in root_map and root_map[cell] != color:
                    continue
                if out[cell] in (BG, BAR, color):
                    out[cell] = color
    return out


def _validate_semantics(inp: np.ndarray, expected: np.ndarray, chains: list[dict]) -> None:
    actual, solved = _solve_scene(inp)
    assert np.array_equal(actual, expected), "solver output mismatches planned output"
    assert np.array_equal(main(inp), expected), "main() mismatches planned output"
    assert len(solved) == len(chains)

    shape_to_color = {chain["shape"].cells: chain["color"] for chain in chains}
    assert len(shape_to_color) == len(chains), "shape matching must be unique"

    horizontal_counterexample = False
    for chain in chains:
        direction = chain["direction"]
        current = set(chain["root_cells"])
        remaining = set(chain["markers"])
        for idx, marker in enumerate(chain["reachable_markers"]):
            front_coord, lanes = _current_front(current, direction)
            picked = _search_marker(remaining, direction, front_coord, lanes, chain["strip"])
            assert picked == marker, "nearest-marker order changed"
            remaining.remove(marker)
            seg = _segment_to_marker(marker, direction, front_coord, chain["strip"])
            current |= seg
            if idx == 0 and chain["decoy_markers"] and direction in ("left", "right"):
                horizontal_counterexample = True
        if direction in ("left", "right") and chain["decoy_markers"]:
            front_coord, lanes = _current_front(set(chain["root_cells"]), direction)
            for decoy in chain["decoy_markers"]:
                assert _search_marker({decoy}, direction, front_coord, lanes, chain["strip"]) is None

    for i, chain_a in enumerate(chains):
        current_a = set(chain_a["root_cells"])
        cells_a = set()
        for marker in chain_a["reachable_markers"]:
            front_a, _ = _current_front(current_a, chain_a["direction"])
            seg_a = _segment_to_marker(marker, chain_a["direction"], front_a, chain_a["strip"])
            current_a |= seg_a
            cells_a |= seg_a
        for chain_b in chains[i + 1 :]:
            current_b = set(chain_b["root_cells"])
            cells_b = set()
            for marker in chain_b["reachable_markers"]:
                front_b, _ = _current_front(current_b, chain_b["direction"])
                seg_b = _segment_to_marker(marker, chain_b["direction"], front_b, chain_b["strip"])
                current_b |= seg_b
                cells_b |= seg_b
            overlap = cells_a & cells_b
            if overlap:
                low = min(chain_a["color"], chain_b["color"])
                for r, c in overlap:
                    assert int(expected[r, c]) == low
    assert expected.shape == inp.shape
    assert max(expected.shape) <= 30
    assert max(inp.shape) <= 30


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(18, 25)
    w = rng.randint(22, 28)
    bar_h = 3
    bar_w = rng.randint(14, min(18, w - 4))
    bar_r0 = rng.randint(6, min(9, h - 9))
    bar_c0 = rng.randint(2, w - bar_w - 2)
    bar_bbox = (bar_r0, bar_c0, bar_r0 + bar_h - 1, bar_c0 + bar_w - 1)
    bar_cells = {
        (r, c)
        for r in range(bar_bbox[0], bar_bbox[2] + 1)
        for c in range(bar_bbox[1], bar_bbox[3] + 1)
    }

    num_chains = rng.choice([2, 2, 3])
    palette = [c for c in range(1, 10) if c != BAR]
    rng.shuffle(palette)

    chains: list[dict] = []
    specs: list[tuple[str, Shape]] = []

    conflict_mode = num_chains == 2 and rng.random() < 0.4
    if num_chains == 3:
        specs.append(("top", _choose_shape(rng, "up", prefer_gated=False)))
        specs.append(("bottom", _choose_shape(rng, "down", prefer_gated=False)))
        third_side = rng.choice(["top", "bottom"])
        if rng.random() < 0.45:
            third_dir = rng.choice(["left", "right"])
        else:
            third_dir = "up" if third_side == "top" else "down"
        specs.append((third_side, _choose_shape(rng, third_dir, prefer_gated=third_dir in ("left", "right"))))
    elif conflict_mode:
        specs.extend(
            [
                ("top", _choose_shape(rng, "right", prefer_gated=True)),
                ("top", _choose_shape(rng, "left", prefer_gated=True)),
            ]
        )
    else:
        for _ in range(num_chains):
            side = rng.choice(["top", "bottom"])
            if rng.random() < 0.55:
                direction = rng.choice(["left", "right"])
            else:
                direction = "up" if side == "top" else "down"
            specs.append((side, _choose_shape(rng, direction, prefer_gated=direction in ("left", "right"))))

    colors = sorted(palette[:num_chains]) if conflict_mode else palette[:num_chains]
    widths = [shape.w for _, shape in specs]
    sides = [side for side, _ in specs]
    shapes = [shape for _, shape in specs]

    if len({shape.cells for shape in shapes}) != len(shapes):
        return None
    min_required_bar_w = sum(widths) + 2 * (num_chains - 1) + 2
    if min_required_bar_w > bar_w:
        return None

    starts = _pick_notch_positions(rng, widths, bar_c0, bar_w)
    if starts is None:
        return None
    if conflict_mode and starts[0] > starts[1]:
        starts[0], starts[1] = starts[1], starts[0]

    for idx in range(num_chains):
        plan = _make_chain_plan(
            rng,
            h,
            w,
            bar_bbox,
            sides[idx],
            colors[idx],
            shapes[idx],
            starts[idx],
        )
        if plan is None:
            return None
        chains.append(plan)

    for chain in chains:
        bar_cells -= set(chain["root_cells"])
    if not _connected(bar_cells):
        return None

    marker_cells: set[tuple[int, int]] = set()
    for chain in chains:
        for marker in chain["markers"]:
            if marker in marker_cells or marker in bar_cells:
                return None
            if not (0 <= marker[0] < h and 0 <= marker[1] < w):
                return None
            if any((marker[0] + dr, marker[1] + dc) in marker_cells for dr, dc in D4):
                return None
            marker_cells.add(marker)

    sprites = _place_sprites(rng, h, w, bar_cells, marker_cells, chains)
    if sprites is None:
        return None

    inp = np.full((h, w), BG, dtype=int)
    for r, c in bar_cells:
        inp[r, c] = BAR
    for sprite in sprites:
        for r, c in sprite["cells"]:
            inp[r, c] = int(sprite["color"])
    for chain in chains:
        for r, c in chain["markers"]:
            if inp[r, c] != BG:
                return None
            inp[r, c] = int(chain["color"])

    expected = _render_expected(h, w, bar_cells, chains)
    _validate_semantics(inp, expected, chains)
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        rng = random.Random()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")


if __name__ == "__main__":
    sample = generate_input()
    result = main(sample)
    print(sample.shape, result.shape, Counter(sample.ravel()))
