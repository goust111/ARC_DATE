from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
MARKER = 5
SEPARATOR = 6
OBJECT_COLORS = (BLUE, RED, GREEN)

COUNT_TO_DIRECTION = {
    1: "down",
    2: "right",
    3: "up",
    4: "left",
}


def _get_bg(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(grid: np.ndarray, *, bg: int = BG) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color == bg:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(pts, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _component_order_key(coords: np.ndarray, direction: str) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    if direction == "down":
        return (-r1, c0, r0, c1)
    if direction == "up":
        return (r0, c0, r1, c1)
    if direction == "right":
        return (-c1, r0, c0, r1)
    assert direction == "left"
    return (c0, r0, c1, r1)


def _shift_coords(coords: np.ndarray, dr: int, dc: int, steps: int) -> np.ndarray:
    return coords + np.array([dr * steps, dc * steps], dtype=int)


def _settle_panel(panel: np.ndarray, direction: str, *, return_meta: bool = False):
    comps = _find_components(panel, bg=BG)
    comps.sort(key=lambda comp: _component_order_key(comp["coords"], direction))

    h, w = panel.shape
    settled = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    move = {
        "down": (1, 0),
        "up": (-1, 0),
        "right": (0, 1),
        "left": (0, -1),
    }[direction]

    meta: list[dict] = []
    for comp in comps:
        coords = comp["coords"]
        color = int(comp["color"])
        shift = 0
        while True:
            cand = _shift_coords(coords, move[0], move[1], shift + 1)
            rows = cand[:, 0]
            cols = cand[:, 1]
            out_of_bounds = bool(rows.min() < 0 or rows.max() >= h or cols.min() < 0 or cols.max() >= w)
            blocked = False if out_of_bounds else bool(np.any(occupied[rows, cols]))
            if out_of_bounds or blocked:
                cause = "boundary" if out_of_bounds else "collision"
                break
            shift += 1

        placed = _shift_coords(coords, move[0], move[1], shift)
        assert not np.any(occupied[placed[:, 0], placed[:, 1]])
        settled[placed[:, 0], placed[:, 1]] = color
        occupied[placed[:, 0], placed[:, 1]] = True
        if return_meta:
            meta.append(
                {
                    "color": color,
                    "original": coords.copy(),
                    "placed": placed.copy(),
                    "shift": shift,
                    "cause": cause,
                }
            )

    return (settled, meta) if return_meta else settled


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count gray markers to decode each panel's gravity, drop the objects, erase the guides, and overlay the panels.

    Input structure:
    - The background is black and exactly two full-width magenta rows split the grid into three equal-height panels.
    - Each panel contains 2-3 non-black foreground objects made of blue, red, or green 4-connected cells, plus isolated gray marker pixels.
    - Marker counts are distinct across panels and each panel is interpreted in its own local coordinate system bounded by the separators or the grid edge.

    Rule:
    - In each panel, count gray marker pixels and map the count to gravity: 1->down, 2->right, 3->up, 4->left.
    - Erase the gray markers, then move every monochromatic 4-connected object rigidly in that direction until the next step would cross the panel boundary or overlap an already settled object.
    - Process objects from nearest to the target boundary to farthest, then delete the separator rows and transparently overlay the three settled panels by local coordinates.

    Key insight:
    - The gray pixels are not part of the final picture; their quantity is a hidden instruction that selects the gravity direction independently for each panel before the panels are merged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h >= 5 and w >= 1

    sep_rows = [r for r in range(h) if np.all(g[r, :] == SEPARATOR)]
    assert len(sep_rows) == 2, "expected exactly two full-width separator rows"
    r1, r2 = sep_rows
    panel_h = r1
    assert panel_h > 0 and r2 == 2 * panel_h + 1 and h == 3 * panel_h + 2, "separator rows must split three equal-height panels"

    panels = [
        g[:panel_h, :].copy(),
        g[r1 + 1 : r2, :].copy(),
        g[r2 + 1 :, :].copy(),
    ]

    out = np.full((panel_h, w), BG, dtype=int)
    for panel in panels:
        marker_count = int(np.count_nonzero(panel == MARKER))
        assert marker_count in COUNT_TO_DIRECTION, f"unsupported marker count: {marker_count}"
        direction = COUNT_TO_DIRECTION[marker_count]
        panel[panel == MARKER] = BG
        settled = _settle_panel(panel, direction)
        overlap = (out != BG) & (settled != BG)
        assert not np.any(overlap), "panel overlay must be conflict-free"
        out[settled != BG] = settled[settled != BG]

    return out


def _random_shape(area: int) -> np.ndarray:
    options = [
        (hh, ww)
        for hh in range(1, 4)
        for ww in range(1, 4)
        if area <= hh * ww and max(hh, ww) <= 3
    ]
    hh, ww = random.choice(options)
    pts = {(random.randrange(hh), random.randrange(ww))}
    while len(pts) < area:
        frontier = []
        for r, c in pts:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < hh and 0 <= nc < ww and (nr, nc) not in pts:
                    frontier.append((nr, nc))
        if not frontier:
            return _random_shape(area)
        pts.add(random.choice(frontier))
    coords = np.array(sorted(pts), dtype=int)
    coords -= coords.min(axis=0)
    if random.choice((False, True)):
        coords[:, 0] = coords[:, 0].max() - coords[:, 0]
    if random.choice((False, True)):
        coords[:, 1] = coords[:, 1].max() - coords[:, 1]
    coords = np.unique(coords, axis=0)
    coords -= coords.min(axis=0)
    return coords


def _drop_from_side(coords0: np.ndarray, direction: str, anchor: int, occupied: np.ndarray) -> np.ndarray | None:
    h, w = occupied.shape
    r0, c0, r1, c1 = _bbox(coords0)
    obj_h = r1 - r0 + 1
    obj_w = c1 - c0 + 1
    if direction == "down":
        coords = coords0 + np.array([0, anchor], dtype=int)
        step = (1, 0)
    elif direction == "up":
        coords = coords0 + np.array([h - obj_h, anchor], dtype=int)
        step = (-1, 0)
    elif direction == "right":
        coords = coords0 + np.array([anchor, 0], dtype=int)
        step = (0, 1)
    else:
        coords = coords0 + np.array([anchor, w - obj_w], dtype=int)
        step = (0, -1)

    rows = coords[:, 0]
    cols = coords[:, 1]
    if rows.min() < 0 or rows.max() >= h or cols.min() < 0 or cols.max() >= w:
        return None
    if np.any(occupied[rows, cols]):
        return None

    while True:
        cand = coords + np.array(step, dtype=int)
        rows = cand[:, 0]
        cols = cand[:, 1]
        if rows.min() < 0 or rows.max() >= h or cols.min() < 0 or cols.max() >= w:
            break
        if np.any(occupied[rows, cols]):
            break
        coords = cand
    return coords


def _overlap_interval(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _chain_anchor_candidates(coords0: np.ndarray, prev: np.ndarray, direction: str, h: int, w: int) -> list[int]:
    _r0, _c0, r1, c1 = _bbox(coords0)
    obj_h = r1 - _r0 + 1
    obj_w = c1 - _c0 + 1
    pr0, pc0, pr1, pc1 = _bbox(prev)
    if direction in ("down", "up"):
        lo = max(0, pc0 - obj_w + 1)
        hi = min(w - obj_w, pc1)
    else:
        lo = max(0, pr0 - obj_h + 1)
        hi = min(h - obj_h, pr1)
    if lo > hi:
        return []
    out = list(range(lo, hi + 1))
    random.shuffle(out)
    return out


def _free_anchor_candidates(coords0: np.ndarray, direction: str, h: int, w: int) -> list[int]:
    _r0, _c0, r1, c1 = _bbox(coords0)
    obj_h = r1 - _r0 + 1
    obj_w = c1 - _c0 + 1
    limit = (w - obj_w) if direction in ("down", "up") else (h - obj_h)
    out = list(range(limit + 1))
    random.shuffle(out)
    return out


def _max_back_shift(coords: np.ndarray, direction: str, h: int, w: int) -> int:
    if direction == "down":
        return int(coords[:, 0].min())
    if direction == "up":
        return int(h - 1 - coords[:, 0].max())
    if direction == "right":
        return int(coords[:, 1].min())
    assert direction == "left"
    return int(w - 1 - coords[:, 1].max())


def _shift_away(coords: np.ndarray, direction: str, amount: int) -> np.ndarray:
    if direction == "down":
        return _shift_coords(coords, -1, 0, amount)
    if direction == "up":
        return _shift_coords(coords, 1, 0, amount)
    if direction == "right":
        return _shift_coords(coords, 0, -1, amount)
    assert direction == "left"
    return _shift_coords(coords, 0, 1, amount)


def _touching_or_overlapping(a: np.ndarray, b: np.ndarray) -> bool:
    dr = np.abs(a[:, None, 0] - b[None, :, 0])
    dc = np.abs(a[:, None, 1] - b[None, :, 1])
    return bool(np.any((dr <= 1) & (dc <= 1)))


def _place_markers(panel: np.ndarray, count: int) -> bool:
    h, w = panel.shape
    occupied = [(int(r), int(c)) for r, c in np.argwhere(panel != BG)]
    candidates = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if int(panel[r, c]) == BG
        and all(max(abs(r - rr), abs(c - cc)) >= 2 for rr, cc in occupied)
    ]
    random.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    for r, c in candidates:
        if all(max(abs(r - rr), abs(c - cc)) >= 2 for rr, cc in chosen):
            chosen.append((r, c))
            if len(chosen) == count:
                break
    if len(chosen) != count:
        return False
    for r, c in chosen:
        panel[r, c] = MARKER
    return True


def _build_panel(
    panel_h: int,
    width: int,
    marker_count: int,
    reserved_output: np.ndarray,
    *,
    force_collision: bool,
) -> tuple[np.ndarray, np.ndarray, dict] | None:
    direction = COUNT_TO_DIRECTION[marker_count]
    n_objects = random.choice((2, 3))
    shapes = [_random_shape(random.randint(3, 6)) for _ in range(n_objects)]
    colors = random.sample(list(OBJECT_COLORS), n_objects)

    occupied = np.zeros((panel_h, width), dtype=bool)
    final_parts: list[dict] = []
    collision_done = False

    for index, (shape, color) in enumerate(zip(shapes, colors)):
        want_chain = force_collision and not collision_done and index > 0
        anchors = []
        if want_chain:
            for prev in reversed(final_parts):
                anchors.extend(_chain_anchor_candidates(shape, prev["final"], direction, panel_h, width))
        anchors.extend(_free_anchor_candidates(shape, direction, panel_h, width))

        tried: set[int] = set()
        placed = None
        collided_here = False
        for anchor in anchors:
            if anchor in tried:
                continue
            tried.add(anchor)
            coords = _drop_from_side(shape, direction, anchor, occupied)
            if coords is None:
                continue
            rows = coords[:, 0]
            cols = coords[:, 1]
            if np.any(reserved_output[rows, cols]):
                continue
            if _max_back_shift(coords, direction, panel_h, width) < index + 1:
                continue
            for prev in final_parts:
                pr0, pc0, pr1, pc1 = _bbox(prev["final"])
                r0, c0, r1, c1 = _bbox(coords)
                if direction in ("down", "up"):
                    if _overlap_interval(c0, c1, pc0, pc1):
                        gap = max(r0, pr0) - min(r1, pr1) - 1
                        collided_here = gap <= 0
                else:
                    if _overlap_interval(r0, r1, pr0, pr1):
                        gap = max(c0, pc0) - min(c1, pc1) - 1
                        collided_here = gap <= 0
            placed = coords
            if want_chain and collided_here:
                break
            if not want_chain:
                break
            placed = None
            collided_here = False

        if placed is None:
            return None

        occupied[placed[:, 0], placed[:, 1]] = True
        final_parts.append(
            {
                "color": color,
                "final": placed,
                "want_collision": want_chain,
                "did_collision": collided_here,
            }
        )
        collision_done = collision_done or collided_here

    if force_collision and not collision_done:
        return None

    input_panel = np.full((panel_h, width), BG, dtype=int)
    output_panel = np.full((panel_h, width), BG, dtype=int)
    input_coords: list[np.ndarray] = []
    input_parts: list[dict] = []

    for index, part in enumerate(final_parts):
        final_coords = part["final"]
        back = index + 1
        coords_in = _shift_away(final_coords, direction, back)
        rows = coords_in[:, 0]
        cols = coords_in[:, 1]
        if rows.min() < 0 or rows.max() >= panel_h or cols.min() < 0 or cols.max() >= width:
            return None
        if np.any(input_panel[rows, cols] != BG):
            return None
        if any(_touching_or_overlapping(coords_in, other) for other in input_coords):
            return None
        input_panel[rows, cols] = int(part["color"])
        output_panel[final_coords[:, 0], final_coords[:, 1]] = int(part["color"])
        input_coords.append(coords_in)
        input_parts.append(
            {
                "color": int(part["color"]),
                "input": coords_in,
                "final": final_coords,
                "shift": back,
            }
        )

    if not _place_markers(input_panel, marker_count):
        return None

    if any(np.any(input_panel[coords[:, 0], coords[:, 1]] == MARKER) for coords in input_coords):
        return None

    meta = {
        "direction": direction,
        "marker_count": marker_count,
        "parts": input_parts,
        "had_collision": collision_done,
    }
    return input_panel, output_panel, meta


def _validate_example(inp: np.ndarray, expected: np.ndarray, panels_meta: list[dict]) -> bool:
    if inp.ndim != 2 or expected.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if _get_bg(inp) != BG:
        return False
    sep_rows = [r for r in range(inp.shape[0]) if np.all(inp[r, :] == SEPARATOR)]
    if len(sep_rows) != 2:
        return False
    panel_h = sep_rows[0]
    if sep_rows[1] != 2 * panel_h + 1 or inp.shape[0] != 3 * panel_h + 2:
        return False

    if not any(meta["had_collision"] for meta in panels_meta):
        return False

    for idx, meta in enumerate(panels_meta):
        if meta["marker_count"] not in COUNT_TO_DIRECTION:
            return False
        row0 = idx * (panel_h + 1)
        panel = inp[row0 : row0 + panel_h, :]
        if int(np.count_nonzero(panel == MARKER)) != meta["marker_count"]:
            return False
        comps = _find_components(np.where(panel == MARKER, BG, panel), bg=BG)
        if len(comps) != len(meta["parts"]):
            return False
        for comp in comps:
            if comp["area"] < 3 or comp["area"] > 6:
                return False
        coords = [comp["coords"] for comp in comps]
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                if _touching_or_overlapping(coords[i], coords[j]):
                    return False

    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    panel_h = random.randint(6, 8)
    width = random.randint(9, 13)
    marker_counts = random.sample([1, 2, 3, 4], 3)
    reserved_output = np.zeros((panel_h, width), dtype=bool)

    input_panels: list[np.ndarray] = []
    output_panels: list[np.ndarray] = []
    panels_meta: list[dict] = []
    collision_panel = random.randrange(3)

    for idx, marker_count in enumerate(marker_counts):
        panel = None
        for _ in range(80):
            panel = _build_panel(
                panel_h,
                width,
                marker_count,
                reserved_output,
                force_collision=(idx == collision_panel),
            )
            if panel is not None:
                break
        if panel is None:
            return None
        inp_panel, out_panel, meta = panel
        reserved_output |= out_panel != BG
        input_panels.append(inp_panel)
        output_panels.append(out_panel)
        panels_meta.append(meta)

    inp = np.full((3 * panel_h + 2, width), BG, dtype=int)
    expected = np.full((panel_h, width), BG, dtype=int)
    for idx, panel in enumerate(input_panels):
        row0 = idx * (panel_h + 1)
        inp[row0 : row0 + panel_h, :] = panel
        if idx < 2:
            inp[row0 + panel_h, :] = SEPARATOR

    for out_panel in output_panels:
        overlap = (expected != BG) & (out_panel != BG)
        if np.any(overlap):
            return None
        expected[out_panel != BG] = out_panel[out_panel != BG]

    if not _validate_example(inp, expected, panels_meta):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid panel-gravity overlay input")
