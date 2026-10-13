from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


BG = 0
FRAME_COLOR = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

# --------------- generator config ---------------
GRID_H_RANGE = (24, 33)
GRID_W_RANGE = (24, 33)
FRAME_COUNT_RANGE = (2, 4)
FRAME_H_RANGE = (7, 11)
FRAME_W_RANGE = (7, 12)
FRAGMENTS_PER_FRAME_RANGE = (2, 4)
SHAPE_RETRIES = 100
PLACEMENT_RETRIES = 250
GEN_RETRIES = 400


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)


def _bbox(coords: list[Coord] | tuple[Coord, ...]) -> BBox:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_hw(box: BBox) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    return r1 - r0 + 1, c1 - c0 + 1


def _lr_tb_key(rc: Coord) -> tuple[int, int]:
    # "left to right, then top to bottom" => primary key is column.
    return rc[1], rc[0]


def _normalize(coords: list[Coord] | tuple[Coord, ...]) -> tuple[tuple[Coord, ...], int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    norm = tuple(sorted((r - r0, c - c0) for r, c in coords))
    return norm, r1 - r0 + 1, c1 - c0 + 1


def _monochrome_components(grid: np.ndarray, mask: np.ndarray | None = None) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    active = np.ones((h, w), dtype=bool) if mask is None else np.asarray(mask, dtype=bool).copy()
    visited = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if not active[r, c] or visited[r, c]:
                continue
            color = int(g[r, c])
            visited[r, c] = True
            q: deque[Coord] = deque([(r, c)])
            coords: list[Coord] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and active[nr, nc] and not visited[nr, nc]:
                        if int(g[nr, nc]) == color:
                            visited[nr, nc] = True
                            q.append((nr, nc))
            box = _bbox(coords)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": box,
                    "area": len(coords),
                }
            )
    return comps


def _border_cells(box: BBox) -> set[Coord]:
    r0, c0, r1, c1 = box
    cells: set[Coord] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _find_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    comps = _monochrome_components(g, mask=(g != BG))
    frames: list[dict] = []

    for comp in comps:
        box = comp["bbox"]
        h, w = _bbox_hw(box)
        if min(h, w) < 5:
            continue
        border = _border_cells(box)
        if set(comp["coords"]) != border:
            continue

        r0, c0, r1, c1 = box
        markers: list[Coord] = []
        marker_colors: set[int] = set()
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                v = int(g[r, c])
                if v == BG:
                    continue
                markers.append((r, c))
                marker_colors.add(v)
        if not markers or len(marker_colors) != 1:
            continue
        if not all(r in (r0 + 1, r1 - 1) or c in (c0 + 1, c1 - 1) for r, c in markers):
            continue

        frames.append(
            {
                "frame_color": int(comp["color"]),
                "bbox": box,
                "markers": markers,
                "marker_color": int(next(iter(marker_colors))),
            }
        )

    return frames


def _find_external_fragments(grid: np.ndarray, frames: list[dict]) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    blocked = np.zeros(g.shape, dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        blocked[r0 : r1 + 1, c0 : c1 + 1] = True

    active = (g != BG) & (~blocked)
    comps = _monochrome_components(g, mask=active)
    frags: list[dict] = []
    for comp in comps:
        norm_coords, hh, ww = _normalize(comp["coords"])
        frags.append(
            {
                "color": int(comp["color"]),
                "coords": comp["coords"],
                "bbox": comp["bbox"],
                "norm_coords": norm_coords,
                "h": hh,
                "w": ww,
            }
        )
    return frags


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each scattered monochrome fragment to the grey frame whose wall markers share its color,
    then translate fragments so their bbox top-left corners align to the sorted marker pixels.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    frames = _find_frames(g)
    assert frames, "expected at least one rectangular frame"

    fragments = _find_external_fragments(g, frames)
    for frag in fragments:
        for r, c in frag["coords"]:
            out[r, c] = BG

    frame_by_color: dict[int, dict] = {}
    for frame in frames:
        color = int(frame["marker_color"])
        assert color not in frame_by_color, "each marker color must identify exactly one frame"
        frame_by_color[color] = frame

    frags_by_color: dict[int, list[dict]] = defaultdict(list)
    for frag in fragments:
        frags_by_color[int(frag["color"])].append(frag)

    for color in frags_by_color:
        assert color in frame_by_color, "every fragment color must have a matching frame"

    for color, frame in frame_by_color.items():
        markers = sorted(frame["markers"], key=_lr_tb_key)
        frags = sorted(
            frags_by_color.get(int(color), []),
            key=lambda frag: _lr_tb_key((frag["bbox"][0], frag["bbox"][1])),
        )
        assert len(frags) == len(markers), "marker count must equal fragment count"

        r0, c0, r1, c1 = frame["bbox"]
        for frag, anchor in zip(frags, markers):
            ar, ac = anchor
            for dr, dc in frag["norm_coords"]:
                rr = ar + dr
                cc = ac + dc
                assert r0 + 1 <= rr <= r1 - 1 and c0 + 1 <= cc <= c1 - 1
                out[rr, cc] = int(color)

    return out


# --------------- generator helpers ---------------


def _box_is_free(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, margin: int = 1) -> bool:
    hh, ww = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + h > hh or c0 + w > ww:
        return False
    rr0 = max(0, r0 - margin)
    cc0 = max(0, c0 - margin)
    rr1 = min(hh, r0 + h + margin)
    cc1 = min(ww, c0 + w + margin)
    return not bool(np.any(occupied[rr0:rr1, cc0:cc1]))


def _reserve_box(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, margin: int = 1) -> None:
    hh, ww = occupied.shape
    rr0 = max(0, r0 - margin)
    cc0 = max(0, c0 - margin)
    rr1 = min(hh, r0 + h + margin)
    cc1 = min(ww, c0 + w + margin)
    occupied[rr0:rr1, cc0:cc1] = True


def _random_connected_shape(rng: random.Random, cover_anchor: bool) -> dict:
    for _ in range(SHAPE_RETRIES):
        if cover_anchor:
            max_h, max_w = rng.choice(
                [(1, 2), (2, 1), (1, 3), (3, 1), (2, 2), (2, 3), (3, 2), (3, 3)]
            )
            start = {(0, 0)}
            target = rng.randint(2, min(max_h * max_w, 5))
        else:
            max_h = rng.randint(2, 3)
            max_w = rng.randint(2, 3)
            start = {(0, 1), (1, 0), (1, 1)}
            target = rng.randint(3, min(max_h * max_w, 5))

        coords = set(start)
        while len(coords) < target:
            frontier: list[Coord] = []
            for r, c in coords:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < max_h and 0 <= nc < max_w and (nr, nc) not in coords:
                        frontier.append((nr, nc))
            if not frontier:
                break
            coords.add(rng.choice(frontier))

        norm, hh, ww = _normalize(tuple(coords))
        if len(norm) < 2:
            continue
        if cover_anchor and (0, 0) not in norm:
            continue
        if not cover_anchor and ((0, 0) in norm or hh < 2 or ww < 2):
            continue
        if len(norm) == hh * ww and len(norm) > 2 and rng.random() < 0.7:
            continue

        return {
            "coords": norm,
            "h": hh,
            "w": ww,
            "cover_anchor": cover_anchor,
        }

    raise RuntimeError("failed to generate connected shape")


def _candidate_anchors(interior_h: int, interior_w: int, shape: dict) -> list[Coord]:
    anchors: list[Coord] = []
    for r in range(interior_h):
        for c in range(interior_w):
            if r not in (0, interior_h - 1) and c not in (0, interior_w - 1):
                continue
            if r + int(shape["h"]) > interior_h or c + int(shape["w"]) > interior_w:
                continue
            anchors.append((r, c))
    return anchors


def _assign_anchors(
    rng: random.Random,
    interior_h: int,
    interior_w: int,
    ordered_shapes: list[dict],
) -> list[Coord] | None:
    candidate_lists = [_candidate_anchors(interior_h, interior_w, shape) for shape in ordered_shapes]
    if any(not options for options in candidate_lists):
        return None

    def rec(
        idx: int,
        prev_key: tuple[int, int],
        occupied_cells: set[Coord],
        anchors: tuple[Coord, ...],
    ) -> list[Coord] | None:
        if idx == len(ordered_shapes):
            return list(anchors)

        shape = ordered_shapes[idx]
        options: list[tuple[Coord, set[Coord]]] = []
        for anchor in candidate_lists[idx]:
            if _lr_tb_key(anchor) <= prev_key:
                continue
            if anchor in occupied_cells:
                continue

            placed = {(anchor[0] + dr, anchor[1] + dc) for dr, dc in shape["coords"]}
            if placed & occupied_cells:
                continue
            if any(old_anchor in placed for old_anchor in anchors):
                continue
            options.append((anchor, placed))

        rng.shuffle(options)
        for anchor, placed in options:
            result = rec(
                idx + 1,
                _lr_tb_key(anchor),
                occupied_cells | placed,
                anchors + (anchor,),
            )
            if result is not None:
                return result
        return None

    return rec(0, (-1, -1), set(), tuple())


def _paint_shape(grid: np.ndarray, top_left: Coord, shape: dict, color: int) -> None:
    r0, c0 = top_left
    for dr, dc in shape["coords"]:
        grid[r0 + dr, c0 + dc] = int(color)


def _place_frames(rng: random.Random, h: int, w: int, count: int) -> tuple[list[dict], np.ndarray] | None:
    occupied = np.zeros((h, w), dtype=bool)
    frames: list[dict] = []

    for _ in range(count):
        placed = False
        for _attempt in range(PLACEMENT_RETRIES):
            fh = rng.randint(FRAME_H_RANGE[0], FRAME_H_RANGE[1])
            fw = rng.randint(FRAME_W_RANGE[0], FRAME_W_RANGE[1])
            r0 = rng.randint(1, h - fh - 1)
            c0 = rng.randint(1, w - fw - 1)
            if not _box_is_free(occupied, r0, c0, fh, fw, margin=1):
                continue

            _reserve_box(occupied, r0, c0, fh, fw, margin=1)
            frames.append(
                {
                    "bbox": (r0, c0, r0 + fh - 1, c0 + fw - 1),
                    "h": fh,
                    "w": fw,
                }
            )
            placed = True
            break
        if not placed:
            return None

    return frames, occupied


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_H_RANGE[0], GRID_H_RANGE[1])
    w = rng.randint(GRID_W_RANGE[0], GRID_W_RANGE[1])
    frame_count = rng.randint(FRAME_COUNT_RANGE[0], FRAME_COUNT_RANGE[1])

    placed = _place_frames(rng, h, w, frame_count)
    if placed is None:
        return None
    frames, occupied = placed

    colors = list(PALETTE)
    rng.shuffle(colors)
    colors = colors[:frame_count]

    grid = np.full((h, w), BG, dtype=int)
    fragment_instances: list[dict] = []

    for frame, color in zip(frames, colors):
        n_frag = rng.randint(FRAGMENTS_PER_FRAME_RANGE[0], FRAGMENTS_PER_FRAME_RANGE[1])
        cover_flags = [True] * (n_frag // 2) + [False] * (n_frag // 2)
        if n_frag % 2 == 1:
            cover_flags.append(bool(rng.getrandbits(1)))
        rng.shuffle(cover_flags)
        if not any(cover_flags):
            cover_flags[0] = True
        if all(cover_flags):
            cover_flags[-1] = False

        shapes = [_random_connected_shape(rng, cover_anchor=flag) for flag in cover_flags]
        current: list[dict] = []
        for shape in shapes:
            placed_shape = False
            for _attempt in range(PLACEMENT_RETRIES):
                top = (
                    rng.randint(0, h - int(shape["h"])),
                    rng.randint(0, w - int(shape["w"])),
                )
                if not _box_is_free(occupied, top[0], top[1], int(shape["h"]), int(shape["w"]), margin=1):
                    continue

                _reserve_box(occupied, top[0], top[1], int(shape["h"]), int(shape["w"]), margin=1)
                current.append(
                    {
                        "color": int(color),
                        "shape": shape,
                        "top_left": top,
                    }
                )
                placed_shape = True
                break
            if not placed_shape:
                return None

        ordered = sorted(current, key=lambda inst: _lr_tb_key(inst["top_left"]))
        interior_h = int(frame["h"]) - 2
        interior_w = int(frame["w"]) - 2
        anchors = _assign_anchors(rng, interior_h, interior_w, [inst["shape"] for inst in ordered])
        if anchors is None:
            return None

        r0, c0, r1, c1 = frame["bbox"]
        frame["marker_color"] = int(color)
        frame["markers"] = [(r0 + 1 + ar, c0 + 1 + ac) for ar, ac in anchors]

        fragment_instances.extend(ordered)

        for c in range(c0, c1 + 1):
            grid[r0, c] = FRAME_COLOR
            grid[r1, c] = FRAME_COLOR
        for r in range(r0, r1 + 1):
            grid[r, c0] = FRAME_COLOR
            grid[r, c1] = FRAME_COLOR

    for frame in frames:
        for mr, mc in frame["markers"]:
            grid[mr, mc] = int(frame["marker_color"])

    for inst in fragment_instances:
        _paint_shape(grid, inst["top_left"], inst["shape"], int(inst["color"]))

    parsed_frames = _find_frames(grid)
    if len(parsed_frames) != frame_count:
        return None
    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
