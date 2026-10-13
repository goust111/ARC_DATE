import numpy as np
from collections import Counter, deque
from typing import Dict, Iterable, List, Tuple


DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR_NAMES = ("up", "down", "left", "right")


def _connected_components(grid: np.ndarray, bg: int) -> Tuple[List[Dict], np.ndarray]:
    h, w = grid.shape
    comp_id = -np.ones((h, w), dtype=np.int32)
    comps: List[Dict] = []
    cid = 0

    for r in range(h):
        for c in range(w):
            if grid[r, c] == bg or comp_id[r, c] != -1:
                continue

            q = deque([(r, c)])
            comp_id[r, c] = cid
            cells: List[Tuple[int, int]] = []
            colors = set()
            min_r = min_c = 10**9
            max_r = max_c = -1

            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                color = int(grid[cr, cc])
                colors.add(color)
                min_r = min(min_r, cr)
                max_r = max(max_r, cr)
                min_c = min(min_c, cc)
                max_c = max(max_c, cc)

                for dr, dc in DIRS:
                    nr, nc = cr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and comp_id[nr, nc] == -1
                        and grid[nr, nc] != bg
                    ):
                        comp_id[nr, nc] = cid
                        q.append((nr, nc))

            comps.append(
                {
                    "id": cid,
                    "cells": tuple(cells),
                    "size": len(cells),
                    "colors": colors,
                    "color": next(iter(colors)) if len(colors) == 1 else None,
                    "bbox": (min_r, max_r, min_c, max_c),
                }
            )
            cid += 1

    return comps, comp_id


def _connected_components_same_color(grid: np.ndarray, bg: int) -> Tuple[List[Dict], np.ndarray]:
    h, w = grid.shape
    comp_id = -np.ones((h, w), dtype=np.int32)
    comps: List[Dict] = []
    cid = 0

    for r in range(h):
        for c in range(w):
            if grid[r, c] == bg or comp_id[r, c] != -1:
                continue

            color = int(grid[r, c])
            q = deque([(r, c)])
            comp_id[r, c] = cid
            cells: List[Tuple[int, int]] = []
            min_r = min_c = 10**9
            max_r = max_c = -1

            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                min_r = min(min_r, cr)
                max_r = max(max_r, cr)
                min_c = min(min_c, cc)
                max_c = max(max_c, cc)

                for dr, dc in DIRS:
                    nr, nc = cr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and comp_id[nr, nc] == -1
                        and grid[nr, nc] == color
                    ):
                        comp_id[nr, nc] = cid
                        q.append((nr, nc))

            comps.append(
                {
                    "id": cid,
                    "cells": tuple(cells),
                    "size": len(cells),
                    "colors": {color},
                    "color": color,
                    "bbox": (min_r, max_r, min_c, max_c),
                }
            )
            cid += 1

    return comps, comp_id


def _is_l_shape(cells: Iterable[Tuple[int, int]]) -> bool:
    cells = tuple(cells)
    if len(cells) != 3:
        return False
    s = set(cells)
    degrees: List[int] = []
    for r, c in cells:
        deg = 0
        for dr, dc in DIRS:
            if (r + dr, c + dc) in s:
                deg += 1
        degrees.append(deg)
    degrees.sort()
    return degrees == [1, 1, 2]


def _is_frame_component(
    comp: Dict,
    comp_id: np.ndarray,
    grid: np.ndarray,
    bg: int,
) -> bool:
    cid = comp["id"]
    r1, r2, c1, c2 = comp["bbox"]
    h = r2 - r1 + 1
    w = c2 - c1 + 1

    if h < 3 or w < 3:
        return False

    expected_border = 2 * (h + w) - 4
    if comp["size"] != expected_border:
        return False

    # Border must be fully occupied by this component and non-bg.
    for c in range(c1, c2 + 1):
        if grid[r1, c] == bg or comp_id[r1, c] != cid:
            return False
        if grid[r2, c] == bg or comp_id[r2, c] != cid:
            return False
    for r in range(r1 + 1, r2):
        if grid[r, c1] == bg or comp_id[r, c1] != cid:
            return False
        if grid[r, c2] == bg or comp_id[r, c2] != cid:
            return False

    # Interior must be bg.
    if h > 2 and w > 2:
        for r in range(r1 + 1, r2):
            if np.any(grid[r, c1 + 1 : c2] != bg):
                return False

    return True


def _frame_info(comp: Dict, grid: np.ndarray) -> Dict:
    colors = [int(grid[r, c]) for r, c in comp["cells"]]
    cnt = Counter(colors)
    dom_color, _ = cnt.most_common(1)[0]
    maroon = sum(v for color, v in cnt.items() if color != dom_color)
    r1, r2, c1, c2 = comp["bbox"]

    return {
        "id": comp["id"],
        "bbox": comp["bbox"],
        "inner_h": (r2 - r1 - 1),
        "inner_w": (c2 - c1 - 1),
        "maroon_count": maroon,
        "dominant_color": dom_color,
    }


def _bbox_dims(comp: Dict) -> Tuple[int, int]:
    r1, r2, c1, c2 = comp["bbox"]
    return c2 - c1 + 1, r2 - r1 + 1

def _first_frame_hit_for_cell(
    start: Tuple[int, int],
    direction: int,
    grid: np.ndarray,
    comp_id: np.ndarray,
    frame_ids: set,
    bg: int,
) -> int:
    h, w = grid.shape
    dr, dc = DIRS[direction]
    r, c = start
    nr, nc = r + dr, c + dc
    while 0 <= nr < h and 0 <= nc < w:
        cell_color = int(grid[nr, nc])
        if cell_color == bg:
            nr, nc = nr + dr, nc + dc
            continue

        fid = int(comp_id[nr, nc])
        if fid in frame_ids:
            return fid
        return -1

    return -1


def _frame_repeat_count(
    frame: Dict,
    src_w: int,
    src_h: int,
    direction: int,
) -> int:
    inner_w = frame["inner_w"]
    inner_h = frame["inner_h"]

    if direction in (0, 1):
        if inner_h != src_h:
            return 0
        if inner_w % src_w != 0:
            return 0
        return inner_w // src_w

    if direction in (2, 3):
        if inner_w != src_w:
            return 0
        if inner_h % src_h != 0:
            return 0
        return inner_h // src_h

    return 0


def _paste_tiles(
    out: np.ndarray,
    src_cells: Iterable[Tuple[int, int]],
    src_color: int,
    frame: Dict,
    direction: int,
    copies: int,
) -> None:
    r1, r2, c1, c2 = frame["bbox"]
    src_rows = [r for r, _ in src_cells]
    src_cols = [c for _, c in src_cells]
    src_top = min(src_rows)
    src_left = min(src_cols)

    rel_cells = [(r - src_top, c - src_left) for r, c in src_cells]
    h_src = max(src_rows) - src_top + 1
    w_src = max(src_cols) - src_left + 1

    base_r = r1 + 1
    base_c = c1 + 1

    for i in range(copies):
        if direction in (0, 1):
            rr = base_r
            cc = base_c + i * w_src
        else:
            rr = base_r + i * h_src
            cc = base_c

        for dr, dc in rel_cells:
            out[rr + dr, cc + dc] = src_color


def main(grid: np.ndarray) -> np.ndarray:
    """Apply arrow-frame tiling for ARC-style pattern transfer.

    Args:
        grid: Integer-coded numpy array representing an ARC grid.

    Returns:
        np.ndarray: Grid after copying source objects into framed regions.

    Assumptions:
        Input follows: red L-arrows point to a single orange frame; frame
        side length is a repeating multiple of source size in one axis; maroon
        markers on the frame border indicate how many copies to place.

    Complexity:
        O(H * W) component scan plus per-arrow directional ray checks,
        with H,W <= 30 so this is tiny in practice.
    """
    if not isinstance(grid, np.ndarray):
        raise TypeError("grid must be np.ndarray")

    grid = np.array(grid, dtype=int, copy=True)
    if grid.ndim != 2:
        raise ValueError("grid must be 2D")

    bg = int(Counter(grid.ravel()).most_common(1)[0][0])
    components, comp_id = _connected_components_same_color(grid, bg)
    all_comps, comp_id_all = _connected_components(grid, bg)

    arrows = []
    source_comps = {}
    frame_comps: Dict[int, Dict] = {}
    frame_ids: set = set()

    for comp in components:
        if _is_l_shape(comp["cells"]):
            arrows.append(comp)
            continue

        width, height = _bbox_dims(comp)
        if (
            comp["size"] <= 25
            and 3 <= width <= 5
            and 3 <= height <= 5
        ):
            source_comps[comp["id"]] = comp

    for comp in all_comps:
        if _is_frame_component(comp, comp_id_all, grid, bg):
            f = _frame_info(comp, grid)
            frame_ids.add(comp["id"])
            frame_comps[comp["id"]] = f
            continue

    out = np.array(grid, dtype=int, copy=True)
    h, w = grid.shape

    for a_comp in arrows:
        # Erase arrow later; still use it to establish pairing.
        cells = a_comp["cells"]
        for rr, cc in cells:
            out[rr, cc] = bg

        # Find a neighboring source component.
        neighbor_ids = set()
        for r, c in cells:
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    nid = int(comp_id[nr, nc])
                    if nid in source_comps:
                        neighbor_ids.add(nid)

        if not neighbor_ids:
            continue
        # If multiple neighbors, choose one that is most likely source.
        src_id = sorted(neighbor_ids)[0]
        src = source_comps[src_id]

        src_w, src_h = _bbox_dims(src)
        src_color = src["color"]
        src_cells = src["cells"]

        best_pair = None
        best_score = -1

        for direction in range(4):
            counts: Dict[int, int] = {}
            for cell in cells:
                fid = _first_frame_hit_for_cell(
                    cell,
                    direction,
                    grid,
                    comp_id_all,
                    frame_ids,
                    bg,
                )
                if fid in frame_ids:
                    counts[fid] = counts.get(fid, 0) + 1

            for fid, score in counts.items():
                frame = frame_comps.get(fid)
                if frame is None:
                    continue

                repeat_total = _frame_repeat_count(frame, src_w, src_h, direction)
                if repeat_total <= 0:
                    continue
                maroon = frame["maroon_count"]
                if not (1 <= maroon <= 4):
                    continue
                copies = min(repeat_total, maroon)
                if copies <= 0:
                    continue

                score_value = score
                if score_value > best_score:
                    best_score = score_value
                    best_pair = (fid, direction, copies)

        if best_pair is None:
            continue

        fid, direction, copies = best_pair
        frame = frame_comps[fid]
        _paste_tiles(out, src_cells, int(src_color), frame, direction, copies)

    return out


def generate_input() -> np.ndarray:
    """Generate one random valid input that respects the described ARC rule set."""
    rng = np.random.default_rng()
    bg = 0
    red = 1
    orange = 2
    maroon = 3

    h = int(rng.integers(14, 28))
    w = int(rng.integers(14, 28))
    # keep within 30x30
    h = min(30, h)
    w = min(30, w)
    grid = np.full((h, w), bg, dtype=int)
    occ = np.zeros((h, w), dtype=bool)

    def _can_place(coords: Iterable[Tuple[int, int]]) -> bool:
        for rr, cc in coords:
            if rr < 0 or rr >= h or cc < 0 or cc >= w:
                return False
            if occ[rr, cc]:
                return False
        return True

    def _mark(coords: Iterable[Tuple[int, int]], color: int) -> None:
        for rr, cc in coords:
            grid[rr, cc] = color
            occ[rr, cc] = True

    def _border_coords(r0: int, r1: int, c0: int, c1: int) -> List[Tuple[int, int]]:
        pts: List[Tuple[int, int]] = []
        for cc in range(c0, c1 + 1):
            pts.append((r0, cc))
            if r1 != r0:
                pts.append((r1, cc))
        for rr in range(r0 + 1, r1):
            pts.append((rr, c0))
            if c1 != c0:
                pts.append((rr, c1))
        return pts

    num_pairs = int(rng.integers(1, 4))
    next_color = 4

    for _ in range(num_pairs):
        success = False
        src_color = next_color
        next_color += 1

        for _ in range(2000):
            obj_h = int(rng.integers(3, 6))
            obj_w = int(rng.integers(3, 6))
            repeats = int(rng.integers(2, 5))
            direction = int(rng.integers(0, 4))

            if direction in (0, 1):
                inner_h = obj_h
                inner_w = obj_w * repeats
            else:
                inner_h = obj_h * repeats
                inner_w = obj_w

            frame_h = inner_h + 2
            frame_w = inner_w + 2

            obj_r = int(rng.integers(0, h - obj_h))
            obj_c = int(rng.integers(0, w - obj_w))

            if direction == 0:  # up
                if obj_r < 2:
                    continue
                arrow_r = obj_r - 2
                arrow_c = obj_c
                frame_r1 = obj_r - 3
                frame_r0 = frame_r1 - frame_h + 1
                frame_c0 = obj_c
                frame_c1 = frame_c0 + frame_w - 1
            elif direction == 1:  # down
                if obj_r + obj_h + 2 >= h:
                    continue
                arrow_r = obj_r + obj_h
                arrow_c = obj_c
                frame_r0 = obj_r + obj_h + 2
                frame_r1 = frame_r0 + frame_h - 1
                frame_c0 = obj_c
                frame_c1 = frame_c0 + frame_w - 1
            elif direction == 2:  # left
                if obj_c < 2:
                    continue
                arrow_r = obj_r
                arrow_c = obj_c - 2
                frame_c1 = obj_c - 3
                frame_c0 = frame_c1 - frame_w + 1
                frame_r0 = obj_r
                frame_r1 = frame_r0 + frame_h - 1
            else:  # right
                if obj_c + obj_w + 2 >= w:
                    continue
                arrow_r = obj_r
                arrow_c = obj_c + obj_w
                frame_c0 = obj_c + obj_w + 2
                frame_c1 = frame_c0 + frame_w - 1
                frame_r0 = obj_r
                frame_r1 = frame_r0 + frame_h - 1

            # Common bounds and coordinate checks
            if not (0 <= arrow_r < h and 0 <= arrow_c + 1 < w):
                continue
            if frame_c0 < 0 or frame_c1 >= w or frame_r0 < 0 or frame_r1 >= h:
                continue

            # Build arrow and object candidate coordinates.
            arrow_cells = [
                (arrow_r, arrow_c),
                (arrow_r + 1, arrow_c),
                (arrow_r, arrow_c + 1),
            ]
            obj_cells = [
                (rr, cc) for rr in range(obj_r, obj_r + obj_h) for cc in range(obj_c, obj_c + obj_w)
            ]
            frame_cells = _border_coords(frame_r0, frame_r1, frame_c0, frame_c1)

            if not _can_place(arrow_cells + obj_cells + frame_cells):
                continue

            # Place object.
            _mark(obj_cells, src_color)

            # Place arrow.
            _mark(arrow_cells, red)

            # Place frame border in orange.
            _mark(frame_cells, orange)

            # Add 1-4 maroon replacements on the border.
            m_cnt = int(rng.integers(1, min(5, len(frame_cells) + 1)))
            idxs = rng.choice(len(frame_cells), size=m_cnt, replace=False)
            for idx in idxs:
                rr, cc = frame_cells[idx]
                grid[rr, cc] = maroon

            success = True
            break

        if not success:
            # If placement failed repeatedly, skip this pair to keep generation moving.
            continue

    return grid


if __name__ == "__main__":
    sample = generate_input()
    print(sample.shape)
    out = main(sample)
    print(out.shape)
