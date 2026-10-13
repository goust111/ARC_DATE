from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
TEAL = 8

BODY_COLORS = [3, 5, 6, 7, 9]
GEN_RETRIES = 400


@dataclass(frozen=True)
class Piece:
    arr: np.ndarray
    bbox: tuple[int, int, int, int]
    reds: tuple[tuple[int, int], ...]
    area: int

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(map(int, self.arr.shape))


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.array(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)


def _piece_from_component(grid: np.ndarray, coords: list[tuple[int, int]]) -> Piece:
    bbox = _bbox(coords)
    arr = _crop(grid, bbox)
    reds = tuple(map(tuple, np.argwhere(arr == RED)))
    return Piece(arr=arr, bbox=bbox, reds=reds, area=int(np.count_nonzero(arr)))


def _extract_objects(grid: np.ndarray) -> list[Piece]:
    return [_piece_from_component(grid, comp) for comp in _components(grid != BLACK)]


def _is_template(piece: Piece) -> bool:
    return piece.arr.shape == (2, 2) and bool(np.all(piece.arr == TEAL))


def _find_anchor_piece(pieces: list[Piece]) -> Piece:
    anchors = [p for p in pieces if np.any(p.arr == YELLOW)]
    assert len(anchors) == 1, "expected exactly one anchor piece"
    yellow = np.argwhere(anchors[0].arr == YELLOW)
    yr0, yc0 = yellow.min(axis=0)
    yr1, yc1 = yellow.max(axis=0)
    assert (int(yr1 - yr0 + 1), int(yc1 - yc0 + 1)) == (2, 2), "anchor must contain a 2x2 yellow block"
    assert np.all(anchors[0].arr[int(yr0) : int(yr1) + 1, int(yc0) : int(yc1) + 1] == YELLOW)
    return anchors[0]


def _can_place(
    canvas: np.ndarray,
    piece: Piece,
    top: int,
    left: int,
    red_counts: dict[tuple[int, int], int],
) -> tuple[bool, int]:
    h, w = canvas.shape
    ph, pw = piece.shape
    if top < 0 or left < 0 or top + ph > h or left + pw > w:
        return False, 0

    overlap_unpaired = 0
    for rr, cc in np.argwhere(piece.arr != BLACK):
        rr = int(rr)
        cc = int(cc)
        pv = int(piece.arr[rr, cc])
        gv = int(canvas[top + rr, left + cc])
        if pv == RED:
            if gv not in (BLACK, RED):
                return False, 0
            prev = red_counts.get((top + rr, left + cc), 0)
            if prev >= 2:
                return False, 0
            if prev == 1:
                overlap_unpaired += 1
        else:
            if gv != BLACK:
                return False, 0

    return True, overlap_unpaired


def _place_piece(canvas: np.ndarray, piece: Piece, top: int, left: int) -> np.ndarray:
    out = canvas.copy()
    for rr, cc in np.argwhere(piece.arr != BLACK):
        rr = int(rr)
        cc = int(cc)
        out[top + rr, left + cc] = int(piece.arr[rr, cc])
    return out


def _solve_assembly(grid: np.ndarray) -> tuple[np.ndarray, int]:
    objects = _extract_objects(grid)
    templates = [obj for obj in objects if _is_template(obj)]
    assert len(templates) == 1, "expected exactly one detached 2x2 teal template"
    pieces = [obj for obj in objects if obj is not templates[0]]
    assert 4 <= len(pieces) <= 6, "expected 4-6 sprites including the anchor"

    anchor = _find_anchor_piece(pieces)
    remaining = [p for p in pieces if p is not anchor]
    canvas = np.full_like(grid, BLACK)
    ar0, ac0, _, _ = anchor.bbox
    canvas = _place_piece(canvas, anchor, ar0, ac0)

    red_counts: dict[tuple[int, int], int] = {}
    for rr, cc in anchor.reds:
        red_counts[(ar0 + rr, ac0 + cc)] = red_counts.get((ar0 + rr, ac0 + cc), 0) + 1

    total_reds = sum(len(p.reds) for p in pieces)
    assert total_reds % 2 == 0, "connector count must be even"
    pair_count = total_reds // 2

    solutions: list[np.ndarray] = []
    seen_signatures: set[bytes] = set()

    def dfs(cur: np.ndarray, rest: list[Piece], rc: dict[tuple[int, int], int]) -> None:
        if len(solutions) >= 2:
            return
        if not rest:
            if rc and all(v == 2 for v in rc.values()):
                sig = cur.tobytes()
                if sig not in seen_signatures:
                    seen_signatures.add(sig)
                    solutions.append(cur.copy())
            return

        unpaired = [coord for coord, count in rc.items() if count == 1]
        if not unpaired:
            return

        best_moves: list[tuple[int, int, int, int]] | None = None
        for target_r, target_c in unpaired:
            moves: list[tuple[int, int, int, int]] = []
            for idx, piece in enumerate(rest):
                for pr, pc in piece.reds:
                    top = target_r - pr
                    left = target_c - pc
                    ok, overlap_count = _can_place(cur, piece, top, left, rc)
                    if ok and overlap_count >= 1:
                        moves.append((idx, top, left, overlap_count))
            if not moves:
                return
            if best_moves is None or len(moves) < len(best_moves):
                best_moves = moves
                if len(best_moves) == 1:
                    break

        assert best_moves is not None
        dedup: dict[tuple[int, int, int], int] = {}
        for idx, top, left, overlap_count in best_moves:
            key = (idx, top, left)
            dedup[key] = max(dedup.get(key, 0), overlap_count)
        move_list = [(idx, top, left, ov) for (idx, top, left), ov in dedup.items()]
        move_list.sort(key=lambda item: (-item[3], -rest[item[0]].area, item[1], item[2], item[0]))

        for idx, top, left, _ in move_list:
            piece = rest[idx]
            next_rc = dict(rc)
            for rr, cc in piece.reds:
                pos = (top + rr, left + cc)
                next_rc[pos] = next_rc.get(pos, 0) + 1
                if next_rc[pos] > 2:
                    break
            else:
                next_canvas = _place_piece(cur, piece, top, left)
                next_rest = rest[:idx] + rest[idx + 1 :]
                dfs(next_canvas, next_rest, next_rc)

    dfs(canvas, remaining, red_counts)
    assert len(solutions) == 1, "expected a unique valid assembly"
    return solutions[0], pair_count


def _fill_enclosed_cavities(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    outside = np.zeros((h, w), dtype=bool)
    q = deque()
    for r in range(h):
        for c in (0, w - 1):
            if g[r, c] == BLACK and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if g[r, c] == BLACK and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and g[nr, nc] == BLACK and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    out = g.copy()
    enclosed = (g == BLACK) & (~outside)
    out[enclosed] = BLUE
    return out


def _stamp_scaled_template(grid: np.ndarray, scale: int) -> np.ndarray:
    out = np.array(grid, dtype=int)
    yellow = np.argwhere(out == YELLOW)
    yr0, yc0 = yellow.min(axis=0)
    stamp_h = stamp_w = 2 * int(scale)
    for rr in range(stamp_h):
        for cc in range(stamp_w):
            gr = int(yr0 + rr)
            gc = int(yc0 + cc)
            if 0 <= gr < out.shape[0] and 0 <= gc < out.shape[1] and out[gr, gc] in (BLACK, BLUE):
                out[gr, gc] = TEAL
    return out


def _crop_nonblack(grid: np.ndarray) -> np.ndarray:
    coords = np.argwhere(grid != BLACK)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return np.array(grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1], dtype=int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble connector-linked sprites, fill the enclosed cavity blue, then stamp a scaled teal template at the yellow anchor.

    Input structure:
    - A black background contains 4-6 scattered sprites plus one detached 2x2 teal template.
    - Exactly one sprite is the anchor: it contains a 2x2 yellow block; every sprite also has red connector pixels on its boundary.
    - The detached teal block is not part of the assembly and the non-anchor sprites can only be translated, not rotated.

    Rule:
    - Keep the anchor sprite fixed and assemble the remaining sprites by translating them so red connector pixels pair up, with overlaps allowed only on red cells.
    - After the unique full assembly is formed, flood-fill every black region enclosed by the assembled non-black pixels with blue.
    - Let k be the number of red connector pairs; stamp the teal 2x2 template scaled by factor k at the yellow block's top-left corner, allowing teal to overwrite only black or blue cells, then crop to the final non-black bounding box.

    Key insight:
    - The connector assembly simultaneously determines the geometry of the enclosed cavity and the connector-pair count that sets the teal stamp scale, so both effects can only be resolved after reconstructing the full composite object.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    assembled, pair_count = _solve_assembly(g)
    filled = _fill_enclosed_cavities(assembled)
    stamped = _stamp_scaled_template(filled, pair_count)
    return _crop_nonblack(stamped)


def _rect_cycle(h: int, w: int, left_conn_r: int) -> list[tuple[int, int]]:
    cycle: list[tuple[int, int]] = []
    for r in range(left_conn_r, -1, -1):
        cycle.append((r, 0))
    for c in range(1, w):
        cycle.append((0, c))
    for r in range(1, h):
        cycle.append((r, w - 1))
    for c in range(w - 2, -1, -1):
        cycle.append((h - 1, c))
    for r in range(h - 2, left_conn_r, -1):
        cycle.append((r, 0))
    return cycle


def _random_partition(total: int, parts: int, lo: int, hi: int) -> list[int] | None:
    if not (lo * parts <= total <= hi * parts):
        return None
    vals = [lo] * parts
    rem = total - lo * parts
    caps = [hi - lo] * parts
    while rem > 0:
        idxs = [i for i, cap in enumerate(caps) if cap > 0]
        if not idxs:
            return None
        idx = random.choice(idxs)
        add = random.randint(1, min(caps[idx], rem))
        vals[idx] += add
        caps[idx] -= add
        rem -= add
    random.shuffle(vals)
    return vals


def _build_piece_arrays(
    piece_cells: list[dict[tuple[int, int], int]],
) -> list[np.ndarray]:
    arrays: list[np.ndarray] = []
    for cell_map in piece_cells:
        coords = list(cell_map)
        r0, c0, r1, c1 = _bbox(coords)
        arr = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
        for (r, c), color in cell_map.items():
            arr[r - r0, c - c0] = int(color)
        arrays.append(arr)
    return arrays


def _assembled_piece_blueprint() -> tuple[list[np.ndarray], np.ndarray]:
    piece_count = random.randint(4, 6)
    for _ in range(200):
        h = random.randint(2 * piece_count + 2, min(16, 2 * piece_count + 4))
        w = random.randint(2 * piece_count + 2, min(16, 2 * piece_count + 5))
        left_conn_r = random.randint(4, min(h - 2, 6))
        top_conn_c = random.randint(4, min(w - 2, 6))
        anchor_len = left_conn_r + top_conn_c - 1
        anchor_main = anchor_len - 3
        if not (4 <= anchor_main <= 10):
            continue

        cycle = _rect_cycle(h, w, left_conn_r)
        perimeter = len(cycle)
        remaining_total = perimeter - piece_count - anchor_len
        rest_lengths = _random_partition(remaining_total, piece_count - 1, 4, 10)
        if rest_lengths is None:
            continue

        connectors = [0, anchor_len + 1]
        cur = connectors[-1]
        for seg_len in rest_lengths[:-1]:
            cur += seg_len + 1
            connectors.append(cur)
        if cur + rest_lengths[-1] + 1 != perimeter:
            continue

        colors = random.sample(BODY_COLORS, k=min(len(BODY_COLORS), piece_count))
        while len(colors) < piece_count:
            colors.append(random.choice(BODY_COLORS))
        random.shuffle(colors)

        piece_maps: list[dict[tuple[int, int], int]] = []
        assembly = np.zeros((h, w), dtype=int)

        segment_lengths = [anchor_len] + rest_lengths
        for idx in range(piece_count):
            start = connectors[idx]
            seg_len = segment_lengths[idx]
            end = start + seg_len + 1
            connector_a = cycle[start]
            connector_b = cycle[end % perimeter]
            internal = cycle[start + 1 : end]
            cell_map: dict[tuple[int, int], int] = {connector_a: RED, connector_b: RED}
            color = colors[idx]
            if idx == 0:
                yellow_cells = {(0, 0), (0, 1), (1, 0), (1, 1)}
                for cell in internal:
                    cell_map[cell] = YELLOW if cell in yellow_cells else color
                cell_map[(1, 1)] = YELLOW
            else:
                for cell in internal:
                    cell_map[cell] = color
            piece_maps.append(cell_map)

        for idx, cell_map in enumerate(piece_maps):
            for (r, c), color in cell_map.items():
                prev = int(assembly[r, c])
                if prev not in (BLACK, RED) and color == RED:
                    raise AssertionError("invalid connector overlap in blueprint")
                if prev not in (BLACK, RED) and color != RED:
                    raise AssertionError("invalid non-red overlap in blueprint")
                if prev == RED and color != RED:
                    raise AssertionError("non-red cannot overwrite a connector")
                assembly[r, c] = int(color)

        if np.count_nonzero(assembly == BLACK) == 0:
            continue
        if np.count_nonzero(assembly == RED) != piece_count:
            continue
        arrays = _build_piece_arrays(piece_maps)
        return arrays, assembly
    raise RuntimeError("failed to build a valid assembled blueprint")


def _far_from_occupied(
    occupied: np.ndarray,
    coords: list[tuple[int, int]],
    top: int,
    left: int,
    gap: int,
) -> bool:
    h, w = occupied.shape
    for rr, cc in coords:
        gr = top + rr
        gc = left + cc
        if not (0 <= gr < h and 0 <= gc < w):
            return False
        for dr in range(-gap, gap + 1):
            for dc in range(-gap, gap + 1):
                nr = gr + dr
                nc = gc + dc
                if 0 <= nr < h and 0 <= nc < w and occupied[nr, nc]:
                    return False
    return True


def _paint_piece(grid: np.ndarray, arr: np.ndarray, top: int, left: int) -> None:
    for rr, cc in np.argwhere(arr != BLACK):
        grid[top + int(rr), left + int(cc)] = int(arr[int(rr), int(cc)])


def _choose_template_corner(h: int, w: int) -> tuple[int, int]:
    return random.choice([(0, 0), (0, w - 2), (h - 2, 0), (h - 2, w - 2)])


def _gen_attempt() -> np.ndarray | None:
    try:
        piece_arrays, assembly = _assembled_piece_blueprint()
    except RuntimeError:
        return None
    ah, aw = assembly.shape
    grid_h = random.randint(max(22, ah + 8), min(30, ah + 14))
    grid_w = random.randint(max(22, aw + 8), min(30, aw + 14))
    grid = np.zeros((grid_h, grid_w), dtype=int)
    occupied = np.zeros_like(grid, dtype=bool)

    template_r, template_c = _choose_template_corner(grid_h, grid_w)
    template_cells = [(template_r + dr, template_c + dc) for dr in range(2) for dc in range(2)]
    for r, c in template_cells:
        occupied[r, c] = True
        grid[r, c] = TEAL

    anchor_arr = piece_arrays[0]
    anchor_coords = [tuple(map(int, rc)) for rc in np.argwhere(anchor_arr != BLACK)]

    for _ in range(200):
        top = random.randint(2, grid_h - anchor_arr.shape[0] - 3)
        left = random.randint(2, grid_w - anchor_arr.shape[1] - 3)
        if _far_from_occupied(occupied, anchor_coords, top, left, gap=2):
            _paint_piece(grid, anchor_arr, top, left)
            for rr, cc in anchor_coords:
                occupied[top + rr, left + cc] = True
            break
    else:
        return None

    for arr in piece_arrays[1:]:
        coords = [tuple(map(int, rc)) for rc in np.argwhere(arr != BLACK)]
        for _ in range(300):
            top = random.randint(0, grid_h - arr.shape[0])
            left = random.randint(0, grid_w - arr.shape[1])
            if _far_from_occupied(occupied, coords, top, left, gap=2):
                _paint_piece(grid, arr, top, left)
                for rr, cc in coords:
                    occupied[top + rr, left + cc] = True
                break
        else:
            return None

    try:
        out = main(grid)
    except (AssertionError, RuntimeError):
        return None

    if TEAL not in out or BLUE not in out or YELLOW not in out or RED not in out:
        return None
    if max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
