from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
RED = 2
YELLOW = 4
FRAME_COLORS = (1, 3, 5, 6, 7, 8, 9)
GEN_RETRIES = 500


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, box)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = [(r, c)]
            seen[r, c] = True
            pts = []
            while q:
                rr, cc = q.pop()
                pts.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def find_components(
    grid: np.ndarray,
    *,
    bg: int = BG,
    monochromatic: bool = False,
) -> list[dict]:
    if monochromatic:
        colors = sorted(set(map(int, grid.reshape(-1))) - {int(bg)})
        out = []
        for color in colors:
            for coords in components_mask(grid == int(color)):
                out.append(
                    {
                        "color": int(color),
                        "coords": coords,
                        "bbox": bbox_coords(coords),
                        "area": int(coords.shape[0]),
                    }
                )
        return out

    mask = grid != int(bg)
    return [
        {
            "coords": coords,
            "bbox": bbox_coords(coords),
            "area": int(coords.shape[0]),
        }
        for coords in components_mask(mask)
    ]


def _extract_pieces(grid: np.ndarray) -> list[dict]:
    pieces = []
    for comp in find_components(grid, bg=BG, monochromatic=False):
        r0, c0, r1, c1 = comp["bbox"]
        arr = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        reds = np.argwhere(arr == RED)
        pieces.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "arr": arr,
                "reds": [(int(r), int(c)) for r, c in reds],
                "shape": (int(arr.shape[0]), int(arr.shape[1])),
                "area": int(np.sum(arr != BG)),
            }
        )
    return pieces


def _try_place(canvas: np.ndarray, arr: np.ndarray, top: int, left: int) -> np.ndarray | None:
    h, w = canvas.shape
    ph, pw = arr.shape
    if top < 0 or left < 0 or top + ph > h or left + pw > w:
        return None

    sub = canvas[top : top + ph, left : left + pw]
    mask = arr != BG
    conflict = mask & (sub != BG) & ~((arr == RED) & (sub == RED))
    if np.any(conflict):
        return None

    out = canvas.copy()
    patch = out[top : top + ph, left : left + pw]
    patch[mask] = arr[mask]
    return out


def _assemble_unique(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    pieces = _extract_pieces(g)
    assert pieces, "expected at least one piece"

    anchors = [p for p in pieces if np.any(p["arr"] == YELLOW)]
    assert len(anchors) == 1, "expected exactly one yellow anchor piece"
    anchor = anchors[0]
    rest = [p for p in pieces if p is not anchor]

    canvas = np.full((h, w), BG, dtype=int)
    ar0, ac0, _, _ = anchor["bbox"]
    placed = _try_place(canvas, anchor["arr"], ar0, ac0)
    assert placed is not None

    total_ports = int(sum(len(p["reds"]) for p in pieces))
    assert total_ports % 2 == 0
    target_unique_red = total_ports // 2

    red_counts: dict[tuple[int, int], int] = {}
    for rr, cc in anchor["reds"]:
        red_counts[(ar0 + rr, ac0 + cc)] = red_counts.get((ar0 + rr, ac0 + cc), 0) + 1

    solutions: list[np.ndarray] = []

    def dfs(cur: np.ndarray, remaining: list[dict], rc: dict[tuple[int, int], int]) -> None:
        if len(solutions) >= 2:
            return
        if not remaining:
            if len(rc) == target_unique_red and all(v == 2 for v in rc.values()):
                solutions.append(cur)
            return
        if len(rc) > target_unique_red:
            return

        unpaired = [coord for coord, count in rc.items() if count == 1]
        if not unpaired:
            return

        open_set = set(unpaired)
        best_moves = None
        for target in unpaired:
            tr, tc = target
            moves = []
            for idx, piece in enumerate(remaining):
                ph, pw = piece["shape"]
                for pr, pc in piece["reds"]:
                    top = tr - pr
                    left = tc - pc
                    if top < 0 or left < 0 or top + ph > h or left + pw > w:
                        continue

                    overlap = 0
                    ok = True
                    for rr, cc in piece["reds"]:
                        gr, gc = top + rr, left + cc
                        prev = rc.get((gr, gc), 0)
                        if prev >= 2:
                            ok = False
                            break
                        if (gr, gc) in open_set:
                            overlap += 1
                    if ok:
                        moves.append((idx, top, left, overlap))

            if not moves:
                return
            if best_moves is None or len(moves) < len(best_moves):
                best_moves = moves
                if len(best_moves) == 1:
                    break

        assert best_moves is not None
        dedup: dict[tuple[int, int, int], int] = {}
        for idx, top, left, overlap in best_moves:
            key = (idx, top, left)
            dedup[key] = max(dedup.get(key, 0), overlap)

        moves2 = [(idx, top, left, overlap) for (idx, top, left), overlap in dedup.items()]
        moves2.sort(key=lambda x: (-x[3], -remaining[x[0]]["area"], x[1], x[2], x[0]))

        for idx, top, left, _ in moves2:
            piece = remaining[idx]
            nxt = _try_place(cur, piece["arr"], top, left)
            if nxt is None:
                continue

            next_counts = dict(rc)
            for rr, cc in piece["reds"]:
                key = (top + rr, left + cc)
                next_counts[key] = next_counts.get(key, 0) + 1

            dfs(nxt, remaining[:idx] + remaining[idx + 1 :], next_counts)
            if len(solutions) >= 2:
                return

    dfs(placed, rest, red_counts)
    assert len(solutions) == 1, "connector pairing must determine a unique assembly"
    return solutions[0]


def _replace_red_connectors(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for r, c in np.argwhere(out == RED):
        neighbor_colors = {
            int(out[nr, nc])
            for nr, nc in neighbors4(int(r), int(c))
            if 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1] and out[nr, nc] not in (BG, RED, YELLOW)
        }
        assert len(neighbor_colors) == 1, "every connector must belong to exactly one frame color"
        out[r, c] = next(iter(neighbor_colors))
    return out


def _frame_from_coords(grid: np.ndarray, coords: np.ndarray, color: int) -> dict | None:
    r0, c0, r1, c1 = bbox_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    if int(coords.shape[0]) != 2 * (h + w) - 4:
        return None

    mask = np.zeros(grid.shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    for rr in range(r0, r1 + 1):
        for cc in range(c0, c1 + 1):
            on_border = rr in (r0, r1) or cc in (c0, c1)
            if bool(mask[rr, cc]) != on_border:
                return None
    return {"color": int(color), "coords": coords, "bbox": (r0, c0, r1, c1)}


def _find_frames(grid: np.ndarray) -> list[dict]:
    frames = []
    for color in sorted(set(map(int, grid.reshape(-1))) - {BG, RED, YELLOW}):
        for coords in components_mask(grid == int(color)):
            frame = _frame_from_coords(grid, coords, int(color))
            if frame is not None:
                frames.append(frame)
    assert frames, "expected assembled rectangles after connector cleanup"
    return frames


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return outer[0] < inner[0] and outer[1] < inner[1] and outer[2] > inner[2] and outer[3] > inner[3]


def _recolor_by_outermost(grid: np.ndarray, frames: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()

    def area(box: tuple[int, int, int, int]) -> int:
        return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)

    for frame in frames:
        containers = [other for other in frames if other is not frame and _contains(other["bbox"], frame["bbox"])]
        if not containers:
            continue
        outer = max(containers, key=lambda item: area(item["bbox"]))
        coords = frame["coords"]
        out[coords[:, 0], coords[:, 1]] = int(outer["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble the scattered red-connected frame fragments, then recolor every inner frame to its outermost container's color.

    Input structure:
    - The background is black (0).
    - One anchor sprite contains a 2x2 yellow (4) block, and the remaining sprites are scattered multicolor fragments.
    - Each sprite is 4-connected and carries one-color rectangular-frame fragments plus red (2) connector pixels on its ends.

    Rule:
    - Keep the yellow-anchor sprite fixed and translate the other sprites so each red connector overlaps exactly one unpaired red connector in the growing assembly.
    - After all sprites are assembled, replace every overlapped red connector with the unique adjacent frame color to recover complete nested rectangular borders.
    - Find each rectangle frame and recolor every frame that is strictly inside another frame with the color of its outermost containing frame, while leaving the yellow block and the black background unchanged.

    Key insight:
    - The nesting hierarchy does not exist in the scattered input; it appears only after the connector-constrained assembly, so the recoloring stage must operate on the assembled structure rather than on the original pieces.
    """
    assembled = _assemble_unique(np.asarray(grid, dtype=int))
    cleaned = _replace_red_connectors(assembled)
    frames = _find_frames(cleaned)
    return _recolor_by_outermost(cleaned, frames)


def _perimeter_path(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, box)
    path = [(r0, c) for c in range(c0, c1 + 1)]
    path.extend((r, c1) for r in range(r0 + 1, r1))
    path.extend((r1, c) for c in range(c1, c0 - 1, -1))
    path.extend((r, c0) for r in range(r1 - 1, r0, -1))
    return path


def _random_arc_lengths(total: int, count: int, lo: int = 4, hi: int = 11) -> list[int] | None:
    order = list(range(count))
    random.shuffle(order)
    vals = [0] * count

    def dfs(idx: int, remaining: int) -> bool:
        if idx == count:
            return remaining == 0
        slot = order[idx]
        rest = count - idx - 1
        low = max(lo, remaining - hi * rest)
        high = min(hi, remaining - lo * rest)
        if low > high:
            return False
        choices = list(range(low, high + 1))
        random.shuffle(choices)
        for value in choices:
            vals[slot] = value
            if dfs(idx + 1, remaining - value):
                return True
        return False

    if dfs(0, int(total)):
        return vals
    return None


def _segment_coords(path: list[tuple[int, int]], start: int, arc_len: int) -> list[tuple[int, int]]:
    n = len(path)
    return [path[(start + step) % n] for step in range(arc_len + 1)]


def _edge_cells(box: tuple[int, int, int, int]) -> dict[str, list[tuple[int, int]]]:
    r0, c0, r1, c1 = map(int, box)
    return {
        "top": [(r0, c) for c in range(c0 + 1, c1)],
        "right": [(r, c1) for r in range(r0 + 1, r1)],
        "bottom": [(r1, c) for c in range(c1 - 1, c0, -1)],
        "left": [(r, c0) for r in range(r1 - 1, r0, -1)],
    }


def _frame_pieces(box: tuple[int, int, int, int], color: int, piece_count: int) -> list[dict] | None:
    path = _perimeter_path(box)
    idx_of = {coord: idx for idx, coord in enumerate(path)}
    r0, c0, r1, c1 = map(int, box)
    corners = {(r0, c0), (r0, c1), (r1, c1), (r1, c0)}
    edges = _edge_cells(box)

    if piece_count == 2:
        side_sets = [("top", "bottom"), ("left", "right")]
    else:
        side_sets = [
            ("top", "right", "bottom"),
            ("right", "bottom", "left"),
            ("bottom", "left", "top"),
            ("left", "top", "right"),
            ("top", "right", "left"),
            ("top", "bottom", "left"),
        ]

    for _ in range(400):
        chosen_sides = random.choice(side_sets)
        if any(not edges[side] for side in chosen_sides):
            continue
        cut_coords = [random.choice(edges[side]) for side in chosen_sides]
        cut_idxs = sorted(idx_of[coord] for coord in cut_coords)

        pieces = []
        lengths = []
        valid = True
        for i, start in enumerate(cut_idxs):
            end = cut_idxs[(i + 1) % len(cut_idxs)]
            arc = (end - start) % len(path)
            if not (4 <= arc <= 11):
                valid = False
                break
            coords = _segment_coords(path, start, arc)
            if not any(coord in corners for coord in coords[1:-1]):
                valid = False
                break
            piece = {
                "frame_color": int(color),
                "path": coords,
                "connectors": {coords[0], coords[-1]},
                "length": len(coords),
            }
            pieces.append(piece)
            lengths.append(len(coords))

        if not valid or len(set(lengths)) != len(lengths):
            continue
        return pieces
    return None


def _yellow_block_cells(
    cell: tuple[int, int],
    side: str,
    *,
    inside: bool,
) -> list[tuple[int, int]]:
    r, c = map(int, cell)
    if side == "top":
        if inside:
            return [(r + 1, c), (r + 1, c + 1), (r + 2, c), (r + 2, c + 1)]
        return [(r - 2, c), (r - 2, c + 1), (r - 1, c), (r - 1, c + 1)]
    if side == "bottom":
        if inside:
            return [(r - 2, c), (r - 2, c + 1), (r - 1, c), (r - 1, c + 1)]
        return [(r + 1, c), (r + 1, c + 1), (r + 2, c), (r + 2, c + 1)]
    if side == "left":
        if inside:
            return [(r, c + 1), (r + 1, c + 1), (r, c + 2), (r + 1, c + 2)]
        return [(r, c - 2), (r + 1, c - 2), (r, c - 1), (r + 1, c - 1)]
    if inside:
        return [(r, c - 2), (r + 1, c - 2), (r, c - 1), (r + 1, c - 1)]
    return [(r, c + 1), (r + 1, c + 1), (r, c + 2), (r + 1, c + 2)]


def _anchor_candidates(
    piece: dict,
    frame_box: tuple[int, int, int, int],
    *,
    inside: bool,
    occupied: set[tuple[int, int]],
    shape: tuple[int, int],
) -> list[list[tuple[int, int]]]:
    r0, c0, r1, c1 = map(int, frame_box)
    h, w = map(int, shape)
    candidates = []
    for cell in piece["path"][1:-1]:
        r, c = cell
        side = None
        if r == r0 and c0 + 1 <= c <= c1 - 2:
            side = "top"
        elif r == r1 and c0 + 1 <= c <= c1 - 2:
            side = "bottom"
        elif c == c0 and r0 + 1 <= r <= r1 - 2:
            side = "left"
        elif c == c1 and r0 + 1 <= r <= r1 - 2:
            side = "right"
        if side is None:
            continue

        block = _yellow_block_cells(cell, side, inside=inside)
        ok = True
        for br, bc in block:
            if not (0 <= br < h and 0 <= bc < w):
                ok = False
                break
            if (br, bc) in occupied:
                ok = False
                break
        if ok:
            candidates.append(block)
    return candidates


def _piece_to_patch(cells: dict[tuple[int, int], int]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    coords = np.array(list(cells), dtype=int)
    box = bbox_coords(coords)
    r0, c0, r1, c1 = box
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    for (r, c), value in cells.items():
        patch[r - r0, c - c0] = int(value)
    return patch, box


def _draw_frame(canvas: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = map(int, box)
    canvas[r0, c0 : c1 + 1] = int(color)
    canvas[r1, c0 : c1 + 1] = int(color)
    canvas[r0 : r1 + 1, c0] = int(color)
    canvas[r0 : r1 + 1, c1] = int(color)


def _choose_layout() -> dict | None:
    return None


def _split_open_chain(
    chain: list[tuple[int, int]],
    corners: set[tuple[int, int]],
) -> list[list[tuple[int, int]]] | None:
    total = len(chain) + 2
    options = []
    for l1 in range(5, 13):
        for l2 in range(5, 13):
            l3 = total - l1 - l2
            if not (5 <= l3 <= 12):
                continue
            if len({l1, l2, l3}) < 2:
                continue
            pieces = [
                chain[:l1],
                chain[l1 - 1 : l1 + l2 - 1],
                chain[l1 + l2 - 2 :],
            ]
            if any(not any(coord in corners for coord in piece[1:-1]) for piece in pieces):
                continue
            options.append(pieces)
    if not options:
        return None
    return random.choice(options)


def _instance_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    outer_h, outer_w = random.choice([(7, 8), (8, 8), (8, 9)])
    outer_color, inner_color = random.sample(FRAME_COLORS, 2)

    base_h = outer_h + 4
    base_w = outer_w + 4
    outer_box = (2, 2, outer_h + 1, outer_w + 1)
    outer_r0, outer_c0, outer_r1, outer_c1 = outer_box
    inner_c0 = random.randint(outer_c0 + 1, outer_c1 - 3)
    inner_box = (outer_r0 + 3, inner_c0, outer_r0 + 5, inner_c0 + 2)

    a = (outer_r0, inner_c0)
    b = (outer_r0, inner_c0 + 1)
    path = _perimeter_path(outer_box)
    idx_a = path.index(a)
    idx_b = path.index(b)
    long_chain = [path[(idx_b + step) % len(path)] for step in range(len(path))]
    while long_chain[-1] != a:
        long_chain.pop()
    outer_corners = {
        (outer_r0, outer_c0),
        (outer_r0, outer_c1),
        (outer_r1, outer_c1),
        (outer_r1, outer_c0),
    }
    outer_piece_paths = _split_open_chain(long_chain, outer_corners)
    if outer_piece_paths is None:
        return None

    anchor_canvas = np.full((base_h, base_w), BG, dtype=int)
    _draw_frame(anchor_canvas, inner_box, inner_color)
    yellow_cells = [
        (outer_r0 + 1, inner_c0),
        (outer_r0 + 1, inner_c0 + 1),
        (outer_r0 + 2, inner_c0),
        (outer_r0 + 2, inner_c0 + 1),
    ]
    for r, c in yellow_cells:
        anchor_canvas[r, c] = YELLOW
    anchor_canvas[a] = RED
    anchor_canvas[b] = RED

    piece_canvases = [anchor_canvas]
    for coords in outer_piece_paths:
        canvas = np.full((base_h, base_w), BG, dtype=int)
        for i, coord in enumerate(coords):
            canvas[coord] = RED if i in (0, len(coords) - 1) else int(outer_color)
        piece_canvases.append(canvas)

    clean_local = np.full((base_h, base_w), BG, dtype=int)
    _draw_frame(clean_local, outer_box, outer_color)
    _draw_frame(clean_local, inner_box, inner_color)
    for r, c in yellow_cells:
        clean_local[r, c] = YELLOW
    out_local = clean_local.copy()
    rr, cc = np.argwhere(
        (np.indices(clean_local.shape)[0] >= inner_box[0])
        & (np.indices(clean_local.shape)[0] <= inner_box[2])
        & (np.indices(clean_local.shape)[1] >= inner_box[1])
        & (np.indices(clean_local.shape)[1] <= inner_box[3])
        & (
            (np.indices(clean_local.shape)[0] == inner_box[0])
            | (np.indices(clean_local.shape)[0] == inner_box[2])
            | (np.indices(clean_local.shape)[1] == inner_box[1])
            | (np.indices(clean_local.shape)[1] == inner_box[3])
        )
    ).T
    out_local[rr, cc] = outer_color

    transform = random.choice(["id", "rot90", "rot180", "rot270", "flip", "flip_rot90"])

    def apply_transform(arr: np.ndarray) -> np.ndarray:
        if transform == "id":
            return arr
        if transform == "rot90":
            return np.rot90(arr, 1)
        if transform == "rot180":
            return np.rot90(arr, 2)
        if transform == "rot270":
            return np.rot90(arr, 3)
        if transform == "flip":
            return np.fliplr(arr)
        return np.rot90(np.fliplr(arr), 1)

    piece_patches: list[np.ndarray] = []
    piece_final_tops: list[tuple[int, int]] = []
    for canvas in piece_canvases:
        transformed = apply_transform(canvas)
        coords = np.argwhere(transformed != BG)
        if coords.size == 0:
            return None
        box = bbox_coords(coords)
        patch = crop_bbox(transformed, box)
        piece_patches.append(patch)
        piece_final_tops.append((box[0], box[1]))

    clean_local = apply_transform(clean_local)
    out_local = apply_transform(out_local)
    local_bbox = bbox_coords(np.argwhere(clean_local != BG))
    bbox_h = local_bbox[2] - local_bbox[0] + 1
    bbox_w = local_bbox[3] - local_bbox[1] + 1
    height = random.randint(max(20, bbox_h + 8), 28)
    width = random.randint(max(20, bbox_w + 8), 28)

    assembly_top = random.randint(1, height - clean_local.shape[0] - 1)
    assembly_left = random.randint(1, width - clean_local.shape[1] - 1)

    expected = np.full((height, width), BG, dtype=int)
    rr, cc = np.argwhere(out_local != BG).T
    expected[rr + assembly_top, cc + assembly_left] = out_local[rr, cc]

    reserved = np.zeros((height, width), dtype=bool)
    rr0 = max(0, assembly_top + local_bbox[0] - 1)
    cc0 = max(0, assembly_left + local_bbox[1] - 1)
    rr1 = min(height - 1, assembly_top + local_bbox[2] + 1)
    cc1 = min(width - 1, assembly_left + local_bbox[3] + 1)
    reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True

    inp = np.full((height, width), BG, dtype=int)
    anchor_top = assembly_top + piece_final_tops[0][0]
    anchor_left = assembly_left + piece_final_tops[0][1]
    paste(inp, piece_patches[0], anchor_top, anchor_left)

    def can_place(patch: np.ndarray, top: int, left: int) -> bool:
        coords = np.argwhere(patch != BG)
        for pr, pc in coords:
            gr = top + int(pr)
            gc = left + int(pc)
            if reserved[gr, gc] or inp[gr, gc] != BG:
                return False
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr = gr + dr
                    nc = gc + dc
                    if 0 <= nr < height and 0 <= nc < width and inp[nr, nc] != BG:
                        return False
        return True

    order = [1, 2, 3]
    random.shuffle(order)
    for idx in order:
        patch = piece_patches[idx]
        ph, pw = patch.shape
        candidates = []
        final_pos = (assembly_top + piece_final_tops[idx][0], assembly_left + piece_final_tops[idx][1])
        for top in range(1, height - ph):
            for left in range(1, width - pw):
                if (top, left) == final_pos:
                    continue
                if can_place(patch, top, left):
                    candidates.append((top, left))
        if not candidates:
            return None
        top, left = random.choice(candidates)
        paste(inp, patch, top, left)

    try:
        actual = main(inp)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None
    if len(_extract_pieces(inp)) != 4:
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        instance = _instance_attempt()
        if instance is not None:
            return instance[0]
    raise RuntimeError("failed to generate a valid nested-frame connector puzzle")
