from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
RED = 2
YELLOW = 4
PIECE_COLORS = (1, 3, 5, 6, 7, 8, 9)
SIDES = ("top", "right", "bottom", "left")
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
SIDE_TO_VEC = {"top": (-1, 0), "right": (0, 1), "bottom": (1, 0), "left": (0, -1)}
ROTATE_CCW_SIDE = {"top": "left", "left": "bottom", "bottom": "right", "right": "top"}
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
CLOCKWISE = {"top": "right", "right": "bottom", "bottom": "left", "left": "top"}
COUNTERCLOCKWISE = {"top": "left", "left": "bottom", "bottom": "right", "right": "top"}

GRID_RANGE = (22, 29)
PIECE_COUNT_RANGE = (2, 3)
PIECE_SIZE_RANGE = (4, 6)
SCATTER_GAP = 2
GEN_RETRIES = 400


def neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= r < h and 0 <= c < w


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def is_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    return len(component_coords(mask)) == 1


def crop_non_bg(arr: np.ndarray) -> np.ndarray:
    pts = np.argwhere(arr != BG)
    r0, c0, r1, c1 = bbox_coords(pts)
    return arr[r0 : r1 + 1, c0 : c1 + 1].copy()


def grid_key(arr: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, arr.shape)), arr.tobytes()


def unique_rotations(arr: np.ndarray) -> list[tuple[int, np.ndarray]]:
    out: list[tuple[int, np.ndarray]] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rot = np.rot90(arr, k).copy()
        key = grid_key(rot)
        if key not in seen:
            seen.add(key)
            out.append((k, rot))
    return out


def contains_yellow_3x3(arr: np.ndarray) -> bool:
    h, w = arr.shape
    for r in range(h - 2):
        for c in range(w - 2):
            if np.all(arr[r : r + 3, c : c + 3] == YELLOW):
                return True
    return False


def extract_pieces(grid: np.ndarray) -> tuple[dict, list[dict]]:
    pieces: list[dict] = []
    for idx, coords in enumerate(component_coords(grid != BG)):
        r0, c0, r1, c1 = bbox_coords(coords)
        arr = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        piece = {
            "id": idx,
            "bbox": (int(r0), int(c0), int(r1), int(c1)),
            "arr": arr,
            "rots": [
                {
                    "k": k,
                    "arr": rot,
                    "reds": [tuple(map(int, rc)) for rc in np.argwhere(rot == RED)],
                }
                for k, rot in unique_rotations(arr)
            ],
            "reds": [tuple(map(int, rc)) for rc in np.argwhere(arr == RED)],
            "is_anchor": contains_yellow_3x3(arr),
        }
        pieces.append(piece)

    anchors = [p for p in pieces if p["is_anchor"]]
    assert len(anchors) == 1, "expected exactly one anchor piece containing a 3x3 yellow block"
    anchor = anchors[0]
    rest = [p for p in pieces if p is not anchor]
    return anchor, rest


def place_at(canvas: np.ndarray, arr: np.ndarray, top: int, left: int) -> np.ndarray | None:
    h, w = arr.shape
    H, W = canvas.shape
    if top < 0 or left < 0 or top + h > H or left + w > W:
        return None
    sub = canvas[top : top + h, left : left + w]
    mask = arr != BG
    overlap = mask & (sub != BG)
    if np.any(overlap):
        vals_a = arr[overlap]
        vals_b = sub[overlap]
        if not (np.all(vals_a == RED) and np.all(vals_b == RED)):
            return None
    out = canvas.copy()
    out_sub = out[top : top + h, left : left + w]
    out_sub[mask] = arr[mask]
    return out


def apply_move(
    canvas: np.ndarray,
    red_counts: dict[tuple[int, int], int],
    arr: np.ndarray,
    top: int,
    left: int,
    frontier: tuple[int, int],
) -> tuple[np.ndarray, dict[tuple[int, int], int]] | None:
    h, w = arr.shape
    H, W = canvas.shape
    if top < 0 or left < 0 or top + h > H or left + w > W:
        return None

    sub = canvas[top : top + h, left : left + w]
    mask = arr != BG
    overlap_pts = np.argwhere(mask & (sub != BG))
    if overlap_pts.shape[0] != 1:
        return None
    rr, cc = map(int, overlap_pts[0])
    if (top + rr, left + cc) != frontier:
        return None
    if arr[rr, cc] != RED or sub[rr, cc] != RED:
        return None
    if red_counts.get(frontier, 0) != 1:
        return None

    next_counts = dict(red_counts)
    for pr, pc in np.argwhere(arr == RED):
        coord = (int(top + pr), int(left + pc))
        if next_counts.get(coord, 0) >= 2:
            return None
        next_counts[coord] = next_counts.get(coord, 0) + 1

    nxt = canvas.copy()
    nxt_sub = nxt[top : top + h, left : left + w]
    nxt_sub[mask] = arr[mask]
    return nxt, next_counts


def enumerate_moves_for_frontier(
    canvas: np.ndarray,
    red_counts: dict[tuple[int, int], int],
    remaining: list[dict],
    frontier: tuple[int, int],
) -> list[dict]:
    out: list[dict] = []
    seen: set[tuple[int, tuple[tuple[int, int], bytes], int, int]] = set()
    tr, tc = frontier
    for idx, piece in enumerate(remaining):
        for rot in piece["rots"]:
            arr = rot["arr"]
            arr_key = grid_key(arr)
            for rr, cc in rot["reds"]:
                top = tr - rr
                left = tc - cc
                dedup_key = (idx, arr_key, int(top), int(left))
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                applied = apply_move(canvas, red_counts, arr, int(top), int(left), frontier)
                if applied is None:
                    continue
                nxt, next_counts = applied
                out.append(
                    {
                        "piece_idx": idx,
                        "piece_id": piece["id"],
                        "rot_k": rot["k"],
                        "arr": arr,
                        "top": int(top),
                        "left": int(left),
                        "canvas": nxt,
                        "red_counts": next_counts,
                    }
                )
    out.sort(key=lambda m: (m["piece_id"], m["rot_k"], m["top"], m["left"]))
    return out


def enumerate_all_moves(
    canvas: np.ndarray,
    red_counts: dict[tuple[int, int], int],
    remaining: list[dict],
) -> list[dict]:
    all_moves: list[dict] = []
    for frontier, count in sorted(red_counts.items()):
        if count == 1:
            all_moves.extend(enumerate_moves_for_frontier(canvas, red_counts, remaining, frontier))
    return all_moves


def assemble_unique(grid: np.ndarray) -> np.ndarray:
    anchor, remaining = extract_pieces(grid)
    H, W = grid.shape
    canvas = np.zeros((H, W), dtype=int)

    ar0, ac0, _, _ = anchor["bbox"]
    placed_anchor = place_at(canvas, anchor["arr"], ar0, ac0)
    assert placed_anchor is not None, "anchor must fit unchanged"

    red_counts: dict[tuple[int, int], int] = {}
    for rr, cc in anchor["reds"]:
        red_counts[(ar0 + rr, ac0 + cc)] = red_counts.get((ar0 + rr, ac0 + cc), 0) + 1

    solutions: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}

    def dfs(cur: np.ndarray, rem: list[dict], counts: dict[tuple[int, int], int]) -> None:
        if not rem:
            if counts and all(v == 2 for v in counts.values()):
                solutions[grid_key(cur)] = cur.copy()
            return

        frontiers = [coord for coord, v in counts.items() if v == 1]
        if not frontiers:
            return

        frontier_moves: list[list[dict]] = []
        for frontier in frontiers:
            moves = enumerate_moves_for_frontier(cur, counts, rem, frontier)
            if not moves:
                return
            frontier_moves.append(moves)
        moves = min(frontier_moves, key=len)

        for move in moves:
            idx = int(move["piece_idx"])
            next_rem = rem[:idx] + rem[idx + 1 :]
            dfs(move["canvas"], next_rem, move["red_counts"])

    dfs(placed_anchor, remaining, red_counts)
    assert solutions, "assembly must have at least one valid final result"
    return solutions[sorted(solutions.keys())[0]]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate scattered connector pieces into a single self-consistent assembly around the yellow anchor.

    Input structure:
    - The background is black.
    - One fixed anchor sprite contains a solid 3x3 yellow block and one or more red connector pixels.
    - Several other colored sprites are scattered elsewhere; each sprite is a single component with red edge connectors.

    Rule:
    - Keep the anchor in its original place.
    - Repeatedly choose a remaining sprite and search over its 0/90/180/270-degree rotations.
    - A placement is valid only when exactly one of the sprite's red pixels overlaps one currently unpaired red pixel in the growing assembly, and every other sprite pixel avoids overlap.
    - Paint the placed sprite into the assembly, keep matched red pixels red, and continue until all sprites are attached.
    - Return the final assembled grid on the original canvas.

    Key insight:
    - The picture never tells you the rotation directly; the only way to recover it is to find the unique red-connector pairing geometry that stays collision-free.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    return assemble_unique(g)


def choose_connector(side: str, h: int, w: int) -> tuple[int, int]:
    if side == "top":
        cols = list(range(1, w - 1)) or list(range(w))
        return 0, random.choice(cols)
    if side == "bottom":
        cols = list(range(1, w - 1)) or list(range(w))
        return h - 1, random.choice(cols)
    if side == "left":
        rows = list(range(1, h - 1)) or list(range(h))
        return random.choice(rows), 0
    rows = list(range(1, h - 1)) or list(range(h))
    return random.choice(rows), w - 1


def inward_neighbor(side: str, cell: tuple[int, int]) -> tuple[int, int]:
    r, c = cell
    if side == "top":
        return r + 1, c
    if side == "bottom":
        return r - 1, c
    if side == "left":
        return r, c + 1
    return r, c - 1


def random_piece_array(degree: int, color: int) -> np.ndarray | None:
    for _ in range(120):
        h = random.randint(*PIECE_SIZE_RANGE)
        w = random.randint(*PIECE_SIZE_RANGE)
        sides = random.sample(SIDES, degree)
        connectors = [choose_connector(side, h, w) for side in sides]

        mask = np.ones((h, w), dtype=bool)
        protected = set(connectors)
        for side, cell in zip(sides, connectors):
            protected.add(inward_neighbor(side, cell))

        remove_goal = random.randint(max(2, (h * w) // 6), max(3, (h * w) // 3))
        removed = 0
        trials = 0
        while removed < remove_goal and trials < remove_goal * 12:
            trials += 1
            r = random.randrange(h)
            c = random.randrange(w)
            if (r, c) in protected or not mask[r, c]:
                continue
            mask[r, c] = False
            if not is_connected(mask):
                mask[r, c] = True
                continue
            if any(not any(mask[nr, nc] for nr, nc in neighbors4(*cell) if 0 <= nr < h and 0 <= nc < w) for cell in connectors):
                mask[r, c] = True
                continue
            removed += 1

        if removed < 2:
            continue
        if int(mask.sum()) < degree + 5:
            continue

        arr = np.zeros((h, w), dtype=int)
        arr[mask] = int(color)
        for cell in connectors:
            arr[cell] = RED
        arr = crop_non_bg(arr)
        if len(unique_rotations(arr)) < 4:
            continue
        reds = np.argwhere(arr == RED)
        if reds.shape[0] != degree:
            continue
        return arr
    return None


def anchor_array() -> np.ndarray:
    arr = np.zeros((5, 5), dtype=int)
    arr[1:4, 1:4] = YELLOW
    side = random.choice(SIDES)
    if side == "top":
        arr[0, 2] = RED
    elif side == "bottom":
        arr[4, 2] = RED
    elif side == "left":
        arr[2, 0] = RED
    else:
        arr[2, 4] = RED
    return arr


def rotate_side(side: str, k: int) -> str:
    out = side
    for _ in range(k % 4):
        out = ROTATE_CCW_SIDE[out]
    return out


def red_coord_on_side(arr: np.ndarray, side: str) -> tuple[int, int]:
    matches = []
    for r, c in np.argwhere(arr == RED):
        r = int(r)
        c = int(c)
        if side == "top" and r == 0:
            matches.append((r, c))
        elif side == "bottom" and r == arr.shape[0] - 1:
            matches.append((r, c))
        elif side == "left" and c == 0:
            matches.append((r, c))
        elif side == "right" and c == arr.shape[1] - 1:
            matches.append((r, c))
    assert len(matches) == 1
    return matches[0]


def side_of_red(arr: np.ndarray, coord: tuple[int, int]) -> str:
    r, c = coord
    if r == 0:
        return "top"
    if r == arr.shape[0] - 1:
        return "bottom"
    if c == 0:
        return "left"
    if c == arr.shape[1] - 1:
        return "right"
    raise AssertionError("red connector must lie on the boundary")


def draw_path(arr: np.ndarray, points: list[tuple[int, int]], color: int) -> None:
    for (r0, c0), (r1, c1) in zip(points, points[1:]):
        if r0 == r1:
            step = 1 if c1 >= c0 else -1
            for c in range(c0, c1 + step, step):
                arr[r0, c] = color
        else:
            step = 1 if r1 >= r0 else -1
            for r in range(r0, r1 + step, step):
                arr[r, c0] = color


def flank_delta(side: str, kind: int) -> tuple[int, int]:
    if side == "top":
        return (0, 1) if kind == 0 else (0, -1)
    if side == "right":
        return (1, 0) if kind == 0 else (-1, 0)
    if side == "bottom":
        return (0, -1) if kind == 0 else (0, 1)
    return (-1, 0) if kind == 0 else (1, 0)


def stamp_flank(arr: np.ndarray, side: str, coord: tuple[int, int], kind: int, color: int) -> bool:
    dr, dc = flank_delta(side, kind)
    rr = coord[0] + dr
    cc = coord[1] + dc
    if not (0 <= rr < arr.shape[0] and 0 <= cc < arr.shape[1]):
        return False
    if arr[rr, cc] == RED:
        return False
    arr[rr, cc] = color
    return True


def canonical_left_to_right(color: int, left_kind: int, right_kind: int) -> np.ndarray | None:
    h = random.randint(5, 6)
    w = random.randint(6, 7)
    arr = np.zeros((h, w), dtype=int)
    r0 = random.choice([1, h - 2])
    r1 = h - 2 if r0 == 1 else 1
    bend = max(2, w - 3)
    draw_path(arr, [(r0, 0), (r0, bend), (r1, bend), (r1, w - 1)], color)
    nub_r = min(h - 1, max(0, r0 + (1 if r0 < h // 2 else -1)))
    nub_c = min(w - 1, 1 + random.randint(0, max(1, bend - 1)))
    arr[nub_r, nub_c] = color
    arr[r0, 0] = RED
    arr[r1, w - 1] = RED
    if not stamp_flank(arr, "left", (r0, 0), left_kind, color):
        return None
    if not stamp_flank(arr, "right", (r1, w - 1), right_kind, color):
        return None
    return crop_non_bg(arr)


def canonical_left_to_top(color: int, left_kind: int, top_kind: int) -> np.ndarray | None:
    h = random.randint(6, 7)
    w = random.randint(6, 7)
    arr = np.zeros((h, w), dtype=int)
    r0 = random.choice([h - 2, h - 3])
    c1 = random.randint(2, w - 3)
    draw_path(arr, [(r0, 0), (r0, c1), (0, c1)], color)
    nub_c = max(1, c1 - random.randint(1, min(2, c1)))
    nub_r = min(h - 1, r0 + 1)
    arr[nub_r, nub_c] = color
    arr[r0, 0] = RED
    arr[0, c1] = RED
    if not stamp_flank(arr, "left", (r0, 0), left_kind, color):
        return None
    if not stamp_flank(arr, "top", (0, c1), top_kind, color):
        return None
    return crop_non_bg(arr)


def canonical_left_to_bottom(color: int, left_kind: int, bottom_kind: int) -> np.ndarray | None:
    base = canonical_left_to_top(color, left_kind, 1 - bottom_kind)
    if base is None:
        return None
    return np.flipud(base).copy()


def canonical_leaf_left(color: int, left_kind: int) -> np.ndarray | None:
    h = random.randint(5, 7)
    w = random.randint(6, 7)
    arr = np.zeros((h, w), dtype=int)
    r0 = random.choice([1, h - 2])
    c1 = random.choice([w - 3, w - 2])
    turn_r = max(1, min(h - 2, r0 + (2 if r0 == 1 else -2)))
    draw_path(arr, [(r0, 0), (r0, c1), (turn_r, c1)], color)
    tail_c = max(1, c1 - random.randint(1, min(2, c1)))
    arr[turn_r, tail_c] = color
    arr[r0, 0] = RED
    if not stamp_flank(arr, "left", (r0, 0), left_kind, color):
        return None
    return crop_non_bg(arr)


def oriented_piece(
    attach_side: str,
    exit_side: str | None,
    color: int,
    attach_kind: int,
    exit_kind: int | None,
) -> np.ndarray | None:
    for _ in range(40):
        k = next(k for k in range(4) if rotate_side("left", k) == attach_side)
        if exit_side is None:
            base = canonical_leaf_left(color, attach_kind)
        else:
            base_exit = rotate_side(exit_side, (-k) % 4)
            if base_exit == "right":
                base = canonical_left_to_right(color, attach_kind, int(exit_kind))
            elif base_exit == "top":
                base = canonical_left_to_top(color, attach_kind, int(exit_kind))
            elif base_exit == "bottom":
                base = canonical_left_to_bottom(color, attach_kind, int(exit_kind))
            else:
                continue
        if base is None:
            continue
        arr = np.rot90(base, k).copy()
        if len(unique_rotations(arr)) < 4:
            continue
        if exit_side is not None:
            reds = np.argwhere(arr == RED)
            if reds.shape[0] != 2:
                continue
            try:
                red_coord_on_side(arr, attach_side)
                red_coord_on_side(arr, exit_side)
            except AssertionError:
                continue
        else:
            if np.count_nonzero(arr == RED) != 1:
                continue
            try:
                red_coord_on_side(arr, attach_side)
            except AssertionError:
                continue
        return arr
    return None


def connector_count(arr: np.ndarray) -> int:
    return int(np.count_nonzero(arr == RED))


def frontier_coords(red_counts: dict[tuple[int, int], int]) -> list[tuple[int, int]]:
    return [coord for coord, count in red_counts.items() if count == 1]


def occupied_bbox(canvas: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(canvas != BG)
    return bbox_coords(pts)


def margin_to_edge(shape: tuple[int, int], coord: tuple[int, int]) -> int:
    H, W = map(int, shape)
    r, c = coord
    return min(r, c, H - 1 - r, W - 1 - c)


def build_output() -> tuple[np.ndarray, list[dict]]:
    for _ in range(GEN_RETRIES):
        hidden_H = hidden_W = 60
        max_span = random.randint(13, 17)
        n_pieces = random.randint(*PIECE_COUNT_RANGE)
        colors = random.sample(list(PIECE_COLORS), n_pieces)

        out = np.zeros((hidden_H, hidden_W), dtype=int)
        anchor_side = random.choice(SIDES)
        frontier_kind = 0
        anchor = np.zeros((5, 5), dtype=int)
        anchor[1:4, 1:4] = YELLOW
        if anchor_side == "top":
            anchor[0, 2] = RED
        elif anchor_side == "bottom":
            anchor[4, 2] = RED
        elif anchor_side == "left":
            anchor[2, 0] = RED
        else:
            anchor[2, 4] = RED
        if not stamp_flank(anchor, anchor_side, red_coord_on_side(anchor, anchor_side), frontier_kind, YELLOW):
            continue
        ah, aw = anchor.shape
        ar0 = hidden_H // 2 - ah // 2
        ac0 = hidden_W // 2 - aw // 2
        placed = place_at(out, anchor, ar0, ac0)
        if placed is None:
            continue
        out = placed

        red_counts: dict[tuple[int, int], int] = {}
        for rr, cc in np.argwhere(anchor == RED):
            red_counts[(ar0 + int(rr), ac0 + int(cc))] = 1
        frontier_side = anchor_side

        pieces_meta = [
            {
                "kind": "anchor",
                "arr": anchor,
                "top": ar0,
                "left": ac0,
            }
        ]

        def extend(
            cur: np.ndarray,
            counts: dict[tuple[int, int], int],
            cur_frontier_side: str,
            idx: int,
            acc: list[dict],
        ) -> list[dict] | None:
            if idx == n_pieces:
                return acc if not frontier_coords(counts) else None

            degree = 2 if idx < n_pieces - 1 else 1
            frontier = frontier_coords(counts)[0]
            attach_side = OPPOSITE[cur_frontier_side]
            rem_after = n_pieces - idx - 1
            requested_exits = [None] if degree == 1 else [s for s in SIDES if s != attach_side]
            random.shuffle(requested_exits)
            candidate_states: list[tuple[np.ndarray, np.ndarray, dict[tuple[int, int], int], int, int, str | None]] = []

            for request_side in requested_exits:
                for _ in range(10):
                    piece = oriented_piece(attach_side, request_side, colors[idx], frontier_kind, frontier_kind if degree == 2 else None)
                    if piece is None:
                        continue
                    piece_variants = [
                        {
                            "k": k,
                            "arr": rot,
                            "reds": [tuple(map(int, rc)) for rc in np.argwhere(rot == RED)],
                        }
                        for k, rot in unique_rotations(piece)
                    ]
                    moves = enumerate_moves_for_frontier(
                        cur,
                        counts,
                        [{"id": idx, "rots": piece_variants}],
                        frontier,
                    )
                    if len(moves) != 1:
                        continue
                    move = moves[0]
                    nxt = move["canvas"]
                    next_counts = move["red_counts"]
                    next_frontiers = frontier_coords(next_counts)
                    if degree == 2 and len(next_frontiers) != 1:
                        continue
                    if degree == 1 and next_frontiers:
                        continue
                    bbox = occupied_bbox(nxt)
                    if bbox[2] - bbox[0] + 1 > max_span or bbox[3] - bbox[1] + 1 > max_span:
                        continue
                    if rem_after > 0 and margin_to_edge((hidden_H, hidden_W), next_frontiers[0]) < 6:
                        continue
                    actual_exit = None
                    if degree == 2:
                        reds_local = [tuple(map(int, rc)) for rc in np.argwhere(move["arr"] == RED)]
                        attach_local = (frontier[0] - move["top"], frontier[1] - move["left"])
                        other_red = next(rc for rc in reds_local if rc != attach_local)
                        actual_exit = side_of_red(move["arr"], other_red)
                    candidate_states.append(
                        (move["arr"].copy(), nxt, next_counts, move["top"], move["left"], actual_exit)
                    )

            random.shuffle(candidate_states)
            seen_cands: set[tuple[tuple[int, int], bytes, int, int, str | None]] = set()
            for piece_arr, nxt, next_counts, top, left, actual_exit in candidate_states:
                dedup = (*grid_key(piece_arr), int(top), int(left), actual_exit)
                if dedup in seen_cands:
                    continue
                seen_cands.add(dedup)
                next_acc = acc + [
                    {
                        "kind": "piece",
                        "arr": piece_arr.copy(),
                        "top": int(top),
                        "left": int(left),
                    }
                ]
                result = extend(
                    nxt,
                    next_counts,
                    actual_exit if actual_exit is not None else cur_frontier_side,
                    idx + 1,
                    next_acc,
                )
                if result is not None:
                    return result
            return None

        built = extend(out, red_counts, frontier_side, 0, pieces_meta)
        if built is None:
            continue
        pieces_meta = built
        out = np.zeros((hidden_H, hidden_W), dtype=int)
        for meta in pieces_meta:
            pasted = place_at(out, meta["arr"], meta["top"], meta["left"])
            assert pasted is not None
            out = pasted

        bbox = occupied_bbox(out)
        r0, c0, r1, c1 = bbox
        crop = out[r0 : r1 + 1, c0 : c1 + 1].copy()
        ch, cw = crop.shape
        min_h = max(GRID_RANGE[0], ch + 8)
        min_w = max(GRID_RANGE[0], cw + 8)
        if min_h > GRID_RANGE[1] or min_w > GRID_RANGE[1]:
            continue

        H = random.randint(min_h, GRID_RANGE[1])
        W = random.randint(min_w, GRID_RANGE[1])
        top0 = random.randint(2, H - ch - 2)
        left0 = random.randint(2, W - cw - 2)
        final_out = np.zeros((H, W), dtype=int)
        final_out[top0 : top0 + ch, left0 : left0 + cw] = crop

        shifted_meta = []
        for meta in pieces_meta:
            shifted_meta.append(
                {
                    "kind": meta["kind"],
                    "arr": meta["arr"].copy(),
                    "top": int(meta["top"] - r0 + top0),
                    "left": int(meta["left"] - c0 + left0),
                }
            )
        return final_out, shifted_meta

    raise RuntimeError("failed to build a valid output assembly")


def bboxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    return a[2] + gap < b[0] or b[2] + gap < a[0] or a[3] + gap < b[1] or b[3] + gap < a[1]


def scatter_input(output: np.ndarray, pieces_meta: list[dict]) -> np.ndarray | None:
    H, W = output.shape
    inp = np.zeros_like(output)
    placed_boxes: list[tuple[int, int, int, int]] = []

    anchor = pieces_meta[0]
    pasted = place_at(inp, anchor["arr"], anchor["top"], anchor["left"])
    if pasted is None:
        return None
    inp = pasted
    ah, aw = anchor["arr"].shape
    placed_boxes.append((anchor["top"], anchor["left"], anchor["top"] + ah - 1, anchor["left"] + aw - 1))

    pieces = pieces_meta[1:]
    random.shuffle(pieces)
    rotated_variants = []
    for meta in pieces:
        arr = np.rot90(meta["arr"], random.randrange(4)).copy()
        rotated_variants.append({"arr": arr})

    rotated_variants.sort(key=lambda item: int(np.count_nonzero(item["arr"] != BG)), reverse=True)

    for meta in rotated_variants:
        arr = meta["arr"]
        h, w = arr.shape
        candidates = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
        random.shuffle(candidates)
        placed = False
        for top, left in candidates:
            box = (top, left, top + h - 1, left + w - 1)
            if any(not bboxes_separated(box, other, SCATTER_GAP) for other in placed_boxes):
                continue
            nxt = place_at(inp, arr, top, left)
            if nxt is None:
                continue
            inp = nxt
            placed_boxes.append(box)
            placed = True
            break
        if not placed:
            return None
    return inp


def greedy_unique_rebuild(inp: np.ndarray, expected: np.ndarray) -> bool:
    anchor, remaining = extract_pieces(inp)
    H, W = inp.shape
    canvas = np.zeros((H, W), dtype=int)
    ar0, ac0, _, _ = anchor["bbox"]
    canvas = place_at(canvas, anchor["arr"], ar0, ac0)
    assert canvas is not None
    red_counts: dict[tuple[int, int], int] = {}
    for rr, cc in anchor["reds"]:
        red_counts[(ar0 + rr, ac0 + cc)] = 1

    rem = remaining[:]
    while rem:
        moves = enumerate_all_moves(canvas, red_counts, rem)
        if len(moves) != 1:
            return False
        move = moves[0]
        idx = int(move["piece_idx"])
        canvas = move["canvas"]
        red_counts = move["red_counts"]
        rem = rem[:idx] + rem[idx + 1 :]

    return not frontier_coords(red_counts) and np.array_equal(canvas, expected)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        output, pieces_meta = build_output()
        inp = scatter_input(output, pieces_meta)
        if inp is None:
            continue
        if np.array_equal(inp, output):
            continue
        try:
            _ = main(inp)
        except AssertionError:
            continue
        return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
