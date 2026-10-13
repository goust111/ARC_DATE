from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7

SIDE_COLORS = {"top": BLUE, "right": GREEN, "bottom": ORANGE, "left": MAGENTA}
SIDES = ("top", "right", "bottom", "left")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
CONNECTOR_POS = {"top": (0, 2), "right": (2, 4), "bottom": (4, 2), "left": (2, 0)}
ADJACENT_BODY_POS = {"top": (1, 2), "right": (2, 3), "bottom": (3, 2), "left": (2, 1)}


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return _bbox_coords(coords)


def _crop_nonzero(arr: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    box = _bbox_mask(arr != BG)
    if box is None:
        raise ValueError("cannot crop an empty patch")
    r0, c0, r1, c1 = box
    return arr[r0 : r1 + 1, c0 : c1 + 1].copy(), box


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _find_pieces(grid: np.ndarray) -> list[dict]:
    pieces = []
    for coords in _components(grid != BG):
        r0, c0, r1, c1 = _bbox_coords(coords)
        arr = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        reds = [tuple(map(int, rc)) for rc in np.argwhere(arr == RED)]
        pieces.append(
            {
                "bbox": (r0, c0, r1, c1),
                "arr": arr,
                "shape": tuple(map(int, arr.shape)),
                "reds": reds,
                "area": int(np.count_nonzero(arr)),
            }
        )
    return pieces


def _has_anchor_block(arr: np.ndarray) -> bool:
    h, w = arr.shape
    for r in range(h - 2):
        for c in range(w - 2):
            if np.all(arr[r : r + 3, c : c + 3] == YELLOW):
                return True
    return False


def _try_place(
    canvas: np.ndarray,
    arr: np.ndarray,
    top: int,
    left: int,
    *,
    require_overlap: bool,
) -> tuple[np.ndarray, tuple[tuple[int, int], ...]] | None:
    h, w = canvas.shape
    ph, pw = arr.shape
    if top < 0 or left < 0 or top + ph > h or left + pw > w:
        return None

    overlaps: list[tuple[int, int]] = []
    sub = canvas[top : top + ph, left : left + pw]
    mask = arr != BG
    coords = np.argwhere(mask)
    for rr, cc in coords:
        av = int(arr[rr, cc])
        cv = int(sub[rr, cc])
        if cv == BG:
            continue
        if av == RED and cv == RED:
            overlaps.append((top + int(rr), left + int(cc)))
            continue
        return None

    if require_overlap and not overlaps:
        return None

    out = canvas.copy()
    patch = out[top : top + ph, left : left + pw]
    for rr, cc in coords:
        if patch[rr, cc] == BG:
            patch[rr, cc] = int(arr[rr, cc])
    out[top : top + ph, left : left + pw] = patch
    return out, tuple(overlaps)


def _assemble_canvas(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    pieces = _find_pieces(g)
    assert pieces, "expected at least one sprite"

    anchors = [piece for piece in pieces if _has_anchor_block(piece["arr"])]
    assert len(anchors) == 1, "expected exactly one anchor sprite with a 3x3 yellow block"
    anchor = anchors[0]
    rest = [piece for piece in pieces if piece is not anchor]

    total_ports = sum(len(piece["reds"]) for piece in pieces)
    assert total_ports % 2 == 0, "red connectors must pair up"
    target_red_sites = total_ports // 2

    canvas = np.full_like(g, BG)
    ar0, ac0, _, _ = anchor["bbox"]
    start = _try_place(canvas, anchor["arr"], ar0, ac0, require_overlap=False)
    assert start is not None
    placed, _ = start

    red_counts: dict[tuple[int, int], int] = {}
    for rr, cc in anchor["reds"]:
        red_counts[(ar0 + rr, ac0 + cc)] = red_counts.get((ar0 + rr, ac0 + cc), 0) + 1

    solutions: dict[bytes, np.ndarray] = {}

    def dfs(cur: np.ndarray, remaining: list[dict], counts: dict[tuple[int, int], int]) -> None:
        if len(solutions) >= 2:
            return
        if not remaining:
            if len(counts) == target_red_sites and counts and all(v == 2 for v in counts.values()):
                solutions.setdefault(cur.tobytes(), cur.copy())
            return

        unpaired = [coord for coord, cnt in counts.items() if cnt == 1]
        if not unpaired:
            return
        if len(counts) > target_red_sites:
            return

        best_moves = None
        for target in unpaired:
            tr, tc = target
            candidate_moves = []
            for idx, piece in enumerate(remaining):
                ph, pw = piece["shape"]
                for pr, pc in piece["reds"]:
                    top = tr - pr
                    left = tc - pc
                    if top < 0 or left < 0 or top + ph > g.shape[0] or left + pw > g.shape[1]:
                        continue
                    ok = True
                    paired = 0
                    for rr, cc in piece["reds"]:
                        abs_rc = (top + rr, left + cc)
                        prev = counts.get(abs_rc, 0)
                        if prev >= 2:
                            ok = False
                            break
                        if prev == 1:
                            paired += 1
                    if not ok:
                        continue
                    candidate_moves.append((idx, top, left, paired))
            if not candidate_moves:
                return
            if best_moves is None or len(candidate_moves) < len(best_moves):
                best_moves = candidate_moves
                if len(best_moves) == 1:
                    break

        assert best_moves is not None
        dedup: dict[tuple[int, int, int], int] = {}
        for idx, top, left, paired in best_moves:
            key = (idx, top, left)
            dedup[key] = max(dedup.get(key, 0), paired)
        ordered = [(idx, top, left, paired) for (idx, top, left), paired in dedup.items()]
        ordered.sort(key=lambda item: (-item[3], -remaining[item[0]]["area"], item[1], item[2], item[0]))

        for idx, top, left, _paired in ordered:
            piece = remaining[idx]
            placed_next = _try_place(cur, piece["arr"], top, left, require_overlap=True)
            if placed_next is None:
                continue
            nxt, overlaps = placed_next
            if any(counts.get(coord, 0) != 1 for coord in overlaps):
                continue

            next_counts = dict(counts)
            for rr, cc in piece["reds"]:
                coord = (top + rr, left + cc)
                next_counts[coord] = next_counts.get(coord, 0) + 1
            if any(v > 2 for v in next_counts.values()):
                continue

            next_remaining = remaining[:idx] + remaining[idx + 1 :]
            dfs(nxt, next_remaining, next_counts)
            if len(solutions) >= 2:
                return

    dfs(placed, rest, red_counts)
    assert len(solutions) == 1, "expected a unique full assembly"
    return next(iter(solutions.values()))


def _recolor_crop(assembled: np.ndarray) -> np.ndarray:
    box = _bbox_mask(assembled != BG)
    assert box is not None
    r0, c0, r1, c1 = box
    crop = assembled[r0 : r1 + 1, c0 : c1 + 1].copy()
    h, w = crop.shape
    for r in range(h):
        for c in range(w):
            color = int(crop[r, c])
            if color in (BG, RED):
                continue
            distances = {
                "top": r,
                "right": w - 1 - c,
                "bottom": h - 1 - r,
                "left": c,
            }
            nearest = min(distances.values())
            best = [side for side, dist in distances.items() if dist == nearest]
            if len(best) == 1:
                crop[r, c] = SIDE_COLORS[best[0]]
    return crop


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble the scattered connector sprites, then recolor each non-connector pixel by its unique nearest bbox side.

    Input structure:
    - The background is black and several disconnected sprites are scattered apart.
    - One anchor sprite contains a solid 3x3 yellow block, while every sprite also carries one or more red connector pixels on its boundary.
    - The movable sprites each use a single body color plus red connectors, and they are separated from one another in the input.

    Rule:
    - Keep the anchor fixed and translate the other sprites one by one so red connector pixels overlap pairwise with currently unpaired red pixels on the growing assembly.
    - After all sprites have been joined, take the tight bounding box of the assembled structure.
    - Keep red connector pixels red, but recolor every other occupied pixel by the color of its uniquely nearest bounding-box side: top blue, right green, bottom orange, left magenta; tied nearest sides keep the pixel's current color.
    - Return only the cropped bounding box of the final assembled structure.

    Key insight:
    - The puzzle is a causal two-stage transform: red connectors first determine the assembled global shape, and only that final shape determines which side-color claims each non-red pixel.
    """
    assembled = _assemble_canvas(np.asarray(grid, dtype=int))
    return _recolor_crop(assembled)


def _random_body(side_set: tuple[str, ...], color: int) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    staging = np.zeros((5, 5), dtype=int)
    body = {ADJACENT_BODY_POS[side] for side in side_set}
    body.add((2, 2))
    target = random.randint(max(3, len(body)), 6)
    core = {(r, c) for r in range(1, 4) for c in range(1, 4)}
    while len(body) < target:
        frontier = sorted(
            {
                nbr
                for r, c in body
                for nbr in _neighbors4(r, c)
                if nbr in core and nbr not in body
            }
        )
        if not frontier:
            break
        body.add(random.choice(frontier))
    for r, c in body:
        staging[r, c] = color
    for side in side_set:
        rr, cc = CONNECTOR_POS[side]
        staging[rr, cc] = RED
    patch, box = _crop_nonzero(staging)
    r0, c0, _, _ = box
    red_map = {side: (CONNECTOR_POS[side][0] - r0, CONNECTOR_POS[side][1] - c0) for side in side_set}
    return patch, red_map


def _make_anchor(sides: tuple[str, ...]) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    staging = np.zeros((5, 5), dtype=int)
    staging[1:4, 1:4] = YELLOW
    for side in sides:
        rr, cc = CONNECTOR_POS[side]
        staging[rr, cc] = RED
    red_map = {side: CONNECTOR_POS[side] for side in sides}
    return staging, red_map


def _compose_lengths(total: int, parts: int) -> list[int]:
    cuts = sorted(random.sample(range(1, total), parts - 1))
    vals = []
    prev = 0
    for cut in cuts + [total]:
        vals.append(cut - prev)
        prev = cut
    return vals


def _build_chain_parents(n_pieces: int) -> tuple[list[int], list[list[int]]]:
    max_chains = min(3, n_pieces)
    chain_options = [k for k in range(1, max_chains + 1)]
    weights = {1: 0.3, 2: 0.45, 3: 0.25}
    chain_count = random.choices(chain_options, weights=[weights[k] for k in chain_options])[0]
    lengths = _compose_lengths(n_pieces, chain_count)
    parent = [-1] * (n_pieces + 1)
    children = [[] for _ in range(n_pieces + 1)]
    node = 1
    for length in lengths:
        prev = 0
        for _ in range(length):
            parent[node] = prev
            children[prev].append(node)
            prev = node
            node += 1
    return parent, children


def _side_preference(attach_side: str) -> list[str]:
    forward = OPPOSITE[attach_side]
    orth = [side for side in SIDES if side not in (attach_side, forward)]
    random.shuffle(orth)
    return [forward] + orth


def _generation_placeable(
    canvas: np.ndarray,
    arr: np.ndarray,
    top: int,
    left: int,
    overlap_target: tuple[int, int],
) -> bool:
    h, w = canvas.shape
    ph, pw = arr.shape
    if top < 0 or left < 0 or top + ph > h or left + pw > w:
        return False
    overlap_count = 0
    occupied = np.argwhere(arr != BG)
    for rr, cc in occupied:
        rr = int(rr)
        cc = int(cc)
        abs_rc = (top + rr, left + cc)
        av = int(arr[rr, cc])
        cv = int(canvas[abs_rc])
        if cv != BG:
            if abs_rc == overlap_target and av == RED and cv == RED:
                overlap_count += 1
            else:
                return False
    if overlap_count != 1:
        return False
    for rr, cc in occupied:
        rr = int(rr)
        cc = int(cc)
        abs_r, abs_c = top + rr, left + cc
        if (abs_r, abs_c) == overlap_target:
            continue
        for nr, nc in _neighbors4(abs_r, abs_c):
            if 0 <= nr < h and 0 <= nc < w and canvas[nr, nc] != BG and (nr, nc) != overlap_target:
                return False
    return True


def _paste_nonzero(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    coords = np.argwhere(patch != BG)
    for rr, cc in coords:
        dst[top + int(rr), left + int(cc)] = int(patch[rr, cc])


def _assemble_generated_structure() -> list[dict] | None:
    n_pieces = random.randint(3, 5)
    colors = random.sample([BLUE, GREEN, GREY, MAGENTA, ORANGE], n_pieces)
    parent, children = _build_chain_parents(n_pieces)
    anchor_sides = tuple(random.sample(SIDES, len(children[0])))
    anchor_arr, anchor_reds = _make_anchor(anchor_sides)

    canvas = np.zeros((30, 30), dtype=int)
    anchor_top, anchor_left = 12, 12
    _paste_nonzero(canvas, anchor_arr, anchor_top, anchor_left)

    pieces: list[dict | None] = [None] * (n_pieces + 1)
    pieces[0] = {
        "id": 0,
        "arr": anchor_arr,
        "reds": anchor_reds,
        "child_sides": anchor_sides,
        "top": anchor_top,
        "left": anchor_left,
        "body_color": YELLOW,
    }
    pending = deque([0])

    while pending:
        node = pending.popleft()
        node_info = pieces[node]
        assert node_info is not None
        child_sides = list(node_info["child_sides"])
        random.shuffle(child_sides)
        assert len(child_sides) == len(children[node])
        for child, parent_side in zip(children[node], child_sides, strict=False):
            attach_side = OPPOSITE[parent_side]
            child_children = children[child]
            base_choices = _side_preference(attach_side)
            side_options = [()] if not child_children else [(side,) for side in base_choices]
            placed = False
            for extra_sides in side_options:
                for _ in range(8):
                    side_set = tuple([attach_side, *extra_sides])
                    arr, red_map = _random_body(side_set, colors[child - 1])
                    parent_abs = (
                        node_info["top"] + node_info["reds"][parent_side][0],
                        node_info["left"] + node_info["reds"][parent_side][1],
                    )
                    child_red = red_map[attach_side]
                    top = parent_abs[0] - child_red[0]
                    left = parent_abs[1] - child_red[1]
                    if not _generation_placeable(canvas, arr, top, left, parent_abs):
                        continue
                    _paste_nonzero(canvas, arr, top, left)
                    pieces[child] = {
                        "id": child,
                        "arr": arr,
                        "reds": red_map,
                        "child_sides": extra_sides,
                        "top": top,
                        "left": left,
                        "body_color": colors[child - 1],
                    }
                    pending.append(child)
                    placed = True
                    break
                if placed:
                    break
            if not placed:
                return None

    out = [piece for piece in pieces if piece is not None]
    assert len(out) == n_pieces + 1
    return out


def _assembled_canvas_from_pieces(pieces: list[dict]) -> np.ndarray:
    canvas = np.zeros((30, 30), dtype=int)
    for piece in pieces:
        _paste_nonzero(canvas, piece["arr"], piece["top"], piece["left"])
    return canvas


def _recolor_stats(assembled: np.ndarray) -> tuple[set[str], int]:
    box = _bbox_mask(assembled != BG)
    assert box is not None
    r0, c0, r1, c1 = box
    crop = assembled[r0 : r1 + 1, c0 : c1 + 1]
    h, w = crop.shape
    sides_hit: set[str] = set()
    tie_count = 0
    for r in range(h):
        for c in range(w):
            color = int(crop[r, c])
            if color in (BG, RED):
                continue
            distances = {
                "top": r,
                "right": w - 1 - c,
                "bottom": h - 1 - r,
                "left": c,
            }
            best = min(distances.values())
            winners = [side for side, dist in distances.items() if dist == best]
            if len(winners) == 1:
                sides_hit.add(winners[0])
            else:
                tie_count += 1
    return sides_hit, tie_count


def _fits_scatter(grid: np.ndarray, arr: np.ndarray, top: int, left: int) -> bool:
    h, w = grid.shape
    ph, pw = arr.shape
    if top < 0 or left < 0 or top + ph > h or left + pw > w:
        return False
    coords = np.argwhere(arr != BG)
    coord_set = {(top + int(rr), left + int(cc)) for rr, cc in coords}
    for abs_r, abs_c in coord_set:
        if grid[abs_r, abs_c] != BG:
            return False
        for nr in range(abs_r - 1, abs_r + 2):
            for nc in range(abs_c - 1, abs_c + 2):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in coord_set and grid[nr, nc] != BG:
                    return False
    return True


def _sample_scatter_position(grid: np.ndarray, arr: np.ndarray) -> tuple[int, int] | None:
    h, w = grid.shape
    ph, pw = arr.shape
    positions = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
    random.shuffle(positions)
    for top, left in positions:
        if _fits_scatter(grid, arr, top, left):
            return top, left
    return None


def _shift_pieces(pieces: list[dict], dr: int, dc: int) -> list[dict]:
    shifted = []
    for piece in pieces:
        shifted.append({**piece, "top": piece["top"] + dr, "left": piece["left"] + dc})
    return shifted


def _pick_global_shift(pieces: list[dict], h: int, w: int) -> tuple[int, int] | None:
    assembled = _assembled_canvas_from_pieces(pieces)
    box = _bbox_mask(assembled != BG)
    assert box is not None
    r0, c0, r1, c1 = box
    ah = r1 - r0 + 1
    aw = c1 - c0 + 1
    if ah > h or aw > w:
        return None
    row_choices = list(range(-r0, h - r1))
    col_choices = list(range(-c0, w - c1))
    random.shuffle(row_choices)
    random.shuffle(col_choices)
    for dr in row_choices[:12]:
        for dc in col_choices[:12]:
            return dr, dc
    return None


def _attempt_generate() -> np.ndarray | None:
    pieces = _assemble_generated_structure()
    if pieces is None:
        return None
    assembled = _assembled_canvas_from_pieces(pieces)
    box = _bbox_mask(assembled != BG)
    assert box is not None
    r0, c0, r1, c1 = box
    if r1 - r0 + 1 < 6 or c1 - c0 + 1 < 6:
        return None
    side_hits, tie_count = _recolor_stats(assembled)
    if side_hits != set(SIDES) or tie_count == 0:
        return None

    h = random.randint(20, 26)
    w = random.randint(20, 26)
    shift = _pick_global_shift(pieces, h, w)
    if shift is None:
        return None
    pieces = _shift_pieces(pieces, shift[0], shift[1])

    grid = np.zeros((h, w), dtype=int)
    anchor = pieces[0]
    _paste_nonzero(grid, anchor["arr"], anchor["top"], anchor["left"])

    movable = pieces[1:]
    random.shuffle(movable)
    for piece in movable:
        pos = _sample_scatter_position(grid, piece["arr"])
        if pos is None:
            return None
        _paste_nonzero(grid, piece["arr"], pos[0], pos[1])

    expected = _recolor_crop(_assembled_canvas_from_pieces(pieces))
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
