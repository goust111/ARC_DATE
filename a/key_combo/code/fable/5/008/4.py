from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]
ShapeSig = tuple[int, int, tuple[Coord, ...]]
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _background(grid: np.ndarray) -> int:
    """Return the visibly dominant (most frequent) color."""
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
    """4-connected components as (N, 2) arrays using (row, column) coordinates."""
    h, w = map(int, mask.shape)
    unseen = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    result: list[np.ndarray] = []
    while unseen:
        start = next(iter(unseen))
        unseen.remove(start)
        queue = deque([start])
        cells = [start]
        while queue:
            r, c = queue.popleft()
            for dr, dc in DIR4:
                nxt = (r + dr, c + dc)
                if 0 <= nxt[0] < h and 0 <= nxt[1] < w and nxt in unseen:
                    unseen.remove(nxt)
                    queue.append(nxt)
                    cells.append(nxt)
        result.append(np.asarray(cells, dtype=int))
    return result


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    return int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])


def _shape_signature(coords: np.ndarray) -> ShapeSig:
    r0, c0, r1, c1 = _bbox(coords)
    rel = tuple(sorted((int(r) - r0, int(c) - c0) for r, c in coords))
    return r1 - r0 + 1, c1 - c0 + 1, rel


def _foreground_components(grid: np.ndarray, bg: int) -> list[dict]:
    records: list[dict] = []
    for color in set(map(int, grid.ravel())) - {int(bg)}:
        for coords in _mask_components(grid == color):
            records.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(len(coords)),
                    "signature": _shape_signature(coords),
                }
            )
    return records


def _parse(grid: np.ndarray) -> tuple[int, dict, list[dict], list[dict]]:
    """Identify the board, its boundary-open notches, and all scattered pieces."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) > 0
    bg = _background(g)
    records = _foreground_components(g, bg)
    assert records, "the grid must contain a board"

    largest_area = max(int(comp["area"]) for comp in records)
    largest = [comp for comp in records if int(comp["area"]) == largest_area]
    assert len(largest) == 1, "the largest monochrome component must be unique"
    board = largest[0]
    pieces = [comp for comp in records if comp is not board]

    r0, c0, r1, c1 = map(int, board["bbox"])
    board_local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    bc = board["coords"]
    board_local[bc[:, 0] - r0, bc[:, 1] - c0] = True

    notches: list[dict] = []
    for local in _mask_components(~board_local):
        rr, cc = local[:, 0], local[:, 1]
        opens = bool(
            np.any(rr == 0)
            or np.any(rr == board_local.shape[0] - 1)
            or np.any(cc == 0)
            or np.any(cc == board_local.shape[1] - 1)
        )
        assert opens, "the board may contain boundary-open notches but no enclosed holes"
        global_coords = local + np.asarray([r0, c0], dtype=int)
        notches.append(
            {
                "coords": global_coords,
                "area": int(len(global_coords)),
                "signature": _shape_signature(global_coords),
            }
        )

    assert notches, "the board must have at least one notch"
    assert len({n["signature"] for n in notches}) == len(notches), (
        "notch shapes must be distinct so every candidate pair has one destination"
    )
    return bg, board, notches, pieces


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each board notch with the uniquely colored matching piece, recolored by its repeated-color decoy.

    Input structure:
    - The most common color is background; the unique largest monochrome component is a solid board whose background holes open to its bounding-box boundary.
    - For every notch, exactly two scattered monochrome sprites have the same untranslated, unrotated shape. One candidate's color forms one component globally, while the other's color forms multiple components.

    Rule:
    - Select the one-component-color candidate as the winner for each exact shape match, but paint its notch with the color of the multiple-component decoy candidate.
    - Erase every scattered sprite, including winners, decoys, and the extra components that reveal repeated colors; retain only the board and the newly filled notches.

    Key insight:
    - Shape alone deliberately gives two answers: component-count multiplicity rejects one candidate, while that rejected candidate still supplies the winner's output color.
    """
    g = np.asarray(grid, dtype=int)
    bg, board, notches, pieces = _parse(g)

    # Multiplicity is defined over the whole grid, so the board component also
    # participates if a hand-authored valid case reuses its color for a decoy.
    color_component_count = Counter(int(piece["color"]) for piece in pieces)
    color_component_count[int(board["color"])] += 1
    out = np.full(g.shape, bg, dtype=int)
    board_coords = board["coords"]
    out[board_coords[:, 0], board_coords[:, 1]] = int(board["color"])

    used_piece_ids: set[int] = set()
    for notch in notches:
        matches = [piece for piece in pieces if piece["signature"] == notch["signature"]]
        assert len(matches) == 2, "each notch must have exactly two same-shape candidates"

        repeated = [p for p in matches if color_component_count[int(p["color"])] > 1]
        unique = [p for p in matches if color_component_count[int(p["color"])] == 1]
        assert len(repeated) == 1 and len(unique) == 1, (
            "each pair must contain one repeated-color decoy and one unique-color winner"
        )
        assert int(repeated[0]["color"]) != int(unique[0]["color"])
        assert id(repeated[0]) not in used_piece_ids and id(unique[0]) not in used_piece_ids
        used_piece_ids.update((id(repeated[0]), id(unique[0])))

        dst = notch["coords"]
        out[dst[:, 0], dst[:, 1]] = int(repeated[0]["color"])

    return out


# ------------------------------- generator -------------------------------

_GENERATION_RETRIES = 400
_NOTCH_PHASE = random.randrange(2)


def _random_polyomino() -> np.ndarray:
    """Generate a small irregular 4-connected mask with a tight 2-4 by 2-4 bbox."""
    for _ in range(200):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        min_area = h + w - 1
        max_area = min(h * w, 8)
        if min_area > max_area:
            continue
        target = random.randint(min_area, max_area)
        cells = {(random.randrange(h), random.randrange(w))}
        while len(cells) < target:
            frontier = {
                (r + dr, c + dc)
                for r, c in cells
                for dr, dc in DIR4
                if 0 <= r + dr < h and 0 <= c + dc < w and (r + dr, c + dc) not in cells
            }
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))
        if len(cells) != target:
            continue
        rows = {r for r, _ in cells}
        cols = {c for _, c in cells}
        if min(rows) != 0 or max(rows) != h - 1 or min(cols) != 0 or max(cols) != w - 1:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if len(_mask_components(mask)) == 1:
            return mask
    raise RuntimeError("failed to sample a connected notch shape")


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = map(int, mask.shape)
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            src_r0, src_r1 = max(0, -dr), h - max(0, dr)
            src_c0, src_c1 = max(0, -dc), w - max(0, dc)
            dst_r0, dst_r1 = max(0, dr), h - max(0, -dr)
            dst_c0, dst_c1 = max(0, dc), w - max(0, -dc)
            out[dst_r0:dst_r1, dst_c0:dst_c1] |= mask[src_r0:src_r1, src_c0:src_c1]
    return out


def _place_notch(
    board_mask: np.ndarray,
    shape: np.ndarray,
    side: str,
    prior_holes: set[Coord],
) -> set[Coord] | None:
    bh, bw = map(int, board_mask.shape)
    h, w = map(int, shape.shape)
    options: list[tuple[int, int]] = []
    if side == "top" and w <= bw - 2:
        options = [(0, c) for c in range(1, bw - w)]
    elif side == "bottom" and w <= bw - 2:
        options = [(bh - h, c) for c in range(1, bw - w)]
    elif side == "left" and h <= bh - 2:
        options = [(r, 0) for r in range(1, bh - h)]
    elif side == "right" and h <= bh - 2:
        options = [(r, bw - w) for r in range(1, bh - h)]
    random.shuffle(options)

    forbidden = {
        (r + dr, c + dc)
        for r, c in prior_holes
        for dr, dc in DIR4 + ((0, 0),)
    }
    rel = np.argwhere(shape)
    for r0, c0 in options:
        cells = {(r0 + int(r), c0 + int(c)) for r, c in rel}
        if cells & forbidden:
            continue
        trial = board_mask.copy()
        for r, c in cells:
            trial[r, c] = False
        if len(_mask_components(trial)) != 1:
            continue
        empties = _mask_components(~trial)
        if len(empties) != len(_mask_components(~board_mask)) + 1:
            continue
        board_mask[:] = trial
        return cells
    return None


def _outside_board_bbox(
    r0: int,
    c0: int,
    h: int,
    w: int,
    board_bbox: tuple[int, int, int, int],
) -> bool:
    br0, bc0, br1, bc1 = board_bbox
    return r0 + h - 1 < br0 - 1 or r0 > br1 + 1 or c0 + w - 1 < bc0 - 1 or c0 > bc1 + 1


def _place_sprite(
    grid: np.ndarray,
    occupancy: np.ndarray,
    mask: np.ndarray,
    color: int,
    board_bbox: tuple[int, int, int, int],
) -> np.ndarray | None:
    h, w = map(int, mask.shape)
    H, W = map(int, grid.shape)
    blocked = _dilate8(occupancy)
    candidates: list[tuple[int, int]] = []
    for r0 in range(1, H - h):
        for c0 in range(1, W - w):
            if not _outside_board_bbox(r0, c0, h, w, board_bbox):
                continue
            if np.any(blocked[r0 : r0 + h, c0 : c0 + w] & mask):
                continue
            candidates.append((r0, c0))
    if not candidates:
        return None

    r0, c0 = random.choice(candidates)
    rel = np.argwhere(mask)
    coords = rel + np.asarray([r0, c0], dtype=int)
    grid[coords[:, 0], coords[:, 1]] = int(color)
    occupancy[coords[:, 0], coords[:, 1]] = True
    return coords


def _witness_mask() -> np.ndarray:
    choices = (
        np.asarray([[1, 1]], dtype=bool),
        np.asarray([[1], [1]], dtype=bool),
        np.asarray([[1, 0], [1, 1]], dtype=bool),
        np.asarray([[1, 1], [0, 1]], dtype=bool),
    )
    return random.choice(choices).copy()


def _semantic_validate(grid: np.ndarray, expected: np.ndarray, meta: dict) -> bool:
    """Check the stated semantics independently of merely closing main(generate_input())."""
    try:
        bg = int(meta["bg"])
        board_color = int(meta["board_color"])
        notches: list[dict] = meta["notches"]
        objects: list[dict] = meta["objects"]

        counts = Counter(map(int, grid.ravel()))
        if counts[bg] <= max(v for k, v in counts.items() if k != bg):
            return False
        if len(notches) not in (2, 3):
            return False
        if len({item["signature"] for item in notches}) != len(notches):
            return False

        parsed_bg, parsed_board, parsed_notches, parsed_pieces = _parse(grid)
        if parsed_bg != bg or int(parsed_board["color"]) != board_color:
            return False
        if {n["signature"] for n in parsed_notches} != {n["signature"] for n in notches}:
            return False

        piece_counts = Counter(int(p["color"]) for p in parsed_pieces)
        decoy_colors = [int(n["decoy_color"]) for n in notches]
        winner_colors = [int(n["winner_color"]) for n in notches]
        if len(set(decoy_colors)) != len(notches) or len(set(winner_colors)) != len(notches):
            return False
        if set(decoy_colors) & set(winner_colors):
            return False

        for notch in notches:
            sig = notch["signature"]
            matches = [p for p in parsed_pieces if p["signature"] == sig]
            if len(matches) != 2:
                return False
            roles = {int(p["color"]): piece_counts[int(p["color"])] for p in matches}
            dc, wc = int(notch["decoy_color"]), int(notch["winner_color"])
            if roles != {dc: 2, wc: 1}:
                return False
            dst = notch["coords"]
            if not np.all(expected[dst[:, 0], dst[:, 1]] == dc):
                return False
            # Counterfactual: keeping the winner's color must change the result.
            if dc == wc:
                return False

        notch_sigs = {n["signature"] for n in notches}
        for obj in objects:
            coords = obj["coords"]
            if not np.all(expected[coords[:, 0], coords[:, 1]] == bg):
                return False
            if obj["role"] == "witness" and obj["signature"] in notch_sigs:
                return False

        if np.array_equal(grid, expected):
            return False
        solved = main(grid)
        if not np.array_equal(solved, expected):
            return False

        # The output must contain every pair-specific loser color and no winner color.
        out_colors = set(map(int, expected.ravel()))
        if not set(decoy_colors).issubset(out_colors) or set(winner_colors) & out_colors:
            return False
        return True
    except (AssertionError, ValueError, RuntimeError):
        return False


def _generation_attempt(notch_count: int) -> np.ndarray | None:
    H = random.randint(27 if notch_count == 3 else 24, 30)
    W = random.randint(27 if notch_count == 3 else 24, 30)
    bh = random.randint(8, min(11, H - 12))
    bw = random.randint(9, min(13, W - 12))

    br_lo, br_hi = 5, H - bh - 5
    bc_lo, bc_hi = 5, W - bw - 5
    if br_lo > br_hi or bc_lo > bc_hi:
        return None
    br0 = random.randint(br_lo, br_hi)
    bc0 = random.randint(bc_lo, bc_hi)
    br1, bc1 = br0 + bh - 1, bc0 + bw - 1

    board_mask = np.ones((bh, bw), dtype=bool)
    sides = random.sample(["top", "bottom", "left", "right"], notch_count)
    signatures: set[ShapeSig] = set()
    prior_holes: set[Coord] = set()
    notch_specs: list[dict] = []

    for side in sides:
        placed = False
        for _ in range(120):
            shape = _random_polyomino()
            sig = _shape_signature(np.argwhere(shape))
            if sig in signatures:
                continue
            cells = _place_notch(board_mask, shape, side, prior_holes)
            if cells is None:
                continue
            global_coords = np.asarray(
                [(br0 + r, bc0 + c) for r, c in sorted(cells)], dtype=int
            )
            signatures.add(sig)
            prior_holes |= cells
            notch_specs.append(
                {
                    "side": side,
                    "shape": shape,
                    "signature": sig,
                    "coords": global_coords,
                }
            )
            placed = True
            break
        if not placed:
            return None

    palette = random.sample(range(10), 2 + 2 * notch_count)
    bg, board_color = palette[:2]
    winner_colors = palette[2 : 2 + notch_count]
    decoy_colors = palette[2 + notch_count :]

    grid = np.full((H, W), bg, dtype=int)
    board_coords_local = np.argwhere(board_mask)
    board_coords = board_coords_local + np.asarray([br0, bc0], dtype=int)
    grid[board_coords[:, 0], board_coords[:, 1]] = board_color
    occupancy = grid != bg

    object_jobs: list[dict] = []
    for i, notch in enumerate(notch_specs):
        notch["winner_color"] = int(winner_colors[i])
        notch["decoy_color"] = int(decoy_colors[i])
        object_jobs.extend(
            [
                {"index": i, "role": "winner", "mask": notch["shape"], "color": winner_colors[i]},
                {"index": i, "role": "decoy", "mask": notch["shape"], "color": decoy_colors[i]},
                {"index": i, "role": "witness", "mask": _witness_mask(), "color": decoy_colors[i]},
            ]
        )
    random.shuffle(object_jobs)

    placed_objects: list[dict] = []
    board_bbox = (br0, bc0, br1, bc1)
    for job in object_jobs:
        coords = _place_sprite(grid, occupancy, job["mask"], int(job["color"]), board_bbox)
        if coords is None:
            return None
        placed_objects.append(
            {
                **job,
                "coords": coords,
                "signature": _shape_signature(coords),
            }
        )

    expected = np.full((H, W), bg, dtype=int)
    expected[board_coords[:, 0], board_coords[:, 1]] = board_color
    for notch in notch_specs:
        coords = notch["coords"]
        expected[coords[:, 0], coords[:, 1]] = int(notch["decoy_color"])

    meta = {
        "bg": bg,
        "board_color": board_color,
        "notches": notch_specs,
        "objects": placed_objects,
    }
    if not _semantic_validate(grid, expected, meta):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate alternating 2/3-notch instances while randomizing all visible geometry and colors."""
    global _NOTCH_PHASE
    notch_count = 2 + (_NOTCH_PHASE % 2)
    for _ in range(_GENERATION_RETRIES):
        grid = _generation_attempt(notch_count)
        if grid is not None:
            _NOTCH_PHASE += 1
            return grid
    raise RuntimeError(f"failed to generate a valid {notch_count}-notch decoy-board instance")
