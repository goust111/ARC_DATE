from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _shape_equal(a: np.ndarray, b: np.ndarray) -> bool:
    for k in range(4):
        rot = np.rot90(b, k)
        if a.shape == rot.shape and np.array_equal(a, rot):
            return True
    return False


def _component_dict(coords: list[tuple[int, int]], color: int | None = None) -> dict:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return {
        "coords": coords,
        "bbox": (r0, c0, r1, c1),
        "mask": mask,
        "area": len(coords),
        "color": color,
    }


def _components_from_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros(grid.shape, dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color == bg:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(_component_dict(coords, color))
    comps.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["area"]))
    return comps


def _find_blob_and_holes(grid: np.ndarray) -> tuple[int, dict, list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    comps = _monochrome_components(g, bg)
    assert comps

    max_area = max(comp["area"] for comp in comps)
    blob_candidates = [comp for comp in comps if comp["area"] == max_area]
    assert len(blob_candidates) == 1
    blob = blob_candidates[0]
    pieces = [comp for comp in comps if comp is not blob]

    blob_mask = np.zeros(g.shape, dtype=bool)
    for r, c in blob["coords"]:
        blob_mask[r, c] = True

    holes: list[dict] = []
    for coords in _components_from_mask(g == bg):
        if any(r == 0 or c == 0 or r == g.shape[0] - 1 or c == g.shape[1] - 1 for r, c in coords):
            continue
        coord_set = set(coords)
        enclosed = True
        for r, c in coords:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in coord_set:
                    continue
                if not blob_mask[nr, nc]:
                    enclosed = False
                    break
            if not enclosed:
                break
        if enclosed:
            holes.append(_component_dict(coords))

    holes.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["area"]))
    pieces.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["area"]))
    return bg, blob, holes, pieces


def _complement_match(hole_mask: np.ndarray, piece_mask: np.ndarray) -> bool:
    hole_cells = {(int(r), int(c)) for r, c in np.argwhere(hole_mask)}
    hole_h, hole_w = hole_mask.shape
    total_area = int(hole_mask.sum() + piece_mask.sum())

    for k in range(4):
        rot = np.rot90(piece_mask, k)
        piece_cells = [(int(r), int(c)) for r, c in np.argwhere(rot)]
        piece_h, piece_w = rot.shape
        for dr in range(-piece_h, hole_h + 1):
            for dc in range(-piece_w, hole_w + 1):
                shifted = {(r + dr, c + dc) for r, c in piece_cells}
                if hole_cells & shifted:
                    continue
                union = hole_cells | shifted
                rs = [r for r, _ in union]
                cs = [c for _, c in union]
                bbox_area = (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1)
                if bbox_area == total_area:
                    return True
    return False


def _unique_matching(adjacency: list[list[int]]) -> dict[int, int]:
    n = len(adjacency)
    order = sorted(range(n), key=lambda idx: (len(adjacency[idx]), idx))
    used = [False] * n
    current: dict[int, int] = {}
    solutions: list[dict[int, int]] = []

    def dfs(pos: int) -> None:
        if len(solutions) > 1:
            return
        if pos == n:
            solutions.append(current.copy())
            return
        hole_idx = order[pos]
        for piece_idx in adjacency[hole_idx]:
            if used[piece_idx]:
                continue
            used[piece_idx] = True
            current[hole_idx] = piece_idx
            dfs(pos + 1)
            del current[hole_idx]
            used[piece_idx] = False

    dfs(0)
    assert len(solutions) == 1
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed hole of the big monochrome blob with the color of its complementary outside key piece.

    Input structure:
    - The grid contains one large single-color connected blob on a background, with several enclosed
      background holes inside it.
    - Outside the blob are the same number of small monochrome colored pieces, one connected component per piece.
    - No piece is shape-equal to a hole, but after translation and rotation each hole has exactly one piece
      whose union with it forms a solid rectangle.

    Rule:
    - Find the largest monochrome component as the blob, then find the background components fully enclosed by it as holes.
    - Match each hole to the unique outside piece whose rotated shape complements the hole to fill some rectangle exactly.
    - Recolor every hole cell with the matched piece color, and erase all outside pieces back to background.

    Key insight:
    - The pairing signal is negative-space complementarity, not same-shape matching: each key is the missing remainder
      of a rectangle around its hole, so the piece color tells which hole should be filled.
    """
    g = np.asarray(grid, dtype=int)
    bg, _blob, holes, pieces = _find_blob_and_holes(g)
    assert holes and len(holes) == len(pieces)

    adjacency: list[list[int]] = []
    for hole in holes:
        matches = [idx for idx, piece in enumerate(pieces) if _complement_match(hole["mask"], piece["mask"])]
        assert matches
        adjacency.append(matches)

    assignment = _unique_matching(adjacency)

    out = g.copy()
    for piece in pieces:
        for r, c in piece["coords"]:
            out[r, c] = bg
    for hole_idx, piece_idx in assignment.items():
        fill_color = int(pieces[piece_idx]["color"])
        for r, c in holes[hole_idx]["coords"]:
            out[r, c] = fill_color
    return out


# --------------- generator ---------------

_SHAPE_SPECS = [
    (3, 3, (1, 1, 2)),
    (3, 3, (2, 1, 1)),
    (3, 3, (2, 1, 2)),
    (3, 4, (1, 3, 1)),
    (4, 3, (1, 1, 1, 2)),
    (4, 4, (1, 1, 2, 2)),
    (4, 4, (1, 1, 3, 2)),
]

_SLOT_ORDER = {
    2: ("tl", "br"),
    3: ("tl", "tr", "bl"),
    4: ("tl", "tr", "br", "bl"),
}


def _make_pair(spec: tuple[int, int, tuple[int, ...]]) -> tuple[np.ndarray, np.ndarray]:
    h, w, widths = spec
    hole = np.zeros((h, w), dtype=bool)
    for r, width in enumerate(widths):
        hole[r, :width] = True
    piece = ~hole
    return _crop_mask(hole), _crop_mask(piece)


_LIBRARY = []
for idx, spec in enumerate(_SHAPE_SPECS):
    hole_mask, piece_mask = _make_pair(spec)
    _LIBRARY.append(
        {
            "index": idx,
            "spec": spec,
            "hole": hole_mask,
            "piece": piece_mask,
            "hole_area": int(hole_mask.sum()),
            "piece_area": int(piece_mask.sum()),
        }
    )


def _validate_library() -> None:
    for i, lhs in enumerate(_LIBRARY):
        assert _complement_match(lhs["hole"], lhs["piece"])
        for j, rhs in enumerate(_LIBRARY):
            if i == j:
                continue
            assert not _complement_match(lhs["hole"], rhs["piece"])
            assert not _shape_equal(lhs["hole"], rhs["piece"])
            assert not _shape_equal(lhs["hole"], rhs["hole"])
            assert not _shape_equal(lhs["piece"], rhs["piece"])


_validate_library()


def _choose_entries(count: int) -> tuple[list[dict], bool]:
    repeat_groups = [[0, 1], [3, 4]]
    require_repeat = count >= 2 and random.random() < 0.8
    if require_repeat and count == 4 and random.random() < 0.4:
        chosen = random.sample(repeat_groups[0], 2) + random.sample(repeat_groups[1], 2)
    elif require_repeat:
        base = random.choice(repeat_groups)
        chosen = random.sample(base, 2)
        remaining = [idx for idx in range(len(_LIBRARY)) if idx not in chosen]
        chosen += random.sample(remaining, count - 2)
    else:
        chosen = random.sample(range(len(_LIBRARY)), count)
    random.shuffle(chosen)
    return [_LIBRARY[idx] for idx in chosen], require_repeat


def _layout_holes(entries: list[dict], bg: int, blob_color: int) -> tuple[np.ndarray, list[dict]]:
    items = []
    for entry in entries:
        hole_rot = random.randrange(4)
        hole_mask = np.rot90(entry["hole"], hole_rot)
        items.append(
            {
                "entry": entry,
                "hole_rot": hole_rot,
                "hole_mask": hole_mask,
                "height": int(hole_mask.shape[0]),
                "width": int(hole_mask.shape[1]),
                "color": None,
            }
        )

    random.shuffle(items)
    count = len(items)
    if count == 2:
        row_sizes = [2] if random.random() < 0.45 else [1, 1]
    elif count == 3:
        row_sizes = [3] if random.random() < 0.35 else random.choice(([1, 2], [2, 1]))
    else:
        row_sizes = random.choice(([2, 2], [1, 3], [3, 1]))

    rows = []
    cursor = 0
    for size in row_sizes:
        rows.append(items[cursor : cursor + size])
        cursor += size

    gap = 2
    margin = 1
    row_widths = [sum(item["width"] for item in row) + gap * (len(row) - 1) for row in rows]
    row_heights = [max(item["height"] for item in row) for row in rows]
    extra_w = random.randint(0, 1)
    extra_h = random.randint(0, 1)
    blob_w = max(row_widths) + 2 * margin + extra_w
    blob_h = sum(row_heights) + 2 * margin + gap * (len(rows) - 1) + extra_h

    blob = np.full((blob_h, blob_w), blob_color, dtype=int)
    out_items: list[dict] = []
    row_top = margin
    for row, row_width, row_height in zip(rows, row_widths, row_heights):
        start_c = margin + random.randint(0, blob_w - 2 * margin - row_width)
        c = start_c
        for item in row:
            r = row_top + random.randint(0, row_height - item["height"])
            mask = item["hole_mask"]
            blob[r : r + item["height"], c : c + item["width"]][mask] = bg
            bbox = (r, c, r + item["height"] - 1, c + item["width"] - 1)
            out_items.append({**item, "bbox": bbox})
            c += item["width"] + gap
        row_top += row_height + gap

    out_items.sort(key=lambda item: (item["bbox"][0], item["bbox"][1], item["entry"]["index"]))
    return blob, out_items


def _piece_rotation(item: dict, rank: int) -> int:
    return (item["hole_rot"] + item["entry"]["index"] + rank + 1) % 4


def _compose_input(entries: list[dict], require_repeat: bool) -> tuple[np.ndarray, np.ndarray] | None:
    bg = 0
    blob_color = random.choice(list(range(1, 10)))
    colors = random.sample([c for c in range(1, 10) if c != blob_color], len(entries))

    blob, hole_items = _layout_holes(entries, bg, blob_color)
    for rank, (item, color) in enumerate(zip(hole_items, colors)):
        item["color"] = color
        piece_rot = _piece_rotation(item, rank)
        item["piece_mask"] = np.rot90(item["entry"]["piece"], piece_rot)

    slots = _SLOT_ORDER[len(entries)]
    slot_to_item = {slot: item for slot, item in zip(slots, hole_items)}
    top_slots = [slot for slot in slots if slot.startswith("t")]
    bottom_slots = [slot for slot in slots if slot.startswith("b")]
    left_slots = [slot for slot in slots if slot.endswith("l")]
    right_slots = [slot for slot in slots if slot.endswith("r")]

    top_band = max((slot_to_item[slot]["piece_mask"].shape[0] for slot in top_slots), default=0) + 1
    bottom_band = max((slot_to_item[slot]["piece_mask"].shape[0] for slot in bottom_slots), default=0) + 1
    left_band = max((slot_to_item[slot]["piece_mask"].shape[1] for slot in left_slots), default=0) + 1
    right_band = max((slot_to_item[slot]["piece_mask"].shape[1] for slot in right_slots), default=0) + 1

    outer_top = random.randint(0, 1)
    outer_bottom = random.randint(0, 1)
    outer_left = random.randint(0, 1)
    outer_right = random.randint(0, 1)

    blob_r0 = outer_top + top_band
    blob_c0 = outer_left + left_band
    grid_h = outer_top + top_band + blob.shape[0] + bottom_band + outer_bottom
    grid_w = outer_left + left_band + blob.shape[1] + right_band + outer_right
    if max(grid_h, grid_w) > 30:
        return None

    inp = np.full((grid_h, grid_w), bg, dtype=int)
    inp[blob_r0 : blob_r0 + blob.shape[0], blob_c0 : blob_c0 + blob.shape[1]] = blob

    for slot, item in slot_to_item.items():
        piece_mask = item["piece_mask"]
        h, w = piece_mask.shape
        if slot == "tl":
            r0 = blob_r0 - 1 - h
            c0 = blob_c0 - 1 - w
        elif slot == "tr":
            r0 = blob_r0 - 1 - h
            c0 = blob_c0 + blob.shape[1] + 1
        elif slot == "br":
            r0 = blob_r0 + blob.shape[0] + 1
            c0 = blob_c0 + blob.shape[1] + 1
        else:
            r0 = blob_r0 + blob.shape[0] + 1
            c0 = blob_c0 - 1 - w
        if r0 < 0 or c0 < 0 or r0 + h > grid_h or c0 + w > grid_w:
            return None
        inp[r0 : r0 + h, c0 : c0 + w][piece_mask] = int(item["color"])
        item["piece_bbox"] = (r0, c0, r0 + h - 1, c0 + w - 1)

    expected = inp.copy()
    expected[blob_r0 : blob_r0 + blob.shape[0], blob_c0 : blob_c0 + blob.shape[1]] = blob
    for item in hole_items:
        hr0, hc0, hr1, hc1 = item["bbox"]
        view = expected[blob_r0 + hr0 : blob_r0 + hr1 + 1, blob_c0 + hc0 : blob_c0 + hc1 + 1]
        view[item["hole_mask"]] = int(item["color"])
    for item in hole_items:
        pr0, pc0, pr1, pc1 = item["piece_bbox"]
        view = expected[pr0 : pr1 + 1, pc0 : pc1 + 1]
        view[item["piece_mask"]] = bg

    if require_repeat:
        piece_areas = [int(item["piece_mask"].sum()) for item in hole_items]
        if len(set(piece_areas)) == len(piece_areas):
            return None

    try:
        bg2, blob2, holes2, pieces2 = _find_blob_and_holes(inp)
        if bg2 != bg or int(blob2["color"]) != blob_color:
            return None
        if len(holes2) != len(hole_items) or len(pieces2) != len(hole_items):
            return None
        for hole in holes2:
            if hole["mask"].all():
                return None
        adjacency = [
            [idx for idx, piece in enumerate(pieces2) if _complement_match(hole["mask"], piece["mask"])]
            for hole in holes2
        ]
        _unique_matching(adjacency)
        for hole in holes2:
            for piece in pieces2:
                if _shape_equal(hole["mask"], piece["mask"]):
                    return None
        if not np.array_equal(main(inp), expected):
            return None
    except Exception:
        return None

    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(400):
        count = random.choice([2, 3, 4])
        entries, require_repeat = _choose_entries(count)
        built = _compose_input(entries, require_repeat)
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate valid input for puzzle 6")
