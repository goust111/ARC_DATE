from __future__ import annotations

import random

import numpy as np


BLACK = 0
PLATE_COLOR = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("left", "right", "top", "bottom")
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

GEN_RETRIES = 400
SHAPE_RETRIES = 120
NOTCH_RETRIES = 120
MARKER_RETRIES = 120
PLACEMENT_RETRIES = 120


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _tight_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (r0, c0)


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    m = np.asarray(mask, dtype=np.uint8)
    return tuple(map(int, m.shape)), m.tobytes()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each edge notch of the large plate with the horizontally flipped matching fragment.

    The fragment's output color is not its own color. Instead, look for the singleton
    marker inside the plate that shares the fragment color, then use the other singleton
    marker on that same row as the fill color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    non_black = [int(color) for color in np.unique(g) if int(color) != BLACK]
    assert non_black, "expected at least one non-background color"
    plate_color = max(non_black, key=lambda color: int(np.count_nonzero(g == color)))

    plate_coords = np.argwhere(g == plate_color)
    assert plate_coords.size, "plate color must be present"
    pr0, pc0, pr1, pc1 = _bbox(plate_coords)
    inside = g[pr0 : pr1 + 1, pc0 : pc1 + 1]

    marker_cells = np.argwhere((inside != BLACK) & (inside != plate_color))
    assert marker_cells.size, "expected interior marker pixels"

    color_map: dict[int, int] = {}
    rows: dict[int, list[tuple[int, int]]] = {}
    for rr, cc in marker_cells:
        rows.setdefault(int(rr + pr0), []).append((int(cc + pc0), int(inside[rr, cc])))
    for row, items in rows.items():
        assert len(items) == 2, "each marker row must contain exactly two pixels"
        items.sort()
        (_, c1), (_, c2) = items
        assert c1 != c2, "paired marker colors must differ"
        assert c1 not in color_map and c2 not in color_map, "marker colors must be unique"
        color_map[c1] = c2
        color_map[c2] = c1

    marker_count = 0
    pieces: list[dict[str, np.ndarray | int]] = []
    object_mask = (g != BLACK) & (g != plate_color)
    for coords in _components(object_mask):
        colors = {int(g[r, c]) for r, c in coords}
        assert len(colors) == 1, "each colored object must be monochrome"
        color = colors.pop()
        inside_plate = (
            np.all(coords[:, 0] >= pr0)
            and np.all(coords[:, 0] <= pr1)
            and np.all(coords[:, 1] >= pc0)
            and np.all(coords[:, 1] <= pc1)
        )
        outside_plate = (
            np.all((coords[:, 0] < pr0) | (coords[:, 0] > pr1) | (coords[:, 1] < pc0) | (coords[:, 1] > pc1))
        )
        assert inside_plate or outside_plate, "objects must lie fully inside or outside the plate box"
        if inside_plate:
            assert len(coords) == 1, "interior colored objects must be singleton markers"
            marker_count += 1
            continue
        mask, _ = _tight_mask(coords)
        pieces.append({"coords": coords, "color": color, "mask": mask})

    assert marker_count == len(marker_cells), "all interior colored pixels must be singleton markers"

    notches: list[dict[str, np.ndarray]] = []
    for coords in _components(inside == BLACK):
        global_coords = coords + np.array([pr0, pc0], dtype=int)
        mask, _ = _tight_mask(global_coords)
        notches.append({"coords": global_coords, "mask": mask})

    assert pieces, "expected at least one external fragment"
    assert len(pieces) == len(notches), "fragment count must equal notch count"

    notch_by_sig: dict[tuple[tuple[int, int], bytes], dict[str, np.ndarray]] = {}
    for notch in notches:
        sig = _mask_signature(notch["mask"])
        assert sig not in notch_by_sig, "notch shapes must be unique"
        notch_by_sig[sig] = notch

    out = g.copy()
    for piece in pieces:
        piece_color = int(piece["color"])
        assert piece_color in color_map, "each fragment color must appear in the plate markers"
        flipped_sig = _mask_signature(np.fliplr(np.asarray(piece["mask"], dtype=bool)))
        assert flipped_sig in notch_by_sig, "each fragment must match a notch after horizontal flip"
        notch = notch_by_sig.pop(flipped_sig)
        coords = np.asarray(piece["coords"], dtype=int)
        out[coords[:, 0], coords[:, 1]] = BLACK
        notch_coords = np.asarray(notch["coords"], dtype=int)
        out[notch_coords[:, 0], notch_coords[:, 1]] = color_map[piece_color]

    assert not notch_by_sig, "all notches must be filled"
    return out


def _canonical_profile() -> np.ndarray | None:
    h = random.randint(3, 5)
    w = random.randint(3, 5)
    depths = [random.randint(1, w - 1)]
    for _ in range(1, h):
        prev = depths[-1]
        lo = max(1, prev - 2)
        hi = min(w, prev + 2)
        depths.append(random.randint(lo, hi))
    depths[random.randrange(h)] = w
    if min(depths) == w:
        return None

    mask = np.zeros((h, w), dtype=bool)
    for r, depth in enumerate(depths):
        mask[r, :depth] = True
    area = int(mask.sum())
    if area < 5 or area > 16:
        return None
    return mask


def _orient(mask: np.ndarray, side: str) -> np.ndarray:
    if side == "left":
        return mask.copy()
    if side == "right":
        return np.fliplr(mask)
    if side == "top":
        return mask.T
    if side == "bottom":
        return np.flipud(mask.T)
    raise ValueError(f"unknown side {side}")


def _choose_shapes(sides: list[str]) -> list[dict[str, object]] | None:
    seen: set[tuple[tuple[int, int], bytes]] = set()
    specs: list[dict[str, object]] = []
    for side in sides:
        placed = False
        for _ in range(SHAPE_RETRIES):
            base = _canonical_profile()
            if base is None:
                continue
            notch_mask = _orient(base, side)
            if np.array_equal(notch_mask, np.fliplr(notch_mask)):
                continue
            sig = _mask_signature(notch_mask)
            if sig in seen:
                continue
            seen.add(sig)
            specs.append({"side": side, "notch_mask": notch_mask, "piece_mask": np.fliplr(notch_mask)})
            placed = True
            break
        if not placed:
            return None
    return specs


def _notch_coords(mask: np.ndarray, top: int, left: int) -> np.ndarray:
    return np.argwhere(mask) + np.array([top, left], dtype=int)


def _can_place_notch(used: np.ndarray, coords: np.ndarray) -> bool:
    h, w = used.shape
    for r, c in coords:
        if used[r, c]:
            return False
        for dr, dc in DIR8:
            nr, nc = int(r) + dr, int(c) + dc
            if 0 <= nr < h and 0 <= nc < w and used[nr, nc]:
                return False
    return True


def _place_notches(plate_h: int, plate_w: int, specs: list[dict[str, object]]) -> list[dict[str, object]] | None:
    for _ in range(NOTCH_RETRIES):
        used = np.zeros((plate_h, plate_w), dtype=bool)
        placed: list[dict[str, object]] = []
        ok = True
        order = specs[:]
        random.shuffle(order)
        for spec in order:
            side = str(spec["side"])
            mask = np.asarray(spec["notch_mask"], dtype=bool)
            mh, mw = mask.shape
            candidates: list[tuple[int, int]] = []
            if side in ("left", "right"):
                if plate_h - mh <= 2:
                    ok = False
                    break
                left = 0 if side == "left" else plate_w - mw
                for top in range(1, plate_h - mh):
                    candidates.append((top, left))
            else:
                if plate_w - mw <= 2:
                    ok = False
                    break
                top = 0 if side == "top" else plate_h - mh
                for left in range(1, plate_w - mw):
                    candidates.append((top, left))
            random.shuffle(candidates)
            chosen = None
            for top, left in candidates:
                coords = _notch_coords(mask, top, left)
                if _can_place_notch(used, coords):
                    chosen = (top, left, coords)
                    used[coords[:, 0], coords[:, 1]] = True
                    break
            if chosen is None:
                ok = False
                break
            top, left, coords = chosen
            entry = dict(spec)
            entry["notch_top"] = top
            entry["notch_left"] = left
            entry["notch_coords_local"] = coords
            placed.append(entry)
        if ok and len(placed) == len(specs):
            return placed
    return None


def _marker_rows(plate_h: int, count: int) -> list[int] | None:
    rows = list(range(2, plate_h - 2))
    random.shuffle(rows)
    chosen: list[int] = []
    for row in rows:
        if any(abs(row - prev) < 2 for prev in chosen):
            continue
        chosen.append(row)
        if len(chosen) == count:
            return chosen
    return None


def _place_markers(plate_h: int, plate_w: int, notch_mask: np.ndarray, color_pairs: list[tuple[int, int]]) -> list[tuple[int, int, int]] | None:
    for _ in range(MARKER_RETRIES):
        rows = _marker_rows(plate_h, len(color_pairs))
        if rows is None:
            return None
        markers: list[tuple[int, int, int]] = []
        ok = True
        for row, (c1, c2) in zip(rows, color_pairs):
            cols = [col for col in range(2, plate_w - 2) if not notch_mask[row, col]]
            candidates = [(a, b) for a in cols for b in cols if a + 1 < b]
            if not candidates:
                ok = False
                break
            col1, col2 = random.choice(candidates)
            markers.append((row, col1, c1))
            markers.append((row, col2, c2))
        if ok:
            return markers
    return None


def _boxes_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _expand_box(box: tuple[int, int, int, int], pad: int, h: int, w: int) -> tuple[int, int, int, int]:
    return (
        max(0, box[0] - pad),
        max(0, box[1] - pad),
        min(h - 1, box[2] + pad),
        min(w - 1, box[3] + pad),
    )


def _place_pieces(
    grid_h: int,
    grid_w: int,
    plate_box: tuple[int, int, int, int],
    specs: list[dict[str, object]],
) -> list[dict[str, object]] | None:
    blocked = [_expand_box(plate_box, 1, grid_h, grid_w)]
    placed: list[dict[str, object]] = []
    order = sorted(specs, key=lambda spec: int(np.asarray(spec["piece_mask"], dtype=bool).sum()), reverse=True)
    for spec in order:
        piece_mask = np.asarray(spec["piece_mask"], dtype=bool)
        ph, pw = piece_mask.shape
        candidates: list[tuple[int, int]] = []
        for top in range(1, grid_h - ph):
            for left in range(1, grid_w - pw):
                box = (top, left, top + ph - 1, left + pw - 1)
                if any(_boxes_intersect(box, forbidden) for forbidden in blocked):
                    continue
                candidates.append((top, left))
        if not candidates:
            return None
        top, left = random.choice(candidates)
        box = (top, left, top + ph - 1, left + pw - 1)
        entry = dict(spec)
        entry["piece_top"] = top
        entry["piece_left"] = left
        entry["piece_box"] = box
        placed.append(entry)
        blocked.append(_expand_box(box, 1, grid_h, grid_w))
    return placed


def _render_case(
    grid_h: int,
    grid_w: int,
    plate_top: int,
    plate_left: int,
    plate_h: int,
    plate_w: int,
    specs: list[dict[str, object]],
    markers_local: list[tuple[int, int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((grid_h, grid_w), BLACK, dtype=int)
    inp[plate_top : plate_top + plate_h, plate_left : plate_left + plate_w] = PLATE_COLOR

    out = inp.copy()
    for spec in specs:
        notch_coords = np.asarray(spec["notch_coords_local"], dtype=int) + np.array([plate_top, plate_left], dtype=int)
        inp[notch_coords[:, 0], notch_coords[:, 1]] = BLACK
        out[notch_coords[:, 0], notch_coords[:, 1]] = int(spec["target_color"])

    for row, col, color in markers_local:
        rr = plate_top + row
        cc = plate_left + col
        inp[rr, cc] = color
        out[rr, cc] = color

    for spec in specs:
        piece_coords = _notch_coords(np.asarray(spec["piece_mask"], dtype=bool), int(spec["piece_top"]), int(spec["piece_left"]))
        inp[piece_coords[:, 0], piece_coords[:, 1]] = int(spec["piece_color"])
        out[piece_coords[:, 0], piece_coords[:, 1]] = BLACK

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    piece_count = random.choice((2, 4))
    sides = random.sample(SIDES, piece_count)
    specs = _choose_shapes(sides)
    if specs is None:
        return None

    max_h = max(int(np.asarray(spec["notch_mask"]).shape[0]) for spec in specs)
    max_w = max(int(np.asarray(spec["notch_mask"]).shape[1]) for spec in specs)
    plate_h = random.randint(max(10, max_h + 5), 16)
    plate_w = random.randint(max(10, max_w + 5), 16)

    placed_notches = _place_notches(plate_h, plate_w, specs)
    if placed_notches is None:
        return None

    notch_mask = np.zeros((plate_h, plate_w), dtype=bool)
    for spec in placed_notches:
        coords = np.asarray(spec["notch_coords_local"], dtype=int)
        notch_mask[coords[:, 0], coords[:, 1]] = True

    colors = random.sample(COLOR_POOL, piece_count)
    random.shuffle(colors)
    color_pairs = [tuple(colors[i : i + 2]) for i in range(0, piece_count, 2)]
    color_map: dict[int, int] = {}
    for c1, c2 in color_pairs:
        color_map[c1] = c2
        color_map[c2] = c1

    markers_local = _place_markers(plate_h, plate_w, notch_mask, color_pairs)
    if markers_local is None:
        return None

    random.shuffle(placed_notches)
    for spec, color in zip(placed_notches, colors):
        spec["piece_color"] = color
        spec["target_color"] = color_map[color]

    grid_h = plate_h + 2 * (max_h + 3) + random.randint(0, 2)
    grid_w = plate_w + 2 * (max_w + 3) + random.randint(0, 2)
    if grid_h > 30 or grid_w > 30:
        return None

    plate_top = (grid_h - plate_h) // 2 + random.randint(-1, 1)
    plate_left = (grid_w - plate_w) // 2 + random.randint(-1, 1)
    plate_top = min(max(3, plate_top), grid_h - plate_h - 3)
    plate_left = min(max(3, plate_left), grid_w - plate_w - 3)
    plate_box = (plate_top, plate_left, plate_top + plate_h - 1, plate_left + plate_w - 1)

    placed_pieces = None
    for _ in range(PLACEMENT_RETRIES):
        placed_pieces = _place_pieces(grid_h, grid_w, plate_box, placed_notches)
        if placed_pieces is not None:
            break
    if placed_pieces is None:
        return None

    inp, out = _render_case(grid_h, grid_w, plate_top, plate_left, plate_h, plate_w, placed_pieces, markers_local)
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
