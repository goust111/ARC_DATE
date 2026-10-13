from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
QUADRANTS = ("UL", "UR", "LL", "LR")
CW_ROTATIONS = {"UL": 0, "UR": 1, "LR": 2, "LL": 3}
GEN_RETRIES = 400


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
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _rotate_cw(patch: np.ndarray, quarter_turns: int) -> np.ndarray:
    turns = quarter_turns % 4
    if turns == 0:
        return patch.copy()
    return np.rot90(patch, -turns)


def _rectangles_overlap(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0
    )


def _center_box(center: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    r, c = center
    h, w = shape
    return (r - h // 2, c - w // 2, r + h // 2, c + w // 2)


def _find_canvas(grid: np.ndarray) -> tuple[int, int, int, int]:
    grey = np.argwhere(grid == GREY)
    assert grey.size > 0, "expected a grey framed canvas"
    r0, c0 = map(int, grey.min(axis=0))
    r1, c1 = map(int, grey.max(axis=0))
    assert r1 - r0 + 1 >= 10 and c1 - c0 + 1 >= 10, "canvas must be at least 10x10"
    frame = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(frame[0, :] == GREY)
    assert np.all(frame[-1, :] == GREY)
    assert np.all(frame[:, 0] == GREY)
    assert np.all(frame[:, -1] == GREY)
    assert not np.any(frame[1:-1, 1:-1] == GREY), "grey is reserved for the canvas border"
    expected = 2 * frame.shape[0] + 2 * frame.shape[1] - 4
    assert int(np.sum(grid == GREY)) == expected, "grey pixels should form one rectangular frame"
    return r0, c0, r1, c1


def _marker_quadrant(
    marker_row: int, marker_col: int, top: int, left: int, bottom: int, right: int
) -> str:
    center_row = (top + bottom) / 2.0
    center_col = (left + right) / 2.0
    above = marker_row < center_row
    below = marker_row > center_row
    before = marker_col < center_col
    after = marker_col > center_col
    assert above or below, "marker must not lie on the horizontal center axis"
    assert before or after, "marker must not lie on the vertical center axis"
    if above and before:
        return "UL"
    if above and after:
        return "UR"
    if below and before:
        return "LL"
    return "LR"


def _extract_markers(
    grid: np.ndarray, canvas: tuple[int, int, int, int]
) -> list[tuple[int, int, int, str]]:
    r0, c0, r1, c1 = canvas
    markers: list[tuple[int, int, int, str]] = []
    colors: set[int] = set()
    for rr in range(r0 + 1, r1):
        for cc in range(c0 + 1, c1):
            value = int(grid[rr, cc])
            if value == BLACK:
                continue
            assert value != GREY
            assert value not in colors, "marker colors must be unique"
            colors.add(value)
            quad = _marker_quadrant(rr, cc, r0 + 1, c0 + 1, r1 - 1, c1 - 1)
            markers.append((rr, cc, value, quad))
    markers.sort(key=lambda item: (item[0], item[1]))
    assert markers, "canvas must contain markers"
    return markers


def _extract_stamps(
    grid: np.ndarray, canvas: tuple[int, int, int, int]
) -> dict[int, np.ndarray]:
    r0, c0, r1, c1 = canvas
    outside = grid != BLACK
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    stamps: dict[int, np.ndarray] = {}

    for comp in _components(outside):
        br0, bc0, br1, bc1 = _bbox(comp)
        patch = grid[br0 : br1 + 1, bc0 : bc1 + 1].copy()
        assert patch.shape[0] % 2 == 1 and patch.shape[1] % 2 == 1, "stamp sizes must have a center pixel"
        ring = np.concatenate((patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]))
        border_colors = {int(v) for v in ring if int(v) != BLACK}
        assert len(border_colors) == 1, "each stamp needs a single border color"
        border_color = border_colors.pop()
        assert border_color != GREY
        assert np.all(patch[0, :] == border_color)
        assert np.all(patch[-1, :] == border_color)
        assert np.all(patch[:, 0] == border_color)
        assert np.all(patch[:, -1] == border_color)
        assert patch.shape[0] >= 3 and patch.shape[1] >= 3
        inner = patch[1:-1, 1:-1]
        assert inner.size > 0 and np.any(inner != BLACK)
        assert np.any(inner != border_color), "stamp interior must differ from its border color"
        assert border_color not in stamps, "border colors must identify stamps uniquely"
        stamps[border_color] = patch

    assert stamps, "expected at least one stamp outside the canvas"
    return stamps


def _stamp_onto_canvas(
    canvas_shape: tuple[int, int],
    markers: list[tuple[int, int, int, str]],
    stamps: dict[int, np.ndarray],
    canvas_top: int,
    canvas_left: int,
) -> np.ndarray:
    out = np.full(canvas_shape, BLACK, dtype=int)
    out[0, :] = GREY
    out[-1, :] = GREY
    out[:, 0] = GREY
    out[:, -1] = GREY

    for abs_r, abs_c, color, quadrant in markers:
        patch = _rotate_cw(stamps[color], CW_ROTATIONS[quadrant])
        ph, pw = patch.shape
        top = abs_r - canvas_top - ph // 2
        left = abs_c - canvas_left - pw // 2
        for pr in range(ph):
            for pc in range(pw):
                value = int(patch[pr, pc])
                if value == BLACK:
                    continue
                rr = top + pr
                cc = left + pc
                if 1 <= rr < canvas_shape[0] - 1 and 1 <= cc < canvas_shape[1] - 1:
                    out[rr, cc] = value
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate color-keyed outside stamps and place them onto the framed canvas at matching markers.

    Input structure:
    - The background is black.
    - One large grey one-pixel rectangular frame encloses a black canvas interior with several singleton colored markers.
    - Outside the frame sit several compact stamp objects; each stamp has a one-pixel border of one color surrounding a multicolor interior.

    Rule:
    - Match each canvas marker to the outside stamp whose border color is the same as the marker color.
    - Use the marker's quadrant relative to the canvas center to choose the rotation: upper-left 0 degrees, upper-right 90 degrees clockwise, lower-right 180 degrees, lower-left 270 degrees clockwise.
    - Rotate the matched stamp, center it on the marker cell, and apply stamps in marker order from top to bottom then left to right so later ones overwrite earlier overlaps.
    - Return only the grey-framed canvas region.

    Key insight:
    - Color determines which stamp belongs to each marker, while the marker's position inside the canvas determines how much that stamp must be rotated before stamping.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    canvas = _find_canvas(g)
    markers = _extract_markers(g, canvas)
    stamps = _extract_stamps(g, canvas)
    marker_colors = {color for _r, _c, color, _q in markers}
    assert marker_colors == set(stamps), "marker colors and stamp border colors must match exactly"

    r0, c0, r1, c1 = canvas
    return _stamp_onto_canvas((r1 - r0 + 1, c1 - c0 + 1), markers, stamps, r0, c0)


def _sample_canvas_geometry(rng: np.random.Generator) -> tuple[int, int, int, int, int, int]:
    for _ in range(200):
        grid_h = int(rng.integers(24, 31))
        grid_w = int(rng.integers(24, 31))
        canvas_h = int(rng.integers(12, min(17, grid_h - 9)))
        canvas_w = int(rng.integers(12, min(17, grid_w - 9)))
        if grid_h - canvas_h < 10 or grid_w - canvas_w < 10:
            continue
        top = int(rng.integers(5, grid_h - canvas_h - 4))
        left = int(rng.integers(5, grid_w - canvas_w - 4))
        if top < 5 or left < 5 or grid_h - (top + canvas_h) < 5 or grid_w - (left + canvas_w) < 5:
            continue
        return grid_h, grid_w, canvas_h, canvas_w, top, left
    raise RuntimeError("failed to sample canvas geometry")


def _build_stamp(
    border_color: int,
    shape: tuple[int, int],
    fill_colors: tuple[int, ...],
    rng: np.random.Generator,
) -> np.ndarray:
    h, w = shape
    patch = np.full((h, w), border_color, dtype=int)
    inner = patch[1:-1, 1:-1]
    base = int(fill_colors[0])
    accent = int(fill_colors[1 if len(fill_colors) > 1 else 0])
    tail = int(fill_colors[2 if len(fill_colors) > 2 else 0])

    inner[:, :] = base
    inner[0, :] = accent
    inner[min(1, inner.shape[0] - 1), 0] = accent
    inner[-1, -1] = tail
    if inner.shape[1] > 1 and rng.random() < 0.5:
        inner[1, -1] = tail
    if inner.shape[0] > 2 and inner.shape[1] > 1 and rng.random() < 0.5:
        inner[-2, 0] = accent

    assert not np.array_equal(patch, _rotate_cw(patch, 1))
    assert not np.array_equal(patch, _rotate_cw(patch, 2))
    return patch


def _candidate_centers(
    inner_h: int, inner_w: int, shape: tuple[int, int], quadrant: str
) -> list[tuple[int, int]]:
    h, w = shape
    hh, hw = h // 2, w // 2
    row_lo, row_hi = hh, inner_h - hh
    col_lo, col_hi = hw, inner_w - hw
    centers: list[tuple[int, int]] = []
    for r in range(row_lo, row_hi):
        if quadrant in ("UL", "UR") and not (r < inner_h // 2):
            continue
        if quadrant in ("LL", "LR") and not (r >= (inner_h + 1) // 2):
            continue
        for c in range(col_lo, col_hi):
            if quadrant in ("UL", "LL") and not (c < inner_w // 2):
                continue
            if quadrant in ("UR", "LR") and not (c >= (inner_w + 1) // 2):
                continue
            centers.append((r, c))
    return centers


def _place_marker_centers(
    specs: list[dict], inner_h: int, inner_w: int, overlap_required: bool, rng: np.random.Generator
) -> dict[int, tuple[int, int]] | None:
    candidates = {
        spec["id"]: _candidate_centers(inner_h, inner_w, spec["placed_shape"], spec["quadrant"])
        for spec in specs
    }
    if any(not pts for pts in candidates.values()):
        return None

    placements: dict[int, tuple[int, int]] = {}
    boxes: dict[int, tuple[int, int, int, int]] = {}

    if overlap_required:
        pair_options = [("UL", "UR"), ("UL", "LL"), ("UR", "LR"), ("LL", "LR")]
        base_lookup = {spec["quadrant"]: spec["id"] for spec in specs[:4]}
        rng.shuffle(pair_options)
        paired = False
        for qa, qb in pair_options:
            ida, idb = base_lookup[qa], base_lookup[qb]
            pts_a = candidates[ida][:]
            pts_b = candidates[idb][:]
            rng.shuffle(pts_a)
            rng.shuffle(pts_b)
            for ca in pts_a[:40]:
                box_a = _center_box(ca, specs[ida]["placed_shape"])
                for cb in pts_b[:40]:
                    if ca == cb:
                        continue
                    box_b = _center_box(cb, specs[idb]["placed_shape"])
                    if _rectangles_overlap(box_a, box_b, pad=0):
                        placements[ida] = ca
                        placements[idb] = cb
                        boxes[ida] = box_a
                        boxes[idb] = box_b
                        paired = True
                        break
                if paired:
                    break
            if paired:
                break
        if not paired:
            return None

    remaining = [spec["id"] for spec in specs if spec["id"] not in placements]
    rng.shuffle(remaining)
    for sid in remaining:
        pts = candidates[sid][:]
        rng.shuffle(pts)
        placed = False
        for center in pts:
            if center in placements.values():
                continue
            box = _center_box(center, specs[sid]["placed_shape"])
            if any(_rectangles_overlap(box, other, pad=1) for other in boxes.values()):
                continue
            placements[sid] = center
            boxes[sid] = box
            placed = True
            break
        if not placed:
            return None

    if overlap_required:
        overlap_count = 0
        ids = list(boxes)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                overlap_count += int(_rectangles_overlap(boxes[ids[i]], boxes[ids[j]], pad=0))
        if overlap_count == 0:
            return None

    return placements


def _place_outside_stamps(
    specs: list[dict],
    grid_h: int,
    grid_w: int,
    canvas_box: tuple[int, int, int, int],
    rng: np.random.Generator,
) -> dict[int, tuple[int, int]] | None:
    placements: dict[int, tuple[int, int]] = {}
    boxes: list[tuple[int, int, int, int]] = []
    order = [spec["id"] for spec in specs]
    rng.shuffle(order)

    for sid in order:
        patch = specs[sid]["original_patch"]
        h, w = patch.shape
        placed = False
        for _ in range(300):
            r0 = int(rng.integers(0, grid_h - h + 1))
            c0 = int(rng.integers(0, grid_w - w + 1))
            box = (r0, c0, r0 + h - 1, c0 + w - 1)
            if _rectangles_overlap(box, canvas_box, pad=1):
                continue
            if any(_rectangles_overlap(box, other, pad=1) for other in boxes):
                continue
            placements[sid] = (r0, c0)
            boxes.append(box)
            placed = True
            break
        if not placed:
            return None
    return placements


def _paint_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    for r in range(h):
        for c in range(w):
            value = int(patch[r, c])
            if value != BLACK:
                grid[top + r, left + c] = value


def _render_expected(scene: dict) -> np.ndarray:
    canvas_h = scene["canvas_h"]
    canvas_w = scene["canvas_w"]
    out = np.full((canvas_h, canvas_w), BLACK, dtype=int)
    out[0, :] = GREY
    out[-1, :] = GREY
    out[:, 0] = GREY
    out[:, -1] = GREY

    markers = sorted(scene["markers"], key=lambda item: (item["abs_row"], item["abs_col"]))
    for marker in markers:
        spec = scene["by_color"][marker["color"]]
        patch = _rotate_cw(spec["original_patch"], CW_ROTATIONS[marker["quadrant"]])
        h, w = patch.shape
        top = marker["rel_row"] + 1 - h // 2
        left = marker["rel_col"] + 1 - w // 2
        for r in range(h):
            for c in range(w):
                value = int(patch[r, c])
                if value == BLACK:
                    continue
                rr = top + r
                cc = left + c
                if 1 <= rr < canvas_h - 1 and 1 <= cc < canvas_w - 1:
                    out[rr, cc] = value
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    grid_h, grid_w, canvas_h, canvas_w, canvas_top, canvas_left = _sample_canvas_geometry(rng)
    if canvas_top + canvas_h > grid_h - 5 or canvas_left + canvas_w > grid_w - 5:
        return None

    inner_h = canvas_h - 2
    inner_w = canvas_w - 2
    overlap_required = bool(rng.integers(0, 2))
    extra_count = int(rng.integers(0, 3))
    stamp_count = 4 + extra_count

    border_colors = [int(v) for v in rng.choice(np.asarray(PALETTE), size=stamp_count, replace=False)]
    remaining_colors = [c for c in PALETTE if c not in border_colors]

    quadrants = list(QUADRANTS)
    extra_quadrants = [str(q) for q in rng.choice(np.asarray(QUADRANTS), size=extra_count, replace=True)]
    marker_quadrants = quadrants + extra_quadrants
    rng.shuffle(marker_quadrants[4:])

    specs: list[dict] = []
    by_color: dict[int, dict] = {}
    shape_choices = ((5, 3), (5, 5))
    for sid, (color, quadrant) in enumerate(zip(border_colors, marker_quadrants)):
        shape = shape_choices[int(rng.integers(0, len(shape_choices)))]
        fill_count = 2 if len(remaining_colors) < 3 or rng.random() < 0.55 else 3
        fills = tuple(int(v) for v in rng.choice(np.asarray(remaining_colors), size=fill_count, replace=False))
        patch = _build_stamp(color, shape, fills, rng)
        placed_patch = _rotate_cw(patch, CW_ROTATIONS[quadrant])
        spec = {
            "id": sid,
            "color": color,
            "quadrant": quadrant,
            "original_patch": patch,
            "placed_shape": tuple(map(int, placed_patch.shape)),
        }
        specs.append(spec)
        by_color[color] = spec

    centers = _place_marker_centers(specs, inner_h, inner_w, overlap_required, rng)
    if centers is None:
        return None

    canvas_box = (canvas_top, canvas_left, canvas_top + canvas_h - 1, canvas_left + canvas_w - 1)
    outside = _place_outside_stamps(specs, grid_h, grid_w, canvas_box, rng)
    if outside is None:
        return None

    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    grid[canvas_top, canvas_left : canvas_left + canvas_w] = GREY
    grid[canvas_top + canvas_h - 1, canvas_left : canvas_left + canvas_w] = GREY
    grid[canvas_top : canvas_top + canvas_h, canvas_left] = GREY
    grid[canvas_top : canvas_top + canvas_h, canvas_left + canvas_w - 1] = GREY

    markers: list[dict] = []
    for spec in specs:
        rel_r, rel_c = centers[spec["id"]]
        abs_r = canvas_top + 1 + rel_r
        abs_c = canvas_left + 1 + rel_c
        grid[abs_r, abs_c] = spec["color"]
        markers.append(
            {
                "abs_row": abs_r,
                "abs_col": abs_c,
                "rel_row": rel_r,
                "rel_col": rel_c,
                "color": spec["color"],
                "quadrant": spec["quadrant"],
            }
        )

    for spec in specs:
        top, left = outside[spec["id"]]
        _paint_patch(grid, spec["original_patch"], top, left)

    scene = {
        "canvas_h": canvas_h,
        "canvas_w": canvas_w,
        "markers": markers,
        "by_color": by_color,
    }
    expected = _render_expected(scene)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
