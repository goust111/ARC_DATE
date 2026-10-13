from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]


def _bbox(coords: list[Coord] | tuple[Coord, ...]) -> tuple[int, int, int, int]:
    rows = [int(r) for r, _ in coords]
    cols = [int(c) for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _unique_mode(values: np.ndarray) -> int:
    counts = Counter(map(int, np.asarray(values).reshape(-1)))
    assert counts, "expected at least one value"
    max_count = max(counts.values())
    winners = [color for color, count in counts.items() if count == max_count]
    assert len(winners) == 1, f"expected a unique dominant color, found {winners}"
    return int(winners[0])


def _background_color(g: np.ndarray) -> int:
    boundary = np.concatenate((g[0, :], g[-1, :], g[:, 0], g[:, -1]))
    try:
        return _unique_mode(boundary)
    except AssertionError:
        return _unique_mode(g)


def _neighbors4(shape: tuple[int, int], r: int, c: int) -> list[Coord]:
    h, w = map(int, shape)
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = int(r + dr)
        cc = int(c + dc)
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _find_components(g: np.ndarray, bg: int) -> list[dict]:
    h, w = map(int, g.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == int(bg) or seen[r, c]:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(g.shape, rr, cc):
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            r0, c0, r1, c1 = _bbox(coords)
            comps.append(
                {
                    "color": color,
                    "coords": tuple(sorted(coords)),
                    "bbox": (r0, c0, r1, c1),
                    "area": len(coords),
                }
            )

    return comps


def _is_frame_component(comp: dict) -> bool:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False

    border = {
        (r0, c)
        for c in range(c0, c1 + 1)
    }
    border |= {
        (r1, c)
        for c in range(c0, c1 + 1)
    }
    border |= {
        (r, c0)
        for r in range(r0, r1 + 1)
    }
    border |= {
        (r, c1)
        for r in range(r0, r1 + 1)
    }
    return int(comp["area"]) == len(border) and set(comp["coords"]) == border


def _shape_from_coords(coords: tuple[Coord, ...], shape_id: int) -> dict:
    r0, c0, r1, c1 = _bbox(list(coords))
    rel = tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))
    return {
        "id": int(shape_id),
        "coords": rel,
        "coord_set": frozenset(rel),
        "h": int(r1 - r0 + 1),
        "w": int(c1 - c0 + 1),
        "area": len(rel),
        "bbox": (r0, c0, r1, c1),
        "abs_coords": coords,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each hollow frame with the uniquely matching outside shape selected by marker positions.

    The frame-internal marker cells specify which outside candidate shape must be moved into
    the frame: after top-left aligning the candidate with the frame interior, every marker
    relative position must land on an occupied shape cell. The matched shape is removed from
    outside, placed in the frame, and recolored by the marker colors while frames, markers,
    and background stay unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 5, "expected a 2D grid with room for frames"
    bg = _background_color(g)

    components = _find_components(g, bg)
    frames = [comp for comp in components if _is_frame_component(comp)]
    assert frames, "expected at least one rectangular frame"

    frame_records = []
    for frame in sorted(frames, key=lambda comp: comp["bbox"]):
        r0, c0, r1, c1 = map(int, frame["bbox"])
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        coords = [(int(r), int(c)) for r, c in zip(*np.where(interior != int(bg)))]
        assert len(coords) in (1, 2), "each frame interior must contain one or two marker pixels"

        markers = []
        for rr, cc in sorted(coords):
            color = int(interior[rr, cc])
            assert color != int(frame["color"]), "marker color must differ from frame color"
            markers.append({"rel": (rr, cc), "color": color})

        frame_records.append(
            {
                "bbox": (r0, c0, r1, c1),
                "inner_top_left": (r0 + 1, c0 + 1),
                "inner_h": int(r1 - r0 - 1),
                "inner_w": int(c1 - c0 - 1),
                "markers": markers,
                "marker_set": frozenset(marker["rel"] for marker in markers),
            }
        )

    candidate_shapes = []
    for comp in components:
        if _is_frame_component(comp):
            continue

        inside = [
            idx
            for idx, frame in enumerate(frame_records)
            if all(
                frame["bbox"][0] < r < frame["bbox"][2] and frame["bbox"][1] < c < frame["bbox"][3]
                for r, c in comp["coords"]
            )
        ]
        if inside:
            assert len(inside) == 1, "marker component must belong to exactly one frame"
            continue

        assert all(
            not (
                frame["bbox"][0] <= r <= frame["bbox"][2]
                and frame["bbox"][1] <= c <= frame["bbox"][3]
            )
            for frame in frame_records
            for r, c in comp["coords"]
        ), "non-marker component must lie outside every frame"
        candidate_shapes.append(_shape_from_coords(comp["coords"], len(candidate_shapes)))

    assert len(candidate_shapes) >= len(frame_records), "expected at least as many outside shapes as frames"

    chosen_shapes: list[dict] = []
    used_ids: set[int] = set()
    for frame in frame_records:
        matches = [
            shape
            for shape in candidate_shapes
            if shape["h"] <= frame["inner_h"]
            and shape["w"] <= frame["inner_w"]
            and frame["marker_set"] <= shape["coord_set"]
        ]
        assert len(matches) == 1, f"expected a unique matching shape, found {len(matches)}"
        match = matches[0]
        assert match["id"] not in used_ids, "the same outside shape cannot fill two frames"
        used_ids.add(int(match["id"]))
        chosen_shapes.append(match)

    out = g.copy()
    for shape in chosen_shapes:
        for r, c in shape["abs_coords"]:
            out[r, c] = int(bg)

    for frame, shape in zip(frame_records, chosen_shapes):
        marker_colors = {marker["rel"]: int(marker["color"]) for marker in frame["markers"]}
        ordered = sorted(frame["markers"], key=lambda marker: (marker["rel"][0], marker["rel"][1]))
        unique_colors = {int(marker["color"]) for marker in ordered}
        if len(ordered) == 1 or len(unique_colors) == 1:
            top_color = bottom_color = int(ordered[0]["color"])
        else:
            top_color = int(ordered[0]["color"])
            bottom_color = int(ordered[1]["color"])
            assert 2 * int(ordered[0]["rel"][0]) < int(shape["h"]), "first marker must lie in the top half"
            assert 2 * int(ordered[1]["rel"][0]) >= int(shape["h"]), "second marker must lie in the bottom half"

        base_r, base_c = map(int, frame["inner_top_left"])
        for dr, dc in shape["coords"]:
            rr = base_r + int(dr)
            cc = base_c + int(dc)
            if (dr, dc) in marker_colors:
                continue
            out[rr, cc] = top_color if 2 * int(dr) < int(shape["h"]) else bottom_color

    return out


# --------------- generator ---------------

COLORS = tuple(range(10))
GRID_H_RANGE = (28, 34)
GRID_W_RANGE = (28, 34)
FRAME_COUNT_RANGE = (2, 4)
EXTRA_SHAPE_RANGE = (1, 3)
SHAPE_DIM_RANGE = (3, 6)
SHAPE_FILL_MIN = 0.42
SHAPE_FILL_MAX = 0.78
FRAME_EXTRA_RANGE = (0, 2)
PLACEMENT_MARGIN = 1
PLACEMENT_PADDING = 1
SHAPE_ATTEMPTS = 80
PLACEMENT_ATTEMPTS = 200
GENERATION_ATTEMPTS = 400


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    r0 = int(rows.min())
    r1 = int(rows.max())
    c0 = int(cols.min())
    c1 = int(cols.max())
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_has_hole(mask: np.ndarray) -> bool:
    empty = ~mask
    seen = np.zeros(mask.shape, dtype=bool)
    q = deque()

    h, w = map(int, mask.shape)
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            if empty[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(mask.shape, r, c):
            if empty[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))

    return bool(np.any(empty & ~seen))


def _shape_from_mask(mask: np.ndarray) -> dict:
    coords = tuple(sorted((int(r), int(c)) for r, c in zip(*np.where(mask))))
    return {
        "coords": coords,
        "coord_set": frozenset(coords),
        "h": int(mask.shape[0]),
        "w": int(mask.shape[1]),
        "area": len(coords),
        "mask": mask,
        "signature": (int(mask.shape[0]), int(mask.shape[1]), coords),
    }


def _generate_shape_mask() -> np.ndarray | None:
    for _ in range(SHAPE_ATTEMPTS):
        raw_h = random.randint(*SHAPE_DIM_RANGE)
        raw_w = random.randint(*SHAPE_DIM_RANGE)
        total = raw_h * raw_w
        lo = max(4, int(np.ceil(total * SHAPE_FILL_MIN)))
        hi = min(total - 1, int(np.floor(total * SHAPE_FILL_MAX)))
        if lo > hi:
            continue
        target = random.randint(lo, hi)

        mask = np.zeros((raw_h, raw_w), dtype=bool)
        start = (random.randrange(raw_h), random.randrange(raw_w))
        mask[start] = True
        frontier = set(_neighbors4(mask.shape, *start))

        while int(mask.sum()) < target and frontier:
            cell = random.choice(tuple(frontier))
            frontier.discard(cell)
            mask[cell] = True
            for nr, nc in _neighbors4(mask.shape, *cell):
                if not mask[nr, nc]:
                    frontier.add((nr, nc))

        if int(mask.sum()) != target:
            continue

        mask = _trim_mask(mask)
        if min(mask.shape) < 2:
            continue
        if bool(np.all(mask)):
            continue
        if _mask_has_hole(mask):
            continue
        return mask

    return None


def _sample_shape(existing: set[tuple], frame_sizes: list[tuple[int, int]] | None = None) -> dict | None:
    for _ in range(SHAPE_ATTEMPTS):
        mask = _generate_shape_mask()
        if mask is None:
            continue
        shape = _shape_from_mask(mask)
        if shape["signature"] in existing:
            continue
        if frame_sizes is not None and not any(
            shape["h"] <= fh and shape["w"] <= fw for fh, fw in frame_sizes
        ):
            continue
        return shape
    return None


def _matching_shape_ids(
    marker_cells: tuple[Coord, ...], frame_size: tuple[int, int], shapes: list[dict]
) -> set[int]:
    fh, fw = map(int, frame_size)
    marker_set = frozenset(marker_cells)
    return {
        int(shape["id"])
        for shape in shapes
        if shape["h"] <= fh and shape["w"] <= fw and marker_set <= shape["coord_set"]
    }


def _valid_single_markers(shape: dict, frame_size: tuple[int, int], shapes: list[dict]) -> list[Coord]:
    coords = list(shape["coords"])
    random.shuffle(coords)
    return [
        coord
        for coord in coords
        if _matching_shape_ids((coord,), frame_size, shapes) == {int(shape["id"])}
    ]


def _valid_double_markers(
    shape: dict, frame_size: tuple[int, int], shapes: list[dict]
) -> list[tuple[Coord, Coord]]:
    top_cells = [coord for coord in shape["coords"] if 2 * int(coord[0]) < int(shape["h"])]
    bottom_cells = [coord for coord in shape["coords"] if 2 * int(coord[0]) >= int(shape["h"])]
    pairs = [(top, bottom) for top in top_cells for bottom in bottom_cells]
    random.shuffle(pairs)
    return [
        pair
        for pair in pairs
        if _matching_shape_ids(pair, frame_size, shapes) == {int(shape["id"])}
    ]


def _place_patch(occupied: np.ndarray, h: int, w: int) -> tuple[int, int] | None:
    grid_h, grid_w = map(int, occupied.shape)
    min_r = PLACEMENT_MARGIN
    max_r = grid_h - h - PLACEMENT_MARGIN
    min_c = PLACEMENT_MARGIN
    max_c = grid_w - w - PLACEMENT_MARGIN
    if min_r > max_r or min_c > max_c:
        return None

    for _ in range(PLACEMENT_ATTEMPTS):
        r = random.randint(min_r, max_r)
        c = random.randint(min_c, max_c)
        rr0 = max(0, r - PLACEMENT_PADDING)
        rr1 = min(grid_h, r + h + PLACEMENT_PADDING)
        cc0 = max(0, c - PLACEMENT_PADDING)
        cc1 = min(grid_w, c + w + PLACEMENT_PADDING)
        if np.any(occupied[rr0:rr1, cc0:cc1]):
            continue
        occupied[rr0:rr1, cc0:cc1] = True
        return int(r), int(c)

    return None


def _render_expected(inp: np.ndarray, bg: int, frames: list[dict], placed_shapes: dict[int, dict]) -> np.ndarray:
    out = inp.copy()
    for frame in frames:
        shape = placed_shapes[int(frame["shape_id"])]
        for r, c in shape["abs_coords"]:
            out[r, c] = int(bg)

    for frame in frames:
        shape = placed_shapes[int(frame["shape_id"])]
        marker_lookup = {marker["rel"]: int(marker["color"]) for marker in frame["markers"]}
        if len(frame["markers"]) == 1:
            top_color = bottom_color = int(frame["markers"][0]["color"])
        else:
            top_color = int(frame["markers"][0]["color"])
            bottom_color = int(frame["markers"][1]["color"])

        base_r, base_c = map(int, frame["inner_top_left"])
        for dr, dc in shape["coords"]:
            rr = base_r + int(dr)
            cc = base_c + int(dc)
            if (dr, dc) in marker_lookup:
                continue
            out[rr, cc] = top_color if 2 * int(dr) < int(shape["h"]) else bottom_color

    return out


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    bg = random.choice(COLORS)
    non_bg = [color for color in COLORS if color != bg]

    n_frames = random.randint(*FRAME_COUNT_RANGE)
    n_extra = random.randint(*EXTRA_SHAPE_RANGE)
    frame_colors = random.sample(non_bg, n_frames)

    existing_signatures: set[tuple] = set()
    assigned_shapes = []
    for _ in range(n_frames):
        shape = _sample_shape(existing_signatures)
        if shape is None:
            return None
        existing_signatures.add(shape["signature"])
        assigned_shapes.append(shape)

    frame_sizes = []
    for shape in assigned_shapes:
        fh = int(shape["h"]) + random.randint(*FRAME_EXTRA_RANGE)
        fw = int(shape["w"]) + random.randint(*FRAME_EXTRA_RANGE)
        frame_sizes.append((fh, fw))

    distractors = []
    for _ in range(n_extra):
        shape = _sample_shape(existing_signatures, frame_sizes=frame_sizes)
        if shape is None:
            return None
        existing_signatures.add(shape["signature"])
        distractors.append(shape)

    all_shapes = assigned_shapes + distractors
    for shape_id, shape in enumerate(all_shapes):
        shape["id"] = int(shape_id)

    frame_records = []
    for idx, shape in enumerate(assigned_shapes):
        frame_h, frame_w = frame_sizes[idx]
        singles = _valid_single_markers(shape, (frame_h, frame_w), all_shapes)
        doubles = _valid_double_markers(shape, (frame_h, frame_w), all_shapes)
        branch_options = []
        if singles:
            branch_options.append("single")
        if doubles:
            branch_options.append("double")
        if not branch_options:
            return None

        branch = random.choice(branch_options)
        frame_color = int(frame_colors[idx])
        marker_palette = [color for color in non_bg if color != frame_color]
        if branch == "single":
            marker = random.choice(singles)
            markers = [{"rel": marker, "color": int(random.choice(marker_palette))}]
        else:
            top_rel, bottom_rel = random.choice(doubles)
            top_color, bottom_color = random.sample(marker_palette, 2)
            markers = [
                {"rel": top_rel, "color": int(top_color)},
                {"rel": bottom_rel, "color": int(bottom_color)},
            ]

        frame_records.append(
            {
                "shape_id": int(shape["id"]),
                "frame_color": frame_color,
                "inner_h": int(frame_h),
                "inner_w": int(frame_w),
                "markers": markers,
            }
        )

    if any(
        not any(shape["h"] <= frame["inner_h"] and shape["w"] <= frame["inner_w"] for frame in frame_records)
        for shape in all_shapes
    ):
        return None

    grid_h = random.randint(*GRID_H_RANGE)
    grid_w = random.randint(*GRID_W_RANGE)
    inp = np.full((grid_h, grid_w), int(bg), dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    frame_order = list(range(len(frame_records)))
    random.shuffle(frame_order)
    for idx in frame_order:
        frame = frame_records[idx]
        outer_h = int(frame["inner_h"]) + 2
        outer_w = int(frame["inner_w"]) + 2
        pos = _place_patch(occupied, outer_h, outer_w)
        if pos is None:
            return None
        r0, c0 = pos
        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1
        color = int(frame["frame_color"])
        inp[r0, c0 : c1 + 1] = color
        inp[r1, c0 : c1 + 1] = color
        inp[r0 : r1 + 1, c0] = color
        inp[r0 : r1 + 1, c1] = color
        frame["bbox"] = (r0, c0, r1, c1)
        frame["inner_top_left"] = (r0 + 1, c0 + 1)
        for marker in frame["markers"]:
            mr = r0 + 1 + int(marker["rel"][0])
            mc = c0 + 1 + int(marker["rel"][1])
            inp[mr, mc] = int(marker["color"])

    placed_shapes: dict[int, dict] = {}
    shape_order = list(range(len(all_shapes)))
    random.shuffle(shape_order)
    for shape_id in shape_order:
        shape = all_shapes[shape_id]
        pos = _place_patch(occupied, int(shape["h"]), int(shape["w"]))
        if pos is None:
            return None
        r0, c0 = pos
        color = int(random.choice(non_bg))
        abs_coords = tuple((r0 + int(dr), c0 + int(dc)) for dr, dc in shape["coords"])
        for rr, cc in abs_coords:
            inp[rr, cc] = color
        placed_shapes[int(shape_id)] = {
            "id": int(shape_id),
            "coords": shape["coords"],
            "coord_set": shape["coord_set"],
            "h": int(shape["h"]),
            "w": int(shape["w"]),
            "abs_coords": abs_coords,
        }

    counts = Counter(map(int, inp.reshape(-1)))
    if counts[int(bg)] != max(counts.values()):
        return None
    if list(counts.values()).count(counts[int(bg)]) != 1:
        return None

    expected = _render_expected(inp, bg, frame_records, placed_shapes)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    """Generate a random valid input grid for the frame-marker shape-matching task."""
    for _ in range(GENERATION_ATTEMPTS):
        pair = _generate_case()
        if pair is not None:
            return pair[0]
    raise AssertionError("failed to generate a valid puzzle instance")
