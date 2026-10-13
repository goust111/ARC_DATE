from __future__ import annotations

import random

import numpy as np


BG = 8
FRAME = 5
MARKER = 9
BODY_COLORS = (0, 1, 2, 3, 4, 6, 7)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

PAIR_COUNT_CHOICES = (2, 3)
RECT_SIZES = tuple(
    (h, w)
    for h in range(4, 7)
    for w in range(4, 8)
    if h * w >= 16
)
PAIR_ATTEMPTS = 250
GENERATION_ATTEMPTS = 400
OUTER_MARGIN = 1
ZONE_GAP = 2


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _crop_mask(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy(), (int(r0), int(c0))


def _is_connected(mask: np.ndarray) -> bool:
    return int(mask.sum()) > 0 and len(_components(mask)) == 1


def _is_solid_rectangle(mask: np.ndarray) -> bool:
    return bool(mask.all())


def _make_piece(
    shape: np.ndarray,
    *,
    marker: tuple[int, int] | None = None,
    body_color: int | None = None,
    global_coords: tuple[tuple[int, int], ...] | None = None,
) -> dict:
    shape = np.asarray(shape, dtype=bool)
    coords = tuple((int(r), int(c)) for r, c in np.argwhere(shape))
    rows = np.array([r for r, _ in coords], dtype=int)
    cols = np.array([c for _, c in coords], dtype=int)
    return {
        "shape": shape,
        "coords": coords,
        "rows": rows,
        "cols": cols,
        "area": int(len(coords)),
        "marker": marker,
        "body_color": body_color,
        "global_coords": global_coords,
    }


def _piece_from_component(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict:
    r0, c0, r1, c1 = _bbox(coords)
    shape = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    marker: tuple[int, int] | None = None
    body_colors: set[int] = set()

    for r, c in coords:
        shape[r - r0, c - c0] = True
        color = int(grid[r, c])
        if color == MARKER:
            assert marker is None, "each fragment must contain one marker"
            marker = (int(r - r0), int(c - c0))
        else:
            body_colors.add(color)

    assert marker is not None, "fragment must contain a maroon marker"
    assert len(body_colors) == 1, "fragment must have one body color"
    return _make_piece(
        shape,
        marker=marker,
        body_color=int(next(iter(body_colors))),
        global_coords=tuple((int(r), int(c)) for r, c in coords),
    )


def _find_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for coords in _components(grid == FRAME):
        r0, c0, r1, c1 = _bbox(coords)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros(sub.shape, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if not np.all(sub[border] == FRAME):
            continue
        if np.any(sub[~border] == FRAME):
            continue
        assert np.all(sub[1:-1, 1:-1] == BG), "frame interiors must be empty"
        frames.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "inner_shape": (int(r1 - r0 - 1), int(c1 - c0 - 1)),
            }
        )

    frames.sort(key=lambda frame: (frame["bbox"][0], frame["bbox"][1]))
    assert frames, "expected at least one frame"
    sizes = [frame["inner_shape"] for frame in frames]
    assert len(sizes) == len(set(sizes)), "frame sizes must be unique"
    return frames


def _find_fragments(grid: np.ndarray) -> list[dict]:
    mask = (grid != BG) & (grid != FRAME)
    fragments = [_piece_from_component(grid, coords) for coords in _components(mask)]
    assert fragments and len(fragments) % 2 == 0, "expected an even number of fragments"
    return fragments


def _placements(piece: dict, rect_shape: tuple[int, int]) -> list[dict]:
    rect_h, rect_w = rect_shape
    piece_h, piece_w = piece["shape"].shape
    if piece_h > rect_h or piece_w > rect_w:
        return []

    rows = piece["rows"]
    cols = piece["cols"]
    placed: list[dict] = []
    for r0 in range(rect_h - piece_h + 1):
        for c0 in range(rect_w - piece_w + 1):
            mask = np.zeros((rect_h, rect_w), dtype=bool)
            mask[rows + r0, cols + c0] = True
            placed.append({"offset": (int(r0), int(c0)), "mask": mask})
    return placed


def _enumerate_tilings(piece_a: dict, piece_b: dict, rect_shape: tuple[int, int]) -> list[dict]:
    rect_h, rect_w = rect_shape
    if piece_a["area"] + piece_b["area"] != rect_h * rect_w:
        return []

    out: list[dict] = []
    placements_a = _placements(piece_a, rect_shape)
    placements_b = _placements(piece_b, rect_shape)
    for placed_a in placements_a:
        for placed_b in placements_b:
            if np.any(placed_a["mask"] & placed_b["mask"]):
                continue
            out.append(
                {
                    "a_offset": placed_a["offset"],
                    "b_offset": placed_b["offset"],
                }
            )
    return out


def _alignment_mode(piece_a: dict, piece_b: dict, tiling: dict) -> str | None:
    marker_a = piece_a["marker"]
    marker_b = piece_b["marker"]
    assert marker_a is not None and marker_b is not None
    ar = tiling["a_offset"][0] + marker_a[0]
    ac = tiling["a_offset"][1] + marker_a[1]
    br = tiling["b_offset"][0] + marker_b[0]
    bc = tiling["b_offset"][1] + marker_b[1]
    if ar == br:
        return "row"
    if ac == bc:
        return "col"
    return None


def _solve_instance(grid: np.ndarray) -> tuple[list[dict], list[dict], dict[int, dict]]:
    frames = _find_frames(grid)
    fragments = _find_fragments(grid)
    assert len(fragments) == 2 * len(frames), "frame count must match fragment pairs"

    options_by_frame: dict[int, list[dict]] = {}
    for frame_idx, frame in enumerate(frames):
        options: list[dict] = []
        for i in range(len(fragments)):
            for j in range(i + 1, len(fragments)):
                for tiling in _enumerate_tilings(fragments[i], fragments[j], frame["inner_shape"]):
                    mode = _alignment_mode(fragments[i], fragments[j], tiling)
                    if mode is None:
                        continue
                    options.append(
                        {
                            "pieces": (int(i), int(j)),
                            "tiling": tiling,
                            "mode": mode,
                        }
                    )
        assert options, "each frame needs a compatible aligned pair"
        options_by_frame[frame_idx] = options

    frame_order = sorted(options_by_frame, key=lambda idx: len(options_by_frame[idx]))
    used = [False] * len(fragments)
    current: dict[int, dict] = {}
    solutions: list[dict[int, dict]] = []

    def backtrack(pos: int) -> None:
        if len(solutions) > 1:
            return
        if pos == len(frame_order):
            if all(used):
                solutions.append(current.copy())
            return

        frame_idx = frame_order[pos]
        for option in options_by_frame[frame_idx]:
            i, j = option["pieces"]
            if used[i] or used[j]:
                continue
            used[i] = True
            used[j] = True
            current[frame_idx] = option
            backtrack(pos + 1)
            del current[frame_idx]
            used[i] = False
            used[j] = False

    backtrack(0)
    assert len(solutions) == 1, "pairing and orientation must be unique"
    return frames, fragments, solutions[0]


def _paint_piece(out: np.ndarray, piece: dict, top_left: tuple[int, int]) -> None:
    r0, c0 = top_left
    out[piece["rows"] + r0, piece["cols"] + c0] = int(piece["body_color"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair complementary fragments into rectangles, resolve the correct tiling by marker alignment,
    and place each reconstructed rectangle into the matching grey frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames, fragments, assignment = _solve_instance(g)
    out = g.copy()

    for fragment in fragments:
        coords = fragment["global_coords"]
        assert coords is not None
        rr = np.array([r for r, _ in coords], dtype=int)
        cc = np.array([c for _, c in coords], dtype=int)
        out[rr, cc] = BG

    for frame_idx, option in assignment.items():
        frame = frames[frame_idx]
        ir0 = frame["bbox"][0] + 1
        ic0 = frame["bbox"][1] + 1
        piece_a = fragments[option["pieces"][0]]
        piece_b = fragments[option["pieces"][1]]
        _paint_piece(
            out,
            piece_a,
            (ir0 + option["tiling"]["a_offset"][0], ic0 + option["tiling"]["a_offset"][1]),
        )
        _paint_piece(
            out,
            piece_b,
            (ir0 + option["tiling"]["b_offset"][0], ic0 + option["tiling"]["b_offset"][1]),
        )

    return out


def _random_connected_subset(shape: tuple[int, int], target: int) -> np.ndarray | None:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    start = (random.randrange(h), random.randrange(w))
    mask[start] = True
    frontier: set[tuple[int, int]] = set()

    def add_frontier(r: int, c: int) -> None:
        for dr, dc in NEIGHBORS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                frontier.add((int(nr), int(nc)))

    add_frontier(*start)
    while int(mask.sum()) < target:
        if not frontier:
            return None
        r, c = random.choice(tuple(frontier))
        frontier.remove((r, c))
        if mask[r, c]:
            continue
        mask[r, c] = True
        add_frontier(r, c)
    return mask


def _generate_pair(rect_shape: tuple[int, int], desired_mode: str) -> dict | None:
    rect_h, rect_w = rect_shape
    area = rect_h * rect_w
    min_part = max(4, area // 4)
    max_part = area - min_part

    for _ in range(PAIR_ATTEMPTS):
        target = random.randint(min_part, max_part)
        mask_a = _random_connected_subset(rect_shape, target)
        if mask_a is None:
            continue
        mask_b = ~mask_a
        if not _is_connected(mask_b):
            continue

        shape_a, _ = _crop_mask(mask_a)
        shape_b, _ = _crop_mask(mask_b)
        if _is_solid_rectangle(shape_a) or _is_solid_rectangle(shape_b):
            continue

        piece_a = _make_piece(shape_a)
        piece_b = _make_piece(shape_b)
        tilings = _enumerate_tilings(piece_a, piece_b, rect_shape)
        if not tilings:
            continue

        marker_pairs = [(ma, mb) for ma in piece_a["coords"] for mb in piece_b["coords"]]
        random.shuffle(marker_pairs)
        for marker_a, marker_b in marker_pairs:
            marked_a = {**piece_a, "marker": marker_a}
            marked_b = {**piece_b, "marker": marker_b}
            aligned: list[tuple[dict, str]] = []
            for tiling in tilings:
                mode = _alignment_mode(marked_a, marked_b, tiling)
                if mode is not None:
                    aligned.append((tiling, mode))
            if len(aligned) == 1 and aligned[0][1] == desired_mode:
                return {
                    "rect_shape": rect_shape,
                    "mode": desired_mode,
                    "piece_a": marked_a,
                    "piece_b": marked_b,
                    "solution": aligned[0][0],
                }
    return None


def _piece_patch(piece: dict) -> np.ndarray:
    patch = np.full(piece["shape"].shape, BG, dtype=int)
    patch[piece["shape"]] = int(piece["body_color"])
    mr, mc = piece["marker"]
    patch[int(mr), int(mc)] = MARKER
    return patch


def _frame_patch(inner_shape: tuple[int, int]) -> np.ndarray:
    h, w = inner_shape
    patch = np.full((h + 2, w + 2), BG, dtype=int)
    patch[0, :] = FRAME
    patch[-1, :] = FRAME
    patch[:, 0] = FRAME
    patch[:, -1] = FRAME
    return patch


def _place_fragment_bboxes(rows: int, left_width: int, patches: list[np.ndarray]) -> list[tuple[int, int]] | None:
    occ = np.zeros((rows, left_width), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(patches)
    order = sorted(range(len(patches)), key=lambda idx: patches[idx].shape[0] * patches[idx].shape[1], reverse=True)

    for idx in order:
        patch = patches[idx]
        ph, pw = patch.shape
        candidates: list[tuple[int, int]] = []
        for r0 in range(OUTER_MARGIN, rows - ph - OUTER_MARGIN + 1):
            for c0 in range(OUTER_MARGIN, left_width - pw - OUTER_MARGIN + 1):
                rr0 = max(0, r0 - 1)
                cc0 = max(0, c0 - 1)
                rr1 = min(rows, r0 + ph + 1)
                cc1 = min(left_width, c0 + pw + 1)
                if occ[rr0:rr1, cc0:cc1].any():
                    continue
                candidates.append((int(r0), int(c0)))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        placements[idx] = (r0, c0)
        occ[max(0, r0 - 1) : min(rows, r0 + ph + 1), max(0, c0 - 1) : min(left_width, c0 + pw + 1)] = True

    return [placement for placement in placements if placement is not None]


def _build_input_and_output(pair_specs: list[dict]) -> tuple[np.ndarray, np.ndarray] | None:
    ordered_specs = pair_specs.copy()
    random.shuffle(ordered_specs)

    frame_stack_h = sum(spec["rect_shape"][0] + 2 for spec in ordered_specs) + max(0, len(ordered_specs) - 1)
    frame_strip_w = max(spec["rect_shape"][1] + 2 for spec in ordered_specs)
    top_margin = random.randint(1, 2)
    bottom_margin = random.randint(1, 2)
    rows = max(frame_stack_h + top_margin + bottom_margin, random.randint(18, 28))
    left_width = random.randint(12, 18)
    cols = OUTER_MARGIN + left_width + ZONE_GAP + frame_strip_w + OUTER_MARGIN

    all_pieces = [spec["piece_a"] for spec in pair_specs] + [spec["piece_b"] for spec in pair_specs]
    patches = [_piece_patch(piece) for piece in all_pieces]
    placements = _place_fragment_bboxes(rows, left_width, patches)
    if placements is None:
        return None

    grid = np.full((rows, cols), BG, dtype=int)
    expected = np.full((rows, cols), BG, dtype=int)

    current_row = top_margin
    frame_positions: dict[int, tuple[int, int]] = {}
    for spec in ordered_specs:
        frame = _frame_patch(spec["rect_shape"])
        fh, fw = frame.shape
        c0 = cols - OUTER_MARGIN - fw
        grid[current_row : current_row + fh, c0 : c0 + fw] = frame
        expected[current_row : current_row + fh, c0 : c0 + fw] = frame
        frame_positions[id(spec)] = (int(current_row), int(c0))
        current_row += fh + 1

    for piece, patch, (r0, c0) in zip(all_pieces, patches, placements):
        mask = patch != BG
        region = grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
        region[mask] = patch[mask]
        region2 = expected[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
        region2[mask] = patch[mask]

    for piece, (r0, c0) in zip(all_pieces, placements):
        expected[piece["rows"] + r0, piece["cols"] + c0] = BG

    for spec in ordered_specs:
        fr0, fc0 = frame_positions[id(spec)]
        ir0 = fr0 + 1
        ic0 = fc0 + 1
        _paint_piece(
            expected,
            spec["piece_a"],
            (ir0 + spec["solution"]["a_offset"][0], ic0 + spec["solution"]["a_offset"][1]),
        )
        _paint_piece(
            expected,
            spec["piece_b"],
            (ir0 + spec["solution"]["b_offset"][0], ic0 + spec["solution"]["b_offset"][1]),
        )

    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        pair_count = random.choice(PAIR_COUNT_CHOICES)
        rect_shapes = random.sample(RECT_SIZES, pair_count)
        modes = [random.choice(("row", "col")) for _ in range(pair_count)]

        pair_specs: list[dict] = []
        ok = True
        for rect_shape, mode in zip(rect_shapes, modes):
            spec = _generate_pair(rect_shape, mode)
            if spec is None:
                ok = False
                break
            pair_specs.append(spec)
        if not ok:
            continue

        colors = random.sample(BODY_COLORS, 2 * pair_count)
        random.shuffle(colors)
        for idx, spec in enumerate(pair_specs):
            spec["piece_a"]["body_color"] = int(colors[2 * idx])
            spec["piece_b"]["body_color"] = int(colors[2 * idx + 1])

        built = _build_input_and_output(pair_specs)
        if built is None:
            continue
        grid, expected = built

        try:
            actual = main(grid)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
