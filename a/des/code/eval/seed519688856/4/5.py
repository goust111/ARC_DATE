import numpy as np


BLACK = 0
GREY = 5
FILL_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
RNG = np.random.default_rng()
MAX_GENERATION_RETRIES = 400
NEXT_OVERLAP_BRANCH = False


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return _bbox_from_coords(coords)


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _bit_count(x: int) -> int:
    return int(x.bit_count())


def _find_canvas_and_stamps(
    grid: np.ndarray,
) -> tuple[np.ndarray, int, list[dict[str, object]]]:
    comps = []
    for coords in _connected_components(grid != BLACK):
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        comps.append({"coords": coords, "bbox": (r0, c0, r1, c1), "patch": patch})

    canvas_candidates = []
    for comp in comps:
        patch = comp["patch"]
        colors = sorted(set(int(x) for x in patch[patch != BLACK].tolist()))
        if (
            len(colors) == 1
            and np.all(patch[0, :] != BLACK)
            and np.all(patch[-1, :] != BLACK)
            and np.all(patch[:, 0] != BLACK)
            and np.all(patch[:, -1] != BLACK)
        ):
            canvas_candidates.append(comp)

    assert len(canvas_candidates) == 1, "expected exactly one canvas"
    canvas = canvas_candidates[0]
    canvas_patch = np.asarray(canvas["patch"], dtype=int)
    grey = int(next(iter(set(int(x) for x in canvas_patch[canvas_patch != BLACK].tolist()))))
    assert grey != BLACK

    stamps: list[dict[str, object]] = []
    for comp in comps:
        if comp is canvas:
            continue
        patch = np.asarray(comp["patch"], dtype=int)
        h, w = patch.shape
        max_t = min(h, w) // 2
        thickness = 0
        for t in range(1, max_t + 1):
            if (
                np.all(patch[:t, :] == grey)
                and np.all(patch[-t:, :] == grey)
                and np.all(patch[:, :t] == grey)
                and np.all(patch[:, -t:] == grey)
            ):
                thickness = t
            else:
                break
        assert thickness >= 1, "stamp must have a grey frame"
        inner = patch[thickness : h - thickness, thickness : w - thickness]
        inner_colors = set(int(x) for x in inner[inner != BLACK].tolist())
        inner_colors.discard(grey)
        assert len(inner_colors) == 1, "stamp must have one fill color"
        hole_mask = inner == BLACK
        hole_box = _bbox(hole_mask)
        assert hole_box is not None, "stamp must have a hole"
        hr0, hc0, hr1, hc1 = hole_box
        shape = hole_mask[hr0 : hr1 + 1, hc0 : hc1 + 1]
        stamps.append(
            {
                "shape": shape,
                "color": int(next(iter(inner_colors))),
                "thickness": int(thickness),
            }
        )

    assert stamps, "expected at least one stamp"
    return canvas_patch, grey, stamps


def _candidate_placements(canvas_black: np.ndarray, shape: np.ndarray) -> list[dict[str, object]]:
    h, w = canvas_black.shape
    sh, sw = shape.shape
    coords = np.argwhere(shape)
    candidates: list[dict[str, object]] = []
    for r0 in range(h - sh + 1):
        for c0 in range(w - sw + 1):
            bits = 0
            ok = True
            for rr, cc in coords:
                r = int(r0 + rr)
                c = int(c0 + cc)
                if not canvas_black[r, c]:
                    ok = False
                    break
                bits |= 1 << (r * w + c)
            if ok:
                candidates.append({"pos": (int(r0), int(c0)), "bits": int(bits)})
    return candidates


def _solve_placements(
    canvas_black: np.ndarray, shapes: list[np.ndarray], max_solutions: int = 2
) -> list[list[tuple[int, int]]]:
    h, w = canvas_black.shape
    union_bits = 0
    for idx, value in enumerate(canvas_black.ravel()):
        if bool(value):
            union_bits |= 1 << idx

    candidates = [_candidate_placements(canvas_black, shape) for shape in shapes]
    assert all(cands for cands in candidates), "every shape must match somewhere on the canvas"

    order = sorted(range(len(shapes)), key=lambda i: (len(candidates[i]), -int(shapes[i].sum())))
    ordered_union = []
    for idx in order:
        bits = 0
        for cand in candidates[idx]:
            bits |= int(cand["bits"])
        ordered_union.append(bits)

    suffix_union = [0] * (len(order) + 1)
    for i in range(len(order) - 1, -1, -1):
        suffix_union[i] = int(suffix_union[i + 1] | ordered_union[i])

    picks: list[dict[str, object] | None] = [None] * len(order)
    solutions: list[list[tuple[int, int]]] = []

    def dfs(i: int, covered: int) -> None:
        if len(solutions) >= max_solutions:
            return
        if (covered | suffix_union[i]) != union_bits:
            return
        if i == len(order):
            if covered == union_bits:
                sol = [(-1, -1)] * len(shapes)
                for rank, cand in enumerate(picks):
                    assert cand is not None
                    sol[order[rank]] = tuple(cand["pos"])
                solutions.append(sol)
            return

        idx = order[i]
        local = sorted(
            candidates[idx],
            key=lambda cand: (-_bit_count(int(cand["bits"]) & ~covered), tuple(cand["pos"])),
        )
        for cand in local:
            new_covered = int(covered | int(cand["bits"]))
            if (new_covered | suffix_union[i + 1]) != union_bits:
                continue
            picks[i] = cand
            dfs(i + 1, new_covered)
            if len(solutions) >= max_solutions:
                return
        picks[i] = None

    dfs(0, 0)
    return solutions


def _paint_canvas(
    canvas_shape: tuple[int, int],
    grey: int,
    stamps: list[dict[str, object]],
    positions: list[tuple[int, int]],
) -> np.ndarray:
    out = np.full(canvas_shape, int(grey), dtype=int)
    order = sorted(range(len(stamps)), key=lambda i: int(stamps[i]["thickness"]))
    for idx in order:
        r0, c0 = positions[idx]
        shape = np.asarray(stamps[idx]["shape"], dtype=bool)
        color = int(stamps[idx]["color"])
        sub = out[r0 : r0 + shape.shape[0], c0 : c0 + shape.shape[1]]
        sub[shape] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each stamp's black hole to a black target on the canvas, then paint the
    canvas in ascending grey-border thickness order so later, thicker stamps overwrite earlier ones.
    """
    g = np.asarray(grid, dtype=int)
    canvas_patch, grey, stamps = _find_canvas_and_stamps(g)
    canvas_black = canvas_patch == BLACK
    shapes = [np.asarray(stamp["shape"], dtype=bool) for stamp in stamps]
    solutions = _solve_placements(canvas_black, shapes, max_solutions=2)
    assert len(solutions) == 1, "target placement must be unique"
    return _paint_canvas(canvas_patch.shape, grey, stamps, solutions[0])


def _random_shape(max_dim: int) -> np.ndarray | None:
    for _ in range(80):
        h = int(RNG.integers(2, max_dim + 1))
        w = int(RNG.integers(2, max_dim + 1))
        area_max = min(h * w - 1, 5)
        if area_max < 3:
            continue
        target_area = int(RNG.integers(3, area_max + 1))
        mask = np.zeros((h, w), dtype=bool)
        start = (int(RNG.integers(h)), int(RNG.integers(w)))
        mask[start] = True
        while int(mask.sum()) < target_area:
            frontier = []
            for rr, cc in np.argwhere(mask):
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        frontier.append((nr, nc))
            if not frontier:
                break
            nr, nc = frontier[int(RNG.integers(len(frontier)))]
            mask[nr, nc] = True
        box = _bbox(mask)
        if box is None:
            continue
        r0, c0, r1, c1 = box
        mask = mask[r0 : r1 + 1, c0 : c1 + 1]
        if min(mask.shape) < 2:
            continue
        if int(mask.sum()) < 3:
            continue
        return mask
    return None


def _sample_shapes(count: int, max_dim: int) -> list[np.ndarray] | None:
    shapes: list[np.ndarray] = []
    seen = set()
    for _ in range(120):
        while len(shapes) < count:
            shape = _random_shape(max_dim)
            if shape is None:
                break
            key = (shape.shape, shape.tobytes())
            if key in seen:
                continue
            seen.add(key)
            shapes.append(shape)
        if len(shapes) == count:
            return shapes
    return None


def _find_non_overlapping_pos(
    shape: np.ndarray, blocked: np.ndarray
) -> tuple[int, int] | None:
    h, w = blocked.shape
    sh, sw = shape.shape
    options = []
    for r0 in range(1, h - sh):
        for c0 in range(1, w - sw):
            if not np.any(blocked[r0 : r0 + sh, c0 : c0 + sw] & shape):
                options.append((r0, c0))
    if not options:
        return None
    return options[int(RNG.integers(len(options)))]


def _place_shapes(
    shapes: list[np.ndarray], canvas_shape: tuple[int, int], require_overlap: bool
) -> list[tuple[int, int]] | None:
    h, w = canvas_shape
    placed = [(-1, -1)] * len(shapes)
    union = np.zeros((h, w), dtype=bool)
    overlap_happened = False

    for idx, shape in enumerate(shapes):
        if idx == 0:
            pos = _find_non_overlapping_pos(shape, np.zeros((h, w), dtype=bool))
            if pos is None:
                return None
            placed[idx] = pos
            r0, c0 = pos
            union[r0 : r0 + shape.shape[0], c0 : c0 + shape.shape[1]] |= shape
            continue

        pos = None
        if require_overlap and not overlap_happened:
            sh, sw = shape.shape
            options = []
            coords = np.argwhere(shape)
            for r0 in range(1, h - sh):
                for c0 in range(1, w - sw):
                    overlap = 0
                    for rr, cc in coords:
                        if union[int(r0 + rr), int(c0 + cc)]:
                            overlap += 1
                    if overlap <= 0 or overlap >= int(shape.sum()):
                        continue
                    candidate_mask = np.zeros_like(union)
                    candidate_mask[r0 : r0 + sh, c0 : c0 + sw] = shape
                    if int((candidate_mask & union).sum()) != overlap:
                        continue
                    if int((candidate_mask & ~union).sum()) == 0:
                        continue
                    options.append((r0, c0))
            if options:
                pos = options[int(RNG.integers(len(options)))]
                overlap_happened = True

        if pos is None:
            pos = _find_non_overlapping_pos(shape, _dilate4(union))
        if pos is None:
            return None
        placed[idx] = pos
        r0, c0 = pos
        union[r0 : r0 + shape.shape[0], c0 : c0 + shape.shape[1]] |= shape

    if require_overlap != overlap_happened:
        return None
    return placed


def _make_stamp(shape: np.ndarray, color: int, thickness: int) -> np.ndarray:
    sh, sw = shape.shape
    stamp = np.full((sh + 2 + 2 * thickness, sw + 2 + 2 * thickness), GREY, dtype=int)
    inner = stamp[thickness:-thickness, thickness:-thickness]
    inner[:, :] = color
    hole = inner[1 : 1 + sh, 1 : 1 + sw]
    hole[shape] = BLACK
    return stamp


def _build_input(stamps: list[np.ndarray], canvas: np.ndarray) -> np.ndarray:
    gap_y = 1
    gap_x = 2
    top_margin = 0
    left_margin = 0
    stamp_width = max(stamp.shape[1] for stamp in stamps)
    stamp_height = sum(stamp.shape[0] for stamp in stamps) + gap_y * (len(stamps) - 1)
    h = max(stamp_height, canvas.shape[0])
    w = stamp_width + gap_x + canvas.shape[1]
    grid = np.full((h, w), BLACK, dtype=int)

    y = top_margin
    layout_order = list(range(len(stamps)))
    RNG.shuffle(layout_order)
    ordered = [stamps[i] for i in layout_order]
    for stamp in ordered:
        sh, sw = stamp.shape
        x = int(RNG.integers(0, stamp_width - sw + 1))
        grid[y : y + sh, x : x + sw] = stamp
        y += sh + gap_y

    cy = int(RNG.integers(0, h - canvas.shape[0] + 1))
    cx = left_margin + stamp_width + gap_x
    grid[cy : cy + canvas.shape[0], cx : cx + canvas.shape[1]] = canvas
    return grid


def _generate_attempt(require_overlap: bool) -> np.ndarray | None:
    count = int(RNG.integers(2, 4))
    max_dim = 3
    shapes = _sample_shapes(count, max_dim)
    if shapes is None:
        return None

    colors = list(RNG.choice(FILL_COLORS, size=count, replace=False))
    thicknesses = list(range(1, count + 1))
    RNG.shuffle(colors)
    RNG.shuffle(thicknesses)
    stamps = [
        {"shape": shape, "color": int(color), "thickness": int(thickness)}
        for shape, color, thickness in zip(shapes, colors, thicknesses)
    ]

    max_h = max(shape.shape[0] for shape in shapes)
    max_w = max(shape.shape[1] for shape in shapes)
    for _ in range(60):
        canvas_h = int(RNG.integers(max(7, max_h + 4), 10))
        canvas_w = int(RNG.integers(max(7, max_w + 4), 10))
        placements = _place_shapes(shapes, (canvas_h, canvas_w), require_overlap=require_overlap)
        if placements is None:
            continue

        canvas_black = np.zeros((canvas_h, canvas_w), dtype=bool)
        for shape, (r0, c0) in zip(shapes, placements):
            canvas_black[r0 : r0 + shape.shape[0], c0 : c0 + shape.shape[1]] |= shape
        grey_mask = np.full((canvas_h, canvas_w), True, dtype=bool)
        grey_mask[canvas_black] = False
        if len(_connected_components(grey_mask)) != 1:
            continue

        solutions = _solve_placements(canvas_black, shapes, max_solutions=2)
        if len(solutions) != 1:
            continue
        if any(tuple(sol) != tuple(act) for sol, act in zip(solutions[0], placements)):
            continue

        canvas = np.full((canvas_h, canvas_w), GREY, dtype=int)
        canvas[canvas_black] = BLACK
        stamp_grids = [_make_stamp(shape, int(color), int(thickness)) for shape, color, thickness in zip(shapes, colors, thicknesses)]
        grid = _build_input(stamp_grids, canvas)
        expected = _paint_canvas(canvas.shape, GREY, stamps, placements)
        if np.array_equal(main(grid), expected):
            return grid
    return None


def generate_input() -> np.ndarray:
    global NEXT_OVERLAP_BRANCH
    want_overlap = bool(NEXT_OVERLAP_BRANCH)
    for branch in (want_overlap, not want_overlap):
        for _ in range(MAX_GENERATION_RETRIES):
            grid = _generate_attempt(branch)
            if grid is not None:
                NEXT_OVERLAP_BRANCH = not branch
                return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
