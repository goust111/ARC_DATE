import numpy as np


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    pts = list(coords)
    r0, c0, _, _ = _bbox(pts)
    return tuple(sorted((r - r0, c - c0) for r, c in pts))


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _component_list(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": len(coords),
                }
            )
    return comps


def _border_cycle(h: int, w: int) -> list[tuple[tuple[int, int], str]]:
    cycle: list[tuple[tuple[int, int], str]] = []
    for c in range(w):
        cycle.append(((0, c), "T"))
    for r in range(1, h - 1):
        cycle.append(((r, w - 1), "R"))
    for c in range(w - 1, -1, -1):
        cycle.append(((h - 1, c), "B"))
    for r in range(h - 2, 0, -1):
        cycle.append(((r, 0), "L"))
    return cycle


def _border_set(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _frame_candidates(grid: np.ndarray, bg: int) -> list[dict]:
    candidates: list[dict] = []
    for color in [int(x) for x in np.unique(grid) if int(x) != bg]:
        coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == color)]
        if not coords:
            continue
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 5 or w < 5:
            continue
        coord_set = set(coords)
        border = _border_set(r0, c0, r1, c1)
        if not coord_set <= border:
            continue
        if coord_set == border:
            continue
        touched = {
            side
            for (rr, cc), side in _border_cycle(h, w)
            if (r0 + rr, c0 + cc) in coord_set
        }
        if touched != {"T", "R", "B", "L"}:
            continue
        hole = sorted(border - coord_set)
        hole_rows = {r for r, _ in hole}
        hole_cols = {c for _, c in hole}
        if len(hole_rows) == 1 or len(hole_cols) == 1:
            continue
        candidates.append(
            {
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "present": coord_set,
                "hole": hole,
                "hole_norm": _normalize(hole),
            }
        )
    return candidates


def _outside_all_frame_boxes(coords: list[tuple[int, int]], frames: list[dict]) -> bool:
    for r, c in coords:
        for frame in frames:
            r0, c0, r1, c1 = frame["bbox"]
            if r0 <= r <= r1 and c0 <= c <= c1:
                return False
    return True


def _key_candidates(grid: np.ndarray, frames: list[dict]) -> list[dict]:
    keys: list[dict] = []
    bg = _get_bg(grid)
    for comp in _component_list(grid, bg):
        if comp["area"] < 2:
            continue
        if not _outside_all_frame_boxes(comp["coords"], frames):
            continue
        keys.append(
            {
                "color": int(comp["color"]),
                "coords": comp["coords"],
                "bbox": comp["bbox"],
                "shape": _normalize(comp["coords"]),
                "area": int(comp["area"]),
            }
        )
    return keys


def _pair_fit(
    hole: list[tuple[int, int]],
    key_a: tuple[tuple[int, int], ...],
    key_b: tuple[tuple[int, int], ...],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    hole_set = set(hole)
    if len(key_a) + len(key_b) != len(hole_set):
        return None
    for hr, hc in hole:
        for ar, ac in key_a:
            dr = hr - ar
            dc = hc - ac
            placed_a = {(r + dr, c + dc) for r, c in key_a}
            if len(placed_a) != len(key_a) or not placed_a <= hole_set:
                continue
            placed_b = hole_set - placed_a
            if len(placed_b) != len(key_b):
                continue
            if _normalize(placed_b) == key_b:
                return placed_a, placed_b
    return None


def _find_unique_solution(grid: np.ndarray) -> dict:
    bg = _get_bg(grid)
    frames = _frame_candidates(grid, bg)
    assert frames, "expected at least one broken rectangular frame"
    keys = _key_candidates(grid, frames)
    assert len(keys) >= 2, "expected at least two key sprites"

    solutions: list[dict] = []
    for frame_idx, frame in enumerate(frames):
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                fit = _pair_fit(frame["hole"], keys[i]["shape"], keys[j]["shape"])
                if fit is not None:
                    solutions.append(
                        {
                            "frame_index": frame_idx,
                            "frame": frame,
                            "key_indices": (i, j),
                            "placements": fit,
                        }
                    )
    assert len(solutions) == 1, "expected exactly one frame/key-pair match"
    return {
        "bg": bg,
        "frames": frames,
        "keys": keys,
        "solution": solutions[0],
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Find the only broken frame whose missing border is exactly the union of two translated keys, repair it, and crop its bbox.

    Input structure:
    - A mostly background grid contains several one-cell-thick monochrome rectangular frames, each missing a bent patch of border pixels.
    - Separate monochrome key sprites lie outside every frame bbox; single-cell speckles may exist as noise but keys are multi-cell components.
    - Frame colors appear only on their own borders, while frame interiors preserve unrelated multicolor content.

    Rule:
    - Detect each broken rectangular frame and compute its missing border cells.
    - Find the unique pair of key sprites that can each be translated, without rotation or overlap, so their union matches exactly one frame hole.
    - Recolor that fitted pair to the frame color, complete the frame, and return the repaired frame's full bbox crop with the interior unchanged.

    Key insight:
    - No single key ever matches a hole; the target is identified only by testing two-key translation unions against the actual missing border geometry of every candidate frame.
    """
    g = np.asarray(grid, dtype=int)
    info = _find_unique_solution(g)
    frame = info["solution"]["frame"]
    r0, c0, r1, c1 = frame["bbox"]
    out = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    for hr, hc in frame["hole"]:
        out[hr - r0, hc - c0] = frame["color"]
    return out


def _shape_from_arc(arc: list[tuple[tuple[int, int], str]]) -> tuple[tuple[int, int], ...]:
    return _normalize([coord for coord, _side in arc])


def _random_fill(rng: np.random.Generator, h: int, w: int, palette: list[int]) -> np.ndarray:
    probs = rng.random(len(palette))
    probs = probs / probs.sum()
    return rng.choice(np.array(palette, dtype=int), size=(h, w), p=probs).astype(int)


def _polyomino(rng: np.random.Generator, area: int) -> tuple[tuple[int, int], ...]:
    cells = {(0, 0)}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) not in cells:
                    frontier.append((nr, nc))
        pick = frontier[int(rng.integers(len(frontier)))]
        cells.add(pick)
    return _normalize(cells)


def _shape_sides(shape: tuple[tuple[int, int], ...]) -> set[str]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    h = max(rows) + 1
    w = max(cols) + 1
    sides: set[str] = set()
    for r, c in shape:
        if r == 0:
            sides.add("T")
        if r == h - 1:
            sides.add("B")
        if c == 0:
            sides.add("L")
        if c == w - 1:
            sides.add("R")
    return sides


def _arc_preserves_all_frame_sides(arc: list[tuple[tuple[int, int], str]], h: int, w: int) -> bool:
    total = {"T": w, "B": w, "L": max(0, h - 2), "R": max(0, h - 2)}
    removed = {side: 0 for side in total}
    for _coord, side in arc:
        removed[side] += 1
    return all(removed[side] < total[side] for side in total)


def _is_frame_like_shape(shape: tuple[tuple[int, int], ...]) -> bool:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    h = max(rows) + 1
    w = max(cols) + 1
    if h < 5 or w < 5:
        return False
    shape_set = set(shape)
    border = _border_set(0, 0, h - 1, w - 1)
    return shape_set <= border and shape_set != border and _shape_sides(shape) == {"T", "R", "B", "L"}


def _generate_target_arc(
    rng: np.random.Generator, h: int, w: int
) -> tuple[list[tuple[tuple[int, int], str]], tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    cycle = _border_cycle(h, w)
    per = len(cycle)
    options: list[
        tuple[list[tuple[tuple[int, int], str]], tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]
    ] = []
    for length in range(6, min(10, per - 2) + 1):
        for start in range(per):
            arc = [cycle[(start + k) % per] for k in range(length)]
            if len({side for _coord, side in arc}) < 2:
                continue
            if not _arc_preserves_all_frame_sides(arc, h, w):
                continue
            for split in range(2, length - 1):
                left = arc[:split]
                right = arc[split:]
                options.append((arc, _shape_from_arc(left), _shape_from_arc(right)))
    if options:
        return options[int(rng.integers(len(options)))]
    raise RuntimeError("failed to generate target hole")


def _generate_decoy_arc(
    rng: np.random.Generator,
    h: int,
    w: int,
    target_len: int,
    forbidden: set[tuple[tuple[int, int], ...]],
) -> tuple[list[tuple[tuple[int, int], str]], tuple[tuple[int, int], ...]]:
    cycle = _border_cycle(h, w)
    per = len(cycle)
    options: list[tuple[list[tuple[tuple[int, int], str]], tuple[tuple[int, int], ...]]] = []
    lengths = {target_len}
    lengths.update(range(5, min(10, per - 2) + 1))
    for length in sorted(lengths):
        for start in range(per):
            arc = [cycle[(start + k) % per] for k in range(length)]
            if len({side for _coord, side in arc}) < 2:
                continue
            if not _arc_preserves_all_frame_sides(arc, h, w):
                continue
            shape = _shape_from_arc(arc)
            if shape in forbidden:
                continue
            options.append((arc, shape))
    if options:
        same_len = [opt for opt in options if len(opt[0]) == target_len]
        pool = same_len if same_len and rng.random() < 0.7 else options
        return pool[int(rng.integers(len(pool)))]
    raise RuntimeError("failed to generate decoy hole")


def _place_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    mask = patch != 0
    region = grid[r0 : r0 + h, c0 : c0 + w]
    region[mask] = patch[mask]


def _can_place_box(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    hh, ww = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + h > hh or c0 + w > ww:
        return False
    pr0 = max(0, r0 - pad)
    pc0 = max(0, c0 - pad)
    pr1 = min(hh, r0 + h + pad)
    pc1 = min(ww, c0 + w + pad)
    return not bool(occupied[pr0:pr1, pc0:pc1].any())


def _mark_box(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    occupied[r0 : r0 + h, c0 : c0 + w] = True


def _shape_patch(shape: tuple[tuple[int, int], ...], color: int) -> np.ndarray:
    h = max(r for r, _ in shape) + 1
    w = max(c for _, c in shape) + 1
    patch = np.zeros((h, w), dtype=int)
    for r, c in shape:
        patch[r, c] = color
    return patch


def _sample_shape_not_in(
    rng: np.random.Generator,
    area: int,
    forbidden: set[tuple[tuple[int, int], ...]],
) -> tuple[tuple[int, int], ...]:
    for _ in range(2000):
        shape = _polyomino(rng, area)
        if shape not in forbidden and not _is_frame_like_shape(shape):
            return shape
    raise RuntimeError("failed to sample decoy key shape")


def _build_frame_patch(
    rng: np.random.Generator,
    h: int,
    w: int,
    frame_color: int,
    hole_arc: list[tuple[tuple[int, int], str]],
    palette: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    interior = _random_fill(rng, h - 2, w - 2, palette)
    patch = np.full((h, w), frame_color, dtype=int)
    patch[1:-1, 1:-1] = interior
    for (r, c), _side in hole_arc:
        patch[r, c] = 0
    solved = patch.copy()
    for (r, c), _side in hole_arc:
        solved[r, c] = frame_color
    return patch, solved


def _semantic_checks(grid: np.ndarray, expected: np.ndarray, target_hole_area: int) -> bool:
    try:
        info = _find_unique_solution(grid)
    except AssertionError:
        return False
    out = main(grid)
    if not np.array_equal(out, expected):
        return False

    holes = [frame["hole_norm"] for frame in info["frames"]]
    keys = info["keys"]
    if any(key["shape"] in holes for key in keys):
        return False
    if not any(key["area"] == target_hole_area for key in keys):
        return False

    target_frame = info["solution"]["frame"]
    same_area_frames = sum(len(frame["hole"]) == len(target_frame["hole"]) for frame in info["frames"])
    if same_area_frames < 2:
        return False

    target_area = len(target_frame["hole"])
    ti, tj = info["solution"]["key_indices"]
    same_sum_decoy = False
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if (i, j) == (ti, tj):
                continue
            if keys[i]["area"] + keys[j]["area"] == target_area:
                same_sum_decoy = True
    if not same_sum_decoy:
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(23, 31))
    w = int(rng.integers(23, 31))
    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    all_colors = list(range(1, 10))
    rng.shuffle(all_colors)
    n_frames = int(rng.integers(2, 4))
    key_count = 5 + int(rng.random() < 0.4)
    need_colors = n_frames + key_count + 2
    if need_colors > len(all_colors):
        return None
    frame_colors = all_colors[:n_frames]
    key_colors = all_colors[n_frames : n_frames + key_count]
    fill_palette = [0] + all_colors[n_frames + key_count : n_frames + key_count + 2]

    frame_specs = []
    forbidden_holes: set[tuple[tuple[int, int], ...]] = set()

    target_dims = (
        int(rng.integers(7, 11)),
        int(rng.integers(7, 12)),
    )
    target_arc, key_a, key_b = _generate_target_arc(rng, *target_dims)
    target_hole_shape = _shape_from_arc(target_arc)
    forbidden_holes.add(target_hole_shape)
    frame_specs.append(
        {
            "dims": target_dims,
            "hole_arc": target_arc,
            "hole_shape": target_hole_shape,
            "color": frame_colors[0],
            "target": True,
            "split_keys": (key_a, key_b),
        }
    )

    for idx in range(1, n_frames):
        dims = (int(rng.integers(6, 11)), int(rng.integers(6, 12)))
        hole_arc, hole_shape = _generate_decoy_arc(rng, *dims, len(target_arc), forbidden_holes)
        forbidden_holes.add(hole_shape)
        frame_specs.append(
            {
                "dims": dims,
                "hole_arc": hole_arc,
                "hole_shape": hole_shape,
                "color": frame_colors[idx],
                "target": False,
            }
        )

    key_shapes = [key_a, key_b]
    forbidden_key_shapes = set(forbidden_holes) | {key_a, key_b}
    key_shapes.append(_sample_shape_not_in(rng, len(target_arc), forbidden_key_shapes))
    forbidden_key_shapes.add(key_shapes[-1])
    key_shapes.append(_sample_shape_not_in(rng, len(key_a), forbidden_key_shapes))
    forbidden_key_shapes.add(key_shapes[-1])
    key_shapes.append(_sample_shape_not_in(rng, len(key_b), forbidden_key_shapes))
    forbidden_key_shapes.add(key_shapes[-1])
    while len(key_shapes) < key_count:
        area = int(rng.integers(2, 6))
        key_shapes.append(_sample_shape_not_in(rng, area, forbidden_key_shapes))
        forbidden_key_shapes.add(key_shapes[-1])

    target_patch = None
    target_solved = None
    target_bbox = None
    for spec in frame_specs:
        fh, fw = spec["dims"]
        patch, solved = _build_frame_patch(rng, fh, fw, spec["color"], spec["hole_arc"], fill_palette)
        placed = False
        for _ in range(300):
            r0 = int(rng.integers(0, h - fh + 1))
            c0 = int(rng.integers(0, w - fw + 1))
            if not _can_place_box(occupied, r0, c0, fh, fw, pad=1):
                continue
            _place_patch(grid, patch, r0, c0)
            _mark_box(occupied, r0, c0, fh, fw)
            spec["placed_bbox"] = (r0, c0, r0 + fh - 1, c0 + fw - 1)
            if spec["target"]:
                target_patch = patch
                target_solved = solved
                target_bbox = spec["placed_bbox"]
            placed = True
            break
        if not placed:
            return None

    for color, shape in zip(key_colors, key_shapes):
        patch = _shape_patch(shape, color)
        kh, kw = patch.shape
        placed = False
        for _ in range(300):
            r0 = int(rng.integers(0, h - kh + 1))
            c0 = int(rng.integers(0, w - kw + 1))
            if not _can_place_box(occupied, r0, c0, kh, kw, pad=1):
                continue
            _place_patch(grid, patch, r0, c0)
            _mark_box(occupied, r0, c0, kh, kw)
            placed = True
            break
        if not placed:
            return None

    for _ in range(int(rng.integers(8, 18))):
        r = int(rng.integers(h))
        c = int(rng.integers(w))
        if grid[r, c] != 0:
            continue
        ok = True
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] != 0:
                ok = False
                break
        if ok:
            grid[r, c] = fill_palette[1 + int(rng.integers(len(fill_palette) - 1))]

    assert target_patch is not None and target_solved is not None and target_bbox is not None
    r0, c0, r1, c1 = target_bbox
    expected = target_solved.copy()
    if not _semantic_checks(grid, expected, len(target_arc)):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(500):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
