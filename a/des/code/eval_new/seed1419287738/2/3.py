import numpy as np


_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_RNG = np.random.default_rng()

_BG = 0
_COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
_GRID_H_RANGE = (20, 31)
_GRID_W_RANGE = (20, 31)
_FRAME_COUNT_CHOICES = (2, 3)
_FRAME_INTERIORS = [
    (2, 4),
    (4, 2),
    (2, 5),
    (5, 2),
    (2, 6),
    (6, 2),
    (3, 4),
    (4, 3),
    (3, 5),
    (5, 3),
    (4, 4),
]
_MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in _N4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _coords_to_mask(coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask, (r0, c0, r1, c1)


def _is_frame_component(grid: np.ndarray, bg: int, color: int, coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    mask, _ = _coords_to_mask(coords)
    if mask.sum() != 2 * (h + w) - 4:
        return False
    if not np.all(mask[0, :]) or not np.all(mask[-1, :]) or not np.all(mask[:, 0]) or not np.all(mask[:, -1]):
        return False
    if np.any(mask[1:-1, 1:-1]):
        return False
    return bool(np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == bg))


def _parse_scene(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    groups: list[dict] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        comps = _components(g == color)
        frame = None
        fragments: list[dict] = []
        for comp in comps:
            if _is_frame_component(g, bg, color, comp):
                assert frame is None, "expected exactly one frame per color"
                mask, box = _coords_to_mask(comp)
                frame = {"coords": comp, "bbox": box, "mask": mask}
            else:
                mask, box = _coords_to_mask(comp)
                fragments.append(
                    {
                        "coords": comp,
                        "bbox": box,
                        "mask": mask,
                        "size": int(mask.sum()),
                        "top": int(box[0]),
                        "left": int(box[1]),
                    }
                )
        assert frame is not None, "missing frame for color"
        assert fragments, "missing fragments for color"
        groups.append({"color": color, "frame": frame, "fragments": fragments})
    assert groups, "expected at least one colored group"
    return bg, groups


def _find_first_fit(occupied: np.ndarray, piece_mask: np.ndarray) -> tuple[int, int] | None:
    ih, iw = occupied.shape
    ph, pw = piece_mask.shape
    for r0 in range(ih - ph + 1):
        for c0 in range(iw - pw + 1):
            ok = True
            for rr, cc in np.argwhere(piece_mask):
                if occupied[r0 + int(rr), c0 + int(cc)]:
                    ok = False
                    break
            if ok:
                return r0, c0
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """Pack scattered fragments into their matching color frames using size-first row-major placement.

    Input structure:
    - The background is black. Several distinct colors form hollow axis-aligned rectangular frames with border width 1.
    - Each frame interior is entirely background in the input.
    - Scattered elsewhere are monochrome 4-connected fragments of those same frame colors, each fragment containing 2 to 6 cells.

    Rule:
    - For each color, identify its unique hollow frame and collect all other same-color components as fragments.
    - Sort a color's fragments by descending area; ties break by smaller top row, then smaller left column.
    - Starting from an empty frame interior, place the fragments in that order. For each fragment, scan frame-interior anchor positions row by row from top-left and choose the first anchor where the fragment's bounding box fits and all occupied cells land on still-empty interior cells.
    - Erase all scattered fragments outside the frames. Keep the frame borders unchanged.

    Key insight:
    - The transformation is a three-step mechanism: color groups choose the destination frame, fragment area determines placement order, and a greedy row-major first-fit scan determines the exact packed positions.
    """
    g = np.asarray(grid, dtype=int)
    bg, groups = _parse_scene(g)

    out = np.full_like(g, bg)
    for group in groups:
        color = int(group["color"])
        frame = group["frame"]
        r0, c0, r1, c1 = frame["bbox"]
        for r, c in frame["coords"]:
            out[r, c] = color

        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        occupied = np.zeros((inner_h, inner_w), dtype=bool)
        fragments = sorted(group["fragments"], key=lambda frag: (-frag["size"], frag["top"], frag["left"]))
        total_area = sum(int(frag["size"]) for frag in fragments)
        assert total_area == inner_h * inner_w, "fragment area must match frame interior"

        for frag in fragments:
            pos = _find_first_fit(occupied, frag["mask"])
            assert pos is not None, "every fragment must fit somewhere in its frame"
            fr, fc = pos
            for rr, cc in np.argwhere(frag["mask"]):
                rr_i = int(rr)
                cc_i = int(cc)
                occupied[fr + rr_i, fc + cc_i] = True
                out[r0 + 1 + fr + rr_i, c0 + 1 + fc + cc_i] = color

        assert bool(np.all(occupied)), "frame interior must be filled exactly"

    return out


def _shape_bbox_stats(shape: tuple[tuple[int, int], ...]) -> tuple[int, int, int]:
    max_r = max(r for r, _ in shape)
    max_c = max(c for _, c in shape)
    area = (max_r + 1) * (max_c + 1)
    return area, max_r, max_c


def _build_shape_library(max_size: int = 6) -> dict[int, list[tuple[tuple[int, int], ...]]]:
    library: dict[int, set[frozenset[tuple[int, int]]]] = {1: {frozenset({(0, 0)})}}
    for size in range(2, max_size + 1):
        new_shapes: set[frozenset[tuple[int, int]]] = set()
        for shape in library[size - 1]:
            frontier: set[tuple[int, int]] = set()
            for r, c in shape:
                for dr, dc in _N4:
                    nr, nc = r + dr, c + dc
                    if nr < 0 or nc < 0 or (nr, nc) in shape:
                        continue
                    frontier.add((nr, nc))
            for cell in frontier:
                new_shapes.add(frozenset(set(shape) | {cell}))
        library[size] = new_shapes

    out: dict[int, list[tuple[tuple[int, int], ...]]] = {}
    for size in range(2, max_size + 1):
        seq = [tuple(sorted(shape)) for shape in library[size]]
        seq.sort(key=lambda shape: (_shape_bbox_stats(shape)[0] - size, _shape_bbox_stats(shape)[0], shape))
        out[size] = seq
    return out


def _shape_to_mask(shape: tuple[tuple[int, int], ...]) -> np.ndarray:
    max_r = max(r for r, _ in shape)
    max_c = max(c for _, c in shape)
    mask = np.zeros((max_r + 1, max_c + 1), dtype=bool)
    for r, c in shape:
        mask[r, c] = True
    return mask


_SHAPE_LIBRARY = _build_shape_library()


def _enumerate_size_sequences(total: int) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []

    def rec(remaining: int, max_part: int, path: list[int]) -> None:
        if remaining == 0:
            if 2 <= len(path) <= 5:
                out.append(tuple(path))
            return
        if len(path) >= 5:
            return
        min_next = 2
        max_next = min(6, max_part, remaining)
        for part in range(max_next, min_next - 1, -1):
            left = remaining - part
            if left != 0:
                min_left = 2 * max(0, 1 - 0)  # keep expression simple for readability
                max_count_left = 5 - (len(path) + 1)
                if left < 2 or max_count_left <= 0:
                    if left != 0:
                        continue
                feasible = False
                for count in range(1, max_count_left + 1):
                    if 2 * count <= left <= 6 * count:
                        feasible = True
                        break
                if not feasible:
                    continue
            rec(left, part, path + [part])

    rec(total, 6, [])
    return out


_SIZE_SEQUENCES = {}
for _ih, _iw in _FRAME_INTERIORS:
    _SIZE_SEQUENCES.setdefault(_ih * _iw, _enumerate_size_sequences(_ih * _iw))


def _empty_component_sizes(empty_mask: np.ndarray) -> list[int]:
    return [len(comp) for comp in _components(empty_mask)]


def _partition_interior(
    inner_h: int, inner_w: int, sizes: tuple[int, ...], rng: np.random.Generator
) -> list[np.ndarray] | None:
    failed: set[tuple[int, bytes]] = set()

    def rec(occupied: np.ndarray, idx: int) -> list[np.ndarray] | None:
        key = (idx, occupied.tobytes())
        if key in failed:
            return None
        if idx == len(sizes):
            return [] if bool(np.all(occupied)) else None

        flat = np.flatnonzero(~occupied.ravel())
        if flat.size == 0:
            failed.add(key)
            return None
        start = int(flat[0])
        r0, c0 = divmod(start, inner_w)
        need = int(sizes[idx])

        candidates: list[tuple[np.ndarray, np.ndarray, int]] = []
        for shape in _SHAPE_LIBRARY[need]:
            ok = True
            new_occ = occupied.copy()
            for dr, dc in shape:
                rr = r0 + dr
                cc = c0 + dc
                if rr >= inner_h or cc >= inner_w or new_occ[rr, cc]:
                    ok = False
                    break
                new_occ[rr, cc] = True
            if not ok:
                continue
            empty = ~new_occ
            if any(size == 1 for size in _empty_component_sizes(empty)):
                continue
            irregularity = _shape_bbox_stats(shape)[0] - need
            candidates.append((new_occ, _shape_to_mask(shape), irregularity))

        if not candidates:
            failed.add(key)
            return None

        order = list(range(len(candidates)))
        rng.shuffle(order)
        order.sort(key=lambda i: candidates[i][2], reverse=True)
        for i in order:
            next_occ, piece_mask, _score = candidates[i]
            tail = rec(next_occ, idx + 1)
            if tail is not None:
                return [piece_mask] + tail

        failed.add(key)
        return None

    return rec(np.zeros((inner_h, inner_w), dtype=bool), 0)


def _choose_partition(
    inner_h: int, inner_w: int, prefer_tie: bool, rng: np.random.Generator
) -> tuple[tuple[int, ...], list[np.ndarray]] | None:
    area = inner_h * inner_w
    candidates = list(_SIZE_SEQUENCES[area])
    if prefer_tie:
        tied = [seq for seq in candidates if len(set(seq)) < len(seq)]
        if tied:
            candidates = tied
    else:
        mixed = [seq for seq in candidates if len(set(seq)) > 1]
        if mixed:
            candidates = mixed
    rng.shuffle(candidates)
    for sizes in candidates[:16]:
        pieces = _partition_interior(inner_h, inner_w, sizes, rng)
        if pieces is not None:
            return sizes, pieces
    return None


def _mark_rect(blocked: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    h, w = blocked.shape
    rr0 = max(0, r0)
    cc0 = max(0, c0)
    rr1 = min(h - 1, r1)
    cc1 = min(w - 1, c1)
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _frame_border_cells(top: int, left: int, inner_h: int, inner_w: int) -> list[tuple[int, int]]:
    r0, c0 = top, left
    r1 = top + inner_h + 1
    c1 = left + inner_w + 1
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _frame_anchor_candidates(blocked: np.ndarray, inner_h: int, inner_w: int) -> list[tuple[int, int]]:
    h, w = blocked.shape
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    candidates: list[tuple[int, int]] = []
    for r0 in range(h - outer_h + 1):
        for c0 in range(w - outer_w + 1):
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(h, r0 + outer_h + 1)
            cc1 = min(w, c0 + outer_w + 1)
            if np.any(blocked[rr0:rr1, cc0:cc1]):
                continue
            candidates.append((r0, c0))
    return candidates


def _fragment_anchor_candidates(
    blocked: np.ndarray, piece_mask: np.ndarray, lower_bound: tuple[int, int] | None = None
) -> list[tuple[int, int]]:
    h, w = blocked.shape
    ph, pw = piece_mask.shape
    candidates: list[tuple[int, int]] = []
    for r0 in range(h - ph + 1):
        for c0 in range(w - pw + 1):
            if lower_bound is not None and (r0, c0) <= lower_bound:
                continue
            ok = True
            for rr, cc in np.argwhere(piece_mask):
                if blocked[r0 + int(rr), c0 + int(cc)]:
                    ok = False
                    break
            if ok:
                candidates.append((r0, c0))
    return candidates


def _mark_fragment(blocked: np.ndarray, top: int, left: int, piece_mask: np.ndarray) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    h, w = blocked.shape
    for rr, cc in np.argwhere(piece_mask):
        r = top + int(rr)
        c = left + int(cc)
        cells.append((r, c))
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    blocked[nr, nc] = True
    return cells


def _build_group_specs(rng: np.random.Generator) -> list[dict] | None:
    group_count = int(rng.choice(_FRAME_COUNT_CHOICES))
    colors = [int(v) for v in rng.choice(_COLOR_POOL, size=group_count, replace=False)]

    tie_flags = [False] * group_count
    tie_flags[int(rng.integers(group_count))] = True
    if group_count > 1:
        tie_flags[int(rng.integers(group_count))] = False

    specs: list[dict] = []
    for i, color in enumerate(colors):
        dims = list(_FRAME_INTERIORS)
        rng.shuffle(dims)
        spec = None
        for inner_h, inner_w in dims:
            chosen = _choose_partition(inner_h, inner_w, tie_flags[i], rng)
            if chosen is None:
                continue
            sizes, pieces = chosen
            spec = {
                "color": color,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "sizes": sizes,
                "pieces": pieces,
            }
            break
        if spec is None:
            return None
        specs.append(spec)
    return specs


def _draw_grid(h: int, w: int, group_specs: list[dict], rng: np.random.Generator) -> np.ndarray | None:
    grid = np.full((h, w), _BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    order = list(range(len(group_specs)))
    rng.shuffle(order)
    for idx in order:
        spec = group_specs[idx]
        anchors = _frame_anchor_candidates(blocked, spec["inner_h"], spec["inner_w"])
        if not anchors:
            return None
        top, left = anchors[int(rng.integers(len(anchors)))]
        spec["frame_top"] = top
        spec["frame_left"] = left
        border_cells = _frame_border_cells(top, left, spec["inner_h"], spec["inner_w"])
        spec["frame_cells"] = border_cells
        for r, c in border_cells:
            grid[r, c] = spec["color"]
        _mark_rect(
            blocked,
            top - 1,
            left - 1,
            top + spec["inner_h"] + 2,
            left + spec["inner_w"] + 2,
        )

    for spec in group_specs:
        pieces = []
        prev_size = None
        prev_anchor = None
        for size, piece_mask in zip(spec["sizes"], spec["pieces"]):
            lower_bound = prev_anchor if prev_size == size else None
            anchors = _fragment_anchor_candidates(blocked, piece_mask, lower_bound)
            if not anchors:
                return None
            pick = anchors[int(rng.integers(len(anchors)))]
            cells = _mark_fragment(blocked, pick[0], pick[1], piece_mask)
            pieces.append({"cells": cells, "top": pick[0], "left": pick[1], "mask": piece_mask})
            prev_size = size
            prev_anchor = pick
        spec["fragment_instances"] = pieces

    for spec in group_specs:
        color = int(spec["color"])
        for frag in spec["fragment_instances"]:
            for r, c in frag["cells"]:
                grid[r, c] = color

    return grid


def _validate_generated_grid(grid: np.ndarray, expected_group_count: int) -> bool:
    bg, groups = _parse_scene(grid)
    if bg != _BG or len(groups) != expected_group_count:
        return False
    if any(len(group["fragments"]) < 2 for group in groups):
        return False
    out = main(grid)
    if out.shape != grid.shape:
        return False
    for group in groups:
        r0, c0, r1, c1 = group["frame"]["bbox"]
        if not np.all(out[r0 + 1 : r1, c0 + 1 : c1] == group["color"]):
            return False
    return bool(np.any(out != grid))


def _gen_attempt() -> np.ndarray | None:
    rng = _RNG
    group_specs = _build_group_specs(rng)
    if group_specs is None:
        return None

    for _ in range(30):
        grid_h = int(rng.integers(*_GRID_H_RANGE))
        grid_w = int(rng.integers(*_GRID_W_RANGE))
        grid = _draw_grid(grid_h, grid_w, [dict(spec) for spec in group_specs], rng)
        if grid is None:
            continue
        if _validate_generated_grid(grid, len(group_specs)):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
