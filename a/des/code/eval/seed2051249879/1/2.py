import random
from collections import deque

import numpy as np


BG = 8
PIPE = 7
SYSTEM_COLORS = (1, 2, 3, 4, 5, 6, 9)
CASES = (
    ((0, 1), (1, 0)),
    ((0, 1), (-1, 0)),
    ((0, -1), (1, 0)),
    ((0, -1), (-1, 0)),
    ((1, 0), (0, 1)),
    ((1, 0), (0, -1)),
    ((-1, 0), (0, 1)),
    ((-1, 0), (0, -1)),
)
GEN_RETRIES = 400
SYSTEM_RETRIES = 200
PLACEMENT_RETRIES = 200
PATCH_PAD = 2


def neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _component_from_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
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
                for nr, nc in neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _translate(coords: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for r, c in coords]


def _reflect_at_bend(
    coords: list[tuple[int, int]],
    bend: tuple[int, int],
    first_dir: tuple[int, int],
) -> list[tuple[int, int]]:
    br, bc = bend
    if first_dir[0] == 0:
        return [(r, 2 * bc - c) for r, c in coords]
    return [(2 * br - r, c) for r, c in coords]


def _frame_border_coords(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    coords: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
        coords.append((r1, c))
    for r in range(r0 + 1, r1):
        coords.append((r, c0))
        coords.append((r, c1))
    return coords


def _is_frame(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    expected = set(_frame_border_coords((r0, c0, r1, c1)))
    return set(coords) == expected


def _pipe_segments(pipe_coords: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    pipe_set = set(pipe_coords)
    deg: dict[tuple[int, int], int] = {}
    endpoints: list[tuple[int, int]] = []
    bends: list[tuple[int, int]] = []
    for cell in pipe_coords:
        nbrs = [n for n in neighbors4(*cell) if n in pipe_set]
        deg[cell] = len(nbrs)
        if len(nbrs) == 1:
            endpoints.append(cell)
        elif len(nbrs) == 2:
            (r1, c1), (r2, c2) = nbrs
            if r1 != r2 and c1 != c2:
                bends.append(cell)
        else:
            raise AssertionError("pipe must be a one-cell-wide chain")
    assert len(endpoints) == 2, "pipe must have exactly two endpoints"
    assert len(bends) == 1, "pipe must have exactly one bend"
    return endpoints[0], endpoints[1], bends[0]


def _direction_toward(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    if dr != 0:
        return (1 if dr > 0 else -1, 0)
    return (0, 1 if dc > 0 else -1)


def _move_to_bend(
    coords: list[tuple[int, int]],
    bend: tuple[int, int],
    direction: tuple[int, int],
) -> list[tuple[int, int]]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    dr, dc = direction
    if dr == 0 and dc == 1:
        return _translate(coords, 0, bend[1] - max(cols))
    if dr == 0 and dc == -1:
        return _translate(coords, 0, bend[1] - min(cols))
    if dr == 1 and dc == 0:
        return _translate(coords, bend[0] - max(rows), 0)
    return _translate(coords, bend[0] - min(rows), 0)


def _move_into_frame(
    coords: list[tuple[int, int]],
    frame_bbox: tuple[int, int, int, int],
    direction: tuple[int, int],
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame_bbox
    ir0, ic0, ir1, ic1 = r0 + 1, c0 + 1, r1 - 1, c1 - 1
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    dr, dc = direction
    if dr == 0 and dc == 1:
        return _translate(coords, 0, ic1 - max(cols))
    if dr == 0 and dc == -1:
        return _translate(coords, 0, ic0 - min(cols))
    if dr == 1 and dc == 0:
        return _translate(coords, ir1 - max(rows), 0)
    return _translate(coords, ir0 - min(rows), 0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide scattered fragments through orange elbow pipes into same-color frames.

    The fragment first translates along the pipe's first leg until its front edge
    reaches the bend, mirrors across the bend row/column, then translates along
    the second leg until its front edge is flush with the frame's far inner wall.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    comps: list[dict] = []
    coord_to_comp: dict[tuple[int, int], int] = {}
    comp_id = 0
    for color in sorted(int(v) for v in np.unique(g) if int(v) not in {BG, PIPE}):
        for coords in _component_from_mask(g == color):
            kind = "frame" if _is_frame(coords) else "fragment"
            comp = {
                "id": comp_id,
                "color": color,
                "coords": coords,
                "coord_set": set(coords),
                "bbox": _bbox(coords),
                "kind": kind,
            }
            for cell in coords:
                coord_to_comp[cell] = comp_id
            comps.append(comp)
            comp_id += 1

    frames_by_color = {comp["color"]: comp for comp in comps if comp["kind"] == "frame"}
    frags_by_color = {comp["color"]: comp for comp in comps if comp["kind"] == "fragment"}
    assert frames_by_color and set(frames_by_color) == set(frags_by_color)

    out = g.copy()
    for frag in frags_by_color.values():
        for r, c in frag["coords"]:
            out[r, c] = BG

    for pipe_coords in _component_from_mask(g == PIPE):
        e1, e2, bend = _pipe_segments(pipe_coords)
        endpoint_info: list[tuple[tuple[int, int], dict]] = []
        for endpoint in (e1, e2):
            attached: set[int] = set()
            for nbr in neighbors4(*endpoint):
                if nbr in coord_to_comp:
                    attached.add(coord_to_comp[nbr])
            assert len(attached) == 1, "each pipe endpoint must touch exactly one colored object"
            endpoint_info.append((endpoint, comps[next(iter(attached))]))

        (p_a, comp_a), (p_b, comp_b) = endpoint_info
        assert comp_a["kind"] != comp_b["kind"], "pipe must link a fragment and a frame"
        assert comp_a["color"] == comp_b["color"], "fragment and frame must share a color"

        if comp_a["kind"] == "fragment":
            frag_endpoint, frame_endpoint = p_a, p_b
            frag_comp, frame_comp = comp_a, comp_b
        else:
            frag_endpoint, frame_endpoint = p_b, p_a
            frag_comp, frame_comp = comp_b, comp_a

        dir1 = _direction_toward(frag_endpoint, bend)
        dir2 = _direction_toward(bend, frame_endpoint)
        moved = _move_to_bend(frag_comp["coords"], bend, dir1)
        turned = _reflect_at_bend(moved, bend, dir1)
        final = _move_into_frame(turned, frame_comp["bbox"], dir2)

        r0, c0, r1, c1 = frame_comp["bbox"]
        ir0, ic0, ir1, ic1 = r0 + 1, c0 + 1, r1 - 1, c1 - 1
        for r, c in final:
            assert ir0 <= r <= ir1 and ic0 <= c <= ic1
            out[r, c] = frag_comp["color"]

    return out


def _normalize(coords: set[tuple[int, int]]) -> list[tuple[int, int]]:
    r0, c0, _, _ = _bbox(list(coords))
    return sorted((r - r0, c - c0) for r, c in coords)


def _random_shape(rng: random.Random) -> list[tuple[int, int]]:
    for _ in range(200):
        area = rng.randint(3, 6)
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for nr, nc in neighbors4(r, c):
                    if (nr, nc) not in cells:
                        frontier.append((nr, nc))
            cells.add(rng.choice(frontier))

        norm = _normalize(cells)
        r0, c0, r1, c1 = _bbox(norm)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h == 1 or w == 1:
            continue
        if h * w == len(norm):
            continue
        if len(set(norm)) != len(norm):
            continue
        return norm
    raise RuntimeError("failed to build a valid fragment shape")


def _edge_cells(coords: list[tuple[int, int]], direction: tuple[int, int]) -> list[tuple[int, int]]:
    if direction == (0, 1):
        edge = max(c for _, c in coords)
        return [(r, c) for r, c in coords if c == edge]
    if direction == (0, -1):
        edge = min(c for _, c in coords)
        return [(r, c) for r, c in coords if c == edge]
    if direction == (1, 0):
        edge = max(r for r, _ in coords)
        return [(r, c) for r, c in coords if r == edge]
    edge = min(r for r, _ in coords)
    return [(r, c) for r, c in coords if r == edge]


def _build_system(
    rng: random.Random,
    color: int,
    case: tuple[tuple[int, int], tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    dir1, dir2 = case
    for _ in range(SYSTEM_RETRIES):
        source0 = _random_shape(rng)
        contact = rng.choice(_edge_cells(source0, dir1))
        source = [(r - contact[0], c - contact[1]) for r, c in source0]

        seg1 = rng.randint(2, 5)
        bend = (dir1[0] * seg1, dir1[1] * seg1)
        moved = _translate(source, bend[0], bend[1])
        turned = _reflect_at_bend(moved, bend, dir1)

        seg2 = rng.randint(2, 5)
        tr0, tc0, tr1, tc1 = _bbox(turned)
        turned_h = tr1 - tr0 + 1
        turned_w = tc1 - tc0 + 1

        if dir2[0] != 0:
            left_margin = rng.randint(0, 2)
            right_margin = rng.randint(0, 2)
            ic0 = tc0 - left_margin
            ic1 = tc1 + right_margin
            if dir2 == (1, 0):
                near_margin = rng.randint(0, 2)
                ir0 = bend[0] + seg2 + 1
                ir1 = ir0 + turned_h + near_margin - 1
                final = _translate(turned, ir1 - tr1, 0)
            else:
                near_margin = rng.randint(0, 2)
                ir1 = bend[0] - seg2 - 1
                ir0 = ir1 - (turned_h + near_margin) + 1
                final = _translate(turned, ir0 - tr0, 0)
        else:
            top_margin = rng.randint(0, 2)
            bottom_margin = rng.randint(0, 2)
            ir0 = tr0 - top_margin
            ir1 = tr1 + bottom_margin
            if dir2 == (0, 1):
                near_margin = rng.randint(0, 2)
                ic0 = bend[1] + seg2 + 1
                ic1 = ic0 + turned_w + near_margin - 1
                final = _translate(turned, 0, ic1 - tc1)
            else:
                near_margin = rng.randint(0, 2)
                ic1 = bend[1] - seg2 - 1
                ic0 = ic1 - (turned_w + near_margin) + 1
                final = _translate(turned, 0, ic0 - tc0)

        fr0, fc0, fr1, fc1 = _bbox(final)
        frame_outer = (ir0 - 1, ic0 - 1, ir1 + 1, ic1 + 1)
        frame_coords = _frame_border_coords(frame_outer)

        if not all(ir0 <= r <= ir1 and ic0 <= c <= ic1 for r, c in final):
            continue
        if dir2 == (1, 0) and fr1 != ir1:
            continue
        if dir2 == (-1, 0) and fr0 != ir0:
            continue
        if dir2 == (0, 1) and fc1 != ic1:
            continue
        if dir2 == (0, -1) and fc0 != ic0:
            continue

        pipe_coords = []
        for k in range(1, seg1 + 1):
            pipe_coords.append((dir1[0] * k, dir1[1] * k))
        for k in range(seg2):
            pipe_coords.append((bend[0] + dir2[0] * k, bend[1] + dir2[1] * k))
        pipe_coords = sorted(set(pipe_coords))

        source_set = set(source)
        frame_set = set(frame_coords)
        pipe_set = set(pipe_coords)
        final_set = set(final)

        if source_set & pipe_set:
            continue
        if source_set & frame_set:
            continue
        if pipe_set & frame_set:
            continue
        if final_set & pipe_set:
            continue
        if final_set & frame_set:
            continue

        all_input = source_set | pipe_set | frame_set
        min_r = min(r for r, _ in all_input)
        min_c = min(c for _, c in all_input)
        max_r = max(r for r, _ in all_input)
        max_c = max(c for _, c in all_input)
        shift_r = PATCH_PAD - min_r
        shift_c = PATCH_PAD - min_c
        patch_h = max_r + shift_r + PATCH_PAD + 1
        patch_w = max_c + shift_c + PATCH_PAD + 1

        inp = np.full((patch_h, patch_w), BG, dtype=int)
        out = np.full((patch_h, patch_w), BG, dtype=int)

        for r, c in frame_set:
            inp[r + shift_r, c + shift_c] = color
            out[r + shift_r, c + shift_c] = color
        for r, c in pipe_set:
            inp[r + shift_r, c + shift_c] = PIPE
            out[r + shift_r, c + shift_c] = PIPE
        for r, c in source_set:
            inp[r + shift_r, c + shift_c] = color
        for r, c in final_set:
            out[r + shift_r, c + shift_c] = color

        return inp, out

    raise RuntimeError("failed to build a valid system")


def _place_patches(
    rng: random.Random,
    patches: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray] | None:
    heights = [inp.shape[0] for inp, _ in patches]
    widths = [inp.shape[1] for inp, _ in patches]
    min_h = max(max(heights) + 4, sum(heights) // 2 + 6)
    min_w = max(max(widths) + 4, sum(widths) // 2 + 6)
    max_h = max(min_h + 6, sum(heights) + 8)
    max_w = max(min_w + 6, sum(widths) + 8)
    if min_h > 30 or min_w > 30:
        return None
    h = rng.randint(min_h, min(max_h, 30))
    w = rng.randint(min_w, min(max_w, 30))

    grid = np.full((h, w), BG, dtype=int)
    expected = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    order = list(range(len(patches)))
    rng.shuffle(order)
    for idx in order:
        inp, out = patches[idx]
        ph, pw = inp.shape
        if ph + 2 > h or pw + 2 > w:
            return None
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            r0 = rng.randint(1, h - ph - 1)
            c0 = rng.randint(1, w - pw - 1)
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(h, r0 + ph + 1)
            cc1 = min(w, c0 + pw + 1)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            occupied[rr0:rr1, cc0:cc1] = True
            grid[r0 : r0 + ph, c0 : c0 + pw] = inp
            expected[r0 : r0 + ph, c0 : c0 + pw] = out
            placed = True
            break
        if not placed:
            return None
    return grid, expected


def _count_system_colors(grid: np.ndarray) -> int:
    colors = {int(v) for v in np.unique(grid) if int(v) not in {BG, PIPE}}
    return len(colors)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_systems = rng.randint(1, 3)
    colors = rng.sample(SYSTEM_COLORS, n_systems)
    cases = rng.sample(CASES, n_systems) if n_systems <= len(CASES) else [rng.choice(CASES) for _ in range(n_systems)]
    patches = [_build_system(rng, color, case) for color, case in zip(colors, cases)]
    placed = _place_patches(rng, patches)
    if placed is None:
        return None
    grid, expected = placed
    if _count_system_colors(grid) != n_systems:
        return None
    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
