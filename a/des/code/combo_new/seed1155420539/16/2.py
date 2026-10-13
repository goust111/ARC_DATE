import random

import numpy as np


BG = 8
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
COLOR_POOL = [0, 1, 2, 3, 4, 5, 6, 7, 9]
PAIR_COUNTS = (2, 3, 4)
GEN_RETRIES = 400
PLACEMENT_RETRIES = 80


def _find_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append({"color": color, "coords": np.asarray(coords, dtype=int)})
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _normalize(coords: np.ndarray | tuple[tuple[int, int], ...]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    if isinstance(coords, np.ndarray):
        r0, c0, r1, c1 = _bbox(coords)
        rel = tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))
        return rel, r1 - r0 + 1, c1 - c0 + 1

    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    rel = tuple(sorted((r - r0, c - c0) for r, c in coords))
    return rel, r1 - r0 + 1, c1 - c0 + 1


def _rotate_once(coords: tuple[tuple[int, int], ...], h: int, w: int) -> tuple[tuple[tuple[int, int], ...], int, int]:
    rotated = tuple(sorted((c, h - 1 - r) for r, c in coords))
    return rotated, w, h


def _rotate_n(coords: tuple[tuple[int, int], ...], h: int, w: int, n: int) -> tuple[tuple[tuple[int, int], ...], int, int]:
    out_coords, out_h, out_w = coords, h, w
    for _ in range(n % 4):
        out_coords, out_h, out_w = _rotate_once(out_coords, out_h, out_w)
    return out_coords, out_h, out_w


def _canonical_signature(coords: tuple[tuple[int, int], ...], h: int, w: int) -> tuple:
    variants = []
    cur_coords, cur_h, cur_w = coords, h, w
    for _ in range(4):
        variants.append((cur_h, cur_w, cur_coords))
        cur_coords, cur_h, cur_w = _rotate_once(cur_coords, cur_h, cur_w)
    return min(variants)


def _complement(coords: tuple[tuple[int, int], ...], h: int, w: int) -> tuple[tuple[int, int], ...]:
    cells = set(coords)
    return tuple(sorted((r, c) for r in range(h) for c in range(w) if (r, c) not in cells))


def _is_connected(coords: tuple[tuple[int, int], ...]) -> bool:
    if not coords:
        return False
    cells = set(coords)
    stack = [next(iter(cells))]
    seen = {stack[0]}
    while stack:
        r, c = stack.pop()
        for dr, dc in DIRS4:
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(cells)


def _touches_all_sides(coords: tuple[tuple[int, int], ...], h: int, w: int) -> bool:
    rows = {r for r, _ in coords}
    cols = {c for _, c in coords}
    return 0 in rows and (h - 1) in rows and 0 in cols and (w - 1) in cols


def _build_shape_catalog() -> dict[tuple, dict]:
    catalog: dict[tuple, dict] = {}
    for h in range(1, 7):
        for w in range(1, 7):
            if h * w > 12:
                continue
            cells = [(r, c) for r in range(h) for c in range(w)]
            count = len(cells)
            for mask in range(1, 1 << count):
                if mask.bit_count() > 6:
                    continue
                coords = tuple(cells[i] for i in range(count) if (mask >> i) & 1)
                if not _touches_all_sides(coords, h, w) or not _is_connected(coords):
                    continue

                sig = _canonical_signature(coords, h, w)
                if sig in catalog:
                    continue

                comp = _complement(coords, h, w)
                target_sig = None
                if comp and _is_connected(comp):
                    comp_rel, comp_h, comp_w = _normalize(comp)
                    target_sig = _canonical_signature(comp_rel, comp_h, comp_w)

                catalog[sig] = {
                    "coords": sig[2],
                    "h": sig[0],
                    "w": sig[1],
                    "target_sig": target_sig,
                }
    return catalog


def _build_pair_catalog() -> tuple[list[dict], dict[tuple, set[tuple]]]:
    shapes = _build_shape_catalog()
    adjacency: dict[tuple, set[tuple]] = {sig: set() for sig in shapes}
    pair_map: dict[tuple[tuple, tuple], dict] = {}

    for sig, info in shapes.items():
        target = info["target_sig"]
        if target is None or target not in shapes or target == sig:
            continue
        edge = tuple(sorted((sig, target)))
        if edge not in pair_map:
            pair_map[edge] = {
                "sig_a": edge[0],
                "sig_b": edge[1],
                "a": shapes[edge[0]],
                "b": shapes[edge[1]],
            }
        adjacency[sig].add(target)
        adjacency[target].add(sig)

    pairs = sorted(pair_map.values(), key=lambda item: (item["sig_a"], item["sig_b"]))
    return pairs, adjacency


PAIR_TEMPLATES, SHAPE_ADJ = _build_pair_catalog()
assert PAIR_TEMPLATES, "expected at least one compatible complement pair"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair scattered fragments by complementary negative space and swap their colors.

    Input structure:
    - The background is azure.
    - The non-background pixels form an even number of 4-connected monochrome fragments, each with a distinct color and at most six cells.
    - Every fragment is interpreted inside its own minimal bounding box, whose empty cells define that fragment's complement shape.

    Rule:
    - For each fragment, compute its minimal bounding box and extract the cells of that box not occupied by the fragment.
    - Match fragments into unique complementary pairs: one fragment's positive shape must exactly realize the other fragment's bounding-box hole, up to rotation.
    - Keep every fragment fixed in place, but recolor it with its partner fragment's original color.

    Key insight:
    - Fragments are not matched by looking alike; each one is identified by the negative space left inside its own bounding box.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    comps = _find_components(g)
    assert comps and len(comps) % 2 == 0

    colors = [int(comp["color"]) for comp in comps]
    assert len(colors) == len(set(colors))

    records = []
    for comp in comps:
        rel, h, w = _normalize(comp["coords"])
        assert len(rel) <= 6
        target_sig = None
        comp_rel = _complement(rel, h, w)
        if comp_rel and _is_connected(comp_rel):
            hole_rel, hole_h, hole_w = _normalize(comp_rel)
            target_sig = _canonical_signature(hole_rel, hole_h, hole_w)
        records.append(
            {
                "coords": comp["coords"],
                "color": int(comp["color"]),
                "shape_sig": _canonical_signature(rel, h, w),
                "target_sig": target_sig,
            }
        )

    adjacency = {idx: set() for idx in range(len(records))}
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            if records[i]["target_sig"] == records[j]["shape_sig"] or records[j]["target_sig"] == records[i]["shape_sig"]:
                adjacency[i].add(j)
                adjacency[j].add(i)

    assert all(len(neighbors) == 1 for neighbors in adjacency.values())

    out = g.copy()
    for idx, record in enumerate(records):
        partner = next(iter(adjacency[idx]))
        coords = record["coords"]
        out[coords[:, 0], coords[:, 1]] = records[partner]["color"]
    return out


def _sample_pair_templates(pair_count: int) -> list[dict] | None:
    order = PAIR_TEMPLATES[:]
    random.shuffle(order)
    chosen: list[dict] = []
    chosen_nodes: set[tuple] = set()

    def backtrack(pos: int) -> bool:
        if len(chosen) == pair_count:
            return True
        if pos == len(order):
            return False

        for idx in range(pos, len(order)):
            pair = order[idx]
            a_sig = pair["sig_a"]
            b_sig = pair["sig_b"]
            new_nodes = {a_sig, b_sig}
            if new_nodes & chosen_nodes:
                continue
            if any(node in SHAPE_ADJ.get(a_sig, set()) or node in SHAPE_ADJ.get(b_sig, set()) for node in chosen_nodes):
                continue

            chosen.append(pair)
            chosen_nodes.update(new_nodes)
            if backtrack(idx + 1):
                return True
            chosen.pop()
            chosen_nodes.difference_update(new_nodes)

        return False

    if backtrack(0):
        return chosen[:]
    return None


def _make_object(shape: dict, color: int) -> dict:
    rot = random.randrange(4)
    coords, out_h, out_w = _rotate_n(shape["coords"], shape["h"], shape["w"], rot)
    return {"coords": coords, "h": out_h, "w": out_w, "color": color}


def _candidate_axis_positions(limit: int, span: int) -> list[int]:
    slack = limit - span
    if slack < 0:
        return []
    if slack >= 2:
        return list(range(1, limit - span))
    return list(range(0, limit - span + 1))


def _can_place(forbidden: np.ndarray, coords: tuple[tuple[int, int], ...], top: int, left: int) -> bool:
    for dr, dc in coords:
        if forbidden[top + dr, left + dc]:
            return False
    return True


def _mark_forbidden(forbidden: np.ndarray, coords: tuple[tuple[int, int], ...], top: int, left: int) -> None:
    h, w = forbidden.shape
    for dr, dc in coords:
        r, c = top + dr, left + dc
        for nr in range(max(0, r - 1), min(h, r + 2)):
            for nc in range(max(0, c - 1), min(w, c + 2)):
                forbidden[nr, nc] = True


def _place_objects(objects: list[dict], rows: int, cols: int) -> np.ndarray | None:
    grid = np.full((rows, cols), BG, dtype=int)
    forbidden = np.zeros((rows, cols), dtype=bool)

    for obj in sorted(objects, key=lambda item: (item["h"] * item["w"], len(item["coords"])), reverse=True):
        row_positions = _candidate_axis_positions(rows, obj["h"])
        col_positions = _candidate_axis_positions(cols, obj["w"])
        positions = [(r, c) for r in row_positions for c in col_positions]
        random.shuffle(positions)

        placed = False
        for top, left in positions:
            if not _can_place(forbidden, obj["coords"], top, left):
                continue
            for dr, dc in obj["coords"]:
                grid[top + dr, left + dc] = obj["color"]
            _mark_forbidden(forbidden, obj["coords"], top, left)
            placed = True
            break
        if not placed:
            return None

    return grid


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choice(PAIR_COUNTS)
    pairs = _sample_pair_templates(pair_count)
    if pairs is None:
        return None

    colors = random.sample(COLOR_POOL, 2 * pair_count)
    objects: list[dict] = []
    for idx, pair in enumerate(pairs):
        objects.append(_make_object(pair["a"], colors[2 * idx]))
        objects.append(_make_object(pair["b"], colors[2 * idx + 1]))
    random.shuffle(objects)

    needed_area = sum((obj["h"] + 2) * (obj["w"] + 2) for obj in objects)
    min_side = max(max(obj["h"] for obj in objects) + 2, max(obj["w"] for obj in objects) + 2, 12)
    base_side = max(min_side, int(np.ceil(np.sqrt(needed_area))))

    for _ in range(PLACEMENT_RETRIES):
        rows = random.randint(base_side, min(30, base_side + 8))
        cols = random.randint(base_side, min(30, base_side + 8))
        if rows * cols < needed_area + 12:
            continue

        grid = _place_objects(objects, rows, cols)
        if grid is None:
            continue

        comps = _find_components(grid)
        if len(comps) != 2 * pair_count:
            continue

        try:
            out = main(grid)
        except AssertionError:
            continue
        if np.array_equal(out, grid):
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid complementary-fragment puzzle")
