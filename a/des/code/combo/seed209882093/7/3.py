import random
from collections import deque

import numpy as np


BG = 0
BODY_COLOR = 8

HFLIP_COLOR = 9
VFLIP_COLOR = 7
ROT90_COLOR = 6

OP_BY_COLOR = {
    HFLIP_COLOR: "hflip",
    VFLIP_COLOR: "vflip",
    ROT90_COLOR: "rot90",
}
COLOR_BY_OP = {op: color for color, op in OP_BY_COLOR.items()}

SEED_COLORS = (1, 2, 3, 4, 5)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return 8-connected components of a boolean mask."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for dr, dc in DIR8:
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _parse_panel_candidate(
    grid: np.ndarray,
    side: str,
) -> tuple[dict[int, str], np.ndarray] | None:
    """Parse a 2-cell-thick edge palette candidate."""
    h, w = grid.shape
    panel_mask = np.zeros((h, w), dtype=bool)
    mapping: dict[int, str] = {}

    if side in ("left", "right") and w < 2:
        return None
    if side in ("top", "bottom") and h < 2:
        return None

    if side == "left":
        positions = [((r, 0), (r, 1)) for r in range(h)]
    elif side == "right":
        positions = [((r, w - 2), (r, w - 1)) for r in range(h)]
    elif side == "top":
        positions = [((0, c), (1, c)) for c in range(w)]
    elif side == "bottom":
        positions = [((h - 2, c), (h - 1, c)) for c in range(w)]
    else:
        raise AssertionError(side)

    for (sr, sc), (tr, tc) in positions:
        seed_color = int(grid[sr, sc])
        op_color = int(grid[tr, tc])
        if seed_color == BG and op_color == BG:
            continue
        if seed_color == BG or op_color == BG:
            return None
        if op_color not in OP_BY_COLOR:
            return None
        if seed_color in OP_BY_COLOR:
            return None
        if seed_color in mapping:
            return None
        mapping[seed_color] = OP_BY_COLOR[op_color]
        panel_mask[sr, sc] = True
        panel_mask[tr, tc] = True

    if not mapping:
        return None
    return mapping, panel_mask


def _candidate_matches_scene(
    grid: np.ndarray,
    mapping: dict[int, str],
    panel_mask: np.ndarray,
) -> bool:
    """A valid palette must explain every non-panel object exactly once."""
    work = np.array(grid, dtype=int)
    work[panel_mask] = BG
    comps = _connected_components(work != BG)
    if len(comps) != len(mapping):
        return False

    palette_seeds = set(mapping)
    used: set[int] = set()
    for comp in comps:
        colors = {int(work[r, c]) for r, c in comp}
        seed_hits = colors & palette_seeds
        if len(seed_hits) != 1:
            return False
        seed_color = next(iter(seed_hits))
        if seed_color in used:
            return False
        if sum(1 for r, c in comp if int(work[r, c]) == seed_color) != 1:
            return False
        if len(comp) < 2:
            return False
        used.add(seed_color)
    return used == palette_seeds


def _parse_panel(grid: np.ndarray) -> tuple[dict[int, str], np.ndarray]:
    candidates: list[tuple[int, str, dict[int, str], np.ndarray]] = []
    for side in ("left", "right", "top", "bottom"):
        parsed = _parse_panel_candidate(grid, side)
        if parsed is None:
            continue
        mapping, panel_mask = parsed
        if _candidate_matches_scene(grid, mapping, panel_mask):
            candidates.append((len(mapping), side, mapping, panel_mask))

    assert candidates, "expected exactly one valid edge palette"
    candidates.sort(key=lambda item: item[0], reverse=True)
    _count, _side, mapping, panel_mask = candidates[0]
    return mapping, panel_mask


def _transform_offset(dr: int, dc: int, op: str) -> tuple[int, int]:
    if op == "hflip":
        return int(dr), int(-dc)
    if op == "vflip":
        return int(-dr), int(dc)
    if op == "rot90":
        return int(dc), int(-dr)
    raise AssertionError(op)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transform each seeded object according to the palette on the edge.

    - maroon (9): horizontal flip about the seed column
    - orange (7): vertical flip about the seed row
    - magenta (6): 90-degree clockwise rotation about the seed pixel
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    mapping, panel_mask = _parse_panel(g)
    work = g.copy()
    work[panel_mask] = BG

    out = np.full((h, w), BG, dtype=int)
    seed_palette = set(mapping)

    for comp in _connected_components(work != BG):
        seed_positions = [(r, c) for r, c in comp if int(work[r, c]) in seed_palette]
        assert len(seed_positions) == 1, "each object must contain exactly one seed pixel"
        sr, sc = seed_positions[0]
        seed_color = int(work[sr, sc])
        out[sr, sc] = seed_color
        op = mapping[seed_color]

        for r, c in comp:
            if (r, c) == (sr, sc):
                continue
            dr = int(r - sr)
            dc = int(c - sc)
            tdr, tdc = _transform_offset(dr, dc, op)
            nr = int(sr + tdr)
            nc = int(sc + tdc)
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = int(work[r, c])

    return out


# --------------- generator ---------------

GRID_H_RANGE = (14, 22)
GRID_W_RANGE = (14, 22)
OBJECT_COUNT_RANGE = (2, 5)
LOCAL_SIZE = 7
LOCAL_CENTER = LOCAL_SIZE // 2
OCCUPIED_COUNT_RANGE = (6, 10)  # includes the seed cell
SHAPE_ATTEMPTS = 200
GEN_ATTEMPTS = 400
CLIP_PROB = 0.35


def _panel_area_mask(h: int, w: int, side: str) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    if side == "left":
        mask[:, :2] = True
    elif side == "right":
        mask[:, w - 2 :] = True
    else:
        raise AssertionError(side)
    return mask


def _transform_body_offsets(body_offsets: list[tuple[int, int]], op: str) -> list[tuple[int, int]]:
    return [_transform_offset(int(dr), int(dc), op) for dr, dc in body_offsets]


def _shape_is_good(body_offsets: list[tuple[int, int]]) -> bool:
    rel = set((int(dr), int(dc)) for dr, dc in body_offsets)
    if len(rel) < 5:
        return False

    center_neighbors = sum((dr, dc) in rel for dr, dc in DIR4)
    if center_neighbors < 2:
        return False

    rows = [0] + [int(dr) for dr, _dc in rel]
    cols = [0] + [int(dc) for _dr, dc in rel]
    if min(rows) >= 0 or max(rows) <= 0:
        return False
    if min(cols) >= 0 or max(cols) <= 0:
        return False

    for op in ("hflip", "vflip", "rot90"):
        if rel == set(_transform_body_offsets(list(rel), op)):
            return False
    return True


def _generate_body_offsets() -> list[tuple[int, int]] | None:
    for _ in range(SHAPE_ATTEMPTS):
        target = random.randint(*OCCUPIED_COUNT_RANGE)
        occ = {(LOCAL_CENTER, LOCAL_CENTER)}

        while len(occ) < target:
            frontier = set()
            for r, c in occ:
                for dr, dc in DIR4:
                    nr = int(r + dr)
                    nc = int(c + dc)
                    if 0 <= nr < LOCAL_SIZE and 0 <= nc < LOCAL_SIZE and (nr, nc) not in occ:
                        frontier.add((nr, nc))
            if not frontier:
                break
            occ.add(random.choice(tuple(frontier)))

        if len(occ) != target:
            continue

        body_offsets = sorted((int(r - LOCAL_CENTER), int(c - LOCAL_CENTER)) for r, c in occ if (r, c) != (LOCAL_CENTER, LOCAL_CENTER))
        if _shape_is_good(body_offsets):
            return body_offsets
    return None


def _can_place(mask: np.ndarray, coords: list[tuple[int, int]], gap: int = 1) -> bool:
    h, w = mask.shape
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w):
            return False
        r0 = max(0, int(r - gap))
        r1 = min(h, int(r + gap + 1))
        c0 = max(0, int(c - gap))
        c1 = min(w, int(c + gap + 1))
        if bool(mask[r0:r1, c0:c1].any()):
            return False
    return True


def _mark(mask: np.ndarray, coords: list[tuple[int, int]]) -> None:
    for r, c in coords:
        mask[int(r), int(c)] = True


def _sample_ops(count: int) -> list[str]:
    ops = [random.choice(["hflip", "vflip", "rot90"]) for _ in range(count)]
    if count > 1 and len(set(ops)) == 1:
        idx = random.randrange(count)
        alternatives = [op for op in ("hflip", "vflip", "rot90") if op != ops[idx]]
        ops[idx] = random.choice(alternatives)
    if count >= 3 and random.random() < 0.6:
        forced = ["hflip", "vflip", "rot90"]
        random.shuffle(forced)
        for i, op in enumerate(forced):
            ops[i] = op
        random.shuffle(ops)
    return ops


def _candidate_positions(h: int, w: int) -> list[tuple[int, int]]:
    coords = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(coords)
    return coords


def _place_objects(
    h: int,
    w: int,
    panel_side: str,
    specs: list[dict[str, object]],
) -> list[dict[str, object]] | None:
    panel_mask = _panel_area_mask(h, w, panel_side)
    input_mask = panel_mask.copy()
    output_mask = panel_mask.copy()
    placed: list[dict[str, object]] = []

    order = sorted(range(len(specs)), key=lambda i: len(specs[i]["body_offsets"]), reverse=True)
    seed_positions = _candidate_positions(h, w)

    for spec_index in order:
        spec = specs[spec_index]
        body_offsets = list(spec["body_offsets"])
        op = str(spec["op"])
        transformed_offsets = _transform_body_offsets(body_offsets, op)
        want_clip = bool(spec["want_clip"])

        preferred = []
        fallback = []
        for sr, sc in seed_positions:
            input_body = []
            input_ok = True
            for dr, dc in body_offsets:
                r = int(sr + dr)
                c = int(sc + dc)
                if not (0 <= r < h and 0 <= c < w) or bool(panel_mask[r, c]):
                    input_ok = False
                    break
                input_body.append((r, c))
            if not input_ok or bool(panel_mask[sr, sc]):
                continue

            output_body = []
            clipped = 0
            output_ok = True
            for dr, dc in transformed_offsets:
                r = int(sr + dr)
                c = int(sc + dc)
                if not (0 <= r < h and 0 <= c < w):
                    clipped += 1
                    continue
                if bool(panel_mask[r, c]):
                    output_ok = False
                    break
                output_body.append((r, c))
            if not output_ok:
                continue
            if len(output_body) < 3:
                continue
            if set(output_body) == set(input_body):
                continue

            input_coords = [(int(sr), int(sc))] + input_body
            output_coords = [(int(sr), int(sc))] + output_body
            if not _can_place(input_mask, input_coords, gap=1):
                continue
            if not _can_place(output_mask, output_coords, gap=1):
                continue

            record = {
                "seed": (int(sr), int(sc)),
                "input_body": input_body,
                "output_body": output_body,
                "clipped": int(clipped),
            }
            fallback.append(record)
            if (want_clip and clipped > 0) or ((not want_clip) and clipped == 0):
                preferred.append(record)

        options = preferred if preferred else fallback
        if not options:
            return None

        chosen = random.choice(options)
        _mark(input_mask, [(int(r), int(c)) for r, c in [chosen["seed"]] + chosen["input_body"]])
        _mark(output_mask, [(int(r), int(c)) for r, c in [chosen["seed"]] + chosen["output_body"]])

        merged = dict(spec)
        merged.update(chosen)
        placed.append(merged)

    return placed


def _render_input(
    h: int,
    w: int,
    panel_side: str,
    panel_rows: list[int],
    objects: list[dict[str, object]],
) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)

    palette_items = objects[:]
    random.shuffle(palette_items)
    for row, obj in zip(panel_rows, palette_items):
        seed_color = int(obj["seed_color"])
        op_color = COLOR_BY_OP[str(obj["op"])]
        if panel_side == "left":
            grid[int(row), 0] = seed_color
            grid[int(row), 1] = op_color
        elif panel_side == "right":
            grid[int(row), w - 2] = seed_color
            grid[int(row), w - 1] = op_color
        else:
            raise AssertionError(panel_side)

    for obj in objects:
        sr, sc = obj["seed"]
        grid[int(sr), int(sc)] = int(obj["seed_color"])
        for r, c in obj["input_body"]:
            grid[int(r), int(c)] = BODY_COLOR

    return grid


def _render_output(h: int, w: int, objects: list[dict[str, object]]) -> np.ndarray:
    out = np.full((h, w), BG, dtype=int)
    for obj in objects:
        sr, sc = obj["seed"]
        out[int(sr), int(sc)] = int(obj["seed_color"])
        for r, c in obj["output_body"]:
            out[int(r), int(c)] = BODY_COLOR
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)

    max_objects = min(OBJECT_COUNT_RANGE[1], len(SEED_COLORS), max(OBJECT_COUNT_RANGE[0], h // 3))
    if max_objects < OBJECT_COUNT_RANGE[0]:
        return None
    obj_count = random.randint(OBJECT_COUNT_RANGE[0], max_objects)

    panel_side = random.choice(["left", "right"])
    usable_rows = list(range(1, h - 1))
    if len(usable_rows) < obj_count:
        return None
    panel_rows = sorted(random.sample(usable_rows, obj_count))

    seed_colors = random.sample(SEED_COLORS, obj_count)
    ops = _sample_ops(obj_count)
    specs = []
    for seed_color, op in zip(seed_colors, ops):
        body_offsets = _generate_body_offsets()
        if body_offsets is None:
            return None
        specs.append(
            {
                "seed_color": int(seed_color),
                "op": op,
                "body_offsets": body_offsets,
                "want_clip": random.random() < CLIP_PROB,
            }
        )

    placed = _place_objects(h, w, panel_side, specs)
    if placed is None:
        return None

    inp = _render_input(h, w, panel_side, panel_rows, placed)
    expected = _render_output(h, w, placed)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid seeded-transform ARC input")
