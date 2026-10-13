import random
from collections import deque

import numpy as np


DIRECTIONS = ("U", "R", "D", "L")
CCW_ORDER = ("U", "L", "D", "R")
ALL_COLORS = tuple(range(10))

# Generator settings.
PAIR_COUNT_RANGE = (2, 4)
GAP_RANGE = (2, 5)
MAX_ATTEMPTS = 500
PAIR_PLACEMENT_ATTEMPTS = 250
MAX_FOREGROUND_PCT = 35


def get_bg(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(values, minlength=10).argmax())


def axis_symmetric(mask: np.ndarray) -> bool:
    return bool(np.array_equal(mask, np.flipud(mask)) or np.array_equal(mask, np.fliplr(mask)))


def find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0] or int(g[r0, c0]) == bg:
                continue
            queue = deque([(r0, c0)])
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            colors = set()

            while queue:
                r, c = queue.popleft()
                coords.append((r, c))
                colors.add(int(g[r, c]))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) == bg:
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))

            assert len(colors) == 1
            rows = [r for r, _ in coords]
            cols = [c for _, c in coords]
            components.append(
                {
                    "coords": coords,
                    "color": next(iter(colors)),
                    "area": len(coords),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )

    return components


def component_mask(component: dict) -> np.ndarray:
    r0, c0, r1, c1 = map(int, component["bbox"])
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in component["coords"]:
        mask[int(r) - r0, int(c) - c0] = True
    return mask


def opening_info(mask: np.ndarray) -> tuple[str, tuple[int, int], dict[str, int]]:
    h, w = mask.shape
    assert mask.ndim == 2 and mask.any()
    assert 1 < h <= 3 and 1 < w <= 3
    assert not axis_symmetric(mask)

    side_cells = {
        "U": [(0, c) for c in range(w) if mask[0, c]],
        "D": [(h - 1, c) for c in range(w) if mask[h - 1, c]],
        "L": [(r, 0) for r in range(h) if mask[r, 0]],
        "R": [(r, w - 1) for r in range(h) if mask[r, w - 1]],
    }
    counts = {side: len(coords) for side, coords in side_cells.items()}
    min_count = min(counts.values())
    opening_sides = [side for side, count in counts.items() if count == min_count]
    assert len(opening_sides) == 1
    opening = opening_sides[0]
    anchor_candidates = side_cells[opening]
    assert anchor_candidates

    if opening in {"U", "D"}:
        mid = (w - 1) / 2.0
        anchor = min(anchor_candidates, key=lambda pos: (abs(pos[1] - mid), pos[1]))
    else:
        mid = (h - 1) / 2.0
        anchor = min(anchor_candidates, key=lambda pos: (abs(pos[0] - mid), pos[0]))

    return opening, anchor, counts


def rotate_side_ccw(side: str, k: int) -> str:
    return CCW_ORDER[(CCW_ORDER.index(side) + k) % 4]


def rotation_k(src_side: str, dst_side: str) -> int:
    for k in range(4):
        if rotate_side_ccw(src_side, k) == dst_side:
            return k
    raise AssertionError("unreachable")


def absolute_coords(mask: np.ndarray, top_left: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = top_left
    return {
        (r0 + int(r), c0 + int(c))
        for r, c in np.argwhere(mask)
    }


def translate_mask_to_anchor(
    mask: np.ndarray,
    anchor: tuple[int, int],
    target: tuple[int, int],
) -> set[tuple[int, int]]:
    tr, tc = target
    ar, ac = anchor
    return absolute_coords(mask, (tr - ar, tc - ac))


def corridor_cells(
    bbox: tuple[int, int, int, int],
    pad: tuple[int, int],
    direction: str,
) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    pr, pc = pad
    if direction == "U":
        return {(r, pc) for r in range(pr + 1, r0)}
    if direction == "D":
        return {(r, pc) for r in range(r1 + 1, pr)}
    if direction == "L":
        return {(pr, c) for c in range(pc + 1, c0)}
    if direction == "R":
        return {(pr, c) for c in range(c1 + 1, pc)}
    raise AssertionError("bad direction")


def detect_pad_candidates(
    grid: np.ndarray,
    bg: int,
    stamp: dict,
    pads: list[dict],
) -> list[tuple[str, dict]]:
    r0, c0, r1, c1 = map(int, stamp["bbox"])
    hits: list[tuple[str, dict]] = []

    for pad in pads:
        pr, pc = map(int, pad["coords"][0])

        if pr < r0 and c0 <= pc <= c1 and np.all(grid[pr + 1 : r0, pc] == bg):
            hits.append(("U", pad))
        if pr > r1 and c0 <= pc <= c1 and np.all(grid[r1 + 1 : pr, pc] == bg):
            hits.append(("D", pad))
        if pc < c0 and r0 <= pr <= r1 and np.all(grid[pr, pc + 1 : c0] == bg):
            hits.append(("L", pad))
        if pc > c1 and r0 <= pr <= r1 and np.all(grid[pr, c1 + 1 : pc] == bg):
            hits.append(("R", pad))

    return hits


def main(grid: np.ndarray) -> np.ndarray:
    """Match each stamp with its pad, rotate the stamp toward the pad, and stamp it in the pad's color."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = get_bg(g)
    components = find_components(g, bg)

    stamps = [comp for comp in components if int(comp["area"]) > 1]
    pads = [comp for comp in components if int(comp["area"]) == 1]
    assert 2 <= len(stamps) <= 4
    assert len(stamps) == len(pads)

    pairings = []
    used_pads: set[tuple[int, int]] = set()

    for stamp in stamps:
        mask = component_mask(stamp)
        opening_side, _, _ = opening_info(mask)
        candidates = detect_pad_candidates(g, bg, stamp, pads)
        assert len(candidates) == 1
        pad_dir, pad = candidates[0]
        pad_coord = tuple(map(int, pad["coords"][0]))
        assert pad_coord not in used_pads
        used_pads.add(pad_coord)
        pairings.append((stamp, mask, opening_side, pad_dir, pad_coord, int(pad["color"])))

    assert len(used_pads) == len(pads)

    out = g.copy()
    h, w = g.shape
    for stamp, mask, opening_side, pad_dir, pad_coord, pad_color in pairings:
        assert pad_color != int(stamp["color"])
        k = rotation_k(opening_side, pad_dir)
        rotated = np.rot90(mask, k)
        rotated_opening, anchor, _ = opening_info(rotated)
        assert rotated_opening == pad_dir
        stamped_coords = translate_mask_to_anchor(rotated, anchor, pad_coord)

        for rr, cc in stamped_coords:
            assert 0 <= rr < h and 0 <= cc < w
            if (rr, cc) != pad_coord:
                assert int(out[rr, cc]) == bg
            out[rr, cc] = pad_color

    return out


def in_bounds(coord: tuple[int, int], shape: tuple[int, int]) -> bool:
    r, c = coord
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def expand_cells(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _connected(mask: np.ndarray) -> bool:
    coords = [tuple(map(int, pos)) for pos in np.argwhere(mask)]
    if not coords:
        return False
    queue = deque([coords[0]])
    seen = {coords[0]}
    h, w = mask.shape
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if not mask[nr, nc] or (nr, nc) in seen:
                continue
            seen.add((nr, nc))
            queue.append((nr, nc))
    return len(seen) == len(coords)


def _enumerate_templates() -> list[dict]:
    templates: list[dict] = []
    for h in (2, 3):
        for w in (2, 3):
            total = h * w
            for bits in range(1 << total):
                mask = np.array([(bits >> i) & 1 for i in range(total)], dtype=bool).reshape(h, w)
                area = int(mask.sum())
                if not (4 <= area <= 7):
                    continue
                if not _connected(mask):
                    continue
                if not mask[0, :].any() or not mask[h - 1, :].any():
                    continue
                if not mask[:, 0].any() or not mask[:, w - 1].any():
                    continue
                if axis_symmetric(mask):
                    continue
                try:
                    opening_side, anchor, counts = opening_info(mask)
                except AssertionError:
                    continue
                if counts[opening_side] != 1:
                    continue
                templates.append(
                    {
                        "mask": mask.copy(),
                        "opening": opening_side,
                        "anchor": anchor,
                    }
                )

    assert templates
    return templates


TEMPLATES = _enumerate_templates()
TEMPLATES_BY_OPENING = {
    side: [template for template in TEMPLATES if template["opening"] == side]
    for side in DIRECTIONS
}


def _candidate_pair(
    shape: tuple[int, int],
    blocked_input: set[tuple[int, int]],
    occupied_input: set[tuple[int, int]],
    reserved_output: set[tuple[int, int]],
) -> dict | None:
    h, w = shape
    opening_side = random.choice(DIRECTIONS)
    template = random.choice(TEMPLATES_BY_OPENING[opening_side])
    mask = template["mask"]
    sh, sw = mask.shape
    pad_dir = random.choice(DIRECTIONS)
    gap = random.randint(*GAP_RANGE)

    sr = random.randint(0, h - sh)
    sc = random.randint(0, w - sw)

    if pad_dir in {"U", "D"}:
        pc = sc + random.randrange(sw)
        pr = sr - gap if pad_dir == "U" else sr + sh - 1 + gap
    else:
        pr = sr + random.randrange(sh)
        pc = sc - gap if pad_dir == "L" else sc + sw - 1 + gap

    pad = (pr, pc)
    if not in_bounds(pad, shape):
        return None

    stamp_coords = absolute_coords(mask, (sr, sc))
    bbox = (sr, sc, sr + sh - 1, sc + sw - 1)
    corridor = corridor_cells(bbox, pad, pad_dir)
    input_cells = stamp_coords | {pad}

    if input_cells & blocked_input:
        return None
    if input_cells & reserved_output:
        return None
    if corridor & occupied_input:
        return None

    k = rotation_k(opening_side, pad_dir)
    rotated = np.rot90(mask, k)
    rotated_opening, anchor, _ = opening_info(rotated)
    assert rotated_opening == pad_dir
    output_coords = translate_mask_to_anchor(rotated, anchor, pad)

    if not all(in_bounds(coord, shape) for coord in output_coords):
        return None
    if output_coords & occupied_input:
        return None
    if output_coords & reserved_output:
        return None
    if output_coords & stamp_coords:
        return None
    if pad not in output_coords:
        return None

    return {
        "mask": mask,
        "stamp_top_left": (sr, sc),
        "bbox": bbox,
        "pad": pad,
        "pad_dir": pad_dir,
        "stamp_coords": stamp_coords,
        "corridor": corridor,
        "output_coords": output_coords,
    }


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    side_min = 10 + 2 * pair_count
    side_max = 14 + 2 * pair_count
    h = random.randint(side_min, side_max)
    w = random.randint(side_min, side_max)
    bg = random.choice(ALL_COLORS)
    palette = [color for color in ALL_COLORS if color != bg]
    random.shuffle(palette)

    grid = np.full((h, w), bg, dtype=int)
    occupied_input: set[tuple[int, int]] = set()
    blocked_input: set[tuple[int, int]] = set()
    reserved_output: set[tuple[int, int]] = set()

    for pair_index in range(pair_count):
        stamp_color = palette[2 * pair_index]
        pad_color = palette[2 * pair_index + 1]
        placed = False

        for _ in range(PAIR_PLACEMENT_ATTEMPTS):
            candidate = _candidate_pair((h, w), blocked_input, occupied_input, reserved_output)
            if candidate is None:
                continue

            for r, c in candidate["stamp_coords"]:
                grid[r, c] = stamp_color
            pr, pc = candidate["pad"]
            grid[pr, pc] = pad_color

            occupied_input |= candidate["stamp_coords"] | {candidate["pad"]}
            blocked_input |= expand_cells(candidate["stamp_coords"] | {candidate["pad"]}, (h, w))
            reserved_output |= candidate["output_coords"]
            placed = True
            break

        if not placed:
            return None

    if get_bg(grid) != bg:
        return None
    if np.count_nonzero(grid != bg) * 100 > MAX_FOREGROUND_PCT * grid.size:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid stamp-pad input")
