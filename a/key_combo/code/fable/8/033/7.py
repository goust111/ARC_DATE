import random
from collections import Counter, deque

import numpy as np


GREY = 5
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]
PLUS_OFFSETS = {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)}


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0:r1 + 1, c0:c1 + 1].copy()


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def component_coords(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    dirs = DIRS4 if connectivity == 4 else DIRS8
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def reflect(direction: tuple[int, int], mirror_kind: str) -> tuple[int, int]:
    dr, dc = direction
    if mirror_kind == "\\":
        return dc, dr
    return -dc, -dr


def is_plus(coords: np.ndarray) -> bool:
    if int(coords.shape[0]) != 5:
        return False
    r0, c0, r1, c1 = bbox_coords(coords)
    if (r1 - r0 + 1, c1 - c0 + 1) != (3, 3):
        return False
    norm = {(int(r - r0), int(c - c0)) for r, c in coords}
    return norm == PLUS_OFFSETS


def detect_plus_color(grid: np.ndarray, bg: int) -> int | None:
    candidates: list[tuple[int, int, int]] = []
    for color in sorted(int(x) for x in np.unique(grid)):
        if color in (bg, GREY):
            continue
        comps = component_coords(grid == color, connectivity=4)
        if comps and all(is_plus(comp) for comp in comps):
            candidates.append((len(comps), sum(int(comp.shape[0]) for comp in comps), color))
    if not candidates:
        return None
    candidates.sort()
    return int(candidates[-1][2])


def plus_labels(grid: np.ndarray, plus_color: int) -> tuple[np.ndarray, list[np.ndarray]]:
    comps = component_coords(grid == plus_color, connectivity=4)
    lab = np.zeros_like(grid, dtype=int)
    for idx, comp in enumerate(comps, start=1):
        for r, c in comp:
            lab[int(r), int(c)] = idx
    return lab, comps


def mirror_map(grid: np.ndarray) -> np.ndarray:
    out = np.full(grid.shape, "", dtype="<U1")
    for comp in component_coords(grid == GREY, connectivity=8):
        diffs = {int(r - c) for r, c in comp}
        sums = {int(r + c) for r, c in comp}
        if len(diffs) == 1:
            kind = "\\"
        elif len(sums) == 1:
            kind = "/"
        else:
            continue
        for r, c in comp:
            out[int(r), int(c)] = kind
    return out


def extract_markers(
    grid: np.ndarray,
    bg: int,
    plus_color: int,
    plus_lab: np.ndarray,
) -> list[dict]:
    mask = (grid != bg) & (grid != plus_color) & (grid != GREY)
    markers: list[dict] = []
    for comp in component_coords(mask, connectivity=4):
        b = bbox_coords(comp)
        tile = crop_bbox(grid, b)
        dirs: set[tuple[int, int]] = set()
        plus_ids: set[int] = set()
        for r, c in comp:
            rr, cc = int(r), int(c)
            for dr, dc in DIRS4:
                nr, nc = rr + dr, cc + dc
                if in_bounds(grid.shape, nr, nc) and plus_lab[nr, nc] > 0:
                    dirs.add((-dr, -dc))
                    plus_ids.add(int(plus_lab[nr, nc]))
        if len(dirs) != 1 or len(plus_ids) != 1:
            continue
        markers.append(
            {
                "bbox": b,
                "tile": tile,
                "direction": next(iter(dirs)),
                "plus_id": next(iter(plus_ids)),
            }
        )
    markers.sort(key=lambda m: m["bbox"])
    return markers


def simulate_marker(
    inp: np.ndarray,
    out: np.ndarray,
    tile: np.ndarray,
    start_bbox: tuple[int, int, int, int],
    direction: tuple[int, int],
    mirrors: np.ndarray,
    protect: np.ndarray,
    bg: int,
) -> None:
    h, w = tile.shape
    r0, c0, _r1, _c1 = start_bbox
    dr, dc = direction
    mask = tile != bg
    while True:
        nr0 = r0 + dr * h
        nc0 = c0 + dc * w
        if nr0 < 0 or nc0 < 0 or nr0 + h > out.shape[0] or nc0 + w > out.shape[1]:
            break
        hit = mirrors[nr0:nr0 + h, nc0:nc0 + w]
        kinds = {x for x in hit.flatten() if x}
        if kinds:
            if len(kinds) != 1:
                break
            dr, dc = reflect((dr, dc), next(iter(kinds)))
            r0, c0 = nr0, nc0
            continue
        write_mask = mask & (~protect[nr0:nr0 + h, nc0:nc0 + w])
        out[nr0:nr0 + h, nc0:nc0 + w][write_mask] = tile[write_mask]
        r0, c0 = nr0, nc0


def main(grid: np.ndarray) -> np.ndarray:
    """Repeat each arm-mounted marker tile away from its '+' source, reflecting at grey diagonal mirrors while preserving the tile cadence.

    Input structure:
    - The background is the most common color.
    - One non-grey color appears only as one or more classic 3x3 '+' shapes.
    - Around the outer side of each arm tip sits a small colored marker tile, and separate grey diagonal segments act as mirrors.

    Rule:
    - For every marker tile, determine the unique cardinal direction from the adjacent '+' arm toward the tile.
    - Copy the whole tile periodically in that direction using the tile's own height/width as the step; if the next copy would overlap a grey diagonal mirror, skip that copy, reflect the travel direction by 90 degrees according to '/' or '\\', and continue from the mirrored beat without resetting phase.
    - Keep all original non-background cells, especially the '+' shapes and mirrors, unchanged; stop when the next full tile copy would leave the canvas.

    Key insight:
    - The marker copies behave like a block-shaped light pulse: the attachment side fixes the initial ray, and mirror hits consume one period while only the direction changes.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = get_bg(g)
    plus_color = detect_plus_color(g, bg)
    if plus_color is None:
        return out
    plus_lab, _pluses = plus_labels(g, plus_color)
    mirrors = mirror_map(g)
    protect = g != bg
    for marker in extract_markers(g, bg, plus_color, plus_lab):
        simulate_marker(
            g,
            out,
            marker["tile"],
            marker["bbox"],
            marker["direction"],
            mirrors,
            protect,
            bg,
        )
    return out


def _rand_tile(rng: random.Random, palette: list[int]) -> np.ndarray:
    while True:
        colors = rng.sample(palette, rng.randint(2, min(4, len(palette))))
        vals = [rng.choice(colors) for _ in range(4)]
        if len(set(vals)) < 2:
            continue
        tile = np.array(vals, dtype=int).reshape(2, 2)
        if np.array_equal(tile[0], tile[1]):
            continue
        if np.array_equal(tile[:, 0], tile[:, 1]):
            continue
        return tile


def _paint_plus(grid: np.ndarray, r0: int, c0: int, color: int) -> None:
    for dr, dc in PLUS_OFFSETS:
        grid[r0 + dr, c0 + dc] = color


def _seed_bbox(plus_r0: int, plus_c0: int, direction: tuple[int, int], offset: int) -> tuple[int, int, int, int]:
    dr, dc = direction
    if (dr, dc) == (-1, 0):
        r0, c0 = plus_r0 - 2, plus_c0 + offset
    elif (dr, dc) == (1, 0):
        r0, c0 = plus_r0 + 3, plus_c0 + offset
    elif (dr, dc) == (0, -1):
        r0, c0 = plus_r0 + offset, plus_c0 - 2
    else:
        r0, c0 = plus_r0 + offset, plus_c0 + 3
    return r0, c0, r0 + 1, c0 + 1


def _paste_tile(grid: np.ndarray, tile: np.ndarray, r0: int, c0: int) -> None:
    grid[r0:r0 + 2, c0:c0 + 2] = tile


def _reserve(blocked: np.ndarray, r0: int, c0: int) -> None:
    blocked[r0:r0 + 2, c0:c0 + 2] = True


def _box_in_bounds(shape: tuple[int, int], r0: int, c0: int) -> bool:
    h, w = shape
    return 0 <= r0 and 0 <= c0 and r0 + 2 <= h and c0 + 2 <= w


def _box_free(blocked: np.ndarray, r0: int, c0: int) -> bool:
    return _box_in_bounds(blocked.shape, r0, c0) and not blocked[r0:r0 + 2, c0:c0 + 2].any()


def _step(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return 2 * dr, 2 * dc


def _distance_to_edge(shape: tuple[int, int], r0: int, c0: int, direction: tuple[int, int]) -> int:
    h, w = shape
    dr, dc = direction
    if dr == -1:
        return r0 // 2
    if dr == 1:
        return (h - (r0 + 2)) // 2
    if dc == -1:
        return c0 // 2
    return (w - (c0 + 2)) // 2


def _paint_mirror(grid: np.ndarray, r0: int, c0: int, kind: str) -> None:
    if kind == "\\":
        grid[r0, c0] = GREY
        grid[r0 + 1, c0 + 1] = GREY
    else:
        grid[r0, c0 + 1] = GREY
        grid[r0 + 1, c0] = GREY


def _plan_arm(
    inp: np.ndarray,
    out: np.ndarray,
    blocked: np.ndarray,
    seed_bbox: tuple[int, int, int, int],
    tile: np.ndarray,
    direction: tuple[int, int],
    forced_turns: list[str],
    forced_after: list[int],
    extra_turn_budget: int,
    rng: random.Random,
) -> dict | None:
    r0, c0, _r1, _c1 = seed_bbox
    cur_dir = direction
    pending = list(forced_turns)
    pending_after = list(forced_after)
    drawn = 0
    turns: list[str] = []
    since_turn = 0
    random_turns = 0

    while True:
        sr, sc = _step(cur_dir)
        nr0, nc0 = r0 + sr, c0 + sc
        if not _box_in_bounds(out.shape, nr0, nc0):
            break
        if not _box_free(blocked, nr0, nc0):
            return None

        chosen: tuple[str, tuple[int, int]] | None = None
        if pending and since_turn >= pending_after[0]:
            kind = pending[0]
            new_dir = reflect(cur_dir, kind)
            tr, tc = _step(new_dir)
            rr0, cc0 = nr0 + tr, nc0 + tc
            if _box_free(blocked, rr0, cc0):
                chosen = (kind, new_dir)
        elif not pending and random_turns < extra_turn_budget and drawn >= 1 and since_turn >= 1 and rng.random() < 0.2:
            kinds = ["/", "\\"]
            rng.shuffle(kinds)
            for kind in kinds:
                new_dir = reflect(cur_dir, kind)
                tr, tc = _step(new_dir)
                rr0, cc0 = nr0 + tr, nc0 + tc
                if _box_free(blocked, rr0, cc0):
                    chosen = (kind, new_dir)
                    break

        if chosen is not None:
            kind, new_dir = chosen
            _paint_mirror(inp, nr0, nc0, kind)
            _paint_mirror(out, nr0, nc0, kind)
            _reserve(blocked, nr0, nc0)
            r0, c0 = nr0, nc0
            cur_dir = new_dir
            turns.append(kind)
            since_turn = 0
            if pending:
                pending.pop(0)
                pending_after.pop(0)
            else:
                random_turns += 1
            continue

        _paste_tile(out, tile, nr0, nc0)
        _reserve(blocked, nr0, nc0)
        r0, c0 = nr0, nc0
        drawn += 1
        since_turn += 1

    if pending or drawn < 2:
        return None
    return {"turns": turns, "drawn": drawn}


def _choose_plus_positions(h: int, w: int, n_plus: int, rng: random.Random) -> list[tuple[int, int]] | None:
    if n_plus == 2:
        if h < 28 or w < 28:
            return None
        band = [6, 7, 8]
        if rng.random() < 0.5:
            return [
                (rng.choice(band), rng.choice(band)),
                (h - 11 - rng.choice([0, 1, 2]), w - 11 - rng.choice([0, 1, 2])),
            ]
        return [
            (rng.choice(band), w - 11 - rng.choice([0, 1, 2])),
            (h - 11 - rng.choice([0, 1, 2]), rng.choice(band)),
        ]

    rows = list(range(6, h - 8))
    cols = list(range(6, w - 8))
    if not rows or not cols:
        return None
    positions: list[tuple[int, int]] = []
    for _ in range(200):
        positions.clear()
        ok = True
        for _k in range(n_plus):
            rng.shuffle(rows)
            rng.shuffle(cols)
            picked = None
            for r0 in rows:
                for c0 in cols:
                    if all(max(abs(r0 - rr), abs(c0 - cc)) >= 10 for rr, cc in positions):
                        picked = (r0, c0)
                        break
                if picked is not None:
                    break
            if picked is None:
                ok = False
                break
            positions.append(picked)
        if ok:
            return positions[:]
    return None


def _arm_specs(arms: list[dict], rng: random.Random) -> list[dict]:
    plus_positions = sorted({arm["plus_pos"] for arm in arms})
    specs = [{"forced": [], "after": [], "extra": 0} for _ in arms]

    if len(plus_positions) == 2:
        (r0, c0), (r1, c1) = sorted(plus_positions)
        if c0 < c1:
            chosen_plus = (r0, c0)
            forced_map = {
                (-1, 0): {"forced": ["\\"], "after": [1], "extra": 0},
                (0, 1): {"forced": ["/"], "after": [1], "extra": 0},
                (1, 0): {"forced": ["/"], "after": [2], "extra": 0},
                (0, -1): {"forced": [], "after": [], "extra": 0},
            }
        else:
            chosen_plus = (r0, c1)
            forced_map = {
                (-1, 0): {"forced": ["/"], "after": [1], "extra": 0},
                (0, -1): {"forced": ["\\"], "after": [1], "extra": 0},
                (1, 0): {"forced": ["\\"], "after": [2], "extra": 0},
                (0, 1): {"forced": [], "after": [], "extra": 0},
            }
        for idx, arm in enumerate(arms):
            if arm["plus_pos"] == chosen_plus:
                specs[idx] = dict(forced_map[arm["direction"]])
        return specs

    order = list(range(len(arms)))
    rng.shuffle(order)
    specs[order[0]] = {"forced": ["/"], "after": [rng.randint(1, 3)], "extra": 0}
    specs[order[1]] = {"forced": ["\\"], "after": [rng.randint(1, 3)], "extra": 0}
    specs[order[2]] = {"forced": [], "after": [], "extra": 0}
    if len(order) > 3 and rng.random() < 0.5:
        specs[order[3]] = {
            "forced": [rng.choice(["/", "\\"]), rng.choice(["/", "\\"])],
            "after": [rng.randint(1, 2), rng.randint(1, 2)],
            "extra": 0,
        }
    for idx in order[4:]:
        specs[idx] = {"forced": [], "after": [], "extra": rng.choice([0, 0, 1])}
    return specs


def _attempt_generation(rng: random.Random) -> tuple[np.ndarray, np.ndarray, dict] | None:
    h = rng.randint(22, 30)
    w = rng.randint(22, 30)
    bg = rng.choice([c for c in range(10) if c != GREY])
    plus_color = rng.choice([c for c in range(10) if c not in (bg, GREY)])
    palette = [c for c in range(10) if c not in (bg, GREY, plus_color)]

    n_plus = 2 if h >= 28 and w >= 28 and rng.random() < 0.55 else 1
    pluses = _choose_plus_positions(h, w, n_plus, rng)
    if pluses is None:
        return None

    inp = np.full((h, w), bg, dtype=int)
    out = inp.copy()
    blocked = np.zeros((h, w), dtype=bool)
    arms: list[dict] = []

    for pr0, pc0 in pluses:
        _paint_plus(inp, pr0, pc0, plus_color)
        _paint_plus(out, pr0, pc0, plus_color)
        for dr, dc in PLUS_OFFSETS:
            blocked[pr0 + dr, pc0 + dc] = True
        for direction in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            offset = rng.choice([0, 1])
            bbox = _seed_bbox(pr0, pc0, direction, offset)
            r0, c0, _r1, _c1 = bbox
            if not _box_free(blocked, r0, c0):
                return None
            tile = _rand_tile(rng, palette)
            _paste_tile(inp, tile, r0, c0)
            _paste_tile(out, tile, r0, c0)
            _reserve(blocked, r0, c0)
            arms.append({"bbox": bbox, "tile": tile, "direction": direction, "plus_pos": (pr0, pc0)})

    specs = _arm_specs(arms, rng)
    order = sorted(range(len(arms)), key=lambda i: (len(specs[i]["forced"]), specs[i]["extra"]), reverse=True)

    turn_kinds: list[str] = []
    straight_arms = 0
    for idx in order:
        plan = _plan_arm(
            inp,
            out,
            blocked,
            arms[idx]["bbox"],
            arms[idx]["tile"],
            arms[idx]["direction"],
            specs[idx]["forced"],
            specs[idx]["after"],
            specs[idx]["extra"],
            rng,
        )
        if plan is None:
            return None
        turn_kinds.extend(plan["turns"])
        if not plan["turns"]:
            straight_arms += 1

    if "/" not in turn_kinds or "\\" not in turn_kinds or straight_arms == 0:
        return None
    if np.array_equal(inp, out):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp, out, {"pluses": n_plus, "turn_kinds": turn_kinds, "straight_arms": straight_arms}


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(500):
        result = _attempt_generation(rng)
        if result is not None:
            inp, _out, _meta = result
            return inp
    raise RuntimeError("generate_input: failed after retries")
