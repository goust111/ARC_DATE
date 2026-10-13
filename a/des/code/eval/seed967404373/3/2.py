import random

import numpy as np


BG_COLOR = 0
RECEIVER_COLOR = 5
TOWER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIR_VECTORS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
COUNT_PATTERNS = {
    2: ((1, 2), (1, 3), (2, 2), (2, 3)),
    3: ((1, 2, 3), (1, 1, 2), (1, 2, 2)),
}
GEN_RETRIES = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
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
                for dr, dc in DIR_VECTORS.values():
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _infer_receiver_color(grid: np.ndarray, bg: int) -> int:
    non_bg = [int(color) for color in np.unique(grid) if int(color) != int(bg)]
    repeated = [color for color in non_bg if int(np.sum(grid == color)) > 1]
    singleton_repeated = [
        color
        for color in repeated
        if all(len(comp) == 1 for comp in _connected_components(grid == color))
    ]
    if len(singleton_repeated) == 1:
        return int(singleton_repeated[0])
    if len(repeated) == 1:
        return int(repeated[0])
    if RECEIVER_COLOR in non_bg:
        return RECEIVER_COLOR
    raise AssertionError("could not infer receiver color")


def _collect_objects(grid: np.ndarray, bg: int, receiver_color: int) -> tuple[list[tuple[int, int]], list[tuple[int, int, int]]]:
    receivers = [(int(r), int(c)) for r, c in np.argwhere(grid == receiver_color)]
    towers: list[tuple[int, int, int]] = []
    for color in np.unique(grid):
        color = int(color)
        if color in (bg, receiver_color):
            continue
        coords = np.argwhere(grid == color)
        assert coords.shape[0] == 1, "each tower color must appear exactly once"
        r, c = map(int, coords[0])
        towers.append((r, c, color))
    assert len(receivers) >= 1, "expected at least one receiver"
    return receivers, towers


def _trace_rays(
    grid: np.ndarray,
    bg: int,
    receivers: list[tuple[int, int]],
    towers: list[tuple[int, int, int]],
) -> tuple[np.ndarray, dict[tuple[int, int], int]]:
    out = np.array(grid, copy=True)
    h, w = out.shape
    receiver_set = set(receivers)
    hits = {receiver: 0 for receiver in receivers}

    for r, c, color in towers:
        for dr, dc in DIR_VECTORS.values():
            rr = r + dr
            cc = c + dc
            while 0 <= rr < h and 0 <= cc < w:
                if (rr, cc) in receiver_set:
                    hits[(rr, cc)] += 1
                    break
                if out[rr, cc] == bg:
                    out[rr, cc] = color
                rr += dr
                cc += dc

    return out, hits


def _expand_receivers(
    out: np.ndarray,
    receivers: list[tuple[int, int]],
    hits: dict[tuple[int, int], int],
    receiver_color: int,
    tower_positions: set[tuple[int, int]],
) -> np.ndarray:
    h, w = out.shape
    receiver_set = set(receivers)
    for r, c in receivers:
        radius = max(0, hits.get((r, c), 0) - 1)
        for rr in range(max(0, r - radius), min(h, r + radius + 1)):
            for cc in range(max(0, c - radius), min(w, c + radius + 1)):
                if (rr, cc) in tower_positions:
                    continue
                if (rr, cc) in receiver_set and (rr, cc) != (r, c):
                    continue
                out[rr, cc] = receiver_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    receiver_color = _infer_receiver_color(g, bg)
    receivers, towers = _collect_objects(g, bg, receiver_color)
    tower_positions = {(r, c) for r, c, _ in towers}

    out, hits = _trace_rays(g, bg, receivers, towers)
    return _expand_receivers(out, receivers, hits, receiver_color, tower_positions)


def _square_bbox(receiver: dict) -> tuple[int, int, int, int]:
    radius = int(receiver["hits"]) - 1
    return (
        int(receiver["r"]) - radius,
        int(receiver["c"]) - radius,
        int(receiver["r"]) + radius,
        int(receiver["c"]) + radius,
    )


def _point_in_expansion(r: int, c: int, receivers: list[dict]) -> bool:
    for receiver in receivers:
        r0, c0, r1, c1 = _square_bbox(receiver)
        if r0 <= r <= r1 and c0 <= c <= c1:
            return True
    return False


def _boxes_too_close(a: dict, b: dict) -> bool:
    ar0, ac0, ar1, ac1 = _square_bbox(a)
    br0, bc0, br1, bc1 = _square_bbox(b)
    return not (ar1 + 1 < br0 or br1 + 1 < ar0 or ac1 + 1 < bc0 or bc1 + 1 < ac0)


def _receiver_candidates(h: int, w: int, spec: dict, placed: list[dict]) -> list[tuple[int, int]]:
    radius = int(spec["hits"]) - 1
    min_r = radius + (2 if "up" in spec["dirs"] else 0)
    max_r = h - 1 - radius - (2 if "down" in spec["dirs"] else 0)
    min_c = radius + (2 if "left" in spec["dirs"] else 0)
    max_c = w - 1 - radius - (2 if "right" in spec["dirs"] else 0)
    if min_r > max_r or min_c > max_c:
        return []

    used_rows = {int(receiver["r"]) for receiver in placed}
    used_cols = {int(receiver["c"]) for receiver in placed}
    candidates: list[tuple[int, int]] = []
    for r in range(min_r, max_r + 1):
        if r in used_rows:
            continue
        for c in range(min_c, max_c + 1):
            if c in used_cols:
                continue
            candidate = {"r": r, "c": c, "hits": spec["hits"]}
            if any(_boxes_too_close(candidate, receiver) for receiver in placed):
                continue
            candidates.append((r, c))
    return candidates


def _place_receivers(h: int, w: int, specs: list[dict]) -> list[dict] | None:
    receivers: list[dict] = []
    order = sorted(range(len(specs)), key=lambda idx: (-int(specs[idx]["hits"]), idx))
    for idx in order:
        candidates = _receiver_candidates(h, w, specs[idx], receivers)
        if not candidates:
            return None
        r, c = random.choice(candidates)
        receivers.append(
            {
                "r": r,
                "c": c,
                "hits": int(specs[idx]["hits"]),
                "dirs": tuple(specs[idx]["dirs"]),
            }
        )
    random.shuffle(receivers)
    return receivers


def _tower_candidates(
    h: int,
    w: int,
    receiver: dict,
    direction: str,
    occupied: set[tuple[int, int]],
    receiver_rows: set[int],
    receiver_cols: set[int],
    receivers: list[dict],
) -> list[tuple[int, int]]:
    r = int(receiver["r"])
    c = int(receiver["c"])
    radius = int(receiver["hits"]) - 1
    candidates: list[tuple[int, int]] = []

    if direction == "up":
        for rr in range(0, r - radius - 1):
            if rr in receiver_rows:
                continue
            if (rr, c) in occupied or _point_in_expansion(rr, c, receivers):
                continue
            candidates.append((rr, c))
    elif direction == "down":
        for rr in range(r + radius + 2, h):
            if rr in receiver_rows:
                continue
            if (rr, c) in occupied or _point_in_expansion(rr, c, receivers):
                continue
            candidates.append((rr, c))
    elif direction == "left":
        for cc in range(0, c - radius - 1):
            if cc in receiver_cols:
                continue
            if (r, cc) in occupied or _point_in_expansion(r, cc, receivers):
                continue
            candidates.append((r, cc))
    else:
        for cc in range(c + radius + 2, w):
            if cc in receiver_cols:
                continue
            if (r, cc) in occupied or _point_in_expansion(r, cc, receivers):
                continue
            candidates.append((r, cc))

    return candidates


def _place_targeted_towers(h: int, w: int, receivers: list[dict]) -> list[tuple[int, int]] | None:
    occupied = {(int(receiver["r"]), int(receiver["c"])) for receiver in receivers}
    receiver_rows = {int(receiver["r"]) for receiver in receivers}
    receiver_cols = {int(receiver["c"]) for receiver in receivers}
    towers: list[tuple[int, int]] = []

    for receiver in sorted(receivers, key=lambda item: (-int(item["hits"]), int(item["r"]), int(item["c"]))):
        for direction in receiver["dirs"]:
            candidates = _tower_candidates(
                h,
                w,
                receiver,
                direction,
                occupied,
                receiver_rows,
                receiver_cols,
                receivers,
            )
            if not candidates:
                return None
            tower = random.choice(candidates)
            towers.append(tower)
            occupied.add(tower)

    return towers


def _place_extra_towers(
    h: int,
    w: int,
    count: int,
    receivers: list[dict],
    occupied: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    receiver_rows = {int(receiver["r"]) for receiver in receivers}
    receiver_cols = {int(receiver["c"]) for receiver in receivers}
    candidates = []
    for r in range(h):
        if r in receiver_rows:
            continue
        for c in range(w):
            if c in receiver_cols or (r, c) in occupied:
                continue
            if _point_in_expansion(r, c, receivers):
                continue
            candidates.append((r, c))
    if len(candidates) < count:
        return None
    return random.sample(candidates, count)


def _render_input(h: int, w: int, receivers: list[dict], towers: list[tuple[int, int]]) -> np.ndarray:
    grid = np.full((h, w), BG_COLOR, dtype=int)
    for receiver in receivers:
        grid[int(receiver["r"]), int(receiver["c"])] = RECEIVER_COLOR
    colors = random.sample(TOWER_COLORS, len(towers))
    random.shuffle(colors)
    for color, (r, c) in zip(colors, towers):
        grid[int(r), int(c)] = int(color)
    return grid


def _receiver_hit_map(grid: np.ndarray) -> dict[tuple[int, int], int]:
    bg = _most_common_color(grid)
    receiver_color = _infer_receiver_color(grid, bg)
    receivers, towers = _collect_objects(grid, bg, receiver_color)
    _, hits = _trace_rays(grid, bg, receivers, towers)
    return hits


def _attempt_generate() -> np.ndarray | None:
    receiver_count = random.choice((2, 3))
    hit_counts = list(random.choice(COUNT_PATTERNS[receiver_count]))
    random.shuffle(hit_counts)

    specs = []
    for hit_count in hit_counts:
        specs.append(
            {
                "hits": hit_count,
                "dirs": tuple(random.sample(tuple(DIR_VECTORS), hit_count)),
            }
        )

    min_size = 15 if receiver_count == 2 else 17
    max_size = 22
    h = random.randint(min_size, max_size)
    w = random.randint(min_size, max_size)

    receivers = _place_receivers(h, w, specs)
    if receivers is None:
        return None

    towers = _place_targeted_towers(h, w, receivers)
    if towers is None:
        return None

    occupied = {(int(receiver["r"]), int(receiver["c"])) for receiver in receivers}
    occupied.update(towers)
    extra_limit = min(2, len(TOWER_COLORS) - len(towers))
    extra_count = 0
    if extra_limit > 0 and random.random() < 0.7:
        extra_count = random.randint(0, extra_limit)
    extra_towers = _place_extra_towers(h, w, extra_count, receivers, occupied)
    if extra_towers is None:
        return None
    towers.extend(extra_towers)

    grid = _render_input(h, w, receivers, towers)
    expected_hits = {(int(receiver["r"]), int(receiver["c"])): int(receiver["hits"]) for receiver in receivers}
    actual_hits = _receiver_hit_map(grid)
    if actual_hits != expected_hits:
        return None

    non_bg = int(np.sum(grid != BG_COLOR))
    if non_bg >= (h * w) // 2:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
