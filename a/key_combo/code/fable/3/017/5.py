import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
SPRITE_COLORS = list(range(2, 10))
NEIGHBORS8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, transparent: int | None = None) -> None:
    h, w = patch.shape
    view = dst[int(r0) : int(r0) + h, int(c0) : int(c0) + w]
    if transparent is None:
        view[:, :] = patch
    else:
        mask = patch != int(transparent)
        view[mask] = patch[mask]


def _connected8(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    start = tuple(map(int, pts[0]))
    seen = {start}
    q: deque[tuple[int, int]] = deque([start])
    H, W = mask.shape
    while q:
        r, c = q.popleft()
        for dr, dc in NEIGHBORS8:
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W and mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                q.append((rr, cc))
    return len(seen) == int(mask.sum())


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    H, W = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        q: deque[tuple[int, int]] = deque([(sr, sc)])
        comp: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for dr, dc in NEIGHBORS8:
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
        comps.append(comp)
    return [np.array(comp, dtype=int) for comp in comps]


def _extract_sprites(g: np.ndarray) -> tuple[list[dict], np.ndarray]:
    H, W = g.shape
    owner = np.full((H, W), -1, dtype=int)
    sprites: list[dict] = []
    for color in [int(x) for x in np.unique(g) if int(x) not in (BLACK, BLUE)]:
        comps = _component_coords(g == int(color))
        assert len(comps) == 1, "each sprite color must form one 8-connected object"
        coords = comps[0]
        box = _bbox(coords)
        r0, c0, r1, c1 = box
        assert (r1 - r0 + 1, c1 - c0 + 1) == (3, 3), "each sprite must have a 3x3 bbox"
        crop = np.full((3, 3), BLACK, dtype=int)
        crop[coords[:, 0] - r0, coords[:, 1] - c0] = int(color)
        assert crop[0].any() and crop[2].any() and crop[:, 0].any() and crop[:, 2].any()
        idx = len(sprites)
        owner[coords[:, 0], coords[:, 1]] = idx
        sprites.append({"color": int(color), "bbox": box, "crop": crop, "coords": coords})
    return sprites, owner


def _trace_ray(g: np.ndarray, sprites: list[dict], owner: np.ndarray) -> tuple[list[int], dict]:
    H, W = g.shape
    blue = np.argwhere(g == BLUE)
    assert blue.shape == (1, 2), "expected exactly one blue start pixel"
    r, c = map(int, blue[0])
    assert r == H - 1, "the blue start pixel must be on the bottom row"

    dc = 1
    collected: list[int] = []
    seen: set[int] = set()
    steps: list[dict] = []
    stats = {
        "wall_bounces": 0,
        "sprite_bounces": 0,
        "vertical_blocks": 0,
        "vertical_sprite_blocks": 0,
    }

    while r > 0:
        nr = r - 1
        pre_r, pre_c, pre_dc = r, c, dc
        wall = False
        if c + dc < 0 or c + dc >= W:
            dc = -dc
            wall = True
            stats["wall_bounces"] += 1

        target_c = c + dc
        assert 0 <= target_c < W, "grid must be at least two columns wide"
        hit = int(owner[nr, target_c])
        step = {
            "pre": (pre_r, pre_c),
            "pre_dc": pre_dc,
            "trial_dc": dc,
            "target": (nr, target_c),
            "wall": wall,
            "hit": hit,
            "new_hit": None,
            "vertical_block": False,
            "vertical_sprite_block": False,
        }

        if hit >= 0:
            stats["sprite_bounces"] += 1
            if hit not in seen:
                seen.add(hit)
                collected.append(hit)
                step["new_hit"] = hit
            dc = -dc
            alt_c = c + dc
            blocked_by_sprite = 0 <= alt_c < W and int(owner[nr, alt_c]) >= 0
            if 0 <= alt_c < W and not blocked_by_sprite:
                c = alt_c
            else:
                stats["vertical_blocks"] += 1
                if blocked_by_sprite:
                    stats["vertical_sprite_blocks"] += 1
                    step["vertical_sprite_block"] = True
                step["vertical_block"] = True
            r = nr
        else:
            c = target_c
            r = nr

        step["post"] = (r, c)
        step["post_dc"] = dc
        steps.append(step)

    stats["steps"] = steps
    stats["collected_colors"] = [sprites[i]["color"] for i in collected]
    return collected, stats


def _pack_sprites(sprites: list[dict], order: list[int]) -> np.ndarray:
    assert order, "expected the ray to collect at least one sprite"
    out = np.full((3, 3 * len(order)), BLACK, dtype=int)
    for i, sprite_idx in enumerate(order):
        _paste(out, sprites[sprite_idx]["crop"], 0, 3 * i)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pack the 3x3 sprites collected by an upward bouncing ray from the blue start pixel.

    Input structure:
    - The background is black, and the bottom row contains exactly one blue start pixel.
    - Every other non-black color is a unique single-color sprite with an 8-connected 3x3 bounding box.
    - The left and right grid edges act as reflection walls; the ray itself is not drawn.

    Rule:
    - Start at the blue pixel with horizontal direction to the right, then repeatedly move one row upward.
    - Before each upward diagonal step, reflect the horizontal direction if that step would cross a side wall.
    - If the upward diagonal target is a sprite pixel, collect that sprite once, reverse horizontal direction, and do not enter the hit cell; if the reversed upward diagonal target is also blocked, move only upward in that step.
    - Stop when the next upward step leaves the top edge, then output the collected 3x3 sprite crops packed left-to-right in hit order.

    Key insight:
    - Sprite hits rewrite the later trajectory, so the packed order is the collision order of the bouncing path, not a spatial reading order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    sprites, owner = _extract_sprites(g)
    order, _stats = _trace_ray(g, sprites, owner)
    return _pack_sprites(sprites, order)


# --------------- generator ---------------

_GEN_CALLS = 0
_GEN_ATTEMPTS = 500
_GEN_H_RANGE = (17, 26)
_GEN_W_RANGE = (7, 13)
_GEN_HIT_RANGE = (3, 5)
_GEN_DISTRACTOR_RANGE = (1, 3)
_MASK_CACHE: dict[tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]], list[np.ndarray]] = {}


def _mask_pool(required: tuple[tuple[int, int], ...], forbidden: tuple[tuple[int, int], ...] = ()) -> list[np.ndarray]:
    key = (tuple(sorted(required)), tuple(sorted(forbidden)))
    if key in _MASK_CACHE:
        return _MASK_CACHE[key]

    out: list[np.ndarray] = []
    req = set(required)
    ban = set(forbidden)
    for bits in range(1, 1 << 9):
        mask = np.array([(bits >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
        area = int(mask.sum())
        if not (4 <= area <= 8):
            continue
        if any(not mask[r, c] for r, c in req):
            continue
        if any(mask[r, c] for r, c in ban):
            continue
        if not (mask[0].any() and mask[2].any() and mask[:, 0].any() and mask[:, 2].any()):
            continue
        if not _connected8(mask):
            continue
        out.append(mask)
    _MASK_CACHE[key] = out
    return out


def _edge_col_for_direction(dc: int) -> int:
    return 0 if int(dc) == 1 else 2


def _can_place_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int) -> bool:
    H, W = grid.shape
    if top < 0 or left < 0 or top + 3 > H or left + 3 > W:
        return False
    for rr, cc in np.argwhere(mask):
        r, c = int(top + rr), int(left + cc)
        if int(grid[r, c]) != BLACK:
            return False
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and int(grid[nr, nc]) not in (BLACK,):
                    return False
    return True


def _place_sprite_for_hit(
    grid: np.ndarray,
    color: int,
    target: tuple[int, int],
    approach_dc: int,
    *,
    require_next_edge: bool = False,
    forbid_next_edge: bool = False,
) -> bool:
    tr, tc = map(int, target)
    edge = _edge_col_for_direction(approach_dc)
    required = [(2, edge)]
    forbidden: list[tuple[int, int]] = []
    if require_next_edge:
        required.append((1, edge))
    if forbid_next_edge:
        forbidden.append((1, edge))

    pool = _mask_pool(tuple(required), tuple(forbidden))
    random.shuffle(pool)
    top = tr - 2
    left = tc - edge
    for mask in pool:
        if not _can_place_mask(grid, mask, top, left):
            continue
        patch = np.where(mask, int(color), BLACK)
        _paste(grid, patch, top, left, transparent=BLACK)
        return True
    return False


def _trace_current(grid: np.ndarray) -> tuple[list[dict], list[int], list[dict], np.ndarray]:
    sprites, owner = _extract_sprites(grid)
    order, stats = _trace_ray(grid, sprites, owner)
    return stats["steps"], order, sprites, owner


def _candidate_steps(
    steps: list[dict],
    remaining_after_event: int,
    *,
    after_step: int,
    need_alt: bool,
    width: int,
) -> list[dict]:
    min_row = 2 + 4 * max(0, int(remaining_after_event))
    candidates: list[dict] = []
    for i, step in enumerate(steps):
        if i <= after_step + 2:
            continue
        if step["hit"] != -1:
            continue
        tr, tc = step["target"]
        if tr < min_row:
            continue
        c = step["pre"][1]
        dc = int(step["trial_dc"])
        if need_alt and not (0 <= c - dc < width):
            continue
        candidates.append(step)
    random.shuffle(candidates)
    return candidates


def _place_planned_hits(
    grid: np.ndarray,
    hit_count: int,
    colors: list[int],
    force_vertical: bool,
) -> bool:
    used = 0
    vertical_done = False
    while used < hit_count:
        try:
            steps, order, _sprites, _owner = _trace_current(grid)
        except AssertionError:
            return False
        if len(order) != used:
            return False

        last_new = -1
        for i, step in enumerate(steps):
            if step["new_hit"] is not None:
                last_new = i

        can_pair = force_vertical and not vertical_done and used + 2 <= hit_count
        if can_pair:
            candidates = _candidate_steps(
                steps,
                hit_count - used - 2,
                after_step=last_new,
                need_alt=True,
                width=grid.shape[1],
            )
            for step in candidates:
                tr, tc = step["target"]
                pre_c = int(step["pre"][1])
                dc = int(step["trial_dc"])
                alt = (int(tr), pre_c - dc)
                trial = grid.copy()
                ok1 = _place_sprite_for_hit(
                    trial,
                    colors[used],
                    (int(tr), int(tc)),
                    dc,
                    forbid_next_edge=True,
                )
                ok2 = ok1 and _place_sprite_for_hit(
                    trial,
                    colors[used + 1],
                    alt,
                    -dc,
                    require_next_edge=True,
                )
                if not ok2:
                    continue
                try:
                    _steps2, order2, _sprites2, _owner2 = _trace_current(trial)
                except AssertionError:
                    continue
                if len(order2) >= used + 2:
                    grid[:, :] = trial
                    used += 2
                    vertical_done = True
                    break
            else:
                return False
            continue

        candidates = _candidate_steps(
            steps,
            hit_count - used - 1,
            after_step=last_new,
            need_alt=False,
            width=grid.shape[1],
        )
        for step in candidates:
            tr, tc = step["target"]
            dc = int(step["trial_dc"])
            trial = grid.copy()
            if not _place_sprite_for_hit(trial, colors[used], (int(tr), int(tc)), dc):
                continue
            try:
                _steps2, order2, _sprites2, _owner2 = _trace_current(trial)
            except AssertionError:
                continue
            if len(order2) >= used + 1:
                grid[:, :] = trial
                used += 1
                break
        else:
            return False

    return (not force_vertical) or vertical_done


def _place_random_distractor(grid: np.ndarray, color: int) -> bool:
    H, W = grid.shape
    pool = _mask_pool(())
    for _ in range(120):
        mask = random.choice(pool)
        top = random.randint(0, H - 3)
        left = random.randint(0, W - 3)
        if not _can_place_mask(grid, mask, top, left):
            continue
        patch = np.where(mask, int(color), BLACK)
        _paste(grid, patch, top, left, transparent=BLACK)
        return True
    return False


def _semantic_accepts(grid: np.ndarray, expected_hits: int, force_vertical: bool) -> bool:
    try:
        sprites, owner = _extract_sprites(grid)
        order, stats = _trace_ray(grid, sprites, owner)
        out = _pack_sprites(sprites, order)
    except AssertionError:
        return False
    if len(order) != int(expected_hits):
        return False
    if len(sprites) <= len(order):
        return False
    if stats["wall_bounces"] < 1 or stats["sprite_bounces"] < len(order):
        return False
    if force_vertical and stats["vertical_sprite_blocks"] < 1:
        return False
    if not force_vertical and stats["vertical_sprite_blocks"] != 0:
        return False
    reading_order = sorted(order, key=lambda idx: sprites[idx]["bbox"][:2])
    if order == reading_order:
        return False
    if not any(int(np.count_nonzero(sprites[idx]["crop"])) < 9 for idx in order):
        return False
    if out.shape != (3, 3 * len(order)):
        return False
    try:
        actual = main(grid)
    except AssertionError:
        return False
    return np.array_equal(actual, out)


def _gen_attempt(force_vertical: bool) -> np.ndarray | None:
    H = random.randint(*_GEN_H_RANGE)
    W = random.randint(*_GEN_W_RANGE)
    hit_count = random.randint(*_GEN_HIT_RANGE)
    distractors = random.randint(*_GEN_DISTRACTOR_RANGE)
    if hit_count + distractors > len(SPRITE_COLORS):
        distractors = len(SPRITE_COLORS) - hit_count
    colors = random.sample(SPRITE_COLORS, hit_count + distractors)

    grid = np.full((H, W), BLACK, dtype=int)
    start_col = random.randint(1, W - 2)
    grid[H - 1, start_col] = BLUE

    if not _place_planned_hits(grid, hit_count, colors[:hit_count], force_vertical):
        return None

    base_order = None
    try:
        _steps, base_order, _sprites, _owner = _trace_current(grid)
    except AssertionError:
        return None

    for color in colors[hit_count:]:
        placed = False
        for _ in range(80):
            trial = grid.copy()
            if not _place_random_distractor(trial, color):
                continue
            try:
                _steps2, order2, _sprites2, _owner2 = _trace_current(trial)
            except AssertionError:
                continue
            if [int(x) for x in order2] == [int(x) for x in base_order]:
                grid[:, :] = trial
                placed = True
                break
        if not placed:
            return None

    if not _semantic_accepts(grid, hit_count, force_vertical):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALLS
    _GEN_CALLS += 1
    force_vertical = (_GEN_CALLS % 3) != 2
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt(force_vertical)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sprite-bounce collection input")
