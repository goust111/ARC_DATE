from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
DIRS = ("up", "down", "left", "right")
SHAPES = ("rect", "L", "T")
MAX_GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[counts.argmax()])


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _rotations(mask: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rotated = np.rot90(mask, k)
        key = (tuple(int(x) for x in rotated.shape), rotated.tobytes())
        if key in seen:
            continue
        seen.add(key)
        out.append(rotated)
    return out


def _classify_object(grid: np.ndarray, color: int) -> dict:
    coords = np.argwhere(grid == int(color))
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0 = int(r0)
    c0 = int(c0)
    r1 = int(r1)
    c1 = int(c1)
    patch = (grid[r0 : r1 + 1, c0 : c1 + 1] == int(color))
    h, w = (int(x) for x in patch.shape)
    area = int(patch.sum())

    if area == h * w - 1:
        missing = np.argwhere(~patch)
        if len(missing) == 1:
            mr, mc = (int(x) for x in missing[0])
            direction = None
            if mr == 0 and w % 2 == 1 and mc == w // 2:
                direction = "up"
            elif mr == h - 1 and w % 2 == 1 and mc == w // 2:
                direction = "down"
            elif mc == 0 and h % 2 == 1 and mr == h // 2:
                direction = "left"
            elif mc == w - 1 and h % 2 == 1 and mr == h // 2:
                direction = "right"
            if direction is not None:
                if direction in {"up", "down"}:
                    target_shape = (h - 1, w)
                else:
                    target_shape = (h, w - 1)
                return {
                    "kind": "envelope",
                    "color": int(color),
                    "bbox": (r0, c0, r1, c1),
                    "direction": direction,
                    "target_shape": tuple(int(x) for x in target_shape),
                }

    return {
        "kind": "paper",
        "color": int(color),
        "bbox": (r0, c0, r1, c1),
        "mask": _tight_mask(patch),
    }


def _place_paper(out: np.ndarray, env: dict, mask: np.ndarray, color: int) -> None:
    r0, c0, r1, c1 = env["bbox"]
    direction = env["direction"]
    h, w = (int(x) for x in mask.shape)
    if direction == "up":
        rr, cc = r0 + 1, c0
    elif direction == "down":
        rr, cc = r0, c0
    elif direction == "left":
        rr, cc = r0, c0 + 1
    else:
        rr, cc = r0, c0
    region = out[rr : rr + h, cc : cc + w]
    region[mask] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pack each scattered paper into the uniquely matching envelope.

    Envelopes are solid rectangles with one missing border-center pixel.
    A paper matches an envelope when some rotation of its tight mask has the
    envelope's target bounding-box shape: the envelope bbox with the gap side's
    full row/column removed. The rotated paper is then placed flush against the
    side opposite the gap, leaving the gap visible.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(_get_bg(g))
    assert bg == BG

    colors = [int(v) for v in np.unique(g) if int(v) != bg]
    envelopes: list[dict] = []
    papers: list[dict] = []
    for color in colors:
        obj = _classify_object(g, color)
        if obj["kind"] == "envelope":
            envelopes.append(obj)
        else:
            papers.append(obj)

    assert envelopes
    assert len(envelopes) == len(papers)

    out = g.copy()
    for paper in papers:
        out[g == int(paper["color"])] = bg

    used_papers: set[int] = set()
    for env in envelopes:
        target_shape = tuple(int(x) for x in env["target_shape"])
        match = None
        placed_mask = None
        for idx, paper in enumerate(papers):
            if idx in used_papers:
                continue
            for rotated in _rotations(paper["mask"]):
                if tuple(int(x) for x in rotated.shape) == target_shape:
                    match = idx
                    placed_mask = rotated
                    break
            if match is not None:
                break
        assert match is not None and placed_mask is not None
        used_papers.add(match)
        _place_paper(out, env, placed_mask, papers[match]["color"])

    return out


def _rect_mask(h: int, w: int) -> np.ndarray:
    return np.ones((int(h), int(w)), dtype=bool)


def _l_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((int(h), int(w)), dtype=bool)
    mask[0, :] = True
    mask[:, 0] = True
    return mask


def _t_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((int(h), int(w)), dtype=bool)
    if w >= h:
        mask[0, :] = True
        mask[:, w // 2] = True
    else:
        mask[:, 0] = True
        mask[h // 2, :] = True
    return mask


def _target_mask(shape_kind: str, h: int, w: int) -> np.ndarray:
    if shape_kind == "rect":
        return _rect_mask(h, w)
    if shape_kind == "L":
        return _l_mask(h, w)
    if shape_kind == "T":
        return _t_mask(h, w)
    raise ValueError(shape_kind)


def _envelope_patch(color: int, direction: str, slot_h: int, slot_w: int) -> np.ndarray:
    if direction in {"up", "down"}:
        patch = np.full((int(slot_h) + 1, int(slot_w)), int(color), dtype=int)
        row = 0 if direction == "up" else patch.shape[0] - 1
        patch[row, patch.shape[1] // 2] = BG
        return patch

    patch = np.full((int(slot_h), int(slot_w) + 1), int(color), dtype=int)
    col = 0 if direction == "left" else patch.shape[1] - 1
    patch[patch.shape[0] // 2, col] = BG
    return patch


def _color_patch(color: int, mask: np.ndarray) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[:-1, :] |= mask[1:, :]
    out[1:, :] |= mask[:-1, :]
    out[:, :-1] |= mask[:, 1:]
    out[:, 1:] |= mask[:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    return out


def _find_location(blocked: np.ndarray, mask: np.ndarray) -> tuple[int, int] | None:
    h, w = blocked.shape
    ph, pw = mask.shape
    if ph > h or pw > w:
        return None
    candidates = [
        (r, c)
        for r in range(h - ph + 1)
        for c in range(w - pw + 1)
        if not np.any(blocked[r : r + ph, c : c + pw] & mask)
    ]
    if not candidates:
        return None
    return random.choice(candidates)


def _sample_sizes(direction: str, used_keys: set[tuple[int, int]]) -> tuple[int, int] | None:
    candidates: list[tuple[int, int]] = []
    if direction in {"up", "down"}:
        for h in range(2, 6):
            for w in (3, 5, 7):
                if w > h and tuple(sorted((h, w))) not in used_keys:
                    candidates.append((h, w))
    else:
        for h in (3, 5, 7):
            for w in range(2, 6):
                if h > w and tuple(sorted((h, w))) not in used_keys:
                    candidates.append((h, w))
    if not candidates:
        return None
    return random.choice(candidates)


def _sample_shape_kinds(pair_count: int) -> list[str]:
    if pair_count == 2:
        kinds = random.sample(SHAPES, 2)
    elif pair_count == 3:
        kinds = list(SHAPES)
        random.shuffle(kinds)
    else:
        kinds = list(SHAPES) + [random.choice(SHAPES)]
        random.shuffle(kinds)
    return kinds


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    pair_count = random.randint(2, 4)
    directions = random.sample(DIRS, pair_count)
    shape_kinds = _sample_shape_kinds(pair_count)
    colors = random.sample(range(1, 10), 2 * pair_count)

    pairs: list[dict] = []
    used_size_keys: set[tuple[int, int]] = set()
    for i in range(pair_count):
        direction = directions[i]
        size = _sample_sizes(direction, used_size_keys)
        if size is None:
            return None
        slot_h, slot_w = size
        used_size_keys.add(tuple(sorted(size)))
        target_mask = _target_mask(shape_kinds[i], slot_h, slot_w)
        env_color = colors[2 * i]
        paper_color = colors[2 * i + 1]
        envelope = _envelope_patch(env_color, direction, slot_h, slot_w)
        input_mask = np.rot90(target_mask, random.randrange(4))
        paper = _color_patch(paper_color, input_mask)
        pairs.append(
            {
                "direction": direction,
                "slot_h": int(slot_h),
                "slot_w": int(slot_w),
                "target_mask": target_mask,
                "env_color": int(env_color),
                "paper_color": int(paper_color),
                "envelope": envelope,
                "paper": paper,
            }
        )

    total_area = sum(int((p["envelope"] != BG).sum() + (p["paper"] != BG).sum()) for p in pairs)
    min_side = max(18, int(np.ceil(np.sqrt(total_area * 2.6))))
    max_side = max(min_side, min(30, min_side + 6))
    h = random.randint(min_side, max_side)
    w = random.randint(min_side, max_side)

    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    env_positions: list[tuple[int, int]] = []
    paper_positions: list[tuple[int, int]] = []

    order = list(range(pair_count))
    random.shuffle(order)
    for idx in order:
        env_mask = pairs[idx]["envelope"] != BG
        loc = _find_location(blocked, env_mask)
        if loc is None:
            return None
        r, c = loc
        patch = pairs[idx]["envelope"]
        grid[r : r + patch.shape[0], c : c + patch.shape[1]][env_mask] = patch[env_mask]
        occ = np.zeros_like(blocked)
        occ[r : r + patch.shape[0], c : c + patch.shape[1]][env_mask] = True
        blocked |= _dilate8(occ)
        env_positions.append((r, c))
    reordered_env_positions = [None] * pair_count
    for pos, idx in enumerate(order):
        reordered_env_positions[idx] = env_positions[pos]
    env_positions = [tuple(int(x) for x in p) for p in reordered_env_positions]

    order = list(range(pair_count))
    random.shuffle(order)
    for idx in order:
        paper_mask = pairs[idx]["paper"] != BG
        loc = _find_location(blocked, paper_mask)
        if loc is None:
            return None
        r, c = loc
        patch = pairs[idx]["paper"]
        grid[r : r + patch.shape[0], c : c + patch.shape[1]][paper_mask] = patch[paper_mask]
        occ = np.zeros_like(blocked)
        occ[r : r + patch.shape[0], c : c + patch.shape[1]][paper_mask] = True
        blocked |= _dilate8(occ)
        paper_positions.append((r, c))
    reordered_paper_positions = [None] * pair_count
    for pos, idx in enumerate(order):
        reordered_paper_positions[idx] = paper_positions[pos]
    paper_positions = [tuple(int(x) for x in p) for p in reordered_paper_positions]

    expected = grid.copy()
    for idx, pair in enumerate(pairs):
        pr, pc = paper_positions[idx]
        paper_mask = pair["paper"] != BG
        expected[pr : pr + paper_mask.shape[0], pc : pc + paper_mask.shape[1]][paper_mask] = BG

    for idx, pair in enumerate(pairs):
        er, ec = env_positions[idx]
        env = {
            "bbox": (
                er,
                ec,
                er + pair["envelope"].shape[0] - 1,
                ec + pair["envelope"].shape[1] - 1,
            ),
            "direction": pair["direction"],
        }
        _place_paper(expected, env, pair["target_mask"], pair["paper_color"])

    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        built = _build_case()
        if built is None:
            continue
        grid, expected = built
        if np.array_equal(main(grid), expected):
            return grid
    raise RuntimeError("failed to generate a valid input")
