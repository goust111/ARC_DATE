import random
from collections import Counter, defaultdict, deque

import numpy as np


BG = 0
ANCHOR = 1
MAIN_COLORS = tuple(range(2, 10))
ORIENTS = ("tl", "tr", "bl", "br")
CORNER_TO_ORIENT = {
    (0, 0): "tl",
    (0, -1): "tr",
    (-1, 0): "bl",
    (-1, -1): "br",
}
MAX_GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _transform(arr: np.ndarray, orient: str) -> np.ndarray:
    if orient == "tl":
        return arr.copy()
    if orient == "tr":
        return np.fliplr(arr)
    if orient == "bl":
        return np.flipud(arr)
    if orient == "br":
        return np.rot90(arr, 2)
    raise ValueError(f"unknown orient: {orient}")


def _anchor_corner(pos: tuple[int, int], shape: tuple[int, int]) -> str:
    r, c = pos
    h, w = shape
    if (r, c) == (0, 0):
        return "tl"
    if (r, c) == (0, w - 1):
        return "tr"
    if (r, c) == (h - 1, 0):
        return "bl"
    if (r, c) == (h - 1, w - 1):
        return "br"
    raise ValueError("anchor must sit at a corner")


def _parse_stamp(grid: np.ndarray, bbox: tuple[int, int, int, int], bg: int) -> dict:
    crop = _crop(grid, bbox)
    colors = [int(v) for v in np.unique(crop) if int(v) != bg]
    assert len(colors) == 2, "each stamp must contain exactly anchor + main colors"
    counts = {color: int(np.sum(crop == color)) for color in colors}
    anchor_colors = [color for color, count in counts.items() if count == 1]
    assert len(anchor_colors) == 1, "anchor color must appear exactly once per stamp"
    anchor_color = anchor_colors[0]
    main_color = [color for color in colors if color != anchor_color][0]
    anchor_pos = tuple(map(int, np.argwhere(crop == anchor_color)[0]))
    orient = _anchor_corner(anchor_pos, crop.shape)
    raw_mask = crop == main_color
    norm_mask = _transform(raw_mask, orient)
    return {
        "bbox": bbox,
        "crop": crop,
        "shape": crop.shape,
        "anchor_color": anchor_color,
        "main_color": main_color,
        "anchor_pos": anchor_pos,
        "orient": orient,
        "raw_mask": raw_mask,
        "norm_mask": norm_mask,
    }


def _collect_stamps(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    comps = _components(g != bg)
    stamps = [_parse_stamp(g, _bbox(comp), bg) for comp in comps]
    assert stamps, "expected at least one stamp"
    shapes = {stamp["shape"] for stamp in stamps}
    assert len(shapes) == 1, "all stamp bounding boxes must share one shape"
    return bg, sorted(stamps, key=lambda x: x["bbox"])


def _vote_truth(stamps: list[dict]) -> np.ndarray:
    stack = np.stack([stamp["norm_mask"] for stamp in stamps], axis=0)
    truth = np.sum(stack, axis=0) > (len(stamps) // 2)
    truth[0, 0] = False
    return truth


def _render_stamp(mask: np.ndarray, main_color: int, orient: str, bg: int = BG, anchor_color: int = ANCHOR) -> np.ndarray:
    patch = np.full(mask.shape, bg, dtype=int)
    oriented = _transform(mask, orient)
    patch[oriented] = main_color
    h, w = patch.shape
    anchor_pos = {
        "tl": (0, 0),
        "tr": (0, w - 1),
        "bl": (h - 1, 0),
        "br": (h - 1, w - 1),
    }[orient]
    patch[anchor_pos] = anchor_color
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """Repair each monochrome stamp by anchor-normalized majority voting within its color group.

    Input structure:
    - The background contains several separated single-color stamps with one special anchor pixel each.
    - Every stamp occupies the same rectangular bounding box size and uses exactly one main color plus the anchor color.
    - Stamps sharing a main color belong to one odd-sized group and are transformed copies of one hidden canonical pattern.

    Rule:
    - For each stamp, use the anchor corner to normalize it into the standard orientation where the anchor is at the top-left.
    - Within each main-color group, majority-vote every normalized cell between main color and background to recover the true pattern.
    - Transform that recovered pattern back into each member's original orientation, repaint its box in place, and keep the anchor at its original corner.

    Key insight:
    - The anchor gives each noisy stamp its private coordinate frame, so voting only becomes meaningful after orientation normalization.
    """
    bg, stamps = _collect_stamps(grid)
    groups: dict[int, list[dict]] = defaultdict(list)
    for stamp in stamps:
        groups[stamp["main_color"]].append(stamp)

    out = np.asarray(grid, dtype=int).copy()
    for members in groups.values():
        assert len(members) >= 3 and len(members) % 2 == 1
        truth = _vote_truth(members)
        for stamp in members:
            patch = _render_stamp(
                truth,
                main_color=stamp["main_color"],
                orient=stamp["orient"],
                bg=bg,
                anchor_color=stamp["anchor_color"],
            )
            r0, c0, r1, c1 = stamp["bbox"]
            out[r0 : r1 + 1, c0 : c1 + 1] = patch
    return out


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] or (r, c) == (0, 0):
                continue
            if any(0 <= nr < h and 0 <= nc < w and mask[nr, nc] for nr, nc in _neighbors4(r, c)):
                cells.append((r, c))
    return cells


def _frontier_mask(mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    for r, c in _frontier(mask):
        out[r, c] = True
    return out


def _gen_truth_mask(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(200):
        bend = random.randint(2, w - 3)
        protected = np.zeros((h, w), dtype=bool)
        protected[1, 0 : bend + 1] = True
        protected[0:h, bend] = True
        protected[h - 1, bend:w] = True
        if h >= 5 and random.random() < 0.7:
            branch_col = random.randint(max(1, bend - 2), min(w - 2, bend + 1))
            branch_top = random.randint(1, h - 3)
            protected[branch_top : h - 1, branch_col] = True
        truth = protected.copy()
        candidates = _frontier(protected)
        random.shuffle(candidates)
        keep_prob = random.uniform(0.35, 0.7)
        for r, c in candidates:
            if random.random() < keep_prob:
                truth[r, c] = True
        truth[0, 0] = False
        if truth.sum() < max(10, (h * w) // 3):
            continue
        if truth.sum() > h * w - 4:
            continue
        if np.array_equal(truth, np.fliplr(truth)):
            continue
        if np.array_equal(truth, np.flipud(truth)):
            continue
        if np.array_equal(truth, np.rot90(truth, 2)):
            continue
        if not _mask_connected(truth):
            continue
        if not np.any(~truth):
            continue
        if not np.any(truth & ~protected):
            continue
        if not np.any(_frontier_mask(protected) & ~truth):
            continue
        return truth, protected
    raise RuntimeError("failed to generate canonical truth mask")


def _pick_orientations(group_size: int) -> list[str]:
    if group_size == 3:
        if random.random() < 0.5:
            return random.sample(list(ORIENTS), 3)
        base = random.sample(list(ORIENTS), 2)
        return [base[0], base[1], random.choice(base)]
    order = list(ORIENTS) + [random.choice(ORIENTS)]
    random.shuffle(order)
    return order


def _corrupt_members(truth: np.ndarray, protected: np.ndarray, count: int) -> list[np.ndarray]:
    members = [truth.copy() for _ in range(count)]
    removable = [tuple(map(int, rc)) for rc in np.argwhere(truth & ~protected)]
    addable = [cell for cell in _frontier(protected) if not truth[cell]]
    assert removable and addable

    random.shuffle(removable)
    random.shuffle(addable)
    remove_sites = removable[: random.randint(1, min(len(removable), max(2, count + 1)))]
    add_sites = addable[: random.randint(1, min(len(addable), max(2, count + 1)))]

    changed_members = set()
    for r, c in remove_sites:
        k = random.randint(1, count // 2)
        for idx in random.sample(range(count), k):
            members[idx][r, c] = False
            changed_members.add(idx)

    for r, c in add_sites:
        k = random.randint(1, count // 2)
        for idx in random.sample(range(count), k):
            members[idx][r, c] = True
            changed_members.add(idx)

    if len(changed_members) < 2:
        idxs = random.sample(range(count), 2)
        r, c = remove_sites[0]
        members[idxs[0]][r, c] = False
        r, c = add_sites[0]
        members[idxs[1]][r, c] = True

    for mask in members:
        mask[0, 0] = False
        if not _mask_connected(mask):
            return []
    return members


def _layout_slots(h: int, w: int, total: int) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    step_r = h + 1
    step_c = w + 1
    candidates = []
    for rows in range(1, total + 1):
        cols = (total + rows - 1) // rows
        if 1 + rows * step_r <= 30 and 1 + cols * step_c <= 30:
            candidates.append((rows, cols))
    rows, cols = random.choice(candidates)
    slots = [(1 + r * step_r, 1 + c * step_c) for r in range(rows) for c in range(cols)]
    chosen = random.sample(slots, total)
    max_r = max(r for r, _ in chosen) + h + random.randint(0, 1)
    max_c = max(c for _, c in chosen) + w + random.randint(0, 1)
    return chosen, (max_r + 1, max_c + 1)


def _build_example(specs: list[dict], h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    total = sum(len(spec["members"]) for spec in specs)
    slots, shape = _layout_slots(h, w, total)
    inp = np.full(shape, BG, dtype=int)
    out = np.full(shape, BG, dtype=int)
    items = []
    for spec in specs:
        for orient, member_mask in zip(spec["orients"], spec["members"]):
            items.append((spec["main_color"], spec["truth"], orient, member_mask))
    random.shuffle(items)
    for (r0, c0), (main_color, truth, orient, member_mask) in zip(slots, items):
        inp[r0 : r0 + h, c0 : c0 + w] = _render_stamp(member_mask, main_color, orient)
        out[r0 : r0 + h, c0 : c0 + w] = _render_stamp(truth, main_color, orient)
    return inp, out


def _validate_example(inp: np.ndarray, out: np.ndarray, truths: dict[int, np.ndarray]) -> bool:
    if inp.ndim != 2 or out.ndim != 2 or max(inp.shape) > 30 or max(out.shape) > 30:
        return False
    try:
        bg, stamps = _collect_stamps(inp)
    except Exception:
        return False
    if bg != BG or not np.array_equal(main(inp), out):
        return False

    groups: dict[int, list[dict]] = defaultdict(list)
    for stamp in stamps:
        groups[stamp["main_color"]].append(stamp)

    for color, members in groups.items():
        if len(members) not in (3, 5) or len(members) % 2 == 0:
            return False
        if len({m["orient"] for m in members}) < 2:
            return False
        truth = truths[color]
        if not np.array_equal(_vote_truth(members), truth):
            return False
        raw_vote = np.sum(np.stack([m["raw_mask"] for m in members], axis=0), axis=0) > (len(members) // 2)
        if np.array_equal(raw_vote, truth):
            return False
        norm_masks = [m["norm_mask"] for m in members]
        has_missing = any(np.any(truth & ~mask) for mask in norm_masks)
        has_extra = any(np.any((~truth) & mask) for mask in norm_masks)
        if not (has_missing and has_extra):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(5, 7)
    w = random.randint(6, 8)
    if h == w:
        w = 8 if w < 8 else 6

    size_patterns = random.choice(([3, 3], [3, 5], [5, 3], [3, 3, 3]))
    main_colors = random.sample(MAIN_COLORS, len(size_patterns))
    specs = []
    truth_signatures = set()
    truths: dict[int, np.ndarray] = {}

    for group_size, main_color in zip(size_patterns, main_colors):
        truth, protected = _gen_truth_mask(h, w)
        signature = tuple(map(tuple, truth.astype(int)))
        if signature in truth_signatures:
            return None
        truth_signatures.add(signature)
        members = _corrupt_members(truth, protected, group_size)
        if not members:
            return None
        specs.append(
            {
                "main_color": main_color,
                "truth": truth,
                "members": members,
                "orients": _pick_orientations(group_size),
            }
        )
        truths[main_color] = truth

    try:
        inp, out = _build_example(specs, h, w)
    except Exception:
        return None
    return inp if _validate_example(inp, out, truths) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_GEN_ATTEMPTS} attempts")
