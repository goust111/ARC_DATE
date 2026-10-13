from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
FOREGROUND_COLORS = tuple(range(1, 10))
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _foreground_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            comps.append({"coords": arr, "bbox": _bbox_coords(arr), "area": int(arr.shape[0])})
    return comps


def _parse_domino(grid: np.ndarray, comp: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = comp["bbox"]
    if r0 == r1 and c1 == c0 + 1:
        new, old = int(grid[r0, c0]), int(grid[r0, c1])
    elif c0 == c1 and r1 == r0 + 1:
        new, old = int(grid[r0, c0]), int(grid[r1, c0])
    else:
        raise AssertionError("legend entries must be isolated horizontal or vertical dominos")
    assert new != old, "legend domino colors must differ"
    return new, old


def _resolve_head(color: int, mapping: dict[int, int]) -> int:
    cur = int(color)
    seen: set[int] = set()
    while cur in mapping:
        assert cur not in seen, "legend replacement graph must be acyclic"
        seen.add(cur)
        cur = int(mapping[cur])
    return cur


def _extract_problem(grid: np.ndarray) -> tuple[dict, np.ndarray, dict[int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    comps = _foreground_components(g, bg)
    assert comps, "expected one target rectangle and at least one legend domino"

    target_candidates = []
    legends = []
    for comp in comps:
        area = int(comp["area"])
        bbox_area = _bbox_area(comp["bbox"])
        if area == 2 and bbox_area == 2:
            legends.append(comp)
        elif area > 2 and area == bbox_area:
            target_candidates.append(comp)
        else:
            raise AssertionError("all non-background components must be the target rectangle or a legend domino")

    assert len(target_candidates) == 1, "expected exactly one solid rectangular target"
    assert legends, "expected at least one legend domino"

    target = target_candidates[0]
    crop = _crop_bbox(g, target["bbox"])
    assert int(target["area"]) == int(crop.size), "target rectangle must fill its bbox"
    assert len(set(map(int, crop.ravel()))) >= 2, "target must be multicolored"

    mapping: dict[int, int] = {}
    new_used: set[int] = set()
    for comp in legends:
        new, old = _parse_domino(g, comp)
        assert old not in mapping, "each old color must appear in at most one legend"
        assert new not in new_used, "legend graph must be a disjoint union of chains"
        mapping[old] = new
        new_used.add(new)

    for old in mapping:
        _resolve_head(old, mapping)

    return target, crop, mapping


def _chain_dominos(nodes: list[int]) -> list[tuple[int, int]]:
    return [(int(nodes[i]), int(nodes[i + 1])) for i in range(len(nodes) - 1)]


def _one_step_replace(crop: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    out = crop.copy()
    for old, new in mapping.items():
        out[crop == int(old)] = int(new)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Crop the rectangular target and collapse every legend-linked color chain to its chain head.

    Input structure:
    - A dominant background separates one solid rectangular multicolor target from several isolated two-cell legend dominos.
    - Each horizontal domino reads left=new, right=old; each vertical domino reads top=new, bottom=old.

    Rule:
    - Output the tight bbox crop of the target rectangle.
    - Interpret every domino as an old->new replacement edge and follow repeated replacements until a color reaches a chain head that appears only as new.
    - Recolor every target pixel whose color lies on a legend chain to that chain head; keep target colors outside all chains unchanged.

    Key insight:
    - The dominos are not a flat one-step lookup table: intermediate colors are only transit points, so the target colors must be chased through the full replacement chain.
    """
    _target, crop, mapping = _extract_problem(np.asarray(grid, dtype=int))
    original = crop.copy()
    out = crop.copy()
    for color in np.unique(original):
        head = _resolve_head(int(color), mapping)
        out[original == int(color)] = head
    return out


def _choose_specs() -> tuple[list[dict], list[dict], list[int]]:
    colors = list(FOREGROUND_COLORS)
    random.shuffle(colors)
    cursor = 0

    def take(n: int) -> list[int]:
        nonlocal cursor
        picked = colors[cursor : cursor + n]
        cursor += n
        return [int(v) for v in picked]

    relevant = [{"nodes": take(random.choice([4, 5, 6])), "head_present": bool(random.getrandbits(1))}]

    remaining = len(colors) - cursor
    if remaining >= 3 and random.random() < 0.55:
        feasible = [span for span in (2, 3) if span <= remaining - 1]
        if feasible:
            relevant.append({"nodes": take(random.choice(feasible)), "head_present": bool(random.getrandbits(1))})

    remaining = len(colors) - cursor
    distractors: list[dict] = []
    if remaining >= 3 and random.random() < 0.70:
        feasible = [span for span in (2, 3) if span <= remaining - 1]
        if feasible:
            distractors.append({"nodes": take(random.choice(feasible))})

    remaining = len(colors) - cursor
    untouched_count = random.randint(1, min(2, remaining))
    untouched = take(untouched_count)
    return relevant, distractors, untouched


def _grow_blob(free: np.ndarray, size: int) -> list[tuple[int, int]] | None:
    starts = np.argwhere(free)
    if starts.size == 0:
        return None

    start_r, start_c = random.choice(starts.tolist())
    coords = [(int(start_r), int(start_c))]
    used = {coords[0]}
    free[int(start_r), int(start_c)] = False

    while len(coords) < size:
        frontier: list[tuple[int, int]] = []
        for rr, cc in coords:
            for nr, nc in _neighbors4(rr, cc):
                if 0 <= nr < free.shape[0] and 0 <= nc < free.shape[1] and free[nr, nc] and (nr, nc) not in used:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        nr, nc = random.choice(frontier)
        coords.append((int(nr), int(nc)))
        used.add((int(nr), int(nc)))
        free[int(nr), int(nc)] = False
    return coords


def _build_target(relevant: list[dict], untouched: list[int]) -> tuple[np.ndarray, int] | None:
    base = int(untouched[0])
    chain_target_colors: list[int] = []
    chain_colors_in_target: set[int] = set()
    for chain in relevant:
        nodes = [int(v) for v in chain["nodes"]]
        start = 0 if chain["head_present"] else 1
        target_colors = nodes[start:]
        chain["target_colors"] = target_colors
        chain_target_colors.extend(target_colors)
        chain_colors_in_target.update(target_colors)

    target_colors = [base, *chain_target_colors, *[int(v) for v in untouched[1:]]]
    blob_colors = [color for color in target_colors if color != base]
    min_sizes = [3 if color in chain_colors_in_target else 4 for color in blob_colors]
    min_blob_sum = sum(min_sizes)

    choices = []
    for h in range(6, 11):
        for w in range(6, 11):
            area = h * w
            base_min = max(8, area // 3)
            if area - base_min >= min_blob_sum:
                choices.append((h, w))
    if not choices:
        return None

    h, w = random.choice(choices)
    area = h * w
    base_min = max(8, area // 3)
    extra_limit = min(area - base_min, min_blob_sum + max(8, area // 4))
    blob_total = random.randint(min_blob_sum, extra_limit)

    sizes = min_sizes[:]
    for _ in range(blob_total - min_blob_sum):
        sizes[random.randrange(len(sizes))] += 1

    target = np.full((h, w), base, dtype=int)
    free = np.ones((h, w), dtype=bool)
    items = list(zip(blob_colors, sizes))
    random.shuffle(items)
    items.sort(key=lambda item: item[1], reverse=True)
    for color, size in items:
        coords = _grow_blob(free, int(size))
        if coords is None:
            return None
        rr, cc = np.array(coords, dtype=int).T
        target[rr, cc] = int(color)
    return target, base


def _place_patch(canvas: np.ndarray, blocked: np.ndarray, patch: np.ndarray, margin: int = 1) -> bool:
    h, w = canvas.shape
    ph, pw = patch.shape
    spots: list[tuple[int, int]] = []
    for r0 in range(margin, h - ph - margin + 1):
        for c0 in range(margin, w - pw - margin + 1):
            rr0 = max(0, r0 - 1)
            rr1 = min(h, r0 + ph + 1)
            cc0 = max(0, c0 - 1)
            cc1 = min(w, c0 + pw + 1)
            if blocked[rr0:rr1, cc0:cc1].any():
                continue
            spots.append((r0, c0))
    if not spots:
        return False
    r0, c0 = random.choice(spots)
    canvas[r0 : r0 + ph, c0 : c0 + pw] = patch
    blocked[max(0, r0 - 1) : min(h, r0 + ph + 1), max(0, c0 - 1) : min(w, c0 + pw + 1)] = True
    return True


def _build_input_grid(target: np.ndarray, domino_pairs: list[tuple[int, int]]) -> np.ndarray | None:
    min_h = max(18, int(target.shape[0]) + 5)
    min_w = max(18, int(target.shape[1]) + 5)
    h = random.randint(min_h, 30)
    w = random.randint(min_w, 30)
    canvas = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    if not _place_patch(canvas, blocked, target):
        return None

    if len(domino_pairs) == 1:
        orientations = [random.choice(["h", "v"])]
    else:
        orientations = ["h", "v"] + [random.choice(["h", "v"]) for _ in range(len(domino_pairs) - 2)]
        random.shuffle(orientations)

    patches: list[np.ndarray] = []
    shuffled_pairs = domino_pairs[:]
    random.shuffle(shuffled_pairs)
    for (new, old), orient in zip(shuffled_pairs, orientations):
        if orient == "h":
            patches.append(np.array([[new, old]], dtype=int))
        else:
            patches.append(np.array([[new], [old]], dtype=int))

    for patch in patches:
        if not _place_patch(canvas, blocked, patch):
            return None
    return canvas


def _expected_from_target(target: np.ndarray, relevant: list[dict]) -> tuple[np.ndarray, dict[int, int]]:
    mapping: dict[int, int] = {}
    for chain in relevant:
        for new, old in _chain_dominos(chain["nodes"]):
            mapping[int(old)] = int(new)

    out = target.copy()
    original = target.copy()
    for color in np.unique(original):
        out[original == int(color)] = _resolve_head(int(color), mapping)
    return out, mapping


def _semantic_check(
    grid: np.ndarray,
    expected: np.ndarray,
    relevant: list[dict],
    distractors: list[dict],
    untouched: list[int],
) -> bool:
    try:
        _target, crop, mapping = _extract_problem(grid)
        actual = main(grid)
    except Exception:
        return False

    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(actual, crop):
        return False
    if np.array_equal(_one_step_replace(crop, mapping), actual):
        return False

    all_chain_colors = {color for chain in relevant + distractors for color in chain["nodes"]}
    target_colors = set(map(int, crop.ravel()))
    if not any(color not in all_chain_colors for color in target_colors):
        return False
    if int(untouched[0]) not in target_colors or int(untouched[0]) not in set(map(int, actual.ravel())):
        return False

    if not any(len(chain["nodes"]) >= 4 for chain in relevant):
        return False

    for chain in distractors:
        if any(int(old) in target_colors for old in chain["nodes"][1:]):
            return False

    return int((grid == BG).sum()) > max(int((grid == c).sum()) for c in np.unique(grid) if int(c) != BG)


def _gen_attempt() -> np.ndarray | None:
    relevant, distractors, untouched = _choose_specs()
    built = _build_target(relevant, untouched)
    if built is None:
        return None
    target, _base = built

    expected, _mapping = _expected_from_target(target, relevant)
    domino_pairs = [pair for chain in relevant for pair in _chain_dominos(chain["nodes"])]
    domino_pairs.extend(pair for chain in distractors for pair in _chain_dominos(chain["nodes"]))

    grid = _build_input_grid(target, domino_pairs)
    if grid is None:
        return None
    if not _semantic_check(grid, expected, relevant, distractors, untouched):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
