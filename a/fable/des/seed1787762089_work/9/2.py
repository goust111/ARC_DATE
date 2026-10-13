from __future__ import annotations

import itertools
import random
from collections import Counter, deque

import numpy as np


BG = 0
CARD = 5
MARK_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 500
TRANSFORMS = [(False, k) for k in range(4)] + [(True, k) for k in range(4)]


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _apply_transform(arr: np.ndarray, op: int) -> np.ndarray:
    flip, rot = TRANSFORMS[op]
    out = np.fliplr(arr) if flip else arr
    return np.rot90(out, rot)


def _rect_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _non_collinear(points: list[tuple[int, int]]) -> bool:
    for (r1, c1), (r2, c2), (r3, c3) in itertools.combinations(points, 3):
        if (r2 - r1) * (c3 - c1) != (c2 - c1) * (r3 - r1):
            return True
    return False


def _find_cards(grid: np.ndarray, bg: int) -> tuple[list[dict], int]:
    comps = _components4(grid != bg)
    groups: dict[tuple[int, int], list[dict]] = {}
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        area = len(comp)
        if area != h * w or area == 1:
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        vals, counts = np.unique(sub, return_counts=True)
        dom = int(vals[np.argmax(counts)])
        marks = area - int(np.max(counts))
        groups.setdefault((h, w), []).append(
            {
                "bbox": (r0, c0, r1, c1),
                "dims": (h, w),
                "dominant": dom,
                "marks": marks,
                "sub": sub.copy(),
            }
        )

    candidates: list[tuple[int, int, int, list[dict]]] = []
    for dims, group in groups.items():
        if len(group) < 2:
            continue
        by_dom = Counter(item["dominant"] for item in group)
        dom, _ = by_dom.most_common(1)[0]
        same = [item for item in group if item["dominant"] == dom and 1 <= item["marks"] <= 3]
        if len(same) >= 2:
            total_marks = sum(item["marks"] for item in same)
            candidates.append((len(same), total_marks, dims[0] * dims[1], same))

    assert candidates, "failed to identify the repeated card rectangles"
    candidates.sort(reverse=True)
    cards = candidates[0][3]
    card_color = int(cards[0]["dominant"])
    return cards, card_color


def _merge_template(cards: list[dict], card_color: int) -> np.ndarray:
    h, w = cards[0]["dims"]
    template = np.full((h, w), card_color, dtype=int)
    seen: dict[tuple[int, int], int] = {}
    for card in cards:
        sub = card["sub"]
        for r in range(h):
            for c in range(w):
                val = int(sub[r, c])
                if val == card_color:
                    continue
                key = (r, c)
                if key in seen:
                    assert seen[key] == val, "card marks conflict at the same relative coordinate"
                seen[key] = val
                template[r, c] = val
    assert 4 <= len(seen) <= 6, "merged template should contain 4 to 6 marks"
    return template


def _extract_fragment_pixels(grid: np.ndarray, bg: int, cards: list[dict]) -> list[tuple[int, int, int]]:
    mask = np.ones(grid.shape, dtype=bool)
    for card in cards:
        r0, c0, r1, c1 = card["bbox"]
        mask[r0 : r1 + 1, c0 : c1 + 1] = False
    pts = np.argwhere((grid != bg) & mask)
    return [(int(r), int(c), int(grid[r, c])) for r, c in pts]


def _find_stamp_candidates(
    grid: np.ndarray,
    bg: int,
    template: np.ndarray,
    card_color: int,
    fragments: list[tuple[int, int, int]],
) -> list[dict]:
    h, w = grid.shape
    candidates: list[dict] = []
    seen = set()
    for op in range(8):
        oriented = _apply_transform(template, op)
        th, tw = oriented.shape
        marks = [(r, c, int(oriented[r, c])) for r in range(th) for c in range(tw) if int(oriented[r, c]) != card_color]
        for fr, fc, color in fragments:
            for mr, mc, mark_color in marks:
                if color != mark_color:
                    continue
                top = fr - mr
                left = fc - mc
                key = (top, left, oriented.shape, oriented.tobytes())
                if key in seen:
                    continue
                seen.add(key)
                if top < 0 or left < 0 or top + th > h or left + tw > w:
                    continue
                sub = grid[top : top + th, left : left + tw]
                existing = np.argwhere(sub != bg)
                covered: set[tuple[int, int]] = set()
                valid = True
                for rr, cc in existing:
                    val = int(sub[rr, cc])
                    want = int(oriented[rr, cc])
                    if want == card_color or want != val:
                        valid = False
                        break
                    covered.add((top + int(rr), left + int(cc)))
                if not valid or not (3 <= len(covered) <= 4):
                    continue
                candidates.append(
                    {
                        "top": top,
                        "left": left,
                        "rect": (top, left, top + th - 1, left + tw - 1),
                        "array": oriented.copy(),
                        "covered": frozenset(covered),
                    }
                )
    return candidates


def _select_stamps(candidates: list[dict], fragment_positions: set[tuple[int, int]]) -> list[dict]:
    solutions: list[tuple[dict, ...]] = []
    for cand in candidates:
        if cand["covered"] == fragment_positions:
            solutions.append((cand,))
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            a = candidates[i]
            b = candidates[j]
            if a["covered"] & b["covered"]:
                continue
            if (a["covered"] | b["covered"]) != fragment_positions:
                continue
            if _rect_overlap(a["rect"], b["rect"]):
                continue
            solutions.append((a, b))
    dedup: dict[tuple, tuple[dict, ...]] = {}
    for sol in solutions:
        key = tuple(
            sorted(
                (
                    item["top"],
                    item["left"],
                    item["array"].shape,
                    item["array"].tobytes(),
                )
                for item in sol
            )
        )
        dedup[key] = sol
    assert len(dedup) == 1, "fragments must admit a unique stamp placement set"
    return list(next(iter(dedup.values())))


def _recover_stamps(grid: np.ndarray) -> tuple[np.ndarray, list[dict], int]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    cards, card_color = _find_cards(g, bg)
    template = _merge_template(cards, card_color)
    fragments = _extract_fragment_pixels(g, bg, cards)
    fragment_positions = {(r, c) for r, c, _ in fragments}
    candidates = _find_stamp_candidates(g, bg, template, card_color, fragments)
    stamps = _select_stamps(candidates, fragment_positions)
    return template, stamps, card_color


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    _, stamps, _ = _recover_stamps(g)
    out = g.copy()
    for stamp in stamps:
        top = int(stamp["top"])
        left = int(stamp["left"])
        arr = stamp["array"]
        h, w = arr.shape
        out[top : top + h, left : left + w] = arr
    return out


def _compositions(total: int, parts: int, lo: int, hi: int) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []

    def rec(rem: int, idx: int, acc: list[int]) -> None:
        if idx == parts:
            if rem == 0:
                out.append(tuple(acc))
            return
        left = parts - idx - 1
        for val in range(lo, hi + 1):
            if rem - val < left * lo or rem - val > left * hi:
                continue
            acc.append(val)
            rec(rem - val, idx + 1, acc)
            acc.pop()

    rec(total, 0, [])
    return out


def _sample_mark_positions(rng: random.Random, h: int, w: int, count: int) -> list[tuple[int, int]] | None:
    cells = [(r, c) for r in range(h) for c in range(w)]
    rng.shuffle(cells)
    marks: list[tuple[int, int]] = []
    for r, c in cells:
        if any(abs(r - rr) + abs(c - cc) == 1 for rr, cc in marks):
            continue
        marks.append((r, c))
        if len(marks) == count:
            break
    if len(marks) != count:
        return None
    return marks


def _eligible_subsets(
    positions: list[tuple[int, int]],
    card_ids: list[int],
    total_marks: int,
) -> list[tuple[int, ...]]:
    subsets: list[tuple[int, ...]] = []
    sizes = [3, 4]
    if total_marks == 4:
        sizes = [3]
    for size in sizes:
        if size > total_marks:
            continue
        for combo in itertools.combinations(range(total_marks), size):
            pts = [positions[i] for i in combo]
            if not _non_collinear(pts):
                continue
            if len({card_ids[i] for i in combo}) < 2:
                continue
            subsets.append(combo)
    return subsets


def _free_positions(
    occupied: np.ndarray,
    box_h: int,
    box_w: int,
    margin: int,
) -> list[tuple[int, int]]:
    h, w = occupied.shape
    locs: list[tuple[int, int]] = []
    for top in range(h - box_h + 1):
        for left in range(w - box_w + 1):
            r0 = max(0, top - margin)
            c0 = max(0, left - margin)
            r1 = min(h, top + box_h + margin)
            c1 = min(w, left + box_w + margin)
            if occupied[r0:r1, c0:c1].any():
                continue
            locs.append((top, left))
    return locs


def _generate_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    h = rng.randint(4, 6)
    w = rng.randint(5, 7)
    if rng.random() < 0.4:
        h, w = w, h
    if h == w:
        w += 1

    num_cards = rng.choice([2, 3])
    total_marks = rng.randint(max(4, num_cards), 6)
    partitions = _compositions(total_marks, num_cards, 1, 3)
    if not partitions:
        return None

    mark_positions = _sample_mark_positions(rng, h, w, total_marks)
    if mark_positions is None:
        return None
    mark_colors = rng.sample(MARK_COLORS, total_marks)

    counts = list(rng.choice(partitions))
    rng.shuffle(counts)
    order = list(range(total_marks))
    rng.shuffle(order)
    card_ids = [-1] * total_marks
    idx = 0
    for card_id, count in enumerate(counts):
        for _ in range(count):
            card_ids[order[idx]] = card_id
            idx += 1

    eligible = _eligible_subsets(mark_positions, card_ids, total_marks)
    if not eligible:
        return None

    num_clusters = rng.choice([1, 2])
    rng.shuffle(eligible)
    if num_clusters == 2:
        distinct = []
        seen = set()
        for item in eligible:
            if item not in seen:
                seen.add(item)
                distinct.append(item)
            if len(distinct) == 2:
                break
        if len(distinct) < 2:
            return None
        subsets = distinct
    else:
        subsets = [eligible[0]]

    template = np.full((h, w), CARD, dtype=int)
    for (r, c), color in zip(mark_positions, mark_colors):
        template[r, c] = color

    card_arrays = [np.full((h, w), CARD, dtype=int) for _ in range(num_cards)]
    for mark_idx, ((r, c), color) in enumerate(zip(mark_positions, mark_colors)):
        card_arrays[card_ids[mark_idx]][r, c] = color

    oriented_templates = []
    for subset in subsets:
        op = rng.randrange(8)
        arr = _apply_transform(template, op)
        color_pos = {
            int(arr[r, c]): (r, c)
            for r in range(arr.shape[0])
            for c in range(arr.shape[1])
            if int(arr[r, c]) != CARD
        }
        oriented_templates.append(
            {
                "array": arr,
                "subset": subset,
                "positions": [(color_pos[mark_colors[i]][0], color_pos[mark_colors[i]][1], mark_colors[i]) for i in subset],
            }
        )

    grid_h = rng.randint(22, 28)
    grid_w = rng.randint(24, 30)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    card_boxes = []
    for _ in range(num_cards):
        locs = _free_positions(occupied, h, w, margin=1)
        if not locs:
            return None
        top, left = rng.choice(locs)
        occupied[top : top + h, left : left + w] = True
        card_boxes.append((top, left))

    stamp_boxes = []
    for oriented in oriented_templates:
        oh, ow = oriented["array"].shape
        locs = _free_positions(occupied, oh, ow, margin=1)
        if not locs:
            return None
        top, left = rng.choice(locs)
        occupied[top : top + oh, left : left + ow] = True
        stamp_boxes.append((top, left))

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for (top, left), arr in zip(card_boxes, card_arrays):
        grid[top : top + h, left : left + w] = arr

    for (top, left), oriented in zip(stamp_boxes, oriented_templates):
        for rr, cc, color in oriented["positions"]:
            grid[top + rr, left + cc] = color

    expected = grid.copy()
    for (top, left), oriented in zip(stamp_boxes, oriented_templates):
        arr = oriented["array"]
        oh, ow = arr.shape
        expected[top : top + oh, left : left + ow] = arr

    try:
        template2, stamps2, card_color = _recover_stamps(grid)
    except AssertionError:
        return None
    if card_color != CARD or not np.array_equal(template2, template):
        return None

    found = {
        (int(item["top"]), int(item["left"]), item["array"].shape, item["array"].tobytes())
        for item in stamps2
    }
    wanted = {
        (top, left, oriented["array"].shape, oriented["array"].tobytes())
        for (top, left), oriented in zip(stamp_boxes, oriented_templates)
    }
    if found != wanted:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        result = _generate_case(rng)
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate a valid sample")
