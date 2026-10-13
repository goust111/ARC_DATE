from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
MARKER_COLORS = (2, 5, 6)
RECT_COLORS = (1, 3, 4, 7, 8, 9)
N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_ATTEMPTS = 400
LOCAL_ATTEMPTS = 240


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    for dr, dc in N4:
        yield r + dr, c + dc


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_single_component(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    start = tuple(map(int, pts[0]))
    target = {tuple(map(int, rc)) for rc in pts}
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in target and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(target)


def _perimeter_coords(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    out: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        out.add((r0, c))
        out.add((r1, c))
    for r in range(r0, r1 + 1):
        out.add((r, c0))
        out.add((r, c1))
    return out


def _edge_contact(marker_rc: tuple[int, int], perimeter: set[tuple[int, int]]) -> int:
    r, c = marker_rc
    return sum((nr, nc) in perimeter for nr, nc in _neighbors4(r, c))


def _rect_colors_in(grid: np.ndarray) -> list[int]:
    colors = sorted(int(v) for v in np.unique(grid))
    return [c for c in colors if c not in (BG, *MARKER_COLORS)]


def _vote_scores(
    grid: np.ndarray, bbox: tuple[int, int, int, int]
) -> tuple[Counter[int], Counter[int], list[tuple[int, int, int, int]]]:
    perimeter = _perimeter_coords(bbox)
    scores: Counter[int] = Counter()
    raw: Counter[int] = Counter()
    voters: list[tuple[int, int, int, int]] = []
    marker_cells = np.argwhere(np.isin(grid, list(MARKER_COLORS)))
    for r, c in marker_cells:
        rr, cc = int(r), int(c)
        contrib = _edge_contact((rr, cc), perimeter)
        if contrib > 0:
            color = int(grid[rr, cc])
            scores[color] += contrib
            raw[color] += 1
            voters.append((rr, cc, color, contrib))
    return scores, raw, voters


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each incomplete rectangle by filling its black gaps with the marker color that wins the local edge-adjacency vote.

    Input structure:
    - The grid is a black background with 2-4 incomplete single-color rectangles, each using a distinct non-marker color.
    - Marker pixels are isolated single cells in colors red(2), grey(5), or pink(6), placed on a rectangle perimeter cell or one step outside it.
    - Each rectangle has black missing cells inside its own bounding box, while nearby markers stay outside the rectangle color set.

    Rule:
    - For each non-marker rectangle color, find the bounding box of all cells of that color.
    - Collect every marker pixel that shares at least one 4-neighbor edge with the bounding-box perimeter cells.
    - Count, per marker color, the total number of such shared edges; one marker can contribute 1 or 2 edges.
    - Choose the unique marker color with the largest edge-contact total, and fill every black cell inside that rectangle bbox with that color.
    - Keep the original rectangle cells and all marker pixels unchanged.

    Key insight:
    - The fill color does not come from the rectangle itself; it is determined by a strictly local vote weighted by how strongly nearby marker pixels touch that rectangle's bounding-box edge.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D ARC grid"

    out = g.copy()
    rect_colors = _rect_colors_in(g)
    assert rect_colors, "expected non-marker rectangle colors"

    for color in rect_colors:
        coords = np.argwhere(g == color)
        assert coords.size > 0
        bbox = _bbox_from_coords(coords)
        scores, _raw, voters = _vote_scores(g, bbox)
        assert len(voters) >= 3, "each rectangle must have at least three nearby markers"
        assert 2 <= len(scores) <= 3, "each rectangle must use two or three marker colors"
        best = max(scores.values())
        winners = [marker_color for marker_color, score in scores.items() if score == best]
        assert len(winners) == 1, "adjacency vote must have a unique winner"
        fill_color = winners[0]

        r0, c0, r1, c1 = bbox
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[patch == BG] = fill_color

    return out


def _boxes_separated(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int = 1
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return (
        ar1 + margin < br0 - margin
        or br1 + margin < ar0 - margin
        or ac1 + margin < bc0 - margin
        or bc1 + margin < ac0 - margin
    )


def _place_boxes(
    h: int, w: int, sizes: list[tuple[int, int]]
) -> list[tuple[int, int, int, int]] | None:
    sizes = sorted(sizes, key=lambda x: x[0] * x[1], reverse=True)
    placed: list[tuple[int, int, int, int]] = []

    def rec(i: int) -> bool:
        if i == len(sizes):
            return True
        rh, rw = sizes[i]
        positions = [
            (r0, c0)
            for r0 in range(1, h - rh)
            for c0 in range(1, w - rw)
        ]
        random.shuffle(positions)
        for r0, c0 in positions:
            box = (r0, c0, r0 + rh - 1, c0 + rw - 1)
            if any(not _boxes_separated(box, other, margin=1) for other in placed):
                continue
            placed.append(box)
            if rec(i + 1):
                return True
            placed.pop()
        return False

    if rec(0):
        return placed.copy()
    return None


def _perimeter_local(h: int, w: int) -> list[tuple[int, int]]:
    out = []
    for c in range(w):
        out.append((0, c))
        out.append((h - 1, c))
    for r in range(1, h - 1):
        out.append((r, 0))
        out.append((r, w - 1))
    return out


def _halo_local(h: int, w: int) -> list[tuple[int, int]]:
    out = []
    for c in range(w):
        out.append((-1, c))
        out.append((h, c))
    for r in range(h):
        out.append((r, -1))
        out.append((r, w))
    return out


def _orth_adj(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def _pick_positions(
    h: int, w: int, tokens: list[tuple[str, int]]
) -> list[tuple[tuple[int, int], str, int]] | None:
    perimeter = _perimeter_local(h, w)
    halo = _halo_local(h, w)

    for _ in range(80):
        chosen: list[tuple[tuple[int, int], str, int]] = []
        used: list[tuple[int, int]] = []
        ok = True
        for role, weight in sorted(tokens, key=lambda item: -item[1]):
            pool = perimeter if weight == 2 else halo
            candidates = [
                pos for pos in pool if pos not in used and all(not _orth_adj(pos, old) for old in used)
            ]
            if not candidates:
                ok = False
                break
            pos = random.choice(candidates)
            used.append(pos)
            chosen.append((pos, role, weight))
        if ok:
            return chosen
    return None


def _grow_missing_blob(h: int, w: int, allow_edge_touch: bool) -> set[tuple[int, int]] | None:
    area = h * w
    max_size = min(max(3, area // 3), area - 3)
    if max_size < 2:
        return None
    target_size = random.randint(2, max_size)
    all_cells = [(r, c) for r in range(h) for c in range(w)]
    interior = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    starts = all_cells if allow_edge_touch or not interior else interior

    for _ in range(80):
        start = random.choice(starts)
        blob = {start}
        frontier = {nbr for nbr in _neighbors4(*start) if 0 <= nbr[0] < h and 0 <= nbr[1] < w}
        while len(blob) < target_size and frontier:
            frontier_list = list(frontier)
            if not allow_edge_touch:
                inner_frontier = [p for p in frontier_list if 0 < p[0] < h - 1 and 0 < p[1] < w - 1]
                if inner_frontier:
                    frontier_list = inner_frontier
            cell = random.choice(frontier_list)
            frontier.remove(cell)
            blob.add(cell)
            for nbr in _neighbors4(*cell):
                if 0 <= nbr[0] < h and 0 <= nbr[1] < w and nbr not in blob:
                    frontier.add(nbr)
        if len(blob) != target_size:
            continue

        remaining = np.ones((h, w), dtype=bool)
        for r, c in blob:
            remaining[r, c] = False

        if not remaining[0].any() or not remaining[h - 1].any():
            continue
        if not remaining[:, 0].any() or not remaining[:, w - 1].any():
            continue
        if not _is_single_component(remaining):
            continue
        return blob
    return None


def _template_tokens(kind: str, use_three_colors: bool) -> list[tuple[str, int]]:
    if kind == "simple_outside":
        tokens = [("win", 1), ("win", 1), ("lose1", 1)]
        if use_three_colors:
            tokens.append(("lose2", 1))
        return tokens
    if kind == "weighted_tie":
        tokens = [("win", 2), ("win", 1), ("lose1", 1), ("lose1", 1)]
        if use_three_colors:
            tokens.append(("lose2", 1))
        return tokens
    tokens = [("win", 2), ("win", 2), ("lose1", 1), ("lose1", 1), ("lose1", 1)]
    if use_three_colors:
        tokens.append(("lose2", 1))
    return tokens


def _role_colors(use_three_colors: bool) -> dict[str, int]:
    chosen = random.sample(list(MARKER_COLORS), 3 if use_three_colors else 2)
    mapping = {"win": chosen[0], "lose1": chosen[1]}
    if use_three_colors:
        mapping["lose2"] = chosen[2]
    return mapping


def _local_vote_summary(
    h: int, w: int, placements: list[tuple[tuple[int, int], int]]
) -> tuple[Counter[int], Counter[int]]:
    perimeter = set(_perimeter_local(h, w))
    scores: Counter[int] = Counter()
    raw: Counter[int] = Counter()
    for pos, color in placements:
        contrib = _edge_contact(pos, perimeter)
        if contrib > 0:
            scores[color] += contrib
            raw[color] += 1
    return scores, raw


def _raw_majority(raw: Counter[int]) -> int | None:
    if not raw:
        return None
    best = max(raw.values())
    winners = [color for color, count in raw.items() if count == best]
    return winners[0] if len(winners) == 1 else None


def _build_rectangle_patch(
    rect_color: int, h: int, w: int, kind: str
) -> tuple[np.ndarray, list[tuple[tuple[int, int], int]], dict] | None:
    use_three_colors = random.random() < 0.45
    tokens = _template_tokens(kind, use_three_colors)

    for _ in range(LOCAL_ATTEMPTS):
        missing = _grow_missing_blob(h, w, allow_edge_touch=(random.random() < 0.35))
        if missing is None:
            continue
        picked = _pick_positions(h, w, tokens)
        if picked is None:
            continue
        role_map = _role_colors(use_three_colors)
        placements = [(pos, role_map[role]) for pos, role, _weight in picked]

        patch = np.full((h, w), rect_color, dtype=int)
        for r, c in missing:
            patch[r, c] = BG
        for (r, c), color in placements:
            if 0 <= r < h and 0 <= c < w:
                patch[r, c] = color

        rect_mask = patch == rect_color
        if not np.any(patch == BG):
            continue
        if not _is_single_component(rect_mask):
            continue
        if _bbox_from_coords(np.argwhere(rect_mask)) != (0, 0, h - 1, w - 1):
            continue

        scores, raw = _local_vote_summary(h, w, placements)
        if len(scores) < 2 or len(scores) > 3:
            continue
        best = max(scores.values())
        winners = [color for color, score in scores.items() if score == best]
        if len(winners) != 1:
            continue
        weighted_winner = winners[0]
        raw_winner = _raw_majority(raw)
        has_weight2 = any(_edge_contact(pos, set(_perimeter_local(h, w))) == 2 for pos, _color in placements)
        mismatch = raw_winner is not None and raw_winner != weighted_winner
        raw_tie = raw_winner is None

        if kind == "simple_outside" and has_weight2:
            continue
        if kind == "weighted_tie" and (not has_weight2 or not raw_tie):
            continue
        if kind == "weighted_mismatch" and not mismatch:
            continue

        meta = {
            "weighted_winner": weighted_winner,
            "raw_winner": raw_winner,
            "mismatch": mismatch,
            "raw_tie": raw_tie,
            "used_colors": tuple(sorted(scores)),
        }
        return patch, placements, meta
    return None


def _analyze_grid(grid: np.ndarray) -> list[dict] | None:
    rect_colors = _rect_colors_in(grid)
    if not (2 <= len(rect_colors) <= 4):
        return None

    marker_claims: Counter[tuple[int, int]] = Counter()
    infos: list[dict] = []
    for rect_color in rect_colors:
        coords = np.argwhere(grid == rect_color)
        if coords.size == 0:
            return None
        bbox = _bbox_from_coords(coords)
        r0, c0, r1, c1 = bbox
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not np.any(patch == BG):
            return None
        if _bbox_from_coords(np.argwhere(patch == rect_color)) != (0, 0, patch.shape[0] - 1, patch.shape[1] - 1):
            return None
        if not _is_single_component(patch == rect_color):
            return None

        scores, raw, voters = _vote_scores(grid, bbox)
        if len(voters) < 3 or not (2 <= len(scores) <= 3):
            return None
        best = max(scores.values())
        winners = [color for color, score in scores.items() if score == best]
        if len(winners) != 1:
            return None
        for r, c, _marker_color, _contrib in voters:
            marker_claims[(r, c)] += 1
        infos.append(
            {
                "rect_color": rect_color,
                "bbox": bbox,
                "scores": scores,
                "raw": raw,
                "winner": winners[0],
                "raw_winner": _raw_majority(raw),
                "voter_colors": set(scores),
            }
        )

    if any(count != 1 for count in marker_claims.values()):
        return None
    return infos


def _grid_passes_constraints(grid: np.ndarray) -> bool:
    infos = _analyze_grid(grid)
    if infos is None:
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False

    winners = [info["winner"] for info in infos]
    if len(set(winners)) < 2:
        return False

    if not any(info["raw_winner"] is not None and info["raw_winner"] != info["winner"] for info in infos):
        return False

    locality_demo = False
    for color in MARKER_COLORS:
        if any(info["winner"] == color for info in infos) and any(
            color in info["voter_colors"] and info["winner"] != color for info in infos
        ):
            locality_demo = True
            break
    if not locality_demo:
        return False

    return True


def _attempt_grid() -> np.ndarray | None:
    h = random.randint(12, 16)
    w = random.randint(12, 16)
    rect_count = random.randint(2, 4)
    rect_colors = random.sample(list(RECT_COLORS), rect_count)

    if rect_count == 4:
        sizes = [(random.randint(3, 4), random.randint(3, 4)) for _ in range(rect_count)]
    else:
        sizes = [(random.randint(3, 6), random.randint(3, 6)) for _ in range(rect_count)]
    boxes = _place_boxes(h, w, sizes)
    if boxes is None:
        return None

    forced_mismatch = random.randrange(rect_count)
    grid = np.zeros((h, w), dtype=int)

    for idx, (box, rect_color) in enumerate(zip(boxes, rect_colors)):
        r0, c0, r1, c1 = box
        rh, rw = r1 - r0 + 1, c1 - c0 + 1
        if idx == forced_mismatch:
            kind = "weighted_mismatch"
        else:
            kind = random.choices(
                ["simple_outside", "weighted_tie", "weighted_mismatch"],
                weights=[3, 3, 2],
                k=1,
            )[0]

        built = _build_rectangle_patch(rect_color, rh, rw, kind)
        if built is None:
            return None
        patch, placements, _meta = built
        grid[r0 : r1 + 1, c0 : c1 + 1] = patch
        for (lr, lc), color in placements:
            gr, gc = r0 + lr, c0 + lc
            if not _in_bounds(grid.shape, gr, gc):
                return None
            if lr < 0 or lr >= rh or lc < 0 or lc >= rw:
                if grid[gr, gc] != BG:
                    return None
                grid[gr, gc] = color

    return grid if _grid_passes_constraints(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_grid()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid adjacency-vote rectangle puzzle")
