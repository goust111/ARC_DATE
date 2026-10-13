from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
GREY = 5

SIDE_TO_DIR = {
    "N": (-1, 0),
    "S": (1, 0),
    "W": (0, -1),
    "E": (0, 1),
}

GEN_RETRIES = 400
GRID_MIN = 18
GRID_MAX = 27
OUTER_MIN = 8
OUTER_MAX = 13
POCKET_COUNT_RANGE = (2, 4)
DISTRACTOR_COUNT_RANGE = (1, 3)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _marker_isolated(markers: list[tuple[int, int]], cand: tuple[int, int]) -> bool:
    r, c = cand
    return all(max(abs(r - rr), abs(c - cc)) >= 2 for rr, cc in markers)


def _blue_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    blue = np.argwhere(grid == BLUE)
    assert blue.size > 0, "expected a blue structure"
    return _bbox_coords(blue)


def _extract_pockets(grid: np.ndarray) -> list[dict]:
    blue_comps = _components(grid == BLUE)
    assert len(blue_comps) == 1, "expected exactly one blue component"
    r0, c0, r1, c1 = _blue_bbox(grid)
    box = grid[r0 : r1 + 1, c0 : c1 + 1]
    pockets = []
    for local in _components(box == BLACK):
        sides = []
        if np.any(local[:, 0] == 0):
            sides.append("N")
        if np.any(local[:, 0] == box.shape[0] - 1):
            sides.append("S")
        if np.any(local[:, 1] == 0):
            sides.append("W")
        if np.any(local[:, 1] == box.shape[1] - 1):
            sides.append("E")
        if len(sides) != 1:
            continue
        side = sides[0]
        dr, dc = SIDE_TO_DIR[side]
        coords = local + np.array([r0, c0], dtype=int)
        cells = {tuple(map(int, rc)) for rc in coords}
        frontier = np.array(
            [
                (r, c)
                for r, c in cells
                if (int(r + dr), int(c + dc)) not in cells
            ],
            dtype=int,
        )
        assert frontier.size > 0, "each pocket needs an opening frontier"
        pr0, pc0, pr1, pc1 = _bbox_coords(coords)
        pockets.append(
            {
                "side": side,
                "dir": (dr, dc),
                "coords": coords.astype(int),
                "frontier": frontier,
                "bbox": (pr0, pc0, pr1, pc1),
            }
        )
    pockets.sort(key=lambda item: (item["bbox"][0], item["bbox"][1], item["side"]))
    assert pockets, "expected at least one edge pocket"
    return pockets


def _landing(grid: np.ndarray, pocket: dict) -> tuple[int, np.ndarray]:
    h, w = grid.shape
    dr, dc = pocket["dir"]
    coords = pocket["coords"]
    for step in range(1, max(h, w) + 1):
        moved = coords + step * np.array([dr, dc], dtype=int)
        rr = moved[:, 0]
        cc = moved[:, 1]
        if rr.min() < 0 or rr.max() >= h or cc.min() < 0 or cc.max() >= w:
            break
        values = grid[rr, cc]
        assert not np.any(values == BLUE), "launched cavity must not overlap blue"
        hit_count = int(np.count_nonzero(values == GREY))
        if hit_count:
            assert hit_count == 1, "first corridor hit must be a single grey marker"
            return step, moved
    raise AssertionError("pocket corridor did not hit a grey marker")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Eject each edge pocket of the blue structure onto the first grey marker in its opening corridor, then heal the pocket in blue.

    Input structure:
    - A black background contains one large blue 4-connected structure.
    - Inside the blue bbox, some black 4-connected cavities touch exactly one bbox side, forming one-direction edge pockets.
    - Grey pixels are isolated single-pixel markers outside the structure; each real pocket has exactly one marker on its outward corridor, while other grey markers are distractors.

    Rule:
    - For every black pocket touching exactly one side of the blue bbox, take the pocket's full cavity shape as the moved piece.
    - Translate that shape outward along its opening direction until the translated shape first covers a grey marker; stamp the whole translated shape grey there.
    - Fill the original pocket cells blue, and leave every grey marker that lies on no pocket corridor unchanged.

    Key insight:
    - The moved object is the structure's negative space: each pocket appears twice in the output, once as a grey emitted copy and once as blue self-repair in the original notch.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(map(int, np.unique(g))) <= {BLACK, BLUE, GREY}

    grey_comps = _components(g == GREY)
    assert grey_comps, "expected grey markers"
    assert all(len(comp) == 1 for comp in grey_comps), "grey markers must be single pixels"

    pockets = _extract_pockets(g)
    out = g.copy()
    for pocket in pockets:
        out[pocket["coords"][:, 0], pocket["coords"][:, 1]] = BLUE

    for pocket in pockets:
        _step, moved = _landing(g, pocket)
        rr = moved[:, 0]
        cc = moved[:, 1]
        assert not np.any(out[rr, cc] == BLUE), "grey stamp must land on background or marker"
        out[rr, cc] = GREY

    return out


def _top_pocket_mask() -> np.ndarray:
    h = random.randint(2, 4)
    w = random.randint(2, 5)
    heights = [random.randint(1, h) for _ in range(w)]
    if max(heights) == 1:
        heights[random.randrange(w)] = h
    if len(set(heights)) == 1 and h > 2 and random.random() < 0.7:
        heights[random.randrange(w)] -= 1
    mask = np.zeros((h, w), dtype=bool)
    for c, depth in enumerate(heights):
        mask[:depth, c] = True
    return mask


def _oriented_mask(side: str) -> np.ndarray:
    base = _top_pocket_mask()
    return np.rot90(base, {"N": 0, "W": 1, "S": 2, "E": 3}[side]).astype(bool)


def _coords_from_mask(mask: np.ndarray, r0: int, c0: int) -> np.ndarray:
    coords = np.argwhere(mask).astype(int)
    coords[:, 0] += int(r0)
    coords[:, 1] += int(c0)
    return coords


def _side_sequence(count: int) -> list[str]:
    sides = random.sample(list(SIDE_TO_DIR), count)
    random.shuffle(sides)
    return sides


def _candidate_origins(
    side: str,
    mask: np.ndarray,
    rect: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    top, left, bottom, right = rect
    h, w = mask.shape
    origins = []
    if side == "N":
        if h > bottom - top or w > right - left - 3:
            return origins
        r0 = top
        for c0 in range(left + 2, right - w):
            origins.append((r0, c0))
    elif side == "S":
        if h > bottom - top or w > right - left - 3:
            return origins
        r0 = bottom - h + 1
        for c0 in range(left + 2, right - w):
            origins.append((r0, c0))
    elif side == "W":
        if w > right - left or h > bottom - top - 3:
            return origins
        c0 = left
        for r0 in range(top + 2, bottom - h):
            origins.append((r0, c0))
    else:
        if w > right - left or h > bottom - top - 3:
            return origins
        c0 = right - w + 1
        for r0 in range(top + 2, bottom - h):
            origins.append((r0, c0))
    return origins


def _can_carve(coords: np.ndarray, carved: set[tuple[int, int]]) -> bool:
    for r, c in coords:
        cell = (int(r), int(c))
        if cell in carved:
            return False
        for nr, nc in _neighbors4(*cell):
            if (nr, nc) in carved:
                return False
    return True


def _translation_sets(coords: np.ndarray, dir_: tuple[int, int], shape: tuple[int, int]):
    h, w = shape
    dr, dc = dir_
    result = []
    for step in range(1, max(h, w) + 1):
        moved = coords + step * np.array([dr, dc], dtype=int)
        rr = moved[:, 0]
        cc = moved[:, 1]
        if rr.min() < 0 or rr.max() >= h or cc.min() < 0 or cc.max() >= w:
            break
        result.append((step, moved, {tuple(map(int, rc)) for rc in moved}))
    return result


def _choose_landings(
    grid: np.ndarray,
    pockets: list[dict],
) -> list[dict] | None:
    translations = [_translation_sets(p["coords"], p["dir"], grid.shape) for p in pockets]
    if any(len(opts) < 2 for opts in translations):
        return None
    healed_blue = {tuple(map(int, rc)) for rc in np.argwhere(grid == BLUE)}
    for pocket in pockets:
        healed_blue |= {tuple(map(int, rc)) for rc in pocket["coords"]}

    corridors = []
    for opts in translations:
        union = set()
        for _step, _moved, cells in opts:
            union |= cells
        corridors.append(union)

    candidates: list[list[dict]] = []
    for i, pocket in enumerate(pockets):
        other_corridors = set()
        for j, cells in enumerate(corridors):
            if i != j:
                other_corridors |= cells
        dr, dc = pocket["dir"]
        options = []
        for step, moved, stamp in translations[i]:
            if step < 2:
                continue
            if stamp & healed_blue:
                continue
            frontier_positions = [
                (int(r + step * dr), int(c + step * dc))
                for r, c in pocket["frontier"]
            ]
            random.shuffle(frontier_positions)
            for marker in frontier_positions:
                if marker in other_corridors:
                    continue
                options.append(
                    {
                        "step": step,
                        "marker": marker,
                        "stamp": stamp,
                        "moved": moved,
                    }
                )
        random.shuffle(options)
        if not options:
            return None
        candidates.append(options)

    order = sorted(range(len(pockets)), key=lambda idx: len(candidates[idx]))
    assignments: list[dict | None] = [None] * len(pockets)

    def search(pos: int, used_stamp: set[tuple[int, int]], used_markers: list[tuple[int, int]]) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        for option in candidates[idx]:
            marker = option["marker"]
            if not _marker_isolated(used_markers, marker):
                continue
            if option["stamp"] & used_stamp:
                continue
            assignments[idx] = option
            if search(pos + 1, used_stamp | option["stamp"], used_markers + [marker]):
                return True
            assignments[idx] = None
        return False

    if not search(0, set(), []):
        return None
    return [assignment for assignment in assignments if assignment is not None]


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    outer_h = random.randint(OUTER_MIN, min(OUTER_MAX, h - 8))
    outer_w = random.randint(OUTER_MIN, min(OUTER_MAX, w - 8))
    top = random.randint(3, h - outer_h - 4)
    left = random.randint(3, w - outer_w - 4)
    rect = (top, left, top + outer_h - 1, left + outer_w - 1)

    inp = np.full((h, w), BLACK, dtype=int)
    inp[top : top + outer_h, left : left + outer_w] = BLUE

    carved: set[tuple[int, int]] = set()
    pocket_count = random.randint(*POCKET_COUNT_RANGE)
    for side in _side_sequence(pocket_count):
        placed = False
        for _ in range(80):
            mask = _oriented_mask(side)
            origins = _candidate_origins(side, mask, rect)
            random.shuffle(origins)
            for r0, c0 in origins:
                coords = _coords_from_mask(mask, r0, c0)
                if not _can_carve(coords, carved):
                    continue
                inp[coords[:, 0], coords[:, 1]] = BLACK
                carved |= {tuple(map(int, rc)) for rc in coords}
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    if len(_components(inp == BLUE)) != 1:
        return None

    pockets = _extract_pockets(inp)
    if len(pockets) != pocket_count:
        return None

    assignments = _choose_landings(inp, pockets)
    if assignments is None or len(assignments) != len(pockets):
        return None

    out = inp.copy()
    grey_markers: list[tuple[int, int]] = []
    corridor_union = set()
    for pocket, assignment in zip(pockets, assignments):
        out[pocket["coords"][:, 0], pocket["coords"][:, 1]] = BLUE
        out[assignment["moved"][:, 0], assignment["moved"][:, 1]] = GREY
        marker = assignment["marker"]
        inp[marker] = GREY
        grey_markers.append(marker)
        corridor_union |= assignment["stamp"]

    r0, c0, r1, c1 = rect
    distractor_cells = []
    for rr in range(h):
        for cc in range(w):
            if inp[rr, cc] != BLACK:
                continue
            if r0 <= rr <= r1 and c0 <= cc <= c1:
                continue
            if (rr, cc) in corridor_union:
                continue
            if not _marker_isolated(grey_markers + distractor_cells, (rr, cc)):
                continue
            distractor_cells.append((rr, cc))
    if len(distractor_cells) < DISTRACTOR_COUNT_RANGE[0]:
        return None

    random.shuffle(distractor_cells)
    for marker in distractor_cells[: random.randint(*DISTRACTOR_COUNT_RANGE)]:
        inp[marker] = GREY
        out[marker] = GREY

    try:
        if not np.array_equal(main(inp), out):
            return None
    except AssertionError:
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _build_case()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid pocket-ejection input")
