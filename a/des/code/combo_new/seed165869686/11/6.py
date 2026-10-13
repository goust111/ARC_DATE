import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
GREY = 5
ORANGE = 7
MAROON = 9

NETWORK_COLORS = {GREY, ORANGE, BLUE}
RESERVED_COLORS = {BG, BLUE, GREY, ORANGE, MAROON}
SOURCE_COLORS = [2, 3, 4, 6, 8]

GEN_RETRIES = 400
LEAF_COUNT_CHOICES = [2, 3, 4]
LEAF_COUNT_WEIGHTS = [3, 3, 2]
FRAME_SIZES = [3, 4]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _paste_nonzero(dst: np.ndarray, src: np.ndarray, top: int, left: int) -> None:
    h, w = src.shape
    patch = dst[top : top + h, left : left + w]
    mask = src != BG
    patch[mask] = src[mask]


def _find_source(grid: np.ndarray) -> tuple[int, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[int, list[tuple[int, int]]]] = []

    for color in np.unique(g):
        color = int(color)
        if color in RESERVED_COLORS:
            continue
        for comp in _components(g == color):
            if len(comp) != 4:
                continue
            r0, c0, r1, c1 = _bbox(comp)
            if r1 - r0 != 1 or c1 - c0 != 1:
                continue
            if not any(
                0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) in NETWORK_COLORS
                for r, c in comp
                for nr, nc in _neighbors4(r, c)
            ):
                continue
            candidates.append((color, comp))

    assert len(candidates) == 1, f"expected exactly one 2x2 source block, found {len(candidates)}"
    source_color, comp = candidates[0]
    mask = np.zeros(g.shape, dtype=bool)
    for r, c in comp:
        mask[r, c] = True
    return int(source_color), mask


def _reachable_network(grid: np.ndarray, starts: list[tuple[int, int]]) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r, c in starts:
        if not (0 <= r < h and 0 <= c < w):
            continue
        value = int(g[r, c])
        if value not in NETWORK_COLORS or value == BLUE or bool(seen[r, c]):
            continue
        seen[r, c] = True
        q.append((int(r), int(c)))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w) or bool(seen[nr, nc]):
                continue
            value = int(g[nr, nc])
            if value == GREY or value == ORANGE:
                seen[nr, nc] = True
                q.append((int(nr), int(nc)))
    return seen


def _find_containers(grid: np.ndarray) -> list[tuple[set[tuple[int, int]], list[tuple[int, int]]]]:
    g = np.asarray(grid, dtype=int)
    containers: list[tuple[set[tuple[int, int]], list[tuple[int, int]]]] = []

    for comp in _components(g == MAROON):
        r0, c0, r1, c1 = _bbox(comp)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height < 3 or width < 3:
            continue
        border_area = 2 * (height + width) - 4
        if len(comp) != border_area:
            continue

        comp_set = set(comp)
        valid = True
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                on_border = rr in (r0, r1) or cc in (c0, c1)
                if on_border != ((rr, cc) in comp_set):
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue

        interior = [
            (rr, cc)
            for rr in range(r0 + 1, r1)
            for cc in range(c0 + 1, c1)
            if int(g[rr, cc]) == BG
        ]
        if not interior:
            continue
        containers.append((comp_set, interior))

    return containers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flow paint through a gated pipe tree and fill only the reachable end containers.

    Input structure:
    - Background is black.
    - A single 2x2 colored source block touches one end of a thin grey pipe tree.
    - Pipe branches may contain orange valve pixels (open) or blue valve pixels (closed).
    - Leaf pipes sit next to small maroon hollow container frames whose interiors are black.

    Rule:
    - Start from the source and traverse the connected pipe network orthogonally.
    - Grey pipe pixels carry paint; orange valve pixels allow traversal but keep their orange color; blue valve pixels block traversal.
    - Every reachable grey pipe pixel is recolored to the source color.
    - If a reachable pipe touches a maroon frame, fill that frame's enclosed black interior with the source color.

    Key insight:
    - The task requires separating pipe topology from valve semantics: first recover the tree, then allow flow only through branches gated by orange valves.
    """
    g = np.asarray(grid, dtype=int)
    source_color, source_mask = _find_source(g)
    h, w = g.shape

    starts: list[tuple[int, int]] = []
    for r, c in np.argwhere(source_mask):
        for nr, nc in _neighbors4(int(r), int(c)):
            if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) in NETWORK_COLORS:
                starts.append((int(nr), int(nc)))
    assert starts, "source block must touch the pipe network"

    reachable = _reachable_network(g, starts)

    out = g.copy()
    out[(g == GREY) & reachable] = source_color

    for border, interior in _find_containers(g):
        active = False
        for r, c in border:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and bool(reachable[nr, nc]):
                    active = True
                    break
            if active:
                break
        if not active:
            continue
        for r, c in interior:
            if int(out[r, c]) == BG:
                out[r, c] = source_color

    return out


def _new_leaf() -> dict:
    return {"kind": "leaf"}


def _make_tree(rng: random.Random) -> dict | None:
    root = _new_leaf()
    leaves: list[tuple[dict, int]] = [(root, 0)]
    target_leaves = rng.choices(LEAF_COUNT_CHOICES, weights=LEAF_COUNT_WEIGHTS, k=1)[0]
    max_depth = 3

    while len(leaves) < target_leaves:
        expandable = [i for i, (_, depth) in enumerate(leaves) if depth < max_depth]
        if not expandable:
            return None
        idx = rng.choice(expandable)
        node, depth = leaves.pop(idx)
        node.clear()
        node["kind"] = "branch"
        node["right"] = _new_leaf()
        node["down"] = _new_leaf()
        leaves.append((node["right"], depth + 1))
        leaves.append((node["down"], depth + 1))

    for leaf, _depth in leaves:
        leaf["kind"] = "leaf"
        leaf["term_dir"] = rng.choice(["right", "down"])
        leaf["tail"] = rng.randint(1, 4)
        leaf["frame_size"] = rng.choice(FRAME_SIZES)

    for _ in range(100):
        _assign_valves(root, rng)
        states: list[bool] = []
        _collect_states(root, states)
        reachable_leaves: list[bool] = []
        _collect_reachability(root, True, reachable_leaves)
        if any(states) and not all(states) and any(reachable_leaves) and not all(reachable_leaves):
            return root
    return None


def _assign_valves(node: dict, rng: random.Random) -> None:
    if node["kind"] == "leaf":
        return
    node["right_open"] = bool(rng.getrandbits(1))
    node["down_open"] = bool(rng.getrandbits(1))
    _assign_valves(node["right"], rng)
    _assign_valves(node["down"], rng)


def _collect_states(node: dict, out: list[bool]) -> None:
    if node["kind"] == "leaf":
        return
    out.append(bool(node["right_open"]))
    out.append(bool(node["down_open"]))
    _collect_states(node["right"], out)
    _collect_states(node["down"], out)


def _collect_reachability(node: dict, live: bool, out: list[bool]) -> None:
    if node["kind"] == "leaf":
        out.append(bool(live))
        return
    _collect_reachability(node["right"], live and bool(node["right_open"]), out)
    _collect_reachability(node["down"], live and bool(node["down_open"]), out)


def _render_frame(arr: np.ndarray, top: int, left: int, size: int) -> None:
    arr[top, left : left + size] = MAROON
    arr[top + size - 1, left : left + size] = MAROON
    arr[top : top + size, left] = MAROON
    arr[top : top + size, left + size - 1] = MAROON


def _render_leaf(node: dict) -> np.ndarray:
    tail = int(node["tail"])
    frame_size = int(node["frame_size"])
    term_dir = str(node["term_dir"])

    if term_dir == "right":
        arr = np.zeros((frame_size, tail + 1 + frame_size), dtype=int)
        arr[0, : tail + 1] = GREY
        _render_frame(arr, 0, tail + 1, frame_size)
        return arr

    arr = np.zeros((tail + 1 + frame_size, frame_size), dtype=int)
    arr[: tail + 1, 0] = GREY
    _render_frame(arr, tail + 1, 0, frame_size)
    return arr


def _render_tree(node: dict, rng: random.Random) -> np.ndarray:
    if node["kind"] == "leaf":
        return _render_leaf(node)

    right = _render_tree(node["right"], rng)
    down = _render_tree(node["down"], rng)
    gap_x = rng.randint(2, 4)
    gap_y = rng.randint(2, 4)
    off_r = down.shape[1] + gap_x
    off_d = right.shape[0] + gap_y

    out = np.zeros((off_d + down.shape[0], off_r + right.shape[1]), dtype=int)
    _paste_nonzero(out, right, 0, off_r)
    _paste_nonzero(out, down, off_d, 0)

    out[0, 0] = GREY
    out[0, 1:off_r] = GREY
    out[1:off_d, 0] = GREY
    out[0, 1] = ORANGE if bool(node["right_open"]) else BLUE
    out[1, 0] = ORANGE if bool(node["down_open"]) else BLUE
    return out


def _place_source(fragment: np.ndarray, source_color: int, rng: random.Random) -> np.ndarray | None:
    fh, fw = fragment.shape
    source_side = rng.choice(["left", "top"])
    row_shift = 2 if source_side == "top" else 0
    col_shift = 2 if source_side == "left" else 0
    inner_h = row_shift + fh
    inner_w = col_shift + fw
    if inner_h > 30 or inner_w > 30:
        return None

    final_h = inner_h + rng.randint(0, min(5, 30 - inner_h))
    final_w = inner_w + rng.randint(0, min(5, 30 - inner_w))
    top_pad = rng.randint(0, final_h - inner_h)
    left_pad = rng.randint(0, final_w - inner_w)

    canvas = np.zeros((final_h, final_w), dtype=int)
    frag_top = top_pad + row_shift
    frag_left = left_pad + col_shift
    _paste_nonzero(canvas, fragment, frag_top, frag_left)

    if source_side == "left":
        source_top = frag_top
        source_left = frag_left - 2
    else:
        source_top = frag_top - 2
        source_left = frag_left
    canvas[source_top : source_top + 2, source_left : source_left + 2] = int(source_color)
    return canvas


def _filled_container_counts(inp: np.ndarray, out: np.ndarray, source_color: int) -> tuple[int, int]:
    filled = 0
    unfilled = 0
    for _border, interior in _find_containers(inp):
        if any(int(out[r, c]) == source_color for r, c in interior):
            filled += 1
        else:
            unfilled += 1
    return filled, unfilled


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    tree = _make_tree(rng)
    if tree is None:
        return None

    fragment = _render_tree(tree, rng)
    if fragment.shape[0] > 28 or fragment.shape[1] > 28:
        return None

    source_color = int(rng.choice(SOURCE_COLORS))
    grid = _place_source(fragment, source_color, rng)
    if grid is None:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    recolored = int(np.count_nonzero((grid == GREY) & (out == source_color)))
    untouched = int(np.count_nonzero((grid == GREY) & (out == GREY)))
    filled, unfilled = _filled_container_counts(grid, out, source_color)

    if recolored == 0 or untouched == 0:
        return None
    if filled == 0 or unfilled == 0:
        return None
    if not np.any(out != grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        result = _gen_attempt(rng)
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
