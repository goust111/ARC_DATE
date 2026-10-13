from __future__ import annotations

from collections import deque
from itertools import product
from pathlib import Path
import random
import sys

import numpy as np


_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import find_components, get_bg  # noqa: E402


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]
DIR4: tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
SIDE_TO_DIR = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
MAX_ATTEMPTS = 700


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= int(r) < h and 0 <= int(c) < w


def _border_cells(bbox: BBox) -> set[Coord]:
    r0, c0, r1, c1 = map(int, bbox)
    cells: set[Coord] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _side_and_dir(bbox: BBox, breach: Coord) -> tuple[str, Coord]:
    r0, c0, r1, c1 = bbox
    r, c = breach
    corner = (r in (r0, r1)) and (c in (c0, c1))
    assert not corner, "the missing wall cell must not be a corner"
    if r == r0:
        return "top", SIDE_TO_DIR["top"]
    if r == r1:
        return "bottom", SIDE_TO_DIR["bottom"]
    if c == c0:
        return "left", SIDE_TO_DIR["left"]
    if c == c1:
        return "right", SIDE_TO_DIR["right"]
    raise AssertionError("breach must lie on the frame boundary")


def _parse_frames(g: np.ndarray) -> list[dict[str, object]]:
    bg = int(get_bg(g))
    frames: list[dict[str, object]] = []
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    assert comps, "expected at least one foreground outline"

    for idx, comp in enumerate(comps):
        color = int(comp["color"])
        bbox = tuple(map(int, comp["bbox"]))
        r0, c0, r1, c1 = bbox
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h >= 3 and w >= 3, "outlines must be at least 3x3"

        actual = {tuple(map(int, p)) for p in comp["coords"]}
        border = _border_cells(bbox)
        assert actual <= border, "outline components may only occupy their rectangle border"

        missing = sorted(border - actual)
        assert len(missing) in (0, 1), "each outline is closed or has one missing wall cell"

        record: dict[str, object] = {
            "id": idx,
            "color": color,
            "bbox": bbox,
            "closed": len(missing) == 0,
            "breach": None,
            "dir": None,
        }
        if missing:
            breach = missing[0]
            _side, direction = _side_and_dir(bbox, breach)
            assert int(g[breach]) == bg
            record["breach"] = breach
            record["dir"] = direction

        frames.append(record)

    return frames


def _ray_from_frame(
    g: np.ndarray, frames: list[dict[str, object]], source_idx: int
) -> tuple[int | None, list[Coord]]:
    frame = frames[source_idx]
    breach = frame["breach"]
    direction = frame["dir"]
    if breach is None or direction is None:
        return None, []

    bg = int(get_bg(g))
    dr, dc = direction  # type: ignore[misc]
    r, c = breach  # type: ignore[misc]
    r += int(dr)
    c += int(dc)
    path: list[Coord] = []

    breach_to_frame: dict[Coord, int] = {}
    for j, other in enumerate(frames):
        if j != source_idx and other["breach"] is not None:
            breach_to_frame[other["breach"]] = j  # type: ignore[index]

    while _in_bounds(g.shape, r, c):
        p = (int(r), int(c))
        if p in breach_to_frame:
            path.append(p)
            return breach_to_frame[p], path
        if int(g[p]) != bg:
            return None, path
        path.append(p)
        r += int(dr)
        c += int(dc)

    return None, path


def _topological_fate(
    broken_ids: list[int], incoming: dict[int, list[int]], outgoing: dict[int, int | None]
) -> dict[int, bool]:
    succ: dict[int, list[int]] = {i: [] for i in broken_ids}
    indeg: dict[int, int] = {i: len(incoming[i]) for i in broken_ids}
    for src, dst in outgoing.items():
        if dst is not None:
            succ[src].append(dst)

    q = deque(sorted(i for i in broken_ids if indeg[i] == 0))
    order: list[int] = []
    while q:
        node = q.popleft()
        order.append(node)
        for dst in succ[node]:
            indeg[dst] -= 1
            if indeg[dst] == 0:
                q.append(dst)

    if len(order) == len(broken_ids):
        plugged: dict[int, bool] = {}
        for node in order:
            plugged[node] = any(not plugged[src] for src in incoming[node])
        return plugged

    # The puzzle promises no directed cycles. Keep a strict unique-fixpoint fallback
    # so invalid hand-authored inputs fail loudly instead of depending on traversal.
    pos = {node: k for k, node in enumerate(broken_ids)}
    fixed: list[dict[int, bool]] = []
    for bits in product((False, True), repeat=len(broken_ids)):
        state = {node: bool(bits[pos[node]]) for node in broken_ids}
        ok = True
        for node in broken_ids:
            rhs = any(not state[src] for src in incoming[node])
            if state[node] != rhs:
                ok = False
                break
        if ok:
            fixed.append(state)
    assert len(fixed) == 1, "plug relation must have a unique stable fate"
    return fixed[0]


def _solve_state(g: np.ndarray, frames: list[dict[str, object]]):
    broken_ids = [i for i, frame in enumerate(frames) if frame["breach"] is not None]
    outgoing: dict[int, int | None] = {}
    rays: dict[int, list[Coord]] = {}
    incoming: dict[int, list[int]] = {i: [] for i in broken_ids}

    for i in broken_ids:
        target, path = _ray_from_frame(g, frames, i)
        outgoing[i] = target
        rays[i] = path
        if target is not None:
            assert target in incoming, "jets may only plug broken outlines"
            incoming[target].append(i)

    plugged = _topological_fate(broken_ids, incoming, outgoing)

    for target in broken_ids:
        if not plugged[target]:
            continue
        hitters = [src for src in incoming[target] if not plugged[src]]
        assert hitters, "a plugged outline needs an active incoming jet"
        hit_colors = {int(frames[src]["color"]) for src in hitters}
        assert len(hit_colors) == 1 or len(hitters) == 1, "plug color must be unambiguous"

    return broken_ids, outgoing, rays, incoming, plugged


def _inside_bbox(p: Coord, bbox: BBox) -> bool:
    r, c = p
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def _nested_bbox_cells(frames: list[dict[str, object]], owner_idx: int, p: Coord) -> bool:
    owner_bbox = frames[owner_idx]["bbox"]  # type: ignore[assignment]
    for j, other in enumerate(frames):
        if j == owner_idx:
            continue
        bbox = other["bbox"]  # type: ignore[assignment]
        if _inside_bbox(p, bbox) and _inside_bbox((bbox[0], bbox[1]), owner_bbox):  # type: ignore[index]
            return True
    return False


def _fill_frame_interior(out: np.ndarray, frames: list[dict[str, object]], idx: int, bg: int) -> None:
    frame = frames[idx]
    color = int(frame["color"])
    r0, c0, r1, c1 = frame["bbox"]  # type: ignore[misc]
    for r in range(int(r0) + 1, int(r1)):
        for c in range(int(c0) + 1, int(c1)):
            p = (r, c)
            if _nested_bbox_cells(frames, idx, p):
                continue
            if int(out[p]) == int(bg):
                out[p] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill sealed outlines and draw the surviving jets after broken outlines plug one another to a stable fate.

    Input structure:
    - A dominant-color background contains separated monochrome rectangular outline figures; outlines may be nested but do not touch.
    - Each outline is either a complete rectangle frame or the same frame with exactly one non-corner wall cell missing.
    - A missing wall cell is the breach, and the side it lies on gives the outward cardinal spray direction.

    Rule:
    - Closed outlines have their background interior cells filled with their own color.
    - Every unblocked broken outline sprays a straight line of its own color from just outside its breach until the grid edge, another wall, or another breach.
    - If a still-spraying line reaches another outline's breach, that breach is plugged by the spray color; the plugged outline fills its own interior and stops spraying.
    - Apply the plug/no-spray dependency to its unique fixpoint, then draw only the jets from outlines that remain unblocked.

    Key insight:
    - A broken outline's final state is decided by the whole directed plug chain: once an outline is plugged, its own downstream jet disappears, which can change the fate of outer outlines.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must be at most 30x30"
    bg = int(get_bg(g))
    frames = _parse_frames(g)
    broken_ids, _outgoing, rays, _incoming, plugged = _solve_state(g, frames)

    out = g.copy()
    for idx, frame in enumerate(frames):
        if bool(frame["closed"]) or bool(plugged.get(idx, False)):
            _fill_frame_interior(out, frames, idx, bg)

    for idx in broken_ids:
        if plugged[idx]:
            continue
        color = int(frames[idx]["color"])
        for r, c in rays[idx]:
            out[r, c] = color

    return out


# ---------------- generator ----------------


def _draw_frame(grid: np.ndarray, bbox: BBox, color: int, breach: Coord | None = None) -> None:
    for r, c in _border_cells(bbox):
        if breach is not None and (r, c) == breach:
            continue
        grid[r, c] = int(color)


def _expanded(bbox: BBox, pad: int, shape: tuple[int, int]) -> BBox:
    h, w = shape
    r0, c0, r1, c1 = bbox
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _overlap(a: BBox, b: BBox) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _can_place_bbox(grid: np.ndarray, bbox: BBox, used: list[BBox], pad: int = 1) -> bool:
    eb = _expanded(bbox, pad, grid.shape)
    if any(_overlap(eb, _expanded(other, 0, grid.shape)) for other in used):
        return False
    r0, c0, r1, c1 = bbox
    return bool(np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == int(get_bg(grid))))


def _odd_choices(lo: int, hi: int) -> list[int]:
    return [x for x in range(int(lo), int(hi) + 1) if x % 2 == 1]


def _place_closed_distractor(
    grid: np.ndarray, used: list[BBox], color: int, rng: random.Random
) -> BBox | None:
    h, w = grid.shape
    for _ in range(300):
        fh = rng.randint(4, 7)
        fw = rng.randint(4, 7)
        if fh > h or fw > w:
            continue
        r0 = rng.randint(0, h - fh)
        c0 = rng.randint(0, w - fw)
        bbox = (r0, c0, r0 + fh - 1, c0 + fw - 1)
        if not _can_place_bbox(grid, bbox, used, pad=1):
            continue
        _draw_frame(grid, bbox, color, None)
        used.append(bbox)
        return bbox
    return None


def _clear_ray_to_top(grid: np.ndarray, breach: Coord) -> bool:
    r, c = breach
    for rr in range(r):
        if int(grid[rr, c]) != int(get_bg(grid)):
            return False
    return True


def _place_standalone_broken(
    grid: np.ndarray, used: list[BBox], color: int, rng: random.Random
) -> BBox | None:
    h, w = grid.shape
    for _ in range(300):
        fh = rng.choice([5, 6, 7])
        fw = rng.choice([5, 7])
        if fh + 3 > h or fw > w:
            continue
        r0 = rng.randint(2, h - fh)
        c0 = rng.randint(0, w - fw)
        bbox = (r0, c0, r0 + fh - 1, c0 + fw - 1)
        breach = (r0, c0 + fw // 2)
        if not _can_place_bbox(grid, bbox, used, pad=1):
            continue
        if not _clear_ray_to_top(grid, breach):
            continue
        _draw_frame(grid, bbox, color, breach)
        used.append(bbox)
        return bbox
    return None


def _build_top_chain(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(27, 30)
    w = rng.randint(27, 30)
    bg = rng.choice([0, 5, 7])
    grid = np.full((h, w), bg, dtype=int)

    depth = rng.choice([3, 4])
    gap = 3
    min_side = 2 * gap * (depth - 1) + (3 if depth == 4 else 5)
    max_side = min(h - 8, w - 8, 23)
    choices = _odd_choices(min_side, max_side)
    if not choices:
        return None

    outer_h = rng.choice(choices)
    outer_w = rng.choice(choices)
    if h - outer_h < 6 or w - outer_w < 6:
        return None

    r0 = rng.randint(3, h - outer_h - 3)
    c0 = rng.randint(2, w - outer_w - 3)
    center_c = c0 + outer_w // 2
    colors = [c for c in range(10) if c != bg]
    rng.shuffle(colors)
    needed = depth + 2
    if len(colors) < needed:
        return None

    used: list[BBox] = []
    chain_colors = colors[:depth]
    for k, color in enumerate(chain_colors):
        rr0 = r0 + k * gap
        cc0 = c0 + k * gap
        rr1 = r0 + outer_h - 1 - k * gap
        cc1 = c0 + outer_w - 1 - k * gap
        if rr1 - rr0 + 1 < 3 or cc1 - cc0 + 1 < 3:
            return None
        breach = (rr0, center_c)
        if not (cc0 < center_c < cc1):
            return None
        bbox = (rr0, cc0, rr1, cc1)
        _draw_frame(grid, bbox, color, breach)
        used.append(bbox)

    if _place_closed_distractor(grid, used, colors[depth], rng) is None:
        return None
    if depth == 4 or rng.random() < 0.6:
        if _place_standalone_broken(grid, used, colors[depth + 1], rng) is None:
            return None

    return grid


def _longest_directed_chain(outgoing: dict[int, int | None]) -> int:
    best = 0
    for start in outgoing:
        seen: set[int] = set()
        node = start
        length = 0
        while node in outgoing and outgoing[node] is not None and node not in seen:
            seen.add(node)
            length += 1
            node = outgoing[node]  # type: ignore[assignment]
        best = max(best, length)
    return best


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        frames = _parse_frames(grid)
        broken_ids, outgoing, rays, _incoming, plugged = _solve_state(grid, frames)
        out = main(grid)
    except Exception:
        return False

    if np.array_equal(out, grid):
        return False
    if len(frames) < 4 or len(broken_ids) < 3:
        return False
    if not any(bool(frame["closed"]) for frame in frames):
        return False
    if _longest_directed_chain(outgoing) < 2:
        return False

    has_retraction = False
    for src, mid in outgoing.items():
        if mid is None or not plugged.get(mid, False):
            continue
        nxt = outgoing.get(mid)
        if nxt is not None and not plugged.get(nxt, False):
            has_retraction = True
            break
    if not has_retraction:
        return False

    active_to_edge = any((not plugged[i]) and outgoing[i] is None and len(rays[i]) >= 2 for i in broken_ids)
    active_plug = any((not plugged[i]) and outgoing[i] is not None for i in broken_ids)
    if not active_plug or not active_to_edge:
        return False

    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = _build_top_chain(rng)
        if grid is None:
            continue
        k = rng.randrange(4)
        if k:
            grid = np.rot90(grid, k).copy()
        if rng.random() < 0.5:
            grid = grid[:, ::-1].copy()
        if max(grid.shape) > 30:
            continue
        if _semantic_checks(grid):
            return grid
    raise RuntimeError("failed to generate a valid jet plug chain input")
