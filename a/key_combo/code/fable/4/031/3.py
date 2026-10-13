"""ARC puzzle 3: portal-guided cup rays fill the first frame they hit."""

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

DIRS: dict[str, Coord] = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}


def _background(grid: np.ndarray) -> int:
    """Use the public ARC toolbox convention: background is the modal color."""
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(coords: set[Coord]) -> BBox:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray, connectivity: int = 4) -> list[set[Coord]]:
    """Return coordinate sets using the toolbox's (row, column) convention."""
    h, w = mask.shape
    if connectivity == 4:
        steps = ((-1, 0), (1, 0), (0, -1), (0, 1))
    else:
        steps = tuple(
            (dr, dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if (dr, dc) != (0, 0)
        )

    unseen = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    result: list[set[Coord]] = []
    while unseen:
        seed = unseen.pop()
        comp = {seed}
        stack = [seed]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in unseen:
                    unseen.remove(q)
                    comp.add(q)
                    stack.append(q)
        result.append(comp)
    return result


def _rectangle_from_component(coords: set[Coord], color: int) -> dict | None:
    if len(coords) < 8:
        return None
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return None
    border = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }
    if coords != border:
        return None
    interior = {
        (r, c)
        for r in range(r0 + 1, r1)
        for c in range(c0 + 1, c1)
    }
    return {
        "color": int(color),
        "bbox": (r0, c0, r1, c1),
        "border": border,
        "interior": interior,
    }


def _flood_inside_bbox(border: set[Coord], box: BBox, side: str) -> set[Coord]:
    """Flood non-border cells from one side, restricted to the object's bbox."""
    r0, c0, r1, c1 = box
    if side == "U":
        seeds = [(r0, c) for c in range(c0, c1 + 1)]
    elif side == "D":
        seeds = [(r1, c) for c in range(c0, c1 + 1)]
    elif side == "L":
        seeds = [(r, c0) for r in range(r0, r1 + 1)]
    else:
        seeds = [(r, c1) for r in range(r0, r1 + 1)]

    seen = {p for p in seeds if p not in border}
    queue = deque(seen)
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIRS.values():
            q = (r + dr, c + dc)
            if (
                r0 <= q[0] <= r1
                and c0 <= q[1] <= c1
                and q not in border
                and q not in seen
            ):
                seen.add(q)
                queue.append(q)
    return seen


def _cup_from_component(
    grid: np.ndarray, coords: set[Coord], border_color: int, bg: int
) -> dict | None:
    """Recognize a cup by its unique one-pixel tail side and colored mouth."""
    if len(coords) < 7:
        return None
    box = _bbox(coords)
    r0, c0, r1, c1 = box
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return None

    side_cells = {
        "U": sorted(p for p in coords if p[0] == r0),
        "D": sorted(p for p in coords if p[0] == r1),
        "L": sorted(p for p in coords if p[1] == c0),
        "R": sorted(p for p in coords if p[1] == c1),
    }
    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    candidates = [
        (side, cells[0])
        for side, cells in side_cells.items()
        if len(cells) == 1 and cells[0] not in corners
    ]

    passing: list[tuple[str, Coord, int, set[Coord]]] = []
    for direction, spike in candidates:
        mouth = _flood_inside_bbox(coords, box, OPPOSITE[direction])
        colored = {
            p
            for p in mouth
            if int(grid[p]) not in (bg, int(border_color))
        }
        colors = {int(grid[p]) for p in colored}
        if len(colors) == 1 and colored:
            passing.append((direction, spike, colors.pop(), colored))

    if len(passing) != 1:
        return None
    direction, spike, content_color, content = passing[0]
    return {
        "border_color": int(border_color),
        "border": coords,
        "bbox": box,
        "direction": direction,
        "spike": spike,
        "content_color": int(content_color),
        "content": content,
    }


def _parse_objects(grid: np.ndarray, bg: int) -> tuple[list[dict], list[dict]]:
    """Classify hollow rectangles first, then cup-shaped 8-components."""
    colors = sorted(set(map(int, grid.ravel())) - {bg})
    frames: list[dict] = []
    frame_components: set[frozenset[Coord]] = set()
    for color in colors:
        for comp in _components(grid == color, connectivity=4):
            frame = _rectangle_from_component(comp, color)
            if frame is not None:
                frames.append(frame)
                frame_components.add(frozenset(comp))

    cups: list[dict] = []
    for color in colors:
        for comp in _components(grid == color, connectivity=8):
            if frozenset(comp) in frame_components:
                continue
            cup = _cup_from_component(grid, comp, color, bg)
            if cup is not None:
                cups.append(cup)
    return frames, cups


def _portal_pairs(
    grid: np.ndarray, bg: int, frames: list[dict], cups: list[dict]
) -> dict[Coord, Coord]:
    """Find colors represented by exactly two isolated non-object pixels."""
    excluded_colors = {int(f["color"]) for f in frames}
    excluded_colors |= {int(c["border_color"]) for c in cups}
    excluded_colors |= {int(c["content_color"]) for c in cups}

    portal_at: dict[Coord, Coord] = {}
    for color in sorted(set(map(int, grid.ravel())) - {bg} - excluded_colors):
        coords = [tuple(map(int, p)) for p in np.argwhere(grid == color)]
        if len(coords) != 2:
            continue
        comps = _components(grid == color, connectivity=8)
        if len(comps) != 2 or any(len(comp) != 1 for comp in comps):
            continue
        a, b = coords
        portal_at[a] = b
        portal_at[b] = a
    return portal_at


def _trace_ray(
    shape: tuple[int, int],
    spike: Coord,
    direction: str,
    portal_at: dict[Coord, Coord],
    border_owner: dict[Coord, int],
) -> int | None:
    """Trace one invisible ray; teleporting does not rotate its direction."""
    h, w = shape
    dr, dc = DIRS[direction]
    r, c = spike
    visited: set[tuple[int, int, str]] = set()
    while True:
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            return None
        state = (r, c, direction)
        if state in visited:
            return None
        visited.add(state)
        if (r, c) in portal_at:
            r, c = portal_at[(r, c)]
            continue
        if (r, c) in border_owner:
            return int(border_owner[(r, c)])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Empty each tailed cup and fill the first hollow rectangle reached by its portal-guided ray.

    Input structure:
    - The modal color is background. Each cup is an 8-connected, uniquely colored open
      frame with a one-pixel tail on one bbox side and one content color in its mouth.
    - Portal pixels are isolated same-colored pairs, and the remaining large monochrome
      components are empty one-cell-thick rectangular frames.

    Rule:
    - Erase every cup's content pixels and cast a straight ray outward from its tail spike.
    - On entering a portal pixel, resume from its same-colored twin without changing the
      row/column direction. Stop at the first rectangular border and fill only that frame's
      interior with the emitting cup's content color; preserve all other pixels.

    Key insight:
    - The apparent rectangle farther along the original line is a decoy: the complete
      same-color portal jump must be followed, and only the first frame after the jump fills.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D ARC grid"
    bg = _background(g)
    frames, cups = _parse_objects(g, bg)
    assert frames, "expected at least one hollow rectangle"
    assert cups, "expected at least one cup"

    portal_at = _portal_pairs(g, bg, frames, cups)
    assert portal_at, "expected at least one portal pair"
    border_owner = {
        p: i for i, frame in enumerate(frames) for p in frame["border"]
    }

    out = g.copy()
    claims: dict[int, int] = {}
    for cup in sorted(cups, key=lambda x: x["spike"]):
        for p in cup["content"]:
            out[p] = bg
        hit = _trace_ray(
            g.shape,
            cup["spike"],
            cup["direction"],
            portal_at,
            border_owner,
        )
        if hit is None:
            continue
        assert hit not in claims, "two cups ambiguously target the same frame"
        claims[hit] = int(cup["content_color"])

    for frame_i, fill_color in claims.items():
        for p in frames[frame_i]["interior"]:
            out[p] = fill_color
    return out


# ---------------------------------------------------------------------------
# Generator: two or three independent horizontal circuits, globally transformed.
# Every circuit includes an original-line decoy and a post-target second frame.
# ---------------------------------------------------------------------------


def _right_facing_cup(color: int) -> tuple[np.ndarray, Coord, list[Coord]]:
    canonical = np.zeros((5, 5), dtype=int)
    border = {
        (0, 2),
        (1, 1), (1, 3),
        (2, 0), (2, 4),
        (3, 0), (3, 4),
        (4, 0), (4, 4),
    }
    for p in border:
        canonical[p] = color
    content_candidates = [(2, 2), (3, 1), (3, 2), (3, 3), (4, 1), (4, 2), (4, 3)]
    patch = np.rot90(canonical, k=-1).copy()
    spike = (2, 4)
    rotated_candidates = [(c, 4 - r) for r, c in content_candidates]
    return patch, spike, rotated_candidates


def _draw_frame(grid: np.ndarray, top: int, left: int, height: int, width: int, color: int) -> dict:
    grid[top, left : left + width] = color
    grid[top + height - 1, left : left + width] = color
    grid[top : top + height, left] = color
    grid[top : top + height, left + width - 1] = color
    border = {
        (r, c)
        for r in range(top, top + height)
        for c in range(left, left + width)
        if r in (top, top + height - 1) or c in (left, left + width - 1)
    }
    interior = {
        (r, c)
        for r in range(top + 1, top + height - 1)
        for c in range(left + 1, left + width - 1)
    }
    return {"border": border, "interior": interior}


def _oracle_trace(
    shape: tuple[int, int],
    spike: Coord,
    direction: str,
    portal_at: dict[Coord, Coord],
    frames: list[dict],
) -> tuple[int | None, list[Coord]]:
    border_owner = {p: i for i, frame in enumerate(frames) for p in frame["border"]}
    h, w = shape
    dr, dc = DIRS[direction]
    r, c = spike
    used: list[Coord] = []
    seen: set[tuple[int, int]] = set()
    while True:
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w) or (r, c) in seen:
            return None, used
        seen.add((r, c))
        if (r, c) in portal_at:
            used.append((r, c))
            r, c = portal_at[(r, c)]
            continue
        if (r, c) in border_owner:
            return border_owner[(r, c)], used


def _build_canonical(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    n_cups = rng.choice((2, 3))
    height = n_cups * 8 + 2
    width = rng.choice((29, 30))
    inp = np.zeros((height, width), dtype=int)

    palette = list(range(1, 10))
    rng.shuffle(palette)
    portal_colors = palette[:n_cups]
    border_colors = palette[n_cups : 2 * n_cups]
    content_colors = palette[2 * n_cups : 3 * n_cups]
    frame_color = palette[3 * n_cups] if n_cups == 2 else border_colors[0]

    frames: list[dict] = []
    cups: list[dict] = []
    pair_map: dict[Coord, Coord] = {}

    for i in range(n_cups):
        source_row = 3 + 8 * i
        target_row = source_row + 4
        cup_left = rng.choice((1, 2))
        patch, spike_local, candidates = _right_facing_cup(border_colors[i])
        cup_top = source_row - spike_local[0]
        mask = patch != 0
        inp[cup_top : cup_top + 5, cup_left : cup_left + 5][mask] = patch[mask]
        spike = (source_row, cup_left + spike_local[1])

        target_width = rng.choice((3, 4))
        content_count = target_width - 1
        content_abs: list[Coord] = []
        for rr, cc in candidates[:content_count]:
            p = (cup_top + rr, cup_left + cc)
            inp[p] = content_colors[i]
            content_abs.append(p)

        entry = (source_row, cup_left + 7)
        twin = (target_row, rng.choice((15, 16)))
        inp[entry] = portal_colors[i]
        inp[twin] = portal_colors[i]
        pair_map[entry] = twin
        pair_map[twin] = entry

        decoy_width = rng.choice((3, 4))
        decoy = _draw_frame(
            inp, source_row - 1, entry[1] + 3, 3, decoy_width, frame_color
        )
        decoy_id = len(frames)
        frames.append(decoy)

        target = _draw_frame(inp, target_row - 1, 19, 3, target_width, frame_color)
        target_id = len(frames)
        frames.append(target)

        far = _draw_frame(inp, target_row - 1, 25, 3, 3, frame_color)
        far_id = len(frames)
        frames.append(far)

        cups.append(
            {
                "spike": spike,
                "direction": "R",
                "content": content_abs,
                "content_color": content_colors[i],
                "entry": entry,
                "decoy_id": decoy_id,
                "target_id": target_id,
                "far_id": far_id,
            }
        )

    expected = inp.copy()
    hit_ids: set[int] = set()
    for cup in cups:
        hit, used = _oracle_trace(inp.shape, cup["spike"], "R", pair_map, frames)
        direct_hit, direct_used = _oracle_trace(inp.shape, cup["spike"], "R", {}, frames)
        if hit != cup["target_id"] or used != [cup["entry"]]:
            return None
        if direct_hit != cup["decoy_id"] or direct_used:
            return None
        if cup["far_id"] == hit or hit in hit_ids:
            return None
        hit_ids.add(int(hit))
        for p in cup["content"]:
            expected[p] = 0
        for p in frames[hit]["interior"]:
            expected[p] = cup["content_color"]

    # Semantic checks independent of main: only content and first-hit interiors change.
    changed = set(map(tuple, np.argwhere(inp != expected)))
    required = {p for cup in cups for p in cup["content"]}
    required |= {p for i in hit_ids for p in frames[i]["interior"]}
    if changed != required:
        return None
    for i, frame in enumerate(frames):
        if i not in hit_ids and any(int(expected[p]) != 0 for p in frame["interior"]):
            return None

    k = rng.randrange(4)
    inp = np.rot90(inp, k=k).copy()
    expected = np.rot90(expected, k=k).copy()
    if rng.random() < 0.5:
        inp = np.flipud(inp).copy()
        expected = np.flipud(expected).copy()
    return inp, expected


def generate_input() -> np.ndarray:
    """Generate a varied, oracle-checked puzzle with explicit shortcut counterexamples."""
    rng = random.Random()
    for _ in range(300):
        built = _build_canonical(rng)
        if built is None:
            continue
        inp, expected = built
        if max(inp.shape) > 30 or max(expected.shape) > 30:
            continue
        if np.array_equal(inp, expected):
            continue
        try:
            actual = main(inp)
        except (AssertionError, ValueError, IndexError):
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("generate_input: failed to build a valid portal-ray puzzle")
