from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
RED = 2
YELLOW = 4
LAMP_COLORS = (1, 3, 5, 6, 7, 8, 9)
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GEN_ATTEMPTS = 400


def _find_source(grid: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(grid == RED)
    assert len(coords) == 1, "expected exactly one red source pixel"
    return int(coords[0, 0]), int(coords[0, 1])


def _is_lamp(value: int) -> bool:
    return int(value) not in (BLACK, RED, YELLOW)


def _scan_ray(
    grid: np.ndarray, start: tuple[int, int], dr: int, dc: int
) -> tuple[list[tuple[int, int]], tuple[int, int] | None]:
    h, w = grid.shape
    r, c = int(start[0]) + int(dr), int(start[1]) + int(dc)
    path: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        if grid[r, c] != BLACK:
            return path, (int(r), int(c))
        path.append((int(r), int(c)))
        r += int(dr)
        c += int(dc)
    return path, None


def _simulate(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.all((0 <= g) & (g <= 9)), "expected ARC color ids"
    assert not np.any(g == YELLOW), "yellow is reserved for output overlaps"

    source = _find_source(g)
    h, w = g.shape

    lamp_positions = {
        (int(r), int(c))
        for r, c in np.argwhere((g != BLACK) & (g != RED))
        if _is_lamp(int(g[r, c]))
    }

    lit_lamps: set[tuple[int, int]] = set()
    seen_emitters = {source}
    depth = {source: 0}
    queue = deque([source])
    covers: dict[tuple[int, int], set[int]] = {}
    edges: list[dict] = []
    boundary_rays: list[tuple[tuple[int, int], tuple[int, int]]] = []

    while queue:
        emitter = queue.popleft()
        emitter_color = int(g[emitter])
        for dr, dc in DIRS:
            path, blocker = _scan_ray(g, emitter, dr, dc)
            for cell in path:
                covers.setdefault(cell, set()).add(emitter_color)

            edge = {
                "emitter": emitter,
                "dir": (int(dr), int(dc)),
                "path": path,
                "blocker": blocker,
                "hit_lamp": False,
            }
            if blocker is None:
                boundary_rays.append((emitter, (int(dr), int(dc))))
            else:
                blocker_color = int(g[blocker])
                if _is_lamp(blocker_color):
                    edge["hit_lamp"] = True
                    if blocker not in lit_lamps:
                        lit_lamps.add(blocker)
                    if blocker not in seen_emitters:
                        seen_emitters.add(blocker)
                        depth[blocker] = depth[emitter] + 1
                        queue.append(blocker)
            edges.append(edge)

    out = g.copy()
    for (r, c), colors in covers.items():
        if g[r, c] != BLACK:
            continue
        if len(colors) == 1:
            out[r, c] = next(iter(colors))
        elif len(colors) > 1:
            out[r, c] = YELLOW

    return {
        "out": out,
        "source": source,
        "lamp_positions": lamp_positions,
        "lit_lamps": lit_lamps,
        "unlit_lamps": lamp_positions - lit_lamps,
        "depth": depth,
        "edges": edges,
        "boundary_rays": boundary_rays,
        "yellow_count": int(np.sum(out == YELLOW)),
        "shape": (h, w),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate the red source through visible lamps, paint every beam, and mark multi-color overlaps yellow.

    Input structure:
    - A black background contains several non-red, non-yellow single-pixel lamps.
    - Exactly one red single-pixel source is present.
    - Lamps may align by row or column, while some others can remain isolated from the source chain.

    Rule:
    - Starting from the red source, emit rays in the four cardinal directions.
    - Along each ray, paint every black cell until the first non-black pixel or the grid boundary.
    - If that first non-black pixel is a lamp, it becomes lit and later emits its own four rays by the same rule; if it is the red source or no object is met, the ray stops there.
    - Keep cascading until no new lamp is lit. Leave every lamp and the source in their original colors.
    - Any black cell covered by beams of exactly one color takes that color; any black cell covered by beams of multiple different colors becomes yellow.

    Key insight:
    - The lit lamps are the transitive closure of the nearest visible lamp relation, not all lamps; yellow records genuine beam overlap rather than a decorative post-process.
    """
    return _simulate(grid)["out"]


def _rand_dist() -> int:
    return random.randint(2, 4)


def _choice(seq):
    return seq[random.randrange(len(seq))]


def _make_core_layout() -> tuple[dict[str, tuple[int, int]], set[str]]:
    hs = _choice((-1, 1))
    vs = _choice((-1, 1))
    d_h = _rand_dist()
    d_v = _rand_dist()
    ext_row = _rand_dist()
    ext_col = _rand_dist()
    ext_cross = _rand_dist()

    lamps = {
        "A": (0, hs * d_h),
        "H": (0, hs * (d_h + ext_row)),
        "C": (vs * d_v, 0),
        "E": (vs * (d_v + ext_col), hs * d_h),
        "F": (vs * d_v, hs * (d_h + ext_cross)),
    }

    optional_names: list[str] = []
    if random.random() < 0.65:
        optional_names.append("J")
        lamps["J"] = (-vs * _rand_dist(), hs * (d_h + ext_row))
    if random.random() < 0.55:
        optional_names.append("B")
        lamps["B"] = (0, -hs * _rand_dist())
    if random.random() < 0.45 and len(optional_names) < 2:
        optional_names.append("D")
        lamps["D"] = (-vs * _rand_dist(), 0)

    if len(optional_names) > 2:
        for name in optional_names[2:]:
            lamps.pop(name)
    active_optional = {name for name in ("J", "B", "D") if name in lamps}
    return lamps, active_optional


def _required_margins(offsets: dict[str, tuple[int, int]]) -> tuple[int, int, int, int]:
    up = down = left = right = 1
    for dr, dc in offsets.values():
        up = max(up, max(0, -dr))
        down = max(down, max(0, dr))
        left = max(left, max(0, -dc))
        right = max(right, max(0, dc))
    return up, down, left, right


def _pick_canvas(
    offsets: dict[str, tuple[int, int]], distractor_count: int
) -> tuple[int, int, int, int] | None:
    up, down, left, right = _required_margins(offsets)
    used_rows = {0, *(dr for dr, _dc in offsets.values())}
    used_cols = {0, *(dc for _dr, dc in offsets.values())}
    min_h = up + down + 1 + distractor_count + max(2, len(used_rows) // 2)
    min_w = left + right + 1 + distractor_count + max(2, len(used_cols) // 2)
    if min_h > 26 or min_w > 26:
        return None

    h = random.randint(max(12, min_h), min(26, max(12, min_h + 6)))
    w = random.randint(max(12, min_w), min(26, max(12, min_w + 6)))
    src_r = random.randint(up, h - 1 - down)
    src_c = random.randint(left, w - 1 - right)
    return h, w, src_r, src_c


def _place_distractors(
    grid: np.ndarray,
    used_rows: set[int],
    used_cols: set[int],
    count: int,
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    h, w = grid.shape
    rows = [r for r in range(h) if r not in used_rows]
    cols = [c for c in range(w) if c not in used_cols]
    if len(rows) < count or len(cols) < count:
        return []

    random.shuffle(rows)
    random.shuffle(cols)
    placed: list[tuple[int, int]] = []
    for r in rows:
        for c in cols:
            cell = (int(r), int(c))
            if cell in blocked:
                continue
            if any(pr == r or pc == c for pr, pc in placed):
                continue
            placed.append(cell)
            blocked.add(cell)
            if len(placed) == count:
                return placed
    return []


def _build_attempt() -> np.ndarray | None:
    offsets, optional_names = _make_core_layout()
    distractor_count = random.randint(1, 3)
    canvas = _pick_canvas(offsets, distractor_count)
    if canvas is None:
        return None
    h, w, src_r, src_c = canvas

    source = (src_r, src_c)
    lamp_positions = {
        name: (src_r + dr, src_c + dc) for name, (dr, dc) in offsets.items()
    }

    cells = [source, *lamp_positions.values()]
    if len(cells) != len(set(cells)):
        return None
    if any(not (0 <= r < h and 0 <= c < w) for r, c in cells):
        return None

    grid = np.zeros((h, w), dtype=int)
    grid[source] = RED

    lit_names = ["A", "H", "C", "E", "F"] + sorted(optional_names)
    if len(lit_names) > len(LAMP_COLORS):
        return None
    lit_colors = random.sample(LAMP_COLORS, len(lit_names))
    for name, color in zip(lit_names, lit_colors):
        grid[lamp_positions[name]] = int(color)

    used_rows = {src_r, *(r for r, _c in lamp_positions.values())}
    used_cols = {src_c, *(c for _r, c in lamp_positions.values())}
    blocked = {source, *lamp_positions.values()}
    distractors = _place_distractors(grid, used_rows, used_cols, distractor_count, blocked)
    if len(distractors) != distractor_count:
        return None

    remaining_colors = list(LAMP_COLORS)
    random.shuffle(remaining_colors)
    for i, cell in enumerate(distractors):
        grid[cell] = int(remaining_colors[i % len(remaining_colors)])

    info = _simulate(grid)
    expected_lit = {lamp_positions[name] for name in lit_names}
    if info["lit_lamps"] != expected_lit:
        return None
    if info["unlit_lamps"] != set(distractors):
        return None

    if max(info["depth"].values()) < (3 if "J" in optional_names else 2):
        return None
    if info["yellow_count"] < 3:
        return None
    if len(info["unlit_lamps"]) < 1:
        return None
    if len(info["boundary_rays"]) < 6:
        return None

    source_hits = [
        edge for edge in info["edges"] if edge["emitter"] == source and edge["hit_lamp"]
    ]
    if len(source_hits) < 2:
        return None

    cascaded_hits = [
        edge
        for edge in info["edges"]
        if edge["emitter"] != source and edge["hit_lamp"] and edge["blocker"] in expected_lit
    ]
    if len(cascaded_hits) < 3:
        return None

    if not any(
        edge["emitter"] == lamp_positions["A"] and edge["blocker"] == lamp_positions["H"]
        for edge in info["edges"]
    ):
        return None
    if not any(
        edge["emitter"] == lamp_positions["A"] and edge["blocker"] == lamp_positions["E"]
        for edge in info["edges"]
    ):
        return None
    if not any(
        edge["emitter"] == lamp_positions["C"] and edge["blocker"] == lamp_positions["F"]
        for edge in info["edges"]
    ):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid light-chain cascade input")
