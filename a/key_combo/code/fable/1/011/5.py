from __future__ import annotations

import itertools

import numpy as np


BACKGROUND = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GEN_RETRIES = 500


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _colors_present(grid: np.ndarray, bg: int) -> list[int]:
    return sorted(int(v) for v in np.unique(grid) if int(v) != int(bg))


def _covers(stripe: dict, r: int, c: int) -> bool:
    if stripe["orient"] == "H":
        return int(stripe["start"]) <= int(r) <= int(stripe["start"]) + 1
    return int(stripe["start"]) <= int(c) <= int(stripe["start"]) + 1


def _paint_stripe(canvas: np.ndarray, stripe: dict) -> None:
    color = int(stripe["color"])
    start = int(stripe["start"])
    if stripe["orient"] == "H":
        canvas[start : start + 2, :] = color
    else:
        canvas[:, start : start + 2] = color


def _render(shape: tuple[int, int], stripes: list[dict], draw_order: list[int], bg: int) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    by_color = {int(stripe["color"]): stripe for stripe in stripes}
    for color in draw_order:
        _paint_stripe(out, by_color[int(color)])
    return out


def _extract_specs(grid: np.ndarray, bg: int) -> list[dict] | None:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    specs: list[dict] = []

    for color in _colors_present(g, bg):
        coords = np.argwhere(g == int(color))
        if coords.size == 0:
            return None
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)

        if h <= 2 and w >= 3:
            rows = sorted(set(int(r) for r in coords[:, 0]))
            candidates: list[int] = []
            if len(rows) == 2:
                if rows[1] != rows[0] + 1:
                    return None
                candidates = [int(rows[0])]
            elif len(rows) == 1:
                row = int(rows[0])
                if row > 0:
                    candidates.append(row - 1)
                if row + 1 < H:
                    candidates.append(row)
            else:
                return None
            candidates = [start for start in candidates if all(start <= row <= start + 1 for row in rows)]
            if not candidates:
                return None
            specs.append(
                {
                    "color": int(color),
                    "orient": "H",
                    "candidates": sorted(set(int(v) for v in candidates)),
                }
            )
            continue

        if w <= 2 and h >= 3:
            cols = sorted(set(int(c) for c in coords[:, 1]))
            candidates = []
            if len(cols) == 2:
                if cols[1] != cols[0] + 1:
                    return None
                candidates = [int(cols[0])]
            elif len(cols) == 1:
                col = int(cols[0])
                if col > 0:
                    candidates.append(col - 1)
                if col + 1 < W:
                    candidates.append(col)
            else:
                return None
            candidates = [start for start in candidates if all(start <= col <= start + 1 for col in cols)]
            if not candidates:
                return None
            specs.append(
                {
                    "color": int(color),
                    "orient": "V",
                    "candidates": sorted(set(int(v) for v in candidates)),
                }
            )
            continue

        return None

    return specs


def _unique_topological_order(colors: list[int], edges: dict[int, set[int]]) -> list[int] | None:
    indegree = {int(color): 0 for color in colors}
    for src in colors:
        for dst in edges[int(src)]:
            indegree[int(dst)] += 1

    ready = [int(color) for color in colors if indegree[int(color)] == 0]
    order: list[int] = []
    while ready:
        if len(ready) != 1:
            return None
        node = int(ready.pop())
        order.append(node)
        for nxt in sorted(edges[node]):
            indegree[int(nxt)] -= 1
            if indegree[int(nxt)] == 0:
                ready.append(int(nxt))

    if len(order) != len(colors):
        return None
    return order


def _infer_stack(grid: np.ndarray) -> dict | None:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    specs = _extract_specs(g, bg)
    if not specs:
        return None

    colors = [int(spec["color"]) for spec in specs]
    solutions: dict[tuple, dict] = {}
    for starts in itertools.product(*(spec["candidates"] for spec in specs)):
        stripes = []
        for spec, start in zip(specs, starts):
            stripes.append(
                {
                    "color": int(spec["color"]),
                    "orient": spec["orient"],
                    "start": int(start),
                }
            )

        edges = {int(color): set() for color in colors}
        valid = True
        for r in range(g.shape[0]):
            for c in range(g.shape[1]):
                cover = [int(stripe["color"]) for stripe in stripes if _covers(stripe, r, c)]
                seen = int(g[r, c])
                if not cover:
                    if seen != int(bg):
                        valid = False
                        break
                    continue
                if seen == int(bg) or seen not in cover:
                    valid = False
                    break
                for other in cover:
                    if int(other) != seen:
                        edges[seen].add(int(other))
            if not valid:
                break
        if not valid:
            continue

        order = _unique_topological_order(colors, edges)
        if order is None:
            continue

        reconstructed = _render(g.shape, stripes, list(reversed(order)), bg)
        if not np.array_equal(reconstructed, g):
            continue

        key = (
            tuple(sorted((int(s["color"]), s["orient"], int(s["start"])) for s in stripes)),
            tuple(int(v) for v in order),
        )
        solutions[key] = {
            "bg": int(bg),
            "stripes": stripes,
            "order": [int(v) for v in order],
        }

    if len(solutions) != 1:
        return None
    return next(iter(solutions.values()))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the global stripe stacking order from occlusion evidence and redraw the scene with the stack reversed.

    Input structure:
    - The grid contains several full-length stripes, each exactly two cells thick and either horizontal or vertical.
    - Stripe colors are all distinct, and a covered cell shows only the color of the highest stripe among the ones passing through it.
    - Some same-direction stripes are offset by one row or column, so they overlap on one shared line in addition to the horizontal/vertical crossings.

    Rule:
    - Recover every stripe's full position from its visible cells, allowing for one hidden row or column inside an overlap.
    - For each cell, compare the displayed color with every other stripe that also covers that cell to collect "above/below" constraints.
    - Use the unique topological order implied by those constraints as the original top-to-bottom stack, then repaint the stripes with that order reversed.

    Key insight:
    - A visible color proves that stripe sits above every other stripe covering the same cell, so the puzzle reduces to recovering stripe locations and then solving a unique global ordering problem by transitive closure.
    """
    solution = _infer_stack(np.asarray(grid, dtype=int))
    assert solution is not None, "expected a uniquely solvable stripe stack"
    return _render(grid.shape, solution["stripes"], solution["order"], solution["bg"])


def _sample_axis_starts(span: int, count: int, overlap: bool, rng: np.random.Generator) -> list[int] | None:
    max_start = int(span) - 2
    if max_start < 1:
        return None

    starts: list[int] = []
    if overlap:
        if max_start < 1:
            return None
        base = int(rng.integers(0, max_start))
        starts.extend([base, base + 1])
    while len(starts) < count:
        choices = [
            start
            for start in range(max_start + 1)
            if start not in starts and all(abs(start - other) >= 2 for other in starts)
        ]
        if not choices:
            return None
        starts.append(int(rng.choice(choices)))
    return sorted(int(v) for v in starts)


def _visible_profile_ok(grid: np.ndarray, stripe: dict) -> bool:
    coords = np.argwhere(grid == int(stripe["color"]))
    if coords.size == 0:
        return False
    rows = sorted(set(int(r) for r in coords[:, 0]))
    cols = sorted(set(int(c) for c in coords[:, 1]))
    if stripe["orient"] == "H":
        return len(rows) in (1, 2) and int(coords[:, 1].max() - coords[:, 1].min() + 1) >= 3
    return len(cols) in (1, 2) and int(coords[:, 0].max() - coords[:, 0].min() + 1) >= 3


def _cover_multiplicity(shape: tuple[int, int], stripes: list[dict]) -> np.ndarray:
    counts = np.zeros(shape, dtype=int)
    for stripe in stripes:
        if stripe["orient"] == "H":
            counts[int(stripe["start"]) : int(stripe["start"]) + 2, :] += 1
        else:
            counts[:, int(stripe["start"]) : int(stripe["start"]) + 2] += 1
    return counts


def _sample_candidate(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, list[int]] | None:
    H = int(rng.integers(12, 25))
    W = int(rng.integers(12, 25))

    h_overlap = bool(int(rng.integers(0, 2)))
    v_overlap = bool(int(rng.integers(0, 2)))
    if not h_overlap and not v_overlap:
        if bool(int(rng.integers(0, 2))):
            h_overlap = True
        else:
            v_overlap = True

    nh = 2 + int(h_overlap and bool(int(rng.integers(0, 2))))
    nv = 2 + int(v_overlap and bool(int(rng.integers(0, 2))))
    if nh + nv > len(PALETTE):
        return None

    row_starts = _sample_axis_starts(H, nh, h_overlap, rng)
    col_starts = _sample_axis_starts(W, nv, v_overlap, rng)
    if row_starts is None or col_starts is None:
        return None

    colors = list(int(v) for v in rng.choice(PALETTE, size=nh + nv, replace=False))
    stripes: list[dict] = []
    for color, start in zip(colors[:nh], row_starts):
        stripes.append({"color": int(color), "orient": "H", "start": int(start)})
    for color, start in zip(colors[nh:], col_starts):
        stripes.append({"color": int(color), "orient": "V", "start": int(start)})

    order_top_to_bottom = [int(v) for v in rng.permutation(colors)]
    inp = _render((H, W), stripes, list(reversed(order_top_to_bottom)), BACKGROUND)
    out = _render((H, W), stripes, order_top_to_bottom, BACKGROUND)

    if np.array_equal(inp, out):
        return None

    if any(not _visible_profile_ok(inp, stripe) for stripe in stripes):
        return None

    overlaps = _cover_multiplicity((H, W), stripes)
    if int(overlaps.max()) < 3:
        return None

    solution = _infer_stack(inp)
    if solution is None:
        return None
    if solution["order"] != order_top_to_bottom:
        return None
    if not np.array_equal(out, main(inp)):
        return None

    return inp, out, order_top_to_bottom


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        sampled = _sample_candidate(rng)
        if sampled is not None:
            inp, _out, _order = sampled
            return inp
    raise RuntimeError("failed to generate a valid pressure-stack stripe input")
