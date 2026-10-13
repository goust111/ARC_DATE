from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
DEFLECTOR = 5
SOURCE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == int(color)
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _normalize(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    base = coords.min(axis=0)
    shifted = coords - base
    return tuple(sorted((int(r), int(c)) for r, c in shifted))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _translate_shape(shape: tuple[tuple[int, int], ...], dr: int, dc: int) -> np.ndarray:
    return np.array([(int(r + dr), int(c + dc)) for r, c in shape], dtype=int)


def _touches_forbidden(coords: np.ndarray, forbidden: set[tuple[int, int]]) -> bool:
    return any((int(r), int(c)) in forbidden for r, c in coords)


def _extend_forbidden(coords: np.ndarray, forbidden: set[tuple[int, int]], shape: tuple[int, int]) -> None:
    h, w = shape
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w:
                    forbidden.add((nr, nc))


def _paint(grid: np.ndarray, coords: np.ndarray, color: int) -> None:
    grid[coords[:, 0], coords[:, 1]] = int(color)


def _shape_library() -> list[tuple[tuple[int, int], ...]]:
    shapes = []
    seen = set()
    cells = [(r, c) for r in range(3) for c in range(3)]
    for area in range(3, 6):
        for combo in itertools.combinations(cells, area):
            arr = np.array(combo, dtype=int)
            patch = np.zeros((3, 3), dtype=int)
            patch[arr[:, 0], arr[:, 1]] = 1
            if len(_components_of_color(patch, 1)) != 1:
                continue
            key = _normalize(arr)
            if key in seen:
                continue
            seen.add(key)
            shapes.append(key)
    return shapes


SHAPES = _shape_library()


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    source_rows = [r for r in range(h) if int(g[r, 0]) not in (BG, DEFLECTOR)]
    assert 2 <= len(source_rows) <= 3, "expected 2-3 left-edge sources"

    sources = []
    used_colors = set()
    for r in source_rows:
        color = int(g[r, 0])
        assert color not in used_colors, "source colors must be distinct"
        used_colors.add(color)
        comps = _components_of_color(g, color)
        owner = None
        for comp in comps:
            if np.any((comp == np.array([r, 0])).all(axis=1)):
                owner = comp
                break
        assert owner is not None, "source must belong to a same-color component"
        assert len(comps) == 1, "each source color should form one connected component"
        template = owner[~((owner[:, 0] == r) & (owner[:, 1] == 0))]
        assert 3 <= len(template) <= 5, "template must be a small connected shape"
        assert np.any((template[:, 0] == r) & (template[:, 1] == 1)), "template must touch the source on its right"
        sources.append(
            {
                "row": int(r),
                "color": color,
                "shape": _normalize(template),
                "coords": owner,
            }
        )

    shape_to_source = {src["shape"]: src for src in sources}
    assert len(shape_to_source) == len(sources), "source template shapes must be unique"

    deflectors = []
    for idx, comp in enumerate(_components_of_color(g, DEFLECTOR)):
        assert 3 <= len(comp) <= 5, "deflectors must be small 4-connected shapes"
        r0, c0, r1, c1 = _bbox(comp)
        assert 0 < r0 and r1 < h - 1 and 0 < c0 and c1 < w - 1, "deflectors must stay away from edges"
        shape = _normalize(comp)
        assert shape in shape_to_source, "every deflector must match one source template shape"
        deflectors.append(
            {
                "id": idx,
                "coords": comp,
                "bbox": (r0, c0, r1, c1),
                "shape": shape,
                "owner_color": int(shape_to_source[shape]["color"]),
            }
        )

    assert 3 <= len(deflectors) <= 5, "expected 3-5 deflectors"
    return {
        "sources": sorted(sources, key=lambda x: x["row"]),
        "deflectors": sorted(deflectors, key=lambda x: (x["bbox"][1], x["bbox"][0])),
    }


def _simulate(grid: np.ndarray, analysis: dict) -> tuple[np.ndarray, dict[int, int], dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    deflectors_by_col: dict[int, list[dict]] = {}
    for d in analysis["deflectors"]:
        deflectors_by_col.setdefault(d["bbox"][1], []).append(d)

    activated: dict[int, int] = {}
    contacts = {
        d["id"]: {"matched": False, "mismatched": False, "touched_by": set(), "owner_hit": False}
        for d in analysis["deflectors"]
    }
    painted_by: dict[tuple[int, int], int] = {}

    for src in analysis["sources"]:
        row = int(src["row"])
        color = int(src["color"])
        shape = src["shape"]
        for c in range(1, w):
            for d in deflectors_by_col.get(c, []):
                r0, _c0, r1, _c1 = d["bbox"]
                if r0 <= row <= r1:
                    contacts[d["id"]]["touched_by"].add(color)
                    if d["shape"] == shape:
                        contacts[d["id"]]["matched"] = True
                        contacts[d["id"]]["owner_hit"] = True
                        row += 1
                        activated[d["id"]] = color
                    else:
                        contacts[d["id"]]["mismatched"] = True
            if not (0 <= row < h):
                break
            if int(out[row, c]) == BG:
                prev = painted_by.get((row, c))
                if prev is not None and prev != color:
                    raise AssertionError("different flows should not need the same background cell")
                painted_by[(row, c)] = color
                out[row, c] = color

    for d in analysis["deflectors"]:
        if d["id"] in activated:
            _paint(out, d["coords"], activated[d["id"]])
    return out, activated, contacts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route each left-edge color stream through only the gray deflectors whose shapes match its local template.

    Input structure:
    - A black background grid has 2-3 colored source pixels on the left edge.
    - Each source is attached to a same-colored 4-connected template immediately to its right.
    - The interior contains 3-5 gray 4-connected deflector shapes that do not touch the border.

    Rule:
    - Normalize every source template and every gray deflector by translating its pixels to the upper-left of its bbox.
    - Each source emits a horizontal stream to the right; when the stream first reaches the left edge column of a deflector whose row span contains the stream, only a shape-matching deflector pushes that stream down by one cell before it continues.
    - Streams color only background cells, matching deflections accumulate, and any deflector that actually caused a push is recolored to that stream's source color.

    Key insight:
    - Shape matching is not the final output step by itself; it gates which deflectors affect which stream, and only those successful gate-triggered deflectors are recolored.
    """
    analysis = _analyze_grid(np.asarray(grid, dtype=int))
    out, _activated, _contacts = _simulate(np.asarray(grid, dtype=int), analysis)
    return out


def _candidate_attach_rows(shape: tuple[tuple[int, int], ...]) -> list[int]:
    min_c = min(c for _, c in shape)
    return [r for r, c in shape if c == min_c]


def _place_sources(h: int, w: int, colors: list[int], shapes: list[tuple[tuple[int, int], ...]]) -> tuple[np.ndarray, list[dict]] | None:
    grid = np.zeros((h, w), dtype=int)
    forbidden: set[tuple[int, int]] = set()
    infos: list[dict] = []

    order = list(range(len(colors)))
    order.sort(key=lambda i: (_shape_dims(shapes[i])[0], _shape_dims(shapes[i])[1]), reverse=True)

    for i in order:
        color = int(colors[i])
        shape = shapes[i]
        sh, _sw = _shape_dims(shape)
        attach_rows = _candidate_attach_rows(shape)
        random.shuffle(attach_rows)
        placed = False
        for attach_r in attach_rows:
            candidate_rows = list(range(attach_r, h - (sh - attach_r) + 1))
            random.shuffle(candidate_rows)
            for source_row in candidate_rows:
                coords = _translate_shape(shape, source_row - attach_r, 1)
                comp = np.vstack([coords, np.array([[source_row, 0]], dtype=int)])
                if _touches_forbidden(comp, forbidden):
                    continue
                if coords[:, 0].min() < 0 or coords[:, 0].max() >= h or coords[:, 1].max() >= w:
                    continue
                _paint(grid, coords, color)
                grid[source_row, 0] = color
                _extend_forbidden(comp, forbidden, (h, w))
                infos.append({"row": int(source_row), "color": color, "shape": shape})
                placed = True
                break
            if placed:
                break
        if not placed:
            return None
    infos.sort(key=lambda x: x["row"])
    return grid, infos


def _slot_rows_before(source_rows: list[int], slot_owners: list[int | None]) -> list[list[int]]:
    rows = list(source_rows)
    snapshots = []
    for owner in slot_owners:
        snapshots.append(rows.copy())
        if owner is not None:
            rows[owner] += 1
    return snapshots


def _top_rows_for_hit(row: int, shape_h: int, h: int) -> list[int]:
    return [top for top in range(1, h - shape_h) if top <= row <= top + shape_h - 1]


def _place_deflectors(grid: np.ndarray, sources: list[dict], active_counts: list[int], inactive_count: int) -> tuple[np.ndarray, list[dict]] | None:
    h, w = grid.shape
    source_rows = [src["row"] for src in sources]
    forbidden = set()
    for r, c in np.argwhere(grid != BG):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = int(r + dr)
                nc = int(c + dc)
                if 0 <= nr < h and 0 <= nc < w:
                    forbidden.add((nr, nc))

    active_owners = []
    for idx, count in enumerate(active_counts):
        active_owners.extend([idx] * int(count))
    random.shuffle(active_owners)

    total = len(active_owners) + inactive_count
    col_pool = list(range(5, w - 3))
    if len(col_pool) < total:
        return None
    cols = sorted(random.sample(col_pool, total))
    inactive_slots = set(random.sample(range(total), inactive_count))

    slot_owners: list[int | None] = []
    active_iter = iter(active_owners)
    for slot in range(total):
        if slot in inactive_slots:
            slot_owners.append(None)
        else:
            slot_owners.append(next(active_iter))

    rows_before = _slot_rows_before(source_rows, slot_owners)
    deflectors = []

    for slot, c0 in enumerate(cols):
        snapshot = rows_before[slot]
        if slot_owners[slot] is not None:
            owner = int(slot_owners[slot])
            shape = sources[owner]["shape"]
            sh, sw = _shape_dims(shape)
            if c0 + sw - 1 >= w - 1:
                return None
            target_row = snapshot[owner]
            tops = _top_rows_for_hit(target_row, sh, h)
            random.shuffle(tops)
            for top in tops:
                coords = _translate_shape(shape, top, c0)
                if _touches_forbidden(coords, forbidden):
                    continue
                _paint(grid, coords, DEFLECTOR)
                _extend_forbidden(coords, forbidden, (h, w))
                deflectors.append({"owner": owner, "coords": coords, "active": True})
                break
            else:
                return None
            continue

        pairs = [(a, b) for a in range(len(sources)) for b in range(len(sources)) if a != b]
        random.shuffle(pairs)
        placed = False
        for owner, witness in pairs:
            shape = sources[owner]["shape"]
            sh, sw = _shape_dims(shape)
            if c0 + sw - 1 >= w - 1:
                continue
            witness_row = snapshot[witness]
            owner_row = snapshot[owner]
            tops = _top_rows_for_hit(witness_row, sh, h)
            tops = [top for top in tops if not (top <= owner_row <= top + sh - 1)]
            random.shuffle(tops)
            for top in tops:
                coords = _translate_shape(shape, top, c0)
                if _touches_forbidden(coords, forbidden):
                    continue
                _paint(grid, coords, DEFLECTOR)
                _extend_forbidden(coords, forbidden, (h, w))
                deflectors.append({"owner": owner, "coords": coords, "active": False, "witness": witness})
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    return grid, deflectors


def _semantic_ok(grid: np.ndarray) -> bool:
    try:
        analysis = _analyze_grid(grid)
        out, activated, contacts = _simulate(grid, analysis)
    except Exception:
        return False

    sources = analysis["sources"]
    deflectors = analysis["deflectors"]
    shape_counts = {src["shape"]: 0 for src in sources}
    for d in deflectors:
        shape_counts[d["shape"]] += 1
    if not all(count >= 1 for count in shape_counts.values()):
        return False

    if not activated or len(activated) == len(deflectors):
        return False

    if not any(len([d for d in deflectors if d["owner_color"] == src["color"] and d["id"] in activated]) >= 1 for src in sources):
        return False

    witness_exists = any(
        d["id"] not in activated and contacts[d["id"]]["mismatched"]
        for d in deflectors
    )
    if not witness_exists:
        return False

    if np.array_equal(out, grid):
        return False

    return True


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(13, 18)
    w = random.randint(15, 20)
    n_sources = random.choice([2, 3])
    colors = random.sample(SOURCE_COLORS, n_sources)
    shapes = random.sample(SHAPES, n_sources)

    placed = _place_sources(h, w, colors, shapes)
    if placed is None:
        return None
    grid, sources = placed

    active_counts = [1] * n_sources
    if random.random() < 0.7 and sum(active_counts) < 4:
        active_counts[random.randrange(n_sources)] += 1
    inactive_count = random.randint(1, min(2, 5 - sum(active_counts)))

    placed_deflectors = _place_deflectors(grid.copy(), sources, active_counts, inactive_count)
    if placed_deflectors is None:
        return None
    grid, _meta = placed_deflectors

    if not _semantic_ok(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
