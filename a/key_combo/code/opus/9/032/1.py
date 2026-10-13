from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
MARKER_COLORS = (BLUE, GREEN, YELLOW)

DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

GEN_RETRIES = 400
GEN_GRID_ROWS = (14, 19)
GEN_GRID_COLS = (18, 24)
GEN_FRAME_COUNT = (2, 4)
GEN_MARKER_COUNT = (3, 6)
GEN_GAP = 2


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            q = deque([(r, c)])
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _frame_candidate_ok(
    grid: np.ndarray,
    comp_set: set[tuple[int, int]],
    bbox: tuple[int, int, int, int],
) -> bool:
    r0, c0, r1, c1 = bbox
    if r1 - r0 < 4 or c1 - c0 < 4:
        return False
    top = grid[r0, c0 : c1 + 1]
    bottom = grid[r1, c0 : c1 + 1]
    left = grid[r0 : r1 + 1, c0]
    right = grid[r0 : r1 + 1, c1]
    sides = (top, bottom, left, right)
    for side in sides:
        bad = [int(v) for v in side if int(v) not in (BG, GREY)]
        if bad:
            return False
        if np.count_nonzero(side == GREY) < len(side) - 1:
            return False
    if int(grid[r0, c0]) != GREY or int(grid[r0, c1]) != GREY or int(grid[r1, c0]) != GREY or int(grid[r1, c1]) != GREY:
        return False

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    if np.any(interior == GREY):
        return False
    red_inside = {(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1) if int(grid[r, c]) == RED}
    if red_inside != comp_set:
        return False
    return True


def _frame_infos(grid: np.ndarray) -> list[dict]:
    infos = []
    used_bboxes: set[tuple[int, int, int, int]] = set()
    for comp in _components_of_color(grid, RED):
        comp_set = set(comp)
        mr0, mc0, mr1, mc1 = _bbox(comp)
        found = None
        for r0 in range(max(0, mr0 - 8), mr0):
            for c0 in range(max(0, mc0 - 8), mc0):
                for r1 in range(mr1 + 1, min(grid.shape[0], mr1 + 10)):
                    for c1 in range(mc1 + 1, min(grid.shape[1], mc1 + 10)):
                        bbox = (r0, c0, r1, c1)
                        if _frame_candidate_ok(grid, comp_set, bbox):
                            if found is None or (r1 - r0 + 1) * (c1 - c0 + 1) < (found[2] - found[0] + 1) * (found[3] - found[1] + 1):
                                found = bbox
        if found is None:
            continue
        if found in used_bboxes:
            continue
        used_bboxes.add(found)
        r0, c0, r1, c1 = found
        infos.append({"bbox": found, "inner": (r0 + 1, c0 + 1, r1 - 1, c1 - 1)})
    infos.sort(key=lambda item: item["bbox"])
    return infos


def _mirror_step(grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    out = np.array(grid, dtype=int, copy=True)
    infos = _frame_infos(out)
    for info in infos:
        ir0, ic0, ir1, ic1 = info["inner"]
        inner_h = ir1 - ir0 + 1
        inner_w = ic1 - ic0 + 1
        vertical = inner_w >= inner_h
        motif = [(r, c) for r in range(ir0, ir1 + 1) for c in range(ic0, ic1 + 1) if int(grid[r, c]) == RED]
        info["vertical"] = vertical
        info["original_red"] = set(motif)
        mirrored: set[tuple[int, int]] = set()
        for r, c in motif:
            rr = r
            cc = ic0 + ic1 - c if vertical else c
            if not vertical:
                rr = ir0 + ir1 - r
            if (rr, cc) not in info["original_red"]:
                mirrored.add((rr, cc))
                out[rr, cc] = ORANGE
        info["mirrored_orange"] = mirrored
    return out, infos


def _ordered_markers(grid: np.ndarray) -> list[tuple[int, int, int, tuple[int, int]]]:
    h, w = grid.shape
    ordered: list[tuple[int, int, int, tuple[int, int]]] = []
    seen: set[tuple[int, int]] = set()

    def add(r: int, c: int, step: tuple[int, int]) -> None:
        if (r, c) in seen:
            return
        color = int(grid[r, c])
        if color in MARKER_COLORS:
            ordered.append((r, c, color, step))
            seen.add((r, c))

    for c in range(w):
        add(0, c, (1, 0))
    for r in range(1, h):
        add(r, w - 1, (0, -1))
    for c in range(w - 2, -1, -1):
        add(h - 1, c, (-1, 0))
    for r in range(h - 2, 0, -1):
        add(r, 0, (0, 1))
    return ordered


def _apply_rays(base_grid: np.ndarray, out: np.ndarray) -> np.ndarray:
    h, w = base_grid.shape
    for r0, c0, color, (dr, dc) in _ordered_markers(base_grid):
        r = r0 + dr
        c = c0 + dc
        while _in_bounds((h, w), r, c):
            if int(base_grid[r, c]) == BG:
                out[r, c] = color
                r += dr
                c += dc
                continue
            if int(out[r, c]) == ORANGE:
                out[r, c] = color
            break
    return out


def _transform(grid: np.ndarray) -> np.ndarray:
    base, _infos = _mirror_step(grid)
    out = base.copy()
    return _apply_rays(base, out)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each framed motif by mirroring it, then let edge rays stop on original pixels but absorb mirrored ones.

    Input structure:
    - A black background contains several separated grey framed regions with one-cell-wide borders and black openings.
    - Each frame encloses a small red 4-connected motif placed on one side of the frame interior.
    - Single blue, green, or yellow marker pixels lie on the outer grid border and fire inward rays.

    Rule:
    - For every grey frame, inspect its interior bounding box: if the interior is wider than tall, or square, mirror the red motif left-right; if taller than wide, mirror it up-down. The mirrored copy is orange and the original red pixels stay red.
    - Process border markers in this exact order: top edge left-to-right, right edge top-to-bottom, bottom edge right-to-left, then left edge bottom-to-top. Each marker paints along the inward ray direction.
    - A ray can traverse any cell that was black immediately after the mirroring step, recoloring that traversed path to the marker color even if an earlier ray already painted there. At the first non-black cell from the mirrored scene, the ray stops one cell before it; the only exception is an orange mirrored pixel, which is recolored to the ray color and then becomes a normal blocker for later rays.

    Key insight:
    - The mirrored orange copy is the only foreground that behaves differently from the original scene: red motif pixels and grey frame pixels merely block rays, while orange pixels are absorbable once, so the frame aspect ratio indirectly controls later ray endpoints.
    """
    return _transform(np.asarray(grid, dtype=int))


def _draw_frame_patch(inner_h: int, inner_w: int, gate_index: int) -> np.ndarray:
    patch = np.zeros((inner_h + 2, inner_w + 2), dtype=int)
    patch[0, :] = GREY
    patch[-1, :] = GREY
    patch[:, 0] = GREY
    patch[:, -1] = GREY
    patch[0, gate_index + 1] = BG
    patch[-1, gate_index + 1] = BG
    return patch


def _random_connected_shape(cells: list[tuple[int, int]], size: int) -> list[tuple[int, int]] | None:
    start = random.choice(cells)
    chosen = {start}
    frontier = set()
    for dr, dc in DIRS4:
        nxt = (start[0] + dr, start[1] + dc)
        if nxt in cells:
            frontier.add(nxt)
    while len(chosen) < size:
        options = [p for p in frontier if p in cells and p not in chosen]
        if not options:
            return None
        cell = random.choice(options)
        chosen.add(cell)
        frontier.discard(cell)
        for dr, dc in DIRS4:
            nxt = (cell[0] + dr, cell[1] + dc)
            if nxt in cells and nxt not in chosen:
                frontier.add(nxt)
    return list(chosen)


def _shape_is_connected(cells: set[tuple[int, int]]) -> bool:
    start = next(iter(cells))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS4:
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return len(seen) == len(cells)


def _make_motif(inner_h: int, inner_w: int, vertical: bool) -> list[tuple[int, int]] | None:
    center_r = inner_h // 2
    center_c = inner_w // 2
    for _ in range(120):
        if vertical:
            side = random.choice(("left", "right"))
            cols = range(0, center_c) if side == "left" else range(center_c + 1, inner_w)
            cells = [(r, c) for r in range(inner_h) for c in cols]
            must_include = random.choice([("row", center_r), ("col", None), ("none", None)])
        else:
            side = random.choice(("top", "bottom"))
            rows = range(0, center_r) if side == "top" else range(center_r + 1, inner_h)
            cells = [(r, c) for r in rows for c in range(inner_w)]
            must_include = random.choice([("col", center_c), ("row", None), ("none", None)])
        if len(cells) < 3:
            continue
        size = random.randint(3, min(5, len(cells)))
        shape = _random_connected_shape(cells, size)
        if shape is None:
            continue
        shape_set = set(shape)
        if vertical:
            mirror = {(r, inner_w - 1 - c) for r, c in shape_set}
        else:
            mirror = {(inner_h - 1 - r, c) for r, c in shape_set}
        if shape_set & mirror:
            continue
        if must_include[0] == "row" and not any(r == must_include[1] for r, _ in shape_set):
            continue
        if must_include[0] == "col":
            if vertical:
                if not any(c in {center_c - 1, center_c + 1} for _, c in shape_set if 0 <= c < inner_w):
                    continue
            else:
                if not any(c == center_c for _, c in shape_set):
                    continue
        if not _shape_is_connected(shape_set):
            continue
        return sorted(shape)
    return None


def _intersects_bbox_path(
    shape: tuple[int, int],
    start: tuple[int, int],
    step: tuple[int, int],
    bbox: tuple[int, int, int, int],
) -> bool:
    h, w = shape
    r, c = start[0] + step[0], start[1] + step[1]
    r0, c0, r1, c1 = bbox
    while 0 <= r < h and 0 <= c < w:
        if r0 <= r <= r1 and c0 <= c <= c1:
            return True
        r += step[0]
        c += step[1]
    return False


def _simulate_candidate(
    base_grid: np.ndarray,
    frames: list[dict],
    start: tuple[int, int],
    step: tuple[int, int],
) -> dict | None:
    h, w = base_grid.shape
    r = start[0] + step[0]
    c = start[1] + step[1]
    traversed: list[tuple[int, int]] = []
    crossed_frames = {
        i
        for i, info in enumerate(frames)
        if _intersects_bbox_path((h, w), start, step, info["bbox"])
    }
    if not crossed_frames:
        return None
    while 0 <= r < h and 0 <= c < w:
        val = int(base_grid[r, c])
        if val == BG:
            traversed.append((r, c))
            r += step[0]
            c += step[1]
            continue
        kind = "orange" if val == ORANGE else "red" if val == RED else "grey" if val == GREY else "other"
        hit_frame = None
        for i, info in enumerate(frames):
            br0, bc0, br1, bc1 = info["bbox"]
            if br0 <= r <= br1 and bc0 <= c <= bc1:
                hit_frame = i
                break
        return {
            "start": start,
            "step": step,
            "path": tuple(traversed),
            "path_set": set(traversed),
            "hit": (r, c),
            "kind": kind,
            "hit_frame": hit_frame,
            "crossed_frames": crossed_frames,
        }
    return {
        "start": start,
        "step": step,
        "path": tuple(traversed),
        "path_set": set(traversed),
        "hit": None,
        "kind": "none",
        "hit_frame": None,
        "crossed_frames": crossed_frames,
    }


def _enumerate_candidates(base_grid: np.ndarray, frames: list[dict]) -> list[dict]:
    h, w = base_grid.shape
    out: list[dict] = []
    for c in range(1, w - 1):
        if int(base_grid[0, c]) == BG:
            cand = _simulate_candidate(base_grid, frames, (0, c), (1, 0))
            if cand is not None:
                cand["edge_rank"] = (0, c)
                out.append(cand)
    for r in range(1, h - 1):
        if int(base_grid[r, w - 1]) == BG:
            cand = _simulate_candidate(base_grid, frames, (r, w - 1), (0, -1))
            if cand is not None:
                cand["edge_rank"] = (1, r)
                out.append(cand)
    for c in range(w - 2, 0, -1):
        if int(base_grid[h - 1, c]) == BG:
            cand = _simulate_candidate(base_grid, frames, (h - 1, c), (-1, 0))
            if cand is not None:
                cand["edge_rank"] = (2, -c)
                out.append(cand)
    for r in range(h - 2, 0, -1):
        if int(base_grid[r, 0]) == BG:
            cand = _simulate_candidate(base_grid, frames, (r, 0), (0, 1))
            if cand is not None:
                cand["edge_rank"] = (3, -r)
                out.append(cand)
    out.sort(key=lambda item: item["edge_rank"])
    return out


def _frame_summary(frames: list[dict]) -> tuple[int, int]:
    vertical = sum(1 for info in frames if info["vertical"])
    horizontal = len(frames) - vertical
    return vertical, horizontal


def _pick_markers(base_grid: np.ndarray, frames: list[dict]) -> list[dict] | None:
    candidates = _enumerate_candidates(base_grid, frames)
    oranges = [c for c in candidates if c["kind"] == "orange"]
    normals = [c for c in candidates if c["kind"] in {"grey", "red"}]
    if not oranges or not normals:
        return None

    selected: list[dict] = []
    used_starts: set[tuple[int, int]] = set()

    def choose(pool: list[dict], *, share_orange_with: tuple[int, int] | None = None, overlap_with: set[tuple[int, int]] | None = None) -> dict | None:
        options = [c for c in pool if c["start"] not in used_starts]
        if share_orange_with is not None:
            shared = [c for c in options if c["hit"] == share_orange_with]
            if shared:
                options = shared
        if overlap_with is not None:
            overlapped = [c for c in options if c["path_set"] & overlap_with]
            if overlapped:
                options = overlapped
        if not options:
            return None
        return random.choice(options)

    first = choose(oranges)
    if first is None:
        return None
    selected.append(first)
    used_starts.add(first["start"])

    second = choose(normals, overlap_with=first["path_set"])
    if second is None:
        second = choose(normals)
    if second is None:
        return None
    selected.append(second)
    used_starts.add(second["start"])

    special = None
    if first["hit"] is not None:
        share = choose(oranges, share_orange_with=first["hit"])
        if share is not None:
            special = share
    if special is None:
        combined = first["path_set"] | second["path_set"]
        special = choose(candidates, overlap_with=combined)
    if special is not None:
        selected.append(special)
        used_starts.add(special["start"])

    target_n = random.randint(*GEN_MARKER_COUNT)
    remain = [c for c in candidates if c["start"] not in used_starts]
    random.shuffle(remain)
    for cand in remain:
        if len(selected) >= target_n:
            break
        selected.append(cand)
        used_starts.add(cand["start"])

    if len(selected) < 3:
        return None
    return selected


def _special_marker_pairs(selected: list[dict]) -> list[tuple[int, int]]:
    pairs = []
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            same_orange = (
                selected[i]["kind"] == "orange"
                and selected[j]["kind"] == "orange"
                and selected[i]["hit"] is not None
                and selected[i]["hit"] == selected[j]["hit"]
            )
            overlap = bool(selected[i]["path_set"] & selected[j]["path_set"])
            if same_orange or overlap:
                pairs.append((i, j))
    return pairs


def _assign_marker_colors(selected: list[dict]) -> list[int] | None:
    pairs = _special_marker_pairs(selected)
    if not pairs:
        return None
    for _ in range(40):
        colors = [random.choice(MARKER_COLORS) for _ in selected]
        if len(set(colors)) == 1:
            colors[random.randrange(len(colors))] = random.choice([c for c in MARKER_COLORS if c != colors[0]])
        if any(colors[i] != colors[j] for i, j in pairs):
            return colors
    return None


def _paint_frames(grid: np.ndarray, specs: list[dict]) -> None:
    for spec in specs:
        r0, c0 = spec["top_left"]
        patch = _draw_frame_patch(spec["inner_h"], spec["inner_w"], spec["gate_index"])
        h, w = patch.shape
        mask = patch == GREY
        grid[r0 : r0 + h, c0 : c0 + w][mask] = GREY
        for rr, cc in spec["motif"]:
            grid[r0 + 1 + rr, c0 + 1 + cc] = RED


def _make_frame_specs(h: int, w: int) -> list[dict] | None:
    count = random.randint(*GEN_FRAME_COUNT)
    kinds = []
    if count >= 2:
        kinds.extend(["wide", "tall"])
    while len(kinds) < count:
        kinds.append(random.choice(["wide", "tall", "square"]))
    random.shuffle(kinds)

    specs: list[dict] = []
    occupied: list[tuple[int, int, int, int]] = []
    for kind in kinds:
        placed = None
        for _ in range(200):
            if kind == "wide":
                inner_h = random.choice((3, 5))
                inner_w = random.choice(tuple(v for v in (5, 7) if v > inner_h))
            elif kind == "tall":
                inner_w = random.choice((3, 5))
                inner_h = random.choice(tuple(v for v in (5, 7) if v > inner_w))
            else:
                inner_h = inner_w = random.choice((3, 5))
            vertical = inner_w >= inner_h
            outer_h = inner_h + 2
            outer_w = inner_w + 2
            if outer_h >= h - 2 or outer_w >= w - 2:
                continue
            r0 = random.randint(1, h - outer_h - 1)
            c0 = random.randint(1, w - outer_w - 1)
            bbox = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
            ok = True
            for br0, bc0, br1, bc1 in occupied:
                if not (bbox[2] + GEN_GAP < br0 or br1 + GEN_GAP < bbox[0] or bbox[3] + GEN_GAP < bc0 or bc1 + GEN_GAP < bbox[1]):
                    ok = False
                    break
            if not ok:
                continue
            motif = _make_motif(inner_h, inner_w, vertical)
            if motif is None:
                continue
            motif_set = set(motif)
            lines = sorted({c for _, c in motif_set} | {inner_w - 1 - c for _, c in motif_set})
            if not lines:
                continue
            placed = {
                "top_left": (r0, c0),
                "bbox": bbox,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "vertical": vertical,
                "gate_index": random.choice(lines),
                "motif": motif,
            }
            break
        if placed is None:
            return None
        specs.append(placed)
        occupied.append(placed["bbox"])
    return specs


def _expected_from_parts(inp: np.ndarray) -> np.ndarray:
    mirrored, _infos = _mirror_step(inp)
    out = mirrored.copy()
    return _apply_rays(mirrored, out)


def _validate_semantics(inp: np.ndarray, out: np.ndarray) -> bool:
    mirrored, infos = _mirror_step(inp)
    frames = infos
    if not frames:
        return False
    verts, hors = _frame_summary(frames)
    if verts == 0 or hors == 0:
        return False
    if not any(info["mirrored_orange"] for info in frames):
        return False

    markers = _ordered_markers(inp)
    if not (3 <= len(markers) <= 6):
        return False

    # Re-run incrementally so branch checks use the true sequential state.
    current = mirrored.copy()
    seen_path_colors: dict[tuple[int, int], int] = {}
    orange_hits = 0
    normal_hits = 0
    overlaps = 0
    shared_orange = 0
    seen_orange_hits: set[tuple[int, int]] = set()
    for r0, c0, color, step in markers:
        r = r0 + step[0]
        c = c0 + step[1]
        traversed: list[tuple[int, int]] = []
        hit = None
        while _in_bounds(current.shape, r, c):
            if int(mirrored[r, c]) == BG:
                traversed.append((r, c))
                r += step[0]
                c += step[1]
                continue
            hit = (r, c)
            val = int(current[r, c])
            base_val = int(mirrored[r, c])
            if base_val == ORANGE and val == ORANGE:
                orange_hits += 1
                if hit in seen_orange_hits:
                    shared_orange += 1
                seen_orange_hits.add(hit)
            elif base_val in (GREY, RED) or (base_val == ORANGE and val != ORANGE):
                normal_hits += 1
            break
        for cell in traversed:
            prev = seen_path_colors.get(cell)
            if prev is not None and prev != color:
                overlaps += 1
            seen_path_colors[cell] = color
        rr, cc = r0 + step[0], c0 + step[1]
        while _in_bounds(current.shape, rr, cc):
            if int(mirrored[rr, cc]) == BG:
                current[rr, cc] = color
                rr += step[0]
                cc += step[1]
                continue
            if int(current[rr, cc]) == ORANGE:
                current[rr, cc] = color
            break

    if orange_hits == 0 or normal_hits == 0:
        return False
    if overlaps == 0 and shared_orange == 0:
        return False
    if np.array_equal(inp, out) or np.array_equal(mirrored, out):
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(*GEN_GRID_ROWS)
    w = random.randint(*GEN_GRID_COLS)
    specs = _make_frame_specs(h, w)
    if specs is None:
        return None

    inp = np.zeros((h, w), dtype=int)
    _paint_frames(inp, specs)

    mirrored, frames = _mirror_step(inp)
    if len(frames) != len(specs):
        return None
    selected = _pick_markers(mirrored, frames)
    if selected is None:
        return None

    colors = _assign_marker_colors(selected)
    if colors is None:
        return None
    for cand, color in zip(selected, colors):
        r, c = cand["start"]
        inp[r, c] = color

    out = _expected_from_parts(inp)
    if not np.array_equal(out, main(inp)):
        return None
    if not _validate_semantics(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC instance")
