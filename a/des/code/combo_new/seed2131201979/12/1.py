from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
FRAME_COLOR = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 500


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
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = _bbox(coords)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _mask_from_coords(coords: tuple[tuple[int, int], ...], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    return mask


def _is_connected(coords: set[tuple[int, int]]) -> bool:
    if not coords:
        return False
    start = next(iter(coords))
    q: deque[tuple[int, int]] = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in coords and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(coords)


def _is_rect_frame(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    expected = 2 * h + 2 * w - 4
    if len(coords) != expected:
        return False
    pts = set(coords)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            border = r in (r0, r1) or c in (c0, c1)
            if border != ((r, c) in pts):
                return False
    return True


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames = []
    for comp in _components(grid == FRAME_COLOR):
        if _is_rect_frame(comp):
            frames.append(_bbox(comp))
    frames.sort()
    assert frames, "expected at least one grey rectangular frame"
    return frames


def _extract_fragments(subgrid: np.ndarray) -> list[dict[str, object]]:
    fragments: list[dict[str, object]] = []
    for comp in _components(subgrid != BG):
        colors = {int(subgrid[r, c]) for r, c in comp}
        assert len(colors) == 1, "each fragment must be monochromatic"
        color = colors.pop()
        assert color != FRAME_COLOR, "frame color must not appear inside frame interiors"
        nrml = _normalize(comp)
        rr0, cc0, rr1, cc1 = _bbox(comp)
        fragments.append(
            {
                "color": int(color),
                "coords": nrml,
                "h": int(rr1 - rr0 + 1),
                "w": int(cc1 - cc0 + 1),
                "area": int(len(comp)),
            }
        )
    fragments.sort(key=lambda frag: int(frag["color"]))
    assert 2 <= len(fragments) <= 3, "expected 2-3 fragments per frame"
    colors = [int(frag["color"]) for frag in fragments]
    assert len(colors) == len(set(colors)), "fragment colors must be distinct"
    return fragments


def _outside_templates(
    grid: np.ndarray, frames: list[tuple[int, int, int, int]]
) -> dict[int, list[dict[str, object]]]:
    outside = np.ones_like(grid, dtype=bool)
    for r0, c0, r1, c1 in frames:
        outside[r0 : r1 + 1, c0 : c1 + 1] = False

    templates: dict[int, list[dict[str, object]]] = {}
    for color in np.unique(grid):
        color = int(color)
        if color in (BG, FRAME_COLOR):
            continue
        comps = _components((grid == color) & outside)
        if not comps:
            continue
        entries: list[dict[str, object]] = []
        for comp in comps:
            r0, c0, r1, c1 = _bbox(comp)
            nrml = _normalize(comp)
            mask = _mask_from_coords(nrml, (r1 - r0 + 1, c1 - c0 + 1))
            entries.append(
                {
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "h": int(r1 - r0 + 1),
                    "w": int(c1 - c0 + 1),
                    "mask": mask,
                }
            )
        templates[color] = entries
    return templates


def _candidate_placements(piece: dict[str, object], target_mask: np.ndarray) -> list[tuple[tuple[int, int], frozenset[tuple[int, int]]]]:
    h, w = target_mask.shape
    ph = int(piece["h"])
    pw = int(piece["w"])
    coords = tuple(piece["coords"])  # type: ignore[arg-type]
    placements: list[tuple[tuple[int, int], frozenset[tuple[int, int]]]] = []
    for r0 in range(h - ph + 1):
        for c0 in range(w - pw + 1):
            cells = frozenset((r0 + dr, c0 + dc) for dr, dc in coords)
            if all(target_mask[r, c] for r, c in cells):
                placements.append(((int(r0), int(c0)), cells))
    return placements


def _solve_packing(
    pieces: list[dict[str, object]],
    target_mask: np.ndarray,
    limit: int = 1,
) -> list[list[tuple[int, int]]]:
    target_area = int(target_mask.sum())
    assert sum(int(piece["area"]) for piece in pieces) == target_area, "piece areas must equal target area"

    all_candidates = [_candidate_placements(piece, target_mask) for piece in pieces]
    assert all(all_candidates), "every piece must have at least one feasible placement"

    placements: list[tuple[int, int] | None] = [None] * len(pieces)
    solutions: list[list[tuple[int, int]]] = []

    def dfs(remaining: set[int], occupied: set[tuple[int, int]]) -> None:
        if len(solutions) >= limit:
            return
        if not remaining:
            solutions.append([pos for pos in placements if pos is not None])
            return

        best_idx = -1
        best_options: list[tuple[tuple[int, int], frozenset[tuple[int, int]]]] | None = None
        for idx in remaining:
            options = [cand for cand in all_candidates[idx] if cand[1].isdisjoint(occupied)]
            if not options:
                return
            if best_options is None or len(options) < len(best_options):
                best_idx = idx
                best_options = options
        assert best_options is not None

        random.shuffle(best_options)
        next_remaining = set(remaining)
        next_remaining.remove(best_idx)
        for offset, cells in best_options:
            placements[best_idx] = offset
            dfs(next_remaining, occupied | set(cells))
            placements[best_idx] = None
            if len(solutions) >= limit:
                return

    dfs(set(range(len(pieces))), set())
    return solutions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble the colored fragments inside each grey frame so they translate into the outline shown by their matching outside templates.

    Input structure:
    - The background is black.
    - One or more grey one-pixel-thick rectangular hollow frames appear in the grid.
    - Each frame interior contains 2-3 separated monochromatic fragments of different colors.
    - Outside the frames, each fragment color also appears once as a monochromatic full-shape template whose bbox matches that frame's interior size.

    Rule:
    - For each frame, collect its interior fragment colors.
    - Match every fragment color to the outside template of the same color whose bbox size equals that frame interior.
    - Treat those matched templates as identical copies of the target silhouette.
    - Translate the interior fragments, without rotating or flipping them, so they tile that silhouette exactly while keeping each fragment's original color.
    - Leave the frame borders, the outside templates, and all other cells unchanged.

    Key insight:
    The outside same-color shapes are not patches to paste into the frames; they are monochrome copies of the final silhouette that the existing fragments must reconstruct by translation-only packing.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames = _find_frames(g)
    templates_by_color = _outside_templates(g, frames)
    out = g.copy()

    for r0, c0, r1, c1 in frames:
        ir0, ic0 = r0 + 1, c0 + 1
        ir1, ic1 = r1 - 1, c1 - 1
        assert ir0 <= ir1 and ic0 <= ic1, "frame must have a nonempty interior"
        interior = g[ir0 : ir1 + 1, ic0 : ic1 + 1]
        fragments = _extract_fragments(interior)
        h, w = interior.shape

        matched_masks = []
        for frag in fragments:
            color = int(frag["color"])
            candidates = [
                entry["mask"]
                for entry in templates_by_color.get(color, [])
                if int(entry["h"]) == h and int(entry["w"]) == w
            ]
            assert len(candidates) == 1, f"expected one matching template for color {color}"
            matched_masks.append(candidates[0])

        target_mask = matched_masks[0]
        for other in matched_masks[1:]:
            assert np.array_equal(target_mask, other), "same-frame templates must encode one common silhouette"

        solutions = _solve_packing(fragments, target_mask, limit=1)
        assert solutions, "frame fragments must tile the matched template"
        placement_by_fragment = solutions[0]

        out[ir0 : ir1 + 1, ic0 : ic1 + 1] = BG
        for frag, (pr, pc) in zip(fragments, placement_by_fragment):
            color = int(frag["color"])
            for dr, dc in frag["coords"]:  # type: ignore[index]
                out[ir0 + pr + dr, ic0 + pc + dc] = color

    return out


def _random_polyomino(area: int, max_h: int, max_w: int) -> tuple[tuple[int, int], ...] | None:
    for _ in range(80):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))

        coords = list(cells)
        if random.random() < 0.5:
            coords = [(-r, c) for r, c in coords]
        if random.random() < 0.5:
            coords = [(r, -c) for r, c in coords]
        for _ in range(random.randrange(4)):
            coords = [(c, -r) for r, c in coords]

        norm = _normalize(coords)
        r0, c0, r1, c1 = _bbox(norm)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h <= max_h and w <= max_w:
            return norm
    return None


def _expanded8(cells: frozenset[tuple[int, int]], h: int, w: int) -> frozenset[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return frozenset(out)


def _all_piece_placements(
    piece_coords: tuple[tuple[int, int], ...], h: int, w: int
) -> list[dict[str, object]]:
    r0, c0, r1, c1 = _bbox(piece_coords)
    ph = r1 - r0 + 1
    pw = c1 - c0 + 1
    out: list[dict[str, object]] = []
    for rr in range(h - ph + 1):
        for cc in range(w - pw + 1):
            cells = frozenset((rr + dr, cc + dc) for dr, dc in piece_coords)
            out.append(
                {
                    "offset": (int(rr), int(cc)),
                    "cells": cells,
                    "halo": _expanded8(cells, h, w),
                }
            )
    return out


def _touches(occupied: set[tuple[int, int]], cells: frozenset[tuple[int, int]]) -> bool:
    for r, c in cells:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in occupied:
                return True
    return False


def _search_final_arrangement(
    piece_coords: list[tuple[tuple[int, int], ...]], h: int, w: int
) -> list[tuple[int, int]] | None:
    options = [_all_piece_placements(coords, h, w) for coords in piece_coords]
    for _ in range(300):
        order = list(range(len(piece_coords)))
        random.shuffle(order)
        placements: list[tuple[int, int] | None] = [None] * len(piece_coords)
        occupied: set[tuple[int, int]] = set()

        first_idx = order[0]
        first = random.choice(options[first_idx])
        placements[first_idx] = first["offset"]  # type: ignore[assignment]
        occupied |= set(first["cells"])  # type: ignore[arg-type]

        ok = True
        for idx in order[1:]:
            candidates = [
                cand
                for cand in options[idx]
                if cand["cells"].isdisjoint(occupied)  # type: ignore[union-attr]
                and _touches(occupied, cand["cells"])  # type: ignore[arg-type]
            ]
            if not candidates:
                ok = False
                break
            chosen = random.choice(candidates)
            placements[idx] = chosen["offset"]  # type: ignore[assignment]
            occupied |= set(chosen["cells"])  # type: ignore[arg-type]

        if ok:
            return [offset for offset in placements if offset is not None]
    return None


def _search_initial_arrangement(
    piece_coords: list[tuple[tuple[int, int], ...]],
    h: int,
    w: int,
    forbidden: tuple[tuple[int, int], ...],
) -> list[tuple[int, int]] | None:
    options = [_all_piece_placements(coords, h, w) for coords in piece_coords]
    for _ in range(400):
        order = list(range(len(piece_coords)))
        random.shuffle(order)
        placements: list[tuple[int, int] | None] = [None] * len(piece_coords)
        occupied: set[tuple[int, int]] = set()
        blocked: set[tuple[int, int]] = set()
        ok = True
        for idx in order:
            candidates = [
                cand
                for cand in options[idx]
                if cand["cells"].isdisjoint(occupied)  # type: ignore[union-attr]
                and cand["cells"].isdisjoint(blocked)  # type: ignore[union-attr]
            ]
            if not candidates:
                ok = False
                break
            chosen = random.choice(candidates)
            placements[idx] = chosen["offset"]  # type: ignore[assignment]
            occupied |= set(chosen["cells"])  # type: ignore[arg-type]
            blocked |= set(chosen["halo"])  # type: ignore[arg-type]
        if not ok:
            continue
        solution = tuple(offset for offset in placements if offset is not None)
        if solution != forbidden:
            return list(solution)
    return None


def _render_frame_patch(
    h: int,
    w: int,
    colors: list[int],
    piece_coords: list[tuple[tuple[int, int], ...]],
    placements: list[tuple[int, int]],
) -> np.ndarray:
    patch = np.zeros((h + 2, w + 2), dtype=int)
    patch[0, :] = FRAME_COLOR
    patch[-1, :] = FRAME_COLOR
    patch[:, 0] = FRAME_COLOR
    patch[:, -1] = FRAME_COLOR
    for color, coords, (r0, c0) in zip(colors, piece_coords, placements):
        for dr, dc in coords:
            patch[1 + r0 + dr, 1 + c0 + dc] = int(color)
    return patch


def _piece_spec(coords: tuple[tuple[int, int], ...], color: int) -> dict[str, object]:
    r0, c0, r1, c1 = _bbox(coords)
    return {
        "color": int(color),
        "coords": coords,
        "h": int(r1 - r0 + 1),
        "w": int(c1 - c0 + 1),
        "area": int(len(coords)),
    }


def _make_case(colors: list[int]) -> dict[str, object] | None:
    count = len(colors)
    assert count in (2, 3)
    if count == 2:
        area_options = ((2, 3), (2, 4), (3, 4), (3, 5))
    else:
        area_options = ((2, 3, 4), (2, 3, 5), (2, 4, 5))
    areas = list(random.choice(area_options))

    piece_coords: list[tuple[tuple[int, int], ...]] = []
    for area in areas:
        coords = _random_polyomino(area, max_h=3, max_w=3)
        if coords is None:
            return None
        piece_coords.append(coords)

    final_places = _search_final_arrangement(piece_coords, 8, 8)
    if final_places is None:
        return None

    target_cells: set[tuple[int, int]] = set()
    for coords, (r0, c0) in zip(piece_coords, final_places):
        target_cells.update((r0 + dr, c0 + dc) for dr, dc in coords)
    rows = [r for r, _ in target_cells]
    cols = [c for _, c in target_cells]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    h = max_r - min_r + 1
    w = max_c - min_c + 1
    if not (4 <= h <= 7 and 4 <= w <= 7):
        return None
    if len(target_cells) >= h * w:
        return None

    final_places_t = tuple((r0 - min_r, c0 - min_c) for r0, c0 in final_places)
    initial_places = _search_initial_arrangement(piece_coords, h, w, final_places_t)
    if initial_places is None:
        return None

    target_mask = np.zeros((h, w), dtype=bool)
    for r, c in target_cells:
        target_mask[r - min_r, c - min_c] = True

    pieces = [_piece_spec(coords, color) for coords, color in zip(piece_coords, colors)]
    solutions = _solve_packing(pieces, target_mask, limit=2)
    if len(solutions) != 1:
        return None

    return {
        "h": int(h),
        "w": int(w),
        "colors": [int(c) for c in colors],
        "piece_coords": piece_coords,
        "initial": initial_places,
        "final": list(final_places_t),
        "target_mask": target_mask,
    }


def _place_boxes(
    sizes: list[tuple[int, int]],
    rows: int,
    cols: int,
    margin: int = 1,
    padding: int = 1,
) -> list[tuple[int, int]] | None:
    occupied = np.zeros((rows, cols), dtype=bool)
    order = list(range(len(sizes)))
    random.shuffle(order)
    order.sort(key=lambda idx: sizes[idx][0] * sizes[idx][1], reverse=True)
    placements: list[tuple[int, int] | None] = [None] * len(sizes)

    for idx in order:
        h, w = sizes[idx]
        candidates = []
        for r0 in range(margin, rows - margin - h + 1):
            for c0 in range(margin, cols - margin - w + 1):
                rr0 = max(0, r0 - padding)
                cc0 = max(0, c0 - padding)
                rr1 = min(rows - 1, r0 + h - 1 + padding)
                cc1 = min(cols - 1, c0 + w - 1 + padding)
                if not occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                    candidates.append((int(r0), int(c0)))
        if not candidates:
            return None
        placements[idx] = random.choice(candidates)
        r0, c0 = placements[idx]
        occupied[r0 : r0 + h, c0 : c0 + w] = True

    return [pos for pos in placements if pos is not None]


def _paste(dst: np.ndarray, top: int, left: int, patch: np.ndarray) -> None:
    h, w = patch.shape
    dst[top : top + h, left : left + w] = patch


def _build_scene() -> tuple[np.ndarray, np.ndarray] | None:
    frame_count = random.choice((1, 2))
    color_pool = list(PALETTE)
    random.shuffle(color_pool)

    cases = []
    cursor = 0
    for _ in range(frame_count):
        count = random.choice((2, 3))
        if cursor + count > len(color_pool):
            return None
        colors = color_pool[cursor : cursor + count]
        cursor += count
        case = _make_case(colors)
        if case is None:
            return None
        cases.append(case)

    objects: list[dict[str, object]] = []
    for idx, case in enumerate(cases):
        h = int(case["h"])
        w = int(case["w"])
        objects.append({"kind": "frame", "case": idx, "h": h + 2, "w": w + 2})
        for color in case["colors"]:  # type: ignore[index]
            objects.append({"kind": "template", "case": idx, "h": h, "w": w, "color": int(color)})

    sizes = [(int(obj["h"]), int(obj["w"])) for obj in objects]
    for _ in range(80):
        rows = random.randint(18, 30)
        cols = random.randint(18, 30)
        placements = _place_boxes(sizes, rows, cols)
        if placements is None:
            continue

        inp = np.zeros((rows, cols), dtype=int)
        out = np.zeros((rows, cols), dtype=int)
        for obj, (r0, c0) in zip(objects, placements):
            case = cases[int(obj["case"])]
            if obj["kind"] == "frame":
                frame_in = _render_frame_patch(
                    int(case["h"]),
                    int(case["w"]),
                    case["colors"],  # type: ignore[arg-type]
                    case["piece_coords"],  # type: ignore[arg-type]
                    case["initial"],  # type: ignore[arg-type]
                )
                frame_out = _render_frame_patch(
                    int(case["h"]),
                    int(case["w"]),
                    case["colors"],  # type: ignore[arg-type]
                    case["piece_coords"],  # type: ignore[arg-type]
                    case["final"],  # type: ignore[arg-type]
                )
                _paste(inp, r0, c0, frame_in)
                _paste(out, r0, c0, frame_out)
            else:
                patch = np.where(case["target_mask"], int(obj["color"]), BG)  # type: ignore[index]
                _paste(inp, r0, c0, patch)
                _paste(out, r0, c0, patch)

        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, out):
            return inp, out
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        candidate = _build_scene()
        if candidate is not None:
            return candidate[0]
    raise RuntimeError("failed to generate a valid puzzle instance")
