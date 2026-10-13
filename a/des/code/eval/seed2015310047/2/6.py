from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
GENERATE_RETRIES = 600
SHAPE_RETRIES = 200
TEETH_RETRIES = 200


def _most_common_color(values: np.ndarray) -> int:
    counts = Counter(int(v) for v in np.asarray(values).ravel())
    assert counts, "expected at least one value"
    return int(counts.most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(int(r), int(c))])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIRS:
                    nr, nc = int(rr + dr), int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            parts.append(np.asarray(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _tighten(patch: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.argwhere(mask)
    assert pts.size > 0, "cannot tighten an empty mask"
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return (
        np.array(patch[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1], dtype=int),
        np.array(mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1], dtype=bool),
    )


def _rotate_patch_mask(patch: np.ndarray, mask: np.ndarray, rot_k: int) -> tuple[np.ndarray, np.ndarray]:
    return _tighten(np.rot90(patch, int(rot_k)), np.rot90(mask, int(rot_k)))


def _build_component_info(grid: np.ndarray, coords: np.ndarray) -> dict:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.array(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return {
        "coords": coords,
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "patch": patch,
        "mask": mask,
        "area": int(coords.shape[0]),
    }


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    comps = [_build_component_info(g, coords) for coords in _components(g != int(bg))]
    assert comps, "expected at least one foreground component"

    frame_idx = max(range(len(comps)), key=lambda i: int(comps[i]["area"]))
    frame = comps[frame_idx]
    fr0, fc0, fr1, fc1 = frame["bbox"]
    frame_color = _most_common_color(g[frame["coords"][:, 0], frame["coords"][:, 1]])

    sub_bg = g[fr0 : fr1 + 1, fc0 : fc1 + 1] == int(bg)
    holes = []
    for coords in _components(sub_bg):
        if (
            np.any(coords[:, 0] == 0)
            or np.any(coords[:, 1] == 0)
            or np.any(coords[:, 0] == sub_bg.shape[0] - 1)
            or np.any(coords[:, 1] == sub_bg.shape[1] - 1)
        ):
            continue
        hr0, hc0, hr1, hc1 = _bbox(coords)
        mask = np.zeros((hr1 - hr0 + 1, hc1 - hc0 + 1), dtype=bool)
        mask[coords[:, 0] - hr0, coords[:, 1] - hc0] = True
        abs_r0 = int(fr0 + hr0)
        abs_c0 = int(fc0 + hc0)
        boundary_teeth: dict[tuple[int, int], int] = {}
        for lr, lc in np.argwhere(mask):
            lr = int(lr)
            lc = int(lc)
            abs_r = int(abs_r0 + lr)
            abs_c = int(abs_c0 + lc)
            for dr, dc in DIRS:
                nr = lr + dr
                nc = lc + dc
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and bool(mask[nr, nc]):
                    continue
                rr = abs_r + dr
                cc = abs_c + dc
                if not (fr0 <= rr <= fr1 and fc0 <= cc <= fc1):
                    continue
                color = int(g[rr, cc])
                if color == int(bg) or color == int(frame_color):
                    continue
                boundary_teeth[(int(rr), int(cc))] = int(color)
        holes.append(
            {
                "bbox": (abs_r0, abs_c0, int(abs_r0 + mask.shape[0] - 1), int(abs_c0 + mask.shape[1] - 1)),
                "mask": mask,
                "boundary_teeth": boundary_teeth,
            }
        )
    holes.sort(key=lambda hole: (int(hole["bbox"][0]), int(hole["bbox"][1])))

    pieces = []
    for idx, comp in enumerate(comps):
        if idx == frame_idx:
            continue
        pieces.append(
            {
                **comp,
                "body": _most_common_color(comp["patch"][comp["mask"]]),
            }
        )
    pieces.sort(key=lambda piece: (int(piece["bbox"][0]), int(piece["bbox"][1])))

    return {
        "bg": int(bg),
        "frame_color": int(frame_color),
        "holes": holes,
        "pieces": pieces,
    }


def _match_piece_to_hole(piece: dict, hole: dict, rot_k: int) -> dict | None:
    patch, mask = _rotate_patch_mask(piece["patch"], piece["mask"], int(rot_k))
    if patch.shape != hole["mask"].shape or not bool(np.array_equal(mask, hole["mask"])):
        return None

    body = int(piece["body"])
    r0, c0, _r1, _c1 = hole["bbox"]
    boundary = hole["boundary_teeth"]

    tooth_pixels = np.argwhere(mask & (patch != int(body)))
    for lr, lc in tooth_pixels:
        lr = int(lr)
        lc = int(lc)
        color = int(patch[lr, lc])
        abs_r = int(r0 + lr)
        abs_c = int(c0 + lc)
        ok = False
        for dr, dc in DIRS:
            if boundary.get((int(abs_r + dr), int(abs_c + dc))) == int(color):
                ok = True
                break
        if not ok:
            return None

    for (br, bc), color in boundary.items():
        ok = False
        for dr, dc in DIRS:
            lr = int(br + dr - r0)
            lc = int(bc + dc - c0)
            if 0 <= lr < mask.shape[0] and 0 <= lc < mask.shape[1] and bool(mask[lr, lc]) and int(patch[lr, lc]) == int(color):
                ok = True
                break
        if not ok:
            return None

    return {"piece": None, "patch": patch, "mask": mask}


def _build_candidates(scene: dict) -> list[list[dict]]:
    candidates: list[list[dict]] = [[] for _ in scene["holes"]]
    for hole_idx, hole in enumerate(scene["holes"]):
        seen = set()
        for piece_idx, piece in enumerate(scene["pieces"]):
            for rot_k in range(4):
                match = _match_piece_to_hole(piece, hole, rot_k)
                if match is None:
                    continue
                key = (int(piece_idx), match["patch"].tobytes())
                if key in seen:
                    continue
                seen.add(key)
                candidates[hole_idx].append(
                    {
                        "piece": int(piece_idx),
                        "patch": match["patch"],
                        "mask": match["mask"],
                    }
                )
        candidates[hole_idx].sort(key=lambda cand: int(cand["piece"]))
    return candidates


def _solve_assignments(candidates: list[list[dict]], max_solutions: int) -> list[list[dict]]:
    if not candidates:
        return [[]]
    if any(not opts for opts in candidates):
        return []

    order = sorted(range(len(candidates)), key=lambda hole_idx: len(candidates[hole_idx]))
    used: set[int] = set()
    current: list[dict | None] = [None] * len(candidates)
    solutions: list[list[dict]] = []

    def dfs(depth: int) -> None:
        if len(solutions) >= int(max_solutions):
            return
        if depth == len(order):
            solutions.append([cand for cand in current if cand is not None])
            return

        hole_idx = int(order[depth])
        for cand in candidates[hole_idx]:
            piece_idx = int(cand["piece"])
            if piece_idx in used:
                continue
            used.add(piece_idx)
            current[hole_idx] = cand
            dfs(depth + 1)
            current[hole_idx] = None
            used.remove(piece_idx)

    dfs(0)
    return solutions


def _render_solution(grid: np.ndarray, scene: dict, assignment: list[dict]) -> np.ndarray:
    out = np.array(grid, dtype=int)
    bg = int(scene["bg"])

    for piece in scene["pieces"]:
        coords = piece["coords"]
        out[coords[:, 0], coords[:, 1]] = int(bg)

    for hole_idx, hole in enumerate(scene["holes"]):
        cand = assignment[hole_idx]
        patch = cand["patch"]
        mask = cand["mask"]
        r0, c0, _r1, _c1 = hole["bbox"]
        sub = out[int(r0) : int(r0) + patch.shape[0], int(c0) : int(c0) + patch.shape[1]]
        sub[mask] = patch[mask]

    return out


def _solve_grid(grid: np.ndarray, max_solutions: int) -> tuple[dict, list[list[dict]]]:
    scene = _parse_scene(grid)
    if len(scene["holes"]) != len(scene["pieces"]):
        return scene, []
    candidates = _build_candidates(scene)
    solutions = _solve_assignments(candidates, max_solutions=max_solutions)
    return scene, solutions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each hole in the large component with the matching external piece.

    Shape equality narrows the candidate hole, and adjacency of the colored edge
    teeth fixes the correct rotation when the shape is rotationally ambiguous.
    """
    g = np.asarray(grid, dtype=int)
    scene, solutions = _solve_grid(g, max_solutions=1)
    assert solutions, "expected at least one valid assignment"
    return _render_solution(g, scene, solutions[0])


def _cell_on_border(mask: np.ndarray, r: int, c: int) -> bool:
    h, w = mask.shape
    for dr, dc in DIRS:
        nr = int(r + dr)
        nc = int(c + dc)
        if not (0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc])):
            return True
    return False


def _is_connected(mask: np.ndarray) -> bool:
    return bool(mask.any()) and len(_components(mask)) == 1


def _eligible_teeth(mask: np.ndarray) -> list[tuple[int, int, int]]:
    h, w = mask.shape
    result = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        exposed = []
        for direction, (dr, dc) in enumerate(DIRS):
            nr = r + dr
            nc = c + dc
            if not (0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc])):
                exposed.append(int(direction))
        if len(exposed) == 1:
            result.append((int(r), int(c), int(exposed[0])))
    return result


def _build_piece_patch(mask: np.ndarray, body_color: int, teeth: list[tuple[int, int, int, int]]) -> np.ndarray:
    patch = np.full(mask.shape, int(body_color), dtype=int)
    for r, c, _direction, color in teeth:
        patch[int(r), int(c)] = int(color)
    return patch


def _boundary_teeth_from_specs(
    top: int, left: int, teeth: list[tuple[int, int, int, int]]
) -> dict[tuple[int, int], int]:
    boundary = {}
    for r, c, direction, color in teeth:
        dr, dc = DIRS[int(direction)]
        boundary[(int(top + r + dr), int(left + c + dc))] = int(color)
    return boundary


def _orientation_match_count(mask: np.ndarray, teeth: list[tuple[int, int, int, int]]) -> int:
    demo_body = 1
    piece = {
        "patch": _build_piece_patch(mask, demo_body, teeth),
        "mask": mask,
        "body": int(demo_body),
    }
    hole = {
        "bbox": (0, 0, int(mask.shape[0] - 1), int(mask.shape[1] - 1)),
        "mask": mask,
        "boundary_teeth": _boundary_teeth_from_specs(0, 0, teeth),
    }
    count = 0
    seen = set()
    for rot_k in range(4):
        match = _match_piece_to_hole(piece, hole, rot_k)
        if match is None:
            continue
        key = match["patch"].tobytes()
        if key in seen:
            continue
        seen.add(key)
        count += 1
    return int(count)


def _random_shape() -> np.ndarray | None:
    for _ in range(SHAPE_RETRIES):
        h = random.randint(2, 5)
        w = random.randint(2, 5)
        if h == 2 and w == 2:
            continue

        mask = np.ones((h, w), dtype=bool)
        max_remove = min(4, int(mask.sum()) - 4)
        if random.random() < 0.45 or max_remove <= 0:
            remove_count = 0
        else:
            remove_count = random.randint(1, max_remove)

        for _step in range(remove_count):
            border_cells = [(r, c) for r in range(h) for c in range(w) if bool(mask[r, c]) and _cell_on_border(mask, r, c)]
            random.shuffle(border_cells)
            changed = False
            for r, c in border_cells:
                test = mask.copy()
                test[int(r), int(c)] = False
                if int(test.sum()) < 4 or not _is_connected(test):
                    continue
                _dummy, tight_mask = _tighten(test.astype(int), test)
                if tight_mask.shape[0] == 1 or tight_mask.shape[1] == 1:
                    continue
                mask = tight_mask
                h, w = mask.shape
                changed = True
                break
            if not changed:
                break

        _dummy, mask = _tighten(mask.astype(int), mask)
        if int(mask.sum()) < 4 or mask.shape[0] == 1 or mask.shape[1] == 1:
            continue
        if len(_eligible_teeth(mask)) < 2:
            continue
        return mask
    return None


def _random_teeth(mask: np.ndarray, tooth_palette: list[int]) -> list[tuple[int, int, int, int]] | None:
    eligible = _eligible_teeth(mask)
    if len(eligible) < 2:
        return None

    for _ in range(TEETH_RETRIES):
        tooth_count = random.randint(2, min(4, len(eligible)))
        chosen = random.sample(eligible, tooth_count)
        colors = [int(random.choice(tooth_palette)) for _ in range(tooth_count)]
        if len(tooth_palette) >= 2 and tooth_count >= 2 and len(set(colors)) == 1:
            colors[0] = int(tooth_palette[0])
            colors[-1] = int(tooth_palette[1])
        teeth = [(int(r), int(c), int(direction), int(color)) for (r, c, direction), color in zip(chosen, colors)]
        if _orientation_match_count(mask, teeth) == 1:
            return teeth
    return None


def _place_rectangles(
    outer_h: int, outer_w: int, sizes: list[tuple[int, int]], reserved: np.ndarray | None = None
) -> dict[int, tuple[int, int]] | None:
    occupied = np.array(reserved, dtype=bool, copy=True) if reserved is not None else np.zeros((outer_h, outer_w), dtype=bool)
    placed: dict[int, tuple[int, int]] = {}
    order = sorted(range(len(sizes)), key=lambda idx: sizes[idx][0] * sizes[idx][1], reverse=True)

    for idx in order:
        h, w = sizes[idx]
        candidates = []
        for top in range(1, outer_h - h):
            for left in range(1, outer_w - w):
                if bool(occupied[top - 1 : top + h + 1, left - 1 : left + w + 1].any()):
                    continue
                candidates.append((int(top), int(left)))
        if not candidates:
            return None
        top, left = random.choice(candidates)
        occupied[top - 1 : top + h + 1, left - 1 : left + w + 1] = True
        placed[int(idx)] = (int(top), int(left))
    return placed


def _gen_attempt() -> np.ndarray | None:
    hole_count = random.randint(2, 4)
    palette = random.sample(list(range(10)), 6)
    bg, frame_color, piece_color = [int(v) for v in palette[:3]]
    tooth_palette = [int(v) for v in palette[3:]]

    hole_defs = []
    for _ in range(hole_count):
        mask = _random_shape()
        if mask is None:
            return None
        teeth = _random_teeth(mask, tooth_palette)
        if teeth is None:
            return None
        hole_defs.append({"mask": mask, "teeth": teeth})

    min_rect_h = max(10, max(int(hole["mask"].shape[0]) + 2 for hole in hole_defs))
    min_rect_w = max(10, max(int(hole["mask"].shape[1]) + 2 for hole in hole_defs))
    rect_h = random.randint(min_rect_h, 16)
    rect_w = random.randint(min_rect_w, 16)

    hole_boxes = [(int(hole["mask"].shape[0]), int(hole["mask"].shape[1])) for hole in hole_defs]
    hole_positions = _place_rectangles(rect_h, rect_w, hole_boxes)
    if hole_positions is None:
        return None

    grid_h = rect_h + random.randint(8, 12)
    grid_w = rect_w + random.randint(8, 12)
    rect_top = random.randint(2, grid_h - rect_h - 2)
    rect_left = random.randint(2, grid_w - rect_w - 2)

    canonical_patches = []
    piece_variants = []
    for hole in hole_defs:
        patch = _build_piece_patch(hole["mask"], piece_color, hole["teeth"])
        canonical_patches.append(patch)
        rot_k = random.randrange(4)
        rot_patch, rot_mask = _rotate_patch_mask(patch, hole["mask"], rot_k)
        piece_variants.append({"patch": rot_patch, "mask": rot_mask})

    piece_reserved = np.zeros((grid_h, grid_w), dtype=bool)
    piece_reserved[
        max(0, rect_top - 1) : min(grid_h, rect_top + rect_h + 1),
        max(0, rect_left - 1) : min(grid_w, rect_left + rect_w + 1),
    ] = True
    reserved_sizes = [(int(piece["mask"].shape[0]), int(piece["mask"].shape[1])) for piece in piece_variants]
    piece_positions = _place_rectangles(grid_h, grid_w, reserved_sizes, reserved=piece_reserved)
    if piece_positions is None:
        return None

    grid = np.full((grid_h, grid_w), int(bg), dtype=int)
    grid[rect_top : rect_top + rect_h, rect_left : rect_left + rect_w] = int(frame_color)

    expected = np.array(grid, dtype=int)

    for hole_idx, hole in enumerate(hole_defs):
        top, left = hole_positions[hole_idx]
        abs_top = int(rect_top + top)
        abs_left = int(rect_left + left)
        mask = hole["mask"]
        grid[abs_top : abs_top + mask.shape[0], abs_left : abs_left + mask.shape[1]][mask] = int(bg)
        expected[abs_top : abs_top + mask.shape[0], abs_left : abs_left + mask.shape[1]][mask] = int(bg)
        for r, c, direction, color in hole["teeth"]:
            dr, dc = DIRS[int(direction)]
            br = int(abs_top + r + dr)
            bc = int(abs_left + c + dc)
            grid[br, bc] = int(color)
            expected[br, bc] = int(color)

    for hole_idx, hole in enumerate(hole_defs):
        top, left = hole_positions[hole_idx]
        abs_top = int(rect_top + top)
        abs_left = int(rect_left + left)
        patch = canonical_patches[hole_idx]
        expected[abs_top : abs_top + patch.shape[0], abs_left : abs_left + patch.shape[1]][hole["mask"]] = patch[hole["mask"]]

    for piece_idx, piece in enumerate(piece_variants):
        top, left = piece_positions[piece_idx]
        grid[top : top + piece["patch"].shape[0], left : left + piece["patch"].shape[1]][piece["mask"]] = piece["patch"][piece["mask"]]

    scene, solutions = _solve_grid(grid, max_solutions=2)
    if len(solutions) != 1:
        return None
    solved = _render_solution(grid, scene, solutions[0])
    if not bool(np.array_equal(solved, expected)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
