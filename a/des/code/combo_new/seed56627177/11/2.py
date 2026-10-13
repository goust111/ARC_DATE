import random
from collections import Counter, defaultdict

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
BG = 0
MAX_RETRIES = 400


def _background_color(grid: np.ndarray) -> int:
    return int(Counter(np.asarray(grid, dtype=int).ravel().tolist()).most_common(1)[0][0])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    comp_id = 0

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            comps.append(
                {
                    "id": comp_id,
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "height": r1 - r0 + 1,
                    "width": c1 - c0 + 1,
                }
            )
            comp_id += 1

    return comps


def _frame_info(grid: np.ndarray, comp: dict, bg: int) -> dict | None:
    g = np.asarray(grid, dtype=int)
    color = int(comp["color"])
    r0, c0, r1, c1 = comp["bbox"]
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return None

    if not np.all(g[r0 : r1 + 1, c0] == color):
        return None
    if not np.all(g[r0 : r1 + 1, c1] == color):
        return None
    if not np.all(g[r0, c0 : c1 + 1] == color):
        return None
    if not np.all(g[r1, c0 : c1 + 1] == color):
        return None

    interior = g[r0 + 1 : r1, c0 + 1 : c1]
    if interior.size == 0:
        return None
    if np.any((interior != bg) & (interior != color)):
        return None

    gap_rows: list[int] = []
    rightmost_rel = -1
    filled_rows = 0
    for rr in range(interior.shape[0]):
        nz = np.where(interior[rr] == color)[0]
        if len(nz) == 0:
            gap_rows.append(r0 + 1 + rr)
            continue
        filled_rows += 1
        rightmost_rel = max(rightmost_rel, int(nz[-1]))

    if filled_rows == 0 or not gap_rows or rightmost_rel < 0:
        return None

    return {
        "id": int(comp["id"]),
        "color": color,
        "bbox": comp["bbox"],
        "gap_rows": gap_rows,
        "insert_col": c0 + 1 + rightmost_rel + 1,
        "inner_right": c1 - 1,
    }


def _fragment_mask(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in comp["coords"]:
        mask[r - r0, c - c0] = True
    return mask


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    out = g.copy()
    comps = _find_components(g, bg)
    by_color: dict[int, list[dict]] = defaultdict(list)
    for comp in comps:
        by_color[int(comp["color"])].append(comp)

    frames = []
    for comp in comps:
        info = _frame_info(g, comp, bg)
        if info is not None:
            frames.append(info)

    for frame in frames:
        color = int(frame["color"])
        gap_count = len(frame["gap_rows"])
        candidates = [
            comp
            for comp in by_color[color]
            if int(comp["id"]) != int(frame["id"])
            and comp["height"] == gap_count
            and _frame_info(g, comp, bg) is None
        ]
        if not candidates:
            continue

        frag = min(candidates, key=lambda comp: (len(comp["coords"]), comp["bbox"][0], comp["bbox"][1]))
        mask = _fragment_mask(frag)
        base_col = int(frame["insert_col"])
        if base_col + mask.shape[1] - 1 > int(frame["inner_right"]):
            continue

        for r, c in frag["coords"]:
            out[r, c] = bg
        for idx, target_r in enumerate(frame["gap_rows"]):
            rel_cols = np.where(mask[idx])[0]
            out[target_r, base_col + rel_cols] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reinsert each exterior fragment into the matching hollow frame by expanding it row-by-row into that frame's blank interior rows.

    Input structure:
    - Background is black.
    - Several colors form 1-pixel-thick rectangular frames whose interiors contain some same-color horizontal stripe rows and some fully black gap rows.
    - Outside the frames, each frame color also appears as one separate 4-connected fragment whose height matches that frame's number of gap rows.

    Rule:
    - Detect each monochrome connected component whose bounding-box perimeter is a hollow rectangle; this is the frame object for that color.
    - For that frame, count the interior rows that are completely black and find the same-color external fragment with exactly that many rows in its bounding box.
    - Clear the external fragment, then place its first row into the topmost gap row, its second row into the next gap row, and so on, keeping the fragment's horizontal offsets inside its own bbox.
    - The fragment's bbox left edge is aligned immediately to the right of the rightmost already-filled interior stripe pixel of the frame.

    Key insight:
    - The decisive match is height: a fragment belongs to the frame whose number of blank interior rows equals the fragment's height, and the fragment is unfolded row-by-row rather than translated as one rigid block.
    """
    return _solve(grid)


def _sample_fragment_mask(height: int, width: int, rng: random.Random) -> np.ndarray:
    intervals = [(s, e) for s in range(width) for e in range(s, width)]
    for _ in range(200):
        rows: list[tuple[int, int]] = [rng.choice(intervals)]
        for _row in range(1, height):
            prev_s, prev_e = rows[-1]
            choices = [(s, e) for s, e in intervals if not (e < prev_s or s > prev_e)]
            rows.append(rng.choice(choices))
        if width > 1:
            if min(s for s, _ in rows) != 0 or max(e for _, e in rows) != width - 1:
                continue
        mask = np.zeros((height, width), dtype=bool)
        for rr, (s, e) in enumerate(rows):
            mask[rr, s : e + 1] = True
        return mask

    mask = np.zeros((height, width), dtype=bool)
    mask[:, :] = True
    return mask


def _frame_patch(interior_h: int, interior_w: int, color: int, gap_rows: set[int], prefix_len: int) -> np.ndarray:
    patch = np.zeros((interior_h + 2, interior_w + 2), dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    for rr in range(interior_h):
        if rr in gap_rows:
            continue
        patch[rr + 1, 1 : 1 + prefix_len] = color
    return patch


def _can_place_rect(used: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> bool:
    h, w = used.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    return not bool(used[r0:r1, c0:c1].any())


def _mark_rect(used: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> None:
    h, w = used.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    used[r0:r1, c0:c1] = True


def _place_patch(grid: np.ndarray, top: int, left: int, patch: np.ndarray) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    h, w = patch.shape
    for r in range(h):
        for c in range(w):
            value = int(patch[r, c])
            if value == BG:
                continue
            grid[top + r, left + c] = value
            coords.append((top + r, left + c))
    return coords


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(16, 30)
    width = rng.randint(16, 30)
    grid = np.zeros((height, width), dtype=int)
    used = np.zeros((height, width), dtype=bool)

    frame_count = rng.randint(1, 3)
    colors = rng.sample(list(range(1, 10)), frame_count)
    specs: list[dict] = []

    for color in colors:
        placed = False
        for _ in range(120):
            interior_h = rng.randint(3, 7)
            interior_w = rng.randint(5, 10)
            if interior_h + 2 > height or interior_w + 2 > width:
                continue

            gap_count = rng.randint(1, interior_h - 1)
            gap_rows = set(rng.sample(list(range(interior_h)), gap_count))
            max_frag_w = min(3, gap_count, interior_w - 1)
            if max_frag_w < 1:
                continue
            frag_w = rng.randint(1, max_frag_w)
            prefix_len = rng.randint(1, interior_w - frag_w)

            frame_patch = _frame_patch(interior_h, interior_w, color, gap_rows, prefix_len)
            frame_h, frame_w = frame_patch.shape
            frame_pos = None
            for _ in range(120):
                top = rng.randint(0, height - frame_h)
                left = rng.randint(0, width - frame_w)
                if _can_place_rect(used, top, left, frame_h, frame_w):
                    frame_pos = (top, left)
                    break
            if frame_pos is None:
                continue

            frag_mask = _sample_fragment_mask(gap_count, frag_w, rng)
            frag_patch = np.zeros((gap_count, frag_w), dtype=int)
            frag_patch[frag_mask] = color
            trial_used = used.copy()
            _mark_rect(trial_used, frame_pos[0], frame_pos[1], frame_h, frame_w)
            frag_pos = None
            for _ in range(160):
                top = rng.randint(0, height - gap_count)
                left = rng.randint(0, width - frag_w)
                if _can_place_rect(trial_used, top, left, gap_count, frag_w):
                    frag_pos = (top, left)
                    break
            if frag_pos is None:
                continue

            frame_coords = _place_patch(grid, frame_pos[0], frame_pos[1], frame_patch)
            _mark_rect(used, frame_pos[0], frame_pos[1], frame_h, frame_w)
            frag_coords = _place_patch(grid, frag_pos[0], frag_pos[1], frag_patch)
            _mark_rect(used, frag_pos[0], frag_pos[1], gap_count, frag_w)

            specs.append(
                {
                    "color": color,
                    "frame_top": frame_pos[0],
                    "frame_left": frame_pos[1],
                    "interior_h": interior_h,
                    "gap_rows": sorted(gap_rows),
                    "prefix_len": prefix_len,
                    "frag_mask": frag_mask.copy(),
                    "frag_coords": frag_coords,
                    "frame_coords": frame_coords,
                }
            )
            placed = True
            break
        if not placed:
            return None

    expected = grid.copy()
    for spec in specs:
        for r, c in spec["frag_coords"]:
            expected[r, c] = BG
        base_col = int(spec["frame_left"] + 1 + spec["prefix_len"])
        for idx, rel_r in enumerate(spec["gap_rows"]):
            target_r = int(spec["frame_top"] + 1 + rel_r)
            rel_cols = np.where(spec["frag_mask"][idx])[0]
            expected[target_r, base_col + rel_cols] = int(spec["color"])

    if _background_color(grid) != BG:
        return None
    if np.array_equal(expected, grid):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {MAX_RETRIES} attempts")
