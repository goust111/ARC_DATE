from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]

GEN_GRID_ROWS = (22, 28)
GEN_GRID_COLS = (22, 28)
GEN_SEED_H = (3, 6)
GEN_SEED_W = (3, 6)
GEN_WINDOW_COUNT = [2, 3, 4]
GEN_WINDOW_WEIGHTS = [2, 5, 3]
GEN_INNER_H = (2, 7)
GEN_INNER_W = (2, 7)
GEN_MAX_ATTEMPTS = 300


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(comp)
    return components


def _is_seed_patch(patch: np.ndarray) -> bool:
    colors = set(map(int, patch.flatten()))
    return BG not in colors and len(colors) >= 2


def _is_frame_patch(patch: np.ndarray) -> bool:
    h, w = patch.shape
    if h < 3 or w < 3:
        return False
    colors = {int(v) for v in np.unique(patch) if int(v) != BG}
    if len(colors) != 1:
        return False
    color = next(iter(colors))
    border = np.zeros_like(patch, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if not np.all(patch[border] == color):
        return False
    return bool(np.all(patch[~border] == BG))


def _parse_scene(grid: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray, list[tuple[int, int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    comps = _components4(g != BG)
    assert comps, "expected at least one foreground component"

    seed_bbox: tuple[int, int, int, int] | None = None
    seed_patch: np.ndarray | None = None
    frames: list[tuple[int, int, int, int]] = []

    for comp in comps:
        bbox = _bbox(comp)
        patch = _crop(g, bbox)
        if len(comp) == patch.size and _is_seed_patch(patch):
            assert seed_bbox is None, "expected exactly one multicolor solid seed rectangle"
            seed_bbox = bbox
            seed_patch = patch
            continue
        if _is_frame_patch(patch):
            frames.append(bbox)
            continue
        raise AssertionError("unexpected foreground component")

    assert seed_bbox is not None and seed_patch is not None
    assert frames, "expected at least one hollow window frame"
    return seed_bbox, seed_patch, frames


def _sample_reflected_tile(seed: np.ndarray, rel_r: int, rel_c: int) -> int:
    h, w = seed.shape
    q_r, rem_r = divmod(int(rel_r), h)
    q_c, rem_c = divmod(int(rel_c), w)
    if q_r % 2:
        rem_r = h - 1 - rem_r
    if q_c % 2:
        rem_c = w - 1 - rem_c
    return int(seed[rem_r, rem_c])


def _render_window(seed: np.ndarray, rel_r0: int, rel_c0: int, h: int, w: int, *, mirrored: bool) -> np.ndarray:
    out = np.zeros((h, w), dtype=int)
    seed_h, seed_w = seed.shape
    for r in range(h):
        for c in range(w):
            rr = rel_r0 + r
            cc = rel_c0 + c
            if mirrored:
                out[r, c] = _sample_reflected_tile(seed, rr, cc)
            else:
                out[r, c] = int(seed[rr % seed_h, cc % seed_w])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Render the seed rectangle as a mirror-flipped wallpaper, but only inside the hollow window frames.

    Input structure:
    - A black background contains one multicolor solid rectangular seed block.
    - It also contains one or more single-color hollow rectangular window frames whose interiors are empty.
    - No wallpaper is drawn explicitly outside the windows.

    Rule:
    - Keep the seed block and every window frame unchanged.
    - For each interior cell of each window, measure its row/column offset from the seed's top-left corner.
    - Divide those offsets by the seed height/width; the quotient parity decides whether that tile is flipped vertically and/or horizontally, and the remainder picks the color inside the flipped seed tile.

    Key insight:
    - The hidden pattern is not plain translational tiling: every step across a tile boundary reflects the motif, so the quotient parity matters as much as the in-tile remainder.
    """
    g = np.asarray(grid, dtype=int)
    seed_bbox, seed_patch, frames = _parse_scene(g)
    seed_r0, seed_c0, _, _ = seed_bbox

    out = g.copy()
    for r0, c0, r1, c1 in frames:
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                rel_r = r - seed_r0
                rel_c = c - seed_c0
                out[r, c] = _sample_reflected_tile(seed_patch, rel_r, rel_c)
    return out


def _has_smaller_period(tile: np.ndarray) -> bool:
    h, w = tile.shape
    for ph in range(1, h + 1):
        for pw in range(1, w + 1):
            if ph == h and pw == w:
                continue
            ok = True
            for r in range(h):
                for c in range(w):
                    if int(tile[r, c]) != int(tile[r % ph, c % pw]):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return True
    return False


def _all_reflections_distinct(tile: np.ndarray) -> bool:
    variants = [
        tile,
        np.fliplr(tile),
        np.flipud(tile),
        np.flipud(np.fliplr(tile)),
    ]
    return len({v.tobytes() for v in variants}) == 4


def _make_seed_tile() -> np.ndarray | None:
    h = random.randint(*GEN_SEED_H)
    w = random.randint(*GEN_SEED_W)
    max_colors = min(5, h * w)
    n_colors = random.randint(3, max_colors)
    palette = random.sample(COLOR_POOL, n_colors)
    tile = np.array([[random.choice(palette) for _ in range(w)] for _ in range(h)], dtype=int)
    missing = [color for color in palette if color not in set(map(int, tile.flatten()))]
    coords = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(coords)
    for color, (r, c) in zip(missing, coords):
        tile[r, c] = color
    if _has_smaller_period(tile):
        return None
    if not _all_reflections_distinct(tile):
        return None
    return tile


def _separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    return a[2] + gap < b[0] or b[2] + gap < a[0] or a[3] + gap < b[1] or b[3] + gap < a[1]


def _window_features(
    seed: np.ndarray,
    seed_top: int,
    seed_left: int,
    top: int,
    left: int,
    inner_h: int,
    inner_w: int,
) -> dict:
    rel_r0 = top + 1 - seed_top
    rel_c0 = left + 1 - seed_left
    seed_h, seed_w = seed.shape
    row_qs = {divmod(rel_r0 + dr, seed_h)[0] for dr in range(inner_h)}
    col_qs = {divmod(rel_c0 + dc, seed_w)[0] for dc in range(inner_w)}
    combos = {
        (divmod(rel_r0 + dr, seed_h)[0] % 2, divmod(rel_c0 + dc, seed_w)[0] % 2)
        for dr in range(inner_h)
        for dc in range(inner_w)
    }
    mirrored = _render_window(seed, rel_r0, rel_c0, inner_h, inner_w, mirrored=True)
    plain = _render_window(seed, rel_r0, rel_c0, inner_h, inner_w, mirrored=False)
    return {
        "bbox": (top, left, top + inner_h + 1, left + inner_w + 1),
        "inner_h": inner_h,
        "inner_w": inner_w,
        "cross_r": len(row_qs) > 1,
        "cross_c": len(col_qs) > 1,
        "negative": any(q < 0 for q in row_qs) or any(q < 0 for q in col_qs),
        "combos": combos,
        "diff_from_plain": not np.array_equal(mirrored, plain),
    }


def _enumerate_window_candidates(
    shape: tuple[int, int],
    seed: np.ndarray,
    seed_bbox: tuple[int, int, int, int],
) -> list[dict]:
    h, w = shape
    seed_top, seed_left, _, _ = seed_bbox
    candidates: list[dict] = []
    for inner_h in range(GEN_INNER_H[0], GEN_INNER_H[1] + 1):
        outer_h = inner_h + 2
        if outer_h > h:
            continue
        for inner_w in range(GEN_INNER_W[0], GEN_INNER_W[1] + 1):
            outer_w = inner_w + 2
            if outer_w > w:
                continue
            for top in range(h - outer_h + 1):
                for left in range(w - outer_w + 1):
                    bbox = (top, left, top + outer_h - 1, left + outer_w - 1)
                    if not _separated(bbox, seed_bbox, gap=1):
                        continue
                    feats = _window_features(seed, seed_top, seed_left, top, left, inner_h, inner_w)
                    if not (feats["cross_r"] or feats["cross_c"] or feats["negative"] or feats["diff_from_plain"]):
                        continue
                    candidates.append(feats)
    random.shuffle(candidates)
    return candidates


def _coverage_ok(selected: list[dict], require_negative: bool) -> bool:
    combos = set()
    for item in selected:
        combos |= item["combos"]
    return (
        any(item["cross_r"] for item in selected)
        and any(item["cross_c"] for item in selected)
        and any(item["cross_r"] and item["cross_c"] for item in selected)
        and any(item["diff_from_plain"] for item in selected)
        and len(combos) == 4
        and ((not require_negative) or any(item["negative"] for item in selected))
    )


def _choose_windows(candidates: list[dict], seed_bbox: tuple[int, int, int, int], count: int, require_negative: bool) -> list[dict] | None:
    for _ in range(400):
        selected: list[dict] = []
        used = [seed_bbox]
        pool = candidates[:]
        while len(selected) < count:
            combos = set()
            for item in selected:
                combos |= item["combos"]
            scored: list[tuple[float, dict]] = []
            have_r = any(item["cross_r"] for item in selected)
            have_c = any(item["cross_c"] for item in selected)
            have_both = any(item["cross_r"] and item["cross_c"] for item in selected)
            have_neg = any(item["negative"] for item in selected)
            have_diff = any(item["diff_from_plain"] for item in selected)
            for cand in pool:
                if not all(_separated(cand["bbox"], bbox, gap=1) for bbox in used):
                    continue
                score = 0.0
                if cand["cross_r"] and not have_r:
                    score += 5.0
                if cand["cross_c"] and not have_c:
                    score += 5.0
                if cand["cross_r"] and cand["cross_c"] and not have_both:
                    score += 4.0
                if cand["diff_from_plain"] and not have_diff:
                    score += 3.0
                if require_negative and cand["negative"] and not have_neg:
                    score += 3.0
                score += 1.5 * len(cand["combos"] - combos)
                score += random.random()
                scored.append((score, cand))
            if not scored:
                break
            scored.sort(key=lambda item: item[0], reverse=True)
            top = [cand for _, cand in scored[: min(12, len(scored))]]
            choice = random.choice(top)
            selected.append(choice)
            used.append(choice["bbox"])
            pool.remove(choice)
        if len(selected) == count and _coverage_ok(selected, require_negative):
            return selected
    return None


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _expected_from_layout(grid: np.ndarray, seed_bbox: tuple[int, int, int, int], seed: np.ndarray, windows: list[dict]) -> np.ndarray:
    out = grid.copy()
    seed_top, seed_left, _, _ = seed_bbox
    for item in windows:
        r0, c0, r1, c1 = item["bbox"]
        patch = _render_window(seed, r0 + 1 - seed_top, c0 + 1 - seed_left, item["inner_h"], item["inner_w"], mirrored=True)
        out[r0 + 1 : r1, c0 + 1 : c1] = patch
    return out


def _validate_semantics(
    inp: np.ndarray,
    out: np.ndarray,
    seed_bbox: tuple[int, int, int, int],
    seed: np.ndarray,
    windows: list[dict],
    require_negative: bool,
) -> bool:
    try:
        parsed_seed_bbox, parsed_seed, parsed_frames = _parse_scene(inp)
    except AssertionError:
        return False
    if parsed_seed_bbox != seed_bbox or not np.array_equal(parsed_seed, seed):
        return False
    if Counter(parsed_frames) != Counter([item["bbox"] for item in windows]):
        return False
    if not _all_reflections_distinct(seed):
        return False
    if _has_smaller_period(seed):
        return False
    if not _coverage_ok(windows, require_negative):
        return False
    if np.array_equal(inp, out):
        return False
    if not np.array_equal(out, _expected_from_layout(inp, seed_bbox, seed, windows)):
        return False
    if not np.array_equal(main(inp), out):
        return False
    for item in windows:
        if not item["diff_from_plain"]:
            continue
        r0, c0, _, _ = item["bbox"]
        mirrored = _render_window(seed, r0 + 1 - seed_bbox[0], c0 + 1 - seed_bbox[1], item["inner_h"], item["inner_w"], mirrored=True)
        plain = _render_window(seed, r0 + 1 - seed_bbox[0], c0 + 1 - seed_bbox[1], item["inner_h"], item["inner_w"], mirrored=False)
        if np.array_equal(mirrored, plain):
            return False
    return True


def _generate_attempt() -> np.ndarray | None:
    seed = _make_seed_tile()
    if seed is None:
        return None

    rows = random.randint(*GEN_GRID_ROWS)
    cols = random.randint(*GEN_GRID_COLS)
    seed_h, seed_w = seed.shape
    if rows < seed_h + 10 or cols < seed_w + 10:
        return None

    seed_top = random.randint(3, rows - seed_h - 4)
    seed_left = random.randint(3, cols - seed_w - 4)
    seed_bbox = (seed_top, seed_left, seed_top + seed_h - 1, seed_left + seed_w - 1)

    candidates = _enumerate_window_candidates((rows, cols), seed, seed_bbox)
    if len(candidates) < 20:
        return None

    count = random.choices(GEN_WINDOW_COUNT, weights=GEN_WINDOW_WEIGHTS, k=1)[0]
    require_negative = random.random() < 0.65
    windows = _choose_windows(candidates, seed_bbox, count, require_negative)
    if windows is None:
        return None

    seed_colors = set(map(int, seed.flatten()))
    frame_palette = [color for color in COLOR_POOL if color not in seed_colors]
    if len(frame_palette) < len(windows):
        return None
    frame_colors = random.sample(frame_palette, len(windows))

    inp = np.zeros((rows, cols), dtype=int)
    inp[seed_top : seed_top + seed_h, seed_left : seed_left + seed_w] = seed
    for item, color in zip(windows, frame_colors):
        _draw_frame(inp, item["bbox"], int(color))

    out = main(inp)
    if not _validate_semantics(inp, out, seed_bbox, seed, windows, require_negative):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid phantom-wallpaper window instance")
