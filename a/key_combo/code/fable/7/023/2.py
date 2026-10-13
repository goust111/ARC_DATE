from __future__ import annotations

import numpy as np


BACKGROUND = 0
PALETTE = tuple(range(1, 10))
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_ATTEMPTS = 600


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()


def find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == int(bg):
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            coords = np.array(cells, dtype=int)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(len(cells)),
                }
            )
    return comps


def _frame_kind(grid: np.ndarray, comp: dict) -> str | None:
    r0, c0, r1, c1 = comp["bbox"]
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    color = int(comp["color"])
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    mask = patch == color
    if mask[1:-1, 1:-1].any():
        return None

    perimeter = np.zeros_like(mask, dtype=bool)
    perimeter[0, :] = True
    perimeter[-1, :] = True
    perimeter[:, 0] = True
    perimeter[:, -1] = True

    if np.array_equal(mask, perimeter):
        return "complete"
    if np.any(mask) and np.all(mask <= perimeter):
        return "broken"
    return None


def _find_separator_col(grid: np.ndarray, bg: int) -> tuple[int, int]:
    g = np.asarray(grid, dtype=int)
    candidates = []
    for c in range(g.shape[1]):
        col = g[:, c]
        if int(col[0]) != int(bg) and np.all(col == col[0]):
            candidates.append((c, int(col[0])))
    assert len(candidates) == 1, "expected exactly one full-height separator column"
    return candidates[0]


def _extract_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    sep_col, sep_color = _find_separator_col(g, bg)
    left = g[:, :sep_col]
    right = g[:, sep_col + 1 :]
    evidence_box = bbox(right != bg)
    assert evidence_box is not None, "expected a non-empty evidence fragment"
    evidence = crop_bbox(right, evidence_box)

    frames = []
    for comp in find_components(left, bg):
        kind = _frame_kind(left, comp)
        if kind is None:
            continue
        r0, c0, r1, c1 = comp["bbox"]
        frames.append(
            {
                "color": int(comp["color"]),
                "bbox": (r0, c0, r1, c1),
                "area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
                "complete": kind == "complete",
                "interior": left[r0 + 1 : r1, c0 + 1 : c1].copy(),
            }
        )

    assert frames, "expected at least one frame on the left side"
    return {
        "grid": g,
        "bg": bg,
        "separator_col": sep_col,
        "separator_color": sep_color,
        "left": left,
        "right": right,
        "evidence": evidence,
        "frames": frames,
    }


def _match_offsets(canvas: np.ndarray, fragment: np.ndarray, bg: int) -> list[tuple[int, int]]:
    ch, cw = canvas.shape
    fh, fw = fragment.shape
    if fh > ch or fw > cw:
        return []
    mask = fragment != bg
    if not np.any(mask):
        return []

    hits = []
    for r0 in range(ch - fh + 1):
        for c0 in range(cw - fw + 1):
            patch = canvas[r0 : r0 + fh, c0 : c0 + fw]
            if np.all(patch[mask] == fragment[mask]):
                hits.append((r0, c0))
    return hits


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Select the unique intact frame whose mirrored interior matches the right-side evidence fragment.

    Input structure:
    - A full-height separator column splits the grid into a left frame field and a right evidence field.
    - The left side contains several one-pixel-thick monochrome rectangular frames with multicolor interior patterns.
    - Some left-side frames are broken distractors with missing border cells, and the largest frame is not the target.

    Rule:
    - Keep only the frames whose rectangular border is fully closed.
    - Mirror each candidate frame interior left-to-right, then test whether the right-side fragment can be overlaid on it so that every non-background evidence pixel matches.
    - Exactly one intact frame fits; output its full mirrored interior at the interior's own size.

    Key insight:
    - The right panel is cropped mirror evidence, so matching must happen in the mirrored interior coordinate system and broken-border lookalikes must be rejected first.
    """
    scene = _extract_scene(np.asarray(input_grid, dtype=int))
    bg = int(scene["bg"])
    evidence = scene["evidence"]

    matches = []
    for frame in scene["frames"]:
        if not frame["complete"]:
            continue
        mirrored = np.fliplr(frame["interior"])
        offsets = _match_offsets(mirrored, evidence, bg)
        if offsets:
            matches.append((frame, mirrored))

    assert len(matches) == 1, "expected exactly one intact matching frame"
    return matches[0][1]


def _random_pattern(
    rng: np.random.Generator,
    h: int,
    w: int,
    palette: list[int],
    *,
    min_cells: int = 5,
) -> np.ndarray:
    for _ in range(200):
        grid = np.full((h, w), BACKGROUND, dtype=int)
        n_colors = int(rng.integers(2, min(4, len(palette)) + 1))
        colors = [int(x) for x in rng.choice(palette, size=n_colors, replace=False)]
        total = int(rng.integers(min_cells, max(min_cells + 1, min(h * w, max(min_cells + 1, (2 * h * w) // 3 + 1)))))
        total = min(total, h * w - 1)
        start = (int(rng.integers(h)), int(rng.integers(w)))
        cells = {start}
        frontier = [start]
        while len(cells) < total:
            if frontier and rng.random() < 0.8:
                rr, cc = frontier[int(rng.integers(len(frontier)))]
                dr, dc = DIRS4[int(rng.integers(4))]
                nr = min(max(rr + dr, 0), h - 1)
                nc = min(max(cc + dc, 0), w - 1)
            else:
                nr = int(rng.integers(h))
                nc = int(rng.integers(w))
            if (nr, nc) not in cells:
                cells.add((nr, nc))
                frontier.append((nr, nc))

        cells = list(cells)
        rng.shuffle(cells)
        split_points = np.linspace(0, len(cells), n_colors + 1, dtype=int)
        for i, color in enumerate(colors):
            part = cells[split_points[i] : split_points[i + 1]]
            if not part:
                continue
            for r, c in part:
                grid[r, c] = color

        non_bg = grid != BACKGROUND
        if non_bg.sum() < min_cells:
            continue
        if len(np.unique(grid[non_bg])) < 2:
            continue
        if np.array_equal(grid, np.fliplr(grid)):
            continue
        if bbox(non_bg) is None:
            continue
        return grid
    raise RuntimeError("failed to sample a non-degenerate interior pattern")


def _remove_border_segment(rng: np.random.Generator, frame: np.ndarray) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape
    side = int(rng.integers(4))
    if side == 0:
        c = int(rng.integers(1, w - 1))
        out[0, c] = BACKGROUND
        if w > 4 and rng.random() < 0.5:
            out[0, min(c + 1, w - 2)] = BACKGROUND
    elif side == 1:
        c = int(rng.integers(1, w - 1))
        out[h - 1, c] = BACKGROUND
        if w > 4 and rng.random() < 0.5:
            out[h - 1, max(c - 1, 1)] = BACKGROUND
    elif side == 2:
        r = int(rng.integers(1, h - 1))
        out[r, 0] = BACKGROUND
        if h > 4 and rng.random() < 0.5:
            out[min(r + 1, h - 2), 0] = BACKGROUND
    else:
        r = int(rng.integers(1, h - 1))
        out[r, w - 1] = BACKGROUND
        if h > 4 and rng.random() < 0.5:
            out[max(r - 1, 1), w - 1] = BACKGROUND
    return out


def _render_frame(obj: dict) -> np.ndarray:
    oh, ow = obj["outer_shape"]
    border_color = int(obj["border_color"])
    tile = np.full((oh, ow), BACKGROUND, dtype=int)
    tile[0, :] = border_color
    tile[-1, :] = border_color
    tile[:, 0] = border_color
    tile[:, -1] = border_color
    tile[1:-1, 1:-1] = obj["interior"]
    if not obj["complete"]:
        tile = _remove_border_segment(obj["rng"], tile)
    return tile


def _place_objects(
    rng: np.random.Generator,
    canvas_h: int,
    canvas_w: int,
    objects: list[dict],
) -> list[tuple[int, int]] | None:
    occ = np.zeros((canvas_h, canvas_w), dtype=bool)
    placements: list[tuple[int, int]] = [(-1, -1)] * len(objects)
    order = sorted(range(len(objects)), key=lambda i: objects[i]["outer_shape"][0] * objects[i]["outer_shape"][1], reverse=True)

    for idx in order:
        oh, ow = objects[idx]["outer_shape"]
        choices = [
            (r, c)
            for r in range(canvas_h - oh + 1)
            for c in range(canvas_w - ow + 1)
            if not occ[max(0, r - 1) : min(canvas_h, r + oh + 1), max(0, c - 1) : min(canvas_w, c + ow + 1)].any()
        ]
        if not choices:
            return None
        r0, c0 = choices[int(rng.integers(len(choices)))]
        placements[idx] = (r0, c0)
        occ[r0 : r0 + oh, c0 : c0 + ow] = True

    return placements


def _candidate_matches(frame: dict, evidence: np.ndarray, bg: int) -> bool:
    return bool(_match_offsets(np.fliplr(frame["interior"]), evidence, bg))


def _choose_fragment(
    rng: np.random.Generator,
    target_mirror: np.ndarray,
    other_complete: list[np.ndarray],
) -> np.ndarray | None:
    coords = np.argwhere(target_mirror != BACKGROUND)
    if coords.shape[0] < 4:
        return None

    for _ in range(400):
        max_take = coords.shape[0]
        min_take = min(max_take, 3)
        if min_take == max_take:
            return None
        take = int(rng.integers(min_take, max_take))
        chosen = coords[rng.choice(max_take, size=take, replace=False)]
        probe = np.full_like(target_mirror, BACKGROUND)
        for r, c in chosen:
            probe[int(r), int(c)] = int(target_mirror[int(r), int(c)])
        box = bbox(probe != BACKGROUND)
        if box is None:
            continue
        fragment = crop_bbox(probe, box)
        mask = fragment != BACKGROUND
        if mask.sum() < 3:
            continue
        if fragment.shape[0] == 1 and fragment.shape[1] == 1:
            continue
        if len(np.unique(fragment[mask])) < 2:
            continue
        if np.array_equal(fragment, target_mirror):
            continue
        match_count = int(_match_offsets(target_mirror, fragment, BACKGROUND) != [])
        for cand in other_complete:
            if _match_offsets(cand, fragment, BACKGROUND):
                match_count += 1
        if match_count == 1:
            return fragment
    return None


def _semantic_check(grid: np.ndarray) -> bool:
    scene = _extract_scene(grid)
    frames = scene["frames"]
    complete = [frame for frame in frames if frame["complete"]]
    broken = [frame for frame in frames if not frame["complete"]]
    evidence = scene["evidence"]
    bg = int(scene["bg"])

    assert len(complete) >= 2, "need at least two intact frames"
    assert broken, "need at least one broken frame"

    complete_matches = [frame for frame in complete if _candidate_matches(frame, evidence, bg)]
    broken_matches = [frame for frame in broken if _candidate_matches(frame, evidence, bg)]
    assert len(complete_matches) == 1, "evidence must select a unique intact frame"
    assert broken_matches, "need a broken distractor that would also match by pattern alone"

    target_area = complete_matches[0]["area"]
    largest_area = max(frame["area"] for frame in frames)
    assert target_area < largest_area, "largest frame must not be the target"

    out = main(grid)
    assert np.array_equal(out, np.fliplr(complete_matches[0]["interior"]))
    assert out.ndim == 2 and max(out.shape) <= 30
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = np.random.default_rng()

        target_h = int(rng.integers(4, 8))
        target_w = int(rng.integers(4, 8))
        target_outer = (target_h + 2, target_w + 2)

        lure_h = int(rng.integers(target_h + 1, min(10, target_h + 4)))
        lure_w = int(rng.integers(target_w + 1, min(10, target_w + 4)))
        lure_outer = (lure_h + 2, lure_w + 2)

        canvas_h = int(rng.integers(max(lure_outer[0] + 2, 16), 25))
        left_w = int(rng.integers(max(lure_outer[1] + 4, 16), 21))

        extra_complete = int(rng.integers(0, 2))
        extra_broken = int(rng.integers(0, 2))
        object_count = 4 + extra_complete + extra_broken
        if object_count + 1 > len(PALETTE) - 2:
            continue

        chosen_colors = [int(x) for x in rng.choice(PALETTE, size=object_count + 1, replace=False)]
        separator_color = chosen_colors[0]
        border_colors = chosen_colors[1:]
        interior_palette = [c for c in PALETTE if c not in chosen_colors]
        if len(interior_palette) < 2:
            continue

        try:
            target_interior = _random_pattern(rng, target_h, target_w, interior_palette)
            lure_interior = _random_pattern(rng, lure_h, lure_w, interior_palette)
        except RuntimeError:
            continue

        objects = [
            {
                "kind": "target",
                "complete": True,
                "outer_shape": target_outer,
                "border_color": border_colors[0],
                "interior": target_interior,
                "rng": rng,
            },
            {
                "kind": "lure",
                "complete": True,
                "outer_shape": lure_outer,
                "border_color": border_colors[1],
                "interior": lure_interior,
                "rng": rng,
            },
            {
                "kind": "broken_clone",
                "complete": False,
                "outer_shape": target_outer,
                "border_color": border_colors[2],
                "interior": target_interior.copy(),
                "rng": rng,
            },
            {
                "kind": "complete_other",
                "complete": True,
                "outer_shape": (
                    int(rng.integers(5, min(10, lure_outer[0]))),
                    int(rng.integers(5, min(10, lure_outer[1]))),
                ),
                "border_color": border_colors[3],
                "interior": None,
                "rng": rng,
            },
        ]

        for i in range(extra_complete):
            objects.append(
                {
                    "kind": f"complete_extra_{i}",
                    "complete": True,
                    "outer_shape": (
                        int(rng.integers(5, min(10, lure_outer[0]))),
                        int(rng.integers(5, min(10, lure_outer[1]))),
                    ),
                    "border_color": border_colors[4 + i],
                    "interior": None,
                    "rng": rng,
                }
            )

        for i in range(extra_broken):
            objects.append(
                {
                    "kind": f"broken_extra_{i}",
                    "complete": False,
                    "outer_shape": (
                        int(rng.integers(5, min(10, lure_outer[0]))),
                        int(rng.integers(5, min(10, lure_outer[1]))),
                    ),
                    "border_color": border_colors[4 + extra_complete + i],
                    "interior": None,
                    "rng": rng,
                }
            )

        valid = True
        for obj in objects:
            oh, ow = obj["outer_shape"]
            if oh < 5 or ow < 5 or oh > canvas_h or ow > left_w:
                valid = False
                break
            if obj["interior"] is None:
                try:
                    obj["interior"] = _random_pattern(rng, oh - 2, ow - 2, interior_palette)
                except RuntimeError:
                    valid = False
                    break
        if not valid:
            continue

        placements = _place_objects(rng, canvas_h, left_w, objects)
        if placements is None:
            continue

        target_mirror = None
        for obj in objects:
            if obj["complete"] and obj["kind"] == "target":
                target_mirror = np.fliplr(obj["interior"])
        assert target_mirror is not None

        other_complete = []
        for obj in objects:
            if obj["complete"] and obj["kind"] != "target":
                other_complete.append(np.fliplr(obj["interior"]))

        fragment = _choose_fragment(rng, target_mirror, other_complete)
        if fragment is None:
            continue

        right_w = int(rng.integers(fragment.shape[1], min(9, 30 - left_w - 1) + 1))
        if left_w + 1 + right_w > 30:
            continue
        if fragment.shape[0] > canvas_h:
            continue

        grid = np.full((canvas_h, left_w + 1 + right_w), BACKGROUND, dtype=int)
        grid[:, left_w] = separator_color

        for obj, (r0, c0) in zip(objects, placements):
            tile = _render_frame(obj)
            oh, ow = tile.shape
            grid[r0 : r0 + oh, c0 : c0 + ow] = tile

        top = int(rng.integers(0, canvas_h - fragment.shape[0] + 1))
        left = left_w + 1 + int(rng.integers(0, right_w - fragment.shape[1] + 1))
        mask = fragment != BACKGROUND
        block = grid[top : top + fragment.shape[0], left : left + fragment.shape[1]]
        block[mask] = fragment[mask]

        try:
            if _semantic_check(grid):
                return grid
        except AssertionError:
            continue

    raise RuntimeError("failed to generate a valid ARC input")
