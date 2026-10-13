import numpy as np


BG = 0
FRAME_COLORS = [1, 2, 3, 4, 5, 6, 7, 8]
LABEL_COLORS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
DIRS = {
    "top": 0,
    "right": 3,   # np.rot90 k=3 == 90 deg clockwise
    "bottom": 2,
    "left": 1,    # np.rot90 k=1 == 270 deg clockwise
}
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for color in sorted(int(x) for x in np.unique(grid) if int(x) != bg):
        for comp in _components(grid == color):
            r0, c0, r1, c1 = _bbox(comp)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 3 or w < 3:
                continue
            need = 2 * (h + w) - 4
            if len(comp) != need:
                continue
            sub = grid[r0 : r1 + 1, c0 : c1 + 1]
            border = np.zeros((h, w), dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if not np.all(sub[border] == color):
                continue
            if not np.all(sub[~border] == bg):
                continue
            frames.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "interior_shape": (h - 2, w - 2),
                }
            )
    return frames


def _label_for_frame(grid: np.ndarray, frame: dict, bg: int) -> tuple[int, str, tuple[int, int]]:
    h, w = grid.shape
    r0, c0, r1, c1 = frame["bbox"]
    hits = []
    if r0 > 0:
        for c in range(c0, c1 + 1):
            v = int(grid[r0 - 1, c])
            if v != bg and v != frame["color"]:
                hits.append((v, "top", (r0 - 1, c)))
    if c1 + 1 < w:
        for r in range(r0, r1 + 1):
            v = int(grid[r, c1 + 1])
            if v != bg and v != frame["color"]:
                hits.append((v, "right", (r, c1 + 1)))
    if r1 + 1 < h:
        for c in range(c0, c1 + 1):
            v = int(grid[r1 + 1, c])
            if v != bg and v != frame["color"]:
                hits.append((v, "bottom", (r1 + 1, c)))
    if c0 > 0:
        for r in range(r0, r1 + 1):
            v = int(grid[r, c0 - 1])
            if v != bg and v != frame["color"]:
                hits.append((v, "left", (r, c0 - 1)))
    assert len(hits) == 1, "each frame must have exactly one adjacent label pixel"
    return hits[0]


def _extract_template(grid: np.ndarray, color: int, blocked: set[tuple[int, int]]) -> dict:
    mask = grid == color
    for r, c in blocked:
        if 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1]:
            mask[r, c] = False
    comps = _components(mask)
    assert len(comps) == 1, "each label color must map to one template component"
    comp = comps[0]
    r0, c0, r1, c1 = _bbox(comp)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    patch[comp[:, 0] - r0, comp[:, 1] - c0] = True
    return {"patch": patch, "coords": [tuple(map(int, x)) for x in comp.tolist()]}


def _center_paste(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    out_h, out_w = shape
    out = np.zeros((out_h, out_w), dtype=bool)
    in_h, in_w = mask.shape
    dst_r0 = max(0, (out_h - in_h) // 2)
    dst_c0 = max(0, (out_w - in_w) // 2)
    src_r0 = max(0, (in_h - out_h) // 2)
    src_c0 = max(0, (in_w - out_w) // 2)
    hh = min(out_h, in_h)
    ww = min(out_w, in_w)
    out[dst_r0 : dst_r0 + hh, dst_c0 : dst_c0 + ww] = mask[src_r0 : src_r0 + hh, src_c0 : src_c0 + ww]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(np.bincount(g.ravel()).argmax())
    frames = _find_frames(g, bg)
    assert frames, "at least one hollow frame is required"

    label_info = []
    blocked: dict[int, set[tuple[int, int]]] = {}
    for frame in frames:
        label_color, side, pos = _label_for_frame(g, frame, bg)
        label_info.append((frame, label_color, side, pos))
        blocked.setdefault(label_color, set()).add(pos)

    templates = {color: _extract_template(g, color, positions) for color, positions in blocked.items()}

    out = g.copy()
    for positions in blocked.values():
        for r, c in positions:
            out[r, c] = bg
    for info in templates.values():
        for r, c in info["coords"]:
            out[r, c] = bg

    for frame, label_color, side, _ in label_info:
        r0, c0, r1, c1 = frame["bbox"]
        ih, iw = frame["interior_shape"]
        rotated = np.rot90(templates[label_color]["patch"], DIRS[side])
        filled = _center_paste(rotated, (ih, iw))
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[:, :] = bg
        interior[filled] = label_color

    return out


def _shape_variants(mask: np.ndarray) -> set[bytes]:
    return {np.rot90(mask, k).tobytes() for k in range(4)}


def _random_connected_mask(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    cells = [(r, c) for r in range(h) for c in range(w)]
    while True:
        target = int(rng.integers(max(3, (h * w) // 3), h * w + 1))
        start = cells[int(rng.integers(len(cells)))]
        chosen = {start}
        frontier = {start}
        while len(chosen) < target and frontier:
            fr, fc = list(frontier)[int(rng.integers(len(frontier)))]
            neigh = []
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = fr + dr, fc + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in chosen:
                    neigh.append((nr, nc))
            if not neigh:
                frontier.remove((fr, fc))
                continue
            nxt = neigh[int(rng.integers(len(neigh)))]
            chosen.add(nxt)
            frontier.add(nxt)
        mask = np.zeros((h, w), dtype=bool)
        for r, c in chosen:
            mask[r, c] = True
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        mask = mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if len(_shape_variants(mask)) < 2:
            continue
        return mask


def _can_place(occ: np.ndarray, r0: int, c0: int, h: int, w: int, margin: int = 1) -> bool:
    H, W = occ.shape
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    if r0 < 0 or c0 < 0 or r1 >= H or c1 >= W:
        return False
    rr0 = max(0, r0 - margin)
    cc0 = max(0, c0 - margin)
    rr1 = min(H - 1, r1 + margin)
    cc1 = min(W - 1, c1 + margin)
    return not occ[rr0 : rr1 + 1, cc0 : cc1 + 1].any()


def _mark_place(occ: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    occ[r0 : r0 + h, c0 : c0 + w] = True


def _paint_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _place_scene(rng: np.random.Generator) -> tuple[np.ndarray, list[dict]] | None:
    h = int(rng.integers(20, 31))
    w = int(rng.integers(20, 31))
    grid = np.full((h, w), BG, dtype=int)
    occ = np.zeros((h, w), dtype=bool)

    n = int(rng.integers(2, 5))
    frame_colors = list(rng.choice(FRAME_COLORS, size=n, replace=False))
    label_colors = list(rng.choice(LABEL_COLORS, size=n, replace=False))
    while set(frame_colors) & set(label_colors):
        label_colors = list(rng.choice(LABEL_COLORS, size=n, replace=False))

    scene = []
    sides = ["top", "right", "bottom", "left"]
    rng.shuffle(sides)
    size_modes = ["smaller", "equal", "larger"]

    for i in range(n):
        side = sides[i % 4] if i < 4 else sides[int(rng.integers(4))]
        mode = size_modes[int(rng.integers(len(size_modes)))]
        ih = int(rng.integers(3, 8))
        iw = int(rng.integers(3, 8))

        if mode == "smaller":
            rh = int(rng.integers(2, ih + 1))
            rw = int(rng.integers(2, iw + 1))
            if rh == ih and rw == iw:
                rh = max(2, rh - 1)
        elif mode == "equal":
            rh, rw = ih, iw
        else:
            rh = int(rng.integers(max(ih, 2), min(ih + 4, 8)))
            rw = int(rng.integers(max(iw, 2), min(iw + 4, 8)))
            if rh == ih and rw == iw:
                if rh < 7:
                    rh += 1
                else:
                    rw += 1

        if side in ("top", "bottom"):
            th, tw = rh, rw
        else:
            th, tw = rw, rh

        template = _random_connected_mask(rng, th, tw)
        if side in ("top", "bottom"):
            rot_view = template
        elif side == "right":
            rot_view = np.rot90(template, 3)
        else:
            rot_view = np.rot90(template, 1)
        if mode == "smaller" and (rot_view.shape[0] > ih or rot_view.shape[1] > iw):
            continue
        if mode == "equal" and rot_view.shape != (ih, iw):
            continue
        if mode == "larger" and not (rot_view.shape[0] > ih or rot_view.shape[1] > iw):
            continue

        fh, fw = ih + 2, iw + 2
        frame_ok = False
        for _ in range(120):
            r0 = int(rng.integers(1, h - fh))
            c0 = int(rng.integers(1, w - fw))
            label_pos = {
                "top": (r0 - 1, c0 + int(rng.integers(fw))),
                "right": (r0 + int(rng.integers(fh)), c0 + fw),
                "bottom": (r0 + fh, c0 + int(rng.integers(fw))),
                "left": (r0 + int(rng.integers(fh)), c0 - 1),
            }[side]
            rr = min(r0, label_pos[0])
            cc = min(c0, label_pos[1])
            hh = max(r0 + fh - 1, label_pos[0]) - rr + 1
            ww = max(c0 + fw - 1, label_pos[1]) - cc + 1
            if _can_place(occ, rr, cc, hh, ww, margin=1):
                frame_ok = True
                break
        if not frame_ok:
            return None

        _paint_frame(grid, (r0, c0, r0 + fh - 1, c0 + fw - 1), frame_colors[i])
        grid[label_pos] = label_colors[i]
        _mark_place(occ, rr, cc, hh, ww)

        tpl_h, tpl_w = template.shape
        tpl_ok = False
        for _ in range(120):
            tr0 = int(rng.integers(1, h - tpl_h))
            tc0 = int(rng.integers(1, w - tpl_w))
            if _can_place(occ, tr0, tc0, tpl_h, tpl_w, margin=1):
                tpl_ok = True
                break
        if not tpl_ok:
            return None

        for pr, pc in np.argwhere(template):
            grid[tr0 + pr, tc0 + pc] = label_colors[i]
        _mark_place(occ, tr0, tc0, tpl_h, tpl_w)

        scene.append(
            {
                "frame_bbox": (r0, c0, r0 + fh - 1, c0 + fw - 1),
                "frame_color": frame_colors[i],
                "label_color": label_colors[i],
                "side": side,
                "label_pos": label_pos,
                "template": template,
                "template_pos": (tr0, tc0),
            }
        )

    return grid, scene


def _expected_output(inp: np.ndarray, scene: list[dict]) -> np.ndarray:
    out = inp.copy()
    for item in scene:
        out[item["label_pos"]] = BG
        tr0, tc0 = item["template_pos"]
        template = item["template"]
        for dr, dc in np.argwhere(template):
            out[tr0 + dr, tc0 + dc] = BG

    for item in scene:
        r0, c0, r1, c1 = item["frame_bbox"]
        ih, iw = r1 - r0 - 1, c1 - c0 - 1
        rotated = np.rot90(item["template"], DIRS[item["side"]])
        filled = _center_paste(rotated, (ih, iw))
        out[r0 + 1 : r1, c0 + 1 : c1] = BG
        out[r0 + 1 : r1, c0 + 1 : c1][filled] = item["label_color"]
    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        made = _place_scene(rng)
        if made is None:
            continue
        inp, scene = made
        exp = _expected_output(inp, scene)
        try:
            got = main(inp)
        except Exception:
            continue
        if np.array_equal(got, exp):
            return inp
    raise RuntimeError("failed to generate valid input")
